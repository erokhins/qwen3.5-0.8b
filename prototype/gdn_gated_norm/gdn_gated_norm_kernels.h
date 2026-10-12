/*
 * prototype/gdn_gated_norm/gdn_gated_norm_kernels.h — §04.5, the GDN gated RMSNorm, k = 1.
 *
 *     n   = o · rsqrt(mean(o², 128) + eps)                    F32, per value head (§04.5)
 *     â   = BF16(n)                                           A-01's "cast the normalized back"
 *     a   = BF16( â · BF16(w_norm) )                          affine, in the activation dtype
 *     ô   = BF16( a · silu(z) )                               gate last, one cast (§04.5)
 *
 * The three BF16 boundaries above are MEASURED, not transcribed: `mirror_probe.py` enumerates
 * the plausible cast orders and only this one reproduces `decode{N}.L{ii}.gdn.gated_out`
 * bit-for-bit — 0 of 294 912 elements differ over dec8 (18 GDN layers × 8 decode steps), and the
 * same on `prefix24dec8`, `smoke`, and 79 776 rows of `p300` prefill. §04.5/A-01 as written
 * ("applies w_norm and silu(z) in F32") puts the affine in F32, and the pins refuse that reading:
 * the same formula with the weight rounded to BF16 and then widened back to F32 misses 74 860 of
 * those elements, with the raw F32 checkpoint weight 105 683. `speccheck.specimpl.rmsnorm_gated`
 * agrees with this policy as speccheck calls it — the weight arrives BF16, so its `w * n` is a
 * BF16 multiply — 0 of 294 912; that count is asserted by the probe. Full write-up, including what
 * each weight dtype costs: results.md §1.
 *
 * `o` arrives F32 because that is what `prototype/gdn_decode`'s recurrence produces, and the
 * kernel rounds it to the activation dtype first: HF's fused delta-rule read casts `o` back to
 * bf16 before the norm (gdn_decode results.md finding 2 — "the dumps are bf16-valued"), so the
 * cast is part of the reference, not a convenience. On the gate's own inputs the cast is
 * invisible (the pins are already BF16-valued), which is why there is no negative for it here;
 * the engine depends on it and results.md says so.
 *
 * One block per value head, one thread per value dim: the reduction is that head's 128 squares,
 * §04.5 normalizes per head and NOT over the flattened 2048 (§05.2 says the same for q/k_norm),
 * and getting that wrong is silent — 16 different scales instead of one. `--neg reduce_wide`
 * cannot be expressed in this block shape, which is why the reduction-side injections are
 * `bf16acc`/`noeps` and the wide-reduction falsification lives in the Python mirror instead
 * (results.md §4).
 *
 * INJECTED BUGS (each must move `gated_out`, and each has a declared must-NOT-catch stage):
 *   w_f32         affine applied with the F32 weight (the A-01 literal reading)
 *   affine_uncast the n·w product stays F32 instead of rounding to the activation dtype
 *   single_cast   everything in F32, one cast at the end (the tempting kernel)
 *   silu_rounded  silu(z) rounded to BF16 before it multiplies
 *   bf16acc       the squared-sum accumulator is BF16 end to end (also moves norm_pre_gate)
 *   noeps         eps dropped from the rsqrt (also moves norm_pre_gate)
 * `norm_pre_gate` is the dumped BF16(n) stage: an injection that only touches the epilogue has
 * to leave it alone, which is what makes the four epilogue negatives locatable rather than
 * decorative (the `mlp_gemv`/`proj_gemv` convention).
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>

#define GN_D_V 128                 // d_v, and the width §04.5 normalizes over, per head
#define GN_HEADS 16                // h_v: 16 blocks, one per value head
#define GN_EPS 1e-6f               // §00.2 rms_norm_eps
#define GN_FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit

enum GdnNeg {
    GN_NONE = 0,
    GN_W_F32,          // affine with the F32 weight, no BF16 round of w or of the product
    GN_AFFINE_UNCAST,  // w is cast, but n·w stays F32
    GN_SINGLE_CAST,    // no intermediate rounding at all
    GN_SILU_ROUNDED,   // silu(z) goes through a BF16 slot before the multiply
    GN_BF16ACC,        // squared-sum accumulated through BF16 slots
    GN_NOEPS,          // rsqrt(mean(o²)) without eps
    GN_ROW_ZERO,       // P6: every token reads row 0's o/z — a tiled kernel that forgot `t`
};

__device__ __forceinline__ float gn_bf(float f) { return float(hip_bfloat16(f)); }

// 128 threads = 4 warps on this target; the same two-ladder shape glue uses.
__device__ __forceinline__ float gn_block_sum(float v, float* red, bool bf16_tree) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) {
        const float n = __shfl_down_sync(GN_FULL_MASK, v, o);
        v = bf16_tree ? gn_bf(v + n) : v + n;
    }
    if ((threadIdx.x & (ws - 1)) == 0) red[threadIdx.x / ws] = v;
    __syncthreads();
    if (threadIdx.x < (unsigned)ws) {
        float x = threadIdx.x < blockDim.x / ws ? red[threadIdx.x] : 0.f;
        for (int o = ws / 2; o; o >>= 1) {
            const float n = __shfl_down_sync(GN_FULL_MASK, x, o);
            x = bf16_tree ? gn_bf(x + n) : x + n;
        }
        if (threadIdx.x == 0) red[0] = x;
    }
    __syncthreads();
    return red[0];
}

/*
 * o:  [tokens][heads][d_v] F32  (the recurrence's output; rounded to the activation dtype here)
 * z:  [tokens][heads][d_v] BF16 (in_proj_z's rows, §04.1 — no cast needed, it is already an activation)
 * w:  [d_v] F32         (linear_attn.norm.weight, §01.6: one of the 36 F32 tensors)
 * out:[tokens][heads][d_v] BF16 (gated_out, which is out_proj's operand)
 * norm_out: optional BF16(n) stage dump, `-` on the plan line to skip
 *
 * `tokens` is P6 (the prefill wave, §06.2): one block per (token, value head) with the head index
 * innermost, so `tokens == 1` degenerates to exactly the `<<<GN_HEADS, GN_D_V>>>` launch this op
 * shipped with — same grid, same 128-thread per-head reduction, same bits — and a wave is that block
 * tiled down the token axis. The norm is per (token, head), so there is no reduction to reorder and
 * the two widths owe each other bit-for-bit; `run.py --tokens` requires it rather than assuming it.
 * `GN_ROW_ZERO` is the bug this shape can have that a wide run could otherwise hide: index the
 * inputs by row 0 while still writing row t. Its locality claim — every row but row 0 moves — is
 * what makes it a negative and not a decoration.
 */
__global__ void gdn_gated_norm(const float* __restrict__ o, const hip_bfloat16* __restrict__ z,
                               const float* __restrict__ w, hip_bfloat16* __restrict__ out,
                               hip_bfloat16* __restrict__ norm_out, int neg, int tokens) {
    __shared__ float red[32];
    const int h = blockIdx.x % GN_HEADS, d = threadIdx.x;
    const int t = (int)(blockIdx.x / GN_HEADS);
    const size_t off = ((size_t)t * GN_HEADS + h) * GN_D_V + d;
    // GN_ROW_ZERO moves the INPUT row only, so the kernel still writes row t of the output. That is
    // the tile bug worth an injection — a wide kernel that forgot to advance the read pointer while
    // advancing the write one — and it is the version with a locality claim a value comparison can see
    // (row 0 exact, every other row wrong). Index the output by row 0 as well and the launch instead
    // shows up in the driver's write-back sentinel, which proves a different thing.
    const size_t in_off = ((size_t)((neg == GN_ROW_ZERO) ? 0 : t) * GN_HEADS + h) * GN_D_V + d;
    (void)tokens;                    // the grid carries it; the bounds are the launcher's to check
    const bool acc_bf16 = (neg == GN_BF16ACC);

    const float x = gn_bf(o[in_off]);                       // activation-dtype cast, see the header
    float acc = x * x;
    if (acc_bf16) acc = gn_bf(acc);
    const float sum = gn_block_sum(acc, red, acc_bf16);
    const float eps = (neg == GN_NOEPS) ? 0.0f : GN_EPS;
    const float r = rsqrtf(sum * (1.0f / GN_D_V) + eps);

    const float n_f32 = x * r;
    const float n = (neg == GN_SINGLE_CAST) ? n_f32 : gn_bf(n_f32);
    if (norm_out) norm_out[off] = hip_bfloat16(n_f32);

    const float wv = w[d];
    float a;
    if (neg == GN_W_F32) {
        a = gn_bf(n * wv);         // only the weight cast is missing: the A-01 literal reading
    } else if (neg == GN_SINGLE_CAST) {
        a = n_f32 * wv;
    } else {
        const float p = n * gn_bf(wv);                        // weight joins as an activation
        a = (neg == GN_AFFINE_UNCAST) ? p : gn_bf(p);
    }

    const float zf = float(z[in_off]);
    float g = zf * (1.0f / (1.0f + expf(-zf)));               // HF's order: z·σ(z), F32
    if (neg == GN_SILU_ROUNDED) g = gn_bf(g);
    out[off] = hip_bfloat16(a * g);
}
