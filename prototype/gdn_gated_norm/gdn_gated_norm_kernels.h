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
 * ("applies w_norm and silu(z) in F32") puts the affine in F32 and misses 105 683 of those
 * elements; `speccheck.specimpl.rmsnorm_gated` implements that reading and is therefore not
 * bit-faithful either. Full write-up, including what the wrong readings cost: results.md.
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
 * o:  [heads][d_v] F32  (the recurrence's output; rounded to the activation dtype here)
 * z:  [heads][d_v] BF16 (in_proj_z's rows, §04.1 — no cast needed, it is already an activation)
 * w:  [d_v] F32         (linear_attn.norm.weight, §01.6: one of the 36 F32 tensors)
 * out:[heads][d_v] BF16 (gated_out, which is out_proj's operand)
 * norm_out: optional BF16(n) stage dump, `-` on the plan line to skip
 */
__global__ void gdn_gated_norm(const float* __restrict__ o, const hip_bfloat16* __restrict__ z,
                               const float* __restrict__ w, hip_bfloat16* __restrict__ out,
                               hip_bfloat16* __restrict__ norm_out, int neg) {
    __shared__ float red[32];
    const int h = blockIdx.x, d = threadIdx.x;
    const bool acc_bf16 = (neg == GN_BF16ACC);

    const float x = gn_bf(o[h * GN_D_V + d]);       // activation-dtype cast, see the header
    float acc = x * x;
    if (acc_bf16) acc = gn_bf(acc);
    const float sum = gn_block_sum(acc, red, acc_bf16);
    const float eps = (neg == GN_NOEPS) ? 0.0f : GN_EPS;
    const float r = rsqrtf(sum * (1.0f / GN_D_V) + eps);

    const float n_f32 = x * r;
    const float n = (neg == GN_SINGLE_CAST) ? n_f32 : gn_bf(n_f32);
    if (norm_out) norm_out[h * GN_D_V + d] = hip_bfloat16(n_f32);

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

    const float zf = float(z[h * GN_D_V + d]);
    float g = zf * (1.0f / (1.0f + expf(-zf)));               // HF's order: z·σ(z), F32
    if (neg == GN_SILU_ROUNDED) g = gn_bf(g);
    out[h * GN_D_V + d] = hip_bfloat16(a * g);
}
