/*
 * prototype/gdn_rec_isa/gdn_rec_isa_kernels.h — the shipped §04.4 recurrence plus MODEs that re-spell
 * one named part of it, so an ISA census can bill one TOKEN-STEP of `gdn_rec` and price each re-spelling.
 *
 * WHY THIS DIRECTORY EXISTS. `bench/performance-status.md` §(the hipBLAS ledger) bills `gdn_rec` at
 * 126.83 ms of a 245.7 ms wave at 1024 lanes (51.2 %) and ~509 ms of ~1,105 ms at 4096 lanes — the
 * biggest family in the wave, 18,432 launches of 6.8 µs each. Every route that has attacked it (P9
 * chunked scan, D-12 route A) was priced bottom-up from an instruction or rate model and missed by
 * ~10×, and `AGENTS.md` already names why: "price a serial kernel in cycles-per-dependency at its
 * realised occupancy, or do not price it". Nobody had counted its instructions. This file exists so
 * the count can be taken the way `prototype/attn_valu_slim/` took attention's: a MODE-0 copy that must
 * census identical to the shipped kernel, one lever per MODE, and the difference in slots per unit.
 *
 * THE UNIT IS NOT THE TOUCH. One `rec_kernel` launch is ONE (token, GDN layer) — grid <<<NH=16, D=128>>>,
 * one block per value head, one thread per d_v column, wave32, so 64 wavefronts per launch. The
 * prologue (two block reductions for the k/q norms, the LDS staging, the scalar `expf`) runs once per
 * token-step and amortises against nothing; the loops are 128 rows of state each. Attention's lever
 * amortised over 4 touches per tile; here the prologue amortises over 1. That asymmetry is the result.
 *
 * MODES — one lever each, so a delta is attributable:
 *   0 `rec0_kernel`   `rec_kernel` copied character for character, shipped `block_sum` called verbatim.
 *                     If MODE 0 does not census identical to the shipped kernel every delta below is
 *                     void — the same control discipline as `attn_valu_slim_kernels.h` MODE 0.
 *   1 `rec1_kernel`   hand-spelled cross-lane fold (`shfl_down_bits`) with the SHIPPED barrier
 *                     structure kept: isolates "the compiler lowered my shuffle and spilled exec
 *                     guards and range-selects", the attention fold lever.
 *   2 `rec2_kernel`   the two block reductions fused into one pass over shared memory with hand-spelled
 *                     shuffles: adds the barrier-count lever (11 barriers/token-step -> 5).
 *   3 `rec3_kernel`   + `gdn_norm3_kernel`: the norms leave the recurrence launch entirely and become
 *                     one pre-pass per token for all 18 layers, so the state walk's launch contains no
 *                     reduction and no barrier at all. This is the geometry lever, and it costs one
 *                     extra launch per token.
 *
 * BIT-IDENTITY. Every MODE keeps each reduction's association identical to the shipped `block_sum`:
 * 5 `shfl_down` steps in the same order in-warp, then the same 5 steps over the per-warp partials.
 * A hand-spelled permute must therefore read the SAME source lane the compiler's lowering reads;
 * `shfl_down_bits` documents the one place where the out-of-range lanes differ and why that is
 * unobservable. Nothing here re-associates a float add, changes an intrinsic, or touches the state.
 */
#pragma once

#include "../../prototype/gdn_decode/gdn_decode_kernels.h"   // rec_kernel, block_sum, D, NH, Q_SCALE

/* The shipped second stage, written once so the copies below cannot drift from the original tree:
 * 5 shfl_down steps over the per-warp partials in lanes 0..warps-1. `warps` is passed explicitly
 * because MODE 2's pre-pass runs 512 threads for one head-group and must reduce 4 partials, exactly
 * like the shipped 128-thread block, not 16. */
#define GDN_ISA_STAGE2(red, warps)                                                           \
    do {                                                                                     \
        float t_ = (threadIdx.x < (warps)) ? (red)[threadIdx.x] : 0.f;                        \
        t_ += __shfl_down_sync(0xffffffffffffffffull, t_, 16);                                 \
        t_ += __shfl_down_sync(0xffffffffffffffffull, t_, 8);                                  \
        t_ += __shfl_down_sync(0xffffffffffffffffull, t_, 4);                                  \
        t_ += __shfl_down_sync(0xffffffffffffffffull, t_, 2);                                  \
        t_ += __shfl_down_sync(0xffffffffffffffffull, t_, 1);                                  \
        __syncthreads();                                                                      \
        if (threadIdx.x == 0) (red)[0] = t_;                                                   \
        __syncthreads();                                                                      \
    } while (0)

/* ---- the hand-spelled fold: see the block below, which carries the constraint write-up. --------- */

/* The hand-spelled fold (the lever of MODEs 1-3).
 *
 * `__shfl_down_sync(mask, v, delta)` on a full wave lowers on gfx1151 to: a range-select that zeroes
 * the delta when the source lane would leave the wave (CUDA's "out of range returns the caller's own
 * value"), a byte-address form of the source lane, the permute — and around all of it an
 * `exec == full` test with a branch to `s_trap 2` (LLVM's divergent-shuffle fallback), a
 * `s_mov_b32 sN, -1`, and the `%Flow` convergence restore. The bookkeeping is what this lever removes;
 * the census calls it `xlane.exec-guard` and it is 127 of 1,523 slots per token-step.
 *
 * WHAT MAY BE REMOVED AND WHAT MAY NOT. The permute must read the source lane the compiler's own
 * lowering reads, at every step, for every lane — not merely for lane 0. That rules out the cute
 * version: `ds_swizzle_b32` with a BITMASK_PERM pattern is exactly the clamp for delta 16 (forcing
 * bit 4 gives lane+16 below 16 and lane itself at or above 16, which is CUDA's rule to the bit) and
 * nothing else, because lane+8 is not a bit pattern. The second stage is where a wrong clamp shows
 * up: only lanes 0..warps-1 hold data, so lane 0's delta-8 read must deliver the zero that lane 8
 * really holds, and an xor-8 permute would deliver lane 0's own value and double the sum.
 * `s_waitcnt lgkmcnt(0)` is written by hand because SIInsertWaitcnts does not insert waits for
 * inline-asm lane ops — the same hazard `prototype/attn_valu_slim` documents for its hand-spelled
 * permute: a missing wait is RACY, and a race shows up as run-to-run disagreement, not a stable
 * wrong answer.
 */
__device__ __forceinline__ float shfl_down_bits(float v, unsigned delta) {
    unsigned lane = threadIdx.x & 31u;
    unsigned bits = __float_as_uint(v);
    unsigned out;
    if (delta == 16u) {          // bit-4-set == lane+16 below 16, lane itself at or above: CUDA's rule
        asm volatile("s_waitcnt lgkmcnt(0)\n\tds_swizzle_b32 %0, %1 offset:swizzle(BITMASK_PERM,\"1pppp\")"
                     : "=v"(out) : "v"(bits));
        return __uint_as_float(out);
    }
    unsigned d = (lane + delta < 32u) ? delta : 0u;      // the same clamp the compiler emits
    unsigned addr = (lane + d) << 2;
    asm volatile("s_waitcnt lgkmcnt(0)\n\tds_bpermute_b32 %0, %1, %2"
                 : "=v"(out) : "v"(addr), "v"(bits));
    return __uint_as_float(out);
}
/* Hand-spelled single-value block reduction, shipped barrier discipline kept (4 barriers: red write,
 * protect red before lane 0 rewrites it, publish red[0], protect red[0] for the next call). MODE 1. */
__device__ __forceinline__ float block_sum_hand(float v, float* red) {
    v += shfl_down_bits(v, 16);
    v += shfl_down_bits(v, 8);
    v += shfl_down_bits(v, 4);
    v += shfl_down_bits(v, 2);
    v += shfl_down_bits(v, 1);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    int warps = blockDim.x >> 5;
    GDN_ISA_STAGE2(red, warps);
    float out = red[0];
    __syncthreads();
    return out;
}

/* Two values, one pass over shared memory. Each value keeps its own 5+5 tree and its own red[] array;
 * what is shared is the barrier each stage needs. 3 barriers for the pair instead of the shipped 8:
 * one after the per-warp writes, one before lane 0 publishes, one after it. The two per-call
 * "protect red[0]" barriers of the shipped form are unnecessary here because each value's caller
 * reads red before this function returns and the arrays are not reused until the next launch. */
__device__ __forceinline__ void block_sum_pair_hand(float v_a, float v_b, float* red_a, float* red_b,
                                                   float* out_a, float* out_b) {
    v_a += shfl_down_bits(v_a, 16);
    v_a += shfl_down_bits(v_a, 8);
    v_a += shfl_down_bits(v_a, 4);
    v_a += shfl_down_bits(v_a, 2);
    v_a += shfl_down_bits(v_a, 1);
    v_b += shfl_down_bits(v_b, 16);
    v_b += shfl_down_bits(v_b, 8);
    v_b += shfl_down_bits(v_b, 4);
    v_b += shfl_down_bits(v_b, 2);
    v_b += shfl_down_bits(v_b, 1);
    if ((threadIdx.x & 31) == 0) {
        red_a[threadIdx.x >> 5] = v_a;
        red_b[threadIdx.x >> 5] = v_b;
    }
    __syncthreads();
    int warps = blockDim.x >> 5;
    float t_a = (threadIdx.x < warps) ? red_a[threadIdx.x] : 0.f;
    float t_b = (threadIdx.x < warps) ? red_b[threadIdx.x] : 0.f;
    t_a += shfl_down_bits(t_a, 16);
    t_b += shfl_down_bits(t_b, 16);
    t_a += shfl_down_bits(t_a, 8);
    t_b += shfl_down_bits(t_b, 8);
    t_a += shfl_down_bits(t_a, 4);
    t_b += shfl_down_bits(t_b, 4);
    t_a += shfl_down_bits(t_a, 2);
    t_b += shfl_down_bits(t_b, 2);
    t_a += shfl_down_bits(t_a, 1);
    t_b += shfl_down_bits(t_b, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        red_a[0] = t_a;
        red_b[0] = t_b;
    }
    __syncthreads();
    *out_a = red_a[0];
    *out_b = red_b[0];
}

/* The same pair, compiler-lowered shuffles — MODE 2's control against MODE 3's spelling, so the
 * barrier lever and the shuffle-spelling lever stay separately attributable. */
__device__ __forceinline__ void block_sum_pair_comp(float v_a, float v_b, float* red_a, float* red_b,
                                                   float* out_a, float* out_b) {
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 16);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 8);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 4);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 2);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 1);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 16);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 8);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 4);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 2);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 1);
    if ((threadIdx.x & 31) == 0) {
        red_a[threadIdx.x >> 5] = v_a;
        red_b[threadIdx.x >> 5] = v_b;
    }
    __syncthreads();
    int warps = blockDim.x >> 5;
    float t_a = (threadIdx.x < warps) ? red_a[threadIdx.x] : 0.f;
    float t_b = (threadIdx.x < warps) ? red_b[threadIdx.x] : 0.f;
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 16);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 16);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 8);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 8);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 4);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 4);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 2);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 2);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 1);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        red_a[0] = t_a;
        red_b[0] = t_b;
    }
    __syncthreads();
    *out_a = red_a[0];
    *out_b = red_b[0];
}

/* ---- MODE 0: the shipped recurrence, verbatim (the control row) --------------------------------- */
__global__ void rec0_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, const float* __restrict__ g,
                            const float* __restrict__ beta, float* __restrict__ S,
                            float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;                       // 128 threads, thread j owns column j
    __shared__ float ks[D], qs[D], red[32];

    float kq = k[h * D + j], qq = q[h * D + j];
    float kn = kq * rsqrtf(block_sum(kq * kq, red) + 1e-6f);
    __syncthreads();
    float qn = qq * rsqrtf(block_sum(qq * qq, red) + 1e-6f) * Q_SCALE;
    __syncthreads();
    ks[j] = kn;
    qs[j] = qn;
    __syncthreads();

    float gam = expf(g[h]), bet = beta[h];
    float* Sh = S + (size_t)h * D * D;

    float r = 0.f;
    for (int i = 0; i < D; ++i) r += Sh[i * D + j] * ks[i];
    float dv = bet * (v[h * D + j] - gam * r);
    float acc = 0.f;
    for (int i = 0; i < D; ++i) {
        float s = gam * Sh[i * D + j];
        s += ks[i] * dv;
        Sh[i * D + j] = s;
        acc += qs[i] * s;
    }
    o[h * D + j] = acc;
}

/* ---- MODE 1: hand-spelled fold, shipped structure (isolates the shuffle lowering) --------------- */
__global__ void rec1_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, const float* __restrict__ g,
                            const float* __restrict__ beta, float* __restrict__ S,
                            float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float ks[D], qs[D], red[32];

    float kq = k[h * D + j], qq = q[h * D + j];
    float kn = kq * rsqrtf(block_sum_hand(kq * kq, red) + 1e-6f);
    __syncthreads();
    float qn = qq * rsqrtf(block_sum_hand(qq * qq, red) + 1e-6f) * Q_SCALE;
    __syncthreads();
    ks[j] = kn;
    qs[j] = qn;
    __syncthreads();

    float gam = expf(g[h]), bet = beta[h];
    float* Sh = S + (size_t)h * D * D;

    float r = 0.f;
    for (int i = 0; i < D; ++i) r += Sh[i * D + j] * ks[i];
    float dv = bet * (v[h * D + j] - gam * r);
    float acc = 0.f;
    for (int i = 0; i < D; ++i) {
        float s = gam * Sh[i * D + j];
        s += ks[i] * dv;
        Sh[i * D + j] = s;
        acc += qs[i] * s;
    }
    o[h * D + j] = acc;
}

/* ---- MODE 2: one fused block reduction + hand spelling (adds the barrier lever) ----------------- */
__global__ void rec2_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, const float* __restrict__ g,
                            const float* __restrict__ beta, float* __restrict__ S,
                            float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float ks[D], qs[D], redk[32], redq[32];

    float kq = k[h * D + j], qq = q[h * D + j];
    float sk, sq;
    block_sum_pair_hand(kq * kq, qq * qq, redk, redq, &sk, &sq);
    float kn = kq * rsqrtf(sk + 1e-6f);
    float qn = qq * rsqrtf(sq + 1e-6f) * Q_SCALE;
    __syncthreads();                       // protects redk/redq; ks/qs are thread-private slots
    ks[j] = kn;
    qs[j] = qn;
    __syncthreads();

    float gam = expf(g[h]), bet = beta[h];
    float* Sh = S + (size_t)h * D * D;

    float r = 0.f;
    for (int i = 0; i < D; ++i) r += Sh[i * D + j] * ks[i];
    float dv = bet * (v[h * D + j] - gam * r);
    float acc = 0.f;
    for (int i = 0; i < D; ++i) {
        float s = gam * Sh[i * D + j];
        s += ks[i] * dv;
        Sh[i * D + j] = s;
        acc += qs[i] * s;
    }
    o[h * D + j] = acc;
}

/* The same pair with compiler shuffles, so MODE 2's delta is not credited to the spelling. */
__global__ void rec2c_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ g,
                             const float* __restrict__ beta, float* __restrict__ S,
                             float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float ks[D], qs[D], redk[32], redq[32];

    float kq = k[h * D + j], qq = q[h * D + j];
    float sk, sq;
    block_sum_pair_comp(kq * kq, qq * qq, redk, redq, &sk, &sq);
    float kn = kq * rsqrtf(sk + 1e-6f);
    float qn = qq * rsqrtf(sq + 1e-6f) * Q_SCALE;
    __syncthreads();
    ks[j] = kn;
    qs[j] = qn;
    __syncthreads();

    float gam = expf(g[h]), bet = beta[h];
    float* Sh = S + (size_t)h * D * D;

    float r = 0.f;
    for (int i = 0; i < D; ++i) r += Sh[i * D + j] * ks[i];
    float dv = bet * (v[h * D + j] - gam * r);
    float acc = 0.f;
    for (int i = 0; i < D; ++i) {
        float s = gam * Sh[i * D + j];
        s += ks[i] * dv;
        Sh[i * D + j] = s;
        acc += qs[i] * s;
    }
    o[h * D + j] = acc;
}
