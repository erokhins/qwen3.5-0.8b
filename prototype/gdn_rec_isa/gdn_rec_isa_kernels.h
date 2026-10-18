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
 *  2c4 `rec2c4_kernel` MODE 2c's fused prologue MERGED with the shipping unroll4 walks
 *                     (`engine/gdn_rec_unroll4_kernels.h`). It exists to be CENSUSED, not to be timed:
 *                     `bench/performance-status.md` §18 / D46 make a compile-only census of this exact
 *                     listing the gate in front of any `rec2c` port, because a merge that loses the
 *                     unroll to register pressure is invisible to a burst of the two unmerged forms.
 *  loop `recloop_kernel` the SHIPPING form (MODE 2c4, which is `engine/gdn_rec_fused_kernels.h`)
 *                     with the TOKEN index moved inside the kernel: one launch walks `tokens` tokens
 *                     instead of one. This is D53's item — launch structure, not math. The per-token
 *                     body is verbatim `rec2c4_kernel`'s (see its own header), the grid is the same
 *                     `<<<NH, D>>>`, the state is the same global buffer at the same addresses, and the
 *                     only thing that moves is how many token-steps one submission carries. Admitted
 *                     by bit-identity like every spelling MODE; its A/B partner is `rec2c4`, the
 *                     spelling the engine runs, and NOT `rec0`.
 *
 * The three DEPENDENCY-BREAK probes at the bottom of this file (`recx_indep`, `recx_unchain`,
 * `recx_both`) are not MODEs in this sense. They are numerically meaningless by design, they are
 * admitted by VOLUME rather than by value, and the block comment above them says what each one breaks
 * and what its result would falsify. Read that block before running one.
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

/* ---- MODE 2c4: MODE 2c's fused prologue MERGED with the SHIPPING unroll4 walks -------------------
 *
 * WHY THIS KERNEL EXISTS, AND WHY IT IS NEVER LAUNCHED. `rec2c`'s burst screen (`bench/performance-status.md`
 * §18: median 1.0268 against D25's 1.02 kill line, one case below it) compared the fused prologue against
 * a PRE-unroll4 base, while the shipped kernel is `engine/gdn_rec_unroll4_kernels.h::rec_unroll4_kernel`
 * (D33). The merge has therefore never existed, and its failure mode is silent: the unroll-4 walks hold
 * four state values plus their addressing live per trip, the fused prologue holds both partial reduction
 * chains live at once, and if the two together exceed what the allocator will keep in VGPRs the compiler
 * gives the unroll back — a kernel that is fused and slower, in a binary no burst has ever run. D46 names
 * this listing, censused compile-only, as the gate in front of any port for exactly that reason.
 *
 * WHAT IS TRANSCRIBED FROM WHERE, because bit-identity is owed to the shipped SOURCE TEXT, not to the
 * algebra (`prototype/gdn_collapse/results.md` §2: a hoisted scalar constant moved 93,522 of 131,072 `o`
 * elements and a CPU mirror could not see it):
 *   * prologue  — MODE 2c's, character for character (`block_sum_pair_comp` + the two `redk`/`redq`
 *                 arrays + the two remaining barriers);
 *   * both state walks — `engine/gdn_rec_unroll4_kernels.h::rec_unroll4_kernel`, character for character
 *                 including `#pragma unroll 4` on both loops and the order of every multiply inside
 *                 every add. That pragma is a measured constraint there, not style: without it LLVM fully
 *                 unrolls loop A and the two forms are no longer the same listing shape.
 * Nothing else moved. Diff this body against those two sources before believing a delta here; a CPU
 * mirror of the merge cannot see an association change, because two sides that share a formula agree.
 *
 * ADMITTED BY BIT-IDENTITY, like MODE 2c: same 5+5 shuffle steps per value in the same order, same
 * reduction association, same state walk terms. If the device ever says it is NOT bit-identical to
 * `rec0`, that is LLVM re-associating under a spelling change (the D25 `q̃` class) and a finding, not
 * something to route around. Censused here and, if the census keeps both halves, in the harness as a
 * MODE before it is ever offered to `engine/`. */
__global__ void rec2c4_kernel(const float* __restrict__ q, const float* __restrict__ k,
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

    // ---- from here to the end: `rec_unroll4_kernel`'s body, verbatim (see the note above) ---------
    float r = 0.f;
#pragma unroll 4                      // keep the loop: without the pragma LLVM fully unrolls this one
    for (int i = 0; i < D; i += 4) {  // side and the two forms stop being bill-matched (see the header)
        r += Sh[(i + 0) * D + j] * ks[i + 0];
        r += Sh[(i + 1) * D + j] * ks[i + 1];
        r += Sh[(i + 2) * D + j] * ks[i + 2];
        r += Sh[(i + 3) * D + j] * ks[i + 3];
    }
    float dv = bet * (v[h * D + j] - gam * r);

    float acc = 0.f;
#pragma unroll 4
    for (int i = 0; i < D; i += 4) {
        float s0 = gam * Sh[(i + 0) * D + j]; s0 += ks[i + 0] * dv; Sh[(i + 0) * D + j] = s0;
        acc += qs[i + 0] * s0;
        float s1 = gam * Sh[(i + 1) * D + j]; s1 += ks[i + 1] * dv; Sh[(i + 1) * D + j] = s1;
        acc += qs[i + 1] * s1;
        float s2 = gam * Sh[(i + 2) * D + j]; s2 += ks[i + 2] * dv; Sh[(i + 2) * D + j] = s2;
        acc += qs[i + 2] * s2;
        float s3 = gam * Sh[(i + 3) * D + j]; s3 += ks[i + 3] * dv; Sh[(i + 3) * D + j] = s3;
        acc += qs[i + 3] * s3;
    }
    o[h * D + j] = acc;
}

/* ---- MODE loop: the shipped form, with the TOKEN index inside the launch (D53) -------------------
 *
 * THE ITEM. `bench/performance-status.md` §22 counts 73,728 `rec_*` launches for a 4096-token prompt
 * (18 GDN layers x 4,096 tokens) at ~5.8 µs each against a measured 1.72 µs empty-launch floor
 * (§11, `tools/submission_probe.cpp`), and §11 established the premise that makes a launch-count item
 * legal at all: submissions are NOT absorbed by a busy stream, which is why deleting 4,080 `copy`
 * launches moved 1.38 % of a wave. The external existence proof walks 512 tokens inside one launch.
 * This MODE is the same shape with our arithmetic: one submission per (layer, span) instead of per
 * (layer, token).
 *
 * WHY THE ARITHMETIC CANNOT MOVE, AND HOW THAT IS CHECKED. The recurrence is already sequential —
 * token t reads the state token t-1 wrote — so running steps t and t+1 in one block instead of in two
 * launches changed the schedule, not the values. That claim is enforced three ways, not asserted:
 *
 *   1. the body below is `rec2c4_kernel`'s text VERBATIM (identical down to the `#pragma unroll 4`
 *      lines and the order of every multiply). The operand pointers are re-declared per iteration in
 *      the inner scope so that not one arithmetic line had to be rewritten to add `t`; the only new
 *      lines are the `for`, the six pointer declarations that name this token's row, and the closing
 *      braces. `check_recloop_transcription.py` strips the wrapper and diffs the body against the
 *      shipping kernel, and it carries its own must-catch mutation so the checker cannot pass vacuously;
 *   2. the CPU witness in `gdn_rec_isa.hip.cpp --selftest-loop` compares the looped ADDRESS mapping
 *      against the per-token mapping element for element, at several span sizes, with defects injected
 *      (t pinned, `o` row stride wrong, state restarted per token, `Q_SCALE` hoisted, `g|beta` stride
 *      wrong) whose must-catch sets are DERIVED from each case's geometry (token count, decay sum,
 *      starting state) rather than hand-listed;
 *   3. the device admission is bit-identity against `rec0` on `o` AND final state, which is the only
 *      gate that can see a compiler-side re-association — the §19 `q̃` class, which no host mirror and
 *      no textual diff can see, because two spellings that share a formula agree.
 *
 * WHAT IS *NOT* DONE HERE, deliberately: no LDS-resident state (that is Route A,
 * `engine/gdn_collapse_kernels.h`, which re-spells the thread mapping, does 4x redundant work per
 * token and changed the association — `prototype/gdn_collapse/results.md` §5 prices what that cost),
 * no grid change, no block size change, no state staging, no prefetch. This MODE isolates ONE axis:
 * submissions per token-step. If it wins, the launch structure was the cost; if it is flat, D53 is
 * falsified in situ and the family is what `prototype/gdn_collapse/results.md` §5 said it is.
 *
 * THE SHARED-LDS REUSE QUESTION, argued because the loop makes it live. `block_sum_pair_comp` ends
 * with `*out_a = red_a[0]` and NO trailing barrier — sound per launch only because nothing writes
 * `red_a` again before the launch ends, which is exactly what its comment claims ("the arrays are not
 * reused until the next launch"). Inside a token loop they ARE reused, so the claim needs the real
 * reason instead: the per-iteration write to `red_a`/`red_b` happens in the next iteration's
 * `block_sum_pair_comp`, which every thread reaches only after passing the `__syncthreads()` that
 * sits between the `red[0]` READ of iteration t (line `*out_a = red_a[0]`) and iteration t+1's write;
 * `ks`/`qs` are covered by the pair of barriers that already bracket their write. All threads run the
 * same trip count (`tokens` is uniform, nothing diverges it), so every barrier is hit by every thread
 * the same number of times. No barrier was added, and if one is ever needed the witness is the
 * harness's own `rerun_ndiff_*` columns (`--repeat 2` bit-compares two runs of the same MODE): a
 * missing barrier is a RACE and a race disagrees with itself before it disagrees with MODE 0.
 *
 * THE STRIDES ARE DERIVED, NOT PASSED. `CH` (6144) and `NH*D` (2048) are the same constants the
 * per-token call sites use to walk `sc.g_u` / `sc.g_o` (`engine/decode_step.cpp`'s
 * `sc.g_u + t*kQkvRows`, `sc.g_o + t*kValueDim`), and the static_asserts below make the equality a
 * compile error instead of a comment. Passing row strides as arguments would let a wrong stride be
 * expressible, and a wrong stride here is a silent numerics change, not a crash.
 */
static_assert(CH == 3 * NH * D, "recloop's q|k|v row stride is CH; must be 3*NH*D = the engine's kQkvRows");
static_assert(NH * D == 2048, "recloop's o row stride is NH*D; must be the engine's kValueDim");

__global__ void recloop_kernel(const float* __restrict__ qtile, const float* __restrict__ ktile,
                               const float* __restrict__ vtile, const float* __restrict__ gtile,
                               const float* __restrict__ betatile, float* __restrict__ S,
                               float* __restrict__ otile, int nheads, int tokens) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float ks[D], qs[D], redk[32], redq[32];

    for (int t = 0; t < tokens; ++t) {
        // This token's rows, named so that every line below is `rec2c4_kernel`'s verbatim. `qtile`/`ktile`/
        // `vtile` are the three 2048-wide slices of a [n][6144] tile, so a token step is `+CH`; `g`/`beta`
        // are [n][16]; `o` is [n][2048]. Same buffers, same addresses the per-token launches were given.
        const float* __restrict__ q = qtile + (size_t)t * CH;
        const float* __restrict__ k = ktile + (size_t)t * CH;
        const float* __restrict__ v = vtile + (size_t)t * CH;
        const float* __restrict__ g = gtile + (size_t)t * NH;
        const float* __restrict__ beta = betatile + (size_t)t * NH;
        float* __restrict__ o = otile + (size_t)t * (NH * D);

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

        // ---- from here to `o[h * D + j] = acc;`: `rec_unroll4_kernel`'s body, verbatim (see the note) --
        float r = 0.f;
#pragma unroll 4                      // keep the loop: without the pragma LLVM fully unrolls this one
        for (int i = 0; i < D; i += 4) {  // side and the two forms stop being bill-matched (see the header)
            r += Sh[(i + 0) * D + j] * ks[i + 0];
            r += Sh[(i + 1) * D + j] * ks[i + 1];
            r += Sh[(i + 2) * D + j] * ks[i + 2];
            r += Sh[(i + 3) * D + j] * ks[i + 3];
        }
        float dv = bet * (v[h * D + j] - gam * r);

        float acc = 0.f;
#pragma unroll 4
        for (int i = 0; i < D; i += 4) {
            float s0 = gam * Sh[(i + 0) * D + j]; s0 += ks[i + 0] * dv; Sh[(i + 0) * D + j] = s0;
            acc += qs[i + 0] * s0;
            float s1 = gam * Sh[(i + 1) * D + j]; s1 += ks[i + 1] * dv; Sh[(i + 1) * D + j] = s1;
            acc += qs[i + 1] * s1;
            float s2 = gam * Sh[(i + 2) * D + j]; s2 += ks[i + 2] * dv; Sh[(i + 2) * D + j] = s2;
            acc += qs[i + 2] * s2;
            float s3 = gam * Sh[(i + 3) * D + j]; s3 += ks[i + 3] * dv; Sh[(i + 3) * D + j] = s3;
            acc += qs[i + 3] * s3;
        }
        o[h * D + j] = acc;
    }
}

/* ============================ THE DEPENDENCY-BREAK PROBES (D30 queue item 1.3) ================== */
/*
 * WHY THESE EXIST, AND WHAT THEY ARE NOT. `results.md` §4 read `gdn_rec` as latency-bound on the
 * per-token dependency chain at ~1.6 *working* wavefronts per CU, and D25's falsifier (rec0/rec2c =
 * 1.0159x) closed the INSTRUCTION route without ever testing that reading: an instruction cut cannot
 * refute a latency model, it can only shorten the stream. The cheap instrument that can is a form
 * whose DEPENDENCY STRUCTURE is broken while its VOLUME is held. That is what these three are.
 *
 * THEY ARE NUMERICALLY MEANINGLESS, AND THAT IS THE POINT. None of them reproduces §04.4 and none is
 * offered as a lever, a candidate, or a numerics proposal. `gdn_rec_isa.hip.cpp` admits MODEs 1/2/2c
 * by bit-identity, admits these by VOLUME (and, for the one that only re-associates, by the §10.1 G3i
 * band), and refuses to time anything whose volume moved. A probe that quietly removed work would come
 * back fast for the wrong reason — that is the `tools/mixed_bw` bug (a per-thread store to a per-block
 * slot made the read checksum report 1/blockDim of the bytes it had read, every warm row failed, and
 * the ms/GB/s column looked entirely plausible), so the volume witness ships INSIDE this directory
 * and must pass before a single launch is timed: `--volume` on the binary (device: elements written,
 * read side untouched where the form claims it, guard bands, the host-derived per-launch byte account
 * compared against MODE 0's) and `traffic_census.py --volume-check` (machine code: `global_load_b32`
 * and `global_store_b32` per token-step and bytes/launch, identical to MODE 0 or non-zero exit).
 *
 * WHAT THE CHAIN IS, from `rec0_kernel`. One launch is one (token, layer); a thread owns one d_v
 * column j and runs prologue (two block reductions) -> loop A (128 state loads, 128 FMAs into ONE
 * accumulator `r`) -> `dv` -> loop B (128 iterations that read Sh[i][j], write Sh[i][j], and feed ONE
 * accumulator `acc`). Two dependencies exist and they are separable:
 *
 *   CROSS-TOKEN. Token t+1's first state load is the address token t's last state store wrote — one
 *     launch per token (`engine/decode_step.cpp:752-762`), so this is a RAW/WAR round trip through
 *     global memory, and it is also what forces the `s_waitcnt vmcnt(0)` store drain inside loop B
 *     (16 per token-step, `results.md` §3 class 3). `recx_indep_kernel` breaks it: read one buffer,
 *     write another, so no launch waits on its predecessor's store.
 *
 *   INTRA-TOKEN. The two serial accumulator chains: 128 dependent FMAs, twice, at ~4-cycle FMA
 *     latency with ~1.6 working wavefronts per CU to hide it (`prototype/gdn_collapse/results.md` §5).
 *     `recx_unchain_kernel` shortens each chain from 128 links to 32 at identical volume. Its ceiling
 *     is calculable and SMALL: ~2 x 96 saved dependent links ~= 0.09-0.17 us of a 6.3-6.9 us launch,
 *     i.e. ~1.015-1.027x. A LARGE `unchain` effect would therefore not be the accumulator chain and
 *     has to be looked for elsewhere; a null is the expected result and is still informative.
 *     `recx_shape_kernel` is its bill-matched control (same unroll-4 spelling, chain NOT broken, and
 *     bit-identical to MODE 0), because the census found the unchain probe 10.5 % cheaper in slots —
 *     see its own comment for the difference table this makes readable.
 *
 * IDENTICAL VOLUME, enforced not claimed: same grid NH x D, same LDS (ks[D], qs[D], red[32]), same two
 * state walks, two state read passes and one state write pass over the same byte count, `block_sum`
 * called verbatim, same operands read, same launch count.
 *
 * NAMING. These are NOT the reserved MODE 3 of the header above (`rec3_kernel` = norms out of the
 * launch, still unwritten, and it is a LEVER that must be bit-identical). The probes are named for
 * what they break, because they are probes and can never be levers.
 */

/* indep: the cross-token dependency, at fixed volume. Reads `S_in`, writes `S_out`; every address this
 * form reads is an address it does not write, so no launch in a sequence waits for an earlier launch's
 * store, and the in-loop WAR that needs the vmcnt(0) drain is gone with it. Bytes are NOT reduced: one
 * read pair over a 1 MiB region plus one write pass over another 1 MiB region is MODE 0's 3 MiB per
 * launch exactly, just split in two. The harness can index a POOL of slots (--pool N) for the literal
 * "indexed per token" form, but the default is ONE slot: the split already breaks the dependency, and
 * a bigger pool only inflates the working set — which is a TRAFFIC change wearing a latency costume,
 * and this directory has the number to prove it: `prototype/gdn_decode/results.md` measures this very
 * access pattern at 876 GB/s with an 18 MiB warm state and 255 GB/s once tiled past ~64 MiB. Footprint
 * belongs behind a flag, never inside the lever. */
__global__ void recx_indep_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                  const float* __restrict__ v, const float* __restrict__ g,
                                  const float* __restrict__ beta, const float* __restrict__ S_in,
                                  float* __restrict__ S_out, float* __restrict__ o, int nheads) {
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
    const float* __restrict__ Sh_in = S_in + (size_t)h * D * D;
    float* __restrict__ Sh_out = S_out + (size_t)h * D * D;

    float r = 0.f;
    for (int i = 0; i < D; ++i) r += Sh_in[i * D + j] * ks[i];
    float dv = bet * (v[h * D + j] - gam * r);
    float acc = 0.f;
    for (int i = 0; i < D; ++i) {
        float s = gam * Sh_in[i * D + j];
        s += ks[i] * dv;
        Sh_out[i * D + j] = s;
        acc += qs[i] * s;
    }
    o[h * D + j] = acc;
}

/* unchain: the intra-token serial chains, shortened at fixed volume. Same buffer, same addresses, same
 * load/store mix, same number of FMAs and adds (four partials carry the same 128 terms and cost 3
 * adds each to combine). What changes is DEPTH: each chain goes from 128 dependent links to 32.
 * Written as ONE loop carrying four explicit element bodies per trip, not four sub-loops, so the
 * listing still has exactly two state walks — `isa_census.py` refuses a listing whose walk count is
 * not two, which is the right way for a geometry change to fail. Re-associates both dots, so it is a
 * numerics change: §04.4 measures these sums cancelling (1 element in 286 losing 84 % of its
 * magnitude), which is why admission is the §10.1 G3i band and never bit-identity. */
__global__ void recx_unchain_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                    const float* __restrict__ v, const float* __restrict__ g,
                                    const float* __restrict__ beta, float* __restrict__ S,
                                    float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;
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

    float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
#pragma unroll 4                      // same shape as `recx_shape_kernel`, so the pair is bill-matched
    for (int i = 0; i < D; i += 4) {
        r0 += Sh[(i + 0) * D + j] * ks[i + 0];
        r1 += Sh[(i + 1) * D + j] * ks[i + 1];
        r2 += Sh[(i + 2) * D + j] * ks[i + 2];
        r3 += Sh[(i + 3) * D + j] * ks[i + 3];
    }
    float r = (r0 + r1) + (r2 + r3);
    float dv = bet * (v[h * D + j] - gam * r);

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int i = 0; i < D; i += 4) {
        float s0 = gam * Sh[(i + 0) * D + j]; s0 += ks[i + 0] * dv; Sh[(i + 0) * D + j] = s0;
        float s1 = gam * Sh[(i + 1) * D + j]; s1 += ks[i + 1] * dv; Sh[(i + 1) * D + j] = s1;
        float s2 = gam * Sh[(i + 2) * D + j]; s2 += ks[i + 2] * dv; Sh[(i + 2) * D + j] = s2;
        float s3 = gam * Sh[(i + 3) * D + j]; s3 += ks[i + 3] * dv; Sh[(i + 3) * D + j] = s3;
        a0 += qs[i + 0] * s0;
        a1 += qs[i + 1] * s1;
        a2 += qs[i + 2] * s2;
        a3 += qs[i + 3] * s3;
    }
    o[h * D + j] = (a0 + a1) + (a2 + a3);
}

/* shape: the BILL-MATCHED CONTROL for `unchain`, and the only one of these that is meant to be
 * bit-identical. Needed because the census caught the probe it controls being CHEAPER, not just
 * shallower: `recx_unchain_kernel` comes out 160 slots (10.5 %) under MODE 0 — four bodies per trip
 * fold the index arithmetic that MODE 0's one-per-trip spelling leaves behind — and D25's measured slot
 * elasticity on this very kernel is ~0.8, so up to 8.4 % of any `rec4` speedup is explainable by its
 * instruction bill alone and could not be attributed to the chain at all.
 *
 * This kernel is `unchain` with the partials folded back into ONE accumulator in MODE 0's own order
 * (i = 0,1,2,3,…), so it keeps the unroll-4 shape and therefore nearly the same bill, and gives back the
 * 128-link chain. The addition tree is MODE 0's addition tree term for term, so it is admitted by
 * BIT-IDENTITY like the levers, not by volume alone: if the device says it is not bit-identical, that is
 * LLVM re-associating under a spelling change and it is a finding about the prologue/barrier lesson
 * (D25's q̃ row in `engine/gdn_collapse_kernels.h`), not something to route around.
 *
 * It turns the two probes into a difference that means something:
 *     rec6 - rec0   the loop-shape and bill effect, chain NOT broken
 *     rec4 - rec6   the chain-depth effect at matched shape and matched bill
 * so a `rec4` win larger than `rec6`'s is the chain, and one equal to it is nothing but the bill. */
__global__ void recx_shape_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                  const float* __restrict__ v, const float* __restrict__ g,
                                  const float* __restrict__ beta, float* __restrict__ S,
                                  float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;
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
#pragma unroll 4                      // keep the loop: see the comment above — this kernel exists to
    for (int i = 0; i < D; i += 4) {   // match `unchain`'s bill, and LLVM fully unrolls it otherwise
        r += Sh[(i + 0) * D + j] * ks[i + 0];
        r += Sh[(i + 1) * D + j] * ks[i + 1];
        r += Sh[(i + 2) * D + j] * ks[i + 2];
        r += Sh[(i + 3) * D + j] * ks[i + 3];
    }
    float dv = bet * (v[h * D + j] - gam * r);

    float acc = 0.f;
#pragma unroll 4
    for (int i = 0; i < D; i += 4) {
        float s0 = gam * Sh[(i + 0) * D + j]; s0 += ks[i + 0] * dv; Sh[(i + 0) * D + j] = s0;
        acc += qs[i + 0] * s0;
        float s1 = gam * Sh[(i + 1) * D + j]; s1 += ks[i + 1] * dv; Sh[(i + 1) * D + j] = s1;
        acc += qs[i + 1] * s1;
        float s2 = gam * Sh[(i + 2) * D + j]; s2 += ks[i + 2] * dv; Sh[(i + 2) * D + j] = s2;
        acc += qs[i + 2] * s2;
        float s3 = gam * Sh[(i + 3) * D + j]; s3 += ks[i + 3] * dv; Sh[(i + 3) * D + j] = s3;
        acc += qs[i + 3] * s3;
    }
    o[h * D + j] = acc;
}

/* both: the two breaks together. Two levers that each hide behind the other read as two nulls; if the
 * cross-token round trip and the FMA chain are both on the critical path, only the pair shows it, and
 * if the pair is a null too, the chain reading needs a different model rather than a wider band. */
__global__ void recx_both_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta, const float* __restrict__ S_in,
                                 float* __restrict__ S_out, float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;
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
    const float* __restrict__ Sh_in = S_in + (size_t)h * D * D;
    float* __restrict__ Sh_out = S_out + (size_t)h * D * D;

    float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
    for (int i = 0; i < D; i += 4) {
        r0 += Sh_in[(i + 0) * D + j] * ks[i + 0];
        r1 += Sh_in[(i + 1) * D + j] * ks[i + 1];
        r2 += Sh_in[(i + 2) * D + j] * ks[i + 2];
        r3 += Sh_in[(i + 3) * D + j] * ks[i + 3];
    }
    float r = (r0 + r1) + (r2 + r3);
    float dv = bet * (v[h * D + j] - gam * r);

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int i = 0; i < D; i += 4) {
        float s0 = gam * Sh_in[(i + 0) * D + j]; s0 += ks[i + 0] * dv; Sh_out[(i + 0) * D + j] = s0;
        float s1 = gam * Sh_in[(i + 1) * D + j]; s1 += ks[i + 1] * dv; Sh_out[(i + 1) * D + j] = s1;
        float s2 = gam * Sh_in[(i + 2) * D + j]; s2 += ks[i + 2] * dv; Sh_out[(i + 2) * D + j] = s2;
        float s3 = gam * Sh_in[(i + 3) * D + j]; s3 += ks[i + 3] * dv; Sh_out[(i + 3) * D + j] = s3;
        a0 += qs[i + 0] * s0;
        a1 += qs[i + 1] * s1;
        a2 += qs[i + 2] * s2;
        a3 += qs[i + 3] * s3;
    }
    o[h * D + j] = (a0 + a1) + (a2 + a3);
}
