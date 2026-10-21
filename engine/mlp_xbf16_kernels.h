/*
 * engine/mlp_xbf16_kernels.h — the MLP's two decode GEMVs with the activation staged as BF16, so the
 * widening launch in front of them can be deleted (route S1 item 1, the `cast` family).
 *
 * WHY A SIBLING HEADER AND NOT AN EDIT. `prototype/mlp_gemv/mlp_gemv_kernels.h` is the gated op source
 * and prototype/README.md forbids editing it for the engine's convenience. This file therefore
 * INCLUDES it — so `bf`, `fb`, `rnd`, `reduce_row` and the `NEG_*` vocabulary are the same spellings the
 * op's own gate ran, not a paraphrase of them — and adds a second kernel per op whose only difference
 * is the dtype of the staged activation row.
 *
 * WHAT CHANGES, exactly one thing. `mlp_gateup` stages `x` into `__shared__ float xs[IN]` and the
 * caller therefore has to hand it F32; on the hand route `decode_step.cpp` buys that F32 with one
 * `kernels_cast.cpp` launch per layer for `sc.norm` (24/token) and one for `sc.mlp_act` (24/token).
 * Here `xs` is `hip_bfloat16` and the widen happens in registers at the use site, so the caller hands
 * the BF16 tile the producer already wrote and those launches go away.
 *
 * WHY THAT IS BIT-IDENTICAL AND NOT "BAND-ADJACENT". `hip_bfloat16 -> float` is exact: same sign, same
 * 8-bit exponent, the 7 mantissa bits zero-extended into 23, no rounding, no reassociation. The value
 * `tof(xs[k])` produces is therefore the same float the cast buffer held at that index, element for
 * element, which means the products are the same floats, the word walk is the same walk, `reduce_row`
 * is the same ladder, and the epilogue below — three roundings over the two dots, nothing to do with
 * `x` — is untouched. `engine/generation-limits.md` §1's `cast` row is the same claim from the other
 * side: those 97 launches move 0.371 MB read / 0.741 MB written per token, which is no bytes of
 * consequence, so this is a submission lever, not an arithmetic one.
 *
 * WHAT IS NOT CLAIMED. That it is FASTER. The staged row is half the LDS and one `v_bfe`-class
 * conversion per element instead of a separate launch, and `engine/results.md`'s cast row records a
 * measured precedent AGAINST that trade for the projection family (`PACK_X` cost +0.278 ms over 48
 * launches to save 48 casts, quoted). Whether the residual survives on the engine's own shape is the
 * wall A/B's question (`prototype/cast_fold/ab_item1.sh`), not this header's.
 *
 * ONE-ROW FORMS ONLY, like `engine/proj_group_kernels.h`. The token-batched pair
 * (`mlp_gateup_tokens`/`mlp_down_tokens`) is what a prefill wave runs and is left alone, so under the
 * fold a wide body keeps its widenings and only a body of one row — a generation step — folds.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

// The gated op source: same bf/fb/rnd/reduce_row, same NEG_* vocabulary, same macro definitions. It is
// already included by the TU that includes this one; the guard makes that free.
#include "prototype/mlp_gemv/mlp_gemv_kernels.h"

namespace engine::mlpxbf16 {

// The prototype's `accum_word` with the staged row typed BF16. One spelling per concept: this is the
// only line in either kernel that knows the operand dtype.
__device__ __forceinline__ void accum_word_xbf16(float& acc, uint4 u,
                                                 const hip_bfloat16* __restrict__ xp) {
    const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += bf(p[j]) * bf(xp[j]);
}

// `mlp_gateup` with a BF16 staged row. Transcribed from the gated kernel body for body; the weight
// walks, the two accumulators, the reduction, the NEG_* vocabulary and the three-rounding SwiGLU
// epilogue are its, unchanged.
template <int IN, int MAP = MAP_INTERLEAVE>
__global__ void mlp_gateup_xbf16(const hip_bfloat16* __restrict__ Wg,
                                 const hip_bfloat16* __restrict__ Wu,
                                 const hip_bfloat16* __restrict__ x,
                                 hip_bfloat16* __restrict__ act, hip_bfloat16* __restrict__ pre,
                                 int rows, int rpb, int neg) {
    __shared__ hip_bfloat16 xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    const uint4* wg = reinterpret_cast<const uint4*>(Wg + (size_t)row * IN);
    const uint4* wu = reinterpret_cast<const uint4*>(Wu + (size_t)row * IN);
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
    float ag = 0.f, au = 0.f;
    if (MAP == MAP_CONTIG) {
#pragma unroll
        for (int v = lane; v < words; v += ws) accum_word_xbf16(ag, wg[v], &xs[v * 8]);
#pragma unroll
        for (int v = lane; v < words; v += ws) accum_word_xbf16(au, wu[v], &xs[v * 8]);
    } else {
#pragma unroll
        for (int v = lane; v < words; v += ws) {
            const hip_bfloat16* xp = &xs[v * 8];
            accum_word_xbf16(ag, wg[v], xp);
            accum_word_xbf16(au, wu[v], xp);
        }
    }
    ag = reduce_row(ag);
    au = reduce_row(au);
    if (lane != 0) return;
    hip_bfloat16 g = rnd(ag, neg), u = rnd(au, neg);
    const int orow = neg == NEG_SWAPROW ? (row ^ 1) : row;
    if (neg == NEG_F32ACT) {              // the wrong reading, for the negative test only
        float a = (ag * (1.0f / (1.0f + expf(-ag)))) * au;
        act[orow] = rnd(a, neg);
    } else {
        float gf = bf(g);
        float sg = bf(fb(gf * (1.0f / (1.0f + expf(-gf)))));
        act[orow] = rnd(sg * bf(u), neg);
    }
    if (pre) {
        const bool sw = neg == NEG_PRESWAP;
        pre[orow] = sw ? u : g;
        pre[rows + orow] = sw ? g : u;
    }
}

// `mlp_down` with a BF16 staged row — the same fold on the other side of `act`.
template <int IN>
__global__ void mlp_down_xbf16(const hip_bfloat16* __restrict__ W,
                               const hip_bfloat16* __restrict__ x,
                               hip_bfloat16* __restrict__ out, int rows, int rpb, int neg) {
    __shared__ hip_bfloat16 xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
    float acc = 0.f;
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
#pragma unroll
    for (int v = lane; v < words; v += ws) {
        uint4 u = wrow[v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        const hip_bfloat16* xp = &xs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += bf(p[j]) * bf(xp[j]);
    }
    acc = reduce_row(acc);
    if (lane == 0) out[neg == NEG_SWAPROW ? (row ^ 1) : row] = rnd(acc, neg);
}

}  // namespace engine::mlpxbf16
