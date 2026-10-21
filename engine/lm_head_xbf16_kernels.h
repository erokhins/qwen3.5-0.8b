/*
 * engine/lm_head_xbf16_kernels.h — the tied logits GEMV with the final norm row staged as BF16, so the
 * last widening launch of a token can be deleted (route S1 item 1, the `cast` family).
 *
 * Same construction and the same reason as `engine/mlp_xbf16_kernels.h`: this INCLUDES the gated op
 * header rather than editing it (`prototype/README.md`), borrows its `HID` and `FULL_MASK`, and adds a
 * kernel whose one difference is the dtype of the staged row. `float(hip_bfloat16)` is exact, so the
 * staged value at every index is the same float the deleted `kernels_cast.cpp` launch used to write,
 * the products and the 16/8/4/2/1 ladder are unchanged, and the logits are the same bits. See that
 * header for the full numerics argument; it is the same argument one row further down the step.
 *
 * WHY THIS ONE LAUNCH IS WORTH NAMING SEPARATELY. It is 1 of the 97, and its consumer is the single
 * family `engine/generation-limits.md` §1/§2 measures at the machine's read roof (1 launch, 509 MiB,
 * 104 % of the plateau, "cannot be faster"). Folding here buys a launch and halves the staged row from
 * 4 KiB to 2 KiB, and costs one conversion per element in the innermost loop of the one kernel with no
 * headroom to give — so the A/B must be read knowing that if the step does not move, this row is the
 * first suspect and `lm_head` itself is the least likely place for the prize.
 *
 * NOT the fused-argmax form, for the standing reason in `kernels_lmhead.cpp`: §03.4 fixes ties to the
 * LOWEST id and the fused kernel resolves them to the highest.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

#include "prototype/lm_head_gemv/lm_head_gemv_kernels.h"   // HID, FULL_MASK, and the header guard

namespace engine::lmheadxbf16 {

template <bool XBF16>
__global__ void lm_head_gemv_full_xbf16(const hip_bfloat16* __restrict__ W,
                                        const void* __restrict__ h, float* __restrict__ out,
                                        int V, int rpb) {
    using Hstaged = typename std::conditional<XBF16, hip_bfloat16, float>::type;
    __shared__ Hstaged hs[HID];
    for (int i = threadIdx.x; i < HID; i += blockDim.x) {
        hs[i] = XBF16 ? ((const hip_bfloat16*)h)[i] : ((const float*)h)[i];
    }
    __syncthreads();

    int row = blockIdx.x * rpb + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= V) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * HID);
    float acc = 0.f;
#pragma unroll
    for (int v = lane; v < HID / 8; v += 32) {          // 128 uint4 per row, 4 per lane
        uint4 u = wrow[v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        const Hstaged* hp = &hs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += float(p[j]) * float(hp[j]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    if (lane == 0) out[row] = acc;
}

}  // namespace engine::lmheadxbf16
