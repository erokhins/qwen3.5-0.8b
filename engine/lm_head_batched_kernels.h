/*
 * engine/lm_head_batched_kernels.h — the tied logits GEMV serving N query rows from ONE read of the
 * 485 MiB embedding table (§03.4; the batched verify body of P7, `prototype/mtp_headbatch`).
 *
 * WHY THIS EXISTS AT ALL. `prototype/mtp_delta/D3-RESULT.md` measured the whole `delta` question as a
 * fork in the road: a 2-row verify body whose two rows share ONE pass over the tied head costs 0.1505 of
 * a step and is a win, and the same body with the head run twice costs 0.4729 and is KILLED. The table is
 * 485 MiB and the head is already at 104 % of the read roof (`engine/generation-limits.md` §1), so a
 * second pass is not a small line — it is the whole route. `prototype/mtp_headbatch` then MEASURED the
 * batched shape: M=2 at ratio 1.0019 of M=1, with `fetch_per_weight = 1.000000` on every M arm (the table
 * is really read once for every row served) and cross-M bit identity at 0 differing elements
 * (`HEADBATCH-RESULT.md`). This header is that shape, in the engine, over the engine's own BF16 norm tile.
 *
 * BIT-IDENTITY, and what is claimed versus what is witnessed. Per output row, the trip order over the
 * weight row, the 16/8/4/2/1 butterfly, the tie-free `lane == 0` store and the per-element
 * `float(hip_bfloat16)` widening are the SAME expressions as `engine/lm_head_xbf16_kernels.h`'s, and each
 * row's accumulator is its own independent FP chain — nothing is shared between rows except the uint4 that
 * the block reads once and both rows multiply by. `--fmad` contraction applies to each chain the same
 * way it does in the one-row kernel, and LLVM cannot reassociate a float add chain without fast-math.
 * That is an ARGUMENT. The witness is a byte compare, and it is a device one: `prototype/mtp_verify`
 * compares row 0 of this kernel against the shipped 1-row head's logits for the same position
 * (`AGENTS.md`: "a CPU mirror of the same recurrence could not see FMA contraction or association" — so
 * the claim is settled by the device run, not by this paragraph).
 *
 * NROWS IS A TEMPLATE PARAMETER AND ONLY ONE VALUE IS INSTANTIATED. The launcher refuses anything but 2:
 * D3's funded shape is k=1, and `prototype/mtp_draft/DRAFT-RESULT.md` closes with the pre-registered
 * instruction "fund a staging pass, do not fund deeper k" (block depth is linear: 1.754× at k=2). A
 * second instantiation would be an unbuilt route sitting next to the funded one, and the head is exactly
 * the family where deeper k looked free (M=4 costs 1.3 %) — which is why it needs a measurement request,
 * not a template argument.
 *
 * SHARED MEMORY: NROWS × HID BF16 = 4 KiB per block at NROWS=2, twice the one-row kernel's 2 KiB. On
 * this part LDS is the binding occupancy constraint for some kernels (`AGENTS.md`'s waves/SIMD row), so
 * the doubling is stated rather than glossed; `HEADBATCH-RESULT.md` measured the consequence at +0.19 %
 * for M=2 over the rpb sweep, which is the number this shape buys with.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "prototype/lm_head_gemv/lm_head_gemv_kernels.h"   // HID, FULL_MASK, and the header guard

namespace engine::lmheadbatched {

// `h` points at `NROWS` contiguous rows of the final-norm tile (row stride HID, which is exactly the
// stride of `DecodeScratch::norm`), and `out` at `NROWS` contiguous logits rows (row stride V, which is
// what `mtp_verify.cpp` allocates and what one D2H of both rows downloads).
template <int NROWS>
__global__ void lm_head_gemv_batched(const hip_bfloat16* __restrict__ W,
                                     const hip_bfloat16* __restrict__ h,
                                     float* __restrict__ out, int V, int rpb) {
    __shared__ hip_bfloat16 hs[NROWS * HID];
    for (int i = threadIdx.x; i < NROWS * HID; i += blockDim.x) hs[i] = h[i];
    __syncthreads();

    const int row = blockIdx.x * rpb + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= V) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * HID);
    float acc[NROWS];
#pragma unroll
    for (int q = 0; q < NROWS; ++q) acc[q] = 0.f;
#pragma unroll
    for (int v = lane; v < HID / 8; v += 32) {          // 128 uint4 per row, 4 per lane — as shipped
        uint4 u = wrow[v];                              // ONE fetch, both rows' dots
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
        for (int q = 0; q < NROWS; ++q) {
            const hip_bfloat16* hp = &hs[q * HID + v * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[q] += float(p[j]) * float(hp[j]);
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
#pragma unroll
        for (int q = 0; q < NROWS; ++q) acc[q] += __shfl_down_sync(FULL_MASK, acc[q], o);
    }
    if (lane == 0) {
#pragma unroll
        for (int q = 0; q < NROWS; ++q) out[(size_t)q * V + row] = acc[q];
    }
}

}  // namespace engine::lmheadbatched
