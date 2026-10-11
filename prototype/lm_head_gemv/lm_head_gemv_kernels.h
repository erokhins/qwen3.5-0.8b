/*
 * prototype/lm_head_gemv/lm_head_gemv_kernels.h — the device code of lm_head_gemv: Tied logits GEMV (S03.4), full-logit and fused-argmax forms.
 *
 * Moved verbatim out of lm_head_gemv.hip.cpp (host main() and the gate's file I/O stayed
 * there) so that lm_head_gemv and `engine/decode_step` compile THE SAME KERNEL SOURCE. An
 * engine TU includes exactly one of these op headers: the ops each spell their own
 * helpers (bf/fb/rnd/block_sum/reduce_row) and define colliding macros (D, H, HID,
 * EPS, SCALE, FULL_MASK), which is fine while they are in separate translation
 * units and is a compile error if they are merged into one.
 *
 * Gate for this file: prototype/lm_head_gemv/ (its run.py/check.py drives the binary in
 * the same directory, which includes this header). Nothing here may be edited for
 * the engine's convenience — see the rule in prototype/README.md.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstring>

#define HID 1024          // model hidden size (§00)
#define ROWS_PER_BLOCK 8  // default: one warp per row, 8 rows per block

// Sign-preserving monotonic map: order-preserving uint32 for IEEE754 floats.
__device__ __forceinline__ unsigned fkey(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float unfkey(unsigned u) {
    unsigned v = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
    float f;
    memcpy(&f, &v, 4);
    return f;
}

// HIP takes the shuffle mask as a 64-bit value (wave32 and wave64 both supported).
#define FULL_MASK 0xffffffffffffffffull

__global__ void lm_head_gemv_full(const hip_bfloat16* __restrict__ W,
                                  const float* __restrict__ h, float* __restrict__ out,
                                  int V, int rpb) {
    __shared__ float hs[HID];
    for (int i = threadIdx.x; i < HID; i += blockDim.x) hs[i] = h[i];
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
        const float* hp = &hs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += float(p[j]) * hp[j];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    if (lane == 0) out[row] = acc;
}

__global__ void lm_head_gemv_fused(const hip_bfloat16* __restrict__ W,
                                   const float* __restrict__ h,
                                   unsigned long long* __restrict__ best, int V, int rpb) {
    __shared__ float hs[HID];
    for (int i = threadIdx.x; i < HID; i += blockDim.x) hs[i] = h[i];
    __syncthreads();

    int row = blockIdx.x * rpb + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= V) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * HID);
    float acc = 0.f;
#pragma unroll
    for (int v = lane; v < HID / 8; v += 32) {
        uint4 u = wrow[v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        const float* hp = &hs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += float(p[j]) * hp[j];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    // 248320 < 2^18, so 20 bits of index fit under 44 bits of float key: one atomic per
    // row and no logits array at all.
    if (lane == 0) {
        unsigned long long key = ((unsigned long long)fkey(acc) << 20) | (unsigned)row;
        atomicMax(best, key);
    }
}
