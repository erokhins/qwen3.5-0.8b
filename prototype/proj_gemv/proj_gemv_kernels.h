/*
 * prototype/proj_gemv/proj_gemv_kernels.h — the device code of proj_gemv: One templated BF16 GEMV for the nine projections of S04.1/S05.1 (plus out_proj/o_proj).
 *
 * Moved verbatim out of proj_gemv.hip.cpp (host main() and the gate's file I/O stayed
 * there) so that proj_gemv and `engine/decode_step` compile THE SAME KERNEL SOURCE. An
 * engine TU includes exactly one of these op headers: the ops each spell their own
 * helpers (bf/fb/rnd/block_sum/reduce_row) and define colliding macros (D, H, HID,
 * EPS, SCALE, FULL_MASK), which is fine while they are in separate translation
 * units and is a compile error if they are merged into one.
 *
 * Gate for this file: prototype/proj_gemv/ (its run.py/check.py drives the binary in
 * the same directory, which includes this header). Nothing here may be edited for
 * the engine's convenience — see the rule in prototype/README.md.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <type_traits>

#define FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit
#define PROBE_GBS 237.0                   // devices/strix-halo-memory-bandwidth.md read probe

enum Neg {
    NEG_NONE = 0,
    NEG_TAIL,        // one word (8 bf16) of K dropped: the subtle reduction-length bug
    NEG_HALF,        // half of K accumulated
    NEG_TRUNC,       // truncate instead of round-to-nearest-even
    NEG_ROWSWAP,     // write out[row] to the neighbouring row
    NEG_TRANSPOSE,   // read W as if it were stored [in, rows]
    NEG_QHALF,       // q_proj rows in the wrong interleave (§05.1)
    NEG_KVS,         // in_proj_qkv k/v sections swapped (§04.1)
};

__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
// The staged x is either BF16 (packed) or F32 (widened); both convert to F32 the same way.
template <typename T> __device__ __forceinline__ float tof(T v) { return float(v); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }

// Truncating cast, NEG_TRUNC only: dropping the low 16 bits of the fp32 payload rounds toward
// zero, the classic way to get a BF16 conversion wrong.
__device__ __forceinline__ hip_bfloat16 fb_trunc(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    unsigned short top = (unsigned short)(u >> 16);
    hip_bfloat16 b;
    memcpy(&b, &top, 2);
    return b;
}

__device__ __forceinline__ hip_bfloat16 rnd(float f, int neg) {
    return neg == NEG_TRUNC ? fb_trunc(f) : fb(f);
}

// Full-wavefront reduction: the shift ladder starts at warpSize/2, so one body is correct on a
// wave32 (this part) and a wave64 target.
__device__ __forceinline__ float reduce_row(float acc) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    return acc;
}

// §05.1: q_proj's 4096 rows are [8 heads][q 256 | gate 256]. §10.5 measured that reading them as
// [all q 2048 | all gate 2048] scores cos -0.14 against the pinned q_proj output, so the layout
// cannot be gotten quietly wrong. NEG_QHALF *is* that reading: every dot is still computed
// correctly, its result is just written where the wrong reading expects it — the same set of
// wrong values an engine that de-interleaves wrongly would hand to §05.2.
__device__ __forceinline__ int qhalf_perm(int row) {
    const int H = 8, DH = 256;
    if (row >= 2 * H * DH) return row;              // past the q_proj region: k/v sections stay
    const int h = row / (2 * DH), j = row % (2 * DH);
    return j < DH ? h * DH + j : H * DH + h * DH + (j - DH);
}

// §04.1: raw = [q 2048 | k 2048 | v 2048]. NEG_KVS swaps the k and v sections of the output.
// Rows past 6144 — the z/b/a sections of a packed in_proj launch — are untouched, which is what
// makes this negative's must-not-catch claim testable instead of decorative.
__device__ __forceinline__ int kv_swap_perm(int row) {
    const int C = 2048;
    if (row >= 3 * C) return row;
    if (row < C) return row;
    return row < 2 * C ? row + C : row - C;
}

// Where this row's result goes. `rows` clamps the neighbour swap so an odd row count cannot
// write past the end of the output buffer.
__device__ __forceinline__ int perm_out_row(int row, int rows, int neg) {
    if (neg == NEG_ROWSWAP) return (row % 2 == 0) ? (row + 1 < rows ? row + 1 : row) : row - 1;
    if (neg == NEG_QHALF) return qhalf_perm(row);
    if (neg == NEG_KVS) return kv_swap_perm(row);
    return row;
}

// One wavefront per output row. PACK_X keeps the staged input as BF16 (2 KiB at in=1024) instead
// of unpacking it to F32 (4 KiB): identical products — every BF16 value is exact in F32 — half
// the shared-memory traffic and half the shared memory per block, which is what limits blocks/CU
// for the 2048-wide shapes. run.py's sweep measures both instead of guessing.
template <int IN, bool PACK_X>
__global__ void proj_gemv(const hip_bfloat16* __restrict__ W, const void* __restrict__ x,
                          hip_bfloat16* __restrict__ out, int rows, int rpb, int neg) {
    __shared__ typename std::conditional<PACK_X, hip_bfloat16, float>::type xs[IN];
    if (PACK_X) {
        const hip_bfloat16* src = (const hip_bfloat16*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = src[i];
    } else {
        const float* src = (const float*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = src[i];
    }
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    float acc = 0.f;
    if (neg == NEG_TRANSPOSE) {
        // Read the same bytes as [in, rows]: in-bounds for any shape (k*rows + row < in*rows)
        // and a completely different set of products.
        for (int k = lane; k < IN; k += ws) acc += bf(W[(size_t)k * rows + row]) * tof(xs[k]);
    } else {
        const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
        int words = IN / 8;
        if (neg == NEG_TAIL) words -= 1;
        else if (neg == NEG_HALF) words /= 2;
#pragma unroll
        for (int v = lane; v < words; v += ws) {
            uint4 u = wrow[v];
            const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += bf(p[j]) * tof(xs[v * 8 + j]);
        }
    }
    acc = reduce_row(acc);
    if (lane == 0) out[perm_out_row(row, rows, neg)] = rnd(acc, neg);
}
