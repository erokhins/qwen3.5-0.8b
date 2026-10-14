/*
 * prototype/glue/glue_kernels.h — the device code of glue: S03.1 zero-centred RMSNorm fused with the S03.2 residual add.
 *
 * Moved verbatim out of glue.hip.cpp (host main() and the gate's file I/O stayed
 * there) so that glue and `engine/decode_step` compile THE SAME KERNEL SOURCE. An
 * engine TU includes exactly one of these op headers: the ops each spell their own
 * helpers (bf/fb/rnd/block_sum/reduce_row) and define colliding macros (D, H, HID,
 * EPS, SCALE, FULL_MASK), which is fine while they are in separate translation
 * units and is a compile error if they are merged into one.
 *
 * Gate for this file: prototype/glue/ (its run.py/check.py drives the binary in
 * the same directory, which includes this header). Nothing here may be edited for
 * the engine's convenience — see the rule in prototype/README.md.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>

#define FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit
#define H 1024                            // §00: the hidden size every §03.1 norm runs over
#define EPS 1e-6f                         // §03.1
#define PROBE_GBS 237.0                   // devices/strix-halo-memory-bandwidth.md read probe

enum Neg {
    NEG_NONE = 0,
    NEG_PLAINW,      // (1 + w) -> w
    NEG_NORMDELTA,   // norm the addend instead of the updated residual
    NEG_NOADD,       // skip s += delta
    NEG_MEAN,        // mean instead of mean-of-squares
    NEG_BF16ACC,     // squared-sum accumulator lives in BF16
    NEG_BF16SUB,     // one BF16 subpartial, F32 ladder
};

__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }

// "Store this into a BF16 slot": the RNE round-trip the reference does when it writes the
// residual back. Also what a BF16 accumulator does to a running sum, which is why the two
// BF16 negatives are written in terms of it.
__device__ __forceinline__ float bf_round(float f) { return float(hip_bfloat16(f)); }

// One-block reduction for a 1024-element row: intra-warp shuffle ladder, then warp partials
// through shared memory. `bf16_tree` is NEG_BF16ACC — the same ladder with the running value
// put back through a BF16 slot at every step.
__device__ __forceinline__ float block_reduce(float v, float* red, bool bf16_tree) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) {
        const float n = __shfl_down_sync(FULL_MASK, v, o);
        v = bf16_tree ? bf_round(v + n) : v + n;
    }
    if ((threadIdx.x & (ws - 1)) == 0) red[threadIdx.x / ws] = v;
    __syncthreads();
    // Second ladder runs in warp 0 with every lane of that warp live, which is what makes the
    // full participating-lane mask correct here rather than a lie.
    if (threadIdx.x < (unsigned)ws) {
        float x = threadIdx.x < blockDim.x / ws ? red[threadIdx.x] : 0.f;
        for (int o = ws / 2; o; o >>= 1) {
            const float n = __shfl_down_sync(FULL_MASK, x, o);
            x = bf16_tree ? bf_round(x + n) : x + n;
        }
        if (threadIdx.x == 0) red[0] = x;
    }
    __syncthreads();
    return red[0];
}

/*
 * The fused op. One BLOCK PER ROW, whole row resident in registers (`v[]`), so `s` and `delta`
 * are each read once and the norm reads the values it already has instead of going back to
 * memory.
 *
 * `blockIdx.x` IS the row (P4, progress/2026-10-12/plan-prefill-2026-10-12.md). The §03.1 reduction is
 * row-local — one block, one 1024-element row, nothing crosses a row boundary — so a
 * `[rows, H]` tile in one launch has to produce the same bits as `rows` launches of one row.
 * At `gridDim.x == 1` every pointer this kernel forms is the pointer it formed before the row
 * index existed, which is why the shipped rows=1 gate stays green by construction rather than by
 * argument; `prototype/glue/run.py --batched` checks the other half on device (one multi-row
 * launch vs one launch per row, bit for bit, against the prefill pins).
 *
 * The norm weight `w` is deliberately NOT row-indexed: one norm vector serves every row.
 */
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
glue_add_norm(const hip_bfloat16* __restrict__ s, const hip_bfloat16* __restrict__ delta,
              const hip_bfloat16* __restrict__ w, hip_bfloat16* __restrict__ s_out,
              hip_bfloat16* __restrict__ n_out, int neg) {
    __shared__ float red[32];   // >= THREADS/warpSize warp partials (1024/32 is the most)
    constexpr int PER = H / THREADS;
    const size_t row = (size_t)blockIdx.x * H;
    s += row;
    if (delta) delta += row;
    if (s_out) s_out += row;
    n_out += row;
    const bool norm_delta = (neg == NEG_NORMDELTA) && delta != nullptr;
    const bool bf16_acc = (neg == NEG_BF16ACC);

    float v[PER];
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;   // consecutive threads, consecutive lanes
        const float sv = bf(s[idx]);
        float m = sv;
        if (delta && neg != NEG_NOADD) {
            m = bf_round(sv + bf(delta[idx]));       // §03.2: the residual add, BF16
            if (s_out) s_out[idx] = hip_bfloat16(m);
        } else if (s_out) {
            s_out[idx] = s[idx];                     // norm-only: the residual is unchanged
        }
        v[i] = m;
        float t = norm_delta ? bf(delta[idx]) : m;
        if (neg != NEG_MEAN) t *= t;                 // §03.1: mean of SQUARES
        acc = bf16_acc ? bf_round(acc + t) : acc + t;
    }
    if (neg == NEG_BF16SUB) acc = bf_round(acc);
    const float sum = block_reduce(acc, red, bf16_acc);
    const float r = rsqrtf(sum * (1.0f / H) + EPS);

#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        const float wv = bf(w[idx]);
        const float aff = (neg == NEG_PLAINW) ? wv : (1.0f + wv);
        n_out[idx] = hip_bfloat16((v[i] * r) * aff);   // §03.1 order, one cast back
    }
}

/*
 * The unfused pair, for measuring what fusion is worth rather than asserting it: `residual_add`
 * writes the new residual, then `norm_only` reads it back from memory and normalises it. Same
 * arithmetic, twice the launches, one extra round trip through DRAM/L2 per row.
 */
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
residual_add(const hip_bfloat16* __restrict__ s, const hip_bfloat16* __restrict__ delta,
             hip_bfloat16* __restrict__ s_out) {
    constexpr int PER = H / THREADS;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        s_out[idx] = hip_bfloat16(bf_round(bf(s[idx]) + bf(delta[idx])));
    }
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
norm_only(const hip_bfloat16* __restrict__ s, const hip_bfloat16* __restrict__ w,
          hip_bfloat16* __restrict__ n_out) {
    __shared__ float red[32];
    constexpr int PER = H / THREADS;
    float v[PER];
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        v[i] = bf(s[idx]);
        acc += v[i] * v[i];
    }
    const float r = rsqrtf(block_reduce(acc, red, false) * (1.0f / H) + EPS);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        n_out[idx] = hip_bfloat16((v[i] * r) * (1.0f + bf(w[idx])));
    }
}
