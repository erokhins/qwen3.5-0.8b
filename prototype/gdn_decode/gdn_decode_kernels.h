/*
 * prototype/gdn_decode/gdn_decode_kernels.h — the device code of gdn_decode: S04.2-S04.4 GDN decode step: gates, causal depthwise conv, gated-delta recurrence.
 *
 * Moved verbatim out of gdn_decode.hip.cpp (host main() and the gate's file I/O stayed
 * there) so that gdn_decode and `engine/decode_step` compile THE SAME KERNEL SOURCE. An
 * engine TU includes exactly one of these op headers: the ops each spell their own
 * helpers (bf/fb/rnd/block_sum/reduce_row) and define colliding macros (D, H, HID,
 * EPS, SCALE, FULL_MASK), which is fine while they are in separate translation
 * units and is a compile error if they are merged into one.
 *
 * Gate for this file: prototype/gdn_decode/ (its run.py/check.py drives the binary in
 * the same directory, which includes this header). Nothing here may be edited for
 * the engine's convenience — see the rule in prototype/README.md.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>

#define D 128          // d_k = d_v = 128
#define NH 16          // value heads
#define CH 6144        // conv channels
#define K 4            // conv kernel
#define Q_SCALE 0.08838834764831845f   // 128^{-1/2}

__device__ __forceinline__ float bf16_round(float x) {
    return float(hip_bfloat16(x));
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    v += __shfl_down_sync(0xffffffffffffffffull, v, 16);
    v += __shfl_down_sync(0xffffffffffffffffull, v, 8);
    v += __shfl_down_sync(0xffffffffffffffffull, v, 4);
    v += __shfl_down_sync(0xffffffffffffffffull, v, 2);
    v += __shfl_down_sync(0xffffffffffffffffull, v, 1);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    // red must be at least blockDim.x/32 entries; reading past it (as a fixed red[8] with
    // 128 threads did) is undefined shared-memory access and showed up as NaNs in one mode
    // and silence in another.
    int warps = blockDim.x >> 5;
    float t = (threadIdx.x < warps) ? red[threadIdx.x] : 0.f;
    t += __shfl_down_sync(0xffffffffffffffffull, t, 16);
    t += __shfl_down_sync(0xffffffffffffffffull, t, 8);
    t += __shfl_down_sync(0xffffffffffffffffull, t, 4);
    t += __shfl_down_sync(0xffffffffffffffffull, t, 2);
    t += __shfl_down_sync(0xffffffffffffffffull, t, 1);
    __syncthreads();
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    float out = red[0];
    __syncthreads();
    return out;
}

/* ---- S04.3 gates ------------------------------------------------------- */
__global__ void gates_kernel(const float* a_raw, const float* b_raw, const float* A_log,
                             const float* dt_bias, float* g, float* beta) {
    int h = threadIdx.x;                       // 16 heads
    float a = a_raw[h] + dt_bias[h];
    float sp = (a > 20.f) ? a : log1pf(expf(a));
    g[h] = -expf(A_log[h]) * sp;
    float b = b_raw[h];
    beta[h] = 1.f / (1.f + expf(-b));
}

/* ---- S04.2 causal depthwise conv, one new token ------------------------ */
// win is [K-1][CH] (oldest row first), raw is [CH]; u is written bf16-rounded-then-silu,
// matching the reference: taps accumulate in F32, round once to bf16, then silu.
__global__ void conv_kernel(const float* __restrict__ raw, const float* __restrict__ win,
                            const float* __restrict__ C, float* __restrict__ u,
                            float* __restrict__ new_win) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        float tap = (j < K - 1) ? win[j * CH + c] : raw[c];
        acc += tap * C[c * K + j];
    }
    u[c] = bf16_round(bf16_round(acc) / (1.f + expf(-bf16_round(acc))));
    // roll: drop the oldest row, append the new pre-conv value. The window is K-1 rows, so
    // the newest slot is K-2 — writing (K-1) here would run one row past the buffer.
#pragma unroll
    for (int j = 0; j < K - 2; ++j) new_win[j * CH + c] = win[(j + 1) * CH + c];
    new_win[(K - 2) * CH + c] = raw[c];
}

/* ---- S04.4 the recurrence, one token, NH heads per block --------------- */
__global__ void rec_kernel(const float* __restrict__ q, const float* __restrict__ k,
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
