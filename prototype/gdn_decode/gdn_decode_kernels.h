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

/* ---- S04.3 gates, T tokens wide ---------------------------------------- */
// `a_raw`/`b_raw`/`g`/`beta` are [tokens][NH]; `blockIdx.x` is the token and `threadIdx.x` the
// value head, so tokens == 1 is the launch this file shipped with (<<<1, NH>>>) and every element
// expression is unchanged. Per-(token, head) work with no reduction: batching the gates cannot
// move a bit, which is what `check.py --tokens` requires rather than assumes.
enum GdnGatesNeg {
    GG_NONE = 0,
    GG_ROW_ZERO,       // every token reads row 0's a_raw/b_raw: a batched kernel that forgot `t`
};

__global__ void gates_kernel(const float* a_raw, const float* b_raw, const float* A_log,
                             const float* dt_bias, float* g, float* beta, int tokens, int neg) {
    const int h = threadIdx.x;                 // 16 heads
    const int t = (neg == GG_ROW_ZERO) ? 0 : blockIdx.x;
    float a = a_raw[(size_t)t * NH + h] + dt_bias[h];
    float sp = (a > 20.f) ? a : log1pf(expf(a));
    g[(size_t)t * NH + h] = -expf(A_log[h]) * sp;
    float b = b_raw[(size_t)t * NH + h];
    beta[(size_t)t * NH + h] = 1.f / (1.f + expf(-b));
}

/* ---- S04.2 causal depthwise conv over a [tokens][CH] tile -------------- */
// ONE body at two widths, the way `prototype/proj_gemv` ships its GEMV: `conv_kernel` at
// tokens = 1 is the launch this file shipped with (CH/256 blocks of 256 threads, same tap order,
// same F32 accumulation, same round-once-then-silu) and a wave is that launch replicated on
// gridDim.y. Nothing about a wider tile changes a per-element reduction, so the two widths owe
// each other bit-for-bit and `check.py --tokens` demands it.
//
// The window is the reason this is not a pure elementwise batch. `win` is [K-1][CH], oldest row
// first, and holds the pre-conv `raw` of the three positions BEFORE this tile (§04.2's
// "incremental state", §08.2, A-04). The tap rule is stated once, in `conv_tap`, for both the
// tile's own tokens and the end condition:
//
//     raw[k] with k < 0  ==  win[k + K - 1]      (the carried window)
//
// At request start the carried window is the zeros §06.1 installs, so "carry" and "zero-pad" are
// the same sentence there and §06.2 lets the engine pick either — but at a WAVE interior, or after
// any decode step, they are not, and a kernel that zero-pads every tile edge is correct on every
// single-wave fixture and wrong everywhere else. That is `GC_ZERO_PAD` below, and why the gate
// feeds a non-zero window (check.py's split-wave case: the pins give a whole prompt's `raw`, so a
// wave can start at any token of it).
//
// The end condition is the other half of §08.6 inv 2 and lives in the same rule: after the tile,
// the window is raw[T-K+1 .. T-1] with the same carry rule, which at tokens = 1 is exactly the
// roll this file used to spell out row by row.
//
// INJECTED BUGS (each moves ONE output, and the must-not-move side is the localisation):
//   tap_stride   the tile is read channel-major ([CH][tokens]) — the wrong K columns, in bounds
//   win_lag      the window is written from raw[T-K .. T-2] instead of raw[T-K+1 .. T-1]
//   win_from_u   the window is written from the POST-conv values (A-04's exact trap: it stores
//                pre-conv `raw`)
//   zero_pad     raw[k<0] is zero instead of the carried window — invisible at request start,
//                which is what makes the gate's non-zero-window case the one that matters
// `win_lag` and `win_from_u` cannot be caught by ANY downstream output of this launch: `u` is
// already computed when the window is written, so the only thing they move is the window itself.
// A conv gate that compares only `u` would pass both.
enum GdnConvNeg {
    GC_NONE = 0,
    GC_TAP_STRIDE,
    GC_WIN_LAG,
    GC_WIN_FROM_U,
    GC_ZERO_PAD,
};

// The one tap rule. `k` is the token offset from this tile's first token; k < 0 is the carried
// window. `zero_pad` is the injected reading that is only right at request start.
__device__ __forceinline__ float conv_tap(const float* __restrict__ raw,
                                          const float* __restrict__ win, int c, int k,
                                          bool zero_pad) {
    if (k < 0) return zero_pad ? 0.f : win[(k + K - 1) * CH + c];
    return raw[(size_t)k * CH + c];
}

// The conv's value at offset k of this tile: taps in F32, one BF16 round, then SiLU (§04.2). The
// tile body and the `GC_WIN_FROM_U` injection both call it, so the injected bug cannot depend on
// another block's already-written output (there is no grid-wide sync to read `u` from). `k` must
// be >= 0, because a tap at k < 0 is a carried window row, not a tile row, and the conv of a
// position before this wave is not this launch's to compute.
// `channel_major` is the `GC_TAP_STRIDE` reading: the same [tokens][CH] buffer indexed as if it
// were still HF's [CH][tokens]. In bounds for every (c, src) this kernel can ask for —
// c*tokens + src <= (CH-1)*tokens + tokens-1 = CH*tokens-1 — and wrong for every token but 0.
__device__ __forceinline__ float conv_value(const float* __restrict__ raw,
                                            const float* __restrict__ win,
                                            const float* __restrict__ C, int c, int k,
                                            int tokens, bool zero_pad, bool channel_major) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const int src = k - (K - 1) + j;
        const float tap = (src < 0) ? conv_tap(raw, win, c, src, zero_pad)
                                    : (channel_major ? raw[(size_t)c * tokens + src]
                                                     : raw[(size_t)src * CH + c]);
        acc += tap * C[c * K + j];
    }
    const float rounded = bf16_round(acc);
    return bf16_round(rounded / (1.f + expf(-rounded)));
}

// win is [K-1][CH] (oldest row first); u is [tokens][CH] and is written bf16-rounded-then-silu.
__global__ void conv_kernel(const float* __restrict__ raw, const float* __restrict__ win,
                            const float* __restrict__ C, float* __restrict__ u,
                            float* __restrict__ new_win, int tokens, int neg) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    const int t = blockIdx.y;
    u[(size_t)t * CH + c] = conv_value(raw, win, C, c, t, tokens, neg == GC_ZERO_PAD,
                                       neg == GC_TAP_STRIDE);
    // The end condition: the window after this tile is its last K-1 raw columns. Every thread owns
    // column `c` on both sides of the write, so this is safe against its own reads whatever the
    // relationship between `win` and `new_win` is — the engine still rolls through a scratch row
    // (`decode_step.cpp` keeps `conv_new`), which is what lets both pointers stay __restrict__.
#pragma unroll
    for (int j = 0; j < K - 1; ++j) {
        const int k = tokens - (K - 1) + j + ((neg == GC_WIN_LAG) ? -1 : 0);
        float v = conv_tap(raw, win, c, k, false);
        // GC_WIN_FROM_U: A-04's exact trap, the window written from the POST-conv value. Only
        // positions inside this tile have one; a carried row is already pre-conv by construction.
        if (neg == GC_WIN_FROM_U && k >= 0) v = conv_value(raw, win, C, c, k, tokens, false, false);
        new_win[j * CH + c] = v;
    }
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
