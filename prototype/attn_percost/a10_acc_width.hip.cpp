// a10_acc_width.hip.cpp — one question, compile-only: does an f16-accumulating D block cost half the
// registers of an f32 one, when what has to be covered is fixed at 16 rows x 256 dims?
//
// WHY THIS FILE EXISTS. `A10-RF-LADDER.md` §2 says yes (f32 accumulator = 128 VGPR/lane, f16 = 64), and that
// is what makes the f16-atom branch fit at ~244 of 256. `prototype/issue_rate_probe/wmma_latency.cpp:24-33`
// says the opposite in its header: "The F16 accumulator does **not** save registers", citing rocwmma
// declaring `CRegsT = AccRegF32x8` for `f16 x f16 -> f16` and its own legs reading near-equal VGPR at equal
// ILP. Both readings can be true at once — the probe's legs hold the same *tile count* in each width, so
// equal per-tile cost says nothing about how much output a tile covers — but only one of them decides whether
// R1+R2+R3 is buildable at all. Nothing in this tree has ever emitted the f16-accumulator form in a kernel
// (`wmma_latency.cpp`'s own header says so), so this is the first compile that asks.
//
// The discriminator has to be a shape clang cannot fold AND that stays under the ceiling, because the §3
// variants pinned at 256 + spill and so could not discriminate (skeletons are not bodies, A10-RF-LADDER.md
// §3). So: hold Q resident (16 operand tuples = 128 VGPR), roll a KV loop whose accumulator is held across
// the loop by construction, and let the two widths differ ONLY in the accumulator's element type and tile
// count. Prediction if A10-RF-LADDER.md §2 is right: `acc_f32` = 128 Q + 128 acc + temps -> over the ceiling,
// spills; `acc_f16` = 128 Q + 64 acc + temps -> ~210, no spill. Prediction if wmma_latency.cpp is right: both
// pin at 256 with similar scratch, and the f16 branch is a source kill too.
//
// Nothing here is launched. Build:
//   hipcc -O3 --offload-arch=gfx1151 --save-temps -c prototype/attn_percost/a10_acc_width.hip.cpp -o /tmp/k.o
// then read the note blocks with prototype/attn_wmma/fattn_residency_read.py --asm <listing> --all.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using A10H = short __attribute__((ext_vector_type(16)));  // one 16x16 F16 tuple = 8 words = 8 VGPR
using A10F = float __attribute__((ext_vector_type(8)));   // one f32 C tile       = 8 words = 8 VGPR

#define A10_DH     256                  // head_dim
#define A10_ROWS   16                   // query rows per wave: forced by the atom
#define A10_DTILES (A10_DH / 16)        // 16 tuples of 16 dims: the Q set, and the f32 accumulator set
#define A10_HTILES (A10_DH / 32)        // 8 tuples of 32 dims: the f16 accumulator set (llama's shape)
#define A10_THREADS 256

// 16 halves per lane = 8 words; loaded as words so the register shape is the real one (a10_rf_ladder.hip.cpp
// uses the same helper shape on purpose).
__device__ __forceinline__ A10H a10w_frag(const unsigned *base, int lane, int word_step) {
    A10H r;
    unsigned *w = (unsigned *)&r;
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = __ldg(base + (size_t)(i * 32 + lane) + word_step);
    return r;
}

__device__ __forceinline__ unsigned a10w_sink(const void *frag, int words) {
    const unsigned *w = (const unsigned *)frag;
    unsigned s = 0;
#pragma unroll 8
    for (int i = 0; i < words; ++i) s += w[i];
    return s;
}

// ---- f32 accumulator: A10_DTILES tiles x 8 f32 = 128 VGPR for 16 rows x 256 dims --------------------
extern "C" __global__ void __launch_bounds__(A10_THREADS) acc_f32(const unsigned *src, unsigned *out,
                                                                   int rounds) {
    A10H q[A10_DTILES];
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < A10_DTILES; ++k) q[k] = a10w_frag(src, lane, k * 32);
    A10F acc[A10_DTILES] = {};                  // brace-init required: v_wmma takes C as a live IN
    const A10H p = q[0];                        // stand-in P operand: footprint, not value
#pragma unroll 1
    for (int r = 0; r < rounds; ++r) {
#pragma unroll
        for (int k = 0; k < A10_DTILES; ++k) {
            const A10H v = a10w_frag(src + 8192 + (size_t)r * 512, lane, k * 32);
            acc[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v, q[k], acc[k]);
        }
    }
    unsigned s = 0;
#pragma unroll
    for (int k = 0; k < A10_DTILES; ++k) s += a10w_sink(&acc[k], 8);
    if (s == 0x1234u) out[threadIdx.x] = s;     // sink: keeps every tile live to the end
}

// ---- f16 accumulator: A10_HTILES tiles, each covering 32 dims, x 8 words = 64 VGPR for the same 16x256
// llama's RDNA form: two issues with opsel 0 then 1 fill the two halves of one D block
// (`engines/llama.cpp/ggml/src/ggml-cuda/mma.cuh:1556-1566`).
extern "C" __global__ void __launch_bounds__(A10_THREADS) acc_f16(const unsigned *src, unsigned *out,
                                                                  int rounds) {
    A10H q[A10_DTILES];
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < A10_DTILES; ++k) q[k] = a10w_frag(src, lane, k * 32);
    A10H acc[A10_HTILES] = {};
    const A10H p = q[0];
#pragma unroll 1
    for (int r = 0; r < rounds; ++r) {
#pragma unroll
        for (int k = 0; k < A10_HTILES; ++k) {
            const A10H v0 = a10w_frag(src + 8192 + (size_t)r * 512, lane, k * 64);
            const A10H v1 = a10w_frag(src + 8192 + (size_t)r * 512, lane, k * 64 + 32);
            acc[k] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(v0, q[2 * k], acc[k], 0);
            acc[k] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(v1, q[2 * k + 1], acc[k], 1);
        }
    }
    unsigned s = 0;
#pragma unroll
    for (int k = 0; k < A10_HTILES; ++k) s += a10w_sink(&acc[k], 8);
    if (s == 0x1234u) out[threadIdx.x] = s;
}

// ---- control: f16 accumulator WITHOUT resident Q, so a gap between the two rows above is attributable
// to the accumulator and not to Q. Q is 128 VGPR; drop it and both widths should fit.
extern "C" __global__ void __launch_bounds__(A10_THREADS) acc_f16_noq(const unsigned *src, unsigned *out,
                                                                      int rounds) {
    A10H acc[A10_HTILES] = {};
    const int lane = threadIdx.x & 31;
#pragma unroll 1
    for (int r = 0; r < rounds; ++r) {
#pragma unroll
        for (int k = 0; k < A10_HTILES; ++k) {
            const A10H v0 = a10w_frag(src + (size_t)r * 512, lane, k * 64);
            const A10H v1 = a10w_frag(src + (size_t)r * 512, lane, k * 64 + 32);
            const A10H q0 = a10w_frag(src + 4096 + (size_t)r * 512, lane, k * 64);
            const A10H q1 = a10w_frag(src + 4096 + (size_t)r * 512, lane, k * 64 + 32);
            acc[k] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(v0, q0, acc[k], 0);
            acc[k] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(v1, q1, acc[k], 1);
        }
    }
    unsigned s = 0;
#pragma unroll
    for (int k = 0; k < A10_HTILES; ++k) s += a10w_sink(&acc[k], 8);
    if (s == 0x1234u) out[threadIdx.x] = s;
}

// ---- control: f32 accumulator without resident Q, the mirror of acc_f16_noq. If the widths really cost
// the same per tile, acc_f32_noq and acc_f16_noq differ by the 64 VGPR of accumulator this file is about.
extern "C" __global__ void __launch_bounds__(A10_THREADS) acc_f32_noq(const unsigned *src, unsigned *out,
                                                                      int rounds) {
    A10F acc[A10_DTILES] = {};
    const int lane = threadIdx.x & 31;
#pragma unroll 1
    for (int r = 0; r < rounds; ++r) {
#pragma unroll
        for (int k = 0; k < A10_DTILES; ++k) {
            const A10H v  = a10w_frag(src + (size_t)r * 512, lane, k * 32);
            const A10H qk = a10w_frag(src + 4096 + (size_t)r * 512, lane, k * 32);
            acc[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v, qk, acc[k]);
        }
    }
    unsigned s = 0;
#pragma unroll
    for (int k = 0; k < A10_DTILES; ++k) s += a10w_sink(&acc[k], 8);
    if (s == 0x1234u) out[threadIdx.x] = s;
}
