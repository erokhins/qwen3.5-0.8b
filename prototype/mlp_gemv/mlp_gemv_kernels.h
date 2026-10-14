/*
 * prototype/mlp_gemv/mlp_gemv_kernels.h — the device code of mlp_gemv: S03.3 MLP decode path: gate+up+SwiGLU fused, then down.
 *
 * Moved verbatim out of mlp_gemv.hip.cpp (host main() and the gate's file I/O stayed
 * there) so that mlp_gemv and `engine/decode_step` compile THE SAME KERNEL SOURCE. An
 * engine TU includes exactly one of these op headers: the ops each spell their own
 * helpers (bf/fb/rnd/block_sum/reduce_row) and define colliding macros (D, H, HID,
 * EPS, SCALE, FULL_MASK), which is fine while they are in separate translation
 * units and is a compile error if they are merged into one.
 *
 * Gate for this file: prototype/mlp_gemv/ (its run.py/check.py drives the binary in
 * the same directory, which includes this header). Nothing here may be edited for
 * the engine's convenience — see the rule in prototype/README.md.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstring>

#define HID 1024      // hidden size (§00)
#define FF 3584       // intermediate size (§00)
#define ROWS_PER_BLOCK 8
#define FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit

// Injected bugs, so run.py --negative can prove the gate rejects them (the G-neg convention
// in speccheck/gates/run_negative.py: a gate that cannot fail verifies nothing).
#define NEG_NONE 0
#define NEG_HALF 1     // accumulate half of K: a wrong reduction length
#define NEG_TRUNC 2    // round toward zero instead of round-to-nearest-even
#define NEG_F32ACT 3   // do the SwiGLU entirely in F32 (the literal A-07 reading)
#define NEG_SWAPROW 4  // write each finished row into its neighbour's slot (wrong output index)
#define NEG_TAIL 5     // drop the LAST word of K: an off-by-one in the reduction bound
#define NEG_PRESWAP 6  // stage gate into the up slot and vice versa in the --pre dump, leaving
                       // `act` correct — the bug class the F1/F2 split exists to detect

// Round-to-nearest-even, same as torch's cast to bf16. The product of two bf16 values must
// round the same way the reference's bf16 multiply does, or `act` drifts by 1 ULP.
// HIP has no `__float2bfloat16_rn` (CUDA-ism, see prototype/README.md): the bf16
// constructor is the RNE conversion, and the gate in run.py would notice truncation
// immediately (it would move ~half of `act` by 1 ULP instead of ~0).
__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }

// Truncating cast, for NEG_TRUNC only: dropping the low 16 bits of the fp32 payload rounds
// toward zero, which is the classic way to get a bf16 conversion wrong.
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

// Full-wavefront reduction helper: the shift ladder starts at warpSize/2, so the same
// code is correct on a wave32 and a wave64 target (the host checks which one this is).
__device__ __forceinline__ float reduce_row(float acc) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    return acc;
}

// One `uint4` of weight (8 BF16 terms) against the staged x row, accumulated into `acc` in the
// order the reference's dot visits them. Shared by every weight-walk mapping below so the
// ARITHMETIC has exactly one spelling and only the walk can differ between them.
__device__ __forceinline__ void accum_word(float& acc, uint4 u, const float* xp) {
    const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += bf(p[j]) * xp[j];
}

// The two weight walks `mlp_gateup` can launch. Which one runs is a mapping choice, not a numerical
// one, and that is the whole reason the second one is legal to ship.
//
// WHY A SECOND ONE EXISTS. `devices/strix-halo-mixed-traffic.md` §Q9 Results: this kernel's own
// per-thread weight arithmetic, replayed as pure reads at byte-identical bytes / grid / launch
// count, runs at **0.470** of a contiguous sweep (0.629 for the single-stream `down` shape), and
// `--stage 0` exonerates the LDS staging (nostage/eng = 1.098, outside the band that would have
// blamed it). Transaction-level coalescing is already 100 % (§"CPU-side verdict on (A)"), so what
// costs the factor of two is the WALK: a lane alternates between two weight blobs 7 MiB apart every
// 16 bytes of each, and the grid covers its entire weight region in one grid pass instead of
// marching a grid-sized window through it.
//
//   MAP_INTERLEAVE  one loop, both streams' words fetched side by side (what has always shipped;
//                   the comment on that loop says why it looked like the right bet).
//   MAP_CONTIG      the gate row end to end, then the up row end to end.
//
// WHY IT IS BIT-IDENTICAL. Inside a stream the loop is still `v = lane; v < words; v += ws` over the
// same `uint4` words in the same order, accumulating into the same `ag`/`au`, and each row is still
// owned end-to-end by one wavefront, so both dots are the same F32 sums in the same order. The only
// free variable is which stream a lane touches when. Device proof:
// `prototype/mlp_gemv/run.py --mapping-identity` (byte-identical `act`/`pre`, and therefore `out`,
// layer by layer), and `--mapping contig` re-runs the whole F1-F4 gate and the negatives under the
// new walk so the claim is not "identical to a wrong kernel" either.
enum MlpMap { MAP_INTERLEAVE = 0, MAP_CONTIG = 1 };

// One wavefront per intermediate row: both projections and the SwiGLU epilogue, so the
// gate/up pair costs one launch and the 3584-element intermediates never reach memory
// unless asked for (--pre, which exists so the gate can compare them).
//
// `MAP` picks the weight walk, not the arithmetic; see `MlpMap` above for what the two walks do and
// why they cannot differ by a bit.
template <int IN, int MAP = MAP_INTERLEAVE>
__global__ void mlp_gateup(const hip_bfloat16* __restrict__ Wg,
                           const hip_bfloat16* __restrict__ Wu, const float* __restrict__ x,
                           hip_bfloat16* __restrict__ act, hip_bfloat16* __restrict__ pre,
                           int rows, int rpb, int neg) {
    __shared__ float xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    // Two independent streams. Under MAP_INTERLEAVE the compiler interleaves the two loads, which is
    // what keeps enough of them in flight to sit near the roofline at only 8 rows per block — and is
    // also the walk §Q9 measured at 0.470 of a contiguous sweep. MAP_CONTIG keeps the same two
    // streams but visits one of them at a time.
    const uint4* wg = reinterpret_cast<const uint4*>(Wg + (size_t)row * IN);
    const uint4* wu = reinterpret_cast<const uint4*>(Wu + (size_t)row * IN);
    // NEG_HALF: half of K. NEG_TAIL: the last word of K (8 of 1024 terms), which is the shape an
    // off-by-one in `v < words` actually takes — small enough that only a gate with a real
    // per-element rule notices, unlike NEG_HALF.
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
    float ag = 0.f, au = 0.f;
    if (MAP == MAP_CONTIG) {
        // Two contiguous passes. Every word of the gate row first, then every word of the up row,
        // so a lane's next load is the next 16 bytes of the stream it is already in rather than the
        // same offset 7 MiB away in the other one. Same words, same order, same accumulators.
#pragma unroll
        for (int v = lane; v < words; v += ws) accum_word(ag, wg[v], &xs[v * 8]);
#pragma unroll
        for (int v = lane; v < words; v += ws) accum_word(au, wu[v], &xs[v * 8]);
    } else {
#pragma unroll
        for (int v = lane; v < words; v += ws) {
            const float* xp = &xs[v * 8];
            accum_word(ag, wg[v], xp);
            accum_word(au, wu[v], xp);
        }
    }
    ag = reduce_row(ag);
    au = reduce_row(au);
    if (lane != 0) return;
    hip_bfloat16 g = rnd(ag, neg), u = rnd(au, neg);
    // NEG_SWAPROW: store the finished row one slot over. rows is a multiple of 3584/1024, so the
    // neighbour always exists; what matters is that every other row is now somebody else's answer.
    const int orow = neg == NEG_SWAPROW ? (row ^ 1) : row;
    // silu in F32 on the rounded gate, rounded back to BF16, THEN multiplied with the rounded
    // `up` and rounded again — three roundings, because HF's F.silu returns BF16 and the
    // product is a BF16 tensor op. Doing the whole thing in F32 (NEG_F32ACT, and the literal
    // reading of A-07) differs from the reference in 27 % of `act` elements; ulp_probe.py
    // measures that and that this three-rounding form is bit-equal to the pin. HF's order is
    // x * (1/(1+exp(-x))); writing it as a division is algebraically identical but cost the
    // first version a 2 ULP outlier at a large element (a tie in the product's BF16 rounding).
    if (neg == NEG_F32ACT) {              // the wrong reading, for the negative test only
        float a = (ag * (1.0f / (1.0f + expf(-ag)))) * au;
        act[orow] = rnd(a, neg);
    } else {
        float gf = bf(g);
        float sg = bf(fb(gf * (1.0f / (1.0f + expf(-gf)))));
        act[orow] = rnd(sg * bf(u), neg);
    }
    if (pre) {
        // NEG_PRESWAP: the two dots and the epilogue are all correct — only the staging of the
        // raw dumps swapped, so `act` matches the reference and `gate_pre` is `up_pre`'s value.
        // A gate that judged `act` alone would sign this off; the F1/F2 split is what catches it.
        const bool sw = neg == NEG_PRESWAP;
        pre[orow] = sw ? u : g;
        pre[rows + orow] = sw ? g : u;
    }
}

template <int IN>
__global__ void mlp_down(const hip_bfloat16* __restrict__ W, const float* __restrict__ x,
                         hip_bfloat16* __restrict__ out, int rows, int rpb, int neg) {
    __shared__ float xs[IN];
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
        const float* xp = &xs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += bf(p[j]) * xp[j];
    }
    acc = reduce_row(acc);
    if (lane == 0) out[neg == NEG_SWAPROW ? (row ^ 1) : row] = rnd(acc, neg);
}

/*
 * ── the token-batched pair (P4, progress/2026-10-12/plan-prefill-2026-10-12.md) ─────────────────────────
 *
 * One block owns `rpb` output rows AND a tile of `TPB` tokens. The weight row is loaded ONCE for
 * the whole tile instead of once per token, which is the entire point: the MLP is 22 MiB of weights
 * per layer and this box's L2 is 2 MiB (measured, /tmp probe: 8060S, warpSize 32, 64 KiB LDS,
 * 20 CUs), so a per-token launch pays that 22 MiB from DRAM for every token. With a TPB-token tile
 * it is paid once per TPB tokens.
 *
 * The arithmetic per output element is UNTOUCHED, and that is what makes this legal to land at
 * all: `x` is staged per token, the word loop is the same `v = lane; v < words; v += ws` over the
 * same `uint4` words in the same order, the inner 8-element loop is the same, and the ladder is the
 * same. Only the accumulator became an array, indexed by token. So output element (t, row) of the
 * batched kernel is obliged to be the bits the single-token kernel already produced for (row) with
 * token t's input — `prototype/mlp_gemv/run.py --tokens` checks exactly that on device, bit for
 * bit, and then checks the result against the prefill pins.
 *
 * Shared memory is the binding constraint on TPB, not registers: the tile is TPB*IN floats, so the
 * two shapes want different tiles (4 x 4 KiB = 16 KiB for gate/up at IN=1024; 2 x 14 KiB = 28 KiB
 * for down at IN=3584, which is within the 48 KiB static limit and leaves two blocks resident per
 * CU). The launcher picks; the kernel only ever sees one TPB.
 */

// Stage `nt` token rows of `x` (the caller's [tokens, IN] F32 tile) into shared, cooperatively.
template <int IN, int TPB>
__device__ __forceinline__ void stage_tokens(const float* __restrict__ x, float (*xs)[IN],
                                             int t0, int nt) {
    for (int i = threadIdx.x; i < nt * IN; i += blockDim.x) {
        const int col = i % IN;          // IN is a template constant: this is a shift, not a divide
        const int t = i / IN;
        xs[t][col] = x[(size_t)(t0 + t) * IN + col];
    }
    __syncthreads();
}

// `MAP` picks the weight walk only, exactly as in `mlp_gateup` above; `run.py --tokens` plus
// `--mapping-identity` is what proves the batched form keeps its bits under both.
template <int IN, int TPB, int MAP = MAP_INTERLEAVE>
__global__ void mlp_gateup_tokens(const hip_bfloat16* __restrict__ Wg,
                                  const hip_bfloat16* __restrict__ Wu,
                                  const float* __restrict__ x, hip_bfloat16* __restrict__ act,
                                  hip_bfloat16* __restrict__ pre, int rows, int tokens, int rpb,
                                  int neg) {
    __shared__ float xs[TPB][IN];
    const int t0 = (int)blockIdx.y * TPB;
    const int nt = min(TPB, tokens - t0);
    stage_tokens<IN, TPB>(x, xs, t0, nt);

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    const uint4* wg = reinterpret_cast<const uint4*>(Wg + (size_t)row * IN);
    const uint4* wu = reinterpret_cast<const uint4*>(Wu + (size_t)row * IN);
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
    float ag[TPB], au[TPB];
#pragma unroll
    for (int t = 0; t < TPB; ++t) { ag[t] = 0.f; au[t] = 0.f; }
    // No unroll pragma on the word loop: its trip count is a runtime value (NEG_HALF/NEG_TAIL make
    // `words` data, not a constant) and asking for a full unroll here only earns a compiler warning.
    // The arithmetic depends on the ORDER this loop visits words, which is the same order the
    // single-token kernel uses; whether the compiler rolls it is not observable in the result.
    if (MAP == MAP_CONTIG) {
        // Two contiguous passes, same trade as `mlp_gateup`: one stream at a time, every token of
        // the tile still accumulated per word in the same order.
        for (int v = lane; v < words; v += ws) {
            uint4 ug = wg[v];                    // once per row for the whole tile
#pragma unroll
            for (int t = 0; t < TPB; ++t) accum_word(ag[t], ug, &xs[t][v * 8]);
        }
        for (int v = lane; v < words; v += ws) {
            uint4 uu = wu[v];
#pragma unroll
            for (int t = 0; t < TPB; ++t) accum_word(au[t], uu, &xs[t][v * 8]);
        }
    } else {
        for (int v = lane; v < words; v += ws) {
            uint4 ug = wg[v], uu = wu[v];        // once per row for the whole tile
#pragma unroll
            for (int t = 0; t < TPB; ++t) {
                const float* xp = &xs[t][v * 8];
                accum_word(ag[t], ug, xp);
                accum_word(au[t], uu, xp);
            }
        }
    }
    // NEG_SWAPROW writes each finished row into its neighbour's slot, in every token's output.
    const int orow = neg == NEG_SWAPROW ? (row ^ 1) : row;
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
        if (t >= nt) break;
        // Same ladder, same order, same three roundings as the single-token epilogue above; see
        // the comment there for why silu is BF16-in/BF16-out and why the product is written as a
        // reciprocal multiply rather than a division.
        const float gs = reduce_row(ag[t]);
        const float us = reduce_row(au[t]);
        if (lane != 0) continue;
        const hip_bfloat16 g = rnd(gs, neg), u = rnd(us, neg);
        const size_t obase = (size_t)(t0 + t) * rows;
        if (neg == NEG_F32ACT) {
            const float a = (gs * (1.0f / (1.0f + expf(-gs)))) * us;
            act[obase + orow] = rnd(a, neg);
        } else {
            const float gf = bf(g);
            const float sg = bf(fb(gf * (1.0f / (1.0f + expf(-gf)))));
            act[obase + orow] = rnd(sg * bf(u), neg);
        }
        if (pre) {
            const size_t pbase = (size_t)(t0 + t) * 2 * rows;
            const bool sw = neg == NEG_PRESWAP;
            pre[pbase + orow] = sw ? u : g;
            pre[pbase + rows + orow] = sw ? g : u;
        }
    }
}

template <int IN, int TPB>
__global__ void mlp_down_tokens(const hip_bfloat16* __restrict__ W, const float* __restrict__ x,
                                hip_bfloat16* __restrict__ out, int rows, int tokens, int rpb,
                                int neg) {
    __shared__ float xs[TPB][IN];
    const int t0 = (int)blockIdx.y * TPB;
    const int nt = min(TPB, tokens - t0);
    stage_tokens<IN, TPB>(x, xs, t0, nt);

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
    float acc[TPB];
#pragma unroll
    for (int t = 0; t < TPB; ++t) acc[t] = 0.f;
    for (int v = lane; v < words; v += ws) {   // see the note in mlp_gateup_tokens
        uint4 u = wrow[v];                                   // once per row for the whole tile
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
        for (int t = 0; t < TPB; ++t) {
            const float* xp = &xs[t][v * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[t] += bf(p[j]) * xp[j];
        }
    }
    const int orow = neg == NEG_SWAPROW ? (row ^ 1) : row;
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
        if (t >= nt) break;
        const float s = reduce_row(acc[t]);
        if (lane == 0) out[(size_t)(t0 + t) * rows + orow] = rnd(s, neg);
    }
}
