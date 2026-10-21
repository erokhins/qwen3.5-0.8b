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
//
// This is the k=1 form. `engine/kernels_proj.cpp` no longer launches it — the engine issues
// `proj_gemv_tokens` for every width, `TPB = 1` included — and it stays here as the gate's
// independent per-row reference: `run.py --tokens` diffs the tile against this kernel run one row
// at a time, which is a different code path rather than the same one with a different template
// argument (`--kernel k1`). See the batched form below.
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

/*
 * ── the token-batched form (P5, progress/2026-10-12/plan-prefill-2026-10-12.md §4 row P5) ───────────────────
 *
 * One block owns `rpb` output rows AND a tile of `TPB` tokens, so each weight row is read ONCE for
 * the whole tile instead of once per token. That is the entire point: the nine projections are
 * 0.4677 GB of the token's bytes, `results.md` measures the family at 63–66 % of the read probe,
 * and the narrow shapes are the weak ones (`k_proj`/`v_proj` 39–40 %, `o_proj`/`out_proj` 46–49 %)
 * because a [512,1024] launch is 16 blocks and cannot fill 20 CUs. Widening a launch along the
 * token axis is the only lever those shapes have; §06.2's "one GEMM per projection" is this kernel.
 *
 * WHY THIS IS ALLOWED TO TOUCH THE LEGAL ROUTE AT ALL: the per-output-element arithmetic is
 * untouched. `x` is staged per token, the word walk is the same `v = lane; v < words; v += ws` over
 * the same `uint4` words in the same order, the inner 8-element loop is the same, the ladder is the
 * same `reduce_row`, and there is still exactly one BF16 rounding per dot. Only the accumulator
 * became an array indexed by token, and no accumulator ever crosses a token — the same
 * "reduction state stays inside one row" rule A-17 states for attention. So output element (t, row)
 * of the tile is OBLIGED to be the bits the k=1 kernel already produced for `row` with token t's
 * input, and `run.py --tokens` checks that on device, bit for bit, before it checks the pins.
 *
 * Unlike `mlp_gemv`'s batched pair this one is not a second kernel the engine also has to keep
 * honest: `engine/kernels_proj.cpp` instantiates THIS template for every width, `tokens == 1`
 * included, where `TPB = 1` and `gridDim.y = 1` give back the k=1 kernel's arithmetic and its launch
 * shape byte for byte (one staged row, one accumulator, the same shared footprint). One arithmetic,
 * two tile widths — which is what the P5 brief asks for and what P4's two-branch launcher did not do.
 *
 * Shared memory is what bounds TPB, not registers: the tile is TPB*IN floats, so the two input
 * widths want different tiles to land on the same 16 KiB budget the gated MLP chose at IN=1024
 * (4 x 4 KiB at IN=1024, 2 x 8 KiB at IN=2048; the device reports 64 KiB of LDS, and staying at
 * 16 KiB keeps several blocks resident per CU). The launcher picks TPB; the kernel sees one.
 * TPB is unmeasured — same as the MLP's — and `--tpb` exists so the main agent's timing pass can
 * sweep it without a rebuild (engine/performance-status.md §6: measurements are the main agent's).
 */
template <int IN, int TPB, bool PACK_X>
__global__ void proj_gemv_tokens(const hip_bfloat16* __restrict__ W, const void* __restrict__ x,
                                 hip_bfloat16* __restrict__ out, int rows, int tokens, int rpb,
                                 int neg) {
    // The staged x is BF16 when PACK_X and F32 otherwise — the same two choices the k=1 kernel offers,
    // and `tof` below converts either to F32 the same way, so the products are identical.
    using Xstaged = typename std::conditional<PACK_X, hip_bfloat16, float>::type;
    __shared__ Xstaged xs[TPB][IN];
    const int t0 = (int)blockIdx.y * TPB;           // this block's first token
    const int nt = min(TPB, tokens - t0);           // grid.y = ceil(tokens/TPB), so nt >= 1
    // Cooperative staging of the tile, cooperatively: the same widening the k=1 kernel does for one
    // row, done for `nt` of them. `IN` is a template constant, so `i % IN` is a shift, not a divide.
    for (int i = threadIdx.x; i < nt * IN; i += blockDim.x) {
        const int col = i % IN;
        const int t = i / IN;
        if (PACK_X) {
            xs[t][col] = ((const hip_bfloat16*)x)[(size_t)(t0 + t) * IN + col];
        } else {
            xs[t][col] = ((const float*)x)[(size_t)(t0 + t) * IN + col];
        }
    }
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    float acc[TPB];
#pragma unroll
    for (int t = 0; t < TPB; ++t) acc[t] = 0.f;

    if (neg == NEG_TRANSPOSE) {
        // Read the same bytes as [in, rows]: in-bounds for any shape, and a different set of
        // products. The weight value is loop-invariant across the tile, so it is hoisted — the
        // per-token product and its order are unchanged.
        for (int k = lane; k < IN; k += ws) {
            const float w = bf(W[(size_t)k * rows + row]);
#pragma unroll
            for (int t = 0; t < TPB; ++t) {
                if (t >= nt) break;
                acc[t] += w * tof(xs[t][k]);
            }
        }
    } else {
        const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
        int words = IN / 8;
        if (neg == NEG_TAIL) words -= 1;
        else if (neg == NEG_HALF) words /= 2;
        // No unroll pragma: `words` is a runtime value once the negatives can shrink it (the note in
        // `mlp_gemv_kernels.h` says the same), and what matters here is the ORDER this loop visits
        // words, which is the k=1 kernel's order.
        for (int v = lane; v < words; v += ws) {
            const uint4 u = wrow[v];                 // ONE weight-word load for the whole tile
            const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int t = 0; t < TPB; ++t) {
                if (t >= nt) break;
                const Xstaged* xp = &xs[t][v * 8];
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[t] += bf(p[j]) * tof(xp[j]);
            }
        }
    }

    // The row permutations (§05.1's qhalf, §04.1's kvswap, the row swap) apply per row, in every
    // token's output — the same `perm_out_row` the k=1 kernel calls, so a negative that localises in
    // one form localises in the other and neither is a special case of the other.
    const int orow = perm_out_row(row, rows, neg);
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
        if (t >= nt) break;
        const float s = reduce_row(acc[t]);
        if (lane == 0) out[(size_t)(t0 + t) * rows + orow] = rnd(s, neg);
    }
}

/*
 * ── the DECODE variants (D1) ────────────────────────────────────────────────────────────────
 *
 * The attribution (`progress/2026-10-21/decode-attribution-2026-10-21.md`) bills the projection
 * family 4.234 ms/token in situ for 466.7 MB of weight reads against a 1.973 ms byte floor, and
 * this op's own `results.md` measures the family at 66-71 % of the 237 GB/s probe — while
 * `lm_head_gemv_full` reads 509 MB at 99.6 % of the same probe. Diffing the two kernels leaves
 * exactly three structural differences, and this section makes each one a separate, gated axis so
 * a sweep can attribute the gap instead of guessing at it:
 *
 *   UNROLL. `lm_head`'s word loop carries `#pragma unroll` over a COMPILE-TIME trip count
 *   (128 words / 32 lanes), so LLVM keeps several 16 B weight loads in flight. `proj_gemv_tokens`
 *   lost its pragma when the injected-bug vocabulary made the word bound a runtime value (see the
 *   note in `mlp_gemv_kernels.h`), so the shipping form pipelines one load per iteration. Here the
 *   bound is compile-time again and the negatives shrink the walk by SKIPPING tail iterations in a
 *   separate runtime-bounded branch, so the code the gate measures is the code the sweep times.
 *   Same terms, same order, same tree: bit-identical to variant 0.
 *
 *   LPW = lanes per row. The shipping form fixes this at one wavefront (32), so a K=1024 row is 4
 *   uint4 words per lane behind a 5-step ladder and a K=2048 row is 8. LPW below 32 puts
 *   32/LPW rows into every wavefront — the "more in-flight rows per block" half of the lever that
 *   `results.md` names as its own untried one — at the cost of a longer per-lane serial sum
 *   (IN/(8*LPW) words) and a shorter ladder (log2 LPW steps). This CHANGES THE PARENTHESES of the
 *   dot: lane l accumulates words {l, l+LPW, ...} instead of {l, l+32, ...} and the tail is a
 *   log2(LPW)-step sum of 32/LPW per-lane partials. Per results.md §F1 a K=1024 dot already moves
 *   >1 BF16 ULP when the reference's own F32 sum is reordered, so LPW < 32 is NOT bit-identical
 *   and is judged by the k=1 gate's own rules — no large element above 1 ULP inside the
 *   cancellation band, ndiff inside the band, cos >= 0.9999, meanrel <= 1e-2 — which is precisely
 *   the standard variant 0 already passes. No band is widened for any variant.
 *
 *   VX = vector-stage `x`. The shipping staging loop moves ONE element per lane per step (a 4-byte
 *   F32 load, a shift and a multiply for the index, one shared store). VX moves a uint4: 4 F32 or 8
 *   BF16 per step. A copy is a copy, so this is bit-identical whatever it feeds.
 *
 * ROWS PER BLOCK AND THREADS PER BLOCK ARE INDEPENDENT. `rowsPerBlock` (`--rows`) is how many output
 * rows one block owns; `blockDim.x` (`--threads`) is how many rows the block walks at a time,
 * `step = blockDim.x / LPW`, so the block loops `rpb` in steps of `step`:
 *
 *     for (int r = threadIdx.x / LPW; r < rpb; r += step) row = blockIdx.x*rpb + r;
 *
 * `threads = rpb*LPW` is therefore one loop trip — the shape `--rows` alone used to produce, and the
 * default when `--threads` is not given, so nothing already measured moves. `threads < rpb*LPW` makes
 * one block walk its rows in several trips (fewer resident rows per block, same work per row); a
 * `--rows-list` sweep at fixed `--threads` is what measures the two separately, which the welded form
 * could not: it forced 256 threads at rpb 8 and 1024 at rpb 32, so a request for "rpb 32 at 512
 * threads" was unspeakable and any published number taken at one thread setting could only be
 * reproduced at the one rpb that implied it.
 *
 * Row mapping is otherwise the shipping form's with 32 replaced by LPW: `lane = threadIdx.x % LPW`, so
 * every row is summed by exactly `LPW` consecutive lanes with the same word walk whatever `--threads`
 * says, and the width-limited `__shfl_down_sync(..., LPW)` stays legal — `--threads` must be a whole
 * number of wavefronts (`% warpSize == 0`, the launcher refuses otherwise), which makes each row's
 * LPW lanes one aligned segment of one wavefront. The trip index `threadIdx.x / LPW` is constant
 * inside a wavefront, so a trip is taken or skipped by whole wavefronts and no lane ever leaves a
 * shuffle early. A trip's out-of-range rows are CLAMPED, not early-returned, and their stores masked.
 */
template <int IN, int TPB, int LPW, bool PACK_X, bool VX>
__global__ void proj_gemv_var(const hip_bfloat16* __restrict__ W, const void* __restrict__ x,
                              hip_bfloat16* __restrict__ out, int rows, int tokens, int rpb,
                              int neg) {
    static_assert(LPW == 4 || LPW == 8 || LPW == 16 || LPW == 32,
                  "__shfl_down_sync width must be 4, 8, 16 or 32");
    static_assert(IN % (8 * LPW) == 0, "LPW must divide the row's uint4 word count");
    using Xstaged = typename std::conditional<PACK_X, hip_bfloat16, float>::type;
    // 16-byte aligned so VX may move uint4s on the shared side as well as the global side. Both
    // staged row strides (IN*4 and IN*2 with IN in {1024, 2048}) are multiples of 16 B.
    __shared__ __align__(16) Xstaged xs[TPB][IN];
    const int t0 = (int)blockIdx.y * TPB;
    const int nt = min(TPB, tokens - t0);
    if (VX) {
        // One uint4 per lane per step, no conversion: the same bytes, four or eight at a time.
        constexpr int VPU = PACK_X ? 8 : 4;      // elements per 16-byte unit
        constexpr int UPR = IN / VPU;            // 16-byte units per token row
        for (int i = threadIdx.x; i < nt * UPR; i += blockDim.x) {
            const int t = i / UPR, u = i % UPR;
            *(uint4*)&xs[t][u * VPU] = ((const uint4*)x)[(size_t)(t0 + t) * UPR + u];
        }
    } else {
        for (int i = threadIdx.x; i < nt * IN; i += blockDim.x) {
            const int col = i % IN, t = i / IN;
            if (PACK_X) xs[t][col] = ((const hip_bfloat16*)x)[(size_t)(t0 + t) * IN + col];
            else xs[t][col] = ((const float*)x)[(size_t)(t0 + t) * IN + col];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % LPW;
    // Rows this block walks at a time. blockDim.x is always a multiple of LPW: LPW is a power of two
    // at or below warpSize and the launcher requires threads % warpSize == 0.
    const int step = (int)blockDim.x / LPW;

    for (int r = threadIdx.x / LPW; r < rpb; r += step) {
    int row = blockIdx.x * rpb + r;
    const bool live = row < rows;
    if (!live) row = rows - 1;                    // clamped: valid loads, masked store

    // Declared INSIDE the trip loop: a block that walks its rows in several trips must start each
    // row's accumulators at zero, or the second trip would add onto the first row's sum. The
    // compiler keeps them in the same registers either way.
    float acc[TPB];
#pragma unroll
    for (int t = 0; t < TPB; ++t) acc[t] = 0.f;

    if (neg == NEG_TRANSPOSE) {
        for (int k = lane; k < IN; k += LPW) {
            const float w = bf(W[(size_t)k * rows + row]);
#pragma unroll
            for (int t = 0; t < TPB; ++t) {
                if (t >= nt) break;
                acc[t] += w * tof(xs[t][k]);
            }
        }
    } else {
        const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
        auto dot8 = [&](int v) {
            const uint4 u = wrow[v];              // ONE 16-B weight load for the whole tile
            const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int t = 0; t < TPB; ++t) {
                if (t >= nt) break;
                const Xstaged* xp = &xs[t][v * 8];
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[t] += bf(p[j]) * tof(xp[j]);
            }
        };
        constexpr int WORDS = IN / 8;
        if (neg == NEG_TAIL || neg == NEG_HALF) {
            // The injected reduction-length bugs, on a runtime bound. Not the timed path.
            int words = WORDS;
            if (neg == NEG_TAIL) words -= 1;
            else words /= 2;
            for (int v = lane; v < words; v += LPW) dot8(v);
        } else {
            // The timed path. Spelled as a COMPILE-TIME trip count over `it`, not a runtime
            // `v < WORDS` bound: `#pragma unroll` on the strided form is silently refused by clang
            // ("loop not unrolled", first build of this file), which would have made variant 1 a
            // no-op experiment. The two spellings visit the same words in the same order for every
            // lane: LPW divides WORDS (static_assert above) and lane < LPW, so every lane runs
            // exactly WORDS/LPW times, and that is also the k=1 kernel's visit order when LPW=32.
            //
            // `unroll 4`, not full unroll: 4 uint4 loads in flight is what `lm_head_gemv_full`
            // actually realises (128 words / 32 lanes), and at LPW=4 with IN=2048 a full unroll
            // would be 64 iterations and 256 live registers.
            constexpr int ITER = WORDS / LPW;
#pragma unroll 4
            for (int it = 0; it < ITER; ++it) dot8(lane + it * LPW);
        }
    }

    const int orow = perm_out_row(row, rows, neg);
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
        if (t >= nt) break;
        float s = acc[t];
        for (int o = LPW / 2; o; o >>= 1) s += __shfl_down_sync(FULL_MASK, s, o, LPW);
        if (lane == 0 && live) out[(size_t)(t0 + t) * rows + orow] = rnd(s, neg);
    }
    }   // for each trip of rows this block owns
}
