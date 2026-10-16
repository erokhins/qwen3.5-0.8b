/*
 * irp_kernels.h — the device side of `prototype/issue_rate_probe`.
 *
 * Two questions, two kernel families. Both exist because a number that gates the whole prefill route
 * ranking has never been measured on this box:
 *
 *  A. the sustained VALU warp-instruction rate per CU per clock. `attn_tile_wave` sits at 95 % of a
 *     floor computed at 1 warp-inst/CU/clk
 *     (progress/2026-10-16/prefill-occupancy-results-2026-10-16.md §3). `rocminfo` says 2 SIMDs per
 *     CU, so that floor is a conservative lower bound and the real number may be 2 — which halves
 *     the instruction wall and re-ranks three funded routes. Every kernel here has a loop whose
 *     warp-instruction count per iteration is read out of the DISASSEMBLY (`isa_census.py` ->
 *     `irp_isa.gen.h`), never counted from this source text — the source text is exactly the kind of
 *     claim that has been wrong twice in this repo.
 *
 *  B. achievable read GB/s as a function of resident working-set SIZE, for the KV access shape. The
 *     prefill byte model bills attention at 237 GB/s (6.45 GB demanded at 1024 lanes), but the
 *     per-layer KV set at that width is only 2 MiB, and devices/strix-halo-mixed-traffic.md shows a
 *     resident ~54 MiB set (the GDN state) served at ~876 GB/s. If a 2 MiB KV-shaped set can be
 *     driven near 876 GB/s the shipped sweep is leaving bytes on the table; if it pins near 240 GB/s
 *     at every set size it is not, and row-tiling is unfunded.
 *
 * WHAT THE TIMING IS NOT. Nothing here takes a timing or prints a millisecond figure. `run.py` does
 * not self-acquire the GPU lock, and this binary is timed only by the main agent in an exclusive
 * window (AGENTS.md, "GPU execution is serialized"). A worker may run `--selftest` and
 * `--selftest-geom`: correctness-only, no ms.
 *
 * THE WITNESS RULE, inherited from tools/mixed_bw.cpp in intent and in two specific fixes. A
 * bandwidth number above ~240 GB/s on this part means skipped work until proven otherwise, so every
 * kernel carries a byte/value account DERIVED ON THE HOST from the same plan the launch is built
 * from — never read back out of the kernel, which is how a check becomes a tautology. The two bugs
 * that file already fixed are not repeated here:
 *   * the block sum is a REAL block reduction (`publish_block_sum`: one slot per thread in shared
 *     memory, then a halving tree). `sums[blockIdx.x] = per_thread_acc` is a per-thread store wearing
 *     a reduction's clothes and reports 1/blockDim of the bytes touched — the bug that made every
 *     warm row of the first `mixed_bw` sweep fail while its ms and GB/s looked plausible;
 *   * in `--selftest` every region carries poison guard bands and each injected bug must be caught by
 *     exactly the witness that owns it, with the must-not-catch side DERIVED from the geometry (an
 *     alias inside the buffer cannot touch a guard byte, so the guard witness must stay silent for
 *     it — a control that fires for everything proves nothing).
 *
 * EXACTNESS, because the witness is an equality and not a band. Every value a kernel consumes is an
 * INTEGER-valued BF16 and every multiply is by an operand whose runtime value is 1.0f, so each
 * partial is an exactly representable F32 integer and the accumulation order cannot matter. That
 * property has a hard boundary: an F32 integer sum is exact only below 2^24, so `plan_check` refuses
 * an `--iters` whose per-thread running sum would cross it. The block-level tree runs in double, so
 * only the per-thread sum constrains the iteration count.
 */
#ifndef IRP_KERNELS_H
#define IRP_KERNELS_H

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace irp {

// The geometry these kernels imitate. Every number names the file it comes from so a later reader can
// re-check the claim instead of trusting this comment.
constexpr int kWarpWidth = 32;                       // wave32 on this part
constexpr int kHeadDim = 256;                        // DH, engine/kernels.h
constexpr int kNumKvHeads = 2;                       // N_KV, engine/kernels.h
constexpr int kGqa = 4;                              // GQA, engine/kernels.h
constexpr int kRowBytes = kHeadDim * 2;              // 512 B: one KV row of BF16 = one warp load
constexpr int kSetBytes = kNumKvHeads * kRowBytes;   // 1024 B of K per lane index, both KV heads
// attn_tile_wave's block LDS: kTileLdsBytes in prototype/attn_decode/attn_decode_kernels.h,
// (kTilePairs*MAXSC + 2*ATTN_TILE_L*DH)*2 = (4*512 + 2*8*256)*2 = 12,288 B per 128-thread block.
constexpr int kAttnTileThreads = 128;
constexpr int kAttnTileLdsBytes = 12288;
constexpr int kAttnTileL = 8;                        // ATTN_TILE_L: lanes resident per sub-block
constexpr int kAttnMaxSc = 512;                      // MAXSC

constexpr double kBenchClkHz = 2894.0e6;             // sclk, prefill-occupancy §3
constexpr double kReadProbeGbs = 237.0;              // devices/strix-halo-memory-bandwidth.md
constexpr double kGdnResidentGbs = 876.0;            // mixed-traffic: resident ~54 MiB tier

// An F32 integer sum is exact below 2^24. Every per-thread running sum must stay under it, or the
// equality the whole witness rests on is a fiction. plan_check() enforces it.
constexpr double kF32ExactInt = 16777216.0;

#define IRP_CK(x)                                                                \
    do {                                                                         \
        hipError_t e_ = (x);                                                     \
        if (e_ != hipSuccess) {                                                  \
            ::fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                      __FILE__, __LINE__);                                       \
            ::exit(1);                                                           \
        }                                                                        \
    } while (0)

// ============================================================================================
// The block reduction, and why it is spelled out.
//
// `sums[blockIdx.x] = per_thread_acc` is NOT a block reduction: every thread of the block stores to
// the same address and the survivor is one thread's total, so the account reports 1/blockDim of the
// bytes touched. That is the bug that made every warm row of the first tools/mixed_bw sweep fail
// while its ms and GB/s looked entirely plausible (devices/strix-halo-mixed-traffic.md, "Row
// corrections after the first attempt"). This is the corrected pattern from
// tools/mixed_bw.cpp:publish_block_sum — one slot per thread in dynamic shared memory, a halving
// tree, one store per block. The values are integer-valued, so this fixed tree order is exact and
// order-independent, and no atomics are involved. blockDim.x must be a power of two (plan_check
// enforces it) or the tree silently drops a tail.
// ============================================================================================
__device__ __forceinline__ void publish_block_sum(double acc, double* __restrict__ sh,
                                                  double* __restrict__ sums) {
    const unsigned t = threadIdx.x, n = blockDim.x;
    sh[t] = acc;
    __syncthreads();
    for (unsigned half = n / 2; half > 0; half /= 2) {
        if (t < half) sh[t] += sh[t + half];
        __syncthreads();
    }
    if (t == 0) sums[blockIdx.x] = sh[0];
}

// One partial per (launch, block): a single-slot account cannot see a middle launch that never ran,
// which is what the `skip_launch` injection exists to prove is caught.
struct Args {
    const float* __restrict__ p;  // operands. p[0] is the FMA MULTIPLIER and its runtime value is
                                  // 1.0f, but the compiler cannot see that, so no fma is folded into
                                  // an add and the instruction mix stays what it claims to be.
    float* __restrict__ sink;     // one element, written only when `verify`
    double* __restrict__ sums;    // [launches] x [blocks], one block-reduced partial per launch
    unsigned long long iters;     // trips of the counted loop: the ONLY thing that scales the work
    unsigned long long launch;    // which row of `sums` this launch owns
    float* __restrict__ each;     // per-thread partials, selftest only (see `each` below)
    int verify;                   // uniform across the block, so the barriers inside are safe
    int inject;                   // Inject, see below
};

enum Inject {
    kInjNone = 0,
    kInjDropChain,  // A kernels: the last FMA chain is never computed
    kInjKvShort,    // kv: the last row index of every pass is never read
    kInjKvAlias,    // kv: the last warp reads row 0 instead of its own row
    kInjKvStray,    // kv: one extra uint4 read past the end of the K array (guard-band control)
    kInjLdsSlotOff, // lds: the last row reads row 0's slot again (two readers, one slot)
    kInjLdsShort,   // lds: one row of the tile is never read
    kInjLdsSwap     // lds: a PURE PERMUTATION of the reads. The block-sum witness is blind to it by
                    // construction and the per-thread witness is what catches it. See `each`.
};

// A defect is a SEPARATE BUILD, never a runtime branch. A branch inside the timed loop would change
// the instruction stream this probe exists to count, which is how a negative test makes the thing it
// measures true; prototype/gdn_collapse/run.py states the same rule. `IRP_NEG` is a compile-time
// constant, so every `if (IRP_NEG == ...)` folds out of the shipping binary and the timed loop is
// instruction-for-instruction identical with and without the flag. build.sh builds one binary per
// defect, and `isa_census.py --check` is what proves the CLEAN binary's loops did not move.
#ifndef IRP_NEG
#define IRP_NEG 0
#endif
// The defect's name, so one binary knows which case is ITS case: a negative that can only be
// observed in the build that carries it must not be scored against the clean cases (it would pass,
// which is the vacuous-control failure mode in tools/mixed_bw and prototype/mlp_gemv).
#ifndef IRP_NEG_NAME
#define IRP_NEG_NAME "none"
#endif

// TWO WITNESSES, because one of them cannot see the bug class that matters most here.
//
// `sums` is a block-reduced total: it bills how many bytes were touched and which residue class of
// value they carried. A PERMUTATION of the same reads leaves it exactly equal — a sum cannot see a
// swap. That is the shape of the `attn_tile_wave` §19/§20 defect (two self-consistent partial layouts
// joined by a branch, invisible to every per-form check), so the selftest also reads per-thread
// partials through `each` and compares them element by element against a host mirror of the index
// arithmetic. `kInjLdsSwap` is the case that proves the difference: it must be invisible to the block
// sum and fatal to the per-thread mirror. A control that both witnesses pass is decoration.

// ============================================================================================
// QUESTION A — kernels with a warp-instruction count per iteration that the DISASSEMBLY owns.
//
// Every loop here is `_Pragma("unroll 1")`: exactly one trip of the loop body is one billed
// iteration, so the count `isa_census.py` reads out of the ISA is the count the host bills. If the
// compiler unrolled one of these loops anyway, the census would see a bigger body and say so; that
// is the forcing, and it is why the count is never copied from the source text.
//
// The operands come from a device buffer, not from literals, for two reasons. (1) Nothing is
// constant-folded away: the loop body the compiler emits is the loop body the machine issues.
// (2) `p[0]`, the FMA multiplier, holds 1.0f at run time, which makes every partial an exact integer
// the host can predict — but the compiler cannot see that, so `__fmaf_rn` stays a real v_fma_f32
// instead of being folded into an add. The seeded addends b_j = j+1 are distinct on purpose: a
// dropped chain changes the witness by its own b_j, so the failure names the chain it lost.
// ============================================================================================

// THE STATE MUST BE PER-LANE OR THE MEASUREMENT IS OF THE WRONG PIPE. The first version seeded every
// lane with the same value, and the disassembly showed what the compiler did with that: `s_fmac_f32`
// and `s_mov_b32`, 29 scalar instructions per iteration and not one VALU instruction, because a
// computation every lane of a wave performs identically is uniform and RDNA scalarizes uniform work.
// The `+ (float)threadIdx.x` seed keeps the accumulators in VGPRs; the multiplicand and addend stay
// warp-uniform on purpose, which is attn's own shape (`q` per lane, the scale not).
#define IRP_A_FMA(NAME, N)                                                        \
__global__ void NAME(const Args a) {                                              \
    extern __shared__ double sh[];                                                \
    float acc[N];                                                                 \
    _Pragma("unroll")                                                             \
    for (int j = 0; j < (N); ++j) acc[j] = a.p[64 + j] + (float)threadIdx.x;    \
    const float m = a.p[0];                                                       \
    const float b = a.p[1];                                                       \
    _Pragma("unroll 1")                                                           \
    for (unsigned long long i = 0; i < a.iters; ++i) {                             \
        _Pragma("unroll")                                                         \
        for (int j = 0; j < (N); ++j) {                                           \
            if (IRP_NEG == kInjDropChain && j == (N) - 1) continue;               \
            acc[j] = __fmaf_rn(m, acc[j], b);                                      \
        }                                                                         \
    }                                                                             \
    float s = 0.0f;                                                               \
    _Pragma("unroll")                                                             \
    for (int j = 0; j < (N); ++j) s += acc[j];                                     \
    if (a.verify) {                                                               \
        if (threadIdx.x == 0 && blockIdx.x == 0 && a.sink) *a.sink = s;             \
        publish_block_sum((double)s, sh, a.sums + a.launch * gridDim.x);           \
    }                                                                             \
}

// N independent chains: the issue-limited end. N = 1 is the same code and doubles as a latency point,
// but `serial` below is the explicit dependent-chain case, because it holds the INSTRUCTION COUNT
// fixed against an independent-chain twin and varies only the dependency graph.
IRP_A_FMA(irp_fma_1, 1)
IRP_A_FMA(irp_fma_4, 4)
IRP_A_FMA(irp_fma_8, 8)
IRP_A_FMA(irp_fma_16, 16)
IRP_A_FMA(irp_fma_32, 32)

// The dependent-chain variant: the same N v_fma_f32 per iteration as `irp_fma_N`, but ONE serial
// chain, so an iteration costs N dependent latencies instead of N issue slots. Paired against
// `irp_fma_N` at equal instruction count it separates the latency-bound floor from the issue-bound
// one, which is the whole point of item (4) of the brief and the lesson of
// prototype/gdn_collapse/results.md §5 (an instruction-count model that assumes issue slots the
// machine does not have at its realised occupancy was 9.8x wrong).
#define IRP_A_SERIAL(NAME, N)                                                     \
__global__ void NAME(const Args a) {                                              \
    extern __shared__ double sh[];                                                \
    float acc = a.p[64] + (float)threadIdx.x;                              \
    const float m = a.p[0];                                                       \
    const float b = a.p[1];                                                       \
    _Pragma("unroll 1")                                                           \
    for (unsigned long long i = 0; i < a.iters; ++i) {                             \
        _Pragma("unroll")                                                         \
        for (int j = 0; j < (N); ++j) {                                           \
            if (IRP_NEG == kInjDropChain && j == (N) - 1) break;                  \
            acc = __fmaf_rn(m, acc, b);                                           \
        }                                                                         \
    }                                                                             \
    if (a.verify) {                                                               \
        if (threadIdx.x == 0 && blockIdx.x == 0 && a.sink) *a.sink = acc;           \
        publish_block_sum((double)acc, sh, a.sums + a.launch * gridDim.x);         \
    }                                                                             \
}
IRP_A_SERIAL(irp_serial_8, 8)
IRP_A_SERIAL(irp_serial_32, 32)

// The bf16 -> f32 conversion tax: item (2) of the brief. Per chain per iteration the loop issues one
// conversion feeding one FMA, which is the attn score shape `dot += bf(k[t]) * q[t]` with the operand
// held in a register instead of LDS (the LDS version is `irp_lds_tile` below).
//
// The bf16 register must change every iteration or the conversion is loop-invariant and LICM hoists it
// into a register, which would silently turn this kernel back into `irp_fma_N`. The liveness is bought
// with a two-instruction rotate (v_cmp + v_cndmask on a 16-bit value) that alternates the operand
// between 1.0 and 2.0; `isa_census.py` prints the family histogram of the emitted body, so the rotate
// is attributable and subtractable rather than hidden inside a claimed count. Nothing in this kernel
// localises WHICH chain moved — the addends are equal by construction — and `RESULT` says so.
#define IRP_A_CVT(NAME, N)                                                        \
__global__ void NAME(const Args a) {                                              \
    extern __shared__ double sh[];                                                \
    const hip_bfloat16 one = hip_bfloat16(1.0f);                           \
    const hip_bfloat16 two = hip_bfloat16(2.0f);                           \
    hip_bfloat16 raw[N];                                                          \
    float acc[N];                                                                 \
    _Pragma("unroll")                                                             \
    for (int j = 0; j < (N); ++j) {                                             \
        acc[j] = a.p[64 + j] + (float)threadIdx.x;                             \
        raw[j] = ((threadIdx.x & 1u) == 0u) ? one : two;                       \
    }                                                                             \
    const float m = a.p[0];                                                       \
    _Pragma("unroll 1")                                                           \
    for (unsigned long long i = 0; i < a.iters; ++i) {                             \
        _Pragma("unroll")                                                         \
        for (int j = 0; j < (N); ++j) {                                           \
            const float x = float(raw[j]);                             \
            if (IRP_NEG == kInjDropChain && j == (N) - 1) continue;               \
            acc[j] = __fmaf_rn(m, x, acc[j]);                                      \
            raw[j] = (raw[j] == one) ? two : one;                                 \
        }                                                                         \
    }                                                                             \
    float s = 0.0f;                                                               \
    _Pragma("unroll")                                                             \
    for (int j = 0; j < (N); ++j) s += acc[j];                                     \
    if (a.verify) {                                                               \
        if (threadIdx.x == 0 && blockIdx.x == 0 && a.sink) *a.sink = s;             \
        publish_block_sum((double)s, sh, a.sums + a.launch * gridDim.x);           \
    }                                                                             \
}
IRP_A_CVT(irp_cvt_8, 8)
IRP_A_CVT(irp_cvt_16, 16)

// ============================================================================================
// The LDS-heavy kernel: item (3) of the brief. One iteration stages EXACTLY attn's block LDS —
// kAttnTileLdsBytes = 12,288 B, the same 128-thread block as `attn_tile_wave` — and reads it back
// with attn's instruction widths: the stage goes in as uint4 (ds_store_b128) and comes out as 48
// individual 16-bit loads (ds_read_u16) feeding one FMA each, which is `dot += bf(k[t]) * q[t]`.
//
// The reduction partials ALIAS the tile instead of adding a second allocation, so the declared LDS
// stays exactly 12,288 B and the residency arithmetic matches the kernel being imitated
// (prototype/attn_decode/results.md §20 counts 5 blocks/CU at that figure). The aliasing is safe by
// barrier: every read of the tile completes before `publish_block_sum` writes into it.
//
// The tile is 24 rows of 256 BF16 (24 x 512 B = 12,288 B) and each thread stages one uint4 per row
// group with the SAME cross-warp shape as attn's `for (rr = s_lo + w; rr < s_hi; rr += nw)`, so a
// thread reads bytes another warp wrote and the barriers are load-bearing rather than decorative.
// `reuse` repeats the read side; 1 is the brief's symmetric 12 KiB write + 12 KiB read, and 4 is
// attn's actual read:write LDS ratio (G = 4 query heads read one staged K row), which is what to
// book if the question is attn's LDS pressure rather than raw LDS bandwidth.
// ============================================================================================
constexpr int kLdsRows = kAttnTileLdsBytes / kRowBytes;          // 24 rows of 256 BF16
constexpr int kLdsWords = kAttnTileLdsBytes / 16;                // 768 uint4 stage words
constexpr int kLdsWordsPerThread = kLdsWords / kAttnTileThreads; // 6 per thread
constexpr int kLdsElemsPerThread = kLdsWordsPerThread * 8;       // 48 BF16 read back per thread

__device__ __forceinline__ float irp_tag(int e) { return (float)(1 + (e % 7)); }

__global__ void irp_lds_tile(const Args a, const int reuse) {
    extern __shared__ __align__(16) unsigned char raw[];
    hip_bfloat16* tile = reinterpret_cast<hip_bfloat16*>(raw);
    double* const sh = reinterpret_cast<double*>(raw);   // valid only after the read-side barrier
    const int lane = threadIdx.x % warpSize;
    const int w = threadIdx.x / warpSize;
    const int nw = (int)blockDim.x / warpSize;
    float acc = 0.0f;
    _Pragma("unroll 1")
    for (unsigned long long i = 0; i < a.iters; ++i) {
        // ---- stage: 6 uint4 per thread, one per row this warp owns (rows w, w+nw, w+2nw, ...).
        // The uint4 is assembled in registers and stored as ONE ds_store_b128, which is the width
        // attn's staging uses (`*reinterpret_cast<uint4*>(&s_k[slot*DH + lane*8]) = ...`).
        _Pragma("unroll")
        for (int k = 0; k < kLdsWordsPerThread; ++k) {
            const int row = w + k * nw;
            union { uint4 u; hip_bfloat16 e[8]; } v;
            const int base = row * kHeadDim + lane * 8;
            _Pragma("unroll")
            for (int t = 0; t < 8; ++t) v.e[t] = hip_bfloat16(irp_tag(base + t));
            *reinterpret_cast<uint4*>(&tile[base]) = v.u;
        }
        __syncthreads();                              // tile published
        // ---- read back: 48 ds_read_u16 per thread, one FMA each, in attn's dot shape
        const float q = a.p[0];
        for (int rep = 0; rep < reuse; ++rep) {
            _Pragma("unroll")
            for (int k = 0; k < kLdsWordsPerThread; ++k) {
                int row = w + k * nw;
                // Two readers, one slot: the last row of the tile re-reads row 0 instead of itself.
                // The read COUNT is unchanged, so only the value account can see it, and because the
                // steal is not a permutation (row 23 goes unread) the block sum sees it too.
                if (IRP_NEG == kInjLdsSlotOff && row == kLdsRows - 1) row = 0;
                // A pure permutation of the READ slots: each thread reads the 8 words its neighbour
                // lane staged, and the neighbour map (lane -> lane^1) is a bijection, so every staged
                // word is still read exactly once overall. The block sum is invariant by construction;
                // only the per-thread mirror can see it. A row-level rotation would NOT do: with
                // nw = threads/32 = 4 the warp row-cosets carry equal tag sums, so a row rotation is
                // invisible to BOTH witnesses and the control would be vacuous (measured: differ=0).
                int off = row * kHeadDim + lane * 8;
                if (IRP_NEG == kInjLdsSwap) off ^= 8;      // lane ^= 1 in element units
                if (IRP_NEG == kInjLdsShort && row == kLdsRows - 1) continue;
                float dot = 0.0f;
                _Pragma("unroll")
                for (int t = 0; t < 8; ++t) dot += float(tile[off + t]) * q;
                acc += dot;
            }
        }
        __syncthreads();   // every read of the tile is done; the reduction may now alias those bytes
    }
    if (a.verify) {
        if (a.each) a.each[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
        publish_block_sum((double)acc, sh, a.sums + a.launch * gridDim.x);
    }
}

// ============================================================================================
// QUESTION B — the KV-shaped streaming read.
//
// The access shape is `attn_tile_wave`'s, not a generic sweep: rows of `kHeadDim` BF16 = 512 B, two
// KV heads, ONE WARP PER ROW, so one warp instruction consumes 32 x 16 B = 512 B = four complete
// 128-byte lines with nothing partially used, and the lane walks the row's uint4 words exactly as the
// engine's GEMVs do. The address expression is spelled the way the cache is indexed in the engine
// (`kcache + ((kv * cap + j) * DH)`, prototype/attn_decode/attn_decode_kernels.h) rather than as a
// flat `r * DH`; they evaluate identically and the compiler may fold one into the other, but the
// version here is the one a reader can check against the shipped kernel.
//
// The load cannot be optimised out because the reduction happens AT THE USE SITE: the row is unpacked
// and dotted against a query vector that was loaded into registers before the loop, which is exactly
// `dot += bf(k[t]) * q[t]` from the score loop. The query comes from device memory, so its value is
// not a compile-time constant and the dot survives.
//
// Working-set SIZE is the independent variable, and the reuse is what makes it one: `passes` re-walks
// the SAME bytes inside one launch, which is the shape of attention re-reading a 2 MiB per-layer KV
// set thousands of times at 1024 lanes. A single cold sweep of 2 MiB would measure dispatch, not the
// memory system.
// ============================================================================================
__global__ void irp_kv_read(const Args a, const hip_bfloat16* __restrict__ kcache,
                            unsigned long long rows, unsigned int passes) {
    extern __shared__ double sh[];
    const int lane = threadIdx.x % warpSize;
    const unsigned long long nwarps = (unsigned long long)gridDim.x * (blockDim.x / warpSize);
    const unsigned long long gw =
        (unsigned long long)blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    const unsigned long long cap = rows / 2;          // lanes per KV head: N_KV = 2
    float q[8];
    _Pragma("unroll")
    for (int t = 0; t < 8; ++t) q[t] = a.p[8 + t];     // the query, hoisted like attn's
    float acc = 0.0f;
    for (unsigned int p = 0; p < passes; ++p) {
        for (unsigned long long r = gw; r < rows; r += nwarps) {
            if (IRP_NEG == kInjKvShort && r == rows - 1) continue;
            unsigned long long rr = r;
            // The last warp reads row 0 again instead of its own row. The read COUNT is identical, so
            // the only thing that can catch it is the value tag, and it can do so only when the two
            // rows carry different tags — see `alias_catchable` on the host, which DERIVES whether
            // this case can fire instead of hand-listing it.
            if (IRP_NEG == kInjKvAlias && gw == nwarps - 1) rr = 0;
            const unsigned long long kv = rr / cap, j = rr % cap;
            unsigned long long off = (kv * cap + j) * (unsigned long long)kHeadDim + lane * 8;
            // The guard band's own negative control: one row PAST the end of the buffer, which is
            // only a legal address because --selftest put poison there. Poison is 512.0, outside the
            // tag domain 1..7, so a stray read moves the checksum instead of matching it.
            if (IRP_NEG == kInjKvStray && gw == 0 && p == 0)
                off = rows * (unsigned long long)kHeadDim + lane * 8;
            union { uint4 u; hip_bfloat16 e[8]; } v;
            v.u = *reinterpret_cast<const uint4*>(kcache + off);
            float dot = 0.0f;
            _Pragma("unroll")
            for (int t = 0; t < 8; ++t) dot += float(v.e[t]) * q[t];
            acc += dot;
        }
    }
    if (a.verify) publish_block_sum((double)acc, sh, a.sums + a.launch * gridDim.x);
}

// Fills, chosen so that an account built on them is an equality and not a shrug:
//   * KV rows carry irp_tag(r) = 1 + (r % 7), integer-valued BF16, so the checksum is exact and a
//     duplicated or skipped row moves it;
//   * guard bands carry 512.0, an exact BF16 integer OUTSIDE the tag domain, so a read that strays
//     out of the region changes the sum instead of coincidentally matching it.
constexpr float kTagMod = 7.0f;
constexpr float kGuardFill = 512.0f;

__device__ __forceinline__ float irp_row_tag(unsigned long long r) {
    return (float)(1 + (int)(r % (unsigned long long)kTagMod));
}

__global__ void fill_rows_k(hip_bfloat16* p, unsigned long long n, unsigned long long mod) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s)
        p[i] = hip_bfloat16((float)(1.0 + (double)((i / kHeadDim) % mod)));
}
__global__ void fill_const_k(hip_bfloat16* p, size_t n, float v) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s)
        p[i] = hip_bfloat16(v);
}

}  // namespace irp

#endif  // IRP_KERNELS_H
