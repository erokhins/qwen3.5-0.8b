// wmma_m16_probe.cpp — D57's compile-only falsifier for item 2.6 (re-tiled attention at M >= 16).
//
// NEVER LAUNCHED. `-c` only: no link step, no main, no launcher. No timing is taken here and none may
// be quoted from anything built from this file. The deliverable is the GCN ISA and the slots-per-touch
// figure `m16_census.py` reads out of it.
//
// THE QUESTION (D57, `progress/2026-10-16/prefill-decision-log-2026-10-16.md`; the gate is its
// second-to-last bullet)
// ---------------------------------------------------------------------------------------------
// Inside a block that owns M >= 16 QUERY ROWS, with K/V staged the way the shipped kernel stages it,
// what does one `(row, lane, head)` touch cost once the score dot AND the warp fold are
// `v_wmma_f32_16x16x16_bf16`?  <= 19 slots/touch -> 2.6 is fundable; >= 30 -> the fragment
// redistribution ate the fold and D30's funded-no stands. `prototype/attn_wmma/results.md` §3 measured
// the score FEED (13.5 slots per 16x16x16 tile) but its drain went into an `f32` row with no BF16
// round, in a clean-room loop with no `s_sc` round and no shipped barrier pair. This file measures the
// same tile inside a block shape that has them.
//
// WHY EVERY VARIANT SHARES ONE BLOCK SHAPE
// ----------------------------------------
// The shipped `attn_slim_wave` (`engine/attn_slim_kernels.h`) is M = 1 (`static_assert(ATTN_TILE_M ==
// 1)`): one query row per block, 128 threads = 4 warps, one warp per query head of the GQA group, a K
// sub-block of ATTN_TILE_L = 8 lanes staged into LDS by all four warps (one warp per lane row, one
// `uint4` per lane at `slot*DH + lane*8`), TWO `__syncthreads()` per sub-block (after the stage, after
// `s_sc` is published), and scores published into `s_sc[h][rr]` as `fb(bf(fb(dot)) * SCALE)` — BF16,
// row stride MAXSC.
//
// A 16x16 WMMA tile needs 16 KV lanes, so the sub-block here is 16 lanes deep (MLANES = 16, not 8); the
// block owns MROWS = 16 query rows, which cannot live in VGPRs (16 rows x 8 dims = 128 f32/lane), so Q
// is staged in LDS in the same row-major-DH layout the K stage uses. Everything else is the shipped
// shape spelled the same way, because the thing under test is the redistribution cost in *that* block
// — not in a loop with no barriers, no score round and no shared K stage.
//
// THE SCORE STAGING IS FORCED TILE-LOCAL, AND THAT IS A FINDING, NOT AN OPTIMISATION
// ----------------------------------------------------------------------------------
// The shipped `s_sc` holds a whole row: `s_sc[ATTN_TILE_G][MAXSC]` = 4 x 512 x 2 B = 4 KiB, affordable
// only because M = 1. At M = 16 the same whole-row staging is 4 heads x 16 rows x 512 lanes x 2 B =
// **65,536 B = the entire 64 KiB LDS budget of this CU** (`tools/kernel_occupancy_probe.cpp`,
// `prototype/attn_wmma/evidence/occupancy.txt`), for a block that also needs s_q + s_k. So the primary
// variants stage the score tile tile-locally (stride MLANES = 16), and `m16_bf16wide` prices the
// 512-wide row stride separately to show the drain's slot cost does not depend on which of the two
// strides is used — the *LDS* cost does, and that is the term that bites (see the brief's step 4).
//
// Per stage round, per warp (one head): 16 rows x 16 KV lanes = **256 touches**, reached as sixteen
// 16-deep K-steps (K = DH = 256) into one accumulator, then ONE drain of the 16x16 f32 D tile into the
// score staging. The denominator every figure in `m16_census.py` prints is therefore 256.
//
// THE VARIANTS (each differenced against `m16_empty`, which carries the shared prologue/epilogue)
// -----------------------------------------------------------------------------------------
//   m16_empty     staging + both barriers + tail sink, no score work at all. The baseline.
//   m16_none      feed + MMA, drained by a 1-add-per-lane sink. Isolates the feed from any drain.
//   m16_f32row    feed + `store_matrix_sync` of the f32 D tile into an f32 score row (ldm = 16).
//                 The drain `results.md` §3 implicitly priced (its `s_sc` was `float`), modulo stride.
//   m16_bf16rt    feed + a LEGAL path to the shipped BF16 round: store the D tile map-aware into a
//                 tile-local f32 scratch, barrier, then convert with the shipped round and store BF16.
//                 No fragment-map assumption: every element's (row, col) comes from *this file's* own
//                 scratch layout, not from the hardware's accumulator map.
//   m16_bf16wide  identical to `m16_bf16rt` but with the shipped 512-element row stride. Stride probe.
//   m16_bf16frag  the FLOOR: raw `__builtin_amdgcn_wmma_*`, hand-fed operands, D registers drained
//                 straight to BF16. Legal only if the hardware's fragment map is what this file
//                 assumes — `prototype/issue_rate_probe` §5.1 forbids assuming that on a shipping
//                 path, so this leg is a bound, labelled as one, like `wm_raw_floor*` in
//                 `wmma_feeder_probe.cpp`.
//   m16_roll      the rolled feed (UN = 1) with V3's drain: the VGPR leg.
//   m16_bf16gq    V3's drain with the Q operand read from GLOBAL instead of an 8 KiB LDS stage: the
//                 LDS leg, because LDS per CU is what binds this kernel, not issue slots.
//   m16_scalar    CONTROL, no WMMA at all: the same 16x16 block geometry scored the shipped way (the
//                 slim dot + the slim fold + the same round and store). It separates "what the
//                 instruction class costs" from "what the re-tile costs", which is the pair step 4 of
//                 the brief needs and `prototype/attn_m2` could only price on a device.
//   m16_scalar_ob CONTROL with every lane's folded result stored, not just lane 0's: the witness that
//                 the fold is really being paid, priced beside `m16_scalar` rather than argued.
//
// WHAT IS *NOT* HERE, so nobody double-books it: pass 2 (softmax + PV) is not re-priced. The 45-slot
// denominator this file compares against is a PASS-1 touch (`prototype/attn_valu_slim/results.md` §2),
// and the PV leg keeps `results.md` §3's measured 53.6 slots/tile = 3.35 slots/touch.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#define DH 256     // engine: BF16 per K/V row -> LDS row stride 512 B
#define MAXSC 512  // engine: BF16 per score row -> LDS row stride 1024 B
#define SCALE 0.0625f  // engine/attn_decode/attn_decode_kernels.h:34 — 1/sqrt(256)
// The engine's row stride for ONE head of a query row is ROW_H = N_Q * DH = 8 * 256 = 2,048 elements
// (4 KiB), NOT DH — `a.qrope + ROW_H * row + head * DH`. `m16_bf16gq` reads A straight from global, so
// it must price the address math of the real layout; using DH here would understate it and imply the 16
// rows were contiguous, which they are not.
#define GQ_LDM 2048

#define MROWS 16   // query rows per block. The whole point: >= 16, or a 16x16 operand cannot be built.
#define MLANES 16  // KV lanes per sub-block. >= 16 because a WMMA B tile is 16 wide; 8 (the shipped
                   // ATTN_TILE_L) would split one tile across two stage rounds.
#define NWARPS 4   // ATTN_TILE_G: one warp per query head of the GQA group, exactly as shipped.
#define THREADS (NWARPS * 32)

// One (row, lane) pair of one head = 256 MACs = one shipped touch. Per warp per stage round:
constexpr int kTouchesPerRound = MROWS * MLANES;   // = 256
constexpr int kKSteps = DH / 16;                  // = 16 tiles of 16 dims into one accumulator

static_assert(MROWS == 16 && MLANES == 16,
              "the falsifier prices the M = 16 tile the gate is about; another tile shape is another "
              "question and needs its own denominator, not a reused 256");
static_assert(NWARPS == 4, "one warp per GQA head, as shipped (ATTN_TILE_G == GQA == 4)");

namespace {

// The engine's own round-trip spellings (`attn_decode_kernels.h:53-54`), so `score_round` below is the
// same expression the shipped `publish` writes.
__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }

// HIP takes the shuffle mask as 64-bit (`attn_decode_kernels.h:22`). The literal matters: the slim
// fold is cheap only because LLVM can prove this mask is the full exec mask.
constexpr unsigned long long kFullMask = 0xffffffffffffffffull;

using AccFrag = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
using AFrag = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t,
                                rocwmma::row_major>;
using BFrag = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t,
                                rocwmma::col_major>;

// The raw builtin's operand types, same as `wmma_feeder_probe.cpp`: A/B are 16 BF16 (32 B) per lane,
// C/D are 8 f32 per lane, taken from rocwmma's own call site in
// /opt/rocm/include/rocwmma/internal/wmma_impl.hpp.
using ARegs = short __attribute__((ext_vector_type(16)));
using CRegs = float __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------------------
// The shipped staging, at the M = 16 shape: one row per warp, one `uint4` per lane, row stride DH —
// the store `attn_slim_kernels.h` issues for a K sub-block
// (`*reinterpret_cast<uint4*>(&s_k[slot*DH + lane*8]) = *...`). Used for both the K sub-block and the
// Q rows, because at M = 16 the query block cannot live in VGPRs and must be staged like a K row.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_rows(hip_bfloat16* dst, const hip_bfloat16* src,
                                           float* sink) {
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
#pragma unroll
    for (int r = w; r < MROWS; r += NWARPS) {
        *reinterpret_cast<uint4*>(&dst[r * DH + lane * 8]) =
            *reinterpret_cast<const uint4*>(&src[(size_t)r * DH + lane * 8]);
    }
    // Keep the staging live: LLVM deletes an LDS object it cannot observe being read (the positive
    // control in `tools/kernel_occupancy_probe.cpp` caught exactly this).
    if (sink != nullptr && *reinterpret_cast<const unsigned short*>(&dst[lane]) == 0x1u) *sink = 1.0f;
}

// ---------------------------------------------------------------------------
// The score tile: A = the M = 16 Q block (row-major, ldm = DH), B = the 16-lane K sub-block read as
// K^T (col_major, ldm = DH: B[d][n] = s_k[n*DH + d]), K = DH = 256 = sixteen 16-deep steps into ONE
// accumulator. Same call shape, same ldm, same LDS layouts as `wmma_feeder_probe.cpp`'s
// `wm_score_k256`, so its feed term must reproduce that file's 13.5 slots/tile — one of the two
// calibration figures `m16_census.py` checks.
// `UN` is the unroll factor of the K loop, and it is a MEASURED knob, not a style choice: fully
// unrolled (UN = 16) the scheduler hoists the operand pairs and the block needs >200 VGPR/lane, which
// on its own costs more occupancy than the instruction cut buys (see the VGPR column of
// `m16_census.py` §1, and the `m16_roll` leg). Rolled (UN = 1) the operand registers are reused.
// ---------------------------------------------------------------------------
template <int UN>
__device__ __forceinline__ void feed_mma(const hip_bfloat16* s_q, const hip_bfloat16* s_k,
                                         AccFrag& fd) {
    rocwmma::fill_fragment(fd, 0.0f);
#pragma unroll UN
    for (int d0 = 0; d0 < DH; d0 += 16) {
        AFrag fa;
        BFrag fb2;
        rocwmma::load_matrix_sync(fa, &s_q[d0], DH);
        rocwmma::load_matrix_sync(fb2, &s_k[d0], DH);
        rocwmma::mma_sync(fd, fa, fb2, fd);       // accumulate in place: one D tile, 16 K-steps
    }
}

// The shipped score round, verbatim (`attn_slim_kernels.h`'s `publish`): one BF16 round, the scale,
// another round — exact in BF16 because SCALE is 2^-4, which is what lets `s_sc` stay 2 bytes and
// still read back the identical F32. Association is load-bearing (the q-clamp trap: a hoisted scalar
// constant is a different number), so this keeps the shipped order.
__device__ __forceinline__ hip_bfloat16 score_round(float dot) {
    return fb(bf(fb(dot)) * SCALE);
}

// --- the shipped VALU touch, transcribed from `engine/attn_slim_kernels.h` (levers 1 and 2) and used
// only by the `m16_scalar` control, so that control bills the 45-slot touch and not the pre-slim 116.
__device__ __forceinline__ float slim_dot(const hip_bfloat16* k, const float* q) {
    const uint4 ku = *reinterpret_cast<const uint4*>(k);
    const hip_bfloat16* kk = reinterpret_cast<const hip_bfloat16*>(&ku);
    float dot = 0.f;
#pragma unroll
    for (int t = 0; t < 8; ++t) dot += bf(kk[t]) * q[t];   // serial in t, §05.5
    return dot;
}

__device__ __forceinline__ float slim_permute_add(float v, unsigned addr) {
    unsigned got;
    const unsigned bits = __builtin_bit_cast(unsigned, v);
    asm volatile("ds_bpermute_b32 %0, %1, %2\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(got)
                 : "v"(addr), "v"(bits));
    return v + __builtin_bit_cast(float, got);
}

__device__ __forceinline__ unsigned slim_perm_addr(int lane, int delta) {
    const int src = (lane + delta < warpSize) ? lane + delta : lane;
    return 4u * (unsigned)src;
}

__device__ __forceinline__ float slim_fold(float v, const unsigned* perm_addr) {
    v = v + __shfl_down_sync(kFullMask, v, 16);
    for (int i = 0; i < 4; ++i) v = slim_permute_add(v, perm_addr[i]);   // deltas 8, 4, 2, 1
    return v;
}

// ---------------------------------------------------------------------------
// The CONTROL's touch loop, spelled ONCE for both control variants so the only thing that can differ
// between them is whether every lane's folded result is observable. `OBS = false` is the shipped shape
// (only lane 0's folded value is published into `s_sc`); `OBS = true` additionally accumulates EACH
// lane's own folded value and stores it unconditionally per lane at the end.
// Why the second spelling exists: with only lane 0's store, "the compiler could not have dropped fold
// work, because every fold step feeds lane 0's value" is an ARGUMENT, and a control that has to be
// argued for is not a control (the `mixed_bw` lesson: a check verified against a buffer the writes never
// store proves nothing). With `OBS = true` the witness is a byte in memory.
// The cost of that witness is billed and reported (`CTRL_OB - CONTROL` in `m16_census.py`), because it
// is one `v_add` per touch plus a tail store, and a control that is quietly dearer than the thing it
// witnesses is the mirror image of the error it guards against.
// ---------------------------------------------------------------------------
template <bool OBS>
__device__ __forceinline__ void control_touches(const hip_bfloat16* s_q, const hip_bfloat16* s_k,
                                               hip_bfloat16* s_sc, float* sink) {
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    unsigned perm_addr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) perm_addr[i] = slim_perm_addr(lane, 8 >> i);   // 8, 4, 2, 1

    float keep = 0.f;                 // this lane's own folded results, when OBS
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
        float qq[8];
        const uint4 qu = *reinterpret_cast<const uint4*>(&s_q[m * DH + lane * 8]);
        const hip_bfloat16* qp = reinterpret_cast<const hip_bfloat16*>(&qu);
#pragma unroll
        for (int t = 0; t < 8; ++t) qq[t] = bf(qp[t]);
#pragma unroll
        for (int n = 0; n < MLANES; ++n) {
            float dot = slim_dot(&s_k[n * DH + lane * 8], qq);
            dot = slim_fold(dot, perm_addr);
            if (OBS) keep += dot;     // every lane's fold result is live, not just lane 0's
            if (lane == 0) s_sc[w * MROWS * MLANES + m * MLANES + n] = score_round(dot);
        }
    }
    if (OBS && sink != nullptr) sink[lane + w * 32] = keep;   // unconditional per-lane store
}

// The raw-builtin operand feed, same hypothesised map as `wmma_feeder_probe.cpp`'s `pack_two_b128`:
// lane owns a contiguous 16-element run. A PRICE FLOOR, not a proven-correct feeder.
__device__ __forceinline__ ARegs pack_two_b128(const hip_bfloat16* p) {
    union { uint4 u[2]; ARegs s; } v;
    v.u[0] = *reinterpret_cast<const uint4*>(p);
    v.u[1] = *reinterpret_cast<const uint4*>(p + 8);
    return v.s;
}

// Shared prologue: Q rows + one 16-lane K sub-block staged as the engine stages them, then the
// shipped "K sub-block published" barrier. Every variant starts here.
__device__ __forceinline__ void prologue(const hip_bfloat16* q, const hip_bfloat16* k,
                                         hip_bfloat16* s_q, hip_bfloat16* s_k, float* sink) {
    stage_rows(s_q, q, sink);
    stage_rows(s_k, k, sink);
    __syncthreads();          // K sub-block published — barrier 1 of the shipped pair
}

// Shared epilogue: the shipped "s_sc published" barrier plus a tail read that keeps the staging live.
// `warp_stride` is the per-head block of the score staging (0 for the single-head stride probe V3b).
__device__ __forceinline__ void epilogue(const hip_bfloat16* s_sc, int scld, int warp_stride,
                                        float* sink) {
    __syncthreads();          // s_sc published — barrier 2 of the shipped pair
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    if (sink != nullptr)
        sink[lane] = (float)s_sc[w * warp_stride + (lane & 15) * scld + lane] +
                     (float)s_sc[w * warp_stride + (lane & 15) * scld + (lane & 15)];
}

}  // namespace

// ===========================================================================
// V0 — baseline: the block shape with no score work at all. Everything the census bills as "marginal"
// is measured against this, so prologue/epilogue/staging cancel by construction.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_empty(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
          float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    prologue(q, k, s_q, s_k, sink);
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
}

// ===========================================================================
// V1 — feed + MMA only. The D tile is consumed by one add-reduce and one 32-bit store per lane, i.e.
// ~9 slots of mandated sink for all 256 touches; anything above `m16_empty` + that sink is the feed.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_none(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
         float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    __shared__ float s_liv[32];
    prologue(q, k, s_q, s_k, sink);

    AccFrag fd;
    feed_mma<16>(s_q, s_k, fd);
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc += fd.x[i];
    s_liv[lane] = acc;                       // keep the MMA live
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
    if (sink != nullptr) sink[lane] += s_liv[lane];
}

// ===========================================================================
// V2 — the drain `results.md` §3 implicitly priced: `store_matrix_sync` of the f32 D tile into a score
// row of the stage. NOTE the destination is F32, not the shipped BF16, so it is not "the same `s_sc`
// round" — V3 is. Kept because §3's 13.5 slots/tile was measured against such a destination and the
// two must stay comparable.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_f32row(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
           float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ float s_scF[NWARPS * MROWS * MLANES];   // f32 score row
    prologue(q, k, s_q, s_k, sink);

    AccFrag fd;
    feed_mma<16>(s_q, s_k, fd);
    const int w = (int)(threadIdx.x >> 5);
    rocwmma::store_matrix_sync(&s_scF[w * MROWS * MLANES], fd, MLANES, rocwmma::mem_row_major);
    epilogue(reinterpret_cast<hip_bfloat16*>(s_scF), MLANES, MROWS * MLANES, sink);
}

// ===========================================================================
// V3 — the LEGAL path to the shipped BF16 round, primary answer of this probe.
//   1. `store_matrix_sync` the f32 D tile into a tile-local 16x16 f32 scratch (ldm = 16). Map-aware;
//      no assumption about which lane holds which element.
//   2. one barrier — this is the redistribution the gate is asking about, and it is priced;
//   3. lane l reads 8 CONTIGUOUS f32 out of the scratch — which by this file's own scratch layout are
//      row `l >> 1`, columns `(l & 1) * 8 .. +7` — applies the shipped round, and writes 8 contiguous
//      BF16 (2 x `ds_load_b128` in, 1 x `ds_store_b128` out).
// The (row, col) map here is the SCRATCH's, chosen by this file, not the hardware accumulator's, so
// unlike V4 this kernel is legal without proving the fragment map.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_bf16rt(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
           float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    __shared__ float s_tile[16 * 16];
    prologue(q, k, s_q, s_k, sink);

    const int w = (int)(threadIdx.x >> 5);
    AccFrag fd;
    feed_mma<16>(s_q, s_k, fd);
    rocwmma::store_matrix_sync(&s_tile[0], fd, 16, rocwmma::mem_row_major);
    __syncthreads();                               // the drain's own barrier: part of the price

    const int lane = threadIdx.x & 31;
    const int row = lane >> 1;
    const int c0 = (lane & 1) * 8;
    float got[8];
    *reinterpret_cast<float4*>(got) = *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0]);
    *reinterpret_cast<float4*>(got + 4) =
        *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0 + 4]);
    hip_bfloat16 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = score_round(got[i]);
    *reinterpret_cast<uint4*>(&s_sc[w * MROWS * MLANES + row * MLANES + c0]) =
        *reinterpret_cast<const uint4*>(out);
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
}

// ===========================================================================
// V3b — identical drain at the SHIPPED 512-element row stride. If V3b and V3 census the same, the
// drain's slot cost is stride-insensitive and the whole-row-vs-tile-local choice is purely an LDS/
// occupancy question (it is: see the header). Sized for ONE head only: four heads' whole rows are
// 65,536 B on top of s_q + s_k + s_tile, which the compiler refuses outright — that refusal is the
// witness `wmma_m16_lds_refusal.cpp` + `check_m16_lds_refusal.sh` assert on, and the reason the primary
// variants stage the score tile tile-locally.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_bf16wide(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
             float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[MROWS * MAXSC];           // shipped stride, whole row, ONE head
    __shared__ float s_tile[16 * 16];
    prologue(q, k, s_q, s_k, sink);

    AccFrag fd;
    feed_mma<16>(s_q, s_k, fd);
    rocwmma::store_matrix_sync(&s_tile[0], fd, 16, rocwmma::mem_row_major);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = lane >> 1;
    const int c0 = (lane & 1) * 8;
    float got[8];
    *reinterpret_cast<float4*>(got) = *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0]);
    *reinterpret_cast<float4*>(got + 4) =
        *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0 + 4]);
    hip_bfloat16 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = score_round(got[i]);
    *reinterpret_cast<uint4*>(&s_sc[row * MAXSC + c0]) =
        *reinterpret_cast<const uint4*>(out);
    epilogue(s_sc, MAXSC, 0, sink);
}

// ===========================================================================
// V3c — the rolled feed (UN = 1) with the SAME drain as V3. This is the occupancy leg: the unroll
// factor is not a style choice on this machine, because the fully unrolled feed needs >200 VGPR/lane
// and the shipped kernel has 47. If `m16_roll` costs more slots than `m16_bf16rt` it is because the
// rolled loop pays per-K-step control it did not have to; if it costs about the same but far fewer
// VGPRs, the unroll is the thing that has to go first.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_roll(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
         float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    __shared__ float s_tile[16 * 16];
    prologue(q, k, s_q, s_k, sink);

    const int w = (int)(threadIdx.x >> 5);
    AccFrag fd;
    feed_mma<1>(s_q, s_k, fd);
    rocwmma::store_matrix_sync(&s_tile[0], fd, 16, rocwmma::mem_row_major);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = lane >> 1;
    const int c0 = (lane & 1) * 8;
    float got[8];
    *reinterpret_cast<float4*>(got) = *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0]);
    *reinterpret_cast<float4*>(got + 4) =
        *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0 + 4]);
    hip_bfloat16 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = score_round(got[i]);
    *reinterpret_cast<uint4*>(&s_sc[w * MROWS * MLANES + row * MLANES + c0]) =
        *reinterpret_cast<const uint4*>(out);
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
}

// ===========================================================================
// V4 — the FLOOR: raw builtin, hand-fed operands, D registers drained straight to BF16. Zero
// redistribution IF the hardware map is "lane owns a contiguous run of its row"; that is the
// assumption `prototype/issue_rate_probe` §5.1 refuses to allow a shipping kernel, so this leg is a
// bound on V3, not a shippable spelling.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_bf16frag(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
             float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    prologue(q, k, s_q, s_k, sink);

    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    const int row = lane & 15;
    CRegs acc{};
#pragma unroll
    for (int d0 = 0; d0 < DH; d0 += 16) {
        const ARegs a = pack_two_b128(&s_q[row * DH + d0]);
        const ARegs bb = pack_two_b128(&s_k[row * DH + d0]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc);
    }
    union { CRegs v; float f[8]; } in;
    in.v = acc;
    hip_bfloat16 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = score_round(in.f[i]);
    *reinterpret_cast<uint4*>(&s_sc[w * MROWS * MLANES + row * MLANES]) =
        *reinterpret_cast<const uint4*>(out);
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
}

// ===========================================================================
// V4b — the Q stage is the largest single LDS object an M = 16 block carries (16 rows x DH = 8 KiB for
// ONE head, and a GQA block that keeps four heads in one block needs 32 KiB). Since LDS per CU, not
// issue slots, is what binds this kernel (§26: 5 blocks/CU at 12,288 B), this variant drops the Q
// stage and feeds A straight from global, which is where `a.qrope` lives anyway. If it censuses at the
// same slots/touch as `m16_bf16rt`, the 8 KiB stage is optional and the occupancy story is salvageable;
// if it does not, the Q stage is mandatory and LDS is the wall.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_bf16gq(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
           float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    __shared__ float s_tile[16 * 16];
    stage_rows(s_k, k, sink);
    __syncthreads();                               // barrier 1 of the shipped pair

    const int w = (int)(threadIdx.x >> 5);
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fd;
    rocwmma::fill_fragment(fd, 0.0f);
#pragma unroll 1
    for (int d0 = 0; d0 < DH; d0 += 16) {
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t,
                          rocwmma::row_major> fa;
        BFrag fb;
        // A from GLOBAL, row-major with the engine's real row stride GQ_LDM (= ROW_H, 4 KiB) — the
        // `qrope` layout for this block's 16 rows of this warp's head. No LDS Q stage exists here.
        rocwmma::load_matrix_sync(fa, q + (size_t)w * MROWS * GQ_LDM + d0, GQ_LDM);
        rocwmma::load_matrix_sync(fb, &s_k[d0], DH);
        rocwmma::mma_sync(fd, fa, fb, fd);
    }
    rocwmma::store_matrix_sync(&s_tile[0], fd, 16, rocwmma::mem_row_major);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = lane >> 1;
    const int c0 = (lane & 1) * 8;
    float got[8];
    *reinterpret_cast<float4*>(got) = *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0]);
    *reinterpret_cast<float4*>(got + 4) =
        *reinterpret_cast<const float4*>(&s_tile[row * 16 + c0 + 4]);
    hip_bfloat16 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = score_round(got[i]);
    *reinterpret_cast<uint4*>(&s_sc[w * MROWS * MLANES + row * MLANES + c0]) =
        *reinterpret_cast<const uint4*>(out);
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
}

// ===========================================================================
// V5b — CONTROL, but with EVERY lane's folded result made observable.
//   Same touch, same fold, same geometry as `m16_scalar`; the only difference is that each lane
//   accumulates ITS OWN folded value into a per-lane register and stores it unconditionally into a real
//   output buffer at the end, instead of only lane 0's value reaching `s_sc`. The reason it exists is a
//   review finding: a control whose only observable result is lane 0's leaves "did the compiler keep the
//   whole fold?" answered by an argument rather than by a witness (the `mixed_bw` lesson — a check
//   verified against a buffer the writes never store proves nothing). `m16_census.py` can now adjudicate
//   it two ways: the lane-movement witness on `m16_scalar` (640 permutes over the 128 touches its static
//   body covers = 5.00/touch, the shipped shape) and the price difference between the two controls,
//   which is the price of making all 32 lanes observable (expected: one v_add per touch plus a tail
//   store). If they census the same, the DCE worry is dead in both directions; if they do not, the
//   difference is the size of what the argument was hiding and the observable control is the one in the
//   verdict.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_scalar_ob(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
              float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    prologue(q, k, s_q, s_k, sink);

    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    unsigned perm_addr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) perm_addr[i] = slim_perm_addr(lane, 8 >> i);   // 8, 4, 2, 1

    float keep = 0.f;                       // THIS lane's running sum of its own folded results
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
        float qq[8];
        const uint4 qu = *reinterpret_cast<const uint4*>(&s_q[m * DH + lane * 8]);
        const hip_bfloat16* qp = reinterpret_cast<const hip_bfloat16*>(&qu);
#pragma unroll
        for (int t = 0; t < 8; ++t) qq[t] = bf(qp[t]);
#pragma unroll
        for (int n = 0; n < MLANES; ++n) {
            float dot = slim_dot(&s_k[n * DH + lane * 8], qq);
            dot = slim_fold(dot, perm_addr);
            keep += dot;                    // every lane's value is live, not just lane 0's
            if (lane == 0) s_sc[w * MROWS * MLANES + m * MLANES + n] = score_round(dot);
        }
    }
    if (sink != nullptr) sink[lane + w * 32] = keep;   // unconditional per-lane store
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
}

// ===========================================================================
// V5 — CONTROL, no WMMA: the same block geometry (M = 16 rows x 16 KV lanes per warp per round, Q
// staged in LDS because 16 rows do not fit in VGPRs) scored with the SHIPPED spellings — the slim dot
// (one `ds_load_b128` of K + 8 unrolled widen+FMA pairs) and the slim fold (one `ds_swizzle` + four
// hand `ds_bpermute`s with hoisted addresses) — and the same round/store. Its slots/touch is what the
// re-tile costs on the VALU path, which is how step 4 separates the class cut from the geometry tax
// without a device run.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
m16_scalar(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
           float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[MROWS * DH];
    __shared__ hip_bfloat16 s_k[MLANES * DH];
    __shared__ hip_bfloat16 s_sc[NWARPS * MROWS * MLANES];
    prologue(q, k, s_q, s_k, sink);

    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    unsigned perm_addr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) perm_addr[i] = slim_perm_addr(lane, 8 >> i);   // 8, 4, 2, 1

#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
        float qq[8];
        const uint4 qu = *reinterpret_cast<const uint4*>(&s_q[m * DH + lane * 8]);
        const hip_bfloat16* qp = reinterpret_cast<const hip_bfloat16*>(&qu);
#pragma unroll
        for (int t = 0; t < 8; ++t) qq[t] = bf(qp[t]);
#pragma unroll
        for (int n = 0; n < MLANES; ++n) {
            float dot = slim_dot(&s_k[n * DH + lane * 8], qq);
            dot = slim_fold(dot, perm_addr);
            if (lane == 0) s_sc[w * MROWS * MLANES + m * MLANES + n] = score_round(dot);
        }
    }
    epilogue(s_sc, MLANES, MROWS * MLANES, sink);
}
