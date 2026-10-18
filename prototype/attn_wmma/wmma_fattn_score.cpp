// wmma_fattn_score.cpp — D62's falsifiable form for prefill queue item 2.6, COMPILE-ONLY.
//
// WHAT IS BEING PRICED, AND BY WHOM
// `progress/2026-10-18/llama-fattn-anatomy.md` §4.2 funds one probe: a single WMMA round that
// replaces BOTH shipped attention passes at <= 20 pipe-weighted slots/touch and >= 4 blocks/CU. The
// shipping shape (engine/attn_slim_kernels.h + engine/kernels_attn.cpp) is two VALU passes: pass 1
// computes one score per (Q row, KV lane) with a serial 256-term dot and publishes it to `s_sc` as
// BF16; pass 2 re-reads that column, folds max/sum with a 5-step cross-lane butterfly, and does P*V
// as another serial dot per output element. Measured whole-kernel cost: 89.9 warp-inst/touch
// (progress/2026-10-16/prefill-occupancy-results-2026-10-16.md §3).
//
// The D58 best form in `wmma_m16_probe.cpp` (`m16_bf16gq`, 1.98 slots/touch, 5 blocks/CU) priced the
// matrix part but did two things this form may not: it ASSUMED the fragment maps (its header says
// "same hypothesised map", and its C drain reads a contiguous 8-run, which is not what the RDNA3 C
// tile gives), and it copied the shipped 5-step butterfly into the WMMA shape instead of using the
// fold the fattn shape allows. This TU changes exactly three things relative to that round and
// nothing else (§4.2):
//   1. score tile = `v_wmma_f32_16x16x16_f16` with the llama.cpp fragment maps TRANSCRIBED from
//      `engines/llama.cpp/ggml/src/ggml-cuda/mma.cuh` (table below), so the drain reads the
//      parity-split column layout the hardware actually hands back;
//   2. fold = ONE cross-lane op per Q row per tile, the `offset_first == offset_last == 16` shape at
//      `fattn-mma-f16.cuh:851-872`, replacing the shipped 5-step butterfly (and the rowsum's combine
//      sits outside the round, as it does at `fattn-mma-f16.cuh:1421`);
//   3. P*V = 16 independent F32 accumulator tiles over DH = 256 (>= the 8-tile `VKQ_C` shape at
//      `fattn-mma-f16.cuh:1013-1041`), with P converted from the C tile to 16-bit by the transcribed
//      `get_half2` exchange (`mma.cuh:729-743`, used at `fattn-mma-f16.cuh:968-977`).
// Everything else is kept from the shipped shape on purpose: NWARPS = 4 warps/block, one warp per
// head of the GQA group; the score round `fb(bf(fb(dot)) * SCALE)` (`engine/attn_decode/attn_decode
// .h:53-54` via `wmma_m16_probe.cpp:178`) so the ONLY numerics difference this form can introduce is
// the dot's summation order plus 16-bit operands; DH = 256; 16 Q rows x 16 KV lanes per warp-round =
// 256 touches, the same denominator as `m16_census.py:98`.
//
// THE TRANSCRIBED MAP (RDNA3 / AMD_WMMA; each row is a transcription, not an inference)
// `mma.cuh:183`   tile<I,J,T>::ne       = I*J/32 for a warp of 32
// `mma.cuh:195`   tile<I,J,T>::get_i(l) = threadIdx.x % 16
// `mma.cuh:207`   float tile, get_j(l)  = 2*l + threadIdx.x/16
//   => C tile `tile<16,16,float>` (mma.cuh:1078 is exactly this typedef for RDNA3): 8 f32 per lane,
//      lane t owns ROW t%16 at the 8 columns {2l + t/16}, i.e. one PARITY of one row, shared with
//      lane t^16 which owns the other parity. A drain must therefore never assume a contiguous 8-run;
//      that assumption is what disqualifies `m16_bf16frag`.
// `mma.cuh:535,577` tile<I,J,half2,MIRRORED>::ne = I*J/32*2 half2, and it delegates to the float tile
// `mma.cuh:552,560` get_i = threadIdx.x % 16, get_j(l) = l
//   => A/B operand tile `tile<16,8,half2>` (mma.cuh:1077 for RDNA3): 8 half2 = 16 CONTIGUOUS halves
//      of row t%16, held identically by both half-warps. This is the map `pack_two_b128` assumed, and
//      the half2 payload is 8 VGPRs either way). THIS form emits `v_wmma_f32_16x16x16_f16 v[1:8],
//      v[15:22], v[23:30], v[1:8]` in its score loop (8-GPR A, 8-GPR B, 8-f32 C in and out), which is the
//      operand arithmetic this map implies; the sibling probe's bf16 line
//      `v_wmma_f32_16x16x16_bf16 v[1:8], v[10:17], v[18:25], v[1:8]` is the same shape.
// `mma.cuh:729-743` get_half2(C) -> B operand: per element l, tmp[t/16] = own, tmp[(t/16)^1] =
//      __shfl_xor_sync(mask, own, 16), then make_half2(tmp[0], tmp[1]). Applied to the C map above
//      this is exactly "collect the partner half of row t%16", and its output shape is exactly the A/B
//      operand shape, which is the cross-check that the two transcriptions agree.
// CORRECTION, 2026-10-18, in place: the C map above is TRANSPOSED for this file's original operand
// order, and the shape cross-check named in the last line above could not catch it — two transposed
// layouts have identical register shapes. The measured law is in `score_leg` below and in
// `prototype/attn_wmma/fattn_map_results.md`; the transcription is kept verbatim above because deleting
// it would hide how the error was made, and because the A/B operand half of it (16 contiguous halves of
// row t%16, k-run contiguous) IS confirmed by the same probe. What remains un-witnessed by that probe —
// the get_half2 C->B relabelling itself — is now witnessed separately and executably by
// `prototype/attn_wmma/fattn_cb_probe.cpp` + `check_fattn_cb.py`, which composes score -> P -> P*V on
// encodable inputs and fails on five injections in this exact function.
// CONSEQUENCES FOR THE STAGING, all derived from the map above and not from taste:
//   - score A operand Q[i=q][k=d]: lane t needs Q[t%16][d0..d0+15]  -> s_q row-major, 2x ds_b128.
//   - score B operand K[k=d][j=kv]: lane t needs K[t%16][d0..d0+15]  -> s_k row-major, 2x ds_b128.
//   - P*V B operand P[k=kv][j=q]:   lane t needs P[t%16][kv 0..15]   -> the get_half2 output, free.
//   - P*V A operand V[i=d][k=kv]:   lane t needs V[kv 0..15][t%16]   -> V MUST BE STAGED TRANSPOSED.
//     (llama.cpp does the transposing load in SRAM instead, `load_ldmatrix_trans_swizzled` at
//     `fattn-mma-f16.cuh:1030`; at DH = 256 that spelling costs 16 narrow LDS reads per lane per
//     tile = 256 per round, against 32 narrow LDS stores per lane per round here. Chosen for the
//     instruction count, and the trade is named in the open risks.)
// ONE ADDRESS EXPRESSION PER STAGED OBJECT, writer and reader alike
// `prototype/attn_decode/results.md` §20.1 is the recorded bug of two self-consistent layouts joined
// by a branch, half of every streaming row coming out as the other group's answer. Every read and
// write of a staged object here goes through `q_addr` / `k_addr` / `v_addr` below, and `s_kv` is ONE
// buffer carrying TWO layouts with a `__syncthreads()` between the last K reader and the first V
// writer — the aliasing llama.cpp does when `nstages == 0` (`fattn-mma-f16.cuh:1252`). The aliasing is
// forced by the LDS budget, see the static_assert at the shared declarations.
//
// WHAT IS *NOT* HERE: pass 2's epilogue (O drain, the /l divide, the pass-2 write-back) is not
// re-priced, per §4.2's "pass 2 is not re-priced". The sink below is the DCE witness the sibling probe
// uses (`m16_none`'s 1-add-per-lane pattern), not a shipping drain, and it is billed in the census.
//
// STATUS: COMPILE-ONLY. Nothing here has been launched or timed. The numerics admission and the burst
// are handed over unrun in `prototype/attn_wmma/fattn_score_handover.md`.
//
// BUILD (compile-only): `bash prototype/attn_wmma/build_fattn_score.sh`
// PRICE IT:               `python3 prototype/attn_wmma/fattn_census.py`

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define DH 256        // engine: head_dim. BF16 in global, so a staged row is 512 B either way.
#define SCALE 0.0625f  // engine/attn_decode/attn_decode_kernels.h:34 — 1/sqrt(256), a power of two
#define MROWS 16      // query rows per block: a 16x16 operand tile cannot be built below this
#define MLANES 16     // KV lanes per round: the C tile is 16 wide (mma.cuh:1078), so 16 or nothing
#define NWARPS 4      // ATTN_TILE_G: one warp per query head of the GQA group, exactly as shipped
#define THREADS (NWARPS * 32)

#define KTILES (DH / 16)   // 16 K-steps of the score leg, one accumulator
#define VTILES (DH / 16)   // 16 P*V output tiles over DH = the ">= 8 chains" bar of §4.2 change (3)

// The price denominator: one warp-round = 16 Q rows x 16 KV lanes x one head = 256 touches, the same
// denominator `m16_census.py:98` uses, so the two censuses are comparable without rescaling.
constexpr int kTouchesPerRound = MROWS * MLANES;
static_assert(VTILES >= 8, "§4.2 change (3): P*V needs >= 8 independent accumulator chains");

// The LDS budget, and why `s_kv` carries two layouts. The KFD says 64 KiB of LDS per real CU
// (/sys/.../nodes/1/properties, lds_size_in_kb 64 — NOT the HIP figure, see AGENTS.md's counter trap).
// >= 4 blocks/CU therefore allows at most 65536/4 = 16,384 B per block, and Q(8 KiB) + K(8 KiB) +
// V(8 KiB) is 24 KiB. So V aliases K, as llama.cpp's `nstages == 0` does. Zero slack: one more byte of
// static LDS drops the form to 3 blocks/CU, and that is a reportable property of the form.
constexpr int kLdsPerBlock = 2 * MROWS * DH * 2;   // = 16,384 B
constexpr int kLdsPerCU = 65536;
static_assert(kLdsPerBlock == 16384, "the 4-blocks/CU budget is exactly two 8 KiB stages");

constexpr unsigned long long kFullMask = 0xffffffffffffffffull;  // AGENTS.md: the mask is 64-bit

// Operand payload types, sized by the transcribed map (a wrong size here is a static_assert, not a
// wrong number: 8 f32 for C, 16 halves for A/B).
using ARegs = short __attribute__((ext_vector_type(16)));
using CRegs = float __attribute__((ext_vector_type(8)));
static_assert(sizeof(CRegs) == 32, "C tile is ne = I*J/32 = 8 f32 per lane (mma.cuh:183)");
static_assert(sizeof(ARegs) == 32, "A/B tile is 8 half2 = 16 halves per lane (mma.cuh:535,577)");

// The engine's own score round, transcribed (`wmma_m16_probe.cpp:178`, from
// `engine/attn_decode/attn_decode_kernels.h:53-54`). Kept verbatim and kept in registers: it is what
// confines this form's numerics difference to the summation order and the 16-bit operands.
namespace {
__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }
__device__ __forceinline__ float score_round(float dot) { return bf(fb(bf(fb(dot)) * SCALE)); }

// FATTN_KQ_MAX_OFFSET, transcribed (fattn-common.cuh:19): a constant added inside the max fold so the
// exp argument stays near zero. It cancels in (s - m), so it is numerically free.
constexpr float kMaxOffset = 3.0f * 0.6931471805599453f;

// ---- one address expression per staged object (see the header note on §20.1) ----
__device__ __forceinline__ int q_addr(int q, int d) { return q * DH + d; }        // [row][dim]
__device__ __forceinline__ int k_addr(int kv, int d) { return kv * DH + d; }      // [kv][dim]
__device__ __forceinline__ int v_addr(int d, int kv) { return d * MLANES + kv; }  // [dim][kv] transposed
static_assert(MLANES * DH == MROWS * DH, "the K and V layouts share one 8 KiB buffer by design");

// ---- the transcribed fragment maps, as functions, so no site can silently disagree ----
// C tile: lane t owns row t%16, column 2l + t/16 (mma.cuh:195,207).
__device__ __forceinline__ int c_row(int lane) { return lane & 15; }
__device__ __forceinline__ int c_col(int lane, int l) { return 2 * l + (lane >> 4); }
// A/B tile: lane t owns 16 contiguous halves of row t%16 (mma.cuh:552,560).
__device__ __forceinline__ int ab_row(int lane) { return lane & 15; }
constexpr int kABHalves = 16;
constexpr int kCElem = 8;

// One 32-byte operand tile: 16 contiguous halves = two 16-B LDS reads.
__device__ __forceinline__ ARegs load_tile(const __half* p) {
    union { uint4 u[2]; ARegs s; } v;
    v.u[0] = *reinterpret_cast<const uint4*>(p);
    v.u[1] = *reinterpret_cast<const uint4*>(p + 8);
    return v.s;
}

// ---------------------------------------------------------------------------
// Staging. Global K/V/Q arrive BF16; a WMMA f16 operand must be F16, so the conversion is paid here,
// once per staged element, which is also where llama.cpp pays it. Row assignment: warp w takes rows
// w, w+NWARPS, ... (16 rows / 4 warps = 4 passes per object).
// ---------------------------------------------------------------------------
// Q: [q][dim], the score leg's A operand. Written through q_addr, read through q_addr.
__device__ __forceinline__ void stage_q_f16(__half* s_q, const hip_bfloat16* src, int w) {
    const int lane = threadIdx.x & 31;
    for (int r = w; r < MROWS; r += NWARPS) {
        const uint4 u = *reinterpret_cast<const uint4*>(src + (size_t)r * DH + lane * 8);
        const hip_bfloat16* b = reinterpret_cast<const hip_bfloat16*>(&u);
        union { __half h[8]; uint4 u; } o;
#pragma unroll
        for (int t = 0; t < 8; ++t) o.h[t] = __float2half_rn(bf(b[t]));
        *reinterpret_cast<uint4*>(&s_q[q_addr(r, lane * 8)]) = o.u;
    }
}

// K: [kv][dim], the score leg's B operand (the operand run is along dim for a B tile, so a row of the
// stage is exactly one operand tile). Written through k_addr, read through k_addr.
__device__ __forceinline__ void stage_k_f16(__half* s_kv, const hip_bfloat16* src, int w) {
    const int lane = threadIdx.x & 31;
    for (int r = w; r < MLANES; r += NWARPS) {
        const uint4 u = *reinterpret_cast<const uint4*>(src + (size_t)r * DH + lane * 8);
        const hip_bfloat16* b = reinterpret_cast<const hip_bfloat16*>(&u);
        union { __half h[8]; uint4 u; } o;
#pragma unroll
        for (int t = 0; t < 8; ++t) o.h[t] = __float2half_rn(bf(b[t]));
        *reinterpret_cast<uint4*>(&s_kv[k_addr(r, lane * 8)]) = o.u;
    }
}

// V is staged transposed because the P*V A operand needs the KV run contiguous (see the map table).
__device__ __forceinline__ void stage_v_transposed_f16(__half* dst, const hip_bfloat16* src, int w) {
    const int lane = threadIdx.x & 31;
    for (int kv = w; kv < MLANES; kv += NWARPS) {
        const uint4 u = *reinterpret_cast<const uint4*>(src + (size_t)kv * DH + lane * 8);
        const hip_bfloat16* b = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d = lane * 8 + t;
            dst[v_addr(d, kv)] = __float2half_rn(bf(b[t]));   // narrow: 8 x ds_store_u16 per row
        }
    }
}

// ---------------------------------------------------------------------------
// The three stages of a round, spelled once and shared by all three kernels so the legs difference.
// Barrier count: 3 per round (Q+K published / K consumed / V published). The shipped pair runs 2.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_qk(__half* s_q, __half* s_kv, const hip_bfloat16* q,
                                         const hip_bfloat16* k) {
    const int w = (int)(threadIdx.x >> 5);
    stage_q_f16(s_q, q, w);
    stage_k_f16(s_kv, k, w);
    __syncthreads();                       // barrier 1: Q and K published
}

__device__ __forceinline__ void stage_v(__half* s_kv, const hip_bfloat16* v) {
    const int w = (int)(threadIdx.x >> 5);
    __syncthreads();                       // barrier 2: every warp is done reading K from s_kv
    stage_v_transposed_f16(s_kv, v, w);    // aliases the K stage; k_addr is dead from here on
    __syncthreads();                       // barrier 3: V published
}

// Score leg: 16 K-steps into ONE f32 accumulator. Rolled (`#pragma unroll 1`) on purpose: D58 found
// the rolled feed is what holds the register count down (`wmma_m16_probe.cpp`'s m16_bf16gq finding),
// and the rolled form is what the engine would ship.
__device__ __forceinline__ CRegs score_leg(const __half* s_q, const __half* s_kv, int lane) {
    CRegs acc{};
#pragma unroll 1
    for (int d0 = 0; d0 < DH; d0 += 16) {
        // CORRECTED 2026-10-18 — the operand SLOTS, and nothing else about this loop moved. The measured
        // C law (`prototype/attn_map_probe.cpp`, `check_fattn_map.py`, written up in
        // `prototype/attn_wmma/fattn_map_results.md`) is
        //     lane t, register l  =  sum_k FIRST[row = 2l + t/16][k] * SECOND[row = t%16][k]
        // so the SECOND operand's row index is the C tile's ROW. Q's rows are the score's rows, so Q
        // belongs in the SECOND slot; the file's original transcription put it first and the drain then
        // read `S[t%16][2l + t/16]`, i.e. the transpose, which folds every row down the wrong axis.
        // Both operands are still "16 contiguous halves of row t%16 with the k-run contiguous", which is
        // why `q_addr`/`k_addr` and the whole staging path are unchanged: a transposed score was never
        // fixable in the addresses, only in which register block is which operand.
        const ARegs a = load_tile(&s_kv[k_addr(ab_row(lane), d0)]);   // FIRST:  rows are KV lanes
        const ARegs b = load_tile(&s_q[q_addr(ab_row(lane), d0)]);    // SECOND: rows are Q rows -> C rows
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    return acc;
}

// Change (2): the fold, transcribed from `fattn-mma-f16.cuh:851-872` with offset_first == offset_last
// == 16, i.e. a serial local max over the lane's own 8 values and then exactly ONE cross-lane op per
// Q row per tile. One 16x16 tile per row per round, so the round's cross-lane max cost is 1 shuffle.
// Returns the row max; `psum` collects the rowsum partial (0 cross-lane ops in the round: the rowsum's
// combine is outside, at `fattn-mma-f16.cuh:1421`), and `pexp` returns the exponentials.
__device__ __forceinline__ float fold_max(const CRegs score, float pexp[kCElem], float* psum) {
    union { CRegs v; float f[kCElem]; } s;
    s.v = score;
    // The round is written IN PLACE into `pexp` rather than into a second live array: the max is taken
    // over the rounded values (which is what pass 2 does, since `s_sc` holds the rounded score), and one
    // live array instead of two is 8 VGPRs — see the header note on the 4-blocks/CU register boundary.
    float m = -__builtin_huge_valf();
#pragma unroll
    for (int l = 0; l < kCElem; ++l) {
        pexp[l] = score_round(s.f[l]);                     // the shipped BF16 score round
        m = fmaxf(m, pexp[l] + kMaxOffset);                // then the max, as pass 2 does it
    }
    m = fmaxf(m, __shfl_xor_sync(kFullMask, m, 16, warpSize));   // ONE cross-lane op per row per tile
    float sum = 0.f;
#pragma unroll
    for (int l = 0; l < kCElem; ++l) {
        pexp[l] = expf(pexp[l] - m);
        sum += pexp[l];
    }
    *psum = sum;
    return m;
}

// Change (3a): P, from the C tile to a 16-bit operand, by the transcribed `get_half2` exchange
// (`mma.cuh:729-743`). 8 cross-lane ops per tile — 8 `shfl_xor(16)` plus the pair packing — which is
// the fragment-relabelling cost the round must pay for keeping P in registers.
__device__ __forceinline__ ARegs p_to_operand(const float pexp[kCElem], int lane) {
    ARegs p;
    const int i = lane >> 4;               // which half of the row pair this lane is the owner of
#pragma unroll
    for (int l = 0; l < kCElem; ++l) {
        float tmp[2];
        tmp[i] = pexp[l];
        tmp[i ^ 1] = __shfl_xor_sync(kFullMask, pexp[l], 16, warpSize);
        p[2 * l + 0] = (short)__half_as_ushort(__float2half_rn(tmp[0]));
        p[2 * l + 1] = (short)__half_as_ushort(__float2half_rn(tmp[1]));
    }
    return p;
}

// Change (3b): P*V as VTILES = 16 independent f32 accumulator chains, FULLY UNROLLED. The roll spelling
// is tried and rejected on evidence: with `#pragma unroll 1` the accumulator array is indexed by the loop
// variable, LLVM puts the 128 accumulator registers in scratch (`private_segment_fixed_size 544 B/thread`,
// `.enable_private_segment 1`, while clang's own remark still reported "VGPRs Spill: 0"), so the chains do
// not exist as registers at all. Unrolled it is 0 B of scratch and the 16 chains are register-resident,
// which is the shape §4.2 change (3) claims. This is the opposite choice from the score leg, where one
// accumulator means the roll costs nothing (D58's `m16_bf16gq` finding).
__device__ __forceinline__ void pv_leg(CRegs out[VTILES], const ARegs p, const __half* s_kv, int lane) {
#pragma unroll
    for (int t = 0; t < VTILES; ++t) {
        // A = V^T tile: row d = 16t + (lane&15), the 16 KV halves contiguous in the transposed stage.
        // The measured C law CONFIRMS this leg's slot assignment rather than leaving it to the
        // transcription: SECOND operand's row index is the C row, and here the SECOND operand is P whose
        // rows are Q rows = the output rows, so output[q][d] lands with row q = t%16 and columns
        // d = 2l + t/16 — the same drain the score tile now uses. The score leg needed the swap because
        // its SECOND operand was the wrong one; this one was already right, for a reason worth naming
        // since it is the only place the original transcription's transpose happened to cancel.
        const ARegs a = load_tile(&s_kv[v_addr(16 * t + ab_row(lane), 0)]);
        out[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, p, out[t]);
    }
}

// The DCE witness. Every accumulator register must reach memory or feed something that does, so each
// is added once and one value per lane is stored. Billed as a sink, not as the shipping drain.
__device__ __forceinline__ void sink(float keep, const CRegs out[VTILES], float* sinkp, int lane, int w) {
#pragma unroll
    for (int t = 0; t < VTILES; ++t) {
#pragma unroll
        for (int l = 0; l < kCElem; ++l) keep += out[t][l];
    }
    if (sinkp != nullptr) sinkp[lane + w * 32] = keep;
}
__device__ __forceinline__ void sink1(float keep, float* sinkp, int lane, int w) {
    if (sinkp != nullptr) sinkp[lane + w * 32] = keep;
}
}  // namespace

// ---------------------------------------------------------------------------
// BASELINE. Same staging, same 3 barriers, same sink, zero matrix work. The legs below are read as
// differences from this kernel, per the §20.5 rule that a traffic claim is a property of the staging
// loops the source contains.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS)
m16_fattn_empty(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
                const hip_bfloat16* __restrict__ v, float* __restrict__ sinkp) {
    __shared__ __align__(16) __half s_q[MROWS * DH];
    __shared__ __align__(16) __half s_kv[MLANES * DH];
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    stage_qk(s_q, s_kv, q, k);
    stage_v(s_kv, v);
    float keep = (float)s_q[q_addr(lane, 0)] + (float)s_kv[v_addr(lane, 0)];
    sink1(keep, sinkp, lane, w);
}

// LEG. Staging + score leg + fold + exp + P, no P*V. Its difference from `m16_fattn_empty` is what the
// matrix-eats-the-dot half of the round costs, including change (2)'s fold deletion.
extern "C" __global__ void __launch_bounds__(THREADS)
m16_fattn_scoreleg(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
                   const hip_bfloat16* __restrict__ v, float* __restrict__ sinkp) {
    __shared__ __align__(16) __half s_q[MROWS * DH];
    __shared__ __align__(16) __half s_kv[MLANES * DH];
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    stage_qk(s_q, s_kv, q, k);
    const CRegs score = score_leg(s_q, s_kv, lane);
    stage_v(s_kv, v);
    float pexp[kCElem], psum;
    const float m = fold_max(score, pexp, &psum);
    const ARegs p = p_to_operand(pexp, lane);
    float keep = m + psum;
#pragma unroll
    // bit witness only: every P register must be observable, this is not a numeric read of the half
    for (int l = 0; l < kCElem; ++l) keep += (float)(int)p[2 * l] + (float)(int)p[2 * l + 1];
    sink1(keep, sinkp, lane, w);
}

// ---------------------------------------------------------------------------
// THE FORM. `m16_fattn_score` = staging + score tile (change 1) + one-shuffle fold (change 2) + 16
// independent f32 P*V chains with the transcribed P conversion (change 3).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS)
m16_fattn_score(const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
                const hip_bfloat16* __restrict__ v, float* __restrict__ sinkp) {
    __shared__ __align__(16) __half s_q[MROWS * DH];
    __shared__ __align__(16) __half s_kv[MLANES * DH];
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);

    stage_qk(s_q, s_kv, q, k);                 // barrier 1
    const CRegs score = score_leg(s_q, s_kv, lane);
    stage_v(s_kv, v);                          // barriers 2, 3 (K stage is dead; V aliases it)

    float pexp[kCElem], psum;
    const float m = fold_max(score, pexp, &psum);      // change (2): 1 cross-lane op per row per tile
    const ARegs p = p_to_operand(pexp, lane);          // change (3a): C -> 16-bit operand

    CRegs out[VTILES];
#pragma unroll
    for (int t = 0; t < VTILES; ++t) out[t] = CRegs{};
    pv_leg(out, p, s_kv, lane);                // change (3b): 16 independent f32 chains

    // The rowsum's cross-lane combine, outside the round (fattn-mma-f16.cuh:1421 shape).
    psum = psum + __shfl_xor_sync(kFullMask, psum, 16, warpSize);
    sink(psum + m, out, sinkp, lane, w);
}
