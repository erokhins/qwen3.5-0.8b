/*
 * attn_fattn_kernels.h — the WMMA attention wave (`v_wmma_f32_16x16x16_f16`), ported from
 * `prototype/attn_wmma/wmma_fattn_score.cpp` into the shipping path behind `--attn-fattn`.
 *
 * WHAT IT IS. One block owns FOUR QUERY HEADS x SIXTEEN QUERY ROWS x a sixteen-lane KV round. The score
 * of a whole 16x16 (row, lane) block is one `v_wmma` per 16-dim step, the online (m, l, acc) fold is one
 * cross-lane max per row per round, and the output accumulator is 16 independent F32 chains over
 * DH = 256. It replaces `attn_slim_wave` for the rows it covers, and it covers exactly the rows the
 * shipped form already STREAMS — see COVERAGE below, which is a decision, not a side effect.
 *
 * THE FRAGMENT MAP THIS CODE IS WRITTEN AGAINST (measured, not transcribed)
 * `prototype/attn_wmma/fattn_map_probe.cpp` + `check_fattn_map.py` measured, on this device, that with both
 * operands staged row-major with the 16-half run along the dot index,
 *
 *     lane t, register l  =  sum_k  FIRST[row = 2l + t/16][k] * SECOND[row = t%16][k]
 *
 * so the SECOND operand's row index is the C tile's ROW. Everything below follows from that one sentence:
 *   * score: FIRST = K (rows are KV lanes), SECOND = Q (rows are query rows) -> C[q][kv] with q = t%16 and
 *     kv = 2l + t/16, i.e. one lane holds one parity of one query row.
 *   * P*V: FIRST = V staged transposed (row = output dim d), SECOND = P (row = query row) -> O[q][d].
 * The prototype's original transcription had Q first, which produced a TRANSPOSED score and folded every
 * row down the wrong axis; `fattn_score_handover.md` carries the correction in place, and
 * `check_fattn_map.py` is a permanent admission case so it cannot come back. `check_fattn_cb.py` is its
 * sibling for the C->B step (drain -> `get_half2` -> operand) and is the witness that the F32 accumulator
 * read back as 16 F16 halves really is the P matrix the next `v_wmma` wants.
 *
 * COVERAGE — WHY ROWS AT OR BELOW MAXSC ARE NOT THIS KERNEL'S (decision (A), 2026-10-18)
 * A WMMA round is streaming by construction: online (m, l, acc), 16 KV lanes at a time, no full-row score
 * staging. The shipped form's §07.3 route for a row of <= MAXSC lanes is the ONE-PASS form, which
 * normalises inside the block, is bit-exact with the pinned dumps, and never calls `attn_combine`. Routing a
 * pinned one-pass row through this kernel would move bits the pins currently judge exactly, so this kernel
 * REFUSES those rows and `engine/kernels_attn.cpp` leaves them on `attn_slim_wave` — and says so in the log
 * (`attn_fattn=1 fattn_rows=n/N`), because a route that changes quietly under a flag is the `--attn-tile`
 * trap wearing a different hat. The refusal is per row and it is a real refusal, not a fallback inside the
 * kernel: a covered row's partial is written, an uncovered row's is not, and `attn_combine` never sees a
 * half-written slot.
 *
 * What that does NOT cover, stated where it can be read: claim (iii)'s family denominator is partly
 * untouched, so the covered share of `attn` ms is printed beside the ms in the burst window rather than
 * being assumed (a 1.5x family result on 70 % coverage is a ~2x result on covered rows, and those are
 * different claims).
 *
 * Q PLACEMENT AND THE LDS BUDGET — one warp per block, Q staged, after two placements failed
 * One warp's M tile is 16 DISTINCT (row, head) pairs, so the Q a warp needs is 16 pairs x 256 dims x 2 B =
 * **8 KiB PER WARP**, not 8 KiB per block. The committed port gave `ATTN_TILE_THREADS` (128 =
 * `ATTN_TILE_G` warps, one per query head of the group) ONE shared 8 KiB `s_q`, so four warps each wrote a
 * different head's rows into one buffer and each then scored its head against a four-head interleaving of
 * it. The prototype never hit that: `prototype/attn_wmma/wmma_fattn_score.cpp` has no head axis — its `q` is
 * one 16-row matrix for the whole block and its output is a DCE sink — so its recorded "Q(8 KiB) + K(8 KiB)
 * = 16,384 B -> 4 blocks/CU" is a fact about a kernel that cannot be routed, and it sat in the handover as
 * if it were a budget. Three placements were considered; two were compiled, and the numbers are in
 * `fattn_ship_results.md` rather than in a comment:
 *
 *   (A) REGISTER-RESIDENT Q, four warps, V un-aliased: Q's 16 operand tiles = 128 VGPR/lane beside the 128
 *       accumulator VGPRs. REJECTED ON MEASUREMENT — 256 live registers is past what this target keeps in
 *       registers at all: `.vgpr_count 264 / .vgpr_spill_count 115 / .private_segment_fixed_size 724`. It
 *       spilled HARDER than the stage it replaced, and scratch traffic lands on the same dependent chain
 *       `bench/performance-status.md` §28 identifies as the mechanism. That is also why llama.cpp's 358
 *       spilled VGPRs are not a number to admire from a distance.
 *   (B) ONE WARP PER BLOCK, Q STAGED, V ALIASED OVER K. **This is what ships.** 32-thread blocks
 *       (`FATTN_THREADS`), and `blockIdx.x` names the query head where the shipped wave names a KV group:
 *         s_q   16 rows x 256 dims x 2 B  = 8192 B   (Q rows, F16: a WMMA operand is F16 when global is BF16)
 *         s_kv  16 lanes x 256 dims x 2 B = 8192 B   (K row-major for the score leg, then V transposed for
 *                                                     P*V — ALIASED — then the F32 O transpose)
 *         total                           = 16384 B  -> 4 blocks/CU on this part's 64 KiB = 4 wf/CU
 *       Two prices, both named rather than buried. `wf_per_cu=4`, and §28's occupancy evidence spans
 *       12->20 wavefronts/CU, so four is BELOW the range in which "occupancy did not matter" was shown; the
 *       banner prints `wf_per_cu` so no burst can read these covered rows against the shipped ~20 without
 *       the annotation. And V aliasing costs a THIRD barrier per round (K published / K consumed / V
 *       published, against the shipped form's 2), which no census has priced — `fattn_ship_census.py`
 *       asserts the count is what the source says it is, and the burst is what will price it.
 *   (C) K and V in separate stages at four warps: 8 KiB x 4 of Q plus 16 KiB = 48 KiB, one block/CU. Not
 *       attempted; same wavefront count as (B) with none of its LDS headroom.
 *
 * CLAIM (i) ARRIVED RED, and that is a result, not an omission: the port as committed (`d7057c4`) compiled
 * to `.vgpr_count 192 / .vgpr_spill_count 242 / .private_segment_fixed_size 332`, so 4 blocks/CU was already
 * false on the shipping path before anything was decided here — from arrays handed to helpers by pointer
 * (`pexp[8]`, `out[16]`) and one dynamically indexed `tmp[i]`, both now gone. The instrument is
 * `prototype/attn_wmma/fattn_ship_census.py`, and it fails on any non-zero spill.
 *
 * There is no score staging, and that part is unchanged: 16 rows x 512 lanes of BF16 scores would be 16 KiB
 * per head at MROWS=16, which does not fit beside the two stages, and it is why the one-pass form cannot be
 * ported here at all — see COVERAGE above.
 *
 * ONE ADDRESS EXPRESSION PER STAGED OBJECT, writer and reader alike (§20.1). `fattn_kv_addr` and
 * `fattn_v_addr` are the only way the two stages are indexed (Q has no stage: `fattn_q_frag` is the only way
 * its operand tiles are produced), and the partial slot expression is the shipped one, verbatim, keyed by
 * the FULL head, so `attn_combine` needs no branch for this form either.
 *
 * COSTS THIS FORM PAYS THAT THE PROTOTYPE CENSUS DID NOT PRICE, named because they will show up in the
 * census of this file and nowhere else:
 *   * Per-row causal/range masking. The shipped form's causal bound is a loop LIMIT (no per-element guard:
 *     `prototype/attn_valu_slim/results.md` §1). Here one round carries 16 rows whose lane ranges differ,
 *     so a cell has to be masked explicitly: two compares per cell (lower/upper bound of this row's slice
 *     within the round). That is per touch, and it is new.
 *   * The per-round rescale of the accumulator: 16 tiles x 8 F32 = 128 F32 multiplies per lane per round.
 *     The prototype had ONE round and therefore priced none of it.
 *   * The O transpose in the epilogue: the accumulator's columns are parity-split across the half-warps
 *     (dim = 16t + 2l + t/16) and a lane holds ONE row of them, while `part` wants that row's whole DH in
 *     blocks of FATTN_TBLK. Staged through the dead V stage in two halves of DH; per half, 16 rows x 128
 *     floats are published by 32 lanes x 8 blocks x 8 dims, which is the thing the first two versions got
 *     wrong — the first by 8 KiB in the wrong direction (a 128-float stage row indexed with `lane*8 + u`,
 *     lane <= 31, writing `part` up to `pp[385]` on a 258-float slot, i.e. trampling the next (row, head)
 *     partial), the second by publishing only 16 of every row's 256 dims and leaving the merge to read
 *     scratch, which is the NaN of §29 of `bench/performance-status.md`. `fattn_transpose_out` is the only
 *     spelling of those addresses, pinned by three `static_assert`s beside them: the widest stage write, the
 *     widest `part` write, and — the one the second version defeated — the COVERAGE of the publish loop.
 *     A bound says where the last index is; only a coverage assert says every index was reached, and the
 *     witness beyond the compile is admission Section 1: a one-pass row sharing a wave with a covered row
 *     must come back bit-identical to `--attn-fattn 0`.
 *   * The rowsum's cross-lane combine. A row's 16 columns live in the lane PAIR (t, t^16), so the max
 *     needs one `shfl_xor(16)` and the rowsum needs one too — deferred outside the round, which is legal
 *     because `rescale` is identical across the pair. `fattn_score_handover.md` §1 counts 10 lane-movement
 *     ops per round as 9 + 1 for exactly this reason; the first port shipped 9 and published half of every
 *     row's denominator. Half a denominator is a near-uniform scale on the row's output, which is the one
  *     row's denominator. Half a denominator is a near-uniform scale on the row's output, which is the one
 *     error a cosine band cannot see at all — hence the magnitude-sensitive checks and the rowsum witness in
 *     `run_fattn_admission.sh`, and `fattn_ship_census.py` asserting the tenth butterfly is present.
 *
 * STATUS: WRITTEN, NOT ADMITTED. Nothing in this file has been numerically admitted on device. The
 * admission sequence is `prototype/attn_wmma/run_fattn_admission.sh` (written, not run) and the timing
 * window is `prototype/attn_wmma/run_fattn_burst_window.sh` (written, not run, main-agent only). Until
 * those are green this path is reachable only by `--attn-fattn 1`, which is not the default anywhere.
 *
 * CENSUS (compile-only, same recipe as `engine/attn_slim_kernels.h`'s header note; the shipped-placement
 * census and its spill gate live in `prototype/attn_wmma/fattn_ship_census.py`, which re-derives the
 * expected instruction counts from THIS file's geometry rather than carrying the probe's):
 *     REPO=$PWD; mkdir -p /tmp/fattn_census && cd /tmp/fattn_census
 *     hipcc -O3 -Wno-unused-command-line-argument -I"$REPO" -I"$REPO/engine" --save-temps \
 *         -c "$REPO/engine/kernels_attn.cpp" -o k.o; cd "$REPO"
 *     python3 prototype/attn_valu_slim/isa_census.py \
 *         --asm /tmp/fattn_census/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s \
 *         --kernels attn_fattn_wave
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>   // the WMMA operands are F16 whatever global is: __half, __float2half_rn
#include <hip/hip_runtime.h>

#include "kernels.h"
#include "prototype/attn_decode/attn_decode_kernels.h"

namespace engine::kernels {
namespace {

using fattn_ARegs = short __attribute__((ext_vector_type(16)));   // one 16x16 F16 operand tile
using fattn_CRegs = float __attribute__((ext_vector_type(8)));    // one 16x16 F32 accumulator tile

#define FATTN_MROWS 16     // query rows per block: the WMMA M dimension. Not a tuning knob.
#define FATTN_MLANES 16    // KV lanes per round: the C tile is 16 wide. Not a tuning knob.
#define FATTN_KTILES (DH / 16)   // 16-dim steps of the score dot: DH/16 = 16
#define FATTN_VTILES (DH / 16)   // output tiles over DH: the >= 8 independent chains §4.2 change (3) asked

static_assert(FATTN_MROWS == 16 && FATTN_MLANES == 16,
              "the WMMA tile is 16x16x16: MROWS below 16 cannot build an A operand and MLANES below 16 "
              "cannot build a C tile");
static_assert(FATTN_VTILES >= 8, "P*V needs >= 8 independent accumulator chains");
static_assert(sizeof(fattn_ARegs) == 32 && sizeof(fattn_CRegs) == 32,
              "one operand tile and one accumulator tile are both 8 VGPRs (measured: the F16 accumulator "
              "would not be smaller — see prototype/issue_rate_probe/wmma_latency.cpp)");

// The two register-resident arrays, as STRUCTS passed by reference rather than as array parameters. The
// first port passed `const float pexp[8]` and `const fattn_CRegs out[16]` as pointers and the compiler put
// both in scratch — `.private_segment_fixed_size 332`, `.vgpr_spill_count 242` on the committed kernel, the
// same failure `fattn_score_handover.md` §1 records for the rolled P·V loop. Every index into either struct
// below is a compile-time constant (every loop over them is `#pragma unroll`), which is the property that
// keeps them in registers, and `prototype/attn_wmma/fattn_ship_census.py` is what notices if they stop
// being. `FattnO` is the P*V accumulator and stays register-resident; `FattnQ` (see the Q-placement note
// at the top of this file) is the placement that FAILED: 128 VGPR of Q plus 128 of accumulators is 256
// live registers per lane, above what this target keeps in registers at all, so the register placement
// spilled harder than the stage it replaced (`.vgpr_count 264 / .vgpr_spill_count 115 / scratch 724 B`,
// `prototype/attn_wmma/fattn_ship_results.md` §1). `FattnQ` therefore names a STAGE here.

struct FattnO { fattn_CRegs t[FATTN_VTILES]; };   // the P*V accumulators over DH: 128 VGPR/lane

// The staged objects, one expression each (§20.1: two self-consistent layouts joined by a branch are
// invisible to both sides — that is §19's aliasing bug). `s_kv` carries TWO layouts (K row-major for the
// score leg, V transposed for P*V) and is legal only because a barrier separates every reader 
// of one from every writer of the other; its third layout is the F32 output transpose, below.
__device__ __forceinline__ int fattn_q_addr(int q, int d) { return q * DH + d; }      // [row][dim]
__device__ __forceinline__ int fattn_kv_addr(int kv, int d) { return kv * DH + d; }   // [lane][dim]
__device__ __forceinline__ int fattn_v_addr(int d, int kv) { return d * FATTN_MLANES + kv; }  // [dim][lane]
// ONE WARP PER BLOCK, a consequence of the Q placement rather than a style choice: per-warp Q data is
// 8 KiB whatever happens, so a shared block buys nothing once Q is per-head, and 32 threads is what keeps
// the LDS-to-wavefront ratio at the 16 KiB / 4 blocks boundary (4 blocks x 1 warp = 4 wf/CU, which the
// launcher's banner prints as `wf_per_cu` because four is below the range §28 measured).
#define FATTN_THREADS 32
// The output transpose stages F32 through the dead K/V stage
// that stage, which is why it is a half of DH at a time and not the whole of it.
#define FATTN_TROW (DH / 2)                       // floats per query row of the transpose stage
static_assert(FATTN_MLANES * DH * 2 == 8192 && FATTN_MROWS * FATTN_TROW * 4 == 8192,
              "the aliased K/V stage is the P*V operand AND the output transpose staging, at exactly 8 KiB");
static_assert(FATTN_MLANES * DH * 2 == 8192,
              "with Q read from global, the aliased K/V stage is the block's entire LDS: 8 KiB, which on this "
              "part is the 8-blocks/CU boundary. The census reads this off the declarations, not from here.");
// HOW THE PARTIAL IS PUBLISHED: `part` wants the row's DH dims in blocks of FATTN_TBLK, the transpose stages
// half of DH at a time, and FATTN_PBLK is the blocks one lane publishes per half. The assert is the
// COVERAGE claim — every dim of every staged row is written exactly once per half — and it is the guard that
// retires the defect described on `fattn_transpose_out`: a publish loop that covers 16 of every row's 256
// dims satisfies every bound-style assert in this file and still merges scratch.
#define FATTN_TBLK 8
#define FATTN_PBLK 8
static_assert(FATTN_MROWS * FATTN_TROW == FATTN_THREADS * FATTN_PBLK * FATTN_TBLK,
              "the publish loop writes every staged float exactly once per half: rows x dims-per-half "
              "must equal lanes x blocks x block-width");
// THE INDEX BOUNDS OF THE EPILOGUE TRANSPOSE, as compile-time facts. Every index into the transpose
// stage and into the partial slot is closed-form in lane, u, t, l and half, so the two bounds
// the first version of `fattn_transpose_out` violated are asserted here rather than watched for at
// run time: the widest stage write is exactly the stage's last float, and the widest `part`
// write is exactly the shipped slot's last float. A poison-band harness would need a debug
// flag the engine does not have; these cannot be skipped by a code path, which is what retires the
// out-of-range class for good. Coverage — that every index BELOW the bound is reached — is the
// `FATTN_PBLK` assert above, and it is the one the second version of the epilogue needed.
static_assert((FATTN_MROWS - 1) * FATTN_TROW + (16 * (FATTN_VTILES / 2 - 1) + 15) + 1
                  == FATTN_MROWS * FATTN_TROW,
              "the widest transpose stage write is the last float of the staging area");
static_assert(2 + (DH - FATTN_TBLK) + (FATTN_TBLK - 1) + 1 == DH + 2,
              "the widest partial write, the last block of the last dim, is the slot's last float");

__device__ __forceinline__ float fattn_bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fattn_fb(float f) { return hip_bfloat16(f); }
// The shipped score round: the dot's F32 result is rounded through BF16, scaled (2^-4, so the scale is
// exact and the staging can stay 2 bytes), and rounded again. Same expression, same association, as
// `attn_slim_wave`'s `fb(bf(fb(dot)) * SCALE)` — association is load-bearing on this device.
__device__ __forceinline__ float fattn_score_round(float dot) {
    return fattn_bf(fattn_fb(fattn_fb(dot) * SCALE));
}

// One 32-byte operand tile out of an F16 LDS stage: 16 contiguous halves = two 16-B LDS reads.
__device__ __forceinline__ fattn_ARegs fattn_load_tile(const __half* p) {
    fattn_ARegs r;
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
    const __half* a = reinterpret_cast<const __half*>(&lo);
    const __half* b = reinterpret_cast<const __half*>(&hi);
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = (short)__half_as_ushort(a[e]);
#pragma unroll
    for (int e = 0; e < 8; ++e) r[8 + e] = (short)__half_as_ushort(b[e]);
    return r;
}

// The row a lane's C register belongs to, and the column its l-th value belongs to. BOTH come from the
// measured law, and `get_i`/`get_j` here are the same functions for the C tile whatever the operand order
// was — it is the OPERAND slots that the measurement corrected, so these two lines are written once and
// used by every consumer (drain, mask, fold, epilogue) instead of being re-derived at each site.
__device__ __forceinline__ int fattn_c_row(int lane) { return lane & 15; }
__device__ __forceinline__ int fattn_c_col(int lane, int l) { return 2 * l + (lane >> 4); }
// The operand map: lane t holds the 16 contiguous halves of row t%16, the run along the dot index.
__device__ __forceinline__ int fattn_ab_row(int lane) { return lane & 15; }

// C (8 F32) -> the next WMMA's operand (16 F16), by the `get_half2` exchange. It lives FUSED into the exp
// loop of `attn_fattn_body` (round -> max -> exp -> operand, one pass over the lane's 8 columns), not as a
// helper taking the exponentials by pointer: the first port did have such a helper, the array it passed
// became scratch, and the kernel shipped with 242 spilled VGPRs.
//
// WHAT THE EXCHANGE DOES, and the measured blind spot that comes with it. A C tile gives lane t one PARITY of
// its row's 16 columns (columns 2l + t/16), and the operand the next `v_wmma` wants is 16 CONTIGUOUS halves
// of row t%16 — so each lane must collect the partner half-warp's parity with one `shfl_xor(…, 16, …)` per
// register. MEASURED, and the measurement is narrower than the check looked: `cb_no_shfl` (the shuffle
// replaced by the lane's own value) leaves the OUTPUT exactly right and only the operand dump wrong,
// because each half-warp supplies its own parity of the KV axis and the exchange's job is only to make BOTH
// parities correct in BOTH half-warps. The blind spot is named in `prototype/attn_wmma/check_fattn_cb.py`'s
// header; do not "simplify" this by deleting the shuffle on the strength of that result — the operand layout
// is what the next tile reads, and `cb_kv_both_relabel` is the case that keeps the claim honest.
//
// The two slots are filled by a SELECT on the lane's half-warp, never by `tmp[i]` with `i = lane >> 4`: a
// dynamically indexed 2-element array is an alloca, and an alloca inside the round loop is the spill.

// One staged row of 8 dims, BF16 in global -> F16 in the stage (a WMMA operand is F16 whatever global is;
// the conversion is paid once per staged element, which is where the shipped form pays its own).
__device__ __forceinline__ void fattn_stage_row(__half* dst, const hip_bfloat16* src, int lane) {
    const uint4 u = *reinterpret_cast<const uint4*>(src + lane * 8);
    const hip_bfloat16* s = reinterpret_cast<const hip_bfloat16*>(&u);
    union { __half h[8]; uint4 o; } o;
#pragma unroll
    for (int t = 0; t < 8; ++t) o.h[t] = __float2half_rn(fattn_bf(s[t]));
    *reinterpret_cast<uint4*>(dst + lane * 8) = o.o;
}

// THE Q OPERAND, read straight from global: the same 16 halves `fattn_load_tile` returns from a stage, built
// here from the two uint4s of the row's 16 BF16 dims. The conversion is the same
// `__float2half_rn(fattn_bf(x))` `fattn_stage_row` applies, moved from once-per-launch to once-per-round, so
// the operand VALUES are identical and no numerics band can move.
//
// WHY: Q's stage is 16 rows x DH x 2 B = 8192 B, and per-warp Q data is 8 KiB whatever the layout (file header,
// placement (B)), so the only lever on it is to stop staging it at all. That takes a block to 8192 B of LDS,
// which on this part is 8 blocks/CU instead of 4 — twice the wavefronts the banner has been printing as
// `wf_per_cu=4`. The price is one Q tile re-read per round instead of per launch, which is what
// `prototype/attn_wmma/fattn_ship_census.py`'s `global_read_B` term exists to price.
__device__ __forceinline__ fattn_ARegs fattn_load_q_tile(const hip_bfloat16* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
    const hip_bfloat16* s0 = reinterpret_cast<const hip_bfloat16*>(&lo);
    const hip_bfloat16* s1 = reinterpret_cast<const hip_bfloat16*>(&hi);
    fattn_ARegs r;
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = (short)__half_as_ushort(__float2half_rn(fattn_bf(s0[e])));
#pragma unroll
    for (int e = 0; e < 8; ++e) r[8 + e] = (short)__half_as_ushort(__float2half_rn(fattn_bf(s1[e])));
    return r;
}

// One row's slice of this launch. Every quantity here is per ROW, and a row belongs to exactly one row-slot
// of the warp (row fattn_c_row(lane), held by the lane pair t and t^16), so the whole per-row geometry
// lives in registers rather than in an array the compiler would have to index dynamically.
struct FattnRow {
    int row;          // the wave row index, or -1 if this lane's row is out of range or uncovered
    int blo, bhi;     // the lanes this block owns of this row
    bool covered;     // false => leave the row to attn_slim_wave; NOTHING is written for it
};

// The shipped rules, called rather than paraphrased, so this form cannot disagree with the rest of the
// engine about which rows stream (`blocks_for_row`) or how many lanes a row may see (`lanes_of`).
__device__ __forceinline__ FattnRow fattn_row_of(const CoreArgs a, int rt, int lane) {
    FattnRow r{};
    r.row = rt * FATTN_MROWS + fattn_c_row(lane);
    r.covered = false;
    r.blo = 0;
    r.bhi = 0;
    if (r.row >= nrows_of(a)) return r;                 // beyond the wave: nothing to do
    const int nrow = lanes_of(a, r.row);
    const int s_row = blocks_for_row(a, r.row);
    // COVERAGE, the decision this file opens with, asked of the ONE predicate the op header owns
    // (`fattn_covers`), because the VALU wave's early-out asks the same question and two spellings of a
    // rule is how a row ends up owned by both kernels or by neither.
    if (!fattn_covers(a, r.row)) return r;
    if ((int)blockIdx.y >= s_row) return r;             // this block does not own this row's slice
    const int span = (nrow + s_row - 1) / s_row;
    r.blo = (int)blockIdx.y * span;
    r.bhi = r.blo + span < nrow ? r.blo + span : nrow;
    if (r.bhi <= r.blo) return r;
    r.covered = true;
    return r;
}

// THE PARTIAL SLOT, one spelling, shared by the (m, l) publish and by the transpose below. It is the
// shipped expression verbatim — `attn_slim_body`'s epilogue and `attn_combine`'s reader, keyed by the FULL
// head and striding by the LAUNCH's split (§20.1: two self-consistent layouts joined by a branch is §19's
// aliasing bug), so `attn_combine` needs no branch for this form.
__device__ __forceinline__ float* fattn_part_slot(const CoreArgs& a, int row, int head) {
    return a.part + (((size_t)row * N_Q + head) * a.split + blockIdx.y) * (DH + 2);
}

// The accumulator transpose, THE only spelling of these addresses (§20.1). The P*V tiles hold columns
// PARITY-SPLIT across the half-warps (dim = 16t + 2l + lane/16) of ONE ROW — lane t holds row t%16 only —
// while `part` wants the row's 256 dims at `[2 + d]`, which is what `attn_slim_body`'s epilogue writes and
// what `attn_combine` reads, and that is not negotiable without editing the merge. So the output is staged
// in F32 through the DEAD V stage, half of DH at a time: 16 rows x (DH/2) floats = 8192 B = exactly that
// stage. A separate output array would have been 8 KiB ON TOP of the stages, i.e. 2 blocks/CU.
//
// WHO PUBLISHES WHAT is the whole content of this function, and three versions of it have now been wrong:
//   1. it indexed the stage with `lane*8 + u` for lane <= 31 in a 128-float row (lanes 16..31 read the NEXT
//      query row's staging) and wrote `pp[2 + half*128 + lane*8 + u]`, up to `pp[385]` on a 258-float slot
//      — out of the partial buffer, into the following (row, head) slot's numbers.
//   2. it kept the shipped `pp[2 + lane*8 + u]` verbatim. That expression is written for a warp that owns
//      ONE row, where 32 lanes x 8 dims = that row's 256 dims. Here one warp owns SIXTEEN rows, and only
//      the two lanes holding row r (lane r and lane r^16) are entitled to r's slot, so the 32 lanes' 8-dim
//      blocks landed as 8 dims of one half plus 8 of the other in each of 16 slots: 16 dims published per
//      row, 240 never written. `attn_combine` merged whatever the scratch held there, and `finish_head`'s
//      `acc / l` over an unwritten slot is how every covered row came out NaN (§29 of
//      `bench/performance-status.md`). The bound-style `static_assert`s could not see it, because the defect
//      was in the SET of indices touched, not in their maximum — hence the coverage assert on `FATTN_PBLK`,
//      which fails if a publish loop stops covering the stage.
//   3. the (m, l) header, published by `lane == 0` alone: legal for a one-row warp, one row of sixteen here.
//
// THE RULE, once. Per half the stage holds 16 rows x FATTN_TROW floats = 2048 floats, and 32 lanes each
// publish FATTN_PBLK blocks of FATTN_TBLK — asserted to be exactly the stage, so coverage is a fact and not
// a reading. Lane L publishes row-slot L/2, blocks 8*(L&1) .. +7 of it, so exactly one lane owns each
// (row, 8-dim block) pair and every destination index is in [2, 2+DH). A row whose bit is not in `row_mask`
// — uncovered, or past `nrows` — publishes NOTHING, which is what makes `attn_slim_wave`'s write of that
// slot the only one; the STAGING still runs for those rows, because the barriers around it are block-wide
// and cannot be taken by some lanes only. `row_mask` is the ballot of `me.covered`, whose bit r is row-slot r
// (lane r and lane r^16 hold the same row and agree), so the publish set IS the coverage predicate rather
// than a range reasoned about next to it.
__device__ __forceinline__ void fattn_transpose_out(__half* stage_as_f32, const FattnO& out,
                                                    const CoreArgs& a, int rt, int head0,
                                                    unsigned row_mask, int lane) {
    float* stage = reinterpret_cast<float*>(stage_as_f32);
    const int srow = lane >> 1;                          // the row-slot this lane publishes
    const int sblk = (lane & 1) * FATTN_PBLK;            // its first 8-dim block within that row
    const bool mine = ((row_mask >> srow) & 1u) != 0u;   // the ballot, per row-slot
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();                     // the previous half's reads (and the last P*V read of V) are done
#pragma unroll
        for (int t = half * (FATTN_VTILES / 2); t < (half + 1) * (FATTN_VTILES / 2); ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int d = 16 * t + fattn_c_col(lane, l) - half * FATTN_TROW;   // always in [0, 128)
                stage[fattn_c_row(lane) * FATTN_TROW + d] = out.t[t][l];
            }
        }
        __syncthreads();
        if (mine) {        // NO barrier below this line inside the loop, so a guarded store cannot diverge one
            float* pp = fattn_part_slot(a, rt * FATTN_MROWS + srow, head0);
#pragma unroll
            for (int k = 0; k < FATTN_PBLK; ++k) {
                const int sbase = (sblk + k) * FATTN_TBLK;   // [0, FATTN_TROW) by the coverage assert
                const int dbase = half * FATTN_TROW + sbase; // the dim this block carries, in [0, DH)
#pragma unroll
                for (int u = 0; u < FATTN_TBLK; ++u) {
                    // The stage row is `srow`, the row-slot this lane PUBLISHES — not `fattn_c_row(lane)`,
                    // the row-slot this lane ACCUMULATED. Those coincide only for lane < 16, and the
                    // published block of a row is written by two lanes (2*srow and 2*srow+1).
                    pp[2 + dbase + u] = stage[srow * FATTN_TROW + sbase + u];
                }
            }
        }
    }
    __syncthreads();                         // the next launch's stage write must not race the last read
}

// The warp-wide min and max of a per-lane quantity, five butterflies each. Written out because the union of
// the 16 rows' slices has to be computed once per launch, and a dynamically indexed array of 16 spans is
// exactly the thing that puts an array into scratch (`fattn_score_handover.md` §1).
__device__ __forceinline__ int fattn_warp_min(int v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v = min(v, __shfl_xor_sync(FULL_MASK, v, d, warpSize));
    return v;
}
__device__ __forceinline__ int fattn_warp_max(int v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v = max(v, __shfl_xor_sync(FULL_MASK, v, d, warpSize));
    return v;
}

__device__ __forceinline__ void attn_fattn_body(const CoreArgs a) {
    // NO Q STAGE. The score leg's SECOND operand comes from global per round by `fattn_load_q_tile`, which
    // costs a re-read and buys 8192 B of LDS: 8 blocks/CU = 8 wavefronts instead of 4, on the driver's own
    // occupancy answer. Everything else about the placement is the shipped one.
    __shared__ __align__(16) __half s_kv[FATTN_MLANES * DH];   // K [lane][dim] for the score leg, then V
                                                               // [dim][lane] for P*V — ALIASED, one barrier
                                                               // between the two users — then the F32 O
                                                               // transpose. Three layouts, one buffer.
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    const int nw = (int)(blockDim.x / warpSize);               // == 1: `FATTN_THREADS` is one warp
    // Widest row first, exactly as `attn_slim_body` does it: the dispatch tail is where the long rows are.
    const int rt = (int)gridDim.z - 1 - (int)blockIdx.z;
    // ONE WARP PER BLOCK: a block is 16 query rows of ONE query head, so `blockIdx.x` names the head where
    // the shipped wave names a KV group (`ATTN_TILE_THREADS` = `ATTN_TILE_G` warps, one per head). Same
    // number of warps either way; this shape is what the spill gate decided, and
    // `prototype/attn_wmma/fattn_ship_results.md` §1 is the measurement that decided it.
    const int head0 = (int)blockIdx.x;
    const int kv_read = kv_of(head0, a.neg);      // the group this head READS; `kv_of` is the shipped
                                                  // injection-aware rule, and the §05.7 append in
                                                  // `attn_prep` keeps using the honest one

    // ---- this block's rows, and whether ANY of them is this kernel's. `fattn_row_of` depends on the ROW
    // and on `blockIdx.y`, never on the warp, so with one warp the ballot below is the whole block's
    // decision — which is what makes the barriers inside the round legal. Divergence WITHIN the warp (a row
    // block that straddles MAXSC, or whose tail runs off `nrows`) is handled by never returning after this
    // point and guarding stores instead.
    //
    // THE BALLOT IS ALSO THE PUBLISH SET. Bit r of it is lane r's answer, and lane r and lane r^16 hold the
    // SAME wave row (`fattn_c_row(lane) = lane & 15`) and compute the same answer, so the low 16 bits are
    // exactly "which of this block's sixteen row-slots does this launch own" — the set `attn_combine`
    // depends on, taken from the one coverage predicate rather than re-derived beside it.
    const FattnRow me = fattn_row_of(a, rt, lane);
    const unsigned row_mask = (unsigned)__ballot_sync(FULL_MASK, me.covered ? 1 : 0);
    if (!row_mask) return;                     // block-uniform, and before any barrier

    // ---- Q is not staged, but its clamp is the shipped one and is still not cosmetic: a row block at the end
    // of a wave can be partly out of range, and reading `ROW_H * row` for a row past `nrows` reads past the Q
    // rows this launch owns. The staged fill clamped its row slot identically; what differs is that a partly
    // out-of-range block now re-reads a clamped row where the staged form left the stage slot unwritten. The
    // cells are masked out below either way (`live`), so no output value changes, and the clamped read is the
    // safer of the two because it never reads an unwritten stage slot.
    const int qrmax = nrows_of(a) - rt * FATTN_MROWS;             // >= 1: the row block has one row at least
    const int qrs = fattn_ab_row(lane) < qrmax ? fattn_ab_row(lane) : qrmax - 1;
    const hip_bfloat16* qrow = a.qrope + (size_t)ROW_H * (rt * FATTN_MROWS + qrs) + (size_t)head0 * DH;

    // The round range: the union of the covered rows' slices. The 16 rows of a block differ in lane count
    // by at most FATTN_MROWS - 1 and share `s_row`, so their `span`s differ by a little and their `blo`s
    // drift apart by `blockIdx.y * (span_max - span_min)`: the union can cost a few fully-masked rounds at
    // either end, bounded by the slice arithmetic and never by the data. Stated because the shipped form,
    // which cuts one row per block, pays nothing for a union.
    const int lo_all = fattn_warp_min(me.covered ? me.blo : 0x7fffffff);
    const int hi_all = fattn_warp_max(me.covered ? me.bhi : 0);

    float m = -1e30f, l = 0.f;
    FattnO out;
#pragma unroll
    for (int t = 0; t < FATTN_VTILES; ++t) out.t[t] = fattn_CRegs{};

    for (int base = lo_all; base < hi_all; base += FATTN_MLANES) {
        // ---- K for this round. `j` is the ABSOLUTE lane index and the stage slot is ROUND-relative
        // (§13/§14: indexing a stage by a slice-absolute row walks off the array and corrupts its
        // neighbour). Out-of-range lanes clamp to a lane this block can read at all; their cells are masked
        // below, so the value is never used.
        for (int slot = w; slot < FATTN_MLANES; slot += nw) {
            int j = base + slot;
            if (j >= hi_all) j = hi_all - 1;
            fattn_stage_row(&s_kv[fattn_kv_addr(slot, 0)],
                            a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH, lane);
        }
        __syncthreads();                                       // barrier 2: K published

        // ---- the score leg: FIRST operand = K rows (KV lanes), SECOND = Q rows. The slot order is the
        // measured correction (`prototype/attn_wmma/fattn_map_results.md`), not the transcription; putting
        // Q first produces S[kv][q] and folds every row down the wrong axis, and `check_fattn_map.py` plus
        // `check_fattn_cb.py`'s cb_score_slot_swap are the required-fail controls for that regression.
        // ROLLED, as the probe's D58 finding prefers (one accumulator, so the roll costs nothing and the
        // register count stays where the LDS-driven occupancy story wants it).
        // INITIALISED, and not for style: `__builtin_amdgcn_wmma_*` takes the running accumulator as a
        // live IN operand, so the FIRST tile of the roll reads whatever the previous dispatch left in those
        // VGPRs. The prototype this file was ported from says `CRegs acc{}` for exactly this reason
        // (`prototype/attn_wmma/wmma_fattn_score.cpp`'s `score_leg`); the port dropped the braces and read
        // an undefined register block into the first `v_wmma`. One NaN cell anywhere in a row's 16-column P
        // tile makes every output dim of that row NaN, because the P·V tile sums over all 16 KV lanes —
        // which is the whole-row, whole-coverage NaN signature recorded in §29 of
        // `bench/performance-status.md` and decision D64.
        fattn_CRegs acc{};
#pragma unroll 1
        for (int t = 0; t < FATTN_KTILES; ++t) {
            const fattn_ARegs first = fattn_load_tile(&s_kv[fattn_kv_addr(fattn_ab_row(lane), 16 * t)]);
            const fattn_ARegs second = fattn_load_q_tile(qrow + 16 * t);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(first, second, acc);
        }

        // ---- mask, fold, exp, P. The bound is per (row, round): a cell is live iff its lane index is
        // inside THIS ROW's slice. Two compares per cell, which is the cost the prototype's one-round form
        // never had to pay and the census of this file must show.
        fattn_CRegs sc;
        float rm = -1e30f;
#pragma unroll
        for (int l2 = 0; l2 < 8; ++l2) {
            const int j = base + fattn_c_col(lane, l2);      // this cell's lane index within the round
            const bool live = me.covered && j >= me.blo && j < me.bhi;
            sc[l2] = live ? fattn_score_round(acc[l2]) : -1e30f;
            rm = fmaxf(rm, sc[l2]);
        }
        rm = fmaxf(rm, __shfl_xor_sync(FULL_MASK, rm, 16, warpSize));   // one cross-lane op per row/round
        const float mn = fmaxf(m, rm);
        const float rescale = expf(m - mn);
        // exp, rowsum and the C->B operand exchange in ONE pass over the lane's 8 columns; the note above
        // `fattn_transpose_out` says what the exchange is for and why it is fused here rather than factored
        // into a helper that takes an array by pointer (the first port had that helper, and the array it
        // passed is where its 332 B of scratch came from). A row's 16 columns live in the lane pair
        // (t, t^16): `e` is column 2l + lane/16 of this row and the partner's `e` is column
        // 2l + (lane/16 ^ 1), which is exactly the contiguous 16 the operand wants.
        float ls = 0.f;
        fattn_ARegs p;
        const bool hi = (lane & 16) != 0;
#pragma unroll
        for (int l2 = 0; l2 < 8; ++l2) {
            const float e = expf(sc[l2] - mn);
            ls += e;
            const float other = __shfl_xor_sync(FULL_MASK, e, 16, warpSize);
            p[2 * l2 + 0] = (short)__half_as_ushort(__float2half_rn(hi ? other : e));
            p[2 * l2 + 1] = (short)__half_as_ushort(__float2half_rn(hi ? e : other));
        }
        l = l * rescale + ls;
        m = mn;
#pragma unroll
        for (int t = 0; t < FATTN_VTILES; ++t) {
#pragma unroll
            for (int l2 = 0; l2 < 8; ++l2) out.t[t][l2] *= rescale;   // the 128 F32 multiplies per round
        }

        // ---- V stage, ALIASED over the K stage. Every K reader has passed barrier 2 and the score leg
        // above, which is what makes it safe — and it is also the third barrier of the round, the cost of
        // the 16 KiB budget (Q 8 + K/V 8). The alternatives were measured or priced: V in its own stage is
        // 24 KiB and 2 blocks/CU, and the register placement that would have freed Q's 8 KiB spilled
        // harder than the stage it replaced (`fattn_ship_results.md` §1). V is staged TRANSPOSED because
        // the P*V operand whose rows are the output dims needs the KV run contiguous: 32 narrow stores per
        // lane per round here against the 256 narrow reads of the row-major alternative, which is the trade
        // §20.5 of `prototype/attn_decode/results.md` names, not hides.
        __syncthreads();                                       // barrier 3: K consumed
        for (int kv = w; kv < FATTN_MLANES; kv += nw) {
            int j = base + kv;
            if (j >= hi_all) j = hi_all - 1;
            const hip_bfloat16* vsrc =
                a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
            const uint4 u = *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            const hip_bfloat16* vv = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                s_kv[fattn_v_addr(lane * 8 + t, kv)] = __float2half_rn(fattn_bf(vv[t]));
            }
        }
        __syncthreads();                                       // barrier 4: V published

        // ---- the P*V leg: FIRST = V (rows are output dims), SECOND = P (rows are query rows), 16
        // independent accumulator chains so the rescale above is the only serial dependency.
        //
        // FULLY unrolled, and the alternative was measured rather than assumed. `#pragma unroll 8` here gives a
        // listing with `.vgpr_spill_count 0` but `.private_segment_fixed_size 544` — 544 B/thread of scratch —
        // which is the probe's own finding ("rolling the scan legs costs 544 B of scratch", §4.2 change (2))
        // reproduced at the real placement, and it is more scratch traffic than the full unroll's 72 B, not
        // less. Full unroll measures `.vgpr_count 256 / .vgpr_spill_count 49 / private 72`. NEITHER clears the
        // spill gate, which is why `--attn-fattn` is default 0 and why `fattn_ship_census.py` fails on either
        // shape; the fix is to shorten the accumulator's live range (drain each half of DH as it completes, or
        // the two half-DH passes of handover §5(e)), not to retune this pragma. The census re-derives the
        // v_wmma site count from the pragmas, so a retune is census-visible instead of silent.
#pragma unroll
        for (int t = 0; t < FATTN_VTILES; ++t) {
            const fattn_ARegs first =
                fattn_load_tile(&s_kv[fattn_v_addr(16 * t + fattn_ab_row(lane), 0)]);
            out.t[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(first, p, out.t[t]);
        }
    }

    // THE DEFERRED ROWSUM COMBINE — the tenth lane-movement op of the handover's round count (9 in the
    // round + 1 here). A row's 16 columns live in the lane pair (t, t^16), so `l` as accumulated per lane
    // is the sum over ONE PARITY of that row's columns: half a denominator, which scales the row's whole
    // output by ~2x, and that is precisely the error a cosine band cannot see, because cosine is
    // scale-invariant. Deferring is legal rather than lucky: `rescale` is pair-uniform (the max already got
    // its butterfly), so the two halves differ by the same product of rescales at every round and one add
    // at the end is exact. It also has to happen BEFORE the epilogue below, so no lane is ever shuffled
    // against a partner that has already exited. `prototype/attn_wmma/fattn_rowsum_witness.py` is the
    // measured witness: against a pinned row it reads cos 1.000000, meanrel 1.000, argmax unchanged.
    l += __shfl_xor_sync(FULL_MASK, l, 16, warpSize);

    // ---- epilogue: the SHIPPED partial slot expression, verbatim (`fattn_part_slot`), keyed by the full
    // head, so `attn_combine` needs no branch for this form (§20.1). An uncovered row writes NOTHING here,
    // which is what makes `attn_slim_wave`'s write of that same slot the only one — and it is a guarded
    // store, not a `return`, because the transpose below carries block-wide barriers.
    //
    // (m, l) IS PAIR-UNIFORM — `rm` took its `shfl_xor(16)` in the round and `l` took the deferred one
    // above — so either lane of a row's pair may publish it, and exactly one must. `lane == 0` published one
    // row-slot of the block's sixteen and left fifteen headers to whatever the scratch held; `lane < 16`
    // names each row-slot exactly once, because lane r's `me.row` is rt*16 + r.
    if (me.covered && lane < FATTN_MROWS) {
        float* pp = fattn_part_slot(a, me.row, head0);
        pp[0] = m;
        pp[1] = l;
    }
    fattn_transpose_out(s_kv, out, a, rt, head0, row_mask, lane);
}

// The staged plan only. In `fused = 0`, `attn_prep` has already written every row's §05.7 append into the
// cache in a launch that finished before this one started, so `kcache`/`vcache` hold every lane this kernel
// reads — including a row's own new lane, which `attn_slim_body` reads from `krope`/`vg` instead. Those are
// the SAME BYTES (`attn_prep` writes both), so reading them from the cache is not a second source of truth;
// a second source is what §20.1 of `prototype/attn_decode/results.md` is about. `kernels_attn.cpp` refuses
// this form in the fused plan, and the refusal below is that statement rather than a paraphrase of it.
__device__ __host__ inline bool fattn_plan_ok(const CoreArgs a, std::string* why) {
    if (a.fused != 0) {
        if (why) *why = "attn_fattn_wave is a staged-plan kernel: in the fused plan each core block appends "
                        "its own lane, so the 16 rows of one fattn block would read lanes that other rows "
                        "have not appended yet.";
        return false;
    }
    if (a.part == nullptr) {
        if (why) *why = "attn_fattn_wave writes the streaming partials; `part` is null, so its output has "
                        "nowhere to go and attn_combine would read uninitialised slots.";
        return false;
    }
    return true;
}

}  // namespace

// `__launch_bounds__(FATTN_THREADS)` is not decoration here, it is the difference between a shipping kernel
// and a spilling one. Without it LLVM prices occupancy against the target's maximum block size (1024
// threads), which budgets a thread 196,608/1024 = 192 VGPRs and then spills everything above that — which
// is exactly what the first two placements of this file measured (`.vgpr_count 192 / spill 240` with Q
// staged, `264 / 115` with Q in registers), and it is why the probe kernel in
// `prototype/attn_wmma/wmma_fattn_score.cpp` showed 193 VGPR and zero scratch: it carries its own
// `__launch_bounds__(THREADS)`. A 32-thread block is allowed 255 VGPRs (the per-thread architectural cap,
// which is a different limit from the per-CU register file and the one that actually binds a 16-tile F32
// accumulator plus a 16-tile operand set), so the declaration is what lets the form exist at all.
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave(CoreArgs a) { attn_fattn_body(a); }

// The launcher's two host helpers live in the op header beside `tile_rtiles_of`: `fattn_rtiles_of` (this
// form's row-block rule, 16 query rows per block) and `fattn_covers` (the ONE coverage predicate both sides
// of the split call, so the VALU wave's early-out and this kernel's coverage test cannot become two rules).

}  // namespace engine::kernels
