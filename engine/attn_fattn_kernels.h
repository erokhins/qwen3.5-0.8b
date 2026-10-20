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
 *   (B) ONE WARP PER BLOCK, Q STAGED, V ALIASED OVER K. **What shipped at `8a7c222`'s parent.** 32-thread blocks
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
 *   (D) THREE WARPS PER BLOCK, Q STAGED PER WARP, ONE SHARED K/V STAGE (Bet 2). 96-thread blocks, each warp
 *       owning 16 query rows of the same head:
 *         s_q   3 x 16 rows x 256 dims x 2 B = 24576 B   (one 8 KiB stage per warp: Q is per-head and per-warp,
 *                                                         so it cannot be shared and cannot be halved)
 *         s_kv  16 lanes x 256 dims x 2 B     =  8192 B   (ONE stage, amortised over the three row-blocks)
 *         total                               = 32768 B  -> 2 blocks/CU on this part's 64 KiB = 6 wf/CU
 *       Three prices, all named: the round's barriers stop being free (a one-warp block emits ZERO `s_barrier`;
 *       this one has real ones, so the round-range must be block-uniform or the block dead-locks); the round
 *       loop's union is 48 rows wide instead of 16, so a sparse wave pays more fully-masked rounds; and the
 *       epilogue's F32 transpose now goes through each warp's own dead Q stage, because one warp's staging is
 *       the whole 8 KiB K/V stage and three warps sharing it would be §20.1's aliasing bug. What it buys is
 *       interleave WITHOUT touching the accumulator's live range, which is the one thing Bet 1 could not do.
 *       The variant that keeps Bet 1's un-staged Q at three warps (V6 ON THE SKELETON: 73 spilled VGPRs,
 *       284 B/lane of scratch) was re-measured on the real body by `prototype/attn_wmma/E1-result.md` §1 and
 *       lands at 184 VGPR / 0 spill / 0 scratch. What the skeleton got wrong is the LDS: the per-warp 8 KiB is
 *       the epilogue's transpose staging whether or not Q is parked in it, so un-staging Q buys a register
 *       question and no residency.
 *   (E) THREE WARPS PER BLOCK, Q UN-STAGED, EPILOGUE STAGED 8 ROWS AT A TIME (`--attn-fattn-qstage
 *       unstaged8`, bet A6-G1). Same block, same grid, same launch count; two things move:
 *         s_warp 3 x 8 rows x 128 dims x 4 B  = 12288 B   (the epilogue's F32 transpose only — Q is read from
 *                                                        global at the use site, so the stage is 4 KiB/warp)
 *         s_kv   16 lanes x 256 dims x 2 B    =  8192 B   (unchanged)
 *         total                               = 20480 B  -> 3 blocks/CU on this part's 64 KiB = 9 wf/CU
 *       (D)'s error was pricing the buffer, not its content: the epilogue needs 16 rows x FATTN_TROW floats
 *       per warp because that is one half of DH of one warp's accumulator, and (D) parked Q in the same bytes.
 *       Un-staging Q alone therefore bought 6 VGPRs and no residency (`E1` §1, 32,768 B unchanged); shrinking
 *       the transpose to 8 rows per pass and running 4 passes instead of 2 is what moves the bytes, and it is
 *       the same candidate — neither half alone changes the block's LDS. Compile-side: 185 VGPR / 0 spill /
 *       0 scratch, and the driver grants 3 blocks/CU at it (`prototype/attn_percost/A6-result.md` §1, §3).
 *       The two prices it pays, both named: the Q operand's 16 loads per lane-round move from `ds` to `global`
 *       (A5 §3.2 prices that at 568 KB/token/layer of DEMANDED traffic, L1/L2 material by construction), and
 *       the epilogue runs twice as many passes, so twice as many staging barriers. It is owed BYTE identity
 *       with (D), which is what `prototype/attn_percost/A6-G1-impl.md` proves on device rather than argues.
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
 * `fattn_v_addr` are the only way the two K/V layouts are indexed; the per-warp stage is sized by ONE formula
 * (`fattn_warp_stage_halves`) and its slots by one (`fattn_tr_addr`), because that buffer now carries TWO
 * layouts across the two Q spellings — Q's F16 operand rows in the default, the F32 output transpose in
 * `unstaged8` — and §20.1 is what happens when two self-consistent spellings of one buffer are joined by a
 * branch; Q's operand tiles come from exactly one place (`fattn_score_q_tile`, which selects stage-or-global at
 * COMPILE time and is the only caller of `fattn_load_q_tile`/`fattn_q_row`); and the partial slot expression is
 * the shipped one, verbatim, keyed by the FULL head, so `attn_combine` needs no branch for this form either.
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
// One 16x32 F16 ACCUMULATOR tile (`--attn-fattn-pvf16 1`): 16 halves per lane = 8 VGPRs, the same register
// shape as an operand tuple but covering TWICE the output dims, because a half2 slot holds two dims. It is
// filled by TWO `v_wmma_f16_16x16x16_f16` issues — `op_sel` 0 then 1, two different A fragments, the same P —
// and the two halves interleave BY HALF OF EACH VGPR (element 2l is op_sel 0's column, 2l+1 is op_sel 1's).
// That sentence is not a transcription: it is what `a10_pvf16_map_probe.cpp` MEASURED on this device, and
// `check_a10_pvf16_map.py` is the permanent witness (`q = t%16, d = 16*(e&1) + 2*(e>>1) + t/16`).
using fattn_HRegs = short __attribute__((ext_vector_type(16)));   // one 16x32 F16 accumulator tile
// The SAME 32 bytes read as numbers rather than as a WMMA operand shape: the rescale fold below is the only
// place that needs it, and keeping the two spellings separate is what stops a `short` multiply (a bit-pattern
// multiply, off by an exponent) from ever appearing.
using fattn_H2Regs = _Float16 __attribute__((ext_vector_type(16)));

#define FATTN_MROWS 16     // query rows per block: the WMMA M dimension. Not a tuning knob.
#define FATTN_MLANES 16    // KV lanes per round: the C tile is 16 wide. Not a tuning knob.
#define FATTN_KTILES (DH / 16)   // 16-dim steps of the score dot: DH/16 = 16
#define FATTN_VTILES (DH / 16)   // output tiles over DH: the >= 8 independent chains §4.2 change (3) asked
#define FATTN_VT16 (DH / 32)     // F16 accumulator tiles over DH: 8, each 32 dims, two `v_wmma` issues each
#define FATTN_HELEMS 16          // halves per lane in one F16 accumulator tile (8 VGPRs, two dims per slot)

// EXPERIMENT KNOB, compile-time only, 0 in every shipping build: extra LDS appended to the aliased K/V stage
// so that ONE paired window can decide which resource binds the shipping kernel. The two machine-side
// readings disagree — the driver's occupancy query grants 3 blocks/CU (9 wf/CU, the triple §53 carries) while
// the dispatch record reserves 248 VGPR per thread (`a11_read_vgpr.py` reads `.amdhsa_next_free_vgpr 241`, the
// driver rounds to 248, and `a11_alloc_check.py` confirms `Accum_VGPR_Count = 0`), whose arithmetic allows 2
// blocks/CU. Padding declared LDS past 65,536/3 = 21,845 B makes the LDS bound allow 2 too: if registers
// already bind at 2 the pad is free and the answer is registers; if the kernel was really resident at 3 the
// loss shows up here. Byte-identical output by construction, because no index reaches the pad.
#ifndef FATTN_LDS_PAD_HALVES
#define FATTN_LDS_PAD_HALVES 0
#endif
static_assert(FATTN_LDS_PAD_HALVES % 8 == 0,
              "pad the stage in halves, a multiple of 8 so the 16 B LDS accesses stay 16 B aligned");

static_assert(FATTN_MROWS == 16 && FATTN_MLANES == 16,
              "the WMMA tile is 16x16x16: MROWS below 16 cannot build an A operand and MLANES below 16 "
              "cannot build a C tile");
static_assert(FATTN_VTILES >= 8, "P*V needs >= 8 independent accumulator chains");
static_assert(FATTN_VT16 * 32 == DH && FATTN_VT16 == FATTN_VTILES / 2,
              "an F16 accumulator tile covers 32 output dims, so the F16 form needs exactly half as many "
              "tiles as the F32 one and they must tile DH the same way");
static_assert(sizeof(fattn_ARegs) == 32 && sizeof(fattn_CRegs) == 32 && sizeof(fattn_HRegs) == 32,
              "one operand tuple, one F32 accumulator tile and one F16 accumulator tile are each 8 VGPRs. "
              "The equality of the LAST two is the whole economy of the F16 accumulator: equal per-tile cost, "
              "twice the dims per tile, so 8 tiles x 8 = 64 VGPR/lane against the F32 form's 16 x 8 = 128 "
              "(`prototype/attn_percost/A10-RF-LADDER.md` §6 measured 219 -> 155 with Q resident, 177 -> 113 "
              "without: 64 both times). This corrects `prototype/issue_rate_probe/wmma_latency.cpp`'s header "
              "claim that the F16 accumulator saves no registers, which held the TILE COUNT fixed instead of "
              "the output coverage and is retired by that §6 measurement.");
static_assert(FATTN_HELEMS == 2 * (sizeof(fattn_CRegs) / sizeof(float)),
              "an F16 accumulator tile is two C-tile halves interleaved by half of each VGPR — the map the "
              "probe measured, and the reason the drain index is `e>>1` with a `16*(e&1)` dim offset");

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

// The F16 accumulator's sibling (`--attn-fattn-pvf16 1`, `prototype/attn_percost/A10-F16-IMPL.md` D1/D3):
// the SAME DH of output in HALF the accumulator registers — 8 tiles x 8 VGPRs = 64 a lane, against 16 x 8 =
// 128. It is not a faster form: D1 counts 16 `v_wmma` issues per round either way and
// `A10-RF-LADDER.md` §7 times the two atoms at the same ~34 clk. And the register saving it was bought for
// does NOT appear at this placement, measured: the shipped F32 spelling reserves 241 VGPR, this one reserves
// 244, and halving the accumulator AGAIN (4 tiles, 32 registers) still reserves 241 — the reservation is set
// by the rest of the round, not by the accumulator, so the freed registers are absorbed rather than freed.
// `prototype/attn_percost/a10_pvf16_isa_notes.py` is that reading and `A10-PVF16.md` is the write-up. It
// qualifies, without refuting on its own, D7's arithmetic that a Q-in-registers spelling needs a half-width
// accumulator first: at 244 the per-thread 256 ceiling has 12 registers free, and 128 registers of resident Q
// does not fit into 12.
struct FattnO16 { fattn_HRegs t[FATTN_VT16]; };   // the P*V accumulators over DH: 64 VGPR/lane

// ONE accumulator type per knob side, so a knob cannot make BOTH accumulators live in the default build —
// the same reason the roll and the Q placement are template parameters rather than runtime branches. A
// runtime branch would move the default's register budget and stop "the default is inert" from being a
// machine-code compare.
template <bool kPvF16> struct FattnAccOf { using type = FattnO; };
template <> struct FattnAccOf<true> { using type = FattnO16; };
template <bool kPvF16> using FattnAcc = typename FattnAccOf<kPvF16>::type;

// THE D-BLOCK MAP of the F16 atom is spelled once, beside `fattn_c_col`, which it extends: see
// `fattn_h_dim` below the C-tile row/column laws.

// The staged objects, one expression each (§20.1: two self-consistent layouts joined by a branch are
// invisible to both sides — that is §19's aliasing bug). `s_kv` carries TWO layouts (K row-major for the
// score leg, V transposed for P*V) and is legal only because a barrier separates every reader 
// of one from every writer of the other; its third layout is the F32 output transpose, below.
__device__ __forceinline__ int fattn_q_addr(int q, int d) { return q * DH + d; }      // [row][dim]
// Q's GLOBAL source row, the one spelling of it (the other spelling is the stage, above, and a lane reaches Q
// by exactly one of the two, chosen at compile time by `fattn_score_q_tile`). `qrow` is an absolute wave-row
// index and the clamp that keeps it inside the wave lives at its single call site in the body — it is not
// cosmetic: at 48 rows per BLOCK a whole warp's row-block can be past `nrows`.
__device__ __forceinline__ const hip_bfloat16* fattn_q_row(const CoreArgs& a, int qrow, int head0) {
    return a.qrope + (size_t)ROW_H * qrow + (size_t)head0 * DH;                    // [row][head][dim]
}
__device__ __forceinline__ int fattn_kv_addr(int kv, int d) { return kv * DH + d; }   // [lane][dim]
__device__ __forceinline__ int fattn_v_addr(int d, int kv) { return d * FATTN_MLANES + kv; }  // [dim][lane]
// THREE WARPS PER BLOCK (Bet 2): each warp owns its OWN 16 query rows of the same head and its own 8 KiB Q
// stage, and the three of them share ONE K/V stage. That sharing is the whole change: 3 x 8192 (Q) + 8192 (K/V)
// = 32768 B, which on this part is 2 blocks/CU x 3 warps = 6 wavefronts/CU on the driver's own answer, against
// Bet 1's 8 wf/CU bought with 50 spilled VGPRs and 72 B/lane of scratch. Interleave without touching the
// accumulator's live range is the point — the per-warp accumulator is 16 tiles and unchanged — and the price is
// that `__syncthreads()` stops being free and the round's range has to be block-uniform, both below.
#define FATTN_WARPS 3
#define FATTN_THREADS 96
// The output transpose stages F32 through the warp's own dead Q stage
// that stage, which is why it is a half of DH at a time and not the whole of it.
#define FATTN_TROW (DH / 2)                       // floats per query row of the transpose stage
#define FATTN_TBLK 8                       // dims one publish block carries (declared here so the pass
                                           // geometry below can be written beside its siblings)
// THE TRANSPOSE PASS GEOMETRY, one place. The SHIPPED epilogue stages all FATTN_MROWS = 16 row-slots of a warp
// at a time, two passes (one per half of DH). G1 (`--attn-fattn-qstage unstaged8`) stages FATTN_TRG = 8 at a
// time, so a warp's stage is 4 KiB instead of 8 and the epilogue runs FATTN_NPASS = 4 passes — HALF-MAJOR
// (`half = pass / FATTN_NGRP`), so a float's destination order is the shipped one and only the staging depth
// moves. `FATTN_LBLK` is the publish blocks one lane owns per PASS: four lanes per staged row where the
// shipped stage used two, and it is asserted below, not assumed.
#define FATTN_TRG  8                       // query rows per transpose pass (G1; the shipped form uses all 16)
#define FATTN_NGRP (FATTN_MROWS / FATTN_TRG)      // row groups per half of DH: 2
#define FATTN_NPASS (2 * FATTN_NGRP)              // 4 passes instead of 2
#define FATTN_LBLK ((FATTN_TRG * (FATTN_TROW / FATTN_TBLK)) / 32)   // publish blocks per lane per pass: 4
// THE PER-WARP STAGE, IN HALVES, spelled once. This buffer carries two layouts across the two Q spellings: the
// default parks Q in it (16 rows x DH F16 = 8,192 B/warp) and reuses it for the F32 transpose afterwards; the
// un-staged spelling parks ONLY the transpose (FATTN_TRG rows x FATTN_TROW F32 = 4,096 B/warp, i.e. two halves
// per float). It has to be ONE array whose size is a constant expression in the spelling, not two `__shared__`
// arrays: LDS is the SUM of a kernel's `__shared__` declarations, so a second array would move the default's
// declared 32,768 B and the banner's `lds=` field with it — the exact number the residency claim is about.
constexpr int fattn_warp_stage_halves(bool q_unstaged8) {
    return q_unstaged8 ? (2 * FATTN_TRG * FATTN_TROW) : (FATTN_MROWS * DH);
}
static_assert(fattn_warp_stage_halves(false) == FATTN_MROWS * DH,
              "the staged spelling's per-warp stride must be the one `fattn_score_q_tile`'s staged branch"
              " indexes verbatim, or the two spellings of one buffer have silently diverged (§20.1)");
static_assert(fattn_warp_stage_halves(true) * 2 == FATTN_TRG * FATTN_TROW * 4,
              "the un-staged spelling's per-warp stage is FATTN_TRG rows x FATTN_TROW F32 = 4,096 B");
static_assert(FATTN_THREADS == 32 * FATTN_WARPS,
              "FATTN_THREADS is the warp count spelled in threads; the two must be one fact");
static_assert(FATTN_ROWS_PER_BLOCK == FATTN_WARPS * FATTN_MROWS,
              "the launcher's row-block rule (`fattn_rtiles_of` counts blocks of FATTN_ROWS_PER_BLOCK rows) and "
              "the kernel's row-block count must be one number. If the kernel carries fewer rows than the "
              "launcher budgeted, some rows of a wave are nobody's and `attn_combine` merges scratch for them "
              "— §20.1's other failure, where a row neither kernel owns is invisible in both");
static_assert(FATTN_MROWS * FATTN_TROW * 4 == FATTN_MROWS * DH * 2,
              "a warp's dead Q stage is exactly its own output transpose staging: 16 rows x 128 F32 = 16 rows "
              "x 256 F16 = 8192 B. PER WARP, not per block: the three warps transpose into three stages at "
              "once, which is what keeps the epilogue from serialising behind one shared stage");
static_assert(FATTN_TRG * FATTN_TROW * 4 == FATTN_TRG * DH * 2,
              "G1's narrower staging, the same statement one row-group at a time: FATTN_TRG rows x 128 F32 = "
              "4,096 B per warp per pass. This is the byte the residency arithmetic turns on, and it is why "
              "un-staging Q alone bought nothing (`prototype/attn_wmma/E1-result.md` §4)");
static_assert(FATTN_MLANES * DH * 2 == 8192,
              "the aliased K/V stage is one 8 KiB buffer shared by the whole block, so that the three "
              "row-blocks amortise its fill instead of each paying for one");
static_assert(FATTN_WARPS * fattn_warp_stage_halves(false) * 2 + FATTN_MLANES * DH * 2 == 32768,
              "the SHIPPED block declares FATTN_WARPS Q stages and ONE aliased K/V stage: 3 x 8192 + 8192 = "
              "32 KiB, which on this part is the 2-blocks/CU boundary, i.e. 6 wavefronts/CU. The census reads "
              "this off the declarations, not from here.");
static_assert(FATTN_WARPS * fattn_warp_stage_halves(true) * 2 + FATTN_MLANES * DH * 2 == 20480,
              "the `unstaged8` block declares FATTN_WARPS transpose stages of FATTN_TRG rows and ONE aliased "
              "K/V stage: 3 x 4096 + 8192 = 20,480 B, which on this part's 64 KiB of LDS (KFD "
              "`lds_size_in_kb 64`, read, not quoted from a ladder) is the 3-blocks/CU boundary, i.e. 9 "
              "wavefronts/CU — the number G1 exists to buy. Same sum, same formula, other side of the knob.");
// HOW THE PARTIAL IS PUBLISHED: `part` wants the row's DH dims in blocks of FATTN_TBLK, the transpose stages
// half of DH at a time, and FATTN_PBLK is the blocks one lane publishes per half in the SHIPPED 16-row stage
// (FATTN_LBLK, defined above, is the same quantity for one G1 pass). The two asserts are the COVERAGE claim —
// every dim of every staged row is written exactly once per half — and they are the guard that retires the
// defect described on `fattn_transpose_out`: a publish loop that covers 16 of every row's 256 dims satisfies
// every bound-style assert in this file and still merges scratch. The G1 assert is the shipped claim RE-DERIVED
// for the narrower stage rather than asserted beside it, because the stage being narrower does not make the
// destination set smaller: both spellings publish 16 rows x 256 dims, they just do it in 2 passes of 16 rows
// or 4 passes of 8.
#define FATTN_PBLK 8
static_assert(FATTN_MROWS * FATTN_TROW == 32 * FATTN_PBLK * FATTN_TBLK,
              "one warp publishes its OWN row-block's stage, every float exactly once per half: rows x "
              "dims-per-half must equal 32 lanes x blocks x block-width. The 32 is a warp and not "
              "FATTN_THREADS, because the stage and the publishing set are both per warp; making it the block "
              "would have three warps publishing into one stage, which is §20.1's aliasing bug wearing a hat");
static_assert(FATTN_MROWS % FATTN_TRG == 0, "the transpose row groups must tile the warp's 16 row-slots");
static_assert(FATTN_TRG * FATTN_TROW == 32 * FATTN_LBLK * FATTN_TBLK,
              "the same coverage claim for one G1 pass: 8 rows x 128 floats = 32 lanes x 4 blocks x 8 dims. "
              "A pass that published the shipped 8 blocks/lane would walk 4,096 B past its own stage — and "
              "into the next warp's, which is the §19/§20.1 aliasing class, caught by a compiler instead of "
              "on a device");
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
static_assert((FATTN_TRG - 1) * FATTN_TROW + (16 * (FATTN_VTILES / 2 - 1) + 15) + 1
                  == FATTN_TRG * FATTN_TROW,
              "the same bound on the narrower (G1) staging area: a row-slot left over from the 16-row geometry "
              "would walk into the next warp's stage, and an assert is the only instrument that sees that");
static_assert(2 + (DH - FATTN_TBLK) + (FATTN_TBLK - 1) + 1 == DH + 2,
              "the widest partial write, the last block of the last dim, is the slot's last float");

// THE SAME TWO INDEX BOUNDS, RE-DERIVED for the F16 accumulator's layout (D3 site 4 says re-derive, not
// inherit). Both spellings publish the same destination set — 16 rows x 128 dims per half of DH into the
// same stage — so the widest within-half dim must be the SAME `FATTN_TROW - 1`, and it is:
//   F32: 16*(VTILES/2 - 1) + 15              = 16*7 + 15 = 127
//   F16: 32*(VT16/2 - 1) + 16 + 14 + 1       = 32*3 + 31 = 127    (last pair, op_sel-1 half, top column,
//                                                                 the +1 is the second lane of a pair)
// and the per-lane write COUNT is 64 both ways (8 tiles x 8 C elements = 4 pairs x 16 halves), which is the
// COVERAGE claim re-derived rather than assumed: 32 lanes x 64 writes = 16 rows x 128 dims, each exactly
// once, because each lane owns one parity class of one row (see `fattn_h_dim`'s derivation above). A relaxed
// coverage assert is how half of every streaming row's denominator once got published (`a6_g1_bits.sh`, and
// §18.1's all-NaN signature), so these are compile-time facts on the F16 spelling too.
static_assert(32 * (FATTN_VT16 / 2 - 1) + 16 + 14 + 1 == 16 * (FATTN_VTILES / 2 - 1) + 15,
              "the F16 drain's widest within-half dim is the F32 drain's widest dim: same destination set, "
              "different order, and the stage bound cannot relax just because the accumulator got narrower");
static_assert((FATTN_VT16 / 2) * FATTN_HELEMS == (FATTN_VTILES / 2) * 8,
              "both accumulator spellings stage the same number of dims per lane per half of DH (64), which "
              "is what makes 32 lanes x 64 = 16 rows x 128 dims coverage hold for the F16 form for the same "
              "reason it holds for the shipped one");
static_assert((FATTN_MROWS - 1) * FATTN_TROW + (32 * (FATTN_VT16 / 2 - 1) + 16 + 14 + 1) + 1
                  == FATTN_MROWS * FATTN_TROW,
              "the widest transpose stage write under the F16 accumulator is the last float of the stage");
static_assert((FATTN_TRG - 1) * FATTN_TROW + (32 * (FATTN_VT16 / 2 - 1) + 16 + 14 + 1) + 1
                  == FATTN_TRG * FATTN_TROW,
              "the same F16-layout bound on the narrower (G1) staging area");

__device__ __forceinline__ float fattn_bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fattn_fb(float f) { return hip_bfloat16(f); }
// An F16 accumulator element as a float, and a float as an F16 accumulator element. The `short`-typed
// register tuple is the WMMA operand/accumulator SHAPE the builtin wants; these two are the only places the
// bits become numbers, and `__builtin_bit_cast` (not a C cast, which would CONVERT rather than reinterpret) is
// what keeps it a register move. `prototype/gdn_gated_norm/results.md` §8 is the caution this obeys: a `.to()`
// where a `.view()` was meant writes garbage that looks like unwritten memory.
__device__ __forceinline__ float fattn_h_get(short h) {
    return __half2float(__builtin_bit_cast(__half, h));
}
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
// THE D-BLOCK MAP of the F16 atom, the one spelling of it, MEASURED not transcribed. Element `e` of tile
// `tau` of lane `t` holds query row `t%16`'s output at dim
//
//     32*tau + 16*(e&1) + 2*(e>>1) + t/16     ==  32*tau + 16*(e&1) + fattn_c_col(t, e>>1)
//
// i.e. the F32 C-tile law above applied to each op_sel half with `l = e>>1`, plus the half-of-the-pair
// offset `16*(e&1)`. The witness is `prototype/attn_percost/check_a10_pvf16_map.py` over
// `a10_pvf16_map_probe.cpp`'s dump in BOTH op_sel orders (0 disagreements over 32 lanes x 16 elements in
// each); a relabelled map is a permuted row of the output, which is the class of defect that once came back
// as all-NaN (§29 of `bench/performance-status.md`), so it is measured and pinned rather than reasoned about.
__device__ __forceinline__ int fattn_h_dim(int lane, int tau, int e) {
    return 32 * tau + 16 * (e & 1) + fattn_c_col(lane, e >> 1);
}
// The operand map: lane t holds the 16 contiguous halves of row t%16, the run along the dot index.
__device__ __forceinline__ int fattn_ab_row(int lane) { return lane & 15; }

// THE SCORE ROLL'S OPERAND TILES, spelled exactly once. This file can launch four spellings — two score rolls
// (`--attn-fattn-roll`) crossed with two places Q lives (`--attn-fattn-qstage`) — and a branch between two
// spellings of the SAME operand is precisely where §20.1's bug lives: two self-consistent address formulas that
// each agree with themselves and disagree with each other. So no spelling writes an address — every one calls
// these two lines, which are the only place the score leg's tile geometry is written. The `16` is the WMMA K
// width, the same 16 that `FATTN_KTILES = DH / 16` divides by: `t` counts k-tiles and a k-tile is 16 halves of
// one row.
__device__ __forceinline__ fattn_ARegs fattn_score_k_tile(const __half* s_kv, int lane, int t) {
    return fattn_load_tile(&s_kv[fattn_kv_addr(fattn_ab_row(lane), 16 * t)]);
}
// THE Q OPERAND, read from global at the use site (`--attn-fattn-qstage unstaged8`). The same 32-byte
// `fattn_ARegs` `fattn_load_tile` returns out of a stage, built from the two uint4s of the lane's row: the
// conversion is exactly the `__float2half_rn(fattn_bf(x))` that `fattn_stage_row` applies on the way in, so the
// operand VALUES are bit-identical and what moves is where the bytes come from — 8 KiB per warp per LAUNCH
// becomes 8 KiB per warp per ROUND, which is a traffic question for the device (`prototype/attn_percost/A5-
// GEOMETRY.md` §3.2 prices the demanded extra) and not a bits question.
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
// The roll's Q tile, THE one spelling for both Q placements. The selection is `if constexpr`, not a runtime
// `?:`, for the reason the roll axis is templated too: a live-in runtime branch would keep the global-address
// arithmetic in the DEFAULT instantiation and move its register budget, which would stop the default being
// HEAD's machine code and therefore stop "the default is inert" from being a byte-compare. In the staged
// instantiation `qrow` is a null that no read reaches (its branch is not in this instantiation at all), and in
// the un-staged one `s_warp`/`w`/`lane` are unread because the stage is the epilogue's, not Q's.
template <bool kQUnstaged8>
__device__ __forceinline__ fattn_ARegs fattn_score_q_tile(const __half* s_warp, int w, int lane,
                                                         const hip_bfloat16* qrow, int t) {
    if constexpr (kQUnstaged8) {
        (void)s_warp; (void)w; (void)lane;
        return fattn_load_q_tile(qrow + 16 * t);
    } else {
        (void)qrow;
        // The per-warp stride comes from the same constexpr as the buffer's declaration — one spelling of the
        // layout, so a stage and its reader cannot be sized differently (§20.1). This branch IS the staged
        // spelling, which is why the argument is `false`; `kQUnstaged8` is what selected this branch.
        return fattn_load_tile(&s_warp[w * fattn_warp_stage_halves(false) +
                                       fattn_q_addr(fattn_ab_row(lane), 16 * t)]);
    }
}

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

// THE TWO Q SPELLINGS, side by side. This file used to say the Q stage was a precondition, and on the evidence
// it had it was right at the time: the one-warp form spilled (Bet 1, 50 spilled VGPRs) and the skeleton's
// three-warp un-stage spilled harder (V6, 73). `prototype/attn_wmma/E1-result.md` §1 re-measured the un-stage
// on the REAL body at 184 VGPR and ZERO spill, so the precondition was a register myth; what it does not buy —
// and this is the other half of E1's finding, and the reason `--attn-fattn-qstage` names a PAIR of changes
// rather than one — is any LDS, because the epilogue stands on the same per-warp 8 KiB whatever Q does. See
// placement (E) in the file header for the byte sums and what they buy.

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

// THE TRANSPOSE STAGE SLOT, one spelling for both epilogues (§20.1, and the reason a narrower stage cannot be
// addressed two ways by the two halves of one epilogue): [row WITHIN THIS PASS][dim within this half of DH].
// Writer and reader both go through it, and rows are renumbered per pass — that renumbering is the whole content
// of G1's staging change, so it is written once here rather than reasoned about at each index site.
__device__ __forceinline__ int fattn_tr_addr(int row_in_pass, int d) { return row_in_pass * FATTN_TROW + d; }

// The accumulator transpose, THE only spelling of these addresses (§20.1). The P*V tiles hold columns
// PARITY-SPLIT across the half-warps (dim = 16t + 2l + lane/16) of ONE ROW — lane t holds row t%16 only —
// while `part` wants the row's 256 dims at `[2 + d]`, which is what `attn_slim_body`'s epilogue writes and
// what `attn_combine` reads, and that is not negotiable without editing the merge. So the output is staged
// in F32 through the CALLER's own per-warp stage, half of DH at a time: the shipped form stages all 16 rows
// (16 x (DH/2) floats = 8192 B = exactly one warp's Q stage, which is dead by now) in TWO passes; `unstaged8`
// stages FATTN_TRG = 8 rows (4,096 B) in FOUR, half-major so a float's destination order is unchanged. It is
// PER WARP either way on purpose: one warp's staging is bigger than a third of the shared K/V stage, so three
// warps transposing through `s_kv` would write one another's rows.
// A separate output array would have been 8 KiB ON TOP of the stages, i.e. 1 block/CU instead of 2.
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
// THE RULE, once. The SHIPPED half-stage holds 16 rows x FATTN_TROW floats = 2048 floats, and 32 lanes each
// publish FATTN_PBLK blocks of FATTN_TBLK — asserted above to be exactly the stage, so coverage is a fact and
// not a reading. Lane L publishes row-slot L/2, blocks 8*(L&1) .. +7 of it, so exactly one lane owns each
// (row, 8-dim block) pair and every destination index is in [2, 2+DH). In a G1 pass the stage holds FATTN_TRG
// rows and FOUR lanes publish each of them, FATTN_LBLK blocks apiece — the same (row, block) pairs, produced by
// a different split, which is what the second coverage assert above pins. A row whose bit is not in `row_mask`
// — uncovered, or past `nrows` — publishes NOTHING, which is what makes `attn_slim_wave`'s write of that
// slot the only one; the STAGING still runs for those rows, because the barriers around it are block-wide
// and cannot be taken by some lanes only. `row_mask` is the ballot of `me.covered`, whose bit r is row-slot r
// (lane r and lane r^16 hold the same row and agree), so the publish set IS the coverage predicate rather
// than a range reasoned about next to it.
//
// WHY TEMPLATED AND NOT BRANCHED: same reason as the score roll and the Q placement. A runtime branch would make
// both staging geometries live in every instantiation and move the default's declared LDS, and `lds=` is the
// field the residency claim and the A/B's canary are read from.
// WHICH ACCUMULATOR ELEMENT CARRIES WHICH DIM, spelled once per accumulator width (D3 site 4). This is the
// ONLY epilogue difference between the two forms. The PUBLISH side of `fattn_transpose_out` below is shared
// deliberately: that side carries the coverage constraint (§18.1's all-NaN was a publish loop that covered 16
// of every row's dims while satisfying every bound assert), and a second spelling of it is the drift the A9
// Q-placement note warns about. Both spellings write the same destination set — 16 rows x 128 dims per half of
// DH — and disagree only about which dim an element carries: `fattn_c_col` for the F32 tile, and the MEASURED
// `fattn_h_dim` for the F16 one.
template <bool kPvF16>
__device__ __forceinline__ void fattn_stage_half(float* stage, int row, const FattnAcc<kPvF16>& out,
                                                int half, int lane) {
    if constexpr (kPvF16) {
#pragma unroll
        for (int tau = half * (FATTN_VT16 / 2); tau < (half + 1) * (FATTN_VT16 / 2); ++tau) {
#pragma unroll
            for (int e = 0; e < FATTN_HELEMS; ++e) {
                stage[fattn_tr_addr(row, fattn_h_dim(lane, tau, e) - half * FATTN_TROW)]
                    = fattn_h_get(out.t[tau][e]);
            }
        }
    } else {
#pragma unroll
        for (int t = half * (FATTN_VTILES / 2); t < (half + 1) * (FATTN_VTILES / 2); ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int d = 16 * t + fattn_c_col(lane, l) - half * FATTN_TROW;  // always in [0, 128)
                stage[fattn_tr_addr(row, d)] = out.t[t][l];
            }
        }
    }
}

template <bool kNarrow, bool kPvF16>
__device__ __forceinline__ void fattn_transpose_out(float* stage, const FattnAcc<kPvF16>& out,
                                                    const CoreArgs& a, int rt, int head0,
                                                    unsigned row_mask, int lane) {
    if constexpr (kNarrow) {
        // G1 (`--attn-fattn-qstage unstaged8`): FATTN_NPASS passes of FATTN_TRG rows. `half` still picks which
        // half of DH this pass stages — HALF-MAJOR, so a float's destination order is the shipped one — and `rg`
        // picks which group of row-slots it stages, so the published VALUES are untouched: same source register,
        // same `part` slot, same (m, l) publish above. What moves is the staging addresses and the barrier count.
#pragma unroll
        for (int pass = 0; pass < FATTN_NPASS; ++pass) {
            const int half = pass / FATTN_NGRP;
            const int rg = pass % FATTN_NGRP;
            const int row0 = rg * FATTN_TRG;              // this pass's first row-slot within the warp
            __syncthreads();      // the previous pass's reads (and the last P*V read of V) are done
            // STAGE. `fattn_c_row(lane)` is the row-slot this lane ACCUMULATED (held by the pair t, t^16), so
            // half the warp's lanes hold rows this pass does not stage and write nothing — a guarded store, not
            // a divergent barrier: the barriers are above and below it, and every lane takes both.
            if (fattn_c_row(lane) >= row0 && fattn_c_row(lane) < row0 + FATTN_TRG) {
                fattn_stage_half<kPvF16>(stage, fattn_c_row(lane) - row0, out, half, lane);
            }
            __syncthreads();
            // PUBLISH: FATTN_LBLK blocks of FATTN_TBLK per lane, four lanes per staged row where the shipped
            // 16-row stage used two.
            const int srow = row0 + (lane >> 2);          // the row-slot THIS lane publishes
            const int sblk = (lane & 3) * FATTN_LBLK;     // its first block within that row's half
            if (((row_mask >> srow) & 1u) != 0u) {   // NO barrier below this line inside the loop
                float* pp = fattn_part_slot(a, rt * FATTN_MROWS + srow, head0);
#pragma unroll
                for (int k = 0; k < FATTN_LBLK; ++k) {
                    const int sbase = (sblk + k) * FATTN_TBLK;   // [0, FATTN_TROW) by the pass coverage assert
                    const int dbase = half * FATTN_TROW + sbase; // the dim this block carries, in [0, DH)
#pragma unroll
                    for (int u = 0; u < FATTN_TBLK; ++u) {
                        // Row-within-pass is `lane >> 2` for the reader exactly as it was `fattn_c_row(lane) -
                        // row0` for the writer — one expression, `fattn_tr_addr`, on both sides (§20.1).
                        pp[2 + dbase + u] = stage[fattn_tr_addr(lane >> 2, sbase + u)];
                    }
                }
            }
        }
    } else {
        const int srow = lane >> 1;                          // the row-slot this lane publishes
        const int sblk = (lane & 1) * FATTN_PBLK;            // its first 8-dim block within that row
        const bool mine = ((row_mask >> srow) & 1u) != 0u;   // the ballot, per row-slot
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            __syncthreads();         // the previous half's reads (and the last P*V read of V) are done
            fattn_stage_half<kPvF16>(stage, fattn_c_row(lane), out, half, lane);
            __syncthreads();
            if (mine) {    // NO barrier below this line inside the loop, so a guarded store cannot diverge one
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
                        pp[2 + dbase + u] = stage[fattn_tr_addr(srow, sbase + u)];
                    }
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

// The body is templated on the score-roll spelling rather than branching on it at RUNTIME, for one reason:
// a runtime `if` would make both accumulator sets live in every build and move the register count of the
// DEFAULT path, which would stop it being the shipped path. Two instantiations are two kernels in this
// binary and the knob selects a LAUNCH (`engine/kernels_attn.cpp`), so the A/B is one window instead of two
// builds — AGENTS: a ruler re-taken in a different window is not an A/B — and the launch count, the grid and
// the declared LDS are the same on both sides either way.
//
// The SECOND parameter is the Q placement (`--attn-fattn-qstage`, bet A6-G1) and is templated for the same
// reason spelled twice in this file already: the two placements differ in the block's declared LDS, and a
// runtime branch would declare both buffers in every build. `false` = Q staged in this warp's per-warp stage
// (SHIPPED, the code path this header carried before either knob existed); `true` = Q read from global at the
// use site and that same stage given over to the epilogue at FATTN_TRG rows a pass. The two are ONE candidate
// — un-staging Q alone moves no byte of LDS, and narrowing the epilogue alone would shrink the buffer Q lives
// in — which is why one knob names the pair.
template <bool kTwoAcc, bool kQUnstaged8, bool kPvF16>
__device__ __forceinline__ void attn_fattn_body(const CoreArgs a) {
    // ONE PER-WARP STAGE, sized by the spelling (see `fattn_warp_stage_halves`: one array, not two, because
    // `__shared__` arrays SUM and would move the default's declared LDS). In the SHIPPED spelling this is the
    // Q stage: the score leg's SECOND operand is this warp's 16 rows, staged once per launch (8 KiB), and the
    // three warps of the block each stage their own row-block. Q cannot be shared: per-warp Q data is 8 KiB
    // whatever the layout (file header, placement (B)). In the G1 spelling Q is read from global per round and
    // the buffer holds only the epilogue's F32 transpose, 4 KiB a warp — which is placement (E) and the whole
    // residency claim of this knob.
    constexpr int warp_stage_halves = fattn_warp_stage_halves(kQUnstaged8);
    __shared__ __align__(16) __half s_warp[FATTN_WARPS * warp_stage_halves];
    // ONE K/V stage per BLOCK, shared by all three row-blocks. This is what the third warp buys: the stage fill
    // runs once for three row-blocks instead of once per row-block. ALIASED uses (K row-major for the score leg,
    // then V transposed for P*V) with a barrier between the two users, and its first ints double as the
    // block-range fold below before the first K fill touches it. Two layouts, one buffer, three readers.
    __shared__ __align__(16) __half s_kv[FATTN_MLANES * DH + FATTN_LDS_PAD_HALVES];  // [lane][dim], then [dim][lane]
    // The pad is spelled as a GROWTH of `s_kv`, not as a second `__shared__` array, and that is an instrument
    // lesson worth keeping: the first attempt declared `s_pad[N]` with a store under `a.cap == 0`, and the
    // declared LDS did not move at all — a write into a shared array nothing ever reads is dead, so the
    // allocation was eliminated and both arms read 20,480 B, which is how a null result nearly got reported as
    // a finding. Growing an array the kernel provably uses cannot be eliminated. The appended tail is
    // unreachable: every index into `s_kv` comes from `fattn_kv_addr`/`fattn_v_addr`, whose widest value the
    // `static_assert`s above pin against the UNPADDED extent, and no code memsets or copies the array
    // wholesale (there is no `sizeof(s_kv)` anywhere in this file).
    static_assert(FATTN_MLANES * DH + FATTN_LDS_PAD_HALVES >= FATTN_MLANES * DH,
                  "the pad only ever grows the stage");
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    const int nw = (int)(blockDim.x / warpSize);              // == FATTN_WARPS: the two s_kv fills stride on it
    // Widest row first, exactly as `attn_slim_body` does it: the dispatch tail is where the long rows are.
    // `rt` names a BLOCK's row tile (FATTN_WARPS row-blocks); `rb` is this warp's 16-row row-block inside it.
    const int rt = (int)gridDim.z - 1 - (int)blockIdx.z;
    const int rb = rt * FATTN_WARPS + w;
    // THREE WARPS PER BLOCK, all of ONE query head, so `blockIdx.x` names the head where the shipped wave names
    // a KV group (`ATTN_TILE_THREADS` = `ATTN_TILE_G` warps, one per head). Same total warp count as the
    // one-warp block; what changed is that the K/V stage is now amortised over three row-blocks and the block is
    // 32 KiB of LDS instead of 8, which is what the driver's `wf_per_cu` answer is about.
    const int head0 = (int)blockIdx.x;
    const int kv_read = kv_of(head0, a.neg);      // the group this head READS; `kv_of` is the shipped
                                                  // injection-aware rule, and the §05.7 append in
                                                  // `attn_prep` keeps using the honest one

    // ---- this warp's rows, and whether ANY of them is this kernel's. `fattn_row_of` depends on the ROW
    // and on `blockIdx.y`, never on the warp, so the ballot is the row-block's decision. Divergence WITHIN the
    // warp (a row block that straddles MAXSC, or whose tail runs off `nrows`) is handled by never returning
    // after this point and guarding stores instead — and what is NEW at three warps is that the RETURN cannot
    // be per-warp either: a warp that leaves before a shared-stage barrier leaves the other two waiting.
    //
    // THE BALLOT IS ALSO THE PUBLISH SET. Bit r of it is lane r's answer, and lane r and lane r^16 hold the
    // SAME wave row (`fattn_c_row(lane) = lane & 15`) and compute the same answer, so the low 16 bits are
    // exactly "which of this row-block's sixteen row-slots does this launch own" — the set `attn_combine`
    // depends on, taken from the one coverage predicate rather than re-derived beside it. Per warp, and the
    // transpose below publishes per warp.
    const FattnRow me = fattn_row_of(a, rb, lane);
    const unsigned row_mask = (unsigned)__ballot_sync(FULL_MASK, me.covered ? 1 : 0);

    // ---- Q, in one of two spellings, and both of them keep the SHIPPED clamp.
    //
    // STAGED (`--attn-fattn-qstage stage`, the default): Q goes into this warp's own 8 KiB once per launch,
    // ALL SIXTEEN ROW-SLOTS, BY THIS WARP ALONE. The fill MUST NOT stride on `w`, and that is the one place the
    // shared-stage pattern must not be copied. `s_kv` is one buffer for three row-blocks, so ITS fill splits
    // across the warps; this warp's own stage is one warp's operand and the score leg below reads all 16 of its
    // row-slots every round. A strided `for (qr = w; qr < FATTN_MROWS; qr += nw)` — the shape the two K/V fills
    // below use — stages 6/5/5 slots for warps 0/1/2 and leaves the other 11 slots of every stage unwritten;
    // the score leg then reads them as whatever LDS held, `live` is true so nothing masks them, and every such
    // row's `l` and accumulator go non-finite (`bench/performance-status.md` §32's all-NaN red). Note what that
    // means for a bisect: at `FATTN_WARPS = 1` the strided loop and this loop are THE SAME LOOP, so a `WARPS=1`
    // probe cannot see this defect, and its green result is not evidence about the per-warp Q stage.
    //
    // UN-STAGED (`unstaged8`, bet A6-G1): there is no fill at all. Each lane keeps ONE pointer — the global row
    // its operand tile carries, read again every round — and nothing about the round sequence, the masks or the
    // fold order moves.
    //
    // THE CLAMP IS THE SHIPPED ONE IN BOTH and it is not cosmetic: a row block at the end of a wave can be
    // partly out of range, and at 48 rows per BLOCK a whole warp's row-block can be PAST `nrows`
    // (`qrmax <= 0`), where `qrmax - 1` alone would index rows before the buffer. The staged form `break`s out
    // of its fill; the un-staged form resolves to row `nrows - 1` there, so every lane dereferences a row that
    // exists. In both, a clamped row's cells are masked out below (`live`), so the bytes it cost are never used
    // for anything.
    const int qrmax = nrows_of(a) - rb * FATTN_MROWS;   // row-slots of this warp that exist; may be <= 0
    const hip_bfloat16* qrow = nullptr;
    if constexpr (kQUnstaged8) {
        const int qslot = fattn_ab_row(lane);              // the row-slot THIS lane's operand tile carries
        const int qrs = qslot < qrmax ? qslot : qrmax - 1;  // the shipped clamp, per lane
        qrow = fattn_q_row(a, rb * FATTN_MROWS + qrs, head0);
    } else {
        for (int qr = 0; qr < FATTN_MROWS; ++qr) {         // every slot of THIS warp's stage, exactly once
            const int qrs = qr < qrmax ? qr : qrmax - 1;
            if (qrs < 0) break;                            // this whole row-block is past the last row
            fattn_stage_row(&s_warp[w * warp_stage_halves + fattn_q_addr(qrs, 0)],
                            a.qrope + (size_t)ROW_H * (rb * FATTN_MROWS + qrs) + (size_t)head0 * DH, lane);
        }
    }

    // ---- THE ROUND RANGE MUST BE BLOCK-UNIFORM, and that is the structural price of the shared stage: if the
    // three warps disagreed on the trip count, the round's barriers would dead-lock one warp against two, which
    // is not a slowdown but a hang (§29's fault-shape, one layer over). So the three warps' warp-level unions
    // are folded through the FRONT OF THE K/V STAGE, which is dead until the first K fill. A separate
    // `__shared__ int[6]` would have cost 24 B — and 32792 B is over the 32 KiB boundary, i.e. 2 blocks/CU
    // becomes 1 and the whole bet becomes 3 wf/CU, so the fold lives in the buffer it already had.
    const int lo_mine = fattn_warp_min(me.covered ? me.blo : 0x7fffffff);
    const int hi_mine = fattn_warp_max(me.covered ? me.bhi : 0);
    int* fold = reinterpret_cast<int*>(s_kv);
    if (lane == 0) {
        fold[w] = lo_mine;
        fold[FATTN_WARPS + w] = hi_mine;
        fold[2 * FATTN_WARPS + w] = (int)row_mask;
    }
    __syncthreads();                                       // barrier 1: Q published, spans published
    int lo_all = 0x7fffffff, hi_all = 0;
    unsigned any_mask = 0;
#pragma unroll
    for (int i = 0; i < FATTN_WARPS; ++i) {   // compile-time trip count, so the fold stays in registers
        lo_all = min(lo_all, fold[i]);
        hi_all = max(hi_all, fold[FATTN_WARPS + i]);
        any_mask |= (unsigned)fold[2 * FATTN_WARPS + i];
    }
    if (!any_mask) return;      // BLOCK-uniform now, so no warp is left holding a barrier
    __syncthreads();            // barrier 2: every reader is done with the fold before the K fill overwrites it

    // The round range: the union of the covered rows' slices, now of the BLOCK's 48 rows rather than one
    // row-block's 16. Rows differ in lane count by at most a few and share `s_row`, so their `span`s drift and
    // the union costs fully-masked rounds at either end, bounded by the slice arithmetic and never by the data
    // — but the bound is now three row-blocks wide, which is a real cost this geometry pays and
    // `prototype/attn_wmma/fattn_ship_census.py` cannot see (it counts static sites, not rounds).

    float m = -1e30f, l = 0.f;
    FattnAcc<kPvF16> out;
    if constexpr (kPvF16) {
#pragma unroll
        for (int tau = 0; tau < FATTN_VT16; ++tau) out.t[tau] = fattn_HRegs{};
    } else {
#pragma unroll
        for (int t = 0; t < FATTN_VTILES; ++t) out.t[t] = fattn_CRegs{};
    }

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
        __syncthreads();                                       // barrier 3: K published (to all three warps)

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
        if constexpr (kTwoAcc) {
            // THE TWO-ACCUMULATOR ROLL (`--attn-fattn-roll two`, bet A4-L1). Gate 0 of
            // `prototype/attn_percost/A4-L1-gates.md` §1 read the shipped roll off the shipping object as ONE
            // `v_wmma` site in a 16-trip rolled loop whose destination tuple is also its accumulator operand
            // — a 16-long loop-carried recurrence with nothing independent between trips. This spelling gives
            // the roll a second accumulator so two chains of 8 progress at once.
            //
            // The ORDER of the four loads is the lever, not the second accumulator: BOTH tiles of a pair are
            // fetched before EITHER matrix op issues, so one LDS round-trip pays for two independent WMATAs
            // and the chain halves. Read back from the emitted loop in `A4-L1-gates.md` §3 — 8 trips, eight
            // `ds_load_b128` before the first `v_wmma`, and the first wait is `lgkmcnt(2)`, not `lgkmcnt(0)`.
            fattn_CRegs acc0{}, acc1{};
            static_assert(FATTN_KTILES % 2 == 0, "the two-accumulator roll pairs k-tiles");
#pragma unroll 1
            for (int t = 0; t < FATTN_KTILES; t += 2) {
                const fattn_ARegs k0 = fattn_score_k_tile(s_kv, lane, t);
                const fattn_ARegs q0 = fattn_score_q_tile<kQUnstaged8>(s_warp, w, lane, qrow, t);
                const fattn_ARegs k1 = fattn_score_k_tile(s_kv, lane, t + 1);
                const fattn_ARegs q1 = fattn_score_q_tile<kQUnstaged8>(s_warp, w, lane, qrow, t + 1);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k0, q0, acc0);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k1, q1, acc1);
            }
            // ONE add after the roll, not one per trip.
            //
            // WHAT THIS COSTS, said in advance rather than discovered afterwards: the shipped dot is a fixed
            // ascending F32 sum of the 16 k-tile partials, and `acc0` (even tiles) + `acc1` (odd) is a
            // RE-ASSOCIATION of it, so `o` moves and bit-identity with the shipped form is gone by
            // construction. `prototype/attn_percost/A4-BET.md` fixes the judge in advance: the existing §10.1
            // cos/meanrel aggregates through `engine/gate.py`, no new band, no widening, no `--ack-spill`.
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = acc0[i] + acc1[i];
        } else {
            // THE SHIPPED ROLL — `--attn-fattn-roll single`, the default, and the code path this file carried
            // before the knob existed, transcribed (the loop, the `#pragma unroll 1` that keeps it one
            // accumulator, and the operand order the measured correction of
            // `prototype/attn_wmma/fattn_map_results.md` fixes). `single` owes HEAD bit-identity, and
            // `prototype/attn_percost/A4-L1-impl.md` is where that is proved, not asserted.
#pragma unroll 1
            for (int t = 0; t < FATTN_KTILES; ++t) {
                const fattn_ARegs first = fattn_score_k_tile(s_kv, lane, t);
                const fattn_ARegs second = fattn_score_q_tile<kQUnstaged8>(s_warp, w, lane, qrow, t);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(first, second, acc);
            }
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
        // THE RESCALE, on the accumulator's critical path in BOTH forms (D3 site 3). The F32 form is the
        // shipped one, 128 `v_mul_f32_e32` per round. The F16 form folds the SAME factor into the SAME live
        // range, on 128 halves held in 64 registers: `rh` is the same `expf(m - mn)`, rounded to F16 once per
        // round, and each product rounds to F16 — so it is the fold's FORMAT that moved, not its math.
        // What the compiler actually emits was read out of the listing rather than assumed, and D1's "half the
        // VALU of that loop" survives the reading: the fold comes out as 64 `v_pk_mul_f16` (two halves per
        // instruction, so the target does have a packed F16 multiply) against the shipped arm's 128
        // `v_mul_f32_e32`, per `prototype/attn_percost/a10_pvf16_isa_notes.py`. What D1 did not count is the
        // drain: 128 `v_cvt_f32_f16_e32` appear per launch to widen the halves back to the F32 transpose stage,
        // so the trade is -64 VALU slots per ROUND against +129 conversion slots per LAUNCH, and a round loop
        // of R rounds pays -64R + 129. The register half of the claim did NOT survive; see the header note and
        // `prototype/attn_percost/A10-PVF16.md`.
        if constexpr (kPvF16) {
            // The SAME factor the F32 form multiplies by, rounded F32->F16 once per round (round to nearest,
            // the rounding `__float2half_rn` performs — and `_Float16` rather than `__half` because only the
            // former is a valid `ext_vector_type` element on this target).
            const _Float16 rh = rescale;
#pragma unroll
            for (int tau = 0; tau < FATTN_VT16; ++tau) {
                fattn_H2Regs v = __builtin_bit_cast(fattn_H2Regs, out.t[tau]);
#pragma unroll
                for (int e = 0; e < FATTN_HELEMS; ++e) v[e] = v[e] * rh;
                out.t[tau] = __builtin_bit_cast(fattn_HRegs, v);
            }
        } else {
#pragma unroll
            for (int t = 0; t < FATTN_VTILES; ++t) {
#pragma unroll
                for (int l2 = 0; l2 < 8; ++l2) out.t[t][l2] *= rescale;   // the 128 F32 multiplies per round
            }
        }

        // ---- V stage, ALIASED over the K stage, and — new at three warps — aliased across THREE readers. Every
        // K reader has passed barrier 3 and the score leg above, which is what makes it safe, and barrier 5 is
        // what makes the V published to all three warps before any of them reads it. This is the third barrier
        // of the round and it is no longer free: at one warp per block LLVM deleted every `__syncthreads()` (the
        // measured listing has zero `s_barrier`), so aliasing V over K cost nothing to sequence; at three warps
        // they are real barriers and the round pays them, which is exactly what `fattn_ship_census.py`'s
        // `s_barrier` term was written to refuse to inherit. The 8 KiB stage is amortised over three row-blocks
        // either way, which is the trade this bet is.
        __syncthreads();                                       // barrier 4: K consumed (all three warps)
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
        __syncthreads();                                       // barrier 5: V published (to all three warps)

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
        if constexpr (kPvF16) {
            // THE TWO-ISSUE SHAPE, llama's `mma.cuh:1553-1566` exactly: two `v_wmma_f16_16x16x16_f16` issues
            // per 32-dim tile, `op_sel` 0 then 1, TWO DIFFERENT A fragments (the low 16 dims of the pair, then
            // the high 16) and the SAME P, accumulating into the SAME D block. The price of the F16 atom is the
            // second issue — the two atoms cost the same per unit of coverage (`~34 clk` each at llama's shape,
            // `prototype/issue_rate_probe/results.md` §2b), so one wide F32 atom is not a cheaper spelling of
            // the same 32 dims and the pair is not overhead. And an F16 accumulator DISCARDS an add once the
            // addend falls below half a ULP of the running value (`issue_rate_probe/results.md` §6, restated as
            // the magnitude risk in `A10-F16-IMPL.md` D5), which is why the rescale above stays exactly where it
            // is instead of being deferred to the epilogue.
            //
            // The A fragments are the same two LDS tiles the F32 leg reads for tiles 2*tau and 2*tau+1: the V
            // stage is addressed by output dim either way, so the stage, its address expression and its two
            // barriers are untouched by this knob.
#pragma unroll
            for (int tau = 0; tau < FATTN_VT16; ++tau) {
                const fattn_ARegs vlo =
                    fattn_load_tile(&s_kv[fattn_v_addr(32 * tau + fattn_ab_row(lane), 0)]);
                const fattn_ARegs vhi =
                    fattn_load_tile(&s_kv[fattn_v_addr(32 * tau + 16 + fattn_ab_row(lane), 0)]);
                out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vlo, p, out.t[tau], 0);
                out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vhi, p, out.t[tau], 1);
            }
        } else {
#pragma unroll
            for (int t = 0; t < FATTN_VTILES; ++t) {
                const fattn_ARegs first =
                    fattn_load_tile(&s_kv[fattn_v_addr(16 * t + fattn_ab_row(lane), 0)]);
                out.t[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(first, p, out.t[t]);
            }
        }
        // ---- THE ROUND'S FOURTH BARRIER, and the reason this bet is not a free lunch. At ONE warp per block
        // the three barriers above are the whole protocol, because there is no other wavefront to race: the
        // aliased V stage is read by the P*V leg and then overwritten by the NEXT round's K fill, and within
        // one wavefront those two are ordered by the `s_waitcnt lgkmcnt(0)` the compiler inserts. With three
        // warps on one shared stage that is a RACE — warp A can be writing round k+1's K into `s_kv` while
        // warp B is still reading round k's V out of it, which is §13's stage-lifetime defect with the polarity
        // reversed (a reader left behind, not a writer too early), and the measured signature is exactly what
        // it produces: every covered row of the wave comes back NaN. The fourth barrier is what makes "V
        // consumed" a block fact before any warp may re-purpose the buffer.
        __syncthreads();                                       // barrier 6: V consumed (all three warps)
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
    // names each row-slot exactly once, because lane r's `me.row` is rb*16 + r — per ROW-BLOCK, which is what
    // makes three warps publish 48 distinct rows rather than the same sixteen three times.
    if (me.covered && lane < FATTN_MROWS) {
        float* pp = fattn_part_slot(a, me.row, head0);
        pp[0] = m;
        pp[1] = l;
    }
    // The transpose stage is this warp's OWN per-warp stage — the dead Q stage in the default spelling, and the
    // only thing living there in `unstaged8`. It has to be per warp either way: one warp's staging is bigger
    // than a third of the shared K/V stage, so three warps transposing through `s_kv` would write each other's
    // rows — §20.1's aliasing bug exactly, and it would then need the warps serialised through the epilogue to
    // be correct. Per warp it costs nothing extra (the Q stage is dead the moment the round loop ends) and the
    // three transposes run independently. The barriers inside stay block-wide because they are cheaper to reason
    // about than to re-derive per warp, and every warp executes the same sequence of them.
    fattn_transpose_out<kQUnstaged8, kPvF16>(reinterpret_cast<float*>(s_warp + w * warp_stage_halves),
                                     out, a, rb, head0, row_mask, lane);
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
                        "its own lane, so the rows of one fattn block would read lanes that other rows "
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
// `__launch_bounds__(THREADS)`. A 32-thread block was allowed 255 VGPRs (the per-thread architectural cap,
// which is a different limit from the per-CU register file and the one that actually bound a 16-tile F32
// accumulator plus a 16-tile operand set), and a 96-thread block is allowed the same 255 because
// 196,608/96 = 2048 is far above the cap — so the declaration is what lets the form exist at all, and
// widening the block from 32 to 96 threads does NOT buy any register headroom. That is the arithmetic Bet 2
// had to check before believing the skeleton's zero-spill claim.
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave(CoreArgs a) {
    attn_fattn_body<false, false, false>(a);
}

// The A4-L1 side of the roll knob: the SAME body, the SAME grid, the SAME declared LDS, the second score-roll
// spelling. A separate `__global__` rather than a runtime branch inside one kernel, because a runtime branch
// would raise the register count of the default side too (see the note above `attn_fattn_body`). The A/B
// reads `vgpr_count` out of both of these note blocks in the same window
// (`prototype/attn_percost/a4_l1_note_read.py`); `engine/build.sh` is not bit-reproducible on this box and a
// canary that lives inside the binary under test is blind to a stale binary (AGENTS, 2026-10-17).
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave_two(CoreArgs a) {
    attn_fattn_body<true, false, false>(a);
}

// THE G1 SIDES of the qstage knob (bet A6-G1, `--attn-fattn-qstage unstaged8`): Q read from global, the
// epilogue staged FATTN_TRG rows at a time, declared LDS 20,480 B instead of 32,768 B — 3 blocks/CU instead of
// 2, i.e. 9 wf/CU instead of 6, at 185 VGPR / spill 0 / scratch 0 on the compile-side read
// (`prototype/attn_percost/A6-result.md` §1). Same grid, same block extent, same launch count, and owed BYTE
// identity with the shipped pair because it moves staging addresses and operand provenance only
// (`prototype/attn_percost/A6-G1-impl.md`, the device proof). The `_two_` variant is the same candidate under
// the two-accumulator roll: the two knobs are orthogonal, every combination is a real kernel, and no combination
// is a route no gate can reach.
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave_u8(CoreArgs a) {
    attn_fattn_body<false, true, false>(a);
}
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave_two_u8(CoreArgs a) {
    attn_fattn_body<true, true, false>(a);
}

// THE F16 P*V ACCUMULATOR SIDES of the third `--attn-fattn` knob (`--attn-fattn-pvf16`,
// `prototype/attn_percost/A10-F16-IMPL.md` D1–D7). Same body, same grid, same block, same declared LDS, same
// score leg and same P; the accumulator is 64 registers instead of 128 and the P*V leg issues two
// `v_wmma_f16_16x16x16_f16` per 32 output dims instead of one `v_wmma_f32_16x16x16_f16` per 16. It is NOT a
// faster form — A10-RF-LADDER §7 times the two atoms at the same ~34 clk for the same coverage, and D1 prices
// 16 issues per round either way — so the expected throughput effect of the knob is ~0 and its purpose is the
// 64 registers it frees for the Q-in-registers spelling D7 says cannot fit today. One `__global__` per
// combination for the same reason as the two knobs above it: a runtime branch inside one kernel would raise
// the register count of the DEFAULT side, and "the default is inert" has to remain a machine-code compare
// rather than a promise. Unlike the qstage/roll pair this knob carries no bit-identity claim: an F16
// accumulator is a different format, and the judge is `engine/gate.py`'s existing cos/meanrel bands through
// `A4-BET.md` — no new band, no widening, no `--ack-spill`.
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave_pvf16(CoreArgs a) {
    attn_fattn_body<false, false, true>(a);
}
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave_two_pvf16(CoreArgs a) {
    attn_fattn_body<true, false, true>(a);
}
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave_u8_pvf16(CoreArgs a) {
    attn_fattn_body<false, true, true>(a);
}
__global__ void __launch_bounds__(FATTN_THREADS) attn_fattn_wave_two_u8_pvf16(CoreArgs a) {
    attn_fattn_body<true, true, true>(a);
}

// The launcher's two host helpers live in the op header beside `tile_rtiles_of`: `fattn_rtiles_of` (this
// form's row-block rule, 16 query rows per block) and `fattn_covers` (the ONE coverage predicate both sides
// of the split call, so the VALU wave's early-out and this kernel's coverage test cannot become two rules).

}  // namespace engine::kernels
