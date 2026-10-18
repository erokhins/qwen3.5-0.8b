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
 * THE LDS BUDGET, and why there is no score staging
 *   s_q   16 rows x 256 dims x 2 B  = 8192 B   (Q rows, F16: a WMMA operand is F16 even when global is BF16)
 *   s_kv  16 lanes x 256 dims x 2 B = 8192 B   (K for the score leg, then ALIASED by V for the P*V leg)
 *   total                           = 16384 B  -> 4 blocks/CU on this part's 64 KiB, driver-confirmed
 * The online state lives in registers instead of the `s_sc` array the shipped form stages into: 16 rows x
 * 512 lanes of BF16 scores would be 16 KiB per head at MROWS=16, which does not fit beside the three
 * stages, and it is exactly why the one-pass form cannot be ported here at all. The accumulator is 16
 * tiles x 8 F32 per lane = 128 VGPR, which is what puts the whole kernel at 193 VGPR and 4 blocks/CU —
 * `evidence/occupancy-fattn-after-swap.txt`, three-control canary included.
 *
 * ONE ADDRESS EXPRESSION PER STAGED OBJECT, writer and reader alike (§20.1). `fattn_q_addr`,
 * `fattn_kv_addr` and `fattn_v_addr` are the only way any of the three stages is indexed, and the partial
 * slot expression is the shipped one, verbatim, keyed by the FULL head, so `attn_combine` needs no branch
 * for this form either.
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
 *     (dim = 16t + 2l + t/16), while `part` wants 8 CONTIGUOUS dims per lane. Staged through the dead K/V
 *     stage in two halves; see `fattn_transpose_out`.
 *
 * STATUS: WRITTEN, NOT ADMITTED. Nothing in this file has been numerically admitted on device. The
 * admission sequence is `prototype/attn_wmma/run_fattn_admission.sh` (written, not run) and the timing
 * window is `prototype/attn_wmma/run_fattn_burst_window.sh` (written, not run, main-agent only). Until
 * those are green this path is reachable only by `--attn-fattn 1`, which is not the default anywhere.
 *
 * CENSUS (compile-only, same recipe as `engine/attn_slim_kernels.h`'s header note):
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

// The three staged objects, one expression each (§20.1: two self-consistent layouts joined by a branch are
// invisible to both sides — that is §19's aliasing bug, and this file has three layouts, not two).
__device__ __forceinline__ int fattn_q_addr(int q, int d) { return q * DH + d; }      // [row][dim]
__device__ __forceinline__ int fattn_kv_addr(int kv, int d) { return kv * DH + d; }   // [lane][dim]
__device__ __forceinline__ int fattn_v_addr(int d, int kv) { return d * FATTN_MLANES + kv; }  // [dim][lane]
static_assert(FATTN_MROWS * DH * 2 + FATTN_MLANES * DH * 2 == 16384,
              "Q stage + aliased K/V stage is exactly 16 KiB, which is the 4-blocks/CU boundary");

__device__ __forceinline__ float fattn_bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fattn_fb(float f) { return hip_bfloat16(f); }
// The shipped score round: the dot's F32 result is rounded through BF16, scaled (2^-4, so the scale is
// exact and the staging can stay 2 bytes), and rounded again. Same expression, same association, as
// `attn_slim_wave`'s `fb(bf(fb(dot)) * SCALE)` — association is load-bearing on this device.
__device__ __forceinline__ float fattn_score_round(float dot) {
    return fattn_bf(fattn_fb(fattn_fb(dot) * SCALE));
}

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

__device__ __forceinline__ void fattn_store_tile(__half* p, const fattn_ARegs& r) {
    uint4 lo, hi;
    __half* a = reinterpret_cast<__half*>(&lo);
    __half* b = reinterpret_cast<__half*>(&hi);
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] = __ushort_as_half((unsigned short)r[e]);
#pragma unroll
    for (int e = 0; e < 8; ++e) b[e] = __ushort_as_half((unsigned short)r[8 + e]);
    *reinterpret_cast<uint4*>(p) = lo;
    *reinterpret_cast<uint4*>(p + 8) = hi;
}

// The row a lane's C register belongs to, and the column its l-th value belongs to. BOTH come from the
// measured law, and `get_i`/`get_j` here are the same functions for the C tile whatever the operand order
// was — it is the OPERAND slots that the measurement corrected, so these two lines are written once and
// used by every consumer (drain, mask, fold, epilogue) instead of being re-derived at each site.
__device__ __forceinline__ int fattn_c_row(int lane) { return lane & 15; }
__device__ __forceinline__ int fattn_c_col(int lane, int l) { return 2 * l + (lane >> 4); }
// The operand map: lane t holds the 16 contiguous halves of row t%16, the run along the dot index.
__device__ __forceinline__ int fattn_ab_row(int lane) { return lane & 15; }

// C (8 F32) -> the next WMMA's operand (16 F16), by the `get_half2` exchange: each lane owns one parity of
// its row's 16 columns, so the partner half-warp's values are collected with one shuffle per register.
// MEASURED, and the measurement is narrower than the check looked: `cb_no_shfl` (the shuffle replaced by the
// lane's own value) leaves the OUTPUT exactly right and only the operand dump wrong, because each half-warp
// supplies its own parity of the KV axis and the exchange's job is only to make BOTH parities correct in
// BOTH half-warps. The blind spot is named in `prototype/attn_wmma/check_fattn_cb.py`'s header; do not
// "simplify" this by deleting the shuffle on the strength of that result — the operand layout is what the
// next tile reads, and `cb_kv_both_relabel` is the case that keeps the claim honest.
__device__ __forceinline__ fattn_ARegs fattn_p_to_operand(const float pexp[8], int lane) {
    fattn_ARegs p;
    const int i = lane >> 4;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        float tmp[2];
        tmp[i] = pexp[l];
        tmp[i ^ 1] = __shfl_xor_sync(FULL_MASK, pexp[l], 16, warpSize);
        p[2 * l + 0] = (short)__half_as_ushort(__float2half_rn(tmp[0]));
        p[2 * l + 1] = (short)__half_as_ushort(__float2half_rn(tmp[1]));
    }
    return p;
}

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

// The accumulator transpose. The P*V tiles hold columns PARITY-SPLIT across the half-warps
// (dim = 16t + 2l + lane/16), while `part` wants 8 CONTIGUOUS dims per lane (`part[2 + lane*8 + u]`, which
// is what `attn_combine` reads and is not negotiable without editing the merge). So the output is staged in
// F32 through the DEAD K/V stage, half of DH at a time: 16 rows x 128 dims x 4 B = 8192 B, which is exactly
// that stage's size — reusing it is what keeps the block at 16 KiB and at 4 blocks/CU. A separate output
// array would have been 8 KiB ON TOP of the stages, i.e. 2 blocks/CU and a different occupancy story.
__device__ __forceinline__ void fattn_transpose_out(__half* stage_as_f32,
                                                    const fattn_CRegs out[FATTN_VTILES], float* pp,
                                                    int lane) {
    float* stage = reinterpret_cast<float*>(stage_as_f32);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();                     // the previous half's reads are done before it is rewritten
#pragma unroll
        for (int t = half * (FATTN_VTILES / 2); t < (half + 1) * (FATTN_VTILES / 2); ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int d = 16 * t + fattn_c_col(lane, l) - half * (DH / 2);
                stage[fattn_c_row(lane) * (DH / 2) + d] = out[t][l];
            }
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int d = half * (DH / 2) + lane * 8 + u;   // the shipped layout: 8 contiguous dims
            pp[2 + d] = stage[fattn_c_row(lane) * (DH / 2) + lane * 8 + u];
        }
    }
    __syncthreads();                         // writers of the next round must not race the last read
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
    __shared__ __align__(16) __half s_q[FATTN_MROWS * DH];
    __shared__ __align__(16) __half s_kv[FATTN_MLANES * DH];   // K, then V, then the F32 output transpose
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    const int nw = (int)(blockDim.x / warpSize);
    // Widest row first, exactly as `attn_slim_body` does it: the dispatch tail is where the long rows are.
    const int rt = (int)gridDim.z - 1 - (int)blockIdx.z;
    const int kv_read = tile_kv_of((int)blockIdx.x, (int)a.neg);
    const int kv_write = (int)blockIdx.x;
    const int head0 = ATTN_TILE_G * kv_write + w;      // warp w owns query head h of the group

    // ---- this warp's rows, and whether ANY of them is this kernel's. `fattn_row_of` depends on the ROW
    // and on `blockIdx.y`, never on the warp, so every warp computes the same covered set and the early
    // return below is BLOCK-uniform — which is what makes the barriers inside the round legal. An early
    // return that diverged between warps would leave the survivors at a barrier no one else reaches.
    const FattnRow me = fattn_row_of(a, rt, lane);
    if (!__ballot_sync(FULL_MASK, me.covered ? 1 : 0)) return;

    // ---- Q stage: this block's 16 rows of each warp's head.
    for (int qr = w; qr < FATTN_MROWS; qr += nw) {
        const int row = rt * FATTN_MROWS + qr;
        if (row >= nrows_of(a)) continue;
        fattn_stage_row(&s_q[fattn_q_addr(qr, 0)],
                        a.qrope + (size_t)ROW_H * row + (size_t)head0 * DH, lane);
    }
    __syncthreads();                                     // barrier 1: Q published

    // The round range: the union of the covered rows' slices. Rows in one tile differ in lane count by at
    // most FATTN_MROWS - 1 and share `s_row`, so their spans differ by at most 1 and the union costs at
    // most one extra round — stated because the shipped form pays nothing for a union.
    const int lo_all = fattn_warp_min(me.covered ? me.blo : 0x7fffffff);
    const int hi_all = fattn_warp_max(me.covered ? me.bhi : 0);

    float m = -1e30f, l = 0.f;
    fattn_CRegs out[FATTN_VTILES];
#pragma unroll
    for (int t = 0; t < FATTN_VTILES; ++t) out[t] = fattn_CRegs{};

    for (int base = lo_all; base < hi_all; base += FATTN_MLANES) {
        // ---- K stage for this round. `j` is the ABSOLUTE lane index; the stage slot is ROUND-relative
        // (§13/§14: indexing a stage by a slice-absolute row walks off the array and corrupts its
        // neighbour). Out-of-range lanes clamp to a lane this block can read at all; their cells are
        // masked below, so the value is never used.
        for (int slot = w; slot < FATTN_MLANES; slot += nw) {
            int j = base + slot;
            if (j >= hi_all) j = hi_all - 1;
            fattn_stage_row(&s_kv[fattn_kv_addr(slot, 0)],
                            a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH, lane);
        }
        __syncthreads();                                 // barrier 2: K published

        // ---- the score leg: FIRST operand = K rows (KV lanes), SECOND = Q rows. Slot order is the
        // measured correction; swapping these two lines is the transposed-score regression, and
        // `check_fattn_cb.py`'s cb_score_slot_swap case is the required-fail control for it.
        fattn_CRegs acc{};
#pragma unroll 1
        for (int d0 = 0; d0 < DH; d0 += 16) {
            const fattn_ARegs first = fattn_load_tile(&s_kv[fattn_kv_addr(fattn_ab_row(lane), d0)]);
            const fattn_ARegs second = fattn_load_tile(&s_q[fattn_q_addr(fattn_ab_row(lane), d0)]);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(first, second, acc);
        }

        // ---- mask, fold, exp. The bound is per (row, round): a cell is live iff its lane index is inside
        // THIS ROW's slice. Two compares per cell, which is the cost the prototype's one-round form never
        // had to pay and the census of this file must show.
        float pexp[8];
        float rm = -1e30f;
#pragma unroll
        for (int l2 = 0; l2 < 8; ++l2) {
            const int col = fattn_c_col(lane, l2);       // this cell's lane index within the round
            const int j = base + col;
            const bool live = me.covered && j >= me.blo && j < me.bhi;
            pexp[l2] = live ? fattn_score_round(acc[l2]) : -1e30f;
            rm = fmaxf(rm, pexp[l2]);
        }
        rm = fmaxf(rm, __shfl_xor_sync(FULL_MASK, rm, 16, warpSize));   // one cross-lane op per row/round
        const float mn = fmaxf(m, rm);
        const float rescale = expf(m - mn);
        float ls = 0.f;
#pragma unroll
        for (int l2 = 0; l2 < 8; ++l2) {
            pexp[l2] = expf(pexp[l2] - mn);
            ls += pexp[l2];
        }
        l = l * rescale + ls;
        m = mn;
#pragma unroll
        for (int t = 0; t < FATTN_VTILES; ++t) {
#pragma unroll
            for (int l2 = 0; l2 < 8; ++l2) out[t][l2] *= rescale;   // the 128 F32 multiplies per round
        }

        // ---- V stage, aliased over the K stage (every K reader has passed barrier 2 and the fold above).
        // V is staged TRANSPOSED: the P*V operand whose rows are the output dims needs the KV run
        // contiguous — 32 narrow stores per lane per round here against the 256 narrow reads of the
        // alternative, which is the trade §20.5 of `prototype/attn_decode/results.md` names, not hides.
        __syncthreads();                                 // barrier 3: K consumed
        for (int kv = w; kv < FATTN_MLANES; kv += nw) {
            int j = base + kv;
            if (j >= hi_all) j = hi_all - 1;
            const hip_bfloat16* vsrc =
                a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
            const uint4 u = *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            const hip_bfloat16* vv = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int d = lane * 8 + t;
                s_kv[fattn_v_addr(d, kv)] = __float2half_rn(fattn_bf(vv[t]));
            }
        }
        __syncthreads();                                 // barrier 4: V published

        // ---- the P*V leg: FIRST = V (rows are output dims), SECOND = P (rows are query rows), 16
        // independent accumulator chains so the rescale above is the only serial dependency.
        const fattn_ARegs p = fattn_p_to_operand(pexp, lane);
#pragma unroll
        for (int t = 0; t < FATTN_VTILES; ++t) {
            const fattn_ARegs first =
                fattn_load_tile(&s_kv[fattn_v_addr(16 * t + fattn_ab_row(lane), 0)]);
            out[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(first, p, out[t]);
        }
    }

    // ---- epilogue: the SHIPPED partial slot expression, verbatim, keyed by the full head, so
    // `attn_combine` needs no branch for this form (§20.1). An uncovered row writes NOTHING here, which is
    // what makes `attn_slim_wave`'s write of that same slot the only one.
    if (!me.covered) return;
    float* pp = a.part + (((size_t)me.row * N_Q + head0) * a.split + blockIdx.y) * (DH + 2);
    if (lane == 0) {
        pp[0] = m;
        pp[1] = l;
    }
    fattn_transpose_out(s_kv, out, pp, lane);
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

__global__ void attn_fattn_wave(CoreArgs a) { attn_fattn_body(a); }

// The launcher's two host helpers live in the op header beside `tile_rtiles_of`: `fattn_rtiles_of` (this
// form's row-block rule, 16 query rows per block) and `fattn_covers` (the ONE coverage predicate both sides
// of the split call, so the VALU wave's early-out and this kernel's coverage test cannot become two rules).

}  // namespace engine::kernels
