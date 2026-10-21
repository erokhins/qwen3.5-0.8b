/*
 * attn_wide_kernels.h — the A12 WIDE attention wave block, moved into the engine behind
 * `--attn-fattn-wide` (0 = the 16-row wave that shipped before A13, 1 = this block, 2 = the A13 port of it,
 * which is the default). One block = 32 query positions x the 4 heads of ONE KV group = 128
 * query row-slots = 8 waves, a 64-position K/V stage, Q converted to F16 once per block and held in
 * registers, and an F16 P*V accumulator.
 *
 * SOURCE: `prototype/attn_wide/attn_wide.h`, moved rather than rewritten, which is the graduation A12's
 * own sequencing named (`prototype/attn_percost/A12-WIDE-BET.md` §3). The three duplicated laws the
 * prototype restated because it had to compile alone — `wide_c_row`/`wide_c_col`/`wide_h_dim` — are GONE:
 * this file calls the engine's single spellings (`fattn_c_row`, `fattn_c_col`, `fattn_h_dim`,
 * `fattn_ab_row`, `fattn_score_round`, `fattn_load_tile`, `fattn_load_q_tile`, `fattn_stage_row`,
 * `fattn_h_get`, `fattn_ARegs`/`fattn_CRegs`/`fattn_HRegs`/`fattn_H2Regs`, `FattnO16`) and takes
 * `CoreArgs`, so the fragment maps and the score round exist in exactly one place each. A relabelled
 * fragment map is a permuted row of the output, and §29 of `progress/2026-10-21/performance-status-2026-10-21.md` is what that
 * costs; nothing below re-derives a map at a use site.
 *
 * INCLUDED AFTER `engine/attn_fattn_kernels.h`, from the one TU that compiles both
 * (`engine/kernels_attn.cpp`). That order is not cosmetic: it reads that header's anonymous-namespace
 * helpers, and a second TU including it would get its own copy of them.
 *
 * WHAT THIS BLOCK DOES THAT THE SHIPPING FORM DOES NOT, and what it therefore stops doing:
 *   * NO LANE SPLIT. A block covers a row's WHOLE causal prefix, in rounds of 64 positions. So it writes
 *     no `part` partial and `attn_combine` has nothing to merge — the launcher skips that launch (see the
 *     routing note in `engine/kernels_attn.cpp`), and this kernel finishes each row itself through
 *     `finish_head`, the engine's §05.6 epilogue, rather than re-spelling normalise-then-gate.
 *   * NO COVERAGE RULE. `--attn-fattn` splits rows with `fattn_covers`: the WMMA form takes the streaming
 *     rows and `attn_slim_wave` keeps the one-pass rows, because a one-pass row is pinned bit-for-bit
 *     against the dumps. The wide form has no such predicate and cannot have a partial one — the timing
 *     rig priced it owning EVERY row of a pass (`prototype/attn_wide/time_wide.cpp`: "the wide form owns
 *     every row: it has no coverage rule to opt out of"), and a block that covered only some rows would
 *     leave the others to a kernel this launch does not run. So at knob 1 `attn_slim_wave` and the eight
 *     `attn_fattn_wave*` spellings are not launched at all, and rows at or below MAXSC move off the
 *     bit-exact one-pass route onto the wide form. THAT IS THE NUMERICALLY MOST SENSITIVE THING THIS KNOB
 *     DOES and it is not hidden: see the risk note below and `prototype/attn_wide/a12_engine_admission.py`.
 *   * THE SPILL BUDGET AND THE RESIDENCY ARE SPENT ON PURPOSE. <= 1,024 B/thread of private scratch is a
 *     budget here, not a taboo (llama's own kernel is at 900 B/thread), and 1 block/CU at 8 wavefronts is
 *     allowed (`A12-WIDE-BET.md` §1). The shipping path keeps all three protections; this file is the one
 *     place they are spent, and `wide_echo` prints the spill so a reader sees the spend.
 *
 * THE RISK THIS FORM NEWLY CARRIES, stated where the code is:
 *   1. A 64-position round RE-ASSOCIATES THE SOFTMAX relative to knob 0 at every row width, so there is NO
 *      bit-identity claim available anywhere — not on a multi-round row, and not on a row that fits in one
 *      round either, for a second independent reason: this form rounds the UN-normalised `exp` to F16 and
 *      divides at the end, while the one-pass route rounds the NORMALISED weight to BF16 before the P*V
 *      (§05.5's rule 4) and the shipping WMMA form's round width is 16, not 64. Knob 1 is therefore judged
 *      by the existing §10.1 cos/meanrel aggregates exactly as `--attn-fattn-roll two` and
 *      `--attn-fattn-pvf16` were (`prototype/attn_percost/A4-BET.md`; AGENTS' "numeric format: delegated"
 *      rule: a green gate set IS the decision procedure). No new band, no widened band, no loosened check.
 *   2. THE PER-ROUND RESCALE IS LOAD-BEARING, not tidiness. An F16 accumulator discards an add once the
 *      addend falls below half a ULP of the running value, and the counterfactual — the same sweep with the
 *      fold removed — is measured twice, once per round width, because the two shapes are different sums:
 *        * the shipping 16-position arm, 53 head-rows: first add dropped at round 1 / 13 / 99
 *          (worst / median / best), most adds dropped by round 53 / 2,035 / 3,552 — `A10-PVF16.md` §4;
 *        * THIS 64-position arm: first add dropped at round 1 in the worst head-row, and the accumulator no
 *          longer finite in fp16 by round 1721 — `prototype/attn_wide/a12_wide_mirror.py` section 4.
 *      A 64-position round carries four times the addends of a 16-wide one into a single accumulator step,
 *      which is the direction that makes the fold more load-bearing here than in the shipping form, not
 *      less. It sits before the round's add, on the accumulator's critical path, where the shipping kernel
 *      keeps it.
 *
 * WHAT IS NOT PRICED HERE, and so is not claimed: the split/combine round-trip (this form writes none),
 * and any interleaving with `attn_combine`, which the timing rig could not price because its wide arm had
 * no combine to price. The epilogue below pays §05.6's gate per element, which the rig's O-buffer
 * epilogue did not — that cost is real, it is one expf + one divide per output dim (exactly what
 * `attn_combine` pays on the shipping route for a streaming row), and it makes this arm's bill slightly
 * HEAVIER than the rig's, never lighter.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>   // the WMMA operands are F16 whatever global is
#include <hip/hip_runtime.h>

#include <string>

namespace engine::kernels {
namespace {

// ---- the shape --------------------------------------------------------------------------------------
// The WMMA tile dimensions are NOT restated here: `FATTN_MROWS` (M = 16 row-slots a wave) and
// `FATTN_MLANES` (N = 16 KV positions a C tile) are one fact with one home, in
// `engine/attn_fattn_kernels.h`, and this shape is built out of them. What is A12's own is how many waves
// a block carries, how deep a round's stage is, and where Q lives.
#define WIDE_WARPS 8              // waves per block (A12's R1). 8 x 16 row-slots = 128 rows per block.
#define WIDE_THREADS (WIDE_WARPS * 32)
#define WIDE_STAGE 64             // KV positions per staged round (A12's R3). 4 C tiles.
#define WIDE_MROWS FATTN_MROWS                        // the WMMA M dimension, from its one home
#define WIDE_LGROUPS (WIDE_STAGE / FATTN_MLANES)      // C tiles per round: 4
#define WIDE_POS_PER_WARP (WIDE_MROWS / GQA)          // 4 query positions per wave (GQA=4 heads each)
#define WIDE_POS_PER_BLOCK (WIDE_WARPS * WIDE_POS_PER_WARP)   // 32 query rows per BLOCK
#define WIDE_ROWS_PER_BLOCK (WIDE_WARPS * WIDE_MROWS)         // 128 row-slots per BLOCK
#define WIDE_TRG 4                // row-slots per epilogue transpose pass (see the coverage assert)
#define WIDE_TROW (DH / 2)        // floats per staged row per pass (half of DH)

// THE ROW RULE, one home. `FATTN_ROWS_PER_BLOCK`'s lesson is the reason this is a function and not a
// number written at the launch site: that fact lived in two headers, one half got committed alone, and the
// tree did not build (`AGENTS.md`, the "two headers, one geometry fact" row). The launcher's grid depth
// and this kernel's `blockIdx.z` reading both come through here, so they cannot disagree, and the asserts
// below are the use-site half of the same guard.
__device__ __host__ inline int wide_rtiles_of(int nrows) {
    return (nrows + WIDE_POS_PER_BLOCK - 1) / WIDE_POS_PER_BLOCK;
}

static_assert(WIDE_THREADS == WIDE_WARPS * 32, "the block extent is the wave count spelled in threads");
static_assert(WIDE_MROWS == FATTN_MROWS, "one WMMA M dimension, spelled once, in attn_fattn_kernels.h");
static_assert(WIDE_STAGE % FATTN_MLANES == 0 && WIDE_LGROUPS * FATTN_MLANES == WIDE_STAGE,
              "a stage must be a whole number of C tiles, and the tile count must reproduce the stage");
static_assert(WIDE_MROWS % GQA == 0,
              "a wave's row-slots must tile whole query positions of ONE KV group, or a wave would share "
              "no staged K/V with its neighbours");
static_assert(WIDE_POS_PER_BLOCK * GQA == WIDE_ROWS_PER_BLOCK,
              "the block's row-slots are its query positions times the GQA heads of one group; the grid "
              "is built in positions and the block is filled in row-slots, so these must be one fact");
static_assert(WIDE_MROWS % WIDE_TRG == 0, "the transpose row groups must tile a wave's row-slots");

// The two staged layouts of the K/V buffer, one expression each — the shipping rule that a staged object
// has ONE address expression shared by its writer and its reader (§20.1 of `prototype/attn_decode/
// results.md`). K is row-major ([lane][dim]), which is the shape the score leg's first operand wants and
// is the SAME expression the shipping form uses (`fattn_kv_addr`); V is transposed ([dim][lane]) and its
// row stride is THIS shape's stage depth, so it needs its own spelling — a second layout of a different
// staged object is not a duplicate of `fattn_v_addr`, which strides by `FATTN_MLANES` = 16. Unpadded on
// purpose: `A11-attribution.md` §5 measured the unpadded stride and priced a pad at "a few-percent lever",
// deliberately not taken, so the ladder prices the shape and not the pad.
__device__ __forceinline__ int wide_v_addr(int d, int j) { return d * WIDE_STAGE + j; }
// The epilogue stage: [row-slot within this pass][dim within this half of DH].
__device__ __forceinline__ int wide_tr_addr(int row_in_pass, int d) { return row_in_pass * WIDE_TROW + d; }

// The LDS the block declares, ONE buffer with TWO layouts separated by barriers, spelled as a growth of one
// array (the shipping `fattn_warp_stage_halves` note says why: `__shared__` arrays SUM, and a second array
// would move the declared LDS, which is the residency claim the number is about; and a pad array nobody
// reads is dead code that the allocator eliminates — AGENTS' `a11_alloc_check` row). Layout 1 is `s_kv`:
// STAGE positions x DH halves, K row-major then V transposed aliased over it. Layout 2 is `s_out`: the
// F32 output transpose, one slice PER WAVE (a shared output slice is §20.1's aliasing bug).
constexpr int wide_lds_kv_halves() { return WIDE_STAGE * DH; }
constexpr int wide_lds_out_halves() { return WIDE_WARPS * WIDE_TRG * WIDE_TROW * 2; }   // F32 = 2 halves
constexpr int wide_lds_halves() {
    return wide_lds_kv_halves() > wide_lds_out_halves() ? wide_lds_kv_halves() : wide_lds_out_halves();
}
static_assert(WIDE_TRG * WIDE_TROW == 32 * 16,
              "one pass stages WIDE_TRG rows x half of DH and 32 lanes publish 16 floats each, so the "
              "publish loop covers the stage exactly. Coverage is a fact, not a reading: §18.1's all-NaN "
              "was a publish loop that touched 16 of every row's dims while satisfying every bound assert");
static_assert(wide_lds_halves() * 2 == 32768,
              "the shipped shape declares exactly 32,768 B, which is what `wide_echo`'s `lds=` field is "
              "expected to read back from the loaded code object. Below the 48 KiB static ceiling, so no "
              "dynamic-allocation opt-in is needed; 2 blocks/CU on LDS arithmetic at 8 warps each");
static_assert(wide_lds_halves() * 2 <= 65536, "the architectural LDS per workgroup is 64 KiB (KFD "
                                              "`lds_size_in_kb 64`, read from the device, not quoted)");

// ---- the body ---------------------------------------------------------------------------------------
// Every loop that touches a register array is `#pragma unroll` without exception, and that is not style: a
// runtime-indexed array is an alloca, and an alloca inside the round loop is the spill this repo has met
// twice. It bites harder here because Q is an array of `FATTN_KTILES` tuples indexed by the roll's `t`:
// rolling that roll would put Q in scratch, which is the traffic this shape exists to delete.
__device__ __forceinline__ void attn_wide_body(const CoreArgs a) {
    __shared__ __align__(16) __half s_mem[wide_lds_halves()];
    __half* const s_kv = s_mem;
    // This wave's OWN slice of the same bytes, reinterpreted as F32. Legal only because the round loop's
    // last barrier ("V consumed") is exactly the condition the epilogue needs before re-purpose.
    float* const s_out = reinterpret_cast<float*>(s_mem) + (threadIdx.x >> 5) * (WIDE_TRG * WIDE_TROW);

    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    const int nw = (int)(blockDim.x >> 5);
    // One KV GROUP per block (grid.x = N_KV), so the block's 4 heads share every staged K/V byte. The read
    // group comes through `tile_kv_of`, the tiled form's group rule, so an injection-aware read that the
    // tiled wave honours is honoured here too rather than paraphrased as `blockIdx.x`.
    const int kv_read = tile_kv_of((int)blockIdx.x, a.neg);
    const int head0 = (int)blockIdx.x * GQA;
    // Widest rows first, as both shipped wave forms do it: the dispatch tail carries the long rows.
    const int rt = (int)gridDim.z - 1 - (int)blockIdx.z;
    const int row0 = rt * WIDE_POS_PER_BLOCK;
    const int nrows = nrows_of(a);
    if (row0 >= nrows) return;                     // the grid is exact; stated, not assumed
    const int rowhi = nrows < row0 + WIDE_POS_PER_BLOCK ? nrows : row0 + WIDE_POS_PER_BLOCK;
    const int warp_row0 = row0 + w * WIDE_POS_PER_WARP;

    // The round range, BLOCK-UNIFORM by construction. This is the piece the prototype got for free from
    // `pos0` and the engine has to compute: row `qrow` may attend `lanes_of(a, qrow)` lanes (§06.3), which
    // advances with the row, so the block's bound is its LAST row's. `lanes_of` is the engine's own
    // predicate rather than a paraphrase of it, and `attn_prep`'s append has already landed in the cache
    // before this launch starts, so reading lanes up to that bound reads bytes somebody wrote. The
    // causal overhang this pays is bounded by WIDE_POS_PER_BLOCK = 32 positions per block — the shipping
    // WMMA form pays 48 — and per-row masking below is what absorbs it.
    const int hi = lanes_of(a, rowhi - 1);
    if (hi <= 0) return;    // block-uniform, and before any barrier, so no wave is left holding one

    // This lane's row-slot: (lane & 15) of the wave's 16, which is WIDE_POS_PER_WARP query positions x the
    // GQA heads of the group. `row_ok` is the tail test; `qrow_read` is the clamp that keeps a Q read inside
    // the buffer for a row-slot that does not exist — not cosmetic, because at 128 row-slots per block a
    // whole WAVE can be past `nrows`, where `nrows - 1` alone would index rows before the buffer.
    const int rslot = fattn_c_row(lane);
    const int rrow = warp_row0 + rslot / GQA;
    const int rhead = head0 + rslot % GQA;
    const bool row_ok = rrow < nrows;
    const int qrow_read = row_ok ? rrow : (rowhi - 1);
    const hip_bfloat16* const qrow = fattn_q_row(a, qrow_read, rhead);
    // The per-row causal bound, read off the engine's rule once per lane rather than re-derived per cell.
    const int rlanes = row_ok ? lanes_of(a, rrow) : 0;

    // Q resident: FATTN_KTILES tuples x 8 VGPR = 128 registers per lane, converted ONCE PER LAUNCH. The two
    // half-warps load the same tuple (lane t and t^16 hold the same row-slot), so this reads 16 KiB per wave
    // to place 8 KiB of distinct Q — the same 2x duplication the shipping form's staged Q read has, paid
    // once per launch here instead of once per round. This is the whole traffic case of the shape: the
    // shipping form re-reads Q at 34.85 B per causal touch, 898 GB per pass at 32,768 lanes.
    fattn_ARegs qv[FATTN_KTILES];
#pragma unroll
    for (int t = 0; t < FATTN_KTILES; ++t) qv[t] = fattn_load_q_tile(qrow + 16 * t);

    float m = -1e30f, l = 0.f;
    FattnO16 out;      // the F16 accumulator: 8 tiles of 16 halves = 64 VGPR/lane (A12's precondition)
#pragma unroll
    for (int tau = 0; tau < FATTN_VT16; ++tau) out.t[tau] = fattn_HRegs{};

    for (int base = 0; base < hi; base += WIDE_STAGE) {
        // ---- K into the stage: one row per wave per trip, all WIDE_WARPS waves covering the stage. A
        // position past the block's bound clamps to a position this block can read at all; the cell is
        // masked out below, so the bytes it cost are never used for anything.
        for (int slot = w; slot < WIDE_STAGE; slot += nw) {
            int j = base + slot;
            if (j >= hi) j = hi - 1;
            fattn_stage_row(&s_kv[fattn_kv_addr(slot, 0)],
                            a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH, lane);
        }
        __syncthreads();                                                 // barrier 1: K published

        // ---- the score roll: FIRST operand = K rows (KV positions), SECOND = Q rows (query positions),
        // so C is [query][kv] and not its transpose — the measured operand order
        // (`prototype/attn_wmma/fattn_map_probe.cpp`, pinned by `check_fattn_map.py`), used not re-derived.
        // WIDE_LGROUPS accumulators, one per 16-position slice of the stage, and ONE Q tuple per trip
        // feeding all four: that is the operand re-use the wide shape buys, 4 C tiles per Q fetch.
        // INITIALISED, for the reason `attn_fattn_body` spells: `__builtin_amdgcn_wmma_*` takes the running
        // accumulator as a live-IN operand, so an uninitialised first tile reads whatever the previous
        // dispatch left in those VGPRs — which came back as whole-row NaN (§29, D64).
        fattn_CRegs acc[WIDE_LGROUPS];
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) acc[L] = fattn_CRegs{};
#pragma unroll
        for (int t = 0; t < FATTN_KTILES; ++t) {
            const fattn_ARegs qt = qv[t];
#pragma unroll
            for (int L = 0; L < WIDE_LGROUPS; ++L) {
                const fattn_ARegs kt = fattn_load_tile(
                    &s_kv[fattn_kv_addr(FATTN_MLANES * L + fattn_ab_row(lane), 16 * t)]);
                acc[L] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kt, qt, acc[L]);
            }
        }

        // ---- mask + running max. The rounded score REPLACES acc in place: a second 32-float array would
        // be 32 more live registers per round for nothing, and the max is all that is read afterwards. A
        // cell is live iff this row-slot exists AND its KV position is inside this row's own causal prefix —
        // `j < rlanes`, the per-row bound, which is what absorbs the block's causal overhang.
        float rm = -1e30f;
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int j = base + FATTN_MLANES * L + fattn_c_col(lane, c);
                const bool live = row_ok && j < rlanes;
                acc[L][c] = live ? fattn_score_round(acc[L][c]) : -1e30f;
                rm = fmaxf(rm, acc[L][c]);
            }
        }
        // A row's columns are split across the lane pair (t, t^16), so the max needs one butterfly. The
        // shipping form's tenth lane-movement op — half a rowsum scales a row by ~2x and cosine cannot
        // see it, because cosine is scale-invariant.
        rm = fmaxf(rm, __shfl_xor_sync(FULL_MASK, rm, 16, warpSize));
        const float mn = fmaxf(m, rm);
        const float rescale = expf(m - mn);

        // ---- exp + the C->B operand exchange, ONE pass, four P tuples (one per 16-position slice). The
        // exchange is the engine's own, fused here for the reason `attn_fattn_body` gives: factoring it into
        // a helper that takes the exponentials by pointer put the array in scratch and cost 242 spilled
        // VGPRs the first time. The P written here is the UN-normalised `exp`, rounded to F16 — which is
        // what makes this form's weights differ from the one-pass route's, and is risk note 1 above.
        fattn_ARegs p[WIDE_LGROUPS];
        float ls = 0.f;
        const bool hi_half = (lane & 16) != 0;
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float e = expf(acc[L][c] - mn);
                ls += e;
                const float other = __shfl_xor_sync(FULL_MASK, e, 16, warpSize);
                p[L][2 * c + 0] = (short)__half_as_ushort(__float2half_rn(hi_half ? other : e));
                p[L][2 * c + 1] = (short)__half_as_ushort(__float2half_rn(hi_half ? e : other));
            }
        }
        l = l * rescale + ls;      // the deferred rowsum; legal because `rescale` is pair-uniform
        m = mn;

        // ---- THE PER-ROUND RESCALE, on the accumulator's critical path, BEFORE the round's add. See the
        // file header: with an F16 accumulator this is what keeps the first add from being dropped at all in
        // the worst head-row, and a 64-position round makes it more load-bearing than the shipping form's 16. `rh` is the SAME F32
        // factor the F32 form multiplies by, rounded to F16 once per round.
        {
            const _Float16 rh = (rescale);
#pragma unroll
            for (int tau = 0; tau < FATTN_VT16; ++tau) {
                fattn_H2Regs v = __builtin_bit_cast(fattn_H2Regs, out.t[tau]);
#pragma unroll
                for (int e = 0; e < FATTN_HELEMS; ++e) v[e] = v[e] * rh;
                out.t[tau] = __builtin_bit_cast(fattn_HRegs, v);
            }
        }

        // ---- V into the stage, ALIASED over K, with a barrier on each side (every K reader passed barrier
        // 1 and the roll above; every V reader gets barrier 3). Eight warps on one stage make the fourth
        // barrier mandatory rather than cheap: without it warp A writes round k+1's K while warp B is still
        // reading round k's V, and the measured signature of that race is every row of the wave NaN.
        __syncthreads();                                                 // barrier 2: K consumed
        for (int slot = w; slot < WIDE_STAGE; slot += nw) {
            int j = base + slot;
            if (j >= hi) j = hi - 1;
            const hip_bfloat16* vsrc = a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
            const uint4 u = *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            const hip_bfloat16* vv = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int t = 0; t < 8; ++t) s_kv[wide_v_addr(lane * 8 + t, slot)] = __float2half_rn(fattn_bf(vv[t]));
        }
        __syncthreads();                                                 // barrier 3: V published

        // ---- P*V: FIRST operand = V staged transposed (rows are output dims), SECOND = P (rows are query
        // rows), and the F16 pair is llama's shape (`mma.cuh:1553-1566`, read out of the shipped listing by
        // `A10-PVF16.md` §3 rather than assumed): two issues into the SAME D block, two different A
        // fragments (the low then the high 16 dims of the 32 the tile covers), the same P. 8 output tiles x
        // 4 KV slices x 2 halves = 64 issues per round against the shipping form's 16 per 16 positions, and
        // 8 independent accumulator chains, so the rescale above is the only serial dependency.
#pragma unroll
        for (int tau = 0; tau < FATTN_VT16; ++tau) {
#pragma unroll
            for (int L = 0; L < WIDE_LGROUPS; ++L) {
                const fattn_ARegs vlo = fattn_load_tile(
                    &s_kv[wide_v_addr(32 * tau + fattn_ab_row(lane), FATTN_MLANES * L)]);
                const fattn_ARegs vhi = fattn_load_tile(
                    &s_kv[wide_v_addr(32 * tau + 16 + fattn_ab_row(lane), FATTN_MLANES * L)]);
                out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vlo, p[L], out.t[tau], 0);
                out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vhi, p[L], out.t[tau], 1);
            }
        }
        __syncthreads();                                                 // barrier 4: V consumed
    }

    // THE DEFERRED ROWSUM COMBINE — `attn_fattn_body`'s tenth lane-movement op, and it has to happen before
    // the epilogue so no lane is ever shuffled against a partner that has already exited.
    l += __shfl_xor_sync(FULL_MASK, l, 16, warpSize);

    // ---- the epilogue. The accumulator's columns are parity-split across the lane pair and each lane holds
    // ONE row-slot, while the output wants that row's whole DH, so stage through LDS WIDE_TRG row-slots x
    // half of DH at a time and divide by `l` on the way out — widen the F16 accumulator to F32 FIRST, then
    // divide, which is the order the CPU mirror models and the order `finish_head` uses.
    //
    // THE PUBLISH IS `finish_head`, NOT A RE-SPELLING OF §05.6. This form has no lane split, so it owns each
    // row's whole `(m, l, acc)` and can write the final output itself instead of leaving it to
    // `attn_combine`; what it must not do is re-state normalise-then-round-then-gate beside the engine's
    // copy of that rule. `finish_head(a, head, qrow, l, acc, d)` computes `bf(fb(acc / l))`, writes `aout`,
    // and writes `gout = fb(av * bf(fb(sigmoid(g))))` from `a.qg` — so passing the already-divided value
    // with `l = 1.0f` reproduces it exactly: a divide by 1.0f is exact, so the rounding, the gate and the
    // BF16 widths are the shipped ones, in one spelling. What that costs relative to the rig's epilogue is
    // one expf and one divide per output dim (see the file header).
#pragma unroll
    for (int pass = 0; pass < (WIDE_MROWS / WIDE_TRG) * 2; ++pass) {
        const int rg = pass >> 1;              // which group of row-slots this pass stages
        const int half = pass & 1;             // which half of DH
        const int r0 = rg * WIDE_TRG;          // this pass's first row-slot within the wave
        __syncthreads();                       // the previous pass's reads are done (first pass: V is)
        if (rslot >= r0 && rslot < r0 + WIDE_TRG) {
            // A guarded store, never a divergent barrier: both barriers are outside the guard and every
            // lane of every wave takes both.
#pragma unroll
            for (int tau = half * (FATTN_VT16 / 2); tau < (half + 1) * (FATTN_VT16 / 2); ++tau) {
#pragma unroll
                for (int e = 0; e < FATTN_HELEMS; ++e) {
                    const int d = fattn_h_dim(lane, tau, e) - half * WIDE_TROW;   // always in [0, TROW)
                    s_out[wide_tr_addr(rslot - r0, d)] = fattn_h_get(out.t[tau][e]) / l;
                }
            }
        }
        __syncthreads();
        // Publish: 32 lanes x 16 floats = this pass's stage, exactly (asserted above), so coverage is a
        // fact and not a reading. Each lane's 16 floats are contiguous in LDS and contiguous in the output
        // row, so they go out as 16 `finish_head` calls on 16 consecutive dims of one (row, head).
        const int prow = r0 + (lane >> 3);              // the wave row-slot this lane publishes
        const int pblk = (lane & 7) * 16;               // its first dim within this half
        const int qrow_pub = warp_row0 + prow / GQA;
        if (qrow_pub < nrows) {           // a row-slot past the last row publishes NOTHING
            const int head_pub = head0 + prow % GQA;
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                ::finish_head(a, head_pub, qrow_pub, 1.0f,
                              s_out[wide_tr_addr(lane >> 3, pblk + u)],
                              half * WIDE_TROW + pblk + u);
            }
        }
    }
    __syncthreads();    // the next launch's stage write must not race the last read
}

// The staged plan only, and only where the launcher can honour it. An error, never a fallback: a wide form
// that silently did not run would price `attn_slim_wave` and report the result as A12's.
//   * `fused` — in the fused plan each core block appends its own lane, so a wide block's rows would read
//     lanes other rows have not appended yet. Same refusal, same reason, as `fattn_plan_ok`.
//   * `score_f32` — the score round here is `fattn_score_round`, the shipped BF16 form. There is no
//     F32-score spelling of this kernel to run, so asking for one is a refusal rather than a silent
//     rounding change.
//   * a 32-lane wave — the fragment maps, the butterflies and `__launch_bounds__` are all built on 32.
// What this list deliberately does NOT contain: `--attn-fattn 0`. The wide block is a spelling of the WMMA
// family and that pair is refused, but a `CoreArgs` has no field for the form the launcher chose, so the
// refusal lives where the choice is visible — at the CLI (keyed on the `_given` flag, the
// `--attn-fattn-qstage` precedent) and on `AttnLayerArgs` in `engine/kernels_attn.cpp`, which is the struct
// that carries it. A guard may only read the struct the launch uses; `CoreArgs` does not carry the choice,
// and inventing a field for it would put a second authority on who owns the wave's rows.
__device__ __host__ inline bool wide_plan_ok(const CoreArgs a, std::string* why) {
    if (a.fused != 0) {
        if (why) *why = "attn_wide_wave is a staged-plan kernel: in the fused plan each core block appends "
                        "its own lane, so the rows of one wide block would read lanes that other rows have "
                        "not appended yet.";
        return false;
    }
    if (a.score_f32 != 0) {
        if (why) *why = "attn_wide_wave rounds its scores through BF16 (fattn_score_round); there is no "
                        "F32-score spelling of this form to launch.";
        return false;
    }
    return true;
}

}  // namespace

// `__launch_bounds__(WIDE_THREADS, 1)`, and the min-blocks argument is spelled rather than left out for one
// reason: without it LLVM prices occupancy against the target's maximum block size (1024 threads), which
// budgets a thread 196,608/1024 = 192 VGPRs and spills everything above — the reading
// `engine/attn_fattn_kernels.h` records and `prototype/attn_wide/wide_ladder.sh` re-measured at 256 threads.
// With the block at 256 threads the arithmetic gives 196,608/256 = 768, so the per-thread architectural
// ceiling of 256 binds instead, and THAT headroom is what A12 spends. Spill is expected here, not a
// failure — which is exactly why `wide_echo` prints `scratch=` and `SPILL` instead of a green tick.
__global__ void __launch_bounds__(WIDE_THREADS, 1) attn_wide_wave(CoreArgs a) { attn_wide_body(a); }

}  // namespace engine::kernels
