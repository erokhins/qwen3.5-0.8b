/*
 * attn_wide_llama_kernels.h — the A13 llama-ported wide block, behind `--attn-fattn-wide 2`, the shipping default.
 *
 * WHAT THIS FILE IS. `engine/attn_wide_kernels.h`'s block with four structural features of llama.cpp's
 * `flash_attn_ext_f16` ported onto it — `prototype/attn_wide/LLAMA-PORT-PLAN.md` items 1, 2, 4 and 5. It is
 * a NEW form behind a NEW knob value, not an edit of the A12 form: nothing here changes what knob 0 or knob 1
 * does, and both kernels are in the binary whatever the flag says, so a paired A/B is one build rather than
 * two windows (AGENTS.md, "a ruler re-taken in a different window is not an A/B").
 *
 * THE CLAIM THIS FORM IS BUILT ON, and it is a strong one: **every ported item here is value-preserving, so
 * knob 2 should produce the same numbers as knob 1, element for element.** That is what makes it cheap to
 * adjudicate — a byte-compare against knob 1 rather than a new band — and each item earns it:
 *   * item 1, the swizzle, permutes bytes INSIDE a stage row. Writer and reader use one expression
 *     (`wll_k_addr`/`wll_v_addr`), so a tile lands exactly where the reader looks — the §20.1 law
 *     (`prototype/attn_decode/results.md`), applied to a permutation rather than a layout.
 *   * item 2, the mid-round prefetch, moves WHEN a load is issued, not WHAT it loads. The next round's K tile
 *     is the same bytes of the same cache, read in the same order, committed before the same barrier.
 *   * item 4, the bounds specialization, deletes a predicate that provably never fires in a full round: a
 *     wide block spans `WIDE_POS_PER_BLOCK` = 32 query positions and `lanes_of` advances by exactly 1 per
 *     row, so a block's causal spread is 31 lanes, LESS than one `WIDE_STAGE` = 64 round — every round but
 *     the last one or two is live for every row-slot of the block. That is a stronger version of llama's
 *     single specialized `last_iter` (`fattn-mma-f16.cuh:1350-1388`, `k_VKQ_sup` clamping at `:1362`).
 *   * item 5, folding `SCALE` into Q at staging, is exact: `SCALE = 0.0625f = 2^-4`
 *     (`prototype/attn_decode/attn_decode_kernels.h:34`); scaling by a power of two commutes with rounding
 *     and with the F32 WMMA sum, so `f16(bf(q)·2⁻⁴) = f16(bf(q))·2⁻⁴`, every product and the accumulated sum
 *     are exactly scaled, and `bf(fb(dot·2⁻⁴)) = bf(fb(dot))·2⁻⁴` — which is the value `fattn_score_round`
 *     computes today, so the score, the max, the `exp` argument and the rowsum come out identical. THE ONE
 *     HOLE: a Q element whose `bf(q)` is under ~2^-10 falls into the F16 SUBNORMAL range once scaled, where
 *     an F16 mantissa has fewer than bf16's 7 bits left. That loses bits in terms below ~1e-6 of the Q scale
 *     and is the only route by which this form's bits can differ from knob 1's.
 *
 * WHAT THIS FORM DELIBERATELY DOES NOT DO, and why:
 *   * NO PADDED STAGE STRIDE. A11 §5 (`prototype/attn_percost/A11-attribution.md`) measured the aliasing that
 *     a 512 B and a 128 B row stride cause and bounded the fix at a few percent of the window; llama's own
 *     RDNA3 answer is a pad, not a swizzle (`ggml_cuda_fattn_mma_get_swizzled` is CUDA-only, so
 *     `stride_tile_K = nbatch_K2 + 4`, and `mma.cuh:792` says `// Equivalent to padding with 16 bytes`). The
 *     swizzle here is a within-row permutation, so declared LDS, the residency claim and `wide_echo`'s
 *     `lds=` field are all untouched; what item 1 buys in SLOTS is the template stride and the hoisted bases.
 *   * NO DOUBLE-BUFFERED K IN LDS. The plan's item 2 wanted the next round's K staged beside the current
 *     round's V, which needs 64 KiB against the 32 KiB this shape declares; the register form is the one that
 *     keeps the residency claim, and it costs 32 VGPR/lane of in-flight tile on a kernel already reserving
 *     256 and spilling 163 (`prototype/gdn_walk/g3_isa_bill.py`'s metadata line). Spill is expected here and
 *     `wide_echo` prints `scratch=`/`SPILL`, so an increase is seen and not inferred.
 *   * NO MASK TILE. llama adds an LDS `half` mask tile to the score (`fattn-mma-f16.cuh:818-823`, the RDNA3
 *     branch) because it carries a real padding mask; our only mask is the causal bound `j < lanes_of(...)`,
 *     one compare per cell off a per-lane scalar. Re-spelling it as an added sentinel costs
 *     `cmp + cndmask + fadd` where the select costs `cmp + cndmask`, so the tail body keeps the select.
 *   * THE EPILOGUE IS CARRIED, NOT SHARED. The transpose-and-`finish_head` block below is the same text as
 *     `attn_wide_kernels.h`'s tail, duplicated on purpose: factoring it would move the machine code of a form
 *     that is mid-admission (`prototype/attn_wide/A12-ADMISSION.md`), and §05.6's contract lives in
 *     `finish_head`, which both forms call. Whoever changes one must change the other until A12 settles.
 *
 * UNCHANGED FROM KNOB 1 because the task named it: the F16 accumulator and its per-round rescale
 * (`attn_wide_kernels.h`'s risk note 2 — an F16 accumulator drops an add once the addend falls below half a
 * ULP, and a 64-position round makes the fold MORE load-bearing, not less); the deferred rowsum and its one
 * `__shfl_xor(16)` combine; and every fragment map, which still comes from `engine/attn_fattn_kernels.h` and
 * is never re-derived here.
 *
 * INCLUDED AFTER `engine/attn_wide_kernels.h`, which it includes itself so the order cannot be got wrong, and
 * from the one TU that compiles both (`engine/kernels_attn.cpp`).
 */
#pragma once

#include "engine/attn_wide_kernels.h"

namespace engine::kernels {
namespace {

// ---- item 1: template strides + XOR swizzle, one address expression per staged object -------------------
// A staged TILE is 16 halves = 32 B, and that is not a taste: it is the granularity `fattn_load_tile` reads
// at (two 16-B LDS reads, one WMMA operand tuple) and the granularity the K writer fills at (one `uint4` per
// lane). Permuting WHOLE tiles inside a row is invisible to both; permuting finer is not — an XOR at 16-B
// granularity swaps the two 8-half halves of a tuple by a row-dependent amount and re-orders the operand's
// k dimension, which is a permuted dot.
constexpr int WLL_TILE = 16;
// Row strides as compile-time constants (K's row is one staged KV position, V's row one output dim). Both
// values are UNCHANGED from knob 1: the swizzle is a permutation, not a pad.
constexpr int WLL_K_STRIDE = DH;
constexpr int WLL_V_STRIDE = WIDE_STAGE;
// llama's `swizzle<>` (`mma.cuh:786-801`) rotates by `i % 8` 16-B units when the row stride is a whole
// multiple of 128 B, which is our case; at 32-B granularity that is `tile ^ (row & 7)`. V has only 4 tiles
// per row, so its mask is 3. Any wider and the XOR could leave the row.
constexpr int WLL_K_TILES = WLL_K_STRIDE / WLL_TILE;
constexpr int WLL_V_TILES = WLL_V_STRIDE / WLL_TILE;
constexpr int WLL_K_XOR = WLL_K_TILES >= 8 ? 7 : WLL_K_TILES - 1;
constexpr int WLL_V_XOR = WLL_V_TILES >= 8 ? 7 : WLL_V_TILES - 1;

template <int STRIDE, int XOR>
__device__ __forceinline__ int wll_swa(int row, int half_in_row) {
    return row * STRIDE + ((((half_in_row >> 4) ^ (row & XOR)) << 4) + (half_in_row & 15));
}
// K: row = staged KV position, column = head dim. THE one spelling of this layout, writer and reader.
__device__ __forceinline__ int wll_k_addr(int kv, int d) { return wll_swa<WLL_K_STRIDE, WLL_K_XOR>(kv, d); }
// V: row = output dim, column = staged KV position. Same law, different stride and mask.
__device__ __forceinline__ int wll_v_addr(int d, int kv) { return wll_swa<WLL_V_STRIDE, WLL_V_XOR>(d, kv); }

static_assert(WLL_K_STRIDE % WLL_TILE == 0 && WLL_V_STRIDE % WLL_TILE == 0,
              "a stage row must be a whole number of swizzle tiles, or the XOR would cross a row");
static_assert((WLL_K_TILES & (WLL_K_TILES - 1)) == 0 && (WLL_V_TILES & (WLL_V_TILES - 1)) == 0,
              "tiles per row must be a power of two: the swizzle is an XOR, and an XOR with a mask only "
              "permutes inside a power-of-two span");
static_assert((WLL_K_XOR & (WLL_K_XOR + 1)) == 0 && (WLL_V_XOR & (WLL_V_XOR + 1)) == 0 &&
              WLL_K_XOR <= WLL_K_TILES - 1 && WLL_V_XOR <= WLL_V_TILES - 1,
              "each XOR mask must be of the form 2^n-1 and fit inside its row's tile count, so no swizzled "
              "address can leave its own row");
static_assert(WIDE_STAGE * WLL_K_STRIDE == wide_lds_kv_halves() &&
              DH * WLL_V_STRIDE == wide_lds_kv_halves(),
              "both staged regions keep knob 1's byte sum and both still fill the whole stage, because K and V "
              "alias one buffer whose size IS the residency claim (`wide_lds_halves`, `wide_echo`'s lds= field) "
              "— the swizzle permutes inside it and pads nothing");
static_assert(WIDE_STAGE % WIDE_WARPS == 0,
              "the K loader gives every wave the same number of staged positions, in both directions");

// ---- item 5: the softmax scale, folded into Q once at staging -------------------------------------------
// `fattn_load_q_tile`'s two `uint4` reads and its `__float2half_rn(fattn_bf(x))` conversion, with the scale
// inside the conversion. A second spelling on purpose: the factor is 2^-4 so the VALUES are the shared
// loader's scaled by a power of two (exact), and editing that loader would have moved the shipping form.
// The score round below must therefore NOT apply `SCALE` again.
__device__ __forceinline__ fattn_ARegs wll_load_q_tile(const hip_bfloat16* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
    const hip_bfloat16* s0 = reinterpret_cast<const hip_bfloat16*>(&lo);
    const hip_bfloat16* s1 = reinterpret_cast<const hip_bfloat16*>(&hi);
    fattn_ARegs r;
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = (short)__half_as_ushort(__float2half_rn(fattn_bf(s0[e]) * SCALE));
#pragma unroll
    for (int e = 0; e < 8; ++e) r[8 + e] = (short)__half_as_ushort(__float2half_rn(fattn_bf(s1[e]) * SCALE));
    return r;
}
// The scale-free sibling of `fattn_score_round`: the factor is already in Q, so the dot's F32 result needs
// only the shipped BF16 round, in the shipped association.
__device__ __forceinline__ float wll_score_round(float dot) { return fattn_bf(fattn_fb(dot)); }

// ---- item 2: the next round's K tile, held in registers while the matrix work runs -----------------------
// cp.async's AMD equivalent. To be exact about the source: llama's own multi-stage pipeline is CUDA-only
// (`CP_ASYNC_AVAILABLE` requires `!GGML_USE_HIP`, `common.cuh:302`, and `ggml_cuda_fattn_mma_get_nstages`
// returns 0 without it), so on this device their K and V are staged synchronously inside the round just as
// knob 1 does. What is ported is the STRUCTURE of `fattn-mma-f16.cuh:980-996` — issue the next round's load
// after the current score tile is consumed and before P*V runs — which is what the brief pre-authorised.
// One wave stages WIDE_STAGE/WIDE_WARPS positions, one row per lane, exactly as knob 1 does; what moves is
// that the global load and the LDS store are separated by a barrier and a matrix roll instead of sitting
// back-to-back in front of one.
constexpr int WLL_KTRIP = WIDE_STAGE / WIDE_WARPS;

__device__ __forceinline__ void wll_load_k(uint4 (&buf)[WLL_KTRIP], const CoreArgs& a, int kv_read,
                                          int base, int hi, int w, int lane) {
#pragma unroll
    for (int s = 0; s < WLL_KTRIP; ++s) {
        // The clamp is what lets the LAST round prefetch a tile that does not exist: a position past the
        // block's bound re-reads the last position that does, and the cell it stages is masked by the tail
        // body. `base`/`hi` are block-uniform, so no wave diverges on it.
        int j = base + w + s * WIDE_WARPS;
        if (j >= hi) j = hi - 1;
        buf[s] = *reinterpret_cast<const uint4*>(
            a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH + lane * 8);
    }
}

__device__ __forceinline__ void wll_store_k(const uint4 (&buf)[WLL_KTRIP], __half* s_kv, int w, int lane) {
#pragma unroll
    for (int s = 0; s < WLL_KTRIP; ++s) {
        const int slot = w + s * WIDE_WARPS;
        const hip_bfloat16* src = reinterpret_cast<const hip_bfloat16*>(&buf[s]);
        union { __half h[8]; uint4 o; } o;
#pragma unroll
        for (int t = 0; t < 8; ++t) o.h[t] = __float2half_rn(fattn_bf(src[t]));
        // One `uint4` per lane, 16 B, at a swizzled tile boundary — `wll_k_addr` keeps 8-half alignment
        // because the XOR moves whole 16-half tiles, so the wide store stays a `ds_store_b128`.
        *reinterpret_cast<uint4*>(&s_kv[wll_k_addr(slot, lane * 8)]) = o.o;
    }
}

// Q in registers, as a STRUCT, because an array parameter is what put `pexp[8]` and `out[16]` in scratch and
// cost 242 spilled VGPRs the first time (`engine/attn_fattn_kernels.h`). Every index below is a compile-time
// constant, which is the property that keeps it in registers.
struct WllQ { fattn_ARegs t[FATTN_KTILES]; };

// ---- THE ROUND, one spelling, instantiated twice ---------------------------------------------------------
// `kMasked` is the item-4 split: `false` for every round the whole block can see, `true` for the one or two
// that cross a row's causal bound. Two instantiations of one inlined body is what llama does — their loop
// calls `flash_attn_ext_f16_iter` with `last_iter=false` for every full tile and `true` for the tail
// (`fattn-mma-f16.cuh:1350-1388`) — and it is why the full body carries no bounds test at all.
// `next` is runtime and block-uniform: the last round of the wave prefetches nothing.
template <bool kMasked>
__device__ __forceinline__ void wll_round(const CoreArgs& a, __half* s_kv, const WllQ& q, FattnO16& out,
                                         float& m, float& l, int base, int hi, int kv_read, int rlanes,
                                         bool row_ok, int w, int lane, bool next) {
    if constexpr (!kMasked) { (void)rlanes; (void)row_ok; }   // only the tail body reads the per-row bound

    __syncthreads();                                          // barrier A: this round's K is published

    // The two hoisted bases. `fattn_ab_row(lane)` is this lane's row inside a 16-row tile, and every tile row
    // this lane ever reads is `ab + 16L`, a multiple of 8 above it — so `row & XOR` is `ab & XOR` for the
    // whole roll and the swizzle collapses to ONE per-lane value hoisted out of both rolls. That is the
    // whole of item 1's ISA case: every tile address below is `base + const`.
    const int kb = fattn_ab_row(lane) * WLL_K_STRIDE;
    const int kx = (fattn_ab_row(lane) & WLL_K_XOR) << 4;

    // ---- the score roll: FIRST operand = K rows (KV positions), SECOND = Q rows, so C is [query][kv].
    // Operand order, initialisation and the four-C-tiles-per-Q-fetch re-use are knob 1's, and for knob 1's
    // reasons (the measured map, and the uninitialised-first-tile NaN of §29/D64).
    fattn_CRegs acc[WIDE_LGROUPS];
#pragma unroll
    for (int L = 0; L < WIDE_LGROUPS; ++L) acc[L] = fattn_CRegs{};
#pragma unroll
    for (int t = 0; t < FATTN_KTILES; ++t) {
        const fattn_ARegs qt = q.t[t];
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) {
            const fattn_ARegs kt = fattn_load_tile(
                &s_kv[kb + FATTN_MLANES * L * WLL_K_STRIDE + ((16 * t) ^ kx)]);
            acc[L] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kt, qt, acc[L]);
        }
    }

    // ---- THE PREFETCH, issued here for the same reason llama issues it here (`:980-996`): the score tile
    // has been consumed, P*V has not run yet, so the global latency of round n+1 hides behind the round's own
    // matrix work instead of sitting in front of a barrier. 32 VGPR/lane stay live across everything below.
    uint4 kbuf[WLL_KTRIP];
    if (next) wll_load_k(kbuf, a, kv_read, base + WIDE_STAGE, hi, w, lane);

    // ---- running max. In the FULL body there is nothing to mask, and that is a proof rather than a hope:
    // every row-slot of the block has a causal bound of at least `lo_bound`, and a full round is one with
    // `base + WIDE_STAGE <= lo_bound`, so all WIDE_STAGE positions are live for every row-slot that exists.
    // A row-slot past `nrows` is no longer forced to -1e30f here; its accumulator holds finite garbage from
    // the clamped Q row, it is confined to that lane's registers (the two butterflies pair (t, t^16), which
    // is ONE row-slot), and the publish guard below is what keeps it out of memory.
    float rm = -1e30f;
#pragma unroll
    for (int L = 0; L < WIDE_LGROUPS; ++L) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if constexpr (kMasked) {
                const int j = base + FATTN_MLANES * L + fattn_c_col(lane, c);
                const bool live = row_ok && j < rlanes;
                acc[L][c] = live ? wll_score_round(acc[L][c]) : -1e30f;
            } else {
                acc[L][c] = wll_score_round(acc[L][c]);
            }
            rm = fmaxf(rm, acc[L][c]);
        }
    }
    // A row's columns are split across the lane pair (t, t^16), so the max needs the one butterfly. Knob 1's
    // tenth lane-movement op, kept verbatim: half a rowsum scales a row and cosine cannot see it.
    rm = fmaxf(rm, __shfl_xor_sync(FULL_MASK, rm, 16, warpSize));
    const float mn = fmaxf(m, rm);
    const float rescale = expf(m - mn);

    // ---- exp + the C->B operand exchange, ONE pass, four P tuples. The fusion and the un-normalised F16
    // weight are knob 1's (`attn_wide_kernels.h` risk note 1: this form rounds the UN-normalised `exp` to F16
    // and divides at the end, which is why it has no bit-identity claim against the one-pass route).
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

    // ---- THE PER-ROUND RESCALE, on the accumulator's critical path, BEFORE the round's add. Untouched from
    // knob 1, and it stays load-bearing: a 64-position round carries four times the addends of a 16-wide one
    // into a single F16 accumulator step (`prototype/attn_wide/a12_wide_mirror.py` §4). LLVM already pairs
    // these into `v_pk_mul_f16` — 64 of them per round, the ideal count — so plan item 3 needed no port.
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

    // ---- V into the stage, ALIASED over the K this round has finished reading. Barrier on each side, for
    // knob 1's reason (eight warps on one stage make the missing barrier a whole-wave NaN, §20.1).
    __syncthreads();                                          // barrier B: K consumed
    for (int slot = w; slot < WIDE_STAGE; slot += WIDE_WARPS) {
        int j = base + slot;
        if (j >= hi) j = hi - 1;
        const hip_bfloat16* vsrc = a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
        const uint4 u = *reinterpret_cast<const uint4*>(vsrc + lane * 8);
        const hip_bfloat16* vv = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
        for (int t = 0; t < 8; ++t) s_kv[wll_v_addr(lane * 8 + t, slot)] = __float2half_rn(fattn_bf(vv[t]));
    }
    __syncthreads();                                          // barrier C: V published

    // ---- P*V: FIRST operand = V staged transposed, SECOND = P, two `v_wmma_f16` issues into one D block
    // (`A10-PVF16.md` §3, read out of the shipped listing). Same hoisted-base trick: the tile rows this lane
    // reads are `ab + 32tau + 16h`, all a multiple of 4 above `ab`, so `row & WLL_V_XOR` is `ab & WLL_V_XOR`.
    const int vb = fattn_ab_row(lane) * WLL_V_STRIDE;
    const int vx = (fattn_ab_row(lane) & WLL_V_XOR) << 4;
#ifdef QWEN_WIDE_DEBUG
    // The hoisted bases are the same law as `wll_k_addr`/`wll_v_addr` written a second time — written that way
    // so a per-lane term is computed once per roll instead of 96 times, and written once in `wll_swa` so the
    // layout has a single statement of record. That is exactly the pair that must not diverge (§20.1, and the
    // `FATTN_ROWS_PER_BLOCK` two-header trap), so the debug build compares the two spellings at EVERY address
    // the roll uses. Compile it with `-DQWEN_WIDE_DEBUG`; the release build carries none of this.
#pragma unroll
    for (int L = 0; L < WIDE_LGROUPS; ++L) {
#pragma unroll
        for (int t = 0; t < FATTN_KTILES; ++t) {
            const int ho = kb + FATTN_MLANES * L * WLL_K_STRIDE + ((16 * t) ^ kx);
            const int law = wll_k_addr(FATTN_MLANES * L + fattn_ab_row(lane), 16 * t);
            if (ho != law) printf("WLL_K_ADDR ho=%d law=%d L=%d t=%d lane=%d\n", ho, law, L, t, lane);
        }
#pragma unroll
        for (int tau = 0; tau < FATTN_VT16; ++tau) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int ho = vb + (32 * tau + 16 * hh) * WLL_V_STRIDE + ((16 * L) ^ vx);
                const int law = wll_v_addr(32 * tau + 16 * hh + fattn_ab_row(lane), 16 * L);
                if (ho != law) printf("WLL_V_ADDR ho=%d law=%d tau=%d h=%d L=%d lane=%d\n",
                                      ho, law, tau, hh, L, lane);
            }
        }
    }
#endif
#pragma unroll
    for (int tau = 0; tau < FATTN_VT16; ++tau) {
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) {
            const fattn_ARegs vlo = fattn_load_tile(
                &s_kv[vb + (32 * tau) * WLL_V_STRIDE + ((16 * L) ^ vx)]);
            const fattn_ARegs vhi = fattn_load_tile(
                &s_kv[vb + (32 * tau + 16) * WLL_V_STRIDE + ((16 * L) ^ vx)]);
            out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vlo, p[L], out.t[tau], 0);
            out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vhi, p[L], out.t[tau], 1);
        }
    }
    __syncthreads();                                          // barrier D: V consumed

    // ... and only now may the prefetched tile land: it goes into the SAME region the V that every wave has
    // just finished reading, so it waits for barrier D like knob 1's next-round store does. The barrier that
    // publishes it is barrier A of the round that reads it.
    if (next) wll_store_k(kbuf, s_kv, w, lane);
}

// ---- the body --------------------------------------------------------------------------------------------
// The block geometry, the row rule, the causal bound and the epilogue contract are knob 1's, unchanged, and
// they are read through the same engine functions rather than paraphrased. What is new is the split of the
// round loop into a predicate-free part and a masked tail, and the pipelined K staging.
__device__ __forceinline__ void attn_wide_llama_body(const CoreArgs a) {
    // ONE buffer, TWO layouts across barriers, sized by knob 1's `wide_lds_halves()` — so the swizzle changes
    // no byte sum and `wide_echo`'s lds= field still reads 32,768.
    __shared__ __align__(16) __half s_mem[wide_lds_halves()];
    __half* const s_kv = s_mem;
    // This wave's OWN slice of the same bytes as F32 (a shared output slice is §20.1's aliasing bug), legal
    // only because barrier D of the last round is exactly the condition the epilogue needs before re-purpose.
    float* const s_out = reinterpret_cast<float*>(s_mem) + (threadIdx.x >> 5) * (WIDE_TRG * WIDE_TROW);

    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    const int kv_read = tile_kv_of((int)blockIdx.x, a.neg);      // knob 1's group rule, not a paraphrase
    const int head0 = (int)blockIdx.x * GQA;
    const int rt = (int)gridDim.z - 1 - (int)blockIdx.z;         // widest rows first, as both wave forms do
    const int row0 = rt * WIDE_POS_PER_BLOCK;
    const int nrows = nrows_of(a);
    if (row0 >= nrows) return;                                   // the grid is exact; stated, not assumed
    const int rowhi = nrows < row0 + WIDE_POS_PER_BLOCK ? nrows : row0 + WIDE_POS_PER_BLOCK;
    const int warp_row0 = row0 + w * WIDE_POS_PER_WARP;

    // The round range, BLOCK-UNIFORM by construction (`hi`), plus the bound that makes item 4's split legal:
    // `lanes_of(a, qrow) = lvalid + qrow` advances by exactly 1 per query row, so the block's widest row is
    // `hi` and its narrowest `lo_bound`, and 32 positions of spread is under one 64-position round.
    const int hi = lanes_of(a, rowhi - 1);
    if (hi <= 0) return;         // block-uniform, and before any barrier, so no wave is left holding one
    const int lo_bound = lanes_of(a, row0);
    const int nfull = lo_bound / WIDE_STAGE;   // the rounds with `base + WIDE_STAGE <= lo_bound`, exactly

    const int rslot = fattn_c_row(lane);
    const int rrow = warp_row0 + rslot / GQA;
    const int rhead = head0 + rslot % GQA;
    const bool row_ok = rrow < nrows;
    // The clamp that keeps a Q read inside the buffer for a row-slot that does not exist. It stays, and it is
    // not cosmetic: a whole WAVE can be past `nrows` at 128 row-slots per block, and the garbage a full round
    // now computes for such a slot has to be FINITE garbage, which a clamped row read is.
    const int qrow_read = row_ok ? rrow : (rowhi - 1);
    const hip_bfloat16* const qrow = fattn_q_row(a, qrow_read, rhead);
    const int rlanes = row_ok ? lanes_of(a, rrow) : 0;          // read by the tail body alone

    // Q resident: FATTN_KTILES tuples x 8 VGPR = 128 registers per lane, converted ONCE PER LAUNCH, and with
    // the softmax scale folded in (item 5). Same traffic argument as knob 1: the shipping form re-reads Q at
    // 34.85 B per causal touch, 898 GB per pass at 32,768 lanes.
    WllQ q;
#pragma unroll
    for (int t = 0; t < FATTN_KTILES; ++t) q.t[t] = wll_load_q_tile(qrow + 16 * t);

    float m = -1e30f, l = 0.f;
    FattnO16 out;
#pragma unroll
    for (int tau = 0; tau < FATTN_VT16; ++tau) out.t[tau] = fattn_HRegs{};

    // THE PIPELINE'S FIRST TILE: round 0 has no earlier round to prefetch it, so it is staged here and
    // published by barrier A of the first round. From there on every round's K arrives from the round before.
    {
        uint4 kbuf[WLL_KTRIP];
        wll_load_k(kbuf, a, kv_read, 0, hi, w, lane);
        wll_store_k(kbuf, s_kv, w, lane);
    }

    // ---- the steady state: every round this block can see in full, and there is no bounds test inside it.
    for (int base = 0; base < nfull * WIDE_STAGE; base += WIDE_STAGE) {
        wll_round<false>(a, s_kv, q, out, m, l, base, hi, kv_read, rlanes, row_ok, w, lane,
                         /*next=*/base + WIDE_STAGE < hi);
    }
    // ---- THE TAIL, at most two rounds by the arithmetic above (`hi - lo_bound <= WIDE_POS_PER_BLOCK - 1 =
    // 31 < WIDE_STAGE`), and the only place the per-row causal bound is evaluated. It runs zero times when the
    // block's bound lands exactly on a round boundary.
    for (int base = nfull * WIDE_STAGE; base < hi; base += WIDE_STAGE) {
        wll_round<true>(a, s_kv, q, out, m, l, base, hi, kv_read, rlanes, row_ok, w, lane,
                        /*next=*/base + WIDE_STAGE < hi);
    }

    // THE DEFERRED ROWSUM COMBINE — knob 1's tenth lane-movement op, before the epilogue so no lane is ever
    // shuffled against a partner that has already exited.
    l += __shfl_xor_sync(FULL_MASK, l, 16, warpSize);

    // ---- the epilogue. CARRIED FROM `engine/attn_wide_kernels.h`, the same text on purpose (the file header
    // says why): stage WIDE_TRG row-slots x half of DH at a time through this wave's own slice, widen the F16
    // accumulator to F32 and divide FIRST, then hand the value to `finish_head` with `l = 1.0f` so §05.6's
    // rounding, gate and BF16 widths are the shipped ones in one spelling rather than in a second one.
#pragma unroll
    for (int pass = 0; pass < (WIDE_MROWS / WIDE_TRG) * 2; ++pass) {
        const int rg = pass >> 1;              // which group of row-slots this pass stages
        const int half = pass & 1;             // which half of DH
        const int r0 = rg * WIDE_TRG;          // this pass's first row-slot within the wave
        __syncthreads();                       // the previous pass's reads are done (first pass: V is)
        if (rslot >= r0 && rslot < r0 + WIDE_TRG) {
            // A guarded store, never a divergent barrier: both barriers are outside the guard and every lane
            // of every wave takes both.
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
        // Publish: 32 lanes x 16 floats = this pass's stage exactly (knob 1's coverage assert on
        // `wide_tr_addr` is what makes that a fact rather than a reading).
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

// The staged-plan and F32-score refusals are RE-USED from `wide_plan_ok` rather than re-spelled: both are
// about the `CoreArgs` the launch dereferences (in the fused plan a wide block would read lanes other rows
// have not appended yet; there is no F32-score spelling of this family — `wll_score_round` rounds through
// BF16 as `fattn_score_round` does), and neither is a property of the form. A second copy of two refusals
// would be a second authority over the same two questions.

}  // namespace

// The same `__launch_bounds__(WIDE_THREADS, 1)` argument knob 1 carries: without the min-blocks argument LLVM
// budgets a thread 196,608/1024 VGPRs and spills everything above; with it the per-thread architectural
// ceiling of 256 binds, and THAT headroom is what the resident Q and the in-flight K tile spend.
__global__ void __launch_bounds__(WIDE_THREADS, 1) attn_wide_llama_wave(CoreArgs a) {
    attn_wide_llama_body(a);
}

}  // namespace engine::kernels
