/*
 * attn_wide.h — the A12 wide-shape attention block: 128 query rows, 8 waves, a 64-position K/V stage,
 * Q converted to F16 once per block and held in registers, F16 P·V accumulator.
 *
 * STANDALONE PROTOTYPE. Nothing here is reachable from `engine/`; A12's own sequencing
 * (`prototype/attn_percost/A12-WIDE-BET.md` §3) says no engine plumbing until the shape wins a paired device
 * window that only the main agent runs. This file graduates by being MOVED into `engine/`, at which point the
 * three duplicated laws below (`wide_c_row`/`wide_c_col`/`wide_h_dim`) collapse into the engine's single
 * spellings — they are restated here because a prototype that `#include`s an anonymous-namespace engine
 * header cannot be compiled alone, and that duplication is named rather than hidden.
 *
 * WHAT IS BEING PRICED, in the BET's words: a 128-query-row × 64-position-staged block clears 6.5 TFLOP/s at
 * 32,768 lanes against today's kernel-only 4.714. Per-round arithmetic (`A10-REWRITE-BRIEF.md` §3 item 1):
 * 128 rows × 64 positions × 512 MAC = 4,194,304 MAC against 64 positions × 2 tensors × 512 B = 65,536 B
 * demanded = 64 MAC per LSU byte, against llama's ≈31 and the shipping form's 9.6. The Q re-read — 898 GB/pass
 * today, 34.85 B per causal touch — is deleted, which is the whole traffic case.
 *
 * THE RELAXATIONS THIS SHAPE SPENDS, and where each is bounded (all three are A12-only; the shipping path keeps
 * all three protections — `A12-WIDE-BET.md` §1):
 *   - spill is a BUDGET, not a taboo: <= 1,024 B/thread of private scratch. The reference point is llama's own
 *     kernel at 900 B/thread (`A2-census.md` §1), re-read with the corrected instrument in
 *     `prototype/attn_percost/A12-WIDE-STEP12.md` §1, reproducible with `prototype/attn_wide/llama_vgpr_check.sh`.
 *   - 1 block/CU at 8 wavefronts is allowed (= llama's residency).
 *   - the P·V accumulator may be F16 — already built and inert as `--attn-fattn-pvf16` (`A10-PVF16.md`).
 *
 * THE REGISTER ARITHMETIC THAT MOTIVATES THE COMBINATION (per lane, one wave):
 *     Q resident, 16 row-slots × 256 dims as F16 operand tuples   16 tuples × 8 VGPR = 128
 *     P·V accumulator, F16, 8 tiles of 32 dims                      8 × 8 =  64
 *     one 64-position round's scores                                  4 C tiles × 8 =  32
 *     the four P operands                                             4 × 8 =  32
 *     running m, l, addressing, stage pointers                                    ≈  20
 *     against the per-thread ceiling                                              256
 * The F32 accumulator would be 128 there instead of 64 and the naive sum is ~308, which is why A10's R2 was
 * source-killed on the shipped atom (`A10-RF-LADDER.md` §6) and why the F16 accumulator is a PRECONDITION of
 * this shape rather than an optimisation. Whether the sum survives contact with LLVM is what
 * `prototype/attn_wide/wide_ladder.sh` measures; spill is expected, not a failure.
 *
 * WHAT IS NOT BEING PRICED HERE: the split/combine path (`part` slots + `attn_combine`), the §05.7 append, the
 * injection-aware KV-group rule, and the one-pass rows at or below MAXSC. The block writes its final O row
 * directly. Those are engine plumbing A12 explicitly does not have yet, and the mirror's numerics are judged on
 * the same narrowed op (`prototype/attn_wide/a12_wide_mirror.py`), so mirror and kernel agree about what the op
 * is — that is the thing §20.1's aliasing bug is about, stated at the boundary of the prototype rather than
 * inside it.
 *
 * THE FRAGMENT MAPS ARE NOT TRANSCRIBED, they are the engine's measured laws, restated:
 *   * C-tile row/column: lane t holds row t%16, column 2l + t/16 — measured by
 *     `prototype/attn_wmma/fattn_map_probe.cpp` and pinned by `check_fattn_map.py` (operand order: FIRST = K
 *     rows-are-KV-lanes, SECOND = Q rows-are-query-rows, so C is [query][kv], not its transpose).
 *   * F16 D-block map: element e of tile τ of lane t holds row t%16's output dim
 *     32τ + 16(e&1) + 2(e>>1) + t/16 — measured by `prototype/attn_percost/a10_pvf16_map_probe.cpp` and pinned
 *     by `check_a10_pvf16_map.py` in BOTH op_sel orders.
 * A relabelled map is a permuted row of the output, which is the class of defect that once came back as all-NaN
 * (§29 of `progress/2026-10-21/performance-status-2026-10-21.md`), so nothing below re-derives either law at a use site.
 *
 * WHY THE ROWS ARE 32 POSITIONS × 4 HEADS (llama's `R5` packing, `A10-REWRITE-BRIEF.md` §3): one staged K/V
 * tile then serves the four query heads of ONE KV group, and — the part that only shows up in the index
 * arithmetic — the block's query positions are a contiguous 32-wide run, so the round range is
 * [0, min(npos, pos0+32)) computed from `pos0` alone. No block-wide fold of per-warp spans is needed, which is
 * what the shipping form pays its `fold[]` buffer and two barriers for. The causal overhang this pays is bounded
 * by 32 positions per block (the shipping form's is bounded by 48), and per-row masking handles it.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>   // the WMMA operands are F16 whatever global is
#include <hip/hip_runtime.h>

// ---- the shape ----------------------------------------------------------------------------------------------
#define WIDE_DH 256              // head_dim (§00.2 / engine/state.h)
#define WIDE_NQ 8                // query heads
#define WIDE_NKV 2               // KV heads
#define WIDE_GQA (WIDE_NQ / WIDE_NKV)          // 4 query heads per KV group
#define WIDE_SCALE 0.0625f       // 1/sqrt(256), §00.2 — exact in binary, so staging can stay 2 bytes

#define WIDE_MROWS 16            // query row-slots per WAVE: the WMMA M dimension, not a tuning knob
#define WIDE_LCOLS 16            // KV positions per C tile: the WMMA N dimension, not a tuning knob
#ifndef WIDE_WARPS
#define WIDE_WARPS 8             // waves per block (A12's R1). Ladder axis.
#endif
#ifndef WIDE_STAGE
#define WIDE_STAGE 64            // KV positions per staged round (A12's R3). Ladder axis.
#endif
#ifndef WIDE_QREG
#define WIDE_QREG 1              // 1 = Q converted once and held in registers (A12's R2). Ladder axis.
#endif
#ifndef WIDE_PVF16
#define WIDE_PVF16 1             // 1 = F16 P·V accumulator (A12's precondition). Ladder axis.
#endif

#define WIDE_THREADS (WIDE_WARPS * 32)
#define WIDE_ROWS (WIDE_WARPS * WIDE_MROWS)                       // 128 query rows per block
#define WIDE_POS_PER_WARP (WIDE_MROWS / WIDE_GQA)                 // 4 query positions per wave
#define WIDE_POS_PER_BLOCK (WIDE_ROWS / WIDE_GQA)                 // 32 query positions per block
#define WIDE_LGROUPS (WIDE_STAGE / WIDE_LCOLS)                    // C tiles per round: 4
#define WIDE_KTILES (WIDE_DH / 16)                                // score roll trips: 16
#define WIDE_VTILES (WIDE_DH / 16)                                // F32 accumulator tiles: 16
#define WIDE_VT16 (WIDE_DH / 32)                                  // F16 accumulator tiles: 8
#define WIDE_HELEMS 16                                            // halves per lane in one F16 tile
#define WIDE_TRG 4                       // row-slots per epilogue transpose pass (see the LDS sum below)
#define WIDE_TROW (WIDE_DH / 2)          // floats per staged row per pass (half of DH)

// The four ladder axes must agree with each other, and each assert below is one of the four shapes the ladder
// compiles. `WIDE_MROWS % WIDE_GQA` is the GQA packing: a wave's 16 row-slots must be a whole number of query
// positions times the 4 heads of one KV group, or a wave would straddle two KV groups and share nothing.
static_assert(WIDE_MROWS == 16 && WIDE_LCOLS == 16,
              "the WMMA tile is 16x16x16 per wave: below 16 there is no A operand and no C tile");
static_assert(WIDE_MROWS % WIDE_GQA == 0, "a wave's row-slots must tile whole query positions of one KV group");
static_assert(WIDE_STAGE % WIDE_LCOLS == 0, "a stage must be a whole number of C tiles");
static_assert(WIDE_THREADS == WIDE_WARPS * 32, "the block extent is the wave count spelled in threads");

// ---- the register shapes ------------------------------------------------------------------------------------
using wide_A = short __attribute__((ext_vector_type(16)));   // one 16x16 F16 operand tuple   = 8 VGPRs
using wide_C = float __attribute__((ext_vector_type(8)));    // one 16x16 F32 C tile           = 8 VGPRs
using wide_H = short __attribute__((ext_vector_type(16)));   // one 16x32 F16 accumulator tile = 8 VGPRs
using wide_H2 = _Float16 __attribute__((ext_vector_type(16)));  // the same 32 bytes as numbers
static_assert(sizeof(wide_A) == 32 && sizeof(wide_C) == 32 && sizeof(wide_H) == 32,
              "an operand tuple, an F32 C tile and an F16 accumulator tile are 8 VGPRs each; the F16 tile "
              "covers 32 output dims because a half2 slot holds two dims, which is the accumulator halving");

struct WideO32 { wide_C t[WIDE_VTILES]; };   // 16 x 8 F32 = 128 VGPR/lane
struct WideO16 { wide_H t[WIDE_VT16]; };     // 8 x 8 F16  =  64 VGPR/lane

// ---- the arguments ------------------------------------------------------------------------------------------
// Q and O are laid out [query position][query head][dim], the same stride the engine's `fattn_q_row` uses
// (`ROW_H * row + head * DH`), so a later move into `engine/` is an index change and not a relayout.
struct WideArgs {
    const hip_bfloat16* q;   // [pos][head][dim]
    const hip_bfloat16* k;   // [kv head][cap][dim]
    const hip_bfloat16* v;   // [kv head][cap][dim]
    float* out;              // [pos][head][dim]
    int npos;                // query positions in this launch
    int cap;                 // lanes per KV head in k/v
};

// ---- the laws, restated once each ---------------------------------------------------------------------------
__device__ __forceinline__ int wide_c_row(int lane) { return lane & 15; }
__device__ __forceinline__ int wide_c_col(int lane, int l) { return 2 * l + (lane >> 4); }
__device__ __forceinline__ int wide_h_dim(int lane, int tau, int e) {
    return 32 * tau + 16 * (e & 1) + wide_c_col(lane, e >> 1);
}
__device__ __forceinline__ int wide_ab_row(int lane) { return lane & 15; }

// The two K/V layouts, one expression each (the shipping form's rule: one address expression per staged
// object, writer and reader alike — §20.1 of prototype/attn_decode/results.md). The V stage's row stride is
// WIDE_STAGE with NO padding: `A11-attribution.md` §5 measured that the unpadded stride makes every wmma
// operand read land in one bank group, and priced a pad at "a few-percent lever". Deliberately not taken here
// so the ladder prices the shape, not the pad.
__device__ __forceinline__ int wide_k_addr(int j, int d) { return j * WIDE_DH + d; }
__device__ __forceinline__ int wide_v_addr(int d, int j) { return d * WIDE_STAGE + j; }
// The epilogue stage: [row-slot within this pass][dim within this half of DH].
__device__ __forceinline__ int wide_tr_addr(int row_in_pass, int d) { return row_in_pass * WIDE_TROW + d; }

// The LDS the block declares, spelled as ONE buffer with TWO layouts, and asserted in bytes. This is the
// shipping form's arrangement (its `fattn_warp_stage_halves` note says why: `__shared__` arrays SUM, so a second
// array would move the block's declared LDS and with it the residency claim the number is about):
//   layout 1 — `s_kv`:  STAGE positions x DH halves, K row-major, then V transposed ALIASED over it, with a
//                        barrier between the users (the shipping kernel's three-reader buffer).
//   layout 2 — `s_out`: the F32 output transpose, WIDE_TRG row-slots x half of DH per pass, ONE SLICE PER WAVE
//                        (a shared slice is §20.1's aliasing bug: two waves writing one stage).
// The two layouts are separated by barriers: the round loop's last barrier is "V consumed", which is exactly the
// condition the epilogue needs before it may re-purpose the buffer, and every pass-to-pass barrier is taken by
// every wave.
constexpr int wide_lds_kv_halves() { return WIDE_STAGE * WIDE_DH; }
constexpr int wide_lds_out_halves() { return WIDE_WARPS * WIDE_TRG * WIDE_TROW * 2; }  // F32 = 2 halves
constexpr int wide_lds_halves() {
    return wide_lds_kv_halves() > wide_lds_out_halves() ? wide_lds_kv_halves() : wide_lds_out_halves();
}
static_assert(WIDE_MROWS % WIDE_TRG == 0, "the transpose row groups must tile a wave's row-slots");
static_assert(WIDE_TRG * WIDE_TROW == 32 * 16,
              "one pass stages WIDE_TRG rows x half of DH, and 32 lanes publish 16 floats each: 4 rows x 128 "
              "floats = 512 = 32 x 16, so coverage is a fact and not a reading (§18.1's all-NaN was a publish "
              "loop that covered 16 of every row's dims while satisfying every bound assert)");
static_assert(wide_lds_halves() * 2 <= 65536,
              "the architectural LDS per workgroup is 64 KiB (KFD `lds_size_in_kb 64`, read not quoted). At stage "
              "64 the aliased buffer is 32,768 B, which is also what makes 2 blocks/CU possible on LDS arithmetic. "
              "Above 49,152 B a launch would additionally need the dynamic-allocation opt-in, so stage 128 is a "
              "COMPILE-ONLY ladder shape and is never launched.");


__device__ __forceinline__ float wide_bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 wide_fb(float f) { return hip_bfloat16(f); }
__device__ __forceinline__ float wide_h_get(short h) { return __half2float(__builtin_bit_cast(__half, h)); }
// The score round, the shipped association: F32 dot -> BF16 -> *SCALE (exact) -> BF16 -> F32.
__device__ __forceinline__ float wide_score_round(float dot) {
    return wide_bf(wide_fb(wide_fb(dot) * WIDE_SCALE));
}

// One 32-byte operand tile out of the F16 stage: 16 contiguous halves = two 16-B LDS reads.
__device__ __forceinline__ wide_A wide_load_tile(const __half* p) {
    wide_A r;
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
// The same tile built from BF16 global: the conversion `__float2half_rn(float(bf16))` the staged form pays on
// its way in, paid at the use site instead. Bit-identical operands, different bytes-source.
__device__ __forceinline__ wide_A wide_load_q_tile(const hip_bfloat16* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
    const hip_bfloat16* s0 = reinterpret_cast<const hip_bfloat16*>(&lo);
    const hip_bfloat16* s1 = reinterpret_cast<const hip_bfloat16*>(&hi);
    wide_A r;
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = (short)__half_as_ushort(__float2half_rn(wide_bf(s0[e])));
#pragma unroll
    for (int e = 0; e < 8; ++e) r[8 + e] = (short)__half_as_ushort(__float2half_rn(wide_bf(s1[e])));
    return r;
}
// One staged KV row: 8 dims per lane, BF16 global -> F16 stage.
__device__ __forceinline__ void wide_stage_row(__half* dst, const hip_bfloat16* src, int lane) {
    const uint4 u = *reinterpret_cast<const uint4*>(src + lane * 8);
    const hip_bfloat16* s = reinterpret_cast<const hip_bfloat16*>(&u);
    union { __half h[8]; uint4 o; } o;
#pragma unroll
    for (int t = 0; t < 8; ++t) o.h[t] = __float2half_rn(wide_bf(s[t]));
    *reinterpret_cast<uint4*>(dst + lane * 8) = o.o;
}

// HIP takes the shuffle mask as 64-bit; an `0xffffffffu` mask is a compile error on this toolchain (the same
// trap `prototype/attn_decode/attn_decode_kernels.h:22` spells for the engine).
#define WIDE_FULL_MASK 0xffffffffffffffffull

// ---- the body -----------------------------------------------------------------------------------------------
// Every loop that touches a REGISTER array is `#pragma unroll`, without exception, and that is not style: a
// runtime-indexed array is an alloca, and an alloca inside the round loop is the spill this repo already met
// (`engine/attn_fattn_kernels.h`, the pointer-parameter story: 242 spilled VGPRs from two arrays and one
// `tmp[i]`). It bites harder here than in the shipping kernel because Q is an array of 16 tuples indexed by the
// roll's `t`: rolling that roll would put Q in scratch, which is the traffic this shape exists to delete.
__device__ __forceinline__ void attn_wide_body(const WideArgs a) {
    // ONE buffer, two layouts (see `wide_lds_halves` above). `s_kv` is the K/V stage; `s_out` is this wave's own
    // slice of the same bytes, reinterpreted as F32, which is legal only because the round loop's last barrier
    // ("V consumed") is exactly the condition the epilogue needs before re-purposing the stage.
    __shared__ __align__(16) __half s_mem[wide_lds_halves()];
    __half* const s_kv = s_mem;
    float* const s_out = reinterpret_cast<float*>(s_mem) + (threadIdx.x >> 5) * (WIDE_TRG * WIDE_TROW);

    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    const int nw = (int)(blockDim.x >> 5);
    const int kv = (int)blockIdx.x;                    // this block serves one KV group
    const int head0 = kv * WIDE_GQA;
    // Widest rows first, as the shipping form does it: the dispatch tail carries the long rows.
    const int pos0 = ((int)gridDim.z - 1 - (int)blockIdx.z) * WIDE_POS_PER_BLOCK;
    if (pos0 >= a.npos) return;                        // the grid is exact; stated, not assumed
    const int hi = a.npos < pos0 + WIDE_POS_PER_BLOCK ? a.npos : pos0 + WIDE_POS_PER_BLOCK;
    const int warp_pos0 = pos0 + w * WIDE_POS_PER_WARP;

    // This lane's row-slot: (lane & 15) of the wave's 16, which is WIDE_POS_PER_WARP query positions x the
    // 4 heads of the group. `row_ok` is the tail test; `pos_read` is the clamp that keeps a Q read inside the
    // buffer for a row-slot that does not exist, exactly as the shipping form's clamp does (not cosmetic: at
    // 128 rows per block a whole WAVE can be past `npos`, where `npos - 1` alone would index before the row).
    const int rslot = wide_c_row(lane);
    const int rpos = warp_pos0 + rslot / WIDE_GQA;
    const int rhead = head0 + rslot % WIDE_GQA;
    const bool row_ok = rpos < a.npos;
    const int pos_read = row_ok ? rpos : (hi - 1);
    const hip_bfloat16* qrow = a.q + ((size_t)pos_read * WIDE_NQ + rhead) * WIDE_DH;

#if WIDE_QREG
    // Q resident: 16 tuples x 8 VGPR = 128 registers per lane, converted once per LAUNCH. The two half-warps
    // load the same tuple (lane t and t^16 hold the same row-slot), so this reads 16 KiB per wave to place
    // 8 KiB of distinct Q — the same 2x duplication the shipping form's staged Q read has, and it is once per
    // launch here instead of once per round.
    wide_A qv[WIDE_KTILES];
#pragma unroll
    for (int t = 0; t < WIDE_KTILES; ++t) qv[t] = wide_load_q_tile(qrow + 16 * t);
#endif

    float m = -1e30f, l = 0.f;
#if WIDE_PVF16
    WideO16 out;
#pragma unroll
    for (int tau = 0; tau < WIDE_VT16; ++tau) out.t[tau] = wide_H{};
#else
    WideO32 out;
#pragma unroll
    for (int t = 0; t < WIDE_VTILES; ++t) out.t[t] = wide_C{};
#endif

    for (int base = 0; base < hi; base += WIDE_STAGE) {
        // ---- K into the stage: one row per wave per trip, all WIDE_WARPS waves covering the stage.
        for (int slot = w; slot < WIDE_STAGE; slot += nw) {
            int j = base + slot;
            if (j >= hi) j = hi - 1;                   // clamp; the cell is masked out below
            wide_stage_row(&s_kv[wide_k_addr(slot, 0)],
                           a.k + ((size_t)kv * a.cap + (size_t)j) * WIDE_DH, lane);
        }
        __syncthreads();                                                 // barrier 1: K published

        // ---- the score roll: FIRST = K rows (KV positions), SECOND = Q rows (query rows) -> C[q][kv].
        // WIDE_LGROUPS accumulators, one per 16-position slice of the stage, and ONE Q tuple per trip feeding
        // all four — that is the operand re-use the wide shape buys: 4 C tiles per Q fetch.
        wide_C acc[WIDE_LGROUPS];
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) acc[L] = wide_C{};
#pragma unroll
        for (int t = 0; t < WIDE_KTILES; ++t) {
#if WIDE_QREG
            const wide_A qt = qv[t];
#else
            const wide_A qt = wide_load_q_tile(qrow + 16 * t);
#endif
#pragma unroll
            for (int L = 0; L < WIDE_LGROUPS; ++L) {
                const wide_A kt = wide_load_tile(
                    &s_kv[wide_k_addr(WIDE_LCOLS * L + wide_ab_row(lane), 16 * t)]);
                acc[L] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kt, qt, acc[L]);
            }
        }

        // ---- mask + running max. The rounded score REPLACES acc in place: a second 32-float array would be
        // 32 more live registers per round for nothing, and the max is all that is read afterwards.
        // A cell is live iff this row-slot exists and its KV position is within this row's causal prefix.
        float rm = -1e30f;
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int j = base + WIDE_LCOLS * L + wide_c_col(lane, c);
                const bool live = row_ok && j <= rpos;
                acc[L][c] = live ? wide_score_round(acc[L][c]) : -1e30f;
                rm = fmaxf(rm, acc[L][c]);
            }
        }
        // A row's columns are split across the lane pair (t, t^16), so the max needs one butterfly. The
        // shipping form's tenth lane-movement op — half a rowsum scales a row by ~2x and cosine cannot see it.
        rm = fmaxf(rm, __shfl_xor_sync(WIDE_FULL_MASK, rm, 16, 32));
        const float mn = fmaxf(m, rm);
        const float rescale = expf(m - mn);

        // ---- exp + the C->B operand exchange, one pass, four P tuples (one per 16-position slice).
        wide_A p[WIDE_LGROUPS];
        float ls = 0.f;
        const bool hi_half = (lane & 16) != 0;
#pragma unroll
        for (int L = 0; L < WIDE_LGROUPS; ++L) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float e = expf(acc[L][c] - mn);
                ls += e;
                const float other = __shfl_xor_sync(WIDE_FULL_MASK, e, 16, 32);
                p[L][2 * c + 0] = (short)__half_as_ushort(__float2half_rn(hi_half ? other : e));
                p[L][2 * c + 1] = (short)__half_as_ushort(__float2half_rn(hi_half ? e : other));
            }
        }
        l = l * rescale + ls;
        m = mn;

        // ---- the accumulator rescale, on the accumulator's critical path, BEFORE the round's add.
        // With an F16 accumulator this is LOAD-BEARING, not tidiness: without it the first add is dropped
        // entirely by round 13 of the median head-row (`A10-PVF16.md` §4, chain section).
#if WIDE_PVF16
        {
            const _Float16 rh = (rescale);   // the same F32 factor, rounded to F16 once per round
#pragma unroll
            for (int tau = 0; tau < WIDE_VT16; ++tau) {
                wide_H2 v = __builtin_bit_cast(wide_H2, out.t[tau]);
#pragma unroll
                for (int e = 0; e < WIDE_HELEMS; ++e) v[e] = v[e] * rh;
                out.t[tau] = __builtin_bit_cast(wide_H, v);
            }
        }
#else
#pragma unroll
        for (int t = 0; t < WIDE_VTILES; ++t) {
#pragma unroll
            for (int c = 0; c < 8; ++c) out.t[t][c] *= rescale;
        }
#endif

        // ---- V into the stage, ALIASED over K, with a barrier on each side (every K reader passed barrier 1
        // and the roll above; every V reader gets barrier 3).
        __syncthreads();                                                 // barrier 2: K consumed
        for (int slot = w; slot < WIDE_STAGE; slot += nw) {
            int j = base + slot;
            if (j >= hi) j = hi - 1;
            const hip_bfloat16* vsrc = a.v + ((size_t)kv * a.cap + (size_t)j) * WIDE_DH;
            const uint4 u = *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            const hip_bfloat16* vv = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int t = 0; t < 8; ++t) s_kv[wide_v_addr(lane * 8 + t, slot)] = __float2half_rn(wide_bf(vv[t]));
        }
        __syncthreads();                                                 // barrier 3: V published

        // ---- P·V: FIRST = V staged transposed (rows are output dims), SECOND = P (rows are query rows).
        // 8 output tiles x 4 KV slices x 2 op_sel halves = 64 issues, the same count the shipping form pays for
        // 16 positions, and 8 independent accumulator chains. The F16 pair is llama's shape: two issues into
        // the SAME D block, two different A fragments (the low then the high 16 dims of the 32 the tile covers),
        // the same P (`A10-PVF16.md` §3, which read the pair out of the shipped listing rather than assume it).
#if WIDE_PVF16
#pragma unroll
        for (int tau = 0; tau < WIDE_VT16; ++tau) {
#pragma unroll
            for (int L = 0; L < WIDE_LGROUPS; ++L) {
                const wide_A vlo = wide_load_tile(
                    &s_kv[wide_v_addr(32 * tau + wide_ab_row(lane), WIDE_LCOLS * L)]);
                const wide_A vhi = wide_load_tile(
                    &s_kv[wide_v_addr(32 * tau + 16 + wide_ab_row(lane), WIDE_LCOLS * L)]);
                out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vlo, p[L], out.t[tau], 0);
                out.t[tau] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(vhi, p[L], out.t[tau], 1);
            }
        }
#else
#pragma unroll
        for (int t = 0; t < WIDE_VTILES; ++t) {
#pragma unroll
            for (int L = 0; L < WIDE_LGROUPS; ++L) {
                const wide_A vt = wide_load_tile(
                    &s_kv[wide_v_addr(16 * t + wide_ab_row(lane), WIDE_LCOLS * L)]);
                out.t[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vt, p[L], out.t[t]);
            }
        }
#endif
        __syncthreads();                                                 // barrier 4: V consumed
    }

    // ---- the deferred rowsum combine (the shipping form's tenth lane-movement op): legal because `rescale`
    // is pair-uniform — the max already took its butterfly — so both halves differ by the same product.
    l += __shfl_xor_sync(WIDE_FULL_MASK, l, 16, 32);

    // ---- the epilogue: the accumulator's columns are parity-split across the lane pair and each lane holds
    // ONE row-slot, while `out` wants that row's whole DH contiguously. So stage through LDS, WIDE_TRG
    // row-slots x half of DH at a time, and divide by `l` on the way out (the mirror divides the same way, in
    // the same order: widen the F16 accumulator to F32 first, then divide).
    // No reciprocal shortcut: the divide is `acc / l` in F32, in that order, because that is the order the CPU
    // mirror models. Widening first and then dividing is also what the shipping form's `finish_head` does.
#pragma unroll
    for (int pass = 0; pass < (WIDE_MROWS / WIDE_TRG) * 2; ++pass) {
        const int rg = pass >> 1;              // which group of row-slots this pass stages
        const int half = pass & 1;             // which half of DH
        const int r0 = rg * WIDE_TRG;          // this pass's first row-slot within the wave
        __syncthreads();                       // the previous pass's reads are done
        if (rslot >= r0 && rslot < r0 + WIDE_TRG) {
            // A guarded store, never a divergent barrier: both barriers are outside the guard and every lane in
            // the block takes both.
#if WIDE_PVF16
#pragma unroll
            for (int tau = half * (WIDE_VT16 / 2); tau < (half + 1) * (WIDE_VT16 / 2); ++tau) {
#pragma unroll
                for (int e = 0; e < WIDE_HELEMS; ++e) {
                    const int d = wide_h_dim(lane, tau, e) - half * WIDE_TROW;   // always in [0, WIDE_TROW)
                    s_out[wide_tr_addr(rslot - r0, d)] =
                        wide_h_get(out.t[tau][e]) / l;
                }
            }
#else
#pragma unroll
            for (int t = half * (WIDE_VTILES / 2); t < (half + 1) * (WIDE_VTILES / 2); ++t) {
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const int d = 16 * t + wide_c_col(lane, c) - half * WIDE_TROW;
                    s_out[wide_tr_addr(rslot - r0, d)] = out.t[t][c] / l;
                }
            }
#endif
        }
        __syncthreads();
        // Publish: 32 lanes x 16 floats = this pass's stage, exactly (asserted above). Each lane's 16 floats
        // are contiguous in LDS and contiguous in `out`, so the global write is one 64-B store per lane.
        const int prow = r0 + (lane >> 3);              // the wave row-slot this lane publishes
        const int pblk = (lane & 7) * 16;               // its first dim within this half
        const int qpos = warp_pos0 + prow / WIDE_GQA;
        if (qpos < a.npos) {
            float* dst = a.out + ((size_t)qpos * WIDE_NQ + head0 + prow % WIDE_GQA) * WIDE_DH
                       + half * WIDE_TROW + pblk;
            const float* src = &s_out[wide_tr_addr(lane >> 3, pblk)];
#pragma unroll
            for (int u = 0; u < 16; ++u) dst[u] = src[u];
        }
    }
    __syncthreads();    // the next launch's stage write must not race the last read
}

// `__launch_bounds__(WIDE_THREADS, WIDE_MINB)`, with the min-blocks argument exposed as a ladder axis because it
// is the open question `A10-REWRITE-BRIEF.md` §3 left behind: "whether clang honours a min-blocks argument on
// gfx1151 at all is precisely what the step-2 compile ladder exists to answer", asked of llama's
// `__launch_bounds__(256, 2)`. The shipping kernel's own note (`engine/attn_fattn_kernels.h`, above its
// `__global__`s) records what happens with NO launch bounds at all: LLVM prices occupancy against the target's
// 1024-thread maximum block and budgets a thread 196,608/1024 = 192 VGPRs, then spills everything above. At
// 256 threads and min-blocks 1 the same arithmetic gives 196,608/256 = 768, i.e. the per-thread architectural
// ceiling of 256 binds instead — which is the register headroom A12 spends, and it arrives from the block size
// alone. Row L of `wide_ladder.sh` sets min-blocks 4, where the arithmetic gives 196,608/1024 = 192 and the two
// readings make different predictions; that is what makes it a measurement rather than an inference.
#ifndef WIDE_MINB
#define WIDE_MINB 1                  // ladder axis: the `__launch_bounds__` min-blocks argument
#endif
__global__ void __launch_bounds__(WIDE_THREADS, WIDE_MINB) attn_wide_wave(WideArgs a) { attn_wide_body(a); }
