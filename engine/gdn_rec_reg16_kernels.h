/*
 * gdn_rec_reg16_kernels.h — bet G4: §04.4's looped recurrence spelled with LLAMA'S THREAD MAP, under OUR
 * recurrence. 128 blocks x 128 threads, 8 lanes per state column, and **16 state elements per thread
 * held in registers for the whole span** — read once at span entry, written once at span exit, and
 * NOTHING moves state between them.
 *
 * WHY THIS MAP AND NOT ANOTHER. `prototype/gdn_walk/G3-llama-kernel.md` read the kernel §38 of
 * `bench/performance-status.md` had only ever inferred from a profiler
 * (`engines/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`, `gdn.cu:203-204`), and found that the one
 * axis no earlier bet had touched is state elements per thread: llama runs 128 blocks x 128 threads =
 * 512 wavefronts per launch = 6.40 wf/SIMD against our 0.80, at 16 elements per thread, and its shard
 * fits in registers (`vgpr 78, spill 0, LDS 0`) so it issues ZERO state memory ops per token. Our
 * shipped walk holds 128 elements per thread and re-reads plus re-writes the whole 1 MiB state from
 * global EVERY token. Instruction count is closed as the explanation — they issue 1.28x MORE slots per
 * token and win by 5.76x — and G2 is now the second sign: it took wf/SIMD the other way (0.8 ->
 * 0.20-0.40) and lost 122 % (`prototype/gdn_walk/BET.md` §11/§12). The map is forced by arithmetic,
 * not taste:
 *
 *     wavefronts x 32 x elements_per_thread == NH * D * D == 262,144
 *
 * so raising wavefronts MEANS lowering elements, and 16 is what fits in a register file that also has
 * to hold this token's k and q slices. The assertion below is that identity; `kGdnR16Elems` is the
 * compile-time constant the whole map hangs off, and every other dimension is derived from it.
 *
 * THE SHAPE, field by field, against `gdn.cu:203-204`:
 *
 *     grid   dim3(NH, D/16)     = (16, 8)      llama: (H, n_seqs, S_v/(4*4)) = (16, 1, 8)
 *     block  dim3(32, 4)        = 128 threads  llama: (warp_size, num_warps) = (32, 4)
 *     waves  16 * 8 * 4         = 512         llama: 512 (G3 §1, all three axes)
 *     columns per block          16           llama: cols_per_warp(4) * num_warps(4) = 16
 *     lanes per column            8           llama: warp_size/cols_per_warp = 8
 *     elements per thread        16           llama: S_v/lanes_per_col = 16
 *
 * `lane` is `threadIdx.x` and `warp` is `threadIdx.y`, exactly as llama indexes them; a column group is
 * the 8 contiguous lanes `lane & 7` and `lane >> 3` picks which of a warp's four columns a lane owns.
 *
 * ONE DIFFERENCE FROM LLAMA, deliberate and forced: OUR state layout stays `[i][j]` (row i, column j,
 * `engine/state.h:107`, `STATEFP`, the decode handoff all read it that way) where llama's is `[col][i]`.
 * So a thread's 16 elements are 16 CONSECUTIVE `i` values of one column — stride `D`*4 B = 512 B, not
 * llama's stride-32 B within a contiguous row — and there is no vector load to be had. That is the
 * price of not transposing, and it is paid ONCE PER SPAN, not per token: 2 MiB read + 2 MiB written per
 * launch per layer against the ~3 MiB/token/layer the shipped walk demands at the LSU (§6 constraint 4
 * of `G3-llama-kernel.md`, which sizes it at ~16 KB/token/layer where we pay ~3 MiB/token/layer).
 * Transposing the state is a later option, not this bet, and not touching the layout is what keeps every
 * consumer working.
 *
 * ONE ADDRESS EXPRESSION, because §20.1 of `prototype/attn_decode/results.md` is a measured failure and
 * not a style note: two self-consistent layouts joined by a branch passed every per-form check while half
 * of every streaming row silently came out as another group's answer. `state_col` below is the ONLY
 * place a state address is formed; the span-entry load, the span-exit store and the static asserts all
 * go through it, and the per-token update never touches an address at all because the shard is in
 * registers. A second spelling of the column base is what would make this kernel two layouts.
 *
 * NORMS: 8-LANE BUTTERFLIES, ZERO STAGING. The 8 lanes of a column group hold the whole 128-element k
 * row between them (each holds its own 16 consecutive elements), so `sum k^2` is a 3-level xor butterfly
 * over thread-local slices: no block reduction, no `__syncthreads()`, no normed copies, no LDS at all.
 * That is G3 §6 constraint 3, and it is the failure G1 measured — hoisting the norms into a pre-pass
 * cost +32 KB/token/layer of staged rows and gave most of its -5.9 % row win back in `gdn_gated_norm`
 * and `glue` (`BET.md` §8). This pays 0 new bytes: the k and q rows are the same rows the shipped walk
 * already reads, and each lane reads the 16 it owns. The cost is the ~40 slots/token/wavefront G3
 * prices, and the fallback it names (stage the two row SUMS per head per token, 128 B/token/layer) was
 * not needed.
 *
 * NUMERICS, PRE-DECLARED (G3 §6 constraint 5, `BET.md` §10 constraint 2). Two sums re-associate, and
 * only these two:
 *   1. the ROW NORMS. The shipped `sk`/`sq` come from `block_sum_pair_f32` at `blockDim.x` = 128 — a
 *      5-level per-warp fold, an LDS hop to 4 partials, a second 5-level fold. Here they are a
 *      16-element thread-local partial folded in 3 xor levels across 8 lanes. Different tree, different
 *      F32 sum. (The per-element products keep the shipped shape: `kn = k_i * rsqrtf(sk + 1e-6f)`,
 *      `qn = (q_i * rsqrtf(sq + 1e-6f)) * Q_SCALE`, association written out because AGENTS.md's `q̃` row
 *      is exactly the case where "equivalent" algebra was a different number on device.)
 *   2. the STATE DOTS. `r_j` and `o_j` are one serial 128-term chain per column in the shipped walk; here
 *      each thread folds its own 16 and the 8 lanes of the column butterfly-together. `o` keeps the
 *      shipped two-walk form (`acc += q̂_i * S_new[i][j]`, i ascending), so unlike G2 there is no closed-form
 *      rewrite of the output and no `sum k_i q_i` scalar in play.
 * So `o` moves and `STATEFP` moves. The band is the EXISTING §10.1 aggregates — cos >= 0.9995,
 * meanrel <= 5 % — judged through `engine/gate.py` on the pinned fixtures, with the shipped side as the
 * comparator for "did the move stay re-association scale". No new band, no widening, no `--ack-spill`.
 * `prototype/gdn_walk/g4_mirror.py` is the CPU size-of-the-move and `gate_g4.py` the device witness.
 *
 * CORRECT AT ANY SPAN, including one token and a ragged tail, because the state hand-off is the kernel's
 * entry and exit rather than a property of the span: load the shard, run `tokens` steps, store the
 * shard. `tokens` is uniform across the grid, so every lane runs the same trip count and no lane can
 * store a partially-updated shard. (`tokens < 2` is still refused by the launcher — not because the
 * kernel would be wrong at 1, it is not, but because a one-row body is `resolve_gdn_loop`'s to route and
 * a decode bit must not reach a kernel the decode pins were not pinned through.)
 *
 * NO INTERFERENCE. A block owns 16 columns of one head's state and no other block touches them, so there
 * is no cross-block ordering to enforce: the span-entry load happens after every write of the PREVIOUS
 * span by the same block (same stream, same kernel, ordered), and within the span only this block's
 * threads ever read or write their own 16 columns. That is the same argument `gdn_rec_wf_kernels.h` and
 * `gdn_rec_hoist_kernels.h` rely on, and it is why zero barriers here is a claim rather than a race.
 *
 * THE REGISTER BUDGET IS THE WHOLE BET, so it is measured, not hoped: `vgpr_spill_count > 0` means the
 * shard lives in scratch and this is route A's failure with extra steps. The peak live payload is 3 x 16
 * floats (the state shard, this token's k slice and its q slice) which is why the k and q slices are
 * NORMALIZED IN PLACE below — `kh`/`qh` hold raw k/q until the norm lands and become k̂/q̂ in the same 16
 * registers, rather than living next to four more arrays. Read the note block before believing any of
 * this: `prototype/attn_wmma/fattn_residency_read.py` on a `-O3 gfx1151` listing of this TU, and
 * `prototype/gdn_walk/G4-result.md` §1 holds the reading this file was written against.
 *
 * WHAT THIS FILE IS NOT. It is not the default: `--gdn-geom` stays `block` and this is a knob first, like
 * G1's `--gdn-norm` and G2's `wf4`/`wf2` before it (the D32 -> D50 -> D54 order — a knob ships as a knob
 * and the paired in-situ A/B on the `gdn_rec` row plus `wall_ms` moves the default, per
 * `prototype/gdn_walk/BET.md` §12: gdn_rec <= 170 live / > 225 dead, wall_ms <= 540 live / > 607 dead,
 * `wall_ms` the ship gate). It reaches the `--gdn-loop` span branch only, and `decode_run` refuses the
 * pairs where it would be inert (`--gdn-loop 1`, `--gdn-scan collapse`, `--gdn-norm hoist`).
 */
#pragma once

#include "prototype/gdn_decode/gdn_decode_kernels.h"   // D, NH, CH, Q_SCALE

// The row strides are the identities `rec_loop_kernel` and `rec_wf_kernel` each restate, restated a
// fourth time here because this kernel indexes them itself: `CH` on the conv's [n][6144] tile and
// `NH*D` on the [n][2048] output. A stride that only looked equal is a silent re-map of every row
// after 0, which is a numerics change with no value to catch it.
static_assert(CH == 3 * NH * D, "rec_reg16's q|k|v row stride is CH; must be 3*NH*D = kQkvRows");
static_assert(NH * D == 2048, "rec_reg16's o row stride is NH*D; must be the engine's kValueDim");

// ---- the map, as compile-time constants -----------------------------------------------------------
//
// Every dimension is DERIVED from `kGdnR16Elems` — the elements per thread is the free parameter the map
// is named after, and the grid falls out of it. The launcher in `engine/kernels_gdn.cpp` launches with
// THESE SAME constants, in the same TU as this kernel, because the AGENTS.md 2026-10-19 row's whole
// lesson is that a geometry fact living in two headers gets one half committed and the tip does not
// build.
constexpr int kGdnR16Elems = 16;            // state elements per thread — the bet's name
constexpr int kGdnR16BlockX = 32;           // lanes in a wavefront (this part's warp size)
constexpr int kGdnR16BlockY = 4;            // wavefronts per block
constexpr int kGdnR16LanesPerCol = D / kGdnR16Elems;   // 8: the group that covers one column
constexpr int kGdnR16ColsPerWarp = kGdnR16BlockX / kGdnR16LanesPerCol;   // 4 columns per wavefront
constexpr int kGdnR16ColsPerBlock = kGdnR16ColsPerWarp * kGdnR16BlockY;
constexpr int kGdnR16GridX = NH;                       // one head per blockIdx.x
constexpr int kGdnR16GridY = D / kGdnR16ColsPerBlock;  // the column blocks that tile a head
constexpr int kGdnR16Waves = kGdnR16GridX * kGdnR16GridY * kGdnR16BlockY;

// The map is FORCED, and the forcing is an assertion rather than a paragraph.
static_assert(kGdnR16Elems == 16, "16 elements per thread is what llama's note block fits; a different"
                                  " value is a different bet and needs its own note-block read");
static_assert(kGdnR16Elems * kGdnR16LanesPerCol == D,
              "the 8 lanes of a column group must cover the column between them, which is what makes"
              " the row norm a 3-level butterfly over thread-local slices and buys the zero staging");
static_assert(kGdnR16LanesPerCol == 8, "llama's map has 8 lanes per state column (`gdn.cu:43`);"
                                       " lanes_per_col = warp_size / cols_per_warp = 32/4");
static_assert(kGdnR16Waves * kGdnR16BlockX * kGdnR16Elems == NH * D * D,
              "THE map identity: wavefronts x 32 x elements_per_thread must tile the whole 262,144-"
              "element state exactly once. This is the equation G3 §6 constraint 1 says to assert; a"
              " launch that covers the state twice or not at all is a compile error, not a bad run.");
static_assert(kGdnR16GridX == 16 && kGdnR16GridY == 8,
              "grid dim3(NH, D/16) = (16, 8) = 128 blocks, `gdn.cu:203`");
static_assert(kGdnR16BlockX == 32 && kGdnR16BlockY == 4,
              "block dim3(32, 4) = 128 threads = 4 wavefronts, `gdn.cu:204`");
static_assert(kGdnR16Waves == 512, "512 wavefronts per launch is the number §43 corrects §38 to;"
                                   " 512 = 6.40 wf/SIMD over KFD's 80 SIMDs, 8x the shipped 0.80");

// `v += __shfl_xor_sync(FULL, v, o)` for o = 4, 2, 1: every lane of an 8-lane column group finishes
// holding the group's whole sum, which is what a broadcast reduction needs and what the shipped
// `block_sum_pair_f32` buys with an LDS hop and three barriers instead. The offsets are < 8 and the
// groups are 8-contiguous and 8-aligned, so an xor butterfly cannot cross a column group — that is the
// whole correctness argument for using the plain un-widthed form, and it is why the same helper serves
// both norms and both dots. Order 4,2,1 is llama's `warp_reduce_sum<8>` (`common.cuh:468-476`, the loop
// runs `offset = width/2` down to 1), kept so the two engines' association is at least the same tree at
// the same width. The 64-bit mask is a HIP requirement on this toolchain, not style (AGENTS.md's HIP row).
__device__ __forceinline__ float gdn_bfly8_sum_f32(float v) {
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 4);
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 2);
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 1);
    return v;
}

// One GDN layer, ONE launch, `tokens` token-steps — the same operand list and the same buffers
// `rec_loop_kernel` is handed, so a diff of the two forms is a claim about the thread map and the
// arithmetic and never about the access pattern. `qtile`/`ktile`/`vtile` are the three 2048-wide slices
// of one [n][6144] tile, `gtile`/`betatile` are [n][16], `otile` is [n][2048], `S` is the layer's
// [16][128][128] block in [i][j] order.
__global__ void rec_reg16_kernel(const float* __restrict__ qtile, const float* __restrict__ ktile,
                                 const float* __restrict__ vtile, const float* __restrict__ gtile,
                                 const float* __restrict__ betatile, float* __restrict__ S,
                                 float* __restrict__ otile, int nheads, int tokens) {
    constexpr int E = kGdnR16Elems;
    const int h = blockIdx.x;                                   // the head, `gdn.cu:203` grid.x
    const int lane = threadIdx.x;                               // 0..31 within the wavefront
    const int warp = threadIdx.y;                               // 0..3 within the block
    // `col_in_warp` and `lane_in_col` are llama's two lane-derived indices, spelled the same way
    // (`gdn.cu:47-49`). `lane_in_col` picks WHICH 16 consecutive rows of the column this lane owns, and
    // `col_in_warp` picks which of the warp's four columns the lane belongs to — the inverse of
    // `gdn.cu:48-49`'s pairing, because our state is [i][j] where theirs is [col][i].
    const int col_in_warp = lane >> 3;                          // 0..3
    const int lane_in_col = lane & 7;                           // 0..7
    const int j = blockIdx.y * kGdnR16ColsPerBlock + warp * kGdnR16ColsPerWarp + col_in_warp;
    const int i0 = lane_in_col * E;                             // this lane's first state row

    // ONE address expression. `state_col[r]` IS the state element S[h][i0 + r][j] — no second spelling
    // of it anywhere in this kernel, per §20.1 (see the header). `i0 * D + j` is the only place the
    // [i][j] stride is written down, and the span-entry load and the span-exit store both index `E`
    // elements of this pointer at stride `D`, so a wrong row/column pair moves the load and the store
    // together rather than desynchronising them.
    float* __restrict__ state_col =
        S + (size_t)h * D * D + (size_t)i0 * D + j;
    float shard[E];
#pragma unroll                          // NOT a hint here: a rolled index into shard[] is a
    for (int r = 0; r < E; ++r) {       // local-memory array, i.e. exactly the spill this bet dies on
        shard[r] = state_col[(size_t)r * D];
    }

    for (int t = 0; t < tokens; ++t) {
        // This token's rows, declared HERE with the `t` advance in the declaration. G1 shipped a
        // strength-reduction bug past every CPU check by hoisting these (`gate_g2.py`'s header names the
        // fingerprint it moved), so the shape is load-bearing, not tidiness.
        const float* __restrict__ q = qtile + (size_t)t * CH + (size_t)h * D;
        const float* __restrict__ k = ktile + (size_t)t * CH + (size_t)h * D;
        const float* __restrict__ v = vtile + (size_t)t * CH + (size_t)h * D;
        const float* __restrict__ g = gtile + (size_t)t * NH;
        const float* __restrict__ beta = betatile + (size_t)t * NH;
        float* __restrict__ o = otile + (size_t)t * (NH * D);

        // ---- this lane's 16 elements of the k and q rows, and the two row sums of squares.
        // `kh`/`qh` hold RAW k/q until the norms land and k̂/q̂ afterwards, in the same registers, so the
        // peak live payload is 3 arrays (shard + kh + qh = 48 floats) instead of 5. That is llama's
        // payload exactly — G3's census counts 48 of their 78 VGPRs on `s_shard`/`k_reg`/`q_reg` — and
        // the in-place normalisation is what keeps a spilled shard from being inevitable.
        float kh[E], qh[E];
        float sk = 0.f, sq = 0.f;
#pragma unroll
        for (int r = 0; r < E; ++r) {
            kh[r] = k[i0 + r];
            qh[r] = q[i0 + r];
            // Accumulated in the same loop that loads, in the same ascending `r` order, so the partial
            // is the SAME sequence of adds as a separate pass would form — fewer instructions, identical
            // bits. Nothing may be hoisted out of here: the norms belong to token `t`.
            sk += kh[r] * kh[r];
            sq += qh[r] * qh[r];
        }
        // The whole 128-element row lives in these 8 lanes, so a 3-level butterfly is the entire
        // reduction: no block reduction, no barrier, no staged norm (`BET.md` §10 constraint 1's
        // "zero bytes", which G1 paid +32 KB/token/layer for).
        sk = gdn_bfly8_sum_f32(sk);
        sq = gdn_bfly8_sum_f32(sq);
        const float rs_k = rsqrtf(sk + 1e-6f);
        const float rs_q = rsqrtf(sq + 1e-6f);
#pragma unroll
        for (int r = 0; r < E; ++r) {
            // The shipped per-element products, association written out (`q̃` row, AGENTS.md).
            kh[r] = kh[r] * rs_k;                       // k̂_i
            qh[r] = (qh[r] * rs_q) * Q_SCALE;           // q̂_i
        }

        const float gam = expf(g[h]), bet = beta[h];

        // ---- walk A: fold this lane's 16, then the column's 8. `shard` is read, never addressed.
        float r = 0.f;
#pragma unroll
        for (int i = 0; i < E; ++i) {
            r += shard[i] * kh[i];
        }
        r = gdn_bfly8_sum_f32(r);
        // `dv` in the shipped shape: r_j first, then β·(v − γ·r), same order of the two subtrahends.
        const float dv = bet * (v[j] - gam * r);

        // ---- walk B: the update and the output dot, in the shipped two-walk form, over the shard in
        // registers. `o` keeps `acc += q̂_i · S_new[i][j]` with i ascending, so the ONLY thing that moves
        // in the output is the association of the fold (16 partials + an 8-lane butterfly), not the
        // algebra — which is what makes this form's numerics smaller than G2's.
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < E; ++i) {
            float s0 = gam * shard[i];
            s0 += kh[i] * dv;
            shard[i] = s0;
            acc += qh[i] * s0;
        }
        acc = gdn_bfly8_sum_f32(acc);
        if (lane_in_col == 0) {
            o[h * D + j] = acc;      // one store per column, like `gdn.cu:168-170`
        }
    }

    // ---- span exit: the shard goes back once, through the same expression the entry read through.
#pragma unroll
    for (int r = 0; r < E; ++r) {
        state_col[(size_t)r * D] = shard[r];
    }
}
