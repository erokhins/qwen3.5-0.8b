#pragma once
// One launch function per graduated kernel, and nothing else.
//
// The kernels themselves live in `prototype/<op>/<op>_kernels.h`, moved verbatim out of the
// binaries each op's own gate signed off against. This header exists so `decode_step` can call
// them without seeing a single macro or device helper from those files: each launcher is its own
// translation unit (`kernels_*.cpp`), because the ops each spell their own `bf`/`fb`/`rnd` and
// define colliding macros (`D`, `H`, `HID`, `EPS`, `SCALE`, `FULL_MASK`) — one TU per family
// keeps that harmless, and merging them is a compile error rather than a cleanup.
//
// Every launcher takes raw device pointers, launches with the configuration its op's gate used,
// and returns. Nothing here syncs: a decode step is submitted as a stream of launches and the
// host waits once, at the logits (§07.1 / W5's "no host round-trip per op").
//
// The `neg` parameter each op carries for its own negative tests is always the clean value here.
// The engine is not a place to inject bugs; the injections stay in the op's gate.

#include <cstddef>
#include <cstdint>
#include <string>

namespace engine::kernels {

// ---- §03.1 + §03.2: residual add fused with the zero-centred RMSNorm (prototype/glue).
// `delta == nullptr` is the norm-only form (§03.2's first call of the stack, and the final norm);
// `s_out == nullptr` skips writing the new residual. 256 threads is the width the op's gate ran at
// (H/THREADS = 4 elements/thread, an 8-partial warp tree). The block width changes the F32 summation
// order of the mean of squares, and the gate has since been re-run at 1024 with an identical legal
// error profile — so 256 is this launcher's measured preference, not a numerical requirement
// (kernels_glue.cpp cites both measurements; engine/results.md E1 is closed).
void glue_add_norm(const uint16_t* s, const uint16_t* delta, const uint16_t* w,
                   uint16_t* s_out, uint16_t* n_out);

// The same op over a [rows, H] tile (P4, §08.2): one block per row, so a tile in one launch is
// obliged to produce the bits of `rows` launches of one row — `prototype/glue/run.py --batched p64`
// requires that on device (adds bit-exact vs the pins, norms inside the floor_probe band, and 612
// 352 elements compared against one launch per row with 0 differing). `rows == 1` is the launch
// above. `n_out` may not be null; `delta == nullptr` is the norm-only form and `s_out == nullptr`
// skips writing the residual, both exactly as in the single-row case.
void glue_add_norm_rows(const uint16_t* s, const uint16_t* delta, const uint16_t* w,
                        uint16_t* s_out, uint16_t* n_out, int rows);

// ---- route (c): the wave matmul route (hipBLAS for the ten fat shapes at wave width).
//
// WHAT THIS IS. `bench/prefill-route-spike-2026-10-13.md` §6 item 1 funds hipBLAS for ten shapes
// (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`,
// `up_proj`, `down_proj` — 150 of a wave's 186 matmul calls) and §"Wave-path attribution" of
// `bench/performance-status.md` says why: matmul is 74.4 % of a 1024-token prefill and the hand
// GEMMs run at ~0.85 TFLOP/s against hipBLAS's ~33 at the same shape. The call form is the probe's:
// `OP_T` on the weight as the shard stores it (no transpose copy), bf16 in, f32 accumulate, **bf16
// `C`** — 16-bit `C` because every shipped pin rounds a projection's product to a 16-bit width, and
// an f32 `C` would move that rounding AND double the activation traffic (§4.3, §3.3).
//
// THE SEAM IS THE ROW COUNT, and it is the same seam the MLP walk and the attention plan already run
// on: a body of ONE row keeps the hand GEMV (decode is bandwidth-bound and the one-row `lm_head` sits
// at 102 % of the stream roofline — there is nothing for a library to win, and W4 measured the
// small-N library call at 12–22× the dispatch floor), and a body of two or more rows may take the
// library. So decode's launch sequence — every bit of it — is untouched by this route by construction.
// `in_proj_a`/`in_proj_b` (N=16, 0.065 B/flop, below the ridge at every width) stay in the hand
// kernel at any width, which is why their call sites call `proj_gemv` and not `proj_gemm`.
//
// `kAuto` does not exist here on purpose: there is no third thing for a row count to pick.
enum class WaveGemm {
    // The shipped default since D-11 admitted it: hipBLAS for a wave body, hand GEMV for a one-row
    // body. Admitted on the audit's terms, and the guarantee is one the reader has to read as weaker
    // than the hand route's: on this route the wave-boundary check is INSIDE A BAND, not
    // byte-identical — see `bench/route-c-audit-2026-10-14.md` and `engine/results.md`, route (c).
    kLibrary,
    // Still first-class, and deleted by nothing: with this set every body takes the hand kernel
    // whatever its width, which is how the old path stays measurable in the same window (§7 W9 is an
    // A/B), how "decode untouched" is checkable end to end, and how a pre-route binary's bits can
    // still be reproduced. `engine/gate.py` judges it by the pre-route bands: its cascade floor is
    // armed for `library` and for nothing else.
    kHand,
};

// "library" / "hand". Throws on anything else: this flag selects which side of an A/B a run is.
WaveGemm parse_wave_gemm(const std::string& name);
const char* wave_gemm_name(WaveGemm route);

// hipBLAS's own version as "major.minor.patch", or "unknown" if the query fails. D-11 makes this
// provenance: the library's solution per (m,n,k) is the numerics, and a bump can change it without
// touching a line of this repository. Queried from the loaded library, not a build-time macro.
std::string gemm_route_version();

// THE ROW-COUNT RULE, in one place, next to `resolve_mlp_gate_map`'s: a one-row body is always the
// hand GEMV — that is what makes "decode byte-identical" a property of the rule rather than of a
// call site remembering it — and any wider body takes what the run asked for.
WaveGemm resolve_wave_gemm(WaveGemm want, int rows);

// One handle and one workspace for the process, created here and pinned: nothing in this route
// allocates per call. Returns the line `decode_run` prints when the flag named the route (one handle,
// one pinned workspace, default stream, nothing per call — said out loud because it is the constraint
// a reviewer cannot check from the call sites). Idempotent; throws if the handle or the workspace
// cannot be created, which is better than a first matmul failing 400 launches into a prefill.
std::string gemm_route_init();
// Tears down what `gemm_route_init` created. Safe to call when it never ran.
void gemm_route_shutdown();

// The matmul itself: `out[m,n] = x[m,k] · weight[n,k]ᵀ`, bf16 operands, f32 accumulate, bf16 `out`.
// `weight` is the shard's [n,k] row-major blob used as stored; `x` is [m,k] row-major and `out` is
// [m,n] row-major, both at their natural leading dimension (`k` and `n`), which is what every
// row-family tile in `DecodeScratch` already is. NOT counted in `engine/launch_stats.h`: this is the
// bare call, and the family counters belong to the routed entry points below so that one projection
// site counts as one site whichever GEMM ran.
void gemm_bf16(const void* weight, const void* x, void* out, int m, int n, int k);

// ---- §04.1 / §05.1: the projection GEMV family (prototype/proj_gemv).
// `in` must be 1024 or 2048 — the only two widths the checkpoint has for these nine shapes, and
// the only two the kernel is instantiated for. `x` is F32: the launcher instantiates PACK_X=false,
// the gated form and the measured-faster one (kernels_proj.cpp cites both numbers; engine/results.md
// E2 closed the A/B in a clean window). Hand it a BF16 buffer and it will read BF16 memory
// as floats, so decode_step must cast where its producer left BF16.
//
// `tokens` is P5's token-batched form (§06.2's "one GEMM per projection"): `x` is then a
// [tokens, in] tile and `out` a [tokens, rows] tile, and each weight row is read once per token tile
// instead of once per token. `tokens == 1` issues the SAME kernel at a tile width of 1, i.e. exactly
// the launch the decode ledger was measured on — one arithmetic, two tile widths, which is why this
// family has no second kernel the way the MLP pair does. The two widths are bit-identical per output
// element: `prototype/proj_gemv/run.py --tokens p64` requires it (138 400 elements per fixture,
// 0 differing) and `gate.py` section (h) requires that a wave is the side that batched.
void proj_gemv(const void* weight, const float* x, uint16_t* out, int rows, int in,
               int tokens = 1);

// The same projection through hipBLAS (route (c)). `x` is the SAME tile `proj_gemv`'s `x` was widened
// from, still in the BF16 glue wrote — the library takes a 16-bit `A` operand, so it reads what the
// producer wrote instead of a widening of it. `tokens` is the body's row count (`m`), and as in
// `proj_gemv` it is what `engine/launch_stats.h` counts this site as: one library call per projection
// per body, the same one-site-per-projection bill §06.2's "one GEMM per projection" already prices.
void proj_gemm(const void* weight, const void* x, uint16_t* out, int rows, int in, int tokens);

// ---- §03.3: gate+up+SwiGLU, then down (prototype/mlp_gemv). Both take the input widened to
// F32; there is no BF16-staged form of these two kernels, so the caller casts (see the launch
// count in results.md — this is 48 casts/token, a named follow-up).
// `tokens` is P4's token-batched form (§08.2): `x` is then a [tokens, in] tile and `act`/`out` a
// [tokens, rows] tile, and each weight row is read once per tile instead of once per token.
// `tokens == 1` issues the single-token kernel and the single-token launch shape, i.e. exactly the
// launch the decode ledger was measured on. The two forms are bit-identical per output element —
// `prototype/mlp_gemv/run.py --tokens p64` requires it (23 040 elements per layer, 0 differing).

// Which weight walk the gate/up launch uses. Q10's A/B switch, and it exists because
// `devices/strix-halo-mixed-traffic.md` §Q9 measured this kernel's own walk at 0.470 of a contiguous
// sweep at identical bytes, grid and launch count, with the LDS staging exonerated.
enum class MlpGateMap {
    // The engine's default, and NOT a third walk: it is a request that the body's row count pick one,
    // which is what `resolve_mlp_gate_map` below does. No launcher may see it — a launcher that
    // received kAuto would have to guess, and a guessed walk in a timing A/B is the wrong side of it.
    kAuto,
    // One row loop, both weight streams' words fetched side by side. What shipped before the row-count
    // default existed, and what `--mlp-map interleave` pins for every body regardless of its size.
    kInterleave,
    // The gate row end to end, then the up row end to end. Bit-identical to the above by construction
    // (same words, same order, same accumulators) — the walk is the only free variable, and
    // `prototype/mlp_gemv/run.py --mapping-identity` proves that on device rather than asserting it.
    kContig,
};

// "auto" / "interleave" / "contig", the latter two the same words the op's gate uses. Throws on
// anything else: this flag selects which side of an A/B a run is, so a silently ignored typo would
// price the wrong thing.
MlpGateMap parse_mlp_gate_map(const std::string& name);
const char* mlp_gate_map_name(MlpGateMap map);

// THE ROW-COUNT RULE, in one place: a one-row body walks `contig`, a body of two or more rows walks
// `interleave`, and an explicitly requested walk is returned unchanged (that is what keeps
// `--mlp-map` an override rather than a second opinion). `rows` is the number of token rows in the
// body about to launch, not the projection's row count.
//
// WHY THE TWO PATHS WANT OPPOSITE WALKS IS NOT ESTABLISHED, and the next reader should not invent a
// reason. Both halves are measured, on an exclusive window, against the same bytes
// (`bench/performance-status.md` §6 "Q10", `engine/results.md` §Q10b): `contig` recovers 0.945 ms of
// the MLP's 2.034 ms in-step gap at 128 lanes of decode, and `interleave` is 5.1 % better on TTFT at
// a 1024-token prompt. A per-path setting that is measured, bit-neutral and documented is what ships
// here — the same shape of decision P7 made for attention's plan and P4 for glue's row batching, both
// made from the call site because that is where the row count is. The mechanism, and what it would
// take to test, are written up in `engine/results.md` §Q10b.
MlpGateMap resolve_mlp_gate_map(MlpGateMap want, int rows);

// `map` must already be resolved: kAuto throws rather than silently becoming one of the walks.
void mlp_gateup(const uint16_t* w_gate, const uint16_t* w_up, const float* x, uint16_t* act,
                int rows, int in, int tokens, MlpGateMap map);
void mlp_down(const uint16_t* w_down, const float* act, uint16_t* out, int rows, int in,
              int tokens = 1);

// The MLP through hipBLAS (route (c)). The library unfuses exactly one epilogue in this model —
// `silu(gate)·up`, §3.3's bill of the one place the engine fuses today — so this issues two matmuls
// into `gate_raw`/`up_raw` and then the epilogue kernel that writes `act`. The epilogue is
// bit-for-bit the fused kernel's: `act = bf16( bf16(silu_f32(g)) * u )` with `g`/`u` the BF16 dots,
// three roundings, the same `expf` expression (`prototype/mlp_gemv/mlp_gemv_kernels.h`, whose F2
// gate is what pinned that policy). `x` and `act` are BF16; `act` is the same buffer the hand form
// writes, so `down`'s two forms read the same tile.
void mlp_gateup_gemm(const void* w_gate, const void* w_up, const void* x, void* gate_raw,
                     void* up_raw, uint16_t* act, int rows, int in, int tokens);
// `act` here is the BF16 `act` itself, not a widening of it — the library reads 16-bit `A`.
void mlp_down_gemm(const void* w_down, const void* act, uint16_t* out, int rows, int in, int tokens);

// ---- §04.2/§04.3/§04.4: conv, gates, recurrence (prototype/gdn_decode). All F32 on the state
// side; `raw` is F32 because the conv kernel reads it that way, which is why decode_step casts the
// BF16 `in_proj_qkv` output once per layer.
// `tokens` is P6's token-parallel form (§06.2): `raw`/`u` are then [tokens, 6144] tiles and
// `a_raw`/`b_raw`/`g`/`beta` [tokens, 16] ones, with the conv reading the [3][6144] window the
// previous wave or decode step left and writing its own last-3-columns window once per wave
// (§08.6 inv 2). `tokens == 1` issues the same grid and the same per-element arithmetic this
// launcher issued before P6 — `prototype/gdn_decode/check.py --tokens` requires the two widths to
// agree bit for bit, and the window rule (raw[k<0] is the CARRIED window, not zero) is the part
// that a wider tile could get wrong while every single-wave fixture still passed.
void gdn_conv(const float* raw, const float* win, const float* conv_weight, float* u,
              float* new_win, int tokens = 1);
void gdn_gates(const float* a_raw, const float* b_raw, const float* a_log, const float* dt_bias,
               float* g, float* beta, int tokens = 1);
// One GDN layer: 16 blocks, thread j owns column j of S. `S` is the layer's [16][128][128] block.
// Per TOKEN, and stays that way: the recurrence is a sequential scan (§04.4), so unlike the three
// launchers above it has no token width. Chunking it is P9, conditionally.
void gdn_rec(const float* q, const float* k, const float* v, const float* g, const float* beta,
             float* S, float* o);

// ---- §04.5: the gated RMSNorm (prototype/gdn_gated_norm). `o` is the recurrence's F32 output
// (the kernel rounds it to the activation dtype itself); `z` is in_proj_z's BF16 rows; `w_norm`
// is F32 in the checkpoint (§01.6) and joins as BF16, which is what the pins carry. `norm_out`
// may be null; decode_step passes the stage pointer through when dumping.
// `tokens` is P6 again: `o`, `z` and `out` are [tokens, 2048] tiles and the launch is one block per
// (token, value head) instead of one per head, so per-head reduction order — and therefore every
// bit — is unchanged at any width.
void gdn_gated_norm(const float* o, const uint16_t* z, const float* w_norm, uint16_t* out,
                    uint16_t* norm_out, int tokens = 1);

// ---- §05.1-§05.7: the whole attention block between the projections and o_proj
// (prototype/attn_decode). ONE QUERY ROW runs the fused launch plan: one launch below the lane
// staging, two above it. A WAVE of query rows runs the staged plan instead — one prep launch that
// stages every row and appends every row's cache lane, then the core (plus the merge above the
// staging) — because row t reads the lanes rows 0..t-1 appended and that ordering needs a launch
// boundary. `lvalid` is the number of cache lanes the FIRST query row may read (prefix_len + 1 during
// decode); row `t` of a body of `nrows` rows may read `lvalid + t`, which is §06.3's causal mask.
//
// TWO FORMS, chosen by the launcher per ROW and by necessity only (§07.3). A row of `lanes <= MAXSC`
// takes the one-pass softmax, which is bit-exact with the pins and is the route every priced fixture
// rides. A longer row cannot be one-pass at all — that form normalises the weight matrix inside one
// block — so it takes the op's split-K online form with `split = ceil(lanes / MAXSC)` blocks per query
// head plus one merge launch, and it needs `part` to put the partials in (see `attn_part_floats`). The
// split is taken from the WIDEST row of the body, so a wave can mix both forms in one grid: each row
// picks its own route and the extra lane blocks of a short row return before they touch anything. The
// lane ceiling is therefore `MAXSC × kAttnMaxSplit`, both constants living in `kernels_attn.cpp` where
// they are enforced, and past it the launcher throws instead of writing past a stage.
struct AttnLayerArgs {
    const uint16_t *q_proj_out, *k_proj_out, *v_proj_out, *q_norm, *k_norm;
    uint16_t *k_cache, *v_cache;            // [2][cap][256], appended at `pos` in place
    int cap, pos, lvalid;
    // P8 (§06.3): the number of QUERY ROWS this call serves, one per wave token, at positions
    // `pos .. pos+nrows-1` and lane counts `lvalid .. lvalid+nrows-1`. `nrows = 1` — a generation
    // step, and every pre-P8 caller — is the fused single-launch form, same grid and same bits as
    // before. Above one row the launcher switches to the staged plan, because a fused wave would have
    // row t read cache lanes that rows 0..t-1 are appending in blocks with no ordering between them
    // (`prototype/attn_decode/attn_decode_kernels.h`'s `attn_prep` note). The three stage buffers and
    // the two outputs are therefore [nrows][...] tiles, and the wave's q/k/v projections are read as
    // rows of the tiles P5's GEMVs wrote.
    int nrows = 1;
    uint16_t *attn_out_pre_gate, *gated;    // [nrows][8][256] each; `gated` is o_proj's operand
    uint16_t *q_rope, *k_rope, *gate_out;   // the §05.2/§05.3 stage tiles, [nrows][...] each. NOT
                                            // optional: they carry the §05.7 append, and for a wave
                                            // they are also the core's INPUT (§05.1-§05.7 staging)
    float *part;                            // [nrows][8][split][258] split-K (max,sum,acc) partials;
                                            // null below the staging, REQUIRED above it — the launcher
                                            // refuses rather than drop a row's partials on the floor
};
void attn_decode_layer(const AttnLayerArgs& a);

// How many F32 elements `AttnLayerArgs::part` must hold for a body of `rows` query rows: one (max,
// sum, acc[head_dim]) tuple per query head per split block, at the maximum split the engine serves.
// Defined in `kernels_attn.cpp`, because the head width and the split cap are that op's constants and
// `decode_step` should not have to know either one to size its scratch.
std::size_t attn_part_floats(int rows = 1);

// ---- §03.4: the tied logits GEMV (prototype/lm_head_gemv), full-logit form. The fused-argmax
// form exists but resolves ties to the highest id, and §03.4 fixes ties to the lowest, so the
// engine computes the logits and takes the argmax host-side (results.md, "the last 2 KB").
void lm_head_full(const uint16_t* table, const float* h, float* logits, int vocab);

// ---- dtype adapters, engine's own (no model math here at all). These exist only because some
// graduated kernels take F32 activations (gdn_decode's conv/gates/rec, mlp_gemv, lm_head_gemv)
// while the activations the other kernels produce are BF16.
void cast_bf16_to_f32(const uint16_t* src, float* dst, int n);
void cast_f32_to_bf16(const float* src, uint16_t* dst, int n);
// Three regions in ONE launch — this is what a per-layer GDN cast costs instead of three launches.
// `kernels_cast.cpp` defines it; a region may be empty (n = 0) but a non-empty region with a null
// pointer is a throw, not a wild write.
void cast_bf16_to_f32_3(const uint16_t* s0, float* d0, int n0, const uint16_t* s1, float* d1,
                        int n1, const uint16_t* s2, float* d2, int n2);

}  // namespace engine::kernels
