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
    // The shipped walk: one row loop, both weight streams' words fetched side by side. The default,
    // so every run that does not ask for the other one is the run it was before this switch.
    kInterleave,
    // The gate row end to end, then the up row end to end. Bit-identical to the above by construction
    // (same words, same order, same accumulators) — the walk is the only free variable, and
    // `prototype/mlp_gemv/run.py --mapping-identity` proves that on device rather than asserting it.
    kContig,
};

// "interleave" / "contig", the same two words the op's gate uses. Throws on anything else: this flag
// selects which side of an A/B a run is, so a silently ignored typo would price the wrong thing.
MlpGateMap parse_mlp_gate_map(const std::string& name);
const char* mlp_gate_map_name(MlpGateMap map);

void mlp_gateup(const uint16_t* w_gate, const uint16_t* w_up, const float* x, uint16_t* act,
                int rows, int in, int tokens = 1,
                MlpGateMap map = MlpGateMap::kInterleave);
void mlp_down(const uint16_t* w_down, const float* act, uint16_t* out, int rows, int in,
              int tokens = 1);

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
