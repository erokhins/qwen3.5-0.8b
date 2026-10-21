#pragma once
// One greedy decode step (§07.1), assembled only out of the launchers in `kernels.h`.
//
// The order of everything in `decode_step.cpp` is §07.1's, and the arithmetic inside each launch is
// the graduated kernel's — nothing here computes a value. What this file does own is the buffer plan,
// the norm/residual pairing, and the dtype seam (which activation has to be widened before which
// kernel will accept it), and those are the parts worth reading here rather than in the kernels.
//
// The §03.2 pairing, which is the one thing a reader can get quietly wrong: layer `l`'s mixer input
// norm has NO residual add in front of it (layer 0's is a norm-only call), while the add after the
// mixer and the add after the MLP are each fused with the NEXT norm — the post-attention norm
// carries the mixer's add, and the next layer's input norm carries the MLP's. That is `prototype/glue`'s
// gated pairing verbatim (`glue_pins.py:w_g1/w_g2/w_in`: call 2 of layer 23 uses `lm.norm.weight`),
// and it is what makes the whole token 1 + 24×2 = 49 glue launches with the final norm falling out
// of the last one for free.

#include <hip/hip_runtime.h>   // hipEvent_t for the embedding-gather id staging below

#include <cstdint>
#include <vector>

#include "engine/kernels.h"   // MlpGateMap: a run-shape choice the launchers take (see below)
#include "engine/state.h"

namespace engine {

class WeightStore;

// Per-step working set. Everything here is reused by every layer of every step; nothing in it
// survives a call to `decode_step`. Sizes are the model's own constants from `state.h`,
// so a shape change cannot leave a short buffer behind. The layout is documented in decode_step.cpp
// next to the buffers' use, because which kernel wants which dtype is the reason each one exists.
//
// P4 split the buffers into two families, and P6 moved one line of that split. The ROW family —
//   embed, s_a, s_b, norm, norm_f32, delta, mlp_act, mlp_act_f32, the projections' inputs and outputs
//   (block_out, the `g_*` rows a projection writes, the `a_*` rows a projection writes), and since P6
//   the GDN mixer's own `g_u`/`g_o`/`g_g`/`g_beta` — is `[wave_rows, H]` (or `[wave_rows, I]`,
//   `[wave_rows, 6144]`, …). A prefill wave (§08.2) fills one row per token and a decode step uses
//   row 0, which is why decode is unchanged by their being taller. P6 moved `g_u`/`g_o`/`g_g`/`g_beta`
//   across because their producing and consuming launches are now one call per wave (the conv writes
//   the whole `u` tile; the gated norm reads the whole `o` tile), which is exactly the test the split
//   uses.
//   * what is left one row wide is what no launch writes in bulk: the conv window's `conv_new` (a
//     K−1-row window, not a row family). Everything the attention block touches is a row family now:
//     P7 split that block's LANE axis (above the staging it is `split` blocks per query head, with
//     `a_part` holding their partials), and P8 gave it the TOKEN axis (§06.3: one site per attention
//     layer per body, so `a_*_rope`/`a_gate`/`a_pre`/`a_gated` are `[wave_rows]` tiles and `a_part` is
//     sized for a whole wave of rows). The GDN recurrence is per token too, but it reads and
//     writes ROWS of the tall tiles above rather than its own scratch, so it did not keep a one-row
//     buffer with it.
constexpr int kMaxWaveRows = 512;   // plan §2.1; decode_step.cpp explains why 512 and not 1024
                                    // `allocate_scratch` accepts less, and `decode_run --wave-rows 1`
                                    // deliberately asks for the least: a prompt fed one token per wave
                                    // is the run `engine/launch_stats.h` exists to be able to see.

struct DecodeScratch {
    // How many rows the row family below holds. `submit_wave_body` refuses a wave longer than this,
    // and `prefill_wave` splits at it, so no launcher here has to re-check.
    int wave_rows = 0;
    // Which weight walk the MLP's gate/up launches use (Q10's A/B switch, `kernels.h:MlpGateMap`).
    // It rides here for the same reason `wave_rows` does: it is a run-shape choice, it has to reach
    // the one call site in `submit_wave_body`, and `decode_run` is the only thing that sets it.
    // The default is `kAuto` — the body's row count picks the walk, per body (Q10b) — so a scratch
    // nobody configured runs the row-count default, and `--mlp-map interleave|contig` pins one walk
    // for every body instead of asking. `submit_wave_body` resolves it where `n` is known.
    kernels::MlpGateMap mlp_gate_map = kernels::MlpGateMap::kAuto;
    // Which GEMM a WIDE body uses (route (c): hipBLAS for the ten fat shapes, hand GEMVs for the rest
    // of the tree's life). It rides here for `mlp_gate_map`'s reason — a run-shape choice that has to
    // reach the one call site where the body's row count is known — with one difference worth naming:
    // `resolve_wave_gemm` overrides it for a one-row body, always, because decode's byte-identity is
    // not a setting. `decode_run --gemm hand` is the A/B side; `submit_wave_body` resolves per body.
    kernels::WaveGemm wave_gemm = kernels::WaveGemm::kLibrary;
    // Which attention FORM a wave body runs (Phase 2, `kernels.h:AttnLayerArgs`). Same riding reason as
    // the two above, and it too is overridden for a one-row body — the fused single-row form is what
    // decode runs whatever this says. The default is the SHIPPED form, not the tiled one: the tiled
    // core's device gate has not been run yet (`prototype/attn_tile_model.py` has priced it on CPU
    // only), and a form that has not met the device does not become the engine's default by having been
    // written. `--attn-tile 1` is the A/B side.
    int attn_tile = 0;
    // Which SPELLING of the tiled wave's touch loop runs (Phase 2 follow-up, `--attn-touch`,
    // `kernels.h:AttnLayerArgs::wave_touch`). Same riding reason as the four above and the same override
    // for a one-row body — the fused single-row form is what decode runs whatever this says, so no decode
    // bit moves under it either. The default is 1, the slim spelling, because the two spellings were
    // proved bit-identical on the prototype harness's 12 plans before being ported; `engine/gate.py`
    // re-proves it on the engine path against the pinned dumps at BOTH values, which is the device pass
    // `progress/2026-10-16/HANDOFF-attn-touch.md` hands over. `--attn-touch 0` is the frozen side of that A/B.
    int attn_touch = 1;
    // Which attention FORM owns a wave body's STREAMING rows (`--attn-fattn`,
    // `kernels.h:AttnLayerArgs::wave_fattn`). Same riding reason as the two above, and the same override
    // for a one-row body — decode runs the fused single-row form whatever this says, so no decode bit
    // moves under it. The default is 0: the WMMA form has been written and its two map checkers are green,
    // but the form itself has not met the device (`prototype/attn_wmma/`), and a form that has not met the
    // device does not become the engine's default by having been written.
    int attn_fattn = 1;   // default since the form met the device: 1.3955x attn at 4096 lanes, wide1024f green both ways, covered rows inside the floors in force
    // Which SPELLING of that form's score roll (`--attn-fattn-roll`,
    // `kernels.h:AttnLayerArgs::fattn_roll`). Same riding reason and the same one-row immunity: a one-row
    // body is the fused `attn_core`, so no decode bit moves under it. `kSingle` is the shipped roll and the
    // default, and unlike the flag above it a default that is also HEAD's code path — the two-accumulator
    // side is bet A4-L1 and is only ever run by naming it (`prototype/attn_percost/A4-BET.md`).
    kernels::FattnRoll attn_fattn_roll = kernels::FattnRoll::kSingle;
    // Where that form keeps Q, and how deep its epilogue stages (`--attn-fattn-qstage`, bet A6-G1; see
    // `kernels.h:AttnLayerArgs::fattn_qstage`). Same riding reason and the same one-row immunity: a one-row
    // body is `attn_core`, so no decode bit moves under it. `kUnstaged8` is the default from A6-G1: the form
    // that takes Q out of LDS and stages the epilogue 8 rows at a time, measured LIVE on all five rows with
    // byte-identical logits and STATEFP (`prototype/attn_percost/A6-G1-verdict.md`), while `kStaged` remains
    // selectable and stays byte-identical to the pre-A6-G1 engine — which is why the default here carries no
    // numerics negotiation at all. Both sides are byte-compares, not bands.
    kernels::FattnQStage attn_fattn_qstage = kernels::FattnQStage::kUnstaged8;
    // What width that form's P*V accumulator is (`--attn-fattn-pvf16`, bet A10-F16; see
    // `kernels.h:AttnLayerArgs::fattn_pvf16`). Same riding reason and same one-row immunity as the two above: a
    // one-row body is `attn_core`, so no decode bit moves under it. 0 — the shipped F32 accumulator, 128
    // registers a lane — is the default and is HEAD's own machine code, so the default side of its A/B is a
    // byte-compare. 1 is the F16 accumulator: 64 registers, the same 256 dims, and NOT owed byte identity (a
    // different format), so it is judged by the bands already in `engine/gate.py` (`prototype/attn_percost/A10-F16-IMPL.md`).
    int attn_fattn_pvf16 = 0;
    // Whether a wave body's core is the A12 wide block instead of the 16-row WMMA one (`--attn-fattn-wide`,
    // bet A12; see `kernels.h:AttnLayerArgs::fattn_wide`). Same riding reason and the same one-row immunity as
    // the four above — a one-row body is `attn_core`, so no decode bit moves under it — and one difference
    // worth naming: the knobs above it choose a SPELLING of the form that ships, this one replaces the whole
    // wave core and drops the split/combine route for that body. This STRUCT default stays 0 on purpose and is
    // NOT the shipping default: `decode_run`'s own option struct carries 2 since A13 shipped, and it assigns
    // this field unconditionally, so the library default only reaches callers that leave it unset — the
    // standalone rigs, whose published baselines were all taken on the inert form and would otherwise move
    // under a flag nobody passed. Inertness at 0 is a byte-compare rather than an argument, because the wide
    // kernel is a separate code object.
    int attn_fattn_wide = 0;
    // How deep the split path may cut a wide row (`--attn-split-cap`, bet A9-L-c;
    // `kernels.h:AttnLayerArgs::split_cap`). Same riding reason as the five above — it has to reach the one
    // call site in `submit_wave_body` — with one difference worth naming: unlike them it is NOT wave-only. A
    // one-row body is the fused `attn_core` whatever it says, but that core has the same split-K streaming
    // route, so a cap re-associates a long DECODE row's merge as well as a wave's (at cap 8 that starts at
    // 4,608 lanes). `kAttnSplitCapOff` (0) is the default and is inert by construction: the split stays
    // `split_for(lanes_last)`, so the shipped bits do not move.
    int attn_split_cap = kernels::kAttnSplitCapOff;
    // Which §04.4 scan a GDN body runs (P9, `kernels.h:GdnScan`). Same riding reason as the three
    // above, and it too is overridden per body — a ONE-ROW body is `gdn_rec` whatever this says,
    // because decode's bits are pinned through the sequential recurrence and a chunked scan of one
    // token has nothing to win. The default is `kSeq`: the chunked form's device gate has not run,
    // and a form that has not met the device does not become the engine's default by being written.
    kernels::GdnScan gdn_scan = kernels::GdnScan::kSeq;
    // Which SPELLING of the sequential recurrence's state walks a `gdn_rec` launch runs
    // (`--gdn-spell`, `kernels.h::GdnRecSpell`). Same riding reason as the four above: it has to reach
    // the launch site, which is inside the body loop. The default is `ship` — the shipped kernel —
    // because the re-spelling's 1.15x is a burst number and D30 item 1.9's in-situ A/B has not been
    // read; `unroll4` is that A/B side and it is bit-identical (`engine/gdn_rec_unroll4_kernels.h`).
    // Inert under `kCollapse`, which submits a different kernel, and `decode_run` refuses the pair
    // rather than price the shipped form twice under the other flag's name.
    kernels::GdnRecSpell gdn_spell = kernels::GdnRecSpell::kShip;
    // How many token-steps one `gdn_rec` launch carries (`--gdn-loop`, D53/D54,
    // `kernels.h:resolve_gdn_loop`). Same riding reason as the five above: it has to reach the launch
    // site, which is inside the body loop, and `submit_wave_body` resolves it where `n` is known.
    // Same one-row override too — a ONE-ROW body walks the per-token loop whatever this says, so no
    // decode bit can move under it. The default is 1, the shipped one-launch-per-token form: D54
    // ships the loop as a knob first (D32 → D50's order again), because its 1.58–1.64x is a harness
    // screen (`progress/2026-10-21/performance-status-2026-10-21.md` §23) and the paired in-situ A/B on the `gdn_rec` family
    // row is what decides whether the default moves — it moved in §24 (1.63× family row, 1,024/1,024
    // prefill logits rows byte-identical), so this mirrors `decode_run`'s option default and the two
    // must move in one commit. Inert under `kCollapse`, which already submits
    // one launch per body, and `decode_run` refuses the pair rather than price route A twice under
    // this flag's name.
    int gdn_loop = 512;
    // Where the looped recurrence's per-token k/q L2-norms are computed (`--gdn-norm`, bet G1,
    // `kernels.h:GdnNorm`). Same riding reason as the five above: it reaches the launch site inside the
    // body loop. Same one-row override too — a ONE-ROW body takes the per-token kernels, which keep
    // their own norm prologue, whatever this says, so no decode bit moves under it. The default is
    // `kInline`, the shipped form: the hoist's slot saving is 9.2 % of the per-token bill
    // (`prototype/gdn_walk/evidence-tokenbill-2026-10-19.txt`) against +2.4 GB of stage traffic per
    // 4,096-lane pass, and BET.md §3's pre-registered lines on the `gdn_rec` family row are what decide
    // whether the default moves. Inert under `kCollapse`, which submits a different kernel.
    kernels::GdnNorm gdn_norm = kernels::GdnNorm::kInline;
    // The BLOCK SHAPE of the looped body (`--gdn-geom`, bet G2, `kernels.h:GdnGeom`). Same riding reason
    // as the six above: it reaches the launch site inside the body loop, and the same one-row override —
    // a ONE-ROW body takes the per-token kernels whatever this says, so no decode bit moves under it.
    // The default is `kBlock`, the shipped `<<<16,128>>>` looped walk; `kWf4` is BET.md §10's
    // pre-registered one-wavefront-per-block form (no LDS, no barrier, norms by warp butterfly) and `kWf2`
    // its split leg. It does NOT claim bit-identity — the state and `o` dots re-associate — and the band is
    // the existing one (§10.1 aggregates via `engine/gate.py`). Inert under `kCollapse` and under
    // `--gdn-loop 1`, and `decode_run` refuses those pairs rather than echo a shape that did not run.
    kernels::GdnGeom gdn_geom = kernels::GdnGeom::kReg16;
    // How a ONE-ROW body GROUPS its projection launches (route D1, `kernels.h:ProjPlan`), and which
    // kernel and block geometry such a launch uses (`kernels.h:ProjShape`). Both ride here for the usual
    // reason — a run-shape choice that has to reach the call sites inside the body loop — and both carry
    // the shipped default, because the prototype's −24.5 % / −11.7 % are burst numbers and the in-situ
    // `proj` family row is what decides whether the defaults move (`HANDOFF-D1-ENGINE.md`).
    //
    // The rule the fields cannot express and `submit_wave_body` enforces: BOTH apply to a body of ONE row
    // and to nothing else. A wide body's projections are already one launch per projection over the whole
    // tile (P5) and usually a library call (route (c)); grouping them would change a prefill number the
    // prototype never measured. `kPacked` is therefore meaningless with `kShipped` — the grouping is a
    // feature of the grouped kernel — and `decode_run` refuses that pair instead of echoing it.
    kernels::ProjPlan proj_plan = kernels::ProjPlan::kSplit;
    kernels::ProjShape proj_shape = kernels::ProjShape::kShipped;
    // How many one-row projection launches asked for a shape this device/width cannot offer and took the
    // shipped kernel instead. Counted, not assumed: `decode_run` prints it, so a run whose A/B side
    // silently fell back says so in its own log instead of reading as a null (`AGENTS.md`: a check whose
    // input can silently fail to arrive must be shown to have arrived).
    long long proj_fallbacks = 0;
    // Route S1 item (1): fold the BF16->F32 activation widening into the kernel that consumes the
    // result, deleting the `cast` launches (`engine/generation-limits.md` §1's 97-launch row, whose
    // 0.371 MiB read / 0.741 MiB written per token is no bytes of consequence — the whole line is
    // per-launch cost, and whether that cost is real in the pipelined shape is what this flag exists
    // to measure).
    //
    // false is the shipped behaviour and HEAD's machine code: every widening is submitted by
    // `kernels_cast.cpp` exactly where it always was. true makes each consumer stage the BF16 tile
    // itself and widen in registers, which is bit-identical because `hip_bfloat16 -> float` is exact
    // (`engine/mlp_xbf16_kernels.h` states the argument once). It is a RUNTIME knob rather than a
    // build define so both sides live in one binary and the paired A/B is one window — the `--attn-touch`
    // precedent, and the reason a base binary and a new binary can be compared without a rebuild.
    //
    // The rule these fields cannot express and `submit_wave_body` enforces, the same one `proj_plan`
    // and `proj_shape` carry: the per-layer folds apply to a body of ONE row and to nothing else, so a
    // prefill wave runs the kernels it always ran. `decode_run` sets this from `QWEN_CAST_FOLD` (see
    // `decode_run.cpp` for why an environment default rather than a command-line flag).
    bool cast_fold = false;
    // The chunk size in tokens, echoed so a run states which chunk it claimed. Only 64 is accepted
    // (`kernels.h:kGdnChunkTokens`); the refusal text carries the measured reason.
    int gdn_chunk = kernels::kGdnChunkTokens;
    // Residual ping-pong. NOT aliased into one buffer: `glue_add_norm`'s `s` and `s_out` are both
    // `__restrict__`, so writing the new residual over the one it is reading is a contract violation
    // even though the same thread touches the same index.
    uint16_t* s_a = nullptr;        // [wave_rows][1024]
    uint16_t* s_b = nullptr;        // [wave_rows][1024]
    uint16_t* norm = nullptr;       // [wave_rows][1024] the glue norm output, BF16 (what the pins
                                    // carry). Row `T-1` is what `finish_token` prices by default;
                                    // a wave under `--prefill-logits all` prices every row.
    float* norm_f32 = nullptr;      // [wave_rows][1024] the same rows widened, for proj_gemv /
                                    // mlp_gateup — one cast launch per wave, not per token
    uint16_t* embed = nullptr;      // [wave_rows][1024] E[y], the residual layer 0 starts from
    uint16_t* delta = nullptr;      // [wave_rows][1024] a block's or the MLP's residual addend
    uint16_t* block_out = nullptr;  // [wave_rows][2048] gated_out / gated, i.e. the 2048-wide proj operand
    float* block_out_f32 = nullptr;  // [wave_rows][2048] the same, widened, for out_proj / o_proj

    // GDN (§07.2). `raw`/`u` are 6144 wide because in_proj_qkv is [q|k|v]; `a`/`b`/`g`/`beta` are the
    // 16 value heads. `conv_new` is the rolled window — see the conv note in decode_step.cpp.
    // The three F32 widenings this struct used to carry (`g_raw_f32`, `g_a_f32`, `g_b_f32`) are gone:
    // `conv_kernel` and `gates_kernel` read the BF16 tiles below and widen their operands
    // in-register (D30 1.7'), so an F32 copy of them had no reader at any width or on either route.
    uint16_t* g_raw = nullptr;      // [wave_rows][6144] in_proj_qkv out, BF16 as the GEMV wrote it;
                                    // `conv_kernel`'s tap operand, widened in the kernel
    uint16_t* g_z = nullptr;        // [wave_rows][2048] in_proj_z out, stays BF16 (§04.5 reads it as BF16)
    uint16_t* g_a = nullptr;        // [wave_rows][16]   in_proj_a out — `gates_kernel`'s operand, as BF16
    uint16_t* g_b = nullptr;        // [wave_rows][16]   in_proj_b out, ditto
    float* g_dt_bias = nullptr;     // [kNumGdnLayers][16]: dt_bias is BF16 in the store (§01.6
                                    // keeps only A_log and linear_attn.norm.weight F32) and
                                    // gates_kernel reads `const float*`, so it is widened ONCE,
                                    // by prepare_weight_adapters, per GDN slot.
    float* g_conv_w = nullptr;      // [kNumGdnLayers][kConvChannels][kConvKernel]: same reason,
                                    // same one-time widening (conv_kernel takes `const float* C`).
    float* g_g = nullptr;           // [16] decay log
    float* g_beta = nullptr;        // [16]
    float* g_u = nullptr;           // [6144] post-conv, q|k|v in one row (§07.2 slices it)
    float* g_o = nullptr;           // [2048] the recurrence's o, F32 (§04.5's input dtype)
    float* conv_new = nullptr;      // [kConvWindow][kConvChannels]

    // P9's chunked-scan staging (§04.7). Body-scoped and reused per layer in stream order, exactly
    // like `g_u`/`g_o`: nothing here is §08.1 state, because the only thing that survives a body is
    // the [16][128][128] state block itself. Written by `gdn_chunk_prep`, read by `gdn_chunk_carry`.
    // `nv` is F32 (a result being subtracted); the other four are BF16 because each is a matmul
    // OPERAND on the carry side, which is where the CPU model rounds them. Allocated unconditionally
    // (see `allocate_scratch`), so a `seq` run never touches it rather than not having it.
    float* g_nv = nullptr;          // [wave_rows][16][128] F32
    uint16_t* g_kcd = nullptr;      // [wave_rows][16][128] BF16
    uint16_t* g_qg = nullptr;       // [wave_rows][16][128] BF16  q~ * Gamma
    uint16_t* g_kh = nullptr;       // [wave_rows][16][128] BF16  k~ * (Gamma_C / Gamma)
    uint16_t* g_intra = nullptr;    // [wave_rows][16][kGdnChunkTokens] BF16
    float* g_cdec = nullptr;        // [ceil(wave_rows/kGdnChunkTokens)][16] F32

    // Bet G1's norm staging (`--gdn-norm hoist`, `engine/gdn_rec_hoist_kernels.h`): the normalised k and
    // q rows and the two scalars the recurrence needs, written by the pre-pass and read by the looped
    // walk, one row per token-step of the body and reused per layer in stream order. Body-scoped
    // scratch like the six above, NOT §08.1 state. At the 512-row ceiling this is 4 MiB + 4 MiB + 64 KiB
    // per request, which `AGENTS.md` is right about: memory capacity is not this model's constraint.
    // Allocated unconditionally for the chunked staging's reason — a conditional allocation would be a
    // path the default run cannot exercise.
    float* g_kn = nullptr;          // [wave_rows][16][128] F32  k * rsqrtf(sum k^2 + eps)
    float* g_qn = nullptr;          // [wave_rows][16][128] F32  q * rsqrtf(sum q^2 + eps) * Q_SCALE
    float* g_gsc = nullptr;         // [wave_rows][2*16] F32     expf(g) for each head, then beta

    // Attention (§07.3). The three stage buffers are required, not optional — `kernels_attn.cpp`
    // says why (the cache append lives behind the same null check as one of them). From P8 they are
    // wave tiles: one attention site serves every row of the body (§06.3), so row t of each is token
    // t's. At wave_rows = 1 they are the shapes they always were.
    uint16_t* a_q = nullptr;        // [wave_rows][4096] q_proj out: [8 heads][q 256 | gate 256], §05.1
    uint16_t* a_k = nullptr;        // [wave_rows][512]
    uint16_t* a_v = nullptr;        // [wave_rows][512]
    uint16_t* a_q_rope = nullptr;   // [wave_rows][2048]
    uint16_t* a_k_rope = nullptr;   // [wave_rows][512]
    uint16_t* a_gate = nullptr;     // [wave_rows][2048] the de-interleaved output gate, §05.6
    uint16_t* a_pre = nullptr;      // [wave_rows][2048] attn_out_pre_gate, §05.5
    uint16_t* a_gated = nullptr;    // [wave_rows][2048] pre * σ(gate), o_proj's operand
    // The split-K online softmax's partials (§07.3, P7): one (max, sum, acc) tuple per query head per
    // lane block, only ever written by a row too long for the one-pass staging. It lives here rather
    // than in `RequestState` because it is not §08.1 state — nothing survives the launch that wrote
    // it, and `(g)`'s fingerprints must keep meaning "model state" — and it is ONE buffer for the
    // whole request because the engine submits in stream order: `attn_combine` reads a row of `part`
    // before the next attention layer is allowed to overwrite it. `kernels_attn.cpp` sizes it for a
    // whole wave of rows, since from P8 a wave is what a site serves.
    float* a_part = nullptr;        // [wave_rows][8][kAttnMaxSplit][258] F32, attn_part_floats(wave_rows)

    // MLP (§03.3). `act` is the F32 intermediate `down_proj` wants, `mlp_act` the BF16 one the
    // gate is pinned against; both exist because the two kernels disagree about the dtype.
    uint16_t* mlp_act = nullptr;    // [wave_rows][3584]
    float* mlp_act_f32 = nullptr;   // [wave_rows][3584]
    // The library route's two raw dots, the price of the one epilogue it unfuses (§3.3): hipBLAS
    // writes `gate` and `up` as BF16 `C` and `mlp_silu_mul_kernel` reads them to write `mlp_act`.
    // Untouched by the hand route, which fuses the epilogue inside the kernel that computes the dots.
    uint16_t* mlp_gate_raw = nullptr;  // [wave_rows][3584]
    uint16_t* mlp_up_raw = nullptr;    // [wave_rows][3584]

    float* logits = nullptr;        // [kVocab] F32 — §03.4's full-logit form, see kernels_lmhead.cpp

    // §02.3's wave gather (D30 4.2′): the body's token ids, where the gather kernel can read them.
    //
    // WHY THESE EXIST AT ALL. `submit_wave_body` is handed a HOST id array, and a gather kernel needs
    // it on the device, so one body costs one H2D upload plus one gather where it used to cost `n`
    // memcpys. What is NOT allowed is a copy per token, which would be the item's own bug in miniature.
    //
    // WHY THE STAGING IS PINNED, AND WHY IT IS A RING. The upload must be `hipMemcpyAsync` from *pinned*
    // host memory: from pageable memory HIP's async copy is synchronous, and a synchronous H2D on the
    // legacy stream drains everything queued in front of it — i.e. the whole previous body — which would
    // flatten exactly the queue depth this item exists to exploit. And a pinned buffer cannot simply be
    // reused by the next body: the host runs bodies ahead of the device, so a single slot would be
    // overwritten while an earlier body's copy still had to read it. Therefore `kEmbedIdSlots` slots,
    // each with an event recorded right after its copy, and a body waits on its slot's event before it
    // writes it. That wait is a correctness bound, and it is honest about what it costs: it caps how far
    // ahead of the device the host may submit at `kEmbedIdSlots` bodies. At 512 rows a body is ~120 ms of
    // device work and the host submits one in well under a millisecond, so the queue still holds seven
    // whole bodies in front of the device at every wait and nothing can starve — the bound bites only on
    // a prompt long enough to wrap the ring, and it buys correctness on exactly the path that wraps it.
    // The first `kEmbedIdSlots` bodies a scratch serves do not attempt the wait at all, and a shipped
    // 4096-lane wave is exactly 8 bodies, so the window this item is measured in is wait-free.
    //
    // Allocated with the scratch and reused by every body of every request, like the row family above;
    // `kEmbedIdSlots * kMaxWaveRows` ints is 16 KiB on each side, which is not what this is about.
    static constexpr int kEmbedIdSlots = 8;
    int* token_ids_dev = nullptr;                    // [kEmbedIdSlots][wave_rows] device
    int* token_ids_pin = nullptr;                    // [kEmbedIdSlots][wave_rows] pinned host
    hipEvent_t token_id_copied[kEmbedIdSlots] = {};  // completed once that slot's upload has run
    long long token_id_bodies = 0;       // bodies served, so the slot is derivable and a first-pass
                                         // body knows its slot was never written, let alone uploaded

    // Set by `prepare_weight_adapters`, which is where `g_dt_bias`/`g_conv_w` get filled. Zeroed by
    // `allocate_scratch`, so a forgotten call is a refusal rather than a read of uninitialised memory.
    bool adapters_ready = false;
};

DecodeScratch allocate_scratch(int wave_rows = kMaxWaveRows);
void free_scratch(DecodeScratch& sc);

struct StepResult {
    int token = -1;            // §03.4 argmax; ties go to the LOWEST id (strict `>` on the host)
    float top_logit = 0.0f;    // the value at the argmax, for the log and the gate's report
};

// §07.5: greedy decoding stops at id 248046 (the end-of-turn token, A-10), at end of context, or at
// the caller's budget. `finish_token` sets `stopped` whenever its argmax IS that id — in either phase,
// because it is not told which — so whether the match ENDS a run is the caller's decision, and for the
// prompt phase (a prefill wave, `engine/prefill.h`) it never is: a prompt-phase sample is discarded, so
// a match there is a diagnostic, not a stop. `decode_run.cpp` owns that rule and says why.
constexpr int kStopToken = 248046;

// ---- the token, in two pieces ---------------------------------------------------------------
//
// The §07.1 order lives in TWO functions, and `decode_step` below is their composition. The split is
// not decoration: a prefill wave (§06.1, `engine/prefill.h`) has to submit T tokens back to back and
// price only the last one, which is §06.5's "only `logits[T-1]` is ever used". With the tail welded
// on, a wave that called `decode_step` per token would pay T `lm_head_full`s, T 1 MB vocabulary D2Hs
// and T syncs to throw T-1 of them away — the single largest piece of waste §1.2 names. So the seam
// is cut here, once, arithmetic-free, and P4/P5/P6 reuse it: they batch the launches *inside*
// `submit_token_body`, and the wave's one-finish shape does not change when they do.
//
// WHO SPLITS THEM: `prefill_wave` (`engine/prefill.cpp`) is the only caller that calls the two
// apart, and it does so T-bodies-then-one-finish. Every other caller uses `decode_step`, which is
// defined as body-then-finish, so there is still exactly ONE implementation of the §07.1 order.

// The body: `x = E[y]` → input norm → 24 blocks (GDN §07.2 or attention §07.3 by layer type, each
// followed by the MLP §03.3 and the fused add+norm) → the final norm falling out of the last glue
// call. Submits 339 of the token's 340 launches (`planned_launch_counts()` derives the bill; the
// 340th is the `lm_head_full` that belongs to the tail) and WAITS FOR NONE OF THEM: no logits, no
// D2H, no sync, no argmax, so bodies can be queued back to back on one stream. Under the pre-D1
// `--proj-plan split` pair the same body submits 405 of that plan's 406; only the `proj` row moves.
//
// It still advances the state, because the next token's position depends on it: `pos` is
// `st.prefix_len` on entry and `st.prefix_len` is `pos + 1` on exit, the GDN state and conv window
// move, and lane `pos` of the KV is written (§08.6's invariants are re-checked before it returns).
// What it does NOT touch is `sampled_token`/`stopped` — there is no sample until `finish_token` —
// and it leaves the normed final row in `sc.norm` for `finish_token` to read.
//
// Throws std::runtime_error on a HIP failure, on a lane count the attention launcher cannot serve
// (staging × split, §07.3 in kernels_attn.cpp), on a token outside [0, vocab), or if
// `prepare_weight_adapters` has not been called on this scratch.
void submit_token_body(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                       int token);

// The wave body (P4, extended by P5): the SAME §07.1 order over `n` consecutive tokens, with the
// families that are legal to batch batched and the mixer left per token. `submit_token_body` is this
// with n = 1, so there is still exactly one implementation of the order — the n = 1 launch sequence is
// the one the decode ledger was measured on, launch for launch.
//
// The batching is layer-major, which is the only way it can be: per layer, all `n` tokens' mixers
// run in position order (the GDN recurrence and the causal KV append need that order and keep it),
// then ONE glue launch over the [n, H] tile, then ONE batched MLP over the tile, then ONE more glue
// launch. Tokens are independent of each other within a layer — that is what §08.2's chunked prefill
// is — and each layer's scan still visits the tokens in ascending position, so the state a layer
// hands to the next one is the same state the per-token order produced.
//
// What is batched, and what is NOT yet: glue (row-indexed, `prototype/glue --batched`), the MLP
// (`--tokens`) and the nine projections (`prototype/proj_gemv --tokens`) are; the GDN conv/gates/rec
// and the attention row are still per token (P6/P7/P8). The batched forms are bit-identical to the
// per-token ones on device (612 352 elements, 0 differing in glue; 23 040 per layer in the MLP;
// 138 400 per fixture in the projections), which is the reason this is allowed to touch the legal
// route at all. Projections are the family whose reduction is a long dot over `hidden`/`heads`, so
// unlike the other two this one COULD legitimately have moved bits; the per-op gate is what rules it
// out (`prototype/proj_gemv/results.md`, P5 section).
//
// `n` may not exceed `sc.wave_rows`; `prefill_wave` splits at that bound so a caller never has to.
// Positions are `st.prefix_len .. st.prefix_len + n - 1`, and `prefix_len` advances by `n` at the
// end, with §08.6's invariants re-checked once for the wave instead of once per token.
void submit_wave_body(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                      const int* tokens, int n);

// The tail: `lm_head_full` over the normed row the last `submit_token_body` left in `sc.norm`, then
// exactly ONE blocking D2H of the vocabulary (this is the sync of the token, and the only one), then
// §03.4's host argmax — a strict `>` scan, so ties go to the LOWEST id. Sets `st.sampled_token`,
// and `st.stopped` when that id is `kStopToken` (§07.5's raw match, any phase: the caller decides
// whether it ENDS anything, and for a prefill wave it never does).
//
// THE CONTRACT, stated because the wave is what leans on it: this prices row `norm_row` of
// `sc.norm` — row 0 for a decode step, and `n-1` for a wave's `y0` (the body leaves every row of the
// wave in the tile). The pair must therefore be called body→finish, and under
// `PrefillLogits::kAll` the caller prices rows 0..n-1 after the wave body, in order. `logits` may be
// null; when it is not, the full vocabulary comes back in it (≈1 MB, which is the price §03.4's tie
// rule and this gate both accept). Nothing here reads the state or the weights' shapes again.
StepResult finish_token(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                        std::vector<float>* logits = nullptr, int norm_row = 0);

// One greedy decode step (§07.1): `submit_token_body` then `finish_token`, which is the whole
// definition. Same launches, same order, one sync (in the tail) as when this was one function.
// `st` is mutated in place: GDN state, conv window and KV advance, `prefix_len` increments, and
// `sampled_token`/`stopped` are set (§07.5).
StepResult decode_step(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                       int token, std::vector<float>* logits = nullptr);

// Two BF16 weight rows that the graduated kernels want in F32, widened once per request instead of
// once per token: `linear_attn.conv1d.weight` and `linear_attn.dt_bias`.
//
// This is not a dtype conversion the model asked for, it is a dtype the two halves disagree about.
// §01.6 keeps exactly two tensors F32 per GDN layer — `A_log` and `linear_attn.norm.weight` — and
// `engine/loader_test.cpp` counts 36 such tensors, so the other two operands `gates_kernel` and
// `conv_kernel` take as `const float*` arrive BF16. Widening them per token would cost 18 launches
// and ~1.8 MB of traffic for 36 rows that never change, so they are widened into the scratch once,
// exactly as `state.h`'s conv-dtype note prices the same trade the other way. `decode_step` refuses
// to run before this is called.
void prepare_weight_adapters(const WeightStore& weights, DecodeScratch& sc);

// The launch plan of one token, by family, derived from the same §00.3 constants `decode_step`'s loop
// iterates over — not a hand-typed total. It is the engine's launch BILL (at the measured 1.85 µs
// dispatch floor, `tools/launch_overhead_hip.cpp`), which is what `engine/results.md`'s two deferrals
// (packed projections, a BF16-staged MLP) are priced against. `gate.py` checks these numbers against
// its own independent reading of §07, so the C++ and the Python cannot silently drift.
//
// `cast_fold` is route S1 item (1): whether the run widens its activations in the kernels that read
// them, in which case a ONE-ROW body — the body this bill describes — submits NO `cast` launches at all,
// because all six per-token sites fold (site by site: `prototype/cast_fold/census_sites.py`). It is a
// parameter with a default rather than a global the function reads for the reason `planned_launch_counts()`
// already exists in this shape: the `LAUNCH` line prints the default (§07's shipped bill, 97/340) in every
// run whatever the flags say, and that is what `gate.py` section (d) re-derives in Python. A bill that
// moved with the flags would make that cross-check a tautology — the `--attn-tile` and `--proj-plan split`
// trap of a gate judging a different plan than the engine ships. What the fold-in-force is a bill of is
// the STAGE clock's plan, and only that clock asks: `stageclock::Clock::set_plan_cast_fold`.
struct LaunchCounts {
    int glue = 0, proj = 0, gdn = 0, attn = 0, mlp = 0, cast = 0, other = 0;
    int total() const { return glue + proj + gdn + attn + mlp + cast + other; }
    std::string describe() const;
};
LaunchCounts planned_launch_counts(bool cast_fold = false);

// ---- route D1: the projection plan, as data (so geometry can be checked without a GPU) -------------
//
// `planned_launch_counts()` bills the SHIPPED plan (`kernels::kProjPlanDefault`, now `packed`) whatever a
// run asks for, for the reason `gate.py` section (d) exists: it re-derives that bill from §07 in Python,
// and a C++ function that quietly returned a different total once a flag was passed would make the
// cross-check a tautology — the `--attn-tile` trap of a gate judging one plan while the engine ships
// another. Before the flip this same paragraph explained why the function stayed on the SPLIT bill while
// `packed` was the flag arm; only which of the two is "shipped" changed, and the reason it must be one
// named constant rather than two call sites' opinions is unchanged. So the plan-in-force is a SEPARATE
// question with its own answer here, derived from the same §00 constants and printable device-free
// (`decode_run --dump-proj-plan`), and the counted witness that it took effect is the `LAUNCHSTAT
// family=proj calls=` line plus `STAGE … family=proj launches_per_step=`.
//
// One layer under `kPacked` is: one launch over the layer's input projections (they read the same row of
// `norm_f32`, and each writes its own row-family tile, so no weight is copied), then one launch for its
// output projection — 48 per token, the shipped bill. Under `kSplit` every tensor is its own launch, i.e.
// `proj`'s 114 — what a WIDE body pays under either plan, since the grouping is a one-row rule.
struct ProjPlanJob {
    int layer = -1;
    int group = 0;             // 0 = this layer's input group, 1 = its output projection
    int seg = 0;               // this tensor's index inside its group
    int segs = 1;              // segments in the launch this tensor becomes
    int rows = 0;
    int in = 0;
    std::string weight;        // full engine name, `lm.layers.N.…`
    const char* dest = "";     // which `DecodeScratch` tile the launch writes
    const char* kind = "";     // "gdn_in" / "attn_in" / "gdn_out" / "attn_out"
    // Whether this projection could be handed to the library route on a WIDE body. `in_proj_a`/`b` are
    // the two that cannot (§3.2, the skinny pair), and the split plan must keep calling the hand kernel
    // for them at any width — the flag must not smuggle a 16-row shape onto hipBLAS.
    bool library_eligible = true;
    // Under `kPacked` a group of tensors becomes ONE launch and only the group's first entry starts it, so
    // "how many launches does a body submit" is a count of these and not a count of entries. Derived, never
    // typed: `proj_launches_per_body` sums this field, which is why 48 and 114 cannot be edited into shape.
    bool starts_launch = true;
};

// The launches one row-priced body submits, in submission order. Pure §00/§07 structure: no device, no
// weights, so `--dump-proj-plan` can be read while another agent holds the GPU lock.
std::vector<ProjPlanJob> proj_plan_jobs(kernels::ProjPlan plan);

// `proj_plan_jobs(plan).size()`, for the one-line echo and for the stage clock's plan bill.
int proj_launches_per_body(kernels::ProjPlan plan);

// Cross-checks the shapes §07 gives this file against the checkpoint the store actually holds, so a
// width typed in `state.h` cannot disagree with the file silently. Returns "" when all match, else a
// message naming the first mismatch. Host-side only; `decode_run` calls it before the first step.
std::string check_weights(const WeightStore& weights);

}  // namespace engine
