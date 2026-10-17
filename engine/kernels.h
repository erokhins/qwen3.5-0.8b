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
// Two launchers in this header are not graduated from a `prototype/<op>/` directory: `cast_*`
// (`kernels_cast.cpp`) and `embed_gather_rows` (`kernels_embed.cpp`). Both are shape the engine grew for
// its own reasons — a dtype seam, and a lookup that used to be one memcpy per token — and both carry no
// model math, which is what makes them safe to hold in a TU with no gate of its own: they move bytes.
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
// WHAT THIS IS. `progress/2026-10-13/prefill-route-spike-2026-10-13.md` §6 item 1 funds hipBLAS for ten shapes
// (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`,
// `up_proj`, `down_proj` — 150 of a wave's 186 matmul calls) and §"Wave-path attribution" of
// progress/2026-10-16/performance-status-2026-10-16.md:912 says why: matmul is 74.4 % of a 1024-token prefill and the hand
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
    // byte-identical — see `progress/2026-10-14/route-c-audit-2026-10-14.md` and `engine/results.md`, route (c).
    kLibrary,
    // The hipBLASLt call path (D26): `hipblasLtMatmul` with one cached algorithm per (shape, M)
    // instead of `hipblasGemmEx`, whose kernel choice the caller can neither see nor pin. Measured at
    // the shipping width M=512, the 150-call mix costs 18.67 ms as `kLibrary`, 17.42 ms through
    // hipBLASLt's heuristic rank-0 and 15.90 ms with a per-shape algo table — 1.072x from the
    // call-path change alone, 1.174x with the table (`prototype/gemm_algo_sweep/results.md` §3a).
    // Three shapes come back `ratio = 1.000` — hipBLASLt's rank-0 is already its own fastest
    // enumerated kernel — while the engine's `hipblasGemmEx` call is still 1.22-1.63x slower than that
    // same kernel, so the gap is hipBLAS-vs-hipBLASLt SELECTION, not kernel quality.
    //
    // NOT the default, and not by oversight: the operator decides when it flips, and it is a runtime
    // knob rather than a build define so both paths live in one binary and can be A/Bed in one process
    // (the `--attn-touch` precedent). The numerics contract is `kLibrary`'s — bf16 in, f32 accumulate,
    // bf16 `C`, one pinned 128 MiB workspace, the default stream — with a different accumulation
    // ORDER, which is a band question and not a byte-identity one: `engine/gate.py` judges `lt` by the
    // same cascade floor it arms for `library`.
    kLibraryLt,
    // Still first-class, and deleted by nothing: with this set every body takes the hand kernel
    // whatever its width, which is how the old path stays measurable in the same window (§7 W9 is an
    // A/B), how "decode untouched" is checkable end to end, and how a pre-route binary's bits can
    // still be reproduced. `engine/gate.py` judges it by the pre-route bands: its cascade floor is
    // armed for a library route and for nothing else.
    kHand,
};

// "library" / "legacy" / "lt" / "hand". Throws on anything else: this flag selects which side of an
// A/B a run is. `legacy` is accepted as a synonym for `library` because the pair being A/Bed is
// legacy-hipBLAS against lt-hipBLASLt and one shell loop should be able to spell both sides;
// `wave_gemm_name` never prints it — the echoed route stays `library`, which is what `gate.py`'s
// route check compares against the log.
WaveGemm parse_wave_gemm(const std::string& name);
const char* wave_gemm_name(WaveGemm route);

// Does a body of `rows` rows submit to a library GEMM? The single spelling of the question
// `submit_wave_body` asks, so that a third route cannot be added here and forgotten there: it is
// `resolve_wave_gemm`'s row-count rule (`rows <= 1` is the hand GEMV whatever the run asked for)
// tested against "is this a library route", which is now two of the four names, not one.
bool wave_gemm_is_library(WaveGemm want, int rows);

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
//
// `want` chooses WHICH library is built: `kLibrary` creates the hipBLAS handle, `kLibraryLt` the
// hipBLASLt one, and `kHand` (or any non-library route) creates neither — a `--gemm hand` run still
// allocates nothing the pre-route binary did not. The two library paths are never both live: one
// process runs one route, which is what makes "which GEMM produced these logits" a property of the
// run and not of the call site. The default argument is `kLibrary` so the pre-lt callers
// (`tools/route_c_stages.cpp`, and every log taken before this flag had a third value) still build
// exactly what they built before.
std::string gemm_route_init(WaveGemm want = WaveGemm::kLibrary);
// Tears down what `gemm_route_init` created. Safe to call when it never ran.
void gemm_route_shutdown();
// Runtime filter over the per-shape algo pin table: "all" (the default), "none", or a comma-separated
// list of pinned shape names. A pin choice is a measurement variable, so it is a knob and not a
// rebuild: pin-vs-rank-0 must be one window with one binary. Throws on an unknown name.
void gemm_pins_select(const std::string& spec);
// Runtime width of the per-(shape,M) heuristic query — how many candidates the library is asked for. The
// engine still takes the first admissible record (rank-0 in every measured field), so this knob changes
// the POOL and not the default pick; it exists so an in-situ A/B can ask for the deep pool and the
// shallow one with one binary, which a rebuilt default could not do (the deep list is not a prefix of
// the 8-wide one: same rank-0, different algorithms from rank ~5 down). Throws outside 1..64, the
// deepest pool measured so far. Echoed as `heur_width=` on the `GEMM` line.
void gemm_heuristic_width_select(int requested);

// PLAN-TIME algo selection for the hipBLASLt route, one call per (shape, M) instead of one per matmul.
// The engine issues 150 library calls per 24-layer wave and a heuristic query costs more than the win
// it buys, so the algorithm is fetched once per (shape, M) and cached; this is where that fetch runs,
// before the first body, so a wave's measured milliseconds contain no library bookkeeping. `rows` is
// the scratch's row width — the M every full body issues — and the seven distinct (shape, M) keys are
// the same census `prototype/gemm_algo_sweep --list` prints. Returns one line per shape built, empty
// if the lt route is not the live one (so a `library`/`hand` run prints nothing new). A body of a
// width nobody prewarmed — a prompt whose last wave is a remainder — still works: `gemm_bf16` builds
// and caches that key on first use, and says so in the same line format.
std::string gemm_route_plan(int rows);

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
// (progress/2026-10-16/performance-status-2026-10-16.md:811 — the frozen ledger's "Q10" — and `engine/results.md` §Q10b): `contig` recovers 0.945 ms of
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
// Which SPELLING of the sequential recurrence's state walks a `gdn_rec` launch runs
// (`decode_run --gdn-spell`). Both kernels are compiled into one binary so the A/B is one window
// (`AGENTS.md`: a ruler re-taken in a different window is not an A/B), and the two are claimed
// BIT-IDENTICAL rather than band-adjacent, which is what makes a knob here safe at all.
//
//   kShip      `prototype/gdn_decode/gdn_decode_kernels.h::rec_kernel`. THE DEFAULT and the shipped
//              form: every decode and prefill pin is pinned through it, and the in-situ A/B that D32
//              demands has not been read yet, so the default does not move on a burst number.
//   kUnroll4   `engine/gdn_rec_unroll4_kernels.h::rec_unroll4_kernel` — same operations, same order,
//              same values, `#pragma unroll 4` on both state walks. Worth 1.149–1.158x at launch
//              level on the harness (`prototype/gdn_rec_isa/results.md` §10, D32) at −7.7 % slots,
//              which is ~2x what D25's measured elasticity predicts, i.e. the prize is the loop
//              schedule and not the instruction bill. It is inert under `--gdn-scan collapse`, which
//              launches a different kernel; `parse_args` refuses that combination rather than letting
//              a timing run price the shipped form twice and call it the re-spelled one.
enum class GdnRecSpell { kShip, kUnroll4 };

// Parse-or-throw, like `parse_gdn_scan`: this names which side of an A/B a run is, and a typo that
// silently became the default would price the shipped spelling twice and report it as the other one.
GdnRecSpell parse_gdn_rec_spell(const std::string& name);
const char* gdn_rec_spell_name(GdnRecSpell s);

// One GDN layer: 16 blocks, thread j owns column j of S. `S` is the layer's [16][128][128] block.
// Per TOKEN, and stays that way for the sequential form: the recurrence is a sequential scan
// (§04.4), so unlike the three launchers above it has no token width. `spell` selects which of the two
// compiled spellings is submitted; the grid, the thread count and every address are the same either
// way, so the flag cannot move a byte of the access pattern (`kernels_gdn.cpp`).
void gdn_rec(const float* q, const float* k, const float* v, const float* g, const float* beta,
             float* S, float* o, GdnRecSpell spell = GdnRecSpell::kShip);

// One GDN layer, one BODY, route A: `kNumValueHeads * (d_v / 32)` blocks = 64, and the state stays
// in LDS for the whole body. `u`/`g`/`beta`/`o` are the same tiles `gdn_rec` is fed one row at a
// time, so the two forms read and write identical addresses; `rows` is the body's row count.
void gdn_rec_collapse(const float* u, const float* g, const float* beta, float* S, float* o,
                      int rows);

// ---- P9: which §04.4 scan a body runs ----------------------------------------------------------
// Two forms of the same recurrence, from one binary, so a before/after attribution is one window.
//
//   kSeq       §04.4's recurrence, one `gdn_rec` launch per token per layer. THE DEFAULT and the
//              shipped form: every decode pin is pinned through it, and `prototype/gdn_decode` gates
//              it.
//   kCollapse  Route A (D-12): the SAME recurrence, resident. One launch per layer per BODY, where
//              kSeq issues one per layer per TOKEN. A block owns 32 of a head's 128 `d_v` columns and
//              holds them in LDS across the whole body, so the state is read once per body instead of
//              twice per token. It is bit-identical to kSeq by construction, not by measurement: the
//              per-step expressions, the ascending `i` order, and the norm reduction (`block_sum` at
//              the same `blockDim.x`, fed the same element-per-thread mapping) are the shipped
//              kernel's, verbatim — only the residency and the launch count move. Why the association
//              cannot move is §04.4's cancellation measurement (one in 286 `o` elements lose 84 % of
//              their magnitude), and `progress/2026-10-14/p9-phase1c-2026-10-14.md` D-12 priced the chunked
//              alternative at 54.03 GFLOP against this route's 33.94.
//   kChunk     §04.7's chunked scan (`prototype/gdn_chunk/gdn_chunk_kernels.h`). KILLED by D-12 and
//              now refuses to be selected: its carry formed the chunk-boundary decay as a quotient of
//              two exps (§1c.1 D6), which is a different expression from the contract's one exp of a
//              difference and returned NaN at `S = 0`, and a wave that fails the wave's own §04.4
//              budget on a bug cannot carry an error budget. The kernel and `phase1.py` stay in the
//              tree as the record of why; `parse_gdn_scan` refuses the name.
//
// `resolve_gdn_scan` is the one place the row-count rule lives, and it is deliberately shaped like
// `resolve_wave_gemm`: a ONE-ROW body is kSeq whatever was asked for. That is not a performance
// opinion. Chunking a scan of one token buys nothing (one chunk, no inter-chunk work) and it would
// move every decode bit through the chunked form's rounding, which is the one thing §04.7's budget
// was never allowed to do — phase1's `onerow` witness compares the two forms on a single state
// update precisely so that this rule is checkable rather than assumed. For the same reason a
// one-row body is never routed through kCollapse: it is bit-identical to kSeq, so the rule costs
// nothing there, and a resident kernel launched for one token would stage 16 KiB of LDS to do one
// state update.
enum class GdnScan { kSeq, kCollapse, kChunk };

// The chunk size, in tokens, that the kernel is instantiated at. One value, compile-time:
// `prototype/gdn_chunk/phase1.py` prices C=64 inside both budget lines and REJECTS C=16 outright
// (per-row 1-cos 3.388e-4 against the 3.2e-4 ceiling), and a second instantiation of a form that is
// already 0.3 % inside a budget line is not something to add without being asked for.
inline constexpr int kGdnChunkTokens = 64;

// Parse-or-throw, like `parse_wave_gemm`: this names which side of an A/B a run is, and a typo that
// silently became the default would price the default twice and report it as the chunked one.
GdnScan parse_gdn_scan(const std::string& name);
const char* gdn_scan_name(GdnScan s);

// The row-count rule, in one place. `rows` is the BODY's row count, known only where the body is
// submitted, which is why this is a function and not a parsed value.
GdnScan resolve_gdn_scan(GdnScan want, int rows);

// Why a `--gdn-chunk` value is not accepted, or null for "accepted". The reason has to be printed
// rather than implied: a bare "unsupported" would not tell a reader that C=16 was measured and
// rejected on the numerics, which is the whole argument for why the default is 64.
const char* gdn_chunk_refusal(int tokens);

// One GDN layer, one body, chunked form: TWO launches (`prep` then `carry`). Everything except `S`
// is body-scoped scratch (`decode_step.cpp:allocate_scratch` allocates the six of them and
// `free_scratch` releases them), reused per layer
// in stream order like `g_u`/`g_o`.
struct GdnChunkLayerArgs {
    const float* u = nullptr;      // [n][6144] post-conv q|k|v, the tile `gdn_rec` reads row by row
    const float* g = nullptr;      // [n][16]
    const float* beta = nullptr;   // [n][16]
    float* S = nullptr;            // [16][128][128] F32 state — the only thing that survives
    float* o = nullptr;            // [n][2048] F32, §04.5's input
    float* nv = nullptr;           // staging: F32 because it is a result, never a rounded operand
    uint16_t* kcd = nullptr;
    uint16_t* qg = nullptr;
    uint16_t* kh = nullptr;
    uint16_t* intra = nullptr;
    float* cdec = nullptr;
    int n = 0;                     // rows in the body; >= 2, because a 1-row body is kSeq
};
void gdn_chunk_prefill(const GdnChunkLayerArgs& a);

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
    // Which wave core runs (`--attn-tile`, Phase 2: §9 of progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md).
    // 1 is `attn_tile_wave`, the G x M tiled form: one K row load feeds G=4 GQA heads x M=4 query rows
    // instead of one (row, head) pair, which is where the wave's 25.79 GB of K/V re-reads come from.
    // 0 is the shipped per-(row,head) `attn_core`. It is a WAVE-only knob: a one-row body is the fused
    // single-row form whatever this says, which is what keeps a decode step untouched by it and lets one
    // binary price both sides of the A/B. The default is 0 — the tiled form has been gated on CPU only
    // (`prototype/attn_tile_model.py`: bit-identical to the shipped form on every pinned wave row) and a
    // form that has not met the device does not become the engine's default by having been written.
    // It also refuses, rather than silently degrading: `score_f32`, a fused plan, and a non-32 warp are
    // each a refusal in `attn_tile_ok`, and the launcher turns a refusal into an error, not a fallback —
    // a fallback would price the shipped form twice and report it as the tiled one.
    int wave_tile = 0;
    // Which SPELLING of the tiled wave's per-touch inner work runs (`--attn-touch`). Only consulted when
    // `wave_tile` picked the tiled form, and only for a wave: a one-row body is `attn_core` whatever this
    // says, so it carries `--attn-tile`'s property — a decode bit cannot move under it — for the same
    // reason and enforced in the same place, the launcher. 1 is `engine/attn_slim_kernels.h`'s
    // `attn_slim_wave`: the score dot's trip count made compile-time and the warp reduce folded by hand,
    // 45 issue slots per (row, lane, head) touch instead of 116, measured 1.52-1.57x on the wave and
    // bit-identical on the prototype harness (`prototype/attn_valu_slim/results.md` §7). 0 is the
    // prototype's `attn_tile_wave`, kept as the frozen side the A/B and the census are made against.
    // The DEFAULT IS 1 because the levers were gated before they were ported, unlike the tiled form
    // itself when `wave_tile`'s default was set; `engine/gate.py` asserts the run's own NUMERICS echo
    // equals the flag it was given, so a binary cannot quietly be judged by the other side.
    int wave_touch = 1;
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

// ---- §02.3: the embedding lookup for a whole wave in ONE launch (D30 4.2′).
// `out[r] = table[ids[r]]` for `rows` rows, BF16, `kHidden` wide — byte-identical to `rows` copies of
// one table row, because there is no arithmetic in it (`kernels_embed.cpp` says why the engine grew a
// kernel for a lookup: 4096 of a wave's copy operations were 2 KiB memcpys and 84 % of each one was
// the submission). `ids` is a DEVICE array of `rows` token ids in [0, vocab); the caller owns it and has
// already range-checked the values, because a bad id here is an out-of-bounds read of the tied table
// rather than a throw.
void embed_gather_rows(const uint16_t* table, const int* ids, uint16_t* out, int rows);

}  // namespace engine::kernels
