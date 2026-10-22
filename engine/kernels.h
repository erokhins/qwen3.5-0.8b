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

// The same GEMV with the activation staged as BF16 and widened in registers (`route S1 item 1`).
// `x` is the BF16 tile the producer wrote — the same bytes the caller would otherwise widen into
// `norm_f32`/`block_out_f32` with a launch of `kernels_cast.cpp`. Bit-identical to `proj_gemv` by the
// exactness of `hip_bfloat16 -> float`, and it is `proj_gemv_tokens<IN, 1, PACK_X=true>`, i.e. the
// prototype's own gated packed staging at the one-row tile width, so nothing new is transcribed here.
// ONE ROW ONLY (`tokens` is fixed at 1 by the launcher): a wider body is a prefill wave and keeps the
// widening, which is why `decode_step.cpp` gates the fold on `n == 1`. The kernel cost of this trade
// has a measured precedent against it in `engine/results.md`'s cast row (`PACK_X` cost +0.278 ms over
// 48 launches to save 48 casts, quoted); the wall A/B decides whether it survives on this shape.
void proj_gemv_xbf16(const void* weight, const uint16_t* x, uint16_t* out, int rows, int in);

// The same projection through hipBLAS (route (c)). `x` is the SAME tile `proj_gemv`'s `x` was widened
// from, still in the BF16 glue wrote — the library takes a 16-bit `A` operand, so it reads what the
// producer wrote instead of a widening of it. `tokens` is the body's row count (`m`), and as in
// `proj_gemv` it is what `engine/launch_stats.h` counts this site as: one library call per projection
// per body, the same one-site-per-projection bill §06.2's "one GEMM per projection" already prices.
void proj_gemm(const void* weight, const void* x, uint16_t* out, int rows, int in, int tokens);

// ---- the decode one-row projection route (route D1, `prototype/proj_gemv/results.md` §D1-W4).
//
// Two independent levers, both measured in one clock-controlled prototype window against THIS shipped
// kernel, both bit-identical to it, and both confined to a body of ONE row (a wider body is a prefill
// wave and takes `proj_gemv`'s token-tiled form or the library, untouched):
//
//   SHAPE    the word loop and block geometry of the winning arm: compile-time trip count with
//            `#pragma unroll 4`, rows/block 8 at 1024 threads. −24.5 % of the family.
//   PLAN     one launch per layer group instead of one per tensor: 48 launches per token instead of
//            the 114 §07 bills, at the winning shape. −11.7 % more, at identical geometry.
//
// Both shipped behind flags that defaulted to HEAD's behaviour, because the prize was a prototype burst
// number and the in-situ A/B (the `proj` family row of a real decode step) is what had to decide whether
// the defaults move — the same order every other lever in this file was shipped in. It did move them:
// the in-situ `proj` family is 4.2015 → 2.9521 → 2.7316 ms/step and the whole step 11.6967 → 10.1851 ms
// (`engine/results.md`, route D1), so `kProjPlanDefault`/`kProjShapeDefault` below name the pair that
// ships, and `kSplit`/`kShipped` are the pre-D1 pair — still reachable, because a pre-D1 log has to stay
// replayable bit for bit.

// How the per-token projection launches are GROUPED. A group is one launch over several weight tensors.
enum class ProjPlan {
    // §07's bill: one launch per projection — 5 per GDN layer (qkv, z, b, a, out_proj) and 4 per
    // attention layer (q, k, v, o_proj), 114 per token. What every gate log taken before route D1 was
    // measured on, and what `--proj-plan split` still replays byte-identically; no longer the default.
    kSplit,
    // One launch for a layer's INPUT projections (they read the same row of `norm_f32` and each writes
    // its own row-family destination, so they need no weight copy — see `ProjSeg`) plus one for its
    // output projection: 2 per layer, 48 per token. THE SHIPPED PLAN (`kProjPlanDefault`).
    kPacked,
};
const char* proj_plan_name(ProjPlan plan);
// "split" / "packed". Throws on anything else: this names which side of an A/B a run is, and a typo that
// silently became the default would price the default twice and report the second price as the plan's win.
ProjPlan parse_proj_plan(const std::string& name);

// Which kernel and block geometry a one-row projection launch uses.
enum class ProjShape {
    // `proj_gemv_tokens<IN, 1, false>` at rows/block 32, i.e. HEAD's machine code, unchanged.
    kShipped,
    // `engine/proj_group_kernels.h`'s grouped kernel, even when it carries ONE segment: unrolled word
    // loop, rows/block 8 at 1024 threads. Bit-identical to `kShipped` by the same argument the
    // prototype's gate accepted for its `v1-unroll` (same weight words, same order, same ladder, one
    // rounding), which is why this lever owes a byte-compare and not a band.
    kUnroll8,
};
const char* proj_shape_name(ProjShape shape);
ProjShape parse_proj_shape(const std::string& name);

// The shipped pair, named ONCE, because three C++ readers have to agree on it and none of them may type a
// launch count: `decode_run`'s option defaults, `engine/decode_step.cpp`'s `planned_launch_counts()` (the
// `LAUNCH` bill `engine/gate.py`'s section (d) polices — which is why that bill is 48 and not 114), and
// `engine/stage_clock.cpp`'s one-row-body self-checks, which have to build a one-row body the way the
// shipped plan builds one. Beware the near-name: `ProjShape::kShipped` is the enum word for the PRE-D1
// shape and is NOT this default; the flag word `shipped` still selects it, and a log that reads
// `shape=shipped` is a parity run rather than the engine.
constexpr ProjPlan kProjPlanDefault = ProjPlan::kPacked;
constexpr ProjShape kProjShapeDefault = ProjShape::kUnroll8;

// One tensor inside a grouped launch. `out` addresses ROW 0 of a row-family tile, which is legal for
// exactly one reason: this route exists for a body of one row (`decode_step.cpp` gates it on `n == 1`),
// so a `[wave_rows, rows]` tile's first row is `rows` contiguous elements starting at `out` and no
// output stride has to be carried. `start` is filled by the launcher as a running sum, never by a caller.
struct ProjSeg {
    const void* weight;   // device, [rows, in] BF16 as stored (§01: y = x · Wᵀ)
    uint16_t* out;        // device, `rows` BF16
    int rows;
    int start;
};

// The kernel argument is passed by value, so it has a fixed size. Four is what a layer needs: the widest
// input group is a GDN layer's {qkv, z, b, a}; an attention layer's is {q, k, v}; an output projection is
// one segment. A fifth would be a plan the engine does not have, and the launcher refuses rather than
// truncating.
constexpr int kProjMaxSegs = 4;
struct ProjSegs {
    ProjSeg s[kProjMaxSegs];
    int n;
    int total;            // sum of rows: the row-index space of the launch
};

// ONE launch over `nsegs` tensors, all of width `in`, all reading the row `x` points at. Refuses (throws)
// on a segment list this route cannot express — empty, over `kProjMaxSegs`, a non-positive row count, a
// width the kernel is not instantiated for — because a silently skipped projection is a wrong token, and
// `proj_gemv`'s own silent no-op on an unknown width is paid for by `check_weights` running first; this
// entry point has no such blanket, since a caller assembles a list here rather than naming one tensor.
// Counts ONE `launchstats::proj()` submission, so `--proj-plan packed` is visible on the `LAUNCHSTAT`
// line and not only in the flag that caused it.
void proj_rows(const ProjSeg* segs, int nsegs, const float* x, int in);

// `proj_rows` with the shared `x` row staged as BF16 instead of taken as a widened F32 tile: the same
// segment list, the same grid, the same geometry, one fewer launch in front of it. Same bit-identity
// argument as `proj_gemv_xbf16`, same one-row-only restriction (`decode_step.cpp` gates it), and the
// staged row is half the LDS per block — a shape effect the A/B reads, not this declaration.
void proj_rows_xbf16(const ProjSeg* segs, int nsegs, const uint16_t* x, int in);

// Can this shape actually run on this device at width `in`? `kShipped` always can. `kUnroll8`'s geometry
// is stated in lanes-per-row terms and was measured on a wave32 device, so a wave64 target gets the
// shipped shape; the caller COUNTS such a refusal and prints it (`decode_step.cpp`, `PROJSHAPE-FALLBACK`),
// because a silently inert A/B side is the failure this file's flags exist to avoid.
bool proj_shape_available(ProjShape want, int in);

// The geometry `proj_rows` will launch with, for the echo that has to agree with the launch
// (`--dump-proj-plan`, `PROJPLAN`). One source for both, so a printed rows/block cannot drift from a
// launched one — the trap AGENTS.md records for a geometry fact that lives in two places.
int proj_shape_rows_per_block(ProjShape shape);
int proj_shape_threads(ProjShape shape);
// The device's warp width as the launcher queried it (the same query `kRowsPerBlock * warp` is built
// from). The echo prints it next to the geometry, so `unroll8` on a wave64 box reads as the fallback it
// is instead of as a shape that ran.
int proj_warp_width();

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

// The same two dots with the activation staged as BF16 (`engine/mlp_xbf16_kernels.h`), which is what
// deletes 48 of the 97 widenings: `x`/`act` are the BF16 tiles `glue` and the gate/up walk already
// wrote instead of F32 copies of them. Bit-identical for the reason that header gives; ONE ROW ONLY,
// so a prefill wave still runs the batched pair above with its widenings in front of it.
void mlp_gateup_xbf16(const uint16_t* w_gate, const uint16_t* w_up, const uint16_t* x,
                      uint16_t* act, int rows, int in, MlpGateMap map);
void mlp_down_xbf16(const uint16_t* w_down, const uint16_t* act, uint16_t* out, int rows, int in);

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

// ---- §04.2/§04.3/§04.4: conv, gates, recurrence (prototype/gdn_decode). The STATE side is all F32
// (window, `u`, `g`, `beta`, `S`, `o`); the gate OPERANDS are the BF16 tiles their projections wrote,
// which the conv and the gates widen in-register (D30 1.7').
// `tokens` is P6's token-parallel form (§06.2): `raw`/`u` are then [tokens, 6144] tiles and
// `a_raw`/`b_raw`/`g`/`beta` [tokens, 16] ones, with the conv reading the [3][6144] window the
// previous wave or decode step left and writing its own last-3-columns window once per wave
// (§08.6 inv 2). `tokens == 1` issues the same grid and the same per-element arithmetic this
// launcher issued before P6 — `prototype/gdn_decode/check.py --tokens` requires the two widths to
// agree bit for bit, and the window rule (raw[k<0] is the CARRIED window, not zero) is the part
// that a wider tile could get wrong while every single-wave fixture still passed.
void gdn_conv(const uint16_t* raw, const float* win, const float* conv_weight, float* u,
              float* new_win, int tokens = 1);
// Route S1's follow-on (`QWEN_COPY_FOLD=1`, one-row bodies only): the same conv with the window roll
// inside it. `win` is the state window and is BOTH operand — no scratch row, no D2D roll-back — which
// is why it needs its own entry point instead of `gdn_conv` called with aliased arguments:
// `conv_kernel`'s `win`/`new_win` are `__restrict__`, and one buffer through two such parameters is
// the case the qualifier forbids, not a warning to walk past. Body and geometry:
// `engine/gdn_conv_roll_kernels.h`; why the values are the same ones the copy used to move:
// `prototype/glue_copy_fold/GLUECOPY-HANDOVER.md` §2.
void gdn_conv_roll(const uint16_t* raw, float* win, const float* conv_weight, float* u);
void gdn_gates(const uint16_t* a_raw, const uint16_t* b_raw, const float* a_log,
               const float* dt_bias, float* g, float* beta, int tokens = 1);
// Which SPELLING of the sequential recurrence's state walks a `gdn_rec` launch runs
// (`decode_run --gdn-spell`). All the spellings are compiled into one binary so an A/B is one window
// (`AGENTS.md`: a ruler re-taken in a different window is not an A/B), and every pair among them is
// claimed BIT-IDENTICAL rather than band-adjacent, which is what makes a knob here safe at all.
//
//   kShip      `prototype/gdn_decode/gdn_decode_kernels.h::rec_kernel`. The pre-D33 shipped form, kept
//              as a first-class A/B side so the claim below stays testable from one binary, and
//              `gdn_rec()`'s optional-argument default. What a run actually submits comes from
//              `decode_run --gdn-spell`, whose default is `kFused` since D51.
//   kUnroll4   `engine/gdn_rec_unroll4_kernels.h::rec_unroll4_kernel` — same operations, same order,
//              same values, `#pragma unroll 4` on both state walks. THE SHIPPING DEFAULT from D33 to
//              D51:
//              worth 1.149–1.158x at launch level on the harness (`prototype/gdn_rec_isa/results.md`
//              §10, D32) at −7.7 % slots and 1.153x on the family row at 4096 lanes in the wave's own
//              interleaving, which is ~2x what D25's measured elasticity predicts, i.e. the prize is
//              the loop schedule and not the instruction bill.
//   kFused     `engine/gdn_rec_fused_kernels.h::rec_fused_kernel` — the unroll4 walks with the q/k norm
//              prologue FUSED into one pass over shared memory (two `red[]` arrays, 3 barriers instead
//              of the two-call form's 8). −32 of 1,406 slots/token-step at UNCHANGED registers (59
//              VGPRs, zero spill) and +128 B of LDS; bit-identical by claim, measured bit-identical in
//              4/4 harness cases, and re-proved per fixture by `engine/gate.py` section (j). **THE
//              DEFAULT since D51**: D50 shipped it as a knob and the paired in-situ A/B on the
//              `gdn_rec` family row read 1.0645 at 4096 lanes (3/3 reps, worst rep 1.0200), flat at
//              1024 lanes and flat on decode — `progress/2026-10-21/performance-status-2026-10-21.md` §20.
// All three are inert under `--gdn-scan collapse`, which launches a different kernel; `parse_args`
// refuses that combination rather than letting a timing run price one kernel twice and call the second
// price the spelling's win.
enum class GdnRecSpell { kShip, kUnroll4, kFused };

// Parse-or-throw, like `parse_gdn_scan`: this names which side of an A/B a run is, and a typo that
// silently became the default would price the shipped spelling twice and report it as the other one.
GdnRecSpell parse_gdn_rec_spell(const std::string& name);
const char* gdn_rec_spell_name(GdnRecSpell s);

// One GDN layer: 16 blocks, thread j owns column j of S. `S` is the layer's [16][128][128] block.
// Per TOKEN, and stays that way for the sequential form: the recurrence is a sequential scan
// (§04.4), so unlike the three launchers above it has no token width. `spell` selects which of the
// compiled spellings is submitted; the grid, the thread count and every address are the same either
// way, so the flag cannot move a byte of the access pattern (`kernels_gdn.cpp`).
void gdn_rec(const float* q, const float* k, const float* v, const float* g, const float* beta,
             float* S, float* o, GdnRecSpell spell = GdnRecSpell::kShip);

// ---- D53/D54: N token-steps of the same recurrence inside ONE launch (`decode_run --gdn-loop`) ----
//
// One GDN layer, one launch, `tokens` CONSECUTIVE token-steps. `rec_loop_kernel`
// (`engine/gdn_rec_loop_kernels.h`) is `rec_fused_kernel`'s body with the token index moved inside
// it, so a span of `n` replaces `n` per-token launches and nothing else: same grid `<<<NH, D>>>`,
// same thread map, same state buffer at the same addresses, same per-token expression trees in the
// same order. Admitted by BIT-identity, not by a band — spans 0/1/3/16 came back
// `ndiff_o_vs0=0 ndiff_state_vs0=0` on the harness (`progress/2026-10-21/performance-status-2026-10-21.md` §23) — and
// `engine/gate.py` section (j) re-proves the claim through this launch path per fixture.
//
// `u`/`g`/`beta`/`o` are the whole BODY tiles, the same ones `gdn_rec_collapse` is fed, and the
// kernel walks their row strides itself (`CH` for the q|k|v tile, `NH*D` for `o`; asserted against
// the engine's own names in `kernels_gdn.cpp`, so a wrong stride is a compile error rather than a
// silent numerics change). Deliberately no default for `spell`, unlike `gdn_rec()`: only one
// spelling of the looped body has been transcribed and admitted, and a side that got its spelling
// from a default is a side the log cannot prove. `gdn_rec_span` therefore REFUSES a non-`kFused`
// spelling for `tokens > 1` rather than silently running a different body under a flag that claims
// to name the looped one.
void gdn_rec_span(const float* u, const float* g, const float* beta, float* S, float* o,
                  int tokens, GdnRecSpell spell);

// How many token-steps one `gdn_rec` launch carries, resolved where the body's row count is known
// (`decode_run --gdn-loop` sets `want`). `1` means "the per-token loop, verbatim" and is the shipped
// default (D54: knob first); anything larger walks that many token-steps per launch, and a span
// wider than the body IS the body, so `--gdn-loop 512` is one launch per (layer, body) at every
// width including a prompt's ragged final body. A span below the body size is allowed and means
// what it says — ceil(rows/span) launches, one per span — because span-invariance is proved
// (`prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp --selftest-loop`, 39.4 M elements bit-compared across
// spans) and a smaller span is the named fallback if a 512-iteration launch behaves badly in situ.
//
// The one-row rule lives here, shaped like `resolve_gdn_scan`'s and for the identical reason: a
// ONE-ROW body takes the per-token path whatever the flag says, so a decode step is the shipped bits
// BY CONSTRUCTION and not by measurement. Every decode pin is pinned through the per-token launch,
// and no decode bit should have to depend on the loop's bit-identity holding.
int resolve_gdn_loop(int want, int rows);

// ---- bet G1: where the per-token k/q L2-norms are computed (`decode_run --gdn-norm`) -------------
//
// Two spellings of the SAME looped body, from one binary, so the A/B is one window:
//
//   kInline  `engine/gdn_rec_loop_kernels.h::rec_loop_kernel` — the shipped form. Each token-step
//            reduces its own k and q rows with `block_sum_pair_f32` inside the walk.
//   kHoist   `engine/gdn_rec_hoist_kernels.h` — a pre-pass computes those two rows' norms (and
//            `expf(g)`, `beta`) once per (token-step, head), and the walk reads the normalised rows and
//            the two scalars. The walk then has no shared memory, no barrier and no cross-lane reduction,
//            which is the point: three barriers no longer sit between token t and token t+1
//            (`prototype/gdn_walk/BET.md` §3).
//
// `kHoist` is claimed BIT-IDENTICAL to `kInline`, not band-adjacent, and the argument is that the
// pre-pass calls the same reduction with the same `blockDim.x` and the walk's arithmetic text is
// unchanged (`engine/gdn_rec_hoist_kernels.h`, `prototype/gdn_walk/check_hoist_transcription.py`).
// Like `--gdn-spell` and `--gdn-loop`, this is a KNOB and `kInline` is the default; the paired in-situ
// A/B on the `gdn_rec` family row decides whether the default moves.
//
// It reaches the `--gdn-loop` span branch only. A one-row body takes the per-token loop whatever this
// says (`resolve_gdn_loop`), and `--gdn-scan collapse` submits a different kernel, so both are inert
// for it — `parse_args` refuses the collapse pair rather than echoing a form that did not run.
enum class GdnNorm { kInline, kHoist };

// Parse-or-throw, for the standing reason: this names which side of an A/B the run is.
GdnNorm parse_gdn_norm(const std::string& name);
const char* gdn_norm_name(GdnNorm n);

// The three stage tiles `gdn_norm_hoist_kernel` writes and `rec_loop_normed_kernel` reads, positioned
// at one body's first row: `kn`/`qn` are [rows][NH*D] F32 normalised rows and `gsc` is
// [rows][2*NH] F32 (`gam` then `beta`). They are scratch, not state — nothing survives the body.
struct GdnNormStage {
    float* kn = nullptr;
    float* qn = nullptr;
    float* gsc = nullptr;
};

// The hoisted pair: TWO launches for what `gdn_rec_span` does in one — the norm pre-pass over the
// body's `tokens` rows, then the walk. Same tiles `gdn_rec_span` is fed, same state buffer, same
// `otile`, and the two launches are adjacent in the stream so the stage tiles are read while they are
// as fresh as they can be. Refuses `tokens < 2` for `gdn_rec_span`'s reason: a one-row body is
// `resolve_gdn_loop`'s to route to the per-token kernels.
void gdn_rec_span_hoisted(const float* u, const float* g, const float* beta, float* S, float* o,
                          const GdnNormStage& stage, int tokens, GdnRecSpell spell);

// ---- bet G2: the geometry of the looped walk (`decode_run --gdn-geom`) ----------------------------
//
// One launch per SPAN in every case, which is what keeps the `gdn_rec` plan row identical across every
// side; what differs is the map.
//
//   kBlock  `<<<NH=16, D=128>>>`, one state column per thread, the two row norms reduced inside the
//           walk by `block_sum_pair_f32` — the SHIPPED form and the default.
//   kWf4    `<<<16, 32>>>`: ONE wavefront per block, 4 state columns per lane, the row norms reduced
//           by warp butterfly from the rows the lane already loads. `LDS = 0`, zero `__syncthreads()`
//           (`engine/gdn_rec_wf_kernels.h`; the shape §38's trace shows llama.cpp running).
//   kWf2    the same kernel at `<<<32, 32>>>`, 2 columns per lane — the split leg, which prices the
//           wavefronts-for-work-per-wavefront trade named in that header.
//   kReg16  `<<<dim3(NH, D/16)=16x8, dim3(32,4)>>>` = 128 blocks x 128 threads = 512 wavefronts = 6.40
//           wf/SIMD against the shipped 0.80: llama's map (`gdn.cu:203-204`) under our recurrence, 8
//           lanes per state column and 16 state elements per thread HELD IN REGISTERS for the whole span
//           (`engine/gdn_rec_reg16_kernels.h`; bet G4, `prototype/gdn_walk/BET.md` §12). Like `wf4` it is
//           NOT bit-identical and cannot be — the row norms and both state dots re-associate — and like
//           every knob on this recurrence it reaches the `--gdn-loop` span branch only.
//
// NOT claimed bit-identical, and it cannot be: the state dot and the `o` dot re-associate (the header
// names the three things that move). BET.md §10 constraint 2 fixes the admission as the EXISTING §10.1
// aggregates — cos >= 0.9995 and meanrel <= 5 % through `engine/gate.py` on the pinned fixtures, no new
// band and no `--ack-spill` — and `prototype/gdn_walk/g2_mirror.py` is the CPU-side size-of-the-move.
// Like every knob on this recurrence it reaches the `--gdn-loop` span branch only: a one-row body takes
// the per-token kernels whatever this says (`resolve_gdn_loop`), so no decode bit can move under it, and
// `parse_args` refuses the combinations where it would be inert (`--gdn-loop 1`, `--gdn-scan collapse`,
// and `--gdn-norm hoist`, whose pre-pass has nothing left to hoist here).
enum class GdnGeom { kBlock, kWf4, kWf2, kReg16 };

// Parse-or-throw, for the standing reason: this names which side of an A/B the run is.
GdnGeom parse_gdn_geom(const std::string& name);
const char* gdn_geom_name(GdnGeom g);

// The barrier-free geometry: ONE launch for what `gdn_rec_span` does in one, over the same tiles at the
// same addresses (`u` is the conv's [n][6144] tile, its q|k|v slices at +0/+kValueDim/+2*kValueDim), the
// same state buffer and the same `otile`. No stage tiles, because the norms are computed in registers.
// Refuses `tokens < 2` and any `spell` without a looped body, for `gdn_rec_span`'s two reasons.
void gdn_rec_span_wf(const float* u, const float* g, const float* beta, float* S, float* o,
                     int tokens, GdnRecSpell spell, GdnGeom geom);

// ---- bet G4: llama's thread map under the same recurrence (`decode_run --gdn-geom reg16`) ----------
//
// ONE launch per SPAN, on the same `gdn_rec` plan row with the same `launches=144` at the A/B shape,
// because a win that moved its milliseconds to another row would not be a win (§40). What differs is
// the map: `<<<dim3(NH, D/16), dim3(32, 4)>>>` = 128 blocks x 128 threads = 512 wavefronts, 8 lanes per
// state column, and **16 state elements per thread held in registers for the whole span** — one state
// load and one state store per (block, span) instead of three passes over 1 MiB per token.
// `engine/gdn_rec_reg16_kernels.h` owns the map, its asserts and its numerics declaration; `kWf4`/`kWf2`
// above are G2's dead one-wavefront form and keep their own launcher so a re-read of G2 stays routable.
void gdn_rec_span_reg16(const float* u, const float* g, const float* beta, float* S, float* o,
                        int tokens, GdnRecSpell spell);

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

// Which score roll the WMMA wave runs (`decode_run --attn-fattn-roll`, bet A4-L1). Two spellings of one
// loop, two kernels, one binary: `kSingle` is the shipped roll (one accumulator, a 16-long `v_wmma`
// recurrence) and `kTwo` is the pair of 8-long recurrences whose note block reads 200 VGPR / 0 spill /
// 0 scratch against the shipped 178 / 0 / 0 — `prototype/attn_percost/A4-L1-gates.md` §2 is that reading,
// and §33's spilled-at-8-lost-to-spill-free-at-6 precedent is why spill is the admission line.
enum class FattnRoll { kSingle, kTwo };

// Where the WMMA wave keeps Q, and how deep its output transpose stages (`decode_run --attn-fattn-qstage`,
// bet A6-G1). Two spellings of one staging plan, two kernels, one binary — and the two knobs above are
// orthogonal to these, so the TU carries all four combinations (`engine/attn_fattn_kernels.h`).
// `kStaged` is SHIPPED: Q in a per-warp 8 KiB stage, the F32 transpose riding that dead stage 16 rows at a
// time, two passes; declared LDS 32,768 B, which on this part's 64 KiB is 2 blocks/CU = 6 wavefronts/CU.
// `kUnstaged8` reads Q from global at the use site and gives the whole per-warp buffer to the transpose at
// FATTN_TRG = 8 rows a pass (4 passes, 4 KiB/warp): declared LDS 20,480 B = 3 blocks/CU = 9 wf/CU at 185 VGPR
// and spill 0 (`prototype/attn_percost/A6-result.md` §1, §3). Unlike the roll knob this one IS owed byte
// identity with the default — same operand values, same order, only the staging addresses and the operand's
// provenance move — and `prototype/attn_percost/A6-G1-impl.md` is where that is proved on device.
enum class FattnQStage { kStaged, kUnstaged8 };

// `AttnLayerArgs::split_cap`'s OFF value, so the default is a named state and not a magic zero.
// It is deliberately NOT spelled as a copy of `kAttnMaxSplit` (`engine/kernels_attn.cpp`, the lane-ceiling
// constant): "uncapped" means the split is the lane geometry's own `split_for(lanes_last)` and nothing
// else, so a cap of 0 cannot drift out of step with a future ceiling raise, and `tools/attn_ceiling_proof.py`
// [A] — which accounts for every CODE use of `kAttnMaxSplit` in `engine/` — keeps its existing census.
// Any value >= 2 is a real cap; a value above the ceiling is inert because `min` cannot bind there.
constexpr int kAttnSplitCapOff = 0;

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
    // Which attention FORM owns the STREAMING rows of a wave body (`--attn-fattn`). 0, the default, is
    // `attn_slim_wave` for every row, which is the shipped path and stays untouched. 1 hands the rows the
    // WMMA form covers — exactly the rows §07.3 already streams, `blocks_for_row > 1` — to
    // `engine/attn_fattn_kernels.h`, and leaves every one-pass row on `attn_slim_wave`. That coverage rule
    // is a decision, not a side effect, and it is written out at the top of that file: a row the one-pass
    // form serves is pinned bit-for-bit against the dumps, and a WMMA round is streaming by construction,
    // so routing a pinned row through it would move bits no band has ever been priced for. Wave-only,
    // like the two above: a one-row body is `attn_core` whatever this says, so no decode bit moves under
    // it. DEFAULT 0 because the form has not been admitted on device yet (`prototype/attn_wmma/`);
    // `engine/gate.py` asserts this echo equals the flag it was given, so a binary cannot be judged by the
    // other side.
    int wave_fattn = 0;

    // Which SPELLING of the WMMA form's score roll runs (`--attn-fattn-roll`, bet A4-L1). Only consulted
    // when `wave_fattn` picked the WMMA form for at least one row, and wave-only like every knob above it:
    // a one-row body is `attn_core`, so no decode bit moves under it. `kSingle` is the shipped roll — one
    // accumulator, 16 dependent `v_wmma` trips, the code path this header carried before the knob existed —
    // and `kTwo` gives the roll a second accumulator and fetches both tiles of a pair before either matrix
    // op issues, so two chains of 8 progress at once (`engine/attn_fattn_kernels.h`,
    // `prototype/attn_percost/A4-BET.md`). NOT bit-identical and cannot be: two accumulators re-associate
    // the fixed ascending F32 dot, so `o` moves, and A4-BET fixes the existing §10.1 cos/meanrel bands as
    // the judge in advance rather than discovering the band afterwards. Both kernels are in the binary
    // whatever this says, which is what makes the paired timing one window instead of two builds.
    FattnRoll fattn_roll = FattnRoll::kSingle;

    // Where the WMMA wave keeps Q, and how deep its output transpose stages (`--attn-fattn-qstage`, bet A6-G1).
    // Wave-only like every knob above it — a one-row body is `attn_core`, so no decode bit moves under it — and
    // consulted only when `wave_fattn` picked the WMMA form for at least one row. `kStaged` is the shipped
    // spelling and the code path this header carried before the knob existed, so the default side of its A/B is
    // HEAD's own machine code and "the default is inert" is a byte-compare rather than an argument.
    // `kUnstaged8` reads Q from global at the use site and stages the epilogue FATTN_TRG rows a pass, which
    // takes the block's declared LDS from 32,768 B to 20,480 B — 3 blocks/CU instead of 2, 9 wf/CU instead of
    // 6 (`engine/attn_fattn_kernels.h` placement (E), `prototype/attn_percost/A5-GEOMETRY.md` §3.2). It is owed
    // byte identity, not a band: a row's round boundaries do not move (`blo = blockIdx.y · span` is per row and
    // geometry-independent), and the operand conversion is the same `bf16 -> f32 -> f16 RN`. Both kernels are in
    // the binary whatever this says, which is what makes the paired timing one window instead of two builds.
    FattnQStage fattn_qstage = FattnQStage::kStaged;

    // What width the WMMA wave's P*V accumulator is (`--attn-fattn-pvf16`, bet A10-F16). Wave-only like every
    // knob above it — a one-row body is `attn_core`, so no decode bit moves under it — and consulted only when
    // `wave_fattn` picked the WMMA form for at least one row. `0` is the shipped F32 accumulator, 16 tiles of 8
    // F32 = 128 registers a lane, and the code path this header carried before the knob existed, so the default
    // side of its A/B is HEAD's own machine code and "the default is inert" is a machine-code compare. `1` is
    // 8 tiles of 16 F16 halves = 64 registers a lane: the same 256 output dims, half the registers, filled by
    // two `v_wmma_f16_16x16x16_f16` issues per 32 dims (`op_sel` 0 then 1, two A fragments, the same P — llama's
    // shape, `mma.cuh:1553-1566`) instead of one `v_wmma_f32_16x16x16_f16` per 16.
    //
    // IT IS AN ENABLER, NOT A SPEED. 16 `v_wmma` issues per round either way (`prototype/attn_percost/A10-F16-IMPL.md`
    // D1), the two atoms time the same per unit of coverage (`A10-RF-LADDER.md` §7), and the expected throughput
    // effect is ~0; what it buys is the 64 registers a later Q-in-registers spelling has to fit through (D7:
    // 128 acc + 128 Q is impossible, 64 + 128 fits). And unlike `fattn_qstage` above it it is NOT owed byte
    // identity — an F16 accumulator is a different format — so its judge is the existing §10.1 cos/meanrel bands
    // through `A4-BET.md`, no new band and no widening. Both spellings are in the binary whatever this says,
    // which is what makes the paired timing one window instead of two builds.
    int fattn_pvf16 = 0;

    // Whether the wave's WMMA core is the A12 WIDE block instead of the 16-row one (`--attn-fattn-wide`, bet
    // A12). Wave-only like every knob above it — a one-row body is `attn_core`, so no decode bit moves under
    // it — and, unlike the three above, it does not re-spell the form that ships: at 1 it REPLACES this body's
    // whole wave core. `engine/attn_wide_kernels.h`'s block covers 32 query positions x the 4 heads of one KV
    // group (128 row-slots) per block, stages 64 KV positions a round, holds Q in F16 registers, and
    // accumulates P*V in F16; it has NO lane split and NO coverage rule, so it owns EVERY row of the body —
    // the rows at or below MAXSC included, which is where the shipping route is bit-exact against the dumps.
    // `attn_slim_wave`, the eight `attn_fattn_wave*` spellings and `attn_combine` are then not launched at all;
    // the whole argument is at the routing site (`engine/kernels_attn.cpp`) and in that header's file comment.
    //
    // NOT bit-identical to either spelling, at any row width, and not owed byte identity: a 64-position round
    // re-associates the softmax and the weight that reaches the accumulator is an un-normalised F16 `exp`
    // divided at the end, where the one-pass route rounds the normalised weight to BF16 before the P*V
    // (§05.5's rule 4). So the judge is `engine/gate.py`'s existing §10.1 cos/meanrel aggregates — no new
    // band, no widening, no loosened check — exactly as `--attn-fattn-roll two` and `--attn-fattn-pvf16` were
    // judged through `prototype/attn_percost/A4-BET.md`.
    //
    // DEFAULT 0, which is the whole point of the knob as it stands: the form has cleared a CPU mirror
    // (`prototype/attn_wide/a12_wide_mirror.py`) and a standalone timing rig, but the engine route has not been
    // admitted on device, and a form that has not been admitted does not become the engine's default by having
    // been written. At 0 nothing here is consulted and the shipped machine code is untouched — the wide kernel
    // is a second code object in the binary, not a branch inside the shipped one.
    //
    // AT 2 it is the SAME block with llama.cpp's round structure ported into it (bet A13,
    // `engine/attn_wide_llama_kernels.h`, `prototype/attn_wide/LLAMA-PORT-PLAN.md` items 1, 2, 4 and 5): a
    // template-stride XOR swizzle inside each staged row, the next round's K tile issued from global into
    // registers in the middle of the current round and committed to the stage only after that round's V has
    // been read, a steady-state round carrying NO per-cell bounds or mask test (the tail rounds, which a wide
    // block can reach at most two of because its causal spread is `WIDE_POS_PER_BLOCK - 1 = 31` lanes and a
    // round is 64, are a second instantiation of the same inlined body), and the softmax scale folded into Q
    // at staging. The routing difference is ONE branch choosing a function pointer on the same grid — nothing
    // about knob 0 or knob 1 moves when this is 2 — and the numbers difference is NONE: every ported item is
    // value-preserving, so 2 owes BYTE identity to 1 (`SCALE` is 2^-4, a swizzle permutes bytes within a stage
    // row, a prefetch moves when a load happens, a specialised predicate that cannot fire computes the same
    // value). `wide_echo` prints `form=wide128llama` and the NUMERICS echo prints `wide_llama` so that claim
    // can be witnessed against the right binary.
    int fattn_wide = 0;

    // How deep the split path may cut a wide query row (`--attn-split-cap`, bet A9-L-c). The lane geometry
    // FORCES `split_for(lanes_last)` lane blocks (§07.3); this is an upper bound on that count, and the
    // launch uses `min(split_for(lanes_last), split_cap)`. `kAttnSplitCapOff` (0) leaves the bound inert,
    // which is the shipped behaviour and what every gate in this tree judges.
    //
    // WHY A CAP EXISTS: the split's partial round-trip is 105.5 GB written + 105.5 GB read back per pass at
    // 32,768 lanes, and `attn_combine` was traced reading it at 208 GB/s = 88 % of this part's DRAM roof —
    // the one item in `prototype/attn_percost/A9-LONGCTX-CENSUS.md` §2.3 whose rate is measured rather than
    // modelled. Capping at 8 removes 77 % of that traffic and costs the widest body 2.13 % more serial
    // depth (§4.2), because `ceil(64/8)` blocks of 4,096 lanes each still feeds the machine 5.87 block-waves.
    //
    // NOT bit-identical, and cannot be: fewer partials re-associates `attn_combine`'s merge, so this is
    // judged by the existing §10.1 cos/meanrel bands exactly as A4-L1's two-accumulator roll was
    // (`prototype/attn_percost/A4-BET.md`), never by a new band and never by loosening one.
    //
    // WHERE A CAP MAY BE ASKED FOR: a capped split makes a block's lane span `ceil(lanes/split)` BIGGER than
    // `MAXSC`, which the WMMA wave handles (it stages `FATTN_MLANES` lanes a round and its round loop just
    // runs longer, `engine/attn_fattn_kernels.h:475-479`, `:771`) and the two VALU tiled spellings do not
    // (their score staging is `s_sc[ATTN_TILE_G][MAXSC]`, `engine/attn_slim_kernels.h:157`, indexed by the
    // whole slice). So `attn_decode_layer` REFUSES a cap that binds under `--attn-fattn 0` instead of
    // overrunning LDS; the refusal is host-side, costs no register and no LDS, and is the same shape as the
    // `attn_m2` prototype's standing rule that a grid shallower than `split_for_lanes` is not launchable.
    // The lane CEILING is priced on the UNCAPPED split, so a cap never widens `MAXSC × kAttnMaxSplit`.
    int split_cap = kAttnSplitCapOff;
};
void attn_decode_layer(const AttnLayerArgs& a);

// The score-roll spellings' parse/name pair, `parse_gdn_geom`'s shape for the same reason: this names which
// side of an A/B a run is, so a typo that silently became the default would price the shipped roll twice and
// report the second price as the second accumulator's win. Defined in `kernels_attn.cpp`, the TU that owns
// the two kernels.
FattnRoll parse_fattn_roll(const std::string& name);
const char* fattn_roll_name(FattnRoll r);

// The Q-placement spellings' parse/name pair, same rule and same reason as the roll's: this names which side of
// an A/B a run is and, unlike the roll, it carries a numerics claim (byte identity with the default), so a value
// that silently fell back to the default would price the shipped form twice and report the second price as the
// un-stage's win. Defined in `kernels_attn.cpp`, the TU that owns the four kernels.
FattnQStage parse_fattn_qstage(const std::string& name);
const char* fattn_qstage_name(FattnQStage s);

// How many F32 elements `AttnLayerArgs::part` must hold for a body of `rows` query rows: one (max,
// sum, acc[head_dim]) tuple per query head per split block, at the maximum split the engine serves.
// Defined in `kernels_attn.cpp`, because the head width and the split cap are that op's constants and
// `decode_step` should not have to know either one to size its scratch.
std::size_t attn_part_floats(int rows = 1);

// ---- §03.4: the tied logits GEMV (prototype/lm_head_gemv), full-logit form. The fused-argmax
// form exists but resolves ties to the highest id, and §03.4 fixes ties to the lowest, so the
// engine computes the logits and takes the argmax host-side (results.md, "the last 2 KB").
void lm_head_full(const uint16_t* table, const float* h, float* logits, int vocab);

// The same GEMV with the norm row staged as BF16, which deletes the last of the 97 widenings
// (`engine/lm_head_xbf16_kernels.h`). `h` is the BF16 norm row itself. Bit-identical for the same
// reason as every other `_xbf16` entry point here; it is the one folded kernel with no headroom in its
// own family, and `prototype/cast_fold/ITEM1-HANDOVER.md` says why that matters for reading the A/B.
void lm_head_full_xbf16(const uint16_t* table, const uint16_t* h, float* logits, int vocab);

// The batched tied head: `nrows` query rows, one pass over the 485 MiB table (§03.4, P7's 2-row verify).
// `h_rows` is `nrows` contiguous rows of the BF16 final-norm tile and `logits_rows` `nrows` contiguous
// vocabulary rows — one launch and one D2H for the pair, which is the whole point (`D3-RESULT.md` priced
// the two-pass alternative at KILLED). `kLmHeadBatchedRows` is the only width instantiated, and the
// launcher refuses any other rather than guessing: k>1 is unfunded
// (`prototype/mtp_draft/DRAFT-RESULT.md`'s closing instruction). Bit-identity of row 0 against
// `lm_head_full_xbf16` is claimed in `engine/lm_head_batched_kernels.h` and witnessed by
// `prototype/mtp_verify`'s byte compare, not by the header.
constexpr int kLmHeadBatchedRows = 2;
void lm_head_batched_xbf16(const uint16_t* table, const uint16_t* h_rows, float* logits_rows,
                           int vocab, int nrows);

// ---- dtype adapters, engine's own (no model math here at all). These exist only because some
// graduated kernels take F32 activations (`proj_gemv` with PACK_X=false, `mlp_gemv`, `lm_head_gemv`)
// while the activations the other kernels produce are BF16. `gdn_decode`'s conv and gates used to be
// on that list; they read the BF16 tiles their projections wrote as of D30 1.7'.
//
// The `_xbf16` entry points above exist to delete these: a consumer that stages the BF16 itself needs
// no widening in front of it, and a BF16 widened in registers is the same value as one widened by a
// launch. What is left here after the fold is the two one-time weight-adapter widenings in
// `decode_step.cpp`'s `prepare_weight_adapters()` and whatever a run with the fold off still submits.
void cast_bf16_to_f32(const uint16_t* src, float* dst, int n);
void cast_f32_to_bf16(const float* src, uint16_t* dst, int n);

// ---- §02.3: the embedding lookup for a whole wave in ONE launch (D30 4.2′).
// `out[r] = table[ids[r]]` for `rows` rows, BF16, `kHidden` wide — byte-identical to `rows` copies of
// one table row, because there is no arithmetic in it (`kernels_embed.cpp` says why the engine grew a
// kernel for a lookup: 4096 of a wave's copy operations were 2 KiB memcpys and 84 % of each one was
// the submission). `ids` is a DEVICE array of `rows` token ids in [0, vocab); the caller owns it and has
// already range-checked the values, because a bad id here is an out-of-bounds read of the tied table
// rather than a throw.
void embed_gather_rows(const uint16_t* table, const int* ids, uint16_t* out, int rows);

// ---- §09.2: the MTP draft block's OWN attention row --------------------------------------------
//
// One query row over the draft block's own 1-layer KV cache, at an ABSOLUTE rope position that is
// DECOUPLED from the lane the k/v go to. That decoupling is the whole reason this is a separate entry
// point and not a call to `attn_decode_layer`: `attn_core` couples the two (`row_pos = a.pos + qrow`
// feeds both `head_norm_rope` and `slot_of(row_pos, cap, …) = pos % cap`), which is right for the
// target — every position writes a lane, so the dense lane index IS the position — and wrong for the
// draft block, whose cache holds one lane per DRAFTED step while the target consumes k+1 per round
// (`speccheck/mtp.py`'s `greedy_mtp` drafts at q+1, q+3, q+5 … at k=1). The gaps cannot be filled: those
// positions were never drafted and a zero lane earns real softmax weight.
//
// `engine/kernels_attn.cpp` defines it as a transcription of `attn_core`'s fused one-pass `full` path
// with that ONE change, and names the witness that licenses the copy (`prototype/mtp_draft`'s `xcheck`
// arm: bit-identity with `attn_core` on the only geometry where the two addressings coincide). It is
// declared HERE rather than included from the op header because `attn_decode_kernels.h` defines its
// `__global__`s with external linkage and `engine/kernels_attn.cpp` already includes it; a second TU
// including it would be a duplicate-symbol link error, which is why this TU is where the kernel lives.
struct MtpAttnRow {
    const uint16_t* qg = nullptr;    // [8 * 2 * 256] this step's q_proj row, [q 256 | gate 256] per head
    const uint16_t* kg = nullptr;    // [2 * 256]
    const uint16_t* vg = nullptr;    // [2 * 256]
    const uint16_t* qn = nullptr;    // [256] q_norm weight
    const uint16_t* kn = nullptr;    // [256] k_norm weight
    uint16_t* kcache = nullptr;      // [2 * cap * 256] plane-major, the target's own layout
    uint16_t* vcache = nullptr;
    uint16_t* aout = nullptr;        // [8 * 256] pre-gate attention out (§05.5)
    uint16_t* gout = nullptr;        // [8 * 256] §05.6 gated out — o_proj's operand
    int cap = 0;                     // lane stride of the cache
    int nlanes = 0;                  // valid lanes AFTER this step's append
    int self_lane = 0;               // the DENSE slot this step's k/v go to
    int pos = 0;                     // ABSOLUTE position, for rope only
};
void mtp_attn_row(const MtpAttnRow& a);

// The longest draft cache the one-pass form can serve (`MAXSC`, that op's constant, read not retyped).
// Past it the streaming/split route would be needed, which is a decision about the draft cache's
// lifetime and not a store the launcher may grow by accident.
int mtp_attn_max_lanes();

}  // namespace engine::kernels
