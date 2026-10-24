// `decode_run` — the driver `engine/gate.py` runs, and the only thing on this box that puts
// `decode_step` through more than one token.
//
// What it is for: `decode_step` is a function, not a program, and a function that has never run cannot
// be gated. This turns a list of token ids into the things a gate can compare against the pinned
// reference dumps: one sampled id and one full logits vector per step, plus the state bookkeeping the
// gate has to tie to the fixture (which position each step ran at, and how big the engine thinks its
// state is).
//
// It deliberately knows NOTHING about the reference fixtures. The token ids come in on the command
// line and the pins are read by the Python side. That split is the point: `gate.py`'s value is that it
// is independent of the engine's arithmetic, and a driver that could read a pin would be a way for
// the engine to accidentally depend on one. `prototype/*/run.py` set this pattern — the C++ emits, the
// Python judges.
//
// Three ways to drive it, and the split between them is deliberate:
//   run      `--ids … --steps N`      the prompt as ONE prefill wave (§06.1, `engine/prefill.h`),
//                                     then free generation: the engine feeds its own samples.
//   replay   `--ids … --replay …`     each generation step is fed the CALLER's token, so a
//                                     divergence shows up at one step instead of cascading
//                                     (that is what `gate.py` uses for the 64-step forced pin;
//                                     §07.5 does not end a replay — the caller owns the stream).
//   session  `--session`              resident line protocol for `engine/chat.sh`: weights and
//                                     `RequestState` stay alive across turns. See `session_loop`.
//
// THE TWO PHASES ARE DIFFERENT CODE ON PURPOSE (§07.5's phase rule). The prompt is a wave and has no
// stop check in it; generation is a `decode_step` loop and is the only place `stop_fired` can be
// computed. Before P2 both were one loop with a `from_prompt` flag, and the first device run armed the
// stop check on the wrong side of that flag. The flag is gone: the phase is now which loop a step came
// out of.
//
// Every number it prints about *time* is provisional by construction: see `engine/performance-status.md`
// §6 for what a clean window takes to arrange, and `AGENTS.md` for why a worker must not arrange one.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "engine/decode_step.h"
#include "engine/launch_stats.h"
#include "engine/loader.h"
#include "engine/mtp_draft.h"
#include "engine/mtp_verify.h"
#include "engine/mtp_accept.h"
#include "engine/mtp_meter.h"
#include "engine/prefill.h"
#include "engine/stage_clock.h"
#include "engine/state.h"

#include <memory>

namespace {

// Slot count a `--session` allocates when `--cap` is not given. Not a limit the engine enforces, and
// it is not the lane ceiling either: attention's ceiling is `MAXSC × kAttnMaxSplit`, which lives in
// `kernels_attn.cpp` and is read (never retyped) by `engine/gate.py::lane_ceiling()`; it now sits well
// above this default, so a session hits this cap first and `--cap` is how it reaches the wider
// geometry.
constexpr int64_t kSessionDefaultCap = 4096;

struct Options {
    std::string model_dir = "models/Qwen3.5-0.8B";
    std::string out_dir;             // where logits/sampled dumps go; empty means do not write
    // `--dump-attn-rows`: also write the attention block's own per-row output tiles into `out_dir` and
    // print one `ATTN_DUMP` line per body (`kernels.h:AttnLayerArgs::attn_dump_dir`). An instrument, and
    // it lives or dies by being byte-identical when absent.
    bool dump_attn_rows = false;
    std::vector<int> prompt;         // the tokens fed before generation starts (the fixture prefill)
    std::vector<int> feed;           // --replay: what to feed each GENERATION step, from outside
    int steps = 8;                   // decode steps after the prompt
    int64_t cap = 0;                 // 0 = derive from prompt+steps
    bool load_verify = false;        // the loader's SHA-256 pass; not this gate's concern
    bool sizes_only = false;         // check §00/§01 against the checkpoint and exit
    bool session = false;            // resident line protocol for engine/chat.sh
    bool no_stop = false;            // §07.5 stays unarmed in generation too (deep smoke runs only)
    bool state_fp = false;           // emit the §08.6 state fingerprints (see the `--state-fp` block)
    // §09: the MTP draft block as a SIDE computation (`engine/mtp_draft.h`). Default OFF, and off means
    // the binary is byte-identical in behaviour to the one that shipped without it: no extra allocation
    // is touched, no extra submission is made, and no line below prints. On, the block reads the
    // finished step's final-norm row and its emitted token and produces a proposal id, and it must not
    // alter one byte the target path consumes — which is exactly why this knob's admission test is
    // `sampled.txt` and every STATEFP row byte-identical with it on and off, and not a band.
    //
    // Environment-defaulted like QWEN_CAST_FOLD, for the reason given there: a gate that runs the
    // shipping path must not have to remember to pass a flag, and the flag below is how an A/B names
    // the side it is on.
    int mtp_draft = 0;
    bool mtp_draft_given = false;
    // §09 / P7: the batched 2-row VERIFY body, as a measured side pass (`engine/mtp_verify.h`). Default
    // OFF, and off means the binary does nothing differently: no allocation, no submission, no line.
    // On, every generation step also runs one 2-row body over the populated KV on a SHADOW state, prices
    // both rows with one pass of the tied head, and reports both argmaxes — and the target's own bytes must
    // still come out byte-identical, because nothing here consumes the result. There is deliberately no
    // accept loop: emitting two tokens per step is the next increment, and the point of this ordering is
    // that the riskiest arithmetic in the repo gets measured before any output depends on it.
    int mtp_verify = 0;
    bool mtp_verify_given = false;
    // The id fed as row 1 at every anchor: `""` (default) means "the token this step was fed" (a real
    // second row at the next position, but a degenerate draft), `"sampled"` means "the token this step
    // predicted" — the TRUE next token, which makes one probe run comparable against a plain run's own
    // logits files at EVERY anchor (`prototype/mtp_verify/gate.py`'s row-1 witness, a reproduction rather
    // than a filename) — and any decimal id is a genuinely wrong candidate, which is the negative: row 1
    // must move and row 0 must not.
    std::string verify_candidate;
    bool verify_candidate_given = false;
    // The witness pair, `--verify-guard 1` (`engine/mtp_verify.h`'s operand/guard section). Per round it
    // downloads what the step was about to read and what the shadow body actually read and byte-compares
    // them, then poisons the one KV lane the body is entitled to write and re-reads every live buffer after
    // it. Default OFF for one reason and one only: it is tens of MiB of downloads per round, and
    // `prototype/mtp_verify/ARMS.md`'s arms must not pay for a witness while timing. The admission gate
    // therefore names it explicitly rather than trusting a default, and the `VERIFY2` line prints which way
    // it was set, so a timing run can be shown to have run without it.
    int verify_guard = 0;
    bool verify_guard_given = false;
    // M2's split admission, `--mtp-midsplit 1` (`prototype/mtp_accept/ACCEPT-BUILD.md` §4 cond. 2).
    // On, the probe runs its 2-row body a SECOND time per anchor with V1's mid-round checkpoint armed,
    // and the two runs are byte-compared against each other and against the step. It is a witness, not a
    // route: it doubles the probe's work per round and prints one line, and it is refused with
    // `--time-stages` for the same reason the probe itself is. Default 0 = the probe runs one body and
    // submits nothing extra.
    int mtp_midsplit = 0;

    // Which weight walk and which GDN scan the MTP body runs — `--mtp-route hand|amortised`
    // (`engine/mtp_verify.h`'s "WHY THAT FORCING IS A FLAG NOW"). `hand` is the default and keeps today's
    // launch sequence byte for byte: the body forces the hand GEMV and the sequential scan so its row 0 is
    // the same kernel chain the one-row step ran, which is what makes `row0_eq_step` a claim at all.
    // `amortised` leaves the scratch's own resolved route in place — the route a wide body takes everywhere
    // else, which `prototype/mtp_accept/ROW-SCALE.md` measured at 1.09–1.13 of a step against the forced
    // route's 2.085 — so the round can be priced on the route that makes speculation pay, before anyone
    // builds a k=2 loop on top of the expensive one. It is not owed bit-identity with the step and gains no
    // new band: the existing gates judge it and its divergence is REPORTED. Parse-or-throw like every flag
    // that names a side of an A/B, and echoed on `VERIFY2`/`MTPSTAT` together with the route that ACTUALLY
    // ran, because a flag that silently did not take would price one route twice and report the second
    // price as the other route's (`AGENTS.md`: a stale binary agrees with itself).
    engine::mtpverify::Route mtp_route = engine::mtpverify::Route::kHand;
    bool mtp_route_given = false;

    // §09 / M2: the ACCEPT loop (`engine/mtp_accept.h`). Speculative decoding at k = 1: the draft's
    // proposal is verified by the 2-row body and accepted or rejected ON DEVICE, so a round emits one or
    // two tokens. Default off, and off means this pointer is never allocated (§4g's inertness argument).
    //
    // It costs a third ~19.27 MiB region per request (the mid-round checkpoint) and it changes the shape
    // of a generation step, which is why it refuses every option whose per-step semantics it does not
    // own — see the refusals below rather than half-supporting them.
    int mtp_accept = 0;
    // B0 of `prototype/mtp_accept/BEAT-MTP.md`: the per-phase ROUND METER (`engine/mtp_meter.h`). Default
    // OFF, and off is one null-pointer read per guard object — no allocation, no sync, no line, so a
    // flag-off run is byte-identical to a binary that does not contain the instrument. On, it prints one
    // `MTPMETER` line per phase of the round shape this run is in and folds a `meter=ok … coverage=…`
    // verdict onto the run's own `RESULT` line.
    //
    // It is a MEASURING instrument with a price: several phases can only be bounded by a device sync the
    // shipping path does not perform, and those lines say `sync_added=1` because such a phase SERIALIZES
    // and therefore INFLATES. That is why it refuses `--time-stages`/`--time-waves` (two instruments, each
    // draining the queue, would price neither shape) and why the metered round's total is an upper bound
    // on a production round rather than the round itself. `prototype/mtp_accept/METER-PLAN.md` states what
    // each phase can under- or over-report; the harness refuses to tabulate a run whose coverage says the
    // phases do not account for the window.
    int mtp_meter = 0;
    // Which round sequence the side computation walks: `loop` reproduces speccheck's greedy_mtp(k=1)
    // round geometry (the one the carried accept rate was measured on), `every` drafts after every
    // step. See `engine::mtpdraft::Schedule`; refused-and-parsed, never defaulted into.
    std::string mtp_sched = "loop";
    bool mtp_sched_given = false;
    // Lanes allocated for the draft's own KV cache. The one-pass attention form serves at most
    // `mtp_attn_max_lanes()` of them, so the usable ceiling is the smaller of the two and a round past
    // it is REFUSED and counted (`DRAFT-SKIP`), not silently served a shorter cache.
    int mtp_cap = 512;
    // How many draft-block ROWS one drafted position gets. `1` is the shipped path and the default: one
    // block call, one proposal. `k>1` feeds each proposal back into the block as the next row's `y_next`
    // and the block's OWN output row `x_d` as the next row's `h` — there is no target row to read behind
    // a token the target never fed — which is the feedback llama.cpp's draft-mtp uses
    // (`engines/llama.cpp/common/speculative.cpp:1770-1783`: `id_sampled` plus
    // `llama_get_embeddings_nextn_ith(ctx_dft, i_last)`, at position `pos0 + i + 1`). Every row is a
    // whole round: 16 submissions including its own head pass and vocabulary D2H
    // (`kSubmissionsPerRound`), so this is a cost of k x 16 per drafted position, not k x a fraction.
    //
    // It exists to MEASURE the second position's accept rate — the number that decides whether a k=2
    // round is worth building — and not to ship speculation: `--mtp-accept` still consumes exactly one
    // candidate per round (its body is `kRows` = 2) and refuses k>1 rather than price rows nothing
    // verifies. Default-off and named-by-flag: with the flag absent nothing below allocates, runs or
    // prints, which is what keeps flag-off byte identity a fact about the code path.
    int mtp_draft_k = 1;
    bool mtp_draft_k_given = false;
    // On an ACCEPT the round consumes two positions and the loop appends one draft lane, so the draft
    // cache skips a position and every lane behind it sits one rope step further from its neighbour than
    // the block was trained to see. llama.cpp's draft context has no such hole: its catch-up feeds
    // `ctx_dft` one row per target row. `--mtp-dense 1` stages the accepted candidate row as a lane too
    // (9 submissions, no head pass — `engine/mtp_draft.h:draft_lane`) so the cache stays dense. Default
    // off: it is a draft-side conditioning change, judged on accept rate, not a shipped improvement.
    int mtp_dense = 0;
    bool mtp_dense_given = false;
    // Seed the draft cache with the last N prompt positions' lanes before generation starts, so the
    // drafter's attention sees the tail of the prompt instead of starting from an empty cache. llama.cpp
    // gets this for free (its catch-up runs over the prompt ubatches); we have to stage it, and the norm
    // rows it needs only exist if they were retained out of the wave scratch — see
    // `engine/prefill.h::NormTailRetention` and the sizing at the prefill call site. 0 = the shipped
    // behaviour, empty cache.
    int mtp_prompt_lanes = 0;
    bool mtp_prompt_lanes_given = false;
    // Added to every rope position the draft block sees, uniformly across the whole cache. The engine's
    // convention (§09.1, and what speccheck/mtp.py and the pinned §09 gates measured) puts the drafted
    // token's position on the row; llama.cpp puts the fed token's position there, one less. Because the
    // block's attention is rope-RELATIVE, a uniform shift must be an exact null — this knob exists to
    // measure that rather than assume it, and to make the alternative convention one flag away if a
    // measurement ever asks for it. Default 0 = the shipped convention.
    int mtp_pos_shift = 0;
    // Which logits a prompt wave gives back (§06.5 / `engine/prefill.h`). `last` is the spec's shape
    // and the default; `all` is the gate's byte-identity mode, which asks for the rows §06.5 throws
    // away so they can be `cmp`'d against what the per-token prompt loop used to write.
    engine::PrefillLogits prefill_logits = engine::PrefillLogits::kLast;
    // How many rows the wave scratch is tall, i.e. how many tokens one wave may batch (§06.1, plan
    // §2.1). The default is the engine's ceiling, `kMaxWaveRows`; `--wave-rows 1` is the P4b
    // anti-vacuity knob — it makes `prefill_wave` split the prompt into waves of one token, so every
    // glue/MLP submission takes the single-row form and the run becomes exactly the "batching is dead
    // code" case that bit-identical output cannot distinguish from a batched run. Running the same
    // prompt both ways and comparing counters AND dumps is `engine/gate.py` section (h).
    int wave_rows = engine::kMaxWaveRows;
    // Which weight walk the MLP's gate/up launches use (Q10's A/B; `kernels.h:MlpGateMap`). The
    // default is kAuto, i.e. the body's row count picks the walk per body (Q10b); naming a walk pins
    // it for every body, which is what makes the flag an override and not a second opinion.
    engine::kernels::MlpGateMap mlp_gate_map = engine::kernels::MlpGateMap::kAuto;
    bool mlp_gate_map_given = false;
    // Which GEMM a WIDE body runs (route (c), `engine/kernels.h:WaveGemm`). The default is LIBRARY:
    // D-11 accepted the route's wave-width dependence and the route-c audit proved the hand<->library
    // delta to be reassociation (progress/2026-10-14/route-c-audit-2026-10-14.md), so the funded route is what ships.
    // The admitted guarantee is a band, not bit-identity — `engine/gate.py` judges a library run against
    // CASCADE_BAND and a hand run byte-identically, which is why `hand` stays a first-class route rather
    // than a legacy alias: it is the A/B side (§7 W9) and the way to reproduce a pre-route binary's bits
    // in the same window. A one-row body is the hand GEMV under either setting (`resolve_wave_gemm`),
    // so this flag cannot move a decode bit either way.
    // Shipped default since `--gemm lt` was measured in situ and its one red row was attributed to a
    // one-ULP tie: GROUP matmul 1.503x at 1024 lanes and the whole wave +11.0% / +8.4% (D28), with the
    // gate green on all nine fixtures at both routes under the ULP flip line (D29). `library` is kept as
    // a first-class value precisely so that claim stays A/B-able in one binary; `legacy` is its alias.
    engine::kernels::WaveGemm wave_gemm = engine::kernels::WaveGemm::kLibraryLt;
    bool wave_gemm_given = false;
    // Which algo pins are in force (`all` / `none` / csv of shape names). Only meaningful with
    // `--gemm lt`; see `kernels.h::gemm_pins_select`.
    std::string gemm_pins = "all";
    // How many candidates the lt heuristic query is asked for per (shape, M). -1 means "not given", so
    // the DEFAULT lives in one place (`kernels_gemm.cpp::kHeuristicWidth`) and cannot drift here the way
    // `gate.py`'s `--attn-tile` default would have if it had been copied instead of read — and the
    // sentinel is -1 and not 0 because 0 is a value worth refusing, not a value to silently ignore.
    // Named only by an A/B that wants a different pool; see `kernels.h::gemm_heuristic_width_select`.
    int gemm_heur = -1;
    // Which attention FORM a wave body runs (Phase 2, `kernels.h:AttnLayerArgs::wave_tile`). 1 is the
    // SHIPPED default since `prototype/attn_decode/results.md` §21: the G x 1 tiled `attn_tile_wave`, which
    // measured 26.70 ms against the shipped form's 107.67 ms at 1024 lanes and is BIT-identical to it on
    // all 656 above-MAXSC rows of both wide anchors. 0 selects the per-(row,head) `attn_core`, kept as the
    // frozen witness the tile form was proved against, so a before/after attribution is still one flag and
    // one binary. A one-row body is the fused single-row form either way, so no decode bit can move under
    // this flag (`attn_decode_layer` enforces that, not the flag parser). Parse-or-throw like `--gemm`:
    // this names which side of an A/B a run is, and a typo that silently became the default would price the
    // default twice and call it the tiled form.
    int attn_tile = 1;
    bool attn_tile_given = false;
    // Which SPELLING of the tiled wave's per-touch inner work runs (`--attn-touch`, the follow-up to
    // `--attn-tile` rather than a third value of it: `--attn-tile 0` has to keep pricing the per-(row,head)
    // `attn_core` exactly as it did before this flag existed, so the two A/Bs stay independent).
    // `1`, the default, runs `engine/attn_slim_kernels.h`'s `attn_slim_wave`: the score dot's trip count
    // made compile-time (it was runtime only because of the `NEG_DIMHALF` injection, and LLVM rolled the
    // hot dot to match) and the warp reduce folded by hand with an explicit `s_waitcnt`, which together
    // take one (row, lane, head) touch from 116 issue slots to 45 and measured 1.52-1.57x on the wave
    // (`prototype/attn_valu_slim/results.md` §7). `0` runs the prototype's `attn_tile_wave`, the frozen
    // side the census and the timing baseline are taken on. Both kernels are in this binary whatever this
    // says, so the paired timing happens in one window — the difference between a number and a claim
    // (`AGENTS.md`: a ruler re-taken in a different window is not an A/B). Same contract as `--attn-tile`:
    // bit-identity, wave-only (a one-row body is the fused single-row form either way, so no decode bit
    // moves), echoed on the NUMERICS line because a log that cannot say which spelling produced its bits
    // adjudicates neither side. Parse-or-throw like every other A/B flag here.
    int attn_touch = 1;
    // Which FORM owns a wave body's STREAMING rows (`--attn-fattn`, Phase: the WMMA port).
    // `prototype/attn_wmma/fattn_score_handover.md` §5a is the contract this flag reaches; the coverage
    // rule and why it is what it is are written at the top of `engine/attn_fattn_kernels.h`. The default
    // is 0: the two fragment-map checkers are green on device, the form itself has not been admitted, and
    // a form that has not met the device does not become the engine's default by having been written.
    // Parse-or-throw like `--attn-tile`, and echoed on the NUMERICS line for the same reason: a log that
    // cannot say which form produced its bits adjudicates neither side.
    int attn_fattn = 1;   // default since the form met the device (1.3955x attn at 4096 lanes, wide1024f green on both flag values, streaming rows inside the floors in force). THIS is the CLI default; the field of the same name in decode_step.h is an internal struct and was flipped alongside it, not instead of it.
    // Which SPELLING of that form's score roll (`--attn-fattn-roll`, bet A4-L1).
    // `single` — one accumulator, 16 dependent `v_wmma` trips — is the code path this header carried before
    // the knob existed, so the default side of the A/B is HEAD's own machine code, which is what makes
    // "default is inert" a byte-compare rather than an argument. `two` gives the roll a second accumulator
    // and fetches both tiles of a pair before either matrix op issues (`engine/attn_fattn_kernels.h`,
    // `prototype/attn_percost/A4-BET.md`); it is NOT bit-identical and cannot be, since two accumulators
    // re-associate the fixed ascending F32 dot, and A4-BET fixes the existing §10.1 bands as the judge.
    // Parse-or-throw like the flags above, and echoed on the same NUMERICS line for the same reason.
    engine::kernels::FattnRoll attn_fattn_roll = engine::kernels::FattnRoll::kSingle;
    // Whether the knob was NAMED, which is the only honest key for the inertness refusal below: `single` is
    // the default, so a guard written as `roll != single` would turn every bare run into a refusal the day
    // the default moved — the `--gdn-geom` precedent, stated at `--gdn-norm hoist`'s refusal above.
    bool attn_fattn_roll_given = false;
    // Where that form keeps Q, and how deep its output transpose stages (`--attn-fattn-qstage`, bet A6-G1).
    // `stage` — Q in a per-warp 8 KiB stage, the epilogue transposing 16 rows at a time — is the code path this
    // header carried before the knob existed, so the default side of the A/B is HEAD's own machine code and its
    // inertness is a byte-compare. `unstaged8` reads Q from global at the use site and stages the epilogue
    // FATTN_TRG = 8 rows a pass, which takes the block's declared LDS from 32,768 B to 20,480 B — 3 blocks/CU
    // instead of 2, 9 wf/CU instead of 6 (`engine/attn_fattn_kernels.h` placement (E),
    // `prototype/attn_percost/A6-BET.md`). Unlike the roll above it, this knob is OWED byte identity with the
    // default rather than judged by a band: it moves staging addresses and where an operand is fetched from,
    // never a value or an order. Parse-or-throw like the flags above, echoed on the same NUMERICS line.
    engine::kernels::FattnQStage attn_fattn_qstage = engine::kernels::FattnQStage::kUnstaged8;
    // Whether the knob was NAMED, for the identical reason as the one above: `stage` IS the default, so a guard
    // keyed on its value would manufacture a refusal for every bare run.
    bool attn_fattn_qstage_given = false;
    // What width the WMMA wave's P*V accumulator is (`--attn-fattn-pvf16`, bet A10-F16). 0 — the shipped F32
    // accumulator, 16 tiles of 8 F32 = 128 registers a lane — is the default and is HEAD's own machine code, so
    // the default side of the A/B is a machine-code compare rather than an argument. 1 is 8 tiles of 16 F16
    // halves = 64 registers for the same 256 output dims, filled by two `v_wmma_f16_16x16x16_f16` issues per 32
    // dims (`engine/attn_fattn_kernels.h`, `prototype/attn_percost/A10-F16-IMPL.md`). It is an ENABLER: 16
    // `v_wmma` issues per round either way, so the expected throughput effect is ~0 and what it buys is the 64
    // registers a later Q-in-registers spelling needs (D7). Unlike `--attn-fattn-qstage` above it it is not owed
    // byte identity — a narrower accumulator is a different format — so its judge is the bands already in
    // `engine/gate.py`, with no new band and no widening. Parse-or-throw like its siblings; echoed on NUMERICS.
    int attn_fattn_pvf16 = 0;
    // Whether the WMMA wave is the A12 WIDE block at all (`--attn-fattn-wide`, bet A12). 0 — the 16-row WMMA
    // wave that ships, with `attn_slim_wave` keeping the one-pass rows and `attn_combine` merging the streaming
    // ones — is the default and is HEAD's own machine code, so the default side of its A/B is a byte-compare
    // rather than an argument. 1 hands EVERY row of a wave body to `engine/attn_wide_kernels.h`'s 128-row block
    // and launches neither `attn_slim_wave` nor `attn_combine`; the consequence, including the fact that the
    // pinned bit-exact one-pass rows move, is at `kernels.h:AttnLayerArgs::fattn_wide` and at the routing site.
    // Parse-or-throw like its siblings and echoed on the same NUMERICS line, because it names which kernel
    // produced a log's wave rows and a silently-defaulted side would price the shipping wave twice and report
    // the second price as the wide block's.
    int attn_fattn_wide = 2;   // A13 ships: the wide block with the ported round (engine/attn_wide_llama_kernels.h).
                               // MUST match `--attn-fattn-wide`'s default in engine/gate.py — one committed half
                               // of the pair makes the gate judge a kernel the engine does not ship.
    // Whether the knob was NAMED, for the identical reason: 0 IS the default, so a guard keyed on its value
    // would manufacture a refusal for every bare run the day the default moved.
    // Whether the knob was NAMED, for the identical reason: 0 IS the default, so a guard keyed on its value
    // would manufacture a refusal for every bare run the day the default moved.
    bool attn_fattn_pvf16_given = false;
    // Whether the knob was NAMED is NOT what the refusal below keys on, and that is a real difference from the
    // three above it: each of those is inert at its default value, so only `given` can separate "asked for and
    // contradictory" from "forwarded by a driver that always forwards every form flag". `--attn-fattn-wide 1`
    // is never inert — it replaces the wave core — so the value itself is the whole question, and a
    // `given`-keyed refusal would reject `--attn-fattn 0 --attn-fattn-wide 0`, which asks for nothing and which
    // `engine/gate.py` forwards as a matter of course.
    // How deep the split path may cut a wide query row (`--attn-split-cap`, bet A9-L-c). Where the lane geometry
    // forces `split_for(lanes_last)` lane blocks per wide row (§07.3), this is an UPPER bound on that count:
    // the launch runs `min(split_for(lanes_last), N)`, so a block's lane slice gets WIDER and the split's
    // partial round-trip — 105.5 GB written and 105.5 GB read back per pass at 32,768 lanes, read back at a
    // traced 88 % of the DRAM roof (`prototype/attn_percost/A9-LONGCTX-CENSUS.md` §2.3) — gets smaller. The
    // default `kAttnSplitCapOff` (0) is inert by construction, which is what makes the default side of the A/B
    // a byte-compare rather than an argument. Like `--attn-fattn-roll` and `--attn-fattn-qstage` it is NOT
    // bit-identical when it binds — fewer partials re-associate `attn_combine`'s merge — so the existing §10.1
    // bands judge it, and unlike them it is not wave-only: a long decode row streams through the same route.
    // Parsed or thrown, because it names which side of an A/B the run is and a silently-defaulted side would
    // reproduce the uncapped run exactly, look green, and be priced as the cap's win.
    int attn_split_cap = engine::kernels::kAttnSplitCapOff;
    // Which §04.4 scan a GDN body runs (P9, §04.7; route A is D-12). `seq` is the shipped recurrence —
    // one launch per token per layer, the form every decode pin is pinned through — and `collapse` is
    // the same recurrence held resident, one launch per layer per BODY. `chunk` is refused at
    // `parse_gdn_scan` and stays in the tree only as the record of why it was killed. One binary carries
    // the forms that live, so the before/after attribution is one window. A one-row body is `seq` under
    // either setting (`resolve_gdn_scan` enforces that, not the parser), so no decode bit moves either
    // way — which is the property that makes it safe to A/B a scan from the same binary at all.
    engine::kernels::GdnScan gdn_scan = engine::kernels::GdnScan::kSeq;
    bool gdn_scan_given = false;
    // Which SPELLING of the sequential recurrence's state walks `gdn_rec` submits (`--gdn-spell`,
    // `kernels.h::GdnRecSpell`). `unroll4` — the shipping spelling from D33 until D51:
    // `engine/gdn_rec_unroll4_kernels.h` is bit-identical to the shipped kernel and measured 1.153x on
    // the family row at 4096 lanes in the wave's own interleaving, not just at launch level. `ship` is
    // kept as a first-class value so the claim stays A/B-able in one binary — the AGENTS.md rule that a
    // ruler re-taken in another window is not an A/B — and `gate.py`'s section (j) runs EVERY side on
    // every fixture and byte-compares them, so "bit-identical" is re-proved per gate rather than
    // inherited from the handoff. Echoed on the NUMERICS line like every other knob here, because a
    // timing log that cannot say which spelling produced its family row is not evidence.
    // `fused` (D50) is the DEFAULT since D51: the fused norm prologue on the unroll4 walks, worth
    // 1.0645 on the `gdn_rec` family row at 4096 lanes (3/3 paired reps, worst rep 1.0200) and
    // −2.96 % of wall in the same window, and bit-identical through the shipping launch path
    // (gate section (j), re-run per fixture). It is flat at 1024 lanes (0.9909, 1/3 reps) and flat
    // on decode (1.0025, 2/3 reps), both inside the documented spread; `bench` §20 holds the rows.
    engine::kernels::GdnRecSpell gdn_spell = engine::kernels::GdnRecSpell::kFused;
    // Whether the flag was actually given, as distinct from the default above. The collapse-refusal
    // predicate below needs this: with a non-`ship` default, `--gdn-scan collapse` alone would
    // otherwise walk into a refusal the user never asked for.
    bool gdn_spell_given = false;
    // How many token-steps ONE `gdn_rec` launch carries (`--gdn-loop`, D32 → D50's knob-then-default
    // order, `engine/gdn_rec_loop_kernels.h`). `512` — the default since §24 — is one launch per
    // (layer, body) at the scratch's full width, the axis the item is on (73,728 launches became 144
    // for a 4,096-token prompt); `1` is the old one-launch-per-token form and stays selectable, because
    // it is the A/B's other side and the form every pre-D54 number in this repo was taken with.
    // What moved the default was not the harness screen (§23, 1.58–1.64x, bit-identity at spans
    // 0/1/3/16) but the paired in-situ family row (§24: `gdn_rec` 449.07 → 266.03 ms at 4,096 lanes,
    // 1024/1024 prefill logits rows byte-identical, decode flat at median 0.9999). A span below the body size means what it says
    // (ceil(rows/span) launches) because span-invariance is proved; a one-row body takes the
    // per-token path whatever this says (`resolve_gdn_loop`), which is what makes decode unchanged by
    // construction. Echoed on the NUMERICS line like every other knob here, because a timing log that
    // cannot say which launch shape produced its family row is not evidence.
    int gdn_loop = 512;
    // Whether the flag was named, as distinct from the default: the collapse refusal below must fire on
    // a run that asked for a span while running route A, not on one that never mentioned either.
    bool gdn_loop_given = false;
    // Where the looped recurrence's per-token k/q norms are computed (`--gdn-norm`, bet G1,
    // `engine/kernels.h:GdnNorm`). `inline` — the shipped `rec_loop_kernel`, which reduces the two rows
    // inside the walk — is the default, and `hoist` is the A/B side: a pre-pass computes the norms and
    // the walk reads them, so the walk carries no shared memory, no barrier and no cross-lane reduction
    // (`engine/gdn_rec_hoist_kernels.h`, `prototype/gdn_walk/BET.md` §3). Claimed bit-identical, like
    // every other knob on this recurrence, and `hoist` reaches the `--gdn-loop` span branch only: a
    // per-token body (`--gdn-loop 1`) runs the per-token kernels, which keep their own prologue, so
    // naming `hoist` there is REFUSED below rather than echoed as a form that did not run.
    engine::kernels::GdnNorm gdn_norm = engine::kernels::GdnNorm::kInline;
    // Whether the flag was named: route A never reaches the span branch at all, so a bare
    // `--gdn-scan collapse` carries an inert `inline` rather than a wrong one, and only a named
    // `--gdn-norm hoist` alongside collapse is the misreport worth refusing.
    bool gdn_norm_given = false;
    // The BLOCK SHAPE of the looped body (`--gdn-geom`, bets G2 and G4, `engine/kernels.h:GdnGeom`).
    // `block` — the shipped `<<<16,128>>>` looped walk — is the default and therefore the shipped
    // behaviour; `wf4` is BET.md §10's pre-registered one-wavefront-per-block form and `wf2` its split leg
    // (`engine/gdn_rec_wf_kernels.h`, measured DEAD in §42); `reg16` is bet G4's register-resident map —
    // 128 blocks of 128 threads, 8 lanes per state column, 16 state elements per thread held in registers
    // for the whole span (`engine/gdn_rec_reg16_kernels.h`, `prototype/gdn_walk/BET.md` §12). Echoed on the
    // NUMERICS line, and it reaches the span branch only: `--gdn-loop 1`, `--gdn-scan collapse` and
    // `--gdn-norm hoist` are each refused below rather than echoed as a shape that did not run.
    engine::kernels::GdnGeom gdn_geom = engine::kernels::GdnGeom::kReg16;
    bool gdn_geom_given = false;
    // The chunk size. Only 64 is accepted, and the refusal prints the measured reason rather than
    // "unsupported": C=16 lost §2.3's per-row budget on the pinned dumps, which is the argument for
    // why 64 is the default at all.
    int gdn_chunk = engine::kernels::kGdnChunkTokens;
    bool gdn_chunk_given = false;
    // How a ONE-ROW body groups its projection launches (`--proj-plan`, route D1, route D1's engine
    // port). `packed` is the shipped default and the pair the in-situ A/B funded: one launch over a
    // layer's input projections plus one for its output projection, 48 per token where §07 bills one
    // launch per projection (`prototype/proj_gemv/results.md` §D1-W4: 2.4942 vs 2.8244 ms at identical
    // geometry, bit-identical, one clock-controlled window; `engine/results.md` for the in-situ step).
    // `split` — §07's bill, 114 per token — is HEAD's behaviour and stays selectable so a pre-D1 log can
    // be replayed byte for byte. The plan is only meaningful with `--proj-shape unroll8`, because the
    // grouping is a feature of the grouped kernel, and the unbuildable pair is REFUSED below rather than
    // echoed as a plan that did not run.
    engine::kernels::ProjPlan proj_plan = engine::kernels::kProjPlanDefault;
    bool proj_plan_given = false;
    // Which kernel and block geometry a one-row projection launch uses (`--proj-shape`). `unroll8` is the
    // shipped default: `engine/proj_group_kernels.h` at rows/block 8 and 1024 threads — the winning arm
    // (§D1-W4: 2.4693 ms against the shipped shape's 3.2692 ms in the same window), bit-identical for the
    // same reason the prototype's gate accepted its `v1-unroll`: the word walk, the ladder and the rounding
    // are unchanged. `shipped` is `proj_gemv_tokens<IN,1,false>` at rows/block 32, i.e. HEAD's machine code,
    // and needs `--proj-plan split` alongside it (the grouped kernel is what does the grouping). Echoed on
    // the NUMERICS line like every other knob here, because a timing log that cannot say which kernel
    // produced its `proj` family row is not evidence, and a stale binary agrees with itself.
    engine::kernels::ProjShape proj_shape = engine::kernels::kProjShapeDefault;
    bool proj_shape_given = false;
    // Route S1 item (1): fold the per-token BF16->F32 activation widenings into the kernels that consume
    // the result, deleting the `cast` family's 97 launches (`engine/generation-limits.md` §1, §4's last
    // paragraph, §7's open item). false = HEAD's behaviour, and the default; see
    // `DecodeScratch::cast_fold` for what each side does and why it is bit-identical.
    //
    // IT IS ENVIRONMENT-GATED, NOT A COMMAND-LINE FLAG, and that is a deliberate difference from every
    // other knob in this struct. `engine/gate.py` builds the engine argv itself and carries its own list
    // of the form flags it must pass through to every side it runs, so a new flag is unreachable inside
    // the gate unless gate.py learns it — and gate.py is the one file in this tree a route may not edit to
    // make its own numbers come out. An environment default reaches the engine through the subprocess
    // environment gate.py already inherits, which means the folded path is gated today:
    //
    //   QWEN_CAST_FOLD=1 .venv/bin/python engine/gate.py --fixture dec8
    //
    // Same "both sides live in one binary, so the A/B is one window" property as `--attn-touch`, and no
    // edit to the judge. It is echoed on its own `CASTFOLD` line (`main`, beside `LAUNCH`) because a
    // setting that is not on the command line is otherwise invisible in a log, which is how a silently
    // inert side is born. Only "0" and "1" are accepted, and anything else is a hard refusal rather than
    // a default, because an unset-by-typo variable reads exactly like an unset-by-choice one.
    bool cast_fold = true;
    // Route S1's follow-on: fold the `copy` family's 19 submissions into the kernels that already touch
    // the same bytes — the embedding-row D2D into the layer-0 glue launch that reads the tied table row,
    // and each GDN layer's conv-window roll-back into the conv that produces the rolled window.
    // false = HEAD's behaviour and the default; `DecodeScratch::copy_fold` carries what each side
    // submits and `prototype/glue_copy_fold/census_sites.py` says which five call sites the 68 are.
    //
    // Environment-gated for the same reason as `cast_fold` immediately above, and it inherits every part
    // of that argument: `engine/gate.py` builds the engine argv itself and may not be edited to teach its
    // own judge a new flag, while an environment default reaches the engine through the environment the
    // gate already inherits. So the folded path is gated today:
    //
    //   QWEN_COPY_FOLD=1 .venv/bin/python engine/gate.py --fixture dec8
    //
    // and it is echoed on its own `COPYFOLD` line for the same reason `CASTFOLD` is: a setting that is
    // not on the command line is otherwise invisible in a log, which is how a silently inert side is born.
    bool copy_fold = true;
    // (no weight load, no HIP call), so geometry can be checked while another agent holds the GPU lock —
    // the same service `prototype/proj_gemv --dump-plan` provides for its own rig.
    bool dump_proj_plan = false;
    // The stage clock (`engine/stage_clock.h`): re-runs the run's generation steps twice more, once
    // with a wait at every family boundary and once with a wait after every submission, and reports
    // them next to the pipelined step. A bench-only observation flag: the run it is attached to still
    // produces the same dumps and the same lines, and the extra passes print nothing but `STAGE*`.
    bool time_stages = false;
    // The wave clock (`engine/stage_clock.h`'s second unit): replays the run's PROMPT wave twice more,
    // once in the production shape and once drained at every family boundary, and attributes the
    // serialized wave's wall across the launcher families. A separate flag from `--time-stages` on purpose:
    // a wave pass is two full prefills of the prompt, so folding it into `--time-stages` would silently
    // make every Q6 row several times more expensive, and the two instruments answer different questions.
    bool time_waves = false;
    bool help = false;               // print_usage and exit 0, before anything is loaded
};

std::vector<int> parse_ids(const std::string& csv) {
    std::vector<int> out;
    size_t i = 0;
    while (i <= csv.size()) {
        const size_t comma = csv.find(',', i);
        const std::string tok = csv.substr(i, comma == std::string::npos ? comma : comma - i);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        i = comma + 1;
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    // The environment-gated side of route S1 item (1); see `Options::cast_fold` for why this one knob
    // is spelled in the environment and not in argv.
    if (const char* env = std::getenv("QWEN_CAST_FOLD"); env != nullptr && env[0] != '\0') {
        const std::string v(env);
        if (v == "1") {
            o.cast_fold = true;
        } else if (v == "0") {
            o.cast_fold = false;
        } else {
            throw std::runtime_error("QWEN_CAST_FOLD takes 0 or 1, got '" + v +
                                     "'; 0 restores the pre-2026-10-21 behaviour (every widening submitted by"
                                     " kernels_cast.cpp) and 1 folds it into the consuming kernel"
                                     " (see engine/mlp_xbf16_kernels.h)");
        }
    }
    if (const char* env = std::getenv("QWEN_COPY_FOLD"); env != nullptr && env[0] != '\0') {
        const std::string v(env);
        if (v == "1") {
            o.copy_fold = true;
        } else if (v == "0") {
            o.copy_fold = false;
        } else {
            throw std::runtime_error("QWEN_COPY_FOLD takes 0 or 1, got '" + v +
                                     "'; 0 restores the pre-2026-10-21 behaviour (the embedding row and each conv"
                                     " window roll are submitted as D2D copies) and 1 folds them into"
                                     " the kernels that read what they write"
                                     " (see engine/gdn_conv_roll_kernels.h)");
        }
    }
    if (const char* env = std::getenv("QWEN_MTP_DRAFT"); env != nullptr && env[0] != '\0') {
        const std::string v(env);
        if (v == "1") {
            o.mtp_draft = 1;
            o.mtp_draft_given = true;      // the env IS the naming, for the `_given` refusals below
        } else if (v == "0") {
            o.mtp_draft = 0;
        } else {
            throw std::runtime_error("QWEN_MTP_DRAFT takes 0 or 1, got '" + v +
                                     "'; 0 is the default and runs no draft at all, 1 runs the §09 block"
                                     " as a read-only side computation over the finished step's hidden"
                                     " row (see engine/mtp_draft.h)");
        }
    }
    if (const char* env = std::getenv("QWEN_MTP_ACCEPT"); env != nullptr && env[0] != '\0') {
        const std::string v(env);
        if (v == "1") {
            o.mtp_accept = 1;
        } else if (v == "0") {
            o.mtp_accept = 0;
        } else {
            throw std::runtime_error("QWEN_MTP_ACCEPT takes 0 or 1, got '" + v +
                                     "'; 0 is the default and runs no accept round at all (see"
                                     " engine/mtp_accept.h)");
        }
    }
    if (const char* env = std::getenv("QWEN_MTP_VERIFY"); env != nullptr && env[0] != '\0') {
        const std::string v(env);
        if (v == "1") {
            o.mtp_verify = 1;
            o.mtp_verify_given = true;      // the env IS the naming, as for the draft above
        } else if (v == "0") {
            o.mtp_verify = 0;
        } else {
            throw std::runtime_error("QWEN_MTP_VERIFY takes 0 or 1, got '" + v +
                                     "'; 0 is the default and runs no verify body at all, 1 runs the"
                                     " batched 2-row verify as a read-only side pass over the step's own"
                                     " state (see engine/mtp_verify.h)");
        }
    }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--model") {
            o.model_dir = next();
        } else if (a == "--ids") {
            o.prompt = parse_ids(next());
        } else if (a == "--ids-file") {
            // The same list, read from a file, because an argument cannot always carry it: Linux caps
            // ONE argv slot at MAX_ARG_STRLEN = 131,072 B (32 pages, independent of `getconf ARG_MAX`,
            // which quotes 2 MiB and misleads everyone), so `--ids` dies at roughly 21,800 ids with
            // exec E2BIG — shell exit 126 and "Argument list too long" — before this binary is
            // reached at all. That is what stopped tools/longctx_check.sh at 32,768 lanes (191,588 B
            // in one field) while 16,384 passed at 95,796 B: the wall was the harness, not the engine.
            const std::string path = next();
            std::ifstream in(path);
            if (!in) throw std::runtime_error("--ids-file: cannot open " + path);
            std::string all, line;
            while (std::getline(in, line)) { all += line; all += ','; }
            o.prompt = parse_ids(all);
            if (o.prompt.empty())
                throw std::runtime_error("--ids-file: " + path + " holds no token ids");
        } else if (a == "--replay") {
            o.feed = parse_ids(next());
        } else if (a == "--steps") {
            o.steps = std::atoi(next().c_str());
        } else if (a == "--out") {
            o.out_dir = next();
        } else if (a == "--cap") {
            o.cap = std::atoll(next().c_str());
        } else if (a == "--load-verify") {
            o.load_verify = true;
        } else if (a == "--sizes-only") {
            o.sizes_only = true;
        } else if (a == "--session") {
            o.session = true;
        } else if (a == "--no-stop") {
            o.no_stop = true;
        } else if (a == "--state-fp") {
            o.state_fp = true;
        } else if (a == "--mtp-draft") {
            // Parses or throws, like every flag that names which side of an A/B a run is on: a typo that
            // silently became "off" would report an off run as an on run, and the two differ by the
            // whole proposal path.
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--mtp-draft takes 0 or 1, got '" + v + "'");
            }
            o.mtp_draft = std::atoi(v.c_str());
            o.mtp_draft_given = true;      // the flag outranks the environment, and it is the naming
        } else if (a == "--mtp-sched") {
            o.mtp_sched = next();
            o.mtp_sched_given = true;
            engine::mtpdraft::parse_schedule(o.mtp_sched);   // parses or throws
        } else if (a == "--mtp-cap") {
            o.mtp_cap = std::atoi(next().c_str());
            if (o.mtp_cap < 1) throw std::runtime_error("--mtp-cap needs at least one lane");
        } else if (a == "--mtp-draft-k") {
            // Parses or throws, like every flag that names which side of an A/B a run is on: a `5` that
            // silently became the default would print a k=1 ladder under a k=5 label. Capped at 4 because
            // the `DRAFTK` line carries exactly four (t,c,hit) triples and a reader that silently got
            // fewer fields than it asked for is this repo's oldest failure.
            const std::string v = next();
            const int kk = std::atoi(v.c_str());
            if (kk < 1 || kk > 4) {
                throw std::runtime_error("--mtp-draft-k takes 1 (the shipped single proposal), 2, 3 or 4"
                                         " draft-block rows per drafted position, got '" +
                                         v + "'");
            }
            o.mtp_draft_k = kk;
            o.mtp_draft_k_given = true;
        } else if (a == "--mtp-dense") {
            // Parses or throws, like every flag that names which side of an A/B a run is on.
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--mtp-dense takes 0 or 1, got '" + v + "'");
            }
            o.mtp_dense = std::atoi(v.c_str());
            o.mtp_dense_given = true;
        } else if (a == "--mtp-prompt-lanes") {
            o.mtp_prompt_lanes = std::atoi(next().c_str());
            if (o.mtp_prompt_lanes < 0)
                throw std::runtime_error("--mtp-prompt-lanes takes 0 or more lanes");
            o.mtp_prompt_lanes_given = true;
        } else if (a == "--mtp-pos-shift") {
            o.mtp_pos_shift = std::atoi(next().c_str());
            if (o.mtp_pos_shift < -100000 || o.mtp_pos_shift > 100000)
                throw std::runtime_error("--mtp-pos-shift is a small offset, not a position");
        } else if (a == "--mtp-verify") {
            // Parses or throws, same rule as `--mtp-draft`: this names which side of an A/B the run is on.
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--mtp-verify takes 0 or 1, got '" + v + "'");
            }
            o.mtp_verify = std::atoi(v.c_str());
            o.mtp_verify_given = true;      // the flag outranks the environment, and it is the naming
        } else if (a == "--mtp-accept") {
            // Parses or throws. `0` is the shipping default and allocates nothing; `1` replaces the
            // generation loop's steps with accept rounds (see `engine/mtp_accept.h`).
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--mtp-accept takes 0 (off, the default) or 1 (k=1 speculative"
                                         " decoding on the draft block's proposals), got '" + v + "'");
            }
            o.mtp_accept = std::atoi(v.c_str());
        } else if (a == "--mtp-midsplit") {
            // Parses or throws, as the other M2 knobs. It is a witness knob and nothing else: it makes
            // the probe run its body twice per anchor and byte-compare the two.
            const std::string v = next();
            if (v != "0" && v != "1" && v != "2") {
                throw std::runtime_error("--mtp-midsplit takes 0 (off), 1 (split arm) or 2 (control"
                                         " arm: the second body runs UNSPLIT), got '" + v + "'");
            }
            o.mtp_midsplit = std::atoi(v.c_str());
        } else if (a == "--mtp-meter") {
            // Parses or throws, like every flag that names which side of a measurement a run is on.
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--mtp-meter takes 0 (off, the default) or 1 (print the round's "
                                         "per-phase milliseconds; see engine/mtp_meter.h)");
            }
            o.mtp_meter = std::atoi(v.c_str());
        } else if (a == "--mtp-route") {
            // Parses or throws: this names which side of the route A/B the run is, and a typo that silently
            // became `hand` would price the hand route twice and report the second price as the amortised
            // one — the exact null this repo has been bitten by before.
            o.mtp_route = engine::mtpverify::parse_route(next());
            o.mtp_route_given = true;
        } else if (a == "--verify-candidate") {
            o.verify_candidate = next();
            o.verify_candidate_given = true;
            // Validated at parse time, not at the first round: a typo that reached the loop would produce
            // a wrong candidate for a whole run before anything complained.
            if (o.verify_candidate != "sampled") {
                try {
                    (void)std::stoi(o.verify_candidate);
                } catch (const std::exception&) {
                    throw std::runtime_error("--verify-candidate takes an id or `sampled`, got '" +
                                             o.verify_candidate + "'");
                }
            }
        } else if (a == "--verify-guard") {
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--verify-guard takes 0 or 1, got '" + v + "'");
            }
            o.verify_guard = std::atoi(v.c_str());
            o.verify_guard_given = true;
        } else if (a == "--prefill-logits") {
            o.prefill_logits = engine::parse_prefill_logits(next());
        } else if (a == "--dump-attn-rows") {
            // A switch, not a value: it writes into `--out`, so there is no second path to keep in step
            // with the first. Whether `--out` was actually given is checked after the parse loop, because
            // flags may arrive in any order.
            o.dump_attn_rows = true;
        } else if (a == "--wave-rows") {
            o.wave_rows = std::atoi(next().c_str());
        } else if (a == "--mlp-map") {
            // Parses or throws: an unrecognised walk must not silently become the default one, or an
            // A/B row would price the default twice and call it a null.
            o.mlp_gate_map = engine::kernels::parse_mlp_gate_map(next());
            o.mlp_gate_map_given = true;
        } else if (a == "--gemm") {
            // Parses or throws, like `--mlp-map`: this names which side of an A/B the run is, so a typo
            // that became the default would price the default twice and report it as a null.
            o.wave_gemm = engine::kernels::parse_wave_gemm(next());
            o.wave_gemm_given = true;
        } else if (a == "--gemm-pins") {
            // Parses or throws, like `--gemm`: an unrecognised name must not silently mean "no pins".
            o.gemm_pins = next();
        } else if (a == "--gemm-heur") {
            // Parsed here, validated by the setter: the bounds belong to the code that owns the pool,
            // and a value outside them is an error at startup rather than a pool of a different size.
            o.gemm_heur = std::atoi(next().c_str());
        } else if (a == "--attn-tile") {
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--attn-tile takes 0 (the shipped attn_core) or 1 (the tiled "
                                         "wave core); got '" + v + "'. Refusing rather than guessing, "
                                         "because this flag names which side of an A/B the run is");
            }
            o.attn_tile = v == "1";
            o.attn_tile_given = true;
        } else if (a == "--attn-touch") {
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--attn-touch takes 0 (the shipped tile spelling, "
                                         "`attn_tile_wave`) or 1 (the slim spelling, "
                                         "`attn_slim_wave`); got '" + v + "'. Refusing rather than"
                                         " guessing, because this names which side of an A/B the run"
                                         " is, and a typo that became the default would price the"
                                         " default twice and report it as the slim form");
            }
            o.attn_touch = v == "1";
        } else if (a == "--attn-fattn") {
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--attn-fattn takes 0 (the shipped VALU tiled wave owns every "
                                         "row) or 1 (the WMMA wave owns the streaming rows); got '" + v +
                                         "'. Refusing rather than guessing, because this names which"
                                         " form produced the bits in this log");
            }
            o.attn_fattn = v == "1";
        } else if (a == "--attn-fattn-roll") {
            // Parses or throws, like `--gdn-geom`: the two sides of this A/B are explicitly NOT bit-identical,
            // so a run that silently took the default would report the shipped roll's family row as the second
            // accumulator's.
            o.attn_fattn_roll = engine::kernels::parse_fattn_roll(next());
            o.attn_fattn_roll_given = true;
        } else if (a == "--attn-fattn-qstage") {
            // Parses or throws, like `--attn-fattn-roll` beside it — and for a sharper reason: this knob's claim
            // is byte identity, so a silently-defaulted side would reproduce the shipped kernel exactly, look
            // perfectly green, and be priced as the un-stage's residency win.
            o.attn_fattn_qstage = engine::kernels::parse_fattn_qstage(next());
            o.attn_fattn_qstage_given = true;
        } else if (a == "--attn-fattn-pvf16") {
            // Parsed or thrown, 0|1 like `--attn-fattn` itself, and for the same reason as `--attn-fattn-qstage`
            // beside it: this names which of the eight WMMA spellings ran, and a value that silently fell back to
            // the default would price the F32 accumulator twice and report the second price as the F16 side's.
            const std::string v = next();
            if (v != "0" && v != "1") {
                throw std::runtime_error("--attn-fattn-pvf16 takes 0 (the shipped F32 P*V accumulator, 16 tiles"
                                         " of 8 F32 = 128 registers a lane — THE DEFAULT and HEAD's own machine"
                                         " code) or 1 (bet A10-F16: the F16 accumulator, 8 tiles of 16 halves ="
                                         " 64 registers for the same 256 output dims, two v_wmma_f16 issues per"
                                         " 32 dims; engine/attn_fattn_kernels.h). Got '" + v + "'. Refusing"
                                         " rather than guessing, because this names which accumulator produced"
                                         " the bits in this log and the register claim is read off its `pv=`");
            }
            o.attn_fattn_pvf16 = v == "1";
            o.attn_fattn_pvf16_given = true;
        } else if (a == "--attn-fattn-wide") {
            // Parsed or thrown, 0|1|2 like `--attn-fattn` and `--attn-fattn-pvf16` beside it, and for the same
            // reason: this names which kernel produced a wave's rows, and a value that silently fell back to the
            // default would price the shipping WMMA wave twice and report the second price as the wide block's.
            // Unlike the two above it it is not a SPELLING of the shipping form: at 1 it replaces the wave core
            // and takes the split/combine route away with it, so the two sides of this A/B differ in which
            // kernels ran at all, which is precisely what an echoed 0/1 cannot be allowed to hide. `2` is the
            // SAME core with llama's structure ported into it (A13), so it inherits all of that and adds a
            // byte-identity claim against `1`; it is a value and not a separate flag precisely so that a driver
            // forwarding every form flag cannot ask for two wave cores at once.
            const std::string v = next();
            if (v != "0" && v != "1" && v != "2") {
                throw std::runtime_error("--attn-fattn-wide takes 0 (the WMMA wave that shipped until A13:"
                                         " the 16-row block on the streaming rows, attn_slim_wave on the"
                                         " one-pass rows, attn_combine merging them)"
                                         " or 1 (bet A12: the wide block, 32 query positions x the 4 heads of"
                                         " one KV group per block, a 64-position stage, Q resident in F16 and"
                                         " an F16 P*V accumulator, owning every row of the body with no lane"
                                         " split and no combine; engine/attn_wide_kernels.h) or 2 (bet A13:"
                                         " that same wide block with llama.cpp's swizzled addressing, mid-round"
                                         " K prefetch, specialised tail round and scale folded into Q; owed"
                                         " byte-identical to 1; engine/attn_wide_llama_kernels.h — THE DEFAULT"
                                         " and HEAD's own machine code). Got '" + v +
                                         "'. Refusing rather than guessing, because this names which wave core"
                                         " produced the bits in this log and the gate asserts the echo");
            }
            o.attn_fattn_wide = std::stoi(v);
            if (o.attn_fattn_wide < 0 || o.attn_fattn_wide > 2) {
                // Unreachable through the test above, and stated anyway because `std::stoi` accepts things the
                // comparison above rejected as strings: the value that reaches `AttnLayerArgs::fattn_wide` must
                // be one of the three the launcher can route, and a wider integer would silently mean "not 2",
                // i.e. knob 1, which is the silently-the-default failure this whole parse exists to prevent.
                throw std::runtime_error("--attn-fattn-wide out of range: " + v);
            }
        } else if (a == "--attn-split-cap") {
            // Parsed AND validated here rather than clamped: a value that silently became the default would
            // price the uncapped run twice and report the second price as the cap's win (the standing reason
            // for parse-or-throw on every A/B flag in this file), and `1` is not a shallow cap but a different
            // route — one lane block on a row wider than the staging is the mis-normalisation §07.3 exists to
            // make unreachable, so the launcher's `split >= 2` precondition is stated as a refusal, not bent.
            const std::string v = next();
            int n = 0;
            bool ok = false;
            try {
                size_t used = 0;
                n = std::stoi(v, &used);
                ok = used == v.size();
            } catch (const std::exception&) {
                ok = false;
            }
            if (!ok) {
                throw std::runtime_error("--attn-split-cap takes a whole number of lane blocks (0 = no"
                                         " cap, the default; 2 or more = a cap); got '" + v +
                                         "'. Refusing rather than guessing, because this names which side"
                                         " of bet A9-L-c produced the bits in this log");
            }
            if (n == 1 || n < 0) {
                throw std::runtime_error("--attn-split-cap " + v + " is not a shallower cut: one lane block"
                                         " on a row wider than the staging (MAXSC=512) normalises a weight"
                                         " matrix inside a block that cannot see the whole row, which is the"
                                         " defect §07.3 exists to make unreachable. Use 0 for no cap, or 2"
                                         " or more");
            }
            o.attn_split_cap = n;
        } else if (a == "--gdn-scan") {
            o.gdn_scan = engine::kernels::parse_gdn_scan(next());
            o.gdn_scan_given = true;
        } else if (a == "--gdn-spell") {
            // Parses or throws, like `--gdn-scan`: this names which side of an A/B the run is.
            o.gdn_spell = engine::kernels::parse_gdn_rec_spell(next());
            o.gdn_spell_given = true;
        } else if (a == "--gdn-loop") {
            // Parsed or thrown, and the number is a SPAN, not a switch: like `--wave-rows` a silently
            // clamped or silently defaulted value would make the A/B compare the wrong two runs, and a
            // run whose log cannot say which launch shape produced its family row is not evidence
            // about the loop.
            const std::string v = next();
            int n = 0;
            try {
                size_t used = 0;
                n = std::stoi(v, &used);
                if (used != v.size()) throw std::invalid_argument(v);
            } catch (const std::exception&) {
                throw std::runtime_error("--gdn-loop takes a token count >= 1 (1 = the shipped"
                                         " one-launch-per-token form (the pre-D54 shape, kept as the"
                                         " A/B's other side); 512 = one launch per layer per body at"
                                         " the full scratch width, the default); got '" + v
                                         + "'");
            }
            if (n < 1) {
                throw std::runtime_error("--gdn-loop " + std::to_string(n) + " names no span: 1 is"
                                         " the per-token loop, 512 the default. Refusing rather"
                                         " than treating a non-positive span as either");
            }
            o.gdn_loop = n;
            o.gdn_loop_given = true;
        } else if (a == "--gdn-norm") {
            // Parses or throws, like `--gdn-spell` and `--gdn-loop`: this names which side of an A/B the
            // run is, and the two sides are claimed bit-identical, so a typo that silently became the
            // default would price one form twice and report the second price as the hoist's win.
            o.gdn_norm = engine::kernels::parse_gdn_norm(next());
            o.gdn_norm_given = true;
        } else if (a == "--gdn-geom") {
            // Parses or throws, for the same reason and with a sharper edge: the two sides of THIS A/B are
            // explicitly not bit-identical, so a run that silently took the default would report the
            // shipped walk's family row as the new geometry's.
            o.gdn_geom = engine::kernels::parse_gdn_geom(next());
            o.gdn_geom_given = true;
        } else if (a == "--gdn-chunk") {
            const std::string v = next();
            int n = 0;
            try {
                size_t used = 0;
                n = std::stoi(v, &used);
                if (used != v.size()) throw std::invalid_argument(v);
            } catch (const std::exception&) {
                throw std::runtime_error("--gdn-chunk takes a token count (64); got '" + v + "'");
            }
            if (const char* why = engine::kernels::gdn_chunk_refusal(n)) {
                throw std::runtime_error("--gdn-chunk " + v + " is refused: " + why);
            }
            o.gdn_chunk = n;
            o.gdn_chunk_given = true;
        } else if (a == "--proj-plan") {
            o.proj_plan = engine::kernels::parse_proj_plan(next());
            o.proj_plan_given = true;
        } else if (a == "--proj-shape") {
            o.proj_shape = engine::kernels::parse_proj_shape(next());
            o.proj_shape_given = true;
        } else if (a == "--dump-proj-plan") {
            o.dump_proj_plan = true;
        } else if (a == "--time-stages") {
            o.time_stages = true;
        } else if (a == "--time-waves") {
            o.time_waves = true;
        } else if (a == "--help" || a == "-h") {
            o.help = true;
        } else {
            throw std::runtime_error("unknown argument " + a);
        }
    }
    if (o.help) return o;   // --help answers before it complains about what is missing
    if (o.wave_rows < 1 || o.wave_rows > engine::kMaxWaveRows) {
        // Refuse rather than clamp: this flag is how a gate forces a particular launch shape, and a
        // silently clamped value would make the A/B compare the wrong two runs.
        throw std::runtime_error("--wave-rows takes 1.." +
                                 std::to_string(engine::kMaxWaveRows) + ", got " +
                                 std::to_string(o.wave_rows) + " (the scratch is sized for "
                                 "kMaxWaveRows rows; `--wave-rows 1` is the one-row-per-launch "
                                 "comparison run)");
    }
    if (!o.sizes_only && !o.session && !o.dump_proj_plan && o.prompt.empty()) {
        throw std::runtime_error(
            "decode_run needs --ids <comma-separated token ids> (the fixture's prefill_ids) or "
            "--ids-file PATH holding the same list (a prompt beyond ~21,800 ids cannot be an argument); "
            "--sizes-only and --session are the two modes that run without them");
    }
    if (!o.sizes_only && !o.session && o.steps < 0) {
        throw std::runtime_error("decode_run needs --steps >= 0");
    }
    if (o.gdn_spell_given && o.gdn_spell != engine::kernels::GdnRecSpell::kShip &&
        o.gdn_scan == engine::kernels::GdnScan::kCollapse) {
        // The re-spelled walks belong to the SEQUENTIAL kernel; `collapse` submits
        // `gdn_rc::rec_collapse_kernel` and never looks at the flag. A run that printed
        // `gdn_spell=unroll4` while running the collapse kernel would price the same kernel twice and
        // report the second price as the spelling's win, which is the failure this whole knob exists to
        // avoid, so it is refused rather than echoed with a caveat. Written as `!= kShip`, so every
        // non-shipping spelling — `unroll4` and `fused` alike — inherits the refusal without this
        // predicate having to be edited each time a spelling is added.
        throw std::runtime_error("--gdn-spell " + std::string(engine::kernels::gdn_rec_spell_name(
                                     o.gdn_spell)) +
                                 " selects the spelling of the sequential recurrence's state walks and"
                                 " is inert under --gdn-scan collapse, which launches a different"
                                 " kernel. Use --gdn-scan seq (the default) for this A/B");
    }
    if (o.gdn_loop_given && o.gdn_loop > 1 &&
        o.gdn_scan == engine::kernels::GdnScan::kCollapse) {
        // The same class of misreport as the pair above, one knob over: route A already submits ONE
        // launch per layer per body, so a `--gdn-loop` span named alongside it is inert, and a log
        // echoing `gdn-loop=512` while running route A would price one kernel twice and call the
        // second price the loop's win. Refused, not echoed with a caveat. Asked on `_given` and NOT on
        // the value — unlike the spelling pair below — because route A never reaches `gdn_rec_span` at
        // all, so a bare `--gdn-scan collapse` run is carrying a span that is inert rather than wrong,
        // and refusing it would refuse a command that never asked for a span. Since the default moved
        // to 512 that run's echoed `gdn_loop=` must be read as inert-by-route, which is what this note
        // is for; the NUMERICS line also echoes `gdn_scan=collapse`, and the two together say it.
        throw std::runtime_error("--gdn-loop " + std::to_string(o.gdn_loop) +
                                 " sets the span of the SEQUENTIAL recurrence's launch and is inert"
                                 " under --gdn-scan collapse, which already submits one launch per"
                                 " layer per body. Use --gdn-scan seq (the default) for this A/B");
    }
    if (o.gdn_loop > 1 && o.gdn_spell != engine::kernels::GdnRecSpell::kFused) {
        // The same class of pair, and this one is refused at parse rather than at the launch even
        // though `gdn_rec_span` also refuses it structurally: a span walks a LOOPED body, only the
        // `fused` body has one, and a run that loaded 1.55 GB of weights before finding that out
        // wastes a device window to say something parse_args already knows (`kernels_gdn.cpp`).
        // The test is on the EFFECTIVE span, not on `_given`, and that matters now that 512 is the
        // default: `--gdn-spell unroll4` alone used to be legal (span defaulted to 1) and now pairs a
        // non-looped spelling with a looped default, so honouring `_given` here would move the refusal
        // into the kernel after the model load and would print the remedy as a stack trace. Hence the
        // two openings below — one for a span the caller named, one for the default doing it.
        throw std::runtime_error(std::string(o.gdn_loop_given
                                                 ? ("--gdn-loop " + std::to_string(o.gdn_loop))
                                                 : "the default --gdn-loop 512") +
                                 " submits the looped body (engine/gdn_rec_loop_kernels.h) and"
                                 " only the `fused` spelling has one; --gdn-spell " +
                                 engine::kernels::gdn_rec_spell_name(o.gdn_spell) +
                                 " has no looped transcription and no bit-identity admission for a"
                                 " multi-token span. Use --gdn-spell fused (the default) or"
                                 " --gdn-loop 1 (the pre-D54 one-launch-per-token form)");
    }
    if (o.gdn_norm_given && o.gdn_norm == engine::kernels::GdnNorm::kHoist &&
        o.gdn_scan == engine::kernels::GdnScan::kCollapse) {
        // The third instance of the same misreport, for the third time refused rather than caveated:
        // route A submits `gdn_rc::rec_collapse_kernel`, which computes its own norms inside its own
        // body and never looks at this flag. A log echoing `gdn_norm=hoist` while running route A would
        // price one kernel twice and call the second price the hoist's win.
        throw std::runtime_error("--gdn-norm hoist moves the norms out of the SEQUENTIAL looped body"
                                 " and is inert under --gdn-scan collapse, which launches a different"
                                 " kernel that keeps its own prologue. Use --gdn-scan seq (the"
                                 " default) for this A/B");
    }
    if (o.gdn_norm == engine::kernels::GdnNorm::kHoist && o.gdn_loop <= 1) {
        // Asked on the EFFECTIVE span, like the spell/loop pair above: `--gdn-loop 1` runs the per-token
        // kernels, whose prologue is what this flag moves, so there is nothing for the hoist to hoist.
        // Refusing at parse is cheaper than loading 1.55 GB of weights to say it, and it is the same
        // claim `--gdn-loop`'s own help makes about a non-looped spelling.
        throw std::runtime_error("--gdn-norm hoist moves the norm prologue out of the LOOPED body"
                                 " (engine/gdn_rec_hoist_kernels.h) and --gdn-loop " +
                                 std::to_string(o.gdn_loop) + " submits the per-token kernels, which"
                                 " keep their own prologue unchanged. Use --gdn-loop 512 (the default)"
                                 " or --gdn-norm inline (the shipped form)");
    }
    // A default must not manufacture a refusal. `reg16` is the shipping geometry now, but `--gdn-norm hoist`
    // — a knob a caller may legitimately pass, because bet G1's form lives there — has no reg16 body: reg16's
    // norms are registers, which is the whole point of the map. When the geometry was NEVER ASKED FOR and that
    // knob WAS, the request is for a body only the pre-G4 map has, so the map follows the request and the echo
    // says `block`: silently, but not falsely, which is the precedent the `--gdn-scan collapse` refusal above
    // already sets ("that refusal now fires only when the flag was actually given, because a default of
    // `unroll4` must not turn a bare `--gdn-scan collapse` into an error"). Name both and the explicit refusal
    // below still fires — an asked-for pair that contradicts itself is a mistake to report, not a default to
    // route around. Verified on device: default echoes `reg16`; `--gdn-loop 1` echoes
    // `na-loop1-per-token-bodies`; `--gdn-norm hoist --gdn-loop 512` echoes `block` and runs G1's form;
    // `--gdn-geom reg16 --gdn-norm hoist` refuses naming registers. (`--gdn-spell` needs no clause here:
    // `--gdn-spell ship|unroll4` at `--gdn-loop > 1` is already refused by the guard above, geometry-free,
    // because only the fused spelling has a looped body at all — checked on device, and a clause for it here
    // would have been a branch that cannot be reached.)
    if (!o.gdn_geom_given && o.gdn_norm_given && o.gdn_norm == engine::kernels::GdnNorm::kHoist) {
        o.gdn_geom = engine::kernels::GdnGeom::kBlock;
    }
    if (o.gdn_geom_given && o.gdn_geom != engine::kernels::GdnGeom::kBlock &&
        o.gdn_scan == engine::kernels::GdnScan::kCollapse) {
        // The fourth instance of the same misreport. Route A submits `gdn_rc::rec_collapse_kernel`, which
        // keeps its own LDS-resident state and its own prologue and never sees this flag; echoing
        // `gdn_geom=wf4` over a route-A run would price one kernel twice and call the second price the
        // geometry's win — the exact failure BET.md §10 is funded to avoid.
        throw std::runtime_error("--gdn-geom " + std::string(engine::kernels::gdn_geom_name(o.gdn_geom)) +
                                 " re-spells the SEQUENTIAL looped body and is inert under --gdn-scan"
                                 " collapse, which launches a different kernel. Use --gdn-scan seq"
                                 " (the default) for this A/B");
    }
    if (o.gdn_geom_given && o.gdn_geom != engine::kernels::GdnGeom::kBlock && o.gdn_loop <= 1) {
        // `--gdn-loop 1` runs the per-token kernels, and neither re-spelled geometry is a per-token body.
        // Refused at parse, like the `--gdn-norm hoist` pair above, rather than loading 1.55 GB of weights
        // to say it.
        //
        // Keyed on `gdn_geom_given` deliberately, because `--gdn-geom` now DEFAULTS to `reg16` (bet G4,
        // bench §44): a guard written as `geom != block` would turn the shipping default into a refusal at
        // `--gdn-loop 1`, which is a legal and pinned route (the decode bits were pinned through the
        // per-token bodies, and `resolve_gdn_loop` routes a one-row body there whatever the geometry says).
        // An EXPLICIT `--gdn-geom reg16 --gdn-loop 1` is still refused, for the standing reason: a flag that
        // names a form and gets another one prices one kernel twice and reports the second price as a win.
        // An unasked default simply runs the per-token bodies, which is what it ran before this flag existed.
        throw std::runtime_error("--gdn-geom " + std::string(engine::kernels::gdn_geom_name(o.gdn_geom)) +
                                 " re-spells the LOOPED body (engine/gdn_rec_wf_kernels.h for wf4/wf2,"
                                 " engine/gdn_rec_reg16_kernels.h for reg16) and --gdn-loop " +
                                 std::to_string(o.gdn_loop) + " submits the per-token"
                                 " kernels unchanged. Use --gdn-loop 512 (the default) or --gdn-geom"
                                 " block (the shipped form)");
    }
    if (o.gdn_geom != engine::kernels::GdnGeom::kBlock &&
        o.gdn_norm == engine::kernels::GdnNorm::kHoist) {
        // The one refusal that is a fact about the arithmetic rather than about inertness: both
        // re-spelled geometries have no norm prologue to hoist (their norms are registers, which is the
        // whole point of both maps), and the launch site picks the geometry first. Two named flags, one
        // kernel: refused.
        throw std::runtime_error("--gdn-geom " + std::string(engine::kernels::gdn_geom_name(o.gdn_geom)) +
                                 " computes the k/q row norms in REGISTERS by warp butterfly, so"
                                 " --gdn-norm hoist has nothing left to hoist. Pick one: --gdn-geom block"
                                 " --gdn-norm hoist (bet G1's form) or --gdn-geom wf4/reg16 with the"
                                 " default --gdn-norm inline (bets G2 and G4)");
    }
    if (o.attn_fattn_roll_given && !o.attn_fattn) {
        // The `--gdn-geom`-under-`--gdn-scan collapse` refusal, one layer over: naming a spelling of a form
        // that does not run would echo a kernel that did not exist in this run and price one launch twice.
        // Keyed on `attn_fattn_roll_given`, NOT on `roll != single`, because `single` IS the default and a
        // default must not manufacture a refusal — a bare `--attn-fattn 0` is the shipped VALU wave path and
        // has been legal since that flag existed. An asked-for pair that contradicts itself is still a
        // mistake to report rather than a default to route around.
        throw std::runtime_error("--attn-fattn-roll " + std::string(engine::kernels::fattn_roll_name(o.attn_fattn_roll)) +
                                 " re-spells the WMMA wave's score roll and --attn-fattn 0 launches no WMMA"
                                 " wave at all, so there is nothing here for it to spell. Use"
                                 " --attn-fattn 1 (the default) with this flag, or drop it for the shipped"
                                 " VALU wave");
    }
    if (o.attn_fattn_qstage_given && !o.attn_fattn) {
        // The refusal above, one knob in: naming a spelling of a form that does not run would echo a kernel that
        // did not exist in this run and price one launch twice. Keyed on `attn_fattn_qstage_given`, NOT on
        // `qstage != stage`, because `stage` IS the default and a default must not manufacture a refusal.
        throw std::runtime_error("--attn-fattn-qstage " +
                                 std::string(engine::kernels::fattn_qstage_name(o.attn_fattn_qstage)) +
                                 " re-spells where the WMMA wave keeps Q and --attn-fattn 0 launches no WMMA"
                                 " wave at all, so there is nothing here for it to spell. Use --attn-fattn 1"
                                 " (the default) with this flag, or drop it for the shipped VALU wave");
    }
    if (o.attn_fattn_pvf16_given && !o.attn_fattn) {
        // The refusal above, one more knob in, and it is the same argument: naming an accumulator width for a
        // form that does not run would echo a kernel that did not exist in this run and price one launch twice.
        // Keyed on `attn_fattn_pvf16_given`, NOT on the value, because 0 IS the default and a default must not
        // manufacture a refusal.
        throw std::runtime_error("--attn-fattn-pvf16 " + std::to_string(o.attn_fattn_pvf16) +
                                 " re-spells the WMMA wave's P*V accumulator and --attn-fattn 0 launches no WMMA"
                                 " wave at all, so there is nothing here for it to spell. Use --attn-fattn 1"
                                 " (the default) with this flag, or drop it for the shipped VALU wave");
    }
    if (o.dump_attn_rows && o.out_dir.empty()) {
        // Checked here rather than at the flag, because `--out` may come before or after it. A dump that
        // had nowhere to go would print a line and no bytes, which is the worst kind of clean result.
        throw std::runtime_error("--dump-attn-rows needs --out DIR: the output tiles go with the other "
                                 "dumps, and a dump that wrote no file must not look like a pass");
    }
    if (o.attn_fattn_wide && !o.attn_fattn) {
        // The wide block is a spelling of the WMMA family (`engine/attn_wide_kernels.h`), and `--attn-fattn 0`
        // has just said the VALU form owns this wave's rows. The other three sub-knobs refuse because naming a
        // spelling of a form that does not run would echo a kernel that did not exist; this one refuses for the
        // stronger reason that it would OVERRIDE a routing decision rather than spell within it — `attn_slim_wave`
        // and `attn_combine` would be skipped by a knob whose name says it only changes the WMMA form. Refusing
        // is the only answer that cannot silently move the engine's routing. Keyed on the VALUE, not on
        // `_given`, because unlike those three this knob has no inert non-default value to excuse: 1 always
        // means the wide block ran, and `engine/gate.py` forwards the flag on every run, so a `given`-keyed
        // refusal would reject `--attn-fattn 0 --attn-fattn-wide 0`, which asks for nothing.
        throw std::runtime_error("--attn-fattn-wide 1 or 2 asks for the wide WMMA wave block and --attn-fattn 0"
                                 " launches no WMMA wave at all: the wide block replaces that wave core"
                                 " outright, taking attn_slim_wave and attn_combine off the wave with it, so"
                                 " there is nothing here for it to spell. Use --attn-fattn 1 (the default)"
                                 " with this flag, or --attn-fattn-wide 0 for the shipping wave");
    }
    if (o.proj_shape_given && o.proj_plan == engine::kernels::ProjPlan::kPacked &&
        o.proj_shape != engine::kernels::ProjShape::kUnroll8) {
        // The grouped kernel is the only thing that can issue a packed plan — a segment list is a kernel
        // argument (`engine/proj_group_kernels.h`), so `--proj-plan packed` on the shipped kernel would
        // either be ignored or would need a second packed form of a kernel this route exists to replace.
        // Either way the log would echo a plan that did not run and price the split form twice, which is
        // the misreport every other refusal above exists to prevent. Keyed on the SHAPE being given:
        // since the flip the default plan is `packed`, so it is a bare `--proj-shape shipped` that asks
        // for the unbuildable pair, and a run that never mentions either flag still gets the shipped
        // default pair. `--proj-plan split --proj-shape shipped` is the pre-D1 form and stays legal.
        throw std::runtime_error("--proj-shape " +
                                 std::string(engine::kernels::proj_shape_name(o.proj_shape)) +
                                 " submits one launch per projection whatever --proj-plan says, and"
                                 " --proj-plan packed groups a layer's input projections into ONE launch;"
                                 " that grouping is a feature of the grouped kernel selected by"
                                 " --proj-shape unroll8 (the default). Use --proj-plan split with " +
                                 std::string(engine::kernels::proj_shape_name(o.proj_shape)) +
                                 " for the pre-D1 pair, or drop --proj-shape for the shipped plan");
    }
    if (o.mtp_draft && o.time_waves) {
        // The WAVE clock segments a prefill pass, and the draft does not run inside one: its rounds follow
        // generation steps, and a step clock's `note()` outside a measured unit is ignored and counted
        // (`engine/stage_clock.h`). So there is nothing for the `mtp` family to bill on a wave, and a wave
        // report that printed no `mtp` row while the run drafted would read as an instrument gap. The
        // family exists, is populated, and belongs to the STEP clock; the refusal stays for the unit that
        // cannot use it, with the reason corrected to say so.
        throw std::runtime_error("--mtp-draft 1 with --time-waves: the wave clock prices a prefill pass,"
                                 " and a draft round follows a generation step, so no draft submission"
                                 " falls inside the unit it measures and the `mtp` family would be"
                                 " declared-and-empty. Use --time-stages (with --mtp-sched every, which"
                                 " is the uniform schedule the clock can bill); --mtp-draft prints its"
                                 " own MTPDRAFT census line either way.");
    }
    if (o.mtp_draft && o.time_stages &&
        engine::mtpdraft::parse_schedule(o.mtp_sched) != engine::mtpdraft::Schedule::kEvery) {
        // The `mtp` family IS declared now, and one real constraint came with declaring it: the clock's
        // per-step division is exact (`Clock::describe()`: "uniform by construction … so an exact integer
        // division is honest here"), and a `loop` round follows a step only when the previous round did
        // not advance two positions, i.e. ~1 round per `1+a` steps. No integer per-step bill exists for
        // that schedule, and billing `rounds x bill / steps` by truncation would print a `plan_per_step`
        // that is neither the plan nor the observation. `every` makes exactly one round per step — and so
        // does the accept loop, which drafts once per round whatever the schedule is printed under — so
        // `every` is what the clock bills, and the default stays `loop` for every correctness pass.
        throw std::runtime_error("--time-stages bills the `mtp` family one draft round per step, which"
                                 " is the uniform shape the clock's per-step division requires;"
                                 " --mtp-sched " + o.mtp_sched +
                                 " makes ~1 round per 1+a steps and has no integer per-step bill. Name"
                                 " --mtp-sched every for a timed run (it is also the accept loop's own"
                                 " shape), or drop --time-stages: the correctness pass prints its own"
                                 " MTPDRAFT census line under either schedule.");
    }
    if (o.mtp_sched_given && !o.mtp_draft) {
        // Naming a schedule for a path that does not run. Keyed on `given` deliberately: `loop` is the
        // default, so a bare run must not be refused, exactly the `--gdn-norm hoist` / `--gdn-geom`
        // precedent about defaults not manufacturing refusals.
        throw std::runtime_error("--mtp-sched " + o.mtp_sched +
                                 " schedules rounds for a draft that is not running; name --mtp-draft 1"
                                 " with it, or drop it");
    }
    if (o.mtp_dense && !o.mtp_accept) {
        // The dense-lane append happens on an ACCEPT, and an accept only exists in the accept loop. The
        // side pass settles a proposal against a later step but never consumes two positions in one
        // round, so there is no skipped position to stage and the flag would be inert — which is a
        // measurement of nothing, not a null result.
        throw std::runtime_error("--mtp-dense stages the lane for the position an ACCEPTED round skipped,"
                                 " and only --mtp-accept 1 runs rounds; name --mtp-accept 1 with it, or"
                                 " drop it (the side pass never skips a position, so the flag cannot"
                                 " have an effect there)");
    }
    if ((o.mtp_dense || o.mtp_prompt_lanes > 0) && o.time_stages) {
        // Both knobs submit block work that is not part of the declared per-step `mtp` bill: a dense lane
        // rides on an accept (so it is not 1-per-step) and prompt lanes ride once on the prompt phase
        // (so they are not per-step at all). The stage clock's division is exact, and the same argument
        // that refuses `--mtp-sched loop` under `--time-stages` refuses these: there is no integer
        // per-step bill, and a truncated one prints a plan that is neither the plan nor the observation.
        throw std::runtime_error("--time-stages bills the `mtp` family one draft round per step; " +
                                 std::string(o.mtp_dense ? "--mtp-dense" : "") +
                                 (o.mtp_dense && o.mtp_prompt_lanes > 0 ? " and " : "") +
                                 (o.mtp_prompt_lanes > 0 ? "--mtp-prompt-lanes" : "") +
                                 " adds block submissions that are not one-per-step (a dense lane rides"
                                 " on an accept, prompt lanes ride on the prompt phase), so no integer"
                                 " per-step bill exists. Drop --time-stages: these knobs are judged on"
                                 " accept rate, and a worker device pass takes no timing anyway.");
    }
    if (o.mtp_dense && o.mtp_meter) {
        // The meter's phase set has no name for a dense-lane append, so its cost would land in
        // `overhead` — which `METER-PLAN.md` §9 names as "a phase waiting to be named, not a band to
        // widen". Refusing is the honest answer until someone gives the phase a boundary and a bias.
        throw std::runtime_error("--mtp-dense with --mtp-meter: the lane append is round work that no "
                                 "metered phase owns, so it would be billed to `overhead`. Name one of "
                                 "the two");
    }
    if (o.mtp_prompt_lanes_given && !o.mtp_draft) {
        throw std::runtime_error("--mtp-prompt-lanes seeds a draft cache that is not allocated; name"
                                 " --mtp-draft 1 with it, or drop it");
    }
    if (o.mtp_prompt_lanes > 0 && o.session) {
        // The seeding retains ONE prompt wave and its last position is what makes the seeded lanes
        // contiguous with the first generated lane. A session prefills one `feed` line at a time
        // (`session_loop`), so there is no such wave and no such position, and the flag would sit there
        // doing nothing while a reader assumed it had taken — the inert-flag-as-a-measurement case
        // `--mtp-dense` without `--mtp-accept` is refused for, and `--mtp-accept` already refuses
        // `--session` for the same family of reason.
        throw std::runtime_error("--mtp-prompt-lanes seeds the draft cache from one prompt wave; --session"
                                 " prefills one `feed` line at a time and has no single prompt wave to"
                                 " retain (the flag could only be inert). Run it as one prompt with"
                                 " --ids-file instead");
    }
    // ---- the multi-row draft ladder's refusals (`--mtp-draft-k`) ----------------------------------
    // Each refusal is a combination in which the extra rows have no object, not a limitation of the rows.
    // All of them are keyed on `given` except the k>1 ones, so a bare run is never refused for a flag it
    // never named (the `--mtp-sched` precedent).
    if (o.mtp_draft_k_given && !o.mtp_draft) {
        throw std::runtime_error("--mtp-draft-k runs extra rows of the draft block, and with --mtp-draft 0"
                                 " there is no draft block to run; name --mtp-draft 1 with it, or drop it");
    }
    if (o.mtp_draft_k_given && o.session) {
        // The ladder's targets are the run's OWN later steps: the proposal made at anchor q is adjudicated
        // by the step at q+1, the second one by q+2, and so on. A session re-anchors the position map at
        // every `feed` and hands the `gen` budget to its caller, so a record opened in one turn would be
        // settled by a token of the next turn and the tail of every ladder would be silently adjudicated
        // against a different sequence. Same family as `--mtp-prompt-lanes` above: inert is bad, but
        // wrong-by-construction is worse, so the flag is refused rather than half-supported.
        throw std::runtime_error("--mtp-draft-k adjudicates each draft row against the run's own later"
                                 " steps; --session re-anchors the position map at every `feed`, so a row"
                                 " from one turn would be settled by a token of the next. Run it as one"
                                 " prompt with --ids-file --steps N instead");
    }
    if (o.mtp_draft_k_given && !o.feed.empty()) {
        // Under `--replay` the per-step tokens are the CALLER's, so `tN` would be a byte of the pin rather
        // than the model's own continuation and the ladder would measure agreement with a transcript. That
        // is a different quantity (and `--mtp-accept` refuses `--feed` for the neighbouring reason); say
        // which one this flag measures instead of letting the same name mean both.
        throw std::runtime_error("--mtp-draft-k's targets are the run's own greedy continuation; --replay"
                                 " feeds the caller's tokens, so the ladder would score the draft against a"
                                 " transcript rather than against this model. Run it free (no --replay)");
    }
    if (o.mtp_draft_k > 1 && o.mtp_accept) {
        // An accept round is a `kRows` = 2 body: one candidate row beside the anchor. A second and third
        // proposal has no row in the body and no verdict, so the extra rounds would be paid for and never
        // adjudicated — priced, not measured. Building k=2 speculation means widening that body, which is
        // the funded experiment this probe exists to price rather than pre-empt.
        throw std::runtime_error("--mtp-accept consumes ONE candidate per round (its body is 2 rows), so a"
                                 " second draft proposal would be drafted, paid for, and never verified;"
                                 " --mtp-draft-k measures those proposals on the side pass. Name"
                                 " --mtp-accept 0 with it (or drop --mtp-accept)");
    }
    if (o.mtp_draft_k > 1 && o.time_stages) {
        // The stage clock's `mtp` family is declared as one round per step precisely so its per-step
        // division is exact (`engine/stage_clock.h`, and the same argument `--mtp-sched loop` already
        // gives). k rows per anchor has no integer per-step bill, and a truncated one prints a plan that is
        // neither the plan nor the observation. This is a rate probe; it is judged on the ladder.
        throw std::runtime_error("--time-stages bills the `mtp` family one draft round per step;"
                                 " --mtp-draft-k " + std::to_string(o.mtp_draft_k) + " runs " +
                                 std::to_string(o.mtp_draft_k) + " per drafted position and has no integer"
                                 " per-step bill. Drop --time-stages: this knob is judged on its ladder,"
                                 " and a worker device pass takes no timing anyway");
    }
    if (o.verify_candidate_given && !o.mtp_verify) {
        throw std::runtime_error("--verify-candidate feeds row 1 of a verify body that is not running;"
                                 " name --mtp-verify 1 with it, or drop it");
    }
    if (o.mtp_midsplit && !o.mtp_verify) {
        // The split is an admission, and the thing it admits is a 2-row body. Arming it with no probe
        // would arm a checkpoint nobody reads, so it is refused rather than silently dropped.
        throw std::runtime_error("--mtp-midsplit witnesses the batched verify body, which is not"
                                 " running; name --mtp-verify 1 with it, or drop it");
    }
    if (o.mtp_midsplit && o.gdn_loop != 1) {
        // The checkpoint forces the recurrence to walk one token-step per launch, because that is the
        // only form in which "the state after row 0" exists at an address. A reference arm left at the
        // shipped `--gdn-loop 512` resolves to a span of 2 at n = 2 and therefore runs the LOOPED
        // recurrence, so the byte-compare would be split-span-against-looped — a different recurrence on
        // the two sides, and it reads as a wholesale logits disagreement (measured: 248 320 and 496 640
        // differing elements, i.e. whole rows, at anchors 18-20, with `--gdn-loop 1` reading 0). Refuse
        // the pair instead of printing a red result that is really a form mismatch, and say which flag
        // makes the two sides the same computation (`ACCEPT-BUILD.md` §4c).
        throw std::runtime_error("--mtp-midsplit pins the reference arm to --gdn-loop 1 (the split arms"
                                 " a one-token-step recurrence, so any other span compares two different"
                                 " recurrences); run it with --gdn-loop 1");
    }
    if (o.verify_guard_given && !o.mtp_verify) {
        // Keyed on `given`, the `--mtp-sched` precedent: the guard's default is 0, so a plain run must not
        // be refused for a flag it never named — but naming it alongside no body is a witness pointed at
        // nothing, and that is worth saying out loud.
        throw std::runtime_error("--verify-guard witnesses a verify body that is not running; name"
                                 " --mtp-verify 1 with it, or drop it");
    }
    if (o.mtp_verify && (o.session || o.sizes_only)) {
        // The verify probe prices one body per generation step against the state that step built, and it
        // reports through the run's own step lines and `--out` dumps. A session's `gen` budget is the
        // caller's and its state is re-anchored at every `feed`, and --sizes-only runs no steps at all —
        // so neither has the object this flag measures. Refusing beats printing nothing.
        throw std::runtime_error("--mtp-verify (or QWEN_MTP_VERIFY=1) probes a run's generation steps;"
                                 " --session has no such list and --sizes-only prefills nothing");
    }
    // ---- the round meter's refusals (`engine/mtp_meter.h`) -----------------------------------------
    // Placed ABOVE the `--mtp-verify`/`--mtp-accept` refusals of these same flags, deliberately: both of
    // those already refuse `--time-stages`, so a meter refusal written below them could never be reached.
    // A refusal that cannot fire is the failure shape this tree keeps recording, and the specific reason a
    // round meter and a step clock collide is worth printing rather than folding into the older message.
    if (o.mtp_meter && !o.mtp_verify && !o.mtp_accept) {
        // The meter has no object otherwise. A plain one-row step is `--time-stages`' subject, and metering
        // it here would produce a second, differently-serialized answer to a question that clock answers.
        throw std::runtime_error("--mtp-meter 1 meters an MTP round and there is none: name --mtp-verify 1"
                                 " (the read-only side pass) or --mtp-draft 1 --mtp-accept 1 (the accept"
                                 " loop) with it. A plain one-row step is --time-stages' subject, not"
                                 " this flag's");
    }
    if (o.mtp_meter && (o.time_stages || o.time_waves)) {
        // The stage clock prices a STEP's launch plan by draining the queue at §07 family boundaries; this
        // meter prices a ROUND by draining it at phase boundaries. Armed together they drain each other's
        // subject and neither report describes a shape the engine runs — a round is not a §07 family, which
        // is what the two refusals below already say for the side pass and the accept loop.
        throw std::runtime_error("--mtp-meter and --time-stages/--time-waves cannot share a run: the"
                                 " stage clock drains the queue at §07 family boundaries to price a"
                                 " STEP, this meter drains it at phase boundaries to price a ROUND, and"
                                 " armed together each serializes the other's subject. Run them as two"
                                 " passes (see prototype/mtp_accept/METER-PLAN.md)");
    }
    if (o.mtp_verify && (o.time_stages || o.time_waves)) {
        // Same rule the draft states above: the verify body's submissions belong to no family of §07's
        // launch bill, and the stage clock would charge them to whatever segment preceded them.
        throw std::runtime_error("--mtp-verify and --time-stages/--time-waves cannot share a run: the"
                                 " verify body is not a family of §07's bill, so the clock would misbill"
                                 " it. Run the correctness pass without the timing flags; the verify"
                                 " body prints its own VERIFY2 census line and its cost is taken as a"
                                 " paired wall-clock A/B (see prototype/mtp_verify/ARMS.md)");
    }
    if (o.mtp_verify && o.wave_rows < engine::mtpverify::kRows) {
        throw std::runtime_error("--mtp-verify needs a scratch at least " +
                                 std::to_string(engine::mtpverify::kRows) + " rows tall; this run's"
                                 " --wave-rows is " + std::to_string(o.wave_rows));
    }
    // ---- the accept loop's refusals (`engine/mtp_accept.h`) --------------------------------------
    // Each of these is a combination whose object the round does not own, not a limitation of the round.
    if (o.mtp_accept && !o.mtp_draft) {
        throw std::runtime_error("--mtp-accept verifies a proposal; with --mtp-draft 0 there is no draft"
                                 " block to propose one. Name --mtp-draft 1 with it (the candidate"
                                 " source is the §09 block, never a guess)");
    }
    if (o.mtp_accept && o.gdn_loop != 1) {
        // Same PIN `--mtp-midsplit` states above, for the same reason: a round's rollback restores the
        // mid-round checkpoint, and that address only exists when the recurrence walks one token-step per
        // launch. A round at any other span would restore a state the body never wrote — which is a wrong
        // token stream, not a red comparison, so it is refused at the flag rather than caught later.
        throw std::runtime_error("--mtp-accept pins --gdn-loop 1 (the mid-round checkpoint it restores"
                                 " from is a one-token-step recurrence's end state); run it with"
                                 " --gdn-loop 1");
    }
    if (o.mtp_accept && (o.session || o.sizes_only)) {
        throw std::runtime_error("--mtp-accept replaces a run's generation steps with rounds; --session"
                                 " hands the token budget to its caller and --sizes-only runs no steps");
    }
    // The boundary this refusal also draws for `DECWALL`: the session REPL prints NO `DECWALL` line, because
    // a session has no run end — `gen` budgets come from the caller and the state is re-anchored at every
    // `feed`, so the honest object there is per-`gen` and belongs to the session protocol, not to a
    // per-run window. Stated here because "the engine prints a decode wall per run" is otherwise true with
    // one silent exception, and a missing line is the worst way for a check to be out of scope.
    if (o.mtp_accept && (o.time_stages || o.time_waves)) {
        // The stage clock's plan is a per-STEP bill (§07's 224 launches) and its replay replays steps.
        // A round is one or two tokens and adds the checkpoint's own submissions, so there is no step
        // record for it to replay and no §07 family to bill it to. The cost is taken as a paired
        // wall-clock A/B by whoever owns timings, not by this flag.
        throw std::runtime_error("--mtp-accept and --time-stages/--time-waves cannot share a run: a round"
                                 " is not a step, so the clock has no per-step plan to replay or bill"
                                 " (the round prints its own MTPSTAT cost instead)");
    }
    if (o.mtp_accept && !o.feed.empty()) {
        // Teacher forcing feeds a token per step. A round consumes one or two, and its reject path emits
        // the token the caller did NOT feed at that index, so a replay under this flag has no defined
        // alignment. The accept path is judged by its own token stream against a free flag-off run.
        throw std::runtime_error("--mtp-accept cannot be combined with --replay/--feed: a round emits 1"
                                 " or 2 tokens, so the caller's per-step token list has no alignment"
                                 " (judge the accept run as a free run)");
    }
    if (o.mtp_accept && o.mtp_verify) {
        // Both arm the same scratch and the same 2-row body; the accept round IS the verify body plus a
        // verdict and a rollback. Running the probe as well would price a third body per round and make
        // the round's own printed cost wrong. `--mtp-midsplit` is the witness for the body's numerics.
        throw std::runtime_error("--mtp-accept already runs the 2-row body as the round itself; the"
                                 " read-only --mtp-verify side pass is not also armed (use"
                                 " --mtp-midsplit for the body's numerics witness)");
    }
    if (o.mtp_accept && o.wave_rows < engine::mtpaccept::kRows) {
        throw std::runtime_error("--mtp-accept needs a scratch at least " +
                                 std::to_string(engine::mtpaccept::kRows) + " rows tall; this run's"
                                 " --wave-rows is " + std::to_string(o.wave_rows));
    }
    // ---- the route switch's refusals (`engine/mtp_verify.h`'s Route) -------------------------------
    if (o.mtp_route_given && !o.mtp_verify && !o.mtp_accept) {
        // The `--mtp-sched` precedent, keyed on `given` so a bare run is never refused: `--mtp-route`
        // chooses the route of a body, and with no body running there is nothing for it to choose. Saying so
        // beats echoing a knob that did nothing.
        throw std::runtime_error("--mtp-route chooses which route a verify/accept body runs; with"
                                 " --mtp-verify 0 and --mtp-accept 0 there is no such body. Name one of"
                                 " them with it, or drop it");
    }
    if (o.mtp_route == engine::mtpverify::Route::kAmortised &&
        o.gdn_scan != engine::kernels::GdnScan::kSeq && (o.mtp_accept || o.mtp_midsplit)) {
        // The route switch moves the route; it does not move the mid-round CHECKPOINT's requirement, which is
        // that the recurrence walk one token-step per launch so "the state after row 0" exists at an address
        // a restore can read — the same reason `--mtp-accept` pins `--gdn-loop 1` above. An amortised round on
        // `collapse` would restore a state the body never wrote, which is a wrong token stream rather than a
        // red comparison, so it is refused at the flag instead of being served a silently re-forced seq scan.
        // `seq` is the default, so the amortised route still moves the GEMM, which is where the cost is.
        throw std::runtime_error("--mtp-route amortised with --gdn-scan " +
                                 std::string(engine::kernels::gdn_scan_name(o.gdn_scan)) +
                                 " cannot also arm a mid-round checkpoint (--mtp-accept / --mtp-midsplit):"
                                 " that checkpoint is a one-token-step sequential recurrence's end state."
                                 " Drop --gdn-scan (seq is the default) to price the amortised route, or"
                                 " keep the scan and use --mtp-route hand");
    }
    if (o.state_fp && (o.session || o.sizes_only)) {
        // A session has no single end condition to fingerprint — `feed` and `gen` boundaries are the
        // caller's, and a resident process cannot know which one a gate wants — and --sizes-only
        // allocates no state at all. Refusing is the honest answer; printing nothing would be the
        // quiet one.
        throw std::runtime_error("--state-fp fingerprints a run's end conditions; --session and"
                                 " --sizes-only have none");
    }
    if (o.session && o.prefill_logits != engine::PrefillLogits::kLast) {
        // Same reasoning, same shape of refusal: a session's `feed` wave has no caller that could do
        // anything with the discarded rows — nothing is dumped and no logits leave the process — so
        // `--prefill-logits all` there would buy T vocabulary D2Hs that no one reads.
        throw std::runtime_error("--prefill-logits all is a run-mode observation flag; a --session"
                                 " feed never dumps logits, so its wave is always the §06.5"
                                 " last-row form");
    }
    if (o.time_stages && (o.session || o.sizes_only)) {
        // The stage clock measures a fixed, replayable list of generation steps against three
        // submission granularities, which needs a run-mode step list: a session's `gen` budget is the
        // caller's and can end anywhere, and --sizes-only runs no steps at all.
        throw std::runtime_error("--time-stages measures a run's generation steps; --session and"
                                 " --sizes-only have no such list to re-run");
    }
    if (o.time_stages && o.steps < 1) {
        throw std::runtime_error("--time-stages needs --steps >= 1 (there is nothing to time in a"
                                 " run that generates nothing)");
    }
    if (o.time_waves && (o.session || o.sizes_only)) {
        // A wave clock prices `prefill_wave` over the run's prompt. A session prefills per `feed` line and
        // has no single prompt wave to replay, and --sizes-only prefills nothing, so neither has the object
        // this flag measures. Refusing beats printing an empty report.
        throw std::runtime_error("--time-waves measures a run's prompt wave; --session has no single"
                                 " prompt wave to replay and --sizes-only prefills nothing");
    }
    // A replay must be able to answer every step it is asked for. Silently falling back to
    // self-feeding after the list runs out would produce a run that is half forced and half
    // free, which is the one thing this mode exists to avoid.
    if (!o.feed.empty() && (int)o.feed.size() < o.steps) {
        throw std::runtime_error("--replay needs at least --steps tokens (have "
                                 + std::to_string(o.feed.size()) + ", steps "
                                 + std::to_string(o.steps) + ")");
    }
    return o;
}

void write_raw(const std::string& dir, const std::string& name, const std::vector<float>& v) {
    std::ofstream f(dir + "/" + name, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + dir + "/" + name);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
}

// ---- the session protocol: what `engine/chat.sh` drives -------------------------------------
// One process, weights and `RequestState` resident, one command per line on stdin:
//
//   feed <id> <id>…   advance the state over the caller's tokens (a new turn's prompt) as ONE
//                     prefill wave — `--prefill-logits last`, the §06.5 shape, because a feed has no
//                     use for any row's logits but the last one's — then
//                     "FED n=<k> prefix_len=<P>". No stop check, exactly as a run's prompt
//                     phase: the caller owns these tokens. It FIRST feeds the token the
//                     previous step sampled but never consumed — the last token the caller was
//                     given — so the transcript in the state is complete. A fresh request after
//                     `reset` has nothing pending and feeds only the caller's tokens.
//   gen <n>           generate at most n tokens, streaming "GEN <id> <pos>" as each one comes
//                     out, then "END steps=<k> stop=<0|1>". §07.5 applies here: the stop id is
//                     emitted and ends the turn.
//   reset             §08.4 fresh request (zero S and the conv window, P=0) -> "RESET prefix_len=0"
//   stats             re-emit the STATE line plus "PREFIX p=<P> sampled=<id>".
//   quit              "BYE" and return.
//
// Why a mode rather than one process per turn: the load is 1.55 GB of `hipMalloc` + `hipMemcpy`
// before the first token, and `RequestState` cannot cross a process boundary, so a per-turn
// process would pay the load again AND re-feed the whole transcript. Resident is the only way
// the per-turn time-to-first-token figure means what it says.
//
// Errors poison the session on purpose. A throw from `decode_step` happens partway through a
// token — the layers run in §07.1 order and `prefix_len` advances only at the end — so the state
// is half-advanced and no longer matches the caller's transcript. Refusing until `reset` is a
// refusal; continuing would be the quiet-corruption case the brief asks this driver to avoid.
// ---- §09: the draft side computation, driver side ------------------------------------------------
//
// Everything the side computation needs that is not in `engine/mtp_draft.h`: the resolved block
// weights, the draft's own cache, the round schedule, and the counters the `MTPDRAFT` line prints.
// Constructed ONLY when the flag is on — with `--mtp-draft 0` no allocation here happens and no line
// below prints, which is what makes "off is byte-identical to the binary that shipped without this
// feature" a fact about the code path rather than a claim about it.
struct MtpSide {
    engine::mtpdraft::Weights w;
    engine::mtpdraft::State st;
    engine::mtpdraft::RoundTracker track{engine::mtpdraft::Schedule::kLoop};
    long long made = 0;           // proposals submitted
    long long adjudicated = 0;    // ... and settled against a later target prediction
    long long accepted = 0;
    long long refused = 0;        // rounds the block could not run (the cache ceiling); counted and
                                  // printed, never silently dropped — a skipped proposal is not a
                                  // rejected one, and the denominator has to be able to say so
    long long unadjudicated = 0;  // proposals left pending when the run ended (no later step)
    // Draft-cache lanes that came from somewhere other than a round: the prompt seeding
    // (`--mtp-prompt-lanes`) and the accepted-candidate lanes (`--mtp-dense`). Both are counted here so
    // the census line can be reconciled against `lanes_max`, which a round-only tally cannot explain.
    long long prompt_lanes = 0;
    int lanes_max = 0;

    // ---- `--mtp-draft-k`: more than one draft row per drafted position -----------------------------
    // `k_given` is what arms the ladder, not `k`: naming `--mtp-draft-k 1` must run EXACTLY the shipped
    // path (one row per anchor, no truncation, nothing extra billed) while still printing the `DRAFTK`
    // lines, because that is how the control "k=1 reproduces the side pass we already quote" is adjudicable
    // rather than tautological.
    int k = 1;
    bool k_given = false;
    long long extra_rows = 0;    // block rows run beyond an anchor's first (each costs a whole round)
    long long row_refused = 0;   // extra rows the cache ceiling refused (`DRAFTK-SKIP`)
    long long lines_printed = 0; // `DRAFTK` lines emitted, closed mid-run or flushed at the end

    // One anchor's proposals, held until the target tokens that adjudicate them have been sampled.
    // The engine cannot print `t1` at draft time: the proposal made at anchor q is settled by the step at
    // q+1, i.e. by a token that does not exist yet — the shipped k=1 path handles this by printing
    // `adjudicated_by_pred_pos` and letting the harness join. Here the join happens in the engine and the
    // line is printed once it can carry every field, so no reader ever sees a short line.
    struct LadderRow {
        int anchor;                 // `pos_before` of the step whose row started the chain
        int pos;                    // the position the FIRST proposal proposes (anchor + 2)
        int rows;                   // rows actually run (fewer than k only if the ceiling refused one)
        std::array<int, 4> c{{-1, -1, -1, -1}};   // the draft's proposals
        std::array<int, 4> t{{-1, -1, -1, -1}};   // the target's tokens, filled as later steps arrive
    };
    std::deque<LadderRow> open;   // ascending in `anchor`; entries leave once `t[rows-1]` arrives
};

// One `DRAFTK` line: every field present on every line, and `-1` meaning "does not exist for this row"
// (past the run's last step for a `t`, refused or not asked for on this `k` for a `c`, and therefore
// `-1` for the corresponding `hit`). A reader that treats `-1` as a token id would be reading a
// different fixture than the one that was run; the harness counts it out of the denominator and prints
// how many it excluded.
void print_ladder_row(const MtpSide::LadderRow& r) {
    std::printf("DRAFTK pos=%d anchor=%d k=%d", r.pos, r.anchor, r.rows);
    for (int n = 0; n < 4; ++n) {
        const int t = n < r.rows ? r.t[n] : -1;
        const int c = n < r.rows ? r.c[n] : -1;
        const int hit = (t >= 0 && c >= 0) ? (c == t ? 1 : 0) : -1;
        std::printf(" t%d=%d c%d=%d hit%d=%d", n + 1, t, n + 1, c, n + 1, hit);
    }
    std::printf("\n");
    std::fflush(stdout);
}

// Feed this step's prediction to every open ladder row and print the ones that are now complete.
// Called once per generation step with the same `(pred_pos, pred_token)` pair the shipped adjudication
// uses, so the ladder and `MTPACCEPT` are joined on one index rather than two that happen to agree.
void ladder_observe(MtpSide& m, int pred_pos, int pred_token) {
    if (!m.k_given) return;
    for (auto& r : m.open) {
        const int n = pred_pos - r.anchor;      // 1-based position within this anchor's ladder
        if (n >= 1 && n <= r.rows) r.t[n - 1] = pred_token;
    }
    // Compact rather than pop the front: a plain run advances one position per step, so rows do close in
    // order, but a check that only works when the ordering assumption holds is a check that fails silently
    // the day it does not.
    auto it = m.open.begin();
    while (it != m.open.end()) {
        if (it->t[it->rows - 1] >= 0) {
            print_ladder_row(*it);
            m.lines_printed++;
            it = m.open.erase(it);
        } else {
            ++it;
        }
    }
}

// The run's last `k-1` anchors have no target behind some of their rows. They are printed with `-1` in
// those fields and counted out of that column's denominator by the reader — a shorter run would be the
// silently-shorter-population failure this driver keeps refusing.
void ladder_flush(MtpSide& m) {
    if (!m.k_given) return;
    for (const auto& r : m.open) {
        print_ladder_row(r);
        m.lines_printed++;
    }
    m.open.clear();
}

// Owns the prompt-norm-row retention buffer (`engine/prefill.h::NormTailRetention`) for one run. It exists
// so a throw between the allocation and the seeding cannot leak device memory, and so the free is not a
// line someone has to remember at the next early exit.
struct NormTailBuf {
    uint16_t* p = nullptr;
    ~NormTailBuf() { if (p) (void)hipFree(p); }
};

// One generation step's worth of side computation.
//
// Returns whether a round was actually submitted (a refusal or a skipped step submits nothing), so the
// stage clock's replay can state its own round count next to the `mtp` row it bills — see `stage_pass`.
//
// `quiet` suppresses the per-round lines. A timed replay of the step list is not the run's own accounting:
// the `MTPDRAFT` census and the `DRAFT round=` lines that witness it are printed once, from the run's
// generation loop, and a second copy from a replay would break the cross-check `gate.py::mtp_plan_verdicts`
// runs (census `rounds=` against the number of `DRAFT` lines) while adding nothing a replay can measure.
// The census counters still move, because they are the block's own and a replay really did run rounds.
//
// `pred_pos` is the position the step's row predicted — its `pos_before`, the same number the STEP /
// GEN lines print — and `pred_token` the token it predicted. `h_norm_row` is the BF16 final-norm row
// the step left in `sc.norm`, which is §09.1's `h` (post-final-norm; A-12's "pre-norm" reading was
// falsified at 0.232 against 0.642).
//
// THE ALIGNMENT, spelled out because getting it wrong is this repo's own recorded trap. A round whose
// target has consumed through position `pred_pos` drafts position `pred_pos + 2` behind the token
// just emitted, and that proposal is settled by the target's own prediction AT position
// `pred_pos + 1` — i.e. by the NEXT step, never this one. So `adjudicated_by_pred_pos` is printed on
// every DRAFT line and the harness joins on that printed position rather than on an index;
// `RoundTracker::after_step` refuses a pending proposal whose recorded position is not this step's.
// A free run aligned on `[0:]` read a false 0/128 here before, and the fix was never to assume an
// index.
//
// A proposal may BE the stop id (gate zero saw the restricted head propose it at dec8 step 0), and it
// settles a round like any other id: §07.5's stop rule applies to tokens the TARGET emitted, in the
// generation loop, and nothing in the side path is allowed to end a run.
bool mtp_after_step(MtpSide& m, const engine::WeightStore& store, const uint16_t* h_norm_row,
                    int pred_pos, int pred_token, bool quiet = false) {
    const auto a = m.track.after_step(pred_pos, pred_token);
    if (a.alignment_broken) {
        // Cannot happen without a bug in this file's own bookkeeping, and if it does happen the accept
        // rate is meaningless rather than merely noisy, so the run aborts instead of printing a number.
        throw std::runtime_error(
            "MTP round alignment is broken: a proposal recorded for position " +
            std::to_string(m.track.pending_pred_pos()) + " was handed the prediction of position " +
            std::to_string(pred_pos) + ". Do not read an accept rate out of this run.");
    }
    // Any ladder row this step just settled. Before the adjudication prints, so the two joins of the same
    // `(pred_pos, pred_token)` pair cannot diverge by being written against different orderings.
    ladder_observe(m, pred_pos, pred_token);
    if (a.adjudicated) {
        m.adjudicated++;
        if (a.accept) m.accepted++;
        if (!quiet)
            std::printf("MTPACCEPT settled_pos=%d proposal=%d target_pred=%d accept=%d\n", a.settled_pos,
                        a.settled_proposal, pred_token, a.accept ? 1 : 0);
    }
    if (a.drop_lane) engine::mtpdraft::truncate_lane(m.st);
    if (!a.draft) return false;          // kLoop skips the step after an accept: q advanced by 2

    const std::string why = engine::mtpdraft::refusal(m.st);
    if (!why.empty()) {
        m.refused++;
        // Printed even under `quiet`, because it is not accounting: a refused round is a step the `mtp`
        // stage row billed and did not get, and a `plan_match=0` with no explanation beside it is the
        // silent-shorter-run failure this driver keeps refusing.
        std::printf("DRAFT-SKIP reason=%s would_adjudicate_at=%d\n", why.c_str(), pred_pos + 1);
        return false;
    }
    const engine::mtpdraft::Proposal p =
        engine::mtpdraft::draft_round(store, m.w, m.st, h_norm_row, pred_token, pred_pos + 2);
    m.track.pending(pred_pos + 1, p.token);
    m.made++;
    if (m.st.lanes_max > m.lanes_max) m.lanes_max = m.st.lanes_max;
    if (!quiet) {
        std::printf("DRAFT round=%lld draft_pos=%d adjudicated_by_pred_pos=%d y_next=%d proposal=%d"
                    " top_logit=%.9g lanes=%d\n",
                    (long long)p.round, pred_pos + 2, pred_pos + 1, pred_token, p.token, p.top_logit,
                    p.nlanes);
        std::fflush(stdout);
    }

    // ---- `--mtp-draft-k`: rows 2..k, the block feeding itself ---------------------------------------
    //
    // WHY `h` IS `x_d` HERE AND NOT A TARGET ROW. Row 1's `h` is the target's final-norm row at `pred_pos`
    // (`sc.norm`, §09.1 as A-12 corrected). Behind a token the target never fed there is no target row to
    // read — `sc.norm` holds only the rows the step actually priced — so row n reads the draft block's own
    // output row, `mtp.norm`'s output `x_d` (§09.2), which is what the block's own head just read and is
    // therefore the same *kind* of object (post-norm, same width, same arena). This is llama.cpp's feedback
    // verbatim in role: `common/speculative.cpp:1770-1783` re-adds `id_sampled` with
    // `llama_get_embeddings_nextn_ith(ctx_dft, i_last)` as the embedding state at position `pos0 + i + 1`.
    // It is a DIFFERENT `h` from the accept loop's (D8 in `prototype/mtp_accept/DRAFT-COND.md`), so every
    // rate printed from here is a side-pass rate and is labelled that way.
    //
    // ALIASING, safe by construction and not by luck: `st.xd` is written only by stage 14
    // (`glue_add_norm(st.z1, st.mlp_o, w.n_mtp, st.z2, st.xd)`) and read only by stage 15's head pass, so
    // passing it as the NEXT call's `h_norm_row` means it is read at that call's stage 3 and overwritten at
    // its own stage 14 — strictly after every read. One stream, in issue order, so no sync is needed and
    // none is taken (a sync here would be a silent rate-changer, since the read is of the previous row).
    //
    // LANES, and the truncation that keeps them a sequence. Each row appends its own lane at position
    // `pos + n - 1`, ascending, which is what `mtp_attn_row` requires (`self_lane` is the newest). But the
    // next anchor is at `pred_pos + 1` or `+ 2`, so its first lane would sit at a position one of THESE rows
    // already used. After the chain, all but the first row's lane are dropped — a length field, 0 bytes
    // (`engine/mtp_draft.h::truncate_lane`) — leaving the cache exactly the sequence the shipped k=1 path
    // builds, one lane per visited anchor. So row n sees the accepted prefix's lanes plus its own anchor's
    // speculative rows, which is what a real k-row round gives it, and the cache never holds a repeated or
    // descending position. Note this leaves the kLoop reject's `truncate_lane` removing precisely the first
    // row's lane, as it does at k=1.
    if (m.k_given) {
        MtpSide::LadderRow row;
        row.anchor = pred_pos;
        row.pos = pred_pos + 2;
        row.rows = 1;
        row.c[0] = p.token;
        const int lanes_after_first = m.st.nlanes;   // AFTER row 1 appended its lane: the frontier to return to
        uint16_t* h_next = m.st.xd;
        int y_next = p.token;
        int lane_pos = row.pos + 1;
        for (int n = 2; n <= m.k; ++n) {
            const std::string row_why = engine::mtpdraft::refusal(m.st);
            if (!row_why.empty()) {
                // Printed, not swallowed: a ladder whose third column is short because the cache filled is a
                // different population from one that is short because the run ended, and only this line says
                // which. `c=-1` on the `DRAFTK` line is the machine-readable half of that statement.
                std::printf("DRAFTK-SKIP row=%d would_propose_for=%d reason=%s\n", n, lane_pos,
                            row_why.c_str());
                m.row_refused++;
                break;
            }
            const engine::mtpdraft::Proposal q =
                engine::mtpdraft::draft_round(store, m.w, m.st, h_next, y_next, lane_pos);
            m.extra_rows++;
            row.c[n - 1] = q.token;
            row.rows = n;
            y_next = q.token;
            lane_pos++;
            if (m.st.lanes_max > m.lanes_max) m.lanes_max = m.st.lanes_max;
        }
        // Return the cache to the frontier row 1 left. Measured, not assumed: leaving ONE lane too many
        // behind (this line read `> lanes_after_first + 1` in the first version) added a lane per anchor whose
        // position sat ahead of the next anchor's own first lane, and the first column's rate moved
        // 0.6641 -> 0.7100 on the corpus. The k=1 control CANNOT see that, because at k=1 the loop above
        // never runs; the witness is that the first column is INVARIANT in k, which is what
        // `prototype/mtp_accept/draftk_ladder.py` compares across the k=1 and k=4 passes.
        while (m.st.nlanes > lanes_after_first) engine::mtpdraft::truncate_lane(m.st);
        m.open.push_back(row);
    }
    return true;
}

// The proposal the last round made and no later step settled. Counted and explained, never dropped: it
// belongs in "proposals made" and not in the numerator, and a tally that cannot say where it ended
// reads as a rate that is simply lower.
void mtp_close_run(MtpSide& m) {
    ladder_flush(m);
    if (m.track.has_pending()) {
        m.unadjudicated++;
        std::printf("DRAFT-UNADJUDICATED adjudicated_by_pred_pos=%d proposal=%d"
                    " reason=no_later_generation_step_in_this_run_or_turn\n",
                    m.track.pending_pred_pos(), m.track.pending_proposal());
        m.track.clear();
    }
}

void mtp_census_line(const MtpSide& m, const char* mode, const char* sched) {
    std::printf("MTPDRAFT on=1 sched=%s mode=%s block=%d+%d head=%d+%d per_round=%d"
                " rounds=%lld adjudicated=%lld accepted=%lld refused=%lld unadjudicated=%lld"
                " lanes_max=%d cap=%d lane_appends=%lld prompt_lanes=%lld pos_shift=%d"
                " tensors=15 block_bytes=%lld\n",
                sched, mode,
                engine::mtpdraft::kBlockCopies, engine::mtpdraft::kBlockLaunches,
                // `head=` is `copies+launches`, the same order `block=` prints, because the gate reads the
                // first number as the count of blocking reads (`engine/gate.py::mtp_plan_verdicts`'s stage
                // row). With both head numbers at 1 the two orders were indistinguishable; the device
                // argmax made them 1 and 3, so the order had to be stated instead of assumed.
                engine::mtpdraft::kHeadCopies,
                engine::mtpdraft::kHeadLaunches + engine::mtpdraft::kArgmaxLaunches,
                engine::mtpdraft::kSubmissionsPerRound,
                (long long)m.made, (long long)m.adjudicated, (long long)m.accepted,
                (long long)m.refused, (long long)m.unadjudicated, m.lanes_max, m.st.cap,
                (long long)m.st.lane_appends, (long long)m.prompt_lanes, m.st.pos_shift,
                m.w.bytes);
    std::fflush(stdout);
}

// The `--mtp-draft-k` census, printed after `MTPDRAFT` at run close and only when the flag was named.
// `rows = anchors + extra_rows` is asserted, not assumed: `draft_round` bumps the block's own row counter
// for every row it runs while the driver's `made` stays one per anchor, so this is the one place the two
// tallies of the same work meet, and a row that ran without being billed or was billed twice fails here
// rather than quietly shifting a denominator (`MTPDRAFT rounds=` still means "one per `DRAFT` line", which
// is what `gate.py`'s census cross-check counts).
void mtp_draftk_line(const MtpSide& m, const char* mode, const char* sched) {
    if (!m.k_given) return;
    const long long rows = m.st.rounds;
    const bool ok = rows == m.made + m.extra_rows && m.lines_printed == m.made;
    std::printf("MTPDRAFTK on=1 k=%d mode=%s sched=%s row_subs=%d extra_subs_per_anchor=%d"
                " lines=%lld anchors=%lld extra_rows=%lld rows=%lld row_refused=%lld lanes_max=%d"
                " identity=%s\n",
                m.k, mode, sched, engine::mtpdraft::kSubmissionsPerRound,
                (m.k - 1) * engine::mtpdraft::kSubmissionsPerRound,
                (long long)m.lines_printed, (long long)m.made, (long long)m.extra_rows,
                (long long)rows, (long long)m.row_refused, m.lanes_max, ok ? "ok" : "BAD");
    std::fflush(stdout);
    if (!ok) {
        // The identity is the ladder's own admission: a BAD here means the per-row tally and the per-anchor
        // tally disagree, so every count in the ladder is suspect. Saying it after printing the line, and
        // then throwing, keeps the evidence on the wire.
        throw std::runtime_error("MTPDRAFTK identity is BAD: the block ran " + std::to_string(rows) +
                                 " rows but the driver billed " + std::to_string(m.made) +
                                 " anchors + " + std::to_string(m.extra_rows) +
                                 " extra rows, or printed " + std::to_string(m.lines_printed) +
                                 " ladder lines for " + std::to_string(m.made) +
                                 " anchors. Do not read a ladder out of this run.");
    }
}

// ---- §09 / P7: the batched verify body, driver side -----------------------------------------------
//
// Everything the probe needs that is not in `engine/mtp_verify.h`: the snapshot, the shadow's logits tile,
// and the lines that make its cost and its byte surface printed numbers. Built ONLY when the flag is on,
// for the same reason `MtpSide` is: with the flag off no allocation here happens and no line below prints,
// which is what makes "off is byte-identical to the binary that shipped without this" a fact about the
// code path rather than a claim about it.
struct VerifySide {
    engine::mtpverify::Snapshot snap;
    engine::mtpverify::LaneStage lanes;
    engine::mtpverify::Stats stats;
    float* logits_dev = nullptr;      // [kRows][kVocab] F32, the shadow's own logits tile
    // Which route this run asked the body to run (`--mtp-route`), and the route the bodies ACTUALLY ran,
    // recorded off the first round's `Result`. Both are printed: the first says what was asked, the second
    // is the canary that it happened (`AGENTS.md`: a stale binary agrees with itself).
    engine::mtpverify::Route route = engine::mtpverify::Route::kHand;
    engine::kernels::WaveGemm gemm_ran = engine::kernels::WaveGemm::kHand;
    engine::kernels::GdnScan scan_ran = engine::kernels::GdnScan::kSeq;
    bool guard = false;               // `--verify-guard 1`: see `engine/mtp_verify.h`'s witness section
    // `--mtp-midsplit 1`: V1's mid-round checkpoint regions, and the device status word the verdict
    // kernel writes. Allocated only under the flag, because a witness that costs memory when off is a
    // witness that will be left on by accident.
    bool midsplit_mode = false;         // any non-zero mode: the second body runs and is compared
    int midsplit = 0;                   // 1 = split arm, 2 = control arm (second body unsplit)
    engine::DecodeScratch::GdnCheckpoint mid;
    int* verdict_dev = nullptr;
    long long mid_bytes = 0;
    long long split_logit_diffs = 0;   // floats that disagreed, split vs batched, whole run
    long long split_end_diffs = 0;     // bytes of end state that disagreed
    long long split_rounds = 0;
    long long verdict_bad = 0;         // rounds where the device verdict disagreed with `argmax_greedy`
};

// One round, run AFTER the step and after the draft have taken everything they need from `sc` (the verify
// body overwrites rows 0..1 of the scratch, including `sc.norm`, which the draft reads). `anchor_pos` is
// the position the step's own row sat at; `fed` the token it was fed; `candidate` row 1's token.
//
// The printed `row0_eq_step` is not a courtesy: it is the per-step behavioural canary that row 0 of the
// batched body reproduces the step it replaces, on the engine's own bits, in the same run that produced
// the cost figure. If it ever reads 0 the batched-body premise is wrong and `prototype/mtp_delta`'s whole
// `delta` has to be re-priced — which is why it is a printed field and not an assert that ends the run.
//
// With the guard on, the same round also answers two questions in bytes rather than in metrics, because a
// metric is what let this body pass four checks while reading the wrong state:
//
//   operands   what the step was about to read, against what the shadow actually held. A row-0 logits gap
//              is "different operands" or it is rounding, and this is the only line that can tell.
//   leak       every live buffer, before the body against after it, plus the poisoned lane. §08.6 makes
//              one lane above the prefix writable and everything below it sacred; this says which of the
//              two the body touched, and the poison says whether it wrote anywhere at all.
//
// `selftest` is the witness that the witnesses bite: the step is required to move the mixer state, so a
// mixer comparison across a step that reads "same" is a comparator that cannot fail, and that is the one
// failure mode this section would otherwise share with the empty-parse gates in `AGENTS.md`.
void verify_round(VerifySide& v, const engine::WeightStore& store, engine::DecodeScratch& sc,
                  const engine::RequestState& st, int anchor_pos, int fed, int candidate,
                  int step_token, int step, const std::string& out_dir,
                  const engine::mtpverify::StateBytes* step_reads,
                  const engine::mtpverify::StateBytes* step_after) {
    const int tokens[engine::mtpverify::kRows] = {fed, candidate};
    engine::mtpverify::StateBytes body_reads;
    // The lane the body is entitled to write: its second row sits at `anchor_pos + 1`, which is
    // `st.prefix_len` now that the step has advanced it. Poisoning it is safe precisely because it is above
    // the live prefix, and it converts "row 1 wrote its own lane" from an assumption into a printed byte
    // count. `poison_kv_lane` refuses anything at or below the prefix, so this cannot drift into poisoning a
    // lane the request reads. The REPORT is taken inside `verify_body`, between the body and the lane revert:
    // the revert puts this lane's pre-body bytes back, so a report read here, after the call, would describe
    // the revert rather than the body — and a witness that measures the repair instead of the write is exactly
    // the decoration this tree keeps refusing.
    const int poison_lane = st.prefix_len;
    engine::mtpverify::StateBytes live_before;
    engine::mtpverify::PoisonReport poison;
    if (v.guard) {
        // `witness` (`engine/mtp_meter.h`): the guard's own downloads and byte-compares. This is the phase
        // that reads 0.000 on a TIMING arm and hundreds of milliseconds on an admission arm, and printing it
        // is what stops a reader from having to guess which kind of run produced the table. The guard is
        // left OFF by every timing pass precisely because of what it costs here.
        const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
        live_before = engine::mtpverify::capture_state_bytes(st, -1);
        engine::mtpverify::poison_kv_lane(st, poison_lane);
    }
            const engine::mtpverify::Result r = engine::mtpverify::verify_body(
                store, st, sc, v.route, v.snap, anchor_pos, tokens, v.logits_dev, v.lanes,
                v.guard ? &body_reads : nullptr, v.guard ? &poison : nullptr, nullptr,
                v.midsplit != 0);
    v.gemm_ran = r.gemm_ran;
    v.scan_ran = r.scan_ran;
    v.stats.rounds++;
    v.stats.snapshot_bytes += v.snap.bytes();
    v.stats.restored_bytes += v.snap.bytes();   // the shadow is seeded every round, not only on a reject
    v.stats.lane_bytes += r.lane_bytes;         // the anchor-lane revert, both directions, this round
    v.stats.kv_bytes_left = engine::mtpverify::kv_bytes_above_prefix(st);
    {
        const engine::mtpmeter::PhaseGuard prints(engine::mtpmeter::kReportPrints);
        std::printf("VERIFY2 step=%d anchor_pos=%d fed=%d candidate=%d row0=%d row1=%d row0_top=%.9g"
                    " row1_top=%.9g row0_eq_step=%d head_passes=%d d2h=%d logits_bytes=%lld"
                    " checkpoint_bytes=%lld lane_revert_bytes=%lld guard=%d"
                    " mtp_route=%s gemm=%s gdn_scan=%s attn_fattn_wide=%s\n",
                    step, anchor_pos, fed, candidate, r.token[0], r.token[1], r.top[0], r.top[1],
                    r.token[0] == step_token ? 1 : 0, r.head_passes, r.d2h_count,
                    (long long)r.bytes_read_back, (long long)v.snap.bytes(), (long long)r.lane_bytes,
                    v.guard ? 1 : 0,
                    engine::mtpverify::route_name(v.route),
                    engine::kernels::wave_gemm_name(r.gemm_ran),
                    engine::kernels::gdn_scan_name(r.scan_ran),
                    engine::kernels::attn_wide_name(r.attn_wide_ran));
        std::fflush(stdout);
    }
    if (v.guard) {
        const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
        const engine::mtpverify::StateBytes live_after =
            engine::mtpverify::capture_state_bytes(st, -1);        const std::string operands = engine::mtpverify::first_state_diff(*step_reads, body_reads);
        const std::string leak = engine::mtpverify::first_state_diff(live_before, live_after);
        const std::string selftest =
            engine::mtpverify::first_mixer_diff(*step_reads, *step_after);
        v.stats.guard_bytes += (long long)live_before.gdn.size() + (long long)live_before.conv.size() +
                               (long long)live_before.kv.size();
        std::printf("VERIFYGUARD step=%d pos=%d poison_lane=%d operands_same=%d leak_clean=%d"
                    " poison_written=%d poison_regions_unwritten=%d/%d poison_survivors=%lld"
                    " poison_lane_bytes=%lld selftest_bites=%d"
                    " operand_first_diff=%s leak_first_diff=%s leak_extent=%s\n",
                    step, anchor_pos, poison_lane, operands.empty() ? 1 : 0, leak.empty() ? 1 : 0,
                    poison.written() ? 1 : 0, poison.regions_unwritten, poison.regions,
                    poison.survivors, poison.lane_bytes, selftest.empty() ? 0 : 1,
                    operands.empty() ? "-" : operands.c_str(), leak.empty() ? "-" : leak.c_str(),
                    engine::mtpverify::state_diff_summary(live_before, live_after).c_str());
        std::fflush(stdout);
    }
    // ---- `--mtp-midsplit 1`: V1's admission, three byte-compares and no metric -------------------
    //
    // The same body, the same anchor, the same two tokens, run a SECOND time with the mid-round
    // checkpoint armed. The split changes the SUBMISSIONS the GDN layers make (conv per row, recurrence
    // per row, a state copy in the gap) and is owed bit-identity, so what this prints is equality of
    // bytes, in three places, each answering a different question:
    //
    //   split_eq_batch   the split body and the batched body agree on BOTH logits and end state. The
    //                    logits alone would not be enough: two routes that disagreed about the state
    //                    they hand the next step could still agree about this step's row.
    //   mid_vs_step      what the captured checkpoint differs from, relative to the mixer state the
    //                    shipping ONE-ROW step left after consuming the same token at the same position.
    //                    Printed as a MEASUREMENT, not an admission: row 0 of a 2-row body runs wave
    //                    attention at rows = 2 while the one-row step runs the fused single-row form, so
    //                    bit equality against the one-row state was never an admissible ask. What IS
    //                    owed bitwise is `split_eq_batch` (the split against the batched body, which is
    //                    the same form on both sides); the argmax-level claim against the step is
    //                    `mtp_verify`'s existing `row0_eq_step`. Printing the difference rather than
    //                    dropping it is deliberate — if that ever becomes zero, the two forms converged.
    //   verdict_eq_host  the device argmax and the accept bit agree with `argmax_greedy` on the same
    //                    logits tile, tie rule included (§03.4's strict `>` / lowest-id-wins).
    //
    // A near-match here is a FAIL, not a band: `ACCEPT-BUILD.md` §4 cond. 2 says so in those words.
    if (v.midsplit != 0) {
        // Mode 2 is the CONTROL: the second body runs UNSPLIT, so a nonzero `logits_ndiff` there is a
        // witness bug — the second call seeing inputs the first did not — and not a result about the
        // split. Running it is what makes mode 1's number readable, and it costs one more body.
        const engine::mtpverify::Result s = engine::mtpverify::verify_body(
            store, st, sc, v.route, v.snap, anchor_pos, tokens, v.logits_dev, v.lanes,
            nullptr, nullptr, v.midsplit == 1 ? &v.mid : nullptr,
            /*capture_end_state=*/true, v.verdict_dev);
        v.split_rounds++;
        // Per row, not pooled: `ndiff == kVocab exactly` is a whole-row difference, and which ROW
        // differs is the diagnosis (row 0 = the step's twin is broken; row 1 = the chained conv
        // window is broken; both = something upstream of the mixer).
        long long logit_diff0 = 0, logit_diff1 = 0;
        for (int q = 0; q < engine::mtpverify::kRows; ++q) {
            for (int i = 0; i < engine::kVocab; ++i) {
                const size_t idx = (size_t)q * engine::kVocab + i;
                if (std::memcmp(&s.logits[idx], &r.logits[idx], sizeof(float)) != 0) {
                    (q == 0 ? logit_diff0 : logit_diff1)++;
                }
            }
        }
        const long long logit_diff = logit_diff0 + logit_diff1;
        long long end_diff = 0;
        if (s.end_state.size() != r.end_state.size()) {
            end_diff = -1;
        } else {
            for (size_t i = 0; i < s.end_state.size(); ++i) {
                if (s.end_state[i] != r.end_state[i]) ++end_diff;
            }
        }
        // The mid checkpoint against the state the shipping ONE-ROW step left, as a MEASUREMENT and not an
        // admission. The body's row 0 runs the wave attention path (`rows = 2` reaches `attn_prep` plus the
        // staged core) while the step ran the fused single-row form, so the two row-0 mixer states are
        // argmax-equivalent and were never bit-equal — `VERIFY2 row0_eq_step=1` is argmax-level by design.
        // Byte equality against `step_after` was therefore not an admissible ask; a non-`-` value here says
        // the two forms' row-0 states differ in bytes, which is expected, and `split_eq_batch` above is
        // what adjudicates the split.
        std::string mid_diff = "no_step_after";
        if (v.midsplit != 1) {
            mid_diff = "control";   // the control arm captures no checkpoint; there is nothing to ask
        } else if (step_after != nullptr) {
            mid_diff = "-";
            const char* mid_g = s.mid_state.data();
            const char* mid_c = s.mid_state.data() + step_after->gdn.size();
            if (s.mid_state.size() != step_after->gdn.size() + step_after->conv.size()) {
                mid_diff = "size";
            } else if (std::memcmp(mid_g, step_after->gdn.data(), step_after->gdn.size()) != 0) {
                mid_diff = "gdn_state_differs";
            } else if (std::memcmp(mid_c, step_after->conv.data(), step_after->conv.size()) != 0) {
                mid_diff = "conv_window_differs";
            }
        }
        const bool verdict_ok = s.verdict[engine::kernels::kVerdictRow0Token] == s.token[0] &&
                                s.verdict[engine::kernels::kVerdictRow1Token] == s.token[1] &&
                                s.verdict[engine::kernels::kVerdictAccept] ==
                                    (s.token[0] == tokens[0] ? 1 : 0);
        v.split_logit_diffs += logit_diff;
        v.split_end_diffs += end_diff;
        if (!verdict_ok) v.verdict_bad++;
        std::printf("MIDSPLIT step=%d anchor_pos=%d mode=%d split_eq_batch=%d logits_ndiff=%lld"
                    " logits_ndiff_row0=%lld logits_ndiff_row1=%lld end_ndiff=%lld mid_vs_step=%s"
                    " verdict_eq_host=%d row0=%d row1=%d candidate=%d accept=%d"
                    " status_bytes=%lld\n",
                    step, anchor_pos, v.midsplit,
                    (logit_diff == 0 && end_diff == 0) ? 1 : 0, (long long)logit_diff,
                    (long long)logit_diff0, (long long)logit_diff1, (long long)end_diff,
                    mid_diff.c_str(),
                    verdict_ok ? 1 : 0, s.token[0], s.token[1], tokens[0],
                    s.verdict[engine::kernels::kVerdictAccept], (long long)s.verdict_bytes);
        std::fflush(stdout);
    }
    if (!out_dir.empty()) {
        // Raw F32, [2][kVocab], the same dtype rule every logits dump in this driver follows (a
        // comparator that guesses the dtype from a file size is `AGENTS.md`'s trap; so is reading F32
        // dumps as BF16).
        write_raw(out_dir, "verify2_" + std::to_string(step) + ".bin", r.logits);
    }
}

void verify_census_line(const VerifySide& v, const std::string& restore_witness) {
    std::printf("VERIFY2DONE on=1 rows=%d rounds=%lld checkpoint_bytes_per_round=%lld"
                " restored_bytes=%lld lane_revert_bytes_per_round=%lld lane_revert_bytes=%lld"
                " kv_bytes_above_prefix=%lld head=batched d2h_per_round=1"
                " mtp_route=%s gemm=%s gdn_scan=%s guard=%d guard_bytes=%lld restore_witness=%s\n",
                engine::mtpverify::kRows, (long long)v.stats.rounds, (long long)v.snap.bytes(),
                (long long)v.stats.restored_bytes,
                v.stats.rounds ? (long long)v.stats.lane_bytes / (long long)v.stats.rounds : 0LL,
                (long long)v.stats.lane_bytes,
                (long long)v.stats.kv_bytes_left,
                engine::mtpverify::route_name(v.route),
                engine::kernels::wave_gemm_name(v.gemm_ran),
                engine::kernels::gdn_scan_name(v.scan_ran),
                v.guard ? 1 : 0, (long long)v.stats.guard_bytes,
                restore_witness.empty() ? "identical" : restore_witness.c_str());
    std::fflush(stdout);
}

void session_loop(const engine::WeightStore& store, engine::RequestState& st,
                  engine::DecodeScratch& sc, MtpSide* mtp) {
    std::printf("READY cap=%lld\n", (long long)st.cap);
    std::fflush(stdout);
    bool poisoned = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream words(line);
        std::string cmd;
        words >> cmd;
        if (cmd.empty()) continue;
        if (cmd == "quit" || cmd == "bye") {
            if (mtp) {
                mtp_close_run(*mtp);
                mtp_census_line(*mtp, "session",
                                engine::mtpdraft::schedule_name(mtp->track.sched));
            }
            std::printf("BYE\n");
            std::fflush(stdout);
            return;
        }
        if (cmd == "reset") {
            st.reset();
            st.sampled_token = -1;
            st.stopped = false;
            if (mtp) {
                // §08.4 fresh request, and the draft's own cache is fresh with it: the block's cache is
                // per-request state like the target's KV, and a stale lane from the previous turn would
                // earn real softmax weight in this turn's first draft.
                engine::mtpdraft::reset(mtp->st);
                mtp->track.clear();
            }
            poisoned = false;
            std::printf("RESET prefix_len=0\n");
            std::fflush(stdout);
            continue;
        }
        if (poisoned) {
            std::printf("ERROR session is poisoned by the failed step: send reset before the "
                        "next feed or gen\n");
            std::fflush(stdout);
            continue;
        }
        // Arguments are validated OUTSIDE the step guard: a typo is the caller's problem and must
        // not poison a session that is otherwise fine. Only a throw from `decode_step` does that.
        if (cmd == "gen" && st.sampled_token < 0) {
            std::printf("ERROR gen with no sampled token: feed this turn's prompt first\n");
            std::fflush(stdout);
            continue;
        }
        int budget = 0;
        if (cmd == "gen" && !(words >> budget)) {
            std::printf("ERROR gen needs a token count\n");
            std::fflush(stdout);
            continue;
        }
        try {
            if (cmd == "feed") {
                // The pending sample first, if there is one: it is the last token the caller was
                // shown and the state has not consumed it (see the invariant above). The wave prices
                // only its last row, so the state's next sample is left pending at the end — which is
                // what `gen` emits first — and the intermediate argmaxes the per-token loop used to
                // compute and throw away are gone (§06.5).
                std::vector<int> tokens;
                if (st.sampled_token >= 0) tokens.push_back(st.sampled_token);
                long id = 0;
                while (words >> id) tokens.push_back((int)id);
                engine::prefill_wave(store, st, sc, tokens, engine::PrefillLogits::kLast);
                std::printf("FED n=%d prefix_len=%d\n", (int)tokens.size(), st.prefix_len);
            } else if (cmd == "gen") {
                int emitted = 0;
                bool stop = false;
                int cur = st.sampled_token;
                int last_gen_pos = -1;      // GEN label of the last token this turn printed
                while (emitted < budget) {
                    const int pos = st.prefix_len - 1;
                    std::printf("GEN %d %d\n", cur, pos);
                    std::fflush(stdout);
                    ++emitted;
                    last_gen_pos = pos;
                    if (cur == engine::kStopToken) {   // §07.5, generation phase
                        stop = true;
                        break;
                    }
                    const engine::StepResult r = decode_step(store, st, sc, cur, nullptr);
                    cur = r.token;
                    if (mtp) {
                        // The step's row predicted position `prefix_len - 1`; `sc.norm` row 0 is the
                        // final-norm row it was priced from, and it is still there (the tail only reads
                        // it). Generation phase only, like §07.5's stop rule on the line above.
                        mtp_after_step(*mtp, store, sc.norm, st.prefix_len - 1, cur);
                    }
                }
                st.sampled_token = stop ? cur : -1;
                // Stop path: `cur` was emitted but never stepped, so it stays pending and the next
                // `feed` steps it — the transcript in the state ends up complete, and the driver
                // does not have to re-send the stop token.
                // Budget path: the last emitted token WAS stepped, so `cur` is a sample nobody
                // was ever shown. Dropping it is the difference between a state that matches the
                // caller's transcript and one holding a token the user never saw; the driver
                // closes the assistant turn instead (`last_stop != "eos"` in engine/chat.py).
                // The turn's last round may have left a proposal that no later step in THIS turn
                // settles; printing that here (before `END`) attributes it to the turn that made it, so
                // the harness's per-line tally and the engine's census agree. In a chat session a
                // following turn's first token would settle it, and the reason text says exactly that
                // rather than claiming the proposal is lost.
                // The budget path stepped the last printed token, so `cur` is a real target sample that
                // this turn never prints (the driver closes the assistant turn instead — see above). The
                // draft may already have had a proposal settled by exactly that sample, and a tally that
                // could not see it would read one settlement short of the engine's own census. Printing
                // it is draft-only, so a flag-off session emits byte-for-byte what it always did, and
                // `chat.py` treats an unrecognised line as a diagnostic (its TERMINAL filter), so the
                // driver protocol is unchanged. `GEN` labels a token by `prefix_len - 1` at the moment it
                // prints, one row behind the sample it shows, so this sample's label is one past the last
                // printed one — which is the same key `mtp_after_step` filed the pending proposal under.
                if (mtp) {
                    if (!stop && emitted > 0)
                        std::printf("LASTSAMPLE pos=%d token=%d\n", last_gen_pos + 1, cur);
                    mtp_close_run(*mtp);
                }
                st.stopped = stop;
                std::printf("END steps=%d stop=%d prefix_len=%d\n", emitted, stop ? 1 : 0,
                            st.prefix_len);
            } else if (cmd == "stats") {
                std::printf("PREFIX p=%d sampled=%d stopped=%d\n", st.prefix_len,
                            st.sampled_token, st.stopped ? 1 : 0);
                if (mtp) {
                    // The census is the caller's per-turn read of the side computation; the per-turn
                    // ACCEPT tally is not, and deliberately is not printed here: the harness joins the
                    // printed positions itself, so that its join and this one are two implementations
                    // that have to agree rather than one number reading its own output.
                    mtp_census_line(*mtp, "session", engine::mtpdraft::schedule_name(mtp->track.sched));
                }
            } else {
                std::printf("ERROR unknown command '%s' (feed|gen|reset|stats|quit)\n",
                            cmd.c_str());
            }
        } catch (const std::exception& e) {
            poisoned = true;
            std::printf("ERROR %s\n", e.what());
        }
        std::fflush(stdout);
    }
}

void print_usage(FILE* out) {
    std::fprintf(out,
                 "usage: decode_run [--model DIR] [--ids ID,ID,...|--ids-file PATH] [--steps N] [--out DIR]"
                 " [--replay ID,ID,...] [--cap N] [--session] [--no-stop] [--state-fp]"
                 " [--prefill-logits last|all] [--wave-rows N]"
                 " [--mlp-map auto|interleave|contig] [--gemm library|legacy|lt|hand] [--attn-tile 0|1]"
                 " [--gemm-pins all|none|<csv>] [--gemm-heur N] [--attn-touch 0|1]"
                 " [--attn-fattn-roll single|two] [--attn-fattn-qstage stage|unstaged8]"
                 " [--attn-fattn-wide 0|1|2]"
                 " [--attn-split-cap N]"
                 " [--gdn-spell ship|unroll4|fused] [--gdn-norm inline|hoist]"
                 " [--gdn-geom block|wf4|wf2|reg16]"
                 " [--mtp-draft 0|1] [--mtp-sched loop|every] [--mtp-cap N] [--mtp-draft-k 1|2|3|4]"
                 " [--mtp-dense 0|1] [--mtp-prompt-lanes N] [--mtp-pos-shift N]"
                 " [--mtp-verify 0|1] [--verify-candidate ID|sampled] [--verify-guard 0|1]"
                 " [--mtp-midsplit 0|1|2] [--mtp-accept 0|1] [--mtp-route hand|amortised]"
                 " [--mtp-meter 0|1]"
                 " [--time-stages] [--time-waves] [--load-verify] [--sizes-only]\n"
                 "  --gemm library|hand  which GEMM a WIDE body runs (route (c)). `library` is the"
                 " default and sends the ten fat prefill"
                 " matmuls to hipBLAS for a body of two or more rows — bf16 in, f32 accumulate, bf16 C,"
                 " the weight used as stored, one handle and one pinned workspace, all on the default"
                 " stream — and keeps the hand GEMVs for in_proj_a/b, every mixer, glue and cast."
                 " `hand` is still fully supported and runs the pre-route path everywhere; it is the"
                 " A/B side and the route whose bits are byte-identical to a pre-route binary. Not"
                 " byte-identical across wave widths on the library route — that check is inside a band"
                 " there (engine/gate.py's CASCADE_BAND, progress/2026-10-14/route-c-audit-2026-10-14.md). A one-row"
                 " body is the hand GEMV under either setting, so no decode step changes shape or bits"
                 " on either side of this flag. `lt` sends the same ten fat shapes to hipBLASLt"
                 " (`hipblasLtMatmul`) instead of hipBLAS, with one algorithm chosen per (shape, M) at"
                 " plan time and printed as a `GEMM-LT` line before the first body; `legacy` is a"
                 " synonym for `library`, so a shell loop can spell both sides of the pair it is timing."
                 " Not the default: same numerics contract as `library` (bf16 in, f32 accumulate, bf16"
                 " C, 128 MiB workspace, default stream) with a different accumulation ORDER, so it is"
                 " judged by the same cascade band and not by byte-identity"
                 " (prototype/gemm_algo_sweep/results.md §3a)\n"
                 "  --gemm-pins all|none|<csv of shape names>  which entries of the lt algo-pin table"
                 " are in force, `all` by default and meaningless unless `--gemm lt` is running. A pin"
                 " is an algorithm chosen in situ for one (shape, M) and a subset is only adjudicable"
                 " in situ, so this is a runtime knob rather than a rebuild-per-subset"
                 " (prototype/gemm_algo_sweep/results.md §3d, D28). The `GEMM` line echoes"
                 " `pins=<in force>/<in the table>` so a log says which config produced its bits, and"
                 " `engine/gate.py` refuses a run whose echo disagrees with what it asked for\n"
                 "  --gemm-heur N  how many candidates the lt heuristic query is asked for per"
                 " (shape, M); ceiling 64, and meaningless unless `--gemm lt` is running. The default is"
                 " `kHeuristicWidth` in `engine/kernels_gemm.cpp` (8 today) and is deliberately not"
                 " repeated here, so this usage line cannot drift from it. The"
                 " engine still runs the FIRST admissible record, so widening the pool does not change"
                 " what an unpinned shape executes — measured, not assumed: rank-0 is the same algorithm"
                 " at width 8 and width 64 for all seven shapes, while rank 2 of the two lists is NOT"
                 " (`deep_pins.py --prefix-check`), so the deep field is a partly different"
                 " list rather than a longer one. Its use is that a deep-field candidate can be"
                 " reproduced in situ, and that a shape whose top ranks turn out inadmissible gets a"
                 " wider net; the `GEMM` line echoes `heur_width=` so every log says which pool it ran"
                 " under\n"
                 "  --attn-tile 0|1  which attention FORM a WAVE body runs (Phase 2, "
                 "progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md §9). `1`, the default since results.md §21, runs"
                 " `attn_tile_wave`, in which one K/V row load feeds the G=4 GQA heads of ONE query row"
                 " (M is fixed at 1: with a per-row lane cut two rows share no window, so row tiling buys"
                 " nothing — `static_assert(ATTN_TILE_M == 1)`), which is where the wave's K/V re-reads"
                 " come from. `0` is the shipped per-(row,head)"
                 " `attn_core`. The contract is bit-identity with the shipped form on both the"
                 " one-pass route and §07.3's streaming route — the five ordered reductions of §05.5 are"
                 " unchanged, only which block a K row comes from. Wave-only: a one-row body is the fused"
                 " single-row form under either setting, so no decode bit moves, and the tiled form"
                 " REFUSES (`score_f32`, a fused plan, a non-32 warp) rather than falling back, because a"
                 " silent fallback would price the shipped form twice and report it as the tiled one\n"
                 "  --attn-touch 0|1  which SPELLING of the tiled wave's touch loop runs "
                 "(prototype/attn_valu_slim/). `1`, the default, runs `attn_slim_wave`: the score dot's"
                 " trip count is made compile-time — it was runtime only because the NEG_DIMHALF injection"
                 " asks for 4 of 8 dims, and LLVM rolled the hot dot to match, at 8 scalar `ds_load_u16`,"
                 " 8 `v_movrels_b32` and 8 `s_waitcnt` per touch — and the warp reduce is folded by hand"
                 " with its own `s_waitcnt lgkmcnt(0)`, because `SIInsertWaitcnts` does not insert waits"
                 " for inline-asm DS ops. 116 issue slots per (row, lane, head) touch become 45"
                 " (prototype/attn_valu_slim/results.md §2) for 1.52-1.57x on the wave (§7). `0` runs the"
                 " prototype's `attn_tile_wave` unchanged. The contract with `1` is bit-identity, not a"
                 " band: same eight products in the same serial order, same five-step butterfly, same"
                 " clamp. Wave-only, like --attn-tile: a one-row body is the fused single-row form under"
                 " either setting, so no decode bit moves, and BOTH spellings are compiled into the binary"
                 " whatever this says, so a paired timing is one window rather than two builds\n"
                 "  --attn-fattn 0|1  which FORM owns a WAVE body's STREAMING rows "
                 "(prototype/attn_wmma/fattn_score_handover.md §5a, engine/attn_fattn_kernels.h). `1` "
                 "launches `attn_fattn_wave` beside `attn_slim_wave`: the WMMA kernel takes the rows §07.3 "
                 "already streams (one 16x16x16 tile per 16 query rows x 16 KV lanes, so the wave's score "
                 "dot becomes one matrix instruction per 16 dims instead of 16 FMAs per touch) and the VALU "
                 "wave keeps the one-pass rows — the split is `fattn_covers`, one predicate both kernels "
                 "call, and the launcher echoes `fattn_rows=n/N` so a covered share is never assumed. `0`, "
                 "the default, is the shipped path untouched: the WMMA form has had its fragment maps "
                 "measured but not its numerics admitted. Wave-only, like the two above; the launcher "
                 "REFUSES (fused plan, null `part`, `--attn-tile 0`, `--attn-touch 0`) rather than "
                 "silently running the shipped form and reporting it as the WMMA one\n"
                 "  --attn-fattn-roll single|two  which SPELLING of that form's score roll runs "
                 "(bet A4-L1, engine/attn_fattn_kernels.h, prototype/attn_percost/A4-BET.md). `single` is"
                 " the SHIPPED roll and the default: one accumulator, `#pragma unroll 1`, a 16-long"
                 " dependent `v_wmma` chain — the same machine code this binary carried before the knob"
                 " existed, which is why the default side is owed byte-identity and gets it checked. `two`"
                 " gives the roll a second accumulator and fetches BOTH tiles of a pair before EITHER"
                 " matrix op issues, so two chains of 8 progress at once and one LDS round-trip pays for"
                 " two WMATAs; it costs +22 VGPR (178 -> 200, spill 0, scratch 0, LDS unchanged at 32,768 B"
                 " per block — A4-L1-gates.md §2) and it RE-ASSOCIATES the score dot, so `o` moves and the"
                 " existing §10.1 cos/meanrel bands judge it (no new band, no `--ack-spill`). Wave-only like"
                 " the flags above, one launch and the same grid either way, and both kernels are in the"
                 " binary whatever this says. `--attn-fattn 0` REFUSES it: naming a spelling of a form that"
                 " does not run would echo a kernel that did not\n"
                 "  --attn-fattn-qstage stage|unstaged8  which SPELLING of that form's staging plan runs "
                 "(bet A6-G1, engine/attn_fattn_kernels.h placement (E),"
                 " prototype/attn_percost/A6-BET.md). `stage` is the SHIPPED plan and the default: Q staged"
                 " in a per-warp 8 KiB buffer, and the epilogue transposing its F32 output through that dead"
                 " buffer 16 rows at a time in 2 passes — declared LDS 32,768 B per block, which on this"
                 " part's 64 KiB is 2 blocks/CU = 6 wf/CU. `unstaged8` reads Q from global at the use site"
                 " (same 16 source bytes, same `bf16 -> f32 -> f16 RN`, so the operand VALUES do not move) and"
                 " gives that whole buffer to the epilogue at FATTN_TRG = 8 rows a pass, 4 passes instead of 2"
                 " — declared LDS 20,480 B = 3 blocks/CU = 9 wf/CU at 185 VGPR, spill 0, scratch 0"
                 " (A6-result.md §1, §3). The two halves are ONE candidate: un-staging Q alone moves no byte"
                 " of LDS, because the epilogue stands on the same buffer either way. Unlike the roll above"
                 " this knob is owed BYTE identity with the default, not a band — it moves staging addresses"
                 " and where an operand is fetched from, never a value or an order — and the device proof is"
                 " prototype/attn_percost/A6-G1-impl.md. Wave-only like the flags above, one launch and the"
                 " same grid either way, all four roll×qstage combinations are real kernels in the binary, and"
                 " the launcher's ATTN_FATTN line reports `q=`/`lds=`/`wf_per_cu=`/`vgpr=` from the kernel that"
                 " actually ran. `--attn-fattn 0` REFUSES it, for the same reason as the roll\n"
                 "  --attn-fattn-pvf16 0|1  what WIDTH of P*V accumulator that form runs (bet A10-F16,"
                 " engine/attn_fattn_kernels.h, prototype/attn_percost/A10-F16-IMPL.md). `0` is the SHIPPED"
                 " accumulator and the default: 16 tiles of 8 F32 = 128 registers a lane, one"
                 " v_wmma_f32_16x16x16_f16 per 16 output dims, and it is HEAD's own machine code, so the"
                 " default side of its A/B is a machine-code compare rather than an argument. `1` is 8 tiles"
                 " of 16 F16 halves = 64 registers a lane for the SAME 256 output dims — half the registers"
                 " for the same coverage, because a half slot holds two dims — filled in llama's exact"
                 " two-issue shape: two v_wmma_f16_16x16x16_f16 per 32 dims, op_sel 0 then 1, two different A"
                 " fragments (the low and high 16 dims of the pair), the same P, the same D block"
                 " (mma.cuh:1553-1566). ENABLER, NOT SPEED: 16 v_wmma issues per round either way and the two"
                 " atoms time the same per unit of coverage (A10-RF-LADDER.md §7), so the expected throughput"
                 " effect is ~0; what it buys is the 64 registers a Q-in-registers-as-F16 spelling has to fit"
                 " through, which cannot be booked at 128+128 registers (D7). NOT owed byte identity — a"
                 " narrower accumulator is a different format — so its judge is the existing §10.1 cos/meanrel"
                 " bands through A4-BET.md, no new band and no widening, and the per-round rescale stays where"
                 " it is because an F16 accumulator discards an add once the addend falls below half a ULP"
                 " (issue_rate_probe/results.md §6). Wave-only like the flags above, one launch and the same"
                 " grid either way, all eight roll×qstage×pv combinations are real kernels in the binary, and"
                 " the launcher's ATTN_FATTN line reports `pv=`/`vgpr=` from the kernel that actually ran."
                 " `--attn-fattn 0` REFUSES it, for the same reason as the roll\n"
                 "  --attn-fattn-wide 0|1|2  whether that wave is the 16-row WMMA block or the A12 WIDE one"
                 " (bet A12, engine/attn_wide_kernels.h, prototype/attn_percost/A12-WIDE-BET.md). `0` is what"
                 " SHIPS and is HEAD's own machine code: the WMMA block takes the rows §07.3 already streams"
                 " (blocks_for_row > 1), attn_slim_wave keeps the one-pass rows, attn_combine merges the"
                 " partials. `1` is not a spelling of that wave but a DIFFERENT CORE: one block covers 32 query"
                 " positions x the 4 heads of ONE KV group — 128 row-slots, 8 wavefronts — stages 64 KV"
                 " positions a round, converts Q to F16 once per block and keeps it in registers, and"
                 " accumulates P*V in F16 (the accumulator halving that makes the shape possible at all). It"
                 " has NO coverage rule and NO lane split, so it owns EVERY row of the body — the rows at or"
                 " below MAXSC included, which is exactly where the shipping route is bit-exact against the"
                 " dumps — and neither attn_slim_wave nor attn_combine is launched for that body, because the"
                 " block normalises and applies §05.6 itself through finish_head. NOT bit-identical at any row"
                 " width and not owed byte identity: a 64-position round re-associates the softmax and the"
                 " weight reaching the accumulator is an un-normalised F16 exp divided at the end, where the"
                 " one-pass route rounds the NORMALISED weight to BF16 before the P*V (§05.5 rule 4). Its judge"
                 " is therefore the existing §10.1 cos/meanrel bands through A4-BET.md — no new band, no"
                 " widening, no loosened check — and the per-round accumulator rescale is mandatory, not"
                 " tidiness: with the fold removed the worst head-row drops its FIRST add at round 1 and the"
                 " accumulator is no longer finite in fp16 by round 1721 of a 64-position round"
                 " (prototype/attn_wide/a12_wide_mirror.py §4; the shipping 16-position arm's own counterfactual"
                 " is round 1 / 13 / 99 worst-median-best, A10-PVF16.md §4). WAVE-ONLY"
                 " like the flags above it, so no decode bit moves under it; `--attn-fattn 0` REFUSES it, and"
                 " the run's own ATTN_WIDE line reports vgpr=/lds=/scratch=/wf_per_cu= from the kernel that"
                 " actually ran, spill included (A12 spends the spill budget on purpose). `2` is the A13 PORT"
                 " of llama.cpp's structure onto that same block (engine/attn_wide_llama_kernels.h,"
                 " prototype/attn_wide/LLAMA-PORT-PLAN.md): a template-stride XOR-swizzled LDS layout, the next"
                 " round's K tile loaded into registers in the middle of the current round and committed to the"
                 " stage only after that round's V is consumed, a steady-state round with NO per-cell bounds or"
                 " mask test in it at all and the comparisons specialised into the one or two tail rounds a"
                 " block can reach (a block spans 32 query positions and lanes_of advances by 1 per row, so its"
                 " causal spread is 31 lanes — under one 64-position round), and the softmax scale folded into"
                 " Q at staging instead of applied to every score element. Unlike `1` against `0`, `2` against"
                 " `1` IS owed byte identity: a swizzle permutes bytes inside a stage row, a prefetch moves"
                 " WHEN a load is issued and not what it loads, a specialised predicate that provably cannot"
                 " fire computes the same value, and SCALE is 2^-4, which is exact. So its adjudicator is a"
                 " byte-compare against `1`, not a band. The ATTN_WIDE line prints form=wide128llama for it,"
                 " which is the behavioural canary: a stale binary cannot price `1` and be tabled as `2`. Still"
                 " WAVE-ONLY, and `2` is the shipped default since A13 (`decode_run.cpp`'s own option field)"
                 " while `1` stays opt-in and moves nothing about `0`.\n"
                 " An MTP ROUND's body is the one wave that does not consult this flag:"
                 " `engine/decode_step.h`'s `kMtpRoundAttnWide` pins a `--mtp-verify` or `--mtp-accept` body"
                 " to the `0` core, because the block's win is measured at PREFILL row counts and the four-arm"
                 " wall A/B (`prototype/mtp_accept/wide_body_ab.sh`) measured it as a large loss at a round's."
                 " A rule that is compiled in has no flag to watch, so the round prints the core it RAN as"
                 " `attn_fattn_wide=` on its `VERIFY2`/`MTPSTAT` line, off the scratch the body was"
                 " configured from — the `gemm=`/`gdn_scan=` precedent. Prefill waves are untouched.\n"
                 "  --attn-split-cap N  an UPPER BOUND on how many lane blocks a wide query row may be cut"
                 " into (bet A9-L-c, prototype/attn_percost/A9-LONGCTX-CENSUS.md §4.2). The lane geometry"
                 " forces `split = ceil(lanes/MAXSC)` blocks per wide row (§07.3); N caps that count, so a"
                 " block's slice becomes `ceil(lanes/min(split,N))` lanes and the split's partial round-trip"
                 " shrinks with it — 105.5 GB/pass of scratch written plus 105.5 GB read back at 32,768 lanes,"
                 " and `attn_combine` was traced reading that at 88 %% of the DRAM roof, which is the one item"
                 " in the census whose rate is measured (A9 §2.3). `0` (the DEFAULT) leaves the geometry's own"
                 " split alone, i.e. today's bits; `1` is REFUSED because one lane block on a wide row is the"
                 " mis-normalisation §07.3 exists to make unreachable; N above the engine's split ceiling is"
                 " inert. NOT bit-identical and cannot be: fewer partials re-associates `attn_combine`'s merge,"
                 " so the existing §10.1 cos/meanrel bands judge it and no band moves (the standing numeric-"
                 " format rule). It binds on a long DECODE row as well as on a wave — the fused `attn_core`"
                 " has the same streaming route — so at N=8 the first rows it moves start at 4,608 lanes. A"
                 " binding cap is REFUSED under `--attn-fattn 0`: the VALU tiled wave stages a whole block"
                 " slice in `s_sc[ATTN_TILE_G][MAXSC]`, so a slice wider than MAXSC is an LDS overrun, while"
                 " the WMMA wave stages 16 lanes a round and absorbs a wider slice as more rounds. Every run"
                 " prints `SPLITCAP split= cap= lanes= natural= rows= span=` once per distinct geometry — the"
                 " behavioural canary an A/B reads to prove the binary carries the knob\n"
                 "  --gdn-spell ship|unroll4|fused  which SPELLING of the sequential recurrence's state"
                 " walks a `gdn_rec` launch runs (`engine/gdn_rec_unroll4_kernels.h`, "
                 "prototype/gdn_rec_isa/results.md §10, decision D32). `unroll4` is the SHIPPED"
                 " DEFAULT: the same operations in the same order with `#pragma unroll 4` on both walks"
                 " — BIT-identical (zero elements of `o` and of the state moved in 4/4 harness cases),"
                 " 7.7%% fewer slots, and 1.149-1.158x at launch level over a resident 1 MiB state; the"
                 " in-situ A/B came back 1.153x on the family row at 4096 lanes (1.100x at 1024), so"
                 " THIS IS THE SHIPPED DEFAULT and `ship` is kept as its A/B side. `ship` is the"
                 " pre-D33 shipped `rec_kernel`, the form every pin was first pinned through. `fused`"
                 " (D50) is the unroll4 walks with the q/k norm prologue FUSED into one pass over"
                 " shared memory — 3 barriers instead of 8, −32 of 1,406 slots/token-step at UNCHANGED"
                 " registers (59 VGPRs, zero spill) and +128 B of LDS (engine/gdn_rec_fused_kernels.h,"
                 " prototype/gdn_rec_isa/results.md §11, progress/2026-10-21/performance-status-2026-10-21.md §19). It is"
                 " BIT-identical too (ndiff 0 on `o` and on the state, 4/4 cases) and it is a KNOB,"
                 " NOT a default: the paired in-situ A/B on the `gdn_rec` family row decides whether"
                 " the default moves. Inert under `--gdn-scan collapse`, which launches a different"
                 " kernel: a bare `collapse` run is legal and echoes `gdn_spell=inert-under-collapse`,"
                 " while naming this flag alongside it is REFUSED rather than echoed with a spelling"
                 " that did not run. All spellings are compiled in either way, so a paired wave timing"
                 " is one window\n"
                 "  --gdn-norm inline|hoist  WHERE the looped recurrence's per-token k-row and q-row"
                 " L2-normalisations are computed (bet G1, engine/gdn_rec_hoist_kernels.h,"
                 " prototype/gdn_walk/BET.md §3). `inline`, the DEFAULT, is the shipped "
                 " `rec_loop_kernel`: every token-step of every head block reduces its own k and q rows"
                 " with `block_sum_pair_f32` before the state walk may start, which is 122 of 1,327"
                 " slots/token-step (9.2 %% of the bill) and three `s_barrier`s on the chain between token t and"
                 " token t+1. `hoist` submits TWO launches per span: a pre-pass of one block per"
                 " (token-step, head) computes those two normalised rows plus `expf(g)` and `beta`, and"
                 " the walk reads them, so its per-token step is read-the-operands, walk, write-o with"
                 " no LDS, no barrier and no cross-lane reduction at all. The contract is BIT-IDENTICAL,"
                 " not a band: the pre-pass calls the same reduction at the same `blockDim.x` and the"
                 " walk's arithmetic text is unchanged (`prototype/gdn_walk/check_hoist_transcription.py"
                 " checks the source and `walk_isa_diff.py` the compiled FP-op census). It costs 2.4 GB"
                 " of stage traffic per 4,096-lane pass, which is why it is a KNOB and why BET.md §3's"
                 " pre-registered lines on the `gdn_rec` family row decide whether the default moves."
                 " Wave-only like --gdn-loop: a one-row body takes the per-token kernels with their own"
                 " prologue, `--gdn-loop 1` is REFUSED with `hoist`, and so is `--gdn-scan collapse`,"
                 " whose kernel keeps its own prologue. Both forms are compiled in either way, so a"
                 " paired wave timing is one window.\n"
                 "  --gdn-geom block|wf4|wf2|reg16  the BLOCK SHAPE of the looped recurrence (bets G2 "
                 "and G4, engine/gdn_rec_wf_kernels.h and engine/gdn_rec_reg16_kernels.h, "
                 "prototype/gdn_walk/BET.md §10 and §12). `block`, the DEFAULT, is"
                 " the shipped `<<<16,128>>>` walk: four wavefronts per block, one state column per"
                 " thread, the row norms reduced by `block_sum_pair_f32` (3 barriers + an LDS round-trip"
                 " per token). `wf4` is one WAVEFRONT per block — `<<<16,32>>>`, 4 state columns per"
                 " lane, LDS = 0, no `__syncthreads()` because a 32-lane block and a wavefront are the"
                 " same set, and both row norms are warp butterflies over the 4-per-lane pieces of the"
                 " rows the lane already loads. That last part is the point of the geometry and the"
                 " reason it costs ZERO staging bytes where G1 paid +32 KB/token/layer. It is NOT"
                 " bit-identical and cannot be: the norm sums, the state dots and the `o` dot all"
                 " re-associate (the `o` dot moves to the closed form Σq̂S_new = Q_SCALE·ι·(γΣqS_old +"
                 " κ(Σkq)dv) because nothing can broadcast q̂ into the second walk without shared"
                 " memory). The admission is the EXISTING §10.1 aggregates — cos >= 0.9995, meanrel <="
                 " 5 %% — through `engine/gate.py` on the pinned fixtures: no new band, no widening, no"
                 " --ack-spill (`prototype/gdn_walk/g2_mirror.py` is the CPU-side size of the move)."
                 " `wf2` is the same kernel at 2 columns per lane, `<<<32,32>>>`: the split leg, which"
                 " prices the wavefronts-for-work-per-wavefront trade the header sizes. `reg16` is llama's"
                 " own thread map under this recurrence (G3 §6, §43 of the ledger):"
                 " `<<<dim3(NH,D/16)=128 blocks, dim3(32,4)=128 threads>>>` = 512 wavefronts per launch ="
                 " 6.40 wf/SIMD against the shipped 0.80, 8 lanes per state column and 16 state elements"
                 " per thread HELD IN REGISTERS for the whole span, so the state is read once at span"
                 " entry and written once at span exit and moves zero times per token. The identity"
                 " wavefronts x 32 x elements == 262,144 is a static_assert in the header, and the"
                 " compiled shape is checked by `walk_isa_diff.py --g4`: the token loop's only global"
                 " store is `o`. The norms stay in the walk as 8-lane butterflies, so nothing is staged"
                 " (where G1 paid +32 KB/token/layer). Same admission as `wf4` — the existing §10.1"
                 " aggregates, no new band — and the state layout does NOT change, so every consumer of"
                 " `engine/state.h`'s [i][j] block keeps working. One launch per"
                 " span on every side, billed on the SAME `gdn_rec` row (a win that moved the"
                 " milliseconds to another row would not be one, which is what §40 is for). Wave-only: a"
                 " one-row body takes the per-token kernels whatever this says, and `--gdn-loop 1`,"
                 " `--gdn-scan collapse` and `--gdn-norm hoist` are each REFUSED with a non-block"
                 " geom.\n"
                 "  --gdn-loop N  how many token-steps ONE `gdn_rec` launch carries (D53/D54, "
                 "engine/gdn_rec_loop_kernels.h). `1`, the default, is the shipped one-launch-per-token"
                 " form: 18 launches per token, 73,728 for a 4,096-token prompt, at the 1.72 us"
                 " measured submission floor (§11). `512` walks a whole wave body inside one launch —"
                 " 18 x bodies, i.e. 144 launches for the same prompt — with the SAME per-token"
                 " arithmetic (the kernel is `rec_fused_kernel`'s body with the token index moved"
                 " inside it, transcribed and checked by check_recloop_transcription.py in"
                 " prototype/gdn_rec_isa), the same grid `<<<16,128>>>` and the same state"
                 " addresses, so it is a launch-structure change and not a numerics one: spans"
                 " 0/1/3/16 came back `ndiff_o_vs0=0 ndiff_state_vs0=0` (engine/performance-status.md"
                 " §23). It is a KNOB, NOT a default — the loop bills +156 of 1,374 slots/token-step"
                 " for recomputing its row addresses, and the paired in-situ A/B on the `gdn_rec`"
                 " family row decides whether the default moves (D54, in D32 → D50's order). A span"
                 " below the body size means what it says (ceil(rows/span) launches, span-invariance"
                 " proved on CPU), and a ONE-ROW body takes the per-token loop whatever this says, so"
                 " no decode bit moves under it. Inert under `--gdn-scan collapse`, which already"
                 " submits one launch per body, and loopable only in the `fused` spelling, which is"
                 " the one body that has been transcribed with the token index inside it: naming this"
                 " flag alongside either of those is REFUSED rather than priced twice\n"
                 "  --gdn-scan seq|chunk  which §04.4 scan a GDN body runs (P9, §04.7,"
                 " progress/2026-10-14/p9-phase1-2026-10-14.md). `seq`, the default, is the shipped recurrence: ONE"
                 " launch per token per layer, 18,432 launches per 1024-token wave, and the form the"
                 " decode pins are pinned through. `chunk` runs the chunked scan — intra-chunk work in"
                 " one launch per body, the inter-chunk carry in a second one that parks the state"
                 " strip in LDS instead of round-tripping it per token (72 launches for the same wave)."
                 " Wave-only: a one-row body is `seq` under either setting, so no decode bit moves."
                 " Band-gated, never bit-gated: a chunked dot's association is not the sequential dot's"
                 " (§10.1), and the budget it was admitted against is Phase 0's §2.1/§2.3 lines as"
                 " measured by prototype/gdn_chunk/phase1.py. It REFUSES rather than falling back, for"
                 " the same reason the tiled attention core does\n"
                 "  --gdn-chunk N  the chunk size in tokens, 64. Smaller is refused with the reason:"
                 " C=16 is measured and outside §2.3's per-row ceiling (1-cos 3.388e-4 > 3.2e-4), and"
                 " C=32 is inside that budget but unbuilt — the kernel is instantiated at one C, and a"
                 " second instantiation of a form already 0.3 %% inside a stage line is not added"
                 " unasked. Larger is unbudgeted: 16/32/64 are the sizes the CPU judge sweeps"
                 "  --proj-plan split|packed  how a ONE-ROW body GROUPS its projection launches (route"
                 " D1). `packed` (the default) folds each layer's input projections into ONE launch — they"
                 " read the same row of `norm_f32` and each writes its own tile, so no weight is copied —"
                 " for 48 per token (`prototype/proj_gemv/results.md` §D1-W4: 2.4942 vs 2.8244 ms at"
                 " identical geometry). `split` is §07's bill: one launch per projection, 114 per token,"
                 " i.e. what shipped before D1, kept so a pre-D1 log replays byte for byte. Requires"
                 " `--proj-shape unroll8`; the other pair is refused rather than echoed. A wider body"
                 " is untouched by it: a wave's projections are already one launch per projection\n"
                 "  --proj-shape shipped|unroll8  which kernel and block geometry a one-row projection"
                 " launch uses. `unroll8` (the default) is `engine/proj_group_kernels.h` at rows/block 8 and"
                 " 1024 threads with a compile-time word walk (§D1-W4's winning arm, 2.4693 vs 3.2692 ms in"
                 " the same window), bit-identical to the shipped form because the word order, the ladder"
                 " and the rounding are unchanged. `shipped` is `proj_gemv_tokens<IN,1,false>` at"
                 " rows/block 32, i.e. HEAD's machine code, and must be asked for together with `--proj-plan"
                 " split`. If the device or the width cannot offer `unroll8`, the run falls"
                 " back to `shipped` and prints `PROJSHAPE-FALLBACK` instead of pricing one kernel twice\n"
                 "  QWEN_CAST_FOLD=0|1  (environment, not a flag) route S1 item (1): fold the per-token"
                 " BF16->F32 activation widenings into the kernels that consume the result and delete the"
                 " `cast` family's 97 launches. `0` (the default) is HEAD's behaviour: `kernels_cast.cpp`"
                 " submits each widening. `1` makes each consumer stage the BF16 tile itself and widen in"
                 " registers, which is bit-identical because `hip_bfloat16 -> float` is exact"
                 " (engine/mlp_xbf16_kernels.h). A ONE-ROW body folds all six sites; a wider body folds"
                 " only the final-norm one. It is read from the environment so `engine/gate.py` can judge"
                 " the folded path by inheriting it, with no edit to the gate; the run echoes "
                 "`CASTFOLD on=... plan=... folded=...` beside `LAUNCH`\n"
                 "  QWEN_COPY_FOLD=0|1  (environment, not a flag) route S1's follow-on: fold the `copy`"
                 " family's 19 submissions into the kernels that already touch the same bytes. `0` (the"
                 " default) is HEAD's behaviour: one D2D copy of the tied embedding row and one D2D"
                 " roll-back of each GDN layer's conv window, 19 per token. `1` points the layer-0 glue"
                 " launch at the table row the embedding copy used to copy, and runs"
                 " `engine/gdn_conv_roll_kernels.h`, whose conv writes its rolled window in place at the"
                 " columns its own lane just read. Both are data movement rather than arithmetic, so the"
                 " admission is a byte-compare of logits AND the whole state table, not a band; a roll"
                 " written to the wrong row would move every later token's conv and could not hide. A"
                 " ONE-ROW body folds both sites; a wider body folds neither (its embedding lookup is one"
                 " gather over the tile, and its conv end condition belongs to one token-block while the"
                 " others are still reading the window). Read from the environment for the same reason as"
                 " `QWEN_CAST_FOLD`, and echoed as `COPYFOLD on=... plan=... folded=...`\n"
                 "  --mtp-draft 0|1  §09's MTP draft block as a READ-ONLY SIDE COMPUTATION"
                 " (`engine/mtp_draft.h`; `QWEN_MTP_DRAFT=0|1` does the same from the environment, and"
                 " the flag outranks it). `0` is the DEFAULT and runs nothing: no allocation, no"
                 " submission, no extra line of output, so an off run is the pre-MTP binary. `1` runs one"
                 " draft round per generation step as the round schedule says — 13 kernel launches and 1"
                 " D2D for the block, 1 `lm_head_full_xbf16` and 1 logits D2H for the proposal, 16"
                 " submissions a round, all of it reading and writing only its own arena, cache and"
                 " logits row. It therefore owes a byte-compare, not a band: `sampled.txt` and every"
                 " `--state-fp` row must be identical with it on and off, and if they are not the draft is"
                 " not a side computation. Prints `DRAFT round= draft_pos= adjudicated_by_pred_pos="
                 " y_next= proposal= top_logit= lanes=` per round, `MTPACCEPT settled_pos= proposal="
                 " target_pred= accept=` when a later step settles one, `DRAFT-SKIP`/`DRAFT-UNADJUDICATED"
                 "` for rounds it could not run or could not settle, and one `MTPDRAFT` census line. It"
                 " is NOT a verify body, an accept loop, or a rollback path: no candidate row is fed to"
                 " the target, nothing is snapshotted, and the target keeps running plain greedy. Refused"
                 " with --time-stages/--time-waves, which would charge the draft's submissions to"
                 " whichever stage-clock segment preceded them\n"
                 "  --mtp-verify 0|1  §09's batched 2-row VERIFY body as a read-only side pass"
                 " (`engine/mtp_verify.h`; `QWEN_MTP_VERIFY=0|1` does the same from the environment, and"
                 " the flag outranks it). `0` is the DEFAULT and runs nothing: no allocation, no"
                 " submission, no extra line. `1` runs, after every generation step, ONE body of two rows"
                 " at positions q and q+1 over the populated KV — the shipping `submit_wave_body` at"
                 " width 2 — on a SHADOW mixer state RESTORED from a per-step checkpoint of the GDN state"
                 " and conv window (repointing the shadow is not seeding it: without the restore the body"
                 " reads memory it never wrote, which is a different operand rather than a worse rounding),"
                 " then prices both rows with ONE pass of the batched tied head"
                 " (`lm_head_batched_xbf16`) and ONE logits download, and argmaxes both with §03.4's rule."
                 " The target's own state is never the shadow, so there is nothing to commit: this is the"
                 " body and its cost, NOT the accept loop, and it changes no emitted token (that loop is"
                 " the next increment). Under --mtp-route hand (the default) it forces the hand GEMM and"
                 " the sequential GDN scan for its own body, because a one-row decode step runs those by"
                 " rule and row 0 has to reproduce them bit for bit; --mtp-route amortised leaves the"
                 " run's own resolved route in place (see below). Prints `VERIFY2 step= anchor_pos= fed="
                 " candidate= row0= row1= row0_top= row1_top= row0_eq_step= head_passes= d2h="
                 " logits_bytes= checkpoint_bytes= guard= mtp_route= gemm= gdn_scan=` per"
                 " round, writes `verify2_<step>.bin` (F32, [2][vocab]) beside the other dumps when --out"
                 " is given, and ends with a `VERIFY2DONE` census line carrying the per-round checkpoint"
                 " bytes, the bytes left above `prefix_len`, and the restore witness. Refused with"
                 " --session (no step list) and with --time-stages/--time-waves (the body is not a family"
                 " of §07's bill, so the clock would misbill it; its cost is a paired wall-clock A/B, see"
                 " prototype/mtp_verify/ARMS.md).\n"
                 "  --verify-candidate ID|sampled  the id fed as row 1 at every anchor. Without the flag"
                 " the candidate is the token the step itself was fed: a real second row at the next"
                 " position, but a degenerate draft. `sampled` uses the token the step PREDICTED, i.e. the"
                 " true next token, which makes ONE probe run directly comparable against a plain run's"
                 " own `logits<step>.bin` files at every anchor — that pairing is `prototype/mtp_verify`'s"
                 " row-1 witness, and it is a reproduction, not a filename. Any decimal id is a genuinely"
                 " wrong candidate, which is the negative: row 1 must move and row 0 must not. Refused"
                 " unless --mtp-verify is on — a flag accepted by silence is the defect this session has"
                 " now recorded three times.\n"
                 "  --verify-guard 0|1  witness the verify body in BYTES rather than in metrics"
                 " (`engine/mtp_verify.h`'s operand/guard section); 0 is the default and submits nothing"
                 " extra. `1` downloads, per round, what the step was about to read and what the shadow"
                 " body actually held, byte-compares them, poisons the one KV lane above `prefix_len` that"
                 " the body is entitled to write, and re-reads every live buffer afterwards. Prints"
                 " `VERIFYGUARD step= pos= poison_lane= operands_same= leak_clean= poison_written="
                 " poison_regions_unwritten=/poison_regions poison_survivors= poison_lane_bytes="
                 " selftest_bites= operand_first_diff= leak_first_diff= leak_extent=` per round."
                 " `selftest_bites=1` is"
                 " the control that says the comparison can fail at all (the step had to move the mixer"
                 " state), and `poison_written` is per-region rather than a byte count because a lane that"
                 " WAS written still contains a few 0xA5 bytes by coincidence. `leak_extent` counts the"
                 " bytes that moved in each buffer and names the lanes they moved in, which is what tells"
                 " 'the body re-staged its own lane' from 'the body wrote all over the cache'. Off during"
                 " timing on"
                 " purpose: it is tens of MiB of downloads per round, so `prototype/mtp_verify/ARMS.md`"
                 " runs it off and the admission gate runs it on. Refused unless --mtp-verify is on.\n"
                 "  --mtp-accept 0|1  k=1 SPECULATIVE DECODING on the draft block's proposals"
                 " (`engine/mtp_accept.h`; `QWEN_MTP_ACCEPT=0|1` does the same from the environment)."
                 " Default 0, and off means the side state is never allocated and the generation loop is"
                 " the one that shipped without this flag — the inertness claim is a byte-compare, not a"
                 " band. On, one round replaces one step: the round feeds the funded 2-row body"
                 " [cur at p, candidate at p+1], prices both norm rows in ONE pass over the tied head,"
                 " and accepts iff argmax(row 0) == candidate, decided ON DEVICE and read back as 4 ints"
                 " (16 B) rather than a vocabulary. An accept emits both tokens and restores nothing; a"
                 " reject emits the verifier's token and restores the body's per-layer mid-round"
                 " checkpoint (20,201,472 B in two bulk copies), which is the state a reject needs and is"
                 " why the recurrence is pinned to --gdn-loop 1 (§4b of prototype/mtp_accept/"
                 " ACCEPT-BUILD.md). Refused with --time-stages/--time-waves, --replay/--feed, --session,"
                 " --sizes-only and --mtp-verify, because a round is not a step and those flags name a"
                 " per-step object it does not own; and it needs --mtp-draft 1, since the candidate must"
                 " come from the draft block rather than a guess. Prints one `ACCEPT round= anchor= fed="
                 " cand= cand_pos= accept= verifier= bonus= emitted= pos=-> restored_bytes=` line per"
                 " round, one `ACCEMIT step= pos= token=` line per EMITTED token (§07.5's stop rule runs"
                 " per emitted token, including the accepted candidate), and a closing `MTPSTAT` line"
                 " with rounds / accepted / rejected / emitted / accept_rate / mean_emit / body_rows /"
                 " captured_bytes / restored_bytes / status_bytes / split_subs. Every run — this flag or"
                 " not — also prints a `DECWALL mode= prefill_ms= decode_ms= bodies= rounds= plain="
                 " accepted= rejected= suppressed= emitted= stop= ended= ms_per_token= tok_per_s= census=` line:"
                 " the wall of THIS run's generation window divided by the tokens that window emitted,"
                 " which is the only decode-rate denominator an accept run has, because --steps counts"
                 " rounds and the loop can end early. It prints NO-RATE rather than a rate over counters"
                 " that disagree. With --out it writes"
                 " `emitted.txt` (one token per line) instead of `sampled.txt` + `logits<N>.bin`, because"
                 " a round has two logits rows and no single top-logit value, and inventing one for a"
                 " five-column row would be a fake number. Note that ON, --steps counts ROUNDS, so a run"
                 " reaches further than --steps tokens; the last lane of the request has room for one row"
                 " only, so there the driver pays a plain step and prints `ROUND-SKIP reason=context_limit`"
                 " rather than refusing the run.\n"
                 "  --mtp-sched loop|every  which round sequence --mtp-draft 1 walks. `loop` (the"
                 " default) reproduces `speccheck/mtp.py`'s `greedy_mtp(k=1)` round geometry on a plain"
                 " greedy stream — draft at q+1, next round at q+2 after an accept and q+1 after a"
                 " reject, the rejected round's draft-cache lane truncated — which is the geometry the"
                 " carried accept rate of 0.7329 was measured on (`prototype/mtp_gatezero/draft_survival.py"
                 "`, whose `verify=\"steps\"` this is the device counterpart of). `every` drafts after"
                 " every step, so its draft cache holds every position rather than the accepted rounds'"
                 " — more context than the loop ever gives a draft, and the gap between the two numbers"
                 " is that difference measured rather than assumed. Parsed or throws; refused if"
                 " --mtp-draft is off\n"
                 "  --mtp-route hand|amortised  which weight walk and which GDN scan the MTP body runs"
                 " (`engine/mtp_verify.h`'s Route). `hand` is the DEFAULT and is today's launch sequence"
                 " byte for byte: the body forces the hand GEMV and the sequential scan so its row 0 is"
                 " the kernel chain the one-row step ran, which is what makes `row0_eq_step` an admissible"
                 " ask. `amortised` leaves the run's own resolved route in place — the route a wide body"
                 " takes everywhere else — which `prototype/mtp_accept/ROW-SCALE.md` measured at 1.09-1.13"
                 " of a step against the forced route's 2.085, i.e. the route the round has to be priced"
                 " on. It is NOT owed bit-identity with the step, gains no new band, and widens no gate"
                 " threshold: the existing gates judge it and its divergence is reported"
                 " (`prototype/mtp_verify/rowdiff.py`). Refused when neither --mtp-verify nor --mtp-accept"
                 " is on (there is no body to route), and refused together with `--gdn-scan collapse`"
                 " under either, because the round's mid-round checkpoint is a one-token-step sequential"
                 " recurrence's end state. Echoed as `mtp_route=` beside `gemm=`/`gdn_scan=` on every"
                 " `VERIFY2` line, on `VERIFY2DONE`, and on `MTPSTAT` — the latter two name the route that"
                 " ACTUALLY ran, not the flag, so a switch that never took cannot report the arm it did"
                 " not run\n"
                 "  --mtp-cap N  lanes allocated for the draft block's own 1-layer KV cache (512 by"
                 " default). The one-pass attention form refuses beyond `mtp_attn_max_lanes()`, and a"
                 " round past the ceiling is counted as `DRAFT-SKIP reason=...` rather than served a"
                 " shorter cache, because a silently shorter run reads as a lower accept rate\n"
                 "  --mtp-draft-k 1|2|3|4  run the draft block k ROWS per drafted position, feeding each"
                 " row's own proposal back as the next row's `y_next` and the block's own output row `x_d`"
                 " as its `h` — there is no target row behind a token the target never fed (llama.cpp's"
                 " draft-mtp feeds the same two objects back, `engines/llama.cpp/common/speculative.cpp:"
                 "1770-1783`). Default 1 = the shipped single proposal, and naming `1` changes no device"
                 " work at all while still printing the ladder fields, which is what makes `k=1` a control"
                 " rather than an assumption. Each row is a whole round — 16 submissions including its own"
                 " head pass and vocabulary D2H — so k costs k x 16 per drafted position. Every row"
                 " appends a draft-cache lane at its own position and all but the first are dropped again,"
                 " so the cache stays the one-lane-per-visited-position sequence the shipped path builds"
                 " (lanes must stay ascending in position: `mtp_attn_row` treats the last as newest)."
                 " Prints one `DRAFTK pos= anchor= k= t1= c1= hit1= ... t4= c4= hit4=` line per drafted"
                 " position — every field on every line, `-1` meaning that field does not exist (past the"
                 " run's last step for a `t`, refused or not asked for on this `k` for a `c`), and one"
                 " `MTPDRAFTK` census line whose `identity=` asserts rows = anchors + extra rows. This"
                 " MEASURES the second position's accept rate on the side pass; it is not speculation and"
                 " it changes no emitted token. Refused with --mtp-accept (its body is 2 rows and would"
                 " verify one candidate while paying for k), --session (positions re-anchor at every"
                 " `feed`, so a row from one turn would be settled by a token of the next), --replay (the"
                 " targets must be this run's own continuation, not a transcript) and --time-stages (k"
                 " rows per anchor has no integer per-step `mtp` bill)\n"
                 "  --mtp-dense 0|1  on an accept, also stage the draft-cache lane for the accepted"
                 " candidate row (default 0). A round appends ONE lane but consumes TWO positions on an"
                 " accept, so the shipped cache skips a position every time one is accepted and the"
                 " rope distance between neighbouring draft lanes is 2 where the block expects 1. A"
                 " dense lane costs 9 submissions and no head pass (`engine/mtp_draft.h:draft_lane`) and"
                 " changes the PROPOSED token only: the body, the verdict and the emitted stream rules"
                 " are untouched, so the accept-rate test is the whole test. Refused without"
                 " --mtp-accept (no round, no skip) and with --time-stages (no integer per-step bill)."
                 " Printed as `lane_appends=` on `MTPDRAFT`\n"
                 "  --mtp-prompt-lanes N  seed the draft cache with the last N prompt positions before"
                 " generation, one lane per position (default 0 = the shipped empty cache; llama.cpp's draft"
                 " sees the prompt because its catch-up runs over the prompt ubatches, and this is our"
                 " counterpart). The norm rows those lanes need do not survive the wave split — `sc.norm`"
                 " keeps only the LAST wave's 46 rows of a 4,142-token prompt — so the rows the fan needs"
                 " are retained out of the scratch during the wave (one D2D per intersecting wave, 2 KiB a"
                 " row; `engine/prefill.h::NormTailRetention`). N is clamped to the retained rows and to"
                 " what the cache ceiling leaves after reserving room for this run's own rounds, and the"
                 " fan that ACTUALLY got staged is printed on `MTPPROMPT` with both clamps, the old"
                 " `last_wave_rows` one beside them, because a silently smaller population is this repo's"
                 " recurring failure. The hard ceiling is the draft attention's one-pass form"
                 " (`mtp_attn_max_lanes()` = 512 lanes), so a whole 4k prompt cannot be seeded at all and a"
                 " wide fan costs 9 submissions a lane once per request. Refused with --time-stages for the"
                 " reason above, and with --session, which has no single prompt wave to retain\n"
                 "  --mtp-pos-shift N  add N to every rope position the draft block sees, uniformly over"
                 " the whole cache (default 0). §09.1 puts the position of the DRAFTED token on the row;"
                 " llama.cpp puts the position of the token the row FEEDS there, one less"
                 " (`engines/llama.cpp/common/speculative.cpp:1830`). Because `mtp_attn_row` rotates q and"
                 " every cached k with the same rope, a score depends only on the DIFFERENCE of positions,"
                 " so a uniform shift must be an exact null — this knob exists to measure that rather than"
                 " assume it. Printed as `pos_shift=` on `MTPDRAFT`\n"
                 "  --mtp-meter 0|1  per-phase milliseconds of the MTP ROUND this run is in"
                 " (`engine/mtp_meter.h`). Default 0, and off means every guard in the engine reads one"
                 " null pointer: no allocation, no added wait, no line, so a flag-off run stays"
                 " byte-identical to a binary without the instrument (verified as an emitted-token and"
                 " `MTPSTAT` diff, not assumed). On, it prints one `MTPMETER role= phase= ms_total="
                 " runs= ms_per_round= status= sync_added= note=` line per phase, plus a derived"
                 " `overhead` line and a `summary` line, and appends `meter=ok rounds="
                 " sum_ms_per_round= wall_ms_per_round= coverage=` to this run's own `RESULT` line, which"
                 " stays the last line of stdout. Two shapes, because the two rounds are different machines:"
                 " `--mtp-verify 1 --mtp-meter 1` meters the read-only side pass (snapshot take, shadow"
                 " seed, witness, lane save, body, lane revert, head, logits D2H, argmax), and"
                 " `--mtp-draft 1 --mtp-accept 1 --gdn-loop 1 --mtp-meter 1` meters the accept loop (body,"
                 " head, verdict kernel, verdict readback, accept path, rollback, lane bookkeeping, the next"
                 " round's draft), with the accept and reject paths as separate phases carrying separate"
                 " run counts. EVERY boundary is labelled: a phase whose close is a call that already"
                 " blocks today prints `sync_added=0` and bounds itself honestly; a phase that is pure"
                 " async submissions prints `sync_added=1` because bounding it needs a device wait the"
                 " shipping path does not perform. What such a wait costs is the OVERLAP it removes, not"
                 " the wait itself — an empty `hipDeviceSynchronize()` is sub-microsecond on this part, the"
                 " run measures its own floor and prints it on the instrument line as `idle_sync_us=` — so"
                 " a `sync_added=1` figure is a bound inflated by serialization, and a phase whose own wait"
                 " fails is reported as `meter=sync_failed` rather than as a zero. The"
                 " accept loop's mid-round checkpoint is printed `status=not_metered` rather than guessed"
                 " at: it is 2 extra submissions per GDN layer INSIDE the body, and a boundary for it would"
                 " need 18 waits inside the call whose pipelining is the thing being priced; it is"
                 " measurable instead as role=verify's `body_with_checkpoint` minus `body` under"
                 " --mtp-midsplit 1. `coverage` is the measured phases over the round's wall, and the"
                 " harness refuses to tabulate a run whose coverage is outside 0.8-1.2, because a phase"
                 " list that does not account for the window is a list with a phase missing. Refused"
                 " without --mtp-verify or --mtp-accept (no round to meter, and a plain step is"
                 " --time-stages' subject), and refused with --time-stages/--time-waves: the stage clock"
                 " drains the queue at §07 family boundaries to price a STEP and this drains it at phase"
                 " boundaries to price a ROUND, so armed together each serializes the other's subject."
                 " What each phase can under- or over-report is written up in"
                 " `prototype/mtp_accept/METER-PLAN.md`; that file, not these milliseconds, is what B1 and"
                 " B2 are sized from\n"
                 "  --dump-proj-plan  print the projection plan one row-priced body would submit (PROJPLAN"
                 " plus one PROJJOB line per tensor) and exit WITHOUT loading the model or touching the GPU,"
                 " so a geometry claim can be checked while another agent holds the device lock\n"
                 "  --time-waves After the run, replay its PROMPT wave from a fresh state at two"
                 " submission granularities — the production shape, queued with one wait at the logits"
                 " (burst), and one wait at every family boundary (stage) — and attribute the serialized"
                 " wave's wall across the launcher families (`engine/stage_clock.h`'s wave unit)."
                 " Prints `WAVEPASS`/`WAVEPARITY` per pass, one `WAVE gran=stage family=` row per"
                 " family (`ms` the family's share of the SERIALIZED wall, `share_pct` that as a"
                 " percent of it, `launches` sites as the marker saw them, `rows_total` the launcher's"
                 " own width sum where one exists, `plan_launches` the derived bill, `counts_match`"
                 " all three agreeing), a `TOTAL` line reconciling the family sum against the pass wall"
                 " (`ungaged_ms`, `within_tol`), a `GROUP` line with `matmul_pct` and `gdn_rec_pct`, and"
                 " `WAVEGAP` with the serialized-vs-burst distance. Timing instrument: its numbers are"
                 " a measurement, so they belong to a clean window, and the run's own outputs and"
                 " `RESULT` line are unchanged by it\n"
                 "  --time-stages After the run, replay its generation steps from a fresh state at"
                 " three submission granularities — one wait per step (burst), per family boundary"
                 " (stage), per submission (kernel) — and print per-stage wall time next to the"
                 " pipelined total (`engine/stage_clock.h`). Timing instrument: the numbers it prints"
                 " are a measurement, so they belong to a clean window, and the run's own dumps are"
                 " unchanged by it\n"
                 "  --replay  feed each generation step the caller's token (teacher forcing; also"
                 " leaves §07.5 unarmed)\n"
                 "  --no-stop leave §07.5 unarmed in a free run too (depth smoke only)\n"
                 "  --state-fp D2H the §08 state at the end of the prompt and at the end of the run"
                 " and print one STATEFP row per pinned fingerprint row (judged by engine/gate.py"
                 " section (g))\n"
                 "  --prefill-logits  `last` (default) samples only the prompt wave's last row, which"
                 " is all §06.5 uses; `all` dumps every row's logits, which is what the wave's"
                 " byte-identity gate compares against\n"
                 "  --dump-attn-rows  also write the attention block's own per-row output tiles"
                 " (pre-gate and gated) for every wave body into --out, one ATTN_DUMP line per body"
                 " naming the core and its full CoreArgs. An instrument for attributing a cross-row"
                 " dependence to a stage; with the flag absent the binary is byte-identical.\n"
                 "  --wave-rows N  rows in the wave scratch (default: the engine's kMaxWaveRows)."
                 " `1` forces one-row glue/MLP launches for every prompt token: the P4b comparison"
                 " run, whose launch counts must differ from the default's and whose outputs must"
                 " not (see engine/launch_stats.h)\n"
                 "  --mlp-map W  which weight walk the MLP's gate/up launches use (Q10, Q10b)."
                 " `auto` (the default) lets the body's row count pick, per body: a one-row body"
                 " walks `contig`, a wave walks `interleave`, because the two paths measurably"
                 " prefer opposite walks (`engine/results.md` §Q10b). `interleave` fetches both"
                 " weight streams' words side by side; `contig` walks the gate row end to end and"
                 " then the up row. The two are bit-identical by construction and proven so on"
                 " device (`prototype/mlp_gemv/run.py --mapping-identity`), which is what makes the"
                 " A/B a timing question and not a numerical one. Naming `interleave` or `contig`"
                 " pins that walk for every body and prints `MLPMAP gate=W` (naming `auto` prints"
                 " `MLPMAP gate=auto`; a run that names nothing prints nothing, so old logs still"
                 " diff clean) - that is the A/B instrument. `down` has one weight stream and is"
                 " unaffected by this flag\n"
                 "  --session resident feed/gen/reset/stats/quit protocol for engine/chat.sh\n");
}

// ---- `--state-fp`: the §08 state, made observable ----------------------------------------------
//
// WHY. G5 (§10) is "prefill `T` then decode `D` equals one prefill of `T+D` (states, logits,
// sampled ids)", and §08.6's four invariants are what "states" means. Until now none of it was
// checkable, because the engine never looked at its own state — `gate.py` could gate ids and logits
// and nothing else (Finding F4 of the prefill plan is the same observation). This block D2Hs the
// §08.1 tuple at the two end conditions a fixture has — after the last prompt token, and after the
// last generation step — and prints one row per pinned fingerprint row.
//
// WHAT THE COLUMNS MEAN, and which code owns them. `tools/state_fp.py::fp_row` is the definition:
// [Σx², max|x|, mean x, Δ²], the parts of one logical state pooled as one multiset, Δ² taken
// part-wise over the common extent of the append axis, and no previous state meaning Δ² = Σx²
// (§08.4 zeroes a fresh request). The fixtures' `state_fp` / `conv_fp` / `kv_fp` arrays were pinned
// with that function, so this block is a mirror of it, not an independent definition — and
// `engine/gate.py` section (g) exists to prove the mirror held (its identity case recomputes the
// pinned arrays with `fp_row` itself and requires bit-equality).
//
// THE ONE DELIBERATE DIFFERENCE: accumulation happens in F64, where `fp_row`'s is F32 (the pins were
// made that way and must not move, §10.5). Sums of 2^21 squares then differ by ~1e-7 relative, which
// is a floor on the `n2` and `mean` columns of every engine-vs-pin comparison here and is printed as
// such rather than folded into a band. `maxabs` is exact. Reproducing torch's F32 reduction order in
// C++ was rejected: it buys bit-identity with the pins at the cost of a second implementation of a
// reduction order, and bit-identity is not what this path is for.
//
// WHAT IT DELIBERATELY DOES NOT DO: judge. This driver knows nothing about the fixtures — that
// ignorance is the split the file header argues for — so it prints and `gate.py` decides, including
// how far the engine's state is allowed to be from the dump's (a route-distance question, D-3/D-5
// territory, not this file's to answer).

// BF16 bits -> F64, by the bit shift. Exact (BF16 is a prefix of F64), and the same widening
// `gate.py:load_pin` and `fp_row`'s `.to(torch.float32)` do.
inline double bf16_to_f64(uint16_t bits) {
    const uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return (double)f;
}

// One D2H of the §08.1 tuple. The KV is compacted to the WRITTEN lanes: the allocation runs to
// `cap`, slots `prefix_len..cap-1` have never been written, and pooling uninitialised slots into a
// moment sum would make the fingerprint meaningless.
struct HostState {
    std::vector<float> gdn;       // [18][16][128][128]
    std::vector<float> conv;      // [18][3][6144], row 0 oldest (§07.2's roll order)
    std::vector<uint16_t> keys;   // [6][2][lanes][256] BF16 bits, post-k_norm post-rope
    std::vector<uint16_t> values;  // [6][2][lanes][256] BF16 bits, raw v
    int64_t lanes = 0;            // prefix_len when this snapshot was taken
};

void d2h(void* dst, const void* src, size_t bytes) {
    const hipError_t e = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        throw std::runtime_error(std::string("--state-fp device-to-host copy failed: ")
                                 + hipGetErrorString(e));
    }
}

HostState download_state(const engine::RequestState& st) {
    const engine::StateSizes s = st.sizes();
    HostState h;
    h.lanes = st.prefix_len;
    h.gdn.resize(s.gdn_state_bytes / sizeof(float));
    h.conv.resize(s.conv_window_bytes / sizeof(float));
    const size_t src_stride = (size_t)engine::kNumKvHeads * (size_t)st.cap * engine::kHeadDim;
    const size_t dst_stride = (size_t)engine::kNumKvHeads * (size_t)h.lanes * engine::kHeadDim;
    h.keys.resize((size_t)engine::kNumAttnLayers * dst_stride);
    h.values.resize((size_t)engine::kNumAttnLayers * dst_stride);
    d2h(h.gdn.data(), st.gdn_state, s.gdn_state_bytes);
    d2h(h.conv.data(), st.conv_window, s.conv_window_bytes);
    // One copy per (layer, K-or-V, head) so each head row lands contiguous at the compacted stride.
    // §08.5's head-major layout is what makes the common-extent Δ² below a contiguous compare.
    for (int l = 0; l < engine::kNumAttnLayers; ++l) {
        for (int which = 0; which < 2; ++which) {
            const uint16_t* src = (which ? st.kv_values : st.kv_keys) + (size_t)l * src_stride;
            uint16_t* dst = (which ? h.values.data() : h.keys.data()) + (size_t)l * dst_stride;
            for (int hd = 0; hd < engine::kNumKvHeads; ++hd) {
                d2h(dst + (size_t)hd * h.lanes * engine::kHeadDim,
                    src + (size_t)hd * st.cap * engine::kHeadDim,
                    (size_t)h.lanes * engine::kHeadDim * sizeof(uint16_t));
            }
        }
    }
    return h;
}

// `fp_row`'s pooled row. `dn2` accumulates only where a previous state reaches, which is the
// part-wise common-extent rule: an appended lane is not a rewritten one.
struct FpPool {
    double n2 = 0.0, sum = 0.0, maxabs = 0.0, dn2 = 0.0;
    size_t n = 0;

    void add(double x, double d) {
        n2 += x * x;
        sum += x;
        const double a = std::fabs(x);
        if (a > maxabs) maxabs = a;
        dn2 += d;
        ++n;
    }
    void f32(const float* x, size_t count, const float* prev, size_t dn_count) {
        for (size_t i = 0; i < count; ++i) {
            const double v = (double)x[i];
            const double d = (prev && i < dn_count) ? v - (double)prev[i] : 0.0;
            add(v, d * d);
        }
    }
    void bf16(const uint16_t* x, size_t count, const uint16_t* prev, size_t dn_count) {
        for (size_t i = 0; i < count; ++i) {
            const double v = bf16_to_f64(x[i]);
            const double d = (prev && i < dn_count) ? v - bf16_to_f64(prev[i]) : 0.0;
            add(v, d * d);
        }
    }
};

// `%.17g` round-trips a double exactly, so the gate reads the number that was accumulated here and
// not a rounded copy of it. `no_prev` is §08.4's convention: with nothing to compare against, Δ² is
// the state itself, so the first row's dn2 equals its n2 — the same convention the pins carry.
void print_fp_row(const char* phase, const char* family, int row, int layer, size_t n,
                  const FpPool& fp, bool no_prev) {
    std::printf("STATEFP phase=%s family=%s row=%d layer=%d n=%zu n2=%.17g maxabs=%.17g"
                " mean=%.17g dn2=%.17g\n",
                phase, family, row, layer, n, fp.n2, fp.maxabs,
                fp.n ? fp.sum / (double)fp.n : 0.0, no_prev ? fp.n2 : fp.dn2);
}

void emit_state_fp(const char* phase, const HostState& cur, const HostState* prev) {
    std::vector<int> gdn_layers, attn_layers;
    for (int l = 0; l < engine::kNumLayers; ++l) {
        (engine::is_full_attention(l) ? attn_layers : gdn_layers).push_back(l);
    }
    std::printf("STATEFP phase=%s prefix=%lld\n", phase, (long long)cur.lanes);

    for (size_t r = 0; r < gdn_layers.size(); ++r) {
        const size_t n = (size_t)engine::kNumValueHeads * engine::kDk * engine::kDv;
        FpPool s;
        s.f32(cur.gdn.data() + r * n, n, prev ? prev->gdn.data() + r * n : nullptr, n);
        print_fp_row(phase, "state", (int)r, gdn_layers[r], n, s, prev == nullptr);

        // The conv row is the engine's §08.2 window: k_conv−1 = 3 rows of 6144, row 0 oldest. It is
        // NOT the same extent as the pinned `conv_fp` row, which pools the REFERENCE's conv buffer —
        // A-04 names why (`causal_conv1d_update` cats `[conv_state, x]`, so HF's buffer is k_conv = 4
        // columns and the engine keeps the newest 3 of them). Printing `cols=3` here is what makes
        // that gap visible in the log instead of looking like a state error; (g) prices it.
        const size_t cn = (size_t)engine::kConvWindow * engine::kConvChannels;
        FpPool c;
        c.f32(cur.conv.data() + r * cn, cn, prev ? prev->conv.data() + r * cn : nullptr, cn);
        std::printf("STATEFP phase=%s family=conv row=%d layer=%d cols=%d ", phase, (int)r,
                    gdn_layers[r], engine::kConvWindow);
        std::printf("n=%zu n2=%.17g maxabs=%.17g mean=%.17g dn2=%.17g\n", cn, c.n2, c.maxabs,
                    c.n ? c.sum / (double)c.n : 0.0, prev == nullptr ? c.n2 : c.dn2);
    }

    const size_t cur_head = (size_t)cur.lanes * engine::kHeadDim;
    for (size_t r = 0; r < attn_layers.size(); ++r) {
        // One row per attention layer with K and V pooled together, exactly as `fp_row` is called
        // with parts=[keys, values]; the Δ² stays part-wise (K against K, V against V) and over the
        // common lane extent only, which is what lets a step append a lane for free.
        const size_t prev_head = prev ? (size_t)prev->lanes * engine::kHeadDim : 0;
        const size_t common_head =
            prev ? (size_t)std::min(cur.lanes, prev->lanes) * engine::kHeadDim : 0;
        const uint16_t* krow = cur.keys.data() + r * 2 * cur_head;
        const uint16_t* vrow = cur.values.data() + r * 2 * cur_head;
        const uint16_t* pkrow = prev ? prev->keys.data() + r * 2 * prev_head : nullptr;
        const uint16_t* pvrow = prev ? prev->values.data() + r * 2 * prev_head : nullptr;
        FpPool kv;
        for (int hd = 0; hd < engine::kNumKvHeads; ++hd) {
            kv.bf16(krow + (size_t)hd * cur_head, cur_head,
                    pkrow ? pkrow + (size_t)hd * prev_head : nullptr, common_head);
            kv.bf16(vrow + (size_t)hd * cur_head, cur_head,
                    pvrow ? pvrow + (size_t)hd * prev_head : nullptr, common_head);
        }
        std::printf("STATEFP phase=%s family=kv row=%d layer=%d lanes=%lld ", phase, (int)r,
                    attn_layers[r], (long long)cur.lanes);
        std::printf("n=%zu n2=%.17g maxabs=%.17g mean=%.17g dn2=%.17g\n", 4 * cur_head, kv.n2,
                    kv.maxabs, kv.n ? kv.sum / (double)kv.n : 0.0,
                    prev == nullptr ? kv.n2 : kv.dn2);
    }
    std::fflush(stdout);
}

// ---- the stage clock's passes (`--time-stages`, `engine/stage_clock.h`) ---------------------
// One pass = rebuild the state this run's generation steps started from (§08.4 reset plus the same
// prompt wave), replay the SAME fed tokens through it, and price each step at one submission
// granularity. Three passes — burst (production shape, one wait per step), stage (a wait at every
// family boundary), kernel (a wait after every submission) — so the three numbers are taken on the same
// tokens at the same lanes in one process. The comparison this instrument exists to make is between
// them, and nothing else about the box is allowed to differ across them.
//
// Why a REPLAY rather than a second free run: a free run feeds its own samples, so a second free run
// could diverge and compare two different tokens; feeding the recorded list makes the work identical by
// construction. The parity line each pass prints (same samples, bit-identical top logits) is what proves
// the granularity did not change the work, which is the difference between a controlled comparison and
// two unrelated runs quoted next to each other.
//
// Why the writes and the gate lines stay with the run's own pass: `engine/gate.py` parses the `decode`/
// `STEP`/`RESULT` lines and the `--out` dumps are pinned bytes. A measurement pass therefore prices a
// step and keeps only its sample id and that sample's logit.

struct StepRec {
    int step = -1;
    int fed = -1;          // what the run fed this step — the replay feeds exactly this
    int sampled = -1;      // what it produced, for the parity check
    float top_logit = 0.0f;
};

struct PassResult {
    const char* gran = "";
    std::vector<double> step_ms;
    bool tokens_ok = true;
    bool logits_ok = true;
    int first_mismatch = -1;
    double mean_ms = 0.0;
};

// The ledger's own spread convention: (max - min) / min, as a percentage.
struct Spread {
    double mean = 0.0;
    double mn = 0.0;
    double mx = 0.0;
    double pct = 0.0;
};

Spread summarize(const std::vector<double>& v) {
    Spread s;
    if (v.empty()) return s;
    double sum = 0.0;
    s.mn = s.mx = v[0];
    for (const double x : v) {
        sum += x;
        if (x < s.mn) s.mn = x;
        if (x > s.mx) s.mx = x;
    }
    s.mean = sum / static_cast<double>(v.size());
    s.pct = s.mn > 0.0 ? 100.0 * (s.mx - s.mn) / s.mn : 0.0;
    return s;
}

// One granularity, priced and printed. `ref` is the run's own step list.
//
// `mtp`, when non-null, is replayed as part of the step it followed — see the note above the loop for why
// the draft has to be INSIDE the measured span rather than beside it, and why the pass counts its own
// rounds instead of borrowing the run's.
PassResult stage_pass(const Options& o, const engine::WeightStore& store, engine::RequestState& st,
                      engine::DecodeScratch& sc, const std::vector<StepRec>& ref,
                      engine::stageclock::Mode mode, MtpSide* mtp) {
    using engine::stageclock::Clock;
    using engine::stageclock::Mode;
    PassResult out;
    out.gran = engine::stageclock::mode_name(mode);

    st.reset();
    st.sampled_token = -1;
    st.stopped = false;
    // The prompt wave again, at `kLast`: the state it leaves is the state the run's first generation
    // step started from, and pricing every row's logits would buy vocabulary D2Hs nobody reads.
    engine::prefill_wave(store, st, sc, o.prompt, engine::PrefillLogits::kLast);

    std::unique_ptr<Clock> clock;
    if (mode != Mode::kBurst) clock = std::make_unique<Clock>(mode);
    // Route D1: a packed step submits fewer projections than §07 bills, and the clock has no wave shape to
    // derive that from, so the driver states it. Leaving it unset would make `plan_match=0` the signature of
    // a legal run, and an instrument that reads broken on a configuration it should merely describe is an
    // instrument nobody trusts on the configuration that matters.
    if (clock) clock->set_plan_proj(engine::proj_launches_per_body(o.proj_plan));
    // Route S1 item (1), stated for the same reason and with the same consequence: a folded step submits
    // no `cast` launch, so a clock that was not told would bill §07's 97 against 243 submissions and read
    // `plan_match=0` on a run that is merely folded. `decode_step.cpp`'s `fold_cast` is the dispatch side
    // of this same fact, and it is a one-row rule — which is fine for a STEP clock, because a step IS a
    // one-row body (`decode_step` submits one token), so the fold either reaches the whole 97 or none.
    if (clock) clock->set_plan_cast_fold(o.cast_fold);
    // The same telling, one family over: a folded step submits no `copy` at all, and a clock left on the
    // shipped bill would read `plan_match=0` against 224 submissions for a run that is merely folded.
    // `decode_step.cpp`'s `fold_copy` is the dispatch side, and it too is a one-row rule, which a STEP
    // clock satisfies by construction because a step IS a one-row body.
    if (clock) clock->set_plan_copy_fold(o.copy_fold);
    // §09 / P7: the draft's declared per-step bill. Set from the §09 constants the `MTPDRAFT` census line
    // prints, and set ONLY when the flag is on: left at 0 the plan is §07's exactly as it was before the
    // `mtp` row existed, which is what makes flag-off `--time-stages` output byte-identical to a pre-MTP
    // binary rather than a run with an extra row and a bigger plan total. The refusals in `validate()`
    // guarantee the uniformity this setter assumes (one round per step), and `mtp_rounds=` below is the
    // in-pass witness that it held.
    if (clock && mtp) clock->set_plan_mtp(engine::mtpdraft::kStageSubsPerRound);
    // The draft's own state starts the pass the way the request's does: a fresh cache and a fresh round
    // tracker, so the replayed schedule is the run's schedule and not a continuation of the previous
    // pass. Without this the second pass would inherit the first pass's lanes and its accept bookkeeping.
    long long pass_rounds = 0;
    if (mtp) {
        engine::mtpdraft::reset(mtp->st);
        mtp->track.clear();
    }
    engine::stageclock::arm(clock.get());
    for (const StepRec& want : ref) {
        if (clock) clock->begin_step();
        const auto t0 = std::chrono::steady_clock::now();
        // logits=nullptr: the sample and its logit come back in `StepResult`; the vocabulary itself is
        // still downloaded and argmaxed by `finish_token`, which is the point — a pass that skipped the
        // D2H would be measuring a step the production path never submits.
        const engine::StepResult r = engine::decode_step(store, st, sc, want.fed, nullptr);
        // INSIDE the measured span, on purpose. A round is submissions the run makes per step, and billing
        // them from a span that ended at `decode_step`'s return would charge them to the host gap between
        // steps — the same misattribution the flag used to be refused for, now priced instead. `sc.norm`
        // row 0 is the step's final-norm row and the tail only reads it, so it is still the right operand
        // here, exactly as it is in the run's own loop.
        if (mtp && mtp_after_step(*mtp, store, sc.norm, st.prefix_len - 1, r.token, /*quiet=*/true)) {
            ++pass_rounds;
        }
        if (clock) clock->end_step();
        out.step_ms.push_back(std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count());
        if (r.token != want.sampled) {
            out.tokens_ok = false;
            if (out.first_mismatch < 0) out.first_mismatch = want.step;
        }
        if (std::memcmp(&r.top_logit, &want.top_logit, sizeof(float)) != 0) out.logits_ok = false;
    }
    engine::stageclock::arm(nullptr);   // never leave the marker armed for an unmeasured pass

    const Spread sp = summarize(out.step_ms);
    out.mean_ms = sp.mean;
    // No `syncs=` here: the wait count is the clock's, and it prints it per step on its own TOTAL line.
    std::printf("STAGEPASS gran=%s steps=%zu ms_mean=%.4f ms_min=%.4f ms_max=%.4f spread_pct=%.2f\n",
                out.gran, out.step_ms.size(), sp.mean, sp.mn, sp.mx, sp.pct);
    std::printf("STAGEPARITY gran=%s tokens_identical=%d logits_bitidentical=%d first_mismatch=%s"
                "%s\n",
                out.gran, out.tokens_ok ? 1 : 0, out.logits_ok ? 1 : 0,
                out.first_mismatch < 0 ? "none" : std::to_string(out.first_mismatch).c_str(),
                (out.tokens_ok && out.logits_ok) ? "" : " note=COMPARISON_VOID");
    if (clock) std::printf("%s\n", clock->describe().c_str());
    // The `mtp` row's own witness, and it is the row that can lie most quietly: `plan_per_step` bills one
    // round per step, and the one way that can stop being true mid-run is a `DRAFT-SKIP` at the cache
    // ceiling, which subtracts 15 submissions from one step and reads as an unexplained `plan_match=0`.
    // So the pass prints its own round count and says whether it equals its own step count. `uniform=0`
    // with `plan_match=0` is a ceiling, and the `DRAFT-SKIP` line above names it; `uniform=1` with
    // `plan_match=0` is a broken declaration, which is the case the wrong-count witness is for.
    if (mtp) {
        std::printf("STAGEMTP gran=%s steps=%zu rounds=%lld bill_per_round=%lld uniform=%d\n",
                    out.gran, ref.size(), pass_rounds,
                    (long long)engine::mtpdraft::kStageSubsPerRound,
                    pass_rounds == (long long)ref.size() ? 1 : 0);
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        std::printf("STAGESTEP gran=%s step=%d fed=%d sampled=%d ms=%.4f\n", out.gran, ref[i].step,
                    ref[i].fed, ref[i].sampled, i < out.step_ms.size() ? out.step_ms[i] : 0.0);
    }
    return out;
}

// ---- the wave clock's passes (`--time-waves`, `engine/stage_clock.h`) ----------------------
// One pass = rebuild the state the run's prompt wave started from (§08.4 reset) and prefill the SAME
// prompt through it, priced at one submission granularity. Two passes, not the step clock's three:
// `burst` is the production wave (every body queued, one wait at the logits) and `stage` is the same
// submissions with a wait at every family boundary. `kernel` is refused outright for a wave —
// `Clock::begin_wave()` says why in the message — because §04.4's recurrence is 18 launches per token and
// draining each of them measures the drain, not the queueing P9 exists to change.
//
// What the two passes are FOR, and what neither can give. Only the serialized pass can say which family
// owns which milliseconds: in a queued wave nothing marks a family boundary except the host's own
// submission order, so a per-family number for the production wave would need a drain (making it the
// serialized pass) or GPU events (which read exactly 0 below ~1 ms on this iGPU — `prototype/README.md`
// rule 7). The burst pass therefore yields ONE number, its wall, and `WAVEGAP` prints the distance between
// the two so a reader can see how much of the serialized wave is the instrument before quoting any share
// against the production wall.
//
// Like the step passes, these mutate `st` (reset + re-prefill) and write nothing to `--out`, so they run
// after everything the run reports about itself; `decode_run` snapshots the `RESULT` fields they move.
struct WavePassResult {
    const char* gran = "";
    double wall_ms = 0.0;
    bool tokens_identical = true;
    bool logits_bitidentical = true;
    int first_mismatch = -1;
    int compared = 0;
};

// One granularity, priced and printed. `ref` is the run's own prompt wave, whose last row is the only row
// both sides price (a pass always prefills at `kLast` — §06.5's shape — because pricing T vocabularies in
// order to compare them would be buying D2Hs the report never reads).
WavePassResult wave_pass(const Options& o, const engine::WeightStore& store, engine::RequestState& st,
                         engine::DecodeScratch& sc, const engine::PrefillResult& ref,
                         engine::stageclock::Mode mode) {
    using engine::stageclock::Clock;
    using engine::stageclock::Mode;
    using engine::stageclock::Unit;
    WavePassResult out;
    out.gran = engine::stageclock::mode_name(mode);

    st.reset();
    st.sampled_token = -1;
    st.stopped = false;

    std::unique_ptr<Clock> clock;
    if (mode != Mode::kBurst) clock = std::make_unique<Clock>(mode, Unit::kWave);
    engine::stageclock::arm(clock.get());
    if (clock) clock->begin_wave();
    const auto t0 = std::chrono::steady_clock::now();
    const engine::PrefillResult pass =
        engine::prefill_wave(store, st, sc, o.prompt, engine::PrefillLogits::kLast);
    // `end_wave()` BEFORE the wall is read: the last segment is the host tail, whose wait is what lands the
    // logits, so closing it here is what makes `sum_ms` and `wall_ms` two readings of one span rather than
    // a sum plus a missing tail.
    if (clock) clock->end_wave();
    out.wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    engine::stageclock::arm(nullptr);   // never leave the marker armed for an unmeasured pass

    const engine::PrefillRow* want = ref.y0();
    const engine::PrefillRow* got = pass.y0();
    if (!want || !got) {
        out.tokens_identical = false;
        out.first_mismatch = 0;
    } else {
        ++out.compared;
        if (want->sampled != got->sampled) {
            out.tokens_identical = false;
            out.first_mismatch = got->index;
        }
        // Bit for bit, not to a tolerance: the whole point of the line is that arming a clock cannot move
        // one bit of the answer, and 1 ULP at the argmax is exactly what a reordering would look like.
        out.logits_bitidentical = want->logits.size() == got->logits.size() &&
                                  std::memcmp(want->logits.data(), got->logits.data(),
                                              want->logits.size() * sizeof(float)) == 0;
    }

    const int bodies =
        (int)((o.prompt.size() + (size_t)sc.wave_rows - 1) / (size_t)sc.wave_rows);
    // How many rows this pass PRICED, which is not how many it has: `pass.rows` is one row per token
    // (§06.5's `PrefillRow`), and under `kLast` only the last of them was ever submitted to `finish_token`
    // — so only the y0 row costs a cast, an `lm_head` and a D2H. Derived from the wave's own y0 rather than
    // typed as 1, so a future `kAll` pass here would report its own bill instead of a stale constant.
    const int priced_rows = pass.y0() ? 1 : 0;
    std::printf("WAVEPASS gran=%s wall_ms=%.4f tokens=%zu bodies=%d wave_rows=%d priced_rows=%d\n",
                out.gran, out.wall_ms, pass.size(), bodies, sc.wave_rows, priced_rows);
    std::printf("WAVEPARITY gran=%s tokens_identical=%d logits_bitidentical=%d rows_compared=%d"
                " first_mismatch=%s%s%s\n",
                out.gran, out.tokens_identical ? 1 : 0, out.logits_bitidentical ? 1 : 0,
                out.compared,
                out.first_mismatch < 0 ? "none" : std::to_string(out.first_mismatch).c_str(),
                (out.tokens_identical && out.logits_bitidentical) ? "" : " note=COMPARISON_VOID",
                o.prefill_logits == engine::PrefillLogits::kAll ? " note=pass_priced_y0_row_only" : "");
    if (clock) {
        engine::stageclock::WaveShape shape;
        shape.tokens = (int)o.prompt.size();
        shape.bodies = bodies;
        shape.wave_rows = sc.wave_rows;
        shape.priced_rows = priced_rows;
        shape.wall_ms = out.wall_ms;
        // Which scan the pass ran, so the plan bills the recurrence the way it was actually submitted:
        // per token for `seq`, per body for the resident form, per chunk for the (refused) chunked one.
        // The single-row-body exception is derived from the shape inside `wave_plan_bill`, not here.
        shape.gdn_chunked = (o.gdn_scan == engine::kernels::GdnScan::kChunk);
        shape.gdn_resident = (o.gdn_scan == engine::kernels::GdnScan::kCollapse);
        // How many token-steps one `gdn_rec` launch carried, so the plan bills the recurrence's
        // SUBMISSIONS the way the body actually submitted them (per token at the default, per span
        // once the loop is on). The span as asked for is enough: `wave_plan_bill` derives the body
        // widths from `tokens`/`bodies`/`wave_rows` and therefore sees the ragged and one-row bodies
        // for itself instead of trusting a caller to announce them.
        shape.gdn_loop_span = o.gdn_loop;
        // Whether the span's body was the hoisted pair rather than the single looped kernel, so the plan
        // bills TWO `gdn_rec` launches per span where the shipped form bills one. Asked as the RUN ASKED
        // for it, like the span above: `wave_plan_bill` derives the one-row bodies for itself, and a
        // one-row body runs the per-token kernel with no pre-pass whatever this says.
        shape.gdn_norm_hoisted = (o.gdn_norm == engine::kernels::GdnNorm::kHoist);
        // Which GEMM a wide body ran, so the plan bills the cast row the way the body submitted it. Asked
        // at `kMaxWaveRows` rather than at a body width on purpose: the only width
        // `resolve_wave_gemm` overrides is `rows <= 1`, and `wave_plan_bill` derives that one-row body
        // from `tokens`/`bodies`/`wave_rows` itself, so what the plan needs from the driver is the route
        // the run asked for and nothing else. Same question the `GEMM` banner asks above.
        shape.gemm_library =
            engine::kernels::wave_gemm_is_library(o.wave_gemm, engine::kMaxWaveRows);
        // Whether one-row bodies grouped their input projections (route D1). As the run ASKED for it, like
        // the two rows above: `wave_plan_bill` applies it only to the bodies `submit_wave_body` would pack
        // (derived from `tokens`/`bodies`/`wave_rows`), so a 512-row wave bills §07's count whatever this
        // says and only a `--wave-rows 1` pass moves — which is the same no-wide-bit-moves property the
        // launcher enforces, witnessed rather than asserted.
        shape.proj_packed = (o.proj_plan == engine::kernels::ProjPlan::kPacked);
        // Route S1 item (1). Asked as the run asked for it, like the two rows above, because the WIDTH rule
        // is `wave_plan_bill`'s to derive: a one-row body folds all six per-token sites, a wide body folds
        // only `finish_token`'s final-norm widening, and this line cannot make a wide pass bill the fold of
        // a site `submit_wave_body` still submitted.
        shape.cast_fold = o.cast_fold;
        shape.copy_fold = o.copy_fold;
        std::printf("%s\n", clock->describe_wave(shape).c_str());
    }
    return out;
}

// The whole `--time-waves` report. Same placement rule as `time_stages_report`: it mutates `st`, it writes
// nothing, and its lines sit above `RESULT` so the last line of a run stays the result line.
void wave_report(const Options& o, const engine::WeightStore& store, engine::RequestState& st,
                 engine::DecodeScratch& sc, const engine::PrefillResult& ref) {
    // The instrument's own overhead, printed first, for the same reason the step clock prints it: every
    // serialized segment ends in one of these waits, so a reader who knows the count (`syncs=` on the
    // TOTAL line) can take the clock back out of the clock's number.
    const double idle_sync_us = engine::stageclock::calibrate_idle_sync(200);
    std::printf("WAVE gran=calib idle_sync_us=%.3f iters=200"
                " note=one_of_these_sits_inside_every_serialized_segment\n", idle_sync_us);

    const WavePassResult burst = wave_pass(o, store, st, sc, ref, engine::stageclock::Mode::kBurst);
    const WavePassResult stage = wave_pass(o, store, st, sc, ref, engine::stageclock::Mode::kStage);

    // The distance between the two waves. The shares on the `stage` rows are shares of
    // `serialized_wall_ms`, and this line is what a reader needs before putting one of them on the
    // production wall: `exposed_ms` is what a drain at every family boundary cost a wave that otherwise
    // had to wait for nothing.
    std::printf("WAVEGAP gran=%s burst_wall_ms=%.4f serialized_wall_ms=%.4f exposed_ms=%.4f ratio=%.3f\n",
                stage.gran, burst.wall_ms, stage.wall_ms, stage.wall_ms - burst.wall_ms,
                burst.wall_ms > 0.0 ? stage.wall_ms / burst.wall_ms : 0.0);
}

// The whole `--time-stages` report. Called after the run has printed everything it normally prints
// about its state, because every pass here MUTATES `st` (reset + re-prefill), and after the run's own
// dumps are closed, because a pass writes nothing.
void time_stages_report(const Options& o, const engine::WeightStore& store, engine::RequestState& st,
                        engine::DecodeScratch& sc, const std::vector<StepRec>& base, MtpSide* mtp) {
    if (base.empty()) {
        std::printf("STAGE gran=calib note=no_generation_steps_ran\n");
        return;
    }
    // The instrument's own overhead, printed first: every serialized segment ends in one of these
    // waits, so a reader can take the clock back out of the clock's number.
    const double idle_sync_us = engine::stageclock::calibrate_idle_sync(200);
    std::printf("STAGE gran=calib idle_sync_us=%.3f iters=200"
                " note=one_of_these_sits_inside_every_serialized_segment\n", idle_sync_us);

    const PassResult burst =
        stage_pass(o, store, st, sc, base, engine::stageclock::Mode::kBurst, mtp);
    const PassResult stage =
        stage_pass(o, store, st, sc, base, engine::stageclock::Mode::kStage, mtp);
    const PassResult kernel =
        stage_pass(o, store, st, sc, base, engine::stageclock::Mode::kKernel, mtp);

    // `exposed` is the finding this whole instrument was built for: what one extra wait per stage (or
    // per kernel) costs, on the same launches at the same lanes. ~0 means there was no overlap to lose
    // and the residue is inside the kernels; a large positive number means the production step is
    // hiding exactly that much per-launch stall. Arithmetic, not interpretation — the reading belongs
    // to whoever is holding a clean window.
    for (const PassResult* p : {&stage, &kernel}) {
        std::printf("STAGEGAP gran=%s burst_ms_mean=%.4f serialized_ms_mean=%.4f exposed_ms=%.4f"
                    " ratio=%.3f\n",
                    p->gran, burst.mean_ms, p->mean_ms, p->mean_ms - burst.mean_ms,
                    burst.mean_ms > 0.0 ? p->mean_ms / burst.mean_ms : 0.0);
    }
}

// ---- route D1: the projection-plan echo -------------------------------------------------------------
//
// One line, formatted once. A normal run prints it next to `LAUNCH` (which stays §07's SPLIT bill,
// because `gate.py` section (d) re-derives that table in Python and would stop being an independent check
// if the same function answered for a flag); `--dump-proj-plan` prints it as the header of a per-launch
// dump and leaves. Every field comes from the flags or from the launcher's own geometry constants, so a
// printed rows/block cannot disagree with a launched one, and neither can be typed into a log by hand.
std::string proj_plan_line(const Options& o) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "PROJPLAN plan=%s shape=%s launches_per_step=%d rpb=%d threads=%d warp=%d segs_max=%d",
                  engine::kernels::proj_plan_name(o.proj_plan),
                  engine::kernels::proj_shape_name(o.proj_shape),
                  engine::proj_launches_per_body(o.proj_plan),
                  engine::kernels::proj_shape_rows_per_block(o.proj_shape),
                  engine::kernels::proj_shape_threads(o.proj_shape),
                  engine::kernels::proj_warp_width(),
                  (int)engine::kernels::kProjMaxSegs);
    return std::string(buf);
}

// The device-free plan dump: one line per projection a row-priced body submits, with the group it belongs
// to. Reads the same `proj_plan_jobs` the stage clock bills itself against, so "48 launches" is a counted
// statement about a list, not a number a reader has to trust.
void print_proj_plan_dump(const Options& o) {
    std::printf("%s\n", proj_plan_line(o).c_str());
    const std::vector<engine::ProjPlanJob> jobs = engine::proj_plan_jobs(o.proj_plan);
    int launches = 0;
    int seq = 0;
    for (const engine::ProjPlanJob& j : jobs) {
        if (j.starts_launch) ++launches;
        std::printf("PROJJOB i=%d layer=%d kind=%s group=%d seg=%d/%d launch=%d rows=%d in=%d"
                    " dest=%s weight=%s library_eligible=%d\n",
                    ++seq, j.layer, j.kind, j.group, j.seg, j.segs, j.starts_launch ? 1 : 0,
                    j.rows, j.in, j.dest, j.weight.c_str(), j.library_eligible ? 1 : 0);
    }
    std::printf("PROJPLAN summary tensors=%zu launches_per_step=%d\n", jobs.size(), launches);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decode_run: %s\n", e.what());
        print_usage(stderr);
        return 2;
    }
    if (o.help) {
        print_usage(stdout);
        return 0;
    }
    if (o.dump_proj_plan) {
        // Before the store is loaded, so before any `hipMalloc`, any upload and any kernel: the plan is
        // §00 structure plus the flags, and that is the whole of what it needs. This is the mode that
        // lets a geometry claim be checked while someone else holds the GPU lock.
        print_proj_plan_dump(o);
        return 0;
    }

    try {
        // `--out DIR` is created here, before anything is loaded, and the failure is reported here too.
        // The alternative is discovering a typo'd or missing path at the first write — after 1.55 GB of
        // weights have been `hipMalloc`'d and copied, and after every step has run — which is a real
        // way to lose a device window: `decode_run FAILED: cannot open /tmp/fr4/sampled.txt` after a
        // silent 3-second load. Creating it is the friendlier half of the fix (a driver writes
        // `--out tmp/freerun` and means it); refusing early is the other half.
        if (!o.out_dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(o.out_dir, ec);
            if (ec || !std::filesystem::is_directory(o.out_dir)) {
                throw std::runtime_error("cannot create --out directory " + o.out_dir + ": "
                                         + (ec ? ec.message() : "it exists and is not a directory"));
            }
        }
        engine::LoadStats stats;
        engine::WeightStore::Options lopts;
        lopts.model_dir = o.model_dir;
        lopts.verify_device_bytes = o.load_verify;
        lopts.measure_steady_h2d = false;   // this driver is not here to measure the load path
        const auto t0 = std::chrono::steady_clock::now();
        engine::WeightStore store = engine::WeightStore::load(lopts, stats);
        const double load_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("loaded %zu tensors, %.4f GB resident in %.1f ms (%zu allocations)\n",
                    stats.n_loaded, store.resident_bytes() / 1e9, load_ms,
                    store.allocation_count());

        // §00 hyperparameters against the checkpoint, before a single launch: every row count and
        // dtype `decode_step` hard-codes goes through this. It is not opt-in, because the failure it
        // catches is "the engine ran and produced plausible garbage".
        const std::string bad = engine::check_weights(store);
        std::printf("check_weights: %s\n", bad.empty() ? "PASS" : ("FAIL — " + bad).c_str());
        if (!bad.empty()) return 1;

        // A session allocates once, up front, and cannot know the transcript length yet; 4096
        // slots is 48 MiB of a pool this model uses 4 % of (§08.1's 12 KiB/token). A run mode can
        // derive the exact need from the prompt and the step budget instead.
        const int64_t cap = o.cap > 0
                                ? o.cap
                                : (o.session ? kSessionDefaultCap
                                             : (int64_t)(o.prompt.size() + (o.steps > 0 ? o.steps : 1) + 1));
        const engine::StateSizes sizes = engine::state_sizes(cap);
        std::printf("state: %s\n", sizes.describe().c_str());
        // Byte-exact, for `gate.py`. `describe()` prints MiB to three decimals, which is a fine
        // human line and a terrible thing to assert on: 1 KiB of drift would round away.
        std::printf("STATE cap=%lld gdn=%llu conv=%llu kv=%llu total=%llu per_token=%llu\n",
                    (long long)sizes.cap, (unsigned long long)sizes.gdn_state_bytes,
                    (unsigned long long)sizes.conv_window_bytes, (unsigned long long)sizes.kv_bytes,
                    (unsigned long long)sizes.total_bytes,
                    (unsigned long long)sizes.per_token_bytes);
        // Printed in every mode, including --sizes-only: it is a statement about the code, not about
        // the run, and `gate.py` asserts it even when the run part is skipped.
        std::printf("LAUNCH %s\n", engine::planned_launch_counts().describe().c_str());
        // Route D1's own plan line, printed beside `LAUNCH` for the reason `LAUNCH` is printed at all: it
        // is a statement about the code and the flags, not about the run, so it appears in every mode and
        // in a `--time-stages` log it is the CLAIM the `STAGE … family=proj launches_per_step=` line is
        // then checked against. Since the flip the two lines AGREE on a default run: `planned_launch_counts()`
        // bills the shipped plan, which is `packed`, so `LAUNCH proj=48` and `PROJPLAN … launches_per_step=48`
        // are one claim spelled twice (typed arithmetic here, a counted job list there). They DISAGREE only
        // on a `--proj-plan split` run, where `LAUNCH` keeps the shipped 48 and `PROJPLAN` says 114 — and
        // `LAUNCH` stays flag-free on purpose, because `gate.py` section (d) re-derives the shipped bill in
        // Python and that cross-check stops being independent the moment the engine's echo is a function of
        // its own flags (the `--attn-tile` trap in AGENTS.md). On a split arm read `PROJPLAN`, or the counted
        // `STAGE`/`LAUNCHSTAT` lines, rather than `LAUNCH`.
        std::printf("%s\n", proj_plan_line(o).c_str());
        // Route S1 item (1)'s own line, printed beside `LAUNCH` for the reason `LAUNCH` is printed at
        // all: a run has to be able to say which of its widenings it submitted. It is a separate line
        // rather than another `NUMERICS` field because `NUMERICS` is compared against
        // `engine/numerics_provenance.json` by `gate.py`'s provenance check, and a field that file has
        // never heard of is a red gate for a reason that has nothing to do with numerics.
        //
        // `plan=` is `planned_launch_counts()`'s §07 bill, derived from the layer counts and NOT edited by
        // this route — `gate.py` section (d) re-derives that table in Python and its independence is worth
        // more than one extra field here. `folded=` is what this run deletes from it and `kept=` is what it
        // submits instead, both taken from `planned_launch_counts(cast_fold)` so this echo and the stage
        // clock's `plan_per_step` are ONE derivation and cannot drift apart; `step_bill=` is that total, and
        // it is the number a `STAGE … TOTAL plan_per_step=` line must equal (the A/B asserts it). A one-row
        // body folds every per-layer site and the final-norm site, so `kept=0` when the knob is on; a wide
        // prefill body folds only the final-norm site, which is what `full_wide=final-norm-only` states
        // instead of leaving it to a comment. `prototype/cast_fold/census_sites.py` is the file that
        // accounts for the claim site by site (six sites, 97 launches, the §1 byte rows).
        {
            const int plan_cast = engine::planned_launch_counts().cast;
            const engine::LaunchCounts bill = engine::planned_launch_counts(o.cast_fold);
            std::printf("CASTFOLD on=%d env=QWEN_CAST_FOLD plan=%d folded=%d kept=%d sites=6"
                        " body=one-row full_wide=%s step_bill=%d\n",
                        o.cast_fold ? 1 : 0, plan_cast, plan_cast - bill.cast, bill.cast,
                        o.cast_fold ? "final-norm-only" : "all", bill.total());
        }
        // Route S1's follow-on, on its own line for the same reason and with the same fields. Here the
        // two derivations it ties together are `stage_clock.cpp`'s wave `copy` row and
        // `planned_launch_counts`'s `c.other`, both of which spell the same five submission sites; the
        // `sites=` field is the census's own count and `census_sites.py` refuses if the source ever
        // disagrees with it, so this line cannot quietly keep claiming five sites after a sixth appears.
        // `step_bill=` is the field the A/B asserts, and it must equal the `STAGE` line's
        // `plan_per_step` in the same run.
        {
            const int plan_copy = engine::planned_launch_counts().other - 1;   // other minus lm_head
            const engine::LaunchCounts bill =
                engine::planned_launch_counts(o.cast_fold, o.copy_fold);
            std::printf("COPYFOLD on=%d env=QWEN_COPY_FOLD plan=%d folded=%d kept=%d sites=5"
                        " body=one-row full_wide=%s step_bill=%d\n",
                        o.copy_fold ? 1 : 0, plan_copy, plan_copy - (bill.other - 1), bill.other - 1,
                        o.copy_fold ? "lm-head-only" : "all", bill.total());
        }
        // D-11's provenance half. The wave-row ceiling and the hipBLAS version are the two numbers that
        // decide what a pinned corpus means once more than one GEMM can produce it: `kMaxWaveRows`
        // because the row-count rule makes it the last width that is still the hand GEMV, hipBLAS's
        // version because §A1 found its solution per (m,n,k) is the numerics and it can change without
        // this repository changing. Printed in every mode, like LAUNCH — a statement about the code and
        // the library it is linked against, not about the run — and read by `gate.py`'s
        // `numerics_provenance_checks` against `engine/numerics_provenance.json`.
        std::printf("NUMERICS wave_rows_cap=%d wave_rows=%d gemm=%s hipblas=%s attn_tile=%d"
                    " attn_touch=%d attn_fattn=%d attn_fattn_roll=%s attn_fattn_qstage=%s"
                    " attn_fattn_pvf16=%s attn_fattn_wide=%s gdn_scan=%s"
                    " gdn_chunk=%d gdn_spell=%s gdn_loop=%d"
                    " gdn_norm=%s gdn_geom=%s proj_plan=%s proj_shape=%s\n",
                    engine::kMaxWaveRows, o.wave_rows,
                    engine::kernels::wave_gemm_name(o.wave_gemm),
                    engine::kernels::gemm_route_version().c_str(), o.attn_tile,
                    o.attn_touch, o.attn_fattn,
                    // Which score roll the WMMA form would run, echoed on the same line for the same reason
                    // as `attn_fattn=` itself: the two sides of A4-L1 are NOT bit-identical, so a family row
                    // that cannot say which roll produced it adjudicates neither. The one case it cannot see
                    // is the one `attn_fattn=` cannot see either — a wave whose rows are all one-pass launches
                    // no WMMA kernel at all, and says so on its own `ATTN_FATTN_SKIP` line rather than having
                    // this field imply a launch that did not happen.
                    engine::kernels::fattn_roll_name(o.attn_fattn_roll),
                    // Where the WMMA wave keeps Q, echoed on the same line for the same reason. This one carries
                    // a numerics claim rather than a band: `unstaged8` is owed BYTE identity with `stage`, so a
                    // log that cannot say which staging plan produced its rows cannot witness that claim in
                    // either direction — and a side that silently took the default would reproduce the shipped
                    // bits exactly, look green, and be priced as the un-stage's residency win. Same blind spot as
                    // `attn_fattn=`/`attn_fattn_roll=`: a wave of all one-pass rows launches no WMMA kernel and
                    // says so on its own `ATTN_FATTN_SKIP` line.
                    engine::kernels::fattn_qstage_name(o.attn_fattn_qstage),
                    // The accumulator width, echoed on the same line for the same reason. This one carries the
                    // register claim rather than a numerics claim: the whole point of the F16 side is 64 VGPR
                    // instead of 128, so a log that cannot say which accumulator ran cannot witness the freeing,
                    // and a side that silently took the default would print the F32 kernel's `vgpr=` and look
                    // like the enabler had failed. Same blind spot as the two fields above it: a wave of all
                    // one-pass rows launches no WMMA kernel and says so on its own `ATTN_FATTN_SKIP` line.
                    o.attn_fattn_pvf16 ? "f16" : "f32",
                    // WHICH WAVE CORE RAN, spelled as the two things it is rather than as the 0/1 the flag
                    // takes: `fattn` is the shipping wave — the 16-row WMMA block on the streaming rows,
                    // attn_slim_wave on the one-pass rows, attn_combine merging them — and `wide` is A12's
                    // 128-row block owning every row of the body with no split and no merge. The echo is
                    // mandatory in `engine/gate.py` for the standing reason: a log that cannot say which wave
                    // core produced its rows cannot be compared against a pinned dump at all, and a side that
                    // silently took the default would price the shipping wave twice and report the second price
                    // as the wide block's. Unlike the three fields above it this one is not blind to an all
                    // one-pass wave: the wide block owns those rows too, so `wide` is true of the run whatever
                    // a later `ATTN_FATTN_SKIP` line says about a kernel this route does not launch. `2` gets
                    // its OWN name rather than riding on `wide`, because the two wide spellings are a
                    // byte-identity pair and a log that cannot tell them apart cannot witness that claim; a
                    // gate that only knows 0 and 1 reading `wide_llama` is a gate being told about a form it
                    // never asked for, which is the failure the echo exists to catch, not a name to shorten.
                    engine::kernels::attn_wide_name(o.attn_fattn_wide),
                    engine::kernels::gdn_scan_name(o.gdn_scan), o.gdn_chunk,
                    // Truthful, not decorative. `collapse` submits `gdn_rc::rec_collapse_kernel`, which
                    // never reads this flag, so an echo naming a spelling here would report a kernel that
                    // did not run - the exact misreport the explicit-pair refusal exists to prevent, and
                    // that refusal now fires only when the flag was actually given, because a default of
                    // `unroll4` must not turn a bare `--gdn-scan collapse` into an error.
                    o.gdn_scan == engine::kernels::GdnScan::kCollapse
                        ? "inert-under-collapse"
                        : engine::kernels::gdn_rec_spell_name(o.gdn_spell),
                    // The span, as given. Under `collapse` the pair with a span >1 is refused above,
                    // so this can only ever echo 1 there — the per-token shape, which is what the
                    // `gdn_scan=collapse` field already says did not run at all. A looped side that
                    // silently took its span from the default would echo 1 and be caught by the
                    // counted `family=gdn_rec launches=` inside the same log (`plan_match` cannot
                    // catch it: it is derived inside this binary and a stale build agrees with
                    // itself — the `17p` retake lesson in AGENTS.md).
                    o.gdn_loop,
                    // Bet G1's form, echoed on the same line for the same reason: a `gdn_rec` family row
                    // that cannot say which body produced it cannot be paired against another one. Under
                    // `collapse` the field is inert — the collapse kernel keeps its own prologue — so it
                    // says so rather than naming a form that did not run.
                    o.gdn_scan == engine::kernels::GdnScan::kCollapse
                        ? "inert-under-collapse"
                        : engine::kernels::gdn_norm_name(o.gdn_norm),
                    // Bet G2's block shape, echoed for the same reason and with the same inert-under-
                    // collapse rule. This is the ONLY field that distinguishes the two sides of the G2
                    // A/B — both submit one launch per span on the same `gdn_rec` row, so the counted
                    // `launches=` line is IDENTICAL on the two sides — which is exactly why canary B of
                    // `prototype/gdn_walk/run_ab_g2.sh` greps this field and cannot grep a launch count.
                    //
                    // And a third inert case, which the default move to `reg16` created: `--gdn-loop 1`
                    // submits the PER-TOKEN bodies (`resolve_gdn_loop`'s routing, and the decode pins were
                    // pinned through it), so no geometry map runs there at all. Since `reg16` is now the
                    // default, echoing `reg16` on a loop-1 log would claim a map that did not run — the same
                    // misreport this field refuses everywhere else, and the reason the parse-level refusal
                    // above keys on `gdn_geom_given` rather than on `!= block`. A named `--gdn-geom` at
                    // loop 1 is refused outright; an unasked default says so here instead of lying.
                    o.gdn_scan == engine::kernels::GdnScan::kCollapse
                        ? "inert-under-collapse"
                        : (o.gdn_loop <= 1
                               ? "na-loop1-per-token-bodies"
                               : engine::kernels::gdn_geom_name(o.gdn_geom)),
                    // Route D1's two levers, echoed like every knob above: both are claimed
                    // BIT-IDENTICAL, so a log that cannot say which projection plan produced its `proj`
                    // family row cannot witness the byte-compare in either direction — and, the reason this
                    // repository has learned the lesson three times, a binary that ignored the flag would
                    // reproduce the shipped bits exactly, look green, and be priced as the win. The counted
                    // counterpart is `PROJPLAN launches_per_step=` and the `STAGE`/`LAUNCHSTAT` lines; this
                    // field is the claim, those are the receipt.
                    engine::kernels::proj_plan_name(o.proj_plan),
                    engine::kernels::proj_shape_name(o.proj_shape));
        if (o.sizes_only) {
            std::printf("RESULT mode=sizes-only tensors=%zu check_weights=PASS\n", stats.n_loaded);
            std::fflush(stdout);
            return 0;
        }

        engine::RequestState st = engine::allocate_state(cap);
        engine::DecodeScratch sc = engine::allocate_scratch(o.wave_rows);
        sc.mlp_gate_map = o.mlp_gate_map;
        sc.wave_gemm = o.wave_gemm;
        sc.attn_tile = o.attn_tile;
        sc.attn_touch = o.attn_touch;
        sc.attn_fattn = o.attn_fattn;
        sc.attn_fattn_roll = o.attn_fattn_roll;
        sc.attn_fattn_qstage = o.attn_fattn_qstage;
        sc.attn_fattn_pvf16 = o.attn_fattn_pvf16;
        sc.attn_fattn_wide = o.attn_fattn_wide;
        sc.attn_split_cap = o.attn_split_cap;
        // The dump sink, read by nothing but the dump itself (see `DecodeScratch::attn_dump_dir`). It
        // points into `o.out_dir`, which lives as long as the run.
        sc.attn_dump_dir = o.dump_attn_rows ? o.out_dir.c_str() : nullptr;
        sc.gdn_scan = o.gdn_scan;
        sc.gdn_chunk = o.gdn_chunk;
        sc.gdn_spell = o.gdn_spell;
        sc.gdn_loop = o.gdn_loop;
        sc.gdn_norm = o.gdn_norm;
        sc.gdn_geom = o.gdn_geom;
        sc.proj_plan = o.proj_plan;
        sc.proj_shape = o.proj_shape;
        sc.cast_fold = o.cast_fold;
        sc.copy_fold = o.copy_fold;
        if (engine::kernels::wave_gemm_is_library(o.wave_gemm, engine::kMaxWaveRows)) {
            // Pin filter first: it decides what `gemm_route_plan` below fetches and what the version
            // line counts, so setting it after the plan would print one route and run another.
            engine::kernels::gemm_pins_select(o.gemm_pins);
            // Only when named, so the engine's own constant stays the single default. Before the plan
            // for the same reason as the pin filter: the pool width is decided when each shape's plan is
            // built, and a setting applied after the plan would print one width and run another.
            if (o.gemm_heur != -1) engine::kernels::gemm_heuristic_width_select(o.gemm_heur);
            // Created here, once, before the first body: the handle and the pinned workspace must not
            // be first touched 400 launches into a prefill. Only when the route is actually on, so a
            // `--gemm hand` run allocates nothing the pre-route run did not.
            const std::string route = engine::kernels::gemm_route_init(o.wave_gemm);
            if (o.wave_gemm_given || o.wave_gemm == engine::kernels::WaveGemm::kLibraryLt) {
                // Printed whenever lt is the route in force, not only when the flag named it: lt is now
                // the shipped default, and a log that does not say which algo each shape ran cannot be
                // re-read next month.
                std::printf("GEMM %s\n", route.c_str());
                // The lt route's algorithm for each (shape, M) is fetched NOW, before the first body,
                // and printed: 150 library calls a wave cannot each pay for a heuristic query, and a
                // timing window must not contain library bookkeeping that the other side of the A/B
                // does not pay. One line per shape, so the algo actually in force is in the log.
                const std::string plan = engine::kernels::gemm_route_plan(o.wave_rows);
                if (!plan.empty()) std::printf("%s", plan.c_str());
            }
        } else if (o.wave_gemm_given) {
            // The A/B side names itself too: a log that does not say which GEMM its wide bodies used
            // cannot be read next to the other side's. Printed only when the flag was named, for
            // `--mlp-map`'s reason — a default run's stdout has to stay byte-stable against the gate
            // logs taken before this flag existed.
            std::printf("GEMM route=hand\n");
        }
        if (o.mlp_gate_map_given) {
            // Only printed when the flag was named: a default run's stdout has to stay byte-stable,
            // because gate logs are diffed against logs taken before this switch existed. A run that
            // did name it says which walk it ran, so no A/B row can be anonymous.
            std::printf("MLPMAP gate=%s\n",
                        engine::kernels::mlp_gate_map_name(sc.mlp_gate_map));
        }
        engine::prepare_weight_adapters(store, sc);

        // §09: the draft side computation, allocated only when it is on. `resolve()` is the block's own
        // shape check against the shard — 15 tensors, rank and both dims stated per tensor — and it
        // throws naming the first disagreement, which is the form AGENTS.md prescribes for a shape claim
        // ("state the rank, and put the contract at the use site"). Nothing here touches `sc` or `st`.
        std::unique_ptr<MtpSide> mtp;
        if (o.mtp_draft) {
            mtp = std::make_unique<MtpSide>();
            mtp->w = engine::mtpdraft::resolve(store);
            mtp->st = engine::mtpdraft::allocate_state(o.mtp_cap);
            // Set before the first `MTPDRAFT` line, so the census can state the convention it used rather
            // than leaving a reader to assume the default. `0` is §09.1's shipped reading.
            mtp->st.pos_shift = o.mtp_pos_shift;
            mtp->track =
                engine::mtpdraft::RoundTracker(engine::mtpdraft::parse_schedule(o.mtp_sched));
            // The ladder's width, and whether it was asked for at all. `k_given` — not `k` — is what arms
            // the extra rows and the `DRAFTK`/`MTPDRAFTK` lines, so an unnamed flag leaves the side pass
            // submitting exactly what it shipped with, and `--mtp-draft-k 1` runs that same path while
            // still printing the ladder fields (which is what makes the k=1 control a measurement).
            mtp->k = o.mtp_draft_k;
            mtp->k_given = o.mtp_draft_k_given;
            // The launch census, stated as its own line rather than smuggled into `LAUNCH`:
            // `planned_launch_counts()` is §07's bill and `gate.py` section (d) re-derives it in Python,
            // so a bill that moved with a flag would make that cross-check a tautology — the `--attn-tile`
            // trap of a gate judging a different plan than the engine ships. This line says what the DRAFT
            // adds, per round, in the vocabulary `CASTFOLD`/`COPYFOLD` already use.
            mtp_census_line(*mtp, o.session ? "session" : "run",
                            engine::mtpdraft::schedule_name(mtp->track.sched));
        }

        // §09 / P7: the verify probe's checkpoint, allocated only when it is on. `snapshot.bytes()` is
        // DERIVED from `state_sizes()` (18.0 MiB of GDN state plus the F32 conv window — 1.265 MiB, not the
        // 0.63 MiB `MTP-IMPL.md` §1 carries, which predates the 2026-10-11 correction to F32), so the
        // printed per-round checkpoint cost cannot be stale in the direction of flattering the route.
        // The bytes are the whole rollback surface a length field cannot undo: KV stays a length
        // (see `engine/mtp_verify.h`), and `download_state` compacts KV to `prefix_len`, so what the shadow
        // body writes above P is neither fingerprinted nor reachable.
        std::unique_ptr<VerifySide> verify;
        if (o.mtp_verify) {
            verify = std::make_unique<VerifySide>();
            verify->route = o.mtp_route;
            verify->guard = o.verify_guard != 0;
            verify->midsplit = o.mtp_midsplit;
            verify->midsplit_mode = o.mtp_midsplit != 0;
            // The mid compare needs the mixer state the step left, and that is captured by the guard, so
            // the witness implies the guard rather than asking for it to be spelled twice. It is stated
            // on the MIDSPLIT lines' own terms: this is a correctness run and it says so.
            if (verify->midsplit_mode) verify->guard = true;
            verify->snap = engine::mtpverify::allocate_snapshot(st);
            verify->lanes = engine::mtpverify::make_lane_stage(st);
            if (verify->midsplit_mode) {
                // V1's third region pair: the mid-round checkpoint, sized from `state_sizes()` so the
                // memory arithmetic is derived, not asserted (per-request model state is 18.66 MiB, so
                // this is a third of that again — fine for this model, and recorded in ACCEPT-BUILD §4).
                const engine::StateSizes sz = st.sizes();
                verify->mid_bytes = (long long)sz.gdn_state_bytes + (long long)sz.conv_window_bytes;
                if (hipMalloc((void**)&verify->mid.mid_gdn, sz.gdn_state_bytes) != hipSuccess ||
                    hipMalloc((void**)&verify->mid.mid_conv, sz.conv_window_bytes) != hipSuccess) {
                    throw std::runtime_error("cannot allocate the mid-round checkpoint");
                }
                if (hipMalloc((void**)&verify->verdict_dev,
                              (size_t)engine::kernels::kVerdictWords * sizeof(int)) !=
                    hipSuccess) {
                    throw std::runtime_error("cannot allocate the verdict status word");
                }
                std::printf("MIDSPLIT on=1 checkpoint_bytes=%lld status_bytes=%zu\n",
                            verify->mid_bytes, (size_t)engine::kernels::kVerdictWords * sizeof(int));
            }
            const hipError_t le = hipMalloc((void**)&verify->logits_dev,
                                            (size_t)engine::mtpverify::kRows * engine::kVocab *
                                            sizeof(float));
            if (le != hipSuccess) {
                throw std::runtime_error(std::string("hip: ") + hipGetErrorString(le) +
                                         " allocating the verify logits tile");
            }
        }

        const std::string inv = st.check_invariants();
        if (!inv.empty()) {
            std::fprintf(stderr, "decode_run: fresh state violates §08.6: %s\n", inv.c_str());
            return 1;
        }

        if (o.session) {
            session_loop(store, st, sc, mtp.get());
            if (mtp) engine::mtpdraft::free_state(mtp->st);
            engine::kernels::gemm_route_shutdown();
            engine::free_scratch(sc);
            engine::free_state(st);
            return 0;
        }

        std::FILE* log = nullptr;
        // A round is not a step, so the per-step dumps (`sampled.txt`, `logits<step>.bin`) are not what an
        // accept run produces: a round prices two logits rows and its verdict carries no top-logit value,
        // and writing a five-column row with a made-up top value would be exactly the fake number this
        // driver keeps refusing. With the flag on, `sampled.txt` is not opened and `emitted.txt` — one
        // emitted token per line, in order — is, because that is the object the losslessness claim is
        // about: the token stream, compared against a flag-off free run.
        if (!o.out_dir.empty() && !o.mtp_accept) {
            log = std::fopen((o.out_dir + "/sampled.txt").c_str(), "w");
            if (!log) throw std::runtime_error("cannot open " + o.out_dir + "/sampled.txt");
            std::fprintf(log, "# step input_token sampled_token top_logit prefix_len_before\n");
        }

        // ---- the accept loop's own side state (`engine/mtp_accept.h`) ------------------------------
        // Allocated on the flag and nowhere else: with `--mtp-accept 0` this pointer is null, no region is
        // reserved, and the generation loop below takes the branch it took before the flag existed.
        std::unique_ptr<engine::mtpaccept::Side> aside;
        engine::mtpaccept::Stats astats;
        // The route the rounds actually ran, off each round's `Round` (it cannot vary within a run: the flag
        // and the scratch are fixed before the first round). Printed on `MTPSTAT` beside the flag so a binary
        // that ignored the switch cannot report the amortised arm while launching the hand chain.
        const char* accept_gemm_ran = "-";
        const char* accept_scan_ran = "-";
        const char* accept_attn_wide_ran = "-";
        std::FILE* emit_log = nullptr;
        int pending_candidate = -1;   // the draft's proposal for the next position; -1 = nothing pending
        int pending_pos = -1;         // the position that proposal is for, echoed on the round line
        if (o.mtp_accept) {
            aside = std::make_unique<engine::mtpaccept::Side>(engine::mtpaccept::allocate_side(st));
            if (!o.out_dir.empty()) {
                emit_log = std::fopen((o.out_dir + "/emitted.txt").c_str(), "w");
                if (!emit_log) throw std::runtime_error("cannot open " + o.out_dir + "/emitted.txt");
                std::fprintf(emit_log, "# emitted_token\n");
            }
            std::printf("MTPACCEPT on=1 rows=%d checkpoint_bytes=%lld status_bytes=%zu"
                        " draft_sched=%s per_step_dumps=0\n",
                        engine::mtpaccept::kRows, aside->bytes(),
                        (size_t)engine::kernels::kVerdictWords * sizeof(int), o.mtp_sched.c_str());
            std::fflush(stdout);
        }

        // ---- the per-phase round meter (`engine/mtp_meter.h`) -------------------------------------
        // One instrument per run, armed here and disarmed after the report, and constructed only when the
        // flag is on: flag-off, every `PhaseGuard`/`WindowGuard` in the engine reads one null pointer and
        // does nothing, which is what keeps a `--mtp-meter 0` run byte-identical to a binary that was not
        // compiled with this line. The role is whichever round shape this run is in — the two are mutually
        // exclusive by the refusals above, so there is exactly one subject to meter and `--mtp-meter` alone
        // is refused rather than metering the plain step.
        //
        // `stage_clock` is deliberately NOT reused. It prices a STEP's §07 launch-plan families and
        // cross-checks them against the compiled plan; a round's phases are not families, and a round's body
        // is one `submit_wave_body` call whose per-token bill is already accounted elsewhere. Folding the
        // two together would bill the body's rows twice and name the second billing as the round.
        std::unique_ptr<engine::mtpmeter::Meter> meter;
        if (o.mtp_meter) {
            meter = std::make_unique<engine::mtpmeter::Meter>(
                o.mtp_accept ? engine::mtpmeter::Role::kAccept : engine::mtpmeter::Role::kVerify);
            engine::mtpmeter::arm(meter.get());
            // The canary line the harness greps for, printed BEFORE the calibration that the report takes:
            // `on=1` has to be a true statement about the run that is about to happen, not about its summary.
            // The idle-sync floor is deliberately NOT measured here — 64 empty waits is a good millisecond of
            // device time, and taking it here would put the instrument's own calibration inside the prefill
            // window every other number in this run is read against. It is taken after the loop instead.
            std::printf("MTPMETER role=%s on=1 sync_added_phases_inflate=1"
                        " plan=prototype/mtp_accept/METER-PLAN.md\n",
                        engine::mtpmeter::Role::kAccept == meter->role() ? "accept" : "verify");
            std::fflush(stdout);
        }

        // ---- the prompt phase: ONE prefill wave (§06.1) ---------------------------------------
        // This was the front of the loop below: a `decode_step` per prompt token, its sample thrown
        // away except at the last position. `prefill_wave` makes the same launches in the same order
        // and gives the phase one owner — the positions, the §06.5 end condition, and the fact that a
        // wave has no stop check to arm. `--prefill-logits all` prices every row, which is the
        // byte-identity mode (same launches per token as the loop it replaces); the default `last`
        // prices only the last row, which is all §06.5 uses.
        const int prompt_len = (int)o.prompt.size();
        const int total_steps = prompt_len + o.steps;
        // ---- the prompt-norm-row retention the draft-cache seeding reads (`--mtp-prompt-lanes`) --------
        // The seeding needs one post-final-norm row per seeded lane and `sc.norm` only ever holds the LAST
        // wave's rows (`PrefillResult::last_wave_rows` — 46 of them for a 4,142-token prompt split at 512),
        // so the rows are copied out of the scratch DURING the wave. The depth is decided here, before the
        // prompt is fed, because the ceiling and the run's own reserve are both known here and nowhere else:
        // sizing the window is the same arithmetic the clamp below applies, and a second spelling of it in
        // a second place is the two-copies-of-one-geometry-fact failure `AGENTS.md` records.
        //
        // OFF IS SILENTLY ABSENT: with `--mtp-prompt-lanes 0` (the default) nothing below allocates and
        // `prefill_wave` receives a null sink, so the prompt phase submits exactly what it used to.
        engine::NormTailRetention keep;
        NormTailBuf keep_mem;
        int seed_ceiling = 0, seed_reserve = 0, seed_room = 0;
        const char* seed_clamp = "none";
        if (mtp && o.mtp_prompt_lanes > 0 && prompt_len > 0) {
            // Lanes this run still needs for its own rounds. A round appends at most two (its own draft
            // lane plus a dense lane when the accept path is on and the round is accepted), and it emits at
            // least one token, so rounds <= steps/(1+a) and the smallest `a` this project admits is the
            // 0.40 M3 kill line, hence 2*steps/1.4. Conservative by construction — a round that is REJECTED
            // appends a lane and then truncates it — and it is what keeps seeding from turning the run into
            // a `DRAFT-SKIP` halfway through, which would be a population change dressed as a rate.
            seed_reserve = (o.steps * 20 + 13) / 14;
            seed_ceiling = engine::mtpdraft::usable_lanes(mtp->st);   // the cache is empty here
            seed_room = seed_ceiling - seed_reserve;
            if (seed_room < 0) seed_room = 0;
            // The draft block's attention is the one-pass form and refuses more than
            // `kernels::mtp_attn_max_lanes()` lanes, so a fan wider than the cache is unreachable however
            // large a prompt is retained; `seed_room` is that ceiling minus what the run still needs.
            const int depth = std::min(prompt_len, std::min(o.mtp_prompt_lanes, seed_room));
            // WHY the window is narrower than the request, decided HERE because this is where the three
            // limits are all visible, and printed below as `clamped=`. Reporting `norm_rows_retained` when
            // the real cause was the cache ceiling would blame the retention for a limit it did not apply.
            seed_clamp = "none";
            if (depth < o.mtp_prompt_lanes) {
                seed_clamp = (seed_room < o.mtp_prompt_lanes) ? "cache_ceiling_minus_reserve"
                                                             : "prompt_shorter_than_request";
            }
            if (depth > 0) {
                if (hipMalloc((void**)&keep_mem.p, (size_t)depth * engine::kHidden * sizeof(uint16_t)) !=
                    hipSuccess) {
                    throw std::runtime_error("--mtp-prompt-lanes: cannot allocate " +
                                             std::to_string((long long)depth * engine::kHidden *
                                             (long long)sizeof(uint16_t)) +
                                             " B for " + std::to_string(depth) +
                                             " retained prompt norm rows");
                }
                keep.buf = keep_mem.p;
                keep.rows = depth;
            }
        }
        std::vector<float> logits;
        double step_ms = 0.0;   // the run's step time, by-product not measurement (§6)
        const auto t_wave = std::chrono::steady_clock::now();
        const engine::PrefillResult wave =
            engine::prefill_wave(store, st, sc, o.prompt, o.prefill_logits,
                                 keep.buf ? &keep : nullptr);
        // Added to the same accumulator the per-step timings went into, so RESULT's `ms_total` still
        // means "this run" rather than silently becoming generation-only. Kept as its own number as well
        // (and not left inside the accumulator) so `DECWALL` can show the prefill leg rather than make a
        // reader recover it by subtracting one run from another — the mistake `DECWALL` exists to retire.
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wave)
                .count();
        step_ms += prefill_ms;
        std::printf("PREFILL tokens=%zu logits=%s prefix_len=%d wave_rows=%d", wave.size(),
                    engine::prefill_logits_name(wave.logits_mode), st.prefix_len, sc.wave_rows);
        if (!wave.empty()) {
            std::printf(" pos=%d..%d", wave.rows.front().pos_before, wave.rows.back().pos_after - 1);
        }
        std::printf("\n");

        for (const engine::PrefillRow& row : wave.rows) {
            // ---- §07.5's stop check is not armed here, and cannot be. -------------------------
            // There is no `stop_fired` in this loop: a prefill row is prompt-phase by construction, so
            // the only place a stop can be raised is the generate loop below. That used to be a
            // `!from_prompt` conjunct on one shared loop and the first device run got it wrong: the
            // `dec8` prompt is the manifest's `sources[0].ids` — a rendered chat template — so it
            // legitimately contains id 248046 (the end-of-turn token `kStopToken` is) at index 8, and
            // 248068/248069 later; and on that first run the engine's own argmax at prompt index 6 WAS
            // 248046, which ended a 17-token prompt after 7 steps, produced 0 generation steps, and
            // failed six gate checks that all said something else. A prompt-phase match is still
            // printed, below, as a diagnostic; it is not a stop and do not "restore" it.
            //
            // One honest consequence of `--prefill-logits last`: an intermediate row was never
            // argmaxed, so this diagnostic can only fire on a row the wave actually priced. The rule
            // does not get weaker for that — there is simply no sample to misread — and
            // `engine/gate.py --break-prompt-stop` injects the bug in the `all` row shape, where the
            // intermediate sample exists again.
            const bool priced = row.sampled >= 0;
            const bool stop_match = row.sampled == engine::kStopToken;
            if (priced) {
                std::printf("prefill %d | fed %7d | sampled %7d | top %.6f | P %d -> %d%s\n",
                            row.index, row.token, row.sampled, row.top_logit, row.pos_before,
                            row.pos_after,
                            // Stays visible as a diagnostic of the prompt path (and a warning that the
                            // fixture's prompt contains the stop id).
                            stop_match ? " [stop id in prompt phase — ignored]" : "");
                // The machine-readable twin. `stop=` is the STOP CHECK (armed, matched, run over), and
                // a prefill step can never arm it: `gate.py` section (a) asserts exactly that.
                std::printf("STEP kind=prefill index=%d input=%d sampled=%d top_logit=%.9g"
                            " pos_before=%d pos_after=%d stop=0\n",
                            row.index, row.token, row.sampled, row.top_logit, row.pos_before,
                            row.pos_after);
            } else {
                // Nothing was priced for this row, so the sample column says so instead of inventing
                // an id. `sampled=-1` is `RequestState::sampled_token`'s existing "nothing pending"
                // marker, and `top_logit=none` keeps the field present for a parser without claiming a
                // value it does not have.
                std::printf("prefill %d | fed %7d | sampled       - | top       - | P %d -> %d\n",
                            row.index, row.token, row.pos_before, row.pos_after);
                std::printf("STEP kind=prefill index=%d input=%d sampled=-1 top_logit=none"
                            " pos_before=%d pos_after=%d stop=0\n",
                            row.index, row.token, row.pos_before, row.pos_after);
            }
            if (log) {
                std::fprintf(log, "%d %d %d %.9g %d\n", row.index, row.token, row.sampled,
                             row.top_logit, row.pos_before);
                if (!row.logits.empty()) {
                    write_raw(o.out_dir, "logits" + std::to_string(row.index) + ".bin", row.logits);
                }
            }
        }

        // `--state-fp` observes the two end conditions rather than deriving them: the prompt-end
        // snapshot is taken at the phase boundary — after the wave, before the first generation step —
        // which is where §08.4's "after the last prompt token" actually is. There is no longer a
        // third case: the wave either completes or throws, so a run cannot end "inside the prompt"
        // any more (the pre-P2 `run_ended_inside_the_prompt` note existed because the stop check could
        // fire mid-prompt, and §07.5 says it must not — now it structurally cannot).
        HostState prompt_end;
        if (o.state_fp) {
            prompt_end = download_state(st);
            if (o.steps > 0) emit_state_fp("prefill", prompt_end, nullptr);
        }

        // ---- the draft cache's prompt seeding (`--mtp-prompt-lanes`, default 0 = nothing happens) ----
        // The shipped draft cache starts EMPTY: the block's first attention row sees one lane, and it
        // only ever accumulates one lane per round. llama.cpp's draft context is not built that way — its
        // catch-up `process()` runs over the prompt ubatches too, so its drafter attends over the whole
        // prompt. This is the counterpart, and it is deliberately placed BEFORE the decode window opens:
        // it is prompt-phase work, and billing it inside `DECWALL`'s window would make a rate probe's
        // tok/s line mean something other than what it usually means.
        //
        // The lane for prompt position p pairs the token AT p with the target's norm row at p-1, and is
        // rope-staged at position p+1 — the same content-to-position map `draft_round` uses (the lane's rope
        // position is the position of the token it DRAFTS, one past the token it feeds; §09.1, and
        // `--mtp-pos-shift` exists because that convention was measured against llama.cpp's one-less and
        // found to be a null), so the seeded lanes and the round lanes are the same kind of object. The last
        // prompt position's lane uses the wave's own `y0` as the token, which is what makes the seeded run
        // contiguous with the first generated lane rather than one position short.
        //
        // THE ROWS COME FROM THE RETENTION, not from `sc.norm`. They used to come from `sc.norm`, which is
        // why the widest seed that ever existed was one wave: the fan was clamped to
        // `PrefillResult::last_wave_rows` — 46 lanes on a 4,142-token prompt, and the reason a
        // `--mtp-prompt-lanes 128` run silently meant 46. Now the rows the wave would otherwise drop are
        // copied into a window before they are overwritten, and what limits the fan is the draft block's own
        // attention form (`mtp_attn_max_lanes()`, the one-pass staging) minus the room this run still needs
        // for its own rounds. Every one of those three numbers is printed, and the old clamp is printed
        // beside them as `last_wave_rows=` so a reader can see WHICH clamp bound this run rather than
        // inferring it.
        if (mtp && o.mtp_prompt_lanes > 0 && !wave.empty()) {
            const int norm_rows = keep.rows_written;     // retained rows that actually exist
            const int h_lo = keep.first_pos;             // prompt position held by retention row 0
            const engine::PrefillRow* y0row = wave.y0();
            const int y0 = y0row ? y0row->sampled : -1;
            const int p_hi = y0 >= 0 ? prompt_len : prompt_len - 1;
            const int p_lo = norm_rows > 0 ? std::max(h_lo + 1, 1) : p_hi + 1;
            const int avail = p_hi - p_lo + 1;
            // `seed_ceiling`/`seed_reserve`/`seed_room` were computed above, before the prompt was fed —
            // the same numbers that sized the retention, so the clamp and the allocation cannot disagree.
            int count = o.mtp_prompt_lanes;
            const char* clamped = "none";
            if (count > avail) {
                count = avail;
                // The retention was sized by the same three limits, so if one of them cut the window the
                // reason was decided at sizing and `seed_clamp` already names it. Only claim the rows
                // themselves when nothing upstream bound.
                clamped = (seed_clamp[0] == 'n') ? (norm_rows > 0 ? "norm_rows_retained" : "nothing_retained")
                                                 : seed_clamp;
            }
            const int room = seed_ceiling - (int)mtp->st.nlanes - seed_reserve;
            if (count > room) { count = room; clamped = "cache_ceiling_minus_reserve"; }
            if (count < 0) count = 0;
            for (int p = p_hi - count + 1; p <= p_hi; ++p) {
                const int tok = (p < prompt_len) ? o.prompt[p] : y0;
                const int hrow_idx = (p - 1) - h_lo;
                // An off-by-one here reads the WRONG norm row — a plausible lane built from the neighbouring
                // position's hidden state, which no device check would present as a fault. So the bound is
                // asserted, not assumed, and the operands it depends on are printed on `MTPPROMPT`.
                if (hrow_idx < 0 || hrow_idx >= norm_rows) {
                    throw std::runtime_error(
                        "--mtp-prompt-lanes: lane for position " + std::to_string(p) + " wants retained row " +
                        std::to_string(hrow_idx) + " but the retention holds " + std::to_string(norm_rows) +
                        " rows from position " + std::to_string(h_lo) + " (window asked: " +
                        std::to_string(keep.rows) + "). Refusing rather than staging a lane from the wrong row.");
                }
                const uint16_t* hrow = keep.buf + (size_t)hrow_idx * engine::kHidden;
                engine::mtpdraft::draft_lane(store, mtp->w, mtp->st, hrow, tok, p + 1);
            }
            mtp->prompt_lanes = count;
            if (mtp->st.lanes_max > mtp->lanes_max) mtp->lanes_max = mtp->st.lanes_max;
            // The cost of this, in COUNTS and not milliseconds (which are not this run's to report):
            // `retain_copies`/`retain_bytes` is what the wave paid to keep the rows — one D2D per wave that
            // intersected the window — and `lane_subs` is the block submissions the staging paid, 9 per
            // lane with no head pass (`engine/mtp_draft.h::kLaneStageSubs`). Against what it is amortised:
            // `rounds=` on this run's own `MTPDRAFT` line.
            std::printf("MTPPROMPT wanted=%d fitted=%d fan=%d norm_rows=%d retained_rows=%d"
                        " retain_copies=%lld retain_bytes=%lld last_wave_rows=%d pos=%d..%d"
                        " lanes_after=%d ceiling=%d reserve=%d lane_subs=%lld clamped=%s\n",
                        o.mtp_prompt_lanes, count, count, norm_rows, keep.rows_written,
                        (long long)keep.copies, (long long)keep.bytes, wave.last_wave_rows,
                        p_hi - count + 1, p_hi, mtp->st.nlanes, seed_ceiling, seed_reserve,
                        (long long)count * engine::mtpdraft::kLaneStageSubs, clamped);
            std::fflush(stdout);
        }

        // ---- the run's own decode window (`DECWALL`, closed and printed after the loop) -------------
        // One wall clock around the generation loop, and the counts that make its rate mean something.
        // Why the counts are here rather than recovered afterwards: a round can end the loop early (§07.5's
        // stop, `RUN-END reason=context_exhausted`, `ROUND-SKIP reason=context_limit`), so `--steps` is not
        // the work a run did, and no arithmetic across two runs can make it so.
        const auto t_decode = std::chrono::steady_clock::now();
        long long gen_emitted = 0;     // tokens handed to the stream, counted AT THE EMIT SITE
        long long gen_bodies = 0;      // bodies executed: one per plain step, one per accept round
        long long gen_plain = 0;       // bodies that were single-row steps (every body, with no flag on)
        long long gen_suppressed = 0;  // an accept's bonus behind a stop: priced, not emitted

        int fed = 0;
        // §06.5: the first token fed here is the wave's own `y0`, produced inside the last wave row,
        // so no extra decode step is spent on the first sampled token.
        int sampled = wave.empty() ? -1 : wave.rows.back().sampled;
        bool run_stopped = false;   // §07.5 reached in the GENERATION phase; see the rule below
        // Which exit the loop actually took. `RESULT stopped=` cannot say: it is one boolean, and both §07.5's
        // stop and a request that ran out of context set it. `DECWALL ended=` names the reason, because a
        // short run is normal for an accept loop (a round consumes one or two lanes) and a reader has to be
        // able to tell "emitted everything there was room for" from "the model ended the turn".
        const char* gen_end = "steps_exhausted";
        // One row per generation step the run actually took, for `--time-stages`, which replays exactly
        // these steps at the other two submission granularities. Inert when the flag is off: a vector of
        // PODs filled from values the loop already had, no device work and no sync.
        std::vector<StepRec> step_records;
        for (int i = 0; i < o.steps; ++i) {
            const int step = prompt_len + i;
            // `--replay`: the caller owns the generation-phase tokens, exactly as it owns the
            // prompt's. The engine still samples every step and prints that sample, so a step
            // whose fed token and sampled token disagree is precisely the divergence the gate is
            // looking for — and because the feeding is not self-propagating, that divergence
            // cannot cascade into the steps after it. Without `--replay` the engine feeds itself,
            // which is what a free run does and what A-18 says cannot be pinned past ~8 tokens.
            const int token = o.feed.empty() ? sampled : o.feed[i];
            const int pos_before = st.prefix_len;
            if (aside) {
                // ---- one accept round, in place of this step (§09 / M2, `engine/mtp_accept.h`) ------
                // `pending_candidate` is the draft block's proposal for position `pos_before + 1`. Where
                // there is none — the first step of every run, and every step after a draft refusal — the
                // run pays a plain one-row step, so speculation is never a precondition for progress.
                //
                // What a round OWES and what it gets for free is in `engine/mtp_accept.h`: an accept costs
                // no restore at all (the body's end state IS the accepted state) and a reject restores the
                // mid-round checkpoint in two bulk copies. Nothing here is printed as a rate without the
                // counters that produced it; `MTPSTAT` at the end carries the same numbers.
                // End of context, handled as an end condition rather than as a throw. `--steps` sizes the
                // request's `cap` for one lane per step, and a round consumes one or two, so a run that
                // speculates successfully can reach `cap` with steps left in its budget — where the plain
                // loop cannot. `submit_wave_body` and `decode_step` both refuse to write past `cap` (that
                // refusal is `§08.6`'s), so the caller owns the stop, same as §07.5's rule.
                if (st.prefix_len >= st.cap) {
                    std::printf("RUN-END reason=context_exhausted pos=%d cap=%d budget_left=%d\n",
                                st.prefix_len, (int)st.cap, o.steps - i - 1);
                    run_stopped = true;
                    gen_end = "context_exhausted";
                    break;
                }
                const uint16_t* draft_norm_row = nullptr;   // the norm row of the last position FED
                // `--mtp-dense`: the lane for the position an accepted round consumed but did not stage.
                // Left at -1 on a reject and on a plain step.
                int dense_token = -1, dense_pos = -1;
                int emitted_first = -1, emitted_second = -1;
                int pos_first = -1, pos_second = -1;
                // Headroom first: a round appends TWO KV lanes, and the last lane of the request has room
                // for one. `accept_round` refuses that case (writing past `cap` is a memory fault), and a
                // refusal thrown at the end of an otherwise good run would report a failure where the run
                // simply ran out of context — so the driver asks before it calls, and pays a plain one-row
                // step. Same fallback shape as "no proposal pending", for a different reason.
                const bool room_for_round =
                    st.prefix_len + engine::mtpaccept::kRows <= st.cap;
                // The meter's window covers THIS branch's whole round: the round itself, the next
                // proposal's draft, and the run's own per-round prints. A plain one-row iteration is not a
                // round and is counted in `plain_steps=` instead of being metered, because pricing it here
                // would mix a step's cost into a round's denominator. `meter_round` says whether a window
                // is open, and it is closed on every exit path below — including the stop-token break.
                bool meter_round = false;
                if (pending_candidate < 0 || !room_for_round) {
                    if (pending_candidate >= 0 && !room_for_round) {
                        std::printf("ROUND-SKIP reason=context_limit anchor=%d cap=%lld rows=%d\n",
                                    st.prefix_len, (long long)st.cap, engine::mtpaccept::kRows);
                    }
                    const engine::StepResult r = engine::decode_step(store, st, sc, token, nullptr);
                    emitted_first = r.token;
                    pos_first = st.prefix_len;
                    draft_norm_row = sc.norm;              // row 0 is the row this step priced
                    gen_bodies++;
                    gen_plain++;
                    if (meter) meter->note_plain_step();
                    // NOT gen_emitted++: this token goes out through the shared `emit_one` tail below, like
                    // a round's first token does. Counting it here as well double-counted every plain step of
                    // an accept run, and `DECWALL note=emitted_vs_verdict_tally` is what caught it.
                } else {
                    if (meter) {
                        meter->round_begin();
                        meter->open_window();
                        meter_round = true;
                    }
                    const engine::mtpaccept::Round ar =
                        engine::mtpaccept::accept_round(store, st, sc, *aside, token,
                                                       pending_candidate, o.mtp_route);
                    accept_gemm_ran = engine::kernels::wave_gemm_name(ar.gemm_ran);
                    accept_scan_ran = engine::kernels::gdn_scan_name(ar.scan_ran);
                    accept_attn_wide_ran = engine::kernels::attn_wide_name(ar.attn_wide_ran);
                    engine::mtpaccept::tally(astats, *aside, ar);
                    gen_bodies++;   // one round is one 2-row body, whatever it emits
                    mtp->adjudicated++;
                    if (ar.accepted) {
                        mtp->accepted++;
                        emitted_first = pending_candidate;
                        emitted_second = ar.bonus_token;
                        pos_first = ar.anchor_after - 1;
                        pos_second = ar.anchor_after;
                        draft_norm_row = sc.norm + engine::kHidden;   // row 1 is position p+1's row
                        if (o.mtp_dense) {
                            // The round fed `cur` at p and the candidate at p+1 and the next draft lane
                            // stands for p+2, so position p+1 is the one the shipped loop skips. Its
                            // content is exactly the row the body fed there — the candidate token with the
                            // ROW-0 norm (h at p) — and both are still in hand: the body wrote row 0 and
                            // row 1 of `sc.norm` and nothing has overwritten the tile since. Staging it is
                            // 9 submissions and no head pass, and it touches no operand of the round.
                            dense_token = pending_candidate;
                            dense_pos = pos_first + 1;    // this engine's content->position map
                        }
                    } else {
                        emitted_first = ar.verifier_token;
                        pos_first = ar.anchor_after;
                        draft_norm_row = sc.norm;
                        // The rejected proposal wrote a draft lane the target never consumed. Truncating
                        // it is a length field and moves 0 bytes (`engine/mtp_draft.h`), and it is what
                        // keeps the draft cache the accepted prefix rather than the attempted one — the
                        // same move the read-only side pass makes on a reject. Charged to
                        // `lane_bookkeeping` beside the §08.6 check, because both are the round's length-
                        // field bookkeeping and neither moves a byte.
                        const engine::mtpmeter::PhaseGuard lane(engine::mtpmeter::kLaneBookkeeping);
                        engine::mtpdraft::truncate_lane(mtp->st);
                    }
                    {
                        const engine::mtpmeter::PhaseGuard prints(engine::mtpmeter::kReportPrints);
                        std::printf("ACCEPT round=%d anchor=%d fed=%d cand=%d cand_pos=%d accept=%d"
                                    " verifier=%d bonus=%d emitted=%d pos=%d->%d restored_bytes=%lld\n",
                                    step, ar.anchor_before, token, pending_candidate, pending_pos,
                                    ar.accepted ? 1 : 0, ar.verifier_token, ar.bonus_token, ar.emitted,
                                    ar.anchor_before, ar.anchor_after, ar.restored_bytes);
                    }
                }                // The next proposal: ONE draft call per round, with the target's final-norm row of the
                // last position the body actually FED (row 1 on an accept, row 0 on a reject or a plain
                // step), `y_next` the last token EMITTED, and the proposed position one past that token.
                // That is `mtp_after_step`'s alignment (`h` at q, `y_next` at q+1, proposal at q+2) with
                // q = the last fed position, and it is one call rather than one per emitted token because
                // an accept's second token was never fed, so no final-norm row of the target exists
                // behind it (`ACCEPT-BUILD.md` §4g.1).
                pending_candidate = -1;
                pending_pos = -1;
                const int tok_next = emitted_second >= 0 ? emitted_second : emitted_first;
                const int draft_pos = (emitted_second >= 0 ? pos_second : pos_first) + 1;
                // The skipped lane goes in BEFORE the round's own lane, because lane order has to stay
                // ascending in position: the attention row treats `self_lane` (the last one) as the newest
                // and reads everything before it.
                if (dense_token >= 0) {
                    const std::string lane_why = engine::mtpdraft::refusal(mtp->st);
                    if (lane_why.empty()) {
                        engine::mtpdraft::draft_lane(store, mtp->w, mtp->st, sc.norm, dense_token,
                                                     dense_pos);
                        if (mtp->st.lanes_max > mtp->lanes_max) mtp->lanes_max = mtp->st.lanes_max;
                    } else {
                        // Refused, not skipped silently: the round still runs, it just keeps the sparse
                        // cache, and the run's rate is therefore a mixture. Printing it is the only way a
                        // reader can tell that from the counters.
                        std::printf("DENSE-SKIP reason=%s would_stage_for=%d\n", lane_why.c_str(),
                                    dense_pos);
                    }
                }
                const std::string why = engine::mtpdraft::refusal(mtp->st);
                if (!why.empty()) {
                    std::printf("DRAFT-SKIP reason=%s would_propose_for=%d\n", why.c_str(), draft_pos);
                } else {
                    // `phase=draft` is the NEXT round's proposal, charged to this round's window because
                    // that is where it is paid for. It ends in the draft block's own blocking vocabulary
                    // D2H, so its close is `sync_added=1` only because the meter does not want the draft's
                    // ~2 ms head pass landing on whichever phase happens to block next.
                    const engine::mtpmeter::PhaseGuard draft(engine::mtpmeter::kDraft);
                    const engine::mtpdraft::Proposal p =
                        engine::mtpdraft::draft_round(store, mtp->w, mtp->st, draft_norm_row, tok_next,
                                                      draft_pos);
                    pending_candidate = p.token;
                    pending_pos = draft_pos;
                    mtp->made++;
                    // `lanes_max` is the census line's only evidence that the draft cache GREW, and in the
                    // accept loop it used to stay 0 because only the side pass updated it — which made
                    // `lanes_max=0 cap=512` a true statement about the counter and a false one about the
                    // cache. Same high-water mark, same field, now from both callers.
                    if (mtp->st.lanes_max > mtp->lanes_max) mtp->lanes_max = mtp->st.lanes_max;
                }
                // §07.5, applied per EMITTED token and in order. The accept's first token is a token the
                // plain loop would never have emitted — it was the draft's guess — and it is exactly as
                // entitled to end the run, because it is now a token the target emitted for real. A bonus
                // behind a stop is not emitted: it was only ever conditional on its predecessor.
                bool stop_fired = false;
                const auto emit_one = [&](int tok, int pos) {
                    if (emit_log) std::fprintf(emit_log, "%d\n", tok);
                    std::printf("ACCEMIT step=%d pos=%d token=%d\n", step, pos, tok);
                    gen_emitted++;   // counted at the emit site, which is what `DECWALL` divides by
                    if (tok == engine::kStopToken && !o.no_stop) stop_fired = true;
                };
                {
                    // The run's own per-round output, named rather than buried in `overhead`. `fflush` is
                    // explicit here because an unflushed stdout lets the writes land in a pipe buffer for
                    // the whole run, and that would be billed to whichever sync follows it.
                    const engine::mtpmeter::PhaseGuard prints(engine::mtpmeter::kReportPrints);
                    emit_one(emitted_first, pos_first);
                    if (emitted_second >= 0 && !stop_fired) {
                        emit_one(emitted_second, pos_second);
                    } else if (emitted_second >= 0) {
                        // The bonus was priced by the body and never emitted, so `MTPSTAT`'s tally (what the
                        // ROUND emitted) and this window (what the RUN emitted) differ by exactly this token.
                        // Counting the difference is what lets `DECWALL` cross-check the two counts instead of
                        // hoping they agree.
                        gen_suppressed++;
                    }
                }
                if (stop_fired) run_stopped = true;
                if (stop_fired) gen_end = "stop_token";
                sampled = emitted_second >= 0 && !stop_fired ? emitted_second : emitted_first;
                if (meter_round) {
                    // Closed here rather than at the top of the next iteration so a stop-terminated run
                    // still has a closed window: an unclosed one throws in the report, and the run would
                    // print a crash where it meant to print a result.
                    meter->close_window();
                    meter->round_end();
                }
                std::fflush(stdout);
                if (run_stopped) break;
                continue;
            }
            // The meter's verify shape: TWO windows, not one. The checkpoint take happens before the plain
            // step (it has to — the step is what destroys the state the body starts from) and the side pass
            // happens after it, so the round's wall is the union of the two spans and the plain step between
            // them is deliberately NOT round work. `overhead` therefore means "host and queue time inside
            // those two spans that no phase owns", and the step's ~10 ms is not in it.
            if (meter) meter->round_begin();
            // The witness's first half, in bytes and on the host: what the STEP is about to read. Held as a
            // copy rather than a pointer because the step is exactly what moves it, and because the whole
            // point of the comparison below is to say which operand the shadow body did not get.
            engine::mtpverify::StateBytes step_reads, step_after;
            if (verify && verify->guard) {
                const engine::mtpmeter::WindowGuard win;
                const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
                step_reads = engine::mtpverify::capture_state_bytes(st, token);
            }
            // The checkpoint, taken BEFORE the body: it has to hold the state the verify body starts from,
            // and after the step that state is gone. This is the per-round cost D3 excluded as out of
            // scope rather than solved, and it is a device-to-device copy of `snapshot.bytes()`.
            if (verify) {
                const engine::mtpmeter::WindowGuard win;
                const engine::mtpmeter::PhaseGuard take(engine::mtpmeter::kSnapTake);
                engine::mtpverify::take_snapshot(st, verify->snap);
            }
            const auto ts = std::chrono::steady_clock::now();
            const engine::StepResult r =
                engine::decode_step(store, st, sc, token,
                                    o.out_dir.empty() ? nullptr : &logits);
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - ts).count();
            step_ms += wall_ms;
            step_records.push_back({step, token, r.token, r.top_logit});
            gen_bodies++;
            gen_plain++;
            gen_emitted++;   // this step's sample goes to the stream, stop token included
            sampled = r.token;
            // The witness's own control: the step is required to have moved the mixer state, so a mixer
            // comparison across it that reads "same" is a comparator that cannot fail. Captured here,
            // adjudicated in `verify_round`, next to the two comparisons it vouches for.
            if (verify && verify->guard) {
                const engine::mtpmeter::WindowGuard win;
                const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
                step_after = engine::mtpverify::capture_state_bytes(st, -1);
            }
            if (mtp) {
                // Generation phase only — this loop is the only place §07.5's stop rule lives, and the
                // draft's rounds start from a decoded step, never from a prompt wave row (whose sample
                // is discarded and must never arm a stop). `r.token` is this step's prediction at
                // position `pos_before`, and `sc.norm` row 0 the row it was priced from.
                mtp_after_step(*mtp, store, sc.norm, pos_before, r.token);
            }
            if (verify) {
                const engine::mtpmeter::WindowGuard win;
                // AFTER the draft, because the draft reads `sc.norm` row 0 and the verify body overwrites
                // rows 0..1 of the scratch. The candidate is the flag's, or the degenerate self-candidate
                // (the token this step fed) when none was named — either way row 1 is a second row at the
                // next position, which is what the body and the batched head are for.
                const int candidate = !o.verify_candidate_given || o.verify_candidate.empty()
                                          ? token
                                      : o.verify_candidate == "sampled" ? r.token
                                                                        : std::stoi(o.verify_candidate);
                verify_round(*verify, store, sc, st, pos_before, token, candidate, r.token, step,
                             o.out_dir, &step_reads, &step_after);
            }
            if (meter) meter->round_end();

            const bool stop_match = r.token == engine::kStopToken;
            // §07.5 ends a FREE run, in this loop and nowhere else (see the prefill loop for the other
            // half of the rule). A replay is not free: the caller feeds the stream, and dec64f's own
            // forced chain steps PAST the stop id (the reference samples it at generation step 24 and
            // teacher forcing keeps going), so firing here would truncate the exact depth the fixture
            // exists to test. `stop_match` is still printed and the gate still sees `stop=0` with
            // `sampled=248046`, which is the same convention the prompt phase uses.
            //
            // `--no-stop` is the same waiver for a free run, and exists for one reason: a smoke run
            // that is supposed to reach a given depth. On dec8's prompt the model emits the stop id
            // after 38 tokens, so `--smoke 300` would otherwise test 38 steps of state growth and
            // call it 300. Nothing that claims parity uses this flag; the gate's parity path never
            // sets it.
            const bool stop_fired = stop_match && o.feed.empty() && !o.no_stop;
            if (stop_fired) run_stopped = true;

            std::printf("decode %d | fed %7d | sampled %7d | top %.6f | P %d -> %d%s\n",
                        step, token, r.token, r.top_logit, pos_before, st.prefix_len,
                        stop_fired ? " [STOP]" : "");
            // The machine-readable twin of the line above, with the step split into a kind so the
            // gate does not have to re-derive which index the reference's `decode{s}` starts at.
            std::printf("STEP kind=decode index=%d input=%d sampled=%d top_logit=%.9g pos_before=%d"
                        " pos_after=%d stop=%d\n",
                        step, token, r.token, r.top_logit, pos_before, st.prefix_len,
                        stop_fired ? 1 : 0);
            if (log) {
                std::fprintf(log, "%d %d %d %.9g %d\n", step, token, r.token, r.top_logit,
                              pos_before);
                write_raw(o.out_dir, "logits" + std::to_string(step) + ".bin", logits);
            }
            if (stop_fired && step + 1 < total_steps) {
                std::printf("stopped at step %d (§07.5: id %d)\n", step, engine::kStopToken);
                gen_end = "stop_token";
                break;
            }
            fed++;
        }
        // The window closes here, and with an explicit device sync: a reject's restore is a
        // `hipMemcpyAsync` issued AFTER the round's status readback, so without this the last round's two
        // bulk copies would be retired by teardown and billed to nothing. A failed sync is not a reason to
        // lose a finished run's output, so it is recorded and reported, not thrown.
        std::string decode_sync_err;
        {
            const hipError_t se = hipDeviceSynchronize();
            if (se != hipSuccess) decode_sync_err = hipGetErrorString(se);
        }
        const double decode_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_decode)
                .count();
        if (log) std::fclose(log);

        // The side computation's own close: the last round's proposal, if the run ended before a later
        // step could settle it, and the census line that states the denominator it is counting.
        if (mtp) {
            mtp_close_run(*mtp);
            mtp_census_line(*mtp, o.feed.empty() ? "run" : "replay",
                            engine::mtpdraft::schedule_name(mtp->track.sched));
            mtp_draftk_line(*mtp, o.feed.empty() ? "run" : "replay",
                            engine::mtpdraft::schedule_name(mtp->track.sched));
        }

        // The verify probe's close. The restore witness is not optional and not per round: it poisons the
        // shadow with a pattern no body writes, restores, and compares EVERY byte, so "the restore was
        // complete" is a witnessed claim rather than "the restore call ran" — and a partial restore would
        // leave a state that looks plausible and is wrong. Costs two downloads of the mixer state once.
        if (verify) {
            std::string witness = "no_rounds_ran";
            if (verify->stats.rounds > 0) {
                // Fresh checkpoint of the FINAL state, so the witness is about real content rather than
                // whatever the last round's snapshot happened to hold. The copy this performs is the
                // witness's own cost and is NOT added to `restored_bytes`, which counts the restores the
                // rounds actually needed — the two figures answer different questions and folding one into
                // the other would make `restored_bytes == rounds × checkpoint` stop being checkable.
                engine::mtpverify::take_snapshot(st, verify->snap);
                witness = engine::mtpverify::restore_witness(verify->snap);
            }
            verify_census_line(*verify, witness);
            engine::mtpverify::free_snapshot(verify->snap);
            engine::mtpverify::free_lane_stage(verify->lanes);
            if (verify->midsplit_mode) {
                // The admission's own close: what it compared, and whether every comparison came out
                // clean. `clean=1` is the conjunction of the three comparisons this admission can make —
                // split-vs-batched logits, split-vs-batched mixer end state, and the device verdict against
                // `argmax_greedy` — so a reader does not have to re-derive it from three counters, and a
                // run that compared zero rounds cannot read as clean. The mid-vs-step byte drift is NOT in
                // the conjunction: the body's row 0 runs the wave attention path at `rows = 2` while the
                // one-row step ran the fused single-row form, so bit-equality against the step's state was
                // never an admissible ask (the probe's own claim is argmax-level, `row0_eq_step=1`). It is
                // still printed per round as `mid_drift_bytes`, because a drift that grew would be the first
                // evidence of a real rollback bug and a deleted check tells you nothing.
                const bool clean = verify->split_rounds > 0 && verify->split_logit_diffs == 0 &&
                                   verify->split_end_diffs == 0 && verify->verdict_bad == 0 &&
                                   verify->midsplit == 1;
                std::printf("MIDSPLITDONE rounds=%lld split_eq_batch=%d logits_ndiff_total=%lld"
                            " end_ndiff_total=%lld verdict_eq_host_bad=%lld"
                            " checkpoint_bytes_per_round=%lld clean=%d\n",
                            verify->split_rounds,
                            (verify->split_logit_diffs == 0 && verify->split_end_diffs == 0) ? 1 : 0,
                            verify->split_logit_diffs, verify->split_end_diffs, verify->verdict_bad,
                            verify->mid_bytes, clean ? 1 : 0);
                if (verify->mid.mid_gdn != nullptr) (void)hipFree(verify->mid.mid_gdn);
                if (verify->mid.mid_conv != nullptr) (void)hipFree(verify->mid.mid_conv);
                if (verify->verdict_dev != nullptr) (void)hipFree(verify->verdict_dev);
            }
            if (verify->logits_dev) (void)hipFree(verify->logits_dev);   // nodiscard, per
                                                                        // `mtp_draft.cpp`'s teardown
            verify->logits_dev = nullptr;
        }

        if (o.state_fp) {
            if (o.steps > 0) {
                const HostState run_end = download_state(st);
                emit_state_fp("end", run_end, &prompt_end);
            } else {
                // The run never generated anything, so there is one state and it is both end
                // conditions: emitted once, as the prompt end, with the reason printed instead of a
                // second identical block.
                std::printf("STATEFP phase=prefill note=no_generation_steps_ran\n");
                emit_state_fp("prefill", prompt_end, nullptr);
            }
        }

        // The denominator keeps its pre-P2 meaning — one token fed, prompt row or generated step — so
        // these figures stay comparable with the ms/token rows already quoted in `engine/results.md`.
        // P2 moved the prompt's work into the wave (timed as one call above, into the same
        // accumulator), it did not take it out of the run. Still a by-product, never a measurement (§6).
        const int n = (int)wave.size() + fed;
        // `stopped=` here is the RUN's §07.5 decision, which is only ever raised by the generate loop
        // above — that is the whole phase rule now, so there is no prompt-phase contribution to it.
        // `st.stopped` is the raw record `finish_token` leaves ("some priced row argmaxed to the stop
        // id, in either phase"), so it can be true while this says 0, and on `dec8` it is; under the
        // default `--prefill-logits last` fewer prompt rows are priced, so fewer rows can set it.
        // What the engine ACTUALLY launched, as opposed to the `LAUNCH` plan line printed before the
        // run. Printed before `RESULT` so the last line of a run stays the result line (the `tail -1`
        // trap in `AGENTS.md`), and printed always rather than on a flag, because a counter an author
        // can switch off is a counter that gets switched off. `engine/launch_stats.h` says why this
        // and not another output comparison is the proof that P4's batched kernels are the ones
        // running.
        std::printf("%s\n", engine::launchstats::describe_run().c_str());
        // Route D1's other witness, and the one that cannot be satisfied by the flags alone: a shape the
        // device or the width could not offer was COUNTED at the call site that fell back. Printed only when
        // non-zero, so silence means "the A/B side ran the shape it asked for" rather than "nobody looked".
        if (sc.proj_fallbacks > 0) {
            std::printf("PROJSHAPE-FALLBACK launches=%lld asked=%s offered=%s warp=%d note=these"
                        " launches took the shipped kernel, so this run is not one clean A/B side\n",
                        (long long)sc.proj_fallbacks, engine::kernels::proj_shape_name(o.proj_shape),
                        engine::kernels::proj_shape_name(engine::kernels::ProjShape::kShipped),
                        engine::kernels::proj_warp_width());
        }

        // The measurement reports below mutate `st` (every pass resets and replays), so two `RESULT` fields
        // are snapshotted here rather than read off the state afterwards: `prefix_len` is what the RUN's
        // tokens left, and a timing pass must not be able to move a field of the result line. `--time-stages`
        // happens to land its last pass back on the run's own end state (it replays every step the run took),
        // but that was an accident of that instrument's shape, and `--time-waves` does not: a wave pass ends
        // where the prompt ends. Same rule as `ms_total` above — a by-product of the run, never of the probe.
        const int result_prefix_len = st.prefix_len;

        // The wave clock's two passes. Placed before the step clock's because a wave is what the run did
        // first, and after `LAUNCHSTAT` for the same load-bearing reason: the wave report prices ITS pass
        // by delta, and `LAUNCHSTAT` must keep meaning the run (§2.4's section-(h) reading of it).
        if (o.time_waves) {
            wave_report(o, store, st, sc, wave);
        }
        // The stage clock's three passes. Their position is load-bearing twice over: they `reset()` and
        // re-prefill `st`, so they run only after everything the RUN reports about its own state
        // (`--state-fp` above) has been taken and after `LAUNCHSTAT` has been printed — otherwise the
        // launch counters would report four passes instead of the run's one, and section (h)'s reading
        // of them would stop meaning the run it is judging. And they write nothing to `--out`, so a
        // `--time-stages` run and a plain run produce identical dumps (proved in `engine/results.md`).
        // Their lines sit between `LAUNCHSTAT` and `RESULT`, which keeps the last line of a run the
        // result line.
        if (o.time_stages) {
            time_stages_report(o, store, st, sc, step_records, mtp.get());
        }

        // The accept loop's own accounting, printed between `LAUNCHSTAT` and `RESULT` so the last line of
        // a run stays the result line. `describe()` refuses a tally whose counters disagree, so a rate on
        // this line is an arithmetically consistent rate or nothing (`engine/mtp_accept.h`).
        if (aside) {
            if (pending_candidate >= 0) mtp->unadjudicated++;   // proposed, never settled: no next round
            std::printf("MTPSTAT route=%s gemm=%s gdn_scan=%s attn_fattn_wide=%s %s\n",
                        engine::mtpverify::route_name(o.mtp_route), accept_gemm_ran, accept_scan_ran,
                        accept_attn_wide_ran,
                        engine::mtpaccept::describe(astats).c_str());
            std::fflush(stdout);
            engine::mtpaccept::free_side(*aside);
        }
        if (emit_log) std::fclose(emit_log);

        // ---- `DECWALL`: the run's own decode window, and the only rate this binary can honestly give ---
        // Why this line exists: an accept run's wall time cannot be recovered by subtracting a `--steps 0`
        // run from a `--steps N` run. The loop can end early (§07.5's stop, `RUN-END`, `ROUND-SKIP`), so the
        // two windows do not cover the same work, and the first harness that did that arithmetic printed
        // 20,689 tok/s for a stream of 45 tokens. The honest object is ONE window around the generation
        // loop, divided by the tokens the loop itself emitted — which is what was closed above.
        //
        // In the window: every body the loop ran, plus the driver's own host work per body (its `STEP`/
        // `ACCEMIT` lines and their fflush) — which is what a server would pay anyway, and is the same
        // order of host work on both arms of an A/B. Out of it: the prefill wave (printed separately as
        // `prefill_ms`, so the split is shown rather than subtracted), the `--state-fp` prompt-end snapshot,
        // and every measurement report below this line: `--time-stages` replays the whole run three more
        // times and none of that may enter a wall that claims to be the run's.
        //
        // Like every other time here this is a by-product of the run, never a measurement (§6): no warm-up,
        // no best-of-N. It is printed on every run with a generation phase and not behind a flag, because a
        // wall an author can switch off is a wall that gets switched off. `census=` is the auditable part:
        // `emitted` is counted at the emit site while `MTPSTAT`'s comes from the device verdicts, so the two
        // agreeing is a cross-check between independent sources rather than an echo. A zero window, a failed
        // sync, a disagreement, or an empty denominator prints NO-RATE instead of a number.
        {
            const double billed_ms = step_ms - prefill_ms;   // the nested per-step timers, if any ran
            std::string bad;
            if (o.steps <= 0) {
                bad = "no_generation_bodies";
            } else if (!decode_sync_err.empty()) {
                bad = "sync_failed:" + decode_sync_err;
            } else if (gen_bodies <= 0) {
                bad = "no_bodies_ran";
            } else if (gen_emitted <= 0) {
                bad = "nothing_emitted";
            } else if (gen_bodies != gen_plain + (aside ? astats.rounds : 0)) {
                bad = "bodies_disagree";
            } else if (!aside && gen_emitted != (long long)step_records.size()) {
                bad = "emitted_vs_step_records";
            } else if (aside && gen_emitted != astats.emitted + gen_plain - gen_suppressed) {
                bad = "emitted_vs_verdict_tally";
            } else if (decode_ms <= 0.0) {
                bad = "zero_window";   // `hipEventElapsedTime`'s 0.000 ms trap, host-side twin
            } else if (decode_ms < billed_ms) {
                bad = "window_smaller_than_billed";
            }
            std::printf("DECWALL mode=%s prefill_ms=%.3f decode_ms=%.3f bodies=%lld rounds=%lld plain=%lld"
                        " accepted=%lld rejected=%lld suppressed=%lld emitted=%lld stop=%d ended=%s",
                        o.feed.empty() ? "run" : "replay", prefill_ms, decode_ms, gen_bodies,
                        aside ? astats.rounds : 0LL, gen_plain, aside ? astats.accepted : 0LL,
                        aside ? astats.rejected : 0LL, gen_suppressed, gen_emitted,
                        run_stopped ? 1 : 0, gen_end);
            if (bad.empty()) {
                std::printf(" ms_per_token=%.4f tok_per_s=%.3f census=ok\n",
                            decode_ms / (double)gen_emitted, 1000.0 * (double)gen_emitted / decode_ms);
            } else {
                std::printf(" note=%s NO-RATE\n", bad.c_str());
            }
            std::fflush(stdout);
        }

        // ---- the meter's report, BEFORE `RESULT`, which stays the run's last stdout line ----------
        // One line per phase in source order (the registry's order, not the accumulated map's), each
        // carrying its own run count and its boundary label, then the derived `overhead` and the meter's own
        // `summary`. Nothing here is printed as a rate over a denominator the line does not show.
        if (meter) {
            // The idle-sync floor, taken after the generation loop rather than before the prompt wave: it is
            // 64 empty waits, and an instrument must not spend its calibration inside the window it calibrates.
            // Reuses `stage_clock`'s calibration rather than inventing a second constant, so both instruments
            // quote the same quantity — the cost of ONE empty `hipDeviceSynchronize()` on this run's own
            // queue. It is printed because the number is conditional: measured on a drained queue after the
            // last round it is well under a microsecond here, and the same wait taken mid-burst is what a
            // `sync_added=1` phase actually pays, which is why the phase figures are bounds and the floor is
            // quoted beside them rather than a constant carried in a header.
            meter->set_idle_sync_ms(engine::stageclock::calibrate_idle_sync(64) / 1000.0);
            meter->finalize();
            std::printf("%s\n", meter->describe().c_str());
            std::fflush(stdout);
        }

        // The meter rides the line it summarizes. A SECOND `RESULT` line would make every
        // `grep -m1 '^RESULT'` harness in the tree read the wrong one — `engine/gate.py` keeps the LAST
        // `RESULT` it sees, so a summary line after the run line would be the one it parsed. Hence fields
        // on the run's own line, which also keeps `RESULT` the last line of stdout by construction.
        const std::string meter_fields = meter ? meter->result_fields() : std::string();
        std::printf("RESULT mode=%s steps=%d ms_total=%.3f ms_per_step=%.3f last_sampled=%d"
                    " prefix_len=%d stopped=%d%s\n",
                    o.feed.empty() ? "run" : "replay", total_steps, step_ms,
                    step_ms / (n > 0 ? n : 1), sampled,
                    result_prefix_len, run_stopped ? 1 : 0, meter_fields.c_str());
        std::fflush(stdout);
        // Disarmed before the destructor for the same reason `stage_clock` is disarmed after its report:
        // anything printed from here on (nothing is, today) must not be able to charge a phase.
        engine::mtpmeter::arm(nullptr);
        if (mtp) engine::mtpdraft::free_state(mtp->st);
        engine::kernels::gemm_route_shutdown();
        engine::free_scratch(sc);
        engine::free_state(st);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decode_run FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
