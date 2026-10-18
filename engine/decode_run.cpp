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
// Every number it prints about *time* is provisional by construction: see `bench/performance-status.md`
// §6 for what a clean window takes to arrange, and `AGENTS.md` for why a worker must not arrange one.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "engine/prefill.h"
#include "engine/stage_clock.h"
#include "engine/state.h"

#include <memory>

namespace {

// Slot count a `--session` allocates when `--cap` is not given. Not a limit the engine enforces, and
// after the 2.5 ceiling raise it is no longer the first bound either: attention now serves 512 x 9 =
// 4608 lanes, so a session hits this cap first and `--cap` is how it reaches the wider geometry.
constexpr int64_t kSessionDefaultCap = 4096;

struct Options {
    std::string model_dir = "models/Qwen3.5-0.8B";
    std::string out_dir;             // where logits/sampled dumps go; empty means do not write
    std::vector<int> prompt;         // the tokens fed before generation starts (the fixture prefill)
    std::vector<int> feed;           // --replay: what to feed each GENERATION step, from outside
    int steps = 8;                   // decode steps after the prompt
    int64_t cap = 0;                 // 0 = derive from prompt+steps
    bool load_verify = false;        // the loader's SHA-256 pass; not this gate's concern
    bool sizes_only = false;         // check §00/§01 against the checkpoint and exit
    bool session = false;            // resident line protocol for engine/chat.sh
    bool no_stop = false;            // §07.5 stays unarmed in generation too (deep smoke runs only)
    bool state_fp = false;           // emit the §08.6 state fingerprints (see the `--state-fp` block)
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
    // The chunk size. Only 64 is accepted, and the refusal prints the measured reason rather than
    // "unsupported": C=16 lost §2.3's per-row budget on the pinned dumps, which is the argument for
    // why 64 is the default at all.
    int gdn_chunk = engine::kernels::kGdnChunkTokens;
    bool gdn_chunk_given = false;
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
        } else if (a == "--prefill-logits") {
            o.prefill_logits = engine::parse_prefill_logits(next());
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
    if (!o.sizes_only && !o.session && o.prompt.empty()) {
        throw std::runtime_error(
            "decode_run needs --ids <comma-separated token ids> (the fixture's prefill_ids); "
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
void session_loop(const engine::WeightStore& store, engine::RequestState& st,
                  engine::DecodeScratch& sc) {
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
            std::printf("BYE\n");
            std::fflush(stdout);
            return;
        }
        if (cmd == "reset") {
            st.reset();
            st.sampled_token = -1;
            st.stopped = false;
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
                while (emitted < budget) {
                    const int pos = st.prefix_len - 1;
                    std::printf("GEN %d %d\n", cur, pos);
                    std::fflush(stdout);
                    ++emitted;
                    if (cur == engine::kStopToken) {   // §07.5, generation phase
                        stop = true;
                        break;
                    }
                    const engine::StepResult r = decode_step(store, st, sc, cur, nullptr);
                    cur = r.token;
                }
                st.sampled_token = stop ? cur : -1;
                // Stop path: `cur` was emitted but never stepped, so it stays pending and the next
                // `feed` steps it — the transcript in the state ends up complete, and the driver
                // does not have to re-send the stop token.
                // Budget path: the last emitted token WAS stepped, so `cur` is a sample nobody
                // was ever shown. Dropping it is the difference between a state that matches the
                // caller's transcript and one holding a token the user never saw; the driver
                // closes the assistant turn instead (`last_stop != "eos"` in engine/chat.py).
                st.stopped = stop;
                std::printf("END steps=%d stop=%d prefix_len=%d\n", emitted, stop ? 1 : 0,
                            st.prefix_len);
            } else if (cmd == "stats") {
                std::printf("PREFIX p=%d sampled=%d stopped=%d\n", st.prefix_len,
                            st.sampled_token, st.stopped ? 1 : 0);
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
                 "usage: decode_run [--model DIR] [--ids ID,ID,...] [--steps N] [--out DIR]"
                 " [--replay ID,ID,...] [--cap N] [--session] [--no-stop] [--state-fp]"
                 " [--prefill-logits last|all] [--wave-rows N]"
                 " [--mlp-map auto|interleave|contig] [--gemm library|legacy|lt|hand] [--attn-tile 0|1]"
                 " [--gemm-pins all|none|<csv>] [--gemm-heur N] [--attn-touch 0|1]"
                 " [--gdn-spell ship|unroll4|fused]"
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
                 " prototype/gdn_rec_isa/results.md §11, bench/performance-status.md §19). It is"
                 " BIT-identical too (ndiff 0 on `o` and on the state, 4/4 cases) and it is a KNOB,"
                 " NOT a default: the paired in-situ A/B on the `gdn_rec` family row decides whether"
                 " the default moves. Inert under `--gdn-scan collapse`, which launches a different"
                 " kernel: a bare `collapse` run is legal and echoes `gdn_spell=inert-under-collapse`,"
                 " while naming this flag alongside it is REFUSED rather than echoed with a spelling"
                 " that did not run. All spellings are compiled in either way, so a paired wave timing"
                 " is one window\n"
                 "  --gdn-loop N  how many token-steps ONE `gdn_rec` launch carries (D53/D54, "
                 "engine/gdn_rec_loop_kernels.h). `1`, the default, is the shipped one-launch-per-token"
                 " form: 18 launches per token, 73,728 for a 4,096-token prompt, at the 1.72 us"
                 " measured submission floor (§11). `512` walks a whole wave body inside one launch —"
                 " 18 x bodies, i.e. 144 launches for the same prompt — with the SAME per-token"
                 " arithmetic (the kernel is `rec_fused_kernel`'s body with the token index moved"
                 " inside it, transcribed and checked by check_recloop_transcription.py in"
                 " prototype/gdn_rec_isa), the same grid `<<<16,128>>>` and the same state"
                 " addresses, so it is a launch-structure change and not a numerics one: spans"
                 " 0/1/3/16 came back `ndiff_o_vs0=0 ndiff_state_vs0=0` (bench/performance-status.md"
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
PassResult stage_pass(const Options& o, const engine::WeightStore& store, engine::RequestState& st,
                      engine::DecodeScratch& sc, const std::vector<StepRec>& ref,
                      engine::stageclock::Mode mode) {
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
    engine::stageclock::arm(clock.get());
    for (const StepRec& want : ref) {
        if (clock) clock->begin_step();
        const auto t0 = std::chrono::steady_clock::now();
        // logits=nullptr: the sample and its logit come back in `StepResult`; the vocabulary itself is
        // still downloaded and argmaxed by `finish_token`, which is the point — a pass that skipped the
        // D2H would be measuring a step the production path never submits.
        const engine::StepResult r = engine::decode_step(store, st, sc, want.fed, nullptr);
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
        // Which GEMM a wide body ran, so the plan bills the cast row the way the body submitted it. Asked
        // at `kMaxWaveRows` rather than at a body width on purpose: the only width
        // `resolve_wave_gemm` overrides is `rows <= 1`, and `wave_plan_bill` derives that one-row body
        // from `tokens`/`bodies`/`wave_rows` itself, so what the plan needs from the driver is the route
        // the run asked for and nothing else. Same question the `GEMM` banner asks above.
        shape.gemm_library =
            engine::kernels::wave_gemm_is_library(o.wave_gemm, engine::kMaxWaveRows);
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
                        engine::DecodeScratch& sc, const std::vector<StepRec>& base) {
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
        stage_pass(o, store, st, sc, base, engine::stageclock::Mode::kBurst);
    const PassResult stage =
        stage_pass(o, store, st, sc, base, engine::stageclock::Mode::kStage);
    const PassResult kernel =
        stage_pass(o, store, st, sc, base, engine::stageclock::Mode::kKernel);

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
        // D-11's provenance half. The wave-row ceiling and the hipBLAS version are the two numbers that
        // decide what a pinned corpus means once more than one GEMM can produce it: `kMaxWaveRows`
        // because the row-count rule makes it the last width that is still the hand GEMV, hipBLAS's
        // version because §A1 found its solution per (m,n,k) is the numerics and it can change without
        // this repository changing. Printed in every mode, like LAUNCH — a statement about the code and
        // the library it is linked against, not about the run — and read by `gate.py`'s
        // `numerics_provenance_checks` against `engine/numerics_provenance.json`.
        std::printf("NUMERICS wave_rows_cap=%d wave_rows=%d gemm=%s hipblas=%s attn_tile=%d"
                    " attn_touch=%d attn_fattn=%d gdn_scan=%s gdn_chunk=%d gdn_spell=%s gdn_loop=%d\n",
                    engine::kMaxWaveRows, o.wave_rows,
                    engine::kernels::wave_gemm_name(o.wave_gemm),
                    engine::kernels::gemm_route_version().c_str(), o.attn_tile,
                    o.attn_touch, o.attn_fattn,
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
                    o.gdn_loop);
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
        sc.gdn_scan = o.gdn_scan;
        sc.gdn_chunk = o.gdn_chunk;
        sc.gdn_spell = o.gdn_spell;
        sc.gdn_loop = o.gdn_loop;
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

        const std::string inv = st.check_invariants();
        if (!inv.empty()) {
            std::fprintf(stderr, "decode_run: fresh state violates §08.6: %s\n", inv.c_str());
            return 1;
        }

        if (o.session) {
            session_loop(store, st, sc);
            engine::kernels::gemm_route_shutdown();
            engine::free_scratch(sc);
            engine::free_state(st);
            return 0;
        }

        std::FILE* log = nullptr;
        if (!o.out_dir.empty()) {
            log = std::fopen((o.out_dir + "/sampled.txt").c_str(), "w");
            if (!log) throw std::runtime_error("cannot open " + o.out_dir + "/sampled.txt");
            std::fprintf(log, "# step input_token sampled_token top_logit prefix_len_before\n");
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
        std::vector<float> logits;
        double step_ms = 0.0;   // the run's step time, by-product not measurement (§6)
        const auto t_wave = std::chrono::steady_clock::now();
        const engine::PrefillResult wave =
            engine::prefill_wave(store, st, sc, o.prompt, o.prefill_logits);
        // Added to the same accumulator the per-step timings went into, so RESULT's `ms_total` still
        // means "this run" rather than silently becoming generation-only.
        step_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wave)
                .count();
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

        int fed = 0;
        // §06.5: the first token fed here is the wave's own `y0`, produced inside the last wave row,
        // so no extra decode step is spent on the first sampled token.
        int sampled = wave.empty() ? -1 : wave.rows.back().sampled;
        bool run_stopped = false;   // §07.5 reached in the GENERATION phase; see the rule below
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
            const auto ts = std::chrono::steady_clock::now();
            const engine::StepResult r =
                engine::decode_step(store, st, sc, token,
                                    o.out_dir.empty() ? nullptr : &logits);
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - ts).count();
            step_ms += wall_ms;
            step_records.push_back({step, token, r.token, r.top_logit});
            sampled = r.token;

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
                break;
            }
            fed++;
        }
        if (log) std::fclose(log);

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
            time_stages_report(o, store, st, sc, step_records);
        }

        std::printf("RESULT mode=%s steps=%d ms_total=%.3f ms_per_step=%.3f last_sampled=%d"
                    " prefix_len=%d stopped=%d\n",
                    o.feed.empty() ? "run" : "replay", total_steps, step_ms,
                    step_ms / (n > 0 ? n : 1), sampled,
                    result_prefix_len, run_stopped ? 1 : 0);
        std::fflush(stdout);
        engine::kernels::gemm_route_shutdown();
        engine::free_scratch(sc);
        engine::free_state(st);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decode_run FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
