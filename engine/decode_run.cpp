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
//   run      `--ids … --steps N`      free generation: the engine feeds its own samples.
//   replay   `--ids … --replay …`     each generation step is fed the CALLER's token, so a
//                                     divergence shows up at one step instead of cascading
//                                     (that is what `gate.py` uses for the 64-step forced pin;
//                                     §07.5 does not end a replay — the caller owns the stream).
//   session  `--session`              resident line protocol for `engine/chat.sh`: weights and
//                                     `RequestState` stay alive across turns. See `session_loop`.
//
// Every number it prints about *time* is provisional by construction: see `bench/performance-status.md`
// §6 for what a clean window takes to arrange, and `AGENTS.md` for why a worker must not arrange one.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/decode_step.h"
#include "engine/loader.h"
#include "engine/state.h"

namespace {

// Slot count a `--session` allocates when `--cap` is not given. Not a limit the engine enforces:
// the attention stage refuses above its own lane ceiling long before 4096 (`kernels_attn.cpp`'s
// MAXSC, written up in `engine/results.md`), and 48 MiB of KV is nothing on this box.
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
        } else if (a == "--help" || a == "-h") {
            o.help = true;
        } else {
            throw std::runtime_error("unknown argument " + a);
        }
    }
    if (o.help) return o;   // --help answers before it complains about what is missing
    if (!o.sizes_only && !o.session && o.prompt.empty()) {
        throw std::runtime_error(
            "decode_run needs --ids <comma-separated token ids> (the fixture's prefill_ids); "
            "--sizes-only and --session are the two modes that run without them");
    }
    if (!o.sizes_only && !o.session && o.steps < 0) {
        throw std::runtime_error("decode_run needs --steps >= 0");
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
//   feed <id> <id>…   advance the state over the caller's tokens (a new turn's prompt), then
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
                int n = 0;
                // The pending sample first, if there is one: it is the last token the caller was
                // shown and the state has not consumed it (see the invariant above). `decode_step`
                // leaves `sampled_token` = whatever its own step argmaxed to, so feeding each
                // token in turn leaves the state's next sample pending at the end — which is what
                // `gen` emits first.
                if (st.sampled_token >= 0) {
                    decode_step(store, st, sc, st.sampled_token, nullptr);
                    ++n;
                }
                long id = 0;
                while (words >> id) {
                    decode_step(store, st, sc, (int)id, nullptr);
                    ++n;
                }
                std::printf("FED n=%d prefix_len=%d\n", n, st.prefix_len);
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
                 " [--replay ID,ID,...] [--cap N] [--session] [--no-stop]"
                 " [--load-verify] [--sizes-only]\n"
                 "  --replay  feed each generation step the caller's token (teacher forcing; also"
                 " leaves §07.5 unarmed)\n"
                 "  --no-stop leave §07.5 unarmed in a free run too (depth smoke only)\n"
                 "  --session resident feed/gen/reset/stats/quit protocol for engine/chat.sh\n");
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
        if (o.sizes_only) {
            std::printf("RESULT mode=sizes-only tensors=%zu check_weights=PASS\n", stats.n_loaded);
            std::fflush(stdout);
            return 0;
        }

        engine::RequestState st = engine::allocate_state(cap);
        engine::DecodeScratch sc = engine::allocate_scratch();
        engine::prepare_weight_adapters(store, sc);

        const std::string inv = st.check_invariants();
        if (!inv.empty()) {
            std::fprintf(stderr, "decode_run: fresh state violates §08.6: %s\n", inv.c_str());
            return 1;
        }

        if (o.session) {
            session_loop(store, st, sc);
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

        // Steps 0 .. prompt_len-1 feed the prompt; the rest feed what the previous step produced.
        // They are the same call: the README's hand-off note is that prefill is token-at-a-time
        // through `decode_step` with no chunked variant, so a second code path here would be a second
        // implementation to keep honest. `prefill.{h,cpp}` remains a throughput item, not this one.
        std::vector<float> logits;
        int fed = 0;
        int sampled = -1;
        bool run_stopped = false;   // §07.5 reached in the GENERATION phase; see the rule below
        double step_ms = 0.0;
        const int total_steps = (int)o.prompt.size() + o.steps;
        for (int i = 0; i < total_steps; ++i) {
            const bool from_prompt = i < (int)o.prompt.size();
            // `--replay`: the caller owns the generation-phase tokens, exactly as it owns the
            // prompt's. The engine still samples every step and prints that sample, so a step
            // whose fed token and sampled token disagree is precisely the divergence the gate is
            // looking for — and because the feeding is not self-propagating, that divergence
            // cannot cascade into the steps after it. Without `--replay` the engine feeds itself,
            // which is what a free run does and what A-18 says cannot be pinned past ~8 tokens.
            const int token = from_prompt
                                  ? o.prompt[i]
                                  : (o.feed.empty() ? sampled
                                                    : o.feed[i - (int)o.prompt.size()]);
            const int pos_before = st.prefix_len;
            const auto ts = std::chrono::steady_clock::now();
            const engine::StepResult r =
                engine::decode_step(store, st, sc, token,
                                    o.out_dir.empty() ? nullptr : &logits);
            step_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  ts).count();
            sampled = r.token;

            // ---- §07.5's stop check is armed in the GENERATION phase only. ---------------
            // While the prompt is being fed, the sample is discarded: the next token fed comes from
            // the caller's id list, not from the model, so what the engine argmaxed to mid-prompt
            // says nothing about whether the model wants to stop and cannot be a stop condition.
            // It is also not hypothetical. The `dec8` prompt is the manifest's `sources[0].ids` — a
            // rendered chat template — so it legitimately contains id 248046 (the end-of-turn token
            // `kStopToken` is) at index 8, and 248068/248069 later; and on the first device run the
            // engine's own argmax at prompt index 6 WAS 248046. Arming the check there ended a
            // 17-token prompt after 7 steps, produced 0 generation steps, and failed six gate
            // checks that all said something else. So: a prompt-phase match is printed and ignored;
            // only a generation-phase match ends the run. Do not "restore" the prompt-phase check.
            const bool stop_match = r.token == engine::kStopToken;
            // §07.5 ends a FREE run. A replay is not free: the caller feeds the stream, and
            // dec64f's own forced chain steps PAST the stop id (the reference samples it at
            // generation step 24 and teacher forcing keeps going), so firing here would truncate
            // the exact depth the fixture exists to test. `stop_match` is still printed and the
            // gate still sees `stop=0` with `sampled=248046`, which is the same convention the
            // prompt phase uses above.
            //
            // `--no-stop` is the same waiver for a free run, and exists for one reason: a smoke run
            // that is supposed to reach a given depth. On dec8's prompt the model emits the stop id
            // after 38 tokens, so `--smoke 300` would otherwise test 38 steps of state growth and
            // call it 300. Nothing that claims parity uses this flag; the gate's parity path never
            // sets it.
            const bool stop_fired = stop_match && !from_prompt && o.feed.empty() && !o.no_stop;
            if (stop_fired) run_stopped = true;

            std::printf("%s %d | fed %7d | sampled %7d | top %.6f | P %d -> %d%s%s\n",
                        from_prompt ? "prompt" : "decode", i, token, r.token, r.top_logit,
                        pos_before, st.prefix_len, stop_fired ? " [STOP]" : "",
                        // The prompt-phase match stays visible: it is a diagnostic of the prompt
                        // path (and a warning that the fixture's prompt contains the stop id).
                        (stop_match && from_prompt) ? " [stop id in prompt phase — ignored]" : "");
            // The machine-readable twin of the line above, with the step split into a kind so the
            // gate does not have to re-derive which index the reference's `decode{s}` starts at.
            // `stop=` is the STOP CHECK (armed, matched, run over) — not the raw match; a prompt-phase
            // match is `stop=0` with `sampled=248046`, which `gate.py` reads as a fact, not a stop.
            std::printf("STEP kind=%s index=%d input=%d sampled=%d top_logit=%.9g pos_before=%d"
                        " pos_after=%d stop=%d\n",
                        from_prompt ? "prompt" : "decode", i, token, r.token, r.top_logit,
                        pos_before, st.prefix_len, stop_fired ? 1 : 0);
            if (log) {
                std::fprintf(log, "%d %d %d %.9g %d\n", i, token, r.token, r.top_logit, pos_before);
                write_raw(o.out_dir, "logits" + std::to_string(i) + ".bin", logits);
            }
            if (stop_fired && i + 1 < total_steps) {
                std::printf("stopped at step %d (§07.5: id %d)\n", i, engine::kStopToken);
                break;
            }
            fed++;
        }
        if (log) std::fclose(log);

        const int n = fed > 0 ? fed : 1;
        // `stopped=` here is the RUN's §07.5 decision (generation phase only), which is what the gate
        // asserts on. `st.stopped` is decode_step's raw record — "some step argmaxed to the stop id,
        // in either phase" — so it can be true while this says 0, and on `dec8` it is.
        std::printf("RESULT mode=%s steps=%d ms_total=%.3f ms_per_step=%.3f last_sampled=%d"
                    " prefix_len=%d stopped=%d\n",
                    o.feed.empty() ? "run" : "replay", total_steps, step_ms, step_ms / n, sampled,
                    st.prefix_len, run_stopped ? 1 : 0);
        std::fflush(stdout);
        engine::free_scratch(sc);
        engine::free_state(st);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decode_run FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
