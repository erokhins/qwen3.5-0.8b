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
// Every number it prints about *time* is provisional by construction: see `bench/performance-status.md`
// §6 for what a clean window takes to arrange, and `AGENTS.md` for why a worker must not arrange one.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/decode_step.h"
#include "engine/loader.h"
#include "engine/state.h"

namespace {

struct Options {
    std::string model_dir = "models/Qwen3.5-0.8B";
    std::string out_dir;             // where logits/sampled dumps go; empty means do not write
    std::vector<int> prompt;         // the tokens fed before generation starts (the fixture prefill)
    int steps = 8;                   // decode steps after the prompt
    int64_t cap = 0;                 // 0 = derive from prompt+steps
    bool load_verify = false;        // the loader's SHA-256 pass; not this gate's concern
    bool sizes_only = false;         // check §00/§01 against the checkpoint and exit
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
        } else {
            throw std::runtime_error("unknown argument " + a);
        }
    }
    if (!o.sizes_only && o.prompt.empty()) {
        throw std::runtime_error(
            "decode_run needs --ids <comma-separated token ids> (the fixture's prefill_ids); "
            "--sizes-only is the only mode that runs without them");
    }
    if (!o.sizes_only && o.steps < 0) {
        throw std::runtime_error("decode_run needs --steps >= 0");
    }
    return o;
}

void write_raw(const std::string& dir, const std::string& name, const std::vector<float>& v) {
    std::ofstream f(dir + "/" + name, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + dir + "/" + name);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decode_run: %s\n", e.what());
        std::fprintf(stderr,
                     "usage: decode_run [--model DIR] [--ids ID,ID,...] [--steps N] [--out DIR]"
                     " [--cap N] [--load-verify] [--sizes-only]\n");
        return 2;
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

        const int64_t cap = o.cap > 0 ? o.cap
                                    : (int64_t)(o.prompt.size() + (o.steps > 0 ? o.steps : 1) + 1);
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

        std::FILE* log = nullptr;
        if (!o.out_dir.empty()) {
            log = std::fopen((o.out_dir + "/sampled.txt").c_str(), "w");
            if (!log) throw std::runtime_error("cannot open " + o.out_dir + "/sampled.txt");
            std::fprintf(log, "# step input_token sampled_token top_logit prefix_len_before\n");
        }

        // step -1 … P-1: the prompt (what a chunked `prefill.{h,cpp}` will eventually do in one
        // pass), then the generation steps. They are the same call: README's hand-off note is that
        // prefill is token-at-a-time through `decode_step` with no chunked variant, so a separate
        // code path here would be a second implementation to keep honest.
        std::vector<float> logits;
        int fed = 0;
        int sampled = -1;
        double step_ms = 0.0;
        const int total_steps = (int)o.prompt.size() + o.steps;
        for (int i = 0; i < total_steps; ++i) {
            const bool from_prompt = i < (int)o.prompt.size();
            const int token = from_prompt ? o.prompt[i] : sampled;
            const int pos_before = st.prefix_len;
            const auto ts = std::chrono::steady_clock::now();
            const engine::StepResult r =
                engine::decode_step(store, st, sc, token,
                                    o.out_dir.empty() ? nullptr : &logits);
            step_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  ts).count();
            sampled = r.token;
            std::printf("%s %d | fed %7d | sampled %7d | top %.6f | P %d -> %d%s\n",
                        from_prompt ? "prompt" : "decode", i, token, r.token, r.top_logit,
                        pos_before, st.prefix_len, r.token == engine::kStopToken ? " [STOP]" : "");
            // The machine-readable twin of the line above, with the step split into a kind so the
            // gate does not have to re-derive which index the reference's `decode{s}` starts at.
            std::printf("STEP kind=%s index=%d input=%d sampled=%d top_logit=%.9g pos_before=%d"
                        " pos_after=%d\n",
                        from_prompt ? "prompt" : "decode", i, token, r.token, r.top_logit,
                        pos_before, st.prefix_len);
            if (log) {
                std::fprintf(log, "%d %d %d %.9g %d\n", i, token, r.token, r.top_logit, pos_before);
                write_raw(o.out_dir, "logits" + std::to_string(i) + ".bin", logits);
            }
            if (st.stopped && i + 1 < total_steps) {
                std::printf("stopped at step %d (§07.5: %d)\n", i, engine::kStopToken);
                break;
            }
            fed++;
        }
        if (log) std::fclose(log);

        const int n = fed > 0 ? fed : 1;
        std::printf("RESULT mode=run steps=%d ms_total=%.3f ms_per_step=%.3f last_sampled=%d"
                    " prefix_len=%d stopped=%d\n",
                    total_steps, step_ms, step_ms / n, sampled, st.prefix_len, st.stopped ? 1 : 0);
        std::fflush(stdout);
        engine::free_scratch(sc);
        engine::free_state(st);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decode_run FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
