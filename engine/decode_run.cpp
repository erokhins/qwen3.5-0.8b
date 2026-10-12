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
#include "engine/loader.h"
#include "engine/prefill.h"
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
    bool state_fp = false;           // emit the §08.6 state fingerprints (see the `--state-fp` block)
    // Which logits a prompt wave gives back (§06.5 / `engine/prefill.h`). `last` is the spec's shape
    // and the default; `all` is the gate's byte-identity mode, which asks for the rows §06.5 throws
    // away so they can be `cmp`'d against what the per-token prompt loop used to write.
    engine::PrefillLogits prefill_logits = engine::PrefillLogits::kLast;
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
                 " [--prefill-logits last|all] [--load-verify] [--sizes-only]\n"
                 "  --replay  feed each generation step the caller's token (teacher forcing; also"
                 " leaves §07.5 unarmed)\n"
                 "  --no-stop leave §07.5 unarmed in a free run too (depth smoke only)\n"
                 "  --state-fp D2H the §08 state at the end of the prompt and at the end of the run"
                 " and print one STATEFP row per pinned fingerprint row (judged by engine/gate.py"
                 " section (g))\n"
                 "  --prefill-logits  `last` (default) samples only the prompt wave's last row, which"
                 " is all §06.5 uses; `all` dumps every row's logits, which is what the wave's"
                 " byte-identity gate compares against\n"
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
        std::printf("PREFILL tokens=%zu logits=%s prefix_len=%d", wave.size(),
                    engine::prefill_logits_name(wave.logits_mode), st.prefix_len);
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
            step_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  ts).count();
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
        std::printf("RESULT mode=%s steps=%d ms_total=%.3f ms_per_step=%.3f last_sampled=%d"
                    " prefix_len=%d stopped=%d\n",
                    o.feed.empty() ? "run" : "replay", total_steps, step_ms,
                    step_ms / (n > 0 ? n : 1), sampled,
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
