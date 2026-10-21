#!/usr/bin/env python3
"""A7 instrument: named host-phase timers + wave repeats, for §49's per-prompt constant.

Why a patcher and not a .patch. The insertions are anchored on real source lines; a unified
diff would fail to apply the moment any of the surrounding comments is re-flowed, and a
silent no-op patch is the empty-parse failure mode `AGENTS.md` records three times. Every
anchor here is asserted to occur EXACTLY ONCE and the script refuses to write anything unless
all of them matched, so a patched tree is either complete or does not exist.

Everything added is gated on an environment variable, so a patched binary with no `A7_*` set
prints exactly what the shipped binary prints. Still: never use a patched binary for a gate
or a pin — these are host wall-clocks, and §6 makes that a by-product, not a measurement.

What it adds
  A7PHASE name=X ms=Y            one per host phase outside the wave, **gated on `A7_PHASE=1`**
                                 (check_weights, state alloc+reset, scratch alloc, gemm route
                                 init, gemm route plan, prepare_weight_adapters, prompt-row
                                 prints, teardown). `load` is NOT duplicated: the shipped
                                 binary already prints it as `loaded … in X ms`.
  A7REPEAT kind=warmup|repeat    another full `prefill_wave` of the same prompt, timed.
        A7_WARMUP=N              N DISCARDED waves before the timed one ⇒ `RESULT ms_total`
                                 becomes the warm number (this is what llama-bench does by
                                 default: `llama-bench.cpp:2415-2428`).
        A7_REPEAT=N              N timed extra waves AFTER the run, printed between
                                 `LAUNCHSTAT` and `RESULT` so `RESULT` keeps meaning the run.
  A7BODY wave=K submit_ms=…      host-submission wall of each body, no sync added
                                 (`A7_BODY=1`). If these stay small while the wave is slow,
                                 the cost is on the device side, not in host bookkeeping.
  A7SYNC1 wave=0 device_ms=…     wall of (first body + one stream sync) (`A7_SYNC1=1`). Cold
                                 vs warm here splits "first wave is slower on the device"
                                 (pages, first launch of each kernel, DVFS) from "the host
                                 stalls before submitting". One line per wave execution, so a
                                 run with `--time-waves` prints one per pass.

Usage — always in a scratch worktree, never in `engine/` in place:

    WT=$(mktemp -d)/a7wt && git worktree add "$WT" HEAD --detach
    python3 prototype/perprompt/a7_phase.py --check            # anchors only, writes nothing
    python3 prototype/perprompt/a7_phase.py --apply "$WT"
    (cd "$WT" && bash engine/build.sh)
    bash prototype/perprompt/a7_host_split.sh --worktree "$WT"

`--check` on the untouched tree must print `anchors: 15/15`; that is the assertion that this
instrument still describes the code it was written against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (file, anchor, replacement). Anchor text is matched exactly and must be unique.
DR = "engine/decode_run.cpp"
PF = "engine/prefill.cpp"

A7_HELPERS = r"""
// ---- A7 (prototype/perprompt/a7_phase.py): named host-phase timers for the §49 constant.
// Instrumentation, not behaviour: every print below is either a phase wall that the shipped
// binary does not otherwise expose, or a wave the caller asked for with A7_WARMUP/A7_REPEAT.
// With no A7_* in the environment this binary's stdout is unchanged. These are host
// wall-clocks and therefore a by-product, never a measurement (`engine/performance-status.md` §6).
namespace a7 {
inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}
inline int env_int(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : 0;
}
// A phase that produced no number is a failure, not a zero: with `A7_PHASE=1` the line is
// unconditional, so a missing line in the log means the phase did not run, and the driver
// aborts on that. Gated like everything else here, so a patched binary with no `A7_*` set
// prints byte-identical stdout to the shipped one.
inline bool on(const char* name) { return std::getenv(name) != nullptr; }
inline void phase(const char* name, std::chrono::steady_clock::time_point t0) {
    static const bool a7_phase_on = on("A7_PHASE");
    if (!a7_phase_on) return;
    std::printf("A7PHASE name=%s ms=%.3f\n", name, ms_since(t0));
    std::fflush(stdout);
}
// One extra execution of the prompt wave. `reset` before, and `reset_after` for a warmup so
// the timed wave still starts at prefix_len 0 — the same reset `wave_pass` does at :1406.
void repeat_wave(const char* kind, int idx, const engine::WeightStore& store,
                 engine::RequestState& st, engine::DecodeScratch& sc,
                 const std::vector<int>& prompt, bool reset_after) {
    st.reset();
    st.sampled_token = -1;
    st.stopped = false;
    const auto t0 = std::chrono::steady_clock::now();
    const engine::PrefillResult pass =
        engine::prefill_wave(store, st, sc, prompt, engine::PrefillLogits::kLast);
    const double wall = ms_since(t0);
    if (reset_after) {
        st.reset();
        st.sampled_token = -1;
        st.stopped = false;
    }
    std::printf("A7REPEAT kind=%s idx=%d wall_ms=%.3f tokens=%zu\n", kind, idx, wall,
                pass.size());
    std::fflush(stdout);
}
}  // namespace a7

"""

EDITS: list[tuple[str, str, str]] = [
    # 1. helper block, immediately above main()
    (
        DR,
        "int main(int argc, char** argv) {",
        A7_HELPERS + "int main(int argc, char** argv) {",
    ),
    # 2. check_weights
    (
        DR,
        "        const std::string bad = engine::check_weights(store);",
        "        const auto a7_t_cw = std::chrono::steady_clock::now();\n"
        "        const std::string bad = engine::check_weights(store);\n"
        "        a7::phase(\"check_weights\", a7_t_cw);",
    ),
    # 3. state allocation, including st.reset()'s two hipMemsets
    (
        DR,
        "        engine::RequestState st = engine::allocate_state(cap);",
        "        const auto a7_t_state = std::chrono::steady_clock::now();\n"
        "        engine::RequestState st = engine::allocate_state(cap);\n"
        "        a7::phase(\"state_alloc_and_reset\", a7_t_state);",
    ),
    # 4. scratch allocation (~45 hipMallocs; first TOUCH stays inside the wave)
    (
        DR,
        "        engine::DecodeScratch sc = engine::allocate_scratch(o.wave_rows);",
        "        const auto a7_t_scr = std::chrono::steady_clock::now();\n"
        "        engine::DecodeScratch sc = engine::allocate_scratch(o.wave_rows);\n"
        "        a7::phase(\"scratch_alloc\", a7_t_scr);",
    ),
    # 5. hipBLASLt handle + workspace
    (
        DR,
        "            const std::string route = engine::kernels::gemm_route_init(o.wave_gemm);",
        "            const auto a7_t_g0 = std::chrono::steady_clock::now();\n"
        "            const std::string route = engine::kernels::gemm_route_init(o.wave_gemm);\n"
        "            a7::phase(\"gemm_route_init\", a7_t_g0);",
    ),
    # 6. the per-shape heuristic queries
    (
        DR,
        "                const std::string plan = engine::kernels::gemm_route_plan(o.wave_rows);",
        "                const auto a7_t_g1 = std::chrono::steady_clock::now();\n"
        "                const std::string plan = engine::kernels::gemm_route_plan(o.wave_rows);\n"
        "                a7::phase(\"gemm_route_plan\", a7_t_g1);",
    ),
    # 7. the dt_bias / conv1d widenings, with their device sync
    (
        DR,
        "        engine::prepare_weight_adapters(store, sc);",
        "        const auto a7_t_ad = std::chrono::steady_clock::now();\n"
        "        engine::prepare_weight_adapters(store, sc);\n"
        "        a7::phase(\"prepare_weight_adapters\", a7_t_ad);",
    ),
    # 8. warmup waves, before the timed one (A7_WARMUP)
    (
        DR,
        "        const auto t_wave = std::chrono::steady_clock::now();",
        "        for (int a7_i = 0; a7_i < a7::env_int(\"A7_WARMUP\"); ++a7_i) {\n"
        "            a7::repeat_wave(\"warmup\", a7_i, store, st, sc, o.prompt, true);\n"
        "        }\n"
        "        const auto t_wave = std::chrono::steady_clock::now();",
    ),
    # 9. start of the row-print/dump span (at --steps 0 this is exactly the prompt rows)
    (
        DR,
        '        std::printf("PREFILL tokens=%zu logits=%s prefix_len=%d wave_rows=%d", wave.size(),',
        "        const auto a7_t_rows = std::chrono::steady_clock::now();\n"
        '        std::printf("PREFILL tokens=%zu logits=%s prefix_len=%d wave_rows=%d", wave.size(),',
    ),
    (
        DR,
        "        if (log) std::fclose(log);",
        "        if (log) std::fclose(log);\n"
        "        a7::phase(\"row_prints_and_dumps\", a7_t_rows);   // --steps 0: the prompt rows",
    ),
    # 10. timed extra waves, after LAUNCHSTAT so RESULT keeps meaning the run (A7_REPEAT)
    (
        DR,
        "        if (o.time_waves) {",
        "        for (int a7_i = 0; a7_i < a7::env_int(\"A7_REPEAT\"); ++a7_i) {\n"
        "            a7::repeat_wave(\"repeat\", a7_i, store, st, sc, o.prompt, false);\n"
        "        }\n"
        "        if (o.time_waves) {",
    ),
    # 11. teardown
    (
        DR,
        "        engine::kernels::gemm_route_shutdown();\n"
        "        engine::free_scratch(sc);\n"
        "        engine::free_state(st);\n"
        "    } catch (const std::exception& e) {",
        "        const auto a7_t_fin = std::chrono::steady_clock::now();\n"
        "        engine::kernels::gemm_route_shutdown();\n"
        "        engine::free_scratch(sc);\n"
        "        engine::free_state(st);\n"
        '        a7::phase("teardown", a7_t_fin);\n'
        "    } catch (const std::exception& e) {",
    ),
]

PREFILL_PRELUDE = r"""
// ---- A7 (prototype/perprompt/a7_phase.py): the wave's own host-submission timeline.
// A7_BODY=1 prints the wall of each body's SUBMISSION with no sync added, so a slow line
// means the host stalled inside a launch (first-time driver or library work) rather than
// the GPU taking longer. A7_SYNC1=1 additionally syncs once after body 0, which is the only
// way to price the first body's device work against a warm body's without changing the rest
// of the wave's queueing. Both are off unless named, and off means zero behaviour change.
namespace a7 {
inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}
inline bool on(const char* name) {
    return std::getenv(name) != nullptr;
}
}  // namespace a7

"""

EDITS += [
    # 12. prefill.cpp: includes + helpers
    (
        PF,
        '#include "engine/decode_step.h"',
        "#include <hip/hip_runtime.h>\n"
        "#include <chrono>\n"
        "#include <cstdio>\n"
        "#include <cstdlib>\n"
        '\n#include "engine/decode_step.h"',
    ),
    (
        PF,
        "PrefillResult prefill_wave(const WeightStore& weights, RequestState& st, DecodeScratch& sc,",
        PREFILL_PRELUDE + "\nPrefillResult prefill_wave(const WeightStore& weights, RequestState& st, DecodeScratch& sc,",
    ),
    (
        PF,
        "        submit_wave_body(weights, st, sc, tokens.data() + begin, n);",
        "        const auto a7_tb = std::chrono::steady_clock::now();\n"
        "        submit_wave_body(weights, st, sc, tokens.data() + begin, n);\n"
        "        if (a7::on(\"A7_BODY\")) {\n"
        '            std::printf("A7BODY wave=%zu rows=%d submit_ms=%.3f\\n", begin / wave_cap, n,\n'
        "                        a7::ms_since(a7_tb));\n"
        "            std::fflush(stdout);\n"
        "        }\n"
        "        if (a7::on(\"A7_SYNC1\") && begin == 0) {\n"
        "            const auto a7_ts = std::chrono::steady_clock::now();\n"
        "            (void)hipStreamSynchronize(nullptr);\n"
        '            std::printf("A7SYNC1 wave=0 device_ms=%.3f\\n", a7::ms_since(a7_ts));\n'
        "            std::fflush(stdout);\n"
        "        }",
    ),
]


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text()


def apply(root: Path, write: bool) -> int:
    texts: dict[str, str] = {}
    needed: dict[str, list[tuple[str, str]]] = {}
    for rel, anchor, repl in EDITS:
        needed.setdefault(rel, []).append((anchor, repl))

    ok, bad = 0, 0
    for rel, pairs in needed.items():
        text = _read(root, rel)
        for anchor, repl in pairs:
            n = text.count(anchor)
            if n != 1:
                print(f"FAIL {rel}: anchor occurs {n} times (need exactly 1):\n{anchor[:110]!r}")
                bad += 1
                continue
            text = text.replace(anchor, repl)
            ok += 1
        texts[rel] = text

    total = sum(len(p) for p in needed.values())
    print(f"anchors: {ok}/{total} matched"
          + (" — tree already patched?" if bad and ok < total else ""))
    if bad:
        print("NOTHING WRITTEN. The instrument no longer matches this source; re-read the "
              "anchors, do not loosen them.")
        return 1
    if write:
        for rel, text in texts.items():
            (root / rel).write_text(text)
            print(f"wrote {root / rel}")
    else:
        print("check only: nothing written")
    print(f"insertions: {len(EDITS)} (A7PHASE phases + A7REPEAT waves + A7BODY/A7SYNC1)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="assert anchors against PATH, write nothing")
    ap.add_argument("--apply", metavar="ROOT", help="patch ROOT (a scratch git worktree)")
    ap.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root used by --check (default: this file's repo)",
    )
    args = ap.parse_args()
    if bool(args.apply) == args.check:
        ap.error("exactly one of --check or --apply ROOT")
    if args.check:
        return apply(Path(args.repo), write=False)
    return apply(Path(args.apply), write=True)


if __name__ == "__main__":
    sys.exit(main())
