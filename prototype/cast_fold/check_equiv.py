#!/usr/bin/env python3
"""check_equiv.py — the cast fold's CPU-side admission, and the wiring proof behind it.

Three things have to be true before route S1 item (1)'s device pass is worth a device, and none of them
needs one:

  1. the POPULATION is the one the fold claims. `census_sites.py` derives it from
     engine/decode_step.cpp + engine/state.h (six call sites, 97 launches/token, the §1 byte rows) and
     refuses if the source moved. Re-run here so a red census cannot be walked past.
  2. the OPERAND VALUES are unchanged: `equiv_check.cpp` proves, over all 65,536 BF16 patterns and
     every tile position, that the consumer's in-register widen is bit-identical to what the deleted
     `kernels_cast.cpp` launch wrote. Compiled and run here.
  3. the WIRING is real: every one of the six call sites is behind the fold predicate, every folded
     consumer entry point is actually called, the counts the C++ check printed match the census, and the
     DECLARED LAUNCH BILL moves with the fold (section 5 — a fold that deletes 97 submissions without
     moving `planned_launch_counts()` reads `plan_match=0` on every folded run, which is a defect in the
     plan derivers and not in the kernel).
     Without this, a refactor that dropped a guard would leave a green numeric check over a fold that
     no longer happens — the empty-parse failure class AGENTS.md records four times.

What stays OUT of reach on CPU, and is therefore this file's final printed handover line: the device
codegen of `float(bf16)` inside a dot's expression tree. That is `engine/gate.py` with
`QWEN_CAST_FOLD=1` plus `ab_item1.sh`'s `sampled.txt` byte-diff, which is a device pass.

    .venv/bin/python prototype/cast_fold/check_equiv.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STEP_CPP = ROOT / "engine" / "decode_step.cpp"
STEP_H = ROOT / "engine" / "decode_step.h"
STAGE_CLOCK_H = ROOT / "engine" / "stage_clock.h"
STAGE_CLOCK_CPP = ROOT / "engine" / "stage_clock.cpp"
DECODE_RUN_CPP = ROOT / "engine" / "decode_run.cpp"
KERNELS_H = ROOT / "engine" / "kernels.h"
CENSUS = HERE / "census_sites.py"
EQUIV_CPP = HERE / "equiv_check.cpp"
# Built under tmp/, not beside the source: `prototype/<op>/` is a source directory and a binary left in
# it is the .gitignore trap AGENTS.md records (a `prototype/**/<op>` line matches the directory too).
EQUIV_BIN = ROOT / "tmp" / "cast_fold" / "equiv_check"

# The six sites, by the guard text that must now be in front of each cast call. Written as the
# substring the shipped source carries, so a reformat that changes the guard's SPELLING fails loudly
# here rather than quietly turning the fold off.
REQUIRED_GUARDS = [
    ("norm_for_input_proj", "if (!fold_cast && (!lib_gemm || !attn_layer))"),
    ("attn_gated_for_o_proj", "if (!fold_cast && !lib_gemm)"),
    ("gdn_out_for_out_proj", "if (!fold_cast && !lib_gemm)"),
    ("norm_for_mlp_gateup", "if (!fold_cast && !lib_gemm)"),
    ("mlp_act_for_mlp_down", "if (!fold_cast && !lib_gemm)"),
]
# The sixth site is in `finish_token`, which has no body width to gate on: it branches on the scratch
# flag directly.
FINISH_GUARD = "if (sc.cast_fold) {"

# Each folded consumer: the entry point the fold calls, and the launcher declaration that must exist.
FOLDED_ENTRY_POINTS = [
    "proj_rows_xbf16",
    "proj_gemv_xbf16",
    "mlp_gateup_xbf16",
    "mlp_down_xbf16",
    "lm_head_full_xbf16",
]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=900)


def step(what: str) -> None:
    print(f"== {what}")


def main() -> int:
    failures: list[str] = []

    step("1. population: prototype/cast_fold/census_sites.py")
    res = run([sys.executable, str(CENSUS)])
    if res.returncode != 0:
        print(res.stdout + res.stderr)
        failures.append("census_sites.py is red; the fold's population is not what its table claims")
        census_sites = 0
        census_launches = 0
    else:
        for line in res.stdout.splitlines():
            m = re.search(r"(\d+) call sites, (\d+) launches/token", line)
            if m:
                census_sites, census_launches = int(m.group(1)), int(m.group(2))
        print("   " + (res.stdout.strip().splitlines()[-1] if res.stdout else ""))
        if census_sites == 0 or census_launches == 0:
            failures.append("census printed no counts — an empty parse cannot fail")

    step("2. operand equivalence: build and run equiv_check (host-only, no device)")
    if not EQUIV_CPP.exists():
        failures.append(f"{EQUIV_CPP} missing")
    else:
        EQUIV_BIN.parent.mkdir(parents=True, exist_ok=True)
        # The rpath is not decoration: `hip_bfloat16.h`'s conversion is inline, but hipcc still links
        # libamdhip64 and the binary then dies at exec with "cannot open shared object file"
        # (prototype/README.md's `-Wl,-rpath,/opt/rocm/lib` rule, and this check is not exempt from it).
        build = run(["hipcc", "-O2", "-Wl,-rpath,/opt/rocm/lib", "-I.", "-Iengine",
                     str(EQUIV_CPP), "-o", str(EQUIV_BIN)])
        if build.returncode != 0:
            print(build.stdout + build.stderr)
            failures.append("equiv_check.cpp did not compile")
        else:
            prog = run([str(EQUIV_BIN)])
            print("\n".join("   " + ln for ln in prog.stdout.strip().splitlines()))
            if prog.returncode != 0:
                failures.append(f"equiv_check exited {prog.returncode}")
            d = dict(re.findall(r"(\w+)=(\d+)", prog.stdout))
            if "SITES_CHECKED" not in d or "LAUNCHES_COVERED" not in d:
                failures.append("equiv_check printed no SITES_CHECKED/LAUNCHES_COVERED — "
                                "a check whose count cannot be read is not a check")
            else:
                got_sites, got_launches = int(d["SITES_CHECKED"]), int(d["LAUNCHES_COVERED"])
                print(f"   equiv_check: sites={got_sites} launches={got_launches}; "
                      f"census: sites={census_sites} launches={census_launches}")
                if got_sites == 0 or got_launches == 0:
                    failures.append("equiv_check checked nothing")
                if (got_sites, got_launches) != (census_sites, census_launches):
                    failures.append(
                        f"the equivalence check covers {got_sites} sites / {got_launches} launches "
                        f"but the fold claims {census_sites} / {census_launches}")

    step("3. wiring: engine/decode_step.cpp actually gates every site on the fold")
    src = STEP_CPP.read_text()
    for name, guard in REQUIRED_GUARDS:
        if guard not in src:
            failures.append(f"{name}: guard `{guard}` is not in decode_step.cpp")
    if FINISH_GUARD not in src:
        failures.append("final_norm_for_lm_head: `finish_token` does not branch on sc.cast_fold")
    if "const bool fold_cast = sc.cast_fold && (n == 1);" not in src:
        failures.append("the fold predicate `fold_cast` is not defined as `sc.cast_fold && (n == 1)`; "
                        "if the one-row rule moved, restate this check with it")
    for entry in FOLDED_ENTRY_POINTS:
        decl = KERNELS_H.read_text()
        if f"void {entry}(" not in decl:
            failures.append(f"kernels.h does not declare {entry}")
        if f"kernels::{entry}(" not in src and f"{entry}(" not in src:
            failures.append(f"{entry} is declared but never called from decode_step.cpp")

    step("4. default side is inert: the shipped path must not consult the fold")
    # `cast_bf16_to_f32` must still be reachable with the flag off, and every per-token call must sit
    # under a `!fold_cast`/`!sc.cast_fold` guard — otherwise the flag does not select a side and the
    # A/B compares one binary against itself. Counted by the callee name, not by a line-shaped regex:
    # two of the eight calls wrap across lines and a `^...;$` pattern silently matches nothing.
    n_cast_calls = src.count("kernels::cast_bf16_to_f32(")
    if n_cast_calls != 8:      # 6 per-token + 2 one-time adapter widenings
        failures.append(f"expected 8 cast_bf16_to_f32 calls in decode_step.cpp, found {n_cast_calls}")
    n_guarded = src.count("if (!fold_cast &&") + src.count("if (sc.cast_fold) {")
    if n_guarded != 6:
        failures.append(f"only {n_guarded} of the 6 per-token cast sites are guarded on the fold")

    step("5. the declared launch bill moves with the fold (the plan self-check must not go red)")
    # Deleting 97 submissions is a change to §07's per-token bill, and the first cut of this route left the
    # declared bill at 340 while 243 were submitted — so the engine's own `plan_match` read 0 on every
    # folded run. `LAUNCH` cannot carry the fold (`gate.py` section (d) re-derives that line in Python and
    # the cross-check stops being independent if the echo is a function of the engine's flags), so the
    # fold-in-force lives in `planned_launch_counts(cast_fold)` and the stage clock is TOLD it, exactly as
    # route D1's `set_plan_proj` works. These rows are the plumbing that makes that true; each one severed
    # reads `plan_match=0` on a folded run, which is the failure this section exists to catch on CPU.
    decl_h = STEP_H.read_text()
    if "planned_launch_counts(bool cast_fold = false)" not in decl_h:
        failures.append("decode_step.h: `planned_launch_counts` does not take a defaulted `cast_fold` "
                        "parameter, so nothing can bill a folded step without editing §07's bill")
    if "c.cast = cast_fold ? 0 :" not in src:
        failures.append("decode_step.cpp: the `cast` row of the plan is not fold-aware")
    sc_h, sc_cpp = STAGE_CLOCK_H.read_text(), STAGE_CLOCK_CPP.read_text()
    for needle, why in [
        ("void set_plan_cast_fold(bool cast_fold)",
         "stage_clock.h: no setter, so a step clock cannot be told the fold and bills 97 widenings anyway"),
        ("bool cast_fold = false;", "stage_clock.h: WaveShape cannot carry the fold, so a wave bill cannot"),
    ]:
        if needle not in sc_h:
            failures.append(why)
    for needle, why in [
        ("planned_launch_counts(cast_fold)",
         "stage_clock.cpp: `plan_bill` does not ask §07's deriver for the folded bill"),
        ("plan_bill(plan_proj_, plan_cast_fold_)",
         "stage_clock.cpp: the step report ignores the fold it was told about, which is the exact defect "
         "that made `plan_match=0` on a folded run"),
        ("sh.cast_fold ? single_row_bodies : 0",
         "stage_clock.cpp: `wave_plan_bill` does not fold a one-row body's casts"),
        ("sh.cast_fold ? 0 : priced",
         "stage_clock.cpp: `wave_plan_bill` does not fold `finish_token`'s widening, which every width loses"),
        ("body1_castfold_plan_matches_token_plan",
         "stage_clock.cpp: the folded wave bill has no self-check, so the step bill and the wave bill could "
         "drift apart with nothing printing it"),
    ]:
        if needle not in sc_cpp:
            failures.append(why)
    run_cpp = DECODE_RUN_CPP.read_text()
    for needle, why in [
        ("set_plan_cast_fold(o.cast_fold)",
         "decode_run.cpp: the step clock is never told, so a folded run reads plan_match=0"),
        ("shape.cast_fold = o.cast_fold", "decode_run.cpp: the wave clock is never told"),
        ("step_bill=%d", "decode_run.cpp: the CASTFOLD line does not print the bill the clock will use, so "
                         "the two print sites could disagree without an observable"),
    ]:
        if needle not in run_cpp:
            failures.append(why)

    print()
    if failures:
        for f in failures:
            print("EQUIV FAIL: " + f)
        return 1
    print("EQUIV OK: 6 sites / 97 launches, operand values bit-identical at every index and every "
          "BF16 value, every site gated on the fold, all five folded entry points called, and the declared "
          "bill moves with the fold on both clocks.")
    print("STILL OPEN ON DEVICE: bit-identity of the two spellings' machine code. That is "
          "`QWEN_CAST_FOLD=1 engine/gate.py` and ab_item1.sh's sampled.txt byte-diff, not this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
