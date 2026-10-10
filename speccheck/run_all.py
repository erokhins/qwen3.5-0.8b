#!/usr/bin/env python3
"""Run every §10 gate over every fixture and print one verdict table.

This exists so that "all gates pass" is a single reproducible claim instead of a
list of commands someone may or may not have run. Exits non-zero if any gate fails.

  .venv/bin/python speccheck/run_all.py            # everything (~3 min on this box)
  .venv/bin/python speccheck/run_all.py --quick    # smoke/dec8 + one form (~1 min)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ALL = ["smoke", "p8", "p64", "p300", "dec8", "prefix24dec8"]
FIXTURES_QUICK = ["smoke", "dec8"]


def run(py, label, argv, expect_fail=False):
    t0 = time.time()
    p = subprocess.run([py] + argv, capture_output=True, text=True)
    out = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    tail = out[-1] if out else (p.stderr or "").strip().splitlines()[-1:]
    ok = (p.returncode != 0) if expect_fail else (p.returncode == 0)
    print("%-30s %s  %5.1fs  %s" % (label, "PASS " if ok else "FAIL ",
                                     time.time() - t0, str(tail)[:110]))
    if not ok:
        for ln in out[-6:]:
            print("      |", ln[:150])
        for ln in (p.stderr or "").splitlines()[-4:]:
            print("      |", ln[:150])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--corpus", default="corpus48")
    ap.add_argument("--floor", default="corpus48sdpa")
    args = ap.parse_args()
    py = args.python
    fixtures = FIXTURES_QUICK if args.quick else FIXTURES_ALL
    forms = ["chunk"] if args.quick else ["recurrent", "chunk"]
    g = ["speccheck", "gates"]
    G = "speccheck/gates"
    results = []

    print("== tensors: speccheck vs reference dumps ==")
    for tag in fixtures:
        import json
        dec = json.loads((ROOT / "data" / "refs" / tag / "manifest.json").read_text()
                         ).get("greedy_continuation", [])
        n = min(max(len(dec) - 1, 0), 8)
        for form in forms:
            eng = f"data/eng/{tag}_{form}.npz"
            run(py, f"dump {tag}/{form}",
                ["speccheck/specimpl.py", "--prompt-ids-file",
                 f"data/refs/{tag}/manifest.json", "--decode", str(n),
                 "--gdn-prefill", form, "--dump", eng])
            results.append(run(py, f"G0-G4 {tag}/{form}",
                               [f"{G}/run_gates.py",
                                "--ref", f"data/refs/{tag}", "--eng", eng, "--top", "0"]))

    print("\n== decision and invariant gates ==")
    results.append(run(py, "G7 tokens (all fixtures)",
                       [f"{G}/run_g7.py", "--fixtures", ",".join(fixtures)]))
    results.append(run(py, "G7 corpus vs self-consistency floor",
                       [f"{G}/run_g7.py", "--corpus", args.corpus,
                        "--floor", args.floor]))
    results.append(run(py, "G6 online softmax",
                       [f"{G}/run_g6.py", "--fixtures", "dec8,p300", "--steps", "12"]))
    for s in (["3"] if args.quick else ["3", "64", "65"]):
        results.append(run(py, f"G5 incremental==batch split={s}",
                           [f"{G}/run_g5.py", "--split", s, "--extend", "4",
                            "--gdn-prefill", "chunk"]))
    for tag in (["smoke"] if args.quick else ["smoke", "p300"]):
        results.append(run(py, f"G3i isolated {tag}",
                           [f"{G}/run_isolated.py", "--fixture", tag]))
        results.append(run(py, f"G3 chunk==recurrent {tag}",
                           [f"{G}/run_chunkcheck.py", "--fixture", tag]))
    results.append(run(py, "G-neg negative controls", [f"{G}/run_negative.py"]))
    results.append(run(py, "G8 MTP (A-12/A-13 accept rate, lossless loop)",
                       [f"{G}/run_mtp.py", "--k", "1", "--steps", "16" if args.quick
                        else "24", "--prompts", "4" if args.quick else "6"]))
    if not args.quick:
        results.append(run(py, "G8 MTP k=2", [f"{G}/run_mtp.py", "--k", "2",
                                              "--steps", "24", "--prompts", "4"]))
    results.append(run(py, "A-11/A-10 tokenizer+render",
                       [f"{G}/run_tok.py", "--fixtures", ",".join(
                           [t for t in fixtures if t not in ("p8", "prefix24dec8")])]))

    nfail = results.count(False)
    print("\n%d gates run, %d failed" % (len(results), nfail))
    print("ALL GATES PASS" if not nfail else "SOME GATES FAILED")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
