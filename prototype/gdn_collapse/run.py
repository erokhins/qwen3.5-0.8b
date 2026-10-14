#!/usr/bin/env python3
"""Gate driver for route A: the collapsed GDN recurrence (one launch per layer per wave body).

WHAT IT RUNS, and what it does not. Everything here is a single GDN layer in a standalone binary
(`gdn_collapse_check.hip.cpp`): both scan forms over the same operands and the same starting state,
compared BIT for BIT, plus the byte-touch checks (every `o` element written, guard bands untouched).
It does NOT take timings — route A's milliseconds come from the engine's `--time-waves` report, on
the `gdn_rec` row, where D-12's win line is written (`bench/p9-route-a-2026-10-14.md` §5). It does
not replace `engine/gate.py --gdn-scan collapse`, which is the end-to-end authority on the shipped
path; this is the fast loop that catches a staging index or an association bug in seconds instead of
a five-minute HSA hang-analysis through the engine.

GPU LOCK: this driver does NOT acquire `tools/gpu_lock.sh`. Per the AGENTS trap about self-locking
gate drivers deadlocking a caller that already holds the lock, the caller (the main agent, the only
one allowed to take device measurements) holds it. Check with `grep -c gpu_lock prototype/*/run.py`.

    ./prototype/gdn_collapse/run.py              # shipping form + the four negatives
    ./prototype/gdn_collapse/run.py --no-neg     # shipping form only

The negatives are separate -D binaries (`EXTRA_FLAGS` in build.sh), never runtime branches, so a
defect is not reachable from the shipping binary. Each one is required to LOCALISE, not merely to
fail, and the localisation is per-defect and derived from what the defect does:

  NEG_RC_STATE_BF16  moves the state AND `o` when the body still carries its incoming state
                     (small decay), and must be CLEAN when it does not — a body that starts from
                     zeros has nothing to round, so staging a rounded zero is still zero. That
                     control is the reason the long-memory cases exist; with the default decay a
                     body erases its incoming state within itself, and this defect then hides in
                     the state and shows only in `o`.
  NEG_RC_STRIP_SHIFT leaves `o` columns UNWRITTEN (two strips of a head collide, four own nothing)
                     and moves the state. Unwritten columns are the localisation: the shape of the
                     output does not change, only who filled it.
  NEG_RC_NORM_HALF   moves everything: `o` and the state, all of it, because k̃ is wrong on every
                     column from the first token.
  NEG_RC_I_DESC      moves a SUBSET — §04.4's dot cancels, so re-associating it moves only the
                     elements with decision room. A defect that moved everything here would be a
                     different defect.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BIN = HERE / "gdn_collapse_check"

CASE_RE = re.compile(
    r"CASE (?P<label>.+?)\s+tokens=\s*(?P<tokens>\d+)\s+body=\s*(?P<body>\d+)\s+"
    r"state=(?P<state>\S+)\s+decay=(?P<decay>\S+)\s+seed=(?P<seed>\d+)\s+"
    r"o bit-diff\s+(?P<odo>-?\d+)/(?P<on>\d+)\s+state bit-diff\s+(?P<dsd>-?\d+)/(?P<sn>\d+)\s+"
    r"o unwritten seq=(?P<uref>\d+) collapse=(?P<unew>\d+)\s+"
    r"guard violations o=(?P<go>\d+) state=(?P<gs>\d+) operands=(?P<gp>\d+)\s+(?P<verdict>\w+)"
)


def run(argv, build_flags=None):
    """Build (optionally with -D flags) and run one invocation; return (stdout, returncode)."""
    env_build = []
    if build_flags:
        # single-quoted, because EXTRA_FLAGS is word-split inside build.sh and two -D flags would
        # otherwise put the second one on the command line as a program name.
        env_build = ["EXTRA_FLAGS='" + " ".join(build_flags) + "'"]
    cmd = ["bash", "-c", " ".join(env_build + ["./prototype/gdn_collapse/build.sh"]) ]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout, p.stderr)
        raise SystemExit(f"build failed ({build_flags}): see above")
    p = subprocess.run([str(BIN)] + argv, cwd=REPO, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    cases = [m.groupdict() for m in CASE_RE.finditer(out)]
    if not cases and p.returncode not in (0, 1):
        print(out)
        raise SystemExit("the check binary failed to run (see above)")
    # A parser that silently returns nothing turns this whole driver into a check that cannot fail —
    # the trap this repo already carries for gate sections that count nothing. Every `CASE` line the
    # binary printed must have been parsed, or the run is an error, not a pass.
    if len(cases) != out.count("CASE "):
        print(out)
        raise SystemExit(f"parsed {len(cases)} of {out.count('CASE ')} CASE lines: the driver's "
                         "regex is out of step with the binary's output, so nothing here was judged")
    return out, p.returncode, cases


def by_label(cases, needle):
    for c in cases:
        if needle in c["label"]:
            return c
    raise SystemExit(f"no case matching {needle!r} in the output")


def moved(c):
    return int(c["odo"]) != 0 or int(c["dsd"]) != 0


def byteclean(c):
    return (int(c["go"]), int(c["gs"]), int(c["gp"])) == (0, 0, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neg", action="store_true", help="skip the four injected-defect builds")
    args = ap.parse_args()

    failures = []

    # ---- 1. the shipping form: every case must be bit-identical and byte-clean.
    out, rc, cases = run([])
    print(out.rstrip())
    if len(cases) != 8:
        failures.append(f"matrix ran {len(cases)} cases, expected 8")
    if rc != 0:
        failures.append("shipping matrix")
    else:
        for c in cases:
            if moved(c) or int(c["uref"]) or int(c["unew"]) or not byteclean(c):
                failures.append(f"shipping case {c['label']!r}")

    if args.no_neg:
        return report(failures)

    # ---- 2. the negatives: each must localise, and where a control exists it must be CLEAN.
    # (build flag, catch args, control args or None, what the catch must show)
    negatives = [
        ("-DNEG_RC_STATE_BF16", ["--tokens", "1024", "--body", "512", "--state", "random",
                                 "--decay", "0.002"],
         ["--tokens", "512", "--body", "512", "--state", "zeros"],
         "state rounding shows in the STATE when the body carries one, and nowhere when it starts "
         "from zeros"),
        ("-DNEG_RC_STRIP_SHIFT", ["--tokens", "512", "--body", "512", "--state", "random"],
         None, "strip base from the head index leaves o columns unwritten"),
        ("-DNEG_RC_NORM_HALF", ["--tokens", "512", "--body", "512", "--state", "random"],
         None, "half-width norm moves o and the state"),
        ("-DNEG_RC_I_DESC", ["--tokens", "1024", "--body", "512", "--state", "random"],
         None, "descending dot moves a subset (the cancellation case)"),
    ]
    for flag, catch, control, why in negatives:
        flags = [flag, "-DNEG_RC_ANY_EXPECT_BREAK"]
        out, _, cases = run(catch, build_flags=flags)
        c = cases[0]
        ok = moved(c) and byteclean(c)
        if flag.endswith("STRIP_SHIFT"):
            ok = ok and int(c["unew"]) > 0
        if flag.endswith("STATE_BF16"):
            ok = ok and int(c["dsd"]) > 0
        if flag.endswith("NORM_HALF"):
            ok = ok and int(c["odo"]) == int(c["on"])
        line = (f"  {flag:26s} o {c['odo']}/{c['on']}  state {c['dsd']}/{c['sn']}  "
                f"unwritten {c['unew']}  -> {'CAUGHT' if ok else 'MISSED'}")
        if control is not None:
            out2, _, cases2 = run(control, build_flags=flags)
            c2 = cases2[0]
            ctrl_ok = not moved(c2) and byteclean(c2)
            line += (f" | control o {c2['odo']}/{c2['on']} state {c2['dsd']}/{c2['sn']}"
                     f" -> {'CLEAN' if ctrl_ok else 'CONTROL MOVED'}")
            ok = ok and ctrl_ok
        print(line)
        print(f"      ({why})")
        if not ok:
            failures.append(flag)

    # Rebuild the shipping binary so the tree is left with the binary the -D-less source builds.
    run(["--tokens", "8", "--body", "8"], build_flags=None)
    return report(failures)


def report(failures) -> int:
    if failures:
        print("GDN COLLAPSE GATE FAIL: " + "; ".join(failures))
        return 1
    print("GDN COLLAPSE GATE PASS (bit-identical to the shipped sequential form; "
          "four injected defects each localised)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
