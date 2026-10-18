#!/usr/bin/env python3
"""check_fattn_cb.py — executes the C->B `get_half2` relabelling instead of arguing it.

WHAT IT DECIDES
`prototype/attn_wmma/wmma_fattn_score.cpp` drains the score tile's 8 F32 registers, converts them to 16
F16 halves with `p_to_operand` (the transcribed `get_half2`, `mma.cuh:729-743`), and hands the result to
the next `v_wmma` as an operand. The score-map probe (`check_fattn_map.py`) says where a C value LANDS and
says nothing about this step, and the handover's own cross-check for it — "its output shape is exactly the
A/B operand shape" — is a shape argument, which is the class of check that just let a transposed score
reach a PASS. This file runs one probe per build and decides the claim by closed form:

    Q[q][0]=16q, Q[q][1]=1; K[kv][0]=1, K[kv][1]=kv  =>  score[q][kv] = 16q + kv   (all values <= 255, so
    V[kv][d] = [kv == d & 15]                          =>  out[t][l]  = 16*(t%16 rows) + (2l + lane/16)

    every step exact in BF16 and F16, so the comparison is equality and no band, floor probe, or cos
    tolerance is involved. A layout claim either lands the cell or does not.

THE STAGE MATRIX, which is the whole point
Verdicts are per STAGE (score / operand / out), not one AND, because the injections are only meaningful if
each one is required to appear WHERE IT LIVES — the rule `prototype/mlp_gemv` and `prototype/attn_tile_model.py`
both record: a negative that catches everything proves nothing, and a negative that loads the wrong layer's
weights catches every injection no matter what the kernel does. The expected matrix is therefore derived
from which stage each mutation touches:

    case                  score   operand   out     what it pins
    clean                 PASS    PASS      PASS    the claim itself
    cb_lane_half_swap     PASS    FAIL      FAIL    the butterfly's partner slot
    cb_pack_swap          PASS    FAIL      FAIL    the pair order inside each packed register
    cb_elem_reverse       PASS    FAIL      FAIL    register order within a lane's 8 values
    cb_no_shfl            PASS    FAIL      PASS    the cross-lane exchange is invisible AT THE OUTPUT.
                                                    get_half2 writes both halves of each pair into BOTH
                                                    half-warps, so without the exchange each half-warp
                                                    still holds its OWN parity correctly. What the
                                                    exchange buys is that both half-warps hold all 16
                                                    values, which is what the operand dump shows broken.
                                                    The operand stage is what catches it, and
                                                    `cb_v_odd_kv_zero` tests the reading this implies.
    cb_score_slot_swap    FAIL    FAIL      FAIL    the PRE-CORRECTION score leg: the transposed-score
                                                    regression, as a required-fail case, so it cannot
                                                    come back unnoticed (`fattn_score_handover.md`
                                                    CORRECTION)
    cb_vrow_off           PASS    PASS      FAIL    V tile -> row map: must move the output and leave the
                                                  operand alone
    cb_kv_both_relabel    PASS    FAIL      PASS    THE DERIVED BLIND SET, required NOT to fail on the
                                                  output: permuting KV identically in the P operand and
                                                  in the V stage cancels inside the dot. The operand
                                                  column MUST be red here, or the injection was never
                                                  expressed. What this probe witnesses is the
                                                  CONSISTENCY of the two layouts; the absolute naming of
                                                  the KV axis is the score-map probe's job.
    cb_v_odd_kv_zero      PASS    PASS      FAIL    zero every odd-KV entry of the staged V. Predicted to
                                                  move the output; if it ever passes while `clean` also
                                                  passes, the odd-KV staged entries are never read and V's
                                                  staged k-run is not the contiguous 16 the form assumes.

Any deviation from the matrix — a must-catch that passes, a must-not-catch that fails, a stage red that the
mutation cannot reach — is reported as a failure of THIS checker, not of the kernel, because a control whose
firing condition is not derived from the geometry is decoration.

DEVICE PROTOCOL (AGENTS.md, "GPU execution is serialized")
Correctness-only: one launch per case, PASS/FAIL, no millisecond figure, and **this script does not take
`tools/gpu_lock.sh` itself** — three gate drivers in this tree self-lock and a caller that already holds the
lock deadlocks it. Each case runs in its own subprocess, because one faulting kernel poisons the context and
every later case in the same process would silently never run.

  python3 prototype/attn_wmma/check_fattn_cb.py              # the clean probe only
  python3 prototype/attn_wmma/check_fattn_cb.py --selftest   # + the seven defect builds
  python3 prototype/attn_wmma/check_fattn_cb.py --expect N   # the check-count guard

Build first: `bash prototype/attn_wmma/build_fattn_cb.sh`. No timing is taken by anything in this file.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTDIR = REPO / "tmp" / "attn_wmma_cb"

# (binary suffix, probe name, expected (score, operand, out), why)
CASES = [
    ("clean", "clean", ("PASS", "PASS", "PASS"), "the claim"),
    ("lane_half_swap", "cb_lane_half_swap", ("PASS", "FAIL", "FAIL"), "butterfly partner slot"),
    ("pack_swap", "cb_pack_swap", ("PASS", "FAIL", "FAIL"), "make_half2 pair order"),
    ("elem_reverse", "cb_elem_reverse", ("PASS", "FAIL", "FAIL"), "register order in a lane's 8"),
    ("no_shfl", "cb_no_shfl", ("PASS", "FAIL", "PASS"),
     "the cross-lane exchange is NOT observable at the output: get_half2 writes both halves of every pair "
     "into both half-warps, so removing the exchange leaves each half-warp's own parity correct, which is "
     "the reading `cb_v_odd_kv_zero` then tests. The operand stage is what catches it"),
    ("score_slot_swap", "cb_score_slot_swap", ("FAIL", "FAIL", "FAIL"), "the transposed-score regression"),
    ("vrow_off", "cb_vrow_off", ("PASS", "PASS", "FAIL"), "V tile->row map, operand untouched"),
    ("kv_both_relabel", "cb_kv_both_relabel", ("PASS", "FAIL", "PASS"),
     "derived blind set: cancels in the dot"),
    ("v_odd_kv_zero", "cb_v_odd_kv_zero", ("PASS", "PASS", "FAIL"),
     "zeroing the odd-KV V entries must change the dot. If this ever PASSes while `clean` also passes, the "
     "odd-KV staged entries are never read, which means V's staged k-run is not the contiguous 16 this form "
     "assumes and the port must not trust it — a result, not a flaky case"),
]

CASE_RE = re.compile(
    r"CASE probe=(?P<name>\S+) lanes=(?P<lanes>\d+) parsed=(?P<parsed>\d+) of (?P<want>\d+) "
    r"bad_score=(?P<bs>\d+) bad_operand=(?P<bo>\d+) bad_out=(?P<bo2>\d+) "
    r"unwritten=(?P<unw>\d+) n_out=(?P<nout>\d+)")
VERDICT_RE = re.compile(
    r"VERDICT probe=(?P<name>\S+) score=(?P<score>PASS|FAIL) operand=(?P<operand>PASS|FAIL) "
    r"out=(?P<out>PASS|FAIL) overall=(?P<overall>PASS|FAIL)")


class Checks:
    def __init__(self) -> None:
        self.n = 0
        self.bad = 0

    def raw(self, what: str, ok: bool, detail: str = "") -> None:
        self.n += 1
        if not ok:
            self.bad += 1
            print(f"  FAIL {what}" + (f"  [{detail}]" if detail else ""))
        else:
            print(f"  pass {what}" + (f"  [{detail}]" if detail else ""))


def run_case(binary: Path) -> tuple[dict | None, dict | None, str]:
    """One subprocess per case: a fault in one must not silence the rest."""
    try:
        p = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, None, f"launch error: {e}"
    case = CASE_RE.search(p.stdout)
    verdict = VERDICT_RE.search(p.stdout)
    if case is None or verdict is None:
        return None, None, (p.stdout.strip() + "\n" + p.stderr.strip()).strip()[-400:]
    return case.groupdict(), verdict.groupdict(), ""


def adjudicate(C: Checks, suffix: str, probe: str, expect: tuple[str, str, str], why: str) -> None:
    binary = OUTDIR / f"cb_{suffix}"
    if not binary.exists():
        C.raw(f"{probe}: binary exists", False,
              f"missing {binary}; run: bash prototype/attn_wmma/build_fattn_cb.sh")
        return
    case, verdict, err = run_case(binary)
    if case is None:
        C.raw(f"{probe}: probe produced a parseable CASE+VERDICT line", False, err)
        return
    # A check whose input list can be empty cannot fail: require every dumped slot to have been written and
    # every record to have been parsed (`prototype/gdn_collapse/run.py`'s rule).
    C.raw(f"{probe}: parsed {case['parsed']} of {case['want']} records",
          case["parsed"] == case["want"], f"lanes={case['lanes']}")
    C.raw(f"{probe}: every output cell was written (n_out={case['nout']}, unwritten={case['unw']})",
          int(case["unw"]) == 0 and int(case["nout"]) == 32 * 16 * 8,
          f"n_out={case['nout']} want={32 * 16 * 8}")
    got = (verdict["score"], verdict["operand"], verdict["out"])
    C.raw(f"{probe}: stage matrix score={expect[0]} operand={expect[1]} out={expect[2]}",
          got == expect, f"got score={got[0]} operand={got[1]} out={got[2]}  ({why})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run the seven defect builds and require the derived stage matrix")
    ap.add_argument("--expect", type=int, default=None,
                    help="abort if the number of checks is not exactly this (default: the full matrix)")
    args = ap.parse_args()
    # The count guard defaults to the whole matrix: three checks per case, every case in the table. An
    # explicit --expect overrides it, and a wrong count is a failure either way (`prototype/gdn_collapse`).
    if args.expect is None:
        args.expect = 3 * (len(CASES) if args.selftest else 1)

    C = Checks()
    print("== check_fattn_cb — the C->B get_half2 relabelling, one launch per build")
    if not args.selftest:
        print("CLEAN PROBE")
        adjudicate(C, "clean", "clean", ("PASS", "PASS", "PASS"), "the claim")
        print("  note: only the clean probe ran. --selftest runs the seven defect builds and requires")
        print("        the derived stage matrix; a clean probe with no controls is an assertion.")
    else:
        print("STAGE MATRIX (must-catch derived from which stage each mutation touches)")
        for suffix, probe, expect, why in CASES:
            adjudicate(C, suffix, probe, expect, why)

    print(f"\n{'FAIL' if C.bad else 'PASS'}: check_fattn_cb — {C.n - C.bad}/{C.n} checks;"
          " correctness only, no timing taken")
    if args.expect is not None and args.expect != C.n:
        print(f"FAIL: expected {args.expect} checks, ran {C.n} — a check list that can be empty"
              " cannot fail")
        return 1
    return 1 if C.bad else 0


if __name__ == "__main__":
    sys.exit(main())
