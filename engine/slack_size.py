#!/usr/bin/env python3
"""The sizing table for the gate's quantization-slack rule: every fixture, every step, one line each.

`progress/2026-10-12/plan-2026-10-12.md` D-9 refused to admit a token the pin never decided until the admitting rule
was SIZED, and the brief for sizing it asked for a table, not an argument: for each judged step, the pin's
own top-1/top-2 margin (in value and in BF16 ULP at that magnitude), the step's licensed element error, and
whether the pin can decide the token under `margin < bound`. This prints that, and nothing else, so the
rule in `engine/gate.py`'s QUANTIZATION SLACK block can be read against the corpus it was derived from.

    .venv/bin/python engine/slack_size.py                       # all eight fixtures
    .venv/bin/python engine/slack_size.py --fixture chunk64b    # one fixture
    .venv/bin/python engine/slack_size.py --engine-log tmp/gate_logs   # add the engine's measured column
    .venv/bin/python engine/slack_size.py --only-close           # only the steps within 5x of the line

CPU-only. It reads pinned dumps and each fixture's cached `route_allowance.json`, starts no binary,
touches no device, prices nothing. The bound is not recomputed here: `gate.step_slack` is the same
function the gate judges with, so this table cannot report a margin against a rule the gate no longer
uses — the pattern `prototype/mlp_gemv/floor_probe.py` sets for a band it does not own.

The `--engine-log` column is the only measured number in the table, and it comes from a gate log rather
than from a second device pass: `engine/gate.py` prints `maxabs` on every step's metrics line, so
pointing this at a directory of gate logs (`dec64f.log`, `chunk64b.log`, …) adds the engine's ACTUAL
element error beside the licence it was granted. That column is what shows the bound is the conservative
of the two — at `chunk64b` step 0 the licence is 0.162049 and the engine's measured error is 0.2509 — and
which steps a measured bound would have excused that this one does not (`dec300f` step 30, maxabs 0.2773
against a margin of 0.25).

What the table is FOR, in one line each:
  * it must show `chunk64b` step 0 inside the undecidable region, and it does (0.125 vs 0.162049);
  * it must show every currently-green step outside it, and it does — the only steps in the corpus inside
    the region are `chunk64b` steps 0, 1, 13 and 22, plus the exact ties that D-1's proved-tie predicate
    already owns (margin 0, a pin with no winner at all is a different question);
  * it shows how near the line the nearest refused step sits (`dec300f` step 30, 1.46x), because a rule
    whose refused steps are all 100x outside it is not being tested.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import gate    # the rule itself; this file only reads it and prints what it says

DEFAULT_FIXTURES = ["dec8", "dec64f", "dec64t", "dec300f", "mt32f", "p128", "wide520f", "chunk64b"]
# `engine/gate.py` prints one of these per judged step; the number is taken verbatim, never recomputed.
MAXABS = re.compile(r"\[PASS\] step (\d+): metrics .* maxabs ([\d.eE+-]+)")


def measured_from_log(path: Path) -> dict:
    """`{step: maxabs}` from one gate log, or {} if it was not given. Parse-only: no re-measurement."""
    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        m = MAXABS.search(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", action="append", default=None,
                    help="a fixture to price; repeatable. Default: the eight the gate runs")
    ap.add_argument("--engine-log", default=None, metavar="DIR",
                    help="directory of gate logs named <fixture>.log, to add the engine's MEASURED "
                         "maxabs column next to the licensed bound")
    ap.add_argument("--only-close", action="store_true",
                    help="print only the steps inside the region or within 5x of its bound")
    args = ap.parse_args()

    print("quantization-slack sizing — bound(j) = meanrel cap in force at step j x mean|pin_j|")
    print("  (the policed band read out in logits units; see engine/gate.py's QUANTIZATION SLACK block)")
    print("  rule: margin < bound  =>  the pin cannot decide the token; the engine's id is then excused")
    print("          only if it is one of the pinned ids inside that margin (D-1's proved-tie policy")
    print("          owns margin == 0, where the pin has no winner at all)")
    corpus_inside, corpus_close = [], []
    for fx in (args.fixture or DEFAULT_FIXTURES):
        gate.set_fixture(fx)
        n = gate.fixture_step_count()
        measured = measured_from_log(Path(args.engine_log) / f"{fx}.log" if args.engine_log else None)
        print(f"\n== {fx}: {n} judged steps"
              + (f", measured maxabs from {args.engine_log}/{fx}.log" if measured else
                 ", no engine log given (licensed bound only)"))
        print(f"  {'step':>4} {'top1':>7} {'v1':>9} {'top2':>7} {'v2':>9} {'margin':>9} {'ULP':>6}"
              f" {'bound':>9} {'cap':>8} {'m/bound':>8}"
              + (f" {'maxabs':>8}" if measured else "") + "   verdict")
        inside, ties, refused = [], [], []
        for j in range(n):
            sl = gate.step_slack(j, gate.pin_row(j))
            if sl is None:
                tie = gate.pin_tie(j)
                if tie is None:
                    print(f"  {j:>4} {'—':>7} {'—':>9} {'—':>7} {'—':>9} {'—':>9} {'—':>6}"
                          f" {'—':>9} {'—':>8} {'—':>8}"
                          + (f" {'—':>8}" if measured else "")
                          + "   NO BAND — the rule refuses (fail closed)")
                    continue
                ties.append(j)
                print(f"  {j:>4} {tie['top1']:>7} {tie['value']:>9} {'(tied)':>7} "
                      f"{tie['value']:>9} {0.0:>9} {0.0:>6} {'—':>9} {'—':>8} {'—':>8}"
                      + (f" {measured.get(j, float('nan')):>8.4f}" if measured else "")
                      + f"   exact {len(tie['ids'])}-way tie — D-1's proved-tie policy, not this rule")
                continue
            ratio = sl["margin"] / sl["bound"]
            verdict = ("INSIDE — pin cannot decide" if sl["margin"] < sl["bound"]
                       else "decidable by the pin")
            line = (f"  {j:>4} {sl['top1']:>7} {sl['v1']:>9.4g} {sl['top2']:>7} {sl['v2']:>9.4g}"
                    f" {sl['margin']:>9.5f} {sl['ulps']:>6.1f} {sl['bound']:>9.5f}"
                    f" {sl['cap']:>8.4%} {ratio:>8.2f}"
                    + (f" {measured.get(j, float('nan')):>8.4f}" if measured else "")
                    + f"   {verdict}")
            if sl["margin"] < sl["bound"]:
                inside.append(j)
                corpus_inside.append(f"{fx} step {j}")
            elif ratio <= 5.0:
                refused.append((ratio, j))
                corpus_close.append((ratio, f"{fx} step {j}"))
            if sl["margin"] < sl["bound"] or ratio <= 5.0 or not args.only_close:
                print(line)
        nearest = min(refused) if refused else None
        print(f"  {fx}: {len(inside)} step(s) inside the region"
              f" ({', '.join(str(j) for j in inside) if inside else 'none'});"
              f" {len(ties)} exact tie(s) ({', '.join(str(j) for j in ties) if ties else 'none'})"
              " belong to D-1; nearest refused step"
              + (f" {nearest[1]} at {nearest[0]:.2f}x its bound" if nearest else " none")
              + (f"; measured maxabs at the inside steps: "
                 + ", ".join(f"{j}={measured[j]:.4f}" for j in inside if j in measured)
                 if inside and measured else ""))
    print(f"\n== corpus: {len(corpus_inside)} step(s) inside the region —"
          f" {', '.join(corpus_inside) if corpus_inside else 'none'}")
    if corpus_close:
        corpus_close.sort()
        print(f"   the five refused steps nearest the line: "
              + ", ".join(f"{what} ({ratio:.2f}x)" for ratio, what in corpus_close[:5]))
    print("   steps outside `chunk64b` the rule could reach: "
          + (", ".join(s for s in corpus_inside if not s.startswith("chunk64b")) or "none"
             ) + "   <- must be empty, or the rule broadens past the fixture D-9 named")
    return 0


if __name__ == "__main__":
    sys.exit(main())
