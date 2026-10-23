#!/usr/bin/env python3
"""prompt_fan_paired.py — is a fan-width difference real, or two binomials passing in the street?

WHY THIS EXISTS. The fan table compares accept FRACTIONS from separate runs, so the honest error bar the
rate probe prints is the pair's binomial half-width (±0.10 at 170 rounds) and every fan arm lands "within
noise" against the loop arm on that bar. But the two arms are not independent samples: they run the same
fixture, the same greedy stream (the admission check requires the emitted stream to be greedy or a near-tie
flip of it), and the same positions. What differs is which settled positions an accepting round landed on.
That makes this a PAIRED comparison, and the paired statistic uses only the rounds where the arms disagree —
so its sensitivity is set by the discordant count, not by the 170 rounds both arms share.

THE TEST. McNemar on the discordant pairs, exact binomial p (two-sided) rather than the chi-square, because
the discordant count is small enough that the asymptotic form is the wrong approximation. The null is "the
fan changes which positions are accepted at not at all"; the alternative is that it does. Reported beside
the raw rates, because a paired test on a fixed fixture says nothing about a different fixture.

PAIRING KEY. `ACCEPT … cand_pos=… accept=…`, one line per round: `cand_pos` is the POSITION whose token the
round tried to land, so the same number in the two logs means the same thing in the run and the join is on a
position rather than on a round ordinal. A round moves the target on by two positions when it accepts and one
when it rejects, so the two arms' geometries diverge the moment their verdicts do and the k-th round of one
arm is not the k-th of the other; joining on a slid index has produced false accept rates in this repo before
(`AGENTS.md`). A position settled by only one arm is NOT a discordant pair — it is a position the other arm
never tried to land, because the arm that accepted got that token as a bonus instead of asking for it — so it
is reported separately and not counted.

NO TIMING: this file reads no clock and prints no millisecond figure.

Usage:
    .venv/bin/python3 prototype/mtp_accept/prompt_fan_paired.py LOOP_LOG WIDE_LOG [--label fan146]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys

ACCEPT_RE = re.compile(r"^ACCEPT round=\d+ anchor=\d+ fed=\d+ cand=\d+ cand_pos=(\d+) accept=(\d)")


def settled(path: pathlib.Path) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for line in path.read_text().splitlines():
        m = ACCEPT_RE.match(line)
        if not m:
            continue
        pos, accept = int(m.group(1)), m.group(2) == "1"
        if pos in out and out[pos] != accept:
            raise SystemExit(f"{path}: position {pos} settled twice with different verdicts; the accept "
                             "tally cannot be paired against a run that adjudicates a position twice")
        out[pos] = accept
    if not out:
        raise SystemExit(f"{path}: no `ACCEPT ... cand_pos= accept=` lines; nothing to pair")
    return out


def two_sided_exact_binom(k: int, n: int) -> float:
    """P(|X - n/2| >= |k - n/2|) for X ~ Binom(n, 1/2) — McNemar's exact form."""
    if n == 0:
        return 1.0
    def pmf(i: int) -> float:
        return math.comb(n, i) / (2 ** n)
    tail = abs(k - n / 2)
    return sum(pmf(i) for i in range(n + 1) if abs(i - n / 2) >= tail - 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("loop_log", type=pathlib.Path, help="the fan-0 arm's captured stdout")
    ap.add_argument("wide_log", type=pathlib.Path, help="the seeded arm's captured stdout")
    ap.add_argument("--label", default="wide", help="name of the wide arm, for the printed line")
    a = ap.parse_args()

    lo, hi = settled(a.loop_log), settled(a.wide_log)
    both = sorted(set(lo) & set(hi))
    b = sum(1 for p in both if not lo[p] and hi[p])       # loop rejected, wide accepted
    c = sum(1 for p in both if lo[p] and not hi[p])       # loop accepted, wide rejected
    agree = sum(1 for p in both if lo[p] == hi[p])
    only_lo, only_hi = len(set(lo) - set(hi)), len(set(hi) - set(lo))
    n = b + c
    p = two_sided_exact_binom(b, n)
    # McNemar's continuity-corrected chi-square, printed alongside the exact p as a cross-check.
    chi = (abs(b - c) - 1) ** 2 / (b + c) if n else float("nan")
    rate = lambda d: sum(d.values()) / len(d)
    print(f"paired accept read: {a.loop_log.name} (a={rate(lo):.4f}, {sum(lo.values())}/{len(lo)}) vs "
          f"{a.label} (a={rate(hi):.4f}, {sum(hi.values())}/{len(hi)})")
    print(f"  positions settled by both: {len(both)}  (agree {agree}, disagree {n})")
    print(f"  discordant: wide-only-accept b={b}   loop-only-accept c={c}")
    print(f"  settled by one arm only: loop={only_lo} wide={only_hi} (round geometry moved; NOT counted)")
    if n == 0:
        print("VERDICT: no discordant position, so the seed moved no verdict on this fixture. That is a null"
              " about the verdicts, not about the proposals — the side-pass control is what shows proposals"
              " changed.")
        return 0
    print(f"  net = {b - c:+d} accepts in favour of {a.label}; exact McNemar p = {p:.4g}"
          f"  (chi2 cc = {chi:.3f}, 1 df)")
    print(f"VERDICT: {'REJECT the null — the fan moves verdicts (p < 0.05)' if p < 0.05 else 'NOT resolved'}"
          f" — {n} discordant position(s) of {len(both)} settled by both arms. Paired sensitivity is set by"
          " the discordant count, which is why this can resolve where the ±binomial pair bar cannot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
