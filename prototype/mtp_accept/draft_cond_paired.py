#!/usr/bin/env python3
"""draft_cond_paired.py — pair two `m3_loop_rate.py` witnesses line by line and ask if the delta is real.

WHY A PAIRED READ IS THE ONLY HONEST ONE HERE. `m3_loop_rate.py` reports one pooled rate per population,
and the two populations are 12 corpus lines of very different lengths — so a pooled delta of +0.026 against
a binomial half-width of +-0.085 "does not clear the noise", and neither does anything short of a hundred
lines. But the two arms ran the SAME 12 prompts against the SAME greedy stream, so the line is the unit of
pairing: a variant that really improves the guess should move the same lines up, not scatter. The sign test
and the sign-flip permutation over per-line deltas are the paired question; the pooled delta is the
headline a funding line would quote, and both are printed because they answer different doubts.

WHAT IT CANNOT SETTLE. The two arms' round anchors diverge as soon as one accept flips, so per-round pairs
are not matched and this is a paired-by-line test, not a McNemar test on matched rounds. That limit is
stated rather than hidden, because a stronger-looking test that is not valid is worse than a weaker one
that is.

USAGE
    prototype/mtp_accept/draft_cond_paired.py /tmp/dcp/m3new/m3_loop_rate.json \
            /tmp/dcp/m3dense/m3_loop_rate.json --label dense
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random


def per_line(p: pathlib.Path):
    """One m3_loop_rate json -> ({line: record}, whole-dict)."""
    d = json.loads(p.read_text())
    return {r["line"]: r for r in d["per_line"]}, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("variant")
    ap.add_argument("--label", default="variant")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    b, bd = per_line(pathlib.Path(args.base))
    v, vd = per_line(pathlib.Path(args.variant))
    lines = sorted(set(b) & set(v))
    if not lines:
        raise SystemExit("no common lines between the two witnesses")

    print(f"base    = {args.base}  a={bd['a']:.4f} ({bd['totals']['accepted']}/{bd['totals']['rounds']})")
    print(f"variant = {args.variant}  a={vd['a']:.4f} ({vd['totals']['accepted']}/{vd['totals']['rounds']})"
          f"  [{args.label}]")
    only = sorted((set(b) | set(v)) - set(lines))
    if only:
        print(f"NOTE: lines present in only one witness, excluded: {only}")

    print(f"\n{'line':>5} {'rounds b/v':>13} {'accept b/v':>13} {'rate b':>8} {'rate v':>8} {'delta':>8}")
    deltas, weights = [], []
    for ln in lines:
        rb, rv = b[ln]["rounds"], v[ln]["rounds"]
        ab, av = b[ln]["accepted"], v[ln]["accepted"]
        if rb == 0 or rv == 0:
            print(f"{ln:>5}  no rounds in one arm — not adjudicable, excluded")
            continue
        xb, xv = ab / rb, av / rv
        deltas.append(xv - xb)
        weights.append(min(rb, rv))
        print(f"{ln:>5} {rb:>6}/{rv:<6} {ab:>6}/{av:<6} {xb:>8.3f} {xv:>8.3f} {xv - xb:>+8.3f}")

    n = len(deltas)
    up = sum(1 for d in deltas if d > 0)
    down = sum(1 for d in deltas if d < 0)
    flat = n - up - down
    pooled_b = bd["totals"]["accepted"] / bd["totals"]["rounds"]
    pooled_v = vd["totals"]["accepted"] / vd["totals"]["rounds"]
    mean_delta = sum(deltas) / n
    # Weighted mean: a 5-round line and a 40-round line should not count the same.
    wsum = sum(weights)
    wmean = sum(d * w for d, w in zip(deltas, weights)) / wsum

    # Sign-flip permutation on the per-line deltas: the null is "the variant moves each line's rate by 0",
    # so flipping a line's delta sign is exactly as likely as keeping it.
    rng = random.Random(args.seed)
    obs = abs(wmean)
    rng_b = 0
    trials = 20000
    for _ in range(trials):
        s = sum((d if rng.random() < 0.5 else -d) * w for d, w in zip(deltas, weights)) / wsum
        if abs(s) >= obs - 1e-12:
            rng_b += 1
    p = (rng_b + 1) / (trials + 1)

    print(f"\nlines paired: {n}   up {up}   down {down}   flat {flat}")
    print(f"pooled rate:   {pooled_b:.4f} -> {pooled_v:.4f}   delta {pooled_v - pooled_b:+.4f}")
    print(f"mean per-line: {mean_delta:+.4f}    round-weighted mean: {wmean:+.4f}")
    print(f"sign test:     {up} up vs {down} down (flat {flat})")
    print(f"sign-flip permutation on the weighted per-line deltas: p = {p:.4f} ({trials} trials, seed {args.seed})")
    print(f"\nREADING: a delta this size with p > 0.05 is a direction, not a measurement — the populations")
    print(f"         here are {bd['totals']['rounds']} and {vd['totals']['rounds']} rounds, and the pair's")
    print(f"         binomial half-width is +-1.96*sqrt(a(1-a)(1/n1+1/n2)) = "
          f"+-{1.96 * ((pooled_v * (1 - pooled_v)) * (1 / bd['totals']['rounds'] + 1 / vd['totals']['rounds'])) ** 0.5:.4f}.")
    print(f"         If the sign test and the per-line direction agree across two independent populations,")
    print(f"         the honest statement is 'consistently positive, not yet sized'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
