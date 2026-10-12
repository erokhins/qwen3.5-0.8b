"""Worst per-step logits margin per fixture, from two gate logs — the §6.0 margin preflight.

Each step's line pair is `[PASS] step N: cos deficit D inside the route allowance C` and
`[PASS] step N: meanrel M% inside the route allowance R%` (`engine/gate.py` section (c)), so D/C and
M/R are how full each step's allowance is. The binding step is the largest ratio, and a package that
moves bits at all moves those numbers; x1.0000 means this one did not. A fixture with nothing priced
(`dec8`) falls back to the flat G5 floors, and says so in its step column rather than pretending the
floor was measured.

Made for P5's before/after table in `engine/results.md`, and general: any two sets of gate logs.
Usage: tools/margin_delta.py <before-log-prefix> <after-log-prefix> <fixture>...
where a log is named `<prefix>_<fixture>.log`, e.g. /tmp/p5_before_dec300f.log.
"""
import re
import sys


def worst(path):
    """(worst cos ratio, its step, its deficit and cap, worst meanrel ratio and step, steps priced).

    Two rule forms, in priority order: a priced `inside the route allowance C` line, or — for a
    fixture with nothing priced, like dec8 — the metrics line's deficit against the flat G5 floors
    (cos 5e-04, meanrel 5 %), labelled as such so a flat floor is never read as a measured one.
    """
    cos, rel, n = None, None, 0
    flat = None  # (step, deficit, meanrel%) from the metrics line, kept for the unpriced form
    with open(path, errors="replace") as fh:
        for line in fh:
            match = re.search(r"step (\d+): metrics — cos [0-9.]+ deficit ([0-9.eE+-]+)"
                              r" meanrel ([0-9.]+)%", line)
            if match:
                flat = (match.group(1), float(match.group(2)), float(match.group(3)))
                n += 1
                continue
            match = re.search(r"step (\d+): cos deficit ([0-9.eE+-]+) inside the route allowance"
                              r" ([0-9.eE+-]+)", line)
            if match:
                step, deficit, cap = match.group(1), float(match.group(2)), float(match.group(3))
                ratio = deficit / cap
                if cos is None or ratio > cos[0]:
                    cos = (ratio, step, deficit, cap)
                continue
            match = re.search(r"step (\d+): meanrel ([0-9.]+)% inside the route allowance"
                              r" ([0-9.]+)%", line)
            if match:
                step, got, cap = match.group(1), float(match.group(2)), float(match.group(3))
                ratio = got / cap
                if rel is None or ratio > rel[0]:
                    rel = (ratio, step, got, cap)
    if cos is None and flat is not None:
        cos = (flat[1] / 5e-04, flat[0] + " (flat G5)", flat[1], 5e-04)
        rel = (flat[2] / 5.0, flat[0] + " (flat G5)", flat[2], 5.0)
    return cos, rel, n


for fixture in sys.argv[3:]:
    (bc, br, nb) = worst("%s_%s.log" % (sys.argv[1], fixture))
    (ac, ar, na) = worst("%s_%s.log" % (sys.argv[2], fixture))
    if not bc or not ac:
        print("%-8s missing or allowance-less log (%s / %s)" % (fixture, nb, na))
        continue
    print("%-8s steps %d->%d" % (fixture, nb, na))
    print("         cos      worst step %s -> %s : deficit %.4e of %.4e -> %.4e of %.4e"
          "  | filled %.4f -> %.4f (x%.4f)"
          % (bc[1], ac[1], bc[2], bc[3], ac[2], ac[3], bc[0], ac[0], ac[0] / bc[0]))
    print("         meanrel worst step %s -> %s : %.4f%% of %.4f%% -> %.4f%% of %.4f%%"
          "  | filled %.4f -> %.4f (x%.4f)"
          % (br[1], ar[1], br[2], br[3], ar[2], ar[3], br[0], ar[0], ar[0] / br[0]))
