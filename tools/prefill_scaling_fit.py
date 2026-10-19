#!/usr/bin/env python3
"""prefill_scaling_fit.py — split a prefill throughput curve into its LINEAR and QUADRATIC parts.

WHY. A ratio at one context width says who is faster there. It cannot say WHY the ratio moves with width, and
the operator's question is exactly that: does llama.cpp degrade more slowly than we do, and if so is it a
constant we pay or a term that grows with context? The two answers imply different work. A two-term model
answers it directly, because prefill cost really is the sum of those two shapes:

    ms_per_prompt(W) = a * W + b * W^2          (a: per-token cost, GEMM/GDN/glue
                                                b: per-pair cost, i.e. attention's quadratic term)

Fit a and b per engine from a window produced by tools/llama_phase_compare.sh (one lock hold, both engines,
same widths), then compare a_ours/a_llama and b_ours/b_llama SEPARATELY. If only `a` differs, the gap is a
constant and the ratio shrinks as context grows. If only `b` differs, the gap is asymptotic and grows without
bound. Both differing — which is what §46 measured — means the ratio has a floor set by a-ratio and no ceiling.

Reads files a measurement window already wrote; it takes no device work and times nothing itself.

Usage:
    python3 tools/prefill_scaling_fit.py --out tmp/gcc/llamaphase \
        --widths 2048,4096,8192,16384,32768 [--reps 3] [--row '-ub 512 -fa on -t 32']
"""
import argparse
import os
import re
import statistics
import sys


def med(vals):
    return statistics.median(vals)


def read_ours(out, width, reps):
    """Our side: RESULT ms_total from each rep's log (median)."""
    vals = []
    for r in range(1, reps + 1):
        path = os.path.join(out, f"our_{width}_{r}.log")
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as fh:
            hits = re.findall(r"^RESULT .*?ms_total=([0-9.]+)", fh.read(), re.M)
        if hits:
            vals.append(float(hits[-1]))
    return med(vals) if vals else None


def read_llama(out, width, reps, row):
    """llama.cpp: the '<tok/s> ± spread' cell, NOT the last float (that is the spread)."""
    path = os.path.join(out, f"llama_{row.replace(' ', '_')}_{width}.txt")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        vals = [float(m) for m in re.findall(r"\|\s*pp\d+\s*\|\s*([0-9][0-9.]*) ±", fh.read())]
    if not vals:
        return None
    return med(vals)  # tok/s


def fit_linear_quadratic(points):
    """Least squares for ms_per_token = a + b*W. points = [(W, ms_per_token)]. Returns (a, b, maxresid)."""
    n = len(points)
    if n < 3:
        raise SystemExit(f"REFUSE: {n} points cannot fit two coefficients and say anything; need >= 3 widths.")
    sx = sum(w for w, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(w * w for w, _ in points)
    sxy = sum(w * y for w, y in points)
    den = n * sxx - sx * sx
    if den == 0:
        raise SystemExit("REFUSE: widths are not distinct.")
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    resid = max(abs(y - (a + b * w)) for w, y in points)
    return a, b, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/gcc/llamaphase")
    ap.add_argument("--widths", default="2048,4096,8192,16384,32768")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--row", default="-ub 512 -fa on -t 32")
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",")]
    rows = []
    for w in widths:
        ours_ms = read_ours(args.out, w, args.reps)
        llama_toks = read_llama(args.out, w, args.reps, args.row)
        if ours_ms is None or llama_toks is None:
            print(f"  W={w}: MISSING (ours={ours_ms} llama={llama_toks}) — refusing to fit around a hole")
            continue
        rows.append((w, ours_ms, llama_toks))
    if len(rows) < 3:
        raise SystemExit("REFUSE: fewer than 3 usable widths. A fit through two points is a line, not evidence.")

    print(f"# window {args.out}, {len(rows)} widths, medians of {args.reps}")
    print(f"{'W':>7} {'ours ms':>10} {'ours ms/tok':>12} {'llama tok/s':>12} {'llama ms/tok':>13} {'ratio':>7}")
    ours_pts, llama_pts = [], []
    for w, ms, toks in rows:
        o = ms / w
        l = 1000.0 / toks
        ours_pts.append((w, o))
        llama_pts.append((w, l))
        print(f"{w:>7} {ms:>10.3f} {o:>12.6f} {toks:>12.2f} {l:>13.6f} {l / o:>7.3f}")

    oa, ob, ores = fit_linear_quadratic(ours_pts)
    la, lb, lres = fit_linear_quadratic(llama_pts)
    print("\n# per-token model: ms/token = a + b*W   (a = context-independent, b = quadratic/attention term)")
    print(f"  ours   a={oa:.6f} ms/tok  b={ob:.3e} ms/tok^2   max residual {ores / oa * 100:.2f}% of a")
    print(f"  llama  a={la:.6f} ms/tok  b={lb:.3e} ms/tok^2   max residual {lres / la * 100:.2f}% of a")
    print(f"\n# THE ANSWER, as two separate ratios:")
    print(f"  LINEAR   ours/llama = {oa / la:.3f}x   (a constant we pay on every token, whatever the context)")
    print(f"  QUADRATIC ours/llama = {ob / lb:.3f}x   (the term that grows with context; attention's shape)")
    print(f"  ratio of ratios = {(ob / lb) / (oa / la):.3f}x — how much faster the gap widens than the constant alone allows")
    print("\n# cannot see: which kernel owns `b`. At these widths attention is the only quadratic term this engine")
    print("#   emits, but 'only quadratic term' is an argument about the code, not a measurement of the kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
