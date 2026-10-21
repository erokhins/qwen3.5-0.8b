#!/usr/bin/env python3
"""fit_shape.py — the only reader of wave_shape's timed output that may print a verdict.

Why this file exists at all: S1 gate zero quoted 13.08 us/launch from a comparison of 340 launches at
grid_max=271 against 48 launches at grid_max=1918, so launch count and workgroups-per-launch moved
together and the number described neither. The rig therefore emits, for every cell, a witness line
(CASE ...) and a timing line (RESULT ...) that carry the geometry as fields, and this fitter refuses
to turn them into a claim unless the input can carry the claim at all.

REFUSALS (each prints `REFUSE <reason>` and exits 2):
  empty        — zero RESULT lines. A parser that matches nothing must not print a fit (AGENTS:
                 `prototype/gdn_collapse/run.py`; a check whose input list can be empty cannot fail).
  partial      — a CASE line without its RESULT, a RESULT without its CASE, a NO-TIMING row, a row
                 whose verify is not ok, or a missing cell from the expected set.
  count        — parsed record count != records printed by the rig (counted from the rig's own
                 `CELLS-PRINTED` / `DOWNS-PRINTED` trailer lines).
  bimodal      — the same arm's ms_min spread across passes exceeds --max-spread. Bimodality here is a
                 property of the window, not of the shape, so the whole matrix is thrown out rather
                 than averaged: launch-bound rows on this box are bimodal (engine/performance-status.md
                 §6), and a bimodal mean is worse than no number.
  footprint    — inside one family, not every row carries the same read_mib/write_mib. The rig's whole
                 contract is "shape varies, bytes do not"; a drifted row makes every delta a traffic
                 delta. Matrix rows and down rows are DIFFERENT families and are never compared as
                 equal-footprint items.

Pre-registered bands (written before any timing was taken; see ITEM2-HANDOVER.md §4):
  WIN   >= 0.30 ms per 4,032-lane step from a shape change at equal bytes AND equal launch count
  KILL  the iso-population diagonal is flat to within the measured noise floor AND split-K buys < 0.15 ms
Anything else is INCONCLUSIVE, which is a result, not a failure.

No timing is taken here; this only reads numbers the rig printed.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

FIELD_RE = re.compile(r"([a-z_0-9]+)=(\S+)")
# Quoted ledger figures. Each is labelled [Qn] and is NOT measured here.
QUOTED = {
    "Q1_us_per_launch": 3.10,      # cast-fold deletion, prototype/launch_price/S1-GATEZERO.md
    "Q3_step_ms_4032": 10.9357,    # engine/performance-status.md §1 headline decode step
    "Q4_down_rows": 1024,          # engine/decode_step.cpp: kHidden rows over 128 blocks
}
WIN_MS = 0.30
KILL_SPLITK_MS = 0.15


def die(msg):
    print("REFUSE " + msg)
    sys.exit(2)


def parse(paths):
    """Returns (rows, trailers). rows = list of dict, one per cell-with-timing."""
    rows, trailers, problems = [], {}, []
    pending = None            # a CASE line waiting for its RESULT
    n_case = n_result = n_notiming = 0
    for path in paths:
        with open(path, errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("CASE "):
                    if pending is not None:
                        problems.append("CASE-without-RESULT name=%s" % pending["name"])
                    n_case += 1
                    pending = {"name": line.split()[1], "case_file": path}
                    continue
                if line.startswith("RESULT "):
                    n_result += 1
                    d = dict(FIELD_RE.findall(line))
                    if pending is None:
                        problems.append("RESULT-without-CASE arm=%s" % d.get("arm", "?"))
                    else:
                        d["name"] = pending["name"]
                        pending = None
                    if d.get("verify") != "ok":
                        problems.append("verify_not_ok arm=%s" % d.get("arm", "?"))
                        continue
                    if "ms_min" not in d:
                        problems.append("result_without_ms arm=%s" % d.get("arm", "?"))
                        continue
                    d["ms_min"] = float(d["ms_min"])
                    d["ms_median"] = float(d.get("ms_median", "nan"))
                    d["ms_max"] = float(d.get("ms_max", "nan"))
                    for k in ("read_mib", "write_mib", "demand_mib", "per_launch_us", "wavefronts_per_cu"):
                        d[k] = float(d.get(k, "nan"))
                    for k in ("launches", "grid", "blocks_total", "wavefronts_total", "slices", "reps", "iters"):
                        try:
                            d[k] = int(float(d[k]))
                        except (KeyError, ValueError):
                            d[k] = None
                    # `pass=` is stamped by sweep.sh; a single-pass file is pass 1.
                    if "pass" not in d:
                        pt = re.search(r"\bpass=(\d+)", line)
                        d["pass"] = int(pt.group(1)) if pt else 1
                    rows.append(d)
                    continue
                if line.startswith("NO-TIMING "):
                    n_notiming += 1
                    d = dict(FIELD_RE.findall(line))
                    problems.append("NO-TIMING arm=%s reason=%s" % (d.get("arm", "?"), d.get("reason", "?")))
                    pending = None
                    continue
                if line.startswith("CELLS-PRINTED") or line.startswith("DOWNS-PRINTED"):
                    key, _, val = line.partition("=")
                    trailers[key.strip()] = int(val.split()[0])
    if pending is not None:
        problems.append("CASE-without-RESULT name=%s" % pending["name"])
    return rows, trailers, problems, (n_case, n_result, n_notiming)


def group_by(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def spread(vals):
    lo, hi = min(vals), max(vals)
    return hi - lo, (0.0 if lo <= 0 else (hi - lo) / lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="files printed by sweep.sh (grep -h RESULT friendly)")
    ap.add_argument("--expect", type=int, default=0,
                    help="required number of DISTINCT timed cells; 0 = accept whatever set arrived")
    ap.add_argument("--max-spread", type=float, default=0.05,
                    help="max relative ms_min spread for one arm across passes before the input is "
                         "called bimodal (default 0.05)")
    ap.add_argument("--json", default="", help="write the whole fit here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows, trailers, problems, counts = parse(args.inputs)
    n_case, n_result, n_notiming = counts

    # ---- refusal gauntlet, in the order a reader would ask about it -------------------------
    if n_result == 0 or not rows:
        die("empty: zero RESULT lines in %d file(s), %d CASE lines seen. Nothing was timed or the "
            "reader matched nothing; either way there is no fit to print." % (len(args.inputs), n_case))
    if problems:
        die("partial/unusable input: %d problem row(s): %s" % (len(problems), "; ".join(problems[:8])))
    # parsed-record-count == records-printed, using the rig's own trailers when present
    for key, want in sorted(trailers.items()):
        fam = {"CELLS-PRINTED": "mat", "DOWNS-PRINTED": "down"}.get(key)
        if fam is None:
            continue
        got = sum(1 for r in rows if r.get("family") == fam)
        if got != want:
            die("count: %s says the rig printed %d but this reader parsed %d" % (key, want, got))
    if n_case != n_result:
        die("count: %d CASE lines vs %d RESULT lines" % (n_case, n_result))

    fams = set(r.get("family") for r in rows)
    for fam in sorted(fams):
        sub = [r for r in rows if r.get("family") == fam]
        reads = set("%.4f" % r["read_mib"] for r in sub)
        writes = set("%.4f" % r["write_mib"] for r in sub)
        if fam == "mat" and (len(reads) != 1 or len(writes) != 1):
            die("footprint: family=mat rows carry %d different read_mib and %d different write_mib "
                "values; every matrix cell must touch identical bytes or no delta is a shape delta"
                % (len(reads), len(writes)))
        if fam == "down":
            wset = set("%.3f" % float(r.get("weight_mib", "nan")) for r in sub)
            if len(wset) != 1:
                die("footprint: family=down rows carry %d different weight_mib values" % len(wset))

    # per-arm spread across passes (also the bimodality guard and the noise floor)
    by_name = group_by(rows, lambda r: r["arm"])
    worst_arm, worst_spread, worst_vals = None, 0.0, []
    for arm, rs in by_name.items():
        if len(rs) < 2:
            continue
        sp, rel = spread([r["ms_min"] for r in rs])
        if rel > worst_spread:
            worst_arm, worst_spread, worst_vals = arm, rel, [r["ms_min"] for r in rs]
    noise_ms = 0.0
    for arm, rs in by_name.items():
        if len(rs) >= 2:
            noise_ms = max(noise_ms, max(r["ms_min"] for r in rs) - min(r["ms_min"] for r in rs))
    if worst_spread > args.max_spread:
        die("bimodal: arm=%s repeats disagree by %.2f%% (%s) > --max-spread %.2f%%. A launch-bound row "
            "on this box is bimodal across windows; re-take the whole matrix inside one lock hold "
            "rather than fitting a mixture." % (worst_arm, 100 * worst_spread,
                                               " ".join("%.4f" % v for v in worst_vals),
                                               100 * args.max_spread))
    if noise_ms == 0.0:
        die("noise: no arm was measured twice, so there is no noise floor and no flatness claim is "
            "adjudicable. sweep.sh must run at least two passes.")

    best = {}
    for arm, rs in by_name.items():
        best[arm] = min(rs, key=lambda r: r["ms_min"])
    if args.expect and len(best) != args.expect:
        die("partial: expected %d distinct timed cells, parsed %d" % (args.expect, len(best)))

    mat = [r for r in best.values() if r.get("family") == "mat"]
    down = [r for r in best.values() if r.get("family") == "down"]
    if not mat:
        die("partial: no matrix cells at all; the (L,G) matrix is the item being priced")
    if not down:
        die("partial: no down_proj cells at all; the split-K arm is the other half of the item")

    # ---- the two one-way tables: the confound is exactly the thing these two must disagree about
    at_fixed_L = group_by(mat, lambda r: r["launches"])
    at_fixed_G = group_by(mat, lambda r: r["grid"])
    iso_pop = group_by(mat, lambda r: r["launches"] * r["grid"])

    def fmt_lg(r):
        return "L=%d G=%d ms=%.4f wf/cu=%.1f" % (r["launches"], r["grid"], r["ms_min"],
                                                 r["wavefronts_per_cu"])

    rep = {"noise_floor_ms": noise_ms, "quoted": QUOTED, "bands": {"win_ms": WIN_MS,
               "kill_splitk_ms": KILL_SPLITK_MS}, "families": {}}

    def out(msg):
        # --quiet drops the tables, never the two lines a reader needs to know what was decided: the
        # noise floor and the VERDICT are printed unconditionally above and below.
        if not args.quiet:
            sys.stdout.write(msg)

    print("== noise floor: %.4f ms (largest within-arm repeat spread over %d arms); quoted step "
          "%.4f ms [Q3]" % (noise_ms, len(by_name), QUOTED["Q3_step_ms_4032"]))

    out("\n== at FIXED launch count L, varying workgroups per launch G  (this is the shape term)\n")
    shape_rows = []
    for L in sorted(at_fixed_L):
        rs = sorted(at_fixed_L[L], key=lambda r: r["grid"])
        if len(rs) < 2:
            continue
        sp, rel = spread([r["ms_min"] for r in rs])
        bestcell = max(rs, key=lambda r: r["ms_min"])
        worstcell = min(rs, key=lambda r: r["ms_min"])
        shape_rows.append({"L": L, "n": len(rs), "spread_ms": sp, "spread_rel": rel,
                           "slowest": fmt_lg(bestcell), "fastest": fmt_lg(worstcell)})
        out("  L=%-5d n=%-3d spread %8.4f ms (%6.2f %%)  slowest %s  fastest %s\n"
            % (L, len(rs), sp, 100 * rel, fmt_lg(bestcell), fmt_lg(worstcell)))
    if not shape_rows:
        die("partial: no L has two different G values, so the shape term is unmeasured. That is exactly "
            "the S1 gate-zero confound; add cells.")
    shape_best = max(shape_rows, key=lambda r: r["spread_ms"])

    out("\n== at FIXED workgroups per launch G, varying launch count L  (the count term, fixed bytes)\n")
    for G in sorted(at_fixed_G):
        rs = sorted(at_fixed_G[G], key=lambda r: r["launches"])
        if len(rs) < 2:
            continue
        sp, rel = spread([r["ms_min"] for r in rs])
        out("  G=%-5d n=%-3d spread %8.4f ms (%6.2f %%)  slowest %s  fastest %s\n"
            % (G, len(rs), sp, 100 * rel,
               fmt_lg(max(rs, key=lambda r: r["ms_min"])), fmt_lg(min(rs, key=lambda r: r["ms_min"]))))

    out("\n== iso-population diagonal (constant L*G = workgroups in the step, bytes constant)\n")
    diag_rows = []
    for pop in sorted(iso_pop):
        rs = sorted(iso_pop[pop], key=lambda r: r["launches"])
        if len(rs) < 2:
            continue
        sp, rel = spread([r["ms_min"] for r in rs])
        flat = sp <= noise_ms
        diag_rows.append({"pop": pop, "n": len(rs), "spread_ms": sp, "spread_rel": rel,
                          "flat_within_noise": flat,
                          "cells": [fmt_lg(r) for r in rs]})
        out("  pop=%-6d n=%-3d spread %8.4f ms (%6.2f %%)  flat_within_noise=%d  %s\n"
            % (pop, len(rs), sp, 100 * rel, flat, " | ".join(fmt_lg(r) for r in rs)))
    if not diag_rows:
        die("partial: no population has two different (L,G) decompositions, so the iso-population "
            "diagonal does not exist in this input")
    diag_flat = all(d["flat_within_noise"] for d in diag_rows)
    diag_worst = max(diag_rows, key=lambda d: d["spread_ms"])

    out("\n== down_proj (one-row): shipped 1-launch form against split-K\n")
    ship = [r for r in down if r.get("arm", "").startswith("down|ship|f32")]
    if not ship:
        die("partial: no down|ship|f32 row to price against")
    ship_ms = min(r["ms_min"] for r in ship)
    tax_note = ("two-pass costs +1 launch (the split rides gridDim.y, the reduce is one more launch); "
                "at the quoted %s us/launch [Q1] that is %.4f ms of tax per down_proj call"
                % (QUOTED["Q1_us_per_launch"], QUOTED["Q1_us_per_launch"] / 1000.0))
    out("  ship ms=%.4f (grid 128x1, %d warps over %s CUs [Q4/Q5])\n"
        % (ship_ms, 1024, "40"))
    splitk_best = 0.0
    sk_rows = []
    for r in sorted(down, key=lambda r: r["arm"]):
        if r["arm"].startswith("down|ship"):
            continue
        gain = ship_ms - r["ms_min"]
        splitk_best = max(splitk_best, gain)
        sk_rows.append({"arm": r["arm"], "ms": r["ms_min"], "gain_ms": gain,
                        "launches": r["launches"], "blocks_total": r["blocks_total"],
                        "wavefronts_per_cu": r["wavefronts_per_cu"],
                        "vs_ship_ndiff": r.get("vs_ship_ndiff"), "max_ulp": r.get("max_ulp")})
        out("  %-16s ms=%.4f gain=%+8.4f ms launches=%-3s wf/cu=%-6.1f stage_kib=%-7s part_kib=%-6s "
            "vs_ship_ndiff=%s max_ulp=%s\n"
            % (r["arm"], r["ms_min"], gain, r["launches"], r["wavefronts_per_cu"],
               r.get("stage_kib"), r.get("part_kib"), r.get("vs_ship_ndiff"), r.get("max_ulp")))
    if not sk_rows:
        die("partial: no split-K down rows to compare against the shipped form")
    out("  %s\n" % tax_note)
    out("  FOOTPRINT NOTE: in the down family the bytes that must be equal are the WEIGHT and the "
        "OUTPUT (7.0000 MiB + 2 KiB for one token's down_proj). stage_kib is deliberately NOT equal "
        "across arms: it is the activation row re-staged by every block, 1.75 MiB for the shipped 128 "
        "blocks and splitK x that for a split shape, which is a property of the shape being priced and "
        "not a drifted fixture. Read it beside every gain before quoting the gain.\n")

    # ---- verdict against the pre-registered bands -------------------------------------------
    shape_gain = shape_best["spread_ms"]
    net_splitk = splitk_best
    verdict, why = "INCONCLUSIVE", []
    if noise_ms >= WIN_MS:
        verdict = "ADJUDICATION-FAILED"
        why.append("the noise floor alone (%.4f ms) is at or above the WIN band (%.2f ms); no shape "
                   "claim is adjudicable in this window" % (noise_ms, WIN_MS))
    else:
        if shape_gain >= WIN_MS:
            verdict = "WIN"
            why.append("shape spread at equal bytes and equal launch count is %.4f ms at %s "
                       ">= the pre-registered %.2f ms" % (shape_gain, "L=%d" % shape_best["L"], WIN_MS))
        elif diag_flat and net_splitk < KILL_SPLITK_MS:
            verdict = "KILL"
            why.append("every iso-population diagonal is flat within noise (worst %.4f ms vs floor "
                       "%.4f ms) AND the best split-K buys %.4f ms < %.2f ms"
                       % (diag_worst["spread_ms"], noise_ms, net_splitk, KILL_SPLITK_MS))
        else:
            verdict = "INCONCLUSIVE"
            why.append("shape spread %.4f ms < %.2f ms, so no WIN; but %s, so no KILL either"
                       % (shape_gain, WIN_MS,
                          ("split-K buys %.4f ms >= %.2f ms" % (net_splitk, KILL_SPLITK_MS))
                          if net_splitk >= KILL_SPLITK_MS else
                          ("the worst iso-population diagonal (%.4f ms at pop=%d) is NOT flat within "
                           "noise" % (diag_worst["spread_ms"], diag_worst["pop"]))))
    out("\n== VERDICT %s\n  %s\n" % (verdict, " ".join(why)))
    print("VERDICT %s %s" % (verdict, " ".join(why)))
    print("inputs=%d rows=%d distinct_cells=%d noise=%.4f ms shape_gain=%.4f ms splitk_gain=%.4f ms"
          % (len(args.inputs), len(rows), len(best), noise_ms, shape_gain, splitk_best))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"noise_floor_ms": noise_ms, "verdict": verdict, "why": " ".join(why),
                       "quoted": QUOTED, "bands": {"win_ms": WIN_MS, "kill_splitk_ms": KILL_SPLITK_MS},
                       "shape_at_fixed_L": shape_rows, "iso_population": diag_rows,
                       "splitk": sk_rows, "ship_ms": ship_ms,
                       "cells": {k: {f: v for f, v in r.items() if f != "name"}
                                 for k, r in best.items()}}, fh, indent=1, sort_keys=True)
        out("wrote %s\n" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
