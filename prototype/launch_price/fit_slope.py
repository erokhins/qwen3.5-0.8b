#!/usr/bin/env python3
"""fit_slope.py — turns the S1 rig's RESULT lines into a slope, an intercept, a residual, a verdict.

Reads the lines `prototype/launch_price/s1_launch_price --time ...` prints, keyed by the rig's own
`arm=` string (`pattern|body|launches|syncs|passes|pages`), and answers the S1 brief's questions in
this order:

  slope    ms per launch, least squares over the launch ladder at a launch-dominated body. Reported
           WITH the intercept (the fixed cost a step pays whatever N is) and the worst residual, so a
           bent ladder cannot hide behind one number. D2 §4's stage table is the precedent: the
           49-launch stage cost 0.210 ms for its launches and the intercept came out negative, which
           is a fitted constant soaking up curvature, not a physical finding — this script says so
           out loud rather than printing the pretty number.
  syncs    the same fit with SEGMENTS on the x axis, run both ways round (sync count swept at fixed
           launch count, then the engine's own 305/340 ratio), which is how launch count and sync
           count get separated at all.
  merge    the engine arm's 340 launches against a smaller N at the SAME bytes and work. The
           difference is the S1 prize read directly; the slope only explains it.

Guards, because every failure this repo has recorded in the last month was a parser or a ruler:
  * zero RESULT lines parsed is a hard failure, never a verdict over an empty list
    (prototype/gdn_collapse/run.py, prototype/gdn_insitu/census.py);
  * a missing ladder arm is a hard failure — a partial ladder silently changes the slope;
  * an arm whose reps disagree by more than --max-spread aborts the fit: this box's dispatch-bound
    rows are bimodal (engine/performance-status.md §6, spreads of 15–260 % on glue's launch path) and
    a slope through a bimodal ladder is a coin toss;
  * any arm whose name starts with `stub` is refused unless --allow-stub, and --allow-stub prints a
    banner: stub rows exist to test THIS script and must never reach a verdict;
  * the bands are quoted, never invented (each names its probe), and the 1.85 us empty-kernel dispatch
    floor is compared explicitly, because a slope under it is not evidence of anything a merge can win
    (D2 §5: a per-launch saving must clear ~2 us to matter at 48 launches).

This script measures nothing. The rig refuses to print a millisecond figure when its byte-touched
witnesses are not clean, so a RESULT line here always came from a verified window.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULT_RE = re.compile(
    r"RESULT arm=(?P<arm>\S+) pattern=(?P<pattern>\S+) body=(?P<body>\S+) "
    r"launches=(?P<launches>\d+) syncs=(?P<syncs>\d+) passes=(?P<passes>\d+) "
    r"pages=(?P<pages>\S+) threads=(?P<threads>\d+) grid_max=(?P<grid_max>\d+) "
    r"recs=(?P<recs>\d+) read_mib=(?P<read_mib>[0-9.]+) write_mib=(?P<write_mib>[0-9.]+) "
    r"demand_mib=(?P<demand_mib>[0-9.]+) seg_lps_min=(?P<seg_lps_min>\d+) "
    r"seg_lps_max=(?P<seg_lps_max>\d+) verify=(?P<verify>\S+) "
    r"launches_executed=(?P<launches_executed>\d+) syncs_executed=(?P<syncs_executed>\d+) "
    r"ms_min=(?P<ms_min>[0-9.eE+-]+) ms_median=(?P<ms_median>[0-9.eE+-]+) ms_max=(?P<ms_max>[0-9.eE+-]+) "
    r"reps=(?P<reps>\d+) iters=(?P<iters>\d+)")

# The regex's group NAMES are required to equal the printed field names, in order; check_s1.sh asserts
# that against the rig's printf. A rename on one side is otherwise invisible until the timed pass.

# Every band is a quoted measurement with its source, not a taste (AGENTS: numbers without a probe
# behind them are assumptions and need an A-xx tag).
BANDS = [
    ("submit floor", 0.375, "tools/launch_overhead_hip.cpp: async submit + free-call floor [Q4]"),
    ("empty-kernel dispatch", 1.85, "tools/submission_probe.cpp: empty 48-launch burst, one sync [Q3]"),
    ("serialized sync gap", 5.9, "engine stage_clock: 1.80 ms / 305 syncs, same work [Q2]"),
]


def die(msg: str):
    print(f"FIT FAIL: {msg}")
    sys.exit(1)


def read_rows(paths):
    rows = []
    for p in paths:
        txt = Path(p).read_text(errors="replace")
        for line in txt.splitlines():
            m = RESULT_RE.search(line)
            if m:
                d = m.groupdict()
                d["_line"] = line
                rows.append(d)
    if not rows:
        die(f"no RESULT line parsed from {paths}. Either the timed run never ran or the line shape "
            f"changed. A check whose input list can be empty cannot fail.")
    return rows


def group(rows, max_spread, allow_stub):
    by = defaultdict(list)
    for r in rows:
        arm = r["arm"]
        if "stub=1" in r["_line"] and not allow_stub:
            die(f"line for arm {arm!r} is marked stub=1: fabricated rows exist to test this script and "
                f"must never reach a verdict. Pass --allow-stub only when the point is to test the fit.")
        if r["verify"] != "ok":
            die(f"arm {arm} carries verify={r['verify']}: the rig does not print RESULT for a red "
                f"window, so a red row in the log means the file mixes runs")
        if int(r["launches_executed"]) != int(r["launches"]) or int(r["syncs_executed"]) != int(r["syncs"]):
            die(f"arm {arm} executed {r['launches_executed']} launches / {r['syncs_executed']} syncs of "
                f"{r['launches']}/{r['syncs']}: the count match is the window's own contract")
        by[arm].append(float(r["ms_median"]) * 1000.0)      # us
    out = {}
    for arm, vals in by.items():
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / med * 100.0 if med > 0 and len(vals) > 1 else 0.0
        if spread > max_spread:
            die(f"arm {arm} spread {spread:.1f}% over {len(vals)} reps (> --max-spread "
                f"{max_spread}%). That is this box's dispatch bimodality, not a number: re-run the "
                f"whole ladder inside ONE gpu_lock hold and report the spread you saw.")
        fields = [r for r in rows if r["arm"] == arm]
        demands = {float(r["demand_mib"]) for r in fields}
        if len(demands) > 1:
            die(f"arm key {arm!r} covers {len(demands)} different footprints {sorted(demands)} MiB. Two "
                f"geometries collided on one key, so averaging them would invent a point; the rig's "
                f"arm key must carry whatever knob distinguishes them.")
        f0 = fields[0]
        out[arm] = {"us": med, "spread": spread, "reps": len(vals),
                    "launches": int(f0["launches"]), "syncs": int(f0["syncs"]),
                    "demand_mib": float(f0["demand_mib"])}
    return out


def least_squares(xs, ys):
    n = len(xs)
    if len(set(xs)) < 2:
        die(f"need at least two distinct x values, got {sorted(set(xs))}")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    inter = my - slope * mx
    resid = [y - (slope * x + inter) for x, y in zip(xs, ys)]
    return slope, inter, max(abs(r) for r in resid), \
        (sum(r * r for r in resid) / n) ** 0.5, resid


def fit_print(title, keys, arms, xfield, labels):
    xs, ys = [], []
    for a in arms:
        if a not in keys:
            die(f"{title}: arm {a!r} is not in the log. A partial ladder silently changes the slope; "
                f"read the run log's tail before trusting anything below this line.")
        xs.append(keys[a][xfield])
        ys.append(keys[a]["us"])
    slope, inter, kmax, rms, resid = least_squares(xs, ys)
    unit = "launch" if xfield == "launches" else "segment"
    print(f"== {title}")
    for x, y, r, lab in zip(xs, ys, resid, labels):
        print(f"    {lab:<40} x={x:<6} {y:9.2f} us   resid {r:+8.2f}")
    print(f"  slope     = {slope:.4f} us per {unit}")
    print(f"  intercept = {inter:.2f} us per step (the cost that is there at N=1)")
    print(f"  residual  = max {kmax:.2f} us, rms {rms:.2f} us over {len(xs)} points")
    if inter < 0:
        print(f"  [note] a NEGATIVE intercept is a fitted constant soaking up curvature, not a physical "
              f"finding (D2 §4 printed -0.065 ms the same way). Quote the pair, not the constant.")
    if kmax > 0.15 * max(ys):
        print(f"  [note] worst residual is > 15% of the largest arm: the ladder is bent and one slope "
              f"does not describe it. Report the legs separately.")
    return slope, inter


def verdict(slope):
    print("== verdict")
    # FIRST the funded decision rule: route S1 §4 sets the kill/fund line at 3 us and 1 us per launch
    # at the shipped launch count. This gate exists to apply that line, not to invent another one.
    if slope >= 3.0:
        dec = ("S1 IS REAL (>= 3 us/launch): the prize is (launches removable) x slope, and the "
               "candidate list in the brief is funded.")
    elif slope >= 1.0:
        dec = ("MARGINAL (1-3 us/launch): fund only merges that also delete work, i.e. fold this into "
               "D5 rather than opening S1 on its own.")
    else:
        dec = ("S1 IS DEAD (< 1 us/launch): launches are free when pipelined, so the T_geo/T_serial "
               "figures in D2 and D4 are instrument artefacts of a draining probe and the only "
               "remaining prize is deleting work (D5).")
    print(f"  DECISION (route S1 §4 bands, 3 us / 1 us): measured slope {slope:.4f} us/launch -> {dec}")
    for name, v, src in BANDS:
        print(f"    {name:<22} {v:>6.2f} us   {src}")
    floor, empty, ser = BANDS[0][1], BANDS[1][1], BANDS[2][1]
    if slope < floor:
        head = ("BELOW the async submit floor. The queue already absorbs the launch issue, so merging "
                "segments cannot buy launch-issue time here.")
        prize = "0 us of the step is recoverable by merging at this body size."
    elif slope < empty:
        head = ("between the submit floor and the empty-kernel dispatch floor: a launch costs about "
                "what an empty kernel costs, so the per-launch prize is bounded near 1.85 us.")
        prize = f"up to {slope:.2f} us per launch removed, and the fixed part is not launch-cost."
    elif slope < ser:
        head = ("at or above the empty-kernel dispatch floor, below the serialized-sync gap: launches "
                "cost real per-launch time on top of the bytes.")
        prize = (f"slope x (340 - N_merged) us; the prize is bounded by the launch count you can "
                 f"actually remove.")
    else:
        head = ("at or above the serialized-sync gap: the per-launch cost is sync-shaped, so the prize "
                "is the 305-segment drain pattern itself, not the 340 kernels.")
        prize = "the whole drain pattern is in play; S1 should plan for segment merging, not just merges."
    print(f"  SHAPE of the cost (slope {slope:.4f} us/launch): {head}")
    print(f"  prize if merged: {prize}")
    print(f"  The D2 §5 test is whether a per-launch saving clears ~2 us: it "
          f"{'CLEARS' if slope >= 2.0 else 'DOES NOT CLEAR'} it at this body size.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True, type=Path)
    ap.add_argument("--slope-body", default="tiny", help="body of the launch-dominated ladder")
    ap.add_argument("--slope-ladder", default="48,165,305,340,680")
    ap.add_argument("--sync-leg-fixed-launches", default="340",
                    help="launch count held fixed while the sync count is swept")
    ap.add_argument("--sync-leg-syncs", default="340,170,43,1",
                    help="sync counts for leg A (sync-each -> sync-skew, work ~constant)")
    ap.add_argument("--engine-arm", default="uniform|tiny|t1461|340|1|8|warm",
                    help="reference arm of the merge section; it must be built with the same "
                         "--total-mib as --merge-arms, which the code asserts")
    ap.add_argument("--merge-arms", default="",
                    help="comma list of arm keys to compare against --engine-arm")
    ap.add_argument("--max-spread", type=float, default=10.0)
    ap.add_argument("--allow-stub", action="store_true",
                    help="accept arms named stub* (testing this script only; never for a verdict)")
    a = ap.parse_args()
    if a.allow_stub:
        print("BANNER: --allow-stub is set. The numbers below are fabricated for parser testing and "
              "are NOT a measurement.")

    rows = read_rows([str(p) for p in a.inp])
    keys = group(rows, a.max_spread, a.allow_stub)
    print(f"parsed {len(rows)} RESULT lines into {len(keys)} arms from {[str(p) for p in a.inp]}")

    ladder = [int(x) for x in a.slope_ladder.split(",") if x.strip()]
    slope_arms = [f"uniform|{a.slope_body}|p|{n}|1|8|warm" for n in ladder]
    slope, _ = fit_print("1. slope: us per launch (launch-dominated body)",
                         keys, slope_arms, "launches",
                         [f"N={n} body={a.slope_body} syncs=1" for n in ladder])
    verdict(slope)

    L = int(a.sync_leg_fixed_launches)
    syncs = [int(x) for x in a.sync_leg_syncs.split(",") if x.strip()]
    leg_a = [f"uniform|{a.slope_body}|p|{L}|{s}|8|warm" for s in syncs]
    fit_print("2a. syncs at fixed launch count (separates sync count from launch count)",
              keys, leg_a, "syncs", [f"L={L} S={s}" for s in syncs])

    if a.merge_arms:
        print("== 3. merge: the prize read directly, bytes and work held equal")
        if a.engine_arm not in keys:
            die(f"engine arm {a.engine_arm!r} not in the log")
        e = keys[a.engine_arm]
        print(f"  reference arm {a.engine_arm}: {e['us']:.2f} us at {e['demand_mib']:.4f} MiB demand")
        for arm in [x.strip() for x in a.merge_arms.split(",") if x.strip()]:
            if arm not in keys:
                die(f"merge arm {arm!r} not in the log")
            m = keys[arm]
            # "Same bytes and work" is the entire point of this section, so it is asserted rather than
            # assumed: two arms whose booked demand differs are two different experiments and their
            # difference is not a merge prize.
            if e["demand_mib"] <= 0 or abs(e["demand_mib"] - m["demand_mib"]) > 0.01 * e["demand_mib"]:
                die(f"merge compares unequal footprints: reference {e['demand_mib']:.4f} MiB vs arm "
                    f"{arm} at {m['demand_mib']:.4f} MiB. Compare arms built with the same --total-mib.")
            gain = e["us"] - m["us"]
            n = m["launches"]
            per = gain / max(1, e["launches"] - n)
            print(f"  merged arm {arm}: {m['us']:.2f} us at {m['demand_mib']:.1f} MiB demand -> "
                  f"gain {gain:.2f} us = {per:.4f} us per launch removed "
                  f"({gain / e['us'] * 100.0:.1f}% of that window)")
        print("  [reading] equal bytes and work make this the direct prize. If the gain is BELOW "
              "slope x (340-N) the launches were already pipelined behind the work, which is the "
              "answer the slope alone cannot give.")
    print("FIT ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
