#!/usr/bin/env python3
"""meter_digest.py — tabulate `--mtp-meter 1` logs into per-phase ms/round tables.

Why this exists: the engine prints ~20 `MTPMETER` lines per run and a `RESULT` verdict, which is unreadable
across 6 runs (2 routes x 3 widths x 2 roles). This turns them into one table per (route, role) with the
round denominators spelled out, and refuses to print a table for a run whose own verdict was not `ok` —
the same rule `meter_round.awk` enforces, duplicated here so a digest cannot quietly include a refused arm.

Contract it parses (`prototype/mtp_accept/METER-PLAN.md`):
  MTPMETER role=R phase=P ms_total=T rounds=N ms_per_round=M runs=K ms_per_run=U sync_added=S status=X note=…
  RESULT … meter=X rounds=N sum_ms_per_round=S wall_ms_per_round=W coverage=C
Canary lines it reads and reports: `mtp_route=` (the route the run ACTUALLY ran, read back off the log — a
flag is not evidence), `MTPSTAT … rounds=` (must equal the meter's rounds), and `mid_checkpoint status=…`.

Derived columns, both labelled as derived and both dividing by `kRows = 2` because every arm here is a k=1
round: `ms/row` = ms_per_round / 2, `us/launch` = ms_per_run * 1000 for phases whose `runs` counts device
submissions (note field says so), blank where it would be a lie (host-only phases, runs=0).

Line-number-free on purpose: citing this file by path cannot rot, so nothing in the tree cites it by line.

Usage: meter_digest.py --logs DIR [--role accept|verify|both] [--trunk-ms X]
  --trunk-ms is the measured ONE-row trunk wall (step minus head minus host_tail); with it the body row
  prints `second row = body - trunk`, which is the quantity B2/B3 are sized from.
"""
import argparse
import collections
import glob
import os
import re

KROWS = 2  # every arm tabulated here is a k=1 round: 1 target row + 1 draft row
PHASES = ["body", "body_with_checkpoint", "head", "draft", "verdict_kernel", "verdict_readback",
          "accept_path", "rollback", "lane_bookkeeping", "snapshot_take", "shadow_seed", "lane_save",
          "lane_revert", "argmax_report", "logits_d2h", "witness", "report_prints", "mid_checkpoint",
          "overhead"]
# Phases whose `runs` counts device submissions, so a per-launch figure means something.
LAUNCHISH = {"body", "head", "draft", "verdict_kernel", "lane_save", "lane_revert", "snapshot_take",
             "shadow_seed", "logits_d2h", "argmax_report", "lane_bookkeeping", "verdict_readback"}

LINE_RE = None  # field order in the engine's line is ms_total rounds ms_per_round sync_added runs ms_per_run
RESULT_RE = re.compile(r"meter=(?P<meter>\w+) rounds=(?P<rounds>\d+) sum_ms_per_round=(?P<sum>\S+) "
                       r"wall_ms_per_round=(?P<wall>\S+) coverage=(?P<cov>\S+)")


def parse(path):
    out = {"phases": {}, "verdict": None, "route": None, "stat_rounds": None, "mid": None}
    for line in open(path, errors="replace"):
        if line.startswith("MTPMETER role="):
            f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
            role, phase = f.get("role"), f.get("phase")
            # Two lines start with `MTPMETER role=` and are NOT phase rows: the `on=1` banner and the
            # `phase=instrument` header. Neither carries `status=` or `ms_per_round`, so defaulting a
            # missing status to "measured" (as the first version did) raised KeyError on the real logs.
            if phase is None or "ms_per_round" not in f:
                continue
            status = f.get("status", "measured")
            if phase == "mid_checkpoint":
                out["mid"] = status
            if status in ("measured", "native"):
                out["phases"][phase] = dict(
                    mpr=float(f["ms_per_round"]), runs=int(f.get("runs", 0)),
                    mprun=float(f.get("ms_per_run", 0.0)), sync=int(f.get("sync_added", 0)),
                    status=status, note=f.get("note", "").replace("_", " "))
        r = RESULT_RE.search(line)
        if r:
            out["verdict"] = dict(meter=r.group("meter"), rounds=int(r.group("rounds")),
                                  sum=float(r.group("sum")), wall=float(r.group("wall")),
                                  cov=float(r.group("cov")))
        if "mtp_route=" in line and out["route"] is None:
            mm = re.search(r"mtp_route=(\w+)", line)
            if mm:
                out["route"] = mm.group(1)
        if "MTPSTAT" in line and out["stat_rounds"] is None:
            mm = re.search(r"MTPSTAT.*? rounds=(\d+)", line)
            if mm:
                out["stat_rounds"] = int(mm.group(1))
    return out


def table(rows, role, trunk=None):
    widths = sorted({w for per in rows.values() for w in per})
    print(f"\n=== role={role}   ms per round (k=1 round, kRows={KROWS})")
    print(f"{'phase':20s}" + "".join(f"{w:>10s}" for w in widths) + f"{'ms/row':>9s}{'us/launch':>11s}")
    for ph in PHASES:
        vals = [(route, w) for route in sorted(rows) for w in widths
                if ph in rows[route].get(w, {}).get("phases", {})]
        if not vals:
            continue
        cells = ""
        for route in sorted(rows):
            for w in widths:
                p = rows[route].get(w, {}).get("phases", {})
                cells += f"{p.get(ph, {}).get('mpr', float('nan')):10.3f}" if ph in p else f"{'-':>10s}"
        last = rows[sorted(rows)[-1]].get(widths[-1], {}).get("phases", {}).get(ph, {})
        msrow = f"{last.get('mpr', 0) / KROWS:9.3f}" if ph in LAUNCHISH and last else f"{'':>9s}"
        usl = f"{last.get('mprun', 0) * 1000:11.1f}" if ph in LAUNCHISH and last.get("runs") else f"{'':>11s}"
        print(f"{ph:20s}{cells}{msrow}{usl}")
    print(f"{'WALL':20s}" + "".join(
        f"{rows[r].get(w, {}).get('verdict', {}).get('wall', float('nan')):10.3f}"
        for r in sorted(rows) for w in widths))
    print(f"{'coverage':20s}" + "".join(
        f"{rows[r].get(w, {}).get('verdict', {}).get('cov', float('nan')):10.4f}"
        for r in sorted(rows) for w in widths))
    print(f"{'meter':20s}" + "".join(f"{rows[r].get(w, {}).get('verdict', {}).get('meter', '?'):>10s}"
                                     for r in sorted(rows) for w in widths))
    if trunk is not None and "body" in PHASES:
        print(f"\n  second-row cost = body - trunk({trunk:.3f} ms measured 1-row trunk wall):")
        for r in sorted(rows):
            for w in widths:
                p = rows[r].get(w, {}).get("phases", {})
                if "body" in p:
                    print(f"    route={r:10s} lanes={w:>5s}: body {p['body']['mpr']:7.3f}  "
                          f"second row {p['body']['mpr'] - trunk:7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--role", default="both", choices=["accept", "verify", "both"])
    ap.add_argument("--trunk-ms", type=float, default=None)
    a = ap.parse_args()
    files = [p for p in sorted(glob.glob(os.path.join(a.logs, "*"))) if not p.endswith("/")]
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    bad = []
    for p in files:
        m = re.search(r"(\d+)_(hand|amortised)_(verify|accept)\.(?:log|txt)$", p)
        if not m:
            continue
        lanes, route, role = m.groups()
        d = parse(p)
        if d["verdict"] is None:
            bad.append((p, "no RESULT meter= line"))
            continue
        if d["verdict"]["meter"] != "ok":
            bad.append((p, f"meter={d['verdict']['meter']} coverage={d['verdict']['cov']}"))
            continue
        if d["stat_rounds"] is not None and d["stat_rounds"] != d["verdict"]["rounds"]:
            bad.append((p, f"meter rounds {d['verdict']['rounds']} != MTPSTAT rounds {d['stat_rounds']}"))
            continue
        by[role][route][lanes] = d
    for role in (["accept", "verify"] if a.role == "both" else [a.role]):
        if by.get(role):
            table(by[role], role, a.trunk_ms)
    if bad:
        print("\nREFUSED (a refused arm is not a measurement):")
        for p, why in bad:
            print(f"  {os.path.basename(p)}: {why}")
    print(f"\nread {len(files)} file(s); routes read back off the logs: "
          f"{sorted({d['route'] for r in by.values() for per in r.values() for d in per.values() if d['route']})}")


if __name__ == "__main__":
    main()
