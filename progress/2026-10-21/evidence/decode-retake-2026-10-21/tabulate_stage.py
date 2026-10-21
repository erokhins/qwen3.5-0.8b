#!/usr/bin/env python3
"""Tabulate the 2026-10-21 decode stage-clock window into medians.

Reads the six `--time-stages` logs the window wrote (128 lanes x3, 4,032 lanes x3) and prints, per
context point: the burst / stage / kernel granularity means per run, the `STAGEGAP` exposed bill,
and the per-family `ms_per_step` at stage granularity with its median of 3 and share of the step.
Nothing here is a new measurement: it is a reader for logs the main agent already took, and the
median-of-3 rule is the ledger's. Asserts the family set is the same in every run (a family that
disappears from one log would otherwise median over a short list and print a plausible number).
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TMP = Path(__file__).resolve().parent
FAMILIES = ["glue", "proj", "gdn", "attn", "mlp", "cast", "copy", "lm_head", "host_tail"]

runs = defaultdict(list)  # point -> list of per-run dicts


def parse(path: Path) -> dict:
    text = path.read_text()
    out = {"file": path.name}
    m = re.search(r"STAGEPASS gran=burst steps=\d+ ms_mean=([\d.]+)", text)
    out["burst"] = float(m.group(1))
    m = re.search(r"STAGEPASS gran=stage steps=\d+ ms_mean=([\d.]+)", text)
    out["stage"] = float(m.group(1))
    m = re.search(r"STAGEPASS gran=kernel steps=\d+ ms_mean=([\d.]+)", text)
    out["kernel"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"STAGEGAP gran=stage .*?exposed_ms=([\d.]+)", text)
    out["exposed"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"STAGE gran=stage TOTAL .*?ms_per_step=([\d.]+).*?launches_per_step=(\d+) "
                  r"plan_per_step=(\d+) plan_match=(\d+) syncs_per_step=(\d+)", text)
    out["total"] = float(m.group(1))
    out["launches"] = int(m.group(2))
    out["plan_match"] = int(m.group(4))
    out["syncs"] = int(m.group(5))
    for fam in FAMILIES:
        m = re.search(r"STAGE gran=stage family=%s\s+ms_per_step=([\d.]+)" % fam, text)
        if not m:
            raise SystemExit(f"{path.name}: no STAGE row for family={fam} — refusing to median a short list")
        out[fam] = float(m.group(1))
    s = sum(out[f] for f in FAMILIES)
    out["sum_families"] = s
    out["ungaged_ms"] = out["total"] - s
    return out


for p in sorted(TMP.glob("stage*_*.txt")):
    point = "128" if "128" in p.name else "4032"
    runs[point].append(parse(p))

for point in ("128", "4032"):
    rs = runs[point]
    assert len(rs) == 3, f"point {point}: expected 3 runs, found {len(rs)}"
    print(f"\n=== {point} lanes ({len(rs)} runs) ===")
    for k in ("burst", "stage", "kernel", "exposed", "total"):
        vals = [r[k] for r in rs]
        print(f"{k:9s} " + " / ".join(f"{v:.4f}" for v in vals) + f"   median {statistics.median(vals):.4f}")
    print(f"launches_per_step {rs[0]['launches']} plan_match {rs[0]['plan_match']} syncs {rs[0]['syncs']}")
    med_stage = statistics.median(r["stage"] for r in rs)
    print(f"{'family':10s} {'med ms/step':>12s} {'share of stage TOTAL':>21s}   three runs")
    for fam in FAMILIES:
        vals = [r[fam] for r in rs]
        med = statistics.median(vals)
        print(f"{fam:10s} {med:12.4f} {100*med/med_stage:20.1f}%   " +
              " / ".join(f"{v:.4f}" for v in vals))
    tot = [r["total"] for r in rs]
    print(f"{'TOTAL':10s} {statistics.median(tot):12.4f}")
    print(f"sum-families minus TOTAL per run: " +
          " ".join(f"{r['ungaged_ms']:+.4f}" for r in rs))
