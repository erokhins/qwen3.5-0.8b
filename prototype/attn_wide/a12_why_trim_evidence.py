#!/usr/bin/env python3
"""Trim the raw rocprofv3 trace to the evidence this finding actually rests on (CPU-only).

Writes, per arm: the trace tool's own summary header + per-kernel table (text), and a compact CSV of
just the attention CORE dispatches with duration and workgroups, which is what
`a12_why_body_compare.py` reads. The raw CSVs are 6 MB each and are not committed.
"""
import csv
import re
import sys
from pathlib import Path

SRC = Path("tmp/a12why")
DST = Path("prototype/attn_wide/evidence")
CORES = ("attn_fattn_wave_u8", "attn_wide_wave")


def trim_log(name: str, out: Path) -> None:
    text = (SRC / f"kt16384_{name}.log").read_text().splitlines()
    keep, in_tab = [], False
    for line in text:
        if line.startswith("---- prefill_kernel_trace ----"):
            in_tab = True
        if in_tab:
            keep.append(line)
            if line.startswith("BY WAVE"):
                break
    out.write_text("\n".join(keep) + "\n")
    print(f"{out}: {len(keep)} lines")


def core_rows(name: str, core: str, out: Path) -> None:
    rows = []
    with (SRC / f"kt16384_{name}" / "prefill_kernel_trace.csv").open() as fh:
        for rec in csv.DictReader(fh):
            if core in rec["Kernel_Name"]:
                wgs = 1
                for a in "XYZ":
                    wgs *= int(rec[f"Grid_Size_{a}"]) // int(rec[f"Workgroup_Size_{a}"])
                rows.append((int(rec["Start_Timestamp"]), int(rec["End_Timestamp"]), wgs,
                             int(rec["Grid_Size_X"]), int(rec["Grid_Size_Y"]), int(rec["Grid_Size_Z"]),
                             rec["Workgroup_Size_X"]))
    rows.sort()
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_ns", "end_ns", "us", "wgs", "grid_x", "grid_y", "grid_z", "threads"])
        for s, e, wgs, gx, gy, gz, t in rows:
            w.writerow([s, e, (e - s) / 1e3, wgs, gx, gy, gz, t])
    print(f"{out}: {len(rows)} core dispatches")


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    trim_log("wide0", DST / "a12_why_kernels_wide0.txt")
    trim_log("wide1", DST / "a12_why_kernels_wide1.txt")
    core_rows("wide0", "attn_fattn_wave_u8", DST / "a12_why_core_wide0.csv")
    core_rows("wide1", "attn_wide_wave", DST / "a12_why_core_wide1.csv")
    sys.exit(0)
