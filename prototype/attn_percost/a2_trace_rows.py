#!/usr/bin/env python3
"""A2 census aid: per-KERNEL geometry rows out of a rocprofv3 kernel_trace.csv.

WHY THIS EXISTS AND WHY IT IS NOT `tools/trace_phase_split.py --geom`. That tool is the right instrument for
phases and for the three-axis geometry of ONE kernel (`--geom SUBSTRING`), and it is what §43 fixed. But its
`--geom` table aggregates by (workgroup, grid, vgpr, lds) and DOES NOT PRINT THE KERNEL NAME, so a run over
`attn` merges four different kernels into geometry-keyed rows — you cannot tell which row is `attn_fattn_wave`.
It also drops three columns this census needs: `Accum_VGPR_Count`, `SGPR_Count`, `Scratch_Size`. On RDNA the arch
and accum VGPR pools are separate, so the (VGPR, Accum_VGPR) pair is what a residency claim must read, and
`Scratch_Size` is the trace's own witness for a spill.

Everything else is deliberately the same as the tool it reuses: `Grid_Size_*` are THREADS per dimension and
blocks = grid_threads / workgroup (AGENTS.md, "a profiler column read on one axis is not a geometry"); CUs and
SIMDs come from KFD, never from `hipGetDeviceProperties`; the pass count is DERIVED from a kernel whose
per-pass site count is known, never typed.

Canaries: refuses a kernel matched 0 times, refuses a non-whole block division, and prints the pass divisor it
derived. CPU-only: reads a CSV that is already on disk, launches nothing, times nothing.

    python3 prototype/attn_percost/a2_trace_rows.py CSV --calibrate attn_prep --per-pass 48 [--match attn]
"""
import argparse
import csv
import glob
import re
import sys
from collections import defaultdict

KFD_NODE_PROPS = "/sys/devices/virtual/kfd/kfd/topology/nodes/*/properties"


def kfd_topology():
    simd = cu_simd = 0
    for path in glob.glob(KFD_NODE_PROPS):
        vals = {}
        try:
            with open(path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 2:
                        vals[parts[0]] = int(parts[1])
        except OSError:
            continue
        simd = max(simd, vals.get("simd_count", 0))
        cu_simd = max(cu_simd, vals.get("simd_per_cu", 0))
    if simd <= 0 or cu_simd <= 0:
        print("# WARN: KFD unreadable, falling back to (80, 2)", file=sys.stderr)
        return 80, 2
    return simd, cu_simd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--calibrate", required=True)
    ap.add_argument("--per-pass", type=int, required=True)
    ap.add_argument("--match", default="", help="substring of the kernel name; empty = every kernel")
    a = ap.parse_args()

    simd, simd_per_cu = kfd_topology()
    cus = max(1, simd // max(1, simd_per_cu))
    n_cal = 0
    per = defaultdict(lambda: {"n": 0, "ms": 0.0})
    with open(a.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["Kernel_Name"]
            if a.calibrate in name:
                n_cal += 1
            if a.match and a.match not in name:
                continue
            dur = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e6
            key = (name,
                   int(row["Workgroup_Size_X"]), int(row["Workgroup_Size_Y"]), int(row["Workgroup_Size_Z"]),
                   int(row["Grid_Size_X"]), int(row["Grid_Size_Y"]), int(row["Grid_Size_Z"]),
                   int(row["VGPR_Count"]), int(row["Accum_VGPR_Count"]), int(row["SGPR_Count"]),
                   int(row["LDS_Block_Size"]), int(row["Scratch_Size"]))
            d = per[key]
            d["n"] += 1
            d["ms"] += dur

    if n_cal == 0:
        sys.exit("REFUSE: calibration kernel %r seen 0 times." % a.calibrate)
    passes = max(1, round(n_cal / a.per_pass))
    print("# trace=%s  passes=%d (derived from %r: %d seen, %d/pass)"
          % (a.csv_path, passes, a.calibrate, n_cal, a.per_pass))
    print("# CUs=%d SIMDs=%d (KFD simd_count / simd_per_cu). Grid_Size_* are THREADS; blocks = grid/workgroup."
          % (cus, simd))
    print("%-42s %-11s %-16s %6s %5s %7s %5s %5s %4s %6s %7s %7s %8s %9s"
          % ("kernel", "workgroup", "grid_threads", "blk", "wf", "wf/SIMD", "blk/CU", "thr/blk",
             "vgpr", "agpr", "sgpr", "lds_B", "scratch", "ms/pass"))
    tot = 0.0
    for key, d in sorted(per.items(), key=lambda kv: -kv[1]["ms"]):
        name, wx, wy, wz, gx, gy, gz, vgpr, agpr, sgpr, lds, scratch = key
        threads = wx * wy * wz
        tgrid = gx * gy * gz
        if threads <= 0 or tgrid % threads:
            sys.exit("REFUSE: grid %d is not a whole number of %d-thread workgroups for %r"
                     % (tgrid, threads, name))
        blocks = tgrid // threads
        wf = blocks * ((threads + 31) // 32)
        ms = d["ms"] / passes
        tot += ms
        print("%-42s %-11s %-16s %6d %5d %7.2f %7.2f %8d %5d %5d %4d %6d %7d %8.2f"
              % (name[:42], "(%d,%d,%d)" % (wx, wy, wz), "(%d,%d,%d)" % (gx, gy, gz), blocks, wf,
                 wf / float(simd), blocks / float(cus), threads, vgpr, agpr, sgpr, lds, scratch, ms))
    print("TOTAL %.2f ms/pass over %d geometry rows" % (tot / 1.0, len(per)))


if __name__ == "__main__":
    main()
