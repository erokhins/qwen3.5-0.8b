#!/usr/bin/env python3
"""Summarise the WMMA latency window's own evidence file into the two comparisons it exists to make.

Why a file and not an awk one-liner: the `WL` rows are `key=value` soup with two occupancy geometries
mixed in, and a rate is only comparable inside one (iters pair, wavefronts/SIMD) — the pair alone is worth
~17 % at ILP 1. Printed as a table so a reader of `evidence/wmma-latency.txt` does not have to re-derive
which rows may be compared with which. Reads only; touches no device.

  python3 prototype/issue_rate_probe/wmma_latency_summary.py [evidence/wmma-latency.txt]
"""
import sys
from collections import OrderedDict

path = sys.argv[1] if len(sys.argv) > 1 else "prototype/issue_rate_probe/evidence/wmma-latency.txt"
rows = OrderedDict()
for line in open(path):
    if not line.startswith("WL "):
        continue
    d = {}
    for kv in line.split():
        if "=" in kv:
            k, _, v = kv.partition("=")
            d[k] = v
    key = (d["kernel"], d["iters_lo"] + "-" + d["iters_hi"], d["wavefronts_per_simd"],
           d["instr_per_step"], d["scale"])
    rows[key] = (float(d["clk_per_wmma_per_simd"]), float(d["Tflops_hi"]))

for wf, label in (("8.00", "shipped occupancy: 8 wavefronts/SIMD"),
                  ("16.00", "saturation geometry: 16 wavefronts/SIMD")):
    print("=== " + label)
    print("%-14s %-10s %-6s %-6s %-6s %10s %10s" %
          ("leg", "pair", "instr", "scale", "", "clk/wmma", "Tflops"))
    for (kern, pair, w, instr, scale), (clk, tf) in rows.items():
        if w != wf:
            continue
        print("%-14s %-10s %-6s %-6s %-6s %10.2f %10.2f" % (kern, pair, instr, scale, "", clk, tf))
    print()
