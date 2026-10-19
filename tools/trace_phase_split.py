#!/usr/bin/env python3
"""trace_phase_split.py — group a rocprofv3 kernel trace into engine phases.

WHY. §22's phase comparison of the two engines was built by hand from a scratch trace. This makes it a tool,
because the operator's queue asks for the comparison to be re-runnable every time the engine moves.

THE RULE IT IS WRITTEN AROUND (rocprofv3 traps already recorded in AGENTS.md):
  * a trace row is per dispatch; a `--time-waves` run replays the wave, so a raw sum is a multiple of a pass.
    The pass count is DERIVED here from a kernel whose per-pass site count is known (--calibrate-kernel),
    never typed in; if that kernel is absent the tool refuses instead of guessing a divisor.
  * llama.cpp's numbers include trace overhead, which grows with launch count — so its totals are UPPER
    BOUNDS and every ratio that uses them is conservative. Stated in the output, not in a footnote.
  * kernel-name matching is on a prefix/substring of the demangled name, because `Kernel_Name` carries the
    signature (`attn_slim_wave(CoreArgs)`) and exact matching silently finds nothing.

Usage:
    tools/trace_phase_split.py tmp/gcc/llamaphase/llama_pp4096/pp4096_kernel_trace.csv \
        --engine llama --calibrate gated_delta_net --per-pass 144 --tokens 4096
    tools/trace_phase_split.py tmp/gcc/llamaphase/our_pp4096/pp4096_kernel_trace.csv \
        --engine ours --calibrate attn_prep --per-pass 48 --tokens 4096
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

# Phase classification, first match wins. Ordered: specific kernels before their families.
PHASES = [
    ("attention",   r"flash_attn|attn_fattn|attn_slim|attn_tile|attn_core|attn_combine|attn_prep|fmha"),
    ("gdn_recur",   r"gated_delta_net|delta_net|gdn_rec|rec_kernel|rec_fused|rec_loop"),
    ("gdn_other",   r"gdn_conv|gdn_gates|gated_norm|conv1d|causal_conv"),
    ("gemm",        r"gemm|wa_dot|scaled_dot|hgemm|Cijk|comgr|sgemm|k_wmma|matmul|dot_main|dot_large"),
    ("norm",        r"rms_norm|layer_norm|norm_f|_norm"),
    ("rope",        r"rope|rope_cache"),
    ("softmax",     r"softmax|soft_max"),
    ("silu_mlp",    r"swiglu|silu|up_silu|ffn|mlp"),
    ("copy",        r"copyBuffer|memcpy|copy_buffer|cast_|_cast"),
    ("sampler",     r"sample|argmax|mul_mat|lm_head"),
]


def classify(name):
    for phase, pat in PHASES:
        if re.search(pat, name, re.I):
            return phase
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--engine", default="?", help="label only")
    ap.add_argument("--calibrate", required=True,
                    help="substring of a kernel whose per-pass site count is known")
    ap.add_argument("--per-pass", type=int, required=True,
                    help="how many times that kernel launches per pass")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--geom", default=None,
                    help="census (workgroup, grid, VGPR, LDS) for kernels matching this substring - the shape "
                         "question a launch-count tie cannot answer")
    a = ap.parse_args()

    n_cal = 0
    per = defaultdict(lambda: [0, 0.0])   # kernel -> [count, ms]
    geom = defaultdict(lambda: [0, 0.0])  # (wg, grid, vgpr, lds) -> [count, ms], for --geom
    with open(a.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Kind", "").startswith("KERNEL"):
                pass
            name = row["Kernel_Name"]
            dur = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e6
            per[name][0] += 1
            per[name][1] += dur
            if a.calibrate in name:
                n_cal += 1
            if a.geom and a.geom in name:
                key = (int(row["Workgroup_Size_X"]), int(row["Grid_Size_X"]),
                       int(row["VGPR_Count"]), int(row["LDS_Block_Size"]))
                geom[key][0] += 1
                geom[key][1] += dur

    if n_cal == 0:
        sys.exit(f"REFUSE: calibration kernel {a.calibrate!r} appears 0 times. A guessed pass divisor is the "
                 f"rocprofv3 trap this tool exists to avoid; name a kernel that is really in the trace.")
    passes = max(1, round(n_cal / a.per_pass))
    print(f"# engine={a.engine} trace={a.csv_path}")
    print(f"# calibration: {a.calibrate!r} seen {n_cal}x, {a.per_pass}/pass => passes = {passes}"
          f"  (derived, not typed)")

    tot_ms = sum(v[1] for v in per.values()) / passes
    tot_launch = sum(v[0] for v in per.values()) / passes
    by_phase = defaultdict(lambda: [0, 0.0])
    for name, (c, ms) in per.items():
        p = classify(name)
        by_phase[p][0] += c / passes
        by_phase[p][1] += ms / passes

    print(f"# total (per pass): {tot_ms:.2f} ms, {tot_launch:.0f} launches, "
          f"{tot_launch / a.tokens:.2f} launches/token")
    print(f"# NOTE: traced ms are UPPER BOUNDS (trace overhead scales with launch count, the axis under study)")
    print(f"{'phase':<12} {'ms/pass':>10} {'share':>7} {'launches/pass':>14}")
    for p, (c, ms) in sorted(by_phase.items(), key=lambda kv: -kv[1][1]):
        print(f"{p:<12} {ms:>10.2f} {100*ms/tot_ms:>6.1f}% {c:>14.1f}")
    print(f"{'TOTAL':<12} {tot_ms:>10.2f} {'100.0%':>7} {tot_launch:>14.0f}")
    print("\n# top kernels by traced ms/pass")
    rows = sorted(per.items(), key=lambda kv: -kv[1][1])[: a.top]
    for name, (c, ms) in rows:
        print(f"  {ms/passes:>9.2f} ms  {c/passes:>7.1f}/pass  {classify(name):<11} {name[:96]}")

    if a.geom:
        print(f"\n# geometry census for kernels matching {a.geom!r} (20 CUs on this SKU, per device_facts)")
        print(f"  {'wgsize':>7} {'grid':>7} {'vgpr':>5} {'lds_B':>7} {'launches':>9} {'us/launch':>10} {'blocks/CU':>10}")
        for key, (c, ms) in sorted(geom.items(), key=lambda kv: -kv[1][1]):
            wg, grid, vgpr, lds = key
            blocks = grid // max(wg, 1)
            print(f"  {wg:>7} {grid:>7} {vgpr:>5} {lds:>7} {c/passes:>9.1f} {ms/max(c,1)*1000:>10.1f} "
                  f"{blocks/20:>10.2f}")


if __name__ == "__main__":
    main()
