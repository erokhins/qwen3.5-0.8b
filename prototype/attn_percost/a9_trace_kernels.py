#!/usr/bin/env python3
"""a9_trace_kernels.py — per-attention-kernel traced ms, whole-prompt-pass basis, for A9 §2.

Reads the traces §51 already captured (CPU only, no device). `tools/trace_phase_split.py` divides by the
derived pass count and so reports a PER-BODY-PASS unit (`attn_prep` is 6 sites per body, not per prompt pass);
this tool multiplies by the body count W/512 so a per-kernel figure can be compared against a per-kernel byte
sum from `a9_waste_model.py`. Passes are still DERIVED from `attn_prep`'s 6 sites per body (the same rule
`tools/trace_phase_split.py` encodes); nothing is summed raw.

Usage: python3 prototype/attn_percost/a9_trace_kernels.py 16384 32768
"""
import collections
import csv
import sys

ATTN = ("attn_prep", "attn_slim_wave", "attn_fattn_wave", "attn_combine", "attn_tile_wave", "attn_core")


def main():
    for width in (int(a) for a in sys.argv[1:] or [16384, 32768]):
        path = f"tmp/gcc/llamaphase/our_pp{width}/pp{width}_kernel_trace.csv"
        count, dur = collections.Counter(), collections.Counter()
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                name = row["Kernel_Name"].split("(")[0].split("<")[0]
                name = name.split("::")[-1]
                d = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e6
                count[name] += 1
                dur[name] += d
        n_prep = count["attn_prep"]
        if n_prep % 6:
            sys.exit(f"REFUSE: attn_prep seen {n_prep}x, not a multiple of 6 sites/body")
        body_passes = n_prep // 6                     # 3 wave replays x W/512 bodies
        bodies = width // 512
        passes_replayed = body_passes // bodies
        print(f"\n=== ours at {width:,} lanes: {bodies} bodies x {passes_replayed} replays "
              f"= {body_passes} body-passes (derived from attn_prep {n_prep}/6) ===")
        print(f"  {'kernel':<20} {'launches/pass':>14} {'ms/body-pass':>13} {'ms/pass':>10}")
        tot = 0.0
        for name in sorted(count, key=lambda n: -dur[n]):
            if not any(a in name for a in ATTN):
                continue
            lp = count[name] / body_passes
            mb = dur[name] / body_passes
            tot += mb * bodies
            print(f"  {name:<20} {lp:>14.2f} {mb:>13.3f} {mb * bodies:>10.2f}")
        print(f"  {'ATTENTION TOTAL':<20} {'':>14} {'':>13} {tot:>10.2f}")
        allms = sum(dur.values()) / body_passes * bodies
        print(f"  whole wave traced {allms:.2f} ms/pass; attention share "
              f"{100 * tot / allms:.1f} %")


if __name__ == "__main__":
    main()
