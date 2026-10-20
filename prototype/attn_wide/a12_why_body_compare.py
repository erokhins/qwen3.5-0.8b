#!/usr/bin/env python3
"""Body-by-body comparison of the attention CORE launches, wide=0 against wide=1, at one width.

CPU-only: it reads the compact per-dispatch CSVs the trace window left in
`prototype/attn_wide/evidence/` (written by `a12_why_trim_evidence.py` from the raw rocprofv3 CSVs).
It groups the core dispatches into wave bodies in dispatch order — six attention layers per body — and
prints per-body mean duration and the workgroups each launch got, so "does the wide shape win, and at
which body width" is answered from the device's own timestamps rather than from a rig.

    .venv/bin/python prototype/attn_wide/a12_why_body_compare.py

The shipping arm does not launch its core for bodies at or below MAXSC (`ATTN_FATTN_SKIP`), so the wide
arm has one more body than the shipping arm and the pairing is offset by that.
"""
import csv
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent / "evidence"
LAYERS = 6          # attention layers per body in this checkpoint (bench §51's launch-site count)
CUS = 40            # devices/strix-halo-compute.md: 40, never the 20 HIP reports


def load(path: Path):
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh)]
    rows.sort(key=lambda r: int(r["start_ns"]))
    return [((int(r["end_ns"]) - int(r["start_ns"])) / 1e3, int(r["wgs"])) for r in rows]


def bodies(rows, per_body=LAYERS):
    out = []
    for i in range(0, len(rows) - per_body + 1, per_body):
        chunk = rows[i:i + per_body]
        out.append((sum(d for d, _ in chunk) / per_body, sum(w for _, w in chunk) / per_body))
    return out


def main() -> int:
    ship = bodies(load(EV / "a12_why_core_wide0.csv"))
    wide = bodies(load(EV / "a12_why_core_wide1.csv"))
    off = len(wide) - len(ship)
    print(f"bodies: wide={len(wide)} shipping={len(ship)} (offset {off}: the bodies at or below MAXSC,")
    print("which the shipping arm does not launch its core for at all)")
    print(f"{'body':>5} {'lanes_last':>10} {'ship us/launch':>15} {'wide us/launch':>15} "
          f"{'wide/ship':>10} {'ship wgs':>9} {'wide wgs':>9}   (wgs = workgroups per launch;"
          f" {CUS} CUs)")
    tot_s = tot_w = 0.0
    for i, w in enumerate(wide):
        if i < off:
            print(f"{i+1:>5} {(i+1)*512:>10} {'--not launched--':>15} {w[0]:>15.3f} "
                  f"{'':>10} {w[1]:>9.0f}")
            continue
        s = ship[i - off]
        tot_s += s[0]
        tot_w += w[0]
        print(f"{i+1:>5} {(i+1)*512:>10} {s[0]:>15.3f} {w[0]:>15.3f} {w[0]/s[0]:>10.4f} "
              f"{s[1]:>9.0f} {w[1]:>9.0f}")
    print(f"SUM over the paired bodies (per-launch means): shipping {tot_s/1e3:.2f} ms  "
          f"wide {tot_w/1e3:.2f} ms  ratio {tot_w/tot_s:.4f}   (each mean is over the 6 attn layers;")
    print("      the SUM is therefore per-launch means summed across bodies, a paired index, not a wall)")
    print("note: the ship-wgs column is the launch's workgroup count and grows with the body's `split`;")
    print("the wide arm's is 2 x ceil(wave_rows/32) = 32 at wave_rows 512, at EVERY body width.")


if __name__ == "__main__":
    sys.exit(main())
