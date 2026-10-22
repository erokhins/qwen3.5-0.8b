#!/usr/bin/env python3
"""attn_dump_shape — of WHAT does the moved row consist? Scale, or direction?

Reads two `--dump-attn-rows` tiles for the same (layer, kind) and, for a row that moved, asks whether the
difference is a per-row (or per-head) MULTIPLICATIVE factor or an additive/directional change. The two
point at different defects, and the distinction is the whole reason this exists:

  * a constant ratio `ref/pert` over the row   -> the softmax DENOMINATOR moved, no extra lane's V
    reached the output. That is a rowsum (`l`) leak: something past the bound changed the normaliser.
  * a direction change (ratios scattered)     -> a weight or a V operand from past the bound reached the
    accumulator, which the cell-level probe in `causal_bound.cpp` would have seen and did not.

Ratios are taken on the F32 widening of the BF16 tiles; `ndiff` and the max |rel| are printed with the
ratio distribution because a ratio histogram that clusters on ONE value is the finding, and one that does
not is a different finding. No timing, no thresholds: this prints measurements, it does not judge them.

Usage: attn_dump_shape.py <dirA> <dirB> --layer 3 --row 4 [--what pre_gate]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

NAME = re.compile(r"^attn(\d+)_L(-?\d+)_(pre_gate|gated)_r(\d+)_lanes(\d+)\.bin$")
ROW_FLOATS = 8 * 256


def load(path: Path, want_layer: int, want_what: str) -> torch.Tensor:
    for p in sorted(path.glob("attn*_L*_*.bin")):
        m = NAME.match(p.name)
        if m and int(m.group(2)) == want_layer and m.group(3) == want_what:
            raw = p.read_bytes()
            rows = int(m.group(4))
            if len(raw) != rows * ROW_FLOATS * 2:
                raise SystemExit(f"{p}: {len(raw)} bytes is not BF16 for {rows} rows")
            return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()
                                   ).view(torch.bfloat16).float().reshape(rows, 8, 256), p.name
    raise SystemExit(f"{path}: no L{want_layer} {want_what} file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--what", default="pre_gate")
    args = ap.parse_args()

    ra, name_a = load(Path(args.a), args.layer, args.what)
    rb, _ = load(Path(args.b), args.layer, args.what)
    x = ra[args.row]           # [8][256]
    y = rb[args.row]
    n = int((x != y).sum())
    print(f"attn_dump_shape L{args.layer} {args.what} row {args.row} ({name_a})")
    print(f"  differing elements = {n} of {x.numel()}")
    if n == 0:
        print("  row identical; nothing to characterise")
        return 0
    nz = (x != 0) & (y != 0)
    r = (x / torch.where(nz, y, torch.ones_like(y)))[nz]
    print(f"  ratio ref/pert over the non-zero pairs: min={float(r.min()):.6f} max={float(r.max()):.6f} "
          f"mean={float(r.mean()):.6f} spread={float(r.max()-r.min()):.3e}")
    print(f"  distinct ratios to 6 decimals = {len(set([f'{float(v):.6f}' for v in r]))} "
          f"(1.0 == a uniform scale on this row)")
    for h in range(8):
        xh, yh = x[h], y[h]
        nh = int((xh != yh).sum())
        if nh == 0:
            print(f"   head {h}: identical")
            continue
        nzh = (xh != 0) & (yh != 0)
        rh = xh[nzh] / yh[nzh]
        print(f"   head {h}: {nh}/256 differ, ratio min={float(rh.min()):.6f} max={float(rh.max()):.6f} "
              f"distinct6={len(set([f'{float(v):.6f}' for v in rh]))}")
    # And the direction question, independent of scale.
    ca = torch.nn.functional.cosine_similarity(x.flatten().unsqueeze(0), y.flatten().unsqueeze(0))
    print(f"  1-cos(row) = {float(1-ca):.3e}   (a pure scale would make this 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
