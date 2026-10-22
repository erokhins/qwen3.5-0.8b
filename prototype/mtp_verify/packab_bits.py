#!/usr/bin/env python3
"""packab_bits.py — bit-level diff of two `pack_ab` output tiles, per row.

Why this exists: `pack_ab`'s CASE lines report THAT a row moved and how many elements, and `report.py` sizes
the aggregate (1-cos, meanrel, max ULP). Neither answers the question that decides the mechanism: **what are
the bits of the moved cells?** A mover whose reference bit is +0 and whose mutant bit is -0 is a signed-zero
carry; a mover at 0x3ecd -> 0x3ecc is a rounding-level change in a non-zero value; those are different defects
and the fix differs. So this prints the bits, for named rows, through `prototype/bf16_ulp.py` (never retyped).

Tile layout, as written by pack_ab's write_tile(): row-major [rows][8 heads][DH=256] BF16, so
element index = (row*8 + head)*256 + dim.

Usage: packab_bits.py A.bin B.bin [--rows 8] [--head 4] [--max 12]
Correctness-only, CPU only. No timing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bf16_ulp  # noqa: E402  (the repo's comparator; imported, not reimplemented)

DH = 256
HEADS = 8


def ulp(x: int, y: int) -> int:
    """BF16 ULP distance between two RAW BIT PATTERNS. `bf16_ulp.ulp_dist` takes BF16 *values* and would
    re-encode these, so the comparator's own `monotone_key` (its law for bit patterns) is what is called."""
    k = bf16_ulp.monotone_key(np.array([x, y], dtype=np.int32))
    return abs(int(k[0]) - int(k[1]))


def load(p: Path) -> list[int]:
    b = p.read_bytes()
    if len(b) % 2:
        raise SystemExit(f"{p}: odd byte count {len(b)} - not a BF16 tile")
    return [int.from_bytes(b[i:i + 2], "little") for i in range(0, len(b), 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rows", default="8", help="comma list of rows to print movers for")
    ap.add_argument("--head", type=int, default=-1, help="-1 = all heads")
    ap.add_argument("--max", type=int, default=12, help="max movers printed per row")
    ns = ap.parse_args()

    A, B = load(Path(ns.a)), load(Path(ns.b))
    if len(A) != len(B):
        raise SystemExit(f"size mismatch {len(A)} vs {len(B)} - two different geometries, not a diff")
    nrows = len(A) // (HEADS * DH)
    rows = [int(x) for x in ns.rows.split(",") if x != ""]
    for r in rows:
        if not 0 <= r < nrows:
            raise SystemExit(f"row {r} outside the tile's {nrows} rows")
    total = 0
    for r in rows:
        heads = range(HEADS) if ns.head < 0 else [ns.head]
        movers = []
        for h in heads:
            base = (r * HEADS + h) * DH
            for d in range(DH):
                x, y = A[base + d], B[base + d]
                if x != y:
                    movers.append((h, d, x, y, ulp(x, y)))
        total += len(movers)
        zeros = sum(1 for _, _, x, y, _ in movers if x in (0x0000, 0x8000) and y in (0x0000, 0x8000))
        print(f"row {r}: {len(movers)} movers, {zeros} of them a +/-0 sign flip")
        for m in movers[: ns.max]:
            h, d, x, y, u = m
            print(f"   head={h:2d} dim={d:3d}  {x:#06x} -> {y:#06x}   ulp={u}")
    print(f"TOTAL movers over requested rows: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
