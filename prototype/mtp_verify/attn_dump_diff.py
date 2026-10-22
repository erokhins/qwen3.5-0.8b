#!/usr/bin/env python3
"""attn_dump_diff — which (layer, row) of the attention block's own output moved?

Reads the `--dump-attn-rows` tiles written by `engine/kernels_attn.cpp:dump_attention_body`
(`<dir>/attn<seq>_L<layer>_<pre_gate|gated>_r<n>_lanes<l>.bin`, BF16 [rows][8][256] — the dtype is read
off the FILE SIZE, AGENTS' dump-dtype row, not assumed) and compares two runs row by row.

This is the instrument that answers the question a logits file cannot: when token p changes, does query
row p-1's ATTENTION output change, and at which layer first. A row that changes here is a row whose
attention block read something past its causal bound; a row that does NOT change here but does change in
the logits localises the mixing DOWNSTREAM of the attention block.

No timing is taken or printed. Usage:
    attn_dump_diff.py <dirA> <dirB> [--rows 0,1,2,3,4] [--what pre_gate,gated]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

NAME = re.compile(r"^attn(\d+)_L(-?\d+)_([a-z0-9_]+)_r(\d+)_lanes(\d+)\.bin$")
ROW_FLOATS = 8 * 256          # ROW_H: [8 heads][256] per query row, `attn_decode_kernels.h`
# Per-file row width, and what a "row" means for it. The cache tiles are indexed by LANE, so a moved "row r"
# there is a moved cache lane r — which is exactly the thing being asked about.
WIDTHS = {
    "pre_gate": ROW_FLOATS, "gated": ROW_FLOATS, "in_q_rope": ROW_FLOATS, "in_gate": ROW_FLOATS,
    "in_k_rope": 2 * 256, "in_kcache0": 256, "in_kcache1": 256, "in_vcache0": 256, "in_vcache1": 256,
}
LANE_INDEXED = {"in_kcache0", "in_kcache1", "in_vcache0", "in_vcache1"}


def load_rows(path: Path) -> torch.Tensor:
    """One file as [rows][width] F32, dtype and width proven by byte count."""
    m = NAME.match(path.name)
    if not m:
        raise SystemExit(f"{path.name} is not an attn dump file (expected attn<seq>_L<layer>_<what>_...)")
    rows, what = int(m.group(4)), m.group(3)
    if what not in WIDTHS:
        raise SystemExit(f"{path.name}: unknown dump kind {what!r}; add it to WIDTHS")
    # The cache tiles carry `lanes` rows, not `rows` rows: lane 0..lanes_last is what the body could see.
    if what in LANE_INDEXED:
        rows = int(m.group(5)) + 1
    width = WIDTHS[what]
    raw = path.read_bytes()
    if len(raw) == rows * width * 2:
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16)
    elif len(raw) == rows * width * 4:
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
    else:
        raise SystemExit(f"{path}: {len(raw)} bytes is neither BF16 nor F32 for {rows} x {width}")
    return t.float().reshape(rows, width)


def one_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0 if torch.equal(a, b) else float("nan")
    return float(1.0 - torch.dot(a, b) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rows", default="0,1,2,3,4")
    ap.add_argument("--what", default="pre_gate,gated,in_q_rope,in_gate,in_k_rope,"
                                      "in_kcache0,in_kcache1,in_vcache0,in_vcache1")
    args = ap.parse_args()

    da, db = Path(args.a), Path(args.b)
    rows = [int(x) for x in args.rows.split(",") if x.strip()]
    whats = [w for w in args.what.split(",") if w.strip()]
    files_a = {}
    for p in sorted(da.glob("attn*_L*_*.bin")):
        m = NAME.match(p.name)
        if m and m.group(3) in whats:
            files_a[(int(m.group(2)), m.group(3))] = p
    if not files_a:
        raise SystemExit(f"{da}: no dump files for {whats}; an empty comparison cannot fail")

    print(f"attn_dump_diff A={da} B={db} rows={rows} what={whats} files={len(files_a)}")
    moved_total = 0
    for (layer, what), pa in sorted(files_a.items()):
        pb = db / pa.name
        if not pb.is_file():
            raise SystemExit(f"{pb} missing: both arms must be dumped the same way")
        ra, rb = load_rows(pa), load_rows(pb)
        unit = "lane" if what in LANE_INDEXED else "row"
        moved = [r for r in rows if not torch.equal(ra[r], rb[r])]
        moved_total += len(moved)
        detail = " ".join(
            f"{unit}{r}:1-cos={one_cos(ra[r].flatten(), rb[r].flatten()):.3e}" for r in moved)
        print(f"  L{layer:<3d} {what:<12s} {unit}s_that_moved={moved if moved else 'none'} {detail}")
    print(f"VERDICT moved_cells_layers={moved_total} "
          f"({'attention output DOES move: the leak is at or before the attention block' if moved_total else 'attention output identical: the mixing is downstream of the attention block'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
