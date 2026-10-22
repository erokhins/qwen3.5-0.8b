#!/usr/bin/env python3
"""pack_ab report — size the rig's per-case tiles in ULPs, with the repo's own comparator.

`pack_ab` (the device rig) prints only byte-level reach counts, because a ULP number printed by the
program under test is a number nobody can re-check. This file does the sizing, on CPU, and it IMPORTS
`prototype/bf16_ulp.py` rather than retyping it: that module exists because the naive BF16 ULP comparator
has a dead branch on a uint16 view and reports nonsense on half the tensor (its docstring), and a rig that
re-derived the comparator would be back to agreeing with itself.

Every dump's dtype is inferred from its byte count and asserted, never assumed (`AGENTS`: reading a BF16
tile as F32, or back, yields a plausible ndiff and a NaN cos).

    .venv/bin/python prototype/mtp_verify/pack_ab/report.py --rig-dir /tmp/packab/rig \
        --ref /tmp/packab/dump-ref --seq 1 --layer 3 --row 8

No GPU work, no timing.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # prototype/, where bf16_ulp lives
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
import bf16_ulp                                                        # noqa: E402  (imported, not retyped)

N_Q, DH, N_KV = 8, 256, 2
ROW_H = N_Q * DH


def load_bf16(path: Path, elems: int) -> torch.Tensor:
    """Raw BF16 tile -> BF16 tensor, with the dtype proved by the byte count."""
    data = path.read_bytes()
    if len(data) != elems * 2:
        raise SystemExit(f"{path.name}: {len(data)} B is not {elems} BF16 elements"
                         f" (nor {elems * 4} B for F32); refusing to guess a dtype")
    bits = np.frombuffer(data, dtype=np.uint16).copy()   # copy: frombuffer is read-only and torch warns
    # `.view`, not `.to`: `torch.Tensor.to(uint16)` CASTS and writes garbage that reads like unwritten
    # memory (prototype/gdn_gated_norm/results.md §8).
    return torch.from_numpy(bits).view(torch.bfloat16)


def rows_of(t: torch.Tensor, rows: int) -> torch.Tensor:
    return t.reshape(rows, N_Q, DH)


def size_row(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """bf16_ulp.report over one row, plus the 1-cos the other instruments here print."""
    r = bf16_ulp.report(got, ref)
    r["one_minus_cos"] = 1.0 - r["cos"]
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig-dir", required=True)
    ap.add_argument("--ref", required=True, help="the dump dir pack_ab was run on")
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--row", type=int, default=8, help="the adjudicated row (bound = lvalid + row)")
    ap.add_argument("--lvalid", type=int, default=1)
    ap.add_argument("--worst", type=int, default=6, help="print this many worst moved rows below")
    ap.add_argument("--detail", default="", help="comma list of case names: dump every differing element of"
                    " the adjudicated row (head, dim, ref/got bits and values, ULP)")
    ap.add_argument("--probe", default="", help="head:dim — print that ONE element of the adjudicated row"
                    " for every case. A monotone ladder of input values against this column is the linear-"
                    " response test: a real weight makes it monotone with a proportional slope.")
    args = ap.parse_args()

    ref_tile = next(Path(args.ref).glob(f"attn{args.seq:03d}_L{args.layer}_pre_gate_r*_lanes*.bin"))
    m = re.search(r"_r(\d+)_lanes(\d+)", ref_tile.name)
    rows, lanes = int(m.group(1)), int(m.group(2))
    ref = load_bf16(ref_tile, rows * ROW_H)
    rr = rows_of(ref, rows)
    bound = args.lvalid + args.row
    jbad = bound
    print(f"report ref={ref_tile.name} rows={rows} lanes={lanes} adjudicated_row={args.row}"
          f" bound={bound} (odd={bound % 2}) disallowed_lane={jbad}")
    print("columns: rows_below = rows < adjudicated whose pre_gate tile differs at all;"
          " ULP/1cos/meanrel are bf16_ulp on the ADJUDICATED row only")

    cases = sorted(p for p in Path(args.rig_dir).glob("*_L*.bin") if p.name != "fid_L%d.bin" % args.layer)
    detail = {s.strip() for s in args.detail.split(",") if s.strip()}
    ph, pd = (-1, -1)
    if args.probe:
        ph, pd = (int(x) for x in args.probe.split(":"))
        print(f"probe element: adjudicated row {args.row} head={ph} dim={pd}"
              f"  ref=0x{bf16_ulp.bf16_bits(rr[args.row].reshape(-1))[ph*DH+pd]:04x}")
    # The disallowed lane's V row, printed next to every DETAIL line: if a mover's magnitude tracks V9 and
    # the case's V9 change, the term is linear in V9 and the dead weight is a number; if not, it is not.
    v9 = {}
    for kv in range(N_KV):
        vp = next(Path(args.ref).glob(f"attn{args.seq:03d}_L{args.layer}_in_vcache{kv}_r*_lanes*.bin"))
        nrv = int(re.search(r"_lanes(\d+)", vp.name).group(1)) + 1
        v9[kv] = load_bf16(vp, nrv * DH).reshape(nrv, DH)[args.lvalid + args.row].float().numpy()
    n_leak = 0
    for p in cases:
        got = load_bf16(p, rows * ROW_H)
        gr = rows_of(got, rows)
        below = [r for r in range(args.row) if not torch.equal(gr[r], rr[r])]
        at = not torch.equal(gr[args.row], rr[args.row])
        above = sum(1 for r in range(args.row + 1, rows) if not torch.equal(gr[r], rr[r]))
        s = size_row(gr[args.row].reshape(-1), rr[args.row].reshape(-1))
        # Per-head detail: a row is 8 heads x DH, and a leak that lands in one head is a different fact
        # from one that lands in all of them.
        heads = [h for h in range(N_Q) if not torch.equal(gr[args.row][h], rr[args.row][h])]
        is_ctrl = p.name.startswith("widen_")
        verdict = ("LEAK" if below else ("null" if not is_ctrl else "CONTROL DID NOT FIRE"))
        if below:
            n_leak += 1
        print(f"  {p.name.replace('_L%d.bin' % args.layer, ''):<11}"
              f" rows_below={len(below):<2} {str(below[:args.worst]):<26}"
              f" row_at={int(at)} heads_moved={str(heads):<14} rows_above={above:<4}"
              f" ndiff={s['ndiff']:<7} maxulp={s['maxulp']:<4} over1={s['over1']:<3}"
              f" n_bad={s['n_bad']:<3} 1-cos={s['one_minus_cos']:.3e}"
              f" meanrel={s['meanrel']:.3e} -> {verdict}")
        if ph >= 0:
            i = ph * DH + pd
            bits = bf16_ulp.bf16_bits(gr[args.row].reshape(-1))[i]
            print(f"    PROBE head={ph} dim={pd}: bits=0x{bits:04x}"
                  f" value={gr[args.row].float().reshape(-1)[i]:+.9g}"
                  f" ulp_vs_ref={int(bf16_ulp.ulp_dist(gr[args.row].reshape(-1), rr[args.row].reshape(-1))[i])}")
        if p.name.replace("_L%d.bin" % args.layer, "") in detail:
            gv = gr[args.row].reshape(-1).float().numpy()
            rv = rr[args.row].reshape(-1).float().numpy()
            gb = bf16_ulp.bf16_bits(gr[args.row].reshape(-1))
            rb = bf16_ulp.bf16_bits(rr[args.row].reshape(-1))
            ul = bf16_ulp.ulp_dist(gr[args.row].reshape(-1), rr[args.row].reshape(-1))
            idx = np.where(ul > 0)[0]
            print(f"    DETAIL {p.name} adjudicated row {args.row}: {len(idx)} differing elements"
                  f" (ulp = BF16 ULP via bf16_ulp.ulp_dist)")
            for i in idx[:40]:
                h, d = int(i) // DH, int(i) % DH
                print(f"    head={h} dim={d:4d} ref_bits=0x{rb[i]:04x} ref={rv[i]:+.9g}"
                      f" got_bits=0x{gb[i]:04x} got={gv[i]:+.9g} ulp={int(ul[i])}"
                      f" |V9[dim]={v9[int(h) // 4][d]:+.6g}")
    print(f"SUMMARY leak_cases={n_leak} of {len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
