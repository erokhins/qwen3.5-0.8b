#!/usr/bin/env python3
"""Byte census of the MTP draft block and of everything a draft token must move.

P7 gate zero, part 1. The draft block is recorded as 39 MiB in `specification/00` / `01`, but the
figure that decides the route is not the block alone: a draft proposal has to become a TOKEN ID, and
the MTP head is TIED to the embedding table (`specification/09-mtp-draft-layer.md` S09.2:
`draft logits = x_d . E^T`), so a proposal that goes through a logits vector re-reads the whole
485 MiB tied table on top of the block. This script enumerates the checkpoint's `mtp.*` tensors,
sizes each one, and prints the two candidate draft-token traffic totals (full tied head vs a
k-row restricted head) as a fraction of the measured decode step.

No timing is taken and no device is touched: every ms figure on output is QUOTED from
`engine/generation-limits.md` and labelled as such on the line.

    .venv/bin/python prototype/mtp_gatezero/mtp_census.py [--fixture-shape 4032]
"""
from __future__ import annotations

import argparse
import json
import pathlib

from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen3.5-0.8B"
SHARD = MODEL / "model.safetensors-00001-of-00001.safetensors"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "BOOL": 1}

# QUOTED, not measured here: engine/generation-limits.md S1 (paired window, 4,032 lanes).
QUOTED = {
    "step_ms": 10.9357,          # decode step at 4,032 lanes
    "step_read_mib": 1532.3,     # per-token weight read
    "lm_head_mib": 509.552,      # the tied-table GEMV
    "lm_head_ms": 2.1685,        # the same GEMV's measured time -- 246 GB/s, 104 % of floor
    "launch_us": 3.10,           # measured cost of one launch (generation-limits S1 `cast` row)
    "plateau_gibs": 237.0,       # copy plateau, generation-limits S0 fact 1
}


def tensors(prefix: str) -> list[tuple[str, str, list[int], int]]:
    with safe_open(str(SHARD), framework="pt") as f:
        keys = [k for k in sorted(f.keys()) if k.startswith(prefix)]
        out = []
        for k in keys:
            sl = f.get_slice(k)
            shape = list(sl.get_shape())
            dt = sl.get_dtype()
            el = 1
            for s in shape:
                el *= s
            out.append((k, dt, shape, el * DTYPE_BYTES[dt]))
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    mtp = tensors("mtp.")
    emb = tensors("model.language_model.embed_tokens.weight")
    if not emb:
        raise SystemExit("no tied embedding table found; census cannot proceed")
    ekey, edt, eshape, ebytes = emb[0]
    vocab, hidden = eshape

    total = sum(b for _, _, _, b in mtp)
    print(f"checkpoint: {SHARD.name}")
    print(f"tied table {ekey}: {edt} {eshape} = {ebytes/2**20:.3f} MiB "
          f"(vocab={vocab}, hidden={hidden})")
    print(f"\nMTP block tensors ({len(mtp)}):")
    groups: dict[str, int] = {}
    for k, dt, shape, b in mtp:
        groups[k.split(".")[1] if k.count(".") > 1 else k] = groups.get(
            k.split(".")[1] if k.count(".") > 1 else k, 0) + b
        print(f"  {k:<58} {dt:<5} {str(shape):<16} {b/2**20:9.4f} MiB")
    for g, b in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  group {g:<20} {b/2**20:9.4f} MiB")
    print(f"\nMTP block total: {total} B = {total/2**20:.4f} MiB = {total/1e6:.4f} MB")
    print(f"  vs the step's measured weight read: {100.0*total/2**20/QUOTED['step_read_mib']:.2f} %")

    # ---- what a draft token must move, two routes -------------------------------------------
    q = QUOTED
    plateau_bps = q["plateau_gibs"] * 2**30
    draft_block = total
    lm_head = int(q["lm_head_mib"] * 2**20)

    def floor_ms(nbytes: int) -> float:
        return 1e3 * nbytes / plateau_bps

    print("\n=== cost of ONE draft proposal, priced from bytes ===")
    print(f"(every measured ms below is QUOTED from engine/generation-limits.md; no timing here)")
    print(f"{'route':<34} {'bytes':>12} {'floor ms':>9} {'c = /step':>10}")
    rows = []
    # route FULL: block + a full tied-head pass, priced at the measured lm_head line, not at floor
    full_ms = q["lm_head_ms"] + floor_ms(draft_block)
    rows.append(("full tied head (x_d . E^T)", draft_block + lm_head, full_ms / q["step_ms"], full_ms))
    for kk in (4, 8, 16, 32, 64, 128, 256):
        cand = kk * hidden * DTYPE_BYTES[edt]
        ms = floor_ms(draft_block + cand)
        rows.append((f"restricted head, top-{kk} rows", draft_block + cand, ms / q["step_ms"], ms))
    for name, nbytes, c, ms in rows:
        print(f"{name:<34} {nbytes/2**20:>9.3f} MiB {ms:>9.4f} {c:>10.4f}")
    print(f"\nThe FULL route is dominated by the tied table: {q['lm_head_ms']:.4f} ms of its "
          f"{rows[0][3]:.4f} ms is the second lm_head pass, quoted at the engine's own measured "
          f"lm_head line (246 GB/s, already 104 % of floor -- nothing here gets cheaper).")

    print("\n=== the win condition ===")
    print("k=1 drafting: tokens per step = 1 + a, cost per step = 1 + c  ->  speedup (1+a)/(1+c)")
    print("win iff a > c")
    for name, nbytes, c, ms in rows:
        for a in (0.30, 0.40, 0.50, 0.582, 0.642):
            pass
        need = c
        print(f"{name:<34} c={c:.4f}  -> needs accept > {need:.3f}"
              f"   speedup at a=0.642: {(1.642)/(1+c):.3f}x   at a=0.50: {(1.50)/(1+c):.3f}x")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            "mtp_block_bytes": total,
            "mtp_tensors": [{"name": k, "dtype": dt, "shape": sh, "bytes": b}
                            for k, dt, sh, b in mtp],
            "embed": {"key": ekey, "dtype": edt, "shape": eshape, "bytes": ebytes},
            "quoted": q,
            "routes": [{"route": n, "bytes": nb, "c": c, "draft_ms": ms}
                       for n, nb, c, ms in rows],
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
