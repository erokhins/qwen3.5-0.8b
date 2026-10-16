#!/usr/bin/env python3
"""How many pinned prefill rows can the reference itself barely decide — the census a row gate needs.

Section (i) of `engine/gate.py` judges every pinned prefill row's argmax, and a row-level argmax rule is
only as strong as the rows' own decision margin. Two classes make a move admissible rather than a defect,
and which class a fixture is full of is a property of the DUMP, not of the engine:

  * an exact TIE at the row's maximum — the reference did not choose, so no engine can be wrong about it;
  * a thin top-1/top-2 ROOM (under `CASCADE_BAND["flip_room_max_ulp"]` BF16 ULP at the row's own top-1 —
    the line the flip rule turns on) — a BF16-class difference in a row that was nearly decided is the
    class of flip the cascade rule exists for.

Print both counts, the gap distribution, and the gaps at the lanes the streaming dispute is actually about.
If a fixture turned out to be mostly thin-room rows, a green row gate on it would be worth nothing; this is
the number that says how much of a given pin is load-bearing.

    .venv/bin/python tools/prefill_row_rooms.py data/refs/wide1024f
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from gate import CASCADE_BAND, bf16_ulp    # the same threshold and step function the gate's flip rule
                                            # uses, not a second copy


def main() -> int:
    where = Path(sys.argv[1] if len(sys.argv) > 1 else "data/refs/wide1024f")
    raw = np.load(where / "logits_rows.npy")            # uint16 = BF16 payload, §10.5's bitcast
    wide = (raw.astype(np.uint32) << 16).view(np.float32)   # gate.py's widening, same arithmetic
    print(f"{where}: {wide.shape[0]} rows x {wide.shape[1]} ids, stored as {raw.dtype}")

    top = wide.max(axis=1, keepdims=True)
    tied = np.flatnonzero((wide == top).sum(axis=1) > 1)
    part = np.partition(wide, -2, axis=1)
    gap = part[:, -1] - part[:, -2]
    # The line in its own unit: each row's gap over one BF16 step at THAT row's top-1, via the same
    # `bf16_ulp` the gate uses, so this census and the gate count the same rows thin. These are the
    # reference's own rows, which is also the reading the gate applies to a pin (`reference_room_ulp`).
    room_ulp = gap / np.vectorize(bf16_ulp, otypes=["float64"])(top[:, 0])
    room = float(CASCADE_BAND["flip_room_max_ulp"])
    thin = np.flatnonzero(room_ulp < room)
    print(f"  exact ties at the maximum: {len(tied)} row(s) — {[int(x) for x in tied[:8]]}"
          + (" …" if len(tied) > 8 else ""))
    print(f"  room under {room:.2f} BF16 ULP at the row's own top-1 (the flip rule's own threshold):"
          f" {len(thin)} row(s) — {[int(x) for x in thin[:8]]}"
          + (" …" if len(thin) > 8 else "")
          + f"   [median room {float(np.median(room_ulp)):.4g} ULP, minimum"
          f" {float(room_ulp.min()):.4g} ULP]")
    print(f"  i.e. {wide.shape[0] - len(set(tied) | set(thin))} of {wide.shape[0]} rows were decided"
          " by the reference with room to spare; that many are what a row gate can actually judge.")
    for lane in (511, 512, 513, wide.shape[0] - 1):
        if 0 <= lane < wide.shape[0]:
            note = " (first row above MAXSC)" if lane == 512 else (
                " (the row §15 disputes)" if lane == wide.shape[0] - 1 else "")
            print(f"  row {lane}: top-1/top-2 room {float(gap[lane]):.4g} logits"
                  f" = {float(room_ulp[lane]):.4g} ULP{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
