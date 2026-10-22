#!/usr/bin/env python3
"""rowdiff — quantify the difference between two logits rows, in the repo's own metrics.

Thin front end over `prototype/bf16_ulp.py`, deliberately: the ULP comparator every other gate uses lives
there (with the sign-correction that a `uint16` view makes a dead branch, and the cancellation band
§10.1 F1–F4 says a BF16 comparison needs), and a second spelling of it is how a gate ends up adjudicating
nothing. What this file adds is the row slicing and the one thing `report` does not print: which token each
side argmaxes to, because a row's *decision* is the quantity the MTP verify contract is written about.

Two traps it exists to avoid, both from `AGENTS.md`:
  * a logits dump's dtype must be read off its SIZE, not assumed — an F32 file read as BF16 still yields a
    plausible-looking ndiff over half its length, and a comparator that cannot produce its headline metric
    still prints a confident number;
  * the argmax pairing needs to say which alignment it used, so `--label` is not decoration.

    .venv/bin/python prototype/mtp_verify/rowdiff.py A.bin B.bin [--row-a 0] [--row-b 0]
                                       [--rows 2] [--rows-b 1] [--label "..."]
No timing taken or printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bf16_ulp  # noqa: E402

VOCAB = 248320
SIZES = {"f32": 4, "bf16": 2}


def load(path: Path, row: int, rows: int):
    """One logits row as float32, dtype read off the file's byte count rather than assumed."""
    import numpy as np
    import torch
    raw = path.read_bytes()
    for name, item in SIZES.items():
        if len(raw) == rows * VOCAB * item:
            dt = {"f32": np.float32, "bf16": None}[name]
            if dt is not None:
                a = np.frombuffer(raw, dtype=dt)[row * VOCAB:(row + 1) * VOCAB]
                return torch.from_numpy(a.astype(np.float32).copy()), name
            u16 = np.frombuffer(raw, dtype=np.uint16)[row * VOCAB:(row + 1) * VOCAB]
            return torch.from_numpy(u16.copy()).view(torch.bfloat16).float(), name
    raise SystemExit(f"{path}: {len(raw)} bytes is none of "
                     + ", ".join(f"{rows * VOCAB * i} ({n})" for n, i in SIZES.items())
                     + f" for {VOCAB}-wide rows; refusing to guess a dtype (AGENTS' dump-dtype row)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--row-a", type=int, default=0)
    ap.add_argument("--row-b", type=int, default=0)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--rows-b", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--ulp-max", type=int, default=1)
    args = ap.parse_args()

    a, da = load(Path(args.a), args.row_a, args.rows)
    b, db = load(Path(args.b), args.row_b, args.rows_b)
    r = bf16_ulp.report(a, b, ulp_max=args.ulp_max)
    am = int(a.argmax()), int(b.argmax())
    print(f"{args.label or 'compare'}: [{da} vs {db}] n={r['n']} ndiff={r['ndiff']}"
          f" maxulp={r['maxulp']} over1={r['over1']} n_bad={r['n_bad']}"
          f" worst_bad_ulp={r['worst_bad_ulp']} n_band={r['n_band']}"
          f" 1-cos={1 - r['cos']:.3e} meanrel={r['meanrel']:.3e}"
          f" argmax={am[0]}/{am[1]}{' SAME' if am[0] == am[1] else ' DIFFER'}"
          f" toplogit={float(a.max()):.6f}/{float(b.max()):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
