#!/usr/bin/env python3
"""Quantify the A13 long-context divergence: knob 1 vs knob 2 at 16,384 lanes. CPU-only.

Byte-identity held at 4,096 lanes and on every pinned fixture up to 1,024 rows, and failed at 16,384. This
says HOW it failed, because the two possible stories have opposite meanings: a late first divergence with a
~1e-5 logit cosine is the bf16 chaos floor (A-18: greedy token equality decays with depth on any bf16
implementation, sdpa-vs-eager floor 0.8776), while an early divergence or a cosine materially below 1 is a
structural bug in the ported full/tail round split -- and the -26.96 % timing win needs the second story to
be true before it is a speedup rather than skipped work.

BF16 dumps are read as uint16 and widened by shifting into the top half of an f32 (exact); never
`astype(float)` on the raw bits, and torch refuses bf16.numpy() for other reasons (prototype/README.md).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp/a13ab"


def load_logits(path: Path) -> np.ndarray:
    """`--out` dumps the row's logits as raw F32 (248,320 x 4 B = 993,280 B, which is what the file is).
    The first version of this script read them as BF16 and produced NaN for every aggregate while still
    printing a confident ndiff — the dtype must be read off the file size, not assumed."""
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        print(f"{name}: SHAPE DIFFERS {a.shape} vs {b.shape} — structural, not drift")
        return
    nd = int((a != b).sum())
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    denom = np.maximum(np.abs(a), 1e-30)
    maxabs = float(np.max(np.abs(a - b)))
    maxrel = float(np.max(np.abs(a - b) / denom))
    meanrel = float(np.mean(np.abs(a - b)) / np.mean(np.abs(a)))
    print(f"{name}: n={a.size} ndiff={nd} ({100.0*nd/a.size:.3f} %)  1-cos={1.0-cos:.3e}  "
          f"max_abs={maxabs:.6g}  max_rel={maxrel:.6g}  meanrel={meanrel:.3e}  "
          f"|a|max={np.max(np.abs(a)):.4g}")
    am, bm = int(np.argmax(a)), int(np.argmax(b))
    print(f"   argmax: knob1={am} knob2={bm} {'SAME' if am == bm else 'DIFFERS'}  "
          f"|  top-5 k1={np.argsort(-a)[:5].tolist()} k2={np.argsort(-b)[:5].tolist()}")


for lanes in (4096, 16384, 32000):
    d1, d2 = OUT / f"id1_{lanes}", OUT / f"id2_{lanes}"
    if not d1.exists():
        continue
    print(f"=== lanes={lanes} ===")
    for f in sorted(d1.glob("logits*.bin")):
        g = d2 / f.name
        if g.exists():
            report(f.name, load_logits(f), load_logits(g))
    s1 = (d1 / "sampled.txt").read_text().split("\n")
    s2 = (d2 / "sampled.txt").read_text().split("\n")
    n = min(len(s1), len(s2))
    first = next((i for i in range(n) if s1[i] != s2[i]), None)
    same = sum(1 for i in range(n) if s1[i] == s2[i])
    print(f"  sampled.txt: {n} lines, {same} equal ({100.0*same/n:.2f} %), "
          f"first difference at index {first}"
          + ("" if first is None else f"  (k1={s1[first]!r} vs k2={s2[first]!r}, {n-first} lines after it)"))
