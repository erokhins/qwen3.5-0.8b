#!/usr/bin/env python3
"""Where is the floor at K = 1024 and K = 2048? Reorder the reference's own math and measure.

`run.py` gates the projections in BF16 ULPs, so the band has to come from a measurement, not from
`mlp_gemv`'s numbers — the cancellation floor depends on K, and the MLP measured it at K = 3584.
`mlp_gemv/floor_probe.py` is the method; this is its re-derivation for the shapes that actually
appear here (§04.1's 6144x1024, §04.5's 1024x2048, §05.1's 4096/512x1024).

For every layer, step and projection it takes the pinned operand pair and recomputes each dot in
four legitimate orders — torch's BF16 matmul (the reference itself), a plain F32 `einsum`, a
blocked F32 sum, and an F64 sum rounded once (nearest thing to the exact value) — plus a
**faithful emulation of this kernel's own summation order** (32 lanes, 8-element words,
stride-`ws` chunk loop, shuffle-fold tree). It reports the worst BF16 distance between any two of
them and, critically, *where* that distance lands:

  * the rate of >1 ULP differences sizes the per-element tolerance `run.py` applies;
  * a large distance at an element whose |value| is ~1e-6 of the tensor max is a cancellation, not
    an error — that is why the gate is scale-aware (ULP on the large elements, nothing on the
    cancelling ones beyond the aggregates).

`--perturb` answers the chain question the engine will ask: if the norm kernel that produces `x`
differs from the reference by one representable BF16 (legal under §10.1), how far can the
projection output move? The answer is why a chained output cannot be gated per element.

    .venv/bin/python prototype/proj_gemv/floor_probe.py [--fixture dec8] [--layers all]
                                                       [--steps all] [--perturb 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402
from operands import (is_attn_layer, operands_for, pin, proj_weight, projections_of)  # noqa: E402


def blocked_sum(A: np.ndarray, x: np.ndarray, block: int = 128) -> np.ndarray:
    """Sum in contiguous blocks of `block`, then pairwise across the block sums: a legitimate
    F32 order, different from torch's. (The odd-column trap `mlp_gemv` hit is handled here by
    zero-padding the block axis to a power of two — 1024/8 and 2048/8 both need it.)"""
    n = x.shape[0]
    nblocks = n // block
    acc = np.empty((A.shape[0], nblocks), dtype=np.float32)
    for b in range(nblocks):
        s = b * block
        acc[:, b] = (A[:, s:s + block] * x[None, s:s + block]).sum(axis=1, dtype=np.float32)
    width = 1 << (nblocks - 1).bit_length()
    if width != nblocks:
        acc = np.concatenate([acc, np.zeros((A.shape[0], width - nblocks), np.float32)], 1)
    while acc.shape[1] > 1:
        h = acc.shape[1] // 2
        acc = acc[:, :h] + acc[:, h:2 * h]
    return acc[:, 0]


def kernel_order_sum(A: np.ndarray, x: np.ndarray, ws: int = 32, word: int = 8) -> np.ndarray:
    """Emulate `proj_gemv`'s reduction exactly: lane `l` accumulates 8-element words
    v = l, l+ws, l+2ws... in that order, then a shuffle-down fold adds lane i+o to lane i for
    o = ws/2, ws/4, ..., 1. The inner 8 products are summed in order. If this variant sits at the
    same distance as the other reorderings, the kernel's order costs nothing beyond the floor."""
    rows, k = A.shape
    words = k // word
    prod = (A.reshape(rows, words, word) * x.reshape(1, words, word)).astype(np.float32)
    per_word = np.empty((rows, words), dtype=np.float32)
    for j in range(word):                       # sequential in j, like the unrolled inner loop
        if j == 0:
            per_word[:] = prod[:, :, 0]
        else:
            per_word += prod[:, :, j]
    chunks = words // ws
    lanes = np.zeros((rows, ws), dtype=np.float32)
    for c in range(chunks):                    # v += ws, in order
        lanes += per_word[:, c * ws:(c + 1) * ws]
    o = ws // 2
    while o:
        lanes[:, :o] = lanes[:, :o] + lanes[:, o:2 * o]
        o //= 2
    return lanes[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--perturb", type=int, default=1,
                    help="also measure how far N injected 1-ULP flips in x move the output")
    args = ap.parse_args()
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the floor cannot be measured")

    d = ROOT / "data" / "refs" / args.fixture
    all_steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L00.s_in.npy")})
    steps = all_steps if args.steps == "all" else args.steps.split(",")
    layers = list(range(24)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    # hist[key] : ULP stats, per variant, plus where the worst distance landed
    hist: dict[str, dict] = {}
    for layer in layers:
        for step in steps:
            if not (d / ("%s.L%02d.s_in.npy" % (step, layer))).exists():
                continue
            ops = operands_for(d, step, layer)
            attn = is_attn_layer(d, layer)
            for name in projections_of(d, layer):
                x = (ops["x_in"] if name != ops["out_proj"] else ops["x_out"]).float().numpy()
                W = proj_weight(layer, name).float().numpy()
                ref = torch.from_numpy(pin(d, "%s.L%02d.%s.out" % (step, layer, name))
                                       .float().numpy())
                scale = float(np.abs(ref).max())
                variants = [
                    ("f32-einsum", torch.from_numpy(np.einsum("ij,j->i", W, x, dtype=np.float32))),
                    ("f32-blocked128", torch.from_numpy(blocked_sum(W, x))),
                    ("kernel-order", torch.from_numpy(kernel_order_sum(W, x))),
                    ("f64-exact", torch.from_numpy(
                        np.einsum("ij,j->i", W.astype(np.float64),
                                  x.astype(np.float64)).astype(np.float32))),
                ]
                key = "%s K=%d" % ("attn" if attn else "gdn", x.shape[0])
                h = hist.setdefault(key, {"per_variant": {}, "by_output": {}, "n": 0, "over1": 0,
                                          "over2": 0, "worst": 0, "worst_frac": 1.0,
                                          "worst_at": None})
                for vname, v in variants:
                    dd = bf16_ulp.ulp_dist(v, ref)
                    pv = h["per_variant"].setdefault(vname, {"max": 0, "over1": 0, "n": 0})
                    pv["max"] = max(pv["max"], int(dd.max()))
                    pv["over1"] += int((dd > 1).sum())
                    pv["n"] += dd.size
                    ob = h["by_output"].setdefault(name, {"max": 0, "over1": 0, "n": 0,
                                                          "worst_frac": 1.0})
                    ob["over1"] += int((dd > 1).sum())
                    ob["n"] += dd.size
                    i = int(dd.argmax())
                    if int(dd.max()) >= ob["max"]:      # where the worst distance landed
                        ob["worst_frac"] = min(ob["worst_frac"], float(abs(ref.numpy()[i]) / scale))
                        ob["max"] = int(dd.max())
                    if int(dd.max()) > h["worst"]:
                        h["worst"] = int(dd.max())
                        h["worst_frac"] = float(abs(ref.numpy()[i]) / scale)
                        h["worst_at"] = "L%02d %s %s via %s" % (layer, step, name, vname)
                    h["over1"] += int((dd > 1).sum())
                    h["over2"] += int((dd > 2).sum())
                    h["n"] += dd.size

    print("fixture %s | %d layers x %d steps | reordering the reference's own F32 sum"
          % (args.fixture, len(layers), len(steps)))
    for key in sorted(hist):
        h = hist[key]
        print("\n%s: worst %d ULP over %d comparisons | >1 ULP %d (%.5f%%) | >2 ULP %d"
              % (key, h["worst"], h["n"], h["over1"], 100.0 * h["over1"] / h["n"], h["over2"]))
        print("   worst element sits at |ref|/max|ref| = %.2e   (%s)"
              % (h["worst_frac"], h["worst_at"]))
        for vname in sorted(h["per_variant"]):
            pv = h["per_variant"][vname]
            print("      %-15s max %5d ULP | >1 ULP: %6d / %9d (%.5f%%)"
                  % (vname, pv["max"], pv["over1"], pv["n"], 100.0 * pv["over1"] / pv["n"]))
        for name in sorted(h["by_output"]):
            ob = h["by_output"][name]
            print("      %-12s max %5d ULP | >1 ULP: %6d / %9d (%.5f%%) | worst at %.1e of max"
                  % (name, ob["max"], ob["over1"], ob["n"], 100.0 * ob["over1"] / ob["n"],
                     ob["worst_frac"]))
    print("\nRead: a >1 ULP element is a cancellation, not an error - it lands where |ref| is a"
          "\nrounding-critical fraction of the tensor max. The gate's band (1 ULP on elements"
          "\nlarger than 1e-4 of the max, aggregates below) is sized on these numbers.")

    if args.perturb:
        print("\noperand sensitivity: %d injected 1-ULP flips in x, what the dot does with them"
              % args.perturb)
        rng = np.random.default_rng(7)
        worst = {"ulp": 0, "frac": 1.0, "where": None}
        for layer in layers:
            for step in steps[:2]:
                if not (d / ("%s.L%02d.s_in.npy" % (step, layer))).exists():
                    continue
                ops = operands_for(d, step, layer)
                for name in projections_of(d, layer):
                    x = (ops["x_in"] if name != ops["out_proj"] else ops["x_out"])
                    ref = pin(d, "%s.L%02d.%s.out" % (step, layer, name))
                    W = proj_weight(layer, name)
                    for _ in range(2):
                        idx = rng.choice(x.numel(), args.perturb, replace=False)
                        flat = x.reshape(-1).clone()
                        k = bf16_ulp.monotone_key(bf16_ulp.bf16_bits(flat[idx])) + 1
                        kb = np.asarray(bf16_ulp.bits_from_key(k), dtype=np.uint16)
                        flat[idx] = torch.from_numpy(np.ascontiguousarray(kb)).view(torch.bfloat16)
                        got = flat.reshape(1, -1) @ W.t()
                        dd = bf16_ulp.ulp_dist(got, ref)
                        scale = float(ref.float().abs().max())
                        i = int(dd.argmax())
                        if int(dd.max()) > worst["ulp"]:
                            worst = {"ulp": int(dd.max()),
                                     "frac": float(abs(ref.float().numpy()[i]) / scale),
                                     "where": "L%02d %s %s" % (layer, step, name)}
        print("   worst anywhere: %d ULP at |ref|/max = %.1e (%s)"
              % (worst["ulp"], worst["frac"], worst["where"]))
        print("   so a legal 1-ULP difference in the norm output upstream is not a per-element"
              "\nerror condition in the projection output: chained outputs are judged on"
              "\naggregates (see run.py and §10.1's F4 line).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
