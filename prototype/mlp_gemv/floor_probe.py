#!/usr/bin/env python3
"""Where is the floor? Reorder the *reference's own* math and see how far BF16 can move.

`run.py` gates the MLP kernel in BF16 ULPs, so it needs a threshold that is not a guess. The
kernel's only license is "same products, F32 accumulation, different summation order", so the
floor is: how far can the reference's own result move when only the summation order changes?

For every layer and every decode step this takes the pinned operands and recomputes each dot
in four orders — torch's bf16 matmul (the reference), numpy F32, a blocked-pairwise F32 sum,
and `math.fsum` (exactly rounded, so it is the truth the others approximate) — and reports the
worst BF16 ULP distance between any two of them. Two things fall out:

  * the distribution of distances, which sizes the per-output threshold;
  * whether a large distance ever lands on an element whose magnitude is a tiny fraction of
    the tensor max. It does: a near-zero dot is a cancellation, its absolute F32 error is
    ordinary but its BF16 ULP distance explodes. That is why the gate in run.py is
    scale-aware (ULP on the big elements, `|Δ| ≤ tol × max|ref|` on the small ones) instead
    of a flat ULP bound that a correct kernel can trip by being unlucky.

    .venv/bin/python prototype/mlp_gemv/floor_probe.py [--fixture dec8] [--layers all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402


def ulp_dist(a, b) -> np.ndarray:
    return bf16_ulp.ulp_dist(a, b)


def blocked_sum(A: np.ndarray, x: np.ndarray, block: int = 128) -> np.ndarray:
    """Sum in blocks of `block`, then pairwise across the block sums: a different, legitimate
    F32 order (roughly what a warp-shuffle reduction does). Two ways to get this wrong, both
    found in the first version of this function: broadcasting a (rows,) block sum onto an
    (rows, nblocks) accumulator adds it to every column and returns the dot times a power of
    two; and folding an odd number of columns silently drops one (in=3584 gives 28 blocks,
    then 14, 7, and a naive half-fold throws one away). Pad to a power of two and fill
    columns explicitly."""
    n = x.shape[0]
    nblocks = n // block
    acc = np.empty((A.shape[0], nblocks), dtype=np.float32)
    for b in range(nblocks):
        s = b * block
        acc[:, b] = (A[:, s:s + block] * x[None, s:s + block]).sum(axis=1, dtype=np.float32)
    width = 1 << (nblocks - 1).bit_length()          # zero pad: 28 -> 32
    if width != nblocks:
        acc = np.concatenate([acc, np.zeros((A.shape[0], width - nblocks), np.float32)], 1)
    while acc.shape[1] > 1:
        h = acc.shape[1] // 2
        acc = acc[:, :h] + acc[:, h:2 * h]
    return acc[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--step", default=None)
    ap.add_argument("--perturb", type=int, default=1,
                    help="also measure how far a 1-ULP error in `act` propagates through "
                         "down_proj, with this many injected elements per trial")
    args = ap.parse_args()
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the floor cannot be measured")

    from safetensors import safe_open
    CKPT = ROOT / "models" / "Qwen3.5-0.8B"
    key = {}
    for f in sorted(CKPT.glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                key[k] = s.get_tensor(k)

    d = ROOT / "data" / "refs" / args.fixture
    steps = [args.step] if args.step else sorted(
        {p.name.split(".")[0] for p in d.glob("decode*.L00.mlp.in.npy")})
    layers = range(24) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    pre = "model.language_model.layers.%d.mlp."

    hist = {k: {"max": 0, "over1": 0, "over2": 0, "n": 0, "worst_frac_of_max": 1.0,
                "worst_at": None, "per_variant": {}} for k in ("gate_pre", "out")}
    for l in layers:
        for step in steps:
            p = "%s.L%02d.mlp." % (step, l)
            if not (d / (p + "in.npy")).exists():
                continue
            refs = {k: np.load(d / (p + k + ".npy")).astype(np.float32).reshape(-1)
                    for k in ("in", "gate_pre", "act", "out")}
            specs = {"gate_pre": (key[pre % l + "gate_proj.weight"], refs["in"]),
                     "out": (key[pre % l + "down_proj.weight"], refs["act"])}
            for nm, (W, x) in specs.items():
                Wf = W.float().numpy()
                ref = torch.from_numpy(refs[nm])
                h = hist[nm]
                variants = [
                    ("f32-einsum", torch.from_numpy(
                        np.einsum("ij,j->i", Wf, x, dtype=np.float32))),
                    ("f32-blocked", torch.from_numpy(blocked_sum(Wf, x))),
                    # The nearest thing to the truth: bf16 x bf16 products are exact in F64 and
                    # 3584 of them sum without a visible F64 rounding error, so this is the
                    # correctly-rounded dot that both torch and the kernel are approximating.
                    ("f64-exact", torch.from_numpy(
                        np.einsum("ij,j->i", W.double(), x.astype(np.float64)).astype(np.float32))),
                ]
                scale = float(np.abs(refs[nm]).max())
                for vname, v in variants:
                    dd = ulp_dist(v, ref)
                    pv = h.setdefault("per_variant", {})
                    pv[vname] = {"max": max(pv.get(vname, {}).get("max", 0), int(dd.max())),
                                 "over1": pv.get(vname, {}).get("over1", 0) + int((dd > 1).sum()),
                                 "n": pv.get(vname, {}).get("n", 0) + dd.size}
                    i = int(dd.argmax())
                    if int(dd.max()) > h["max"]:
                        h["max"] = int(dd.max())
                        h["worst_frac_of_max"] = float(abs(refs[nm][i]) / scale)
                        h["worst_at"] = "L%02d %s %s via %s" % (l, step, nm, vname)
                    h["over1"] += int((dd > 1).sum())
                    h["over2"] += int((dd > 2).sum())
                    h["n"] += dd.size

    print("fixture %s | steps %s | layers %s" % (args.fixture, len(steps), args.layers))
    for nm, h in hist.items():
        print("\n%s: reordering the reference's own F32 sum moves" % nm)
        print("   worst distance      %d ULP (over %d comparisons)" % (h["max"], h["n"]))
        print("   >1 ULP              %d / %d (%.4f%%)   >2 ULP: %d"
              % (h["over1"], h["n"], 100.0 * h["over1"] / h["n"], h["over2"]))
        print("   worst element |ref|/max|ref| = %.2e  (%s)"
              % (h["worst_frac_of_max"], h["worst_at"]))
        for vname, pv in sorted(h.get("per_variant", {}).items()):
            print("      %-12s max %5d ULP | >1 ULP: %d / %d (%.4f%%)"
                  % (vname, pv["max"], pv["over1"], pv["n"], 100.0 * pv["over1"] / pv["n"]))
    print("\nRead: a large ULP distance is a cancellation, not an error — it lands where "
          "|ref| is a rounding-critical fraction of the tensor max. A gate that forbids it "
          "outright forbids correctness.")

    if args.perturb:
        # Chain sensitivity: F4 feeds the kernel's own `act` into the down GEMV, so a legal
        # 1-ULP difference in act is not allowed to produce a 1-ULP difference in out. Inject
        # k one-ULP errors into the reference act and measure what down_proj does with them.
        print("\nact -> out propagation, %d injected 1-ULP elements per trial:" % args.perturb)
        rng = np.random.default_rng(7)
        worst = {"ulp": 0, "frac": 1.0, "where": None}
        for l in layers:
            for step in steps:
                p = "%s.L%02d.mlp." % (step, l)
                if not (d / (p + "act.npy")).exists():
                    continue
                act = torch.from_numpy(np.load(d / (p + "act.npy"))).to(torch.bfloat16)
                out_ref = torch.from_numpy(np.load(d / (p + "out.npy"))).to(torch.bfloat16)
                Wd = key[pre % l + "down_proj.weight"]
                for _ in range(4):
                    idx = rng.choice(act.numel(), args.perturb, replace=False)
                    flat = act.view(-1).clone()
                    # step each chosen element exactly one representable BF16 up, via the
                    # monotone key (np.nextafter in F32 would change a value that then rounds
                    # back to the same BF16, i.e. inject nothing at all)
                    k = bf16_ulp.monotone_key(bf16_ulp.bf16_bits(flat[idx])) + 1
                    kb = np.asarray(bf16_ulp.bits_from_key(k), dtype=np.uint16)
                    flat[idx] = torch.from_numpy(np.ascontiguousarray(kb.view(np.uint16))).view(
                        torch.bfloat16)
                    got = flat.reshape(act.shape) @ Wd.t()
                    dd = ulp_dist(got, out_ref)
                    scale = float(out_ref.float().abs().max())
                    i = int(dd.argmax())
                    if int(dd.max()) > worst["ulp"]:
                        worst = {"ulp": int(dd.max()),
                                 "frac": float(abs(out_ref.float().reshape(-1)[i].numpy())
                                               / scale),
                                 "where": "L%02d %s" % (l, step)}
                    nd = int((dd > 1).sum())
                    if nd:
                        print("   L%02d %s: %d/%d out elements >1 ULP (max %d) from %d "
                              "1-ULP act flips" % (l, step, nd, dd.size, int(dd.max()),
                                                   args.perturb))
                        break
            else:
                continue
            break
        print("   worst seen anywhere: %d ULP at |ref|/max = %.1e (%s)"
              % (worst["ulp"], worst["frac"], worst["where"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
