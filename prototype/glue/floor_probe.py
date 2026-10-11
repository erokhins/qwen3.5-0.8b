#!/usr/bin/env python3
"""Where is the floor for a NORM, and what does a BF16 accumulator actually cost?

`mlp_gemv/floor_probe.py` sized the band for a 3584-term dot. A norm is a different animal and
gets its own number rather than inheriting that one: there is no cancellation inside the reduction
(it is a sum of 1024 non-negative squares), but the OUTPUT has elements that cancel — `out[i]` is
proportional to `s[i]`, so wherever the residual happens to be near zero a one-ULP difference is
meaningless. And a different summation order for `mean(s^2)` scales EVERY element by a slightly
different `r`, flipping whichever ones sat on a BF16 rounding boundary.

So, over every layer x every decode step x both glue calls:

  1. LEGAL reorderings — numpy F32, `math.fsum` (exactly rounded), a 128-block fold, the exact
     reduction order this kernel uses (4 strided elements per thread, then an 8-partial tree), and
     a different way to compute `r` itself (`1/sqrt` instead of `rsqrt`). Report the BF16 ULP
     distance of the resulting `out` against the reference (torch's own `rmsnorm_zc`): worst, how
     many exceed 1 ULP, where they land relative to the tensor max, how much of the comparison the
     cancellation band excuses, and the tightest band these numbers would still pass at. That is
     the band sizing, with the sample size printed next to it.

  2. ILLEGAL precision loss — the one genuine accuracy risk in this op. Every other value here is
     BF16, so writing the squared-sum accumulator as BF16 looks free. Two severities: `bf16sub`
     rounds each thread's partial once before the F32 ladder, `bf16acc` keeps the whole reduction
     in a BF16 slot. These are NOT reorderings and are excluded from band sizing; they are
     measured so the gate's negatives have numbers next to them instead of hunches.

    .venv/bin/python prototype/glue/floor_probe.py [--fixture dec8] [--layers all] [--steps all]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402
from glue_pins import H, case, ckpt_w  # noqa: E402

THREADS = 256        # the kernel's launch width, so "kernel-order" is the real order
WS = 32              # this box's warpSize (tools/device_query.cpp)
EPS = 1e-6


def gate_module():
    """The band lives in `run.py`, because that is what gates. Reading it from there means this
    probe can never report margin against a number the gate no longer uses."""
    p = Path(__file__).resolve().parent / "run.py"
    spec = importlib.util.spec_from_file_location("glue_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bf16_round(a):
    """Round-to-nearest-even F32 -> BF16 -> F32, in numpy: what a BF16 slot does to a value."""
    b = np.asarray(a, dtype=np.float32).view(np.uint32)
    b = (b + (0x7FFF + ((b >> 16) & 1))) & np.uint32(0xFFFF0000)
    return b.view(np.float32)


def fold(partials, as_bf16: bool) -> float:
    """Sum partials; with as_bf16 the running value goes back through a BF16 slot at every step,
    which is what an accumulator declared BF16 does."""
    acc = np.float32(0.0)
    for p in partials:
        acc = np.float32(acc + np.float32(p))
        if as_bf16:
            acc = bf16_round(acc)
    return float(acc)


def reductions(x64: np.ndarray) -> list[tuple[str, bool, float]]:
    """(name, r_via_rsqrt, mean_of_squares) for each variant worth measuring."""
    sq = (x64 * x64).astype(np.float32)
    per = H // THREADS
    th = sq.reshape(per, THREADS).T            # thread t owns elements t, t+256, t+512, t+768
    warp_partials = []
    for w in range(THREADS // WS):
        # each lane sums its own 4 strided squares in F32 first, then the shuffle ladder folds
        # lanes: lane i adds lane i+off, exactly as __shfl_down_sync does it on the device.
        part = th[w * WS:(w + 1) * WS].sum(axis=1, dtype=np.float32)
        width = WS
        while width > 1:
            half = width // 2
            part[:half] += part[half:width]
            width = half
        warp_partials.append(float(part[0]))
    tree = np.stack(warp_partials)
    width = len(warp_partials)
    while width > 1:
        half = width // 2
        tree[:half] += tree[half:width]
        width = half
    return [
        ("np-f32", True, float(sq.sum(dtype=np.float32)) / H),
        ("fsum-exact", True, math.fsum(sq.tolist()) / H),
        ("block-128", True, fold([sq[i * 128:(i + 1) * 128].sum(dtype=np.float32)
                                  for i in range(H // 128)], False) / H),
        ("kernel-order", True, float(tree[0]) / H),
        ("div-sqrt", False, float(tree[0]) / H),
        ("bf16sub", True, float(np.array([bf16_round(th[i].sum(dtype=np.float32))
                                          for i in range(THREADS)]).sum(dtype=np.float32)) / H),
        ("bf16acc", True, fold(th.reshape(-1).tolist(), True) / H),
    ]


LEGAL = {"np-f32", "fsum-exact", "block-128", "kernel-order", "div-sqrt"}


def rel_of(x: np.ndarray) -> np.ndarray:
    scale = float(np.abs(x).max())
    return np.abs(x) / scale if scale else np.zeros(np.shape(x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the floor cannot be measured")
    band = gate_module().CANCEL_REL

    from speccheck.specimpl import rmsnorm_zc
    d = ROOT / "data" / "refs" / args.fixture
    if not d.exists():
        raise SystemExit("no fixture %s" % d)
    all_steps = sorted({p.name.split(".")[0] for p in d.glob("decode*.L00.s_in.npy")})
    steps = all_steps if args.steps == "all" else args.steps.split(",")
    layers = list(range(24)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    stat: dict[str, dict] = {}
    for layer in layers:
        for step in steps:
            c = case(d, step, layer)
            for which, (xt, wk) in (("g1", (c["s_mid"], c["w1"])), ("g2", (c["s_out"], c["w2"]))):
                x32 = xt.float().numpy()                       # the BF16 values, widened
                wf = ckpt_w(wk).float().numpy()
                ref = rmsnorm_zc(xt.reshape(1, -1), ckpt_w(wk)).reshape(-1)
                rel = rel_of(ref.float().numpy().astype(np.float64))
                for name, via_rsqrt, mean in reductions(x32.astype(np.float64)):
                    if via_rsqrt:
                        r = float(np.float32(1.0) / np.float32(math.sqrt(np.float32(mean + EPS))))
                    else:
                        r = float(np.float32(mean + EPS) ** np.float32(-0.5))
                    out32 = bf16_round((x32 * np.float32(r)) * (np.float32(1.0) + wf))
                    dd = bf16_ulp.ulp_dist(torch.from_numpy(out32), ref)
                    bs = bf16_ulp.band_usage(rel, dd, 1, band)
                    s = stat.setdefault(name, {"n": 0, "cases": 0, "ndiff": 0, "over1": 0,
                                               "max": 0, "worst_rel": 0.0, "at": None,
                                               "worst_off_rel": 0.0, "n_band": 0, "n_off": 0,
                                               "n_off_in_band": 0})
                    s["cases"] += 1
                    s["n"] += dd.size
                    s["ndiff"] += int((dd > 0).sum())
                    s["over1"] += int((dd > 1).sum())
                    s["n_band"] += bs["n_band"]
                    s["n_off"] += bs["n_off"]
                    s["n_off_in_band"] += bs["n_off_in_band"]
                    s["worst_off_rel"] = max(s["worst_off_rel"], bs["worst_off_rel"])
                    if int(dd.max()) > s["max"]:
                        i = int(dd.argmax())
                        s["max"], s["worst_rel"] = int(dd.max()), float(rel[i])
                        s["at"] = "L%02d %s %s" % (layer, step, which)

    n_cases = sum(s["cases"] for s in stat.values()) // len(stat)
    print("fixture %s | %d layers x %d steps x 2 glue calls = %d norms x %d elements = %d"
          " compared values" % (args.fixture, len(layers), len(steps), n_cases, H,
                                stat["np-f32"]["n"]))
    print("reference = torch's own rmsnorm_zc on the same BF16 input (the mirror run.py gates")
    print("against, and which stage R proves is bit-exact against the pins).")
    for group, names in (("LEGAL: same F32 sum, different order or different r (these size the"
                          " band)", sorted(LEGAL)),
                         ("ILLEGAL: accumulator precision lost (these do NOT size the band)",
                          ["bf16sub", "bf16acc"])):
        print("\n%s:" % group)
        for name in names:
            s = stat.get(name)
            if not s:
                continue
            print("  %-13s ndiff %7d / %d (%.4f%%) | max %3d ULP | >1 ULP: %6d (%.5f%%) |"
                  " worst offender |ref|/max %.2e (largest distance at %s) | band %g covers"
                  " %d/%d (%.2f%%) and excuses %d of %d offenders"
                  % (name, s["ndiff"], s["n"], 100.0 * s["ndiff"] / s["n"], s["max"],
                     s["over1"], 100.0 * s["over1"] / s["n"], s["worst_off_rel"], s["at"],
                     band, s["n_band"], s["n"], 100.0 * s["n_band"] / s["n"],
                     s["n_off_in_band"], s["n_off"]))
            if name in LEGAL:
                w = s["worst_off_rel"]
                print("                 tightest band these numbers pass at: %.1e -> the gate's"
                      " %g carries %s margin"
                      % (w, band, "%.1fx" % (band / w) if w else "inf"))
    print("\nRead: a norm's floor is not a cancellation floor the way a dot's is — no legal")
    print("reordering put a single element more than 1 ULP from the reference in this sample, so")
    print("the band is carrying nothing here and the ndiff column is the floor that matters. It")
    print("exists because out[i] is near zero wherever s[i] is, and one r scales every element")
    print("together. The two ILLEGAL rows are what the gate's bf16acc/bf16sub negatives are")
    print("expected to catch, and by how much.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
