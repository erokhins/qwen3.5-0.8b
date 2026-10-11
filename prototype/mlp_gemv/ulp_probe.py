#!/usr/bin/env python3
"""What may a GEMV differ by? Measure the reference's own rounding floor for §03.3.

The HIP kernel accumulates each dot product in F32 in a warp-shuffle order that is not
torch's order, so it cannot be expected to reproduce `x @ W.T` bit for bit even though
both round once to BF16 at the end. This probe measures the floor before the kernel
exists, so the gate in `run.py` is sized against a number and not against a hope.

Three questions, all on real reference data (the pinned `L{ll}.mlp.*` operands):

1. Do the pins mean what §03.3 says they mean? Recompute gate_pre/up_pre/act/out from
   `mlp.in` + checkpoint weights with speccheck's own `mlp()` rounding policy and check
   **bit equality** with the dumps. If this fails the pins are not what I think they are.
2. How many elements move when the same dot is summed in a different order (numpy F32
   matmul of the same values, rounded to BF16 once)? That is the kernel's floor.
3. Which rounding policy does the reference use for the SwiGLU epilogue: BF16 values in
   and out of `silu`, then a BF16 multiply, or F32 throughout (the A-07 reading)? The
   answer picks the kernel's epilogue, and §03.3 is ambiguous enough to need measurement.

    .venv/bin/python prototype/mlp_gemv/ulp_probe.py [--fixture dec8] [--layers 0,11,23]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CKPT = ROOT / "models" / "Qwen3.5-0.8B"


def ckpt_tensor(key: str) -> torch.Tensor:
    for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            if key in s.keys():
                return s.get_tensor(key)
    raise KeyError(key)


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    """BF16 payload as uint16, so equality is bit equality."""
    return t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()


def ulp_report(got: torch.Tensor, ref: torch.Tensor) -> str:
    """Compare as BF16 bit patterns and report distance in ULPs of BF16."""
    g = bf16_bits(got).astype(np.int32)
    r = bf16_bits(ref).astype(np.int32)
    # BF16 is IEEE-truncated-fp32-in-miniature: for same-sign values the integer distance
    # of the bit patterns is the ULP distance. Sign-flip to a monotone order first.
    gm = np.where(g < 0, 0 - g, g ^ 0x8000)          # uint16 -> monotone int
    rm = np.where(r < 0, 0 - r, r ^ 0x8000)
    d = np.abs(gm - rm)
    n = d.size
    return ("ndiff %d/%d (%.3f%%) | max %d ULP | >1 ULP: %d | cos %.8f"
            % (int((d > 0).sum()), n, 100.0 * (d > 0).mean(), int(d.max()),
               int((d > 1).sum()),
               float(got.float().flatten() @ ref.float().flatten()
                     / (got.float().flatten().norm() * ref.float().flatten().norm()))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--step", default="decode0")
    ap.add_argument("--layers", default="0,3,11,23")
    args = ap.parse_args()

    d = ROOT / "data" / "refs" / args.fixture
    ids = json.loads((d / "manifest.json").read_text())["prefill_ids"]
    print("fixture %s (prefill %d tokens) step %s layers %s"
          % (args.fixture, len(ids), args.step, args.layers))
    pre = "model.language_model.layers.%d.mlp."

    for l in [int(x) for x in args.layers.split(",")]:
        p = "%s.L%02d.mlp." % (args.step, l)
        x = torch.from_numpy(np.load(d / (p + "in.npy"))).to(torch.bfloat16)   # [1,1,1024]
        gate_ref = torch.from_numpy(np.load(d / (p + "gate_pre.npy"))).to(torch.bfloat16)
        up_ref = torch.from_numpy(np.load(d / (p + "up_pre.npy"))).to(torch.bfloat16)
        act_ref = torch.from_numpy(np.load(d / (p + "act.npy"))).to(torch.bfloat16)
        out_ref = torch.from_numpy(np.load(d / (p + "out.npy"))).to(torch.bfloat16)
        Wg = ckpt_tensor(pre % l + "gate_proj.weight")
        Wu = ckpt_tensor(pre % l + "up_proj.weight")
        Wd = ckpt_tensor(pre % l + "down_proj.weight")

        # (1) pins vs a fresh recompute in speccheck's policy
        gate = x @ Wg.t()
        up = x @ Wu.t()
        act = torch.nn.functional.silu(gate) * up
        out = act @ Wd.t()
        print("\nL%02d  shapes x%s gate%s out%s" % (l, tuple(x.shape), tuple(gate.shape),
                                                    tuple(out.shape)))
        for nm, got, ref in (("gate_pre", gate, gate_ref), ("up_pre", up, up_ref),
                             ("act", act, act_ref), ("out", out, out_ref)):
            bit = "BIT-EQUAL" if np.array_equal(bf16_bits(got), bf16_bits(ref)) else "differs"
            print("  recompute %-8s %s | %s" % (nm, bit, ulp_report(got, ref)))

        # (2) reordered F32 accumulation: what a warp-shuffle kernel will do
        xf = x.float().reshape(-1)
        gate_re = torch.from_numpy((Wg.float().numpy() @ xf.numpy()).astype(np.float32)
                                   ).to(torch.bfloat16).reshape(gate_ref.shape)
        print("  reordered-F32 gate_pre     %s" % ulp_report(gate_re, gate_ref))
        act_re = torch.nn.functional.silu(gate_re) * up
        out_re = act_re @ Wd.t()
        print("  -> propagated through act/out: %s" % ulp_report(out_re, out_ref))

        # (3) epilogue rounding policy: bf16 silu then bf16 mul, vs everything in F32
        act_f32 = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
        print("  act policy bf16(silu)xbf16   %s" % ulp_report(act, act_ref))
        print("  act policy all-f32 (A-07)    %s" % ulp_report(act_f32, act_ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
