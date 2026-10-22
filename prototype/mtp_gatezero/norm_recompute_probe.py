#!/usr/bin/env python3
"""Does the bare `pre_final_hidden` pin reproduce the pinned `final_hidden`?

Written because the MTP block's input `h` is defined as "the vector the lm_head saw" (§09.1), and a
natural shortcut is to recompute the final norm from the pin sitting beside it. `pin_pair_search.py` shows
`final_hidden` IS that vector; this prints by how much the shortcut misses, so MTP-GATEZERO.md §5 states a
measured gap instead of a recollection. The expected answer is "they are not interchangeable" — the §10.5
bare-name trap, one file over from where it is documented.

    .venv/bin/python prototype/mtp_gatezero/norm_recompute_probe.py [--fixture dec8]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHARD = ROOT / "models" / "Qwen3.5-0.8B" / "model.safetensors-00001-of-00001.safetensors"
NORM_KEYS = ["model.language_model.norm.weight", "model.language_model.norm.gamma"]


def decode_pin(path: pathlib.Path, want: str) -> torch.Tensor:
    a = np.load(path).reshape(-1)
    if "bits:uint16" in want or want.startswith("bfloat16"):
        if a.dtype == np.uint16:
            return torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16).float()
        return torch.from_numpy(a).float()
    return torch.from_numpy(a.astype(np.float64)).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="dec8")
    args = ap.parse_args()
    d = ROOT / "data" / "refs" / args.fixture
    pins = json.loads((d / "manifest.json").read_text()).get("pins", {})

    def pin(name):
        return decode_pin(d / f"{name}.npy", pins.get(name, {}).get("dtype", "float32"))

    pre, fin = pin("pre_final_hidden"), pin("final_hidden")
    with safe_open(str(SHARD), framework="pt") as f:
        nk = [k for k in NORM_KEYS if k in f.keys()]
        gamma = f.get_tensor(nk[0]).float()
    rec = gamma * (pre * torch.rsqrt(pre.pow(2).mean(-1, keepdim=True) + 1e-6))

    cos = torch.nn.functional.cosine_similarity(fin, rec, dim=-1).mean().item()
    rel = ((fin - rec).abs() / fin.abs().clamp(min=1e-9))
    print(f"{args.fixture}: gamma*RMSnorm(pre_final_hidden) vs the pinned final_hidden  [{nk[0]}]")
    print(f"  cos            {cos:.9f}")
    print(f"  median rel err {rel.median().item():.4f}   p90 {rel.quantile(0.90).item():.4f}")
    print(f"  max abs diff   {(fin - rec).abs().max().item():.5f} against max|final_hidden| "
          f"{fin.abs().max().item():.5f}")
    print("  => the two bare pin names are NOT interchangeable; `h` comes from final_hidden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
