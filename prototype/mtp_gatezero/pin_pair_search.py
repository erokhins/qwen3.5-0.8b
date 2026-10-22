#!/usr/bin/env python3
"""Which pinned file is the lm_head's input? A pair search, not a filename assumption.

`§10.5` (bare dump names are last-writer-wins) and `prototype/lm_head_gemv/results.md` (the bare pair
reproduces at cos 0.25) say a name cannot be trusted as "the input that produced that output". The
`final_hidden` / `pre_final_hidden` files are exactly such a bare pair, and the S09.1 drafter input
(`h` = the vector the lm_head saw) rests on it, so this script decides the question the only way it
can be decided: build every candidate x, multiply by the tied table, and report the cos against every
pinned logits row in the fixture. Whatever reproduces a pinned row at cos ~ 1 IS that row's input.

    .venv/bin/python prototype/mtp_gatezero/pin_pair_search.py [--fixture dec8]
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
    man = json.loads((d / "manifest.json").read_text())
    pins = man.get("pins", {})

    with safe_open(str(SHARD), framework="pt") as f:
        keys = set(f.keys())
        nk = [k for k in NORM_KEYS if k in keys]
        gamma = f.get_tensor(nk[0]).float() if nk else None
        E = f.get_tensor("model.language_model.embed_tokens.weight")
    print(f"norm weight key: {nk[:1]}   table {tuple(E.shape)}")

    cands: dict[str, torch.Tensor] = {}
    for name in ("final_hidden", "pre_final_hidden"):
        p = d / f"{name}.npy"
        if not p.exists():
            continue
        v = decode_pin(p, pins.get(name, {}).get("dtype", "float32"))
        cands[name] = v
        if gamma is not None:
            x = v.float()
            cands[f"gamma*RMSnorm({name})"] = (gamma * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)))).reshape(-1)
            cands[f"RMSnorm({name})"] = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))).reshape(-1)

    logits = {p.name[:-len(".npy")]: decode_pin(p, pins.get(p.name[:-len('.npy')], {}).get("dtype", "float32")).reshape(-1)
              for p in sorted(d.glob("*logits_last.npy"))}
    print(f"candidate x's: {list(cands)}")
    print(f"pinned logits rows: {list(logits)}\n")

    V = E.shape[0]
    best = (-1.0, None, None)
    for xn, xv in cands.items():
        xv = xv.reshape(-1)
        if xv.numel() != E.shape[1]:
            print(f"  {xn:<34} width {xv.numel()} != {E.shape[1]}, skipped")
            continue
        pred = torch.empty(len(logits), V, dtype=torch.float32)
        Eb = E.float()
        for i in range(0, V, 32768):
            pred[:, i:i + 32768] = Eb[i:i + 32768] @ xv
        for ln, lg in logits.items():
            n = min(lg.numel(), V)
            cos = float(torch.nn.functional.cosine_similarity(pred[:, :n], lg[:n].unsqueeze(0), dim=-1).mean())
            top = ", " if cos < 0.9 else "  <<<"
            print(f"  {xn:<34} x {ln:<28} cos={cos:.9f}{top}")
            if cos > best[0]:
                best = (cos, xn, ln)
    print(f"\nbest pair: {best[1]} x {best[2]} at cos={best[0]:.9f}")
    if best[0] < 0.9999:
        print("NO pair reproduces. The fixture does not contain the lm_head input/output pair under")
        print("these names, so any 'h is the vector the lm_head sees' claim must be re-derived.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
