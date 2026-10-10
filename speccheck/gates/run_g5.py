#!/usr/bin/env python3
"""G5 (§10.1) — the incremental == batch invariant of §08.6.

Feeds a token sequence two ways and requires the engine state and the final
logits to agree:

  A) prefill(all tokens)                        — one batch pass
  B) prefill(prefix) then decode(rest)          — incremental

This is the invariant the real engine depends on: if it holds, the GDN recurrent
state, the conv window and the KV cache are sufficient to resume a request
mid-stream, and preemption can be a memcpy (§08.4).
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

from speccheck.specimpl import GDN_LAYERS, ATTN_LAYERS, Forward, Weights  # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1).double()
    b = b.detach().float().reshape(-1).double()
    d = a.norm() * b.norm()
    return float(a @ b / d) if d > 0 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--prompt-ids-file", default=str(ROOT / "data/refs/smoke/manifest.json"))
    ap.add_argument("--split", type=int, default=8, help="prefix length kept in prefill")
    ap.add_argument("--extend", type=int, default=5, help="tokens appended incrementally")
    ap.add_argument("--cos-min", type=float, default=0.999)
    ap.add_argument("--gdn-prefill", default="recurrent", choices=["recurrent", "chunk"],
                    help="pass the chunked form to BOTH sides; the point of the test is "
                         "that the state at a chunk boundary is the same either way")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = Weights(Path(args.model))
    ids = json.loads(Path(args.prompt_ids_file).read_text())["prefill_ids"]
    # extend the sequence deterministically: greedy continuation from the batch pass
    ref = Forward(w, gdn_form=args.gdn_prefill)
    gen = ref.greedy(ids, max(args.extend, 1))
    seq = ids + gen[:args.extend]
    T = min(args.split, len(seq) - 1)
    print(f"seq={len(seq)} tokens, prefill {T} then {len(seq)-T} incremental steps "
          f"(gdn prefill form={args.gdn_prefill})")

    a = Forward(w, gdn_form=args.gdn_prefill)
    na = a.logits(a.prefill(torch.tensor(seq)))
    b = Forward(w, gdn_form=args.gdn_prefill)
    b.prefill(torch.tensor(seq[:T]))
    for tok in seq[T:]:
        nb = b.logits(b.decode(tok))

    fails = []
    print(f"{'state':<26} {'cos':>10}  ok")
    for l in GDN_LAYERS:
        for name, x, y in (("gdn.state", a.rec[l], b.rec[l]),
                           ("gdn.conv_win", a.conv_win[l], b.conv_win[l])):
            if x is None or y is None:
                fails.append(f"L{l:02d}.{name}: missing")
                continue
            c = cos(x, y)
            print(f"L{l:02d}.{name:<16} {c:>10.6f}  {'ok' if c >= args.cos_min else 'FAIL'}")
            if c < args.cos_min:
                fails.append(f"L{l:02d}.{name}: cos={c:.6f}")
    for l in ATTN_LAYERS:
        for name in ("k", "v"):
            x, y = torch.cat(a.kv[l][name], 0), torch.cat(b.kv[l][name], 0)
            if x.shape != y.shape:
                fails.append(f"L{l:02d}.cache_{name}: {tuple(x.shape)} != {tuple(y.shape)}")
                continue
            c = cos(x, y)
            print(f"L{l:02d}.cache_{name:<11} {c:>10.6f}  {'ok' if c >= args.cos_min else 'FAIL'}")
            if c < args.cos_min:
                fails.append(f"L{l:02d}.cache_{name}: cos={c:.6f}")
    if a.P != b.P:
        fails.append(f"prefix_len {a.P} != {b.P}")
    else:
        print(f"prefix_len            {a.P:>10}  ok")
    c = cos(na[-1], nb[-1])
    same_tok = int(na[-1].argmax()) == int(nb[-1].argmax())
    print(f"{'final logits':<26} {c:>10.6f}  {'ok' if c >= args.cos_min and same_tok else 'FAIL'}"
          f"   argmax {int(na[-1].argmax())} / {int(nb[-1].argmax())}")
    if c < args.cos_min or not same_tok:
        fails.append(f"logits: cos={c:.6f} argmax match={same_tok}")

    if fails:
        print(f"\nG5 FAILED {len(fails)}:")
        for f in fails[:15]:
            print("  ", f)
        return 1
    print("\nG5 PASS: incremental decode reproduces the batch pass (state + logits + argmax)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
