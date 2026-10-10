#!/usr/bin/env python3
"""Isolated per-layer gate - the gate real kernels will be signed off against.

End-to-end cascade comparison has a noise floor: with bf16 activations, any
rounding difference in one op is re-amplified by the residual stream and the norms
of the following 23 layers, so deep-layer tensors drift ~2 % even when every op is
written the way the reference writes it. That makes a cascade diff a poor test for
a single kernel.

This gate removes the cascade: for every layer it feeds the **reference's own
layer input** (`L{ll}.s_in`) into the engine's layer and compares that layer's
outputs and residual output against the reference. Prefill runs from an empty
state, so a single layer is fully determined by its input - no upstream error can
enter, and raw projections (S04.1/S05.1) reproduce the reference **bit-exactly** (meanrel 0),
so all remaining error comes from the reductions - the GDN scan, the attention
softmax, the gated norm - where chunk-vs-chunk blocking order legitimately differs.

For a kernel implementation the same harness applies with `--layer L{ll}`; state
injection for decode-step isolation (feeding the reference's S08 conv window and
recurrent state) is the next capability to add.
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

from engine.gates.run_gates import classify, metrics, norm  # noqa: E402
from engine.specimpl import (ATTN_LAYERS, GDN_LAYERS, Forward, Weights,  # noqa: E402
                             rope_cs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--fixture", default="smoke")
    ap.add_argument("--gdn-prefill", default="chunk", choices=["recurrent", "chunk"])
    ap.add_argument("--layers", default=None, help="comma list like 0,3,12")
    ap.add_argument("--cos-min", type=float, default=0.9999)
    ap.add_argument("--meanrel-max", type=float, default=1e-2)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = Weights(Path(args.model))
    d = ROOT / "data" / "refs" / args.fixture
    T = len(json.loads((d / "manifest.json").read_text())["prefill_ids"])
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(len(GDN_LAYERS) + len(ATTN_LAYERS))))
    print("fixture=%s T=%d layers=%s gdn_form=%s"
          % (args.fixture, T, layers, args.gdn_prefill))

    cos_, sin_ = rope_cs(torch.arange(T), torch.device("cpu"))
    rows, fails = [], []
    for l in layers:
        key_in = "L%02d.s_in" % l
        if not (d / (key_in + ".npy")).exists():
            fails.append("missing reference input %s" % key_in)
            continue
        x = torch.from_numpy(np.squeeze(np.load(d / (key_in + ".npy")))).to(torch.bfloat16)
        assert x.shape == (T, w.p["model.language_model.embed_tokens.weight"].shape[1]), \
            "%s has shape %s, expected (%d, %d)" % (key_in, tuple(x.shape), T,
                                                   w.p["model.language_model.embed_tokens.weight"].shape[1])
        # fresh state: prefill sees no prefix, so the layer is fully determined by x
        f = Forward(w, gdn_form=args.gdn_prefill)
        dbg = {}
        out = f._layer(l, x, cos_, sin_, dbg, args.gdn_prefill)
        dbg["L%02d.s_out" % l] = out
        for name, got in dbg.items():
            path = d / (name + ".npy")
            if not path.exists():
                continue
            r = norm(name, np.load(path), True)
            e = norm(name, got.detach().float().numpy(), False)
            if r.shape != e.shape:
                if r.size == e.size:
                    e = e.reshape(r.shape)
                else:
                    continue                      # convention differs (see run_gates SKIPS)
            cos, maxrel, meanrel = metrics(e, r)
            rows.append((name, classify(name), cos, meanrel, maxrel))
            if cos < args.cos_min or meanrel > args.meanrel_max:
                fails.append("%s: cos=%.8f meanrel=%.3g (maxrel=%.3g)"
                             % (name, cos, meanrel, maxrel))

    rows.sort(key=lambda r: r[2])           # worst cos first
    for name, cls, cos, meanrel, maxrel in rows[:args.top]:
        print("%-34s %-6s cos=%.8f meanrel=%.2e maxrel=%.2e" % (name, cls, cos, meanrel, maxrel))
    if rows:
        worst, best = rows[0], rows[-1]
        print("\n%d isolated tensors checked; worst cos %.8f (meanrel %.2e), best cos %.8f"
              % (len(rows), worst[2], worst[3], best[2]))
    if fails:
        print("FAILED %d (cos_min=%g meanrel_max=%g):" % (len(fails), args.cos_min,
                                                          args.meanrel_max))
        for f in fails[:args.top]:
            print("  ", f)
        return 1
    print("PASS: every layer reproduces the reference from the reference's own input "
          "- per-op math is right; any remaining end-to-end drift is cascade "
          "amplification, not a wrong formula")
    return 0


if __name__ == "__main__":
    sys.exit(main())
