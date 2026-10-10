#!/usr/bin/env python3
"""G6 (S10.1) - online (block-wise, (max, sum)) softmax == one torch.softmax.

A decode attention reduction over a long prefix is written as a blocked loop with
`(max, sum)` rescaling - online softmax, the thing flash-decoding / split-K kernels
do - instead of one materialized `softmax` over P lanes. Mathematically identical;
the only question is whether the blocked F32 summation changes any decision.

What this gate is NOT about: accumulators carried between decode steps. Each decode
step has a new query, so its softmax is a new reduction; per-step accumulators are
reduction state, not request state. An implementation that carried them across steps
was tried and is wrong (attention cos 0.876 after one step, output matched the
previous prefix); A-17 records it.

Requirements: sampled ids identical, logits cos >= cos_min, and the two paths must
not be bit-identical everywhere (if they are, the online branch probably never ran
and the test is vacuous).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speccheck.specimpl import Forward, Weights  # noqa: E402


def cos_rel(b, a):
    b = b.detach().float().reshape(-1).double()
    a = a.detach().float().reshape(-1).double()
    return (float(b @ a / (b.norm() * a.norm() + 1e-30)),
            float((b - a).abs().max() / (a.abs().max() + 1e-30)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--fixtures", default="dec8,p300")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gdn-prefill", default="chunk", choices=["recurrent", "chunk"])
    ap.add_argument("--cos-min", type=float, default=0.9999,
                    help="logit-level floor; bf16 re-amplifies summation order through "
                         "23 layers (see S10.2), so the binding requirement is identical ids")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = Weights(Path(args.model))
    fails = []
    for tag in [t for t in args.fixtures.split(",") if t]:
        ids = json.loads((ROOT / "data" / "refs" / tag / "manifest.json").read_text()
                         )["prefill_ids"]
        res = {}
        for mode in ("full", "online"):
            f = Forward(w, gdn_form=args.gdn_prefill, attn_softmax=mode)
            dp, dd = {}, {}
            gen = f.greedy(ids, args.steps, dp, dd, stop_on_eos=False)
            res[mode] = (gen, dp, dd)
        same_ids = res["full"][0] == res["online"][0]
        worst, diffs, ndiff = (1.0, 0.0), 0, 0
        for step in sorted(res["full"][2], key=lambda s: int(s[6:])):
            a = res["full"][2][step]["logits_last"]
            b = res["online"][2][step]["logits_last"]
            c, mx = cos_rel(b, a)
            ndiff += int(c < 1.0)
            if c < worst[0]:
                worst = (c, mx)
            for k in res["full"][2][step]:
                x, y = res["full"][2][step][k], res["online"][2][step][k]
                if x.shape == y.shape and x.numel() > 8:
                    cc, mm = cos_rel(y, x)
                    diffs = max(diffs, mm)
        print("%-6s steps=%d  ids identical=%s  worst decode logits cos=%.8f "
              "(maxrel %.1e)  attention maxrel=%.1e  logits differing in %d/%d steps"
              % (tag, args.steps, same_ids, worst[0], worst[1], diffs, ndiff,
                 len(res["full"][2])))
        if not same_ids:
            fails.append("%s: ids differ: full=%s online=%s"
                         % (tag, res["full"][0][:8], res["online"][0][:8]))
        if worst[0] < args.cos_min:
            fails.append("%s: logits cos %.8f < %g" % (tag, worst[0], args.cos_min))
        if ndiff == 0:
            fails.append("%s: every step bit-identical - the online branch may not "
                         "have run (vacuous test)" % tag)
        if diffs == 0.0:
            fails.append("%s: no attention difference at all - suspicious" % tag)
    if fails:
        print("G6 FAILED:")
        for f in fails:
            print("  ", f)
        return 1
    print("G6 PASS: blocked online reduction reproduces the reference softmax within "
          "F32 rounding; every sampled id identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
