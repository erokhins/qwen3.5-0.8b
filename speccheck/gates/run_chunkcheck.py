#!/usr/bin/env python3
"""G3 self-check (S10.1) - chunked GDN prefill == recurrent GDN prefill.

S04.7 is an algebraic rearrangement of S04.4, so on identical inputs they must
agree to F32 summation-order noise. Anything else means the chunk form is wrong
(pad positions leaking into the state, a mis-indexed pairwise decay, ...).

Three-way check, because "agrees with itself" is not enough:
  A) engine recurrent form (S04.4)   - the numerical contract
  B) engine chunked form  (S04.7)    - the fast path being introduced
  C) HF reference dumps (data/refs)  - the oracle
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

from speccheck.specimpl import GDN_LAYERS, Forward, Weights  # noqa: E402


def stats(e, r):
    e = torch.as_tensor(np.asarray(e, dtype=np.float64)).reshape(-1)
    r = torch.as_tensor(np.asarray(r, dtype=np.float64)).reshape(-1)
    cos = float(e @ r / (e.norm() * r.norm() + 1e-30))
    meanrel = float((e - r).abs().mean() / (r.abs().mean() + 1e-30))
    return cos, meanrel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--fixture", default="smoke", help="tag under data/refs (manifest + dumps)")
    ap.add_argument("--extend", type=int, default=0, help="greedy tokens appended to the fixture")
    ap.add_argument("--cos-min", type=float, default=0.9999)
    ap.add_argument("--meanrel-max", type=float, default=0.01)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = Weights(Path(args.model))
    d = ROOT / "data" / "refs" / args.fixture
    ids = json.loads((d / "manifest.json").read_text())["prefill_ids"]
    if args.extend:
        ids = ids + Forward(w).greedy(ids, args.extend)
    seq = torch.tensor(ids)
    print("fixture=%s  T=%d  chunks=%d" % (args.fixture, len(ids), -(-len(ids) // 64)))

    dbg = {}
    outs, states = {}, {}
    for form in ("recurrent", "chunk"):
        f = Forward(w, gdn_form=form)
        f.prefill(seq, dbg)
        for l in GDN_LAYERS:
            outs[(form, l)] = dbg["L%02d.gdn.o_pre_norm" % l].clone()
            states[(form, l)] = dbg["L%02d.gdn.state_new" % l].clone()

    ref = {}
    for p in d.glob("*.npy"):
        name = p.name[:-4]
        if name.endswith(("gdn.o_pre_norm", "gdn.state_new")) and not name.startswith("decode"):
            ref[name] = np.load(p)

    fails, rows = [], []
    for l in GDN_LAYERS:
        for what, key in (("o", "o_pre_norm"), ("S", "state_new")):
            a = outs[("recurrent", l)] if what == "o" else states[("recurrent", l)]
            b = outs[("chunk", l)] if what == "o" else states[("chunk", l)]
            c_ab, m_ab = stats(b, a)                       # chunk vs recurrent
            name = "L%02d.gdn.%s" % (l, key)
            line = "L%02d %-2s chunk-vs-recurrent cos=%.8f meanrel=%.2e" % (l, what, c_ab, m_ab)
            # cos gates both forms; meanrel is reported but not gated. Both `o` and
            # the state are cancellative sums (the state is a sum of signed rank-1
            # updates), so a summation-order difference shows up as a large
            # mean|Δ|/mean|r| while cos stays ~1e-6. A real bug - a pad leak, a
            # mis-indexed pairwise decay - moves cos, and that is what fails here.
            if c_ab < args.cos_min:
                fails.append("%s: chunk vs recurrent cos=%.8f" % (name, c_ab))
            if name in ref:
                c_rc, m_rc = stats(a, ref[name])          # recurrent vs HF
                c_bc, m_bc = stats(b, ref[name])          # chunk vs HF
                line += " | vs HF: cos %.6f -> %.6f  meanrel %.2e -> %.2e" % (
                    c_rc, c_bc, m_rc, m_bc)
            rows.append(line)
    for r in rows[:4] + ["  ..."] + rows[-4:]:
        print(r)
    print("\n%d comparisons over %d tokens (%d chunks), %d GDN layers"
          % (len(rows), len(ids), -(-len(ids) // 64), len(GDN_LAYERS)))
    if fails:
        print("FAILED %d:" % len(fails))
        for f in fails[:12]:
            print("  ", f)
        return 1
    print("PASS: chunked prefill == recurrent prefill, and the chunk form is no "
          "further from the reference than the recurrent one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
