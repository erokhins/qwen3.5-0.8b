#!/usr/bin/env python3
"""Compare greedy continuations between two corpus manifests (or a manifest and a
speccheck run) and report mean prefix agreement.

Used to separate two things that look identical in a table:
  speccheck vs HF-eager  -> is our implementation diverging?
  HF-sdpa vs HF-eager    -> how much does ANY legal rounding difference diverge?
                            (the chaos floor of greedy decoding in bf16)
If the second number is close to the first, the divergence is chaos, not error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def agreement(a: Path, b: Path):
    A = json.loads(a.read_text())
    B = json.loads(b.read_text())
    ka = "prompts" if "prompts" in A else "prompts"
    assert len(A[ka]) == len(B["prompts"]), "different corpus sizes"
    tot = ag = 0.0
    worst = []
    for x, y in zip(A[ka], B["prompts"]):
        assert x["text"] == y["text"], (x["text"], y["text"])
        u, v = x["continuation"], y["continuation"]
        n = min(len(u), len(v))
        k = 0
        while k < n and u[k] == v[k]:
            k += 1
        ag += k / max(len(u), 1)
        tot += 1
        if k < len(u):
            worst.append((k / len(u), x["text"][:44], k, len(u)))
    return ag / tot, tot, sorted(worst)[:5]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    pa = Path(args.a) if Path(args.a).is_absolute() else ROOT / "data" / "refs" / args.a
    pb = Path(args.b) if Path(args.b).is_absolute() else ROOT / "data" / "refs" / args.b
    for p in (pa, pb):
        p = p / "manifest.json" if p.is_dir() else p
    frac, n, worst = agreement(pa if pa.is_file() else pa / "manifest.json",
                               pb if pb.is_file() else pb / "manifest.json")
    print("%s %s vs %s: mean prefix agreement %.4f over %d prompts"
          % (args.label, args.a, args.b, frac, n))
    for a, txt, k, ln in worst:
        print("   %.2f  %-44s diverged at %d/%d" % (a, txt, k, ln))
    return 0


if __name__ == "__main__":
    sys.exit(main())
