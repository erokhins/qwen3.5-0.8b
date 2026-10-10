#!/usr/bin/env python3
"""G7 (S10.1) - end-to-end greedy token equality against the reference.

The tensor gates say "the numbers are close"; this gate says the thing users
actually depend on: the sampled ids are identical, token for token, including the
stop behaviour (A-10). It reads `generated_ids` from a speccheck dump and compares
with `manifest.greedy_continuation`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def run_corpus(args):
    """G7 at scale (§10). Plain-text prompts, no per-tensor pins, so hundreds are
    cheap. Agreement per prompt = matched-prefix length / reference length.

    An absolute 95 % threshold is NOT achievable by any implementation: greedy
    continuations of this model in bf16 are chaotic beyond ~8 tokens. Measured, the
    reference disagrees with *itself* (sdpa vs eager attention, same weights, same
    dtype) at 0.8776 mean prefix agreement, while speccheck reaches 0.9124. So the
    requirement is relative: speccheck must be at least as close to the pinned oracle
    as a legitimate rounding variant of the oracle is (A-18)."""
    import torch
    from speccheck.specimpl import Forward, Weights
    torch.set_grad_enabled(False)
    m = json.loads((ROOT / "data" / "refs" / args.corpus / "manifest.json").read_text())
    w = Weights(Path(args.model))
    total, agree_sum, worst = 0, 0.0, []
    for it in m["prompts"]:
        want = list(it["continuation"])
        f = Forward(w, gdn_form=args.gdn_prefill, attn_softmax=args.attn_softmax)
        got = f.greedy(it["ids"], max(len(want) - 1, 0), stop_on_eos=False)
        n = min(len(got), len(want))
        k = 0
        while k < n and got[k] == want[k]:
            k += 1
        agree_sum += k / max(len(want), 1)
        total += 1
        if k < len(want):
            worst.append((k / len(want), it["text"][:40], k, len(want), want[k], got[k]
                          if k < len(got) else None))
    frac = agree_sum / total if total else 0.0
    print("corpus=%s prompts=%d steps=%d  speccheck mean prefix agreement=%.4f"
          % (args.corpus, total, m["steps"], frac))
    for a, txt, k, n, want, got in sorted(worst)[:3]:
        print("   %.2f  %-40s diverged at %d/%d (ref=%d spec=%s)" % (a, txt, k, n, want, got))
    if args.floor:
        fm = json.loads((ROOT / "data" / "refs" / args.floor / "manifest.json").read_text())
        fl, tot = 0.0, 0
        for x, y in zip(fm["prompts"], m["prompts"]):
            u, v = x["continuation"], y["continuation"]
            n = min(len(u), len(v))
            k = 0
            while k < n and u[k] == v[k]:
                k += 1
            fl += k / max(len(u), 1)
            tot += 1
        floor = fl / tot
        print("floor=%s (a legal rounding variant of the reference vs the reference)"
              " mean prefix agreement=%.4f" % (args.floor, floor))
        if frac + 1e-9 < floor - args.margin:
            print("G7 corpus FAILED: speccheck %.4f is below the reference self-consistency "
                  "floor %.4f by more than %.2f" % (frac, floor, args.margin))
            return 1
        print("G7 corpus PASS: speccheck is at least as faithful to the oracle as the "
              "oracle is to itself (%.4f >= %.4f - %.2f)" % (frac, floor, args.margin))
        return 0
    if frac < args.min_agreement:
        print("G7 corpus: %.4f < absolute %.2f (no floor given; pass --floor corpus48sdpa)"
              % (frac, args.min_agreement))
        return 1
    print("G7 corpus PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default="smoke,p64,p300,dec8,p8,prefix24dec8")
    ap.add_argument("--gdn-prefill", default="chunk", choices=["recurrent", "chunk"])
    ap.add_argument("--attn-softmax", default="full", choices=["full", "running"])
    ap.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--corpus", default=None, help="corpus fixture from tools/corpus_dump.py")
    ap.add_argument("--floor", default=None, help="reference self-consistency fixture, e.g. "
                                                 "corpus48sdpa (see A-18)")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--min-agreement", type=float, default=0.95)
    args = ap.parse_args()

    if args.corpus:
        sys.path.insert(0, str(ROOT))
        return run_corpus(args)

    fails = []
    for tag in [t for t in args.fixtures.split(",") if t]:
        m = json.loads((ROOT / "data" / "refs" / tag / "manifest.json").read_text())
        want = list(m["greedy_continuation"])
        ids = m["prefill_ids"]
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=True) as tf:
            cmd = [args.python, str(ROOT / "speccheck" / "specimpl.py"),
                   "--prompt-ids-file", str(ROOT / "data" / "refs" / tag / "manifest.json"),
                   "--decode", str(max(len(want) - 1, 0)), "--gdn-prefill", args.gdn_prefill,
                   "--attn-softmax", args.attn_softmax, "--dump", tf.name]
            subprocess.run(cmd, check=True, capture_output=True)
            got = [int(t) for t in np.load(tf.name)["generated_ids"]]
        # the reference continuation is unbounded; compare the overlap, and require
        # at least the reference length unless the engine stopped on EOS (A-10)
        n = min(len(got), len(want))
        bad = [i for i in range(n) if got[i] != want[i]]
        eos = [i for i, t in enumerate(got) if t in (248046, 248044)]
        stopped = bool(eos) and len(got) <= len(want)
        ok = not bad and (len(got) >= len(want) or stopped)
        print("%-6s ref=%d got=%d  first mismatch=%s  %s"
              % (tag, len(want), len(got), ("idx %d" % bad[0]) if bad else "none",
                 "ok" if ok else "FAIL"))
        if not ok:
            fails.append("%s: %s (got %s vs ref %s)"
                         % (tag, ("mismatch at %s" % bad[0]) if bad else "length",
                            got[:12], want[:12]))
    if fails:
        print("G7 FAILED:")
        for f in fails:
            print("  ", f)
        return 1
    print("G7 PASS: greedy ids identical to the reference on %s "
          "(gdn=%s, attn=%s)" % (args.fixtures, args.gdn_prefill, args.attn_softmax))
    return 0


if __name__ == "__main__":
    sys.exit(main())
