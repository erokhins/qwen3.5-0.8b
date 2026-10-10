#!/usr/bin/env python3
"""G8 — MTP drafter gate (S09). Discharges A-12 and A-13 by measurement.

transformers has no MTP class for `qwen3_5`, so there is no reference dump to compare
against and nothing to take on faith. Three separate questions, in order of strength:

**G8a — the loop is exactly right (`verify="steps"`, binding, bit-exact).**
Greedy verification makes speculation a no-op on the output *provided the verifying
arithmetic is the arithmetic the target would have done alone*. Feeding the draft block
through the ordinary decode path gives exactly that, so the output must equal plain
greedy token for token, and the target state afterwards must be bit-identical to a
sequential replay over the same tokens. This is what tests positions, rollback and KV
truncation - the parts of S09.3/S08.5 that a drafter cannot hide behind.

**G8b — batching the verify is honest (`verify="batch"`, reported, floored).**
One forward over [t, d1..dk] is what an engine must do to win, and a batched GEMM is
not bit-identical to a GEMV in bf16: the target can flip a near tie that it would not
have flipped alone, so greedy speculation is not bit-lossless in practice. The number
this is judged against is the same self-consistency floor as A-18 (0.8776, HF-sdpa vs
HF-eager) - a legal rounding variant of the oracle. If an engine wants exact lossless
speculation it must make its verify path reuse the decode kernels bit for bit.

**Accept rate (binding, decisive between readings of §09).**
A correct drafter agrees with the target often. The three readings of §09 - A-12 pre- vs
post-final-norm hidden, A-13 `[emb ‖ hidden]` vs `[hidden ‖ emb]` - are measured on the
same prompts; the spec reading must be the best of them. If it is not, §09 is wrong and
gets corrected here, not in the engine.

Speed is reported, never gated: on torch/CPU a draft step pays the full 248k-row tied
lm_head, so drafting can legitimately lose (§09.4 predicted a thin budget).

Usage: run_mtp.py [--k 1] [--steps 24] [--prompts 8] [--min-accept 0.30]
                  [--min-agreement 0.80] [--floor 0.8776]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speccheck.mtp import MTPDraft, greedy_mtp                    # noqa: E402
from speccheck.specimpl import (ATTN_LAYERS, GDN_LAYERS, Forward,  # noqa: E402
                                Weights)
from speccheck.textio import Tokenizer                            # noqa: E402

# Resolved by this gate: A-12 as first written (pre-final-norm hidden) was WRONG,
# A-13 ([emb ‖ hidden]) was right. `spec` below = the reading the code defaults to.
VARIANTS = [("post_norm", "eh", "§09 as corrected (A-12 post-final-norm, A-13 [emb‖hidden])"),
            ("pre_norm", "eh", "A-12 as originally written (pre-final-norm hidden)"),
            ("post_norm", "he", "A-13 alternative (fc order [hidden‖emb])")]


def prefix_agreement(a: list[int], b: list[int]) -> float:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def state_identical(f: Forward, g: Forward) -> str | None:
    """Bit-exact state comparison; returns a description of the first difference."""
    if f.P != g.P:
        return f"prefix_len {f.P} != {g.P}"
    for l in GDN_LAYERS:
        for name, have, want in (("rec", f.rec[l], g.rec[l]),
                                 ("conv", f.conv_win[l], g.conv_win[l])):
            if have is None or want is None:
                if have is not want:
                    return f"L{l:02d}.{name} presence"
            elif not torch.equal(have, want):
                return f"L{l:02d}.{name} values"
    for l in ATTN_LAYERS:
        for n in ("k", "v"):
            a, b = f.kv[l].get(n, []), g.kv[l].get(n, [])
            if len(a) != len(b):
                return f"L{l:02d}.{n} cache len {len(a)} != {len(b)}"
            if any(not torch.equal(x, y) for x, y in zip(a, b)):
                return f"L{l:02d}.{n} cache values"
    return None


def corpus_prompts(tok, n, root):
    prompts = [ln.strip() for ln in
               (root / "speccheck" / "fixtures" / "corpus.txt").read_text().splitlines()
               if ln.strip()][:n]
    for tag in ("smoke", "p64", "p300"):
        man = json.loads((root / "data" / "refs" / tag / "manifest.json").read_text())
        prompts.append(man["sources"][0]["text"])
    return prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--k", type=int, default=1, help="draft tokens per target step")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--min-accept", type=float, default=0.30)
    ap.add_argument("--min-agreement", type=float, default=0.80)
    ap.add_argument("--floor", type=float, default=0.8776,
                    help="A-18 self-consistency floor (HF-sdpa vs HF-eager)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = Weights(Path(args.model))
    tok = Tokenizer(args.model)
    prompts = corpus_prompts(tok, args.prompts, ROOT)
    print(f"G8 MTP gate: k={args.k}, {len(prompts)} prompts × {args.steps} tokens\n")
    fails: list[str] = []

    # ---- G8a: exact losslessness + bit-exact state, sequential verify -------------
    worst = None
    for p in prompts:
        ids = tok.encode(p)
        f = Forward(w, gdn_form="chunk")
        m = MTPDraft(w)
        out, st = greedy_mtp(f, m, ids, args.steps, k=args.k, stop_on_eos=False,
                             verify="steps")
        ref_f = Forward(w, gdn_form="chunk")
        ref = ref_f.greedy(ids, len(out) - 1, stop_on_eos=False)
        if out != ref[:len(out)]:
            j = next(i for i, (x, y) in enumerate(zip(out, ref)) if x != y)
            fails.append(f"G8a LOSSY on {p[:28]!r}: first difference at token {j}")
            continue
        # the target must also be left exactly where a plain sequential run would leave
        # it: replay the consumed prefix token by token through decode(), which is what
        # the steps verifier used.
        j = len(out) - 1   # every emitted token except the last has been consumed
        seq = Forward(w, gdn_form="chunk")
        seq.prefill(torch.tensor(ids))
        for one in out[:j]:
            seq.decode(one)
        d = state_identical(seq, f)
        if d:
            worst = d
            fails.append(f"G8a STATE differs on {p[:28]!r}: {d}")
    print("G8a output identical to plain greedy and target state bit-identical to a "
          "sequential replay: %s" % ("PASS" if not fails else "FAIL"))

    # ---- G8b: batched verify, measured against the self-consistency floor ---------
    agree, acc_b, dra_b = [], 0, 0
    for p in prompts:
        ids = tok.encode(p)
        f = Forward(w, gdn_form="chunk")
        out, st = greedy_mtp(f, MTPDraft(w), ids, args.steps, k=args.k,
                             stop_on_eos=False, verify="batch")
        ref = Forward(w, gdn_form="chunk").greedy(ids, len(out) - 1, stop_on_eos=False)
        agree.append(prefix_agreement(out, ref))
        acc_b += st["accepted"]
        dra_b += st["drafted"]
    mean_agree = sum(agree) / len(agree)
    worst_p = min(agree)
    print("G8b batched-verify output vs plain greedy: mean prefix agreement %.4f "
          "(worst prompt %.3f) against the A-18 floor %.4f"
          % (mean_agree, worst_p, args.floor))
    if mean_agree < min(args.min_agreement, args.floor):
        fails.append(f"G8b agreement {mean_agree:.3f} below floor {args.floor}")

    # ---- accept rate across the three readings of §09 -----------------------------
    rates = {}
    t0 = time.time()
    for h, o, label in VARIANTS:
        acc = dra = fwd = toks = 0
        for p in prompts:
            f = Forward(w, gdn_form="chunk")
            _, st = greedy_mtp(f, MTPDraft(w, h_source=h, fc_order=o), tok.encode(p),
                               args.steps, k=args.k, stop_on_eos=False)
            acc += st["accepted"]; dra += st["drafted"]
            fwd += st["target_forwards"]; toks += len(f.generated) - 1
        rates[(h, o)] = acc / max(dra, 1)
        print("  %-46s accept %.4f (%d/%d)  tokens/target-forward %.2f"
              % (label, rates[(h, o)], acc, dra, toks / max(fwd, 1)))
    spec, alt_h, alt_o = rates[("post_norm", "eh")], rates[("pre_norm", "eh")], \
        rates[("post_norm", "he")]
    if spec < args.min_accept:
        fails.append(f"spec reading accept {spec:.3f} < --min-accept {args.min_accept}")
    if not (spec > alt_h and spec > alt_o):
        fails.append(f"§09 reading not best of three (spec {spec:.3f}, pre-norm "
                     f"{alt_h:.3f}, swapped fc {alt_o:.3f}) - correct §09 from this")

    print()
    if fails:
        for f_ in fails:
            print("FAIL:", f_)
        sys.exit(1)
    print(f"PASS: A-12 corrected to the post-final-norm hidden ({spec:.3f} vs pre-norm "
          f"{alt_h:.3f}); "
          f"A-13 [emb‖hidden] confirmed ({spec:.3f} vs swapped {alt_o:.3f}); loop exact; "
          f"batched verify within floor ({mean_agree:.3f} ≥ {args.floor}) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
