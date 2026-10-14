#!/usr/bin/env python3
"""Reference-side decision margins for a corpus manifest — the other half of D-4.

A free-run agreement number means nothing without knowing how close the decisions were. `dec64t` and
`dec300f` agree 64/64 and 128/128 because their next tokens are near-certain (D-4 of
`progress/2026-10-12/plan-2026-10-12.md`), and the instrument that knows that is `engine/gate.py::metrics()`'s
`gap12`: the pinned logits' top1 minus top2. The corpus fixtures (`tools/corpus_dump.py`) pin only
ids — 49 prompts x 24 steps of full logits would be ~700 MB and was never worth it — so the margin
has to be re-derived from the reference on CPU, at the same place in the chain.

Per prompt, along the REFERENCE's own greedy continuation (forced, so the chain cannot drift from
itself), this records for each of the 24 steps:

    gaps[i]     top1 - top2 of the reference's logits, logits units (BF16 payload, F64 subtract)
    top1[i]     the id that won, which must equal the pinned continuation[i] — checked, not assumed
    top2[i]     the runner-up id, so a divergent step can be read as "the engine took the runner-up"

and it refuses, per prompt, if its own greedy chain does not reproduce the pinned continuation
(`--allow-pin-mismatch` to override, which is what you would need if the corpus manifest itself were
regenerated under a different attention implementation).

CPU-only. No device, no timings. The gap is defined on the reference's logits, so a worker may run it
while someone else holds the GPU; it just costs a minute of CPU per corpus.

    .venv/bin/python tools/corpus_margin.py --corpus corpus48 --out data/refs/corpus48-margins.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--corpus", default="corpus48")
    ap.add_argument("--out", default="data/refs/corpus48-margins.json")
    ap.add_argument("--attn-implementation", default="eager",
                    help="eager is the oracle corpus48 was pinned with; sdpa reproduces corpus48sdpa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--allow-pin-mismatch", action="store_true",
                    help="write margins even where the reproduced chain differs from the pin")
    args = ap.parse_args()

    src = resolve(args.corpus)
    if src.is_dir():
        src = src / "manifest.json"
    elif src.suffix != ".json":
        src = ROOT / "data" / "refs" / args.corpus / "manifest.json"
    ref = json.loads(src.read_text())
    prompts = ref["prompts"][args.start: args.start + args.limit if args.limit else None]
    steps = ref["steps"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    model.eval()

    items, bad = [], []
    t0 = time.time()
    K = 8            # how far down the reference's own ranking to keep (see the note below)
    for i, p in enumerate(prompts):
        ids = list(p["ids"])
        pin = list(p["continuation"])
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=True)
            cache = o.past_key_values
            gaps, top1, top_ids, top_vals, gap_dist = [], [], [], [], []
            cur = ids
            for s in range(steps):
                lg = o.logits[:, -1, :].float()          # BF16 payload, F64-comparable
                # argmax for the WINNER, topk only for the ranking. Not interchangeable: BF16 logits
                # over a 248k vocab do tie at the top (dec8 step 11 is a pinned exact tie), `generate`
                # and `reference_dump.py` both take argmax — which returns the LOWEST tied index —
                # while topk's order among equal values is unspecified. Selecting with topk here would
                # make this pass disagree with the pin for a reason that is the tool's, not the model's.
                win = int(torch.argmax(lg[0]))
                tv = torch.topk(lg[0], k=K)
                vals = [float(v) for v in tv.values]
                tids = [int(t) for t in tv.indices]
                gaps.append(vals[0] - vals[1])
                # The margin to the first DISTINCT value below the top. `gaps` is the gate's gap12 and
                # is the right number for "could any implementation flip this" (0 = a tie, and a tie is
                # flipped by an epsilon), but it cannot say how far the engine's error would have had
                # to reach for a flip: that is this one. Only as good as K — if all 8 stored values tie
                # at the top this reads 0.0 rather than lying.
                below = [v for v in vals if v < vals[0]]
                gap_dist.append(vals[0] - below[0] if below else 0.0)
                top1.append(win)
                top_ids.append(tids)
                top_vals.append(vals)
                cur = [win]
                if s < steps - 1:
                    with torch.no_grad():
                        o = model(input_ids=torch.tensor([cur], dtype=torch.long),
                                  past_key_values=cache, use_cache=True)
                    cache = o.past_key_values
        match = sum(1 for a, b in zip(top1, pin) if a == b)
        if match != len(pin):
            bad.append((p["text"], match, len(pin)))
        items.append({"text": p["text"], "ids": ids, "gaps": gaps, "gap_distinct": gap_dist,
                      "top1": top1, "top_ids": top_ids, "top_vals": top_vals,
                      "pin_match": match, "pin_len": len(pin)})
        print(f"  {args.start + i + 1}/{len(prompts)}  {p['text'][:40]!r}"
              f"  pin_match {match}/{len(pin)}  median gap {sorted(gaps)[len(gaps)//2]:.3f}")

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "corpus-margins", "corpus": str(src), "model": args.model,
        "attn_implementation": args.attn_implementation, "steps": steps, "top_k": K,
        "n": len(items), "seconds": round(time.time() - t0, 1),
        "note": ("per step of the reference's OWN greedy chain: gaps = top1-top2 of its logits "
                 "(engine/gate.py's gap12, logits units, 0 means an exact BF16 tie); gap_distinct = "
                 "top1 minus the first distinct value below it; top_ids/top_vals = its top-K ranking, "
                 "enough to say where the engine's token sat in the reference's own order"),
        "prompts": items}, indent=1))
    print(f"wrote {out}: {len(items)} prompts x {steps} steps")
    if bad:
        print(f"WARNING: {len(bad)} prompt(s) where the reproduced chain differs from the pin"
              " — the margins describe the reproduced chain, not the pinned one:")
        for text, m, n in bad[:10]:
            print(f"   {m}/{n}  {text[:50]!r}")
        if not args.allow_pin_mismatch:
            raise SystemExit(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
