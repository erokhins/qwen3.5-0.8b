#!/usr/bin/env python3
"""G7 corpus fixture: reference greedy continuations for many plain-text prompts.

`reference_dump.py` pins per-tensor intermediates, which is far too expensive to do
for hundreds of prompts. This dumps only what the corpus half of G7 needs — prompt
text, prompt ids, and the reference greedy continuation — so that speccheck can be
measured for prefix agreement at scale (§10 G7: >= 95 % before any perf claim).

Cost on this box: one model load, then ~1-2 s per prompt for a 24-token greedy
continuation with eager attention on CPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--corpus", default=str(ROOT / "speccheck" / "fixtures" / "corpus.txt"))
    ap.add_argument("--tag", default="corpus48")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of prompts")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--attn-implementation", default="eager",
                    help="eager = the pinned oracle; sdpa is a legitimate different "
                         "rounding of the same math, used to measure the chaos floor")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation)
    model.eval()

    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines() if ln.strip()]
    if args.limit:
        prompts = prompts[:args.limit]
    out = ROOT / "data" / "refs" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    items, t0 = [], time.time()
    for i, text in enumerate(prompts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        gen = model.generate(torch.tensor([ids]), max_new_tokens=args.steps,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        cont = [int(t) for t in gen[0][len(ids):]]
        items.append({"text": text, "ids": [int(x) for x in ids], "continuation": cont})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    (out / "manifest.json").write_text(json.dumps({
        "kind": "corpus", "model": str(args.model), "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "steps": args.steps, "n": len(items), "seconds": round(time.time() - t0, 1),
        "note": "plain-text prompts (no chat template); greedy continuation from HF-ref",
        "prompts": items}, indent=1))
    print(f"wrote {out}/manifest.json: {len(items)} prompts, "
          f"{args.steps} tokens each, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
