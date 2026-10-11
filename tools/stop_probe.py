#!/usr/bin/env python3
"""How far does the reference run before it stops on its own? A pre-flight for choosing a deep fixture.

`dec64f` is teacher-forced, so its chain runs straight through the end-of-turn token (the stop id,
248046, at generation step 24) and keeps going. That is what makes it a 64-step pin, and it is also what
makes one of its findings uninterpretable on its own: steps 27-34 are simultaneously the window where the
reference's GDN state energy halves AND the window after the model has been driven past its own stop, and
`engine/route_price.py` shows both legal reference routes going under G5 there too. To separate "the state
got small" from "we are asking the model to continue a turn it already finished", the fixture has to be one
whose own chain never reaches the stop inside the pinned window.

So: ask the reference, before pinning anything. One model load, greedy for `--max-new` tokens, and the
report is the FIRST stop index plus what it generated. Rendering is byte-identical to
`tools/reference_dump.py`'s (`apply_chat_template(..., add_generation_prompt=True,
enable_thinking=False)`), because a fixture is only comparable with `dec8`/`dec64f` if the prompt went in
the same way.

    .venv/bin/python tools/stop_probe.py --message "Count from 1 to 80, separating the numbers with commas."
    .venv/bin/python tools/stop_probe.py --message "..." --max-new 300

It generates through `model.generate`, and that is the trap this file exists to record: the first version
hand-rolled the decode loop and fed one token at a time **without `past_key_values`**, so every step saw a
one-token context and every prompt looked like a degenerate four-token loop. The same prompt counted
correctly one directory over, in the dumper's own cached loop. A probe that mis-states the model's behaviour
is worse than no probe, so this one uses the library loop and cannot be wrong that way again. Stop ids come
from `speccheck/specimpl.py`'s own constants (248046 primary, 248044 legacy — A-10) rather than being
re-typed or written as template text, per the repo rule that specials are referred to by id.

Nothing is written here. The decision it feeds — "does this prompt run long enough to be a no-crossing deep
fixture?" — is the operator's, and the number to look at is `first stop at step` (or `none in N`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "speccheck"))
from specimpl import EOS_PRIMARY, EOS_LEGACY      # 248046, 248044 (A-10)  # noqa: E402

torch.set_grad_enabled(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--message", action="append", required=True)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--enable-thinking", action="store_true",
                    help="render with enable_thinking=True. The pinned fixtures use False"
                         " (tools/reference_dump.py); this flag exists to measure what the other"
                         " render does, not to change a fixture's rendering by accident.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").eval()
    print(f"loaded in {time.time() - t0:.1f}s | config eos_token_id ="
          f" {getattr(model.config, 'eos_token_id', None)} | stop id used here {EOS_PRIMARY}"
          f" (legacy {EOS_LEGACY})")

    for text in args.message:
        rendered = tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=False,
                                           enable_thinking=args.enable_thinking)
        if isinstance(rendered, list):
            rendered = "".join(rendered)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        print(f"\n=== {text!r}\n  prompt: {len(ids)} tokens = {ids}")
        t0 = time.time()
        # `eos_token_id=None` so generation does NOT stop early: the question is where the chain FIRST
        # hits each stop id, and a loop that stops on the legacy one cannot answer that for the other.
        out = model.generate(torch.tensor([ids], dtype=torch.long), max_new_tokens=args.max_new,
                             do_sample=False, eos_token_id=None, pad_token_id=None)
        gen = [int(t) for t in out[0][len(ids):]]
        stop_at = [i for i, t in enumerate(gen) if t == EOS_PRIMARY]
        legacy_at = [i for i, t in enumerate(gen) if t == EOS_LEGACY]
        print(f"  generated {len(gen)} tokens in {time.time() - t0:.1f}s")
        print(f"  FIRST STOP ({EOS_PRIMARY}) AT STEP:"
              f" {stop_at[0] if stop_at else f'none in {len(gen)}'}"
              "   <- a no-crossing deep fixture wants this well past its window")
        print(f"  first legacy stop {EOS_LEGACY} (config eos, A-10) at step:"
              f" {legacy_at[0] if legacy_at else 'none'}")
        print(f"  ids: {gen}")
        print(f"  text: {tok.decode(gen)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
