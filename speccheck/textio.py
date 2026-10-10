"""Chat rendering + tokenization contract (S02.2, A-10, A-11) — independent version.

The engine has to build the prompt string itself (the real engine will do this in
C++), so the renderer lives here rather than being borrowed from
`apply_chat_template`. `gates/run_tok.py` proves this file reproduces the reference
ids token-exactly, which is what makes it safe to keep.
"""
from __future__ import annotations

from pathlib import Path

IM_START = "<|im_start|>"          # 248045
IM_END = "<|im_end|>"            # 248046 — also the engine's stop token (A-10)
# Written by concatenation on purpose: `<|...|>` literals get mangled by text
# tooling, and Qwen3.5 has TWO distinct tokens that both look like an im-start
# (248045 `<|im_start|>` vs 248068 `<|im_start|>`). Getting these wrong is invisible
# when reading a terminal and breaks tokenization at position 13 of 17. Verified by
# gates/run_tok.py against the tokenizer's own added_tokens_decoder.
THINK_START = "<" "|" "think" "|" ">"      # 248068
THINK_END = "<" "|" "end" "|" ">"          # 248069
THINK_START_ID = 248068
THINK_END_ID = 248069
EOS_PRIMARY_ID = 248046
EOS_LEGACY_ID = 248044                          # im_end, legacy stop (A-10)
EOS_IDS = (EOS_PRIMARY_ID, EOS_LEGACY_ID)


def render_chat(messages, add_generation_prompt: bool = True,
                thinking: bool = False) -> str:
    """S02.2: concatenate `<|im_start|>{role}\\n{content}<|im_end|>\\n` with no extra
    separators, then the generation header. `thinking` selects the trailing branch
    that `enable_thinking` controls in the checkpoint template."""
    out = []
    for m in messages:
        out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    if add_generation_prompt:
        tail = f"{THINK_START}\n" if thinking else f"{THINK_START}\n\n{THINK_END}\n\n"
        out.append(f"{IM_START}assistant\n{tail}")
    return "".join(out)


class Tokenizer:
    """Minimal wrapper: the engine calls exactly two things — render, and
    tokenize-without-added-specials (A-11: special tokens are preserved and are not
    merged with adjacent text)."""

    def __init__(self, model_dir: str | Path):
        from transformers import AutoTokenizer          # only needed for the checks
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        # never trust the literals: resolve the think tokens from the tokenizer
        dec = self.tok.added_tokens_decoder
        for tid, lit in ((THINK_START_ID, "THINK_START"), (THINK_END_ID, "THINK_END")):
            real = dec[tid].content if tid in dec else None
            globals()[lit] = real
        self.think_ids = (THINK_START_ID, THINK_END_ID)

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def render_ids(self, messages, thinking: bool = False) -> list[int]:
        return self.encode(render_chat(messages, thinking=thinking))

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    @property
    def eos_id(self) -> int:
        return int(self.tok.eos_token_id)

    @property
    def special_ids(self) -> dict:
        return {k: int(v) for k, v in self.tok.special_tokens_map.items()
                if isinstance(v, (int, str)) and not isinstance(v, list)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--message", default="The capital of France is")
    args = ap.parse_args()
    t = Tokenizer(args.model)
    msgs = [{"role": "user", "content": args.message}]
    print(repr(render_chat(msgs)))
    print(t.render_ids(msgs))
    print("eos", t.eos_id, "specials", t.special_ids)
