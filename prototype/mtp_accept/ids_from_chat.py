#!/usr/bin/env python3
"""Tokenize a text file as ONE chat-template turn, so an accept rate can be compared against an engine that
decoded chat prose rather than raw continuation.

WHY THIS EXISTS. Every llama.cpp number in this repo came out of `llama-cli`, whose conversation mode is
selected by the presence of a chat template in the GGUF (this build has no `-no-cnv`; `--chat-template ""` and
`--no-skip-chat-parsing` both still apply the template, verified 2026-10-24). So llama.cpp's accept ladder of
0.750 is a rate on TEMPLATE-WRAPPED THINKING PROSE, while ours was measured on the same bytes as a RAW
continuation. Prose that follows a rigid template is easier to predict, so the two numbers were never directly
comparable -- the caveat is recorded in prototype/mtp_llama/LLAMA-MTP-RESULT.md but was never priced. This
script builds the missing arm: the identical text, wrapped the way the model's own template wraps it.

The render is `tools/chat_render.render_transcript` -- the same function `tools/reference_dump.py` uses to
make the pinned fixtures -- so the template here is the one the pins were made with, not a hand-written guess.

Like `ids_from_text.py`, the output is COMMA-separated (engine `--ids-file` splits on commas; a space-joined
file reads as one token and the run still "succeeds" in ~1 s) and the field count written to disk is asserted,
because the file IS the claim about how many lanes the fixture has.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from chat_render import render_transcript  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="file whose contents become the user turn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--thinking", type=int, default=1,
                    help="1 = enable_thinking=True, which is what llama.cpp's decoded stream was")
    a = ap.parse_args()

    raw = open(a.text, encoding="utf-8", errors="replace").read()
    tok = AutoTokenizer.from_pretrained(a.model)
    rendered = render_transcript(tok, [{"role": "user", "content": raw}],
                                 thinking=bool(a.thinking))
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    if not ids:
        raise SystemExit("rendered to zero ids")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(",".join(str(i) for i in ids))
    back = open(a.out).read().strip().split(",")
    if len(back) != len(ids):
        raise SystemExit(f"wrote {len(ids)} ids, read back {len(back)}")
    print(f"ids={len(ids)} thinking={a.thinking} bytes={os.path.getsize(a.out)} out={a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
