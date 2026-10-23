#!/usr/bin/env python3
"""ids_from_text.py — tokenise a text file into a `decode_run --ids-file` CSV (comma-separated ids).

Why this exists: llama.cpp's MTP A/B (`prototype/mtp_llama/llama_mtp_ab.sh:52`) uses
`head -c 10200 progress/2026-10-21/performance-status-2026-10-21.md` as its long-context input — a markdown
ledger document, which is far more predictable prose than our chat-template corpus. That makes the two
engines' accept rates incomparable as measured, so the only honest fix is to run our accept loop on the SAME
text. Every existing ids fixture was built by an earlier session's scratch and no writer survived in-tree;
this is that writer.

Choices, and why:
  * NO chat template. The input is a raw document continuation; adding a chat template would compare our
    templated continuation against llama.cpp's templated continuation of a DIFFERENT rendering, which is the
    confound this exists to remove. (llama.cpp does apply its own template; the token counts differ by ~100
    and that is labelled where the comparison is made, not hidden.)
  * splits on COMMAS, one field per token — `decode_run --ids-file` splits on commas, and a space-joined
    fixture reads as ONE token and silently turns a 4k run into a 1-token run (AGENTS.md trap).
  * asserts the field count it wrote, read back off the file, because the builder's nominal width is not a
    fact about the file.
"""
import argparse
import os
import sys
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=0)
    a = ap.parse_args()

    raw = open(a.text, "r", errors="replace").read()
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(raw, add_special_tokens=False)["input_ids"]
    if a.max_tokens:
        ids = ids[: a.max_tokens]
    if not ids:
        sys.exit("no tokens encoded")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(",".join(str(i) for i in ids) + "\n")

    back = open(a.out).read().strip().split(",")
    if len(back) != len(ids):
        sys.exit(f"field count mismatch: wrote {len(ids)}, read back {len(back)}")
    print(f"ids={len(ids)} bytes={os.path.getsize(a.out)} out={a.out}")
    # A one-field file is the known silent failure (space-joined fixture), so refuse loudly here.
    if len(ids) < 8:
        sys.exit(f"suspiciously few fields ({len(ids)}) — check the input really has text")


if __name__ == "__main__":
    main()
