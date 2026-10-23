#!/usr/bin/env python3
"""emitted_text.py — decode an `emitted.txt` (or `sampled.txt`) back to text, to characterise a fixture.

WHY. An accept rate is a property of the CONTINUATION, not only of the engine, and a number like 0.94 on one
fixture beside 0.51 on another is a claim about two texts before it is a claim about a draft block. The rate
tables in this directory are read as if they were interchangeable; this is the 5-second check that says what
the run actually generated. It prints the decoded stream and one line of shape (unique-token share, longest
repeated 8-gram), which is what separates "the drafter got good at prose" from "this continuation is the same
sentence eleven times".

It reads the tokenizer from the checkpoint directory (the same one `ids_from_text.py` used to build the
fixture, so the round trip is exact) and it prints nothing about time.

Usage:
    .venv/bin/python3 prototype/mtp_accept/emitted_text.py /tmp/pf/fan/C/acc_loop/emitted.txt
    .venv/bin/python3 prototype/mtp_accept/emitted_text.py --head 1200 FILE
"""
from __future__ import annotations

import argparse
import collections
import glob
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--model", default="")
    ap.add_argument("--head", type=int, default=700, help="characters of decoded text to print")
    ap.add_argument("--gram", type=int, default=8, help="n-gram length for the repetition statistic")
    a = ap.parse_args()

    model = a.model or sorted(glob.glob("models/Qwen*"))[0]
    lines = [ln.strip() for ln in a.path.read_text().splitlines() if ln.strip()]
    # `emitted.txt`/`sampled.txt` carry a `# emitted_token` header line (and a probe may append notes);
    # skipping non-numeric lines is the difference between a decoder and a ValueError on the header.
    ids = [int(t) for t in lines if not t.startswith("#") and t.lstrip("-").isdigit()]
    if not ids:
        raise SystemExit(f"{a.path}: no numeric id lines (header-only file?)")
    from transformers import AutoTokenizer        # only needed for this tool
    tok = AutoTokenizer.from_pretrained(model)
    text = tok.decode(ids, skip_special_tokens=False)
    grams = collections.Counter(tuple(ids[i:i + a.gram]) for i in range(len(ids) - a.gram + 1))
    top, n = (grams.most_common(1) or [((), 0)])[0]
    print(f"# {a.path}: {len(ids)} tokens, model={model}")
    print(f"# unique tokens {len(set(ids))}/{len(ids)} = {len(set(ids))/len(ids):.2f};"
          f" most repeated {a.gram}-gram appears {n}x ({n/max(1,len(grams)):.2f} of positions)")
    print(text[: a.head])
    if len(text) > a.head:
        print(f"... [{len(text) - a.head} more characters]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
