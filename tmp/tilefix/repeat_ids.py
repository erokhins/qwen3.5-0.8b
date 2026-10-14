#!/usr/bin/env python3
"""Repeat a comma-separated id file to a target length — a long-prompt A/B needs 3072 lanes and
the longest real fixture on the box is 1024. tools/prefill_sweep.py synthesises repeated prose
for exactly this reason (per-token cost at a lane count does not depend on the text's meaning),
so repetition is the same honesty the ledger's own ruler uses, and is stated as such wherever a
number taken from this file appears."""
import sys

src, n, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
toks = open(src).read().replace(",", " ").split()
out = []
while len(out) < n:
    out.extend(toks)
open(dst, "w").write(",".join(out[:n]) + "\n")
print(f"{dst}: {n} ids by repeating {len(toks)} ({'real text' if n <= len(toks) else 'REPEATED TEXT, not real prose'})")
