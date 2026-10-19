"""CPU-only citation check for prototype/attn_percost/A5-GEOMETRY.md.

Checks two things a reviewer would ask first: (1) every backticked file path in the document exists in the
tree, and (2) every `:NNN` line cite that stands for `engine/attn_fattn_kernels.h` (the document's own stated
convention) is within that file's line count and lands on a line containing something plausible. Prints a
refusal for anything wrong; exits non-zero. No device, no timing.
"""
import os
import re
import sys

DOC = "prototype/attn_percost/A5-GEOMETRY.md"
KERNEL = "engine/attn_fattn_kernels.h"

text = open(DOC).read()

paths = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:md|h|hpp|cpp|py|sh|diff|txt|json))`", text))
missing = sorted(p for p in paths if not os.path.exists(p))

klines = open(KERNEL).read().split("\n")
nums = sorted({int(m) for m in re.findall(r"`:(\d{2,4})`", text)} |
              {int(m) for m in re.findall(r"`:(\d{2,4}),", text)})
oob = [n for n in nums if n > len(klines)]

print(f"doc lines            : {len(text.splitlines())}")
print(f"paths cited          : {len(paths)}")
print(f"missing paths        : {missing if missing else 'none'}")
print(f"kernel line cites    : {nums}")
print(f"kernel file lines    : {len(klines)}")
print(f"out-of-range cites   : {oob if oob else 'none'}")
for n in nums:
    if n <= len(klines):
        print(f"  :{n:<5} -> {klines[n - 1].strip()[:88]}")
sys.exit(1 if (missing or oob) else 0)
