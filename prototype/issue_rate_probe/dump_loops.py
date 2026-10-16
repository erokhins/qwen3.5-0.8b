#!/usr/bin/env python3
"""Diagnostic: print every backward branch of each kernel in a header, with its histogram."""
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isa_census as ic  # noqa: E402

tmp = Path(tempfile.mkdtemp())
sfile = ic.compile_asm(tmp)
kernels = ic.slice_kernels(sfile.read_text())
names = sys.argv[1:] or sorted(kernels)
for name in names:
    body = kernels.get(name)
    if body is None:
        print("no kernel", name, "have:", ", ".join(sorted(kernels)))
        continue
    print("####", name)
    for L in ic.loops_of(body):
        span = body[L["start"]:L["stop"] + 1]
        hist = Counter()
        for line in span:
            m = re.match(r"([a-z][a-z0-9_.]*)", line.strip())
            if m:
                hist[m.group(1)] += 1
        print("  loop %-12s lines %d..%d insts=%d sig=%s" %
              (L["target"], L["start"], L["stop"], L["insts"],
               ", ".join("%s x%d" % kv for kv in sorted(hist.items(), key=lambda kv: -kv[1]))))
        if L.get("_show"):
            for line in span:
                print("     ", line.rstrip())
