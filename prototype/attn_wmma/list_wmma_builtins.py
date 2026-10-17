#!/usr/bin/env python3
"""DEAD END: this toolchain's `clang -cc1` has no `-dump-builtins` (error: unknown argument), so the
builtin list cannot be enumerated this way; `rocwmma`'s headers were read instead. Kept as the record.

List the WMMA builtins this toolchain declares for gfx1151, and the fragment shapes they take.

D30 item 2.4 has to price a WMMA path's FEEDER cost (how the fragments get filled), not just the
matrix instruction's rate; the rate is already measured (`prototype/issue_rate_probe/results.md` §1,
26.31 T-MAC/s at 88.8 % of roof). What was never established is whether the compiler offers a
fragment load/store at all on this target, or only the raw `wmma` op — which is the difference
between "the feeder is the compiler's problem (cheap)" and "the feeder is hand-written ds_loads
and shuffles (the cost that decides the price)".
"""
import re
import subprocess
import sys

CLANG = "/opt/rocm/lib/llvm/bin/clang++"
out = subprocess.run([CLANG, "-cc1", "-dump-builtins"], capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(f"clang -cc1 -dump-builtins failed: {out.stderr[:400]}")
names = [l.split()[0] for l in out.stdout.splitlines() if "wmma" in l.lower()]
print(f"{len(names)} builtins mention 'wmma'")
for pre in ("__builtin_amdgcn_wmma_load", "__builtin_amdgcn_wmma_store", "__builtin_amdgcn_wmma"):
    hit = sorted({n for n in names if n.startswith(pre)})
    print(f"\n{pre}*  -> {len(hit)}")
    for n in hit[:60]:
        print("   ", n)
sys.exit(0)
