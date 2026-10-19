#!/usr/bin/env bash
# A4-L1 Gate 1 build: compile the shipped score roll and the two-accumulator variant, CPU only.
#
# Two sides, built IDENTICALLY so the only difference between them is the patch in a4_l1_two_acc.py:
#   base/  = a copy of engine/kernels_attn.cpp + an unmodified copy of engine/attn_fattn_kernels.h
#   new/   = the same copy + the patched header (acc0/acc1 over alternating k-tiles, summed after the roll)
# The copy is compiled from a scratch dir so `#include "engine/attn_fattn_kernels.h"` resolves to the copy
# (quote-includes search the includer's directory first). engine/ is never written to, nothing is launched,
# no timing is taken, and nothing lands in the tree.
#
# The base side doubles as the behavioural canary (AGENTS, "a paired A/B against a stale binary reads as a clean
# null"): if base does not reproduce the shipped note block, the harness changed the code and the delta is void.
#
#   bash prototype/attn_percost/a4_l1_build.sh [outdir]        # default /tmp/a4l1v
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
OUT=${1:-/tmp/a4l1v}

for side in base new; do
    rm -rf "$OUT/$side"
    mkdir -p "$OUT/$side/engine"
    cp "$REPO/engine/kernels_attn.cpp" "$OUT/$side/kernels_attn.cpp"
    cp "$REPO/engine/attn_fattn_kernels.h" "$OUT/$side/engine/attn_fattn_kernels.h"
done
python3 "$REPO/prototype/attn_percost/a4_l1_two_acc.py" --out "$OUT/new/engine/attn_fattn_kernels.h"

for side in base new; do
    (cd "$OUT/$side" && hipcc -O3 -Wno-unused-command-line-argument \
        -I"$OUT/$side" -I"$REPO" -I"$REPO/engine" --save-temps -c kernels_attn.cpp -o k.o) >/dev/null
    /opt/rocm/llvm/bin/llvm-objdump -d "$OUT/$side/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.o" \
        > "$OUT/$side/objdump.s"
    echo "built $side: $OUT/$side/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s"
done
