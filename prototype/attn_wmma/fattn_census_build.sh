#!/usr/bin/env bash
# Compile-only census build of the engine attention TU. CPU only: no kernel is launched, no timing.
# Usage: prototype/attn_wmma/fattn_census_build.sh [outdir]
#
# This is the gate's own build, so it lives in the tree beside the gate that runs it: while it lived in
# `tmp/gcc/` (gitignored) the tracked census could not be reproduced from a clone, which is the same rule that
# puts every other measurement behind a tool in `tools/`. Behaviour is byte-identical to the scratch copy that
# `fattn_ship_census.py` used to shell out to, and that copy is left in place as scratch.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
OUT=${1:-/tmp/fattn_census}
rm -rf "$OUT"
mkdir -p "$OUT"
hipcc -O3 -Wno-unused-command-line-argument -I"$REPO" -I"$REPO/engine" --save-temps \
    -c "$REPO/engine/kernels_attn.cpp" -o "$OUT/k.o"
ls -l "$OUT" | head
