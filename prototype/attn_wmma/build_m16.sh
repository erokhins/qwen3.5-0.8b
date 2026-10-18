#!/usr/bin/env bash
# Compile-only build for the D57 / item-2.6 M = 16 WMMA falsifier. Deliberately no link step and no
# launcher: the item is priced from the ISA, and a device run would need the GPU lock and would produce
# a timing, which this item must not contain. Same protocol as `build.sh` beside this file.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wmma_m16}
mkdir -p "$out"
cd "$out"
exec hipcc -O3 -gline-tables-only --offload-arch=gfx1151 --save-temps \
     -Wno-unused-command-line-argument \
     -I"$repo" -I"$repo/engine" \
     -c "$here/wmma_m16_probe.cpp" -o wmma_m16_probe.o
