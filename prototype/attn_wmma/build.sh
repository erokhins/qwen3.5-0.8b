#!/usr/bin/env bash
# Compile-only build for the WMMA feeder price probe. There is deliberately no link step and no
# launcher: D30 item 2.4 is priced from the ISA, and a device run would need the GPU lock and would
# produce a timing, which this item must not contain.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wmma_price}
mkdir -p "$out"
cd "$out"
exec hipcc -O3 -gline-tables-only --offload-arch=gfx1151 --save-temps \
     -Wno-unused-command-line-argument \
     -I"$repo" -I"$repo/engine" \
     -c "$here/wmma_feeder_probe.cpp" -o wmma_feeder_probe.o
