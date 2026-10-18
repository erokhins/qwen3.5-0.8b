#!/usr/bin/env bash
# Build the driver-authority occupancy probe for D62's `m16_fattn_score`: links
# `wmma_fattn_score.cpp` (the kernels) with `occupancy_fattn_probe.cpp` (a host main that only queries
# the driver). Compile+link only; the binary launches no kernel and takes no timing.
#
#   bash prototype/attn_wmma/build_occupancy_fattn.sh [outdir]
#
# Running it creates a device context, so it belongs under `tools/gpu_lock.sh` (see the file header).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wmma_fattn}
mkdir -p "$out"
# -Wl,-rpath is required for HIP builds on this box (prototype/README.md).
exec hipcc -O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib \
     -Wno-unused-command-line-argument \
     -I"$repo" -I"$repo/engine" \
     "$here/occupancy_fattn_probe.cpp" "$here/wmma_fattn_score.cpp" \
     -o "$out/occupancy_fattn_probe"
