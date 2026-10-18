#!/usr/bin/env bash
# Build the WMMA fragment-map probe (`fattn_map_probe.cpp`) — a LINKED binary, unlike every other build
# script in this directory, because this one's whole purpose is to launch the instruction once and read
# the result back. Correctness-only: no timing, no best-of-N, no warm loop.
#
#   bash prototype/attn_wmma/build_map_probe.sh [outdir]
#
# Leaves `$outdir` (default `tmp/attn_wmma_map`) with the binary and its listing. The listing is kept
# because the probe's own claim is about the instruction that gets emitted: if the compiler turned the
# wmma into something else (a scalar fallback, a DCE'd shell), the probe would be measuring the wrong
# thing, so the build FAILS unless the listing contains the same `v_wmma_f32_16x16x16_f16` the shipping
# form emits.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wmma_map}
mkdir -p "$out"
cd "$out"
hipcc -O3 --offload-arch=gfx1151 --save-temps -Wno-unused-command-line-argument \
      -I"$repo" -I"$repo/engine" \
      -c "$here/fattn_map_probe.cpp" -o fattn_map_probe.o
# -Wl,-rpath is not decoration: without it the linked binary dies at exec with
# "error while loading shared libraries: libamdhip64.so.7" (the standing HIP-build trap in AGENTS.md).
hipcc --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib fattn_map_probe.o -o fattn_map_probe
grep -q "v_wmma_f32_16x16x16_f16" fattn_map_probe-hip-amdgcn-amd-amdhsa-gfx1151.s \
  || { echo "ABORT: no v_wmma_f32_16x16x16_f16 in the listing — the probe would launch nothing matrix-like" >&2; exit 1; }
# The DCE witness: four tiles were asked for, and a probe whose four wmma calls collapsed into one would
# silently stop testing the accumulation pattern P2. Count, do not assume.
n=$(grep -c "v_wmma_f32_16x16x16_f16" fattn_map_probe-hip-amdgcn-amd-amdhsa-gfx1151.s)
[ "$n" -ge 17 ] || { echo "ABORT: listing has $n v_wmma lines; the four patterns need >= 17 (1 + 16)" >&2; exit 1; }
echo "wrote $out/fattn_map_probe ($n v_wmma_f32_16x16x16_f16 in the listing)"
