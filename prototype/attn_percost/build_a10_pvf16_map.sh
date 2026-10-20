#!/usr/bin/env bash
# Build the F16-accumulator D-block map probe. Correctness-only: it launches one warp once and reads the
# raw D registers back. No timing, no best-of-N, no warm loop (AGENTS: a worker's device check is correctness
# only). Mirrors `prototype/attn_wmma/build_map_probe.sh`, including its two build-time refusals: the
# listing must contain the instruction under test, and it must contain BOTH issues, or the probe would be
# measuring a DCE'd shell.
#
#   bash prototype/attn_percost/build_a10_pvf16_map.sh [outdir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/pvf16_map}
mkdir -p "$out"
cd "$out"
hipcc -O3 --offload-arch=gfx1151 --save-temps -Wno-unused-command-line-argument \
      -I"$repo" -I"$repo/engine" \
      -c "$here/a10_pvf16_map_probe.cpp" -o a10_pvf16_map_probe.o
# -Wl,-rpath is not decoration (the standing HIP-build trap).
hipcc --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib a10_pvf16_map_probe.o -o a10_pvf16_map_probe
L=a10_pvf16_map_probe-hip-amdgcn-amd-amdhsa-gfx1151.s
grep -q "v_wmma_f16_16x16x16_f16" "$L" \
  || { echo "ABORT: no v_wmma_f16_16x16x16_f16 in the listing — the probe launches nothing matrix-like" >&2; exit 1; }
n=$(grep -c "v_wmma_f16_16x16x16_f16" "$L")
[ "$n" -ge 2 ] || { echo "ABORT: listing has $n v_wmma_f16 lines; the two-op_sel shape needs >= 2" >&2; exit 1; }
echo "wrote $out/a10_pvf16_map_probe ($n v_wmma_f16_16x16x16_f16 sites; op_sel sites below)"
grep -n "v_wmma_f16_16x16x16_f16" "$L" | sed 's/^/  /'
