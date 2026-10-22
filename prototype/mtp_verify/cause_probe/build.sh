#!/usr/bin/env bash
# Build `cause_probe` — does the engine's own P*V roll shape (4 groups x 2 issues into one D tuple, C = +0,
# a mostly-dead k map) move its F16 result on the sign of a dead k-term's V cell?
# See prototype/mtp_verify/cause_probe/cause_probe.cpp and prototype/mtp_verify/CAUSE.md.
#
# Same standing rules as `wmma_sign`'s build: `-Wl,-rpath,/opt/rocm/lib`, and the build REFUSES if the emitted
# code object does not contain the WMMA sites the source says it runs — a probe whose tiles were optimised
# away measures nothing and prints a confident null.
#
#   bash prototype/mtp_verify/cause_probe/build.sh [outdir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${1:-$repo/tmp/mtp_verify/cause_probe}
mkdir -p "$out"
cd "$out"

hipcc -O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument \
      -Wl,-rpath,/opt/rocm/lib --save-temps \
      -I"$repo" "$here/cause_probe.cpp" -o "$out/cause_probe" 2>&1 | tee "$out/build.log"

L=$(ls "$out"/*-hip-amdgcn-amd-amdhsa-gfx1151.s 2>/dev/null | head -1 || true)
if [ -z "${L:-}" ]; then echo "ABORT: no gfx1151 device listing next to $out" >&2; exit 1; fi
grep -q "k_roll" "$L" || { echo "ABORT: code object has no k_roll entry point" >&2; exit 1; }
# The roll is 4 groups x 2 issues = 8 v_wmma_f16 sites; a build that folded any of them away is not the shape.
w=$(grep -c 'v_wmma_f16' "$L" || true)
[ "$w" -ge 8 ] || { echo "ABORT: listing has $w v_wmma_f16 sites, expected >= 8 (4 groups x 2 issues)" >&2; exit 1; }
echo "listing=$(basename "$L")  v_wmma_f16_sites=$w"
echo "wrote $out/cause_probe"
