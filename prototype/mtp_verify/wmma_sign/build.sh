#!/usr/bin/env bash
# Build `wmma_sign` — is the sign of a zero k-term visible in an F16 wmma result? See wmma_sign.cpp's header
# and prototype/mtp_verify/FIX-ZERO.md.
#
# Same two standing rules as every other HIP build here: `-Wl,-rpath,/opt/rocm/lib` is not decoration
# (AGENTS' HIP-build trap), and the build REFUSES if the emitted code object does not contain the sites the
# source says it runs — a probe whose WMMA sites were optimised away measures nothing (AGENTS: a check whose
# input list can be empty cannot fail). The site census is printed, because the point of this probe is the
# emitted instruction sequence as much as the numbers.
#
#   bash prototype/mtp_verify/wmma_sign/build.sh [outdir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${1:-$repo/tmp/mtp_verify/wmma_sign}
mkdir -p "$out"
cd "$out"      # --save-temps writes the listing into the CURRENT directory

hipcc -O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument \
      -Wl,-rpath,/opt/rocm/lib --save-temps \
      -I"$repo" "$here/wmma_sign.cpp" -o "$out/wmma_sign" 2>&1 | tee "$out/build.log"

L=$(ls "$out"/*-hip-amdgcn-amd-amdhsa-gfx1151.s 2>/dev/null | head -1 || true)
if [ -z "${L:-}" ]; then echo "ABORT: no gfx1151 device listing next to $out" >&2; exit 1; fi
for k in k_wmma_h k_wmma_f32 k_micro; do
  grep -q "^\.globl.*$k\|\.globl.*$k" "$L" || { echo "ABORT: code object has no $k entry point" >&2; exit 1; }
done
w=$(grep -c 'v_wmma' "$L" || true)
[ "$w" -gt 0 ] || { echo "ABORT: listing has no v_wmma sites; the tiles were optimised away" >&2; exit 1; }
echo "listing=$(basename "$L")"
echo "v_wmma sites=$w  v_pk_fma_f16=$(grep -c 'v_pk_fma_f16' "$L" || true)  v_fma_f16=$(grep -c 'v_fma_f16' "$L" || true)  v_pk_mul_f16=$(grep -c 'v_pk_mul_f16' "$L" || true)  v_add_f16=$(grep -c 'v_add_f16' "$L" || true)"
echo "wrote $out/wmma_sign"
