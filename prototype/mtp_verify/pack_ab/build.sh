#!/usr/bin/env bash
# Build `pack_ab` — the isolated cross-row rig (see pack_ab.cpp's header, prototype/mtp_verify/ISOLATED.md).
#
# Same two standing rules as every other HIP build in this tree: `-Wl,-rpath,/opt/rocm/lib` is not
# decoration (AGENTS' HIP-build trap), and the build refuses if the emitted code object does not contain
# the kernels the source says it launches — a probe whose WMMA sites were optimised away measures nothing
# (`build_causal_bound.sh` refuses the same way for the same reason).
#
#   bash prototype/mtp_verify/pack_ab/build.sh [outdir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${1:-$repo/tmp/mtp_verify/pack_ab}
mkdir -p "$out"
cd "$out"      # `--save-temps` writes the listing into the CURRENT directory, not next to -o

hipcc -O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument \
      -Wl,-rpath,/opt/rocm/lib --save-temps \
      -I"$repo" "$here/pack_ab.cpp" -o "$out/pack_ab" 2>&1 | tee "$out/build.log"

L=$(ls "$out"/*-hip-amdgcn-amd-amdhsa-gfx1151.s 2>/dev/null | head -1 || true)
if [ -z "${L:-}" ]; then echo "ABORT: no gfx1151 device listing next to $out" >&2; exit 1; fi
for k in attn_wide_llama_wave attn_wide_wave; do
  grep -q "\.globl.*$k" "$L" || { echo "ABORT: code object has no $k entry point" >&2; exit 1; }
done
w=$(grep -c 'v_wmma' "$L" || true)
[ "$w" -gt 0 ] || { echo "ABORT: listing has no v_wmma sites; the wide bodies were optimised away" >&2; exit 1; }
echo "kernels present in $(basename "$L"): attn_wide_llama_wave attn_wide_wave; v_wmma sites=$w"
echo "wrote $out/pack_ab"
