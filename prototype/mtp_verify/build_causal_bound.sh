#!/usr/bin/env bash
# Build `causal_bound` — the engine's own attention cores, launched from a probe that reads out which
# cache lanes each query row attended (see causal_bound.cpp's header for how).
#
# Same two standing rules as every other HIP build in this tree: `-Wl,-rpath,/opt/rocm/lib` is not
# decoration (AGENTS' HIP-build trap), and the build refuses if the emitted code object does not contain
# the kernels the source says it launches — a probe whose WMMA sites were optimised away measures nothing
# (`prototype/attn_wide/build_wide.sh` refuses the same way for the same reason).
#
#   bash prototype/mtp_verify/build_causal_bound.sh [outdir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/mtp_verify}
mkdir -p "$out"
cd "$out"      # `--save-temps` writes the listing into the CURRENT directory, not next to -o

# `-Wl,-rpath,/opt/rocm/lib` must reach the LINK step, so compile and link stay one invocation.
hipcc -O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument \
      -Wl,-rpath,/opt/rocm/lib --save-temps \
      -I"$repo" "$here/causal_bound.cpp" -o "$out/causal_bound" 2>&1 | tee "$out/build.log"

L=$(ls "$out"/*-hip-amdgcn-amd-amdhsa-gfx1151.s 2>/dev/null | head -1 || true)
if [ -z "${L:-}" ]; then echo "ABORT: no gfx1151 device listing next to $out — cannot check what was emitted" >&2; exit 1; fi
for k in attn_wide_llama_wave attn_wide_wave attn_slim_wave; do
  grep -q "\.globl.*$k" "$L" || { echo "ABORT: code object has no $k entry point" >&2; exit 1; }
done
# The wide forms are WMMA kernels; if their matrix sites vanished the probe would be timing/asking a shell.
w=$(grep -c 'v_wmma' "$L" || true)
[ "$w" -gt 0 ] || { echo "ABORT: listing has no v_wmma sites; the wide bodies were optimised away" >&2; exit 1; }
echo "kernels present in $(basename "$L"): attn_wide_llama_wave attn_wide_wave attn_slim_wave; v_wmma sites=$w"
echo "wrote $out/causal_bound"
