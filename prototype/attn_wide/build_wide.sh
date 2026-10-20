#!/usr/bin/env bash
# Build the A12 wide-shape prototype: the correctness binary, and (with --compile-only) the LLVM listing the
# register ladder reads.
#
#   bash prototype/attn_wide/build_wide.sh [outdir]                  # linked binary, correctness self-test
#   bash prototype/attn_wide/build_wide.sh [outdir] --compile-only   # -c --save-temps, nothing linked
#   WIDE_WARPS=3 WIDE_STAGE=16 WIDE_QREG=0 WIDE_PVF16=0 WIDE_MINB=4 ... same, which is how wide_ladder.sh
#   drives it (every axis is one env var, so the ladder stays one line per variant)
#
# WIDE_MINB is the `__launch_bounds__` min-blocks argument, and it is a ladder axis in its own right: A10 left
# open "whether clang honours a min-blocks argument on gfx1151 at all", and row L of `wide_ladder.sh` is the one
# compile that answers it (min-blocks 4 budgets 196,608/(256x4) = 192 VGPRs, which the observed 256 does not).
#
# Two refusals, both earned elsewhere in this tree:
#   * the listing must contain the WMMA sites the source says it emits. A probe that launches a DCE'd shell
#     measures nothing (`prototype/attn_percost/build_a10_pvf16_map.sh` refuses the same way), and every loop
#     over a register array in `attn_wide.h` is `#pragma unroll` precisely so the site counts stay static.
#   * `-Wl,-rpath,/opt/rocm/lib` is not decoration (the standing HIP-build trap).
# `--save-temps` writes into the CURRENT directory, which is why this cds into the outdir.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wide}
mode=${2:-}
mkdir -p "$out"
cd "$out"

FLAGS=(-O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument -I"$here" -I"$repo")
for v in WIDE_WARPS WIDE_STAGE WIDE_QREG WIDE_PVF16 WIDE_MINB; do
  if [[ -n ${!v:-} ]]; then FLAGS+=(-D$v=${!v}); fi
done
hipcc "${FLAGS[@]}" --save-temps -c "$here/attn_wide.hip.cpp" -o attn_wide.o

L=attn_wide.hip-hip-amdgcn-amd-amdhsa-gfx1151.s
warps=${WIDE_WARPS:-8}; stage=${WIDE_STAGE:-64}; pvf16=${WIDE_PVF16:-1}
groups=$(( stage / 16 ))                       # C tiles per round: the score roll's inner trip count
want_f32=$(( 16 * groups ))                    # the score roll: 16 k-tiles x one C tile per slice
want_f16=$(( pvf16 == 1 ? 16 * groups : 0 ))   # P·V on the F16 atom: 8 tiles x 2 op_sel x slices
want_f32=$(( want_f32 + (pvf16 == 1 ? 0 : 16 * groups) ))   # P·V on the F32 atom: 16 tiles x slices
n32=$(grep -c 'v_wmma_f32_16x16x16_f16' "$L" || true)
n16=$(grep -c 'v_wmma_f16_16x16x16_f16' "$L" || true)
{ [ "$n32" -eq "$want_f32" ] && [ "$n16" -eq "$want_f16" ]; } \
  || { echo "ABORT: listing has $n32 v_wmma_f32 and $n16 v_wmma_f16 sites; the wide body emits $want_f32 and $want_f16 (score roll 16 k-tiles x $groups slices, plus the P·V leg)" >&2; exit 1; }
echo "wmma sites: $n32 v_wmma_f32 + $n16 v_wmma_f16 (warps=$warps stage=$stage qreg=${WIDE_QREG:-1} pvf16=$pvf16)"
[ "$mode" == "--compile-only" ] && exit 0

hipcc -Wl,-rpath,/opt/rocm/lib attn_wide.o -o attn_wide
echo "wrote $out/attn_wide  (listing $out/$L)"
