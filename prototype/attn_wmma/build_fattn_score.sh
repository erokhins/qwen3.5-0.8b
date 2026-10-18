#!/usr/bin/env bash
# Compile-only build for D62's `m16_fattn_score` form (progress/2026-10-18/llama-fattn-anatomy.md
# §4.2). Deliberately no link step and no launcher: the form is priced from the ISA, and a device run
# would need the GPU lock and would produce a timing, which this item must not contain. Same protocol
# as `build_m16.sh` beside this file; separate script because a separate TU (see that file's header for
# why: the D57 instrument's recorded census is the calibration witness and stays byte-identical).
#
#   bash prototype/attn_wmma/build_fattn_score.sh [outdir]
#
# Leaves in `outdir` (default `tmp/attn_wmma_fattn`):
#   wmma_fattn_score-hip-amdgcn-amd-amdhsa-gfx1151.s   the device listing `fattn_census.py` reads
#   resource_usage.txt                                 clang's own -Rpass-analysis VGPR/LDS remarks
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wmma_fattn}
mkdir -p "$out"
cd "$out"
# The listing and the resource remarks in one compile: the remarks are emitted for the device pass, so
# they must come from the same compile that produces the listing the census counts. -O3, never -O0
# (AGENTS.md: never benchmark an -O0 build). The remarks are the route the task asks for; the listing's
# own `.vgpr_count`/`.lds_byte_size` note block is the cross-check, and `fattn_census.py` FAILs if the
# two disagree.
rc=0
hipcc -O3 -gline-tables-only --offload-arch=gfx1151 --save-temps \
      -Rpass-analysis=kernel-resource-usage \
      -Wno-unused-command-line-argument \
      -I"$repo" -I"$repo/engine" \
      -c "$here/wmma_fattn_score.cpp" -o wmma_fattn_score.o > resource_usage.txt 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  cat resource_usage.txt >&2
  echo "ABORT: compile failed with $rc (full remarks/diagnostics left in $out/resource_usage.txt)" >&2
  exit "$rc"
fi
grep -q "v_wmma_f32_16x16x16_f16" \
    wmma_fattn_score-hip-amdgcn-amd-amdhsa-gfx1151.s \
  || { echo "ABORT: the listing has no v_wmma_f32_16x16x16_f16 — nothing matrix-like survived" >&2; exit 1; }
echo "wrote $out/wmma_fattn_score-hip-amdgcn-amd-amdhsa-gfx1151.s"
echo "wrote $out/resource_usage.txt"
