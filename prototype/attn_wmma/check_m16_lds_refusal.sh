#!/usr/bin/env bash
# check_m16_lds_refusal.sh — assert the witness behind the largest M = 16 cost.
#
# The claim: an M = 16 attention block CANNOT keep the shipped whole-row score staging
# (`s_sc[ATTN_TILE_G][MAXSC]`, affordable only because `ATTN_TILE_M == 1`). At M = 16 it is
# 4 heads x 16 rows x 512 lanes x 2 B = 65,536 B, and with the Q and K stages on top the compiler
# refuses the TU. CPU-only compile, no device, no lock.
#
# This script FAILS if the TU compiles (the cost is gone, or the TU stopped expressing it) or if the
# compiler's reason is not the local-memory limit — a refusal for an unrelated typo would otherwise
# read as the same result.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wmma_m16}
mkdir -p "$out"

err=$(cd "$out" && hipcc -O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument \
        -I"$repo" -I"$repo/engine" -c "$here/wmma_m16_lds_refusal.cpp" -o /dev/null 2>&1)
rc=$?

if [ "$rc" -eq 0 ]; then
    echo "REFUSAL-CHECK FAIL: wmma_m16_lds_refusal.cpp COMPILED. The whole-row score staging at"
    echo "M = 16 no longer exceeds the LDS limit, so the cost it witnesses is gone (or the TU no"
    echo "longer expresses it). Re-read it before citing the 65,536 B figure."
    exit 1
fi
if ! printf '%s\n' "$err" | grep -q "local memory ([0-9]*) exceeds limit"; then
    echo "REFUSAL-CHECK FAIL: the TU failed, but not for the LDS reason. Compiler said:"
    printf '%s\n' "$err" | head -5
    exit 1
fi
echo "REFUSAL-CHECK PASS: $(printf '%s\n' "$err" | grep -m1 -oE "local memory \([0-9]+\) exceeds limit \([0-9]+\) in '[^']+'")"
echo "  -> an M = 16 block must stage scores tile-locally or fuse the two passes: whole-row s_sc at"
echo "     4 heads x 16 rows x 512 lanes is 65,536 B on top of the Q and K stages. The drain's SLOT cost"
echo "     does not care which stride it uses (STRIDE leg ~ 0 in m16_census.py); its LDS cost does."
