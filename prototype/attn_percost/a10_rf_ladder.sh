#!/usr/bin/env bash
# a10_rf_ladder.sh — A10 step 2: compile the R1+R2+R3 register ladder and read its note blocks. CPU ONLY.
#
# Nothing here launches a kernel, creates a device context, or takes a timing. The two readings are the code
# object's own note block (`.vgpr_count`, `.vgpr_spill_count`, `group_segment_fixed_size`) parsed by
# `prototype/attn_wmma/fattn_residency_read.py` — the tree's one note-block reader, which imports
# `fattn_ship_census.py`'s parser rather than writing a second one (AGENTS: list the census tools before
# writing one). Unlike `a6_rf_ladder.sh` there is no RUN=1 side: a driver occupancy query needs a context, and
# A10 step 2 is compile-only by construction.
#
#   bash prototype/attn_percost/a10_rf_ladder.sh [outdir]        # default /tmp/a10rf
#
# CANARY, asserted before the ladder means anything. The same clang, in the same run, must still read the
# SHIPPING kernel `attn_fattn_wave_u8` at 185 VGPR / 0 spill / 20,480 B LDS — the figures
# `A10-REWRITE-BRIEF.md` §3 carries and §48 shipped on. If that triple does not print, the ladder below is a
# prediction from a compiler nobody calibrated (AGENTS: a check whose input can fail to arrive must be shown to
# have arrived, in the window that produces the number).
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
OUT=${1:-/tmp/a10rf}
rm -rf "$OUT"; mkdir -p "$OUT/ladder" "$OUT/canary"

echo "== CANARY: the shipping kernel, same clang, same flags"
( cd "$OUT/canary" && hipcc -O3 -Wno-unused-command-line-argument -I"$REPO" -I"$REPO/engine" --save-temps \
      -c "$REPO/engine/kernels_attn.cpp" -o k.o ) >"$OUT/canary/compile.log" 2>&1 \
    || { echo "CANARY BUILD FAILED"; tail -30 "$OUT/canary/compile.log"; exit 1; }
python3 "$REPO/prototype/attn_wmma/fattn_residency_read.py" --asm "$OUT/canary/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s" \
    --all 2>/dev/null | grep -E "attn_fattn_wave" | sed 's/^/  /'

CAN=$(python3 "$REPO/prototype/attn_wmma/fattn_residency_read.py" --asm \
        "$OUT/canary/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s" --all 2>/dev/null \
      | grep -E "^\s+attn_fattn_wave_u8\s" | sed -E 's/.*vgpr_count=([0-9]+).*vgpr_spill_count=([0-9]+).*group_segment_fixed_size=([0-9]+)/\1 \2 \3/')
read -r CAN_VGPR CAN_SPILL CAN_LDS <<<"$CAN"
if [ "${CAN_VGPR:-0}" = 185 ] && [ "${CAN_SPILL:-0}" = 0 ] && [ "${CAN_LDS:-0}" = 20480 ]; then
    echo "  CANARY PASS: attn_fattn_wave_u8 (the shipping default, --attn-fattn-qstage unstaged8) reads"
    echo "  CANARY PASS: 185 VGPR / 0 spill / 20,480 B, the same triple §53's inertness witness printed"
else
    echo "  CANARY FAIL: attn_fattn_wave_u8 read vgpr=${CAN_VGPR:-nothing} spill=${CAN_SPILL:-nothing} lds=${CAN_LDS:-nothing}."
    echo "  CANARY FAIL: the ladder below would be a prediction from an uncalibrated compiler; stop and read why."
    exit 1
fi

echo
echo "== LADDER build (compile only, nothing launched)"
( cd "$OUT/ladder" && hipcc -O3 -Wno-unused-command-line-argument --offload-arch=gfx1151 \
      -Wl,-rpath,/opt/rocm/lib --save-temps -c "$REPO/prototype/attn_percost/a10_rf_ladder.hip.cpp" -o k.o ) \
    >"$OUT/ladder/compile.log" 2>&1 || { echo "LADDER BUILD FAILED"; tail -40 "$OUT/ladder/compile.log"; exit 1; }
S=$(ls "$OUT/ladder"/*gfx1151.s)
echo "  listing: $S"
grep -c "failed to meet occupancy target" "$OUT/ladder/compile.log" | sed 's/^/  llvm occupancy refusals: /'

echo
echo "== LADDER note blocks"
python3 "$REPO/prototype/attn_wmma/fattn_residency_read.py" --asm "$S" --all 2>/dev/null \
    | grep -E "a10_" \
    | sed -E 's/^ +//; s/vgpr_spill_count/spill/; s/sgpr_count.*sgpr_spill_count=[0-9]+ +//; s/private_segment_fixed_size/priv/; s/group_segment_fixed_size/lds/'

echo
echo "== per-kernel derived residency (LDS term from KFD's 64 KiB; the register file per CU stays UNKNOWN,"
echo "   per prototype/attn_wmma/fattn_regfile_ladder.md §3, so no register-derived block count is printed)"
for k in a10_stub_0 a10_stub_8192 a10_stub_20480 a10_stub_32768 a10_stub_36096 a10_stub_65536 \
         a10_cur_lb0 a10_cur_lb1 a10_cur_lb2 a10_cur_lb3 a10_cur_lb4 \
         a10_r1_f32_lb0 a10_r1_f32_lb1 a10_r1_f32_lb2 a10_r1_f32_lb3 a10_r1_f32_lb4 \
         a10_r1_f16_lb0 a10_r1_f16_lb1 a10_r1_f16_lb2 a10_r1_f16_lb3 a10_r1_f16_lb4 \
         a10_r1_f32_noq_lb0 a10_r1_f16_noq_lb0 a10_r1_f16_split_lb0 a10_3w_f32_lb0 a10_3w_f16_lb0; do
    python3 "$REPO/prototype/attn_wmma/fattn_residency_read.py" --asm "$S" --kernel "$k" 2>/dev/null \
        | awk '/^kernel |vgpr_count|vgpr_spill_count|private_segment|group_segment|derived blocks|binding limit|derived wf/ {printf "%s", $0 " | "} END {print ""}'
done
