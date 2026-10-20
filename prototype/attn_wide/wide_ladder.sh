#!/usr/bin/env bash
# The A12 compile-only register/scratch/LDS ladder. NOTHING IS LAUNCHED: this is `A10-RF-LADDER.md`'s method
# (build the variant, read the code object, no device), applied to the wide shape because A12's whole premise is
# that the shape fits once spill is a budget and the accumulator is F16, and "fits" is a compile question.
#
#   bash prototype/attn_wide/wide_ladder.sh [outdir]      # ~8 hipcc compiles, all CPU
#
# The numbers come from `.amdhsa_next_free_vgpr` / `.amdhsa_private_segment_fixed_size` /
# `.amdhsa_group_segment_fixed_size` via `wide_regread.py`, never from the LLVM note block's `.vgpr_count`
# (AGENTS.md, the note-block row) and never from `-Rpass-analysis=kernel-resource-usage`, whose
# `Occupancy [waves/SIMD]` is a constant on this SKU.
#
# The axes, one at a time off the shipping shape:
#   warps 8 -> 4 -> 3   (R1: the block size, which is what buys 256 VGPR/thread)
#   stage 64 -> 32 -> 16 (R3: positions per round, i.e. MAC per staged byte)
#   qreg 1 -> 0          (R2: Q resident vs re-read per round — the row that shows what Q-in-registers COSTS)
#   pvf16 1 -> 0         (the accumulator's format: 64 registers vs 128)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wide_ladder}
rm -rf "$out"; mkdir -p "$out"

# name | WIDE_WARPS | WIDE_STAGE | WIDE_QREG | WIDE_PVF16 | what the row asks
rows=(
  "A_a12_wide          8 64 1 1  the A12 shape as funded: 128 rows, 64-stage, Q resident, F16 acc"
  "B_q_from_global     8 64 0 1  same, Q re-read from global every round: what Q-in-registers costs"
  "C_f32_accumulator   8 64 1 0  same, F32 accumulator (128 regs instead of 64): the A10 source-kill shape"
  "D_qreg0_f32         8 64 0 0  neither: Q from global AND F32 accumulator"
  "E_stage32           8 32 1 1  half the stage: 32 positions per round"
  "F_stage16           8 16 1 1  the shipping stage depth at 8 waves: separates rows-per-block from stage depth"
  "G_warps4            4 64 1 1  half the block: 64 rows, 4 waves, stage 64"
  "H_warps3_stage64    3 64 1 1  the shipping block size with the deep stage and Q resident"
  "I_warps3_like_eng   3 16 1 1  the closest this body gets to the shipping kernel's shape"
  "J_stage128_f16      8 128 1 1  twice the stage: how big the spill gets when the stage doubles (compile-only, LDS 64 KiB)"
  "K_stage128_f32      8 128 1 0  the same deep stage on the F32 accumulator: the accumulator's contribution to the spill"
  "L_minblocks4        8 64 1 1  the min-blocks question A10 left open: llama declares occupancy 2 and gets 256 VGPR at 256 threads, so does a min-blocks argument bind at all? At 4, LLVM's own arithmetic (196,608 / (256x4)) predicts 192 where the unbounded shape says 256"
)
# `__launch_bounds__` min-blocks per row; a row not listed here is compiled with 1, which is what every other row
# wants because 196,608/(256x1) = 768 is above the 256 per-thread ceiling and therefore binds nothing.
declare -A MINB=([L_minblocks4]=4)

printf "%-18s %-6s %-6s %-5s %-6s %-5s | %6s %6s %8s %7s %6s  binder\n" \
       "variant" "warps" "stage" "qreg" "pvf16" "minb" "vGPU" "alloc" "scratch" "LDS" "blk/CU"
for r in "${rows[@]}"; do
  set -- $r
  name=$1; warps=$2; stage=$3; qreg=$4; pvf16=$5; shift 5
  desc="$*"
  minb=${MINB[$name]:-1}
  d="$out/$name"
  if ! WIDE_WARPS=$warps WIDE_STAGE=$stage WIDE_QREG=$qreg WIDE_PVF16=$pvf16 WIDE_MINB=$minb \
       bash "$here/build_wide.sh" "$d" --compile-only > "$d.log" 2>&1; then
    printf "%-18s %-6s %-6s %-5s %-6s %-5s | BUILD FAILED (see %s.log)\n" "$name" "$warps" "$stage" "$qreg" "$pvf16" "$minb" "$d"
    continue
  fi
  line=$(python3 "$here/wide_regread.py" "$d/attn_wide.hip-hip-amdgcn-amd-amdhsa-gfx1151.s" \
           --match attn_wide_wave --threads $((warps * 32)) | tail -1)
  printf "%-18s %-6s %-6s %-5s %-6s %-5s | %s\n" "$name" "$warps" "$stage" "$qreg" "$pvf16" "$minb" "$line"
  echo "                     └ $desc   [$(grep -o 'wmma sites:.*' "$d.log")]"
done
