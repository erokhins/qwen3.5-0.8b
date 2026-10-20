#!/usr/bin/env bash
# A11 §6's owed measurement, run as one held window: does the LDS bound bind the shipping kernel at all?
#
# Arm BASE = shipping build, declared LDS 20,480 B (LDS allows 3 blocks/CU, registers allow 2).
# Arm PAD  = same source + -DFATTN_LDS_PAD_HALVES=6144, declared LDS 32,768 B (LDS allows 2, registers 2).
# Same 241 VGPR on both arms (verified compile-side), so the ONLY difference is which resource binds.
#
#   PAD ~= BASE  -> registers bind at 2 blocks/CU: the kernel was never resident at 3, and a register
#                   saving (the f16 accumulator) is the lever that can move residency.
#   PAD  worse   -> the LDS bound was live at 3 blocks: residency is LDS-gated and the register story is wrong.
#
# Interleaved per repetition so any drift in the box hits both arms, and best-of-N per the ledger convention.
# Main agent only, blocking, zero subagents, one gpu_lock hold, no backgrounding.
set -u
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
PY=.venv/bin/python
LANES=${LANES:-16384}
REPS=${REPS:-3}
TAG=a11pad
tools/gpu_lock.sh acquire "$TAG" || { echo "LOCK FAILED"; exit 1; }
echo "util at acquire: $(tools/gpu_lock.sh util 2>/dev/null)"
for lane in 3072 "$LANES"; do
  echo "=== lanes=$lane ==="
  for i in $(seq 1 "$REPS"); do
    for arm in base pad; do
      # NOTE (trap class recorded in AGENTS.md): this tool prints a MARKDOWN TABLE, not a `RESULT` line, so a
      # grep for RESULT matches nothing and every field comes back empty while the run itself was fine. Match
      # the table row for this lane count instead, and refuse to print a number when the row is absent.
      line=$($PY tools/prefill_sweep.py --binary "/tmp/ab_res/${arm}_run" --lanes "$lane" --runs 1 2>&1 | grep -E "^\| *${lane} \|")
      if [ -z "$line" ]; then echo "rep$i $arm: NO-ROW (unparsed, not a zero)"; else echo "rep$i $arm: $line"; fi
    done
  done
done
echo "util at release: $(tools/gpu_lock.sh util 2>/dev/null)"
tools/gpu_lock.sh release "$TAG"
echo "WINDOW DONE"
