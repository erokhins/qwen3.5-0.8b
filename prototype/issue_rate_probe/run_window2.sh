#!/usr/bin/env bash
# Second half of the device window: the legs run_window.sh's grep pattern missed, plus the mix
# leg that produced no row at all. Same rules: main agent, gpu_lock held, zero subagents, blocking.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
OUT=${OUT:?set OUT to an existing writable path}
mkdir -p "$(dirname "$OUT")" || exit 1

echo "=== mix leg, run directly (run.py produced no row for it)" | tee -a "$OUT"
timeout 300 ./issue_rate_probe --kernel irp_mix_8 --iters 20000 --launches 20 --reps 5 2>&1 |
  tail -3 | tee -a "$OUT"

echo "=== attn dot-shape ILP and latency legs (grep fixed: rows are prefixed 'OK   IRP')" | tee -a "$OUT"
for k in irp_fma_1 irp_fma_4 irp_fma_8 irp_fma_16 irp_fma_32 irp_serial_8 irp_serial_32 \
         irp_cvt_8 irp_cvt_16; do
  timeout 300 python3 run.py --device --only "$k" --launches 20 --reps 5 2>&1 |
    grep -E "kernel=$k" >> "$OUT"
done

echo "=== WMMA leg again at three iters, to see whether the row is stable" | tee -a "$OUT"
for it in 500 2000 8000; do
  timeout 300 ./issue_rate_probe --kernel irp_wmma_4 --iters "$it" --launches 20 --reps 5 2>&1 |
    grep -E "^IRP|^OK" >> "$OUT"
done

n=$(grep -cE "^(OK )?IRP" "$OUT" || true)
echo "=== rows now in $OUT: $n"
echo "=== util at end: $(bash ../../tools/gpu_lock.sh util | tail -1)" | tee -a "$OUT"
