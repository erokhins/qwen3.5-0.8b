#!/usr/bin/env bash
# The device window, run by the main agent with tools/gpu_lock.sh held and zero subagents
# active. Prints raw rows only; no verdict arithmetic is done here because the binary's own
# pct_of_ceiling column is wrong (see results.md), and raw ms + geometry is what survives.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
OUT=${OUT:-/tmp/irp_window_raw.txt}
mkdir -p "$(dirname "$OUT")" || exit 1
: > "$OUT" || exit 1
# A window that runs and saves nothing is worse than one that fails loudly: the device time is
# spent and the evidence is zero. The first attempt at this script did exactly that, because the
# output directory did not exist and every append silently failed.
test -w "$OUT" || { echo "FATAL: cannot write $OUT" >&2; exit 1; }

echo "=== arithmetic legs (scalar floor to matrix roof)" | tee -a "$OUT"
for k in irp_f32_8 irp_mix_8 irp_pkf16_8 irp_d2f16_8 irp_d2bf16_8 irp_d4i8_8 irp_d8i4_8 \
         irp_u4u8_8 irp_u8u4_8 irp_wmma_4; do
  timeout 300 python3 run.py --device --only "$k" --launches 20 --reps 5 2>&1 |
    grep -E "kernel=$k" >> "$OUT"
done

echo "=== attn dot-shape ILP and latency legs" | tee -a "$OUT"
timeout 900 python3 run.py --device \
  --only irp_fma_1 --only irp_fma_4 --only irp_fma_8 --only irp_fma_16 --only irp_fma_32 \
  --only irp_serial_8 --only irp_serial_32 --only irp_cvt_8 --only irp_cvt_16 \
  --launches 20 --reps 5 2>&1 | grep -E "^IRP" >> "$OUT"

echo "=== KV read GB/s against resident footprint" | tee -a "$OUT"
for m in 2 8 32 128 512; do
  timeout 300 ./issue_rate_probe --kernel irp_kv_read --iters 1 --passes 2 \
    --footprint-mib "$m" --launches 20 --reps 5 2>&1 | grep -E "^IRP" >> "$OUT"
done

n=$(grep -c "^IRP" "$OUT" || true)
if [ "${n:-0}" -lt 15 ]; then
  echo "FATAL: window produced only $n IRP rows; expected >= 15 (10 arithmetic + 9 attn + 5 KV)" >&2
fi
echo "=== util at end: $(bash ../../tools/gpu_lock.sh util | tail -1)" | tee -a "$OUT"
grep -c "^IRP" "$OUT"
