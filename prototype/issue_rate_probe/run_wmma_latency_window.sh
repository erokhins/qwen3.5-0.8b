#!/usr/bin/env bash
# The device window for the WMMA LATENCY leg. MAIN AGENT ONLY: run it blocking, with tools/gpu_lock.sh
# already acquired and zero subagents active. A worker prepares this script and hands it over; it does
# NOT acquire the lock itself, because a caller that holds the lock and then invokes a self-locking driver
# deadlocks the box silently (AGENTS.md, the gpu_lock deadlock row) — so lock ownership stays with the
# caller on both sides.
#
#   bash prototype/issue_rate_probe/build_wmma_latency.sh
#   tools/gpu_lock.sh acquire wmma-latency
#   tools/gpu_lock.sh util
#   bash prototype/issue_rate_probe/run_wmma_latency_window.sh
#   tools/gpu_lock.sh release
#
# WHAT IT PRINTS, and the reading order. Rows come out in the order the questions should be answered:
#   1. wl_empty_i1                — loop overhead. Every wmma row is read as (row - this).
#   2. wl_f32_i{1,2,4,8}          — the F32 accumulator latency curve at the shipped occupancy.
#   3. wl_f16_i{1,2,4,8}          — the same curve with the F16 accumulator.
#   4. the pair-stability probe — the bill is the SLOPE of the leg's time against its own chain length, so
#                                   two different (iters-lo, iters-hi) pairs must give the same clock
#                                   figure. If they do not, the leg is throughput- or dispatch-limited and
#                                   the ILP=1 number is not a latency. A single --iters divided into itself
#                                   is never used, because that charges dispatch, the fragment loads and
#                                   first touch to the instruction.
#   5. the high-occupancy cross-check — the ILP=4 rows at 32 wavefronts/CU must land near the existing
#                                   52.63 TFLOP/s WMMA roof row in prototype/issue_rate_probe/results.md §1. If it does not, this
#                                   tool and that row are not measuring the same instruction stream and
#                                   nothing here is comparable to the ledger.
# REFUSED rows are rows the witnesses rejected (spill, exactness range, read-back mismatch). They are
# evidence, not noise, and a window that prints zero WL rows is a red result: the count check at the end
# says so out loud instead of leaving a short log to be misread.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
BIN=../../tmp/issue_rate_probe_latency/wmma_latency
test -x "$BIN" || { echo "FATAL: $BIN missing. Run: bash build_wmma_latency.sh" >&2; exit 1; }

OUT=${OUT:-evidence/wmma-latency.txt}
mkdir -p "$(dirname "$OUT")" || exit 1
: > "$OUT" || exit 1
test -w "$OUT" || { echo "FATAL: cannot write $OUT — a window that saves nothing spent the device time for nothing" >&2; exit 1; }

say() { echo "$@" | tee -a "$OUT"; }

# The clock divisor is printed because every `clk_per_wmma_*` column is that divisor divided into a wall
# time: 2894 MHz is a MAX clock and `devices/strix-halo-compute.md` measured 2.206 GHz in ordinary
# operation, so a row read against the wrong divisor is wrong by up to 24 %. It is a device query, not a
# timing — the same class as the `gpu_lock.sh util` line beside it.
sclk() { rocm-smi --showclocks 2>/dev/null | grep -o "sclk clock level:.*" | head -1; }

say "=== WMMA latency leg — RAW rows, no verdict arithmetic here"
say "# util at start: $(bash ../../tools/gpu_lock.sh util | tail -1)"
say "# sclk at start: $(sclk)"
say "# occupancy default: 160 blocks x 128 threads = 640 wavefronts over the KFD 40 CUs ="
say "#   -> 16 wavefronts/CU, 8 wavefronts/SIMD (KFD simd_per_cu 2). The cross-check below uses"
say "#   --blocks 320 for 32 wavefronts/CU, which is the rate probe's saturation geometry."
say "# clocks: --clock-mhz 2894 (the divisor prototype/issue_rate_probe/README.md books (a MAX clock; sclk is printed with the rows)), same divisor as prototype/issue_rate_probe/results.md §1"

say "=== 1. loop-overhead baseline (subtract this from every row below)"
timeout 300 "$BIN" --only wl_empty_i1 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"

say "=== 2. F32 accumulator, the ILP curve (i1 is the latency, i8 approaches the roof)"
timeout 300 "$BIN" --only wl_f32_i1 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
timeout 300 "$BIN" --only wl_f32_i2 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
timeout 300 "$BIN" --only wl_f32_i4 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
timeout 300 "$BIN" --only wl_f32_i8 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"

say "=== 3. F16 accumulator, same geometry (exact range caps --iters at 512 with --scale 0.5)"
timeout 300 "$BIN" --only wl_f16_i1 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
timeout 300 "$BIN" --only wl_f16_i2 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
timeout 300 "$BIN" --only wl_f16_i4 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
timeout 300 "$BIN" --only wl_f16_i8 --launches 20 --reps 5 2>&1 | grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"

say "=== 4. slope stability: the same legs at two DIFFERENT pairs. The bill is the slope between"
say "===    iters-lo and iters-hi, so two pairs must agree; if they do not, the leg is not latency-shaped"
say "===    and neither number can be quoted (the recorded reason --iters is never divided into itself)."
for pair in "64 128" "256 512"; do
  set -- $pair
  for k in wl_f32_i1 wl_f16_i1 wl_empty_i1; do
    timeout 300 "$BIN" --only "$k" --iters-lo "$1" --iters-hi "$2" --launches 20 --reps 5 2>&1 |
      grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
  done
done

say "=== 5. cross-check against the ledger's own WMMA roof: 32 wavefronts/CU (= 320 blocks), ILP 4"
say "# wl_f32_i4 here should reproduce the 52.63 TFLOP/s row in prototype/issue_rate_probe/results.md §1; if it does not, these"
say "# rows and that row are not the same measurement and none of this is comparable."
timeout 300 "$BIN" --only wl_f32_i4 --blocks 320 --launches 20 --reps 5 2>&1 |
  grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"
timeout 300 "$BIN" --only wl_f16_i4 --blocks 320 --launches 20 --reps 5 2>&1 |
  grep -E "^(WL|REFUSED|IRP-ERROR)" >> "$OUT"

say "# util at end: $(bash ../../tools/gpu_lock.sh util | tail -1)"
say "# sclk at end: $(sclk)"

n=$(grep -c "^WL " "$OUT" || true)
r=$(grep -c "^REFUSED" "$OUT" || true)
say "=== rows: $n WL, $r REFUSED (18 rows expected: 1 baseline + 8 curve + 6 pair-stability + 3 cross-check)"
if [ "${n:-0}" -lt 18 ]; then
  say "FATAL: window produced $n of 15 expected WL rows. A short log is a red result, not a quiet one:"
  say "       read the REFUSED lines before drawing anything from the rows that did print."
  exit 1
fi
