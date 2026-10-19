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
#
# WHY `run_leg` EXISTS — the first run of this window burned a device window and told us nothing. Each
# invocation used to be `timeout ... "$BIN" ... | grep -E ... >> "$OUT"`, which throws away BOTH the exit
# status and every line the regex did not match. The binary segfaulted on its first-ever run (a `__global__`
# called as a host function), printed nothing, and the window's only symptom was "0 WL rows" with no reason
# attached — the row-count check did catch the run, but the diagnosis needed a second device pass. `run_leg`
# keeps rc, keeps the unmatched output, and fails LOUDLY per invocation: a crash is now a row in the log, not
# an absence of one. The final check is also derived rather than hand-counted: every invocation increments
# WANT, so the parser can no longer disagree with an expectation someone typed.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
BIN=../../tmp/issue_rate_probe_latency/wmma_latency
test -x "$BIN" || { echo "FATAL: $BIN missing. Run: bash build_wmma_latency.sh" >&2; exit 1; }

OUT=${OUT:-evidence/wmma-latency.txt}
mkdir -p "$(dirname "$OUT")" || exit 1
: > "$OUT" || exit 1
test -w "$OUT" || { echo "FATAL: cannot write $OUT — a window that saves nothing spent the device time for nothing" >&2; exit 1; }

RAW=$(mktemp) || exit 1
trap 'rm -f "$RAW"' EXIT
WANT=0     # invocations attempted; the row count must equal it
BAD=0      # invocations that crashed, timed out, or printed no row

say() { echo "$@" | tee -a "$OUT"; }

# run_leg <label> <args...> — one invocation, one row, no silent failure.
run_leg() {
    local label=$1; shift
    WANT=$((WANT + 1))
    timeout 300 "$BIN" "$@" > "$RAW" 2>&1
    local rc=$?
    local nrows
    nrows=$(grep -cE "^(WL|REFUSED|IRP-ERROR)" "$RAW" || true)
    if [ "$rc" -ne 0 ] || [ "${nrows:-0}" -eq 0 ]; then
        BAD=$((BAD + 1))
        say "INVOKED-FAILED leg=$label rc=$rc rows=$nrows args: $*"
        sed 's/^/#> /' "$RAW" >> "$OUT"      # keep whatever it did print, marked so it cannot be parsed
        return 1
    fi
    grep -E "^(WL|REFUSED|IRP-ERROR)" "$RAW" >> "$OUT"
    return 0
}

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
run_leg wl_empty_i1 --only wl_empty_i1 --launches 20 --reps 5

say "=== 2. F32 accumulator, the ILP curve (i1 is the latency, i8 approaches the roof)"
for k in wl_f32_i1 wl_f32_i2 wl_f32_i4 wl_f32_i8; do
    run_leg "$k" --only "$k" --launches 20 --reps 5
done

say "=== 2b. the SAME F32 legs on the SHORT pair (64/128), because the F16 legs can only be run there"
say "===     (see the refusal note above): a cross-width rate comparison is only honest inside one pair,"
say "===     and the pair itself is worth ~17 % at ILP 1 (42.4 vs 51.3 TFLOP/s at i1), which is more than"
say "===     the difference being asked about."
for k in wl_f32_i1 wl_f32_i4; do
    run_leg "$k@64-128" --only "$k" --iters-lo 64 --iters-hi 128 --launches 20 --reps 5
done

say "=== 3. F16 accumulator, same geometry. --scale 0.0625 (2^-4, exact in F16) is not arbitrary: the"
say "===    exactness budget is on the accumulated VALUE (F16 holds integers exactly to 2048), the step is"
say "===    128*ilp*scale^2 per iteration, so 0.5 would cap iters-hi at 64/ilp -- too short for a slope."
say "===    At 0.0625 the cap clears every pair this window uses at every ILP."
say "===    --iters pair 64/128 rather than the default 128/512: every F16 spelling refused at iters-hi=512"
say "===    with relative drift EXACTLY 0.5 (single op_sel and the two-op_sel form alike) while the same leg"
say "===    read back exactly at iters-hi=128. That is the leg's uniform-fragment value model, not the atom;"
say "===    the slope method makes the pair choice free, so the pair is taken where the witness is green."
for k in wl_f16_i1 wl_f16_i2 wl_f16_i4 wl_f16_i8; do
    run_leg "$k" --only "$k" --scale 0.0625 --iters-lo 64 --iters-hi 128 --launches 20 --reps 5
done

say "=== 3b. F16 in the form the engine would actually ship: op_sel 0 then 1 per chain step (llama's RDNA"
say "===     spelling, mma.cuh:1556-1566). If these pass where section 3 refused, the refusal was the"
say "===     single-op_sel D-block layout and not a broken chain."
for k in wl_f16x2_i1 wl_f16x2_i2 wl_f16x2_i4 wl_f16x2_i8; do
    run_leg "$k" --only "$k" --scale 0.0625 --iters-lo 64 --iters-hi 128 --launches 20 --reps 5
done

say "=== 4. slope stability: the same legs at two DIFFERENT pairs. The bill is the slope between"
say "===    iters-lo and iters-hi, so two pairs must agree; if they do not, the leg is not latency-shaped"
say "===    and neither number can be quoted (the recorded reason --iters is never divided into itself)."
for pair in "64 128" "256 512"; do
  set -- $pair
  for k in wl_f32_i1 wl_f16_i1 wl_empty_i1; do
    case "$k" in
      *f16*) sc="--scale 0.0625";;
      *)     sc="";;
    esac
    run_leg "$k@$1-$2" --only "$k" $sc --iters-lo "$1" --iters-hi "$2" --launches 20 --reps 5
  done
done

say "=== 4b. the same stability question for the two-op_sel form: two different pairs must agree"
for pair in "64 128" "256 512"; do
  set -- $pair
  run_leg "wl_f16x2_i1@$1-$2" --only wl_f16x2_i1 --scale 0.0625 --iters-lo "$1" --iters-hi "$2" \
          --launches 20 --reps 5
done

say "=== 5. cross-check against the ledger's own WMMA roof: 32 wavefronts/CU (= 320 blocks), ILP 4"
say "# wl_f32_i4 here should reproduce the 52.63 TFLOP/s row in prototype/issue_rate_probe/results.md §1; if it does not, these"
say "# rows and that row are not the same measurement and none of this is comparable."
run_leg wl_f32_i4@320 --only wl_f32_i4 --blocks 320 --launches 20 --reps 5
run_leg wl_f16_i4@320 --only wl_f16_i4 --scale 0.0625 --iters-lo 64 --iters-hi 128 --blocks 320 --launches 20 --reps 5
run_leg wl_f16x2_i4@320 --only wl_f16x2_i4 --scale 0.0625 --iters-lo 64 --iters-hi 128 --blocks 320 --launches 20 --reps 5

say "# util at end: $(bash ../../tools/gpu_lock.sh util | tail -1)"
say "# sclk at end: $(sclk)"

n=$(grep -c "^WL " "$OUT" || true)
r=$(grep -c "^REFUSED" "$OUT" || true)
f=$(grep -c "^INVOKED-FAILED" "$OUT" || true)
say "=== invocations: $WANT   rows: $n WL, $r REFUSED, $f INVOKED-FAILED   (a REFUSED row is a witness that fired,"
say "===    an INVOKED-FAILED row is the tool itself failing; only the first is a measurement result)"
# Every invocation must be ACCOUNTED FOR, which is not the same as every invocation passing: a REFUSED row
# is a witness doing its job (the F16 legs at iters-hi=512 are refused by design, see the note above the
# F16 section), and an INVOKED-FAILED row is the tool breaking. The old check demanded WL == WANT, which
# can never hold while a documented refusal is in the matrix -- so a correct window printed FATAL and a
# reader had to know that was wrong. Now: WL + REFUSED must equal the invocations attempted, and no
# invocation may have crashed or printed nothing.
if [ "$((n + r))" -ne "$WANT" ] || [ "${BAD:-0}" -ne 0 ]; then
  say "FATAL: $WANT invocations accounted for $((n + r)) results ($n WL, $r REFUSED) and $BAD tool failures."
  say "       A short log is a red result, not a quiet one: read the INVOKED-FAILED and REFUSED lines before"
  say "       drawing anything from the rows that did print."
  exit 1
fi
say "OK: every invocation produced exactly one accounted-for row ($n measured, $r refused by witness)."
