#!/usr/bin/env bash
# R1 — the wave-width probe: is a 512-row body the right size, or just the size we picked?
#
# WHY THIS EXISTS. A12's attribution window measured the OTHER direction: at fixed total lanes,
# `--wave-rows` 512 -> 256 costs +10.8 %, -> 128 costs +29 %, -> 64 costs +86 %. That is per-body
# overhead, and it is convex, which says a body BIGGER than 512 may be cheaper — but `--wave-rows` is
# capped at `kMaxWaveRows`, so the interesting direction cannot be measured by a flag. It needs three
# builds of one source line. `kMaxWaveRows` is also the thing that keeps the A12 wide block at 32
# workgroups per launch (2 x ceil(rows/32)), so if it is cheap it is a win twice: per-body overhead on
# the shipping form, and the precondition for the wide shape's residency (>= 1,280 rows/body).
#
# WHAT IT DOES NOT DECIDE. Numerics. A wider body is NOT bit-identical: the library GEMM route is
# explicitly not byte-identical across wave widths (judged by the cascade band), and attention bodies
# change shape too. This window is timing only; a winner goes through `engine/gate.py` before it ships.
#
# CANARY. Every run asserts its own `wave_rows_cap=` (which binary it is) and its own `wave_rows=`
# (which body it ran) BEFORE the value is recorded — the stale-binary failure this box has been bitten
# by twice. A run whose canary does not read as expected aborts the window.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-tmp/wave_rows}
mkdir -p "$OUT"
TAG=${TAG:-r1-wavewidth}
REPS=${REPS:-3}
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

# arm = <binary-label> <wave-rows>
ARMS="w512:512 w1024:1024 w2048:2048 w2048c:512"

# one run -> "<ms_total>" and the canary check inline
run() {   # run <label> <wave_rows> <lanes>
  local bin="$OUT/decode_run.$1" rows="$2" lanes="$3" cap="${1#w}"
  [ "$1" = "w2048c" ] && { bin="$OUT/decode_run.w2048"; cap=2048; }
  "$bin" --mtp-accept 0 --mtp-draft 0 --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$lanes.txt" --steps 1 \
         --wave-rows "$rows" --out "$OUT/tmpout" > "$OUT/run.log" 2>&1 || fail "run $1 rows=$rows lanes=$lanes rc=$?"
  # TWO canaries, both read off the run's own lines, before the number is kept:
  #   `NUMERICS wave_rows_cap=` is compiled into the binary, so it is what distinguishes the three
  #     builds from each other -- a stale build agrees with itself and cannot be caught any other way;
  #   `PREFILL ... wave_rows=` is what the run actually ran its bodies at.
  local echo_cap echo_rows
  echo_cap=$(grep -E "^NUMERICS" "$OUT/run.log" | head -1 | sed -E 's/.*wave_rows_cap=([0-9]+).*/\1/')
  echo_rows=$(grep -E "^PREFILL" "$OUT/run.log" | head -1 | sed -E 's/.*wave_rows=([0-9]+).*/\1/')
  [ "$echo_cap" = "$cap" ] || fail "canary: lanes=$lanes $1 printed wave_rows_cap=$echo_cap, built $cap"
  [ "$echo_rows" = "$rows" ] || fail "canary: lanes=$lanes $1 printed wave_rows=$echo_rows, asked $rows"
  grep -E "^RESULT" "$OUT/run.log" | sed -E 's/.*ms_total=([0-9.]+).*/\1/'
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

for lanes in 4096 16384 32768; do
  reps=$REPS; [ "$lanes" = 32768 ] && reps=2
  : > "$OUT/rows_$lanes.txt"
  for rep in $(seq "$reps"); do
    for arm in $ARMS; do
      lbl=${arm%%:*}; rows=${arm##*:}
      echo "$lbl $rows $(run "$lbl" "$rows" "$lanes")" >> "$OUT/rows_$lanes.txt"
    done
  done
  echo "=== lanes=$lanes  (best of $reps; canary wave_rows asserted per run) ==="
  for arm in $ARMS; do
    lbl=${arm%%:*}; rows=${arm##*:}
    best=$(awk -v l="$lbl" '$1==l{if(m==""||$3<m)m=$3}END{printf "%.1f", m}' "$OUT/rows_$lanes.txt")
    base=$(awk '$1=="w512"{if(m==""||$3<m)m=$3}END{printf "%.1f", m}' "$OUT/rows_$lanes.txt")
    tok=$(awk -v b="$best" -v n="$lanes" 'BEGIN{printf "%.0f", n/b*1000}')
    printf "  %-6s rows=%-5s %8s ms  %6s %% vs w512  %6s tok/s\n" "$lbl" "$rows" "$best" \
           "$(awk -v a="$best" -v b="$base" 'BEGIN{printf "%.2f", (a-b)/b*100}')" "$tok"
  done
done
