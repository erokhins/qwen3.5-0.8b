#!/usr/bin/env bash
# llama_phase_compare.sh — re-run the §21/§22 comparison of this engine against llama.cpp, by phases.
#
# WHY. §21 established llama.cpp prefills this model 1.7-2.3x faster; §22 localised it to two gaps: GDN
# (428 ms vs <=47.4, a LAUNCH-STRUCTURE gap) and attention (263 ms vs <=52.5, a KERNEL-SHAPE gap). Since
# then GDN got the looped launch (§24: 449 -> 266 ms) and attention got the WMMA round (§33: 263 -> 132.6 ms).
# This re-measures the ranking with the same harness and the same matched row, because the ranking is what
# decides the next funded bet: if GDN is still the bigger gap, the money goes to a chunked GDN walk and not
# to a third attention pass.
#
# THE RULES THAT MAKE THE COMPARISON MEAN SOMETHING (all three are §21/§22's, kept deliberately):
#   * everything inside ONE lock hold, ours re-taken in the same window (§6: a ruler from another window is
#     not an A/B);
#   * our family ms come UNTRACED from --time-waves; llama.cpp's traced numbers are UPPER BOUNDS, because
#     trace overhead scales with launch count, which is the axis under study — so every "N x slower" that
#     uses a traced llama.cpp figure is a conservative floor, never a measurement of theirs;
#   * the matched row is -ub 512 -fa on -t 32, because kMaxWaveRows = 512 and §21 showed -ub is a huge lever
#     (their -ub 128 row is SLOWER than us). The -t 4 row is still recorded: it beat -t 32 by 24 % and §21
#     refused to drop a row because it flattered the comparison.
#
# MAIN-AGENT DEVICE WINDOW: blocking, zero subagents, one `tools/gpu_lock.sh` hold.
# Usage:
#   bash tools/llama_phase_compare.sh            # both legs
#   LEGS=bench bash tools/llama_phase_compare.sh # headline rows only (no rocprofv3)
#   LEGS=trace bash tools/llama_phase_compare.sh # traces only
set -uo pipefail
cd "$(dirname "$0")/.."
LB=${LB:-engines/llama.cpp/build-hip/bin/llama-bench}
LM=${LM:-models/gguf/Qwen3.5-0.8B.f16.gguf}
BIN=engine/build/decode_run
PY=${PY:-.venv/bin/python}
MODEL=models/Qwen3.5-0.8B
IDS=$(cat progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt)
OURWIDTHS=${OURWIDTHS:-"512 1024 4096"}
REPS=${REPS:-3}
LEGS=${LEGS:-bench trace}
TAG=${TAG:-llama-phase}
OUT=tmp/gcc/llamaphase
mkdir -p "$OUT"

clk() { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk" | tr '\n' ' '; }
res() { grep -h "^RESULT" "$1" | sed -E 's/.*ms_total=([0-9.]+).*/\1/' | tail -1; }
median() { sort -g "$@" | awk '{a[NR]=$1} END{if(NR==0){print "NA";exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

[ -x "$LB" ] || fail "llama-bench not built at $LB"
[ -f "$LM" ] || fail "gguf not found at $LM"
[ -x "$BIN" ] || fail "engine binary missing"

tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
echo "CLK@start: $(clk)"
echo "llama-bench: $("$LB" --version 2>/dev/null | head -1)"
echo "model: $LM"

if [[ " $LEGS " == *" bench "* ]]; then
  echo
  echo "== leg 1: throughput rows, one hold, medians of $REPS =="
  for ROW in "-ub 512 -fa on -t 32" "-ub 512 -fa off -t 32" "-ub 128 -fa on -t 32" "-ub 512 -fa on -t 4"; do
    for PP in 512 1024 4096; do
      for R in $(seq 1 "$REPS"); do
        timeout 900 "$LB" -m "$LM" -p "$PP" -n 0 $ROW -ngl 99 -b 2048 -r 1 2>>"$OUT/llama.err" \
            | tee -a "$OUT/llama_${ROW// /_}_${PP}.txt" >/dev/null || fail "llama-bench pp$PP row '$ROW' rep $R"
        sleep 3
      done
    done
  done
  # ours, same window, untraced, and PROMPT-ONLY: llama's row is `-n 0`, so ours must be `--steps 0`.
  # (`tools/prefill_sweep.py` was tried here and is the wrong instrument: its ms_total carries one priced
  # generation step, ~20 % more time at these widths, and every ratio would have inherited that.)
  # Ids: 4,096 uses the pinned ids4096.txt as §21 did; 512/1,024 use synthetic 1..W, which is CLOSER to
  # what llama-bench feeds than the pinned prose is, and §21's own non-comparable note is honoured.
  for W in $OURWIDTHS; do
    IDS_W=$( [ "$W" = "4096" ] && printf '%s' "$IDS" || seq -s, 1 "$W" )
    for R in $(seq 1 "$REPS"); do
      "$BIN" --model "$MODEL" --ids "$IDS_W" --steps 0 --gemm lt --gemm-pins all --gdn-loop 512 \
          --out "$OUT/our_${W}_$R" > "$OUT/our_${W}_$R.log" 2>&1 || fail "ours W=$W rep $R"
      sleep 3
    done
  done
  echo "CLK@mid: $(clk)"
  echo
  echo "-- llama.cpp pp medians (tok/s) --"
  for ROW in "-ub 512 -fa on -t 32" "-ub 512 -fa off -t 32" "-ub 128 -fa on -t 32" "-ub 512 -fa on -t 4"; do
    line="llama $ROW:"
    for PP in 512 1024 4096; do
      # llama-bench prints "<tok/s> ± <spread>" in the test column; reading the last float grabs the
      # ± spread, which is how a comparison harness reports 0.00 tok/s for a perfectly healthy engine.
      V=$(sed -n "s/.*| *pp[0-9]* *| *\([0-9][0-9.]*\) ±.*/\1/p" "$OUT/llama_${ROW// /_}_${PP}.txt" | tail -1)
      [ -n "$V" ] || fail "could not parse a throughput out of llama pp$PP row '$ROW' - read the log, do not guess"
      line="$line pp$PP=$V"
    done
    echo "$line"
  done
  echo "-- ours (prompt-only, --steps 0, same hold) and the matched ratio --"
  for W in $OURWIDTHS; do
    M=$(for R in $(seq 1 "$REPS"); do res "$OUT/our_${W}_${R}.log"; done | median -)
    OUR_TOKS=$(awk -v m="$M" -v w="$W" 'BEGIN{printf "%.0f", w*1000/m}')
    LM=$(sed -n "s/.*| *pp[0-9]* *| *\([0-9][0-9.]*\) ±.*/\1/p" "$OUT/llama_-ub_512_-fa_on_-t_32_${W}.txt" | tail -1)
    echo "  W=$W ours ${M} ms = ${OUR_TOKS} tok/s ; llama(-ub512 fa-on t32) ${LM} tok/s ; llama/ours = "\
        "$(awk -v l="$LM" -v o="$OUR_TOKS" 'BEGIN{printf "%.2f", l/o}')"x
  done
  echo
fi

if [[ " $LEGS " == *" trace "* ]]; then
  echo
  echo "== leg 2: rocprofv3 kernel traces (SHARES and LAUNCH COUNTS only; llama.cpp ms are upper bounds) =="
  command -v rocprofv3 >/dev/null || fail "rocprofv3 not on PATH"
  timeout 1800 rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$OUT/llama_pp4096" -o pp4096 -- \
      "$LB" -m "$LM" -p 4096 -n 0 -ub 512 -fa on -ngl 99 -t 32 -b 2048 -r 1 -o csv \
      > "$OUT/llama_pp4096.stdout" 2>&1 || fail "llama trace"
  timeout 1800 rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$OUT/our_pp4096" -o pp4096 -- \
      "$BIN" --model "$MODEL" --ids "$IDS" --steps 0 --time-waves --gemm lt --gemm-pins all --gdn-loop 512 \
      --out "$OUT/our_trace" > "$OUT/our_pp4096.stdout" 2>&1 || fail "engine trace"
  echo "traces in $OUT/{llama,our}_pp4096 — parse with the §22 pass-count division (llama.cpp holds warmup+timed = /2; ours holds three passes = /3)."
fi

echo "CLK@end: $(clk)"
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
echo "DONE. logs in $OUT/"
