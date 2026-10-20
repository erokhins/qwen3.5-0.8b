#!/usr/bin/env bash
# A13 timing window — does the llama-port wide form (knob 2) beat the A12 wide form (knob 1) and the
# shipping WMMA wave (knob 0)?  Main-agent-only: this takes timings, so zero subagents, blocking, one
# gpu_lock hold, best-of-N, paired and interleaved inside the window (bench SS57's "a ruler taken in
# another window is not an A/B").
#
# WHAT THE CHANGE IS. `prototype/attn_wide/evidence/a13_worker_notes.md` bills the ported round at 20.56 non-matrix
# slots per matrix slot against A12's 24.05 (round-scoped), and the worker's own residual risk is the price:
# spilled VGPRs 163 -> 920, private segment 176 -> 720 B a thread. Whether the slot cut pays for the spill is
# exactly the question no CPU instrument can answer. Knob 2 is BYTE-IDENTICAL to knob 1 on five geometries
# (1,960 files), so no numerics band is in play: this is a pure speed A/B with a third arm for the shipper.
#
# CANARIES, asserted on every run before its number is kept (AGENTS' 2026-10-17 row: a behavioural field the
# change is REQUIRED to move, read in the same breath as the measurement, because engine/build.sh is not
# bit-reproducible and a stale binary agrees with itself):
#   1. `NUMERICS attn_fattn_wide=`   must read fattn | wide | wide_llama per arm;
#   2. `ATTN_WIDE form=`             must read wide128 for knob 1 and wide128llama for knob 2;
#   3. `ATTN_WIDE scratch=`          the required-to-move field: knob 2 must spill strictly more than
#                                    knob 1, and both must be > 0. Equal values abort the window, because
#                                    that is what a stale or mis-selected launch looks like.
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-engine/build/decode_run}
OUT=${OUT:-tmp/a13ab}
TAG=${TAG:-a13-ab}
mkdir -p "$OUT"
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

# arm -> wanted attn_fattn_wide / wanted form / wanted scratch
want_core() { case "$1" in 0) echo fattn ;; 1) echo wide ;; 2) echo wide_llama ;; esac; }
want_form() { case "$1" in 0) echo - ;; 1) echo wide128 ;; 2) echo wide128llama ;; esac; }

run() {   # run <knob> <lanes> -> ms ; appends the canary fields to $OUT/canaries.txt
  local k="$1" lanes="$2" log="$OUT/run_${k}_${lanes}.log"
  "$BIN" --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$lanes.txt" --steps 1 \
         --attn-fattn 1 --attn-fattn-wide "$k" --out "$OUT/tmpout" > "$log" 2>&1 \
      || fail "knob=$k lanes=$lanes rc=$? (see $log)"
  local core form scratch prev_scratch
  # NOTE on shape: run() is called inside $(...), so `fail` here would exit a SUBSHELL and the window
  # would keep going and print a table of zeros (seen on this harness's first run). Every canary below
  # therefore `return 1`s and the CALLER aborts the script.
  local tk want_tk; want_tk=$(awk -F',' '{print NF}' "$OUT/ids$lanes.txt")
  tk=$(grep -E "^PREFILL" "$log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
  [ "$tk" = "$want_tk" ] || { echo "CANARY-FAIL: knob $k lanes $lanes prefilled tokens=$tk, wanted $want_tk" >&2; return 1; }
  core=$(grep -E "^NUMERICS" "$log" | head -1 | sed -E 's/.*attn_fattn_wide=([a-z_]+).*/\1/')
  [ "$core" = "$(want_core "$k")" ] || { echo "CANARY-FAIL: knob $k lanes $lanes attn_fattn_wide=$core, wanted $(want_core "$k")" >&2; return 1; }
  if [ "$k" = 0 ]; then
    form="-"; scratch=0
  else
    form=$(grep -E "^ATTN_WIDE" "$log" | head -1 | sed -E 's/.*form=([a-z0-9]+).*/\1/')
    scratch=$(grep -E "^ATTN_WIDE" "$log" | head -1 | sed -E 's/.*scratch=([0-9]+).*/\1/')
    [ "$form" = "$(want_form "$k")" ] || { echo "CANARY-FAIL: knob $k lanes $lanes form=$form, wanted $(want_form "$k")" >&2; return 1; }
    [ "$scratch" -gt 0 ] 2>/dev/null || { echo "CANARY-FAIL: knob $k scratch=$scratch (expected a spill)" >&2; return 1; }
    if [ "$k" = 2 ]; then
      prev_scratch=$(awk '$1==1{print $3; exit}' "$OUT/canaries.txt" 2>/dev/null)
      if [ -n "$prev_scratch" ] && [ "$scratch" -le "$prev_scratch" ]; then
        echo "CANARY-FAIL: knob 2 scratch=$scratch <= knob 1 scratch=$prev_scratch — not two forms" >&2; return 1; fi
    fi
  fi
  echo "$k $form $scratch" >> "$OUT/canaries.txt"
  local ms
  ms=$(grep -E "^RESULT" "$log" | tail -1 | sed -E 's/.*ms_total=([0-9.]+).*/\1/')
  case "$ms" in ''|*[!0-9.]*) echo "CANARY-FAIL: knob $k lanes $lanes no ms parsed" >&2; return 1;; esac
  echo "$ms"
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1
: > "$OUT/canaries.txt"

for lanes in ${LANES:-512 1024 4096 16384 32000}; do
  reps=${REPS:-3}; [ "$lanes" = 32000 ] && reps=2
  : > "$OUT/ab_$lanes.txt"
  for rep in $(seq "$reps"); do
    for k in 0 1 2; do
      ms=$(run "$k" "$lanes") || fail "window aborted at knob $k lanes $lanes (see CANARY-FAIL above)"
      echo "$k $ms" >> "$OUT/ab_$lanes.txt"
    done
  done
  echo "=== lanes=$lanes  best of $reps  (canaries core/form/scratch asserted every run) ==="
  for k in 0 1 2; do
    best=$(awk -v a="$k" '$1==a{if(m==""||$2<m)m=$2}END{printf "%.2f", m}' "$OUT/ab_$lanes.txt")
    base=$(awk '$1==0{if(m==""||$2<m)m=$2}END{printf "%.2f", m}' "$OUT/ab_$lanes.txt")
    lo=$(awk -v a="$k" '$1==a{if(m==""||$2<m)m=$2}END{printf "%.1f", m}' "$OUT/ab_$lanes.txt")
    hi=$(awk -v a="$k" '$1==a{if(m==""||$2>m)m=$2}END{printf "%.1f", m}' "$OUT/ab_$lanes.txt")
    printf "  knob %s  %9s ms  [%s, %s]  %6s %% vs knob0  %7s tok/s\n" "$k" "$best" "$lo" "$hi" \
      "$(awk -v a="$best" -v b="$base" 'BEGIN{printf "%+.2f", (a-b)/b*100}')" \
      "$(awk -v b="$best" -v n="$lanes" 'BEGIN{printf "%.0f", n/b*1000}')"
  done
done
echo "=== canary fields seen (knob form scratch) ==="
sort -u "$OUT/canaries.txt"
