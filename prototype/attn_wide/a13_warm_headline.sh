#!/usr/bin/env bash
# A13 warm headline — the §51 convention applied to the ported form.
#
# CONVENTION (progress/2026-10-21/performance-status-2026-10-21.md §51, user decision 2026-10-19): the headline is the WARM number,
# `WAVEPASS gran=burst wall_ms` — `st.reset()` then `prefill_wave` again on the same prompt, first-execution
# cost excluded — medians of 3, one process per width and arm. Cold `RESULT ms_total` stays first-request
# latency and must never be compared against another engine's warm figure, which is why this script prints
# both and gates the headline on the warm one.
#
# Baseline it has to beat (§51, same convention, knob 0): 512 -> 10,637 tok/s; 1,024 -> 11,276;
# 4,096 -> 9,790; 16,384 -> 6,125; 32,768 -> 3,924.
#
# CANARIES per run, before any number is kept: `NUMERICS attn_fattn_wide=`, `ATTN_WIDE form=` and
# `PREFILL tokens=` equal to the width read off the ids file. A missing `WAVEPASS` line is a FAILURE, not a
# zero (prototype/perprompt/a7_lib.sh's whole reason for existing).
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-engine/build/decode_run}
OUT=${OUT:-tmp/a13ab}
TAG=${TAG:-a13-warm}
WIDTHS=${WIDTHS:-"512 1024 4096 16384 32000"}
REPS=${REPS:-3}
KNOBS=${KNOBS:-"0 2"}
mkdir -p "$OUT"
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
want_core() { case "$1" in 0) echo fattn ;; 1) echo wide ;; 2) echo wide_llama ;; esac; }
want_form() { case "$1" in 0) echo - ;; 1) echo wide128 ;; 2) echo wide128llama ;; esac; }

one() {   # one <knob> <width> -> "<cold_ms> <warm_ms>"
  local k="$1" w="$2" log="$OUT/warm_${k}_${w}_$RANDOM.log"
  "$BIN" --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$w.txt" --steps 1 --time-waves \
         --attn-fattn 1 --attn-fattn-wide "$k" --out "$OUT/tmpout" > "$log" 2>&1 || return 1
  local want_tk; want_tk=$(awk -F',' '{print NF}' "$OUT/ids$w.txt")
  local got; got=$(grep -E "^PREFILL" "$log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
  [ "$got" = "$want_tk" ] || { echo "canary tokens: knob $k w=$w prefilled $got want $want_tk" >&2; return 1; }
  got=$(grep -E "^NUMERICS" "$log" | head -1 | sed -E 's/.*attn_fattn_wide=([a-z_]+).*/\1/')
  [ "$got" = "$(want_core "$k")" ] || { echo "canary core: knob $k w=$w echoed $got" >&2; return 1; }
  if [ "$k" != 0 ]; then
    got=$(grep -E "^ATTN_WIDE" "$log" | head -1 | sed -E 's/.*form=([a-z0-9]+).*/\1/')
    [ "$got" = "$(want_form "$k")" ] || { echo "canary form: knob $k w=$w echoed $got" >&2; return 1; }
  fi
  local cold warm
  cold=$(grep -E "^RESULT" "$log" | tail -1 | sed -E 's/.*ms_total=([0-9.]+).*/\1/')
  warm=$(grep "^WAVEPASS gran=burst " "$log" | head -1 | sed -E 's/.*wall_ms=([0-9.]+).*/\1/')
  for v in "$cold" "$warm"; do case "$v" in ''|*[!0-9.]*|0|0.0) echo "no cold/warm number for knob $k w=$w (see $log)" >&2; return 1;; esac; done
  echo "$cold $warm"
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

for w in $WIDTHS; do
  : > "$OUT/warm_$w.txt"
  for rep in $(seq "$REPS"); do
    for k in $KNOBS; do
      pair=$(one "$k" "$w") || fail "knob $k width $w produced no usable pair (canary or missing WAVEPASS)"
      echo "$k $pair" >> "$OUT/warm_$w.txt"
    done
  done
  echo "=== width=$w  warm = median of $REPS \`WAVEPASS gran=burst wall_ms\` (cold shown, not headlined) ==="
  for k in $KNOBS; do
    med=$(awk -v a="$k" '$1==a{print $3}' "$OUT/warm_$w.txt" | sort -n | awk '{x[NR]=$1}END{printf "%.3f", x[int((NR+1)/2)]}')
    cold=$(awk -v a="$k" '$1==a{print $2}' "$OUT/warm_$w.txt" | sort -n | awk '{x[NR]=$1}END{printf "%.1f", x[1]}')
    base=$(awk '$1==0{print $3}' "$OUT/warm_$w.txt" | sort -n | awk '{x[NR]=$1}END{printf "%.3f", x[int((NR+1)/2)]}')
    printf "  knob %s  warm %9s ms  (cold best %s)  %8s prompt tok/s  %7s %% vs knob0\n" "$k" "$med" "$cold" \
      "$(awk -v m="$med" -v n="$w" 'BEGIN{printf "%.0f", n/m*1000}')" \
      "$(awk -v m="$med" -v b="$base" 'BEGIN{printf "%+.2f", (m-b)/b*100}')"
  done
done
