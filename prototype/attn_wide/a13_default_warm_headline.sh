#!/usr/bin/env bash
# The WARM HEADLINE as the engine now ships it — no form flags at all.
#
# WHY THIS EXISTS SEPARATELY FROM a13_warm_headline.sh: that script prices knobs against each other and so
# must name them. This one passes nothing, which makes it the only witness that the committed DEFAULT really
# routes to the form the headline claims — the failure mode AGENTS records as "a gate default and an engine
# default that moved apart", caught here by asserting the run's own `attn_fattn_wide=` and `form=` echoes
# rather than trusting a flag. §51's convention: warm = `WAVEPASS gran=burst wall_ms` after `st.reset()`,
# medians of 3, one process per width; a missing WAVEPASS line is a failure, never a zero.
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-engine/build/decode_run}
OUT=${OUT:-tmp/a13ab}
TAG=${TAG:-a13-default-warm}
WIDTHS=${WIDTHS:-"512 1024 4096 16384 32000"}
REPS=${REPS:-3}
mkdir -p "$OUT"
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

one() {   # one <width> -> "<cold> <warm>"
  local w="$1" log="$OUT/default_warm_$w.log"
  "$BIN" --mtp-accept 0 --mtp-draft 0 --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$w.txt" --steps 1 --time-waves \
         --out "$OUT/tmpout" > "$log" 2>&1 || return 1
  local want_tk; want_tk=$(awk -F',' '{print NF}' "$OUT/ids$w.txt")
  local got; got=$(grep -E "^PREFILL" "$log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
  [ "$got" = "$want_tk" ] || { echo "width $w prefilled $got, file has $want_tk" >&2; return 1; }
  got=$(grep -E "^NUMERICS" "$log" | head -1 | sed -E 's/.*attn_fattn_wide=([a-z_]+).*/\1/')
  [ "$got" = wide_llama ] || { echo "width $w: default echoes attn_fattn_wide=$got, not wide_llama — the flip did not land" >&2; return 1; }
  got=$(grep -E "^ATTN_WIDE" "$log" | head -1 | sed -E 's/.*form=([a-z0-9]+).*/\1/')
  [ "$got" = wide128llama ] || { echo "width $w: default echoes form=$got" >&2; return 1; }
  local cold warm
  cold=$(grep -E "^RESULT" "$log" | tail -1 | sed -E 's/.*ms_total=([0-9.]+).*/\1/')
  warm=$(grep "^WAVEPASS gran=burst " "$log" | head -1 | sed -E 's/.*wall_ms=([0-9.]+).*/\1/')
  case "$warm" in ''|*[!0-9.]*|0|0.0) echo "width $w: no WAVEPASS warm number ($log)" >&2; return 1;; esac
  echo "$cold $warm"
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

for w in $WIDTHS; do
  : > "$OUT/default_warm_$w.txt"
  for rep in $(seq "$REPS"); do
    pair=$(one "$w") || fail "default warm leg failed at width $w"
    echo "$pair" >> "$OUT/default_warm_$w.txt"
  done
  med=$(awk '{print $2}' "$OUT/default_warm_$w.txt" | sort -n | awk '{x[NR]=$1}END{printf "%.3f", x[int((NR+1)/2)]}')
  lo=$(awk '{print $2}' "$OUT/default_warm_$w.txt" | sort -n | head -1)
  hi=$(awk '{print $2}' "$OUT/default_warm_$w.txt" | sort -n | tail -1)
  printf "width %-6s warm %9s ms  [%s, %s]  %8s prompt tok/s\n" "$w" "$med" "$lo" "$hi" \
         "$(awk -v m="$med" -v n="$w" 'BEGIN{printf "%.0f", n/m*1000}')"
done
