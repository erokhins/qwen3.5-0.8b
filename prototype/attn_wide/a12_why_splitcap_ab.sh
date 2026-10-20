#!/usr/bin/env bash
# W4 — LEG C: the wide route's real engine gain is ~2 %, and the trace says where it comes from: it does
# not launch `attn_combine` (71.6 ms at 16,384 lanes) or `attn_slim_wave` (13.4 ms), i.e. it skips the
# split's partial round-trip. `--attn-split-cap` is the SHIPPED knob that shrinks that same round-trip on
# the shipping kernel. If a cap reproduces the wide arm's ~2 %, then the wide form pays numerics risk for
# a win the default route can already buy. 2 reps, interleaved, one lock, blocking, main agent.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=tmp/a12why
mkdir -p "$OUT"
BIN=engine/build/decode_run
TAG=${TAG:-a12-why-cap}
LANES=${LANES:-16384}
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
run() {  # run <wide> <cap> -> ms_total
  "$BIN" --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$LANES.txt" --steps 1 --attn-fattn 1 \
         --attn-fattn-wide "$1" --attn-split-cap "$2" --out "$OUT/tmpout" > "$OUT/cap.log" 2>&1 \
      || fail "wide=$1 cap=$2 rc=$?"
  grep -E "^RESULT" "$OUT/cap.log" | sed -E 's/.*ms_total=([0-9.]+).*/\1/'
}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

: > "$OUT/legC.txt"
for rep in 1 2 3; do
  echo "ship_nocap $(run 0 0)"      >> "$OUT/legC.txt"
  echo "wide1      $(run 1 0)"      >> "$OUT/legC.txt"
  for cap in 2 4 8 16; do echo "cap$cap $(run 0 "$cap")" >> "$OUT/legC.txt"; done
done
echo "lanes=$LANES  (best of 3, ms_total; canary: the SPLITCAP line each run printed)"
for k in ship_nocap wide1 cap2 cap4 cap8 cap16; do
  best=$(awk -v k="$k" '$1==k{if(m==""||$2<m)m=$2}END{printf "%.1f", m}' "$OUT/legC.txt")
  base=$(awk '$1=="ship_nocap"{if(m==""||$2<m)m=$2}END{printf "%.1f", m}' "$OUT/legC.txt")
  all=$(awk -v k="$k" '$1==k{printf "%s ", $2}' "$OUT/legC.txt")
  printf "%-11s %8s ms  %7s %% vs ship  reps: %s\n" "$k" "$best" \
         "$(awk -v a="$best" -v b="$base" 'BEGIN{printf "%.2f", (a-b)/b*100}')" "$all"
done
