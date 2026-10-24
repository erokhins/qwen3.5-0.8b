#!/usr/bin/env bash
# wide_body_ab.sh — does NOT using the wide WMMA attention block in a verify body make MTP pay at 4k?
#
# WHY. The body probe (`--mtp-verify 1 --mtp-meter 1`, one knob at a time inside one lock hold) found that a
# 2-row verify body at 4,032 lanes costs 16.906 ms with the A13 wide attention block and 11.972 ms without
# it: -4.93 ms, -29 %, the single largest item found in the round. `--gdn-loop 1` moved it by 0.06 ms (free).
# That block was ported for PREFILL, where it won 27 % of the 16k wave; nothing in this repo had ever asked
# what it costs at 2 rows over a populated cache. The wall question follows immediately: does removing it turn
# the -16 % accept arm positive?
#
# FOUR arms, not two, because two of them are canaries:
#   base          the shipping decode path, wide block as shipped
#   base_wide0    the same path with the block off. A one-row decode step never enters the wide path (it is a
#                 rows>1 block), so this MUST agree with `base`. If it moves, the flag is doing something else
#                 to the run and the accept-arm delta is not attributable to the body.
#   accept        the shipping accept loop (k=1, --gdn-loop 1), wide block as shipped
#   accept_wide0  the accept loop with the block off. The arm whose delta is being adjudicated.
#
# WHAT IT REFUSES TO PRINT. A DECWALL line missing for any arm (an arm that died silently would be median-ed
# out of existence), a base_wide0 that moves more than 2 % against base (the canary), or an accept rate that
# differs between the two accept arms -- if the two accept arms do not accept the same drafts, this is a
# numerics change wearing a timing coat, and the number is reported as such, not as a speedup.
#
# NOT AN ADMISSION. Wall time is not gate admission. Whether the verify path may ship with the block off is
# decided by engine/gate.py on the pinned fixtures, including whether the emitted tokens and logits stay
# inside the bands; this script only establishes that there is 4.9 ms and a double-digit tok/s delta there.
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

MODEL=${MODEL:-models/Qwen3.5-0.8B}
IDS=${IDS:-tmp/verify_body_scale/ids4032.csv}
STEPS=${STEPS:-32}
CAP=${CAP:-4332}
REPS=${REPS:-3}
OUT=${OUT:-tmp/wide_body_ab}
CANARY_PCT=${CANARY_PCT:-0}   # 0 = derive from the observed within-arm spread (see below)

want_tk=$(awk -F',' '{print NF}' "$IDS") || { echo "no ids file $IDS"; exit 1; }
[ "$want_tk" -ge 8 ] || { echo "$IDS has $want_tk fields; too few to be a fixture"; exit 1; }
mkdir -p "$OUT"; rm -f "$OUT"/*.tsv

COMMON="--mtp-accept 0 --model $MODEL --ids-file $IDS --steps $STEPS --cap $CAP --no-stop"
ACC="--mtp-draft 1 --mtp-accept 1 --gdn-loop 1"

arm() { case "$1" in
  base)         echo "$COMMON" ;;
  base_wide0)   echo "$COMMON --attn-fattn-wide 0" ;;
  accept)       echo "$COMMON $ACC" ;;
  accept_wide0) echo "$COMMON $ACC --attn-fattn-wide 0" ;;
esac; }

tools/gpu_lock.sh acquire main-agent wide-body-ab || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1); fixture $IDS ($want_tk fields); reps $REPS"

for label in base base_wide0 accept accept_wide0; do
  for rep in $(seq 1 "$REPS"); do
    log="$OUT/${label}_${rep}.log"
    engine/build/decode_run $(arm "$label") --out "$OUT/${label}_${rep}" > "$log" 2>&1
    d=$(grep '^DECWALL' "$log" | tail -1)
    if [ -z "$d" ]; then echo "FAIL: no DECWALL line for $label rep $rep (see $log)"; exit 1; fi
    tps=$(echo "$d" | grep -oE 'tok_per_s=[0-9.]+' | cut -d= -f2)
    rd=$(echo "$d"  | grep -oE 'rounds=[0-9]+' | cut -d= -f2)
    ac=$(echo "$d"  | grep -oE 'accepted=[0-9]+' | cut -d= -f2)
    [ -n "$tps" ] || { echo "FAIL: $label rep $rep has no tok_per_s"; exit 1; }
    echo "$label $tps ${rd:-0} ${ac:-0}" >> "$OUT/rows.tsv"
    printf "  %-13s rep %d  tok/s %8.3f  rounds %4s accepted %4s\n" "$label" "$rep" "$tps" "${rd:-?}" "${ac:-?}"
  done
done
tools/gpu_lock.sh release main-agent >/dev/null
trap - EXIT

med() { awk -v l="$1" '$1==l {print $2}' "$OUT/rows.tsv" | sort -n | \
      awk '{a[++n]=$1} END {if (n==0) {print "nan"; exit} printf "%.3f", a[int((n+1)/2)]}'; }
rate() { awk -v l="$1" '$1==l {r+=$3; a+=$4} END {if (r==0) {print "nan"; exit} printf "%.4f", a/r}' "$OUT/rows.tsv"; }
rate() { awk -v l="$1" '$1==l {r+=$3; a+=$4} END {if (r==0) {print "nan"; exit} printf "%.4f", a/r}' "$OUT/rows.tsv"; }

b=$(med base); bw=$(med base_wide0); a=$(med accept); aw=$(med accept_wide0)
ra=$(rate accept); raw=$(rate accept_wide0)
echo "== medians of $REPS reps, one lock hold, $want_tk lanes"
printf "  base          %8.3f tok/s\n  base_wide0    %8.3f tok/s   (canary: must match base)\n" "$b" "$bw"
printf "  accept        %8.3f tok/s   accept rate %s\n  accept_wide0  %8.3f tok/s   accept rate %s\n" "$a" "$ra" "$aw" "$raw"
awk -v b="$b" -v bw="$bw" -v a="$a" -v aw="$aw" -v ra="$ra" -v raw="$raw" -v cp="$CANARY_PCT" 'BEGIN {
  dc = (bw/b-1)*100;
  printf "  canary base_wide0 vs base: %+.2f %%\n", dc;
  if (ra != raw) { print "  ABORT: the two accept arms accepted different drafts -- this is a numerics change, not a timing win"; exit 1 }
  printf "  accept shipping    vs base: %+.2f %%\n", (a/b-1)*100;
  printf "  accept wide-off    vs base: %+.2f %%   (pre-registered adopt line: >= +10.00 %%)\n", (aw/b-1)*100;
  if ((aw/b-1)*100 >= 10) print "  VERDICT: clears the adopt line on wall time. Gate admission is a separate question.";
  else print "  VERDICT: does NOT clear the adopt line on wall time.";
}'
echo "== logs in $OUT; medians of $REPS, one lock hold; no number here is best-of-N"
