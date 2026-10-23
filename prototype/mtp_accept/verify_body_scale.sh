#!/usr/bin/env bash
# verify_body_scale.sh — how does a batched 2-row body scale with lanes, and is the round's cost the body?
#
# WHY. `prototype/mtp_accept/M3-RESULT.md` §5 ends with one open number: at 4,000 lanes an accept round
# costs 2.029 steps while the known extras (mid-checkpoint capture+restore ~19.3 MiB, verdict kernel + 16 B
# readback, one draft ~0.44 ms) are sub-millisecond in total, so the ~11.6 ms is the BODY — and the same
# body amortises ~16 % at short context (1.691 round/step) but nothing at 4k. llama.cpp's `draft-mtp` on
# this same box and model prices its own k=1 round at 1.579 steps at 4,152 lanes, of which its trace log
# says 3.42 ms is DRAFTING and the rest (≈1.26 steps) is the 2-row verification (`prototype/mtp_llama/`).
# So the shape is achievable here; what is unknown is which term of OUR body grows with context.
#
# INSTRUMENT. `--mtp-verify 1` runs the batched 2-row body once per generation step as a READ-ONLY side
# pass and consumes nothing, so the difference between it and a plain step at identical flags IS one body
# (plus an argmax report). It refuses `--time-stages` (line 1157 of decode_run.cpp: a side pass has no
# per-step stage bill), which is why this script prices it from DECWALL's decode window rather than from
# stage clocks. Three arms per width, one variable each:
#   base    : one-row step                                     -> ms/step
#   verify  : one-row step + one 2-row body per step           -> minus base = one body
#   accept  : the real round (body + checkpoint + verdict + draft) -> minus base = round overhead
# and the whole point is the SHAPE of each ratio against lanes: flat = weight-walk-bound (fixable by
# sharing the walk), rising with lanes = per-row traffic (needs the KV read to be shared).
#
# CANARIES, per rep, before the milliseconds are used: exactly one DECWALL; census=ok; PREFILL tokens= equal
# to the field count read off the ids FILE (--ids-file splits on commas — AGENTS.md); VERIFY2 present on the
# verify arm and absent on base; MTPSTAT rounds>0 on accept; and equal prompt token counts across the three
# arms of a width.
#
# Protocol: main agent, blocking, one lock hold, util at both ends and discounted, medians of REPS, no
# best-of-N. `--gdn-loop 1` on every arm (the accept loop pins it, so the pair must not differ in it).
#
# Usage: prototype/mtp_accept/verify_body_scale.sh [width...]   # default: 128 1024 4032
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

BIN=${BIN:-engine/build/decode_run}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
OUT=${OUT:-tmp/verify_body_scale}
REPS=${REPS:-2}
STEPS=${STEPS:-32}
SRC_IDS=${SRC_IDS:-/tmp/dec8_ids.csv}
mkdir -p "$OUT"; rm -f "$OUT"/*.tsv "$OUT"/*.log 2>/dev/null

ids_of() { # $1 width -> a comma-separated ids file of EXACTLY that many fields, cycled from a real fixture
  local w="$1" f="$OUT/ids$1.csv" n
  awk -v n="$w" 'BEGIN{RS="[,\n]"} {ids[c++]=$0} END{if(c<1){exit 1} for(i=0;i<n;i++) printf "%s%s", ids[i%c], (i+1<n?",":"")}' \
      "$SRC_IDS" > "$f"
  n=$(awk -F',' '{print NF}' "$f")
  [ "$n" = "$w" ] || { echo "ABORT: $f has $n fields, wanted $w (a short fixture silently feeds a smaller prefill)"; exit 1; }
  echo "$f"
}

run() { # $1 arm $2 width $3 ids-file $4 rep -> log
  local arm="$1" w="$2" ids="$3" rep="$4" extra="" log
  case "$arm" in
    base)   extra="" ;;
    verify) extra="--mtp-verify 1" ;;
    accept) extra="--mtp-draft 1 --mtp-accept 1" ;;
  esac
  log="$OUT/${w}_${arm}_${rep}.log"; mkdir -p "$OUT/out_${w}_${arm}_${rep}"
  "$BIN" --model "$MODEL" --ids-file "$ids" --steps "$STEPS" --cap "$((w + 300))" \
        --out "$OUT/out_${w}_${arm}_${rep}" --no-stop --gdn-loop 1 $extra > "$log" 2>&1
  echo "$log"
}

kv() { grep -m1 "^$1 " "$2" | tr ' \t' '\n\n' | sed -n "s|^$3=||p"; }

if [ "$#" -gt 0 ]; then WIDTHS="$*"; else WIDTHS="128 1024 4032"; fi
tools/gpu_lock.sh acquire main-agent verify-body-scale || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1); steps=$STEPS reps=$REPS src=$SRC_IDS"

for w in $WIDTHS; do
  ids=$(ids_of "$w")
  echo
  echo "=== lanes=$w (ids file $ids, $(awk -F',' '{print NF}' "$ids") fields, cap $((w + 300))) ==="
  for rep in $(seq 1 "$REPS"); do
    for arm in base verify accept; do
      log=$(run "$arm" "$w" "$ids" "$rep")
      [ "$(grep -c '^DECWALL' "$log")" = 1 ] || { echo "ABORT: $w/$arm/rep$rep printed no single DECWALL"; exit 1; }
      [ "$(kv DECWALL "$log" census)" = "ok" ] || { echo "ABORT: $w/$arm/rep$rep census=$(kv DECWALL "$log" census)"; exit 1; }
      tk=$(kv PREFILL "$log" tokens); [ "$tk" = "$w" ] || { echo "ABORT: $w/$arm/rep$rep prefilled $tk tokens, file has $w"; exit 1; }
      case "$arm" in
        verify) grep -q '^VERIFY2' "$log" || { echo "ABORT: $w/$arm printed no VERIFY2 line: the side pass never ran, so this arm is a duplicate of base, not a slower one"; exit 1; } ;;
        base)   grep -q '^VERIFY2' "$log" && { echo "ABORT: the base arm printed VERIFY2 — the flag leaked"; exit 1; } ;;
        accept) [ "$(kv MTPSTAT "$log" rounds)" -gt 0 ] 2>/dev/null || { echo "ABORT: $w/$arm rounds not > 0"; exit 1; } ;;
      esac
      dms=$(kv DECWALL "$log" decode_ms); em=$(kv DECWALL "$log" emitted); rd=$(kv DECWALL "$log" rounds)
      [ -n "$dms" ] || { echo "ABORT: $w/$arm/rep$rep decode_ms empty"; exit 1; }
      bodies=$([ "$arm" = accept ] && echo "$rd" || echo "$STEPS")
      printf '%-6s rep %s decode %8.2f ms / %s %s  emitted %3s  %8.3f ms per %s\n' \
        "$arm" "$rep" "$dms" "$bodies" "$([ "$arm" = accept ] && echo rounds || echo steps)" "$em" \
        "$(awk -v d="$dms" -v b="$bodies" 'BEGIN{printf "%.4f", d/b}')" "$([ "$arm" = accept ] && echo round || echo step)"
      echo "$dms $bodies" >> "$OUT/${w}_${arm}_${rep}.tsv"
    done
  done
  med() { cat "$OUT"/${w}_$1_*.tsv | awk -v c="$2" '{print $c}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
  cost() { awk -v d="$(med "$1" 1)" -v b="$(med "$1" 2)" 'BEGIN{printf "%.6f", d/b}'; }
  step=$(cost base); body=$(cost verify); round=$(cost accept)
  awk -v w="$w" -v step="$step" -v body="$body" -v round="$round" 'BEGIN{
    printf "  -- medians, lanes=%d --\n", w;
    printf "   base    %7.3f ms/step\n", step;
    printf "   verify  %7.3f ms/step  =>  one 2-row BODY = %6.3f ms = %.3f steps\n", body, body-step, (body-step)/step;
    printf "   accept  %7.3f ms/round =>  round OVERHEAD beyond the body = %6.3f ms\n", round, round-body;
    printf "   round/step = %.3f   (body alone = %.3f of it)\n", round/step, (body-step)/step; }'
done
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
