#!/usr/bin/env bash
# MTP cost measurement at 4,000 lanes (main agent, blocking, one lock hold, medians of 3).
# Arms: base (flags off), draft (QWEN_MTP_DRAFT=1), verify (QWEN_MTP_VERIFY=1), both.
# No band/threshold touched. This is a TIMING window: run only with zero subagents active.
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1
MODEL=${MODEL:-models/Qwen3.5-0.8B}
IDS=${IDS:-tmp/glue_copy_fold/ab/ids4032.csv}
STEPS=${STEPS:-32}
REPS=${REPS:-3}
ARMS=${ARMS:-base draft accept}
EXTRA=${EXTRA:-}
OUT=tmp/mtp_speed
BIN=${BIN:-engine/build/decode_run}
mkdir -p "$OUT"
want_tk=$(awk -F',' '{print NF}' "$IDS")
[ "$want_tk" -gt 1000 ] || { echo "ABORT: id file holds $want_tk fields; a space-joined fixture reads as ONE token"; exit 1; }
echo "== ids=$IDS fields=$want_tk steps=$STEPS reps=$REPS bin=$BIN"

tools/gpu_lock.sh acquire main-agent mtp-speed || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util 2>&1 | tail -1)"

for arm in $ARMS; do
  case "$arm" in
    base)    envline="env" ;;
    draft)   envline="env QWEN_MTP_DRAFT=1" ;;
    accept)  envline="env QWEN_MTP_DRAFT=1 QWEN_MTP_VERIFY=1 QWEN_MTP_ACCEPT=1" ;;
  esac
  for r in $(seq 1 "$REPS"); do
    log="$OUT/${arm}_${r}.log"; od="$OUT/out_${arm}_${r}"; mkdir -p "$od"
    $envline "$BIN" --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --time-stages --out "$od" $EXTRA > "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then echo "ARM $arm rep $r: rc=$rc (see $log)"; tail -3 "$log" | sed 's/^/    /'; continue; fi
    tk=$(grep -E '^PREFILL' "$log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
    [ "$tk" = "$want_tk" ] || { echo "ARM $arm rep $r: CANARY-FAIL tokens=$tk want=$want_tk"; continue; }
    parity=$(grep -c 'STAGEPARITY .*tokens_identical=1 logits_bitidentical=1' "$log")
    burst=$(grep -E 'gran=burst' "$log" | head -1)
    stage=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1)
    mtpstat=$(grep '^MTPSTAT' "$log" | head -1)
    echo "ARM $arm rep $r: tokens=$tk parity=$parity"
    echo "    $burst"
    echo "    $stage"
    [ -n "$mtpstat" ] && echo "    $mtpstat" || echo "    (no MTPSTAT line: arm is off, as expected for base/draft)"
  done
done
echo "== util at release: $(tools/gpu_lock.sh util 2>&1 | tail -1)"
