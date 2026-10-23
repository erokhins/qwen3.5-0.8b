#!/usr/bin/env bash
# prompt_fan_paired.sh — paired (McNemar) read of fan widths against the fan-0 loop arm, in one window.
#
# WHY NOT JUST THE TABLE IN `prompt_fan_sweep.sh`. The sweep compares accept fractions from separate runs,
# so the bar it prints is the pair's binomial half-width (±0.10 at ~170 rounds) and a +0.04 move reads
# "within noise" no matter whether it is real. The arms are not independent samples: same fixture, same
# greedy stream (the admission requires it), same positions — only WHICH positions an accepting round landed
# on differs. `prompt_fan_paired.py` does the paired test on the settled positions, and its sensitivity is
# set by the discordant count.
#
# ONE WINDOW MATTERS: every arm here runs inside a single GPU-lock hold, back to back, on the same
# fixture — so no cross-window ruler is being compared against another (`AGENTS.md`: a ruler re-taken in a
# different window is not an A/B).
#
# NO TIMING: no millisecond figure is read or printed anywhere. The wall at the end is this harness's own
# seconds, and the runs carry `--gdn-loop 1 --no-stop` because they are rate runs, not shipped-path runs.
#
# Usage:
#   prototype/mtp_accept/prompt_fan_paired.sh                       # 4k fixture, 256 steps, fans 46 96 146
#   FIXTURE=tmp/llama_input/ids_long.csv STEPS=128 FANS="146 329" \
#       prototype/mtp_accept/prompt_fan_paired.sh
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

FIXTURE=${FIXTURE:-tmp/llama_input/ids_long.csv}
STEPS=${STEPS:-256}
FANS=${FANS:-"46 96 146"}
BIN=${BIN:-engine/build/decode_run}
MODEL=${MODEL:-$(ls -d models/Qwen*)}
OUT=${OUT:-/tmp/pf/paired}
N=$(awk -F',' '{print NF}' "$FIXTURE" 2>/dev/null || echo 0)
[ "$N" -ge 8 ] || { echo "$FIXTURE has $N comma fields — that is the one-token silent failure"; exit 1; }

rm -rf /tmp/pf/paired
mkdir -p "$OUT"
TAG=${TAG:-worker prompt-fan-paired}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"
echo "== fixture=$FIXTURE fields=$N steps=$STEPS fans='$FANS' — rate-only, NO TIMING taken"
s=$(date +%s)

BASE="--mtp-draft 1 --mtp-accept 1 --gdn-loop 1 --no-stop"
run_arm() {   # run_arm NAME [extra...]
    local name=$1; shift
    "$BIN" --model "$MODEL" --ids-file "$FIXTURE" --steps "$STEPS" $BASE "$@" \
        > "$OUT/$name.log" 2>&1 || { echo "RUN FAIL arm=$name"; tail -4 "$OUT/$name.log"; return 1; }
    grep -h "^MTPSTAT" "$OUT/$name.log" | sed "s/^/  [$name] /"
    grep -h "^MTPPROMPT" "$OUT/$name.log" | sed "s/^/  [$name] /"
}

run_arm loop || exit 1
for f in $FANS; do
    run_arm "fan$f" --mtp-prompt-lanes "$f" || exit 1
    echo
    .venv/bin/python3 prototype/mtp_accept/prompt_fan_paired.py \
        "$OUT/loop.log" "$OUT/fan$f.log" --label "fan$f"
    echo
done
echo "== harness wall $(( $(date +%s) - s ))s (its own seconds); NO TIMING taken"
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
