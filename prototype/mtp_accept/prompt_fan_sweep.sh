#!/usr/bin/env bash
# prompt_fan_sweep.sh — a(fan_width): how the accept rate moves as the draft cache's prompt seed widens.
#
# WHY A SEPARATE SWEEP AND NOT JUST MORE ARMS ON `draft_cond_probe.sh`. The knob it sweeps is bounded by
# three different ceilings at once, and two of them depend on `--steps`:
#
#   * the draft attention's one-pass form: `mtp_attn_max_lanes()` = 512 lanes, a hard wall — the block's
#     own kernel stages `MAXSC` scores in shared memory and REFUSES more, so "seed the whole prompt" is
#     unreachable at any prompt length and the widest possible fan is 512 minus what the run still needs;
#   * the seeding's own reserve: a run must leave room in the cache for its own rounds or it starts
#     printing `DRAFT-SKIP`, which changes the accept-rate population mid-run. The reserve is
#     ceil(2*steps/1.4), so a 256-step run leaves 146 lanes and a 128-step run leaves 329;
#   * the prompt length itself (a 4k prompt can supply all of either).
#
# So a fan ladder at ONE `--steps` cannot reach past ~146 lanes, and comparing fans ACROSS `--steps` values
# compares different denominators (`DRAFT-COND.md` §3: the same loop arm reads 0.4713 at 256 steps and
# 0.3953 at 120). Both are reported, and each panel is internally comparable: every arm inside a panel shares
# `--steps`, the fixture, the window, and the flag-off pair run.
#
# HOW TO READ THE OUTPUT. `a` is the accept fraction; `FAN` is printed per arm and says what the seed
# ACTUALLY staged, which is the number the curve is a function of, together with `clamped=` (which limit
# bound) and `last_wave_rows=` (the 46-row scratch clamp that used to be the binding one and is now only a
# floor). `lane_subs=` is the staging's block submissions for that arm, 9 per lane, once per request.
#
# NO TIMING: no millisecond figure is read or printed anywhere in this pass. The harness seconds it prints
# are its own wall, not an engine measurement, and the runs carry `--no-stop --gdn-loop 1` because they are
# rate runs, not shipped-path runs.
#
# Usage (one panel per invocation, so one GPU-lock hold per panel):
#   prototype/mtp_accept/prompt_fan_sweep.sh A            # the 4k reporting width, 256 steps
#   prototype/mtp_accept/prompt_fan_sweep.sh B            # the 4k reporting width, 128 steps (wider fans)
#   prototype/mtp_accept/prompt_fan_sweep.sh C            # mid-length fixture
#   prototype/mtp_accept/prompt_fan_sweep.sh ALL
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-.venv/bin/python3}
PROBE=prototype/mtp_accept/draft_cond_probe.py
LONG=${LONG:-tmp/llama_input/ids_long.csv}      # 4,142 ids — the width the headline is set at
MID=${MID:-/tmp/pf/mid.csv}                    # 1,285 ids — a mid-length prose fixture
OUTROOT=${OUTROOT:-/tmp/pf/fan}
MODEL=${MODEL:-$(ls -d models/Qwen*)}
for f in "$LONG" "$MID"; do
    n=$(awk -F',' '{print NF}' "$f" 2>/dev/null || echo 0)
    [ "$n" -ge 8 ] || { echo "fixture $f has $n comma fields — that is the one-token silent failure"; exit 1; }
    echo "== fixture $f fields=$n"
done

panel() {           # panel NAME FIXTURE STEPS "arm specs" "control specs..."
    local name=$1 fix=$2 steps=$3 arms=$4; shift 4
    local out=$OUTROOT/$name
    rm -rf "$out"; mkdir -p "$out"
    echo
    echo "######## PANEL $name  fixture=$fix steps=$steps ########"
    local aargs=()
    # Word-split on purpose: each whitespace-separated item is one `name=extra,args` arm spec.
    for a in $arms; do aargs+=(--arm "$a"); done
    local cargs=()
    for c in "$@"; do cargs+=(--control "$c"); done
    local s=$(date +%s)
    $PY "$PROBE" --fixture "$fix" --steps "$steps" --binary engine/build/decode_run \
        --model "$MODEL" --out "$out" --json "$out/$name.json" "${aargs[@]}" "${cargs[@]}" \
        > "$out/$name.log" 2>&1
    local rc=$?
    cat "$out/$name.log"
    echo "-------- panel $name rc=$rc harness wall $(( $(date +%s) - s ))s (its own seconds; NO TIMING taken)"
    # The logits dumps are the admission's raw material DURING the run and 250 MB of them per arm after
    # it; the log and the json are the record. `/` is 98 % full on this box.
    find "$out" -maxdepth 2 -name 'logits*.bin' -delete 2>/dev/null
    return $rc
}

CTRL_PLAIN="--mtp-draft,1,--mtp-sched,every,--gdn-loop,1,--no-stop"

P=${1:-ALL}
rc=0
tools/gpu_lock.sh acquire "worker prompt-fan-sweep" || exit 1
trap 'tools/gpu_lock.sh release "worker prompt-fan-sweep" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as 'worker prompt-fan-sweep'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"
echo "== rate-only pass: accept fractions and counts, no millisecond figure anywhere ($(date -u +%FT%TZ))"

case "$P" in
A)  # 4,142-lane width, the step budget the rate table is carried at. Reserve=366 of the 512-lane cache,
    # so the ladder stops at 146 — the ceiling is the CACHE, not the prompt, and `clamped=` says so.
    panel A "$LONG" 256 \
        "loop= dense=--mtp-dense,1 pl46=--mtp-prompt-lanes,46 pl96=--mtp-prompt-lanes,96 pl146=--mtp-prompt-lanes,146" \
        "side=$CTRL_PLAIN" "sidepl146=$CTRL_PLAIN,--mtp-prompt-lanes,146" || rc=1 ;;
B)  # same width, half the step budget: reserve falls to 183, so the fan can reach 329. The loop arm is
    # re-measured here because a cross-panel fan comparison is meaningless without it.
    panel B "$LONG" 128 \
        "loop= pl146=--mtp-prompt-lanes,146 pl329=--mtp-prompt-lanes,329" || rc=1 ;;
C)  # mid-length prose fixture, same 128-step budget: does the fan do more work when the prompt is short
    # enough that a 329-lane seed is a quarter of the context rather than a twelfth?
    panel C "$MID" 128 \
        "loop= pl46=--mtp-prompt-lanes,46 pl146=--mtp-prompt-lanes,146 pl329=--mtp-prompt-lanes,329" || rc=1 ;;
ALL) panel A "$LONG" 256 \
        "loop= dense=--mtp-dense,1 pl46=--mtp-prompt-lanes,46 pl96=--mtp-prompt-lanes,96 pl146=--mtp-prompt-lanes,146" \
        "side=$CTRL_PLAIN" "sidepl146=$CTRL_PLAIN,--mtp-prompt-lanes,146" || rc=1
     panel B "$LONG" 128 "loop= pl146=--mtp-prompt-lanes,146 pl329=--mtp-prompt-lanes,329" || rc=1
     panel C "$MID" 128 "loop= pl46=--mtp-prompt-lanes,46 pl146=--mtp-prompt-lanes,146 pl329=--mtp-prompt-lanes,329" || rc=1 ;;
*) echo "usage: $0 [A|B|C|ALL]"; exit 2 ;;
esac

echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
