#!/usr/bin/env bash
# run_locked.sh — the MTP draft-block timing harness. ONE `gpu_lock.sh` hold, blocking (never
# backgrounded), admission-gated, canary-asserted, two interleaved passes.
#
# WHO RUNS IT: the main agent. A worker may run `gate.py` (correctness, no timings) but no number from
# this script may come from a subagent — `engine/performance-status.md` is main-agent-only territory.
#
# ARMS
#   block-k1|k2|k4   the whole draft block, k chained draft steps, burst-timed (--time-burst)
#   empty            14 no-op launches per rep, the launch-overhead term measured in the SAME window
#                    as the block instead of carried in from another day (the "ruler re-taken in a
#                    different window is not an A/B" rule).
#
# WHAT EACH RULE IS A DEFENCE AGAINST
#   * admission-gated: the CPU selftest and the numeric+structural gate must be green before a single ms
#     is printed, and a red admission prints `NO-TIMING: admission is red` and exits (`prototype/attn_m2
#     /burst_m16.sh`; a millisecond figure quoted over a red admission is the failure, not the number).
#   * each arm states its OWN admission level. dec8's pinned golden is two steps, so k=4 has no numeric
#     band to be admitted against and is admitted by the byte-level set only (poison/guard/launch census/
#     cache-tail). A wider case list than the gate's is exactly how the M=16 null got reported.
#   * behavioural canary asserted before any timing is believed: the RESULT line must carry
#     `poison_left=0 guard_left=12288` and a launch count equal to the plan. A stale or half-built binary
#     agrees with itself, so self-consistency cannot catch it and a canary must (AGENTS, 2026-10-17).
#   * two interleaved passes, both quoted, min reported — a single pass on this box is dispatch-bimodal.
#   * `--time-burst` and `--dump` are mutually exclusive in the rig, so no timing run ever writes dumps.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/../.."

BIN=prototype/mtp_draft/mtp_draft
WORK=${WORK:-/tmp/mtp_draft}
FIXTURE=${FIXTURE:-dec8}
REPS=${REPS:-20}
PY=${PY:-.venv/bin/python3}
TAG="mtp_draft run_locked ${FIXTURE} $(date -u +%FT%TZ)"

ARM_NAMES=(block-k1 block-k2 block-k4 empty)
arm_flags() {
    case "$1" in
        block-k1) echo "--mode chain --k 1" ;;
        block-k2) echo "--mode chain --k 2" ;;
        block-k4) echo "--mode chain --k 4" ;;
        empty)    echo "--mode launch" ;;
    esac
}
arm_adm() { case "$1" in block-k4|empty) echo structural ;; *) echo numeric ;; esac; }
arm_k() { case "$1" in block-k1) echo 1 ;; block-k2) echo 2 ;; block-k4) echo 4 ;; *) echo 0 ;; esac; }

declare -A BEST MIN_SEEN
UTIL_START=$(tools/gpu_lock.sh util || echo unknown)
tools/gpu_lock.sh acquire "$TAG"
trap 'tools/gpu_lock.sh release "$TAG"' EXIT
echo "LOCK held tag=$TAG util_at_acquire=$UTIL_START"

# Build a per-arm work directory at the arm's own draft depth. The fixture's inputs.txt only carries the
# steps the pinned golden covers (dec8: two), so a deeper arm needs its step list continued — the last
# pinned token repeated and the position advanced. That is legitimate for a STRUCTURAL admission (bytes,
# guards, launch census) and is precisely why k=4 may not be quoted as a numeric headline.
prep_work() {
    local k="$1" d="$WORK/k$1"
    mkdir -p "$d/out"
    for f in block.txt block.bin h_post.bin h_pre.bin W.bin; do
        if [ ! -s "$d/$f" ]; then
            ln "$WORK/$f" "$d/$f" 2>/dev/null || cp "$WORK/$f" "$d/$f"
        fi
    done
    awk -v k="$k" 'NR==1{print; next} {n++; if (n<=k) print; last=$0}
                  END{ for (i=n+1; i<=k; ++i) { split(last, a, " "); print a[1], a[2] + (i-n) } }' \
        "$WORK/inputs.txt" > "$d/inputs.txt"
    local got; got=$(( $(wc -l < "$d/inputs.txt") - 1 ))
    [ "$got" = "$k" ] || { echo "prep_work: wrote $got steps for k=$k"; return 1; }
    echo "$d"
}

# ── admission: nothing below this point may print a millisecond unless all of it is green ──────────
if ! "$PY" prototype/mtp_draft/gate.py --selftest; then
    echo "NO-TIMING: admission is red (cpu selftest)"; exit 1
fi
if ! "$PY" prototype/mtp_draft/gate.py --fixture "$FIXTURE" --k 2 --negatives; then
    echo "NO-TIMING: admission is red (chain k=2 + derived negatives)"; exit 1
fi
# The structural admission for the arms the numeric set cannot reach (k=4: beyond the pinned golden).
for a in block-k4 empty; do
    adm_work=$(prep_work 4)
    if ! "$BIN" --work "$adm_work" $(arm_flags "$a") >/tmp/mtp_draft.adm.$$ 2>&1; then
        echo "NO-TIMING: admission is red ($a structural)"; cat /tmp/mtp_draft.adm.$$; exit 1
    fi
    case "$a" in
        # The block arm's structural canary is its byte invariants. The launch arm has no arena, so its
        # canary is the launch census itself — and note `timed=0`: the admission run must not be able to
        # produce a duration at all, or the admission pass would itself be an unattributed measurement.
        block-k4) grep -q 'poison_left=0 guard_left=12288' /tmp/mtp_draft.adm.$$ || {
                      echo "NO-TIMING: $a structural canary failed"; cat /tmp/mtp_draft.adm.$$; exit 1; } ;;
        empty)    grep -q 'mode=launch launches=14 timed=0' /tmp/mtp_draft.adm.$$ || {
                      echo "NO-TIMING: $a structural canary failed"; cat /tmp/mtp_draft.adm.$$; exit 1; } ;;
    esac
done
rm -f /tmp/mtp_draft.adm.$$
echo "ADMISSION green: numeric(dec8 k<=2, 21 observables + 9 derived negatives) "
echo "                 structural(k=4: poison/guard/launch-census/cache-tail; empty: n/a)"

# ── the timed passes ──────────────────────────────────────────────────────────────────────────────
run_arm() {
    local name="$1" flags work; flags=$(arm_flags "$name")
    work=$(prep_work "$(arm_k "$name")") || return 1
    local out rc
    out=$("$BIN" --work "$work" $flags --time-burst --repeat "$REPS" 2>&1) || rc=$?
    if [ "${rc:-0}" != 0 ]; then echo "ARM $name FAILED (exit ${rc})"; return 1; fi
    if [ "$name" = empty ]; then
        local us; us=$(sed -n 's/.*burst_us_per_launch=\([0-9.]*\).*/\1/p' <<<"$out")
        # A zero here is a broken timer, not a fast launch (`hipEventElapsedTime` reads 0 for sub-ms
        # work; the ledger says a zero means the measurement did not happen).
        if [ -z "$us" ] || [ "$us" = "0.0000" ]; then echo "ARM $name timer read zero"; return 1; fi
        echo "$us"
        return 0
    fi
    # The canary, in the same breath as the measurement: plan, poison, guards.
    grep -q 'poison_left=0 guard_left=12288' <<<"$out" || { echo "ARM $name canary: guards"; return 1; }
    local want got
    want=$((14 * $(arm_k "$name") * REPS))
    got=$(sed -n 's/.*RESULT mode=chain arm=block burst_ms=[0-9.]* reps=[0-9]* k=[0-9]* launches=\([0-9]*\).*/\1/p' <<<"$out")
    [ "$got" = "$want" ] || { echo "ARM $name canary: launches=$got want=$want"; return 1; }
    sed -n 's/.*burst_ms=\([0-9.]*\).*/\1/p' <<<"$out"
}

EV="${EV:-/tmp/mtp_draft/evidence}"
for pass in 1 2; do
    mkdir -p "$EV"; if [ "$pass" = 1 ]; then : > "$EV/log.txt"; fi
    for a in "${ARM_NAMES[@]}"; do
        v=$(run_arm "$a") || { echo "NO-TIMING: arm $a failed its canary in pass $pass"; exit 1; }
        echo "  pass$pass $(printf '%-8s' "$a") admission=$(arm_adm "$a") $v" | tee -a "$EV/log.txt"
        MIN_SEEN[$a]="${MIN_SEEN[$a]:-} $v"
        if [ -z "${BEST[$a]:-}" ]; then BEST[$a]="$v"; else
            BEST[$a]=$(awk -v x="${BEST[$a]}" -v y="$v" 'BEGIN{printf "%.4f", (y+0<x+0)?y:x}')
        fi
    done
done

echo "RESULT summary (reps=$REPS fixture=$FIXTURE; passes:$(echo "${MIN_SEEN[*]}")"
for a in "${ARM_NAMES[@]}"; do
    echo "  $a  min=${BEST[$a]}  passes=(${MIN_SEEN[$a]})  admission=$(arm_adm $a)"
done
echo "DRAFT-PRICE.md's model is filled from these lines; nothing here enters the ledger by itself."
tools/gpu_lock.sh release "$TAG"
