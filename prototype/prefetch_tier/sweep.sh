#!/usr/bin/env bash
# sweep.sh — THE TIMED RUN for route P1. Main-agent only.
#
# Every number here is a timing, it must hold the iGPU exclusively, and a contaminated number is worse
# than no number (AGENTS.md: "GPU execution is serialized and belongs to the main agent"). Run it
# BLOCKING, in the foreground, with ZERO subagents active. Never `&`, never nohup, never backgrounded:
# this box serves this session's own tokens on the same GPU, so a backgrounded run contends with itself.
#
# =========================================================================================
# PRE-REGISTERED BEFORE ANY TIMING WAS TAKEN (queue P1 + P1-ARITHMETIC.md; do not re-cut after seeing
# the numbers — if a line cannot be adjudicated, report that instead of moving the line)
#
#   Prize bar            0.30 ms on a 10.9357 ms step [quoted, queue P1 preamble]
#   Honest envelope      0.2-0.6 ms as the queue states it; the CPU transfer model in
#                        prototype/prefetch_tier/P1-ARITHMETIC.md puts the NEW money at 0.161 ms and
#                        says the only way past it is exposed latency in a latency-bound family
#   Sub-targets          GDN state ~0.07 ms [quoted, D4 S7: cold penalty saturating at 32 MiB];
#                        per-layer KV unknown, bounded above by attn's non-floor time 1.4031 ms
#
#   WIN       (cold - polluted) >= 0.30 ms/step at an engine-shaped stratum, adjudicable against that
#             stratum's OWN relative noise floor. The engine reading, not the best-case reading.
#   KILL-1    |cold - warm| <= noise at EVERY stratum. An explicit prefetch changes nothing at the use
#             site: the mechanism is not present on this part. Prefetch does not belong in this engine.
#   KILL-2    |cold - polluted| <= noise at every stratum while some stratum shows
#             cold - warm > noise. The bytes ARE fetched and are GONE by the use site — the exact
#             failure P1 was built to dodge, and the answer to the user's question "does the early
#             prefetch win us something": no, because the phase that follows evicts it.
#   INCONCLUSIVE  everything else, reported with the stratum that failed to adjudicate and what a
#             re-take would need. A null is a result here; three routes in this repo were killed
#             cheaply and that was the good outcome (P3, P4, split-K, glue fold).
#
#   WHICH READINGS THE ARMS ARE                    Only the USE-SITE READ is timed; the evictor, the
#   prefetch and the intervening stream run before the timer and are synced outside it. A prefetch is an
#   extra read and the consumer's read still happens, so timing the whole sequence would bill the warm arm
#   for an extra trip to memory and make cold - warm negative whatever the cache does. Timed this way:
#     cold - warm     ceiling — what a staged slice is worth with nothing following the prefetch.
#     cold - polluted engine reading — does the slice survive the traffic that really sits in between.
#   The timed quantity is therefore 0.03-0.2 ms, a sub-ms figure on a box whose event timer returns exactly
#   0 below ~1 ms, which is why every row carries samples/reps/iters/spread_rel and fit_prefetch.py
#   refuses a row whose own min/max spread is above 30 % (see prototype/gdn_insitu for the same shape).
#
# WHICH MEASUREMENT DECIDES WHICH OUTCOME
#   dose curve on --setup-mib 32/64/256 : is the cold arm actually cold? If 32 MiB and 256 MiB cost the
#                                         same, the setup never evicted the tier and every delta below
#                                         is unreadable (D4 measured saturation at exactly 32 MiB; this
#                                         rig must reproduce that shape or explain it).
#   --intervene-mib 7/21/43/64          : the residency cliff. Where cold-polluted collapses toward
#                                         cold-warm's mirror is the amount of traffic the tier survives,
#                                         which is the number the engine's staging window has to fit in.
#   --regime idle vs spin               : whether "issue the prefetch while a phase computes" differs
#                                         from "issue it while the GPU is idle" (the host_tail case). If
#                                         they differ, the prize is window-shaped, not residency-shaped,
#                                         and it must be re-filed as an overlap route.
# =========================================================================================
set -u
export LC_ALL=C
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
cd "$ROOT" || exit 2
BIN="$HERE/p1_rig"
REPS=${REPS:-5}
ITERS=${ITERS:-3}
OUT=${OUT:-$HERE/out}
mkdir -p "$OUT"
RES="$OUT/results.txt"
LOG="$OUT/sweep.log"
: > "$RES"; : > "$LOG"
ROWS=0

die() { echo "ABORT: $*" | tee -a "$LOG"; }

# Sizes are DERIVED, never typed: one slice of KV is 7.875 MiB only because six attention layers at 4,032
# lanes say so, and the same number spelled in a shell script is a number that rots (the AGENTS rank-3
# tensor row is the same failure one level over). Every MiB below comes from size_targets.py, which reads
# engine/state.h and runs tools/step_traffic_model.py.
PY=${PY:-$ROOT/.venv/bin/python}
[ -x "$PY" ] || PY=python3
size_mib() {
    local v
    v=$("$PY" "$HERE/size_targets.py" --bytes-mib "$1" 2>&1)
    case "$v" in
      *[!0-9.]*|"") echo "ABORT: size_targets.py --bytes-mib $1 did not print a number: $v" | tee -a "$LOG"
             return 2 ;;
    esac
    printf '%s' "$v"
}
STATE_MIB=$(size_mib gdn_state_all18)   || exit 2   # the whole per-step GDN state working set
STATE_SLICE_MIB=$(size_mib gdn_state_1layer) || exit 2
KV_SLICE_MIB=$(size_mib kv_1layer_at_4k) || exit 2
WEIGHTS_MIB=$(size_mib weights_1gdn_layer) || exit 2   # one layer's block: does NOT fit the tier
TIER_BYTES=33554432   # the ~32 MiB fast tier (devices/strix-halo-memory-bandwidth.md: 876.7 GB/s holding to 32 MiB, cliff to 239.4 by 64 MiB). Anything larger cannot be staged resident, so it is not a residency candidate.
echo "derived target sizes (MiB): state=$STATE_MIB state-slice=$STATE_SLICE_MIB kv-slice=$KV_SLICE_MIB weights=$WEIGHTS_MIB" | tee -a "$LOG"

[ -x "$BIN" ] || { echo "ABORT: $BIN not built; run $HERE/build.sh"; exit 2; }
# The lock protocol, checked rather than trusted: a self-locking child parked on a held lock looks
# exactly like a hung box (AGENTS: three gate drivers self-lock, and a worker deadlocked one).
if [ -f "$BIN" ] && grep -q gpu_lock "$BIN"; then
    echo "ABORT: $BIN self-locks; holding the lock here would deadlock"; exit 2
fi
TOOLS=tools/gpu_lock.sh
[ -x "$TOOLS" ] || { echo "ABORT: $TOOLS not found"; exit 2; }

TAG="p1 prefetch_tier timed matrix (main agent, exclusive)"
echo "util at acquire: $("$TOOLS" util)" | tee -a "$LOG"
"$TOOLS" acquire "$TAG" || { echo "ABORT: could not acquire the GPU lock"; exit 2; }
trap '"$TOOLS" release "$TAG" >/dev/null 2>&1 || true' EXIT INT TERM

# ---- admission: no timing is taken for a red rig ------------------------------------------------
if ! "$BIN" --selftest-geom 2>&1 | tee -a "$LOG" | grep -q "^SELFTEST-GEOM ok=1"; then
    die "the geometry selftest is not ok. No timing is taken."
    exit 1
fi
if ! "$BIN" --selftest 2>&1 | tee -a "$LOG" | grep -q "^SELFTEST ok "; then
    die "--selftest is red: the byte witnesses do not hold. No timing is taken (an admission-red run " \
        "must not print a millisecond figure: prototype/attn_m2 burst_m16.sh is the precedent)."
    exit 1
fi

# ---- behavioural canary, read in the same breath as the measurement ------------------------------
# engine/build.sh is not bit-reproducible on this box, so comparing binaries proves nothing; a stale
# binary would time the wrong arms and every self-consistency check inside it would agree with itself
# (AGENTS 2026-10-17 row). The canary here is the one field the arms are REQUIRED to differ in: the
# cold arm issues no prefetch (pre_rd=-1), the warm arm does (pre_rd=1), the polluted arm runs the
# intervening stream (inter_rd=1). If a build ever made the three arms the same code path, every row
# would read a clean null and the canary is what catches it.
canary() {  # $1=arm $2=expected field
    local line
    line=$("$BIN" --target state --target-mib "$STATE_MIB" --arm "$1" --verify 2>&1)
    case "$line" in
      *"$2"*) return 0 ;;
      *) echo "CANARY-FAIL arm=$1 expected '$2' in:"; printf '%s\n' "$line" | sed 's/^/  /'
         return 1 ;;
    esac
}
canary cold "pre_rd=-1"     || { die "canary: the cold arm is not prefetch-free"; exit 1; }
canary warm "pre_rd=1"      || { die "canary: the warm arm issues no prefetch"; exit 1; }
canary polluted "inter_rd=1" || { die "canary: the polluted arm runs no intervening stream"; exit 1; }
echo "CANARY ok: cold pre_rd=-1, warm pre_rd=1, polluted inter_rd=1" | tee -a "$LOG"

# ---- the matrix ---------------------------------------------------------------------------------
# run <target> <target_mib> <arm> <regime> <setup_mib> <intervene_mib> <pass>
run() {
    local tgt=$1 tmib=$2 arm=$3 regime=$4 setup=$5 inter=$6 pass=$7 line res
    line=$("$BIN" --target "$tgt" --target-mib "$tmib" --arm "$arm" --regime "$regime" \
                --setup-mib "$setup" --intervene-mib "$inter" --time --reps "$REPS" \
                --iters "$ITERS" 2>&1)
    res=$(printf '%s\n' "$line" | grep '^RESULT' | head -1)
    if [ -z "$res" ]; then
        printf '%s\n' "$line" | sed 's/^/RAW /' >> "$RES"
        die "no RESULT for target=$tgt arm=$arm regime=$regime setup=$setup inter=$inter"
        return 1
    fi
    # the witness line must still say green in the timed path, or this row is a broken-rig number
    if printf '%s\n' "$line" | grep '^WITNESS' | grep -qE 'use_rd=0|setup_rd=0|pre_rd=0|inter_rd=0|guard=0|sentinel=0'; then
        die "witness went red mid-run for $tgt/$arm/$regime; NO-TIMING is the only honest label"
        printf 'NO-TIMING target=%s arm=%s regime=%s reason=witness_red\n' "$tgt" "$arm" "$regime" >> "$RES"
        return 1
    fi
    printf 'CASE %s\n%s pass=%s\n' "$tgt" "$res" "$pass" >> "$RES"
    ROWS=$((ROWS + 1))
    return 0
}

BAD=0
for PASS in 1 2; do
    # A: the engine-shaped matrix. Two staged forms that fit the tier — the whole GDN state, and one
    # attention layer's KV slice (P1-ARITHMETIC.md S4: the per-layer KV slice is the only engine-shaped
    # staging that survives the intervening traffic on arithmetic) — in both window regimes.
    for spec in "state $STATE_MIB" "kv-layer $KV_SLICE_MIB"; do
        set -- $spec
        for regime in idle spin; do
            for arm in cold warm polluted; do
                run "$1" "$2" "$arm" "$regime" 64 21 "$PASS" || BAD=1
            done
        done
    done
    # B: the coldness dose curve. Must reproduce D4's saturation at 32 MiB or explain it.
    for setup in 32 64 256; do
        for arm in cold warm polluted; do
            run state "$STATE_MIB" "$arm" idle "$setup" 21 "$PASS" || BAD=1
        done
    done
    # C: the residency cliff. Where the prize dies is the engine's staging budget.
    for inter in 7 21 43 64; do
        for arm in cold warm polluted; do
            run state "$STATE_MIB" "$arm" idle 64 "$inter" "$PASS" || BAD=1
        done
    done
    # D: a staged form larger than the tier. Retired by arithmetic rather than by timing: one layer's
    # weight block is 41.076157 MiB, which is bigger than the ~32 MiB fast tier (devices/
    # strix-halo-memory-bandwidth.md), so it is not a residency candidate at all. The rig also refuses
    # to witness it: that byte count is not a multiple of a 64 B line, so an over-read witness would be
    # ambiguous (`REFUSE invariants: region not a multiple of a 64 B line`). Excluded by a DERIVED test
    # and printed, not dropped silently, so a reader can tell "not a candidate" from "not run".
    for arm in cold warm polluted; do
        verdict=$(awk -v m="$WEIGHTS_MIB" -v tier="$TIER_BYTES" 'BEGIN{b=int(m*1048576); printf "%d %d %d", b, (b<=tier?1:0), (b%64==0?1:0)}')
        set -- $verdict
        if [ "$2" = 0 ] || [ "$3" = 0 ]; then
            [ "$arm" = cold ] && echo "EXCLUDED target=weights bytes=$1 tier_bytes=$TIER_BYTES fits_tier=$2 line_multiple=$3 — not a residency candidate and not witnessable; no timing taken" | tee -a "$LOG"
        else
            run weights "$WEIGHTS_MIB" "$arm" idle 64 21 "$PASS" || BAD=1
        fi
    done
done
echo "RESULTS-PRINTED=$ROWS" >> "$RES"
echo "util at release: $("$TOOLS" util)" | tee -a "$LOG"
"$TOOLS" release "$TAG" | tee -a "$LOG"
trap - EXIT INT TERM

echo "wrote $RES ($ROWS timed rows, 2 passes inside one lock hold)" | tee -a "$LOG"
if [ "$BAD" -ne 0 ]; then
    echo "SWEEP INCOMPLETE: at least one row printed no RESULT or went red. Do not fit it." | tee -a "$LOG"
    exit 1
fi
echo "Next: python3 $HERE/fit_prefetch.py $RES" | tee -a "$LOG"
