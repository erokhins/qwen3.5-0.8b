#!/usr/bin/env bash
# flagoff_bytecheck.sh — does a draft-side patch move a single bit of the TARGET's own output?
#
# The invariant every MTP conditioning variant here must satisfy: a legitimate variant changes only the
# PROPOSED token. The strongest device-side statement of that is byte identity on the flag-off path, so
# this runs the same fixture through the pre-change binary and the patched one with no MTP flag at all and
# `diff -r`s the dump directories: `sampled.txt`, `emitted.txt` and every `logits*.bin`.
#
# Why this is the control and not a formality: `engine/build.sh` is not bit-reproducible on this box, so
# comparing BINARY files proves only that they are two builds. This compares what the two binaries COMPUTE.
# A patched engine that reached into the body, the verdict or the state would show here; a better guess
# cannot.
#
# NO TIMING: no millisecond figure is read or printed. `wall` below is this harness's own seconds.
#
# CANARY (optional): a flag the new binary must accept and the old must reject, e.g.
#   CANARY="--mtp-draft-k 2" CANARY_MARK="^DRAFTK pos=" prototype/mtp_accept/flagoff_bytecheck.sh
# Without it the byte compare cannot tell "the patch does not move the target" from "both sides are the
# same build", which is the failure this file's own header warns about.
#
# Usage:
#   prototype/mtp_accept/flagoff_bytecheck.sh                      # HEAD worktree vs working tree, dec8
#   prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run \
#           engine/build/decode_run /tmp/dcp/dec8.csv 32
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

OLD=${1:-/tmp/prewt/engine/build/decode_run}
NEW=${2:-engine/build/decode_run}
IDS=${3:-/tmp/dcp/dec8.csv}
STEPS=${4:-32}
OUT=/tmp/flagoff_bytecheck   # fixed literal: the cleanup below and this path must be the same object
MODEL=${MODEL:-models/Qwen3.5-0.8B}

[ -x "$OLD" ] || { echo "no pre-change binary at $OLD (git worktree add /tmp/prewt HEAD --detach && (cd /tmp/prewt && engine/build.sh))"; exit 1; }
[ -x "$NEW" ] || { echo "no patched binary at $NEW"; exit 1; }
[ -f "$IDS" ] || { echo "no ids file at $IDS"; exit 1; }
N_IDS=$(awk -F',' '{print NF}' "$IDS")
[ "$N_IDS" -ge 8 ] || { echo "$IDS has $N_IDS comma fields — that is the one-token silent failure"; exit 1; }

TAG=${TAG:-worker flagoff-bytecheck}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"
echo "== fixture=$IDS fields=$N_IDS steps=$STEPS"

s=$(date +%s)
rc=0
# Literal path on purpose (the safety net cannot verify a variable target, and a stale file from an earlier
# run would make the compare mean something other than what these two runs wrote).
rm -rf /tmp/flagoff_bytecheck
for arm in old new; do
    bin=$OLD; [ "$arm" = new ] && bin=$NEW
    out=$OUT/$arm
    mkdir -p "$out"
    echo "  $ $bin --model $MODEL --ids-file $IDS --steps $STEPS --no-stop --out $out"
    "$bin" --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --no-stop --out "$out" \
        > "$OUT/$arm.log" 2>&1 || { echo "RUN FAIL arm=$arm rc=$?"; tail -4 "$OUT/$arm.log"; rc=1; }
done
echo "--- flag-off byte compare: $OLD vs $NEW"
if [ "$rc" = 0 ]; then
    if diff -r "$OUT/old" "$OUT/new" > "$OUT/diff.txt" 2>&1; then
        echo "CONTROL: IDENTICAL — $(ls "$OUT/old" | wc -l) files byte-identical (sampled.txt, emitted.txt, every logits*.bin). The patch moves no bit of what the TARGET computes."
    else
        echo "CONTROL: DIFFERS — the patch reached the target path. Do not read any accept rate from it."
        head -20 "$OUT/diff.txt"
        rc=1
    fi
fi

# --- the behavioural canary (`AGENTS.md`: a paired A/B against a stale binary reads as a clean null) -----
# A byte-identical pair is also what two copies of the SAME binary produce, and `engine/build.sh` is not
# bit-reproducible here, so "identical" alone cannot say which side carries the change. `CANARY` is a flag
# the NEW binary must accept and produce `CANARY_MARK` with, and the OLD one must reject outright — then the
# pair above is demonstrably pre-change vs post-change, and the identical bytes are the post-change
# binary's refusal to move the target rather than two runs of one build. Generic on purpose: any
# default-off knob can be named, and with no CANARY set this section says it was skipped instead of
# pretending to have run.
if [ "$rc" = 0 ] && [ -n "${CANARY:-}" ]; then
    MARK=${CANARY_MARK:-}
    if "$OLD" --model "$MODEL" --ids-file "$IDS" --steps 4 $CANARY > "$OUT/canary_old.log" 2>&1; then
        echo "CANARY: VACUOUS — the pre-change binary ACCEPTED '$CANARY', so it is not a pre-change build"
        echo "        and the byte compare above adjudicates nothing."
        rc=1
    else
        echo "CANARY: the pre-change binary rejects '$CANARY' (as it must: it predates the knob)"
    fi
    if ! "$NEW" --model "$MODEL" --ids-file "$IDS" --steps 4 $CANARY > "$OUT/canary_new.log" 2>&1; then
        echo "CANARY: FAIL — the new binary rejected '$CANARY', so the patch is not in the binary that ran the byte compare."
        tail -3 "$OUT/canary_new.log"
        rc=1
    elif [ -n "$MARK" ] && ! grep -q "$MARK" "$OUT/canary_new.log"; then
        echo "CANARY: FAIL — the new binary accepted '$CANARY' but printed no '$MARK' line: the flag took but the feature did not."
        rc=1
    else
        echo "CANARY: the new binary accepts '$CANARY'$( [ -n "$MARK" ] && printf " and prints %s lines" "$MARK" )"
        echo "CONTROL: CANARY-OK — the two sides of the byte compare are demonstrably different builds, one without the knob and one with it armed."
    fi
fi
echo "== harness wall $(( $(date +%s) - s ))s (its own seconds, not an engine timing); no timing taken"
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
