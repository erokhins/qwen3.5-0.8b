#!/usr/bin/env bash
# flagoff_bytecheck.sh — does a draft-side patch move a single bit of the TARGET's own output?
#
# The invariant every MTP conditioning variant here must satisfy: a legitimate variant changes only the
# PROPOSED token. The strongest device-side statement of that is byte identity on the flag-off path, so
# this runs the same fixture through the pre-change binary and the patched one with no MTP flag at all and
# `diff -r`s the dump directories: `sampled.txt`, `emitted.txt` and every `logits*.bin`.
#
# MODE=accept runs the OTHER half of the claim: both arms run the SHIPPING ACCEPT LOOP
# (`--gdn-loop 1 --mtp-draft 1 --mtp-accept 1`) and the comparison is of the DECISIONS — every `ACCEPT`,
# `ACCEMIT`, `MTPACCEPT`, `DRAFT`, `MTPSTAT` line and `emitted.txt`. That is the admission
# `prototype/mtp_accept/ROUND-TAX.md` pre-registered for removing the round tax: an accept decision or a
# draft proposal that moves by one token fails the change, because the argmax rule the patch claims to
# reproduce is a total order and an exact-equality claim is admitted by a byte-compare, never by a band.
# `MTPDRAFT` is deliberately NOT compared in this mode: its `head=`/`per_round=` fields are the DECLARED §09
# plan, which the round tax legitimately changes (two launches join the head group), and `engine/gate.py`
# adjudicates that line — comparing it here would make a correct patch look like a defect.
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
# NEW_MARK / OLD_MARK (optional, the same job for a patch that adds NO flag): a pattern the NEW binary must
# print and the OLD one must not, e.g.
#   NEW_MARK='head=1\+3 per_round=18'   # the §09 census after the device argmax
# A round-tax patch changes no flag and no decision, so the census line — the engine's own statement of its
# per-round plan — is what can show the two sides are different builds and which one carries the work.
#
# Usage:
#   prototype/mtp_accept/flagoff_bytecheck.sh                      # HEAD worktree vs working tree, dec8
#   prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run \
#           engine/build/decode_run /tmp/dcp/dec8.csv 32
#   MODE=accept prototype/mtp_accept/flagoff_bytecheck.sh <old> <new> <ids.csv> <steps>
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

OLD=${1:-/tmp/prewt/engine/build/decode_run}
NEW=${2:-engine/build/decode_run}
IDS=${3:-/tmp/dcp/dec8.csv}
STEPS=${4:-32}
OUT=/tmp/flagoff_bytecheck   # fixed literal: the cleanup below and this path must be the same object
MODEL=${MODEL:-models/Qwen3.5-0.8B}
MODE=${MODE:-flagoff}
# The accept loop's two-row body needs room for the prompt plus its own rows, and a too-small `cap` makes
# every round throw `context limit` — a failure that reads as a slow run rather than a wrong one.
CAP=${CAP:-4300}
case "$MODE" in
  flagoff) RUN_FLAGS=() ;;
  accept)  RUN_FLAGS=(--gdn-loop 1 --mtp-draft 1 --mtp-accept 1 --cap "$CAP") ;;
  *) echo "MODE must be flagoff or accept, got '$MODE'"; exit 2 ;;
esac

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
    echo "  $ $bin --model $MODEL --ids-file $IDS --steps $STEPS --no-stop --out $out ${RUN_FLAGS[*]:-}"
    "$bin" --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --no-stop --out "$out" \
        ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} > "$OUT/$arm.log" 2>&1 \
        || { echo "RUN FAIL arm=$arm rc=$?"; tail -4 "$OUT/$arm.log"; rc=1; }
done
echo "--- byte compare ($MODE mode): $OLD vs $NEW"
if [ "$rc" = 0 ]; then
    if diff -r "$OUT/old" "$OUT/new" > "$OUT/diff.txt" 2>&1; then
        # What the directory holds depends on the mode: a flag-off run dumps `sampled.txt` and one
        # `logits*.bin` per step, an accept round dumps only `emitted.txt` (it writes no per-step logits),
        # so the claim names the mode rather than reciting a list that is not always true.
        echo "CONTROL: IDENTICAL — $(ls "$OUT/old" | wc -l) dump file(s) byte-identical in $MODE mode"
        echo "        ($(ls "$OUT/old" | tr '\n' ' ')). The patch moves no bit of what the TARGET computes."
    else
        echo "CONTROL: DIFFERS — the patch reached the target path. Do not read any accept rate from it."
        head -20 "$OUT/diff.txt"
        rc=1
    fi
fi

# --- the accept loop's decisions, in accept mode ----------------------------------------------------------
# The dump directories above compare what the TARGET emitted; this compares what the ROUND decided.
# `ACCEPT` is the verdict itself (candidate, verifier token, bonus token, who was emitted, what was
# restored), `ACCEMIT` is the emitted stream with positions, `MTPACCEPT`/`DRAFT` are the adjudication and
# the proposal, `MTPSTAT` is the tally the accept rate is read from. They are pulled out by line prefix
# rather than diffing whole logs because a log also carries duration fields, and a duration is allowed to
# move. `MTPDRAFT` is NOT in the set: see the header for why the declared §09 plan is the gate's business.
#
# The must-catch set is derived from the geometry, not hoped for: the `ACCEPT` lines ARE the verdict
# kernel's output, so if no round ran the compare has nothing to catch and a green result would be vacuous
# (`AGENTS.md`: a negative control with an empty must-catch set is not a control). Hence the refusal below.
if [ "$rc" = 0 ] && [ "$MODE" = accept ]; then
    for arm in old new; do
        grep -E '^(ACCEPT |ACCEMIT |MTPACCEPT |DRAFT |DRAFT-SKIP |DRAFTK |MTPSTAT )' "$OUT/$arm.log" \
            > "$OUT/$arm.decisions"
        # Declared-bill fields come OUT, for the same reason `MTPDRAFT`'s whole line is absent: a round's
        # submission bill is DECLARED PLAN — `engine/gate.py`'s §09 census adjudicates it — and a byte
        # compare that reddens on the change it was built to make adjudicates nothing. Not a theory: defect
        # (B) replaces the verdict's one launch with a strip pass and a combine, `split_subs` moved by
        # exactly one per round (2420 -> 2483 over 63 rounds) while every `ACCEPT` line and `emitted.txt`
        # stayed byte-identical, and the compare called that a divergence. Everything decision-bearing on
        # the line stays in: `accepted=`, `rejected=`, `emitted=`, `accept_rate=`, `mean_tokens_per_round=`.
        sed -E 's/[[:space:]](split_subs|stage_subs_per_round)=[0-9]+//; s/[[:space:]]+$//' \
            "$OUT/$arm.decisions" > "$OUT/$arm.decisions.n" && mv "$OUT/$arm.decisions.n" "$OUT/$arm.decisions"
    done
    N_DEC=$(wc -l < "$OUT/old.decisions")
    N_ACCEPT=$(grep -c '^ACCEPT ' "$OUT/old.log")
    if [ "$N_ACCEPT" -lt 1 ]; then
        echo "CONTROL: VACUOUS — the pre-change run produced 0 ACCEPT lines at steps=$STEPS on $IDS, so"
        echo "        there are no decisions to compare. Raise STEPS; a compare over nothing passes always."
        rc=1
    elif diff "$OUT/old.decisions" "$OUT/new.decisions" > "$OUT/decisions.diff" 2>&1; then
        echo "CONTROL: DECISIONS IDENTICAL — $N_DEC lines (verdicts, emits, proposals, the tally) plus"
        echo "        emitted.txt, byte-identical on $IDS over $N_ACCEPT rounds."
    else
        echo "CONTROL: DECISIONS DIFFER — the argmax rule is not what the spec claims. Revert; do not band"
        echo "        it and do not wave it through with a tolerance."
        head -20 "$OUT/decisions.diff"
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
# --- the same job for a patch that adds NO flag: `NEW_MARK` ------------------------------------------------
# A round-tax patch introduces no new knob, so the `CANARY` form above has nothing to name. The evidence is
# still needed — "both sides printed identical decisions" is also what two runs of ONE binary prints — so
# `NEW_MARK` is a pattern the NEW binary must print in the run that just happened and the OLD one must not,
# read off the logs already captured above (no extra device runs). For the round tax the natural mark is the
# engine's own §09 census (`head=1+3 per_round=18` after the draft argmax moved on device), which is a
# statement about the binary's plan rather than about the fixture.
if [ "$rc" = 0 ] && [ -n "${NEW_MARK:-}" ]; then
    if ! grep -q "$NEW_MARK" "$OUT/new.log"; then
        echo "NEW_MARK: FAIL — the new binary printed no '$NEW_MARK' in its run, so the patch is not in the"
        echo "          binary that produced the byte compare above."
        rc=1
    elif grep -q "$NEW_MARK" "$OUT/old.log"; then
        echo "NEW_MARK: VACUOUS — the pre-change binary ALSO prints '$NEW_MARK', so it is not pre-change"
        echo "          and the identity above adjudicates nothing."
        rc=1
    else
        echo "NEW_MARK: ok — the new side prints '$NEW_MARK', the pre-change side does not: two different"
        echo "          builds, and the identical decisions are the new one refusing to move a decision."
    fi
fi
echo "== harness wall $(( $(date +%s) - s ))s (its own seconds, not an engine timing); no timing taken"
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
