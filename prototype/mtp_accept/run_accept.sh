#!/usr/bin/env bash
# P7 / MTP — the accept rate `a` on the engine's own bits, in ONE GPU-lock hold.
#
# WHAT THIS IS. `run_accept.py` is the instrument and this is its protocol: build, admit, measure,
# tally, all inside a single `tools/gpu_lock.sh` acquisition so no second device user can land between
# the admission and the measurement, and so the lock cannot be left held by a stage that failed. It is
# blocking by construction — no `&`, no `nohup`, no backgrounded run, per the standing rule that a
# backgrounded device run keeps the serving stream alive and contaminates itself.
#
# ORDER, and why it is this order. Admission first: a measurement taken on a binary that changed the
# target's bits is worthless, so the gate runs with the draft flag OFF and then ON, and the target's
# own output must be byte-identical between the two (`sampled.txt` and every `STATEFP` row). Only then
# does the corpus pass run, and only then is the accept tally printed. If admission goes red the script
# stops and prints NO-NUMBER, because a number from a binary that moved the target is not a number.
#
# WALL, per stage, so a long run can be split where it is cheap to split (measured in a worker window,
# util 81-93 % at acquire — a worker's window, so treat these as lower bounds for an exclusive run):
#     build            ~18 s   (`engine/build.sh`, both binaries)
#     engine gate      the dominant term by far; `engine/gate.py` runs the full numerics pass per
#                      fixture and is the main agent's measurement to take, not a worker's. It is the
#                      first thing to drop if a sitting window is short: STAGES=admission,corpus.
#     byte identity    ~12 s   (four `decode_run` runs at 24 steps, off and on)
#     corpus pass      ~19 s per schedule (49 lines x 32 steps, one resident session)
#     witness          ~110 s torch CPU, no GPU at all, and skippable (`STAGES` without `witness`)
#   Everything except the gate fits inside ~3 minutes, so the default stage set stays well under the
#   10-minute split line; the gate is what pushes it past, and it is optional per invocation. The
#   witness stage is torch CPU and runs AFTER the lock is released, so the box is not held for it.
#
# USAGE
#   tools/gpu_lock.sh status            # check nobody else holds it (and no STALE holder)
#   prototype/mtp_accept/run_accept.sh                      # everything except the gate
#   RUN_GATE=1 prototype/mtp_accept/run_accept.sh           # with the engine gate, off and on
#   STAGES=corpus SCHED=every prototype/mtp_accept/run_accept.sh    # one split piece
#
# Stages: build, gate, identity, corpus, witness, tally. `STAGES` is a comma list; the default is
# `build,identity,corpus,witness` (the gate is opt-in because of its wall, the tally always runs with
# the corpus pass). NO TIMING IS TAKEN from any of it: the printed walls are the harness's own minutes,
# labelled as such, and nothing here prints a millisecond figure for the engine.set -euo pipefail
export LC_ALL=C

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PY=${PY:-.venv/bin/python3}
BIN=${BIN:-engine/build/decode_run}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
OUT=${OUT:-/tmp/mtp_accept}
STEPS=${STEPS:-32}
SCHED=${SCHED:-loop}
GATE_FIXTURE=${GATE_FIXTURE:-dec8}
RUN_GATE=${RUN_GATE:-0}
STAGES=${STAGES:-build,identity,corpus,witness}
TAG=${TAG:-p7 mtp accept on engine bits}

case "$STEPS" in
    ''|*[!0-9]*) echo "STEPS must be a integer, got '$STEPS'" >&2; exit 2 ;;
esac
if [ "$STEPS" -lt 24 ]; then
    echo "STEPS must be >= 24: the protocol prices >= 24 generated tokens per line" >&2
    exit 2
fi
mkdir -p "$OUT"

has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

T0=$(date +%s)
stage_start() { STAGE_T0=$(date +%s); printf '\n=== %s ===\n' "$1"; }
stage_end() { printf -- '--- %s: wall %ss (harness minutes, not an engine timing) ---\n' \
                       "$1" "$(( $(date +%s) - STAGE_T0 ))"; }

fail() { printf '\nADMISSION FAILED: %s\nNO-NUMBER: the accept rate was not measured, because a binary\n' \
                 "that moved the target's bits cannot be the one that is quoted.\n" "$1" >&2; exit 1; }

tools/gpu_lock.sh acquire "$TAG" || exit 1
released=0
release() { [ "$released" = 1 ] || { tools/gpu_lock.sh release "$TAG"; released=1; }; }
trap release EXIT INT TERM
echo "lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util)"

if has_stage build; then
    stage_start "build (the tree as it stands, not the working tree you remember)"
    bash engine/build.sh | tail -2
    stage_end build
fi

if has_stage gate; then
    stage_start "engine gate, draft flag OFF ($GATE_FIXTURE)"
    $PY engine/gate.py --binary "$BIN" --model "$MODEL" --fixture "$GATE_FIXTURE" \
        > "$OUT/gate_off.log" 2>&1 || fail "the engine gate is red with --mtp-draft off; see $OUT/gate_off.log"
    tail -3 "$OUT/gate_off.log"
    stage_end "gate off"

    stage_start "engine gate, draft flag ON ($GATE_FIXTURE)"
    # The flag reaches the gate through the environment, which is how every other engine knob gets in
    # (`QWEN_GEMM_*`, `QWEN_COPY_FOLD`); gate.py is not taught a flag it does not need, and the point of
    # running it green on both sides is that the DRAFT lines are invisible to the target's own checks.
    # The schedule is deliberately NOT set from here: `QWEN_MTP_DRAFT` is the only environment knob the
    # engine has, and `--mtp-sched` is a command-line flag, so a gate run adjudicates the default
    # schedule rather than one an env var invented.
    QWEN_MTP_DRAFT=1 $PY engine/gate.py --binary "$BIN" --model "$MODEL" \
        --fixture "$GATE_FIXTURE" > "$OUT/gate_on.log" 2>&1 \
        || fail "the engine gate is red with QWEN_MTP_DRAFT=1; see $OUT/gate_on.log"
    grep -q '^MTPDRAFT on=1' "$OUT/gate_on.log" \
        || fail "gate ran with QWEN_MTP_DRAFT=1 but no MTPDRAFT line came back: the flag did not reach the engine, so this 'green' adjudicates nothing"
    tail -3 "$OUT/gate_on.log"
    stage_end "gate on"
fi

if has_stage identity; then
    stage_start "byte identity: flag off vs flag on on the same prompts"
    # The D1 admission test, and the only one that cannot be satisfied by a number: the side computation
    # may read the finished step's hidden state and emit a proposal id, and it may not alter ONE BYTE
    # the target path consumes. `sampled.txt` is the token stream and `STATEFP` is the per-step state
    # fingerprint, so together they are the whole observable surface of the target's bits.
    #
    # Two prompts, chosen for what each can falsify. `dec8`'s own prefill contains id 248046 in the
    # MIDDLE of the prompt (§07.5: a prompt-phase match must never stop anything), so if the draft path
    # had disturbed the stop rule the flag-on side would end early and `sampled.txt` would differ in
    # length. The corpus line is ordinary prose, which is what the accept pass runs on.
    $PY -c "import json,pathlib;pathlib.Path('$OUT/ad_dec8.txt').write_text(','.join(str(i) for i in json.load(open('data/refs/dec8/manifest.json'))['prefill_ids'])+'\n')"
    $PY -c "import json,pathlib;pathlib.Path('$OUT/ad_corpus0.txt').write_text(','.join(str(i) for i in json.load(open('data/refs/corpus48/manifest.json'))['prompts'][0]['ids'])+'\n')"
    for name in ad_dec8 ad_corpus0; do
        pf="$OUT/$name.txt"
        [ -s "$pf" ] || fail "could not write the prompt id file $pf"
        "$BIN" --model "$MODEL" --ids-file "$pf" --steps 24 --out "$OUT/id_off_$name" \
            --state-fp > "$OUT/id_off_$name.log" 2>&1 || fail "flag-off run failed on $name"
        "$BIN" --model "$MODEL" --ids-file "$pf" --steps 24 --out "$OUT/id_on_$name" \
            --state-fp --mtp-draft 1 --mtp-sched "$SCHED" > "$OUT/id_on_$name.log" 2>&1 \
            || fail "flag-on run failed on $name"
        cmp "$OUT/id_off_$name/sampled.txt" "$OUT/id_on_$name/sampled.txt" \
            || fail "sampled.txt differs with the draft flag on, on $name"
        grep '^STATEFP' "$OUT/id_off_$name.log" > "$OUT/off_fp.txt"
        grep '^STATEFP' "$OUT/id_on_$name.log"  > "$OUT/on_fp.txt"
        cmp "$OUT/off_fp.txt" "$OUT/on_fp.txt" \
            || fail "a STATEFP fingerprint row differs with the draft flag on, on $name"
        [ "$(grep -c '^STEP' "$OUT/id_off_$name.log")" = "$(grep -c '^STEP' "$OUT/id_on_$name.log")" ] \
            || fail "the two runs printed a different number of STEP lines on $name: the target's own transcript moved"
        grep -q '^MTPDRAFT on=1' "$OUT/id_on_$name.log" \
            || fail "flag-on run on $name printed no MTPDRAFT line; nothing was drafted, so the pair proves nothing"
        # The census is asserted against what the code declares, so a printed bill cannot drift from a
        # later build of the launch chain.
        grep '^MTPDRAFT on=1' "$OUT/id_on_$name.log" | tail -1
        grep -q 'block=1+13 head=1+1 per_round=16' "$OUT/id_on_$name.log" \
            || fail "the census line does not declare the launch plan the draft driver submits"
        printf '  %s: sampled.txt and %s STATEFP rows byte-identical off vs on\n' \
               "$name" "$(wc -l < "$OUT/off_fp.txt")"
    done

    # The stronger form of the same claim, when a pre-MTP binary is about: off-vs-on inside ONE binary
    # could in principle be satisfied by a change that moves both sides. Against a binary built before the
    # draft path existed, it cannot. Asserted with a behavioural canary, because "two files differ" is not
    # proof either way (`AGENTS.md`: a paired A/B against a stale binary reads as a clean null, and the
    # canary is what catches that the two sides are not what they were said to be).
    if [ -x engine/build/decode_run.premtp ]; then
        for name in ad_dec8 ad_corpus0; do
            pf="$OUT/$name.txt"
            engine/build/decode_run.premtp --model "$MODEL" --ids-file "$pf" --steps 24 \
                --out "$OUT/pre_$name" --state-fp > "$OUT/pre_$name.log" 2>&1 \
                || fail "the pre-MTP binary failed on $name"
            cmp "$OUT/pre_$name/sampled.txt" "$OUT/id_off_$name/sampled.txt" \
                || fail "$name: today's flag-off sampled.txt is not the pre-MTP binary's"
            grep '^STATEFP' "$OUT/pre_$name.log" > "$OUT/pre_fp.txt"
            grep '^STATEFP' "$OUT/id_off_$name.log" > "$OUT/off_fp.txt"
            cmp "$OUT/pre_fp.txt" "$OUT/off_fp.txt" \
                || fail "$name: today's flag-off STATEFP rows are not the pre-MTP binary's"
            diff <(grep '^STEP' "$OUT/pre_$name.log") <(grep '^STEP' "$OUT/id_off_$name.log") >/dev/null \
                || fail "$name: a STEP line moved relative to the pre-MTP binary"
            # Written as an `if` rather than `cmd && fail`, because the point is that this command MUST
            # fail, and a construct whose expected outcome is a non-zero exit should not depend on how
            # `set -e` treats a short-circuited list.
            if engine/build/decode_run.premtp --mtp-draft 1 --ids-file "$pf" --steps 1 >/dev/null 2>&1
            then
                fail "the pre-MTP binary ACCEPTED --mtp-draft, so it is not a pre-MTP build and the pair above adjudicates nothing"
            fi
            printf '  %s: flag-off today == the pre-MTP binary (sampled.txt, %s STATEFP rows, every STEP line); canary: that binary rejects --mtp-draft\n' \
                   "$name" "$(wc -l < "$OUT/pre_fp.txt")"
        done
    else
        echo "  note: no engine/build/decode_run.premtp to compare against, so only the off-vs-on form of"
        echo "        the identity check ran. Save a copy of the binary BEFORE the first MTP build if the"
        echo "        stronger form is wanted: cp engine/build/decode_run engine/build/decode_run.premtp"
    fi
    echo "ADMISSION: PASS — the draft is a strict side computation on these fixtures."
    stage_end identity
fi

if has_stage corpus; then
    stage_start "corpus pass + accept tally ($SCHED, $STEPS steps per line)"
    $PY prototype/mtp_accept/run_accept.py --binary "$BIN" --model "$MODEL" --steps "$STEPS" \
        --sched "$SCHED" --json "$OUT/engine_$SCHED.json" | tee "$OUT/engine_$SCHED.log"
    stage_end corpus
fi

if has_stage witness; then
    # Out of the lock on purpose: torch CPU holds nothing on the GPU, and holding the box through 2
    # minutes of CPU work would starve every other device user for no reason. `release` is idempotent
    # and the ERR/INT trap still covers a failure in any earlier stage, so the lock cannot leak.
    release
    stage_start "witness: the same protocol on torch bits (CPU only, no device work in this stage)"
    eng=""
    [ -f "$OUT/engine_$SCHED.json" ] && eng="--engine-json $OUT/engine_$SCHED.json"
    $PY prototype/mtp_accept/witness_torch.py --protocol same --sched "$SCHED" --steps "$STEPS" \
        --json "$OUT/witness_same.json" $eng | tail -4
    stage_end witness
fi

printf '\n=== total: %ss of harness wall ===\n' "$(( $(date +%s) - T0 ))"
echo "artifacts in $OUT: engine_$SCHED.{json,log}, and witness_same.json where the witness ran."
