#!/usr/bin/env bash
# ab_item1.sh — the paired wall A/B for route S1 item (1): delete the 97 `cast` launches by folding the
# BF16->F32 widen into the kernels that consume the result. MAIN AGENT ONLY: this takes milliseconds and
# a worker may not (AGENTS.md, "GPU execution is serialized and belongs to the main agent"). It acquires
# tools/gpu_lock.sh itself and holds it for the whole window; run it BLOCKING, never backgrounded, with
# zero subagents active.
#
#   bash prototype/cast_fold/ab_item1.sh
#   REPS=5 GATE_FIXTURES="dec8 wide4096f" bash prototype/cast_fold/ab_item1.sh
#
# ============================================================================ WHAT IS BEING TESTED
# `engine/generation-limits.md` §1 bills the `cast` family 97 launches, 0.371 MiB read and 0.741 MiB
# written a token (quoted), at a 0.7775 ms stage line (quoted) — i.e. no bytes of consequence and 7.78 us
# a launch, of which the stage clock's own ~5.3 us/segment (quoted, §4) is the instrument. The residual,
# ~3.8 us a launch, is the thing nobody has ever measured on the shipping path. Item (1) is the direct
# test: fold the widen into each consumer — bit-identical, because a BF16 widened to F32 is the same real
# number (`prototype/cast_fold/equiv_check.cpp` proves the operand values equal element for element, at
# every index and every one of the 65,536 BF16 patterns) — delete all 97 launches, and read the WALL.
#
# POPULATION, and why the lines need no scaling: `prototype/cast_fold/census_sites.py` derives six call
# sites from engine/decode_step.cpp and all six are foldable-now, so for the thing being timed — a decode
# step, a body of ONE row — the folded count is 97 of 97 and the pre-registered lines stand as written.
# The fold is a one-row rule (`sc.cast_fold && (n == 1)`); a wide prefill body folds only `finish_token`'s
# final-norm widening, so TTFT is not what this window prices. If a later refactor folds fewer sites, the
# script scales the lines by the `folded=` field it reads out of its own runs rather than trusting 97.
#
# ============================================================================ THE PRE-REGISTERED READING
# Quoted from `engine/generation-limits.md` §7 (the arm, and the lines, were set before this existed):
#     saves >= 0.24 ms on the wall -> the ~3.8 us/launch residual is real and `glue`/`copy` (~116 more
#                                    tiny launches) become a funded route
#     saves <= 0.05 ms             -> launches are free when pipelined, the small-op tail is an
#                                    instrument artefact, and the route closes
#     between                      -> neither, reported as neither
# Lines are on the DECODE STEP WALL at 4,032 lanes (4,000 prompt + 32 steps), median of REPS paired reps,
# scaled by folded/97 (which this build reads as 97, so the scaling is 1.000 and the lines are unchanged).
# At 128 lanes there is NO line: D3 established short widths are one round at either residency and the
# S1 brief §4 says a flat 128 row is a predicted PASS. Report it beside the headline, never instead.
#
# ONE HONEST CAVEAT ABOUT WHAT A NULL WOULD MEAN. The consumer-side staging has a measured precedent
# against it in `engine/results.md`'s cast row, which priced the projection family's `PACK_X` staging as
# a net LOSS (the number is in that row, quoted) — that measurement was made on `proj_gemv_tokens` at
# rows/block 32, before route D1 shipped the grouped kernel this fold now rides on, and it halved the LDS
# per block while it did. So a null or a negative here is ambiguous between "launches are free when
# pipelined" and "the fold costs more in-register than the launches were worth", and the discriminator is
# printed with the verdict: the per-family stage lines. If `proj`/`mlp`/`lm_head` rose while `cast` fell to
# nothing, it is the staging's cost; if every family line is flat and the wall is flat, it is the launch
# price being instrument, which is the reading §7 asked for.
#
# ============================================================================ THE CANARIES
# `engine/build.sh` is not bit-reproducible on this box, so `cmp` between two builds proves only that they
# are two builds, and any self-check compiled into the binary under test is blind to the one failure mode
# that matters (AGENTS.md 2026-10-17: a D30 A/B timed a pre-change binary as the `new` side and reported a
# clean null while every in-binary parity field stayed green). So these are asserted inside the window,
# before any millisecond is spent:
#
#   1. THE FLAG TOOK EFFECT, read two ways, because a printed field the binary chose to print is the weak
#      form of evidence:
#        a. `CASTFOLD on=... plan=... folded=... kept=...` is printed by the new binary and by NO base
#           build — a stale "new" binary looks exactly like base and is caught here. `plan=` must equal the
#           `cast=` field of the untouched `LAUNCH` plan line (two derivations, one from
#           `planned_launch_counts()` and one from the stage marker, that must agree), and with the fold on
#           `folded=plan` and `kept=0`.
#        b. THE STAGE CLOCK'S OWN OBSERVATION of submissions, plus its own plan self-check: base prints
#           `STAGE gran=stage family=cast` and `launches_per_step=340 plan_per_step=340 plan_match=1`; the
#           folded run must print NO cast row and `launches_per_step=243 plan_per_step=243 plan_match=1`.
#           `plan_per_step` is `planned_launch_counts()` in its fold-aware form and `launches_per_step`
#           counts `stageclock::note` calls at run time, so `plan_match` is the instrument agreeing with
#           the dispatch, and the A/B refuses any arm that reads `plan_match=0`. (These two fields moved
#           in the same change as the fold: the parent's review caught a first cut that left the declared
#           bill at 340 while 243 were submitted, i.e. a folded run whose own self-check went red. A
#           witness break — `planned_launch_counts(true)` forced to answer 97 — reads `plan_match=0` and
#           `body1_castfold_plan_matches_token_plan=0`, which is what makes those two fields teeth rather
#           than decoration. `plan_per_step` still reads 340 with the flag off, which is what keeps the
#           base arm's canary a statement about the fold and not about a bill that moved under both arms.)
#           The §07 `LAUNCH` line is deliberately NOT the carrier: `engine/gate.py` section (d) re-derives
#           that table in Python and its independence is worth more than a tidier echo, so `LAUNCH` keeps
#           `cast=97 total=340` in both arms and `CASTFOLD ... step_bill=` carries the bill in force.
#   2. THE EXERCISED PATH, on EVERY timed run: `PREFILL tokens=` equals the id FILE's field count
#      (`--ids-file` splits on commas, and a nominal width that is not the field count is the AGENTS
#      E2BIG/one-token trap), the run is reproducible against itself (`STAGEPARITY ... tokens_identical=1
#      logits_bitidentical=1`), and no parser accepts an empty or non-numeric field.
#   3. NUMERICS BEFORE TIMING: `engine/gate.py` per fixture for base, for new-with-the-fold-off (the
#      inertness claim: the default side must be HEAD's behaviour, `CASTFOLD on=0 folded=0 kept=97`) and
#      for new-with-the-fold-on (the claim itself, `CASTFOLD on=1 folded=97 kept=0`). Any red aborts the
#      window with `NO-TIMING` and no millisecond figure is printed — the `prototype/attn_m2/burst_m16.sh`
#      discipline. Then `sampled.txt` byte-identity base vs new-on, at BOTH 128 and 4,032 lanes, plus
#      new-off vs base to witness that the flag rather than the rebuild is the only variable.
#   4. WHAT IS AND IS NOT A PRIZE FIGURE. Only `STAGEPASS gran=burst ms_mean` is the production shape. The
#      `STAGE gran=stage TOTAL ms_per_step` figure is NOT: stage granularity syncs per segment, and a fold
#      deletes segments along with launches (measured on a contended box: 305 syncs/step at flag-off
#      against 184 at flag-on), so the serialized total falls for a reason that has nothing to do with the
#      wall. Both are printed, and the report says which one the verdict is read off.
#
# WHAT IS TIMED, AND HOW. `engine/build/decode_run --time-stages`, and the headline is
# `STAGEPASS gran=burst steps=N ms_mean=` — the pipelined production shape, one sync at the logits, which
# is what `engine/generation-limits.md`'s operating point is quoted from. Arms are INTERLEAVED rep by rep
# so clock drift shows as disagreement between reps of one arm rather than as a window effect, and each
# arm is a median with its spread, never a best.
set -uo pipefail
cd "$(dirname "$0")/../.."
export LC_ALL=C          # the box's locale uses a comma decimal separator; sort -n and awk's strtod
                         # both truncate at the '.' and the medians would be computed off integers.
VENV=${VENV:-.venv/bin/python}
OUT=${OUT:-tmp/cast_fold/ab}
TAG=${TAG:-s1-item1-castfold-ab}
REPS=${REPS:-3}                       # reps per arm, per width (>= 3, pre-registered)
GATE_FIXTURES=${GATE_FIXTURES:-dec8 wide4096f}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
BASE_BIN="$OUT/base_decode_run"; NEW_BIN="$OUT/new_decode_run"
BASE_WT="$OUT/base"
WIN_MS=${WIN_MS:-0.24}                # quoted: engine/generation-limits.md §7
KILL_MS=${KILL_MS:-0.05}              # quoted: engine/generation-limits.md §7
CAST_PLAN=${CAST_PLAN:-97}            # quoted: engine/generation-limits.md §1, re-derived by census_sites.py
mkdir -p "$OUT"
# OUT must be ABSOLUTE: the base build runs `( cd "$BASE_WT" && ... > "$OUT/build_base.log" )`, and a
# relative OUT cannot resolve from inside the worktree, which reads as "the base build failed".
OUT=$(cd "$OUT" && pwd) || { echo "ABORT: cannot resolve OUT" >&2; exit 1; }
NUM_OK=0

fail() { echo "ABORT: $*" >&2; echo "NO-TIMING: no millisecond figure is printed from this run" >&2;
         tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
med() { sort -n "$1" | awk '{a[NR]=$1} END{ if (NR==0) {print "nan"; exit}
        m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "%.4f", m }'; }
spread() { sort -n "$1" | awk '{if(m==""||$1<m)m=$1; if($1>x)x=$1} END{printf "%s..%s", m, x}'; }
want_n() { awk -F',' '{print NF}' "$1"; }
num() { case "$1" in ''|*[!0-9.]*) return 1;; *) return 0;; esac; }

# ------------------------------------------------------------------ CPU work, before the lock ----
echo "== CPU: population + equivalence admission (no device)"
$VENV prototype/cast_fold/check_equiv.py > "$OUT/equiv.log" 2>&1 \
    || fail "prototype/cast_fold/check_equiv.py is red — do not price a fold that is not wired: see $OUT/equiv.log"
grep -E "^(SITES_CHECKED|EQUIV OK)" "$OUT/equiv.log" | sed 's/^/   /'

echo "== CPU: building NEW from the working tree"
bash engine/build.sh > "$OUT/build_new.log" 2>&1 || fail "engine/build.sh (new) failed: $OUT/build_new.log"
cp engine/build/decode_run "$NEW_BIN" || fail "no $NEW_BIN"
if [ -x "$BASE_BIN" ] && [ "${REBUILD_BASE:-0}" != 1 ]; then
    echo "== CPU: reusing $BASE_BIN from an earlier run (REBUILD_BASE=1 forces a fresh HEAD build)"
else
    echo "== CPU: building BASE from HEAD in a scratch worktree (never from the touched working tree)"
    # `git worktree prune` then `rm -rf`, not `git worktree remove --force`: the force flag can delete
    # uncommitted work and this box's safety net (rightly) refuses it, and nothing here needs it — the
    # worktree is disposable and carries no edits of its own.
    git worktree prune >/dev/null 2>&1
    rm -rf "$BASE_WT"
    git worktree add "$BASE_WT" HEAD --detach > "$OUT/worktree.log" 2>&1 || fail "worktree add failed"
    ( cd "$BASE_WT" && bash engine/build.sh > "$OUT/build_base.log" 2>&1 ) \
        || fail "engine/build.sh (base) failed: $OUT/build_base.log"
    cp "$BASE_WT/engine/build/decode_run" "$BASE_BIN" || fail "no $BASE_BIN"
fi
grep -c . "$BASE_BIN" >/dev/null || fail "base binary is empty"
cmp -s "$BASE_BIN" "$NEW_BIN" && fail "base and new are the same bytes — nothing is being A/Bed"

# ---------------------------------------------------------------- the flag canary, no device -----
# `--sizes-only` loads nothing and touches nothing, but it prints the plan lines, which is where the flag
# canary lives. Run all three arms here so a stale binary is caught before a device window is opened.
plan_lines() {   # plan_lines <arm> -> the log it printed; returns 1 if the canary fields are wrong
    local arm="$1" log="$OUT/plan_$1.log" bin env
    case "$arm" in
        base) bin="$BASE_BIN"; env="" ;;
        new-off) bin="$NEW_BIN"; env="" ;;
        new-on) bin="$NEW_BIN"; env="QWEN_CAST_FOLD=1" ;;
        *) fail "plan_lines: unknown arm $arm" ;;
    esac
    env $env "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --sizes-only > "$log" 2>&1 \
        || { echo "  plan $arm: run failed (see $log)" >&2; return 1; }
    local launch_cast
    launch_cast=$(grep -E '^LAUNCH ' "$log" | head -1 | sed -E 's/.* cast=([0-9]+).*/\1/')
    num "${launch_cast:-x}" || { echo "  plan $arm: no cast= on the LAUNCH line" >&2; return 1; }
    [ "$launch_cast" = "$CAST_PLAN" ] \
        || { echo "  plan $arm: LAUNCH bills cast=$launch_cast, the §07 plan is $CAST_PLAN" >&2; return 1; }
    if [ "$arm" = base ]; then
        grep -q '^CASTFOLD ' "$log" \
            && { echo "  plan base: a base build printed a CASTFOLD line — BASE IS NOT HEAD, or NEW is what you built into it" >&2; return 1;
                 true; }
        echo "  plan base:    no CASTFOLD line, LAUNCH cast=$launch_cast (expected: base does not know the fold)"
        return 0
    fi
    local cf on plan folded kept
    cf=$(grep -E '^CASTFOLD ' "$log" | head -1) || return 1
    [ -n "$cf" ] || { echo "  plan $arm: the new binary printed no CASTFOLD line — THE FOLD IS NOT IN THIS BINARY" >&2; return 1; }
    on=$(echo "$cf" | sed -E 's/.* on=([01]+).*/\1/');     plan=$(echo "$cf" | sed -E 's/.* plan=([0-9]+).*/\1/')
    folded=$(echo "$cf" | sed -E 's/.* folded=([0-9]+).*/\1/'); kept=$(echo "$cf" | sed -E 's/.* kept=([0-9]+).*/\1/')
    for v in "$on" "$plan" "$folded" "$kept"; do num "$v" || { echo "  plan $arm: unparsable CASTFOLD line: $cf" >&2; return 1; }; done
    [ "$plan" = "$launch_cast" ] \
        || { echo "  plan $arm: CASTFOLD plan=$plan disagrees with LAUNCH cast=$launch_cast" >&2; return 1; }
    if [ "$arm" = new-on ]; then
        { [ "$on" = 1 ] && [ "$folded" = "$plan" ] && [ "$kept" = 0 ]; } \
            || { echo "  plan $arm: QWEN_CAST_FOLD=1 but CASTFOLD reads on=$on folded=$folded kept=$kept — the env did not reach the engine, or the fold does not cover the plan" >&2; return 1; }
        FOLDED="$folded"; PLAN="$plan"
    else
        { [ "$on" = 0 ] && [ "$folded" = 0 ]; } \
            || { echo "  plan $arm: no env but CASTFOLD reads on=$on folded=$folded — the default side is not inert" >&2; return 1; }
    fi
    echo "  plan $arm:  $cf"
    return 0
}
echo "== CPU: the flag canary on all three arms (no device, --sizes-only)"
for arm in base new-off new-on; do
    plan_lines "$arm" || fail "flag canary failed on the $arm arm — nothing below would be an A/B"
done
echo "  CANARY ok: base has no fold, new-off is inert, new-on folds $FOLDED of $PLAN cast launches"

# ------------------------------------------------------------------ numerics, before any ms ------
echo "== numerics on identical fixtures, three arms (fixture list: $GATE_FIXTURES)"
for fx in $GATE_FIXTURES; do
    for arm in base new-off new-on; do
        bin="$BASE_BIN"; env=""
        [ "$arm" = new-off ] && bin="$NEW_BIN"
        [ "$arm" = new-on  ] && { bin="$NEW_BIN"; env="QWEN_CAST_FOLD=1"; }
        if env $env $VENV engine/gate.py --binary "$bin" --fixture "$fx" \
                --work "$OUT/gate_${arm}_${fx}" > "$OUT/gate_${arm}_${fx}.log" 2>&1; then
            echo "  gate $arm $fx: $(grep -E '^ENGINE GATE' "$OUT/gate_${arm}_${fx}.log" | tail -1)"
        else
            echo "  gate $arm $fx: $(grep -E '^ENGINE GATE' "$OUT/gate_${arm}_${fx}.log" | tail -1)"
            fail "engine/gate.py $arm $fx went red — a numerics red outranks this timing window entirely"
        fi
    done
done

# Byte claim on the tokens themselves, same prompt, same step count, at BOTH widths. Two comparisons per
# width: base vs new-on is the fold's decision-neutrality claim, new-off vs base witnesses that the rebuild
# alone moved nothing, so a difference on the first pair is attributable to the flag and not to the build.
for w in 128 4032; do
    if [ "$w" = 128 ]; then ids="$OUT/ids128.csv"; steps=16; else ids="$OUT/ids4032.csv"; steps=32; fi
    [ -f "$ids" ] || $VENV prototype/attn_decode_wide/mkids_d3.py "$OUT" > "$OUT/mkids.log" 2>&1 \
        || fail "mkids_d3.py failed: $OUT/mkids.log"
    ntok=$(want_n "$ids")
    for arm in base new-off new-on; do
        bin="$BASE_BIN"; env=""
        [ "$arm" = new-off ] && bin="$NEW_BIN"
        [ "$arm" = new-on  ] && { bin="$NEW_BIN"; env="QWEN_CAST_FOLD=1"; }
        od="$OUT/byte_${w}_$arm"; mkdir -p "$od"
        env $env "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$ids" --steps "$steps" --out "$od" \
            > "$OUT/byte_${w}_$arm.log" 2>&1 || fail "byte-identity run ($arm, $w lanes) failed"
    done
    for pair in "base new-on" "base new-off"; do
        set -- $pair
        if diff -q "$OUT/byte_${w}_$1/sampled.txt" "$OUT/byte_${w}_$2/sampled.txt" >/dev/null 2>&1; then
            echo "  BYTE-IDENTICAL sampled.txt ($1 vs $2) at $w lanes: $ntok prompt + $steps steps, $(wc -l < "$OUT/byte_${w}_$2/sampled.txt") lines, last step $((ntok + steps)) lanes"
        else
            fail "sampled.txt differs between $1 and $2 at $w lanes — the fold is not decision-neutral"
        fi
    done
done
NUM_OK=1

# ----------------------------------------------------------------------------------- timing ------
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
echo "== lock held: $(tools/gpu_lock.sh status 2>&1 | head -1) ; util at acquire: $(tools/gpu_lock.sh util 2>&1 | tail -1)"

run() {   # run <arm> <idsfile> <steps> -> "<burst_ms> <stage_total_ms> <attn_ms>", or fails a canary
    local arm="$1" ids="$2" steps="$3"
    local bin="$BASE_BIN" env="" log od
    [ "$arm" = new-on ] && { bin="$NEW_BIN"; env="QWEN_CAST_FOLD=1"; }
    [ "$arm" = new-off ] && bin="$NEW_BIN"
    log="$OUT/run_${arm}_$(basename "$ids" .csv)_${REPCUR}.log"; od="$OUT/out_${arm}_$(basename "$ids" .csv)_${REPCUR}"
    mkdir -p "$od"
    env $env "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$ids" --steps "$steps" --time-stages --out "$od" \
        > "$log" 2>&1 || { echo "CANARY-FAIL: $arm run rc=$? (see $log)" >&2; return 1; }
    # --- exercised-path canaries, on EVERY run ----------------------------------------------------
    local want_tk tk cf plan folded kept bill burst stage attn subs pm syncs
    want_tk=$(want_n "$ids")
    tk=$(grep -E '^PREFILL' "$log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
    [ "${tk:-x}" = "$want_tk" ] || { echo "CANARY-FAIL: $arm prefilled tokens=$tk, the id file holds $want_tk fields" >&2; return 1; }
    grep -q 'STAGEPARITY .*tokens_identical=1 logits_bitidentical=1' "$log" \
        || { echo "CANARY-FAIL: $arm's own replay is not reproducible — the box or the run is unwell" >&2; return 1; }
    # The engine's OWN plan self-check, on both arms. `plan_per_step` is `planned_launch_counts()` (in its
    # fold-aware form) and `launches_per_step` counts `stageclock::note` calls at run time, so agreement is
    // the instrument saying the declared bill and the actual submissions are the same 97-rows-worth of
    // launches. It is derived inside the binary and so cannot catch a stale build — the CASTFOLD checks
    // below are what catch that — but a fold that moved submissions without moving the declared bill reads
    // 0 here, and this window refuses to price such a run.
    subs=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.* launches_per_step=([0-9]+).*/\1/')
    pm=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.* plan_match=([0-9]+).*/\1/')
    syncs=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.* syncs_per_step=([0-9]+).*/\1/')
    for v in "$subs" "$pm" "$syncs"; do num "$v" || { echo "CANARY-FAIL: $arm parsed an empty or non-numeric field on the stage TOTAL line" >&2; return 1; }; done
    [ "$pm" = 1 ] || { echo "CANARY-FAIL: $arm's own plan self-check reads plan_match=$pm against $subs launches/step — the declared bill and the submissions disagree, so there is no well-defined launch count to price" >&2; return 1; }
    if [ "$arm" = base ]; then
        # HEAD prints no CASTFOLD line — it does not know the route exists — so its half of the flag canary
        # is the §07 bill line plus the presence of the widenings themselves.
        grep -q '^CASTFOLD ' "$log" \
            && { echo "CANARY-FAIL: base printed a CASTFOLD line — BASE IS NOT HEAD" >&2; return 1; }
        grep -q 'STAGE gran=stage family=cast' "$log" \
            || { echo "CANARY-FAIL: base emitted no `cast` stage row — the widenings were never submitted, so the §1 row this A/B prices does not exist" >&2; return 1; }
        [ "$subs" = 340 ] || { echo "CANARY-FAIL: base submitted $subs launches a step, expected 340" >&2; return 1; }
    else
        cf=$(grep -E '^CASTFOLD ' "$log" | head -1)
        [ -n "$cf" ] || { echo "CANARY-FAIL: $arm printed no CASTFOLD line" >&2; return 1; }
        plan=$(echo "$cf" | sed -E 's/.* plan=([0-9]+).*/\1/')
        folded=$(echo "$cf" | sed -E 's/.* folded=([0-9]+).*/\1/')
        kept=$(echo "$cf" | sed -E 's/.* kept=([0-9]+).*/\1/')
        bill=$(echo "$cf" | sed -E 's/.* step_bill=([0-9]+).*/\1/')
        for v in "$plan" "$folded" "$kept" "$bill"; do num "$v" || { echo "CANARY-FAIL: unparsable CASTFOLD line: $cf" >&2; return 1; }; done
        # The declared step bill and the stage clock's plan must be the same number, printed by two
        # different lines of the same run. If they ever differ, `plan_match` is being computed off one and
        # the launcher's claim is the other, and neither is a bill.
        [ "$bill" = "$((340))" ] || [ "$bill" = "$subs" ] \
            || { echo "CANARY-FAIL: $arm's CASTFOLD step_bill=$bill disagrees with the stage clock's launches_per_step=$subs" >&2; return 1; }
        if [ "$arm" = new-on ]; then
            { [ "$folded" = "$plan" ] && [ "$kept" = 0 ] && [ "$plan" = "$CAST_PLAN" ]; } \
                || { echo "CANARY-FAIL: new-on reads folded=$folded kept=$kept of plan=$plan, expected folded=$CAST_PLAN kept=0" >&2; return 1; }
            [ "$subs" = "$((340 - CAST_PLAN))" ] \
                || { echo "CANARY-FAIL: new-on submitted $subs launches a step, expected $((340 - CAST_PLAN)) (340 plan minus $CAST_PLAN folded)" >&2; return 1; }
            grep -q 'STAGE gran=stage family=cast' "$log" \
                && { echo "CANARY-FAIL: new-on still emitted a `cast` stage row — the stage clock saw widenings the flag claims it deleted" >&2; return 1; }
            true
        else
            { [ "$folded" = 0 ] && [ "$kept" = "$plan" ] && [ "$plan" = "$CAST_PLAN" ]; } \
                || { echo "CANARY-FAIL: new-off reads folded=$folded kept=$kept of plan=$plan, expected folded=0 kept=$CAST_PLAN — the default side is not HEAD's behaviour" >&2; return 1; }
            grep -q 'STAGE gran=stage family=cast' "$log" \
                || { echo "CANARY-FAIL: new-off emitted no `cast` stage row — the default side is not HEAD's submissions" >&2; return 1; }
            [ "$subs" = 340 ] || { echo "CANARY-FAIL: $arm submitted $subs launches a step, expected 340" >&2; return 1; }
        fi
    fi
    burst=$(grep 'STAGEPASS gran=burst' "$log" | head -1 | sed -E 's/.*ms_mean=([0-9.]+).*/\1/')
    stage=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.*ms_per_step=([0-9.]+).*/\1/')
    attn=$(grep 'STAGE gran=stage family=attn' "$log" | head -1 | sed -E 's/.*ms_per_step=([0-9.]+).*/\1/')
    for v in "$burst" "$stage" "$attn"; do num "$v" || { echo "CANARY-FAIL: $arm parsed an empty or non-numeric stage figure from $log" >&2; return 1; }; done
    echo "$burst $stage $attn $subs $pm $syncs"
}

for width in 4032 128; do
    [ "$width" = 4032 ] && { ids="$OUT/ids4032.csv"; steps=32; } || { ids="$OUT/ids128.csv"; steps=16; }
    : > "$OUT/burst_$width.txt"; : > "$OUT/stage_$width.txt"; : > "$OUT/attn_$width.txt"
    for rep in $(seq "$REPS"); do
        REPCUR="$rep"
        for arm in base new-on; do
            line=$(run "$arm" "$ids" "$steps") \
                || fail "window aborted: $arm at $width lanes (CANARY-FAIL above) — NO-TIMING"
            echo "$arm $line" >> "$OUT/burst_$width.txt"
            echo "$arm $line" | awk '{print $1, $3}' >> "$OUT/stage_$width.txt"
            echo "$arm $line" | awk '{print $1, $4}' >> "$OUT/attn_$width.txt"
        done
        echo "  rep $rep/$REPS at $width lanes done"
    done
done
tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1
trap - EXIT

# ----------------------------------------------------------------------- the report ---------------
[ "$NUM_OK" = 1 ] || { echo "NO-TIMING: numerics did not pass"; exit 1; }
SCALE=$(awk -v f="$FOLDED" -v p="$PLAN" 'BEGIN{ if (p+0==0) {print "nan"} else printf "%.4f", f/p }')
for width in 4032 128; do
    [ "$width" = 4032 ] && label="4,032 lanes (4,000 prompt + 32 steps) — the pre-registered width" \
                        || label="128 lanes (112 prompt + 16 steps) — NO LINE HERE: D3/S1 §4 put short widths out of reach of this route in advance, so FLAT IS A PREDICTED PASS"
    echo
    echo "=== $width lanes — $label"
    printf "%-8s %14s %14s %12s %14s %10s %10s\n" arm burst_ms/tok spread tok_per_s stage_total_ms subs/step syncs/step
    for arm in base new-on; do
        awk -v a="$arm" '$1==a{print $2}' "$OUT/burst_$width.txt" > "$OUT/.b_$arm"
        awk -v a="$arm" '$1==a{print $3}' "$OUT/burst_$width.txt" > "$OUT/.s_$arm"
        bm=$(med "$OUT/.b_$arm"); bs=$(spread "$OUT/.b_$arm"); sm=$(med "$OUT/.s_$arm")
        sc=$(awk -v a="$arm" '$1==a{print $5; exit}' "$OUT/burst_$width.txt")
        sy=$(awk -v a="$arm" '$1==a{print $7; exit}' "$OUT/burst_$width.txt")
        num "${sc:-x}" >/dev/null 2>&1; num "${sy:-x}" >/dev/null 2>&1
        printf "%-8s %14s %14s %12s %14s %10s %10s\n" "$arm" "$bm" "$bs" \
            "$(awk -v m="$bm" 'BEGIN{printf "%.1f", 1000/m}')" "$sm" "$sc" "$sy"
    done
    echo "  stage_total_ms is NOT a prize figure and is printed for shape only: stage granularity pays one"
    echo "    instrument sync per segment, so deleting a launch deletes a sync too (syncs/step falls with"
    echo "    subs/step) and the serialized total drops mechanically. The only prize figure is burst_ms/tok,"
    echo "    which is the production shape: the whole step submitted, one sync at the logits."
    med "$OUT/.b_base" > "$OUT/.bm"; med "$OUT/.b_new-on" > "$OUT/.bn"
    awk -v b="$(cat "$OUT/.bm")" -v n="$(cat "$OUT/.bn")" -v w="$width" -v win="$WIN_MS" \
        -v kill="$KILL_MS" -v scale="$SCALE" -v folded="$FOLDED" -v plan="$PLAN" 'BEGIN{
        save = b - n
        printf "  step wall: %s -> %s ms/token, saving %.4f ms (%+.2f %%)\n", b, n, save, (n-b)/b*100
        printf "  folded %s of %s cast launches, so the lines scale by %s\n", folded, plan, scale
        if (w != 4032) { print "  128-lane row: no pre-registered verdict; report it beside the headline"; exit }
        wl = win * scale; kl = kill * scale
        printf "  pre-registered lines at 4,032, scaled: WIN >= %.4f ms saved, KILL <= %.4f ms saved (engine/generation-limits.md §7, quoted)", wl, kl, ""
        print ""
        if (save >= wl)      print "  VERDICT at 4,032: the per-launch residual is REAL — glue/copy (~116 more tiny launches) become a funded route"
        else if (save <= kl) print "  VERDICT at 4,032: launches are FREE when pipelined — the small-op tail is an instrument artefact and the route closes"
        else                 print "  VERDICT at 4,032: NEITHER (between the lines), reported as neither"
        printf "  read the family lines before accepting either: if proj/mlp/lm_head rose while cast fell to nothing, this is the staging'"'"'s cost, not the launch price (engine/results.md cast row)\n"
    }'
done
echo
echo "# each arm is a pair with its spread, never a best (engine/performance-status.md §6)."
echo "# the family stage lines, so the caveat above is checkable without opening two logs:"
for arm in base new-on; do
    echo "#   $arm, last rep at 4,032 lanes:"
    grep 'STAGE gran=stage family=' "$OUT/run_${arm}_ids4032_${REPS}.log" \
        | sed -E 's/STAGE gran=stage family=/#     /; s/ launches_per_step=/ launches=/; s/ ms_per_step=/ ms/' \
        | sed 's/ rows_per_step.*//'
done
echo "# raw per-rep logs: $OUT/run_*.log; medians: $OUT/burst_*.txt; gate logs: $OUT/gate_*.log"
echo "# canaries asserted inside this window: CPU equivalence + census, three-arm CASTFOLD/plan agreement,"
echo "#   base has no CASTFOLD line, new-off is inert, the stage clock's own cast row present on base and"
echo "#   new-off and ABSENT on new-on, stage TOTAL launches_per_step = 340 - folded, PREFILL tokens= vs"
echo "#   the id file, STAGEPARITY, gate.py per fixture, sampled.txt byte-diff at 128 and 4,032 lanes"
echo "#   against both base and new-off."
