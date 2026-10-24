#!/usr/bin/env bash
# ab_gluecopy.sh — the paired wall A/B for route S1's follow-on: delete 19 of the 68 `glue`+`copy`
# launches by folding the embedding-row D2D into the glue launch that reads the table row, and each
# GDN layer's conv-window roll-back into the conv that produces the rolled window. MAIN AGENT ONLY:
# this takes milliseconds and a worker may not (AGENTS.md, "GPU execution is serialized and belongs to
# the main agent"). It acquires tools/gpu_lock.sh itself and holds it for the whole window; run it
# BLOCKING, never backgrounded, with zero subagents active.
#
#   bash prototype/glue_copy_fold/ab_gluecopy.sh
#   REPS=5 GATE_FIXTURES="dec8 wide4096f" bash prototype/glue_copy_fold/ab_gluecopy.sh
#
# ============================================================================ SAY IT PLAINLY FIRST
# THIS ROUTE'S FULL WIN IS ABOUT 1 % OF THE STEP, NOT 2.7 %. `engine/generation-limits.md` §1 bills the
# whole `glue`+`copy` population at 0.407 + 0.060 ms of a ~10 ms step, and `prototype/glue_copy_fold/
# census_sites.py` shows why that ceiling is unreachable: 49 of the 68 are `glue` launches that CANNOT
# fold, because a fused add+norm's row norm needs the whole row and every consumer block owns one output
# row — riding it recomputes the norm once per consumer block, 64,136 blocks per token, 394 MB of extra
# reads to buy 0.152 ms, 10.9x the prize it is buying (the census prints the walk). What folds is the
# 19 `copy` submissions, 19 of 68, so the ceiling on THIS route is 19 x 3.10 us = 0.0589 ms on a
# ~10 ms step at 4,032 lanes, about 0.6 % — and the honest expectation is less. Its value is that it is
# the last per-launch population left in the decode step, not that it is large. Report it that way.
#
# ============================================================================ WHAT IS BEING TESTED
# Two sites, both data movement rather than arithmetic:
#   copy_embed_row      (1/token)  the tied table's row was D2D-copied into `sc.embed`; folded, the
#                                  layer-0 glue launch reads the table row directly.
#   copy_conv_rollback  (18/token) `conv_kernel` wrote the rolled window to a scratch row and a D2D copy
#                                  moved it into `st.conv_window_layer(layer)`; folded,
#                                  `engine/gdn_conv_roll_kernels.h::conv_roll_kernel` writes it in place
#                                  at the columns its own lane just read.
# Both are exact: same values, one address shorter, so the admission is a byte-compare and NOT a band
# (`prototype/glue_copy_fold/equiv_copy_fold.py` makes the CPU half; the fold is a one-row rule,
# `sc.copy_fold && (n == 1)`, so a prefill wave runs the kernels it always ran and TTFT is not priced).
#
# ============================================================================ THE PRE-REGISTERED READING
# Quoted from `engine/generation-limits.md` §7, on the cast-fold arm: 97 launches saved >= 0.24 ms and
# <= 0.05 ms closes the route, i.e. WIN = 0.10 ms and KILL = 0.03 ms PER 97 LAUNCHES on the 4,032-lane
# decode-step wall. Scaled to the population that actually folds, read out of THIS script's own runs
# (folded/68, 19/68 on this build):
#       WIN  >= 0.10 x 19/68 = 0.0279 ms saved on the 4,032-lane step wall
#       KILL <= 0.03 x 19/68 = 0.0084 ms saved on the 4,032-lane step wall
#       between                -> neither, reported as neither
# The scaling is computed from the `folded=` field the script parses from its own `COPYFOLD` line, never
# typed, so if a later refactor folds fewer sites the lines move with it. At 128 lanes there is NO line:
# a flat per-launch saving is a smaller fraction of a 4k step than of a 128 one
# (`engine/generation-limits.md`, "Decode is measured and reported at 4k context"), and a short-width row
# is reported beside the headline, never instead of it.
#
# ============================================================================ THE CANARIES
# `engine/build.sh` is not bit-reproducible on this box, so `cmp` between two builds proves only that they
# are two builds, and any self-check compiled into the binary under test is blind to the one failure mode
# that matters (AGENTS.md 2026-10-17: a D30 A/B timed a pre-change binary as the `new` side and reported a
# clean null while every in-binary parity field stayed green). So these are asserted inside the window,
# before any millisecond is spent:
#
#   1. THE FLAG TOOK EFFECT, read two ways. `COPYFOLD on=... plan=... folded=... kept=... step_bill=...`
#      is printed by the new binary and by NO base build — a stale "new" binary looks exactly like base
#      and is caught here. `plan=` must equal the `other=` field of the untouched `LAUNCH` line minus the
#      lm_head launch, which is a second, independent derivation of the same 19.
#   2. THE FOLD FAMILY DISAPPEARS FROM THE STAGE CLOCK. On base and new-off a
#      `STAGE gran=stage family=copy` row exists; on new-on it must be ABSENT, because `stageclock::note(
#      kCopy)` is called 19 times per step shipped and zero times folded. A build where the flag prints
#      the line but the clock still sees copies is a fold that did not happen.
#   3. THE DECLARED BILL AND THE COUNTED SUBMISSIONS AGREE, and across the flag they must differ by
#      exactly the folded count: base/new-off `launches_per_step=340`, new-on 321. `plan_match=1` alone
#      cannot show this (a stale binary agrees with itself), so both the value and the difference are
#      asserted, and `step_bill=` from the COPYFOLD line must equal the stage clock's number.
#   4. THE PATH WAS EXERCISED: `PREFILL tokens=` equals the field count of the id file the run was handed
#      (a space-joined fixture reads as ONE token and the run then succeeds, AGENTS.md), the fold is a
#      one-row rule so the byte-identity runs use one-row bodies at BOTH widths, and `STAGEPARITY
#      tokens_identical=1 logits_bitidentical=1` says the run reproduced itself.
#   5. BYTE-IDENTITY WHERE EXACT. This fold is exact, so `sampled.txt` AND the state dump must match base
#      byte for byte at every width tested — and the state dump matters more here than anywhere else in
#      this route family, because the thing the fold rewrites in place IS state. A re-associated or
#      lagged window would move every later token's conv, so if the width is beyond where byte-identity
#      is claimed, the decision-level fields (argmax, top-5, whole greedy continuation) are printed and
#      the byte claim is stated with its width, never dropped (`AGENTS.md`: byte-identity on short
#      fixtures is not a witness for long context).
#
# Any canary failure prints NO-TIMING and exits before a millisecond is spent.
#
# ============================================================================ what this script does NOT do
# It does not touch `engine/performance-status.md` or any ledger: a number from this window enters a
# timing doc only through the main agent. `REPS` defaults to 3 and both arms are paired inside ONE lock
# hold, because a ruler re-taken in a different window is not an A/B.

set -u
cd "$(dirname "$0")/../.."
export LC_ALL=C          # the box's locale uses a comma decimal separator; sort -n and awk's strtod
                         # both truncate at the '.' and the medians would be computed off integers.
VENV=${VENV:-.venv/bin/python}
OUT=${OUT:-tmp/glue_copy_fold/ab}
TAG=${TAG:-s1-gluecopy-ab}
REPS=${REPS:-3}                       # reps per arm, per width (>= 3, pre-registered)
GATE_FIXTURES=${GATE_FIXTURES:-dec8 wide4096f}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
BASE_BIN="$OUT/base_decode_run"; NEW_BIN="$OUT/new_decode_run"
BASE_WT="$OUT/base"
WIN_MS=${WIN_MS:-0.10}                # quoted: engine/generation-limits.md §7, on the 4,032-lane step
KILL_MS=${KILL_MS:-0.03}              # quoted: same
POPULATION=${POPULATION:-68}          # quoted: §1's glue 49 + copy 19; census_sites.py re-derives it
COPY_PLAN=${COPY_PLAN:-19}            # quoted: §1's copy row, re-derived by census_sites.py
STEP_PLAN=${STEP_PLAN:-340}           # quoted: §1's total step bill
mkdir -p "$OUT"
# OUT must be ABSOLUTE: the base build runs `( cd "$BASE_WT" && ... > "$OUT/build_base.log" )`, and a
# relative OUT cannot resolve from inside the worktree, which reads as "the base build failed".
OUT=$(cd "$OUT" && pwd) || { echo "ABORT: cannot resolve OUT" >&2; exit 1; }
NUM_OK=0
FOLDED=0; PLAN=0

fail() { echo "ABORT: $*" >&2; echo "NO-TIMING: no millisecond figure is printed from this run" >&2;
         tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
med() { sort -n "$1" | awk '{a[NR]=$1} END{ if (NR==0) {print "nan"; exit}
        m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "%.4f", m }'; }
spread() { sort -n "$1" | awk '{if(m==""||$1<m)m=$1; if($1>x)x=$1} END{printf "%s..%s", m, x}'; }
want_n() { awk -F',' '{print NF}' "$1"; }
num() { case "$1" in ''|*[!0-9.]*) return 1;; *) return 0;; esac; }
arm_bin() { case "$1" in base) echo "$BASE_BIN";; *) echo "$NEW_BIN";; esac; }
arm_env() { case "$1" in new-on) echo "QWEN_COPY_FOLD=1";; *) echo "";; esac; }

# ------------------------------------------------------------------ CPU work, before the lock ----
echo "== CPU: census + equivalence admission (no device)"
$VENV prototype/glue_copy_fold/census_sites.py > "$OUT/census.log" 2>&1 \
    || fail "prototype/glue_copy_fold/census_sites.py is red — the population is not what this script prices: see $OUT/census.log"
grep -E "^SITES ok" "$OUT/census.log" | sed 's/^/   /'
grep -E "^FOLDED" "$OUT/census.log" | sed 's/^/   /'
$VENV prototype/glue_copy_fold/census_sites.py --selftest > "$OUT/census_selftest.log" 2>&1 \
    || fail "the census's own selftest is red: see $OUT/census_selftest.log"
$VENV prototype/glue_copy_fold/equiv_copy_fold.py > "$OUT/equiv.log" 2>&1 \
    || fail "prototype/glue_copy_fold/equiv_copy_fold.py is red — do not price a fold that is not proven: see $OUT/equiv.log"
$VENV prototype/glue_copy_fold/equiv_copy_fold.py --selftest > "$OUT/equiv_selftest.log" 2>&1 \
    || fail "the equivalence checker's own selftest is red: see $OUT/equiv_selftest.log"
grep -E "^EQUIV ok" "$OUT/equiv.log" | sed 's/^/   /'

echo "== CPU: building NEW from the working tree"
bash engine/build.sh > "$OUT/build_new.log" 2>&1 || fail "engine/build.sh (new) failed: $OUT/build_new.log"
cp engine/build/decode_run "$NEW_BIN" || fail "no $NEW_BIN"
if [ -x "$BASE_BIN" ] && [ "${REBUILD_BASE:-0}" != 1 ]; then
    echo "== CPU: reusing $BASE_BIN from an earlier run (REBUILD_BASE=1 forces a fresh HEAD build)"
else
    echo "== CPU: building BASE from HEAD in a scratch worktree (never from the touched working tree)"
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
plan_lines() {   # plan_lines <arm> -> the log it printed; returns 1 if the canary fields are wrong
    local arm="$1" log="$OUT/plan_$1.log" bin env
    case "$arm" in
        base) bin="$BASE_BIN"; env="" ;;
        new-off) bin="$NEW_BIN"; env="" ;;
        new-on) bin="$NEW_BIN"; env="QWEN_COPY_FOLD=1" ;;
        *) fail "plan_lines: unknown arm $arm" ;;
    esac
    env $env "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --sizes-only > "$log" 2>&1 \
        || { echo "  plan $arm: run failed (see $log)" >&2; return 1; }
    local launch_other copy_plan
    launch_other=$(grep -E '^LAUNCH ' "$log" | head -1 | sed -E 's/.* other=([0-9]+).*/\1/')
    num "${launch_other:-x}" || { echo "  plan $arm: no other= on the LAUNCH line" >&2; return 1; }
    copy_plan=$((launch_other - 1))      # `other` is the 19 copies plus the lm_head launch
    [ "$copy_plan" = "$COPY_PLAN" ] \
        || { echo "  plan $arm: LAUNCH bills other=$launch_other, so the copy row is $copy_plan and §1 says $COPY_PLAN" >&2; return 1; }
    if [ "$arm" = base ]; then
        grep -q '^COPYFOLD ' "$log" \
            && { echo "  plan base: a base build printed a COPYFOLD line — BASE IS NOT HEAD, or NEW is what you built into it" >&2; return 1;
                 true; }
        echo "  plan base:    no COPYFOLD line, LAUNCH other=$launch_other (expected: base does not know the fold)"
        return 0
    fi
    local cf on plan folded kept
    cf=$(grep -E '^COPYFOLD ' "$log" | head -1)
    [ -n "$cf" ] || { echo "  plan $arm: the new binary printed no COPYFOLD line — THE FOLD IS NOT IN THIS BINARY" >&2; return 1; }
    on=$(echo "$cf" | sed -E 's/.* on=([01]+).*/\1/');     plan=$(echo "$cf" | sed -E 's/.* plan=([0-9]+).*/\1/')
    folded=$(echo "$cf" | sed -E 's/.* folded=([0-9]+).*/\1/'); kept=$(echo "$cf" | sed -E 's/.* kept=([0-9]+).*/\1/')
    for v in "$on" "$plan" "$folded" "$kept"; do num "$v" || { echo "  plan $arm: unparsable COPYFOLD line: $cf" >&2; return 1; }; done
    [ "$plan" = "$copy_plan" ] \
        || { echo "  plan $arm: COPYFOLD plan=$plan disagrees with LAUNCH other=$launch_other minus lm_head" >&2; return 1; }
    if [ "$arm" = new-on ]; then
        { [ "$on" = 1 ] && [ "$folded" = "$plan" ] && [ "$kept" = 0 ]; } \
            || { echo "  plan $arm: QWEN_COPY_FOLD=1 but COPYFOLD reads on=$on folded=$folded kept=$kept — the env did not reach the engine, or the fold does not cover the plan" >&2; return 1; }
        FOLDED="$folded"; PLAN="$plan"
    else
        { [ "$on" = 0 ] && [ "$folded" = 0 ]; } \
            || { echo "  plan $arm: no env but COPYFOLD reads on=$on folded=$folded — the default side is not inert" >&2; return 1; }
    fi
    echo "  plan $arm:  $cf"
    return 0
}
echo "== CPU: the flag canary on all three arms (no device, --sizes-only)"
for arm in base new-off new-on; do
    plan_lines "$arm" || fail "flag canary failed on the $arm arm — nothing below would be an A/B"
done
echo "  CANARY ok: base has no fold, new-off is inert, new-on folds $FOLDED of $PLAN copy launches"

# ------------------------------------------------------------------ numerics, before any ms ------
echo "== numerics on identical fixtures, three arms (fixture list: $GATE_FIXTURES)"
for fx in $GATE_FIXTURES; do
    for arm in base new-off new-on; do
        bin=$(arm_bin "$arm"); env=$(arm_env "$arm")
        if env $env $VENV engine/gate.py --binary "$bin" --fixture "$fx" \
                --work "$OUT/gate_${arm}_${fx}" > "$OUT/gate_${arm}_${fx}.log" 2>&1; then
            echo "  gate $arm $fx: $(grep -E '^ENGINE GATE' "$OUT/gate_${arm}_${fx}.log" | tail -1)"
        else
            echo "  gate $arm $fx: $(grep -E '^ENGINE GATE' "$OUT/gate_${arm}_${fx}.log" | tail -1)"
            fail "engine/gate.py $arm $fx went red — a numerics red outranks this timing window entirely"
        fi
    done
done

# Byte claim on the tokens AND the state, same prompt, same step count, at both widths. Two comparisons
# per width: base vs new-on is the fold's exactness claim, new-off vs base witnesses that the rebuild
# alone moved nothing, so a difference on the first pair is attributable to the flag and not to the
# build. The state dump is compared alongside `sampled.txt` because on THIS route the folded write
# targets state: a window written one row late or one row wrong is invisible in the first token's logits
# and decisive in the tenth.
for w in 128 4032; do
    if [ "$w" = 128 ]; then ids="$OUT/ids128.csv"; steps=16; else ids="$OUT/ids4032.csv"; steps=32; fi
    [ -f "$ids" ] || $VENV prototype/attn_decode_wide/mkids_d3.py "$OUT" > "$OUT/mkids.log" 2>&1 \
        || fail "mkids_d3.py failed: $OUT/mkids.log"
    ntok=$(want_n "$ids")
    for arm in base new-off new-on; do
        bin=$(arm_bin "$arm"); env=$(arm_env "$arm")
        od="$OUT/byte_${w}_$arm"; mkdir -p "$od"
        env $env "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$ids" --steps "$steps" --state-fp \
            --out "$od" > "$OUT/byte_${w}_$arm.log" 2>&1 \
            || fail "byte-identity run ($arm, $w lanes) failed"
    done
    for pair in "base new-on" "base new-off"; do
        set -- $pair
        if diff -q "$OUT/byte_${w}_$1/sampled.txt" "$OUT/byte_${w}_$2/sampled.txt" >/dev/null 2>&1; then
            echo "  BYTE-IDENTICAL sampled.txt ($1 vs $2) at $w lanes: $ntok prompt + $steps steps, $(wc -l < "$OUT/byte_${w}_$2/sampled.txt") lines, last step $((ntok + steps)) lanes"
        else
            fail "sampled.txt differs between $1 and $2 at $w lanes — the fold is not exact at $w; state the width the byte claim covers and judge decisions separately, do not loosen this"
        fi
        # The state fingerprints, which on THIS route are the half that matters: `STATEFP family=conv`
        # is the conv window, and the fold rewrites that buffer in place. A window written one row late,
        # or one row wrong, can leave the first token's logits untouched and is decisive by the tenth, so
        # comparing logits alone would be a witness for the wrong thing.
        grep '^STATEFP ' "$OUT/byte_${w}_$1.log" > "$OUT/.fp_$1"; grep '^STATEFP ' "$OUT/byte_${w}_$2.log" > "$OUT/.fp_$2"
        fpn=$(wc -l < "$OUT/.fp_$2")
        [ "$fpn" -gt 0 ] || fail "($1 vs $2) at $w lanes printed no STATEFP rows — the state was never fingerprinted, so there is no witness for the folded write"
        if diff -q "$OUT/.fp_$1" "$OUT/.fp_$2" >/dev/null 2>&1; then
            echo "  BYTE-IDENTICAL STATEFP ($1 vs $2) at $w lanes: $fpn fingerprint rows, conv windows included"
        else
            fail "STATEFP differs between $1 and $2 at $w lanes — $(diff "$OUT/.fp_$1" "$OUT/.fp_$2" | head -3 | tr '\n' ' ')"
        fi
    done
done
NUM_OK=1

# ----------------------------------------------------------------------------------- timing ------
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
echo "== lock held: $(tools/gpu_lock.sh status 2>&1 | head -1) ; util at acquire: $(tools/gpu_lock.sh util 2>&1 | tail -1)"

run() {   # run <arm> <idsfile> <steps> -> "<burst_ms> <stage_total_ms> <copy_ms> <subs> <pm> <syncs>"
    local arm="$1" ids="$2" steps="$3"
    local bin env log od
    bin=$(arm_bin "$arm"); env=$(arm_env "$arm")
    log="$OUT/run_${arm}_$(basename "$ids" .csv)_${REPCUR}.log"
    od="$OUT/out_${arm}_$(basename "$ids" .csv)_${REPCUR}"; mkdir -p "$od"
    env $env "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$ids" --steps "$steps" --time-stages --out "$od" \
        > "$log" 2>&1 || { echo "CANARY-FAIL: $arm run rc=$? (see $log)" >&2; return 1; }
    # --- exercised-path canaries, on EVERY run ----------------------------------------------------
    local want_tk tk cf plan folded kept bill burst stage copysub subs pm syncs
    want_tk=$(want_n "$ids")
    tk=$(grep -E '^PREFILL' "$log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
    [ "${tk:-x}" = "$want_tk" ] || { echo "CANARY-FAIL: $arm prefilled tokens=$tk, the id file holds $want_tk fields (a space-joined fixture reads as ONE token)" >&2; return 1; }
    grep -q 'STAGEPARITY .*tokens_identical=1 logits_bitidentical=1' "$log" \
        || { echo "CANARY-FAIL: $arm's own replay is not reproducible — the box or the run is unwell" >&2; return 1; }
    subs=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.* launches_per_step=([0-9]+).*/\1/')
    pm=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.* plan_match=([0-9]+).*/\1/')
    syncs=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.* syncs_per_step=([0-9]+).*/\1/')
    for v in "$subs" "$pm" "$syncs"; do num "$v" || { echo "CANARY-FAIL: $arm parsed an empty or non-numeric field on the stage TOTAL line" >&2; return 1; }; done
    [ "$pm" = 1 ] || { echo "CANARY-FAIL: $arm's own plan self-check reads plan_match=$pm against $subs launches/step — the declared bill and the submissions disagree, so there is no well-defined launch count to price" >&2; return 1; }
    if [ "$arm" = base ]; then
        # HEAD prints no COPYFOLD line — it does not know the route exists — so its half of the flag
        # canary is the §07 bill line plus the presence of the copies themselves.
        grep -q '^COPYFOLD ' "$log" \
            && { echo "CANARY-FAIL: base printed a COPYFOLD line — BASE IS NOT HEAD" >&2; return 1; }
        grep -q 'STAGE gran=stage family=copy' "$log" \
            || { echo "CANARY-FAIL: base emitted no `copy` stage row — the D2D copies this A/B prices were never submitted" >&2; return 1; }
        [ "$subs" = "$STEP_PLAN" ] || { echo "CANARY-FAIL: base submitted $subs launches a step, expected $STEP_PLAN" >&2; return 1; }
    else
        cf=$(grep -E '^COPYFOLD ' "$log" | head -1)
        [ -n "$cf" ] || { echo "CANARY-FAIL: $arm printed no COPYFOLD line" >&2; return 1; }
        plan=$(echo "$cf" | sed -E 's/.* plan=([0-9]+).*/\1/')
        folded=$(echo "$cf" | sed -E 's/.* folded=([0-9]+).*/\1/')
        kept=$(echo "$cf" | sed -E 's/.* kept=([0-9]+).*/\1/')
        bill=$(echo "$cf" | sed -E 's/.* step_bill=([0-9]+).*/\1/')
        for v in "$plan" "$folded" "$kept" "$bill"; do num "$v" || { echo "CANARY-FAIL: unparsable COPYFOLD line: $cf" >&2; return 1; }; done
        # The declared step bill and the stage clock's plan must be the same number, printed by two
        # different lines of the same run. If they ever differ, `plan_match` is computed off one and the
        # launcher's claim is the other, and neither is a bill.
        [ "$bill" = "$subs" ] \
            || { echo "CANARY-FAIL: $arm's COPYFOLD step_bill=$bill disagrees with the stage clock's launches_per_step=$subs" >&2; return 1; }
        if [ "$arm" = new-on ]; then
            { [ "$folded" = "$plan" ] && [ "$kept" = 0 ] && [ "$plan" = "$COPY_PLAN" ]; } \
                || { echo "CANARY-FAIL: new-on reads folded=$folded kept=$kept of plan=$plan, expected folded=$COPY_PLAN kept=0" >&2; return 1; }
            [ "$subs" = "$((STEP_PLAN - COPY_PLAN))" ] \
                || { echo "CANARY-FAIL: new-on submitted $subs launches a step, expected $((STEP_PLAN - COPY_PLAN)) ($STEP_PLAN plan minus $COPY_PLAN folded)" >&2; return 1; }
            grep -q 'STAGE gran=stage family=copy' "$log" \
                && { echo "CANARY-FAIL: new-on still emitted a `copy` stage row — the stage clock saw D2D submissions the flag claims it deleted, so the fold did not happen" >&2; return 1; }
            true
        else
            { [ "$folded" = 0 ] && [ "$kept" = "$plan" ] && [ "$plan" = "$COPY_PLAN" ]; } \
                || { echo "CANARY-FAIL: new-off reads folded=$folded kept=$kept of plan=$plan, expected folded=0 kept=$COPY_PLAN — the default side is not HEAD's behaviour" >&2; return 1; }
            grep -q 'STAGE gran=stage family=copy' "$log" \
                || { echo "CANARY-FAIL: new-off emitted no `copy` stage row — the default side is not HEAD's submissions" >&2; return 1; }
            [ "$subs" = "$STEP_PLAN" ] || { echo "CANARY-FAIL: $arm submitted $subs launches a step, expected $STEP_PLAN" >&2; return 1; }
        fi
    fi
    burst=$(grep 'STAGEPASS gran=burst' "$log" | head -1 | sed -E 's/.*ms_mean=([0-9.]+).*/\1/')
    stage=$(grep 'STAGE gran=stage TOTAL' "$log" | head -1 | sed -E 's/.*ms_per_step=([0-9.]+).*/\1/')
    copysub=$(grep 'STAGE gran=stage family=copy' "$log" | head -1 | sed -E 's/.*ms_per_step=([0-9.]+).*/\1/')
    if [ "$arm" = new-on ]; then
        # The folded arm has no `copy` row by construction — that is canary 2 — so the field is 0.0 here
        # rather than unparsable, and the report says so instead of printing a blank.
        copysub=0.0
    else
        num "$copysub" || { echo "CANARY-FAIL: $arm parsed an empty or non-numeric copy figure" >&2; return 1; }
    fi
    for v in "$burst" "$stage"; do num "$v" || { echo "CANARY-FAIL: $arm parsed an empty or non-numeric stage figure from $log" >&2; return 1; }; done
    echo "$burst $stage $copysub $subs $pm $syncs"
}

for width in 4032 128; do
    [ "$width" = 4032 ] && { ids="$OUT/ids4032.csv"; steps=32; } || { ids="$OUT/ids128.csv"; steps=16; }
    : > "$OUT/burst_$width.txt"; : > "$OUT/stage_$width.txt"
    for rep in $(seq "$REPS"); do
        REPCUR="$rep"
        for arm in base new-on; do
            line=$(run "$arm" "$ids" "$steps") \
                || fail "window aborted: $arm at $width lanes (CANARY-FAIL above) — NO-TIMING"
            echo "$arm $line" >> "$OUT/burst_$width.txt"
            echo "$arm $line" | awk '{print $1, $3}' >> "$OUT/stage_$width.txt"
        done
        echo "  rep $rep/$REPS at $width lanes done"
    done
done
tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1
trap - EXIT

# ----------------------------------------------------------------------- the report ---------------
[ "$NUM_OK" = 1 ] || { echo "NO-TIMING: numerics did not pass"; exit 1; }
SCALE=$(awk -v f="$FOLDED" -v p="$POPULATION" 'BEGIN{ if (p+0==0) {print "nan"} else printf "%.4f", f/p }')
for width in 4032 128; do
    [ "$width" = 4032 ] && label="4,032 lanes (4,000 prompt + 32 steps) — the pre-registered width" \
                        || label="128 lanes (112 prompt + 16 steps) — NO LINE HERE: a flat per-launch saving is a smaller fraction of a 4k step than of a 128 one; report beside the headline, never instead"
    echo
    echo "=== $width lanes — $label"
    printf "%-8s %14s %14s %12s %14s %10s %10s %10s\n" arm burst_ms/tok spread tok_per_s stage_total_ms copy_ms subs/step syncs/step
    for arm in base new-on; do
        awk -v a="$arm" '$1==a{print $2}' "$OUT/burst_$width.txt" > "$OUT/.b_$arm"
        awk -v a="$arm" '$1==a{print $3}' "$OUT/burst_$width.txt" > "$OUT/.s_$arm"
        bm=$(med "$OUT/.b_$arm"); bs=$(spread "$OUT/.b_$arm"); sm=$(med "$OUT/.s_$arm")
        cm=$(awk -v a="$arm" '$1==a{print $4; exit}' "$OUT/burst_$width.txt")
        sc=$(awk -v a="$arm" '$1==a{print $5; exit}' "$OUT/burst_$width.txt")
        sy=$(awk -v a="$arm" '$1==a{print $7; exit}' "$OUT/burst_$width.txt")
        num "${sc:-x}" >/dev/null 2>&1; num "${sy:-x}" >/dev/null 2>&1
        printf "%-8s %14s %14s %12s %14s %10s %10s %10s\n" "$arm" "$bm" "$bs" \
            "$(awk -v m="$bm" 'BEGIN{printf "%.1f", 1000/m}')" "$sm" \
            "$( [ "$arm" = new-on ] && echo "0 (folded)" || echo "$cm" )" "$sc" "$sy"
    done
    echo "  stage_total_ms is NOT a prize figure and is printed for shape only: stage granularity pays one"
    echo "    instrument sync per segment, so deleting a launch deletes a sync too (syncs/step falls with"
    echo "    subs/step) and the serialized total drops mechanically. The only prize figure is burst_ms/tok,"
    echo "    which is the production shape: the whole step submitted, one sync at the logits."
    med "$OUT/.b_base" > "$OUT/.bm"; med "$OUT/.b_new-on" > "$OUT/.bn"
    awk -v b="$(cat "$OUT/.bm")" -v n="$(cat "$OUT/.bn")" -v w="$width" -v win="$WIN_MS" \
        -v kill="$KILL_MS" -v scale="$SCALE" -v folded="$FOLDED" -v pop="$POPULATION" 'BEGIN{
        save = b - n
        printf "  step wall: %s -> %s ms/token, saving %.4f ms (%+.2f %%)\n", b, n, save, (n-b)/b*100
        printf "  folded %s of %s launches in this population; the other 49 are glue launches that cannot ride a consumer (the census prices them at 394 MB of extra reads per token against a 0.152 ms prize), so the lines scale by folded/68 = %s\n", folded, pop, scale
        if (w != 4032) { print "  128-lane row: no pre-registered verdict; report it beside the headline"; exit }
        wl = win * scale; kl = kill * scale
        printf "  pre-registered lines at 4,032, scaled: WIN >= %.4f ms saved, KILL <= %.4f ms saved (engine/generation-limits.md §7, quoted, scaled by 19/68)\n", wl, kl
        if (save >= wl)      print "  VERDICT at 4,032: the per-launch residual survives into the copy population — the fold pays, and the remaining question is whether any of the 49 glue launches can be made to ride a consumer at all"
        else if (save <= kl) print "  VERDICT at 4,032: launches are free when pipelined, here too — the route closes on its own number and the small-op tail stays an instrument artefact"
        else                 print "  VERDICT at 4,032: NEITHER (between the lines), reported as neither"
        printf "  and the size to keep in the sentence: at most %s x 3.10 us = %.4f ms, about 1 %% of a %.1f ms step, not 2.7 %%\n", folded, folded * 3.10 / 1000.0, b
    }'
done
echo
echo "# each arm is a pair with its spread, never a best (engine/performance-status.md §6)."
echo "# the family stage lines, so the copy row's disappearance is checkable without opening two logs:"
for arm in base new-on; do
    echo "#   $arm, last rep at 4,032 lanes:"
    grep 'STAGE gran=stage family=' "$OUT/run_${arm}_ids4032_${REPS}.log" \
        | sed -E 's/STAGE gran=stage family=/#     /; s/ launches_per_step=/ launches=/; s/ ms_per_step=/ ms/' \
        | sed 's/ rows_per_step.*//'
done
echo "# raw per-rep logs: $OUT/run_*.log; medians: $OUT/burst_*.txt; gate logs: $OUT/gate_*.log"
echo "# canaries asserted inside this window: CPU census + census selftest + equivalence + equivalence"
echo "#   selftest, three-arm COPYFOLD/LAUNCH agreement, base has no COPYFOLD line, new-off is inert, the"
echo "#   stage clock's own copy row present on base and new-off and ABSENT on new-on, stage TOTAL"
echo "#   launches_per_step = 340 - 19 on the folded arm and step_bill equal to it, PREFILL tokens= vs the"
echo "#   id file, STAGEPARITY, gate.py per fixture, and a recursive byte-diff of sampled.txt AND the state"
echo "#   dumps at 128 and 4,032 lanes against both base and new-off."
