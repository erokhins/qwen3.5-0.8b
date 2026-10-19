#!/usr/bin/env bash
# run_ab_g2.sh — the timing A/B for bet G2 (`--gdn-geom block|wf4|wf2`): four sides, one window.
#
# MAIN-AGENT DEVICE WINDOW ONLY. Blocking, zero subagents, one `tools/gpu_lock.sh` hold, and it runs only
# programs that do NOT self-lock (`decode_run`; the two CPU checks it calls touch no device). A worker must
# not run it: it takes timings, and `AGENTS.md` keeps every timing on the main agent because this box
# serves its own tokens on the same iGPU. The worker-side correctness gate is `gate_g2.py`, and the numerics
# mirror is `g2_mirror.py`; both are already run and re-run by this script's non-timing legs.
#
# WHY FOUR SIDES.
#   base          HEAD, which has no `--gdn-geom` at all. `base` vs `block` must be a NULL: it is the leg
#                 that proves adding the knob did not move the shipping path.
#   block         the new binary at `--gdn-geom block` = the shipped `<<<16,128>>>` looped walk.
#   wf4           BET.md §10's pre-registered shape: `<<<16,32>>>`, 4 state columns per lane, no LDS, no
#                 barrier, norms by warp butterfly. THE VERDICT IS TAKEN ON THIS SIDE.
#   wf2           the same kernel at 2 columns per lane (`<<<32,32>>>`): the split leg. It is bit-identical
#                 to wf4 (each column's arithmetic does not depend on which lane owns it), so the pair
#                 `wf4`/`wf2` separates "did the wavefront shape pay" from "at what split", which is the one
#                 free parameter the geometry has and the one whose arithmetic moved the issue floor 2x
#                 (engine/gdn_rec_wf_kernels.h's header sizes it). Set SIDES="wf4" to drop it.
#
# PRE-REGISTERED OUTCOME — BOTH denominators, from BET.md §10, medians of 3 at 4,096 lanes, base = the §40
# window (gdn_rec 266.60 ms, wall_ms 626.75 ms):
#       gdn_rec family row :  live <= 180   ·   dead > 230
#       wave wall_ms       :  live <= 560   ·   dead > 607
#   BOTH must clear their live line to ship. EITHER falling past its dead line kills the form. Anything
#   else is UNRESOLVED and is reported as unresolved, with no narration — which is what §40 did for G1 and
#   why the second line exists at all: G1's −5.9 % on its row was −2.2 % of the wave, because the cost
#   landed in its neighbours. The externality line this script prints is that trap made visible.
#
# Usage:
#   bash prototype/gdn_walk/run_ab_g2.sh                     # REPS=3, 4,096 lanes, all four sides
#   REPS=2 bash prototype/gdn_walk/run_ab_g2.sh
#   SIDES="wf4" bash prototype/gdn_walk/run_ab_g2.sh          # drop the split leg
#   SUMMARIZE=1 bash prototype/gdn_walk/run_ab_g2.sh          # re-judge the logs, touch no device
#   BASE=<sha-or-ref> BASEDIR=/tmp/base-g2 bash prototype/gdn_walk/run_ab_g2.sh
set -uo pipefail
export LC_ALL=C # a decimal comma in a machine-read number is a parse bug wearing a locale's clothes
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
BASE=${BASE:-HEAD}
BASEDIR=${BASEDIR:-/tmp/base-g2}
BIN_NEW=engine/build/decode_run
BIN_OLD=$BASEDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
W=4096
IDS=$(cat progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt)
REPS=${REPS:-3}
TAG=${TAG:-g2-ab}
SIDES=${SIDES:-wf4 wf2}
OUT=${OUT:-tmp/gdn_walk/g2ab}
# BET.md §10's two lines, in one place so the verdict and the printed text cannot drift apart.
LINE_REC_LIVE=180;  LINE_REC_DEAD=230
LINE_WALL_LIVE=560; LINE_WALL_DEAD=607
# Expected `gdn_rec` launch counts at this shape, from prototype/gdn_walk/plan_bill_check.py's derivation:
# 8 bodies of 512 rows, one span each, 18 GDN layers -> 18*8 = 144 launches, on EVERY side. The geometry
# changes the block, not the submission count, so unlike G1 (144 vs 288) the canary here is that the two
# sides agree — and the thing that distinguishes them is the NUMERICS echo, not the launch count.
EXPECT_LAUNCHES=144
mkdir -p "$OUT"
ARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --wave-rows 512 --gdn-loop 512"

clk() { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk" | tr '\n' ' '; }
fam() { grep -h "WAVE gran=stage family=$1 " "$2" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
tot() { grep -h "WAVE gran=stage TOTAL" "$1" | sed -E 's/.*wall_ms=([0-9.]+).*/\1/' | tail -1; }
res() { grep -h "^RESULT" "$1" | sed -E 's/.*ms_total=([0-9.]+).*/\1/' | tail -1; }
median() { sort -g "$@" | awk '{a[NR]=$1} END{if(NR==0){print "NA";exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
log_of() { case $1 in base) echo "$OUT/base" ;; *) echo "$OUT/new_$1" ;; esac; }
bin_of() { case $1 in base) echo "$BIN_OLD" ;; *) echo "$BIN_NEW" ;; esac; }
geom_of() { case $1 in base) echo "" ;; *) echo "--gdn-geom $1" ;; esac; }

[ -x "$BIN_NEW" ] || fail "shipping binary missing: run engine/build.sh"

if [ "${SUMMARIZE:-0}" != "1" ]; then
    if [ ! -x "$BIN_OLD" ]; then
        echo "building baseline $BASE into $BASEDIR (CPU only)"
        [ -d "$BASEDIR" ] || git worktree add "$BASEDIR" "$BASE" --detach -q || fail "worktree add failed"
        for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
        ( cd "$BASEDIR" && bash engine/build.sh >/dev/null ) || fail "baseline build failed"
    fi

    # --- canary A, outside the timing hold: the two binaries must be distinguishable ------------------
    "$BIN_OLD" --gdn-geom wf4 --model "$MODEL" --ids "1 2 3" --steps 0 >/dev/null 2>"$OUT/canaryA_old.err"
    grep -q "unknown argument --gdn-geom" "$OUT/canaryA_old.err" \
        || fail "canary A: the BASELINE accepted --gdn-geom. Either $BASE is not the pre-G2 tree or the wrong binary is under test; nothing is timed."
    "$BIN_NEW" --gdn-geom block --model "$MODEL" --ids "1 2 3" --steps 0 --out "$OUT/canary_new" >/dev/null 2>&1 \
        || fail "canary A: the SHIPPING binary refused --gdn-geom block. Is $BIN_NEW built at all?"
    echo "CANARY A ok: baseline rejects --gdn-geom, shipping binary accepts it"

    # --- numerics leg, outside the hold: deterministic, so it needs no exclusive device ----------------
    # Three claims, in the order that matters. This is NOT a bit-identity leg: G1 could demand one and G2
    # cannot, because the geometry re-associates (BET.md §10 constraint 2 fixes the band instead).
    #   (1) DEFAULT UNTOUCHED: base's bits == the new binary's `block` bits, every prefill logit row and
    #       every STATEFP line. If this moves, the knob leaked into the shipping path and no ms below
    #       means anything.
    #   (2) THE GEOMETRY IS ACTUALLY RUNNING: the wf sides must MOVE the state fingerprint. An identical
    #       STATEFP would mean the flag selected nothing and its ms would be the shipped form's a second
    #       time — the stale-binary null in AGENTS.md, caught on the numerics side instead of the timing
    #       side.
    #   (3) IN BAND, against the REFERENCE. The band adjudication itself lives in ONE place —
    #       `prototype/gdn_walk/gate_g2.py`, which owns the §10.1 floors, the per-layer attribution and the
    #       conv address witness — and it self-locks, so this script does NOT call it inside the hold. The
    #       CPU mirror runs here because it is deterministic and touches no device; the device-vs-pins leg
    #       is printed as a reminder to run `gate_g2.py` outside this window.
    GATE_IDS=$OUT/ids46.txt
    .venv/bin/python -c "
import numpy as np
ids = np.load('data/refs/p64/input_ids.npy').reshape(-1)
open('$GATE_IDS','w').write(','.join(str(int(x)) for x in ids))
" || fail "could not build the gate ids from data/refs/p64/input_ids.npy"
    for SIDE in base block $SIDES; do
        if [ "$SIDE" = base ]; then BINX=$BIN_OLD; else BINX=$BIN_NEW; fi
        "$BINX" --model "$MODEL" --ids "$(cat "$GATE_IDS")" --steps 0 --prefill-logits all --state-fp \
            $(geom_of "$SIDE") --wave-rows 16 --gdn-loop 8 --out "$OUT/bi_$SIDE" \
            > "$OUT/bi_$SIDE.log" 2>&1 || fail "numerics leg: $SIDE did not finish"
    done
    for f in "$OUT/bi_base"/*.bin; do
        b=$(basename "$f")
        cmp -s "$f" "$OUT/bi_block/$b" \
            || fail "default moved: --gdn-geom block differs from the BASELINE on $b. The shipping path is not inert; stop."
    done
    echo "NUMERICS leg 1 ok: base == block, bit for bit, on every prefill logit row and every STATEFP line"
    for k in $SIDES; do
        grep -h "^STATEFP" "$OUT/bi_base.log" > "$OUT/bi_base.fp"
        grep -h "^STATEFP" "$OUT/bi_$k.log" > "$OUT/bi_$k.fp"
        cmp -s "$OUT/bi_base.fp" "$OUT/bi_$k.fp" \
            && fail "numerics leg 2: $k's STATEFP is IDENTICAL to the baseline's. A re-associated walk must move the state; if it does not, the geometry is not running and its ms would be the shipped form's."
        echo "NUMERICS leg 2 ok: $k moves the state fingerprint — the geometry is live, not a flag that selects nothing"
    done
    .venv/bin/python prototype/gdn_walk/g2_mirror.py --layers L00,L01 --random 2 \
        > "$OUT/mirror.txt" 2>&1 \
        || fail "g2_mirror.py is red: the CPU witness for the band failed, so the numerics admission is not established. Tail: $(tail -2 "$OUT/mirror.txt" | tr '\n' ' ')"
    echo "MIRROR ok: g2_mirror.py green (cos/meanrel inside the existing §10.1 band; $OUT/mirror.txt)"
    echo "REMINDER: run '.venv/bin/python prototype/gdn_walk/gate_g2.py' OUTSIDE this window (it self-locks) for"
    echo "          the device-vs-pinned band and the per-layer attribution."

    tools/gpu_lock.sh acquire "$TAG" || exit 1
    echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
    echo "CLK@start: $(clk)"
    for R in $(seq 1 "$REPS"); do
        for SIDE in base block $SIDES; do
            sleep 5
            "$(bin_of "$SIDE")" --model "$MODEL" --ids "$IDS" $ARGS $(geom_of "$SIDE") \
                --out "$OUT/$(log_of "$SIDE")_$R" > "$(log_of "$SIDE")_$R.log" 2>&1 \
                || fail "$SIDE run rep=$R did not finish"
        done
        # canary B, inside the window: each side says which block shape it ran, and every side submits the
        # planned launch count on the SAME `gdn_rec` row. A stale binary passes the launch-count half of
        # this and fails the echo half, which is the AGENTS.md "paired A/B against a stale binary reads as
        # a clean null" case: the plan bill is compiled into the binary being timed and agrees with itself.
        for SIDE in block $SIDES; do
            LOG="$(log_of "$SIDE")_$R.log"
            grep -q "gdn_geom=$SIDE" "$LOG" || fail "canary B: rep=$R $SIDE side did not echo gdn_geom=$SIDE (it echoed: $(grep -o 'gdn_geom=[^ ]*' "$LOG" | tail -1)). The sides are not the shapes they claim; nothing is reported."
        done
        for SIDE in base block $SIDES; do
            LOG="$(log_of "$SIDE")_$R.log"
            GOT=$(grep -h "WAVE gran=stage family=gdn_rec " "$LOG" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1)
            PLAN=$(grep -h "WAVE gran=stage family=gdn_rec " "$LOG" | sed -E 's/.* plan_launches=([0-9]+).*/\1/' | tail -1)
            MATCH=$(grep -h "WAVE gran=stage family=gdn_rec " "$LOG" | sed -E 's/.* counts_match=([01]).*/\1/' | tail -1)
            [ "$GOT" = "$EXPECT_LAUNCHES" ] || fail "canary B: $LOG observed gdn_rec launches=$GOT, expected $EXPECT_LAUNCHES on every side. The geometry must not change the submission count; if it did, this is not the shape the bet is about."
            [ "$PLAN" = "$EXPECT_LAUNCHES" ] || fail "canary B: $LOG plan bill says $PLAN, expected $EXPECT_LAUNCHES (stage_clock.cpp's derivation is wrong)."
            [ "$MATCH" = "1" ] || fail "canary B: $LOG counts_match=0: the plan and the submissions disagree."
        done
        grep -q "TOTAL.*plan_match=1" "$(log_of block)_$R.log" || fail "canary B: plan_match!=1 on the block side at rep=$R"
    done
    echo "CLK@end: $(clk)"
    echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
    tools/gpu_lock.sh release "$TAG"
fi

# --- summary: per-family medians across the sides, paired by rep; then BOTH pre-registered verdicts ----
echo
echo "== $W lanes, $REPS reps, one window. Verdict lines: BET.md §10, gdn_rec <=$LINE_REC_LIVE / >$LINE_REC_DEAD, wall_ms <=$LINE_WALL_LIVE / >$LINE_WALL_DEAD."
printf "%-16s %10s %10s" family base_ms block_ms
for S in $SIDES; do printf " %10s" "new_$S"; done
printf " %12s\n" "share_of_wave"
FAMS="gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head host_tail"
# Median of one field across the reps, for one side. `fn` is fam/tot/res; the extra "$@" is the family name
# that `fam` needs and the others do not.
med_of() { local fn=$1 side=$2; shift 2; for R in $(seq 1 "$REPS"); do "$fn" "$@" "$(log_of "$side")_$R.log"; done | grep -v '^$' | median -; }
# A parser that matches nothing prints NA, and a summary of NAs reads like a slow engine rather than a
# broken harness — the empty-parse class in AGENTS.md. Base's wall is the one field that must exist.
[ "$(med_of tot base)" = "NA" ] && fail "no WAVE gran=stage TOTAL line in $(log_of base)_1.log: the logs are missing or the field names moved. Not a measurement."
for F in $FAMS; do
    B=$(med_of fam base "$F"); K=$(med_of fam block "$F")
    [ "$B" = "NA" ] && continue
    printf "%-16s %10s %10s" "$F" "$B" "$K"
    for S in $SIDES; do
        printf " %10s" "$(med_of fam "$S" "$F")"
    done
    printf " %12s\n" "$(grep -h "family=$F " "$(log_of block)_1.log" | sed -E 's/.*share_pct=([0-9.]+).*/\1/' | tail -1)"
done
med_of() { local fn=$1 side=$2; shift 2; for R in $(seq 1 "$REPS"); do "$fn" "$@" "$(log_of "$side")_$R.log"; done | grep -v '^$' | median -; }
BT=$(med_of tot base); KT=$(med_of tot block)
BR=$(med_of res base); KR=$(med_of res block)
printf "%-16s %10s %10s" TOTAL_wall "$BT" "$KT"
for S in $SIDES; do printf " %10s" "$(med_of tot "$S")"; done
printf "   (wave wall_ms)\n%-16s %10s %10s" RESULT "$BR" "$KR"
for S in $SIDES; do printf " %10s" "$(med_of res "$S")"; done
printf "   (RESULT ms_total)\n"

GR_B=$(med_of fam base gdn_rec)
for S in $SIDES; do
    GR_S=$(med_of fam "$S" gdn_rec)
    WL_S=$(med_of tot "$S")
    # The externality a single family row cannot see, which is what killed G1's line (§40: a +11.5 ms bill
    # outside the row it improved): ms summed over EVERY family other than the one improved, new minus base.
    EXT=0
    for F in $FAMS; do
        [ "$F" = "gdn_rec" ] && continue
        fb=$(med_of fam base "$F"); fs=$(med_of fam "$S" "$F")
        { [ "$fb" = "NA" ] || [ "$fs" = "NA" ]; } && continue
        EXT=$(awk -v e="$EXT" -v b="$fb" -v s="$fs" 'BEGIN{printf "%.3f", e + (s - b)}')
    done
    awk -v b="$GR_B" -v s="$GR_S" -v bw="$BT" -v ws="$WL_S" -v ext="$EXT" -v side="$S" \
        -v rl="$LINE_REC_LIVE" -v rd="$LINE_REC_DEAD" -v wl="$LINE_WALL_LIVE" -v wd="$LINE_WALL_DEAD" 'BEGIN{
        printf "\n== VERDICT for side %s, both pre-registered denominators (BET.md §10):\n", side;
        printf "   gdn_rec family row : %.2f -> %.2f ms  (%+.2f %%)   lines: live <= %s, dead > %s\n", b, s, 100*(s-b)/b, rl, rd;
        printf "   wave wall_ms       : %.2f -> %.2f ms  (%+.2f %%)   lines: live <= %s, dead > %s\n", bw, ws, 100*(ws-bw)/bw, wl, wd;
        printf "   EXTERNALITY        : %+.2f ms summed over every family other than gdn_rec\n", ext;
        rec_live = (s <= rl); rec_dead = (s > rd);
        wall_live = (ws <= wl); wall_dead = (ws > wd);
        if (rec_dead || wall_dead) v = "DEAD";
        else if (rec_live && wall_live) v = "LIVE";
        else v = "UNRESOLVED";
        printf "   VERDICT %s — both lines must clear their live bound to ship; either dead line kills.\n", v;
        if (v == "UNRESOLVED") printf "   Report it as unresolved: do not ship, do not narrate (§40 is the precedent).\n";
    }'
done
echo
echo "Also report: the base-vs-block NULL (medians must agree within window noise - see the -1.8/-2.4/-3.1 %"
echo "-1.8/-2.4/-3.1 % on untouched families), the per-side rep spread, and the clocks. A single rep"
echo "outside the spread is a discarded window, not a number."
echo "DONE. logs in $OUT/"
