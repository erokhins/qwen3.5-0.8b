#!/usr/bin/env bash
# run_ab_g4.sh — the timing A/B for bet G4 (`--gdn-geom reg16`): three sides, one window, one geometry read.
#
# MAIN-AGENT DEVICE WINDOW ONLY. Blocking, zero subagents, ONE `tools/gpu_lock.sh` hold, and it runs only
# programs that do not self-lock (`decode_run`, `tools/prefill_kernel_trace.py`; the CPU legs touch no
# device). A worker must not run it: it takes timings, and `AGENTS.md` keeps every timing on the main agent
# because this box serves its own tokens on the same iGPU. The worker-side correctness gate is
# `gate_g4.py` and the numerics mirror is `g4_mirror.py`; both self-lock or touch no device, and both are
# re-run by this script's non-timing legs.
#
# THE THREE SIDES.
#   base    HEAD (`git worktree`, built into $BASEDIR). It knows `--gdn-geom` (that shipped in 8b65fb0) and
#           REFUSES `reg16`. `base` vs `block` must be a NULL: the leg that proves adding G4 did not move
#           the shipping path.
#   block   the new binary at `--gdn-geom block` = the shipped `<<<16,128>>>` looped walk. The comparator.
#   reg16   BET.md §12's pre-registered shape: `<<<dim3(16,8), dim3(32,4)>>>` = 128 blocks x 128 threads
#           = 512 wavefronts, 16 state elements per thread in registers for the whole span.
#           THE VERDICT IS TAKEN ON THIS SIDE.
#
# PRE-REGISTERED OUTCOME — BOTH denominators, BET.md §12 / G3-llama-kernel.md §6, medians of 3 at 4,096
# lanes in one hold, base = §42's window (`gdn_rec` 264.00 ms, `wall_ms` 626.44 ms; §40's bases give the
# same verdict, which is why the lines were written that way):
#       gdn_rec family row :  live <= 170   ·   dead > 225
#       wave wall_ms       :  live <= 540   ·   dead > 607
#   BOTH must clear their live line to ship; EITHER past its dead line kills the form; anything else is
#   UNRESOLVED and is reported as unresolved with no narration (§40's rule, and the reason the second line
#   exists: G1's -5.9 % on its own row was -2.2 % of the wave). `wall_ms` is the ship gate. The externality
#   line printed below is that trap made visible.
#
# Usage:
#   bash prototype/gdn_walk/run_ab_g4.sh
#   REPS=2 bash prototype/gdn_walk/run_ab_g4.sh
#   SUMMARIZE=1 bash prototype/gdn_walk/run_ab_g4.sh        # re-judge the logs, touch no device
#   BASE=<sha> BASEDIR=/tmp/base-g4 bash prototype/gdn_walk/run_ab_g4.sh
set -uo pipefail
export LC_ALL=C # a decimal comma in a machine-read number is a parse bug wearing a locale's clothes
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
BASE=${BASE:-HEAD}
BASEDIR=${BASEDIR:-/tmp/base-g4}
BIN_NEW=engine/build/decode_run
BIN_OLD=$BASEDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
W=4096
IDS=$(cat progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt)
REPS=${REPS:-3}
TAG=${TAG:-g4-ab}
NEW_SIDES=${NEW_SIDES:-reg16}
OUT=${OUT:-tmp/gdn_walk/g4ab}
# BET.md §12's two lines, in one place so the verdict and the printed text cannot drift apart.
LINE_REC_LIVE=170;  LINE_REC_DEAD=225
LINE_WALL_LIVE=540; LINE_WALL_DEAD=607
# 8 bodies of 512 rows, one span each, 18 GDN layers -> 18*8 = 144 launches on EVERY side
# (`prototype/gdn_walk/plan_bill_check.py`'s derivation). G4 moves submissions to another SHAPE, never to
# another ROW, so the launch-count canary is equality; what separates the sides is the NUMERICS echo and the
# geometry census below (`engine/stage_clock.h`: billing a moved submission on the same row is what keeps
# the win line honest, §40).
EXPECT_LAUNCHES=144
mkdir -p "$OUT"
ARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --wave-rows 512 --gdn-loop 512"
# The trace leg does not need `--time-waves`; leaving it off keeps `passes` at 1 and the derived divisor honest.
TRACE_ARGS="--steps 0 --gemm lt --gemm-pins all --wave-rows 512 --gdn-loop 512"

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

    # --- CANARY A, outside the hold: the two binaries must be distinguishable BY THE NAME G4 ADDED -------
    # Not `grep -q "unknown argument --gdn-geom"`, which is what run_ab_g2.sh could use: `--gdn-geom` itself
    # shipped in 8b65fb0, so a correct G4 baseline knows the flag and refuses only the VALUE. A canary that
    # looks for the wrong refusal text reads as "stale baseline" and aborts a valid window, and the reverse
    # (accepting any non-zero) would pass a binary that failed for an unrelated reason.
    "$BIN_OLD" --gdn-geom reg16 --model "$MODEL" --ids "1,2,3" --steps 0 >/dev/null 2>"$OUT/canaryA_old.err"
    [ $? -ne 0 ] || fail "canary A: the BASELINE ACCEPTED --gdn-geom reg16. Either $BASE is not a pre-G4 tree or the wrong binary is under test; nothing is timed."
    grep -q "got 'reg16'" "$OUT/canaryA_old.err" \
        || fail "canary A: the baseline failed, but not by refusing the NAME (no \"got 'reg16'\" in $OUT/canaryA_old.err). A refusal for some other reason proves nothing about the tree."
    "$BIN_NEW" --gdn-geom reg16 --model "$MODEL" --ids "1,2,3" --steps 0 --out "$OUT/canaryA_new" \
        >/dev/null 2>&1 || fail "canary A: the SHIPPING binary refused --gdn-geom reg16. Is $BIN_NEW built from this tree at all?"
    echo "CANARY A ok: baseline refuses the NAME reg16 for the right reason; shipping binary accepts it"

    # --- NUMERICS LEGS, outside the hold: deterministic, so they need no exclusive device -----------------
    # Three claims, in the order that matters. This is NOT a bit-identity leg: G4 re-associates (BET §12
    # constraint 5 fixes the band instead of bit-identity), so the three legs are (1) the default is inert,
    # (2) the new map is actually running, (3) the band holds — and (3) is owned by `gate_g4.py`, which
    # self-locks and therefore runs OUTSIDE this window.
    GATE_IDS=$OUT/ids46.txt
    .venv/bin/python -c "
import numpy as np
ids = np.load('data/refs/p64/input_ids.npy').reshape(-1)
open('$GATE_IDS','w').write(','.join(str(int(x)) for x in ids))
" || fail "could not build the gate ids from data/refs/p64/input_ids.npy"
    for SIDE in base block $NEW_SIDES; do
        if [ "$SIDE" = base ]; then BINX=$BIN_OLD; else BINX=$BIN_NEW; fi
        "$BINX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$GATE_IDS")" --steps 0 --prefill-logits all --state-fp \
            $(geom_of "$SIDE") --wave-rows 16 --gdn-loop 8 --out "$OUT/bi_$SIDE" \
            > "$OUT/bi_$SIDE.log" 2>&1 || fail "numerics leg: $SIDE did not finish"
    done
    for f in "$OUT/bi_base"/*.bin; do
        b=$(basename "$f")
        cmp -s "$f" "$OUT/bi_block/$b" \
            || fail "default moved: --gdn-geom block differs from the BASELINE on $b. The shipping path is not inert; stop."
    done
    echo "NUMERICS leg 1 ok: base == block, bit for bit, on every prefill logit row and every STATEFP line"
    grep -h "^STATEFP" "$OUT/bi_base.log" > "$OUT/bi_base.fp"
    for k in $NEW_SIDES; do
        grep -h "^STATEFP" "$OUT/bi_$k.log" > "$OUT/bi_$k.fp"
        cmp -s "$OUT/bi_base.fp" "$OUT/bi_$k.fp" \
            && fail "numerics leg 2: $k's STATEFP is IDENTICAL to the baseline's. A register-resident walk must move the state; if it does not, the geometry is not running and its ms is the shipped form's a second time (the stale-binary null, caught on the numerics side)."
        echo "NUMERICS leg 2 ok: $k moves the state fingerprint — the map is live, not a flag that selects nothing"
    done
    .venv/bin/python prototype/gdn_walk/g4_mirror.py --layers L00,L01 --random 2 \
        > "$OUT/mirror.txt" 2>&1 \
        || fail "g4_mirror.py is red: the CPU witness for the band failed, so the numerics admission is not established. Tail: $(tail -2 "$OUT/mirror.txt" | tr '\n' ' ')"
    .venv/bin/python prototype/gdn_walk/g4_mirror.py --geom >/dev/null 2>&1 \
        || fail "g4_mirror.py --geom is red: the thread map is not a bijection over the state, so nothing else here is about the intended geometry"
    echo "MIRROR ok: g4_mirror.py green, map is a bijection (cos/meanrel inside the existing §10.1 band; $OUT/mirror.txt)"
    echo "REMINDER: run '.venv/bin/python prototype/gdn_walk/gate_g4.py --base-bin $BIN_OLD' OUTSIDE this window"
    echo "          (it self-locks) for the device-vs-pinned band, the span-invariance legs, token parity and"
    echo "          the per-layer attribution. Its far-field block prints BOTH readings; read the set."

    tools/gpu_lock.sh acquire "$TAG" || exit 1
    echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
    echo "CLK@start: $(clk)"

    # --- CANARY C, inside the hold: the GEOMETRY ITSELF, from a trace, before any timing ------------------
    # G3 §6 constraint 6 requires this and it is the only leg that sees the launch shape: the launch count is
    # identical on every side (144), so canary B cannot tell the two maps apart, and `engine/build.sh` is not
    # bit-reproducible on this box, so comparing binaries proves nothing (AGENTS.md, 2026-10-17: "a paired
    # A/B against a stale binary reads as a clean null"). rocprofv3's Grid_Size_* are per-dimension THREAD
    # counts and `trace_phase_split.py --geom` now keys on all three axes (fixed at 879ef31); an X-only read
    # prints 16 blocks for a 3-D launch, which is how §38 got its geometry wrong (§43's retirement).
    for SIDE in block $NEW_SIDES; do
        .venv/bin/python tools/prefill_kernel_trace.py --waves 8 --out "$OUT/trace_$SIDE" -- \
            "./$BIN_NEW" --model "$MODEL" --ids "$IDS" $TRACE_ARGS $(geom_of "$SIDE") \
            --out "$OUT/trace_${SIDE}_out" > "$OUT/trace_$SIDE.log" 2>&1 \
            || fail "geometry canary: the traced $SIDE run did not finish (see $OUT/trace_$SIDE.log)"
        CSV=$(ls "$OUT/trace_$SIDE"/*_kernel_trace.csv 2>/dev/null | head -1)
        [ -n "$CSV" ] || fail "geometry canary: no kernel_trace.csv for $SIDE. The census did not run, so the geometry is unverified and nothing here is timed."
        .venv/bin/python tools/trace_phase_split.py "$CSV" --engine "g4-$SIDE" \
            --calibrate attn_prep --per-pass 48 --geom rec_ > "$OUT/geom_$SIDE.txt" 2>&1 \
            || fail "geometry canary: trace_phase_split.py refused the $SIDE census (see $OUT/geom_$SIDE.txt)"
    done
    # `rec_loop_kernel` must read (128,1,1) x 16 blocks; `rec_reg16_kernel` must read (32,4,1) x 128 blocks.
    # Assert on the census ROWS, not on a substring of the file: a canary that greps the word "reg16" passes
    # when the kernel is missing entirely, because the header line names it.
    awk 'NR>1 && $4+0>0 {print}' "$OUT/geom_block.txt" | grep -q "(128,1,1)" \
        || fail "geometry canary: the \`block\` census has no (128,1,1) row. The shipped walk is not the shape the bet is compared against; see $OUT/geom_block.txt"
    grep -q "(32,4,1)" "$OUT/geom_$NEW_SIDES.txt" \
        || fail "geometry canary: the \`$NEW_SIDES\` census has no (32,4,1) row, so the traced kernel is NOT llama's map. The bet is about the block shape and nothing below measures it. See $OUT/geom_$NEW_SIDES.txt"
    BLOCKS_NEW=$(awk '/\(32,4,1\)/ {print $4; exit}' "$OUT/geom_$NEW_SIDES.txt")
    [ "$BLOCKS_NEW" = "128" ] || fail "geometry canary: the $NEW_SIDES census reports $BLOCKS_NEW blocks/launch, expected 128 (grid (16,8) x workgroup (32,4)). Not llama's 128 blocks."
    grep -q "rec_reg16_kernel" "$OUT/geom_$NEW_SIDES.txt" \
        || fail "geometry canary: the $NEW_SIDES census names no rec_reg16_kernel, so the traced side ran some other body"
    echo "GEOMETRY CANARY ok: block=(128,1,1)x16 blocks, $NEW_SIDES=(32,4,1)x$BLOCKS_NEW blocks, read in this hold ($OUT/geom_block.txt, $OUT/geom_$NEW_SIDES.txt)"

    for R in $(seq 1 "$REPS"); do
        for SIDE in base block $NEW_SIDES; do
            sleep 5
            "$(bin_of "$SIDE")" --model "$MODEL" --ids "$IDS" $ARGS $(geom_of "$SIDE") \
                --out "$OUT/$(log_of "$SIDE")_$R" > "$(log_of "$SIDE")_$R.log" 2>&1 \
                || fail "$SIDE run rep=$R did not finish"
        done
        # CANARY B, inside the window: each side says which map it ran, and every side submits the planned
        # launch count on the SAME `gdn_rec` row. The launch-count half cannot distinguish the sides — the
        # geometry moved submissions to a shape, not to a row — so the echo is the discriminator, and the
        # geometry canary above is the one that cannot be satisfied by a stale binary.
        for SIDE in block $NEW_SIDES; do
            LOG="$(log_of "$SIDE")_$R.log"
            grep -q "gdn_geom=$SIDE" "$LOG" || fail "canary B: rep=$R $SIDE side did not echo gdn_geom=$SIDE (it echoed: $(grep -o 'gdn_geom=[^ ]*' "$LOG" | tail -1)). The sides are not the shapes they claim; nothing is reported."
        done
        for SIDE in base block $NEW_SIDES; do
            LOG="$(log_of "$SIDE")_$R.log"
            GOT=$(grep -h "WAVE gran=stage family=gdn_rec " "$LOG" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1)
            PLAN=$(grep -h "WAVE gran=stage family=gdn_rec " "$LOG" | sed -E 's/.* plan_launches=([0-9]+).*/\1/' | tail -1)
            MATCH=$(grep -h "WAVE gran=stage family=gdn_rec " "$LOG" | sed -E 's/.* counts_match=([01]).*/\1/' | tail -1)
            [ "$GOT" = "$EXPECT_LAUNCHES" ] || fail "canary B: $LOG observed gdn_rec launches=$GOT, expected $EXPECT_LAUNCHES on every side. The geometry must not move submissions off the row it bills."
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
echo "== $W lanes, $REPS reps, one window. Verdict lines (BET.md §12): gdn_rec <=$LINE_REC_LIVE live /"
echo "   >$LINE_REC_DEAD dead; wave wall_ms <=$LINE_WALL_LIVE live / >$LINE_WALL_DEAD dead."
printf "%-16s %10s %10s" family base_ms block_ms
for S in $NEW_SIDES; do printf " %10s" "new_$S"; done
printf " %14s\n" "share_of_wave"
FAMS="gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head host_tail"
# Median of one field across the reps, for one side. `fn` is fam/tot/res; the extra "$@" is the family name
# that `fam` needs and the others do not. Defined BEFORE first use (mk_parsecheck's first run found that bug).
med_of() { local fn=$1 side=$2; shift 2; for R in $(seq 1 "$REPS"); do "$fn" "$@" "$(log_of "$side")_$R.log"; done | grep -v '^$' | median -; }
# A parser that matches nothing prints NA, and a table of NAs reads like a slow engine rather than a broken
# harness — the empty-parse class in AGENTS.md. Base's wall is the one field that must exist.
[ "$(med_of tot base)" = "NA" ] && fail "no WAVE gran=stage TOTAL line in $(log_of base)_1.log: the logs are missing or the field names moved. Not a measurement."
for F in $FAMS; do
    B=$(med_of fam base "$F"); K=$(med_of fam block "$F")
    [ "$B" = "NA" ] && continue
    printf "%-16s %10s %10s" "$F" "$B" "$K"
    for S in $NEW_SIDES; do printf " %10s" "$(med_of fam "$S" "$F")"; done
    printf " %14s\n" "$(grep -h "family=$F " "$(log_of block)_1.log" | sed -E 's/.*share_pct=([0-9.]+).*/\1/' | tail -1)"
done
BT=$(med_of tot base); KT=$(med_of tot block)
BR=$(med_of res base); KR=$(med_of res block)
printf "%-16s %10s %10s" TOTAL_wall "$BT" "$KT"
for S in $NEW_SIDES; do printf " %10s" "$(med_of tot "$S")"; done
printf "   (wave wall_ms)\n%-16s %10s %10s" RESULT "$BR" "$KR"
for S in $NEW_SIDES; do printf " %10s" "$(med_of res "$S")"; done
printf "   (RESULT ms_total)\n"

GR_B=$(med_of fam base gdn_rec)
for S in $NEW_SIDES; do
    GR_S=$(med_of fam "$S" gdn_rec)
    WL_S=$(med_of tot "$S")
    # The externality a single family row cannot see, which is what killed G1's line (§40: a +11.5 ms bill
    # outside the row it improved). G4 is the form most likely to buy an externality of this kind: staging
    # no LDS means the norms are paid in global traffic, so the sum over EVERY other family is the check.
    EXT=0
    for F in $FAMS; do
        [ "$F" = "gdn_rec" ] && continue
        fb=$(med_of fam base "$F"); fs=$(med_of fam "$S" "$F")
        { [ "$fb" = "NA" ] || [ "$fs" = "NA" ]; } && continue
        EXT=$(awk -v e="$EXT" -v b="$fb" -v s="$fs" 'BEGIN{printf "%.3f", e + (s - b)}')
    done
    awk -v b="$GR_B" -v s="$GR_S" -v bw="$BT" -v ws="$WL_S" -v ext="$EXT" -v side="$S" \
        -v rl="$LINE_REC_LIVE" -v rd="$LINE_REC_DEAD" -v wl="$LINE_WALL_LIVE" -v wd="$LINE_WALL_DEAD" 'BEGIN{
        printf "\n== VERDICT for side %s, both pre-registered denominators (BET.md §12):\n", side;
        printf "   gdn_rec family row : %.2f -> %.2f ms  (%+.2f %%)   lines: live <= %s, dead > %s\n", b, s, 100*(s-b)/b, rl, rd;
        printf "   wave wall_ms       : %.2f -> %.2f ms  (%+.2f %%)   lines: live <= %s, dead > %s   [SHIP GATE]\n", bw, ws, 100*(ws-bw)/bw, wl, wd;
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
echo "Also report: the base-vs-block NULL (medians must agree within window noise), the per-side rep spread,"
echo "the clocks, and the two --geom census files (the shape the bet is about, read in this window). A single"
echo "rep outside the spread is a discarded window, not a number. Timings from this script are the ONLY ones"
echo "that may enter the ledger; a worker's device run may not."
echo "DONE. logs in $OUT/"
