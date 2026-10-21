#!/usr/bin/env bash
# run_ab_a6g1.sh — the device A/B that adjudicates `prototype/attn_percost/A6-BET.md` for G1: Q un-staged plus the
# epilogue staged 8 rows at a time, on the engine path behind `--attn-fattn-qstage stage|unstaged8`.
#
# MAIN-AGENT DEVICE WINDOW ONLY. Blocking, zero subagents, ONE `tools/gpu_lock.sh` hold for the timed legs. The
# numerics legs (which include `a6_g1_bits.sh`, and which SELF-LOCK) run BEFORE the hold is taken — AGENTS'
# deadlock row: three gate drivers acquire `tools/gpu_lock.sh` themselves, so a caller that holds the lock and
# then runs one of them parks silently. This harness self-locks, so it must never be run from inside a hold, and
# it must not call a self-locking script while holding one.
#
# WHAT IS BEING DECIDED — A6-BET's G1 bar, "same five rows, same ratios", fixed before any measurement:
#   attn family, untraced ms/pass   live <= 0.8947 x base   dead > 0.9699 x base
#   attention traced ms/pass        live <= 0.8991 x base   dead > 0.9697 x base   [must agree in direction]
#   wave wall_ms @4,096             live <= 0.9435 x base   dead > 0.9699 x base   [SHIP GATE]
#   16,384-lane wave ms             live <= 0.9189 x base   dead > 0.9701 x base   [SHIP GATE]
#   32,768-lane wave ms             live <= 0.8997 x base   dead > 0.9700 x base   [SHIP GATE]
# The verdict, stated here so it cannot be re-sized afterwards (A6-BET: "if it lands inside them, it is a win,
# and if it lands between live and dead it is unresolved and not shipped"):
#   LIVE        family <= live line AND all three ship gates <= their live lines AND the traced row improves
#   DEAD        family > dead line OR any ship gate > its dead line
#   UNRESOLVED  anything in between. Not shipped, not narrated (A4-BET's words, and the §47 precedent of
#               engine/performance-status.md).
# Unlike G0, G1 HAS an inert side, so A4's NULL leg is restored: `stage` (the new binary, knob unnamed) against
# `base` must read the same milliseconds within NULL_TOL on the family row and on every ship gate, or the window
# is unpaired and the harness refuses to print a verdict at all. Bit-identity of that side is a separate,
# deterministic claim and is taken outside the hold by `a6_g1_bits.sh`.
#
# THE THREE SIDES. base is a `git worktree` at the pinned PRE-change sha built as-is; stage and u8 are the SAME
# new binary, differing only by `--attn-fattn-qstage unstaged8`. One binary for two sides is the point: both
# kernels are in it, so the comparison is one window, not two rulers re-taken in different weather (AGENTS: a
# ruler re-taken in a different window is not an A/B).
#
# WHAT IT REFUSES TO DO (each refusal is a lesson from progress/2026-10-21/performance-status-2026-10-21.md §47 / §37 / §44):
#   * it does not time a binary it has not identified. CANARY A reads the NOTE BLOCKS of both LINKED binaries
#     (base `attn_fattn_wave` 178/0/0/32768; new `attn_fattn_wave` the same and `attn_fattn_wave_u8`
#     185/0/0/20480 — `A6-result.md` §1, re-read on the built tree in `A6-G1-impl.md` §2). CANARY B reads the
#     launcher's own `q= lds= wf_per_cu= vgpr=` from the run that produced the milliseconds and requires them to
#     agree with those note blocks: `lds=32768 wf_per_cu=6` on base and stage, `lds=20480 wf_per_cu=9` on u8.
#     A side whose banner disagrees with its note block is the wrong binary (progress/2026-10-21/performance-status-2026-10-21.md §44's
#     stale-build trap, and
#     `engine/build.sh` is not bit-reproducible).
#   * it does not time a run whose launch counts or coverage fields differ (progress/2026-10-21/performance-status-2026-10-21.md §41 closed
#     launch count as a variable and this bet must not reopen it; `fattn_covers` does not read a staging plan).
#   * it does not pass an empty parse: every field is `need`-checked, and PARSE_CHECK=1 colour-tests the verdict
#     parser against fixtures GENERATED from the ratio constants in this file (`mk_a6g1_parsecheck.py`), never
#     hand-typed — `mk_a4l1_parsecheck.py`'s history is the reason.
#   * it does not take the wide legs through argv: one argv slot caps at MAX_ARG_STRLEN = 131,072 B, so a
#     32,768-id prompt cannot be passed at all (progress/2026-10-21/performance-status-2026-10-21.md §39: shell exit 126 in 0 s, read as
#     "the engine crashes at 32k").
#     Wide legs go via --ids-file and the field count is asserted.
#   * it does not print a verdict from an unpaired window (the NULL leg), nor from reps that disagree by more than
#     SPREAD_TOL, nor past a red numerics leg.
#
# EXTERNALITY is printed and decides nothing, as A6-BET pre-registered no line for it. It is printed against
# G0's reading — `A6-G0-verdict.md`: +6.80 / +74.72 / -1.03 ms at 4,096 / 16,384 / 32,768, where the 16k number
# was a 224-thread block making its neighbours slower. G1 keeps 96-thread blocks and only drops LDS, so it should
# not reproduce that; if it does, that is a co-residency finding worth its own line in the report.
#
# Usage:
#   bash prototype/attn_percost/run_ab_a6g1.sh                      # the window (main agent, blocking)
#   BASE=<sha> bash prototype/attn_percost/run_ab_a6g1.sh           # pin the pre-change sha
#   NEWDIR=/path/to/changed/tree bash prototype/attn_percost/run_ab_a6g1.sh
#   REPS=2 REPS_WIDE=2 bash prototype/attn_percost/run_ab_a6g1.sh
#   SKIP_NUMERICS=1 bash prototype/attn_percost/run_ab_a6g1.sh      # timing only (bit-identity taken separately)
#   DRY_RUN=1    bash prototype/attn_percost/run_ab_a6g1.sh         # print the plan; builds nothing, touches no device
#   SUMMARIZE=1  bash prototype/attn_percost/run_ab_a6g1.sh         # re-judge the logs, touch no device
#   PARSE_CHECK=1 bash prototype/attn_percost/run_ab_a6g1.sh        # colour-test the parser, touch no device
set -uo pipefail
export LC_ALL=C          # a decimal comma in a machine-read number is a parse bug wearing a locale's clothes
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD

# ---- pins ----------------------------------------------------------------------------
# A PRE-change sha: `base` must be the thing the bet is against, not the thing that already contains it. Re-pin
# this to the sha the tree ships at when the knob lands; `HEAD` is not a baseline, because after the landing HEAD
# *is* the change.
BASE=${BASE:-ed658fe}
SHORT=$(git rev-parse --short "$BASE" 2>/dev/null || echo "$BASE")
BASEDIR=${BASEDIR:-/tmp/a6g1-base-$SHORT}
NEWDIR=${NEWDIR:-$REPO}                        # the changed tree; a worktree of it works too
BIN_BASE=$BASEDIR/engine/build/decode_run
BIN_NEW=$NEWDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
PY=.venv/bin/python
REPS=${REPS:-3}
REPS_WIDE=${REPS_WIDE:-2}
W=4096
WIDE_WIDTHS=${WIDE_WIDTHS:-"16384 32768"}
TAG=${TAG:-a6g1-ab}
OUT=${OUT:-tmp/attn_percost/a6g1ab}
IDS4096=${IDS4096:-progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt}
mkdir -p "$OUT"
# Absolute, because the build legs `cd` into a worktree before redirecting into $OUT: a relative log path
# resolves inside the worktree, where tmp/ does not exist, and the build then fails for a bug that has nothing to
# do with the code under test (the f736e27 lesson).
OUT="$(cd "$OUT" && pwd)"

# ---- the lines: A6-BET's G1 bar (source: prototype/attn_percost/A6-BET.md), nothing else --------------
RAT_A_LIVE=${RAT_A_LIVE:-0.8947};   RAT_A_DEAD=${RAT_A_DEAD:-0.9699}
RAT_AT_LIVE=${RAT_AT_LIVE:-0.8991}; RAT_AT_DEAD=${RAT_AT_DEAD:-0.9697}
RAT_W_LIVE=${RAT_W_LIVE:-0.9435};   RAT_W_DEAD=${RAT_W_DEAD:-0.9699}
RAT_16_LIVE=${RAT_16_LIVE:-0.9189}; RAT_16_DEAD=${RAT_16_DEAD:-0.9701}
RAT_32_LIVE=${RAT_32_LIVE:-0.8997}; RAT_32_DEAD=${RAT_32_DEAD:-0.9700}
NULL_TOL=${NULL_TOL:-2.0}                     # %: how far the inert side may move before the window is unpaired
SPREAD_TOL=${SPREAD_TOL:-3.0}                 # %: per-rep spread tolerated inside one window
# G0's externality, for the comparison line only (`A6-G0-verdict.md`). Informational.
EXT_G0_16=${EXT_G0_16:-74.72}
FAMS="gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head host_tail"
ARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --wave-rows 512"
TRACE_ARGS="--steps 0 --gemm lt --gemm-pins all --wave-rows 512"
QSTAGE_ARG=" --attn-fattn-qstage unstaged8"
# Note-block expectations, read from `A6-result.md` §1 and re-confirmed on the built tree in `A6-G1-impl.md` §2.
NOTE_STAGE_WANT=${NOTE_STAGE_WANT:-"178 0 0 32768"}    # vgpr spill private group
NOTE_U8_WANT=${NOTE_U8_WANT:-"185 0 0 20480"}
BAN_STAGE_WANT=${BAN_STAGE_WANT:-"lds=32768 wf_per_cu=6"}
BAN_U8_WANT=${BAN_U8_WANT:-"lds=20480 wf_per_cu=9"}

clk()    { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk" | tr '\n' ' '; }
fam()    { grep -h "WAVE gran=stage family=$1 " "$2" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
tot()    { grep -h "WAVE gran=stage TOTAL" "$1" | sed -E 's/.*wall_ms=([0-9.]+).*/\1/' | tail -1; }
traced() { awk '$1=="attention"{print $2}' "$1" | tail -1; }
median() { sort -g "$@" | awk '{a[NR]=$1} END{if(NR==0){print "NA";exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
fail()   { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
need()   { [ -n "$1" ] || fail "empty parse where a number was required ($2). A verdict computed from nothing is the empty-parse failure AGENTS.md records; not a measurement."; }
med_of()   { local fn=$1 side=$2; shift 2
    for R in $(seq 1 "$REPS"); do "$fn" "$@" "$OUT/${side}_$R.log"; done | grep -v '^$' | median -; }
med_wide() { local WID=$1 side=$2
    for R in $(seq 1 "$REPS_WIDE"); do tot "$OUT/wide${WID}_${side}_$R.log"; done | grep -v '^$' | median -; }
med_wide_fam() { local WID=$1 side=$2 F=$3
    for R in $(seq 1 "$REPS_WIDE"); do grep -h "family=$F " "$OUT/wide${WID}_${side}_$R.log" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; done | grep -v '^$' | median -; }
med_trace(){ local side=$1
    for R in $(seq 1 "$REPS"); do traced "$OUT/trace_${side}_$R.txt"; done | grep -v '^$' | median -; }
spread_of(){ local fn=$1 side=$2; shift 2
    local vals med lo hi
    vals=$(for R in $(seq 1 "$REPS"); do "$fn" "$@" "$OUT/${side}_$R.log"; done | grep -v '^$')
    med=$(echo "$vals" | median -); lo=$(echo "$vals" | sort -g | head -1); hi=$(echo "$vals" | sort -g | tail -1)
    awk -v m="$med" -v l="$lo" -v h="$hi" 'BEGIN{ if (m==0||m=="NA"){print "NA"} else printf "%.2f", 100*(h-l)/m }'
}
note_of() { "$PY" prototype/attn_percost/a4_l1_note_read.py "$1" --kernel "$2" 2>&1 \
            | sed -n "s/^KERNEL $2 .*vgpr_count=\([0-9]*\) vgpr_spill_count=\([0-9]*\).*private_segment_fixed_size=\([0-9]*\) group_segment_fixed_size=\([0-9]*\).*/\1 \2 \3 \4/p"; }
pct_move() { awk -v b="$1" -v n="$2" 'BEGIN{ if (b==0||b=="NA"||n=="NA"){print "NA"} else printf "%.2f", 100*(n-b)/b }'; }

ids_file_of() { local N=$1 DST=$2
    if [ -f "$IDS4096" ] && [ "$N" -le 4096 ]; then cut -d, -f"1-$N" "$IDS4096" > "$DST"
    else
        "$PY" - "$N" "$DST" <<'PYIDS' || fail "could not generate the wide id file"
import random, sys
n, path = int(sys.argv[1]), sys.argv[2]
rng = random.Random(20261019)              # longctx_check.sh's seed, so wide rows stay comparable
with open(path, "w") as f:
    f.write(",".join(str(rng.randrange(1000, 60000)) for _ in range(n)))
PYIDS
    fi
    local have; have=$("$PY" -c "import sys;print(len(open(sys.argv[1]).read().split(',')))" "$DST")
    [ "$have" = "$N" ] || fail "width $N needs $N ids and $DST holds $have — refusing to judge a run whose prompt is not the width under test"
}

# ============================ DRY_RUN: print the plan, touch nothing ==============================
# The brief for this harness is "hand the main agent a window it can audit before it spends it". DRY_RUN=1 prints
# every command line, pin and refusal the window would use, and exits without building, locking or running.
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN — nothing executed, no lock taken, no binary built."
    echo "  base      : $BASE  -> $BIN_BASE   (worktree at $BASEDIR, built if absent)"
    echo "  new       : $NEWDIR -> $BIN_NEW"
    echo "  pins      : default note blocks must read [$NOTE_STAGE_WANT] and unstaged8 [$NOTE_U8_WANT] (vgpr spill private group)"
    echo "  banners   : stage/base [$BAN_STAGE_WANT], u8 [$BAN_U8_WANT], each with vgpr equal to its own note block"
    echo "  numerics  : a6_g1_bits.sh BEFORE the hold (it self-locks; never call it from inside one)"
    echo "  legs @4096 : REPS=$REPS  args [$ARGS]  and for u8 [$ARGS$QSTAGE_ARG]"
    for SIDE in base stage u8; do
        case $SIDE in
            base)  EX="$BIN_BASE --model $MODEL --ids \$(cat $OUT/ids$W.csv) $ARGS --out $OUT/${SIDE}_R" ;;
            stage) EX="$BIN_NEW --model $MODEL --ids \$(cat $OUT/ids$W.csv) $ARGS --out $OUT/${SIDE}_R" ;;
            u8)    EX="$BIN_NEW --model $MODEL --ids \$(cat $OUT/ids$W.csv) $ARGS$QSTAGE_ARG --out $OUT/${SIDE}_R" ;;
        esac
        echo "    $SIDE: $EX"
    done
    echo "  traced    : tools/prefill_kernel_trace.py --waves 8 -- $TRACE_ARGS   (base and u8, REPS=$REPS)"
    echo "              then tools/trace_phase_split.py --calibrate attn_prep --per-pass 48 --tokens $W"
    echo "  wide      : REPS_WIDE=$REPS_WIDE widths [$WIDE_WIDTHS] via --ids-file (never argv: one argv slot caps at"
    echo "              131072 B — progress/2026-10-21/performance-status-2026-10-21.md §39)"
    for WID in $WIDE_WIDTHS; do echo "    $WID: $BIN_NEW --model $MODEL --ids-file $OUT/ids$WID.csv --cap $WID $ARGS$QSTAGE_ARG --out $OUT/wide${WID}_u8_R"; done
    echo "  verdict   : LIVE needs family<=$RAT_A_LIVE and walls<=$RAT_W_LIVE/$RAT_16_LIVE/$RAT_32_LIVE and traced<=$RAT_AT_LIVE;"
    echo "              DEAD if family>$RAT_A_DEAD or a ship gate > its dead line; otherwise UNRESOLVED (not shipped)."
    echo "              NULL leg |move| <= $NULL_TOL % on the family row and every ship gate, else UNPAIRED and no verdict."
    echo "              externality printed against G0's $EXT_G0_16 ms at 16,384; it decides nothing."
    echo "  refuses   : stale/mismatched binaries (note block vs banner), a launch-count or coverage-field difference,"
    echo "              an empty parse, a rep spread > $SPREAD_TOL %, a wide prompt whose id count is not the width."
    echo "  logs      : $OUT/   (lock tag $TAG)"
    exit 0
fi

# ============================ PARSE_CHECK: colour-test the verdict parser ==============================
if [ "${PARSE_CHECK:-0}" = "1" ]; then
    for SCEN in live unresolved shipgate_blocks dead traced_disagrees null_unpaired; do
        case $SCEN in
            live)             WANT=LIVE ;;
            unresolved)       WANT=UNRESOLVED ;;
            shipgate_blocks)  WANT=UNRESOLVED ;;
            traced_disagrees) WANT=UNRESOLVED ;;
            dead)             WANT=DEAD ;;
            null_unpaired)    WANT=UNPAIRED ;;
        esac
        SCEN=$SCEN OUT="$OUT/pc_$SCEN" REPS=1 REPS_WIDE=1 \
            "$PY" prototype/attn_percost/mk_a6g1_parsecheck.py \
            || fail "parse-check generator failed for scenario $SCEN"
        env -u PARSE_CHECK OUT="$OUT/pc_$SCEN" REPS=1 REPS_WIDE=1 SKIP_NUMERICS=1 SUMMARIZE=1 bash "$0" 2>&1 \
            > "$OUT/pc_$SCEN.summary" || true
        GOT=$(sed -n 's/^== VERDICT for G1: \([A-Z_]*\).*/\1/p' "$OUT/pc_$SCEN.summary" | tail -1)
        [ -n "$GOT" ] || GOT=$(grep -o 'UNPAIRED' "$OUT/pc_$SCEN.summary" | head -1)
        [ "$GOT" = "$WANT" ] || fail "parse-check: scenario $SCEN printed '${GOT:-<nothing>}', expected $WANT. The verdict parser is not reading its own table; no real window may be run."
        echo "PARSE-CHECK ok: scenario $SCEN -> $GOT (expected $WANT)"
    done
    echo "PARSE-CHECK PASS: the parser distinguishes LIVE / UNRESOLVED (family row, ship gate, traced disagreement) / DEAD / and refuses an unpaired NULL. This says nothing about the engine."
    exit 0
fi

[ "${SUMMARIZE:-0}" = "1" ] && [ ! -f "$OUT/base_1.log" ] && fail "SUMMARIZE over $OUT, which holds no base_1.log: a verdict computed from an empty log set is the empty-parse failure AGENTS.md records."

if [ "${SUMMARIZE:-0}" != "1" ]; then
    [ -f "$IDS4096" ] || fail "no 4,096-id file at $IDS4096"
    [ -x "$BIN_NEW" ] || fail "no changed-tree binary at $BIN_NEW — run engine/build.sh first"
    [ -d "$BASEDIR" ] || git worktree add "$BASEDIR" "$BASE" --detach -q || fail "worktree add failed for $BASE"
    for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
    [ -x "$BIN_BASE" ] || { echo "building base $BASE into $BASEDIR (CPU only)"; ( cd "$BASEDIR" && bash engine/build.sh >"$OUT/build_base.log" 2>&1 ) || fail "baseline build failed (see $OUT/build_base.log)"; }
    [ "$BIN_BASE" -nt "$BASEDIR/engine/attn_fattn_kernels.h" ] || fail "the base binary is older than the source it came from"
    [ "$BIN_NEW"  -nt "$NEWDIR/engine/attn_fattn_kernels.h"   ] || fail "the new binary is older than the changed source it came from"

    # ---- CANARY A: the NOTE BLOCKS of the two LINKED binaries (the stale-binary guard) ------------------
    NB=$(note_of "$BIN_BASE" attn_fattn_wave);   need "$NB" "the base binary's attn_fattn_wave note block"
    NN=$(note_of "$BIN_NEW"  attn_fattn_wave);   need "$NN" "the new binary's attn_fattn_wave note block"
    NU=$(note_of "$BIN_NEW"  attn_fattn_wave_u8); need "$NU" "the new binary's attn_fattn_wave_u8 note block"
    [ "$NB" = "$NOTE_STAGE_WANT" ] || fail "canary A: base attn_fattn_wave reads '$NB', want '$NOTE_STAGE_WANT' (vgpr spill private group). The comparator is not the shipped form; nothing is timed."
    [ "$NN" = "$NOTE_STAGE_WANT" ] || fail "canary A: the new binary's DEFAULT kernel reads '$NN', want '$NOTE_STAGE_WANT'. The knob moved the default's budget, so `stage` is not the shipped form and the NULL leg below cannot exist."
    [ "$NU" = "$NOTE_U8_WANT" ] || fail "canary A: the new binary's unstaged8 kernel reads '$NU', want '$NOTE_U8_WANT' (A6-result.md §1). The variant the bet is about is not what got compiled, and its ms would be the shipped form's a second time."
    echo "CANARY A ok: base $NB | new stage-side $NN | new unstaged8 $NU  (vgpr spill private group, off the LINKED binaries in this window)"

    # ---- NUMERICS: byte identity, OUTSIDE the hold and BEFORE it (a6_g1_bits.sh self-locks) -------------
    # A6-BET owes G1 byte-identical logits and STATEFP. It is deterministic, so it does not belong inside the
    # timing window at all — and it must not be run while the lock is held, because it acquires the lock itself.
    if [ "${SKIP_NUMERICS:-0}" != "1" ]; then
        BASE="$BASE" BIN_NEW="$BIN_NEW" OUT="$OUT/bits" TAG="${TAG}-bits" \
            bash prototype/attn_percost/a6_g1_bits.sh > "$OUT/bits.log" 2>&1 \
            || fail "numerics: a6_g1_bits.sh refused the pair. That is a RED numerics result — G1's premise is that nothing moves. Report it red; do not time a variant whose bits moved."
        grep -E "^(BYTE-IDENTICAL|A6-G1-M1)" "$OUT/bits.log" || fail "numerics: a6_g1_bits.sh exited 0 but printed no BYTE-IDENTICAL line. A check that prints nothing cannot pass."
        echo "NUMERICS ok: logits and STATEFP byte-identical (base vs stage, base vs unstaged8) — see $OUT/bits.log"
    else
        echo "NUMERICS SKIPPED by SKIP_NUMERICS=1: nothing in this window speaks to bit-identity. Say so in the report."
    fi

    ids_file_of "$W" "$OUT/ids$W.csv"
    tools/gpu_lock.sh acquire "$TAG" || exit 1
    echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
    echo "CLK@start: $(clk)"

    # One timed leg: $1 = side (base|stage|u8), $2 = rep.
    run_leg() {
        local SIDE=$1 R=$2
        case $SIDE in
            base)  "$BIN_BASE" --model "$MODEL" --ids "$(cat "$OUT/ids$W.csv")" $ARGS --out "$OUT/${SIDE}_$R" > "$OUT/${SIDE}_$R.log" 2>&1 ;;
            stage) "$BIN_NEW"  --model "$MODEL" --ids "$(cat "$OUT/ids$W.csv")" $ARGS --out "$OUT/${SIDE}_$R" > "$OUT/${SIDE}_$R.log" 2>&1 ;;
            u8)    "$BIN_NEW"  --model "$MODEL" --ids "$(cat "$OUT/ids$W.csv")" $ARGS$QSTAGE_ARG --out "$OUT/${SIDE}_$R" > "$OUT/${SIDE}_$R.log" 2>&1 ;;
        esac || fail "$SIDE rep=$R did not finish"
    }
    banner() { local SIDE=$1 R=$2; for F in q lds wf_per_cu vgpr; do printf "%s=%s " "$F" "$(grep -o "$F=[^ ]*" "$OUT/${SIDE}_$R.log" | head -1 | cut -d= -f2)"; done; }

    for R in $(seq 1 "$REPS"); do
        for SIDE in base stage u8; do sleep 5; run_leg "$SIDE" "$R"; done
        # --- CANARY B: the geometry of the kernel that ACTUALLY RAN, agreeing with its own note block -----
        BB=$(banner base "$R"); BS=$(banner stage "$R"); BU=$(banner u8 "$R")
        need "$(echo "$BB" | grep -o 'lds=[0-9]*')" "the base banner's lds= field"
        need "$(echo "$BU" | grep -o 'wf_per_cu=[0-9]*')" "the u8 banner's wf_per_cu= field"
        echo "$BB" | grep -q "q=stage $(echo "$BAN_STAGE_WANT" | cut -d' ' -f1) $(echo "$BAN_STAGE_WANT" | cut -d' ' -f2) vgpr=$(echo "$NB" | cut -d' ' -f1)" \
            || fail "canary B: rep=$R base banner [$BB], want q=stage $BAN_STAGE_WANT vgpr=$(echo "$NB" | cut -d' ' -f1). The baseline is not the shipped form."
        echo "$BS" | grep -q "q=stage $(echo "$BAN_STAGE_WANT" | cut -d' ' -f1) $(echo "$BAN_STAGE_WANT" | cut -d' ' -f2) vgpr=$(echo "$NN" | cut -d' ' -f1)" \
            || fail "canary B: rep=$R stage banner [$BS], want q=stage $BAN_STAGE_WANT vgpr=$(echo "$NN" | cut -d' ' -f1). The inert side is not running the default kernel, so the NULL leg below is not a NULL."
        echo "$BU" | grep -q "q=unstaged8 $(echo "$BAN_U8_WANT" | cut -d' ' -f1) $(echo "$BAN_U8_WANT" | cut -d' ' -f2) vgpr=$(echo "$NU" | cut -d' ' -f1)" \
            || fail "canary B: rep=$R u8 banner [$BU], want q=unstaged8 $BAN_U8_WANT vgpr=$(echo "$NU" | cut -d' ' -f1) (A6-result.md §1's compile reading). Either the build is stale or the driver refused 20,480 B; in both cases the ms below is not a residency measurement."
        # --- CANARY C: observed launches and the plan agree, and are the SAME on all three sides -----------
        #   (progress/2026-10-21/performance-status-2026-10-21.md §41 closed launch count as a variable; this bet may not reopen it)
        for SIDE in base stage u8; do
            LOG="$OUT/${SIDE}_$R.log"
            GOT=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1)
            PLAN=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* plan_launches=([0-9]+).*/\1/' | tail -1)
            MATCH=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* counts_match=([01]).*/\1/' | tail -1)
            need "$GOT" "the attn launch count on $SIDE"; need "$PLAN" "the attn plan bill on $SIDE"
            [ "$MATCH" = "1" ] || fail "canary C: $LOG counts_match=0 on the attn row: the plan and the submissions disagree."
            [ "$PLAN" = "$GOT" ] || fail "canary C: $LOG plans $PLAN attn launches and submits $GOT."
            grep -q "TOTAL.*plan_match=1" "$LOG" || fail "canary C: plan_match!=1 on $SIDE at rep=$R"
            echo "$GOT" > "$OUT/${SIDE}_$R.launches"
            : > "$OUT/${SIDE}_$R.fields"
            for F in vtiles wave_rows split fattn_rows fallback_rows covered_rows_pct covered_work_pct; do
                grep -o "$F=[^ ]*" "$LOG" | head -1 >> "$OUT/${SIDE}_$R.fields"
            done
            [ "$(wc -l < "$OUT/${SIDE}_$R.fields")" = "7" ] \
                || fail "canary C: $LOG parsed $(wc -l < "$OUT/${SIDE}_$R.fields") of 7 coverage fields. A check over a list that can be empty cannot fail."
        done
        for SIDE in stage u8; do
            cmp -s "$OUT/base_$R.launches" "$OUT/${SIDE}_$R.launches" \
                || fail "canary C: rep=$R $SIDE submits $(cat "$OUT/${SIDE}_$R.launches") attn launches against base's $(cat "$OUT/base_$R.launches"). G1 changes which LDS buffer an operand lives in, not how many launches a layer submits; a launch-count move means something else changed too, and this bet may not move launches (progress/2026-10-21/performance-status-2026-10-21.md §41)."
            cmp -s "$OUT/base_$R.fields" "$OUT/${SIDE}_$R.fields" \
                || fail "canary C: rep=$R $SIDE differs from base in the form's coverage fields: $(paste -d'/' "$OUT/base_$R.fields" "$OUT/${SIDE}_$R.fields" | tr '\n' ' '). The coverage predicate does not read a staging plan, so this is a ROUTING change and the milliseconds below would be comparing different rows."
        done
    done

    # --- the traced leg: the `attention` PHASE row (A6-BET's direction-agreement row) --------------------
    for R in $(seq 1 "$REPS"); do
        for SIDE in base u8; do
            sleep 5
            # The knob is a SEPARATE argv word, expanded unquoted like $TRACE_ARGS: folding it into "$BX"
            # makes rocprofv3 exec a file called "decode_run --attn-fattn-qstage unstaged8", which is an
            # argument-quoting bug wearing the costume of a traced-leg failure.
            BX=$BIN_BASE; KX=""; [ "$SIDE" = u8 ] && { BX="$BIN_NEW"; KX="$QSTAGE_ARG"; }
            "$PY" tools/prefill_kernel_trace.py --waves 8 --out "$OUT/trace_${SIDE}_$R" -- \
                "$BX" --model "$MODEL" --ids "$(cat "$OUT/ids$W.csv")" $TRACE_ARGS $KX \
                --out "$OUT/trace_${SIDE}_${R}_out" > "$OUT/trace_${SIDE}_$R.log" 2>&1 \
                || fail "traced leg: the $SIDE rep=$R traced run did not finish"
            CSV=$(ls "$OUT/trace_${SIDE}_$R"/*_kernel_trace.csv 2>/dev/null | head -1)
            need "$CSV" "kernel_trace.csv for $SIDE rep=$R"
            # The divisor is DERIVED from `attn_prep`, whose per-pass site count is 48 at 4,096 lanes and which
            # G1 does not touch at all. Per the AGENTS rocprofv3 trap row: derive the pass count, never type a
            # number beside it.
            "$PY" tools/trace_phase_split.py "$CSV" --engine ours --calibrate attn_prep --per-pass 48 \
                --tokens "$W" > "$OUT/trace_${SIDE}_$R.txt" 2>&1 \
                || fail "traced leg: trace_phase_split.py refused the $SIDE rep=$R census"
            need "$(traced "$OUT/trace_${SIDE}_$R.txt")" "the attention phase row on $SIDE rep=$R"
        done
    done

    # --- the wide legs, through --ids-file (progress/2026-10-21/performance-status-2026-10-21.md §39) -------------------------------
    for WID in $WIDE_WIDTHS; do
        ids_file_of "$WID" "$OUT/ids$WID.csv"
        for R in $(seq 1 "$REPS_WIDE"); do
            for SIDE in base stage u8; do
                sleep 5
                case $SIDE in
                    base)  "$BIN_BASE" --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS --out "$OUT/wide${WID}_${SIDE}_$R" > "$OUT/wide${WID}_${SIDE}_$R.log" 2>&1 ;;
                    stage) "$BIN_NEW"  --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS --out "$OUT/wide${WID}_${SIDE}_$R" > "$OUT/wide${WID}_${SIDE}_$R.log" 2>&1 ;;
                    u8)    "$BIN_NEW"  --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS$QSTAGE_ARG --out "$OUT/wide${WID}_${SIDE}_$R" > "$OUT/wide${WID}_${SIDE}_$R.log" 2>&1 ;;
                esac || fail "wide leg $WID: $SIDE rep=$R did not finish"
            done
        done
    done
    echo "CLK@end: $(clk)"
    echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
    tools/gpu_lock.sh release "$TAG"
fi

# ================================== SUMMARY: five rows, one verdict ====================================
BA=$(med_of fam base attn); BN=$(med_of fam stage attn); BU=$(med_of fam u8 attn)
BT=$(med_trace base);      TU=$(med_trace u8)
BW=$(med_of tot base);     WN=$(med_of tot stage);    WU=$(med_of tot u8)
need "$BA" "base attn family ms"; need "$BN" "the stage (inert) attn family ms"; need "$BU" "the u8 attn family ms"
need "$BT" "base traced attention ms"; need "$TU" "the u8 traced attention ms"
need "$BW" "base wall_ms at 4,096"; need "$WN" "the stage wall_ms at 4,096"; need "$WU" "the u8 wall_ms at 4,096"
for F in $FAMS; do
    [ "$(med_of fam base "$F")" = "NA" ] && fail "family row $F parsed empty on base. The logs are missing or the field names moved; this is not a measurement."
    [ "$(med_of fam stage "$F")" = "NA" ] && fail "family row $F parsed empty on stage."
    [ "$(med_of fam u8 "$F")" = "NA" ] && fail "family row $F parsed empty on u8."
done
W16B=$(med_wide 16384 base); W16N=$(med_wide 16384 stage); W16U=$(med_wide 16384 u8)
W32B=$(med_wide 32768 base); W32N=$(med_wide 32768 stage); W32U=$(med_wide 32768 u8)
for V in W16B W16N W16U W32B W32N W32U; do need "${!V}" "the wide ms at ${V:1:5} (${V})"; done

# The window's own tightness, and the NULL leg. G1 HAS an inert side, so A4's NULL returns: `stage` is the new
# binary with the knob unnamed, and any move it makes is either the binary differing from base or the window being
# unpaired. Either way there is no verdict to print.
NULL_FAM=$(pct_move "$BA" "$BN"); NULL_W=$(pct_move "$BW" "$WN")
NULL_16=$(pct_move "$W16B" "$W16N"); NULL_32=$(pct_move "$W32B" "$W32N")
for V in NULL_FAM NULL_W NULL_16 NULL_32; do need "${!V}" "the NULL agreement on $V"; done
UNPAIRED=$(awk -v a="$NULL_FAM" -v w="$NULL_W" -v w16="$NULL_16" -v w32="$NULL_32" -v tol="$NULL_TOL" \
    'BEGIN{ split(a" "w" "w16" "w32, v, " "); mx=0; for (i=1;i<=4;i++){ x=v[i]+0; if(x<0)x=-x; if(x>mx)mx=x }
      print (mx>tol)?"UNPAIRED":"paired" }')
SP_B=$(spread_of fam base attn); SP_N=$(spread_of fam stage attn); SP_U=$(spread_of fam u8 attn)
need "$SP_B" "the base rep spread"; need "$SP_U" "the u8 rep spread"
WIDE_SPREAD=$(awk -v a="$SP_B" -v b="$SP_N" -v c="$SP_U" -v tol="$SPREAD_TOL" 'BEGIN{print (a>tol||b>tol||c>tol)?1:0}')

echo
echo "== A6 G1 — Q un-staged + the 8-row epilogue, behind --attn-fattn-qstage. $REPS reps at $W lanes"
echo "   (untraced + traced), $REPS_WIDE reps at [$WIDE_WIDTHS] lanes, all inside one window. base = $BASE,"
echo "   stage = the new binary with the knob UNSET (the NULL), u8 = the same binary with unstaged8."
echo "   Lines are A6-BET's G1 ratios over the base measured IN THIS WINDOW."
printf "   NULL tolerance %s%%  |  rep-spread tolerance %s%%\n" "$NULL_TOL" "$SPREAD_TOL"

printf "\n%-16s %10s %10s %10s %14s\n" family base stage u8 share_of_wave
for F in $FAMS; do
    printf "%-16s %10s %10s %10s %14s\n" "$F" "$(med_of fam base "$F")" "$(med_of fam stage "$F")" \
        "$(med_of fam u8 "$F")" "$(grep -h "family=$F " "$OUT/base_1.log" | sed -E 's/.*share_pct=([0-9.]+).*/\1/' | tail -1)"
done

# Externality: sum of (u8 - base) over every family OTHER than attn. Informational, and printed against G0.
ext_of() { local WID=$1 E=0 F fb fu
    for F in $FAMS; do
        [ "$F" = attn ] && continue
        if [ "$WID" = "$W" ]; then fb=$(med_of fam base "$F"); fu=$(med_of fam u8 "$F")
        else fb=$(med_wide_fam "$WID" base "$F"); fu=$(med_wide_fam "$WID" u8 "$F"); fi
        { [ -z "$fb" ] || [ -z "$fu" ] || [ "$fb" = "NA" ] || [ "$fu" = "NA" ]; } && continue
        E=$(awk -v e="$E" -v b="$fb" -v u="$fu" 'BEGIN{printf "%.3f", e + (u - b)}')
    done
    echo "$E"
}
E4=$(ext_of "$W"); E16=$(ext_of 16384); E32=$(ext_of 32768)

printf "\n%-26s %10s %10s %10s\n" "attn family, untraced" "$BA" "$BN" "$BU"
printf "%-26s %10s %10s %10s\n" "attention phase, traced" "$BT" "-" "$TU"
printf "%-26s %10s %10s %10s\n" "wave wall_ms @4,096" "$BW" "$WN" "$WU"
for WID in 16384 32768; do
    BB=$(med_wide "$WID" base); BNW=$(med_wide "$WID" stage); BUW=$(med_wide "$WID" u8)
    printf "%-26s %10s %10s %10s\n" "wave ms @${WID}" "$BB" "$BNW" "$BUW"
done
printf "\nNULL (new default vs base): family %s%%  wall@4096 %s%%  wall@16384 %s%%  wall@32768 %s%%  -> %s\n" \
    "$NULL_FAM" "$NULL_W" "$NULL_16" "$NULL_32" "$UNPAIRED"
awk -v a="$SP_B" -v b="$SP_N" -v c="$SP_U" -v tol="$SPREAD_TOL" \
    'BEGIN{ w=(a>tol||b>tol||c>tol)?"SPREAD TOO WIDE":"ok";
      printf "WINDOW %s: attn family rep spread base %.2f%%, stage %.2f%%, u8 %.2f%% (tolerance %s%%)\n", w, a, b, c, tol }'

if [ "$UNPAIRED" = "UNPAIRED" ]; then
    echo "== VERDICT for G1: UNPAIRED — the NULL leg moved by more than ${NULL_TOL}% on the family row or on a ship gate, so the two binaries are not comparable in this window. Report the NULL row, re-take; do NOT read a verdict from this window."
    exit 1
fi
[ "$WIDE_SPREAD" = "1" ] && fail "window spread above ${SPREAD_TOL}% on the attn family (base $SP_B stage $SP_N u8 $SP_U). Treat this window as unpaired and re-take; do not read a verdict from it."

awk -v a="$BA" -v u="$BU" -v at="$BT" -v tu="$TU" -v w="$BW" -v wu="$WU" -v w16="$W16B" -v w16u="$W16U" \
    -v w32="$W32B" -v w32u="$W32U" -v e4="$E4" -v e16="$E16" -v e32="$E32" -v eg0="$EXT_G0_16" \
    -v ra_l="$RAT_A_LIVE" -v ra_d="$RAT_A_DEAD" -v rat_l="$RAT_AT_LIVE" -v rat_d="$RAT_AT_DEAD" \
    -v rw_l="$RAT_W_LIVE" -v rw_d="$RAT_W_DEAD" -v r16_l="$RAT_16_LIVE" -v r16_d="$RAT_16_DEAD" \
    -v r32_l="$RAT_32_LIVE" -v r32_d="$RAT_32_DEAD" 'BEGIN{
    fa = u / a; ft = tu / at; fw = wu / w; f16 = w16u / w16; f32 = w32u / w32;
    # LIVE requires EVERY pre-registered line cleared: the family row, all three ship gates, and the traced row
    # improving at least to its own live line (A6-BET: it "must agree in direction with the family row").
    # DEAD is the family row or a ship gate past its dead line. The traced row cannot decide DEAD on its own —
    # a worse traced row beside a clearing family is a disagreement to report, which is UNRESOLVED by construction.
    live = (fa <= ra_l) && (fw <= rw_l) && (f16 <= r16_l) && (f32 <= r32_l) && (ft <= rat_l);
    dead = (fa >  ra_d) || (fw >  rw_d) || (f16 >  r16_d) || (f32 >  r32_d);
    v = "UNRESOLVED"; if (dead) v = "DEAD"; else if (live) v = "LIVE";
    why = "";
    if (fa > ra_l && fa <= ra_d) why = why " family row between the lines;";
    if (ft > rat_l) why = why " traced row does not clear its live line (direction disagreement);";
    if (fw > rw_l && fw <= rw_d) why = why " wave wall@4096 short of its live line;";
    if (f16 > r16_l && f16 <= r16_d) why = why " wave ms@16384 short of its live line;";
    if (f32 > r32_l && f32 <= r32_d) why = why " wave ms@32768 short of its live line;";
    printf "\n== VERDICT for G1: %s\n", v;
    printf "   attn family (THE ROW)   : %.2f -> %.2f ms  (%+.2f %% ms)  ratio %.4f   live <= %s, dead > %s\n",
           a, u, 100*(u-a)/a, fa, ra_l, ra_d;
    printf "   attention traced        : %.2f -> %.2f ms  (%+.2f %%)   ratio %.4f   live <= %s, dead > %s  [direction agreement required for LIVE]\n",
           at, tu, 100*(tu-at)/at, ft, rat_l, rat_d;
    printf "   wave wall_ms @4,096     : %.2f -> %.2f ms  (%+.2f %%)   ratio %.4f   live <= %s, dead > %s  [SHIP GATE]\n",
           w, wu, 100*(wu-w)/w, fw, rw_l, rw_d;
    printf "   wave ms @16,384         : %.2f -> %.2f ms  (%+.2f %%)   ratio %.4f   live <= %s, dead > %s  [SHIP GATE]\n",
           w16, w16u, 100*(w16u-w16)/w16, f16, r16_l, r16_d;
    printf "   wave ms @32,768         : %.2f -> %.2f ms  (%+.2f %%)   ratio %.4f   live <= %s, dead > %s  [SHIP GATE]\n",
           w32, w32u, 100*(w32u-w32)/w32, f32, r32_l, r32_d;
    printf "   externality ms          : %+.2f / %+.2f / %+.2f  (4,096 / 16,384 / 32,768)   [informational, decides nothing]; G0 read %+.2f at 16,384\n",
           e4, e16, e32, eg0;
    if (e16 > 20) printf "   EXTERNALITY NOTE: +%.2f ms at 16,384 is the same shape G0 measured (+%.2f) — neighbours slowed, not the attention family. G1 keeps 96-thread blocks and only drops LDS, so this is a co-residency finding worth its own line, not noise.\n", e16, eg0;
    if (dead) printf "   -> DEAD at A6-BET G1 line as pre-registered: not shipped. Say which row missed.\n";
    else if (live) printf "   -> LIVE: the family row and all three ship gates cleared, traced row agreeing. Ship it (the default stays `stage`; adopting unstaged8 is a separate commit with its own gate echo).\n";
    else printf "   -> UNRESOLVED:%s between live and dead, or a ship gate short of its line. Not shipped, not narrated — the same rule A6-BET pre-registered for this bet.\n", why;
}'
echo
echo "Also report with this: the two note-block readings (CANARY A), the three banners (CANARY B: q=/lds=/"
echo "wf_per_cu=/vgpr=), the NULL row and the rep spreads, the clocks, the traced census files, the externality"
echo "against G0's $EXT_G0_16 ms at 16,384, and — unless SKIP_NUMERICS=1 — the byte-identity result from"
echo "$OUT/bits.log. The shipping-tree gate is .venv/bin/python engine/gate.py --attn-fattn-qstage unstaged8 on"
echo "the changed tree (argv passthrough; the default side is what gate.py already judges). DONE. logs in $OUT/"
