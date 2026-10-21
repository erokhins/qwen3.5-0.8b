#!/usr/bin/env bash
# run_ab_a9lc.sh — the device A/B that adjudicates `prototype/attn_percost/A9-BET.md` for L-c: cap the attention
# split (`--attn-split-cap N`) so the split's partial round-trip stops being the largest measured byte in the
# long-context attention phase.
#
# MAIN-AGENT DEVICE WINDOW ONLY. Blocking, zero subagents, ONE `tools/gpu_lock.sh` hold for the timed legs.
# This harness self-locks, so it must never be run from inside another hold, and it must not call a self-locking
# script (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`, engine/gate.py)
# while holding one — AGENTS' deadlock row.
#
# WHAT IS BEING DECIDED — A9-BET's two pre-registered lines, both at the deciding width, fixed before measurement:
#   LINE 1  attn family ms (untraced)   ratio <= 0.9700
#   LINE 2  wave TOTAL wall_ms          ratio <= 0.9700   [SHIP GATE]
#   NULL arm: the changed binary with the cap UNNAMED against BASE, both lines |move| <= 2 %.
# The verdict, stated here so it cannot be re-sized afterwards:
#   SHIP        family <= 0.97 AND wall <= 0.97
#   FALSIFIED   family >  0.97 — the byte mechanism did not deliver, and A9 §2.3's "attn_combine reads the
#               partials back at 88 % of the DRAM roof" reading is retired for this kernel (A9 §4.2's falsifier)
#   DISPLACED   family <= 0.97 but wall > 0.97 — the win is real inside the attention family and did not survive
#               into the wave, i.e. it landed in a neighbour. Not shipped. This is the whole reason A9-BET
#               pre-registers TWO lines: a single family row can be fooled by cost moving next door (AGENTS).
#   UNPAIRED    the NULL arm moved: the two binaries are not comparable in this window and no verdict is printed.
#
# THE SIDES. `base` is a `git worktree` at the pinned PRE-change sha, built as-is; `cap` is the changed tree's
# binary run with `--attn-split-cap $CAP`; `null` is that same binary with the flag unnamed, and it exists only
# to prove the window is paired. One binary for two arms is the point: both geometries are in it, so the
# comparison is one window and not two rulers re-taken in different weather (AGENTS).
#
# WHAT IT REFUSES TO DO:
#   * it does not time a binary it has not identified. CANARY 1 requires the changed binary to know the flag and
#     the BASE binary NOT to know it. A base that accepts `--attn-split-cap` is not the pre-change tree; a new
#     binary that rejects it is a stale build — and a byte-compare cannot say either, because `engine/build.sh`
#     is not bit-reproducible on this box (AGENTS 2026-10-17).
#   * it does not time a run that has not shown the behaviour. CANARY 2 reads, from the logs that produced the
#     milliseconds: base prints NO `SPLITCAP` line and its `ATTN_FATTN split=` is the geometry's own; the capped
#     arm prints `SPLITCAP cap= split= natural= span=` agreeing with the harness's own `$CAP`, with `span` above
#     `MAXSC` and with its own banner; the null arm prints the same geometry as base. A cap that did not bind
#     times the uncapped run twice and reports the second price as the cap's win.
#   * it does not pass an empty parse. Every field is `need`-checked, the family table must parse one row per name
#     in `$FAMS`, and a missing `SPLITCAP` record is an abort — a check over a list that can be empty cannot fail.
#   * it does not take a wide prompt through argv. One argv slot caps at MAX_ARG_STRLEN = 131,072 B, so 32,768 ids
#     cannot be passed at all (progress/2026-10-21/performance-status-2026-10-21.md §39: shell exit 126 in 0 s, read as "the engine crashes
#     at 32k"). Wide legs go via `--ids-file`, and the file's field count is asserted to equal the width.
#   * it does not print a verdict from an unpaired window, nor from reps that disagree by more than SPREAD_TOL,
#     nor from a run whose attention launch count differs between arms (progress/2026-10-21/performance-status-2026-10-21.md §41 closed
#     launch count as a variable and this lever may not reopen it).
#
# Usage:
#   bash prototype/attn_percost/run_ab_a9lc.sh                      # the window (main agent, blocking)
#   BASE=8b0a3b4 CAP=8 bash prototype/attn_percost/run_ab_a9lc.sh   # re-pin the baseline / the candidate
#   NEWDIR=/path/to/changed/tree bash prototype/attn_percost/run_ab_a9lc.sh
#   REPS=3 bash prototype/attn_percost/run_ab_a9lc.sh
#   WIDTHS="16384 32768" bash prototype/attn_percost/run_ab_a9lc.sh # second width; DECIDE still decides
#   DRY_RUN=1  bash prototype/attn_percost/run_ab_a9lc.sh           # print the plan; build nothing, no device
#   SELFTEST=1 bash prototype/attn_percost/run_ab_a9lc.sh           # colour-test the parser + canaries, no device
set -uo pipefail
export LC_ALL=C          # a decimal comma in a machine-read number is a parse bug wearing a locale's clothes
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD

# ---- pins ----------------------------------------------------------------------------
# A PRE-change sha, never HEAD: after this knob lands HEAD *is* the change, so a baseline read from HEAD would
# compare the lever with itself and print a clean null (the §13 trap). This is the sha the tree shipped at when
# A9-L-c was written (progress/2026-10-21/performance-status-2026-10-21.md §52's funding decision).
BASE=${BASE:-8b0a3b4}
SHORT=$(git rev-parse --short "$BASE" 2>/dev/null || echo "$BASE")
BASEDIR=${BASEDIR:-/tmp/a9lc-base-$SHORT}
NEWDIR=${NEWDIR:-$REPO}                        # the changed tree; a worktree of it works too
BIN_BASE=$BASEDIR/engine/build/decode_run
BIN_NEW=$NEWDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

CAP=${CAP:-8}                                  # the candidate A9-BET funds; every colour below reads $CAP
WIDTHS=${WIDTHS:-"32768"}                     # the widths to run
DECIDE=${DECIDE:-32768}                       # the width the verdict is read at
REPS=${REPS:-2}
TAG=${TAG:-a9lc-ab}
OUT=${OUT:-tmp/attn_percost/a9lcab}
mkdir -p "$OUT"
# Absolute BEFORE anything `cd`s: the build leg enters the base worktree and redirects into $OUT, and a relative
# log path resolves inside the worktree, where tmp/ does not exist; the build then fails for a bug that has
# nothing to do with the code under test (the f736e27 lesson).
OUT="$(cd "$OUT" && pwd)"

# ---- the lines: A9-BET's bar, nothing else -------------------------------------------------------------
RAT_LIVE=${RAT_LIVE:-0.9700}                   # -3.0 %, on the family row AND on the wave wall
NULL_TOL=${NULL_TOL:-2.0}                      # %: how far the inert arm may move before the window is unpaired
SPREAD_TOL=${SPREAD_TOL:-3.0}                  # %: per-rep spread tolerated inside one window
MAXSC=${MAXSC:-512}                            # `prototype/attn_decode/attn_decode_kernels.h:32`, for canary 2
NATURAL32=${NATURAL32:-64}                     # split_for(32768): base's own split at the deciding width
FAMS="gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head host_tail"
ARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --wave-rows 512"
CAPARG=" --attn-split-cap $CAP"

fam()    { grep -h "WAVE gran=stage family=$1 " "$2" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
tot()    { grep -h "WAVE gran=stage TOTAL" "$1" | sed -E 's/.*wall_ms=([0-9.]+).*/\1/' | tail -1; }
median() { sort -g "$@" | awk '{a[NR]=$1} END{if(NR==0){print "NA";exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
fail()   { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
need()   { [ -n "$1" ] || fail "empty parse where a value was required ($2). A verdict computed from nothing is the empty-parse failure AGENTS.md records; not a measurement."; }
pct_move() { awk -v b="$1" -v n="$2" 'BEGIN{ if (b==0||b=="NA"||n=="NA"){print "NA"} else printf "%.2f", 100*(n-b)/b }'; }
LOGOF()  { echo "$OUT/wide$1_$2_$3.log"; }              # width side rep
med_wide()      { local WID=$1 side=$2; for R in $(seq 1 "$REPS"); do tot "$OUT/wide${WID}_${side}_$R.log"; done | grep -v '^$' | median -; }
med_wide_fam()  { local WID=$1 side=$2 F=$3; for R in $(seq 1 "$REPS"); do grep -h "family=$F " "$OUT/wide${WID}_${side}_$R.log" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; done | grep -v '^$' | median -; }
spread_of()     { local WID=$1 side=$2 vals med lo hi
    vals=$(for R in $(seq 1 "$REPS"); do fam attn "$OUT/wide${WID}_${side}_$R.log"; done | grep -v '^$')
    med=$(echo "$vals" | median -); lo=$(echo "$vals" | sort -g | head -1); hi=$(echo "$vals" | sort -g | tail -1)
    awk -v m="$med" -v l="$lo" -v h="$hi" 'BEGIN{ if (m==0||m=="NA"){print "NA"} else printf "%.2f", 100*(h-l)/m }'
}
# The canary fields, read out of one log. A missing record is an abort, never a silent empty string.
splitcap_of()    { grep -h "^SPLITCAP " "$1" 2>/dev/null | tail -1; }
splitcap_field() { splitcap_of "$1" | tr ' ' '\n' | sed -n "s/^$2=//p"; }
# The LAST banner, not the first: a 32,768-lane prefill is 64 bodies and the deduped `ATTN_FATTN`/`SPLITCAP`
# lines print one per distinct geometry, so the first is body 1 (split 2, where no cap can bind) and the last is
# the widest body (split 64, where cap 8 does). Comparing the first would compare a geometry the cap cannot reach.
ban_split()      { grep -h "^ATTN_FATTN " "$1" 2>/dev/null | tail -1 | tr ' ' '\n' | sed -n 's/^split=//p'; }
launches_of()    { grep -h "WAVE gran=stage family=attn " "$1" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1; }
plan_of()        { grep -h "WAVE gran=stage family=attn " "$1" | sed -E 's/.* plan_launches=([0-9]+).*/\1/' | tail -1; }

ids_file_of() { local N=$1 DST=$2
    # awk, not a python heredoc: the generator must be inspectable in the log and must not depend on a stdin
    # consumer. The seed is `longctx_check.sh`'s so wide rows stay comparable across harnesses; the sequence
    # itself does not have to match python's, only be the same sequence on both arms — which the field-count
    # assert below plus one shared file guarantees.
    awk -v n="$N" 'BEGIN{srand(20261019); for (i=0;i<n;i++) printf "%s%d", (i?",":""), 1000+int(rand()*59000); print ""}' > "$DST"
    local have; have=$("$PY" -c "import sys;print(len(open(sys.argv[1]).read().split(',')))" "$DST")
    [ "$have" = "$N" ] || fail "width $N needs $N ids and $DST holds $have — refusing to judge a run whose prompt is not the width under test"
}

# ---- CANARY 2: the behaviour, asserted from the logs that produced the numbers -------------------------
# Split out so SELFTEST can colour-test it against fixtures; a canary nobody can test is a comment.
canary2() {   # $1 = width, $2 = rep
    local WID=$1 R=$2
    local LB LC LN BS CS CC CN CP CL NS
    LB=$(LOGOF "$WID" base "$R"); LC=$(LOGOF "$WID" cap "$R"); LN=$(LOGOF "$WID" null "$R")
    need "$(fam attn "$LB")" "the base attn family row at $WID"
    need "$(fam attn "$LC")" "the capped attn family row at $WID"
    # base: no SPLITCAP line at all (the pre-change binary cannot print one), banner shows the natural split.
    [ -z "$(splitcap_of "$LB")" ] \
        || fail "canary 2: the BASE log $LB prints a SPLITCAP line. That is the changed binary wearing the baseline's name, and every ms below is the same arm twice."
    BS=$(ban_split "$LB"); need "$BS" "the base ATTN_FATTN split= field at $WID"
    [ "$BS" -gt "$CAP" ] || fail "canary 2: base's own split at $WID lanes is $BS, which is not above the cap $CAP, so there is nothing to cap at this width. Raise the width (natural split = ceil(lanes/$MAXSC))."
    # cap: the line exists and its fields are the harness's constants.
    local SC; SC=$(splitcap_of "$LC")
    need "$SC" "the capped arm's SPLITCAP line at $WID. Zero records parsed: the knob is absent from this binary, the run died before the attention launcher, or the flag did not reach it."
    CS=$(splitcap_field "$LC" split); CC=$(splitcap_field "$LC" cap)
    CN=$(splitcap_field "$LC" natural); CP=$(splitcap_field "$LC" span); CL=$(splitcap_field "$LC" lanes)
    need "$CS" "SPLITCAP split="; need "$CC" "SPLITCAP cap="; need "$CN" "SPLITCAP natural="
    need "$CP" "SPLITCAP span="; need "$CL" "SPLITCAP lanes="
    [ "$CC" = "$CAP" ] || fail "canary 2: the capped arm reports cap=$CC, want $CAP — the flag did not reach the launcher."
    [ "$CS" = "$CAP" ] || fail "canary 2: the capped arm reports split=$CS with cap=$CAP at $CL lanes. The cap did not bind (it must give split = min(natural,cap) = $CAP), so both arms ran one geometry and the ms below compare a run with itself."
    [ "$CN" -gt "$CAP" ] || fail "canary 2: natural=$CN is not above the cap $CAP at $CL lanes — nothing to cap here."
    [ "$CP" -gt "$MAXSC" ] || fail "canary 2: span=$CP does not exceed MAXSC=$MAXSC, so this cap changed no block geometry and the timing would measure nothing."
    [ "$(ban_split "$LC")" = "$CS" ] || fail "canary 2: the capped arm's ATTN_FATTN split=$(ban_split "$LC") disagrees with its SPLITCAP split=$CS. Two spellings of one number may not differ (§20.1's class)."
    [ "$CL" -le "$WID" ] || fail "canary 2: the capped arm's widest row is $CL lanes at a $WID-lane prompt; the prompt is not the width under test."
    # null: the changed binary's DEFAULT must reproduce base's geometry exactly — the inertness claim, read
    # off the geometry rather than asserted in a comment.
    NS=$(splitcap_field "$LN" split)
    [ -n "$NS" ] || fail "canary 2: the NULL arm printed no SPLITCAP line ($LN): the changed binary does not echo the knob at all, so the inert-default claim is uncheckable."
    [ "$NS" = "$BS" ] || fail "canary 2: the NULL arm's split is $NS against base's $BS. The default moved the geometry, so there is no inert side and this window cannot pair."
    # The launch plan must not have moved: a cap changes a grid EXTENT, not how many launches a layer submits.
    local SIDE G P
    for SIDE in base cap null; do
        local L; L=$(LOGOF "$WID" "$SIDE" "$R")
        G=$(launches_of "$L"); P=$(plan_of "$L")
        need "$G" "the attn launch count on $SIDE at $WID"; need "$P" "the attn plan bill on $SIDE at $WID"
        [ "$G" = "$P" ] || fail "canary 2: $L plans $P attn launches and submits $G."
        grep -q "TOTAL.*plan_match=1" "$L" || fail "canary 2: plan_match != 1 on $SIDE at $WID rep=$R"
        echo "$G" > "$OUT/wide${WID}_${SIDE}_$R.launches"
    done
    cmp -s "$OUT/wide${WID}_base_$R.launches" "$OUT/wide${WID}_cap_$R.launches" \
        || fail "canary 2: at $WID lanes the capped arm submits $(cat "$OUT/wide${WID}_cap_$R.launches") attn launches against base's $(cat "$OUT/wide${WID}_base_$R.launches"). Capping the split changes a grid extent and the bytes of a partial, not how many launches a layer submits (progress/2026-10-21/performance-status-2026-10-21.md §41)."
    echo "CANARY 2 ok @${WID}: base split=$BS (no SPLITCAP) | cap split=$CS cap=$CC natural=$CN span=$CP | null split=$NS"
}

# ============================ DRY_RUN: print the plan, touch nothing ==============================
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN — nothing executed, no lock taken, no binary built."
    echo "  base   : $BASE -> $BIN_BASE   (worktree at $BASEDIR, built if absent)"
    echo "  cap    : $NEWDIR -> $BIN_NEW  run with --attn-split-cap $CAP"
    echo "  null   : the same binary, cap UNNAMED (the pairing check)"
    echo "  widths : [$WIDTHS]; the verdict is read at $DECIDE; REPS=$REPS"
    echo "  lines  : LINE 1 attn family ms, LINE 2 wave TOTAL wall_ms, both at $DECIDE; live ratio <= $RAT_LIVE"
    for WID in $WIDTHS; do
        echo "    $WID base: $BIN_BASE --model $MODEL --ids-file $OUT/ids$WID.csv --cap $WID $ARGS --out $OUT/wide${WID}_base_R"
        echo "    $WID cap : $BIN_NEW --model $MODEL --ids-file $OUT/ids$WID.csv --cap $WID $ARGS$CAPARG --out $OUT/wide${WID}_cap_R"
        echo "    $WID null: $BIN_NEW --model $MODEL --ids-file $OUT/ids$WID.csv --cap $WID $ARGS --out $OUT/wide${WID}_null_R"
    done
    echo "  canaries: 1) the flag is known to BIN_NEW and unknown to BIN_BASE."
    echo "            2) base log: no SPLITCAP line, ATTN_FATTN split=$NATURAL32 at 32768."
    echo "               cap  log: SPLITCAP cap=$CAP split=$CAP natural>$CAP span>$MAXSC, banner agrees."
    echo "               null log: SPLITCAP split == base's split (the default is inert on the geometry)."
    echo "               attn launches identical on all three; plan_match=1 everywhere."
    echo "  refuses: a base that knows the flag or a new binary that does not; a cap that did not bind; an"
    echo "           empty parse (including zero SPLITCAP records on the capped arm); a wide prompt whose id count"
    echo "           is not the width; a rep spread > $SPREAD_TOL %; an unpaired NULL; a launch-count difference."
    echo "  verdict: SHIP <= $RAT_LIVE on BOTH lines; FALSIFIED if family > $RAT_LIVE; DISPLACED if the family"
    echo "           clears and the wall does not; UNPAIRED prints no verdict at all."
    echo "  logs   : $OUT/   (lock tag $TAG)"
    exit 0
fi

# ============================ SELFTEST: colour-test the parser and the canary ==============================
# Fixtures are GENERATED from the ratio constants in this file, never hand-typed, and one scenario (displaced)
# is built so a parser that read only the family row would call it a ship. `nocanary` proves the canary aborts.
if [ "${SELFTEST:-0}" = "1" ]; then
    mk_fixture() {   # $1 dir, $2 base attn family ms, $3 cap family ratio, $4 cap wall ratio,
                     # $5 null family ratio, $6 null wall ratio, $7 emit SPLITCAP on the cap arm (1/0)
        local D=$1 FB=$2 RC=$3 RW=$4 RN=$5 RWL=$6 SC=$7 SIDE FM WM SP
        for SIDE in base cap null; do
            FM=$(awk -v f="$FB" 'BEGIN{printf "%.4f", f}')
            WM=$(awk 'BEGIN{printf "%.4f", 4000}')
            [ "$SIDE" = cap ]  && FM=$(awk -v f="$FB" -v r="$RC" 'BEGIN{printf "%.4f", f*r}')
            [ "$SIDE" = cap ]  && WM=$(awk -v r="$RW" 'BEGIN{printf "%.4f", 4000*r}')
            [ "$SIDE" = null ] && FM=$(awk -v f="$FB" -v r="$RN" 'BEGIN{printf "%.4f", f*r}')
            [ "$SIDE" = null ] && WM=$(awk -v r="$RWL" 'BEGIN{printf "%.4f", 4000*r}')
            SP=$NATURAL32; [ "$SIDE" = cap ] && SP=$CAP
            {
                local Fx
                for Fx in $FAMS; do
                    V=10.0; [ "$Fx" = attn ] && V=$FM
                    echo "WAVE gran=stage family=$Fx ms=$V launches=6 plan_launches=6 counts_match=1 share_pct=1.0"
                done
                echo "WAVE gran=stage TOTAL wall_ms=$WM plan_match=1"
                echo "ATTN_FATTN form=fattn16 q=unstaged8 roll=single warps=3 vgpr=185 lds=20480 scratch=0 vtiles=16 wf_per_cu=9 wave_rows=512 split=$SP fattn_rows=512/512 fallback_rows=0 covered_rows_pct=100.0 covered_work_pct=100.0"
                if [ "$SIDE" = base ]; then :; elif [ "$SIDE" = cap ] && [ "$SC" = 1 ]; then
                    echo "SPLITCAP split=$CAP cap=$CAP lanes=$DECIDE natural=$NATURAL32 rows=512 span=$(( (DECIDE + CAP - 1) / CAP ))"
                elif [ "$SIDE" = null ]; then
                    echo "SPLITCAP split=$NATURAL32 cap=0 lanes=$DECIDE natural=$NATURAL32 rows=512 span=$(( (DECIDE + NATURAL32 - 1) / NATURAL32 ))"
                fi
            } > "$D/wide${DECIDE}_${SIDE}_1.log"
        done
    }
    for SCEN in ship falsified displaced unpaired nocanary; do
        case $SCEN in
            ship)      FB=1000 RC=0.90 RW=0.91 RN=1.000 RWL=1.000 SC=1 WANT=SHIP ;;
            falsified) FB=1000 RC=1.02 RW=1.02 RN=1.000 RWL=1.000 SC=1 WANT=FALSIFIED ;;
            displaced) FB=1000 RC=0.94 RW=1.01 RN=1.000 RWL=1.000 SC=1 WANT=DISPLACED ;;
            unpaired)  FB=1000 RC=0.90 RW=0.90 RN=1.000 RWL=1.060 SC=1 WANT=UNPAIRED ;;
            nocanary)  FB=1000 RC=0.90 RW=0.90 RN=1.000 RWL=1.000 SC=0 WANT=ABORT ;;
        esac
        D="$OUT/selftest_$SCEN"; mkdir -p "$D"
        mk_fixture "$D" "$FB" "$RC" "$RW" "$RN" "$RWL" "$SC"
        ENVF="OUT=$D REPS=1 DECIDE=$DECIDE CAP=$CAP RAT_LIVE=$RAT_LIVE NULL_TOL=$NULL_TOL SPREAD_TOL=$SPREAD_TOL FAMS=$FAMS WIDTHS=$DECIDE"
        # (a) the canary, run over the same fixtures the verdict will be read from.
        COUT=$(env -u SELFTEST OUT="$D" REPS=1 DECIDE="$DECIDE" CAP="$CAP" MAXSC="$MAXSC" SKIP_DEVICE=1 CANARY_ONLY=1 \
                  bash "$0" 2>&1); CRC=$?
        if [ "$SCEN" = nocanary ]; then
            [ "$CRC" -ne 0 ] && echo "$COUT" | grep -q "^ABORT" \
                || fail "selftest: scenario $SCEN printed '${COUT}', expected an ABORT from the canary. A canary that stays silent when the capped arm prints no SPLITCAP line is a comment, not a control."
            echo "SELFTEST ok: scenario $SCEN -> ABORT (the canary fired)"; continue
        fi
        [ "$CRC" = 0 ] || fail "selftest: scenario $SCEN's canary aborted: $COUT"
        # (b) the verdict, from the same logs.
        GOT=$(env -u SELFTEST OUT="$D" REPS=1 DECIDE="$DECIDE" CAP="$CAP" MAXSC="$MAXSC" SKIP_DEVICE=1 bash "$0" 2>&1 \
              | sed -n 's/^== VERDICT for A9-L-c[^:]*: \([A-Z]*\).*/\1/p' | tail -1)
        [ "$GOT" = "$WANT" ] || fail "selftest: scenario $SCEN printed '${GOT:-<nothing>}', expected $WANT. The verdict parser is not reading its own table; no real window may be run."
        echo "SELFTEST ok: scenario $SCEN -> ${WANT} (canary ok)"
    done
    echo "SELFTEST PASS: the parser distinguishes SHIP / FALSIFIED / DISPLACED (family clears, wall does not) /"
    echo "UNPAIRED, and the canary aborts when the capped arm prints no SPLITCAP line. This says nothing about"
    echo "the engine."
    exit 0
fi

# The canary alone, over whatever logs are in $OUT — the hook SELFTEST uses to colour-test it, and the hook a
# reviewer uses to re-check a taken window without re-running the device.
if [ "${CANARY_ONLY:-0}" = "1" ]; then
    for WID in $WIDTHS; do canary2 "$WID" 1; done
    exit 0
fi

# ================================== THE WINDOW ==================================
# `SKIP_DEVICE=1` judges the logs already in $OUT and touches nothing — the reviewer's re-summarise path, and
# what SELFTEST uses so the parser can be colour-tested without a device or a lock.
if [ "${SKIP_DEVICE:-0}" != "1" ]; then
[ -x "$BIN_NEW" ] || fail "no changed-tree binary at $BIN_NEW — run engine/build.sh first"
[ -d "$BASEDIR" ] || git worktree add "$BASEDIR" "$BASE" --detach -q || fail "worktree add failed for $BASE"
for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
[ -x "$BIN_BASE" ] || { echo "building base $BASE into $BASEDIR (CPU only)"; ( cd "$BASEDIR" && bash engine/build.sh >"$OUT/build_base.log" 2>&1 ) || fail "baseline build failed (see $OUT/build_base.log)"; }
[ "$BIN_BASE" -nt "$BASEDIR/engine/kernels_attn.cpp" ] || fail "the base binary is older than the source it came from"
[ "$BIN_NEW"  -nt "$NEWDIR/engine/kernels_attn.cpp"   ] || fail "the new binary is older than the changed source it came from"

# ---- CANARY 1: the flag exists on one side and not the other ------------------------------------------
# The usage text is captured into a variable rather than piped into `grep -q`: this script runs with
# `pipefail`, and a big --help closing the pipe early makes the WRITER die of SIGPIPE, which pipefail reports as
# a failed pipeline even when the match was found. A canary that fails on the baseline's own success would be
# worse than no canary.
HELP_NEW=$("$BIN_NEW" --help 2>&1 || true)
HELP_BASE=$("$BIN_BASE" --help 2>&1 || true)
case $HELP_NEW in
    *--attn-split-cap*) ;;
    *) fail "canary 1: the changed-tree binary's usage has no --attn-split-cap. That binary predates the knob; a window run on it times the lever's absence and calls it the baseline." ;;
esac
case $HELP_BASE in
    *--attn-split-cap*) fail "canary 1: the BASE binary at $BASE already knows --attn-split-cap, so it is not the pre-change tree and this window has no baseline. Re-pin BASE to a sha before the knob landed." ;;
esac
echo "CANARY 1 ok: the flag is known to $BIN_NEW and unknown to $BIN_BASE (base pinned at $BASE)"

for WID in $WIDTHS; do ids_file_of "$WID" "$OUT/ids$WID.csv"; done

tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
echo "CLK@start: $(clk)"

for R in $(seq 1 "$REPS"); do
    for WID in $WIDTHS; do
        for SIDE in base cap null; do
            sleep 5
            LOG=$(LOGOF "$WID" "$SIDE" "$R")
            case $SIDE in
                base) "$BIN_BASE" --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS \
                          --out "$OUT/wide${WID}_base_$R" > "$LOG" 2>&1 || fail "base rep=$R at $WID did not finish" ;;
                cap)  "$BIN_NEW"  --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS$CAPARG \
                          --out "$OUT/wide${WID}_cap_$R"  > "$LOG" 2>&1 || fail "cap rep=$R at $WID did not finish" ;;
                null) "$BIN_NEW"  --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS \
                          --out "$OUT/wide${WID}_null_$R" > "$LOG" 2>&1 || fail "null rep=$R at $WID did not finish" ;;
            esac
        done
        canary2 "$WID" "$R"
    done
done
echo "CLK@end: $(clk)"
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
else
    echo "SKIP_DEVICE=1: no device touched, no lock taken; judging the logs already in $OUT"
fi

# ================================== SUMMARY: two lines, one verdict ====================================
BA=$(med_wide_fam "$DECIDE" base attn); BC=$(med_wide_fam "$DECIDE" cap attn); BN=$(med_wide_fam "$DECIDE" null attn)
WB=$(med_wide "$DECIDE" base);         WC=$(med_wide "$DECIDE" cap);         WN=$(med_wide "$DECIDE" null)
need "$BA" "the base attn family ms at $DECIDE"; need "$BC" "the capped attn family ms at $DECIDE"
need "$WB" "the base wave wall_ms at $DECIDE";   need "$WC" "the capped wave wall_ms at $DECIDE"
need "$BN" "the NULL arm's attn family ms at $DECIDE"; need "$WN" "the NULL arm's wave wall_ms at $DECIDE"
for F in $FAMS; do
    [ "$(med_wide_fam "$DECIDE" base "$F")" = "NA" ] && fail "family row $F parsed empty on base at $DECIDE. The logs are missing or the field names moved; this is not a measurement."
    [ "$(med_wide_fam "$DECIDE" cap "$F")" = "NA" ] && fail "family row $F parsed empty on the capped arm at $DECIDE."
done

NULL_FAM=$(pct_move "$BA" "$BN"); NULL_WALL=$(pct_move "$WB" "$WN")
UNPAIRED=$(awk -v a="$NULL_FAM" -v w="$NULL_WALL" -v tol="$NULL_TOL" \
    'BEGIN{ x=a+0; if (x<0) x=-x; y=w+0; if (y<0) y=-y; m=(x>y)?x:y; print (m>tol)?"UNPAIRED":"paired" }')
SP_B=$(spread_of "$DECIDE" base); SP_C=$(spread_of "$DECIDE" cap)
[ "$SP_B" != "NA" ] || fail "the base rep spread at $DECIDE parsed empty; there are no reps to median"
WIDE_SPREAD=$(awk -v a="$SP_B" -v b="$SP_C" -v tol="$SPREAD_TOL" 'BEGIN{print (a>tol||b>tol)?1:0}')

# Externality: the sum of (cap - base) over every family OTHER than attn — the number that explains a
# DISPLACED verdict, which is why A9-BET pre-registers the wall as a second line at all.
ext_of() { local E=0 F fb fc
    for F in $FAMS; do
        [ "$F" = attn ] && continue
        fb=$(med_wide_fam "$DECIDE" base "$F"); fc=$(med_wide_fam "$DECIDE" cap "$F")
        { [ -z "$fb" ] || [ -z "$fc" ] || [ "$fb" = "NA" ] || [ "$fc" = "NA" ]; } && continue
        E=$(awk -v e="$E" -v b="$fb" -v c="$fc" 'BEGIN{printf "%.3f", e + (c - b)}')
    done
    echo "$E"
}
EXT=$(ext_of)

echo
echo "== A9-L-c — cap the attention split, behind --attn-split-cap $CAP. $REPS reps at [$WIDTHS] lanes,"
echo "   one window; the verdict is read at $DECIDE. base = $BASE (pre-change, no such flag),"
echo "   cap = the changed binary with --attn-split-cap $CAP, null = the same binary with the cap UNNAMED."
echo "   Live line: ratio <= $RAT_LIVE on BOTH the attn family row and the wave wall (A9-BET, -3.0 %)."
printf "   NULL tolerance %s%%  |  rep-spread tolerance %s%%\n" "$NULL_TOL" "$SPREAD_TOL"
printf "\n%-24s %10s %10s %10s\n" "attn family ms @${DECIDE}" base cap null
printf "%-24s %10s %10s %10s\n" "attn family" "$BA" "$BC" "$BN"
printf "%-24s %10s %10s %10s\n" "wave wall_ms" "$WB" "$WC" "$WN"
for WID in $WIDTHS; do
    [ "$WID" = "$DECIDE" ] && continue
    printf "%-24s %10s %10s %10s\n" "wave ms @${WID}" "$(med_wide "$WID" base)" "$(med_wide "$WID" cap)" "$(med_wide "$WID" null)"
done
printf "\n%-22s %10s %10s\n" family base cap
for F in $FAMS; do
    printf "%-22s %10s %10s\n" "$F" "$(med_wide_fam "$DECIDE" base "$F")" "$(med_wide_fam "$DECIDE" cap "$F")"
done
printf "\nNULL (changed binary, cap unnamed, vs base): family %s%%  wall %s%%  -> %s\n" "$NULL_FAM" "$NULL_WALL" "$UNPAIRED"
printf "Externality (sum of non-attn families, cap - base): %s ms  [explains a DISPLACED verdict; decides nothing alone]\n" "$EXT"
awk -v a="$SP_B" -v b="$SP_C" -v tol="$SPREAD_TOL" \
    'BEGIN{ w=(a>tol||b>tol)?"SPREAD TOO WIDE":"ok"; printf "WINDOW %s: attn family rep spread base %.2f%%, cap %.2f%% (tolerance %s%%)\n", w, a, b, tol }'

if [ "$UNPAIRED" = "UNPAIRED" ]; then
    echo "== VERDICT for A9-L-c: UNPAIRED — the NULL arm moved by more than ${NULL_TOL}% on one of the two lines, so the binaries are not comparable in this window. Report the NULL row and re-take; do NOT read a verdict from it."
    exit 1
fi
[ "$WIDE_SPREAD" = "1" ] && fail "window spread above ${SPREAD_TOL}% on the attn family (base $SP_B cap $SP_C). Treat this window as unpaired and re-take; do not read a verdict from it."

awk -v a="$BA" -v c="$BC" -v wb="$WB" -v wc="$WC" -v ext="$EXT" -v r="$RAT_LIVE" -v cap="$CAP" -v w="$DECIDE" 'BEGIN{
    f = c / a; wl = wc / wb;
    v = "DISPLACED";
    if (f > r) v = "FALSIFIED"; else if (wl <= r) v = "SHIP";
    printf "\n== VERDICT for A9-L-c (cap %s at %s lanes): %s\n", cap, w, v;
    printf "   LINE 1 attn family : %.2f -> %.2f ms  (%+.2f %%)  ratio %.4f   live <= %s\n", a, c, 100*(c-a)/a, f, r;
    printf "   LINE 2 wave wall   : %.2f -> %.2f ms  (%+.2f %%)  ratio %.4f   live <= %s   [SHIP GATE]\n", wb, wc, 100*(wc-wb)/wb, wl, r;
    printf "   externality (non-attn families): %+.2f ms\n", ext;
    if (v == "SHIP") print "   -> SHIP: both pre-registered lines cleared in one paired window, so the byte mechanism of A9-LONGCTX-CENSUS.md §4.2 converted into milliseconds.";
    else if (v == "FALSIFIED") print "   -> FALSIFIED: the family row did not reach -3 %. The scratch round-trip is hidden behind compute after all, and the 88 %-of-roof reading in A9-LONGCTX-CENSUS.md §2.3 is retired for this kernel. Do not re-run at another cap hoping for a different sign.";
    else print "   -> DISPLACED: attention got faster and the wave did not. Not shipped as a wave lever; report which family absorbed the saving — that is a finding about the co-scheduling of the phase, not noise.";
}'
echo
echo "Also report: both canaries (the flag's absence on base; the capped arm's SPLITCAP split=/cap=/natural=/span=,"
echo "and the null arm's inert split), the clocks, util at acquire and release, the NULL row, the rep spreads and"
echo "the externality — and the numerics basis: this knob is NOT bit-identical. It is judged by the existing bands,"
echo "the cascade band of engine/gate.py (a reassociating route) and §10.1 of"
echo "specification/10-reference-dumps-and-gates.md, and the gate set must be green at the shipping default (cap"
echo "inert) AND at the candidate cap before a SHIP verdict is honoured. Those commands are in"
echo "prototype/attn_percost/A9-LC-impl.md. logs in $OUT/"
