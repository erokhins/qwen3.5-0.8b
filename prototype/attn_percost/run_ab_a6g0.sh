#!/usr/bin/env bash
# run_ab_a6g0.sh — the device A/B that adjudicates `prototype/attn_percost/A6-BET.md` for G0, the 7-warp falsifier.
#
# MAIN-AGENT DEVICE WINDOW ONLY. Blocking, zero subagents, ONE `tools/gpu_lock.sh` hold, and it runs only
# programs that do not self-lock (`engine/build.sh`, `decode_run`, `tools/prefill_kernel_trace.py`; the numerics
# legs are deterministic and run outside the hold). A worker must not run it: it takes timings, and AGENTS.md
# keeps every timing on the main agent because this box serves its own tokens on the same iGPU.
#
# WHAT IS BEING DECIDED — AND WHY IT IS NOT THE WAVE ROW. A6-BET's falsifier rule, verbatim from that file:
#   family improves >= 10 %                 -> residency thesis holds -> FUND G1
#   family improvement < 3 % (inside §37's
#   paired floor)                           -> THESIS FALSIFIED, G1 is not funded
#   between 3 % and 10 %                    -> UNRESOLVED, and G1 is funded ONLY if the traced `attention`
#                                              row independently improves >= 5 % (the condition is recorded
#                                              here, not discovered later)
# G0's predicted wave move (A5 §4.3: -3.2 %) sits INSIDE the paired floor, so the wave row cannot adjudicate a
# 6 -> 7 wf/CU step; the ship-gate rows are printed with their lines as INFORMATION about what G1 would face, and
# they do not decide G0. That is the whole design of the verdict below, and a harness that "helpfully" gated G0 on
# wall_ms would be re-litigating A6-BET.
#
# THE TWO SIDES. There is no runtime knob for a warp count, so this is a two-BINARY A/B, not a two-flag one:
#   base  a `git worktree` at the pinned pre-change sha, built as-is           -> 3 warps, 96 thr, LDS 32,768
#   g0    a `git worktree` at the SAME sha, with prototype/attn_percost/a6_g0_warps7.py applied, built
#         -> 7 warps, 224 thr, LDS 65,536, 1 block/CU -> 7 wf/CU
# Both are built from the same sha so the ONLY difference between the binaries is that patch — which is also why
# BASEDIR/NEWDIR carry the sha in their names (a stale worktree from an earlier BASE cannot be reused as a fresh
# baseline, and `HEAD` is not a baseline: once G0 lands, HEAD *is* the change).
#
# PRE-REGISTERED LINES — A6-BET's table, nothing else, applied as RATIOS of A6-BET's own number over A6-BET's own
# base times the base measured IN THIS WINDOW (the §47 re-basing defect is exactly what this discipline refuses,
# and the ratios were fixed before any measurement). Overridable by env so harness and verdict cannot drift:
#       attn family, untraced ms/pass   live <= 0.8947 x base   dead > 0.9699 x base
#       attention traced ms/pass        live <= 0.8991 x base   dead > 0.9697 x base
#       wave wall_ms @4,096             live <= 0.9435 x base   dead > 0.9699 x base   [G1 ship gate]
#       16,384-lane wave ms             live <= 0.9189 x base   dead > 0.9701 x base   [G1 ship gate]
#       32,768-lane wave ms             live <= 0.8997 x base   dead > 0.9700 x base   [G1 ship gate]
#   and the falsifier's own three constants, which are what decide it:
#       family improvement >= FAMILY_LIVE_IMPROVE (0.10) -> LIVE; < FAMILY_DEAD_IMPROVE (0.03) -> DEAD;
#       otherwise the traced row must improve >= TRACE_TIEBREAK (0.05) for LIVE, else UNRESOLVED.
#
# WHAT IT REFUSES TO DO (each refusal is a §47/§37 lesson, and the A/B cannot print a verdict past any of them):
#   * it does not time a binary it has not identified: the note block of the LINKED binary (group_segment 32,768
#     vs 65,536, spill 0, private 0) and the launcher's own `warps= / lds= / wf_per_cu=` from the run that produced
#     the ms are both asserted in-window. `engine/build.sh` is not bit-reproducible and every check compiled INTO
#     a binary agrees with itself (AGENTS, 2026-10-17).
#   * it does not judge a geometry change on a launch-count canary alone: G0 keeps the launch COUNT and changes
#     grid.z (fattn_rtiles_of 11 -> 5 at 512 rows), so the plan/launch equality is asserted as invariance and the
#     live-change canaries are the geometry fields.
#   * it does not pass an empty parse: every field is `need`-checked, and PARSE_CHECK=1 colour-tests the verdict
#     parser against fixtures GENERATED from the ratio constants in this file (`mk_a6g0_parsecheck.py`), never
#     hand-typed — `mk_a4l1_parsecheck.py`'s history is the reason.
#   * it does not take the wide legs through argv: one argv slot caps at MAX_ARG_STRLEN = 131,072 B, so a 32,768-id
#     prompt cannot be passed at all (§39: shell exit 126 in 0 s, "the engine crashes at 32k"). Wide legs go via
#     --ids-file and the field count is asserted.
#   * it does not claim bit-identity for G0. G0 re-splits the round, so A6-BET expects bits to move and puts the
#     existing §10.1 aggregates in charge; that is `engine/attn_form_band.py` over the two dump dirs, run OUTSIDE
#     the hold because it is deterministic.
#
# Usage:
#   bash prototype/attn_percost/run_ab_a6g0.sh
#   BASE=<sha> bash prototype/attn_percost/run_ab_a6g0.sh
#   REPS=2 bash prototype/attn_percost/run_ab_a6g0.sh
#   SUMMARIZE=1  bash prototype/attn_percost/run_ab_a6g0.sh     # re-judge the logs, touch no device
#   PARSE_CHECK=1 bash prototype/attn_percost/run_ab_a6g0.sh    # colour-test the parser, touch no device
set -uo pipefail
export LC_ALL=C          # a decimal comma in a machine-read number is a parse bug wearing a locale's clothes
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD

# ---- pin the baseline -------------------------------------------------------------
# Re-pin this to the sha the tree ships at when G0 lands. It must be a PRE-change sha: `base` has to be the thing
# the bet is against, not the thing that already contains it.
BASE=${BASE:-9a4bbce}
SHORT=$(git rev-parse --short "$BASE" 2>/dev/null || echo "$BASE")
BASEDIR=${BASEDIR:-/tmp/a6g0-base-$SHORT}
NEWDIR=${NEWDIR:-/tmp/a6g0-g0-$SHORT}
BIN_BASE=$BASEDIR/engine/build/decode_run
BIN_G0=$NEWDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
PY=.venv/bin/python
REPS=${REPS:-3}
REPS_WIDE=${REPS_WIDE:-2}
W=4096
WIDE_WIDTHS=${WIDE_WIDTHS:-"16384 32768"}
TAG=${TAG:-a6g0-ab}
OUT=${OUT:-tmp/attn_percost/a6g0ab}
IDS4096=${IDS4096:-progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt}

# ---- the lines, as A6-BET's ratios (source: prototype/attn_percost/A6-BET.md) -----
RAT_A_LIVE=${RAT_A_LIVE:-0.8947};   RAT_A_DEAD=${RAT_A_DEAD:-0.9699}
RAT_AT_LIVE=${RAT_AT_LIVE:-0.8991}; RAT_AT_DEAD=${RAT_AT_DEAD:-0.9697}
RAT_W_LIVE=${RAT_W_LIVE:-0.9435};   RAT_W_DEAD=${RAT_W_DEAD:-0.9699}
RAT_16_LIVE=${RAT_16_LIVE:-0.9189}; RAT_16_DEAD=${RAT_16_DEAD:-0.9701}
RAT_32_LIVE=${RAT_32_LIVE:-0.8997}; RAT_32_DEAD=${RAT_32_DEAD:-0.9700}
# The falsifier's own constants (A6-BET, "G0's falsifier rule, stated now").
FAMILY_LIVE_IMPROVE=${FAMILY_LIVE_IMPROVE:-0.10}
FAMILY_DEAD_IMPROVE=${FAMILY_DEAD_IMPROVE:-0.03}
TRACE_TIEBREAK=${TRACE_TIEBREAK:-0.05}
SPREAD_TOL=${SPREAD_TOL:-3.0}          # %: the per-rep spread tolerated inside one window (there is no inert side)
EXT_TOL=${EXT_TOL:-0.03}               # informational only
FAMS="gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head host_tail"
ARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --wave-rows 512"
TRACE_ARGS="--steps 0 --gemm lt --gemm-pins all --wave-rows 512"
# Note-block expectations, read from `prototype/attn_percost/A6-result.md` §1/§2 rather than invented here.
NOTE_BASE_WANT=${NOTE_BASE_WANT:-"178 0 0 32768"}          # vgpr spill private group
NOTE_G0_GROUP=${NOTE_G0_GROUP:-65536}
mkdir -p "$OUT"
# Absolute, because the build legs `cd` into a worktree before redirecting into $OUT: a relative log path
# resolves inside the worktree, where tmp/ does not exist, and the build then fails with "No such file or
# directory" for a bug that has nothing to do with the code under test.
OUT="$(cd "$OUT" && pwd)"

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
med_trace(){ local side=$1
    for R in $(seq 1 "$REPS"); do traced "$OUT/trace_${side}_$R.txt"; done | grep -v '^$' | median -; }
# One field across reps of one side, as a spread percentage: the substitute for A4's NULL leg.
spread_of(){ local fn=$1 side=$2; shift 2
    local vals med lo hi
    vals=$(for R in $(seq 1 "$REPS"); do "$fn" "$@" "$OUT/${side}_$R.log"; done | grep -v '^$')
    med=$(echo "$vals" | median -); lo=$(echo "$vals" | sort -g | head -1); hi=$(echo "$vals" | sort -g | tail -1)
    awk -v m="$med" -v l="$lo" -v h="$hi" 'BEGIN{ if (m==0||m=="NA"){print "NA"} else printf "%.2f", 100*(h-l)/m }'
}

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

# ============================ PARSE_CHECK: colour-test the verdict parser ==============================
if [ "${PARSE_CHECK:-0}" = "1" ]; then
    for SCEN in live unresolved live_by_trace dead; do
        case $SCEN in
            live)          WANT=LIVE ;;
            unresolved)    WANT=UNRESOLVED ;;
            live_by_trace) WANT=LIVE ;;
            dead)          WANT=DEAD ;;
        esac
        SCEN=$SCEN OUT="$OUT/pc_$SCEN" REPS=1 REPS_WIDE=1 \
            "$PY" prototype/attn_percost/mk_a6g0_parsecheck.py \
            || fail "parse-check generator failed for scenario $SCEN"
        GOT=$(env -u PARSE_CHECK OUT="$OUT/pc_$SCEN" REPS=1 REPS_WIDE=1 SUMMARIZE=1 bash "$0" 2>&1 \
              | sed -n 's/^== VERDICT for G0: \([A-Z_]*\).*/\1/p' | tail -1)
        [ "$GOT" = "$WANT" ] || fail "parse-check: scenario $SCEN printed '${GOT:-<nothing>}', expected $WANT. The verdict parser is not reading its own table; no real window may be run."
        echo "PARSE-CHECK ok: scenario $SCEN -> $GOT (expected $WANT)"
    done
    echo "PARSE-CHECK PASS: the parser distinguishes LIVE / UNRESOLVED / DEAD and the trace tie-break on synthetic logs. This says nothing about the engine."
    exit 0
fi

# ============================== NUMERICS LEGS (deterministic: outside the hold) ========================
# A6-BET, "Numerics": G0 re-splits the round, so bit-identity is NOT expected and the existing §10.1 aggregates
# govern (cos >= 0.9995, meanrel <= 5 %), with no band widened. `engine/attn_form_band.py` is the instrument that
# already imports those thresholds from the file that owns them, refuses a missing row instead of calling a
# crashed run a band difference, and applies the flip-room rule. Its --ship/--tile flag names are inherited from
# the tiled-form A/B it was written for; here they are base and g0.
band_leg() {
    local N=$1
    ids_file_of "$N" "$OUT/ids${N}_band.csv"
    for SIDE in base g0; do
        BX=$BIN_BASE; [ "$SIDE" = g0 ] && BX=$BIN_G0
        "$BX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$OUT/ids${N}_band.csv")" --steps 0 \
            --prefill-logits all --state-fp --out "$OUT/band${N}_$SIDE" > "$OUT/band${N}_$SIDE.log" 2>&1 \
            || fail "numerics leg $N: $SIDE did not finish"
    done
    "$PY" engine/attn_form_band.py --ship "$OUT/band${N}_base" --tile "$OUT/band${N}_g0" --prompt-len "$N" \
        > "$OUT/band${N}_verdict.txt" 2>&1 \
        || fail "numerics leg $N: attn_form_band.py refused the pair (see $OUT/band${N}_verdict.txt). That is a red numerics result, not a parse problem."
    grep -h "^STATEFP" "$OUT/band${N}_base.log" > "$OUT/band${N}_base.fp"
    grep -h "^STATEFP" "$OUT/band${N}_g0.log"   > "$OUT/band${N}_g0.fp"
    if cmp -s "$OUT/band${N}_base.fp" "$OUT/band${N}_g0.fp"; then
        # NOT a failure, and say why: the banner canaries (warps/lds/wf_per_cu) are the witness that the geometry
        # is live. Identical fingerprints here would only mean the re-split did not move THIS fixture's arithmetic.
        echo "NUMERICS $N note: STATEFP identical on base and g0 — the re-split did not move this fixture's arithmetic. Not a defect (the in-window banner canary is what proves the geometry is live), but read the band row as vacuous for this width."
    else
        echo "NUMERICS $N: STATEFP moved between base and g0, as A6-BET predicts for a warp-count change."
    fi
    echo "NUMERICS $N ok: attn_form_band.py adjudicated every dumped row against the bands it imports (see $OUT/band${N}_verdict.txt)"
}

[ -f "$IDS4096" ] || fail "no 4,096-id file at $IDS4096"

if [ "${SUMMARIZE:-0}" != "1" ]; then
    # ---- build both sides from the SAME sha (CPU only; nothing here touches the device) ----------------
    [ -d "$BASEDIR" ] || git worktree add "$BASEDIR" "$BASE" --detach -q || fail "worktree add failed for $BASE"
    for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
    [ -x "$BIN_BASE" ] || { echo "building base $BASE into $BASEDIR"; ( cd "$BASEDIR" && bash engine/build.sh >"$OUT/build_base.log" 2>&1 ) || fail "baseline build failed (see $OUT/build_base.log)"; }
    [ -d "$NEWDIR" ] || git worktree add "$NEWDIR" "$BASE" --detach -q || fail "worktree add failed for the g0 side"
    for D in data models; do [ -e "$NEWDIR/$D" ] || ln -s "$REPO/$D" "$NEWDIR/$D"; done
    python3 prototype/attn_percost/a6_g0_warps7.py \
        --src-header "$NEWDIR/engine/attn_fattn_kernels.h" \
        --src-decode "$NEWDIR/prototype/attn_decode/attn_decode_kernels.h" \
        --out-header "$NEWDIR/engine/attn_fattn_kernels.h" \
        --out-decode "$NEWDIR/prototype/attn_decode/attn_decode_kernels.h" \
        || fail "could not apply the G0 patch to $NEWDIR (the anchors moved; re-read the headers)"
    [ -x "$BIN_G0" ] || { echo "building g0 in $NEWDIR"; ( cd "$NEWDIR" && bash engine/build.sh >"$OUT/build_g0.log" 2>&1 ) || fail "g0 build failed (see $OUT/build_g0.log)"; }
    [ "$BIN_BASE" -nt "$BASEDIR/engine/attn_fattn_kernels.h" ] || fail "the base binary is older than the source it came from"
    [ "$BIN_G0"   -nt "$NEWDIR/engine/attn_fattn_kernels.h"   ] || fail "the g0 binary is older than the patched source it came from"

    # ---- CANARY A: the NOTE BLOCKS OF THE TWO BUILT BINARIES (the stale-binary guard) ------------------
    note() { "$PY" prototype/attn_percost/a4_l1_note_read.py "$1" --kernel attn_fattn_wave 2>&1 \
             | sed -n "s/^KERNEL attn_fattn_wave .*vgpr_count=\([0-9]*\) vgpr_spill_count=\([0-9]*\).*private_segment_fixed_size=\([0-9]*\) group_segment_fixed_size=\([0-9]*\).*/\1 \2 \3 \4/p"; }
    NB=$(note "$BIN_BASE"); need "$NB" "the base binary's attn_fattn_wave note block"
    NG=$(note "$BIN_G0");   need "$NG" "the g0 binary's attn_fattn_wave note block"
    [ "$NB" = "$NOTE_BASE_WANT" ] || fail "canary A: base attn_fattn_wave reads '$NB', want '$NOTE_BASE_WANT' (vgpr spill private group). The comparator is not the shipped 3-warp form; nothing is timed."
    [ "$(echo "$NG" | awk '{print $3" "$4}')" = "0 $NOTE_G0_GROUP" ] \
        || fail "canary A: the g0 binary reads '$NG' — want spill 0 and group_segment_fixed_size $NOTE_G0_GROUP (A6-result.md §2). The variant the bet is about is not what got compiled, and its ms would be the shipped form's a second time."
    echo "CANARY A ok: base $NB (spill 0, group 32,768); g0 $NG (spill 0, group $NOTE_G0_GROUP) — read off the LINKED binaries in this window"

    band_leg 640
    band_leg 1111

    tools/gpu_lock.sh acquire "$TAG" || exit 1
    echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
    echo "CLK@start: $(clk)"

    for R in $(seq 1 "$REPS"); do
        for SIDE in base g0; do
            sleep 5
            BX=$BIN_BASE; [ "$SIDE" = g0 ] && BX=$BIN_G0
            "$BX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$IDS4096")" $ARGS --out "$OUT/${SIDE}_$R" \
                > "$OUT/${SIDE}_$R.log" 2>&1 || fail "$SIDE run rep=$R did not finish"
        done
        # --- CANARY B: the geometry of the kernel that ACTUALLY RAN, from the launcher's own banner ------
        # `warps= / vgpr= / lds= / wf_per_cu=` come from `hipFuncGetAttributes` and
        # `hipOccupancyMaxActiveBlocksPerMultiprocessor` on the LAUNCHED function (`engine/kernels_attn.cpp`
        # `fattn_echo`), not from a name typed beside the print. warps/lds/wf_per_cu are the fields G0 is
        # REQUIRED to move; grid.z moves too (A6-result.md §2) and is deliberately not a canary because it is
        # not printed — launch-count invariance is asserted instead by CANARY C.
        WB=$(grep -o 'warps=[0-9]*' "$OUT/base_$R.log" | head -1);  WG=$(grep -o 'warps=[0-9]*' "$OUT/g0_$R.log" | head -1)
        LB=$(grep -o 'lds=[0-9]*'  "$OUT/base_$R.log" | head -1);   LG=$(grep -o 'lds=[0-9]*'  "$OUT/g0_$R.log" | head -1)
        FB=$(grep -o 'wf_per_cu=[0-9]*' "$OUT/base_$R.log" | head -1); FG=$(grep -o 'wf_per_cu=[0-9]*' "$OUT/g0_$R.log" | head -1)
        GB=$(grep -o 'vgpr=[0-9]*' "$OUT/base_$R.log" | head -1);    GG=$(grep -o 'vgpr=[0-9]*' "$OUT/g0_$R.log" | head -1)
        need "$GB" "the base side's launched vgpr="; need "$GG" "the g0 side's launched vgpr="
        # The banner's `vgpr=` is `hipFuncGetAttributes` on the LAUNCHED function; CANARY A's number came from the
        # note block of the same file on disk. Requiring them to agree is what makes "the binary that printed these
        # milliseconds is the binary whose note block I read" a checked statement rather than an assumption.
        [ "$GB" = "vgpr=$(echo "$NB" | awk '{print $1}')" ] \
            || fail "canary B: rep=$R base banner $GB disagrees with the base note block ($NB). Two objects are in play."
        [ "$GG" = "vgpr=$(echo "$NG" | awk '{print $1}')" ] \
            || fail "canary B: rep=$R g0 banner $GG disagrees with the g0 note block ($NG). Two objects are in play."
        need "$WB" "the base side's warps= field"; need "$WG" "the g0 side's warps= field"
        need "$LB" "the base side's lds= field";   need "$LG" "the g0 side's lds= field"
        need "$FB" "the base side's wf_per_cu= field"; need "$FG" "the g0 side's wf_per_cu= field"
        [ "$WB" = "warps=3" ] && [ "$LB" = "lds=32768" ] && [ "$FB" = "wf_per_cu=6" ] \
            || fail "canary B: rep=$R base printed $WB $LB $FB, want warps=3 lds=32768 wf_per_cu=6. The baseline is not the shipped 3-warp form."
        [ "$WG" = "warps=7" ] && [ "$LG" = "lds=65536" ] && [ "$FG" = "wf_per_cu=7" ] \
            || fail "canary B: rep=$R g0 printed $WG $LG $FG, want warps=7 lds=65536 wf_per_cu=7 (A6-result.md §2's compile reading). Either the build is stale or the driver refused the shape; in both cases the ms below is not a residency measurement."
        # --- CANARY C: the launch plan is identical on both sides (A6-BET's harness requirement) ---------
        for SIDE in base g0; do
            LOG="$OUT/${SIDE}_$R.log"
            GOT=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1)
            PLAN=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* plan_launches=([0-9]+).*/\1/' | tail -1)
            MATCH=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* counts_match=([01]).*/\1/' | tail -1)
            need "$GOT" "the attn launch count on $SIDE"; need "$PLAN" "the attn plan bill on $SIDE"
            [ "$MATCH" = "1" ] || fail "canary D: $LOG counts_match=0 on the attn row: the plan and the submissions disagree."
            [ "$PLAN" = "$GOT" ] || fail "canary D: $LOG plans $PLAN attn launches and submits $GOT."
            grep -q "TOTAL.*plan_match=1" "$LOG" || fail "canary D: plan_match!=1 on $SIDE at rep=$R"
            case $SIDE in base) LB2=$GOT ;; g0) LG2=$GOT ;; esac
        done
        [ "$LB2" = "$LG2" ] || fail "canary D: rep=$R attn launches differ (base $LB2 g0 $LG2). G0 changes which ROWS one block owns and grid.z, not how many launches a layer submits; a launch-count move means something else changed too."
        # --- CANARY E: the fields G0 must NOT move (coverage is a §07.3 predicate, not a warp-count one) --
        for F in vtiles wave_rows split fattn_rows fallback_rows covered_rows_pct covered_work_pct; do
            VB=$(grep -o "$F=[^ ]*" "$OUT/base_$R.log" | head -1); VG=$(grep -o "$F=[^ ]*" "$OUT/g0_$R.log" | head -1)
            need "$VB" "the $F field on base"; need "$VG" "the $F field on g0"
            [ "$VB" = "$VG" ] || fail "canary E: rep=$R field $F differs (base ${VB:-<none>} g0 ${VG:-<none>}). G0 may not change which rows the form owns — that is `fattn_covers`' decision and it does not read the warp count."
        done
    done

    # --- the traced leg: the `attention` PHASE, A6-BET's tie-break row ---------------------------------
    for R in $(seq 1 "$REPS"); do
        for SIDE in base g0; do
            sleep 5
            BX=$BIN_BASE; [ "$SIDE" = g0 ] && BX=$BIN_G0
            "$PY" tools/prefill_kernel_trace.py --waves 8 --out "$OUT/trace_${SIDE}_$R" -- \
                "$BX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$IDS4096")" $TRACE_ARGS \
                --out "$OUT/trace_${SIDE}_${R}_out" > "$OUT/trace_${SIDE}_$R.log" 2>&1 \
                || fail "traced leg: the $SIDE rep=$R traced run did not finish"
            CSV=$(ls "$OUT/trace_${SIDE}_$R"/*_kernel_trace.csv 2>/dev/null | head -1)
            need "$CSV" "kernel_trace.csv for $SIDE rep=$R"
            # The divisor is DERIVED from `attn_prep`, whose per-pass site count is 48 at 4,096 lanes and which
            # G0 does not touch at all (A6-result.md §1.3's drift check: one TU line differs, and it is not this
            # kernel). Per the AGENTS rocprofv3 trap row: derive the pass count, never type a number beside it.
            "$PY" tools/trace_phase_split.py "$CSV" --engine ours --calibrate attn_prep --per-pass 48 \
                --tokens "$W" > "$OUT/trace_${SIDE}_$R.txt" 2>&1 \
                || fail "traced leg: trace_phase_split.py refused the $SIDE rep=$R census"
            need "$(traced "$OUT/trace_${SIDE}_$R.txt")" "the attention phase row on $SIDE rep=$R"
        done
    done

    # --- the wide legs -----------------------------------------------------------------------------------
    for WID in $WIDE_WIDTHS; do
        ids_file_of "$WID" "$OUT/ids$WID.csv"
        for R in $(seq 1 "$REPS_WIDE"); do
            for SIDE in base g0; do
                sleep 5
                BX=$BIN_BASE; [ "$SIDE" = g0 ] && BX=$BIN_G0
                "$BX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS --out "$OUT/wide${WID}_${SIDE}_$R" \
                    > "$OUT/wide${WID}_${SIDE}_$R.log" 2>&1 || fail "wide leg $WID: $SIDE rep=$R did not finish"
            done
        done
    done
    echo "CLK@end: $(clk)"
    echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
    tools/gpu_lock.sh release "$TAG"
fi

# ================================== SUMMARY: five rows, one verdict ====================================
BA=$(med_of fam base attn); BG=$(med_of fam g0 attn)
BT=$(med_trace base);      TG=$(med_trace g0)
BW=$(med_of tot base);     BWG=$(med_of tot g0)
need "$BA" "base attn family ms"; need "$BG" "the g0 attn family ms"
need "$BT" "base traced attention ms"; need "$TG" "the g0 traced attention ms"
need "$BW" "base wall_ms at 4,096"; need "$BWG" "the g0 wall_ms at 4,096"
for F in $FAMS; do
    [ "$(med_of fam base "$F")" = "NA" ] && fail "family row $F parsed empty on base. The logs are missing or the field names moved; this is not a measurement."
    [ "$(med_of fam g0 "$F")" = "NA" ] && fail "family row $F parsed empty on g0."
done
SP_A=$(spread_of fam base attn); SP_AG=$(spread_of fam g0 attn)
need "$SP_A" "the base rep spread"; need "$SP_AG" "the g0 rep spread"

echo
echo "== A6 G0 — the 7-warp falsifier. $REPS reps at $W lanes (untraced + traced), $REPS_WIDE reps at"
echo "   [$WIDE_WIDTHS] lanes, all inside one window. Lines are A6-BET.md's ratios over its own bases,"
echo "   applied to the base measured IN THIS WINDOW; the DECISION is the family row alone (A6-BET's"
echo "   falsifier rule), and the wave/16k/32k rows are G1's ship gates printed as information."
printf "   family row:  LIVE if it improves >= %s   DEAD if it improves < %s   else the traced row must improve >= %s\n" "$FAMILY_LIVE_IMPROVE" "$FAMILY_DEAD_IMPROVE" "$TRACE_TIEBREAK"

printf "\n%-16s %10s %10s %14s\n" family base_ms g0_ms share_of_wave
for F in $FAMS; do
    printf "%-16s %10s %10s %10s %14s\n" "$F" "$(med_of fam base "$F")" "$(med_of fam g0 "$F")" \
        "$(grep -h "family=$F " "$OUT/base_1.log" | sed -E 's/.*share_pct=([0-9.]+).*/\1/' | tail -1)"
done
printf "\n%-26s %10s %10s %10s\n" "attn family, untraced" "$BA" "$BG" ""
printf "%-26s %10s %10s %10s\n" "attention phase, traced" "$BT" "$TG" ""
printf "%-26s %10s %10s %10s\n" "wave wall_ms @4,096" "$BW" "$BWG" ""
for WID in $WIDE_WIDTHS; do
    BB=$(med_wide "$WID" base); BGW=$(med_wide "$WID" g0)
    need "$BB" "base wide ms at $WID"; need "$BGW" "the g0 wide ms at $WID"
    printf "%-26s %10s %10s %10s\n" "wave ms @${WID}" "$BB" "$BGW" ""
done
# The window's own tightness. G0 has no inert side to NULL against (a warp count is not a flag), so what
# replaces A4's base-vs-single leg is a refusal to read a window whose reps disagree.
awk -v a="$SP_A" -v b="$SP_AG" -v tol="$SPREAD_TOL" 'BEGIN{ w=(a>tol||b>tol)?"SPREAD TOO WIDE":"ok";
    printf "\nWINDOW %s: base attn family spread %.2f%%, g0 spread %.2f%% across reps (tolerance %s%%). No inert side exists for a compile-time geometry change, so this is the pairing check.\n", w, a, b, tol }'
[ "$(awk -v a="$SP_A" -v b="$SP_AG" -v tol="$SPREAD_TOL" 'BEGIN{print (a>tol||b>tol)?1:0}')" = "1" ] \
    && fail "window spread above ${SPREAD_TOL}% on the attn family (base $SP_A g0 $SP_AG). Treat this window as unpaired and re-take; do not read a verdict from it."

# Externality: sum of (g0 - base) over every family OTHER than attn. Informational, as A6-BET pre-registered no
# line for it and wall_ms already contains every family.
ext_of() { local WID=$1 E=0 F fb fg
    for F in $FAMS; do
        [ "$F" = attn ] && continue
        if [ "$WID" = "$W" ]; then fb=$(med_of fam base "$F"); fg=$(med_of fam g0 "$F")
        else fb=$(grep -h "family=$F " "$OUT/wide${WID}_base_1.log" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1)
             fg=$(grep -h "family=$F " "$OUT/wide${WID}_g0_1.log"   | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1); fi
        { [ -z "$fb" ] || [ -z "$fg" ]; } && continue
        E=$(awk -v e="$E" -v b="$fb" -v g="$fg" 'BEGIN{printf "%.3f", e + (g - b)}')
    done
    echo "$E"
}
E4=$(ext_of "$W"); E16=$(ext_of 16384); E32=$(ext_of 32768)
T16=$(med_wide 16384 g0); T32=$(med_wide 32768 g0)
B16=$(med_wide 16384 base); B32=$(med_wide 32768 base)
need "$T16" "the g0 16,384-lane ms"; need "$T32" "the g0 32,768-lane ms"

awk -v a="$BG" -v ba="$BA" -v at="$TG" -v bat="$BT" -v w="$BWG" -v bw="$BW" \
    -v w16="$T16" -v bw16="$B16" -v w32="$T32" -v bw32="$B32" \
    -v e4="$E4" -v e16="$E16" -v e32="$E32" \
    -v ra_l="$RAT_A_LIVE" -v ra_d="$RAT_A_DEAD" -v rat="$RAT_AT_LIVE" -v ratd="$RAT_AT_DEAD" \
    -v rw_l="$RAT_W_LIVE" -v rw_d="$RAT_W_DEAD" -v r16_l="$RAT_16_LIVE" -v r16_d="$RAT_16_DEAD" \
    -v r32_l="$RAT_32_LIVE" -v r32_d="$RAT_32_DEAD" \
    -v fl="$FAMILY_LIVE_IMPROVE" -v fd="$FAMILY_DEAD_IMPROVE" -v tb="$TRACE_TIEBREAK" 'BEGIN{
    fi = (ba - a) / ba;                 # family IMPROVEMENT, positive = faster
    ti = (bat - at) / bat;              # traced improvement
    v = "UNRESOLVED";
    if (fi >= fl)                       v = "LIVE";
    else if (fi < fd)                   v = "DEAD";
    else if (ti >= tb)                  v = "LIVE";
    printf "\n== VERDICT for G0: %s\n", v;
    printf "   attn family (THE ROW)   : %.2f -> %.2f ms  (%+.2f %% ms; improvement %.1f%%)  LIVE at >= %.0f%%, DEAD below %.0f%%   [A6-BET ratios: live <= %s, dead > %s]\n",
           ba, a, 100*(a-ba)/ba, 100*fi, 100*fl, 100*fd, ra_l, ra_d;
    printf "   attention traced        : %.2f -> %.2f ms  (%+.2f %%)   tie-break needs >= %.0f%% when the family row is in between\n",
           bat, at, 100*(at-bat)/bat, 100*tb;
    printf "   wave wall_ms @4,096     : %.2f -> %.2f ms  (%+.2f %%)   live <= %s, dead > %s  [G1 ship gate, informational here]\n",
           bw, w, 100*(w-bw)/bw, rw_l, rw_d;
    printf "   wave ms @16,384         : %.2f -> %.2f ms  (%+.2f %%)   live <= %s, dead > %s\n",
           bw16, w16, 100*(w16-bw16)/bw16, r16_l, r16_d;
    printf "   wave ms @32,768         : %.2f -> %.2f ms  (%+.2f %%)   live <= %s, dead > %s\n",
           bw32, w32, 100*(w32-bw32)/bw32, r32_l, r32_d;
    printf "   externality ms          : %+.2f / %+.2f / %+.2f  (4,096 / 16,384 / 32,768)  [informational, decides nothing]\n", e4, e16, e32;
    if (v == "LIVE" && fi < fl) printf "   LIVE by the traced tie-break, not by the family row: A6-BET records that condition, quote it with the number.\n";
    if (v == "LIVE")     printf "   -> residency thesis holds at engine geometry: FUND G1 (prototype/attn_percost/A6-BET.md).\n";
    else if (v == "DEAD")printf "   -> thesis falsified: G1 is NOT funded, and attention'\''s remaining route is the llama-shaped rewrite A5 §5 puts to the user as a scope decision.\n";
    else                 printf "   -> UNRESOLVED: report it as unresolved, do not ship, do not narrate (A4-BET'\''s own words, G1 as precedent).\n";
}'
echo
echo "Also report with this: the two note-block readings (the stale-binary canary), the two banners"
echo "(warps=/lds=/wf_per_cu=), the rep spreads, the clocks, the traced census files, and the two"
echo "attn_form_band.py verdicts. A4-L1's sibling gate is .venv/bin/python engine/gate.py --fixture"
echo "wide640f on the SHIPPING tree — G0 has no knob, so gate.py judges the base side only; the variant's"
echo "band row is the attn_form_band.py output above. DONE. logs in $OUT/"
