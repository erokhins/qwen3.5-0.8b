#!/usr/bin/env bash
# run_ab_a4l1.sh — the device A/B that adjudicates A4-BET Gate 2 for the two-accumulator score roll.
#
# MAIN-AGENT DEVICE WINDOW ONLY. Blocking, zero subagents, ONE `tools/gpu_lock.sh` hold, and it runs only
# programs that do not self-lock (`engine/build/decode_run`, `tools/prefill_kernel_trace.py`; the CPU legs
# touch no device). A worker must not run it: it takes timings, and `AGENTS.md` keeps every timing on the
# main agent because this box serves its own tokens on the same iGPU. The worker-side correctness gate is
# `engine/gate.py --attn-fattn-roll two` (self-locking, so it runs OUTSIDE this window) plus the NUMERICS
# block below, which is deterministic and therefore needs no exclusive device.
#
# THE THREE SIDES.
#   base    HEAD (`git worktree`, built into $BASEDIR). It has NO `--attn-fattn-roll` at all, and its
#           `attn_fattn_wave` carries the shipped roll. The comparator AND the inertness reference.
#   single  the new binary at the default. `base` vs `single` must be a NULL — same bits (checked outside
#           the hold) and the same milliseconds (the NULL line below). This is the leg that proves graduating
#           the knob did not move the shipping path.
#   two     the new binary at `--attn-fattn-roll two`: two accumulators, both tiles of a pair fetched before
#           either `v_wmma` issues, two dependent chains of 8 instead of one of 16. THE VERDICT IS ON THIS.
#
# PRE-REGISTERED LINES. These are A4-BET.md's Gate-2 lines and nothing else: the worker's first harness
# carried a "re-based" set (91.27 / 73.34 / 1,196.89 ms) attributed to the supervisor, and the supervisor
# never fixed any such set - the 1,196.89 wall base is 2.6x the ~460 ms this very fixture produces, so it was
# not a re-measurement of the same denominator either. A4-BET's lines are absolute numbers over its own §45
# bases; since this harness measures base and two in ONE window, each line is applied as the pre-registered
# RATIO of A4-BET's number to A4-BET's base (dead lands at base-3% on every row, which is the intent) times
# the base measured IN THIS WINDOW. That keeps the pre-registration (the ratios were fixed before any
# measurement) and removes window drift from the comparison. No figure from the `two` side was read before
# this amendment was made. Every line stays env-overridable so harness and printed verdict cannot drift.
#       attn family, untraced ms/pass  : live <= 119/133 = 0.8947 x base   dead > 129/133 = 0.9699 x base
#       attention phase, traced ms/pass: live <= 140/155.71 = 0.8991      dead > 151/155.71 = 0.9697
#       wave wall_ms at 4,096          : live <= 428/453.65 = 0.9435      dead > 440/453.65 = 0.9699 [SHIP GATE]
#       16,384-lane wave ms            : live <= 3050/3319.25 = 0.9189    dead > 3220/3319.25 = 0.9701
#       32,768-lane wave ms            : live <= 9400/10448.13 = 0.8997   dead > 10135/10448.13 = 0.9700
#   ALL FIVE must clear their live line for LIVE; ANY dead line kills; anything else is UNRESOLVED and is
#   reported as unresolved with no narration (A4-BET Gate 2, G1's precedent).
#   EXTERNALITY is printed but does NOT decide: A4-BET pre-registered no externality line, and the wave
#   wall_ms - which contains every family - is the pre-registered ship gate. A3's trap is still visible here
#   (a family row that improves while the wave does not), so it is reported, not hidden.
#
# Usage:
#   bash prototype/attn_percost/run_ab_a4l1.sh
#   REPS=2 bash prototype/attn_percost/run_ab_a4l1.sh
#   SUMMARIZE=1 bash prototype/attn_percost/run_ab_a4l1.sh          # re-judge the logs, touch no device
#   PARSE_CHECK=1 bash prototype/attn_percost/run_ab_a4l1.sh        # colour-test the parser, touch no device
#   BASE=<sha> bash prototype/attn_percost/run_ab_a4l1.sh   # override the pinned pre-knob baseline
set -uo pipefail
export LC_ALL=C  # a decimal comma in a machine-read number is a parse bug wearing a locale's clothes
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
# The baseline is the tree as shipped BEFORE the knob landed, not HEAD. Once the knob is committed, HEAD
# carries --attn-fattn-roll, so canary A ("a baseline that accepts the flag is the wrong binary") fires by
# construction and nothing gets timed - which is the canary working, not the harness failing. Re-pin this
# sha when the knob ships. BASEDIR is derived from the sha so a stale worktree from an earlier BASE can
# never be reused as a fresh baseline.
BASE=${BASE:-47d390e}
BASEDIR=${BASEDIR:-/tmp/base-a4l1-$(git rev-parse --short "$BASE" 2>/dev/null || echo "$BASE")}
BIN_NEW=engine/build/decode_run
BIN_OLD=$BASEDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
PY=.venv/bin/python
REPS=${REPS:-3}                    # paired reps inside the window; medians, not single shots
REPS_WIDE=${REPS_WIDE:-2}          # the wide legs cost seconds each; median those too
W=4096
WIDE_WIDTHS=${WIDE_WIDTHS:-"16384 32768"}
TAG=${TAG:-a4l1-ab}
OUT=${OUT:-tmp/attn_percost/a4l1ab}
IDS4096=${IDS4096:-progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt}

# Ratios from A4-BET.md's Gate-2 table (its number / its base). LINE_* are derived from the IN-WINDOW base
# medians further down unless the caller overrides them explicitly.
RAT_A_LIVE=${RAT_A_LIVE:-0.8947};  RAT_A_DEAD=${RAT_A_DEAD:-0.9699}
RAT_AT_LIVE=${RAT_AT_LIVE:-0.8991}; RAT_AT_DEAD=${RAT_AT_DEAD:-0.9697}
RAT_W_LIVE=${RAT_W_LIVE:-0.9435};  RAT_W_DEAD=${RAT_W_DEAD:-0.9699}
RAT_16_LIVE=${RAT_16_LIVE:-0.9189}; RAT_16_DEAD=${RAT_16_DEAD:-0.9701}
RAT_32_LIVE=${RAT_32_LIVE:-0.8997}; RAT_32_DEAD=${RAT_32_DEAD:-0.9700}
EXT_TOL=${EXT_TOL:-0.03}           # informational band: |externality| vs the non-attn share of the base wave
NULL_TOL=${NULL_TOL:-2.0}           # per cent, base vs `single` on the attn family: above it the window is unpaired
FAMS="gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head host_tail"
ARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --wave-rows 512"
TRACE_ARGS="--steps 0 --gemm lt --gemm-pins all --wave-rows 512"
mkdir -p "$OUT"

clk()     { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk" | tr '\n' ' '; }
fam()     { grep -h "WAVE gran=stage family=$1 " "$2" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
tot()     { grep -h "WAVE gran=stage TOTAL" "$1" | sed -E 's/.*wall_ms=([0-9.]+).*/\1/' | tail -1; }
traced()  { awk '$1=="attention"{print $2}' "$1" | tail -1; }   # trace_phase_split table: phase ms/pass share launches/pass
median()  { sort -g "$@" | awk '{a[NR]=$1} END{if(NR==0){print "NA";exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
fail()    { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
need()    { [ -n "$1" ] || fail "empty parse where a number was required ($2). A verdict computed from nothing is the empty-parse failure AGENTS.md records; not a measurement."; }
tag_of()  { case $1 in base) echo base ;; *) echo "new_$1" ;; esac; }
roll_of() { case $1 in base) echo "" ;; *) echo "--attn-fattn-roll $1" ;; esac; }
# Median of one field across the reps for one side. `fn` is fam/tot and the extra "$@" is the family name
# `fam` needs and `tot` does not.
med_of()  { local fn=$1 side=$2; shift 2
    for R in $(seq 1 "$REPS"); do "$fn" "$@" "$OUT/$(tag_of "$side")_$R.log"; done | grep -v '^$' | median -; }
med_wide(){ local WID=$1 side=$2
    for R in $(seq 1 "$REPS_WIDE"); do tot "$OUT/wide${WID}_$(tag_of "$side")_$R.log"; done | grep -v '^$' | median -; }
med_trace(){ local side=$1
    for R in $(seq 1 "$REPS"); do traced "$OUT/trace_$(tag_of "$side")_$R.txt"; done | grep -v '^$' | median -; }

# --- ids. The wide legs MUST go through a file: one argv slot caps at MAX_ARG_STRLEN = 131,072 B, so a
# 32,768-id prompt (191,588 B) cannot be passed at all — `--ids` at that width dies with shell exit 126 in 0 s
# and presents as "the engine crashes at 32k" (§39). The field count is asserted, because a silently short
# prompt is worse than a missing width: it judges another prompt while printing the wide label.
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

if [ "${PARSE_CHECK:-0}" = "1" ]; then
    # --- COLOUR TEST of the verdict parser: three synthetic windows, three colours, no device --------------
    # An A/B script is judged by a parser, and a parser that matches nothing prints NA and still prints a
    # verdict (AGENTS.md, the empty-parse class: a check whose input list can be empty cannot fail). Each
    # scenario below names the verdict it MUST produce; the harness re-runs itself with SUMMARIZE=1 over
    # generated logs and refuses if the colour is wrong.
    for SCEN in live unresolved dead; do
        WANT=LIVE; [ "$SCEN" = unresolved ] && WANT=UNRESOLVED; [ "$SCEN" = dead ] && WANT=DEAD
        SCEN=$SCEN OUT="$OUT/pc_$SCEN" REPS=1 REPS_WIDE=1 "$PY" prototype/attn_percost/mk_a4l1_parsecheck.py \
            || fail "parse-check generator failed for scenario $SCEN"
        # `env -u PARSE_CHECK`: the child would otherwise inherit the exported PARSE_CHECK=1 and re-enter this
        # very block, so every level would re-run the colour test one deep and none of them would print a
        # verdict — a recursion that presents as "the parser matched nothing".
        GOT=$(env -u PARSE_CHECK OUT="$OUT/pc_$SCEN" REPS=1 REPS_WIDE=1 SUMMARIZE=1 bash "$0" 2>&1 \
              | sed -n 's/^== VERDICT for two: \([A-Z]*\).*/\1/p' | tail -1)
        [ "$GOT" = "$WANT" ] || fail "parse-check: scenario $SCEN printed '${GOT:-<nothing>}', expected $WANT. The verdict parser is not reading its own table; no real window may be run."
        echo "PARSE-CHECK ok: scenario $SCEN -> $GOT (expected $WANT)"
    done
    echo "PARSE-CHECK PASS: the parser distinguishes all three colours on synthetic logs. This says nothing about the engine."
    exit 0
fi

[ -x "$BIN_NEW" ] || fail "shipping binary missing: run engine/build.sh"
[ -f "$IDS4096" ] || fail "no 4,096-id file at $IDS4096"

if [ "${SUMMARIZE:-0}" != "1" ]; then
    if [ ! -x "$BIN_OLD" ]; then
        echo "building baseline $BASE into $BASEDIR (CPU only)"
        [ -d "$BASEDIR" ] || git worktree add "$BASEDIR" "$BASE" --detach -q || fail "worktree add failed"
        for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
        ( cd "$BASEDIR" && bash engine/build.sh >/dev/null ) || fail "baseline build failed"
    fi

    # --- CANARY A, outside the hold: the knob, and the refusal direction -----------------------------------
    # `base` predates the knob, so it must REFUSE `two` — and it must refuse for the right reason. Accepting
    # any non-zero exit passes a baseline that died of an unrelated cause; a baseline that ACCEPTED `two` means
    # the wrong binary is under test and the window would price one build twice.
    "$BIN_OLD" --attn-fattn-roll two --model "$MODEL" --ids "1,2,3" --steps 0 >/dev/null 2>"$OUT/canaryA_old.err"
    [ $? -ne 0 ] || fail "canary A: the BASELINE accepted --attn-fattn-roll two. $BASE is not a pre-knob tree, or the wrong binary is under test; nothing is timed."
    "$BIN_NEW" --attn-fattn-roll two --model "$MODEL" --ids "1,2,3" --steps 0 --out "$OUT/canaryA_new" >/dev/null 2>&1 \
        || fail "canary A: the SHIPPING binary refused --attn-fattn-roll two. Is $BIN_NEW built from this tree at all?"
    if "$BIN_NEW" --attn-fattn-roll sine --model "$MODEL" --ids "1,2,3" --steps 0 >/dev/null 2>"$OUT/canaryA_bad.err"; then
        fail "canary A: the new binary ACCEPTED --attn-fattn-roll sine. Parse-by-refusal is not what it claims."
    fi
    grep -q "Got 'sine'" "$OUT/canaryA_bad.err" \
        || fail "canary A: 'sine' was rejected but not named in the refusal (see $OUT/canaryA_bad.err) — a refusal that does not name the value is a usage dump, not a parse."
    echo "CANARY A ok: baseline refuses the knob; shipping binary accepts two and refuses a nonsense value by naming it"

    # --- CANARY B, outside the hold: the NOTE BLOCKS OF THE TWO BUILT BINARIES -----------------------------
    # The stale-binary canary. `engine/build.sh` is not bit-reproducible on this box, so `cmp` between two
    # binaries proves only that they are two builds, and every self-consistency check compiled INTO the binary
    # under test (a plan bill, a launch counter) agrees with itself and is blind to staleness. The register
    # budget is the field this change is REQUIRED to move: `attn_fattn_wave` must read 178/0/0/32768 in BOTH
    # binaries (the default side must not pay for the second instantiation) and `attn_fattn_wave_two` must
    # exist only in the new one at 200/0/0/32768 — Gate 1's reading, from the linked host binary.
    note() { "$PY" prototype/attn_percost/a4_l1_note_read.py "$1" --kernel "$2" 2>&1 \
             | sed -n "s/^KERNEL $2 .*vgpr_count=\([0-9]*\) vgpr_spill_count=\([0-9]*\).*private_segment_fixed_size=\([0-9]*\) group_segment_fixed_size=\([0-9]*\).*/\1 \2 \3 \4/p"; }
    NOTE_BASE=$(note "$BIN_OLD" attn_fattn_wave);     need "$NOTE_BASE" "the base binary's attn_fattn_wave note block"
    NOTE_NEW=$(note "$BIN_NEW" attn_fattn_wave);      need "$NOTE_NEW" "the new binary's attn_fattn_wave note block"
    NOTE_TWO=$(note "$BIN_NEW" attn_fattn_wave_two);  need "$NOTE_TWO" "the new binary's attn_fattn_wave_two note block"
    [ "$NOTE_BASE" = "178 0 0 32768" ] || fail "canary B: base attn_fattn_wave reads '$NOTE_BASE', want '178 0 0 32768' (vgpr spill scratch lds). The comparator is not the shipped roll."
    [ "$NOTE_NEW" = "178 0 0 32768" ] || fail "canary B: the new binary's DEFAULT kernel reads '$NOTE_NEW', want '178 0 0 32768'. Adding the knob moved the register budget of the shipping path: the default side is not inert and its ms is not base's."
    [ "$NOTE_TWO" = "200 0 0 32768" ] || fail "canary B: attn_fattn_wave_two reads '$NOTE_TWO', want '200 0 0 32768' (Gate 1's reading). The variant the bet is about is not what got compiled."
    "$PY" prototype/attn_percost/a4_l1_note_read.py "$BIN_OLD" --kernel attn_fattn_wave_two >/dev/null 2>&1 \
        && fail "canary B: the BASELINE binary contains attn_fattn_wave_two. It is not a pre-knob build; every leg below would compare one binary with itself."
    echo "CANARY B ok: base 178/0/0/32768 with no second kernel; new default 178/0/0/32768 (inert); attn_fattn_wave_two 200/0/0/32768 (live) — read off the linked binaries in this window"

    # --- NUMERICS LEGS, outside the hold: deterministic, so they need no exclusive device ------------------
    # Not a bit-identity leg for `two`: the second accumulator RE-ASSOCIATES the score dot. The three claims
    # are (1) the default is inert bit for bit, (2) `two` actually moves bits — a flag that selected nothing
    # would print base's milliseconds a second time and read as a clean null — and (3) the band holds, which
    # is owned by `engine/gate.py --attn-fattn-roll two` (self-locking: run it OUTSIDE this window).
    # 640 ids, i.e. above MAXSC 512: at or below the staging the WMMA form owns NO rows (it prints
    # ATTN_FATTN_SKIP and is not launched), so a short id set would let `two` pass without running the kernel
    # the bet is about — which is exactly how a 500-token fixture passed the same knob in the worker's gate.
    "$PY" - "$OUT/ids640.txt" <<'PYIDS' || fail "could not build the 640-id numerics file"
import numpy as np, sys
ids = np.load('data/refs/wide4096p/input_ids.npy').reshape(-1)
open(sys.argv[1], 'w').write(','.join(str(int(x)) for x in ids[:640]))
PYIDS
    [ "$("$PY" -c "import sys;print(len(open(sys.argv[1]).read().split(',')))" "$OUT/ids640.txt")" = "640" ] \
        || fail "the numerics id file does not hold 640 ids"
    for SIDE in base single two; do
        BINX=$BIN_NEW; [ "$SIDE" = base ] && BINX=$BIN_OLD
        "$BINX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$OUT/ids640.txt")" --steps 0 \
            --prefill-logits all --state-fp $(roll_of "$SIDE") --out "$OUT/bi_$SIDE" \
            > "$OUT/bi_$SIDE.log" 2>&1 || fail "numerics leg: $SIDE did not finish"
    done
    for f in "$OUT/bi_base"/*; do
        b=$(basename "$f")
        cmp -s "$f" "$OUT/bi_single/$b" \
            || fail "default moved: --attn-fattn-roll single differs from the BASELINE on $b. The shipping path is not inert; stop."
    done
    grep -h "^STATEFP" "$OUT/bi_base.log"   > "$OUT/bi_base.fp"
    grep -h "^STATEFP" "$OUT/bi_single.log" > "$OUT/bi_single.fp"
    grep -h "^STATEFP" "$OUT/bi_two.log"    > "$OUT/bi_two.fp"
    cmp -s "$OUT/bi_base.fp" "$OUT/bi_single.fp" \
        || fail "default moved: the single side's STATEFP lines are not the baseline's. Inertness is the default's whole point."
    echo "NUMERICS leg 1 ok: base == single, bit for bit, on every prefill logit row and every STATEFP line"
    cmp -s "$OUT/bi_base.fp" "$OUT/bi_two.fp" \
        && fail "numerics leg 2: the two side's STATEFP is IDENTICAL to the baseline's. A second accumulator must move the score; if it does not, the roll is not running and its ms is the shipped roll's a second time."
    echo "NUMERICS leg 2 ok: two moves the state fingerprint — the second accumulator is live, not a flag that selects nothing"

    tools/gpu_lock.sh acquire "$TAG" || exit 1
    echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
    echo "CLK@start: $(clk)"

    for R in $(seq 1 "$REPS"); do
        for SIDE in base single two; do
            sleep 5
            BINX=$BIN_NEW; [ "$SIDE" = base ] && BINX=$BIN_OLD
            "$BINX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$IDS4096")" $ARGS $(roll_of "$SIDE") \
                --out "$OUT/$(tag_of "$SIDE")_$R" > "$OUT/$(tag_of "$SIDE")_$R.log" 2>&1 \
                || fail "$SIDE run rep=$R did not finish"
        done
        # --- CANARY C, inside the window: the banner of the kernel that ACTUALLY RAN ------------------------
        # `vgpr=` comes from `hipFuncGetAttributes` on the launched function, not from a name typed beside the
        # banner. The geometry fields (warps/lds/vtiles/wave_rows/split/fattn_rows) must be IDENTICAL on all
        # three sides — the knob picks a function, never a grid — and `vgpr=` is what separates single from two.
        # A side that echoed the other roll's banner would make the two halves of this A/B indistinguishable
        # while their milliseconds differed: the attribution failure AGENTS.md records for a check whose
        # independent variable is not the variable under test.
        for SIDE in single two; do
            LOG="$OUT/$(tag_of "$SIDE")_$R.log"
            grep -q "attn_fattn_roll=$SIDE" "$LOG" || fail "canary C: rep=$R $SIDE did not echo attn_fattn_roll=$SIDE (it echoed: $(grep -o 'attn_fattn_roll=[^ ]*' "$LOG" | tail -1)). The sides are not the rolls they claim; nothing is reported."
            grep -qE "ATTN_FATTN .*roll=$SIDE( .*vgpr=[0-9]+)+" "$LOG" || fail "canary C: rep=$R $SIDE printed no ATTN_FATTN banner naming roll=$SIDE. Either the WMMA form never launched at this width — then this bet has nothing to measure — or the banner does not name the roll."
        done
        VG=$(grep -oE "roll=single.*vgpr=[0-9]*" "$OUT/new_single_$R.log" | head -1 | sed 's/.*vgpr=//')
        VT=$(grep -oE "roll=two.*vgpr=[0-9]*"    "$OUT/new_two_$R.log"    | head -1 | sed 's/.*vgpr=//')
        VB=$(grep -o "form=fattn16 q=stage warps=[0-9]* vgpr=[0-9]*" "$OUT/base_$R.log" | head -1 | sed 's/.*vgpr=//')
        need "$VG" "the single side's launched vgpr"; need "$VT" "the two side's launched vgpr"; need "$VB" "the base side's launched vgpr"
        [ "$VG" = "178" ] || fail "canary C: rep=$R the single side's launched kernel reports vgpr=$VG, want 178."
        [ "$VT" = "200" ] || fail "canary C: rep=$R the two side's launched kernel reports vgpr=$VT, want 200."
        [ "$VB" = "178" ] || fail "canary C: rep=$R the base side's launched kernel reports vgpr=$VB, want 178."
        for F in warps lds vtiles wave_rows split fattn_rows fallback_rows; do
            GB=$(grep -o "$F=[^ ]*" "$OUT/base_$R.log"       | head -1)
            GS=$(grep -o "$F=[^ ]*" "$OUT/new_single_$R.log" | head -1)
            GT=$(grep -o "$F=[^ ]*" "$OUT/new_two_$R.log"    | head -1)
            need "$GS" "the $F field on the single side"; need "$GT" "the $F field on the two side"
            [ "$GB" = "$GS" ] && [ "$GB" = "$GT" ] \
                || fail "canary C: rep=$R geometry field $F differs across sides (base ${GB:-<none>} single $GS two $GT). The knob must not move the grid or the stage; if it did, this A/B compares two geometries and not two rolls."
        done
        # --- CANARY D: the launch plan is identical on every side (A4-BET's harness requirement) -----------
        for SIDE in base single two; do
            LOG="$OUT/$(tag_of "$SIDE")_$R.log"
            GOT=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1)
            PLAN=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* plan_launches=([0-9]+).*/\1/' | tail -1)
            MATCH=$(grep -h "WAVE gran=stage family=attn " "$LOG" | sed -E 's/.* counts_match=([01]).*/\1/' | tail -1)
            need "$GOT" "the attn launch count on $SIDE"; need "$PLAN" "the attn plan bill on $SIDE"
            [ "$MATCH" = "1" ] || fail "canary D: $LOG counts_match=0 on the attn row: the plan and the submissions disagree."
            [ "$PLAN" = "$GOT" ] || fail "canary D: $LOG plans $PLAN attn launches and submits $GOT."
            case $SIDE in base) LB=$GOT ;; single) LS=$GOT ;; two) LT=$GOT ;; esac
        done
        [ "$LB" = "$LS" ] && [ "$LB" = "$LT" ] \
            || fail "canary D: rep=$R attn launches differ across sides (base $LB single $LS two $LT). The knob submits one kernel either way; a launch-count move means it is not the only thing that changed."
        grep -q "TOTAL.*plan_match=1" "$OUT/new_two_$R.log" || fail "canary D: plan_match!=1 on the two side at rep=$R"
    done

    # --- the traced leg: the `attention` PHASE, the second line A4-BET adjudicates -------------------------
    # The untraced family row is what the stage clock bills; the traced phase row is what the kernels actually
    # spent. `--time-waves` is OFF here so the pass count stays honest, and the divisor is DERIVED from
    # `attn_prep` (48 sites per pass at 4,096 lanes) rather than typed, per the rocprofv3 trap in AGENTS.md.
    for R in $(seq 1 "$REPS"); do
        for SIDE in base single two; do
            sleep 5
            BINX=$BIN_NEW; [ "$SIDE" = base ] && BINX=$BIN_OLD
            "$PY" tools/prefill_kernel_trace.py --waves 8 --out "$OUT/trace_$(tag_of "$SIDE")_$R" -- \
                "$BINX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$IDS4096")" $TRACE_ARGS $(roll_of "$SIDE") \
                --out "$OUT/trace_$(tag_of "$SIDE")_${R}_out" \
                > "$OUT/trace_$(tag_of "$SIDE")_$R.log" 2>&1 \
                || fail "traced leg: the $SIDE rep=$R traced run did not finish"
            CSV=$(ls "$OUT/trace_$(tag_of "$SIDE")_$R"/*_kernel_trace.csv 2>/dev/null | head -1)
            need "$CSV" "the kernel_trace.csv for $SIDE rep=$R"
            "$PY" tools/trace_phase_split.py "$CSV" --engine ours --calibrate attn_prep --per-pass 48 \
                --tokens "$W" > "$OUT/trace_$(tag_of "$SIDE")_$R.txt" 2>&1 \
                || fail "traced leg: trace_phase_split.py refused the $SIDE rep=$R census (see $OUT/trace_$(tag_of "$SIDE")_$R.log)"
            need "$(traced "$OUT/trace_$(tag_of "$SIDE")_$R.txt")" "the attention phase row on $SIDE rep=$R"
        done
    done

    # --- the wide legs: where the quadratic lives, and A4-BET calls both of them ship gates -----------------
    for WID in $WIDE_WIDTHS; do
        ids_file_of "$WID" "$OUT/ids$WID.csv"
        for R in $(seq 1 "$REPS_WIDE"); do
            for SIDE in base single two; do
                sleep 5
                BINX=$BIN_NEW; [ "$SIDE" = base ] && BINX=$BIN_OLD
                "$BINX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$OUT/ids$WID.csv" --cap "$WID" $ARGS $(roll_of "$SIDE") \
                    --out "$OUT/wide${WID}_$(tag_of "$SIDE")_$R" \
                    > "$OUT/wide${WID}_$(tag_of "$SIDE")_$R.log" 2>&1 \
                    || fail "wide leg $WID: $SIDE rep=$R did not finish"
            done
            for SIDE in single two; do
                grep -q "attn_fattn_roll=$SIDE" "$OUT/wide${WID}_$(tag_of "$SIDE")_$R.log" \
                    || fail "wide leg $WID: the $SIDE side did not echo its roll at rep=$R. A wide ms from an unknown roll adjudicates nothing."
            done
        done
    done
    echo "CLK@end: $(clk)"
    echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
    tools/gpu_lock.sh release "$TAG"
fi

# ============================== SUMMARY: one window, five lines, one verdict =============================
echo
echo "== A4-L1 two-accumulator score roll. $REPS reps at $W lanes (untraced + traced), $REPS_WIDE reps at"
echo "   [$WIDE_WIDTHS] lanes, all inside one window. Lines are A4-BET.md's Gate-2 ratios (its number over its"
echo "   own base) applied to the base measured IN THIS WINDOW; absolutes are printed with the verdict:"
printf "   attn untraced        live <= %s x base   dead > %s x base   (A4-BET.md)\n" "$RAT_A_LIVE" "$RAT_A_DEAD"
printf "   attention traced     live <= %s x base   dead > %s x base   (A4-BET.md)\n" "$RAT_AT_LIVE" "$RAT_AT_DEAD"
printf "   wave wall_ms @%-6s live <= %s x base   dead > %s x base   (A4-BET.md) [SHIP GATE]\n" "$W" "$RAT_W_LIVE" "$RAT_W_DEAD"
printf "   wave ms @16,384      live <= %s x base   dead > %s x base   (A4-BET.md)\n" "$RAT_16_LIVE" "$RAT_16_DEAD"
printf "   wave ms @32,768      live <= %s x base   dead > %s x base   (A4-BET.md)\n" "$RAT_32_LIVE" "$RAT_32_DEAD"
printf "   externality          within %s x the non-attn base, INFORMATIONAL ONLY (not pre-registered)\n" "$EXT_TOL"

printf "\n%-16s %10s %10s %10s %14s\n" family base_ms single_ms two_ms share_of_wave
for F in $FAMS; do
    B=$(med_of fam base "$F"); S=$(med_of fam single "$F"); T=$(med_of fam two "$F")
    { [ "$B" = "NA" ] || [ "$T" = "NA" ]; } && fail "family row $F parsed empty (base=$B two=$T). The logs are missing or the field names moved; this is not a measurement."
    printf "%-16s %10s %10s %10s %14s\n" "$F" "$B" "$S" "$T" \
        "$(grep -h "family=$F " "$OUT/base_1.log" | sed -E 's/.*share_pct=([0-9.]+).*/\1/' | tail -1)"
done
BW=$(med_of tot base); SW=$(med_of tot single); TW=$(med_of tot two)
BA=$(med_of fam base attn); SA=$(med_of fam single attn); TA=$(med_of fam two attn)
BT=$(med_trace base); ST=$(med_trace single); TT=$(med_trace two)
need "$BW" "base wall_ms at 4,096"; need "$BA" "base attn family ms"; need "$TA" "the two side's attn family ms"
need "$BT" "base traced attention ms"; need "$TT" "the two side's traced attention ms"
# Absolute lines = A4-BET's pre-registered ratios x the base measured in THIS window (header block explains
# why ratios and not the §45 absolutes). An explicit env override still wins, unchanged.
LINE_A_LIVE=${LINE_A_LIVE:-$(awk -v b="$BA" -v r="$RAT_A_LIVE" 'BEGIN{printf "%.2f", b*r}')}
LINE_A_DEAD=${LINE_A_DEAD:-$(awk -v b="$BA" -v r="$RAT_A_DEAD" 'BEGIN{printf "%.2f", b*r}')}
LINE_AT_LIVE=${LINE_AT_LIVE:-$(awk -v b="$BT" -v r="$RAT_AT_LIVE" 'BEGIN{printf "%.2f", b*r}')}
LINE_AT_DEAD=${LINE_AT_DEAD:-$(awk -v b="$BT" -v r="$RAT_AT_DEAD" 'BEGIN{printf "%.2f", b*r}')}
LINE_W_LIVE=${LINE_W_LIVE:-$(awk -v b="$BW" -v r="$RAT_W_LIVE" 'BEGIN{printf "%.2f", b*r}')}
LINE_W_DEAD=${LINE_W_DEAD:-$(awk -v b="$BW" -v r="$RAT_W_DEAD" 'BEGIN{printf "%.2f", b*r}')}
printf "\n%-24s %10s %10s %10s\n" "attn family, untraced" "$BA" "$SA" "$TA"
printf "%-24s %10s %10s %10s\n" "attention phase, traced" "$BT" "$ST" "$TT"
printf "%-24s %10s %10s %10s\n" "wave wall_ms @4,096" "$BW" "$SW" "$TW"

# THE NULL: base vs single, same family, same window. A real move here means the window is not a pair, and
# every ratio below would then be about window noise rather than about the roll.
NULL_PCT=$(awk -v b="$BA" -v s="$SA" 'BEGIN{printf "%.2f", 100*(s-b)/b}')
awk -v p="$NULL_PCT" -v tol="$NULL_TOL" 'BEGIN{ if ((p<0?-p:p) > tol) \
    printf "NULL BROKEN: base vs single differ by %s%% on the attn family, above the %s%% tolerance. Treat this window as unpaired and re-take; do not read the verdict.\n", p, tol; \
    else printf "NULL ok: base vs single %.2f%% on the attn family (tolerance %s%%), and bit-identical outside the hold.\n", p, tol }'

for WID in $WIDE_WIDTHS; do
    BB=$(med_wide "$WID" base); SS=$(med_wide "$WID" single); TTW=$(med_wide "$WID" two)
    need "$BB" "base wide ms at $WID"; need "$TTW" "the two side's wide ms at $WID"
    printf "%-24s %10s %10s %10s\n" "wave ms @${WID}" "$BB" "$SS" "$TTW"
done

# Externality: the sum of (two - base) over every family OTHER than attn, per width. A3's trap was a family
# row that improved while the wave did not, and the complement of the row is the only honest witness.
ext_of() { local WID=$1 E=0 F fb fs
    for F in $FAMS; do
        [ "$F" = attn ] && continue
        if [ "$WID" = "$W" ]; then fb=$(med_of fam base "$F"); fs=$(med_of fam two "$F")
        else fb=$(grep -h "family=$F " "$OUT/wide${WID}_base_1.log" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1)
             fs=$(grep -h "family=$F " "$OUT/wide${WID}_new_two_1.log" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1); fi
        { [ -z "$fb" ] || [ -z "$fs" ]; } && continue
        E=$(awk -v e="$E" -v b="$fb" -v s="$fs" 'BEGIN{printf "%.3f", e + (s - b)}')
    done
    echo "$E"
}
E4=$(ext_of "$W"); E16=$(ext_of 16384); E32=$(ext_of 32768)
need "$E4" "the externality sum at 4,096"
T16=$(med_wide 16384 two); T32=$(med_wide 32768 two)
B16=$(med_wide 16384 base); B32=$(med_wide 32768 base)
need "$T16" "the two side's 16,384-lane ms"; need "$T32" "the two side's 32,768-lane ms"
LINE_16_LIVE=${LINE_16_LIVE:-$(awk -v b="$B16" -v r="$RAT_16_LIVE" 'BEGIN{printf "%.2f", b*r}')}
LINE_16_DEAD=${LINE_16_DEAD:-$(awk -v b="$B16" -v r="$RAT_16_DEAD" 'BEGIN{printf "%.2f", b*r}')}
LINE_32_LIVE=${LINE_32_LIVE:-$(awk -v b="$B32" -v r="$RAT_32_LIVE" 'BEGIN{printf "%.2f", b*r}')}
LINE_32_DEAD=${LINE_32_DEAD:-$(awk -v b="$B32" -v r="$RAT_32_DEAD" 'BEGIN{printf "%.2f", b*r}')}
# informational externality band: a fraction of the non-attn share of the base wave at that width
XB4=$(awk -v w="$BW" -v a="$BA" -v t="$EXT_TOL" 'BEGIN{printf "%.2f", (w-a)*t}')
XB16=$(awk -v w="$B16" -v a="$(grep -h 'family=attn ' "$OUT/wide16384_base_1.log" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1)" -v t="$EXT_TOL" 'BEGIN{printf "%.2f", (w-a)*t}')
XB32=$(awk -v w="$B32" -v a="$(grep -h 'family=attn ' "$OUT/wide32768_base_1.log" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1)" -v t="$EXT_TOL" 'BEGIN{printf "%.2f", (w-a)*t}')

awk -v a="$TA" -v at="$TT" -v w="$TW" -v w16="$T16" -v w32="$T32" \
    -v ba="$BA" -v bat="$BT" -v bw="$BW" -v bw16="$B16" -v bw32="$B32" \
    -v e4="$E4" -v e16="$E16" -v e32="$E32" \
    -v al="$LINE_A_LIVE" -v ad="$LINE_A_DEAD" -v atl="$LINE_AT_LIVE" -v atd="$LINE_AT_DEAD" \
    -v wl="$LINE_W_LIVE" -v wd="$LINE_W_DEAD" -v l16="$LINE_16_LIVE" -v d16="$LINE_16_DEAD" \
    -v l32="$LINE_32_LIVE" -v d32="$LINE_32_DEAD" \
    -v x4="$XB4" -v x16="$XB16" -v x32="$XB32" 'BEGIN{
    live = (a <= al) && (at <= atl) && (w <= wl) && (w16 <= l16) && (w32 <= l32);
    dead = (a > ad) || (at > atd) || (w > wd) || (w16 > d16) || (w32 > d32);
    v = dead ? "DEAD" : (live ? "LIVE" : "UNRESOLVED");
    printf "\n== VERDICT for two: %s\n", v;
    printf "   attn untraced ms/pass : %.2f -> %.2f  (%+.2f %%)   live <= %s, dead > %s\n", ba, a, 100*(a-ba)/ba, al, ad;
    printf "   attention traced      : %.2f -> %.2f  (%+.2f %%)   live <= %s, dead > %s\n", bat, at, 100*(at-bat)/bat, atl, atd;
    printf "   wave wall_ms @4,096   : %.2f -> %.2f  (%+.2f %%)   live <= %s, dead > %s  [SHIP GATE]\n", bw, w, 100*(w-bw)/bw, wl, wd;
    printf "   wave ms @16,384       : %.2f -> %.2f  (%+.2f %%)   live <= %s, dead > %s\n", bw16, w16, 100*(w16-bw16)/bw16, l16, d16;
    printf "   wave ms @32,768       : %.2f -> %.2f  (%+.2f %%)   live <= %s, dead > %s\n", bw32, w32, 100*(w32-bw32)/bw32, l32, d32;
    printf "   externality ms        : %+.2f / %+.2f / %+.2f  (4,096 / 16,384 / 32,768)  band +/-%s / %s / %s  [informational, not a line]\n", e4, e16, e32, x4, x16, x32;
    printf "   All five rows must clear their live line for LIVE; any dead line kills. Externality decides nothing (A4-BET pre-registered no line for it; wall_ms already contains every family).\n";
    if (v == "UNRESOLVED") printf "   Report it as unresolved: do not ship, do not narrate (A4-BET Gate 2, G1 precedent).\n";
}'
echo
echo "Also report, with this: the base-vs-single NULL (above), the per-side rep spread, the clocks, the three"
echo "note-block readings (the canary that catches a stale binary), and the traced census files. A single rep"
echo "outside the spread is a discarded window, not a number. Run"
echo "  .venv/bin/python engine/gate.py --fixture wide640f --attn-fattn-roll two"
echo "OUTSIDE this window (it self-locks) for the numerics admission against the pins."
echo "DONE. logs in $OUT/"
