#!/usr/bin/env bash
# a6_g1_bits.sh — A6-G1's M1: the device bit-identity gate for `--attn-fattn-qstage stage|unstaged8`.
#
# WHAT IT ADJUDICATES. `prototype/attn_percost/A6-BET.md`, "Numerics", G1: the unstaged8 side is owed
# BYTE-IDENTICAL logits and STATEFP against the shipped default, on >= 2 shapes including a ragged tail, plus
# default-inertness of the knob. Unlike G0 (a warp count, which re-splits the round and is judged by the
# existing aggregates), G1 claims to move staging addresses and operand provenance and NOTHING else. So this
# script has no band in it and cannot be talked into one: either every byte matches or the premise is false, and
# a red here is the finding, not an inconvenience (AGENTS: "A red result reported red is a success").
#
# THREE SIDES PER SHAPE, and the two comparisons are different claims:
#   base   a `git worktree` at the pinned PRE-CHANGE sha, built as-is          -> HEAD's machine code
#   stage  the changed tree's binary, knob UNSET                              -> must equal base (inertness)
#   u8     the changed tree's binary, `--attn-fattn-qstage unstaged8`          -> must equal base (the bet)
# Transitivity is deliberate: comparing u8 against `stage` alone would pass if BOTH had moved away from HEAD in
# the same way, and comparing stage against HEAD alone would pass if the knob were a no-op.
#
# NON-VACUITY, which is the whole difference between this gate and a green light. Byte-identity is trivially
# satisfied by a kernel that never ran, and AGENTS records that class three times over (the `partalias` control
# scoring 0 of 1,024, the CPU mirror that agreed with itself, the empty-parse gate that printed PASS). So every
# leg must show, in its own log:
#   * `q=stage` / `q=unstaged8` naming the spelling that ran;
#   * `lds=` / `wf_per_cu=` ACTUALLY DIFFERENT between the two spellings (32768/6 vs 20480/9), read off the
#     kernel that ran — if they agree, the knob did not select a kernel and the identity below is arithmetic on
#     one run taken twice;
#   * `vgpr=` agreeing with the note block of the BUILT binary (the stale-binary guard, AGENTS 2026-10-17);
#   * `fattn_rows=n/N` with n > 0: a wave of only one-pass rows runs no WMMA kernel and proves nothing;
#   * identical launch counts and coverage fields on all three sides (this bet must not move launches,
#     progress/2026-10-21/performance-status-2026-10-21.md §41).
#
# WHAT IT REFUSES: an empty parse (every field is `need`-checked); a base binary whose note block disagrees with
# the shipped reading; a side whose banner disagrees with its own note block; a shape where the WMMA form owns
# zero rows; a moved byte (it prints the first differing file and exits 1).
#
# CORRECTNESS ONLY. It takes no timing, prints no ms, and must be run blocking under one `tools/gpu_lock.sh`
# hold. The script's own wall is printed because AGENTS asks a worker device run to report command + wall + util
# + "no timing taken"; it is not a measurement of anything.
#
# Usage:
#   bash prototype/attn_percost/a6_g1_bits.sh                  # shapes 640 and 1,111 (ragged tail)
#   BASE=<sha> bash prototype/attn_percost/a6_g1_bits.sh
#   SHAPES="640 1111 2048" bash prototype/attn_percost/a6_g1_bits.sh
#   BUILD_ONLY=1 bash prototype/attn_percost/a6_g1_bits.sh     # CPU only: build base, read note blocks
set -uo pipefail
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD

# ---- pins ------------------------------------------------------------------------------
# A PRE-change sha: `base` must be the thing the bet is against, not the thing that already contains it.
BASE=${BASE:-ed658fe}
SHORT=$(git rev-parse --short "$BASE" 2>/dev/null || echo "$BASE")
BASEDIR=${BASEDIR:-/tmp/a6g1-bits-base-$SHORT}
BIN_BASE=$BASEDIR/engine/build/decode_run
BIN_NEW=${BIN_NEW:-$REPO/engine/build/decode_run}
MODEL=models/Qwen3.5-0.8B
PY=.venv/bin/python
SHAPES=${SHAPES:-"640 1111"}
TAG=${TAG:-a6g1-bits}
IDS4096=${IDS4096:-progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt}
OUT=${OUT:-/tmp/a6g1-bits}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)          # absolute: a relative log path resolves inside a worktree

# Note-block expectations, read from `prototype/attn_percost/A6-result.md` §1 and re-confirmed against the
# built tree in `prototype/attn_percost/A6-G1-impl.md` §2 — not invented here.
NOTE_STAGE_WANT=${NOTE_STAGE_WANT:-"178 0 0 32768"}     # vgpr spill private group
NOTE_U8_WANT=${NOTE_U8_WANT:-"185 0 0 20480"}
BAN_STAGE_WANT=${BAN_STAGE_WANT:-"lds=32768 wf_per_cu=6"}
BAN_U8_WANT=${BAN_U8_WANT:-"lds=20480 wf_per_cu=9"}

fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
need() { [ -n "$1" ] || fail "empty parse where a field was required ($2). A check whose input can be empty cannot fail."; }
field() { grep -o "$2=[^ ]*" "$1" | head -1; }
note_of() { "$PY" prototype/attn_percost/a4_l1_note_read.py "$1" --kernel "$2" 2>&1 \
            | sed -n "s/^KERNEL $2 .*vgpr_count=\([0-9]*\) vgpr_spill_count=\([0-9]*\).*private_segment_fixed_size=\([0-9]*\) group_segment_fixed_size=\([0-9]*\).*/\1 \2 \3 \4/p"; }

ids_file_of() { local N=$1 DST=$2
    [ -f "$IDS4096" ] || fail "no id file at $IDS4096"
    cut -d, -f"1-$N" "$IDS4096" > "$DST"
    local have; have=$("$PY" -c "import sys;print(len(open(sys.argv[1]).read().split(',')))" "$DST")
    [ "$have" = "$N" ] || fail "width $N needs $N ids and $DST holds $have — refusing to judge a run whose prompt is not the width under test"
}

# ---- build the pre-change baseline (CPU only; nothing here touches the device) ----------
[ -x "$BIN_NEW" ] || fail "no changed-tree binary at $BIN_NEW — run engine/build.sh first (this script does not build the change)"
[ -d "$BASEDIR" ] || git worktree add "$BASEDIR" "$BASE" --detach -q || fail "worktree add failed for $BASE"
for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
[ -x "$BIN_BASE" ] || { echo "building base $BASE into $BASEDIR (CPU only)"; ( cd "$BASEDIR" && bash engine/build.sh >"$OUT/build_base.log" 2>&1 ) || fail "baseline build failed (see $OUT/build_base.log)"; }
[ "$BIN_BASE" -nt "$BASEDIR/engine/attn_fattn_kernels.h" ] || fail "the base binary is older than the source it came from"
[ "$BIN_NEW"  -nt "$REPO/engine/attn_fattn_kernels.h"    ] || fail "the changed-tree binary is older than the changed source it came from"

# ---- CANARY 1: the note blocks of the two LINKED binaries ------------------------------
NB_STAGE=$(note_of "$BIN_BASE" attn_fattn_wave);  need "$NB_STAGE" "the base binary's attn_fattn_wave note block"
NN_STAGE=$(note_of "$BIN_NEW"  attn_fattn_wave);  need "$NN_STAGE" "the new binary's attn_fattn_wave note block"
NN_U8=$(note_of "$BIN_NEW"     attn_fattn_wave_u8); need "$NN_U8" "the new binary's attn_fattn_wave_u8 note block"
[ "$NB_STAGE" = "$NOTE_STAGE_WANT" ] || fail "canary 1: base attn_fattn_wave reads '$NB_STAGE', want '$NOTE_STAGE_WANT'. The comparator is not the shipped form; nothing may be compared."
[ "$NN_STAGE" = "$NOTE_STAGE_WANT" ] || fail "canary 1: the new binary's DEFAULT kernel reads '$NN_STAGE', want '$NOTE_STAGE_WANT'. Adding the knob moved the default's budget, so the default side is no longer HEAD's machine code and inertness cannot be claimed."
[ "$NN_U8" = "$NOTE_U8_WANT" ] || fail "canary 1: the new binary's unstaged8 kernel reads '$NN_U8', want '$NOTE_U8_WANT' (A6-result.md §1: spill 0 and group_segment 20,480). The variant the bet is about is not what got compiled."
echo "CANARY 1 ok: base attn_fattn_wave $NB_STAGE | new attn_fattn_wave $NN_STAGE | new attn_fattn_wave_u8 $NN_U8   (vgpr spill private group, read off the LINKED binaries)"

if [ "${BUILD_ONLY:-0}" = "1" ]; then
    echo "BUILD_ONLY: built and read note blocks, no device work."
    exit 0
fi

# ---- the legs --------------------------------------------------------------------------
# One side's run: $1 = shape, $2 = side (base|stage|u8). Writes $OUT/s${1}_$2.log and $OUT/s${1}_$2/.
leg() {
    local N=$1 SIDE=$2 BX BXK QARG
    case $SIDE in
        base)  BX=$BIN_BASE; QARG=;                             BXK=attn_fattn_wave ;;
        stage) BX=$BIN_NEW;  QARG=;                             BXK=attn_fattn_wave ;;
        u8)    BX=$BIN_NEW;  QARG="--attn-fattn-qstage unstaged8"; BXK=attn_fattn_wave_u8 ;;
        *) fail "unknown side $SIDE" ;;
    esac
    "$BX" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$(cat "$OUT/ids$N.csv")" --steps 0 \
          --prefill-logits all --state-fp $QARG --out "$OUT/s${N}_$SIDE" \
        > "$OUT/s${N}_$SIDE.log" 2>&1 || fail "shape $N side $SIDE: decode_run did not finish (see $OUT/s${N}_$SIDE.log)"
    # CANARY 2: the spelling that ran, the geometry it ran with, and the note block it must agree with.
    local q vgpr lds wf cov want_ban
    q=$(field "$OUT/s${N}_$SIDE.log" q);        need "$q" "the q= field on shape $N side $SIDE"
    vgpr=$(field "$OUT/s${N}_$SIDE.log" vgpr); need "$vgpr" "the vgpr= field on shape $N side $SIDE"
    lds=$(field "$OUT/s${N}_$SIDE.log" lds);   need "$lds" "the lds= field on shape $N side $SIDE"
    wf=$(field "$OUT/s${N}_$SIDE.log" wf_per_cu); need "$wf" "the wf_per_cu= field on shape $N side $SIDE"
    cov=$(grep -o 'fattn_rows=[^ ]*' "$OUT/s${N}_$SIDE.log" | head -1); need "$cov" "the fattn_rows= field on shape $N side $SIDE"
    want_ban=$BAN_STAGE_WANT; [ "$SIDE" = u8 ] && want_ban=$BAN_U8_WANT
    [ "$q" = "q=$( [ "$SIDE" = u8 ] && echo unstaged8 || echo stage )" ] \
        || fail "canary 2: shape $N side $SIDE printed '$q'. The knob did not name the spelling that ran."
    [ "$lds $wf" = "$want_ban" ] || fail "canary 2: shape $N side $SIDE printed '$lds $wf', want '$want_ban'. Either the build is stale or the driver refused the shape — in both cases the bytes below are not evidence about the unstaged8 form."
    [ "$vgpr" = "vgpr=$(echo "$([ "$SIDE" = u8 ] && echo "$NN_U8" | cut -d' ' -f1 || echo "$NOTE_STAGE_WANT" | cut -d' ' -f1)")" ] \
        || fail "canary 2: shape $N side $SIDE banner $vgpr disagrees with that kernel's note block. Two objects are in play."
    local ncov; ncov=$(echo "$cov" | sed -E 's/fattn_rows=([0-9]+).*/\1/')
    [ "${ncov:-0}" -gt 0 ] || fail "non-vacuity: shape $N side $SIDE covers 0 rows with the WMMA form (fattn_rows=$cov). Byte-identity over rows this kernel never wrote proves nothing; pick a wider shape."
    # launch plan + coverage fields: this bet must not move launches (progress/2026-10-21/performance-status-2026-10-21.md §41) nor which
    # rows the form owns.
    # This is the PLANNED bill (`LAUNCH`, printed in every mode); the OBSERVED count (`family=attn launches=`)
    # needs --time-waves and is therefore asserted inside the A/B window, where timing is the point. A plan bill
    # is compiled into the binary and agrees with itself, so it is a pairing check here, not the canary — the
    # canaries are the note block and the banner above.
    local lau; lau=$(grep -h "^LAUNCH " "$OUT/s${N}_$SIDE.log" | sed -E 's/.* attn=([0-9]+).*/\1/' | tail -1)
    need "$lau" "the planned attn launch bill on shape $N side $SIDE"
    echo "$lau" > "$OUT/s${N}_$SIDE.launches"
    # The fields this form must NOT move: which rows it owns and how it slices them. `fattn_covers` does not
    # read a staging plan, so a difference here means the knob changed routing, not staging.
    : > "$OUT/s${N}_$SIDE.fields"
    for F in vtiles wave_rows split fattn_rows fallback_rows covered_rows_pct covered_work_pct; do
        field "$OUT/s${N}_$SIDE.log" "$F" >> "$OUT/s${N}_$SIDE.fields"
    done
    [ "$(wc -l < "$OUT/s${N}_$SIDE.fields")" = "7" ] \
        || fail "non-vacuity: shape $N side $SIDE parsed $(wc -l < "$OUT/s${N}_$SIDE.fields") of 7 coverage fields from its log. A comparison over a list that can be empty cannot fail."
    grep -h "^STATEFP" "$OUT/s${N}_$SIDE.log" > "$OUT/s${N}_$SIDE.statefp"
    [ -s "$OUT/s${N}_$SIDE.statefp" ] || fail "non-vacuity: shape $N side $SIDE wrote no STATEFP row, so --state-fp did not run and the state half of the claim is unadjudicated."
    echo "LEG $N $SIDE: $q $lds $wf $vgpr $cov plan_attn_launches=$lau statefp_rows=$(wc -l < "$OUT/s${N}_$SIDE.statefp")"
}

# BYTE comparison: every dumped file, then the STATEFP rows.
cmp_sides() {
    local N=$1 A=$2 B=$3
    local d1=$OUT/s${N}_$A d2=$OUT/s${N}_$B
    local nf; nf=$(find "$d1" -type f | wc -l)
    [ "$nf" -gt 0 ] || fail "shape $N side $A dumped 0 files into $d1; there is nothing to compare (a comparison over an empty set cannot fail)."
    local nd; nd=$(find "$d2" -type f | wc -l)
    [ "$nf" = "$nd" ] || fail "shape $N: side $A dumped $nf files and side $B dumped $nd. A missing file is a moved byte with extra steps."
    local bad; bad=$(diff -rq "$d1" "$d2" | head -5)
    [ -z "$bad" ] || fail "shape $N: $A vs $B DIFFER in the dumped tensors. First differences: $bad . G1 is claimed to move staging addresses only; a moved byte falsifies that premise. STOP and report."
    cmp -s "$OUT/s${N}_$A.statefp" "$OUT/s${N}_$B.statefp" \
        || fail "shape $N: $A vs $B STATEFP rows differ. Same verdict as a moved logit: the premise is false, report it red."
    [ "$(cat "$OUT/s${N}_$A.launches")" = "$(cat "$OUT/s${N}_$B.launches")" ] \
        || fail "shape $N: $A and $B submitted different attn launch counts ($(cat "$OUT/s${N}_$A.launches") vs $(cat "$OUT/s${N}_$B.launches")). progress/2026-10-21/performance-status-2026-10-21.md §41 closed launch count as a variable and this bet may not reopen it."
    local nf2; nf2=$(find "$d2" -type f | wc -l)
    cmp -s "$OUT/s${N}_$A.fields" "$OUT/s${N}_$B.fields" \
        || fail "shape $N: $A and $B differ in the form's coverage fields: $(paste -d'/' "$OUT/s${N}_$A.fields" "$OUT/s${N}_$B.fields" | tr '\n' ' '). The coverage predicate does not read a staging plan, so a difference here means the knob changed ROUTING and the byte equality above is judging different rows."
    echo "BYTE-IDENTICAL $N: $A vs $B — $nf dumped files equal, $(wc -l < "$OUT/s${N}_$A.statefp") STATEFP rows equal, attn plan bill equal ($(cat "$OUT/s${N}_$A.launches")), coverage fields equal ($(tr '\n' ' ' < "$OUT/s${N}_$A.fields"))"
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
START=$(date +%s)
for N in $SHAPES; do
    ids_file_of "$N" "$OUT/ids$N.csv"
    for SIDE in base stage u8; do leg "$N" "$SIDE"; done
    cmp_sides "$N" base stage      # default-inertness: the knob, unnamed, changed nothing
    cmp_sides "$N" base u8         # THE BET: unstaged8 is byte-identical to the shipped form
done
WALL=$(( $(date +%s) - START ))
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
echo "A6-G1-M1 PASS: default-inert and unstaged8 byte-identical on shapes [$SHAPES] (logits files + STATEFP), with the geometry canaries proving the unstaged8 kernel is the one that ran."
echo "PROTOCOL: command=bash prototype/attn_percost/a6_g1_bits.sh SHAPES=\"$SHAPES\"; wall=${WALL}s (script wall, correctness only); NO TIMING TAKEN — no ms figure exists in this output and none may be copied from it."
