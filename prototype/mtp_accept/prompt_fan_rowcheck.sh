#!/usr/bin/env bash
# prompt_fan_rowcheck.sh — does a retained norm row hold the position its index claims?
#
# THE QUESTION. `--mtp-prompt-lanes` used to read `sc.norm` directly, which capped the fan at the last
# prefill wave's row count (46 lanes on a 4,142-token prompt). It now reads a retained window copied out of
# the scratch during the wave (`engine/prefill.h::NormTailRetention`), and a window index that is off by one
# does not crash: it stages a lane built from the NEIGHBOURING position's hidden state — a plausible draft
# conditioned on the wrong token, which no device check presents as a fault. So the addressing needs a
# witness, not a review comment.
#
# TWO LEGS, because the failure modes are different and one control cannot catch both.
#
# LEG 1 — cross-binary, single-wave window. On a fixture where the fan fits inside the last wave, the OLD
# binary's addressing is legitimate, so the two binaries must seed the same lanes and therefore PROPOSE the
# same tokens: `emitted.txt` byte-identical and `MTPSTAT accept_rate=` equal. Any row off by one, a window
# anchored on the wrong end, or a copy landing at the wrong offset changes a proposal. The precondition
# (fan <= the rows the OLD side found) is READ off the old run's `MTPPROMPT` and the script refuses if it
# does not hold, because then the two sides seeded different populations and equality would prove nothing.
#
# LEG 2 — same binary, window spanning several waves. Leg 1 can only ever test a window that lives in ONE
# wave, which is precisely the case that used to work. The multi-wave path — one D2D per intersecting wave,
# each landing at `position - first_pos` — has no pre-change counterpart, so its witness is that the window's
# CONTENT must not depend on how the prompt happened to be split: `--wave-rows` moves the split and nothing
# else (a wave body is bit-identical to the per-row forms it replaces — `engine/prefill.h`), so the same fan
# through a 1-wave window and a many-wave window has to emit the same tokens. If a slice lands at the wrong
# offset when it comes from an earlier wave, the streams diverge.
#
# NO TIMING: no millisecond figure is read or printed. The wall at the end is this harness's own seconds.
#
# Usage:
#   prototype/mtp_accept/prompt_fan_rowcheck.sh                    # defaults below
#   prototype/mtp_accept/prompt_fan_rowcheck.sh /tmp/prewt/engine/build/decode_run \
#       engine/build/decode_run /tmp/pf/mid.csv 146 128
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

OLD=${1:-/tmp/prewt/engine/build/decode_run}
NEW=${2:-engine/build/decode_run}
IDS=${3:-/tmp/pf/mid.csv}
FAN=${4:-146}                    # leg 1: must fit the old side's last wave (mid fixture: 261 rows)
STEPS=${5:-128}
FAN2=${6:-261}                   # leg 2: wider than the 64-row split's last wave, so it spans several
MODEL=${MODEL:-$(ls -d models/Qwen*)}

[ -x "$OLD" ] || { echo "no pre-change binary at $OLD (git worktree add /tmp/prewt HEAD --detach && (cd /tmp/prewt && engine/build.sh))"; exit 1; }
[ -x "$NEW" ] || { echo "no patched binary at $NEW"; exit 1; }
N=$(awk -F',' '{print NF}' "$IDS" 2>/dev/null || echo 0)
[ "$N" -ge 8 ] || { echo "$IDS has $N comma fields — that is the one-token silent failure"; exit 1; }
echo "== fixture=$IDS fields=$N fan(leg1)=$FAN fan(leg2)=$FAN2 steps=$STEPS"

rm -rf /tmp/pf/rowcheck
mkdir -p /tmp/pf/rowcheck/l1_old /tmp/pf/rowcheck/l1_new \
         /tmp/pf/rowcheck/l2_zero_wide /tmp/pf/rowcheck/l2_zero_narrow \
         /tmp/pf/rowcheck/l2_wide /tmp/pf/rowcheck/l2_narrow
TAG=${TAG:-worker prompt-fan-rowcheck}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"
s=$(date +%s)
rc=0

run() {   # run OUT BINARY FAN [extra...]
    local out=$1 bin=$2 fan=$3; shift 3
    "$bin" --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" \
        --mtp-draft 1 --mtp-accept 1 --gdn-loop 1 --no-stop \
        --mtp-prompt-lanes "$fan" "$@" --out "$out" > "$out/run.log" 2>&1 \
        || { echo "RUN FAIL out=$out"; tail -4 "$out/run.log"; return 1; }
    grep -h "^MTPPROMPT" "$out/run.log" | sed "s|^|  [$(basename "$out")] |"
    grep -h "^MTPSTAT"   "$out/run.log" | sed "s|^|  [$(basename "$out")] |"
}

field() { tr ' ' '\n' < "$1" | sed -n "s/^$2=//p" | head -1; }

echo "--- LEG 1: cross-binary, window inside one wave (old addressing is legitimate here) ---"
run /tmp/pf/rowcheck/l1_old "$OLD" "$FAN" || exit 1
run /tmp/pf/rowcheck/l1_new "$NEW" "$FAN" || exit 1
old_rows=$(field /tmp/pf/rowcheck/l1_old/run.log norm_rows)
old_fitted=$(field /tmp/pf/rowcheck/l1_old/run.log fitted)
if [ -z "$old_rows" ] || [ "$FAN" -gt "$old_rows" ] || [ "$old_fitted" != "$FAN" ]; then
    echo "LEG 1 INCONCLUSIVE: fan=$FAN, old side found $old_rows rows and fitted $old_fitted. The old side"
    echo "                    clamped, so the two sides did not seed the same population. Use a fan <= $old_rows."
    exit 3
fi
if cmp -s /tmp/pf/rowcheck/l1_old/emitted.txt /tmp/pf/rowcheck/l1_new/emitted.txt; then
    echo "LEG 1: IDENTICAL — $(wc -l < /tmp/pf/rowcheck/l1_new/emitted.txt) emitted tokens;"
    echo "       retained row i holds position first_pos+i (same lanes the scratch used to hand over)."
else
    echo "LEG 1: DIFFERS — a retained row is not the row the scratch held. Diff the MTPPROMPT lines and the"
    echo "       first diverging ACCEMIT line before reading any fan curve."; rc=1
fi
oa=$(field /tmp/pf/rowcheck/l1_old/run.log accept_rate); na=$(field /tmp/pf/rowcheck/l1_new/run.log accept_rate)
echo "       accept_rate old=$oa new=$na"; [ "$oa" = "$na" ] || rc=1

echo "--- LEG 2: same binary, window spanning 1 wave vs spanning many (split must not reach the content) ---"
# Leg 2's inference needs the TARGET to be split-invariant, or a divergence would be blameable on the body
# rather than on the retention. So the precondition is measured, not assumed: the SAME pair of wave splits at
# fan 0 must emit the same target stream. If that pair already differs, leg 2 is inconclusive by construction.
run /tmp/pf/rowcheck/l2_zero_wide "$NEW" 0 || exit 1
run /tmp/pf/rowcheck/l2_zero_narrow "$NEW" 0 --wave-rows 64 || exit 1
if cmp -s /tmp/pf/rowcheck/l2_zero_wide/emitted.txt /tmp/pf/rowcheck/l2_zero_narrow/emitted.txt; then
    echo "LEG 2 precondition: fan-0 target streams are split-invariant, so any difference below is the seed's."
else
    echo "LEG 2 INCONCLUSIVE: the TARGET itself moves between wave splits on this fixture, so a seeded-pair"
    echo "                    difference would not be attributable. Leg 1 still adjudicates the row index."
    rc=3
fi
run /tmp/pf/rowcheck/l2_wide "$NEW" "$FAN2" || exit 1                      # wave_rows=512: one wave holds it
run /tmp/pf/rowcheck/l2_narrow "$NEW" "$FAN2" --wave-rows 64 || exit 1     # 64-row waves: it spans several
if cmp -s /tmp/pf/rowcheck/l2_wide/emitted.txt /tmp/pf/rowcheck/l2_narrow/emitted.txt; then
    echo "LEG 2: IDENTICAL — a $FAN2-lane window assembled from one wave and from many waves seeds the same"
    echo "       cache, so each wave's slice lands at position - first_pos, as engine/prefill.cpp claims."
else
    echo "LEG 2: DIFFERS — the retention depends on the wave split, i.e. a slice landed at the wrong offset."; rc=1
fi
echo "== harness wall $(( $(date +%s) - s ))s (its own seconds); NO TIMING taken"
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
