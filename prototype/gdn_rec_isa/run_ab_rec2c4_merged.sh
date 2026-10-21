#!/usr/bin/env bash
# Paired A/B for the MERGED `rec2c` form: MODE `rec2c4` (fused prologue + the shipping unroll4 walks)
# against MODE `receng` (`engine/gdn_rec_unroll4_kernels.h::rec_unroll4_kernel`, the spelling the engine
# actually runs). Written for the main agent; nothing in it has been run.
#
# WHY A CANARY, AND WHY IT IS NOT THE LAUNCH BILL. `progress/2026-10-21/performance-status-2026-10-21.md` §18 killed the previous
# screen because it compared two pre-unroll4 spellings, and `AGENTS.md` ("a paired A/B against a stale
# binary reads as a clean null") requires a behavioural canary read inside the window. The launch bill is
# unusable here: `gdn_rec` is deliberately NOT a `launch_stats` family (`engine/launch_stats.h:23`) and the
# merged form changes no launch count and no grid, so `plan_match`/`launchstats_match` are both green for a
# binary that never got the merge. The property the merge is REQUIRED to move is in the listing of the very
# binary being timed, so the canary is ISA-derived and read at build time, in this script, before any
# launch:
#
#   rec2c4_kernel loopB state-accesses/trip == rec_unroll4_kernel's  (the unroll survived the merge)
#   rec2c4_kernel barrier.sync              ==  5  while rec_unroll4_kernel's == 11  (prologue fused)
#   rec2c4_kernel slots                     == 1374 and rec_unroll4_kernel's == 1406
#
# A stale harness fails the mode name outright (`--modes rec2c4` exits 2), which is loud; a harness built
# from source where the merge silently gave the unroll back would still answer, and that is what the ISA
# canary catches. Both sides live in ONE binary and alternate inside every rep, so the window is one window.
#
# Run blocking, zero subagents, inside ONE `gpu_lock` hold, per AGENTS.md. Read the result as
# best_receng_ms / best_rec2c4_ms — the `FALSIFIER … rec0_over_mode=` lines compare against `rec0`, the
# PRE-unroll4 reference, and §18 is exactly the mistake that column invites.
set -u
cd "$(dirname "$0")/../.."
HARNESS=prototype/gdn_rec_isa/gdn_rec_isa
LISTING=tmp/gdn_isa/gdn_rec_isa.s
OUT=tmp/gcc/rec2c4_merged
LOG="$OUT/ab.log"
mkdir -p "$OUT"
TAG=rec2c4_merged_ab

fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

echo "== build (compile+link, SAVE_TEMPS so the listing under test is this build's)"
SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh || fail "harness build failed"
[ -f "$LISTING" ] || fail "no listing at $LISTING — the canary cannot be read"

echo
echo "== CANARY (ISA, read at build time, before any device launch)"
prototype/gdn_rec_isa/merged_census.py "$LISTING" \
    rec_unroll4_kernel,rec2c_kernel,rec2c4_kernel --expect merged || fail "merged-form canary failed"
# The numbers are pinned, not just the shapes: a build whose merged kernel bills something else is a
# different compiler decision than the one the census recorded, and a burst would then be pricing an
# unlabelled third form.
can=$(prototype/gdn_rec_isa/merged_census.py "$LISTING" rec_unroll4_kernel,rec2c4_kernel \
        | awk '/^rec_unroll4_kernel |^rec2c4_kernel /{print $1, $2, $3, $4, $5}')
echo "$can"
echo "$can" | grep -q "^rec_unroll4_kernel 326 352 728 1406$" || fail "rec_unroll4 bill moved off 1406: re-census before timing anything"
echo "$can" | grep -q "^rec2c4_kernel 294 352 728 1374$" || fail "rec2c4 bill moved off 1374: re-census before timing anything"
echo "CANARY OK — merged form is fused AND unrolled, and both bills are the recorded ones"

echo
echo "== CPU-only selftests (no device)"
"$HARNESS" --selftest-lane   | tail -1 || fail "--selftest-lane"
"$HARNESS" --selftest-mirror | tail -1 || fail "--selftest-mirror"

echo
echo "== lock"
tools/gpu_lock.sh acquire "$TAG" || fail "gpu_lock acquire"
tools/gpu_lock.sh util | tee "$OUT/util.txt"

echo
echo "== volume witness (no timing): the merged form must touch MODE 0's bytes, no more, no less"
"$HARNESS" --modes receng,rec2c4 --volume | tee "$OUT/volume.log" || fail "volume run"
# `rec0` is always active as the reference, so three MODEs are accounted, not two — and a count of the
# MODEs rather than a bare grep is the point: a MODE that never ran must not read as a pass.
n_vol=$(grep -c "^VOLUME mode=" "$OUT/volume.log")
[ "$n_vol" -eq 3 ] || fail "expected 3 VOLUME lines (rec0, receng, rec2c4), got $n_vol"
n_ok=$(grep -c '^VOLUME mode=.*verdict=VOLUME_OK' "$OUT/volume.log")
[ "$n_ok" -eq 3 ] || fail "expected 3 per-MODE verdict=VOLUME_OK lines, got $n_ok"
grep -q '^RESULT mode=volume-only failures=0 verdict=VOLUME_OK' "$OUT/volume.log" \
  || fail "volume RESULT line absent or not failures=0"
grep "VOLUME summary" "$OUT/volume.log" | grep -q "ok=1" || fail "volume summary not ok"

echo
echo "== admission (no timing): bit-identity against rec0, the only admission a spelling can have"
"$HARNESS" --modes receng,rec2c4 | tee "$OUT/admission.log" || fail "admission run"
n_bad=$(grep -c "mode=rec2c4 .*ndiff_o_vs0=0 ndiff_state_vs0=0" "$OUT/admission.log")
n_all=$(grep -c "mode=rec2c4" "$OUT/admission.log")
[ "$n_all" -gt 0 ] || fail "no rec2c4 RESULT lines at all — the binary does not carry the merged MODE"
[ "$n_bad" -eq "$n_all" ] || fail "rec2c4 is NOT bit-identical to rec0 in $((n_all-n_bad))/$n_all cases"
grep -E "RESULT mode=summary" "$OUT/admission.log" | grep -q "admission=PASS" || fail "admission not PASS"
echo "ADMISSION OK — rec2c4 bit-identical to rec0 in $n_all/$n_all cases (and receng too)"

echo
echo "== the paired burst: one window, both MODEs inside every rep, medians of --reps"
tools/gpu_lock.sh util | tee -a "$OUT/util.txt"
"$HARNESS" --modes receng,rec2c4 --volume --burst 40 --reps 7 | tee "$LOG" || fail "burst"
grep "BURST-LINES" "$LOG" | grep "ok=1" || fail "BURST-LINES not fully timed"
tools/gpu_lock.sh util | tee -a "$OUT/util.txt"
tools/gpu_lock.sh release "$TAG" || fail "gpu_lock release"
tools/gpu_lock.sh status

echo
echo "== summary (receng = the shipped spelling; rec2c4 = the merged form)"
python3 - "$LOG" <<'PY'
import re, statistics, sys
lines = open(sys.argv[1]).read().splitlines()
per = {}
for ln in lines:
    m = re.match(r"BURST case=(\S+) rep=\d+ mode=(\S+) ms_per_sequence=([\d.]+)", ln)
    if m:
        per.setdefault((m.group(1), m.group(2)), []).append(float(m.group(3)))
cases = sorted({c for c, _ in per})
for case in cases:
    a = per.get((case, "receng"), [])
    b = per.get((case, "rec2c4"), [])
    if not a or not b:
        print(f"{case}: missing data a={len(a)} b={len(b)}"); continue
    ma, mb = statistics.median(a), statistics.median(b)
    pairs = sum(1 for x, y in zip(a, b) if y < x)
    print(f"{case:24} receng med {ma:.4f}  rec2c4 med {mb:.4f}  ratio {ma/mb:.4f}"
          f"  pairs favouring rec2c4 {pairs}/{len(a)}")
print("kill line: D25's 1.02 on receng_over_rec2c4 (NOT rec0_over_rec2c4 — §18).")
print("census predicted ~1.018-1.019 from -32 slots at elasticity 0.8; the §18 screen read 1.0268")
print("against the pre-unroll4 base, so anything at or under 1.018 here says the merge bought the")
print("bill and nothing else, and rec2c closes for good.")
PY

echo
echo "NOT TIMED BY THIS SCRIPT'S AUTHOR: no device run in this window was read by the worker who wrote"
echo "it. Numbers from here belong in engine/performance-status.md only if the main agent took them."
