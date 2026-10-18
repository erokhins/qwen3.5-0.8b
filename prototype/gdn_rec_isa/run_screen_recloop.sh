#!/usr/bin/env bash
# Harness screen for D53 MODE `recloop` (N token-steps inside ONE launch) against `rec2c4`, the per-token
# spelling the engine actually runs (`bench/performance-status.md` §22, decision log D53).
#
# WRITTEN, NOT RUN. No device run in this file has been executed by the worker who wrote it: no launch,
# no burst, no ms. Everything below is for the main agent, blocking, zero subagents, inside ONE
# `tools/gpu_lock.sh` hold (AGENTS.md). This binary does NOT self-lock — check with
# `grep -c gpu_lock prototype/gdn_rec_isa/run.py` for the drivers that do; holding the lock around this
# one is correct.
#
# WHAT DECIDES WHAT.
#   * the CPU checks and the transcription diff are PRE-FLIGHTS: they say the source did not move, and
#     they run before the lock is taken so a device window is never spent finding a typo;
#   * the volume witness says the looped MODE touches MODE 0's bytes PER TOKEN-STEP (same traffic,
#     fewer submissions) — a loop that looked fast by doing less work fails here, not in the burst;
#   * the admission says bit-identity, which is the whole claim of the item: `ndiff_o_vs0=0
#     ndiff_state_vs0=0` in every case, at every span, with `rerun_ndiff=0` (a racy barrier disagrees
#     with itself before it disagrees with anyone, and the looped form reuses `redk`/`redq` across
#     iterations — see the barrier argument in `gdn_rec_isa_kernels.h`);
#   * the burst is a SCREEN at 128 tokens, not the verdict. The claim lives on the engine's `gdn_rec`
#     family row in situ (`run_ab_gdnloop.sh`), and §10 is the precedent that a burst winner can be an
#     in-situ loser.
#
# THE CANARY IS NOT A PLAN BILL. `gdn_rec` is deliberately not a `launch_stats` family and a `recloop`
# launch has no plan-bill row, so nothing in a plan table can prove the looped MODE is present. The three
# witnesses here are: the mode name itself (a binary without it exits 2), the transcription diff and the
# ISA resource/bill assertions read off the listing of the binary being timed (rec2c4 must still bill
# 1,374 and read 59 VGPR — if adding the looped kernel perturbed the shipping side, the A/B is no longer
# one-knob), and `recloop` at 1,530 / 75 VGPR / 0 spill / occupancy 16.
set -u
cd "$(dirname "$0")/../.."
HARNESS=prototype/gdn_rec_isa/gdn_rec_isa
LISTING=tmp/gdn_isa/gdn_rec_isa.s
OUT=tmp/gcc/recloop_screen
LOG="$OUT/burst.log"
mkdir -p "$OUT"
TAG=recloop_screen_ab

fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

echo "== build (SAVE_TEMPS so the listing under test is this build's)"
SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh || fail "harness build failed"
[ -f "$LISTING" ] || fail "no listing at $LISTING — the canary cannot be read"

echo
echo "== CANARY 1 (build time): the looped body is the shipping body, line for line"
.venv/bin/python prototype/gdn_rec_isa/check_recloop_transcription.py || fail "transcription diff"
.venv/bin/python prototype/gdn_rec_isa/check_recloop_transcription.py --selftest \
    || fail "the transcription checker cannot see its own mutations — it is decoration"

echo
echo "== CANARY 2 (build time): the census, on the listing of this binary"
# rec2c4 must still bill the recorded 1,374 with the unrolled walk shapes 16x8 / 16x8 (the A/B side is
# unchanged), and recloop must bill 1,530 with the SAME walk shapes (so its +156 slots are per-token
# addressing and loop control, not state math). If either shape moved, `loop_price.py`'s penalty term is
# describing a different kernel and must be re-taken before the burst is read.
.venv/bin/python prototype/gdn_rec_isa/isa_census.py --asm "$LISTING" \
    --kernels rec_kernel,rec2c4_kernel,recloop_kernel | tee "$OUT/census.txt" || fail "census"
grep -q "rec2c4_kernel: 1374 slots" "$OUT/census.txt" || fail "rec2c4 no longer bills 1374"
grep -q "recloop_kernel: 1530 slots" "$OUT/census.txt" || fail "recloop does not bill 1530"
n_walk=$(grep -c "walk loopB_state_update.*state-accesses/trip=16 trips=8" "$OUT/census.txt")
# Two, not three: `rec_kernel` is the pre-D33 form whose loopB is 8-per-trip x 16 trips, and the whole
# point of counting is that BOTH the shipping side and the looped side show the unrolled 16x8 shape.
[ "$n_walk" -eq 2 ] || fail "expected 2 kernels with an unrolled loopB (16x8), got $n_walk"

echo
echo "== CANARY 3 (build time): resources, from a compile pass — zero spill, unchanged occupancy"
hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage \
      -c prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp -o /dev/null -I. -Iengine 2>&1 \
    | grep -A12 -E "Function Name: (_Z13rec2c4_kernel|_Z14recloop_kernel)" \
    > "$OUT/resources.txt" || fail "resource pass"
cat "$OUT/resources.txt"
grep -A12 -E "Function Name: _Z14recloop_kernel" "$OUT/resources.txt" | grep -q "VGPRs Spill: 0" \
    || fail "recloop spills — register pressure is now the story, not the launch count"
grep -A12 -E "Function Name: _Z14recloop_kernel" "$OUT/resources.txt" | grep -q "Occupancy \[waves/SIMD\]: 16" \
    || fail "recloop occupancy moved off 16 waves/SIMD"

echo
echo "== CPU-only selftests (no device, no lock)"
"$HARNESS" --selftest-lane   | tee "$OUT/selftest-lane.txt"   | grep "ok=1" || fail "--selftest-lane"
"$HARNESS" --selftest-mirror | tee "$OUT/selftest-mirror.txt" | grep "ok=1" || fail "--selftest-mirror"
"$HARNESS" --selftest-loop   | tee "$OUT/selftest-loop.txt"
grep -q "SELFTEST loop.*bad=0 ok=1" "$OUT/selftest-loop.txt" || fail "--selftest-loop not clean"
# A witness whose lists can be empty proves nothing, so the counts are asserted, not admired.
grep -q "span_runs=32 want_span_runs=32" "$OUT/selftest-loop.txt" || fail "loop span matrix incomplete"
grep -q "defect_runs=40 want_defect_runs=40" "$OUT/selftest-loop.txt" || fail "loop defect matrix incomplete"
grep -q "missed_catch=0 wrong_catch=0 defects_unreachable=0" "$OUT/selftest-loop.txt" \
    || fail "a loop defect was missed, fired where it must not, or was unreachable"

echo
echo "== lock"
tools/gpu_lock.sh acquire "$TAG" || fail "gpu_lock acquire"
tools/gpu_lock.sh util | tee "$OUT/util.txt"

echo
echo "== VOLUME (no timing): same bytes per token-step as rec0, at three spans"
# Span 0 = one launch for the whole 8-token volume sequence; span 3 = a ragged 3+3+2, which is the case
# that proves a chunked loop still writes every `o` row and every state element (unwritten_o must be 0).
for loop in 0 3; do
    "$HARNESS" --modes rec2c4,recloop --loop "$loop" --volume | tee "$OUT/volume-loop$loop.log" \
        || fail "volume run at --loop $loop"
    n=$(grep -c "^VOLUME mode=" "$OUT/volume-loop$loop.log")
    [ "$n" -eq 3 ] || fail "expected 3 VOLUME lines (rec0, rec2c4, recloop), got $n"
    grep "^VOLUME mode=" "$OUT/volume-loop$loop.log" \
        | grep -c "account_same_per_token_step_as_rec0=1" | grep -q "^3$" \
        || fail "a MODE's per-token-step traffic moved off rec0's (--loop $loop)"
    grep "^VOLUME mode=recloop" "$OUT/volume-loop$loop.log" | grep -q "verdict=VOLUME_OK" \
        || fail "recloop volume mismatch at --loop $loop"
done
echo "VOLUME OK at spans 0 and 3 — the loop moves submissions, not bytes"

echo
echo "== ADMISSION: bit-identity against rec0, at every span, asserted per line"
for loop in 0 1 3 16; do
    "$HARNESS" --modes rec2c4,recloop --loop "$loop" | tee "$OUT/admission-loop$loop.log" \
        || fail "admission run at --loop $loop"
    all=$(grep -c "mode=recloop" "$OUT/admission-loop$loop.log")
    bit=$(grep -c "mode=recloop .*ndiff_o_vs0=0 ndiff_state_vs0=0" "$OUT/admission-loop$loop.log")
    [ "$all" -gt 0 ] || fail "no recloop RESULT lines at --loop $loop — the binary does not carry the MODE"
    [ "$all" -eq "$bit" ] || fail "recloop is NOT bit-identical to rec0 in $((all-bit))/$all cases at --loop $loop"
    rer=$(grep -c "mode=recloop .*rerun_ndiff_o=0 rerun_ndiff_state=0" "$OUT/admission-loop$loop.log")
    [ "$rer" -eq "$all" ] || fail "recloop disagrees with its own rerun at --loop $loop: a barrier is" \
        " wrong (see the barrier argument in gdn_rec_isa_kernels.h), not the arithmetic"
    grep "RESULT mode=summary" "$OUT/admission-loop$loop.log" | grep -q "admission=PASS" \
        || fail "admission not PASS at --loop $loop"
    echo "admission --loop $loop: $all/$all cases bit-identical, reruns clean"
done

echo
echo "== THE PAIRED BURST: one window, both MODEs inside every rep (main agent only)"
tools/gpu_lock.sh util | tee -a "$OUT/util.txt"
"$HARNESS" --modes rec2c4,recloop --volume --burst 40 --reps 7 | tee "$LOG" || fail "burst loop=0"
grep "BURST-LINES" "$LOG" | grep "ok=1" || fail "BURST-LINES not fully timed"
"$HARNESS" --modes rec2c4,recloop --loop 32 --volume --burst 40 --reps 7 \
    | tee "$OUT/burst-loop32.log" || fail "burst loop=32"
grep "BURST-LINES" "$OUT/burst-loop32.log" | grep "ok=1" || fail "BURST-LINES not fully timed (loop=32)"
tools/gpu_lock.sh util | tee -a "$OUT/util.txt"
tools/gpu_lock.sh release "$TAG" || fail "gpu_lock release"
tools/gpu_lock.sh status

echo
echo "== summary (rec2c4 = 128 launches/sequence today; recloop = --loop'ed)"
python3 - "$LOG" "$OUT/burst-loop32.log" <<'PY'
import re, statistics, sys
for path in sys.argv[1:]:
    per = {}
    for ln in open(path).read().splitlines():
        m = re.match(r"BURST case=(\S+) rep=\d+ mode=(\S+) ms_per_sequence=([\d.]+)"
                     r" ms_per_launch=([\d.]+) launches=(\d+)", ln)
        if m:
            per.setdefault((m.group(1), m.group(2)), []).append(
                (float(m.group(3)), float(m.group(4)), int(m.group(5))))
    print(f"-- {path}")
    for case in sorted({c for c, _ in per}):
        a = per.get((case, "rec2c4"), []); b = per.get((case, "recloop"), [])
        if not a or not b:
            print(f"   {case}: missing data a={len(a)} b={len(b)}"); continue
        sa, sb = statistics.median(x[0] for x in a), statistics.median(x[0] for x in b)
        la, lb = statistics.median(x[1] for x in a), statistics.median(x[1] for x in b)
        pairs = sum(1 for x, y in zip(a, b) if y[0] < x[0])
        print(f"   {case:22} seq ms {sa:.4f} -> {sb:.4f}  ratio {sa/sb:.4f}"
              f"   per-launch ms {la:.5f} -> {lb:.5f}  ({b[0][2]} launches vs {a[0][2]})"
              f"   pairs favouring recloop {pairs}/{len(a)}")
print("read it as: does one launch covering the span beat 128 launches covering it?")
print("the ratio on ms_per_sequence is the screen; the per-launch ms is NOT comparable across the two")
print("(different amounts of work per launch). Engine verdict comes from the in-situ family row.")
PY

echo
echo "NOT TIMED BY THIS SCRIPT'S AUTHOR: no device run in this window was read by the worker who wrote"
echo "it. Numbers from here belong in bench/performance-status.md only if the main agent took them."
