#!/usr/bin/env bash
# Screen for the address-spelling MODE `recptr` (row pointers advanced by their strides instead of being
# re-materialised from the token index) against `recloop`, the shipped looped spelling it is derived from.
#
# WRITTEN, NOT RUN. The worker who wrote this ran no launch, no burst and no timing. The only device thing
# it ran is `occupancy_probe`, which launches nothing (it is a host-side calculator); everything below is
# for the main agent, blocking, zero subagents, inside ONE `tools/gpu_lock.sh` hold (AGENTS.md). Nothing in
# this directory self-locks — `grep -rl gpu_lock prototype/gdn_rec_isa/` finds only the screen scripts — so
# holding the lock here is correct and releasing it on every failure path is mandatory.
#
# WHAT THIS WINDOW IS FOR, AND WHAT THE CENSUS ALREADY SAID. The compile-only census (`token_bill.py`, and
# `RESULTS_addr.md` in this directory) says the address-spelling saves 4 flat slots of 1,530 per token-step
# (+0.26% of the bill goes the RIGHT way, but the per-token part goes the wrong way by 9), that
# address arithmetic is 93 slots/token-step (6.1%), not the ~156 the queue line implied, and that the
# runtime calculator gives both looped forms the same residency. So the honest expectation is a null. The
# window is worth taking only if the reviewer wants that null measured rather than inferred; the admission
# half is cheap and is the part that decides whether the MODE is even correct.
#
# THE CANARY IS BEHAVIOURAL, NOT A FILE COMPARE (`AGENTS.md`, the stale-binary row: `engine/build.sh` is not
# bit-reproducible, so a saved binary can pass as the new one and every parity field inside it agrees with
# itself). Before any timing is trusted this script requires, IN THE SAME WINDOW, that:
#   * the transcription checker passes AND its own mutation selftest passes (a checker that cannot fail is
#     decoration);
#   * `token_bill.py` prints DIFFERENT per-token bills for the two sides (1,327 for `recloop`, 1,336 for
#     `recptr`). If the two sides bill identically, the binary is stale or the MODE collapsed into the other
#     one, and a null from it means nothing;
#   * the resource rows differ in the one field the change is required to move (VGPRs 75 vs 69) and agree in
#     the ones it is not (LDS 1,280 B, spill 0).
set -u
cd "$(dirname "$0")/../.."
HARNESS=prototype/gdn_rec_isa/gdn_rec_isa
LISTING=tmp/gdn_isa/gdn_rec_isa.s
OUT=tmp/gcc/recptr_screen
mkdir -p "$OUT"
LOG="$OUT/burst.log"
TAG=recptr_screen_ab

fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

echo "== build (SAVE_TEMPS so the listing under test is this build's, not an older one)"
SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh || fail "harness build failed"
[ -f "$LISTING" ] || fail "no listing at $LISTING — the census canary cannot be read"

echo
echo "== CANARY 1 (build time): only the addressing moved, and the checker can see that"
.venv/bin/python prototype/gdn_rec_isa/check_recloop_transcription.py || fail "recloop transcription"
.venv/bin/python prototype/gdn_rec_isa/check_recptr_transcription.py  || fail "recptr transcription"
.venv/bin/python prototype/gdn_rec_isa/check_recptr_transcription.py --selftest \
    || fail "the recptr transcription checker cannot see its own mutations — it is decoration"

echo
echo "== CANARY 2 (build time): the per-token bill line DIFFERS between the two sides"
# This is the field the change is required to move. `recloop` must read 1,327 per-token / 1,530 flat and
# `recptr` 1,336 per-token / 1,526 flat, on the listing of the binary that is about to be timed.
.venv/bin/python prototype/gdn_rec_isa/token_bill.py --asm "$LISTING" \
    --kernels rec2c4_kernel,recloop_kernel,recptr_kernel --selftest \
    > "$OUT/token_bill.txt" 2>&1 || fail "token_bill selftest (the classifier cannot move)"
cat "$OUT/token_bill.txt"
for pair in "recloop_kernel:1327" "recptr_kernel:1336"; do
    k=${pair%%:*}; want=${pair##*:}
    grep -A8 "===== $k" "$OUT/token_bill.txt" | grep -q "PER-TOKEN bill        : $want slots" \
        || fail "$k does not bill $want slots/token on this build — stale binary or the MODE moved"
done
grep -A8 "===== recloop_kernel" "$OUT/token_bill.txt" | grep -q "FLAT token-step bill  : 1530" \
    || fail "recloop no longer bills 1530 flat; the shipped side moved and the A/B is no longer one-knob"

echo
echo "== CANARY 3 (build time): resources — VGPRs differ, LDS and spill do not"
hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage \
      -c prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp -o /dev/null -I. -Iengine 2>&1 \
    | grep -A12 -E "Function Name: (_Z14recloop_kernel|_Z13recptr_kernel)" > "$OUT/resources.txt" \
    || fail "resource pass"
for k in recloop_kernel recptr_kernel; do
    grep -A12 -E "Function Name: _Z1[34]$k" "$OUT/resources.txt" | grep -q "Spill: 0" \
        || fail "$k spills"
done
[ "$(grep -A12 -E 'Function Name: _Z14recloop_kernel' "$OUT/resources.txt" | grep -m1 'VGPRs:' \
     | tr -dc '0-9')" = "75" ] || fail "recloop VGPRs moved off 75"
[ "$(grep -A12 -E 'Function Name: _Z13recptr_kernel' "$OUT/resources.txt" | grep -m1 'VGPRs:' \
     | tr -dc '0-9')" = "69" ] || fail "recptr VGPRs are not 69 — the addressing did not move the way the" \
    " census read it, so nothing below is attributable to this item"

echo
echo "== residency (host-side calculator; launches nothing, times nothing)"
hipcc -O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib -I. -Iengine \
      prototype/gdn_rec_isa/occupancy_probe.cpp -o "$OUT/occupancy_probe" || fail "probe build"
"$OUT/occupancy_probe" | tee "$OUT/occupancy.txt" | grep -E "^OCC" || fail "occupancy canary"
grep -q "ok=1" "$OUT/occupancy.txt" || fail "the occupancy probe cannot see static LDS; its rows are void"

echo
echo "== CPU-only selftests (no device, no lock)"
"$HARNESS" --selftest-lane   | tee "$OUT/selftest-lane.txt"   | grep "ok=1" || fail "--selftest-lane"
"$HARNESS" --selftest-mirror | tee "$OUT/selftest-mirror.txt" | grep "ok=1" || fail "--selftest-mirror"
"$HARNESS" --selftest-loop   | tee "$OUT/selftest-loop.txt"
grep -q "SELFTEST loop.*bad=0 ok=1" "$OUT/selftest-loop.txt" || fail "--selftest-loop not clean"
grep -q "span_runs=32 want_span_runs=32" "$OUT/selftest-loop.txt" || fail "loop span matrix incomplete"
grep -q "defect_runs=40 want_defect_runs=40" "$OUT/selftest-loop.txt" || fail "loop defect matrix incomplete"

echo
echo "== lock"
tools/gpu_lock.sh acquire "$TAG" || fail "gpu_lock acquire"
tools/gpu_lock.sh util | tee "$OUT/util.txt"

echo
echo "== VOLUME (no timing): recptr touches rec0's bytes PER TOKEN-STEP, at a whole and a ragged span"
for loop in 0 3; do
    "$HARNESS" --modes rec2c4,recloop,recptr --loop "$loop" --volume \
        | tee "$OUT/volume-loop$loop.log" || fail "volume run at --loop $loop"
    n=$(grep -c "^VOLUME mode=" "$OUT/volume-loop$loop.log")
    [ "$n" -eq 4 ] || fail "expected 4 VOLUME lines (rec0, rec2c4, recloop, recptr), got $n"
    grep "^VOLUME mode=" "$OUT/volume-loop$loop.log" \
        | grep -c "account_same_per_token_step_as_rec0=1" | grep -q "^4$" \
        || fail "a MODE's per-token-step traffic moved off rec0's (--loop $loop)"
done
echo "VOLUME OK at spans 0 and 3 — the pointer form walks the same bytes it always walked"

echo
echo "== ADMISSION: bit-identity against rec0, at every span, asserted per line"
# The admission shape is `run_screen_recloop.sh`'s, unchanged: spans 0/1/3/16, and for every `recptr`
# result line both ndiff fields must be 0 and the rerun must reproduce it. Bit-identity is the whole claim
# of an addressing change: same addresses or nothing else was preserved.
for loop in 0 1 3 16; do
    "$HARNESS" --modes rec2c4,recloop,recptr --loop "$loop" \
        | tee "$OUT/admission-loop$loop.log" || fail "admission run at --loop $loop"
    for m in recloop recptr; do
        all=$(grep -c "mode=$m" "$OUT/admission-loop$loop.log")
        bit=$(grep -c "mode=$m .*ndiff_o_vs0=0 ndiff_state_vs0=0" "$OUT/admission-loop$loop.log")
        rer=$(grep -c "mode=$m .*rerun_ndiff_o=0 rerun_ndiff_state=0" "$OUT/admission-loop$loop.log")
        [ "$all" -gt 0 ] || fail "no $m RESULT lines at --loop $loop — the binary does not carry the MODE"
        [ "$all" -eq "$bit" ] || fail "$m is NOT bit-identical to rec0 in $((all-bit))/$all cases at --loop $loop"
        [ "$rer" -eq "$all" ] || fail "$m disagrees with its own rerun at --loop $loop"
    done
    grep "RESULT mode=summary" "$OUT/admission-loop$loop.log" | grep -q "admission=PASS" \
        || fail "admission not PASS at --loop $loop"
    echo "admission --loop $loop: both looped forms bit-identical, reruns clean"
done

echo
echo "== THE PAIRED BURST: one window, all three MODEs inside every rep (main agent only)"
# `recloop` vs `recptr` is the one-knob pair (addressing only); `rec2c4` rides along as the shipping
# per-token reference. Report ms per SEQUENCE and per LAUNCH: the two looped forms have equal launch
# counts at equal --loop, so unlike the D53 screen these per-launch numbers ARE comparable.
tools/gpu_lock.sh util | tee -a "$OUT/util.txt"
"$HARNESS" --modes rec2c4,recloop,recptr --loop 32 --volume --burst 40 --reps 7 \
    | tee "$LOG" || fail "burst"
grep "BURST-LINES" "$LOG" | grep "ok=1" || fail "BURST-LINES not fully timed"
tools/gpu_lock.sh util | tee -a "$OUT/util.txt"
tools/gpu_lock.sh release "$TAG" || fail "gpu_lock release"
tools/gpu_lock.sh status

echo
echo "== summary"
python3 - "$LOG" <<'PY'
import re, statistics, sys
per = {}
for ln in open(sys.argv[1]).read().splitlines():
    m = re.match(r"BURST case=(\S+) rep=\d+ mode=(\S+) ms_per_sequence=([\d.]+) ms_per_launch=([\d.]+)"
                 r" launches=(\d+)", ln)
    if m:
        per.setdefault((m.group(1), m.group(2)), []).append(
            (float(m.group(3)), float(m.group(4)), int(m.group(5))))
for case in sorted({c for c, _ in per}):
    a = per.get((case, "recloop"), []); b = per.get((case, "recptr"), [])
    if not a or not b:
        print(f"   {case}: missing data a={len(a)} b={len(b)}"); continue
    sa, sb = statistics.median(x[0] for x in a), statistics.median(x[0] for x in b)
    pairs = sum(1 for x, y in zip(a, b) if y[0] < x[0])
    print(f"   {case:22} recloop {sa:.4f} ms -> recptr {sb:.4f} ms  ratio {sa/sb:.4f}"
          f"   pairs favouring recptr {pairs}/{len(a)}")
print("read it against the census: the address-spelling is billed at 4 slots of 1,530 per token-step, so a")
print("pair ratio inside the rep-to-rep spread IS the null, and a ratio outside it means the census missed")
print("something (most likely a scheduling effect of the 6-VGPR drop, which the residency probe says is not")
print("a residency change at all).")
PY

echo
echo "NOT TIMED BY THIS SCRIPT'S AUTHOR: no device timing in this file was produced or read by the worker"
echo "who wrote it. Numbers from here belong in bench/performance-status.md only if the main agent took them."
