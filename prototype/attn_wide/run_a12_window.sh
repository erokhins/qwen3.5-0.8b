#!/usr/bin/env bash
# run_a12_window.sh — the A12 device window: the wide-shape block and the shipping 16-row WMMA form, paired at
# one shape, at as many lane counts as asked for. `prototype/attn_percost/A12-WIDE-BET.md` §3 step 3.
#
# MAY BE RUN ONLY BY THE MAIN AGENT — blocking, never backgrounded, with ZERO subagents active. Every millisecond
# this prints is main-agent-only by the standing rule (`AGENTS.md`, "GPU execution is serialized": a worker may
# run short correctness gates but no timing, and a backgrounded run keeps this session's own token stream alive
# and contaminates the number). A worker must not execute this file.
#
#   bash prototype/attn_wide/run_a12_window.sh                       # LANES="3072 16384 32768" REPS=3
#   LANES=32768 REPS=5 bash prototype/attn_wide/run_a12_window.sh    # the deciding row alone
#   OUT=tmp/attn_wide_time TAG=a12-wide-window bash prototype/attn_wide/run_a12_window.sh
#   WIDE_WARPS=8 WIDE_STAGE=64 ... same, and they go to BOTH builds (the rig refuses a driver built for other axes)
#
# IT TAKES `tools/gpu_lock.sh` ITSELF, acquire/release by TAG, and prints util at acquire and at release (recorded
# and then discounted: identical clean runs on this box read 24-91 %, `bench/performance-status.md` §6). Nothing it
# calls takes the lock — `time_wide` and `attn_wide` are self-contained binaries, not one of the three self-locking
# gate drivers that a lock-holding caller deadlocks (AGENTS.md); the release runs on every exit path via the trap.
#
# IN-WINDOW, IN THIS ORDER:
#   1. rebuild BOTH artifacts from source in one environment, so a stale `new` side cannot be timed: the
#      correctness driver and the rig, each binary and each listing from a single hipcc invocation, then an
#      mtime assert that the artifacts are newer than every source they are built from (a `cmp` between two builds
#      proves nothing here — `engine/build.sh` is not bit-reproducible on this box).
#   2. read the register reservation out of the CODE OBJECT, never the LLVM note block: `a11_read_vgpr.py` for the
#      96-thread shipping kernel (`.amdhsa_next_free_vgpr`, the field the driver actually reserves) and
#      `wide_regread.py` for both, because it is the same parser plus `.amdhsa_private_segment_fixed_size` — the
#      spill budget A12 spends its relaxation on — and because `a11_read_vgpr.py`'s blocks/CU arithmetic multiplies
#      by a hardcoded 96 threads per block, which is the other kernel's block size. The LDS the rig reports for
#      itself is then compared against `listing_lds.py`'s reading of the same code object.
#   3. hand each lane count to `time_wide`, which re-runs the correctness driver itself and prints NO ms figure
#      at all unless that pass and its own structural checks went green (`NO-TIMING:`, exit 2, treated as an abort
#      here). The canary lines and the rig's self-reported LDS are checked against the listing from THIS build.
#
# WHAT IT DELIBERATELY DOES NOT DO: it takes no timing on the engine (no `decode_run`, no `--ids-file` — the rig
# launches the two kernels directly at a synthetic causal shape), it does not touch the default trio or any knob,
# and it books nothing into `bench/performance-status.md`; the ledger rows are the main agent's to write.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-tmp/attn_wide_time}
LANES=${LANES:-"3072 16384 32768"}
REPS=${REPS:-3}
TAG=${TAG:-a12-wide-window}
LIST=$OUT/time_wide-hip-amdgcn-amd-amdhsa-gfx1151.s
WIDE_THREADS=$(( ${WIDE_WARPS:-8} * 32 ))

fail() { echo "ABORT: $*" >&2; exit 1; }
# The lock is released on EVERY exit path, including a `fail` from the middle of the window, and only if this
# script actually holds it — a failed acquire must not let us release somebody else's lock. (`ERR` is not used:
# without `set -e` it fires on the expected-false branch of `grep ... && fail`, which would drop the lock in the
# middle of a good window.)
HELD=0
release() { if [ "$HELD" = 1 ]; then HELD=0; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1 || true; fi; }
trap 'if [ "$HELD" = 1 ]; then echo "ABORT: the window ended before the timing finished; lock released" >&2; release; fi' EXIT

echo "== lock: acquire as '$TAG'"
tools/gpu_lock.sh status
if ! tools/gpu_lock.sh acquire "$TAG"; then exit 1; fi
HELD=1
echo "util at acquire: $(tools/gpu_lock.sh util | tail -1)   [recorded, then discounted]"

mkdir -p "$OUT"
echo
echo "== 1. rebuild both artifacts in-window (one hipcc call per artifact: binary and listing cannot disagree)"
bash prototype/attn_wide/build_wide.sh "$OUT"        || fail "build_wide.sh (the correctness driver) failed"
bash prototype/attn_wide/build_time_wide.sh "$OUT"   || fail "build_time_wide.sh (the rig) failed"

newest=$(ls -t prototype/attn_wide/time_wide.cpp prototype/attn_wide/attn_wide.h \
            prototype/attn_wide/attn_wide.hip.cpp prototype/attn_wide/build_wide.sh \
            prototype/attn_wide/build_time_wide.sh engine/attn_fattn_kernels.h \
            prototype/attn_decode/attn_decode_kernels.h | head -1)
for art in "$OUT/time_wide" "$OUT/attn_wide" "$LIST"; do
    [ -e "$art" ] || fail "missing $art"
    [ "$art" -nt "$newest" ] || fail "$art is not newer than $newest: this window would time a stale artifact"
done
echo "fresh: time_wide, attn_wide and the listing are all newer than $newest"

echo
echo "== 2. the register reservation, from the code object (.amdhsa_*), never the LLVM note block"
A11=$(python3 prototype/attn_percost/a11_read_vgpr.py "$LIST" attn_fattn_wave_u8) \
    || fail "a11_read_vgpr.py could not read $LIST"
echo "$A11"
echo "$A11" | grep -q attn_fattn_wave_u8 \
    || fail "a11_read_vgpr.py printed no row for attn_fattn_wave_u8: a table with no row is not a measurement"
for spec in "attn_fattn_wave_u8 96" "attn_wide_wave $WIDE_THREADS"; do
    set -- $spec
    echo "-- wide_regread.py --match $1 --threads $2  (reads private-scratch too; a11_read_vgpr.py does not)"
    python3 prototype/attn_wide/wide_regread.py "$LIST" --match "$1" --threads "$2" \
        || fail "wide_regread.py found no $1 block in $LIST"
done
EXPECT=$(python3 prototype/attn_wide/listing_lds.py "$LIST") \
    || fail "cannot read .amdhsa_group_segment_fixed_size for both kernels out of $LIST"
EXP_WIDE_LDS=${EXPECT%% *}
EXP_F_LDS=${EXPECT##* }
echo "listing LDS: attn_wide_wave=$EXP_WIDE_LDS B   attn_fattn_wave_u8=$EXP_F_LDS B"

for L in $LANES; do
    echo
    echo "== 3. lanes=$L reps=$REPS"
    LOG=$OUT/lanes${L}.log
    "$OUT/time_wide" --lanes "$L" --reps "$REPS" --driver "$OUT/attn_wide" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || fail "time_wide exited $rc at lanes=$L. If it printed NO-TIMING, the unmet precondition is named there."
    if grep -q "NO-TIMING" "$LOG"; then fail "NO-TIMING at lanes=$L; nothing was measured. Read the reason above."; fi
    grep -q "driver_witness: PASS" "$LOG" || fail "no driver witness in $LOG: the rig cannot have timed anything"
    for side in wide fattn_u8; do
        grep -q "CANARY form=$side" "$LOG" || fail "no CANARY line for $side in $LOG: which kernel ran?"
    done
    GOT_WIDE_LDS=$(grep -o "CANARY form=wide .*" "$LOG" | head -1 | sed -E 's/.*lds_reported=([0-9]+).*/\1/')
    GOT_F_LDS=$(grep -o "CANARY form=fattn_u8 .*" "$LOG" | head -1 | sed -E 's/.*lds_reported=([0-9]+).*/\1/')
    [ "$GOT_WIDE_LDS" = "$EXP_WIDE_LDS" ] \
        || fail "the rig reports the wide kernel at $GOT_WIDE_LDS B of LDS while THIS build's listing says $EXP_WIDE_LDS B: two different builds are in one window, and a paired A/B across them is not an A/B"
    [ "$GOT_F_LDS" = "$EXP_F_LDS" ] \
        || fail "the rig reports attn_fattn_wave_u8 at $GOT_F_LDS B of LDS, listing says $EXP_F_LDS B: same failure, other side"
    grep -E "^(RESULT|PAIR|BET)" "$LOG" || fail "no RESULT lines in $LOG: a run that prints no number printed nothing"
done

echo
echo "== close: util at release, then release '$TAG'"
echo "util at release: $(tools/gpu_lock.sh util | tail -1)"
release
tools/gpu_lock.sh status
echo "done. The ms figures above are the main agent's to book into bench/performance-status.md; the rig's own"
echo "limits block (printed per lane count) names what they do not mean."
