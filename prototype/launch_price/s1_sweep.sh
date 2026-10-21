#!/usr/bin/env bash
# s1_sweep.sh — the TIMED arm matrix for S1, and the command list for the main agent to run.
#
# WHO RUNS THIS. Every line it emits carries --time, so every line is a measurement. AGENTS.md reserves
# measurements to the main agent (a worker may run correctness-only device gates under a minute and
# prints no ms figure); this script is therefore a MAIN-AGENT script, and its default mode prints the
# commands instead of running them so a worker can hand the list over without touching the device.
#   ./s1_sweep.sh            print the command list (no device, no timing)
#   ./s1_sweep.sh --run      run it: acquires tools/gpu_lock.sh once and holds it for the whole matrix,
#                            because a ruler re-taken in a different window is not an A/B (AGENTS, the
#                            prefill_sweep row: 708 vs 770 ms for identical configs, across windows)
#
# WHAT THE ARMS ARE FOR:
#   ladder    N in {1,48,165,305,340,680} at the launch-dominated body, one drain per step. The slope
#             through these is the us-per-launch tax. {165,305,340} are in the set because the brief
#             asks for a ladder spanning the engine's 340, and 680 to show whether the tax doubles.
#   body      the same N at the byte-dominated bodies: if the slope there is ~0, dispatch was hidden
#             behind work and S1's prize is the interstice, not the issue cost.
#   drains    the engine arm's OWN 340 launches with 1, 305 and 340 drains. Same launches, same bytes,
#             same work, only the segment count moves — this is the D2 §5 experiment re-run at the real
#             launch count instead of 48, and it is the single most decision-relevant triple here.
#   syncs     launch count held at 340 while the sync count moves 340 -> 1: the axis that separates a
#             launch cost from a sync cost.
#   merge     the same total bytes at 340 / 8 / 1 launches. Their difference IS the prize, read
#             directly instead of inferred from a slope. Kept at a total that fits one launch's grid
#             (grid_cap), which is why the total is 64 MiB and not the step's 1.43 GiB.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")"
BIN=./s1_launch_price
REPS=${REPS:-5}
ITERS=${ITERS:-5}

run=0
[ "${1:-}" = "--run" ] && run=1

rows=()
add() { rows+=("$*"); }

# 1. the launch ladder, one drain per step
for n in 1 48 165 305 340 680; do
    add "--launches $n --syncs 1 --body tiny"
done
# 2. the same launch counts at the byte-dominated bodies
for b in mid fat; do
    for n in 48 340 680; do
        add "--launches $n --syncs 1 --body $b"
    done
done
# 3. the engine's own shape at three drain counts (D2 §5 at N=340)
for s in 1 305 340; do
    add "--pattern engine --body-mix engine --syncs $s"
done
# 4. sync count swept with the launch count pinned
for s in 340 170 43 1; do
    add "--launches 340 --syncs $s --body tiny"
done
# 5. merge: equal bytes and work, unequal launch counts. Two pairs. One launch cannot carry the step:
#    the per-launch ceiling is grid_cap(2048) x threads(256) x passes(8) x 8 B = 33.55 MiB, so the
#    step's 1461 MiB needs at least 44 launches. The small pair (64 MiB, down to N=1) prices the pure
#    launch-count effect; the step-scale pair (1461 MiB at 340 vs 48) prices it at the shipping
#    footprint, where the queue depth is the thing that differs.
for n in 340 8 1; do
    add "--launches $n --syncs 1 --total-mib 64"
done
for n in 340 48; do
    add "--launches $n --syncs 1 --total-mib 1461"
done
# 6. a cold-cache read of the shipping pattern is REFUSED by design (--pages cold aborts), so the
#    warm-up path is the fill plus one untimed step and there is nothing to add here.

if [ "$run" = "0" ]; then
    echo "# $(printf '%s\n' "${rows[@]}" | wc -l) timed arms. Run as:"
    echo "#   prototype/launch_price/s1_sweep.sh --run"
    echo "# or one at a time, inside a single gpu_lock hold, prefixing each with $BIN and suffixing:"
    echo "#   --time --reps $REPS --iters $ITERS | tee -a /tmp/s1_sweep.log"
    printf '%s\n' "${rows[@]}"
    exit 0
fi

# The timed pass. One lock hold for the whole matrix; blocking, never backgrounded (AGENTS).
cd ../..
tools/gpu_lock.sh acquire "S1 slope matrix (main agent, timed)"
echo "util at acquire: $(tools/gpu_lock.sh util)"
log=/tmp/s1_sweep.log
: > "$log"
for args in "${rows[@]}"; do
    # shellcheck disable=SC2086
    prototype/launch_price/s1_launch_price $args --time --reps "$REPS" --iters "$ITERS" 2>&1 | tee -a "$log"
done
echo "util at release: $(tools/gpu_lock.sh util)"
tools/gpu_lock.sh release "S1 slope matrix (main agent, timed)"
cd prototype/launch_price
echo
echo "# now fit it (no device). Each merge comparison is asserted equal-footprint, so both arms of a"
echo "# pair must share a --total-mib; run the two pairs as two invocations:"
echo "python3 prototype/launch_price/fit_slope.py --in $log \\\n    --engine-arm 'uniform|tiny|t1461|340|1|8|warm' --merge-arms 'uniform|tiny|t1461|48|1|8|warm'"
echo "python3 prototype/launch_price/fit_slope.py --in $log \\\n    --engine-arm 'uniform|tiny|t64|340|1|8|warm' --merge-arms 'uniform|tiny|t64|8|1|8|warm,uniform|tiny|t64|1|1|8|warm'"
