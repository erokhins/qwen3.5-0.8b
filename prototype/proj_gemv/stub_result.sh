#!/usr/bin/env bash
# prototype/proj_gemv/stub_result.sh — a stand-in for ./proj_gemv that prints RESULT lines and does no
# GPU work. It exists because the sweep's timed path (interleaved order, per-row clock, VOID and LOWCLK
# exclusion, pooled min, winner choice, the plan pair) cannot otherwise be run by the worker who wrote
# it: AGENTS.md forbids this worker to take a timing, and a plumbing path that has never been executed is
# a path whose bugs get found by the main agent inside a measurement window.
#
#   BIN=$PWD/prototype/proj_gemv/stub_result.sh SKIP_GATE=1 bash prototype/proj_gemv/sweep_decode.sh 1 1 decode0
#
# The numbers are a fixed function of the requested geometry — never a measurement, never quoted, and
# deliberately ordered so a smoke run can check which arm a rig would pick:
#   * ms grows with rows/block and with threads, so the winner is predictable;
#   * an unpacked plan is 28 % slower than the packed one at the same shape, so the plan pair is exercised
#     with the right sign and a ratio that is CLEARLY NOT the banked real figure (-16.2 %), so nobody reads
#     a dry run as a measurement;
#   * --rows 32 with --threads 512 alone reports max/min 4.0, so one point must come back VOID;
#   * a clock below the floor is simulated with CLK_STUB (see sweep usage), not here.
set -u

# It also doubles as CLK_TOOL so a dry run can exercise the clock path too:
#   CLK_TOOL=<this file> STUB_CLK=800  -> every row comes back LOWCLK, on purpose.
if [ "${1:-}" = "--showclocks" ]; then
  printf 'GPU[0]\t\t: sclk clock level: 1: (%sMhz)\n' "${STUB_CLK:-2700}"
  exit 0
fi

mode=burst; plan=; variant=v0-baseline; rows=8; threads=0; iters=30; reps=3; warm=2; burn=0; preburn=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) mode="$2"; shift 2;;
    --plan) plan="$2"; shift 2;;
    --variant) variant="$2"; shift 2;;
    --rows) rows="$2"; shift 2;;
    --threads) threads="$2"; shift 2;;
    --iters) iters="$2"; shift 2;;
    --reps) reps="$2"; shift 2;;
    --warm) warm="$2"; shift 2;;
    --burn) burn="$2"; shift 2;;
    --preburn) preburn="$2"; shift 2;;
    *) shift;;
  esac
done
# The per-arm burn is echoed as the real launcher echoes it, so the sweep's "did the burn actually run"
# canary has something to check in a dry run too.
pre_plans=$(awk -v s="$preburn" 'BEGIN{printf "%d", s*380}')
pre_launches=$(awk -v s="$preburn" 'BEGIN{printf "%d", s*380*48}')

lanes=32
[ "$threads" = 0 ] && threads=$(( rows * lanes ))
# Order matters: "unpacked" contains "packed", and the first dry run proved it — the plan pair's unpacked
# arm came back with 48 launches and tripped its own canary.
case "$plan" in *unpacked*) launches=114; base=3.20;; *mixed*) launches=114; base=3.10;; *packed*) launches=48; base=2.5;; *) launches=48; base=2.5;; esac
[ "$variant" = v0-baseline ] && base=$(awk -v b="$base" 'BEGIN{printf "%.4f", b+0.5}')
# The point of the stub: a deterministic ms, so the rig's own choices can be checked, not guessed.
ms=$(awk -v b="$base" -v r="$rows" -v t="$threads" 'BEGIN{printf "%.6f", b + r/100 + t/100000}')
ratio=1.02; pt=ok
if [ "$rows" = 32 ] && [ "$threads" = 512 ]; then ratio=4.000; pt=void; fi
mx=$(awk -v m="$ms" -v q="$ratio" 'BEGIN{printf "%.6f", m*q}')

if [ "$mode" = burn ]; then
  printf 'RESULT mode=burn label=%s variant=stub rpb=%d plans=1234 launches=59232 wall_s=%s\n' \
         "$variant" "$rows" "${burn:-0}"
  exit 0
fi
printf 'RESULT mode=burst label=%s variant=stub rpb=%d lanes_per_row=%d vx=0 threads=%d jobs=%d launches=%d' \
       "$variant" "$rows" "$lanes" "$threads" "$launches" "$launches"
printf ' preburn_s=%.2f preburn_plans=%d preburn_launches=%d' "$preburn" "$pre_plans" "$pre_launches"
printf ' ms_per_plan=%s ms_min=%s ms_max=%s reps=%d warm=%d ratio=%s pt=%s us_per_launch=60.000' \
       "$ms" "$ms" "$mx" "$reps" "$warm" "$ratio" "$pt"
if [ "$pt" = void ]; then printf ' gbs=VOID probe_pct=VOID'; else
  printf ' gbs=150.00 probe_pct=62.00'; fi
printf ' x_packed=0 tile=1 neg=none\n'
