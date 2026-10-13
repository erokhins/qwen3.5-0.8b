#!/usr/bin/env bash
# One `engine/gate.py` fixture run, with the GPU lock held inside the same call and the run's own
# provenance written into the log it produces.
#
#     tools/gate_locked_run.sh <fixture> [<out-dir>]        # out-dir default: tmp/gate_logs
#
# Why it exists: AGENTS.md requires a worker's device check to report the exact command, the wall time
# and `tools/gpu_lock.sh util` at the time, and a log that carries none of those is a number with no
# provenance. A bare `gate.py --fixture X > x.log` also invites running two of them concurrently; the
# lock is taken here so the protocol cannot be forgotten, and released on EXIT with the tag it acquired
# (`gpu_lock.sh release` refuses a tagless call — a script that forgets the tag leaves a stale holder and
# the next caller merely reclaims it by PID-liveness, so the serialization looks intact by accident).
#
# What it is NOT: a timing harness. The wall in the header is this script's own span and the gate prints
# `ms/step` figures the gate itself disclaims; neither belongs in `bench/performance-status.md`, which
# belongs to the main agent's exclusive-window protocol (bench/performance-status.md §6). This runs a
# correctness gate and nothing else.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1
fx="${1:?usage: gate_locked_run.sh <fixture> [<out-dir>]}"
out="${2:-tmp/gate_logs}"
mkdir -p "$out"
tag="gate-$fx"
tools/gpu_lock.sh acquire "$tag" || exit 1
trap "tools/gpu_lock.sh release '$tag'" EXIT
u0=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
t0=$(date +%s.%N)
.venv/bin/python engine/gate.py --fixture "$fx" > "$out/$fx.log" 2>&1
rc=$?
t1=$(date +%s.%N)
u1=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
{ echo "# cmd: .venv/bin/python engine/gate.py --fixture $fx"
  echo "# gpu_lock util at acquire:$u0 at release:$u1 | wall=${wall}s (by-product, not a measurement)"
  cat "$out/$fx.log"; } > "$out/$fx.log.tmp" && mv "$out/$fx.log.tmp" "$out/$fx.log"
echo "$fx rc=$rc util_acquire=$u0 util_release=$u1 wall=${wall}s log=$out/$fx.log"
echo "PASS lines: $(grep -c '\[PASS\]' "$out/$fx.log")  FAIL lines: $(grep -c '\[FAIL\]' "$out/$fx.log")"
grep -E '^  \[(TIE|TIE-NOTE|SLACK|SLACK-NOTE|expected-red|NEW-FAILURE)\]|^ENGINE GATE' "$out/$fx.log"
exit "$rc"
