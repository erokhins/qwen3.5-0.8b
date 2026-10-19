#!/usr/bin/env bash
# a6_rf_run.sh — run the A6 Step 3 driver queries under the GPU lock, and report what AGENTS requires a worker
# device run to report: the exact command, its wall, `util` at acquire and release, and the words NO TIMING TAKEN.
#
# WHAT THE PROGRAM DOES AND DOES NOT DO. `a6_rf_probe` calls `hipSetDevice`, `hipGetDeviceProperties`,
# `hipFuncGetAttributes` and `hipOccupancyMaxActiveBlocksPerMultiprocessor` and nothing else: no kernel launch, no
# allocation, no memcpy, no burst, no best-of-N. A driver occupancy query is host-side arithmetic over the loaded
# code object (`tools/kernel_occupancy_probe.cpp` and `prototype/attn_wmma/occupancy_fattn_probe.cpp` are the
# precedents in this tree). It is therefore a device FACT read, not a measurement, and nothing printed here may
# enter a timing document. The `wall` line is the wall clock of the script — reported because AGENTS asks for it,
# not because anything is being timed.
#
#   bash prototype/attn_percost/a6_rf_run.sh [outdir]        # default /tmp/a6rf
set -euo pipefail
export LC_ALL=C        # the wall line below is printed with awk's %.2f; a locale that writes 0,25 is a number a reader cannot parse
REPO=$(git rev-parse --show-toplevel)
OUT=${1:-/tmp/a6rf}
TAG=${TAG:-a6rf-ladder}
cd "$REPO"
for side in base g1; do
    [ -x "$OUT/$side/a6_rf_probe" ] || { echo "no probe at $OUT/$side/a6_rf_probe — run a6_rf_ladder.sh first"; exit 1; }
done
tools/gpu_lock.sh acquire "$TAG"
echo "util@acquire=$(tools/gpu_lock.sh util | tail -1)"
start=$(date +%s.%N)
for side in base g1; do
    echo "-- $side"
    rc=0
    "$OUT/$side/a6_rf_probe" || rc=$?
    echo "exit=$rc   (0 = every stub reproduced A5 §4.1's derived LDS column)"
done
end=$(date +%s.%N)
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
awk -v a="$start" -v b="$end" \
    'BEGIN{printf "wall of the two driver-query runs: %.2f s — SCRIPT wall, reported per AGENTS; no kernel timing taken\n", b-a}'
tools/gpu_lock.sh release "$TAG"
echo "NO TIMING TAKEN: no kernel was launched; every number above is a driver answer about a code object."
