#!/usr/bin/env bash
# tools/cold_restore.sh — cold weight-restore: how long to make the text weights resident again
# on a device pool that holds none? Owned by `progress/2026-10-12/plan-2026-10-12.md` Q4 ("cold weight-restore
# row (~1 s load) into the perf ledger"); the row itself belongs to `bench/performance-status.md`
# §6, which only the main agent writes, so this script produces the line and the operator decides.
#
# WHAT IS BEING MEASURED, exactly. `engine/loader_test` (built to `engine/build/loader_test` by
# `engine/build.sh`) already times every phase of a load:
# header parse, `hipHostMalloc` staging, `hipMalloc` of the device slabs (pages reserved, not
# transferred), file -> pinned staging, the cold H2D into fresh destination pages, and the first
# GPU read pass over them — that whole sequence IS a restore from nothing, because a fresh process
# starts with an empty device pool. This script runs it N times in N separate processes and reports
# the spread; it adds nothing to what loader_test measures, and re-parses nothing.
#
# TWO DIFFERENT COLDS, and a row that does not say which it measured is worth nothing:
#   * device-cold — always. Fresh process, fresh `hipMalloc`, destination pages fresh. This is what
#     `H2D cold` and `cold load` mean in loader_test's output.
#   * host-cold — the 1.5457 GB of shard bytes are also out of the page cache, so the file -> staging
#     read goes to the NVMe instead to RAM. Needs root (`--drop-caches`); without it the host page
#     cache stays warm and the row must be labelled "device-cold, host-warm". `devices/` measured the
#     first GPU pass over fresh pages at 7x slow, which is why the distinction is spelled out here
#     rather than in the ledger prose.
#
# LOCKING: deliberately does NOT take `tools/gpu_lock.sh` itself — `prototype/glue/run.py`,
# `prototype/proj_gemv/run.py` and `prototype/gdn_gated_norm/run.py` self-lock and deadlock a caller
# that already holds it, and a script that guesses is worse than one that states the protocol. The
# caller holds the lock:
#     tools/gpu_lock.sh acquire cold-restore && tools/cold_restore.sh --runs 5 ; tools/gpu_lock.sh release cold-restore
#
# Usage:  tools/cold_restore.sh [--runs N] [--drop-caches] [--verify] [--model-dir DIR]

set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

RUNS=3
DROP=0
VERIFY=0
MODEL=models/Qwen3.5-0.8B
BIN=engine/build/loader_test   # engine/build.sh writes it here; fall back below if built elsewhere
[ -x "$BIN" ] || BIN=engine/loader_test

while [ $# -gt 0 ]; do
    case "$1" in
        --runs) RUNS=$2; shift 2 ;;
        --drop-caches) DROP=1; shift ;;
        --verify) VERIFY=1; shift ;;
        --model-dir) MODEL=$2; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [ ! -x "$BIN" ]; then
    echo "$BIN is missing or not executable; build it first with engine/build.sh" >&2
    exit 1
fi
echo "using $BIN"

VFLAG="--no-verify"
[ "$VERIFY" = 1 ] && VFLAG=""
if [ "$DROP" = 1 ] && [ "$(id -u)" != 0 ]; then
    echo "--drop-caches needs root and this shell is not root; falling back to host-warm." >&2
    DROP=0
fi

echo "== cold weight-restore, $RUNS fresh process(es) of $BIN $VFLAG, model $MODEL =="
echo "   kind: $([ "$DROP" = 1 ] && echo 'device-cold AND host-cache-dropped' || echo 'device-cold, host page cache warm')"
echo "   gpu util before: $(tools/gpu_lock.sh util 2>/dev/null || echo 'lock tool unavailable')"

best=""
: > /tmp/cold_restore_runs.txt
for i in $(seq 1 "$RUNS"); do
    if [ "$DROP" = 1 ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches || { echo "drop_caches failed" >&2; exit 1; }
    fi
    out=$("$BIN" $VFLAG --model-dir "$MODEL" 2>&1) || { echo "loader_test failed:"; echo "$out"; exit 1; }
    cold=$(printf '%s\n' "$out" | sed -n 's/^ *cold load *\([0-9.]*\) ms.*/\1/p' | head -1)
    wall=$(printf '%s\n' "$out" | sed -n 's/^ *load() wall *\([0-9.]*\) ms.*/\1/p' | head -1)
    h2d=$(printf '%s\n' "$out" | sed -n 's/^ *H2D cold *\([0-9.]*\) ms.*/\1/p' | head -1)
    stage=$(printf '%s\n' "$out" | sed -n 's/^ *stage *\([0-9.]*\) ms.*/\1/p' | head -1)
    warm=$(printf '%s\n' "$out" | sed -n 's/^ *warm *\([0-9.]*\) ms.*/\1/p' | head -1)
    if [ -z "$cold" ] || [ -z "$wall" ]; then
        echo "run $i: could not find the phase lines in loader_test's output; raw output follows"
        echo "$out"
        exit 1
    fi
    printf '   run %d: cold load %s ms | load() wall %s ms | H2D cold %s | stage %s | warm pass %s\n' \
        "$i" "$cold" "$wall" "$h2d" "$stage" "$warm"
    printf '%s %s\n' "$cold" "$wall" >> /tmp/cold_restore_runs.txt
    best=$(printf '%s\n%s\n' "$best" "$cold" | sed '/^$/d' | sort -n | head -1)
done

echo "   gpu util after:  $(tools/gpu_lock.sh util 2>/dev/null || echo 'lock tool unavailable')"
echo
echo "best-of-$RUNS cold load: ${best} ms   (raw list in /tmp/cold_restore_runs.txt)"
echo
echo "Ledger line for bench/performance-status.md §6 — fill in the spread and the util pair:"
KIND="device-cold, host-warm"
[ "$DROP" = 1 ] && KIND="device-cold and host-cache-dropped"
VRF="no verify"
[ "$VERIFY" = 1 ] && VRF="with byte verification"
FLAGS="--runs $RUNS"
[ "$DROP" = 1 ] && FLAGS="$FLAGS --drop-caches"
[ "$VERIFY" = 1 ] && FLAGS="$FLAGS --verify"
echo "  | cold weight-restore ($VRF, $KIND) | ${best} ms best of $RUNS | <spread> | engine/loader_test via tools/cold_restore.sh $FLAGS |"
