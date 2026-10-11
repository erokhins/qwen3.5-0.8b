#!/usr/bin/env bash
# gpu_lock.sh — serialize anything that measures the iGPU.
#
# Why: two processes on this box do not just share the memory controller, they lie about it.
# A clean single-process MLP cold sweep is 2.66-2.69 ms/token; with a second gate running
# concurrently the same command reports 2.61-3.59 ms across four configs (documented value
# 2.650 ± 0.02), and the config ranking changes as well as the absolute number. Correctness
# results are unaffected, timings are not, so timing has to be exclusive.
#
# Note that "is the GPU idle" is not answerable by assumption here: flash_serve, the server
# generating this session's own model tokens, holds the iGPU at ~98% utilisation from its
# polling loop whether or not it is emitting tokens. Utilisation is therefore reported
# (`util`) and not used as a gate; in-flight concurrency is what corrupts a measurement.
#
#   tools/gpu_lock.sh acquire [tag]      wait until free, then hold it, recording `tag`
#   tools/gpu_lock.sh acquire-noblock[t] succeed immediately or exit 1
#   tools/gpu_lock.sh release [tag]      release; the tag must be the one acquired with
#   tools/gpu_lock.sh steal              force-release (for a lock you know is orphaned)
#   tools/gpu_lock.sh status             HELD / STALE / free
#   tools/gpu_lock.sh util               current rocm-smi GPU utilisation, for the record
#
# Ownership is the TAG, not a pid. A pid cannot work: every call is a separate short-lived
# process, and `bash -c 'a; b'` execs its last command, so acquire and release from the same
# script legitimately report different $PPIDs (observed: release refusing its own lock, and a
# leaked lock then wedging the next agent's sweep loop). A dead holder's lock is always
# reclaimable, so one SIGKILLed agent cannot stop the queue.
#
# Typical use:
#   tag="proj_gemv cold sweep"
#   tools/gpu_lock.sh acquire "$tag"; trap "tools/gpu_lock.sh release '$tag'" EXIT
#   ./prototype/proj_gemv/proj_gemv --time ...
#
# Do NOT hold the lock during CPU-only work (torch fixture dumps, speccheck, editing).
set -uo pipefail

LOCKDIR=/tmp/pi_gpu.lock
TAGFILE=$LOCKDIR/holder

now() { date -u +%FT%TZ; }
holder() { cat "$TAGFILE" 2>/dev/null || true; }
holder_pid() { holder | tr '\t' '\n' | sed -n 's/^pid=//p' | head -1; }
holder_alive() { local p; p=$(holder_pid); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
drop() { rm -f "$TAGFILE"; rmdir "$LOCKDIR" 2>/dev/null || true; }
util() { rocm-smi --showuse 2>/dev/null | awk -F: '/GPU use/ {gsub(/[^0-9]/, "", $NF); print $NF; exit}'; }

case "${1:-status}" in
  acquire)
    tag="${2:-untagged}"
    n=0
    while ! mkdir "$LOCKDIR" 2>/dev/null; do
      if ! holder_alive; then
        echo "gpu_lock: clearing lock left by a dead holder ($(holder))" >&2
        drop
        continue
      fi
      if [ $((n % 15)) -eq 0 ]; then echo "gpu_lock: waiting on $(holder)" >&2; fi
      n=$((n+1)); sleep 2
    done
    # pid is informational (who is holding it right now), never an ownership claim.
    printf '%s\t%s\tpid=%s\n' "$(now)" "$tag" "$PPID" > "$TAGFILE"
    ;;
  acquire-noblock)
    tag="${2:-untagged}"
    if mkdir "$LOCKDIR" 2>/dev/null; then
      printf '%s\t%s\tpid=%s\n' "$(now)" "$tag" "$PPID" > "$TAGFILE"
    else
      echo "gpu_lock: busy — $(holder)" >&2
      exit 1
    fi
    ;;
  release)
    tag="${2:-}"
    if [ ! -d "$LOCKDIR" ]; then
      exit 0
    elif ! holder_alive; then
      echo "gpu_lock: releasing lock left by a dead holder ($(holder))" >&2
      drop
    elif [ -n "$tag" ] && holder | grep -qF "	$tag"; then
      drop
    else
      echo "gpu_lock: refused; held as ($(holder)). Release with the tag you acquired with, or use \`steal\`." >&2
      exit 1
    fi
    ;;
  steal)
    echo "gpu_lock: forcing release of $(holder)" >&2
    drop
    ;;
  status)
    if [ ! -d "$LOCKDIR" ]; then echo "free"
    elif holder_alive; then echo "HELD: $(holder)"
    else echo "STALE (holder pid dead): $(holder) — acquire will reclaim it"; fi
    ;;
  util) echo "gpu use: $(util)%" ;;
  *) echo "usage: gpu_lock.sh {acquire|acquire-noblock|release|steal|status|util} [tag]" >&2; exit 2 ;;
esac
