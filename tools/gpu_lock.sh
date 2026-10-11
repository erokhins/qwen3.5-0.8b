#!/usr/bin/env bash
# gpu_lock.sh — serialize anything that touches the iGPU.
#
# Why: two processes on this box do not just share the memory controller, they lie about
# it. A clean single-process MLP cold sweep is 2.66-2.69 ms/token; with a second gate
# running concurrently the same command reports 3.37 ms (66% of the probe instead of 84%),
# and the config ranking changes as well as the absolute number. Correctness results are
# unaffected, timings are not, so timing must be exclusive.
#
# It is an mkdir lock: atomic, no stale-pid guessing, and a killed holder is cleaned by
# its own trap. If a holder is SIGKILLed the lock survives; `gpu_lock.sh steal` clears it.
#
#   tools/gpu_lock.sh acquire [tag]     waits until free, then holds it for this call
#   tools/gpu_lock.sh acquire-noblock   exits 1 if busy (for a quick "may I time now?")
#   tools/gpu_lock.sh release [tag]     release (normally the trap does it)
#   tools/gpu_lock.sh steal             force-release a lock left by a killed holder
#   tools/gpu_lock.sh status            who holds it
#
# Typical use inside a script that is about to measure:
#   tools/gpu_lock.sh acquire "proj_gemv sweep"
#   trap 'tools/gpu_lock.sh release "proj_gemv sweep"' EXIT
#   ./prototype/proj_gemv/proj_gemv --time ...
#
# Do NOT hold the lock while doing CPU-only work (torch fixture dumps, speccheck, editing).
set -uo pipefail

LOCKDIR=/tmp/pi_gpu.lock
TAGFILE=$LOCKDIR/holder

now() { date -u +%FT%TZ; }

case "${1:-status}" in
  acquire)
    tag="${2:-untagged}"
    n=0
    while ! mkdir "$LOCKDIR" 2>/dev/null; do
      if [ $((n % 15)) -eq 0 ]; then
        echo "gpu_lock: waiting on $(cat "$TAGFILE" 2>/dev/null || echo unknown)" >&2
      fi
      n=$((n+1)); sleep 2
    done
    printf '%s\t%s\tpid=%s\n' "$(now)" "$tag" "$$" > "$TAGFILE"
    ;;
  acquire-noblock)
    tag="${2:-untagged}"
    if mkdir "$LOCKDIR" 2>/dev/null; then
      printf '%s\t%s\tpid=%s\n' "$(now)" "$tag" "$$" > "$TAGFILE"
    else
      echo "gpu_lock: busy — $(cat "$TAGFILE" 2>/dev/null || echo unknown)" >&2
      exit 1
    fi
    ;;
  release)
    # Release only if we can prove it is ours or untagged; never steal someone else's.
    if [ -f "$TAGFILE" ] && grep -q "pid=$$" "$TAGFILE"; then
      rm -f "$TAGFILE"; rmdir "$LOCKDIR" 2>/dev/null || true
    elif [ ! -f "$TAGFILE" ]; then
      rmdir "$LOCKDIR" 2>/dev/null || true
    else
      echo "gpu_lock: not ours to release ($(cat "$TAGFILE"))" >&2
      exit 1
    fi
    ;;
  steal)
    echo "gpu_lock: forcing release of $(cat "$TAGFILE" 2>/dev/null || echo 'no lock')" >&2
    rm -f "$TAGFILE"; rmdir "$LOCKDIR" 2>/dev/null || true
    ;;
  status)
    if [ -d "$LOCKDIR" ]; then echo "HELD: $(cat "$TAGFILE" 2>/dev/null || echo 'no tag')"
    else echo "free"; fi
    ;;
  *) echo "usage: gpu_lock.sh {acquire|acquire-noblock|release|steal|status} [tag]" >&2; exit 2 ;;
esac
