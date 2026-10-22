#!/usr/bin/env bash
# run_locked.sh — one gpu_lock hold around ONE blocking command, for a worker's correctness-only
# device check.
#
# Why a script instead of an inline `acquire … ; cmd ; release`: the harness that gates this agent's
# shell calls cannot analyse a compound command with a subshell and `time` in it, and AGENTS.md
# already requires the lock to be acquired and released inside one shell block. This file is that
# block. It also refuses to run a command whose binary self-locks, because a self-locking child parked
# on a held lock looks exactly like a hung box (AGENTS: three gate drivers self-lock; grep -c gpu_lock).
#
# Worker contract (AGENTS: "GPU execution is serialized and belongs to the main agent"):
#   * correctness only, never timing — this script prints no duration, and nothing it runs may print
#     an ms/GB/s figure;
#   * under ~60 s wall;
#   * util is printed at acquire and release so the run is attributed.
set -uo pipefail
export LC_ALL=C
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../.." && pwd)
cd "$root" || exit 2
LOCK=tools/gpu_lock.sh
[ -x "$LOCK" ] || { echo "ABORT: $LOCK not found"; exit 2; }
[ $# -ge 2 ] || { echo "usage: run_locked.sh <tag-words> <command> [args…]"; exit 2; }
TAG=$1
shift
BIN=$1
# The self-lock check is on the file name we were given; a python driver is not a binary.
if [ -f "$BIN" ] && grep -q gpu_lock "$BIN" 2>/dev/null; then
    echo "ABORT: $BIN self-locks gpu_lock; holding it here would deadlock"
    exit 2
fi
echo "UTIL-AT-ACQUIRE $("$LOCK" util)"
"$LOCK" acquire "$TAG" || { echo "ABORT: could not acquire the GPU lock"; exit 2; }
trap '"$LOCK" release "$TAG" >/dev/null 2>&1 || true' EXIT INT TERM
"$@"
rc=$?
echo "UTIL-AT-RELEASE $("$LOCK" util)"
echo "RUN-EXIT rc=$rc"
"$LOCK" release "$TAG" >/dev/null 2>&1 || true
trap - EXIT INT TERM
exit $rc
