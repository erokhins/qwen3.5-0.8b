#!/usr/bin/env bash
# session_lossless.sh — device gate: `--session --mtp-accept 1` must emit the same tokens, at the same
# positions, with the same `prefix_len`, as `--session --mtp-accept 0 --mtp-draft 0`.
#
# This is the session-path sibling of `flagoff_bytecheck.sh` (which admits the RUN path by diffing dump
# directories). A session has no dumps: its output IS the line protocol, so what gets compared is the
# `GEN`/`FED`/`END`/`PREFIX` stream. See `session_lossless.py` for the claim, the derived population and
# the structural cases; this file only holds the GPU lock and runs it.
#
# NO TIMING. No millisecond figure is read or printed. `wall` below is this harness's own seconds, and
# the run is correctness-only (`AGENTS.md`: short, device, PASS/FAIL, never a number).
#
# This script is SELF-LOCKING — do not run it while holding tools/gpu_lock.sh (AGENTS.md: a caller that
# holds the lock and then invokes a self-locking driver parks both silently).
#
# Usage:
#   prototype/mtp_accept/session_lossless.sh                 # full derived population
#   prototype/mtp_accept/session_lossless.sh --quick --long-gen 40
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
TAG=${TAG:-"mtp session lossless (session_lossless.sh)"}
PY=${PY:-$root/.venv/bin/python3}
[[ -x "$PY" ]] || PY=python3

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $($root/tools/gpu_lock.sh util | tail -1); no timing taken"
s=$SECONDS
"$PY" "$here/session_lossless.py" "$@"
rc=$?
echo "== harness wall $((SECONDS - s))s (its own wall, not a measurement)"
exit $rc
