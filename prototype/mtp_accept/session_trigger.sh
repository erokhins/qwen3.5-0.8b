#!/usr/bin/env bash
# session_trigger.sh — take the GPU lock and run `session_trigger.py`. The python tool does not lock (same
# split as `session_lossless.sh` / `m3_loop_rate.sh`: a short-lived subprocess would record a dead holder).
# Correctness only: no millisecond figure anywhere, and the tool it runs prints none.
# SELF-LOCKING: do not run this while holding tools/gpu_lock.sh (`grep -c gpu_lock` before acquiring).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root" || exit 1
PY=${PY:-"$root/.venv/bin/python3"}
TAG=${TAG:-"mtp session trigger table (session_trigger.sh)"}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $("$root/tools/gpu_lock.sh" util | tail -1); no timing taken"
"$PY" "$here/session_trigger.py" "$@"
