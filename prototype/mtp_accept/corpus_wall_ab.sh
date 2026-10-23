#!/usr/bin/env bash
# corpus_wall_ab.sh — hold the GPU for the corpus-representative wall A/B, nothing else.
#
# The main agent runs every measurement (AGENTS.md): blocking, zero subagents, one lock hold so every arm of
# every rep shares one window. `corpus_wall_ab.py` does the work and owns the refusals; this script owns the
# lock and the util stamp at both ends, and refuses to start if the lock is held (a caller that already holds
# it would deadlock the child — the trap this repo already documents for the self-locking gate drivers).
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-.venv/bin/python3}
TAG=${TAG:-main-agent corpus wall ab}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"

$PY prototype/mtp_accept/corpus_wall_ab.py --out "${OUT:-/tmp/corpus_wall_ab}" "$@"
rc=$?
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
