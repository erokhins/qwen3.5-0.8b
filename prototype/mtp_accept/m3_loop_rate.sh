#!/usr/bin/env bash
# m3_loop_rate.sh — hold the GPU for `m3_loop_rate.py`, nothing else.
#
# The lock is acquired HERE, in a shell that lives for the whole pass, and not inside the Python: a
# short-lived subprocess records a dead holder pid, and `tools/gpu_lock.sh` deliberately reclaims a dead
# holder's lock — so a Python-side acquire would advertise this pass as stealable while it was running.
# Blocking by construction: no `&`, no `nohup`, no backgrounded run.
#
# Usage (env overrides pass through to the Python):
#   prototype/mtp_accept/m3_loop_rate.sh                       # all 49 lines, both arms
#   LIMIT=8 STEPS=32 prototype/mtp_accept/m3_loop_rate.sh      # a slice
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-.venv/bin/python3}
ARGS=()
[ -n "${STEPS:-}" ] && ARGS+=(--steps "$STEPS")
[ -n "${START:-}" ] && ARGS+=(--start "$START")
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
[ -n "${LOSSLESS:-}" ] && ARGS+=(--lossless "$LOSSLESS")
[ -n "${MIN_ROUNDS:-}" ] && ARGS+=(--min-rounds "$MIN_ROUNDS")
# Draft-side conditioning knobs for the accept arm, e.g. EXTRA="--mtp-dense 1". Passed as ONE argv entry
# with `=` so argparse does not read a leading `--` as the next option. Never applied to the flag-off pair
# run, which is the greedy reference the stream is adjudicated against.
[ -n "${EXTRA:-}" ] && ARGS+=(--extra="$EXTRA")
OUT=${OUT:-/tmp/m3_loop_rate}
ARGS+=(--out "$OUT" --json "$OUT/m3_loop_rate.json")
ARGS+=("$@")

TAG=${TAG:-main-agent m3 loop accept rate}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"
echo "== binaries: ${BIN:-engine/build/decode_run}  ($(date -u +%FT%TZ))"
[ -n "${BIN:-}" ] && ARGS+=(--binary "$BIN")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")

$PY prototype/mtp_accept/m3_loop_rate.py "${ARGS[@]}"
rc=$?
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
