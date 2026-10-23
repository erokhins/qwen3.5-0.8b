#!/usr/bin/env bash
# draft_cond_probe.sh — hold the GPU for `draft_cond_probe.py`, nothing else.
#
# Same lock protocol as `m3_loop_rate.sh` and for the same reason: the acquire happens in a shell that
# lives for the whole pass, because a short-lived subprocess records a dead holder pid and
# `tools/gpu_lock.sh` deliberately reclaims a dead holder's lock. Blocking by construction: no `&`,
# no `nohup`, no backgrounded run, and this harness takes no timing of any kind.
#
# Usage (env overrides pass through to the Python):
#   prototype/mtp_accept/draft_cond_probe.sh
#   FIXTURE=/tmp/dcp/dec8.csv STEPS=32 prototype/mtp_accept/draft_cond_probe.sh
#   ARMS="loop=--mtp-draft,1 --mtp-accept,1 dense=--mtp-draft,1,--mtp-accept,1,--mtp-dense,1" \
#       prototype/mtp_accept/draft_cond_probe.sh
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-.venv/bin/python3}
ARGS=()
[ -n "${FIXTURE:-}" ] && ARGS+=(--fixture "$FIXTURE")
[ -n "${STEPS:-}" ] && ARGS+=(--steps "$STEPS")
[ -n "${BINARY:-}" ] && ARGS+=(--binary "$BINARY")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")
[ "${ADMISSION:-1}" = "0" ] && ARGS+=(--no-admission)
OUT=${OUT:-/tmp/draft_cond_probe}
ARGS+=(--out "$OUT" --json "$OUT/draft_cond_probe.json")
if [ -n "${ARMS:-}" ]; then
    # Word-split on purpose: each whitespace-separated item is one `name=extra,args` arm.
    for a in $ARMS; do ARGS+=(--arm "$a"); done
fi
ARGS+=("$@")

TAG=${TAG:-worker draft-cond-probe}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"
echo "== no timing taken: this pass reports accept fractions only ($(date -u +%FT%TZ))"

$PY prototype/mtp_accept/draft_cond_probe.py "${ARGS[@]}"
rc=$?
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
