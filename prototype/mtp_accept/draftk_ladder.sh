#!/usr/bin/env bash
# draftk_ladder.sh — hold the GPU for `draftk_ladder.py`, nothing else.
#
# The lock is acquired HERE, in a shell that lives for the whole pass, exactly as `m3_loop_rate.sh` does:
# a short-lived subprocess records a dead holder pid and `tools/gpu_lock.sh` reclaims a dead holder, so a
# Python-side acquire would advertise the pass as stealable while it ran. Blocking by construction: no
# `&`, no `nohup`, nothing backgrounded.
#
# NO TIMING: no millisecond figure is read or printed anywhere in this pass. The wall quoted at the end is
# this harness's own seconds, labelled as that, and the rates come from token agreement only.
#
# USAGE
#   prototype/mtp_accept/draftk_ladder.sh --fixture tmp/llama_input/ids_long_chat.csv --k 4 --steps 256 --tag templated
#   K=2 STEPS=32 prototype/mtp_accept/draftk_ladder.sh --corpus 12 --tag corpus12
#   GDN='' K=1 STEPS=32 prototype/mtp_accept/draftk_ladder.sh --corpus 12 --tag control-k1   # shipped defaults
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-.venv/bin/python3}
ARGS=()
[ -n "${K:-}" ] && ARGS+=(--k "$K")
[ -n "${STEPS:-}" ] && ARGS+=(--steps "$STEPS")
[ -n "${SCHED:-}" ] && ARGS+=(--sched "$SCHED")
[ -n "${SHIP_M1:-}" ] && ARGS+=(--ship-m1 "$SHIP_M1")
[ -n "${EXTRA:-}" ] && ARGS+=(--extra="$EXTRA")
# `GDN=''` means "do not pass --gdn-loop at all", i.e. the engine default — which is how the carried
# side-pass rate of 0.6641 was taken, and the control must be taken that way to be comparable with it.
if [ -n "${GDN+set}" ]; then GDN_ARG="$GDN"; else GDN_ARG=1; fi
ARGS+=(--gdn-loop "$GDN_ARG")
OUT=${OUT:-/tmp/draftk}
ARGS+=(--out "$OUT")
ARGS+=("$@")
# One JSON per tag, so one population's ladder cannot overwrite another's. A caller that passed its own
# --json wins (the parse keeps the last occurrence).
case " ${ARGS[*]} " in
    *" --json "*) ;;
    *) ARGS+=(--json "$OUT/ladder.json") ;;
esac

TAG=${TAG:-worker draftk ladder}
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $(tools/gpu_lock.sh util | tail -1)"
echo "== binary: ${BIN:-engine/build/decode_run}  ($(date -u +%FT%TZ))  rates only, no timing taken"
[ -n "${BIN:-}" ] && ARGS+=(--binary "$BIN")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")

s=$(date +%s)
$PY prototype/mtp_accept/draftk_ladder.py "${ARGS[@]}"
rc=$?
echo "== harness wall $(( $(date +%s) - s ))s (its own seconds, not an engine timing); no timing taken"
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
exit $rc
