#!/usr/bin/env bash
# W1 — the coverage/grid census the tip commit asked for: which rows does the wide block own in the
# engine, and what grid does it get? Echo lines only; no timing taken here.
# Main agent only: blocking, one gpu_lock hold, zero subagents.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=tmp/a12why
mkdir -p "$OUT"
BIN=engine/build/decode_run
TAG=${TAG:-a12-why-census}
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

for lanes in 4096 16384; do
  for wide in 0 1; do
    log=$OUT/echo_${lanes}_wide${wide}.log
    /usr/bin/time -f "wall=%e s" "$BIN" --mtp-accept 0 --mtp-draft 0 --model models/Qwen3.5-0.8B \
        --ids-file "$OUT/ids$lanes.txt" --steps 1 --attn-fattn 1 --attn-fattn-wide "$wide" \
        --out "$OUT/out_${lanes}_wide${wide}" > "$log" 2>"$log.time" || fail "run lanes=$lanes wide=$wide rc=$?"
    echo "=== lanes=$lanes wide=$wide  ($(cat "$log.time" | tail -1)) ==="
    grep -E "^(ATTN_WIDE|ATTN_FATTN|SPLITCAP)" "$log" | head -8
    echo "    distinct ATTN_WIDE geometries : $(grep -c '^ATTN_WIDE' "$log")"
    echo "    distinct ATTN_FATTN geometries: $(grep -c '^ATTN_FATTN' "$log")"
    grep -E "^RESULT" "$log"
  done
done
