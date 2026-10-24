#!/usr/bin/env bash
# W2 — in-situ per-kernel attribution of one 16,384-lane prefill, wide=0 vs wide=1, in ONE lock hold.
# Instrument: tools/prefill_kernel_trace.py (rocprofv3 kernel dispatch timestamps). Shares/ratios are the
# product; absolute ms under a trace are not ledger ms (the tool says so itself).
# Main agent only: blocking, zero subagents, one gpu_lock hold.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=tmp/a12why
mkdir -p "$OUT"
BIN=engine/build/decode_run
TAG=${TAG:-a12-why-trace}
LANES=${LANES:-16384}
WAVES=$(( (LANES + 511) / 512 ))
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

for wide in 0 1; do
  d=$OUT/kt${LANES}_wide$wide
  rm -rf "$d" "${d}_out"
  .venv/bin/python tools/prefill_kernel_trace.py --waves "$WAVES" --out "$d" -- \
      "$BIN" --mtp-accept 0 --mtp-draft 0 --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$LANES.txt" --steps 1 \
             --attn-fattn 1 --attn-fattn-wide "$wide" --out "${d}_out" > "$d.log" 2>&1 \
      || fail "trace wide=$wide rc=$? (see $d.log)"
  echo "############ wide=$wide  lanes=$LANES  ($d.log) ############"
  sed -n '/---- prefill_kernel_trace ----/,/^PER-KERNEL/p' "$d.log" | grep -E "CU count|kernel records|span of kernel|Σ kernel duration|DEVICE-BUSY|Σ idle"
  echo "--- attention-family rows (per-kernel table) ---"
  awk '/^PER-KERNEL/,/^BY WAVE/' "$d.log" | grep -E "^  *[0-9]" | grep -Ei "attn|wide" | head -14
done
