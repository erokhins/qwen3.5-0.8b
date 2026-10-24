#!/usr/bin/env bash
# bench §36 banner re-take, ruler leg: the §25 ruler on the shipped default, clock-documented.
# Main-agent window: blocking, one lock hold, zero subagents. Does NOT self-acquire around other
# self-locking drivers (it only runs decode_run, which does not self-lock).
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=models/Qwen3.5-0.8B
IDS=$(cat progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt)
BIN=engine/build/decode_run
OUT=tmp/gcc/bench36
TAG=bench36-ruler
REPS=${REPS:-3}
mkdir -p "$OUT"

clk() { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk|mclk" | tr '\n' ' '; }

tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
echo "CLK@start: $(clk)"
for R in $(seq 1 "$REPS"); do
    sleep 10
    echo "CLK@rep$R pre: $(clk)"
    "$BIN" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$IDS" --steps 0 --time-waves --gemm lt \
        --gemm-pins all --gdn-loop 512 --out "$OUT/r$R" > "$OUT/r$R.log" 2>&1 \
        || { echo "ABORT: rep $R did not finish"; tools/gpu_lock.sh release "$TAG"; exit 1; }
    echo "CLK@rep$R post: $(clk)"
    grep -E "WAVE gran=stage TOTAL|^RESULT" "$OUT/r$R.log" | tail -2
done
echo "CLK@end: $(clk)"
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
echo "WINDOW CLOSED"
