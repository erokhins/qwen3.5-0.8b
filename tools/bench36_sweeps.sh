#!/usr/bin/env bash
# bench §36 banner re-take, §1 sweep legs: prefill TTFT and decode marginal cost, clock-documented.
# Main-agent window: blocking, one lock hold, zero subagents. Neither sweep self-acquires the lock
# (their `gpu_lock` mentions are usage lines), so this script holds it for both.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=models/Qwen3.5-0.8B
PY=.venv/bin/python
OUT=tmp/gcc/bench36
TAG=bench36-sweeps

clk() { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk" | tr '\n' ' '; }

mkdir -p "$OUT"
tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
echo "CLK@start: $(clk)"
echo "--- prefill_sweep --lanes 64,256,1024,3072 --runs 3 ---"
"$PY" tools/prefill_sweep.py --model "$MODEL" --lanes 64,256,1024,3072 --runs 3 \
    > "$OUT/prefill.log" 2>&1 || { echo "ABORT prefill_sweep"; tools/gpu_lock.sh release "$TAG"; exit 1; }
tail -14 "$OUT/prefill.log"
echo "CLK@mid: $(clk)"
echo "--- context_sweep --lanes 128,512,1024,2048 --steps 32 --runs 3 ---"
"$PY" tools/context_sweep.py --model "$MODEL" --lanes 128,512,1024,2048 --steps 32 --runs 3 \
    > "$OUT/context.log" 2>&1 || { echo "ABORT context_sweep"; tools/gpu_lock.sh release "$TAG"; exit 1; }
tail -16 "$OUT/context.log"
echo "CLK@end: $(clk)"
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
echo "WINDOW CLOSED"
