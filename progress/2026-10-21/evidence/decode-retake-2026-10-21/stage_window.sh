#!/usr/bin/env bash
# Decode stage-clock attribution window, 2026-10-21. Main agent, blocking, zero subagents active.
# Same instrument and same two points as progress/bench/evidence/ledger-2026-10-16/ §3: 128 lanes
# (112 prompt + 16 steps) and 4,032 lanes (4,000 + 32), medians of 3.
set -e
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
bash tools/gpu_lock.sh acquire gen-stage
bash tools/gpu_lock.sh util
for i in 1 2 3; do timeout 900 ./engine/build/decode_run --ids-file tmp/gen-2026-10-21-decode/ids128.csv --steps 16 --time-stages > tmp/gen-2026-10-21-decode/stage128_$i.log 2>&1; done
for i in 1 2 3; do timeout 900 ./engine/build/decode_run --ids-file tmp/gen-2026-10-21-decode/ids4032.csv --steps 32 --time-stages > tmp/gen-2026-10-21-decode/stage4032_$i.log 2>&1; done
bash tools/gpu_lock.sh util
bash tools/gpu_lock.sh release gen-stage
echo DONE
