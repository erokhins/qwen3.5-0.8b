#!/bin/bash
# Correctness-only device witness for the accept loop (§4i of ACCEPT-BUILD.md). No timing is read from
# this run: the binary prints its own ms fields and none of them are used here.
#
# Needs the `dec8` prompt as a COMMA-separated id file (`--ids-file` splits on commas, so a space-joined
# file reads as one token and the run then "succeeds" at width 1 — the trap is in AGENTS.md). Rebuild it
# from the fixture's own ids if /tmp/dec8_ids.csv is gone.
# Lock protocol: acquire and release inside this one process, because the holder is this shell's pid.
set -u
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
export LC_ALL=C
BIN=engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
IDS=/tmp/dec8_ids.csv

tools/gpu_lock.sh acquire mtp2-al5 >/dev/null
tools/gpu_lock.sh util

rm -rf /tmp/bl0 /tmp/bl1
mkdir -p /tmp/bl0 /tmp/bl1

$BIN --model $MODEL --ids-file $IDS --steps 16 --gdn-loop 1 --out /tmp/bl0 >/tmp/bl0/log.txt 2>&1
echo "off rc=$?"

$BIN --model $MODEL --ids-file $IDS --steps 16 --gdn-loop 1 --mtp-draft 1 --mtp-accept 1 --out /tmp/bl1 >/tmp/bl1/log.txt 2>&1
echo "on rc=$?"

tools/gpu_lock.sh release mtp2-al5 >/dev/null
tools/gpu_lock.sh status

echo "OFF_TOKENS: $(grep -o 'sampled=[0-9]*' /tmp/bl0/log.txt | grep -o '[0-9]*' | tr '\n' ' ')"
echo "ON_TOKENS: $(grep -v '^#' /tmp/bl1/emitted.txt | tr '\n' ' ')"
echo "ON_ROUNDS: $(grep -c '^ACCEPT round' /tmp/bl1/log.txt)"
echo "ON_ACCEPTS: $(grep -c 'accept=1' /tmp/bl1/log.txt)"
echo "ON_REJECTS: $(grep -c 'accept=0' /tmp/bl1/log.txt)"
echo "ON_SKIPS: $(grep -c '^ROUND-SKIP' /tmp/bl1/log.txt)"
grep '^MTPSTAT' /tmp/bl1/log.txt
grep -c 'DRAFT-SKIP' /tmp/bl1/log.txt
