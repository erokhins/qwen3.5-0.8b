#!/usr/bin/env bash
# W9 window (main agent, timings) + the fixture matrix the worker left unfinished.
set -uo pipefail
cd "$(dirname "$0")/.."
LIB=tmp/decode_run_library.sh
HAND=tmp/decode_run_hand.sh
ls "$LIB" "$HAND" >/dev/null 2>&1 || { echo "MISSING WRAPPERS"; exit 1; }
chmod +x "$LIB" "$HAND"
echo "### fixture matrix, remaining four, both sides (correctness only)"
for fx in dec300f wide520f mt32f chunk64b; do
    for side in hand library; do
        [ "$side" = hand ] && bin="$HAND" || bin="$LIB"
        out=$(timeout 1200 tmp/gate_ab.sh "$side" "$fx" "$bin" 2>&1 | tail -3)
        printf '%-9s %-8s %s\n' "$fx" "$side" "$(echo "$out" | tr '\n' ' ' | cut -c1-110)"
    done
done
echo "### W9 TTFT, hand path"
.venv/bin/python tools/prefill_sweep.py --binary "$HAND" --lanes 64,256,1024,3072 --runs 3 2>&1 | grep -E '^\|'
echo "### W9 TTFT, library path"
.venv/bin/python tools/prefill_sweep.py --binary "$LIB" --lanes 64,256,1024,3072 --runs 3 2>&1 | grep -E '^\|'
echo "### wave attribution, library path, 1024-token prompt"
timeout 1200 ./engine/build/decode_run --ids "$(cat tmp/q10/ids1024.txt)" --steps 1 --time-waves --gemm library > tmp/wave1024_lib.log 2>&1
grep -E "WAVEPASS gran=stage|WAVEPARITY gran=stage|WAVE gran=stage family" tmp/wave1024_lib.log | cut -c1-118
