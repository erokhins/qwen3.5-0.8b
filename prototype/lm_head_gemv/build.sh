#!/usr/bin/env bash
# Build the first engine kernel. -O3 is not optional: this is timed code, and the
# project rule is never to benchmark an unoptimized build.
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o lm_head_gemv lm_head_gemv.hip.cpp
echo "built ./lm_head_gemv"
