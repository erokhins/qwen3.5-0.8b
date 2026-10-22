#!/usr/bin/env bash
# Build the batched tied-head rig. -O3 is not optional (project rule: never benchmark an -O0 build),
# and -Wl,-rpath,/opt/rocm/lib is required for HIP builds on this box (AGENTS trap table).
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o mtp_headbatch mtp_headbatch.hip.cpp
echo "built ./mtp_headbatch"
