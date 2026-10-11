#!/usr/bin/env bash
# -O3 is not optional (timed code; AGENTS.md) and the rpath is required or the binary dies at
# load looking for libamdhip64.so.7 (prototype/README.md). Links engine/loader.cpp because the
# one weight this op reads (linear_attn.norm.weight, §01.6 F32) arrives through the loader, the
# same way prototype/glue proved it for the 49 norm vectors.
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib -I../.. -o gdn_gated_norm gdn_gated_norm.hip.cpp ../../engine/loader.cpp
echo "built ./gdn_gated_norm"
