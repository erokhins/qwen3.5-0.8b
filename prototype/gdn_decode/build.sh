#!/usr/bin/env bash
# Build the GDN decode kernel. -O3 always: timing an unoptimized build is banned (AGENTS.md).
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o gdn_decode gdn_decode.hip.cpp
echo "built ./gdn_decode"
