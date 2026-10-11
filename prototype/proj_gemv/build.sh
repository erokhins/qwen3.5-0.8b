#!/usr/bin/env bash
# -O3 is not optional: this is timed code, and the project rule is never to benchmark an
# unoptimized build. The rpath is required or the binary dies at load looking for
# libamdhip64.so.7 (prototype/README.md).
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o proj_gemv proj_gemv.hip.cpp
echo "built ./proj_gemv"
