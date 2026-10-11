#!/usr/bin/env bash
# -O3 is not optional (project rule: never benchmark an unoptimized build) and the rpath
# is required or the binary dies at load looking for libamdhip64.so.7 (prototype/README.md).
# hipcc also compiles the host TU: loader.cpp carries the warm-read kernel.
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o loader_test loader_test.cpp loader.cpp
echo "built ./loader_test"
