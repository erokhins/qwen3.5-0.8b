#!/usr/bin/env bash
# -O3 is not optional: this is timed code, and the project rule is never to benchmark an
# unoptimized build. The rpath is required or the binary dies at load looking for
# libamdhip64.so.7 (prototype/README.md).
#
# -I../.. so the source can say #include "engine/loader.h" the way any other consumer will, and
# ../../engine/loader.cpp is compiled in: this op is the loader's first consumer, so it builds
# against it rather than hand-rolling an H2D for the 49 norm vectors.
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib -I../.. -o glue glue.hip.cpp ../../engine/loader.cpp
echo "built ./glue"
