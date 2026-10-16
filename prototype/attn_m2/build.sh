#!/usr/bin/env bash
# -O3 is not optional: this is timed code and the project rule is never to benchmark an unoptimized
# build. The rpath is required or the binary dies at load looking for libamdhip64.so.7
# (prototype/README.md). `--offload-arch=gfx1151` matches the Framework Desktop target; without it the
# build silently fat-pack-for-everything and the resource census of `isa` is not the shipped kernel's.
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 -o attn_m2 attn_m2.hip.cpp
echo "built ./attn_m2"
