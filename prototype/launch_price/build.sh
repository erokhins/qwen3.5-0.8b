#!/usr/bin/env bash
# build.sh — the S1 gate-zero rig at -O3, gfx1151.
#
# -O3 is not optional (AGENTS.md: never time an -O0 build), and the rpath is required or the binary
# dies at load looking for libamdhip64.so.7 (prototype/README.md).
#
# This rig owns its kernel and nothing else: it measures the machine's launch/segment cost, not any
# engine kernel, so there is no engine header to #include and therefore no second copy of anything to
# drift. That is a deliberate difference from prototype/mlp_insitu and prototype/gdn_insitu, whose whole
# point was to run the shipping kernels.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
GPU=${GPU_TARGET:-gfx1151}
hipcc -O3 --offload-arch=$GPU -Wl,-rpath,/opt/rocm/lib -o "$here/s1_launch_price" "$here/s1_launch_price.hip.cpp"
echo "built $here/s1_launch_price"
