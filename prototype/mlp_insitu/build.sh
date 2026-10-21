#!/usr/bin/env bash
# build.sh — mlp_insitu at -O3, gfx1151.
#
# -O3 is not optional: this rig times the shipping MLP kernel, and the project rule is never to
# benchmark an -O0 build. The rpath is required or the binary dies at load looking for
# libamdhip64.so.7 (prototype/README.md).
#
# The rig `#include`s ../mlp_gemv/mlp_gemv_kernels.h — the SAME header `engine/kernels_mlp.cpp`
# includes — so there is no copy of the kernel here to drift. That is also why this TU must stay
# alone in its translation unit: the op headers define colliding macros (HID, FF, FULL_MASK, NEG_*),
# which is fine across TUs and a compile error if merged (`engine/kernels_mlp.cpp` says the same).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
GPU=${GPU_TARGET:-gfx1151}
hipcc -O3 --offload-arch=$GPU -Wl,-rpath,/opt/rocm/lib -o "$here/mlp_insitu" "$here/mlp_insitu.hip.cpp"
echo "built $here/mlp_insitu"
