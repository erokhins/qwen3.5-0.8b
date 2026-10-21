#!/usr/bin/env bash
# build.sh — gdn_insitu at -O3, gfx1151.
#
# -O3 is not optional: this rig launches the shipping GDN recurrence kernels and the project rule is
# never to time an -O0 build (prototype/README.md). The rpath is required or the binary dies at load
# looking for libamdhip64.so.7.
#
# The rig #includes the same headers engine/kernels_gdn.cpp includes, by repo-root-relative path, so
# it is compiled with the repo root on the include path. There is no copy of any kernel here to drift.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
GPU=${GPU_TARGET:-gfx1151}
hipcc -O3 --offload-arch=$GPU -I"$root" -Wl,-rpath,/opt/rocm/lib \
      -o "$here/gdn_insitu" "$here/gdn_insitu.hip.cpp"
echo "built $here/gdn_insitu"
