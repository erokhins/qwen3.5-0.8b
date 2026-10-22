#!/usr/bin/env bash
# build.sh — the P1 prefetch rig at -O3, gfx1151.
#
# AGENTS.md: never benchmark an -O0 build, and a HIP build needs -Wl,-rpath,/opt/rocm/lib or the binary
# dies at load looking for libamdhip64. The binary is named with its full path in .gitignore, never with
# a `prototype/**/<op>` pattern — that matches the DIRECTORY the sources live in, which is how two op
# directories once went untracked (AGENTS trap row).
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
GPU=${GPU_TARGET:-gfx1151}
for src in p1c_query.cpp p1_rig.hip.cpp; do
    out="${src%.cpp}"
    out="${out%.hip}"
    hipcc -O3 --offload-arch="$GPU" -Wl,-rpath,/opt/rocm/lib -o "$here/$out" "$here/$src"
    echo "built $here/$out"
done
