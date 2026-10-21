#!/usr/bin/env bash
# build.sh — attn_decode_wide at -O3, gfx1151.
#
# Same flags as `prototype/attn_decode/build.sh`, including the rpath: without it the binary finds a
# different libhipcxx and crashes 40 layers later than this (prototype/README.md).
#
# `--save-temps=obj` is left ON, deliberately: the residency claim in the header (6,176 B of LDS, not
# 38,944 B) is read out of the generated `.s` by `census.sh`, and a build that throws its temporaries
# away makes the only witness for the whole route disappear.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
GPU=${GPU_TARGET:-gfx1151}
hipcc -O3 --offload-arch=$GPU -Wl,-rpath,/opt/rocm/lib --save-temps=obj \
    -o "$here/attn_decode_wide" "$here/attn_decode_wide.hip.cpp"
echo "built $here/attn_decode_wide"
