#!/usr/bin/env bash
# Build the slim-vs-shipped A/B harness. Same flags as prototype/attn_decode/build.sh: -O3 is a
# standing rule (never benchmark or gate an -O0 build) and the rpath is what makes the binary run
# against /opt/rocm/lib without an LD_LIBRARY_PATH dance (prototype/README.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 -o "$here/attn_valu_slim" "$here/attn_valu_slim.hip.cpp"
echo "built $here/attn_valu_slim"
