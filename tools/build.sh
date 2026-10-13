#!/usr/bin/env bash
# Build the HIP probes in tools/ into tools/build/ (gitignored).
#
# Two flags are not negotiable on this box (AGENTS.md, prototype/README.md):
#   -O3                  never benchmark an -O0/-O1 build; the probes time the memory system, and a
#                        non-optimized host loop changes what the GPU sees.
#   -Wl,-rpath,/opt/rocm/lib   without it the binary dies at load with a missing-libhipamd error.
#
#   tools/build.sh                  # builds mixed_bw
#   tools/build.sh mixed_bw bw_probe_hip     # or name targets explicitly
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=(mixed_bw)
for t in "${targets[@]}"; do
    echo "hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o build/$t $t.cpp"
    hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o "build/$t" "$t.cpp"
done
