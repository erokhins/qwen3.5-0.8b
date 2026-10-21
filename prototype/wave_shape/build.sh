#!/usr/bin/env bash
# build.sh — the wave-shape rig at -O3, gfx1151 (AGENTS: never time an -O0 build; the rpath is required
# or the binary dies at load looking for libamdhip64).
#
# This rig owns its kernels: `down_ship` is a transcription of the shipped one-token down_proj, not an
# #include of it, because prototype/README.md forbids editing prototype/mlp_gemv for the engine's
# convenience and this rig must be able to run the frozen spelling beside its own variants. The
# transcription is what --selftest's pair_intra1_eq_ship case exists to witness: if the transcription
# drifted, the degenerate split stops reproducing the frozen form bit for bit.
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
GPU=${GPU_TARGET:-gfx1151}
hipcc -O3 --offload-arch=$GPU -Wl,-rpath,/opt/rocm/lib -o "$here/wave_shape" "$here/wave_shape.hip.cpp"
echo "built $here/wave_shape"
