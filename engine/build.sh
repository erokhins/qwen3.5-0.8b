#!/usr/bin/env bash
# Build the engine binaries with hipcc.
#
#   ./engine/build.sh              # release build (-O3), outputs in engine/build/
#   ./engine/build.sh --debug      # -O0 -g, for line-numbered HIP errors; NEVER for timing
#   ./engine/build.sh --clean      # remove engine/build/
#
# Two binaries, because that is two questions:
#   loader_test   — does §01 hold? sizes, the dtype census (exactly 36 F32 tensors), the byte-level
#                   assertions, and the load-path timings that want the GPU to themselves.
#   decode_run    — does a token come out? drives `decode_step` over a list of token ids and emits
#                   one sampled id + one full logits vector per step for engine/gate.py to judge.
#                   Run it through gate.py, not by hand: the assertions live in the Python.
#
# Engine rules inherited from prototype/README.md: -O3 always, -Wl,-rpath,/opt/rocm/lib always
# (without it the binary dies with a linker error about libhsa-runtime4.so), and the include path is
# the REPO ROOT so that kernels include "engine/..." and the graduated kernels' own includes of
# "prototype/..." both resolve.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=engine/build
FLAGS=(-O3 -Wno-unused-command-line-argument -I. -Iengine -Wl,-rpath,/opt/rocm/lib)

case "${1:-}" in
  --clean) rm -rf "$BIN"; echo "removed $BIN"; exit 0 ;;
  --debug) FLAGS=(-O0 -g -Wno-unused-command-line-argument -I. -Iengine -Wl,-rpath,/opt/rocm/lib) ;;
  "") ;;
  *) echo "usage: engine/build.sh [--debug|--clean]" >&2; exit 2 ;;
esac

mkdir -p "$BIN"

# The §01 load path. loader.cpp is the only TU it needs.
hipcc "${FLAGS[@]}" engine/loader_test.cpp engine/loader.cpp -o "$BIN/loader_test"
echo "built $BIN/loader_test"

# The token loop. One TU per kernel family on purpose — see engine/kernels.h for why the families
# cannot share a translation unit. `prefill.cpp` is the prompt-phase wave (§06.1) and is a separate TU
# from `decode_step.cpp` because it is a different question: the step is one token's §07.1 order, the
# wave is what a request's whole prompt looks like when it is made of those steps.
# `launch_stats.cpp` is the observed-launch counter (`engine/launch_stats.h`): the two kernel TUs
# increment it, the driver prints it, and it has no business being anywhere else.
hipcc "${FLAGS[@]}" \
  engine/decode_run.cpp engine/decode_step.cpp engine/prefill.cpp engine/state.cpp engine/loader.cpp \
  engine/launch_stats.cpp \
  engine/kernels_glue.cpp engine/kernels_proj.cpp engine/kernels_mlp.cpp \
  engine/kernels_gdn.cpp engine/kernels_attn.cpp engine/kernels_lmhead.cpp \
  engine/kernels_cast.cpp \
  -o "$BIN/decode_run"
echo "built $BIN/decode_run"
