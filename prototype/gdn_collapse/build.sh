#!/usr/bin/env bash
# Build the route A device gate. -O3 always: timing an unoptimized build is banned (AGENTS.md), and
# this one is not a timing gate at all, but the kernel under test must be the -O3 kernel the engine
# builds with FMA contraction included — a -O0 collapse kernel is a different set of bits.
#
#   ./build.sh                                  # the shipping form
#   EXTRA_FLAGS=-DNEG_RC_STATE_BF16 ./build.sh  # one injected defect, its own binary (run.py)
#
# The include path is the repo root, same as engine/build.sh, because the kernel lives in engine/ and
# includes its op header out of prototype/.
set -euo pipefail
cd "$(dirname "$0")/../.."

FLAGS=(-O3 -Wno-unused-command-line-argument -I. -Iengine -Wl,-rpath,/opt/rocm/lib)
if [[ -n ${EXTRA_FLAGS:-} ]]; then FLAGS+=($EXTRA_FLAGS); fi

mkdir -p prototype/gdn_collapse
hipcc "${FLAGS[@]}" prototype/gdn_collapse/gdn_collapse_check.hip.cpp \
  -o prototype/gdn_collapse/gdn_collapse_check
echo "built prototype/gdn_collapse/gdn_collapse_check"
