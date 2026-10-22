#!/usr/bin/env bash
# Build the MTP draft-block rig.
#
# Six TUs, on purpose. The shipped op headers each define colliding macros and same-named helpers (their
# own header comment says merging them is a compile error), so one TU per family is both the engine's own
# build shape and the only thing that compiles. `td_modes.cpp` is the single home of
# `mtp_draft_kernels.h`, whose __global__ definitions must not be emitted twice.
#
# -O3 is not optional (project rule: never benchmark an -O0 build) and -Wl,-rpath,/opt/rocm/lib is
# required for HIP builds on this box (AGENTS trap table). The include path is the repo root so the rig
# can say #include "prototype/glue/glue_kernels.h" / "engine/mlp_xbf16_kernels.h" the way the engine does.
set -euo pipefail
cd "$(dirname "$0")/../.."
export LC_ALL=C
OUT=prototype/mtp_draft/mtp_draft
FLAGS=(-O3 -Wno-unused-command-line-argument -I. -Iengine -Iprototype/mtp_draft -Wl,-rpath,/opt/rocm/lib)
TUS=(prototype/mtp_draft/mtp_draft.hip.cpp prototype/mtp_draft/td_modes.cpp
     prototype/mtp_draft/td_glue_tu.cpp prototype/mtp_draft/td_proj_tu.cpp
     prototype/mtp_draft/td_mlp_tu.cpp prototype/mtp_draft/td_attn_tu.cpp)
OBJS=()
for tu in "${TUS[@]}"; do
    o="$(mktemp -d)/$(basename "$tu").o"
    hipcc "${FLAGS[@]}" -c "$tu" -o "$o"
    OBJS+=("$o")
done
hipcc "${FLAGS[@]}" -o "$OUT" "${OBJS[@]}"
rm -f "${OBJS[@]}"
echo "built $OUT"
