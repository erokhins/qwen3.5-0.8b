#!/usr/bin/env bash
# Build gemm_algo_sweep.
#
#   bash prototype/gemm_algo_sweep/build.sh
#
# Two flags are not negotiable on this box (AGENTS.md, prototype/README.md):
#   -O3                        never benchmark a non-optimized build; this tool exists to time.
#   -Wl,-rpath,/opt/rocm/lib   without it the binary dies at load on a missing libhipamd/hipblaslt.
#
# Both libraries are linked, and each is there for a named reason:
#   -lhipblaslt   the enumeration and the per-algorithm launch path (hipblasLtMatmulAlgoGetHeuristic,
#                 hipblaslt_ext::getAllAlgos, hipblasLtMatmul).
#   -lhipblas     the engine's OWN call, `gemm_common::gemm_stored_weight` -> hipblasGemmEx, timed
#                 alongside so a hipBLASLt ratio can be attributed to the engine rather than to
#                 hipBLASLt's own default. Same reason tools/build.sh links -lhipblas for gemm_peak
#                 and prefill_mix.
#
# -I<repo root> because the tool includes tools/gemm_common.h by its repo-relative name, which is how
# engine/kernels_gemm.cpp includes it too: one call form, one definition (AGENTS.md, no duplicates).
#
# Nothing here runs a kernel. The binary only touches the device when invoked, and README.md says
# which modes allocate what.
set -euo pipefail
cd "$(dirname "$0")"
: "${HIPCC:=hipcc}"
ROOT="$(cd ../.. && pwd)"

${HIPCC} -O3 -Wno-unused-command-line-argument -Wl,-rpath,/opt/rocm/lib \
    -I"${ROOT}" \
    -o gemm_algo_sweep gemm_algo_sweep.cpp \
    -lhipblaslt -lhipblas

echo "BUILD ok: prototype/gemm_algo_sweep/gemm_algo_sweep (links libhipblaslt + libhipblas; does NOT self-lock)"
