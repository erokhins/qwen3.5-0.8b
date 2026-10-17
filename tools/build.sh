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
    # gemm_peak and prefill_mix need hipBLAS (the library whose kernels are being measured); everything
    # else links nothing beyond the HIP runtime. mfma_probe is rocwmma (header-only) and pins the arch on
    # the command line, because "what does this target emit" is the question it asks.
    libs=""
    extra=""
    src="$t.cpp"
    case "$t" in
        gemm_peak|prefill_mix) libs="-lhipblas" ;;
        # route_c_stages drives the engine's own wave body, so it links the engine TUs that
        # engine/build.sh links for decode_run (minus decode_run's main). The include paths are
        # repo-root-relative because this script runs from tools/.
        route_c_stages)
            libs="-lhipblas"
            extra="-I.. -I../engine"
            src="$t.cpp ../engine/decode_step.cpp ../engine/prefill.cpp ../engine/state.cpp \
                 ../engine/loader.cpp ../engine/launch_stats.cpp ../engine/stage_clock.cpp \
                 ../engine/kernels_glue.cpp ../engine/kernels_proj.cpp ../engine/kernels_mlp.cpp \
                 ../engine/kernels_gdn.cpp ../engine/kernels_attn.cpp ../engine/kernels_lmhead.cpp \
                 ../engine/kernels_cast.cpp ../engine/kernels_embed.cpp ../engine/kernels_gemm.cpp"
            ;;
    esac
    [ "$t" = "mfma_probe" ] && extra="--offload-arch=gfx1151"
    echo "hipcc -O3 -Wl,-rpath,/opt/rocm/lib $libs $extra -o build/$t $src"
    hipcc -O3 -Wno-unused-command-line-argument -Wl,-rpath,/opt/rocm/lib $libs $extra -o "build/$t" $src
done
