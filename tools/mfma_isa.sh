#!/usr/bin/env bash
# mfma_isa.sh — did the compiler actually give this target matrix instructions?
#
# Compile-only. No device is touched, so this can run any time, next to anything, and it takes no
# timing. It answers a question a runtime PASS cannot: a `rocwmma::mma_sync` call can compile and run
# and still be *lowered to scalar FMA*, in which case the "matrix unit route works" evidence is worth
# nothing. So the probe prints both halves: `tools/mfma_probe.cpp` runs on the device and says whether
# the answer is right; this says what the hardware was asked to do.
#
#   tools/mfma_isa.sh [tools/mfma_probe.cpp]
#
# Method: `--save-temps=<dir>` makes the HIP driver leave the device assembly behind (the file named
# `*-hip-amdgcn-amd-amdhsa-<arch>.s`), and this slices that file per kernel and counts instruction
# families. `fma_gemm` is in the same translation unit precisely so that the table shows what a kernel
# that must be scalar FMA looks like next to the one that should not be.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-$here/mfma_probe.cpp}
# Absolute before the cd below: the compiler runs in a scratch dir and a relative path would not
# survive it.
case "$src" in /*) ;; *) src=$(pwd)/$src ;; esac
arch=${MFMA_ISA_ARCH:-gfx1151}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "compiling $src --offload-arch=$arch (compile only; no device touched)"
# The HIP driver writes `--save-temps=<dir>` files into the CWD anyway (ROCm 7.15 ignores the dir
# argument), so run it inside the temp dir instead of arguing with it. The stray
# `mlp_gemv.hip-hip-amdgcn-amd-amdhsa-gfx1151.o` sitting in this repo's root is what happens when
# someone argues with it.
cd "$tmp" || exit 1
if ! hipcc -O3 --offload-arch="$arch" -c "$src" -o out.o -save-temps 2>err; then
    echo "COMPILE FAILED — that is itself the answer. Compiler said:"
    sed 's/^/  /' err
    exit 1
fi
sfile=$(find "$tmp" -name "*-${arch}.s" | head -1)
if [ -z "$sfile" ]; then
    echo "no device assembly produced for $arch; files were:"
    find "$tmp" -maxdepth 1 -printf '  %f\n'
    exit 1
fi

echo "target line: $(grep -m1 amdgcn_target "$sfile" | tr -d '\t')"
echo "notes target: $(grep -m1 'amdhsa.target' "$sfile" | tr -d '\t')"
echo
printf '%-40s %7s %6s %6s %6s %8s   %s\n' \
       kernel instrs v_wmma v_mfma v_fma loads verdict
awk '
    function demangle(n) {                       # only enough to read the table
        if (n ~ /9wmma_gemm/) {
            if (n ~ /12hip_bfloat16fEE/) return "wmma_gemm<bf16, f32 acc>"
            if (n ~ /DF16_DF16_EE/)       return "wmma_gemm<fp16, fp16 acc>"
            if (n ~ /DF16_fEE/)           return "wmma_gemm<fp16, f32 acc>"
            return "wmma_gemm<?>"
        }
        if (n ~ /8fma_gemm/)  return "fma_gemm<fp16, f32> (the control)"
        gsub(/^_ZN12_GLOBAL__N_1/, "", n); return substr(n, 1, 40)
    }
    /-- Begin function/ { name = $NF; next }
    name != "" {
        tot[name]++
        if ($1 ~ /^v_wmma/) { wmm[name]++; if (mn_set[name] == "") mn_set[name] = $1; else if (index(mn_set[name], $1) == 0) mn_set[name] = mn_set[name] "," $1 }
        else if ($1 ~ /^v_mfma/) { mfma[name]++; if (mn_set[name] == "") mn_set[name] = $1; else if (index(mn_set[name], $1) == 0) mn_set[name] = mn_set[name] "," $1 }
        else if ($1 ~ /^v_fma/) fma[name]++
        else if ($1 ~ /^(global|flat)_load/) ld[name]++
        if ($1 == ".size") {
            v = (wmm[name] || mfma[name]) ? "MATRIX UNIT (" mn_set[name] ")" \
                : (fma[name] ? "scalar/vector FMA only" : "no arithmetic counted")
            printf "%-40s %7d %6d %6d %6d %8d   %s\n", demangle(name), tot[name], wmm[name], \
                   mfma[name], fma[name], ld[name], v
            name = ""
        }
    }
' "$sfile"
echo
echo "Read: a kernel whose verdict is MATRIX UNIT emitted the instruction named; one that says"
echo "scalar/vector FMA only had its mma_sync lowered to ordinary arithmetic, which would make a"
echo "runtime PASS worthless as matrix-unit evidence."
