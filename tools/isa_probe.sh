#!/usr/bin/env bash
# isa_probe.sh — which arithmetic instructions does this part HAVE, and which does the shipped
# library actually USE. Pure compile/disassemble; touches no device, needs no GPU lock.
#
# Why this exists: the repo's MFMA route was priced against "the MFMA peak of this part is
# 14.85 TFLOPS (1:1 with FP32) and we realize 38 % of it" (prototype/compute_bw/results.md §10.1).
# That sentence has two errors this script proves or refutes: (1) there is no MFMA on RDNA3.5 at
# all — the matrix instruction here is WMMA, and it is not 1:1 with FP32 but 4× it; (2) our library
# GEMM number is not a matrix measurement, because the shipped gfx1151 bf16 kernels contain no
# matrix instruction to measure.
#
# Evidence captured on 2026-10-16 with ROCm 10.0.0-4 / LLVM 23.0.0git, recorded in
# devices/strix-halo-official-specs.md §6.
set -uo pipefail
MC="${MC:-/opt/rocm/lib/llvm/bin/llvm-mc}"
OBJDUMP="${OBJDUMP:-/opt/rocm/lib/llvm/bin/llvm-objdump}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
LIBDIR="${LIBDIR:-/opt/rocm/lib/rocblas/library/gfx1151}"
CPU="${CPU:-gfx1151}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

FAMS='v_wmma[a-z0-9_]*|v_mfma[a-z0-9_]*|v_dot[0-9][a-z0-9_]*|v_pk_fma[a-z0-9_]*|v_fma_mix_f32|v_fmac_f32_e32|v_fma_f32|v_pack_b32_f16'

test_asm() {  # name, mnemonic-with-operands
  local s="$TMP/a.s"; printf '\t.text\n%s\n' "$2" > "$s"
  local out; out=$("$MC" -triple=amdgcn -mcpu="$CPU" -assemble "$s" -o /dev/null 2>&1)
  if [ -z "$out" ]; then printf "  %-34s PRESENT\n" "$1"
  else printf "  %-34s %s\n" "$1" "$(echo "$out" | head -1 | sed -E 's/.*error: //; s/^/      /')"; fi
}

echo "### 1. ISA inventory for $CPU (llvm-mc accepts the mnemonic for this CPU or not)"
test_asm "v_mfma_f32_16x16x16f16 (CDNA)" "v_mfma_f32_16x16x16f16 v4, v0, v1, v2"
test_asm "v_mfma_f32_16x16x16bf16_1k"    "v_mfma_f32_16x16x16bf16_1k v4, v0, v1, v2"
test_asm "v_wmma_f32_16x16x16_f16"       "v_wmma_f32_16x16x16_f16 v[0:7], v[0:3], v[4:7], v[8:15]"
test_asm "v_dot2_f32_f16"                "v_dot2_f32_f16 v2, v0, v1, v2"
test_asm "v_dot2_f32_bf16"               "v_dot2_f32_bf16 v2, v0, v1, v2"
test_asm "v_dot4_i32_i8 (dp4a)"          "v_dot4_i32_i8 v2, v0, v1, v2"
test_asm "v_dot8_i32_i4"                 "v_dot8_i32_i4 v2, v0, v1, v2"
test_asm "v_dot4_f32_fp8_fp8 (fp8)"      "v_dot4_f32_fp8_fp8 v2, v0, v1, v2"
test_asm "v_dot4_f32_fp8_bf8"            "v_dot4_f32_fp8_bf8 v2, v0, v1, v2"
test_asm "v_pk_fma_f16 (packed fp16)"    "v_pk_fma_f16 v2, v0, v1, v2"
test_asm "v_pk_fma_bf16 (packed bf16)"   "v_pk_fma_bf16 v2, v0, v1, v2"
test_asm "v_fma_mix_f32 (bf16->f32)"     "v_fma_mix_f32 v2, v0, v1, v2"
test_asm "v_fmac_f32_e32 (scalar f32)"   "v_fmac_f32_e32 v2, v0, v1"

echo
echo "### 2. the matrix instruction is reachable from our compiler (correct RDNA3 spelling)"
cat > "$TMP/w.cu" <<'EOF'
#include <hip/hip_runtime.h>
typedef _Float16 half16_t  __attribute__((ext_vector_type(16)));
typedef float    float8_t  __attribute__((ext_vector_type(8)));
extern "C" __global__ void gemm_f16_wmma(const half16_t* W, const half16_t* X, float* Y) {
  float8_t acc = {0,0,0,0,0,0,0,0};
  half16_t a = W[threadIdx.x], b = X[threadIdx.x];
  // gfx11 spelling: _f16_w32. The CDNA-style names (__builtin_amdgcn_wmma_*_16f16_w32) do not
  // exist here; an earlier version of this probe used one and wrongly concluded WMMA was absent.
  acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
  float s = 0; for (int i = 0; i < 8; i++) s += acc[i];
  Y[threadIdx.x] = s;
}
EOF
if out=$(timeout 300 "$HIPCC" --offload-arch="$CPU" --save-temps -c "$TMP/w.cu" -o "$TMP/w.o" 2>&1) && [ -z "$out" ]; then
  dev=$(ls "$(dirname "$(find "$TMP" -name "*${CPU}.o" | head -1)")"/*"${CPU}.o" 2>/dev/null | head -1)
  echo "  rocwmma-style WMMA kernel COMPILES for $CPU; device object ${dev:+$dev}"
  echo "  instructions emitted: $("$OBJDUMP" -d "$dev" 2>/dev/null | grep -oE "$FAMS" | sort | uniq -c | tr -s ' \n' ' ')"
else
  echo "  WMMA kernel FAILED: $(echo "$out" | grep -m1 'error:')"
fi

echo
echo "### 3. what the SHIPPED rocBLAS kernels for $CPU actually execute (per data type)"
printf "  %-10s %-52s %s\n" TYPE "library file" "arithmetic instructions found"
for t in BB BS CC DD HH HS SS ZZ 4xi8I I8I8I; do
  for f in "$LIBDIR"/TensileLibrary_Type_${t}_*.hsaco; do
    [ -e "$f" ] || continue
    fams=$(timeout 200 "$OBJDUMP" -d "$f" 2>/dev/null | grep -oE "$FAMS" | sort | uniq -c | sort -rn | head -3 | tr -s ' ' | tr '\n' ';' | sed 's/;$//')
    printf "  %-10s %-52s %s\n" "$t" "$(basename "$f" | sed 's/TensileLibrary_Type_//; s/_Contraction_l_/ /; s/_Cijk_Dijk//; s/_fallback_gfx1151.hsaco//')" "${fams:-none matched}"
  done
done | head -20
echo "  (bb = bf16 in/bf16 out, hh = fp16, ss = fp32, 4xi8I = int8; HPA = high-precision accumulate)"
echo
echo "  matrix-instruction census across every $CPU code object shipped with rocBLAS:"
n_with=0; n_tot=0
for f in "$LIBDIR"/*.hsaco; do
  [ -e "$f" ] || continue
  n_tot=$((n_tot+1))
  c=$(timeout 200 "$OBJDUMP" -d "$f" 2>/dev/null | grep -cE "v_wmma|v_mfma")
  [ "${c:-0}" -gt 0 ] && { n_with=$((n_with+1)); echo "    $f: $c"; }
done
echo "    $n_with of $n_tot code objects contain a matrix instruction."
echo
echo "  NOTE: the per-kernel .co files Tensile writes are CCO containers (magic 'CCO\003'), which
  llvm-objdump cannot read; scan the .hsaco/.dat libraries above instead, or use roc-obj."
