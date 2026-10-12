// The engine's own dtype adapters, and the only TU in engine/ that contains a kernel it did not
// lift from a prototype. `no model math here at all` is the rule for this file: a cast moves bytes
// and rounds, and anything more belongs in an op directory with a gate in front of it.
//
// Why they exist at all: the graduated kernels disagree about activation dtype. `glue`,
// `proj_gemv` and `gdn_gated_norm` write BF16 activations, while `proj_gemv` (PACK_X=false),
// `mlp_gemv` and `gdn_decode`'s conv/gates/rec read them as `const float*` — so every producer/
// consumer pair across that seam needs a widening. That is 115 launches/token in `decode_step`
// (the `cast` row of the table at the top of `decode_step.cpp`, which `planned_launch_counts()`
// derives and `gate.py` section (d) re-derives), which is the price of the seam, not of these
// kernels; `engine/results.md` keeps the deferral and the fix (a BF16-staged `mlp_gateup`/
// `mlp_down`, which `proj_gemv` proved is bit-identical for a dot) on the books.
//
// The rounding is not free to get wrong: `hip_bfloat16(float)` is round-to-nearest-even, which is
// what torch's cast does and what every prototype's `fb()`/`bf()` helper uses (`hip_bfloat16`'s
// constructor is the RNE conversion; `__float2bfloat16_rn` is a CUDA-ism and does not exist here —
// prototype/README.md). A truncating cast here would drift half of every widened tensor by 1 ULP.
//
// Its own TU for the same reason as every other kernels_*.cpp: nothing here needs a macro, and
// keeping it macro-free means it can never collide with an op's `D`/`H`/`HID`/`FULL_MASK`.
#include "kernels.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>

namespace {
constexpr int kThreads = 256;

__global__ void cast_bf16_to_f32_kernel(const hip_bfloat16* __restrict__ src,
                                        float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = float(src[i]);
}

__global__ void cast_f32_to_bf16_kernel(const float* __restrict__ src,
                                        hip_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = hip_bfloat16(src[i]);
}

// Three regions in ONE launch. A GDN layer needs the same widening for in_proj_qkv's 6144 raw
// columns and for two 16-element gate rows; three launches would each pay the ~1.85 µs dispatch
// floor to move as little as 32 B. Regions are addressed by a linear index over n0+n1+n2, so a
// zero-length region costs nothing and the caller does not need a second entry point.
__global__ void cast_bf16_to_f32_3_kernel(const hip_bfloat16* __restrict__ s0,
                                          float* __restrict__ d0, int n0,
                                          const hip_bfloat16* __restrict__ s1,
                                          float* __restrict__ d1, int n1,
                                          const hip_bfloat16* __restrict__ s2,
                                          float* __restrict__ d2, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n0) {
        d0[i] = float(s0[i]);
    } else if ((i -= n0) < n1) {
        d1[i] = float(s1[i]);
    } else if ((i -= n1) < n2) {
        d2[i] = float(s2[i]);
    }
}

int blocks_for(int n) { return (n + kThreads - 1) / kThreads; }
}  // namespace

namespace engine::kernels {

void cast_bf16_to_f32(const uint16_t* src, float* dst, int n) {
    if (n <= 0) return;   // a zero-length cast is a caller bug worth noticing, but never a launch
    cast_bf16_to_f32_kernel<<<blocks_for(n), kThreads>>>((const hip_bfloat16*)src, dst, n);
}

void cast_f32_to_bf16(const float* src, uint16_t* dst, int n) {
    if (n <= 0) return;
    cast_f32_to_bf16_kernel<<<blocks_for(n), kThreads>>>(src, (hip_bfloat16*)dst, n);
}

void cast_bf16_to_f32_3(const uint16_t* s0, float* d0, int n0, const uint16_t* s1, float* d1,
                        int n1, const uint16_t* s2, float* d2, int n2) {
    const long total = (long)n0 + n1 + n2;
    if (total <= 0) return;
    // A null pointer with a non-zero count would be a wild write inside a region the caller
    // promised exists; refuse it here rather than inside the kernel.
    if ((n0 > 0 && (s0 == nullptr || d0 == nullptr)) ||
        (n1 > 0 && (s1 == nullptr || d1 == nullptr)) ||
        (n2 > 0 && (s2 == nullptr || d2 == nullptr))) {
        throw std::runtime_error("cast_bf16_to_f32_3: a non-empty region has a null pointer");
    }
    cast_bf16_to_f32_3_kernel<<<blocks_for((int)total), kThreads>>>(
        (const hip_bfloat16*)s0, d0, n0, (const hip_bfloat16*)s1, d1, n1,
        (const hip_bfloat16*)s2, d2, n2);
}

}  // namespace engine::kernels
