// The engine's own dtype adapters, and the only TU in engine/ that contains a kernel it did not
// lift from a prototype. `no model math here at all` is the rule for this file: a cast moves bytes
// and rounds, and anything more belongs in an op directory with a gate in front of it.
//
// Why they exist at all: the graduated kernels disagree about activation dtype. `glue`,
// `proj_gemv` and `gdn_gated_norm` write BF16 activations, while `proj_gemv` (PACK_X=false),
// `mlp_gemv` and `lm_head_gemv` read them as `const float*` — so every producer/consumer pair across
// that seam needs a widening. That is 97 launches/token in `decode_step`
// (the `cast` row of the table at the top of `decode_step.cpp`, which `planned_launch_counts()`
// derives and `gate.py` section (d) re-derives), which is the price of the seam, not of these
// kernels; `engine/results.md` keeps the deferral and the fix (a BF16-staged `mlp_gateup`/
// `mlp_down`, which `proj_gemv` proved is bit-identical for a dot) on the books.
//
// The seam used to be wider. `gdn_decode`'s conv and gates read F32 too, which cost one packed
// 3-region widening per GDN layer per body — 18 of the old 115 casts/token, and the largest single
// byte item in the family at wave width. D30 1.7' closed it by having both kernels widen their own
// BF16 operands in-register, and the packed form below went with it: a widening that no consumer
// reads is not a dtype adapter, it is a store to an empty buffer.
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

}  // namespace engine::kernels
