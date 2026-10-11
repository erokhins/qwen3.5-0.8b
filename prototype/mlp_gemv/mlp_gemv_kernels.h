/*
 * prototype/mlp_gemv/mlp_gemv_kernels.h — the device code of mlp_gemv: S03.3 MLP decode path: gate+up+SwiGLU fused, then down.
 *
 * Moved verbatim out of mlp_gemv.hip.cpp (host main() and the gate's file I/O stayed
 * there) so that mlp_gemv and `engine/decode_step` compile THE SAME KERNEL SOURCE. An
 * engine TU includes exactly one of these op headers: the ops each spell their own
 * helpers (bf/fb/rnd/block_sum/reduce_row) and define colliding macros (D, H, HID,
 * EPS, SCALE, FULL_MASK), which is fine while they are in separate translation
 * units and is a compile error if they are merged into one.
 *
 * Gate for this file: prototype/mlp_gemv/ (its run.py/check.py drives the binary in
 * the same directory, which includes this header). Nothing here may be edited for
 * the engine's convenience — see the rule in prototype/README.md.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstring>

#define HID 1024      // hidden size (§00)
#define FF 3584       // intermediate size (§00)
#define ROWS_PER_BLOCK 8
#define FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit

// Injected bugs, so run.py --negative can prove the gate rejects them (the G-neg convention
// in speccheck/gates/run_negative.py: a gate that cannot fail verifies nothing).
#define NEG_NONE 0
#define NEG_HALF 1     // accumulate half of K: a wrong reduction length
#define NEG_TRUNC 2    // round toward zero instead of round-to-nearest-even
#define NEG_F32ACT 3   // do the SwiGLU entirely in F32 (the literal A-07 reading)
#define NEG_SWAPROW 4  // write each finished row into its neighbour's slot (wrong output index)
#define NEG_TAIL 5     // drop the LAST word of K: an off-by-one in the reduction bound
#define NEG_PRESWAP 6  // stage gate into the up slot and vice versa in the --pre dump, leaving
                       // `act` correct — the bug class the F1/F2 split exists to detect

// Round-to-nearest-even, same as torch's cast to bf16. The product of two bf16 values must
// round the same way the reference's bf16 multiply does, or `act` drifts by 1 ULP.
// HIP has no `__float2bfloat16_rn` (CUDA-ism, see prototype/README.md): the bf16
// constructor is the RNE conversion, and the gate in run.py would notice truncation
// immediately (it would move ~half of `act` by 1 ULP instead of ~0).
__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }

// Truncating cast, for NEG_TRUNC only: dropping the low 16 bits of the fp32 payload rounds
// toward zero, which is the classic way to get a bf16 conversion wrong.
__device__ __forceinline__ hip_bfloat16 fb_trunc(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    unsigned short top = (unsigned short)(u >> 16);
    hip_bfloat16 b;
    memcpy(&b, &top, 2);
    return b;
}

__device__ __forceinline__ hip_bfloat16 rnd(float f, int neg) {
    return neg == NEG_TRUNC ? fb_trunc(f) : fb(f);
}

// Full-wavefront reduction helper: the shift ladder starts at warpSize/2, so the same
// code is correct on a wave32 and a wave64 target (the host checks which one this is).
__device__ __forceinline__ float reduce_row(float acc) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    return acc;
}

// One wavefront per intermediate row: both projections and the SwiGLU epilogue, so the
// gate/up pair costs one launch and the 3584-element intermediates never reach memory
// unless asked for (--pre, which exists so the gate can compare them).
template <int IN>
__global__ void mlp_gateup(const hip_bfloat16* __restrict__ Wg,
                           const hip_bfloat16* __restrict__ Wu, const float* __restrict__ x,
                           hip_bfloat16* __restrict__ act, hip_bfloat16* __restrict__ pre,
                           int rows, int rpb, int neg) {
    __shared__ float xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    // Two independent streams: the compiler interleaves the loads, which is what keeps
    // enough of them in flight to sit on the roofline at only 8 rows per block.
    const uint4* wg = reinterpret_cast<const uint4*>(Wg + (size_t)row * IN);
    const uint4* wu = reinterpret_cast<const uint4*>(Wu + (size_t)row * IN);
    // NEG_HALF: half of K. NEG_TAIL: the last word of K (8 of 1024 terms), which is the shape an
    // off-by-one in `v < words` actually takes — small enough that only a gate with a real
    // per-element rule notices, unlike NEG_HALF.
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
    float ag = 0.f, au = 0.f;
#pragma unroll
    for (int v = lane; v < words; v += ws) {
        uint4 ug = wg[v], uu = wu[v];
        const hip_bfloat16* pg = reinterpret_cast<const hip_bfloat16*>(&ug);
        const hip_bfloat16* pu = reinterpret_cast<const hip_bfloat16*>(&uu);
        const float* xp = &xs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            ag += bf(pg[j]) * xp[j];
            au += bf(pu[j]) * xp[j];
        }
    }
    ag = reduce_row(ag);
    au = reduce_row(au);
    if (lane != 0) return;
    hip_bfloat16 g = rnd(ag, neg), u = rnd(au, neg);
    // NEG_SWAPROW: store the finished row one slot over. rows is a multiple of 3584/1024, so the
    // neighbour always exists; what matters is that every other row is now somebody else's answer.
    const int orow = neg == NEG_SWAPROW ? (row ^ 1) : row;
    // silu in F32 on the rounded gate, rounded back to BF16, THEN multiplied with the rounded
    // `up` and rounded again — three roundings, because HF's F.silu returns BF16 and the
    // product is a BF16 tensor op. Doing the whole thing in F32 (NEG_F32ACT, and the literal
    // reading of A-07) differs from the reference in 27 % of `act` elements; ulp_probe.py
    // measures that and that this three-rounding form is bit-equal to the pin. HF's order is
    // x * (1/(1+exp(-x))); writing it as a division is algebraically identical but cost the
    // first version a 2 ULP outlier at a large element (a tie in the product's BF16 rounding).
    if (neg == NEG_F32ACT) {              // the wrong reading, for the negative test only
        float a = (ag * (1.0f / (1.0f + expf(-ag)))) * au;
        act[orow] = rnd(a, neg);
    } else {
        float gf = bf(g);
        float sg = bf(fb(gf * (1.0f / (1.0f + expf(-gf)))));
        act[orow] = rnd(sg * bf(u), neg);
    }
    if (pre) {
        // NEG_PRESWAP: the two dots and the epilogue are all correct — only the staging of the
        // raw dumps swapped, so `act` matches the reference and `gate_pre` is `up_pre`'s value.
        // A gate that judged `act` alone would sign this off; the F1/F2 split is what catches it.
        const bool sw = neg == NEG_PRESWAP;
        pre[orow] = sw ? u : g;
        pre[rows + orow] = sw ? g : u;
    }
}

template <int IN>
__global__ void mlp_down(const hip_bfloat16* __restrict__ W, const float* __restrict__ x,
                         hip_bfloat16* __restrict__ out, int rows, int rpb, int neg) {
    __shared__ float xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
    float acc = 0.f;
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
#pragma unroll
    for (int v = lane; v < words; v += ws) {
        uint4 u = wrow[v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        const float* xp = &xs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += bf(p[j]) * xp[j];
    }
    acc = reduce_row(acc);
    if (lane == 0) out[neg == NEG_SWAPROW ? (row ^ 1) : row] = rnd(acc, neg);
}
