// a10_pvf16_map_probe.cpp — measure the D-block fragment map of `v_wmma_f16_16x16x16_f16` (w32) in the
// SHAPE THE ENGINE WOULD EMIT: two issues per 32-dim output tile, op_sel 0 then 1, two different A
// fragments and the same P operand (`engines/llama.cpp/ggml/src/ggml-cuda/mma.cuh:1556-1566`).
//
// WHY THIS EXISTS AND NOT A COMMENT
// `engine/attn_fattn_kernels.h` drains its F32 accumulator with a map that was MEASURED on this device by
// `prototype/attn_wmma/fattn_map_probe.cpp` + `check_fattn_map.py`, not transcribed — that probe exists
// because a transcription of `mma.cuh` was wrong in exactly this way (it produced a TRANSPOSED score and
// folded every row down the wrong axis). An F16 accumulator is a DIFFERENT fragment layout: 16 halves per
// lane instead of 8 F32, and the builtin carries a fourth `op_sel` argument with no F32 counterpart. Nothing
// in this tree has ever emitted it against known operands, so the epilogue's `(lane, element) -> (row, dim)`
// law is UNMEASURED, and the epilogue is the site where a half-published row once produced NaN for every
// covered row (`bench/performance-status.md` §29). This file measures it instead of assuming it.
//
// THE PATTERN (same design rule as `fattn_map_probe.cpp`: every cell holds a number that ENCODES ITS OWN
// COORDINATES, so any relabelling of (row, dim) shows up as a wrong integer rather than as a plausible sum)
//
//   P[q][0] = 1,  P[q][1] = 32q            (row = query row q, run along the KV lane)
//   V[d][0] = d,  V[d][1] = 1              (row = output dim d, run along the KV lane; d < 32)
//   everything else 0
//     => O[q][d] = 1*d + 32q*1 = d + 32q, unique per (q, d), every value an exact integer below 2^11, so
//     a wrong number is never a rounding artefact and never a re-labelled-but-plausible one. q is the high
//     part, d the low part: q = v/32, d = v%32.
//
// The two A fragments are the two 16-dim halves of the 32-dim output tile, exactly as the kernel's
// `fattn_v_addr(d, kv) = d*FATTN_MLANES + kv` stages them: lane r = lane%16 reads the 16 contiguous halves
// of row `16t + r`, for t = 2τ (op_sel 0 side) and t = 2τ+1 (op_sel 1 side).
//
// WHAT THE CHECKER GETS FROM THIS
//   * which lane/element holds which (q, d) — the drain's address law;
//   * which half of the D block (op_sel 0 or op_sel 1) carries dims 0..15 vs 16..31 — the P*V leg's operand
//     pairing, i.e. which A fragment goes with which `opsel`;
//   * whether the two issues are independent (`--opsel high-first` swaps them: if the two halves are truly
//     independent tiles the map mirrors; if they are not, it does not, and that is also worth knowing).
//
// BUILD: bash prototype/attn_percost/build_a10_pvf16_map.sh      (linked binary; it launches once)
// RUN:   python3 prototype/attn_percost/check_a10_pvf16_map.py    (PASS/FAIL on the derived map; no timing)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#define DIMS 32          // output dims in the probe's transposed-V stage (two 16-dim halves)
#define KV 16            // KV lanes in one round: the WMMA K dimension
#define LANES 32
#define ELEMS 16         // halves per lane in one F16 D block (8 VGPRs)

namespace {

using f16x16 = _Float16 __attribute__((ext_vector_type(16)));   // one A/B operand, one F16 D block

__device__ __forceinline__ f16x16 load_tile(const __half* p) {
    union { uint4 u[2]; f16x16 v; } x;
    x.u[0] = *reinterpret_cast<const uint4*>(p);
    x.u[1] = *reinterpret_cast<const uint4*>(p + 8);
    return x.v;
}

// One 32-dim output tile, two issues, llama's shape. `low_first` picks which op_sel gets the LOW A fragment,
// so the checker can tell "the two halves are independent tiles" from "they are one thing in a fixed order".
extern "C" __global__ void a10_pvf16_map_probe(const __half* __restrict__ vt,   // [d][kv], d < DIMS
                                               const __half* __restrict__ pm,   // [q][kv], q < 16
                                               unsigned short* __restrict__ out,  // [32][16] raw bits
                                               int low_first) {
    const int lane = threadIdx.x & (LANES - 1);
    const int r = lane & 15;                       // `fattn_ab_row(lane)`
    const f16x16 a_lo = load_tile(vt + (size_t)r * KV);          // dims  0..15
    const f16x16 a_hi = load_tile(vt + (size_t)(16 + r) * KV);    // dims 16..31
    const f16x16 b = load_tile(pm + (size_t)r * KV);              // P: row = query row
    f16x16 c{};                                                    // brace-init: C is a live IN operand
    if (low_first) {
        c = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_lo, b, c, 0);
        c = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_hi, b, c, 1);
    } else {
        c = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_hi, b, c, 0);
        c = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_lo, b, c, 1);
    }
    union { f16x16 v; unsigned short s[ELEMS]; } d;
    d.v = c;
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) out[lane * ELEMS + e] = d.s[e];
}

// Exact F16 bit pattern for a small non-negative integer, written longhand on the host (the same trick
// `fattn_map_probe.cpp` uses: every value here is an integer below 2048, so it is exact by construction and
// the host needs no `__half` conversion intrinsic whose availability varies).
unsigned short half_bits(int v) {
    if (v == 0) return 0;
    int msb = 0;
    while ((1 << (msb + 1)) <= v) ++msb;
    const int guard = 10 - msb;
    const unsigned int mant = (unsigned int)((v - (1 << msb)) << guard);
    return (unsigned short)(((msb + 15) << 10) | (int)mant);
}

#define CK(x)                                                                             \
    do {                                                                                  \
        const hipError_t e_ = (x);                                                        \
        if (e_ != hipSuccess) {                                                           \
            std::printf("PROBE-ERROR %s at %s:%d\n", hipGetErrorString(e_), __FILE__,     \
                        __LINE__);                                                        \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    int low_first = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--opsel-high-first")) low_first = 0;
        else { std::printf("PROBE-ERROR unknown argument %s\n", argv[i]); return 2; }
    }

    static unsigned short host[DIMS * KV];
    // V[d][0] = d, V[d][1] = 1  (row = output dim, run along the KV lane: the transposed-V stage)
    for (int i = 0; i < DIMS * KV; ++i) host[i] = half_bits(i % KV == 0 ? i / KV : (i % KV == 1 ? 1 : 0));
    __half* vt = nullptr;
    CK(hipMalloc(&vt, sizeof(host)));
    CK(hipMemcpy(vt, host, sizeof(host), hipMemcpyHostToDevice));

    // P[q][0] = 1, P[q][1] = 32q
    for (int i = 0; i < DIMS * KV; ++i) {
        const int q = i / KV, k = i % KV;
        host[i] = half_bits(k == 0 ? 1 : (k == 1 ? 32 * q : 0));
    }
    __half* pm = nullptr;
    CK(hipMalloc(&pm, sizeof(host)));
    CK(hipMemcpy(pm, host, sizeof(host), hipMemcpyHostToDevice));

    unsigned short* out = nullptr;
    CK(hipMalloc(&out, (size_t)LANES * ELEMS * sizeof(unsigned short)));
    CK(hipMemset(out, 0, (size_t)LANES * ELEMS * sizeof(unsigned short)));

    a10_pvf16_map_probe<<<1, LANES>>>(vt, pm, out, low_first);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    static unsigned short hout[LANES * ELEMS];
    CK(hipMemcpy(hout, out, sizeof(hout), hipMemcpyDeviceToHost));

    std::printf("PVF16MAP low_first=%d lanes=%d elems=%d\n", low_first, LANES, ELEMS);
    for (int t = 0; t < LANES; ++t) {
        std::printf("LANE %d", t);
        for (int e = 0; e < ELEMS; ++e) std::printf(" %u", hout[t * ELEMS + e]);
        std::printf("\n");
    }
    std::printf("PVF16MAP DONE\n");
    return 0;
}
