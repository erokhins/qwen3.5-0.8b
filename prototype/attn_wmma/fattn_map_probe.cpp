// fattn_map_probe.cpp — the fragment-map WITNESS that `m16_fattn_score` never had.
//
// WHY THIS EXISTS
// `prototype/attn_wmma/fattn_score_handover.md` §5d states the gap in one line: everything the form is
// built on is "transcription plus a consistency argument" — the C map and the A/B map were read out of
// `engines/llama.cpp/ggml/src/ggml-cuda/mma.cuh:183,195,207,552,560` and cross-checked against each other
// (the `get_half2` output shape equals the A/B operand shape, the emitted operand register counts are
// 8/8/8 -> 8, `sizeof` is static_asserted). Nothing has ever been *executed against*. That is precisely
// the defect that disqualified D58's `m16_bf16frag`, whose header said "same hypothesised map" and whose
// C drain read a contiguous 8-run. A consistency argument between two transcriptions cannot tell a
// correct pair from a pair that is wrong in the same way, so this file replaces the argument with a
// measurement: one warp, one launch, patterned operands, every lane's 8 C values and that lane's own id
// stored to memory, and the assertion computed off-device by `check_fattn_map.py`.
//
// THE CLAIM UNDER TEST, one sentence per tile
//   C tile (f32):   lane t holds C[t%16][2l + t/16] for l = 0..7   (mma.cuh:195,207 — one PARITY of one
//                   row, shared with lane t^16, which holds the other parity).
//   A/B tile (f16): lane t holds operand[t%16][run of 16 contiguous halves]  (mma.cuh:552,560), and a
//                   wmma's B operand is spelled with the SAME shape, its "row" being the N index.
//
// WHY THE PATTERNS ARE BUILT THE WAY THEY ARE
// `Q[q][d] = q, K[kv][d] = kv` — the pattern §5d suggests — is USELESS as a falsifier: it gives
// C[q][kv] = 256*q*kv, which is symmetric in q and kv, so a transposed C tile satisfies it identically
// and the probe would pass on a map that is transposed. Every pattern here instead makes each tile cell
// hold a value that ENCODES ITS OWN COORDINATES, so any relabelling of (row, col) — transpose, parity
// swap, lane permutation, register-order error — shows up as a wrong number rather than as a
// plausible-looking sum. The encoding is exact in f16 and in f32 (every product is an integer below
// 2^24), so a mismatch is never a rounding artefact.
//
//   P1  A = Q1[q][0]=q, Q1[q][1]=1        B = K1[kv][0]=16, K1[kv][1]=kv      ONE tile
//       => C[q][kv] = 16q + kv, unique per cell, row-weight on A, col-weight on B.
//   P2  A = Q2[q][d]=q+1 (all 256 dims)   B = K2[kv][d]=kv+1 (all 256 dims)   16 tiles accumulated
//       => C[q][kv] = 256(q+1)(kv+1). Symmetric, so it CANNOT adjudicate orientation — deliberately. It
//          witnesses coverage and accumulation: each of the 256 dims contributes exactly once across the
//          16 K-steps (a doubled or dropped dim changes the total), and the accumulator chain carries
//          across the steps. `check_fattn_map.py` is told which pattern adjudicates what and refuses to
//          let a symmetric pattern claim an orientation result.
//   P3  A = Q3[q][0]=q, Q3[q][1]=1        B = K3[kv][0]=1, K3[kv][1]=16kv     ONE tile
//       => C[q][kv] = 16kv + q: P1 with the axis weights exchanged, so an A/B asymmetry in the hardware
//          maps moves P1 and P3 in OPPOSITE directions. That is what lets a failure be attributed to one
//          operand side instead of being left as "something about the tile".
//   P4  A = Vt[d][0]=d, Vt[d][1]=1 (row stride 16 = the transposed V stage)                 ONE tile
//       B = Pm[q][0]=1, Pm[q][1]=16q     (row stride 16 = P, q-major, kv contiguous)
//       => C[d][q] = d + 16q. This one is about the PORT's own spellings, not about the hardware: it
//          exercises the score tile's roles exchanged — A's row index is a head_dim (the `v_addr(d,kv)`
//          transposed-V claim of `wmma_fattn_score.cpp`), B's row index is a query row, and the drain
//          reads the C tile as C[dim][row]. A wrong `v_addr` (row-major V instead of transposed) is a
//          whole-tensor wrong answer here, not a subtle one. WHAT IT DOES NOT WITNESS: the `get_half2`
//          C -> B relabelling itself (that exchange moves values across lanes; it is admitted by the
//          §5b numerics gate, not here), and anything about LDS — the operands are read from global
//          addresses built by the same index expressions the port uses.
//
// THE INJECTIONS are store-side permutations of the raw dump (`--inject=<name>`). They model the ways an
// implementation loses a map — a lane-half swap, a register-order reversal, an off-by-one lane — and they
// exist so `check_fattn_map.py` can show its comparator is not vacuous end-to-end (device -> dump ->
// compare). A wrong *map* is injected host-side instead, by rebuilding the expectation from the wrong
// formula, because a store permutation cannot emulate a relabelled tile without moving data across
// lanes — and an LDS round trip that writes and reads with the same index formula would cancel by
// construction, which is the trap class this probe exists to avoid.
//
// BUILD:  bash prototype/attn_wmma/build_map_probe.sh          (link step included; it launches)
// RUN:    python3 prototype/attn_wmma/check_fattn_map.py       (the checker; PASS/FAIL, no timing)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#define DH 256
#define PATTERNS 4
#define CELLS 8       // ne = I*J/32 = 8 f32 per lane for a 16x16 f32 tile (mma.cuh:183)
#define LANES 32

// The operand payload types, sized by the transcription under test — the same static_asserts the form
// carries, so a wrong size is a compile error here rather than a wrong number.
using ARegs = short __attribute__((ext_vector_type(16)));
using CRegs = float __attribute__((ext_vector_type(8)));
static_assert(sizeof(ARegs) == 32, "A/B tile is 8 half2 = 16 halves per lane (mma.cuh:535,577)");
static_assert(sizeof(CRegs) == 32, "C tile is ne = I*J/32 = 8 f32 per lane (mma.cuh:183)");

namespace {

// The store-side injections. `none` is the truth; the others permute WHERE each lane's 8 raw values
// land, which is the failure shape of a lane/register mislabelling.
enum Inject {
    INJ_NONE = 0,
    INJ_LANE_HALF_SWAP = 1,   // lane t's values land in lane t^16's slots
    INJ_REG_REVERSE = 2,      // the 8 register slots are written in reverse order
    INJ_LANE_ROT = 3,         // every value lands one lane up (an off-by-one in the lane index)
};

__device__ __forceinline__ ARegs load_tile(const __half* p) {
    union { uint4 u[2]; ARegs s; } v;
    v.u[0] = *reinterpret_cast<const uint4*>(p);
    v.u[1] = *reinterpret_cast<const uint4*>(p + 8);
    return v.s;
}

// One 16x16x(16*steps) product with the A/B operand spelling under test: lane t reads the 16 contiguous
// halves of ROW t%16 at run offset 16*step, for both operands, exactly as `score_leg` and `pv_leg` do.
//
// `swap` exchanges the two wmma OPERAND SLOTS while leaving both operand SPELLINGS (and the C drain the
// checker assumes) untouched. It exists because the first run of this probe falsified the transcription,
// and two readings of that falsification produce identical predictions for every pattern: "the C map is
// transposed" and "the operand roles are exchanged". The swap mode is what makes the finding actionable
// either way — it shows that the transcribed C drain becomes exactly right when the operand order is
// exchanged, which is the rule a port has to implement, with or without a verdict on which sentence of
// `mma.cuh` the transcription misread.
__device__ __forceinline__ CRegs tile_product(const __half* A, int aStride, const __half* B,
                                             int bStride, int steps, int swap) {
    CRegs c{};
    const int r = threadIdx.x & 15;
    for (int s = 0; s < steps; ++s) {
        const ARegs a = load_tile(A + (size_t)r * aStride + 16 * s);
        const ARegs b = load_tile(B + (size_t)r * bStride + 16 * s);
        c = swap ? __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a, c)
                 : __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
    return c;
}

// Where lane t's l-th C value is written, and where lane t writes its OWN id. The id moves with the
// values under `none` and stays put under the injections, which is what makes the dump's own
// lane/value association checkable rather than assumed.
__device__ __forceinline__ void publish(float* out, int* laneids, int pattern, const CRegs c,
                                        int inject) {
    const int t = threadIdx.x & (LANES - 1);
#pragma unroll
    for (int l = 0; l < CELLS; ++l) {
        int slot = t * CELLS + l;
        if (inject == INJ_LANE_HALF_SWAP) slot = (t ^ 16) * CELLS + l;
        else if (inject == INJ_REG_REVERSE) slot = t * CELLS + (CELLS - 1 - l);
        else if (inject == INJ_LANE_ROT) slot = ((t + 1) & (LANES - 1)) * CELLS + l;
        out[pattern * LANES * CELLS + slot] = c[l];
    }
    if (pattern == 0) laneids[t] = t;
}

const char* inject_name(int inject) {
    switch (inject) {
        case INJ_LANE_HALF_SWAP: return "lane-half-swap";
        case INJ_REG_REVERSE: return "reg-reverse";
        case INJ_LANE_ROT: return "lane-rot";
        default: return "none";
    }
}

}  // namespace

// One warp, one block, one launch. Four tiles, four patterns, 4*32*8 floats plus the lane ids.
extern "C" __global__ void fattn_map_probe(const __half* __restrict__ q1, const __half* __restrict__ k1,
                                           const __half* __restrict__ q2, const __half* __restrict__ k2,
                                           const __half* __restrict__ q3, const __half* __restrict__ k3,
                                           const __half* __restrict__ vt, const __half* __restrict__ pm,
                                           float* __restrict__ out, int* __restrict__ laneids,
                                           int inject, int swap) {
    const int t = threadIdx.x & (LANES - 1);
    publish(out, laneids, 0, tile_product(q1, DH, k1, DH, 1, swap), inject);        // P1: cell-id
    publish(out, laneids, 1, tile_product(q2, DH, k2, DH, DH / 16, swap), inject);  // P2: coverage
    publish(out, laneids, 2, tile_product(q3, DH, k3, DH, 1, swap), inject);        // P3: exchanged
    publish(out, laneids, 3, tile_product(vt, 16, pm, 16, 1, swap), inject);        // P4: port spellings
}

#define CK(x)                                                                        \
    do {                                                                             \
        const hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                      \
            std::printf("PROBE-ERROR %s at %s:%d\n", hipGetErrorString(e_),         \
                        __FILE__, __LINE__);                                         \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

// Host-side BF16-free F16 bit pattern for a small non-negative integer, written out longhand: every
// pattern value is an exact integer below 2048, so this is exact by construction and it keeps the host
// free of `__half` conversion intrinsics whose host availability varies across ROCm versions. Bit
// layout is the standard one: sign 0, exponent biased by 15, 10 explicit mantissa bits.
unsigned short half_bits(int v) {
    if (v == 0) return 0;
    int msb = 0;
    while ((1 << (msb + 1)) <= v) ++msb;
    const int guard = 10 - msb;                    // explicit mantissa bits available
    const unsigned int mant = (unsigned int)((v - (1 << msb)) << guard);
    return (unsigned short)(((msb + 15) << 10) | (int)mant);
}

}  // namespace

int main(int argc, char** argv) {
    int inject = INJ_NONE;
    int swap = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--swap=", 7)) {
            swap = atoi(argv[i] + 7);
        } else if (!strncmp(argv[i], "--inject=", 9)) {
            const char* v = argv[i] + 9;
            if (!strcmp(v, "none")) inject = INJ_NONE;
            else if (!strcmp(v, "lane-half-swap")) inject = INJ_LANE_HALF_SWAP;
            else if (!strcmp(v, "reg-reverse")) inject = INJ_REG_REVERSE;
            else if (!strcmp(v, "lane-rot")) inject = INJ_LANE_ROT;
            else {
                std::printf("PROBE-ERROR unknown --inject=%s "
                            "(none|lane-half-swap|reg-reverse|lane-rot)\n", v);
                return 2;
            }
        } else {
            std::printf("PROBE-ERROR unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    // ---- the patterned operands, built on the host and copied up. Nothing generates the pattern on the
    // ---- device, so what the checker expects and what the device saw come from two different programs.
    static unsigned short host[16 * DH];
    auto fill = [&](int rows, int stride, auto fn) {
        for (int r = 0; r < rows; ++r)
            for (int d = 0; d < stride; ++d) host[r * stride + d] = half_bits(fn(r, d));
    };
    fill(16, DH, [](int q, int d) { return d == 0 ? q : (d == 1 ? 1 : 0); });     // Q1
    __half* q1 = nullptr;
    CK(hipMalloc(&q1, sizeof(host)));
    CK(hipMemcpy(q1, host, sizeof(host), hipMemcpyHostToDevice));
    fill(16, DH, [](int kv, int d) { return d == 0 ? 16 : (d == 1 ? kv : 0); });  // K1
    __half* k1 = nullptr;
    CK(hipMalloc(&k1, sizeof(host)));
    CK(hipMemcpy(k1, host, sizeof(host), hipMemcpyHostToDevice));
    fill(16, DH, [](int q, int d) { return q + 1; });                             // Q2
    __half* q2 = nullptr;
    CK(hipMalloc(&q2, sizeof(host)));
    CK(hipMemcpy(q2, host, sizeof(host), hipMemcpyHostToDevice));
    fill(16, DH, [](int kv, int d) { return kv + 1; });                           // K2
    __half* k2 = nullptr;
    CK(hipMalloc(&k2, sizeof(host)));
    CK(hipMemcpy(k2, host, sizeof(host), hipMemcpyHostToDevice));
    fill(16, DH, [](int q, int d) { return d == 0 ? q : (d == 1 ? 1 : 0); });     // Q3 (as Q1)
    __half* q3 = nullptr;
    CK(hipMalloc(&q3, sizeof(host)));
    CK(hipMemcpy(q3, host, sizeof(host), hipMemcpyHostToDevice));
    fill(16, DH, [](int kv, int d) { return d == 0 ? 1 : (d == 1 ? 16 * kv : 0); });  // K3
    __half* k3 = nullptr;
    CK(hipMalloc(&k3, sizeof(host)));
    CK(hipMemcpy(k3, host, sizeof(host), hipMemcpyHostToDevice));
    fill(16, 16, [](int d, int kv) { return kv == 0 ? d : (kv == 1 ? 1 : 0); });  // Vt (transposed)
    __half* vt = nullptr;
    CK(hipMalloc(&vt, 16 * 16 * sizeof(short)));
    CK(hipMemcpy(vt, host, 16 * 16 * sizeof(short), hipMemcpyHostToDevice));
    fill(16, 16, [](int q, int kv) { return kv == 0 ? 1 : (kv == 1 ? 16 * q : 0); });  // Pm (q-major)
    __half* pm = nullptr;
    CK(hipMalloc(&pm, 16 * 16 * sizeof(short)));
    CK(hipMemcpy(pm, host, 16 * 16 * sizeof(short), hipMemcpyHostToDevice));

    float* out = nullptr;
    int* laneids = nullptr;
    const size_t nout = (size_t)PATTERNS * LANES * CELLS;
    CK(hipMalloc(&out, nout * sizeof(float)));
    CK(hipMalloc(&laneids, LANES * sizeof(int)));
    CK(hipMemset(out, 0xff, nout * sizeof(float)));     // NaN: an unwritten cell is not a plausible 0
    CK(hipMemset(laneids, 0x7f, LANES * sizeof(int)));

    fattn_map_probe<<<1, LANES>>>(q1, k1, q2, k2, q3, k3, vt, pm, out, laneids, inject, swap);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    static float hout[2048];
    int hlids[LANES];
    CK(hipMemcpy(hout, out, nout * sizeof(float), hipMemcpyDeviceToHost));
    CK(hipMemcpy(hlids, laneids, LANES * sizeof(int), hipMemcpyDeviceToHost));

    std::printf("MAPPROBE inject=%s swap=%d lanes=%d cells=%d patterns=%d\n", inject_name(inject),
                swap, LANES, CELLS, PATTERNS);
    for (int t = 0; t < LANES; ++t) {
        std::printf("LANE %d laneid=%d", t, hlids[t]);
        for (int p = 0; p < PATTERNS; ++p) {
            std::printf(" p%d=", p + 1);
            for (int l = 0; l < CELLS; ++l) {
                const float v = hout[(size_t)p * LANES * CELLS + t * CELLS + l];
                std::printf(l ? " %g" : "%g", (double)v);
            }
        }
        std::printf("\n");
    }
    std::printf("MAPPROBE DONE\n");
    return 0;
}
