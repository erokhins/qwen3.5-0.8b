// tools/gemm_common.h — the pieces more than one hipBLAS probe in tools/ needs.
//
// Lifted out of tools/gemm_peak.cpp when tools/prefill_mix.cpp arrived, so that the 16-bit-C
// comparison prefill_mix --check makes IS the comparison gemm_peak --check makes rather than a
// near-copy of it (AGENTS.md: do not duplicate logic). gemm_peak's behaviour is unchanged by the
// move, including the byte patterns its --determinism hashes, which are quoted in
// progress/2026-10-13/prefill-route-spike-2026-10-13.md §3.1 and must keep reproducing.
//
// Build: this is a header; the probes that use it are built by tools/build.sh.
#ifndef TOOLS_GEMM_COMMON_H
#define TOOLS_GEMM_COMMON_H

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef CK
#define CK(x)                                                              \
    do {                                                                   \
        hipError_t e_ = (x);                                               \
        if (e_ != hipSuccess) {                                            \
            std::printf("HIP_ERROR %s:%d %s\n", __FILE__, __LINE__,        \
                        hipGetErrorString(e_));                            \
            std::exit(2);                                                  \
        }                                                                  \
    } while (0)
#endif
#ifndef BK
#define BK(x)                                                              \
    do {                                                                   \
        hipblasStatus_t s_ = (x);                                          \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                \
            std::printf("HIPBLAS_ERROR %s:%d status=%d\n", __FILE__,       \
                        __LINE__, (int)s_);                                \
            std::exit(3);                                                  \
        }                                                                  \
    } while (0)
#endif

namespace gemm_common {

// FNV-1a over the raw output bytes. A hash rather than a diff because the question is "are these the
// same bytes" at 100 MB, and because printing one line per shape is what makes a cross-PROCESS
// comparison possible: run the tool three times and compare the hashes.
inline uint64_t fnv1a_from(uint64_t h, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
inline uint64_t fnv1a(const uint8_t *p, size_t n) {
    return fnv1a_from(1469598103934665603ull, p, n);
}

inline uint16_t bits_of_bf16(float f) {
    hip_bfloat16 b = hip_bfloat16(f);
    uint16_t out;
    __builtin_memcpy(&out, &b, 2);
    return out;
}
inline uint16_t bits_of_fp16(float f) {
    __half h = __float2half_rn(f);
    uint16_t out;
    __builtin_memcpy(&out, &h, 2);
    return out;
}
inline float fp16_to_f32(uint16_t bits) {
    __half h;
    __builtin_memcpy(&h, &bits, 2);
    return __half2float(h);
}
inline float bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}
inline float bf2f(hip_bfloat16 x) {
    uint16_t bits;
    __builtin_memcpy(&bits, &x, 2);
    return bf16_to_f32(bits);
}

// The two width-paired helpers. `is_bf` selects the format for BOTH directions in one flag because a
// comparison that mixes them (encode bf16, decode fp16) is the bug this pairing exists to prevent, and
// because the routes use the widths in pairs: bf16 in / bf16 C, fp16 in / fp16 C. Asking hipBLAS for
// bf16 in / fp16 C is a refusal (status 7), not a conversion — §4.3.
inline uint16_t enc16(float f, bool is_bf) { return is_bf ? bits_of_bf16(f) : bits_of_fp16(f); }
inline float dec16(uint16_t bits, bool is_bf) { return is_bf ? bf16_to_f32(bits) : fp16_to_f32(bits); }

// A cheap fill whose values are exactly representable in both 16-bit widths: multiples of 1/256 in
// [-100, 100)/256. Exactly representable matters — it is what makes a bf16 run and an fp16 run
// multiply the same numbers, so any hash difference between the two can only be a different kernel
// being chosen (the reading recorded in the spike memo §3.1). No subnormals, no infinities.
inline void fill_16bit(uint8_t *p, size_t count, bool is_bf, unsigned salt) {
    for (size_t i = 0; i < count; ++i) {
        const float v = (float)(((int)(i * salt + 11) % 200) - 100) / 256.f;
        ((uint16_t *)p)[i] = enc16(v, is_bf);
    }
}

inline size_t dtype_bytes(hipDataType dt) { return dt == HIP_R_32F ? 4u : 2u; }

// The form the engine would actually call: the weight as the checkpoint stores it, [n, k] row-major,
// used directly as the transposed operand, so no transpose copy is ever made.
//
// Derivation, because getting this backwards is silent: BLAS is column-major and computes
// C_cm(m×n) = op(X)(m×k)·op(Y)(k×n). We want C_rm[m,n] = A_rm[m,k]·W_rm[n,k]^T, i.e.
// C_cm = W·Aᵗ, so m_blas=n, n_blas=m, X = W with OP_T (a row-major [n,k] buffer IS a cm k×n matrix,
// and transposing it gives the n×k we need, ld=k), Y = A with OP_N (a row-major [m,k] buffer is a cm
// k×m matrix, ld=k), and C with ldc=n.
inline hipblasStatus_t gemm_stored_weight(hipblasHandle_t h, int m, int n, int k, hipDataType dt,
                                          const void *A, const void *W, void *C,
                                          hipDataType dt_out) {
    const float alpha = 1.0f, beta = 0.0f;
    return hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k, &alpha, W, dt, k,
                         A, dt, k, &beta, C, dt_out, n,
                         HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
}

// Does a 16-bit `C` round where a host round would? The comparator is the reference rounded the same
// way, so the question is "same rounding point", not "exact product" — the reference is the dot of the
// values the device was actually given, in double, which is what leaves only the output rounding to
// compare.
//
// The ULP yardstick comes from the format, not from a guess: adding 1 to the raw bits of a positive
// 16-bit float is its next representable neighbour. A per-element relative metric would be wrong
// here (AGENTS.md, the cancellation lesson: a cancelled element divided by itself produced 6e12 "ULP"
// in an earlier draft of this check).
struct Bits16 {
    size_t ndiff;
    double worst_ulp;
};
inline Bits16 compare_16bit_output(const uint8_t *raw, const double *ref, size_t n, bool is_bf) {
    Bits16 out{0, 0.0};
    for (size_t i = 0; i < n; ++i) {
        uint16_t b;
        __builtin_memcpy(&b, raw + i * 2, 2);
        const double got = dec16(b, is_bf);
        const double want = (double)dec16(enc16((float)ref[i], is_bf), is_bf);
        const double d = std::fabs(got - want);
        const uint16_t wb = enc16((float)std::fabs(want), is_bf);
        const double ulp = std::max(
            std::fabs((double)dec16((uint16_t)(wb + 1), is_bf) - (double)dec16(wb, is_bf)), 1e-30);
        out.worst_ulp = std::max(out.worst_ulp, d / ulp);
        out.ndiff += (got != want);
    }
    return out;
}

// min / median / max of a sample set. Used by the timing modes only; the correctness modes never
// reach it.
struct Stat {
    double lo, med, hi;
    size_t n;
};
inline Stat stat_of(std::vector<double> v) {
    if (v.empty()) return Stat{0.0, 0.0, 0.0, 0};
    std::sort(v.begin(), v.end());
    return Stat{v.front(), v[v.size() / 2], v.back(), v.size()};
}

}  // namespace gemm_common

#endif  // TOOLS_GEMM_COMMON_H
