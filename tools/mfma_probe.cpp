// tools/mfma_probe.cpp — does the matrix unit execute on `gfx1151` at all? Correctness only.
//
// Why this exists: route (c) of the funded prefill package (`bench/plan-prefill-2026-10-12.md`,
// "Funded: prefill compute") is hand-tiled matrix-unit work, and there is a public report that the
// support is not there — rocm-libraries issue 4618, "gfx1151 support is still broken in practice as
// of February 2026". `rocwmma.hpp` IS installed (2.2.1), so the question is answerable at the cost of
// one small program instead of one half-ported kernel. This is that program: the smallest real test —
// a 16x16x16-tile GEMM through `rocwmma::mma_sync`, fp16 and bf16 in, f32 accumulate, checked against
// a CPU reference at several sizes. A failure here is the result, not a setback.
//
// What is NOT in this file, deliberately: no timing, no best-of-N, no warm loop, no GB/s, no TFLOP/s.
// AGENTS.md: a worker's device run is a deterministic PASS/FAIL gate under ~60 s and takes no
// number that smells like a rate — `devices/strix-halo-compute.md` already carries the rate rows,
// and they are the main agent's to take. The clock question, if it matters, is theirs too.
//
// Build:  tools/build.sh mfma_probe
// ISA:    tools/mfma_isa.sh        (compile-only; prints whether the target got v_wmma_* or v_fma_*)
// Run:    tools/build/mfma_probe [ --case MxNxK ] [ --dtypes fp16,bf16,fp16acc ] [ --fma ]
//
// Layout choice, because it is the one thing this spike tells the implementation for free: the
// checkpoint stores a projection as [out, in] row-major, and C[M,N] = A[M,K] · W[N,K]ᵗ means W is
// *already* the col-major KxN operand WMMA wants. So the weight tensor goes to
// `load_matrix_sync` untransposed, with ldm = K. That is the shape tested here.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma-version.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                              \
    do {                                                                   \
        hipError_t e_ = (x);                                               \
        if (e_ != hipSuccess) {                                            \
            std::printf("HIP_ERROR %s:%d %s\n", __FILE__, __LINE__,        \
                        hipGetErrorString(e_));                            \
            return 2;                                                      \
        }                                                                  \
    } while (0)

namespace {

// One spelling per format for "to float" and "from float", so the kernel bodies never rely on an
// implicit conversion the type does not offer (hip_bfloat16 refuses `Acc(0.f)` outright, which is the
// kind of thing a bf16/fp16 pair always does).
template <typename T>
__host__ __device__ inline float to_f(T v) { return (float)v; }
template <typename T>
__host__ __device__ inline T from_f(float v) { return T(v); }

// One wavefront computes one 16x16 output tile; the K axis is walked in 16-wide steps and the
// accumulator stays in registers the whole time (that is the point of the exercise).
//
// Sized for M, N, K that are multiples of 16. A partial tile reads outside its buffer, which is an
// implementation problem for whoever builds the route (pad, or mask the IO) and not a capability
// question, so it is stated here rather than silently avoided.
template <typename T, typename Acc>
__global__ void wmma_gemm(const T* __restrict__ A, const T* __restrict__ W,
                          Acc* __restrict__ C, int M, int N, int K, int tiles_n)
{
    const uint32_t tile = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t m0 = (tile / tiles_n) * 16;
    const uint32_t n0 = (tile % tiles_n) * 16;
    if (tile * 0 + m0 >= (uint32_t)M || n0 >= (uint32_t)N) return;   // warp-uniform, so legal

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, T, rocwmma::row_major> a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, T, rocwmma::col_major> b;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, Acc, rocwmma::row_major> c;
    rocwmma::fill_fragment(c, from_f<Acc>(0.f));

    for (int k0 = 0; k0 < K; k0 += 16) {
        rocwmma::load_matrix_sync(a, A + (size_t)m0 * K + k0, K);
        rocwmma::load_matrix_sync(b, W + (size_t)n0 * K + k0, K);
        rocwmma::mma_sync(c, a, b, c);
    }
    rocwmma::store_matrix_sync(C + (size_t)m0 * N + n0, c, N);
    __syncwarp();
}

// The control kernel: same product, one thread per output element, plain scalar FMA. Its only jobs
// are (a) to be an on-device reference that does not depend on WMMA being real, and (b) to give
// `tools/mfma_isa.sh` something in the same translation unit that MUST compile to v_fma_f32, so the
// WMMA kernel's instruction mix can be read as evidence rather than as an absence of evidence.
template <typename T, typename Acc>
__global__ void fma_gemm(const T* __restrict__ A, const T* __restrict__ W,
                         Acc* __restrict__ C, int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N, n = idx % N;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s += to_f(A[(size_t)m * K + k]) * to_f(W[(size_t)n * K + k]);
    C[idx] = from_f<Acc>(s);
}

// ---------------------------------------------------------------- host-side dtype plumbing
// The buffers hold 16-bit patterns; the reference converts with the HIP host intrinsics so there is
// one spelling of each format and no implicit float narrowing in the comparison path.
using bits16 = uint16_t;

inline float to_f32_fp16(bits16 b) {
    __half h;
    __builtin_memcpy(&h, &b, 2);
    return __half2float(h);
}
inline float to_f32_bf16(bits16 b) {
    hip_bfloat16 h;
    __builtin_memcpy(&h, &b, 2);
    return float(h);
}
inline bits16 from_f32_fp16(float f) {
    __half h = __float2half(f);
    bits16 b;
    __builtin_memcpy(&b, &h, 2);
    return b;
}
inline bits16 from_f32_bf16(float f) {
    hip_bfloat16 h = hip_bfloat16(f);
    bits16 b;
    __builtin_memcpy(&b, &h, 2);
    return b;
}

struct Case {
    int m, n, k;
};

struct Stats {
    double cos, meanrel, maxabs, maxrel, worst_at;
    bool ok;
};

// Aggregates only. `prototype/mlp_gemv/results.md` F4 is the reason: max-ULP and maxrel are blind to
// what a long dot actually does (cancellation puts the biggest relative error on elements that
// cancelled to ~1e-6 of the tensor max), so the gate keys on cos and meanrel and prints maxrel with
// the element it came from rather than pretending it is a bound.
Stats compare(const std::vector<float>& got, const std::vector<float>& ref) {
    double dot = 0, na = 0, nb = 0, sum_abs_err = 0, sum_abs_ref = 0;
    double maxabs = 0, maxrel = 0, worst_at = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = ref[i], d = a - b;
        dot += a * b; na += a * a; nb += b * b;
        sum_abs_err += std::fabs(d);
        sum_abs_ref += std::fabs(b);
        if (std::fabs(d) > maxabs) maxabs = std::fabs(d);
        const double r = std::fabs(d) / (std::fabs(b) + 1e-3);
        if (r > maxrel) { maxrel = r; worst_at = b; }
    }
    Stats s;
    s.cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 1.0;
    s.meanrel = sum_abs_err / (sum_abs_ref + 1e-30);
    s.maxabs = maxabs; s.maxrel = maxrel; s.worst_at = worst_at;
    s.ok = false;
    return s;
}

template <typename T, typename Acc>
bool run_one(const char* dname, T* dA, T* dW, Acc* dC, const std::vector<bits16>& hA,
             const std::vector<bits16>& hW, const std::vector<float>& ref, Case c,
             float (*dec)(bits16), Stats& out) {
    const int tiles_n = c.n / 16;
    const int warps = (c.m / 16) * tiles_n;
    const int threads = 256;                        // 8 wavefronts per block on this part
    const int blocks = (warps * 32 + threads - 1) / threads;
    std::vector<float> got((size_t)c.m * c.n, -1.0f);
    std::vector<bits16> got16;
    if (sizeof(Acc) == 2) got16.assign(got.size(), 0);
    CK(hipMemset(dC, 0, (size_t)c.m * c.n * sizeof(Acc)));
    wmma_gemm<T, Acc><<<blocks, threads>>>(dA, dW, dC, c.m, c.n, c.k, tiles_n);
    const hipError_t le = hipGetLastError();
    const hipError_t se = hipDeviceSynchronize();
    if (le != hipSuccess || se != hipSuccess) {
        std::printf("  %-22s %5dx%-5d x%-5d  LAUNCH/RUN FAILED  last=%s sync=%s\n", dname, c.m,
                    c.n, c.k, hipGetErrorString(le), hipGetErrorString(se));
        out.ok = false;
        return false;
    }
    // The accumulator width decides how the output comes back, and reading a 2-byte accumulator into a
    // 4-byte float buffer is a silent bitcast-with-wrong-stride: every value lands in the low half of a
    // float slot and the comparison then reports 1e2-sized "errors" for a kernel that was fine. It cost
    // a false FAIL on the first run of this probe. Only an f32 accumulator is already floats.
    if (sizeof(Acc) == 2) {
        CK(hipMemcpy(got16.data(), dC, (size_t)c.m * c.n * 2, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < got.size(); ++i) got[i] = to_f32_fp16(got16[i]);
    } else {
        CK(hipMemcpy(got.data(), dC, (size_t)c.m * c.n * 4, hipMemcpyDeviceToHost));
    }
    Stats s = compare(got, ref);
    // f32 accumulate over a K-length dot of fp16/bf16 inputs is a near-exact route: the tolerance is
    // the f32 accumulation noise, not a fudge. fp16 accumulate is a different product and gets its
    // own, looser, line.
    const bool f32acc = std::strstr(dname, "acc16") == nullptr;
    s.ok = (s.cos >= (f32acc ? 0.999999 : 0.999)) && (s.meanrel <= (f32acc ? 1e-3 : 5e-2));
    out = s;
    std::printf("  %-22s %5dx%-5d x%-5d  %-4s cos=%.9f meanrel=%.3e maxabs=%.3e "
                "maxrel=%.3e (at ref=%.6g)\n",
                dname, c.m, c.n, c.k, s.ok ? "PASS" : "FAIL", s.cos, s.meanrel, s.maxabs,
                s.maxrel, s.worst_at);
    (void)dec;
    return s.ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<Case> cases = {{16, 16, 16}, {64, 96, 128}, {128, 512, 1024}};
    bool do_fp16 = true, do_bf16 = true, do_fp16acc = true, do_fma = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dtypes") && i + 1 < argc) {
            do_fp16 = do_bf16 = do_fp16acc = false;
            std::string s = argv[++i];
            if (s.find("fp16") != std::string::npos) do_fp16 = true;
            if (s.find("bf16") != std::string::npos) do_bf16 = true;
            if (s.find("fp16acc") != std::string::npos) do_fp16acc = true;
        } else if (!std::strcmp(argv[i], "--case") && i + 1 < argc) {
            Case c{};
            if (std::sscanf(argv[++i], "%dx%dx%d", &c.m, &c.n, &c.k) != 3) {
                std::printf("--case wants MxNxK\n");
                return 2;
            }
            cases = {c};
        } else if (!std::strcmp(argv[i], "--fma")) {
            do_fma = true;
        }
    }
    for (auto& c : cases) {
        if (c.m % 16 || c.n % 16 || c.k % 16) {
            std::printf("size %dx%dx%d is not a multiple of 16 in every axis; this probe does not "
                        "pad partial tiles (see the header note)\n", c.m, c.n, c.k);
            return 2;
        }
    }

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s  gcnArchName=%s  CUs(HIP)=%d  wavefront=%d  sharedMemPerBlock=%zu\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.warpSize,
                prop.sharedMemPerBlock);
    std::printf("rocwmma %d.%d.%d\n", ROCWMMA_VERSION_MAJOR, ROCWMMA_VERSION_MINOR,
                ROCWMMA_VERSION_PATCH);
    std::printf("correctness only — no timing taken, no rate printed\n");

    int fails = 0;
    for (Case c : cases) {
        std::mt19937 rng(20261013);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        std::vector<bits16> hA((size_t)c.m * c.k), hW((size_t)c.n * c.k);
        for (auto& x : hA) x = from_f32_fp16(u(rng));
        for (auto& x : hW) x = from_f32_fp16(u(rng));
        std::vector<bits16> hA_b(hA.size()), hW_b(hW.size());
        for (size_t i = 0; i < hA.size(); ++i) hA_b[i] = from_f32_bf16(to_f32_fp16(hA[i]));
        for (size_t i = 0; i < hW.size(); ++i) hW_b[i] = from_f32_bf16(to_f32_fp16(hW[i]));

        std::printf("\n=== %dx%dx%d  (C[m,n] = A[m,k] * W[n,k]^T, W row-major = col-major B)\n",
                    c.m, c.n, c.k);

        // References, computed once per size in double so both device routes are judged from above.
        auto reference = [&](const std::vector<bits16>& A, const std::vector<bits16>& W,
                             float (*dec)(bits16)) {
            std::vector<float> ref((size_t)c.m * c.n);
            for (int m = 0; m < c.m; ++m)
                for (int n = 0; n < c.n; ++n) {
                    double s = 0.0;
                    for (int k = 0; k < c.k; ++k)
                        s += (double)dec(A[(size_t)m * c.k + k]) * (double)dec(W[(size_t)n * c.k + k]);
                    ref[(size_t)m * c.n + n] = (float)s;
                }
            return ref;
        };
        const auto ref16 = reference(hA, hW, to_f32_fp16);
        const auto refb = reference(hA_b, hW_b, to_f32_bf16);
        // The fp16-accumulator line is judged against the same product rounded once to fp16, not
        // against the f32-accumulated one: the whole point of that line is accumulation noise, and
        // judging accumulation noise against an exact reference measures the wrong thing. The width
        // of the gap is printed, because it is the number that says whether fp16 accumulation could
        // ever be trusted — it is NOT a candidate for this engine (D-3 pins F32 accumulation), and the
        // line is here to show the instruction exists and what it costs.
        auto ref16a = ref16;
        for (auto &x : ref16a) x = to_f32_fp16(from_f32_fp16(x));

        void *dA16, *dW16, *dAb, *dWb;
        CK(hipMalloc(&dA16, hA.size() * 2)); CK(hipMalloc(&dW16, hW.size() * 2));
        CK(hipMalloc(&dAb, hA.size() * 2)); CK(hipMalloc(&dWb, hW.size() * 2));
        CK(hipMemcpy(dA16, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dW16, hW.data(), hW.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dAb, hA_b.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dWb, hW_b.data(), hW.size() * 2, hipMemcpyHostToDevice));
        float* dCf; rocwmma::float16_t* dCh;
        CK(hipMalloc(&dCf, (size_t)c.m * c.n * 4));
        CK(hipMalloc(&dCh, (size_t)c.m * c.n * 2));
        Stats s{};
        if (do_fp16)
            fails += !run_one<rocwmma::float16_t, float>("fp16 in / f32 acc",
                         (rocwmma::float16_t*)dA16, (rocwmma::float16_t*)dW16, dCf, hA, hW, ref16,
                         c, to_f32_fp16, s);
        if (do_bf16)
            fails += !run_one<rocwmma::bfloat16_t, float>("bf16 in / f32 acc",
                         (rocwmma::bfloat16_t*)dAb, (rocwmma::bfloat16_t*)dWb, dCf, hA_b, hW_b,
                         refb, c, to_f32_bf16, s);
        if (do_fp16acc)
            fails += !run_one<rocwmma::float16_t, rocwmma::float16_t>("fp16 in / fp16 acc16",
                         (rocwmma::float16_t*)dA16, (rocwmma::float16_t*)dW16, dCh, hA, hW, ref16a,
                         c, to_f32_fp16, s);

        if (do_fma) {
            std::vector<float> got((size_t)c.m * c.n, -1.0f);
            const int threads = 256;
            fma_gemm<rocwmma::float16_t, float><<<(c.m * c.n + threads - 1) / threads, threads>>>(
                (const rocwmma::float16_t*)dA16, (const rocwmma::float16_t*)dW16, dCf, c.m, c.n,
                c.k);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(got.data(), dCf, got.size() * 4, hipMemcpyDeviceToHost));
            Stats f = compare(got, ref16);
            f.ok = f.cos >= 0.999999;
            std::printf("  %-22s %5dx%-5d x%-5d  %-4s cos=%.9f meanrel=%.3e   (scalar control)\n",
                        "fp16 in / f32 fma", c.m, c.n, c.k, f.ok ? "PASS" : "FAIL", f.cos,
                        f.meanrel);
            fails += !f.ok;
        }
        CK(hipFree(dA16)); CK(hipFree(dW16)); CK(hipFree(dAb)); CK(hipFree(dWb));
        CK(hipFree(dCf)); CK(hipFree(dCh));
    }

    std::printf("\nMFMA PROBE %s — %d failing case/dtype line(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
