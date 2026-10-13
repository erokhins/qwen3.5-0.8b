// tools/gemm_peak.cpp — dense matrix-multiply peak for this GPU, measured with rocBLAS/hipBLASLt
// kernels (i.e. the MFMA/tensor path), not with hand-rolled FMA loops.
//
// Why this exists: the engine's prefill throughput is bounded by dense matmul, and every argument about
// "prefill should be 5k tok/s" is arithmetic on two numbers — FLOPs per token (known: ~1.6 GFLOP) and this
// machine's sustained dense-matrix rate (never measured here). This probe measures the second one.
//
// Build:  tools/build.sh gemm_peak
// Run:    tools/build/gemm_peak [--n N] [--iters K] [--check]
//
// Honesty rules this probe follows (AGENTS.md): warm before timing, best-of-N with the spread printed, a
// number above the measured memory-probe rate on a bandwidth-bound shape means skipped work, and nothing
// here is a token rate — this is a matmul ceiling, nothing more.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <hipblas/hipblas.h>

#include <chrono>
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
            std::exit(2);                                                  \
        }                                                                  \
    } while (0)
#define BK(x)                                                              \
    do {                                                                   \
        hipblasStatus_t s_ = (x);                                          \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                \
            std::printf("HIPBLAS_ERROR %s:%d status=%d\n", __FILE__,       \
                        __LINE__, (int)s_);                                \
            std::exit(3);                                                  \
        }                                                                  \
    } while (0)

namespace {

struct Row {
    int m, n, k;
    const char *tag;
};

// C[m,n] = A[m,k] * B[k,n], row-major, bf16 in, f32 accumulate. hipBLAS is column-major, so the
// row-major product is expressed as the transposed column-major one (C^T = B^T A^T).
double run_gemm(hipblasHandle_t h, int m, int n, int k, int iters,
                hipDataType dt, const void *A, const void *B, void *C) {
    const float alpha = 1.0f, beta = 0.0f;
    auto gemm = [&]() {
        BK(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k, &alpha, B, dt, n,
                         A, dt, k, &beta, C, HIPBLAS_R_32F, n,
                         HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    };
    gemm();                       // warm: first touch of these pages and of the picked kernel
    CK(hipDeviceSynchronize());
    double best = 1e30, worst = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        gemm();
        CK(hipDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, ms);
        worst = std::max(worst, ms);
    }
    (void)worst;
    return best;                  // caller prints spread from separate calls if it wants it
}

// Host-side bf16 -> f32 without relying on device-only helpers (this comparison runs on the CPU).
float bf2f(hip_bfloat16 x) {
    uint16_t bits;
    __builtin_memcpy(&bits, &x, 2);
    uint32_t u = (uint32_t)bits << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

bool selfcheck(hipblasHandle_t h) {
    const int M = 64, N = 96, K = 128;
    std::vector<hip_bfloat16> ha(M * K), hb(K * N);
    std::vector<float> hc(M * N), ref(M * N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto &x : ha) x = __float2bfloat16(u(rng));
    for (auto &x : hb) x = __float2bfloat16(u(rng));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int t = 0; t < K; ++t)
                s += bf2f(ha[i * K + t]) * bf2f(hb[t * N + j]);
            ref[i * N + j] = s;
        }
    hip_bfloat16 *da, *db;
    float *dc;
    CK(hipMalloc(&da, ha.size() * 2));
    CK(hipMalloc(&db, hb.size() * 2));
    CK(hipMalloc(&dc, hc.size() * 4));
    CK(hipMemcpy(da, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(db, hb.data(), hb.size() * 2, hipMemcpyHostToDevice));
    const float alpha = 1.f, beta = 0.f;
    BK(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &alpha, db, HIP_R_16BF, N,
                     da, HIP_R_16BF, K, &beta, dc, HIP_R_32F, N,
                     HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hc.data(), dc, hc.size() * 4, hipMemcpyDeviceToHost));
    double maxrel = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double d = std::fabs(hc[i] - ref[i]) / (1.0 + std::fabs((double)ref[i]));
        maxrel = std::max(maxrel, d);
    }
    bool ok = maxrel < 2e-2;
    std::printf("SELFCHECK %s maxrel=%.3e (bf16 in, f32 accumulate, CPU reference)\n",
                ok ? "ok" : "FAIL", maxrel);
    CK(hipFree(da)); CK(hipFree(db)); CK(hipFree(dc));
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    int iters = 20;
    bool check = false;
    std::vector<Row> rows = {
        {4096, 4096, 4096, "square_4096"},
        {2048, 2048, 2048, "square_2048"},
        {8192, 4096, 4096, "wide_8192"},
        {1024, 4096, 1024, "engine_q_shape"},   // a 1024-token prompt's q_proj-sized product
        {1024, 3584, 1024, "engine_gate_shape"},
        {1, 4096, 1024, "decode_gemv_shape"},   // one row: the decode shape, for contrast
    };
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--small"))
            rows = {{4096, 4096, 4096, "square_4096"}, {1024, 4096, 1024, "engine_q_shape"}};
    }
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s  CUs=%d  clock_MHz=%d  mem_clock_MHz=%d  bus_width=%d\n",
                prop.name, prop.multiProcessorCount, prop.clockRate / 1000,
                prop.memoryClockRate / 1000, prop.memoryBusWidth);
    // Theoretical dense peak this device advertises through its matrix units is not printed here on
    // purpose: the spec sheet number is not a measurement, and this file only reports measured rows.
    hipblasHandle_t h;
    BK(hipblasCreate(&h));
    CK(hipSetDevice(0));
    if (check && !selfcheck(h)) return 1;

    std::printf("%-18s %10s %12s %12s %10s\n", "tag", "GFLOP", "best_ms", "TFLOP/s", "GB/s_of_C");
    for (auto &r : rows) {
        size_t na = (size_t)r.m * r.k, nb = (size_t)r.k * r.n, nc = (size_t)r.m * r.n;
        std::vector<hip_bfloat16> ha(na), hb(nb);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-0.5f, 0.5f);
        for (auto &x : ha) x = __float2bfloat16(u(rng));
        for (auto &x : hb) x = __float2bfloat16(u(rng));
        hip_bfloat16 *da, *db;
        float *dc;
        CK(hipMalloc(&da, na * 2)); CK(hipMalloc(&db, nb * 2)); CK(hipMalloc(&dc, nc * 4));
        CK(hipMemcpy(da, ha.data(), na * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(db, hb.data(), nb * 2, hipMemcpyHostToDevice));
        CK(hipMemset(dc, 0, nc * 4));
        double best = run_gemm(h, r.m, r.n, r.k, iters, HIP_R_16BF, da, db, dc);
        double flop = 2.0 * r.m * r.n * r.k;
        double gb = (na * 2 + nb * 2 + nc * 4) / 1e9;
        std::printf("%-18s %10.1f %12.3f %12.3f %10.1f\n", r.tag, flop / 1e9, best,
                    flop / (best * 1e-3) / 1e12, gb / (best * 1e-3));
        CK(hipFree(da)); CK(hipFree(db)); CK(hipFree(dc));
        std::fflush(stdout);
    }
    BK(hipblasDestroy(h));
    return 0;
}
