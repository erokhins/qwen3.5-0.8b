// tmp/gemm_probe.cpp — does `engine::kernels::gemm_bf16` compute the product the engine means?
//
// Links the engine's own route TU, so this exercises the real handle setup, the real workspace pin,
// the real stream choice and the real call — not a re-typed copy of them. Correctness only: one pass
// per shape, a CPU dot as the reference, no timing.
//
// What "right product" means here is SIZE of disagreement, not a zero count: reordering a k=1024 fp32
// sum already moves a few elements by a bf16 ULP or two against a double reference
// (prototype/mlp_gemv/floor_probe.py's F1), so ndiff==0 is the wrong gate. The gate is cosine, and the
// negative control is the same bytes judged against a weight with two rows swapped — a call that
// computed nothing, or left the poison in the buffer, would otherwise look "nearly right".
//
// build: hipcc -O3 -Wno-unused-command-line-argument -I. -Iengine -Wl,-rpath,/opt/rocm/lib \
//          tmp/gemm_probe.cpp engine/kernels_gemm.cpp -lhipblas -o tmp/gemm_probe
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "engine/kernels.h"
#include "tools/gemm_common.h"

#define CKP(x)                                                                        \
    do {                                                                              \
        const hipError_t e_ = (x);                                                     \
        if (e_ != hipSuccess) {                                                        \
            std::printf("HIP_ERROR %s:%d %s\n", __FILE__, __LINE__,                    \
                        hipGetErrorString(e_));                                         \
            return 2;                                                                  \
        }                                                                              \
    } while (0)

using namespace gemm_common;

static int one(int m, int n, int k) {
    std::vector<uint16_t> hA((size_t)m * k), hW((size_t)n * k);
    fill_16bit((uint8_t*)hA.data(), (size_t)m * k, true, 17u);
    fill_16bit((uint8_t*)hW.data(), (size_t)n * k, true, 23u);
    void *dA, *dW, *dC;
    CKP(hipMalloc(&dA, hA.size() * 2));
    CKP(hipMalloc(&dW, hW.size() * 2));
    CKP(hipMalloc(&dC, (size_t)m * n * 2));
    CKP(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    CKP(hipMemcpy(dW, hW.data(), hW.size() * 2, hipMemcpyHostToDevice));
    const std::vector<uint16_t> poison((size_t)m * n, 0x7F7F);   // a value no product stores
    CKP(hipMemcpy(dC, poison.data(), poison.size() * 2, hipMemcpyHostToDevice));

    engine::kernels::gemm_bf16(dW, dA, dC, m, n, k);
    CKP(hipDeviceSynchronize());

    std::vector<uint16_t> got((size_t)m * n);
    CKP(hipMemcpy(got.data(), dC, got.size() * 2, hipMemcpyDeviceToHost));

    size_t survivors = 0, ndiff_good = 0, ndiff_bad = 0;
    double worst_ulp = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t t = 0; t < got.size(); ++t) {
        if (got[t] == 0x7F7F) ++survivors;
        const double have = bf16_to_f32(got[t]);
        na += have * have;
    }
    for (int i = 0; i < m; ++i) {
        for (int jj = 0; jj < n; ++jj) {
            double acc = 0.0, acc_bad = 0.0;
            for (int p = 0; p < k; ++p) {
                const double xv = (double)bf16_to_f32(hA[(size_t)i * k + p]);
                acc += xv * (double)bf16_to_f32(hW[(size_t)jj * k + p]);
                acc_bad += xv * (double)bf16_to_f32(hW[(size_t)(jj ^ 1) * k + p]);
            }
            const uint16_t want = bits_of_bf16((float)acc);
            const uint16_t badw = bits_of_bf16((float)acc_bad);
            const uint16_t have = got[(size_t)i * n + jj];
            if (have != want) ++ndiff_good;
            if (have != badw) ++ndiff_bad;
            const double hv = bf16_to_f32(have), wv = bf16_to_f32(want);
            dot += hv * wv;
            nb += wv * wv;
            const double ulp = std::fabs(bf16_to_f32((uint16_t)(want + 1)) - bf16_to_f32(want));
            if (ulp > 0) worst_ulp = std::max(worst_ulp, std::fabs(hv - wv) / ulp);
        }
    }
    const double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    const bool ok = survivors == 0 && cos > 0.999999 && ndiff_bad > 0;
    std::printf("PROBE m=%d n=%d k=%d ndiff=%zu survivors=%zu wrongref_ndiff=%zu of %zu "
                "worst_ulp=%.2f cos_vs_double=%.9f %s\n",
                m, n, k, ndiff_good, survivors, ndiff_bad, got.size(), worst_ulp, cos,
                ok ? "PASS" : "FAIL");
    CKP(hipFree(dA));
    CKP(hipFree(dW));
    CKP(hipFree(dC));
    return ok ? 0 : 1;
}

int main() {
    std::printf("%s\n", engine::kernels::gemm_route_init().c_str());
    int rc = 0;
    rc |= one(8, 4096, 1024);      // q_proj at an 8-row wave (dec8's shape)
    rc |= one(8, 6144, 1024);      // in_proj_qkv at 8 rows
    rc |= one(295, 512, 1024);     // k_proj/v_proj at dec300f's wave width
    rc |= one(295, 3584, 1024);    // gate/up
    rc |= one(295, 1024, 3584);    // down
    rc |= one(512, 1024, 2048);    // o_proj/out_proj at the wave ceiling
    return rc;
}
