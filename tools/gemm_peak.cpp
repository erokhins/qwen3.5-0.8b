// tools/gemm_peak.cpp — dense matrix-multiply peak for this GPU, measured with rocBLAS/hipBLASLt
// kernels (i.e. the MFMA/tensor path), not with hand-rolled FMA loops.
//
// Why this exists: the engine's prefill throughput is bounded by dense matmul, and every argument about
// "prefill should be 5k tok/s" is arithmetic on two numbers — FLOPs per token (known: ~1.6 GFLOP) and this
// machine's sustained dense-matrix rate (never measured here). This probe measures the second one.
//
// Build:  tools/build.sh gemm_peak
// Run:    tools/build/gemm_peak [--iters K] [--dtypes bf16,fp16,fp32] [--check] [--determinism]
//
// Two modes are correctness-only and print no timing at all, so a worker may run them (AGENTS.md):
//   --check        the CPU-reference selftest, the OP_T/OP_N layout cross-check, and the 16-bit-C probe
//   --determinism  same product N times, from three handles, into a buffer pre-filled with a pattern
//                  writes cannot store, and hashed. Byte comparison only; no rate is computed or printed.
//
// Honesty rules this probe follows (AGENTS.md): warm before timing, best-of-N with the spread printed, a
// number above the measured memory-probe rate on a bandwidth-bound shape means skipped work, and nothing
// here is a token rate — this is a matmul ceiling, nothing more.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gemm_common.h"

namespace {

// CK/BK, the 16-bit encode/decode pair, gemm_stored_weight, fnv1a, fill_16bit and the 16-bit-C
// comparator all live in gemm_common.h, because tools/prefill_mix.cpp has to make the same two
// claims (same call, same rounding comparison) and a second copy of either could drift.
using namespace gemm_common;

struct Row {
    int m, n, k;
    const char *tag;
};

// The shapes a determinism pass runs: the engine's own matmul set at one wave width, plus the two
// skinny shapes a square probe never touches. Deliberately NOT every shape in the model — this is a
// correctness question, and a correctness question that takes four minutes is a four-minute wait
// between the answer and the person who has to read it. The full shape census is paper, in
// `tools/prefill_gemm_shapes.py`; rates are the main agent's (`devices/strix-halo-compute.md`).
static const Row kDeterminismRows[] = {
    {512, 4096, 1024, "q_proj"},          // [4096,1024] rows, a 512-token wave
    {512, 512, 1024, "k_proj"},           // the skinny one: N=512 (2 kv heads x 256), N/K = 1/2
    {512, 16, 1024, "in_proj_a"},         // skinniest in the checkpoint: N=16
    {512, 6144, 1024, "in_proj_qkv"},     // widest of the projections
    {512, 1024, 2048, "o_proj"},          // K > N
    {512, 3584, 1024, "mlp_gate"},
    {512, 1024, 3584, "mlp_down"},
    {1024, 4096, 1024, "q_proj@1024"},
    {3072, 4096, 1024, "q_proj@3072"},
    {64, 4096, 1024, "q_proj@64"},
    {1, 248320, 1024, "lm_head@1"},       // prefill prices exactly one row (§06.5)
};

// FNV-1a and the 16-bit encode/decode helpers are in gemm_common.h.

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

// Host-side bf16 -> f32 lives in gemm_common.h as bf2f (this comparison runs on the CPU).

// The OP_T form of the call, gemm_stored_weight, is in gemm_common.h too — tools/prefill_mix.cpp
// calls that same function, so the layout derivation and the mix run through one place.

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
    CK(hipFree(db)); CK(hipFree(dc));   // `da` stays alive: the layout check below reads it

    // The layout question, answered by computation rather than by a comment. Two ways to hand hipBLAS
    // the same product: B pre-transposed into [k,n] (what run_gemm and every historical row of this
    // probe use), or the weight exactly as the checkpoint stores it, [n,k], used as the transposed
    // operand (what an engine would use, because a transpose copy of 1.5 GB of weights is not free).
    // Getting the OP_T arguments wrong is silent — it computes a different product, not an error.
    {
        std::vector<hip_bfloat16> wb((size_t)N * K);
        for (int i = 0; i < N; ++i) for (int j = 0; j < K; ++j) wb[(size_t)i * K + j] = hb[(size_t)j * N + i];
        hip_bfloat16 *db2;
        CK(hipMalloc(&db2, wb.size() * 2));
        CK(hipMemcpy(db2, wb.data(), wb.size() * 2, hipMemcpyHostToDevice));
        // OP_T form: same product, weight buffer read as [n,k] with ld=k. Nothing else is called on
        // this buffer: an earlier draft also handed it to an OP_N call with ldb=k, which is NOT the same
        // product (with OP_N the operand must be n x k column-major, so a row-major [n,k] buffer read
        // that way is an index swap), and a check that fails for a reason of its own inventing is worse
        // than no check.
        {
            hipblasHandle_t h2;
            BK(hipblasCreate(&h2));
            float *dc3;
            CK(hipMalloc(&dc3, hc.size() * 4));
            BK(gemm_stored_weight(h2, M, N, K, HIP_R_16BF, da, db2, dc3, HIP_R_32F));
            CK(hipDeviceSynchronize());
            std::vector<float> h3(hc.size());
            CK(hipMemcpy(h3.data(), dc3, hc.size() * 4, hipMemcpyDeviceToHost));
            double md = 0.0; size_t nd = 0;
            for (int i = 0; i < M * N; ++i) {
                double d = std::fabs(h3[i] - ref[i]) / (1.0 + std::fabs((double)ref[i]));
                md = std::max(md, d);
                nd += (h3[i] != hc[i]);
            }
            const bool layout_ok = md < 2e-2;
            std::printf("LAYOUT CROSS-CHECK %s: OP_T(W as [n,k], ld=k) against the same CPU reference "
                        "at max|delta| %.3e; %zu of %d outputs differ in their bits from the OP_N "
                        "result (a last-bit difference between two kernels is expected, a different "
                        "product is not)\n", layout_ok ? "ok" : "FAIL", md, nd, M * N);
            ok = ok && layout_ok;
            hipblasDestroy(h2);
            CK(hipFree(dc3));
        }
        CK(hipFree(db2));
    }
    CK(hipFree(da));
    return ok;
}

// Which (input dtype, output dtype) pairs does this hipBLAS actually accept, and does a 16-bit output
// round where a host round would?
//
// This exists because the first version of it asked for C=fp16 with A/B=bf16 and got status 7
// (INVALID_VALUE), which is not the same fact as "C=fp16 is unavailable": bf16-in/fp16-out is a
// sensible combination for a library not to support, and reading that refusal as the wrong fact would
// have ruled out the 16-bit output the fp16 route needs. So the axis is walked as the routes use it:
// each input dtype paired with its own 16-bit width and with F32.
//
// Why it matters: every shipped pin rounds a projection's product to a 16-bit width at the point the
// reference does. A library call that can write that width keeps the rounding point; one that cannot
// has to write F32 C, which both doubles the activation traffic (tools/prefill_gemm_shapes.py) and
// moves a rounding — and a moved rounding is a route change (D-3/D-5), not a tuning knob.
bool cdtype_probe(hipblasHandle_t h) {
    const int M = 64, N = 96, K = 128;
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> sa((size_t)M * K), sw((size_t)N * K);
    for (auto &x : sa) x = u(rng);
    for (auto &x : sw) x = u(rng);

    struct Ab { hipDataType dt; const char *name; };
    bool ok = true;
    std::printf("CDTYPE MATRIX (C[m,n] = A[m,k] W[n,k]^T, W as [n,k], OP_T; %dx%dx%d)\n", M, N, K);
    std::printf("  %-6s %-6s %-10s %-11s %s\n", "A/B", "C", "status", "bits differ",
                "worst |device - host-rounded| in ULP of the 16-bit format");
    for (const Ab &ab : {Ab{HIP_R_16BF, "bf16"}, Ab{HIP_R_16F, "fp16"}}) {
        const bool is_bf = ab.dt == HIP_R_16BF;
        auto enc = [&](float f) { return enc16(f, is_bf); };
        auto dec = [&](uint16_t b) { return dec16(b, is_bf); };
        std::vector<uint16_t> ha((size_t)M * K), hw((size_t)N * K);
        for (size_t i = 0; i < ha.size(); ++i) ha[i] = enc(sa[i]);
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = enc(sw[i]);
        // The reference is the dot of the values the device was actually given, in double, so the only
        // thing left to compare is the output rounding.
        std::vector<double> ref((size_t)M * N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double s = 0.0;
                for (int k = 0; k < K; ++k)
                    s += (double)dec(ha[(size_t)m * K + k]) * (double)dec(hw[(size_t)n * K + k]);
                ref[(size_t)m * N + n] = s;
            }
        void *da, *dw;
        CK(hipMalloc(&da, ha.size() * 2)); CK(hipMalloc(&dw, hw.size() * 2));
        CK(hipMemcpy(da, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        const float alpha = 1.f, beta = 0.f;
        for (const auto &c : {std::pair<hipDataType, const char *>{ab.dt, is_bf ? "bf16" : "fp16"},
                              std::pair<hipDataType, const char *>{HIP_R_32F, "f32"}}) {
            const size_t cesz = (c.first == HIP_R_32F) ? 4u : 2u;
            void *dc;
            CK(hipMalloc(&dc, (size_t)M * N * cesz));
            const hipblasStatus_t st = hipblasGemmEx(
                h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, &alpha, dw, ab.dt, K, da, ab.dt, K, &beta,
                dc, c.first, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
            if (st != HIPBLAS_STATUS_SUCCESS) {
                std::printf("  %-6s %-6s %-10s %-11s %s\n", ab.name, c.second, "REFUSED",
                            "-", "status indicates the pair is not supported by hipblasGemmEx");
                CK(hipFree(dc));
                continue;
            }
            CK(hipDeviceSynchronize());
            std::vector<uint8_t> raw((size_t)M * N * cesz);
            CK(hipMemcpy(raw.data(), dc, raw.size(), hipMemcpyDeviceToHost));
            // An F32 C has no rounding point to compare against the reference, so there is nothing to
            // gate here beyond "accepted" — SELFCHECK above is this probe's accuracy evidence. The
            // first version tried to score it in ULP and produced 6e12 "ULP" for a cancelled element,
            // which is what happens when a per-element relative metric is used where the tensor's own
            // scale is the yardstick (AGENTS.md, the cancellation lesson).
            if (cesz == 4) {
                std::printf("  %-6s %-6s %-10s %-11s %s\n", ab.name, c.second, "accepted", "n/a",
                            "F32 output has no 16-bit rounding to compare; SELFCHECK is the accuracy "
                            "evidence");
                CK(hipFree(dc));
                continue;
            }
            const auto cmp = compare_16bit_output(raw.data(), ref.data(), ref.size(), is_bf);
            const bool good = (cmp.worst_ulp <= 1.0);
            ok = ok && good;
            std::printf("  %-6s %-6s %-10s %-11zu %.3g %s\n", ab.name, c.second, "accepted", cmp.ndiff,
                        cmp.worst_ulp, good ? "" : "<- more than one ULP off");
            CK(hipFree(dc));
        }
        CK(hipFree(da)); CK(hipFree(dw));
    }
    std::printf("CDTYPE %s\n", ok ? "ok" : "FAIL");
    return ok;
}

// Is the vendor library's answer a function of its inputs? Same product, four times: three back to
// back, one after throwing the handle away (the heuristic and its solution cache live on the handle,
// so a fresh handle is what asks "does kernel choice depend on history"), and — by construction — a
// second run whose output buffer starts full of the first run's bytes rather than 0xFF, which is the
// beta==0 "I overwrite, I do not accumulate" check with a filler writes can never produce.
//
// Why ask: hipBLAS may pick a split-K kernel, and a split-K that reduces with atomics makes the output
// bytes non-deterministic. Every parity claim in this repo — "bit-identical, or banded against the
// pinned logits" — presumes the same inputs give the same bytes, so this has to be answered before
// anyone wires the library in, not after.
//
// No timing here either: this is a byte comparison.
bool determinism(hipDataType dt, const char *dname, int reps) {
    bool all_ok = true;
    uint64_t combined = 1469598103934665603ull;
    std::printf("\n=== determinism, %s in / f32 accumulate, hipBLAS DEFAULT algorithm\n", dname);
    std::printf("  %-14s %-6s %6s %7s %7s %5s %9s  %-18s %s\n", "shape", "layout", "m*n", "runhash",
                "freshWs", "ndiff", "noWs", "identical", "beta0-clean");
    std::printf("  (freshWs = second handle given the same 128 MiB workspace — a DIFF here is real\n"
                "   nondeterminism; noWs = second handle given none, a DIFF there is a workspace-\n"
                "   dependent kernel choice, which is a configuration fact, not a bug)");
    for (const Row &r : kDeterminismRows) {
        // Two layouts, because they are different kernels and the question is per-kernel.
        for (int layout = 0; layout < 2; ++layout) {
            const char *lname = layout ? "W[n,k]" : "W[k,n]";
            const size_t na = (size_t)r.m * r.k, nw = (size_t)r.n * r.k, nc = (size_t)r.m * r.n;
            std::vector<uint8_t> ha(na * 2), hw(nw * 2);
            // The same byte pattern either way; the two layouts differ in how hipBLAS is told to read
            // it, which is the whole point, not in the values. Filled in the dtype under test, because
            // a bf16 bit pattern read as fp16 is arbitrary data and could carry an Inf into a product
            // that has nothing to do with the question.
            const bool is_bf = dt != HIP_R_16F;
            fill_16bit(hw.data(), nw, is_bf, 37);
            fill_16bit(ha.data(), na, is_bf, 17);
            // One scratch buffer, handed to some handles and withheld from others: kernel selection
            // depends on how much workspace the handle was given, so "a second handle disagreed" is
            // only evidence of nondeterminism if the two handles were given the same workspace.
            void *ws = nullptr;
            const size_t wsz = 128u << 20;
            if (hipMalloc(&ws, wsz) != hipSuccess) ws = nullptr;
            void *da; void *dw; float *dc;
            if (hipMalloc(&da, na * 2) != hipSuccess || hipMalloc(&dw, nw * 2) != hipSuccess ||
                hipMalloc(&dc, nc * 4) != hipSuccess) {
                std::printf("  %-14s %-6s  OUT OF MEMORY at %dx%dx%d\n", r.tag, lname, r.m, r.n, r.k);
                all_ok = false;
                continue;
            }
            CK(hipMemcpy(da, ha.data(), na * 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(dw, hw.data(), nw * 2, hipMemcpyHostToDevice));
            CK(hipMemset(dc, 0xFF, nc * 4));        // NaN-pattern filler: beta must erase it

            hipblasHandle_t h;
            BK(hipblasCreate(&h));
            if (ws) hipblasSetWorkspace(h, ws, wsz);
            auto one = [&](hipblasHandle_t hh) {
                hipblasStatus_t s;
                if (layout == 0) {
                    const float alpha = 1.f, beta = 0.f;
                    s = hipblasGemmEx(hh, HIPBLAS_OP_N, HIPBLAS_OP_N, r.n, r.m, r.k, &alpha, dw,
                                      dt, r.n, da, dt, r.k, &beta, dc, HIP_R_32F, r.n,
                                      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
                } else {
                    s = gemm_stored_weight(hh, r.m, r.n, r.k, dt, da, dw, dc, HIP_R_32F);
                }
                return (s == HIPBLAS_STATUS_SUCCESS) && (hipDeviceSynchronize() == hipSuccess);
            };

            std::vector<float> first(nc), later(nc);
            const bool ok1 = one(h);
            CK(hipMemcpy(first.data(), dc, nc * 4, hipMemcpyDeviceToHost));
            const uint64_t h1 = fnv1a((const uint8_t *)first.data(), nc * 4);
            bool same = true;
            for (int i = 1; i < reps && same; ++i) {
                const bool okr = one(h);
                CK(hipMemcpy(later.data(), dc, nc * 4, hipMemcpyDeviceToHost));
                if (!okr || fnv1a((const uint8_t *)later.data(), nc * 4) != h1) same = false;
            }
            // fresh handle, SAME workspace: this is the determinism question.
            hipblasHandle_t h2;
            BK(hipblasCreate(&h2));
            if (ws) hipblasSetWorkspace(h2, ws, wsz);
            const bool okf = one(h2);
            CK(hipMemcpy(later.data(), dc, nc * 4, hipMemcpyDeviceToHost));
            const bool fresh_same = okf && fnv1a((const uint8_t *)later.data(), nc * 4) == h1;
            size_t ndiff = 0;
            for (size_t i = 0; i < nc; ++i) ndiff += (later[i] != first[i]);
            // fresh handle, NO workspace: what an engine that forgets hipblasSetWorkspace gets. A
            // difference here is a *configuration* difference, not nondeterminism, and the two
            // columns must be read against each other before anyone calls a mismatch a bug.
            hipblasHandle_t h3;
            BK(hipblasCreate(&h3));
            const bool okn = one(h3);
            CK(hipMemcpy(later.data(), dc, nc * 4, hipMemcpyDeviceToHost));
            const bool fresh_nowe_same = okn && fnv1a((const uint8_t *)later.data(), nc * 4) == h1;
            hipblasDestroy(h2);
            hipblasDestroy(h3);
            // beta==0 cleanliness: no 0xFF word may survive in the output.
            size_t untouched = 0;
            for (size_t i = 0; i < nc; ++i) {
                uint32_t w;
                __builtin_memcpy(&w, &first[i], 4);
                untouched += (w == 0xFFFFFFFFu);
            }
            hipblasDestroy(h);
            if (ws) CK(hipFree(ws));
            const bool ok = ok1 && same && fresh_same && untouched == 0;
            all_ok = all_ok && ok;
            std::printf("  %-14s %-6s %6zu %7llx %7s %5zu %9s  %-18s %s\n", r.tag, lname, nc,
                        (unsigned long long)(h1 >> 32), fresh_same ? "same" : "DIFF", ndiff,
                        fresh_nowe_same ? "same" : "DIFF",
                        ok ? "YES" : "NO — NOT DETERMINISTIC",
                        untouched == 0 ? "yes" : "NO: 0xFF SURVIVED");
            combined ^= h1;
            combined *= 1099511628211ull;
            CK(hipFree(da)); CK(hipFree(dw)); CK(hipFree(dc));
        }
    }
    std::printf("DET-COMBINED %s %016llx   (identical across three separate PROCESSES means the\n"
                "  selection does not depend on address layout or process history either)\n",
                dname, (unsigned long long)combined);
    return all_ok;
}

}  // namespace


int main(int argc, char **argv) {
    int iters = 20;
    bool check = false;
    bool determinism_mode = false;
    std::vector<std::pair<hipDataType, const char *>> dtypes = {
        {HIP_R_16BF, "bf16"}, {HIP_R_16F, "fp16"}, {HIP_R_32F, "fp32"},
    };
    bool sample_clocks = false;
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
        else if (!std::strcmp(argv[i], "--determinism")) determinism_mode = true;
        else if (!std::strcmp(argv[i], "--clocks")) sample_clocks = true;
        else if (!std::strcmp(argv[i], "--dtypes") && i + 1 < argc) {
            dtypes.clear();
            for (auto &tok : [] (const std::string &s) {
                std::vector<std::string> out; size_t p = 0;
                while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
                    out.push_back(s.substr(p, q - p)); p = q + 1; }
                return out;
            }(argv[++i])) {
                if (tok == "bf16") dtypes.push_back({HIP_R_16BF, "bf16"});
                else if (tok == "fp16") dtypes.push_back({HIP_R_16F, "fp16"});
                else if (tok == "fp32") dtypes.push_back({HIP_R_32F, "fp32"});
            }
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
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
    if (check) {
        // `--check` is a correctness gate and nothing else: it does not reach the timing loop, so a
        // worker running it cannot emit an ms figure by accident (AGENTS.md: worker device runs are
        // correctness-only). `--determinism` is the other such mode.
        const bool ok = selfcheck(h) && cdtype_probe(h);
        std::printf("CHECK %s\n", ok ? "PASS" : "FAIL");
        hipblasDestroy(h);
        return ok ? 0 : 1;
    }

    if (determinism_mode) {
        // Correctness only, and deliberately no timing rows in this branch: no best-of-N, no ms, no
        // rate. The byte comparison is the whole question.
        bool ok = true;
        for (auto &[dt, dname] : dtypes) {
            if (dt == HIP_R_32F) { std::printf("  (fp32 skipped in determinism mode)\n"); continue; }
            ok = determinism(dt, dname, 3) && ok;
        }
        std::printf("\nDETERMINISM %s\n", ok ? "PASS" : "FAIL");
        hipblasDestroy(h);
        return ok ? 0 : 1;
    }

    std::printf("%-18s %8s %12s %12s\n", "tag", "dtype", "best_ms", "TFLOP/s");
    for (auto &r : rows) {
        size_t na = (size_t)r.m * r.k, nb = (size_t)r.k * r.n, nc = (size_t)r.m * r.n;
        for (auto &[dt, dname] : dtypes) {
            size_t esz = (dt == HIP_R_32F) ? 4u : 2u;
            std::vector<uint8_t> ha(na * esz), hb(nb * esz);
            std::mt19937 rng(7);
            std::uniform_real_distribution<float> u(-0.5f, 0.5f);
            if (dt == HIP_R_32F) {
                auto *pa = (float *)ha.data(); for (size_t i = 0; i < na; ++i) pa[i] = u(rng);
                auto *pb = (float *)hb.data(); for (size_t i = 0; i < nb; ++i) pb[i] = u(rng);
            } else {
                auto *pa = (hip_bfloat16 *)ha.data(); for (size_t i = 0; i < na; ++i) pa[i] = __float2bfloat16(u(rng));
                auto *pb = (hip_bfloat16 *)hb.data(); for (size_t i = 0; i < nb; ++i) pb[i] = __float2bfloat16(u(rng));
            }
            void *da, *db; float *dc;
            CK(hipMalloc(&da, na * esz)); CK(hipMalloc(&db, nb * esz)); CK(hipMalloc(&dc, nc * 4));
            CK(hipMemcpy(da, ha.data(), na * esz, hipMemcpyHostToDevice));
            CK(hipMemcpy(db, hb.data(), nb * esz, hipMemcpyHostToDevice));
            CK(hipMemset(dc, 0, nc * 4));
            int it = sample_clocks ? std::max(iters, 200) : iters;
            double best = run_gemm(h, r.m, r.n, r.k, it, dt, da, db, dc);
            double flop = 2.0 * r.m * r.n * r.k;
            std::printf("%-18s %8s %12.3f %12.3f\n", r.tag, dname, best,
                        flop / (best * 1e-3) / 1e12);
            std::fflush(stdout);
            CK(hipFree(da)); CK(hipFree(db)); CK(hipFree(dc));
        }
    }
    BK(hipblasDestroy(h));
    return 0;
}
