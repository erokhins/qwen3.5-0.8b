/*
 * tools/bw_probe_hip.cpp — GPU-side memory bandwidth probe for the engine's roofline.
 *
 * §07.4 turns bandwidth into the decode ceiling (tok/s = BW / bytes_per_token, and the
 * engine's byte count is ~1.565 GB/token bf16 for Qwen3.5-0.8B). The CPU-side probe
 * (`tools/bw_probe.py`) reaches ~117 GB/s here; this measures what the iGPU actually
 * pulls, which is the number the engine will be gated against. On Strix Halo the iGPU
 * and the CPU share the same LPDDR5x package, so the two probes compete for one pool —
 * run them with nothing else loading the machine, and never both at once.
 *
 * Build and run:
 *     hipcc -O2 -o /tmp/bw_probe_hip tools/bw_probe_hip.cpp && /tmp/bw_probe_hip
 *
 * What it measures, and why each one is here:
 *   read   — grid-stride sum over the whole buffer with 16 B loads. One add per 16 B
 *            cannot be compute-bound at any plausible bandwidth, so this is the number
 *            a decode step's weight streaming looks like. Reported as the max over a
 *            block-size sweep, because a small grid leaves the memory controller idle.
 *   copy   — device-to-device copy: pays read + write traffic, i.e. what a
 *            state-updating kernel (KV append, GDN state write) really costs.
 *   H2D / D2H — host↔device over the unified pool; a CPU->GPU staging path in the
 *            engine (e.g. uploading logits) is bounded by this, not by the read number.
 *
 * Allocations fall back to hipMallocManaged if the 2 GiB VRAM carve-out refuses, which
 * on this box it will for big buffers — it is the same physical LPDDR either way, but
 * record which one was used, because managed memory can behave differently at the first
 * touch.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                              \
    do {                                                                   \
        hipError_t e_ = (x);                                               \
        if (e_ != hipSuccess) {                                            \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_), \
                    __FILE__, __LINE__);                                   \
        }                                                                  \
    } while (0)

__global__ void read_kernel(const float4* __restrict__ p, size_t n4, float* out) {
    size_t stride = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (size_t i = stride; i < n4; i += step) {
        float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5678f) *out = acc;   // never true; keeps the loads alive
}

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                            size_t n4) {
    size_t stride = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    for (size_t i = stride; i < n4; i += step) dst[i] = src[i];
}

double time_ms(void (*launch)(void*, void*, size_t, int, int), void* a, void* b,
               size_t n4, int blocks, int threads, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        hipEvent_t s, e;
        CK(hipEventCreate(&s));
        CK(hipEventCreate(&e));
        CK(hipEventRecord(s));
        launch(a, b, n4, blocks, threads);
        CK(hipEventRecord(e));
        CK(hipEventSynchronize(e));
        float ms = 0.f;
        CK(hipEventElapsedTime(&ms, s, e));
        best = std::min(best, (double)ms);
        CK(hipEventDestroy(s));
        CK(hipEventDestroy(e));
    }
    return best;
}

static void launch_read(void* a, void*, size_t n4, int blocks, int threads) {
    hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (const float4*)a, n4, nullptr);
}
static void launch_copy(void* a, void* b, size_t n4, int blocks, int threads) {
    hipLaunchKernelGGL(copy_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (const float4*)a, (float4*)b, n4);
}

int main(int argc, char** argv) {
    double gib = (argc > 1) ? atof(argv[1]) : 1.0;      // buffer size per test
    int reps = (argc > 2) ? atoi(argv[2]) : 5;
    double bytes_per_token = (argc > 3) ? atof(argv[3]) : 1.565e9;

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device: %s | %d CUs | L2 %d KB | VRAM carve-out %.2f GiB%s\n", prop.name,
           prop.multiProcessorCount, prop.l2CacheSize / 1024,
           prop.totalGlobalMem / 1073741824.0,
           prop.unifiedAddressing ? " | unified addressing" : "");
    printf("buffer %.2f GiB per test, best of %d reps\n\n", gib, reps);

    size_t bytes = (size_t)(gib * 1073741824.0);
    size_t n4 = bytes / sizeof(float4);

    for (int try_managed = 0; try_managed < 2; ++try_managed) {
        void *a = nullptr, *b = nullptr;
        hipError_t r1, r2;
        if (try_managed) {
            r1 = hipMallocManaged(&a, bytes);
            r2 = hipMallocManaged(&b, bytes);
            if (r1 == hipSuccess && r2 == hipSuccess)
                printf("(using hipMallocManaged: the VRAM carve-out refused %.2f GiB)\n",
                       gib);
        } else {
            r1 = hipMalloc(&a, bytes);
            r2 = hipMalloc(&b, bytes);
        }
        if (r1 != hipSuccess || r2 != hipSuccess) {
            if (r1 != hipSuccess) hipGetLastError();
            if (r2 != hipSuccess) hipGetLastError();
            if (!try_managed) continue;
            fprintf(stderr, "cannot allocate %.2f GiB either way\n", gib);
            return 1;
        }
        CK(hipMemset(a, 0x3f, bytes));
        CK(hipMemset(b, 0x00, bytes));

        printf("  %-6s %-8s %10s %12s %12s\n", "blocks", "threads", "read GB/s",
               "copy GB/s", "tok/s cap");
        double best_read = 0, best_copy = 0;
        for (int waves : {1, 4, 8, 16, 32}) {
            int blocks = prop.multiProcessorCount * waves;
            for (int threads : {128, 256, 512, 1024}) {
                double tr = time_ms(launch_read, a, nullptr, n4, blocks, threads, reps);
                double tc = time_ms(launch_copy, a, b, n4, blocks, threads, reps);
                double rb = (double)bytes / (tr / 1e3) / 1e9;
                double cb = 2.0 * bytes / (tc / 1e3) / 1e9;
                best_read = std::max(best_read, rb);
                best_copy = std::max(best_copy, cb);
                printf("  %-6d %-8d %10.1f %12.1f %12.0f\n", blocks, threads, rb, cb,
                       rb * 1e9 / bytes_per_token);
            }
        }
        printf("\n  BEST read %.1f GB/s -> %.0f tok/s at %.3f GB/token"
               " | best copy %.1f GB/s\n", best_read,
               best_read / bytes_per_token * 1e9, bytes_per_token / 1e9, best_copy);

        // host<->device over the unified pool
        void* h = nullptr;
        CK(hipMallocHost(&h, bytes));
        double best_h2d = 0, best_d2h = 0;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            CK(hipMemcpy(a, h, bytes, hipMemcpyHostToDevice));
            auto t1 = std::chrono::steady_clock::now();
            CK(hipMemcpy(h, a, bytes, hipMemcpyDeviceToHost));
            auto t2 = std::chrono::steady_clock::now();
            best_h2d = std::max(best_h2d, bytes /
                std::chrono::duration<double>(t1 - t0).count() / 1e9);
            best_d2h = std::max(best_d2h, bytes /
                std::chrono::duration<double>(t2 - t1).count() / 1e9);
        }
        printf("  H2D %.1f GB/s | D2H %.1f GB/s (staging paths are bounded by this)\n",
               best_h2d, best_d2h);
        CK(hipFreeHost(h));
        CK(hipFree(a));
        CK(hipFree(b));
        break;
    }
    return 0;
}
