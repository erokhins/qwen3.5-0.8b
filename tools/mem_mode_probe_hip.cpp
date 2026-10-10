/*
 * tools/mem_mode_probe_hip.cpp — which allocation mode puts device-visible memory where,
 * and does it cost bandwidth? Written because the engine's weights (1.55 GB today, ~70 GB
 * for the M4 MoE target) must not depend on a VRAM carve-out.
 *
 * On this box the iGPU has no local memory: the "VRAM" the driver reports is a *reservation
 * out of LPDDR5x* (2 GiB by default) and everything else is reached through the GTT
 * (Graphics Translation Table), i.e. ordinary system RAM with GPU page tables. hipMalloc
 * silently over-commits the carve-out and hands GTT pages back, which is why `hipDeviceProp
 * erties.totalGlobalMem` says ~110 GiB while `rocm-smi --showmeminfo vram` says 2 GiB. This
 * probe measures each mode and shows, from sysfs, where the pages actually came from.
 *
 * Build and run:
 *     hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/mem_mode_probe tools/mem_mode_probe_hip.cpp
 *     /tmp/mem_mode_probe [size_GiB] [reps]
 *
 * Modes measured, all with the same grid-stride read kernel as tools/bw_probe_hip.cpp:
 *   carve  hipMalloc of a small buffer that fits the 2 GiB carve-out
 *   gtt    hipMalloc larger than the carve-out (the default path for our weights)
 *   man    hipMallocManaged, reported twice: first touch (page migration on demand) and
 *          best-of-N after it is resident — the gap is the trap this doc exists to warn about
 *   pin    hipHostMalloc, i.e. pinned host memory touched from the kernel through unified
 *          addressing (zero-copy); should be worst, and is the pattern to avoid for weights
 *
 * sysfs counters (mem_info_vram_used / mem_info_gtt_used) are sampled around each allocation
 * so the placement claim is measured, not assumed.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CK(x)                                                                \
    do {                                                                     \
        hipError_t e_ = (x);                                                 \
        if (e_ != hipSuccess) {                                              \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_), \
                    __FILE__, __LINE__);                                     \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void read_kernel(const float4* __restrict__ p, size_t n4, float* out) {
    size_t stride = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (size_t i = stride; i < n4; i += step) {
        float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5678f) *out = acc;
}

static long read_sysfs(const char* pat) {
    for (int i = 0; i < 8; ++i) {
        std::string p = "/sys/class/drm/card" + std::to_string(i) + "/device/" + pat;
        std::ifstream f(p);
        long v = -1;
        if (f.good() && (f >> v) && v >= 0) return v;
    }
    return -1;
}

int main(int argc, char** argv) {
    double gib = (argc > 1) ? atof(argv[1]) : 1.0;
    int reps = (argc > 2) ? atoi(argv[2]) : 5;
    double bytes_per_token = (argc > 3) ? atof(argv[3]) : 1.565e9;
    size_t bytes = (size_t)(gib * 1073741824.0);
    size_t n4 = bytes / sizeof(float4);

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device %s | %d CUs | visible device memory %.1f GiB\n", prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    printf("carve-out total %.2f GiB | GTT total %.2f GiB\n\n",
           read_sysfs("mem_info_vram_total") / 1073741824.0,
           read_sysfs("mem_info_gtt_total") / 1073741824.0);
    printf("%-6s %10s %10s %12s %12s %10s\n", "mode", "vram+GiB", "gtt+GiB",
           "read GB/s", "1st-run GB/s", "tok/s cap");

    const int blocks = 640, threads = 512;
    const size_t GiB1 = 1073741824ull;

    // A real model is not one 40 GiB tensor; it is thousands of them. So the single-
    // allocation ceiling found above (hipMalloc fails somewhere in 32..40 GiB) is not the
    // engine's ceiling. `many` allocates buffers until `--many` GiB is resident and reads
    // each one, which is the pattern of per-layer weight tensors.
    if (argc > 4 && std::string(argv[4]) == "many") {
        int nbuf = (argc > 5) ? atoi(argv[5]) : 8;
        std::vector<void*> ps;
        long v0 = read_sysfs("mem_info_vram_used"), g0 = read_sysfs("mem_info_gtt_used");
        for (int i = 0; i < nbuf; ++i) {
            void* q = nullptr;
            if (hipMalloc(&q, bytes) != hipSuccess) {
                printf("many: hipMalloc #%d of %d (%.1f GiB each) FAILED\n", i + 1, nbuf,
                       gib);
                break;
            }
            ps.push_back(q);
        }
        long v1 = read_sysfs("mem_info_vram_used"), g1 = read_sysfs("mem_info_gtt_used");
        printf("many: %d x %.1f GiB resident (vram +%.2f GiB, gtt +%.2f GiB)\n",
               (int)ps.size(), gib, (v1 - v0) / 1073741824.0, (g1 - g0) / 1073741824.0);
        double sum = 0;
        for (size_t i = 0; i < ps.size(); ++i) {
            hipEvent_t s, e;
            hipEventCreate(&s);
            hipEventCreate(&e);
            hipEventRecord(s);
            read_kernel<<<blocks, threads>>>((const float4*)ps[i], n4, nullptr);
            hipEventRecord(e);
            hipEventSynchronize(e);
            float ms = 0.f;
            hipEventElapsedTime(&ms, s, e);
            double gb = bytes / (ms / 1e3) / 1e9;
            sum += gb;
            printf("  buf %2d: %6.1f GB/s\n", (int)i, gb);
            hipEventDestroy(s);
            hipEventDestroy(e);
        }
        if (!ps.empty()) printf("  mean %.1f GB/s over %.1f GiB resident\n",
                                sum / ps.size(), gib * ps.size());
        for (void* q : ps) hipFree(q);
        return 0;
    }

    struct Row { const char* name; int kind; };
    // kind: 0 hipMalloc, 1 managed, 2 host-pinned
    for (Row r : {Row{"carve", 0}, Row{"gtt", 0}, Row{"man", 1}, Row{"pin", 2}}) {
        // `carve` deliberately uses a small buffer so it can land inside the reservation;
        // the others use the requested size and therefore spill into GTT.
        size_t mb = bytes;
        if (std::string(r.name) == "carve") mb = std::min<size_t>(bytes, GiB1);
        size_t mn4 = mb / sizeof(float4);
        long v0 = read_sysfs("mem_info_vram_used"), g0 = read_sysfs("mem_info_gtt_used");
        void* p = nullptr;
        if (r.kind == 0) CK(hipMalloc(&p, mb));
        else if (r.kind == 1) CK(hipMallocManaged(&p, mb));
        else CK(hipHostMalloc(&p, mb, hipHostMallocMapped));
        long v1 = read_sysfs("mem_info_vram_used"), g1 = read_sysfs("mem_info_gtt_used");

        // CPU-side fill of the same pages: this is the weight-loading path. If the engine
        // can write the shard bytes straight into the device-visible buffer (managed) it
        // skips a 76-82 GB/s hipMemcpy staging copy entirely - but only if this is fast.
        double fill_gbs = 0.0;
        if (r.kind == 1 || r.kind == 2) {
            auto f0 = std::chrono::steady_clock::now();
            volatile uint64_t* q = (volatile uint64_t*)p;
            for (size_t i = 0; i < mb / 8; ++i) q[i] = 0x3f803f803f803f80ull;
            double fs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - f0).count();
            fill_gbs = mb / fs / 1e9;
        }

        // First touch: for managed memory this is where pages migrate, so time it alone.
        auto t0 = std::chrono::steady_clock::now();
        read_kernel<<<blocks, threads>>>((const float4*)p, mn4, nullptr);
        CK(hipDeviceSynchronize());
        double first = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();

        double best = 1e30;
        for (int i = 0; i < reps; ++i) {
            hipEvent_t s, e;
            CK(hipEventCreate(&s));
            CK(hipEventCreate(&e));
            CK(hipEventRecord(s));
            read_kernel<<<blocks, threads>>>((const float4*)p, mn4, nullptr);
            CK(hipEventRecord(e));
            CK(hipEventSynchronize(e));
            float ms = 0.f;
            CK(hipEventElapsedTime(&ms, s, e));
            best = std::min(best, (double)ms);
            CK(hipEventDestroy(s));
            CK(hipEventDestroy(e));
        }
        double rb = mb / (best / 1e3) / 1e9;
        printf("%-6s %10.2f %10.2f %12.1f %12.1f %10.0f   (%.2f GiB buf, cpu fill %.1f GB/s)\n",
               r.name, (v1 - v0) / 1073741824.0, (g1 - g0) / 1073741824.0, rb,
               mb / first / 1e9, rb * 1e9 / bytes_per_token, mb / 1073741824.0, fill_gbs);
        if (r.kind == 2) CK(hipHostFree(p));
        else CK(hipFree(p));
    }
        printf("\n`carve` deliberately allocates <= 1 GiB to try to stay inside the reservation;\n");
    printf("`gtt`/`man`/`pin` use the requested %.1f GiB. The vram+/gtt+ columns are sysfs\n", gib);
    printf("counters sampled around the allocation, so placement is measured, not assumed.\n");
    printf("tok/s cap = read BW / %.3f GB per decode token (spec 01.7).\n",
           bytes_per_token / 1e9);
    return 0;
}
