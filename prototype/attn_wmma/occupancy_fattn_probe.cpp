// occupancy_fattn_probe.cpp — driver-authority occupancy for `m16_fattn_score` (D62, §4.2 claim (i)).
//
// WHY THIS EXISTS AND WHAT IT IS NOT
// Claim (i) is ">= 4 blocks/CU and the wavefronts that implies", from `-Rpass-analysis=kernel-resource-
// usage` VGPR/LDS plus an occupancy rule validated against the shipped `attn_slim_wave` row
// (47 VGPR / 12,288 B / 5 blocks / 20 wavefronts/CU, prototype/attn_wmma/evidence/occupancy-m16.txt).
// A rule is not enough here, and the reason is the finding: every recorded row in that file is LDS-bound,
// so none of them separates the two candidate register-file figures — the retired 512-lane-VGPR-per-SIMD
// budget used by `m16_census.py:325` and the driver's `MaxRegistersPerMultiprocessor = 196,608` that
// progress/2026-10-18/llama-fattn-anatomy.md §3.4 declares authoritative. `m16_fattn_score` is the first
// form whose register term is the one that could bind (193 VGPR against a 16,384 B LDS term that allows
// exactly 4), so the claim has to be read from the driver, not from a rule the evidence cannot pin down.
// §3.4's own words: driver authority is `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
//
// WHAT IT DOES: one `hipOccupancyMaxActiveBlocksPerMultiprocessor` per kernel, no kernel launched, no
// timing of any kind. `tools/kernel_occupancy_probe.cpp` is the precedent that this is a device FACT
// query and not a measurement ("one hipOccupancyMaxActiveBlocksPerMultiprocessor call, which is host-side
// driver arithmetic and does not launch a kernel"). It does create a device context, so it is run under
// `tools/gpu_lock.sh` and its wall time and `util` are reported by whoever runs it.
//
// THE CANARY, same shape as `evidence/occupancy-m16.txt`'s: two stubs that must return different answers
// (a 65,536-B LDS stub must give 1, a 0-B LDS stub must give 16 at 128 threads) and a stub carrying the
// shipped kernel's LDS figure, which must return 5 if this query reproduces the recorded shipped row. If
// the stubs agree with each other, or the 12,288-B stub does not print 5, the numbers below are worthless.
//
// BUILD + RUN: `bash prototype/attn_wmma/build_occupancy_fattn.sh` (prints the table itself).

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>

#define THREADS 128  // NWARPS = 4 x 32, the shipped block shape (prototype/.../wmma_fattn_score.cpp)

#define CK(x)                                                        \
    do {                                                             \
        const hipError_t e_ = (x);                                   \
        if (e_ != hipSuccess) {                                      \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            return 1;                                                \
        }                                                            \
    } while (0)

// The three kernels under test, defined in `wmma_fattn_score.cpp`.
extern "C" __global__ void m16_fattn_empty(const hip_bfloat16*, const hip_bfloat16*,
                                           const hip_bfloat16*, float*);
extern "C" __global__ void m16_fattn_scoreleg(const hip_bfloat16*, const hip_bfloat16*,
                                              const hip_bfloat16*, float*);
extern "C" __global__ void m16_fattn_score(const hip_bfloat16*, const hip_bfloat16*,
                                           const hip_bfloat16*, float*);

// The controls. `lds_size` is static shared, so the driver sees it without any launch.
__global__ void stub_lds_0(float* o) { o[threadIdx.x] = 1.f; }
__global__ void stub_lds_12288(float* o) {
    __shared__ float s[12288 / 4];   // the shipped attn_slim_wave LDS figure
    s[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();
    o[threadIdx.x] = s[threadIdx.x];
}
__global__ void stub_lds_65536(float* o) {
    __shared__ float s[65536 / 4];   // the whole CU budget: must return 1 block
    s[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();
    o[threadIdx.x] = s[threadIdx.x];
}

namespace {
struct Row {
    const char* name;
    const void* fn;
    int threads;
    int lds;   // for the printed arithmetic only; the driver reads the real figure itself
    int vgpr;  // ditto, from the listing's note block
};

int report(const Row& r) {
    int blocks = 0;
    // dynamicLDS is 0 on purpose: every byte these kernels use is STATIC shared, declared in the kernel.
    // Passing r.lds here as well charges it twice, and the first run of this probe did exactly that — it
    // read 2 blocks for the 12,288 B control instead of the recorded 5, which looks like a machine fact
    // and is an argument error.
    const hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, r.fn, r.threads, 0);
    if (e != hipSuccess) {
        std::printf("%-26s QUERY FAILED: %s\n", r.name, hipGetErrorString(e));
        return 1;
    }
    // The wavefronts figure uses the KFD's 32 wavefronts per real CU (simd_per_cu 2 x 16 waves), not
    // HIP's halved view (AGENTS.md: HIP reports against a halved SM count on this SKU).
    std::printf("%-26s %5d %9d %9d %9d %7d   %s\n", r.name, r.threads, r.vgpr, r.lds, blocks,
                blocks * (r.threads / 32), blocks >= 4 ? "" : "< 4 blocks/CU: claim (i) fails");
    return 0;
}
}  // namespace

int main() {
    CK(hipSetDevice(0));
    hipDeviceProp_t p{};
    CK(hipGetDeviceProperties(&p, 0));
    // Printed as the arithmetic's inputs, halved-SM-count warning included (AGENTS.md counter trap).
    std::printf("device                          %s\n", p.name);
    std::printf("  hip multiProcessorCount       %d  (halved on this SKU; KFD says 40 CUs)\n",
                p.multiProcessorCount);
    std::printf("  hip sharedMemPerMultiprocessor %d B\n", (int)p.sharedMemPerMultiprocessor);
    std::printf("  hip MaxRegistersPerMultiprocessor %d lane-VGPRs\n", p.regsPerMultiprocessor);
    std::printf("  hip MaxThreadsPerMultiprocessor   %d\n", p.maxThreadsPerMultiProcessor);
    std::printf("  hip MaxThreadsPerBlock          %d\n", p.maxThreadsPerBlock);
    std::printf("  sharedMemPerBlock               %d B\n", (int)p.sharedMemPerBlock);
    std::printf("\nNo kernel is launched and no timing is taken: this is a driver query only.\n\n");
    std::printf("%-26s %5s %9s %9s %9s %7s\n", "kernel", "thread", "vgpr", "lds B", "blk/CU", "wf/CU");

    const Row rows[] = {
        {"control: 0-B LDS stub", (const void*)&stub_lds_0, THREADS, 0, 0},
        {"control: 12288-B stub", (const void*)&stub_lds_12288, THREADS, 12288, 47},
        {"control: 65536-B stub", (const void*)&stub_lds_65536, THREADS, 65536, 0},
        {"m16_fattn_empty", (const void*)&m16_fattn_empty, THREADS, 16384, 21},
        {"m16_fattn_scoreleg", (const void*)&m16_fattn_scoreleg, THREADS, 16384, 31},
        {"m16_fattn_score", (const void*)&m16_fattn_score, THREADS, 16384, 193},
    };
    int bad = 0;
    for (const Row&r : rows) bad += report(r);

    // CANARY: the query must be able to tell 64 KiB from 0, and must reproduce the recorded shipped row.
    // The stubs' LDS is static, so dynamicLDS stays 0 here too.
    int b0 = 0, b64 = 0, b12 = 0;
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&b0, (const void*)&stub_lds_0, THREADS, 0));
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&b64, (const void*)&stub_lds_65536, THREADS, 0));
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&b12, (const void*)&stub_lds_12288, THREADS, 0));
    std::printf("\nCANARY\n");
    std::printf("%s: 65,536-B stub returns 1 (got %d)\n", b64 == 1 ? "PASS" : "FAIL", b64);
    std::printf("%s: 0-B stub returns 16 (got %d)\n", b0 == 16 ? "PASS" : "FAIL", b0);
    std::printf("%s: 12,288-B stub returns 5, the recorded attn_slim_wave row (got %d)\n",
                b12 == 5 ? "PASS" : "FAIL", b12);
    return bad || (b64 != 1) || (b0 != 16) || (b12 != 5);
}
