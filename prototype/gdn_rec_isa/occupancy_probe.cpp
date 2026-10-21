// occupancy_probe.cpp — how many wavefronts of the `gdn_rec_isa` spelling MODEs can be resident per CU?
//
// WHY THIS EXISTS, AND WHY IT IS NOT A DRIVER ASSUMPTION. The pointer-increment MODE (`recptr_kernel`)
// compiles with FEWER vector registers than the shipped looped form it is a spelling of (69 VGPR against
// 75, from `-Rpass-analysis=kernel-resource-usage`). A VGPR drop is a lever only if it changes what the
// runtime can keep resident — and this repo has been bitten by trusting the compiler's own remark: clang
// prints `Occupancy [waves/SIMD]: 16` for EVERY kernel in the tree, at 38,944 B of static LDS and at 0 B
// alike (AGENTS.md, the counter/occupancy trap row; `tools/kernel_occupancy_probe.cpp` says the same for
// the attention kernels). So the question is put to the runtime's own calculator,
// `hipOccupancyMaxActiveBlocksPerMultiprocessor`, which reads the code object's `.vgpr_count` and
// `.group_segment_fixed_size` against the device's per-CU budget and returns what would actually be
// scheduled.
//
// NOTHING IS LAUNCHED, ALLOCATED, OR TIMED. The query is host-side: no kernel runs, no buffer is allocated,
// no counter is read. It produces no number that belongs in `engine/performance-status.md` and no number
// that a contended window could corrupt — it is a census of the code object, the same class of thing as
// reading the `.s`, which is why a worker may run it (AGENTS.md: short, correctness-shaped, no timing).
//
// THE CANARY IS THE POINT. Two stubs bound the answer: a kernel whose static LDS is the whole 64 KiB budget
// must come back at 1 block/CU, and a kernel with no LDS at all must come back at least as high as the real
// ones. If the real kernels match the whole-budget stub, this probe is not seeing LDS/VGPR at all and every
// row it printed must be thrown away. A probe that cannot fail is the failure mode this repo keeps logging
// (`tools/mixed_bw`, `prototype/gdn_collapse/run.py`, `prototype/gdn_rec_isa/run.py`'s CASE_RE).
//
// Build and run (no device work is issued; the device is queried for its own facts):
//   hipcc -O3 --offload-arch=gfx1151 -I. -Iengine \
//         prototype/gdn_rec_isa/occupancy_probe.cpp -o tmp/gcc/gdn_addr/occupancy_probe
//   tmp/gcc/gdn_addr/occupancy_probe
#include <hip/hip_runtime.h>
#include <cstdio>

#include "prototype/gdn_rec_isa/gdn_rec_isa_kernels.h"

// `simd_per_cu` is 2 on gfx1151 per KFD (`simd_count 80` over 40 CUs). HIP's own `multiProcessorCount`
// reports a HALVED SM count on this SKU and its `maxThreadsPerMultiProcessor` is inconsistent with the
// architecture (AGENTS.md, counter/occupancy trap row), so waves per SIMD are computed from the KFD number
// and the C-side constant the kernels themselves use for the launch geometry, not from a HIP property.
static const int kSimdPerCu = 2;
static const int kThreadsPerBlock = D;          // every MODE launches <<<NH, D>>>

// The whole-LDS control. It must WRITE the array before reading it or LLVM deletes the object and the
// control passes vacuously — the exact mistake `tools/kernel_occupancy_probe.cpp` records. It is sized
// 64 KiB minus 1 KiB because the runtime reserves a page per block: a full-64 KiB request cannot fit any
// block at all and the query returns an error rather than 0, which is what the first run of this probe
// found and why its canary had to be re-derived rather than assumed.
__global__ void lds_whole_budget(int* o) {
    __shared__ char pad[65536 - 1024];
    for (int i = threadIdx.x; i < 65536 - 1024; i += blockDim.x) pad[i] = (char)i;
    __syncthreads();
    if (o) o[0] = pad[threadIdx.x];
}
// A half-budget control: if the probe can tell 32 KiB from 63 KiB, it is really reading static LDS.
__global__ void lds_half_budget(int* o) {
    __shared__ char pad[32768];
    for (int i = threadIdx.x; i < 32768; i += blockDim.x) pad[i] = (char)i;
    __syncthreads();
    if (o) o[0] = pad[threadIdx.x];
}
// The no-LDS control: the residency ceiling the real kernels must sit under.
__global__ void lds_none(int* o) { if (o) o[0] = 1; }

namespace {

int query(const void* k) {
    int nb = -1;
    const hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, const_cast<void*>(k),
                                                                    kThreadsPerBlock, 0);
    if (e != hipSuccess) {
        std::fprintf(stderr, "OCC query failed: %s\n", hipGetErrorString(e));
        return -1;
    }
    return nb;
}

struct Row { const char* name; const void* fn; };

}  // namespace

int main() {
    // Asking the device for its properties is what makes the runtime load the code objects; no kernel is
    // launched and nothing is allocated, which is what makes this legal for a worker to run (AGENTS.md).
    int dev = 0, ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) { std::fprintf(stderr, "no device\n"); return 1; }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { std::fprintf(stderr, "no props\n"); return 1; }

    const Row stubs[] = {{"stub_lds_63k", (const void*)lds_whole_budget},
                         {"stub_lds_32k", (const void*)lds_half_budget},
                         {"stub_lds_none", (const void*)lds_none}};
    const Row real[] = {{"rec_kernel", (const void*)rec_kernel},
                        {"rec2c4_kernel", (const void*)rec2c4_kernel},
                        {"recloop_kernel", (const void*)recloop_kernel},
                        {"recptr_kernel", (const void*)recptr_kernel}};

    std::printf("OCC device=%s threads_per_block=%d simd_per_cu=%d\n", prop.name,
                kThreadsPerBlock, kSimdPerCu);
    // HIP's own geometry, printed rather than hidden, because on this SKU it is 2x off the architecture
    // (AGENTS.md): a per-CU thread maximum of 2,048 is 64 wavefronts against a real maximum of 32, so any
    // row that lands on that ceiling is a GEOMETRY limit and says nothing about LDS or VGPRs.
    std::printf("OCC HIP geometry (2x-off on this SKU): multiProcessorCount=%d"
                " maxThreadsPerMultiProcessor=%d sharedMemPerMultiprocessor=%zu\n",
                prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
                prop.sharedMemPerMultiprocessor);
    int bad = 0;
    int nb_stub_full = -1, nb_stub_none = -1, nb_stub_half = -1;
    for (const auto& r : stubs) {
        const int nb = query(r.fn);
        std::printf("OCC %-18s blocks_per_CU=%d\n", r.name, nb);
        if (r.name[9] == '6') nb_stub_full = nb;
        else if (r.name[9] == '3') nb_stub_half = nb;
        else nb_stub_none = nb;
    }
    int worst_real = 0;
    for (const auto& r : real) {
        const int nb = query(r.fn);
        if (nb < 1) bad = 1;
        if (nb > worst_real) worst_real = nb;
        std::printf("OCC %-18s blocks_per_CU=%2d waves_per_CU=%2d waves_per_SIMD=%d\n",
                    r.name, nb, nb * (kThreadsPerBlock / 32), nb * (kThreadsPerBlock / 32) / kSimdPerCu);
    }
    if (nb_stub_full != 1) {
        std::printf("OCC CANARY FAIL: the 63 KiB-LDS stub returned %d, not 1 — the probe is not seeing"
                    " static LDS, so its rows are worthless\n", nb_stub_full);
        bad = 1;
    }
    if (nb_stub_half != 2 * nb_stub_full) {
        std::printf("OCC CANARY FAIL: the 32 KiB stub returned %d, not twice the 63 KiB stub (%d) — the"
                    " LDS reading is not linear, so residency numbers are not trustworthy\n",
                    nb_stub_half, nb_stub_full);
        bad = 1;
    }
    if (nb_stub_none < worst_real) {
        std::printf("OCC CANARY FAIL: the 0-LDS stub (%d) is below a real kernel (%d) — same verdict\n",
                    nb_stub_none, worst_real);
        bad = 1;
    }
    std::printf("OCC canary stubs 63k=%d 32k=%d none=%d real_max=%d ok=%d claim=these rows come from static"
                " LDS and VGPR pressure (the two LDS stubs are read correctly), not from a driver default\n",
                nb_stub_full, nb_stub_half, nb_stub_none, worst_real, bad ? 0 : 1);
    return bad;
}
