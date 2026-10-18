// attn_occupancy.cpp — how many wavefronts of the SHIPPED attention kernels can be resident per CU?
//
// WHY THIS IS NOT A DEVICE MEASUREMENT: it launches no kernel, allocates nothing and times nothing.
// `hipOccupancyMaxActiveBlocksPerMultiprocessor` is a host-side calculator — it reads the code
// object's own `.vgpr_count` / `.group_segment_fixed_size` and the device's LDS-per-CU and returns
// what the runtime would schedule. That is the number the realized-occupancy discipline in
// `prototype/gdn_collapse/results.md` §4-§5 asks for ("price cycles-per-dependency at realised
// occupancy or do not price it"), and it has been the one occupancy input this repo never had:
// clang's `Occupancy [waves/SIMD]` remark prints 16 for EVERY kernel in the tree — attn_core at
// 38,944 B/block and attn_combine at 0 B both print 16 — so it is the machine maximum, not a
// per-kernel result, and `prototype/attn_valu_slim/results.md` §"Lever 0" citing "occupancy
// unchanged at 16 waves/SIMD" is citing a constant.
//
// It compiles the SAME headers `engine/kernels_attn.cpp` compiles, so the kernels measured here are
// the shipping ones, not look-alikes. Nothing under engine/ is modified; this TU only includes them.
//
// CANARY: the calculator must return >= 1 for every kernel and must return exactly 1 for a stub
// whose static LDS is the whole 64 KiB budget. If both attn kernels come back at the same value as
// the 0-LDS stub, this probe is not seeing static LDS and its numbers must be thrown away.
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <string>
#include <cstdlib>

#include "prototype/attn_decode/attn_decode_kernels.h"
#include "engine/attn_slim_kernels.h"

// The M = 16 WMMA probe variants, linked in from `prototype/attn_wmma/wmma_m16_probe.cpp`, which has
// no `main()` by design precisely so it can be linked into this TU. The query needs only the function
// symbols: `hipOccupancyMaxActiveBlocksPerMultiprocessor` reads the code object's `.vgpr_count` and
// `.group_segment_fixed_size`, which the compiler fixed at compile time — nothing here launches.
// They are asked about because D57's item-2.6 gate turned on whether a 16-row tile can hold its slot
// win AND stay resident; a slot count from a 1-block/CU kernel is not the same currency as the shipped
// 45 (`prototype/gdn_collapse/results.md` §4–§5 is that lesson).
extern "C" __global__ void m16_empty(const hip_bfloat16 *, const hip_bfloat16 *, float *);
extern "C" __global__ void m16_bf16rt(const hip_bfloat16 *, const hip_bfloat16 *, float *);
extern "C" __global__ void m16_roll(const hip_bfloat16 *, const hip_bfloat16 *, float *);
extern "C" __global__ void m16_bf16gq(const hip_bfloat16 *, const hip_bfloat16 *, float *);
extern "C" __global__ void m16_scalar_ob(const hip_bfloat16 *, const hip_bfloat16 *, float *);

// A stub with the whole LDS budget, as the probe's own positive control. It must WRITE the array
// before reading it: the first version read uninitialised shared memory and LLVM deleted the whole
// LDS object, so the control returned 16 blocks/CU and the canary correctly refused the result —
// a control that passes vacuously is exactly the failure this repo keeps logging.
__global__ void lds_whole_budget(int *o) {
    __shared__ char pad[65536];
    for (int i = threadIdx.x; i < 65536; i += blockDim.x) pad[i] = (char)i;
    __syncthreads();
    if (o) o[0] = pad[threadIdx.x];
}
// A stub with no LDS at all, as the negative control the two attention kernels must NOT match.
__global__ void lds_none(int *o) { o[0] = 1; }

static int blocks_per_sm(const void *k, int threads) {
    int nb = -1;
    const hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, const_cast<void *>(k),
                                                                    threads, 0);
    if (e != hipSuccess) { fprintf(stderr, "occupancy query failed: %s\n",
                                   hipGetErrorString(e)); return -1; }
    return nb;
}

int main() {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) { printf("no device\n"); return 1; }
    // KFD is the authority on this box: /sys/.../kfd/topology/nodes/1/properties says
    // lds_size_in_kb 64, simd_per_cu 2, max_waves_per_simd 16, simd_count 80 -> 40 CU x 64 lanes.
    // HIP's multiProcessorCount (20) and sharedMemPerMultiprocessor (65536) are both per-2-CU here,
    // the same halving recorded for multiProcessorCount in devices/strix-halo-compute.md.
    printf("device                 %s\n", p.name);
    printf("  HIP multiProcessorCount (halved, do NOT use for roofs)  %d\n", p.multiProcessorCount);
    printf("  HIP sharedMemPerMultiprocessor                          %zu B\n",
           p.sharedMemPerMultiprocessor);
    printf("  KFD lds_size_in_kb per CU (sysfs, static)               64\n");
    printf("  KFD simd_per_cu / max_waves_per_simd / simd_count        2 / 16 / 80\n\n");

    struct Row { const char *name; const void *k; int threads; };
    const Row rows[] = {
        {"m16_bf16gq   (WMMA, global-A, rolled)", (const void *)m16_bf16gq, 128},
        {"m16_roll     (WMMA, staged, rolled)", (const void *)m16_roll, 128},
        {"m16_bf16rt   (WMMA, staged, UNROLLED)", (const void *)m16_bf16rt, 128},
        {"m16_empty    (WMMA-deleted baseline)", (const void *)m16_empty, 128},
        {"m16_scalar_ob (M=16 VALU control, DCE-proof)", (const void *)m16_scalar_ob, 128},
        {"attn_slim_wave (SHIPPED wave body, --attn-touch 1)", (const void *)attn_slim_wave,
         ATTN_TILE_THREADS},
        {"attn_tile_wave (--attn-touch 0, frozen witness)", (const void *)attn_tile_wave,
         ATTN_TILE_THREADS},
        {"attn_core    (one-row / --attn-tile 0)", (const void *)attn_core, 512},
        {"attn_prep", (const void *)attn_prep, 256},
        {"attn_combine", (const void *)attn_combine, 256},
        {"control: 0-LDS stub", (const void *)lds_none, 128},
        {"control: 65536-B LDS stub (must be 1)", (const void *)lds_whole_budget, 128},
    };
    printf("%-48s %-6s %8s %10s %14s\n", "kernel", "thread", "blk/CU", "wf/CU", "wf/CU of max 32");
    int slim_wf = -1;
    for (const Row &r : rows) {
        const int nb = blocks_per_sm(r.k, r.threads);
        const int wf = nb * (r.threads / 32);
        printf("%-48s %-6d %8d %10d %14s\n", r.name, r.threads, nb, wf,
               nb < 0 ? "-" : (std::to_string(100.0 * wf / 32.0).substr(0, 5) + " %").c_str());
        if (r.name[5] == 's') slim_wf = wf;   // attn_slim_wave row
    }
    // ---- canary -----------------------------------------------------------------------
    const int whole = blocks_per_sm((const void *)lds_whole_budget, 128);
    const int none = blocks_per_sm((const void *)lds_none, 128);
    const int slim = blocks_per_sm((const void *)attn_slim_wave, ATTN_TILE_THREADS);
    printf("\nCANARY\n");
    if (whole != 1) { printf("FAIL: 64 KiB-LDS stub returned %d, expected 1 — the query is not "
                             "reading LDS at all\n", whole); return 1; }
    if (slim == none) { printf("FAIL: attn_slim_wave reports the same blocks/CU as a 0-LDS stub — "
                               "static LDS is invisible to this query\n"); return 1; }
    if (slim < 1) { printf("FAIL: attn_slim_wave returned %d\n", slim); return 1; }
    printf("PASS: the query distinguishes a 64 KiB stub (1 block) from a 0-LDS stub (%d) from "
           "attn_slim_wave (%d blocks = %d wavefronts/CU)\n", none, slim, slim_wf);
    printf("\nReading: attn_slim_wave is %d wavefronts/CU of the 32 the part allows (%.1f %%), and "
           "the two caps are independent: LDS 12,288 B/block allows 5 blocks, and 47 VGPR (granularity "
           "8) allows %d waves/SIMD.\n", slim_wf, 100.0 * slim_wf / 32.0, 512 / 48);
    return 0;
}
