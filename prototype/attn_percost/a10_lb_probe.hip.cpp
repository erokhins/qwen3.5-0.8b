// a10_lb_probe.hip.cpp — does clang honour `__launch_bounds__(threads, minBlocks)` on gfx1151? COMPILE ONLY.
//
// TWO QUESTIONS THIS ANSWERS, both of which the A10 ladder otherwise rests on air.
//
// (1) THE LADDER'S OWN VALIDITY. `a6_rf_ladder.sh`'s rung mechanism, and A10 step 2's instruction to report "a
//     `__launch_bounds__` occupancy ladder", both assume the min-blocks argument makes the allocator *fit*
//     N blocks. If it does not, a rung that prints "fits" is printing the request back, which is the AGENTS
//     failure class "a check whose input can silently fail to arrive". A10's first ladder run had every rung at
//     `vgpr_count=256` regardless of the hint, which is what prompted this file.
//
// (2) A PROVENANCE QUESTION §3 of `A10-REWRITE-BRIEF.md` left open. llama's RDNA row for our shape declares
//     `occupancy 2` (`fattn-mma-f16.cuh:164`) and feeds it to `__launch_bounds__(nthreads, occupancy)` at
//     `:1797`, yet the traced kernel reports 256 VGPR/thread (`A2-census.md:78`) — 2 blocks of 256 threads at
//     256 VGPR each would need twice the register file. Either the trace column is wrong or the min-blocks hint
//     asked the allocator for nothing. This file distinguishes them on a body whose register demand is known.
//
// METHOD: the same N-fragment hold probe as `a10_frag_cost.hip.cpp`, at N chosen to need ~128 VGPRs, emitted at
// min-blocks 1..4 with 256 threads (the implied ceilings are 256 / 128 / 85 / 64). If the hint binds, `vgpr_count`
// falls to the ceiling and the spill count rises; if it is inert, every rung reads 128/0 and the rung ladder is
// decorative. Read with:
//   python3 prototype/attn_wmma/fattn_residency_read.py --asm <listing> --all | grep _lb

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using A16 = short __attribute__((ext_vector_type(16)));

template <int N>
__device__ __forceinline__ void lb_body(const uint* __restrict__ in, float* __restrict__ sink) {
    A16 f[N]{};
#pragma unroll
    for (int i = 0; i < N; ++i) {
        uint* w = (uint*)&f[i];
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = __ldg(in + (size_t)(i * 8 + j) * 32 + threadIdx.x);
    }
    // Same barrier discipline as `a10_frag_cost.hip.cpp`: without it clang sinks each pure load to its use and
    // the probe measures address arithmetic instead of storage.
    asm volatile("" ::: "memory");
    unsigned acc = 0u;
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int j = 0; j < 16; ++j) acc += (unsigned)f[i][j];
    sink[threadIdx.x] = (float)acc;
}

#define LB_EMIT(THREADS, MINB, NAME)                                                        \
    __global__ void __launch_bounds__(THREADS, MINB) NAME(const uint* i, float* s) { lb_body<16>(i, s); }
LB_EMIT(256, 1, lb256_m1)
LB_EMIT(256, 2, lb256_m2)
LB_EMIT(256, 3, lb256_m3)
LB_EMIT(256, 4, lb256_m4)
LB_EMIT(256, 8, lb256_m8)
LB_EMIT(96, 1, lb96_m1)
LB_EMIT(96, 3, lb96_m3)
LB_EMIT(96, 7, lb96_m7)
#undef LB_EMIT

// And the same demand with NO second argument at all, as the reference point.
__global__ void __launch_bounds__(256) lb256_none(const uint* i, float* s) { lb_body<16>(i, s); }
__global__ void __launch_bounds__(96) lb96_none(const uint* i, float* s) { lb_body<16>(i, s); }
