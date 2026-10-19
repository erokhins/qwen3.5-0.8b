// a10_frag_cost.hip.cpp — what ONE WMMA fragment costs in VGPRs on this toolchain. COMPILE ONLY, nothing launched.
//
// WHY THIS EXISTS. `a10_rf_ladder.sh`'s first run put every variant at `vgpr_count=256` WITH several hundred
// bytes of spill — including the variant built to model the shipped geometry, which really reads 185 VGPR / 0
// spill. Before that can be read as "the geometry cannot fit", the ladder's unit has to be calibrated. The
// A10 live set is Q (16 operand tiles) + the P·V accumulator (16 f32 tiles, or 8 f16 tiles in llama's RDNA3
// spelling) + a round's scores (4 f32 tiles), and its storage is 8·36 = 288 VGPRs if a 16-half operand tuple
// costs 8 VGPRs, or 144 if clang packs those halves into 4. Those two answers are "spills for sure" and
// "fits with room", and no prose picks between them: LLVM's builtin declares `<16 x half>` for A/B and
// `<8 x float>` for C, and the allocation is a codegen fact.
//
// METHOD. Hold N fragments of one class live (load them all, then fold them all into a sink), so the marginal
// VGPRs over N=0 divided by N is the class's per-fragment cost. No WMMA call in these probes on purpose: the
// pure storage question has to be separated from the tuple-alignment pressure the `v_wmma` instructions add,
// and the ladder file carries the variant that has both.
//
// Read with:  python3 prototype/attn_wmma/fattn_residency_read.py --asm <listing> --all | grep _hold

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using A16 = short __attribute__((ext_vector_type(16)));  // the engine's fattn_ARegs (`engine/attn_fattn_kernels.h:184`)
using A8  = short __attribute__((ext_vector_type(8)));   // half as wide — does the narrower tuple cost half?
using C32 = float __attribute__((ext_vector_type(8)));   // the engine's fattn_CRegs (`:185`), one 16x16 f32 tile
using C64 = float __attribute__((ext_vector_type(16)));  // twice as wide, for the slope to be unambiguous

__device__ __forceinline__ uint h_ld(const uint* p, int lane, int i) { return __ldg(p + (size_t)(i * 32 + lane)); }

// THE BARRIER THAT MAKES THE MEASUREMENT MEAN ANYTHING. Without it the first run of this file read 82 VGPR for
// a probe holding 32 fragments of 8 words each (256 expected): a pure `__ldg` has no side effects, so clang was
// free to sink each load to just before its use and reuse four registers forever — the probe was measuring the
// address arithmetic, not the storage. A `volatile` asm with a `memory` clobber between the loads and the fold
// forbids that reordering, so every loaded word is live across it and `.vgpr_count` is the storage cost.
#define H_BARRIER() asm volatile("" ::: "memory")

#define H_STORE(TYPE, NAME, WORDS, ELEM)                                                     \
    template <int N>                                                                          \
    __device__ __forceinline__ void NAME##_body(const uint* __restrict__ in, float* __restrict__ sink) { \
        TYPE f[N]{};                                                                           \
        _Pragma("unroll")                                                                       \
        for (int i = 0; i < N; ++i) {                                                            \
            uint* w = (uint*)&f[i];                                                               \
            _Pragma("unroll")                                                                     \
            for (int j = 0; j < WORDS; ++j) w[j] = h_ld(in, threadIdx.x, i * WORDS + j);           \
        }                                                                                          \
        H_BARRIER();                                                                                 \
        unsigned acc = 0u;                                                                          \
        _Pragma("unroll")                                                                            \
        for (int i = 0; i < N; ++i)                                                                   \
            _Pragma("unroll")                                                                          \
            for (int j = 0; j < ELEM; ++j) acc += (unsigned)f[i][j];                                  \
        sink[threadIdx.x] = (float)acc;                                                               \
    }

#define H_EMIT(TYPE, NAME, WORDS, ELEM)                                                    \
    H_STORE(TYPE, NAME, WORDS, ELEM)                                                        \
    __global__ void NAME##_hold_1(const uint* i, float* s) { NAME##_body<1>(i, s); }          \
    __global__ void NAME##_hold_4(const uint* i, float* s) { NAME##_body<4>(i, s); }          \
    __global__ void NAME##_hold_8(const uint* i, float* s) { NAME##_body<8>(i, s); }          \
    __global__ void NAME##_hold_16(const uint* i, float* s) { NAME##_body<16>(i, s); }        \
    __global__ void NAME##_hold_32(const uint* i, float* s) { NAME##_body<32>(i, s); }

// Read with N=1 as the baseline (a zero-length array is not legal device code): the slope is
// (vgpr(N) - vgpr(1)) / (N - 1) per fragment of the class.
// 16 halves per lane = 32 bytes = 8 words if packed, 8 VGPR. The operand class Q lives in.
H_EMIT(A16, a16, 8, 16)
// 8 halves per lane. If the f16 accumulator's tiles are allocated at half the cost of the f32 ones, this is
// the shape that shows the packing is real; if it lands at half of A16, packing is real; if it lands equal,
// clang spent a whole word per half and llama's f16 accumulator buys nothing on this compiler.
H_EMIT(A8, a8, 4, 8)
// The f32 accumulator class.
H_EMIT(C32, cf, 8, 8)
// A double-width f32 vector: if the slope here is double the C32 slope, allocation is linear in elements.
H_EMIT(C64, c64, 16, 16)
