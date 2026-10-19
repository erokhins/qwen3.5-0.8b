// a10_rf_ladder.hip.cpp — A10 step 2: the COMPILE-ONLY register/LDS ladder for the R1+R2+R3 geometry.
//
// WHAT IT IS. A skeleton body whose live set is what `A10-REWRITE-BRIEF.md` §6 R1+R2+R3 forces a wave to hold
// at one time — P·V accumulator + Q fragments + a round's scores + staging — compiled at several
// `__launch_bounds__` rungs, plus LDS-only stubs. NOTHING IS LAUNCHED. The build is `-c` with `--save-temps`;
// the reading is the note block (`.vgpr_count`, `.vgpr_spill_count`, `group_segment_fixed_size`) via
// `prototype/attn_wmma/fattn_residency_read.py`, the same parser `fattn_ship_census.py` uses.
//
// WHY A SKELETON AND NOT THE REAL KERNEL. The real body is the ~1.1k-line rewrite step 3 is not allowed to
// start before step 2 answers its question. The question is arithmetic the compiler can answer for free: can
// 256 VGPR/thread hold 128 accumulator + Q + scores + addressing at 256 threads and 1 block/CU. A skeleton
// answers it as a LOWER BOUND, and the file says so instead of pretending otherwise: a skeleton always reads
// cheaper than the body it models, because it carries no mask logic, no split-partial bookkeeping, no free-row
// edge cases and no GQA index arithmetic. `a10_cur` is the witness for how far below: it models the SHIPPED
// geometry, whose real body reads 178 (staged) / 185 (`u8`, the default) VGPR, so the gap between `a10_cur`
// and those two numbers is the size of the skeleton's blind spot, and it is measured rather than assumed.
//
// THE GEOMETRY, from §3 SETTLED and §6 of the brief:
//   256 threads = 8 waves; 16 query rows per wave (the atom); 128 query rows per block; a 64-position KV
//   stage staged synchronously (`nstages_target = 1`, `fattn-mma-f16.cuh:164`); Q held in registers for the
//   whole sweep; rescale batched over the 64 positions (R4 falls out of R3).
//
// THE TWO ACCUMULATOR SPELLINGS ARE THE POINT OF THE LADDER, not decoration. Our engine's atom is
// `v_wmma_f32_16x16x16_f16` (`engine/attn_fattn_kernels.h:818`), whose 16-row x 256-dim P·V accumulator costs
// 128 f32 per lane. §3 SETTLED item 3 established llama's RDNA3 path uses `tile<16,16,half2>` +
// `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(..., opsel)` (`mma.cuh:1556-1566`), i.e. an f16-accumulating
// form covering 16 rows x 32 dims per 16-half-per-lane tile: 8 tiles, 128 halves per lane. Whether clang
// PACKS those halves into 64 VGPR or leaves them at 128 is exactly the kind of thing a register ladder decides
// and prose cannot. This file never claims the answer in a comment; it prints it.
//
// WHAT IS DELIBERATELY NOT MODELLED, so the read is not over-believed:
//   * mask / causal-edge state, split partials, GQA row indexing, the epilogue transpose buffers (all body
//     overhead — the `a10_cur` gap is the only handle on them);
//   * the mask LDS (~2.3 KiB in llama's shape, `fattn-mma-f16.cuh:2036`) — LDS figures here are stage-only;
//   * the register FILE per CU, which this box has never been made to print
//     (`prototype/attn_wmma/fattn_regfile_ladder.md` §3), so blocks/CU below is an LDS derivation and the
//     register bound is read from the rung plateau, never from a register-file assumption.
//
// The fragment types are the engine's own spellings (`engine/attn_fattn_kernels.h:184-185`) so the numbers are
// comparable with the shipping kernel rather than with a second ABI invented here.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using A10H = short __attribute__((ext_vector_type(16)));  // one 16x16 F16 tile: operand A/B, or f16 C (llama shape)
using A10F = float __attribute__((ext_vector_type(8)));   // one 16x16 F32 C tile (engine's fattn_CRegs)

#define A10_DH 256                    // head_dim
#define A10_ROWS 16                   // query rows per wave: forced by the atom
#define A10_DTILES (A10_DH / 16)      // 16 slices of 16 dims: the Q operand set, and the f32 accumulator set
#define A10_F16_TILES (A10_DH / 32)   // 8 tiles of 32 dims each: the f16 accumulator set

// ---------------------------------------------------------------------------------------------
// fragment loading. A fragment is 16 halves per lane = 8 32-bit words; loading it as words keeps the register
// shape honest (8 VGPR per fragment) instead of synthesising it from 16 half-width loads.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ A10H a10_frag_lds(const uint* base, int lane, int word_step) {
    A10H r;
    uint* w = (uint*)&r;
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = base[(size_t)(i * 32 + lane) + word_step];
    return r;
}

__device__ __forceinline__ A10H a10_frag_gld(const uint* base, int lane, int word_step) {
    A10H r;
    uint* w = (uint*)&r;
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = __ldg(base + (size_t)(i * 32 + lane) + word_step);
    return r;
}

// Pack a round's f32 scores into the F16 P operand the PV leg consumes. The engine does the same "16
// contiguous halves" packing (`engine/attn_fattn_kernels.h:419-424`); the scaling is a fake, the SHAPE is not.
__device__ __forceinline__ A10H a10_pack_p(const A10F& sc) {
    A10H p;
    short* h = (short*)&p;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __half2 hp = __floats2half2_rn(sc[i], sc[(i + 4) & 7]);
        h[2 * i] = __low2half(hp);
        h[2 * i + 1] = __high2half(hp);
    }
    return p;
}

// ---------------------------------------------------------------------------------------------
// the round body.
//   kAccF16   : P·V accumulates in F16 (llama's RDNA3 form) instead of F32 (ours).
//   kQReg     : Q is held in registers for the whole sweep (R2); false re-reads it every sub-step.
//   kStageHalf: LDS halves in the aliased K/V stage.  64 positions x 256 dims = 32,768 B aliased (llama's
//               alias, `:1251-1252`); 65,536 B is the non-aliased spelling that pays for K and V separately.
//   kThreads / kStagePos: the shipped shape is reprised with 96 threads and a 16-position stage.
// ---------------------------------------------------------------------------------------------

template <bool kAccF16, bool kQReg, int kStageHalf, int kThreads, int kStagePos>
__device__ __forceinline__ void a10_body(const uint* __restrict__ gq, const uint* __restrict__ gk,
                                         const uint* __restrict__ gv, float* __restrict__ sink, int nrounds) {
    __shared__ __align__(16) uint s_kv[kStageHalf / 2];
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int stage_words = kStagePos * (A10_DH / 2);          // uints per staged tensor
    const uint* qw = gq + (size_t)wave * (A10_ROWS * (A10_DH / 2));

    // ---- R2: Q resident, 16 rows x 256 dims = 16 operand tiles = 128 halves per lane. When Q is NOT resident
    // the array has size 1 and is never used, so the register cost of residency is the only thing the two
    // variants differ by (a size-0 array is not legal C++, hence the conditional bound).
    A10H q[kQReg ? A10_DTILES : 1];
    if (kQReg) {
#pragma unroll
        for (int t = 0; t < A10_DTILES; ++t) q[t] = a10_frag_gld(qw, lane, t * 8);
    }

    // ---- the accumulator: 16 rows x 256 dims, in F32 (128 f32/lane) or F16 (128 halves/lane). Braces, for the
    // engine's own reason (`engine/attn_fattn_kernels.h:790-798`): `v_wmma` takes the accumulator as a live IN
    // operand, so an uninitialised one reads the previous dispatch's registers.
    A10F accf[A10_DTILES]{};
    A10H accc[A10_F16_TILES]{};

    float run = 0.f;

#pragma unroll 1
    for (int kb = 0; kb < nrounds; ++kb) {
        // ---- R3: stage K synchronously into the block's one buffer, then read it back per 16-position step.
#pragma unroll 1
        for (int i = threadIdx.x; i < stage_words; i += kThreads) s_kv[i] = __ldg(gk + (size_t)kb * stage_words + i);
        __syncthreads();

        const int sub = kStagePos / 16;                          // 16-position steps per round: 4 at 64, 1 at 16
        A10F sc[4]{};
#pragma unroll
        for (int s = 0; s < 4; ++s) sc[s] = A10F();
#pragma unroll
        for (int s = 0; s < sub; ++s) {
#pragma unroll
            for (int t = 0; t < A10_DTILES; ++t) {
                const A10H kf = a10_frag_lds(s_kv, lane, s * 64 + t * 4);
                const A10H qf = kQReg ? q[kQReg ? t : 0] : a10_frag_gld(qw, lane, t * 8);
                sc[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf, qf, sc[s]);
            }
        }

        // ---- R4: the running max / rescale fires once per ROUND (kStagePos positions), not per 16.
        A10H p[4]{};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            float mx = sc[s][0];
#pragma unroll
            for (int i = 1; i < 8; ++i) mx = fmaxf(mx, sc[s][i]);
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) sum += expf(sc[s][i] - mx);
            run += sum;
            p[s] = a10_pack_p(sc[s]);
        }

        // ---- stage V into the SAME buffer once K is dead (the alias llama pays for with a second load).
        __syncthreads();
#pragma unroll 1
        for (int i = threadIdx.x; i < stage_words; i += kThreads) s_kv[i] = __ldg(gv + (size_t)kb * stage_words + i);
        __syncthreads();

        if (kAccF16) {
#pragma unroll
            for (int t = 0; t < A10_F16_TILES; ++t) {
#pragma unroll
                for (int s = 0; s < sub; ++s) {
                    // two 16-dim slices per tile, opsel 0 then 1, exactly `mma.cuh:1564-1565`
                    const A10H v0 = a10_frag_lds(s_kv, lane, s * 64 + t * 4);
                    const A10H v1 = a10_frag_lds(s_kv, lane, s * 64 + t * 4 + 128);
                    accc[t] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(v0, p[s], accc[t], 0);
                    accc[t] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(v1, p[s], accc[t], 1);
                }
            }
        } else {
#pragma unroll
            for (int t = 0; t < A10_DTILES; ++t) {
#pragma unroll
                for (int s = 0; s < sub; ++s) {
                    const A10H vf = a10_frag_lds(s_kv, lane, s * 64 + t * 4);
                    accf[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf, p[s], accf[t]);
                }
            }
        }
        __syncthreads();
    }

    // ---- the epilogue sink: touch every accumulator element so none of it can be dead-code eliminated.
    float acc = run;
    if (kAccF16) {
#pragma unroll
        for (int t = 0; t < A10_F16_TILES; ++t)
#pragma unroll
            for (int i = 0; i < 16; ++i) acc += (float)accc[t][i];
    } else {
#pragma unroll
        for (int t = 0; t < A10_DTILES; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc += accf[t][i];
    }
    sink[threadIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------------
// the rungs. `__launch_bounds__` is per-kernel, so each (variant, minBlocks) pair is its own `__global__`.
// minBlocks 1..4 at 256 threads asks clang to fit 256 / 128 / 85 / 64 VGPR per thread; at 96 threads the same
// rungs ask 256 / 85 / 56 / 42. The plateau is where the shape stops being able to fit, and the VGPR+scratch
// pair per rung is the price of each squeeze — the same reading discipline as `prototype/attn_percost/a6_rf_
// ladder.sh`, whose rung mechanism this copies and whose `attn_fattn_body<false>` call site is stale (§see
// A10-RF-LADDER.md).
// ---------------------------------------------------------------------------------------------

#define A10_STAGE64_HALF (64 * A10_DH)       // 32,768 B staged  (aliased K/V)
#define A10_STAGE64_SPLIT (128 * A10_DH)     // 65,536 B staged  (K and V both resident)
#define A10_STAGE16_HALF (16 * A10_DH)       //  8,192 B staged  (the shipped round)

#define A10_VARIANTS(PREFIX, ACCF16, QREG, STAGE_HALF, THREADS, STAGE_POS)                    \
    __global__ void __launch_bounds__(THREADS) PREFIX##_lb0(const uint* a, const uint* b,      \
                                                            const uint* c, float* d, int e) { \
        a10_body<ACCF16, QREG, STAGE_HALF, THREADS, STAGE_POS>(a, b, c, d, e);                 \
    }                                                                                         \
    __global__ void __launch_bounds__(THREADS, 1) PREFIX##_lb1(const uint* a, const uint* b,   \
                                                               const uint* c, float* d, int e) { \
        a10_body<ACCF16, QREG, STAGE_HALF, THREADS, STAGE_POS>(a, b, c, d, e);                  \
    }                                                                                         \
    __global__ void __launch_bounds__(THREADS, 2) PREFIX##_lb2(const uint* a, const uint* b,   \
                                                               const uint* c, float* d, int e) { \
        a10_body<ACCF16, QREG, STAGE_HALF, THREADS, STAGE_POS>(a, b, c, d, e);                  \
    }                                                                                         \
    __global__ void __launch_bounds__(THREADS, 3) PREFIX##_lb3(const uint* a, const uint* b,   \
                                                               const uint* c, float* d, int e) { \
        a10_body<ACCF16, QREG, STAGE_HALF, THREADS, STAGE_POS>(a, b, c, d, e);                  \
    }                                                                                         \
    __global__ void __launch_bounds__(THREADS, 4) PREFIX##_lb4(const uint* a, const uint* b,   \
                                                               const uint* c, float* d, int e) { \
        a10_body<ACCF16, QREG, STAGE_HALF, THREADS, STAGE_POS>(a, b, c, d, e);                  \
    }

// THE WITNESS: the shipped geometry — 96 threads / 3 waves, Q staged (not resident), a 16-position stage.
// Its real body reads 178 (staged) and 185 (u8) VGPR; the gap is the skeleton's blind spot, measured.
A10_VARIANTS(a10_cur, false, false, A10_STAGE16_HALF, 96, 16)
// THE BET: R1+R2+R3 at 256 threads / 8 waves / 64-position stage, both accumulator spellings, Q resident.
A10_VARIANTS(a10_r1_f32, false, true, A10_STAGE64_HALF, 256, 64)
A10_VARIANTS(a10_r1_f16, true, true, A10_STAGE64_HALF, 256, 64)
// The Q axis on its own, at the same shape: what residency actually costs in the ladder.
A10_VARIANTS(a10_r1_f32_noq, false, false, A10_STAGE64_HALF, 256, 64)
A10_VARIANTS(a10_r1_f16_noq, true, false, A10_STAGE64_HALF, 256, 64)
// The LDS axis on its own: the same f16 shape with K and V both resident (L-b's mistake at a deeper stage).
A10_VARIANTS(a10_r1_f16_split, true, true, A10_STAGE64_SPLIT, 256, 64)
// The 3-wave spelling of the same per-wave live set: the shape the brief says was rejected before on a spill.
A10_VARIANTS(a10_3w_f32, false, true, A10_STAGE64_HALF, 96, 64)
A10_VARIANTS(a10_3w_f16, true, true, A10_STAGE64_HALF, 96, 64)

// ---------------------------------------------------------------------------------------------
// LDS-only stubs: the LDS term, so a real kernel that reports fewer blocks than its own LDS allows has been
// refused by registers and not by something guessed. Same role as the stubs in `a6_rf_probe.inc`, at the
// sizes this geometry needs. Compile-only: their value here is `group_segment_fixed_size`, which the note
// block prints without a driver.
// ---------------------------------------------------------------------------------------------

#define A10_LDS_STUB(NAME, BYTES)                                                     \
    __global__ void __launch_bounds__(256) NAME(float* o) {                           \
        __shared__ __align__(16) char s[BYTES];                                       \
        s[threadIdx.x] = (char)threadIdx.x;                                           \
        __syncthreads();                                                              \
        o[threadIdx.x] = (float)s[threadIdx.x];                                       \
    }
A10_LDS_STUB(a10_stub_0, 1)
A10_LDS_STUB(a10_stub_8192, 8192)
A10_LDS_STUB(a10_stub_20480, 20480)
A10_LDS_STUB(a10_stub_32768, 32768)
A10_LDS_STUB(a10_stub_36096, 36096)
A10_LDS_STUB(a10_stub_65536, 65536)
#undef A10_LDS_STUB
