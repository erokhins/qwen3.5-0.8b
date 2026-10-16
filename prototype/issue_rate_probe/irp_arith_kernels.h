/*
 * irp_arith_kernels.h — one kernel per ARITHMETIC PATH this part offers.
 *
 * This file exists because `devices/strix-halo-official-specs.md` §7 proved the repo has never
 * issued a matrix or packed instruction: everything it ships — and everything rocBLAS ships for
 * gfx1151 in bf16 — executes scalar `v_fmac_f32_e32` at 1 MAC per lane per clock, and 0 of the 55
 * shipped gfx1151 code objects contain a matrix instruction. Question A of this probe is now "what
 * does each path actually sustain", and that needs one loop per path.
 *
 * THE CEILINGS. 40 CU, 2.894 GHz, 64 FP32 lane slots per CU (CU count and SIMDs from `rocminfo`,
 * clock from `rocm-smi`; the MAC/lane/clk column is `devices/strix-halo-official-specs.md` §7):
 *
 *   mnemonic                     MAC/lane/clk   ceiling        kernel here
 *   v_fma_f32 / v_fmac_f32_e32        1         14.82 TFLOP/s   irp_fma_*, irp_serial_*, irp_cvt_*
 *   v_fma_mix_f32                     1         14.82           irp_mix_8
 *   v_pk_fma_f16                      2         29.66           irp_pkf16_8
 *   v_dot2_f32_f16                    2         29.66           irp_d2f16_8
 *   v_dot2_f32_bf16                   2         29.66           irp_d2bf16_8
 *   v_dot4_i32_i8                     4         59.35 TOPS      irp_d4i8_8
 *   v_dot8_i32_i4                     8        118.70 TOPS      irp_d8i4_8
 *   v_wmma_f32_16x16x16_f16           4         59.36 TFLOP/s   irp_wmma_4
 *
 * A MAC is two FLOPs, and MAC/lane/clk counts LANE SLOTS (64 per CU), so the driver prints, on every
 * line so no reader has to redo it by hand:
 *
 *     mac_per_lane_clk = flop_per_s / (2 * cu_count * 64 * clock_hz)
 *     pct_of_ceiling   = 100 * mac_per_lane_clk / ceiling_mac_per_lane_clk
 *
 * which puts the scalar FMA roof at exactly 1.0000 and the WMMA roof at exactly 4.0000. One
 * warp-instruction's work per lane is `macs_per_lane_per_op`: 1 for a scalar FMA, 2 for packed f16
 * and the dot2 forms, 4 and 8 for the integer dots, 128 for a WMMA (4096 MAC over 32 lanes).
 *
 * HOW EACH MNEMONIC IS REACHED — established by compiling a one-loop kernel and objdumping it, not
 * by reading documentation, and re-verified on every build because `isa_census.py` refuses to bill a
 * loop that does not contain the claimed mnemonic.
 *
 *   v_fma_f32       `__fmaf_rn` on a per-lane-seeded float. Seed every lane equally and RDNA
 *                   scalarizes the loop into `s_fmac_f32` (the first version of this probe did
 *                   exactly that), so the per-lane seed is load-bearing, not decoration.
 *   v_fma_mix_f32   NOT reachable from source on this compiler: `float(bf16)` + `__fmaf_rn` compiles
 *                   to `v_lshlrev_b32` + `v_fma_f32` — LLVM prefers the shift to the mixed
 *                   instruction. Inline asm. The mnemonic assembles for gfx1151 (`tools/isa_probe.sh`
 *                   §1) and is what rocBLAS's HS_HPA library executes.
 *   v_pk_fma_f16    NOT reachable either: `_Float16` `a*b+c` on an `ext_vector_type(2)` is promoted
 *                   to f32 FMAs (observed: `v_fma_mix_f32`), never `v_pk_fma_f16`, even with
 *                   `-fno-experimental-isel`. Inline asm.
 *   v_dot2_f32_f16  inline asm, deliberately the PLAIN form and not the acc form (next paragraph).
 *   v_dot2_f32_bf16 `__builtin_amdgcn_fdot2_f32_bf16(i2,i2,f,0)` emits it verbatim; kept as source.
 *   v_dot4_i32_i8   inline asm. `__builtin_amdgcn_sdot4` exists but clang gates it behind the
 *                   `dot1-insts` subtarget feature and rejects it for gfx1151, and
 *                   `-Xclang -target-feature -Xclang +dot1-insts` does not survive the HIP offload
 *                   driver (the kernel silently vanishes from the code object). The unsigned builtin
 *                   `__builtin_amdgcn_udot4` IS accepted and emits `v_dot4_u32_u8`, the same
 *                   4 MAC/lane/clk class, so both spellings are kept side by side: irp_d4i8_8 is the
 *                   exact mnemonic from the ceiling table, irp_u4u8_8 is what a frontend program can
 *                   actually ask for.
 *   v_dot8_i32_i4   inline asm; `__builtin_amdgcn_udot8` (v_dot8_u32_u4) is the frontend twin.
 *   v_wmma_f32_16x16x16_f16
 *                   `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)` with
 *                   `ext_vector_type(16)` fp16 A and B and `ext_vector_type(8)` f32 C/D. The
 *                   CDNA-style `..._16f16_w32` spelling does not exist here — the trap that made an
 *                   earlier probe conclude WMMA was missing (`tools/isa_probe.sh` §2).
 *
 * WHY THE PLAIN DOT FORMS AND NOT THE `acc` ONES. `__builtin_amdgcn_fdot2` emits
 * `v_dot2acc_f32_f16`, and the `_acc` forms fold a scaled accumulator term, so both their per-op MAC
 * count and their accumulate semantics differ from the plain rows of the ceiling table. The plain
 * forms are exactly `acc += dot(...)`, which is what lets the host mirror be a closed form instead
 * of a guess about a doubling term. Switching to an acc form is a one-line change and the witness
 * would report it immediately; that is the forcing, not a caveat.
 *
 * INLINE ASM AND THE LIVE CHAIN. Every asm op is `mnemonic %0, %1, %2, %0` with a `+v` tied
 * accumulator, so the destination is also an input: the op sits in the loop-carried dependency chain
 * and cannot be hoisted, dead-coded or rotated out of the billed span. That is the difference between
 * measuring an instruction and measuring the absence of one — BLOCKER 1 of the previous run was
 * exactly a hoisted read, which is a broken measurement, not a fast kernel. The billed per-iteration
 * op count still comes from the disassembly, so if an asm leg ever stopped emitting its mnemonic the
 * census would stop billing it.
 *
 * WHAT THE INTEGER LEGS DO AND DO NOT MEASURE. irp_d4i8_8 and irp_d8i4_8 accumulate into their own
 * i32 register, so they measure the RAW dot rate: 4 and 8 MAC per lane per clock. A real quantized
 * GEMM adds, on top: a zero-point correction per output element (for asymmetric quantization
 * `acc += K*128*128 - 128*rowsum_a - 128*colsum_b`, which needs row/column sums of the operands),
 * then a dequantize pass (`v_cvt` + `v_mul`) into f32 before the residual add, plus the i32->f32
 * conversion of the accumulator if the epilogue is fused rather than run once at the end. Those are
 * per-output costs, so their weight falls as K grows, but they land on the same VALU pipe this probe
 * counts and they are not zero. The integer rows here are therefore a CEILING, not an expectation: a
 * quantized-kernel plan that promises the ceiling has not booked the epilogue. And there is no FP8 to
 * plan around instead — `v_dot4_f32_fp8_*` does not assemble for gfx1151 — so every "8-bit" idea in
 * this repo is an int8 idea with scales attached.
 */
#ifndef IRP_ARITH_KERNELS_H
#define IRP_ARITH_KERNELS_H

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

namespace irp_arith {

using irp::Args;
using irp::publish_block_sum;

// An fp16 accumulator holds 11 significand bits, so it carries exact integers only below 2^11; an
// f32 accumulator only below 2^24 (irp::kF32ExactInt). plan_check refuses an --iters that would
// cross the bound for the kernel's own accumulator type instead of letting the witness rot.
constexpr double kF16ExactInt = 2048.0;
constexpr double kI32ExactInt = 2147483648.0;

// The WMMA operand types. These ARE the layout the instruction requires on wave32: 16 fp16 per lane
// for A and B, 8 f32 per lane for C/D. `irp_wmma_frag` takes them as flat per-lane arrays so the
// host can write a fragment lane by lane without the kernel claiming a tile layout.
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef short i2 __attribute__((ext_vector_type(2)));   // the two bf16 halves the fdot2 builtin wants
constexpr int kWmmaLanes = 32;        // wave32 on this part
constexpr int kWmmaAFrag = 16;        // fp16 per lane, A and B
constexpr int kWmmaDFrag = 8;         // f32 per lane, C and D

// Bit patterns for the packed operands, as u32, so one table covers fp16 pairs, bf16 pairs, int8
// bytes and int4 nibbles. Every value is 1 in every field, which makes each op's contribution to the
// tallied accumulator exactly its MAC count: 2 for a packed fp16 FMA (1 per half), 2 for a dot2,
// 4 for dp4a, 8 for dp8a, 16 per D element for a WMMA. One exception: `kMixB` zeroes the high bf16
// so the mixed FMA's increment is 1 under either reading of its operand semantics (see irp_mix_8).
constexpr unsigned kPk1 = 0x3C003C00u;    // two fp16 1.0 : 1.0
constexpr unsigned kBf1 = 0x3F803F80u;    // two bf16 1.0 : 1.0
constexpr unsigned kBf0 = 0x00003F80u;    // bf16 pair (lo = 1.0, hi = +0.0)
constexpr unsigned kI8_1 = 0x01010101u;   // four int8 1s
constexpr unsigned kI4_1 = 0x11111111u;   // eight int4 1s

// Operand slots in Args::p, which is a float[128] the host fills. They come from device memory, not
// from literals, so the compiler cannot fold the operand values into the loop or replace an op with
// an add. Slot 16 holds the m word and 17 the b word for the packed legs; the WMMA legs use 16..23
// (A) and 24..31 (B) as 16 fp16 halves each.
constexpr int kWordM = 16;
constexpr int kWordB = 17;

__device__ __forceinline__ unsigned irp_word(const float* p, int slot) {
    unsigned w = 0;
    memcpy(&w, &p[slot], sizeof(w));
    return w;
}

// ---------------------------------------------------------------------------------------------
// The inline-asm spellings: one instruction each, accumulator tied in-out so it is in the live chain.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void irp_op_pk_fma_f16(unsigned& acc, unsigned a, unsigned b) {
    asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
}
__device__ __forceinline__ void irp_op_dot2_f32_f16(float& acc, unsigned a, unsigned b) {
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
}
__device__ __forceinline__ void irp_op_fma_mix_f32(float& acc, unsigned a, unsigned b) {
    asm volatile("v_fma_mix_f32 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
}
__device__ __forceinline__ void irp_op_dot4_i32_i8(int& acc, unsigned a, unsigned b) {
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
}
__device__ __forceinline__ void irp_op_dot8_i32_i4(int& acc, unsigned a, unsigned b) {
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
}
__device__ __forceinline__ void irp_op_dot2_f32_bf16(float& acc, unsigned a, unsigned b) {
    i2 x, y;
    memcpy(&x, &a, sizeof(x));
    memcpy(&y, &b, sizeof(y));
    acc = __builtin_amdgcn_fdot2_f32_bf16(x, y, acc, 0);
}
// The unsigned dot builtins accumulate in u32; these legs keep an i32 accumulator so the tally is the
// integer sum (one contribution per byte or nibble field) rather than a packed fp16 pair read, which
// is what the ceiling table's 4 and 8 MAC rows are about.
__device__ __forceinline__ void irp_op_dot4_u32_u8(int& acc, unsigned a, unsigned b) {
    unsigned u = (unsigned)acc;
    u = __builtin_amdgcn_udot4(a, b, u, 0);
    acc = (int)u;
}
__device__ __forceinline__ void irp_op_dot8_u32_u4(int& acc, unsigned a, unsigned b) {
    unsigned u = (unsigned)acc;
    u = __builtin_amdgcn_udot8(a, b, u, 0);
    acc = (int)u;
}
__device__ __forceinline__ void irp_op_fma_f32(float& acc, float m, float b) {
    acc = __fmaf_rn(m, acc, b);
}

// How a chain's accumulator contributes to the value the block publishes. A packed fp16 pair tallies
// as the SUM OF ITS TWO HALVES, which is what makes one v_pk_fma_f16 worth 2 to the witness, exactly
// as it is worth 2 MACs to the ceiling table.
__device__ __forceinline__ double irp_tally(float v) { return (double)v; }
__device__ __forceinline__ double irp_tally(int v) { return (double)v; }
__device__ __forceinline__ double irp_tally(unsigned v) {
    _Float16 h[2];
    memcpy(h, &v, sizeof(h));
    return (double)h[0] + (double)h[1];
}
__device__ __forceinline__ double irp_tally(f8 v) {
    double s = 0.0;
    for (int i = 0; i < kWmmaDFrag; ++i) s += (double)v[i];
    return s;
}

// THE SHAPE OF EVERY LEG, identical across legs on purpose so the instruction stream differs only in
// the op: one `_Pragma("unroll 1")` loop, N independent accumulators, exactly one op per accumulator
// per iteration, so the per-iteration op count equals N and the census can confirm it from the ISA.
// `DECL` declares and seeds the accumulators (per lane, so RDNA cannot scalarize; per chain, so a
// dropped chain moves the witness by its own amount and the failure names the chain it lost).
#define IRP_ARITH_KERNEL(NAME, TYPE, N, DECL, OP)                              \
__global__ void NAME(const Args a) {                                           \
    extern __shared__ double sh[];                                             \
    /* `kN`, not the macro parameter: a macro parameter does not substitute      \
       inside the replacement list of the separate SEED macros, so the seeds     \
       would otherwise see a bare `N` that was never declared. */               \
    constexpr int kN = (N);                                                     \
    const unsigned m_w = irp_word(a.p, irp_arith::kWordM);                      \
    const unsigned b_w = irp_word(a.p, irp_arith::kWordB);                      \
    const float m_f = a.p[0];                                                   \
    const float b_f = a.p[1];                                                   \
    (void)m_w; (void)b_w; (void)m_f; (void)b_f;                                 \
    TYPE acc[kN];                                                               \
    DECL                                                                        \
    _Pragma("unroll 1")                                                         \
    for (unsigned long long i = 0; i < a.iters; ++i) {                          \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < kN; ++j) {                                          \
            if (IRP_NEG == irp::kInjDropChain && j == kN - 1) continue;          \
            OP;                                                                 \
        }                                                                       \
    }                                                                           \
    double s = 0.0;                                                             \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < kN; ++j) s += irp_arith::irp_tally(acc[j]);              \
    if (a.verify) {                                                             \
        /* Per-thread account as well as the block account: a permutation that    \
           leaves the block total alone still moves this slot. */                 \
        if (a.each) a.each[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = (float)s; \
        publish_block_sum(s, sh, a.sums + a.launch * gridDim.x);                 \
    }                                                                           \
}

// Per-lane seeds, one per accumulator type. All three use the same integers (chain index + 1, plus
// the lane index) so one host closed form covers every leg: the per-thread seed tally is
// `seed_mul * (kN(kN+1)/2 + kN*lane)`, which is what the driver's `seed0`/`seed_lane` columns carry.
#define IRP_SEED_F32                                   \
    _Pragma("unroll")                                 \
    for (int j = 0; j < kN; ++j) acc[j] = a.p[64 + j] + (float)threadIdx.x;
#define IRP_SEED_I32                                   \
    _Pragma("unroll")                                 \
    for (int j = 0; j < kN; ++j) acc[j] = (int)(a.p[64 + j]) + (int)threadIdx.x;
#define IRP_SEED_U32                                                        \
    _Pragma("unroll")                                                      \
    for (int j = 0; j < kN; ++j) {                                          \
        const _Float16 h[2] = {(_Float16)(a.p[64 + j] + (float)threadIdx.x),  \
                               (_Float16)(a.p[64 + j] + (float)threadIdx.x)}; \
        memcpy(&acc[j], h, sizeof(acc[j]));                                   \
    }

IRP_ARITH_KERNEL(irp_pkf16_8, unsigned, 8, IRP_SEED_U32, irp_op_pk_fma_f16(acc[j], m_w, b_w))
IRP_ARITH_KERNEL(irp_d2f16_8, float, 8, IRP_SEED_F32, irp_op_dot2_f32_f16(acc[j], m_w, b_w))
IRP_ARITH_KERNEL(irp_d2bf16_8, float, 8, IRP_SEED_F32, irp_op_dot2_f32_bf16(acc[j], m_w, b_w))
// The mixed FMA is bf16-in / f32-out and documentation in this repo has it booked at 1 MAC/lane/clk
// (devices/strix-halo-official-specs.md §7.1). Its operand semantics are ambiguous between one
// product and a two-element bf16 dot, so the b operand carries a zero in its high half: under either
// reading the increment is exactly 1, the witness stays a closed form, and the ceiling column still
// follows §7.1. If the high half mattered, the witness would move and say so.
IRP_ARITH_KERNEL(irp_mix_8, float, 8, IRP_SEED_F32, irp_op_fma_mix_f32(acc[j], m_w, b_w))
IRP_ARITH_KERNEL(irp_d4i8_8, int, 8, IRP_SEED_I32, irp_op_dot4_i32_i8(acc[j], m_w, b_w))
IRP_ARITH_KERNEL(irp_d8i4_8, int, 8, IRP_SEED_I32, irp_op_dot8_i32_i4(acc[j], m_w, b_w))
IRP_ARITH_KERNEL(irp_u4u8_8, int, 8, IRP_SEED_I32, irp_op_dot4_u32_u8(acc[j], m_w, b_w))
IRP_ARITH_KERNEL(irp_u8u4_8, int, 8, IRP_SEED_I32, irp_op_dot8_u32_u4(acc[j], m_w, b_w))
// The scalar baseline spelled through the same macro as every other leg, so the ceiling table has a
// row per path and `irp_fma_*`/`irp_serial_*` remain the instruction-count/LDS-shaped versions.
IRP_ARITH_KERNEL(irp_f32_8, float, 8, IRP_SEED_F32, irp_op_fma_f32(acc[j], m_f, b_f))

// =============================================================================================
// The WMMA legs.
//
// irp_wmma_4 is the K loop: A and B are 16-wide fragments held in registers and each iteration
// accumulates one more 16-deep K slab into the same D, which is exactly what a tile GEMM or an
// attention score loop does over K. Four independent D accumulators, so the per-iteration op count is
// 4 and the wavefront has independent matrix work to issue rather than a single dependent chain.
//
// D is f32, so the accumulation is exact for integer operands: with A and B both all-ones fp16 every
// D element grows by exactly 16 per op (16 MACs of 1x1), which the host mirror predicts in closed
// form. A dropped or duplicated WMMA moves the block sum by exactly 128 per D element.
// =============================================================================================
#define IRP_WMMA_KERNEL(NAME, N)                                                \
__global__ void NAME(const Args a) {                                            \
    extern __shared__ double sh[];                                              \
    constexpr int kN = (N);                                                     \
    h16 A, B;                                                                   \
    _Pragma("unroll")                                                           \
    for (int i = 0; i < irp_arith::kWmmaAFrag; ++i) {                           \
        A[i] = a.frag[i];                                                       \
        B[i] = a.frag[irp_arith::kWmmaAFrag + i];                               \
    }                                                                           \
    f8 acc[kN];                                                                 \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < kN; ++j) {                                              \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < irp_arith::kWmmaDFrag; ++i) {                       \
            acc[j][i] = (float)(j * irp_arith::kWmmaDFrag + i + 1)              \
                        + (float)threadIdx.x;                                   \
        }                                                                       \
    }                                                                           \
    _Pragma("unroll 1")                                                         \
    for (unsigned long long i = 0; i < a.iters; ++i) {                          \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < kN; ++j) {                                          \
            if (IRP_NEG == irp::kInjDropChain && j == kN - 1) continue;          \
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, acc[j]);   \
        }                                                                       \
    }                                                                           \
    double s = 0.0;                                                             \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < kN; ++j) s += irp_arith::irp_tally(acc[j]);              \
    if (a.verify) {                                                             \
        /* Per-thread account as well as the block account, as in IRP_ARITH_KERNEL. \
           A WMMA is all 8 D elements at once, so the D-vector's sum is the chain. */ \
        if (a.each) a.each[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = (float)s; \
        publish_block_sum(s, sh, a.sums + a.launch * gridDim.x);                 \
    }                                                                           \
}

IRP_WMMA_KERNEL(irp_wmma_4, 4)

// =============================================================================================
// The WMMA correctness kernel, and why it is shaped this way.
//
// The lane layout is the part that goes silently wrong, and this worker takes no device
// measurements, so the last thing this file should do is hard-code a transcription of the wave32
// fragment layout and call it verified. It does not. `irp_wmma_frag` performs ONE wmma on raw
// per-lane fragments supplied by the host and writes raw per-lane D back, so no tile layout is
// claimed anywhere in the kernel. The host then DERIVES the layout from the device instead of
// asserting one:
//
//   `ones`   A and B fragments all 1.0, C = 0   -> every D element must be exactly 16.0. Layout-free:
//            it proves the instruction is a 16-deep accumulating dot and that D is 8 f32 per lane.
//   `rows`   A fragment of lane p = (p+1) in every slot, B all 1.0, C = 0 -> each D element is
//            16*(p+1) for the lane p that feeds it, so the D slot reveals which A lane fed it.
//   `cols`   the same with the roles swapped.
//   `tile`   rank-1 fp16 tiles built from the derived feeder map, compared against a naive triple-loop
//            fp16 matmul of the same tile computed on the host, exactly.
//
// From `rows` and `cols` the host also checks that the 256 D slots pair 16 A-feeders with 16 B-feeders
// as 256 DISTINCT pairs, which is the statement "this instruction really computes a 16x16 grid of
// dots" — derived from the part, not transcribed from a blog post. The tiles are rank-1 along K for
// the same reason: a general tile would require claiming which k each fragment slot carries, and the
// derived map above does not establish that. See HANDOFF: if a real WMMA kernel is going to be built,
// the next step is to derive the k-slot order the same way, not to trust a layout table.
// =============================================================================================
__global__ void irp_wmma_frag(const __half* __restrict__ afrag, const __half* __restrict__ bfrag,
                              const float* __restrict__ cfrag, float* __restrict__ dfrag) {
    const int lane = threadIdx.x % warpSize;
    h16 A, B;
    f8 C;
    _Pragma("unroll")
    for (int i = 0; i < irp_arith::kWmmaAFrag; ++i) {
        A[i] = afrag[lane * irp_arith::kWmmaAFrag + i];
        B[i] = bfrag[lane * irp_arith::kWmmaAFrag + i];
    }
    _Pragma("unroll")
    for (int i = 0; i < irp_arith::kWmmaDFrag; ++i)
        C[i] = cfrag ? cfrag[lane * irp_arith::kWmmaDFrag + i] : 0.0f;
    const f8 D = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, C);
    // Every D element is written, so a layout mistake shows up as a wrong value rather than as
    // uninitialised memory: the caller pre-fills dfrag with a value the kernel cannot produce.
    _Pragma("unroll")
    for (int i = 0; i < irp_arith::kWmmaDFrag; ++i)
        dfrag[lane * irp_arith::kWmmaDFrag + i] = D[i];
}

}  // namespace irp_arith

#endif  // IRP_ARITH_KERNELS_H
