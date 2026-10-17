// wmma_feeder_probe.cpp — COMPILE-ONLY price probe for D30 item 2.4 ("price a WMMA path for attn").
//
// NEVER LAUNCHED, AND CANNOT BE LAUNCHED BY THIS FILE'S BUILD (`-c` only, no link, no main).
// No timing is taken and none may be quoted from anything built here. The deliverable is the GCN
// ISA and the slot counts `feeder_census.py` reads out of it.
//
// WHY THIS FILE EXISTS
// --------------------
// `prototype/issue_rate_probe/results.md` §5.1 measured the WMMA *rate* (52.63 TFLOP/s, 88.8 % of
// its roof) but refused the feeder question: what does it cost, in issue slots, to get a 16x16 bf16
// tile from an engine-shaped buffer into the WMMA operand registers and the f32 result back out,
// and does the compiler have to move lanes around to satisfy the fragment map? Its conclusion was
// "no shipping WMMA kernel may assume a hand-rolled fragment layout".
//
// That refusal can be narrowed without a device. `rocwmma` is installed and its WMMA path is
// explicitly enabled for this arch — `enable_gfx11_t` in
// /opt/rocm/include/rocwmma/internal/wmma_impl.hpp lists AMDGCN_ARCH_ID_GFX1151 — and rocwmma's
// `load_matrix_sync`/`store_matrix_sync` are the map-aware feeders: they compute the per-lane
// addresses the ISA actually wants. So compiling them for gfx1151 and reading the ISA reveals the
// map's cost without assuming it. The raw-builtin variants below are the comparison: they are what
// a hand-rolled "lane l owns row l" feeder would cost, i.e. the floor IF the map were the engine's
// own layout. Their numerical correctness is deliberately NOT claimed (that is the device
// falsifier's job); only their ISA shape is used.
//
// THE LAYOUTS FED FROM (the engine's, not an idealised one)
// --------------------------------------------------------
// `engine/attn_slim_kernels.h` stages K and V as `s_k[ATTN_TILE_L * DH]` / `s_v[...]`: DH = 256
// BF16 per row, so a row stride of 512 B, each row written by one lane as one `uint4` at
// `slot * DH + lane * 8`. Scores are staged as `s_sc[ATTN_TILE_G][MAXSC]` (MAXSC = 512 BF16,
// 1024 B stride). A 16x16 tile is therefore a *sub-block of a wide row*, never a tight 32 B row:
//   score leg   S[m][n] = sum_d Q[m][d] * K[n][d]
//     A = Q  : row_major, ldm = 256      (a Q row is 256 contiguous BF16)
//     B = K^T as K x N: B[d][n] = K[n][d] at n*256 + d  -> col_major, ldm = 256
//     D = S  : f32 accumulator, stored row-major, ldm = 512 (s_sc's row width)
//   PV leg      O[m][d] = sum_n W[m][n] * V[n][d]
//     A = W  : row_major, ldm = 512, and **BF16** — see the format note in the handoff
//     B = V  : row_major, ldm = 256
//     D = O  : f32, row-major, ldm = 256
// K = 256 means the score leg is SIXTEEN 16-deep K-steps into ONE accumulator, so the D store
// happens once per 16 tiles, not per tile; `wm_score_k256` prices that shape explicitly.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#define DH 256     // engine: BF16 per K/V row    -> LDS row stride 512 B
#define MAXSC 512  // engine: BF16 per score row  -> LDS row stride 1024 B

// The raw builtin's operand types, taken from rocwmma's own call site
// (/opt/rocm/include/rocwmma/internal/wmma_impl.hpp, `amdgcn_wmma<bfloat16_t,...>::exec`:
// A/B are 8 f32-sized registers reinterpreted as `VecT<short, 16>` = 16 BF16 = 32 B per lane;
// C/D are 8 f32 per lane).
namespace {

using ARegs = short __attribute__((ext_vector_type(16)));
using CRegs = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ ARegs pack_two_b128(const hip_bfloat16* p) {
    // 32 B per lane = two 16 B LDS loads. This is the hypothesised "lane owns a contiguous
    // 16-element run" map; it is a PRICE FLOOR, not a proven-correct feeder. The union is so the
    // raw builtin sees the register group as 16 shorts with no conversion VALU at all.
    union { uint4 u[2]; ARegs s; } v;
    v.u[0] = *reinterpret_cast<const uint4*>(p);
    v.u[1] = *reinterpret_cast<const uint4*>(p + 8);
    return v.s;
}

__device__ __forceinline__ void stage_rows(hip_bfloat16* dst, const hip_bfloat16* src, float* sink) {
    // Engine-shaped staging: one row per lane, one `uint4` per lane, `DH` row stride —
    // the same write `s_k[slot * DH + lane * 8] = ...` performs.
    const int lane = threadIdx.x & 31;
    const int warp = (int)(threadIdx.x >> 5);
    const int nw = (int)(blockDim.x >> 5);
    for (int r = warp; r < 16; r += nw) {
        *reinterpret_cast<uint4*>(&dst[r * DH + lane * 8]) =
            *reinterpret_cast<const uint4*>(&src[(size_t)r * DH + lane * 8]);
    }
    // Keep the staging live: the compiler may not delete a store whose data can never be observed.
    if (sink != nullptr && *reinterpret_cast<const unsigned short*>(&dst[lane]) == 0x1u) *sink = 1.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
// V1 — score leg, map-aware feed from engine-shaped LDS, ONE 16-deep K-step.
// The quantity of interest: ds_load widths + any lane movement the map demands.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128)
wm_score_tile(const hip_bfloat16* __restrict__ q,
              const hip_bfloat16* __restrict__ k,
              float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[16 * DH];
    __shared__ hip_bfloat16 s_k[16 * DH];
    __shared__ float s_sc[16 * MAXSC];
    stage_rows(s_q, q, sink);
    stage_rows(s_k, k, sink);
    __syncthreads();

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::col_major> fb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc, fd;
    rocwmma::fill_fragment(fc, 0.0f);
    rocwmma::load_matrix_sync(fa, &s_q[0], DH);
    rocwmma::load_matrix_sync(fb, &s_k[0], DH);
    rocwmma::mma_sync(fd, fa, fb, fc);
    rocwmma::store_matrix_sync(&s_sc[0], fd, MAXSC, rocwmma::mem_row_major);

    const int lane = threadIdx.x & 31;
    if (sink != nullptr) sink[lane] = s_sc[(lane & 15) * MAXSC + lane] + s_sc[lane & 15];
}

// ---------------------------------------------------------------------------
// V2 — score leg at the real depth: K = 256 = sixteen 16-deep steps into ONE
// accumulator, then a single D store. This is the shape the shipped pass 1 covers
// with 16 touches, so its slot count is the one to compare against 45 x 16.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128)
wm_score_k256(const hip_bfloat16* __restrict__ q,
              const hip_bfloat16* __restrict__ k,
              float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[16 * DH];
    __shared__ hip_bfloat16 s_k[16 * DH];
    __shared__ float s_sc[16 * MAXSC];
    stage_rows(s_q, q, sink);
    stage_rows(s_k, k, sink);
    __syncthreads();

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc, fd;
    rocwmma::fill_fragment(fc, 0.0f);
#pragma unroll
    for (int d0 = 0; d0 < DH; d0 += 16) {
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> fa;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::col_major> fb;
        rocwmma::load_matrix_sync(fa, &s_q[d0], DH);
        rocwmma::load_matrix_sync(fb, &s_k[d0], DH);
        rocwmma::mma_sync(fd, fa, fb, fc);
        fc = fd;
    }
    rocwmma::store_matrix_sync(&s_sc[0], fd, MAXSC, rocwmma::mem_row_major);

    const int lane = threadIdx.x & 31;
    if (sink != nullptr) sink[lane] = s_sc[(lane & 15) * MAXSC + lane] + s_sc[lane & 15];
}

// ---------------------------------------------------------------------------
// V3 — PV leg: A = the staged BF16 scores (row-major, ldm = MAXSC), B = V
// (row-major, ldm = DH), D = output dims (row-major, ldm = DH).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128)
wm_pv_tile(const hip_bfloat16* __restrict__ w_src,
           const hip_bfloat16* __restrict__ v_src,
           float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_w[16 * MAXSC];
    __shared__ hip_bfloat16 s_v[16 * DH];
    __shared__ float s_o[16 * DH];
    stage_rows(s_v, v_src, sink);
    {
        const int lane = threadIdx.x & 31;
        const int warp = (int)(threadIdx.x >> 5);
        for (int r = warp; r < 16; r += (int)(blockDim.x >> 5)) {
            *reinterpret_cast<uint4*>(&s_w[r * MAXSC + lane * 8]) =
                *reinterpret_cast<const uint4*>(&w_src[(size_t)r * MAXSC + lane * 8]);
        }
    }
    __syncthreads();

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> fb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc, fd;
    rocwmma::fill_fragment(fc, 0.0f);
    rocwmma::load_matrix_sync(fa, &s_w[0], MAXSC);
    rocwmma::load_matrix_sync(fb, &s_v[0], DH);
    rocwmma::mma_sync(fd, fa, fb, fc);
    rocwmma::store_matrix_sync(&s_o[0], fd, DH, rocwmma::mem_row_major);

    const int lane = threadIdx.x & 31;
    if (sink != nullptr) sink[lane] = s_o[(lane & 15) * DH + lane] + s_o[lane & 15];
}

// ---------------------------------------------------------------------------
// V3b — PV leg at its real width: N = 256 output dims = SIXTEEN 16-wide N-tiles
// over one 16-deep KV slice, A fed from the BF16 score row (ldm = MAXSC). If the
// A feed degrades to per-element `ds_load_u16` here (it does at ldm = MAXSC in
// V3), the PV leg's feeder is a different price from the score leg's, and this is
// the leg that also changes the P operand format.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128)
wm_pv_n256(const hip_bfloat16* __restrict__ w_src,
           const hip_bfloat16* __restrict__ v_src,
           float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_w[16 * MAXSC];
    __shared__ hip_bfloat16 s_v[16 * DH];
    __shared__ float s_o[16 * DH];
    stage_rows(s_v, v_src, sink);
    {
        const int lane = threadIdx.x & 31;
        const int warp = (int)(threadIdx.x >> 5);
        for (int r = warp; r < 16; r += (int)(blockDim.x >> 5)) {
            *reinterpret_cast<uint4*>(&s_w[r * MAXSC + lane * 8]) =
                *reinterpret_cast<const uint4*>(&w_src[(size_t)r * MAXSC + lane * 8]);
        }
    }
    __syncthreads();

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc;
    rocwmma::fill_fragment(fc, 0.0f);
    rocwmma::load_matrix_sync(fa, &s_w[0], MAXSC);
#pragma unroll
    for (int n0 = 0; n0 < DH; n0 += 16) {
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> fb;
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fd;
        rocwmma::load_matrix_sync(fb, &s_v[n0], DH);
        rocwmma::mma_sync(fd, fa, fb, fc);
        rocwmma::store_matrix_sync(&s_o[n0], fd, DH, rocwmma::mem_row_major);
    }

    const int lane = threadIdx.x & 31;
    if (sink != nullptr) sink[lane] = s_o[(lane & 15) * DH + lane] + s_o[lane & 15];
}

// ---------------------------------------------------------------------------
// V4 — the hand-rolled floor: raw builtin fed by two `ds_load_b128`-class LDS
// reads per operand and drained by two 16 B stores, i.e. ZERO lane movement.
// Legal only if the hardware map really is "lane l owns a contiguous 16-element
// run"; correctness is the device falsifier's job, this variant prices the shape.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128)
wm_raw_floor_k256(const hip_bfloat16* __restrict__ q,
                  const hip_bfloat16* __restrict__ k,
                  float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[16 * DH];
    __shared__ hip_bfloat16 s_k[16 * DH];
    __shared__ float s_sc[16 * MAXSC];
    stage_rows(s_q, q, sink);
    stage_rows(s_k, k, sink);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = lane & 15;
    CRegs acc{};
#pragma unroll
    for (int d0 = 0; d0 < DH; d0 += 16) {
        ARegs a = pack_two_b128(&s_q[row * DH + d0]);
        ARegs b = pack_two_b128(&s_k[row * DH + d0]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    union { CRegs v; float4 f[2]; } out;
    out.v = acc;
    *reinterpret_cast<float4*>(&s_sc[row * MAXSC]) = out.f[0];
    *reinterpret_cast<float4*>(&s_sc[row * MAXSC + 4]) = out.f[1];
    __syncthreads();
    if (sink != nullptr) sink[lane] = s_sc[(lane & 15) * MAXSC + lane] + s_sc[lane & 15];
}

extern "C" __global__ void __launch_bounds__(128)
wm_raw_floor(const hip_bfloat16* __restrict__ q,
             const hip_bfloat16* __restrict__ k,
             float* __restrict__ sink) {
    __shared__ hip_bfloat16 s_q[16 * DH];
    __shared__ hip_bfloat16 s_k[16 * DH];
    __shared__ float s_sc[16 * MAXSC];
    stage_rows(s_q, q, sink);
    stage_rows(s_k, k, sink);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    
    const int row = lane & 15;
    ARegs a = pack_two_b128(&s_q[row * DH]);
    ARegs b = pack_two_b128(&s_k[row * DH]);
    CRegs c{};
    CRegs d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    // Drain: 8 f32 = 32 B per lane. Two 16 B stores if contiguous, which is the question.
    union { CRegs v; float4 f[2]; } out;
    out.v = d;
    *reinterpret_cast<float4*>(&s_sc[row * MAXSC]) = out.f[0];
    *reinterpret_cast<float4*>(&s_sc[row * MAXSC + 4]) = out.f[1];
    __syncthreads();
    if (sink != nullptr) sink[lane] = s_sc[(lane & 15) * MAXSC + lane] + s_sc[lane & 15];
}
