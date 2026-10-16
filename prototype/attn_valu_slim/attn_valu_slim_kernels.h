/*
 * attn_valu_slim_kernels.h — can the per-touch instruction bill of the shipped tiled attention wave
 * be cut without spending a numerics decision?
 *
 * D20 (`progress/2026-10-16/prefill-decision-log-2026-10-16.md`) funded this route on the claim that
 * "the causal-mask guards cost 20 VALU instructions per touch — more than the dot they protect — and
 * the bf16->f32 conversions 8 more", and that removing them is bit-identity preserving by
 * construction. The first act of this file is to find out, in the disassembly, what those 20 VALU
 * actually are. They are NOT a causal mask: this form's causal bound is already a loop limit
 * (`s_lo`/`s_hi` bound the walk by the row's own slice `ns = bhi - blo`), and there is no
 * per-element causal guard to remove. What the 20 VALU are is the per-element machinery the compiler
 * puts around the two things the score loop *does* contain: the `nd`-bounded dot and the
 * `__shfl_down_sync` butterfly. Both are removable; neither is a causal mask. `results.md` §1-§3.
 *
 * Two levers, each one a bit-identity-preserving re-spelling, each independently selectable by a
 * MODE bit so the A/B stays single-variable:
 *
 *   MODE bit 0 (SLIM_DOT)  — the score dot's trip count is `nd`, a RUNTIME value, because
 *       `NEG_DIMHALF` asks for 4 of the 8 dims. LLVM therefore keeps the dot ROLLED: 8 scalar
 *       `ds_load_u16` of the staged K row, 8 `v_movrels_b32` reads of a dynamically indexed `q[]`,
 *       8 `s_waitcnt lgkmcnt(0)`, 8 back edges. The PV dot, whose trip count is the literal 8,
 *       compiles as ONE `ds_load_b128` plus eight unrolled convert+FMA pairs. This lever gives the
 *       score dot that shape and moves `NEG_DIMHALF` into a second copy of the loop behind a uniform
 *       branch, so the injection keeps its exact 4-term semantics without its trip count deciding
 *       the hot path's shape. Eight products, same serial t order, same F32 add chain.
 *
 *   MODE bit 1 (SLIM_FOLD) — four of the five `__shfl_down_sync` steps of `warp_sum` compile to
 *       `ds_bpermute_b32` plus a participation select, a permute-address add, an out-of-range
 *       compare, and an `exec == mask` guard with its slow-path branch. The five quantities involved
 *       depend only on the lane number and the delta, so they are loop-invariant and hoisted out of
 *       the touch loop here; the permute itself is spelled once, as inline `ds_bpermute_b32` with
 *       the same clamp the compiler emits (`delta -> 0` for an out-of-range lane, i.e. the lane
 *       reads itself). Same permute, same add, same tree, same clamp: bit-identical per lane, not
 *       merely at lane 0.
 *
 * What MODE=0 is: `attn_tile_body` from `prototype/attn_decode/attn_decode_kernels.h`, copied, with
 * `if constexpr` scaffolding added and nothing else changed. The shipped header is frozen and is
 * `#include`d rather than edited, so the A/B is one body in one file and the two forms cannot drift.
 *
 * Bit-identity claims, and the witness for each (no device run is claimed here — a worker takes no
 * timings and this tree's device gates are the main agent's):
 *   - lever 0: the expression tree is unchanged; `touch_probe.py --dot` executes both spellings in
 *     F32 with the shipped order and requires bit equality on every one of 200k random touches.
 *   - lever 1: same for the fold, `touch_probe.py --fold`, including the clamp's ±0 corners.
 *   - on device: `attn_valu_slim --slim 1` (or 2/3) must print `ndiff_aout=0 ndiff_gated=0` for
 *     every plan in `HANDOFF.md`, because the only consumer of a folded score is lane 0's store and
 *     the only consumer of the dot is that store too.
 *
 * Build (compile-only; the census needs no device):
 *   hipcc -O3 --offload-arch=gfx1151 --save-temps -c isa_tu.cpp -o isa_tu.o
 * Harness: `attn_valu_slim.hip.cpp`, `--slim 0|1|2|3` = MODE 0 (shipped copy) / 1 (both levers) /
 * 2 (dot only) / 3 (fold only). Device A/B commands: `HANDOFF.md`.
 */
#ifndef ATTN_VALU_SLIM_KERNELS_H
#define ATTN_VALU_SLIM_KERNELS_H

#include "../attn_decode/attn_decode_kernels.h"

#include <algorithm>
#include <cassert>

// MODE bit 0: unrolled score dot. MODE bit 1: guard-free butterfly.
#define VSLIM_DOT 1
#define VSLIM_FOLD 2

// One 128-bit LDS read of the staged K row's 8 BF16 dims, then the serial-in-t dot the reference
// form uses (`prototype/attn_decode/reference.py`, and §05.5). With a compile-time trip count the
// eight terms unroll and `q[]` stays in registers; `ND` is a template parameter precisely so that
// `NEG_DIMHALF` cannot make the hot trip count dynamic again.
template <int ND>
__device__ __forceinline__ float vslim_dot(const hip_bfloat16* k, const float* q) {
    static_assert(ND == 4 || ND == 8, "the only two trip counts this form has");
    const uint4 ku = *reinterpret_cast<const uint4*>(k);   // 8 BF16 = 16 B, in-line with a K row
    const hip_bfloat16* kk = reinterpret_cast<const hip_bfloat16*>(&ku);
    float dot = 0.f;
#pragma unroll
    for (int t = 0; t < ND; ++t) dot += bf(kk[t]) * q[t];   // serial in t, §05.5
    return dot;
}

// `ds_bpermute_b32 dst, addr, src`: read lane `addr/4`'s copy of `src`. This is exactly what
// `__shfl_down_sync` lowers to on GCN; spelling it directly buys the removal of the compiler's
// participation select, its permute-address math and its `exec == mask` guard, all of which are
// loop-invariant here (they depend on `lane` and the delta only).
//
// The `s_waitcnt lgkmcnt(0)` is part of the instruction, not decoration: SIMachineFunctionInfo
// inserts LDS waitcnts for instructions it recognises, and an inline-asm DS op is not one of them,
// so without it the consumer of `got` can issue before the permute lands. That is the one real risk
// this lever carries and `HANDOFF.md` step 2 is the run that would catch it (a racy permute is
// nondeterministic, so the A/B is required twice, and `--neg` runs are unaffected because an
// injection that moves the output moves it every time).
__device__ __forceinline__ float vslim_permute_add(float v, unsigned addr) {
    unsigned got;
    const unsigned bits = __builtin_bit_cast(unsigned, v);
    asm volatile("ds_bpermute_b32 %0, %1, %2\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(got)
                 : "v"(addr), "v"(bits));
    return v + __builtin_bit_cast(float, got);
}

// The permute address `__shfl_down_sync(v, delta)` uses for a 32-lane warp, with the compiler's own
// clamp: an out-of-range lane gets `delta -> 0`, i.e. it reads itself. Reproducing the clamp exactly
// is the difference between "bit-identical at lane 0, which is all the consumer reads" and
// "bit-identical in every lane", and the second is cheaper to argue about.
__device__ __forceinline__ unsigned vslim_permute_addr(int lane, int delta) {
    const int src = (lane + delta < warpSize) ? lane + delta : lane;
    return 4u * (unsigned)src;
}

// The 5-step butterfly, one warp per head, 32 lanes x 8 dims. Delta 16 is left to the compiler in
// both MODEs, because there it emits a `ds_swizzle_b32` with an immediate permute and no address
// math at all — the cheap case, and the one case where hand-rolling could only lose.
template <int MODE>
__device__ __forceinline__ float vslim_fold(float v, const unsigned* pa) {
    if constexpr ((MODE & VSLIM_FOLD) == 0) return warp_sum(v);   // the shipped spelling, verbatim
    v = v + __shfl_down_sync(FULL_MASK, v, 16);
    for (int i = 0; i < 4; ++i) v = vslim_permute_add(v, pa[i]);   // deltas 8,4,2,1
    return v;
}

template <int MODE>
__device__ __forceinline__ void attn_slim_body(const CoreArgs a) {
    __shared__ hip_bfloat16 s_sc[ATTN_TILE_G][MAXSC];   // exact-BF16 score staging, one row per head
    __shared__ hip_bfloat16 s_k[ATTN_TILE_L * DH];      // K rows, L lanes deep (pass 1)
    __shared__ hip_bfloat16 s_v[ATTN_TILE_L * DH];      // V rows, L lanes deep (pass 2)
    const int row = gridDim.z - 1 - blockIdx.z;   // widest row first: §2's dispatch tail, priced
    const int kv_read = tile_kv_of(blockIdx.x, a.neg);  // group whose K/V this block reads
    const int kv_write = blockIdx.x;                    // group whose lanes this wave appends
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;               // == ATTN_TILE_G, by the static_assert
    const int nrows = nrows_of(a);
    const int row_pos = a.pos + row;
    const int nrow = lanes_of(a, row);
    const int s_row = blocks_for_row(a, row);           // THE shipped form's rule, one call
    const int span = (nrow + s_row - 1) / s_row;        // ... and the shipped form's slice
    const int blo = blockIdx.y * span;
    const int bhi = std::min(nrow, blo + span);
    const int ns = bhi - blo;                           // lanes this block owns of this row
    const int tile = (a.tile > 0 && a.tile <= TILE) ? a.tile : TILE;   // bounded by `float e[TILE]`
    const int nd = (a.neg == NEG_DIMHALF) ? 4 : 8;      // of the 8 BF16 a lane owns
    if (row >= nrows || blockIdx.y >= s_row || ns <= 0) return;        // before any barrier: uniform
    const int h = w % ATTN_TILE_G;                      // warp w owns query head h of the group
    const int head = ATTN_TILE_G * kv_write + h;

    // The four permute addresses for the folded deltas. Loop-invariant by construction: they are
    // functions of `lane` alone. Held in 4 VGPRs across the pass-1 loops, which is the price of
    // MODE bit 1 and the reason the resource census in `results.md` §4 is part of the evidence.
    unsigned pa[4];
    for (int i = 0; i < 4; ++i) pa[i] = vslim_permute_addr(lane, 8 >> i);

    // This head's query, in registers: lane owns dims lane*8..lane*8+7 — the same 512-byte read a K
    // or V row needs, and the same chunk `attn_core` scores with.
    float q[8];
    {
        const uint4 u = *reinterpret_cast<const uint4*>(
            a.qrope + (size_t)ROW_H * row + (size_t)head * DH + lane * 8);
        const hip_bfloat16* qq = reinterpret_cast<const hip_bfloat16*>(&u);
        for (int t = 0; t < 8; ++t) q[t] = bf(qq[t]);
    }

    // ---- pass 1: stage K a sub-block of L lanes at a time — ONE warp per lane, so the G heads read
    // one load each — then score those lanes for this warp's head. Barriers are block-uniform: every
    // quantity that bounds them (`s_row`, `span`, `ns`, `tile`) is a per-block value, not a per-warp
    // one, which is the other half of what M = 1 buys.
    for (int s_lo = 0; s_lo < ns; s_lo += ATTN_TILE_L) {
        const int s_hi = std::min(ns, s_lo + ATTN_TILE_L);
        for (int rr = s_lo + w; rr < s_hi; rr += nw) {
            const int slot = rr - s_lo;      // a SLOT in `s_k`, in [0, ATTN_TILE_L). Indexing the
                                             // stage by the slice-absolute `rr` walks off the array
                                             // and corrupts its neighbour: §13/§14, and
                                             // `attn_tile_model.py lds` reads this expression.
            const int j = blo + rr;
            const hip_bfloat16* ksrc =
                (j == row_pos)
                    ? a.krope + (size_t)ROW_KV * row + (size_t)kv_write * DH
                    : a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
            *reinterpret_cast<uint4*>(&s_k[slot * DH + lane * 8]) =
                *reinterpret_cast<const uint4*>(ksrc + lane * 8);
        }
        __syncthreads();                                   // K sub-block published

        // One (row, lane, head) touch = one dot + one fold + one BF16 round. The causal bound is the
        // `rr < s_hi` limit in both MODEs, as it already is in the shipped form: there is no
        // per-element causal guard here to convert into a loop bound, which is finding R1.
        auto publish = [&](int rr, float dot) {
            dot = vslim_fold<MODE>(dot, pa);
            // One BF16 round, then the scale. Exact in BF16 because SCALE is 2^-4, which is what lets
            // the staging be 2 bytes and still read back the identical F32.
            if (lane == 0) s_sc[h][rr] = fb(bf(fb(dot)) * SCALE);
        };
        if constexpr (MODE & VSLIM_DOT) {
            // The dim-half injection gets its own copy of the loop rather than a runtime trip count.
            // `nd` is warp-uniform (it is a function of `a.neg`), so this branch costs one uniform
            // compare outside the touch loop and buys a constant trip count inside it.
            if (nd == 8) {
                for (int rr = s_lo; rr < s_hi; ++rr) {
                    publish(rr, vslim_dot<8>(&s_k[(rr - s_lo) * DH + lane * 8], q));
                }
            } else {
                for (int rr = s_lo; rr < s_hi; ++rr) {
                    publish(rr, vslim_dot<4>(&s_k[(rr - s_lo) * DH + lane * 8], q));
                }
            }
        } else {
            for (int rr = s_lo; rr < s_hi; ++rr) {
                const hip_bfloat16* k = &s_k[(rr - s_lo) * DH + lane * 8];
                float dot = 0.f;
                for (int t = 0; t < nd; ++t) dot += bf(k[t]) * q[t];   // serial in t, §05.5
                publish(rr, dot);
            }
        }
        __syncthreads();                                   // s_sc published for pass 2
    }

    // ---- pass 2: this warp's head alone, in the shipped form's order, staging V the same block-level
    // way. The state is warp-private, so there is no cross-warp hand-off and nothing to leak between
    // heads; `one_pass` is `attn_core`'s own condition, not a paraphrase of it.
    //
    // UNCHANGED IN EVERY MODE. It is the half of the touch that D20's 24 % does not name (the V
    // conversions and the 8 PV FMAs are inside the 72.6 "by subtraction"), and the PV dot cannot be
    // touched by either lever here: `acc[u] += wt * bf(v[u])` needs eight accumulators, so there is
    // nothing for a packed instruction to share, and its eight conversions are already dual-issued
    // with the FMAs they feed (`results.md` §3).
    float m = 0.f, l = 0.f, acc[8];
    for (int u = 0; u < 8; ++u) acc[u] = 0.f;
    if (s_row == 1 && a.full) {
        // The reference form: max, then sum-of-exp, then the BF16-rounded normalised weight and P·V,
        // each serial in index order, over the whole row.
        for (int t = 0; t < ns; ++t) m = fmaxf(m, bf(s_sc[h][t]));
        for (int t = 0; t < ns; ++t) l += expf(bf(s_sc[h][t]) - m);
        for (int s_lo = 0; s_lo < ns; s_lo += ATTN_TILE_L) {
            const int s_hi = std::min(ns, s_lo + ATTN_TILE_L);
            for (int rr = s_lo + w; rr < s_hi; rr += nw) {
                const int slot = rr - s_lo;
                const int j = blo + rr;
                const hip_bfloat16* vsrc =
                    (j == row_pos)
                        ? a.vg + (size_t)ROW_KV * row + (size_t)kv_write * DH
                        : a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
                *reinterpret_cast<uint4*>(&s_v[slot * DH + lane * 8]) =
                    *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            }
            __syncthreads();
            for (int rr = s_lo; rr < s_hi; ++rr) {
                const float wt = bf(fb(expf(bf(s_sc[h][rr]) - m) / l));
                const hip_bfloat16* v = &s_v[(rr - s_lo) * DH + lane * 8];
                for (int u = 0; u < 8; ++u) acc[u] += wt * bf(v[u]);   // serial in t, §05.5
            }
            __syncthreads();                               // before the next stage overwrites
        }
        // The weights were already normalised and rounded (that IS the reference form), so the
        // epilogue must not divide again — and on the streaming route below the divide is left to
        // `attn_combine`, so a weight is divided exactly once either way.
        l = 1.0f;
    } else {
        // The streaming form: `attn_core`'s tile-of-`tile` loop over the SAME [blo, bhi) it would be
        // handed, including the `expf(m - mn)` rescale algebra, so this route lands on the bits the
        // shipped form already streams instead of opening a second set.
        for (int base = blo; base < bhi; base += tile) {
            const int nt = std::min(tile, bhi - base);
            for (int rr = base + w; rr < base + nt; rr += nw) {
                const int slot = rr - base;                // a SLOT in `s_v`; `nt <= ATTN_TILE_L`
                const int j = rr;
                const hip_bfloat16* vsrc =
                    (j == row_pos)
                        ? a.vg + (size_t)ROW_KV * row + (size_t)kv_write * DH
                        : a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
                *reinterpret_cast<uint4*>(&s_v[slot * DH + lane * 8]) =
                    *reinterpret_cast<const uint4*>(vsrc + lane * 8);
            }
            __syncthreads();
            float mt = -1e30f;
            for (int slot = 0; slot < nt; ++slot) mt = fmaxf(mt, bf(s_sc[h][base - blo + slot]));
            const float mn = fmaxf(m, mt);
            const float rescale = expf(m - mn);
            float e[TILE], ls = 0.f;
            for (int slot = 0; slot < nt; ++slot) {
                e[slot] = expf(bf(s_sc[h][base - blo + slot]) - mn);
                ls += e[slot];
            }
            l = l * rescale + ls;
            for (int u = 0; u < 8; ++u) acc[u] *= rescale;
            for (int slot = 0; slot < nt; ++slot) {
                const hip_bfloat16* v = &s_v[slot * DH + lane * 8];
                for (int u = 0; u < 8; ++u) acc[u] += e[slot] * bf(v[u]);
            }
            m = mn;
            __syncthreads();                               // before the next stage overwrites
        }
    }

    // ---- epilogue: identical arithmetic to `finish_head`, 8 dims per lane, and the SAME output
    // destinations and partial slot expression the shipped form uses.
    if (s_row == 1) {
        hip_bfloat16 o[8], g8[8];
        const uint4 gu = *reinterpret_cast<const uint4*>(
            a.qg + (size_t)ROW_Q * row + g_off(head, a.neg) + lane * 8);
        const hip_bfloat16* gg = reinterpret_cast<const hip_bfloat16*>(&gu);
        for (int u = 0; u < 8; ++u) {
            const float av = bf(fb(acc[u] / l));
            o[u] = fb(av);
            const float g = bf(gg[u]);
            g8[u] = fb(av * bf(fb(1.0f / (1.0f + expf(-g)))));
        }
        *reinterpret_cast<uint4*>(a.aout + (size_t)ROW_H * row + (size_t)head * DH + lane * 8) =
            *reinterpret_cast<uint4*>(o);
        *reinterpret_cast<uint4*>(a.gout + (size_t)ROW_H * row + (size_t)head * DH + lane * 8) =
            *reinterpret_cast<uint4*>(g8);
    } else {
        // One partial per (row, head, lane block) — `attn_core`'s expression and `attn_combine`'s
        // reader, which is the point: the two forms differ in who loads K/V, and a merge that guessed
        // about a private layout is how §19's aliasing happened. Striding by the LAUNCH's `split`
        // (not by the row's `s_row`) is what makes a row's slab contiguous and allocatable from one
        // expression; a row that used fewer blocks simply leaves the tail unwritten.
        float* pp = a.part + (((size_t)row * N_Q + head) * a.split + blockIdx.y) * (DH + 2);
        if (lane == 0) { pp[0] = m; pp[1] = l; }
        for (int u = 0; u < 8; ++u) pp[2 + lane * 8 + u] = acc[u];
    }
}

// The four entry points. MODE 0 is the shipped body; a zero-initialised CoreArgs still selects
// `attn_core`, so this file never replaces the one-row decode path.
__global__ void attn_slim0_wave(CoreArgs a) { attn_slim_body<0>(a); }
__global__ void attn_slim1_wave(CoreArgs a) { attn_slim_body<VSLIM_DOT | VSLIM_FOLD>(a); }
__global__ void attn_slim2_wave(CoreArgs a) { attn_slim_body<VSLIM_DOT>(a); }
__global__ void attn_slim3_wave(CoreArgs a) { attn_slim_body<VSLIM_FOLD>(a); }

inline int attn_valu_slim_mode_ok(int mode) { return mode >= 0 && mode <= 3; }

// The MODE each `--slim` value selects, in one place so the harness and the census cannot drift.
inline int attn_valu_slim_mode_of(int slim) {
    return slim == 1 ? (VSLIM_DOT | VSLIM_FOLD) : slim == 2 ? VSLIM_DOT : slim == 3 ? VSLIM_FOLD : 0;
}

#endif  // ATTN_VALU_SLIM_KERNELS_H
