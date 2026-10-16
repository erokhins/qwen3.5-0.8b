/*
 * attn_slim_kernels.h — the engine's own copy of the tiled attention wave, with the two per-touch
 * instruction levers applied: `attn_slim_wave` scores and folds one (row, lane, head) touch in 45 issue
 * slots where the shipped form spends 116, and it is bit-identical while doing it
 * (`prototype/attn_valu_slim/results.md` §2 and §7).
 *
 * WHY A COPY RATHER THAN AN EDIT. The engine shipped this kernel by including the prototype's header:
 * `engine/kernels_attn.cpp` launches `attn_tile_wave`, which is `attn_tile_body` in
 * `prototype/attn_decode/attn_decode_kernels.h`. That header declares itself frozen — "Nothing here may
 * be edited for the engine's convenience", the graduation rule in `prototype/README.md` — and it is also
 * the form the §18/§19/§20 witnesses are made against, while `prototype/attn_m2/` compares
 * prototype-to-prototype through it. Porting into the shipping path therefore meant the engine growing
 * its own copy, not un-freezing the reference the port is proved against. The body below is
 * `attn_tile_body` copied and changed in exactly two places, both inside the pass-1 score loop; the
 * census command at the end of this note is what enforces that sentence, not this sentence.
 * (`engine/gdn_collapse_kernels.h` is the precedent for an engine-local kernel header living beside the
 * prototype includes; `engine/kernels_attn.cpp` including a prototype header was the anomaly.)
 *
 * KEEPING THE COPY HONEST. BOTH kernels are compiled into the TU unconditionally. The runtime switch
 * (`--attn-touch`, `AttnLayerArgs::wave_touch`) decides only which one the LAUNCHER calls, never which
 * one exists, so `isa_census.py` on the engine listing prints the shipped and slim rows out of one
 * binary:
 *
 *     REPO=$PWD; mkdir -p /tmp/attn_census && cd /tmp/attn_census
 *     hipcc -O3 -Wno-unused-command-line-argument -I"$REPO" -I"$REPO/engine" --save-temps \
 *         -c "$REPO/engine/kernels_attn.cpp" -o k.o; cd "$REPO"
 *     .venv/bin/python prototype/attn_valu_slim/isa_census.py \
 *         --asm /tmp/attn_census/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s \
 *         --kernels attn_tile_wave,attn_slim_wave
 *
 * (`--save-temps` writes the listing into the CWD, not beside `-o`, hence the scratch directory.)
 *
 * Expected, and this is the drift guard: `attn_tile_wave` 116 slots / 55 VALU / 14 LDS / 47 SALU per
 * touch, `attn_slim_wave` 45 / 28 / 7 / 10. A shipped row that moves is the prototype's kernel changing
 * under this copy — which is the failure a "copied from line N" comment cannot catch and a census can.
 * Like every op header in this tree this file is included by exactly ONE translation unit.
 *
 * THE TWO LEVERS. Neither needs a numerics decision, neither costs occupancy, and both were proved on
 * the prototype harness before being transplanted here:
 *
 *   LEVER 1 — THE DOT'S TRIP COUNT BECOMES COMPILE-TIME. `nd` is `(a.neg == NEG_DIMHALF) ? 4 : 8`, a
 *     runtime value, so LLVM kept the score dot rolled: per element one scalar `ds_load_u16` of the
 *     staged K row, one `v_movrels_b32_e32` for the dynamically indexed `q[t]`, one widen, one FMA, one
 *     `s_waitcnt lgkmcnt(0)` and a back edge — eight of each per touch. The PV dot two hundred lines
 *     below, whose trip count is the literal 8, compiles as ONE `ds_load_b128` plus eight unrolled
 *     convert+FMA pairs that dual-issue with the converts they feed. `attn_slim_dot<ND>` gives the score
 *     dot that shape, and `NEG_DIMHALF` moves into a second copy of the loop behind a uniform branch
 *     (`nd` is a function of `a.neg`, so it is warp-uniform) instead of deciding the hot path's shape.
 *     Same eight products, same serial-in-`t` order, same single F32 accumulator. Census: dot.lds-load
 *     8 → 1, dot.q-indexed 8 → 0, wait.lgkmcnt 13 → 6.
 *
 *   LEVER 2 — THE WARP REDUCE IS FOLDED BY HAND, WITH ITS OWN WAIT. Four of `warp_sum`'s five
 *     `__shfl_down_sync` steps lower to `ds_bpermute_b32` plus four loop-invariant guards: a
 *     participation select, a permute-address add, an out-of-range compare, and an `exec == mask` test
 *     with its slow-path branch — 16 `s_cmp_eq_u32 exec_lo, -1` / `s_cbranch_scc0` pairs per touch, more
 *     than the dot they protect. Every one of those four quantities depends on `lane` and the delta only,
 *     so they are computed once before the pass-1 loops (`perm_addr`) and the permute is spelled once, as
 *     inline `ds_bpermute_b32`. The out-of-range clamp is the compiler's own (`delta -> 0`, i.e. a lane
 *     reads itself), which is why the claim is "bit-identical in every lane" and not the weaker
 *     "bit-identical at lane 0, which is all the consumer reads". Delta 16 stays `__shfl_down_sync`: there
 *     LLVM already emits a `ds_swizzle_b32` with an immediate permute and no address math or guard, so
 *     hand-rolling could only lose. Census: fold.exec-guard 16 → 2, fold.range-select 7 → 0,
 *     fold.addr-math 4 → 0; fold.permute 5 and fold.add 4 unchanged, because those ARE the reduction.
 *     THE ONE REAL RISK IN THIS FILE: `SIInsertWaitcnts` inserts LDS waits for the instructions it
 *     recognises and an inline-asm DS op is not one of them, so the `s_waitcnt lgkmcnt(0)` is written
 *     inside the asm block. A missing or misplaced wait is not a wrong-but-stable number, it is
 *     NONDETERMINISM — hence the gate requirement that the same plans be run twice and the logs diffed.
 *
 * WHAT NEITHER LEVER TOUCHES, so nobody re-books it as a win later: the BF16→F32 widening stays (two
 * BF16 operands cannot multiply without being widened, and the only instructions that widen for free are
 * `v_dot2_f32_bf16`, rejected in D20 because it would spend m2's bit-identity admission, and WMMA) — the
 * census moves 9 → 8, not 9 → 0. Pass 2 is untouched: the V widens, the eight PV FMAs, `expf`, the
 * rescale. Whole-kernel static ops go UP (1454 → 1565) because the score loop is now emitted twice;
 * per-touch slots are what fall, and what prices the route is the paired burst, not the count.
 *
 * MEASURED, on the harness this was written in: `ndiff_aout=0 ndiff_gated=0` on all 12 plans,
 * byte-identical logs across two runs (the `s_waitcnt` race check), 0 disagreements over 36 injection
 * cells including `dimhalf`, and 1.536 / 1.547 / 1.521 / 1.572× at wave64-1024 / 2048 / 4096 /
 * wave16-4096 (`prototype/attn_valu_slim/results.md` §7). Resources, same census: VGPRs 47 → 47, LDS
 * 12,288 B unchanged, occupancy unchanged at 16 waves/SIMD, SGPRs 52 → 54.
 */
#pragma once

#include <cstdint>

// The shipped body's own vocabulary (`CoreArgs`, `bf`, `fb`, `warp_sum`, `ATTN_TILE_*`, `MAXSC`,
// `lanes_of`, `blocks_for_row`, `g_off`, …) comes from the op header this file copies out of, and it must
// already be in scope: include `prototype/attn_decode/attn_decode_kernels.h` BEFORE this one, which is
// what `engine/kernels_attn.cpp` does. Getting the order wrong is a compile error, not a silent hole.
// There is no include-guard macro to test — the op header uses `#pragma once` — so the probe is a macro
// only its tiled section defines.
#ifndef ATTN_TILE_THREADS
#error "attn_slim_kernels.h is a copy-with-levers of attn_tile_body and needs the op header it copies already included (see the note above)"
#endif

// One staged K row, eight BF16 dims per lane, compile-time trip count. `ND` is a template parameter
// precisely so that `NEG_DIMHALF` cannot make the hot trip count dynamic again: the whole point of the
// lever is that the injection keeps its exact 4-term semantics without its runtime value deciding the
// shape of the loop every clean token runs through.
template <int ND>
__device__ __forceinline__ float attn_slim_dot(const hip_bfloat16* k, const float* q) {
    static_assert(ND == 4 || ND == 8, "the only two trip counts this form has");
    const uint4 ku = *reinterpret_cast<const uint4*>(k);   // 8 BF16 = 16 B, in line with a K row
    const hip_bfloat16* kk = reinterpret_cast<const hip_bfloat16*>(&ku);
    float dot = 0.f;
#pragma unroll
    for (int t = 0; t < ND; ++t) dot += bf(kk[t]) * q[t];   // serial in t, §05.5 — same order, same tree
    return dot;
}

// `ds_bpermute_b32 dst, addr, src`: read lane `addr/4`'s copy of `src`. That is exactly what
// `__shfl_down_sync` lowers to on GCN, and spelling it directly buys the removal of the compiler's
// participation select, its permute-address math and its `exec == mask` guard — all loop-invariant here,
// because they depend on `lane` and the delta and nothing else.
//
// The `s_waitcnt lgkmcnt(0)` is part of the instruction, not decoration: SIMachineFunctionInfo inserts
// LDS waitcnts for instructions it recognises and an inline-asm DS op is not one of them, so without it
// the consumer of `got` can issue before the permute lands. See this file's header note: the failure
// mode of getting this wrong is run-to-run disagreement, which is why the device gate is run twice.
__device__ __forceinline__ float attn_slim_permute_add(float v, unsigned addr) {
    unsigned got;
    const unsigned bits = __builtin_bit_cast(unsigned, v);
    asm volatile("ds_bpermute_b32 %0, %1, %2\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(got)
                 : "v"(addr), "v"(bits));
    return v + __builtin_bit_cast(float, got);
}

// The permute address `__shfl_down_sync(v, delta)` uses for a 32-lane warp, with the compiler's own
// clamp: an out-of-range lane gets `delta -> 0`, i.e. it reads itself. Reproducing the clamp is the
// difference between "bit-identical at lane 0" and "bit-identical in every lane", and the second is
// cheaper to argue about.
__device__ __forceinline__ unsigned attn_slim_perm_addr(int lane, int delta) {
    const int src = (lane + delta < warpSize) ? lane + delta : lane;
    return 4u * (unsigned)src;
}

// The 5-step butterfly, one warp per head, 32 lanes x 8 dims. Delta 16 is left to the compiler because
// there it emits `ds_swizzle_b32` with an immediate permute and no address math at all — the cheap case,
// and the one case where hand-rolling could only lose. `perm_addr` holds the four folded deltas
// (8, 4, 2, 1), which are per-lane constants and therefore hoisted out of the touch loops by the caller.
__device__ __forceinline__ float attn_slim_fold(float v, const unsigned* perm_addr) {
    v = v + __shfl_down_sync(FULL_MASK, v, 16);
    for (int i = 0; i < 4; ++i) v = attn_slim_permute_add(v, perm_addr[i]);   // deltas 8, 4, 2, 1
    return v;
}

// ---- the copy. Everything below is `prototype/attn_decode/attn_decode_kernels.h`'s `attn_tile_body`,
// transcribed character for character apart from the two lever sites inside the pass-1 score loop, and
// the four hoisted permute addresses lever 2 needs. Association is load-bearing here: an expression that
// is algebraically equal but parenthesised differently is a different number on this device
// (`engine/gdn_collapse_kernels.h`'s q̃ note — a hoisted `SCALE`-shaped constant moved 93,522 of 131,072
// elements while leaving the state bit-exact), and a CPU mirror cannot see either the contraction or the
// association. Do not "tidy" this body.
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

    // LEVER 2, part 1 — the four folded permute addresses, hoisted. `lane` is constant for the lifetime
    // of the thread, so these four VGPRs are loop-invariant across every touch this block makes; leaving
    // them inline is what made the compiler recompute the address, the participation select, the
    // out-of-range compare and the `exec == mask` guard on every one of those touches.
    unsigned perm_addr[4];
    for (int i = 0; i < 4; ++i) perm_addr[i] = attn_slim_perm_addr(lane, 8 >> i);   // 8, 4, 2, 1

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
    // one, which is the other half of what M=1 buys.
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
        // `rr < s_hi` loop limit in BOTH forms, exactly as it is in the shipped body: this form has no
        // per-element causal guard to convert into a bound (`prototype/attn_valu_slim/results.md` §1
        // retires D20's "causal-mask guards" reading — the 20 VALU were shuffle-lowering, not masking).
        auto publish = [&](int rr, float dot) {
            dot = attn_slim_fold(dot, perm_addr);                    // LEVER 2, part 2
            // One BF16 round, then the scale. Exact in BF16 because SCALE is 2^-4, which is what lets
            // the staging be 2 bytes and still read back the identical F32.
            if (lane == 0) s_sc[h][rr] = fb(bf(fb(dot)) * SCALE);
        };
        // LEVER 1 — `nd` is warp-uniform (a function of `a.neg`), so the dim-half injection gets its own
        // copy of the loop behind one uniform compare instead of lending this one a runtime trip count.
        // Which is how the shipped form ends up rolling the dot: 8 scalar `ds_load_u16`s, 8
        // `v_movrels_b32_e32`s for the dynamically indexed `q[t]` and 8 back edges per touch, where the
        // PV loop below — trip count literal 8 — gets one `ds_load_b128` and eight dual-issued pairs.
        if (nd == 8) {
            for (int rr = s_lo; rr < s_hi; ++rr) {
                publish(rr, attn_slim_dot<8>(&s_k[(rr - s_lo) * DH + lane * 8], q));
            }
        } else {
            for (int rr = s_lo; rr < s_hi; ++rr) {
                publish(rr, attn_slim_dot<4>(&s_k[(rr - s_lo) * DH + lane * 8], q));
            }
        }
        __syncthreads();                                   // s_sc published for pass 2
    }

    // ---- pass 2: this warp's head alone, in the shipped form's order, staging V the same block-level
    // way. The state is warp-private, so there is no cross-warp hand-off and nothing to leak between
    // heads; `one_pass` is `attn_core`'s own condition, not a paraphrase of it.
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

// The engine's wave entry point for the slim spelling. Same `CoreArgs`, same grid, same block extent,
// same `part` slot expression as `attn_tile_wave` — the two differ only in the two spellings above, so a
// launcher can pick between them without a second launch plan and `attn_combine` needs no branch (see
// `engine/kernels_attn.cpp`'s note on why a merge that guesses about a private layout is how §19's
// aliasing happened).
__global__ void attn_slim_wave(CoreArgs a) { attn_slim_body(a); }
