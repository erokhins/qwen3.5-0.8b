/*
 * gdn_rec_unroll4_kernels.h — §04.4's shipped recurrence, RE-SPELLED: same operations, same order,
 * same values, with `#pragma unroll 4` on the two state walks. This is the engine's copy of
 * `prototype/gdn_rec_isa/gdn_rec_isa_kernels.h::recx_shape_kernel`, which is the probe that measured
 * **1.149–1.158x over the shipped spelling at launch level while moving zero bits**
 * (`prototype/gdn_rec_isa/results.md` §10, decision log D32 of
 * `progress/2026-10-16/prefill-decision-log-2026-10-16.md`).
 *
 * WHY A SPELLING IS WORTH A FILE AT ALL, when the census said instructions do not matter here.
 * `prototype/gdn_rec_isa/isa_census.py` bills the shipped kernel at 1,523 slots per token-step and
 * D25 measured a slot elasticity of ~0.8 on this very kernel (a −2.0 % bill returned −1.6 % of time).
 * This kernel removes ~7.7 % of the bill, which by that elasticity predicts ~6 %; the burst returned
 * ~15 %. So the win is NOT the instruction count — it is that four bodies per trip fold the index
 * arithmetic out of the serial chain, which is the one axis D25's fusion-and-hand-fold probe never
 * touched (D32's restatement: "the axis that pays is spelling, which a census cannot see because a
 * spelling change looks like a no-op to an instruction count"). That is also why this file is behind a
 * runtime knob and not simply substituted for `rec_kernel`: the burst ran on a 1 MiB single-layer state
 * that stays resident for 128 tokens, the engine walks 18 layers per token, and D28 is the precedent
 * that two of three burst winners were in-situ losers. The knob makes the in-situ A/B one flag; the
 * default stays `ship` until that A/B is read.
 *
 * BIT-IDENTITY IS THE ADMISSION, NOT A HOPE. `rec_kernel`'s two dots are ascending F32 sums that
 * cancel — §04.4 measured 1 element in 286 of `o` losing 84 % of its magnitude to that cancellation —
 * so re-associating them is a different function, not a rounding choice. What changes here is the LOOP
 * SPACING and nothing else:
 *   * one accumulator each (`r`, `acc`), NOT four partials. `recx_unchain_kernel` is the four-partial
 *     form and it is a numerics change (admitted in `prototype/gdn_rec_isa/` only, inside the §10.1
 *     GDN band, never here);
 *   * the terms arrive in the shipped order. Trip `i` adds i+0, i+1, i+2, i+3 in that order and
 *     `i += 4`, so the sum is `((…((t0 + t1) + t2) + t3) + t4 …)` — `rec_kernel`'s tree term for term;
 *   * every expression is transcribed character for character, including which multiply lands inside
 *     which add. That is the D25 hoisted-`q̃` class of bug
 *     (`engine/gdn_collapse_kernels.h`'s note, `prototype/gdn_collapse/results.md` §2): writing
 *     `qv * (rsqrt * SCALE)` where the shipped kernel writes `(qv * rsqrt) * SCALE` is a different
 *     product tree, and on device it moved 93,522 of 131,072 `o` elements while leaving the state
 *     bit-exact. A CPU mirror cannot see it — two sides that share a formula agree — so the rule here
 *     is textual: diff this body against `recx_shape_kernel`, do not "improve" it.
 *
 * `#pragma unroll 4` IS MANDATORY ON BOTH LOOPS, and that is a measured constraint, not style. Without
 * it LLVM fully unrolls loop A (141 static state loads against 29) and the two forms are no longer the
 * same listing shape; `traffic_census.py --volume-check` refused that listing outright, which is the
 * right way for a geometry change to fail (`prototype/gdn_rec_isa/results.md` §9). The pragma is on
 * loop B too for the same reason: the pair must stay bill-matched with the probe the number came from.
 *
 * WHAT THIS FILE IS NOT. It is not a second recurrence and it does not replace the shipped kernel —
 * `prototype/gdn_decode/gdn_decode_kernels.h::rec_kernel` stays compiled and stays the default
 * (`engine/kernels.h::GdnRecSpell`, `decode_run --gdn-spell`). Both forms live in one binary so the
 * in-situ A/B is one window (`AGENTS.md`: a ruler re-taken in a different window is not an A/B). The
 * launch geometry is `rec_kernel`'s exactly — `<<<NH, D>>>`, one block per value head, one thread per
 * `d_v` column, same grid, same LDS, same operands — because a timing delta between two launch plans
 * measures the plans.
 */
#pragma once

#include "prototype/gdn_decode/gdn_decode_kernels.h"   // rec_kernel, block_sum, D, NH, Q_SCALE

// One GDN layer, one token, one (token, layer) launch — the same signature as `rec_kernel`, so the
// launcher's choice of spelling is one call and cannot move an address. Body is
// `recx_shape_kernel`'s; keep them identical.
__global__ void rec_unroll4_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                   const float* __restrict__ v, const float* __restrict__ g,
                                   const float* __restrict__ beta, float* __restrict__ S,
                                   float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;                       // 128 threads, thread j owns column j
    __shared__ float ks[D], qs[D], red[32];

    float kq = k[h * D + j], qq = q[h * D + j];
    float kn = kq * rsqrtf(block_sum(kq * kq, red) + 1e-6f);
    __syncthreads();
    float qn = qq * rsqrtf(block_sum(qq * qq, red) + 1e-6f) * Q_SCALE;
    __syncthreads();
    ks[j] = kn;
    qs[j] = qn;
    __syncthreads();

    float gam = expf(g[h]), bet = beta[h];
    float* Sh = S + (size_t)h * D * D;

    float r = 0.f;
#pragma unroll 4                      // keep the loop: without the pragma LLVM fully unrolls this one
    for (int i = 0; i < D; i += 4) {  // side and the two forms stop being bill-matched (see the header)
        r += Sh[(i + 0) * D + j] * ks[i + 0];
        r += Sh[(i + 1) * D + j] * ks[i + 1];
        r += Sh[(i + 2) * D + j] * ks[i + 2];
        r += Sh[(i + 3) * D + j] * ks[i + 3];
    }
    float dv = bet * (v[h * D + j] - gam * r);

    float acc = 0.f;
#pragma unroll 4
    for (int i = 0; i < D; i += 4) {
        float s0 = gam * Sh[(i + 0) * D + j]; s0 += ks[i + 0] * dv; Sh[(i + 0) * D + j] = s0;
        acc += qs[i + 0] * s0;
        float s1 = gam * Sh[(i + 1) * D + j]; s1 += ks[i + 1] * dv; Sh[(i + 1) * D + j] = s1;
        acc += qs[i + 1] * s1;
        float s2 = gam * Sh[(i + 2) * D + j]; s2 += ks[i + 2] * dv; Sh[(i + 2) * D + j] = s2;
        acc += qs[i + 2] * s2;
        float s3 = gam * Sh[(i + 3) * D + j]; s3 += ks[i + 3] * dv; Sh[(i + 3) * D + j] = s3;
        acc += qs[i + 3] * s3;
    }
    o[h * D + j] = acc;
}
