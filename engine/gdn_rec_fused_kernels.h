/*
 * gdn_rec_fused_kernels.h — §04.4's shipped recurrence, RE-SPELLED AGAIN: the same operations, in the
 * same order, with the two block reductions of the q/k norm prologue FOLDED INTO ONE pass over shared
 * memory, on top of the `#pragma unroll 4` state walks that already ship. This is the engine's copy of
 * `prototype/gdn_rec_isa/gdn_rec_isa_kernels.h::rec2c4_kernel` (MODE 2c4), the merged form that
 * cleared D46's compile-only census gate and then D50's device admission
 * (`prototype/gdn_rec_isa/results.md` §11, `bench/performance-status.md` §19, decision log D50 of
 * `progress/2026-10-16/prefill-decision-log-2026-10-16.md`).
 *
 * WHAT IS TRANSCRIBED FROM WHERE, and the rule that comes with it. Bit-identity is owed to the shipped
 * SOURCE TEXT, not to the algebra, so the two halves of this body have two different parents and both
 * must stay character-for-character:
 *   * prologue — `prototype/gdn_rec_isa/gdn_rec_isa_kernels.h::rec2c_kernel`'s (MODE 2c): the
 *                `block_sum_pair_comp` call, now named `block_sum_pair_f32` for the engine, plus the
 *                `redk`/`redq` arrays and the two barriers that survive the merge;
 *   * both state walks — `engine/gdn_rec_unroll4_kernels.h::rec_unroll4_kernel`, verbatim, including
 *                `#pragma unroll 4` on both loops and the order of every multiply inside every add.
 * Diff this body against those two sources before believing any delta here; do not "improve" it. That
 * discipline is not ceremony: `prototype/gdn_collapse/results.md` §2, quoted in
 * `engine/gdn_rec_unroll4_kernels.h`'s own header, is the case where hoisting one scalar into a
 * different product tree moved 93,522 of 131,072 `o` elements on device while leaving the state
 * bit-exact, and a CPU mirror could not see it because two sides that share a formula agree.
 *
 * THE CENSUS FIGURES, recorded so a future LLVM bump that quietly changes the merge is caught against
 * numbers and not against impressions (`prototype/gdn_rec_isa/merged_census.py`, recorded run
 * `prototype/gdn_rec_isa/merged_census.output.txt`; slots are per token-step per wavefront, 64
 * wavefronts per launch):
 *
 *   bill            1,374 slots/token-step  = 294 prologue + 352 loop A + 728 loop B
 *                   against the shipping `rec_unroll4_kernel`'s 1,406 = 326 + 352 + 728, i.e. −32
 *                   (−2.276 %); every state-walk category is IDENTICAL between the two
 *                   (state.fma 377, state.load 261, state.store 129, state.coef-lds-load 96,
 *                   xlane.permute 20), so the whole delta is barrier bookkeeping in the prologue.
 *   loop B shape    16 state accesses per trip × 8 trips (4 bodies) — THE UNROLL SURVIVED THE MERGE.
 *                   The rolled fallback would read 8 × 16, which is what MODE 2c alone shows.
 *   prologue        5 `s_barrier` (shipped form: 11), 3 permute-then-barrier groups (shipped: 5),
 *                   all 20 cross-lane permutes intact — the fused pair's fingerprint.
 *   VGPRs           59, the same 59 `rec_unroll4_kernel` already needs; vgpr_spill = 0, ScratchSize = 0,
 *                   occupancy 16 waves/SIMD. The merge spends nothing here because the fused pair's
 *                   live ranges retire before the walk's peak live set begins.
 *   LDS             1,280 B/block — +128 B over the shipped 1,152 B, which is the second reduction
 *                   array `redq[32]`. Non-binding at `<<<NH=16, D=128>>>` on this part.
 *   SGPRs           24.
 *
 * WHY THE PLAN BILL CANNOT BE THE CANARY FOR AN A/B OF THIS SPELLING. The launch is
 * `rec_fused_kernel<<<NH, D>>>` with `rec_kernel`'s operand list and its launch count — one launch per
 * (token, layer) — so the flag moves no grid, no address, and no launch. `gdn_rec` is also deliberately
 * NOT a `launch_stats` family (`engine/launch_stats.h`: a recurrence launch has no row width to
 * report, and its launch count is the thing under test), so there is no per-token plan-bill entry and
 * no launcher counter that this spelling can move. A run of a binary that never got the merge therefore
 * prints `plan_match=1 launchstats_match=1` exactly like one that did, which is the AGENTS.md case
 * ("a paired A/B against a stale binary reads as a clean null") and the reason an A/B of this knob
 * takes a BEHAVIOURAL canary inside its own window instead: the binary must accept AND echo
 * `gdn_spell=fused` — a pre-port binary throws in `parse_gdn_rec_spell`, which is loud — and the ISA
 * assertion in `prototype/gdn_rec_isa/merged_census.py --expect merged`, read off the listing of the
 * very binary being timed, is the build-time half.
 *
 * ADMISSION, and where it is re-proved. On device the merged form is bit-identical to the reference
 * spelling (`ndiff_o_vs0=0 ndiff_state_vs0=0`, 4/4 harness cases, `bench/performance-status.md` §19),
 * and `engine/gate.py` section (j) re-runs that claim per fixture through the SHIPPING launch path by
 * dumping every logits row from a run of each spelling and byte-comparing them. A per-fixture green at
 * `--gdn-spell fused` is not that proof — the pins are judged inside bands wide enough to admit a
 * re-associated dot — so the gate, not the band, is the witness.
 *
 * WHAT THIS FILE IS NOT. It is not the default. `GdnRecSpell::kUnroll4` stays the shipping default and
 * `fused` ships as a knob first (D50's order, which is D32 → D33 repeated deliberately: a knob ships as
 * a knob, and the paired in-situ A/B on the `gdn_rec` family row — not a harness burst median of
 * 1.0415 — decides whether the default moves). It is inert under `--gdn-scan collapse`, which submits a
 * different kernel; `decode_run` refuses that pair rather than echoing a spelling that did not run.
 */
#pragma once

#include "prototype/gdn_decode/gdn_decode_kernels.h"   // block_sum's shape, D, NH, Q_SCALE

/* Two values, one pass over shared memory — `block_sum_pair_comp` verbatim, renamed only. Each value
 * keeps its own 5+5 shuffle tree and its own `red[]` array; what is shared is the barrier each stage
 * needs, so the pair costs 3 barriers instead of the shipped two-call form's 8. The two per-call
 * "protect red[0]" barriers of `block_sum` are unnecessary here because each value's caller reads its
 * `red` before this returns and neither array is reused before the next launch. The association of
 * each value's sum is `block_sum`'s, step for step, which is the whole reason this is an admission and
 * not a numerics change. */
__device__ __forceinline__ void block_sum_pair_f32(float v_a, float v_b, float* red_a, float* red_b,
                                                   float* out_a, float* out_b) {
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 16);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 8);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 4);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 2);
    v_a += __shfl_down_sync(0xffffffffffffffffull, v_a, 1);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 16);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 8);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 4);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 2);
    v_b += __shfl_down_sync(0xffffffffffffffffull, v_b, 1);
    if ((threadIdx.x & 31) == 0) {
        red_a[threadIdx.x >> 5] = v_a;
        red_b[threadIdx.x >> 5] = v_b;
    }
    __syncthreads();
    int warps = blockDim.x >> 5;
    float t_a = (threadIdx.x < warps) ? red_a[threadIdx.x] : 0.f;
    float t_b = (threadIdx.x < warps) ? red_b[threadIdx.x] : 0.f;
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 16);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 16);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 8);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 8);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 4);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 4);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 2);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 2);
    t_a += __shfl_down_sync(0xffffffffffffffffull, t_a, 1);
    t_b += __shfl_down_sync(0xffffffffffffffffull, t_b, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        red_a[0] = t_a;
        red_b[0] = t_b;
    }
    __syncthreads();
    *out_a = red_a[0];
    *out_b = red_b[0];
}

// One GDN layer, one token, one (token, layer) launch — the same signature as `rec_kernel` and
// `rec_unroll4_kernel`, so the launcher's choice of spelling is one call and cannot move an address.
// Body is `rec2c4_kernel`'s; keep them identical (see the header for the two parents of each half).
__global__ void rec_fused_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta, float* __restrict__ S,
                                 float* __restrict__ o, int nheads) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float ks[D], qs[D], redk[32], redq[32];

    float kq = k[h * D + j], qq = q[h * D + j];
    float sk, sq;
    block_sum_pair_f32(kq * kq, qq * qq, redk, redq, &sk, &sq);
    float kn = kq * rsqrtf(sk + 1e-6f);
    float qn = qq * rsqrtf(sq + 1e-6f) * Q_SCALE;
    __syncthreads();
    ks[j] = kn;
    qs[j] = qn;
    __syncthreads();

    float gam = expf(g[h]), bet = beta[h];
    float* Sh = S + (size_t)h * D * D;

    // ---- from here to the end: `rec_unroll4_kernel`'s body, verbatim (see the header) -------------
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
