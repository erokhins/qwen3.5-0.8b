/*
 * gdn_rec_hoist_kernels.h — bet G1: the same §04.4 looped recurrence with the per-token k/q L2-norms
 * computed OUTSIDE it. Two kernels, one pair of launches per (layer, body):
 *
 *   `gdn_norm_hoist_kernel`  — one block per (token-step, head), 128 threads. It computes exactly the
 *                              prologue `rec_loop_kernel` computes today — the two row sums of squares
 *                              with `block_sum_pair_f32`, the two normalised rows, `expf(g[h])` and
 *                              `beta[h]` — and writes them to three stage tiles.
 *   `rec_loop_normed_kernel` — the shipped looped walk with the prologue deleted: per token-step it
 *                              reads the two normalised rows and the two scalars and then walks the
 *                              state. It has NO shared memory, NO `__syncthreads()` and NO cross-lane
 *                              reduction, so nothing separates token t's last store from token t+1's
 *                              first load.
 *
 * WHY. `prototype/gdn_walk/evidence-tokenbill-2026-10-19.txt`: the shipping walk bills 1,327
 * slots/token-step/wavefront and the per-token cross-lane norm machinery inside it is 122 slots
 * (permute 20 + add 19 + exec-guard 57 + LDS round-trip 5 + barrier 10 + range-select 7 + addr 4)
 * = 9.2 % of the bill. BET.md §3 (`prototype/gdn_walk/BET.md`) is the funded item and its kill lines;
 * the part that mattered more than the slots is the SHAPE: `block_sum_pair_f32` ends with a barrier,
 * then `ks`/`qs` are bracketed by two more, so three barriers sit on the chain between token t and
 * token t+1 in a kernel that realises ~3.2 wavefronts per CU — the thing that forbids pipelining the
 * walk across iterations (BET.md §2). The norms are a row-wise map over tiles an earlier pass already
 * read; they were never part of the recurrence.
 *
 * BIT-IDENTITY IS THE ADMISSION, AND IT IS WHY THE CODE LOOKS REDUNDANT. `block_sum_pair_f32`'s
 * summation association is what makes the shipped bits (`prototype/gdn_collapse/results.md` §2,
 * quoted in `engine/gdn_rec_fused_kernels.h`: a merely hoisted scalar constant moved 93,522 of
 * 131,072 `o` elements on device while a CPU mirror watched two spellings of one formula agree). So
 * nothing here is "reduced equivalently":
 *   * the pre-pass calls the SAME `block_sum_pair_f32` with the SAME thread map (blockDim.x == D == 128,
 *     thread j owns element j, `warps = blockDim.x >> 5 == 4`), so the 64-bit shuffle mask, the
 *     16/8/4/2/1 order, the per-warp partial layout in `redk`/`redq`, the `(threadIdx.x < warps)` gate
 *     and the final `red_a[0]` fold are the shipped ones because they are the SAME CODE, not a copy;
 *   * `kn`/`qn`/`gam`/`bet` are stored from the shipped expressions verbatim — `kq * rsqrtf(sk + 1e-6f)`,
 *     `qq * rsqrtf(sq + 1e-6f) * Q_SCALE`, `expf(g[h])`, `beta[h]` — and a F32 store/load round-trip is
 *     exact, so the walk consumes bit-identical operands;
 *   * the walk's operand pointers are NAMED `ks`/`qs` (`const float* __restrict__ ks = knorm + h*D`)
 *     precisely so that every state-walk line is `rec_loop_kernel`'s character for character. The walk
 *     text is checked, not hoped: `prototype/gdn_walk/check_hoist_transcription.py` extracts the walk
 *     region of both kernels, requires them identical including the `#pragma unroll 4` lines, and
 *     carries must-catch mutations so it cannot pass vacuously. Keeping the text identical is what keeps
 *     clang's `-ffp-contract` decisions identical — contraction is chosen from the expression, so a
 *     re-spelled `ks[i] * x` could become or stop being an FMA, and that is exactly the class of change
 *     a CPU mirror cannot see. `prototype/gdn_walk/walk_isa_diff.py` is the device-independent half of
 *     that argument: it compares the FP-opcode census of the shipped kernel against the two new ones.
 *
 * THE TRAFFIC THIS BUYS, arithmetic not measurement, per 4,096-token prefill at 18 GDN layers
 * (`prototype/gdn_walk/G1-result.md` carries the same figure with its reasoning):
 *   pre-pass reads q|k            4096 × 18 × (2048+2048) F32 = 1.19 GB
 *   pre-pass writes kn|qn|gsc     4096 × 18 × (2048+2048+32) F32 = 1.20 GB
 *   the walk's own reads          UNCHANGED: it reads 16 KiB of kn/qn where it read 16 KiB of q/k, plus
 *                                 the same v row, plus 128 B of scalars instead of reading `g`/`beta`
 *   ⇒ +2.4 GB of DEMANDED bytes per pass, against the walk's own ~155 GB of demand (the layer state is
 *     1 MiB and is walked once per token, so most of THAT demand is L2-served). But these 2.4 GB are
 *     written once and read once per body and a body's stage tiles are 8.4 MiB against a GPU L2 of
 *     2 MiB (`devices/strix-halo-official-specs.md` §GPU cache sizes), so unlike the state they cannot
 *     be L2-absorbed: at the measured 237 GB/s ceiling (AGENTS.md) that is ≈ +10 ms of DRAM on a
 *     266.24 ms row. §38's lesson cuts both ways — demanded bytes are not DRAM bytes — and here it cuts
 *     AGAINST the change. The pre-registered lines in BET.md §3 are the judge; if the result is between
 *     them, these bytes are the first suspect.
 *
 * WHAT THIS FILE IS NOT. It is not the default. `--gdn-norm` defaults to `inline`, which submits
 * `rec_loop_kernel` untouched; `hoist` is a knob and the paired in-situ A/B on the `gdn_rec` family row
 * decides whether the default moves (D50/D54's order, which this repo repeats deliberately for every
 * recurrence change). It reaches ONE path: the `--gdn-loop` span branch of `decode_step.cpp`'s GDN site.
 * `--gdn-spell ship|unroll4|fused` (one launch per token) and `--gdn-scan collapse` do not consult it and
 * stay bit-identical because their kernels are not edited here. `block_sum_pair_f32` is shared with
 * `rec_fused_kernel` and is NOT modified by this file.
 */
#pragma once

#include "engine/gdn_rec_loop_kernels.h"   // block_sum_pair_f32 (via gdn_rec_fused_kernels.h), D, NH, CH, Q_SCALE

static_assert(CH == 3 * NH * D, "the hoist pass reads q|k rows at the CH stride, as rec_loop_kernel does");
static_assert(NH * D == 2048, "the kn/qn stage row stride is NH*D = the engine's kValueDim");

// The scalar stage row: `gam` for the NH heads, then `beta` for the NH heads, one row of 2*NH F32 per
// token-step. Two half-rows rather than a struct of arrays per head because the walk reads exactly one
// of each per token and the offsets must be as boring as possible.
constexpr int kGdnScalarRow = 2 * NH;

// One token-step, one head, 128 threads: everything `rec_loop_kernel` computes before its first state
// access, and nothing else. `grid = <<<dim3(NH, tokens), D>>>` — `blockIdx.x` is the head and
// `blockIdx.y` the token-step so the walk's `h = blockIdx.x` / row-address spellings carry over, and
// `blockDim.x == D == 128` is LOAD-BEARING, not incidental: `block_sum_pair_f32`'s second stage folds
// `warps = blockDim.x >> 5` partials, so a different block width would be a different summation tree
// and therefore different bits.
//
// `qtile`/`ktile` are two 2048-wide slices of the conv's [n][6144] tile, `gtile`/`betatile` are [n][16]
// — the same operands `gdn_rec_span` hands the looped kernel — and `knorm`/`qnorm` are [n][2048] F32
// stage tiles whose row t is read by the walk as if it were the `ks`/`qs` shared arrays.
__global__ void gdn_norm_hoist_kernel(const float* __restrict__ qtile, const float* __restrict__ ktile,
                                      const float* __restrict__ gtile,
                                      const float* __restrict__ betatile,
                                      float* __restrict__ knorm, float* __restrict__ qnorm,
                                      float* __restrict__ gsctile, int nheads, int tokens) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float redk[32], redq[32];       // the pair reduction's own arrays, as in the shipped form

    const float* __restrict__ q = qtile + (size_t)(blockIdx.y) * CH;
    const float* __restrict__ k = ktile + (size_t)(blockIdx.y) * CH;
    const float* __restrict__ g = gtile + (size_t)(blockIdx.y) * NH;
    const float* __restrict__ beta = betatile + (size_t)(blockIdx.y) * NH;
    float* __restrict__ kn = knorm + (size_t)(blockIdx.y) * (NH * D);
    float* __restrict__ qn = qnorm + (size_t)(blockIdx.y) * (NH * D);
    float* __restrict__ gsc = gsctile + (size_t)(blockIdx.y) * kGdnScalarRow;

    // ---- `rec_loop_kernel`'s per-token prologue, verbatim (see the header) ------------------------
    float kq = k[h * D + j], qq = q[h * D + j];
    float sk, sq;
    block_sum_pair_f32(kq * kq, qq * qq, redk, redq, &sk, &sq);
    kn[h * D + j] = kq * rsqrtf(sk + 1e-6f);
    qn[h * D + j] = qq * rsqrtf(sq + 1e-6f) * Q_SCALE;
    // The two scalars the prologue forces. `expf` is deterministic per target, so evaluating it once per
    // block instead of once per thread cannot move a bit; the walk reads the value as its `gam`.
    if (j == 0) {
        gsc[h] = expf(g[h]);
        gsc[NH + h] = beta[h];
    }
}

// The looped walk with its prologue hoisted: one GDN layer, ONE launch, `tokens` token-steps, the same
// grid `<<<NH, D>>>`, the same `S`, the same `otile`, the same per-token arithmetic. `knorm`/`qnorm` are
// what `gdn_norm_hoist_kernel` just wrote for this body and `gsctile` its scalar rows; `vtile` is still
// row 0 of the v slice of the conv tile, because `v` was never part of the norm.
//
// The per-token text below is `engine/gdn_rec_loop_kernels.h`'s loop body from `float gam =` through
// `o[h * D + j] = acc;` verbatim. What is gone is everything above it: no `__shared__`, no
// `block_sum_pair_f32`, no barriers, and the row pointers `ks`/`qs` are the stage tiles' rows instead of
// shared arrays so the two walks' lines stay identical. That no cross-thread coupling is left is not an
// optimisation trick: thread j's state columns are `Sh[i*D + j]`, which only thread j ever reads or
// writes, and the only values it needs from other threads are now values it loads.
__global__ void rec_loop_normed_kernel(const float* __restrict__ vtile,
                                       const float* __restrict__ knorm,
                                       const float* __restrict__ qnorm,
                                       const float* __restrict__ gsctile,
                                       float* __restrict__ S, float* __restrict__ otile,
                                       int nheads, int tokens) {
    int h = blockIdx.x;
    int j = threadIdx.x;

    for (int t = 0; t < tokens; ++t) {
        const float* __restrict__ v = vtile + (size_t)t * CH;
        const float* __restrict__ g = gsctile + (size_t)t * kGdnScalarRow;
        const float* __restrict__ beta = g + NH;
        float* __restrict__ o = otile + (size_t)t * (NH * D);
        // NAMED `ks`/`qs`, and declared INSIDE the loop for the same reason `rec_loop_kernel` declares
        // `q`/`k`/`v`/`o` inside its loop: not one line of the walk below had to be rewritten to add `t`.
        // Hoisting these two declarations above the loop — which reads like a legal strength reduction and
        // is what this file originally did — silently walks every token with token 0's normalised rows and
        // was caught only by the device gate (layer 0's post-prefill state fingerprint moved from
        // n2=221.737 to n2=45.290; `prototype/gdn_walk/G1-result.md`). The row advance belongs HERE, in
        // the declaration, because the walk text must not change.
        // `__restrict__` on both keeps them from aliasing `S`/`otile`, which is what lets the loads move
        // as freely as the shared-array reads they replace.
        const float* __restrict__ ks = knorm + (size_t)t * (NH * D) + (size_t)h * D;
        const float* __restrict__ qs = qnorm + (size_t)t * (NH * D) + (size_t)h * D;

        float gam = g[h], bet = beta[h];
        float* Sh = S + (size_t)h * D * D;

        // ---- from here to the end: `rec_loop_kernel`'s per-token walk, verbatim (see the header) --
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
}
