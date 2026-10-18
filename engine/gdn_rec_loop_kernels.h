/*
 * gdn_rec_loop_kernels.h — §04.4's shipped recurrence with the TOKEN INDEX moved INSIDE the launch:
 * one submission walks `tokens` token-steps instead of one. This is the engine's copy of
 * `prototype/gdn_rec_isa/gdn_rec_isa_kernels.h::recloop_kernel` (MODE `recloop`), which is itself
 * `rec_fused_kernel`'s body — the spelling that ships since D51 — walked over N tokens. Nothing else
 * moved: same grid `<<<NH=16, D=128>>>`, same thread map, same global state buffer at the same
 * addresses, same per-token arithmetic in the same order. The ONLY thing this form changes is how many
 * token-steps one submission carries, which is what makes it a launch-structure item and not a
 * numerics one (D53, `progress/2026-10-16/prefill-decision-log-2026-10-16.md`; the screen is
 * `bench/performance-status.md` §23; the port order — knob first, `--gdn-loop`, default 1 — is D54).
 *
 * WHAT IS TRANSCRIBED FROM WHERE, and the rule that comes with it. Bit-identity is owed to the shipped
 * SOURCE TEXT, not to the algebra, so the loop body is `engine/gdn_rec_fused_kernels.h::rec_fused_kernel`
 * character for character. The operand pointers are re-declared in the loop's inner scope
 * (`q = qtile + t*CH`, …) precisely so that NOT ONE arithmetic line had to be rewritten to add `t`;
 * the only new lines in the kernel are the `for`, those six declarations and the braces. That is
 * checked, not hoped: `prototype/gdn_rec_isa/check_recloop_transcription.py` extracts the per-token
 * core of this kernel, `rec2c_kernel`'s merged descendant in the harness, and the shipping engine
 * header, requires them identical including the `#pragma unroll 4` lines, and carries four must-catch
 * mutations so it cannot pass vacuously. It cannot see the COMPILER — that is the device bit-identity
 * gate — and it cannot see association (`prototype/gdn_collapse/results.md` §2, quoted in
 * `engine/gdn_rec_fused_kernels.h`: hoisting one scalar into a different product tree moved 93,522 of
 * 131,072 `o` elements on device while a CPU mirror watched two spellings of one formula agree).
 *
 * THE CENSUS AND SCREEN FIGURES, recorded so a future LLVM bump that quietly changes the loop is caught
 * against numbers and not against impressions (`prototype/gdn_rec_isa/isa_census.py` and
 * `-Rpass-analysis=kernel-resource-usage` on the listing of the binary that was timed; the screen ran
 * `prototype/gdn_rec_isa/gdn_rec_isa --modes rec2c4,recloop`, `bench/performance-status.md` §23):
 *
 *   bill            1,530 slots/token-step against `rec_fused_kernel`'s 1,374 — +156, +11.4 %. It is
 *                   NOT state math: both kernels show the same walk shapes 16x8 / 16x8 (the unroll
 *                   survived), and the delta is per-token addressing plus loop control
 *                   (`salu.mov` 7→82, `loop.salu` 57→79, `salu.branch` 3→21, prologue 294→458), all of
 *                   it from recomputing `t*CH` per iteration. Known and ACCEPTED for this port: the
 *                   addressing-spelling variant (increment the six row pointers instead of recomputing
 *                   them) is integer-only, so bit-neutral, but it is a separate lever with its own
 *                   transcription + bit-identity gates. Do not fold it in here.
 *   VGPRs / spill   75 VGPR, 40 SGPR, vgpr_spill = 0, ScratchSize = 0, occupancy UNCHANGED at 16
 *                   waves/SIMD (`rec_fused_kernel`: 59 VGPR, 24 SGPR, 0 spill, same 16). The launch
 *                   puts 64 wavefronts on the device (16 blocks x 4), so +16 registers do not bind at
 *                   this grid.
 *   LDS             1,280 B/block — identical to the shipped form, and per LAUNCH, not per token.
 *                   Nothing is staged: the state stays in global memory exactly as before. A variant
 *                   that staged state in LDS is Route A's geometry (`engine/gdn_collapse_kernels.h`)
 *                   and re-opens the association question; do not slip it in.
 *   device screen   whole-body span (1 launch vs 128): 1.6363 / 1.5834 / 1.5924 / 1.5955 on the four
 *                   carried/zeros x d0.002/d0.4 cases; span 32 (4 launches): 1.4713–1.5553. 8/8 cases,
 *                   7/7 pairs each. Per-token cost 5.43–5.74 µs → 3.43–3.51 µs.
 *   admission       bit-identical to `rec0` on `o` AND final state at spans 0, 1, 3 and 16 —
 *                   `ndiff_o_vs0=0 ndiff_state_vs0=0`, 4/4 cases each span, reruns clean — plus
 *                   `VOLUME_OK` at spans 0 and 3 on a per-TOKEN-STEP byte account.
 *
 * WHY NO BARRIER WAS ADDED, which is the one thing here that is easy to get wrong and silently wrong.
 * `block_sum_pair_f32` ends with `*out_a = red_a[0]` and NO trailing barrier; its own comment
 * justifies that by "neither array is reused before the next launch", and inside a token loop that
 * claim is false. The real reason it is still safe: every thread's read of `red_a[0]`/`red_b[0]` in
 * iteration t precedes the `__syncthreads()` that sits between it and iteration t+1's write to those
 * arrays, and `ks`/`qs` are covered by the pair of barriers that already bracket their write. All
 * threads run the same trip count — `tokens` is uniform and nothing diverges it — so every barrier is
 * hit by every thread the same number of times. The witness if that argument is ever broken is a
 * rerun disagreement (`--repeat 2`, `rerun_ndiff_*`), because a missing barrier is a RACE and a race
 * disagrees with itself before it disagrees with MODE 0.
 *
 * THE STRIDES ARE DERIVED, NOT PASSED. `CH` (6144) and `NH*D` (2048) are the same constants the
 * per-token call site uses to walk `sc.g_u` / `sc.g_o` (`engine/decode_step.cpp`'s
 * `sc.g_u + t*kQkvRows`, `sc.g_o + t*kValueDim`), and the static_asserts below make the equality a
 * compile error instead of a comment; `engine/kernels_gdn.cpp::gdn_rec_span` asserts the same
 * identities against the engine's own names. Passing row strides as arguments would let a wrong stride
 * be expressible, and a wrong stride here is a silent numerics change, not a crash.
 *
 * WHAT THIS FILE IS NOT. It is not the default. `--gdn-loop` defaults to 1 — i.e. this kernel is not
 * submitted at all until the paired in-situ A/B on the `gdn_rec` family row says otherwise (D54's
 * order, which is D32 → D50 repeated deliberately), and a ONE-ROW body never reaches it whatever the
 * flag says (`kernels.h:resolve_gdn_loop`), which is what makes a decode step unchanged by
 * construction rather than by measurement. It is inert under `--gdn-scan collapse`, which submits a
 * different kernel and already batches per body; `decode_run` refuses that pair.
 */
#pragma once

#include "engine/gdn_rec_fused_kernels.h"   // block_sum_pair_f32, and with it D, NH, CH, Q_SCALE

// The q|k|v row stride is `CH` because `u` is the conv's [n][6144] tile, and the o row stride is
// `NH*D` because §04.5's input is [n][2048]. Compile errors, not comments — see the header.
static_assert(CH == 3 * NH * D, "rec_loop's q|k|v row stride is CH; must be 3*NH*D = the engine's kQkvRows");
static_assert(NH * D == 2048, "rec_loop's o row stride is NH*D; must be the engine's kValueDim");

// One GDN layer, ONE launch, `tokens` consecutive token-steps of the §04.4 scan. `qtile`/`ktile`/
// `vtile` are the three 2048-wide slices of one [n][6144] tile (row 0 of each), `gtile`/`betatile`
// are [n][16], `otile` is [n][2048], and `S` is the layer's [16][128][128] block — the same buffers
// the per-token launches were given, so the two forms touch identical addresses and a diff of their
// outputs is a claim about the schedule rather than about the access pattern.
__global__ void rec_loop_kernel(const float* __restrict__ qtile, const float* __restrict__ ktile,
                                const float* __restrict__ vtile, const float* __restrict__ gtile,
                                const float* __restrict__ betatile, float* __restrict__ S,
                                float* __restrict__ otile, int nheads, int tokens) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float ks[D], qs[D], redk[32], redq[32];

    for (int t = 0; t < tokens; ++t) {
        // This token's rows, named so that every line below is `rec_fused_kernel`'s verbatim.
        // `qtile`/`ktile`/`vtile` are the three 2048-wide slices of a [n][6144] tile, so a token step
        // is `+CH`; `g`/`beta` are [n][16]; `o` is [n][2048]. Same buffers, same addresses the
        // per-token launches were given.
        const float* __restrict__ q = qtile + (size_t)t * CH;
        const float* __restrict__ k = ktile + (size_t)t * CH;
        const float* __restrict__ v = vtile + (size_t)t * CH;
        const float* __restrict__ g = gtile + (size_t)t * NH;
        const float* __restrict__ beta = betatile + (size_t)t * NH;
        float* __restrict__ o = otile + (size_t)t * (NH * D);

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

        // ---- from here to the end: `rec_unroll4_kernel`'s body, verbatim (see the header) ---------
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
