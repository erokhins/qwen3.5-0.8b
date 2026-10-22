/*
 * prototype/mtp_draft/td_attn_tu.cpp — the §05 attention row for the MTP block.
 *
 * TWO ENTRY POINTS, ONE ARITHMETIC.
 *
 *   shipped_attn_row  — `attn_core` from `prototype/attn_decode/attn_decode_kernels.h`, one row, fused
 *                       plan, one-pass `full` softmax (the form `attn_decode`'s own gate calls
 *                       "bit-exact with the pins"). Used as the reference side of a bit-identity check.
 *   mtp_attn_row      — the same arithmetic with ONE addressing change, needed because the MTP cache is
 *                       not the target's cache.
 *
 * WHY THE CHANGE IS UNAVOIDABLE. In `attn_core`, `row_pos = a.pos + qrow` does two jobs at once: it is
 * the position fed to `head_norm_rope` (§05.2/§05.3) and, through `slot_of(row_pos, cap, ...)`, the cache
 * slot of the appended lane (§05.7). For the target those coincide — every position writes a lane, so the
 * dense index IS the position. For the MTP block they do not: the drafter appends one lane per DRAFTED
 * token while the target consumes k+1 tokens per round, so at k=1 the MTP cache holds positions
 * q+1, q+3, q+5, … (read `speccheck/mtp.py:greedy_mtp`: drafts are fed at `q+1+j`, the verify consumes
 * `q..q+k`). The gaps cannot be filled — those positions were never drafted, and a zero lane earns real
 * softmax weight — so the MTP cache is a dense list with per-lane absolute positions, and only the rope
 * needs the position. `mtp_attn_row` therefore takes `self_lane` (where the lane goes) and `pos` (what
 * rope uses) as SEPARATE fields. That seam is the engine's decision to make when this graduates
 * (DRAFT-HANDOVER.md §3); here it is a rig kernel, and its fidelity to the shipped arithmetic is
 * DISPROVED independently of any number: on a case whose positions are contiguous from 0 the two
 * addressings coincide, and `--mode xcheck` requires `mtp_attn_row` to be bit-identical to `attn_core`
 * there, lane by lane and step by step. Anywhere the transcription drifted, that check is where it shows.
 *
 * Everything else is transcribed character for character, including the things that look accidental:
 * the score dot's `for (t = 0; t < 8; ++t) if (t < nd)` literal trip + guard (the "D3 LDS" note at the
 * foot of `attn_decode_kernels.h` says the obvious spelling cost a promote-alloca-to-LDS), the
 * `bf(fb(dot)) * SCALE` rounding site, the tile-of-8 scoring with warp w owning lane base+w, the
 * max-then-sum-then-BF16-rounded-weight-then-PV order of the `full` pass 2 (that round IS §05.5's
 * reference form), `l = 1.0f` after it so `finish_head` does not divide twice, and the §05.6 epilogue's
 * `av * bf(fb(1/(1+exp(-g))))`. The gdn_collapse lesson is why the copy keeps the parentheses: bit
 * identity is owed to the source text's association, not to the algebra, and a CPU mirror cannot see a
 * re-association.
 */
#include "td_launchers.h"

#include <stdexcept>

#include "prototype/attn_decode/attn_decode_kernels.h"

namespace td {
namespace {

struct MtpRowArgs {
    const hip_bfloat16 *qg, *kg, *vg, *qn, *kn;
    hip_bfloat16 *kcache, *vcache, *aout, *gout;
    hip_bfloat16 *qrope_out, *krope_out, *gate_out;
    int cap, nlanes, self_lane, pos;
};

__global__ void __launch_bounds__(THREADS)
mtp_attn_row_kernel(MtpRowArgs a) {
    __shared__ float sq[DH], sk[DH], sv[DH], zs[DH], red[MAXW];
    __shared__ float sc[MAXSC];                       // MAXSC >= TILE; the one-pass row lives here
    const int head = blockIdx.x;
    const int kv = head / GQA;                        // kappa(i) = i/4, §05.5
    const int d = threadIdx.x;
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;
    const float pos = (float)a.pos;                   // ABSOLUTE position: rope only
    const int nrow = a.nlanes;                        // dense lane count, incl. this step's append

    // §05.2 + §05.3 + the v stage. `head_norm_rope` is the SHIPPED function, called exactly as
    // attn_core's fused branch calls it, so the norm and the rope are not a copy at all.
    head_norm_rope(a.qg + (size_t)head * (2 * DH), a.qn, pos, NEG_NONE, sq, zs, red, nw);
    head_norm_rope(a.kg + (size_t)kv * DH, a.kn, pos, NEG_NONE, sk, zs, red, nw);
    for (int i = d; i < DH; i += blockDim.x) sv[i] = bf(a.vg[(size_t)kv * DH + i]);
    __syncthreads();

    // The §05.7 append, plus the staged dumps attn_core's `stage_head` produces (they are the exact
    // witnesses for §05.2/§05.3, so the rig keeps them). ONE DIFFERENCE from `slot_of(row_pos, cap,…)`:
    // the slot is the dense append index.
    a.qrope_out[(size_t)head * DH + d] = fb(sq[d]);
    a.gate_out[(size_t)head * DH + d] = a.qg[(size_t)head * (2 * DH) + DH + d];
    if (head % GQA == 0) {
        a.krope_out[(size_t)kv * DH + d] = fb(sk[d]);
        a.kcache[((size_t)kv * a.cap + a.self_lane) * DH + d] = fb(sk[d]);
        a.vcache[((size_t)kv * a.cap + a.self_lane) * DH + d] = fb(sv[d]);
    }

    float qreg[8];
    for (int t = 0; t < 8; ++t) qreg[t] = sq[lane * 8 + t];

    const int nd = 8;                                 // all 8 BF16 of a lane's 512-byte read
    const int tile = TILE;
    float m = 0.f, l = 0.f, acc = 0.f;                // neutral seed, §07.3's note

    // ---- pass 1: score tiles of `tile`, warp w scores lane base+w
    int ns = 0;
    for (int base = 0; base < nrow; base += tile) {
        const int nt = std::min(tile, nrow - base);
        __syncthreads();
        if (w < nt) {
            const int j = base + w;
            float dot = 0.f;
            if (j == a.self_lane) {                   // the appended lane comes from shared
                for (int t = 0; t < 8; ++t) if (t < nd) dot += sq[lane * 8 + t] * sk[lane * 8 + t];
            } else {
                const hip_bfloat16* kr = a.kcache + ((size_t)kv * a.cap + (size_t)j) * DH;
                uint4 u = *reinterpret_cast<const uint4*>(kr + lane * 8);
                const hip_bfloat16* k = reinterpret_cast<const hip_bfloat16*>(&u);
                for (int t = 0; t < 8; ++t) if (t < nd) dot += bf(k[t]) * qreg[t];
            }
            dot = warp_sum(dot);
            if (lane == 0) sc[ns + w] = bf(fb(dot)) * SCALE;
        }
        ns += nt;
        __syncthreads();
    }
    // ---- pass 2 (reference form): max, sum, BF16 weights, P.V. The weights were already normalised
    // and rounded, so the epilogue must not divide again; hence l = 1 below.
    for (int t = 0; t < ns; ++t) m = fmaxf(m, sc[t]);
    for (int t = 0; t < ns; ++t) l += expf(sc[t] - m);
    for (int t = 0; t < ns; ++t) {
        const int j = t;
        const float wt = bf(fb(expf(sc[t] - m) / l));
        const float v = (j == a.self_lane)
                            ? sv[d]
                            : bf(a.vcache[((size_t)kv * a.cap + (size_t)j) * DH + d]);
        acc += wt * v;
    }
    l = 1.0f;

    // §05.6 epilogue, verbatim from finish_head().
    const float av = bf(fb(acc / l));
    a.aout[(size_t)head * DH + d] = fb(av);
    const float g = bf(a.qg[(size_t)head * (2 * DH) + DH + d]);
    a.gout[(size_t)head * DH + d] = fb(av * bf(fb(1.0f / (1.0f + expf(-g)))));
}

MtpRowArgs to_row(const AttnRow& r) {
    MtpRowArgs a;
    a.qg = (const hip_bfloat16*)r.qg;  a.kg = (const hip_bfloat16*)r.kg;
    a.vg = (const hip_bfloat16*)r.vg;  a.qn = (const hip_bfloat16*)r.qn;
    a.kn = (const hip_bfloat16*)r.kn;
    a.kcache = (hip_bfloat16*)r.kcache; a.vcache = (hip_bfloat16*)r.vcache;
    a.aout = (hip_bfloat16*)r.aout;     a.gout = (hip_bfloat16*)r.gout;
    a.qrope_out = (hip_bfloat16*)r.qrope_out;
    a.krope_out = (hip_bfloat16*)r.krope_out;
    a.gate_out = (hip_bfloat16*)r.gate_out;
    a.cap = r.cap; a.nlanes = r.nlanes; a.self_lane = r.self_lane; a.pos = r.pos;
    return a;
}

}  // namespace

void mtp_attn_row(const AttnRow& r) {
    // The one-pass row stages its scores in MAXSC floats, exactly as `attn_core` does, and its launcher
    // refuses anything wider for the same reason: `attn_decode`'s split/streaming route is a different
    // form with its own gate, and an MTP cache longer than MAXSC lanes is the engine's decision to make
    // with the streaming route in hand (DRAFT-HANDOVER.md §3), not something this rig may reach by
    // walking off the end of a shared array.
    if (r.nlanes > MAXSC)
        throw std::runtime_error("mtp_attn_row: " + std::to_string(r.nlanes) +
                                 " lanes exceeds the one-pass form's MAXSC; the streaming route is a "
                                 "separate decision");
    mtp_attn_row_kernel<<<dim3(N_Q, 1, 1), THREADS>>>(to_row(r));
    td_launches += 1;
}

void shipped_attn_row(const AttnRow& r) {
    CoreArgs a{};                       // zero-init: tile_kernel/fattn_cover/rope pointers off
    a.qg = (hip_bfloat16*)r.qg;   a.kg = (hip_bfloat16*)r.kg;   a.vg = (hip_bfloat16*)r.vg;
    a.kcache = (hip_bfloat16*)r.kcache; a.vcache = (hip_bfloat16*)r.vcache;
    a.qn = (hip_bfloat16*)r.qn;   a.kn = (hip_bfloat16*)r.kn;
    a.qrope = nullptr; a.krope = nullptr;                 // fused plan: nothing staged to read
    a.aout = (hip_bfloat16*)r.aout; a.gout = (hip_bfloat16*)r.gout;
    a.qrope_out = (hip_bfloat16*)r.qrope_out;             // staged dumps ON, which is also the append
    a.krope_out = (hip_bfloat16*)r.krope_out;
    a.gate_out = (hip_bfloat16*)r.gate_out;
    a.kc_as = (hip_bfloat16*)r.kcache; a.vc_as = (hip_bfloat16*)r.vcache;
    a.part = nullptr; a.stale = nullptr;
    a.cap = r.cap;
    a.pos = r.pos;                                          // contiguous-from-0 case: pos == self_lane
    a.lvalid = r.nlanes;
    a.split = 1; a.neg = NEG_NONE; a.full = 1; a.score_f32 = 0; a.fused = 1; a.tile = TILE;
    a.nrows = 1; a.tile_kernel = 0; a.fattn_cover = 0;
    ::attn_core<<<dim3(N_Q, 1, 1), THREADS>>>(a);
    td_launches += 1;
}

}  // namespace td
