// Launchers for the GDN family: prototype/gdn_decode (§04.2–§04.4) and prototype/gdn_gated_norm
// (§04.5). Separate TU so their macros (D, NH, CH, K, GN_D_V, …) cannot collide with another op's.
#include "kernels.h"

#include "engine/launch_stats.h"
#include "engine/gdn_collapse_kernels.h"
#include "engine/gdn_rec_unroll4_kernels.h"
#include "prototype/gdn_chunk/gdn_chunk_kernels.h"
#include "prototype/gdn_decode/gdn_decode_kernels.h"
#include "prototype/gdn_gated_norm/gdn_gated_norm_kernels.h"

namespace engine::kernels {

// §04.2 over a wave. grid.x is the channel strip — the single-token launch's 24 blocks of 256 — and
// grid.y the token, so `tokens == 1` is the same grid, not a grid padded to one, and the per-element
// arithmetic (F32 taps in j order, one BF16 round, SiLU) is the one `prototype/gdn_decode/check.py`
// has always pinned against `q/k/v_post_conv`. What the wave adds is the window rule: the three
// columns before row 0 come from `win` (§08.2's carried window), never from zeros, and the launch
// writes the window's new contents once for the whole wave instead of once per token — the copy
// `decode_step.cpp` used to issue per token is what collapses here.
void gdn_conv(const float* raw, const float* win, const float* conv_weight, float* u,
              float* new_win, int tokens) {
    conv_kernel<<<dim3(CH / 256, tokens), 256>>>((const float*)raw, win, conv_weight, u, new_win,
                                                 tokens, GCK_NONE);
    launchstats::gdn_conv().record(tokens);
}

// §04.3: `<<<tokens, NH>>>`, one thread per (token, value head), so a decode step is still the
// one-block launch it was and no element expression moved.
void gdn_gates(const float* a_raw, const float* b_raw, const float* a_log, const float* dt_bias,
               float* g, float* beta, int tokens) {
    gates_kernel<<<tokens, NH>>>(a_raw, b_raw, a_log, dt_bias, g, beta, tokens, GG_NONE);
    launchstats::gdn_gates().record(tokens);
}

void gdn_rec(const float* q, const float* k, const float* v, const float* g, const float* beta,
             float* S, float* o, GdnRecSpell spell) {
    // One layer per launch: the model interleaves GDN and attention layers (§03.5), so a token
    // never has two adjacent GDN layers to batch. `all` mode (18 layers, one launch) in the
    // prototype is a bandwidth-measurement shape, not a token shape.
    //
    // Which SPELLING runs (`--gdn-spell`, `kernels.h::GdnRecSpell`). The two launches differ in the
    // called kernel and in nothing else — same `<<<NH, D>>>`, same operands, same state buffer, so the
    // flag cannot move an address and an A/B between them measures the schedule rather than the plan
    // (the `attn_slim_kernels.h` / `--attn-touch` shape, one level down the tree).
    if (spell == GdnRecSpell::kUnroll4) {
        rec_unroll4_kernel<<<NH, D>>>(q, k, v, g, beta, S, o, NH);
    } else {
        rec_kernel<<<NH, D>>>(q, k, v, g, beta, S, o, NH);
    }
    // Not counted: this launcher has no token width (see `engine/launch_stats.h`), and a family
    // that can only ever report rows=1 would give section (h) a check that cannot fail.
}

// `--gdn-spell`, and it parses or throws for the standing reason: the flag names which side of an A/B
// the run is. `ship` is the default and stays the default until D30 item 1.9's in-situ A/B is read;
// `unroll4` is the bit-identical re-spelling worth 1.149-1.158x at launch level and possibly nothing
// in situ (`prototype/gdn_rec_isa/results.md` §10).
GdnRecSpell parse_gdn_rec_spell(const std::string& name) {
    if (name == "ship") return GdnRecSpell::kShip;
    if (name == "unroll4") return GdnRecSpell::kUnroll4;
    throw std::runtime_error("--gdn-spell takes `ship` (the shipped rec_kernel, the default) or "
                             "`unroll4` (engine/gdn_rec_unroll4_kernels.h, bit-identical, unroll-4 "
                             "state walks); got '" + name + "'. Refusing rather than guessing, because "
                             "this flag names which side of an A/B the run is");
}

const char* gdn_rec_spell_name(GdnRecSpell s) {
    switch (s) {
        case GdnRecSpell::kShip: return "ship";
        case GdnRecSpell::kUnroll4: return "unroll4";
    }
    return "invalid";
}

// Route A, D-12: the same recurrence, one launch per BODY. `dim3(kStrips, kNH)` = (4, 16) = 64 blocks
// of 128 threads where `gdn_rec` issues 16 blocks of 128 per TOKEN, and `u`/`g`/`beta`/`o` are the
// whole body tiles rather than row t of them — the two forms therefore touch identical addresses, which
// is what makes `engine/gdn_collapse_kernels.h`'s bit-identity claim checkable at all.
//
// Not counted in `launch_stats` for the same reason `gdn_rec` is not: the recurrence's launch count is
// the thing under test, and the wave report's `gdn_rec` row already carries it as `rows_total`
// (`engine/stage_clock.h`: "a reader who wants a launcher-side witness for the recurrence has to count
// launches"). Adding a family here would move the count out of the row the win line is written on.
void gdn_rec_collapse(const float* u, const float* g, const float* beta, float* S, float* o,
                      int rows) {
    gdn_rc::rec_collapse_kernel<<<dim3(gdn_rc::kStrips, gdn_rc::kNH), gdn_rc::kBlockDim>>>(
        u, g, beta, S, o, rows);
}

// ---- P9: §04.7's chunked scan ----------------------------------------------------------------
//
// Three things this launcher is responsible for, in the order the code does them: refuse a body the
// form cannot serve, size nothing (the scratch is `decode_step`'s), and issue the TWO launches in
// the order the scan depends on — `carry` reads what `prep` wrote, and same-stream ordering is what
// orders them, exactly as `attn_prep`/`attn_core` are ordered in `kernels_attn.cpp`.
GdnScan parse_gdn_scan(const std::string& name) {
    if (name == "seq") return GdnScan::kSeq;
    if (name == "collapse") return GdnScan::kCollapse;
    if (name == "chunk") {
        // FAILED CLOSED, deliberately, and this is the note rather than a comment on a throw: the
        // chunked scan is a killed route, not an experimental one. `progress/2026-10-14/p9-phase1-2026-10-14.md`
        // D-12 dropped it for launch-bound arithmetic (54.03 GFLOP per wave against route A's 33.94),
        // and `progress/2026-10-14/p9-phase1c-2026-10-14.md` §1c.1 D6 found the reason it could not simply be
        // repaired: `gdn_chunk_carry` formed the chunk-boundary decay as a quotient of two exps, which
        // is not the contract's one exp of a difference, and it returned NaN where the incoming state
        // was zero. A form that fails its own §04.4 budget because of an expression bug cannot carry an
        // error budget, so it does not get to look selectable from a flag. The kernel, `phase1.py` and
        // `kernel_price.py` stay in the tree as the record; the one-line repair D6 names is written
        // there for whoever re-funds the route.
        throw std::runtime_error(
            "--gdn-scan chunk is refused: the chunked scan was killed by progress/2026-10-14/p9-phase1-2026-10-14.md "
            "D-12 and its carry form is known-wrong (progress/2026-10-14/p9-phase1c-2026-10-14.md §1c.1 D6: boundary "
            "decay as gamC/gam[i] underflows to NaN at S=0). Use `seq` (the shipped form) or `collapse` "
            "(route A, bit-identical to seq). The prototype still runs it: prototype/gdn_chunk/run.py.");
    }
    throw std::runtime_error("--gdn-scan takes `seq` (§04.4's recurrence, one launch per token) or "
                             "`collapse` (route A: the same recurrence, resident, one launch per "
                             "body); got '" + name + "'. Refusing rather than guessing, because this "
                             "flag names which side of an A/B the run is");
}

const char* gdn_scan_name(GdnScan s) {
    switch (s) {
        case GdnScan::kSeq: return "seq";
        case GdnScan::kCollapse: return "collapse";
        case GdnScan::kChunk: return "chunk";
    }
    return "invalid";
}

GdnScan resolve_gdn_scan(GdnScan want, int rows) {
    // The one-row rule, stated once. See `kernels.h:resolve_gdn_scan` for why it is a necessity and
    // not a preference: decode's bit-identity is not a setting, and a one-token chunk has no
    // inter-chunk work for the chunked form to win.
    return rows == 1 ? GdnScan::kSeq : want;
}

const char* gdn_chunk_refusal(int tokens) {
    if (tokens == kGdnChunkTokens) return nullptr;
    if (tokens < kGdnChunkTokens) {
        // The number, not a bare "unsupported": C=16 was measured, on the pinned dumps, and lost.
        return "chunk sizes below 64 are measured and rejected, not unsupported: "
               "C=16 is outside §2.3's per-row budget (1-cos 3.388e-4 against the 3.2e-4 ceiling, "
               "progress/2026-10-14/p9-phase1-2026-10-14.md parent verification and prototype/gdn_chunk/phase1.py "
               "rows). C=32 IS inside that budget (best row number of the three) but the kernel is "
               "instantiated at C=64 only, and a second instantiation of a form already 0.3 % inside "
               "a stage line on p64 is not added unasked.";
    }
    return "the chunked kernel is instantiated at C=64 (GC_C), and nothing above 64 has been priced "
           "against §2.1/§2.3 — the CPU judge sweeps 16/32/64 only. A larger chunk is a new numerics "
           "claim, so it needs a new budget line, not a constant edit.";
}

// NOTE: unreachable from a flag since `parse_gdn_scan` refuses `chunk` (see the D-12 note there).
// The launcher and its scratch contract stay, because D-12 dropped the route for arithmetic reasons
// and the next funding round needs the form to be re-runnable, not reconstructed.
void gdn_chunk_prefill(const GdnChunkLayerArgs& a) {
    if (a.n < 2) {
        throw std::runtime_error("gdn_chunk_prefill: n=" + std::to_string(a.n) + "; a one-row body is"
                                 " `resolve_gdn_scan`'s to route to gdn_rec, and reaching this"
                                 " launcher with n<2 means a caller bypassed that rule. Decode bits"
                                 " depend on it.");
    }
    if (a.u == nullptr || a.g == nullptr || a.beta == nullptr || a.S == nullptr || a.o == nullptr ||
        a.nv == nullptr || a.kcd == nullptr || a.qg == nullptr || a.kh == nullptr ||
        a.intra == nullptr || a.cdec == nullptr) {
        throw std::runtime_error(
            "gdn_chunk_prefill: a required buffer is null. The staging is not optional: `carry` reads"
            " every array `prep` writes, and a null one is a dropped stage rather than a skipped"
            " dump — `decode_step.cpp` allocates all six in `allocate_scratch` and frees all six.");
    }
    ::GdnChunkArgs k{};                      // the op's own struct, spelled in its header
    k.u = a.u;
    k.g = a.g;
    k.beta = a.beta;
    k.S = a.S;
    k.o = a.o;
    k.nv = a.nv;
    k.kcd = reinterpret_cast<hip_bfloat16*>(a.kcd);
    k.qg = reinterpret_cast<hip_bfloat16*>(a.qg);
    k.kh = reinterpret_cast<hip_bfloat16*>(a.kh);
    k.intra = reinterpret_cast<hip_bfloat16*>(a.intra);
    k.cdec = a.cdec;
    k.n = a.n;
    k.nc = (a.n + GC_C - 1) / GC_C;        // the ragged tail is the kernel's to pad (§04.7 A-08)
    k.neg = GCK_NONE;                       // the engine is not a place to inject bugs
    // grid (chunks, heads): every block independent, none of them touches S.
    gdn_chunk_prep<<<dim3(static_cast<unsigned>(k.nc), GC_NH), GC_THREADS>>>(k);
    // grid (d_v/32, heads) = 64 blocks: the ONLY serial part, one block per state strip, walking
    // the chunk axis with that strip resident in LDS.
    gdn_chunk_carry<<<dim3(GC_D / GC_STRIP, GC_NH), GC_THREADS>>>(k);
}

void gdn_gated_norm(const float* o, const uint16_t* z, const float* w_norm, uint16_t* out,
                    uint16_t* norm_out, int tokens) {
    // One block per (token, value head) with the heads innermost, so tokens == 1 degenerates to the
    // `<<<GN_HEADS, GN_D_V>>>` this launcher has always issued — same grid, same per-head 128-thread
    // reduction, same bits — and a wave is that block repeated down the token axis.
    ::gdn_gated_norm<<<GN_HEADS * tokens, GN_D_V>>>(o, (const hip_bfloat16*)z, w_norm,
                                                   (hip_bfloat16*)out,
                                                   (hip_bfloat16*)norm_out, GN_NONE, tokens);
    launchstats::gdn_gated_norm().record(tokens);
}

}  // namespace engine::kernels
