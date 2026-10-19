// Launchers for the GDN family: prototype/gdn_decode (§04.2–§04.4) and prototype/gdn_gated_norm
// (§04.5). Separate TU so their macros (D, NH, CH, K, GN_D_V, …) cannot collide with another op's.
#include "kernels.h"

#include "engine/launch_stats.h"
#include "engine/state.h"   // kQkvRows / kValueDim: the tile row strides the call site walks
#include "engine/gdn_collapse_kernels.h"
#include "engine/gdn_rec_fused_kernels.h"
#include "engine/gdn_rec_hoist_kernels.h"
#include "engine/gdn_rec_loop_kernels.h"
#include "engine/gdn_rec_unroll4_kernels.h"
#include "prototype/gdn_chunk/gdn_chunk_kernels.h"
#include "prototype/gdn_decode/gdn_decode_kernels.h"
#include "prototype/gdn_gated_norm/gdn_gated_norm_kernels.h"

namespace engine::kernels {

// §04.2 over a wave. grid.x is the channel strip — CH/(256*CONV_VEC) blocks, i.e. the 24 the
// single-token launch used at one channel per lane, four lanes wide now — and grid.y the token, so
// `tokens == 1` is the same grid, not a grid padded to one, and the per-element arithmetic (F32 taps
// in j order, one BF16 round, SiLU) is the one `prototype/gdn_decode/check.py` has always pinned
// against `q/k/v_post_conv`. What the wave adds is the window rule: the three columns before row 0
// come from `win` (§08.2's carried window), never from zeros, and the launch writes the window's new
// contents once for the whole wave instead of once per token — the copy `decode_step.cpp` used to
// issue per token is what collapses here.
//
// `raw` is the BF16 tile `in_proj_qkv` wrote, not a widening of it (D30 1.7'): the kernel widens the
// taps in-register, a BF16->F32 widening is exact, and so the F32 values the taps take are the same
// numbers the deleted `cast_bf16_to_f32_3` used to store. `win`/`new_win` stay F32 — that is §08.1's
// state dtype and a separate question from the tile's (§08.3, `engine/state.h`'s note).
void gdn_conv(const uint16_t* raw, const float* win, const float* conv_weight, float* u,
              float* new_win, int tokens) {
    // A lane owns CONV_VEC consecutive channels, so the strip is CONV_VEC blocks narrower than the
    // 24 it was: 6 blocks of 256 for the whole [CH] row, and `tokens == 1` is still that same grid.
    // The divisor is asserted here rather than trusted, because a non-dividing CH would silently hand
    // the last channels to nobody (the kernel's own guard returns instead of wrapping).
    static_assert(CH % (256 * CONV_VEC) == 0, "conv_kernel's grid must be a whole number of blocks");
    conv_kernel<<<dim3(CH / (256 * CONV_VEC), tokens), 256>>>((const hip_bfloat16*)raw, win,
                                                             conv_weight, u, new_win, tokens,
                                                             GCK_NONE);
    launchstats::gdn_conv().record(tokens);
}

// §04.3: `<<<tokens, NH>>>`, one thread per (token, value head), so a decode step is still the
// one-block launch it was and no element expression moved. `a_raw`/`b_raw` are the BF16 tiles
// `in_proj_a`/`in_proj_b` wrote, widened in-register for the same reason and to the same effect as
// the conv's `raw` above.
void gdn_gates(const uint16_t* a_raw, const uint16_t* b_raw, const float* a_log,
               const float* dt_bias, float* g, float* beta, int tokens) {
    gates_kernel<<<tokens, NH>>>((const hip_bfloat16*)a_raw, (const hip_bfloat16*)b_raw, a_log,
                                 dt_bias, g, beta, tokens, GG_NONE);
    launchstats::gdn_gates().record(tokens);
}

void gdn_rec(const float* q, const float* k, const float* v, const float* g, const float* beta,
             float* S, float* o, GdnRecSpell spell) {
    // One layer per launch: the model interleaves GDN and attention layers (§03.5), so a token
    // never has two adjacent GDN layers to batch. `all` mode (18 layers, one launch) in the
    // prototype is a bandwidth-measurement shape, not a token shape.
    //
    // Which SPELLING runs (`--gdn-spell`, `kernels.h::GdnRecSpell`). The launches differ in the
    // called kernel and in nothing else — same `<<<NH, D>>>`, same operands, same state buffer, so the
    // flag cannot move an address and an A/B between them measures the schedule rather than the plan
    // (the `attn_slim_kernels.h` / `--attn-touch` shape, one level down the tree). The third branch is
    // the merged fused-prologue form (`engine/gdn_rec_fused_kernels.h`): same grid, same operands, and
    // its own +128 B of static LDS is the whole resource delta, which does not bind at this geometry.
    if (spell == GdnRecSpell::kUnroll4) {
        rec_unroll4_kernel<<<NH, D>>>(q, k, v, g, beta, S, o, NH);
    } else if (spell == GdnRecSpell::kFused) {
        rec_fused_kernel<<<NH, D>>>(q, k, v, g, beta, S, o, NH);
    } else {
        rec_kernel<<<NH, D>>>(q, k, v, g, beta, S, o, NH);
    }
    // Not counted: this launcher has no token width (see `engine/launch_stats.h`), and a family
    // that can only ever report rows=1 would give section (h) a check that cannot fail.
}

// D53/D54: the same recurrence, `tokens` token-steps per submission (`--gdn-loop`). One launch
// `<<<NH, D>>>` over the whole tile, which is the shape `gdn_rec_collapse` already submits — the two
// batched forms therefore agree on which buffers they are handed, and the only difference between
// them is that this one keeps the shipping per-token arithmetic verbatim (transcribed and checked,
// see engine/gdn_rec_loop_kernels.h) while route A re-spells the thread mapping.
//
// Not counted in `launch_stats` for `gdn_rec`'s reason: the recurrence's launch count is the thing
// under test, and it is already the `gdn_rec` row's `rows_total` (`engine/stage_clock.h`). The
// stage-clock marker stays at the CALL SITE in `decode_step.cpp`, which is what makes the counted
// `family=gdn_rec launches=` the behavioural canary for this flag — one note per launch, so the
// counted row goes from `G x tokens` to `G x bodies` because the launches really did fall, not
// because a bill line was edited.
void gdn_rec_span(const float* u, const float* g, const float* beta, float* S, float* o,
                  int tokens, GdnRecSpell spell) {
    // The kernel walks `CH` per token-step on the q|k|v tile and `NH*D` on `o`; the call site walks
    // `kQkvRows` and `kValueDim`. Those are the same numbers, and this is where the equality is
    // worth making a compile error: a stride that only looked equal is a silent re-map of every
    // operand after row 0, which is a numerics change with no value to catch it. `weights_contract.py`
    // states the same rule for a shape claim: state the rank at the use site, never let the checker
    // guess it.
    static_assert(CH == kQkvRows, "rec_loop_kernel's q|k|v row stride (CH) must be the tile stride"
                                  " the call site walks (kQkvRows)");
    static_assert(NH * D == kValueDim, "rec_loop_kernel's o row stride (NH*D) must be the tile"
                                       " stride the call site walks (kValueDim)");
    if (tokens < 2) {
        // Not a style rule. `resolve_gdn_loop` is the one place the one-row exception lives, and a
        // caller that reached here with one row bypassed it — which would move every decode bit
        // through a kernel the decode pins were not pinned through, silently. `gdn_chunk_prefill`
        // refuses its own bypass for the same reason.
        throw std::runtime_error("gdn_rec_span: tokens=" + std::to_string(tokens) + "; a one-row"
                                 " body is `resolve_gdn_loop`'s to route to `gdn_rec`, and reaching"
                                 " this launcher with <2 rows means a caller bypassed that rule."
                                 " Decode bits depend on it.");
    }
    if (spell != GdnRecSpell::kFused) {
        // The same discipline as `parse_gdn_rec_spell` and `parse_gdn_scan`: this flag names which
        // side of an A/B the run is, so a launcher that quietly ran a different body than the one it
        // was asked for would price one kernel twice and report the second price as the loop's win.
        // Only the fused body has a looped transcription (`engine/gdn_rec_loop_kernels.h`) and only
        // the fused form has device bit-identity admission at spans 0/1/3/16, so `ship`/`unroll4`
        // under `--gdn-loop N>1` is refused rather than served by the shipping kernel.
        throw std::runtime_error(std::string("gdn_rec_span: --gdn-spell ") + gdn_rec_spell_name(spell)
                                 + " has no looped kernel and no bit-identity admission for a"
                                 " multi-token span. --gdn-loop N>1 runs"
                                 " engine/gdn_rec_loop_kernels.h, which is the `fused` body; use"
                                 " --gdn-spell fused (the default) or --gdn-loop 1");
    }
    // `u` is the conv's [n][6144] tile and the three slices are row 0 of q, k and v; the kernel adds
    // `CH` per token-step itself, exactly as the call site used to add it per launch.
    rec_loop_kernel<<<NH, D>>>(u, u + kValueDim, u + 2 * kValueDim, g, beta, S, o, NH, tokens);
}

// bet G1: the same span, TWO launches. The ordering is a dependency, not a preference — the walk reads
// the stage tiles the pre-pass writes — and same-stream order is what orders them, exactly as
// `attn_prep`/`attn_core` are ordered above (`kernels_attn.cpp`). Same-stream order is also why the two
// launches are issued adjacently per SPAN rather than one pre-pass for a whole body: the stage tiles
// stay as small as a span and the walk touches them while they are the freshest bytes on the chip.
void gdn_rec_span_hoisted(const float* u, const float* g, const float* beta, float* S, float* o,
                          const GdnNormStage& stage, int tokens, GdnRecSpell spell) {
    // The same three refusals `gdn_rec_span` makes, for the same three reasons: a one-row body is
    // `resolve_gdn_loop`'s to route, a non-`fused` spelling has no looped transcription to hoist a
    // prologue out of, and a stage tile is not optional — a null one is a dropped launch that looks
    // like a skipped stage rather than a crash.
    if (tokens < 2) {
        throw std::runtime_error("gdn_rec_span_hoisted: tokens=" + std::to_string(tokens) +
                                 "; a one-row body is `resolve_gdn_loop`'s to route to the per-token"
                                 " kernels, which keep their own norm prologue. Decode bits depend on"
                                 " that routing.");
    }
    if (stage.kn == nullptr || stage.qn == nullptr || stage.gsc == nullptr) {
        throw std::runtime_error("gdn_rec_span_hoisted: a stage tile is null. The norm pre-pass is not"
                                 " optional under --gdn-norm hoist; the walk reads exactly what it"
                                 " writes.");
    }
    if (spell != GdnRecSpell::kFused) {
        throw std::runtime_error(std::string("gdn_rec_span_hoisted: --gdn-spell ") +
                                 gdn_rec_spell_name(spell) + " has no looped body to hoist; the"
                                 " hoisted walk is the `fused` body's ("
                                 "engine/gdn_rec_hoist_kernels.h). Use --gdn-spell fused (the"
                                 " default) or --gdn-loop 1");
    }
    static_assert(CH == kQkvRows, "the hoist pass's q|k row stride (CH) must be kQkvRows");
    static_assert(NH * D == kValueDim, "the kn/qn stage row stride (NH*D) must be kValueDim");
    static_assert(kGdnScalarRow == 2 * kNumValueHeads,
                  "the scalar stage row is gam then beta for every value head");
    // `<<<dim3(NH, tokens), D>>>`: one block per (head, token-step), 128 threads. The kernel reads
    // `h = blockIdx.x` and the token-step from `blockIdx.y` — the same axis order the walk's grid uses,
    // so a swap here is not a wrong-value bug you can spot in a logits diff, it is the pre-pass writing
    // normalised rows for head 46 into a 16-head buffer. `block_sum_pair_f32`'s fold width is
    // `blockDim.x >> 5` and the shipped prologue's fold is a 4-warp one, so `D` as the block width is
    // part of the bit-identity argument, not a convenience.
    gdn_norm_hoist_kernel<<<dim3(NH, tokens), D>>>(u, u + kValueDim, g, beta, stage.kn, stage.qn,
                                                   stage.gsc, NH, tokens);
    // The walk never sees q|k: `v` is still the conv tile's third slice, and the two normalised rows
    // plus the scalar row are the stage tiles this launch just filled.
    rec_loop_normed_kernel<<<NH, D>>>(u + 2 * kValueDim, stage.kn, stage.qn, stage.gsc, S, o,
                                      NH, tokens);
}

// `--gdn-norm`, bet G1. Parses or throws, like every other knob that names a side of an A/B.
GdnNorm parse_gdn_norm(const std::string& name) {
    if (name == "inline") return GdnNorm::kInline;
    if (name == "hoist") return GdnNorm::kHoist;
    throw std::runtime_error("--gdn-norm takes `inline` (the shipped looped body, which reduces each"
                             " token's k and q rows inside the walk — THE DEFAULT) or `hoist`"
                             " (engine/gdn_rec_hoist_kernels.h: a pre-pass computes those norms and"
                             " the walk reads them, so the walk carries no barrier). Refusing"
                             " rather than guessing, because this flag names which side of an A/B the"
                             " run is");
}

const char* gdn_norm_name(GdnNorm n) {
    switch (n) {
        case GdnNorm::kInline: return "inline";
        case GdnNorm::kHoist: return "hoist";
    }
    return "invalid";
}

int resolve_gdn_loop(int want, int rows) {
    // The one-row rule, stated once. `rows <= 1` covers both a decode step and `--wave-rows 1`, and
    // `want <= 1` is the shipped default; either way the answer is the per-token loop, verbatim.
    // A span wider than the body is left alone on purpose — `ceil(rows/span) == 1`, so the body is
    // one launch and `--gdn-loop 512` means "one launch per (layer, body)" at every width.
    return (rows <= 1 || want <= 1) ? 1 : want;
}

// `--gdn-spell`, and it parses or throws for the standing reason: the flag names which side of an A/B
// the run is. `unroll4` is the shipping default (D33); `ship` is kept so that claim stays A/B-able from
// one binary; `fused` is D50's knob — the fused prologue on the unroll4 walks, bit-identical, and NOT a
// default until its in-situ A/B is read (`prototype/gdn_rec_isa/results.md` §11,
// `bench/performance-status.md` §19).
GdnRecSpell parse_gdn_rec_spell(const std::string& name) {
    if (name == "ship") return GdnRecSpell::kShip;
    if (name == "unroll4") return GdnRecSpell::kUnroll4;
    if (name == "fused") return GdnRecSpell::kFused;
    throw std::runtime_error("--gdn-spell takes `ship` (the pre-D33 `rec_kernel`, kept as an A/B side), "
                             "`unroll4` (engine/gdn_rec_unroll4_kernels.h, bit-identical, unroll-4 "
                             "state walks, THE DEFAULT) or `fused` (engine/gdn_rec_fused_kernels.h, "
                             "bit-identical, the unroll4 walks with the q/k norm prologue fused); got "
                             "'" + name + "'. Refusing rather than guessing, because "
                             "this flag names which side of an A/B the run is");
}

const char* gdn_rec_spell_name(GdnRecSpell s) {
    switch (s) {
        case GdnRecSpell::kShip: return "ship";
        case GdnRecSpell::kUnroll4: return "unroll4";
        case GdnRecSpell::kFused: return "fused";
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
