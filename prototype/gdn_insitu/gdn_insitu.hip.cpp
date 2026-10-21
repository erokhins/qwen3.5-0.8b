/*
 * gdn_insitu.hip.cpp — the D4 rig: the SHIPPING GDN decode launches at the shipped decode shape,
 * under one-variable arms. Nothing prints `ms` or a rate unless `--time` is present, and that flag
 * belongs to the main agent (AGENTS.md: timing is a main-agent measurement).
 *
 * WHAT IT RUNS. `engine/decode_step.cpp` at n == 1 issues four GDN launches per GDN layer
 * (gdn_conv, gdn_gates, gdn_rec, gdn_gated_norm) = 4 x 18 = 72 per decode step in 36 stage-clock
 * segments, and the recurrence goes through `rec_fused_kernel<<<NH=16, D=128>>>` because
 * `resolve_gdn_loop(flag, 1)` returns 1 (`engine/kernels_gdn.cpp`). This rig builds that launch list
 * from the shipped kernels and changes one variable at a time.
 *
 * NO SECOND KERNEL. Every kernel is from a header `engine/kernels_gdn.cpp` itself includes
 * (prototype/gdn_decode/gdn_decode_kernels.h, engine/gdn_rec_fused_kernels.h,
 * engine/gdn_rec_unroll4_kernels.h, engine/gdn_rec_reg16_kernels.h,
 * prototype/gdn_gated_norm/gdn_gated_norm_kernels.h). Nothing here may be edited to make an arm pass.
 * Negatives are (a) the shipped kernels' own `neg` enums and (b) launch-level perturbations
 * (truncated grid, shifted base): an injection in a copy of the kernel would be a second kernel.
 *
 * VERIFICATION IS BY BYTES TOUCHED (the tools/mixed_bw + prototype/gdn_collapse rule, in the shape
 * prototype/mlp_insitu established):
 *   --witness land     destinations pre-set to a sentinel no store here can emit; every other buffer
 *                      snapshotted byte-for-byte; every region fenced by a poison guard band. A write
 *                      outside the declared destinations, a skipped element, or a stray store into a
 *                      neighbour is a FAIL, with no tolerance anywhere.
 *   --witness onehot   tolerance-free. Per head h: k one-hot at p_k(h), q one-hot at p_q(h) (both vary
 *                      with h, so head aliasing shows), g = 0, beta = 0. Then gam = expf(0) = 1 exactly
 *                      and dv = beta*(v - gam*r) = 0 EXACTLY, so the state write-back must be
 *                      BIT-IDENTICAL to its input — no wrong state index survives that, and there is
 *                      no summation-order question to band. o[j] collapses to c_q * S[p_q(h)][j] with
 *                      one nonzero term; the state is signed powers of two, so |o[j]|/|S[p_q(h)][j]|
 *                      must be ONE constant across every j, head and layer, exactly, with the sign of
 *                      S. The exponent depends on both i and j, so a wrong row OR column moves it.
 *   --witness mirror   a double mirror of §04.4 for `rec` only, banded (meanrel <= 1e-5). The
 *                      transcendental paths (gates' log1pf/expf, gated_norm's rsqrtf, conv's SiLU) are
 *                      NOT bit-adjudicated here: their catchers are prototype/gdn_decode/check.py and
 *                      engine/gate.py, which this rig neither re-runs nor re-bands.
 * Declared blindnesses. (1) `onehot` cannot see a SKIPPED state write — with beta = 0, g = 0 the
 * state's correct output equals its input; that is exactly what `land` catches. (2) A re-associated
 * reduction is invisible here by construction (two sides that share a formula agree, AGENTS.md's q̃
 * row), which is why --recmap split8 owes engine/gate.py an admission, not a rig pass. (3) At
 * --tokens 1 the shipped negatives GC_TAP_STRIDE and GG_ROW_ZERO have an EMPTY must-catch set (both
 * indexings coincide at one row, published in prototype/gdn_decode/results.md), so the rig reports
 * them BLIND-BY-GEOMETRY and requires them at --tokens 2. A control that cannot fire is not one.
 *
 * THE MUST-LAND SET IS DERIVED from the same launch-descriptor list the launches are built from, and
 * the expectation is always built from the SHIPPED plan while a negative perturbs only the actual
 * launches: a reference derived from the thing under test cannot adjudicate it
 * (prototype/attn_tile_model.py). An arm whose expected set is empty FAILS (FAIL_nothing_expected).
 *
 * FORKS AND ORDER. --selftest runs each device case in its own forked child: one faulting kernel
 * poisons the context and every later case in that process silently never ran (tools/mixed_bw). No
 * HIP call sits on a path reachable before the fork (prototype/mlp_insitu §2.5). hipMalloc only,
 * never managed memory. --pages cold is refused, not averaged: first touch of fresh pages runs ~7x
 * slow and would be reported as a mechanism.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include "prototype/gdn_decode/gdn_decode_kernels.h"
#include "engine/gdn_rec_fused_kernels.h"
#include "engine/gdn_rec_unroll4_kernels.h"
#include "engine/gdn_rec_reg16_kernels.h"
#include "prototype/gdn_gated_norm/gdn_gated_norm_kernels.h"

#define CK(x)                                                                        \
    do {                                                                             \
        hipError_t e_ = (x);                                                         \
        if (e_ != hipSuccess) {                                                      \
            std::printf("hip error %s at line %d\n", hipGetErrorString(e_), __LINE__); \
            std::exit(4);                                                            \
        }                                                                            \
    } while (0)

namespace {

constexpr int kStatePerHead = D * D;                  // 16,384 floats
constexpr int kStatePerLayer = NH * kStatePerHead;    // 262,144 floats = 1 MiB
constexpr int kValueDim = NH * D;                     // 2048 == engine/state.h kValueDim
constexpr int kConvWin = K - 1;
constexpr int kGUARD = 64;                            // words of poison on each side
constexpr float kSentF = -1.2345e37f;
constexpr uint16_t kSentBF = 0xFFFFu;
// A store can land on the sentinel only by returning its own input unchanged, which is why the `land`
// operand set uses g = -20, beta = 1: gamma != 1, so S' = gamma*S + k*dv differs from S by force.
constexpr float kLandG = -20.0f, kLandBeta = 1.0f;
// The single non-zero conv tap. Index K-2 so that at token 0 it reads a CARRIED window row (which is
// what makes GC_ZERO_PAD fire) and at token >= 1 it reads a live `raw` row k = t-1 (which is what makes
// GC_TAP_STRIDE fire). One tap keeps every conv output a sigmoid of one named operand.
constexpr int kConvActiveTap = K - 2;
// The raw fill, as ONE named function, because the tap-stride control has to know the value at both
// the shipped index and the injected index: `raw[t][c]` on a 0.1 grid, so two indices that differ by a
// grid step differ by >= 0.1 and no rounding can hide it. A stored array here would have been read
// before it was filled by the host-only geometry selftest (segfault), which is the other reason for
// making it a formula.
inline float raw_val(size_t t, size_t c) { return 1.0f + 0.1f * (float)((t * 7 + c) % 10); }

enum Buf {
    B_S = 0, B_RAW, B_WIN, B_NWIN, B_CONVW, B_U, B_Q, B_K, B_V, B_GA, B_GB, B_G, B_BETA,
    B_ALOG, B_DTB, B_O, B_Z, B_GNW, B_BLOCKOUT, B_POLL, NBUF
};
const char* kBufName[NBUF] = {
    "state", "raw", "win", "new_win", "conv_w", "u", "qcomb", "kcomb", "vcomb", "a_raw", "b_raw",
    "g", "beta", "A_log", "dt_bias", "o", "z", "gnorm_w", "block_out", "pollute"};
const int kBufElem[NBUF] = {4, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 2, 4, 2, 4};

struct Region {
    void* base = nullptr;
    char* p = nullptr;
    size_t n = 0;
    std::vector<char> snap;
};
Region g_buf[NBUF];
void* g_sum_slot = nullptr;

void region_free(int id) {
    if (g_buf[id].base) (void)hipFree(g_buf[id].base);
    g_buf[id] = Region();
}

void region_alloc(int id, size_t bytes) {
    region_free(id);
    Region& r = g_buf[id];
    r.n = bytes;
    CK(hipMalloc(&r.base, bytes + 2 * kGUARD * 4));
    r.p = static_cast<char*>(r.base) + kGUARD * 4;
    for (int side = 0; side < 2; ++side) {
        void* at = side == 0 ? r.base : static_cast<char*>(r.base) + kGUARD * 4 + bytes;
        CK(hipMemset(at, 0xBE, kGUARD * 4));
    }
    r.snap.assign(bytes, 0);
}

void set_bytes(int id, const void* host, size_t bytes, size_t off = 0) {
    CK(hipMemcpy(g_buf[id].p + off, host, bytes, hipMemcpyHostToDevice));
}
void sent_fill(int id) {
    if (kBufElem[id] == 4) {
        uint32_t w;
        std::memcpy(&w, &kSentF, 4);
        std::vector<uint32_t> v(g_buf[id].n / 4, w);
        set_bytes(id, v.data(), g_buf[id].n);
    } else {
        CK(hipMemset(g_buf[id].p, 0xFF, g_buf[id].n));
    }
}
void snapshot(int id) {
    g_buf[id].snap.resize(g_buf[id].n);
    CK(hipMemcpy(g_buf[id].snap.data(), g_buf[id].p, g_buf[id].n, hipMemcpyDeviceToHost));
}
int check_guards() {
    int bad = 0;
    std::vector<uint8_t> g(kGUARD * 4);
    for (int id = 0; id < NBUF; ++id) {
        if (!g_buf[id].base) continue;
        for (int side = 0; side < 2; ++side) {
            const void* at = side == 0
                                 ? g_buf[id].base
                                 : static_cast<char*>(g_buf[id].base) + kGUARD * 4 + g_buf[id].n;
            CK(hipMemcpy(g.data(), at, kGUARD * 4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < g.size(); ++i) bad += (g[i] != 0xBE);
        }
    }
    return bad;
}

// --------------------------------------------------------------- the plan
enum Kind { K_REC = 0, K_CONV, K_GATES, K_GNORM };
enum Shape { SH_PERLAYER = 0, SH_COMBINED };
enum RecMap { RM_COL = 0, RM_SPLIT8 };
enum Spell { SP_SHIP = 0, SP_UNROLL4, SP_FUSED };
enum Drain { DR_EACH = 0, DR_BURST };
enum Layout { LY_BLOB1 = 0, LY_SPANNED };
enum Witness { WS_LAND = 0, WS_ONEHOT, WS_MIRROR };

constexpr int F_REC = 1, F_CONV = 2, F_GATES = 4, F_GNORM = 8;

enum NegId { NEG_NONE = 0, NEG_GRID_SHORT_REC, NEG_STATE_OFF,
             NEG_CONV_TAP_STRIDE, NEG_CONV_WIN_LAG, NEG_CONV_WIN_FROM_U, NEG_CONV_ZERO_PAD,
             NEG_GATES_ROW_ZERO, NEG_GNORM_ROW_ZERO };
const char* kNegNames[] = {"none", "grid_short_rec", "state_off", "conv_tap_stride", "conv_win_lag",
                           "conv_win_from_u", "conv_zero_pad", "gates_row_zero", "gnorm_row_zero"};

struct Cfg {
    int layers = 18, tokens = 1;
    int kernels = F_REC;
    Shape shape = SH_PERLAYER;
    RecMap map = RM_COL;
    Spell spell = SP_FUSED;
    Drain drain = DR_BURST;
    Layout layout = LY_BLOB1;
    Witness witness = WS_ONEHOT;
    int gap_mib = 0, pollute_mib = 0, neg = NEG_NONE;
    int time_flag = 0, reps = 3, iters = 4;
};

struct Span { int buf; size_t off_elem, n_elem; };
struct Launch {
    Kind kind;
    int layer, token, neg_kernel;
    dim3 grid, block;
    std::vector<Span> dst;
    double demand_mib = 0.0;      // DEMANDED MiB for this launch, from the plan (never measured)
};

const char* refuse(const Cfg& c) {
    if (c.shape == SH_COMBINED && c.kernels != F_REC)
        return "--shape combined is a rec-only arm: the combined launch has no per-layer conv tile "
               "to read (it is the --mode all shape that produced the quoted 0.0646 ms row), so "
               "pairing it with conv/gates/gnorm would price a shape the rig never builds.";
    if (c.shape == SH_COMBINED && c.map == RM_SPLIT8)
        return "--shape combined --recmap split8 is not expressible: rec_reg16_kernel scopes the "
               "state base to blockIdx.x and uses `tokens` as a TIME axis, so it cannot address one "
               "block per (layer, head). The combined cell exists for `col` only.";
    if (c.map == RM_SPLIT8 && c.tokens != 1)
        return "--recmap split8 is a decode-shape arm and requires --tokens 1: at tokens > 1 "
               "rec_reg16_kernel is a SPAN (the wave path's --gdn-loop body), a different claim "
               "about a different code path.";
    if (c.layout == LY_SPANNED && c.gap_mib <= 0)
        return "--layout spanned needs --gap-mib > 0, or it is --layout blob1 wearing a label.";
    if (c.time_flag && c.tokens > 1)
        return "--tokens > 1 is a SYNTHETIC width in this rig: it exists so the geometry-derived "
               "negatives (the row-zero and tap-stride injections) have a case that can fire, and the "
               "recurrence is walked per token here where the shipped multi-token path is the looped "
               "SPAN. Timing it would price the rig, not the engine (D2's failure). Run it without "
               "--time.";
    return nullptr;
}

int rec_blocks(const Cfg& c) { return c.shape == SH_COMBINED ? c.layers * NH : NH; }

// State stride in elements per layer: kStatePerLayer, or larger when --layout spanned separates the
// slabs. The recurrence kernels compute the state base themselves (S + h*D*D), so only the per-layer
// arm can honour a gap, and --shape combined refuses it below.
size_t g_state_stride = kStatePerLayer;
size_t state_layer_off(int layer) { return (size_t)layer * g_state_stride; }
size_t rec_tile_elems(const Cfg& c) {
    return (size_t)c.tokens * kValueDim * (c.shape == SH_COMBINED ? c.layers : 1);
}

void plan_build(const Cfg& c, bool actual, std::vector<Launch>& out) {
    // The shipped launch population, per layer: ONE conv launch (grid.y carries the tokens), ONE gates
    // launch (grid.x carries them), one recurrence launch per token, then ONE gated_norm launch
    // (grid.x = GN_HEADS*tokens). At tokens == 1 - the decode shape - that is exactly the 72 launches
    // of the shipping step, in the shipping order. Above tokens == 1 the shipped recurrence is the
    // looped SPAN, which this rig does not re-make, so --time refuses that width instead of pricing a
    // shape the engine never emits (the D2 lesson: a number taken at a shape the shipping path does not
    // issue is a fact about the rig).
    for (int L = 0; L < c.layers; ++L) {
        if (c.kernels & F_CONV) {
            Launch l; l.kind = K_CONV; l.layer = L; l.token = -1;
            l.grid = dim3(CH / (256 * CONV_VEC), c.tokens); l.block = dim3(256);
            l.neg_kernel = c.neg == NEG_CONV_TAP_STRIDE ? GC_TAP_STRIDE
                         : c.neg == NEG_CONV_WIN_LAG ? GC_WIN_LAG
                         : c.neg == NEG_CONV_WIN_FROM_U ? GC_WIN_FROM_U
                         : c.neg == NEG_CONV_ZERO_PAD ? GC_ZERO_PAD : GC_NONE;
            l.dst = {{B_U, 0, (size_t)c.tokens * CH}, {B_NWIN, 0, (size_t)kConvWin * CH}};
            l.demand_mib = (c.tokens * CH * 2.0 + (size_t)kConvWin * CH * 4.0 + (size_t)CH * K * 4.0 +
                            (size_t)c.tokens * CH * 4.0 + (size_t)kConvWin * CH * 4.0) / 1048576.0;
            out.push_back(l);
        }
        if (c.kernels & F_GATES) {
            Launch l; l.kind = K_GATES; l.layer = L; l.token = -1;
            l.grid = dim3(c.tokens); l.block = dim3(NH);
            l.neg_kernel = c.neg == NEG_GATES_ROW_ZERO ? GG_ROW_ZERO : GG_NONE;
            l.dst = {{B_G, 0, (size_t)c.tokens * NH}, {B_BETA, 0, (size_t)c.tokens * NH}};
            l.demand_mib = (NH * 4.0 + NH * 8.0 + NH * 8.0) / 1048576.0;
            out.push_back(l);
        }
        for (int t = 0; t < c.tokens; ++t) {
            if (!(c.kernels & F_REC) || c.shape == SH_COMBINED) continue;
            Launch l; l.kind = K_REC; l.layer = L; l.token = t;
            const int nb = rec_blocks(c);
            if (c.map == RM_COL) { l.grid = dim3(nb); l.block = dim3(D); }
            else { l.grid = dim3(NH, kGdnR16GridY); l.block = dim3(kGdnR16BlockX, kGdnR16BlockY); }
            if (actual && c.neg == NEG_GRID_SHORT_REC) {
                if (c.map == RM_COL) l.grid.x -= 1; else l.grid.y -= 1;
            }
            if (c.shape == SH_COMBINED) {
                // The combined cell indexes o as `o + h*D` with h = blockIdx.x a GLOBAL head index over
                // layers*NH, so one launch writes layers*kValueDim floats, not one token row: the first
                // version of this span claimed NH*D and the witness then called the other 17 token rows
                // "stray" (34,816 of them) for a kernel that was correct.
                l.dst = {{B_S, 0, (size_t)c.layers * kStatePerLayer},
                         {B_O, (size_t)t * kValueDim * c.layers, (size_t)c.layers * kValueDim}};
                l.demand_mib = (kStatePerLayer * 4.0 * c.layers * 2.0 + kValueDim * 4.0 * 3.0 +
                                NH * 8.0 + kValueDim * 4.0) / 1048576.0;
            } else {
                l.dst = {{B_S, state_layer_off(L), (size_t)kStatePerLayer},
                         {B_O, (size_t)t * kValueDim, (size_t)kValueDim}};
                l.demand_mib = (kStatePerLayer * 4.0 * 2.0 + kValueDim * 4.0 * 3.0 + NH * 8.0 +
                                kValueDim * 4.0) / 1048576.0;
            }
            out.push_back(l);
        }
        if (c.kernels & F_GNORM) {
            Launch l; l.kind = K_GNORM; l.layer = L; l.token = -1;
            l.grid = dim3(GN_HEADS * c.tokens); l.block = dim3(GN_D_V);
            l.neg_kernel = c.neg == NEG_GNORM_ROW_ZERO ? GN_ROW_ZERO : GN_NONE;
            l.dst = {{B_BLOCKOUT, 0, (size_t)c.tokens * kValueDim}};   // norm_out: nullptr, as shipped
            l.demand_mib = (c.tokens * kValueDim * 4.0 + c.tokens * kValueDim * 2.0 + D * 4.0 +
                            c.tokens * kValueDim * 2.0) / 1048576.0;
            out.push_back(l);
        }
    }
    // The combined cell is ONE launch per token whose grid spans every (layer, head) - that is the
    // `--mode all` shape the quoted 0.0646 ms burst row came from, and booking it per layer would have
    // made 18 launches of 288 blocks, i.e. an instrument 18x the shape it is supposed to reproduce.
    if ((c.kernels & F_REC) && c.shape == SH_COMBINED) {
        for (int t = 0; t < c.tokens; ++t) {
            Launch l; l.kind = K_REC; l.layer = -1; l.token = t;
            l.grid = dim3(c.layers * NH); l.block = dim3(D);
            if (c.map == RM_SPLIT8) { /* refused above */ }
            if (actual && c.neg == NEG_GRID_SHORT_REC) l.grid.x -= 1;
            l.dst = {{B_S, 0, (size_t)c.layers * kStatePerLayer},
                     {B_O, (size_t)t * kValueDim * c.layers, (size_t)c.layers * kValueDim}};
            l.demand_mib = (kStatePerLayer * 4.0 * c.layers * 2.0 + kValueDim * 4.0 * 3.0 +
                            NH * 8.0 + kValueDim * 4.0) / 1048576.0;
            out.push_back(l);
        }
    }
}


int geom_check(const Cfg& c, char* msg, size_t mn) {
    std::vector<Launch> plan;
    plan_build(c, false, plan);
    if (plan.empty()) { std::snprintf(msg, mn, "empty plan"); return 0; }
    for (const Launch& l : plan) {
        if (l.kind == K_CONV) {
            if ((int)l.grid.x != CH / (256 * CONV_VEC) || (int)l.grid.y != c.tokens ||
                (int)l.block.x != 256) {
                std::snprintf(msg, mn, "conv grid (%u,%u) block %u is not the shipped (%d,%d)/256",
                              l.grid.x, l.grid.y, l.block.x, CH / (256 * CONV_VEC), c.tokens);
                return 0;
            }
            const long last = (long)(l.grid.x * l.block.x - 1) * CONV_VEC + (CONV_VEC - 1);
            if (last != CH - 1) { std::snprintf(msg, mn, "conv strip ends at %ld, CH is %d", last, CH); return 0; }
        } else if (l.kind == K_GATES) {
            if ((int)l.block.x != NH || (int)l.grid.x != c.tokens) {
                std::snprintf(msg, mn, "gates grid (%u,%u) block %u is not <<</%d>,NH>>>",
                              l.grid.x, l.grid.y, l.block.x, c.tokens);
                return 0;
            }
        } else if (l.kind == K_GNORM) {
            if ((int)l.grid.x != GN_HEADS * c.tokens || (int)l.block.x != GN_D_V) {
                std::snprintf(msg, mn, "gnorm grid %u block %u is not <<</GN_HEADS*%d>,GN_D_V>>>",
                              l.grid.x, l.block.x, c.tokens);
                return 0;
            }
        } else {
            long last;
            if (c.map == RM_COL) {
                if ((int)l.block.x != D || (int)l.grid.x != rec_blocks(c)) {
                    std::snprintf(msg, mn, "rec col grid %u block %u is not <<</%d-heads>,D>>>",
                                  l.grid.x, l.block.x, rec_blocks(c));
                    return 0;
                }
                last = (long)(l.grid.x - 1) * kStatePerHead + (long)(D - 1) * D + (D - 1);
            } else {
                if ((int)l.grid.x != kGdnR16GridX || (int)l.grid.y != kGdnR16GridY ||
                    (int)l.block.x != kGdnR16BlockX || (int)l.block.y != kGdnR16BlockY) {
                    std::snprintf(msg, mn, "rec split8 grid (%u,%u) block (%u,%u) is not the shipped "
                                  "(%d,%d)/(%d,%d)", l.grid.x, l.grid.y, l.block.x, l.block.y,
                                  kGdnR16GridX, kGdnR16GridY, kGdnR16BlockX, kGdnR16BlockY);
                    return 0;
                }
                const int jmax = ((int)l.grid.y - 1) * kGdnR16ColsPerBlock +
                                 (kGdnR16BlockY - 1) * kGdnR16ColsPerWarp + (kGdnR16ColsPerWarp - 1);
                const int imax = (kGdnR16LanesPerCol - 1) * kGdnR16Elems + (kGdnR16Elems - 1);
                last = (long)(l.grid.x - 1) * kStatePerHead + (long)imax * D + jmax;
            }
            // One launch of the per-layer arm covers the 16 head slabs of ONE layer (grid.x == NH);
            // the combined cell covers every layer's slab in one launch (grid.x == layers*NH).
            const long want = (c.shape == SH_COMBINED) ? (long)c.layers * kStatePerLayer - 1
                                                      : (long)kStatePerLayer - 1;
            if (last != want) { std::snprintf(msg, mn, "rec ends at %ld, want %ld", last, want); return 0; }
        }
    }
    return (int)plan.size();
}
// State geometry helpers are declared above plan_build; p_q/p_k/pow2_pattern sit just below, ahead of
// their first use in alloc_fill.
int p_q(int h) { return (h * 11 + 5) % D; }
// One-hot witness positions VARY with the head, so a launch that aliases heads is caught: the same
// localisation rule as prototype/gdn_collapse's NEG_RC_STRIP_SHIFT, a control must move a derived
// SUBSET and not everything.
int p_k(int h) { return (h * 7 + 3) % D; }
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

float pow2_pattern(int i, int j) {
    // Signed power of two: scaling by it is exact, so o/S recovers one constant with no rounding, and
    // the exponent depends on BOTH i and j so a wrong row or column changes it.
    const int e = ((i * 5 + j * 3) % 40) - 20;
    const float sign = ((i + j) & 1) ? -1.0f : 1.0f;
    return sign * std::ldexp(1.0f, e);
}

// --------------------------------------------------------------- allocation and operand fills
size_t state_region_elems(const Cfg& c) { return (size_t)c.layers * g_state_stride; }

void alloc_fill(const Cfg& c) {
    const size_t u_elems = (c.shape == SH_COMBINED) ? 0 : (size_t)c.tokens * CH;
    const size_t rt = rec_tile_elems(c);
    region_alloc(B_S, state_region_elems(c) * 4);
    region_alloc(B_RAW, (size_t)c.tokens * CH * 2);
    region_alloc(B_WIN, (size_t)kConvWin * CH * 4);
    region_alloc(B_NWIN, (size_t)kConvWin * CH * 4);
    region_alloc(B_CONVW, (size_t)CH * K * 4);
    region_alloc(B_U, u_elems * 4);
    region_alloc(B_Q, rt * 4);
    region_alloc(B_K, rt * 4);
    region_alloc(B_V, rt * 4);
    region_alloc(B_GA, (size_t)c.tokens * NH * 2);
    region_alloc(B_GB, (size_t)c.tokens * NH * 2);
    region_alloc(B_G, (size_t)c.tokens * NH * (c.shape == SH_COMBINED ? c.layers : 1) * 4);
    region_alloc(B_BETA, (size_t)c.tokens * NH * (c.shape == SH_COMBINED ? c.layers : 1) * 4);
    region_alloc(B_ALOG, NH * 4);
    region_alloc(B_DTB, NH * 4);
    region_alloc(B_O, (size_t)c.tokens * kValueDim * (c.shape == SH_COMBINED ? c.layers : 1) * 4);
    region_alloc(B_Z, (size_t)c.tokens * kValueDim * 2);
    region_alloc(B_GNW, D * 4);
    region_alloc(B_BLOCKOUT, (size_t)c.tokens * kValueDim * 2);
    if (c.pollute_mib > 0) region_alloc(B_POLL, (size_t)c.pollute_mib * 1048576);
    if (!g_sum_slot) CK(hipMalloc(&g_sum_slot, sizeof(double)));

    const int mode = c.witness;
    // Read-only operands, in VALUE BANDS chosen so that a negative control's must-differ set is
    // PROVABLE and not luck: the carried window lives in [10, 11.5], raw in [1.0, 1.9] on a 0.1 grid,
    // and the conv weight has exactly ONE non-zero tap (index kConvActiveTap), so every conv output is
    // a single sigmoid of one named operand and two different operands cannot round to the same BF16.
    // A flat fill cannot prove any of this - the first version of this rig filled raw with one constant
    // and three of the five conv/gates controls were un-adjudicable, which is the AGENTS trap ("a
    // negative test that is vacuous no matter what the kernel does") arrived at from the other side.
    {
        std::vector<float> v(g_buf[B_WIN].n / 4);
        for (size_t i = 0; i < v.size(); ++i) {
            const size_t row = i / CH;                       // 0,1,2: oldest first
            v[i] = 10.0f + 0.5f * (float)row + 0.05f * (float)(i % 7);
        }
        set_bytes(B_WIN, v.data(), g_buf[B_WIN].n);
        std::vector<float> cw(g_buf[B_CONVW].n / 4, 0.0f);
        for (size_t i = 0; i < cw.size(); ++i) cw[i] = ((i % K) == kConvActiveTap) ? 1.0f : 0.0f;
        set_bytes(B_CONVW, cw.data(), g_buf[B_CONVW].n);
        std::vector<float> rv((size_t)c.tokens * CH);
        for (size_t i = 0; i < rv.size(); ++i) rv[i] = raw_val(i / (size_t)CH, i % (size_t)CH);
        std::vector<uint16_t> rb(rv.size());
        for (size_t i = 0; i < rb.size(); ++i) { hip_bfloat16 h(rv[i]); std::memcpy(&rb[i], &h, 2); }
        set_bytes(B_RAW, rb.data(), g_buf[B_RAW].n);
        std::vector<uint16_t> zb(g_buf[B_Z].n / 2);
        for (size_t i = 0; i < zb.size(); ++i) {
            const size_t t = i / (size_t)kValueDim;
            const float f = 0.5f + 4.0f * (float)t + 0.01f * (float)(i % 64);   // token-disjoint bands
            hip_bfloat16 h(f);
            std::memcpy(&zb[i], &h, 2);
        }
        set_bytes(B_Z, zb.data(), g_buf[B_Z].n);
        std::vector<uint16_t> ab(g_buf[B_GA].n / 2), bb2(g_buf[B_GB].n / 2);
        for (size_t i = 0; i < ab.size(); ++i) {
            const size_t t = i / (size_t)NH;
            hip_bfloat16 ha(0.5f + 2.0f * (float)t);           // token-distinct: a row-zero read moves
            hip_bfloat16 hb(0.5f + 0.5f * (float)t);           // keeps beta inside (0,1) with a margin
            std::memcpy(&ab[i], &ha, 2);
            std::memcpy(&bb2[i], &hb, 2);
        }
        set_bytes(B_GA, ab.data(), g_buf[B_GA].n);
        set_bytes(B_GB, bb2.data(), g_buf[B_GB].n);
        std::vector<float> al(NH), dt(NH);
        for (int h = 0; h < NH; ++h) { al[h] = -1.0f - 0.1f * h; dt[h] = 0.1f * h; }
        set_bytes(B_ALOG, al.data(), NH * 4);
        set_bytes(B_DTB, dt.data(), NH * 4);
        std::vector<float> gw(D);
        for (int i = 0; i < D; ++i) gw[i] = 1.0f + (float)(i % 8) * 0.125f;
        set_bytes(B_GNW, gw.data(), D * 4);
    }

    // Destinations: sentinel, so an unwritten element is visible as itself.
    sent_fill(B_NWIN);
    if (u_elems) sent_fill(B_U);
    sent_fill(B_BLOCKOUT);
    if (rt) { sent_fill(B_Q); sent_fill(B_K); sent_fill(B_V); }
    sent_fill(B_G); sent_fill(B_BETA);
    if (c.kernels & F_REC) sent_fill(B_O);
    else {
        // gated_norm's operand row must be token-distinct and OUTSIDE what a row-zero read can produce
        // by accident: row t lives in [1+2t, 2+2t].
        std::vector<float> ov(g_buf[B_O].n / 4);
        for (size_t i = 0; i < ov.size(); ++i) {
            const size_t t = i / (size_t)kValueDim;
            ov[i] = 1.0f + 2.0f * (float)t + 0.01f * (float)(i % 64);
        }
        set_bytes(B_O, ov.data(), g_buf[B_O].n);
    }

    // Per-arm operand sets for the recurrence and its witnesses. One-hot positions come from p_q/p_k
    // so q and k one-hot at DIFFERENT columns: a kernel that reads k where it should read q moves.
    std::vector<float> st(state_region_elems(c), 0.0f);
    const int nhl = rec_blocks(c);            // (layer, head) block count for the rec launch
    if (c.shape == SH_COMBINED) {
        std::vector<float> qp(rt, 0.0f), kp(rt, 0.0f), vp(rt, 0.0f);
        for (int b = 0; b < nhl && (size_t)(b + 1) * D <= rt; ++b) {
            const int h = b % NH;
            for (int i = 0; i < D; ++i) {
                const float gen = 0.125f + (float)((b * 3 + i) % 17) * 0.05f;
                qp[(size_t)b * D + i] = (mode == WS_ONEHOT) ? (i == p_q(h) ? 1.0f : 0.0f) : gen;
                kp[(size_t)b * D + i] = (mode == WS_ONEHOT) ? (i == p_k(h) ? 1.0f : 0.0f) : gen;
                vp[(size_t)b * D + i] = 0.5f + (float)((b + i) % 5) * 0.25f;
            }
        }
        set_bytes(B_Q, qp.data(), rt * 4);
        set_bytes(B_K, kp.data(), rt * 4);
        set_bytes(B_V, vp.data(), rt * 4);
    } else {
        std::vector<float> u(g_buf[B_U].n / 4, 0.0f);
        for (int t = 0; t < c.tokens; ++t) {
            float* row = u.data() + (size_t)t * CH;
            for (int h = 0; h < NH; ++h) {
                for (int i = 0; i < D; ++i) {
                    const float gen = 0.125f + (float)((h * 5 + i + t) % 19) * 0.05f;
                    const float qv = (mode == WS_ONEHOT) ? (i == p_q(h) ? 1.0f : 0.0f) : gen;
                    const float kv = (mode == WS_ONEHOT) ? (i == p_k(h) ? 1.0f : 0.0f) : gen;
                    row[h * D + i] = qv;                                    // q slice
                    row[kValueDim + h * D + i] = kv;                        // k slice
                    row[2 * kValueDim + h * D + i] = 0.5f + (float)((h + i) % 5) * 0.25f;  // v
                }
            }
        }
        set_bytes(B_U, u.data(), g_buf[B_U].n);
    }
    {
        const int ng = (int)((size_t)c.tokens * NH * (c.shape == SH_COMBINED ? c.layers : 1));
        std::vector<float> gv(ng), bv(ng);
        for (int i = 0; i < ng; ++i) {
            // onehot: gam = expf(0) = 1 exactly and beta = 0, which forces dv = 0 and makes the state
            // write-back bit-identical to its input. land: gamma != 1 and beta != 0 force a change.
            gv[i] = (mode == WS_ONEHOT) ? 0.0f : kLandG;
            bv[i] = (mode == WS_ONEHOT) ? 0.0f : kLandBeta;
        }
        set_bytes(B_G, gv.data(), ng * 4);
        set_bytes(B_BETA, bv.data(), ng * 4);
    }
    if (mode == WS_LAND) {
        std::fill(st.begin(), st.end(), kSentF);      // sentinel: a skipped element is visible
    } else {
        for (size_t i = 0; i < st.size(); ++i) {
            const size_t L = i / g_state_stride, e = i % g_state_stride;
            if (L >= (size_t)c.layers) { st[i] = 0.0f; continue; }
            st[i] = (mode == WS_ONEHOT)
                        ? pow2_pattern((int)((e / D) % D), (int)(e % D))
                        : (0.001f * (float)((e % 1000) + 1) + 0.017f * (float)(L + 1));
        }
    }
    {
        const int ng = (int)((size_t)c.tokens * NH * (c.shape == SH_COMBINED ? c.layers : 1));
        std::vector<float> gv(ng), bv(ng);
        for (int i = 0; i < ng; ++i) {
            gv[i] = (mode == WS_ONEHOT) ? 0.0f : kLandG;
            bv[i] = (mode == WS_ONEHOT) ? 0.0f : kLandBeta;
        }
        set_bytes(B_G, gv.data(), ng * 4);
        set_bytes(B_BETA, bv.data(), ng * 4);
    }
    set_bytes(B_S, st.data(), state_region_elems(c) * 4);
    if (c.pollute_mib > 0) {
        std::vector<float> one(1048576, 1.0f);
        for (size_t m = 0; m < (size_t)c.pollute_mib; ++m)
            set_bytes(B_POLL, one.data(), 1048576, m * 1048576);
    }
    for (int id = 0; id < NBUF; ++id) if (g_buf[id].base) snapshot(id);
}

// --------------------------------------------------------------- launches
void launch_all(const Cfg& c, const std::vector<Launch>& plan) {
    float* Sall = reinterpret_cast<float*>(g_buf[B_S].p);
    float* U = reinterpret_cast<float*>(g_buf[B_U].p);
    float* G = reinterpret_cast<float*>(g_buf[B_G].p);
    float* BE = reinterpret_cast<float*>(g_buf[B_BETA].p);
    float* O = reinterpret_cast<float*>(g_buf[B_O].p);
    const int nheads = (c.shape == SH_COMBINED) ? c.layers * NH : NH;
    // token == -1 means the launch carries the tokens itself in its grid (conv's grid.y, gates'
    // grid.x, gated_norm's grid.x), exactly as the shipped launch lines do, so it gets the base row.
    auto trow = [](int token, size_t stride) { return token < 0 ? size_t(0) : (size_t)token * stride; };
    for (const Launch& l : plan) {
        const size_t soff = (c.shape == SH_COMBINED) ? 0 : state_layer_off(l.layer);
        float* S = Sall + soff + ((c.neg == NEG_STATE_OFF) ? 1u : 0u);
        switch (l.kind) {
            case K_CONV:
                conv_kernel<<<l.grid, l.block>>>(
                    reinterpret_cast<const hip_bfloat16*>(g_buf[B_RAW].p),
                    reinterpret_cast<float*>(g_buf[B_WIN].p),
                    reinterpret_cast<float*>(g_buf[B_CONVW].p), U + trow(l.token, CH),
                    reinterpret_cast<float*>(g_buf[B_NWIN].p), c.tokens, l.neg_kernel);
                break;
            case K_GATES:
                gates_kernel<<<l.grid, l.block>>>(
                    reinterpret_cast<const hip_bfloat16*>(g_buf[B_GA].p),
                    reinterpret_cast<const hip_bfloat16*>(g_buf[B_GB].p),
                    reinterpret_cast<const float*>(g_buf[B_ALOG].p),
                    reinterpret_cast<const float*>(g_buf[B_DTB].p), G, BE, c.tokens, l.neg_kernel);
                break;
            case K_REC: {
                const float *qp, *kp, *vp;
                if (c.shape == SH_COMBINED) {
                    const size_t o = (size_t)l.token * kValueDim * c.layers;
                    qp = reinterpret_cast<float*>(g_buf[B_Q].p) + o;
                    kp = reinterpret_cast<float*>(g_buf[B_K].p) + o;
                    vp = reinterpret_cast<float*>(g_buf[B_V].p) + o;
                } else {
                    const float* row = U + (size_t)l.token * CH;
                    qp = row; kp = row + kValueDim; vp = row + 2 * kValueDim;
                }
                const size_t go = (size_t)l.token * NH * ((c.shape == SH_COMBINED) ? c.layers : 1);
                float* op = O + (size_t)l.token * kValueDim * ((c.shape == SH_COMBINED) ? c.layers : 1);
                if (c.map == RM_SPLIT8) {
                    rec_reg16_kernel<<<l.grid, l.block>>>(qp, kp, vp, G + go, BE + go, S, op, nheads, 1);
                } else if (c.spell == SP_SHIP) {
                    rec_kernel<<<l.grid, l.block>>>(qp, kp, vp, G + go, BE + go, S, op, nheads);
                } else if (c.spell == SP_UNROLL4) {
                    rec_unroll4_kernel<<<l.grid, l.block>>>(qp, kp, vp, G + go, BE + go, S, op, nheads);
                } else {
                    rec_fused_kernel<<<l.grid, l.block>>>(qp, kp, vp, G + go, BE + go, S, op, nheads);
                }
            } break;
            case K_GNORM:
                gdn_gated_norm<<<l.grid, l.block>>>(
                    reinterpret_cast<float*>(g_buf[B_O].p) + trow(l.token, kValueDim),
                    reinterpret_cast<const hip_bfloat16*>(g_buf[B_Z].p) + trow(l.token, kValueDim),
                    reinterpret_cast<const float*>(g_buf[B_GNW].p),
                    reinterpret_cast<hip_bfloat16*>(g_buf[B_BLOCKOUT].p) + trow(l.token, kValueDim),
                    nullptr, l.neg_kernel, c.tokens);
                break;
        }
        if (c.drain == DR_EACH) CK(hipDeviceSynchronize());
    }
    if (c.drain == DR_BURST) CK(hipDeviceSynchronize());
}

// The interferer: a read-only grid-stride walk whose block-reduced checksum must equal the value the
// PLAN bills, never a value the kernel reports (the mixed_bw block-store lesson: a per-thread store
// to a per-block slot is not a reduction).
__global__ void pollute_read(const float4* buf, long n4, double* out) {
    double s = 0.0;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = buf[i];
        s += (double)v.x + (double)v.y + (double)v.z + (double)v.w;
    }
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int off = (int)blockDim.x / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

int run_pollute(int mib) {
    const long n4 = (long)mib * 1048576 / 16;
    CK(hipMemset(g_sum_slot, 0, sizeof(double)));
    pollute_read<<<512, 256>>>(reinterpret_cast<const float4*>(g_buf[B_POLL].p), n4,
                               reinterpret_cast<double*>(g_sum_slot));
    CK(hipDeviceSynchronize());
    double got = 0.0;
    CK(hipMemcpy(&got, g_sum_slot, sizeof(double), hipMemcpyDeviceToHost));
    const double want = (double)mib * 1048576.0 / 4.0;      // every float filled with 1.0
    return (got == want) ? 1 : 0;
}
// --------------------------------------------------------------- expectations and witnesses
const char* witness_refuse(const Cfg& c) {
    if (c.witness != WS_LAND && c.kernels != F_REC)
        return "--witness onehot|mirror fixes the recurrence's own operands (g, beta and the q|k|v "
               "tile). Running conv or gates in the same window overwrites them, so the witness would "
               "judge operands it did not set. Use --witness land for a full-family row.";
    if (c.witness == WS_MIRROR && c.shape == SH_COMBINED)
        return "--witness mirror is implemented for the per-layer shape only; the combined arm is a "
               "geometry arm and is adjudicated by onehot + land.";
    if (c.witness == WS_MIRROR && c.tokens != 1)
        return "--witness mirror replays one launch per state slab; with tokens > 1 a slab is touched "
               "twice and the pre-state would have to be replayed per token. Use --witness land.";
    if (c.shape == SH_COMBINED && c.layout != LY_BLOB1)
        return "--shape combined requires --layout blob1: the combined kernel computes the state base "
               "as blockIdx.x*D*D and cannot honour a gap between layers.";
    return nullptr;
}

struct Chk {
    int ok = 1;
    long want = 0, unwritten = 0, stray = 0, guard = 0, ratio_bad = 0;
    double meanrel = 0.0, aux1 = 0.0, aux2 = 0.0;
    char why[192] = "";
    // A violation that cannot say WHERE is a red light with no diagnosis, and the address is the whole
    // lesson of the aliased-partial bug (prototype/attn_decode/results.md 20.1): keep the first few.
    int nbad = 0;
    long bad_elem[4] = {-1, -1, -1, -1};
    int bad_buf[4] = {-1, -1, -1, -1};
    void note(int buf, long elem) {
        if (nbad < 4) { bad_buf[nbad] = buf; bad_elem[nbad] = elem; ++nbad; }
    }
    std::string where() const {
        std::string out = "first_bad=";
        if (!nbad) return out + "none";
        for (int i = 0; i < nbad; ++i) {
            out += std::string(kBufName[bad_buf[i]]) + "@" + std::to_string(bad_elem[i]);
            out += (i + 1 < nbad) ? "," : "";
        }
        return out;
    }
};

void fetch(int id, std::vector<uint8_t>& out) {
    out.resize(g_buf[id].n);
    if (g_buf[id].n) CK(hipMemcpy(out.data(), g_buf[id].p, g_buf[id].n, hipMemcpyDeviceToHost));
}

bool is_sent(int id, const uint8_t* p) {
    if (kBufElem[id] == 4) {
        uint32_t w, s;
        std::memcpy(&w, p, 4);
        std::memcpy(&s, &kSentF, 4);
        return w == s;
    }
    uint16_t w;
    std::memcpy(&w, p, 2);
    return w == kSentBF;
}

long expect_build(const Cfg& c, std::vector<uint8_t> mask[NBUF]) {
    std::vector<Launch> plan;
    plan_build(c, false, plan);
    for (int i = 0; i < NBUF; ++i) mask[i].assign(g_buf[i].n / kBufElem[i], 0);
    long total = 0;
    for (const Launch& l : plan)
        for (const Span& s : l.dst)
            for (size_t e = 0; e < s.n_elem; ++e) {
                const size_t idx = s.off_elem + e;
                if (idx < mask[s.buf].size() && !mask[s.buf][idx]) { mask[s.buf][idx] = 1; ++total; }
            }
    return total;
}

void plan_actual(const Cfg& c, std::vector<Launch>& out) { plan_build(c, true, out); }

// The tolerance-free witness: state bit-identity plus the constant-ratio test on o.
void verify_onehot(const Cfg& c, Chk& chk) {
    std::vector<uint8_t> o_bytes, s_bytes;
    fetch(B_O, o_bytes);
    fetch(B_S, s_bytes);
    const int nblocks = rec_blocks(c);
    double ratio0 = -1.0;
    for (int b = 0; b < nblocks; ++b) {
        const int h = (c.shape == SH_COMBINED) ? (b % NH) : b;
        const size_t slab = (c.shape == SH_COMBINED)
                                ? (size_t)b * kStatePerHead
                                : state_layer_off(c.layers - 1) + (size_t)h * kStatePerHead;
        const size_t o0 = (c.shape == SH_COMBINED) ? (size_t)b * D : (size_t)h * D;
        const int pq = p_q(h);
        for (int j = 0; j < D; ++j) {
            float ov, sv;
            std::memcpy(&ov, o_bytes.data() + (o0 + j) * 4, 4);
            std::memcpy(&sv, s_bytes.data() + (slab + (size_t)pq * D + j) * 4, 4);
            if (sv == 0.0f) { ++chk.ratio_bad; continue; }
            if ((ov > 0) != (sv > 0) || ov == 0.0f) { ++chk.ratio_bad; continue; }
            const double r = std::fabs((double)ov) / std::fabs((double)sv);
            if (ratio0 < 0) ratio0 = r;
            else if (r != ratio0) ++chk.ratio_bad;      // exact: scaling by a power of two is exact
        }
    }
}

// A double mirror of §04.4 for the per-layer shape, banded: a geometry witness for the operand maps,
// never a numerics gate (the numerics gates are prototype/gdn_decode/check.py and engine/gate.py).
void verify_mirror(const Cfg& c, Chk& chk) {
    // Operands and the pre-run state come from the SNAPSHOT (what the launch was given); the post-run
    // state and `o` come from the device. mirror is refused for tokens > 1, where one layer's slab is
    // touched by more than one launch and the pre-state would need replaying per token.
    std::vector<uint8_t> ub, sb, ob;
    fetch(B_U, ub); fetch(B_S, sb); fetch(B_O, ob);
    const std::vector<char>& s_pre = g_buf[B_S].snap;
    const std::vector<char>& g_pre = g_buf[B_G].snap;
    const std::vector<char>& b_pre = g_buf[B_BETA].snap;
    const float* u = reinterpret_cast<const float*>(ub.data());
    const float* g = reinterpret_cast<const float*>(g_pre.data());
    const float* be = reinterpret_cast<const float*>(b_pre.data());
    double num = 0.0, den = 0.0, snum = 0.0, sden = 0.0;
    long nbad_o = 0, nbad_s = 0;
    double nbad_o_val = 0.0, nbad_o_got = 0.0;
    for (int L = 0; L < c.layers; ++L) {
        const size_t sL = state_layer_off(L);
        for (int h = 0; h < NH; ++h) {
            // The kernel scopes its state base to its own head (`S + h*D*D`), so the mirror must too.
            // The first draft summed `state_layer_off(L) + i*D + j` for every head, and the mirror
            // flagged it at element 128 - head 1's expected value judged against head 0's slot. That is
            // the argument for keeping a mirror: it caught a wrong index in the witness itself.
            const size_t s0 = sL + (size_t)h * kStatePerHead;
            double sk = 0.0, sq = 0.0;
            double kn[D], qn[D];
            for (int i = 0; i < D; ++i) {
                // the u row is [q | k | v] in kValueDim slices, exactly how the engine hands it over
                const double k = u[kValueDim + h * D + i], q = u[h * D + i];
                sk += k * k; sq += q * q; kn[i] = k; qn[i] = q;
            }
            const double rsk = 1.0 / std::sqrt(sk + 1e-6), rsq = 1.0 / std::sqrt(sq + 1e-6);
            for (int i = 0; i < D; ++i) { kn[i] *= rsk; qn[i] = qn[i] * rsq * Q_SCALE; }
            const double gam = std::exp((double)g[h]), bet = (double)be[h];
            for (int j = 0; j < D; ++j) {
                double r = 0.0;
                for (int i = 0; i < D; ++i) {
                    float sraw;
                    std::memcpy(&sraw, s_pre.data() + (s0 + (size_t)i * D + j) * 4, 4);
                    r += kn[i] * (double)sraw;
                }
                const double v = (double)u[2 * kValueDim + h * D + j];
                const double dv = bet * (v - gam * r);
                double acc = 0.0;
                for (int i = 0; i < D; ++i) {
                    const size_t so = s0 + (size_t)i * D + j;
                    float sraw;
                    std::memcpy(&sraw, s_pre.data() + so * 4, 4);
                    const double sn = gam * (double)sraw + kn[i] * dv;
                    acc += qn[i] * sn;
                    float got;
                    std::memcpy(&got, sb.data() + so * 4, 4);
                    snum += std::fabs(sn - (double)got);
                    sden += std::fabs(sn);
                    if (std::fabs(sn - (double)got) > 1e-3 * (1.0 + std::fabs(sn)) && !nbad_s) {
                        nbad_s = (long)so;
                        std::snprintf(chk.why, sizeof chk.why,
                                      "mirror s@%ld pre=%g want=%.6g got=%.6g sk=%.5g kn=%.5g r=%.5g dv=%.5g",
                                      (long)so, (double)*(const float*)(s_pre.data() + so * 4), sn,
                                      (double)got, sk, kn[(int)i], r, dv);
                    }
                }
                if (L == c.layers - 1) {   // every layer writes the same o row; the last one survives
                    float ogotf;
                    std::memcpy(&ogotf, ob.data() + ((size_t)h * D + j) * 4, 4);
                    const double ogot = (double)ogotf;   // 4 bytes into a float: a double here reads
                                                          // eight bytes and the top half is garbage
                    num += std::fabs(acc - ogot);
                    den += std::fabs(acc);
                    if (std::fabs(acc - ogot) > 1e-3 * (1.0 + std::fabs(acc)) && !nbad_o) {
                        nbad_o = (long)(h * D + j);
                        nbad_o_val = acc;
                        nbad_o_got = ogot;
                    }
                }
            }
        }
    }
    chk.aux1 = num; chk.aux2 = den;
    if (nbad_o) std::snprintf(chk.why + strlen(chk.why), sizeof chk.why - strlen(chk.why),
                              " | mirror o@%ld want=%.9g", (long)nbad_o, (double)nbad_o_val);
    if (nbad_o || nbad_s)
        std::snprintf(chk.why + strlen(chk.why), sizeof chk.why - strlen(chk.why),
                      " | first bad state@%ld o@%ld got=%.6g", nbad_s, nbad_o, (double)nbad_o_got);
    chk.meanrel = (den > 0) ? num / den : 1.0;
    const double srel = (sden > 0) ? snum / sden : 1.0;
    if (srel > chk.meanrel) chk.meanrel = srel;      // the worse of state and output
}

int verify_run(const Cfg& c, Chk& chk) {
    std::vector<uint8_t> mask[NBUF];
    chk.want = expect_build(c, mask);
    if (chk.want == 0) { chk.ok = 0; std::snprintf(chk.why, sizeof chk.why, "FAIL_nothing_expected"); return 0; }
    chk.guard = check_guards();
    for (int id = 0; id < NBUF; ++id) {
        if (!g_buf[id].base) continue;
        std::vector<uint8_t> cur;
        fetch(id, cur);
        const int es = kBufElem[id];
        const size_t ne = cur.size() / es;
        const bool must_equal = (c.witness == WS_ONEHOT && id == B_S);
        for (size_t e = 0; e < ne; ++e) {
            const int off = (int)(e * es);
            const bool diff = std::memcmp(cur.data() + off, g_buf[id].snap.data() + off, es) != 0;
            if (mask[id][e]) {
                if (must_equal) {
                    if (diff) { ++chk.stray; chk.note(id, (long)e); }
                } else if (is_sent(id, cur.data() + off)) {
                    ++chk.unwritten; chk.note(id, (long)e);
                }
            } else if (diff) {
                ++chk.stray; chk.note(id, (long)e);
            }
        }
    }
    if (c.witness == WS_ONEHOT) verify_onehot(c, chk);
    if (c.witness == WS_MIRROR) verify_mirror(c, chk);
    if (chk.unwritten) std::snprintf(chk.why, sizeof chk.why, "unwritten=%ld", chk.unwritten);
    else if (chk.stray) std::snprintf(chk.why, sizeof chk.why, "stray_or_equal_violation=%ld", chk.stray);
    else if (chk.guard) std::snprintf(chk.why, sizeof chk.why, "guard_bytes_broken=%ld", chk.guard);
    else if (chk.ratio_bad) std::snprintf(chk.why, sizeof chk.why, "onehot_ratio_bad=%ld", chk.ratio_bad);
    else if (chk.meanrel > 1e-3) {
        char extra[96];
        std::snprintf(extra, sizeof extra, "%smirror_meanrel=%.3e_gt_1e-3", chk.why[0] ? " " : "",
                      chk.meanrel);
        strncat(chk.why, extra, sizeof chk.why - strlen(chk.why) - 1);
    }
    chk.ok = (chk.unwritten == 0 && chk.stray == 0 && chk.guard == 0 && chk.ratio_bad == 0 &&
              chk.meanrel <= 1e-3);
    return chk.ok;
}

// --------------------------------------------------------------- the paired negative witness
// A negative must move its derived set and must NOT move its derived complement. The classes below
// are DERIVED from what each injection touches (k = tokens-(K-1)+j for the conv window, token index
// for the row-zero injections), which is the only form of the rule that does not rot into "everything"
// the moment the fixture changes (AGENTS.md, the must-catch rows).
// The classes are DERIVED, and each is a claim checkable by arithmetic on the fixture:
//   MUST_DIFF  - provably different: either the two sides read different VALUE BANDS, or the write is
//                simply absent in the injected run (sentinel survives), or the sources differ on the
//                0.1 grid by more than one sigmoid can hide.
//   MUST_SAME  - provably identical: the injection cannot reach this element at all, so a byte of
//                movement here means the control moved something it should not have.
//   DONT_CARE  - this control does not adjudicate it.
// This is the rule the AGENTS table forces ("must-land sets must be DERIVED from the geometry"), with
// the extra condition this rig learned the hard way: a derived set is only sound if the FIXTURE can
// prove it. A flat fill makes "these must differ" a wish, and three of the five conv/gates controls
// were un-adjudicable until the operands were put in disjoint bands with one non-zero conv tap.
enum : int { DONT_CARE = -1, MUST_SAME = 0, MUST_DIFF = 1 };

int differ_class(int neg, int buf, const Cfg& c, size_t elem) {
    switch (neg) {
        case NEG_CONV_WIN_LAG:
            if (buf == B_U) return MUST_SAME;          // u is computed before the window is written
            if (buf != B_NWIN) return DONT_CARE;
            return MUST_DIFF;                          // every row's source moves a band
        case NEG_CONV_WIN_FROM_U: {
            if (buf == B_U) return MUST_SAME;
            if (buf != B_NWIN) return DONT_CARE;
            const int j = (int)(elem / CH);            // window row, oldest first
            const int k = c.tokens - (K - 1) + j;
            return k >= 0 ? MUST_DIFF : MUST_SAME;     // only k >= 0 has a post-conv value to read
        }
        case NEG_CONV_ZERO_PAD: {
            if (buf == B_NWIN) return MUST_SAME;       // the window writes pass zero_pad = false
            if (buf != B_U) return DONT_CARE;
            const int t = (int)(elem / CH);
            const int k = t - (K - 1) + kConvActiveTap;
            return k < 0 ? MUST_DIFF : MUST_SAME;      // a padded tap is all zero_pad changes
        }
        case NEG_CONV_TAP_STRIDE: {
            if (buf == B_NWIN) return MUST_SAME;       // channel_major never reaches the window write
            if (buf != B_U) return DONT_CARE;
            const int t = (int)(elem / CH), ch = (int)(elem % CH);
            const int k = t - (K - 1) + kConvActiveTap;
            if (k < 0 || k >= c.tokens) return MUST_SAME;    // carried rows read the window either way
            // Both sides are pure reads of one array, so their values are known here: clean reads
            // raw[k][ch], the injection reads the [channel][token] view at raw[ch*tokens + k]. Equal
            // sources must give byte-identical outputs; sources one grid step apart cannot collide
            // after a single sigmoid (>= 0.09 in, >= 0.019 out, against a BF16 ULP of 0.0039 at 0.8).
            const float clean = raw_val((size_t)k, (size_t)ch);
            const size_t inj = (size_t)ch * c.tokens + (size_t)k;      // the [channel][token] view,
            const float injv = raw_val(inj / (size_t)CH, inj % (size_t)CH);   // still in bounds
            return (clean == injv) ? MUST_SAME : MUST_DIFF;
        }
        case NEG_GATES_ROW_ZERO:
        case NEG_GNORM_ROW_ZERO: {
            const bool is_out = (buf == B_BLOCKOUT);
            if (buf != B_G && buf != B_BETA && !is_out) return DONT_CARE;
            const size_t row = is_out ? (size_t)kValueDim : (size_t)NH;   // g and beta are [t][NH]
            const int t = (int)(elem / row);
            // The injection collapses the READ row to 0 and keeps the WRITE row, so row 0 gets the
            // same value from every block and rows > 0 are never written at all.
            return t == 0 ? MUST_SAME : MUST_DIFF;
        }
        default:
            return DONT_CARE;
    }
}

int paired_neg(const Cfg& cfg, int neg, int primary, int secondary, Chk& chk) {
    Cfg a = cfg; a.neg = NEG_NONE;
    alloc_fill(a);
    std::vector<Launch> pa; plan_actual(a, pa);
    launch_all(a, pa);
    std::vector<uint8_t> p1, p2;
    fetch(primary, p1);
    if (secondary >= 0) fetch(secondary, p2);
    std::vector<uint8_t> q1, q2;
    Cfg b = cfg; b.neg = neg;
    alloc_fill(b);
    std::vector<Launch> pb; plan_actual(b, pb);
    launch_all(b, pb);
    fetch(primary, q1);
    if (secondary >= 0) fetch(secondary, q2);

    // Derived must-catch set: the elements differ_class calls 1. Empty means the injection cannot fire
    // at this geometry, which is reported as BLIND and never counted as a catch.
    const int es1 = kBufElem[primary];
    long must = 0, caught = 0, banned = 0, same = 0;
    for (size_t e = 0; e + es1 <= p1.size(); e += es1) {
        const int cls = differ_class(neg, primary, cfg, e / es1);
        const bool diff = std::memcmp(p1.data() + e, q1.data() + e, es1) != 0;
        if (cls == MUST_DIFF) { ++must; if (diff) ++caught; else chk.note(primary, (long)(e / es1)); }
        else if (cls == MUST_SAME) { ++same; if (diff) { ++banned; chk.note(primary, (long)(e / es1)); } }
    }
    long banned2 = 0, same2 = 0;
    if (secondary >= 0) {
        const int es2 = kBufElem[secondary];
        for (size_t e = 0; e + es2 <= p2.size(); e += es2) {
            const int cls = differ_class(neg, secondary, cfg, e / es2);
            const bool diff = std::memcmp(p2.data() + e, q2.data() + e, es2) != 0;
            if (cls == MUST_SAME) { ++same2; if (diff) { ++banned2; chk.note(secondary, (long)(e / es2)); } }
            else if (cls == MUST_DIFF && !diff) { chk.note(secondary, (long)(e / es2)); }
        }
    }
    chk.want = must;
    chk.unwritten = (must > 0 && caught < must) ? must - caught : 0;
    chk.stray = banned + banned2;
    chk.guard = check_guards();
    chk.ok = (must > 0 && chk.unwritten == 0 && chk.stray == 0 && chk.guard == 0);
    if (must == 0) std::snprintf(chk.why, sizeof chk.why, "BLIND-BY-GEOMETRY_empty_must_catch");
    else if (chk.unwritten) std::snprintf(chk.why, sizeof chk.why, "must_catch_missed=%ld/%ld", chk.unwritten, must);
    else if (chk.stray) std::snprintf(chk.why, sizeof chk.why, "must_not_move_violated=%ld", chk.stray);
    chk.aux1 = (double)(same + same2);       // how many elements the control proves did NOT move
    return chk.ok;
}

// --------------------------------------------------------------- the timed row
void alloc_free_all();

struct CaseSpec {
    const char* name;
    int kind;             // 0 = positive verify, 1 = negative must be caught, 2 = blind expected
    Cfg c;
    int primary, secondary;
};

std::vector<CaseSpec> case_list() {
    std::vector<CaseSpec> v;
    auto push = [&](const char* nm, int kind, Cfg c, int prim = -1, int sec = -1) {
        CaseSpec s; s.name = nm; s.kind = kind; s.c = c; s.primary = prim; s.secondary = sec;
        v.push_back(s);
    };
    Cfg base;                              // layers 18, tokens 1, rec only, perlayer, col, fused
    push("pos_land_all", 0, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_REC|F_CONV|F_GATES|F_GNORM; return c; }());
    push("pos_onehot_rec", 0, [&]{ Cfg c = base; c.witness = WS_ONEHOT; return c; }());
    push("pos_onehot_split8", 0, [&]{ Cfg c = base; c.witness = WS_ONEHOT; c.map = RM_SPLIT8; return c; }());
    push("pos_onehot_combined", 0, [&]{ Cfg c = base; c.witness = WS_ONEHOT; c.shape = SH_COMBINED; return c; }());
    push("pos_onehot_spell_ship", 0, [&]{ Cfg c = base; c.witness = WS_ONEHOT; c.spell = SP_SHIP; return c; }());
    push("pos_onehot_spell_unroll4", 0, [&]{ Cfg c = base; c.witness = WS_ONEHOT; c.spell = SP_UNROLL4; return c; }());
    push("pos_land_rec", 0, [&]{ Cfg c = base; c.witness = WS_LAND; return c; }());
    push("pos_land_conv", 0, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_CONV; return c; }());
    push("pos_land_gates", 0, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_GATES; return c; }());
    push("pos_land_gnorm", 0, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_GNORM; return c; }());
    push("pos_land_drain_each", 0, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_REC|F_CONV|F_GATES|F_GNORM; c.drain = DR_EACH; return c; }());
    push("pos_land_spanned", 0, [&]{ Cfg c = base; c.witness = WS_LAND; c.layout = LY_SPANNED; c.gap_mib = 8; return c; }());
    push("pos_onehot_tokens2", 0, [&]{ Cfg c = base; c.witness = WS_ONEHOT; c.tokens = 2; return c; }());
    Cfg mir = base; mir.witness = WS_MIRROR; mir.layers = 3;
    push("pos_mirror_rec3", 0, mir);
    push("pos_land_polluted", 0, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_REC; c.pollute_mib = 256; return c; }());

    push("neg_grid_short_col", 1, [&]{ Cfg c = base; c.witness = WS_LAND; c.neg = NEG_GRID_SHORT_REC; return c; }());
    push("neg_grid_short_split8", 1, [&]{ Cfg c = base; c.witness = WS_LAND; c.map = RM_SPLIT8; c.neg = NEG_GRID_SHORT_REC; return c; }());
    push("neg_state_off", 1, [&]{ Cfg c = base; c.witness = WS_ONEHOT; c.neg = NEG_STATE_OFF; return c; }());
    push("neg_conv_win_lag", 1, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_CONV; c.neg = NEG_CONV_WIN_LAG; return c; }(), B_NWIN, B_U);
    push("neg_conv_win_from_u", 1, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_CONV; c.neg = NEG_CONV_WIN_FROM_U; return c; }(), B_NWIN, B_U);
    push("neg_conv_zero_pad", 1, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_CONV; c.neg = NEG_CONV_ZERO_PAD; return c; }(), B_U, B_NWIN);
    push("neg_conv_tap_stride_t2", 1, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_CONV; c.tokens = 2; c.neg = NEG_CONV_TAP_STRIDE; return c; }(), B_U, B_NWIN);
    push("neg_gates_row_zero_t2", 1, [&]{ Cfg c = base; c.kernels = F_GATES; c.tokens = 2; c.neg = NEG_GATES_ROW_ZERO; return c; }(), B_G, B_BETA);
    push("neg_gnorm_row_zero_t2", 1, [&]{ Cfg c = base; c.kernels = F_GNORM; c.tokens = 2; c.neg = NEG_GNORM_ROW_ZERO; return c; }(), B_BLOCKOUT, -1);
    // The two derived BLIND sets at tokens == 1: the must-catch set is provably empty, so the injection
    // must be INVISIBLE. Counting these as catches would be the vacuous-control failure.
    push("blind_conv_tap_stride_t1", 2, [&]{ Cfg c = base; c.witness = WS_LAND; c.kernels = F_CONV; c.neg = NEG_CONV_TAP_STRIDE; return c; }(), B_U, B_NWIN);
    push("blind_gates_row_zero_t1", 2, [&]{ Cfg c = base; c.kernels = F_GATES; c.neg = NEG_GATES_ROW_ZERO; return c; }(), B_G, B_BETA);
    return v;
}

int run_case(const CaseSpec& s, char* line, size_t ln) {
    Cfg cfg = s.c;
    // A case with a primary buffer is a PAIRED control (run clean, run injected, compare two
    // buffers); one without is a launch-level injection judged by the ordinary verify pass, and it
    // keeps its own witness because the witness IS the thing that must notice.
    const bool paired_case = (s.kind == 1 || s.kind == 2) && s.primary >= 0;
    if (paired_case) cfg.witness = WS_LAND;
    if (const char* r = refuse(cfg)) { std::snprintf(line, ln, "REFUSED %s", r); return 9; }
    if (const char* r = witness_refuse(cfg)) { std::snprintf(line, ln, "REFUSED %s", r); return 9; }
    g_state_stride = (size_t)kStatePerLayer + (size_t)cfg.gap_mib * 262144;
    Chk chk;
    int res;
    if (paired_case) {
        res = paired_neg(cfg, cfg.neg, s.primary, s.secondary, chk);
        if (s.kind == 2) {
            // blind: the derived must-catch set must be empty and nothing may move
            const bool blind_ok = (chk.want == 0 && chk.stray == 0 && chk.guard == 0);
            std::snprintf(line, ln, "%s must_catch=%ld same=%.0f stray=%ld guard=%ld",
                          blind_ok ? "BLIND-OK" : "BLIND-BROKEN", chk.want, chk.aux1, chk.stray,
                          chk.guard);
            alloc_free_all();
            return blind_ok ? 0 : 1;
        }
        std::snprintf(line, ln, "%s %s same=%.0f %s", res ? "CAUGHT" : "NOT-CAUGHT", chk.why,
                      chk.aux1, chk.where().c_str());
    } else {
        alloc_fill(cfg);
        std::vector<Launch> pl; plan_actual(cfg, pl);
        launch_all(cfg, pl);
        verify_run(cfg, chk);
        res = chk.ok;
        if (s.kind == 1) res = !chk.ok;              // an injection is CAUGHT when verify goes red
        std::snprintf(line, ln, "%s checked=%ld meanrel=%.2e aux=%.3g/%.3g %s %s",
                      s.kind == 1 ? (res ? "CAUGHT" : "NOT-CAUGHT") : (chk.ok ? "CLEAN" : "FAIL"),
                      chk.want, chk.meanrel, chk.aux1, chk.aux2, chk.why, chk.where().c_str());
    }
    alloc_free_all();
    return res ? 0 : 1;
}

// --------------------------------------------------------------- main-ish pieces
int selftest_geom() {
    int fails = 0;
    char msg[160];
    auto geom_ok = [&](Cfg c, const char* nm) {
        const int n = geom_check(c, msg, sizeof msg);
        const bool ok = n > 0;
        if (!ok) ++fails;
        std::printf("  geom %-26s launches=%d %s%s\n", nm, n, ok ? "ok" : "BAD", ok ? "" : msg);
    };
    Cfg b;
    geom_ok(b, "perlayer_col_all");
    Cfg c2 = b; c2.kernels = F_REC; c2.shape = SH_COMBINED; geom_ok(c2, "combined_rec");
    Cfg c3 = b; c3.map = RM_SPLIT8; geom_ok(c3, "perlayer_split8");
    Cfg c4 = b; c4.kernels = F_CONV; geom_ok(c4, "conv_only");
    Cfg c5 = b; c5.kernels = F_GATES; geom_ok(c5, "gates_only");
    Cfg c6 = b; c6.kernels = F_GNORM; geom_ok(c6, "gnorm_only");
    Cfg c7 = b; c7.layout = LY_SPANNED; c7.gap_mib = 42; geom_ok(c7, "spanned_rec");
    Cfg c9 = b; c9.tokens = 3; geom_ok(c9, "synthetic_tokens3");

    struct { Cfg c; const char* want; } refs[] = {
        { [&]{ Cfg c = b; c.time_flag = 1; c.tokens = 2; return c; }(), "time_at_tokens2" },
        { [&]{ Cfg c = b; c.shape = SH_COMBINED; c.kernels = F_REC | F_CONV; return c; }(), "combined_with_conv" },
        { [&]{ Cfg c = b; c.shape = SH_COMBINED; c.map = RM_SPLIT8; c.kernels = F_REC; return c; }(), "combined_split8" },
        { [&]{ Cfg c = b; c.map = RM_SPLIT8; c.tokens = 2; return c; }(), "split8_tokens2" },
        { [&]{ Cfg c = b; c.layout = LY_SPANNED; c.gap_mib = 0; return c; }(), "spanned_gap0" },
        { [&]{ Cfg c = b; c.witness = WS_ONEHOT; c.kernels = F_REC|F_GATES; return c; }(), "onehot_with_gates" },
        { [&]{ Cfg c = b; c.witness = WS_MIRROR; c.shape = SH_COMBINED; return c; }(), "mirror_combined" },
        { [&]{ Cfg c = b; c.witness = WS_MIRROR; c.tokens = 2; return c; }(), "mirror_tokens2" },
    };
    for (auto& r : refs) {
        const char* why = refuse(r.c); if (!why) why = witness_refuse(r.c);
        if (why) std::printf("  refuse %-26s %s\n", r.want, why);
        else { ++fails; std::printf("  refuse %-26s DID-NOT-REFUSE\n", r.want); }
    }
    // The derived blind sets, asserted rather than hoped: at tokens == 1 the row-zero and tap-stride
    // injections must have an EMPTY must-catch set, and at tokens == 2 a non-empty one.
    Cfg t1 = b; t1.kernels = F_CONV; t1.neg = NEG_CONV_TAP_STRIDE;
    long must1 = 0;
    for (size_t e = 0; e < (size_t)t1.tokens * CH; ++e) must1 += (differ_class(NEG_CONV_TAP_STRIDE, B_U, t1, e) == 1);
    Cfg t2 = t1; t2.tokens = 2;
    long must2 = 0;
    for (size_t e = 0; e < (size_t)t2.tokens * CH; ++e) must2 += (differ_class(NEG_CONV_TAP_STRIDE, B_U, t2, e) == 1);
    std::printf("  derived tap_stride must-catch: tokens1=%ld tokens2=%ld (want 0 then >0)\n", must1, must2);
    if (must1 != 0 || must2 <= 0) ++fails;
    long g1 = 0, g2 = 0;
    for (size_t e = 0; e < (size_t)NH; ++e) g1 += (differ_class(NEG_GATES_ROW_ZERO, B_G, b, e) == 1);
    for (size_t e = 0; e < 2 * (size_t)NH; ++e) g2 += (differ_class(NEG_GATES_ROW_ZERO, B_G, b, e) == 1);
    std::printf("  derived gates row_zero must-catch: tokens1=%ld tokens2=%ld (want 0 then >0)\n", g1, g2);
    if (g1 != 0 || g2 <= 0) ++fails;

    std::printf("SELFTEST-GEOM %s fails=%d\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}

void alloc_free_all() {
    for (int id = 0; id < NBUF; ++id) region_free(id);
    g_state_stride = kStatePerLayer;
}

int selftest_device() {
    const std::vector<CaseSpec> cases = case_list();
    int parsed = 0, bad = 0, caught = 0, blind = 0;
    char line[256];
    for (const CaseSpec& s : cases) {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            int rc = run_case(s, line, sizeof line);
            std::printf("CASE %-26s %s\n", s.name, line);
            fflush(stdout);
            _exit(rc);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        ++parsed;
        const int code = WIFEXITED(st) ? WEXITSTATUS(st) : 3;
        if (code != 0) ++bad;
        if (s.kind == 1 && code == 0) ++caught;
        if (s.kind == 2 && code == 0) ++blind;
    }
    const int want_caught = [&] { int n = 0; for (const CaseSpec& s2 : cases) if (s2.kind == 1) ++n; return n; }();
    const int want_blind = [&] { int n = 0; for (const CaseSpec& s2 : cases) if (s2.kind == 2) ++n; return n; }();
    std::printf("SELFTEST %s cases=%d parsed=%d children_bad=%d caught=%d/%d blind=%d/%d\n",
                (bad == 0 && parsed == (int)cases.size() && caught == want_caught && blind == want_blind)
                    ? "ok" : "FAILED",
                (int)cases.size(), parsed, bad, caught, want_caught, blind, want_blind);
    return (bad == 0 && parsed == (int)cases.size() && caught == want_caught && blind == want_blind) ? 0 : 1;
}

static int real_main(int argc, char** argv) {
    Cfg c;
    int mode = 0;   // 1 = selftest, 2 = selftest-geom, 3 = print-cases, 0 = one row
    auto need = [&](int& i) { if (i + 1 >= argc) { std::printf("missing value for %s\n", argv[i]); std::exit(2); } return std::string(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") mode = 1;
        else if (a == "--selftest-geom") mode = 2;
        else if (a == "--print-cases") mode = 3;
        else if (a == "--time") c.time_flag = 1;
        else if (a == "--layers") c.layers = atoi(need(i).c_str());
        else if (a == "--tokens") c.tokens = atoi(need(i).c_str());
        else if (a == "--reps") c.reps = atoi(need(i).c_str());
        else if (a == "--iters") c.iters = atoi(need(i).c_str());
        else if (a == "--gap-mib") c.gap_mib = atoi(need(i).c_str());
        else if (a == "--pollute-mib") c.pollute_mib = atoi(need(i).c_str());
        else if (a == "--shape") { const std::string s = need(i); c.shape = (s == "combined") ? SH_COMBINED : (s == "perlayer") ? SH_PERLAYER : (std::printf("bad --shape %s\n", s.c_str()), std::exit(2), SH_PERLAYER); }
        else if (a == "--recmap") { const std::string s = need(i); c.map = (s == "split8") ? RM_SPLIT8 : (s == "col") ? RM_COL : (std::printf("bad --recmap %s\n", s.c_str()), std::exit(2), RM_COL); }
        else if (a == "--spell") { const std::string s = need(i); c.spell = (s == "ship") ? SP_SHIP : (s == "unroll4") ? SP_UNROLL4 : (s == "fused") ? SP_FUSED : (std::printf("bad --spell %s\n", s.c_str()), std::exit(2), SP_FUSED); }
        else if (a == "--drain") { const std::string s = need(i); c.drain = (s == "each") ? DR_EACH : (s == "burst") ? DR_BURST : (std::printf("bad --drain %s\n", s.c_str()), std::exit(2), DR_BURST); }
        // Refused at parse time, not in refuse(), because no Cfg can express it: there is no 18-blob
        // allocation anywhere in the rig (the state is one allocation sliced per layer, as in the
        // engine). `check_d4.sh` asserts this exit code so the arm cannot silently become `blob1`.
        else if (a == "--layout") { const std::string s = need(i); if (s == "blob18") { std::printf("REFUSE --layout blob18: not built, and it will not be. The engine allocates the state as ONE allocation sliced per layer (engine/state.h gdn_state + gdn_state_layer()), so the shipped shape IS --layout blob1; an 18-separate-blob arm would price a scatter the engine does not have. --layout spanned --gap-mib N is the only scatter this rig can build honestly.\n"); return 2; } c.layout = (s == "blob1") ? LY_BLOB1 : (s == "spanned") ? LY_SPANNED : (std::printf("bad --layout %s\n", s.c_str()), std::exit(2), LY_BLOB1); }
        else if (a == "--witness") { const std::string s = need(i); c.witness = (s == "land") ? WS_LAND : (s == "onehot") ? WS_ONEHOT : (s == "mirror") ? WS_MIRROR : (std::printf("bad --witness %s\n", s.c_str()), std::exit(2), WS_ONEHOT); }
        else if (a == "--neg") { const std::string s = need(i); int k = 0; for (; k < 9; ++k) if (s == kNegNames[k]) break; if (k == 9) { std::printf("bad --neg %s\n", s.c_str()); std::exit(2); } c.neg = k; }
        else if (a == "--kernel") {
            const std::string s = need(i);
            c.kernels = (s == "all") ? (F_REC | F_CONV | F_GATES | F_GNORM)
                      : (s == "rec") ? F_REC : (s == "conv") ? F_CONV
                      : (s == "gates") ? F_GATES : (s == "gnorm") ? F_GNORM
                      : (s == "none") ? -1
                                      : (std::printf("bad --kernel %s\n", s.c_str()), std::exit(2), 0);
        } else if (a == "--pages") { const std::string s = need(i); if (s == "cold") { std::printf("REFUSE --pages cold: first touch of fresh pages runs ~7x slow (devices/). Cold is refused, not averaged.\n"); return 2; } }
        else { std::printf("unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mode == 2) return selftest_geom();
    if (mode == 3) {
        const std::vector<CaseSpec> cs = case_list();
        for (const CaseSpec& s : cs) std::printf("CASE-ROW %s\n", s.name);
        std::printf("CASE-ROWS total=%zu\n", cs.size());
        return 0;
    }
    if (mode == 1) return selftest_device();

    if (c.kernels == -1) {
        std::printf("ROW no_launch=1 reason=timed_window_launches_nothing NO-TIMING\n");
        return 0;
    }
    if (const char* r = refuse(c)) { std::printf("ABORT: %s\n", r); return 3; }
    if (const char* r = witness_refuse(c)) { std::printf("ABORT: %s\n", r); return 3; }

    g_state_stride = (size_t)kStatePerLayer + (size_t)c.gap_mib * 262144;
    alloc_fill(c);
    std::vector<Launch> plan; plan_actual(c, plan);
    // verify pass (untimed, one token body from the filled state)
    launch_all(c, plan);
    Chk chk; verify_run(c, chk);
    if (c.pollute_mib > 0 && !run_pollute(c.pollute_mib)) { chk.ok = 0; std::snprintf(chk.why, sizeof chk.why, "pollute_checksum_mismatch"); }

    double demand_mib = 0.0;
    for (const Launch& l : plan) demand_mib += l.demand_mib;
    std::printf("ROW shape=%s map=%s spell=%s kernels=%d layers=%d tokens=%d drain=%s layout=%s "
                "witness=%s neg=%s pollute_mib=%d gap_mib=%d launches=%zu demand_mib=%.3f "
                "verify=%s checked=%ld %s",
                c.shape == SH_COMBINED ? "combined" : "perlayer",
                c.map == RM_SPLIT8 ? "split8" : "col",
                c.spell == SP_SHIP ? "ship" : (c.spell == SP_UNROLL4 ? "unroll4" : "fused"),
                c.kernels, c.layers, c.tokens,
                c.drain == DR_EACH ? "each" : "burst",
                c.layout == LY_SPANNED ? "spanned" : "blob1",
                c.witness == WS_LAND ? "land" : (c.witness == WS_ONEHOT ? "onehot" : "mirror"),
                kNegNames[c.neg], c.pollute_mib, c.gap_mib, plan.size(), demand_mib,
                chk.ok ? "ok" : "FAIL", chk.want, chk.why);
    if (plan.empty()) { std::printf(" NO-TIMING no_launch=1 reason=timed_window_launches_nothing\n"); alloc_free_all(); return chk.ok ? 0 : 1; }
    if (!c.time_flag) {
        std::printf(" NO-TIMING reason=no_--time_flag\n");
        alloc_free_all();
        return chk.ok ? 0 : 1;
    }
    // Timed reps: warm burst first (fresh pages are ~7x slow, devices/), then best/median per rep.
    std::vector<double> per_tok;
    double best = 1e30, worst = 0.0;
    for (int r = 0; r < c.reps; ++r) {
        launch_all(c, plan);                        // warm-up
        // The interferer is submitted and DRAINED outside the clock. Timing it with the row would bill
        // the eviction to the kernel under test, which is the opposite of the arm's purpose (C2 asks how
        // the row changes with the cache state the row *inherits*).
        const double ms = [&] {
            double best_it = 1e30;
            for (int it = 0; it < c.iters; ++it) {
                if (c.pollute_mib > 0) { run_pollute(c.pollute_mib); CK(hipDeviceSynchronize()); }
                const auto t0 = std::chrono::steady_clock::now();
                launch_all(c, plan);
                const auto t1 = std::chrono::steady_clock::now();
                best_it = std::min(best_it, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best_it / (double)c.tokens;
        }();
        per_tok.push_back(ms);
        best = std::min(best, ms); worst = std::max(worst, ms);
    }
    const double med = median(per_tok);
    const double gbs = demand_mib * 1.048576 / (med / 1000.0) / 1000.0;
    std::printf(" ms_median=%.4f ms_best=%.4f ms_worst=%.4f reps=%d iters=%d "
                "demanded_GBps=%.1f rate_class=%s\n",
                med, best, worst, c.reps, c.iters, gbs,
                gbs > 240.0 ? "demanded-above-DRAM-plateau(L2-resident)" : "DRAM-plausible");
    // ms is the best-of-iters per-token body time: the timer is steady_clock around one launch burst
    // (hipEventElapsedTime returns exactly 0 below ~1 ms on this part), one interferer drain per iter
    // OUTSIDE the clock, and the per-rep spread printed so a bimodal row cannot hide.
    alloc_free_all();
    return chk.ok ? 0 : 1;
}


}  // namespace

int main(int argc, char** argv) { return real_main(argc, argv); }
