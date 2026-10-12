// Launcher for prototype/attn_decode: one whole attention layer per launch (two, above the lane
// staging), in the FUSED launch plan. Its own TU — attn_decode defines N_Q, N_KV, DH, ROT,
// THREADS, TILE, MAXSC, SCALE, EPS, FULL_MASK and its own bf/fb/rbf/block_sum, every one of which
// collides with another op's spelling if these files are merged.
//
// Three configuration choices, each made by a gate rather than here:
//
//   * `fused=1` — the core recomputes its own head norm + rope and appends to the cache inline,
//     instead of `attn_prep` staging them. The op's gate runs `staged` as its headline plan and
//     requires `fused` to agree with it BIT FOR BIT (`run.py`'s cross-check fails the gate if the
//     two launch plans differ at all), so the fused form is gated against the same pins, and it is
//     the one-launch form W5 wants at these cache lengths.
//   * `full`/`split` are chosen per row by necessity and nothing else — see the lane geometry below.
//     One-pass (`full=1, split=1`) wherever a row fits the staging, because it is bit-exact with the
//     pins and it is the route every priced fixture rides: §07.3's decision, and the reason
//     `full=0` is only ever reached by being forced. The blocked `(max,sum)` form cannot reproduce the
//     pins at any tile size — §05.5 rounds the *normalised* weight and a streaming row cannot round a
//     weight it has not divided yet — measured on the dec8 pins: one-pass 0 of 98 304 elements
//     differ, blocked 25 745. Its own band is §07.3's pre-authorized one, gated in
//     `prototype/attn_decode/run.py` (results.md §10), and it is the form that can be split.
//   * `score_f32=0` — the score is rounded through BF16 before the scale, which is the form the pins
//     carry (see G6's history in `bench/gate-audit.md`: the band that hid this was an audit finding).
#include "kernels.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "prototype/attn_decode/attn_decode_kernels.h"

namespace engine::kernels {
namespace {
// ---- the lane geometry (P7, §07.3) -------------------------------------------------------------
// The op stages one block's scores in shared memory: `MAXSC` lanes, a #define the kernel owns
// (`prototype/attn_decode/attn_decode_kernels.h:32`). The ONE-PASS form cannot cut a row across
// blocks — it normalises the weight matrix inside one block (§05.5), which is why the prototype
// refuses `--softmax full --split > 1` outright — so above the staging the engine switches to the
// streaming form and splits the lane axis into `split` blocks whose `(max,sum,Σ w·v)` partials
// `attn_combine` merges with the same rescaling (A-17: that state belongs to one query row and dies
// inside this launch, which is what makes it reduction state and not carry state).
//
// `kAttnMaxSplit` is how many blocks one query row may be cut into, and it is the second half of the
// engine's lane ceiling: `MAXSC × kAttnMaxSplit`. It is what makes that ceiling a bounded,
// allocatable number rather than "as much shared memory as the row happens to need", and going past
// it is a refusal, never an overrun: `attn_decode_layer` throws rather than write past the staging or
// past the `part` scratch. 8 is the number that makes the ceiling the context the engine already
// hands out by default (`decode_run`'s `kSessionDefaultCap` = 4096 slots); it is not a limit the
// arithmetic has, and a longer request gets this comment, not a wrong token.
//
// `engine/gate.py::lane_ceiling()` READS both halves of that product out of the files that enforce
// them, so neither constant can be retyped into the gate.
constexpr int kAttnMaxSplit = 8;

// The split a row of `lanes` needs: `split = ceil(lanes / staging)`, so no block is handed more lanes
// than the staging width the split was derived from (`ceil(n / ceil(n/s)) <= s`).
int split_for(int lanes) { return (lanes + MAXSC - 1) / MAXSC; }

int warp_width() {
    static int w = [] {
        int v = 32;
        (void)hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, 0);
        return v > 0 ? v : 32;
    }();
    return w;
}

// The engine's public boundary is `uint16_t*` for BF16 buffers (kernels.h keeps hip_bfloat16 out of
// the host API); CoreArgs is the op's own struct and wants `hip_bfloat16*`. Same 2 bytes, and one
// named cast here so no individual field assignment looks like a type error being papered over. The
// cast drops const deliberately and once: `CoreArgs` — verbatim from the pinned prototype — spells its
// read-only members (`qn`, `kn`, the three projections) non-const, and editing a kernel this box has
// no device to re-gate is the worse price for const-correctness here.
hip_bfloat16* bf16(const uint16_t* p) {
    return reinterpret_cast<hip_bfloat16*>(const_cast<uint16_t*>(p));
}
}  // namespace

std::size_t attn_part_floats() {
    // `attn_core`'s own partial layout: one `(max, sum, acc[DH])` tuple per (query head, split
    // block), F32 — `a.part + (head*split + blockIdx.y)*(DH + 2)` in the kernel. Sized for
    // `kAttnMaxSplit`, which is what makes the lane ceiling allocatable instead of hopeful.
    return static_cast<size_t>(N_Q) * kAttnMaxSplit * (DH + 2);
}

void attn_decode_layer(const AttnLayerArgs& a) {
    if (a.q_proj_out == nullptr || a.k_proj_out == nullptr || a.v_proj_out == nullptr ||
        a.q_norm == nullptr || a.k_norm == nullptr || a.k_cache == nullptr ||
        a.v_cache == nullptr || a.attn_out_pre_gate == nullptr || a.gated == nullptr ||
        a.q_rope == nullptr || a.k_rope == nullptr || a.gate_out == nullptr) {
        throw std::runtime_error(
            "attn_decode_layer: a required buffer is null. The three stage buffers are NOT "
            "optional in the fused plan: attn_decode_kernels.h's stage_head() returns early when "
            "q_rope is null, and the §05.7 cache append lives behind that same test, so nulling "
            "them to 'skip the dump' silently skips writing K and V into the cache.");
    }
    if (a.lvalid < 0 || a.pos < 0 || a.pos >= a.cap || a.lvalid > a.cap) {
        throw std::runtime_error("attn_decode_layer: pos=" + std::to_string(a.pos) +
                                 " lvalid=" + std::to_string(a.lvalid) +
                                 " does not fit cap=" + std::to_string(a.cap));
    }

    // ---- the form, chosen by necessity (§07.3) ------------------------------------------------
    // One-pass below the staging and split-K streaming above it, never the other way round. The
    // one-pass form is bit-exact with the pins, so a row that fits it must use it: that is what keeps
    // every priced fixture's route — and therefore every shipped margin — untouched by this launcher.
    // Above the staging there is no choice to make, only a way to be wrong: `full=1` with a row
    // longer than `MAXSC` would have `attn_core` write past `sc[]`, so the split is forced and
    // `part` becomes a required buffer rather than an optional one.
    const bool split_lane = a.lvalid > MAXSC;
    const int split = split_lane ? split_for(a.lvalid) : 1;
    if (split_lane) {
        if (split > kAttnMaxSplit) {
            throw std::runtime_error("attn_decode_layer: lvalid=" + std::to_string(a.lvalid) +
                                     " needs split=" + std::to_string(split) +
                                     ", more blocks than the engine's kAttnMaxSplit=" +
                                     std::to_string(kAttnMaxSplit) + " (lane ceiling " +
                                     std::to_string(MAXSC * kAttnMaxSplit) + " = staging MAXSC " +
                                     std::to_string(MAXSC) + " x split). Raising it means raising the"
                                     " `part` scratch too — `attn_part_floats()` sizes it, and"
                                     " `engine/gate.py::lane_ceiling()` reads both constants rather"
                                     " than trusting a number");
        }
        if (a.part == nullptr) {
            throw std::runtime_error("attn_decode_layer: lvalid=" + std::to_string(a.lvalid) +
                                     " exceeds the one-pass staging (MAXSC=" +
                                     std::to_string(MAXSC) + ") and needs the split-K online form"
                                     " (§07.3), which has nowhere to put its partials: `part` is"
                                     " null. Allocate it — `attn_part_floats()` says how big.");
        }
    }

    CoreArgs c{};
    c.qg = bf16(a.q_proj_out);
    c.kg = bf16(a.k_proj_out);
    c.vg = bf16(a.v_proj_out);
    c.kcache = bf16(a.k_cache);
    c.vcache = bf16(a.v_cache);
    c.qn = bf16(a.q_norm);
    c.kn = bf16(a.k_norm);
    c.aout = bf16(a.attn_out_pre_gate);
    c.gout = bf16(a.gated);
    // The §05.7 append writes the cache IN PLACE (§08.5's head-major layout is exactly what
    // attn_core indexes), so kc_as/vc_as are the cache itself, as in the gated host.
    c.kc_as = c.kcache;
    c.vc_as = c.vcache;
    c.qrope_out = bf16(a.q_rope);
    c.krope_out = bf16(a.k_rope);
    c.gate_out = bf16(a.gate_out);
    // rope_out is the cos||sin table the gate dumps to prove the in-kernel transcendentals; the
    // engine has no use for it. `stale` is NEG_CARRY plumbing (an injected bug this launcher never
    // injects); `part` is the split-K partial staging, null whenever one block covers the row.
    c.rope_out = nullptr;
    c.part = split_lane ? a.part : nullptr;
    c.stale = nullptr;
    c.cap = a.cap;
    c.pos = a.pos;
    c.lvalid = a.lvalid;
    c.split = split;
    c.neg = NEG_NONE;
    c.full = split_lane ? 0 : 1;
    c.score_f32 = 0;
    c.fused = 1;
    // One cache lane per warp, capped by the block's warp count — `attn_decode.hip.cpp:226`
    // (`tile = std::min(TILE, nw)`), where nw is THREADS/warpSize. On this wave32 part that is 8.
    c.tile = std::min(TILE, THREADS / warp_width());

    // 8 query-head blocks, `split` deep in y. At split 1 that is the single lane of y the launcher
    // shipped before P7; above it `stage_head`'s `blockIdx.y != 0` early-out is what makes the extra
    // blocks safe to run concurrently — exactly one of them writes the stage buffers and performs the
    // §05.7 append, and none of them reads the appended lane back out of the cache (the core takes
    // lane `pos` from shared memory, which every block computed for itself), so there is no
    // cross-block dependency to order.
    attn_core<<<dim3(N_Q, split), THREADS>>>(c);
    if (split > 1) {
        // The merge: `(max,sum,acc)` per block → one row, then §05.6's gate. Same algebra as the
        // tile loop, same lifetime (A-17: it never leaves this launch). This is the one extra launch
        // the long-context form costs per attention layer.
        attn_combine<<<N_Q, THREADS>>>(c);
    }
}

}  // namespace engine::kernels
