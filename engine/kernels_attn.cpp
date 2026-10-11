// Launcher for prototype/attn_decode: one whole attention layer per launch, in the FUSED plan and
// the ONE-PASS softmax form (§07.3's decision). Its own TU — attn_decode defines N_Q, N_KV, DH, ROT,
// THREADS, TILE, MAXSC, SCALE, EPS, FULL_MASK and its own bf/fb/rbf/block_sum, every one of which
// collides with another op's spelling if these files are merged.
//
// Two configuration choices, both made by a gate rather than here:
//
//   * `fused=1` — the core recomputes its own head norm + rope and appends to the cache inline,
//     instead of `attn_prep` staging them. The op's gate runs `staged` as its headline plan and
//     requires `fused` to agree with it BIT FOR BIT (`run.py`'s cross-check fails the gate if the
//     two launch plans differ at all), so the fused form is gated against the same pins, and it is
//     the one-launch form W5 wants at these cache lengths.
//   * `full=1` (one `softmax` over the row, weights rounded to BF16 before P·V) rather than the
//     blocked `(max,sum)` form. §07.3: the two do NOT agree bit-for-bit at any tile size, because
//     §05.5 rounds the *normalised* weight and a streaming row cannot round a weight it has not
//     divided yet. Measured on the dec8 pins: one-pass 0 of 98 304 elements differ, blocked 25 745.
//     The one-pass form is also the only one that keeps `split` at 1 — it normalises the weight
//     matrix inside one block, so it cannot be split across blocks (the prototype refuses that
//     combination outright).
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
    // The shared score staging is MAXSC lanes per block and `split` stays 1, so lvalid cannot
    // exceed it — the prototype's own loader refuses the case rather than reading past `sc[]`.
    if (a.lvalid < 0 || a.pos < 0 || a.pos >= a.cap || a.lvalid > a.cap) {
        throw std::runtime_error("attn_decode_layer: pos=" + std::to_string(a.pos) +
                                 " lvalid=" + std::to_string(a.lvalid) +
                                 " does not fit cap=" + std::to_string(a.cap));
    }
    if (a.lvalid > MAXSC) {
        throw std::runtime_error("attn_decode_layer: lvalid=" + std::to_string(a.lvalid) +
                                 " exceeds MAXSC=" + std::to_string(MAXSC) +
                                 "; the one-pass softmax cannot be split across blocks (§07.3), so"
                                 " a longer context needs the split-K / flash-decoding path, not"
                                 " a wider stage");
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
    // engine has no use for it. part/stale are split-K and NEG_CARRY plumbing, unused at split 1.
    c.rope_out = nullptr;
    c.part = nullptr;
    c.stale = nullptr;
    c.cap = a.cap;
    c.pos = a.pos;
    c.lvalid = a.lvalid;
    c.split = 1;
    c.neg = NEG_NONE;
    c.full = 1;
    c.score_f32 = 0;
    c.fused = 1;
    // One cache lane per warp, capped by the block's warp count — `attn_decode.hip.cpp:226`
    // (`tile = std::min(TILE, nw)`), where nw is THREADS/warpSize. On this wave32 part that is 8.
    c.tile = std::min(TILE, THREADS / warp_width());

    // 8 query-head blocks; gridDim.y = split = 1 (and stage_head's `blockIdx.y != 0` early-out
    // depends on that being the only lane of y).
    attn_core<<<dim3(N_Q, 1), THREADS>>>(c);
}

}  // namespace engine::kernels
