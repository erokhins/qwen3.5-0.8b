/*
 * prototype/mtp_draft/td_launchers.h — the one declaration set the host TU sees. The four shipped
 * kernel families are compiled in their own translation units (td_glue_tu.cpp, td_proj_tu.cpp,
 * td_mlp_tu.cpp, td_attn_tu.cpp) because each op header spells its own helpers and defines colliding
 * macros (D, H, HID, EPS, SCALE, FULL_MASK); that is stated in every op header and it is a compile error
 * if they are merged. One TU per family is also what `engine/` does, so the rig's build shape matches the
 * thing it is a rehearsal for.
 *
 * Every launcher increments `td_launches`. The counter lives here rather than at the call sites because
 * `engine/launch_stats.h` gives its reason: "counted where the launch is, not at the call site, so no
 * caller can decline to report itself". D2's "~18 launches" is replaced by this counter (DRAFT-HANDOVER.md
 * §5), and the price sheet is built from the counted number.
 */
#pragma once

#include <cstdint>

extern long td_launches;      // defined in mtp_draft.hip.cpp
extern int td_rpb_proj;       // proj rows per block, overridable so the timing pass can sweep it

namespace td {

// ── prototype/glue/glue_kernels.h ──────────────────────────────────────────────────────────────
// Norm only (delta == nullptr is the shipped norm-only call; engine/kernels_glue.cpp's
// glue_add_norm() passes exactly that for layer 0's input norm). `n_out` is the normalised row.
void norm_only(const uint16_t* s, const uint16_t* w, uint16_t* n_out);
// The §03.2 residual add fused with the NEXT norm (the engine's pairing: post-attention norm carries
// the mixer's add, and the norm after the MLP carries the MLP's).
void add_norm(const uint16_t* s, const uint16_t* delta, const uint16_t* w,
              uint16_t* s_out, uint16_t* n_out);

// ── prototype/proj_gemv/proj_gemv_kernels.h ───────────────────────────────────────────────────
// One output row per wavefront, tokens == 1 (the engine's decode shape). `pack_x` is the staging
// choice: true stages the operand row BF16 (the `proj_gemv_xbf16` route, and what this rig uses so the
// chain carries no cast launch), false stages it F32 (the engine's shipped default). `gemv_f32` exists
// so `--mode stagecheck` can require the two to be bit-identical on this block's operands.
// IN is the input width: 1024 for q/k/v_proj, 2048 for fc and o_proj.
void gemv(const uint16_t* W, const uint16_t* x, uint16_t* out, int rows, int in,
          bool pack_x = true);
void gemv_f32(const uint16_t* W, const float* x, uint16_t* out, int rows, int in);

// ── prototype/mlp_gemv + engine/mlp_xbf16_kernels.h ───────────────────────────────────────────
// gate/up + SwiGLU fused (act never reaches memory), then down. The MTP MLP's shapes are the
// target MLP's shapes exactly (3584x1024, 1024x3584), which is why nothing new is needed. `in` is
// the input width, so the pair is gateup(in=1024) then down(in=3584).
void gateup(const uint16_t* wg, const uint16_t* wu, const uint16_t* x, uint16_t* act, int in);
void gateup_f32(const uint16_t* wg, const uint16_t* wu, const float* x, uint16_t* act, int in);
void down(const uint16_t* wd, const uint16_t* act, uint16_t* out, int in);
void down_f32(const uint16_t* wd, const float* act, uint16_t* out, int in);

// ── the MTP attention row, and the shipped row it must agree with ──────────────────────────────
struct AttnRow {
    const uint16_t* qg;      // [4096] this step's q_proj row: [8 heads][q 256 | gate 256]
    const uint16_t* kg;      // [512]
    const uint16_t* vg;      // [512]
    const uint16_t* qn;      // [256] q_norm weight
    const uint16_t* kn;      // [256] k_norm weight
    uint16_t* kcache;        // [N_KV * cap * 256]
    uint16_t* vcache;
    uint16_t* aout;          // [2048] pre-gate attention out
    uint16_t* gout;          // [2048] §05.6 gated out, i.e. o_proj's operand
    uint16_t* qrope_out;     // staged dumps (the §05.2/§05.3 witnesses), may be null
    uint16_t* krope_out;
    uint16_t* gate_out;
    int cap;                 // lane stride of the cache
    int nlanes;              // lanes valid in the cache AFTER this step's append
    int self_lane;           // the dense slot this step's k/v go to
    int pos;                 // ABSOLUTE position for rope (decoupled from self_lane, see handover §3)
};
void mtp_attn_row(const AttnRow& a);
// The shipped `attn_core`, one row, fused plan with the staged append on, for the cross-kernel
// bit-identity witness (valid only where position == dense lane index, i.e. contiguous from 0).
void shipped_attn_row(const AttnRow& a);

}  // namespace td
