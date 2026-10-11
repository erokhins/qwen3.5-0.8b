#pragma once
// One launch function per graduated kernel, and nothing else.
//
// The kernels themselves live in `prototype/<op>/<op>_kernels.h`, moved verbatim out of the
// binaries each op's own gate signed off against. This header exists so `decode_step` can call
// them without seeing a single macro or device helper from those files: each launcher is its own
// translation unit (`kernels_*.cpp`), because the ops each spell their own `bf`/`fb`/`rnd` and
// define colliding macros (`D`, `H`, `HID`, `EPS`, `SCALE`, `FULL_MASK`) — one TU per family
// keeps that harmless, and merging them is a compile error rather than a cleanup.
//
// Every launcher takes raw device pointers, launches with the configuration its op's gate used,
// and returns. Nothing here syncs: a decode step is submitted as a stream of launches and the
// host waits once, at the logits (§07.1 / W5's "no host round-trip per op").
//
// The `neg` parameter each op carries for its own negative tests is always the clean value here.
// The engine is not a place to inject bugs; the injections stay in the op's gate.

#include <cstdint>

namespace engine::kernels {

// ---- §03.1 + §03.2: residual add fused with the zero-centred RMSNorm (prototype/glue).
// `delta == nullptr` is the norm-only form (§03.2's first call of the stack, and the final norm);
// `s_out == nullptr` skips writing the new residual. 256 threads is the width the op's gate ran at
// (H/THREADS = 4 elements/thread, an 8-partial warp tree). The block width changes the F32 summation
// order of the mean of squares, and the gate has since been re-run at 1024 with an identical legal
// error profile — so 256 is this launcher's measured preference, not a numerical requirement
// (kernels_glue.cpp cites both measurements; engine/results.md E1 is closed).
void glue_add_norm(const uint16_t* s, const uint16_t* delta, const uint16_t* w,
                   uint16_t* s_out, uint16_t* n_out);

// ---- §04.1 / §05.1: the projection GEMV family (prototype/proj_gemv).
// `in` must be 1024 or 2048 — the only two widths the checkpoint has for these nine shapes, and
// the only two the kernel is instantiated for. `x` is F32: the launcher instantiates PACK_X=false,
// the gated form and the measured-faster one (kernels_proj.cpp cites both numbers; engine/results.md
// E2 closed the A/B in a clean window). Hand it a BF16 buffer and it will read BF16 memory
// as floats, so decode_step must cast where its producer left BF16.
void proj_gemv(const void* weight, const float* x, uint16_t* out, int rows, int in);

// ---- §03.3: gate+up+SwiGLU, then down (prototype/mlp_gemv). Both take the input widened to
// F32; there is no BF16-staged form of these two kernels, so the caller casts (see the launch
// count in results.md — this is 48 casts/token, a named follow-up).
void mlp_gateup(const uint16_t* w_gate, const uint16_t* w_up, const float* x, uint16_t* act,
                int rows, int in);
void mlp_down(const uint16_t* w_down, const float* act, uint16_t* out, int rows, int in);

// ---- §04.2/§04.3/§04.4: conv, gates, recurrence (prototype/gdn_decode). All F32 on the state
// side; `raw` is F32 because the conv kernel reads it that way, which is why decode_step casts the
// BF16 `in_proj_qkv` output once per layer.
void gdn_conv(const float* raw, const float* win, const float* conv_weight, float* u,
              float* new_win);
void gdn_gates(const float* a_raw, const float* b_raw, const float* a_log, const float* dt_bias,
               float* g, float* beta);
// One GDN layer: 16 blocks, thread j owns column j of S. `S` is the layer's [16][128][128] block.
void gdn_rec(const float* q, const float* k, const float* v, const float* g, const float* beta,
             float* S, float* o);

// ---- §04.5: the gated RMSNorm (prototype/gdn_gated_norm). `o` is the recurrence's F32 output
// (the kernel rounds it to the activation dtype itself); `z` is in_proj_z's BF16 rows; `w_norm`
// is F32 in the checkpoint (§01.6) and joins as BF16, which is what the pins carry. `norm_out`
// may be null; decode_step passes the stage pointer through when dumping.
void gdn_gated_norm(const float* o, const uint16_t* z, const float* w_norm, uint16_t* out,
                    uint16_t* norm_out);

// ---- §05.1-§05.7: the whole attention block between the projections and o_proj
// (prototype/attn_decode, fused launch plan, one-pass softmax — the form §07.3 now prescribes and
// the only one that reproduces `attn_out_pre_gate` bit-for-bit). One launch, 8 blocks.
// `lvalid` is the number of cache lanes this query may read (prefix_len + 1 during decode).
struct AttnLayerArgs {
    const uint16_t *q_proj_out, *k_proj_out, *v_proj_out, *q_norm, *k_norm;
    uint16_t *k_cache, *v_cache;            // [2][cap][256], appended at `pos` in place
    int cap, pos, lvalid;
    uint16_t *attn_out_pre_gate, *gated;    // [8][256] each; `gated` is o_proj's operand
    uint16_t *q_rope, *k_rope, *gate_out;   // stage dumps (may be null)
};
void attn_decode_layer(const AttnLayerArgs& a);

// ---- §03.4: the tied logits GEMV (prototype/lm_head_gemv), full-logit form. The fused-argmax
// form exists but resolves ties to the highest id, and §03.4 fixes ties to the lowest, so the
// engine computes the logits and takes the argmax host-side (results.md, "the last 2 KB").
void lm_head_full(const uint16_t* table, const float* h, float* logits, int vocab);

// ---- dtype adapters, engine's own (no model math here at all). These exist only because some
// graduated kernels take F32 activations (gdn_decode's conv/gates/rec, mlp_gemv, lm_head_gemv)
// while the activations the other kernels produce are BF16.
void cast_bf16_to_f32(const uint16_t* src, float* dst, int n);
void cast_f32_to_bf16(const float* src, uint16_t* dst, int n);
// Three regions in ONE launch — this is what a per-layer GDN cast costs instead of three launches.
// `kernels_cast.cpp` defines it; a region may be empty (n = 0) but a non-empty region with a null
// pointer is a throw, not a wild write.
void cast_bf16_to_f32_3(const uint16_t* s0, float* d0, int n0, const uint16_t* s1, float* d1,
                        int n1, const uint16_t* s2, float* d2, int n2);

}  // namespace engine::kernels
