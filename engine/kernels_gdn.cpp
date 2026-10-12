// Launchers for the GDN family: prototype/gdn_decode (§04.2–§04.4) and prototype/gdn_gated_norm
// (§04.5). Separate TU so their macros (D, NH, CH, K, GN_D_V, …) cannot collide with another op's.
#include "kernels.h"

#include "engine/launch_stats.h"
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
                                                 tokens, GC_NONE);
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
             float* S, float* o) {
    // One layer per launch: the model interleaves GDN and attention layers (§03.5), so a token
    // never has two adjacent GDN layers to batch. `all` mode (18 layers, one launch) in the
    // prototype is a bandwidth-measurement shape, not a token shape.
    rec_kernel<<<NH, D>>>(q, k, v, g, beta, S, o, NH);
    // Not counted: this launcher has no token width (see `engine/launch_stats.h`), and a family
    // that can only ever report rows=1 would give section (h) a check that cannot fail.
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
