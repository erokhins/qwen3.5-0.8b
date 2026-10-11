// Launchers for the GDN family: prototype/gdn_decode (§04.2–§04.4) and prototype/gdn_gated_norm
// (§04.5). Separate TU so their macros (D, NH, CH, K, GN_D_V, …) cannot collide with another op's.
#include "kernels.h"

#include "prototype/gdn_decode/gdn_decode_kernels.h"
#include "prototype/gdn_gated_norm/gdn_gated_norm_kernels.h"

namespace engine::kernels {

void gdn_conv(const float* raw, const float* win, const float* conv_weight, float* u,
              float* new_win) {
    conv_kernel<<<CH / 256, 256>>>((const float*)raw, win, conv_weight, u, new_win);
}

void gdn_gates(const float* a_raw, const float* b_raw, const float* a_log, const float* dt_bias,
               float* g, float* beta) {
    gates_kernel<<<1, NH>>>(a_raw, b_raw, a_log, dt_bias, g, beta);
}

void gdn_rec(const float* q, const float* k, const float* v, const float* g, const float* beta,
             float* S, float* o) {
    // One layer per launch: the model interleaves GDN and attention layers (§03.5), so a token
    // never has two adjacent GDN layers to batch. `all` mode (18 layers, one launch) in the
    // prototype is a bandwidth-measurement shape, not a token shape.
    rec_kernel<<<NH, D>>>(q, k, v, g, beta, S, o, NH);
}

void gdn_gated_norm(const float* o, const uint16_t* z, const float* w_norm, uint16_t* out,
                    uint16_t* norm_out) {
    ::gdn_gated_norm<<<GN_HEADS, GN_D_V>>>(o, (const hip_bfloat16*)z, w_norm,
                                          (hip_bfloat16*)out, (hip_bfloat16*)norm_out, GN_NONE);
}

}  // namespace engine::kernels
