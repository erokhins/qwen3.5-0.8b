/*
 * prototype/mtp_draft/td_glue_tu.cpp — the §03.1/§03.2 family for the MTP block, compiled from the
 * shipped kernel source. The block has five norms (pre_fc_norm_embedding, pre_fc_norm_hidden,
 * input_layernorm, post_attention_layernorm, mtp.norm), all over H=1024, so `prototype/glue`'s gated
 * kernel serves all of them and nothing in this file is new arithmetic.
 *
 * The pre-FC pair is TWO launches, not one: `glue_add_norm` deliberately does NOT row-index its weight
 * ("one norm vector serves every row"), and the two halves of the fc operand have different weights, so
 * batching them into one launch would feed one weight to both rows. That is a launch for the census, and
 * the census is counted, not assumed.
 *
 * The fc INPUT CONCAT is this file's only piece of new thinking, and it is a pointer, not a kernel:
 * `norm_only(e, w_pre_fc_norm_embedding)` writes to `pair + 0` and
 * `norm_only(h, w_pre_fc_norm_hidden)`  writes to `pair + 1024`, which is §09.2's `[emb || hidden]`
 * order against `W_fc [1024, 2048]`. `speccheck/mtp.py`'s `fc_order` knob (measured 0.642 accept for
 * "eh" against 0.014 for "he", §09.5) is therefore an operand-staging fact the gate can flip and must
 * catch, and the A-13 reading is `pair + 1024` first.
 */
#include "td_launchers.h"

#include "prototype/glue/glue_kernels.h"

namespace td {

void norm_only(const uint16_t* s, const uint16_t* w, uint16_t* n_out) {
    ::glue_add_norm<256><<<1, 256>>>((const hip_bfloat16*)s, nullptr, (const hip_bfloat16*)w,
                                     nullptr, (hip_bfloat16*)n_out, NEG_NONE);
    td_launches += 1;
}

void add_norm(const uint16_t* s, const uint16_t* delta, const uint16_t* w,
              uint16_t* s_out, uint16_t* n_out) {
    ::glue_add_norm<256><<<1, 256>>>((const hip_bfloat16*)s, (const hip_bfloat16*)delta,
                                     (const hip_bfloat16*)w, (hip_bfloat16*)s_out,
                                     (hip_bfloat16*)n_out, NEG_NONE);
    td_launches += 1;
}

}  // namespace td
