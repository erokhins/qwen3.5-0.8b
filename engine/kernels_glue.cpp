// Launcher for prototype/glue: the fused residual add + zero-centred RMSNorm (§03.2 + §03.1).
// Its own TU: glue defines `enum Neg`, H, EPS and bf()/bf_round()/block_reduce(), which several
// other ops define too.
#include "kernels.h"

#include "prototype/glue/glue_kernels.h"

namespace engine::kernels {

void glue_add_norm(const uint16_t* s, const uint16_t* delta, const uint16_t* w, uint16_t* s_out,
                   uint16_t* n_out) {
    // 1024 threads: H/THREADS = 1 element per thread, the width prototype/glue's sweep prefers
    // (results.md there); one block per row. `delta == nullptr` is the norm-only form, and
    // `s_out == nullptr` with it is exactly the plan line the op's gate calls "-".
    ::glue_add_norm<1024><<<1, 1024>>>((const hip_bfloat16*)s, (const hip_bfloat16*)delta,
                                       (const hip_bfloat16*)w, (hip_bfloat16*)s_out,
                                       (hip_bfloat16*)n_out, NEG_NONE);
}

}  // namespace engine::kernels
