// Launcher for prototype/glue: the fused residual add + zero-centred RMSNorm (§03.2 + §03.1).
// Its own TU: glue defines `enum Neg`, H, EPS and bf()/bf_round()/block_reduce(), which several
// other ops define too.
#include "kernels.h"

#include "prototype/glue/glue_kernels.h"

namespace engine::kernels {

void glue_add_norm(const uint16_t* s, const uint16_t* delta, const uint16_t* w, uint16_t* s_out,
                   uint16_t* n_out) {
    // 256 threads = the configuration this op's 784-comparison correctness gate actually ran at
    // (`run.py:471` default, all six negatives included), which is the config whose numbers are in
    // `results.md`. The width is not cosmetic here: `glue_kernels.h:83-102` derives
    // `PER = H / THREADS`, so 256 → 4 elements/thread and an 8-partial warp tree while 1024 →
    // 1 element/thread and a 32-partial tree — a DIFFERENT F32 summation order for the mean of
    // squares. A norm's legal floor is 1 ULP with no cancellation band, so a changed order shows up
    // only as ndiff, and `results.md:113` names ndiff "the criterion doing the work": the floor
    // sample (3 / 393 216, `results.md:102` "4 strided/lane … 8-partial tree") was measured on the
    // 256-thread order and nothing else.
    //
    // 1024 is faster (~2.78 vs 3.75 µs/launch, `results.md` §4 — ≈0.05 ms/token over the 49 glue
    // launches), but `results.md:170` says it plainly — "Re-run the gate at 1024 before W5 hard-codes
    // a width" — and that re-run has not happened (engine/results.md, experiment E1). Until it does,
    // the engine launches the gated width.
    ::glue_add_norm<256><<<1, 256>>>((const hip_bfloat16*)s, (const hip_bfloat16*)delta,
                                     (const hip_bfloat16*)w, (hip_bfloat16*)s_out,
                                     (hip_bfloat16*)n_out, NEG_NONE);
}

}  // namespace engine::kernels
