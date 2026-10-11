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
    // `results.md`. The width is not cosmetic on its way in: `glue_kernels.h:83-102` derives
    // `PER = H / THREADS`, so 256 → 4 elements/thread and an 8-partial warp tree while 1024 →
    // 1 element/thread and a 32-partial tree — a DIFFERENT F32 summation order for the mean of
    // squares. A norm's legal floor is 1 ULP with no cancellation band, so ndiff is the only
    // criterion that could see the difference, and `results.md:113` names ndiff "the criterion doing
    // the work".
    //
    // That question is now measured, not assumed: the gate re-ran at 1024 in the 2026-10-11 clean
    // window (0 failures over 784 comparisons, `g1_n` gated 0 / raw 1, adds bit-exact, every negative
    // still landing on its own stage) and the error profile is identical to 256's. So <256> here is a
    // PREFERENCE, not a requirement — `engine/results.md` E1, now closed. 1024 measures faster
    // (~2.78 vs 3.75 µs/launch, `results.md` §4 — ≈0.05 ms/token over the 49 glue launches) and is
    // admissible; switching is a performance decision for the timing pass, and it inherits the
    // bimodality caveat `bench/performance-status.md` §6 attaches to launch-bound numbers.
    ::glue_add_norm<256><<<1, 256>>>((const hip_bfloat16*)s, (const hip_bfloat16*)delta,
                                     (const hip_bfloat16*)w, (hip_bfloat16*)s_out,
                                     (hip_bfloat16*)n_out, NEG_NONE);
}

}  // namespace engine::kernels
