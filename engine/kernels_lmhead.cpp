// Launcher for prototype/lm_head_gemv: the tied logits GEMV in its FULL-LOGIT form (§03.4). Its own
// TU — lm_head_gemv defines HID, ROWS_PER_BLOCK, FULL_MASK and fkey/unfkey, and `HID` there is the
// same spelling mlp_gemv uses for the same number with a different comment, which is exactly the
// kind of duplicate that must not meet in one translation unit.
#include "kernels.h"

#include <hip/hip_runtime.h>

#include <stdexcept>

#include "engine/lm_head_xbf16_kernels.h"   // the BF16-staged form (route S1 item 1)
#include "prototype/lm_head_gemv/lm_head_gemv_kernels.h"

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
}  // namespace

void lm_head_full(const uint16_t* table, const float* h, float* logits, int vocab) {
    // Unlike proj_gemv and mlp_gemv, this kernel maps rows with a literal `>> 5` and a 16/8/4/2/1
    // shuffle ladder, so it is a wave32 kernel by construction (that is what its 242 GB/s gate row
    // was measured on). Saying so here beats producing a wrong histogram of logits on a wave64 part.
    if (warp_width() != 32) {
        throw std::runtime_error("lm_head_full: the gated kernel assumes a 32-lane wavefront; this"
                                 " device reports " + std::to_string(warp_width()));
    }
    if (vocab <= 0) throw std::runtime_error("lm_head_full: vocab must be > 0");
    constexpr int kRowsPerBlock = ROWS_PER_BLOCK;   // the prototype's default and its gated width
    const int threads = 32 * kRowsPerBlock;
    const int blocks = (vocab + kRowsPerBlock - 1) / kRowsPerBlock;
    // The FULL form, not `lm_head_gemv_fused`: the fused kernel resolves its argmax with
    // `atomicMax` over a (float-key, index) word, which returns the HIGHEST id on a tie, and §03.4
    // fixes ties to the LOWEST. The engine therefore writes the logits and takes the argmax on the
    // host with a strict `>` (decode_step.cpp), costing ~1 MB D2H + a host scan per token — a cost
    // §07.4 corrected after measurement: fusing argmax is worth ~1 % on this GPU because the 485 MB
    // read, not the 1 MB write, is the whole price of the head.
    lm_head_gemv_full<<<blocks, threads>>>((const hip_bfloat16*)table, h, logits, vocab,
                                           kRowsPerBlock);
}

// The BF16-staged form (route S1 item 1): the identical launch — same grid, same block, same
// wave32-only refusal, same literal 16/8/4/2/1 ladder — with the final-norm row staged as BF16 and
// widened in registers, so `decode_step`'s `finish_token` no longer needs the widening launch in front
// of it. That launch is 1 of the 97 and this kernel is the family the ledger measures at the read
// roof, so the honest statement is the one in `engine/lm_head_xbf16_kernels.h`: it is bit-identical by
// construction and its timing effect is the A/B's question, not this launcher's claim.
void lm_head_full_xbf16(const uint16_t* table, const uint16_t* h, float* logits, int vocab) {
    if (warp_width() != 32) {
        throw std::runtime_error("lm_head_full_xbf16: the gated kernel assumes a 32-lane wavefront;"
                                 " this device reports " + std::to_string(warp_width()));
    }
    if (vocab <= 0) throw std::runtime_error("lm_head_full_xbf16: vocab must be > 0");
    if (h == nullptr) throw std::runtime_error("lm_head_full_xbf16: h is null");
    constexpr int kRowsPerBlock = ROWS_PER_BLOCK;
    const int threads = 32 * kRowsPerBlock;
    const int blocks = (vocab + kRowsPerBlock - 1) / kRowsPerBlock;
    engine::lmheadxbf16::lm_head_gemv_full_xbf16<true><<<blocks, threads>>>(
        (const hip_bfloat16*)table, h, logits, vocab, kRowsPerBlock);
}

}  // namespace engine::kernels
