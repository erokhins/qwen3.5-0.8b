// Launcher for prototype/lm_head_gemv: the tied logits GEMV in its FULL-LOGIT form (§03.4). Its own
// TU — lm_head_gemv defines HID, ROWS_PER_BLOCK, FULL_MASK and fkey/unfkey, and `HID` there is the
// same spelling mlp_gemv uses for the same number with a different comment, which is exactly the
// kind of duplicate that must not meet in one translation unit.
#include "kernels.h"

#include <hip/hip_runtime.h>

#include <stdexcept>

#include "engine/lm_head_batched_kernels.h"  // the N-rows-from-one-table-read form (P7's verify body)
#include "engine/lm_head_xbf16_kernels.h"   // the BF16-staged form (route S1 item 1)
#include "engine/mtp_verdict_kernels.h"    // the accept round's device verdict (M2, D-b)
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

// The batched form (P7's verify body): N query rows of the tied head from ONE pass over the table, which
// is the difference between `delta` = 0.1505 of a step and `delta` = 0.4729 = KILLED
// (`prototype/mtp_delta/D3-RESULT.md`), measured as a shape at 1.0019x of the one-row pass for N=2
// (`prototype/mtp_headbatch/HEADBATCH-RESULT.md`). Same grid, same block, same wave32 refusal, same
// rows-per-block constant as the shipped one-row launch — only the work per block differs, so the two are
// directly comparable in a paired A/B.
//
// NROWS is accepted only as 2, and says so rather than falling back: a deeper verify (k>1) is unfunded
// work (`prototype/mtp_draft/DRAFT-RESULT.md`: "fund a staging pass, do not fund deeper k"), and a
// launcher that quietly served k=3 with a k=2 kernel would be the silent-substitution failure this repo
// keeps recording.
void lm_head_batched_xbf16(const uint16_t* table, const uint16_t* h_rows, float* logits_rows,
                           int vocab, int nrows) {
    if (warp_width() != 32) {
        throw std::runtime_error("lm_head_batched_xbf16: the gated kernel assumes a 32-lane wavefront;"
                                 " this device reports " + std::to_string(warp_width()));
    }
    if (vocab <= 0) throw std::runtime_error("lm_head_batched_xbf16: vocab must be > 0");
    if (h_rows == nullptr) throw std::runtime_error("lm_head_batched_xbf16: h_rows is null");
    if (nrows != kLmHeadBatchedRows) {
        throw std::runtime_error("lm_head_batched_xbf16: this build instantiates NROWS=" +
                                 std::to_string(kLmHeadBatchedRows) + ", asked for " +
                                 std::to_string(nrows) + ". A deeper verify is a funding decision, not"
                                 " a template argument.");
    }
    constexpr int kRowsPerBlock = ROWS_PER_BLOCK;
    const int threads = 32 * kRowsPerBlock;
    const int blocks = (vocab + kRowsPerBlock - 1) / kRowsPerBlock;
    engine::lmheadbatched::lm_head_gemv_batched<kLmHeadBatchedRows><<<blocks, threads>>>(
        (const hip_bfloat16*)table, (const hip_bfloat16*)h_rows, logits_rows, vocab, kRowsPerBlock);
}

// The accept round's verdict (M2, D-b): each logits row is argmaxed by the SHARED pair of launches — the
// same `argmax_partials_kernel` the draft's proposal uses, then `verdict_combine_kernel` — and the round's
// whole readback is `kVerdictWords` ints: two ids, the accept bit, and the candidate it was compared
// against. `nrows` is the verify body's width and nothing else is accepted, for the same reason
// `lm_head_batched_xbf16` refuses other widths: the shape that was priced is the shape that ships.
//
// The two rows share ONE launch pair (`gridDim.y` is the row) rather than one pair per row, because §20.1's
// aliasing bug came from two spellings of one address; here there is one writer of `partials` and one reader.
//
// Deliberately NOT billed on a stage-clock row by this launcher: it belongs to the accept round, and
// the round states its own per-round submission bill (`MTPSTAT`'s `subs_per_round`) rather than pretending
// to be one of §07's nine families.
void mtp_accept_verdict(const float* logits_rows, int vocab, int nrows, int candidate,
                        int* out_dev, ArgmaxWord* partials_dev) {
    if (vocab <= 0) throw std::runtime_error("mtp_accept_verdict: vocab must be > 0");
    if (nrows != kVerdictRows) {
        throw std::runtime_error("mtp_accept_verdict: this build reduces " +
                                 std::to_string(kVerdictRows) +
                                 " logits rows, asked for " + std::to_string(nrows) +
                                 ". A deeper verify is a funding decision, not a grid dimension.");
    }
    if (nrows > kMaxArgmaxRows) {
        throw std::runtime_error("mtp_accept_verdict: " + std::to_string(nrows) +
                                 " rows exceeds kMaxArgmaxRows; the caller's partials buffer is sized by "
                                 "that constant and would be overrun.");
    }
    if (vocab % (kArgmaxChunks * 4) != 0) {
        throw std::runtime_error("mtp_accept_verdict: vocab " + std::to_string(vocab) +
                                 " does not split into " + std::to_string(kArgmaxChunks) +
                                 " strips of whole float4 vectors.");
    }
    if (logits_rows == nullptr || out_dev == nullptr || partials_dev == nullptr) {
        throw std::runtime_error("mtp_accept_verdict: logits, out, or partials is null");
    }
    engine::mtpverdict::argmax_partials_kernel<<<dim3(kArgmaxChunks, nrows),
                                                engine::mtpverdict::kVerdictThreads>>>(
        logits_rows, vocab, vocab / kArgmaxChunks, kArgmaxChunks, partials_dev);
    engine::mtpverdict::verdict_combine_kernel<<<nrows, engine::mtpverdict::kVerdictThreads>>>(
        reinterpret_cast<const engine::mtpverdict::detail::Best*>(partials_dev), kArgmaxChunks,
        candidate, out_dev);
}

// One logits row, argmaxed on device into one `ArgmaxWord`: `kArgmaxChunks` strip blocks, then one combine
// block. No atomics anywhere, because the whole claim is that the fold's result does not depend on the order
// its operands arrive in (`engine/mtp_verdict_kernels.h`), and an arrival-order-dependent combine would
// trade a vocabulary download for a nondeterministic id — a bad trade in a build whose gates are admitted by
// exact equality.
//
// Deliberately NOT billed on a stage-clock row by this launcher, exactly like `mtp_accept_verdict` above:
// these belong to the round, and the round declares its own per-round bill (`kStageSubsPerRound` in
// `engine/mtp_draft.h`, which counts these two launches).
void mtp_argmax_row(const float* logits_row, int vocab, ArgmaxWord* out_dev, ArgmaxWord* partials_dev) {
    if (vocab <= 0) throw std::runtime_error("mtp_argmax_row: vocab must be > 0");
    if (vocab % (kArgmaxChunks * 4) != 0) {
        throw std::runtime_error("mtp_argmax_row: vocab=" + std::to_string(vocab) + " does not split into "
                                 + std::to_string(kArgmaxChunks)
                                 + " strips of a whole number of float4s ("
                                 + std::to_string(kArgmaxChunks * 4)
                                 + " elements per strip). The strips are read vectorised, so an uneven split"
                                   " is a wrong answer, not merely a slower one — refuse rather than"
                                   " silently drop a tail.");
    }
    if (logits_row == nullptr || out_dev == nullptr || partials_dev == nullptr)
        throw std::runtime_error("mtp_argmax_row: logits, out or partials is null");
    const int per_chunk = vocab / kArgmaxChunks;
    // grid.x = the chunk, grid.y = the row (one row here), which is the same (chunk, row) mapping the
    // verdict path uses, so one kernel serves both shapes.
    engine::mtpverdict::argmax_partials_kernel
        <<<dim3(kArgmaxChunks, 1, 1), engine::mtpverdict::kVerdictThreads>>>(
            logits_row, vocab, per_chunk, kArgmaxChunks, partials_dev);
    engine::mtpverdict::row_argmax_combine_kernel<<<1, engine::mtpverdict::kVerdictThreads>>>(
        partials_dev, kArgmaxChunks, out_dev);
}

}  // namespace engine::kernels
