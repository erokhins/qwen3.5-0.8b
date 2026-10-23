// §09 / M2 — the accept round, implemented. `engine/mtp_accept.h` carries the contract: what a round
// is, why it runs on the live state rather than a shadow, what the mid-round checkpoint costs, and what
// this file is NOT allowed to decide (bands, kill lines, timing).
//
// No arithmetic lives here. The body is `submit_wave_body` at the funded width, the head is
// `kernels::lm_head_batched_xbf16`, the verdict is `kernels::mtp_accept_verdict`, and the rollback is
// two bulk device copies. What this file owns is the ORDER and the FORCING, which is where the risk is.
#include "engine/mtp_accept.h"

#include <hip/hip_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include "engine/kernels.h"
#include "engine/loader.h"

#define ACCEPT_CK(x)                                                              \
    do {                                                                          \
        hipError_t e_ = (x);                                                      \
        if (e_ != hipSuccess) {                                                   \
            throw std::runtime_error(std::string("hip: ") + hipGetErrorString(e_) \
                                     + " at " __FILE__ + ":" + #x);               \
        }                                                                         \
    } while (0)

namespace engine::mtpaccept {

namespace {
constexpr const char* kHead = "lm_head.weight";   // the tied table, as `decode_step.cpp` resolves it
}  // namespace

Side allocate_side(const RequestState& st) {
    const StateSizes s = st.sizes();
    Side side;
    side.bytes_gdn = (long long)s.gdn_state_bytes;
    side.bytes_conv = (long long)s.conv_window_bytes;
    ACCEPT_CK(hipMalloc((void**)&side.ck.mid_gdn, s.gdn_state_bytes));
    ACCEPT_CK(hipMalloc((void**)&side.ck.mid_conv, s.conv_window_bytes));
    ACCEPT_CK(hipMalloc((void**)&side.logits, (size_t)kRows * kVocab * sizeof(float)));
    ACCEPT_CK(hipMalloc((void**)&side.verdict_dev,
                        (size_t)engine::kernels::kVerdictWords * sizeof(int)));

    // THE SPLIT'S BILL, derived from the same constants `decode_step.cpp`'s split sites use rather than
    // typed in, and corrected on this pass to the DERIVATION form (§4c): the shipped batched conv is
    // untouched, so per GDN layer the checkpoint costs exactly TWO extra submissions — the
    // `conv_mid_window_kernel` derivation and the 1 MiB mid-state capture — plus two per round for the
    // verdict kernel and its 16-byte status readback. A reject restores the two regions in two bulk D2Ds.
    //
    // The figure this replaces (`3 * kNumGdnLayers`) billed a recurrence launch the pinned form does not
    // add: an accept round runs the per-token recurrence, which is what the shipped one-row step runs and
    // what `--gdn-loop 1` pins the reference arm to, so there is no extra walk to pay for. If a future
    // round ever resolves a span of 2, add `kNumGdnLayers` back here and say so on the `MTPSTAT` line.
    side.subs_per_round = 2 * kNumGdnLayers + 2;
    side.subs_per_reject = 2;
    return side;
}

void free_side(Side& s) {
    if (s.ck.mid_gdn != nullptr) (void)hipFree(s.ck.mid_gdn);
    if (s.ck.mid_conv != nullptr) (void)hipFree(s.ck.mid_conv);
    if (s.logits != nullptr) (void)hipFree(s.logits);
    if (s.verdict_dev != nullptr) (void)hipFree(s.verdict_dev);
    s = Side{};
}

Round accept_round(const WeightStore& weights, RequestState& st, DecodeScratch& sc, Side& side,
                   int cur, int candidate) {
    if (sc.wave_rows < kRows) {
        throw std::runtime_error("accept_round: the scratch is " + std::to_string(sc.wave_rows) +
                                 " rows tall and the round needs " + std::to_string(kRows));
    }
    if (cur < 0 || cur >= kVocab || candidate < 0 || candidate >= kVocab) {
        throw std::runtime_error("accept_round: token out of range (cur=" + std::to_string(cur) +
                                 ", candidate=" + std::to_string(candidate) + ")");
    }
    const int anchor = st.prefix_len;
    if (anchor + kRows > st.cap) {
        // The round appends two KV lanes; refusing is the only honest answer, because a round that
        // wrote past `cap` is a memory fault, and a silently truncated one is a wrong answer.
        throw std::runtime_error("accept_round: anchor " + std::to_string(anchor) + " + " +
                                 std::to_string(kRows) + " rows is outside cap " +
                                 std::to_string(st.cap) + " — the run has reached its context limit");
    }

    Round r;
    r.anchor_before = anchor;

    // THE ROUTE FORCING, for the same reason and with the same words as `mtp_verify.cpp`: a one-row
    // decode body is the hand GEMV and the sequential scan BY RULE, so those are the bits row 0 has to
    // be, and a width-2 body left on the library route would take hipBLAS's N=2 solution — legal,
    // in-band, and not the kernel chain the shipping step ran. The forced `kSeq` is also what the
    // checkpoint requires: `kCollapse` and `kChunk` submit one recurrence launch per body and so have
    // no row-0 boundary for the copy to sit at.
    const kernels::WaveGemm gemm = sc.wave_gemm;
    const kernels::GdnScan scan = sc.gdn_scan;
    const DecodeScratch::GdnCheckpoint* saved_ckpt = sc.gdn_ckpt;
    sc.wave_gemm = kernels::WaveGemm::kHand;
    sc.gdn_scan = kernels::GdnScan::kSeq;
    sc.gdn_ckpt = &side.ck;

    const auto restore_knobs = [&sc, gemm, scan, saved_ckpt] {
        sc.wave_gemm = gemm;
        sc.gdn_scan = scan;
        sc.gdn_ckpt = saved_ckpt;
    };

    try {
        const int tokens[kRows] = {cur, candidate};
        // The round IS the step: one body, two rows, at positions `anchor` and `anchor + 1`, on the
        // request's own state, appending its own two KV lanes. `submit_wave_body` advances
        // `prefix_len` to anchor + 2, which is the accepted answer; the reject branch below takes it
        // back to anchor + 1.
        submit_wave_body(weights, st, sc, tokens, kRows);

        // ONE pass over the 485 MiB table for both rows, then ONE 16-byte readback: the verdict kernel
        // argmaxes both rows and resolves the accept test on device, so the round's whole
        // vocabulary-sized download is four ints (`engine/mtp_verdict_kernels.h` for why the tie rule
        // is the hard part, not the reduction).
        kernels::lm_head_batched_xbf16((const uint16_t*)weights.at(kHead).device, sc.norm,
                                       side.logits, kVocab, kRows);
        kernels::mtp_accept_verdict(side.logits, kVocab, kRows, candidate, side.verdict_dev);
        ACCEPT_CK(hipMemcpy(side.verdict_host, side.verdict_dev,
                            (size_t)engine::kernels::kVerdictWords * sizeof(int),
                            hipMemcpyDeviceToHost));
    } catch (...) {
        restore_knobs();
        throw;
    }
    restore_knobs();

    // The word offsets come from `engine/kernels.h`, not from the kernel header: a header whose
    // namespace holds a `__global__` may be included by exactly one translation unit
    // (`ACCEPT-BUILD.md` §4e trap 1, whose witness is this build's own duplicate-`__device_stub__` link
    // error), and `engine/kernels_lmhead.cpp` is that TU for the verdict kernel. This file launches it
    // through `kernels::mtp_accept_verdict` and reads its word back, and needs no kernel of its own.
    r.verifier_token = side.verdict_host[engine::kernels::kVerdictRow0Token];
    r.bonus_token = side.verdict_host[engine::kernels::kVerdictRow1Token];
    r.accepted = side.verdict_host[engine::kernels::kVerdictAccept] != 0;
    // The word the kernel echoed is the word it was given. Cheap, and it is the only field that can
    // prove the comparison used THIS round's candidate rather than a stale one.
    if (side.verdict_host[engine::kernels::kVerdictCandidate] != candidate) {
        throw std::runtime_error("accept_round: the verdict echoes candidate " +
                                 std::to_string(side.verdict_host[engine::kernels::kVerdictCandidate]) +
                                 ", asked " + std::to_string(candidate));
    }

    if (r.accepted) {
        // Nothing to undo. The body's end state IS S_{p+2}, both norm rows were priced from it, and
        // row 1's KV lane is the accepted candidate's own lane at its own position.
        r.emitted = 2;
        r.anchor_after = anchor + kRows;
        if (st.prefix_len != r.anchor_after) {
            throw std::runtime_error("accept_round: an accepted round left prefix_len at " +
                                     std::to_string(st.prefix_len) + ", expected " +
                                     std::to_string(r.anchor_after));
        }
    } else {
        // A reject: the candidate is discarded and the verifier's own token takes position p+1, so the
        // mixer state has to be S_{p+1} — the per-layer mid-checkpoint, restored in two bulk copies
        // because both regions are contiguous with per-layer strides identical to the live state's.
        // The KV lane row 1 wrote is NOT restored: with `prefix_len` back to p+1 it is above the prefix,
        // which §08.6 inv. 4 makes unread, and the next round's row 0 re-stages it with the real
        // token's K/V. That is the same 0-bytes argument `engine/mtp_verify.h` makes for the probe, and
        // here it is load-bearing rather than a witness.
        ACCEPT_CK(hipMemcpyAsync(st.gdn_state, side.ck.mid_gdn, (size_t)side.bytes_gdn,
                                 hipMemcpyDeviceToDevice));
        ACCEPT_CK(hipMemcpyAsync(st.conv_window, side.ck.mid_conv, (size_t)side.bytes_conv,
                                 hipMemcpyDeviceToDevice));
        st.prefix_len = anchor + 1;
        r.emitted = 1;
        r.anchor_after = anchor + 1;
        r.restored_bytes = side.bytes();
    }

    // §08.6's invariants, on the host, after EITHER branch. A restore that landed a shifted region or
    // a prefix that disagrees with the state is exactly the failure a plausible-looking next round hides.
    const std::string broken = st.check_invariants();
    if (!broken.empty()) {
        throw std::runtime_error("accept_round: §08.6 invariant broken after the " +
                                 std::string(r.accepted ? "accept" : "reject") + ": " + broken);
    }
    return r;
}

void tally(Stats& s, const Side& side, const Round& r) {
    s.rounds++;
    s.body_rows += kRows;
    s.emitted += r.emitted;
    s.captured_bytes += side.bytes();
    s.verdict_bytes += engine::kernels::kVerdictWords * sizeof(int);
    s.split_subs += side.subs_per_round;
    if (r.accepted) {
        s.accepted++;
    } else {
        s.rejected++;
        s.restored_bytes += r.restored_bytes;
        s.split_subs += side.subs_per_reject;
    }
}

std::string describe(const Stats& s) {
    // The counters are not allowed to disagree with each other, and a rate printed over inconsistent
    // counters is worse than no rate: a round is exactly one accept or one reject, and an accept emits
    // 2 while a reject emits 1, so `emitted` is determined by the other two.
    if (s.rounds != s.accepted + s.rejected) {
        throw std::runtime_error("mtpaccept: rounds=" + std::to_string(s.rounds) + " != accepted " +
                                 std::to_string(s.accepted) + " + rejected " +
                                 std::to_string(s.rejected) + "; do not read a rate out of this run");
    }
    if (s.emitted != 2 * s.accepted + s.rejected) {
        throw std::runtime_error("mtpaccept: emitted=" + std::to_string(s.emitted) +
                                 " is not 2*accepted + rejected; the emit counts are wrong");
    }
    std::ostringstream os;
    os << "rounds=" << s.rounds << " accepted=" << s.accepted << " rejected=" << s.rejected
       << " emitted=" << s.emitted << " accept_rate=" << s.accept_rate()
       << " mean_tokens_per_round=" << s.mean_emit() << " body_rows=" << s.body_rows
       << " captured_bytes=" << s.captured_bytes << " restored_bytes=" << s.restored_bytes
       << " status_d2h_bytes=" << s.verdict_bytes << " split_subs=" << s.split_subs;
    return os.str();
}

}  // namespace engine::mtpaccept
