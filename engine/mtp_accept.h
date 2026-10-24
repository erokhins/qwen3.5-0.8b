#pragma once
// §09 / M2 — the ACCEPT loop: k=1 speculative decoding on the engine's own bits, behind a flag that
// defaults off (`--mtp-accept 1` / `QWEN_MTP_ACCEPT`, `prototype/mtp_accept/ACCEPT-BUILD.md` §4).
//
// WHAT ONE ROUND IS. The request sits at state S_p with a token `cur` to feed at position p and a
// CANDIDATE `d` that the §09 draft block proposed for position p+1. The round feeds the funded 2-row
// body — `[cur at p, d at p+1]` — through `submit_wave_body`, which is the shipping §07.1 order at
// width 2 (`engine/mtp_verify.h`: `submit_token_body` IS `submit_wave_body` with n = 1, so there is
// no second code path through the 24 layers). Both norm rows are priced by ONE pass over the tied
// head, and the accept test is:
//
//     accept  <=>  argmax(logits row 0) == d
//
// evaluated ON DEVICE (`engine/mtp_verdict_kernels.h`) and read back as four ints, so the round never
// pays a vocabulary-sized download to ask a question about two integers.
//
//  accept   emit `d` at p+1 and row 1's argmax at p+2; the state the body left IS S_{p+2}; nothing is
//           restored. Row 1's own KV lane is the accepted token's lane, so it stays.
//  reject   emit row 0's argmax at p+1; the state must become S_{p+1}, which is what the body's
//           per-layer mid-checkpoint holds; restore it, and pull `prefix_len` back to p+1.
//
// WHY THE ROUND RUNS ON THE LIVE STATE AND NOT ON A SHADOW. `engine/mtp_verify.h` built a shadow
// because it was a SIDE pass that had to leave the run untouched. An accept round is not a side pass:
// on an accept its result IS the run, so copying the whole mixer state back afterwards would bill
// 19.27 MiB for work the body already did in place. Living on the state means an accept costs ZERO
// restore bytes and only a reject pays, which is the cheaper schedule at every accept rate the plan
// admits (the draft's own profile: 0.640/0.578/0.684/0.718 by draft position). The shadow's lane
// save/restore is also not needed here and is not regressed: row 0 re-stages the lane the step already
// wrote with the identical bytes, and row 1's lane on a reject is ABOVE `prefix_len`, which §08.6 inv.
// 4 makes unread and the next append overwrites — 0 bytes restored, the same argument
// `engine/mtp_verify.h`'s rollback table makes for the probe.
//
// THE CHECKPOINT, and why it is not a pre-body snapshot. A reject needs S_{p+1}: every layer's mixer
// state AFTER row 0 consumed `cur` and BEFORE the candidate was seen. A pre-body snapshot gives S_p and
// the body's end gives S_{p+2}; the body is layer-major so "every layer, after row 0" is a thing that
// exists at no single address in the batched stream. `submit_wave_body` now makes it exist per layer
// when `sc.gdn_ckpt` is armed (see `decode_step.h`'s `GdnCheckpoint` and the two sites in
// `decode_step.cpp`): each GDN layer gets ONE `conv_mid_window_kernel` derivation — the shipped batched
// conv is untouched — and each layer's per-token-step recurrence is followed by a 1 MiB state copy
// before row 1 runs. The per-token recurrence is a PIN, not a preference: `--mtp-midsplit` already
// refuses a reference arm that is not at `--gdn-loop 1`, for the reason `ACCEPT-BUILD.md` §4b records.
// The bill is stated in `Stats` and printed, because a cost that only exists in prose is a cost nobody
// priced:
//
//   every round   +18 conv-window derivations, +18 mid-state captures of 1 MiB, +1 verdict kernel, +1
//                 status readback: 20,201,472 B captured, 16 B read back (§4d's itemised table).
//   on a reject   the same 20,201,472 B copied back, in TWO bulk D2Ds (both regions are contiguous).
//
// And the memory it costs, on the record for this model (per-request state 18.66 MiB, the whole bf16
// checkpoint ~4 % of the device pool): a THIRD ~19.27 MiB region per in-flight request, alongside
// `mtp_verify`'s checkpoint and shadow sides when both features are on. Nothing about that implies
// paging, eviction or a weights-on-disk path — see `AGENTS.md`'s scope table.
//
// WHAT THIS FILE IS NOT. No timing. No new numerics band. The bands that judge it are the ones that
// already exist (`engine/gate.py`), and the accept rate's kill line (`a < 0.4568`) is
// `prototype/mtp_accept/ACCEPT-PLAN.md`'s, not this file's to move.

#include <cstdint>
#include <string>

#include "engine/decode_step.h"
#include "engine/mtp_verify.h"
#include "engine/state.h"

namespace engine {
class WeightStore;
namespace mtpaccept {

// The funded width: one value, because one shape was priced. It is `mtp_verify`'s constant rather than
// a second 2, so a change to the verify width cannot leave this file behind.
constexpr int kRows = engine::mtpverify::kRows;

// Everything a round needs that outlives it: the mid-checkpoint regions, the logits tile both rows are
// priced into, the device status word, and the host copy of that word.
struct Side {
    // The mid-round checkpoint: S_{p+1}, written by the body between row 0 and row 1.
    DecodeScratch::GdnCheckpoint ck;
    float* logits = nullptr;        // [kRows][kVocab] F32, device
    int* verdict_dev = nullptr;     // mtpverdict::kVerdictWords ints, device
    // The verdict reduction's partials tile: `kernels::kMaxArgmaxRows * kernels::kArgmaxChunks` argmax
    // words. Nothing outside the two verdict launches reads it, but the round may not `hipMalloc` it — a
    // mid-round allocation is a sync, and the round tax is the thing being removed — so it lives here.
    engine::kernels::ArgmaxWord* verdict_partials = nullptr;
    int verdict_host[4] = {0, 0, 0, 0};
    long long bytes_gdn = 0;        // DERIVED from `state_sizes()`, never typed in
    long long bytes_conv = 0;
    long long bytes() const { return bytes_gdn + bytes_conv; }
    // The per-round capture/restore figures, and the +N submissions the split costs. Derived from the
    // same constants the split uses, so the printed line cannot disagree with the code.
    int subs_per_round = 0;
    int subs_per_reject = 0;
};

Side allocate_side(const RequestState& st);
void free_side(Side& s);

// One round's result, as the driver needs it.
struct Round {
    bool accepted = false;
    int verifier_token = -1;   // argmax of row 0 — the verifier's own prediction for position p+1
    int bonus_token = -1;      // argmax of row 1; meaningful only when `accepted`
    int emitted = 0;           // 2 on an accept, 1 on a reject. Never 0.
    long long restored_bytes = 0;
    int anchor_before = 0;     // the position the round started at
    int anchor_after = 0;      // p+2 on an accept, p+1 on a reject
    // The route the round ACTUALLY ran, off the scratch the body was configured from, not off the flag —
    // the same canary `mtpverify::Result` carries, for the same reason: a stale or inert binary agrees
    // with itself about everything except what its launchers were told.
    kernels::WaveGemm gemm_ran = kernels::WaveGemm::kHand;
    kernels::GdnScan scan_ran = kernels::GdnScan::kSeq;
    // Which attention CORE the round ran, for the same reason and printed on the same `MTPSTAT` line.
    // Since `kMtpRoundAttnWide` the round's form is a compiled-in rule, not a flag, so an echo read off the
    // scratch is the only thing that can show it took: a binary that predated the rule would otherwise
    // print an identical run (`AGENTS.md`: assert a behavioural canary inside the window).
    int attn_wide_ran = 0;
};

// Run one accept round on the LIVE state. `candidate` is the draft block's proposal for position
// `st.prefix_len + 1`; `cur` is the token the request was already going to feed. Both must be in
// [0, kVocab) and the request must have room for two more lanes.
//
// Route forcing lives here, not in the driver: a 2-row body that took hipBLAS or a chunked scan would
// not be the kernel chain the one-row decode step runs, and the whole claim that row 0 is the shipping
// step rests on that forcing (`engine/mtp_verify.h`'s last paragraph). The knobs are restored on every
// exit path including a throw, for the same reason.
//
// `route` is `mtpverify::Route` rather than a second spelling: `kHand` applies that forcing (the default,
// and today's launch sequence byte for byte); `kAmortised` leaves the scratch's own resolved route alone,
// which is what `prototype/mtp_accept/ROW-SCALE.md` needs priced. The mid-round checkpoint this round
// restores from is a one-token-step recurrence's product, so an amortised round over a non-`seq` scan
// throws rather than restoring a state the body never wrote.
Round accept_round(const WeightStore& weights, RequestState& st, DecodeScratch& sc, Side& side,
                   int cur, int candidate, engine::mtpverify::Route route);

// Round counters, printed as one line so the cost and the accept rate are both numbers a reviewer can
// re-derive. `mean_emit` is the headline: tokens emitted per round, which is what the plan's `(1+a)`
// term actually measured.
struct Stats {
    long long rounds = 0;
    long long accepted = 0;
    long long rejected = 0;
    long long emitted = 0;
    long long captured_bytes = 0;
    long long restored_bytes = 0;
    long long verdict_bytes = 0;    // the ONE status word per round, counted so a reader can see that
                                    // it is not a vocabulary
    long long body_rows = 0;        // rows the body was fed: 2 per round, the `kRows` denominator
    long long split_subs = 0;       // the split's extra submissions over the whole run

    double accept_rate() const { return rounds > 0 ? (double)accepted / (double)rounds : 0.0; }
    double mean_emit() const { return rounds > 0 ? (double)emitted / (double)rounds : 0.0; }
};

// Fold one round into the counters. A free function rather than inline arithmetic in the driver so the
// relation `rounds == accepted + rejected` and `emitted == 2*accepted + rejected` is one place's
// responsibility, and `describe()` below can assert it rather than echo it.
void tally(Stats& s, const Side& side, const Round& r);

// The `MTPSTAT` line's payload (the caller prints the prefix). Refuses to describe a tally whose
// counters disagree, because a rate out of inconsistent counters is worse than no rate.
std::string describe(const Stats& s);

}  // namespace mtpaccept
}  // namespace engine
