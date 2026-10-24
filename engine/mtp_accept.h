#pragma once
// §09 / M2 — the ACCEPT loop: k=1 speculative decoding on the engine's own bits, behind a flag that
// defaults off (`--mtp-accept 1` / `QWEN_MTP_ACCEPT`, `prototype/mtp_accept/ACCEPT-BUILD.md` §4).
//
// WHAT ONE ROUND IS. The request sits at state S_p with a token `cur` to feed at position p and up to two
// CANDIDATES `d1`, `d2` that the §09 draft block proposed for positions p+1 and p+2. `--mtp-k` picks the
// width: k=1 (the shipped default) feeds the funded 2-row body `[cur at p, d1 at p+1]`; k=2 feeds the
// 3-row body `[cur at p, d1 at p+1, d2 at p+2]`. Both go through `submit_wave_body`, which is the shipping
// §07.1 order at that width (`engine/mtp_verify.h`: `submit_token_body` IS `submit_wave_body` with n = 1,
// so there is no second code path through the 24 layers). Every norm row is priced by ONE pass over the
// tied head, and the accept test is the LONGEST CORRECT PREFIX, spelled once just below:
//
//     accept d1  <=>  argmax(row 0) == d1
//     accept d2  <=>  that AND argmax(row 1) == d2
//
// evaluated ON DEVICE (`engine/mtp_verdict_kernels.h`) as one int equality per row and read back as a
// handful of ints (`kVerdictWordsFor(rows)`: 4 at k=1, 7 at k=2), so the round never pays a
// vocabulary-sized download to ask a question about a few integers.
//
//  n == k   emit `d1..dn` then row n's argmax; the state the body left IS S_{p+n+1}; nothing is restored.
//           Every candidate row's KV lane is an accepted token's lane, so all of them stay.
//  n < k    emit `d1..dn` then row n's argmax; the state must become S_{p+n+1}, which is mid-checkpoint
//           slot n; restore it, and pull `prefix_len` back to p+n+1. At k=1 that is exactly the old
//           single-reject case: slot 0, one restore, one emitted token.
//
// The KV lanes the un-accepted candidate rows wrote are NOT restored: with `prefix_len` back below them
// they are above the prefix, which §08.6 inv. 4 makes unread and the next append overwrites.
//
// WHY THE ROUND RUNS ON THE LIVE STATE AND NOT ON A SHADOW. `engine/mtp_verify.h` built a shadow
// because it was a SIDE pass that had to leave the run untouched. An accept round is not a side pass:
// on an accept its result IS the run, so copying the whole mixer state back afterwards would bill
// 19.27 MiB for work the body already did in place. Living on the state means a full accept costs ZERO
// restore bytes and only a partial one pays, which is the cheaper schedule at every accept rate the plan
// admits (the draft's own profile: 0.640/0.578/0.684/0.718 by draft position). The shadow's lane
// save/restore is also not needed here and is not regressed: row 0 re-stages the lane the step already
// wrote with the identical bytes, and any row above the accepted prefix is ABOVE `prefix_len`, which §08.6
// inv. 4 makes unread and the next append overwrites — 0 bytes restored, the same argument
// `engine/mtp_verify.h`'s rollback table makes for the probe.
//
// THE CHECKPOINT, and why it is not a pre-body snapshot. A round that stops with n candidates accepted
// needs S_{p+n+1}: every layer's mixer state AFTER row n consumed its token and before row n+1 was seen.
// A pre-body snapshot gives S_p and the body's end gives S_{p+rows}; the body is layer-major so "every
// layer, after row n" is a thing that exists at no single address in the batched stream — except for the
// LAST row, where it is not needed at all, because that state is the body's end state. `submit_wave_body`
// makes the others exist per layer
// when `sc.gdn_ckpt` is armed (see `decode_step.h`'s `GdnCheckpoint` and the two sites in
// `decode_step.cpp`): each GDN layer gets ONE `conv_mid_window_kernel` derivation per slot — the shipped
// batched conv is untouched — and each layer's per-token-step recurrence is followed by a 1 MiB state copy
// after every row that has a slot. The per-token recurrence is a PIN, not a preference: `--mtp-midsplit`
// already refuses a reference arm that is not at `--gdn-loop 1`, for the reason `ACCEPT-BUILD.md` §4b
// records. The bill is stated in `Stats` and printed, because a cost that only exists in prose is a cost
// nobody priced, and `slots = rows - 1` (1 at k=1, 2 at k=2):
//
//   every round   +18*slots conv-window derivations, +18*slots mid-state captures of 1 MiB, +1 verdict
//                 launch pair, +1 status readback: slots * 20,201,472 B captured and
//                 `kVerdictWordsFor(rows)` ints (16 B at k=1, 28 B at k=2) read back — §4d's itemised
//                 table, with `slots` as its only variable.
//   on a partial  ONE slot copied back: 20,201,472 B in TWO bulk D2Ds (both regions are contiguous per
//   accept        slot), whichever slot the accepted prefix stopped at. A full accept restores nothing.
//
// And the memory it costs, on the record for this model (per-request state 18.66 MiB, the whole bf16
// checkpoint ~4 % of the device pool): a THIRD ~19.27 MiB region per in-flight request at k=1, and two of
// them at k=2, alongside `mtp_verify`'s checkpoint and shadow sides when both features are on. Nothing
// about that implies paging, eviction or a weights-on-disk path — see `AGENTS.md`'s scope table.
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

// The two funded widths. `kRowsK1` is the shipped round — one candidate, two body rows — and `kRowsK2` is
// the k=2 round this build adds: three body rows `[cur at p, cand1 at p+1, cand2 at p+2]`. Both are
// derived from `mtp_verify`'s 2 rather than typed twice, so a change to the verify width cannot leave this
// file behind.
constexpr int kRowsK1 = engine::mtpverify::kRows;
constexpr int kRowsK2 = kRowsK1 + 1;
// The widest round this build can run, and the only values `--mtp-k` may name: 1 and 2.
constexpr int kMaxRows = kRowsK2;
constexpr int kMaxK = kMaxRows - 1;      // at most two candidates
// The narrowest funded round, and the LOWER BOUND of what `--mtp-k` may name. Deliberately its own constant
// rather than `kDefaultK`: the default moved to 2, and `--mtp-k 1` has to stay nameable or every wall A/B,
// gate (m) run and lossless witness in this repo that names k=1 would be refused by the flag parser.
// Two range checks read this (`decode_run.cpp`'s `--mtp-k` parse, `allocate_side`), so one edit moves both.
constexpr int kMinK = 1;
// The shipped round. k=2 is the measured optimum, not a guess and not a taste:
// `prototype/mtp_accept/K2-RESULT.md` §7 — +16.53/+17.76 % at 4,152 lanes (+20.68 % with
// `--mtp-dense --mtp-prompt-lanes 146`) against k=1's +9.75…+10.38 %, round/step 1.82-1.84 under the 1.95
// kill line, lossless 49/49 on the full corpus, and gate (m) admits rounds at k=1 and k=2. k=3 projects
// ≈ +18 % off the nested ladder, which is inside k=2's measured spread on a geometry that is not
// implemented, not swept and not gated, so it is NOT the default.
constexpr int kDefaultK = kMaxK;

// The candidate count a body of `rows` rows speculates on, and the `--mtp-k` value it corresponds to.
constexpr int kOfRows(int rows) { return rows - 1; }
constexpr int rows_of_k(int k) { return k + 1; }

// THE ACCEPT RULE, in the one place that spells it, because three files would otherwise have to agree:
// a round accepts the LONGEST CORRECT PREFIX of its candidates, never a subset.
//
//     hit[i]  <=>  argmax(logits row i) == cand(i+1)      (device, one int equality per row)
//     n       =  the leading run of hits: hit[0], and only then hit[1]
//
// and the round then emits `cand(1..n)` followed by `argmax(row n)`, and restores the mid-checkpoint of
// row `n` — except when `n == k`, where the body's end state is already the state after the last row and
// nothing is restored. `n < k` is therefore the whole condition for a restore, and `n` is the slot index.
//
// THE SAME RULE IN llama.cpp, which is the outside witness that this is the accepted semantics and not an
// invention: `common/speculative.cpp:1858-1871` (`mtp`'s `accept()`) counts `i_prev` while
// `id_keep.seq_id == id_prop.seq_id && i_prev == n_keep`, i.e. a prefix walked in order and stopped at the
// first mismatch, then keeps `verify_h[min(n_accepted, n_rows - 1)]`
// (`tools/server/server-context.cpp:4590`, `common/sampling.cpp:690-723`). `min(n_accepted, rows-1)` is
// this file's "slot `n`, and no slot when `n == k`" in the other engine's spelling: the row whose state
// you keep is the last row the target agreed with, and if it agreed with all of them that is the body's
// last row, whose state is already live.
constexpr int kVerdictRowsFor(int k) { return k + 1; }
static_assert(kVerdictRowsFor(kMinK) == kRowsK1, "k=1 is the 2-row round");
static_assert(kVerdictRowsFor(kDefaultK) == kRowsK2, "the default round is the 3-row k=2 round");
static_assert(kVerdictRowsFor(2) == kRowsK2, "k=2 is the 3-row round");
// A mid-checkpoint slot per row except the last (`decode_step.h`'s rule), stated once so the side, the
// body and the rollback cannot each invent their own count.
constexpr int kMidSlotsFor(int rows) { return rows - 1; }
static_assert(kMidSlotsFor(kRowsK2) == 2,
             "the 3-row body checkpoints after rows 0 and 1 and NOT after row 2: the state after the last"
             " row is the body's end state");

// Everything a round needs that outlives it: the mid-checkpoint regions, the logits tile the rows are
// priced into, the device status word, and the host copy of that word.
struct Side {
    // The mid-round checkpoint: one state per row except the last, written by the body between rows.
    // `ck.slots` is set by `allocate_side` and `submit_wave_body` refuses a body whose row count does not
    // match it, so the buffer count and the geometry are one fact.
    DecodeScratch::GdnCheckpoint ck;
    // Both fields default to the SAME constant, and `rows` is DERIVED from it rather than typed: a default
    // `Side` whose `rows` and `k` disagreed would be a geometry bug that only a 3-row run could show.
    int rows = rows_of_k(kDefaultK);  // the body width this side was allocated for: 2 (k=1) or 3 (k=2)
    int k = kDefaultK;                // candidates per round = rows - 1
    int words = engine::kernels::kVerdictWords;
    float* logits = nullptr;        // [rows][kVocab] F32, device
    int* verdict_dev = nullptr;     // `words` ints, device
    // The verdict reduction's partials tile: `kernels::kMaxArgmaxRows * kernels::kArgmaxChunks` argmax
    // words. Nothing outside the two verdict launches reads it, but the round may not `hipMalloc` it — a
    // mid-round allocation is a sync, and the round tax is the thing being removed — so it lives here.
    engine::kernels::ArgmaxWord* verdict_partials = nullptr;
    int verdict_host[engine::kernels::kVerdictWordsFor(kMaxRows)] = {0, 0, 0, 0, 0, 0, 0};
    // ONE slot's two regions. The capture writes `ck.slots` of these per round; a restore copies exactly
    // one (the slot the accepted prefix ended at), which is why the two numbers are not the same field.
    long long bytes_gdn = 0;        // DERIVED from `state_sizes()`, never typed in
    long long bytes_conv = 0;
    long long slot_bytes() const { return bytes_gdn + bytes_conv; }
    // What the side OWNS, which is also what a round CAPTURES into: `ck.slots` slots. At k=1 this is the
    // 20,201,472 B `MTPACCEPT checkpoint_bytes=` has always printed, because a k=1 side has one slot.
    long long bytes() const { return (long long)(ck.slots > 0 ? ck.slots : 1) * slot_bytes(); }
    long long captured_bytes() const { return bytes(); }
    // The per-round capture/restore figures, and the +N submissions the split costs. Derived from the
    // same constants the split uses, so the printed line cannot disagree with the code.
    int subs_per_round = 0;
    int subs_per_reject = 0;
};

// Allocate a side for a round of `k` candidates (`--mtp-k`). The width is fixed for the life of the Side
// on purpose: a run whose geometry could change mid-flight would make `body_rows` a sum over mixed shapes
// and the rate a mixture, which is the one thing the win bar may not be.
Side allocate_side(const RequestState& st, int k = kDefaultK);
void free_side(Side& s);

// One round's result, as the driver needs it.
struct Round {
    int rows = rows_of_k(kDefaultK);
    int k = kDefaultK;
    // The device verdict, unpacked. `row_token[i]` is row i's argmax (§03.4's rule, on device) and
    // `hit[i]` is the device's `row_token[i] == cand(i+1)`; `n_accepted` is the leading run of hits,
    // which the host folds because the two blocks that produced the hits have no order between them.
    int row_token[kMaxRows] = {-1, -1, -1};
    int hit[kMaxK] = {0, 0};
    int n_accepted = 0;
    // True iff the WHOLE prefix was accepted (`n_accepted == k`). At k=1 this is exactly the old accept
    // bit, which is what lets `ACCEPT accept=` keep its meaning unchanged at the shipped width.
    bool accepted = false;
    int verifier_token = -1;   // argmax of row 0 — the verifier's own prediction for position p+1
    int bonus_token = -1;      // argmax of row 1; the token an accept emits after its candidate
    // The emitted stream, in order: the accepted candidates then the verifier's token for the position
    // after them. `n_emit == n_accepted + 1`, always at least 1.
    int emit[kMaxRows] = {-1, -1, -1};
    int n_emit = 0;
    int emitted = 0;           // == n_emit. Never 0.
    long long restored_bytes = 0;
    int restored_slot = -1;    // which mid-checkpoint was copied back; -1 when nothing was (full accept)
    int anchor_before = 0;     // the position the round started at
    int anchor_after = 0;      // p + n_accepted + 1 — p+2 on a k=1 accept, p+1 on a k=1 reject
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

// Run one accept round on the LIVE state. `cur` is the token the request was already going to feed, at
// position `st.prefix_len`; `cand1` is the draft block's proposal for `st.prefix_len + 1`, and `cand2`
// its proposal for `st.prefix_len + 2` — pass -1 for `cand2` on a 2-row (k=1) round, and it is refused on
// a 3-row one. All must be in [0, kVocab) and the request must have room for `side.rows` more lanes.
//
// Route forcing lives here, not in the driver: a body that took hipBLAS or a chunked scan would not be the
// kernel chain the one-row decode step runs, and the whole claim that row 0 is the shipping step rests on
// that forcing (`engine/mtp_verify.h`'s last paragraph). The knobs are restored on every exit path
// including a throw, for the same reason.
//
// `route` is `mtpverify::Route` rather than a second spelling: `kHand` applies that forcing (the default,
// and today's launch sequence byte for byte); `kAmortised` leaves the scratch's own resolved route alone,
// which is what `prototype/mtp_accept/ROW-SCALE.md` needs priced. The mid-round checkpoint this round
// restores from is a one-token-step recurrence's product, so an amortised round over a non-`seq` scan
// throws rather than restoring a state the body never wrote.
Round accept_round(const WeightStore& weights, RequestState& st, DecodeScratch& sc, Side& side,
                   int cur, int cand1, int cand2, engine::mtpverify::Route route);

// Round counters, printed as one line so the cost and the accept rate are both numbers a reviewer can
// re-derive. `mean_emit` is the headline: tokens emitted per round, which is what the plan's `(1+a)`
// term actually measured.
struct Stats {
    long long rounds = 0;
    long long accepted = 0;         // rounds that accepted the WHOLE prefix (n == k)
    long long rejected = 0;         // rounds that stopped short of it (n < k)
    // Rounds whose FIRST candidate was accepted. At k=1 this equals `accepted` exactly; at k=2 it is the
    // depth-1 rate the draft ladder's `n1` column measures, and without it a reader can see `accepted`
    // (= the depth-2 rate) and `mean_tokens_per_round` but cannot check one against the other.
    long long depth1 = 0;
    // Sum of `n_accepted` over all rounds: the candidates the round consumed. The identity that ties the
    // counters together at either width is `emitted == candidates + rounds`.
    long long candidates = 0;
    long long emitted = 0;
    long long captured_bytes = 0;
    long long restored_bytes = 0;
    long long verdict_bytes = 0;    // the ONE status word per round, counted so a reader can see that
                                    // it is not a vocabulary
    long long body_rows = 0;        // rows the body was fed: `rows` per round, the `kRows` denominator
    long long split_subs = 0;       // the split's extra submissions over the whole run
    // The geometry these counters describe, so `describe()` can say which fields mean what. A run has one
    // width for its whole life (`allocate_side` fixes it), so this is a constant and not an average.
    int k = kDefaultK;

    double accept_rate() const { return rounds > 0 ? (double)accepted / (double)rounds : 0.0; }
    double depth1_rate() const { return rounds > 0 ? (double)depth1 / (double)rounds : 0.0; }
    double mean_emit() const { return rounds > 0 ? (double)emitted / (double)rounds : 0.0; }
};

// Fold one round into the counters. A free function rather than inline arithmetic in the driver so the
// relation `rounds == accepted + rejected` and `emitted == candidates + rounds` is one place's
// responsibility, and `describe()` below can assert it rather than echo it.
void tally(Stats& s, const Side& side, const Round& r);

// The `MTPSTAT` line's payload (the caller prints the prefix). Refuses to describe a tally whose
// counters disagree, because a rate out of inconsistent counters is worse than no rate.
//
// At the shipped width (k=1) the line carries EXACTLY the fields it always did: invariant 1 of
// `prototype/mtp_accept/K2-RESULT.md` is that a `--mtp-k 1` run is byte-identical to the binary that
// predates k=2, and this line is in the compared set. The k>1 fields (`k=`, `depth1=`, `candidates=`) are
// appended only when the geometry is k>1, because otherwise they would be new bytes on a run whose whole
// claim is that nothing moved. The geometry is still stated unconditionally at k=1, on the `MTPGEOM` line
// (`decode_run.cpp`) — a NEW line no comparison covers — so "every log says which geometry produced its
// bits" is satisfied without touching a byte of `MTPSTAT` or `MTPACCEPT`.
std::string describe(const Stats& s);

}  // namespace mtpaccept
}  // namespace engine
