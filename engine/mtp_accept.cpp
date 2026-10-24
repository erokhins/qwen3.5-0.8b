// §09 / M2 — the accept round, implemented. `engine/mtp_accept.h` carries the contract: what a round
// is, why it runs on the live state rather than a shadow, what the mid-round checkpoint costs, and what
// this file is NOT allowed to decide (bands, kill lines, timing).
//
// No arithmetic lives here. The body is `submit_wave_body` at the funded width, the head is
// `kernels::lm_head_batched_xbf16`, the verdict is `kernels::mtp_accept_verdict`, and the rollback is
// two bulk device copies. What this file owns is the ORDER and the FORCING, which is where the risk is.
#include "engine/mtp_accept.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#include "engine/kernels.h"
#include "engine/loader.h"
#include "engine/mtp_meter.h"   // the default-off per-phase round meter; that header owns the semantics

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

Side allocate_side(const RequestState& st, int k) {
    if (k < kMinK || k > kMaxK) {
        throw std::runtime_error("allocate_side: --mtp-k " + std::to_string(k) + " is not a funded"
                                 " width (" + std::to_string(kMinK) + " or " + std::to_string(kMaxK) +
                                 "); the body width, the checkpoint slot count and the"
                                 " verdict word all derive from it, so an unfunded value is not a"
                                 " wider round but an unallocated one");
    }
    const StateSizes s = st.sizes();
    Side side;
    side.k = k;
    side.rows = rows_of_k(k);
    side.words = engine::kernels::kVerdictWordsFor(side.rows);
    side.ck.slots = kMidSlotsFor(side.rows);
    side.bytes_gdn = (long long)s.gdn_state_bytes;
    side.bytes_conv = (long long)s.conv_window_bytes;
    // One region set per slot, each laid out with the live state's own per-layer strides, so a restore of
    // slot `n` is two bulk copies and not a per-layer loop (`accept_round` below).
    ACCEPT_CK(hipMalloc((void**)&side.ck.mid_gdn, (size_t)side.ck.slots * s.gdn_state_bytes));
    ACCEPT_CK(hipMalloc((void**)&side.ck.mid_conv, (size_t)side.ck.slots * s.conv_window_bytes));
    ACCEPT_CK(hipMalloc((void**)&side.logits, (size_t)side.rows * kVocab * sizeof(float)));
    ACCEPT_CK(hipMalloc((void**)&side.verdict_dev, (size_t)side.words * sizeof(int)));
    ACCEPT_CK(hipMalloc((void**)&side.verdict_partials,
                        (size_t)engine::kernels::kMaxArgmaxRows *
                            engine::kernels::kArgmaxChunks *
                            sizeof(engine::kernels::ArgmaxWord)));

    // THE SPLIT'S BILL, derived from the same constants `decode_step.cpp`'s split sites use rather than
    // typed in, and corrected to the DERIVATION form (§4c): the shipped batched conv is untouched, so per
    // GDN layer per SLOT the checkpoint costs exactly TWO extra submissions — the `conv_mid_window_kernel`
    // derivation and the 1 MiB mid-state capture — plus THREE per round for the verdict (its strip launch,
    // its combine since `ROUND-TAX.md` (B) split the row across blocks) and its status readback. A partial
    // accept restores the ONE slot the prefix stopped at, in two bulk D2Ds.
    //
    // At k=1 (slots=1) this is `2 * kNumGdnLayers + 3` = 39, the figure §4d itemises; at k=2 (slots=2) it
    // is 75. The figure this replaces (`3 * kNumGdnLayers`) billed a recurrence launch the pinned form
    // does not add: a round runs the per-token recurrence, which is what the shipped one-row step runs and
    // what `--gdn-loop 1` pins the reference arm to, so there is no extra walk to pay for.
    side.subs_per_round = 2 * kNumGdnLayers * side.ck.slots + 3;
    side.subs_per_reject = 2;
    return side;
}

void free_side(Side& s) {
    if (s.ck.mid_gdn != nullptr) (void)hipFree(s.ck.mid_gdn);
    if (s.ck.mid_conv != nullptr) (void)hipFree(s.ck.mid_conv);
    if (s.logits != nullptr) (void)hipFree(s.logits);
    if (s.verdict_dev != nullptr) (void)hipFree(s.verdict_dev);
    if (s.verdict_partials != nullptr) (void)hipFree(s.verdict_partials);
    s = Side{};
}

Round accept_round(const WeightStore& weights, RequestState& st, DecodeScratch& sc, Side& side,
                   int k, int cur, int cand1, int cand2, engine::mtpverify::Route route) {
    // The per-round GEOMETRY, derived from the k THIS round was asked for and never from `side.rows`.
    // `side` owns the buffers (allocated for the run's ceiling) and this block owns the widths, which is
    // what makes one side legally serve both depths (`engine/mtp_accept.h`: "a prefix, except the slot
    // count"). The ceiling check is a throw, not an assert: a round wider than the side would write past
    // the logits tile and past the checkpoint, and only the caller would notice.
    if (k < kMinK || k > side.k) {
        throw std::runtime_error("accept_round: asked for a " + std::to_string(k) + "-candidate round on"
                                 " a side allocated for " + std::to_string(side.k) + "; the widest round"
                                 " a side can serve is the k it was allocated with, because the logits"
                                 " tile, the checkpoint slots and the verdict words are all sized from it");
    }
    const int rows = rows_of_k(k);
    const int words = engine::kernels::kVerdictWordsFor(rows);
    // THE PER-ROUND CHECKPOINT VIEW, and the one piece of legality the allocation cannot express: the
    // buffers hold the ceiling's slots, but `submit_wave_body` requires `slots == rows - 1` EXACTLY, so a
    // k=1 round on a k=2 side has to arm one slot, not two. Slot 0 means the same thing at either width
    // ("the state after row 0"), so sharing the buffers is correct rather than merely in-bounds.
    DecodeScratch::GdnCheckpoint ck;
    ck.mid_gdn = side.ck.mid_gdn;
    ck.mid_conv = side.ck.mid_conv;
    ck.slots = kMidSlotsFor(rows);
    if (sc.wave_rows < rows) {
        throw std::runtime_error("accept_round: the scratch is " + std::to_string(sc.wave_rows) +
                                 " rows tall and the round needs " + std::to_string(rows));
    }
    // The candidate list, built once so the body, the verdict and the host fold all read the same array
    // rather than three spellings of "candidate i". `cand2` is -1 exactly when the geometry has no second
    // candidate, and that is checked rather than assumed: a k=2 round fed a missing second proposal would
    // speculate on -1 and call it a token id.
    const int cands[kMaxK] = {cand1, cand2};
    if (cur < 0 || cur >= kVocab) {
        throw std::runtime_error("accept_round: token out of range (cur=" + std::to_string(cur) + ")");
    }
    for (int i = 0; i < k; ++i) {
        if (cands[i] < 0 || cands[i] >= kVocab) {
            throw std::runtime_error("accept_round: candidate " + std::to_string(i + 1) + " is " +
                                     std::to_string(cands[i]) + ", outside the tied table — a " +
                                     std::to_string(k) + "-candidate round needs all " +
                                     std::to_string(k) + " proposals");
        }
    }
    const int anchor = st.prefix_len;
    if (anchor + rows > st.cap) {
        // The round appends `rows` KV lanes; refusing is the only honest answer, because a round that
        // wrote past `cap` is a memory fault, and a silently truncated one is a wrong answer.
        throw std::runtime_error("accept_round: anchor " + std::to_string(anchor) + " + " +
                                 std::to_string(rows) + " rows is outside cap " +
                                 std::to_string(st.cap) + " — the run has reached its context limit");
    }
    Round r;
    r.anchor_before = anchor;
    r.rows = rows;
    r.k = k;

    // THE ROUTE FORCING, for the same reason and with the same words as `mtp_verify.cpp`: under `kHand` a
    // one-row decode body is the hand GEMV and the sequential scan BY RULE, so those are the bits row 0 has
    // to be, and a width-2 body left on the library route would take hipBLAS's N=2 solution — legal,
    // in-band, and not the kernel chain the shipping step ran. Under `kAmortised` the scratch's own resolved
    // route stands, which is the route `prototype/mtp_accept/ROW-SCALE.md` measured at 1.09-1.13 of a step
    // and therefore the route the round has to be priced on.
    const kernels::WaveGemm gemm = sc.wave_gemm;
    const kernels::GdnScan scan = sc.gdn_scan;
    const DecodeScratch::GdnCheckpoint* saved_ckpt = sc.gdn_ckpt;
    if (route == engine::mtpverify::Route::kHand) {
        sc.wave_gemm = kernels::WaveGemm::kHand;
        sc.gdn_scan = kernels::GdnScan::kSeq;
    } else if (sc.gdn_scan != kernels::GdnScan::kSeq) {
        // The other half of the forcing the route switch does NOT reach, and deliberately a refusal rather
        // than a silent override: this round always arms the mid-round checkpoint, and that checkpoint only
        // exists as an address when the recurrence walks one token-step per launch (`kCollapse` submits one
        // recurrence launch per body, so there is no row-0 boundary for the copy to sit at). Serving a
        // `collapse` round a seq scan would make a named flag inert; restoring a state the body never wrote
        // would emit wrong tokens. `decode_run.cpp` refuses the pair at the flag; this is the guard for any
        // caller that did not come through the flag.
        throw std::runtime_error("accept_round: an amortised round restores a mid-round checkpoint, which"
                                 " is a one-token-step sequential recurrence's end state; the driver"
                                 " resolved gdn_scan="
                                 + std::string(kernels::gdn_scan_name(sc.gdn_scan)));
    }
    sc.gdn_ckpt = &ck;
    // And the attention form, which is not route-dependent: a round's body is a speculation round whatever
    // weight walk it rides, and the wide wave block was priced at PREFILL row counts and is a loss here
    // (`decode_step.h`'s `kMtpRoundAttnWide`, whose comment carries the wall A/B numbers and the reason a
    // row-count predicate would have moved prefill bits instead). The flag arm cannot see this: the round
    // reaches `attn_decode_layer` through the same `submit_wave_body` a prefill wave does, and the launcher
    // has no other way to tell the two apart.
    const int saved_wide = arm_mtp_round_attn_form(sc);
    // The route in force for this round, read off the scratch AFTER the decision above and before anything
    // restores it, so the driver's `MTPSTAT` echo reports the launch sequence rather than the flag.
    r.gemm_ran = sc.wave_gemm;
    r.scan_ran = sc.gdn_scan;
    r.attn_wide_ran = sc.attn_fattn_wide;

    const auto restore_knobs = [&sc, gemm, scan, saved_ckpt, saved_wide] {
        sc.wave_gemm = gemm;
        sc.gdn_scan = scan;
        sc.gdn_ckpt = saved_ckpt;
        sc.attn_fattn_wide = saved_wide;
    };

    try {
        const int tokens[kMaxRows] = {cur, cand1, cand2};
        // The round IS the step: one body, `rows` rows, at positions `anchor` .. `anchor+rows-1`, on the
        // request's own state, appending its own KV lanes. `submit_wave_body` advances `prefix_len` to
        // anchor + rows, which is the answer only if EVERY candidate was accepted; the fold below takes it
        // back to the last position the verifier agreed with.
        //
        // `phase=body` here carries the mid-round checkpoint's own submissions (2 per GDN layer per slot),
        // and it is NOT split out: the capture is interleaved per layer inside this one call, so a boundary
        // would mean 18 device drains inside the body whose pipelining is the thing being priced. The meter
        // prints that as `status=not_metered` rather than guessing at it — see `engine/mtp_meter.h`.
        {
            const engine::mtpmeter::PhaseGuard body(engine::mtpmeter::kBody);
            submit_wave_body(weights, st, sc, tokens, rows);
        }

        // ONE pass over the 485 MiB table for all the rows, then ONE readback: the verdict kernel argmaxes
        // every row and resolves every candidate comparison on device, so the round's whole vocabulary-sized
        // download is a handful of ints (`engine/mtp_verdict_kernels.h` for why the tie rule is the hard
        // part, not the reduction).
        //
        // Three phases, not one, because the funding question is exactly this split: `head` and
        // `verdict_kernel` close on an ADDED sync (`sync_added=1` — they are pure submissions today, and
        // without a wait their device time would be charged to the readback that happens to block), while
        // `verdict_readback` is a blocking D2H already (`sync_added=0`) and is the round's ENTIRE
        // vocabulary-sized download: 4 ints at k=1, 7 at k=2, not 2 MB. That contrast is what
        // `--mtp-verify`'s `logits_d2h` line is the counterpoint to.
        {
            const engine::mtpmeter::PhaseGuard head(engine::mtpmeter::kHead);
            kernels::lm_head_batched_xbf16((const uint16_t*)weights.at(kHead).device, sc.norm,
                                           side.logits, kVocab, rows);
        }
        {
            const engine::mtpmeter::PhaseGuard verdict(engine::mtpmeter::kVerdict);
            kernels::mtp_accept_verdict(side.logits, kVocab, rows, cand1, cand2, side.verdict_dev,
                                        side.verdict_partials);
        }
        {
            const engine::mtpmeter::PhaseGuard readback(engine::mtpmeter::kReadback);
            ACCEPT_CK(hipMemcpy(side.verdict_host, side.verdict_dev,
                                (size_t)words * sizeof(int), hipMemcpyDeviceToHost));
        }
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
    //
    // THE FOLD, in the order the rule is written: read every row's argmax, read every per-row equality
    // bit, then walk the prefix and stop at the first miss. The walk is on the host for one reason and it is
    // not cost: the block that resolved row 1's bit has no order with the block that resolved row 0's, and
    // inventing one (a third launch, or an atomic) would make the PREFIX depend on arrival order for the
    // sake of moving three integer comparisons. What must be on device — the argmax, whose tie rule §03.4
    // fixes — is on device, and each block echoes the candidate it compared, so a stale proposal is still
    // caught here rather than agreed with.
    {
        // `accept_path` is the host tail of the round on BOTH branches: decoding the verdict words and the
        // candidate-echo check. It is metered on its own so a full accept's near-zero host cost is a printed
        // fact next to a partial accept's restore, rather than a paragraph about it.
        const engine::mtpmeter::PhaseGuard accept_path(engine::mtpmeter::kAcceptPath);
        for (int i = 0; i < rows; ++i) r.row_token[i] = side.verdict_host[i];
        for (int i = 0; i < k; ++i) {
            r.hit[i] = side.verdict_host[engine::kernels::kVerdictHitAt(rows, i)];
            // The word the kernel echoed is the word it was given. Cheap, and it is the only field that can
            // prove the comparison used THIS round's candidate rather than a stale one.
            const int echo = side.verdict_host[engine::kernels::kVerdictEchoAt(rows, i)];
            if (echo != cands[i]) {
                throw std::runtime_error("accept_round: the verdict echoes candidate " +
                                         std::to_string(i + 1) + " as " + std::to_string(echo) +
                         ", asked " + std::to_string(cands[i]));
            }
        }
        while (r.n_accepted < k && r.hit[r.n_accepted] != 0) r.n_accepted++;
        r.accepted = r.n_accepted == k;
        // The two fields the k=1 log line has always printed, spelled as the rows they are: row 0's argmax
        // (`verifier=`) and row 1's (`bonus=`, the token an accept emits after its candidate). At k=1
        // `bonus=` is what gets emitted; on a reject it is the row the round did NOT use, exactly as it was
        // before a third row existed.
        r.verifier_token = r.row_token[0];
        r.bonus_token = r.row_token[1];
        // The emitted stream, in the order it goes out: the accepted candidates, then the verifier's token
        // for the position after them. `emit[n]` is the bonus, and at k=1 this is [cand] + [row1 argmax] on
        // an accept and [row0 argmax] on a reject — today's behaviour, derived rather than branched.
        for (int i = 0; i < r.n_accepted; ++i) r.emit[i] = cands[i];
        r.emit[r.n_accepted] = r.row_token[r.n_accepted];
        r.n_emit = r.n_accepted + 1;
        r.emitted = r.n_emit;
    }

    if (r.accepted) {
        // Nothing to undo. The body's end state IS S_{p+rows}, every norm row was priced from it, and each
        // candidate row's KV lane is an accepted token's own lane at its own position.
        r.anchor_after = anchor + rows;
        if (st.prefix_len != r.anchor_after) {
            throw std::runtime_error("accept_round: an accepted round left prefix_len at " +
                                     std::to_string(st.prefix_len) + ", expected " +
                                     std::to_string(r.anchor_after));
        }
    } else {
        // The prefix stopped at n < k, so the emitted bonus sits at p+n+1 and the mixer state has to be
        // S_{p+n+1} — mid-checkpoint slot n, restored in two bulk copies because both regions are contiguous
        // within a slot with per-layer strides identical to the live state's. At k=1, n is 0 and this is
        // byte-for-byte the old single-slot restore.
        //
        // The KV lanes the rejected candidate rows wrote are NOT restored: with `prefix_len` back to p+n+1
        // they are above the prefix, which §08.6 inv. 4 makes unread, and the next round's row 0 re-stages
        // them with the real tokens' K/V. That is the same 0-bytes argument `engine/mtp_verify.h` makes for
        // the probe, and here it is load-bearing rather than a witness.
        const int slot = r.n_accepted;
        {
            const engine::mtpmeter::PhaseGuard rollback(engine::mtpmeter::kRollback);
            // THE ONE SPELLING OF THE ROLLBACK, now shared with the session driver's stop cut
            // (`restore_prefix`, whose comment records why the offsets are BYTE offsets). The copies and
            // the `prefix_len` assignment are moved verbatim, so a partial accept restores exactly what it
            // restored before the extraction.
            r.restored_bytes = restore_prefix(st, side, anchor, slot, rows);
            r.anchor_after = anchor + r.n_accepted + 1;
            r.restored_slot = slot;
        }
    }

    // §08.6's invariants, on the host, after EITHER branch. A restore that landed a shifted region or
    // a prefix that disagrees with the state is exactly the failure a plausible-looking next round hides.
    std::string broken;
    {
        const engine::mtpmeter::PhaseGuard lane(engine::mtpmeter::kLaneBookkeeping);
        broken = st.check_invariants();
    }
    if (!broken.empty()) {
        throw std::runtime_error("accept_round: §08.6 invariant broken after the " +
                                 std::string(r.accepted ? "accept" : "partial accept") + ": " +
                                 broken);
    }
    return r;
}

void tally(Stats& s, const Side& side, const Round& r) {
    // EVERY per-round figure comes off the ROUND, not off the side. With `--mtp-adaptive 1` one side serves
    // both depths, so a cost read from `side.ck.slots` would bill a 2-row round the 3-row capture it did not
    // pay for (`captured_bytes`, `status_d2h_bytes` and `split_subs` would each over-report, and the round
    // tax that `ROUND-TAX.md` priced would stop being the round's). At flag-off and at a fixed
    // `--mtp-k` the round's width IS the side's, so every line here is the number it was before.
    const int slots = r.rows - 1;   // the mid-checkpoint this body actually captured
    s.rounds++;
    s.k = side.k;
    if (s.k_low == 0 || r.k < s.k_low) s.k_low = r.k;
    if (r.k > s.k_high) s.k_high = r.k;
    s.body_rows += r.rows;
    s.emitted += r.emitted;
    s.candidates += r.n_accepted;
    s.captured_bytes += (long long)slots * side.slot_bytes();
    s.verdict_bytes += (long long)engine::kernels::kVerdictWordsFor(r.rows) * (long long)sizeof(int);
    s.split_subs += 2 * kNumGdnLayers * slots + 3;
    if (r.hit[0] != 0) s.depth1++;
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
    // counters is worse than no rate: a round is exactly one full accept or one partial one, and a round
    // emits its accepted candidates plus its own bonus token, so `emitted` is determined by the others.
    // At k=1 `candidates == accepted` and this reduces to the old `emitted == 2*accepted + rejected`.
    if (s.rounds != s.accepted + s.rejected) {
        throw std::runtime_error("mtpaccept: rounds=" + std::to_string(s.rounds) + " != accepted " +
                                 std::to_string(s.accepted) + " + rejected " +
                                 std::to_string(s.rejected) + "; do not read a rate out of this run");
    }
    if (s.emitted != s.candidates + s.rounds) {
        throw std::runtime_error("mtpaccept: emitted=" + std::to_string(s.emitted) +
                                 " is not candidates_accepted " + std::to_string(s.candidates) +
                                 " + rounds " + std::to_string(s.rounds) +
                                 "; the emit counts are wrong");
    }
    if (s.k > 1 && s.depth1 < s.accepted) {
        // A depth-2 accept is by definition also a depth-1 accept, so this cannot happen and its presence
        // would mean the prefix rule was not applied — the round would have accepted a candidate behind a
        // rejected one. Checked rather than assumed because the whole losslessness argument rests on it.
        throw std::runtime_error("mtpaccept: depth1=" + std::to_string(s.depth1) +
                                 " is below the full-prefix count " + std::to_string(s.accepted) +
                                 "; a candidate behind a rejected one was accepted");
    }
    std::ostringstream os;
    os << "rounds=" << s.rounds << " accepted=" << s.accepted << " rejected=" << s.rejected
       << " emitted=" << s.emitted << " accept_rate=" << s.accept_rate()
       << " mean_tokens_per_round=" << s.mean_emit() << " body_rows=" << s.body_rows
       << " captured_bytes=" << s.captured_bytes << " restored_bytes=" << s.restored_bytes
       << " status_d2h_bytes=" << s.verdict_bytes << " split_subs=" << s.split_subs;
    // Appended, never inserted: everything before this point is the byte-identical k=1 line. `depth1` and
    // `candidates` are the two numbers that make the k=2 rate readable as n1 + n2 rather than as one
    // aggregate, and `k=` says which geometry produced all of them (the same statement the run's `MTPGEOM`
    // line makes at k=1, where these fields would be new bytes on a byte-compared line).
    if (s.k > 1) {
        os << " k=" << s.k << " depth1=" << s.depth1
           << " depth1_rate=" << s.depth1_rate() << " candidates_accepted=" << s.candidates;
    }
    // Appended only when the run's rounds did NOT all share one depth, which is `--mtp-adaptive 1`'s
    // signature and never happens flag-off. Without it a reader would see `k=2` (the allocation) above a
    // `mean_tokens_per_round` that belongs to a mixture of a 3-row and a 2-row body, and the rate-vs-cost
    // arithmetic that judges this feature would be unreadable — the same reason `body_rows` exists at all.
    if (s.k_low != s.k_high) {
        os << " k_ran=" << s.k_low << ".." << s.k_high;
    }
    return os.str();
}

long long restore_prefix(RequestState& st, const Side& side, int anchor, int slot, int rows) {
    // THE SLOT EXISTS OR THIS IS A DEVICE COPY FROM OUTSIDE THE BUFFER. A body of `rows` rows captures
    // `rows - 1` mid-checkpoints (`kMidSlotsFor`), which with `--mtp-adaptive 1` can be narrower than the
    // side's allocation, so the bound is the ROUND's width and not the side's.
    if (rows < kMinK + 1 || slot < 0 || slot >= rows - 1) {
        throw std::runtime_error("restore_prefix: slot " + std::to_string(slot) + " of a " +
                                 std::to_string(rows) + "-row round is not a captured checkpoint"
                                 " (a body of rows rows writes slots 0..rows-2); refusing rather than"
                                 " copying an uninitialised state over the live one");
    }
    // BYTE arithmetic on the slot offset, not element arithmetic: `mid_gdn`/`mid_conv` are `float *` and
    // `bytes_gdn`/`bytes_conv` are byte counts taken from `state_sizes()`, so `mid_conv + slot *
    // bytes_conv` would advance four bytes per element and read a quarter of the buffer past the end at
    // slot 1 (witness: the k=2 restore faulted in the runtime's own memcpy on the second slot, while slot
    // 0 — offset zero — was fine, which is exactly the shape a k=1-only gate cannot see).
    ACCEPT_CK(hipMemcpyAsync(st.gdn_state,
                             (const char*)side.ck.mid_gdn + (size_t)slot * (size_t)side.bytes_gdn,
                             (size_t)side.bytes_gdn, hipMemcpyDeviceToDevice));
    ACCEPT_CK(hipMemcpyAsync(st.conv_window,
                             (const char*)side.ck.mid_conv + (size_t)slot * (size_t)side.bytes_conv,
                             (size_t)side.bytes_conv, hipMemcpyDeviceToDevice));
    // The KV lanes the body wrote above this position are NOT restored: above `prefix_len` they are unread
    // (§08.6 inv. 4) and the next append overwrites them — the same 0-bytes argument `accept_round`'s
    // partial-accept branch has always made.
    st.prefix_len = anchor + slot + 1;
    // ONE slot was copied back, not the side's whole capture: at k=2 a round captures two slots and a stop
    // restores the one the transcript ended at.
    return side.slot_bytes();
}

namespace {

// The draft cache refuses only when its lane ceiling is reached, and it has no eviction
// (`engine/mtp_draft.h` offers `reset` and the LIFO `truncate_lane`). A session therefore needs a
// policy for a full cache, and the policy is: reset once, print it, and retry — so a long single turn
// that fills the 512 usable lanes keeps speculating instead of printing `DRAFT-SKIP` forever. Returns
// the refusal that SURVIVES the reset ("" when the draft may run); a non-empty answer here is a genuine
// geometry problem and is printed by the caller, never swallowed.
std::string refused_after_reset(Session& ss, int for_pos) {
    std::string why = engine::mtpdraft::refusal(*ss.draft);
    if (why.empty()) return why;
    std::printf("DRAFT-RESET reason=%s would_propose_for=%d\n", why.c_str(), for_pos);
    std::fflush(stdout);
    engine::mtpdraft::reset(*ss.draft);
    ss.draft_resets++;
    return engine::mtpdraft::refusal(*ss.draft);
}

// The next round's proposals, spelled as the run loop spells them: one draft row per candidate of the
// round that is about to run, the first from the target's own final-norm row of the last position the
// body FED and agreed with, the second from the draft block's own output row `x_d` with the previous
// proposal as `y_next` (`prototype/mtp_accept/DRAFT-COND.md` D4/D5). `need_next` is the NEXT round's
// depth, which the policy may have just demoted — with the policy off it is the same number as `need`.
void produce_proposals(Session& ss, const WeightStore& store, const uint16_t* h_row,
                       int tok_next, int draft_pos, int need_next) {
    ss.pending = -1;
    ss.pending_pos = -1;
    ss.pending2 = -1;
    ss.pending2_pos = -1;
    if (need_next <= 0 || h_row == nullptr) return;
    const std::string why = refused_after_reset(ss, draft_pos);
    if (!why.empty()) {
        std::printf("DRAFT-SKIP reason=%s would_propose_for=%d\n", why.c_str(), draft_pos);
        std::fflush(stdout);
        return;
    }
    const engine::mtpdraft::Proposal p =
        engine::mtpdraft::draft_round(store, *ss.draft_w, *ss.draft, h_row, tok_next, draft_pos);
    ss.pending = p.token;
    ss.pending_pos = draft_pos;
    ss.made++;
    ss.lanes_max = std::max(ss.lanes_max, (long long)ss.draft->lanes_max);
    if (need_next > 1) {
        const std::string why2 = refused_after_reset(ss, draft_pos + 1);
        if (!why2.empty()) {
            std::printf("DRAFT-SKIP reason=%s would_propose_for=%d row=2\n", why2.c_str(),
                        draft_pos + 1);
            std::fflush(stdout);
            return;
        }
        // ALIASING, safe by construction and copied from the run loop: `xd` is written at stage 14 and
        // read at stage 3 of the next call, one stream in issue order, so no sync is needed and none is
        // taken (a sync here would be a silent rate-changer).
        const engine::mtpdraft::Proposal q = engine::mtpdraft::draft_round(
            store, *ss.draft_w, *ss.draft, ss.draft->xd, p.token, draft_pos + 1);
        ss.pending2 = q.token;
        ss.pending2_pos = draft_pos + 1;
        ss.lanes_max = std::max(ss.lanes_max, (long long)ss.draft->lanes_max);
    }
}

}  // namespace

void session_new_turn(Session& ss) {
    if (ss.draft != nullptr) engine::mtpdraft::reset(*ss.draft);
    if (ss.adapt) {
        ss.adapt = std::make_unique<engine::mtpadapt::Tracker>(ss.k_ceiling);
    }
    ss.pending = ss.pending_pos = ss.pending2 = ss.pending2_pos = -1;
    ss.said_plain = false;
    ss.stats = Stats{};
    // Pre-state the geometry so `describe()` of a turn that ran no round cannot report a depth it did
    // not run (`tally` would otherwise leave `k` at the kDefaultK it is initialised to).
    if (ss.side != nullptr) {
        ss.stats.k = ss.side->k;
        ss.stats.k_low = ss.side->k;
        ss.stats.k_high = ss.side->k;
    }
}

Body session_body(Session& ss, const WeightStore& store, RequestState& st, DecodeScratch& sc,
                  int cur, int budget_left) {
    Body b;
    const int anchor = st.prefix_len;       // the absolute position `cur` occupies
    const int round_k = ss.adapt ? ss.adapt->k() : ss.side->k;
    // `kPlain` is not a narrower round: it means the policy abandoned speculation for this turn, and the
    // driver then takes the branch it already had for "no proposal pending".
    const bool speculate = round_k > engine::mtpadapt::kPlain;
    const int need = round_k;
    const int rows = rows_of_k(round_k);
    // Headroom first: a round appends `rows` KV lanes and `accept_round` refuses to write past `cap`
    // (that refusal is §08.6's), so the driver asks before it calls and pays a plain step otherwise.
    const bool room_for_round = anchor + rows <= st.cap;
    // The budget rule, stated in `session_body`'s contract: a round may emit `rows` tokens, so it runs
    // only when the turn can print all of them. No budget rollback exists.
    const bool budget_fits = budget_left >= rows;
    // The norm row the NEXT draft stands from: the target's final-norm row of the last position the body
    // FED and the verifier agreed with — row `n_accepted` of the tile the round just wrote, which is
    // llama.cpp's `verify_h[min(n_accepted, rows-1)]` in the other engine's spelling.
    const uint16_t* draft_norm_row = nullptr;
    int n_accepted = 0;
    engine::mtpaccept::Round ar{};
    bool cut_stopped = false;
    int n_print = 0;

    if (!speculate || ss.pending < 0 || (need > 1 && ss.pending2 < 0) || !room_for_round ||
        !budget_fits) {
        // Every skip is printed: a round that did not run is a body the caller is being charged one
        // token for, and a silently shorter speculation is the empty-parse failure class.
        if (ss.pending >= 0 && !room_for_round) {
            std::printf("ROUND-SKIP reason=context_limit anchor=%d cap=%lld rows=%d\n", anchor,
                        (long long)st.cap, rows);
        } else if (ss.pending >= 0 && room_for_round && !budget_fits) {
            std::printf("ROUND-SKIP reason=budget_tail anchor=%d budget_left=%d rows=%d\n", anchor,
                        budget_left, rows);
        } else if (ss.pending >= 0 && need > 1 && ss.pending2 < 0) {
            std::printf("ROUND-SKIP reason=short_proposal_set anchor=%d pending=%d need=%d rows=%d\n",
                        anchor, 1, need, rows);
            engine::mtpdraft::truncate_lane(*ss.draft);   // the orphan proposal's lane
        }
        std::fflush(stdout);
        const engine::StepResult r = engine::decode_step(store, st, sc, cur, nullptr);
        b.n_emit = 1;
        b.tok[0] = r.token;
        b.label[0] = st.prefix_len - 1;
        // The row this step priced: `finish_token`'s own logits buffer, one row, and the label its token
        // printed under. Read only by the opt-in session dump (`Body`'s field comment for the contract).
        b.logits_dev = sc.logits;
        b.logits_rows = 1;
        b.logits_pos0 = st.prefix_len - 1;
        b.logits_emitted = 1;
        draft_norm_row = sc.norm;                        // row 0 is the row this step priced
        ss.plain_steps++;
    } else {
        ar = accept_round(store, st, sc, *ss.side, round_k, cur, ss.pending,
                          need > 1 ? ss.pending2 : -1, ss.route);
        tally(ss.stats, *ss.side, ar);
        ss.rounds++;
        ss.adjudicated++;
        if (ar.accepted) ss.accepted++;
        n_accepted = ar.n_accepted;
        draft_norm_row = sc.norm + (size_t)n_accepted * kHidden;
        if (n_accepted < need) {
            // The proposals the round did not consume wrote lanes the target never swallowed; they are
            // the LAST ones appended, so a LIFO truncate removes precisely them.
            for (int i = n_accepted; i < need; ++i) engine::mtpdraft::truncate_lane(*ss.draft);
        }
        // §07.5 per emitted token, in order, generation phase only — the run loop's `emit_one` tail, but
        // decided here because the state has to end where the transcript does.
        n_print = ar.n_emit;
        for (int i = 0; i < ar.n_emit; ++i) {
            if (ar.emit[i] == engine::kStopToken) {
                n_print = i + 1;
                cut_stopped = true;
                break;
            }
        }
        if (cut_stopped && n_print < ar.n_emit) {
            // The stop sits at emit `n_print - 1`, absolute position `anchor + n_print`, and the state
            // must end with that token PENDING (printed, never stepped) exactly as a flag-off stop does.
            const long long cut = restore_prefix(st, *ss.side, anchor, n_print - 1, ar.rows);
            ss.stats.restored_bytes += cut;
            ss.suppressed += ar.n_emit - n_print;
        }
        for (int i = 0; i < n_print; ++i) {
            b.tok[i] = ar.emit[i];
            b.label[i] = anchor + i;      // GEN label = absolute position minus one
        }
        b.n_emit = n_print;
        b.was_round = true;
        b.stopped = cut_stopped;
        // The rows the round priced: `side.logits` is `[rows][kVocab]` F32 and is NOT rewritten by
        // anything between here and the caller (the draft's head pass has its own logits buffer), so the
        // dump below can still read it. `n_print` rows decided a printed token; the ones above it were
        // priced and dropped, and a dump that dropped them would hide which row a verdict threw away.
        b.logits_dev = ss.side->logits;
        b.logits_rows = ar.rows;
        b.logits_pos0 = anchor;
        b.logits_emitted = n_print;
        {
            std::printf("ACCEPT round=%lld anchor=%d fed=%d cand=%d cand_pos=%d accept=%d k=%d"
                        " n_accepted=%d emitted=%d pos=%d->%d restored_bytes=%lld suppressed=%d",
                        (long long)ss.rounds, ar.anchor_before, cur, ss.pending,
                        ss.pending_pos, ar.hit[0], ar.k, ar.n_accepted, ar.emitted, ar.anchor_before,
                        ar.anchor_after, ar.restored_bytes, ar.n_emit - n_print);
            if (ar.k > 1) {
                std::printf(" accept2=%d cand2=%d cand2_pos=%d row2=%d restored_slot=%d", ar.hit[1],
                            ss.pending2, ss.pending2_pos, ar.row_token[2], ar.restored_slot);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
        if (ss.adapt) {
            const engine::mtpadapt::Change ch = ss.adapt->note(round_k, ar.emitted);
            if (ch.fired) {
                std::printf("MTPADAPT round=%lld k=%d est=%.4f prev_k=%d window_rounds=%d plain=%d\n",
                            (long long)ss.rounds, ch.to, ch.est, ch.from, ch.seen,
                            ch.to == engine::mtpadapt::kPlain ? 1 : 0);
                std::fflush(stdout);
            }
        }
    }

    // The next proposals. A stopped turn is over — the token that ended it is pending and the next `feed`
    // resets the draft cache — so drafting behind a stop would pay a whole head pass for a proposal no
    // body will ever see.
    if (b.stopped) {
        ss.pending = ss.pending_pos = ss.pending2 = ss.pending2_pos = -1;
        return b;
    }
    const int need_next = ss.adapt ? (ss.adapt->abandoned() ? 0 : ss.adapt->k()) : need;
    if (need_next == 0) {
        if (!ss.said_plain) {
            ss.said_plain = true;      // once per turn: an event, not a heartbeat
            std::printf("MTPADAPT-PLAIN reason=abandoned round=%lld draft_work_stopped=1\n",
                        (long long)ss.rounds);
            std::fflush(stdout);
        }
        ss.pending = ss.pending_pos = ss.pending2 = ss.pending2_pos = -1;
        return b;
    }
    produce_proposals(ss, store, draft_norm_row, b.tok[b.n_emit - 1],
                      b.label[b.n_emit - 1] + 2, need_next);
    return b;
}

}  // namespace engine::mtpaccept
