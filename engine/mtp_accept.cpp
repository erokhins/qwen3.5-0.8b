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
    if (k < kDefaultK || k > kMaxK) {
        throw std::runtime_error("allocate_side: --mtp-k " + std::to_string(k) + " is not a funded"
                                 " width (1 or 2); the body width, the checkpoint slot count and the"
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
                   int cur, int cand1, int cand2, engine::mtpverify::Route route) {
    const int rows = side.rows;
    const int k = side.k;
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
    sc.gdn_ckpt = &side.ck;
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
                                (size_t)side.words * sizeof(int), hipMemcpyDeviceToHost));
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
            // BYTE arithmetic on the slot offset, not element arithmetic: `mid_gdn`/`mid_conv` are
            // `float *` and `bytes_gdn`/`bytes_conv` are byte counts taken from `state_sizes()`, so
            // `mid_conv + slot * bytes_conv` would advance four bytes per element and read a quarter of
            // the buffer past the end at slot 1 (witness: the k=2 restore faulted in the runtime's own
            // memcpy on the second slot, while slot 0 — offset zero — was fine, which is exactly the
            // shape a k=1-only gate cannot see).
            ACCEPT_CK(hipMemcpyAsync(st.gdn_state,
                                     (const char*)side.ck.mid_gdn + (size_t)slot * (size_t)side.bytes_gdn,
                                     (size_t)side.bytes_gdn, hipMemcpyDeviceToDevice));
            ACCEPT_CK(hipMemcpyAsync(st.conv_window,
                                     (const char*)side.ck.mid_conv
                                         + (size_t)slot * (size_t)side.bytes_conv,
                                     (size_t)side.bytes_conv, hipMemcpyDeviceToDevice));
            st.prefix_len = anchor + r.n_accepted + 1;
            r.anchor_after = anchor + r.n_accepted + 1;
            // ONE slot was copied back, not the side's whole capture: at k=2 the round captured two slots
            // and restored the one the prefix stopped at, and `MTPSTAT captured_bytes=` keeps that
            // distinction (the capture is per round, the rollback is per stop).
            r.restored_bytes = side.slot_bytes();
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
    s.rounds++;
    s.k = side.k;
    s.body_rows += r.rows;
    s.emitted += r.emitted;
    s.candidates += r.n_accepted;
    s.captured_bytes += side.captured_bytes();
    s.verdict_bytes += (long long)side.words * (long long)sizeof(int);
    s.split_subs += side.subs_per_round;
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
    return os.str();
}

}  // namespace engine::mtpaccept
