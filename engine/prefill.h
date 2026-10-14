#pragma once
// The prefill wave (§06.1), in its P2 shape: `T` tokens through the existing per-token kernels under
// one submission stream, with the logits taken once at the end (§06.5).
//
// What a wave is NOT yet, said here so nobody has to discover it in a diff: no batched kernel and no
// chunked GDN scan. Every launch a wave makes is a launch `decode_step` already made for that one
// token, in the same order, so the wave is byte-identical to the per-token prompt loop it replaces —
// that equivalence IS the package's gate (`progress/2026-10-12/plan-prefill-2026-10-12.md` §4, row P2). P4/P5/P6
// batch the families *inside* `submit_token_body`; the wave's shape is what lets them do that without
// also restructuring the driver, and that is why P2 lands before them.
//
// What the wave does own, and why it is not just a for-loop left in the driver:
//   * positions. The wave is the only place that decides which token sits at which §02.4 `P`; under
//     waves (plan §2.1) that is `P..P+T_w-1`, and it must not be re-derived at every call site.
//   * the §06.5 end condition. One row is sampled — the last — and handed back as `y0`. Nothing
//     crosses the prefill/decode boundary except the §08.1 state and `prefix_len`, and this file
//     cannot leak anything else across it because it returns no state of its own.
//   * one sync instead of T. `submit_token_body` never waits, so the wave queues every body and then
//     calls `finish_token` once. Under `kAll` that saving is deliberately given back (see below).
//   * the §07.5 phase split. A wave contains no stop check at all — there is nothing here that could
//     end a run — so "a stop id inside the prompt never ends the run" is structural rather than a
//     condition someone has to remember to write. That was the first device run's bug;
//     `engine/decode_run.cpp` owns the rule for the generate phase, and it is the only phase that can
//     have one.
//
// What it deliberately does NOT do yet, named so nobody goes looking for it: no wave splitting
// (plan §2.1's one wave for `T <= 1024`, then waves of 512 at multiples of 64). Nothing in P2 bounds
// the scratch — every buffer is still one token wide — so one call is one wave however long the token
// list is. When P4 gives the wave `[T_w]`-sized buffers the split belongs HERE, so callers never have
// to change.

#include <cstddef>
#include <string>
#include <vector>

#include "engine/decode_step.h"
#include "engine/state.h"

namespace engine {

class WeightStore;

// Which logits a wave gives back. §06.5 says only `logits[T-1]` is ever used, and that is the
// default; the other setting exists because the gate has to be able to look at the rows that are
// normally thrown away.
enum class PrefillLogits {
    // `last` — §06.5's shape, and `decode_run`'s default: one `lm_head_full`, one vocabulary D2H,
    // one host argmax, over the final normed row of the last token. Every earlier row is submitted
    // and never read back, which is where the "one submission stream, one sync" of P2's goal
    // sentence lives. Rows before the last carry no sample (`PrefillRow::sampled` is -1), so nothing
    // can print an intermediate prompt-phase argmax — a stop id least of all.
    kLast,
    // `all` — every row gets its own `lm_head` + D2H + argmax, i.e. exactly the launches the pre-wave
    // per-token prompt loop made, in the same order. That is the point: it is the mode the
    // byte-identity comparison runs in, because it can be `cmp`'d against a dump taken before P2
    // existed. It costs T vocabulary D2Hs and keeps T x 1 MB of logits on the host, so it is a gate
    // mode and not the production one.
    kAll,
};

// The two spellings `decode_run --prefill-logits` takes, and their inverse. Parsing lives here so a
// typo is refused at the boundary instead of quietly becoming the default.
PrefillLogits parse_prefill_logits(const std::string& name);
const char* prefill_logits_name(PrefillLogits mode);

// One wave position as the driver sees it. `pos_before`/`pos_after` are what §02.4's `P` was, and is
// after this token — the pair the gate's `STEP` line prints and checks.
struct PrefillRow {
    int index = 0;        // wave-local, 0..T-1
    int token = -1;       // the id fed here
    int pos_before = 0;
    int pos_after = 0;
    // This row's argmax, or -1 when the row was never priced (`kLast`, and not the last row). -1 is
    // not a token id: it is `RequestState::sampled_token`'s existing "nothing pending" value.
    int sampled = -1;
    float top_logit = 0.0f;   // only meaningful when `sampled >= 0`
    // The row's vocabulary, when the wave produced it: every row under `kAll`, and the last row under
    // `kLast` too, because that one is §06.5's `y0` and the driver dumps it either way.
    std::vector<float> logits;
};

// The wave's output: one row per token fed, in order. `y0()` is §06.5's first sampled token — the
// last row's sample — and is null for an empty wave (a wave of nothing advances nothing and samples
// nothing, which is what `feed` with no pending sample and no ids means).
struct PrefillResult {
    std::vector<PrefillRow> rows;
    PrefillLogits logits_mode = PrefillLogits::kLast;

    const PrefillRow* y0() const { return rows.empty() ? nullptr : &rows.back(); }
    size_t size() const { return rows.size(); }
    bool empty() const { return rows.empty(); }
};

// Feed `tokens` through the stack as one prefill wave (§06.1), mutating `st` in place: its GDN
// state, conv window and KV advance token by token, and `prefix_len` ends at
// `prefix_len_before + tokens.size()` (§06.5's "nothing but the §08.1 state and `prefix_len` crosses
// the boundary" is precisely the rule this function is not allowed to violate).
//
// Throws what `submit_token_body`/`finish_token` throw: a HIP failure, a position outside the
// allocation, attention's lane ceiling, or a token outside `[0, vocab)`. A throw leaves the state
// advanced by the tokens that did complete — the wave is not transactional — and its callers treat a
// throw as fatal (the session poisons the turn, the run exits), which is the same rule a failed
// `decode_step` has always followed.
PrefillResult prefill_wave(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                           const std::vector<int>& tokens, PrefillLogits mode);

}  // namespace engine
