// The prefill wave, in the two calls `decode_step.h` documents. Read that header first: it owns the
// contract this file implements, and `engine/prefill.h`'s header comment owns what a wave is for.
//
// The whole of this file's arithmetic is other people's kernels. Nothing here computes a value,
// chooses a reduction order, or converts a dtype: the wave is `submit_token_body` per token and
// `finish_token` once, which is why a wave and a run of single-token steps cannot disagree — there
// is one implementation of the §07.1 order, in `decode_step.cpp`, and this file calls pieces of it.
//
// The one thing worth reading closely is the `price_this_row` line. Under `kLast` the wave prices only
// its last row, so a T-token prompt is T body submissions and ONE vocabulary D2H — §06.5's "only
// `logits[T-1]` is ever used", and the piece of §1.2's waste this package is allowed to remove. Under
// `kAll` every row is priced, which reproduces the per-token prompt loop's launch sequence exactly;
// that mode exists for the byte-identity gate and pays for it in syncs and host memory.
#include "engine/prefill.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "engine/decode_step.h"

namespace engine {

PrefillLogits parse_prefill_logits(const std::string& name) {
    if (name == "last") return PrefillLogits::kLast;
    if (name == "all") return PrefillLogits::kAll;
    throw std::runtime_error("--prefill-logits takes `last` (§06.5: only the last row is ever used)"
                             " or `all` (every row: the gate's byte-identity mode), not '"
                             + name + "'");
}

const char* prefill_logits_name(PrefillLogits mode) {
    return mode == PrefillLogits::kAll ? "all" : "last";
}

PrefillResult prefill_wave(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                           const std::vector<int>& tokens, PrefillLogits mode) {
    PrefillResult out;
    out.logits_mode = mode;
    out.rows.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        PrefillRow row;
        row.index = static_cast<int>(i);
        row.token = tokens[i];
        row.pos_before = st.prefix_len;   // §02.4: P is the 0-based position of the token being fed
        // Submits the token and waits for nothing, so the stream keeps filling. This is where the
        // position advances (`prefix_len = pos + 1`) and where §08.6's invariants are re-checked.
        submit_token_body(weights, st, sc, tokens[i]);
        row.pos_after = st.prefix_len;

        // The last row is always priced — §06.5's `y0` has to come from somewhere — and under `kAll`
        // so is every other one. Body-then-finish for one row is exactly what `decode_step` is, so in
        // `kAll` the wave's launch sequence is the old prompt loop's, launch for launch, which is
        // what makes the gate's `cmp` a test rather than a restatement.
        const bool price_this_row = (mode == PrefillLogits::kAll) || (i + 1 == tokens.size());
        if (price_this_row) {
            const StepResult sampled = finish_token(weights, st, sc, &row.logits);
            row.sampled = sampled.token;
            row.top_logit = sampled.top_logit;
        }
        out.rows.push_back(std::move(row));
    }
    return out;
}

}  // namespace engine
