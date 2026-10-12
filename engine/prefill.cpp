// The prefill wave, in the two calls `decode_step.h` documents. Read that header first: it owns the
// contract this file implements, and `engine/prefill.h`'s header comment owns what a wave is for.
//
// The whole of this file's arithmetic is other people's kernels. Nothing here computes a value,
// chooses a reduction order, or converts a dtype: the wave is `submit_wave_body` per wave and
// `finish_token` per priced row, which is why a wave and a run of single-token steps cannot disagree
// — there is one implementation of the §07.1 order, in `decode_step.cpp`, and this file calls pieces
// of it.
//
// Two things here are worth reading closely.
//
// The `price_this_row` decision. Under `kLast` the wave prices only its last row, so a T-token prompt
// is one body submission per wave and ONE vocabulary D2H — §06.5's "only `logits[T-1]` is ever
// used", and the piece of §1.2's waste this package is allowed to remove. Under `kAll` every row is
// priced, which is the mode the byte-identity gate runs in and pays for it in syncs and host memory.
//
// The wave split. P4 gave the scratch `[T_w]` rows, so a prompt longer than the scratch is several
// waves; splitting lives here so no caller changed. Under `kAll` a wave prices its own rows before
// the next wave overwrites the tile, which is why pricing sits inside the loop rather than after it.
#include "engine/prefill.h"

#include <algorithm>
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

    // ---- the split (plan §2.1). The scratch holds `sc.wave_rows` rows, so a prompt longer than that
    // is more than one wave and the caller must not have to know. The plan's shape is "one wave for
    // T ≤ 1024, then waves of 512 at multiples of 64"; the engine cannot yet honour the first half of
    // that sentence, because attention's one-pass softmax stages MAXSC=512 lanes and refuses more
    // (`kernels_attn.cpp`), so the wave bound is `min(wave_rows, MAXSC)` in effect and a prompt past
    // the lane ceiling stops there with that message whatever this loop does. Splitting at
    // `wave_rows` = 512 is a multiple of 64, so the second half of the sentence is satisfied by
    // construction. Raising the lane ceiling is P7's, not this file's.
    const size_t wave_cap = (size_t)sc.wave_rows;
    for (size_t begin = 0; begin < tokens.size(); begin += wave_cap) {
        const int n = (int)std::min(wave_cap, tokens.size() - begin);
        const size_t base = out.rows.size();
        const int pos0 = st.prefix_len;   // §02.4: P of this wave's first token
        for (int t = 0; t < n; ++t) {
            PrefillRow row;
            row.index = (int)(base + t);
            row.token = tokens[begin + t];
            row.pos_before = pos0 + t;
            row.pos_after = pos0 + t + 1;   // the body advances one position per row
            out.rows.push_back(std::move(row));
        }

        // Submits the whole wave and waits for nothing, so the stream keeps filling. This is where the
        // position advances (`prefix_len = pos0 + n`) and where §08.6's invariants are re-checked.
        submit_wave_body(weights, st, sc, tokens.data() + begin, n);

        // The LAST row of the whole prompt is always priced — §06.5's `y0` has to come from somewhere
        // — and under `kAll` so is every other one. Pricing a row now reads that row out of the wave's
        // norm tile (`finish_token`'s row argument), which is what lets `kAll` reproduce the pre-wave
        // per-token prompt loop's launches without interleaving a finish into every body: row `t` here
        // holds exactly the bits the old loop's `sc.norm` held after its `t`-th body, because glue's
        // and the MLP's batched forms are bit-identical to the per-row ones (`prototype/glue/run.py
        // --batched`, `prototype/mlp_gemv/run.py --tokens`). That equivalence is the gate's `cmp`.
        if (mode == PrefillLogits::kAll) {
            for (int t = 0; t < n; ++t) {
                PrefillRow& row = out.rows[base + t];
                const StepResult sampled = finish_token(weights, st, sc, &row.logits, t);
                row.sampled = sampled.token;
                row.top_logit = sampled.top_logit;
            }
        } else if (begin + (size_t)n == tokens.size()) {
            PrefillRow& row = out.rows.back();
            const StepResult sampled = finish_token(weights, st, sc, &row.logits, n - 1);
            row.sampled = sampled.token;
            row.top_logit = sampled.top_logit;
        }
    }
    return out;
}

}  // namespace engine
