#pragma once
// §09 / M4 — ADAPTIVE SPECULATION DEPTH: the policy that decides, from what the rounds have already
// accepted, whether the next round runs at k=2, at k=1, or not at all (`--mtp-adaptive 1`, default OFF).
//
// WHY THIS EXISTS. The shipping default (`--mtp-accept 1 --mtp-k 2`) is a win wherever acceptance is
// decent and a pure tax where it is not. `engine/performance-status.md` §9's four paired rows, in its own
// numbers: a = 0.5269 → +17.87 %, a = 0.529 → +23.44 %, a = 0.400 → +12.88 %, a = 0.0323 → **−28.28 %**.
// A round costs 1.83–1.97 of a plain step, so a round that accepts nothing spends two steps to emit one
// token. The default is therefore only safe if acceptance is not near zero, and the engine cannot know
// that before the run — the DRAFT cannot know it, the flags cannot know it, but the rounds do, one round
// at a time. This file is the rule that spends what the rounds already know. llama.cpp has no such policy
// (`common/speculative.cpp` drafts at a fixed depth forever); this is about making OUR default defensible,
// not about parity.
//
// WHAT IT MAY AND MAY NOT TOUCH. The policy changes GEOMETRY — how many candidates the next body
// speculates on, or whether the loop speculates at all. It never touches a verdict: every token any of the
// three shapes emits is the target's own argmax or a candidate the device proved equal to one
// (`engine/mtp_accept.h`'s accept rule), so losslessness is a property of the round, not of the depth.
// That is the whole reason a depth policy is admissible at all, and it is what
// `engine/gate.py` section (m) adjudicates on both sides of the flag.
//
// WHY NO NEW SYNC. The policy is fed from the round's own verdict readback — `mtp_accept.cpp` already
// blocks on one `hipMemcpy` of `kVerdictWordsFor(rows)` ints per round, and `n_accepted` and `emitted`
// fall out of it. The decision for round N+1 is therefore host arithmetic on values the host already
// holds: zero added device→host traffic, zero added waits. A policy that needed to ASK the device
// something the round did not already answer would have been refused rather than written.
//
// THE ESTIMATOR, in one sentence: the mean tokens per step-equivalent over the last `kWindowRounds`
// rounds, where a round's step-equivalent is the measured cost of ITS OWN depth (so a mixed window is a
// fair comparison, not a mixture). It is a ratio of two tallies, not a mean of ratios, so it is exactly
// "what this window paid per step" rather than an average of per-round rates. `prototype/mtp_accept/
// ADAPTIVE.md` carries the full arithmetic — the cost bands these constants are midpoints of, the σ
// separations that sized the observation windows, and the four §9 rows this rule classifies correctly.
//
// NO MILLISECONDS ANYWHERE IN THIS FILE, same rule as `engine/mtp_accept.h`. The cost constants below are
// RATIOS of a round to a step, measured by the main agent and cited by section; they are not timings and
// they are not this file's to move.
//
// WHAT THIS FILE IS NOT: not a scheduler, not a per-request priority scheme, not a k=3 ladder (k=3 is not
// implemented — `engine/mtp_accept.h`'s `kMaxK` is 2 and this file does not invent a third depth), and
// not a way to make a slow content fast. It can only stop paying for speculation that is not paying.

#include <string>

namespace engine {
namespace mtpadapt {

// ---- the measured cost of one round, in units of one plain decode step -----------------------------
// Both are MIDPOINTS of measured bands, and the bands are written out because a midpoint with no band
// beside it is a taste. A cost is a function of the BODY WIDTH, not of the policy: a 3-row body is a k=2
// round and a 2-row body is a k=1 round, whatever asked for it.
//
//   k=1 (2-row body): `prototype/mtp_accept/ROUND-TAX.md` §"The pre-registered adopt line" reads
//     round/step 1.472 and 1.481 at templated 4,152 lanes (1.530 with `--mtp-dense --mtp-prompt-lanes`);
//     `prototype/mtp_accept/CORPUS-WALL.md` reads 1.686 and 1.701 at the corpus's natural (short) width.
//     Band [1.47, 1.70] → midpoint 1.585, used as 1.58.
//   k=2 (3-row body): `engine/performance-status.md` §9's own round/step column: 1.829 (templated 4,152),
//     1.841 (raw 4,142), 1.889 (corpus case at 4,032), 1.971 (corpus 12-case population).
//     Band [1.829, 1.971] → midpoint 1.90.
//
// The cost is what the estimator divides by, so the direction of its error is worth stating: billing a
// round CHEAPER than it is makes the estimate read BETTER than reality and delays a demotion (the unsafe
// direction), while billing it dearer demotes a little early (the safe direction, since the fallback is a
// plain step and a plain step is the shipping path). The band midpoints sit within ±0.06 of either end of
// the k=1 band and ±0.07 of either end of the k=2 band, and `prototype/mtp_accept/ADAPTIVE.md` §3 shows
// every §9 row still classifying correctly at the pessimistic end of both bands.
constexpr double kRoundCostStepsK1 = 1.58;
constexpr double kRoundCostStepsK2 = 1.90;

// The cost of one round at depth `k` (1 or 2). `k == 0` is a plain step and costs exactly one step.
constexpr double round_cost_steps(int k) { return k >= 2 ? kRoundCostStepsK2 : kRoundCostStepsK1; }

// ---- the window and the observation minimums --------------------------------------------------------
// `kWindowRounds` bounds the estimate literally: a round older than the window is not in the tally at
// all, so the policy cannot be held hostage by the first hundred tokens of a long request.
//
// The two minimums are sample sizes, and they are sized by σ, not by feel. Per-round emitted tokens are
// bounded (1..3 at k=2, 1..2 at k=1), so the standard error of the window mean is σ/√N divided by the
// round cost. Taking the §9 rows' own variances (templated k=2: p1 = 0.729, p2 = 0.529 → per-round σ ≈
// 0.86 tokens; k=1 at p1 = 0.32 → σ ≈ 0.47):
//
//   DEMOTE (k=2 → k=1) at N = 16: se(est) = 0.86 / √16 / 1.90 = 0.11. The weakest WINNING §9 row is raw
//     long prose at est = (1 + 0.677 + 0.400) / 1.90 = 1.093, which is (1.093 − 0.95) / 0.11 = 1.3 σ above
//     the line — thin, which is why the demotion is cheap and reversible-in-cost (k=1 gives 1.06 on the
//     same content, a −3 % mistake, not a −28 % one).
//   ABANDON (k=1 → plain) at N = 24: se(est) = 0.47 / √24 / 1.58 = 0.06. The weakest winning k=1 row is
//     the same content at est = (1 + 0.677) / 1.58 = 1.061, i.e. (1.061 − 0.90) / 0.06 = 2.7 σ of clearance
//     before a sticky, cannot-be-undone decision. That is the whole argument for the second, larger
//     minimum: the irreversible step gets the bigger sample.
constexpr int kWindowRounds = 32;
constexpr int kMinObserveRounds = 16;
constexpr int kMinObserveBeforeAbandon = 24;

// ---- the thresholds, and the break-even they come from ----------------------------------------------
// A plain step emits exactly 1 token per step, so a round is worth running iff its tokens-per-step-
// equivalent is above 1.0. Break-even at k=2 therefore needs `1 + p1 + p2 ≥ 1.90`, i.e. an expected 0.90
// accepted candidates out of 2; at k=1 it needs `1 + p1 ≥ 1.58`, i.e. p1 ≥ 0.58. Both thresholds sit
// BELOW 1.0 on purpose, and the gap is the hysteresis:
//
//   `kDemoteEst = 0.95` — demote only when the round is *measurably* worse than a plain step (>5 %), not
//     merely at break-even, so noise around 1.0 cannot flip it. One-way: there is NO re-promotion, so a
//     request's depth sequence is monotone (2 → 1 → plain) and has at most two changes. That is what makes
//     round-to-round oscillation structurally impossible rather than merely discouraged, and it is why no
//     second threshold (a re-promote line above the demote line) is needed or present.
//   `kAbandonEst = 0.90` — the sticky decision needs a wider margin than the cheap one. Abandonment is
//     final for the request (`abandoned()` stays true), so it is taken only when speculation is at least
//     10 % below break-even, not when it is borderline.
constexpr double kDemoteEst = 0.95;
constexpr double kAbandonEst = 0.90;

// The three shapes. `kPlain` is not a depth: it means "stop speculating, run plain one-row steps for the
// rest of the request", i.e. the loop's existing no-proposal branch, which is the pre-MTP shipping path.
constexpr int kPlain = 0;

// One policy decision, as the driver prints it. `fired == false` means "no change", and the driver prints
// nothing in that case: an event line is an event, not a heartbeat.
struct Change {
    bool fired = false;
    int from = 0;
    int to = 0;
    double est = 0.0;
    int seen = 0;       // rounds in the window when the decision was taken
};

// The tracker: one per request, fed once per round. Header-only and allocation-free by construction — it
// is two fixed arrays and four scalars, so the policy cannot be the reason a round allocates.
class Tracker {
  public:
    // `start_k` is the CEILING this run may use, which is the requested `--mtp-k`: `--mtp-k 1` plus adaptive
    // starts at 1 and can only abandon. The side buffers are allocated for this same value
    // (`decode_run.cpp`), so no round is ever wider than the run asked for.
    explicit Tracker(int start_k) : k_(start_k), k_ceiling_(start_k) {}

    // Fed after EVERY round, with what that round emitted and the depth it ran at. Returns the decision,
    // if any. Sticky once abandoned: a plain step is not fed here at all (there is no round), so nothing
    // can walk the estimate back up.
    Change note(int k_round, int emitted) {
        rounds_[k_round < 0 || k_round > 2 ? 0 : k_round]++;
        if (k_ == kPlain) return Change();
        push(emitted, round_cost_steps(k_round));
        if (count_ < min_observe()) return Change();
        const double e = est();
        int to = k_;
        if (k_ >= 2 && e < kDemoteEst) {
            to = k_ - 1;                       // 2 → 1
        } else if (k_ == 1 && e < kAbandonEst) {
            to = kPlain;                       // 1 → plain, and final
        }
        if (to == k_) return Change();
        const Change c{true, k_, to, e, count_};
        k_ = to;
        changes_++;
        // THE WINDOW CLEARS ON A CHANGE, and that is the second half of the hysteresis. Every round in a
        // window therefore ran under ONE geometry, so a decision never adjudicates the old depth's rounds
        // as evidence against the new one — and because the minimum sample must be rebuilt from scratch,
        // two changes can never be closer together than `min_observe()` rounds. With no re-promotion that
        // bounds the whole request at two changes.
        head_ = 0;
        count_ = 0;
        return c;
    }

    int k() const { return k_; }
    int ceiling() const { return k_ceiling_; }
    bool abandoned() const { return k_ == kPlain; }
    int changes() const { return changes_; }
    int seen() const { return count_; }
    long long rounds_at(int k) const { return k < 0 || k > 2 ? 0 : rounds_[k]; }

    // The window estimate: tokens emitted per step-equivalent. Zero when the window is empty, which is
    // also what "no evidence yet" must print — a policy that invented a starting estimate would be judging
    // rounds it never saw.
    double est() const {
        double emitted = 0.0, cost = 0.0;
        tally(emitted, cost);
        return cost > 0.0 ? emitted / cost : 0.0;
    }

    // The config line, printed once when the policy is armed. Every threshold the policy will apply is on
    // this line so the run's own log states the rule that produced its bits — and so `engine/gate.py`
    // parses its arithmetic off the engine instead of typing a copy of these constants (a gate that
    // re-types a threshold drifts from the thing it judges; `AGENTS.md`'s `--attn-tile` rule).
    std::string config_line() const {
        return "on=1 k_start=" + std::to_string(k_ceiling_) + " floor=1 ceiling=" +
               std::to_string(k_ceiling_) + " plain=" + std::to_string(kPlain) +
               " window=" + std::to_string(kWindowRounds) +
               " min_observe=" + std::to_string(kMinObserveRounds) +
               " min_observe_abandon=" + std::to_string(kMinObserveBeforeAbandon) +
               " demote_est=" + fixed3(kDemoteEst) + " abandon_est=" + fixed3(kAbandonEst) +
               " cost_k1=" + fixed2(kRoundCostStepsK1) + " cost_k2=" + fixed2(kRoundCostStepsK2) +
               " re_promote=0";
    }

    // The run's attribution line. `plain_steps` comes from the driver's own counter (`DECWALL plain=`),
    // because a plain step is not a round and the policy does not invent a second count of it.
    std::string describe(int plain_steps) const {
        double e_sum = 0.0, c_sum = 0.0;
        tally(e_sum, c_sum);
        return std::string("rounds_k2=") + std::to_string(rounds_[2]) + " rounds_k1=" +
               std::to_string(rounds_[1]) + " plain=" + std::to_string(plain_steps) +
               " changes=" + std::to_string(changes_) + " final_k=" + std::to_string(k_) +
               " ceiling=" + std::to_string(k_ceiling_) + " window_rounds=" + std::to_string(count_) +
               " window_emitted=" + fixed3(e_sum) + " window_cost_steps=" + fixed3(c_sum) +
               " est=" + fixed3(est()) +
               " est_all=" + fixed3(all_emitted_ / all_cost_);
    }

  private:
    int min_observe() const { return k_ >= 2 ? kMinObserveRounds : kMinObserveBeforeAbandon; }

    void push(double emitted, double cost) {
        emit_[head_] = emitted;
        cost_[head_] = cost;
        head_ = (head_ + 1) % kWindowRounds;
        if (count_ < kWindowRounds) count_++;
        all_emitted_ += emitted;
        all_cost_ += cost;
    }

    void tally(double& emitted, double& cost) const {
        emitted = 0.0;
        cost = 0.0;
        for (int i = 0; i < count_; ++i) {
            emitted += emit_[i];
            cost += cost_[i];
        }
    }

    static std::string fixed2(double v) { return fixed(v, 2); }
    static std::string fixed3(double v) { return fixed(v, 3); }
    static std::string fixed(double v, int places) {
        long long scale = 1;
        for (int i = 0; i < places; ++i) scale *= 10;
        long long scaled = static_cast<long long>(v * static_cast<double>(scale) + (v < 0 ? -0.5 : 0.5));
        std::string s = std::to_string(scaled < 0 ? -scaled : scaled);
        while (static_cast<int>(s.size()) <= places) s = "0" + s;
        return (v < 0 ? "-" : "") + s.substr(0, s.size() - places) + "." +
               s.substr(s.size() - places);
    }

    double emit_[kWindowRounds] = {0.0};
    double cost_[kWindowRounds] = {0.0};
    int head_ = 0;
    int count_ = 0;
    int k_ = 0;
    int k_ceiling_ = 0;
    int changes_ = 0;
    long long rounds_[3] = {0, 0, 0};   // rounds fed to `note()`, indexed by the depth they ran at
    double all_emitted_ = 0.0;          // never cleared, unlike the window: the run's own realised rate
    double all_cost_ = 0.0;
};

}  // namespace mtpadapt
}  // namespace engine
