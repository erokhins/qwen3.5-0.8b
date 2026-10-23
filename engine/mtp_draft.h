#pragma once
// §09 — the MTP draft block, as a SIDE computation.
//
// WHAT THIS IS ALLOWED TO DO, and it is a narrow allowance. The block reads the finished step's
// final-norm hidden row and the token that step emitted, and it produces one proposal id. It writes
// nothing the target path consumes: its weights, its intermediates, its own 1-layer KV cache and its
// own logits row live in allocations this file owns, and every byte the target reads is untouched.
// That is what makes the admission test brutal rather than aspirational — `sampled.txt` and the §08.6
// STATEFP rows must come out BYTE-identical with the side computation on and off, and if they do not,
// the draft is not a side computation and this increment is wrong by construction.
//
// WHAT IT IS NOT: a verify body, an accept loop, or a rollback path. There is no 2-row body here, no
// snapshot of the mixer state, and no candidate row fed to the target. `RoundTracker` below decides
// which generation steps get a draft and whether a rejected round's cache lane is truncated; it
// changes nothing about the target's own state, which keeps running plain greedy.
//
// ONE DRAFT ROUND, submission by submission (ported from `prototype/mtp_draft`, whose gate is what
// bit-gated these stages against the pinned MTP reference — see that directory's `results.md`):
//
//  1  copy     e    <- E[y_next]                       one 2 KiB D2D, the mechanism a shipped one-row
//                                                     body uses for its embedding lookup when
//                                                     QWEN_COPY_FOLD is off
//  2  norm     e'   = RMSnorm(e)   pre_fc_norm_embedding     -> pair[   0:1024]
//  3  norm     h'   = RMSnorm(h)   pre_fc_norm_hidden        -> pair[1024:2048]
//     TWO launches because `glue_add_norm` deliberately does NOT row-index its weight, so batching
//     the halves would feed one weight to both halves. The CONCAT ORDER IS THESE TWO DESTINATION
//     POINTERS (§09.5: "eh" measures 0.642 accept against "he" 0.014), which is why they are written
//     out here rather than folded into one call.
//  4  proj     u    = pair · W_fcᵀ            [1,2048] -> [1,1024]
//  5  norm     u'   = RMSnorm(u)   input_layernorm
//  6  proj     qg   = u' · W_qᵀ               [4096] = 8 heads x (q 256 | gate 256), §05.1
//  7  proj     kg   = u' · W_kᵀ               [512]
//  8  proj     vg   = u' · W_vᵀ               [512]
//  9  attn     the block's OWN cache, at the drafted ABSOLUTE position, dense append slot
// 10  proj     attn_o = gout · W_oᵀ           [1024]   (gout is §05.6's gated out)
// 11  add+norm z1, u2 = post_attention_layernorm(u + attn_o)      — §03.2's pairing
// 12  mlp      act  = SwiGLU(u2 · [gate|up])  [3584], never reaches memory
// 13  mlp      mlp_o = act · W_downᵀ          [1024]
// 14  add+norm z2, x_d = mtp.norm(z1 + mlp_o)  — x_d IS the draft's output row (§09.2)
// ── then the proposal:
// 15  lm_head  logits = x_d · Eᵀ             the same tied table, one more full pass
// 16  D2H      one blocking read of the vocabulary, then §03.4's argmax via `argmax_greedy`
//
// So 14 submissions for the block (1 copy + 13 kernel launches) and 2 more for the head, 16 per
// round. `prototype/mtp_draft` counted 14 for the block too; its 14th was a gather KERNEL where this
// uses the shipped one-row body's D2D, so "14 submissions" is the same bill by a different mechanism
// and the engine's own line states which it paid.
//
// Activations stay BF16 end to end, which is what deletes the cast launches: `hip_bfloat16 -> float`
// is exact, and `prototype/mtp_draft`'s `stagecheck` arm MEASURED that the two stagings are
// bit-identical on this block's own operands (`fc_diff=0 act_diff=0 down_diff=0`, results.md §4)
// rather than citing the argument.
//
// WHY `h` IS THE ROW IT IS. §09.1's `h` is the hidden state the target's own lm_head saw, i.e. AFTER
// the final RMSNorm — A-12 as originally written said "before" and was falsified (post-norm cos 0.642
// against pre-norm 0.232, `specification/09-mtp-draft-layer.md` and `prototype/mtp_gatezero`). The
// engine already has that row: it is what `submit_token_body` leaves in `sc.norm`, so `h` costs no
// dump and no extra pass.

#include <cstdint>
#include <string>
#include <vector>

namespace engine {

class WeightStore;

namespace mtpdraft {

// ---- the census, in one place so the printed line cannot disagree with the code -------------------
constexpr int kBlockCopies = 1;       // the E[y_next] D2D
constexpr int kBlockLaunches = 13;    // 2 norms + fc + norm + q,k,v + attn + o + add_norm + gateup
                                      // + down + add_norm
constexpr int kHeadLaunches = 1;      // lm_head_full_xbf16
constexpr int kHeadCopies = 1;        // the one blocking logits D2H
constexpr int kSubmissionsPerRound =
    kBlockCopies + kBlockLaunches + kHeadLaunches + kHeadCopies;

// The same bill as `stageclock` counts it, which is NOT the same quantity, and the difference is one
// submission. The stage clock's rows are launches and device-to-device copies; its `kHostTail` row is
// noted with ZERO submissions precisely so that "launches" means the same thing on every row
// (`engine/stage_clock.h`). So the blocking vocabulary D2H — the 16th submission, and the one thing about
// a round that is guaranteed to synchronise — is billed nothing there and one here. A single constant
// shared by both instruments would have made one of the two wrong, which is why there are two named ones
// and the census line prints both spellings it is asked to.
constexpr long long kStageSubsPerRound =
    static_cast<long long>(kBlockCopies) + kBlockLaunches + kHeadLaunches;

// The block's 15 tensors, as the shard stores them (BF16, [out, in] for every matrix). Resolved once
// per run: `resolve()` re-reads each claim from the store and throws naming the FIRST mismatch, with
// rank and both dims stated, because a shape claim written as a guess is a bug with a compile-time
// guarantee it will be wrong eventually (`engine/weights_contract.py`'s reason for existing, and this
// is the use-site form of it that AGENTS.md prescribes for a kernel's own operands).
struct Weights {
    const uint16_t* fc = nullptr;        // mtp.fc.weight                      [1024,2048]
    const uint16_t* q = nullptr;         // …self_attn.q_proj.weight           [4096,1024]
    const uint16_t* k = nullptr;         // …self_attn.k_proj.weight           [512,1024]
    const uint16_t* v = nullptr;         // …self_attn.v_proj.weight           [512,1024]
    const uint16_t* o = nullptr;         // …self_attn.o_proj.weight           [1024,2048]
    const uint16_t* gate = nullptr;      // …mlp.gate_proj.weight              [3584,1024]
    const uint16_t* up = nullptr;        // …mlp.up_proj.weight                [3584,1024]
    const uint16_t* down = nullptr;      // …mlp.down_proj.weight              [1024,3584]
    const uint16_t* n_emb = nullptr;     // mtp.pre_fc_norm_embedding.weight   [1024]
    const uint16_t* n_hid = nullptr;     // mtp.pre_fc_norm_hidden.weight      [1024]
    const uint16_t* n_in = nullptr;      // …input_layernorm.weight            [1024]
    const uint16_t* n_paln = nullptr;    // …post_attention_layernorm.weight   [1024]
    const uint16_t* n_mtp = nullptr;     // mtp.norm.weight                    [1024]
    const uint16_t* n_q = nullptr;       // …self_attn.q_norm.weight           [256]
    const uint16_t* n_k = nullptr;       // …self_attn.k_norm.weight           [256]
    long long bytes = 0;                 // the sum of their nbytes; the census line prints it
};

// Resolves all 15 by engine name (§02.5: `mtp.layers.0.*` normalises to `mtp.layer0.*`, `mtp.*`
// passes through) and asserts each one's rank, dtype and both dims. Throws std::runtime_error naming
// the tensor and both the claim and the fact.
Weights resolve(const WeightStore& weights);

// The draft's own mutable state. NOT part of `RequestState`: the §08.6 state fingerprints have to keep
// meaning "model state", and a cache that only the side computation reads would move them for no
// reason (the same reasoning `DecodeScratch::a_part` gives for living in scratch).
struct State {
    // The block's 1-layer KV cache (§09.1), plane-major in the target's own layout
    // `[kv_head][cap][256]`, dense in DRAFTED STEPS rather than in positions — see `kernels::MtpAttnRow`
    // for why those two differ and `RoundTracker` for which steps append.
    uint16_t* kcache = nullptr;
    uint16_t* vcache = nullptr;
    int cap = 0;            // lanes allocated per KV head
    int nlanes = 0;         // lanes in use; a reject truncates it, which moves 0 bytes
    int lanes_max = 0;      // the high-water mark, for the census line

    // One arena of intermediates, sub-allocated from a table (`regions()` in the .cpp, so the layout
    // and the pointers cannot drift apart).
    uint16_t* arena = nullptr;
    long long arena_bytes = 0;
    uint16_t* e = nullptr;      // E[y_next]
    uint16_t* pair = nullptr;   // [e' || h'], the fc operand — THE CONCAT ORDER LIVES IN THESE WRITES
    uint16_t* u = nullptr;      // fc out
    uint16_t* nrm = nullptr;    // input_layernorm out
    uint16_t* qg = nullptr;     // q_proj out (q|gate interleaved per head, §05.1)
    uint16_t* kg = nullptr;
    uint16_t* vg = nullptr;
    uint16_t* aout = nullptr;   // §05.5 pre-gate attention out
    uint16_t* gout = nullptr;   // §05.6 gated out — o_proj's operand
    uint16_t* attn_o = nullptr;
    uint16_t* z1 = nullptr;     // residual after the mixer
    uint16_t* u2 = nullptr;     // post-attention-norm row
    uint16_t* act = nullptr;    // SwiGLU intermediate
    uint16_t* mlp_o = nullptr;
    uint16_t* z2 = nullptr;     // residual after the MLP
    uint16_t* xd = nullptr;     // x_d: mtp.norm's output, the head's operand

    float* logits = nullptr;    // [kVocab] F32, the draft's own logits row

    // Counters for the census line. `rounds` is what this state actually drafted; `skipped` counts the
    // rounds the caller asked for and this file refused (see `refusal()`).
    long long rounds = 0;
    long long refused = 0;
};

// `cap` lanes per KV head. The one-pass attention form can only serve `mtp_attn_max_lanes()` of them,
// so the usable ceiling is the smaller of the two and `usable_lanes()` says which.
State allocate_state(int cap);
void free_state(State& st);
void reset(State& st);                 // a fresh request: the cache is empty again
int usable_lanes(const State& st);

// A rejected round's lane is truncated, which is a length field and moves 0 bytes — the same argument
// `MTP-IMPL.md` §1 makes for the target's own KV on a reject ("§08.6 inv. 3 permits re-appending at
// that position, and inv. 4 makes the stale lane unreachable"). The lane is overwritten by the next
// round before anything could read it, and nothing reads past `nlanes`.
void truncate_lane(State& st);

// A round the side computation could not run, or "" when it can. The caller prints it and counts the
// proposal as SKIPPED with this reason — a proposal that was never made must not be able to read as a
// rejected proposal, and a silently shorter run is the empty-parse failure class again.
std::string refusal(const State& st);

// One draft round. `h_norm_row` is a DEVICE pointer to the BF16 final-norm row the step left in
// `sc.norm` (§09.1's `h`, post-final-norm); `y_next` the token that step emitted; `draft_pos` the
// ABSOLUTE position being drafted. Advances `st.nlanes` by one on success.
//
// Throws std::runtime_error on a HIP failure, on an out-of-range id, or on a cache the form cannot
// serve (check `refusal()` first: it is the polite version).
struct Proposal {
    int token = -1;         // §03.4's argmax over the draft's logits row (ties -> LOWEST id)
    float top_logit = 0.0f;
    int nlanes = 0;         // cache lanes after this round's append
    long long round = 0;    // 1-based, per State
};
Proposal draft_round(const WeightStore& weights, const Weights& w, State& st,
                     const uint16_t* h_norm_row, int y_next, int draft_pos,
                     std::vector<float>* logits_out = nullptr);

// ---- which generation steps get a draft -------------------------------------------------------------
//
// Two schedules, because the question "what is the accept rate" has two answers worth separating, and
// the difference between them is a measurement rather than a reading.
//
// kLoop reproduces `speccheck/mtp.py`'s `greedy_mtp(k=1)` round sequence on a plain greedy token
// stream. In that loop a round at position q drafts at q+1, and the next round is at q+2 when the draft
// was accepted and q+1 when it was not, so q advances by 1 or 2 and the draft cache holds only the
// ACCEPTED rounds' lanes (`greedy_mtp` calls `m.restore(msnap)` on a reject, which drops the lane).
// A plain greedy run emits exactly the token stream that loop would emit — greedy speculation cannot
// change the output, which is the whole point of §09.4 — so walking the same round sequence from the
// engine's own samples reproduces the loop's draft contexts exactly, without a verify body or a
// rollback: this is the engine's counterpart of `verify="steps"`, which is what the carried accept
// rate of 0.7329 was measured with (`prototype/mtp_gatezero/draft_survival.py`).
//
// kEvery drafts after EVERY step, i.e. it prices every draft context the stream contains instead of
// the subsequence the loop visits. Its cache therefore holds every position, which is more context than
// the loop gives a draft. So the gap between the two numbers is the size of that difference, measured,
// and neither one is silently substituted for the other.
enum class Schedule { kLoop, kEvery };
const char* schedule_name(Schedule s);
Schedule parse_schedule(const std::string& name);

class RoundTracker {
public:
    explicit RoundTracker(Schedule s) : sched(s) {}
    Schedule sched;

    struct After {
        bool alignment_broken = false;   // the pending proposal belonged to a different position
        bool adjudicated = false;        // this step's prediction settles the pending proposal
        bool accept = false;             // ... and it agrees with it
        int settled_pos = -1;            // the position whose prediction settled it
        int settled_proposal = -1;
        bool draft = false;              // submit a draft after this step
        bool drop_lane = false;          // a reject: truncate the draft cache by one lane
    };

    // Called once per generation step, with the position that step's row predicted (`pos_before`) and
    // the token it predicted. The pending proposal's own recorded position is compared against
    // `pred_pos` rather than assumed to line up: an index that quietly slid by one has produced false
    // 0/128 accept rates in this repo's history, so the join asserts and does not hope.
    After after_step(int pred_pos, int pred_token);

    // Called after a round was submitted, to record what to adjudicate later.
    void pending(int pred_pos, int proposal);

    bool has_pending() const { return have_pending; }
    int pending_pred_pos() const { return pending_pred_pos_; }
    int pending_proposal() const { return pending_proposal_; }
    void clear() { have_pending = false; pending_pred_pos_ = -1; pending_proposal_ = -1; }

private:
    bool have_pending = false;
    int pending_pred_pos_ = -1;
    int pending_proposal_ = -1;
};

}  // namespace mtpdraft
}  // namespace engine
