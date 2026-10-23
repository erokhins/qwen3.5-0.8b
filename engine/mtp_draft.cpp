// §09 — the MTP draft block, submitted. Read `engine/mtp_draft.h` first: it owns the submission list,
// the side-computation contract, and what each stage is for.
//
// What this file adds over the list is the plumbing: the arena layout, the weight claims, and one
// refusal path that a caller can count. It contains no arithmetic of its own, exactly like
// `decode_step.cpp` — every value comes from a kernel that has its own gate against a pinned reference.
//
// PROVENANCE OF THE STAGES. Every launch here is a launcher `engine/kernels.h` already declares and the
// shipping path already uses, except `kernels::mtp_attn_row`. The arithmetic of the block was gated in
// `prototype/mtp_draft/` before it was allowed here: per-observable bands imported from the ops that
// measured them, the mirror witnessed bit-identical against `speccheck.mtp.MTPDraft`, the attention
// transcription witnessed bit-identical against the shipped `attn_core`, and nine derived negative
// controls caught by a check rather than by reading (`prototype/mtp_draft/results.md` §2–§5). Porting
// those launchers rather than rewriting them is the point: a rewrite would have had no witness.
#include "engine/mtp_draft.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "engine/decode_step.h"
#include "engine/kernels.h"
#include "engine/loader.h"
#include "engine/stage_clock.h"
#include "engine/state.h"

#define MTP_CK(x)                                                                     \
    do {                                                                              \
        hipError_t e_ = (x);                                                          \
        if (e_ != hipSuccess) {                                                       \
            throw std::runtime_error(std::string("hip: ") + hipGetErrorString(e_) +   \
                                     " at " __FILE__ ":" + #x);                       \
        }                                                                             \
    } while (0)

namespace engine::mtpdraft {
namespace {

// The two names the target path already uses for the shared table (§01.1: there is no second table, so
// `lm_head.weight` is an alias of `lm.embed_tokens.weight` and one blob answers both). §09.1 says the
// draft shares it too, which is why the draft's cost is a table PASS and not a new 485 MiB blob.
constexpr const char* kEmbedding = "lm.embed_tokens.weight";
constexpr const char* kHead = "lm_head.weight";

[[noreturn]] void die(const std::string& m) { throw std::runtime_error("mtp_draft: " + m); }

// The round's own submission tally, and the reason it is an object rather than a bare marker call at
// each site.
//
// `stageclock` bills the `mtp` row from a DECLARED per-step total (`Clock::set_plan_mtp`), because the
// clock has nothing to derive a side computation's bill from. A declaration that drifts from the sites is
// the failure this repo keeps recording — a plan that agrees with itself while the dispatch does something
// else — so the sites themselves are counted here and required to add up to the same number the census
// line prints. It is a runtime check with a name, not a comment: it runs on every round, armed or not, so
// a launch added, dropped or double-marked fails the run that did it rather than a later gate.
//
// Marking a site is two things at once and the helper does both: tell the clock which family the
// submissions that FOLLOW belong to (`engine/stage_clock.h`: the marker precedes the submission, and
// getting that order wrong silently charges every segment to its predecessor), and add to the tally the
// round is then checked against. A site that launched without being marked shows up as a tally BELOW the
// declared total; a site marked twice shows up as one ABOVE. Both name themselves at the round that did
// it, on the host, before any number is printed.
class RoundTally {
public:
    void mark(stageclock::Stage s, int submissions) {
        stageclock::note(s, submissions);
        subs_ += submissions;
        ++sites_;
    }

    // Called at the end of a round, after the D2H (which is marked with zero submissions, the way
    // `kHostTail` is, so that "launches" means one thing on every row of the report).
    void check(const char* what, int expected) const {
        if (subs_ == expected) return;
        std::ostringstream os;
        os << what << ": marked " << subs_ << " billed submissions across " << sites_
           << " site(s), but the declared bill is " << expected << " (for a §09 round: block "
           << kBlockCopies << " copies + " << kBlockLaunches << " launches, head " << kHeadLaunches
           << " launch; the " << kHeadCopies << " blocking D2H is billed 0 here and 1 on the census"
              " line — see kStageSubsPerRound; for a lane append: kLaneStageSubs)."
              " A launch added or dropped inside the unit without the plan being told: the stage clock's"
              " `mtp` row and the `MTPDRAFT` census would both still agree with themselves.";
        die(os.str());
    }

private:
    int subs_ = 0;
    int sites_ = 0;
};

// One arena region. The layout and the named pointers are both derived from this table, so a resize
// cannot leave a short buffer behind and `arena_bytes` is not a typed-in total.
constexpr int kHiddenW = kHidden;                 // 1024
constexpr int kQgW = kQprojRows;                 // 4096 = 8 heads x (256 q | 256 gate)
constexpr int kKvW = kKvProjRows;                // 512
constexpr int kQwW = kQwidth;                    // 2048 = the attention block's output width
constexpr int kFfW = kIntermediate;              // 3584

struct Region {
    const char* name;
    int n;
    uint16_t* State::*slot;
};
constexpr Region kRegions[] = {
    {"e", kHiddenW, &State::e},     {"pair", 2 * kHiddenW, &State::pair},
    {"u", kHiddenW, &State::u},     {"nrm", kHiddenW, &State::nrm},
    {"qg", kQgW, &State::qg},       {"kg", kKvW, &State::kg},
    {"vg", kKvW, &State::vg},       {"aout", kQwW, &State::aout},
    {"gout", kQwW, &State::gout},   {"attn_o", kHiddenW, &State::attn_o},
    {"z1", kHiddenW, &State::z1},   {"u2", kHiddenW, &State::u2},
    {"act", kFfW, &State::act},     {"mlp_o", kHiddenW, &State::mlp_o},
    {"z2", kHiddenW, &State::z2},   {"xd", kHiddenW, &State::xd},
};

// The rank-and-shape claim for one block tensor: what the kernel about to read it needs, stated out
// loud. `rank` is a parameter rather than something inferred from `shape.size()` because
// `linear_attn.conv1d.weight` is the tensor this repo got wrong by having no rank field, and the fix
// that held was saying the rank (AGENTS.md).
void check_shape(const Weight& wt, int rank, int d0, int d1) {
    if ((int)wt.shape.size() != rank) {
        std::ostringstream os;
        os << wt.name << ": rank " << wt.shape.size() << " " << wt.shape_string()
           << ", but the kernel that reads it needs rank " << rank << " [" << d0 << "," << d1 << "]";
        die(os.str());
    }
    if (wt.dtype != Dtype::BF16) die(std::string(wt.name) + ": stored as " + dtype_name(wt.dtype) +
                                     ", and these kernels take BF16 weights");
    if (wt.shape[0] != d0 || (rank == 2 && wt.shape[1] != d1)) {
        std::ostringstream os;
        os << wt.name << ": " << wt.shape_string() << ", but the kernel that reads it needs [" << d0
           << "," << d1 << "]";
        die(os.str());
    }
}

const uint16_t* want(const WeightStore& ws, const char* name, int rank, int d0, int d1) {
    const Weight& wt = ws.at(name);        // throws and names the tensor if it is not resident
    check_shape(wt, rank, d0, d1);
    return (const uint16_t*)wt.device;
}

// The guards every block submission needs, in ONE place so a lane append cannot be served a row or an id
// that a round would have refused. `what` names the caller in the message, because a refusal thrown from
// a shared helper and blamed on the wrong unit sends the reader to the wrong call site.
void check_operands(const char* what, const State& st, const uint16_t* h_norm_row, int y_next) {
    if (h_norm_row == nullptr) die(std::string(what) + ": h_norm_row is null");
    const std::string why = refusal(st);
    if (!why.empty()) die(std::string(what) + ": " + why);
    if (y_next < 0 || y_next >= kVocab)
        die(std::string(what) + ": y_next=" + std::to_string(y_next) + " is outside the tied table");
}

// The block's stages 1-9 — embedding copy, the two pre-FC norms, `fc`, the input norm, q/k/v, and the
// attention row that APPENDS this step's lane. Shared by `draft_round` and `draft_lane` on purpose: two
// spellings of the same nine submissions is how a lane and a round drift apart, and the drift would show
// up as a draft cache whose lanes are not the same objects depending on which call made them.
//
// `tally` is the caller's, so each unit checks its own bill; `draft_round` continues with stages 10-16
// afterwards and `draft_lane` stops here.
void submit_block_prefix(const WeightStore& weights, const Weights& w, State& st,
                         const uint16_t* h_norm_row, int y_next, int draft_pos,
                         RoundTally& tally) {
    const uint16_t* table = (const uint16_t*)weights.at(kEmbedding).device;

    // Every site below is marked into the `mtp` stage family — see `RoundTally`: the marker names the
    // family the FOLLOWING submissions belong to, and the tally is what the unit is checked against at
    // the end. The marks are in the order of the block, which is §09's order as spelled in the header;
    // nothing here reorders a launch, so the clock's segments tile the round the way `decode_step.cpp`'s
    // markers tile a step.

    // 1  E[y_next]. A 2 KiB D2D — the mechanism `submit_wave_body` uses for a one-row body when the
    // copy fold is off, so this is the engine's own embedding lookup rather than a third shape.
    tally.mark(stageclock::kMtp, 1);
    MTP_CK(hipMemcpyAsync(st.e, table + (size_t)y_next * kHidden, (size_t)kHidden * 2,
                          hipMemcpyDeviceToDevice, nullptr));

    // 2-3  the two pre-FC norms, straight into the halves of the fc operand. `pair[0:1024]` is the
    // EMBEDDING half and `pair[1024:2048]` the HIDDEN half: that is §09.2's `[emb || hidden]` order
    // against `W_fc [1024, 2048]`, and `speccheck/mtp.py`'s `fc_order` knob measures it (0.642 accept
    // for "eh" against 0.014 for "he", §09.5). An order bug here is invisible to any max-ULP check on
    // `u` and blindingly obvious to the pinned golden, which is why it is spelled twice.
    tally.mark(stageclock::kMtp, 1);
    kernels::glue_add_norm(st.e, nullptr, w.n_emb, nullptr, st.pair);
    tally.mark(stageclock::kMtp, 1);
    kernels::glue_add_norm(h_norm_row, nullptr, w.n_hid, nullptr, st.pair + kHidden);

    // 4  the fusion fc: [1, 2048] -> [1, 1024]
    tally.mark(stageclock::kMtp, 1);
    kernels::proj_gemv_xbf16(w.fc, st.pair, st.u, 1024, 2048);

    // 5-8  the §05 block's operands
    tally.mark(stageclock::kMtp, 1);
    kernels::glue_add_norm(st.u, nullptr, w.n_in, nullptr, st.nrm);
    tally.mark(stageclock::kMtp, 1);
    kernels::proj_gemv_xbf16(w.q, st.nrm, st.qg, kQprojRows, kHidden);
    tally.mark(stageclock::kMtp, 1);
    kernels::proj_gemv_xbf16(w.k, st.nrm, st.kg, kKvProjRows, kHidden);
    tally.mark(stageclock::kMtp, 1);
    kernels::proj_gemv_xbf16(w.v, st.nrm, st.vg, kKvProjRows, kHidden);

    // 9  the attention row, over the block's OWN cache, at the drafted ABSOLUTE position. The append
    // slot is the dense lane index; `draft_pos` is only what rope sees.
    kernels::MtpAttnRow ar;
    ar.qg = st.qg; ar.kg = st.kg; ar.vg = st.vg;
    ar.qn = w.n_q; ar.kn = w.n_k;
    ar.kcache = st.kcache; ar.vcache = st.vcache;
    ar.aout = st.aout; ar.gout = st.gout;
    ar.cap = st.cap;
    ar.nlanes = st.nlanes + 1;
    ar.self_lane = st.nlanes;
    ar.pos = draft_pos + st.pos_shift;   // `pos_shift` shifts every lane uniformly — see State
    tally.mark(stageclock::kMtp, 1);
    kernels::mtp_attn_row(ar);
    st.nlanes += 1;
    if (st.nlanes > st.lanes_max) st.lanes_max = st.nlanes;
}

}  // namespace

Weights resolve(const WeightStore& weights) {
    Weights w;
    w.fc = want(weights, "mtp.fc.weight", 2, 1024, 2048);
    w.q = want(weights, "mtp.layer0.self_attn.q_proj.weight", 2, kQprojRows, kHidden);
    w.k = want(weights, "mtp.layer0.self_attn.k_proj.weight", 2, kKvProjRows, kHidden);
    w.v = want(weights, "mtp.layer0.self_attn.v_proj.weight", 2, kKvProjRows, kHidden);
    w.o = want(weights, "mtp.layer0.self_attn.o_proj.weight", 2, kHidden, kQwidth);
    w.gate = want(weights, "mtp.layer0.mlp.gate_proj.weight", 2, kIntermediate, kHidden);
    w.up = want(weights, "mtp.layer0.mlp.up_proj.weight", 2, kIntermediate, kHidden);
    w.down = want(weights, "mtp.layer0.mlp.down_proj.weight", 2, kHidden, kIntermediate);
    w.n_emb = want(weights, "mtp.pre_fc_norm_embedding.weight", 1, kHidden, 0);
    w.n_hid = want(weights, "mtp.pre_fc_norm_hidden.weight", 1, kHidden, 0);
    w.n_in = want(weights, "mtp.layer0.input_layernorm.weight", 1, kHidden, 0);
    w.n_paln = want(weights, "mtp.layer0.post_attention_layernorm.weight", 1, kHidden, 0);
    w.n_mtp = want(weights, "mtp.norm.weight", 1, kHidden, 0);
    w.n_q = want(weights, "mtp.layer0.self_attn.q_norm.weight", 1, kHeadDim, 0);
    w.n_k = want(weights, "mtp.layer0.self_attn.k_norm.weight", 1, kHeadDim, 0);
    // The census: sum the nbytes the store actually holds for those 15 names, so the printed byte
    // total is read off the shard and not off this file's opinion of it.
    static const char* kNames[] = {
        "mtp.fc.weight",
        "mtp.layer0.self_attn.q_proj.weight", "mtp.layer0.self_attn.k_proj.weight",
        "mtp.layer0.self_attn.v_proj.weight", "mtp.layer0.self_attn.o_proj.weight",
        "mtp.layer0.mlp.gate_proj.weight",    "mtp.layer0.mlp.up_proj.weight",
        "mtp.layer0.mlp.down_proj.weight",    "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",      "mtp.layer0.input_layernorm.weight",
        "mtp.layer0.post_attention_layernorm.weight", "mtp.norm.weight",
        "mtp.layer0.self_attn.q_norm.weight", "mtp.layer0.self_attn.k_norm.weight"};
    for (const char* n : kNames) w.bytes += (long long)weights.at(n).nbytes;
    return w;
}

State allocate_state(int cap) {
    if (cap < 1) die("allocate_state: cap must be at least one lane");
    State st;
    st.cap = cap;
    long long total = 0;
    for (const Region& r : kRegions) total += r.n;
    st.arena_bytes = total * 2;
    MTP_CK(hipMalloc((void**)&st.arena, (size_t)st.arena_bytes));
    long long off = 0;
    for (const Region& r : kRegions) {
        st.*(r.slot) = st.arena + off;
        off += r.n;
    }
    const size_t plane = (size_t)cap * kHeadDim;
    MTP_CK(hipMalloc((void**)&st.kcache, 2 * plane * 2));
    MTP_CK(hipMalloc((void**)&st.vcache, 2 * plane * 2));
    MTP_CK(hipMemset(st.kcache, 0, 2 * plane * 2));
    MTP_CK(hipMemset(st.vcache, 0, 2 * plane * 2));
    MTP_CK(hipMalloc((void**)&st.logits, (size_t)kVocab * sizeof(float)));
    st.lanes_max = 0;
    return st;
}

void free_state(State& st) {
    // The HIP headers mark hipFree nodiscard; a teardown failure surfaces on the next ck() rather
    // than being silently swallowed, which is what `prototype/mtp_draft` does for the same reason.
    (void)hipFree(st.arena);
    (void)hipFree(st.kcache);
    (void)hipFree(st.vcache);
    (void)hipFree(st.logits);
    st = State{};
}

void reset(State& st) {
    st.nlanes = 0;
    st.lanes_max = 0;
    st.rounds = 0;
    st.refused = 0;
    st.lane_appends = 0;
}

void truncate_lane(State& st) {
    // A length field, not a copy: the rejected round's lane is overwritten by the next round's append
    // before anything can read it, exactly the argument `MTP-IMPL.md` §1 makes for the target's own KV.
    if (st.nlanes > 0) --st.nlanes;
}

int usable_lanes(const State& st) {
    const int form = kernels::mtp_attn_max_lanes();
    return st.cap < form ? st.cap : form;
}

std::string refusal(const State& st) {
    if (st.nlanes + 1 > usable_lanes(st)) {
        std::ostringstream os;
        os << "draft cache would hold " << (st.nlanes + 1) << " lanes and the ceiling is "
           << usable_lanes(st) << " (cap=" << st.cap
           << ", one-pass form MAXSC=" << kernels::mtp_attn_max_lanes() << ")";
        return os.str();
    }
    return "";
}

Proposal draft_round(const WeightStore& weights, const Weights& w, State& st,
                     const uint16_t* h_norm_row, int y_next, int draft_pos,
                     std::vector<float>* logits_out) {
    check_operands("draft_round", st, h_norm_row, y_next);

    RoundTally tally;
    submit_block_prefix(weights, w, st, h_norm_row, y_next, draft_pos, tally);

    // 10-11  o_proj, then §03.2's add fused with the next norm
    tally.mark(stageclock::kMtp, 1);
    kernels::proj_gemv_xbf16(w.o, st.gout, st.attn_o, kHidden, kQwidth);
    tally.mark(stageclock::kMtp, 1);
    kernels::glue_add_norm(st.u, st.attn_o, w.n_paln, st.z1, st.u2);

    // 12-14  the MLP (the target's shapes verbatim, so the gated pair applies with no new kernel),
    // then the second add fused with `mtp.norm`, whose output IS x_d (§09.2).
    const kernels::MlpGateMap map = kernels::resolve_mlp_gate_map(kernels::MlpGateMap::kAuto, 1);
    tally.mark(stageclock::kMtp, 1);
    kernels::mlp_gateup_xbf16(w.gate, w.up, st.u2, st.act, kIntermediate, kHidden, map);
    tally.mark(stageclock::kMtp, 1);
    kernels::mlp_down_xbf16(w.down, st.act, st.mlp_o, kHidden, kIntermediate);
    tally.mark(stageclock::kMtp, 1);
    kernels::glue_add_norm(st.z1, st.mlp_o, w.n_mtp, st.z2, st.xd);

    // 15  the tied head over x_d — a second full pass over the same table, which is what `c` is made
    // of (§09's largest single term). Read-only on the table, and the ONLY writer of `st.logits`.
    tally.mark(stageclock::kMtp, 1);
    kernels::lm_head_full_xbf16((const uint16_t*)weights.at(kHead).device, st.xd, st.logits, kVocab);

    // 16  one blocking D2H and §03.4's argmax — the same function `finish_token` uses, so the draft's
    // proposal cannot be produced by a second standard. Marked with ZERO billed submissions: a D2H is not
    // a launch, which is exactly how `kHostTail` is handled in a step, and the census line still counts it
    // (see `kStageSubsPerRound`).
    tally.mark(stageclock::kMtp, 0);
    Proposal p;
    std::vector<float> host(kVocab);
    MTP_CK(hipMemcpy(host.data(), st.logits, kVocab * sizeof(float), hipMemcpyDeviceToHost));
    tally.check("draft_round", kStageSubsPerRound);
    p.token = argmax_greedy(host.data(), kVocab);
    p.top_logit = host[p.token];
    p.nlanes = st.nlanes;
    p.round = ++st.rounds;
    if (logits_out) *logits_out = std::move(host);
    return p;
}

void draft_lane(const WeightStore& weights, const Weights& w, State& st,
                const uint16_t* h_norm_row, int y_next, int pos) {
    check_operands("draft_lane", st, h_norm_row, y_next);
    RoundTally tally;
    submit_block_prefix(weights, w, st, h_norm_row, y_next, pos, tally);
    // Stops at the append: no o_proj, no MLP, no head pass, no vocabulary D2H. The lane's K/V is a
    // function of its own (h, y) pair only, so nothing after stage 9 can change a lane that is already
    // in the cache — which is the whole reason this unit is allowed to stop early.
    tally.check("draft_lane", kLaneStageSubs);
    st.lane_appends++;
}

// ---- the round schedule ---------------------------------------------------------------------------
RoundTracker::After RoundTracker::after_step(int pred_pos, int pred_token) {
    After a;
    bool settled = false;
    bool accepted = false;
    if (have_pending) {
        if (pending_pred_pos_ != pred_pos) {
            // The proposal on hand is not this position's. That is the index slide this repo has been
            // bitten by, and it is fatal to the number rather than cosmetic, so it is reported rather
            // than papered over: the caller aborts the run.
            a.alignment_broken = true;
            return a;
        }
        settled = true;
        accepted = (pending_proposal_ == pred_token);
        a.adjudicated = true;
        a.accept = accepted;
        a.settled_pos = pending_pred_pos_;
        a.settled_proposal = pending_proposal_;
        have_pending = false;
    }
    if (!settled) {
        a.draft = true;              // the previous step drafted nothing, so this one is a round
    } else if (sched == Schedule::kEvery) {
        a.draft = true;              // every draft context is priced; nothing is skipped
    } else if (accepted) {
        a.draft = false;             // kLoop: q advanced by 2, so one step carries no round
    } else {
        a.drop_lane = true;          // kLoop: greedy_mtp's `m.restore(msnap)` on a reject
        a.draft = true;              // ... and the next round is at the very next position
    }
    return a;
}

void RoundTracker::pending(int pred_pos, int proposal) {
    have_pending = true;
    pending_pred_pos_ = pred_pos;
    pending_proposal_ = proposal;
}

const char* schedule_name(Schedule s) { return s == Schedule::kEvery ? "every" : "loop"; }

Schedule parse_schedule(const std::string& name) {
    if (name == "loop") return Schedule::kLoop;
    if (name == "every") return Schedule::kEvery;
    throw std::runtime_error(
        "--mtp-sched takes `loop` (THE DEFAULT: reproduce speccheck/mtp.py's greedy_mtp(k=1) round "
        "sequence — draft at q+1, next round at q+2 on an accept and q+1 on a reject, the rejected "
        "round's cache lane truncated, which is the round geometry the carried accept rate of 0.7329 "
        "was measured on) or `every` (draft after every step, i.e. price every draft context the "
        "stream contains, whose cache therefore holds more lanes than the loop ever gives a draft). "
        "Got '" + name + "'. Parse-or-throw like every other A/B flag: a typo that silently became the "
        "default would price one geometry and report the other.");
}

}  // namespace engine::mtpdraft
