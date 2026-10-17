// The stage clock `engine/stage_clock.h` documents. Read that first: it owns WHY the three
// granularities exist, what each one's number is allowed to be called, and why the marker precedes the
// submission instead of following it.
#include "engine/stage_clock.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>

#include "engine/decode_step.h"   // planned_launch_counts(), for the report's plan cross-check
#include "engine/launch_stats.h"   // the launcher-side counts the wave report reconciles against
#include "engine/state.h"          // §00's layer counts, which the wave plan bill is derived from

namespace engine::stageclock {
namespace {

// One global, one writer thread: the engine submits from a single host thread, so the arm/read pair
// needs no atomics (same reasoning as `engine/launch_stats.cpp`).
Clock* g_clock = nullptr;

double us_since(const std::chrono::steady_clock::time_point& from,
                const std::chrono::steady_clock::time_point& to) {
    return std::chrono::duration<double, std::micro>(to - from).count();
}

// The wait. A HIP failure here is a real fault, not a slow step, and it throws: a segment measured
// across a faulted queue would be a number about nothing.
void drain() {
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        throw std::runtime_error(std::string("stage clock sync: hip: ") + hipGetErrorString(e));
    }
}

// The plan bill, indexed by `Stage` rather than listed in its order, so the report can put an observed
// count next to the number `gate.py` section (d) polices without a positional correspondence that could
// drift. `copy` is `other` minus the lm_head launch (the 19 D2D copies), `host_tail` is a D2H and a host
// argmax and therefore not a launch, and every row is per TOKEN — which is what `Clock::describe()`
// divides the observed counts by before comparing.
std::array<int, kStageCount> plan_bill() {
    const LaunchCounts c = planned_launch_counts();
    std::array<int, kStageCount> p{};
    p[kGlue] = c.glue;
    p[kProj] = c.proj;
    p[kGdn] = c.gdn;
    p[kAttn] = c.attn;
    p[kMlp] = c.mlp;
    p[kCast] = c.cast;
    p[kCopy] = c.other - 1;
    p[kLmHead] = 1;
    p[kHostTail] = 0;
    return p;
}

// The wave's plan bill, derived from §00's layer counts for the shape the driver describes, in the
// launcher families the wave report is written in. Every row is a count of SUBMISSION SITES, which is
// what `note()` counts and what `launch_stats.h` counts for attention (its header: sites, not kernels —
// one attention site issues 1-3 launches by lane depth), so the three instruments are counting the same
// thing and their agreement means something. Read against `planned_launch_counts()`'s table:
//
//   glue  1 + 2 x 24 per body                                proj  5 x 18 + 4 x 6 per body
//   gdn_conv / gdn_gates / gdn_gated_norm  1 x 18 per body   gdn_rec  1 x 18 PER TOKEN (a scan)
//   attn  6 per body (sites)                                 mlp_gateup / mlp_down  24 per body each
//   cast  3 x 24 + 1 x 18 + 6 per body on the hand route, 1 x 18 on a library body,
//         + 1 per priced row
//   copy  18 per body (the conv-window rolls) + the embedding lookup: 1 per one-row body,
//         2 per wider body (one id upload + one gather), so tokens + 18 for a per-token pass
//   lm_head  1 per priced row                               host_tail  0 (a D2H and a host loop)
//
// The `cast` row is the one worth reading twice: per layer the body casts the norm tile for the mixer
// (24), the MLP norm tile (24) and `mlp_act` (24), a GDN layer adds the gated-norm output cast (18),
// an attention layer adds the gated-output cast (6) — 3x24 + 1x18 + 6 = 96 per body. For a body of one
// token with one priced row that must be, and is checked to be, the per-token plan's 97.
//
// The hand bill is the whole story only on the hand route. `submit_wave_body` guards every widening whose
// sole reader is a hand kernel on the same predicate its call sites use, so on a library route the only
// cast a wide body still submits is the per-GDN-layer one whose readers are the 16-row
// `in_proj_b`/`in_proj_a` pair: 1x18 = 18 per body. The GDN layer's other cast, the packed
// `g_raw│g_a│g_b` widening, is gone at EVERY width and on BOTH routes (D30 1.7'): `conv_kernel` and
// `gates_kernel` read the BF16 tiles and widen in-register, so nothing ever reads the F32 copy again.
// That is why this row moved instead of growing a second route flag the way 4.1-a's guard did, and it is
// what made 1.7' a §07 launch-bill change rather than a route guard. The `priced` term is route-blind:
// `finish_token`'s widening feeds `lm_head_full`, which is a hand kernel at every width.
struct WavePlanBill {
    std::array<long long, kStageCount> subs{};
};

WavePlanBill wave_plan_bill(const WaveShape& sh) {
    const long long bodies = sh.bodies > 0 ? sh.bodies : 0;
    const long long tokens = sh.tokens > 0 ? sh.tokens : 0;
    const long long priced = sh.priced_rows > 0 ? sh.priced_rows : 0;
    const long long L = kNumLayers, G = kNumGdnLayers, A = kNumAttnLayers;
    WavePlanBill p;
    p.subs[kGlue] = (1 + 2 * L) * bodies;
    p.subs[kProj] = (5 * G + 4 * A) * bodies;
    p.subs[kGdnConv] = G * bodies;
    p.subs[kGdnGates] = G * bodies;
    p.subs[kGdnRec] = G * tokens;
    p.subs[kGdnRec] = G * tokens;
    // The two scan forms, billed by what each one submits. Sequential recurrence: one site per layer
    // per TOKEN, because token t cannot start before t-1 finished writing S. Chunked scan: one site per
    // layer per BODY, and the body's rows that fit in one chunk is where the serial part stops (§04.7).
    // The one-row exception is derived here rather than trusted to the caller: with `bodies` and
    // `wave_rows` this file can see that the FINAL body of a run is one row (tokens ==
    // (bodies-1)*wave_rows + 1) and therefore ran `gdn_rec` for its one token even in a
    // `--gdn-scan chunk` run — the same rule `resolve_gdn_scan` applies, restated from the shape the
    // driver reports rather than from the flag, which is what makes the plan's row agree with the
    // counted one when the two are compared.
    if (sh.gdn_resident) {
        // Route A: one launch per GDN layer per body, on the `gdn_rec` row. The final body of a run can
        // be one row, and `resolve_gdn_scan` sends a one-row body through the per-token form — which for
        // one row is also exactly one launch, so the count is `G * bodies` with no exception to derive.
        p.subs[kGdnRec] = G * bodies;
        p.subs[kGdnChunk] = 0;
    } else if (sh.gdn_chunked) {
        const long long last_body_rows =
            sh.bodies > 0 ? sh.tokens - (sh.bodies - 1) * sh.wave_rows : 0;
        const long long single = (last_body_rows == 1) ? 1 : 0;   // that body is seq's, by the rule
        p.subs[kGdnRec] = G * single;        // its one token, and nothing else in a chunked run
        p.subs[kGdnChunk] = G * (sh.bodies - single);
    } else {
        p.subs[kGdnChunk] = 0;
    }
    p.subs[kGdnGatedNorm] = G * bodies;
    p.subs[kAttn] = A * bodies;
    p.subs[kMlpGateup] = L * bodies;
    p.subs[kMlpDown] = L * bodies;
    // The one row the GEMM route moves, and it moves because of who READS the widening, not who wrote
    // it (`decode_step.cpp` guards each site on `wave_gemm_is_library`, the predicate its `project()` /
    // `mlp_*` branches already use). Hand: §07's 3L + G + A = 96 per body. Library: G = 18, the one
    // per-GDN-layer cast whose readers (the two 16-row `proj_gemv`s) are not GEMMs at any width — the
    // GDN projection widening is not in either count any more, because no consumer reads an F32 copy of
    // those tiles at any width (D30 1.7'). A one-row body owes the HAND count whatever the run asked
    // for — `resolve_wave_gemm` again — so that body is derived from the shape here rather than trusted
    // to the caller, exactly as the chunked branch above derives it.
    //
    // How many of this pass's bodies were ONE ROW — the one derivation two rows below depend on, so it
    // lives here once rather than twice. Both are rules about a body's WIDTH rather than about a flag,
    // and both therefore have to re-derive the exception from the shape the driver reports instead of
    // trusting a caller to announce it: `resolve_wave_gemm` gives a one-row body the hand GEMV whatever
    // was asked, and `submit_wave_body` gives a one-row body the embedding MEMCPY whatever its gather
    // rule would otherwise allow. Same rule the chunked branch above applies to `gdn_rec`.
    const long long last_body_rows =
        sh.bodies > 0 ? sh.tokens - (sh.bodies - 1) * sh.wave_rows : 0;
    const long long single_row_bodies =
        sh.wave_rows <= 1 ? bodies : ((last_body_rows == 1) ? 1 : 0);
    const long long casts_hand = 3 * L + G + A;
    const long long casts_library = G;
    long long lib_bodies = 0;
    long long hand_bodies = bodies;
    if (sh.gemm_library) {
        lib_bodies = bodies - single_row_bodies;
        hand_bodies = single_row_bodies;
    }
    p.subs[kCast] = casts_hand * hand_bodies + casts_library * lib_bodies + priced;
    // `copy` is the conv-window rolls plus the §02.3 embedding lookup, and the LOOKUP half is now a
    // row-count row too (D30 4.2′): a one-row body still submits the one `hipMemcpyAsync` §07 billed
    // (so a decode step's bill did not move, which `body1_plan_matches_token_plan` below proves), and a
    // wider body submits TWO operations — one upload of the body's token ids and one gather over the
    // table — where it used to submit one 2 KiB copy per row. That is the whole of the item: a 512-row
    // body went from 512 operations to 2, and the wave's 4096 embedding copies became 16.
    // Route-blind (unlike `cast` above): which form a body runs depends on `n` and on nothing else.
    p.subs[kCopy] = single_row_bodies + 2 * (bodies - single_row_bodies) + G * bodies;
    p.subs[kLmHead] = priced;
    p.subs[kHostTail] = 0;
    return p;
}

// Which `launch_stats.h` family, if any, observes this stage. Eight of the fifteen have a twin; the
// recurrence, the casts, the copies, the lm_head and the host tail do not, and the report prints
// `launchstats_calls=-1` for those rather than inventing a number. Taking the address of the accessor's
// reference is safe: each is a function-local static (`engine/launch_stats.cpp`).
launchstats::Family* launchstats_twin(Stage s) {
    switch (s) {
        case kGlue: return &launchstats::glue();
        case kProj: return &launchstats::proj();
        case kMlpGateup: return &launchstats::mlp_gateup();
        case kMlpDown: return &launchstats::mlp_down();
        case kGdnConv: return &launchstats::gdn_conv();
        case kGdnGates: return &launchstats::gdn_gates();
        case kGdnGatedNorm: return &launchstats::gdn_gated_norm();
        case kAttn: return &launchstats::attn();
        default: return nullptr;
    }
}

// The order the wave report prints families in: §07's stack order, mixer internals in launch order, and
// the three non-launch rows last. Not the enum order, because the enum keeps §07's nine rows first so
// the STEP report's row order is untouched.
const Stage kWaveOrder[] = {kGlue,   kProj,   kGdnConv,        kGdnGates, kGdnRec,
                            kGdnChunk, kGdnGatedNorm, kAttn, kMlpGateup,  kMlpDown,  kCast,
                            kCopy,   kLmHead, kHostTail};

// The reconciliation band for `ungaged_ms`: everything the segments do not cover — the host work before
// the first marker of the pass (the body's precondition and token-validation loops) and after the last
// drain (the row bookkeeping and the driver's own return). It is scalar host work with no device wait in
// it, so on a wave it should be far under 1 %, and the point of the band is not to grade it but to make a
// MISSING MARKER loud: a family that is never noted is not merely absent from the report, its time is
// charged to a neighbour and only the residue gives it away. `cast` alone is a few percent of a fat wave,
// so a dropped marker of that size trips a 1 % band; a 1 ms floor keeps a short wave from tripping it for
// arithmetic noise. Deliberately tight, and stated in ms and percent on the TOTAL line so a reader can
// re-check the verdict by hand.
double wave_reconcile_tol_ms(double wall_ms) {
    return std::max(1.0, 0.01 * wall_ms);
}

}  // namespace

const char* stage_name(Stage s) {
    switch (s) {
        case kGlue: return "glue";
        case kProj: return "proj";
        case kGdn: return "gdn";
        case kAttn: return "attn";
        case kMlp: return "mlp";
        case kCast: return "cast";
        case kCopy: return "copy";
        case kLmHead: return "lm_head";
        case kHostTail: return "host_tail";
        // Spelled exactly as `engine/launch_stats.h` names the same family, so one word is one concept
        // across the two instruments and across `LAUNCHSTAT`. `gdn_rec` has no launcher counter (see
        // `launch_stats.h`: no row width, so a counter for it could not fail) but it keeps the spelling
        // the launch bill and P9 use.
        case kGdnConv: return "gdn_conv";
        case kGdnGates: return "gdn_gates";
        case kGdnRec: return "gdn_rec";
        case kGdnChunk: return "gdn_chunk";
        case kGdnGatedNorm: return "gdn_gated_norm";
        case kMlpGateup: return "mlp_gateup";
        case kMlpDown: return "mlp_down";
        case kStageCount: break;
    }
    return "invalid";
}

Stage coarse_stage(Stage s) {
    switch (s) {
        case kGdnConv:
        case kGdnGates:
        case kGdnRec:
        case kGdnChunk:
        case kGdnGatedNorm:
            return kGdn;
        case kMlpGateup:
        case kMlpDown:
            return kMlp;
        default:
            return s;   // already one of §07's nine bill rows
    }
}

const char* unit_name(Unit u) { return u == Unit::kWave ? "wave" : "step"; }

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::kBurst: return "burst";
        case Mode::kStage: return "stage";
        case Mode::kKernel: return "kernel";
    }
    return "invalid";
}

void arm(Clock* c) { g_clock = c; }
Clock* armed() { return g_clock; }

Clock::Clock(Mode mode, Unit unit) : mode_(mode), unit_(unit) {}

void Clock::begin_unit() {
    in_step_ = true;
    open_ = false;
}

void Clock::end_unit(const char* what) {
    if (!in_step_) {
        throw std::runtime_error(std::string("stage clock ") + what + " without a matching begin");
    }
    if (open_) close_segment();
    in_step_ = false;
}

void Clock::begin_step() { begin_unit(); }

void Clock::end_step() {
    end_unit("end_step()");
    ++steps_;
}

void Clock::begin_wave() {
    // See the header: a per-submission drain on a wave drains §04.4's recurrence launch by launch, and
    // the recurrence's queueing is the thing P9 would be buying. Refuse instead of reporting a number
    // whose instrument is bigger than its subject.
    if (mode_ == Mode::kKernel) {
        throw std::runtime_error(
            "stage clock: gran=kernel is not a wave granularity. gdn_rec is 18 launches per token, so a "
            "per-submission drain on a wide wave adds ~18k waits per body and measures the wait, not the "
            "recurrence's queueing. Use gran=stage (a drain at every family boundary), or "
            "--time-stages for the per-launch view of a one-row body.");
    }
    begin_unit();
    // Snapshot the launcher-side counters so the report can price THIS pass and not the whole run: the
    // wave report runs after `LAUNCHSTAT` has printed the run's totals (`decode_run.cpp`), exactly as
    // the step clock's passes do, and a delta is what makes the two instruments comparable.
    for (int i = 0; i < kStageCount; ++i) {
        const Stage s = static_cast<Stage>(i);
        if (launchstats::Family* f = launchstats_twin(s)) {
            ls_calls_[i] = f->calls;
            ls_rows_[i] = f->rows_total;
        }
    }
}

void Clock::end_wave() { end_unit("end_wave()"); }

void Clock::note(Stage s, int submissions) {
    if (!in_step_) {   // a wave body under a step clock, or any body outside a measured unit
        ++notes_ignored_;
        return;
    }
    // Folded HERE, before the segment comparison, so a step clock cannot tell the fine names were
    // introduced: conv/gates/rec/gated_norm still merge into one `gdn` segment and gateup/down into one
    // `mlp` segment exactly as they did before, with the same sync count and the same nine report rows.
    if (unit_ == Unit::kStep) s = coarse_stage(s);
    if (open_ && (mode_ == Mode::kKernel || s != stage_)) close_segment();
    if (!open_) {
        stage_ = s;
        open_subs_ = submissions;
        open_ = true;
        seg_start_ = std::chrono::steady_clock::now();
    } else {
        open_subs_ += submissions;   // same family, so the segment keeps running (stage mode only)
    }
}

void Clock::close_segment() {
    drain();
    const auto now = std::chrono::steady_clock::now();
    stage_us_[stage_] += us_since(seg_start_, now);
    stage_subs_[stage_] += open_subs_;
    ++stage_segs_[stage_];
    ++syncs_;
    seg_start_ = now;      // the next segment starts at the end of this wait, so the segments tile
    open_ = false;
}

std::string Clock::describe() const {
    const std::array<int, kStageCount> plan = plan_bill();
    std::string out;
    char buf[512];
    long long subs = 0;
    long long plan_subs = 0;      // the plan is a PER-STEP bill, so everything it is compared against
    double total_us = 0.0;        // here is per step too, never a pass total
    for (int i = 0; i < kStageCount; ++i) {
        const Stage s = static_cast<Stage>(i);
        total_us += stage_us_[s];
        subs += stage_subs_[s];
        plan_subs += plan[i];
    }
    // Uniform by construction — every generation step submits the same plan — so an exact integer
    // division is honest here, and `plan_match` is the line that says it was.
    const int steps = steps_ > 0 ? steps_ : 1;
    const double step_div = static_cast<double>(steps);
    for (int i = 0; i < kStageCount; ++i) {
        const Stage s = static_cast<Stage>(i);
        if (stage_segs_[s] == 0) continue;
        // No field widths after a `=`: these are key=value lines meant to be split on whitespace, and a
        // right-justified number prints as `launches_per_step= 49`, which parses as an empty value.
        // (`%-9s` on the family name is fine — that pads after the value, not before it.)
        std::snprintf(buf, sizeof buf,
                      "STAGE gran=%s family=%-9s ms_per_step=%.4f ms_total=%.3f"
                      " launches_per_step=%lld segments_per_step=%lld us_per_launch=%.2f",
                      mode_name(mode_), stage_name(s), stage_us_[s] / 1000.0 / step_div,
                      stage_us_[s] / 1000.0, stage_subs_[s] / steps, stage_segs_[s] / steps,
                      stage_subs_[s] ? stage_us_[s] / static_cast<double>(stage_subs_[s]) : 0.0);
        out += buf;
        out += "\n";
    }
    std::snprintf(buf, sizeof buf,
                  "STAGE gran=%s TOTAL steps=%d ms_per_step=%.4f ms_total=%.3f"
                  " launches_per_step=%lld plan_per_step=%lld plan_match=%d syncs_per_step=%lld",
                  mode_name(mode_), steps, total_us / 1000.0 / step_div, total_us / 1000.0,
                  subs / steps, plan_subs, subs / steps == plan_subs ? 1 : 0, syncs_ / steps);
    out += buf;
    if (notes_ignored_) {
        std::snprintf(buf, sizeof buf, " notes_ignored=%lld", notes_ignored_);
        out += buf;
    }
    return out;
}

std::string Clock::describe_wave(const WaveShape& shape) const {
    if (unit_ != Unit::kWave) {
        throw std::runtime_error(std::string("stage clock describe_wave() called on a ") +
                                 unit_name(unit_) + " clock");
    }
    const WavePlanBill plan = wave_plan_bill(shape);
    const double wall_ms = shape.wall_ms;
    const double tol_ms = wave_reconcile_tol_ms(wall_ms);
    std::string out;
    char buf[640];

    double sum_ms = 0.0;
    long long subs = 0;
    long long plan_subs = 0;
    for (int i = 0; i < kStageCount; ++i) {
        sum_ms += stage_us_[i] / 1000.0;
        subs += stage_subs_[i];
        plan_subs += plan.subs[i];
    }

    int families = 0;
    int launchstats_checked = 0;
    bool launchstats_ok = true;
    for (const Stage s : kWaveOrder) {
        if (stage_segs_[s] == 0) continue;
        ++families;
        const double ms = stage_us_[s] / 1000.0;
        const long long launches = stage_subs_[s];
        launchstats::Family* twin = launchstats_twin(s);
        // The delta over this pass, not the run's total: `begin_wave()` took the snapshot.
        const long long ls_calls = twin ? twin->calls - ls_calls_[s] : -1;
        const long long ls_rows = twin ? twin->rows_total - ls_rows_[s] : -1;
        if (twin) {
            ++launchstats_checked;
            // Two independent instruments over the same submissions: this clock counts a site where the
            // marker sits, `launch_stats.h` counts it where the launch happens. A marker on the wrong
            // line, a family that was never noted, or a site that was noted twice all separate them.
            if (ls_calls != launches || ls_rows < 0) launchstats_ok = false;
        }
        const bool counts_ok = (ls_calls < 0 || ls_calls == launches) && launches == plan.subs[s];
        std::snprintf(buf, sizeof buf,
                      "WAVE gran=%s family=%-15s ms=%.4f share_pct=%.2f launches=%lld segments=%lld"
                      " ms_per_launch=%.5f rows_total=%lld launchstats_calls=%lld plan_launches=%lld"
                      " counts_match=%d",
                      mode_name(mode_), stage_name(s), ms,
                      wall_ms > 0.0 ? 100.0 * ms / wall_ms : 0.0, launches, stage_segs_[s],
                      launches ? ms / static_cast<double>(launches) : 0.0,
                      // `rows_total` is the launcher's own width sum where one exists (P4-P8's batching
                      // evidence, same number `LAUNCHSTAT` prints), and for the recurrence it is one row
                      // per launch BY CONSTRUCTION — which is the reason it has no launcher counter
                      // (`launch_stats.h`) and the reason P9 exists. Nothing is claimed about the widths
                      // of the cast/copy/lm_head rows, so they say so instead of printing a plausible 0.
                      twin ? ls_rows : (s == kGdnRec ? launches : 0), ls_calls, plan.subs[s],
                      counts_ok ? 1 : 0);
        out += buf;
        if (!twin) {
            out += s == kGdnRec ? " rows_note=one_row_per_launch_by_construction"
                                : " rows_note=no_launcher_row_width";
        }
        out += "\n";
    }

    const double ungaged_ms = wall_ms - sum_ms;
    std::snprintf(buf, sizeof buf,
                  "WAVE gran=%s TOTAL tokens=%d bodies=%d wave_rows=%d priced_rows=%d"
                  " wall_ms=%.4f sum_ms=%.4f ungaged_ms=%.4f ungaged_pct=%.3f tol_ms=%.4f"
                  " within_tol=%d launches=%lld plan_launches=%lld plan_match=%d"
                  " launchstats_families=%d launchstats_match=%d syncs=%lld",
                  mode_name(mode_), shape.tokens, shape.bodies, shape.wave_rows, shape.priced_rows,
                  wall_ms, sum_ms, ungaged_ms, wall_ms > 0.0 ? 100.0 * ungaged_ms / wall_ms : 0.0,
                  tol_ms, (ungaged_ms >= -tol_ms && ungaged_ms <= tol_ms) ? 1 : 0, subs, plan_subs,
                  subs == plan_subs ? 1 : 0, launchstats_checked, launchstats_ok ? 1 : 0, syncs_);
    out += buf;
    // A self-check of the DERIVATION, not of the run: the wave bill for one body of one token with one
    // priced row must reproduce §07's per-token bill, which `gate.py` section (d) polices from the other
    // side. If someone edits one of the two formulas and not the other, this reads 0 in every run.
    {
        WaveShape one;
        one.tokens = one.bodies = one.wave_rows = one.priced_rows = 1;
        const WavePlanBill w1 = wave_plan_bill(one);
        const std::array<int, kStageCount> tok = plan_bill();
        // `gdn` and `mlp` are the two rows the tables spell differently: §07 bills them as 72 and 48 under
        // one name each, the wave bill carries them under six. Fold, then compare everything.
        const auto matches_token_plan = [&](const WavePlanBill& w) {
            const long long gdn_wave = w.subs[kGdnConv] + w.subs[kGdnGates] + w.subs[kGdnRec] +
                                       w.subs[kGdnChunk] + w.subs[kGdnGatedNorm];
            const long long mlp_wave = w.subs[kMlpGateup] + w.subs[kMlpDown];
            return w.subs[kGlue] == tok[kGlue] && w.subs[kProj] == tok[kProj] &&
                   gdn_wave == tok[kGdn] && w.subs[kAttn] == tok[kAttn] &&
                   mlp_wave == tok[kMlp] && w.subs[kCast] == tok[kCast] &&
                   w.subs[kCopy] == tok[kCopy] && w.subs[kLmHead] == tok[kLmHead];
        };
        const bool same = matches_token_plan(w1);
        // The same one-row body on a library route. `resolve_wave_gemm` gives a one-row body the hand
        // GEMV whatever the run asked for, so the route-aware `cast` row must fold back to §07's 97
        // here: this is the witness that the guard is a row-count rule and not a flag rule, and a guard
        // that could fire at n == 1 — i.e. one that could move a decode step — would read 0 in
        // every run instead of being a claim nobody can check.
        WaveShape one_library = one;
        one_library.gemm_library = true;
        const bool same_library = matches_token_plan(wave_plan_bill(one_library));
        std::snprintf(buf, sizeof buf,
                      " body1_plan_matches_token_plan=%d body1_library_plan_matches_token_plan=%d",
                      same ? 1 : 0, same_library ? 1 : 0);
        out += buf;
    }
    if (notes_ignored_) {
        std::snprintf(buf, sizeof buf, " notes_ignored=%lld", notes_ignored_);
        out += buf;
    }

    // The rollups the funding decision is actually taken on, printed rather than left to whoever is
    // reading eleven rows off a screen: `matmul_pct` is the library-route question, `gdn_rec_pct` is the
    // P9 question, and `nonmatmul_pct` is 100 − matmul by construction (the eight groups below partition
    // the families, so their pcts sum to the wall and `matmul_pct + nonmatmul_pct` sums to 100 of it).
    auto share = [&](std::initializer_list<Stage> ss) {
        double ms = 0.0;
        for (const Stage s : ss) ms += stage_us_[s] / 1000.0;
        return ms;
    };
    const double matmul_ms = share({kProj, kMlpGateup, kMlpDown});
    const double gdn_all_ms = share({kGdnConv, kGdnGates, kGdnRec, kGdnGatedNorm});
    const double gdn_batched_ms = gdn_all_ms - share({kGdnRec});
    const auto pct = [&](double ms) { return wall_ms > 0.0 ? 100.0 * ms / wall_ms : 0.0; };
    std::snprintf(buf, sizeof buf,
                  "\nWAVE gran=%s GROUP matmul_ms=%.4f matmul_pct=%.2f nonmatmul_pct=%.2f"
                  " gdn_all_pct=%.2f gdn_rec_pct=%.2f gdn_batched_pct=%.2f attn_pct=%.2f glue_pct=%.2f"
                  " cast_pct=%.2f copy_pct=%.2f lm_head_pct=%.2f host_tail_pct=%.2f",
                  mode_name(mode_), matmul_ms, pct(matmul_ms),
                  wall_ms > 0.0 ? 100.0 - pct(matmul_ms) : 0.0, pct(gdn_all_ms), pct(share({kGdnRec})),
                  pct(gdn_batched_ms), pct(share({kAttn})), pct(share({kGlue})), pct(share({kCast})),
                  pct(share({kCopy})), pct(share({kLmHead})), pct(share({kHostTail})));
    out += buf;
    return out;
}

double calibrate_idle_sync(int iters) {
    if (iters < 1) iters = 1;
    drain();   // make sure nothing real is still in flight before timing the empty wait
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) drain();
    const auto t1 = std::chrono::steady_clock::now();
    return us_since(t0, t1) / static_cast<double>(iters);
}

}  // namespace engine::stageclock
