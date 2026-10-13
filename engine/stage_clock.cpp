// The stage clock `engine/stage_clock.h` documents. Read that first: it owns WHY the three
// granularities exist, what each one's number is allowed to be called, and why the marker precedes the
// submission instead of following it.
#include "engine/stage_clock.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <stdexcept>

#include "engine/decode_step.h"   // planned_launch_counts(), for the report's plan cross-check

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
        case kStageCount: break;
    }
    return "invalid";
}

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

Clock::Clock(Mode mode) : mode_(mode) {}

void Clock::begin_step() {
    in_step_ = true;
    open_ = false;
}

void Clock::end_step() {
    if (!in_step_) {
        throw std::runtime_error("stage clock end_step() without begin_step()");
    }
    if (open_) close_segment();
    ++steps_;
    in_step_ = false;
}

void Clock::note(Stage s, int submissions) {
    if (!in_step_) {   // a wave body, or any body submitted outside a measured step
        ++notes_ignored_;
        return;
    }
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

double calibrate_idle_sync(int iters) {
    if (iters < 1) iters = 1;
    drain();   // make sure nothing real is still in flight before timing the empty wait
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) drain();
    const auto t1 = std::chrono::steady_clock::now();
    return us_since(t0, t1) / static_cast<double>(iters);
}

}  // namespace engine::stageclock
