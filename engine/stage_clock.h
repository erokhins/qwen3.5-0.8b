#pragma once
// engine/stage_clock.h — the instrument that can tell which of two stories a decode step is.
//
// WHY THIS EXISTS. `bench/performance-status.md` §6 and `engine/results.md`'s Q6 section leave a
// ~3.4-3.6 ms residue inside a ~12.9 ms token: the named families account for ~9.4 ms, and all 424
// launches priced at the block width each site actually uses are only ~1.6 ms of dispatch. Two
// stories fit that residue and they need OPPOSITE fixes:
//
//   A. EXPOSED SUBMISSION. Each launch is individually cheap, but nothing is queued behind it, so the
//      step pays a stall per launch instead of the 2.9-3.8 us it costs when it is merely queued.
//      The fix is submission-side: fewer, wider launches, or a captured graph (§6 already priced the
//      graph form at ~4 %, so this has to be tested before it is assumed).
//   B. SLOW IN SITU. Each kernel really does take that long inside a step — longer than its own
//      isolated burst said. The fix is memory behaviour, and nothing submission-side would help.
//
// Story A is a property of the STEP SHAPE, and no per-family burst can see it in either direction: a
// burst submits the same kernel N times back to back, so its queue is never empty and an exposed
// stall cannot appear at all. The cheapest measurement that separates the two is the real step at
// three submission granularities — one burst with a single sync, one sync per stage, one sync per
// kernel — which is what this file provides and `decode_run --time-stages` drives.
//
// THE THREE QUANTITIES, named honestly, because a stage clock that reports only its own per-stage
// numbers would be reporting a SERIALIZED step and calling it the step:
//
//   burst      the production shape: every launch of the step submitted to one stream, ONE wait (the
//              logits D2H in `finish_token`). This is what the ledger's ms/token rows price.
//   stage      the same launches, but the stream is drained at every family boundary. Each stage is
//              therefore measured alone, with nothing queued in front of it or behind it.
//   kernel     the same launches drained after every single submission.
//   `serialized_sum` — the sum of the stages of one step. Because a segment runs from the end of the
//              previous sync to the end of this one, the segments tile the step: the sum IS that
//              pass's wall clock, which is the cross-check the report prints.
//   `exposed`  — `serialized_sum - burst`, for the SAME launches at the same lanes. If it is ~0 there
//              is no overlap to win and the residue is inside the kernels (story B). If it is large,
//              the stall is exposed and submission is where the time goes (story A).
//
// WHY THE MARKER PRECEDES THE SUBMISSION, which is the one thing here that is easy to get wrong and
// silently wrong. A segment has to be closed BEFORE the next one's work is queued: sync-after-note
// would drain the note's own launch and charge it to the previous stage, silently shifting every
// boundary by one launch (each stage would report the previous stage's work, and on a step where most
// segments are one launch that is invisible). So `note(s)` means "the submissions that FOLLOW belong
// to family s", and it is called immediately before each launch/memcpy site in `decode_step.cpp`.
//
// WHAT A STAGE IS: the families `planned_launch_counts()` already names, plus `copy` (the D2D memcpys)
// and `host_tail` (the logits D2H and the argmax), so a stage row can be set against the launch bill
// line for line. `note()` also counts submissions, and at a lane count inside the attention staging the
// observed total equals the plan's 424 exactly — the report prints both and says whether they match,
// which is what makes the instrument checkable rather than plausible. Above the staging an attention
// site legitimately issues 2 kernels instead of 1 (`kernels_attn.cpp`'s launch plan), so the observed
// total exceeds the plan there; that is reported, not smoothed.
//
// WHY `steady_clock` AND NOT EVENTS: `hipEventElapsedTime` returns exactly 0 for anything below ~1 ms
// on this iGPU (`prototype/README.md` rule 7 — "a zero is a broken timer, not a fast kernel"), and
// every stage here is far below 1 ms. So: wall clock around a real wait. The wait is
// `hipDeviceSynchronize()`, which is the one form that cannot lie about the engine's queue: every
// submission in a step goes to the legacy default stream, including the `hipMemcpyAsync` calls, and a
// device-wide sync is the wait that provably covers memcpys as well as kernels. Its own cost is not
// free and is not hidden: `calibrate_idle_sync()` measures an idle sync and the report prints it next
// to the sync count, so a reader can subtract the instrument from the instrument's number.
//
// THE COST IT ADDS, and the rule that keeps the production path clean. When no clock is armed, the
// body's marker is one read of a null pointer: no sync, no launch, no reordering, no readback. That is
// the whole of what the instrument costs when it is off, and `decode_run --time-stages` is proven not
// to change a single output byte (engine/results.md, Q6.1). When it IS armed, it adds one wait per
// stage (or per kernel) to THAT PASS — which is exactly why the burst pass is a separate pass with no
// clock at all, rather than the same pass with the clock switched off halfway.
//
// SCOPE, stated so nobody has to infer it: one-row bodies only. The clock is armed by the driver around
// a generation step (`decode_step`, i.e. `submit_token_body` + `finish_token`); a wave body's notes are
// ignored — see `Clock::note()`'s "outside a step" rule — so a mis-armed clock shows up as
// `notes_ignored` in the report instead of as a plausible wrong number.

#include <chrono>
#include <cstdint>
#include <string>

namespace engine::stageclock {

// One row of the launch bill. `kCopy` is `LaunchCounts::other` minus the `lm_head` launch (the
// embedding row and the 18 conv-window rolls), and `kHostTail` is not a submission at all: it is the
// blocking D2H plus the host argmax, noted with zero submissions so the totals still mean launches.
enum Stage : uint8_t {
    kGlue = 0,
    kProj,
    kGdn,
    kAttn,
    kMlp,
    kCast,
    kCopy,
    kLmHead,
    kHostTail,
    kStageCount,
};
const char* stage_name(Stage s);

// The three submission granularities. `kBurst` is not a clock mode — it is the production path, timed
// by the driver around the whole step — but it belongs in this enum because the report compares all
// three and every caller has to name them the same way.
enum class Mode : uint8_t { kBurst, kStage, kKernel };
const char* mode_name(Mode m);

class Clock;

// The armed clock, or null. `decode_step.cpp`'s markers read it through `note()` below.
void arm(Clock* c);
Clock* armed();

// One measured pass of serialized submission. Constructed per pass, armed for its duration; not
// thread-safe because the engine submits from one host thread (`engine/launch_stats.h` makes the same
// note about its counters).
class Clock {
public:
    explicit Clock(Mode mode);

    Clock(const Clock&) = delete;
    Clock& operator=(const Clock&) = delete;

    // Call around one step. `note()` outside a step is IGNORED and counted, which is what keeps a
    // prefill wave or a stray body from being charged to a step that is not being measured.
    void begin_step();
    void end_step();

    // "The submissions that follow belong to family `s`." Called immediately BEFORE each launch/memcpy
    // site; see the header note for why the order is load-bearing.
    void note(Stage s, int submissions = 1);

    Mode mode() const { return mode_; }

    // `STAGE gran=… family=…` lines, one per family that ran, plus the totals and the plan cross-check.
    // No trailing newline; `decode_run` prints it ahead of `LAUNCHSTAT`/`RESULT` so the last line of a
    // run stays the result line (the `tail -1` trap in `AGENTS.md`).
    //
    // There is deliberately no per-step accessor: the driver times each step with the same clock and
    // prints those (`STAGEPASS`/`STAGESTEP`), and the segments tile the step, so `TOTAL ms_per_step`
    // here and the driver's `ms_mean` for the same `gran=` are two readings of one quantity. They agree
    // to the sync's own cost, and a disagreement would be an unaccounted host gap between segments.
    std::string describe() const;

private:
    void close_segment();   // the wait + the charge; see `note()` for when it runs

    Mode mode_;
    bool in_step_ = false;
    bool open_ = false;
    Stage stage_ = kStageCount;
    int open_subs_ = 0;
    std::chrono::steady_clock::time_point seg_start_{};

    double stage_us_[kStageCount] = {0.0};
    long long stage_subs_[kStageCount] = {0};
    long long stage_segs_[kStageCount] = {0};
    int steps_ = 0;
    long long syncs_ = 0;
    long long notes_ignored_ = 0;
};

// The marker every submission site in `decode_step.cpp` calls, immediately BEFORE its launch. When no
// clock is armed this is a null-pointer read and a return, which is the whole of what the instrument
// costs the production path; the header's note on why the marker precedes the submission is the other
// half of what makes it safe to leave in every build.
inline void note(Stage s, int submissions = 1) {
    if (Clock* c = armed()) c->note(s, submissions);
}

// The instrument's own overhead: the mean microseconds of a `hipDeviceSynchronize()` on an already
// idle queue. Every serialized segment ends in one of those, so a per-stage figure contains roughly one
// of them, and a reader who knows the count can take the instrument back out. Measured as a burst of
// `iters` idle syncs, one wall-clock reading around the lot (`prototype/README.md` rule 7 again: this
// is sub-millisecond per call, so it has to be a burst).
double calibrate_idle_sync(int iters);

}  // namespace engine::stageclock
