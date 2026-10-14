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
//   `exposed`  — `serialized_sum - burst`, for the SAME launches at the same lanes. In serialized
//              passes a launch pays its dispatch, its execution and its wait without overlapping, so
//              `serialized ≈ Σexec + dispatch + waits`, while the pipelined burst pays
//              `Σexec + whatever dispatch it FAILS to hide`. A LARGE gap therefore means the queue was
//              already hiding dispatch behind execution — nothing submission-side left to win, the
//              residue is inside the kernels (story B). A gap of ~0 means the real step's queue is
//              already empty launch by launch, so dispatch is exposed in the open and submission is
//              where the time goes (story A). Same reasoning as `bench/plan-2026-10-12.md` Q6 item 1,
//              whose first sketch had this polarity inverted; corrected in place 2026-10-13.
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
// SCOPE, for a STEP clock: one-row bodies only. The clock is armed by the driver around
// a generation step (`decode_step`, i.e. `submit_token_body` + `finish_token`); a wave body's notes are
// ignored — see `Clock::note()`'s "outside a step" rule — so a mis-armed clock shows up as
// `notes_ignored` in the report instead of as a plausible wrong number.
//
// WHY THERE IS A SECOND UNIT, and what a WAVE clock is allowed to claim. A wave is the same marker set
// over `ceil(T / wave_rows)` fat bodies, and the question it has to answer is not "how much dispatch is
// exposed" but "which FAMILY owns the wave's milliseconds" — the fork between replacing the hand
// matmuls with a library route and chunking §04.4's recurrence (P9) turns on that split and nothing else
// in the tree can produce it. Everything above applies unchanged, with three differences that come from
// the wave being wide rather than from any new idea:
//
//   * FINER FAMILIES. A wave's bill is `engine/launch_stats.h`'s family list, not §07's per-token table,
//     because the decision needs `gdn_conv`/`gdn_gates`/`gdn_gated_norm` (which batch, so P6 already made
//     them cheap) separated from `gdn_rec` (which does not batch, so it is what P9 would change), and
//     `mlp_gateup` from `mlp_down`. The markers name those fine families and a STEP clock folds them back
//     to `gdn`/`mlp` in `note()` — see `coarse_stage()` — so `--time-stages` reports exactly the nine
//     rows it always has, unchanged, while `--time-waves` gets the launcher's eleven.
//   * `gdn_rec` HAS NO `LAUNCHSTAT` TWIN, deliberately (`engine/launch_stats.h`: a recurrence launch has
//     no row width that could batch, so a launcher counter for it could not fail). For the wave report it
//     is the opposite of decoration: the recurrence's own milliseconds ARE the P9 question, so the clock
//     carries it as a family with `rows_total == launches` (one row per launch by construction) and
//     `launchstats_calls=-1` (no twin). A reader who wants a launcher-side witness for the recurrence has
//     to count launches, and `plan_launches`/`launches` is that comparison.
//   * THE COST OF A DRAIN IS SMALLER HERE, WHICH IS WHY THIS INSTRUMENT CAN BE BELIEVED. In a one-row
//     step a family is often a 30 us launch, so a several-microsecond wait inside it is a few percent of
//     the number. A 512-row wave body's families are hundreds of microseconds to milliseconds each, so
//     the same wait is a much smaller fraction — the serialized wave is a less distorted picture of its
//     subject than the serialized step is of the step. That is an order-of-magnitude argument, not a
//     proof, which is why `WAVEGAP` prints the serialized-vs-burst difference next to the shares and a
//     reader can see the distortion instead of being told to trust it.
//
// WHY THERE IS NO `gran=kernel` FOR A WAVE, which is the one place this file refuses a question rather
// than answering it. `gdn_rec` is 18 launches per token per body (§04.4, a scan), so a per-submission
// drain on a 512-row body would add ~9 200 waits per layer-run — tens of milliseconds of instrument —
// and, worse, would drain the very queue whose pipelining P9's chunked scan is trying to buy. The
// recurrence's shape is the thing under test; measuring it with a wait after every one of its launches
// measures the wait. `Clock::begin_wave()` therefore throws for `Mode::kKernel`.
//
// WHAT THE SHARES ARE SHARES OF, said before anyone quotes one. `ms` and `share_pct` come from the
// SERIALIZED wave pass: family X's `ms` is what family X took when nothing else was in the queue. The
// production wave is the burst pass, and per-family numbers are not obtainable from it without either a
// drain (which makes it the serialized pass) or GPU events (untrustworthy below ~1 ms on this iGPU —
// `prototype/README.md` rule 7). So a share is a share of `serialized_wall_ms`, and `burst_wall_ms` /
// `exposed_ms` on the `WAVEGAP` line are what convert it: multiply a share by `burst_wall_ms` to put it
// on the production wall, which assumes the serialization cost is spread in proportion to family time.
// That assumption is exactly what `exposed_ms` sizes; if it is large, say so with the number rather than
// quoting shares against the burst wall.

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
    kGdn,           // the §07 bill row: conv+gates+rec+gated_norm together
    kAttn,
    kMlp,           // the §07 bill row: gateup+down together
    kCast,
    kCopy,
    kLmHead,
    kHostTail,
    // The launcher families below spell themselves exactly as `engine/launch_stats.h` names them, so one
    // word names one concept in both instruments and in `LAUNCHSTAT`. A step clock folds them into the
    // rows above; a wave clock reports them as they are.
    kGdnConv,
    kGdnGates,
    kGdnRec,        // §04.4's recurrence. Not a `LAUNCHSTAT` family, on purpose; see the header.
    kGdnChunk,      // P9's §04.7 chunked scan: ONE site (two launches) per GDN layer per BODY, where
                    // kGdnRec is one per layer per TOKEN. Which of the two rows is non-zero is the
                    // resolved form's doing (`kernels.h:resolve_gdn_scan`), and a step — a one-token
                    // body — can only ever fill kGdnRec, which is why the step plan bill above needs
                    // no knowledge of the form at all.
    kGdnGatedNorm,
    kMlpGateup,
    kMlpDown,
    kStageCount,
};
const char* stage_name(Stage s);

// Which §07 bill row a launcher family belongs to: the six families above map onto `kGdn`/`kMlp` and
// every other stage is its own row. A STEP clock folds at `note()` time — before the segment comparison,
// which is what keeps `--time-stages`' segments, sync count and nine report rows byte-for-byte what they
// were before the fine names existed — and a WAVE clock does not fold.
Stage coarse_stage(Stage s);

// The three submission granularities. `kBurst` is not a clock mode — it is the production path, timed
// by the driver around the whole step — but it belongs in this enum because the report compares all
// three and every caller has to name them the same way.
enum class Mode : uint8_t { kBurst, kStage, kKernel };
const char* mode_name(Mode m);

class Clock;

// The armed clock, or null. `decode_step.cpp`'s markers read it through `note()` below.
void arm(Clock* c);
Clock* armed();

// Which body of work a clock is segmenting. It changes two things and nothing else: whether the launcher
// families are folded (§07's bill rows are the step's, the launcher's are the wave's) and which report
// function may be called on it.
enum class Unit : uint8_t { kStep, kWave };
const char* unit_name(Unit u);

// What a wave pass is, so the wave report can derive its own expected launch bill from §00's layer
// counts instead of trusting a typed number or the instrument's own reading — the same reason
// `plan_bill()` exists for a step. `bodies` is the piece a step plan does not have: `prefill_wave`
// splits a prompt taller than the scratch into that many `submit_wave_body` calls, and every family
// that batches is billed per body while `gdn_rec` and the embedding copies are billed per token.
struct WaveShape {
    int tokens = 0;        // prompt tokens the pass prefills
    int bodies = 0;        // `submit_wave_body` calls that takes = ceil(tokens / wave_rows)
    int wave_rows = 0;     // the scratch's row ceiling it split at (`sc.wave_rows`)
    int priced_rows = 0;   // `finish_token` calls: 1 under `kLast`, `tokens` under `kAll`
    double wall_ms = 0.0;  // the driver's wall around the pass, filled in after it ran
    // Which §04.4 scan the pass ran, as the RESOLVED form rather than as the flag: the plan needs the
    // form because `gdn_rec` and `gdn_chunk` bill differently (per token vs per body), and it needs
    // the resolved one because a one-row final body is `gdn_rec` even in a `--gdn-scan chunk` run.
    // `wave_plan_bill` derives the split from `tokens`/`bodies`/`wave_rows`, so the plan and the
    // dispatch cannot disagree about how many bodies were big enough to chunk.
    bool gdn_chunked = false;
};

// One measured pass of serialized submission. Constructed per pass, armed for its duration; not
// thread-safe because the engine submits from one host thread (`engine/launch_stats.h` makes the same
// note about its counters).
class Clock {
public:
    // A step clock folds the launcher families into §07's bill rows; a wave clock reports them.
    explicit Clock(Mode mode, Unit unit = Unit::kStep);

    Clock(const Clock&) = delete;
    Clock& operator=(const Clock&) = delete;

    // Call around one step. `note()` outside a step is IGNORED and counted, which is what keeps a
    // prefill wave or a stray body from being charged to a step that is not being measured.
    void begin_step();
    void end_step();

    // Call around one whole wave pass (every body of `prefill_wave`, and the finish that prices it).
    // One begin/end for the pass rather than one per body, because the families are what is being
    // attributed and a body boundary is not a family boundary: the host work between bodies belongs to
    // whichever segment is open and shows up in `ungaged`/the neighbour row, never as a hole.
    // Throws for `Mode::kKernel` — see the header for why a wave refuses that granularity.
    void begin_wave();
    void end_wave();

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

    // `WAVE gran=… family=…` lines, the `TOTAL` line that reconciles them against `shape.wall_ms`, and
    // the `GROUP` line that adds the families the funding decision is actually made on. Reads the shape
    // the driver filled in (tokens/bodies/priced_rows/wall_ms), because the expected launch bill is a
    // property of the pass, not of the clock. Only valid on a wave clock; throws otherwise so a caller
    // cannot print a wave report of a step.
    std::string describe_wave(const WaveShape& shape) const;

private:
    void close_segment();   // the wait + the charge; see `note()` for when it runs

    void begin_unit();
    void end_unit(const char* what);

    Mode mode_;
    Unit unit_;
    bool in_step_ = false;
    bool open_ = false;
    Stage stage_ = kStageCount;
    int open_subs_ = 0;
    std::chrono::steady_clock::time_point seg_start_{};

    double stage_us_[kStageCount] = {0.0};
    long long stage_subs_[kStageCount] = {0};
    long long stage_segs_[kStageCount] = {0};
    // `launch_stats.h`'s counters as they stood at `begin_wave()`, so the wave report can price this
    // pass by delta. Unused by a step clock (its plan cross-check is per step, and the run's own
    // `LAUNCHSTAT` line is already the launcher-side witness there).
    long long ls_calls_[kStageCount] = {0};
    long long ls_rows_[kStageCount] = {0};
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
