#pragma once
// engine/mtp_meter.h — where an MTP round's milliseconds actually go. B0 of `prototype/mtp_accept/
// BEAT-MTP.md`.
//
// WHY THIS EXISTS. A k=1 round at 4,032 lanes costs 10.916 ms more than a plain step, but the per-family
// kernel marginals account for only 4.3-5.5 ms of it (`BEAT-MTP.md`). The missing half is PRESUMED to be
// the round's own machinery — the mid-checkpoint, the shadow re-seed, the lane staging, the verdict
// kernel, the blocking logits D2H, the sync bubbles. "Presumed" is the phrase this file exists to delete:
// B1 and B2 are funded against that split, so the split has to be measured, not inferred.
//
// WHAT IT IS NOT. It is not `--time-stages`. The stage clock segments ONE body by §07's family list and
// reconciles it against a per-step launch plan; a round is not a step, has no §07 plan to reconcile
// against, and its machinery (checkpoint, shadow seed, lane staging, verdict, rollback) belongs to no §07
// family — which is exactly why `--mtp-verify` and `--mtp-accept` already refuse `--time-stages`. The
// stage clock's `kMtp` row bills a DRAFT's submissions per step and nothing else. So this is a second,
// tiny instrument rather than a contortion of the first.
//
// TWO SHAPES, because they are different machines and one number would hide the difference:
//
//   role=verify  the read-only side pass (`--mtp-verify 1`): the pre-step snapshot take, then one shadow
//                body with its seed restore, lane staging, batched head, blocking vocabulary D2H, lane
//                revert and report. The plain step sits BETWEEN the snapshot and the body and is NOT in
//                the wall: the wall is the union of those two windows and nothing else.
//   role=accept  the accept loop (`--mtp-draft 1 --mtp-accept 1 --gdn-loop 1`): one round = the body
//                (which carries the mid-round checkpoint inside its own launches), the batched head, the
//                on-device verdict, the 16-byte readback, the accept OR rollback branch (separate lines,
//                each with its own run count), lane bookkeeping, the next round's draft, and the driver's
//                own prints. Iterations where the loop pays a plain one-row step instead of a round are
//                counted in `plain_steps=` and are NOT metered — they are the ledger's step, not a round.
//
// ─── HOW EVERY BOUNDARY WORKS, and what it can therefore under- or over-report ────────────────────────
// NO hipEVENTS, and that is a measured constraint, not a preference: `hipEventElapsedTime` returns EXACTLY
// 0 for work below ~1 ms on this iGPU and has been seen reading 0.000 for a 2.1 ms kernel
// (`engine/performance-status.md` §6, `prototype/README.md` rule 7; `stage_clock.h`'s "WHY THERE IS A
// SECOND UNIT" made the same call for the same reason). Several phases here are well under 1 ms, so an
// event pair would silently report zero for exactly the phases the funding decision turns on. Every
// boundary is therefore a HOST `steady_clock` pair, and each phase's `sync_added=` field says what makes
// its close meaningful:
//
//   sync_added=1  THE METER ADDS A `hipDeviceSynchronize()` AT THE CLOSE. Without it the phase would
//                 measure only host submission time, because launches are asynchronous — so this is the
//                 only way to charge a phase its DEVICE work. It also SERIALIZES: a production round
//                 overlaps dispatch with execution and a metered round cannot. So every number on such a
//                 line is an UPPER BOUND on that phase's share of a pipelined round, and
//                 `wall_ms_per_round` is an upper bound on the round. The instrument's own floor is
//                 printed (`idle_sync_us`, `syncs_added=`) so a reader can subtract it.
//   sync_added=0  No sync added. Either the phase's last site ALREADY blocks today — `hipMemcpy` D2D and
//                 `hipMemcpy` D2H both return to the host only once the copy has completed: the snapshot
//                 take, the shadow seed, the lane staging, the vocabulary D2H, the verdict word — or the
//                 phase is pure host work (argmax, prints, the §08.6 invariant check). These lines cost
//                 the run nothing it was not already paying, and they are the trustworthy ones.
//   One honest caveat on the blocking-copy lines: because a `hipMemcpy` drains the stream it was enqueued
//   on, a native-block phase also ABSORBS whatever device work was still queued when it ran. In the
//   shipping order nothing is queued at those points; if a number there looks implausible, read it against
//   its neighbour before believing it.
//
// COVERAGE, and why it can fail. `wall_ms_per_round` is the sum of the round's WINDOWS — disjoint host
// spans the driver opens (two per verify round: the snapshot take and the side pass; one per accept
// round). Phases are charged inside windows; whatever a window holds that no phase claimed is printed as
// `phase=overhead status=derived`. So
//
//     coverage = Σ measured phases / wall = 1 − overhead/wall
//
// and coverage is NOT 1 by construction: it falls when a phase was forgotten, or when a flag adds
// per-round work nobody metered. `prototype/mtp_accept/meter_round.sh` refuses to print a table outside
// 0.8-1.2. Double counting cannot reach coverage > 1 either: `Meter::begin()` refuses a nested phase
// rather than charging a window twice, and a window opened inside a window is refused the same way.
//
// TWO ROWS ARE OUT OF THE SUM BESIDES `overhead`: `summary`, and any `Kind::kSubphase` row. A sub-phase is
// a BREAKDOWN of milliseconds an enclosing phase already owns (`draft_vocab_d2h` and `draft_argmax_host`
// inside `draft`, which is where `prototype/mtp_accept/ROUND-TAX.md` §1 says the draft's vocabulary
// crossing lives), so charging one into the sum would bill the same span twice and redden the identity
// check in `meter_table.awk` — which is how that awk's own selftest proves the exclusion is load-bearing
// rather than decorative.
//
// WHAT CANNOT BE METERED WITH TODAY'S STRUCTURE, said rather than guessed:
//
//   * role=accept `mid_checkpoint`. The mid-round checkpoint is not a copy at the round's level: it is
//     TWO extra submissions PER GDN LAYER (a `conv_mid_window_kernel` derivation plus a 1 MiB state
//     capture — `engine/decode_step.cpp`'s two `sc.gdn_ckpt` sites) interleaved inside `submit_wave_body`.
//     A boundary around it means a sync inside the per-layer loop, i.e. 18 drains inside the body whose
//     pipelining is the thing being priced. It is metered INSIDE `phase=body` and printed as
//     `status=not_metered`. The honest way to price it is the pair role=verify already can, in ONE window
//     at ONE anchor: `--mtp-midsplit 1` gives `body_with_checkpoint` (second body, checkpoint armed)
//     beside `body` (first body, not armed).
//   * The dispatch EXPOSED by the added syncs is not separable from device time on this part. It is
//     bounded, not hidden: `syncs_added × idle_sync_us` is printed, and `--time-stages`' `exposed_ms` is
//     the same quantity for a step.
//
// NUMERICS. The meter adds device waits, reads clocks and counts. It submits nothing, allocates nothing
// per round, reorders nothing and touches no operand, so the emitted token stream and every `STATEFP`
// fingerprint must be identical with it on and off. That is its admission test, and
// `prototype/mtp_accept/meter_identity.sh` runs it field by field (both shapes, OFF vs ON; `DECWALL` and
// `RESULT` compared per field so a moved counter cannot hide among the durations). With the flag OFF the
// instrument costs one null-pointer read per guard object, which is what makes flag-off output
// byte-identical to a binary that does not contain this file.

#include <chrono>
#include <cstdint>
#include <string>

namespace engine::mtpmeter {

// Which round shape this run is metering. Derived from the flag that armed the meter and never asked for:
// `--mtp-verify` and `--mtp-accept` already refuse each other, so one run has exactly one shape.
enum Role : uint8_t { kVerify = 0, kAccept = 1, kRoleCount };
const char* role_name(Role r);

// The phases, in registry order. One spelling per phase, in ONE table that `describe()` prints from and
// that a `static_assert` ties to this enum, so a printed `phase=` cannot drift from the site that records
// it. `--time-stages`' `Stage` enum is the other instrument's family list and is not reused here: a
// round's phases are not §07 families, and folding them into it would bill the step's rows twice.
enum PhaseId : uint8_t {
    // role=verify
    kSnapTake = 0,   // the pre-step checkpoint take (19.27 MiB at this model)
    kShadowSeed,     // checkpoint -> shadow: the copy a reject would perform
    kWitness,        // --verify-guard / --mtp-midsplit downloads and byte-compares; 0 when both are off
    kLaneSave,       // save each body row's KV lane out of the live cache
    kLaneRevert,     // put those lanes back
    kLogitsD2H,      // the blocking vocabulary download
    kArgmax,         // host §03.4 argmax over the downloaded rows
    // role=accept
    kMidCheckpoint,  // printed `status=not_metered`; see the header
    kVerdict,        // the on-device verdict kernel
    kReadback,       // the four-int verdict word; this path has NO logits D2H
    kAcceptPath,     // ran on an accept only
    kRollback,       // ran on a reject only: two bulk D2D restores
    kLaneBookkeeping,  // §08.6 invariant check + the draft lane's length field; moves 0 bytes
    kDraft,          // the NEXT round's draft submissions
    // BREAKDOWNS inside `kDraft`, not siblings of it. See `Kind::kSubphase` for why they cannot be
    // siblings, and `prototype/mtp_accept/ROUND-TAX.md` §1 for why they exist at all.
    kDraftVocabD2H,  // the draft's vocabulary-sized download (a sub-phase of `kDraft`)
    kDraftArgmaxHost,// the host §03.4 scan over the downloaded row (a sub-phase of `kDraft`)
    // shared by both roles
    kBody,           // `submit_wave_body`; carries the mid-checkpoint when one is armed
    kBodyCkpt,       // the `--mtp-midsplit 1` second body: the body WITH the checkpoint armed
    kHead,           // one batched pass over the 485 MiB tied table, serving both rows
    kReportPrints,   // the run's own per-round printf+fflush: host work, named not buried
    kOverhead,       // DERIVED: window minus the measured phases
    kSummary,        // DERIVED: the meter's own verdict line; excluded from the sum
    kPhaseCount,
};

// How a phase's close is made meaningful. This IS the sync labelling the header explains, and it lives in
// the registry rather than at the call sites so a line cannot be printed with a label its author forgot.
enum class Kind : uint8_t {
    kNativeBlock,   // the phase ends in a HIP call that already blocks today; nothing added
    kHostOnly,      // pure host work; nothing to drain
    kSyncAdded,     // the meter adds a hipDeviceSynchronize(): serializes, so it INFLATES
    kNotMetered,    // no honest boundary exists for this at today's call sites
    kDerived,       // computed from the windows, not timed
    // A BREAKDOWN of one span of a `kSyncAdded`/`kNativeBlock` phase, charged inside that phase rather
    // than beside it, and therefore EXCLUDED from the sum exactly like `kDerived` — a nested span added
    // to the sum would make coverage exceed the window it lives in and turn the coverage check into a
    // number that cannot be read. `parent` on the registry row names the phase it breaks down; the
    // printer prints it as `inside=`, so a reader never has to guess which line owns the milliseconds.
    // Its close adds no wait: whatever makes the enclosing phase's close meaningful already covers it.
    kSubphase,
};

// The registry row. `note` is printed on the phase's line, so a reader never has to open this header to
// know what a number bounds; the prose version is in `prototype/mtp_accept/METER-PLAN.md`.
struct PhaseSpec {
    Role role;             // kRoleCount = shared by both roles
    const char* name;
    Kind kind;
    const char* note;
    // Only read for `Kind::kSubphase` rows: the phase this one is a breakdown of. Left defaulted on
    // every other row, and `sub_begin` refuses a sub-phase whose declared parent is not the open phase,
    // so a sub-phase row that forgot to name its parent fails the run that arms it rather than silently
    // charging its milliseconds to whoever happened to be open.
    PhaseId parent = kPhaseCount;
};

const PhaseSpec* specs();   // `kPhaseCount` rows, indexable by `PhaseId`

// The meter. One per run, constructed and armed by the driver when `--mtp-meter 1`. Call `finalize()`
// once after the generation loop, then `describe()` and `result_fields()`. Not thread-safe: the engine
// submits from one host thread (`engine/launch_stats.h` and `stage_clock.h` make the same note).
class Meter {
public:
    explicit Meter(Role role);

    // A window is a host span belonging to a round. Windows are disjoint and additive and the wall is
    // their sum; both throw if misused, because overlapping windows would double-count the wall and a
    // check that cannot be broken is not a check.
    void open_window();
    void close_window();
    bool window_open() const { return window_open_; }

    // A round is the unit every `ms_per_round=` divides by. Windows live inside rounds.
    void round_begin();
    void round_end();
    long long rounds() const { return rounds_; }

    void begin(PhaseId p);
    void end(PhaseId p);

    // A sub-phase: a span inside the phase that is currently open, named by the registry row's `parent`.
    // Throws unless (a) `p` is declared a sub-phase, (b) its declared parent IS the open phase, and
    // (c) no other sub-phase is open — a sub-phase nested in a sub-phase would charge one span twice and
    // is refused for the same reason `begin()` refuses a nested phase.
    void sub_begin(PhaseId p);
    void sub_end(PhaseId p);

    // True when `p` is the phase currently open. The sub-phase guard uses it to stay inert where a
    // sub-phase has no owner in this run (a draft submitted outside a metered round), which is the one
    // case where silence is honest rather than a missing row.
    bool phase_open(PhaseId p) const { return open_phase_ == p; }

    // A one-row step taken where a round could have been: counted, never metered.
    void note_plain_step() { plain_steps_++; }

    // Reconciles sum against wall. Call once, after the generation loop, before describing.
    void finalize();
    bool ok() const { return ok_; }
    double coverage() const { return coverage_; }
    Role role() const { return role_; }

    // The measured cost of ONE empty `hipDeviceSynchronize()` on this run's own queue, from
    // `stageclock::calibrate_idle_sync`. Printed on the instrument line because a `sync_added=1` phase
    // cannot be read without it: below this floor a phase's milliseconds are the wait, not the work.
    void set_idle_sync_ms(double ms) { idle_sync_ms_ = ms; }

    // The `MTPMETER ...` lines: one per phase of this role, then the instrument's own line. No trailing
    // newline — the driver prints it before `RESULT`, which must stay the run's last line.
    std::string describe() const;
    // The fields appended to the run's OWN `RESULT` line, so the last line of stdout stays ONE line that
    // names both the run and the meter: `... meter=ok rounds=N sum_ms_per_round=… wall_ms_per_round=…
    // coverage=…`. A second `RESULT` line would make every `grep -m1 '^RESULT'` harness in the tree read
    // the wrong one, so the summary rides the line it summarizes.
    std::string result_fields() const;

private:
    struct Rec {
        double ms = 0.0;
        long long runs = 0;
    };
    double sum_ms() const;
    bool in_band(double coverage) const;

    Role role_;
    double window_ms_ = 0.0;
    long long windows_ = 0;
    bool window_open_ = false;
    std::chrono::steady_clock::time_point window_t0_{};
    std::chrono::steady_clock::time_point phase_t0_{};
    std::chrono::steady_clock::time_point sub_t0_{};   // its own, or a breakdown would re-start the clock
                                                       // of the phase that owns it and shorten that phase
    PhaseId open_phase_ = kPhaseCount;   // kPhaseCount = none open
    PhaseId open_sub_ = kPhaseCount;     // the same, for the one sub-phase slot
    bool in_round_ = false;
    long long rounds_ = 0;
    long long plain_steps_ = 0;
    long long syncs_added_ = 0;
    long long sync_errors_ = 0;
    std::string sync_err_;
    double idle_sync_ms_ = 0.0;   // `stageclock::calibrate_idle_sync`, in ms
    bool ok_ = false;
    double coverage_ = 0.0;
    double wall_per_round_ = 0.0;
    double sum_per_round_ = 0.0;
    Rec recs_[kPhaseCount];
};

// The armed meter, or null. The guards read it once per construction, so an unarmed instrument costs one
// null-pointer read per phase — the `stageclock::note()` discipline.
void arm(Meter* m);
Meter* armed();

// RAII, so a `throw` on a guard path cannot leave a phase open and mis-charge the next one. Both guards
// are inert when no meter is armed; `PhaseGuard` is also inert when no window is open, which is what lets
// the accept loop wrap a draft call that plain iterations make too. `SubPhaseGuard` is inert in addition
// when the phase its registry row names as its parent is not the open one.
class WindowGuard {
public:
    WindowGuard();
    ~WindowGuard();
    WindowGuard(const WindowGuard&) = delete;
    WindowGuard& operator=(const WindowGuard&) = delete;

private:
    Meter* m_;
};

class PhaseGuard {
public:
    explicit PhaseGuard(PhaseId p);
    ~PhaseGuard();
    PhaseGuard(const PhaseGuard&) = delete;
    PhaseGuard& operator=(const PhaseGuard&) = delete;

private:
    Meter* m_;
    PhaseId p_;
};

// One breakdown span inside an enclosing phase (`Kind::kSubphase`). Same discipline as `PhaseGuard`, and
// inert unless a meter is armed, a window is open, and the phase its registry row names as its parent is
// the one open.
class SubPhaseGuard {
public:
    explicit SubPhaseGuard(PhaseId p);
    ~SubPhaseGuard();
    SubPhaseGuard(const SubPhaseGuard&) = delete;
    SubPhaseGuard& operator=(const SubPhaseGuard&) = delete;

private:
    Meter* m_;
    PhaseId p_;
};

}  // namespace engine::mtpmeter
