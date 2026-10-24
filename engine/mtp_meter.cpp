// The per-phase MTP-round meter `engine/mtp_meter.h` documents. Read that first: it owns WHY a second
// instrument exists beside `stage_clock.h`, why every boundary is a host clock pair rather than a HIP
// event, which phases therefore inflate, and which phases today's structure cannot meter at all.
#include "engine/mtp_meter.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>

#include "engine/stage_clock.h"   // calibrate_idle_sync(), reused for this instrument's own floor

namespace engine::mtpmeter {
namespace {

Meter* g_meter = nullptr;

// The registry, in `PhaseId` order; the static_assert is what stops the enum and this table becoming two
// lists that drift. Notes are compact and space-free so a line stays field-wise greppable —
// `prototype/mtp_accept/METER-PLAN.md` carries the prose version of every row, including what each one
// can under- or over-report.
const PhaseSpec kSpecs[] = {
    // clang-format off
    {Role::kVerify, "snapshot_take",   Kind::kNativeBlock, "take_snapshot:_the_two_blocking_hipMemcpy_D2D_of_sizes()_bytes_(19.27_MiB_at_this_model).Already_blocking_today,_so_sync_added=0_and_the_line_bounds_its_own_device_work"},
    {Role::kVerify, "shadow_seed",     Kind::kNativeBlock, "restore_snapshot:_checkpoint_to_shadow,_same_bytes._This_is_what_a_reject_would_cost_and_the_side_pass_pays_unconditionally"},
    {Role::kVerify, "witness",         Kind::kNativeBlock, "the_--verify-guard_and_--mtp-midsplit_downloads_and_byte-compares._Reads_0.000_with_both_off,_which_is_the_only_arm_a_timing_pass_should_use:_it_is_tens_of_MiB_of_D2H_per_round"},
    {Role::kVerify, "lane_save",       Kind::kNativeBlock, "save_each_body_row's_KV_lane_out_of_the_live_cache:_24_blocking_512B_hipMemcpy_D2D_per_row_(6_attn_layers_x_2_kv_heads_x_2_buffers).Launch-count_bound,_not_byte_bound,_so_it_is_the_phase_a_fused_staging_kernel_targets"},
    {Role::kVerify, "lane_revert",     Kind::kNativeBlock, "put_those_lanes_back:_same_count_of_copies,_other_direction"},
    {Role::kVerify, "logits_d2h",      Kind::kNativeBlock, "the_blocking_vocabulary_download:_kRows_x_kVocab_F32_(1.99_MiB_here)_into_a_host_buffer._This_is_the_side_pass's_vocabulary_tax_and_the_accept_loop_has_none_of_it"},
    {Role::kVerify, "argmax_report",   Kind::kHostOnly,    "host_argmax_over_the_downloaded_rows._Pure_host_work,_so_no_wait_and_no_inflation;_2_rows_x_248320_f32_is_a_memory-bound_host_loop,_not_free"},
    {Role::kAccept, "mid_checkpoint",  Kind::kNotMetered,  "NOT_A_SEPARATE_COPY_AT_THIS_LEVEL:_2_extra_submissions_per_GDN_layer_PER_MID-CHECKPOINT_SLOT_inside_submit_wave_body_(conv_mid_derivation_plus_the_state_capture),and_the_slot_count_is_the_body's_rows_minus_1:_1_slot_at_k=1,_2_at_k=2._Bounding_it_would_need_18_waits_inside_the_call_whose_pipelining_is_the_thing_being_priced,_so_it_lives_inside_phase=body._Price_it_as_role=verify's_body_with_checkpoint_MINUS_body_under_--mtp-midsplit_1"},
    {Role::kAccept, "verdict_kernel",  Kind::kSyncAdded,   "mtp_accept_verdict_over_the_round's_logits_rows_on_device_(2_rows_at_k=1,_3_at_k=2)._INFLATES:_submissions_only_today,_the_meter's_wait_makes_the_argmax_serial._Its_real_cost_is_bounded_below_by_this_and_above_by_the_readback_it_replaces"},
    {Role::kAccept, "verdict_readback", Kind::kNativeBlock, "the_verdict_word,_blocking_D2H:_4_ints_at_k=1_(16_B),_7_at_k=2_(28_B)._This_path_carries_NO_logits_D2H_at_all:_the_vocabulary_never_leaves_the_device._The_16_B_line_beside_verify's_logits_d2h_IS_the_accept_loop's_traffic_argument"},
    {Role::kAccept, "accept_path",     Kind::kHostOnly,    "RAN_ON_AN_ACCEPT_ONLY:_decode_the_verdict_words_and_check_the_candidate_echoes._An_accept_restores_nothing,_so_a_near-zero_here_is_the_expected_result_and_not_a_missing_phase"},
    {Role::kAccept, "rollback",        Kind::kSyncAdded,   "RAN_ON_A_REJECT_ONLY:_two_bulk_D2D_restores_of_ONE_mid-checkpoint_slot_(19.27_MiB_here,_whichever_slot_the_accepted_prefix_stopped_at;_a_k=2_depth-2_reject_restores_slot_1,_not_slot_0)._INFLATES_HARD:_these_are_hipMemcpyAsync_today_and_the_round_lets_them_overlap_the_next_draft,_so_the_meter's_wait_serializes_a_copy_the_real_round_hides._Read_with_accept_rate,_never_alone"},
    {Role::kAccept, "lane_bookkeeping", Kind::kHostOnly,   "the_§08.6_invariant_check_and_the_draft_lane's_length-field_truncate._Moves_0_bytes;_host-only,_so_no_wait"},
    {Role::kAccept, "draft",           Kind::kSyncAdded,   "the_NEXT_round's_draft_round_submissions,_ending_in_the_draft's_own_blocking_vocabulary_D2H._Sync_added_so_the_draft's_head_pass_does_not_land_on_whichever_phase_happens_to_block_next;_the_D2H_already_would,_so_the_inflation_is_small"},
    // The two BREAKDOWNS of `draft`, added so the vocabulary crossing `ROUND-TAX.md` §1 names is priced
    // before it is optimized rather than after. Both are excluded from the sum (`Kind::kSubphase`), so
    // they cannot move coverage and cannot double-charge: whatever they print, `draft` already printed.
    // After the device argmax lands, `draft_vocab_d2h` bills a small status word instead of 993,280 B and
    // `draft_argmax_host` loses its site — a 0.000 line with runs=0 there IS the result of that change,
    // which is why the rows stay in the registry instead of being deleted with the host scan.
    {Role::kAccept, "draft_vocab_d2h",  Kind::kSubphase,
     "BREAKDOWN_inside_phase=draft._The_draft's_blocking_download_of_its_own_logits_row:_today_kVocab_F32_=_993,280_B_per_round,_which_is_the_second_of_the_two_vocabulary_crossings_ROUND-TAX.md_§1_names_(the_first_is_verify's_logits_d2h)._sync_added=0_because_the_copy_already_blocks._After_the_device_argmax_this_line_bills_one_small_status_word_and_a_0.000_there_is_the_result_not_a_missing_phase",
     kDraft},
    {Role::kAccept, "draft_argmax_host", Kind::kSubphase,
     "BREAKDOWN_inside_phase=draft_and_inside_the_span_draft_vocab_d2h_closes._The_host_§03.4_argmax_over_the_downloaded_row:_248,320_f32_scanned_on_one_CPU_core,_pure_host_work,_sync_added=0._Removed_by_resolving_the_argmax_on_device;_runs=0_means_the_site_is_gone,_not_that_it_was_missed",
     kDraft},
    {Role::kRoleCount, "body",                  Kind::kSyncAdded, "submit_wave_body_at_the_round's_row_count._INFLATES:today_the_body's_submissions_flow_into_the_head_launch_and_the_next_wait,_so_a_wait_at_its_close_removes_pipelining_the_engine_relies_on._It_also_carries_the_mid-checkpoint's_extra_submissions_when_one_is_armed"},
    {Role::kRoleCount, "body_with_checkpoint",  Kind::kSyncAdded, "the_--mtp-midsplit_1_second_body:_same_anchor,_same_tokens,_WITH_the_mid-checkpoint_armed._MINUS_phase=body_it_is_that_checkpoint's_price_measured_in_ONE_window_at_ONE_anchor,_which_is_the_only_honest_way_to_ask"},
    {Role::kRoleCount, "head",                  Kind::kSyncAdded, "one_lm_head_batched_xbf16_pass_over_the_485_MiB_tied_table,_serving_the_round's_rows_(2_at_k=1,_3_at_k=2,_the_second_instantiation_of_the_same_kernel_template)._INFLATES_in_a_pipelined_round_by_the_wait,_but_NOT_for_free_in_a_serial_one:_on_the_verify_path_the_logits_D2H_already_waited_for_this_launch_on_the_same_stream,_so_the_wait_it_replaces_was_already_paid"},
    {Role::kRoleCount, "report_prints",         Kind::kHostOnly,  "the_run's_own_per-round_lines_(printf_plus_explicit_fflush)._Named_rather_than_left_in_overhead_so_a_reader_can_see_whether_the_reporting_is_the_cost;_host-only,_no_wait"},
    {Role::kRoleCount, "overhead",              Kind::kDerived,   "DERIVED,_not_timed:_the_round's_windows_minus_every_measured_phase._Round_cost_that_no_site_owns._Excluded_from_the_sum,_which_is_what_makes_coverage_a_real_check_instead_of_1.000_by_construction"},
    {Role::kRoleCount, "summary",               Kind::kDerived,   "the_meter's_own_verdict_line:_wall,_sum,_coverage_and_the_sync_floor._Excluded_from_the_sum_like_overhead"},
    // clang-format on
};
static_assert(sizeof(kSpecs) / sizeof(kSpecs[0]) == kPhaseCount,
              "the phase registry and the PhaseId enum must be one list, not two that can drift");

const PhaseId kVerifyOrder[] = {
    kSnapTake, kShadowSeed, kWitness, kLaneSave, kBody, kBodyCkpt, kLaneRevert, kHead,
    kLogitsD2H, kArgmax, kReportPrints, kOverhead, kSummary,
};
const PhaseId kAcceptOrder[] = {
    kBody, kMidCheckpoint, kHead, kVerdict, kReadback, kAcceptPath, kRollback, kLaneBookkeeping,
    kDraft, kDraftVocabD2H, kDraftArgmaxHost, kReportPrints, kOverhead, kSummary,
};

// The wait. Unlike `stage_clock`'s drain this does NOT throw: a faulted queue must still let the run print
// its result line, so the failure is counted and named on the summary and in `meter=`.
void drain(std::string& first_err, long long& errors) {
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        if (first_err.empty()) first_err = hipGetErrorString(e);
        errors++;
    }
}

double ms_since(const std::chrono::steady_clock::time_point& from) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
}

bool is_role_row(Role mine, Role row) { return row == mine || row == Role::kRoleCount; }

}  // namespace

const char* role_name(Role r) {
    switch (r) {
        case Role::kVerify: return "verify";
        case Role::kAccept: return "accept";
        case Role::kRoleCount: break;
    }
    return "invalid";
}

const PhaseSpec* specs() { return kSpecs; }

Meter::Meter(Role role) : role_(role) {
    // The instrument's own floor, printed rather than assumed: an idle `hipDeviceSynchronize()` is the
    // minimum cost of every added boundary, and `stage_clock` calibrates it the same way.
    idle_sync_ms_ = stageclock::calibrate_idle_sync(64) / 1000.0;
}

void Meter::open_window() {
    if (window_open_) {
        throw std::runtime_error("mtp meter: a window is already open; windows must be disjoint or the "
                                 "wall double-counts and coverage means nothing");
    }
    window_open_ = true;
    window_t0_ = std::chrono::steady_clock::now();
}

void Meter::close_window() {
    if (!window_open_) {
        throw std::runtime_error("mtp meter: close_window called with no window open");
    }
    window_ms_ += ms_since(window_t0_);
    windows_++;
    window_open_ = false;
}

void Meter::round_begin() {
    if (in_round_) throw std::runtime_error("mtp meter: nested rounds");
    in_round_ = true;
}

void Meter::round_end() {
    if (!in_round_) throw std::runtime_error("mtp meter: round_end with no round open");
    if (window_open_) {
        throw std::runtime_error("mtp meter: round_end with a window still open; the wall would be "
                                 "missing a round's last span");
    }
    in_round_ = false;
    rounds_++;
}

void Meter::begin(PhaseId p) {
    if (open_phase_ != kPhaseCount) {
        throw std::runtime_error(std::string("mtp meter: phase ") + kSpecs[p].name
                                 + " opened while " + kSpecs[open_phase_].name
                                 + " was still open; a nested phase would charge the window twice");
    }
    open_phase_ = p;
    phase_t0_ = std::chrono::steady_clock::now();
}

void Meter::end(PhaseId p) {
    if (open_phase_ != p) {
        throw std::runtime_error(std::string("mtp meter: end(") + kSpecs[p].name
                                 + ") with no such phase open");
    }
    open_phase_ = kPhaseCount;
    // The wait comes BEFORE the clock read, or the phase would close before its own work was done — and it
    // is added ONLY for `kSyncAdded`. A `kNativeBlock` phase already ends in a call that blocks (that is
    // what makes its host pair a bound), and adding a second wait there would be the one way this
    // instrument could silently change the thing it is measuring. `syncs_added=` counts what was added.
    if (kSpecs[p].kind == Kind::kSyncAdded) {
        syncs_added_++;
        drain(sync_err_, sync_errors_);
    }
    Rec& r = recs_[p];
    r.ms += ms_since(phase_t0_);
    r.runs++;
}

void Meter::sub_begin(PhaseId p) {
    const PhaseSpec& s = kSpecs[p];
    if (s.kind != Kind::kSubphase) {
        throw std::runtime_error(std::string("mtp meter: sub_begin(") + s.name
                                 + ") on a phase the registry does not declare a breakdown; a sub-phase"
                                     " is out of the sum by kind, so a real phase opened this way would"
                                     " vanish from coverage");
    }
    if (open_phase_ != s.parent) {
        // The guard stays inert rather than reaching here unless its parent is open, so arriving with the
        // wrong parent means a call site that bypassed the guard. Charging the span to whatever happened
        // to be open is the mis-attribution this instrument refuses everywhere else.
        throw std::runtime_error(std::string("mtp meter: sub_begin(") + s.name + ") declares its parent as "
                                 + (s.parent == kPhaseCount ? std::string("<none>") : kSpecs[s.parent].name)
                                 + " but "
                                 + (open_phase_ == kPhaseCount ? std::string("<no phase>")
                                                               : kSpecs[open_phase_].name)
                                 + " is open");
    }
    if (open_sub_ != kPhaseCount) {
        throw std::runtime_error(std::string("mtp meter: sub-phase ") + s.name
                                 + " opened while " + kSpecs[open_sub_].name
                                 + " was still open; a nested breakdown would charge one span twice");
    }
    open_sub_ = p;
    sub_t0_ = std::chrono::steady_clock::now();
}

void Meter::sub_end(PhaseId p) {
    if (open_sub_ != p) {
        throw std::runtime_error(std::string("mtp meter: sub_end(") + kSpecs[p].name
                                 + ") with no such sub-phase open");
    }
    open_sub_ = kPhaseCount;
    // No drain, and that is not an oversight: a sub-phase lives inside a span whose close is already made
    // meaningful, and its own two sites are a blocking copy and a host loop (`ROUND-TAX.md` §1). Adding a
    // wait here would serialize inside a phase the enclosing meter already bounds.
    Rec& r = recs_[p];
    r.ms += ms_since(sub_t0_);
    r.runs++;
}

double Meter::sum_ms() const {
    double total = 0.0;
    for (int i = 0; i < kPhaseCount; ++i) {
        const Kind k = kSpecs[i].kind;
        // `kSubphase` is out of the sum for the same reason `kDerived` is: its milliseconds belong to a
        // phase that is already in there, so adding them would bill one span twice and push coverage past
        // the window it lives in. `meter_table.awk` excludes the same rows for the same reason, and its
        // selftest reddens a log where that exclusion is broken.
        if (k == Kind::kNotMetered || k == Kind::kDerived || k == Kind::kSubphase) continue;
        total += recs_[i].ms;
    }
    return total;
}

void Meter::finalize() {
    if (rounds_ > 0) {
        wall_per_round_ = window_ms_ / static_cast<double>(rounds_);
        sum_per_round_ = sum_ms() / static_cast<double>(rounds_);
        coverage_ = window_ms_ > 0.0 ? sum_ms() / window_ms_ : 0.0;
    }
    ok_ = rounds_ > 0 && sync_errors_ == 0 && in_band(coverage_);
}

bool Meter::in_band(double coverage) const { return coverage >= 0.8 && coverage <= 1.2; }

std::string Meter::describe() const {
    const PhaseId* order = role_ == Role::kVerify ? kVerifyOrder : kAcceptOrder;
    const size_t n = role_ == Role::kVerify ? sizeof(kVerifyOrder) / sizeof(PhaseId)
                                            : sizeof(kAcceptOrder) / sizeof(PhaseId);
    const double rounds = rounds_ > 0 ? static_cast<double>(rounds_) : 0.0;
    const double measured = sum_ms();
    std::string out;
    char buf[1200];

    for (size_t i = 0; i < n; ++i) {
        const PhaseId p = order[i];
        const PhaseSpec& s = kSpecs[p];
        if (!is_role_row(role_, s.role)) continue;   // one shared table, two printed shapes
        if (s.kind == Kind::kNotMetered) {
            // No ms_total= field: printing a number for a phase that has no boundary is exactly how a
            // meter would lie. The line exists so the gap is visible and greppable.
            std::snprintf(buf, sizeof buf, "MTPMETER role=%s phase=%s status=not_metered note=%s\n",
                          role_name(role_), s.name, s.note);
            out += buf;
            continue;
        }
        double ms = recs_[p].ms;
        long long runs = recs_[p].runs;
        if (p == kOverhead) {
            ms = window_ms_ - measured;
            runs = rounds_;
        } else if (p == kSummary) {
            ms = measured;
            runs = rounds_;
        }
        const double per_round = rounds > 0 ? ms / rounds : 0.0;
        const double per_run = runs > 0 ? ms / static_cast<double>(runs) : 0.0;
        // TWO denominators, printed under names that say which is which, because a round pays some phases
        // more or less than once and a reader who mixes these up gets a wrong per-round cost quietly:
        //   rounds= is THE RUN'S ROUND COUNT, the denominator `ms_per_round=` actually divided by, and it is
        //           the same number on every phase line of a run (and on `RESULT`), so a phase that fires on
        //           one reject in six reads as the diluted per-round cost it is — which is what "what does a
        //           round cost" asks. `ms_per_run=` is the price of one execution of the site.
        //   runs=   is the number of SITE EXECUTIONS, so `runs > rounds` means more than one print/copy per
        //           round (report_prints) and `runs < rounds` means a branch not every round takes (rollback).
        // It deliberately does NOT print "how many rounds charged this phase", because that number divided
        // into ms_total is the per-execution price wearing a round's label.
        std::snprintf(buf, sizeof buf,
                      "MTPMETER role=%s phase=%s ms_total=%.4f rounds=%lld ms_per_round=%.4f "
                      "sync_added=%d runs=%lld ms_per_run=%.4f",
                      role_name(role_), s.name, ms, (long long)rounds_, per_round,
                      s.kind == Kind::kSyncAdded ? 1 : 0, (long long)runs, per_run);
        out += buf;
        if (s.kind == Kind::kDerived) out += " status=derived";
        if (s.kind == Kind::kSubphase) {
            // `inside=` names the phase whose milliseconds this row breaks down, so the line is readable
            // on its own and a reader never has to know that `draft_vocab_d2h` lives under `draft`.
            out += " status=subphase inside=";
            out += kSpecs[s.parent].name;
        }
        if (p == kSummary) {
            std::snprintf(buf, sizeof buf, " wall_ms_per_round=%.4f coverage=%.4f", wall_per_round_,
                          coverage_);
            out += buf;
        }
        out += " note=";
        out += s.note;
        out += "\n";
    }

    std::snprintf(buf, sizeof buf,
                  "MTPMETER role=%s phase=instrument windows=%lld rounds=%lld plain_steps=%lld "
                  "syncs_added=%lld idle_sync_us=%.2f sync_errors=%lld "
                  "note=windows_are_the_wall_and_are_disjoint_by_construction;idle_sync_us_is_the_floor_"
                  "of_one_added_sync;plain_steps_are_one_row_steps_taken_instead_of_a_round_and_are_not_"
                  "metered\n",
                  role_name(role_), windows_, (long long)rounds_, plain_steps_, syncs_added_,
                  idle_sync_ms_ * 1000.0, sync_errors_);
    out += buf;
    if (!sync_err_.empty()) {
        std::snprintf(buf, sizeof buf, "MTPMETER role=%s phase=instrument sync_error=%s\n",
                      role_name(role_), sync_err_.c_str());
        out += buf;
    }
    while (!out.empty() && out.back() == '\n') out.pop_back();   // driver prints it before RESULT
    return out;
}

std::string Meter::result_fields() const {
    char buf[256];
    if (rounds_ == 0) {
        std::snprintf(buf, sizeof buf, " meter=no_rounds rounds=0 plain_steps=%lld", plain_steps_);
        return buf;
    }
    std::snprintf(buf, sizeof buf,
                  " meter=%s rounds=%lld sum_ms_per_round=%.4f wall_ms_per_round=%.4f coverage=%.4f",
                  ok_ ? "ok" : (sync_errors_ > 0 ? "sync_failed" : "out_of_band"),
                  (long long)rounds_, sum_per_round_, wall_per_round_, coverage_);
    return buf;
}

void arm(Meter* m) { g_meter = m; }
Meter* armed() { return g_meter; }

WindowGuard::WindowGuard() : m_(nullptr) {
    if (Meter* m = armed()) {
        m->open_window();
        m_ = m;
    }
}

WindowGuard::~WindowGuard() {
    if (m_) m_->close_window();
}

PhaseGuard::PhaseGuard(PhaseId p) : m_(nullptr), p_(p) {
    Meter* m = armed();
    if (m != nullptr && m->window_open()) {
        m->begin(p);
        m_ = m;
    }
}

PhaseGuard::~PhaseGuard() {
    if (m_) m_->end(p_);
}

SubPhaseGuard::SubPhaseGuard(PhaseId p) : m_(nullptr), p_(p) {
    Meter* m = armed();
    // Inert, in this order: no meter, no window, or the phase this breakdown belongs to is not the one
    // open. The last case is not an error — a draft submitted outside a metered round (the side pass, a
    // `--mtp-draft-k` ladder row) has no round window to charge it to — and the phase line's own `runs=`
    // is what shows a breakdown that never got an owner anywhere in the run.
    if (m != nullptr && m->window_open() && m->phase_open(kSpecs[p].parent)) {
        m->sub_begin(p);
        m_ = m;
    }
}

SubPhaseGuard::~SubPhaseGuard() {
    if (m_) m_->sub_end(p_);
}

}  // namespace engine::mtpmeter
