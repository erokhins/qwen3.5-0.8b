/*
 * prototype/mtp_draft/td_host.h — the host-side data the rig shares between main() and the mode
 * implementations. Kept in one header so `td_modes.cpp` (the only TU that includes
 * `mtp_draft_kernels.h`, whose `__global__` definitions must have exactly one home) and `main` agree on
 * the shape of the block plan without a second spelling of it.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Defined in the launcher TUs' shared header (`td_launchers.h`); declared here so main() can set the
// sweep knob without pulling the device headers into the host TU.
extern long td_launches;
extern int td_rpb_proj;

// ── the derived work census ──────────────────────────────────────────────────────────────
// D2 priced the block with "~18 launches" and said in terms that the number was not a measured census
// (prototype/mtp_gatezero/MTP-GATEZERO.md, the same file that flags staging as the thing that decides
// whether a rig measures the block or measures itself). The rig therefore COUNTS its launches at the
// launch sites, the way `engine/launch_stats.h` does, and asserts the total against this plan: one draft
// step is 14 submissions (DRAFT-HANDOVER.md §0) — a plan the counter is compared against, not a plan the
// counter is assumed to satisfy. The tensor and byte constants are the census in
// `prototype/mtp_gatezero/mtp_census.json` (15 tensors, 40,905,728 B), restated here as an assertion.
namespace tdplan {
constexpr int kLaunchesPerDraftStep = 14;
constexpr int kBlockTensors = 15;
constexpr long long kBlockBytes = 40905728ll;
}  // namespace tdplan

struct Blob {
    std::string name;
    long long offset = 0, bytes = 0;
    int rank = 0;
    long long dim[3] = {0, 0, 0};
};

// The block arena, with a poison band on each side. `dev` is between them, so a store that runs off the
// end of the block lands in poison and is counted, not absorbed.
struct Plan {
    std::vector<Blob> blobs;
    long long total = 0;
    uint16_t* dev = nullptr;
    uint32_t* guard_lo = nullptr;
    uint32_t* guard_hi = nullptr;
    long guard_words = 4096;
    long guard_left() const;             // poison words surviving in both guards
};

struct StepIn {
    long long y_next = 0;
    long pos = 0;
};

struct Opts {
    std::string work, out, mode = "chain";
    int k = 1, repeat = 1;
    bool dump = false;
    bool time_burst = false;      // main agent only: this rig prints no ms without this flag
    std::string fc_order = "eh", h_source = "post";
    long pos_shift = 0, pos_last = 0;
    std::string perturb;
};

// Filled by main(), consumed by the modes.
struct Inputs {
    std::vector<StepIn> steps;
    uint16_t* h_post = nullptr;
    uint16_t* h_pre = nullptr;
    uint16_t* table = nullptr;
    long long vocab = 0;
};

void mode_chain(const Opts&, const Plan&, const Inputs&);
void mode_launch(const Opts&);
void mode_witness(const Opts&, const Plan&);
void mode_xcheck(const Plan&);
void mode_stagecheck(const Plan&, const Inputs&);

// Poison helpers, defined in td_modes.cpp (the TU that owns mtp_draft_kernels.h).
void td_poison_range(uint32_t* d, long words);
long td_count_poison(const uint32_t* d, long words);
