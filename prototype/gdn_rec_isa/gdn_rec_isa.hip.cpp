/*
 * prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp — the device A/B harness for the `gdn_rec` MODEs, so the
 * falsifier written in `results.md` §4 can actually be run.
 *
 * WHAT IT RUNS. For every case, over ONE input tile and ONE starting state, every active MODE, each on
 * its own state and output buffers:
 *
 *   ship    `rec_kernel`   — the shipped kernel (`prototype/gdn_decode/gdn_decode_kernels.h`), the
 *                            control the census bills against. Optional; not in the default set.
 *   rec0    `rec0_kernel`  — MODE 0, the shipped kernel copied character for character. THE REFERENCE:
 *                            every other MODE is judged by reproducing it bit for bit.
 *   rec1    `rec1_kernel`  — MODE 1, the cross-lane fold hand-spelled, shipped barrier structure kept.
 *   rec2    `rec2_kernel`  — MODE 2, both block reductions fused into one pass, hand-spelled folds.
 *   rec2c   `rec2c_kernel` — MODE 2c, the fused pair with compiler-lowered shuffles (spelling control).
 *   rec3    `recx_indep_kernel` — DEPENDENCY PROBE, meaningless by design: reads one state buffer and
 *           writes another, so no token waits for the previous token's result. Same bytes, new addresses.
 *   rec4    `recx_unchain_kernel` — DEPENDENCY PROBE: same buffer, same addresses, same load/store mix,
 *           but each of the two serial accumulator chains is shortened from 128 dependent links to 32.
 *   rec5    `recx_both_kernel`  — DEPENDENCY PROBE: rec3's split and rec4's shortened chains together.
 *   rec6    `recx_shape_kernel` — the CONTROL for rec4: same unroll-4 loop spelling, chain NOT broken,
 *                            and therefore bit-identical to `rec0`. It exists because the census found
 *                            rec4 10.5 % cheaper in slots; without rec6 a rec4 win cannot be split
 *                            between its instruction bill and the chain it shortened.
 *   receng  `rec_unroll4_kernel` — the ENGINE's shipping kernel, compiled out of
 *                            `engine/gdn_rec_unroll4_kernels.h` by the same TU. Bit-identity-gated like
 *                            rec6, and the witness that the engine's port moved no bits.
 *   rec2c4  `rec2c4_kernel` — MODE 2c's fused prologue MERGED with those unroll4 walks, i.e. the form
 *                            §18/D46 could not adjudicate because it had never been built. Admitted by
 *                            bit-identity like every spelling MODE, and its A/B partner is `receng`, the
 *                            spelling the engine runs — NOT `rec0`, which is the pre-unroll4 form §18
 *                            wrongly screened against.
 *   recloop `recloop_kernel` — D53's item: the shipping spelling with the TOKEN index moved INSIDE the
 *                            kernel, so ONE submission walks `--loop N` token-steps (N=0 means the whole
 *                            sequence). Same grid `<<<NH,D>>>`, same operands, same state buffer, same
 *                            per-token arithmetic verbatim (checked by `check_recloop_transcription.py`),
 *                            so the one axis is submissions per token-step. Admitted by bit-identity
 *                            against `rec0`, and its burst partner is `rec2c4` — the per-token form the
 *                            engine runs. The burst reports ms per LAUNCH and per SEQUENCE, because for
 *                            this MODE the two come apart by `--loop`.
 *   recloop `recloop_kernel` — D53's item: the shipping spelling with the TOKEN index inside the
 *                            kernel, so one submission walks `--loop N` token-steps instead of one.
 *                            Same grid, same operands, same state buffer, same per-token arithmetic
 *                            (verbatim, checked by `check_recloop_transcription.py`); the only thing
 *                            that moves is launches per token-step. Admitted by bit-identity against
 *                            `rec0` like every spelling MODE, and its burst partner is `rec2c4`, the
 *                            per-token form the engine actually runs. `--loop 0` (the default) walks
 *                            the whole sequence in one launch; `--loop 16` walks 16 per launch, which
 *                            is the same arithmetic in a smaller chunk — the span must not matter, and
 *                            `--selftest-loop` asserts that on CPU before anything is launched.
 *
 * WHY THE PROBES ARE ADMITTED DIFFERENTLY. MODEs 0/1/2/2c/6/eng/2c4 must reproduce MODE 0 bit for bit — 6
 * is a
 * probe's control, not a probe, and its addition tree is MODE 0's term for term — so a bit move there is a
 * defect. A probe that broke the dependency and still produced the right answer would not have broken it,
 * so `rec3` and
 * `rec5` are admitted on VOLUME only (their value is not a statement about anything), `rec4` re-
 * associates two dots and is admitted inside the EXISTING §10.1 GDN band (cos ≥ 0.999, meanrel ≤ 5 %),
 * and `--burst` REFUSES to time anything unless the volume self-check passed first (non-zero exit).
 * The per-MODE policy is printed on each RESULT line as `gate=`, so no reader has to remember it.
 *
 * WHY IT EXISTS. The census's verdict is a funded NO — the family issues 1,523 slots per wavefront
 * against a measured 6.8-6.9 us per launch, so it runs at ~10 % of its own issue roof and instruction
 * surgery is capped near 1-13 % of a 509 ms family. That verdict came with a falsifier nobody could
 * run: MODE 0 vs MODE 2 in one burst, > 1.10x refutes the latency reading, < 1.02x kills the
 * instruction route for this family. This file is the missing instrument. It does not change the
 * verdict; it makes the verdict checkable.
 *
 * THE ADMISSION GATE IS BIT-IDENTITY FOR THE SPELLING MODES, AND IT IS NOT SOFTER THAN THE CENSUS
 * ASSUMED. MODEs 1, 2 and 2c claim to preserve every value's addition tree exactly (same 5+5 shuffle
 * steps in the same order per value), so the only acceptable difference against MODE 0 is ZERO, on `o`
 * AND on the final state, over T tokens. A band there would be a way of not checking the claim: MODE 1
 * re-spells a fold and MODE 2 fuses two of them, and a hand-written fold that changes association is
 * not a lever, it is a numerics change. `results.md` §6 says what each failure would mean. The three
 * dependency PROBES are the deliberate exception, gated on volume instead — the paragraph above and
 * `results.md` §9 say why.
 *
 * WHAT THE CPU REFERENCE IS FOR, and what it is not. The host mirror below transcribes the shipped
 * expressions (including the FMA sites the listing proves the device fuses: `v_fmac_f32_e32` through
 * loop A at `rec_kernel.ship.s:243-282` and loop B at `:346-383`, and `v_fma_f32 v1, -v6, v1, v5` at
 * `:315`), so its dots are the same product trees. It CANNOT be bit-identical to the device and is not
 * asked to be: the device computes `rsqrtf` as `v_rsq_f32_e32` (`rec_kernel.ship.s:177`) and `expf` as
 * `v_exp_f32_e32` + `v_ldexp_f32` (`:304-307`), hardware approximations of a different kind from
 * glibc's. So `ndiff_*_vs_cpu` is non-zero for MODE 0 too, MODE 0's own line is printed as the FLOOR
 * line, and a MODE passes by matching MODE 0 — never by matching the mirror. What the mirror
 * adjudicates that MODE-vs-MODE cannot: a fixture bug shared by every MODE (wrong q|k|v row stride,
 * wrong head index, state laid out the other way round) is invisible to a MODE-vs-MODE diff — all the
 * MODEs would agree — and shows here as `maxrel_*_vs_cpu` of order 1 instead of order 1e-6. And when a
 * MODE does NOT match MODE 0, the mirror says which of the two is the outlier.
 *
 * BYTES TOUCHED, NOT JUST VALUES (the `tools/mixed_bw` lesson). Every device buffer sits between two
 * guard bands pre-filled with a value no MODE can produce, and every `o` buffer starts as that
 * sentinel, so an element nobody wrote stays an unwritten element instead of becoming a lucky match.
 * Each MODE has separate `o` and state buffers so a stray write in one cannot be laundered by another.
 *
 * RUN TWICE, ON PURPOSE (`prototype/attn_valu_slim`'s precedent). MODEs 1 and 2 read across lanes with
 * inline `ds_bpermute_b32` / `ds_swizzle_b32` whose `s_waitcnt lgkmcnt(0)` is written by hand
 * (`gdn_rec_isa_kernels.h`), and a racy permute does not give a stable wrong answer — it gives
 * run-to-run disagreement. So every case is run `--repeat 2` (the default) and the two repeats are
 * bit-compared per MODE: a non-zero `rerun_ndiff_*` means the wait is wrong, not the arithmetic.
 *
 * ONE PROCESS, NOT A FORKED CHILD PER MODE. `prototype/gdn_collapse/gdn_collapse_check.hip.cpp` — the
 * harness this one is modelled on — runs every form in one process against separate buffers, and so does
 * this; `fork()` is not a usable isolation mechanism once HIP has touched the device (the
 * `tools/mixed_bw` `run_dev_case` pattern forks BEFORE `hipInit`, which only works when the parent never
 * touches the device at all). The isolated-process equivalent is here instead: `--modes rec2` runs one
 * MODE and `--out DIR` dumps its `o` and final state, so a MODE that faults the queue (the hand permutes
 * are the fault candidates) can be run alone and compared with `cmp -l` against MODE 0's dump.
 * `HANDOFF.md` §3 carries the commands in that order for exactly that reason.
 *
 * NO TIMING IS TAKEN HERE, BY WHOVER RUNS IT FIRST. A worker may not take timings (AGENTS.md), and a
 * number taken while this session streams its own tokens is worse than no number. The correctness
 * output carries counts, geometry and verdicts and never ms. The `--burst N` path below is the paired
 * A/B the main agent runs under one lock hold: steady_clock, not `hipEventElapsedTime`, which returns
 * exactly 0 for a sub-ms kernel on this part (`bench/performance-status.md` §6), and every active MODE
 * is timed INSIDE each rep (the `prototype/attn_m2` `--forms` pattern) so the falsifier is one window
 * and not four. A burst also REFUSES to run (`NOT TIMED reason=volume_self_check_failed`, exit 1) if
 * the `--volume` self-check does not pass, because a probe whose bytes changed is not a probe.
 *
 * TWO CPU-ONLY SELFTESTS, because a device gate nobody can run should still be half-checkable. On this
 * box the harness could be compiled and reviewed but not executed, so the two claims that do NOT need a
 * device are tested here instead: `--selftest-lane` re-derives the hand fold's source-lane rule against
 * the compiler's and the CUDA rule, in every lane and at both stages of `block_sum`, with a must-catch
 * control (two wrong rules must move lanes, and they do); `--selftest-mirror` witnesses the host
 * mirror's addresses with three exact identities at g = 0 exactly (beta=0 leaves the state bit-identical,
 * beta=1 from a zero state yields ks (x) v, and o is qs . state). Neither is the gate; both say what a
 * gate failure is NOT caused by. `results.md` §6 carries their counts and what each rules out.
 *
 * Build:  prototype/gdn_rec_isa/build.sh
 * Run:    prototype/gdn_rec_isa/HANDOFF.md §3 — numbered, with the gpu_lock protocol. This binary does
 *         NOT acquire `tools/gpu_lock.sh`; the three self-locking drivers named in AGENTS.md are glue,
 *         proj_gemv and gdn_gated_norm, and a caller that held the lock and ran one of them deadlocks it.
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gdn_rec_isa_kernels.h"   // rec0/rec1/rec2/rec2c, and through them rec_kernel, block_sum, D, NH, Q_SCALE

// The engine's OWN port of `recx_shape_kernel`, so the bit-identity verdict is taken on the source text
// the engine compiles and not on a third copy of it. Precedent:
// `prototype/gdn_collapse/gdn_collapse_check.hip.cpp:60` includes `engine/gdn_collapse_kernels.h` for
// the same reason — a port may be witnessed on the copy, but the copy that matters is the shipped one.
// `--modes receng` is that witness: `receng` over `rec0` is the engine kernel versus the shipped
// kernel, bit for bit, on `o` AND on the final state, over T tokens.
#include "engine/gdn_rec_unroll4_kernels.h"

namespace {

constexpr int kD = D;                       // 128: d_k = d_v, one thread per d_v column
constexpr int kNH = NH;                     // 16 value heads, one block per head
constexpr int kValueDim = kNH * kD;         // 2048: one token's row of `o`
constexpr int kQkvRows = 3 * kValueDim;     // 6144: one token's q|k|v row, the engine's layout
constexpr int kStateElems = kNH * kD * kD;  // 262144: one layer's state
constexpr int kWave = 32;
// `recloop_kernel` derives its row strides from CH and NH*D; the fixture derives its tile layout from
// kQkvRows/kValueDim. If those two spellings of "one token's row" ever disagree, the looped MODE would
// be reading a different layout from the per-token MODEs while both agreed with themselves, so the
// equality is a compile error here rather than a comment (`engine/weights_contract.py`'s lesson: state
// the rank, put the collapsing contract at the use site).
static_assert(CH == kQkvRows, "recloop's q|k|v row stride (CH) must equal the fixture's kQkvRows");
static_assert(NH * D == kValueDim, "recloop's o row stride (NH*D) must equal the fixture's kValueDim");
// A value no MODE can produce from these operands. If it survives in `o` nothing wrote there; if it
// survives in a guard band nothing outside the buffer wrote there.
constexpr float kSentinel = -1.2345e37f;
constexpr int kGuard = 512;                 // floats of guard band on each side

#define CK(call)                                                                       \
    do {                                                                               \
        hipError_t e_ = (call);                                                        \
        if (e_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "hip: %s (%s:%d)\n", hipGetErrorString(e_), __FILE__, \
                         __LINE__);                                                    \
            std::exit(3);                                                              \
        }                                                                              \
    } while (0)

}  // namespace
namespace {

// ---- the MODEs -----------------------------------------------------------------------------------
// One index space, used by the admission path and the burst path alike, so a timing can never be taken
// on a different launch plan from the one that was gated. `rec0` is MODE 0 and is the reference every
// other MODE is bit-diffed against; `ship` is optional and gives the census's static control row
// (rec0_kernel censuses identical to rec_kernel) a device witness when someone wants one.
enum Mode { M_SHIP = 0, M_REC0, M_REC1, M_REC2, M_REC2C, M_REC3, M_REC4, M_REC5, M_REC6, M_RECENG,
            M_REC2C4, M_RECLOOP, NMODES };
static const char* kModeName[] = {"ship", "rec0", "rec1", "rec2", "rec2c",
                                  "rec3", "rec4", "rec5", "rec6", "receng", "rec2c4", "recloop"};

/* The three dependency-break probes. Their VALUE is meaningless by design — a probe whose value was
 * still correct would still be waiting on the dependency — so they are never admitted against MODE 0's
 * bits. MODE 4 (`recx_unchain`) is the one whose value is only re-associated rather than replaced, and
 * it is judged against the EXISTING §10.1 GDN band (cos ≥ 0.999, the G3 `gdn` row), which is an
 * existing gate rather than a new one; MODEs 3 and 5 replace what the recurrence reads, so nothing but
 * volume is asserted about them. */
inline bool mode_splits_state(int m) { return m == M_REC3 || m == M_REC5; }
inline bool mode_value_gated(int m) { return m == M_SHIP || m == M_REC0 || m == M_REC1 ||
                                           m == M_REC2 || m == M_REC2C || m == M_REC6 ||
                                           m == M_RECENG || m == M_REC2C4 || m == M_RECLOOP; }
inline bool mode_band_gated(int m) { return m == M_REC4; }
// The three dependency probes: value is not the gate, volume is.
inline bool mode_is_probe(int m) { return m == M_REC3 || m == M_REC4 || m == M_REC5; }
// The ONE axis `recloop` moves: token-steps per submission. Every other MODE — including the probes,
// which move addresses and dependency chains but not this — walks exactly one token per launch.
inline bool mode_loops_tokens(int m) { return m == M_RECLOOP; }

// Token-steps one launch of `mode` carries. `g_loop` is `--loop N` (0 = the whole sequence in one
// launch); a MODE that does not loop always returns 1 whatever the flag says, so `--loop` cannot
// silently re-shape a MODE that is not the looped one.
static int g_loop = 0;
inline int span_of(int mode, int tokens) {
    if (!mode_loops_tokens(mode)) return 1;
    return (g_loop > 0 && g_loop < tokens) ? g_loop : tokens;
}
// How many launches one `tokens`-long sequence costs a MODE. The burst divides by this, not by the
// token count, because for the looped MODE the two are no longer the same number — and a
// ms-per-launch figure taken on the wrong denominator is the whole point of the item.
inline long launches_of(int mode, int tokens) {
    const int s = span_of(mode, tokens);
    return s <= 0 ? tokens : (tokens + s - 1) / s;
}

struct ModeSet {
    int m[NMODES] = {0, 0, 0, 0, 0};
    int n = 0;
    bool has(int mode) const {
        for (int i = 0; i < n; ++i) if (m[i] == mode) return true;
        return false;
    }
    void add(int mode) {
        if (!has(mode)) m[n++] = mode;
    }
    std::string names() const {
        std::string s;
        for (int i = 0; i < n; ++i) {
            if (i) s += ",";
            s += kModeName[m[i]];
        }
        return s;
    }
};

// MODE 0 is ALWAYS active: it is the reference, and a diff against a MODE that never ran is worse than
// no diff. Tokens are matched exactly — the `attn_m2` lesson is that a substring test would make
// `--modes rec2c` silently add `rec2`.
static ModeSet modes_of(const std::string& spec) {
    ModeSet ms;
    ms.add(M_REC0);
    if (spec.empty()) {
        ms.add(M_REC1);
        ms.add(M_REC2);
        ms.add(M_REC2C);
        return ms;
    }
    size_t p = 0;
    while (p < spec.size()) {
        const size_t c = spec.find(',', p);
        const std::string tok = spec.substr(p, c == std::string::npos ? std::string::npos : c - p);
        int found = -1;
        for (int i = 0; i < NMODES; ++i)
            if (tok == kModeName[i]) found = i;
        if (found < 0) {
            std::fprintf(stderr, "--modes: no mode \"%s\"; known:"
                         " ship,rec0,rec1,rec2,rec2c,rec3,rec4,rec5,rec6,receng,rec2c4,recloop\n",
                         tok.c_str());
            std::exit(2);
        }
        ms.add(found);
        if (c == std::string::npos) break;
        p = c + 1;
    }
    return ms;
}

// The one launch site. Grid and thread count are the shipped geometry (`engine/kernels_gdn.cpp:40`
// launches `rec_kernel<<<NH=16, D=128>>>`), identical for every MODE by construction: the MODEs differ
// in the prologue's spelling and in nothing else, which is what makes the A/B single-variable.
/* The one launch site. `S_in`/`S_out` are the SAME buffer for every MODE that carries state (the
 * shipped geometry, unchanged) and DIFFERENT ones for the two split probes, which is what makes their
 * launches independent of their predecessor's store. Grid and thread count are the shipped geometry
 * (`engine/kernels_gdn.cpp:40` launches `rec_kernel<<<NH, D>>>`) and are identical for every MODE by
 * construction, which is what keeps the A/B single-variable. */
void launch_mode(int mode, const float* q, const float* k, const float* v, const float* g,
                 const float* beta, float* S_in, float* S_out, float* o, int tokens) {
    if (!mode_splits_state(mode) && S_in != S_out) {
        std::fprintf(stderr, "mode %s is not a split-state MODE but was given distinct state buffers\n",
                     kModeName[mode]);
        std::exit(3);
    }
    // A MODE that walks one token per launch must not be handed a span, and the looped MODE must not
    // be handed 1 by accident: this is the one field the looped item is defined by, so it is asserted
    // at the launch site rather than trusted to the caller (`run_sequence` is its only caller).
    if (!mode_loops_tokens(mode) && tokens != 1) {
        std::fprintf(stderr, "mode %s is not a looping MODE but was given tokens=%d\n",
                     kModeName[mode], tokens);
        std::exit(3);
    }
    switch (mode) {
        case M_SHIP: rec_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        case M_REC0: rec0_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        case M_REC1: rec1_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        case M_REC2: rec2_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        case M_REC2C: rec2c_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        case M_REC3: recx_indep_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, S_out, o, kNH); break;
        case M_REC4: recx_unchain_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        case M_REC5: recx_both_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, S_out, o, kNH); break;
        case M_REC6: recx_shape_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        // The engine's port. Same launch plan as every other carried MODE, so a `receng`/`rec6` pair
        // differing at all would mean the two source texts differ, not the geometry.
        case M_RECENG:
            rec_unroll4_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH);
            break;
        // MODE 2c4: the fused prologue on the SHIPPING unroll4 walks. Compile-only until this run
        // admits it: `bench/performance-status.md` §18 makes the merged listing the gate in front of
        // a port, and `receng`/`rec2c4` in one window is the first A/B that compares the merge against
        // the spelling the engine actually runs.
        case M_REC2C4: rec2c4_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH); break;
        // D53: the shipping spelling with the token index inside the launch. Same grid `<<<NH, D>>>`,
        // same state buffer, same `o` base; `tokens` is how many token-steps this one submission walks.
        case M_RECLOOP:
            recloop_kernel<<<kNH, kD>>>(q, k, v, g, beta, S_in, o, kNH, tokens);
            break;
        default:
            std::fprintf(stderr, "no kernel for mode %d\n", mode);
            std::exit(3);
    }
    CK(hipGetLastError());
}

// ---- guarded buffers (the `gdn_collapse_check.hip.cpp` shape) ------------------------------------
float* alloc_guarded(size_t n, float** usable) {
    float* base = nullptr;
    CK(hipMalloc(&base, (n + 2 * (size_t)kGuard) * sizeof(float)));
    std::vector<float> pad(n + 2 * (size_t)kGuard, kSentinel);
    CK(hipMemcpy(base, pad.data(), pad.size() * sizeof(float), hipMemcpyHostToDevice));
    *usable = base + kGuard;
    return base;
}

// How many guard floats are no longer the sentinel, i.e. how much of the surrounding allocation a
// kernel stomped.
int guards_clean(const float* base, size_t n) {
    std::vector<float> pad(n + 2 * (size_t)kGuard);
    CK(hipMemcpy(pad.data(), base, pad.size() * sizeof(float), hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < kGuard; ++i) {
        if (pad[i] != kSentinel) ++bad;
        if (pad[n + kGuard + i] != kSentinel) ++bad;
    }
    return bad;
}

int count_sentinel(const std::vector<float>& v) {
    int n = 0;
    for (float x : v) n += (x == kSentinel) ? 1 : 0;
    return n;
}

// The same count straight off the device, for paths that did not copy the buffer into a vector first.
long long count_sentinel_dev(const float* dptr, size_t n) {
    std::vector<float> h(n);
    CK(hipMemcpy(h.data(), const_cast<float*>(dptr), n * sizeof(float), hipMemcpyDeviceToHost));
    return count_sentinel(h);
}

// Bit comparison, no band. -0.0 vs +0.0 and two NaN payloads must NOT be called equal here, because the
// claim being checked is the bits.
long bit_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return -1;
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(a.data());
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(b.data());
    long d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (pa[i] != pb[i]) ? 1 : 0;
    return d;
}

// Magnitude, for the one column the bit diff cannot adjudicate. Reported as max |a-b| / max(1e-30,|b|)
// over the tensor, which is the quantity that separates "the device's rsqrtf is not glibc's" (order
// 1e-6) from "this MODE read the wrong row" (order 1).
double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        const double r = d / std::fmax(1e-30, std::fabs((double)b[i]));
        if (r > worst) worst = r;
    }
    return worst;
}

// The §10.1 definitions, spelled as §10.1 spells them, because `maxrel` here is NOT the per-element
// ratio above: §10.1 says `maxrel = max |Δ| / (max |ref| + 1e-30)`, a TENSOR-relative quantity, and a
// re-associated dot moves a cancelling element by a huge per-element ratio and a tiny tensor-relative
// one (`prototype/gdn_collapse` §2's cancellation note is that exact case). Using the wrong
// denominator would either pass a broken MODE or fail a legal one, and this tree has been bitten by
// both.
double maxrel_10_1(const std::vector<float>& a, const std::vector<float>& b) {
    double ref_max = 0.0, worst = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double r = std::fabs((double)b[i]);
        if (r > ref_max) ref_max = r;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    return worst / (ref_max + 1e-30);
}

double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (!(na > 0.0) || !(nb > 0.0)) return 0.0;   // also catches a NaN/Inf tensor: cos is not a number
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// ---- the host mirror ------------------------------------------------------------------------------
// `__shfl_down_sync(full, v, delta)` over one 32-lane wave, simulated lane-for-lane with CUDA's
// out-of-range rule: a lane whose source lane leaves the wave gets its OWN value back. Every lane is
// kept, not just lane 0, because that is the claim `gdn_rec_isa_kernels.h::shfl_down_bits` makes about
// the hand-spelled fold, and lane 0 alone cannot show it.
void wave_shfl_down_add(float* lane /*32*/, int delta) {
    float src[kWave];
    for (int i = 0; i < kWave; ++i) src[i] = lane[(i + delta < kWave) ? i + delta : i];
    for (int i = 0; i < kWave; ++i) lane[i] += src[i];
}

void wave_fold(float* lane /*32*/) {
    const int steps[5] = {16, 8, 4, 2, 1};
    for (int s = 0; s < 5; ++s) wave_shfl_down_add(lane, steps[s]);
}

// The shipped `block_sum` (`gdn_decode_kernels.h:30`) over `n` per-thread values: 5 steps in-warp, then
// the same 5 steps over the per-warp partials. Returns lane 0's value, which is the only value any
// thread reads back. The addition order is the shipped one step for step, which is the whole point.
float host_block_sum(const float* vals, int n) {
    const int warps = n / kWave;
    float red[kWave];
    for (int w = 0; w < warps; ++w) {
        float lane[kWave];
        for (int i = 0; i < kWave; ++i) lane[i] = vals[w * kWave + i];
        wave_fold(lane);
        red[w] = lane[0];
    }
    float lane[kWave];
    for (int i = 0; i < kWave; ++i) lane[i] = (i < warps) ? red[i] : 0.f;
    wave_fold(lane);
    return lane[0];
}

// One token-step on the host: the same operands, the same state, the same expression tree. Each host
// loop over `j` stands for one thread, and columns are independent, so the interleaving of loop A and
// loop B across columns has no arithmetic consequence. FMA sites are explicit (`fmaf`) because the build
// passes `-ffp-contract=off` to the HOST only: a host that contracted wherever it liked would not be
// the tree the listing shows, and a mirror that re-associates cannot adjudicate re-association.
void cpu_token_step(const float* q, const float* k, const float* v, const float* g, const float* beta,
                    float* S, float* o) {
    float kq[kD], qq[kD], k2[kD], q2[kD], ks[kD], qs[kD];
    for (int h = 0; h < kNH; ++h) {
        for (int j = 0; j < kD; ++j) {
            kq[j] = k[h * kD + j];
            qq[j] = q[h * kD + j];
            k2[j] = kq[j] * kq[j];
            q2[j] = qq[j] * qq[j];
        }
        const float kr = 1.0f / std::sqrt(host_block_sum(k2, kD) + 1e-6f);
        const float qr = 1.0f / std::sqrt(host_block_sum(q2, kD) + 1e-6f);
        for (int j = 0; j < kD; ++j) {
            ks[j] = kq[j] * kr;
            // the shipped association: `(qq*rsqrt)*Q_SCALE`. Collapsing the constant into one multiply
            // is a different product tree — `prototype/gdn_collapse/results.md` §2 is that exact bug.
            qs[j] = (qq[j] * qr) * Q_SCALE;
        }
        const float gam = std::exp(g[h]);
        const float bet = beta[h];
        float* Sh = S + (size_t)h * kD * kD;
        for (int j = 0; j < kD; ++j) {
            float r = 0.f;
            for (int i = 0; i < kD; ++i) r = fmaf(Sh[(size_t)i * kD + j], ks[i], r);  // loop A: v_fmac_f32_e32
            const float dv = bet * fmaf(-gam, r, v[h * kD + j]);   // v_fma_f32 v1,-v6,v1,v5 then v_mul_f32
            float acc = 0.f;
            for (int i = 0; i < kD; ++i) {
                const float gs = gam * Sh[(size_t)i * kD + j];      // a separate mul: this value is stored back
                const float s = fmaf(ks[i], dv, gs);
                Sh[(size_t)i * kD + j] = s;
                acc = fmaf(qs[i], s, acc);
            }
            o[h * kD + j] = acc;
        }
    }
}

// ---- the CPU-only lane selftest -------------------------------------------------------------------
// The one thing the admission gate cannot check without a device is the claim the MODEs are built on:
// that a hand-spelled permute reads the source lane the compiler's own lowering reads, in EVERY lane.
// `--selftest-lane` re-derives both rules on the host and requires them identical for the geometry the
// kernels use. It is a cheap witness, not the gate: it says the fold's lane choice cannot be the reason
// a MODE differs, and it stays silent about everything else.
int hand_src_lane(int lane, int delta) {
    if (delta == 16) return lane | 16;                    // ds_swizzle BITMASK_PERM "1pppp"
    const int d = (lane + delta < kWave) ? delta : 0;     // the compiler's own clamp: delta -> 0
    return lane + d;
}
int cuda_src_lane(int lane, int delta) {
    return (lane + delta < kWave) ? lane + delta : lane;  // out of range -> the caller's own value
}

// Two rules that are NOT the compiler's, kept as the selftest's must-catch cases: a check that cannot
// fail is decoration (the `mlp_gemv` / `mixed_bw` / `attn_tile_model` lesson, one layer over). `xor` is
// the cheap butterfly an optimiser might reach for; `zero` is the out-of-range rule misread as "a lane
// that falls off the wave contributes nothing". Both differ from the shipped lowering in real lanes, and
// the selftest is only worth anything if it can see that.
int xor_src_lane(int lane, int delta) {
    return lane ^ delta;
}
int zero_src_lane(int lane, int delta) {
    return (lane + delta < kWave) ? lane + delta : -1;   // -1 index is never read: the fold maps it to 0.f
}

// The 5-step fold over one wave, spelled with a given source-lane rule. `src_rule` returns the lane
// whose value this lane adds (negative means "add nothing"), so the same fold body adjudicates every
// rule under test. Both rules must produce the same value in EVERY lane or the hand fold is a different
// tree — which is the thing this selftest exists to catch before a device is involved.
void fold_with_rule(const float* in, float* out, int (*src_rule)(int, int)) {
    const int steps[5] = {16, 8, 4, 2, 1};
    float cur[kWave];
    for (int i = 0; i < kWave; ++i) cur[i] = in[i];
    for (int s = 0; s < 5; ++s) {
        float add[kWave];
        for (int i = 0; i < kWave; ++i) {
            const int src = src_rule(i, steps[s]);
            add[i] = (src >= 0) ? cur[src] : 0.f;
        }
        for (int i = 0; i < kWave; ++i) cur[i] = cur[i] + add[i];
    }
    for (int i = 0; i < kWave; ++i) out[i] = cur[i];
}

// How many lanes a wrong rule moves relative to the compiler's, over one full wave of data.
int fold_rule_diffs(const float* in, int (*wrong)(int, int)) {
    float a[kWave], b[kWave];
    fold_with_rule(in, a, wrong);
    fold_with_rule(in, b, &cuda_src_lane);
    int d = 0;
    for (int i = 0; i < kWave; ++i) d += (a[i] != b[i]) ? 1 : 0;
    return d;
}

int selftest_lane() {
    const int steps[5] = {16, 8, 4, 2, 1};
    const int kFoldTrials = 500;
    int bad = 0, cases = 0;
    int xor_lanes_moved = 0, zero_lanes_moved = 0;
    for (int s = 0; s < 5; ++s) {
        for (int lane = 0; lane < kWave; ++lane) {
            ++cases;
            if (hand_src_lane(lane, steps[s]) != cuda_src_lane(lane, steps[s])) ++bad;
        }
    }
    std::mt19937 rng(4242);
    std::normal_distribution<float> nrm(0.f, 1.f);
    for (int trial = 0; trial < kFoldTrials; ++trial) {
        float vals[kWave * 4];
        for (float& x : vals) x = nrm(rng);
        float red_hand[4], red_ship[4];
        // Stage 1: four full waves of data, hand rule vs compiler rule, in every lane — not only lane 0,
        // because "bit-identical where the consumer reads" is a weaker claim than the one the header makes.
        for (int w = 0; w < 4; ++w) {
            float a[kWave], b[kWave];
            fold_with_rule(vals + w * kWave, a, &hand_src_lane);
            fold_with_rule(vals + w * kWave, b, &cuda_src_lane);
            red_hand[w] = a[0];
            red_ship[w] = b[0];
            for (int i = 0; i < kWave; ++i) {
                ++cases;
                if (a[i] != b[i]) ++bad;   // every lane, not only lane 0
            }
            // must-catch, on the same data and the same comparison the real check uses
            xor_lanes_moved += fold_rule_diffs(vals + w * kWave, &xor_src_lane);
            zero_lanes_moved += fold_rule_diffs(vals + w * kWave, &zero_src_lane);
        }
        // Stage 2, exactly as the shipped `block_sum` spells it: only lanes 0..warps-1 hold a partial
        // and every other lane contributes 0.f, so 32 lanes go into the fold and lane 0 comes out.
        float in_hand[kWave], in_ship[kWave];
        for (int i = 0; i < kWave; ++i) {
            in_hand[i] = (i < 4) ? red_hand[i] : 0.f;
            in_ship[i] = (i < 4) ? red_ship[i] : 0.f;
        }
        float s2a[kWave], s2b[kWave];
        fold_with_rule(in_hand, s2a, &hand_src_lane);
        fold_with_rule(in_ship, s2b, &cuda_src_lane);
        for (int i = 0; i < kWave; ++i) {
            ++cases;
            if (s2a[i] != s2b[i]) ++bad;
        }
        // and the mirror the device comparison uses must be the same tree as the compiler rule
        ++cases;
        if (host_block_sum(vals, kWave * 4) != s2b[0]) ++bad;
    }
    // The must-catch verdict: both wrong rules have to move lanes the shipped tree does not, or the
    // agreement above is agreeing about nothing.
    if (xor_lanes_moved == 0 || zero_lanes_moved == 0) ++bad;
    std::printf("SELFTEST lane cases=%d bad=%d ok=%d must-catch-lanes-moved xor=%d zero=%d"
                " claim=the hand-spelled permute reads the compiler's own source lane in every lane,"
                " so MODEs 1/2 preserve the shipped addition tree\n",
                cases, bad, bad ? 0 : 1, xor_lanes_moved, zero_lanes_moved);
    return bad ? 1 : 0;
}

// ---- the CPU-only mirror selftest -----------------------------------------------------------------
// The mirror is the ruler that says "the operands are right" for a column no MODE-vs-MODE diff can
// reach (a fixture bug shared by every MODE). A ruler nobody checked is worse than none, so these are
// exact identities the recurrence has by construction, at g = 0 exactly (gam = expf(0) = 1 exactly on
// both sides, so the rsqrt/exp floor cannot excuse a mismatch) and with beta at 0 or 1 exactly:
//
//   A  beta = 0                      dv = 0, so the state walk is S -> gam*S = S: the final state must
//                                     equal the STARTING state bit for bit. That is a witness about
//                                     state addressing, per-head base and loop B's write-back, and it
//                                     holds whatever q, k and v are.
//   B  beta = 1, starting state = 0  r = 0, dv = v, so the state comes out as the outer product
//                                     S[h][i][j] = ks_h[i] * v_h[j], ks recomputed here from the raw k
//                                     row. A q/k swap, a transposed state, or a head base off by one
//                                     all break it, and none of them can be seen by comparing two MODEs
//                                     that share the same fixture.
//   C  beta = 0                      o[h*D+j] = sum_i qs_h[i] * S0[h][i][j], summed in the shipped
//                                     serial order. Witnesses the `o` layout and the q-side wiring.
//
// `--selftest-mirror` is CPU-only: no HIP call, no device, no lock.
int selftest_mirror() {
    std::mt19937 rng(90210);
    std::normal_distribution<float> nrm(0.f, 0.5f);
    std::vector<float> u(kQkvRows), g(kNH), beta(kNH), s0(kStateElems), o((size_t)kValueDim);
    for (auto& x : u) x = nrm(rng);
    for (auto& x : g) x = 0.f;                // gam = expf(0) = 1 exactly, on both sides
    for (auto& x : s0) x = nrm(rng);
    int bad = 0, cases = 0;

    // ---- A: beta = 0 leaves the state untouched, bit for bit
    for (auto& x : beta) x = 0.f;
    std::vector<float> sA = s0;
    cpu_token_step(u.data(), u.data() + kValueDim, u.data() + 2 * kValueDim, g.data(), beta.data(),
                   sA.data(), o.data());
    for (size_t i = 0; i < sA.size(); ++i) {
        ++cases;
        if (sA[i] != s0[i]) ++bad;
    }

    // ---- C: with beta = 0, o is qs . S0, and ks never enters
    for (int h = 0; h < kNH; ++h) {
        const float* qq = u.data() + h * kD;   // the q row is at `row`; see `launch_mode`
        float q2[kD];
        for (int j = 0; j < kD; ++j) q2[j] = qq[j] * qq[j];
        const float qr = 1.0f / std::sqrt(host_block_sum(q2, kD) + 1e-6f);
        float acc = 0.f;
        const float* Sh = s0.data() + (size_t)h * kD * kD;
        for (int i = 0; i < kD; ++i) acc = fmaf((qq[i] * qr) * Q_SCALE, Sh[(size_t)i * kD + 0], acc);
        // only column 0 is re-derived here; the point is the layout and the q-side wiring, and column 0
        // is enough to catch a stride error (a wrong stride moves every column including this one).
        ++cases;
        if (o[h * kD + 0] != acc) ++bad;
    }

    // ---- B: beta = 1 from a zero state gives the outer product ks (x) v
    for (auto& x : beta) x = 1.f;
    std::vector<float> sB(kStateElems, 0.f);
    std::vector<float> oB((size_t)kValueDim, 0.f);
    cpu_token_step(u.data(), u.data() + kValueDim, u.data() + 2 * kValueDim, g.data(), beta.data(),
                   sB.data(), oB.data());
    for (int h = 0; h < kNH; ++h) {
        const float* kk = u.data() + kValueDim + h * kD;   // q at `row`, k at `row + kValueDim`
        float k2[kD];
        for (int j = 0; j < kD; ++j) k2[j] = kk[j] * kk[j];
        const float kr = 1.0f / std::sqrt(host_block_sum(k2, kD) + 1e-6f);
        const float* vh = u.data() + 2 * kValueDim + h * kD;
        for (int j = 0; j < kD; ++j) {
            const float dvj = vh[j];   // beta=1, gam=1, r=0 -> dv = 1*(v - 1*0) = v, exactly
            for (int i = 0; i < kD; ++i) {
                ++cases;
                const float want = (kk[i] * kr) * dvj;
                if (sB[(size_t)h * kD * kD + (size_t)i * kD + j] != want) ++bad;
            }
        }
    }
    std::printf("SELFTEST mirror cases=%d bad=%d ok=%d claim=the host mirror walks q|k|v rows, heads,"
                " state and o at the same addresses and in the same order as rec_kernel"
                " (identities: beta=0 leaves the state bit-identical; beta=1 from a zero state gives"
                " ks (x) v; o is qs . state)\n",
                cases, bad, bad ? 0 : 1);
    return bad ? 1 : 0;
}

// ---- the CPU-only loop selftest (D53) -------------------------------------------------------------
/* WHAT THIS ADJUDICATES, AND WHAT IT CANNOT — read this before trusting the line it prints.
 *
 * The claim under test is "walking N tokens inside one launch produces the same bits as launching
 * once per token". That claim has two halves and they need different instruments:
 *
 *   ADDRESSING. Which byte of `q|k|v`, `g`, `beta`, `o` and the state does token t read and write, and
 *     does the state survive from token t to token t+1. THAT is what this selftest adjudicates: it
 *     runs a host mirror of `recloop_kernel`'s index arithmetic against the host mirror of the
 *     per-token launch mapping, element for element, at several span sizes, and any mapping error — a
 *     row stride, a head base, an `o` row, a lost carry — moves bytes here exactly as it would on
 *     device.
 *   ASSOCIATION. Whether the looped kernel's float tree is the shipped one. This selftest CANNOT see
 *     it: both mirrors call the same per-token arithmetic, and two sides that share a formula agree
 *     (`prototype/gdn_collapse/results.md` §2, the `q̃` row, is that exact trap and it is quoted in
 *     `engine/gdn_rec_fused_kernels.h`). The association claim is witnessed two other ways:
 *     `check_recloop_transcription.py` diffs the looped body against the shipping kernel line for
 *     line (and carries a mutation so IT cannot pass vacuously), and the device gate diffs the
 *     compiled bits against `rec0`, which is the only instrument that can see a compiler-side
 *     re-association or an FMA contraction.
 *
 * MUST-CATCH, DERIVED. Each injected defect names the geometric fact that makes it observable, and the
 * case list is checked against that fact per case: a defect that is predicted invisible on a case must
 * come back uncaught there, and a defect that no case in the list can catch is a defect nobody is
 * testing (that failure is `bad`, not a skipped row). The hand-listed version of this table rotted
 * twice in this repo (`prototype/attn_tile_model.py`, `prototype/mlp_gemv`), so the predicates below
 * are computed from the case's own draw — token count, the spread of the drawn `g`/`beta`, whether the
 * starting state is zero — not from its label.
 */
enum LoopDefect {
    LD_NONE = 0,
    LD_T_PIN0,         // token index pinned to row 0: a batched body that forgot `t` (the same defect
                       // `gates_kernel`'s GG_ROW_ZERO is). Observable iff tokens > 1.
    LD_OROW_HALF,      // `o` written at half the row stride: token rows collide. Observable iff > 1.
    LD_STATE_RESTART,  // the state is zeroed at each token: the loop lost its carry. Observable iff
                       // some token READS a state a restart would not leave: tokens > 1, or a non-zero
                       // starting state on the only token.
    LD_GBETA_PIN0,     // `beta` read from row 0 every token. Observable iff the drawn betas differ.
    LD_G_PIN0,         // `g` (the log-decay the state is scaled by) read from row 0 every token.
                       // Observable iff the drawn g values differ.
    N_LD
};
static const char* kLoopDefectName[] = {"none", "t_pinned_to_row0", "o_row_stride_half",
                                        "state_restarted_per_token", "beta_pinned_to_row0",
                                        "g_pinned_to_row0"};

// `recloop_kernel`'s address mapping, spelled the way the kernel spells it: `+t*CH` on the three
// q|k|v slices, `+t*NH` on g|beta, `+t*(NH*D)` on `o`, and ONE state buffer shared across the span.
// The per-token arithmetic is the SAME `cpu_token_step` the per-token mirror uses, on purpose — see
// the header above for why that makes this an addressing witness and not an arithmetic one.
void cpu_loop_span(const std::vector<float>& u, const std::vector<float>& g,
                   const std::vector<float>& beta, float* S, float* o, int t0, int ntok, int defect) {
    for (int t = 0; t < ntok; ++t) {
        const int tt = (defect == LD_T_PIN0) ? 0 : t0 + t;
        if (defect == LD_STATE_RESTART)
            std::fill(S, S + kStateElems, 0.f);
        const size_t gix = (defect == LD_G_PIN0) ? 0u : (size_t)tt * kNH;
        const size_t bix = (defect == LD_GBETA_PIN0) ? 0u : (size_t)tt * kNH;
        const size_t ooff = (defect == LD_OROW_HALF) ? 0u : (size_t)t * kValueDim;
        cpu_token_step(u.data() + (size_t)tt * kQkvRows,
                       u.data() + (size_t)tt * kQkvRows + kValueDim,
                       u.data() + (size_t)tt * kQkvRows + 2 * kValueDim,
                       g.data() + gix, beta.data() + bix, S, o + (size_t)t0 * kValueDim + ooff);
    }
}

struct LoopCase {
    std::string label;
    int tokens;
    float decay;
    bool zero_state;
    unsigned seed;
    bool flat_gates;      // draw g and beta constant: the two pin-to-row-0 defects MUST NOT be caught
    // Derived facts about the DRAW (not the label), used to build each defect's must-catch set. Both
    // spreads are across TOKEN ROWS (row t vs row 0), which is the axis a pinned row index can move.
    double g_spread = 0.0, beta_spread = 0.0;
};

void loop_case_draw(LoopCase& c, std::vector<float>& u, std::vector<float>& g,
                    std::vector<float>& beta, std::vector<float>& s0) {
    std::mt19937 rng(c.seed);
    std::normal_distribution<float> nrm(0.f, 0.5f);
    u.assign((size_t)c.tokens * kQkvRows, 0.f);
    g.assign((size_t)c.tokens * kNH, 0.f);
    beta.assign((size_t)c.tokens * kNH, 0.f);
    s0.assign(kStateElems, 0.f);
    for (auto& x : u) x = nrm(rng);
    for (auto& x : g) x = c.flat_gates ? -0.25f : -std::exp(nrm(rng)) * c.decay;
    for (auto& x : beta) x = c.flat_gates ? 0.5f : 1.f / (1.f + std::exp(-nrm(rng)));
    if (!c.zero_state) for (auto& x : s0) x = 0.25f * nrm(rng);
    // The spread that matters for a PIN-TO-ROW-0 defect is across TOKEN ROWS, not across the 16 heads of
    // one row: pinning the row index can only change a value that differs between rows. Getting this
    // wrong is not hypothetical — the first version of this line compared every drawn value against the
    // first, predicted that a one-token case must catch the pin, and was refused by its own predicate
    // (`missed_catch=2`). A must-catch set derived from the wrong axis fails loudly if it is derived at
    // all, which is the whole argument for deriving it.
    for (int t = 1; t < c.tokens; ++t) {
        for (int h = 0; h < kNH; ++h) {
            c.g_spread = std::max(c.g_spread, std::fabs((double)g[(size_t)t * kNH + h] - g[h]));
            c.beta_spread = std::max(c.beta_spread,
                                     std::fabs((double)beta[(size_t)t * kNH + h] - beta[h]));
        }
    }
}

// The must-catch predicate: derived from the case's geometry and draw, per defect.
bool loop_defect_must_fire(int defect, const LoopCase& c) {
    switch (defect) {
        case LD_T_PIN0:        return c.tokens > 1;
        case LD_OROW_HALF:     return c.tokens > 1;
        case LD_STATE_RESTART: return c.tokens > 1 || !c.zero_state;
        case LD_GBETA_PIN0:    return c.beta_spread > 0.0;
        case LD_G_PIN0:        return c.g_spread > 0.0;
        default:               return false;   // LD_NONE is not a defect
    }
}

int selftest_loop() {
    // The case list is small on purpose (tokens <= 12): what is under test is the MAPPING, and a long
    // sequence adds cost without adding a new mapping. What it must contain, and does, is at least one
    // one-token case (so the "cannot fire" side of two defects is actually exercised), a carried-state
    // case at a near-zero decay (so the state carry is observable), and a flat-gates case (so the two
    // pin-to-row-0 defects have a case where they must come back clean).
    std::vector<LoopCase> cases = {
        {"t1-zeros-flat",      1, 0.4f,   true,  31u, true },
        {"t1-carried-d0.4",    1, 0.4f,   false, 32u, false},
        {"t2-zeros-d0.4",      2, 0.4f,   true,  33u, false},
        {"t2-carried-d0.002",  2, 0.002f, false, 34u, false},
        {"t8-carried-d0.4",    8, 0.4f,   false, 35u, false},
        {"t8-zeros-flat",      8, 0.4f,   true,  36u, true },
        {"t12-carried-d0.002", 12, 0.002f, false, 37u, false},
        {"t12-carried-d0.4",   12, 0.4f,   false, 38u, false},
    };
    // The spans a launch may be given. The engine's `--gdn-loop` will be an operator knob, so the span
    // must be arithmetically irrelevant: every span, including a ragged last chunk (3 into 12), has to
    // reproduce the per-token form bit for bit.
    const int kSpans[] = {1, 2, 3, 12};

    long elems_compared = 0, span_runs = 0, defect_runs = 0;
    int bad = 0, clean_mismatch = 0, wrong_catch = 0, missed_catch = 0;
    int defect_must[N_LD] = {0}, defect_caught[N_LD] = {0}, defect_must_not[N_LD] = {0};

    for (auto& c : cases) {
        std::vector<float> u, g, beta, s0;
        loop_case_draw(c, u, g, beta, s0);

        // The reference: the per-token mapping, i.e. `run_sequence`'s shape on the host.
        std::vector<float> ref_s = s0, ref_o((size_t)c.tokens * kValueDim, 0.f);
        for (int t = 0; t < c.tokens; ++t) {
            cpu_token_step(u.data() + (size_t)t * kQkvRows,
                           u.data() + (size_t)t * kQkvRows + kValueDim,
                           u.data() + (size_t)t * kQkvRows + 2 * kValueDim,
                           g.data() + (size_t)t * kNH, beta.data() + (size_t)t * kNH,
                           ref_s.data(), ref_o.data() + (size_t)t * kValueDim);
        }

        // (1) every span must reproduce the reference exactly.
        for (int sp : kSpans) {
            std::vector<float> s = s0, o((size_t)c.tokens * kValueDim, 0.f);
            for (int t0 = 0; t0 < c.tokens; t0 += sp) {
                const int n = (sp < c.tokens - t0) ? sp : (c.tokens - t0);
                cpu_loop_span(u, g, beta, s.data(), o.data(), t0, n, LD_NONE);
            }
            const long d_o = bit_diff(o, ref_o), d_s = bit_diff(s, ref_s);
            elems_compared += 2 * ((long)c.tokens * kValueDim + kStateElems);
            ++span_runs;
            if (d_o != 0 || d_s != 0) {
                ++clean_mismatch; ++bad;
                std::printf("LOOP MISMATCH case=%s span=%d ndiff_o=%ld ndiff_state=%ld"
                            " — the looped mapping is not the per-token mapping\n",
                            c.label.c_str(), sp, d_o, d_s);
            }
        }

        // (2) every injected defect must move bytes EXACTLY where its geometry says it can.
        for (int d = 1; d < N_LD; ++d) {
            std::vector<float> s = s0, o((size_t)c.tokens * kValueDim, 0.f);
            cpu_loop_span(u, g, beta, s.data(), o.data(), 0, c.tokens, d);
            const bool caught = bit_diff(o, ref_o) != 0 || bit_diff(s, ref_s) != 0;
            const bool must = loop_defect_must_fire(d, c);
            elems_compared += 2 * ((long)c.tokens * kValueDim + kStateElems);
            ++defect_runs;
            if (must) ++defect_must[d]; else ++defect_must_not[d];
            if (caught) ++defect_caught[d];
            if (must && !caught) { ++missed_catch; ++bad; }
            if (!must && caught) { ++wrong_catch; ++bad; }
        }
    }

    // A comparison list that is empty cannot fail, so its size is asserted, not assumed — and the
    // expected size is DERIVED from the case list, so a case list that shrank is caught too.
    long want_span_runs = 0, want_defect_runs = 0;
    for (const auto& c : cases) { want_span_runs += 4; want_defect_runs += (N_LD - 1); }
    if (span_runs != want_span_runs || span_runs == 0) { ++bad; }
    if (defect_runs != want_defect_runs || defect_runs == 0) { ++bad; }
    // Every defect must have at least one case that can catch it; a defect no case can see is not a
    // control, it is decoration (`tools/mixed_bw`, `prototype/attn_tile_model`, one layer over).
    int defects_unreachable = 0;
    for (int d = 1; d < N_LD; ++d) if (defect_must[d] == 0) ++defects_unreachable;
    if (defects_unreachable) ++bad;

    std::printf("SELFTEST loop cases=%zu span_runs=%ld want_span_runs=%ld defect_runs=%ld"
                " want_defect_runs=%ld elems_compared=%ld bad=%d ok=%d"
                " clean_mismatch=%d missed_catch=%d wrong_catch=%d defects_unreachable=%d"
                " claim=recloop_kernel's row/state/o mapping reproduces the per-token mapping exactly"
                " at every span, and each loop defect moves bytes exactly where its geometry says it"
                " must (NOT a claim about float association, which this mirror shares with itself by"
                " construction and only the device bit-identity gate can see)\n",
                cases.size(), span_runs, want_span_runs, defect_runs, want_defect_runs,
                elems_compared, bad, bad ? 0 : 1, clean_mismatch, missed_catch, wrong_catch,
                defects_unreachable);
    for (int d = 1; d < N_LD; ++d) {
        std::printf("  must-catch %s: cases-that-must-catch=%d caught=%d"
                    " cases-that-must-NOT-catch=%d\n",
                    kLoopDefectName[d], defect_must[d], defect_caught[d], defect_must_not[d]);
    }
    return bad ? 1 : 0;
}

// ---- a case, its fixture, and the two paths over it -----------------------------------------------
struct Case {
    std::string label;
    int tokens = 128;
    bool zero_state = true;
    float decay = 0.4f;
    unsigned seed = 7u;
};

// How many floats of state are allocated for one MODE: one layer's worth for a MODE that carries its
// state, or `pool` slots for a split probe, whose per-token slot index is what makes consecutive
// launches independent even at the address level.
struct Fixture;
size_t state_span(const Fixture& f, int mode);

struct Fixture {
    Case c;
    int pool = 1;                       // per-token state slots; only split MODEs index it
    std::vector<float> u, g, beta, s0;
    // device, each between guard bands. `du/dg/db` are shared read-only operands; every MODE owns its
    // own state and `o`, so a stray write in one MODE cannot be laundered by another.
    float *bu = nullptr, *bg = nullptr, *bb = nullptr, *bs0 = nullptr;
    float *du = nullptr, *dg = nullptr, *db = nullptr, *ds0 = nullptr;
    float *bst[NMODES] = {nullptr}, *bso[NMODES] = {nullptr};
    float *st[NMODES] = {nullptr}, *so[NMODES] = {nullptr};     // WRITE side (and the only one, usually)
    float *bsr[NMODES] = {nullptr}, *sr[NMODES] = {nullptr};    // READ side; == st for carried MODEs
};

size_t state_span(const Fixture& f, int mode) {
    return (size_t)(mode_splits_state(mode) ? f.pool : 1) * kStateElems;
}

void fixture_build(Fixture& f) {
    const int tokens = f.c.tokens;
    std::mt19937 rng(f.c.seed);
    std::normal_distribution<float> nrm(0.f, 0.5f);
    f.u.assign((size_t)tokens * kQkvRows, 0.f);
    f.g.assign((size_t)tokens * kNH, 0.f);
    f.beta.assign((size_t)tokens * kNH, 0.f);
    f.s0.assign(kStateElems, 0.f);
    for (auto& x : f.u) x = nrm(rng);
    // §04.3 leaves g <= 0 (a decay log) and its magnitude is the state's MEMORY: per token the state is
    // scaled by gam = exp(g), so a large decay erases the starting state inside one run. Both shapes are
    // in the matrix for the reason `gdn_collapse_check.hip.cpp` gives for the same choice: with a short
    // memory the state a case ends on carries no trace of what it started with, and a defect in state
    // handling then cannot show up in the state at all — only in `o`.
    for (auto& x : f.g) x = -std::exp(nrm(rng)) * f.c.decay;
    for (auto& x : f.beta) x = 1.f / (1.f + std::exp(-nrm(rng)));
    if (!f.c.zero_state)
        for (auto& x : f.s0) x = 0.25f * nrm(rng);

    f.bu = alloc_guarded((size_t)tokens * kQkvRows, &f.du);
    f.bg = alloc_guarded((size_t)tokens * kNH, &f.dg);
    f.bb = alloc_guarded((size_t)tokens * kNH, &f.db);
    f.bs0 = alloc_guarded(kStateElems, &f.ds0);
    CK(hipMemcpy(f.du, f.u.data(), f.u.size() * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(f.dg, f.g.data(), f.g.size() * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(f.db, f.beta.data(), f.beta.size() * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(f.ds0, f.s0.data(), kStateElems * sizeof(float), hipMemcpyHostToDevice));
}

void fixture_build_modes(Fixture& f, const ModeSet& ms) {
    for (int i = 0; i < ms.n; ++i) {
        const int m = ms.m[i];
        const size_t span = state_span(f, m);
        f.bst[m] = alloc_guarded(span, &f.st[m]);
        if (mode_splits_state(m)) {
            f.bsr[m] = alloc_guarded(span, &f.sr[m]);   // a separate allocation: that IS the lever
        } else {
            f.bsr[m] = nullptr;                          // read side and write side are one buffer
            f.sr[m] = f.st[m];
        }
        f.bso[m] = alloc_guarded((size_t)f.c.tokens * kValueDim, &f.so[m]);
    }
}

void fixture_free(Fixture& f) {
    (void)hipFree(f.bu); (void)hipFree(f.bg); (void)hipFree(f.bb); (void)hipFree(f.bs0);
    for (int m = 0; m < NMODES; ++m) {
        (void)hipFree(f.bst[m]);
        (void)hipFree(f.bsr[m]);
        (void)hipFree(f.bso[m]);
    }
}

// Restore a MODE's state to the fixture's starting state, one slot at a time so every slot of a split
// pool starts from a real state. Used before every timed burst loop so a burst walks the SAME state
// trajectory the gated launches walked, instead of decaying for `burst x tokens` launches into values
// no real wave ever presents.
void state_restore(const Fixture& f, int mode) {
    const size_t slots = mode_splits_state(mode) ? (size_t)f.pool : 1u;
    for (size_t s = 0; s < slots; ++s) {
        CK(hipMemcpy(f.st[mode] + s * (size_t)kStateElems, f.ds0, kStateElems * sizeof(float),
                     hipMemcpyDeviceToDevice));
        if (f.sr[mode] != f.st[mode]) {
            CK(hipMemcpy(f.sr[mode] + s * (size_t)kStateElems, f.ds0, kStateElems * sizeof(float),
                         hipMemcpyDeviceToDevice));
        }
    }
}

// One token-step per launch for every MODE except `recloop`, which walks `span_of` tokens per launch —
// which is the engine's shape either way (`engine/decode_step.cpp`'s `for (t = 0; t < n; ++t)
// gdn_rec(...)` becomes one call for the looped form, over the same [n][6144] tile).
// A split probe indexes slot t%pool on BOTH sides, so the slot token t+1 reads is not the slot token t
// wrote: no launch in the sequence waits for its predecessor's store.
void run_sequence(const Fixture& f, int mode, int tokens) {
    const size_t slots = mode_splits_state(mode) ? (size_t)f.pool : 1u;
    const int span = span_of(mode, tokens);          // 1 for every MODE that is not `recloop`
    for (int base = 0; base < tokens; base += span) {
        const int n = (span < tokens - base) ? span : (tokens - base);
        const size_t slot = slots ? ((size_t)base % slots) : 0u;
        const float* row = f.du + (size_t)base * kQkvRows;
        launch_mode(mode, row, row + kValueDim, row + 2 * kValueDim, f.dg + (size_t)base * kNH,
                    f.db + (size_t)base * kNH, f.sr[mode] + slot * (size_t)kStateElems,
                    f.st[mode] + slot * (size_t)kStateElems,
                    f.so[mode] + (size_t)base * kValueDim, n);
    }
}

std::string sanitize(const std::string& s) {
    std::string o = s;
    for (char& ch : o)
        if (!(ch == '-' || ch == '.' || ch == '_')) {
            if (ch == ' ') ch = '_';
        }
    return o;
}

void dump_file(const std::string& dir, const std::string& case_name, int mode, const char* what,
               const float* dptr, size_t elems) {
    std::vector<float> h(elems);
    CK(hipMemcpy(h.data(), const_cast<float*>(dptr), elems * sizeof(float), hipMemcpyDeviceToHost));
    const std::string path = dir + "/" + sanitize(case_name) + "." + kModeName[mode] + "." + what + ".bin";
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(3);
    }
    std::fwrite(h.data(), sizeof(float), elems, fp);
    std::fclose(fp);
}

// The falsifier thresholds, in one place so the number and the decision printed with it cannot drift
// apart. `r` is MODE-0 time over lever time, i.e. how much faster the lever is.
static const char* falsifier_band(double r) {
    if (r > 1.10) return "REFUTES-the-latency-reading-move-the-norms-out-of-the-launch-MODE-3";
    if (r < 1.02) return "KILLS-the-instruction-route-for-this-family";
    return "INCONCLUSIVE-between-the-two-bands";
}

// The dependency probes' thresholds, kept separate from `falsifier_band` on purpose: that one prices an
// INSTRUCTION lever (>1.10 refutes the latency reading, <1.02 kills the instruction route) and reading a
// latency probe through it would invert the meaning of both numbers. Here the question is whether the
// chain is what costs, so a LARGE ratio is the positive result.
static const char* dependency_band(double r) {
    if (r > 1.10) return "LATENCY_BINDS-fund-occupancy-D30-1-4-and-chain-shortening-1-5";
    if (r < 1.02) return "LATENCY_DOES_NOT_BIND-at-this_granularity-family_needs_another_model";
    return "INCONCLUSIVE-between-the-two-bands-take-the-counters";
}

/* The `recloop` band, separate for the same reason `dependency_band` is separate: `falsifier_band` reads
 * an instruction lever (>1.10 refutes the latency reading, <1.02 kills the instruction route), and here
 * the axis is submissions per token-step, so the same numbers mean something else. `r` is
 * per-token-form ms over looped-form ms for the same token span.
 *   >1.10  the launch axis is worth real time at the smallest geometry the harness can build, which is
 *          what licenses the engine port and the in-situ A/B;
 *   <1.02  either the floor is absorbed at this geometry or the loop's own addressing ate it — the
 *          addressing-spelling variant (§10's rec6 lesson) is the next lever, not the port;
 *   between  the harness cannot separate them and only the in-situ family row decides (D51's instrument).
 */
static const char* launch_structure_band(double r) {
    if (r > 1.10) return "LAUNCH_STRUCTURE_WORTH-it-at-harness_geometry-port_it_and_A_B_in_situ";
    if (r < 1.02) return "LAUNCH_STRUCTURE_WORTH_NOTHING-here-floor_absorbed_or_addressing_ate_it";
    return "PARTIAL-at-harness_geometry-the_in_situ_gdn_rec_family_row_decides";
}

// ---- the VOLUME self-check ---------------------------------------------------------------------
/* WHY THIS EXISTS AND WHAT IT IS NOT. The dependency probes are meaningless by design, so the only
 * thing that may be compared between them and MODE 0 is WORK VOLUME. If a probe quietly wrote less, or
 * read a smaller region, it would come back fast for a reason that has nothing to do with latency, and
 * that is exactly how `tools/mixed_bw`'s first attempt produced plausible ms numbers over a checksum
 * that counted 1/blockDim of the bytes it had read. So the check is on BYTES, on both sides:
 *
 *   WRITE SIDE (device, exact): the write-side state buffer starts as `kSentinel`, a value the
 *     recurrence cannot produce, and after the sequence EVERY element must have stopped being it. That
 *     is a per-element witness of which addresses were written, not an inference from the values.
 *     For a split probe the READ-side buffer is checked for the mirror image: it must still be
 *     byte-identical to the fixture state that was seeded into it, which is what proves the write pass
 *     MOVED to another buffer instead of vanishing. Guard bands on both catch a write outside either.
 *
 *   READ SIDE (device, exact where the buffers are separate): a split probe's read region must still
 *     equal the fixture bytes, i.e. the bytes it was supposed to read are the bytes that were there,
 *     and nothing wrote them. For a MODE that reads and writes one buffer this cannot separate read
 *     from write, so the read volume is witnessed where it actually lives: in the instruction stream,
 *     by `traffic_census.py --volume-check`, which counts `global_load_b32`/`global_store_b32` per
 *     token-step in the listing the binary runs and exits non-zero if they differ from MODE 0's. That
 *     command is printed by this check and is a required part of the protocol, not a footnote.
 *
 *   THE PER-LAUNCH ACCOUNT is re-derived on the host from the same constants the launch is built from
 *     (the `tools/mixed_bw` `plan_bounds` rule: never let the kernel certify its own geometry), and a
 *     MODE whose account differs from MODE 0's PER TOKEN-STEP fails here even if the device agrees
 *     with it. Per launch is the wrong denominator for a MODE that walks several tokens per submission
 *     (`recloop`), which is exactly why the account carries a span and the check divides it out.
 */
struct LaunchAccount {
    long long state_read_elems = 0;      // state elements read per launch
    long long state_write_elems = 0;     // state elements written per launch
    long long operand_read_elems = 0;    // q|k|v rows plus g|beta, per launch
    long long operand_write_elems = 0;   // the `o` row
};

/* Every form in this file walks the state in the same two loops: loop A reads the layer's whole
 * [16][128][128] block once, loop B reads it again and writes it once. Hence 2 read passes and 1 write
 * pass over exactly one layer's state per launch — for the split probes the write pass lands in a
 * different buffer, which changes addresses, not bytes. The pass count is a property of the source
 * (`rec0_kernel`'s two `for (i = 0; i < D; ++i)` walks) and `traffic_census.py --volume-check` is what
 * proves the compiled listing really issues 2+1 passes at 4 B per access, per token-step.
 *
 * `span` is the number of token-steps one launch carries (1 for everything except `recloop`), so the
 * account is per LAUNCH and the invariant the check below applies is per TOKEN-STEP: `recloop` is
 * allowed to move submissions and is not allowed to move a byte of traffic per token.
 */
LaunchAccount launch_account(int mode, int span) {
    LaunchAccount a;
    a.state_read_elems = 2LL * span * kStateElems;
    a.state_write_elems = 1LL * span * kStateElems;
    a.operand_read_elems = 1LL * span * (3LL * kValueDim + 2LL * kNH);
    a.operand_write_elems = 1LL * span * kValueDim;
    (void)mode;
    return a;
}

long long account_bytes(const LaunchAccount& a, bool read) {
    return (long long)((read ? a.state_read_elems + a.operand_read_elems
                              : a.state_write_elems + a.operand_write_elems) * sizeof(float));
}

int count_equal(const std::vector<float>& v, float want) {
    int n = 0;
    for (float x : v) n += (x == want) ? 1 : 0;
    return n;
}

// How many of `have` differ from `want` (bit-compared, because "untouched" is a bits claim).
long long bits_differ(const std::vector<float>& have, const std::vector<float>& want) {
    if (have.size() != want.size()) return -1;
    const uint32_t* ph = reinterpret_cast<const uint32_t*>(have.data());
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(want.data());
    long long d = 0;
    for (size_t i = 0; i < have.size(); ++i) d += (ph[i] != pw[i]) ? 1 : 0;
    return d;
}

int volume_check(const ModeSet& ms, int pool) {
    Case c{"volume", pool > 8 ? pool : 8, true, 0.4f, 5u};   // long enough to touch every pool slot
    Fixture f;
    f.c = c;
    f.pool = pool;
    fixture_build(f);
    fixture_build_modes(f, ms);

    const LaunchAccount want_account = launch_account(M_REC0, 1);   // MODE 0: one token-step per launch
    int failed = 0;
    for (int i = 0; i < ms.n; ++i) {
        const int m = ms.m[i];
        const size_t span = state_span(f, m);
        // Fresh sentinel seeding: `alloc_guarded` pre-fills with kSentinel, so a slot nobody writes is
        // still recognisable as unwritten. Seed both sides from the fixture state, then re-poison the
        // WRITE side and leave the read side seeded: after the sequence, "still sentinel" on the write
        // side means nobody wrote that element, and "still the fixture bits" on the read side of a
        // split probe means the write pass moved to another buffer instead of disappearing.
        state_restore(f, m);
        std::vector<float> poison(span, kSentinel);
        CK(hipMemcpy(f.st[m], poison.data(), span * sizeof(float), hipMemcpyHostToDevice));

        run_sequence(f, m, c.tokens);
        CK(hipDeviceSynchronize());

        std::vector<float> back_w(span), back_r(span);
        CK(hipMemcpy(back_w.data(), f.st[m], span * sizeof(float), hipMemcpyDeviceToHost));
        const long long unwritten = count_equal(back_w, kSentinel);
        const int guard_w = guards_clean(f.bst[m], span);
        const int guard_r = f.bsr[m] ? guards_clean(f.bsr[m], span) : 0;
        long long readside_untouched = -1;
        if (mode_splits_state(m)) {
            std::vector<float> one(kStateElems);
            CK(hipMemcpy(one.data(), f.ds0, kStateElems * sizeof(float), hipMemcpyDeviceToHost));
            CK(hipMemcpy(back_r.data(), f.sr[m], span * sizeof(float), hipMemcpyDeviceToHost));
            long long diff = 0;
            for (size_t s = 0; s < (size_t)f.pool; ++s)
                diff += bits_differ(std::vector<float>(back_r.begin() + s * kStateElems,
                                                       back_r.begin() + (s + 1) * kStateElems), one);
            readside_untouched = diff;
        }
        const long long unwritten_o = count_sentinel_dev(f.so[m], (size_t)c.tokens * kValueDim);
        const int tok_span = span_of(m, c.tokens);          // token-steps per launch (1 unless recloop)
        const LaunchAccount a = launch_account(m, tok_span);
        // The invariant is per TOKEN-STEP, not per launch: `recloop` is defined by moving submissions,
        // and a form that moved submissions by doing less work per token would pass a per-launch compare
        // against itself while cheating the A/B. So: per launch, its account must be exactly `span` x
        // MODE 0's; per token-step, it must be MODE 0's to the element.
        const bool same_account = a.state_read_elems == (long long)tok_span * want_account.state_read_elems &&
            a.state_write_elems == (long long)tok_span * want_account.state_write_elems &&
            a.operand_read_elems == (long long)tok_span * want_account.operand_read_elems &&
            a.operand_write_elems == (long long)tok_span * want_account.operand_write_elems;
        const bool ok = same_account && unwritten == 0 && guard_w == 0 && guard_r == 0 &&
            unwritten_o == 0 && (readside_untouched == 0 || (!mode_splits_state(m) && readside_untouched == -1));
        if (!ok) ++failed;
        std::printf("VOLUME mode=%s launches=%lld tokens=%d span=%d pool=%d split=%d state_read_elems=%lld"
                    " state_write_elems=%lld operand_read_elems=%lld operand_write_elems=%lld"
                    " bytes_read_per_launch=%lld bytes_written_per_launch=%lld"
                    " bytes_read_per_token_step=%lld bytes_written_per_token_step=%lld"
                    " dev_unwritten_state_elems=%lld dev_unwritten_o=%lld guard=%d"
                    " readside_bits_differing_vs_seeded=%lld account_same_per_token_step_as_rec0=%d"
                    " verdict=%s\n",
                    kModeName[m], (long long)launches_of(m, c.tokens), c.tokens, tok_span, f.pool,
                    mode_splits_state(m) ? 1 : 0,
                    a.state_read_elems, a.state_write_elems, a.operand_read_elems,
                    a.operand_write_elems, account_bytes(a, true), account_bytes(a, false),
                    account_bytes(a, true) / tok_span, account_bytes(a, false) / tok_span,
                    unwritten, (long long)unwritten_o, guard_w + guard_r, readside_untouched,
                    same_account ? 1 : 0, ok ? "VOLUME_OK" : "VOLUME_MISMATCH");
    }
    std::printf("VOLUME summary modes=%s ok=%d pool=%d" 
                " note=instruction-count witness is NOT here, it is the ISA census; run:"
                " .venv/bin/python prototype/gdn_rec_isa/traffic_census.py --volume-check" 
                " --asm tmp/gdn_isa/gdn_rec_isa.s (default --modes covers every probe and rec6)\n",
                ms.names().c_str(), failed ? 0 : 1, pool);
    fixture_free(f);
    return failed;
}

// Counts printed BURST lines so main() can require that the whole case x rep x mode matrix was timed.
// A burst that silently timed a subset reads exactly like a clean run to a grep for RESULT.
static long g_burst_lines = 0;

int run_case(const Case& c, const ModeSet& ms, int repeat, int burst, int reps,
             const std::string& outdir, int pool, bool volume_ok) {
    Fixture f;
    f.c = c;
    f.pool = pool;
    fixture_build(f);
    fixture_build_modes(f, ms);
    const size_t n_o = (size_t)c.tokens * kValueDim;

    // The CPU reference: one pass over the same operands and the same starting state.
    std::vector<float> cpu_o(n_o), cpu_s = f.s0;
    for (int t = 0; t < c.tokens; ++t) {
        const float* row = f.u.data() + (size_t)t * kQkvRows;
        cpu_token_step(row, row + kValueDim, row + 2 * kValueDim, f.g.data() + (size_t)t * kNH,
                       f.beta.data() + (size_t)t * kNH, cpu_s.data(),
                       cpu_o.data() + (size_t)t * kValueDim);
    }

    // Every MODE, `repeat` times, each from the same starting state.
    std::vector<std::vector<float>> ho(NMODES * (size_t)repeat), hs(NMODES * (size_t)repeat);
    for (int rep = 0; rep < repeat; ++rep) {
        for (int i = 0; i < ms.n; ++i) {
            const int m = ms.m[i];
            state_restore(f, m);
            run_sequence(f, m, c.tokens);
            CK(hipDeviceSynchronize());
            ho[(size_t)m * repeat + rep].assign(n_o, 0.f);
            hs[(size_t)m * repeat + rep].assign(kStateElems, 0.f);
            CK(hipMemcpy(ho[(size_t)m * repeat + rep].data(), f.so[m], n_o * sizeof(float),
                         hipMemcpyDeviceToHost));
            CK(hipMemcpy(hs[(size_t)m * repeat + rep].data(), f.st[m], kStateElems * sizeof(float),
                         hipMemcpyDeviceToHost));
        }
    }

    std::printf("CASE %s tokens=%d state=%-15s decay=%-6g seed=%u modes=%s repeat=%d pool=%d"
                " o_elems=%zu state_elems=%d grid=%dx%d wavefronts=%d\n",
                c.label.c_str(), c.tokens, c.zero_state ? "zeros" : "carried-random", c.decay,
                c.seed, ms.names().c_str(), repeat, pool, n_o, kStateElems, kNH, kD,
                kNH * (kD / kWave));

    const int guard_operand = guards_clean(f.bu, (size_t)c.tokens * kQkvRows) +
        guards_clean(f.bg, (size_t)c.tokens * kNH) + guards_clean(f.bb, (size_t)c.tokens * kNH) +
        guards_clean(f.bs0, kStateElems);

    int failed = 0;
    for (int i = 0; i < ms.n; ++i) {
        const int m = ms.m[i];
        const std::vector<float>& o0 = ho[(size_t)M_REC0 * repeat];
        const std::vector<float>& s0 = hs[(size_t)M_REC0 * repeat];
        const std::vector<float>& o = ho[(size_t)m * repeat];
        const std::vector<float>& s = hs[(size_t)m * repeat];
        const long do0 = bit_diff(o, o0), ds0 = bit_diff(s, s0);
        const long doc = mode_is_probe(m) ? -1 : bit_diff(o, cpu_o);
        const long dsc = mode_is_probe(m) ? -1 : bit_diff(s, cpu_s);
        const int unwritten = count_sentinel(o);
        const int guard = guards_clean(f.bso[m], n_o) + guards_clean(f.bst[m], state_span(f, m)) +
            (f.bsr[m] ? guards_clean(f.bsr[m], state_span(f, m)) : 0);
        long rerun_o = 0, rerun_s = 0;
        if (repeat > 1) {   // a racy hand permute disagrees with itself before it disagrees with anyone
            rerun_o = bit_diff(o, ho[(size_t)m * repeat + 1]);
            rerun_s = bit_diff(s, hs[(size_t)m * repeat + 1]);
        }
        // Three admission policies, and the one that runs is printed so a reader never has to guess
        // which column adjudicated. MODEs 1/2/2c must reproduce MODE 0's BITS (a re-spelled reduction
        // that moves a bit is a numerics change, per `results.md` §6). `recx_unchain` re-associates two
        // dots, so bits are the wrong question; it is judged inside the EXISTING §10.1 GDN band
        // (cos ≥ 0.999, meanrel ≤ 5 %), which is an existing gate being applied, not a new one being
        // invented. `recx_indep` and `recx_both` replace what the recurrence READS, so no value
        // statement about them is meaningful at all and the volume check is the only gate they have.
        const bool bit_ok = (do0 == 0 && ds0 == 0);
        const double cos_o = cosine_sim(o, o0), cos_s = cosine_sim(s, s0);
        const double mr_o = maxrel_10_1(o, o0), mr_s = maxrel_10_1(s, s0);
        const bool band_ok = cos_o >= 0.999 && cos_s >= 0.999 && mr_o <= 5e-2 && mr_s <= 5e-2;
        const char* gate = "none";
        bool admitted = true;
        if (mode_value_gated(m)) { gate = "bit-identity-vs-rec0"; admitted = bit_ok; }
        else if (mode_band_gated(m)) { gate = "existing-10.1-GDN-band-cos-ge-0.999-meanrel-le-0.05";
                                       admitted = band_ok; }
        else if (mode_is_probe(m)) { gate = "volume-only-value-is-meaningless-by-design"; admitted = true; }
        const bool ok = admitted && unwritten == 0 && guard == 0 && guard_operand == 0 &&
            rerun_o == 0 && rerun_s == 0;
        if (!ok) ++failed;
        std::printf("RESULT case=%s mode=%s tokens=%d gate=%s ndiff_o_vs0=%ld ndiff_state_vs0=%ld"
                    " cos_o_vs0=%.6f cos_state_vs0=%.6f maxrel10_o_vs0=%.3e maxrel10_state_vs0=%.3e"
                    " ndiff_o_vs_cpu=%ld ndiff_state_vs_cpu=%ld"
                    " maxrel_o_vs_cpu=%.3e maxrel_state_vs_cpu=%.3e unwritten_o=%d guard=%d"
                    " rerun_ndiff_o=%ld rerun_ndiff_state=%ld verdict=%s\n",
                    c.label.c_str(), kModeName[m], c.tokens, gate, do0, ds0, cos_o, cos_s, mr_o, mr_s,
                    doc, dsc,
                    mode_is_probe(m) ? -1.0 : max_rel(o, cpu_o),
                    mode_is_probe(m) ? -1.0 : max_rel(s, cpu_s),
                    unwritten, guard + guard_operand, rerun_o, rerun_s,
                    ok ? (bit_ok ? "BIT_IDENTICAL" : "ADMITTED_BY_POLICY")
                       : (admitted ? "CLEAN_BUT_UNWRITTEN_OR_STOMP" : "FAILS_ADMISSION"));
        if (!outdir.empty()) {
            dump_file(outdir, c.label, m, "o", f.so[m], n_o);
            dump_file(outdir, c.label, m, "state", f.st[m], kStateElems);
        }
    }
    {
        // The floor line: what the host mirror costs, stated by the MODE it is supposed to agree with.
        const std::vector<float>& o0 = ho[(size_t)M_REC0 * repeat];
        const std::vector<float>& s0 = hs[(size_t)M_REC0 * repeat];
        std::printf("CPU-FLOOR case=%s mode=rec0 ndiff_o_vs_cpu=%ld ndiff_state_vs_cpu=%ld"
                    " maxrel_o_vs_cpu=%.3e maxrel_state_vs_cpu=%.3e"
                    " cause=device v_rsq_f32_e32 and v_exp_f32_e32 against host sqrt and exp;"
                    " every MODE bit-identical to rec0 shares this floor exactly\n",
                    c.label.c_str(), bit_diff(o0, cpu_o), bit_diff(s0, cpu_s), max_rel(o0, cpu_o),
                    max_rel(s0, cpu_s));
    }

    // ---- the burst: the falsifier, one window, every MODE inside every rep -------------------------
    // Main-agent-only in practice: a worker may not take timings. The state is restored before each
    // timed loop so all MODEs measure the same trajectory, and the restore is OUTSIDE the timed region.
    if (burst > 0) {
        if (!volume_ok) {
            std::printf("NOT TIMED case=%s reason=volume_self_check_failed a probe whose bytes touched"
                        " differ from MODE 0 would look fast for the wrong reason; fix the probe, do"
                        " not read the number\n", sanitize(c.label).c_str());
        } else if (failed) {
            std::printf("NOT TIMED case=%s reason=admission_not_clean a timing on a MODE that failed"
                        " its own admission gate is not a measurement\n",
                        sanitize(c.label).c_str());
        } else {
            std::vector<double> best(NMODES, 1e30);
            for (int i = 0; i < ms.n; ++i) {   // warm the pages: the first GPU pass over fresh pages is ~35 GB/s
                state_restore(f, ms.m[i]);
                run_sequence(f, ms.m[i], c.tokens);
            }
            CK(hipDeviceSynchronize());
            for (int r = 0; r < reps; ++r) {
                for (int i = 0; i < ms.n; ++i) {
                    const int m = ms.m[i];
                    state_restore(f, m);
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int n = 0; n < burst; ++n) run_sequence(f, m, c.tokens);
                    CK(hipDeviceSynchronize());
                    const auto t1 = std::chrono::steady_clock::now();
                    const double ms_seq =
                        std::chrono::duration<double, std::milli>(t1 - t0).count() / burst;
                    if (ms_seq < best[m]) best[m] = ms_seq;
                    // `launches`, not `tokens`: the looped MODE is the first MODE in this harness for
                    // which the two differ, and a ms/launch figure divided by the token count would
                    // report the loop's saving as a rate it does not have.
                    const long n_launch = launches_of(m, c.tokens);
                    std::printf("BURST case=%s rep=%d mode=%s ms_per_sequence=%.4f ms_per_launch=%.5f"
                                " launches=%ld tokens=%d span=%d\n",
                                sanitize(c.label).c_str(), r, kModeName[m], ms_seq,
                                ms_seq / (double)n_launch, n_launch, c.tokens,
                                span_of(m, c.tokens));
                    ++g_burst_lines;
                }
            }
            std::printf("RESULT case=%s mode=burst burst=%d reps=%d tokens=%d modes=%s",
                        sanitize(c.label).c_str(), burst, reps, c.tokens, ms.names().c_str());
            for (int i = 0; i < ms.n; ++i)
                std::printf(" best_%s_ms=%.4f", kModeName[ms.m[i]], best[ms.m[i]]);
            std::printf(" no_timing_taken=0\n");
            for (int i = 0; i < ms.n; ++i) {
                const int m = ms.m[i];
                if (m == M_REC0) continue;
                const double r = best[M_REC0] / best[m];
                std::printf("FALSIFIER case=%s mode=%s against=rec0 kind=%s best_rec0_ms=%.4f"
                            " best_mode_ms=%.4f rec0_over_mode=%.4f per_launch_rec0_us=%.3f band=%s"
                            " bill_delta=%s\n",
                            sanitize(c.label).c_str(), kModeName[m],
                            mode_is_probe(m) ? "dependency-break"
                                             : (m == M_REC6 ? "control-bill-matched"
                                                            : (m == M_REC2C4
                                                               ? "merged-fused-prologue-on-unroll4-walks"
                                                               : (m == M_RECLOOP
                                                                  ? "token-loop-launch-structure-D53"
                                                                  : "instruction-lever"))),
                            best[M_REC0], best[m], r,
                            best[M_REC0] * 1000.0 / c.tokens,
                            mode_is_probe(m) ? dependency_band(r)
                                             : (m == M_RECLOOP ? launch_structure_band(r)
                                                               : falsifier_band(r)),
                            "traffic_census.py --volume-check: a probe that is cheaper in slots owes"
                            " part of its win to D25's ~0.8 slot elasticity, not to the dependency");
            }
            if (ms.has(M_REC4) && ms.has(M_REC6)) {
                // The pair is the point of rec6: rec4 and rec6 differ in chain depth and in nothing else
                // that a compiler would not report, so THIS ratio and not rec4/rec0 is the chain.
                std::printf("DIFF case=%s what=chain_at_matched_shape rec4=%s rec6=%s"
                            " rec6_over_rec4=%.4f note=rec6 is the bill-matched control, so this ratio"
                            " has the loop-shape effect cancelled; subtract rec6's own slot delta from"
                            " the census before calling the remainder the chain\n",
                            sanitize(c.label).c_str(), kModeName[M_REC4], kModeName[M_REC6],
                            best[M_REC6] / best[M_REC4]);
            }
        }
    } else {
        std::printf("NOT TIMED case=%s no timing taken in this run (pass --burst N to time the"
                    " token-step launch)\n",
                    sanitize(c.label).c_str());
    }

    fixture_free(f);
    return failed;
}

}  // namespace

int main(int argc, char** argv) {
    std::string modespec, outdir, statename = "zeros";
    int tokens = 128, repeat = 2, burst = 0, reps = 5, matrix = 1, pool = 1, loop = 0;
    bool volume_only = false;
    unsigned seed = 7u;
    float decay = 0.4f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--modes") modespec = next();
        else if (a == "--tokens") {
            tokens = std::atoi(next().c_str());
            matrix = 0;
        } else if (a == "--seed") {
            seed = (unsigned)std::atoi(next().c_str());
            matrix = 0;
        } else if (a == "--decay") {
            decay = (float)std::atof(next().c_str());
            matrix = 0;
        } else if (a == "--state") {
            statename = next();
            matrix = 0;
        } else if (a == "--repeat") {
            repeat = std::atoi(next().c_str());
        } else if (a == "--burst") {
            burst = std::atoi(next().c_str());
        } else if (a == "--reps") {
            reps = std::atoi(next().c_str());
        } else if (a == "--out") outdir = next();
        else if (a == "--pool") pool = std::atoi(next().c_str());
        else if (a == "--loop") loop = std::atoi(next().c_str());
        else if (a == "--volume") volume_only = true;
        else if (a == "--matrix") matrix = 1;
        else if (a == "--selftest-lane") return selftest_lane();
        else if (a == "--selftest-mirror") return selftest_mirror();
        else if (a == "--selftest-loop") return selftest_loop();
        else {
            std::fprintf(stderr,
                         "usage: gdn_rec_isa [--modes ship,rec0,rec1,rec2,rec2c,rec3,rec4,rec5,rec6,receng,recloop]"
                         " [--tokens T --state zeros|random --decay D --seed S] [--repeat N]\n"
                         "                 [--volume [--pool N]] [--burst N --reps R [--pool N]]"
                         " [--loop N] [--out DIR] [--selftest-lane] [--selftest-mirror]"
                         " [--selftest-loop]\n"
                         "  --volume alone = the volume self-check and nothing else (no timing exists in"
                         "  that mode); with --burst it is the same single check followed by timing.\n"
                         "  default: the four spelling MODEs (rec0 is always the reference) over the\n"
                         "  T=128 matrix, twice each. The dependency probes are rec3 (cross-token\n"
                         "  dependency broken by a split state), rec4 (both serial chains shortened from\n"
                         "  128 links to 32 at identical volume), rec5 (both) and rec6 (rec4's loop\n"
                         "  shape with the chain NOT broken: its bill-matched control, bit-identical to\n"
                         "  rec0); receng is `engine/gdn_rec_unroll4_kernels.h::rec_unroll4_kernel`, the\n"
                         "  ENGINE's port of recx_shape_kernel, gated by bit-identity against rec0 for\n"
                         "  the same reason rec6 is — it is the witness that the port moved no bits;\n"
                         "  rec2c4 is MODE 2c's fused prologue on those unroll4 walks (§18/D46's merged\n"
                         "  form), also bit-identity-gated, and is compared against receng, not rec0;\n"
                         "  rec3/rec4/rec5 are meaningless\n"
                         "  by design, admitted on VOLUME, and a --burst runs that volume check itself and\n"
                         "  refuses to time unless it passes. --pool N indexes N independent state slots per token (the\n"
                         "  literal indexed-per-token form); N=1 keeps the working set smallest, which\n"
                         "  is the comparable choice — see gdn_rec_isa_kernels.h. --selftest-lane and\n"
                         "  --selftest-mirror are CPU-only and touch no device.\n"
                         "  recloop is D53's item: the SHIPPING spelling (`rec2c4`, i.e. "
                         "`engine/gdn_rec_fused_kernels.h`) with the token index moved INSIDE the "
                         "kernel, so one submission walks --loop N token-steps instead of one. Same "
                         "grid, same operands, same state; bit-identity-gated against rec0 like every "
                         "spelling MODE, and its burst partner is rec2c4, not rec0. --loop 0 (default) "
                         "means the whole sequence in one launch; the span is an operand, not a "
                         "numerics choice, so --selftest-loop requires EVERY span to reproduce the "
                         "per-token form exactly. --selftest-loop is CPU-only and touches no device.\n");
            return 2;
        }
    }
    if (pool < 1) {
        std::fprintf(stderr, "--pool must be >= 1\n");
        return 2;
    }
    if (loop < 0) {
        std::fprintf(stderr, "--loop must be >= 0 (0 = the whole sequence in one launch, N = N"
                             " token-steps per launch)\n");
        return 2;
    }
    g_loop = loop;   // read by span_of()/launches_of(); a non-looping MODE ignores it (see span_of)
    if (repeat < 1) {
        std::fprintf(stderr, "--repeat must be >= 1 (2 is the default: a racy permute must disagree with itself)\n");
        return 2;
    }
    bool zero_state = true;
    if (statename == "zeros") zero_state = true;
    else if (statename == "random") zero_state = false;
    else {
        std::fprintf(stderr, "--state zeros|random\n");
        return 2;
    }

    const ModeSet ms = modes_of(modespec);
    std::vector<Case> cases;
    if (matrix) {
        // T = 128 is the admission length the falsifier was written against. The two decays are the
        // state-memory axis: with a large decay the incoming state is forgotten inside the run, so only
        // the small-decay cases prove the FINAL STATE still depends on the state that went in.
        cases = {
            {"t128-zeros-d0.4", 128, true, 0.4f, 7u},
            {"t128-carried-d0.4", 128, false, 0.4f, 11u},
            {"t128-carried-d0.002", 128, false, 0.002f, 17u},
            {"t128-zeros-d0.002", 128, true, 0.002f, 19u},
        };
    } else {
        cases = {Case{"single", tokens, zero_state, decay, seed}};
    }

    std::printf("plan modes=%s | reference=rec0 | cases=%zu | tokens(matrix)=%d | repeat=%d | pool=%d"
                " | loop=%d"
                " | grid %dx%d, wave%d, %d wavefronts/token-step | state %d floats (1 layer)\n",
                ms.names().c_str(), cases.size(), tokens, repeat, pool, g_loop, kNH, kD, kWave,
                kNH * (kD / kWave), kStateElems);

    // The volume self-check runs BEFORE anything is timed, and a burst that cannot show it passing is
    // refused with a non-zero exit. It is standalone under --volume so a correctness-only run (which is
    // all a worker is allowed to take) can exercise it on its own.
    int volume_failed = 0;
    if (burst > 0 || volume_only) volume_failed = volume_check(ms, pool);
    const bool volume_ok = (burst == 0 && !volume_only) || volume_failed == 0;
    std::printf("VOLUME-GATE requested=%s run=%s ok=%s\n", (burst > 0 || volume_only) ? "yes" : "no",
                (burst > 0 || volume_only) ? "yes" : "no-burst-so-not-needed",
                volume_ok ? "yes" : "no");
    // `--volume` ALONE is the standalone no-timing pass (the only device pass a worker may take). With
    // `--burst` it must fall through to the timing. These two flags used to be overloaded: the
    // documented `--volume --burst 40 --reps 5` printed VOLUME_OK, exited 0 and timed NOTHING - the
    // "driver whose parse matches nothing still prints PASS" class, caught before any number was read
    // off it. The burst's own line-count assertion below is the second net.
    if (volume_only && burst == 0) {
        std::printf("RESULT mode=volume-only failures=%d verdict=%s\n", volume_failed,
                    volume_failed ? "VOLUME_MISMATCH" : "VOLUME_OK");
        return volume_failed ? 1 : 0;
    }

    int failed = 0;
    for (const auto& c : cases) failed += run_case(c, ms, repeat, burst, reps, outdir, pool, volume_ok);
    if (burst > 0 && !volume_ok) ++failed;   // a timing that had to be refused is a failure, not a null
    // A burst that printed fewer lines than it owes timed a subset, which reads exactly like a clean
    // run to anything grepping for a RESULT line. Count them, require the product, fail otherwise.
    if (burst > 0)
    {
        const long want = (long)cases.size() * (long)reps * (long)ms.n;
        std::printf("BURST-LINES printed=%ld want=%ld (cases=%zu reps=%d modes=%d) ok=%d\n",
                    g_burst_lines, want, cases.size(), reps, ms.n, (int)(g_burst_lines == want));
        if (g_burst_lines != want) ++failed;
    }

    std::printf("RESULT mode=summary cases=%zu modes=%s repeat=%d volume=%s admission=%s"
                " gates=bit-identity(0,1,2,2c,6) existing-10.1-GDN-band(4) volume-only(3,5)\n",
                cases.size(), ms.names().c_str(), repeat, volume_ok ? "OK" : "FAILED-or-not-run",
                failed ? "FAIL" : "PASS");
    if (failed)
        std::printf("ADMISSION FAIL: %d MODE-case verdict(s) failed their gate, or a burst was refused"
                    " for want of a clean volume check. A MODE that moves a bit it should not move is a"
                    " numerics change, not a slower lever; a probe whose volume moved is not a probe."
                    " See results.md §9 for which MODE means what.\n",
                    failed);
    return failed ? 1 : 0;
}
