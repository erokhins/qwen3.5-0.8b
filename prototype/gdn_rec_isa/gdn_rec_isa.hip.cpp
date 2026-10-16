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
 *
 * WHY IT EXISTS. The census's verdict is a funded NO — the family issues 1,523 slots per wavefront
 * against a measured 6.8-6.9 us per launch, so it runs at ~10 % of its own issue roof and instruction
 * surgery is capped near 1-13 % of a 509 ms family. That verdict came with a falsifier nobody could
 * run: MODE 0 vs MODE 2 in one burst, > 1.10x refutes the latency reading, < 1.02x kills the
 * instruction route for this family. This file is the missing instrument. It does not change the
 * verdict; it makes the verdict checkable.
 *
 * THE ADMISSION GATE IS BIT-IDENTITY, AND IT IS NOT SOFTER THAN THE CENSUS ASSUMED. Every MODE claims
 * to preserve each value's addition tree exactly (same 5+5 shuffle steps in the same order per value),
 * so the only acceptable difference against MODE 0 is ZERO, on `o` AND on the final state, over T
 * tokens. A band here would be a way of not checking the claim: MODE 1 re-spells a fold and MODE 2
 * fuses two of them, and a hand-written fold that changes association is not a lever, it is a numerics
 * change. `results.md` §6 says what each failure would mean.
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
 * and not four.
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

namespace {

constexpr int kD = D;                       // 128: d_k = d_v, one thread per d_v column
constexpr int kNH = NH;                     // 16 value heads, one block per head
constexpr int kValueDim = kNH * kD;         // 2048: one token's row of `o`
constexpr int kQkvRows = 3 * kValueDim;     // 6144: one token's q|k|v row, the engine's layout
constexpr int kStateElems = kNH * kD * kD;  // 262144: one layer's state
constexpr int kWave = 32;
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
enum Mode { M_SHIP = 0, M_REC0, M_REC1, M_REC2, M_REC2C, NMODES };
static const char* kModeName[] = {"ship", "rec0", "rec1", "rec2", "rec2c"};

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
            std::fprintf(stderr, "--modes: no mode \"%s\"; known: ship,rec0,rec1,rec2,rec2c\n",
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
void launch_mode(int mode, const float* q, const float* k, const float* v, const float* g,
                 const float* beta, float* S, float* o) {
    switch (mode) {
        case M_SHIP: rec_kernel<<<kNH, kD>>>(q, k, v, g, beta, S, o, kNH); break;
        case M_REC0: rec0_kernel<<<kNH, kD>>>(q, k, v, g, beta, S, o, kNH); break;
        case M_REC1: rec1_kernel<<<kNH, kD>>>(q, k, v, g, beta, S, o, kNH); break;
        case M_REC2: rec2_kernel<<<kNH, kD>>>(q, k, v, g, beta, S, o, kNH); break;
        case M_REC2C: rec2c_kernel<<<kNH, kD>>>(q, k, v, g, beta, S, o, kNH); break;
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

// ---- a case, its fixture, and the two paths over it -----------------------------------------------
struct Case {
    std::string label;
    int tokens = 128;
    bool zero_state = true;
    float decay = 0.4f;
    unsigned seed = 7u;
};

struct Fixture {
    Case c;
    std::vector<float> u, g, beta, s0;
    // device, each between guard bands. `du/dg/db` are shared read-only operands; every MODE owns its
    // own state and `o`, so a stray write in one MODE cannot be laundered by another.
    float *bu = nullptr, *bg = nullptr, *bb = nullptr, *bs0 = nullptr;
    float *du = nullptr, *dg = nullptr, *db = nullptr, *ds0 = nullptr;
    float *bst[NMODES] = {nullptr}, *bso[NMODES] = {nullptr};
    float *st[NMODES] = {nullptr}, *so[NMODES] = {nullptr};
};

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
        f.bst[m] = alloc_guarded(kStateElems, &f.st[m]);
        f.bso[m] = alloc_guarded((size_t)f.c.tokens * kValueDim, &f.so[m]);
    }
}

void fixture_free(Fixture& f) {
    (void)hipFree(f.bu); (void)hipFree(f.bg); (void)hipFree(f.bb); (void)hipFree(f.bs0);
    for (int m = 0; m < NMODES; ++m) {
        (void)hipFree(f.bst[m]);
        (void)hipFree(f.bso[m]);
    }
}

// Restore a MODE's state to the fixture's starting state. Used before every timed burst loop so a burst
// walks the SAME state trajectory the gated launches walked, instead of decaying for `burst x tokens`
// launches into values no real wave ever presents.
void state_restore(const Fixture& f, int mode) {
    CK(hipMemcpy(f.st[mode], f.ds0, kStateElems * sizeof(float), hipMemcpyDeviceToDevice));
}

// One token-step per token, which is the engine's shape: `rec_kernel` is launched once per
// (token, GDN layer), so a sequence of T tokens is T launches and the unit the census bills.
void run_sequence(const Fixture& f, int mode, int tokens) {
    for (int t = 0; t < tokens; ++t) {
        const float* row = f.du + (size_t)t * kQkvRows;
        launch_mode(mode, row, row + kValueDim, row + 2 * kValueDim, f.dg + (size_t)t * kNH,
                    f.db + (size_t)t * kNH, f.st[mode], f.so[mode] + (size_t)t * kValueDim);
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

int run_case(const Case& c, const ModeSet& ms, int repeat, int burst, int reps,
             const std::string& outdir) {
    Fixture f;
    f.c = c;
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

    std::printf("CASE %s tokens=%d state=%-15s decay=%-6g seed=%u modes=%s repeat=%d"
                " o_elems=%zu state_elems=%d grid=%dx%d wavefronts=%d\n",
                c.label.c_str(), c.tokens, c.zero_state ? "zeros" : "carried-random", c.decay,
                c.seed, ms.names().c_str(), repeat, n_o, kStateElems, kNH, kD,
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
        const long doc = bit_diff(o, cpu_o), dsc = bit_diff(s, cpu_s);
        const int unwritten = count_sentinel(o);
        const int guard = guards_clean(f.bso[m], n_o) + guards_clean(f.bst[m], kStateElems);
        long rerun_o = 0, rerun_s = 0;
        if (repeat > 1) {   // a racy hand permute disagrees with itself before it disagrees with anyone
            rerun_o = bit_diff(o, ho[(size_t)m * repeat + 1]);
            rerun_s = bit_diff(s, hs[(size_t)m * repeat + 1]);
        }
        const bool bit_ok = (do0 == 0 && ds0 == 0);
        const bool ok =
            bit_ok && unwritten == 0 && guard == 0 && guard_operand == 0 && rerun_o == 0 && rerun_s == 0;
        if (!ok) ++failed;
        std::printf("RESULT case=%s mode=%s tokens=%d ndiff_o_vs0=%ld ndiff_state_vs0=%ld"
                    " ndiff_o_vs_cpu=%ld ndiff_state_vs_cpu=%ld maxrel_o_vs_cpu=%.3e"
                    " maxrel_state_vs_cpu=%.3e unwritten_o=%d guard=%d rerun_ndiff_o=%ld"
                    " rerun_ndiff_state=%ld verdict=%s\n",
                    c.label.c_str(), kModeName[m], c.tokens, do0, ds0, doc, dsc, max_rel(o, cpu_o),
                    max_rel(s, cpu_s), unwritten, guard + guard_operand, rerun_o, rerun_s,
                    ok ? "BIT_IDENTICAL" : (bit_ok ? "CLEAN_BUT_UNWRITTEN_OR_STOMP" : "DIFFERS"));
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
        if (failed) {
            std::printf("NOT TIMED case=%s reason=admission_not_clean a timing on a MODE that is not"
                        " bit-identical to MODE 0 is not a lever measurement\n",
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
                    std::printf("BURST case=%s rep=%d mode=%s ms_per_sequence=%.4f ms_per_launch=%.5f\n",
                                sanitize(c.label).c_str(), r, kModeName[m], ms_seq,
                                ms_seq / (double)c.tokens);
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
                std::printf("FALSIFIER case=%s mode=%s against=rec0 best_rec0_ms=%.4f"
                            " best_mode_ms=%.4f rec0_over_mode=%.4f per_launch_rec0_us=%.3f band=%s\n",
                            sanitize(c.label).c_str(), kModeName[m], best[M_REC0], best[m], r,
                            best[M_REC0] * 1000.0 / c.tokens, falsifier_band(r));
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
    int tokens = 128, repeat = 2, burst = 0, reps = 5, matrix = 1;
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
        else if (a == "--matrix") matrix = 1;
        else if (a == "--selftest-lane") return selftest_lane();
        else if (a == "--selftest-mirror") return selftest_mirror();
        else {
            std::fprintf(stderr,
                         "usage: gdn_rec_isa [--modes ship,rec0,rec1,rec2,rec2c] [--tokens T --state"
                         " zeros|random --decay D --seed S] [--repeat N] [--burst N --reps R]"
                         " [--out DIR] [--selftest-lane] [--selftest-mirror]\n"
                         "  default: the four MODEs (rec0 is always the reference) over the T=128 case\n"
                         "  matrix, twice each. --selftest-lane and --selftest-mirror are CPU-only and\n"
                         "  touch no device.\n");
            return 2;
        }
    }
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

    std::printf("plan modes=%s | reference=rec0 | cases=%zu | tokens(matrix)=%d | repeat=%d"
                " | grid %dx%d, wave%d, %d wavefronts/token-step | state %d floats (1 layer)\n",
                ms.names().c_str(), cases.size(), tokens, repeat, kNH, kD, kWave,
                kNH * (kD / kWave), kStateElems);

    int failed = 0;
    for (const auto& c : cases) failed += run_case(c, ms, repeat, burst, reps, outdir);

    std::printf("RESULT mode=summary cases=%zu modes=%s repeat=%d admission=%s"
                " gate=bit-identity-on-o-and-final-state-against-rec0\n",
                cases.size(), ms.names().c_str(), repeat, failed ? "FAIL" : "PASS");
    if (failed)
        std::printf("ADMISSION FAIL: %d MODE-case verdict(s) off bit-identity, unwritten, stomped or"
                    " run-to-run unstable. A MODE that moves a bit is a numerics change, not a lever;"
                    " see results.md §6 for which MODE means what.\n",
                    failed);
    return failed ? 1 : 0;
}
