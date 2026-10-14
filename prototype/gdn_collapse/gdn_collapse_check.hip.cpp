/*
 * prototype/gdn_collapse/gdn_collapse_check.hip.cpp — device gate for route A (§04.4's recurrence
 * with the state held resident in LDS: ONE launch per GDN layer per wave body).
 *
 * WHAT IT JUDGES. `engine/gdn_collapse_kernels.h`'s claim is BIT identity with the shipped
 * sequential form, not closeness — the recurrence is the same arithmetic and only the residency and
 * the launch shape moved (`progress/2026-10-14/p9-route-a-2026-10-14.md` D-12). So this file runs BOTH forms over
 * the SAME operands and the SAME starting state and compares `o` and the resulting state bit for
 * bit: no band, no cos, no max-ULP. A band here would be a way of not checking the claim.
 *
 *   reference side : `rec_kernel` from prototype/gdn_decode/gdn_decode_kernels.h, launched once per
 *                    token per layer — verbatim what `engine/decode_step.cpp`'s `seq` branch does.
 *   route A side   : `gdn_rc::rec_collapse_kernel`, launched once per BODY, with the state carried
 *                    from one body to the next exactly as the engine carries it between bodies.
 *
 * WHY IT EXISTS AS A SEPARATE BINARY. The end-to-end gate for this route is `engine/gate.py
 * --gdn-scan collapse` against `--gdn-scan seq`, and that is the right instrument for the shipped
 * path — but a device fault on the collapse path costs a 5-minute HSA hang-analysis round trip
 * through the engine (the queue dump has to time out), and the first version of this kernel faulted.
 * A gate that takes five minutes to say FAIL gets run once; this one takes seconds and touches one
 * layer, which is also what makes it usable as a red/green pair while the kernel is being repaired.
 * `engine/gate.py` stays the authority on the shipped route; this is the fast loop.
 *
 * BYTES TOUCHED, NOT JUST VALUES (the `tools/mixed_bw` lesson: verify what a kernel touched, not only
 * what it computed). Three checks the value comparison alone cannot make:
 *   - every element of `o` is written. Both `o` buffers are pre-filled with a sentinel neither form
 *     can produce, so a form that silently skips a strip shows up as an unwritten element instead of
 *     as a lucky match against an equally-unwritten reference element.
 *   - nothing outside `o` and the state is written: every device buffer is surrounded by guard bands
 *     pre-filled with the same sentinel. The bug this harness was written for — using `tid` (0..127)
 *     as a column index into a 32-column strip — read and wrote up to 96 floats past every head's
 *     state, i.e. past the END of the tensor for the last head. That is what a guard band is for.
 *   - the two forms run on SEPARATE state and output buffers, so a stray write in one cannot be
 *     laundered by the other, and the guards are checked on both.
 *
 * NEGATIVES are compile-time `-D` builds (never a runtime branch), taken from
 * `engine/gdn_collapse_kernels.h`'s header: NEG_RC_STATE_BF16, NEG_RC_STRIP_SHIFT, NEG_RC_NORM_HALF,
 * NEG_RC_I_DESC. `run.py` builds each into its own binary with `-DNEG_RC_ANY_EXPECT_BREAK`, which
 * flips the verdict so the ONLY acceptable outcome is a bit difference — and the byte-touch checks
 * must still be clean, because a negative that "fails" by writing outside its buffer proves nothing
 * about the defect it names. Note the trap already recorded for the CPU mirror: NEG_RC_STATE_BF16 is
 * invisible from a ZERO starting state (rounding a staged zero is still zero), which is why every
 * case here is run from a carried non-zero state as well as from zeros, and why a two-body case is in
 * the matrix at all: only a body that stages a previous body's state can see state staging.
 *
 * NO TIMING. This answers "same bits?" and nothing else (AGENTS.md). The milliseconds for this route
 * are taken from `--time-waves` in the engine, where the win line is written against the `gdn_rec`
 * row; `progress/2026-10-14/p9-route-a-2026-10-14.md` §5 carries those commands.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "engine/gdn_collapse_kernels.h"   // pulls prototype/gdn_decode/gdn_decode_kernels.h: rec_kernel, block_sum, D, NH

namespace {

constexpr int kQkvRows = gdn_rc::kQkvRows;    // 6144: one token's q|k|v row of `u`
constexpr int kValueDim = gdn_rc::kValueDim;  // 2048: one token's row of `o`
constexpr int kNH = gdn_rc::kNH;              // 16 value heads
constexpr int kD = gdn_rc::kD;                // 128
constexpr int kStateElems = kNH * kD * kD;    // one layer's state
// A value neither form can produce from these operands. If it survives anywhere in `o` nothing wrote
// there; if it survives in a guard band nothing outside the buffer wrote there.
constexpr float kSentinel = -1.2345e37f;
constexpr int kGuard = 512;                   // floats of guard band on each side

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        hipError_t e_ = (call);                                                       \
        if (e_ != hipSuccess) {                                                       \
            std::fprintf(stderr, "hip: %s (%s:%d)\n", hipGetErrorString(e_), __FILE__, \
                         __LINE__);                                                   \
            std::exit(3);                                                             \
        }                                                                             \
    } while (0)

// A buffer with guard bands: `n` usable floats between two regions pre-filled with the sentinel.
// Returns the BASE (what gets freed) and hands back the usable pointer.
float* alloc_guarded(size_t n, float** usable) {
    float* base = nullptr;
    CUDA_CHECK(hipMalloc(&base, (n + 2 * kGuard) * sizeof(float)));
    std::vector<float> pad(n + 2 * kGuard, kSentinel);
    CUDA_CHECK(hipMemcpy(base, pad.data(), pad.size() * sizeof(float), hipMemcpyHostToDevice));
    *usable = base + kGuard;
    return base;
}

// Guard-band integrity after a run: how many guard floats are no longer the sentinel, i.e. how much
// of the surrounding allocation the kernel stomped.
int guards_clean(const float* base, size_t n) {
    std::vector<float> pad(n + 2 * kGuard);
    CUDA_CHECK(hipMemcpy(pad.data(), base, pad.size() * sizeof(float), hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < kGuard; ++i) {
        if (pad[i] != kSentinel) ++bad;                 // front guard
        if (pad[n + kGuard + i] != kSentinel) ++bad;    // back guard, AFTER the usable region
    }
    return bad;
}

int count_sentinel(const std::vector<float>& v) {
    int n = 0;
    for (float x : v) n += (x == kSentinel) ? 1 : 0;
    return n;
}

// Bit comparison, no band: -0.0 vs +0.0 and two NaN payloads must NOT be called equal here, because
// the claim being checked is the bits.
long bit_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return -1;
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(a.data());
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(b.data());
    long d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (pa[i] != pb[i]) ? 1 : 0;
    return d;
}

struct Case {
    std::string label;
    int tokens;
    int body;
    bool zero_state;
    unsigned seed;
    float decay = 0.4f;   // magnitude of the decay log; see --decay
};

int run_case(const Case& c) {
    const int tokens = c.tokens, body = c.body;
    if (body < 1 || body > tokens || tokens % body != 0) {
        std::fprintf(stderr, "--body must divide --tokens (both are the engine's body shape)\n");
        return 2;
    }

    // ---- host operands: one GDN layer's worth of post-conv q|k|v, decay log and beta.
    std::mt19937 rng(c.seed);
    std::normal_distribution<float> nrm(0.f, 0.5f);
    std::vector<float> h_u((size_t)tokens * kQkvRows), h_g((size_t)tokens * kNH),
        h_beta((size_t)tokens * kNH), h_s0(kStateElems);
    for (auto& x : h_u) x = nrm(rng);
    // §04.3 leaves g <= 0 (a decay log), and its magnitude is the state's MEMORY: per token the state is
    // scaled by gam = exp(g), so a body of T tokens forgets its incoming state by roughly exp(-T*decay).
    // The default 0.4 with T=128 erases the incoming state inside one body, which is a fixture property
    // and not a kernel property — and it is what made the first run of the NEG_RC_STATE_BF16 build report
    // `state bit-diff 0` while `o` moved 222k elements: the rounding was there, the memory of it was not.
    // `--decay 0.002` keeps the state long-lived (which is what a real GDN head does) and is the shape
    // the state-staging negative has to be run on.
    for (auto& x : h_g) x = -std::exp(nrm(rng)) * c.decay;
    for (auto& x : h_beta) x = 1.f / (1.f + std::exp(-nrm(rng)));       // sigmoid(beta_raw)
    if (!c.zero_state) for (auto& x : h_s0) x = 0.25f * nrm(rng);

    // ---- device: separate guarded buffers per form, so a stray write in one cannot hide in the other.
    float *bu = nullptr, *bg = nullptr, *bb = nullptr, *bo_ref = nullptr, *bo_new = nullptr;
    float *bs_ref = nullptr, *bs_new = nullptr;
    float *u, *g, *beta, *o_ref, *o_new, *s_ref, *s_new;
    bu = alloc_guarded((size_t)tokens * kQkvRows, &u);
    bg = alloc_guarded((size_t)tokens * kNH, &g);
    bb = alloc_guarded((size_t)tokens * kNH, &beta);
    bo_ref = alloc_guarded((size_t)tokens * kValueDim, &o_ref);
    bo_new = alloc_guarded((size_t)tokens * kValueDim, &o_new);
    bs_ref = alloc_guarded(kStateElems, &s_ref);
    bs_new = alloc_guarded(kStateElems, &s_new);
    CUDA_CHECK(hipMemcpy(u, h_u.data(), h_u.size() * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(g, h_g.data(), h_g.size() * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(beta, h_beta.data(), h_beta.size() * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(s_ref, h_s0.data(), kStateElems * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(s_new, h_s0.data(), kStateElems * sizeof(float), hipMemcpyHostToDevice));

    // ---- reference side: one launch per token per layer, `decode_step.cpp`'s `seq` branch verbatim.
    for (int t = 0; t < tokens; ++t) {
        const float* row = u + (size_t)t * kQkvRows;
        rec_kernel<<<kNH, kD>>>(row, row + kValueDim, row + 2 * kValueDim, g + (size_t)t * kNH,
                                beta + (size_t)t * kNH, s_ref, o_ref + (size_t)t * kValueDim, kNH);
    }
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());

    // ---- route A: one launch per BODY, state carried between bodies as the engine carries it.
    for (int off = 0; off < tokens; off += body) {
        gdn_rc::rec_collapse_kernel<<<dim3(gdn_rc::kStrips, gdn_rc::kNH), gdn_rc::kBlockDim>>>(
            u + (size_t)off * kQkvRows, g + (size_t)off * kNH, beta + (size_t)off * kNH, s_new,
            o_new + (size_t)off * kValueDim, body);
        CUDA_CHECK(hipGetLastError());
    }
    CUDA_CHECK(hipDeviceSynchronize());

    std::vector<float> ho_ref((size_t)tokens * kValueDim), ho_new((size_t)tokens * kValueDim);
    std::vector<float> hs_ref(kStateElems), hs_new(kStateElems);
    CUDA_CHECK(hipMemcpy(ho_ref.data(), o_ref, ho_ref.size() * sizeof(float), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(ho_new.data(), o_new, ho_new.size() * sizeof(float), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(hs_ref.data(), s_ref, kStateElems * sizeof(float), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(hs_new.data(), s_new, kStateElems * sizeof(float), hipMemcpyDeviceToHost));

    const long do_bit = bit_diff(ho_ref, ho_new);
    const long ds_bit = bit_diff(hs_ref, hs_new);
    const int unwritten_ref = count_sentinel(ho_ref);
    const int unwritten_new = count_sentinel(ho_new);
    const int guard_o = guards_clean(bo_ref, (size_t)tokens * kValueDim) +
        guards_clean(bo_new, (size_t)tokens * kValueDim);
    const int guard_s = guards_clean(bs_ref, kStateElems) + guards_clean(bs_new, kStateElems);
    const int guard_operand = guards_clean(bu, (size_t)tokens * kQkvRows) +
        guards_clean(bg, (size_t)tokens * kNH) + guards_clean(bb, (size_t)tokens * kNH);

    bool pass = do_bit == 0 && ds_bit == 0 && unwritten_ref == 0 && unwritten_new == 0 &&
        guard_o == 0 && guard_s == 0 && guard_operand == 0;
#ifdef NEG_RC_ANY_EXPECT_BREAK
    // Built as a negative: the ONLY acceptable outcome is a defect that SHOWS, which is bits moved OR
    // columns left unwritten (that last one is how NEG_RC_STRIP_SHIFT fails — four strips of a head own
    // no columns, so nobody writes those `o` elements at all), while the byte-touch checks must STILL be
    // clean: a negative that "fails" by writing outside its buffers proves nothing about the defect it
    // names. `run.py` holds the per-defect signature and the control that must NOT move.
    pass = (do_bit != 0 || ds_bit != 0 || unwritten_new != 0) && unwritten_ref == 0 && guard_o == 0 &&
        guard_s == 0 && guard_operand == 0;
#endif

    std::printf("CASE %-40s tokens=%4d body=%3d state=%-13s decay=%-6g seed=%u\n"
                "     o bit-diff %8ld/%zu   state bit-diff %8ld/%d\n"
                "     o unwritten seq=%d collapse=%d   guard violations o=%d state=%d operands=%d\n"
                "     %s\n",
                c.label.c_str(), tokens, body, c.zero_state ? "zeros" : "carried-random", c.decay,
                c.seed, do_bit, ho_ref.size(), ds_bit, kStateElems, unwritten_ref, unwritten_new,
                guard_o, guard_s, guard_operand, pass ? "PASS" : "FAIL");

    // (void) because hipFree is [[nodiscard]] and a free of a buffer whose kernel already reported its
    // own error is not a new error; the verdict above already has every signal this case produces.
    (void)hipFree(bu); (void)hipFree(bg); (void)hipFree(bb);
    (void)hipFree(bo_ref); (void)hipFree(bo_new); (void)hipFree(bs_ref); (void)hipFree(bs_new);
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<Case> cases;
    Case one{"single", 64, 64, true, 7, 0.4f};
    bool matrix = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_int = [&](const char* name) -> int {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); std::exit(2); }
            return std::atoi(argv[++i]);
        };
        if (a == "--tokens") { one.tokens = next_int("--tokens"); matrix = false; }
        else if (a == "--body") { one.body = next_int("--body"); matrix = false; }
        else if (a == "--seed") { one.seed = (unsigned)next_int("--seed"); matrix = false; }
        else if (a == "--decay") {
            if (i + 1 >= argc) { std::fprintf(stderr, "--decay needs a value\n"); return 2; }
            one.decay = (float)std::atof(argv[++i]);
            matrix = false;
        } else if (a == "--state") {
            const std::string s = argv[++i];
            if (s == "zeros") one.zero_state = true;
            else if (s == "random") one.zero_state = false;
            else { std::fprintf(stderr, "--state zeros|random\n"); return 2; }
            matrix = false;
        } else if (a == "--single") matrix = false;
        else if (a == "--matrix") matrix = true;
        else {
            std::fprintf(stderr,
                         "usage: gdn_collapse_check [--tokens N --body B --state zeros|random --decay D "
                         "--seed S]\n"
                         "  default --matrix: the body shapes the engine runs at 1024 lanes, the two-body\n"
                         "  carried-state case, and a long-memory case (small decay) where the state\n"
                         "  itself still depends on what was staged\n");
            return 2;
        }
    }
    if (matrix) {
        cases = {
            {"one body of 512, fresh state", 512, 512, true, 7},
            {"two bodies of 512, fresh state", 1024, 512, true, 7},
            {"two bodies of 512, carried state", 1024, 512, false, 11},
            {"one body of 512, carried state", 512, 512, false, 11},
            {"four bodies of 128, carried state", 512, 128, false, 13},
            {"one body of 8, fresh state", 8, 8, true, 3},
            // The long-memory cases: decay small enough that the state at the end of a body still
            // carries its incoming state, so state staging is observable in the state and not only in
            // `o`. A real GDN head is like this (§04.3's A_log spans decades); the default decay of 0.4
            // erases an incoming state within one body and therefore CANNOT see state staging at all.
            {"two bodies of 512, long memory", 1024, 512, false, 17, 0.002f},
            {"one body of 1024, long memory", 1024, 1024, false, 19, 0.002f},
        };
    } else {
        if (one.body > one.tokens) one.body = one.tokens;
        if (one.tokens % one.body != 0) one.body = one.tokens;
        cases = {one};
    }

    int failed = 0;
    for (const auto& c : cases) failed += (run_case(c) != 0);
    std::printf("%s (%d/%zu cases)\n", failed ? "COLLAPSE CHECK FAIL" : "COLLAPSE CHECK PASS",
                (int)cases.size() - failed, cases.size());
    return failed ? 1 : 0;
}
