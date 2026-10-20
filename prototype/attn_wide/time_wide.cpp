/*
 * time_wide.cpp — the A12 long-context timing rig: the wide-shape block and the shipping 16-row WMMA form,
 * paired at one shape inside ONE process. Written, NOT RUN (`AGENTS.md`, "GPU execution is serialized": every
 * measurement belongs to the main agent, and `prototype/attn_percost/A12-WIDE-BET.md` §3 step 3 books this
 * window to it). Nothing in this file has ever printed a millisecond figure.
 *
 * WHAT IT ANSWERS. The BET: does `attn_wide_wave` (128 query rows x a 64-position stage, 8 waves, F16 P·V
 * accumulator) clear **6.5 TFLOP/s at 32,768 lanes**, against the shipping form's kernel-only 4.714
 * (`A12-WIDE-BET.md` §2, whose denominator is `A11-attribution.md` §0: 26.389 TFLOP/pass of useful work over
 * 5,598.220 ms of `attn_fattn_wave_u8` dispatches). The rig prints both forms' ms (best-of-N and every rep),
 * their MAC counts and rates, the driver's occupancy answer, and the geometry triple that tells a reader the
 * two sides are not the same kernel.
 *
 * IT IS NOT AN ENGINE RUN. One shape, launched directly, no GEMMs, no rope, no cache append, no `attn_combine`,
 * no O(n²) reference. What the shipping side pays here is its own kernel and nothing else, which is exactly what
 * A11's 4.714 is (a sum over `attn_fattn_wave_u8` dispatches) and therefore the only comparator that is not a
 * different claim; the excluded `attn_prep`/`attn_combine` are named in the printed note rather than hidden —
 * they are the shipping form's cost, and omitting them flatters the baseline, not the wide form.
 *
 * THE PAIRED BASELINE IS THE SHIPPING KERNEL ITSELF, not a paraphrase. `engine/attn_fattn_kernels.h` is
 * `#include`d read-only and `engine::kernels::attn_fattn_wave_u8` is launched with a `CoreArgs` built from the op
 * header's OWN host-callable rules (`split_for_lanes`, `blocks_for_row`, `lanes_of`, `fattn_covers`,
 * `fattn_rtiles_of` — all `__device__ __host__`), so this rig cannot disagree with the engine about lane blocks,
 * coverage or row tiles the way a hand-written mirror would. Nothing in `engine/` is edited. The wide side comes
 * from `prototype/attn_wide/attn_wide.h`, unchanged, so `attn_wide`'s correctness witness still describes the
 * kernel being timed.
 *
 * THE SHAPE, and the three asymmetries it carries (all printed, none averaged away):
 *   `--lanes N` = query positions = final cache depth; row i attends lanes 0..i (causal), so the useful work is
 *   NQ x 2 x DH x N(N+1)/2 MAC — NQ=8 query heads, NKV=2 KV heads, GQA 4, DH 256, the checkpoint's shape
 *   (`engine/state.h:51` kNumQueryHeads = 8, §00.3). NOT 16 heads / 4 KV groups: that is not this model, and
 *   both kernels hard-code 8/2 (`prototype/attn_decode/attn_decode_kernels.h:22-23`, `attn_wide.h`'s WIDE_NQ /
 *   WIDE_NKV), so a rig that asked for 16/4 would be pricing a model that does not exist.
 *   (1) The wide arm runs the whole causal problem in ONE launch (n = N, grid z = N / WIDE_POS_PER_BLOCK). It has
 *       no lane-split concept — `attn_wide.h` says outright that the `part`/`attn_combine` route is not what it
 *       prices — and no causal-offset field, so it cannot be cut into 512-row waves without editing the kernel
 *       under test, which would make its own correctness witness describe a different kernel.
 *   (2) The shipping arm runs the engine's own geometry: `ceil(N/512)` sequential waves of at most
 *       `kMaxWaveRows = 512` query rows (`engine/decode_step.h:49`) at `split = split_for_lanes(lanes_last)`
 *       lane blocks, because a single 32,768-row launch would need a 17 GB `part` buffer and breaks the wave
 *       rule (AGENTS.md: "a sweep timed at a width the shipping path never issues is a fact about the library,
 *       not the engine"). The arms therefore differ in launch count (1 vs ceil(N/512)); both counts are printed
 *       and the inter-launch-gap share is stated, since this side is wall-clock over a pass and A11's rate was a
 *       sum of pmc dispatch intervals.
 *   (3) The shipping form owns NO rows at or below MAXSC (the `fattn_covers` decision, `attn_fattn_kernels.h`
 *       COVERAGE), so below 1,024 lanes there is nothing to pair and the rig refuses. Above it the covered share
 *       is printed per pass, because a rate on a partly-inert arm is a different claim than a rate on a fully
 *       covered one (the `run_fattn_burst_window.sh` lesson: state the denominator).
 *
 * THE REFUSALS. Every one prints `NO-TIMING: <why>`, exits 2, and prints no ms figure. Each is reachable, and
 * each names the failure mode of this tree it exists to catch:
 *   R1 the correctness driver (`attn_wide`, built from the same source in the same invocation) did not run, did
 *      not print its banner, printed a banner whose ladder axes disagree with this binary's compiled axes,
 *      printed a case list that is not the geometry-derived ten, or printed anything but `SELFTEST PASS … 0
 *      failed`. A driver whose output a parser silently matches nothing is the recorded failure (AGENTS.md, the
 *      gdn_collapse row), so the parse REQUIRES ten case lines equal to the derived list, plus a trailing count
 *      that equals the parsed count.
 *   R2 the timed shape is at/below MAXSC, above the engine's lane ceiling `MAXSC x kAttnMaxSplit = 32,768`, or
 *      otherwise not pairable (`geometry_refusals`).
 *   R3 a launch faulted; or the output buffer still holds the sentinel neither kernel can store (it wrote
 *      nothing, which would read as a large speed-up); or it wrote a non-finite value; or a sampled row misses the
 *      gross-error guard against the driver's reference form.
 *   R4 the two forms' geometry triples are identical or carry a zero field — which is exactly what a stale binary
 *      that ran one side's kernel twice would look like (AGENTS.md, "a paired A/B against a stale binary reads as
 *      a clean null"). `run_a12_window.sh` additionally compares this binary's declared LDS against the LDS in the
 *      code-object listing produced by the SAME build invocation.
 *
 * THE BANDS, and which one is a gate. The driver's bands (worst cosine 0.9995, meanrel 2e-2) are calibrated at
 * <= 512 lanes, as far as its O(n²) reference reaches. At 32,768 lanes the wide form runs 512 rounds instead of
 * 8 with an F16 accumulator, so the streaming-vs-one-pass residual is expected to be larger there for a reason
 * that is not a bug — and a rig that refuses the deciding window on an expected drift is worse than no rig. So:
 * the calibrated aggregates are ALWAYS printed with a verdict, and the hard gate by default is the gross-error
 * guard (cos >= 0.9, meanrel <= 0.15), ~50x looser than the band and still tripped by every named defect class —
 * a permuted row gives "a cosine of a few tenths", half a denominator gives meanrel ~0.5 (`attn_wide.hip.cpp`,
 * the bands' own comment). `A12_TIGHT_BAND=1` promotes the calibrated band to a refusal, e.g. at <= 512 lanes.
 *
 * WHAT IT CANNOT DO (also printed at the end of every run): no engine plumbing, no `attn_prep`/`attn_combine`
 * cost, no numerics admission (the pinned-dump judge is `a12_wide_mirror.py`), no bandwidth claim, no claim that
 * either form is the default, no token-level claim, and no comparison against the PINNED dumps at the timed shape
 * — no pinned prompt is anywhere near 512 lanes, let alone 32,768.
 *
 * Build:  bash prototype/attn_wide/build_time_wide.sh [outdir]    # also writes the .s the register read uses
 * Run:    main agent only, inside one `tools/gpu_lock.sh` hold — `run_a12_window.sh` carries that rule.
 *         <outdir>/time_wide --lanes 32768 --reps 3 --driver <outdir>/attn_wide
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include <hip/hip_runtime.h>

#include "prototype/attn_wide/attn_wide.h"
#include "engine/attn_fattn_kernels.h"      // read-only: the paired baseline IS the shipping kernel

namespace ek = engine::kernels;

namespace {

// ---- the two constants this rig restates, each with the file that owns it ---------------------------------
// `kWaveRows` and `kMaxSplit` are `constexpr` in scopes this TU cannot reach without pulling in the whole engine
// launcher, so they are restated and then asserted against the arithmetic that makes them load-bearing here:
// their product is the engine's lane ceiling, and that is what `--lanes` is measured against.
constexpr int kWaveRows = 512;        // engine/decode_step.h:49 (kMaxWaveRows)
constexpr int kMaxSplit = 64;         // engine/kernels_attn.cpp:82 (kAttnMaxSplit)
static_assert(MAXSC * kMaxSplit == 32768,
              "the engine's lane ceiling is MAXSC x kAttnMaxSplit; if this moved, the 32,768-lane row this rig "
              "exists to price is no longer the widest body the engine can emit");
static_assert(WIDE_NQ == N_Q && WIDE_NKV == N_KV && WIDE_DH == DH,
              "both forms must be priced at ONE model shape. If this fires the rig is timing two different "
              "models and calling the ratio a form comparison");
static_assert(WIDE_SCALE == SCALE, "the score scale is one fact; a different scale is a different op");

constexpr float kSentinel = 12345.0f;      // a value neither kernel can store (outputs are O(1) softmax means)
constexpr double kGrossCosMin = 0.9;       // the hard gate; see the header's band section
constexpr double kGrossMeanRelMax = 0.15;
constexpr double kBandCos = 0.9995;        // the driver's calibrated bands: printed always
constexpr double kBandMeanRel = 2e-2;
constexpr int kDriverCases = 10;           // attn_wide.hip.cpp's derived `cases` list
const char* const kKfdNode = "/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties";

bool tight_band_on() { return getenv("A12_TIGHT_BAND") != nullptr; }

void refuse(const char* fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
void refuse(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fputs("NO-TIMING: ", stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputs("\n  No ms figure was printed: a precondition of this window is unadjudicated.\n", stderr);
    va_end(ap);
    std::exit(2);
}

#define HIP(call)                                                                        \
    do {                                                                                 \
        hipError_t e_ = (call);                                                          \
        if (e_ != hipSuccess)                                                            \
            refuse("hip call `%s` failed: %s (a fault before the timed region means the "  \
                   "shape, not the kernel, is what would have been timed)", #call,         \
                   hipGetErrorString(e_));                                               \
    } while (0)

// Checks that hold whatever the device says, run before a byte is allocated or a kernel is loaded.
void geometry_refusals(int lanes) {
    if (lanes <= 0) refuse("--lanes must be positive");
    if (lanes > MAXSC * kMaxSplit)
        refuse("--lanes %d exceeds the engine's lane ceiling MAXSC x kAttnMaxSplit = %d; the shipping form "
               "cannot be launched deeper, so there is no paired baseline at this width", lanes,
               MAXSC * kMaxSplit);
    if (lanes <= MAXSC)
        refuse("--lanes %d is at or below MAXSC %d, where `fattn_covers` gives the shipping form NO rows: the "
               "baseline would be an inert launch and its ratio a slowdown artefact. Pair only rows that stream.",
               lanes, MAXSC);
    if (lanes % 2 != 0)
        refuse("--lanes %d must be even: with an odd count the last wave's rows do not tile the 16-row M "
               "dimension and the two arms are not comparable row for row", lanes);
}

// ---- allocation ledger: what was allocated, against what the pool has -----------------------------------
struct Ledger {
    size_t total = 0;
    void note(const char* what, size_t bytes) {
        std::printf("  %-26s %10.2f MiB\n", what, double(bytes) / (1024.0 * 1024.0));
        total += bytes;
    }
};

// ---- the reference passes: the driver's policy, sampled so the cost is O(S x n) and not O(n^2) ------------
// Same rounding policy as `attn_wide.hip.cpp`'s reference: score round `bf(bf(dot)*SCALE)`, P rounded to F16,
// `l` summed over the UNROUNDED exp, output `(sum f16(p) * v) / l` in F32 after widening. Restating it here is
// what lets the SAME policy reach the timed shape: the driver's version is one thread per (row, head) and that
// is exactly why it stops at 512. A truncated-attention bug needs no separate control: a form that stopped at
// the first 512 lanes produces a row whose cosine against this full-prefix reference is far below the guard.
__global__ void chk_score_kernel(const WideArgs a, const int* __restrict__ rows, int nsamp, float* pbuf) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long tot = (long long)nsamp * WIDE_NQ * a.npos;
    if (idx >= tot) return;
    const int j = (int)(idx % a.npos);
    const int h = (int)((idx / a.npos) % WIDE_NQ);
    const int s = (int)(idx / ((long long)a.npos * WIDE_NQ));
    const int i = rows[s];
    float* p = pbuf + ((long long)s * WIDE_NQ + h) * a.npos;
    if (j > i) { p[j] = -1e30f; return; }                      // causal: out of the max and of the sum
    const hip_bfloat16* qr = a.q + ((size_t)i * WIDE_NQ + h) * WIDE_DH;
    const hip_bfloat16* kr = a.k + ((size_t)(h / WIDE_GQA) * a.cap + (size_t)j) * WIDE_DH;
    float dot = 0.f;
    for (int d = 0; d < WIDE_DH; ++d) dot += wide_bf(qr[d]) * wide_bf(kr[d]);
    p[j] = wide_score_round(dot);
}

__global__ void chk_norm_kernel(const WideArgs a, const int* __restrict__ rows, int nsamp,
                                float* pbuf, float* lbuf) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;      // one thread per (sample, head)
    if (idx >= nsamp * WIDE_NQ) return;
    const int h = idx % WIDE_NQ;
    const int s = idx / WIDE_NQ;
    const int i = rows[s];
    float* p = pbuf + (long long)idx * a.npos;
    float mx = -1e30f;
    for (int j = 0; j <= i; ++j) mx = fmaxf(mx, p[j]);
    float l = 0.f;
    for (int j = 0; j <= i; ++j) {
        const float e = expf(p[j] - mx);
        l += e;                                                // unrounded, as in the kernel
        p[j] = __half2float(__float2half_rn(e));                // rounded, as in the kernel's P operand
    }
    lbuf[idx] = l;
}

__global__ void chk_out_kernel(const WideArgs a, const int* __restrict__ rows, int nsamp,
                               const float* __restrict__ pbuf, const float* __restrict__ lbuf,
                               float* __restrict__ ref) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;      // over (sample, head, dim)
    const int rh = idx / WIDE_DH;
    if (rh >= nsamp * WIDE_NQ) return;
    const int d = idx % WIDE_DH;
    const int h = rh % WIDE_NQ;
    const int s = rh / WIDE_NQ;
    const int i = rows[s];
    const hip_bfloat16* vb = a.v + (size_t)(h / WIDE_GQA) * a.cap * WIDE_DH;
    const float* p = pbuf + (long long)rh * a.npos;
    float acc = 0.f;
    for (int j = 0; j <= i; ++j) acc += p[j] * wide_bf(vb[(size_t)j * WIDE_DH + d]);
    ref[(long long)rh * WIDE_DH + d] = acc / lbuf[rh];          // widen, then divide: the driver's order
}

// One pass over a float buffer: how many elements still hold the sentinel (nothing wrote them) and how many are
// not finite. On device, because reading 256 MiB back to the host to ask one integer is the slow way round.
__global__ void audit_kernel(const float* __restrict__ p, size_t n, unsigned long long* sentinel_ct,
                             unsigned long long* nonfinite_ct) {
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = p[i];
        if (v == kSentinel) atomicAdd(sentinel_ct, 1ULL);
        if (!isfinite(v)) atomicAdd(nonfinite_ct, 1ULL);
    }
}

__global__ void fill_sentinel_kernel(float* p, size_t n) {
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = kSentinel;
}

// Lanes past the causal shape are POISON, not zeros: an over-read that lands on zeros merely looks small, one
// that lands on 9000 is unmistakable in the finiteness and cosine checks (`attn_wide.hip.cpp`'s rule).
__global__ void fill_poison_kernel(hip_bfloat16* p, size_t n) {
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = hip_bfloat16(9000.0f);
}

// ---- the two arms ---------------------------------------------------------------------------------------
struct WideRun {
    WideArgs a{};
    dim3 grid{WIDE_NKV, 1, 1};
    void launch() const { attn_wide_wave<<<grid, dim3(WIDE_THREADS)>>>(a); }
};

// The shipping arm: ceil(n/kWaveRows) sequential waves, one launch each, with the CoreArgs the engine's own
// wave launcher fills (`engine/kernels_attn.cpp`: staged plan — fused=0, tile_kernel=1, fattn_cover=1, full=1).
struct FattnRun {
    CoreArgs base{};
    hip_bfloat16* qbase = nullptr;      // the staging buffer `attn_prep` would have written; non-const because
                                        // CoreArgs::qrope is (the engine's launcher passes a non-const pointer too)
    int n = 0, waves = 0;
    CoreArgs wave(int w) const {
        const int pos0 = w * kWaveRows;
        const int nrows = std::min(kWaveRows, n - pos0);
        CoreArgs c = base;
        c.qrope = qbase + (size_t)pos0 * ROW_H;   // a staged plan has READ pointers as well as write pointers
        c.pos = pos0;                             // (AGENTS.md: the null-read half of a plan is a hang, not a
        c.lvalid = pos0 + 1;                      //  crash); row qrow attends pos0+qrow+1 lanes, decode_step.cpp:687
        c.nrows = nrows;
        c.split = std::min(kMaxSplit, split_for_lanes(pos0 + nrows));
        return c;
    }
    static dim3 grid_of(const CoreArgs& c) { return dim3(N_Q, c.split, fattn_rtiles_of(c.nrows)); }
    void launch() const {
        for (int w = 0; w < waves; ++w) {
            const CoreArgs c = wave(w);
            ek::attn_fattn_wave_u8<<<grid_of(c), dim3(FATTN_THREADS)>>>(c);
        }
    }
    void launch_wave(int w) const {
        const CoreArgs c = wave(w);
        ek::attn_fattn_wave_u8<<<grid_of(c), dim3(FATTN_THREADS)>>>(c);
    }
};

// ---- work accounting, and the denominator each rate is on ------------------------------------------------
struct Work {
    long long launches = 0, rows = 0, covered_rows = 0;
    long long mac_useful = 0, mac_issued = 0;
    long long active_blocks = 0, rounds_total = 0;      // the shipping arm's round model, for its own witness
};

// Useful MAC over rows [0,n): NQ query heads x 2 matmuls (score + P·V) x DH x n(n+1)/2 causal (row, lane) pairs.
// This is A11 §0's basis — its 26.389 TFLOP/pass is exactly this at 32,768 rows over six layers — so a rate on
// it is the number the BET's 4.714 and 6.5 are stated in. Not counting the P·V half (i.e. DH, not 2·DH) would
// halve every rate here and silently move the abort line. The identity that witnesses the definition: this formula
// at 32,768 rows gives 4.3982 TFLOP per layer, and A11's traced 5,598.220 ms over six layers is 933.0 ms per
// layer, 4.3982/0.9330 = 4.714 TFLOP/s — the BET's own number, reproduced by the rig's denominator arithmetic.
long long useful_mac(int n) {
    return (long long)N_Q * 2 * DH * ((long long)n * (n + 1) / 2);
}

// The wide arm's ISSUED MAC, read off its own loop: `for (base = 0; base < hi; base += WIDE_STAGE)` with
// hi = min(n, pos0 + WIDE_POS_PER_BLOCK), and one round = WIDE_ROWS rows x WIDE_STAGE positions x 2·DH MAC
// (`attn_wide.h`'s per-round arithmetic: 128 x 64 x 512 = 4,194,304). grid.x = WIDE_NKV blocks cover the other
// half of the heads. The gap to useful MAC is the causal overhang the wide shape pays: <= WIDE_STAGE positions
// per block, which is the price its own header books against the shipping form's 48-row union.
long long wide_issued_mac(int n) {
    long long mac = 0;
    for (int pos0 = 0; pos0 < n; pos0 += WIDE_POS_PER_BLOCK) {
        const int hi = std::min(n, pos0 + WIDE_POS_PER_BLOCK);
        const long long rounds = (hi + WIDE_STAGE - 1) / WIDE_STAGE;
        mac += rounds * (long long)WIDE_ROWS * WIDE_STAGE * (2 * WIDE_DH) * WIDE_NKV;
    }
    return mac;
}

// The shipping arm's ISSUED MAC, from the source rule rather than a guess: `attn_fattn_kernels.h`'s "THE ROUND
// RANGE MUST BE BLOCK-UNIFORM" fold makes a block's round range the union over its three row-blocks (48 row-slots)
// of the covered rows' [blo, bhi), rounded up to FATTN_MLANES = 16; a fully-masked row still costs its issue, so
// a round is 48 row-slots x 16 lanes x 2·DH MAC. The model self-witnesses: A11 §1 counted 33.29 rounds/block on
// the widest traced body against `ceil(span/16) ~= 32`, and the mean printed here should land in the same place.
Work fattn_work(const CoreArgs& base, int n) {
    Work w;
    w.mac_useful = useful_mac(n);
    for (int w0 = 0; w0 < n; w0 += kWaveRows) {
        const int nrows = std::min(kWaveRows, n - w0);
        CoreArgs c = base;
        c.pos = w0;
        c.lvalid = w0 + 1;
        c.nrows = nrows;
        c.split = std::min(kMaxSplit, split_for_lanes(w0 + nrows));
        const int rt = fattn_rtiles_of(nrows);
        w.launches += (long long)N_Q * c.split * rt;
        for (int r = 0; r < nrows; ++r) {
            ++w.rows;
            if (fattn_covers(c, r)) ++w.covered_rows;
        }
        for (int b = 0; b < c.split; ++b) {
            for (int t = 0; t < rt; ++t) {
                int lo = 1 << 30, hi = 0;
                const int r_end = std::min(nrows, (t + 1) * FATTN_ROWS_PER_BLOCK);
                for (int r = t * FATTN_ROWS_PER_BLOCK; r < r_end; ++r) {
                    if (!fattn_covers(c, r)) continue;
                    const int s_row = blocks_for_row(c, r);
                    const int nrow = lanes_of(c, r);
                    const int span = (nrow + s_row - 1) / s_row;
                    const int blo = b * span;
                    const int bhi = std::min(blo + span, nrow);
                    if (bhi <= blo) continue;
                    lo = std::min(lo, blo);
                    hi = std::max(hi, bhi);
                }
                if (hi <= lo) continue;                    // `if (!any_mask) return;` — this block does nothing
                const long long rounds = (hi - lo + FATTN_MLANES - 1) / FATTN_MLANES;
                ++w.active_blocks;
                w.rounds_total += rounds;
                w.mac_issued += rounds * (long long)FATTN_ROWS_PER_BLOCK * FATTN_MLANES * (2 * DH);
            }
        }
    }
    return w;
}

Work wide_work(int n) {
    Work w;
    w.mac_useful = useful_mac(n);
    w.mac_issued = wide_issued_mac(n);
    w.launches = 1;
    w.rows = n;
    w.covered_rows = n;              // the wide form owns every row: it has no coverage rule to opt out of
    return w;
}

// ---- R1: the correctness driver's witness, taken from the driver itself ----------------------------------
// The rig runs the driver rather than trusting a file, so the refusal cannot be talked around by a stale log.
std::vector<int> expected_driver_rows() {
    const int P = WIDE_POS_PER_BLOCK, S = WIDE_STAGE;
    std::vector<int> v = {1, P - 1, P, P + 1, S - 1, S, S + 1, 2 * S + 2, 300, 512};
    std::sort(v.begin(), v.end());
    return v;
}

void driver_witness(const std::string& path, int lanes) {
    if (access(path.c_str(), X_OK) != 0)
        refuse("correctness driver `%s` is not executable. Build it with `bash prototype/attn_wide/build_wide.sh "
               "<outdir>` in the same invocation and the same WIDE_* axes as this binary (run_a12_window.sh does "
               "both). Without it nothing here has ever been checked against attention.", path.c_str());
    FILE* f = popen(("\"" + path + "\" 2>&1").c_str(), "r");
    if (!f) refuse("cannot run the correctness driver `%s`", path.c_str());
    char line[1024];
    bool have_banner = false, have_final = false;
    int bw = -1, bt = -1, brs = -1, bs = -1, bq = -1, bp = -1, bpp = -1;
    std::vector<int> got;
    int n_pass = 0, n_fail = 0, final_cases = -1, final_failed = -1;
    bool final_pass = false;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, "attn_wide correctness:", 22) == 0) {
            if (std::sscanf(line, "attn_wide correctness: warps=%d threads=%d rows=%d stage=%d qreg=%d pvf16=%d "
                                  "pos_per_block=%d", &bw, &bt, &brs, &bs, &bq, &bp, &bpp) != 7)
                refuse("could not parse the driver's banner (`%s`); a banner the rig cannot read is a witness the "
                       "rig does not have", line);
            have_banner = true;
        }
        if (const char* np = std::strstr(line, "npos=")) {
            int v = -1;
            if (std::sscanf(np, "npos=%d", &v) == 1 && v >= 0) {
                got.push_back(v);
                if (std::strstr(line, " PASS")) ++n_pass;
                if (std::strstr(line, " FAIL")) ++n_fail;
            }
        }
        if (std::strstr(line, "case(s),")) {
            have_final = true;
            final_pass = std::strstr(line, "SELFTEST PASS") != nullptr;
            if (std::sscanf(line, "%*[^:]: %d case(s), %d failed", &final_cases, &final_failed) != 2)
                refuse("could not parse the driver's verdict line (`%s`)", line);
        }
    }
    const int status = pclose(f);
    if (!have_banner) refuse("the driver printed no `attn_wide correctness:` banner — did it run at all?");
    if (!have_final) refuse("the driver printed no verdict line; a driver that ends silently has not adjudicated "
                            "anything (status %d)", status);
    if (bw != WIDE_WARPS || bt != WIDE_THREADS || brs != WIDE_ROWS || bs != WIDE_STAGE || bq != WIDE_QREG ||
        bp != WIDE_PVF16 || bpp != WIDE_POS_PER_BLOCK)
        refuse("the driver that passed was compiled for a different shape than the kernel being timed: driver "
               "warps=%d threads=%d rows=%d stage=%d qreg=%d pvf16=%d pos_per_block=%d vs this binary warps=%d "
               "threads=%d rows=%d stage=%d qreg=%d pvf16=%d pos_per_block=%d. Build both from one invocation "
               "(run_a12_window.sh does).", bw, bt, brs, bs, bq, bp, bpp, WIDE_WARPS, WIDE_THREADS, WIDE_ROWS,
               WIDE_STAGE, WIDE_QREG, WIDE_PVF16, WIDE_POS_PER_BLOCK);
    std::sort(got.begin(), got.end());
    const std::vector<int> want = expected_driver_rows();
    if (got != want) {
        std::string s;
        for (int x : got) s += std::to_string(x) + ",";
        std::string t;
        for (int x : want) t += std::to_string(x) + ",";
        refuse("the driver's case list is not the geometry-derived ten (got [%s] want [%s]). A driver whose cases "
               "a parser silently does not match is the recorded empty-parse failure; refusing rather than "
               "reporting PASS over a list that was not read.", s.c_str(), t.c_str());
    }
    if (n_fail != 0 || n_pass != kDriverCases || final_failed != 0 || final_cases != kDriverCases || !final_pass)
        refuse("the correctness driver did not pass (%d pass, %d fail, verdict says %d case(s) %d failed). The "
               "kernel being timed is not adjudicated; nothing is timed.", n_pass, n_fail, final_cases,
               final_failed);
    const int reach = got.back();
    std::printf("driver_witness: PASS cases=%d (all of %s) axes match this binary (warps=%d stage=%d qreg=%d "
                "pvf16=%d)\n", kDriverCases, "[1,POS-1,POS,POS+1,STAGE-1,STAGE,STAGE+1,2*STAGE+2,300,512]",
                WIDE_WARPS, WIDE_STAGE, WIDE_QREG, WIDE_PVF16);
    if (lanes > reach)
        std::printf("driver_witness: its O(n^2) reference reaches %d lanes, the timed shape is %d. Numerics at "
                    "the timed shape are adjudicated by the sampled-row form of the SAME reference policy (below) "
                    "against the gross-error guard cos>=%.2f / meanrel<=%.2f; the calibrated band (%.4f / %.0e) is "
                    "printed and %s. Set A12_TIGHT_BAND=1 to gate on it.\n",
                    reach, lanes, kGrossCosMin, kGrossMeanRelMax, kBandCos, kBandMeanRel,
                    tight_band_on() ? "GATES" : "is advisory");
    else
        std::printf("driver_witness: the timed shape (%d) is inside the driver's own reach (%d lanes).\n", lanes,
                    reach);
    if (WIDE_MINB != 1)
        std::printf("note: the driver's banner carries no WIDE_MINB field, so this witness cannot tell two builds "
                    "apart that differ only in __launch_bounds__ min-blocks (compiled here: WIDE_MINB=%d). Pass the "
                    "same WIDE_MINB to build_wide.sh and build_time_wide.sh.\n", WIDE_MINB);
}

// The rows the sampled check runs, DERIVED from the geometry rather than hand-listed (the repo's rule for a
// must-catch set): the wide block edge, the round/stage edge, the MAXSC coverage edge, mid-range and the last
// round's edges. Every one is a place a mask, a clamp or a round boundary can be wrong and nowhere else is.
std::vector<int> sampled_rows(int n) {
    const int B = WIDE_POS_PER_BLOCK, S = WIDE_STAGE, M = MAXSC;
    const int cand[] = {0, 1, B - 1, B, B + 1, S - 1, S, S + 1, M - 1, M, M + 1,
                        n / 2 - B, n / 2, n / 2 + B, n - S - 1, n - S, n - S + 1, n - 2, n - 1};
    std::vector<int> v;
    for (int x : cand) if (x >= 0 && x < n) v.push_back(x);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    if (v.empty()) refuse("the derived sample-row list is empty at n=%d; a check with no cases cannot fail", n);
    return v;
}

// ---- the driver's occupancy query, and the device facts it must NOT come from -----------------------------
struct Occ {
    int blocks_per_cu = 0, threads = 0;
    long long wf_per_cu = 0;
    int regs = 0;
    size_t shared = 0, local = 0;
};

template <typename T>
Occ occupancy(T* fn, int threads, const char* name) {
    Occ o;
    o.threads = threads;
    HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&o.blocks_per_cu, fn, threads, 0));
    hipFuncAttributes at{};
    // The HIP API takes the kernel as `const void*`; the occupancy call above has a template overload and this
    // one does not, hence the cast (the same form HIP's own examples use).
    if (hipFuncGetAttributes(&at, (const void*)fn) != hipSuccess)
        refuse("hipFuncGetAttributes failed for %s; without its declared LDS the canary cannot compare this "
               "binary against the listing of the same build", name);
    o.regs = at.numRegs;
    o.shared = at.sharedSizeBytes;
    o.local = at.localSizeBytes;
    o.wf_per_cu = (long long)o.blocks_per_cu * threads / 32;
    return o;
}

// CU and SIMD counts come from KFD, never from `hipGetDeviceProperties`, which on this SKU reports a halved CU
// count and a per-thread waves/SIMD ceiling (AGENTS.md, the counter/occupancy row). Printed as context: the
// occupancy API above answers blocks/CU directly, so nothing here needs a CU count to compute wavefronts/CU.
void kfd_facts() {
    FILE* f = std::fopen(kKfdNode, "r");
    if (!f) {
        std::printf("device facts: KFD %s unreadable — CU/SIMD/LDS counts NOT printed (never taken from "
                    "hipGetDeviceProperties, which reads 2x off on this SKU)\n", kKfdNode);
        return;
    }
    char line[256];
    long simd = 0, per_cu = 0, lds_kb = 0;
    while (std::fgets(line, sizeof line, f)) {
        long v;
        if (std::sscanf(line, "simd_count %ld", &v) == 1) simd = v;
        if (std::sscanf(line, "simd_per_cu %ld", &v) == 1) per_cu = v;
        if (std::sscanf(line, "lds_size_in_kb %ld", &v) == 1) lds_kb = v;
    }
    std::fclose(f);
    if (simd <= 0 || per_cu <= 0)
        std::printf("device facts: KFD node parsed but simd_count/simd_per_cu missing — not printed\n");
    else
        std::printf("device facts (KFD %s): SIMDs=%ld CUs=%ld LDS/CU=%ld KiB\n", kKfdNode, simd, simd / per_cu,
                    lds_kb);
}

// ---- the sampled comparison ------------------------------------------------------------------------------
struct Lcg {                       // the driver's generator, so the input distribution is the same one the
    unsigned long long s = 88172645463325252ull;   // bands were calibrated against
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long r = (s >> 33) & 0xffff;
        return 6.0f * ((float)r / 65536.0f) - 3.0f;
    }
};

struct Agg {
    double worst_cos = 1.0, worst_meanrel = 0.0;
    int nonfinite = 0, rows = 0, argmove = 0;
};

// Per (sample, head) row over DH, exactly as `attn_wide.hip.cpp` aggregates: the worst row is what is gated,
// because a wrong row/head/dim mapping moves ONE row's cosine and a whole-tensor aggregate averages it away.
Agg compare_rows(const float* got, const float* want, int nrows) {
    Agg a;
    for (int i = 0; i < nrows; ++i) {
        const float* x = got + (size_t)i * WIDE_DH;
        const float* y = want + (size_t)i * WIDE_DH;
        double dot = 0, nx = 0, ny = 0, num = 0, den = 0;
        double bg = -1e30, br = -1e30;
        int ag = 0, ar = 0;
        for (int d = 0; d < WIDE_DH; ++d) {
            if (!std::isfinite(x[d])) ++a.nonfinite;
            dot += (double)x[d] * y[d]; nx += (double)x[d] * x[d]; ny += (double)y[d] * y[d];
            num += std::fabs((double)x[d] - y[d]); den += std::fabs(y[d]);
            if (x[d] > bg) { bg = x[d]; ag = d; }
            if (y[d] > br) { br = y[d]; ar = d; }
        }
        const double cos = (nx > 0 && ny > 0) ? dot / (sqrt(nx) * sqrt(ny)) : 0.0;
        const double mr = den > 0 ? num / den : 0.0;
        a.worst_cos = std::min(a.worst_cos, cos);
        a.worst_meanrel = std::max(a.worst_meanrel, mr);
        if (ag != ar) ++a.argmove;           // printed, not gated: the guard is the row cosine and meanrel
        ++a.rows;
    }
    return a;
}

// Device-side audit of a buffer: how many elements still hold the sentinel (nothing wrote them) and how many are
// not finite, each compared against an EXPECTED count derived from the geometry. Refuses on a mismatch, because
// the failure each one hides is a fast-looking number: a form that wrote nothing is very fast, and a form that
// wrote NaNs is not an attention result at all. The expected sentinel count is not always zero — a row the
// shipping form does not own is supposed to keep it, and that count is arithmetic on `fattn_covers` rather than
// a loosened check.
void audit(const float* p, size_t n, const char* what, unsigned long long expect_sentinel) {
    unsigned long long* d = nullptr;
    HIP(hipMalloc((void**)&d, 2 * sizeof(unsigned long long)));
    HIP(hipMemset(d, 0, 2 * sizeof(unsigned long long)));
    audit_kernel<<<2048, 256>>>(p, n, d, d + 1);
    HIP(hipDeviceSynchronize());
    unsigned long long sent = 0, nonf = 0;
    HIP(hipMemcpy(&sent, d, sizeof sent, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(&nonf, d + 1, sizeof nonf, hipMemcpyDeviceToHost));
    HIP(hipFree(d));
    if (nonf) refuse("%s wrote %llu non-finite values at the timed shape; that is not an attention result", what, nonf);
    if (sent != expect_sentinel)
        refuse("%s left %llu unwritten elements where the geometry predicts %llu of %zu (the sentinel neither "
               "kernel can store survived somewhere it should not have, or a slot that must be written was not). "
               "An arm that does less work than it claims looks exactly this fast.", what, sent, expect_sentinel, n);
    std::printf("audit %-26s elements=%zu sentinel_left=%llu (expected %llu) nonfinite=0 COVERAGE OK\n", what, n,
                sent, expect_sentinel);
}

void fill_sentinel(float* p, size_t n) { fill_sentinel_kernel<<<2048, 256>>>(p, n); HIP(hipDeviceSynchronize()); }

template <typename F>
double timed_ms(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    HIP(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void print_limits() {
    std::printf(
        "limits: this rig cannot (a) price attn_prep/attn_combine or the §05.7 append, so the shipping side is\n"
        "        its kernel only — which is what A11's 4.714 is, and it FLATTERS the baseline, not the wide form;\n"
        "        (b) admit numerics: the pinned-dump judge is prototype/attn_wide/a12_wide_mirror.py and no pinned\n"
        "        prompt reaches 512 lanes, let alone 32,768; (c) make any bandwidth or occupancy claim beyond the\n"
        "        driver's blocks/CU answer; (d) tell you which form should be the default — the default trio and\n"
        "        the >= 3 %% deciding-row rule from the BET are the main agent's call, not this file's;\n"
        "        (e) distinguish two builds that differ only in WIDE_MINB (the driver's banner does not carry it);\n"
        "        (f) put the shipping arm's uncovered rows anywhere: its mac_useful covers every row of the pass\n"
        "            because that is A11's basis, and in an engine run the rows at or below MAXSC are\n"
        "            `attn_slim_wave`'s and are not free.\n");
}

void usage(const char* argv0) {
    std::printf(
        "time_wide — A12 wide-vs-shipping paired attention timing rig. MAIN AGENT ONLY, blocking, one gpu_lock\n"
        "hold, zero subagents. Run it through prototype/attn_wide/run_a12_window.sh, which owns the lock, the\n"
        "build and the register read. Refuses (exit 2, `NO-TIMING:`, no ms printed) unless the correctness driver\n"
        "passed at these compiled axes and both arms pass a structural check at the timed shape.\n"
        "\n"
        "  %s --lanes N --reps R [--driver PATH] [--abort-tflops X] [--help]\n"
        "\n"
        "  --lanes N          query positions = final cache depth, causal row i attends 0..i. Must be > MAXSC\n"
        "                     (%d) and <= the engine's lane ceiling MAXSC x kAttnMaxSplit = %d. The wide arm runs\n"
        "                     it as ONE launch; the shipping arm runs ceil(N/%d) waves of <= %d rows, as shipped.\n"
        "  --reps R           timed repetitions per arm, >= 2. Every rep is printed and the best is reported.\n"
        "  --driver PATH      the attn_wide correctness binary (built from the same source, same WIDE_* axes).\n"
        "  --abort-tflops X   the BET's abort line, reported against the wide arm (default 6.5).\n"
        "\n"
        "Env: A12_TIGHT_BAND=1 gates the sampled-row numerics on the driver's calibrated bands (%.4f / %.0e)\n"
        "     instead of the gross-error guard (cos >= %.2f, meanrel <= %.2f).\n"
        "     WIDE_WARPS / WIDE_STAGE / WIDE_QREG / WIDE_PVF16 / WIDE_MINB are build-time, via build_time_wide.sh.\n",
        argv0, MAXSC, MAXSC * kMaxSplit, kWaveRows, kWaveRows, kBandCos, kBandMeanRel, kGrossCosMin,
        kGrossMeanRelMax);
}

}  // namespace

int main(int argc, char** argv) {
    int lanes = 0, reps = 0;
    std::string driver = "attn_wide";
    double abort_tflops = 6.5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) refuse("%s needs a value", name);
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--lanes") lanes = std::atoi(val("--lanes").c_str());
        else if (a == "--reps") reps = std::atoi(val("--reps").c_str());
        else if (a == "--driver") driver = val("--driver");
        else if (a == "--abort-tflops") abort_tflops = std::atof(val("--abort-tflops").c_str());
        else refuse("unknown argument `%s` (see --help)", a.c_str());
    }
    if (lanes == 0) refuse("--lanes is required (see --help)");
    if (reps == 0) refuse("--reps is required: a timing rig with an implicit repetition count reports a "
                          "best-of-one and calls it a best");
    if (abort_tflops <= 0) refuse("--abort-tflops must be positive");
    geometry_refusals(lanes);
    if (reps < 2) refuse("--reps must be >= 2: with one rep `best` and `only` are the same number and the spread "
                         "this box is known for (bench/performance-status.md §6) is invisible");

    // ---- R1: the correctness driver's witness, BEFORE anything is allocated or timed ----------------------
    driver_witness(driver, lanes);

    HIP(hipSetDevice(0));
    size_t free_b = 0, total_b = 0;
    HIP(hipMemGetInfo(&free_b, &total_b));

    const int n = lanes;
    const int cap_alloc = n + 64;                        // poison tail: an over-read is then unmistakable
    const int waves = (n + kWaveRows - 1) / kWaveRows;
    const std::vector<int> samps = sampled_rows(n);
    const int ns = (int)samps.size();

    const size_t el_q = (size_t)n * ROW_H;
    const size_t el_kv = (size_t)N_KV * cap_alloc * DH;
    const size_t el_out = (size_t)n * N_Q * DH;
    const int split_max = std::min(kMaxSplit, split_for_lanes(n));
    const size_t el_part = (size_t)kWaveRows * N_Q * split_max * (DH + 2);
    const size_t el_pbuf = (size_t)ns * N_Q * n;
    const size_t bytes = el_q * 2 + 2 * el_kv * 2 + el_out * 4 + el_part * 4 + el_pbuf * 4
                       + (size_t)ns * N_Q * 4 * 2 + (size_t)ns * N_Q * DH * 4;

    std::printf("=== time_wide: A12 wide form vs the shipping 16-row WMMA form, paired in one process ===\n");
    std::printf("compiled: WIDE_WARPS=%d WIDE_THREADS=%d WIDE_ROWS=%d WIDE_STAGE=%d WIDE_QREG=%d WIDE_PVF16=%d "
                "WIDE_MINB=%d\n", WIDE_WARPS, WIDE_THREADS, WIDE_ROWS, WIDE_STAGE, WIDE_QREG, WIDE_PVF16,
                WIDE_MINB);
    std::printf("shape:    NQ=%d NKV=%d GQA=%d DH=%d MAXSC=%d kWaveRows=%d kMaxSplit=%d  (the checkpoint's 8 "
                "query heads / 2 KV heads, engine/state.h:51)\n", N_Q, N_KV, GQA, DH, MAXSC, kWaveRows, kMaxSplit);
    std::printf("timed:    lanes=%d reps=%d  wide: 1 launch (n=%d, grid %d x %d)   shipping: %d wave(s) of <= %d "
                "rows, split up to %d\n", n, reps, n, WIDE_NKV,
                (n + WIDE_POS_PER_BLOCK - 1) / WIDE_POS_PER_BLOCK, waves, kWaveRows, split_max);
    std::printf("pool:     total=%.2f GiB free=%.2f GiB (device-visible pool ~34 GiB, "
                "devices/strix-halo-unified-memory.md — this model uses ~4%% of it)\n",
                double(total_b) / (1024.0 * 1024.0 * 1024.0), double(free_b) / (1024.0 * 1024.0 * 1024.0));
    std::printf("allocations (synthetic, one set shared by BOTH arms — same bytes, two kernels):\n");
    Ledger led;
    led.note("q (bf16 [row][head][dim])", el_q * 2);
    led.note("k cache (bf16 [kv][cap][dim])", el_kv * 2);
    led.note("v cache (bf16 [kv][cap][dim])", el_kv * 2);
    led.note("out F32 (wide arm)", el_out * 4);
    led.note("part F32 (shipping arm)", el_part * 4);
    led.note("sampled reference", el_pbuf * 4 + (size_t)ns * N_Q * DH * 4 + (size_t)ns * N_Q * 8);
    std::printf("  %-26s %10.2f MiB  = %.2f %% of the pool\n", "TOTAL", double(bytes) / (1024.0 * 1024.0),
                100.0 * double(bytes) / double(total_b));
    if (bytes + (1ull << 30) > free_b)
        refuse("the rig needs %.2f MiB plus a 1 GiB margin against %.2f MiB free; refusing to allocate into a "
               "nearly-full pool (hipMalloc fails loudly where managed memory would degrade silently)",
               double(bytes) / (1024.0 * 1024.0), double(free_b) / (1024.0 * 1024.0));

    hip_bfloat16 *q = nullptr, *k = nullptr, *v = nullptr;
    float *out = nullptr, *part = nullptr, *pbuf = nullptr, *lbuf = nullptr, *ref = nullptr;
    int* drows = nullptr;
    HIP(hipMalloc((void**)&q, el_q * 2));
    HIP(hipMalloc((void**)&k, el_kv * 2));
    HIP(hipMalloc((void**)&v, el_kv * 2));
    HIP(hipMalloc((void**)&out, el_out * 4));
    HIP(hipMalloc((void**)&part, el_part * 4));
    HIP(hipMalloc((void**)&pbuf, el_pbuf * 4));
    HIP(hipMalloc((void**)&lbuf, (size_t)ns * N_Q * 4));
    HIP(hipMalloc((void**)&ref, (size_t)ns * N_Q * DH * 4));
    HIP(hipMalloc((void**)&drows, (size_t)ns * sizeof(int)));

    // One deterministic fill shared by both arms (the driver's generator, so the input distribution the bands
    // were calibrated against is the one being timed).
    {
        Lcg rng;
        std::vector<hip_bfloat16> tmp(1 << 20);
        auto put = [&](hip_bfloat16* dst, size_t count) {
            while (count) {
                const size_t c = std::min(count, tmp.size());
                for (size_t i = 0; i < c; ++i) tmp[i] = hip_bfloat16(rng.next());
                HIP(hipMemcpy(dst, tmp.data(), c * 2, hipMemcpyHostToDevice));
                dst += c;
                count -= c;
            }
        };
        put(q, el_q);
        put(k, el_kv);
        put(v, el_kv);
        for (int h = 0; h < N_KV; ++h) {
            fill_poison_kernel<<<64, 256>>>(k + ((size_t)h * cap_alloc + n) * DH, (size_t)64 * DH);
            fill_poison_kernel<<<64, 256>>>(v + ((size_t)h * cap_alloc + n) * DH, (size_t)64 * DH);
        }
        HIP(hipDeviceSynchronize());
    }
    HIP(hipMemcpy(drows, samps.data(), (size_t)ns * sizeof(int), hipMemcpyHostToDevice));

    // ---- the two arms ----------------------------------------------------------------------------------
    WideRun wide;
    wide.a.q = q; wide.a.k = k; wide.a.v = v; wide.a.out = out;
    wide.a.npos = n; wide.a.cap = cap_alloc;
    wide.grid = dim3(WIDE_NKV, 1, (n + WIDE_POS_PER_BLOCK - 1) / WIDE_POS_PER_BLOCK);

    FattnRun fat;
    fat.qbase = q; fat.n = n; fat.waves = waves;
    CoreArgs& c = fat.base;
    c.kcache = k; c.vcache = v; c.part = part;
    c.cap = cap_alloc; c.neg = 0; c.full = 1; c.fused = 0; c.score_f32 = 0;
    c.tile = std::min(TILE, THREADS / 32);
    c.tile_kernel = 1; c.fattn_cover = 1;
    {
        std::string why;
        if (!ek::fattn_plan_ok(fat.base, &why))
            refuse("the shipping form refuses this launch plan: %s", why.c_str());
    }
    if (waves < 2)
        refuse("lanes=%d is %d wave(s); at one wave every row is at or below MAXSC and the shipping form owns "
               "nothing", n, waves);

    const Work ww = wide_work(n);
    const Work fw = fattn_work(fat.base, n);
    const Occ ow = occupancy(attn_wide_wave, WIDE_THREADS, "attn_wide_wave");
    const Occ of = occupancy(ek::attn_fattn_wave_u8, FATTN_THREADS, "attn_fattn_wave_u8");
    const CoreArgs lastc = fat.wave(waves - 1);

    // ---- R4: the canary triple, printed before any timing and required to DIFFER --------------------------
    std::printf("CANARY form=wide     kernel=attn_wide_wave threads=%d rows_per_block=%d lanes_per_round=%d "
                "warps=%d stage=%d qreg=%d pvf16=%d minb=%d grid=(%d,%d,%d) launches_per_pass=%lld "
                "lds_declared=%d lds_reported=%zu scratch_reported=%zu regs_reported=%d blocks_per_CU=%d "
                "wf_per_CU=%lld\n",
                WIDE_THREADS, WIDE_ROWS, WIDE_STAGE, WIDE_WARPS, WIDE_STAGE, WIDE_QREG, WIDE_PVF16, WIDE_MINB,
                wide.grid.x, wide.grid.y, wide.grid.z, ww.launches, wide_lds_halves() * 2, ow.shared, ow.local,
                ow.regs, ow.blocks_per_cu, ow.wf_per_cu);
    std::printf("CANARY form=fattn_u8 kernel=attn_fattn_wave_u8 threads=%d rows_per_block=%d lanes_per_round=%d "
                "warps=%d stage=16 qreg=0 pvf16=0 minb=0 grid=(%d,%d,%d) launches_per_pass=%lld "
                "lds_declared=%d lds_reported=%zu scratch_reported=%zu regs_reported=%d blocks_per_CU=%d "
                "wf_per_CU=%lld\n",
                FATTN_THREADS, FATTN_ROWS_PER_BLOCK, FATTN_MLANES, FATTN_WARPS, N_Q, lastc.split,
                fattn_rtiles_of(lastc.nrows), fw.launches,
                FATTN_WARPS * ek::fattn_warp_stage_halves(true) * 2 + FATTN_MLANES * DH * 2, of.shared,
                of.local, of.regs, of.blocks_per_cu, of.wf_per_cu);
    std::printf("CANARY note: grid for the shipping arm is its WIDEST wave (%d rows, split %d); it varies per "
                "wave, and %d waves are issued per pass\n", lastc.nrows, lastc.split, waves);
    if (ow.shared != (size_t)wide_lds_halves() * 2)
        refuse("the wide kernel's declared LDS is %zu B but `wide_lds_halves()` says %d B: this binary was not "
               "built from the header whose stage this prices", ow.shared, wide_lds_halves() * 2);
    if (of.shared != (size_t)(FATTN_WARPS * ek::fattn_warp_stage_halves(true) * 2 + FATTN_MLANES * DH * 2))
        refuse("attn_fattn_wave_u8 declares %zu B of LDS, not the 20,480 B the `unstaged8` spelling declares — "
               "the launch would be the wrong spelling of the shipping form (staged `attn_fattn_wave` declares "
               "32,768 B)", of.shared);
    // The triple has to DIFFER, as a triple. Requiring every field to differ would be stricter than the failure
    // it guards and would refuse a legitimate ladder row (`WIDE_STAGE=16` is the shipping stage depth, so a
    // stage-16 wide block would trip a field-by-field test); requiring only that they are not all equal is what a
    // stale binary cannot fake — one kernel launched twice reports one triple, and a zero field means the canary
    // was never filled in, which is the same vacuity wearing a different hat.
    if (WIDE_THREADS <= 0 || WIDE_ROWS <= 0 || WIDE_STAGE <= 0 || FATTN_THREADS <= 0 || FATTN_ROWS_PER_BLOCK <= 0
        || FATTN_MLANES <= 0)
        refuse("a canary geometry field is zero (threads %d/%d, rows %d/%d, lanes per round %d/%d): an unfilled "
               "canary cannot witness anything", WIDE_THREADS, FATTN_THREADS, WIDE_ROWS, FATTN_ROWS_PER_BLOCK,
               WIDE_STAGE, FATTN_MLANES);
    if (WIDE_THREADS == FATTN_THREADS && WIDE_ROWS == FATTN_ROWS_PER_BLOCK && WIDE_STAGE == FATTN_MLANES)
        refuse("the two arms report the same geometry triple (threads %d, rows per block %d, lanes per round %d): "
               "one binary launching the same kernel on both sides looks exactly like this, and a null result "
               "would then be indistinguishable from a stale build. Two shapes this alike also have no business "
               "being paired — pick a wide shape the shipping form is not.", WIDE_THREADS, WIDE_ROWS, WIDE_STAGE);
    if (ow.blocks_per_cu <= 0 || of.blocks_per_cu <= 0)
        refuse("the driver's occupancy query returned %d and %d blocks/CU; a zero would make every wavefront "
               "count below a guess", ow.blocks_per_cu, of.blocks_per_cu);
    kfd_facts();
    std::printf("registers/scratch: the authoritative `.amdhsa_next_free_vgpr` and private-scratch figures come "
                "from the code-object listing (a11_read_vgpr.py / wide_regread.py, printed by "
                "run_a12_window.sh against THIS build); `*_reported` above is `hipFuncGetAttributes` and is a "
                "cross-check, not the citation.\n");

    // ---- R3: both arms at the timed shape, once, structurally -------------------------------------------
    fill_sentinel(out, el_out);
    wide.launch();
    HIP(hipGetLastError());
    HIP(hipDeviceSynchronize());
    audit(out, el_out, "wide out", 0);

    // The sampled reference: the driver's policy at the timed shape. Cost is O(ns x n), not O(n^2).
    {
        const long long tot = (long long)ns * N_Q * n;
        chk_score_kernel<<<(unsigned)((tot + 255) / 256), 256>>>(wide.a, drows, ns, pbuf);
        chk_norm_kernel<<<(unsigned)((ns * N_Q + 127) / 128), 128>>>(wide.a, drows, ns, pbuf, lbuf);
        chk_out_kernel<<<(unsigned)(((long long)ns * N_Q * DH + 255) / 256), 256>>>(wide.a, drows, ns, pbuf, lbuf, ref);
        HIP(hipGetLastError());
        HIP(hipDeviceSynchronize());
        std::vector<float> want((size_t)ns * N_Q * DH), got((size_t)ns * N_Q * DH);
        HIP(hipMemcpy(want.data(), ref, want.size() * 4, hipMemcpyDeviceToHost));
        for (int s = 0; s < ns; ++s)
            HIP(hipMemcpy(got.data() + (size_t)s * N_Q * DH, out + (size_t)samps[s] * N_Q * DH,
                          (size_t)N_Q * DH * 4, hipMemcpyDeviceToHost));
        const Agg ag = compare_rows(got.data(), want.data(), ns * N_Q);
        const bool inside = ag.worst_cos >= kBandCos && ag.worst_meanrel <= kBandMeanRel;
        std::printf("numerics sampled rows=%d (derived: block/stage/MAXSC/last-round edges) worst_cos=%.7f "
                    "worst_meanrel=%.3e argmax_moved=%d/%d nonfinite=%d | guard cos>=%.2f meanrel<=%.2f %s | "
                    "calibrated band %.4f / %.0e: %s%s\n",
                    ns, ag.worst_cos, ag.worst_meanrel, ag.argmove, ns * N_Q, ag.nonfinite, kGrossCosMin,
                    kGrossMeanRelMax, "OK", kBandCos, kBandMeanRel, inside ? "inside" : "OUTSIDE (advisory)",
                    tight_band_on() ? " — GATED by A12_TIGHT_BAND" : "");
        if (ag.nonfinite) refuse("the wide arm wrote %d non-finite sampled values at lanes=%d", ag.nonfinite, n);
        if (ag.worst_cos < kGrossCosMin)
            refuse("sampled-row guard: worst_cos=%.6f < %.2f at lanes=%d. A permuted or truncated row is the "
                   "class this catches (a relabelled fragment map gives a cosine of a few tenths); nothing is "
                   "timed.", ag.worst_cos, kGrossCosMin, n);
        if (ag.worst_meanrel > kGrossMeanRelMax)
            refuse("sampled-row guard: worst_meanrel=%.3e > %.2f at lanes=%d (half a denominator gives ~0.5). "
                   "Nothing is timed.", ag.worst_meanrel, kGrossMeanRelMax, n);
        if (tight_band_on() && !inside)
            refuse("A12_TIGHT_BAND=1: the calibrated band (cos >= %.4f, meanrel <= %.0e) failed at lanes=%d with "
                   "cos=%.7f meanrel=%.3e. That band is calibrated at <= 512 lanes; either the shape moved the "
                   "streaming residual or a real regression landed — read it before re-running without the env.",
                   kBandCos, kBandMeanRel, n, ag.worst_cos, ag.worst_meanrel);
    }

    // The shipping arm: sentinel-fill the partials, run the widest wave alone, and require every promised slot
    // to have been written. Rows the form does not own are expected to keep the sentinel, and that expectation is
    // arithmetic on `fattn_covers`, not a loosened check.
    {
        fill_sentinel(part, el_part);
        fat.launch_wave(waves - 1);
        HIP(hipGetLastError());
        HIP(hipDeviceSynchronize());
        // The exact promise, per row: a covered row's slot is written once per lane block that owns a non-empty
        // slice of it, and a slice `b*span >= nrow` is empty by construction (span = ceil(nrow/s_row), so with a
        // large split and a short row the top blocks own nothing — `fattn_row_of`'s `bhi <= blo` return). Counting
        // those as unwritten failures would be this rig inventing a bug; counting them as written would be a check
        // that cannot fail, so they are subtracted from the arithmetic and printed.
        long long covered = 0, written_row_blocks = 0;
        for (int r = 0; r < lastc.nrows; ++r) {
            if (!fattn_covers(lastc, r)) continue;
            ++covered;
            const int nrow = lanes_of(lastc, r);
            const int s_row = blocks_for_row(lastc, r);          // == lastc.split for a covered row
            const int span = (nrow + s_row - 1) / s_row;
            written_row_blocks += span > 0 ? std::min(s_row, (nrow + span - 1) / span) : 0;
        }
        const long long range_row_blocks = (long long)lastc.nrows * lastc.split;
        const long long range_slots = range_row_blocks * N_Q;
        const long long expect_sentinel = (range_slots - written_row_blocks * N_Q) * (DH + 2);
        if (covered == 0)
            refuse("the widest wave has 0 rows the shipping form owns; there is nothing to pair at lanes=%d", n);
        if (written_row_blocks == 0)
            refuse("every covered row of the widest wave owns an empty lane slice; the audit would be vacuous");
        audit(part, (size_t)(range_slots * (DH + 2)), "fattn part (widest wave)",
              (unsigned long long)expect_sentinel);
        std::printf("coverage widest wave: rows=%d covered=%d (%.1f %%) split=%d slots=%lld empty_row_blocks=%lld "
                    "(unwritten by design, subtracted from the audit)\n", lastc.nrows, (int)covered,
                    100.0 * covered / lastc.nrows, lastc.split, range_slots, range_row_blocks - written_row_blocks);
    }
    // One warm pass over every wave of the shipping arm: a fault in an early wave must show up here, not inside
    // a timed rep (a path only the tail waves reach is invisible to a one-wave check — AGENTS.md, the wave-only
    // row).
    fat.launch();
    HIP(hipGetLastError());
    HIP(hipDeviceSynchronize());
    wide.launch();
    HIP(hipGetLastError());
    HIP(hipDeviceSynchronize());

    // ---- the timed reps: paired and interleaved inside this one process ---------------------------------
    std::printf("\n== timing: best-of-%d, steady_clock wall over one whole pass per arm, reps interleaved ==\n",
                reps);
    std::printf("note: A11's 4.714 TFLOP/s is a sum of pmc DISPATCH intervals; this is wall over a pass, so it "
                "also carries the %d inter-launch gaps of the shipping arm (~4 us each) and is the conservative "
                "side of the comparison.\n", (int)fw.launches);
    std::vector<double> mw(reps), mf(reps);
    for (int r = 0; r < reps; ++r) {
        mw[r] = timed_ms([&] { wide.launch(); });
        mf[r] = timed_ms([&] { fat.launch(); });
        std::printf("  rep %d: wide %10.4f ms    fattn_u8 %10.4f ms\n", r + 1, mw[r], mf[r]);
    }
    const double bw = *std::min_element(mw.begin(), mw.end());
    const double bf = *std::min_element(mf.begin(), mf.end());
    const double wf = *std::max_element(mf.begin(), mf.end());
    const double wwf = *std::max_element(mw.begin(), mw.end());
    const double flop = 2.0 * (double)ww.mac_useful;
    std::printf("\nRESULT lanes=%d form=wide     best_ms=%.4f worst_ms=%.4f launches/pass=%lld "
                "mac_useful=%.6g mac_issued=%.6g (overhang %.3fx) TFLOP/pass=%.4f TFLOP/s=%.3f "
                "TFLOP/s_on_issued=%.3f blocks_per_CU=%d wf_per_CU=%lld lds=%zu\n",
                n, bw, wwf, ww.launches, (double)ww.mac_useful, (double)ww.mac_issued,
                double(ww.mac_issued) / double(ww.mac_useful), flop / 1e12, flop / 1e12 / (bw / 1e3),
                2.0 * ww.mac_issued / 1e12 / (bw / 1e3), ow.blocks_per_cu, ow.wf_per_cu, ow.shared);
    std::printf("RESULT lanes=%d form=fattn_u8 best_ms=%.4f worst_ms=%.4f launches/pass=%lld "
                "mac_useful=%.6g mac_issued(model of the block-union round rule)=%.6g (overhang %.3fx, mean "
                "rounds/active block %.2f — A11 §1 traced 33.29 on the widest body) TFLOP/pass=%.4f TFLOP/s=%.3f "
                "covered_rows=%lld/%lld (%.1f %%) blocks_per_CU=%d wf_per_CU=%lld lds=%zu\n",
                n, bf, wf, fw.launches, (double)fw.mac_useful, (double)fw.mac_issued,
                double(fw.mac_issued) / double(fw.mac_useful),
                fw.active_blocks ? double(fw.rounds_total) / double(fw.active_blocks) : 0.0, flop / 1e12,
                flop / 1e12 / (bf / 1e3), fw.covered_rows, fw.rows, 100.0 * fw.covered_rows / fw.rows,
                of.blocks_per_cu, of.wf_per_cu, of.shared);
    std::printf("PAIR lanes=%d wide/shipping = %.4fx on wall time (%.4f vs %.4f ms); the shipping arm is "
                "inert on %lld of %lld rows, so read the two RESULT lines' own denominators before calling this "
                "a ratio of like for like\n", n, bf / bw, bw, bf, fw.rows - fw.covered_rows, fw.rows);
    std::printf("PAIR both lines' mac_useful covers EVERY row of the pass (A11's basis: its 4.714 divides the "
                "whole pass's useful work by `attn_fattn_wave_u8` dispatches alone), which is generous to the "
                "baseline by exactly the uncovered share above — in an engine run those rows are `attn_slim_wave`'s "
                "and are not free. The wide arm owns every row, so its line carries no such share.\n");
    std::printf("BET  abort line %.1f TFLOP/s (A12-WIDE-BET.md §2, stated at 32,768 lanes): wide reads %.3f -> "
                "%s%s. Report-only: the >= 3 %% deciding-row rule and the DISPLACED-vs-shipped call are the main "
                "agent's.\n", abort_tflops, flop / 1e12 / (bw / 1e3),
                flop / 1e12 / (bw / 1e3) >= abort_tflops ? "CLEARED" : "NOT CLEARED",
                n == 32768 ? "" : " (this row is not 32,768 lanes, so treat it as a sanity row, not the verdict)");
    print_limits();
    std::printf("REMINDER: run this only from run_a12_window.sh — main agent, blocking, one gpu_lock hold, zero "
                "subagents, and the ms figures above are the main agent's to book, never a worker's.\n");
    HIP(hipFree(q)); HIP(hipFree(k)); HIP(hipFree(v)); HIP(hipFree(out)); HIP(hipFree(part));
    HIP(hipFree(pbuf)); HIP(hipFree(lbuf)); HIP(hipFree(ref)); HIP(hipFree(drows));
    return 0;
}
