/*
 * s1_launch_price.hip.cpp — ROUTE S1 GATE ZERO: what does a launch cost on the wall when nothing
 * drains between them?
 *
 * THE QUESTION (progress/2026-10-21/route-S1-segments-2026-10-21.md §3). The engine emits 340
 * launches and 305 stage-clock syncs per decode step (quoted, labelled below), and the serialized
 * stage shape costs 1.80 ms more than the burst shape for identical work, ~5.9 us per sync — but the
 * stage clock and `tools/context_sweep.py` drain per segment BY CONSTRUCTION, so part of that gap may
 * be an instrument cost of MEASURING rather than a production cost. This rig launches N kernels
 * back-to-back in the production shape (one sync for the whole burst), sweeps N, and fits a line, so
 * the marginal wall cost of a launch is READ OFF THE CURVE (slope, with residual and intercept) rather
 * than assumed from two points. The slope decides S1: >= 3 us funds a candidate list, 1-3 us folds it
 * into D5, below 1 us S1 IS DEAD and every T_geo/T_serial figure in D2 and D4 is an instrument artefact.
 *
 * WHAT THE FIT GIVES, and why three quantities come out of one ladder:
 *     ms_per_step  =  intercept  +  slope * N
 *   slope      = the marginal wall cost of ONE MORE LAUNCH at that body size and sync shape.
 *   intercept  = the per-step cost that does not scale with N: the one drain plus the host loop.
 *   the sync axis (fixed N, varying S) has its own slope = the marginal cost of one more DRAIN, which
 *                is the number D2 billed at 0.2709 ms for 48 segments and D4 at 5.32-5.80 us/sync.
 * Two body sizes are mandatory, because they answer different questions:
 *   `empty`/`tiny` (launch-dominated) — is the DISPATCH itself cheap when the queue runs ahead?
 *   `mid`/`fat` (byte-dominated)      — is dispatch HIDDEN behind real work? If the slope here is ~0
 *                while the tiny-body slope is not, launches are paid only when nothing overlaps them,
 *                which is the engine's small-launch population, not its mlp/proj/attn population.
 * `--pattern engine --body-mix engine` puts the same fit at the operating point: 340 launches in 305
 * segments with each launch carrying its own family's real per-launch bytes from the step census.
 *
 * WHAT IS QUOTED, never measured here. Every millisecond and microsecond figure in this file and in
 * S1-GATEZERO.md is a QUOTE from the ledger or the S1 brief, labelled as such, and the rig prints no
 * `ms` at all without `--time` — that flag belongs to the main agent (AGENTS.md: timing is a
 * main-agent measurement; a worker's device gate prints no number).
 *   [Q1] 340 launches / 305 stage-clock syncs per step, plan_per_step=340 — the S1 brief §2, echoing
 *        `decode_run`'s own `LAUNCH glue=49 proj=48 gdn=72 attn=6 mlp=48 cast=97 other=20 total=340`
 *        and `STAGE ... launches_per_step=406 plan_per_step=340 syncs_per_step=305`.
 *   [Q2] serialized stage TOTAL minus burst = 1.80 ms for identical work, ~305 syncs => 5.9 us/sync
 *        — S1 brief §2 (quoted, NOT re-taken here).
 *   [Q3] per-segment constant the ledger bills: 7.2 us/segment cross-family, 5.64 us/segment measured
 *        on the gdn family itself — `prototype/gdn_insitu/D4-DIAGNOSIS.md` §0(i)/§6 (quoted).
 *   [Q4] empty/tiny kernel dispatch floor 1.85 us/launch, graph replay 1.78 us (4 % better) —
 *        `devices/strix-halo-memory-bandwidth.md`, `engine/performance-status.md` §"kernel launch
 *        floor" (quoted). `tools/launch_overhead_hip.cpp` is its instrument: it prices ONE EMPTY
 *        KERNEL in a burst and has no byte body, no segment/sync axis, no engine population and no
 *        verification of any kind. What this rig adds is the marginal slope at a real launch
 *        population with a real byte body and a separate sync axis; check_s1.sh's G-dispatch gate
 *        re-reads that quoted constant as the reproduction gate that ties the two instruments together.
 *
 * LAUNCH COUNT AND SYNC COUNT ARE DIFFERENT AXES and the route's two sources conflate them, so the
 * rig has both: `--launches N` and `--syncs S`, with S drains distributed over the burst. That makes
 * all four discriminating arms expressible: same N with {1,N} syncs; same S with different N.
 *
 * VERIFICATION, in the `prototype/gdn_insitu` / `prototype/mlp_insitu` discipline, because a launch
 * rig that silently launches fewer kernels than it claims would print a beautiful, wrong slope.
 *   - EVERY LAUNCH AND EVERY BLOCK REPORTS BYTES TOUCHED. One record per (launch, block): {read sum,
 *     read count, write sum, write count, tag}. The expected values are DERIVED ON THE HOST from the
 *     same launch struct the launcher is built from (`plan_build` + `plan_invariants`), never from
 *     what the kernel reported — a check whose reference side comes from the thing under test cannot
 *     adjudicate it (prototype/attn_tile_model.py). The read side is witnessed by a sum over a
 *     fill pattern chosen so its range sum has a CLOSED FORM and integer addition is associative, so
 *     there is no floating-point association question (AGENTS' q̃ row) and no readback of the read
 *     region is needed.
 *   - THE WRITES THEMSELVES ARE CHECKED BYTE FOR BYTE, because a record is self-consistent about a
 *     wrong ADDRESS: `neg_lying_record` is a control that reports perfect counts while storing
 *     nothing, and only the content comparison can see it.
 *   - MUST-LAND SETS ARE DERIVED from the plan actually emitted, and an empty one FAILS
 *     (`FAIL_nothing_expected`) rather than passing vacuously. Must-catch and must-not-catch sets for
 *     every negative are DERIVED from the injection's geometry too, and `--selftest-geom` refuses a
 *     control whose derived sets are empty in either direction.
 *   - GUARD BANDS of poison around every region; everything outside the must-land set must be
 *     byte-identical to the snapshot, so a stray write is caught, not merely a missing one.
 *   - EVERY DEVICE CASE RUNS IN ITS OWN FORKED CHILD (one faulting kernel poisons the context and
 *     every later case in that process silently never ran — tools/mixed_bw), and no HIP call sits on
 *     a path reachable before the fork (prototype/mlp_insitu §2.5).
 *   - A WINDOW THAT LAUNCHES NOTHING PRINTS NO NUMBER. `--launches 0` prints `NO-TIMING no_launch=1`;
 *     this part's timer returns ~0 below a millisecond, and a reader tabulates 0.0001 as "free"
 *     (prototype/mlp_insitu §2.6).
 *   - NO RATE IS PRINTED, at all. A bandwidth figure here would be a DEMANDED rate above the DRAM
 *     plateau and would look like a new record (AGENTS: >240 GB/s is skipped work). The rig prints
 *     bytes and milliseconds and leaves the division to the reader.
 *
 * DECLARED BLINDNESSES, stated rather than discovered in review:
 *   - A DUPLICATED launch is invisible to the device witness (it rewrites the same records with the
 *     same values). It is caught by the printed host submission counter instead, which increments
 *     only after a launch that returned no error, and which `check_s1.sh`/`s1_sweep.sh` assert
 *     against the requested count. The two mechanisms are independent and agree or the row is refused.
 *   - A READ-SET PERMUTATION inside one block's own chunk is invisible BY CONSTRUCTION (the witness is
 *     a sum). `blind_permute_reads` requires that invisibility, so the blindness is asserted, not
 *     hoped for. Writes are index-valued, so a permuted WRITE is caught.
 *   - A STRAY WRITE into the read region is caught only if it lands before the element is read; the
 *     guard bands cover the region ends.
 *   - The byte arm is read-dominated on purpose (the step is 62.8:1 read:write, `step_traffic_model`),
 *     but it is not the engine's kernel: it moves bytes and this rig therefore says NOTHING about what
 *     any one kernel costs. That is D2's and D4's territory, not this route's.
 *   - ORDER within the step is not reproduced: the engine interleaves families, this rig groups them.
 *     Launch order is a refuted factor on this box (D2 §1: Q9's B_blk/B_rr rows, ratio 1.005 on an
 *     identical launch multiset), so grouping is a license, and it is stated instead of smuggled.
 *   - SEGMENTS ARE COUNTS, not family names. The rig reproduces the engine's per-segment launch
 *     POPULATION (340 in 305: 36 segments of 2 launches, 268 of 1, one with none) and not which
 *     family owns a segment, because the quantity under test is a drain count.
 *
 * MECHANICS. `hipMalloc` only, never managed memory (AGENTS: managed degrades silently near 32 GiB and
 * this rig legitimately allocates ~1.5 GiB). Fresh pages are warmed by the fill and by an untimed step
 * before any timing; `--pages cold` is REFUSED, not averaged (first touch runs ~7x slow). The timer is
 * steady_clock around one whole step — `hipEventElapsedTime` returns exactly 0 below ~1 ms on this
 * part, and a step at N=1 is far below that.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#define CK(x)                                                                          \
    do {                                                                               \
        hipError_t e_ = (x);                                                           \
        if (e_ != hipSuccess) {                                                        \
            std::printf("hip error %s at line %d\n", hipGetErrorString(e_), __LINE__); \
            std::exit(4);                                                              \
        }                                                                              \
    } while (0)

namespace {

// ---------------------------------------------------------------- the step's real population [Q1]
// engine/state.h:41-43. The per-family launch bill is the same arithmetic `planned_launch_counts()`
// performs (engine/decode_step.cpp), spelled here so this rig can be checked against it without
// linking the engine; `census.py` asserts all of it against `tools/step_traffic_model.py` and against
// `--print-engine-spec`, and the totals are asserted to be the quoted 340 and 305.
constexpr int kNumLayers = 24, kNumGdnLayers = 18, kNumAttnLayers = 6;
constexpr int kEngineLaunches = 340;
constexpr int kEngineSegments = 305;

enum Fam { F_GLUE = 0, F_PROJ, F_GDN, F_ATTN, F_MLP, F_CAST, F_COPY, F_LMHEAD, F_TAIL, NFAM };
const char* kFamName[NFAM] = {"glue", "proj",       "gdn",      "attn", "mlp",
                              "cast", "copy_19", "lm_head", "host_tail"};
struct FamRow {
    int launches;
    int segments;
    double read_mb;     // whole family, per step, in the census's MB (1e6 B), NOT MiB
    double write_mb;
};
// Totals, not per-launch: per-launch bytes are read_mb/launches, which is the rule census.py uses.
// The unit is the CENSUS's unit (MB = 1e6 bytes), and the byte model inside the rig is 8-byte
// elements, so the conversion MB*1e6/8 happens exactly once, where the family rows are expanded.
// An earlier version labelled these numbers MiB and printed them as such, which made the engine arm
// 4.6 % heavier than the census it claims to reproduce (lm_head reads 509.552 MB = 485.947 MiB).
// `norms` (0.106 MB read) is folded into glue — those launches are glue's add+norm and final norm.
const FamRow kFam[NFAM] = {
    {1 + 2 * kNumLayers, 1 + 2 * kNumLayers, 0.301 + 0.106, 0.201},
    {2 * kNumLayers, 2 * kNumLayers, 378.667 + 88.080 + 0.811, 0.407},
    {4 * kNumGdnLayers, 2 * kNumGdnLayers, 22.642, 20.425},
    {kNumAttnLayers, kNumAttnLayers, 1.573 + 0.049, 0.080 + 0.012},
    {2 * kNumLayers, 2 * kNumLayers, 528.482 + 0.344, 0.221},
    {4 * kNumGdnLayers + 4 * kNumAttnLayers + 1, 4 * kNumGdnLayers + 4 * kNumAttnLayers + 1, 0.371,
     0.741},
    {kNumGdnLayers + 1, kNumGdnLayers + 1, 1.327, 1.327},
    {1, 1, 508.559 + 0.993, 0.993},
    {0, 1, 0.0, 0.0},
};

// ---------------------------------------------------------------- byte model
// Element = 8 bytes. word(p) = K*(2p+1) over the read region, so:
//   * the value depends on the GLOBAL index: a wrong base, a short range or a dropped block changes
//     the sum, and integer addition is associative so the host can predict the sum exactly with
//     (a) no floating-point association risk and (b) a closed form: sum_{p0<=p<q} word(p) = K*(q^2-p^2)
//   * K is odd, so s = K^-1 mod 2^64 inverts it exactly: a sum therefore knows its element COUNT and
//     first index exactly, which is what makes a short or misbased read observable.
constexpr unsigned long long kWordK = 0x9E3779B97F4A7C15ull;
constexpr unsigned long long kWordInv = 0xF1DE83E19937733Dull;   // K * kWordInv == 1 (mod 2^64)
constexpr unsigned long long kSentW = 0xFFFFFFFFFFFFFFFFull;
constexpr int kGUARD = 64;                       // poison words on each side of every region
__host__ __device__ inline unsigned long long word_at(long long p) {
    return kWordK * (2ull * (unsigned long long)p + 1ull);
}
// Closed form for sum_{p0<=p<hi} word(p).
inline unsigned long long range_sum(long long p0, long long hi) {
    const unsigned long long h = (unsigned long long)hi, a = (unsigned long long)p0;
    return kWordK * (h * h - a * a);
}

struct Rec {
    unsigned long long rsum, wsum;
    unsigned rcount, wcount, tag, pad;
};
static_assert(sizeof(Rec) == 32, "record size is part of the region arithmetic");
constexpr unsigned kRecSent = 0xFFFFFFFFu;      // a real rcount never reaches this (see invariants)

// ---------------------------------------------------------------- kernel
// One record per (launch, block). The block owns a CONTIGUOUS chunk of its launch's read range and of
// its write range, aligned to the block size, so (i) accesses stay coalesced, (ii) each block's
// contribution has a closed form the host can predict, and (iii) a dropped block is a missing record
// rather than a coincidentally equal sum.
enum NegKind {
    NG_NONE = 0,
    NG_LIE_WRITE,        // accumulate and report, do not store: only a content check can see it
    NG_PERMUTE_READS,    // rotate the block's own read set: provably invisible to a sum (blind case)
    NG_SHORT_READ,       // drop the last element of one named block's read chunk
    NG_SHORT_WRITE,      // drop the last element of one named block's write chunk
    NG_TAG_CONST,        // use a constant tag in BOTH the values and the record: self-consistent
};

__device__ inline unsigned long long warp_red(unsigned long long v) {
    // AGENTS: on this target the shuffle mask is 64 bits even for wave32 builds.
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffffffffffull, v, (unsigned)off);
    return v;
}

extern "C" __global__ void s1_touch(const unsigned long long* __restrict__ src,
                                    unsigned long long* __restrict__ dst,
                                    long r_lo, long r_n, long w_lo, long w_n, unsigned tag,
                                    Rec* __restrict__ recs, long rec_lo, int neg, long inj_rec,
                                    long rchunk, long wchunk)
{
    const int tid = (int)threadIdx.x, nth = (int)blockDim.x;
    const long r_end = r_lo + r_n, w_end = w_lo + w_n;
    // The chunk sizes are BOOKED numbers passed in, not derived from gridDim.x: that is what lets
    // `neg_grid_short` be a surgical control. A grid that launches one block short then leaves exactly
    // that block's chunk untouched instead of silently enlarging every other block's work, which is
    // both the truth about a dropped block and the only version with a derivable must-catch set.
    const long rb = rec_lo + blockIdx.x;
    const bool inj = (rb == inj_rec);

    unsigned long long rsum = 0.0, wsum = 0.0, rcnt = 0, wcnt = 0;

    // Every kernel-level injection is localized to ONE block (`inj`), so a control's must-catch set is
    // one block's bytes and its must-not-catch set is everything else — the only form of the rule that
    // does not rot into "moves everything" (AGENTS, the derived must-catch rows).
    const bool lie = (neg == NG_LIE_WRITE && inj);
    const bool rot = (neg == NG_PERMUTE_READS && inj);

    {
        long lo = r_lo + (long)blockIdx.x * rchunk;
        long hi = lo + rchunk;
        if (hi > r_end) hi = r_end;
        if (lo > r_end) lo = r_end;
        if (neg == NG_SHORT_READ && inj && hi > lo) hi -= 1;   // this block's own last element
        if (hi > lo) {
            const long m = hi - lo;
            for (long i = lo + tid; i < hi; i += nth) {
                const unsigned long long v = rot ? src[lo + ((i - lo + 1) % m)] : src[i];
                rsum += v;
                rcnt += 1;
            }
        }
    }

    const unsigned etag = (neg == NG_TAG_CONST) ? 7u : tag;
    {
        long lo = w_lo + (long)blockIdx.x * wchunk;
        long hi = lo + wchunk;
        if (hi > w_end) hi = w_end;
        if (lo > w_end) lo = w_end;
        if (neg == NG_SHORT_WRITE && inj && hi > lo) hi -= 1;  // this block's own last element
        if (hi > lo) {
            // The written value is indexed by the element's offset inside ITS OWN booked write range,
            // not by the address it lands at. That is deliberate: a launch pointed at a neighbour's
            // range then stores the same values it would have stored at home, so its own record stays
            // byte-identical and ONLY the byte-content check can see the alias. An address-indexed
            // value made the aliased launch's sum move, which is the witness doing the launcher's job.
            for (long i = lo + tid; i < hi; i += nth) {
                const unsigned long long v = word_at(i - w_lo) + etag;
                wsum += v;
                wcnt += 1;
                if (!lie) dst[i] = v;
            }
        }
    }

    // Block reduction (warp butterfly, then a shared tree) — a per-thread store to a per-block slot is
    // NOT a reduction (tools/mixed_bw's measured failure). One record per block, from thread 0.
    __shared__ unsigned long long sh[4][32];
    const int lane = tid % warpSize, wid = tid / warpSize;
    rsum = warp_red(rsum); wsum = warp_red(wsum); rcnt = warp_red(rcnt); wcnt = warp_red(wcnt);
    if (lane == 0) {
        sh[0][wid] = rsum; sh[1][wid] = wsum; sh[2][wid] = rcnt; sh[3][wid] = wcnt;
    }
    __syncthreads();
    if (tid == 0) {
        const int nw = (nth + warpSize - 1) / warpSize;
        unsigned long long a = 0, b = 0, c = 0, d = 0;
        for (int w = 0; w < nw; ++w) { a += sh[0][w]; b += sh[1][w]; c += sh[2][w]; d += sh[3][w]; }
        Rec r;
        r.rsum = a; r.wsum = b;
        r.rcount = (unsigned)c; r.wcount = (unsigned)d;
        r.tag = etag; r.pad = 0u;
        recs[rb] = r;
    }
}

extern "C" __global__ void fill_word(unsigned long long* src, long n)
{
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) src[i] = word_at(i);
}

// ---------------------------------------------------------------- regions
enum Reg { R_SRC = 0, R_DST, R_REC, NREG };
const char* kRegName[NREG] = {"src", "dst", "rec"};
struct Region {
    void* base = nullptr;
    char* p = nullptr;
    size_t bytes = 0;
    std::vector<unsigned long long> snap;   // dst only; src is a formula
};
Region g_reg[NREG];

void reg_free(int id) {
    if (g_reg[id].base) (void)hipFree(g_reg[id].base);
    g_reg[id] = Region();
}
void reg_alloc(int id, size_t bytes) {
    reg_free(id);
    if (bytes == 0) return;
    Region& r = g_reg[id];
    r.bytes = bytes;
    CK(hipMalloc(&r.base, bytes + 2 * kGUARD * 8));
    r.p = static_cast<char*>(r.base) + kGUARD * 8;
    for (int side = 0; side < 2; ++side) {
        void* at = side == 0 ? r.base : static_cast<char*>(r.base) + kGUARD * 8 + bytes;
        CK(hipMemset(at, 0xBE, kGUARD * 8));
    }
}
int check_guards() {
    int bad = 0;
    std::vector<unsigned char> g(kGUARD * 8);
    for (int id = 0; id < NREG; ++id) {
        if (!g_reg[id].base) continue;
        for (int side = 0; side < 2; ++side) {
            const void* at = side == 0 ? g_reg[id].base
                                       : static_cast<char*>(g_reg[id].base) + kGUARD * 8 +
                                             g_reg[id].bytes;
            CK(hipMemcpy(g.data(), at, kGUARD * 8, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < g.size(); ++i) bad += (g[i] != 0xBE);
        }
    }
    return bad;
}

// ---------------------------------------------------------------- config
enum Pattern { PT_UNIFORM = 0, PT_ENGINE };
enum Body { BD_TINY = 0, BD_EMPTY, BD_MID, BD_FAT, BD_ENGINE, BD_CUSTOM };
enum { NEG_NONE = 0, NEG_DROP_LAST, NEG_DROP_MID, NEG_GRID_SHORT, NEG_SHORT_WRITE, NEG_SHORT_READ,
       NEG_ALIAS, NEG_TAG_CONST, NEG_LIE_WRITE, NEG_PERMUTE_READS, NEG_COUNT };
const char* kNegName[NEG_COUNT] = {"none",       "drop_last",  "drop_mid",   "grid_short", "short_write",
                                   "short_read", "alias_write", "tag_const",  "lie_write", "permute_reads"};
int neg_kernel_id(int neg) {
    switch (neg) {
        case NEG_LIE_WRITE: return NG_LIE_WRITE;
        case NEG_PERMUTE_READS: return NG_PERMUTE_READS;
        case NEG_SHORT_READ: return NG_SHORT_READ;
        case NEG_SHORT_WRITE: return NG_SHORT_WRITE;
        case NEG_TAG_CONST: return NG_TAG_CONST;
        default: return NG_NONE;
    }
}

// Body presets. The byte-dominated ones are sized from the step: `mid` is a small proj-sized launch and
// `fat` is a small mlp-sized one, so the arm name says which family's launch it resembles. All four
// carry a non-zero write as well, so every arm is byte-verified. `tiny` is 16 KiB of read, which at the
// quoted 237 GB/s plateau is ~0.07 us of transfer against a quoted 1.85 us dispatch [Q4] — the
// launch-dominated point, by 26x. `empty` is that shape with no bytes at all, so the fit can be tied
// directly to [Q4].
struct BodyPreset { double read_kib, write_kib; };
BodyPreset body_preset(int body) {
    switch (body) {
        case BD_EMPTY: return {0.0, 0.0};
        case BD_TINY: return {16.0, 0.5};
        case BD_MID: return {1024.0, 16.0};
        case BD_FAT: return {11264.0, 176.0};      // the mlp launch's ~11.02 MiB read, 64:1
        default: return {16.0, 0.5};
    }
}
const char* body_name(int b) {
    switch (b) {
        case BD_EMPTY: return "empty"; case BD_TINY: return "tiny"; case BD_MID: return "mid";
        case BD_FAT: return "fat"; case BD_ENGINE: return "engine"; default: return "custom";
    }
}

struct Cfg {
    int launches = 340;
    int syncs = 1;
    int pattern = PT_UNIFORM;
    int body = BD_TINY;
    double read_kib = 0, write_kib = 0;   // BD_CUSTOM per launch; 0 means "take the preset"
    double total_mib = 0.0;               // merge arm: fixed step bytes, split over N launches
    int threads = 256;
    int grid_cap = 2048;
    int passes = 8;                       // passes per block before the grid is widened
    int time_flag = 0, reps = 3, iters = 5;
    int neg = NEG_NONE;
    int cold = 0;                 // only --pages warm is accepted; the field keys the arm name
    std::string arm;              // arm_key() of this config; printed in ROW and RESULT lines
};

// The arm key is part of the output contract: fit_slope.py groups its ladder on this exact string, so
// the rig composes it once, after every knob is known. Seven fields:
//   pattern | body | size | launches | syncs | passes | pages
// `size` distinguishes two arms that share a body name but not a footprint: a preset body is `p`, a
// custom one is c<readKiB>_<writeKiB>, and a fixed-total (merge) arm is t<totalMiB>. Without it
// `--launches 340 --body tiny` and `--launches 340 --total-mib 64` would collide on one key and the
// fitter would average two different geometries into one point.
static std::string arm_key(const Cfg& c) {
    std::string size = "p";
    if (c.total_mib > 0) {
        char buf[48];
        std::snprintf(buf, sizeof buf, "t%g", c.total_mib);
        size = buf;
    } else if (c.body == BD_CUSTOM) {
        char buf[48];
        std::snprintf(buf, sizeof buf, "c%g_%g", c.read_kib, c.write_kib);
        size = buf;
    }
    return std::string(c.pattern == PT_ENGINE ? "engine" : "uniform") + "|" + body_name(c.body) +
           "|" + size + "|" + std::to_string(c.launches) + "|" + std::to_string(c.syncs) + "|" +
           std::to_string(c.passes) + "|warm";
}

const char* refuse(const Cfg& c) {
    if (c.launches < 0) return "--launches must be >= 0; --launches 0 is the declared no-launch arm.";
    if (c.syncs < 0) return "--syncs must be >= 0.";
    if (c.launches > 0 && c.syncs == 0)
        return "--syncs 0 times submission alone, not a step: the wall would exclude completion and "
               "the queue depth would be unbounded. Submission cost alone is "
               "tools/launch_overhead_hip.cpp's measurement [Q4]; this rig measures a step, and a "
               "step ends in a drain.";
    if (c.launches == 0 && c.syncs > 0)
        return "--launches 0 with --syncs > 0 drains a step that has nothing in it; the no-launch arm "
               "is --launches 0 --syncs 0, and it prints NO-TIMING, never a number.";
    if (c.launches > 0 && c.syncs > c.launches)
        return "--syncs > --launches is not a step the engine emits. The engine's one segment with no "
               "launch is `host_tail` (0 launches, 1 segment, [Q1]) and only --pattern engine carries "
               "it; a uniform arm with more drains than kernels prices a drain that owns nothing.";
    if (c.pattern == PT_ENGINE) {
        if (c.launches != kEngineLaunches)
            return "--pattern engine IS the shipped launch population (340, quoted [Q1]); another "
                   "--launches with it would print the engine's segment count for a different launch "
                   "count, which is the mistake this route exists to stop.";
        if (c.syncs != kEngineSegments && c.syncs != kEngineLaunches && c.syncs != 1)
            return "--pattern engine takes --syncs 305 (as the engine prints [Q1]), 340 (drain every "
                   "launch, the gran=kernel shape) or 1 (all 340 in one drain, the production burst "
                   "shape). Any other S would be a segment count the engine does not have.";
        if (c.body != BD_ENGINE)
            return "--pattern engine pairs with --body-mix engine: the shipped population priced with "
                   "a synthetic uniform body is a population arm, not the engine arm, and it belongs "
                   "in the ladder with --pattern uniform.";
    } else if (c.body == BD_ENGINE) {
        return "--body-mix engine needs --pattern engine: the per-family byte mix is defined on the "
               "engine's launch order.";
    }
    if (c.total_mib > 0 && (c.body == BD_ENGINE))
        return "--total-mib and --body-mix engine both define the bytes; pick one.";
    if (c.total_mib > 0 && c.launches == 0)
        return "--total-mib with --launches 0 emits bytes without a launcher, which prices nothing.";
    if (c.threads < 64 || c.threads > 1024 || (c.threads % 64) != 0)
        return "--threads must be a multiple of 64 in [64,1024] (wave64 part, 1024 = 16 wavefronts).";
    if (c.passes < 1) return "--passes must be >= 1.";
    return nullptr;
}

// ---------------------------------------------------------------- the plan
// One entry per launch, exactly as the launcher will build it. Ranges are disjoint and packed: each
// launch owns [r_lo, r_lo+r_n) of the read region and [w_lo, w_lo+w_n) of the write region.
struct LS {
    int family;                 // engine family, or -1 for a uniform arm
    int seg;                    // which drain this launch precedes
    long r_lo, r_n, w_lo, w_n;
    int grid;
    unsigned tag;
    long rec_lo;
};
struct Plan {
    std::vector<LS> ls;
    long n_src = 0, n_dst = 0, n_rec = 0;
    int segments = 0;
    std::vector<int> seg_lps;   // launches per segment, derived
    long read_bytes = 0, write_bytes = 0;
};

int pick_grid(const Cfg& c, long n_elem) {
    if (n_elem <= 0) return 1;
    const long per_block = (long)c.threads * c.passes;
    long g = (n_elem + per_block - 1) / per_block;
    if (g < 1) g = 1;
    if (g > (long)c.grid_cap) g = c.grid_cap;
    return (int)g;
}

// Per-launch read/write element counts for this config, BEFORE base assignment.
struct RW { long r, w; };
std::vector<RW> body_sizes(const Cfg& c) {
    std::vector<RW> v;
    if (c.pattern == PT_ENGINE) {
        for (int f = 0; f < NFAM; ++f) {
            const int nl = kFam[f].launches;
            // Round each launch, not the family total: the last launch of a family absorbs the
            // remainder so the family total is exact (asserted in plan_invariants).
            long r_tot = (long)llround(kFam[f].read_mb * 1e6 / 8.0);      // census MB -> elements
            long w_tot = (long)llround(kFam[f].write_mb * 1e6 / 8.0);
            for (int i = 0; i < nl; ++i) {
                long r = r_tot / nl, w = w_tot / nl;
                if (i == nl - 1) { r = r_tot - r * (nl - 1); w = w_tot - w * (nl - 1); }
                v.push_back({r, w});
            }
        }
        return v;
    }
    BodyPreset bp = body_preset(c.body);
    double rk = (c.body == BD_CUSTOM && c.read_kib > 0) ? c.read_kib : bp.read_kib;
    double wk = (c.body == BD_CUSTOM && c.write_kib > 0) ? c.write_kib : bp.write_kib;
    if (c.total_mib > 0) {   // merge arm: total bytes fixed, launch count is the variable
        const long tot = (long)llround(c.total_mib * 131072.0);
        const long r_tot = (long)llround(c.total_mib * 131072.0 * 63.0 / 64.0);
        (void)tot;
        for (int i = 0; i < c.launches; ++i) {
            long r = r_tot / c.launches, w = (tot - r_tot) / c.launches;
            if (i == c.launches - 1) { r = r_tot - r * (c.launches - 1); w = (tot - r_tot) - w * (c.launches - 1); }
            v.push_back({r, w});
        }
        return v;
    }
    const long r = (long)llround(rk * 128.0);      // KiB -> 8 B elements
    const long w = (long)llround(wk * 128.0);
    for (int i = 0; i < c.launches; ++i) v.push_back({r, w});
    return v;
}

void segments_uniform(const Cfg& c, Plan& p) {
    const int S = c.syncs, N = c.launches;
    p.segments = S;
    p.seg_lps.assign(S, 0);
    if (S == 0 || N == 0) return;
    const int base = N / S, rem = N % S;
    for (int s = 0; s < S; ++s) p.seg_lps[s] = base + (s < rem ? 1 : 0);
}

void segments_engine(const Cfg& c, Plan& p) {
    p.segments = c.syncs;
    if (c.syncs == kEngineSegments) {
        for (int f = 0; f < NFAM; ++f)
            for (int s = 0; s < kFam[f].segments; ++s) p.seg_lps.push_back(kFam[f].launches / kFam[f].segments);
    } else if (c.syncs == kEngineLaunches) {
        p.seg_lps.assign(kEngineLaunches, 1);      // drain every launch: the gran=kernel shape
    } else {
        p.seg_lps.assign(1, kEngineLaunches);      // one drain for the whole step: production burst
    }
}

// The plan is the address map: every launch's read range, write range, record range, grid and tag.
// NO injection perturbs it — the negatives perturb SUBMISSION (a launch never submitted, a grid one
// block short, a base pointed elsewhere) or the kernel, never the booked map, so the expectation always
// describes the step the arm claims to have run (prototype/gdn_insitu's rule that a reference derived
// from the thing under test cannot adjudicate it).
void plan_build(const Cfg& c, Plan& p) {
    const std::vector<RW> sz = body_sizes(c);
    if (c.pattern == PT_ENGINE) segments_engine(c, p); else segments_uniform(c, p);

    std::vector<int> fam_of;      // per launch, index into kFam (engine order: family blocks, in the
    if (c.pattern == PT_ENGINE) { // order the stage report prints them in)
        for (int f = 0; f < NFAM; ++f) for (int i = 0; i < kFam[f].launches; ++i) fam_of.push_back(f);
    }
    std::vector<int> seg_of;      // per launch, the drain it precedes
    for (int i = 0; i < (int)p.seg_lps.size(); ++i)
        for (int j = 0; j < p.seg_lps[i]; ++j) seg_of.push_back(i);

    long r_cur = 0, w_cur = 0, rec_cur = 0;
    for (size_t i = 0; i < sz.size(); ++i) {
        LS l;
        l.family = (c.pattern == PT_ENGINE) ? fam_of[i] : -1;
        l.seg = (i < seg_of.size()) ? seg_of[i] : (p.segments > 0 ? p.segments - 1 : 0);
        l.r_lo = r_cur; l.r_n = sz[i].r; r_cur += sz[i].r;
        l.w_lo = w_cur; l.w_n = sz[i].w; w_cur += sz[i].w;
        l.grid = pick_grid(c, std::max(l.r_n, l.w_n));
        l.tag = (unsigned)(i + 1);   // tag 7 is launch index 6: tag_const's derived must-not-move slot
        l.rec_lo = rec_cur;
        rec_cur += l.grid;
        p.ls.push_back(l);
    }
    p.n_src = r_cur; p.n_dst = w_cur; p.n_rec = rec_cur;
    for (const LS& l : p.ls) { p.read_bytes += l.r_n * 8; p.write_bytes += l.w_n * 8; }
}


// Re-derive on the host the index arithmetic the kernel will run, from the same struct the launch is
// built from, and refuse a plan that does not hold (tools/mixed_bw: "the solver and the kernel meant
// two different things by `rounds`"). Returns nullptr when the plan is sound, else the reason.
const char* plan_invariants(const Cfg& c, const Plan& p) {
    if ((int)p.ls.size() != c.launches) return "shipped plan size does not equal --launches";
    if (p.segments != c.syncs) return "segment count does not equal --syncs";
    long seg_sum = 0;
    for (int s : p.seg_lps) seg_sum += s;
    if (seg_sum != (long)p.ls.size()) return "segment populations do not sum to the launch count";
    long r_cur = 0, w_cur = 0, rec_cur = 0;
    std::vector<char> dst_own((size_t)std::max<long>(p.n_dst, 1), 0);
    for (const LS& l : p.ls) {
        if (l.r_lo != r_cur || l.w_lo != w_cur || l.rec_lo != rec_cur)
            return "ranges are not packed as the builder assigned them";
        if (l.r_n < 0 || l.w_n < 0) return "negative range";
        if (l.grid < 1) return "grid below 1";
        // A block's chunk is a whole number of block sizes, so no chunk can start misaligned; the
        // record field is an int, so a chunk that could overflow it is refused, never truncated.
        const long chunk = (std::max(l.r_n, l.w_n) + (long)l.grid * c.threads - 1) /
                                   ((long)l.grid * c.threads) * c.threads;
        if (chunk > (long)kRecSent - 2) return "a block's chunk exceeds the record count sentinel";
        for (long e = l.w_lo; e < l.w_lo + l.w_n; ++e) {
            if (dst_own[(size_t)e]) return "two launches own one write element";   // §20.1's alias, booked
            dst_own[(size_t)e] = 1;
        }
        r_cur += l.r_n; w_cur += l.w_n; rec_cur += l.grid;
    }
    if (r_cur != p.n_src || rec_cur != p.n_rec) return "totals disagree with the walk";
    if (c.pattern == PT_ENGINE) {
        // The engine's own populations, re-derived: 340 launches in 305 segments; per family,
        // launches % segments == 0 (gdn 72 in 36 = two per drain, the rest one, host_tail none).
        long fl = 0, fs = 0;
        for (int f = 0; f < NFAM; ++f) {
            if (kFam[f].segments > 0 && kFam[f].launches % kFam[f].segments)
                return "an engine family's launches are not a whole number per segment";
            fl += kFam[f].launches; fs += kFam[f].segments;
        }
        if (fl != kEngineLaunches || fs != kEngineSegments) return "engine family totals are not 340/305";
        int fam_seen[NFAM] = {0};
        for (const LS& l : p.ls) if (l.family >= 0) fam_seen[l.family]++;
        for (int f = 0; f < NFAM; ++f)
            if (fam_seen[f] != kFam[f].launches) return "engine plan lost or gained a family launch";
        double rd = 0, wr = 0, tr = 0, tw = 0;
        for (const LS& l : p.ls) { rd += l.r_n * 8.0; wr += l.w_n * 8.0; }
        for (int f = 0; f < NFAM; ++f) { tr += kFam[f].read_mb; tw += kFam[f].write_mb; }
        rd /= 1e6; wr /= 1e6;                      // both sides in the census's MB
        if (std::fabs(rd - tr) > 0.02 * tr || std::fabs(wr - tw) > 0.02 * tw) {
            // The numbers go in the refusal: "drifted by more than 2 %" without the pair is the
            // kind of failure message that costs a debugging cycle to interpret.
            static std::string msg;
            msg = "engine byte mix drifted from the census by more than 2 % (plan read=" +
                  std::to_string(rd) + " MB vs census " + std::to_string(tr) + ", plan write=" +
                  std::to_string(wr) + " MB vs census " + std::to_string(tw) + ")";
            return msg.c_str();
        }
    }
    // The merge arm must really emit the bytes it booked, or "same total work" is a claim about a
    // number that never reached the device.
    if (c.total_mib > 0) {
        const double got = (double)(p.n_src + p.n_dst) * 8.0 / 1048576.0;
        if (std::fabs(got - c.total_mib) > 0.005 * c.total_mib)
            return "--total-mib did not survive rounding to whole elements per launch";
    }
    return nullptr;
}

// A block's own chunk bounds, as ONE function shared by the launcher, the expectation and the negative
// derivations. If the kernel and the checker ever disagree, they disagree here first.
void chunk_of(const Cfg& c, const LS& l, int b, long& r_lo, long& r_hi, long& w_lo, long& w_hi) {
    const long rc = (l.r_n + (long)l.grid * c.threads - 1) / ((long)l.grid * c.threads) * c.threads;
    const long wc = (l.w_n + (long)l.grid * c.threads - 1) / ((long)l.grid * c.threads) * c.threads;    r_lo = l.r_lo + (long)b * rc;
    if (r_lo > l.r_lo + l.r_n) r_lo = l.r_lo + l.r_n;
    r_hi = std::min(r_lo + rc, l.r_lo + l.r_n);
    if (r_hi < r_lo) r_hi = r_lo;
    w_lo = l.w_lo + (long)b * wc;
    if (w_lo > l.w_lo + l.w_n) w_lo = l.w_lo + l.w_n;
    w_hi = std::min(w_lo + wc, l.w_lo + l.w_n);
    if (w_hi < w_lo) w_hi = w_lo;
}

// The booked chunk sizes themselves, so the launcher can hand the kernel exactly the arithmetic the
// expectation predicts (see s1_touch: chunks are passed in, never re-derived from gridDim.x).
void chunks_of(const Cfg& c, const LS& l, long& rc, long& wc) {
    rc = (l.r_n + (long)l.grid * c.threads - 1) / ((long)l.grid * c.threads) * c.threads;
    wc = (l.w_n + (long)l.grid * c.threads - 1) / ((long)l.grid * c.threads) * c.threads;
}

// Which launch a launch-level or element-level injection picks, and which of its blocks: the middle
// launch, its first block. Derived, so a control's must-catch set is arithmetic, not a typed index.
long inj_launch_of(const Plan& ship) {
    for (size_t i = 0; i < ship.ls.size(); ++i)
        if ((int)i == (int)ship.ls.size() / 2) return (long)i;
    return -1;
}
long inj_rec_of(const Plan& ship) {
    const long i = inj_launch_of(ship);
    return i < 0 ? -1 : ship.ls[(size_t)i].rec_lo;
}
// The block-level injections pick a block that exists; for a one-block launch the tag/alias controls
// need no block at all.

// ---------------------------------------------------------------- allocation and fill
void alloc_fill(const Plan& ship) {
    reg_alloc(R_SRC, (size_t)ship.n_src * 8);
    reg_alloc(R_DST, (size_t)ship.n_dst * 8);
    reg_alloc(R_REC, (size_t)ship.n_rec * sizeof(Rec));
    if (ship.n_src > 0) {
        fill_word<<<512, 256>>>(reinterpret_cast<unsigned long long*>(g_reg[R_SRC].p), ship.n_src);
        CK(hipDeviceSynchronize());
        // Spot-check the fill formula at both ends. If the fill kernel and the host expectation ever
        // disagreed, every read sum would compare a formula against itself and pass vacuously.
        const long idx[2] = {0, ship.n_src - 1};
        for (long i : idx) {
            unsigned long long v = 0;
            CK(hipMemcpy(&v, g_reg[R_SRC].p + (size_t)i * 8, 8, hipMemcpyDeviceToHost));
            if (v != word_at(i)) { std::printf("FAIL_fill_mismatch at %ld\n", i); std::exit(4); }
        }
    }
    if (ship.n_dst > 0) {
        CK(hipMemset(g_reg[R_DST].p, 0xFF, (size_t)ship.n_dst * 8));
        g_reg[R_DST].snap.assign((size_t)ship.n_dst, kSentW);
    }
    if (ship.n_rec > 0) CK(hipMemset(g_reg[R_REC].p, 0xFF, (size_t)ship.n_rec * sizeof(Rec)));
}
void alloc_free_all() { for (int id = 0; id < NREG; ++id) reg_free(id); }

// ---------------------------------------------------------------- running a step
struct Counts { long launches_want = 0, launches_got = 0, syncs_want = 0, syncs_got = 0; };

// Submit one step: every launch in order, with a drain at the end of each segment (so `--syncs 1` IS
// the production burst shape: everything queued, one drain). The counters are incremented only on a
// submission that returned no error, and the printed values come from here, not from the config — that
// is what makes "the launch count an arm emitted must be printed by the arm" a measurement rather than
// a comment. The launch-level injections live here, on the submission side: a launch never submitted, a
// grid one block short, a base pointed at a neighbour's write range.
Counts run_step(const Cfg& c, const Plan& p) {
    Counts ct;
    ct.launches_want = (long)p.ls.size();
    ct.syncs_want = p.segments;
    const int nk = neg_kernel_id(c.neg);
    const long inj_l = inj_launch_of(p), inj_rec = inj_rec_of(p);
    size_t i = 0;
    for (int s = 0; s < (int)p.seg_lps.size(); ++s) {
        const int take = p.seg_lps[s];
        for (int j = 0; j < take && i < p.ls.size(); ++j, ++i) {
            const LS& l = p.ls[i];
            if ((c.neg == NEG_DROP_LAST && (long)i == (long)p.ls.size() - 1) ||
                (c.neg == NEG_DROP_MID && (long)i == inj_l))
                continue;                                  // never submitted, but still booked
            int grid = l.grid;
            if (c.neg == NEG_GRID_SHORT && (long)i == inj_l && grid > 1) grid -= 1;
            long wlo = l.w_lo, wn = l.w_n;
            if (c.neg == NEG_ALIAS && (long)i == inj_l && l.w_n > 0 && i > 0 && p.ls[i - 1].w_n > 0) {
                wlo = p.ls[i - 1].w_lo;                     // two writers, one address (attn §20.1)
                wn = std::min(l.w_n, p.ls[i - 1].w_n);
            }
            long rc, wc;
            chunks_of(c, l, rc, wc);
            s1_touch<<<grid, c.threads>>>(
                reinterpret_cast<const unsigned long long*>(g_reg[R_SRC].p),
                reinterpret_cast<unsigned long long*>(g_reg[R_DST].p), l.r_lo, l.r_n, wlo, wn,
                l.tag, reinterpret_cast<Rec*>(g_reg[R_REC].p), l.rec_lo, nk, inj_rec, rc, wc);
            if (hipGetLastError() == hipSuccess) ++ct.launches_got;
        }
        CK(hipDeviceSynchronize());
        ++ct.syncs_got;
    }
    return ct;
}

// ---------------------------------------------------------------- verification
struct Chk {
    int ok = 1;
    long rec_want = 0, rec_missing = 0, rec_wrong = 0;
    long w_want = 0, w_missing = 0, w_wrong = 0, stray = 0, guard = 0;
    long must_total = 0, must_caught = 0, same_total = 0, same_violated = 0;
    char why[224] = "";
    int nbad = 0;
    char bad[3][96];
    void note(const char* what, long at) {
        if (nbad < 3) { std::snprintf(bad[nbad], sizeof bad[nbad], "%s@%ld", what, at); ++nbad; }
    }
    std::string where() const {
        std::string out = "first_bad=";
        if (!nbad) return out + "none";
        for (int i = 0; i < nbad; ++i) { out += bad[i]; out += (i + 1 < nbad) ? "," : ""; }
        return out;
    }
};

// Which launch booked a write element, and which launch+block owns a record. Built from the shipped
// plan once, then used by both the content check and the derived must-catch classes.
struct Owners {
    std::vector<int> w_owner;      // element -> launch index, -1 for nobody
    std::vector<int> rec_launch;   // record -> launch index, -1 for nobody
    std::vector<int> rec_block;
};

void owners_build(const Plan& ship, Owners& ow) {
    ow.w_owner.assign((size_t)std::max<long>(ship.n_dst, 0), -1);
    ow.rec_launch.assign((size_t)std::max<long>(ship.n_rec, 0), -1);
    ow.rec_block.assign(ow.rec_launch.size(), -1);
    for (size_t i = 0; i < ship.ls.size(); ++i) {
        const LS& l = ship.ls[i];
        for (long e = l.w_lo; e < l.w_lo + l.w_n; ++e) ow.w_owner[(size_t)e] = (int)i;
        for (int b = 0; b < l.grid; ++b) {
            ow.rec_launch[(size_t)(l.rec_lo + b)] = (int)i;
            ow.rec_block[(size_t)(l.rec_lo + b)] = b;
        }
    }
}

// The three-way class, DERIVED from the injection's geometry — the same rule prototype/gdn_insitu
// enforces, and for the same reason: a hand-listed must-catch set rots into "everything" the moment the
// arm changes, and a control that catches everything proves nothing.
enum : int { DONT_CARE = -1, MUST_SAME = 0, MUST_DIFF = 1 };

int rec_class(int neg, const Plan& ship, long li, long b) {
    const long inj = inj_launch_of(ship);
    switch (neg) {
        case NEG_DROP_LAST:
            return li == (long)ship.ls.size() - 1 ? MUST_DIFF : MUST_SAME;
        case NEG_DROP_MID:
            return li == inj ? MUST_DIFF : MUST_SAME;
        case NEG_GRID_SHORT:
            if (li != inj) return MUST_SAME;
            return b == ship.ls[(size_t)li].grid - 1 ? MUST_DIFF : MUST_SAME;
        case NEG_SHORT_WRITE:
        case NEG_SHORT_READ:
            // The element-level controls shorten ONE block, and a shortened loop also shortens that
            // block's own reported counts: the record differs too. The split that matters is on the
            // WRITE side below — a short READ must leave every written byte identical, which is what
            // makes the read witness load-bearing rather than decorative.
            return (li == inj && b == 0) ? MUST_DIFF : MUST_SAME;
        case NEG_ALIAS:
        case NEG_LIE_WRITE:
            // The defect is an address or a missing store. The record reports what the block
            // accumulated, so it is self-consistent and provably UNMOVED: only the content check can
            // see either one, which is why both witnesses exist.
            return MUST_SAME;
        case NEG_TAG_CONST:
            return ship.ls[(size_t)li].tag == 7u ? MUST_SAME : MUST_DIFF;
        case NEG_PERMUTE_READS:
            // A rotation of one block's own read set preserves that block's multiset, so its sum and
            // count are provably unchanged, and no write is touched. BLIND BY CONSTRUCTION.
            return MUST_SAME;
        default:
            return DONT_CARE;
    }
}

int welem_class(int neg, const Cfg& c, const Plan& ship, const Owners& ow, long e) {
    const long li = ow.w_owner[(size_t)e];
    if (li < 0) return MUST_SAME;                       // nobody booked it: nothing may move
    const long inj = inj_launch_of(ship);
    const LS& l = ship.ls[(size_t)li];
    long rl, rh, wl, wh;
    chunk_of(c, l, 0, rl, rh, wl, wh);                  // only the injected block's chunk matters
    (void)rl; (void)rh;
    switch (neg) {
        case NEG_DROP_LAST:
            return li == (long)ship.ls.size() - 1 ? MUST_DIFF : MUST_SAME;
        case NEG_DROP_MID:
            return li == inj ? MUST_DIFF : MUST_SAME;
        case NEG_GRID_SHORT: {
            if (li != inj) return MUST_SAME;
            long d0, d1, a0, a1, b0, b1;
            chunk_of(c, l, l.grid - 1, d0, d1, a0, a1);
            return (e >= a0 && e < a1) ? MUST_DIFF : MUST_SAME;   // the dropped block's write chunk
        }
        case NEG_SHORT_WRITE: {
            if (li != inj) return MUST_SAME;
            return (e == wh - 1 && wh > wl) ? MUST_DIFF : MUST_SAME;   // the one dropped element
        }
        case NEG_SHORT_READ:
            // The read side only: every written byte must be IDENTICAL. A rig whose only witness were
            // the writes would call this clean, and that is the blindness this control declares.
            return MUST_SAME;
        case NEG_ALIAS: {
            if (li == inj) return MUST_DIFF;            // its own range never got its own values
            if (li == inj - 1) {
                const long stolen = std::min(l.w_n, ship.ls[(size_t)inj].w_n);
                return (e < l.w_lo + stolen) ? MUST_DIFF : MUST_SAME;   // overwritten by the alias
            }
            return MUST_SAME;
        }
        case NEG_LIE_WRITE: {
            if (li != inj) return MUST_SAME;
            return (e >= wl && e < wh) ? MUST_DIFF : MUST_SAME;   // that block's chunk was never stored
        }
        case NEG_TAG_CONST:
            return l.tag == 7u ? MUST_SAME : MUST_DIFF;
        case NEG_PERMUTE_READS:
            return MUST_SAME;
        default:
            return DONT_CARE;
    }
}

// What the kernel must have left behind, derived from the shipped plan and the chunk arithmetic — never
// from a record the kernel reported.
void verify_run(const Cfg& c, const Plan& ship, Owners& ow, Chk& chk) {
    // ---- records
    std::vector<Rec> recs((size_t)std::max<long>(ship.n_rec, 0));
    if (ship.n_rec > 0)
        CK(hipMemcpy(recs.data(), g_reg[R_REC].p, (size_t)ship.n_rec * sizeof(Rec),
                     hipMemcpyDeviceToHost));
    for (size_t i = 0; i < ship.ls.size(); ++i) {
        const LS& l = ship.ls[i];
        for (int b = 0; b < l.grid; ++b) {
            long rl, rh, wl, wh;
            chunk_of(c, l, b, rl, rh, wl, wh);
            const Rec& got = recs[(size_t)(l.rec_lo + b)];
            ++chk.rec_want;
            const bool sent = (got.rcount == kRecSent && got.wcount == kRecSent);
            // The kernel writes word_at(offset inside its own booked range) + tag, so the prediction is
            // the range sum over the OFFSETS, not over the addresses.
            const unsigned long long wexp =
                range_sum(wl - l.w_lo, wh - l.w_lo) +
                (unsigned long long)l.tag * (unsigned long long)(wh - wl);
            const int cls = rec_class(c.neg, ship, (long)i, b);
            const bool diff = sent || got.rsum != range_sum(rl, rh) || got.rcount != (unsigned)(rh - rl) ||
                              got.wsum != wexp || got.wcount != (unsigned)(wh - wl) || got.tag != l.tag;
            if (cls == MUST_DIFF) { ++chk.must_total; if (diff) ++chk.must_caught; else chk.note("rec", l.rec_lo + b); }
            else if (cls == MUST_SAME) { ++chk.same_total; if (diff) { ++chk.same_violated; chk.note("rec", l.rec_lo + b); } }
            if (sent) { ++chk.rec_missing; chk.note("rec_unwritten", l.rec_lo + b); }
            else if (diff) { ++chk.rec_wrong; chk.note("rec_wrong", l.rec_lo + b); }
        }
    }
    // ---- written bytes
    if (ship.n_dst > 0) {
        std::vector<unsigned long long> dst((size_t)ship.n_dst);
        CK(hipMemcpy(dst.data(), g_reg[R_DST].p, (size_t)ship.n_dst * 8, hipMemcpyDeviceToHost));
        for (long e = 0; e < ship.n_dst; ++e) {
            const int li = ow.w_owner[(size_t)e];
            const unsigned long long expv =
                (li < 0) ? g_reg[R_DST].snap[(size_t)e]
                         : word_at(e - ship.ls[(size_t)li].w_lo) +
                               (unsigned long long)ship.ls[(size_t)li].tag;
            ++chk.w_want;
            const bool diff = (dst[(size_t)e] != expv);
            const bool unwritten = (dst[(size_t)e] == kSentW);
            const int cls = welem_class(c.neg, c, ship, ow, e);
            if (cls == MUST_DIFF) { ++chk.must_total; if (diff) ++chk.must_caught; else chk.note("dst", e); }
            else if (cls == MUST_SAME) { ++chk.same_total; if (diff) { ++chk.same_violated; chk.note("dst", e); } }
            if (li >= 0 && diff) {
                if (unwritten) { ++chk.w_missing; chk.note("dst_unwritten", e); }
                else { ++chk.w_wrong; chk.note("dst_wrong", e); }
            } else if (li < 0 && diff) {
                ++chk.stray; chk.note("stray", e);
            }
        }
    }
    chk.guard = check_guards();
    if (chk.rec_want + chk.w_want == 0) {
        chk.ok = 0;
        std::snprintf(chk.why, sizeof chk.why, "FAIL_nothing_expected");
        return;
    }
    if (c.neg == NEG_NONE) {
        chk.ok = (chk.rec_missing == 0 && chk.rec_wrong == 0 && chk.w_missing == 0 &&
                  chk.w_wrong == 0 && chk.stray == 0 && chk.guard == 0);
        if (!chk.ok) std::snprintf(chk.why, sizeof chk.why,
                                   "rec_missing=%ld rec_wrong=%ld dst_unwritten=%ld dst_wrong=%ld stray=%ld guard=%ld",
                                   chk.rec_missing, chk.rec_wrong, chk.w_missing, chk.w_wrong,
                                   chk.stray, chk.guard);
    } else if (c.neg == NEG_PERMUTE_READS) {
        // The declared-blind case: the derived must-catch set is provably empty AND nothing may move.
        chk.ok = (chk.must_total == 0 && chk.same_violated == 0 && chk.rec_missing == 0 &&
                  chk.rec_wrong == 0 && chk.w_missing == 0 && chk.w_wrong == 0 && chk.guard == 0);
        std::snprintf(chk.why, sizeof chk.why,
                      "blind_must_catch=%ld same=%ld same_violated=%ld", chk.must_total,
                      chk.same_total, chk.same_violated);
    } else {
        chk.ok = (chk.must_total > 0 && chk.must_caught == chk.must_total && chk.same_violated == 0 &&
                  chk.guard == 0 && chk.stray == 0);
        if (chk.must_total == 0) std::snprintf(chk.why, sizeof chk.why, "BLIND-BY-GEOMETRY_empty_must_catch");
        else if (chk.must_caught != chk.must_total)
            std::snprintf(chk.why, sizeof chk.why, "must_catch_missed=%ld/%ld same_violated=%ld",
                          chk.must_total - chk.must_caught, chk.must_total, chk.same_violated);
        else if (chk.same_violated)
            std::snprintf(chk.why, sizeof chk.why, "must_not_move_violated=%ld", chk.same_violated);
        else if (chk.guard) std::snprintf(chk.why, sizeof chk.why, "guard_bytes_broken=%ld", chk.guard);
        else if (chk.stray) std::snprintf(chk.why, sizeof chk.why, "stray_writes=%ld", chk.stray);
    }
}

// A negative's derived sets, computed without a device so `--selftest-geom` can refuse a control that
// cannot fire — and one whose must-not-move set is empty, which is the other half of the same trap.
struct NegSets { long must = 0, same = 0, must_rec = 0, must_w = 0; int is_blind = 0; };
NegSets neg_derive(const Cfg& c, const Plan& ship) {
    Owners ow;
    owners_build(ship, ow);
    NegSets ns;
    for (size_t i = 0; i < ship.ls.size(); ++i)
        for (int b = 0; b < ship.ls[i].grid; ++b) {
            const int cls = rec_class(c.neg, ship, (long)i, b);
            ns.must += (cls == MUST_DIFF); ns.same += (cls == MUST_SAME);
            ns.must_rec += (cls == MUST_DIFF);
        }
    for (long e = 0; e < ship.n_dst; ++e) {
        const int cls = welem_class(c.neg, c, ship, ow, e);
        ns.must += (cls == MUST_DIFF); ns.same += (cls == MUST_SAME);
        ns.must_w += (cls == MUST_DIFF);
    }
    ns.is_blind = (c.neg == NEG_PERMUTE_READS);
    return ns;
}

// ---------------------------------------------------------------- engine spec echo
void print_engine_spec() {
    long fl = 0, fs = 0;
    for (int f = 0; f < NFAM; ++f) {
        std::printf("ESPEC family=%s launches=%d segments=%d read_mb=%.3f write_mb=%.3f\n",
                    kFamName[f], kFam[f].launches, kFam[f].segments, kFam[f].read_mb,
                    kFam[f].write_mb);
        fl += kFam[f].launches; fs += kFam[f].segments;
    }
    std::printf("ESPEC-TOTAL launches=%ld segments=%ld (quoted [Q1]: 340 and 305)\n", fl, fs);
}

// ---------------------------------------------------------------- host geometry selftest
int selftest_geom() {
    int fails = 0;
    auto need = [&](bool cond, const char* what) {
        std::printf("  %-62s %s\n", what, cond ? "ok" : "FAIL");
        if (!cond) ++fails;
    };

    // 1. The engine's own populations, re-derived and equal to what the engine prints.
    std::printf("== engine population, re-derived (quoted [Q1] = 340 launches / 305 segments)\n");
    need(kWordK * kWordInv == 1ull,
         "the read-fill multiplier really is invertible mod 2^64 (so a sum knows its element count)");
    need(range_sum(0, 3) == word_at(0) + word_at(1) + word_at(2) &&
         range_sum(7, 40) == range_sum(0, 40) - range_sum(0, 7),
         "the closed-form range sum equals the elementwise sum it replaces");
    Cfg ec; ec.launches = kEngineLaunches; ec.syncs = kEngineSegments;
    ec.pattern = PT_ENGINE; ec.body = BD_ENGINE;
    Plan ep; plan_build(ec, ep);
    const char* bad = plan_invariants(ec, ep);
    need(bad == nullptr, "engine plan passes invariants");
    if (bad) std::printf("      reason: %s\n", bad);
    need((int)ep.ls.size() == kEngineLaunches && ep.segments == kEngineSegments,
         "engine plan is 340 launches in 305 segments");
    long two = 0, one = 0, zero = 0;
    for (int s : ep.seg_lps) { two += (s == 2); one += (s == 1); zero += (s == 0); }
    need(two == 36 && one == 268 && zero == 1,
         "engine segments: 36 of two launches (gdn), 268 of one, 1 with none (host_tail)");
    double rd = 0, wr = 0;
    for (const LS& l : ep.ls) { rd += l.r_n * 8.0 / 1e6; wr += l.w_n * 8.0 / 1e6; }   // census unit
    std::printf("      engine step demands read=%.1f MB write=%.2f MB (census TOTAL 1532.311 / 24.409)\n",
                rd, wr);
    need(std::fabs(rd - 1532.311) < 1.0 && std::fabs(wr - 24.409) < 0.5,
         "engine byte mix matches the step census within rounding");

    // 2. Every arm shape the sweeps use must pass invariants.
    std::printf("== arm shapes\n");
    struct Shape { const char* name; Cfg c; };
    Shape shapes[] = {
        {"ladder N=1",       [&]{ Cfg c; c.launches = 1;   c.syncs = 1; c.body = BD_TINY; return c; }()},
        {"ladder N=48",      [&]{ Cfg c; c.launches = 48;  c.syncs = 1; c.body = BD_TINY; return c; }()},
        {"ladder N=165",     [&]{ Cfg c; c.launches = 165; c.syncs = 1; c.body = BD_MID; return c; }()},
        {"ladder N=305",     [&]{ Cfg c; c.launches = 305; c.syncs = 1; c.body = BD_EMPTY; return c; }()},
        {"ladder N=340",     [&]{ Cfg c; c.launches = 340; c.syncs = 1; c.body = BD_MID; return c; }()},
        {"ladder N=680",     [&]{ Cfg c; c.launches = 680; c.syncs = 1; c.body = BD_TINY; return c; }()},
        {"sync axis S=N",    [&]{ Cfg c; c.launches = 340; c.syncs = 340; c.body = BD_TINY; return c; }()},
        {"sync axis S=8",    [&]{ Cfg c; c.launches = 340; c.syncs = 8; c.body = BD_TINY; return c; }()},
        {"skew S=7 of 100",  [&]{ Cfg c; c.launches = 100; c.syncs = 7; c.body = BD_TINY; return c; }()},
        {"merge L=1",        [&]{ Cfg c; c.launches = 1;   c.syncs = 1; c.total_mib = 1024; return c; }()},
        {"merge L=340",      [&]{ Cfg c; c.launches = 340; c.syncs = 1; c.total_mib = 1024; return c; }()},
        {"engine 305 drains", ec},
        {"engine 1 drain",   [&]{ Cfg c = ec; c.syncs = 1; return c; }()},
        {"engine 340 drains", [&]{ Cfg c = ec; c.syncs = kEngineLaunches; return c; }()},
    };
    for (Shape& s : shapes) {
        Plan p;
        plan_build(s.c, p);
        const char* b = plan_invariants(s.c, p);
        char msg[160];
        std::snprintf(msg, sizeof msg, "%s (launches=%zu segments=%d)", s.name, p.ls.size(), p.segments);
        need(b == nullptr, msg);
        if (b) std::printf("      reason: %s\n", b);
    }
    {
        Cfg c; c.launches = 100; c.syncs = 7;
        Plan p; plan_build(c, p);
        long sum = 0; int lo = 1 << 30, hi = 0;
        for (int s : p.seg_lps) { sum += s; lo = std::min(lo, s); hi = std::max(hi, s); }
        need(sum == 100 && hi - lo <= 1 && (int)p.seg_lps.size() == 7,
             "uniform 100 launches / 7 drains distributes 15,15,14,14,14,14,14");
    }

    // 3. Refusals. An option that cannot refuse is not an option (gdn_insitu's blob18 lesson).
    std::printf("== refusals (each must refuse)\n");
    struct Bad { Cfg c; const char* name; };
    Bad bads[] = {
        {[&]{ Cfg c; c.syncs = 0; return c; }(), "syncs 0 (submission-only is another tool)"},
        {[&]{ Cfg c; c.launches = 4; c.syncs = 9; return c; }(), "syncs > launches (uniform)"},
        {[&]{ Cfg c = ec; c.launches = 300; return c; }(), "engine pattern with --launches != 340"},
        {[&]{ Cfg c = ec; c.syncs = 100; return c; }(), "engine pattern with a foreign segment count"},
        {[&]{ Cfg c = ec; c.body = BD_TINY; return c; }(), "engine pattern with a synthetic body"},
        {[&]{ Cfg c; c.body = BD_ENGINE; return c; }(), "engine body mix without the engine pattern"},
        {[&]{ Cfg c = ec; c.total_mib = 64; return c; }(), "total-mib and engine mix both booked"},
        {[&]{ Cfg c; c.threads = 100; return c; }(), "threads not a multiple of 64"},
    };
    for (Bad& b : bads) {
        const char* r = refuse(b.c);
        char msg[160];
        std::snprintf(msg, sizeof msg, "refuse: %s", b.name);
        need(r != nullptr, msg);
        if (!r) std::printf("      DID NOT REFUSE\n");
    }

    // 4. The derived control sets. A negative whose must-catch set is empty cannot fire, and one whose
    //    must-not-move set is empty catches nothing either; the blind case must be provably blind.
    std::printf("== derived control sets (must-catch AND must-not-move, per control)\n");
    struct NegCase { const char* name; int neg; int body; int n; };
    NegCase ncs[] = {
        {"drop_last",     NEG_DROP_LAST,     BD_TINY, 32},
        {"drop_mid",      NEG_DROP_MID,      BD_TINY, 32},
        {"grid_short",    NEG_GRID_SHORT,    BD_MID,  32},
        {"short_write",   NEG_SHORT_WRITE,   BD_MID,  32},
        {"short_read",    NEG_SHORT_READ,    BD_MID,  32},
        {"alias_write",   NEG_ALIAS,         BD_TINY, 32},
        {"tag_const",     NEG_TAG_CONST,     BD_TINY, 32},
        {"lie_write",     NEG_LIE_WRITE,     BD_MID,  32},
        {"permute_reads", NEG_PERMUTE_READS, BD_MID,  32},
    };
    for (NegCase& nc : ncs) {
        Cfg c; c.launches = nc.n; c.syncs = 4; c.body = nc.body; c.neg = nc.neg;
        Plan p; plan_build(c, p);
        const NegSets ns = neg_derive(c, p);
        char msg[160];
        std::snprintf(msg, sizeof msg, "%-14s must=%ld (rec=%ld w=%ld) same=%ld", nc.name, ns.must,
                      ns.must_rec, ns.must_w, ns.same);
        if (nc.neg == NEG_PERMUTE_READS) need(ns.must == 0 && ns.same > 0, msg);
        else need(ns.must > 0 && ns.same > 0, msg);
    }
    {   // The three structural claims the witness design rests on, asserted rather than argued.
        Cfg base; base.launches = 32; base.syncs = 4; base.body = BD_MID;
        auto derive = [&](int neg) {
            Cfg d = base; d.neg = neg;
            Plan p; plan_build(d, p);
            return neg_derive(d, p);
        };
        const NegSets alias = derive(NEG_ALIAS);
        const NegSets lie = derive(NEG_LIE_WRITE);
        const NegSets sr = derive(NEG_SHORT_READ);
        need(alias.must_rec == 0 && alias.must_w > 0,
             "alias_write moves NO record: only the byte check can catch an aliased address");
        need(lie.must_rec == 0 && lie.must_w > 0,
             "lie_write moves NO record: a record reporting its own arithmetic cannot catch itself");
        need(sr.must_rec > 0 && sr.must_w == 0,
             "short_read moves records but NO written byte: the read-sum witness is load-bearing");
    }
    {   // The no-launch arm has nothing to expect: a declared state, not a vacuous pass.
        Cfg c; c.launches = 0; c.syncs = 0;
        Plan p; plan_build(c, p);
        need(p.ls.empty() && p.n_rec == 0, "--launches 0 books an empty window (row prints NO-TIMING)");
    }
    std::printf("SELFTEST-GEOM %s fails=%d\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------- device cases
struct CaseSpec {
    const char* name;
    int kind;   // 0 = positive, must verify clean; 1 = negative, must be caught; 2 = must be blind
    Cfg c;
};

std::vector<CaseSpec> case_list() {
    std::vector<CaseSpec> v;
    auto push = [&](const char* nm, int kind, Cfg c) { v.push_back({nm, kind, c}); };
    // Positives, across every shape the sweeps can set. Each is byte-verified, not counted.
    push("pos_empty_340", 0, [&]{ Cfg c; c.launches = 340; c.syncs = 1; c.body = BD_EMPTY; return c; }());
    push("pos_tiny_340", 0, [&]{ Cfg c; c.launches = 340; c.syncs = 1; c.body = BD_TINY; return c; }());
    push("pos_tiny_sync_each", 0, [&]{ Cfg c; c.launches = 48; c.syncs = 48; c.body = BD_TINY; return c; }());
    push("pos_tiny_skew", 0, [&]{ Cfg c; c.launches = 100; c.syncs = 7; c.body = BD_TINY; return c; }());
    push("pos_mid_48", 0, [&]{ Cfg c; c.launches = 48; c.syncs = 1; c.body = BD_MID; return c; }());
    push("pos_merge_8", 0, [&]{ Cfg c; c.launches = 8; c.syncs = 1; c.total_mib = 64; return c; }());
    push("pos_engine", 0, [&]{ Cfg c; c.launches = kEngineLaunches; c.syncs = kEngineSegments;
                               c.pattern = PT_ENGINE; c.body = BD_ENGINE; return c; }());
    // Negatives: each must move its derived set and leave its derived complement alone.
    push("neg_drop_last", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_TINY; c.neg = NEG_DROP_LAST; return c; }());
    push("neg_drop_mid", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_TINY; c.neg = NEG_DROP_MID; return c; }());
    push("neg_grid_short", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_MID; c.neg = NEG_GRID_SHORT; return c; }());
    push("neg_short_write", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_MID; c.neg = NEG_SHORT_WRITE; return c; }());
    push("neg_short_read", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_MID; c.neg = NEG_SHORT_READ; return c; }());
    push("neg_alias_write", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_TINY; c.neg = NEG_ALIAS; return c; }());
    push("neg_tag_const", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_TINY; c.neg = NEG_TAG_CONST; return c; }());
    push("neg_lie_write", 1, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_MID; c.neg = NEG_LIE_WRITE; return c; }());
    // The declared-blind case: invisible by proof, and the case requires that invisibility.
    push("blind_permute_reads", 2, [&]{ Cfg c; c.launches = 32; c.syncs = 4; c.body = BD_MID; c.neg = NEG_PERMUTE_READS; return c; }());
    return v;
}

// One case, one process. Exit 0 means "behaved as its kind demands": a positive verifies clean, a
// negative goes red on its own derived sets, the blind case verifies untouched.
int run_case(const CaseSpec& s, char* line, size_t ln) {
    if (const char* r = refuse(s.c)) { std::snprintf(line, ln, "REFUSED %s", r); return 9; }
    Plan ship;
    plan_build(s.c, ship);
    if (const char* b = plan_invariants(s.c, ship)) { std::snprintf(line, ln, "PLAN-BAD %s", b); return 9; }
    alloc_fill(ship);
    const Counts ct = run_step(s.c, ship);          // the step that gets verified, untimed
    Owners ow; owners_build(ship, ow);
    Chk chk;
    verify_run(s.c, ship, ow, chk);
    std::snprintf(line, ln,
                  "%s verify=%s launches=%ld/%ld syncs=%ld/%ld rec_bad=%ld/want %ld "
                  "w_bad=%ld/want %ld must=%ld/%ld same=%ld/viol %ld %s %s",
                  s.kind == 0 ? (chk.ok ? "CLEAN" : "FAIL")
                  : s.kind == 2 ? (chk.ok ? "BLIND-OK" : "BLIND-BROKEN")
                                : (chk.ok ? "CAUGHT" : "NOT-CAUGHT"),
                  chk.ok ? "ok" : "FAIL", ct.launches_got, ct.launches_want, ct.syncs_got,
                  ct.syncs_want, chk.rec_missing + chk.rec_wrong, chk.rec_want,
                  chk.w_missing + chk.w_wrong, chk.w_want, chk.must_caught, chk.must_total,
                  chk.same_total, chk.same_violated, chk.why, chk.where().c_str());
    alloc_free_all();
    // chk.ok already means "this case behaved as its kind demands": clean for a positive, derived set
    // satisfied for a negative, nothing moved for the blind case. So the exit code is the same 0/1.
    return chk.ok ? 0 : 1;
}

int selftest_device() {
    const std::vector<CaseSpec> cases = case_list();
    int want_caught = 0, want_blind = 0, want_clean = 0;
    for (const CaseSpec& s : cases) {
        if (s.kind == 1) ++want_caught; else if (s.kind == 2) ++want_blind; else ++want_clean;
    }
    int parsed = 0, bad = 0, caught = 0, blind = 0, clean = 0;
    char line[512];
    for (const CaseSpec& s : cases) {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            const int rc = run_case(s, line, sizeof line);
            std::printf("CASE %-20s %s\n", s.name, line);
            fflush(stdout);
            _exit(rc);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        ++parsed;
        const int code = WIFEXITED(st) ? WEXITSTATUS(st) : 3;
        if (s.kind == 1) { if (code == 0) ++caught; else ++bad; }
        else if (s.kind == 2) { if (code == 0) ++blind; else ++bad; }
        else { if (code == 0) ++clean; else ++bad; }
    }
    const bool ok = (bad == 0 && parsed == (int)cases.size() && caught == want_caught &&
                     blind == want_blind && clean == want_clean);
    std::printf("SELFTEST %s cases=%d parsed=%d children_bad=%d clean=%d/%d caught=%d/%d blind=%d/%d\n",
                ok ? "ok" : "FAILED", (int)cases.size(), parsed, bad, clean, want_clean, caught,
                want_caught, blind, want_blind);
    return ok ? 0 : 1;
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------- main
int real_main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);      // lines survive a forked child that dies in a fault
    Cfg c;
    int mode = 0;                             // 1 selftest, 2 selftest-geom, 3 print-cases, 4 engine spec
    auto need_val = [&](int& i) {
        if (i + 1 >= argc) { std::printf("ABORT missing value for %s\n", argv[i]); std::exit(2); }
        return std::string(argv[++i]);
    };
    auto need_num = [&](int& i, const char* flag) {
        const std::string s = need_val(i);
        char* end = nullptr;
        const double d = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || (end != nullptr && *end != '\0')) {
            std::printf("ABORT %s wants a number, got '%s'\n", flag, s.c_str());
            std::exit(2);
        }
        return d;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") mode = 1;
        else if (a == "--selftest-geom") mode = 2;
        else if (a == "--print-cases") mode = 3;
        else if (a == "--print-engine-spec") mode = 4;
        else if (a == "--time") c.time_flag = 1;
        else if (a == "--launches") c.launches = (int)need_num(i, "--launches");
        else if (a == "--syncs") c.syncs = (int)need_num(i, "--syncs");
        else if (a == "--threads") c.threads = (int)need_num(i, "--threads");
        else if (a == "--grid-cap") c.grid_cap = (int)need_num(i, "--grid-cap");
        else if (a == "--passes") c.passes = (int)need_num(i, "--passes");
        else if (a == "--reps") c.reps = (int)need_num(i, "--reps");
        else if (a == "--iters") c.iters = (int)need_num(i, "--iters");
        else if (a == "--read-kib") { c.read_kib = need_num(i, "--read-kib"); c.body = BD_CUSTOM; }
        else if (a == "--write-kib") { c.write_kib = need_num(i, "--write-kib"); c.body = BD_CUSTOM; }
        else if (a == "--total-mib") c.total_mib = need_num(i, "--total-mib");
        else if (a == "--pattern") {
            const std::string s = need_val(i);
            if (s == "uniform") c.pattern = PT_UNIFORM;
            else if (s == "engine") { c.pattern = PT_ENGINE; c.launches = kEngineLaunches; c.syncs = kEngineSegments; }
            else { std::printf("ABORT bad --pattern %s (uniform|engine)\n", s.c_str()); return 2; }
        } else if (a == "--body") {
            const std::string s = need_val(i);
            if (s == "tiny") c.body = BD_TINY;
            else if (s == "empty") c.body = BD_EMPTY;
            else if (s == "mid") c.body = BD_MID;
            else if (s == "fat") c.body = BD_FAT;
            else if (s == "custom") c.body = BD_CUSTOM;
            else { std::printf("ABORT bad --body %s (tiny|empty|mid|fat|custom)\n", s.c_str()); return 2; }
        } else if (a == "--body-mix") {
            const std::string s = need_val(i);
            if (s == "engine") c.body = BD_ENGINE;
            else { std::printf("ABORT bad --body-mix %s (engine)\n", s.c_str()); return 2; }
        } else if (a == "--neg") {
            const std::string s = need_val(i);
            int k = 0;
            for (; k < NEG_COUNT; ++k) if (s == kNegName[k]) break;
            if (k == NEG_COUNT) { std::printf("ABORT bad --neg %s\n", s.c_str()); return 2; }
            c.neg = k;
        } else if (a == "--pages") {
            const std::string s = need_val(i);
            if (s == "cold") {
                std::printf("ABORT REFUSED --pages cold: first touch of fresh pages runs ~7x slow "
                            "(devices/), so cold is refused, not averaged. The fill plus one untimed "
                            "step is this rig's warm-up.\n");
                return 2;
            }
            if (s != "warm") {
                std::printf("ABORT bad --pages %s (warm; cold is refused by design)\n", s.c_str());
                return 2;
            }
        } else { std::printf("ABORT unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mode == 2) return selftest_geom();
    if (mode == 4) { print_engine_spec(); return 0; }
    if (mode == 3) {
        const std::vector<CaseSpec> cs = case_list();
        for (const CaseSpec& s : cs) std::printf("CASE-ROW %s kind=%d\n", s.name, s.kind);
        std::printf("CASE-ROWS total=%zu\n", cs.size());
        return 0;
    }
    if (mode == 1) return selftest_device();

    if (const char* r = refuse(c)) { std::printf("ABORT: %s\n", r); return 3; }
    c.arm = arm_key(c);          // composed once, after every knob is known
    Plan ship;
    plan_build(c, ship);
    if (const char* b = plan_invariants(c, ship)) { std::printf("ABORT plan: %s\n", b); return 3; }

    if (ship.ls.empty()) {
        // The no-launch arm. There is no wall to report, and this part's timer would invent one.
        std::printf("ROW pattern=%s body=%s launches=0 syncs=0 NO-TIMING no_launch=1 "
                    "reason=timed_window_launches_nothing\n",
                    c.pattern == PT_ENGINE ? "engine" : "uniform", body_name(c.body));
        return 0;
    }

    alloc_fill(ship);
    const Counts ct = run_step(c, ship);            // the verified step, untimed
    Owners ow; owners_build(ship, ow);
    Chk chk;
    verify_run(c, ship, ow, chk);

    int seg_min = 1 << 30, seg_max = 0, grid_max = 0;
    for (int s : ship.seg_lps) { seg_min = std::min(seg_min, s); seg_max = std::max(seg_max, s); }
    for (const LS& l : ship.ls) grid_max = std::max(grid_max, l.grid);
    const double read_mib = (double)ship.read_bytes / 1048576.0;
    const double write_mib = (double)ship.write_bytes / 1048576.0;
    std::printf("ROW arm=%s pattern=%s body=%s launches=%ld syncs=%ld want_launches=%d want_syncs=%d "
                "count_match=%d seg_lps_min=%d seg_lps_max=%d threads=%d grid_max=%d recs=%ld "
                "read_mib=%.4f write_mib=%.4f demand_mib=%.4f verify=%s checked_recs=%ld "
                "checked_w=%ld %s %s",
                c.arm.c_str(),
                c.pattern == PT_ENGINE ? "engine" : "uniform", body_name(c.body), ct.launches_got,
                ct.syncs_got, c.launches, c.syncs,
                (ct.launches_got == c.launches && ct.syncs_got == c.syncs) ? 1 : 0,
                seg_min, seg_max, c.threads, grid_max, (long)ship.n_rec, read_mib, write_mib,
                read_mib + write_mib, chk.ok ? "ok" : "FAIL", chk.rec_want, chk.w_want, chk.why,
                chk.where().c_str());
    if (!c.time_flag) {
        std::printf(" NO-TIMING reason=no_--time_flag\n");
        alloc_free_all();
        return chk.ok ? 0 : 1;
    }
    if (!chk.ok) {
        // Never print a millisecond figure for a row the witnesses did not clear.
        std::printf(" NO-TIMING reason=verify_not_ok\n");
        alloc_free_all();
        return 1;
    }
    // Timed reps. steady_clock, because hipEventElapsedTime returns exactly 0 below ~1 ms on this part
    // and a step at N=1 is far below that. One warm step already ran (the verified one); each rep warms
    // again and then takes the best of `iters` whole-step walls, and the per-rep spread is printed so a
    // bimodal row cannot hide (AGENTS: dispatch-bound timings do not repeat).
    std::vector<double> per;
    double best = 1e30, worst = 0.0;
    for (int r = 0; r < c.reps; ++r) {
        run_step(c, ship);                          // warm within the rep
        double bm = 1e30;
        for (int it = 0; it < c.iters; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            run_step(c, ship);
            const auto t1 = std::chrono::steady_clock::now();
            bm = std::min(bm, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        per.push_back(bm);
        best = std::min(best, bm);
        worst = std::max(worst, bm);
    }
    // One RESULT line per invocation, printed AFTER the ROW line because a harness greps for RESULT and
    // a CPU-side line printed after it would swallow it (AGENTS, the lm_head_gemv `tail -1` trap).
    std::printf("RESULT arm=%s pattern=%s body=%s launches=%d syncs=%d passes=%d pages=warm "
                "threads=%d grid_max=%d recs=%ld read_mib=%.4f write_mib=%.4f demand_mib=%.4f "
                "seg_lps_min=%d seg_lps_max=%d verify=ok launches_executed=%ld syncs_executed=%ld "
                "ms_min=%.6f ms_median=%.6f ms_max=%.6f reps=%d iters=%d\n",
                c.arm.c_str(), c.pattern == PT_ENGINE ? "engine" : "uniform", body_name(c.body),
                c.launches, c.syncs, c.passes, c.threads, grid_max, (long)ship.n_rec,
                read_mib, write_mib, read_mib + write_mib, seg_min, seg_max,
                ct.launches_got, ct.syncs_got, best, median_of(per), worst, c.reps, c.iters);
    alloc_free_all();
    return 0;
}

}  // namespace

int main(int argc, char** argv) { return real_main(argc, argv); }
