/*
 * prototype/wave_shape/wave_shape.hip.cpp — ROUTE S1 ITEM (2): is a launch expensive because it is a
 * LAUNCH, or because a one-row launch has too few workgroups to hide anything behind?
 *
 * WHY THIS RIG EXISTS. `prototype/launch_price` (gate zero) settled the *count* term by deletion: the
 * cast fold removed 97 launches and moved the wall −0.3008 ms at 4,032 lanes = **3.10 us per launch**
 * (`engine/generation-limits.md` §1, §4, §7 item 1 — quoted, never re-taken here). What gate zero could
 * NOT settle is the *shape* term, because its merge arm compared 340 launches at `grid_max=271` against
 * 48 at `grid_max=1918`: launch count and workgroups-per-launch moved together, so its 13.08 us/launch
 * is uninterpretable (`prototype/launch_price/S1-GATEZERO.md` §1, and the same reading in
 * `progress/2026-10-21/route-S1-segments-2026-10-21.md` §7). The A12 rule is the one this rig exists to
 * honour: **read workgroups-per-launch on both sides** (`AGENTS.md`, the A12 row).
 *
 * TWO ARMS, one rig, because the two questions share the witness machinery:
 *
 *   ARM 1 — the (L, G) matrix at fixed total bytes. Total READ bytes are pinned to the step's own read
 *     census (1461.3 MiB, `engine/generation-limits.md` §1 via `tools/step_traffic_model.py`), and
 *     launch count L and blocks-per-launch G vary INDEPENDENTLY: every reachable cell is reachable at
 *     several different L and several different G, so the axes can be told apart. The cells sharing a
 *     block total B = L×G form the ISO-POPULATION DIAGONAL, and on a diagonal the byte total, the block
 *     total, the record count and the per-block work are all *equal by construction* — only how the same
 *     work is divided into launches differs. Gate zero's one confounded pair was one diagonal pair; this
 *     matrix has the diagonal AND the iso-L rows (G moves at fixed L: workgroups per launch, count
 *     fixed) AND the iso-G columns (L moves at fixed G), which is what breaks the confound apart.
 *
 *   ARM 2 — split-K on the shape a one-token `down_proj` really presents, reproduced from source:
 *     `engine/decode_step.cpp:1109` calls `kernels::mlp_down(w_down, act_f32, delta, kHidden,
 *     kIntermediate, n)` with `kHidden = 1024`, `kIntermediate = 3584` (`engine/state.h:53-54`) and
 *     `engine/kernels_mlp.cpp:240-241` sets threads = kRowsPerBlock × warp_width and blocks =
 *     ceil(rows / kRowsPerBlock) with `ROWS_PER_BLOCK 8`
 *     (`prototype/mlp_gemv/mlp_gemv_kernels.h:23`), and warp width is 32 on this part
 *     (`/sys/.../nodes/1/properties` → `wave_front_size 32`). So **down_proj = 1,024 rows over 128
 *     blocks of 256 threads**, which is the figure D2 §5 quotes, not the 3,584 one — 3,584 is
 *     `gateup`'s row count and `down_proj`'s K. The physical asymmetry this arm is about: a one-row
 *     GEMV gives each output row exactly ONE warp (32 lanes over 448 `uint4` words = 14 dependent
 *     accumulations per lane), so a one-token down_proj puts 128 blocks × 8 wavefronts = 1,024
 *     wavefronts on a 40-CU machine = 25.6 of an architectural 64 wavefronts/CU
 *     (`max_waves_per_simd 16` × 2 simds/CU). Half the slots are empty and there is nothing else in
 *     flight. Splitting K is the only lever that fills them, because it multiplies the WARP count at
 *     constant bytes.
 *
 * WHAT EACH ARM CLAIMS TO ISOLATE, exactly:
 *   arm 1, iso-L row      = workgroups per launch, launch count held. The A12 axis.
 *   arm 1, iso-G column   = launch count, workgroups per launch held.
 *   arm 1, iso-B diagonal = both at once (gate zero's confound), readable here because the row and
 *                           column through each point are also in the matrix.
 *   arm 2, wpr=1          = the shipped kernel, re-transcribed, and it must reproduce it bit for bit.
 *   arm 2, wpr=2/4/8      = K split INSIDE the launch: same launch count, same threads, same bytes,
 *                           2/4/8× the wavefronts, 1/2/4/8 the serial accumulation per lane. A pure
 *                           shape change at equal L, which is exactly the comparison gate zero lacked.
 *   arm 2, two:S          = K split ACROSS launches with a real two-pass reduction (partials, not
 *                           atomics). Buys S× the wavefronts of any single launch, at a cost of one
 *                           extra launch per down_proj = 24 launches/step; at the quoted 3.10 us/launch
 *                           that tax is 0.0744 ms/step and the fitter states it beside the gain.
 *
 * BIT-IDENTICAL, AND WHERE IT CANNOT BE. `down_ship` is a character-for-character transcription of the
 * shipped `mlp_down<FF>` (`prototype/mlp_gemv/mlp_gemv_kernels.h:191-211`, with the F32 act that ships
 * today because `Options::cast_fold` defaults false — `engine/decode_run.cpp:336`), and it is required
 * to reproduce the CPU mirror bit for bit. `down_intra<1>` is required to reproduce `down_ship` bit for
 * bit, which is the template's own degenerate case and therefore a real witness that the parameterised
 * form degenerates correctly. **wpr>1 and two:S cannot be bit-identical to the shipped form**, and this
 * is arithmetic, not sloppiness: the shipped reduction is 448 words over 32 lanes with one strided walk
 * and one 5-step ladder, and any K split replaces that with wpr (or S) partial sums each covering a
 * contiguous slice; F32 addition is not associative, so the products are the same floats but the sum is
 * a different sum. The rig therefore reports, for every split arm, device-vs-shipped-device `ndiff`,
 * max ULP (with the sign-correct BF16 ULP map of `prototype/bf16_ulp.py`, not a `uint16` view — that
 * bug is in `AGENTS.md`), and mean |Δ| ÷ max |ref|, and it refuses to call a band "1 ULP" without the
 * floor probe beside it.
 *
 * VERIFICATION. The witness discipline is `prototype/launch_price`'s, carried over: one record per
 * (launch, block) holding {read sum, read count, write sum, write count, tag}; the read fill is
 * `word(p) = K·(2p+1)` with K odd, so a range sum has a CLOSED FORM, integer addition is associative,
 * and K's invertibility mod 2⁶⁴ means a sum also knows its element count (asserted, not assumed).
 * Expectations are DERIVED ON THE HOST from the same struct the launcher is built from, never from what
 * the kernel reported. Written values are indexed by offset inside the writer's own booked range, so an
 * aliased launch stays byte-identical in its own record and only the content check can see it. Chunk
 * sizes are BOOKED numbers passed to the kernel, never re-derived from `gridDim.x`. Poison guard bands
 * on both sides of every region. Every must-catch and must-not-catch set is DERIVED from the injection's
 * geometry, `--selftest-geom` refuses any control whose derived set is empty on either side, and every
 * device case runs in its own forked child (one faulting kernel poisons the context and every later
 * case silently never ran — `tools/mixed_bw`).
 *
 * THE MATRIX ARM'S WITNESS IS THE WHOLE POINT, so it is spelled out: a cell is admissible only if
 *   - every block's read sum equals the closed form over exactly its own booked range (so a cell that
 *     skipped a range, based a range wrong, or dropped a block cannot print a row at all),
 *   - every written byte equals the value its owner must have stored, at the owner's offsets, and every
 *     byte outside a booked range is untouched (write-landing read-back with a byte count that CAN
 *     differ from what was booked: `N_ROW_DOUBLE` doubles one block's write and is caught as a stray),
 *   - no guard byte moved,
 *   - the printed per-cell totals equal the plan's totals, so "the step really read 1461.3 MiB" is a
 *     measurement from the records, not a comment about the argv.
 *
 * NO TIMINGS BY THE BUILDER. `--time` exists for the main agent and nothing prints `ms` without it; a
 * row whose witnesses are not clean prints `NO-TIMING reason=verify_not_ok`, a window that launches
 * nothing prints `NO-TIMING no_launch=1`, and `--pages cold` is refused outright (first touch of fresh
 * pages runs ~7× slow). No bandwidth rate is printed anywhere in this file: a demanded rate here would
 * exceed the measured plateau and read as a record (`AGENTS.md`: >240 GB/s is skipped work). Every
 * microsecond and millisecond figure quoted above is labelled and comes from the ledger, not from this
 * rig. `sweep.sh` is the main agent's timed matrix and takes `tools/gpu_lock.sh` itself; `check_wave_shape.sh`
 * is the correctness gate, also self-locking, because a caller holding the lock and then running a
 * self-locking driver deadlocks it (`AGENTS.md`, the three-gate-drivers-self-lock row).
 *
 * QUOTED, never measured here (labelled [Qx] wherever used):
 *   [Q1] 3.10 us per launch, from deleting 97 cast launches: −0.3008 ms at 4,032 lanes —
 *        `engine/generation-limits.md` §1 `cast` row, §7 item (1).
 *   [Q2] step read census 1532.311 MB = 1461.33 MiB; write 24.409 MB = 23.28 MiB; mlp family 528.482 MB
 *        read over 48 launches at 3.3628 ms stage line (70 % of floor), gdn 30 % — same §1, floors from
 *        §0 fact 1's 237 GB/s plateau.
 *   [Q3] step wall 10.9357 ms/token at 4,032 lanes; 102.2 tok/s at 128 — `engine/performance-status.md`
 *        §1 via `engine/generation-limits.md` "Operating point".
 *   [Q4] down_proj = 1,024 rows / 128 blocks on 40 CUs, and that is where D2 left the unpriced term —
 *        `prototype/mlp_insitu/D2-DIAGNOSIS.md` §5.
 *   [Q5] machine shape: 40 CUs (simd_count 80 / simd_per_cu 2 from KFD — HIP reports HALF this on this
 *        SKU, `AGENTS.md`), 64 KiB LDS/CU, 16 waves/SIMD → 64 wavefronts/CU, wavefront 32, DRAM plateau
 *        ~237-240 GB/s, and a big cache tier of ~32-48 MiB that reads ~877 GB/s. The first four fields
 *        come from `/sys/devices/virtual/kfd/kfd/topology/nodes/NODENUM/properties`, read by `--print-device`;
 *        the plateau and the tier are `devices/strix-halo-memory-bandwidth.md`.
 *
 * WHY THE READ FOOTPRINT IS 1.4 GiB AND NOT "ONE WEIGHTS-LENGTH BUFFER". The big cache tier is
 * 32-48 MiB [Q5] and the engine streams 1.46 GiB of weights per step through it, so any arm whose reads
 * fit in the tier measures the tier (~877 GB/s) and not the engine (~237 GB/s). Every matrix cell
 * therefore streams the full pinned read total — 30-60x the tier — from one buffer, and the write side
 * is spread over a 22.9 MiB region rather than a small reused one, which is the same order as the
 * engine's own 23.28 MiB write census [Q2]. A tier-isolation arm is deliberately NOT built here: it
 * would price cache residency, which is D4's term (~0.07 ms of L2 residency for gdn's state, quoted),
 * not this route's shape question, and a tier win reported as a shape win is exactly the substitution
 * this rig exists to prevent.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
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

// ---------------------------------------------------------------- machine facts [Q5]
// Read from KFD, never from HIP: `hipGetDeviceProperties` reports a HALVED CU count on this SKU
// (multiProcessorCount 20 against an architectural 40), which is recorded in AGENTS.md and which would
// put every blocks-per-CU figure out by 2x. The fallback constants are the measured ones, and the
// printed line says which source was used.
struct DevFacts {
    int cus = 40, waves_per_simd = 16, simd_per_cu = 2, wave = 32, lds_kib = 64, threads_per_cu = 2048;
    // A std::string, not a const char*: the first draft kept `path.c_str()` from a loop-local string and
    // printed a dangling pointer (visible as mojibake in `kfd_source=` on the very first device run).
    // The default is one token because every field here is parsed as `key=value` with no spaces.
    std::string src = "fallback_kfd_sysfs_unreadable";
    int hip_cus = 0, hip_warpsize = 0;
};

DevFacts dev_facts() {
    DevFacts d;
    std::vector<std::string> candidates;
    for (int n = 0; n < 6; ++n)
        candidates.push_back("/sys/devices/virtual/kfd/kfd/topology/nodes/" + std::to_string(n) +
                             "/properties");
    for (const auto& path : candidates) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) continue;
        char key[128], val[128];
        int simd_count = 0, simd_per_cu = 0, lds = 0, wave = 0, waves = 0, tpc = 0;
        while (std::fscanf(f, "%127s %127s", key, val) == 2) {
            const std::string k(key);
            const int v = std::atoi(val);
            if (k == "simd_count") simd_count = v;
            else if (k == "simd_per_cu") simd_per_cu = v;
            else if (k == "lds_size_in_kb") lds = v;
            else if (k == "wave_front_size") wave = v;
            else if (k == "max_waves_per_simd") waves = v;
        }
        std::fclose(f);
        if (simd_count > 0 && simd_per_cu > 0) {
            d.cus = simd_count / simd_per_cu;
            d.simd_per_cu = simd_per_cu;
            d.waves_per_simd = waves > 0 ? waves : d.waves_per_simd;
            if (lds > 0) d.lds_kib = lds;
            if (wave > 0) d.wave = wave;
            d.threads_per_cu = d.wave * d.waves_per_simd * d.simd_per_cu;
            d.src = path;
            break;
        }
    }
    int hc = 0, hw = 0;
    if (hipDeviceGetAttribute(&hc, hipDeviceAttributeMultiprocessorCount, 0) == hipSuccess)
        d.hip_cus = hc;
    if (hipDeviceGetAttribute(&hw, hipDeviceAttributeWarpSize, 0) == hipSuccess)
        d.hip_warpsize = hw;
    return d;
}

void print_device(const DevFacts& d) {
    std::printf(
        "DEVICE kfd_source=%s cus=%d simd_per_cu=%d wave=%d waves_per_simd=%d waves_per_cu=%d "
        "threads_per_cu=%d lds_kib=%d\n",
        d.src.c_str(), d.cus, d.simd_per_cu, d.wave, d.waves_per_simd,
        d.waves_per_simd * d.simd_per_cu, d.threads_per_cu, d.lds_kib);
    std::printf(
        "DEVICE hip_multiprocessor_count=%d hip_warp_size=%d  NOTE: HIP reports HALF the CU count on "
        "this SKU (AGENTS.md); every blocks-per-CU number here uses the KFD value (%d), not HIP's (%d)\n",
        d.hip_cus, d.hip_warpsize, d.cus, d.hip_cus);
}

// ---------------------------------------------------------------- the pinned byte totals [Q2]
// The census's own numbers, converted once, in one place, with the deviation asserted rather than
// waved at. The census is in MB (1e6 B) and the rig works in 8-byte elements.
constexpr double kCensusReadMB = 1532.311;
constexpr double kCensusWriteMB = 24.409;
inline long long census_el_read() { return (long long)llround(kCensusReadMB * 1e6 / 8.0); }
inline long long census_el_write() { return (long long)llround(kCensusWriteMB * 1e6 / 8.0); }

// The totals are the census rounded to the nearest multiple of the LCM of every planned block total B,
// times 64 so that the write total (= read/64) is ALSO divisible by every B. Rounding is +0.18 % on
// the read and −1.7 % on the write, both printed on every row and asserted in plan_invariants().
constexpr long long kTotRead = 191889408;          // 1_532_362_752 B = 1461.38 MiB
constexpr long long kTotWrite = kTotRead / 64;     // 23,986,176 B = 22.88 MiB
constexpr int kThreads = 256;                      // the engine's mlp launch width: 8 rows x 32 lanes

double tot_read_mib() { return (double)kTotRead * 8.0 / 1048576.0; }
double tot_write_mib() { return (double)kTotWrite * 8.0 / 1048576.0; }
double dev_read_vs_census() { return (double)kTotRead / (double)census_el_read() - 1.0; }
double dev_write_vs_census() { return (double)kTotWrite / (double)census_el_write() - 1.0; }

// ---------------------------------------------------------------- the byte witness (arm 1)
constexpr unsigned long long kWordK = 0x9E3779B97F4A7C15ull;
constexpr unsigned long long kWordInv = 0xF1DE83E19937733Dull;   // K*kWordInv == 1 mod 2^64
constexpr unsigned long long kSentW = 0xFFFFFFFFFFFFFFFFull;
constexpr int kGUARD = 64;
__host__ __device__ inline unsigned long long word_at(long long p) {
    return kWordK * (2ull * (unsigned long long)p + 1ull);
}
inline unsigned long long range_sum(long long p0, long long hi) {
    const unsigned long long h = (unsigned long long)hi, a = (unsigned long long)p0;
    return kWordK * (h * h - a * a);
}

struct MRec {
    unsigned long long rsum, wsum;
    unsigned rcount, wcount, tag, pad;
};
static_assert(sizeof(MRec) == 32, "record size is part of the region arithmetic");
constexpr unsigned kRecSent = 0xFFFFFFFFu;

enum {
    MN_NONE = 0,
    MN_BLOCK_SHORT,     // grid one block short on one launch: that block never ran
    MN_RANGE_SHORT,     // one block reads one element fewer: records move, NO byte moves
    MN_ROW_DOUBLE,      // one block writes twice its booked write range: a stray write
    MN_ALIAS,           // one launch writes its neighbour's range at its own offsets
    MN_LIE_WRITE,       // accumulate and report, never store: only the content check sees it
    MN_TAG_CONST,       // a constant tag in both values and records: self-consistent
    MN_PERMUTE_READS,   // rotate one block's own read set: provably invisible to a sum (blind)
    MN_COUNT
};
const char* kMNName[MN_COUNT] = {"none",          "block_short", "range_short", "row_double",
                                 "alias_write", "lie_write",    "tag_const",   "permute_reads"};

enum NegKind {
    NG_NONE = 0,
    NG_LIE_WRITE = 2,      // keep the numbering stable with MN_* where the kernel cares
    NG_PERMUTE_READS = 3,
    NG_RANGE_SHORT = 4,
    NG_ROW_DOUBLE = 5,
    NG_TAG_CONST = 6,
};
int neg_kernel_id(int neg) {
    switch (neg) {
        case MN_LIE_WRITE: return NG_LIE_WRITE;
        case MN_PERMUTE_READS: return NG_PERMUTE_READS;
        case MN_RANGE_SHORT: return NG_RANGE_SHORT;
        case MN_ROW_DOUBLE: return NG_ROW_DOUBLE;
        case MN_TAG_CONST: return NG_TAG_CONST;
        default: return NG_NONE;
    }
}

__device__ inline unsigned long long warp_red_u64(unsigned long long v) {
    // Offsets start at warpSize/2, NOT at 32: `__shfl_down_sync` returns the lane's OWN value when
    // lane+delta passes the segment, so starting at 32 on a wave32 part doubles every lane on the first
    // step. (The mask is 64-bit even on a wave32 build; the mask and the offsets are separate facts and
    // conflating them is how this rig's first device run reported every record at exactly 2x.)
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffffffffffull, v, (unsigned)off);
    return v;
}

// One launch of a matrix cell. Block gb owns a contiguous U_R-element read range and U_W write range.
// `rec_lo` is the launch's first record slot AND, because records are 1:1 with blocks, the launch's
// first GLOBAL block index; the injection names a global block index, so `inj` is exact for one block
// of one launch and no other.
extern "C" __global__ void ws_touch(const unsigned long long* __restrict__ src,
                                   unsigned long long* __restrict__ dst, long r_lo, long w_lo,
                                   long u_r, long u_w, unsigned tag, MRec* __restrict__ recs,
                                   long rec_lo, int neg, long inj_glob)
{
    const int tid = (int)threadIdx.x, nth = (int)blockDim.x;
    const bool inj = ((long)(rec_lo + blockIdx.x) == inj_glob);
    const long lo = r_lo + (long)blockIdx.x * u_r, hi = lo + u_r;
    const long wlo = w_lo + (long)blockIdx.x * u_w, whi = wlo + u_w;
    const bool lie = (neg == NG_LIE_WRITE && inj);
    const bool rot = (neg == NG_PERMUTE_READS && inj);
    const unsigned mult = (neg == NG_ROW_DOUBLE && inj) ? 2u : 1u;
    const bool short_r = (neg == NG_RANGE_SHORT && inj);
    const unsigned etag = (neg == NG_TAG_CONST) ? 7u : tag;

    unsigned long long rsum = 0, wsum = 0, rcnt = 0, wcnt = 0;
    const long rhi = (short_r && hi > lo) ? hi - 1 : hi;
    for (long i = lo + tid; i < rhi; i += nth) {
        const long idx = rot ? lo + ((i - lo + 1) % (rhi - lo)) : i;
        rsum += src[idx];
        rcnt += 1;
    }
    // Values are indexed by the offset from the LAUNCH's booked write base (the argument), not from the
    // block's base: a block's own chunk is at launch offsets [b*u_w, (b+1)*u_w), and an expectation that
    // mixed the two origins would make the record check and the content check disagree with each other
    // while each agreed with itself. An aliased launch indexes from the base it was handed, so it stores
    // exactly what it would have stored at home and only a content comparison can see the alias.
    //
    // MN_ROW_DOUBLE reports TWICE the booked write count while storing exactly the booked bytes at
    // exactly the booked addresses. That is deliberate: the control exists to prove the record's
    // write-COUNT is load-bearing (a rig that compared content only would pass it), and writing to a
    // second address instead would either race a neighbour block or need a second checked region, and a
    // flaky control is worse than no control.
    for (long i = wlo + tid; i < whi; i += nth) {
        const unsigned long long v = word_at(i - w_lo) + etag;
        wsum += v * mult;
        wcnt += mult;
        if (!lie) dst[i] = v;
    }

    __shared__ unsigned long long sh[4][32];
    const int lane = tid % warpSize, wid = tid / warpSize;
    rsum = warp_red_u64(rsum); wsum = warp_red_u64(wsum);
    rcnt = warp_red_u64(rcnt); wcnt = warp_red_u64(wcnt);
    if (lane == 0) {
        sh[0][wid] = rsum; sh[1][wid] = wsum; sh[2][wid] = rcnt; sh[3][wid] = wcnt;
    }
    __syncthreads();
    if (tid == 0) {
        unsigned long long a = 0, b = 0, c = 0, d = 0;
        const int nw = (nth + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) { a += sh[0][w]; b += sh[1][w]; c += sh[2][w]; d += sh[3][w]; }
        MRec r;
        r.rsum = a; r.wsum = b;
        r.rcount = (unsigned)c; r.wcount = (unsigned)d;
        r.tag = etag; r.pad = 0u;
        recs[rec_lo + blockIdx.x] = r;
    }
}

extern "C" __global__ void fill_word(unsigned long long* src, long n)
{
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) src[i] = word_at(i);
}

// ---------------------------------------------------------------- regions + guards
enum Reg { R_SRC = 0, R_DST, R_REC, R_W, R_X, R_OUT, R_PART, NREG };
const char* kRegName[NREG] = {"src", "dst", "rec", "weights", "act", "out", "partials"};
struct Region {
    void* base = nullptr;
    char* p = nullptr;
    size_t bytes = 0;
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
void reg_fill(int id, unsigned byte) {
    if (g_reg[id].base) CK(hipMemset(g_reg[id].p, byte, g_reg[id].bytes));
}
void alloc_free_all() { for (int id = 0; id < NREG; ++id) reg_free(id); }

// Best-of-`iters` whole-step walls inside best-of-`reps`, steady_clock (hipEventElapsedTime returns
// exactly 0 below ~1 ms on this part). Forward-declared so the case runners, which print the RESULT
// line, can call it; the definition is beside main().
double timed_steps(const std::function<void()>& step, int reps, int iters, double* med, double* worst);

// ---------------------------------------------------------------- the matrix cell
struct MCfg {
    int L = 128;             // launches per step
    int G = 64;              // blocks per launch (workgroups per launch — the A12 field)
    long long u_r = 0, u_w = 0;
    int neg = MN_NONE;
    int time_flag = 0, reps = 3, iters = 5;
    std::string arm;
};

const char* mrefuse(const MCfg& c) {
    if (c.L < 1) return "--launches must be >= 1 for a matrix cell (there is no no-launch cell: a "
                        "window that launches nothing prints NO-TIMING, and the arm is --down none)";
    if (c.G < 1) return "--grid must be >= 1: a launch with no workgroups is not a launch the engine emits";
    const long long B = (long long)c.L * c.G;
    if (kTotRead % B != 0) {
        static std::string msg;
        msg = "block total L*G=" + std::to_string(B) + " does not divide the pinned read total " +
              std::to_string(kTotRead) + " exactly; every cell must touch the SAME bytes, so a cell "
              "that needs rounding is refused rather than approximated. Run --print-plan for the "
              "reachable cells.";
        return msg.c_str();
    }
    if (kTotWrite % B != 0) return "block total does not divide the pinned write total exactly";
    if ((long long)c.G * kThreads > kTotRead) return "one launch would have more blocks than bytes";
    if (c.G > 65535) return "--grid above 65535 is the 2-D-grid arm, which this rig does not emit";
    return nullptr;
}

struct MPlan {
    long long u_r = 0, u_w = 0, blocks = 0;
    long n_rec = 0;
    long long read_bytes = 0, write_bytes = 0;
};

void mplan_build(const MCfg& c, MPlan& p) {
    p.blocks = (long long)c.L * c.G;
    p.u_r = kTotRead / p.blocks;
    p.u_w = kTotWrite / p.blocks;
    p.n_rec = p.blocks;
    p.read_bytes = kTotRead * 8;
    p.write_bytes = kTotWrite * 8;
}

const char* mplan_invariants(const MCfg& c, const MPlan& p, const DevFacts& d) {
    if ((long long)c.L * c.G != p.blocks) return "block total disagrees with L*G";
    if (p.u_r * p.blocks != kTotRead) return "the per-block read work does not reconstruct the pinned total";
    if (p.u_w * p.blocks != kTotWrite) return "the per-block write work does not reconstruct the pinned total";
    if (p.u_r * 8 % 64 != 0) return "a block's read range is not a whole number of cache lines";
    if (std::fabs(dev_read_vs_census()) > 0.005) return "pinned read total drifted >0.5 % from the census";
    if (std::fabs(dev_write_vs_census()) > 0.025) return "pinned write total drifted >2.5 % from the census";
    if (p.u_r / (long long)kThreads < 1) return "a block has fewer bytes than threads";
    // 256 threads = 8 wavefronts at a 32-lane wavefront; the residency ceiling is the wavefront one
    // unless LDS or registers bind first (this kernel declares no shared memory).
    const long long waves = p.blocks * (kThreads / d.wave);
    if (waves < (long long)d.cus) return "fewer wavefronts in the cell than there are CUs";
    return nullptr;
}

std::string marm_key(const MCfg& c) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "mat|%d|%d|b%lld", c.L, c.G, (long long)c.L * c.G);
    return buf;
}

// ---------------------------------------------------------------- arm 1: run + verify
struct Counts { long want = 0, got = 0, syncs = 0; };

long m_inj_glob(const MCfg& c, const MPlan& p) {
    // The injected block is the FIRST block of the middle launch: one block's bytes are the must-catch
    // set, everything else is the must-not-move set, and both are derived, never typed.
    const long linj = c.L / 2;
    return linj * (long)c.G;
}
long m_inj_launch_of(const MCfg& c) { return c.L / 2; }

Counts mrun(const MCfg& c, const MPlan& p) {
    Counts ct;
    ct.want = c.L;
    const int nk = neg_kernel_id(c.neg);
    const long inj_glob = (c.neg == MN_NONE) ? -1 : m_inj_glob(c, p);
    const long linj = m_inj_launch_of(c);
    for (int i = 0; i < c.L; ++i) {
        int grid = c.G;
        if (c.neg == MN_BLOCK_SHORT && i == linj && grid > 1) grid -= 1;
        long w_lo = (long)((long long)i * c.G * p.u_w);
        if (c.neg == MN_ALIAS && i == linj && i > 0)
            w_lo = (long)((long long)(i - 1) * c.G * p.u_w);   // two writers, one address
        ws_touch<<<grid, kThreads>>>(
            reinterpret_cast<const unsigned long long*>(g_reg[R_SRC].p),
            reinterpret_cast<unsigned long long*>(g_reg[R_DST].p),
            (long)((long long)i * c.G * p.u_r), w_lo, (long)p.u_r, (long)p.u_w,
            (unsigned)(i + 1), reinterpret_cast<MRec*>(g_reg[R_REC].p), (long)((long long)i * c.G),
            nk, inj_glob);
        if (hipGetLastError() == hipSuccess) ++ct.got;
    }
    CK(hipDeviceSynchronize());
    ct.syncs = 1;
    return ct;
}

// The three-way class, DERIVED. DONT_CARE = -1 (a positive run: everything must match exactly),
// MUST_SAME = 0, MUST_DIFF = 1.
enum : int { DONT_CARE = -1, MUST_SAME = 0, MUST_DIFF = 1 };

int mrec_class(const MCfg& c, long linj, long linj_n, long li, long b) {
    switch (c.neg) {
        case MN_BLOCK_SHORT:
            return (li == linj && b == linj_n - 1) ? MUST_DIFF : MUST_SAME;
        case MN_RANGE_SHORT:
        case MN_ROW_DOUBLE:
            return (li == linj && b == 0) ? MUST_DIFF : MUST_SAME;
        case MN_ALIAS:
        case MN_LIE_WRITE:
            // self-consistent records: only the byte-content check can see either one
            return MUST_SAME;
        case MN_TAG_CONST:
            return li == 6 ? MUST_SAME : MUST_DIFF;      // launch 6 carries tag 7
        case MN_PERMUTE_READS:
            return MUST_SAME;                            // blind by proof
        default:
            return DONT_CARE;
    }
}

int mblock_class(const MCfg& c, long linj, long linj_n, long li, long b) {
    // Which blocks' WRITTEN bytes must move.
    switch (c.neg) {
        case MN_BLOCK_SHORT:
            return (li == linj && b == linj_n - 1) ? MUST_DIFF : MUST_SAME;
        case MN_RANGE_SHORT:
            return MUST_SAME;              // a short READ moves no written byte: the sum is load-bearing
        case MN_ROW_DOUBLE:
            // the doubled block stored exactly the booked bytes at the booked addresses; only its
            // record can show the doubling, so NO written byte may differ
            return MUST_SAME;
        case MN_ALIAS:
            // the aliased launch never got its own values; its predecessor's range was overwritten
            return (li == linj) ? MUST_DIFF : (li == linj - 1 ? MUST_DIFF : MUST_SAME);
        case MN_LIE_WRITE:
            return (li == linj && b == 0) ? MUST_DIFF : MUST_SAME;
        case MN_TAG_CONST:
            return li == 6 ? MUST_SAME : MUST_DIFF;
        case MN_PERMUTE_READS:
            return MUST_SAME;
        default:
            return DONT_CARE;
    }
}

struct Chk {
    int ok = 1;
    long rec_want = 0, rec_bad = 0, rec_missing = 0;
    long w_want = 0, w_bad = 0, w_missing = 0, stray = 0, guard = 0;
    long must_total = 0, must_caught = 0, same_total = 0, same_violated = 0;
    long read_bytes_checked = 0, write_bytes_checked = 0;
    char why[208] = "";
    char diag[320] = "";              // the first mismatch, got-vs-expected, field by field
    void diag_rec(const MRec& g, const MRec& e) {
        if (diag[0]) return;
        std::snprintf(diag, sizeof diag,
                      "rec0 got(rs=%llu rc=%u ws=%llu wc=%u tg=%u) exp(rs=%llu rc=%u ws=%llu wc=%u tg=%u)",
                      g.rsum, g.rcount, g.wsum, g.wcount, g.tag, e.rsum, e.rcount, e.wsum, e.wcount,
                      e.tag);
    }
    void diag_dst(long e, unsigned long long got, unsigned long long exp) {
        if (diag[0]) return;
        std::snprintf(diag, sizeof diag, "dst0 e=%ld got=%llu exp=%llu", e, got, exp);
    }
    int nbad = 0;
    char bad[3][80] = {};
    void note(const char* what, long at) {
        if (nbad < 3) { std::snprintf(bad[nbad], sizeof bad[nbad], "%s@%ld", what, at); ++nbad; }
    }
    std::string where() const {
        std::string out = "first_bad=";
        for (int i = 0; i < nbad; ++i) { out += bad[i]; out += (i + 1 < nbad) ? "," : ""; }
        return nbad ? out : out + "none";
    }
};

void mverify(const MCfg& c, const MPlan& p, Chk& chk) {
    const long linj = m_inj_launch_of(c), G = c.G;
    std::vector<MRec> recs((size_t)p.n_rec);
    CK(hipMemcpy(recs.data(), g_reg[R_REC].p, (size_t)p.n_rec * sizeof(MRec), hipMemcpyDeviceToHost));
    // ---- records: exact read checksum per block, predicted from the plan, never from the kernel
    for (long i = 0; i < c.L; ++i) {
        const long r_lo = i * G * p.u_r;
        const long w_lo = i * G * p.u_w;
        for (long b = 0; b < G; ++b) {
            const long lo = r_lo + b * p.u_r, hi = lo + p.u_r;
            const MRec& got = recs[(size_t)(i * G + b)];
            ++chk.rec_want;
            const bool sent = (got.rcount == kRecSent && got.wcount == kRecSent);
            // The block's own chunk sits at LAUNCH offsets [b*u_w, (b+1)*u_w) and the stored value is
            // indexed from the launch base, so the record's expected write sum is that sub-range, not
            // [0,u_w). Getting this wrong makes the record check and the content check disagree with
            // each other while each agrees with itself, which is the worst possible failure: one of the
            // two witnesses is always green.
            const unsigned long long wexp =
                range_sum(b * p.u_w, (b + 1) * p.u_w) +
                (unsigned long long)(i + 1) * (unsigned long long)p.u_w;
            const bool diff = sent || got.rsum != range_sum(lo, hi) ||
                              got.rcount != (unsigned)p.u_r || got.wsum != wexp ||
                              got.wcount != (unsigned)p.u_w || got.tag != (unsigned)(i + 1);
            const int cls = mrec_class(c, linj, G, i, b);
            if (cls == MUST_DIFF) { ++chk.must_total; if (diff) ++chk.must_caught; else chk.note("rec", i * G + b); }
            else if (cls == MUST_SAME) { ++chk.same_total; if (diff) { ++chk.same_violated; chk.note("rec", i * G + b); } }
            if (diff) {
                MRec exp; exp.rsum = range_sum(lo, hi); exp.rcount = (unsigned)p.u_r;
                exp.wsum = wexp; exp.wcount = (unsigned)p.u_w; exp.tag = (unsigned)(i + 1); exp.pad = 0;
                chk.diag_rec(got, exp);
            }
            if (sent) { ++chk.rec_missing; chk.note("rec_unwritten", i * G + b); }
            else if (diff) { ++chk.rec_bad; chk.note("rec_wrong", i * G + b); }
            else { chk.read_bytes_checked += p.u_r * 8; chk.write_bytes_checked += p.u_w * 8; }
        }
    }
    // ---- written bytes: write-landing read-back, every byte of the booked region, plus strays
    std::vector<unsigned long long> dst((size_t)kTotWrite);
    CK(hipMemcpy(dst.data(), g_reg[R_DST].p, (size_t)kTotWrite * 8, hipMemcpyDeviceToHost));
    const long per_launch_w = G * p.u_w;
    for (long e = 0; e < kTotWrite; ++e) {
        const long li = e / per_launch_w;
        const long b = (e - li * per_launch_w) / p.u_w;
        // The expectation is the SHIPPED plan's value, always. Predicting the injected value instead
        // makes the control invisible: the first draft of this line carried `MN_TAG_CONST ? 7 : li+1`
        // into the expectation and tag_const then reported must_catch_missed over 360k bytes while the
        // injection was working perfectly.
        const unsigned long long expv = word_at(e - li * per_launch_w) +
                                        (unsigned long long)(li + 1);
        const bool diff = (dst[(size_t)e] != expv);
        const bool unwritten = (dst[(size_t)e] == kSentW);
        ++chk.w_want;
        const int cls = mblock_class(c, linj, G, li, b);
        if (cls == MUST_DIFF) { ++chk.must_total; if (diff) ++chk.must_caught; else chk.note("dst", e); }
        else if (cls == MUST_SAME) { ++chk.same_total; if (diff) { ++chk.same_violated; chk.note("dst", e); } }
        if (diff) {
            if (unwritten) ++chk.w_missing;
            ++chk.w_bad;
            chk.diag_dst(e, dst[(size_t)e], expv);
            chk.note("dst_wrong", e);
        }
    }
    chk.guard = check_guards();
    if (chk.rec_want + chk.w_want == 0) {
        chk.ok = 0;
        std::snprintf(chk.why, sizeof chk.why, "FAIL_nothing_expected");
        return;
    }
    if (c.neg == MN_NONE) {
        chk.ok = (chk.rec_missing == 0 && chk.rec_bad == 0 && chk.w_bad == 0 && chk.guard == 0 &&
                  chk.read_bytes_checked == kTotRead * 8 && chk.write_bytes_checked == kTotWrite * 8);
        if (!chk.ok)
            std::snprintf(chk.why, sizeof chk.why,
                          "rec_missing=%ld rec_bad=%ld dst_bad=%ld guard=%ld read_checked=%lld/%lld "
                          "write_checked=%lld/%lld",
                          chk.rec_missing, chk.rec_bad, chk.w_bad, chk.guard,
                          (long long)chk.read_bytes_checked, (long long)(kTotRead * 8),
                          (long long)chk.write_bytes_checked, (long long)(kTotWrite * 8));
    } else if (c.neg == MN_PERMUTE_READS) {
        chk.ok = (chk.must_total == 0 && chk.same_violated == 0 && chk.rec_missing == 0 &&
                  chk.rec_bad == 0 && chk.w_bad == 0 && chk.guard == 0);
        std::snprintf(chk.why, sizeof chk.why, "blind_must_catch=%ld same=%ld same_violated=%ld",
                      chk.must_total, chk.same_total, chk.same_violated);
    } else {
        chk.ok = (chk.must_total > 0 && chk.must_caught == chk.must_total &&
                  chk.same_violated == 0 && chk.guard == 0);
        if (chk.must_total == 0) std::snprintf(chk.why, sizeof chk.why, "BLIND-BY-GEOMETRY_empty_must_catch");
        else if (chk.must_caught != chk.must_total)
            std::snprintf(chk.why, sizeof chk.why, "must_catch_missed=%ld/%ld same_violated=%ld",
                          chk.must_total - chk.must_caught, chk.must_total, chk.same_violated);
        else if (chk.same_violated) std::snprintf(chk.why, sizeof chk.why, "must_not_move_violated=%ld", chk.same_violated);
        else if (chk.guard) std::snprintf(chk.why, sizeof chk.why, "guard_bytes_broken=%ld", chk.guard);
    }
}

struct MNegSets { long must = 0, same = 0, must_rec = 0, must_w = 0; int blind = 0; };
MNegSets mneg_derive(const MCfg& c, const MPlan& /*p*/) {
    MNegSets ns;
    const long linj = m_inj_launch_of(c), G = c.G;
    for (long i = 0; i < c.L; ++i)
        for (long b = 0; b < G; ++b) {
            const int r = mrec_class(c, linj, G, i, b);
            ns.must += (r == MUST_DIFF); ns.same += (r == MUST_SAME); ns.must_rec += (r == MUST_DIFF);
        }
    for (long i = 0; i < c.L; ++i)
        for (long b = 0; b < G; ++b) {
            const int w = mblock_class(c, linj, G, i, b);
            ns.must += (w == MUST_DIFF); ns.same += (w == MUST_SAME); ns.must_w += (w == MUST_DIFF);
        }
    ns.blind = (c.neg == MN_PERMUTE_READS);
    return ns;
}

// ================================================================ ARM 2: the one-row down_proj shape
// The engine's shape, reproduced from source (see the file header): rows = kHidden = 1024,
// K = kIntermediate = 3584, ROWS_PER_BLOCK = 8, warp 32 -> 128 blocks x 256 threads, and each output
// row is one warp walking 448 `uint4` words with 14 dependent accumulations per lane.
constexpr int kDownRows = 1024;
constexpr int kDownK = 3584;
constexpr int kDownWords = kDownK / 8;        // 448 uint4 words per row
constexpr int kDownRpb = 8;                   // ROWS_PER_BLOCK

// Transcribed from prototype/mlp_gemv/mlp_gemv_kernels.h:46,56,62 (bf / fb / reduce_row) so the
// arithmetic has ONE spelling in this rig and it is the shipped spelling.
__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }
__device__ __forceinline__ float reduce_row(float acc) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, (unsigned)o);
    return acc;
}

// Down-arm injections. Each is localised to ONE block (and within it, for the word/row cases, one warp),
// so the must-catch set is that block's rows and the must-not-move set is every other row: derived,
// non-empty on both sides, and impossible for a control to rot into "moves everything".
enum {
    DN_NONE = 0,
    DN_TRUNC,       // intra only: one warp computes a quarter of the words it was booked -> its own row
    DN_ROW_ALIAS,      // intra only: one block stores its rows shifted one slot -> unwritten + stolen
    DN_POISON,         // intra and two-pass: a partial slot holds a poison value instead of a partial
    DN_SLICE_DROP,     // two-pass only: one block writes NO partial (its slots keep the sentinel)
    DN_COUNT
};
const char* kDNName[DN_COUNT] = {"none", "trunc", "row_alias", "poison", "slice_drop"};

// The shipped one-token down_proj, character for character from mlp_gemv_kernels.h:191-211 with
// neg = NEG_NONE (the F32 act that ships today; --down-act bf16 builds the cast-fold twin).
template <int IN, bool ACT_BF16>
__global__ void down_ship(const hip_bfloat16* __restrict__ W, const void* __restrict__ x,
                          hip_bfloat16* __restrict__ out, int rows, int rpb)
{
    __shared__ float xs_f[IN];
    __shared__ hip_bfloat16 xs_b[IN];
    if (ACT_BF16) {
        const hip_bfloat16* xb = (const hip_bfloat16*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs_b[i] = xb[i];
        __syncthreads();
    } else {
        const float* xf = (const float*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs_f[i] = xf[i];
        __syncthreads();
    }

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
    float acc = 0.f;
    const int words = IN / 8;
#pragma unroll
    for (int v = lane; v < words; v += ws) {
        uint4 u = wrow[v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        float xv[8];
        if (ACT_BF16) {
            const hip_bfloat16* xp = &xs_b[v * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[j] = bf(xp[j]);
        } else {
            const float* xp = &xs_f[v * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[j] = xp[j];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += bf(p[j]) * xv[j];
    }
    acc = reduce_row(acc);
    if (lane == 0) out[row] = fb(acc);
}

// K split INSIDE the launch: WPR warps share one row, each owning the contiguous word slice
// [w*(words/WPR), (w+1)*(words/WPR)) walked with the same strided form. WPR == 1 degenerates to the
// shipped walk, the shipped ladder and the shipped store, which is what makes the degenerate case a
// witness rather than a hope. The cross-warp combine is a real block reduction through shared memory
// (partial slots written by each warp's lane 0, summed in slice order by one thread) — never a
// per-thread store to one slot (the tools/mixed_bw bug, AGENTS.md).
template <int IN, int WPR, bool ACT_BF16>
__global__ void down_intra(const hip_bfloat16* __restrict__ W, const void* __restrict__ x,
                           hip_bfloat16* __restrict__ out, int rows, int neg, long inj_blk)
{
    constexpr int BLOCK_ROWS = (256 / 32) / WPR;      // warps per block is 8 at a 32-lane wavefront
    __shared__ float xs_f[IN];
    __shared__ hip_bfloat16 xs_b[IN];
    __shared__ float part[BLOCK_ROWS][WPR];
    if (ACT_BF16) {
        const hip_bfloat16* xb = (const hip_bfloat16*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs_b[i] = xb[i];
        __syncthreads();
    } else {
        const float* xf = (const float*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs_f[i] = xf[i];
        __syncthreads();
    }

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int wl = threadIdx.x / ws;                    // warp index inside the block
    const int rl = wl / WPR;                            // row slot inside the block
    const int slice = wl % WPR;                          // which K slice this warp owns
    const int row = blockIdx.x * BLOCK_ROWS + rl;
    if (row >= rows) return;

    const int sw = IN / 8 / WPR;                         // words in this slice
    const int base = slice * sw;
    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
    const bool inj = ((long)blockIdx.x == inj_blk);
    // One warp of one block, so the must-catch set is exactly one row (dn_derive says which one). Two
    // things this had to learn the hard way: (a) the lane that owns a strided walk's last word is
    // (sw-1) % warpSize, so a `lane == 0` gate drops nothing at all; (b) dropping ONE word of a 448-word
    // dot moves the row by ~0.2 % of itself, which is under a BF16 output quantum (1/128 of an octave),
    // so the control is invisible to the only output there is and reports NOT-CAUGHT while the injection
    // works. Truncating the slice to a quarter is above the quantum for every row but a total
    // cancellation, and a control must fire on the geometry, not on luck.
    int upto = sw;
    if (neg == DN_TRUNC && inj && wl == 0 && sw >= 4) upto = sw / 4;

    float acc = 0.f;
#pragma unroll
    for (int v = lane; v < upto; v += ws) {
        uint4 u = wrow[base + v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        float xv[8];
        if (ACT_BF16) {
            const hip_bfloat16* xp = &xs_b[(base + v) * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[j] = bf(xp[j]);
        } else {
            const float* xp = &xs_f[(base + v) * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[j] = xp[j];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += bf(p[j]) * xv[j];
    }
    acc = reduce_row(acc);
    if (neg == DN_POISON && inj && slice == 0) acc = 1234.5f;   // same slot, wrong value, no race
    if (lane == 0) part[rl][slice] = acc;
    __syncthreads();
    // The combiner is each row's slice-0 warp (not warp 0 of the block, which would combine one row and
    // leave the block's other rows unwritten).
    if (slice == 0 && lane == 0) {
        float s = part[rl][0];
#pragma unroll
        for (int w = 1; w < WPR; ++w) s += part[rl][w];
        const int dst = (neg == DN_ROW_ALIAS && inj) ? (row ^ 1) : row;   // in bounds, no double writer
        out[dst] = fb(s);
    }
}

// K split ACROSS the reduction: pass 1 writes F32 partials[slice][row] (one warp per (row, slice), so
// a block is 8 rows x 1 slice — the same work distribution as down_intra at the same split), pass 2
// sums the slices in ascending order and rounds once. Two launches; a real reduction, not atomics.
template <int IN, int S, bool ACT_BF16>
__global__ void down_part(const hip_bfloat16* __restrict__ W, const void* __restrict__ x,
                          float* __restrict__ part, int rows, int neg, long inj_blk)
{
    constexpr int BLOCK_ROWS = 256 / 32;
    __shared__ float xs_f[IN];
    __shared__ hip_bfloat16 xs_b[IN];
    if (ACT_BF16) {
        const hip_bfloat16* xb = (const hip_bfloat16*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs_b[i] = xb[i];
        __syncthreads();
    } else {
        const float* xf = (const float*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs_f[i] = xf[i];
        __syncthreads();
    }
    const int slice = blockIdx.y;
    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * BLOCK_ROWS + threadIdx.x / ws;
    if (row >= rows) return;
    const bool inj = ((long)blockIdx.x == inj_blk && slice == 0);
    if (neg == DN_SLICE_DROP && inj) return;                    // writes NO partial at all

    const int sw = IN / 8 / S;
    const int base = slice * sw;
    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
    float acc = 0.f;
#pragma unroll
    for (int v = lane; v < sw; v += ws) {
        uint4 u = wrow[base + v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        float xv[8];
        if (ACT_BF16) {
            const hip_bfloat16* xp = &xs_b[(base + v) * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[j] = bf(xp[j]);
        } else {
            const float* xp = &xs_f[(base + v) * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[j] = xp[j];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += bf(p[j]) * xv[j];
    }
    acc = reduce_row(acc);
    if (neg == DN_POISON && inj && slice == 0) { acc = 1234.5f; }   // same slot, wrong value, no race
    if (lane == 0) part[(size_t)slice * rows + row] = acc;
}

template <int S>
__global__ void down_reduce(const float* __restrict__ part, hip_bfloat16* __restrict__ out, int rows)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float s = part[row];
#pragma unroll
    for (int sl = 1; sl < S; ++sl) s += part[(size_t)sl * rows + row];
    out[row] = fb(s);
}

// ---------------------------------------------------------------- down-arm config + geometry
enum DForm { DF_SHIP = 0, DF_INTRA, DF_TWO };
struct DCfg {
    int form = DF_SHIP;
    int wpr = 1;                 // DF_INTRA: warps per row (the K split inside the launch)
    int spl = 1;                 // DF_TWO: slices (the K split across the reduction)
    int act_bf16 = 0;            // 0 = the F32 act that ships today; 1 = the cast-fold twin
    int neg = DN_NONE;
    int time_flag = 0, reps = 3, iters = 5;
    std::string arm;
};

struct DGeom {
    int launches = 1, grid_x = 128, grid_y = 1, blocks_total = 128, block_rows = kDownRpb;
    int slices = 1, warps_total = kDownRows;
    long long stage_bytes = 0, weight_bytes = 0, part_bytes = 0, out_bytes = 0;
    int inj_block = 0, inj_row = 0, inj_rows = 1;      // the injection target, derived
};

int dgeom_spl(const DCfg& c) { return c.form == DF_TWO ? c.spl : c.wpr; }

DGeom dgeom(const DCfg& c) {
    DGeom g;
    const int slices = dgeom_spl(c);
    g.slices = slices;
    g.weight_bytes = (long long)kDownRows * kDownK * 2;
    g.out_bytes = (long long)kDownRows * 2;
    if (c.form == DF_SHIP) {
        g.block_rows = kDownRpb; g.grid_x = kDownRows / kDownRpb; g.grid_y = 1;
    } else if (c.form == DF_INTRA) {
        g.block_rows = kDownRpb / c.wpr; g.grid_x = kDownRows / g.block_rows; g.grid_y = 1;
    } else {
        g.block_rows = kDownRpb; g.grid_x = kDownRows / kDownRpb; g.grid_y = c.spl;
    }
    g.blocks_total = g.grid_x * g.grid_y + (c.form == DF_TWO ? (kDownRows + 255) / 256 : 0);
    g.launches = (c.form == DF_TWO) ? 2 : 1;
    g.warps_total = g.grid_x * g.grid_y * (kThreads / 32) +
                    (c.form == DF_TWO ? ((kDownRows + 255) / 256) * (kThreads / 32) : 0);
    // Every block stages the whole activation row into shared memory; that is demanded traffic the
    // shipped shape pays 128 times and a split shape pays more, so it is booked, not hidden.
    g.stage_bytes = (long long)g.grid_x * g.grid_y * kDownK * (c.act_bf16 ? 2 : 4);
    g.part_bytes = (c.form == DF_TWO) ? (long long)c.spl * kDownRows * 4 * 2 : 0;  // write + read back
    // The injection is the middle block of the first grid row: one block's rows are the must-catch set.
    g.inj_block = (c.form == DF_TWO) ? (g.grid_x / 2) : (g.grid_x / 2);
    if (g.inj_block >= g.grid_x) g.inj_block = g.grid_x - 1;
    g.inj_row = g.inj_block * g.block_rows;           // the slice-0, lane-0 row of that block
    g.inj_rows = (c.form == DF_TWO) ? kDownRpb : g.block_rows;
    return g;
}

const char* drefuse(const DCfg& c) {
    if (c.form == DF_SHIP && c.neg != DN_NONE)
        return "the frozen form takes no injection: the shipped spelling is only ever run as the "
               "reference, and a control must not be able to move the ruler it is judged against "
               "(AGENTS: the frozen form gets its own spelling).";
    if (c.form == DF_INTRA) {
        if (c.wpr < 1 || c.wpr > 8) return "--wpr must be in 1..8 (8 warps per block at 256 threads)";
        if (kDownRpb % c.wpr) return "--wpr must divide ROWS_PER_BLOCK=8 so every row slot is whole";
        if (kDownWords % c.wpr) return "--wpr must divide the 448 words of K so no slice is ragged";
        if (kDownWords / c.wpr < 32) return "--wpr leaves a warp with fewer words than lanes";
    }
    if (c.form == DF_TWO) {
        const int ok_spl[5] = {2, 4, 7, 8, 14};
        bool good = false;
        for (int s : ok_spl) if (s == c.spl) good = true;
        if (!good) return "--split must be one of 2,4,7,8,14: the divisors of the 448 words of K that "
                          "leave at least 32 words per slice, and the only ones instantiated here";
        if (c.neg == DN_TRUNC || c.neg == DN_ROW_ALIAS)
            return "that control's injection point lives in down_intra. On the two-pass form the word "
                   "walk and the store index are witnessed differently: the store index by the frozen "
                   "form reproducing the mirror bit for bit, which is stronger than an injection, and "
                   "the walk by DN_POISON and DN_SLICE_DROP, which is why those two exist here";
    } else if (c.neg == DN_SLICE_DROP) {
        return "--neg slice_drop only exists on the two-pass form (there is no partial to skip)";
    }
    return nullptr;
}

std::string darm_key(const DCfg& c) {
    char buf[96];
    if (c.form == DF_SHIP) std::snprintf(buf, sizeof buf, "down|ship|%s", c.act_bf16 ? "bf16" : "f32");
    else if (c.form == DF_INTRA)
        std::snprintf(buf, sizeof buf, "down|intra%d|%s", c.wpr, c.act_bf16 ? "bf16" : "f32");
    else std::snprintf(buf, sizeof buf, "down|two%d|%s", c.spl, c.act_bf16 ? "bf16" : "f32");
    return buf;
}

// ---------------------------------------------------------------- the deterministic fixture
// A synthetic fixture, stated as such: this rig does NOT adjudicate a numerics band against the pinned
// HF dumps (that is speccheck/ and prototype/mlp_gemv's gate). What it adjudicates is whether a claimed
// reduction order is the order the kernel executes, and for that the fixture only has to be
// row-distinct, sign-mixed and reproducible.
unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline unsigned short fixture_w(int r, int k) {
    const unsigned long long h = mix64(0x5EEDull + (unsigned long long)r * 4096u + (unsigned)k);
    const int q = (int)(h % 2001ull) - 1000;              // [-1000, 1000]
    const float v = (float)q * 0.0005f;                   // |w| <= 0.5
    hip_bfloat16 b = hip_bfloat16(v);
    unsigned short bits; std::memcpy(&bits, &b, 2);
    return bits;
}
inline float fixture_x(int k) {
    const unsigned long long h = mix64(0xC0FFEEull + (unsigned)k * 8192ull);
    const int q = (int)(h % 2001ull) - 1000;
    // Round the fixture through BF16 on purpose. The engine's activation row IS BF16 (`sc.mlp_act`)
    // and the F32 path only ever hands the kernel a widened copy of it, so the two dtypes must carry
    // the SAME numbers here or the cast-fold twin is judged against a fixture it never saw.
    hip_bfloat16 b = hip_bfloat16((float)q * 0.001f);
    unsigned short bits; std::memcpy(&bits, &b, 2);
    const unsigned widened = ((unsigned)bits) << 16;
    float v; std::memcpy(&v, &widened, 4);
    return v;
}

// bf16 bit helpers. bf16_rank() is the monotone ordered-integer map: for a sign-magnitude float,
// rank = (b & 0x8000) ? ~b : b + 0x8000. Writing it as `np.where(x < 0, …)` over a uint16 view is the
// dead-branch bug in AGENTS.md (never true for uint16, every negative collapses to +0.0 and the gate
// passes vacuously), so the map is asserted in --selftest-geom, not trusted.
long bf16_rank(unsigned short b) {
    return (b & 0x8000u) ? (long)(unsigned short)(b ^ 0xFFFFu) : (long)(unsigned short)(b ^ 0x8000u);
}
float bf16_val(unsigned short b) {
    const unsigned u = ((unsigned)b) << 16;
    float f;
    std::memcpy(&f, &u, 4);        // a bitcast, not a conversion: (float)u would convert the integer
    return f;
}
unsigned short f32_to_bf16_bits(float f) {
    hip_bfloat16 b = hip_bfloat16(f);           // the same RNE constructor the device uses
    unsigned short bits; std::memcpy(&bits, &b, 2);
    return bits;
}
unsigned short bf16_bits(const hip_bfloat16* p) { unsigned short b; std::memcpy(&b, p, 2); return b; }

// The warp ladder's association, on the host, restricted to the lanes lane 0 can read: after the
// step at offset o, a[l] (l < o) holds exactly what lane l's register holds on the device, and the
// lanes the device doubled (l >= ws-o) are never read by lane 0's chain.
static float host_ladder(std::vector<float> a) {
    const int ws = (int)a.size();
    for (int o = ws / 2; o; o >>= 1) {
        const std::vector<float> b = a;              // shfl reads every lane's value simultaneously
        for (int l = 0; l < o; ++l) a[l] = b[l] + b[l + o];
    }
    return a[0];
}

// The CPU mirror of one arm's arithmetic, including its reduction order. `use_fma` selects between
// the contracted product the GPU almost certainly emits and the two-rounding product a plain host sum
// makes; whichever one reproduces the shipped form bit for bit is the mode every claim in this run is
// made in, and the rig prints which one that was instead of assuming it.
void down_mirror(const std::vector<unsigned short>& w, const std::vector<float>& x, int slices,
                 bool use_fma, std::vector<unsigned short>& out) {
    out.assign(kDownRows, 0);
    const int sw = kDownWords / slices;
    std::vector<float> lanes(32);
    for (int r = 0; r < kDownRows; ++r) {
        float total = 0.f;
        for (int s = 0; s < slices; ++s) {
            for (int lane = 0; lane < 32; ++lane) {
                float acc = 0.f;
                for (int v = lane; v < sw; v += 32) {
                    const int word = s * sw + v;
                    for (int j = 0; j < 8; ++j) {
                        const float wv = bf16_val(w[(size_t)r * kDownK + word * 8 + j]);
                        const float xv = x[word * 8 + j];
                        acc = use_fma ? __builtin_fmaf(wv, xv, acc) : (acc + wv * xv);
                    }
                }
                lanes[lane] = acc;
            }
            const float part = host_ladder(lanes);
            total = (s == 0) ? part : total + part;
        }
        out[(size_t)r] = f32_to_bf16_bits(total);
    }
}

// The derived row sets for a down-arm injection. Built from the same geometry constants the launcher
// is built from, so a control's must-catch set is arithmetic and cannot rot.
struct DRowSets { long must = 0, same = 0; };
DRowSets dn_derive(const DCfg& c) {
    const DGeom g = dgeom(c);
    std::vector<char> diff(kDownRows, 0);
    if (c.neg == DN_NONE) return {0, (long)kDownRows};
    if (c.neg == DN_TRUNC) {
        diff[(size_t)g.inj_row] = 1;
    } else if (c.neg == DN_ROW_ALIAS) {
        for (int i = 0; i < g.inj_rows; ++i) {
            const int r = g.inj_block * g.block_rows + i;
            if (r < kDownRows) { diff[(size_t)r] = 1; diff[(size_t)(r ^ 1)] = 1; }
        }
    } else if (c.neg == DN_POISON || c.neg == DN_SLICE_DROP) {
        for (int i = 0; i < g.inj_rows; ++i) {
            const int r = g.inj_block * g.block_rows + i;
            if (r < kDownRows) diff[(size_t)r] = 1;
        }
    }
    DRowSets rs;
    for (int r = 0; r < kDownRows; ++r) { if (diff[r]) ++rs.must; else ++rs.same; }
    return rs;
}

// ---------------------------------------------------------------- down-arm run
struct DownRun {
    std::vector<float> part;          // the F32 partials, if the form has them
    std::vector<unsigned short> out;  // what the arm under test stored, as BF16 bits
    int guards = 0;
    Counts ct;
};

long d_inj_block(const DCfg& c) { return dgeom(c).inj_block; }

// The act region holds whichever dtype the arm stages; the template parameter decides the reading,
// so the buffer and the template can never disagree (a `float*` handed to a BF16 kernel is the
// uint16-view class of bug in a different costume).
void run_down_one(const DCfg& c, DownRun& rr, const std::vector<unsigned char>& xbytes,
                  bool prefill = true)
{
    const DGeom g = dgeom(c);
    const long inj = d_inj_block(c);
    const hip_bfloat16* W = reinterpret_cast<const hip_bfloat16*>(g_reg[R_W].p);
    const void* X = g_reg[R_X].p;
    hip_bfloat16* OUT = reinterpret_cast<hip_bfloat16*>(g_reg[R_OUT].p);
    float* PART = c.form == DF_TWO ? reinterpret_cast<float*>(g_reg[R_PART].p) : nullptr;
    // The sentinels are for the verified step only: leaving them in the timed loop would bill a memset
    // to the arm under test.
    if (prefill) {
        CK(hipMemset(g_reg[R_OUT].p, 0xA5, (size_t)kDownRows * 2));   // a pattern no arm stores
        if (c.form == DF_TWO) reg_fill(R_PART, 0xFF);                  // NaN: an unwritten slot is loud
    }

    if (c.form == DF_SHIP) {
        if (c.act_bf16) down_ship<kDownK, true><<<g.grid_x, kThreads>>>(W, X, OUT, kDownRows, kDownRpb);
        else down_ship<kDownK, false><<<g.grid_x, kThreads>>>(W, X, OUT, kDownRows, kDownRpb);
        rr.ct.got += 1;
    } else if (c.form == DF_INTRA) {
#define WS_INTRA(WPR)                                                                     \
        do {                                                                              \
            if (c.act_bf16) down_intra<kDownK, WPR, true><<<g.grid_x, kThreads>>>(W, X, OUT, kDownRows, c.neg, inj); \
            else down_intra<kDownK, WPR, false><<<g.grid_x, kThreads>>>(W, X, OUT, kDownRows, c.neg, inj);          \
        } while (0)
        switch (c.wpr) {
            case 1: WS_INTRA(1); break;
            case 2: WS_INTRA(2); break;
            case 4: WS_INTRA(4); break;
            default: WS_INTRA(8); break;
        }
#undef WS_INTRA
        rr.ct.got += 1;
    } else {
        const dim3 grid(g.grid_x, g.grid_y);
        const int red_grid = (kDownRows + kThreads - 1) / kThreads;
#define WS_PART(S)                                                                        \
        do {                                                                              \
            if (c.act_bf16) down_part<kDownK, S, true><<<grid, kThreads>>>(W, X, PART, kDownRows, c.neg, inj); \
            else down_part<kDownK, S, false><<<grid, kThreads>>>(W, X, PART, kDownRows, c.neg, inj);           \
            if (c.act_bf16) down_reduce<S><<<red_grid, kThreads>>>(PART, OUT, kDownRows);                      \
            else down_reduce<S><<<red_grid, kThreads>>>(PART, OUT, kDownRows);                                 \
        } while (0)
        switch (c.spl) {
            case 2: WS_PART(2); break;
            case 4: WS_PART(4); break;
            case 7: WS_PART(7); break;
            case 8: WS_PART(8); break;
            default: WS_PART(14); break;
        }
#undef WS_PART
        rr.ct.got += 2;
    }
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    rr.ct.want += g.launches;
    rr.ct.syncs += 1;
    rr.out.assign((size_t)kDownRows, 0);
    if (!prefill) return;                     // a timed run reads nothing back
    CK(hipMemcpy(rr.out.data(), g_reg[R_OUT].p, (size_t)kDownRows * 2, hipMemcpyDeviceToHost));
    if (c.form == DF_TWO) {
        rr.part.resize((size_t)c.spl * kDownRows);
        CK(hipMemcpy(rr.part.data(), g_reg[R_PART].p, (size_t)c.spl * kDownRows * sizeof(float),
                     hipMemcpyDeviceToHost));
    }
    rr.guards = check_guards();
}

// ---------------------------------------------------------------- down-arm comparison
struct DChk {
    int ok = 1;
    int mode_fma = 0;                 // 1 if the contracted mirror reproduced the frozen form
    int mode_nofma = 0;
    long ndiff_ship = 0, ulp_max = 0, ulp_over1 = 0, unwritten = 0;
    double meanrel = 0.0, maxabs = 0.0;
    long must_total = 0, must_caught = 0, same_total = 0, same_violated = 0;
    int guards = 0, part_unwritten = 0, act_moved = 0, nan_seen = 0;
    char why[208] = "";
    int nbad = 0;
    char bad[3][64] = {};
    void note(const char* what, long at) {
        if (nbad < 3) { std::snprintf(bad[nbad], sizeof bad[nbad], "%s@%ld", what, at); ++nbad; }
    }
    std::string where() const {
        std::string out = "first_bad=";
        for (int i = 0; i < nbad; ++i) { out += bad[i]; out += (i + 1 < nbad) ? "," : ""; }
        return nbad ? out : out + "none";
    }
};

// ulp distance in the BF16 lattice via the monotone rank map; the map itself is asserted in
// --selftest-geom because the naive uint16 form has a dead branch (AGENTS.md).
long bf16_ulp(unsigned short a, unsigned short b) {
    const long ra = bf16_rank(a), rb = bf16_rank(b);
    return ra > rb ? ra - rb : rb - ra;
}

void dcompare(const std::vector<unsigned short>& got, const std::vector<unsigned short>& ref,
              const DRowSets& rs, const DCfg& c, DChk& ck, const std::vector<char>& must_diff) {
    double denom = 0.0, acc = 0.0;
    for (size_t r = 0; r < got.size(); ++r) {
        const double v = std::fabs(bf16_val(ref[r]));
        if (v > denom) denom = v;
    }
    if (denom == 0.0) denom = 1.0;
    for (size_t r = 0; r < got.size(); ++r) {
        const bool diff = (got[r] != ref[r]);
        const bool sent = (got[r] == 0xA5A5);
        const int cls = must_diff[r] ? MUST_DIFF : MUST_SAME;
        if (cls == MUST_DIFF) { ++ck.must_total; if (diff) ++ck.must_caught; else ck.note("row", (long)r); }
        else { ++ck.same_total; if (diff) { ++ck.same_violated; ck.note("row", (long)r); } }
        if (sent) { ++ck.unwritten; ck.note("unwritten", (long)r); }
        if (diff) {
            ++ck.ndiff_ship;
            const long u = bf16_ulp(got[r], ref[r]);
            ck.ulp_max = std::max(ck.ulp_max, u);
            ck.ulp_over1 += (u > 1);
            const double d = std::fabs(bf16_val(got[r]) - bf16_val(ref[r]));
            if (!(d == d)) { ck.nan_seen = 1; }         // never print a NaN aggregate (AGENTS.md)
            else { ck.maxabs = std::max(ck.maxabs, d); acc += d; }
        }
    }
    ck.meanrel = acc / (double)got.size() / denom;
    (void)c;
}

// ---------------------------------------------------------------- allocation
void matrix_alloc_fill(const MPlan& p) {
    reg_alloc(R_SRC, (size_t)kTotRead * 8);
    reg_alloc(R_DST, (size_t)kTotWrite * 8);
    reg_alloc(R_REC, (size_t)p.n_rec * sizeof(MRec));
    fill_word<<<512, 256>>>(reinterpret_cast<unsigned long long*>(g_reg[R_SRC].p), (long)kTotRead);
    CK(hipDeviceSynchronize());
    // Spot-check the fill at both ends, or every read sum compares a formula against itself.
    const long long idx[2] = {0, kTotRead - 1};
    for (long long i : idx) {
        unsigned long long v = 0;
        CK(hipMemcpy(&v, g_reg[R_SRC].p + (size_t)i * 8, 8, hipMemcpyDeviceToHost));
        if (v != word_at(i)) { std::printf("FAIL_fill_mismatch at %lld\n", i); std::exit(4); }
    }
    reg_fill(R_DST, 0xFF);
    reg_fill(R_REC, 0xFF);
}

void down_alloc_fill(const DCfg& c, std::vector<unsigned char>& xbytes,
                     std::vector<unsigned short>& wb) {
    reg_alloc(R_W, (size_t)kDownRows * kDownK * 2);
    const size_t xlen = (size_t)kDownK * (c.act_bf16 ? 2 : 4);
    reg_alloc(R_X, xlen);
    reg_alloc(R_OUT, (size_t)kDownRows * 2 + 64);
    reg_alloc(R_PART, c.form == DF_TWO ? (size_t)c.spl * kDownRows * 4 : 0);
    std::vector<unsigned short> wbits((size_t)kDownRows * kDownK);
    for (int r = 0; r < kDownRows; ++r)
        for (int k = 0; k < kDownK; ++k) wbits[(size_t)r * kDownK + k] = fixture_w(r, k);
    wb = wbits;      // the mirror and the device must read ONE fixture, not two generations of it
    CK(hipMemcpy(g_reg[R_W].p, wbits.data(), wbits.size() * 2, hipMemcpyHostToDevice));
    xbytes.assign(xlen, 0);
    if (c.act_bf16) {
        for (int k = 0; k < kDownK; ++k) {
            const unsigned short b = f32_to_bf16_bits(fixture_x(k));
            std::memcpy(&xbytes[(size_t)k * 2], &b, 2);
        }
    } else {
        for (int k = 0; k < kDownK; ++k) {
            const float v = fixture_x(k);
            std::memcpy(&xbytes[(size_t)k * 4], &v, 4);
        }
    }
    CK(hipMemcpy(g_reg[R_X].p, xbytes.data(), xlen, hipMemcpyHostToDevice));
    reg_fill(R_OUT, 0xA5);
    if (c.form == DF_TWO) reg_fill(R_PART, 0xFF);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------- one matrix cell
int run_matrix_case(const MCfg& c, char* line, size_t ln, DevFacts& df, int* ok_out = nullptr,
                    MPlan* plan_out = nullptr, bool timed = false,
                    char* tline = nullptr, size_t tn = 0) {
    if (const char* r = mrefuse(c)) { std::snprintf(line, ln, "REFUSED %s", r); return 9; }
    MPlan p; mplan_build(c, p);
    if (const char* b = mplan_invariants(c, p, df)) { std::snprintf(line, ln, "PLAN-BAD %s", b); return 9; }
    matrix_alloc_fill(p);
    Counts ct = mrun(c, p);
    Chk chk; mverify(c, p, chk);
    const long long waves = p.blocks * (kThreads / df.wave);
    std::snprintf(line, ln,
                  "%s arm=%s launches=%ld/%d grid=%d blocks_total=%lld wavefronts_total=%lld "
                  "wavefronts_per_cu=%.2f blocks_per_cu=%.2f recs=%lld read_mib=%.4f write_mib=%.4f "
                  "demand_mib=%.4f dev_vs_census_read=%+.5f%% verify=%s read_checked_bytes=%lld "
                  "must=%ld/%ld same=%ld/viol %ld %s %s%s%s",
                  (c.neg == MN_NONE ? (chk.ok ? "CLEAN" : "FAIL")
                                    : (c.neg == MN_PERMUTE_READS ? (chk.ok ? "BLIND-OK" : "BLIND-BROKEN")
                                                                 : (chk.ok ? "CAUGHT" : "NOT-CAUGHT"))),
                  marm_key(c).c_str(), ct.got, c.L, c.G, (long long)p.blocks, (long long)waves,
                  (double)waves / (double)df.cus, (double)p.blocks / (double)df.cus,
                  (long long)p.n_rec, (double)p.read_bytes / 1048576.0,
                  (double)p.write_bytes / 1048576.0, (double)(p.read_bytes + p.write_bytes) / 1048576.0,
                  100.0 * dev_read_vs_census(), chk.ok ? "ok" : "FAIL",
                  (long long)chk.read_bytes_checked, chk.must_caught, chk.must_total, chk.same_total,
                  chk.same_violated, chk.why, chk.where().c_str(),
                  chk.diag[0] ? " diag=" : "", chk.diag[0] ? chk.diag : "");
    if (ok_out) *ok_out = chk.ok;
    if (plan_out) *plan_out = p;
    if (timed) {
        // ORDER MATTERS: the witness line must reach the reader before or together with the timing, and
        // the timing must never be printed for a row the witnesses did not clear (AGENTS: a timing
        // harness that prints ms for a red admission row is the `burst_m16` trap).
        char* out = tline;
        size_t on = tn;
        if (!out) { out = line + std::strlen(line); on = ln - std::strlen(line); }
        if (!chk.ok) {
            std::snprintf(out, on, "NO-TIMING arm=%s reason=verify_not_ok", marm_key(c).c_str());
        } else if (!c.time_flag) {
            std::snprintf(out, on, "NO-TIMING arm=%s reason=no_--time_flag", marm_key(c).c_str());
        } else {
            double med = 0.0, worst = 0.0;
            const double best = timed_steps([&] { mrun(c, p); }, c.reps, c.iters, &med, &worst);
            std::snprintf(out, on,
                          "RESULT arm=%s family=mat launches=%d grid=%d blocks_total=%lld "
                          "wavefronts_total=%lld wavefronts_per_cu=%.2f read_mib=%.4f write_mib=%.4f "
                          "demand_mib=%.4f verify=ok launches_executed=%ld syncs_executed=%ld "
                          "ms_min=%.6f ms_median=%.6f ms_max=%.6f per_launch_us=%.4f "
                          "per_block_us=%.4f reps=%d iters=%d",
                          marm_key(c).c_str(), c.L, c.G, (long long)p.blocks, (long long)waves,
                          (double)waves / (double)df.cus, (double)p.read_bytes / 1048576.0,
                          (double)p.write_bytes / 1048576.0,
                          (double)(p.read_bytes + p.write_bytes) / 1048576.0,
                          ct.got, ct.syncs, best, med, worst,
                          best * 1000.0 / (double)c.L, best * 1000.0 / (double)p.blocks,
                          c.reps, c.iters);
        }
    }
    alloc_free_all();
    return chk.ok ? 0 : 1;
}

// ---------------------------------------------------------------- one down arm
struct DownEval {
    DCfg c;
    DGeom g;
    DownRun arm, ship;
    std::vector<unsigned short> mir_fma, mir_nofma, mir_arm;
    DChk ck;                        // arm vs the mirror of the arm's own order
    DChk vs_ship;                   // arm vs the shipped form, device against device
    int ship_clean = 0;
    long frozen_ndiff_fma = 0, frozen_ndiff_nofma = 0;
};

// The whole numerics story for one arm, in four ordered steps. The ORDER matters: the mirror's own
// floor is established on the frozen form BEFORE any split arm is judged against it, so a device/host
// product-policy difference can never be read as a split-K defect.
void down_eval(DownEval& de, const std::vector<unsigned char>& xbytes,
               const std::vector<float>& xf, const std::vector<unsigned short>& wb) {
    de.g = dgeom(de.c);

    // (1) the frozen shipped form: its own spelling, never injected.
    DCfg sc; sc.form = DF_SHIP; sc.act_bf16 = de.c.act_bf16;
    run_down_one(sc, de.ship, xbytes);

    // (2) the mirror of the frozen form in both product policies; whichever reproduces it EXACTLY is
    //     the policy the device emitted, and that is the mode every later claim in this run uses.
    down_mirror(wb, xf, 1, true, de.mir_fma);
    down_mirror(wb, xf, 1, false, de.mir_nofma);
    long d_fma = 0, d_nofma = 0;
    for (size_t r = 0; r < de.ship.out.size(); ++r) {
        d_fma += (de.ship.out[r] != de.mir_fma[r]);
        d_nofma += (de.ship.out[r] != de.mir_nofma[r]);
    }
    de.frozen_ndiff_fma = d_fma;
    de.frozen_ndiff_nofma = d_nofma;
    de.ship_clean = (d_fma == 0 || d_nofma == 0);
    const bool use_fma = (d_fma == 0);
    de.ck.mode_fma = use_fma ? 1 : 0;
    de.ck.mode_nofma = (!use_fma && d_nofma == 0) ? 1 : 0;

    // (3) the arm under test, mirrored in that mode with the arm's own reduction order.
    run_down_one(de.c, de.arm, xbytes);
    down_mirror(wb, xf, dgeom_spl(de.c), use_fma, de.mir_arm);
    const DRowSets rs = dn_derive(de.c);
    std::vector<char> must(kDownRows, 0);
    if (de.c.neg == DN_TRUNC) must[(size_t)de.g.inj_row] = 1;
    else if (de.c.neg == DN_ROW_ALIAS) {
        for (int i = 0; i < de.g.inj_rows; ++i) {
            const int r = de.g.inj_block * de.g.block_rows + i;
            if (r < kDownRows) { must[r] = 1; must[r ^ 1] = 1; }
        }
    } else if (de.c.neg == DN_POISON || de.c.neg == DN_SLICE_DROP) {
        for (int i = 0; i < de.g.inj_rows; ++i) {
            const int r = de.g.inj_block * de.g.block_rows + i;
            if (r < kDownRows) must[r] = 1;
        }
    }
    // The must-set is derived twice, here and in dn_derive(), from the same geometry constants; if the
    // two spellings ever disagree the row verdict is meaningless, so it is refused rather than picked.
    {
        long here = 0;
        for (int r = 0; r < kDownRows; ++r) here += must[r];
        if (here != rs.must) {
            std::printf("FAIL_must_set_disagrees here=%ld dn_derive=%ld\n", here, rs.must);
            std::exit(4);
        }
    }
    dcompare(de.arm.out, de.mir_arm, rs, de.c, de.ck, must);
    de.ck.guards = de.arm.guards;
    de.ck.ok = (de.c.neg == DN_NONE)
                   ? (de.ck.same_violated == 0 && de.ck.unwritten == 0 && de.ck.guards == 0)
                   : (de.ck.must_total > 0 && de.ck.must_caught == de.ck.must_total &&
                      de.ck.same_violated == 0 && de.ck.guards == 0);
    if (de.c.neg != DN_NONE && de.ck.must_total == 0)
        std::snprintf(de.ck.why, sizeof de.ck.why, "BLIND-BY-GEOMETRY_empty_must_catch");

    // (4) arm against the shipped form, device against device. This is the reassociation size with no
    //     CPU mirror in the comparison at all, which is what makes it a claim about the kernels.
    const std::vector<char> none(kDownRows, 0);
    dcompare(de.arm.out, de.ship.out, DRowSets{0, (long)kDownRows}, de.c, de.vs_ship, none);
    if (de.vs_ship.nan_seen)
        std::snprintf(de.vs_ship.why, sizeof de.vs_ship.why, "NaN_in_a_compared_value aggregates_printed_as_minus_one");
}

int run_down_case(const DCfg& c, char* line, size_t ln, DevFacts& df,
                  std::vector<unsigned short>* out_bits = nullptr, int* ok_out = nullptr,
                  bool timed = false, char* tline = nullptr, size_t tn = 0) {
    if (const char* r = drefuse(c)) { std::snprintf(line, ln, "REFUSED %s", r); return 9; }
    std::vector<unsigned char> xbytes;
    std::vector<unsigned short> wbits;
    down_alloc_fill(c, xbytes, wbits);
    std::vector<float> xf((size_t)kDownK);
    std::vector<unsigned short> xb((size_t)kDownK);
    for (int k = 0; k < kDownK; ++k) { xf[k] = fixture_x(k); xb[k] = f32_to_bf16_bits(fixture_x(k)); }
    DownEval de; de.c = c;
    down_eval(de, xbytes, xf, wbits);
    const int clean = de.ship_clean && de.ck.ok;
    std::snprintf(line, ln,
                  "%s arm=%s launches=%ld/%d grid=%dx%d blocks_total=%d wavefronts_total=%d "
                  "wavefronts_per_cu=%.2f slices=%d act=%s weight_mib=%.3f stage_kib=%.1f "
                  "part_kib=%.1f mirror=%s frozen_fma=%ld frozen_nofma=%ld mirror_ndiff=%ld "
                  "vs_ship_ndiff=%ld max_ulp=%ld ulp_over1=%ld meanrel=%s guards=%d "
                  "must=%ld/%ld same=%ld/viol %ld %s %s",
                  (c.neg == DN_NONE ? (clean ? "CLEAN" : "FAIL")
                                    : (de.ck.ok ? "CAUGHT" : "NOT-CAUGHT")),
                  darm_key(c).c_str(), de.arm.ct.got, de.g.launches, de.g.grid_x, de.g.grid_y,
                  de.g.blocks_total, de.g.warps_total,
                  (double)de.g.warps_total / (double)df.cus, de.g.slices,
                  c.act_bf16 ? "bf16" : "f32", (double)de.g.weight_bytes / 1048576.0,
                  (double)de.g.stage_bytes / 1024.0, (double)de.g.part_bytes / 1024.0,
                  de.ck.mode_fma ? "fma" : (de.ck.mode_nofma ? "nofma" : "NEITHER"),
                  (long)de.frozen_ndiff_fma, (long)de.frozen_ndiff_nofma, (long)de.ck.ndiff_ship,
                  (long)de.vs_ship.ndiff_ship, (long)de.vs_ship.ulp_max, (long)de.vs_ship.ulp_over1,
                  de.vs_ship.nan_seen ? "NaN-refused" : std::to_string(de.vs_ship.meanrel).c_str(),
                  de.arm.guards, de.ck.must_caught, de.ck.must_total, de.ck.same_total,
                  de.ck.same_violated, de.ck.why, de.ck.where().c_str());
    if (out_bits) *out_bits = de.arm.out;
    if (ok_out) *ok_out = clean ? 1 : 0;
    if (timed) {
        char* out = tline;
        size_t on = tn;
        if (!out) { out = line + std::strlen(line); on = ln - std::strlen(line); }
        if (!clean) {
            std::snprintf(out, on, "NO-TIMING arm=%s reason=verify_not_ok", darm_key(c).c_str());
        } else if (!c.time_flag) {
            std::snprintf(out, on, "NO-TIMING arm=%s reason=no_--time_flag", darm_key(c).c_str());
        } else {
            DownRun warm;
            double med = 0.0, worst = 0.0;
            const double best = timed_steps(
                [&] { run_down_one(c, warm, xbytes, false); }, c.reps, c.iters, &med, &worst);
            std::snprintf(out, on,
                          "RESULT arm=%s family=down launches=%d grid=%dx%d blocks_total=%d "
                          "wavefronts_total=%d wavefronts_per_cu=%.2f slices=%d act=%s "
                          "weight_mib=%.3f stage_kib=%.1f part_kib=%.1f verify=ok mirror=%s "
                          "vs_ship_ndiff=%ld max_ulp=%ld launches_executed=%ld syncs_executed=%ld "
                          "ms_min=%.6f ms_median=%.6f ms_max=%.6f per_launch_us=%.4f "
                          "per_block_us=%.4f reps=%d iters=%d",
                          darm_key(c).c_str(), de.g.launches, de.g.grid_x, de.g.grid_y,
                          de.g.blocks_total, de.g.warps_total, (double)de.g.warps_total / df.cus,
                          de.g.slices, c.act_bf16 ? "bf16" : "f32",
                          (double)de.g.weight_bytes / 1048576.0,
                          (double)de.g.stage_bytes / 1024.0, (double)de.g.part_bytes / 1024.0,
                          de.ck.mode_fma ? "fma" : (de.ck.mode_nofma ? "nofma" : "NEITHER"),
                          (long)de.vs_ship.ndiff_ship, (long)de.vs_ship.ulp_max,
                          de.arm.ct.got, de.arm.ct.syncs, best, med, worst,
                          best * 1000.0 / (double)de.g.launches,
                          best * 1000.0 / (double)de.g.blocks_total, c.reps, c.iters);
        }
    }
    alloc_free_all();
    return clean ? 0 : 1;
}

// ---------------------------------------------------------------- the plan, in ONE spelling
// plan_table.py re-derives this table independently and diffs it line for line; a cell that exists in
// one spelling and not the other is a geometry bug, not a typo.
static const int kMatL[] = {16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
static const int kMatG[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
static const int kDownWpr[] = {1, 2, 4, 8};
static const int kDownSpl[] = {2, 4, 7, 8, 14};

void print_plan(const DevFacts& df) {
    // The gap against the step's census is printed in PERCENT here, the same scale the per-cell
    // `dev_vs_census_read` uses. A plan line that carries a fraction while the result lines carry a
    // percent is a unit mismatch waiting to be misread by 100x.
    std::printf("PLAN_TOTALS read_el=%lld read_mib=%.4f write_el=%lld write_mib=%.4f "
                "census_read_mib=%.4f census_write_mib=%.4f dev_read_pct=%+.5f dev_write_pct=%+.5f "
                "threads=%d cus=%d wave=%d\n",
                (long long)kTotRead, tot_read_mib(), (long long)kTotWrite, tot_write_mib(),
                (double)census_el_read() * 8.0 / 1048576.0, (double)census_el_write() * 8.0 / 1048576.0,
                100.0 * dev_read_vs_census(), 100.0 * dev_write_vs_census(), kThreads, df.cus, df.wave);
    int n = 0, nre = 0;
    for (int li = 0; li < (int)(sizeof(kMatL) / sizeof(int)); ++li) {
        for (int gi = 0; gi < (int)(sizeof(kMatG) / sizeof(int)); ++gi) {
            const long long B = (long long)kMatL[li] * kMatG[gi];
            const bool reach = (kTotRead % B == 0) && (kTotWrite % B == 0);
            MCfg c; c.L = kMatL[li]; c.G = kMatG[gi];
            if (reach) {
                MPlan p; mplan_build(c, p);
                const long long waves = p.blocks * (kThreads / df.wave);
                std::printf("PLAN_CELL L=%d G=%d B=%lld blocks_per_launch=%d per_launch_mib=%.6f "
                            "per_block_kib=%.4f read_mib=%.4f write_mib=%.4f demand_mib=%.4f "
                            "wavefronts=%lld wavefronts_per_cu=%.2f reachable=1\n",
                            kMatL[li], kMatG[gi], B, kMatG[gi],
                            (double)p.read_bytes / kMatL[li] / 1048576.0,
                            (double)p.u_r * 8.0 / 1024.0, (double)p.read_bytes / 1048576.0,
                            (double)p.write_bytes / 1048576.0,
                            (double)(p.read_bytes + p.write_bytes) / 1048576.0, (long long)waves,
                            (double)waves / (double)df.cus);
                ++nre;
            } else {
                std::printf("PLAN_CELL L=%d G=%d B=%lld blocks_per_launch=%d reachable=0 "
                            "reason=block_total_does_not_divide_the_pinned_bytes\n",
                            kMatL[li], kMatG[gi], B, kMatG[gi]);
            }
            ++n;
        }
    }
    std::printf("PLAN_CELLS total=%d reachable=%d\n", n, nre);
    // How many ways each block total can be reached: the iso-population diagonal needs at least three,
    // and a diagonal that can only be reached one way is not a diagonal.
    for (long long B = 128; B <= 16384; B *= 2) {
        int ways = 0;
        for (int li = 0; li < (int)(sizeof(kMatL) / sizeof(int)); ++li)
            if (B % kMatL[li] == 0) {
                const long long G = B / kMatL[li];
                for (int gi = 0; gi < (int)(sizeof(kMatG) / sizeof(int)); ++gi)
                    if ((long long)kMatG[gi] == G) ++ways;
            }
        if (ways) std::printf("PLAN_DIAG B=%lld ways=%d\n", B, ways);
    }
    DCfg d;
    d.form = DF_SHIP;
    {
        DGeom g = dgeom(d);
        std::printf("PLAN_DOWN arm=%s launches=%d grid=%dx%d blocks_total=%d wavefronts_total=%d "
                    "slices=%d weight_mib=%.4f stage_kib=%.1f part_kib=%.1f\n",
                    darm_key(d).c_str(), g.launches, g.grid_x, g.grid_y, g.blocks_total,
                    g.warps_total, g.slices, (double)g.weight_bytes / 1048576.0,
                    (double)g.stage_bytes / 1024.0, (double)g.part_bytes / 1024.0);
    }
    for (int w : kDownWpr) {
        DCfg c; c.form = DF_INTRA; c.wpr = w;
        DGeom g = dgeom(c);
        std::printf("PLAN_DOWN arm=%s launches=%d grid=%dx%d blocks_total=%d wavefronts_total=%d "
                    "slices=%d weight_mib=%.4f stage_kib=%.1f part_kib=%.1f\n",
                    darm_key(c).c_str(), g.launches, g.grid_x, g.grid_y, g.blocks_total,
                    g.warps_total, g.slices, (double)g.weight_bytes / 1048576.0,
                    (double)g.stage_bytes / 1024.0, (double)g.part_bytes / 1024.0);
    }
    for (int s : kDownSpl) {
        DCfg c; c.form = DF_TWO; c.spl = s;
        DGeom g = dgeom(c);
        std::printf("PLAN_DOWN arm=%s launches=%d grid=%dx%d blocks_total=%d wavefronts_total=%d "
                    "slices=%d weight_mib=%.4f stage_kib=%.1f part_kib=%.1f\n",
                    darm_key(c).c_str(), g.launches, g.grid_x, g.grid_y, g.blocks_total,
                    g.warps_total, g.slices, (double)g.weight_bytes / 1048576.0,
                    (double)g.stage_bytes / 1024.0, (double)g.part_bytes / 1024.0);
    }
    DCfg b; b.form = DF_INTRA; b.wpr = 4; b.act_bf16 = 1;
    {
        DGeom g = dgeom(b);
        std::printf("PLAN_DOWN arm=%s launches=%d grid=%dx%d blocks_total=%d wavefronts_total=%d "
                    "slices=%d weight_mib=%.4f stage_kib=%.1f part_kib=%.1f\n",
                    darm_key(b).c_str(), g.launches, g.grid_x, g.grid_y, g.blocks_total,
                    g.warps_total, g.slices, (double)g.weight_bytes / 1048576.0,
                    (double)g.stage_bytes / 1024.0, (double)g.part_bytes / 1024.0);
    }
}

// ---------------------------------------------------------------- selftest: geometry (no device)
int selftest_geom() {
    int fails = 0;
    auto need = [&](bool cond, const char* what) {
        std::printf("  %-76s %s\n", what, cond ? "ok" : "FAIL");
        if (!cond) ++fails;
    };
    DevFacts df = dev_facts();
    std::printf("== the pinned bytes\n");
    need(kWordK * kWordInv == 1ull,
         "the read-fill multiplier really is invertible mod 2^64, so a sum knows its element count");
    need(range_sum(0, 3) == word_at(0) + word_at(1) + word_at(2) &&
             range_sum(7, 40) == range_sum(0, 40) - range_sum(0, 7),
         "the closed-form range sum equals the elementwise sum it replaces");
    need(std::fabs(dev_read_vs_census()) <= 0.005 && std::fabs(dev_write_vs_census()) <= 0.025,
         "the pinned totals are within 0.5 % (read) and 2.5 % (write) of the step census [Q2]");
    need(kTotRead % 1024 == 0 && kTotWrite % 1024 == 0, "both pinned totals are whole numbers of KiB");

    std::printf("== the BF16 comparator (the dead-branch bug this repo has been bitten by)\n");
    need(bf16_rank(0x3F80u) > bf16_rank(0x3F00u), "rank(1.0) > rank(0.5): positives increase with rank");
    need(bf16_rank(0xBF80u) < bf16_rank(0x8000u),
         "rank(-1.0) < rank(-0.0): NEGATIVES DO NOT COLLAPSE TO ZERO (the uint16-view bug)");
    need(bf16_rank(0xBF80u) < bf16_rank(0x3F80u) && bf16_ulp(0xBF80u, 0xBF81u) == 1,
         "ulp(-1.0, -1.0078125) == 1 and rank(-1.0) < rank(+1.0): the map is monotone across zero");
    need(bf16_ulp(0x3F80u, 0x3F80u) == 0 && bf16_ulp(0xBF80u, 0xBF00u) == 128,
         "ulp(x,x)==0 and one BF16 octave is 128 steps (7 mantissa bits), not 8");
    need(bf16_val(0x0000u) == 0.0f && bf16_val(0x8000u) == 0.0f,
         "both zero signs decode to zero (the rank map puts them one step apart; the value does not)");
    need(std::fabs(bf16_val(0x3F80u) - 1.0f) < 1e-9f && std::fabs(bf16_val(0xBF80u) + 1.0f) < 1e-9f,
         "bf16_val decodes both signs (a uint16 view would have said -1.0 is +0.0)");

    std::printf("== the matrix cells\n");
    int reachable = 0, diag_ok = 0;
    for (int L : kMatL)
        for (int G : kMatG) {
            MCfg c; c.L = L; c.G = G;
            if (mrefuse(c)) continue;
            MPlan p; mplan_build(c, p);
            ++reachable;
            if (mplan_invariants(c, p, df)) { std::printf("   invariant FAIL L=%d G=%d: %s\n", L, G, mplan_invariants(c, p, df)); ++fails; }
            if (p.u_r * p.blocks != kTotRead || p.u_w * p.blocks != kTotWrite) {
                std::printf("   exactness FAIL L=%d G=%d\n", L, G);
                ++fails;
            }
        }
    need(reachable >= 60, "at least 60 (L,G) cells are reachable at the pinned byte totals");
    for (long long B = 256; B <= 16384; B *= 2) {
        int ways = 0;
        for (int L : kMatL)
            if (B % L == 0) {
                const long long G = B / L;
                for (int g : kMatG) if ((long long)g == G) ++ways;
            }
        if (ways >= 3) ++diag_ok;
    }
    need(diag_ok >= 6, "at least six block totals are reachable three different ways (iso-population diagonals)");
    {
        MCfg c; c.L = 100; c.G = 7;      // 700 blocks: not a divisor of the pinned bytes
        need(mrefuse(c) != nullptr, "a cell whose block total does not divide the pinned bytes is refused");
        MCfg d; d.L = 0; need(mrefuse(d) != nullptr, "L<1 is refused");
        MCfg e; e.G = 0; need(mrefuse(e) != nullptr, "G<1 is refused");
    }

    std::printf("== the down_proj geometry, from source (rows 1024, K 3584, rpb 8, warp 32)\n");
    {
        DCfg s; s.form = DF_SHIP;
        DGeom g = dgeom(s);
        need(g.grid_x == 128 && g.block_rows == 8 && g.launches == 1 && g.warps_total == 1024,
             "the shipped shape reproduces: 128 blocks x 8 rows x 1 warp, 1024 wavefronts [Q4]");
        need(g.weight_bytes == 7340032, "down_proj's weight bytes are 7.0 MiB (1024 x 3584 x 2)");
        for (int w : kDownWpr) {
            DCfg c; c.form = DF_INTRA; c.wpr = w;
            DGeom h = dgeom(c);
            char msg[160];
            std::snprintf(msg, sizeof msg,
                          "intra wpr=%d: grid %d (x%d), %d rows/block, %d wavefronts, 1 launch",
                          w, h.grid_x, h.grid_y, h.block_rows, h.warps_total);
            need(h.grid_x == 128 * w && h.warps_total == 1024 * w && h.launches == 1 &&
                     h.grid_x * h.block_rows == kDownRows, msg);
        }
        for (int s2 : kDownSpl) {
            DCfg c; c.form = DF_TWO; c.spl = s2;
            DGeom h = dgeom(c);
            char msg[160];
            std::snprintf(msg, sizeof msg, "two-pass split=%d: pass1 grid 128x%d + reduce, 2 launches",
                          s2, h.grid_y);
            need(h.grid_x == 128 && h.grid_y == s2 && h.launches == 2 &&
                     h.warps_total == 1024 * s2 + 4 * 8, msg);
        }
    }
    std::printf("== the down-arm refusals\n");
    {
        struct Bad { DCfg c; const char* why; };
        Bad bads[] = {
            {[&]{ DCfg c; c.form = DF_SHIP; c.neg = DN_TRUNC; return c; }(),
             "the frozen form takes no injection"},
            {[&]{ DCfg c; c.form = DF_INTRA; c.wpr = 3; return c; }(), "--wpr=3 does not divide 8 rows"},
            {[&]{ DCfg c; c.form = DF_INTRA; c.wpr = 8; c.neg = DN_TRUNC; return c; }(),
             "wpr=8 word_short is legal, so this one must NOT refuse"},
            {[&]{ DCfg c; c.form = DF_TWO; c.spl = 3; return c; }(), "--split=3 is not a divisor of 448"},
            {[&]{ DCfg c; c.form = DF_TWO; c.spl = 5; return c; }(), "--split=5 is not instantiated"},
            {[&]{ DCfg c; c.form = DF_TWO; c.spl = 4; c.neg = DN_ROW_ALIAS; return c; }(),
             "row_alias is an intra injection point"},
            {[&]{ DCfg c; c.form = DF_INTRA; c.wpr = 4; c.neg = DN_SLICE_DROP; return c; }(),
             "slice_drop needs the two-pass form"},
        };
        for (Bad& b : bads) {
            const char* r = drefuse(b.c);
            const bool expect_refuse = std::strstr(b.why, "must NOT refuse") == nullptr;
            char msg[180];
            std::snprintf(msg, sizeof msg, "%s", b.why);
            need(expect_refuse ? (r != nullptr) : (r == nullptr), msg);
        }
    }
    std::printf("== derived control sets, matrix\n");
    for (int neg = 1; neg < MN_COUNT; ++neg) {
        MCfg c; c.L = 32; c.G = 8; c.neg = neg;
        MPlan p; mplan_build(c, p);
        MNegSets ns = mneg_derive(c, p);
        char msg[160];
        std::snprintf(msg, sizeof msg, "%-14s must=%ld (rec=%ld w=%ld) same=%ld%s", kMNName[neg],
                      ns.must, ns.must_rec, ns.must_w, ns.same,
                      neg == MN_PERMUTE_READS ? " [declared blind]" : "");
        if (neg == MN_PERMUTE_READS) need(ns.must == 0 && ns.same > 0, msg);
        else need(ns.must > 0 && ns.same > 0, msg);
    }
    std::printf("== derived control sets, down arms\n");
    {
        struct NC { const char* nm; DCfg c; };
        std::vector<NC> v;
        { DCfg c; c.form = DF_INTRA; c.wpr = 1; c.neg = DN_TRUNC; v.push_back({"intra1 trunc", c}); }
        { DCfg c; c.form = DF_INTRA; c.wpr = 4; c.neg = DN_TRUNC; v.push_back({"intra4 trunc", c}); }
        { DCfg c; c.form = DF_INTRA; c.wpr = 2; c.neg = DN_ROW_ALIAS; v.push_back({"intra2 row_alias", c}); }
        { DCfg c; c.form = DF_INTRA; c.wpr = 8; c.neg = DN_ROW_ALIAS; v.push_back({"intra8 row_alias", c}); }
        { DCfg c; c.form = DF_INTRA; c.wpr = 4; c.neg = DN_POISON; v.push_back({"intra4 poison", c}); }
        { DCfg c; c.form = DF_TWO; c.spl = 4; c.neg = DN_POISON; v.push_back({"two4 poison", c}); }
        { DCfg c; c.form = DF_TWO; c.spl = 8; c.neg = DN_SLICE_DROP; v.push_back({"two8 slice_drop", c}); }
        for (NC& nc : v) {
            const DRowSets rs = dn_derive(nc.c);
            char msg[160];
            std::snprintf(msg, sizeof msg, "%-18s must=%ld same=%ld", nc.nm, rs.must, rs.same);
            need(rs.must > 0 && rs.same > 0, msg);
        }
    }
    std::printf("SELFTEST-GEOM %s fails=%d\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------- device selftest
struct CaseSpec {
    const char* name;
    int kind;          // 0 matrix clean · 1 matrix caught · 2 matrix blind · 3 down clean
                       // 4 down caught · 5 pair must be bit-identical
    int which = 0;     // 0 = matrix, 1 = down, 2 = pair
    MCfg m;
    DCfg d, d2;
};

std::vector<CaseSpec> case_list() {
    std::vector<CaseSpec> v;
    auto push = [&](const char* nm, int kind, int which, MCfg m, DCfg d, DCfg d2) {
        CaseSpec s; s.name = nm; s.kind = kind; s.which = which; s.m = m; s.d = d; s.d2 = d2;
        v.push_back(s);
    };
    MCfg mb; mb.L = 32; mb.G = 8;                       // B=256, the cheap witness cell
    // Matrix positives across the shapes the sweep can set, including an iso-population pair
    // (L=32,G=192 and L=48,G=128 are both 6144 blocks at the same bytes).
    push("pos_128x8", 0, 0, [] { MCfg c; c.L = 128; c.G = 8; return c; }(), DCfg{}, DCfg{});
    push("pos_16x64", 0, 0, [] { MCfg c; c.L = 16; c.G = 64; return c; }(), DCfg{}, DCfg{});
    push("pos_1024x8", 0, 0, [] { MCfg c; c.L = 1024; c.G = 8; return c; }(), DCfg{}, DCfg{});
    push("pos_48x128", 0, 0, [] { MCfg c; c.L = 48; c.G = 128; return c; }(), DCfg{}, DCfg{});
    push("pos_32x192", 0, 0, [] { MCfg c; c.L = 32; c.G = 192; return c; }(), DCfg{}, DCfg{});
    push("pos_16x1024", 0, 0, [] { MCfg c; c.L = 16; c.G = 1024; return c; }(), DCfg{}, DCfg{});
    for (int neg = 1; neg < MN_COUNT; ++neg) {
        MCfg c = mb; c.neg = neg;
        CaseSpec s; s.name = kMNName[neg]; s.kind = (neg == MN_PERMUTE_READS) ? 2 : 1;
        s.which = 0; s.m = c; v.push_back(s);
    }
    // Down arms.
    push("down_ship", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_SHIP; return c; }(), DCfg{});
    push("down_intra1", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_INTRA; c.wpr = 1; return c; }(), DCfg{});
    push("down_intra2", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_INTRA; c.wpr = 2; return c; }(), DCfg{});
    push("down_intra4", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_INTRA; c.wpr = 4; return c; }(), DCfg{});
    push("down_intra8", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_INTRA; c.wpr = 8; return c; }(), DCfg{});
    push("down_two2", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_TWO; c.spl = 2; return c; }(), DCfg{});
    push("down_two4", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_TWO; c.spl = 4; return c; }(), DCfg{});
    push("down_two8", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_TWO; c.spl = 8; return c; }(), DCfg{});
    push("down_two14", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_TWO; c.spl = 14; return c; }(), DCfg{});
    push("down_ship_bf16", 3, 1, MCfg{}, [] { DCfg c; c.form = DF_SHIP; c.act_bf16 = 1; return c; }(), DCfg{});
    push("down_intra4_bf16", 3, 1, MCfg{},
         [] { DCfg c; c.form = DF_INTRA; c.wpr = 4; c.act_bf16 = 1; return c; }(), DCfg{});
    {
        struct NN { const char* nm; DCfg c; };
        std::vector<NN> negs;
        negs.push_back({"dn_trunc", [] { DCfg c; c.form = DF_INTRA; c.wpr = 4; c.neg = DN_TRUNC; return c; }()});
        negs.push_back({"dn_row_alias", [] { DCfg c; c.form = DF_INTRA; c.wpr = 2; c.neg = DN_ROW_ALIAS; return c; }()});
        negs.push_back({"dn_poison_intra", [] { DCfg c; c.form = DF_INTRA; c.wpr = 4; c.neg = DN_POISON; return c; }()});
        negs.push_back({"dn_poison_two", [] { DCfg c; c.form = DF_TWO; c.spl = 4; c.neg = DN_POISON; return c; }()});
        negs.push_back({"dn_slice_drop", [] { DCfg c; c.form = DF_TWO; c.spl = 8; c.neg = DN_SLICE_DROP; return c; }()});
        for (NN& n : negs) { CaseSpec s; s.name = n.nm; s.kind = 4; s.which = 1; s.d = n.c; v.push_back(s); }
    }
    // The two structural claims, as cases that run both arms and compare bits.
    {
        CaseSpec s; s.name = "pair_intra1_eq_ship"; s.kind = 5; s.which = 2;
        s.d = [] { DCfg c; c.form = DF_INTRA; c.wpr = 1; return c; }();
        s.d2 = [] { DCfg c; c.form = DF_SHIP; return c; }();
        v.push_back(s);
    }
    {
        CaseSpec s; s.name = "pair_intra4_eq_two4"; s.kind = 5; s.which = 2;
        s.d = [] { DCfg c; c.form = DF_INTRA; c.wpr = 4; return c; }();
        s.d2 = [] { DCfg c; c.form = DF_TWO; c.spl = 4; return c; }();
        v.push_back(s);
    }
    {
        CaseSpec s; s.name = "pair_intra8_eq_two8"; s.kind = 5; s.which = 2;
        s.d = [] { DCfg c; c.form = DF_INTRA; c.wpr = 8; return c; }();
        s.d2 = [] { DCfg c; c.form = DF_TWO; c.spl = 8; return c; }();
        v.push_back(s);
    }
    {
        CaseSpec s; s.name = "pair_f32_eq_bf16act"; s.kind = 5; s.which = 2;
        s.d = [] { DCfg c; c.form = DF_INTRA; c.wpr = 1; return c; }();
        s.d2 = [] { DCfg c; c.form = DF_INTRA; c.wpr = 1; c.act_bf16 = 1; return c; }();
        v.push_back(s);
    }
    return v;
}

int run_case(const CaseSpec& s, char* line, size_t ln, DevFacts& df) {
    if (s.which == 0) {
        int ok = 0;
        const int rc = run_matrix_case(s.m, line, ln, df, &ok);
        if (rc == 9) return 9;
        // kind 0 wants CLEAN, 1 CAUGHT, 2 BLIND; run_matrix_case already chose the word from cfg.neg,
        // and `ok` means "behaved as its kind demands", so the exit code is the same for all three.
        return ok ? 0 : 1;
    }
    if (s.which == 1) {
        int ok = 0;
        const int rc = run_down_case(s.d, line, ln, df, nullptr, &ok);
        if (rc == 9) return 9;
        return ok ? 0 : 1;
    }
    std::vector<unsigned short> a, b;
    char l1[640], l2[640];
    int ok1 = 0, ok2 = 0;
    if (run_down_case(s.d, l1, sizeof l1, df, &a, &ok1) == 9) { std::snprintf(line, ln, "%s", l1); return 9; }
    if (run_down_case(s.d2, l2, sizeof l2, df, &b, &ok2) == 9) { std::snprintf(line, ln, "%s", l2); return 9; }
    long ndiff = 0, first = -1;
    for (size_t r = 0; r < a.size() && r < b.size(); ++r)
        if (a[r] != b[r]) { ++ndiff; if (first < 0) first = (long)r; }
    const bool same = (a.size() == b.size()) && ndiff == 0 && ok1 && ok2;
    std::snprintf(line, ln, "%s arms=%s vs %s rows=%zu ndiff=%ld first_row=%ld both_clean=%d/%d",
                  same ? "EQUAL" : "DIFFER", darm_key(s.d).c_str(), darm_key(s.d2).c_str(),
                  a.size(), ndiff, first, ok1, ok2);
    return same ? 0 : 1;
}

int selftest_device() {
    const std::vector<CaseSpec> cases = case_list();
    int want[6] = {0, 0, 0, 0, 0, 0};
    for (const CaseSpec& s : cases) ++want[s.kind];
    int parsed = 0, bad = 0, got[6] = {0, 0, 0, 0, 0, 0};
    char line[640];
    for (const CaseSpec& s : cases) {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            DevFacts df = dev_facts();
            const int rc = run_case(s, line, sizeof line, df);
            std::printf("CASE %-20s %s\n", s.name, line);
            fflush(stdout);
            _exit(rc);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        ++parsed;
        const int code = WIFEXITED(st) ? WEXITSTATUS(st) : 3;
        if (code == 0) ++got[s.kind]; else ++bad;
    }
    const int want_total = (int)cases.size();
    const bool ok = (bad == 0 && parsed == want_total && want[0] == got[0] && want[1] == got[1] &&
                     want[2] == got[2] && want[3] == got[3] && want[4] == got[4] && want[5] == got[5]);
    std::printf("SELFTEST %s cases=%d parsed=%d children_bad=%d clean=%d/%d caught=%d/%d blind=%d/%d "
                "down_clean=%d/%d down_caught=%d/%d pairs_equal=%d/%d\n",
                ok ? "ok" : "FAILED", want_total, parsed, bad, got[0], want[0], got[1], want[1],
                got[2], want[2], got[3], want[3], got[4], want[4], got[5], want[5]);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- main
double timed_steps(const std::function<void()>& step, int reps, int iters, double* med, double* worst) {
    std::vector<double> per;
    double best = 1e30, w = 0.0;
    for (int r = 0; r < reps; ++r) {
        step();                                    // warm within the rep
        double bm = 1e30;
        for (int it = 0; it < iters; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            step();
            const auto t1 = std::chrono::steady_clock::now();
            bm = std::min(bm, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        per.push_back(bm);
        best = std::min(best, bm);
        w = std::max(w, bm);
    }
    *med = median_of(per);
    *worst = w;
    return best;
}

int real_main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    MCfg mc; DCfg dc;
    int mode = 0;                       // 1 selftest 2 selftest-geom 3 print-cases 4 print-plan 5 print-device
    int have_cell = 0, have_down = 0;
    std::string down_name;
    auto num = [&](int& i, const char* flag) {
        if (i + 1 >= argc) { std::printf("ABORT missing value for %s\n", argv[i]); std::exit(2); }
        const std::string s = argv[++i];
        char* end = nullptr;
        const double d = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0') {
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
        else if (a == "--print-plan") mode = 4;
        else if (a == "--print-device") mode = 5;
        else if (a == "--time") { mc.time_flag = dc.time_flag = 1; }
        else if (a == "--reps") { mc.reps = (int)num(i, "--reps"); dc.reps = mc.reps; }
        else if (a == "--iters") { mc.iters = (int)num(i, "--iters"); dc.iters = mc.iters; }
        else if (a == "--launches" || a == "--L") { mc.L = (int)num(i, a.c_str()); have_cell = 1; }
        else if (a == "--grid" || a == "--G") { mc.G = (int)num(i, a.c_str()); have_cell = 1; }
        else if (a == "--neg") {
            const std::string s = argv[++i];
            int k = 0;
            for (; k < MN_COUNT; ++k) if (s == kMNName[k]) break;
            if (k < MN_COUNT) { mc.neg = k; }
            else {
                int j = 0;
                for (; j < DN_COUNT; ++j) if (s == kDNName[j]) break;
                if (j == DN_COUNT) { std::printf("ABORT bad --neg %s\n", s.c_str()); return 2; }
                dc.neg = j;
            }
        } else if (a == "--down") { down_name = argv[++i]; have_down = 1; }
        else if (a == "--wpr") { dc.wpr = (int)num(i, "--wpr"); dc.form = DF_INTRA; have_down = 1; }
        else if (a == "--split") { dc.spl = (int)num(i, "--split"); dc.form = DF_TWO; have_down = 1; }
        else if (a == "--down-act") {
            const std::string s = argv[++i];
            if (s == "f32") dc.act_bf16 = 0;
            else if (s == "bf16") dc.act_bf16 = 1;
            else { std::printf("ABORT bad --down-act %s (f32|bf16)\n", s.c_str()); return 2; }
        } else if (a == "--pages") {
            const std::string s = argv[++i];
            if (s == "cold") {
                std::printf("ABORT REFUSED --pages cold: first touch of fresh pages runs ~7x slow "
                            "(devices/), so cold is refused, not averaged. The fill plus the verified "
                            "untimed step is this rig's warm-up.\n");
                return 2;
            }
            if (s != "warm") { std::printf("ABORT bad --pages %s (warm; cold is refused)\n", s.c_str()); return 2; }
        } else { std::printf("ABORT unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mode == 2) return selftest_geom();          // no device, no HIP call on this path
    if (mode == 4) { print_plan(dev_facts()); return 0; }
    if (mode == 5) { print_device(dev_facts()); return 0; }
    if (mode == 3) {
        const std::vector<CaseSpec> cs = case_list();
        for (const CaseSpec& s : cs) std::printf("CASE-ROW %s kind=%d which=%d\n", s.name, s.kind, s.which);
        std::printf("CASE-ROWS total=%zu\n", cs.size());
        return 0;
    }
    if (mode == 1) return selftest_device();

    if (have_down == have_cell) {
        std::printf("ABORT exactly one of --down / --launches+--grid is required (or --print-plan, "
                    "--selftest-geom, --selftest, --print-device). Got %d down and %d cell flags.\n",
                    have_down, have_cell);
        return 2;
    }
        // Every check that needs no device runs before the device is touched: a bad cell or a bad split is a
    // CLI error in milliseconds, not a HIP context plus a 1.5 GiB fill followed by an error.
    if (have_down) {
        if (down_name == "ship") dc.form = DF_SHIP;
        else if (down_name == "intra") dc.form = DF_INTRA;
        else if (down_name == "two") dc.form = DF_TWO;
        else { std::printf("ABORT bad --down %s (ship|intra|two)\n", down_name.c_str()); return 2; }
        if (const char* r = drefuse(dc)) { std::printf("ABORT: %s\n", r); return 3; }
        dc.arm = darm_key(dc);
    } else {
        if (const char* r = mrefuse(mc)) { std::printf("ABORT: %s\n", r); return 3; }
        mc.arm = marm_key(mc);
    }
    DevFacts df = dev_facts();
    if (have_down) {
        char wl[900], tl[900];
        if (run_down_case(dc, wl, sizeof wl, df, nullptr, nullptr, true, tl, sizeof tl)) {
            std::printf("CASE %s\n%s\n", darm_key(dc).c_str(), wl);
            std::printf("%s\n", tl);
            return 1;
        }
        std::printf("CASE %s\n%s\n", darm_key(dc).c_str(), wl);
        std::printf("%s\n", tl);
        return 0;
    }
    char wl[900], tl[900];
    if (run_matrix_case(mc, wl, sizeof wl, df, nullptr, nullptr, true, tl, sizeof tl)) {
        std::printf("CASE %s\n%s\n", marm_key(mc).c_str(), wl);
        std::printf("%s\n", tl);
        return 1;
    }
    std::printf("CASE %s\n%s\n", marm_key(mc).c_str(), wl);
    std::printf("%s\n", tl);
    return 0;
}

}  // namespace

int main(int argc, char** argv) { return real_main(argc, argv); }
