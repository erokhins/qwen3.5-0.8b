/*
 * prototype/mlp_insitu/mlp_insitu.hip.cpp — why does the MLP one-row walk lose ~1.09 ms in the step?
 *
 * D2 (progress/2026-10-21/decode-attribution-2026-10-21.md §5) is a DIAGNOSIS, not an optimization.
 * The quoted facts it has to explain are QUOTED, not measured here — every millisecond in this route
 * belongs to the main agent:
 *
 *   burst   2.275 ms/token  `engine/performance-status.md` §4: same 48 launches, 504 MiB, `contig`.
 *   in-situ 3.3628 ms/token at 4,032 lanes today (3.3880 in the route line): `decode_run
 *           --time-stages`, the line `STAGE gran=stage family=mlp`.
 *   penalty ~1.09 ms, on a kernel that sits at 98 % of the 237 GB/s read probe when run alone.
 *
 * WHAT THIS IS. A rig that runs THE SHIPPING KERNEL — it `#include`s
 * `prototype/mlp_gemv/mlp_gemv_kernels.h`, the same header `engine/kernels_mlp.cpp` includes, so the
 * object diagnosed and the object that ships are one source — under a matrix of conditions that
 * differ by exactly one variable, and that variable is always a FLAG. There is no second copy of the
 * kernel here, and nothing in `engine/` is touched.
 *
 * ============================================================================================
 * THE DECOMPOSITION — the rig's job is to turn a difference of two numbers into a SUM of named terms
 * so that an unexplained residue stays visible instead of being absorbed into a verdict.
 * ============================================================================================
 *   P              = R_in-situ-shaped − R_burst-shaped      (must reproduce ~1.09 or the rig is not
 *                                                          the step: NO-DIAGNOSIS, attribute nothing)
 *   T_serial       = drain-each − drain-burst, MLP alone   (48 drains + 48 exposed dispatches + an
 *                                                          empty queue per launch. NOT a memory
 *                                                          mechanism, and see the note below.)
 *   T_region       = 72 blobs − 3 blobs, same bytes        (region count / a walk split into pieces)
 *   T_span         = shipped 1.5 GiB span − packed 504 MiB (the resident set the step really
 *                                                          translates and evicts: candidate (b))
 *   T_readcarry    = read interferer before each drained   (candidate (a) in the only form a
 *                                                          launch, dose-swept          DRAINED family
 *                                                          can see — see the note)
 *   T_writecarry   = write interferer, same dose, same     (dirty-line writeback tax: the ONLY
 *                                                          drain                        way writes
 *                                                          can reach a drained number)
 *   T_ownwrites    = act/delta stores moved so they cannot (candidate (c) as posed: 0.221 MB/token,
 *                                                          stay in cache, dose-swept    shared vs spread)
 *   residue        = P − Σ(named terms), printed, never absorbed.
 *
 * NOTE ON "IN SITU", which changes what the three candidates even mean. The line D2 is priced against
 * comes from the SERIALIZED stage pass: `engine/stage_clock.h` — "family X's ms is what family X took
 * when nothing else was in the queue", and `note()` closes a segment whenever the family changes, so a
 * per-layer `mlp_gateup` and `mlp_down` are each their own drained segment (`segments_per_step` on the
 * STAGE line is the witness, and the main agent should print it). Consequence: candidate (a) in its
 * literal form — other families' traffic in FLIGHT with the MLP's — is EXCLUDED BY THE INSTRUMENT, not
 * measured by it. What a drained family can still inherit is STATE: cache-tier contents, page tables,
 * and lines left dirty by the step's 24.4 MB of writes. That is why `T_readcarry`/`T_writecarry` are
 * drained arms (`--if-when before`) and the true-concurrency arm (`--if-when concurrent`, two
 * non-default streams) is labelled a BOUND on (a), not a reproduction of it.
 *
 * ============================================================================================
 * WHY A NEW RIG — the census of instruments, taken before adding one (AGENTS.md: "before writing a
 * census tool, list the census tools").
 * ============================================================================================
 *   `tools/mixed_bw.cpp`            the traffic-CURVE tool (read/write/mix/mixlaunch/interleave over
 *                                   synthetic footprints). It runs a COPY of the engine's read
 *                                   mapping with the maths removed, so it can price a pattern but not
 *                                   a kernel; it cannot place blobs at shipped spacing, cannot put a
 *                                   store in the window, and cannot drain between them. Its
 *                                   `B_blk`/`B_rr` rows already refuted launch ORDER (ratio 1.005),
 *                                   which is why this rig does NOT re-test order.
 *   `tools/step_traffic_model.py`   the byte census parsed from `engine/state.h`. This rig is fed by
 *                                   `census.py`, which reads it; sizes are never typed in here.
 *   `prototype/mlp_gemv/`           the F1–F4 numerics gate over this exact kernel header, 192 cases.
 *                                   This rig does not re-run it, re-band it, or invent a band.
 *   `tools/bw_probe_hip.cpp`        the 237 GB/s read probe, quoted as the denominator.
 * New here, and only these: the real kernel in the window; blob placement (`--layout`); drained
 * state-carry-over with a dose (`--if-*`); store placement (`--writes`). The mixed_bw reference rows
 * stay in the main agent's window, because a ratio against a historical row is not a ratio.
 *
 * ============================================================================================
 * RULES THIS FILE OBEYS (each one is load-bearing; every one is an AGENTS.md trap row)
 * ============================================================================================
 *   * `steady_clock` bursts only, one sync at the end of a timed segment. `hipEventElapsedTime`
 *     returns exactly 0 below ~1 ms on this part and a per-launch MLP segment is 30–70 µs.
 *   * no ms is printed unless `--time` is given. `--selftest` prints no ms and no GB/s: a gate that
 *     also reports a bandwidth number is a measurement wearing a costume (`mixed_bw`).
 *   * a demanded-bytes rate is labelled, never quoted as a DRAM rate: `label=demanded_not_dram=1`
 *     rides on any row whose implied rate exceeds `--suspect-gbs` (default 237), `--strict` errors.
 *   * the byte account is a real block reduction (double per-thread partials, shared tree, one
 *     publish per block) — NOT `sums[blockIdx.x] = per_thread_acc`, which keeps one thread's total
 *     and failed every warm row it shipped (`tools/mixed_bw.cpp`, `publish_block_sum`).
 *   * writes are proven by landing in an arena pre-filled with a value the writer never stores; every
 *     region carries a poison guard band, so a stray read moves a checksum and a stray write stops
 *     being poison.
 *   * the geometry billed on the host is re-derived from the SAME struct the launch is built from
 *     (`bounds_for`), and the device selftest verifies BYTES TOUCHED (a bit-exact one-hot read,
 *     write-landing, guard bands) rather than arithmetic. Each device case runs in its own forked
 *     child, because one faulting kernel poisons the context and every case after it silently never
 *     ran (`mixed_bw` SELFTEST).
 *   * the must-catch / must-not-catch sets in `case_table()` are DERIVED from the geometry (word
 *     count, warp width, which buffer an injection can reach), and `--selftest-geom` asserts no
 *     injection has an empty must-catch set — an empty must-catch set is not a control.
 *   * `hipMalloc` only, never managed memory (silent degradation, `devices/strix-halo-unified-memory.md`).
 *   * cold pages are refused, not averaged: every region is warmed by an untimed pass and
 *     `--pages cold` exits 2, because the first GPU pass over fresh pages runs ~35 GB/s.
 *
 * USAGE. This is a timing tool: `--time` belongs to the main agent under `tools/gpu_lock.sh`, and the
 * program deliberately does NOT self-lock — the D2 window is one lock hold across many rows.
 *   prototype/mlp_insitu/mlp_insitu --dry-run          # plan + byte bill, no device, no timing
 *   prototype/mlp_insitu/mlp_insitu --selftest-geom    # CPU invariants + derived case list
 *   prototype/mlp_insitu/mlp_insitu --selftest         # untimed device witnesses
 *   prototype/mlp_insitu/mlp_insitu --time --tag R0 --drain burst --layout blob3 ...
 *
 * EXIT CODES: 0 ok; 2 bad arguments / impossible geometry; 3 a verification failed; 4 a suspect
 * (>237 GB/s demanded) row under `--strict`; 1 a HIP error.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>
#include <wait.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

// THE shipping kernel. Same header `engine/kernels_mlp.cpp` includes, so what is diagnosed is what
// ships. Brings HID=1024, FF=3584, ROWS_PER_BLOCK=8, NEG_*, MAP_*, bf/fb/rnd/reduce_row.
#include "../mlp_gemv/mlp_gemv_kernels.h"

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace {

constexpr double MIB = 1048576.0;
// QUOTED from `devices/strix-halo-memory-bandwidth.md`: the denominator, not a measurement here.
constexpr double READ_PROBE_GBS = 237.0;

constexpr int kGateRows = FF;    // mlp_gateup: 3584 rows of 1024
constexpr int kGateIn = HID;
constexpr int kDownRows = HID;   // mlp_down:   1024 rows of 3584
constexpr int kDownIn = FF;
constexpr size_t kGateBlob = (size_t)kGateRows * kGateIn * 2;   // 7 MiB exactly
constexpr size_t kDownBlob = (size_t)kDownRows * kDownIn * 2;   // 7 MiB exactly

// Fill values, as `mixed_bw`: readable, never-stored, poison.
constexpr float kFillVal = 1.0f;
constexpr float kUnwrittenVal = 0.0f;
constexpr uint16_t kBF16One = 0x3F80;        // BF16 1.0, so products through it are exact
constexpr unsigned kGuard = 1024;            // bytes of 0xFF on each side of every region

enum : int {
    kInjNone = 0,
    kInjShortGrid = 1,   // one block FEWER than the plan bills: a grid/rows mistake
    kInjShortRead = 2,   // interferer reads one float4 short: the read account catches
    kInjMissWrite = 3,   // interferer skips the last block's store: the landing check catches
    kInjStrayRead = 4,   // one float4 over the read region's edge: only the guard band sees it
    kInjStrayWrite = 5,  // one float4 over the write arena's edge: only the guard band sees it
    kInjSwapRow = 6,     // kernel NEG_SWAPROW: exact-pre catches; landing is blind BY CONSTRUCTION
    kInjTail = 7,        // kernel NEG_TAIL: only a one-hot at j = in-1 can catch it
    kInjHalf = 8,        // kernel NEG_HALF: only a one-hot at j >= in/2 can catch it
    kInjF32Act = 9,      // kernel NEG_F32ACT: derived blindness of both witnesses (must-not-catch)
};

/* --------------------------------------------------------------- interferer kernels --------- */

// One block publishes ONE double: per-thread partials, shared-mem tree, lane 0 writes. The
// per-thread-store version looked like a partial and was not one (`mixed_bw`).
__device__ __forceinline__ void publish_block_sum(double acc, double* __restrict__ sh,
                                                 double* __restrict__ sums) {
    const unsigned t = threadIdx.x, n = blockDim.x;   // n a power of two (host-checked)
    sh[t] = acc;
    __syncthreads();
    for (unsigned half = n / 2; half > 0; half /= 2) {
        if (t < half) sh[t] += sh[t + half];
        __syncthreads();
    }
    if (t == 0) sums[blockIdx.x] = sh[0];
}

__global__ void read_k(const float4* __restrict__ src, size_t n4, double* __restrict__ sums,
                       int verify, int inject) {
    extern __shared__ double sh[];
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double acc = 0.0;
    // short_read: one whole grid-stride ROUND is never read (every thread stops `stride` short). It
    // stays inside the region, so only the byte account can see it. The first version dropped
    // `n4 - 1` for a hand-picked thread, which dropped exactly zero float4s in practice because the
    // thread that owns index n4-1 was not in the set it selected — a negative control that could not
    // fire, which is the failure mode AGENTS.md records three times.
    const size_t last = (inject == kInjShortRead) ? (n4 > s ? n4 - s : 0) : n4;
    for (size_t i = t; i < last; i += s) {
        const float4 v = src[i];
        acc += (double)(v.x + v.y + v.z + v.w);
    }
    if (inject == kInjStrayRead && threadIdx.x == 0) {
        const float4 v = src[n4];              // first float4 of the guard band
        acc += (double)(v.x + v.y + v.z + v.w);
    }
    if (verify) publish_block_sum(acc, sh, sums);
}

__global__ void write_k(float4* __restrict__ dst, size_t n4, float v, int inject) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float4 val = make_float4(v, v, v, v);
    // miss_write: the LAST block does not store. Every other block does, so only the landing
    // read-back can see the hole.
    if (inject == kInjMissWrite && blockIdx.x + 1 == gridDim.x) return;
    for (size_t i = t; i < n4; i += s) dst[i] = val;
    if (inject == kInjStrayWrite && threadIdx.x == 0) dst[n4] = val;
}

__global__ void fill_k(float4* __restrict__ dst, size_t n4, float v) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float4 val = make_float4(v, v, v, v);
    for (size_t i = t; i < n4; i += s) dst[i] = val;
}

}  // namespace
namespace {

/* ------------------------------------------------------------------------- options ---------- */

struct Opt {
    int layers = 24;
    std::string kernels = "both";      // gateup | down | both
    std::string layout = "blob3";      // blob3 | blob72 | shipped72
    std::string drain = "burst";       // burst (one sync/token) | each (a sync per launch)
    std::string mapping = "contig";    // the shipped one-row walk is `contig`
    std::string writes = "shared";     // shared (as shipped) | spread (cannot stay in cache)
    std::string if_when = "before";    // before (drained) | concurrent (second stream)
    std::string if_sched = "perlayer"; // perlayer | once
    std::string pages = "warm";
    std::string geometry = "thin";     // thin = 48 launches, the shipped shape. fat = the shape the
                                       // published 2.275 ms burst was taken on: one launch per kernel
                                       // covering every layer, so grid is 24x bigger and launches 24x
                                       // fewer for the same bytes. See results.md provenance.
    int token_burst = 1;               // tokens submitted between syncs (1 = a drain per token, the
                                       // shipped measure; >1 = the reference's never-drained burst)
    std::string inject = "none";
    std::string tag = "row";
    int mlp = 1;                       // 0 = interferer alone (the cross-term needs it)
    double if_read_mib = 0.0;          // interferer read bytes per token
    double if_write_mib = 0.0;         // interferer write bytes per token
    double head_mib = 0.0;             // one whole-token blob per token (lm_head stand-in)
    double gap_mib = 42.0;             // per-layer other-weight span, layout=shipped72
    int write_mul = 1;                 // store dose: MiB of stride per layer when writes=spread
    int iters = 8;
    int reps = 3;
    int blocks_cap = 448;
    int time = 0;
    int verify = 1;
    int strict = 0;
    int selftest = 0, selftest_case = -1, geom_only = 0, dry = 0, print_cases = 0;
    double suspect_gbs = READ_PROBE_GBS;
};

struct Blob {
    void* p = nullptr;          // guard-band start
    void* data = nullptr;       // usable region
    size_t bytes = 0;
    double mib = 0;
    std::string name;
};

struct Plan {
    std::vector<Blob> allocs;
    std::vector<void*> wgu, wup, wdn;        // per layer
    float* x = nullptr;                      // gateup input, F32 [kGateIn]
    float* xa = nullptr;                     // down input (the widened act), F32 [kDownIn]
    Blob act, delta;                         // the shipped shared scratch, written 24x a token
    Blob act_spread, delta_spread;            // writes=spread destinations, one slot per layer
    std::vector<void*> act_slot, delta_slot; // writes=spread destinations
    Blob if_read, if_write, head;
    double* sums = nullptr;
    size_t sums_n = 0;
    int if_blocks = 256;                      // fixed launch width: one sums slot range per launch
    int if_seq = 0;                           // running launch index inside one verify pass
    size_t if_read_billed = 0, if_write_billed = 0, head_billed = 0;   // exact bytes, one pass
    int threads = 0, grid_gateup = 0, grid_down = 0;
    // billed per token, DEMANDED bytes (not a DRAM claim)
    double mlp_read_mib = 0, mlp_write_mib = 0, if_read_mib = 0, if_write_mib = 0;
    int launches_mlp = 0, launches_if = 0;
    double alloc_mib = 0;
};

struct Config {
    Opt o;
    Plan p;
};

// The warp width, cached. `--selftest` forks a child per case, and FORKING AFTER THE HIP RUNTIME HAS
// BEEN INITIALISED IN THE PARENT makes every child fail to allocate (observed: 76 children, 76
// `hip error out of memory`, none of them on a real memory limit). So no code path reachable from
// `--selftest-geom`/`--selftest` before the fork may touch the runtime; the query lives here and is
// called only by a process that is about to use the device.
int g_warp_width = 0;
int warp_width() {
    if (g_warp_width > 0) return g_warp_width;
    int w = 32;
    (void)hipDeviceGetAttribute(&w, hipDeviceAttributeWarpSize, 0);
    g_warp_width = w > 0 ? w : 32;
    return g_warp_width;
}
int rpb_threads() { return ROWS_PER_BLOCK * warp_width(); }

/*
 * The two launch geometries, in ONE place, because the whole point of the `fat` arm is that it is the
 * SAME bytes through the SAME kernel and only the launch granularity differs.
 *
 * The published `mlp` floor (engine/performance-status.md §4: 528 MB, `contig`, 2.275 ms, 98 % of
 * probe) was produced by `prototype/mlp_gemv/run.py --layers all --tile 24 --mapping contig`, and in
 * `mlp_gemv.hip.cpp` the row count is DERIVED FROM THE WEIGHT FILE SIZE (`rows = wbytes / (in*2)`,
 * line 219). With `--tile 24` the file is 24 replicas concatenated, so that burst is TWO launches of
 * rows=86,016 (grid 10,752) and rows2=24,576 (grid 3,072) per 24-layer iteration, and run.py divides
 * the iteration time by 24 to call it "per token". The engine emits 48 launches of grid 448 and 128.
 * Same bytes, same kernel, same mapping, same rows/block — 24x the work per launch and 1/24 the
 * launches. `thin` is the shipped shape; `fat` is the published one.
 */
inline bool is_fat(const Opt& o) { return o.geometry == "fat"; }
int geom_rows_gate(const Opt& o) { return (is_fat(o) ? o.layers : 1) * kGateRows; }
int geom_rows_down(const Opt& o) { return (is_fat(o) ? o.layers : 1) * kDownRows; }
size_t geom_gate_blob(const Opt& o) { return (size_t)(is_fat(o) ? o.layers : 1) * kGateBlob; }
size_t geom_down_blob(const Opt& o) { return (size_t)(is_fat(o) ? o.layers : 1) * kDownBlob; }
int geom_launches(const Opt& o, int layers) {
    if (!o.mlp) return 0;
    const bool do_g = o.kernels != "down";
    const bool do_d = o.kernels != "gateup";
    if (is_fat(o)) return (do_g ? 1 : 0) + (do_d ? 1 : 0);
    return layers * ((do_g ? 1 : 0) + (do_d ? 1 : 0));
}
// The must-LAND set, DERIVED from the arm's own launch list rather than hand-listed. `act` is
// gate/up's output and `delta` is down_proj's, so a `--kernels gateup` arm has no delta to land and a
// `--kernels down` arm has no act. The hand-listed version made row RAg print
// `verify=okFAIL_delta_unwritten` for a kernel that never ran, which is the failure class AGENTS.md
// records three times over (mixed_bw, mlp_gemv's own negative table, attn_tile_model).
enum { kLandAct = 1, kLandDelta = 2 };
int must_land_mask(const Opt& o) {
    if (!o.mlp) return 0;                       // interferer-only arm: nothing to land
    int m = 0;
    if (o.kernels != "down") m |= kLandAct;
    if (o.kernels != "gateup") m |= kLandDelta;
    return m;
}
// The fat launch walks the 24 layers as ONE region, which is only true if the per-layer slices are
// adjacent inside one allocation — that is `blob3`'s definition and not `blob72`/`shipped72`'s.
int fat_refusable(const Opt& o, char* why, size_t n) {
    if (!is_fat(o)) return 0;
    if (o.layout != "blob3") {
        snprintf(why, n, "--geometry fat requires --layout blob3: only there are the %d layers' "
                        "slices of one tensor adjacent inside one allocation, which is what a single "
                        "rows=%d launch walks. In blob72/shipped72 they are separate hipMallocs.",
                 o.layers, geom_rows_gate(o));
        return 2;
    }
    if (o.writes == "spread") {
        snprintf(why, n, "--geometry fat with --writes spread is not a question anyone is asking: "
                        "fat already writes every layer's act to its own slot, which IS the spread "
                        "arm. Pick one.");
        return 2;
    }
    return 0;
}

int parse_inject(const std::string& s) {
    if (s == "none") return kInjNone;
    if (s == "short_grid") return kInjShortGrid;
    if (s == "short_read") return kInjShortRead;
    if (s == "miss_write") return kInjMissWrite;
    if (s == "stray_read") return kInjStrayRead;
    if (s == "stray_write") return kInjStrayWrite;
    if (s == "swaprow") return kInjSwapRow;
    if (s == "tail") return kInjTail;
    if (s == "half") return kInjHalf;
    if (s == "f32act") return kInjF32Act;
    fprintf(stderr, "--inject: unknown name %s\n", s.c_str());
    exit(2);
}
const char* inject_name(int i) {
    switch (i) {
        case kInjNone: return "none";
        case kInjShortGrid: return "short_grid";
        case kInjShortRead: return "short_read";
        case kInjMissWrite: return "miss_write";
        case kInjStrayRead: return "stray_read";
        case kInjStrayWrite: return "stray_write";
        case kInjSwapRow: return "swaprow";
        case kInjTail: return "tail";
        case kInjHalf: return "half";
        case kInjF32Act: return "f32act";
    }
    return "?";
}
int kernel_neg(int inject) {
    switch (inject) {
        case kInjSwapRow: return NEG_SWAPROW;
        case kInjTail: return NEG_TAIL;
        case kInjHalf: return NEG_HALF;
        case kInjF32Act: return NEG_F32ACT;
        default: return NEG_NONE;
    }
}
size_t mib_bytes(double m) { return ((size_t)(m * MIB)) & ~((size_t)4095); }

/* ---------------------------------------------------------------------- plan bounds --------- */

/*
 * bounds_for — the host re-derivation of the kernel's index arithmetic, from the same fields the
 * launch is built from. `mlp_gateup` computes `row = blockIdx.x*rpb + threadIdx.x/ws`, then walks
 * `v = lane; v < IN/8; v += ws` over uint4 words of `W + row*IN`; `mlp_down` is the same with one
 * stream. So the last byte a launch may touch in a weight blob is
 *     (blocks*rpb - 1)*IN*2 + (IN/8 - 1)*16 + 16,
 * and the highest output slot it may write is min(blocks*rpb, rows) - 1. Deriving both from
 * rows/in/rpb — what the launcher actually passes — is what keeps a plan and a kernel from meaning
 * two different things again (the `mixed_bw` rounds lesson). `main()` refuses a plan where the grid
 * does not cover every row, and `kInjShortGrid` proves the refusal has a witness.
 */
struct Bounds {
    int blocks = 0, rpb = 0, rows = 0, in = 0;
    long long wavefronts = 0, rows_covered = 0, last_weight_byte = 0, out_slots = 0;
    int covers_rows = 0, exact_bytes = 0, one_grid_pass = 0;
};

Bounds bounds_for(int rows, int in, int rpb, int blocks_cap, size_t blob_bytes) {
    Bounds b;
    b.rows = rows;
    b.in = in;
    b.rpb = rpb;
    b.blocks = (rows + rpb - 1) / rpb;
    if (blocks_cap > 0 && b.blocks > blocks_cap) b.blocks = blocks_cap;
    b.wavefronts = (long long)b.blocks * b.rpb;
    b.rows_covered = std::min<long long>(b.wavefronts, rows);
    b.last_weight_byte = (long long)(b.blocks * b.rpb - 1) * in * 2 + (in / 8 - 1) * 16 + 16;
    b.out_slots = b.rows_covered;
    b.covers_rows = b.wavefronts >= rows;
    b.exact_bytes = (size_t)rows * (size_t)in * 2 == blob_bytes;
    b.one_grid_pass = b.wavefronts == rows;
    return b;
}

// The derived one-hot positions. `words = in/8` and the lane stride is the warp width, so the
// positions that can tell an off-by-one from a correct walk are the word edges, one warp's worth of
// words, the half-way word, and the last element of K. These come out of `words`/`ws`; they are not
// a hand list.
std::vector<int> onehot_positions(int in) {
    const int ws = 32;
    std::vector<int> cand = {0, 7, 8, ws * 8 - 1, ws * 8, in / 2, in - 1};
    std::vector<int> out;
    for (int j : cand)
        if (j >= 0 && j < in && std::find(out.begin(), out.end(), j) == out.end()) out.push_back(j);
    return out;
}

/* --------------------------------------------------------------------- allocation ----------- */

Blob alloc_region(const std::string& name, size_t bytes, char fill) {
    Blob b;
    b.name = name;
    b.bytes = bytes;
    b.mib = bytes / MIB;
    if (bytes == 0) return b;
    CK(hipMalloc(&b.p, bytes + 2 * kGuard));
    b.data = (char*)b.p + kGuard;
    if (fill == '0') CK(hipMemset(b.data, 0, bytes));
    if (fill == 'F') CK(hipMemset(b.data, 0xFF, bytes));
    // Guard bands: 0xFF bytes. As F32 that is a NaN, as BF16 a NaN too, so either a stray read moves
    // a checksum or a stray write stops being poison.
    CK(hipMemset(b.p, 0xFF, kGuard));
    CK(hipMemset((char*)b.data + bytes, 0xFF, kGuard));
    return b;
}

bool guard_intact(const Blob& b) {
    if (b.bytes == 0) return true;
    std::vector<unsigned char> buf(kGuard);
    CK(hipMemcpy(buf.data(), b.p, kGuard, hipMemcpyDeviceToHost));
    for (unsigned char c : buf)
        if (c != 0xFF) return false;
    CK(hipMemcpy(buf.data(), (char*)b.data + b.bytes, kGuard, hipMemcpyDeviceToHost));
    for (unsigned char c : buf)
        if (c != 0xFF) return false;
    return true;
}

void free_all(Plan& p) {
    for (Blob& b : p.allocs)
        if (b.p) (void)hipFree(b.p);
    if (p.x) (void)hipFree(p.x);
    if (p.xa) (void)hipFree(p.xa);
    if (p.sums) (void)hipFree(p.sums);
}

/*
 * build_weights — the placement arm, which is candidate (b) made testable.
 *   blob3     three contiguous blobs (all gate, all up, all down). This is the shape of the PUBLISHED
 *             2.275 ms burst, whose weights are one layer's tensors replicated 24x and concatenated
 *             (`prototype/mlp_gemv/run.py --tile 24`), so this row is the rig's reproduction of it.
 *   blob72    72 separate 7 MiB blobs, adjacent. Same bytes, same total span; only the region COUNT
 *             and the per-region base changes.
 *   shipped72 those 72 blobs separated by `--gap-mib` per layer of warm, allocated-but-never-read
 *             stand-ins for the ~1 GiB of OTHER weights the engine keeps resident
 *             (`engine/loader.cpp`: one hipMalloc per checkpoint tensor, 335 of them). Bytes
 *             billed are identical in all three; what differs is placement and the span the walk and
 *             the TLB see. `--gap-mib` is the dose knob for that.
 */
void build_weights(Config& c) {
    const Opt& o = c.o;
    Plan& p = c.p;
    const size_t gap = mib_bytes(o.gap_mib);
    const int L = o.layers;
    if (o.layout == "blob3") {
        Blob g = alloc_region("weights.gate", (size_t)L * kGateBlob, '1');
        Blob u = alloc_region("weights.up", (size_t)L * kGateBlob, '1');
        Blob d = alloc_region("weights.down", (size_t)L * kDownBlob, '1');
        for (int i = 0; i < L; ++i) {
            p.wgu.push_back((char*)g.data + (size_t)i * kGateBlob);
            p.wup.push_back((char*)u.data + (size_t)i * kGateBlob);
            p.wdn.push_back((char*)d.data + (size_t)i * kDownBlob);
        }
        p.allocs.push_back(g);
        p.allocs.push_back(u);
        p.allocs.push_back(d);
    } else {
        const bool gapd = (o.layout == "shipped72");
        for (int i = 0; i < L; ++i) {
            char nm[64];
            snprintf(nm, sizeof nm, "L%02d.gate", i);
            Blob g = alloc_region(nm, kGateBlob, '1');
            snprintf(nm, sizeof nm, "L%02d.up", i);
            Blob u = alloc_region(nm, kGateBlob, '1');
            snprintf(nm, sizeof nm, "L%02d.down", i);
            Blob d = alloc_region(nm, kDownBlob, '1');
            p.wgu.push_back(g.data);
            p.wup.push_back(u.data);
            p.wdn.push_back(d.data);
            p.allocs.push_back(g);
            p.allocs.push_back(u);
            p.allocs.push_back(d);
            if (gapd && i + 1 < L) {
                snprintf(nm, sizeof nm, "gap%02d", i);
                p.allocs.push_back(alloc_region(nm, gap, '0'));
            }
        }
    }
}

void build_rest(Config& c) {
    const Opt& o = c.o;
    Plan& p = c.p;
    if (o.if_read_mib > 0 || o.head_mib > 0) {
        p.if_read = alloc_region("if.read", mib_bytes(o.if_read_mib), '0');
        p.allocs.push_back(p.if_read);
        p.head = alloc_region("if.head", mib_bytes(o.head_mib), '0');
        p.allocs.push_back(p.head);
    }
    if (o.if_write_mib > 0) {
        p.if_write = alloc_region("if.write", mib_bytes(o.if_write_mib), '0');
        p.allocs.push_back(p.if_write);
    }
    CK(hipMalloc((void**)&p.x, kGateIn * 4));
    CK(hipMalloc((void**)&p.xa, kDownIn * 4));
    // act/delta are sized by the GEOMETRY. A thin gateup launch rewrites one 3,584-slot buffer 24 times;
    // a fat launch writes one slot per row of the whole 24-layer region (86,016 slots) once. The BYTES
    // per token are identical — that is what keeps the two arms comparable — but a fat arm that reused
    // the thin buffer would write 165 KB past it, and the landing witness would read a bigger region
    // than the allocation holds and report it as a memcpy error, or worse, as a pass.
    p.act = alloc_region("act", (size_t)geom_rows_gate(o) * 2, '0');
    p.delta = alloc_region("delta", (size_t)geom_rows_down(o) * 2, '0');
    p.allocs.push_back(p.act);
    p.allocs.push_back(p.delta);
    if (o.writes == "spread") {
        // Same stores, same byte count, different base addresses per layer, so a line written by
        // layer i cannot still be resident when layer i+1 stores. `--write-mul N` is the store dose
        // in MiB of destination stride per layer: it changes WHERE the stores land, never HOW MANY.
        const size_t stride = mib_bytes((double)o.write_mul);
        p.act_spread = alloc_region("act.spread", stride * o.layers, '0');
        p.delta_spread = alloc_region("delta.spread", stride * o.layers, '0');
        for (int i = 0; i < o.layers; ++i) {
            p.act_slot.push_back((char*)p.act_spread.data + (size_t)i * stride);
            p.delta_slot.push_back((char*)p.delta_spread.data + (size_t)i * stride);
        }
        p.allocs.push_back(p.act_spread);
        p.allocs.push_back(p.delta_spread);
    }
    // One sums slot range per interferer launch, so a launch that never ran, ran short, or ran twice
    // is visible: a single-slot account cannot see a skipped middle launch (`mixed_bw`).
    p.if_seq = 0;
    p.sums_n = (size_t)(2 * o.layers + 2) * p.if_blocks;
    CK(hipMalloc((void**)&p.sums, p.sums_n * sizeof(double)));
    CK(hipMemset(p.sums, 0, p.sums_n * sizeof(double)));
    // Warm every page this rig will time. A cold row is not a cold-page measurement, it is a wrong
    // one (~35 GB/s first pass), so warming is unconditional and `--pages cold` is refused.
    for (Blob& b : p.allocs) {
        if (b.bytes / 16 == 0) continue;
        fill_k<<<256, 256>>>((float4*)b.data, b.bytes / 16,
                             b.name.rfind("gap", 0) == 0 ? kUnwrittenVal : kFillVal);
    }
    std::vector<float> hx(kGateIn, 1.0f), hxa(kDownIn, 1.0f);
    CK(hipMemcpy(p.x, hx.data(), kGateIn * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(p.xa, hxa.data(), kDownIn * 4, hipMemcpyHostToDevice));
    CK(hipDeviceSynchronize());
}

/* ------------------------------------------------------------------------ billing ----------- */

void bill(Config& c) {
    const Opt& o = c.o;
    Plan& p = c.p;
    const bool do_g = o.kernels != "down";
    const bool do_d = o.kernels != "gateup";
    const int L = o.layers;
    // bill() stays device-free, because `--dry-run` and `--selftest-geom` are CPU-only paths and a
    // hipDeviceGetAttribute here would initialise the runtime in the SELFTEST parent before the fork
    // (that is how 76 forked children all reported "out of memory"). `run_arm()` fills the cache
    // before calling this; a CPU-only path reports the wave32 default, which is the same value this
    // part returns and is marked by the caller printing its own path.
    p.threads = ROWS_PER_BLOCK * (g_warp_width > 0 ? g_warp_width : 32);
    // `--blocks-cap` is a thin-shape lever (it emulates a grid that does not cover every row). On the
    // fat arm it would silently turn the published shape back into a capped one — the geometry
    // selftest caught exactly that, printing grid 448 where the published burst issues 10,752 — so the
    // fat arm always issues its full grid.
    const int cap = is_fat(o) ? 0 : o.blocks_cap;
    p.grid_gateup = bounds_for(geom_rows_gate(o), kGateIn, ROWS_PER_BLOCK, cap, geom_gate_blob(o))
                        .blocks;
    p.grid_down =
        bounds_for(geom_rows_down(o), kDownIn, ROWS_PER_BLOCK, cap, geom_down_blob(o)).blocks;
    if (o.mlp) {
        // The byte bill is geometry-free on purpose: fat and thin read and write the SAME bytes, and
        // `geom=` on the row is the only difference. If these two lines ever need a geometry term the
        // fat arm has stopped being a launch-granularity control and become a different workload.
        p.mlp_read_mib =
            (do_g ? 2.0 * L * kGateBlob : 0.0) / MIB + (do_d ? (double)L * kDownBlob / MIB : 0.0);
        p.mlp_write_mib = (do_g ? (double)L * kGateRows * 2 : 0.0) / MIB +
                          (do_d ? (double)L * kDownRows * 2 : 0.0) / MIB;
        p.launches_mlp = geom_launches(o, L);
    }
    p.if_read_mib = o.if_read_mib + o.head_mib;
    p.if_write_mib = o.if_write_mib;
    p.launches_if = (p.if_read_mib > 0 ? o.layers + 1 : 0) + (p.if_write_mib > 0 ? o.layers : 0);
    double m = 0;
    for (const Blob& b : p.allocs) m += b.mib;
    p.alloc_mib = m;
    if (p.alloc_mib > 8192) {
        fprintf(stderr,
                "ABORT: this run would allocate %.0f MiB, over the 8 GiB self-cap. Lower "
                "--layers/--gap-mib/--head-mib. (Device pool is ~34 GiB; the cap is a mistake "
                "detector, not a machine limit.)\n",
                p.alloc_mib);
        exit(2);
    }
}

void print_plan(const Config& c) {
    const Opt& o = c.o;
    const Plan& p = c.p;
    const int cap = is_fat(o) ? 0 : o.blocks_cap;      // must match bill(), or PLAN lies about the grid
    const Bounds bg = bounds_for(geom_rows_gate(o), kGateIn, ROWS_PER_BLOCK, cap, geom_gate_blob(o));
    const Bounds bd = bounds_for(geom_rows_down(o), kDownIn, ROWS_PER_BLOCK, cap, geom_down_blob(o));
    printf("PLAN arm tag=%s layout=%s drain=%s kernels=%s mapping=%s writes=%s write_mul=%d "
           "geom=%s token_burst=%d if_r=%.1f if_w=%.1f head=%.1f mlp=%d\n",
           o.tag.c_str(), o.layout.c_str(), o.drain.c_str(), o.kernels.c_str(), o.mapping.c_str(),
           o.writes.c_str(), o.write_mul, o.geometry.c_str(), o.token_burst, o.if_read_mib,
           o.if_write_mib, o.head_mib, o.mlp);
    printf("PLAN gateup grid=%d threads=%d rows=%d in=%d wavefronts=%lld covers=%d one_grid_pass=%d "
           "exact_bytes=%d last_weight_byte=%lld\n",
           bg.blocks, p.threads, geom_rows_gate(o), kGateIn, bg.wavefronts, bg.covers_rows,
           bg.one_grid_pass, bg.exact_bytes, bg.last_weight_byte);
    printf("PLAN down grid=%d threads=%d rows=%d in=%d wavefronts=%lld covers=%d one_grid_pass=%d "
           "exact_bytes=%d last_weight_byte=%lld\n",
           bd.blocks, p.threads, geom_rows_down(o), kDownIn, bd.wavefronts, bd.covers_rows,
           bd.one_grid_pass, bd.exact_bytes, bd.last_weight_byte);
    printf("PLAN billed_per_token DEMANDED mlp_read_mib=%.3f mlp_write_mib=%.3f if_read_mib=%.3f "
           "if_write_mib=%.3f launches_mlp=%d launches_if=%d alloc_mib=%.1f\n",
           p.mlp_read_mib, p.mlp_write_mib, p.if_read_mib, p.if_write_mib, p.launches_mlp,
           p.launches_if, p.alloc_mib);
}

/* ----------------------------------------------------------------------- launching ---------- */

int map_of(const std::string& s) {
    if (s == "contig") return MAP_CONTIG;
    if (s == "interleave") return MAP_INTERLEAVE;
    fprintf(stderr, "--mapping must be contig|interleave\n");
    exit(2);
}

// One token: per layer, gateup then down — the shipped alternation. Bursting one kernel and then the
// other measures two machines no token runs on (`prototype/mlp_gemv/results.md`).
void launch_token(const Config& c, int inject, int sync_each, hipStream_t stream) {
    const Opt& o = c.o;
    const Plan& p = c.p;
    if (!o.mlp) return;
    const int neg = kernel_neg(inject);
    const int short_by = inject == kInjShortGrid ? 1 : 0;
    const bool do_g = o.kernels != "down";
    const bool do_d = o.kernels != "gateup";
    const int map = map_of(o.mapping);
    if (is_fat(o)) {
        // One launch per kernel covering all `--layers` layers — the published burst's shape, and the
        // ONLY thing that differs from the thin arm: same bytes, same kernel, same mapping, same
        // rows/block, 24x the work per launch and 1/24 the launches. It walks the blob end to end,
        // which run_arm has checked is what the per-layer slices make when they are adjacent.
        if (do_g) {
            if (map == MAP_CONTIG)
                mlp_gateup<kGateIn, MAP_CONTIG><<<p.grid_gateup, p.threads, 0, stream>>>(
                    (const hip_bfloat16*)p.wgu[0], (const hip_bfloat16*)p.wup[0], p.x,
                    (hip_bfloat16*)p.act.data, nullptr, geom_rows_gate(o), ROWS_PER_BLOCK, neg);
            else
                mlp_gateup<kGateIn, MAP_INTERLEAVE><<<p.grid_gateup, p.threads, 0, stream>>>(
                    (const hip_bfloat16*)p.wgu[0], (const hip_bfloat16*)p.wup[0], p.x,
                    (hip_bfloat16*)p.act.data, nullptr, geom_rows_gate(o), ROWS_PER_BLOCK, neg);
            if (sync_each) CK(hipStreamSynchronize(stream));
        }
        if (do_d) {
            mlp_down<kDownIn><<<p.grid_down, p.threads, 0, stream>>>(
                (const hip_bfloat16*)p.wdn[0], p.xa, (hip_bfloat16*)p.delta.data,
                geom_rows_down(o), ROWS_PER_BLOCK, neg);
            if (sync_each) CK(hipStreamSynchronize(stream));
        }
        // A failed launch is otherwise invisible until some later call trips on the sticky error, and a
        // row that launches nothing times as ~0 ms — the failure `mlp_gemv`'s own loop guards against.
        CK(hipGetLastError());
        return;
    }
    for (int L = 0; L < o.layers; ++L) {
        hip_bfloat16* act = (hip_bfloat16*)(o.writes == "spread" ? p.act_slot[L] : p.act.data);
        hip_bfloat16* del = (hip_bfloat16*)(o.writes == "spread" ? p.delta_slot[L] : p.delta.data);
        if (do_g) {
            if (map == MAP_CONTIG)
                mlp_gateup<kGateIn, MAP_CONTIG><<<p.grid_gateup - short_by, p.threads, 0, stream>>>(
                    (const hip_bfloat16*)p.wgu[L], (const hip_bfloat16*)p.wup[L], p.x, act, nullptr,
                    kGateRows, ROWS_PER_BLOCK, neg);
            else
                mlp_gateup<kGateIn, MAP_INTERLEAVE><<<p.grid_gateup - short_by, p.threads, 0,
                                                      stream>>>(
                    (const hip_bfloat16*)p.wgu[L], (const hip_bfloat16*)p.wup[L], p.x, act, nullptr,
                    kGateRows, ROWS_PER_BLOCK, neg);
            if (sync_each) CK(hipStreamSynchronize(stream));
        }
        if (do_d) {
            mlp_down<kDownIn><<<p.grid_down - short_by, p.threads, 0, stream>>>(
                (const hip_bfloat16*)p.wdn[L], p.xa, del, kDownRows, ROWS_PER_BLOCK, neg);
            if (sync_each) CK(hipStreamSynchronize(stream));
        }
    }
}

// The interferer's bytes are billed exactly: `--if-read-mib`/`--if-write-mib` are per TOKEN, split
// across layers (or issued once), and each region is sized to exactly that many bytes, so the read
// account's expected value comes from the bill and not from the kernel. Every launch publishes into
// its OWN slice of `sums` (`if_blocks` doubles per launch, fixed on the host), so a launch that never
// ran, ran short, or ran twice cannot hide behind another launch's total. The write launches walk
// FORWARD through the arena (offset L*wbytes) instead of rewriting one hot spot, so the landing
// read-back proves the whole billed volume was stored exactly once.
void launch_interferer(Config& c, int inject, hipStream_t stream) {
    const Opt& o = c.o;
    Plan& p = c.p;
    const int parts = (o.if_sched == "once") ? 1 : o.layers;
    const size_t rbytes = mib_bytes(o.if_read_mib / parts);
    const size_t wbytes = mib_bytes(o.if_write_mib / parts);
    const size_t hbytes = mib_bytes(o.head_mib);
    const int blocks = p.if_blocks;
    // The expected checksum is derived from these, so it cannot pass by construction.
    p.if_read_billed = (rbytes && p.if_read.data) ? (o.if_sched == "once" ? rbytes * parts
                                                                           : rbytes * o.layers) : 0;
    p.if_write_billed =
        (wbytes && p.if_write.data) ? (o.if_sched == "once" ? wbytes * parts : wbytes * o.layers) : 0;
    p.head_billed = (hbytes && p.head.data) ? hbytes : 0;
    auto rd = [&](const void* base, size_t bytes) {
        const size_t n4 = bytes / 16;
        if (n4 == 0 || base == nullptr) return;
        const int seq = p.if_seq++;
        if ((size_t)(seq + 1) * blocks > p.sums_n) {
            fprintf(stderr, "ABORT: interferer launch %d exceeds the sums slab (%zu doubles).\n",
                    seq, p.sums_n);
            exit(2);
        }
        read_k<<<blocks, 256, 256 * sizeof(double), stream>>>(
            (const float4*)base, n4, p.sums + (size_t)seq * blocks, o.verify, inject);
    };
    auto wr = [&](size_t bytes, size_t offset) {
        const size_t n4 = bytes / 16;
        if (n4 == 0 || p.if_write.data == nullptr) return;
        write_k<<<blocks, 256, 0, stream>>>((float4*)((char*)p.if_write.data + offset), n4, kFillVal,
                                            inject);
    };
    if (o.if_sched == "once") {
        rd(p.if_read.data, rbytes * parts);
        wr(wbytes * parts, 0);
    } else {
        for (int L = 0; L < o.layers; ++L) {
            rd((char*)p.if_read.data + (size_t)L * rbytes, rbytes);
            wr(wbytes, (size_t)L * wbytes);
        }
    }
    rd(p.head.data, hbytes);      // the one-whole-token blob (lm_head stand-in)
}

}  // namespace
namespace {

/* ------------------------------------------------------------------- witnesses -------------- */

// Every element of a BF16 array was written. The buffer was pre-set to 0xFFFF in every byte, a bit
// pattern no store in this kernel can produce (every store is rnd()/RNE of a finite dot), so an
// unwritten slot is visible rather than merely unwitnessed.
bool all_written(const void* dev, size_t elems, const char* what = "?") {
    // The `what` label exists because a bare hipMemcpy failure here is undebuggable: it is the readback
    // of a buffer several arms size differently, and "invalid argument" without the pointer and the
    // element count is a dead end.
    if (elems == 0) return true;
    if (!dev) { fprintf(stderr, "all_written(%s): null device pointer\n", what); return false; }
    std::vector<uint16_t> buf(elems);
    const hipError_t e = hipMemcpy(buf.data(), dev, elems * 2, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        fprintf(stderr, "all_written(%s): hipMemcpy(%p, %zu elems = %zu B) -> %s\n", what, dev,
                elems, elems * 2, hipGetErrorString(e));
        return false;
    }
    size_t first = elems, n = 0;
    for (size_t i = 0; i < elems; ++i)
        if ((buf[i] & 0xFFFF) == 0xFFFF) { ++n; if (first == elems) first = i; }
    if (n) {
        // The count and the first index are the whole diagnostic: "all rows unwritten" is a launch that
        // never ran or a pointer that is not the one the kernel got, while "rows 3,584..86,015" is a
        // launch that covered only the first layer. Guessing between those two costs a rebuild.
        fprintf(stderr, "all_written(%s): %zu of %zu unwritten, first at %zu\n", what, n, elems, first);
        return false;
    }
    return true;
}

bool arena_all(const Blob& b, float expect) {
    if (b.bytes == 0) return true;
    std::vector<float> buf(b.bytes / 4);
    CK(hipMemcpy(buf.data(), b.data, b.bytes, hipMemcpyDeviceToHost));
    for (float f : buf)
        if (f != expect) return false;
    return true;
}

// The EXACT witness. x = one-hot(j) makes every dot exactly ONE product: bf(W[i][j]) * 1.0, which is
// that BF16 value unchanged, so the expected slot is a known weight word — NO tolerance, no
// summation-order question, no expf on the host, and nothing to re-band. A wrong word, a wrong row,
// a dropped word, or a store into the wrong slot all show up as a wrong 16-bit pattern. This is a
// GEOMETRY witness; it is not and does not replace `prototype/mlp_gemv/run.py`'s F1–F4 numerics gate.
//
// `tag_bits(row, j)` is a distinct finite non-zero BF16 per (row, column) and never 0xFFFF (the
// unwritten sentinel). The ROW has to be part of it: with a row-independent tag, `NEG_SWAPROW` — which
// permutes rows — is invisible because every row's expected answer is the same number. That is exactly
// how the first version of this witness scored swaprow as a false clean pass: a control's pattern must
// differ wherever the bug moves things.
// BF16 1.0 + k/256 is exactly representable for k in [0,255], so 0x3F80+k is a real, distinct, finite,
// non-zero BF16 for every k > 0.
uint16_t tag_bits(int row, int j) {
    const unsigned k = 1u + (unsigned)(((long)row * 37 + (long)j * 11) % 255);
    return (uint16_t)(0x3F80u + k);
}

bool exact_expect(const void* dev, size_t slots, int which, int j) {
    std::vector<uint16_t> got(slots);
    CK(hipMemcpy(got.data(), dev, slots * 2, hipMemcpyDeviceToHost));
    for (size_t s = 0; s < slots; ++s) {
        // gateup's `pre` is [gate rows][up rows] and down's `out` is one slot per row. The modulo is
        // per-LAYER, not per-arm, because a `--geometry fat` launch writes layers x rows slots: layer
        // L's rows sit at [L*kGateRows, (L+1)*kGateRows) and each layer carries the same per-row tag,
        // so slot s belongs to row s % rows_per_layer for both geometries.
        const int row = which ? (int)(s % (size_t)kDownRows) : (int)(s % (size_t)kGateRows);
        if (got[s] != tag_bits(row, j)) return false;
    }
    return true;
}

// The weight pattern that makes `exact_expect` true: every word is BF16 1.0 EXCEPT column j of every
// row, which is `tag_bits(row, j)`. Uploading it is the only host->device copy a witness case needs.
void upload_pattern(int which, int j, void* dst) {
    const int in = which ? kDownIn : kGateIn;
    const int rows = which ? kDownRows : kGateRows;
    std::vector<uint16_t> h((size_t)rows * in);
    for (int i = 0; i < rows; ++i) {
        const uint16_t tag = tag_bits(i, j);
        for (int k = 0; k < in; ++k) h[(size_t)i * in + k] = (k == j) ? tag : kBF16One;
    }
    CK(hipMemcpy(dst, h.data(), h.size() * 2, hipMemcpyHostToDevice));
}

/* ---------------------------------------------------------------- witness case table -------- */

struct DevCase {
    const char* name;
    int which;              // 0 gateup shape, 1 down shape, -1 interferer accounts
    int j;                  // one-hot position (unused by the interferer cases)
    int inject;
    int expect_exact;       // 1 = the exact witness MUST fail
    int expect_landing;     // 1 = the landing witness MUST fail
    int expect_read;        // 1 = the interferer read account MUST fail
    int expect_write;       // 1 = the interferer write landing MUST fail
    int expect_guard;       // 1 = a guard band MUST be broken
    const char* why;
};

// The case list is DERIVED: the one-hot positions come from `words = in/8` and the warp width (see
// `onehot_positions`), and each injection's must-catch set is the set of geometries in which the bug
// can change an observable at all. A hand-listed must-catch set rots into "everything" the moment the
// fixture changes (AGENTS.md), so every row states its `why` and `geom_selftest` rejects an injection
// whose must-catch set is empty.
std::vector<DevCase> case_table() {
    std::vector<DevCase> t;
    for (int which = 0; which < 2; ++which) {
        const int in = which ? kDownIn : kGateIn;
        for (int j : onehot_positions(in)) {
            t.push_back({"clean", which, j, kInjNone, 0, 0, 0, 0, 0,
                         "control: a correct walk must satisfy every witness"});
            // NEG_SWAPROW permutes the output slots; EVERY slot is still written, so `landing` is
            // blind by construction and only the exact pre-image can see a permutation.
            t.push_back({"swaprow", which, j, kInjSwapRow, 1, 0, 0, 0, 0,
                         "rows^1 moves every answer: exact catches, landing structurally cannot"});
            // NEG_TAIL drops the LAST word of K. Only a one-hot sitting in that word can see it: for
            // every other j the dropped word multiplies a zero, so the must-catch set is exactly
            // {j == in-1} — read off words = in/8, not hand-picked.
            t.push_back({"tail", which, j, kInjTail, j == in - 1 ? 1 : 0, 0, 0, 0, 0,
                         "NEG_TAIL drops word in/8-1: only j=in-1 can see it"});
            // NEG_HALF keeps words [0, in/16), so the derived blind set is j < in/2.
            t.push_back({"half", which, j, kInjHalf, j >= in / 2 ? 1 : 0, 0, 0, 0, 0,
                         "NEG_HALF keeps the first half of K: only j >= in/2 can see it"});
            // NEG_F32ACT changes `act`, which no witness here reads (`pre` is written from g/u BEFORE
            // the epilogue; `landing` sees only that a slot was written). Derived blindness, asserted
            // so that no witness is claimed to be omniscient.
            t.push_back({"f32act", which, j, kInjF32Act, 0, 0, 0, 0, 0,
                         "NEG_F32ACT changes act only: both witnesses are blind by construction"});
        }
    }
    // Rig-level injections: about THIS file's geometry and accounts, not the kernel's.
    t.push_back({"short_grid", 0, 0, kInjShortGrid, 1, 1, 0, 0, 0,
                 "one block fewer: the last rows are never written, so both witnesses catch it"});
    t.push_back({"short_grid", 1, 0, kInjShortGrid, 1, 1, 0, 0, 0,
                 "one block fewer: the last rows are never written, so both witnesses catch it"});
    t.push_back({"short_read", -1, -1, kInjShortRead, 0, 0, 1, 0, 0,
                 "interferer reads one float4 short: only the read account can see it"});
    t.push_back({"stray_read", -1, -1, kInjStrayRead, 0, 0, 1, 0, 0,
                 "one float4 over the read edge: the poison in the band moves the sum. A READ cannot "
                 "break a band, so the band is a write witness only and the account is the catcher"});
    t.push_back({"miss_write", -1, -1, kInjMissWrite, 0, 0, 0, 1, 0,
                 "last block stores nothing: only the arena landing read-back can see it"});
    t.push_back({"stray_write", -1, -1, kInjStrayWrite, 0, 0, 0, 0, 1,
                 "one float4 over the arena edge: the guard band is the only witness"});
    return t;
}

int must_catch(const DevCase& dc) {
    return dc.expect_exact || dc.expect_landing || dc.expect_read || dc.expect_write ||
           dc.expect_guard;
}

// One device case, in its own process. Prints `CASE ...`; returns the number of anomalies, where an
// anomaly includes a control that PASSED when the table said it must be caught — the vacuous
// direction, which is the one a must-catch-only harness never sees.
int run_dev_case(const DevCase& dc) {
    int anomalies = 0;
    if (dc.which >= 0) {
        const int in = dc.which ? kDownIn : kGateIn;
        const int rows = dc.which ? kDownRows : kGateRows;
        const size_t blob = (size_t)rows * in * 2;
        void *dw = nullptr, *du = nullptr;
        float* dx = nullptr;
        Blob dact = alloc_region("wit.act", (size_t)kGateRows * 2, 'F');
        Blob dpre = alloc_region("wit.pre", (size_t)kGateRows * 2 * 2, 'F');
        Blob ddout = alloc_region("wit.out", (size_t)kDownRows * 2, 'F');
        CK(hipMalloc(&dw, blob));
        CK(hipMalloc(&du, blob));
        CK(hipMalloc((void**)&dx, (size_t)in * 4));
        upload_pattern(dc.which, dc.j, dw);
        upload_pattern(dc.which, dc.j, du);
        std::vector<float> hx(in, 0.f);
        hx[dc.j] = 1.0f;
        CK(hipMemcpy(dx, hx.data(), (size_t)in * 4, hipMemcpyHostToDevice));
        const int neg = kernel_neg(dc.inject);
        const int short_by = dc.inject == kInjShortGrid ? 1 : 0;
        const int grid = dc.which
                             ? bounds_for(kDownRows, kDownIn, ROWS_PER_BLOCK, 0, kDownBlob).blocks
                             : bounds_for(kGateRows, kGateIn, ROWS_PER_BLOCK, 0, kGateBlob).blocks;
        if (dc.which == 0) {
            mlp_gateup<kGateIn, MAP_CONTIG><<<grid - short_by, rpb_threads()>>>(
                (const hip_bfloat16*)dw, (const hip_bfloat16*)du, dx, (hip_bfloat16*)dact.data,
                (hip_bfloat16*)dpre.data, kGateRows, ROWS_PER_BLOCK, neg);
        } else {
            mlp_down<kDownIn><<<grid - short_by, rpb_threads()>>>(
                (const hip_bfloat16*)dw, dx, (hip_bfloat16*)ddout.data, kDownRows, ROWS_PER_BLOCK,
                neg);
        }
        CK(hipDeviceSynchronize());
        // WITNESS exact. `short_grid` needs no special case: the unwritten slots still read 0xFFFF,
        // which is not a tag, so exact and landing both fail on their own.
        const int exact_slots = dc.which ? kDownRows : 2 * kGateRows;
        const int exact_fail =
            !exact_expect(dc.which ? (const void*)ddout.data : (const void*)dpre.data,
                          (size_t)exact_slots, dc.which, dc.j);
        // WITNESS landing.
        const int landing_fail = dc.which
                                     ? !all_written(ddout.data, kDownRows, "wit.out")
                                     : (!all_written(dact.data, kGateRows, "wit.act") ||
                                        !all_written(dpre.data, 2 * (size_t)kGateRows, "wit.pre"));
        const int guards_ok = guard_intact(dact) && guard_intact(dpre) && guard_intact(ddout);
        const int ok = exact_fail == dc.expect_exact && landing_fail == dc.expect_landing && guards_ok;
        if (!ok) ++anomalies;
        printf("CASE name=%s which=%d j=%d inject=%s exact_fail=%d expect_exact=%d landing_fail=%d "
               "expect_landing=%d guards_intact=%d as_expected=%d why=%s\n",
               dc.name, dc.which, dc.j, inject_name(dc.inject), exact_fail, dc.expect_exact,
               landing_fail, dc.expect_landing, guards_ok, ok, dc.why);
        (void)hipFree(dw);
        (void)hipFree(du);
        (void)hipFree(dx);
        for (Blob* b : {&dact, &dpre, &ddout})
            if (b->p) (void)hipFree(b->p);
    } else {
        // The interferer accounts. One read region and one write arena, each sized to the billed
        // bytes, and the expected checksum comes from the bill rather than from the kernel.
        const double mib = 8.0;
        Blob rd = alloc_region("if.read", mib_bytes(mib), '0');
        Blob wr = alloc_region("if.write", mib_bytes(mib), '0');
        fill_k<<<256, 256>>>((float4*)rd.data, rd.bytes / 16, kFillVal);
        CK(hipDeviceSynchronize());
        const int blocks = 256;
        double* sums = nullptr;
        CK(hipMalloc((void**)&sums, blocks * sizeof(double)));
        CK(hipMemset(sums, 0, blocks * sizeof(double)));
        const size_t n4r = rd.bytes / 16;
        read_k<<<blocks, 256, 256 * sizeof(double)>>>((const float4*)rd.data, n4r, sums, 1,
                                                      dc.inject);
        CK(hipDeviceSynchronize());
        std::vector<double> hs(blocks);
        CK(hipMemcpy(hs.data(), sums, blocks * sizeof(double), hipMemcpyDeviceToHost));
        double got = 0;
        for (double d : hs) got += d;
        const double want = 4.0 * (double)n4r;         // every float4 holds four 1.0f
        const int read_fail = (got != want);
        const size_t n4w = wr.bytes / 16;
        write_k<<<blocks, 256>>>((float4*)wr.data, n4w, kFillVal, dc.inject);
        CK(hipDeviceSynchronize());
        const int write_fail = !arena_all(wr, kFillVal);
        const int guard_fail = !guard_intact(rd) || !guard_intact(wr);
        const int ok = read_fail == dc.expect_read && write_fail == dc.expect_write &&
                       guard_fail == dc.expect_guard;
        if (!ok) ++anomalies;
        printf("CASE name=%s which=-1 j=-1 inject=%s read_sum=%.0f want=%.0f read_fail=%d "
               "expect_read=%d write_fail=%d expect_write=%d guard_fail=%d expect_guard=%d "
               "as_expected=%d why=%s\n",
               dc.name, inject_name(dc.inject), got, want, read_fail, dc.expect_read, write_fail,
               dc.expect_write, guard_fail, dc.expect_guard, ok, dc.why);
        (void)hipFree(sums);
        if (rd.p) (void)hipFree(rd.p);
        if (wr.p) (void)hipFree(wr.p);
    }
    fflush(stdout);
    return anomalies;
}

/* --------------------------------------------------------------- CPU geometry selftest ------ */

/*
 * geom_selftest — the checks that need no device, plus the anti-vacuity rules for the witness table.
 * A geometry check that cannot fail is the bug: `mixed_bw`'s first `--selftest-geom` printed ok=1 for
 * exactly the configurations whose kernel then read past its own buffer and faulted. So every
 * assertion prints the two numbers it compared, and the table is REJECTED if any injection present in
 * it has an empty must-catch set, or if it declares no derived blindness at all.
 */
int geom_selftest() {
    int fails = 0;
    auto expect = [&](int ok, const char* what, long long got, long long want) {
        printf("GEOM %-52s got=%lld want=%lld ok=%d\n", what, got, want, ok);
        if (!ok) ++fails;
    };
    const Bounds g = bounds_for(kGateRows, kGateIn, ROWS_PER_BLOCK, 448, kGateBlob);
    const Bounds d = bounds_for(kDownRows, kDownIn, ROWS_PER_BLOCK, 448, kDownBlob);
    expect(g.covers_rows, "gateup grid covers every row", g.covers_rows, 1);
    expect(d.covers_rows, "down grid covers every row", d.covers_rows, 1);
    expect(g.one_grid_pass, "gateup wavefronts == rows (one grid pass)", g.one_grid_pass, 1);
    expect(d.one_grid_pass, "down wavefronts == rows (one grid pass)", d.one_grid_pass, 1);
    expect(g.exact_bytes, "gateup blob bytes == rows*in*2", g.exact_bytes, 1);
    expect(d.exact_bytes, "down blob bytes == rows*in*2", d.exact_bytes, 1);
    expect(g.last_weight_byte == (long long)kGateBlob, "gateup last byte touched == blob end",
           g.last_weight_byte, (long long)kGateBlob);
    expect(d.last_weight_byte == (long long)kDownBlob, "down last byte touched == blob end",
           d.last_weight_byte, (long long)kDownBlob);
    expect(g.wavefronts == (long long)g.blocks * ROWS_PER_BLOCK, "gateup wavefronts == blocks*rpb",
           g.wavefronts, (long long)g.blocks * ROWS_PER_BLOCK);
    expect(g.blocks == kGateRows / ROWS_PER_BLOCK, "gateup blocks == rows/ROWS_PER_BLOCK (448)",
           g.blocks, kGateRows / ROWS_PER_BLOCK);
    expect(d.blocks == kDownRows / ROWS_PER_BLOCK, "down blocks == rows/ROWS_PER_BLOCK (128)",
           d.blocks, kDownRows / ROWS_PER_BLOCK);
    // The placement arm is only admissible if all three layouts bill the SAME MLP bytes; otherwise a
    // layout row is a byte difference wearing a costume.
    const double mlp_bytes = 24.0 * kGateBlob * 2 + 24.0 * kDownBlob;
    double billed[3];
    for (int i = 0; i < 3; ++i) {
        Config c;
        c.o.layout = (i == 0) ? "blob3" : (i == 1) ? "blob72" : "shipped72";
        c.o.gap_mib = 42.0;
        bill(c);
        billed[i] = c.p.mlp_read_mib * MIB;
    }
    expect(billed[0] == billed[1] && billed[1] == billed[2], "all three layouts bill identical "
           "MLP read bytes", (long long)billed[0], (long long)billed[2]);
    expect(mlp_bytes == billed[0], "billed MLP read bytes == 24*(2*gate + down)",
           (long long)billed[0], (long long)mlp_bytes);
    // The interferer's expected checksum is derived from billed bytes, not from a kernel.
    expect(mib_bytes(8.0) % 16 == 0, "interferer region is a whole number of float4s",
           (long long)(mib_bytes(8.0) % 16), 0);
    // Case table: derived, and every injection present must be DECLARED as either a catch control or
    // a blindness control. A catch control with an empty must-catch set is the vacuous negative test
    // AGENTS.md records twice; a blindness control is the opposite — a row that proves a witness is
    // not omniscient — and it must be empty on purpose, not by omission.
    const std::vector<DevCase> t = case_table();
    // The classification is a decision on the record, not an inference: an injection is either a CATCH
    // control (it must be caught somewhere in the derived case list) or a BLINDNESS control (it is
    // required to be invisible here, because its catcher lives in another gate — `NEG_F32ACT` is a
    // numerics case and `prototype/mlp_gemv/run.py` is that gate). Anything else is a defect.
    const int blindness_controls[] = {kInjF32Act};
    for (int inj = 1; inj <= 9; ++inj) {
        int must = 0, blind = 0;
        for (const DevCase& dc : t)
            if (dc.inject == inj) {
                if (must_catch(dc)) ++must;
                else ++blind;
            }
        if (must + blind == 0) {
            printf("GEOM injection %-12s not present in the table (unused, not a defect)\n",
                   inject_name(inj));
            continue;
        }
        const bool declared_blind =
            std::find(blindness_controls, blindness_controls + 1, inj) != blindness_controls + 1;
        if (declared_blind) {
            expect(blind > 0 && must == 0,
                   (std::string("blindness control ") + inject_name(inj) +
                    " is invisible everywhere in this rig").c_str(), must + blind, blind);
        } else {
            expect(must > 0, (std::string("catch control ") + inject_name(inj) +
                              " has a non-empty must-catch set").c_str(), must, must);
        }
        printf("GEOM   injection %-12s kind=%s rows=%d must-catch=%d must-not-catch=%d\n",
               inject_name(inj), declared_blind ? "blindness" : "catch", must + blind, must, blind);
    }
    // And the derived must-catch SETS, not just their non-emptiness: `tail` may catch exactly the
    // rows with j == in-1, `half` exactly those with j >= in/2. Checked against the geometry, so a
    // hand-edited table that quietly widened a catch set fails here.
    int want_tail = 0, want_half = 0;
    for (int which = 0; which < 2; ++which) {
        const int in = which ? kDownIn : kGateIn;
        for (int j : onehot_positions(in)) {
            if (j == in - 1) ++want_tail;
            if (j >= in / 2) ++want_half;
        }
    }
    int got_tail = 0, got_half = 0;
    for (const DevCase& dc : t) {
        if (dc.inject == kInjTail && must_catch(dc)) ++got_tail;
        if (dc.inject == kInjHalf && must_catch(dc)) ++got_half;
    }
    expect(got_tail == want_tail, "NEG_TAIL's must-catch set == {j == in-1} in both shapes", got_tail,
           want_tail);
    expect(got_half == want_half, "NEG_HALF's must-catch set == {j >= in/2} in both shapes", got_half,
           want_half);
    int blind_rows = 0;
    for (const DevCase& dc : t)
        if (dc.inject != kInjNone && !must_catch(dc)) ++blind_rows;
    expect(blind_rows > 0, "the table declares derived blindness (no omniscient witness)",
           blind_rows, blind_rows);
    printf("GEOM cases=%d one-hot positions: gateup=%zu down=%zu (words=%d/%d)\n", (int)t.size(),
           onehot_positions(kGateIn).size(), onehot_positions(kDownIn).size(), kGateIn / 8,
           kDownIn / 8);

    /*
     * Geometry checks, added after the first window aborted on canary C2. `--geometry fat` tests ONE
     * claim: that the published 2.275 ms floor is a launch-granularity artefact. For that to be a
     * controlled comparison the fat arm must be the same workload at a different granularity and
     * nothing else — same bytes billed, same kernel, only rows-per-launch and launches-per-token
     * different — and it must be REFUSED where its contiguity premise is false.
     */
    {
        Config thin, fat;
        thin.o.layers = fat.o.layers = 24;
        thin.o.geometry = "thin";
        fat.o.geometry = "fat";
        bill(thin);
        bill(fat);
        expect(fat.p.mlp_read_mib == thin.p.mlp_read_mib, "fat bills identical MLP read bytes to thin",
               (long long)(fat.p.mlp_read_mib * 1000), (long long)(thin.p.mlp_read_mib * 1000));
        expect(fat.p.mlp_write_mib == thin.p.mlp_write_mib,
               "fat bills identical MLP write bytes to thin",
               (long long)(fat.p.mlp_write_mib * 1000), (long long)(thin.p.mlp_write_mib * 1000));
        expect(thin.p.launches_mlp, "thin = 48 launches/token (shipped)", thin.p.launches_mlp, 48);
        expect(fat.p.launches_mlp, "fat = 2 launches/token (the published burst)", fat.p.launches_mlp,
               2);
        expect(fat.p.grid_gateup, "fat gateup grid = 86,016 rows / 8 = the published 10,752",
               fat.p.grid_gateup, 10752);
        expect(fat.p.grid_down, "fat down grid = 24,576 rows / 8 = the published 3,072",
               fat.p.grid_down, 3072);
        expect(thin.p.grid_gateup == 448 && thin.p.grid_down == 128,
               "thin grids stay 448/128 (one row, as shipped)",
               (long long)thin.p.grid_gateup * 10000 + thin.p.grid_down, 4480128LL);
        char why[320];
        Config bad = fat;
        bad.o.layout = "blob72";
        expect(fat_refusable(bad.o, why, sizeof why) != 0,
               "fat is REFUSED on blob72, where the slices are separate hipMallocs", 0, 0);
        Config bad2 = fat;
        bad2.o.writes = "spread";
        expect(fat_refusable(bad2.o, why, sizeof why) != 0,
               "fat is REFUSED with --writes spread (fat already spreads act)", 0, 0);
        Config fine = fat;
        fine.o.geometry = "thin";
        fine.o.writes = "spread";
        expect(fat_refusable(fine.o, why, sizeof why) == 0, "the fat guards never fire on a thin arm",
               0, 0);
    }

    // The must-land set is DERIVED and asserted NON-EMPTY for every arm — the anti-vacuity half of
    // defect 1: a verify pass that expects nothing passes by construction.
    {
        const std::string ks[] = {"both", "gateup", "down"};
        int empty = 0, wrong = 0;
        for (const std::string& k : ks) {
            Opt o;
            o.kernels = k;
            const int m = must_land_mask(o);
            if (!m) ++empty;
            const int want = k == "gateup" ? kLandAct
                                           : (k == "down" ? kLandDelta : (kLandAct | kLandDelta));
            if (m != want) ++wrong;
        }
        expect(empty == 0, "no arm has an EMPTY must-land set", empty, 0);
        expect(wrong == 0, "gateup lands act only, down lands delta only, both lands both", wrong, 0);
        Opt nomlp;
        nomlp.mlp = 0;
        expect(must_land_mask(nomlp) == 0,
               "--mlp 0 expects nothing and must print no_launch=1, not ms", must_land_mask(nomlp), 0);
    }
    printf("SELFTEST-GEOM %s fails=%d\n", fails ? "FAILED" : "ok", fails);
    fflush(stdout);
    return fails;
}

// One child per case: a faulting kernel poisons the context and every later case in that process
// silently never ran, which is how `mixed_bw`'s first `--selftest` shipped half-broken.
int selftest_device() {
    const std::vector<DevCase> t = case_table();
    int anomalies = 0, children_bad = 0, ran = 0;
    for (size_t i = 0; i < t.size(); ++i) {
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) {
            const int a = run_dev_case(t[i]);
            _exit(a > 125 ? 125 : a);
        }
        if (pid < 0) {
            fprintf(stderr, "fork failed\n");
            return 1;
        }
        int status = 0;
        waitpid(pid, &status, 0);
        ++ran;
        if (!WIFEXITED(status)) {
            printf("CASE child %zu (%s/%s) died without exiting cleanly\n", i, t[i].name,
                   inject_name(t[i].inject));
            ++children_bad;
            continue;
        }
        anomalies += WEXITSTATUS(status);
    }
    if (ran != (int)t.size()) {
        printf("SELFTEST case-count MISMATCH ran=%d table=%zu\n", ran, t.size());
        ++anomalies;
    }
    const int fails = anomalies + children_bad;
    printf("SELFTEST %s fails=%d cases=%d children_bad=%d\n", fails ? "FAILED" : "ok", fails, ran,
           children_bad);
    fflush(stdout);
    return fails;
}

/* --------------------------------------------------------------------- the timed run -------- */

double ms_since(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

/*
 * One iteration of the arm, timed on the segment the row is about.
 *
 * `--if-when before` submits the interferer and DRAINS it before the timed window opens, so the
 * interferer's own milliseconds are outside the measurement by construction: what is measured is the
 * MLP's time as a function of the STATE the interferer left behind. That is the only form candidate
 * (a) can take for a family the stage clock measures drained, and it is what makes the dose slope
 * interpretable as carry-over rather than as bandwidth sharing.
 *
 * `--if-when concurrent` runs the MLP on stream A and the interferer on stream B (both non-default,
 * because the legacy default stream implicitly synchronizes with every blocking stream and would
 * serialize the pair into a null result). There the observable is the UNION time, so the estimator is
 * the cross-term t(A,B) − t(A) − t(B) across three rows, and the number is a bound on contention, not
 * a reproduction of the shipped measure.
 */
double timed_iteration(Config& c, hipStream_t sA, hipStream_t sB) {
    const Opt& o = c.o;
    Plan& p = c.p;
    const int sync_each = (o.drain == "each");
    const bool has_if = (o.if_read_mib > 0 || o.if_write_mib > 0 || o.head_mib > 0);
    if (has_if && o.if_when == "before") {
        p.if_seq = 0;
        launch_interferer(c, kInjNone, 0);
        CK(hipDeviceSynchronize());
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (has_if && o.if_when == "concurrent") {
        launch_token(c, kInjNone, 0, sA);
        p.if_seq = 0;
        launch_interferer(c, kInjNone, sB);
        CK(hipDeviceSynchronize());
        return ms_since(t0);
    }
    // `--token-burst N` submits N tokens between syncs. At N=1 (the default, and the shipped shape)
    // every token is its own drained burst, which is what the stage clock's family row is too. At N>1
    // the queue never drains between tokens, which is how the published 2.275 ms burst was taken: 30
    // iterations submitted, ONE sync. Whether R0's 48 launches are being drained despite
    // `--drain burst` is therefore not a guess to argue about — it is the difference between these
    // two rows, and it is measured, not assumed.
    for (int t = 0; t < o.token_burst; ++t) launch_token(c, kInjNone, sync_each, 0);
    CK(hipDeviceSynchronize());
    return ms_since(t0) / (double)o.token_burst;
}

/*
 * The untimed witness pass that must accompany every timed row. Each account is pre-set to a value the
 * writer never stores, then the arm runs once, then the bytes are read back: a millisecond figure from
 * a run whose bytes were not accounted is a measurement of something unknown.
 */
std::string verify_pass(Config& c) {
    const Opt& o = c.o;
    Plan& p = c.p;
    std::vector<std::string> bad;
    auto add = [&](const char* s) { bad.push_back(s); };
    if (p.if_read.bytes) fill_k<<<256, 256>>>((float4*)p.if_read.data, p.if_read.bytes / 16, kFillVal);
    if (p.head.bytes) fill_k<<<256, 256>>>((float4*)p.head.data, p.head.bytes / 16, kFillVal);
    if (p.if_write.bytes) CK(hipMemset(p.if_write.data, 0, p.if_write.bytes));   // 0.0f: never stored
    if (o.mlp) {
        CK(hipMemset(p.act.data, 0xFF, p.act.bytes));
        CK(hipMemset(p.delta.data, 0xFF, p.delta.bytes));
        if (o.writes == "spread") {
            CK(hipMemset(p.act_spread.data, 0xFF, p.act_spread.bytes));
            CK(hipMemset(p.delta_spread.data, 0xFF, p.delta_spread.bytes));
        }
    }
    CK(hipMemset(p.sums, 0, p.sums_n * sizeof(double)));
    CK(hipDeviceSynchronize());
    p.if_seq = 0;
    launch_interferer(c, kInjNone, 0);
    launch_token(c, kInjNone, 0, 0);
    CK(hipDeviceSynchronize());

    if (o.verify && p.if_read_billed + p.head_billed > 0) {
        std::vector<double> hs(p.sums_n);
        CK(hipMemcpy(hs.data(), p.sums, p.sums_n * sizeof(double), hipMemcpyDeviceToHost));
        double got = 0;
        for (size_t i = 0; i < p.sums_n; ++i) got += hs[i];
        const double want = 4.0 * (double)((p.if_read_billed + p.head_billed) / 16);
        if (got != want) {
            char buf[128];
            snprintf(buf, sizeof buf, "FAIL_reads_short(got=%.0f want=%.0f)", got, want);
            bad.push_back(buf);
        }
        if (!guard_intact(p.if_read) || !guard_intact(p.head)) add("FAIL_guard_read");
    }
    if (o.verify && p.if_write_billed > 0) {
        if (!arena_all(p.if_write, kFillVal)) add("FAIL_writes_missing");
        if (!guard_intact(p.if_write)) add("FAIL_guard_write");
    }
    if (o.mlp) {
        const int land = must_land_mask(o);
        const size_t act_elems = (size_t)geom_rows_gate(o);
        const size_t delta_elems = (size_t)geom_rows_down(o);
        // Structural, not cosmetic: reading `elems` from a buffer sized for another geometry either
        // faults or silently reads someone else's bytes, and the first fat arm of this rig did the
        // latter for `act` and passed it vacuously. The witness must not be able to look past the region.
        if (act_elems * 2 > p.act.bytes) add("FAIL_act_buffer_smaller_than_geometry");
        if (delta_elems * 2 > p.delta.bytes) add("FAIL_delta_buffer_smaller_than_geometry");
        int checked = 0;
        if (o.writes == "shared") {
            if (land & kLandAct) {
                ++checked;
                if (!all_written(p.act.data, act_elems, "act")) add("FAIL_act_unwritten");
            }
            if (land & kLandDelta) {
                ++checked;
                if (!all_written(p.delta.data, delta_elems, "delta")) add("FAIL_delta_unwritten");
            }
        } else {
            int short_act = 0, short_del = 0;
            for (int i = 0; i < o.layers; ++i) {
                if ((land & kLandAct) && !all_written(p.act_slot[i], kGateRows, "act.slot")) ++short_act;
                if ((land & kLandDelta) && !all_written(p.delta_slot[i], kDownRows, "delta.slot"))
                    ++short_del;
            }
            checked += ((land & kLandAct) ? 1 : 0) + ((land & kLandDelta) ? 1 : 0);
            if (short_act) add("FAIL_act_slots_unwritten");
            if (short_del) add("FAIL_delta_slots_unwritten");
        }
        // A verify pass that expected nothing would pass by construction, which is the same empty-list
        // failure as a parser that matches zero records and prints PASS. Say what was expected.
        if (!checked) add("FAIL_nothing_expected");
        printf("VERIFY land=%s%s elems=%zu,%zu checked=%d geom=%s\n",
               (land & kLandAct) ? "act" : "-", (land & kLandDelta) ? "+delta" : "-no-delta",
               act_elems, delta_elems, checked, o.geometry.c_str());
        fflush(stdout);
        if (!guard_intact(p.act) || !guard_intact(p.delta)) add("FAIL_guard_act");
        if (o.writes == "spread" && (!guard_intact(p.act_spread) || !guard_intact(p.delta_spread)))
            add("FAIL_guard_spread");
        int bad_w = 0;
        for (const Blob& b : p.allocs)
            if (b.name.size() > 2 && b.name[0] == 'L' && !guard_intact(b)) ++bad_w;
        if (bad_w) add("FAIL_guard_weight");
    }
    // `verify=okFAIL_act_unwritten,...` was what the first window actually printed, and a reader (or a
    // `grep -c verify=ok`) scores that as ok. Either it is ok or it is the list of what failed.
    std::string s;
    for (size_t i = 0; i < bad.size(); ++i) s += (i ? "," : "") + bad[i];
    if (s.empty()) s = "ok";
    return s;
}

int run_arm(Config& c) {
    Opt& o = c.o;
    Plan& p = c.p;
    warp_width();      // query now, once, before anything bills or prints geometry
    if (o.if_when == "concurrent" && o.drain == "each") {
        fprintf(stderr,
                "ABORT: --if-when concurrent with --drain each is a contradiction: syncing every "
                "launch serializes the two streams and measures nothing. Use --drain burst.\n");
        return 2;
    }
    if (o.writes == "spread" && o.write_mul < 1) {
        fprintf(stderr, "ABORT: --writes spread needs --write-mul >= 1 (MiB of stride per layer).\n");
        return 2;
    }
    char why[320];
    if (int rc = fat_refusable(o, why, sizeof why)) {
        fprintf(stderr, "ABORT: %s\n", why);
        return rc;
    }
    if (o.if_when == "concurrent" && o.token_burst != 1) {
        fprintf(stderr,
                "ABORT: --if-when concurrent with --token-burst N>1 folds two variables into one row: "
                "the concurrency arm's observable is the union time of ONE token against the "
                "interferer. Keep --token-burst 1 there.\n");
        return 2;
    }
    if (o.token_burst < 1) {
        fprintf(stderr, "ABORT: --token-burst must be >= 1.\n");
        return 2;
    }
    build_weights(c);
    if (is_fat(o)) {
        // The fat launch walks the blob end to end, so the claim "those 24 slices are contiguous"
        // must be checked against the pointers actually built, not assumed from the layout name.
        int bad = 0;
        for (int L = 1; L < o.layers; ++L) {
            if ((char*)p.wgu[L] != (char*)p.wgu[0] + (size_t)L * kGateBlob) ++bad;
            if ((char*)p.wup[L] != (char*)p.wup[0] + (size_t)L * kGateBlob) ++bad;
            if ((char*)p.wdn[L] != (char*)p.wdn[0] + (size_t)L * kDownBlob) ++bad;
        }
        if (bad) {
            fprintf(stderr,
                    "ABORT: --geometry fat needs every layer of one tensor adjacent inside one "
                    "allocation and %d pointer checks failed; the single rows=%d launch would read "
                    "the wrong bytes and still time plausibly.\n", bad, geom_rows_gate(o));
            return 2;
        }
    }
    build_rest(c);
    bill(c);
    print_plan(c);
    if (is_fat(o)) {
        printf("PLAN fat geom_contig=1 rows_gate=%d rows_down=%d grid_gateup=%d grid_down=%d "
               "launches_mlp=%d (the published burst's shape: 24x the work per launch, 1/24 the "
               "launches, same bytes)\n", geom_rows_gate(o), geom_rows_down(o), p.grid_gateup,
               p.grid_down, p.launches_mlp);
        fflush(stdout);
    }
    hipStream_t sA = 0, sB = 0;
    if (o.if_when == "concurrent") {
        CK(hipStreamCreateWithFlags(&sA, hipStreamNonBlocking));
        CK(hipStreamCreateWithFlags(&sB, hipStreamNonBlocking));
    }
    for (int i = 0; i < 2; ++i) timed_iteration(c, sA, sB);      // warm-up, outside the timed region
    std::vector<double> reps;
    for (int r = 0; r < o.reps; ++r) {
        double acc = 0;
        for (int i = 0; i < o.iters; ++i) acc += timed_iteration(c, sA, sB);
        reps.push_back(acc / o.iters);
    }
    const std::string v = o.verify ? verify_pass(c) : "skipped";
    const double lo = *std::min_element(reps.begin(), reps.end());
    const double hi = *std::max_element(reps.begin(), reps.end());
    std::sort(reps.begin(), reps.end());
    const double med = (reps.size() % 2) ? reps[reps.size() / 2]
                                         : (reps[reps.size() / 2 - 1] + reps[reps.size() / 2]) / 2.0;
    const double read_mib = p.mlp_read_mib + p.if_read_mib;
    const double all_mib = read_mib + p.mlp_write_mib + p.if_write_mib;
    const double gbs_mlp = p.mlp_read_mib / med * 1000.0;
    const double gbs_all = all_mib / med * 1000.0;
    const int suspect = (gbs_mlp > o.suspect_gbs) || (gbs_all > o.suspect_gbs);
    if (p.launches_mlp == 0) {
        // DEFECT 3, from the first window: row R9 (`--mlp 0`) printed `0.0001 ms`. An arm whose timed
        // window launches nothing has NO milliseconds — this part's timer returns ~0 for sub-ms work
        // (AGENTS.md: `hipEventElapsedTime` returns exactly 0 below ~1 ms), and a reader tabulates
        // 0.0001 as "essentially free" and it becomes a term in a subtraction. The arm is still worth
        // running: with no MLP in flight it is the witness that the interferer's bytes are accounted on
        // their own. So it says what it is and prints no number.
        printf("NO-TIMING tag=%s no_launch=1 mlp=0 geom=%s if_r_mib=%.3f if_w_mib=%.3f "
               "head_mib=%.3f verify=%s reason=timed_window_launches_nothing (a sub-ms empty window "
               "reads as ~0 on this part and would be tabulated as free)\n",
               o.tag.c_str(), o.geometry.c_str(), o.if_read_mib, o.if_write_mib, o.head_mib,
               v.c_str());
    } else if (o.time) {
        printf("INSITU tag=%s when=%s sched=%s layout=%s drain=%s kernels=%s mapping=%s writes=%s"
               " mul=%d mlp=%d geom=%s token_burst=%d pages=%s if_r_mib=%.3f if_w_mib=%.3f"
               " head_mib=%.3f ms_per_token=%.4f"
               " spread=%.4f..%.4f us_per_mlp_launch=%.2f mlp_read_mib=%.3f total_mib=%.3f"
               " gbs_mlp_demanded=%.1f gbs_all_demanded=%.1f suspect=%d label=demanded_not_dram=%d"
               " verify=%s alloc_mib=%.0f launches_mlp=%d\n",
               o.tag.c_str(), o.if_when.c_str(), o.if_sched.c_str(), o.layout.c_str(), o.drain.c_str(),
               o.kernels.c_str(), o.mapping.c_str(), o.writes.c_str(), o.write_mul, o.mlp,
               o.geometry.c_str(), o.token_burst, o.pages.c_str(),
               o.if_read_mib, o.if_write_mib, o.head_mib, med, lo, hi,
               p.launches_mlp ? med * 1000.0 / p.launches_mlp : 0.0, p.mlp_read_mib, all_mib, gbs_mlp,
               gbs_all, suspect, suspect, v.c_str(), p.alloc_mib, p.launches_mlp);
        printf("NOTE gbs_* are DEMANDED bytes over wall time, not a DRAM rate (L1/L2 absorb reuse).\n"
               "NOTE median of %d reps, best %.4f worst %.4f — quote the spread, never the best.\n",
               o.reps, lo, hi);
    } else {
        printf("NO-TIMING tag=%s verify=%s (add --time to print milliseconds; that is the main "
               "agent's flag, not a worker's)\n", o.tag.c_str(), v.c_str());
    }
    fflush(stdout);
    if (sA) (void)hipStreamDestroy(sA);
    if (sB) (void)hipStreamDestroy(sB);
    free_all(p);
    if (v != "ok" && v != "skipped") {
        fprintf(stderr, "verification failed: %s\n", v.c_str());
        return 3;
    }
    if (o.strict && suspect) return 4;
    return 0;
}

/* ---------------------------------------------------------------------------- main ---------- */

void usage() {
    printf(
        "mlp_insitu — the D2 diagnosis rig for the MLP in-situ penalty (file header has the design).\n"
        "  --dry-run | --selftest-geom | --selftest | --print-cases     no ms, worker-safe\n"
        "  --time                       print milliseconds (main agent only, under gpu_lock)\n"
        "  --tag NAME                   row label, echoed on the INSITU line\n"
        "  --layout blob3|blob72|shipped72   weight placement: published-burst shape / 72 regions /\n"
        "                                    72 regions separated by --gap-mib of warm stand-ins\n"
        "  --drain burst|each           one sync per token, or one sync per MLP launch (the\n"
        "                                    `gran=stage` shape the route line is priced against)\n"
        "  --kernels both|gateup|down   the fixed-per-launch vs bytes-proportional discriminator\n"
        "  --mapping contig|interleave  both shipped walks; one row ships `contig`\n"
        "  --writes shared|spread --write-mul N   store placement; N = MiB stride per layer\n"
        "  --if-read-mib N --if-write-mib N --head-mib N   interferer bytes per token\n"
        "  --if-when before|concurrent  drained carry-over, or true concurrency on 2 streams\n"
        "  --if-sched perlayer|once     split across the 24 layers, or one blob per token\n"
        "  --mlp 0                      interferer alone (the cross-term needs this row)\n"
        "  --layers N --iters N --reps N --gap-mib F --blocks-cap N --suspect-gbs F --verify 0|1\n"
        "  --geometry thin|fat          thin = 48 launches of grid 448/128, the shipped shape.\n"
        "                               fat = the shape the published 2.275 ms burst was taken in: one\n"
        "                               launch per kernel over all layers (grid 10,752/3,072), same\n"
        "                               bytes, same kernel. Requires --layout blob3.\n"
        "  --token-burst N              tokens submitted between syncs. 1 = a drained token (default,\n"
        "                               the shipped measure); N>1 = the never-drained burst the\n"
        "                               published floor was taken in.\n"
        "  --inject none|short_grid|short_read|miss_write|stray_read|stray_write|swaprow|tail|"
        "half|f32act\n"
        "  --pages warm                 cold is refused: a cold row is void by rule, not a result\n");
}

}  // namespace

int main(int argc, char** argv) {
    Config c;
    Opt& o = c.o;
    int i = 1;
    auto next = [&](const char* a) {
        if (i + 1 >= argc) {
            fprintf(stderr, "%s needs a value\n", a);
            exit(2);
        }
        return std::string(argv[++i]);
    };
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else if (a == "--tag") o.tag = next("--tag");
        else if (a == "--layout") o.layout = next("--layout");
        else if (a == "--drain") o.drain = next("--drain");
        else if (a == "--kernels") o.kernels = next("--kernels");
        else if (a == "--mapping") o.mapping = next("--mapping");
        else if (a == "--writes") o.writes = next("--writes");
        else if (a == "--write-mul") o.write_mul = atoi(next("--write-mul").c_str());
        else if (a == "--if-when") o.if_when = next("--if-when");
        else if (a == "--if-sched") o.if_sched = next("--if-sched");
        else if (a == "--if-read-mib") o.if_read_mib = atof(next("--if-read-mib").c_str());
        else if (a == "--if-write-mib") o.if_write_mib = atof(next("--if-write-mib").c_str());
        else if (a == "--head-mib") o.head_mib = atof(next("--head-mib").c_str());
        else if (a == "--gap-mib") o.gap_mib = atof(next("--gap-mib").c_str());
        else if (a == "--mlp") o.mlp = atoi(next("--mlp").c_str());
        else if (a == "--layers") o.layers = atoi(next("--layers").c_str());
        else if (a == "--geometry") o.geometry = next("--geometry");
        else if (a == "--token-burst") o.token_burst = atoi(next("--token-burst").c_str());
        else if (a == "--iters") o.iters = atoi(next("--iters").c_str());
        else if (a == "--reps") o.reps = atoi(next("--reps").c_str());
        else if (a == "--blocks-cap") o.blocks_cap = atoi(next("--blocks-cap").c_str());
        else if (a == "--suspect-gbs") o.suspect_gbs = atof(next("--suspect-gbs").c_str());
        else if (a == "--pages") o.pages = next("--pages");
        else if (a == "--inject") o.inject = next("--inject");
        else if (a == "--verify") o.verify = atoi(next("--verify").c_str());
        else if (a == "--strict") o.strict = 1;
        else if (a == "--time") o.time = 1;
        else if (a == "--selftest") o.selftest = 1;
        else if (a == "--selftest-geom") o.geom_only = 1;
        else if (a == "--dry-run") o.dry = 1;
        else if (a == "--print-cases") o.print_cases = 1;
        else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (o.geometry != "thin" && o.geometry != "fat") {
        fprintf(stderr,
                "--geometry must be thin (48 launches, shipped) or fat (one launch per kernel over "
                "all layers, the published burst's shape); got '%s'. A typo here would silently run "
                "the thin arm and be read as the fat one.\n", o.geometry.c_str());
        return 2;
    }
    if (o.pages != "warm") {
        fprintf(stderr,
                "--pages must be warm. The first GPU pass over fresh pages runs ~35 GB/s, so a cold "
                "row here is not a cold-page measurement, it is a void row (AGENTS.md).\n");
        return 2;
    }
    if (o.drain != "burst" && o.drain != "each") {
        fprintf(stderr, "--drain must be burst|each\n");
        return 2;
    }
    if (o.layout != "blob3" && o.layout != "blob72" && o.layout != "shipped72") {
        fprintf(stderr, "--layout must be blob3|blob72|shipped72\n");
        return 2;
    }
    if (o.writes != "shared" && o.writes != "spread") {
        fprintf(stderr, "--writes must be shared|spread\n");
        return 2;
    }
    if (o.if_when != "before" && o.if_when != "concurrent") {
        fprintf(stderr, "--if-when must be before|concurrent\n");
        return 2;
    }
    if (o.kernels != "both" && o.kernels != "gateup" && o.kernels != "down") {
        fprintf(stderr, "--kernels must be both|gateup|down\n");
        return 2;
    }
    if (o.print_cases) {
        const std::vector<DevCase> t = case_table();
        for (const DevCase& dc : t)
            printf("CASE-ROW name=%s which=%d j=%d inject=%s must_exact=%d must_landing=%d "
                   "must_read=%d must_write=%d must_guard=%d\n",
                   dc.name, dc.which, dc.j, inject_name(dc.inject), dc.expect_exact,
                   dc.expect_landing, dc.expect_read, dc.expect_write, dc.expect_guard);
        printf("CASE-ROWS total=%d\n", (int)t.size());
        return 0;
    }
    if (o.geom_only) return geom_selftest() ? 3 : 0;
    if (o.selftest) {
        int f = geom_selftest();
        printf("\n");
        f += selftest_device();
        return f ? 3 : 0;
    }
    if (o.dry) {
        bill(c);
        print_plan(c);
        printf("DRY no device touched, no timing, no allocation.\n");
        return 0;
    }
    return run_arm(c);
}
