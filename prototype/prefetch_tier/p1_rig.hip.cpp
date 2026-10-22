// p1_rig.hip.cpp — does a prefetched line still live in the 32 MiB tier at the use site?
//
// ROUTE. P1 / P1a / P1b of progress/2026-10-21/generation-idea-queue-2026-10-21.md. P1c (p1c_query.cpp)
// settled the MECHANISM: a HIP persisting/access-policy window on this part addresses 2.000 MiB
// (attr PersistingL2CacheMaxSize == attr L2CacheSize) against a 32.0 MiB tier (KFD node1 cache level 3),
// and hipStreamSetAttribute rejects every window size tried, so the only mechanism available is EXPLICIT
// PREFETCH: issue a real load early and hope the bytes are still there when the compute arrives. That
// hope is what this rig measures. The claim is NOT "a warm loop is fast" — it is "the bytes are still
// resident at the use site", and the polluted arm is what forces the difference.
//
// THE THREE ARMS, per (target, regime, setup). All three run identical setup bytes, so a paired
// difference between them is a residency effect and not a traffic effect:
//   cold      setup(P)                                    then use     <- tier forced over first
//   warm      setup(P)   PREFETCH(target)                 then use     <- nothing in between
//   polluted  setup(P)   PREFETCH(target)   INTERVENE(Q)  then use     <- the traffic that really sits
//                                                                        between the window and the use
// How to read the pair:
//   (cold - warm)     = the prize if the prefetch is issued immediately before the use. The upper bound
//                       on the mechanism, and an upper bound only in this rig's geometry.
//   (cold - polluted) = the prize with the real intervening stream in the way. THIS is the engine number.
//   If (cold - warm) is large while (cold - polluted) is ~0, the bytes are evicted before the use site
//   and prefetch does not belong in this engine. That is the KILL line, written in sweep.sh BEFORE any
//   timing was taken.
//
// WHY SETUP IS COMMON, AND WHY THE POSELLER IS SWEPT. The timed unit is setup + use, so every byte of
// the instrument stays inside the timed quantity. Pricing the use alone would need a separately timed
// setup subtracted from it, and that is exactly how the D4 attribution produced a derived "wall share"
// two instruments disagreed about (prototype/gdn_insitu/D4-DIAGNOSIS.md §0(i)). Common setup bytes is
// also what lets fit_prefetch.py refuse a footprint mismatch. Sweeping P doubles as the dose curve: D4
// measured the state-coldness penalty saturating at exactly 32 MiB, and a rig that saturates somewhere
// else is measuring something else.
//
// BYTE WITNESSES, NOT ARITHMETIC WITNESSES. AGENTS.md records two rigs that shipped confidently wrong
// geometry because a host-side solver agreed with itself (tools/mixed_bw.cpp, prototype/gdn_walk). So:
//   * every region is laid out [guard][target][sentinel][guard]. The guards hold the bit pattern
//     0xDEADBEEF, which no kernel here ever stores, so a stray write is visible on read-back.
//   * the region immediately AFTER every target is filled with a 1e30 sentinel. Reads do not modify
//     memory, so "did the prefetch drag in more than it was told" cannot be witnessed by writes. It is
//     witnessed by the checksum instead: every walk publishes an exact per-block double partial sum of
//     what it actually read, and the host re-derives the expected partials by replaying the SAME index
//     arithmetic over the plan it built the launch from (`expect_sum`). The sentinel's magnitude makes
//     any over-read impossible to miss, and `--inject overread` proves that.
//   * the block reduction is a real block reduction (dynamic shared, one double per thread, butterfly
//     tail). `out[blockIdx] = per_thread_acc` looks like a partial and is not one — AGENTS records
//     mixed_bw reporting 1/256 of the bytes it had read while its ms looked entirely plausible — so the
//     rig ships `--inject badreduce`, whose only job is to prove that witness can fail.
//   * every device case runs in its own forked child (see run_case_child): one faulting launch poisons
//     the context and every later case in the same process silently never runs, which is how mixed_bw's
//     first --selftest shipped half-broken. HIP is only ever initialised inside a child, because
//     hipHostMalloc and friends also fail after a fork of a process that already has a context.
//
// NO TIMINGS UNLESS --time, and this program's author never uses it: AGENTS.md gives every measurement
// to the main agent. --selftest and --verify print booleans and byte counts only.
//
// build: ./build.sh
#include <hip/hip_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const size_t MIB = 1024ull * 1024ull;
static const uint32_t GUARD_BITS = 0xDEADBEEFu;   // no kernel in this file ever stores it
static const float SENTINEL_F = 1.0e30f;         // a dragged-in line cannot hide next to this
static const size_t GUARD_BYTES = 4096;
static const size_t SENTINEL_BYTES = 65536;
static const int THREADS = 256;
static const size_t TIER_BYTES = 32ull * MIB;     // KFD level 3, as p1c_query prints

#define MUST(x)                                                                    \
    do {                                                                           \
        hipError_t e_ = (x);                                                       \
        if (e_ != hipSuccess) {                                                    \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),      \
                    __FILE__, __LINE__);                                           \
            return 7;                                                              \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------------------------
// Content. One pattern function, used by the device filler and by the host expectation, so the witness
// compares the kernel against the PLAN rather than against a second invention.
// ---------------------------------------------------------------------------------------------
enum Kind { KIND_TARGET = 0, KIND_POLLUTE = 1 };

__host__ __device__ __forceinline__ double pat(Kind k, unsigned long long i) {
    if (k == KIND_POLLUTE) return (double)(unsigned)(i & 1023ull) * 0.5;
    unsigned long long h = (i * 2654435761ull) ^ (i >> 7);
    return (double)(unsigned)((h >> 8) & 0xFFFFu) * 0.001953125;   // [0,128)
}

__global__ void fill_kernel(float* p, unsigned long long n, int kind) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) p[i] = (float)pat((Kind)kind, i);
}

__global__ void fill_bits_kernel(float* p, unsigned long long n, unsigned bits) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float v;
        memcpy(&v, &bits, 4);
        p[i] = v;
    }
}

// One double per block, reduced properly. `sh` is dynamic shared of blockDim.x * 8 bytes.
__device__ __forceinline__ void publish_block_sum(double acc, double* sh, double* out) {
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

// A strided float4 walk over [p, p+n_f4) that publishes an exact per-block checksum of what it read.
// The SAME kernel serves the prefetch, the use site and both polluters, so "which read was it" can
// never be confused with "what work did it do".
__global__ void walk_kernel(const float4* __restrict__ p, unsigned long long n_f4,
                            double* __restrict__ out, int badreduce) {
    extern __shared__ double sh[];
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    double acc = 0.0;
    for (; i < n_f4; i += stride) {
        const float4 v = p[i];
        acc += (double)v.x + (double)v.y + (double)v.z + (double)v.w;
    }
    if (badreduce) { out[blockIdx.x] = acc; return; }    // the mixed_bw bug, kept as a must-catch
    publish_block_sum(acc, sh, out);
}

// A resident compute phase with no memory traffic: the host for a concurrent prefetch. Bounded by an
// iteration count, never by a deadline, so no setting of it can wedge a queue.
__global__ void window_kernel(unsigned long long iters, float* sink) {
    float a = 1.0f + (float)threadIdx.x * 1e-6f, b = 1.0000001f;
    for (unsigned long long i = 0; i < iters; ++i) {
        a = fmaf(a, b, 1e-9f);
        b = fmaf(b, a, 1e-9f);
    }
    if (a == 12345.678f) *sink = a + b;                  // unreachable: keeps the chain live
}

// ---------------------------------------------------------------------------------------------
// Regions and plan
// ---------------------------------------------------------------------------------------------
struct Region {
    float* alloc = nullptr;
    float* target = nullptr;
    size_t target_bytes = 0, alloc_bytes = 0;
    Kind kind = KIND_TARGET;
    std::string name;
};

struct Plan {
    Region target, setup, intervene;
    int blocks = 0;
    int threads = THREADS;
    unsigned long long window_iters = 0;
    size_t witness_max_bytes = 64 * MIB;   // above this the per-block witness is SAMPLED, and says so
};

struct Bounds {
    unsigned long long n_f4 = 0, rd_bytes = 0;
    bool ok = false;
};

static Bounds plan_bounds(const Region& r) {
    Bounds b;
    const size_t f4 = 16;
    b.n_f4 = r.target_bytes / f4;
    b.rd_bytes = b.n_f4 * f4;
    b.ok = (r.target_bytes % f4 == 0) && (r.target_bytes % 64 == 0) &&
           (b.rd_bytes <= r.target_bytes) && (r.target_bytes > 0);
    return b;
}

// Geometry invariants. Shared by --selftest-geom (host only) and every real case, so a refused plan is
// refused identically whether or not a device is involved.
static bool invariants(const Plan& p, std::string* why) {
    auto bad = [&](const char* m) { if (why) *why = m; return false; };
    for (const Region* r : {&p.target, &p.setup, &p.intervene}) {
        if (r->target_bytes == 0) continue;
        if (r->target_bytes % 64) return bad("region not a multiple of a 64 B line: an over-read "
                                            "witness would be ambiguous");
        if (!plan_bounds(*r).ok) return bad("plan_bounds refused a region");
    }
    if (p.target.target_bytes == 0) return bad("empty target: a check over zero bytes cannot fail");
    if (p.blocks < 1) return bad("blocks < 1");
    const Bounds b = plan_bounds(p.target);
    if ((unsigned long long)p.blocks * p.threads > 4ull * b.n_f4)
        return bad("grid is >4x the work: most blocks would contribute an empty partial and the "
                   "witness would be measuring the partial buffer, not the read");
    if (p.setup.target_bytes == 0) return bad("no setup region: the three arms would not share bytes");
    return true;
}

static int blocks_for(size_t target_bytes) {
    unsigned long long n_f4 = target_bytes / 16;
    unsigned long long want = (n_f4 + THREADS * 4 - 1) / (THREADS * 4);   // >= 4 float4 per thread
    if (want < 1) want = 1;
    if (want > 1280) want = 1280;
    return (int)want;
}

static int alloc_region(Region& r, size_t bytes, Kind kind, const char* name) {
    // build_plan already named the region after the target under test; a region that is renamed here
    // loses its identity in the output, and fit_prefetch.py stratifies on that name.
    if (r.name.empty()) r.name = name;
    r.kind = kind;
    r.target_bytes = bytes;
    r.alloc_bytes = GUARD_BYTES + bytes + SENTINEL_BYTES + GUARD_BYTES;
    if (bytes == 0) { r.alloc = nullptr; r.target = nullptr; return 0; }
    if (hipMalloc((void**)&r.alloc, r.alloc_bytes) != hipSuccess) return 1;
    r.target = r.alloc + GUARD_BYTES / sizeof(float);
    return 0;
}

static int seed_region(const Region& r, Kind fill_as) {
    if (!r.target_bytes) return 0;
    const unsigned long long n = r.target_bytes / sizeof(float);
    fill_kernel<<<1280, THREADS>>>(r.target, n, (int)fill_as);
    float* sent = r.target + n;
    const unsigned sbits = *(const unsigned*)&SENTINEL_F;
    fill_bits_kernel<<<32, THREADS>>>(sent, SENTINEL_BYTES / sizeof(float), sbits);
    fill_bits_kernel<<<4, THREADS>>>(r.alloc, GUARD_BYTES / sizeof(float), GUARD_BITS);
    fill_bits_kernel<<<4, THREADS>>>(sent + SENTINEL_BYTES / sizeof(float), GUARD_BYTES / sizeof(float),
                                     GUARD_BITS);
    return hipDeviceSynchronize() == hipSuccess ? 0 : 1;
}

// The witness oracle. Replays the kernel's visit pattern on the host, over indices, and returns the
// per-block partial sums the kernel MUST publish. Derived from the plan, so it cannot pass by
// construction, and it needs no host mirror of a 256 MiB polluter.
struct Expected {
    std::vector<double> per_block;
    int blocks = 0;
};

static Expected expect_sum(const Region& r, int blocks, int threads) {
    const Bounds b = plan_bounds(r);
    Expected e;
    e.blocks = blocks;
    e.per_block.assign((size_t)blocks, 0.0);
    const unsigned long long stride = (unsigned long long)threads * blocks;
    for (int blk = 0; blk < blocks; blk++) {
        double acc = 0.0;
        for (unsigned long long t = 0; t < (unsigned long long)threads; t++) {
            for (unsigned long long i = (unsigned long long)blk * threads + t; i < b.n_f4;
                 i += stride) {
                acc += pat(r.kind, i * 4 + 0) + pat(r.kind, i * 4 + 1) +
                       pat(r.kind, i * 4 + 2) + pat(r.kind, i * 4 + 3);
            }
        }
        e.per_block[(size_t)blk] = acc;
    }
    return e;
}

struct Check {
    int rd_ok = 0;
    int guard_ok = 0;
    int sentinel_ok = 0;
    int blocks_checked = 0;
    int blocks_expected = 0;
    double worst_rel = 0.0;
    int first_bad_block = -1;
    double got = 0.0, want = 0.0;
};

static bool bits_untouched(float* dev, size_t bytes, unsigned want) {
    std::vector<float> buf(bytes / sizeof(float), 0.0f);
    if (hipMemcpy(buf.data(), dev, bytes, hipMemcpyDeviceToHost) != hipSuccess) return false;
    for (float f : buf) {
        unsigned bits;
        memcpy(&bits, &f, 4);
        if (bits != want) return false;
    }
    return true;
}

// A tiny relative tolerance is not a numerics band: the walk sums the same values in a different order
// on the host than on the device, so the witness admits F64 reassociation and nothing else. A 1-ULP-of-
// F32 content change in a 4.7 M-element sum is far below this; a wrong base, wrong stride, missing
// block or dragged-in sentinel is orders of magnitude above it.
static const double REL_TOL = 1e-12;

static void check_walk(const Region& r, const Plan& p, const double* dev_partials, Check& out) {
    Expected e = expect_sum(r, p.blocks, p.threads);
    std::vector<double> got((size_t)p.blocks, 0.0);
    out.blocks_expected = p.blocks;
    if (hipMemcpy(got.data(), dev_partials, (size_t)p.blocks * sizeof(double),
                  hipMemcpyDeviceToHost) != hipSuccess) { out.rd_ok = 0; return; }
    for (int i = 0; i < p.blocks; i++) {
        const double want = e.per_block[(size_t)i];
        if (want == 0.0) continue;                 // an empty block contributes nothing to prove
        out.blocks_checked++;
        const double rel = fabs(got[(size_t)i] - want) / fabs(want);
        if (rel > out.worst_rel) out.worst_rel = rel;
        if (rel > REL_TOL && out.first_bad_block < 0) {
            out.first_bad_block = i;
            out.got = got[(size_t)i];
            out.want = want;
        }
    }
    // A witness that checked nothing is not a witness (AGENTS: a parser that matches zero records
    // prints PASS). blocks_checked is asserted against the non-empty expectation count.
    out.rd_ok = (out.first_bad_block < 0 && out.blocks_checked > 0) ? 1 : 0;
    out.guard_ok = bits_untouched(r.alloc, GUARD_BYTES, GUARD_BITS) &&
                   bits_untouched(r.target + r.target_bytes / sizeof(float) +
                                  SENTINEL_BYTES / sizeof(float), GUARD_BYTES, GUARD_BITS);
    out.sentinel_ok = bits_untouched(r.target + r.target_bytes / sizeof(float), 4096,
                                     *(const unsigned*)&SENTINEL_F);
}

// ---------------------------------------------------------------------------------------------
// One case
// ---------------------------------------------------------------------------------------------
enum Arm { ARM_COLD = 0, ARM_WARM = 1, ARM_POLLUTED = 2 };
static const char* arm_name(Arm a) { return a == ARM_COLD ? "cold" : (a == ARM_WARM ? "warm" : "polluted"); }
enum Regime { REG_IDLE = 0, REG_SPIN = 1 };
static const char* regime_name(Regime r) { return r == REG_IDLE ? "idle" : "spin"; }
enum Inject { INJ_NONE = 0, INJ_BADBASE = 1, INJ_OVERREAD = 2, INJ_BADREDUCE = 3, INJ_WRONGFILL = 4 };
static const char* inj_name(Inject i) {
    return i == INJ_NONE ? "none" : (i == INJ_BADBASE ? "badbase"
                                      : (i == INJ_OVERREAD ? "overread"
                                           : (i == INJ_BADREDUCE ? "badreduce" : "wrongfill")));
}

struct Parts {
    double* use = nullptr;
    double* pre = nullptr;
    double* setup = nullptr;
    double* inter = nullptr;
    void free_all() {
        hipFree(use); hipFree(pre); hipFree(setup); hipFree(inter);
        use = pre = setup = inter = nullptr;
    }
};

static double* alloc_partials(int blocks) {
    double* p = nullptr;
    if (hipMalloc((void**)&p, (size_t)blocks * sizeof(double)) != hipSuccess) return nullptr;
    if (hipMemset(p, 0, (size_t)blocks * sizeof(double)) != hipSuccess) return nullptr;
    return p;
}

struct Args {
    std::string target_name = "state";
    // 18.0 MiB = 18,874,368 B, the GDN recurrent state as derived by size_targets.py from
    // engine/state.h (18 layers x 16 heads x 128 x 128 x F32). The ledger's "18.9 MB" is the same
    // bytes read as decimal MB.
    size_t target_bytes = 18ull * MIB;
    size_t setup_bytes = 64 * MIB;        // > 2x the tier, so "cold" means cold
    size_t intervene_bytes = 21 * MIB;    // ~ one MLP weight read per layer (see P1-ARITHMETIC.md)
    std::string arm = "cold", regime = "idle", inject = "none";
    unsigned long long window_iters = 200000;
    int do_time = 0, verify = 0, reps = 5, iters = 3;
};

// One timed/measured unit: setup, [prefetch], [intervene], use. Identical setup bytes on every arm.
// One arm, split where the timer belongs.
//   run_prefix: setup evictor P, then the prefetch (warm/polluted), then the intervening stream Q.
//               All of it is UNTIMED and synced before the timer starts.
//   run_use:    the single use-site read of the target — the only timed quantity.
//
// WHY THE SPLIT MATTERS. An earlier draft timed setup+prefetch+use together and compared the arms. That
// cannot work: a prefetch is an EXTRA read, the consumer's read still happens afterwards, so the warm arm
// moves strictly more bytes than the cold arm and `cold - warm` is negative by construction whatever the
// cache does. The engine prize was never "fewer bytes"; it is "the bytes the consumer read came from the
// tier instead of DRAM", and that is only visible when the consumer's read is what is timed. This is also
// the shape prototype/gdn_insitu used to measure the +0.070 ms state penalty [Q9]: polluter outside, the
// thing of interest timed. And because the setup is never timed, no subtraction of a separately timed
// setup appears anywhere — the move D4 S0(i) records as making a "wall share" unreadable.
static int run_prefix(const Plan& p, Arm arm, Regime regime, const Parts& pt, Inject inj,
                      hipStream_t s_main, hipStream_t s_pre, hipStream_t s_win) {
    const size_t smem = (size_t)p.threads * sizeof(double);
    if (regime == REG_SPIN && p.window_iters) {
        window_kernel<<<64, THREADS, 0, s_win>>>(p.window_iters, nullptr);
        if (hipGetLastError() != hipSuccess) return 1;
    }
    {   // 1. setup — identical bytes on all three arms
        const Bounds b = plan_bounds(p.setup);
        walk_kernel<<<p.blocks, p.threads, smem, s_main>>>(
            (const float4*)(inj == INJ_BADBASE ? p.setup.target - 4 : p.setup.target),
            b.n_f4, pt.setup, inj == INJ_BADREDUCE);
        if (hipGetLastError() != hipSuccess) return 1;
        if (hipStreamSynchronize(s_main) != hipSuccess) return 1;
    }
    if (arm != ARM_COLD) {   // 2. the explicit early load that is the whole point of P1
        const Bounds b = plan_bounds(p.target);
        walk_kernel<<<p.blocks, p.threads, smem, s_pre>>>(
            (const float4*)p.target.target, inj == INJ_OVERREAD ? b.n_f4 + 1 : b.n_f4,
            pt.pre, inj == INJ_BADREDUCE);
        if (hipGetLastError() != hipSuccess) return 1;
        if (hipStreamSynchronize(s_pre) != hipSuccess) return 1;
    }
    if (arm == ARM_POLLUTED && p.intervene.target_bytes) {   // 3. the real gap
        const Bounds b = plan_bounds(p.intervene);
        walk_kernel<<<p.blocks, p.threads, smem, s_main>>>(
            (const float4*)p.intervene.target, b.n_f4, pt.inter, inj == INJ_BADREDUCE);
        if (hipGetLastError() != hipSuccess) return 1;
        if (hipStreamSynchronize(s_main) != hipSuccess) return 1;
    }
    if (regime == REG_SPIN) (void)hipStreamSynchronize(s_win);
    return 0;
}

static int run_use(const Plan& p, const Parts& pt, hipStream_t s_main) {
    const size_t smem = (size_t)p.threads * sizeof(double);
    // 4. the use site. This read is what the route is about.
    const Bounds b = plan_bounds(p.target);
    walk_kernel<<<p.blocks, p.threads, smem, s_main>>>(
        (const float4*)p.target.target, b.n_f4, pt.use, 0);
    if (hipGetLastError() != hipSuccess) return 1;
    if (hipStreamSynchronize(s_main) != hipSuccess) return 1;
    return 0;
}

static int run_unit(const Plan& p, Arm arm, Regime regime, const Parts& pt, Inject inj,
                    hipStream_t s_main, hipStream_t s_pre, hipStream_t s_win) {
    if (run_prefix(p, arm, regime, pt, inj, s_main, s_pre, s_win)) return 1;
    return run_use(p, pt, s_main);
}

static void print_usage() {
    printf("usage: p1_rig [--target NAME] [--target-mib F] [--setup-mib F] [--intervene-mib F]\n"
           "              [--arm cold|warm|polluted] [--regime idle|spin] [--inject none|badbase|\n"
           "              overread|badreduce|wrongfill] [--window-iters N] [--verify] [--time]\n"
           "              [--reps N] [--iters N] | --selftest-geom | --selftest | --list-cases\n"
           "No timing is printed without --time, and --time belongs to the main agent. The timed\n"
           "quantity is ONE use-site read; evictor, prefetch and intervening stream all run outside it,\n"
           "because a prefetch is an extra read and would otherwise be billed against its own win.\n");
}

// The case matrix, in one place so --selftest, the sweep and the fitter all agree on the population.
struct CaseRow {
    const char* target;
    double target_mib;
    const char* arm;
    const char* regime;
    const char* inject;
    int expect_ok;          // 1 = a clean case, 0 = an injection that MUST be caught
};
static const CaseRow CASES[] = {
    {"state",       18.0, "cold",     "idle", "none",      1},
    {"state",       18.0, "warm",     "idle", "none",      1},
    {"state",       18.0, "polluted", "idle", "none",      1},
    {"state",       18.0, "warm",     "spin", "none",      1},
    {"kv-layer",     7.875, "cold",     "idle", "none",    1},
    {"kv-layer",     7.875, "warm",     "idle", "none",    1},
    {"kv-layer",     7.875, "polluted", "idle", "none",    1},
    {"weights",     41.0, "cold",     "idle", "none",      1},
    {"weights",     41.0, "warm",     "idle", "none",      1},
    {"weights",     41.0, "polluted", "idle", "none",      1},
    {"tiny",         0.0625, "warm",   "idle", "none",      1},   // degenerate geometry control
    {"state",       18.0, "polluted", "idle", "badbase",     0},
    {"state",       18.0, "warm",     "idle", "overread",    0},
    {"state",       18.0, "warm",     "idle", "badreduce",   0},
    {"state",       18.0, "polluted", "idle", "badreduce",   0},
    {"state",       18.0, "cold",     "idle", "wrongfill",   0},
};
static const int NCASES = (int)(sizeof CASES / sizeof CASES[0]);

static void list_cases() {
    for (int i = 0; i < NCASES; i++) {
        printf("CASE case%02d target=%s target_mib=%g arm=%s regime=%s inject=%s expect=%s\n", i,
               CASES[i].target, CASES[i].target_mib, CASES[i].arm, CASES[i].regime,
               CASES[i].inject, CASES[i].expect_ok ? "ok" : "caught");
    }
    printf("CASES-PRINTED=%d\n", NCASES);
}

// Builds a Plan from one matrix row. Shared by --selftest's children and by a hand-run single case, so
// a case cannot be run with a geometry the matrix does not describe.
static size_t mib_to_bytes(double mib) { return (size_t)(mib * MIB + 0.5); }

static int build_plan(const Args& a, Plan& p) {
    p.target.name = a.target_name;
    p.setup.name = "setup";
    p.intervene.name = "intervene";
    if (a.target_bytes == 0 || a.setup_bytes == 0) return 1;
    p.target.target_bytes = a.target_bytes;
    p.setup.target_bytes = a.setup_bytes;
    p.intervene.target_bytes = a.intervene_bytes;
    p.target.kind = KIND_TARGET;
    p.setup.kind = KIND_POLLUTE;
    p.intervene.kind = KIND_POLLUTE;
    p.threads = THREADS;
    p.blocks = blocks_for(p.target.target_bytes);
    p.window_iters = a.window_iters;
    return 0;
}

// A case runs entirely inside a forked child: a fault poisons only that child, and HIP is initialised
// fresh each time. Returns the case's exit code: 0 clean-and-verified, 4 witness-red, other = error.
static int case_body(const Args& a) {
    Plan p;
    if (build_plan(a, p)) { printf("REFUSE bad plan for target=%s\n", a.target_name.c_str()); return 2; }
    std::string why;
    if (!invariants(p, &why)) { printf("REFUSE invariants: %s\n", why.c_str()); return 2; }
    Arm arm = a.arm == "cold" ? ARM_COLD : (a.arm == "warm" ? ARM_WARM : ARM_POLLUTED);
    Regime regime = a.regime == "spin" ? REG_SPIN : REG_IDLE;
    Inject inj = a.inject == "none" ? INJ_NONE
               : (a.inject == "badbase" ? INJ_BADBASE
                          : (a.inject == "overread" ? INJ_OVERREAD
                             : (a.inject == "badreduce" ? INJ_BADREDUCE : INJ_WRONGFILL)));
    if (alloc_region(p.target, p.target.target_bytes, KIND_TARGET, a.target_name.c_str())) return 7;
    if (alloc_region(p.setup, p.setup.target_bytes, KIND_POLLUTE, "setup-region")) return 7;
    if (alloc_region(p.intervene, p.intervene.target_bytes, KIND_POLLUTE, "intervene-region")) return 7;
    // --inject wrongfill corrupts the target's CONTENT, not its address: the filler writes the
    // polluter's pattern under the target's name, which is the drift a transcription bug produces. The
    // PLAN still books the target as KIND_TARGET, and it is the plan the witness expects, so the
    // injection has to be caught - which is the whole point of having it.
    const Kind fill_target_as = (inj == INJ_WRONGFILL) ? KIND_POLLUTE : KIND_TARGET;
    if (seed_region(p.target, fill_target_as)) return 7;
    if (seed_region(p.setup, KIND_POLLUTE)) return 7;
    if (seed_region(p.intervene, KIND_POLLUTE)) return 7;

    Parts pt;
    pt.use = alloc_partials(p.blocks);
    pt.setup = alloc_partials(p.blocks);
    pt.pre = (arm != ARM_COLD) ? alloc_partials(p.blocks) : nullptr;
    pt.inter = (arm == ARM_POLLUTED) ? alloc_partials(p.blocks) : nullptr;
    if (!pt.use || !pt.setup || (arm != ARM_COLD && !pt.pre) ||
        (arm == ARM_POLLUTED && !pt.inter)) { pt.free_all(); return 7; }

    hipStream_t s_main = nullptr, s_pre = nullptr, s_win = nullptr;
    hipStreamCreate(&s_main);
    hipStreamCreate(&s_pre);
    hipStreamCreate(&s_win);
    const int rc = run_unit(p, arm, regime, pt, inj, s_main, s_pre, s_win);
    const double mib = p.target.target_bytes / (double)MIB;
    printf("CASE %s target=%s target_mib=%.4f arm=%s regime=%s inject=%s blocks=%d threads=%d "
           "wg_per_launch=%d setup_mib=%.4f intervene_mib=%.4f bounds_ok=%d\n",
           p.target.name.c_str(), p.target.name.c_str(), mib, arm_name(arm), regime_name(regime),
           inj_name(inj), p.blocks, p.threads, p.blocks, p.setup.target_bytes / (double)MIB,
           p.intervene.target_bytes / (double)MIB, plan_bounds(p.target).ok ? 1 : 0);
    if (rc) { printf("LAUNCH-FAIL rc=%d\n", rc); pt.free_all(); return 6; }

    Check cu, cp, cs, ci;
    check_walk(p.target, p, pt.use, cu);
    check_walk(p.setup, p, pt.setup, cs);
    if (arm != ARM_COLD) check_walk(p.target, p, pt.pre, cp);
    if (arm == ARM_POLLUTED) check_walk(p.intervene, p, pt.inter, ci);
    printf("WITNESS %s arm=%s inject=%s use_rd=%d setup_rd=%d pre_rd=%d inter_rd=%d guard=%d "
           "sentinel=%d worst_rel=%.3e blocks_checked=%d/%d bad_block=%d got=%.6f want=%.6f\n",
           a.target_name.c_str(), arm_name(arm), inj_name(inj), cu.rd_ok, cs.rd_ok,
           arm != ARM_COLD ? cp.rd_ok : -1, arm == ARM_POLLUTED ? ci.rd_ok : -1,
           cu.guard_ok && cs.guard_ok, cu.sentinel_ok && cs.sentinel_ok, cu.worst_rel,
           cu.blocks_checked, cu.blocks_expected, cu.first_bad_block, cu.got, cu.want);

    int verdict_ok = cu.rd_ok && cs.rd_ok && cu.guard_ok && cs.guard_ok && cu.sentinel_ok &&
                     cs.sentinel_ok && (arm == ARM_COLD || cp.rd_ok) &&
                     (arm != ARM_POLLUTED || ci.rd_ok);
    // --verify forces the untimed path so a worker's gate cannot accidentally time anything.
    const int do_time = a.verify ? 0 : a.do_time;
    if (!do_time) {
        printf("NO-TIMING %s arm=%s reason=no --time flag (timings belong to the main agent)\n",
               a.target_name.c_str(), arm_name(arm));
    } else {
        // Best-of-N over individual use-site reads, each preceded by its own full prefix. The timed
        // quantity is ONE read of the target, i.e. 0.03-0.2 ms for the engine's real slices. A sub-ms
        // figure, so sample size and spread are part of the result and not a footnote: hipEventElapsedTime
        // returns exactly 0 below ~1 ms on this box and a burst is the only timer to trust (AGENTS), and
        // prototype/gdn_insitu priced a +0.070 ms delta on a 0.12 ms row the same way — reps x iters
        // samples, min and median reported, spread carried to the fitter, which refuses a bimodal stratum.
        std::vector<double> per;
        for (int r = 0; r < a.reps; r++) {
            for (int it = 0; it < a.iters; it++) {
                if (run_prefix(p, arm, regime, pt, inj, s_main, s_pre, s_win)) { pt.free_all(); return 6; }
                const auto t0 = std::chrono::steady_clock::now();
                if (run_use(p, pt, s_main)) { pt.free_all(); return 6; }
                const auto t1 = std::chrono::steady_clock::now();
                per.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        std::sort(per.begin(), per.end());
        const double best = per.front(), worst = per.back();
        const double median = per[per.size() / 2];
        printf("RESULT %s target=%s target_mib=%.4f arm=%s regime=%s inject=%s setup_mib=%.4f "
               "intervene_mib=%.4f blocks=%d threads=%d wg_per_launch=%d verify=%s "
               "ms_min=%.4f ms_median=%.4f ms_max=%.4f samples=%d reps=%d iters=%d spread_rel=%.4f\n",
               p.target.name.c_str(), p.target.name.c_str(), mib, arm_name(arm), regime_name(regime),
               inj_name(inj), p.setup.target_bytes / (double)MIB,
               p.intervene.target_bytes / (double)MIB, p.blocks, p.threads, p.blocks,
               verdict_ok ? "ok" : "FAIL", best, median, worst, (int)per.size(), a.reps, a.iters,
               best > 0.0 ? (worst - best) / best : 0.0);
    }
    // Exit codes are the selftest's contract and mean one thing each: 0 = the witnesses this case
    // expects are green, 4 = at least one witness is red. For an injection red IS the expectation, so
    // the selftest maps rc==4 to "caught" and rc==0 to "NOT-CAUGHT".
    pt.free_all();
    hipStreamDestroy(s_main);
    hipStreamDestroy(s_pre);
    hipStreamDestroy(s_win);
    return verdict_ok ? 0 : 4;
}

static int run_child(Args a) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        int rc = case_body(a);
        fflush(stdout);
        _exit(rc & 0x7f);
    }
    if (pid < 0) return -1;
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {}
    if (WIFSIGNALED(status)) {
        printf("CHILD-KILLED signal=%d\n", WTERMSIG(status));
        return 100;
    }
    return WEXITSTATUS(status);
}

// ---------------------------------------------------------------------------------------------
// CPU-only geometry selftest: the invariants over a table of geometries, plus the anti-vacuity pair
// (every injected geometry fault must be refused, and the clean plan must be accepted).
// ---------------------------------------------------------------------------------------------
static int selftest_geom() {
    int fails = 0, checks = 0;
    auto expect = [&](bool cond, const char* what) {
        checks++;
        if (!cond) { printf("GEOM-FAIL %s\n", what); fails++; }
    };
    struct G { double target_mib, setup_mib, inter_mib; int ok; const char* why; };
    const G table[] = {
        {18.0, 64, 21, 1, "the state target as shipped"},
        {7.875, 64, 21, 1, "one attention layer's KV slice at 4k"},
        {41.0, 64, 21, 1, "one GDN layer's weight block (bigger than the tier, and legal here)"},
        {0.0625, 64, 21, 1, "tiny target, degenerate grid"},
        {0.0, 64, 21, 0, "empty target: a zero-byte witness cannot fail"},
        {18.0, 0, 21, 0, "no setup: the arms would not share bytes"},
        {18.0, 64, 0, 1, "no intervene stream is legal: that is the warm arm"},
        {18.0 + 32.0 / (double)MIB, 64, 21, 0, "target not line-multiple: over-read witness fuzzy"},
    };
    for (const G& g : table) {
        Plan p;
        p.target.name = "t"; p.setup.name = "s"; p.intervene.name = "i";
        p.target.target_bytes = mib_to_bytes(g.target_mib);
        p.setup.target_bytes = mib_to_bytes(g.setup_mib);
        p.intervene.target_bytes = mib_to_bytes(g.inter_mib);
        p.target.kind = KIND_TARGET;
        p.setup.kind = KIND_POLLUTE;
        p.intervene.kind = KIND_POLLUTE;
        p.threads = THREADS;
        p.blocks = blocks_for(std::max<size_t>(16, p.target.target_bytes));
        std::string why;
        const bool got = invariants(p, &why);
        expect(got == (g.ok != 0), g.why);
    }
    // The witness oracle must agree with itself only in the benign direction: a plan whose grid splits
    // the target differently must produce a DIFFERENT per-block partition and the same grand total,
    // because the sum is over the same bytes exactly once. If this ever fails, expect_sum double-counts
    // or drops a block and every byte witness downstream is worthless.
    {
        Plan p;
        p.target.name = "t";
        p.target.target_bytes = mib_to_bytes(0.5);
        p.target.kind = KIND_TARGET;
        const Expected a = expect_sum(p.target, blocks_for(p.target.target_bytes), THREADS);
        const Expected b = expect_sum(p.target, 7, THREADS);
        double sa = 0, sb = 0;
        for (double v : a.per_block) sa += v;
        for (double v : b.per_block) sb += v;
        const double rel = fabs(sa - sb) / fabs(sa);
        checks++;
        if (!(sa > 0 && rel < 1e-12)) { printf("GEOM-FAIL partition-sum invariant rel=%.3e\n", rel); fails++; }
        // anti-vacuity: a plan that drops the last block must change the total
        Expected c = a;
        c.per_block.back() = 0.0;
        double sc = 0;
        for (double v : c.per_block) sc += v;
        checks++;
        if (sc == sa) { printf("GEOM-FAIL dropping a block did not change the sum\n"); fails++; }
    }
    printf("SELFTEST-GEOM ok=%d checks=%d fails=%d\n", fails == 0 ? 1 : 0, checks, fails);
    return fails ? 1 : 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    Args a;
    int do_selftest = 0, do_geom = 0, do_list = 0, one_case = -1;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&](double& d) {
            if (i + 1 >= argc) { printf("REFUSE %s needs a value\n", s.c_str()); exit(2); }
            d = atof(argv[++i]);
        };
        auto str = [&](std::string& d) {
            if (i + 1 >= argc) { printf("REFUSE %s needs a value\n", s.c_str()); exit(2); }
            d = argv[++i];
        };
        if (s == "--target") str(a.target_name);
        else if (s == "--target-mib") { double d; next(d); a.target_bytes = mib_to_bytes(d); }
        else if (s == "--setup-mib") { double d; next(d); a.setup_bytes = mib_to_bytes(d); }
        else if (s == "--intervene-mib") { double d; next(d); a.intervene_bytes = mib_to_bytes(d); }
        else if (s == "--window-iters") { double d; next(d); a.window_iters = (unsigned long long)d; }
        else if (s == "--reps") { double d; next(d); a.reps = (int)d; }
        else if (s == "--iters") { double d; next(d); a.iters = (int)d; }
        else if (s == "--arm") str(a.arm);
        else if (s == "--regime") str(a.regime);
        else if (s == "--inject") str(a.inject);
        else if (s == "--time") a.do_time = 1;
        else if (s == "--verify") a.verify = 1;
        else if (s == "--selftest") do_selftest = 1;
        else if (s == "--selftest-geom") do_geom = 1;
        else if (s == "--list-cases") do_list = 1;
        else if (s == "--case") { double d; next(d); one_case = (int)d; }
        else { print_usage(); return 2; }
        if (a.reps < 1 || a.iters < 1) { printf("REFUSE reps and iters must be >= 1\n"); return 2; }
    }
    if (do_geom) return selftest_geom();
    if (do_list) { list_cases(); return 0; }
    if (do_selftest) {
        int cases = 0, parsed = 0, bad_children = 0, caught = 0, must_catch = 0, blind = 0;
        for (int i = 0; i < NCASES; i++) {
            const CaseRow& r = CASES[i];
            Args ca;
            ca.target_name = r.target;
            ca.target_bytes = mib_to_bytes(r.target_mib);
            ca.setup_bytes = 64 * MIB;
            ca.intervene_bytes = 21 * MIB;
            ca.arm = r.arm;
            ca.regime = r.regime;
            ca.inject = r.inject;
            const int rc = run_child(ca);
            cases++;
            if (rc < 0 || rc >= 100) { bad_children++; continue; }
            parsed++;
            if (r.expect_ok == 0) {
                must_catch++;
                if (rc == 4) caught++;
                else printf("NOT-CAUGHT case%02d inject=%s rc=%d\n", i, r.inject, rc);
            } else if (rc != 0) {
                blind++;
                printf("CLEAN-CASE-RED case%02d %s/%s/%s rc=%d\n", i, r.target, r.arm, r.inject, rc);
            }
        }
        printf("SELFTEST %s cases=%d parsed=%d children_bad=%d caught=%d/%d clean_red=%d\n",
               (bad_children == 0 && parsed == cases && caught == must_catch && blind == 0) ? "ok"
                                                                                             : "FAIL",
               cases, parsed, bad_children, caught, must_catch, blind);
        return (bad_children == 0 && parsed == cases && caught == must_catch && blind == 0) ? 0 : 1;
    }
    if (one_case >= 0 && one_case < NCASES) {
        const CaseRow& r = CASES[one_case];
        Args ca;
        ca.target_name = r.target;
        ca.target_bytes = mib_to_bytes(r.target_mib);
        ca.arm = r.arm;
        ca.regime = r.regime;
        ca.inject = r.inject;
        ca.do_time = a.do_time;
        ca.reps = a.reps;
        ca.iters = a.iters;
        return run_child(ca);
    }
    return case_body(a);
}
