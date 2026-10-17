/*
 * tools/mixed_bw.cpp — is 237 GB/s even the right ceiling for a decode step?
 *
 * Every roofline in this repo is a READ number: `tools/bw_probe_hip.cpp` streams a 1 GiB buffer
 * with a grid-stride sum and the box answers 237 GB/s, and `bench/performance-status.md` then bills
 * ~1.5 GB of weights against it. But a decode step is not a read-only workload: it streams weights
 * WHILE writing activations, the 97 bf16->f32 cast outputs (115 before `62b73e9`), the KV append and the GDN state
 * (18.87 MB read + 18.87 MB written per token, plus the conv window). Q7's M1 refuted the
 * fresh-window story — the same MLP kernels at a 528 MB cold window run at 93 % of the probe in
 * isolation and 52 % inside the step — so the surviving candidate for the ~3.2 ms residue is that
 * write traffic is taxing the read traffic. That is only true if the memory system falls off a
 * cliff between pure read and the step's own mix. This tool finds the curve; the argument is
 * `tools/step_traffic_model.py`'s job.
 *
 * What it measures, one line per run so a shell loop can sweep it:
 *   read     pure read of the footprint (reproduces `bw_probe_hip`'s number at blocks>=80)
 *   write    pure write (full 64 B lines, no read-modify-write)
 *   mix      reads and writes in ONE kernel at exactly R:W bytes, interleaved round by round
 *   wthenr   a write pass immediately followed by a read pass, timed as one pair AND separately,
 *            so the cost of reading while dirty lines are being written back is the difference
 *   copy     read + write of the same bytes (the 1:1 reference `bw_probe_hip` reports as ~210)
 *   interleave  a POOL of same-sized read buffers, touched in one of two orders at identical
 *               total bytes: `blocked` (each buffer re-walked back-to-back, stand-ins first) or
 *               `roundrobin` (pass-major, stand-ins embedded between the engine-shape launches).
 *               Q9's inter-kernel-locality question, with the ORDER as the only free variable.
 * And the read stream has a pattern flag: `contiguous` (the grid-stride float4 sweep every read
 * row above uses) or `engine-mlp` (the engine's real per-thread weight arithmetic, math removed).
 * Q9's intra-kernel question, with the MAPPING as the only free variable.
 *
 * Rules this file obeys, all of them load-bearing (AGENTS.md trap rows):
 *   * `steady_clock` bursts only. `hipEventElapsedTime` returns exactly 0.000 ms for sub-millisecond
 *     work on this iGPU — an event timer here reports inf GB/s. One sync at the end of the burst,
 *     warm-up pass outside the timed region.
 *   * cold-page numbers are their OWN rows, never averaged with warm ones: the first GPU pass over
 *     freshly allocated pages runs ~35 GB/s (`devices/strix-halo-unified-memory.md`), so mixing the
 *     two makes up a number. `--pages both` prints one row per state, cold first, and the cold row
 *     is necessarily a single iteration (after one pass the pages are warm by definition).
 *   * >240 GB/s means skipped work, not a fast kernel. So EVERY row is gated on a byte account, and
 *     the gate runs inside the timed burst rather than beside it (`--verify 1`, the default), so a
 *     number can only be printed by a kernel that proved what it touched:
 *       - reads: the source buffer is filled with 1.0f and each block reduces its per-thread
 *         partials in DOUBLE — integer-valued, so the sum is exactly 4.0 x float4s-read and the
 *         accumulation order cannot matter. Short or skipped work prints `checksum=FAIL_reads_short`.
 *         It has to be a real block reduction: `sums[blockIdx.x] = per_thread_acc` looks like a
 *         partial but keeps one thread's value, and that version failed every warm row it shipped.
 *       - writes: the arena is pre-filled with a value the write stream never stores (0.0f), so an
 *         untimed read-back after the burst proves every arena byte was written rather than assumed.
 *         A missing write prints `checksum=FAIL_writes_missing`.
 *       - `--pages cold` cannot verify either account: the pre-fill would warm the pages it is
 *         measuring, so a cold row prints `checksum=unverifiable_cold_pages` and the process exits
 *         non-zero. Read rows add the one witness cold pages allow, `fresh_zero_sum=1` (the bytes
 *         read were all zero, i.e. the pages were fresh); a cold WRITE pass overwrites the only
 *         evidence, so its freshness is structural (`fresh_alloc_untouched=1`: this process issues no
 *         access to those pages before the timed launch) and it is still a non-zero exit.
 *     A row whose bytes were not verified is printed with a field that cannot be misread as `ok`, and
 *     never with exit code 0 (see EXIT CODES below): per rule 11 of
 *     `devices/strix-halo-mixed-traffic.md`, no conclusion may be drawn from such a row.
 *     Rows above `--suspect-gbs` (default 240) carry `suspect=1`; a small footprint can legitimately
 *     beat the ceiling (the 18 MiB GDN state reads at ~918 GB/s warm), which is why that is a flag on
 *     the row and not a deleted number.
 *   * the read account covers EVERY read launch of one iteration, not just the last one. Each read
 *     launch reduces into its own slot of the `sums` array, so a launch that never ran, ran short,
 *     or ran twice is visible; a single-slot account cannot see a skipped middle launch, which is
 *     exactly the bug `--inject skip_launch` exists to prove is caught (see SELFTEST).
 *   * the geometry the solver produces is checked against the indexing the kernels actually compute
 *     (`plan_bounds`), not against a second arithmetic of the host's own invention. The first version
 *     of this file had `rounds` meaning "grid passes per buffer" to the solver and "trips of the
 *     ratio loop" to `mix_k`; the two agreed only at ratio 1:1, so every `mix` row with ratio > 1
 *     read 62x past the end of its own buffer and faulted, while `--selftest-geom` printed
 *     `ratio_act=62.0000 ok=1` for exactly those configurations. That is why `--selftest` (device)
 *     is a byte witness and not a geometry print: see SELFTEST below.
 *   * `hipMalloc` only, never managed memory (managed degrades SILENTLY near the pool limit).
 *
 * EXIT CODES: 0 = every printed row verified; 2 = bad arguments or an impossible geometry;
 * 3 = a verification failed OR a row was printed whose bytes could not be verified; 4 = suspect row
 * with `--strict`. `--selftest` exits non-zero if any case fails, any child dies, or any injected bug
 * escapes the one account that is supposed to catch it.
 *
 * Build:  tools/build.sh            (-> tools/build/mixed_bw, -O3 + rpath)
 * Run:    it is a timing tool, so it belongs to the main agent under tools/gpu_lock.sh.
 *     tools/build/mixed_bw --mode mix --ratio 20 --footprint-mib 528 --pages both --iters 8
 * Self-checks that touch no device and take no timing:  --dry-run prints the geometry it would run;
 * --selftest-geom runs the geometry invariants (CPU only) over the sweep's own configuration table.
 * SELFTEST: `--selftest` runs --selftest-geom and then one device case per child process (the parent
 * never touches the device, so a faulting kernel costs one child and is still attributed and counted).
 * Each child allocates with poison guard bands around every region, runs ONE untimed pass, and proves
 * the bytes it touched: the read checksum equals the plan's read account exactly, the write landing
 * read-back equals the write account, and no guard byte was read (they would move the sum) or written
 * (they would stop being poison). It prints no ms and no GB/s, because a gate that also reports a
 * bandwidth number is a measurement wearing a costume.
 */
#include <hip/hip_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#define CK(x)                                                                \
    do {                                                                     \
        hipError_t e_ = (x);                                                 \
        if (e_ != hipSuccess) {                                              \
            fprintf(stderr, "hip error %s at %s:%d\n",                       \
                    hipGetErrorString(e_), __FILE__, __LINE__);              \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

namespace {

constexpr double MI = 1048576.0;
// The read-only reference number of this box (devices/strix-halo-memory-bandwidth.md).
constexpr double READ_PROBE_GBS = 237.0;

// The three values the buffers can hold, chosen so that each is observable and none is confusable:
// 1.0f is what the read regions hold and what the write stream stores; 0.0f pre-fills the write
// arena, so a write that never happened is visible in the read-back; -1.0f is the guard band, so a
// stray read moves the checksum and a stray write is found on inspection.
constexpr float kFillVal = 1.0f;
constexpr float kUnwrittenVal = 0.0f;
constexpr float kGuardVal = -1.0f;

// Anti-vacuity injections (see SELFTEST in the file header). `short_read` and `miss_write` are
// bugs that stay INSIDE their buffers: a bug that faults would be caught by the fault, not by the
// accounting, and that proves nothing about the accounting. `stray_read` and `stray_write` are their
// mirror image: they step exactly one float4 over the edge on purpose, which is only safe inside
// `--selftest`, where the guard band catches it. Without the bands the same injection faults, and
// that is the point — it is the negative control that shows the guard witness can actually fire.
enum : int {
    kInjNone = 0, kInjShortRead = 1, kInjMissWrite = 2, kInjStrayRead = 3, kInjStrayWrite = 4,
    // `short_row` is the pattern kernel's own short read: one whole weight row of the last pass is
    // never read. It stays inside the region, so only the byte account can see it.
    // `skip_launch` is the schedule's: one read launch of the iteration does not run at all, which
    // is the failure mode a last-launch-only checksum is blind to.
    kInjShortRow = 5, kInjSkipLaunch = 6
};

// ---- kernels -------------------------------------------------------------------------
// Every stream is lane-coalesced. One "round" reads R float4 per thread at a stride of the whole
// grid and then writes W float4 per thread, contiguous across the warp. R:W is therefore exact by
// construction (no rounding, no partial lines) and a single-lane partial-line write — which would
// turn the write stream into read-modify-write traffic and make the answer an artifact — cannot
// happen. This is the whole design: measure the mix, do not fight the coalescer.

// Reduce the block's per-thread partials into sums[blockIdx.x]. This has to be a real reduction:
// `sums[blockIdx.x] = acc`, which looks like a per-block partial and is not one, leaves whichever
// thread wrote last, i.e. it reports 1/blockDim of the bytes read (measured: every warm row of the
// first sweep failed with FAIL_reads_short for exactly this reason). The values are integer-valued
// doubles, so this fixed tree order is exact and order-independent, and no atomics are involved.
// Shared memory is dynamically sized — one double per thread — so a 256-thread launch pays 2 KB.
// The call sits inside `if (verify)`, which is uniform across the block, so __syncthreads is safe.
// `sh` is a parameter rather than an `extern __shared__` inside this function because the pattern
// kernel keeps its activation staging and its reduction partials in ONE dynamic allocation.
__device__ __forceinline__ void publish_block_sum(double acc, double* __restrict__ sh,
                                                 double* __restrict__ sums) {
    const unsigned t = threadIdx.x, n = blockDim.x;   // n is a power of two (checked on the host)
    sh[t] = acc;
    __syncthreads();
    for (unsigned half = n / 2; half > 0; half /= 2) {
        if (t < half) sh[t] += sh[t + half];
        __syncthreads();
    }
    if (t == 0) sums[blockIdx.x] = sh[0];
}

/*
 * `--pattern engine-mlp` — the engine's own per-thread weight arithmetic, with the math taken out.
 *
 * Why this exists and what it can prove. `C_mixlaunch_62` walks 48 x 11.01 MiB of contiguous
 * float4s at the step's ratio and gets 86.6 % of the read probe, while the MLP kernels in a real
 * step run the same bytes at 52 % (bench/performance-status.md Q6.1, Q8). One surviving candidate
 * is that the kernel's own mapping is what costs — which threads touch which bytes, in what order,
 * with what else in flight — and the only way to tell that from "the memory system was cold" is to
 * walk the same bytes with the engine's real mapping in a probe that changes nothing else. So this
 * is a COPY of `prototype/mlp_gemv/mlp_gemv_kernels.h`'s addressing, not an approximation of it:
 *
 *   * one wavefront per weight row:   row = blockIdx.x*rpb + threadIdx.x/warpSize,   the engine's
 *     own expression, with rpb = blockDim.x/warpSize = the launcher's rows/block (256 threads on
 *     this part -> 8, exactly `ROWS_PER_BLOCK`); a wavefront that lands past the end returns, which
 *     is the engine's tail and is reproduced rather than smoothed over;
 *   * inside the row the lane walks uint4 words at the engine's stride: v = lane; v < words;
 *     v += warpSize, so one warp instruction consumes warpSize consecutive 16-byte words = 512
 *     bytes = four complete 128-byte transactions, and a row is IN bf16 = IN/8 words = IN*2 bytes;
 *   * `--engine-in 1024` is `mlp_gateup`'s shape: TWO weight streams read interleaved by the same
 *     threads (row i of Wg and row i of Wu), so a read launch PAIRS two regions — two separate
 *     hipMallocs, like the two blobs — and each wavefront pulls 2 KiB from each;
 *   * `--engine-in 3584` is `mlp_down`'s shape: one stream, 7 KiB rows, 128 blocks at the engine's
 *     own grid;
 *   * `--stage 1` reproduces the activation staging too. The engine's `mlp_gateup`/`mlp_down` declare
 *     `__shared__ float xs[IN]`, stage one x row into it, `__syncthreads()`, then re-read it as
 *     `&xs[v*8]` — 8 F32 reads per 16-byte weight word, i.e. 32 B of LDS traffic per 16 B of weight,
 *     on every launch, with the addressing that puts lane l and lane l+4 on the same bank. The plain
 *     reader never touches LDS at all, so this has to be a switch and not an assumption. The staged
 *     row here is the same width and element type as the engine's (`stage_f4 = IN/4` float4s = IN
 *     floats: 4 KiB at `--engine-in 1024`, 14 KiB at 3584), so the footprint and the bank pattern
 *     match the engine's rather than approximating it. `--stage 0` is the row that separates the two
 *     explanations.
 *
 * What is deliberately NOT copied: the dot products, the SwiGLU epilogue, the BF16 output store.
 * The question is which bytes arrive at what rate; arithmetic does not change the byte stream, and
 * a probe that carried the FMAs could not tell a mapping loss from an ALU loss. Reads are never
 * skipped and the checksum still proves every byte landed — with one exception that is a feature:
 * the staged activation is pre-filled with 0.0f, so it contributes exactly 0.0 to a sum that must
 * stay exact (it is traffic, not evidence). The staging is LDS traffic, not DRAM traffic, so `gbs`
 * does not bill it — the same convention the engine's own roofline uses, which is billed on weight
 * bytes alone. The row prints `stage_mib=` for its size and `stage_witness=zero_by_construction` to
 * say which account does not cover it.
 */
__global__ void read_engine_k(const float4* __restrict__ a, const float4* __restrict__ b,
                             size_t n4, size_t row_f4, size_t stage_f4,
                             const float4* __restrict__ stage_src, double* __restrict__ sums,
                             int verify, int inject) {
    extern __shared__ float smraw[];                 // [staged x row][reduction partials]
    double* const sh = reinterpret_cast<double*>(smraw + stage_f4 * 4);
    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int rpb = (int)(blockDim.x / ws);           // the launcher's rows/block
    const size_t wf_pass = (size_t)gridDim.x * (size_t)rpb;   // rows one grid pass covers
    const size_t rows = row_f4 ? n4 / row_f4 : 1;
    const size_t gw = (size_t)blockIdx.x * (size_t)rpb + threadIdx.x / ws;  // the engine's row index
    if (stage_f4) {                                   // the engine stages x, then __syncthreads
        const float* src = reinterpret_cast<const float*>(stage_src);
        const size_t nf = stage_f4 * 4;
        for (size_t i = threadIdx.x; i < nf; i += blockDim.x) smraw[i] = src[i];
        __syncthreads();
    }
    double acc = 0.0;
    for (size_t base = 0; base < rows; base += wf_pass) {
        const size_t row = base + gw;
        if (row >= rows) break;                       // the engine's `if (row >= rows) return;`
        // kInjShortRow: the last row of the region is never read. Whole, inside, and invisible to
        // every witness but the read account.
        if (inject == kInjShortRow && row + 1 == rows) continue;
        const float4* ra = a + row * row_f4;
        const float4* rb = b ? b + row * row_f4 : nullptr;
        for (size_t v = lane; v < row_f4; v += ws) {
            const float4 va = ra[v];
            acc += (double)(va.x + va.y + va.z + va.w);
            if (rb) {
                const float4 vb = rb[v];
                acc += (double)(vb.x + vb.y + vb.z + vb.w);
            }
            if (stage_f4) {                           // the engine's &xs[v*8 + j] reads
                const float* xp = &smraw[v * 8];
#pragma unroll
                for (int j = 0; j < 8; ++j) acc += (double)xp[j];
            }
        }
    }
    if (verify) publish_block_sum(acc, sh, sums);
}

__global__ void read_k(const float4* __restrict__ src, size_t n4, double* __restrict__ sums,
                       int verify) {
    extern __shared__ double sh[];
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double acc = 0.0;
    for (size_t i = t; i < n4; i += s) {
        const float4 v = src[i];
        acc += (double)(v.x + v.y + v.z + v.w);
    }
    if (verify) publish_block_sum(acc, sh, sums);
}

__global__ void write_k(float4* __restrict__ dst, size_t n4, float v) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float4 val = make_float4(v, v, v, v);
    for (size_t i = t; i < n4; i += s) dst[i] = val;
}

// rrd reads per rwr writes, both lane-coalesced, in ONE kernel so the two streams are genuinely
// in flight together (separate launches on one stream would serialize and measure nothing).
// `rounds` is the trip count of the ratio loop and the ONLY meaning this field has: the read region
// is exactly rounds*rrd*stride float4s, and `plan_bounds` re-derives that index arithmetic on the
// host so the solver and the kernel cannot drift into meaning two different things again.
// `wrounds` is how many write passes the arena can hold; the write index cycles through it, so a
// small write arena is rewritten again and again while the read stream runs — which is the engine's
// real shape (a few MB of activations and one GDN-state arena, dirtied 424 times a token).
__global__ void mix_k(const float4* __restrict__ src, float4* __restrict__ dst, size_t rounds,
                      size_t rrd, size_t rwr, size_t wrounds, double* __restrict__ sums,
                      int verify, int inject) {
    extern __shared__ double sh[];
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float4 one = make_float4(kFillVal, kFillVal, kFillVal, kFillVal);
    double acc = 0.0;
    for (size_t j = 0; j < rounds; ++j) {
        const bool last = (j + 1 == rounds);
        // kInjShortRead: the last round reads nothing. Inside the buffer, invisible to every check
        // but the read checksum.
        if (!(inject == kInjShortRead && last)) {
            for (size_t r = 0; r < rrd; ++r) {
                const float4 v = src[(j * rrd + r) * s + t];
                acc += (double)(v.x + v.y + v.z + v.w);
            }
        }
        // kInjMissWrite: one block skips its store on the last round. Every other block still
        // writes, so only the write-side landing check can see it.
        const bool miss_wr = (inject == kInjMissWrite && last && blockIdx.x + 1 == gridDim.x);
        if (!miss_wr) {
            const size_t wr = (wrounds ? (j % wrounds) : j) * rwr;
            for (size_t w = 0; w < rwr; ++w) dst[(wr + w) * s + t] = one;
        }
    }
    // The two strays. Each is one float4, taken by lane 0 of each block, so it lands in the first
    // slot of the guard band rather than `stride` float4s over the edge (every other index in this
    // file is lane-scaled). A stray read moves the checksum because the guard is not 1.0f; a stray
    // write is found because the guard is no longer poison.
    if (inject == kInjStrayRead && threadIdx.x == 0) {
        const float4 v = src[rounds * rrd * s];
        acc += (double)(v.x + v.y + v.z + v.w);
    }
    if (inject == kInjStrayWrite && threadIdx.x == 0) {
        dst[wrounds * rwr * s] = one;
    }
    if (verify) publish_block_sum(acc, sh, sums);
}

// ---- configuration ---------------------------------------------------------------------
struct Cfg {
    std::string mode = "read";       // read | write | mix | mixlaunch | wthenr | interleave
    std::string tag = "-";
    std::string pages = "both";      // both | warm | cold
    long rrd = 20, rwr = 1;          // read:write in float4 units -> --ratio 20 or --ratio 1:2
    double footprint_mib = 512.0;    // bytes touched per iteration (read + write)
    double buf_mib = 0.0;            // per-read-buffer size (>0: footprint = nbuf * buf_mib)
    double wr_arena_mib = 0.0;       // mix only: arena the write stream cycles through
    int nbuf = 1;                    // separate read buffers, each walked once per iteration
    int blocks = 0;                  // 0 = auto: 640 (one buffer) / 448 (many), mlp_gateup's shape
    int threads = 256;
    int iters = 8;                   // timed iterations per burst, ONE sync at the end
    int pair_sync = 0;               // wthenr: sync between the write pass and the read pass
    int overlap = 1;                 // wthenr: read back the lines just dirtied
    int verify = 1;                  // checksum the read stream (file header explains why)
    double suspect_gbs = 240.0;
    int strict = 0, selftest = 0, device_check = 1, dry = 0;
    int only_case = -1;              // >=0: run exactly this --selftest device case (the child's path)
    std::string inject = "none";     // none|short_read|miss_write|stray_read|stray_write|short_row|skip_launch
    // ---- Q9: the read stream's mapping (which thread touches which byte) ------------------
    std::string pattern = "contiguous";   // contiguous | engine-mlp (see read_engine_k)
    int engine_in = 1024;                 // 1024 = gate/up (two streams, 2 KiB rows, paired regions)
                                          // 3584 = down     (one stream, 7 KiB rows)
    bool engine_in_given = false;         // --engine-in passed: its row width then sizes the regions
    int stage = 1;                        // engine-mlp: stage one x row into LDS and re-read it
    // ---- Q9: the schedule (in which order the regions are touched) -------------------------
    int revisits = 1;                     // interleave: engine passes over its buffers per iteration
    double dummy_mib = 0.0;               // interleave: size of EACH stand-in buffer
    int dummy_nbuf = 0;                   // interleave: how many stand-in buffers
    std::string order = "roundrobin";     // interleave: roundrobin | blocked
    int dummy_same = 0;                   // interleave: stand-ins use the engine mapping too
};

// A resolved geometry: what gets allocated and launched, with exact byte counts. `sched` IS the
// launch order inside one iteration, so the host never re-derives the interleaving at launch time
// and the reported byte counts cannot disagree with what was actually launched.
struct Plan {
    struct Step {
        int kind = 0;        // 0 = read pass, 1 = write pass
        int obj = 0;         // read-buffer index (reads)
        int obj2 = -1;       // second read buffer: the gate/up shape's OTHER stream (Wu)
        int slot = 0;        // which slice of `sums` this read launch writes (see sum_slots)
        int engine = 0;      // 1 = walk this region with the engine's mapping, not the sweep
        size_t off = 0;      // float4 offset into the write arena (writes)
        size_t n4 = 0;       // float4s this launch touches (per region, when it reads two)
        int blocks = 1;      // grid for THIS launch: a 4 KB write must not launch a 640-block grid
    };
    int blocks = 0, threads = 256, nbuf = 1;
    size_t stride = 0;               // threads per full-size launch = blocks*threads
    size_t rd_n4 = 0;                // float4s per read buffer
    int rd_regions = 0;              // regions allocated (launches x 2 when gate/up pairs them)
    int sum_slots = 1;               // slices of `sums`: one per read launch, so a skipped one shows
    size_t stage_f4 = 0;             // float4s of x staged into LDS per block (engine pattern)
    size_t row_f4 = 0;               // float4s per weight row (engine pattern)
    size_t dummy_rd_n4 = 0;          // interleave: float4s per stand-in buffer
    int dummy_launches = 0;          // interleave: stand-in read launches per iteration
    int rd_engine_regions = 0;       // regions sized rd_n4; the rest are the interleave stand-ins
    size_t rounds = 0;               // mix only: trips of the ratio loop inside the one kernel
    size_t passes = 0;               // float4s each thread touches = rd_n4 / stride
    size_t wr_n4 = 0;                // float4s written per iteration = counted write traffic
    size_t wr_buf_n4 = 0;            // arena size; wr_n4 may exceed it (`mix` cycles it)
    size_t wrounds = 0;              // mix: write passes the arena holds before it wraps
    size_t wr_chunk_n4 = 0;          // float4s per write launch
    size_t chk_rd_n4 = 0;            // checksum target: float4s read by ONE iteration, all launches
    int rd_launches = 0, wr_launches = 0;
    double read_bytes = 0, write_bytes = 0, ratio_act = 0;
    std::string fit;                 // what the solver gave up; "-" means nothing
    std::vector<Step> sched;         // launch order for one iteration (empty for `mix`: one kernel)

    // Region i's size. Regions past the engine group are the interleave mode's stand-ins.
    size_t region_n4(int i) const {
        return (rd_engine_regions > 0 && i >= rd_engine_regions) ? dummy_rd_n4 : rd_n4;
    }
};

// The bounds model: the index arithmetic the kernels compute, re-derived on the host from the very
// Plan they were built from, as EXCLUSIVE ends relative to each region. `--selftest-geom` used to
// print `ratio_act=62.0000 ok=1` for geometries whose kernel ran off the end of its own buffer,
// because the ratio was checked by one arithmetic and the indexing by another. They cannot disagree
// again: invariants() compares both of these against what the Plan actually allocates.
struct Bounds {
    size_t rd_issued = 0, rd_end = 0, wr_issued = 0, wr_end = 0;
    bool ok = true;
    std::string why = "-";
};

Bounds plan_bounds(const Cfg& c, const Plan& p) {
    Bounds b;
    if (c.mode == "mix") {
        // mix_k reads (j*rrd + r)*stride + t for j < rounds, r < rrd, and writes
        // ((j % wrounds)*rwr + w)*stride + t, so the read end is exactly the read region and the
        // write end is what the arena holds within `rounds` trips.
        b.rd_issued = p.rounds * (size_t)c.rrd * p.stride;
        b.rd_end = b.rd_issued;
        // Every round writes; only the SLOT cycles, so the traffic issued is rounds deep while the
        // arena's high-water mark is what the arena can hold.
        b.wr_issued = p.rounds * (size_t)c.rwr * p.stride;
        b.wr_end = std::min(p.rounds, p.wrounds) * (size_t)c.rwr * p.stride;
    } else {
        for (const Plan::Step& st : p.sched) {
            if (st.kind == 0) {
                size_t issued = st.n4, end = st.n4;
                if (st.engine) {
                    // read_engine_k walks whole rows: `rows = n4/row_f4` of them, `row_f4` float4s
                    // each, and it breaks out on `row >= rows`. So a region that is not a whole
                    // number of rows is read PARTLY, and the row's byte account would then bill
                    // bytes the kernel never touched. This is where that gets caught.
                    const size_t rows = p.row_f4 ? st.n4 / p.row_f4 : 0;
                    issued = rows * p.row_f4;
                    end = issued;
                }
                if (st.obj2 >= 0) issued *= 2;        // the paired region is walked too, same span
                b.rd_issued += issued;
                b.rd_end = std::max(b.rd_end, end);
            } else {
                b.wr_issued += st.n4;
                b.wr_end = std::max(b.wr_end, st.off + st.n4);
            }
        }
    }
    const size_t rd_want = (c.mode == "mix") ? p.rd_n4 : p.chk_rd_n4;
    const size_t wr_want = p.wr_n4;
    if (b.rd_issued != rd_want) { b.ok = false; b.why = "read_issued_does_not_match_read_account"; }
    else if (b.rd_end > std::max(p.rd_n4, p.dummy_rd_n4)) { b.ok = false; b.why = "read_index_past_region"; }
    else if (b.wr_issued != wr_want) { b.ok = false; b.why = "write_issued_does_not_match_write_account"; }
    else if (b.wr_end > p.wr_buf_n4) { b.ok = false; b.why = "write_index_past_arena"; }
    return b;
}

bool parse(int argc, char** argv, Cfg& c) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto sval = [&](const char* f) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", f); exit(2); }
            return argv[++i];
        };
        auto dval = [&](const char* f) -> double { return atof(sval(f).c_str()); };
        auto ival = [&](const char* f) -> int { return (int)atol(sval(f).c_str()); };
        if (a == "--mode") c.mode = sval("--mode");
        else if (a == "--tag") c.tag = sval("--tag");
        else if (a == "--pages") c.pages = sval("--pages");
        else if (a == "--pattern") c.pattern = sval("--pattern");
        else if (a == "--engine-in") { c.engine_in = ival("--engine-in"); c.engine_in_given = true; }
        else if (a == "--stage") c.stage = ival("--stage");
        else if (a == "--revisits") c.revisits = ival("--revisits");
        else if (a == "--interleave") c.order = sval("--interleave");
        else if (a == "--dummy-mib") c.dummy_mib = dval("--dummy-mib");
        else if (a == "--dummy-nbuf") c.dummy_nbuf = ival("--dummy-nbuf");
        else if (a == "--dummy-same") c.dummy_same = ival("--dummy-same");
        else if (a == "--ratio") {                       // "20" -> 20:1 ; "1:2" -> 1:2 write-heavy
            const std::string v = sval("--ratio");
            const size_t colon = v.find(':');
            if (colon == std::string::npos) { c.rrd = atol(v.c_str()); c.rwr = 1; }
            else { c.rrd = atol(v.substr(0, colon).c_str()); c.rwr = atol(v.substr(colon + 1).c_str()); }
        } else if (a == "--footprint-mib") c.footprint_mib = dval("--footprint-mib");
        else if (a == "--buf-mib") c.buf_mib = dval("--buf-mib");
        else if (a == "--wr-arena-mib") c.wr_arena_mib = dval("--wr-arena-mib");
        else if (a == "--nbuf") c.nbuf = ival("--nbuf");
        else if (a == "--blocks") c.blocks = ival("--blocks");
        else if (a == "--threads") c.threads = ival("--threads");
        else if (a == "--iters") c.iters = ival("--iters");
        else if (a == "--pair-sync") c.pair_sync = ival("--pair-sync");
        else if (a == "--overlap") c.overlap = ival("--overlap");
        else if (a == "--verify") c.verify = ival("--verify");
        else if (a == "--suspect-gbs") c.suspect_gbs = dval("--suspect-gbs");
        else if (a == "--strict") c.strict = 1;
        else if (a == "--inject") c.inject = sval("--inject");
        else if (a == "--selftest") c.selftest = 1;
        else if (a == "--selftest-case") {          // one device case, in THIS process (see SELFTEST)
            c.selftest = 1; c.device_check = 1; c.only_case = ival("--selftest-case");
        } else if (a == "--selftest-geom") { c.selftest = 1; c.device_check = 0; }
        else if (a == "--dry-run") c.dry = 1;
        else if (a == "--help" || a == "-h") return false;
        else { fprintf(stderr, "unknown flag %s\n", a.c_str()); exit(2); }
    }
    if (c.mode != "read" && c.mode != "write" && c.mode != "mix" && c.mode != "mixlaunch" &&
        c.mode != "wthenr" && c.mode != "interleave") {
        fprintf(stderr, "--mode must be read|write|mix|mixlaunch|wthenr|interleave\n"); exit(2);
    }
    if (c.pattern != "contiguous" && c.pattern != "engine-mlp") {
        fprintf(stderr, "--pattern must be contiguous|engine-mlp\n"); exit(2);
    }
    if (c.pattern == "engine-mlp") {
        if (c.mode != "read" && c.mode != "mixlaunch" && c.mode != "interleave") {
            fprintf(stderr, "--pattern engine-mlp is a read-mapping flag: use it with --mode "
                            "read|mixlaunch|interleave, not %s\n", c.mode.c_str());
            exit(2);
        }
        // Only the two widths the engine instantiates. 1024 is `mlp_gateup` (gate and up, two
        // streams, so a launch PAIRS two regions); 3584 is `mlp_down` (one stream). Nothing else
        // is a shape this checkpoint has, and a made-up width would be a pattern nobody runs.
        if (c.engine_in != 1024 && c.engine_in != 3584) {
            fprintf(stderr, "--engine-in must be 1024 (mlp_gateup) or 3584 (mlp_down)\n"); exit(2);
        }
        if (c.stage < 0 || c.stage > 1) { fprintf(stderr, "--stage must be 0 or 1\n"); exit(2); }
        // The two-stream shape pairs regions, so it needs an even region count.
        if (c.engine_in == 1024 && std::max(1, c.nbuf) % 2) {
            fprintf(stderr, "--pattern engine-mlp --engine-in 1024 reads gate and up as a PAIR of "
                            "regions, so --nbuf must be even (48 blobs = 24 launches)\n");
            exit(2);
        }
        if (c.engine_in == 1024 && c.mode == "interleave") {
            fprintf(stderr, "--mode interleave keeps one region per launch so the two orders are "
                            "the same schedule; use --engine-in 3584 there, or test the paired "
                            "gate/up shape with --mode read\n");
            exit(2);
        }
    }
    if (c.order != "roundrobin" && c.order != "blocked") {
        fprintf(stderr, "--interleave must be roundrobin|blocked (the order the read regions are touched in)\n"); exit(2);
    }
    if (c.mode == "interleave") {
        if (c.buf_mib <= 0.0 || c.dummy_mib <= 0.0 || c.dummy_nbuf < 1 || c.nbuf < 2) {
            fprintf(stderr, "--mode interleave needs --nbuf >= 2 --buf-mib B --dummy-nbuf >= 1 "
                            "--dummy-mib D: the engine-shaped buffers and the stand-in stream, "
                            "without them the order cannot matter\n");
            exit(2);
        }
        if (c.revisits < 1) { fprintf(stderr, "--revisits must be >= 1\n"); exit(2); }
    }
    if (c.pages != "both" && c.pages != "warm" && c.pages != "cold") {
        fprintf(stderr, "--pages must be both|warm|cold\n"); exit(2);
    }
    if (c.rrd < 0 || c.rwr < 0 || (c.rrd == 0 && c.rwr == 0)) {
        fprintf(stderr, "--ratio needs a positive side\n"); exit(2);
    }
    // The ratio sides go to the kernel as size_t, but a side this large means the flag was a typo,
    // and a silently truncated ratio is exactly the lie this tool exists to avoid.
    if (c.rrd > (1L << 24) || c.rwr > (1L << 24)) {
        fprintf(stderr, "--ratio sides must be <= 2^24\n"); exit(2);
    }
    if (c.mode == "mix" || c.mode == "mixlaunch") {
        if (c.rrd == 0 || c.rwr == 0) {
            fprintf(stderr, "%s needs a two-sided --ratio\n", c.mode.c_str()); exit(2);
        }
    }
    // The block reduction halves the partials, so the thread count must be a power of two.
    if (c.threads < 32 || c.threads > 1024 || (c.threads & (c.threads - 1))) {
        fprintf(stderr, "--threads must be a power of two in [32,1024] (the checksum reduction "
                        "halves the block)\n");
        exit(2);
    }
    if (c.inject != "none" && c.inject != "short_read" && c.inject != "miss_write" &&
        c.inject != "stray_read" && c.inject != "stray_write" && c.inject != "short_row" &&
        c.inject != "skip_launch") {
        fprintf(stderr, "--inject must be none|short_read|miss_write|stray_read|stray_write|"
                        "short_row|skip_launch\n"); exit(2);
    }
    if (c.inject != "none" && c.mode != "mix" && c.inject != "short_row" &&
        c.inject != "skip_launch") {
        fprintf(stderr, "--inject %s is only implemented for --mode mix (it needs both streams to "
                        "prove which account caught the bug)\n", c.inject.c_str());
        exit(2);
    }
    // The two new bugs are each specific to one new mechanism, and each is the anti-vacuity proof for
    // a witness that only exists for it: `short_row` is the pattern kernel's short read (the whole-
    // row bounds check is what makes it billable at all) and `skip_launch` is a launch that never
    // ran, which only the per-launch checksum slots can see.
    if (c.inject == "short_row" && (c.pattern != "engine-mlp" ||
                                    (c.mode != "read" && c.mode != "mixlaunch"))) {
        fprintf(stderr, "--inject short_row needs --pattern engine-mlp with --mode read|mixlaunch\n");
        exit(2);
    }
    if (c.inject == "skip_launch" && c.mode != "interleave") {
        fprintf(stderr, "--inject skip_launch needs --mode interleave\n"); exit(2);
    }
    // The strays step past the region: safe only where --selftest put a guard band to catch them.
    if ((c.inject == "stray_read" || c.inject == "stray_write") && c.only_case < 0) {
        fprintf(stderr, "--inject %s is a --selftest-case injection: outside the guard bands it "
                        "faults by design\n", c.inject.c_str());
        exit(2);
    }
    if (c.mode == "wthenr" && c.overlap) c.nbuf = 1;     // same lines => one region
    c.iters = std::max(1, c.iters);
    return true;
}

// Solve the geometry: allocations, grid, and the exact launch schedule for one iteration.
//
// Three constraints drive it.
//   * `mix`: one round is rrd float4 reads plus rwr float4 writes, one of each per thread, so a
//     round costs stride*(rrd+rwr)*16 bytes. A steep ratio in a small buffer is therefore a
//     GRID-SIZE problem, not a byte problem: the solver trades block count away to keep the ratio
//     exact, because a silently rounded ratio is how a mix probe lies.
//   * `mixlaunch` / `write` / `wthenr`: each launch carries its own grid, so a write only has to be
//     a multiple of one thread-pass (4 KB at 256 threads) to stay coalesced and never trigger a
//     partial-line read-modify-write — which would turn 1 byte of write traffic into 1.25 bytes of
//     DRAM traffic and silently change what is being measured.
//   * everything: pick the grid that fits the most bytes, and among the grids within 0.5 % of the
//     best fit take the LARGEST. A tiny grid is dispatch-limited rather than bandwidth-bound, and a
//     406-launch decode step already has plenty of dispatch limitation to study elsewhere.
Plan resolve(const Cfg& c) {
    const int nbuf = (c.mode == "mix" || c.mode == "wthenr") ? 1 : std::max(1, c.nbuf);
    // The gate/up shape reads TWO tensors per wavefront, so a launch pairs two regions and each
    // region is half the launch. `down` and every plain row are one region per launch.
    const int streams = (c.pattern == "engine-mlp" && c.engine_in == 1024) ? 2 : 1;
    const bool engine_pat = (c.pattern == "engine-mlp");
    // The row width is a GEOMETRY fact (it comes from --engine-in), not a kernel fact: a contiguous twin
    // passed --engine-in rounds its regions to the same whole rows, so the two sides of a mapping
    // comparison bill identical bytes. --pattern alone decides which kernel runs.
    const size_t row_f4 = c.engine_in_given ? (size_t)c.engine_in / 8 : 0;
    const size_t stage_f4 = (engine_pat && c.stage) ? (size_t)c.engine_in / 4 : 0;
    const int revisit = std::max(1, c.revisits);
    const double total = (c.buf_mib > 0.0 ? c.buf_mib * nbuf : c.footprint_mib) * MI;
    double rb, wb;
    if (c.mode == "read" || c.mode == "interleave") { rb = total; wb = 0.0; }
    else if (c.mode == "write") { rb = 0.0; wb = total; }
    else if (c.mode == "wthenr" && c.overlap) { rb = wb = total; }   // read back the same lines
    else { rb = total * (double)c.rrd / (double)(c.rrd + c.rwr); wb = total - rb; }
    const double arena = c.wr_arena_mib > 0.0 ? c.wr_arena_mib * MI : wb;
    const size_t wgran = (size_t)c.threads;      // float4 per write-launch unit: 4 KB at 256
    const int want = c.blocks > 0 ? c.blocks : (nbuf > 1 ? 448 : 640);
    auto dn = [](size_t v, size_t unit) { return unit ? (v / unit) * unit : 0; };
    // Round to a grid pass first (that is what every committed row did), then to a whole number of
    // weight rows, so an engine-shaped launch can never read a partial row. Both divisors divide the
    // real blob sizes exactly: a gate/up/down blob is 7 MiB = 458752 float4s, divisible by 448*256
    // and by 128 (gate/up, 3584 rows) and by 128*256 and by 448 (down, 1024 rows).
    auto rd_unit = [&](size_t v, size_t s) { return row_f4 ? dn(dn(v, s), row_f4) : dn(v, s); };

    std::vector<Plan> cands;
    for (int b = std::min(want, 4096); b >= 8; --b) {
        Plan q;
        q.blocks = b; q.threads = c.threads; q.nbuf = nbuf;
        q.row_f4 = row_f4; q.stage_f4 = stage_f4;
        const size_t s = q.stride = (size_t)b * (size_t)c.threads;
        size_t rd_n4 = 0, wr_chunk = 0, wr_total = 0, wr_arena = 0, dummy_n4 = 0;
        int rdl = 0, wrl = 0, dummy_rl = 0;
        int rd_regions_read = 0;                // engine regions walked per iteration
        int wafter = 0;                       // 1 = write launch after each read launch
        std::vector<size_t> wr_group;         // per read launch: float4s to write after it
        if (c.mode == "mix") {                // one kernel: exact ratio, one read region
            q.rounds = (s * (size_t)c.rrd * 16) ? (size_t)(rb / (s * (size_t)c.rrd * 16)) : 0;
            rd_n4 = q.rounds * (size_t)c.rrd * s;
            rdl = q.rd_launches = q.rounds ? 1 : 0;
            rd_regions_read = q.rounds ? 1 : 0;
            wr_chunk = (size_t)c.rwr * s;
            wr_total = q.rounds * wr_chunk;
            wr_arena = std::min(wr_total, std::max(wr_chunk, dn((size_t)(arena / 16), wr_chunk)));
        } else if (c.mode == "read") {
            rd_n4 = rd_unit((size_t)(rb / nbuf) / 16, s);
            rdl = rd_n4 ? nbuf / streams : 0;
            rd_regions_read = rd_n4 ? nbuf : 0;
        } else if (c.mode == "write") {
            wr_chunk = std::max(wgran, dn((size_t)(wb / nbuf) / 16, wgran));
            wrl = wr_chunk ? nbuf : 0;
            wr_total = wr_chunk * (size_t)wrl;
            wr_arena = wr_total;
        } else if (c.mode == "interleave") {
            rd_n4 = rd_unit((size_t)(rb / nbuf) / 16, s);
            dummy_n4 = dn((size_t)(c.dummy_mib * MI / 16), s);
            rdl = rd_n4 ? nbuf * revisit : 0;
            dummy_rl = (rd_n4 && dummy_n4) ? c.dummy_nbuf : 0;
            rd_regions_read = rd_n4 ? nbuf * revisit : 0;
        } else if (c.mode == "mixlaunch") {    // the engine's shape: serialized launches, one stream
            rd_n4 = rd_unit((size_t)(rb / nbuf) / 16, s);
            rdl = rd_n4 ? nbuf / streams : 0;
            rd_regions_read = rd_n4 ? nbuf : 0;
            // Write launches are distributed across the read launches instead of rounded per
            // buffer: the whole iteration hits the requested ratio to within one 4 KB unit, so the
            // ratio error no longer grows with the ratio (per-buffer rounding made a 62:1 mix
            // silently become 63:1).
            const double target = (double)rd_n4 * (double)std::max(1, rdl) * streams
                                * (double)c.rwr / (double)std::max(1L, c.rrd);
            const size_t units = std::max<size_t>((size_t)std::max(1, rdl),
                                                  (size_t)(target / (double)wgran + 0.5));
            wr_group.assign(std::max(1, rdl), 0);
            for (int i = 0; i < rdl; ++i) {
                const size_t upto = (size_t)((long)(i + 1) * units / std::max(1, rdl));
                const size_t was = (size_t)((long)i * units / std::max(1, rdl));
                wr_group[i] = (upto - was) * wgran;
            }
            wafter = 1;
            for (int i = 0; i < rdl; ++i) if (wr_group[i]) { wr_total += wr_group[i]; ++wrl; }
            wr_chunk = rdl ? wr_group[0] : 0;
            wr_arena = wr_total;
        } else {                               // wthenr: write pass first, then the read pass
            rd_n4 = dn((size_t)(rb) / 16, s);
            rdl = rd_n4 ? 1 : 0;
            rd_regions_read = rdl;
            wr_chunk = c.overlap ? rd_n4 : std::max(wgran, dn((size_t)(wb) / 16, wgran));
            wrl = wr_chunk ? 1 : 0;
            wr_total = wr_chunk;
            wr_arena = wr_chunk;               // the same arena is dirtied again each pass
        }
        q.rd_n4 = rd_n4;
        q.dummy_rd_n4 = dummy_n4;
        q.dummy_launches = dummy_rl;
        q.rd_engine_regions = (c.mode == "interleave") ? nbuf : rd_regions_read;
        q.rd_regions = (c.mode == "interleave") ? nbuf + dummy_rl
                       : (c.mode == "write" ? 0 : std::max(1, rd_regions_read));
        q.wr_launches = wrl;
        // `passes` is reporting only. It must NOT feed back into `rounds`: for `mix`, `rounds` is the
        // kernel's loop count and rd_n4 is rounds*rrd*s, so conflating the two multiplies the kernel's
        // read window by rrd — the bug that faulted every `mix` row with ratio > 1 in the first Q8
        // sweep while --selftest-geom called those same geometries ok.
        q.passes = s ? rd_n4 / s : 0;
        q.wr_chunk_n4 = wr_chunk;
        q.wr_n4 = wr_total;
        q.wr_buf_n4 = wr_arena;
        q.wrounds = wr_chunk ? wr_arena / wr_chunk : 0;
        size_t woff = 0;
        int rslot = 0;
        auto push_wr = [&](size_t n4) {
            if (!n4) return;
            Plan::Step st; st.kind = 1; st.off = woff; st.n4 = n4;
            st.blocks = (int)std::min<size_t>(8192, std::max<size_t>(1, (n4 + wgran - 1) / wgran));
            q.sched.push_back(st); woff += n4;
        };
        // One read launch. `region` (/`+1` when paired) indexes the region vectors and `n4` is per
        // REGION, so a paired gate/up launch bills 2*n4. Every launch gets its own `sums` slot,
        // which is what makes a launch that never ran visible to the byte account instead of
        // invisible to it.
        auto push_rd = [&](size_t region, size_t n4, bool paired, bool engine) {
            if (!n4) return;
            Plan::Step st;
            st.kind = 0; st.obj = (int)region; st.n4 = n4; st.blocks = b;
            st.slot = rslot++; st.engine = engine ? 1 : 0;
            if (paired) st.obj2 = (int)(region + 1);
            q.sched.push_back(st);
        };
        if (c.mode == "wthenr") { push_wr(wr_chunk); push_rd(0, rd_n4, false, false); }
        else if (c.mode == "write") { for (int i = 0; i < wrl; ++i) push_wr(wr_chunk); }
        else if (c.mode == "interleave") {
            // The two orders launch the SAME MULTISET — same regions, same sizes, same count — so
            // any difference between the two rows is the order alone: how recently each buffer was
            // read. `blocked` is the best case for locality (each region revisited back to back);
            // `roundrobin` is the step's shape, where other families' traffic sits between two
            // touches of the same weights. The stand-ins are spread evenly through the round-robin
            // order, which is what lets them stand in for those families.
            const int E = std::max(1, rdl);
            int pushed_d = 0;
            auto push_dummies_up_to = [&](int upto) {
                while (pushed_d < upto)
                    { push_rd(nbuf + pushed_d, dummy_n4, false, c.dummy_same != 0); ++pushed_d; }
            };
            if (c.order == "blocked") {
                push_dummies_up_to(dummy_rl);
                for (int i = 0; i < nbuf; ++i)
                    for (int r = 0; r < revisit; ++r) push_rd(i, rd_n4, false, engine_pat);
            } else {
                for (int r = 0; r < revisit; ++r) {
                    for (int i = 0; i < nbuf; ++i) {
                        push_rd(i, rd_n4, false, engine_pat);
                        push_dummies_up_to((int)((long)(r * nbuf + i + 1) * dummy_rl / E));
                    }
                }
                push_dummies_up_to(dummy_rl);
            }
        } else {
            for (int i = 0; i < rdl; ++i) {
                push_rd((size_t)(streams == 2 ? 2 * i : i), rd_n4, streams == 2, engine_pat);
                if (wafter) push_wr(wr_group[i]);
            }
        }
        q.rd_launches = rslot;
        q.sum_slots = std::max(1, rslot);
        // The checksum target: every float4 every read launch of one iteration has to touch.
        q.chk_rd_n4 = rd_n4 * (size_t)rd_regions_read + dummy_n4 * (size_t)dummy_rl;
        q.read_bytes = (double)q.chk_rd_n4 * 16.0;
        q.write_bytes = (double)wr_total * 16.0;
        q.ratio_act = q.write_bytes > 0 ? q.read_bytes / q.write_bytes : 0.0;
        cands.push_back(q);
    }

    double best_cov = 0.0;
    for (const Plan& q : cands) best_cov = std::max(best_cov, q.read_bytes + q.write_bytes);
    if (best_cov <= 0.0) { fprintf(stderr, "geometry too small: no block count 8..%d fits\n", want); exit(2); }
    Plan best = cands.front();
    for (const Plan& q : cands) {                    // candidates run largest grid first
        if (q.read_bytes + q.write_bytes >= 0.995 * best_cov) { best = q; break; }
    }
    if (best.rd_n4 == 0 && c.mode != "write") {
        fprintf(stderr, "read region smaller than one grid pass (or smaller than one %zu-float4 "
                        "weight row at --pattern %s)\n", row_f4, c.pattern.c_str());
        exit(2);
    }
    if (c.mode == "interleave" && best.dummy_rd_n4 == 0) {
        fprintf(stderr, "stand-in buffer smaller than one grid pass\n"); exit(2);
    }
    if (best.wr_n4 == 0 && c.mode != "read" && c.mode != "interleave") {
        fprintf(stderr, "write region smaller than one launch\n"); exit(2);
    }
    if (c.mode == "mix" && c.nbuf > 1) best.fit = "nbuf_forced_1(mix_is_one_region)";
    if (best.wr_buf_n4 > 0 && best.wr_n4 > best.wr_buf_n4 && best.fit.empty()) best.fit = "write_arena_cycled";
    if (best.blocks < 80 && best.fit.empty() && (c.mode == "mix" || c.mode == "mixlaunch")) {
        best.fit = "blocks_below_80(ratio_too_steep_for_footprint)";
    }
    return best;
}

// ---- device side -----------------------------------------------------------------------
// A region is a usable span with an optional poison band on each side. The bands exist so that a
// kernel which wanders is caught by the byte account rather than by luck: a stray READ lands on
// -1.0f and moves the checksum, a stray WRITE is found where it should still be poison. Only
// --selftest asks for bands (they are extra pages and extra bytes on the allocation, and a timing
// row must not pay for them).
struct Region {
    float4* base = nullptr;      // what hipMalloc returned (guards included)
    float4* p = nullptr;         // first usable float4
    size_t n4 = 0;               // usable float4s
    size_t guard4 = 0;           // poison float4s on each side
};

constexpr size_t kGuard4 = 2048;   // 32 KiB of poison per side

struct Dev {
    std::vector<Region> rd;     // read buffers (nbuf of them; for wthenr+overlap, wr aliases rd[0])
    Region wr;                  // write arena
    bool wr_alias = false;      // wr IS rd[0]: the just-dirtied lines are the lines being read
    Region stage;               // engine pattern: the all-zero stand-in for the staged x row
    double* sums = nullptr;     // per-READ-LAUNCH x per-block partials (sum_slots x blocks)
};

inline float4* first_rd(const Dev& d) { return d.rd.empty() ? nullptr : d.rd[0].p; }

int inject_code(const Cfg& c) {
    if (c.inject == "short_read") return kInjShortRead;
    if (c.inject == "miss_write") return kInjMissWrite;
    if (c.inject == "stray_read") return kInjStrayRead;
    if (c.inject == "stray_write") return kInjStrayWrite;
    if (c.inject == "short_row") return kInjShortRow;
    if (c.inject == "skip_launch") return kInjSkipLaunch;   // host-side: launch_once drops it
    return kInjNone;
}

// `fill` writes the values the byte accounts expect, and that write pass is what warms the pages; a
// cold run passes fill=false and touches nothing, which is the whole point of its row. `guards`
// adds the poison bands (selftest only, see Region).
Dev alloc_dev(const Cfg& c, const Plan& p, bool fill, bool guards = false) {
    Dev d;
    const size_t g4 = guards ? kGuard4 : 0;
    const bool alias = (c.mode == "wthenr" && c.overlap);
    auto make = [&](size_t n4) {
        Region r; r.n4 = n4; r.guard4 = g4;
        CK(hipMalloc(&r.base, (g4 + n4 + g4) * 16));           // hipMalloc, never managed
        r.p = r.base + g4;
        return r;
    };
    auto band = [&](float4* at, size_t n4, float v) {
        if (n4) write_k<<<64, 256>>>(at, n4, v);
    };
    for (int i = 0; i < p.rd_regions; ++i) d.rd.push_back(make(p.region_n4(i)));
    d.wr_alias = alias;
    if (alias) { d.wr = d.rd.empty() ? Region() : d.rd[0]; }    // read back the just-dirtied lines
    else if (p.wr_buf_n4) d.wr = make(p.wr_buf_n4);
    if (p.stage_f4) {
        // The stand-in for the activation row the engine stages into LDS. Zeros, like the write arena
        // is pre-filled with a value writes never store: the staged bytes must contribute EXACTLY 0.0
        // to the checksum, so they can never make it pass by construction.
        d.stage = make(p.stage_f4);
        band(d.stage.p, d.stage.n4, kUnwrittenVal);
    }
    // One block-reduced partial per READ LAUNCH per block, not one per block: billing every launch is
    // what makes a launch that never ran move the checksum. See the Q8 lesson in the file header.
    CK(hipMalloc(&d.sums, sizeof(double) * (size_t)p.sum_slots * (size_t)p.blocks));
    CK(hipMemset(d.sums, 0, sizeof(double) * (size_t)p.sum_slots * (size_t)p.blocks));
    for (const Region& r : d.rd) {                              // read side: 1.0f, the checksum source
        band(r.base, r.guard4, kGuardVal);
        band(r.p + r.n4, r.guard4, kGuardVal);
        if (fill) band(r.p, r.n4, kFillVal);
    }
    if (!alias && d.wr.n4) {                                    // write side: a value writes never store
        band(d.wr.base, d.wr.guard4, kGuardVal);
        band(d.wr.p + d.wr.n4, d.wr.guard4, kGuardVal);
        if (fill) band(d.wr.p, d.wr.n4, kUnwrittenVal);
    }
    if (d.stage.n4) {
        band(d.stage.base, d.stage.guard4, kGuardVal);
        band(d.stage.p + d.stage.n4, d.stage.guard4, kGuardVal);
    }
    CK(hipDeviceSynchronize());                                 // one sync: this pass is the warming
    return d;
}

void free_dev(Dev& d) {
    for (const Region& r : d.rd) if (r.base) CK(hipFree(r.base));
    if (!d.wr_alias && d.wr.base) CK(hipFree(d.wr.base));
    if (d.stage.base) CK(hipFree(d.stage.base));
    if (d.sums) CK(hipFree(d.sums));
    d = Dev();
}

// Did every byte of poison survive? A kernel that wrote outside its region shows up here, and a
// witness that cannot fail is not a witness, so the caller gets the region name it found dirty.
bool guards_intact(const Dev& d, std::string* why) {
    // A float4 band is 4 floats per slot: the host buffer has to be sized in FLOATS, not in float4s,
    // or the copy runs 4x past a heap allocation and the witness corrupts its own process instead of
    // reporting the kernel that strayed.
    std::vector<float> h(kGuard4 * 4);
    auto check = [&](const char* name, const Region& r) {
        if (!r.base || !r.guard4) return true;
        const float4* bands[2] = {r.base, r.p + r.n4};
        for (const float4* b : bands) {
            CK(hipMemcpy(h.data(), b, r.guard4 * 16, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < r.guard4 * 4; ++i) {
                if (h[i] != kGuardVal) {
                    *why = std::string("poison_written_by_stray_store(") + name + ")";
                    return false;
                }
            }
        }
        return true;
    };
    for (size_t i = 0; i < d.rd.size(); ++i)
        if (!check(("rd" + std::to_string(i)).c_str(), d.rd[i])) return false;
    if (!d.wr_alias && d.wr.base && !check("wr", d.wr)) return false;
    if (d.stage.base && !check("stage", d.stage)) return false;
    return true;
}

// One iteration of the mode's launch schedule, exactly as resolve() laid it out. Kept apart from
// the timing so the cold / warm / write-then-read variants all share one launch shape.
//
// Note on what `mixlaunch` measures: launches on one stream are serialized, so its read and write
// passes do NOT overlap in flight — that is the engine's own shape (one stream, 406 launches per
// token), and it is a different question from `mix`, where both streams are in flight inside one
// kernel. Rows carry --mode for exactly this reason; do not compare across them as if they were one
// experiment.
void launch_once(const Cfg& c, const Plan& p, Dev& d, bool reads, bool writes) {
    // Dynamic shared for the block reduction: one double per thread, and only when verifying.
    const size_t smem = c.verify ? (size_t)p.threads * sizeof(double) : 0;
    if (c.mode == "mix") {
        if (reads && writes)
            mix_k<<<p.blocks, p.threads, smem>>>(first_rd(d), d.wr.p, p.rounds, (size_t)c.rrd,
                                                 (size_t)c.rwr, p.wrounds,
                                                 c.verify ? d.sums : nullptr, c.verify,
                                                 inject_code(c));
        return;
    }
    int nrd = 0;
    for (const Plan::Step& st : p.sched) {
        if (st.kind == 0) {
            if (!reads) continue;
            ++nrd;
            // kInjSkipLaunch: drop ONE read launch of the schedule (the second). A whole launch that
            // never ran is the Q8 failure mode, and it is invisible to an account that only bills the
            // last pass, so this injection is the anti-vacuity proof for the per-launch slots.
            const bool skip = (c.inject == "skip_launch" && nrd == 2);
            if (skip) continue;
            double* sums = c.verify ? d.sums + (size_t)st.slot * (size_t)p.blocks : nullptr;
            if (st.engine) {
                read_engine_k<<<st.blocks, p.threads,
                                p.stage_f4 * 16 + smem>>>(
                    d.rd[st.obj].p, st.obj2 >= 0 ? d.rd[st.obj2].p : nullptr, st.n4, p.row_f4,
                    p.stage_f4, d.stage.p, sums, c.verify, inject_code(c));
            } else {
                read_k<<<st.blocks, p.threads, smem>>>(d.rd[st.obj].p, st.n4, sums, c.verify);
            }
        } else {
            if (!writes) continue;
            write_k<<<st.blocks, p.threads>>>(d.wr.p + st.off, st.n4, kFillVal);
            if (c.mode == "wthenr" && c.pair_sync) CK(hipDeviceSynchronize());
        }
    }
}


// steady_clock burst, warm-up OUTSIDE the timed region, ONE sync at the end. Event timers are not
// an option here (they read exactly 0 below ~1 ms on this iGPU).
double burst(const Cfg& c, const Plan& p, Dev& d, bool reads, bool writes, int iters, bool warmup) {
    if (warmup) {
        for (int i = 0; i < 2; ++i) { launch_once(c, p, d, reads, writes); CK(hipDeviceSynchronize()); }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) launch_once(c, p, d, reads, writes);
    CK(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)iters;
}

// Exact checksum, summed over the slots of `lo` .. `hi-1`: the buffers hold 1.0f, every block reduced
// its own partial in double, so the total is an exact integer equal to 4.0 x float4s-read and the
// order of accumulation cannot matter. Nothing here has a tolerance to loosen.
bool readback(const Dev& d, const Plan& p, int lo, int hi, double* sum_out) {
    const size_t n = (size_t)(hi - lo) * p.blocks;
    std::vector<double> h(n, 0.0);
    CK(hipMemcpy(h.data(), d.sums + (size_t)lo * p.blocks, sizeof(double) * n, hipMemcpyDeviceToHost));
    double s = 0.0;
    for (double v : h) s += v;
    *sum_out = s;
    return true;
}

// The byte accounts of one finished run. Two separate questions, because one row has to be able to
// fail one of them: did the kernel READ the bytes it claims, and did it WRITE them. `--selftest`
// injects a bug into each side and requires the other side to stay green, which is the only way to
// show these are two accounts and not one alarm that rings for anything.
struct Accounts {
    bool has_read = false, has_write = false;   // is that side verifiable at all
    bool read_ok = true, write_ok = true, guards_ok = true;
    double rd_sum = 0, rd_want = 0, wr_sum = 0, wr_want = 0;
    std::string write_note = "none";           // how the write side was proved
    std::string why = "-";                     // guard detail
    bool ok() const { return read_ok && write_ok && guards_ok; }
};

// Read the accounts back off the device. `sums` holds one set of block partials PER READ LAUNCH of
// the last read pass, so the read target is every float4 the iteration's read launches touch
// (`chk_rd_n4`), not just the last launch's: a build that never launched 47 of 96 reads passed a
// checksum that only billed the last pass, so this account bills the whole schedule. The write arena
// is then read back UNTIMED into slot 0 and must hold the value the write stream stores in every byte
// (it was pre-filled with a value writes never store, so a missing write is visible rather than
// merely unwitnessed). For a cycling arena the proof is weaker in exactly one way - a slot written
// twice is indistinguishable from a slot written once - and `write_note` says so instead of
// rounding the claim up.
Accounts verify_accounts(const Cfg& c, const Plan& p, Dev& d, bool reads, bool writes, bool guards) {
    Accounts a;
    if (guards && !guards_intact(d, &a.why)) a.guards_ok = false;
    if (!c.verify) {
        a.write_note = "verify_off";
        return a;
    }
    const size_t smem = (size_t)p.threads * sizeof(double);
    if (reads && p.chk_rd_n4) {
        a.has_read = true;
        readback(d, p, 0, p.sum_slots, &a.rd_sum);
        a.rd_want = 4.0 * (double)p.chk_rd_n4;
        a.read_ok = (a.rd_sum == a.rd_want);
    }
    if (!writes || !p.wr_buf_n4) {
        a.write_note = "no_write_stream";
    } else if (d.wr_alias) {
        // wthenr --overlap 1 reads back the lines it just dirtied, so the read checksum cannot tell
        // a landed write from the pre-fill. Say that instead of claiming a proof that does not exist.
        a.write_note = "unverifiable_lines_are_the_read_lines";
    } else {
        a.has_write = true;
        read_k<<<p.blocks, p.threads, smem>>>(d.wr.p, p.wr_buf_n4, d.sums, 1);
        CK(hipDeviceSynchronize());
        readback(d, p, 0, 1, &a.wr_sum);
        a.wr_want = 4.0 * (double)p.wr_buf_n4;
        a.write_ok = (a.wr_sum == a.wr_want);
        // Never label a short sum "exact": the landed= field is what a reader of the log trusts, and a
        // clean-looking word next to a mismatching account is how an instrument starts lying.
        a.write_note = (!a.write_ok)
                           ? ((p.wr_n4 > p.wr_buf_n4) ? "short_within_cycled_arena" : "short")
                           : ((p.wr_n4 > p.wr_buf_n4) ? "arena_slots_only_cycled" : "exact");
    }
    return a;
}

// The single field a reader of a MIXEDBW line gates on, and the exit code behind it. `ok` is the
// only value that means verified; every other value names WHICH account is missing rather than
// gesturing at "skipped", and none of them may coexist with exit code 0 (rule 11 of
// devices/strix-halo-mixed-traffic.md: a row whose bytes were not verified carries no conclusion).
std::string checksum_field(const Accounts& a, bool cold) {
    if (cold) return "unverifiable_cold_pages";      // pre-filling would warm the pages being measured
    if (!a.guards_ok) return "FAIL_" + a.why;
    if (!a.read_ok) return "FAIL_reads_short";
    if (!a.write_ok) return "FAIL_writes_missing";
    if (!a.has_read && !a.has_write) return "verify_off";
    if (!a.has_write && a.write_note == "unverifiable_lines_are_the_read_lines")
        return "ok_writes_aliased";                 // read side verified, write side cannot be
    return "ok";
}

// A row may only print with exit code 0 if its bytes were verified. `ok` and `ok_writes_aliased`
// (read side proven, write side physically unprovable because it aliases the read lines) are the
// two acceptable values; anything else is a loud non-zero, because a field a reader can walk past is
// how the first Q8 sweep turned 13 plausible rows into zero conclusions.
bool row_accepted(const std::string& ck) {
    return ck == "ok" || ck == "ok_writes_aliased";
}

int g_suspect = 0, g_bad = 0;
constexpr double COPY_PROBE_GBS = 210.0;    // same file: iGPU copy (1:1 read+write) best

void row(const Cfg& c, const Plan& p, const char* pages, int iters, double ms, double rb,
         double wb, const std::string& extra) {
    const double tot = rb + wb;
    const double gbs = tot / (ms * 1e6);              // bytes / (ms/1000) / 1e9
    const double rd_gbs = rb / (ms * 1e6), wr_gbs = wb / (ms * 1e6);
    const int suspect = gbs > c.suspect_gbs;
    if (suspect) ++g_suspect;
    // The staged x row is LDS traffic: real, paid for, and deliberately NOT part of `gbs`, which
    // counts DRAM bytes only — the same convention the engine's own roofline uses (weight bytes).
    const double stage_mib = p.stage_f4 * 16.0 * p.blocks * p.rd_launches / MI;
    printf("MIXEDBW tag=%s mode=%s ratio=%ld:%ld ratio_act=%.3f pages=%s iters=%d blocks=%d "
           "threads=%d launches=%d footprint_mib=%.3f read_mib=%.3f write_mib=%.3f wrbuf_mib=%.3f "
           "rounds=%zu passes=%zu wrounds=%zu pattern=%s row_f4=%zu stage_mib=%.2f "
           "stage_witness=%s order=%s revisits=%d slots=%d ms=%.4f gbs=%.1f rd_gbs=%.1f "
           "wr_gbs=%.1f vs_read_probe_pct=%.1f vs_copy_probe_pct=%.1f suspect=%d %s\n",
           c.tag.c_str(), c.mode.c_str(), c.rrd, c.rwr, p.ratio_act, pages, iters, p.blocks,
           p.threads, p.rd_launches + p.wr_launches, tot / MI, rb / MI, wb / MI,
           p.wr_buf_n4 * 16.0 / MI, p.rounds, p.passes, p.wrounds,
           c.pattern.c_str(), p.row_f4, stage_mib,
           p.stage_f4 ? "zero_by_construction" : "none",
           c.mode == "interleave" ? c.order.c_str() : "-", c.revisits, p.sum_slots,
           ms, gbs, rd_gbs, wr_gbs,
           100.0 * gbs / READ_PROBE_GBS, 100.0 * gbs / COPY_PROBE_GBS, suspect, extra.c_str());
    fflush(stdout);
    if (suspect)
        fprintf(stderr, "NOTE: %.1f GB/s is above the %.0f GB/s measured ceiling: either the kernel "
                        "skipped work or the footprint is cache-resident (the 18 MiB GDN state reads "
                        "at ~918 GB/s warm). Treat as suspect, not as a new record.\n",
                gbs, c.suspect_gbs);
}

// ---- reporting / self-checks -----------------------------------------------------------
// The invariants that make a row trustworthy. Shared by --dry-run (no device) and --selftest.
// One of them is the bounds model, i.e. the indexing the kernels will actually compute; the rest are
// the geometry the solver promised.
bool invariants(const Cfg& c, const Plan& p, std::string* why) {
    auto bad = [&](const std::string& s) { *why = s; return false; };
    if (p.blocks < 8) return bad("blocks_collapsed_below_8");
    if (c.mode == "read" && (p.rd_n4 == 0 || p.wr_n4 != 0)) return bad("read_side_empty");
    if (c.mode == "interleave" && p.wr_n4 != 0) return bad("interleave_has_no_write_stream");
    if (c.mode == "write" && p.wr_n4 == 0) return bad("write_side_empty");
    if (c.mode != "read" && c.mode != "write" && c.mode != "interleave" && p.wr_n4 == 0)
        return bad("write_stream_dropped");
    if (p.wr_n4 > 0 && p.wr_buf_n4 == 0) return bad("no_write_arena");
    if (c.mode == "mix") {
        if (p.rounds < 1) return bad("zero_rounds");
        const double want = (double)c.rrd / (double)c.rwr;
        if (std::fabs(p.ratio_act - want) > 1e-9 * want) return bad("ratio_not_exact");
    }
    if (c.mode == "mixlaunch" && p.ratio_act <= 0.0) return bad("no_traffic");
    if (p.fit == "write_stream_dropped") return bad("write_stream_dropped");
    if (p.read_bytes + p.write_bytes < 1.0 * MI) return bad("footprint_below_1MiB");
    // The wiring the two Q9 witnesses depend on. One checksum slot per read launch, or a skipped
    // launch could share a slot with another and hide; and the engine mapping must be whole rows of
    // the width the engine actually instantiates, or it is a mapping nobody runs.
    int rd_steps = 0, wr_steps = 0, rd_engine_steps = 0, rd_engine_wanted = 0;
    for (const Plan::Step& st : p.sched) {
        if (st.kind == 0) {
            if (st.slot >= p.sum_slots) return bad("checksum_slot_out_of_range");
            // Regions below rd_engine_regions are the ones this configuration is meant to walk with
            // the engine's mapping; the interleave mode's stand-ins are other families and stay plain
            // unless --dummy-same says otherwise.
            if (st.obj < p.rd_engine_regions) ++rd_engine_wanted;
            if (st.engine) {
                ++rd_engine_steps;
                if (!p.row_f4) return bad("engine_pattern_without_row_width");
                if (st.n4 % p.row_f4) return bad("engine_pattern_partial_row");
                if (st.obj2 >= 0 && st.obj2 != st.obj + 1) return bad("engine_pattern_pair_not_adjacent");
            }
            ++rd_steps;
        } else {
            ++wr_steps;
        }
    }
    // The flag has to reach the launches that stand for the MLP. `--pattern engine-mlp` that silently
    // still ran the plain reader is this instrument's vacuous form, and the byte account cannot catch
    // it: both readers touch the same bytes, so only the geometry can say the mapping was applied.
    if (c.pattern == "engine-mlp" && rd_engine_wanted == 0) return bad("engine_pattern_has_no_read_launches");
    if (c.pattern == "engine-mlp" && rd_engine_steps != rd_engine_wanted)
        return bad("engine_pattern_not_applied_to_every_read_launch");
    if (p.rd_launches != rd_steps) return bad("read_launch_count_disagrees_with_schedule");
    if (p.wr_launches != wr_steps) return bad("write_launch_count_disagrees_with_schedule");
    if (p.rd_regions < 1 && c.mode != "write") return bad("no_read_regions_allocated");
    if (c.mode == "interleave") {
        if (p.dummy_launches < 1 || p.dummy_rd_n4 == 0) return bad("interleave_without_stand_ins");
        if (p.rd_regions != p.rd_engine_regions + p.dummy_launches) return bad("interleave_regions_mismatch");
    }
    const Bounds b = plan_bounds(c, p);
    if (!b.ok) return bad(b.why);
    return true;
}

void usage() {
    fprintf(stderr,
        "usage: mixed_bw [flags]   (one machine-readable MIXEDBW line per page state)\n"
        "  --mode read|write|mix|mixlaunch|wthenr\n"
        "      read/write    pure stream over --footprint-mib\n"
        "      mix           rrd reads per rwr writes in ONE kernel, one read region + a cycling\n"
        "                    write arena (the primary ratio instrument)\n"
        "      mixlaunch     nbuf read launches with proportional write launches interleaved\n"
        "                    (the MLP's 48 x 11.01 MB shape at the step's ratio)\n"
        "      wthenr        write pass immediately followed by a read pass; also times each alone\n"
        "      interleave    a pool of read buffers touched in ONE of two orders at identical total\n"
        "                    bytes, the only free variable being the order: `blocked` revisits each\n"
        "                    buffer back to back (best locality), `roundrobin` walks the pool pass-major\n"
        "                    with --dummy-mib/--dummy-nbuf stand-ins embedded between the launches\n"
        "  --pattern contiguous|engine-mlp  WHICH thread touches WHICH byte on the read side.\n"
        "                    engine-mlp walks the bytes with mlp_gateup/mlp_down's own weight\n"
        "                    addressing (math removed): --engine-in 1024 = gate+up, two regions per\n"
        "                    launch; --engine-in 3584 = down, one. --stage 0|1 turns the engine's\n"
        "                    `__shared__ float xs[IN]` staging and &xs[v*8] re-reads on or off.\n"
        "  --interleave roundrobin|blocked  the order interleave touches its regions in: roundrobin\n"
        "                    walks the pool pass-major with the stand-ins spread between launches;\n"
        "                    blocked re-walks each region back to back (best locality)\n"
        "  --revisits K      interleave: engine passes over its buffers per iteration\n"
        "  --dummy-mib D --dummy-nbuf Z      interleave stand-in stream (other families' traffic)\n"
        "  --dummy-same 0|1  interleave: stand-ins use the same mapping as the engine buffers\n"
        "  --ratio R|N:D     read:write bytes (20, or 1:2 for a write-heavy cast-like mix)\n"
        "  --footprint-mib F | --nbuf N --buf-mib B   geometry; B>0 sets footprint = N*B. The\n"
        "                    footprint_mib in the output line is COUNTED bytes = read_mib +\n"
        "                    write_mib; with --overlap 1 the same lines count once as written\n"
        "                    and once as read.\n"
        "  --wr-arena-mib A  mix: size of the arena the writes cycle through (0 = one write pass)\n"
        "  --pages both|warm|cold   cold = first GPU pass over fresh pages, its OWN row, 1 iteration\n"
        "  --iters N --blocks B --threads T           grid/burst (B=0 auto: 640, or 448 if nbuf>1)\n"
        "  --pair-sync 0|1   wthenr: sync between the write and read passes (0 = let them pipeline)\n"
        "  --overlap 0|1     wthenr: read back the just-dirtied lines (default 1)\n"
        "  --verify 0|1 --suspect-gbs G --strict      byte accounts + >240 GB/s flag (see file header)\n"
        "  --inject short_read|miss_write  mix only; the anti-vacuity bugs --selftest uses (each is\n"
        "                    caught by exactly one of the two byte accounts, and must be)\n"
        "  --inject short_row  engine-mlp only: one whole weight row of one launch is never read.\n"
        "                    Inside the buffer, whole, and invisible to every witness but the read\n"
        "                    account -- the anti-vacuity proof for the engine-pattern bounds model\n"
        "  --inject skip_launch  interleave only: one read launch of the schedule never runs, which\n"
        "                    only the per-launch checksum slots can see (a last-pass account cannot)\n"
        "  --inject stray_read|stray_write the guard band's own negative control; needs\n"
        "                    --selftest-case, because without the bands they fault (loudly) by design\n"
        "  --tag S | --dry-run (print the geometry, touch no device) | --selftest-geom (CPU) |\n"
        "  --selftest (CPU geometry + one untimed device case per child process; prints no ms/GB/s) |\n"
        "  --selftest-case N (device case N alone; what --selftest forks)\n"
        "checksum= on every row is the byte account: ok is the only value that means verified.\n"
        "FAIL_reads_short / FAIL_writes_missing / unverifiable_cold_pages / verify_off all exit 3.\n"
        "EXIT: 0 all rows verified | 2 bad args or geometry | 3 a check failed or a row went\n"
        "unverified | 4 suspect row with --strict\n");
}

void print_plan(const Cfg& c, const Plan& p) {
    std::string why;
    const bool ok = invariants(c, p, &why);
    printf("PLAN mode=%s blocks=%d threads=%d stride=%zu nbuf=%d rd_n4=%zu rd_launches=%d "
           "rd_mib=%.3f wr_n4=%zu wr_chunk_n4=%zu wr_launches=%d wr_mib=%.3f wrbuf_mib=%.3f "
           "rounds=%zu passes=%zu wrounds=%zu ratio_req=%ld:%ld ratio_act=%.4f footprint_mib=%.3f "
           "pattern=%s engine_in=%d stage=%d row_f4=%zu stage_f4=%zu rdregions=%d slots=%d "
           "dummy_rd_n4=%zu dummy_launches=%d order=%s revisits=%d "
           "rd_end=%zu wr_end=%zu fit=%s\n",
           c.mode.c_str(), p.blocks, p.threads, p.stride, p.nbuf, p.rd_n4, p.rd_launches,
           p.read_bytes / MI, p.wr_n4, p.wr_chunk_n4, p.wr_launches, p.write_bytes / MI,
           p.wr_buf_n4 * 16.0 / MI, p.rounds, p.passes, p.wrounds, c.rrd, c.rwr, p.ratio_act,
           (p.read_bytes + p.write_bytes) / MI,
           c.pattern.c_str(), c.engine_in, c.stage, p.row_f4, p.stage_f4, p.rd_regions,
           p.sum_slots, p.dummy_rd_n4, p.dummy_launches, c.order.c_str(), c.revisits,
           plan_bounds(c, p).rd_end, plan_bounds(c, p).wr_end,
           p.fit.empty() ? "-" : p.fit.c_str());
    printf("PLAN_CHECK mode=%s ok=%d %s\n", c.mode.c_str(), ok ? 1 : 0, ok ? "-" : why.c_str());
    fflush(stdout);
}

// ---- one run ---------------------------------------------------------------------------
int run(const Cfg& c) {
    const Plan p = resolve(c);
    if (c.dry) {
        std::string why;
        print_plan(c, p);
        return invariants(c, p, &why) ? 0 : 3;
    }
    CK(hipInit(0));
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    fprintf(stderr, "# %s | mode=%s ratio=%ld:%ld footprint=%.1f MiB pages=%s\n",
            prop.name, c.mode.c_str(), c.rrd, c.rwr,
            (p.read_bytes + p.write_bytes) / MI, c.pages.c_str());

    const bool rd = (c.mode != "write");
    const bool wr = (c.mode != "read");
    const bool do_cold = (c.pages == "both" || c.pages == "cold");
    const bool do_warm = (c.pages == "both" || c.pages == "warm");

    if (do_cold) {
        // Pages allocated and never touched: any pre-fill or read-back that could prove the bytes
        // would warm them first, so this row is UNVERIFIABLE by construction and says so in the one
        // field a reader gates on, and exits non-zero. The freshness witness is only measurable for a
        // read row (the bytes it read were all zero); a write pass overwrites the only evidence, so
        // its claim is structural: this process issued no access to those pages before the launch.
        Dev d = alloc_dev(c, p, false);                  // pages allocated and never touched
        const int it = 1;                                // one pass; a second pass would be warm
        const double ms = burst(c, p, d, rd, wr, it, false);   // no warm-up: that IS the measurement
        char buf[192];
        if (rd && c.verify) {
            double sum = 0.0;
            readback(d, p, 0, p.sum_slots, &sum);
            snprintf(buf, sizeof buf, "raw_sum=%.0f fresh_zero_sum=%d", sum, sum == 0.0 ? 1 : 0);
            if (sum != 0.0) ++g_bad;                     // rule 10: pages were not fresh, throw it out
        } else {
            snprintf(buf, sizeof buf, "fresh_zero_sum=not_observable_write_pass "
                                      "fresh_alloc_untouched=1");
        }
        const std::string ck = checksum_field(Accounts(), true);
        ++g_bad;                                         // cold rows never exit 0; see header
        row(c, p, "cold", it, ms, p.read_bytes, p.write_bytes,
            "checksum=" + ck + " " + buf);
        fprintf(stderr, "NOTE: the cold row's bytes could not be verified (verifying them would warm "
                        "them). Per rule 11 no conclusion may be drawn from it; exit is non-zero.\n");
        free_dev(d);
    }

    if (do_warm) {
        Dev d = alloc_dev(c, p, true);                   // the fill pass is what warms the pages
        std::string extra;
        double ms = 0.0;
        if (c.mode == "wthenr") {
            // Three bursts, one row: what the write costs alone, what the read costs alone, and the
            // pair. The byte accounts come off the PAIR, which is the number the row reports.
            const double ms_w = burst(c, p, d, false, true, c.iters, true);
            const double ms_r = burst(c, p, d, true, false, c.iters, true);
            ms = burst(c, p, d, true, true, c.iters, true);
            const Accounts a = verify_accounts(c, p, d, true, true, false);
            const std::string ck = checksum_field(a, false);
            if (!row_accepted(ck)) ++g_bad;
            char buf[384];
            snprintf(buf, sizeof buf,
                     "checksum=%s rd_sum=%.0f rd_want=%.0f wr_landed=%s ms_write_alone=%.4f "
                     "ms_read_alone=%.4f ms_pair=%.4f pair_credit_ms=%.4f "
                     "read_dirty_penalty_pct=%.1f pair_sync=%d overlap=%d",
                     ck.c_str(), a.rd_sum, a.rd_want, a.write_note.c_str(), ms_w, ms_r, ms,
                     ms_w + ms_r - ms, ms_r > 0 ? 100.0 * (ms - ms_w) / ms_r : 0.0,
                     c.pair_sync, c.overlap);
            extra = buf;
        } else {
            ms = burst(c, p, d, rd, wr, c.iters, true);
            const Accounts a = verify_accounts(c, p, d, rd, wr, false);
            const std::string ck = checksum_field(a, false);
            if (!row_accepted(ck)) ++g_bad;
            char buf[256];
            snprintf(buf, sizeof buf, "checksum=%s rd_sum=%.0f rd_want=%.0f wr_sum=%.0f wr_want=%.0f "
                                      "wr_landed=%s",
                     ck.c_str(), a.rd_sum, a.rd_want, a.wr_sum, a.wr_want, a.write_note.c_str());
            extra = buf;
            if (ck == "unverifiable_cold_pages" || ck == "verify_off" || ck.rfind("FAIL", 0) == 0)
                fprintf(stderr, "NOTE: checksum=%s - the kernel did not touch exactly the bytes the "
                                "row bills. No conclusion may be drawn from this line.\n", ck.c_str());
        }
        row(c, p, "warm", c.iters, ms, p.read_bytes, p.write_bytes, extra);
        free_dev(d);
    }
    if (g_bad) return 3;
    if (g_suspect && c.strict) return 4;
    return 0;
}

// ---- selftest --------------------------------------------------------------------------
// Three halves, and the split is the point.
//
//   * The geometry half (`--selftest-geom`, CPU) runs the invariants — including the bounds model,
//     i.e. the index arithmetic the kernels will actually compute — over the sweep's own config
//     table. Cheap, touches no device, catches a rounded ratio or a dropped write stream.
//   * The bounds model's own anti-vacuity (CPU): restore the exact bug that shipped, where `rounds`
//     meant "grid passes" to the solver and "trips of the ratio loop" to the kernel, and require the
//     model to REJECT it. A check that never fires is not a check (AGENTS.md's negative test that
//     loads another layer's weights "catches" every bug and therefore nothing).
//   * The device half is a byte witness, not an arithmetic print. For every configuration that
//     faulted or failed in the first Q8 sweep it allocates with poison guards, runs ONE untimed pass,
//     and demands three things: the read checksum equals the plan's read account exactly, every arena
//     byte holds the value the write stream stores, and no poison byte was read or written. It prints
//     no ms and no GB/s, because a gate that also reports a bandwidth number is a measurement wearing
//     a costume and bandwidth on this box belongs to the main agent.
//
// Each device case runs in a forked child, and that is not decoration: a device memory fault poisons
// the context, so in one process the first dead configuration ends the run and every configuration
// after it is silently unreported — which is how `--selftest` shipped: the first sweep's log shows it
// printing one failing line and then faulting. The parent never touches the device, so each child
// starts from a clean context and one command yields a verdict per configuration. `--selftest-case N`
// is the child's entry point, and a convenient way to look at one configuration by hand.
struct GeomCase {
    const char* mode; long rrd, rwr; double mib, buf_mib, arena_mib; int nbuf; int overlap;
    // Q9 additions. Trailing members are zero/null for the rows that predate them, and cfg_of /
    // selftest_geom only copy them when they are set, so the existing table rows keep their meaning.
    const char* pattern = nullptr; int engine_in = 0; int stage = -1; int blocks = 0;
    int revisits = 0; double dummy_mib = 0.0; int dummy_nbuf = 0; const char* order = nullptr;
};
static const GeomCase kGeomCases[] = {
    //   mode         rrd  rwr  mib   buf_mib arena nbuf overlap
    {"read",      20, 1, 128,  0,     0,  1, 1},
    {"read",      20, 1, 528,  0,     0, 48, 1},
    {"read",      20, 1,  0,   11.01, 0, 48, 1},   // C_read_48x11
    {"write",     20, 1, 128,  0,     0,  1, 1},
    {"write",     20, 1, 528,  0,     0,  1, 1},   // A_write_512m
    {"mix",        1, 1, 528,  0,     0,  1, 1},   // B_mix_*: every row that faulted
    {"mix",        4, 1, 528,  0,     0,  1, 1},
    {"mix",       10, 1, 528,  0,     0,  1, 1},
    {"mix",       20, 1, 528,  0,     0,  1, 1},
    {"mix",       32, 1, 528,  0,     0,  1, 1},
    {"mix",       62, 1, 528,  0,     0,  1, 1},
    {"mix",      128, 1, 528,  0,     0,  1, 1},
    {"mix",       62, 1, 528,  0,     2,  1, 1},   // D_62_arena2m: the cycling write arena
    {"mix",       20, 1, 528,  0,     2,  1, 1},   // D_20_arena2m
    {"mix",       20, 1, 512,  0,     0,  1, 1},
    {"mix",       10, 1, 512,  0,     0,  1, 1},
    {"mix",        1, 2, 128,  0,     0,  1, 1},   // the write-heavy cast-like mix
    {"mix",       20, 1, 528,  0,     0, 48, 1},   // nbuf is forced to 1: mix is one region
    {"mixlaunch", 62, 1,  0,   11.01, 0, 48, 1},   // C_mixlaunch_62: the step's ratio, MLP shape
    {"mixlaunch",  1, 1,  0,   11.01, 0, 48, 1},
    {"mixlaunch", 43, 2, 528,  0,     0, 48, 1},   // the step's write VOLUME on that read shape
    {"mixlaunch", 20, 1, 528,  0,     0, 48, 1},
    {"mixlaunch",128, 1, 528,  0,     0, 48, 1},
    {"wthenr",    20, 1, 512,  0,     0,  1, 1},   // E_same: read back the just-dirtied lines
    {"wthenr",    62, 1, 528,  0,     0,  1, 0},   // E_62_async / E_62_synced: separate regions
    // ---- Q9: the engine's own read mapping, at the engine's own launch geometries -----------
    // gate/up: 48 blobs of exactly 7 MiB = 3584 rows of 2 KiB, read two tensors per wavefront, in
    // 24 launches of 448 blocks x 256 threads — the shape engine/kernels_mlp.cpp launches.
    {"read", 20, 1, 0, 7, 0, 48, 1, "engine-mlp", 1024, 1, 448},
    {"read", 20, 1, 0, 7, 0, 48, 1, "engine-mlp", 1024, 0, 448},  // --stage 0: no LDS re-read
    // down: 24 blobs of 7 MiB = 1024 rows of 7 KiB, one tensor per wavefront, 128 blocks.
    {"read", 20, 1, 0, 7, 0, 24, 1, "engine-mlp", 3584, 1, 128},
    // The decisive C_mixlaunch_62 geometry (48 x 11.01 MiB at 62:1) with the engine's mapping and
    // the step's write stream: 96 regions of 5.505 MiB, paired two per launch.
    {"mixlaunch", 62, 1, 0, 5.505, 0, 96, 1, "engine-mlp", 1024, 1, 0},
    // ---- Q9: the schedule. Same launch multiset, two orders; see resolve()'s interleave branch.
    {"interleave", 20, 1, 0, 32, 0, 8, 1, nullptr, 0, 0, 0, 2, 96, 4, "roundrobin"},
    {"interleave", 20, 1, 0, 32, 0, 8, 1, nullptr, 0, 0, 0, 2, 96, 4, "blocked"},
    {"interleave", 20, 1, 0, 11.01, 0, 48, 1, nullptr, 0, 0, 0, 1, 0.44, 420, "roundrobin"},
    {"interleave", 20, 1, 0, 11.01, 0, 48, 1, nullptr, 0, 0, 0, 1, 0.44, 420, "blocked"},
    {"interleave", 20, 1, 0, 7, 0, 48, 1, "engine-mlp", 3584, 1, 0, 1, 0.44, 420, "roundrobin"},
};

struct DevCase {
    const char* name;
    const char* mode; long rrd, rwr; double mib, buf_mib, arena_mib; int nbuf; int overlap;
    const char* pages; const char* inject; const char* expect;
    // expect: clean | caught_read | caught_write | caught_guards | caught_read_sole
    //   (caught_read needs BOTH streams present and only the read one to fire, which is how the two
    //    accounts are shown to be two; caught_read_sole is for a mode with no write stream at all.)
    // Q9 additions, trailing so the rows above keep their meaning.
    const char* pattern = nullptr; int engine_in = 0; int stage = -1; int blocks = 0;
    int revisits = 0; double dummy_mib = 0.0; int dummy_nbuf = 0; const char* order = nullptr;
};
static const DevCase kDevCases[] = {
    //  name                 mode          rrd  rwr  mib   buf    arena nbuf ov pages inject     expect
    {"read_warm",        "read",       20, 1,  64,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"write_warm",       "write",      20, 1,  64,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"wthenr_overlap",   "wthenr",     20, 1,  64,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"wthenr_62_split",  "wthenr",     62, 1, 512,  0,     0,   1, 0, "warm", "none",       "clean"},
    {"mix_1",            "mix",         1, 1, 528,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"mix_4",            "mix",         4, 1, 528,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"mix_10",           "mix",        10, 1, 528,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"mix_20",           "mix",        20, 1, 528,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"mix_32",           "mix",        32, 1, 528,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"mix_62",           "mix",        62, 1, 528,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"mix_128",          "mix",       128, 1, 528,  0,     0,   1, 1, "warm", "none",       "clean"},
    {"mix_62_arena2m",   "mix",        62, 1, 528,  0,     2,   1, 1, "warm", "none",       "clean"},
    {"mix_20_arena2m",   "mix",        20, 1, 528,  0,     2,   1, 1, "warm", "none",       "clean"},
    {"mixlaunch_48x11_62","mixlaunch", 62, 1,  0,   11.01, 0,  48, 1, "warm", "none",       "clean"},
    {"mixlaunch_48x11_1","mixlaunch",   1, 1,  0,   11.01, 0,  48, 1, "warm", "none",       "clean"},
    {"write_cold",       "write",      20, 1, 512,  0,     0,   1, 1, "cold", "none",       "clean"},
    {"inj_short_read",   "mix",        20, 1,  64,  0,     0,   1, 1, "warm", "short_read", "caught_read"},
    {"inj_miss_write",   "mix",        20, 1,  64,  0,     0,   1, 1, "warm", "miss_write", "caught_write"},
    // The guard bands' own negative control: one float4 over each edge, which only the bands make
    // survivable. Without them these two configurations fault, which is the same loudness by a
    // different route; with them the witness has to be the thing that reports it.
    {"inj_stray_read",   "mix",        20, 1,  64,  0,     0,   1, 1, "warm", "stray_read",  "caught_read"},
    {"inj_stray_write",  "mix",        20, 1,  64,  0,     0,   1, 1, "warm", "stray_write", "caught_guards"},
    // ---- Q9: the engine mapping, at the engine's geometry (8 blobs of 7 MiB: gate/up is 3584 rows
    // of 2 KiB read two-per-wavefront in a 448-block grid, down is 1024 rows of 7 KiB in 128 blocks).
    {"engine_mlp_gateup",     "read",  20, 1,  0,   7,   0,   8, 1, "warm", "none", "clean",
         "engine-mlp", 1024, 1, 448},
    {"engine_mlp_gateup_nostage", "read", 20, 1, 0,  7,   0,   8, 1, "warm", "none", "clean",
         "engine-mlp", 1024, 0, 448},
    {"engine_mlp_down",       "read",  20, 1,  0,   7,   0,   8, 1, "warm", "none", "clean",
         "engine-mlp", 3584, 1, 128},
    // A whole weight row never read: inside the buffer, whole, and invisible to every witness but the
    // per-launch read account.
    {"inj_short_row",         "read",  20, 1,  0,   7,   0,   8, 1, "warm", "short_row",
         "caught_read_sole", "engine-mlp", 1024, 1, 448},
    // ---- Q9: the schedule. `clean` here is the witness that the per-launch slots close for BOTH
    // orders, which is what makes the two orders comparable at all.
    {"interleave_rr",         "interleave", 20, 1, 0, 32, 0,  4, 1, "warm", "none", "clean",
         nullptr, 0, 0, 0, 2, 96, 4, "roundrobin"},
    {"interleave_blocked",    "interleave", 20, 1, 0, 32, 0,  4, 1, "warm", "none", "clean",
         nullptr, 0, 0, 0, 2, 96, 4, "blocked"},
    // One read launch of the schedule never runs. Before the per-launch slots existed this was the
    // exact hole that let a build missing 47 of 96 launches print `ok`.
    {"inj_skip_launch",       "interleave", 20, 1, 0, 32, 0,  4, 1, "warm", "skip_launch",
         "caught_read_sole", nullptr, 0, 0, 0, 2, 96, 4, "roundrobin"},
};

Cfg cfg_of(const DevCase& t) {
    Cfg c;
    c.mode = t.mode; c.rrd = t.rrd; c.rwr = t.rwr; c.footprint_mib = t.mib; c.buf_mib = t.buf_mib;
    c.wr_arena_mib = t.arena_mib; c.nbuf = t.nbuf; c.overlap = t.overlap;
    c.pages = t.pages; c.inject = t.inject;
    if (t.pattern) c.pattern = t.pattern;
    if (t.engine_in) { c.engine_in = t.engine_in; c.engine_in_given = true; }
    if (t.stage >= 0) c.stage = t.stage;
    if (t.blocks) c.blocks = t.blocks;
    if (t.revisits) c.revisits = t.revisits;
    if (t.dummy_mib) c.dummy_mib = t.dummy_mib;
    if (t.dummy_nbuf) c.dummy_nbuf = t.dummy_nbuf;
    if (t.order) c.order = t.order;
    c.iters = 1; c.verify = 1; c.tag = "selftest";
    return c;
}

// One device case, in this (child) process: geometry -> guarded allocation -> one untimed pass ->
// byte accounts. Exits non-zero on any failure; a kernel fault makes CK print the hip error and exit
// 1, which the parent reports as that case failing.
int run_dev_case(const DevCase& t) {
    CK(hipInit(0));
    const Cfg c = cfg_of(t);
    const Plan p = resolve(c);
    std::string why;
    if (!invariants(c, p, &why)) {
        printf("SELFTEST device case=%s ok=0 geometry=%s\n", t.name, why.c_str());
        fflush(stdout);
        return 1;
    }
    const bool rd = (c.mode != "write"), wr = (c.mode != "read");
    Dev d = alloc_dev(c, p, c.pages == "warm", /*guards=*/true);
    launch_once(c, p, d, rd, wr);
    CK(hipDeviceSynchronize());                        // a fault dies here, loudly, naming itself
    const Accounts a = verify_accounts(c, p, d, rd, wr, /*guards=*/true);
    std::string verdict = "ok", detail;
    if (!a.guards_ok) verdict = "guards";
    else if (!a.read_ok) verdict = "read_short";
    else if (!a.write_ok) verdict = "writes_missing";
    // Judged on the accounts, not on whether something screamed: the injected bugs must be caught by
    // ONE account and must NOT be caught by the other. Caught-by-both means the two accounts are one
    // alarm; caught-by-neither means they are decoration. Both are a broken selftest.
    bool pass = (verdict == "ok");
    const std::string want = t.expect;
    if (want == "caught_read") {
        pass = !a.read_ok && a.has_read && a.has_write && a.write_ok && a.guards_ok;
        if (a.read_ok) detail = "not_caught_by_read_account(gate_is_vacuous)";
        else if (!a.write_ok || !a.guards_ok) detail = "caught_by_more_than_the_read_account";
    } else if (want == "caught_write") {
        pass = !a.write_ok && a.has_write && a.read_ok && a.guards_ok;
        if (a.write_ok) detail = "not_caught_by_write_account(gate_is_vacuous)";
        else if (!a.read_ok || !a.guards_ok) detail = "caught_by_more_than_the_write_account";
    } else if (want == "caught_guards") {
        // A write past the arena: the byte accounts cannot see it (the arena itself is fine) and only
        // the guard band can, so this is the case that proves the band is wired up rather than
        // decorative.
        pass = !a.guards_ok && a.read_ok && a.write_ok;
        if (a.guards_ok) detail = "not_caught_by_the_guard_band(gate_is_vacuous)";
        else if (!a.read_ok || !a.write_ok) detail = "caught_by_more_than_the_guard_band";
    } else if (want == "caught_read_sole") {
        // A mode with no write stream at all (the two new modes): the read account must be the ONLY
        // account, it must fire, and the guards must stay intact — otherwise "caught" would just mean
        // "something somewhere disagreed".
        pass = !a.read_ok && a.has_read && !a.has_write && a.guards_ok;
        if (a.read_ok) detail = "not_caught_by_read_account(gate_is_vacuous)";
        else if (!a.guards_ok) detail = "caught_by_more_than_the_read_account";
    }
    char buf[320];
    snprintf(buf, sizeof buf,
             "SELFTEST device case=%s mode=%s expect=%s ratio_act=%.3f blocks=%d pages=%s "
             "rd=%.0f/%.0f wr=%.0f/%.0f wr_landed=%s guards=%s verdict=%s ok=%d%s%s\n",
             t.name, c.mode.c_str(), t.expect, p.ratio_act, p.blocks, c.pages.c_str(),
             a.rd_sum, a.rd_want, a.wr_sum, a.wr_want, a.write_note.c_str(),
             a.guards_ok ? "intact" : "written", verdict.c_str(), pass ? 1 : 0,
             pass ? "" : " why=", pass ? "" : detail.c_str());
    fputs(buf, stdout);
    fflush(stdout);
    free_dev(d);
    return pass ? 0 : 1;
}

// CPU half: the geometry invariants over the sweep's own configurations, plus the anti-vacuity of
// the bounds model itself.

// Are two plans the same MULTISET of launches (kind, region, bytes, pairing)? The two interleave
// orders must be, or a rate difference between them would be a byte-count difference in costume.
struct SchedKey {
    int kind, obj, obj2; size_t n4;
    bool operator<(const SchedKey& o) const {
        return std::tie(kind, obj, obj2, n4) < std::tie(o.kind, o.obj, o.obj2, o.n4);
    }
    bool operator==(const SchedKey& o) const {
        return kind == o.kind && obj == o.obj && obj2 == o.obj2 && n4 == o.n4;
    }
};
bool same_multiset(const Plan& x, const Plan& y) {
    auto keys = [](const Plan& q) {
        std::vector<SchedKey> v;
        for (const Plan::Step& s : q.sched) v.push_back({s.kind, s.obj, s.obj2, s.n4});
        std::sort(v.begin(), v.end());
        return v;
    };
    return x.chk_rd_n4 == y.chk_rd_n4 && keys(x) == keys(y);
}

int selftest_geom() {
    int fails = 0;
    for (const GeomCase& t : kGeomCases) {
        Cfg c;
        c.mode = t.mode; c.rrd = t.rrd; c.rwr = t.rwr; c.footprint_mib = t.mib;
        c.buf_mib = t.buf_mib; c.wr_arena_mib = t.arena_mib; c.nbuf = t.nbuf;
        c.overlap = t.overlap; c.tag = "selftest";
        if (t.pattern) c.pattern = t.pattern;
        if (t.engine_in) { c.engine_in = t.engine_in; c.engine_in_given = true; }
        if (t.stage >= 0) c.stage = t.stage;
        if (t.blocks) c.blocks = t.blocks;
        if (t.revisits) c.revisits = t.revisits;
        if (t.dummy_mib) c.dummy_mib = t.dummy_mib;
        if (t.dummy_nbuf) c.dummy_nbuf = t.dummy_nbuf;
        if (t.order) c.order = t.order;
        const Plan p = resolve(c);
        std::string why;
        const bool ok = invariants(c, p, &why);
        const Bounds b = plan_bounds(c, p);
        printf("SELFTEST geom mode=%s ratio=%ld:%ld mib=%.0f buf_mib=%.2f arena_mib=%.0f nbuf=%d "
               "blocks=%d rounds=%zu passes=%zu wrounds=%zu ratio_act=%.4f rd_mib=%.1f wr_mib=%.3f "
               "rd_end=%zu wr_end=%zu wr_buf_n4=%zu pattern=%s engine_in=%d stage=%d row_f4=%zu "
               "slots=%d rdlaunches=%d order=%s revisits=%d dummy_mib=%.2f dummy_nbuf=%d "
               "fit=%s ok=%d %s\n",
               t.mode, t.rrd, t.rwr, t.mib, t.buf_mib, t.arena_mib, t.nbuf, p.blocks, p.rounds,
               p.passes, p.wrounds, p.ratio_act, p.read_bytes / MI, p.write_bytes / MI,
               b.rd_end, b.wr_end, p.wr_buf_n4,
               c.pattern.c_str(), c.engine_in, c.stage, p.row_f4, p.sum_slots, p.rd_launches,
               c.order.c_str(), c.revisits, c.dummy_mib, c.dummy_nbuf,
               p.fit.empty() ? "-" : p.fit.c_str(), ok ? 1 : 0, ok ? "-" : why.c_str());
        if (!ok) ++fails;
    }
    // Anti-vacuity for the ORDER comparison: the two orders must launch the same multiset of reads,
    // and dropping one launch has to break that equality. Otherwise "round-robin is slower" could
    // silently mean "round-robin read more bytes".
    {
        Cfg base;
        base.mode = "interleave"; base.buf_mib = 32; base.nbuf = 8; base.revisits = 2;
        base.dummy_mib = 96; base.dummy_nbuf = 4; base.tag = "selftest";
        Cfg blk = base, rr = base;
        blk.order = "blocked"; rr.order = "roundrobin";
        const Plan pb = resolve(blk), pr = resolve(rr);
        const bool same = same_multiset(pb, pr);
        Plan broken = pr;
        for (size_t i = 0; i < broken.sched.size(); ++i) {
            if (broken.sched[i].kind == 0 && broken.sched[i].obj >= pb.rd_engine_regions) {
                broken.sched.erase(broken.sched.begin() + i);
                break;
            }
        }
        const bool dropped_caught = !same_multiset(pb, broken);
        std::string w;
        const bool both_ok = invariants(blk, pb, &w) && invariants(rr, pr, &w);
        printf("SELFTEST anti-vacuity order: same_multiset ok=%d same=%d dropped_dummy_caught=%d "
               "both_orders_geom_ok=%d rd_mib=%.1f\n",
               (same && dropped_caught && both_ok) ? 1 : 0, same ? 1 : 0, dropped_caught ? 1 : 0,
               both_ok ? 1 : 0, pr.read_bytes / MI);
        if (!(same && dropped_caught && both_ok)) ++fails;
    }
    // Anti-vacuity for the MAPPING comparison, three ways. (1) At the geometry the A-test uses, the
    // engine mapping must bill exactly the same bytes as the contiguous twin, so a rate difference
    // cannot be a byte difference — for both shapes, with the twin carrying --engine-in so the two sides
    // round their regions to the same whole rows. (2) The row widths must be the ones the engine
    // instantiates: 1024 bf16 = 128 float4s for gate/up, 3584 = 448 for down, each a whole number of
    // rows per launch. (3) The invariant that the pattern reached the launches must reject a plan whose
    // engine flags were cleared, because a `--pattern engine-mlp` that silently ran the plain reader is
    // this instrument's vacuous form and the byte account cannot see it.
    {
        Cfg base; base.mode = "read"; base.tag = "selftest";
        Cfg gu = base, gu_plain = base, dn = base, dn_plain = base;
        gu.nbuf = 48; gu.buf_mib = 7;   gu.pattern = "engine-mlp"; gu.engine_in = 1024;
        gu.engine_in_given = true; gu.blocks = 448;
        gu_plain.nbuf = 24; gu_plain.buf_mib = 14; gu_plain.engine_in = 1024;
        gu_plain.engine_in_given = true; gu_plain.blocks = 448;
        dn.nbuf = 24; dn.buf_mib = 7;   dn.pattern = "engine-mlp"; dn.engine_in = 3584;
        dn.engine_in_given = true; dn.blocks = 128;
        dn_plain.nbuf = 24; dn_plain.buf_mib = 7; dn_plain.engine_in = 3584;
        dn_plain.engine_in_given = true; dn_plain.blocks = 128;
        const Plan pgu = resolve(gu), pgu_plain = resolve(gu_plain);
        const Plan pdn = resolve(dn), pdn_plain = resolve(dn_plain);
        const bool gu_same = pgu.read_bytes == pgu_plain.read_bytes
                          && pgu.rd_launches == pgu_plain.rd_launches;
        const bool dn_same = pdn.read_bytes == pdn_plain.read_bytes
                          && pdn.rd_launches == pdn_plain.rd_launches;
        const bool named_widths = pgu.row_f4 == 128 && pdn.row_f4 == 448
                               && pgu.rd_n4 % pgu.row_f4 == 0 && pdn.rd_n4 % pdn.row_f4 == 0;
        std::string w;
        const bool geom_ok = invariants(gu, pgu, &w) && invariants(dn, pdn, &w)
                          && invariants(gu_plain, pgu_plain, &w) && invariants(dn_plain, pdn_plain, &w);
        // (3): clear the engine flags on a copy of a clean engine plan. Same bytes, so only the new
        // invariant can notice that the mapping was never applied.
        Plan not_applied = pdn;
        for (Plan::Step& st : not_applied.sched) st.engine = 0;
        const bool caught = !invariants(dn, not_applied, &w);
        printf("SELFTEST anti-vacuity mapping: same_bytes_as_contiguous ok=%d gu=%d dn=%d "
               "named_widths=%d geom_ok=%d pattern_reaches_launches=%d gu_mib=%.1f dn_mib=%.1f\n",
               (gu_same && dn_same && named_widths && geom_ok && caught) ? 1 : 0,
               gu_same ? 1 : 0, dn_same ? 1 : 0, named_widths ? 1 : 0, geom_ok ? 1 : 0,
               caught ? 1 : 0, pgu.read_bytes / MI, pdn.read_bytes / MI);
        if (!(gu_same && dn_same && named_widths && geom_ok && caught)) ++fails;
    }
    // Anti-vacuity for the bounds model: put the shipped bug back — the host setting `rounds` to
    // rd_n4/stride while the kernel reads rounds*rrd*stride float4s — and the model must reject it.
    {
        Cfg c;
        c.mode = "mix"; c.rrd = 62; c.rwr = 1; c.footprint_mib = 64; c.tag = "selftest";
        const Plan clean = resolve(c);
        Plan broken = resolve(c);
        broken.rounds = broken.passes;               // the Q8 collision, restored on purpose
        const bool clean_ok = plan_bounds(c, clean).ok;
        const Bounds inj = plan_bounds(c, broken);
        const bool caught = !inj.ok;
        printf("SELFTEST anti-vacuity bounds: rounds/grid_passes collision rejected ok=%d "
               "clean_ok=%d broken_ok=%d why=%s\n",
               (clean_ok && caught) ? 1 : 0, clean_ok ? 1 : 0, inj.ok ? 1 : 0, inj.why.c_str());
        if (!(clean_ok && caught)) ++fails;
    }
    return fails;
}

int selftest(int device, int only_case) {
    if (only_case >= 0) {
        if (only_case >= (int)(sizeof(kDevCases) / sizeof(kDevCases[0]))) {
            fprintf(stderr, "--selftest-case out of range\n");
            return 2;
        }
        return run_dev_case(kDevCases[only_case]);
    }
    int fails = selftest_geom();
    if (!device) {
        printf("SELFTEST geom-only (no device touched) fails=%d\n", fails);
        fflush(stdout);
        return fails ? 1 : 0;
    }
    // The parent deliberately never initialises HIP before forking.
    for (size_t i = 0; i < sizeof(kDevCases) / sizeof(kDevCases[0]); ++i) {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid < 0) { fprintf(stderr, "fork failed\n"); ++fails; continue; }
        if (pid == 0) _exit(run_dev_case(kDevCases[i]) ? 1 : 0);
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) { fprintf(stderr, "waitpid failed\n"); ++fails; continue; }
        if (WIFSIGNALED(status)) {
            printf("SELFTEST device case=%s ok=0 child_died_signal=%d (treat as a device fault)\n",
                   kDevCases[i].name, WTERMSIG(status));
            ++fails;
        } else if (WEXITSTATUS(status) != 0) {
            ++fails;
        }
    }
    fflush(stdout);
    printf("SELFTEST %s fails=%d\n", fails ? "FAIL" : "ok", fails);
    fflush(stdout);
    return fails ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    Cfg c;
    if (!parse(argc, argv, c)) { usage(); return 0; }
    if (c.selftest) return selftest(c.device_check != 0, c.only_case);
    return run(c);
}
