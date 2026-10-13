/*
 * tools/mixed_bw.cpp — is 237 GB/s even the right ceiling for a decode step?
 *
 * Every roofline in this repo is a READ number: `tools/bw_probe_hip.cpp` streams a 1 GiB buffer
 * with a grid-stride sum and the box answers 237 GB/s, and `bench/performance-status.md` then bills
 * ~1.5 GB of weights against it. But a decode step is not a read-only workload: it streams weights
 * WHILE writing activations, the ~115 bf16->f32 cast outputs, the KV append and the GDN state
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
 * Geometry is a flag: one big streamed buffer, or N separate buffers walked once per iteration —
 * the MLP's actual shape is 48 x 11.01 MB, which is not obviously the same machine as "one 528 MB
 * window", and Q7 showed the difference is worth measuring rather than assuming.
 *
 * Rules this file obeys, all of them load-bearing (AGENTS.md trap rows):
 *   * `steady_clock` bursts only. `hipEventElapsedTime` returns exactly 0.000 ms for sub-millisecond
 *     work on this iGPU — an event timer here reports inf GB/s. One sync at the end of the burst,
 *     warm-up pass outside the timed region.
 *   * cold-page numbers are their OWN rows, never averaged with warm ones: the first GPU pass over
 *     freshly allocated pages runs ~35 GB/s (`devices/strix-halo-unified-memory.md`), so mixing the
 *     two makes up a number. `--pages both` prints one row per state, cold first, and the cold row
 *     is necessarily a single iteration (after one pass the pages are warm by definition).
 *   * >240 GB/s means skipped work, not a fast kernel. So every read pass carries a checksum: the
 *     source buffer is filled with 1.0f and the kernel sums it in DOUBLE, giving an exact integer
 *     result that is compared against 4.0 x float4s-read. A short or skipped read shows up as
 *     `checksum=FAIL` and a non-zero exit. Rows above `--suspect-gbs` (default 240) carry
 *     `suspect=1`; a small footprint can legitimately beat the ceiling (the 18 MiB GDN state reads
 *     at ~918 GB/s warm), which is why that is a flag on the row and not a deleted number.
 *   * `hipMalloc` only, never managed memory (managed degrades SILENTLY near the pool limit).
 *
 * Build:  tools/build.sh            (-> tools/build/mixed_bw, -O3 + rpath)
 * Run:    it is a timing tool, so it belongs to the main agent under tools/gpu_lock.sh.
 *     tools/build/mixed_bw --mode mix --ratio 20 --footprint-mib 528 --pages both --iters 8
 * Self-check that needs no device and no timing:  --dry-run prints the geometry it would run.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// ---- kernels -------------------------------------------------------------------------
// Every stream is lane-coalesced. One "round" reads R float4 per thread at a stride of the whole
// grid and then writes W float4 per thread, contiguous across the warp. R:W is therefore exact by
// construction (no rounding, no partial lines) and a single-lane partial-line write — which would
// turn the write stream into read-modify-write traffic and make the answer an artifact — cannot
// happen. This is the whole design: measure the mix, do not fight the coalescer.
__global__ void read_k(const float4* __restrict__ src, size_t n4, double* __restrict__ sums,
                       int verify) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double acc = 0.0;
    for (size_t i = t; i < n4; i += s) {
        float4 v = src[i];
        acc += (double)(v.x + v.y + v.z + v.w);
    }
    if (verify) sums[blockIdx.x] = acc;      // exact integer-valued partial: order-independent
}

__global__ void write_k(float4* __restrict__ dst, size_t n4) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float4 one = make_float4(1.f, 1.f, 1.f, 1.f);
    for (size_t i = t; i < n4; i += s) dst[i] = one;
}

// rrd reads per rwr writes, both lane-coalesced, in ONE kernel so the two streams are genuinely
// in flight together (separate launches on one stream would serialize and measure nothing).
// `wrounds` is how many write passes the arena can hold; the write index cycles through it, so a
// small write arena is rewritten again and again while the read stream runs — which is the engine's
// real shape (a few MB of activations and one GDN-state arena, dirtied 424 times a token).
__global__ void mix_k(const float4* __restrict__ src, float4* __restrict__ dst, size_t rounds,
                      int rrd, int rwr, size_t wrounds, double* __restrict__ sums, int verify) {
    const size_t s = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float4 one = make_float4(1.f, 1.f, 1.f, 1.f);
    double acc = 0.0;
    for (size_t j = 0; j < rounds; ++j) {
        for (int r = 0; r < rrd; ++r) {
            float4 v = src[((size_t)j * rrd + r) * s + t];
            acc += (double)(v.x + v.y + v.z + v.w);
        }
        const size_t wr = (wrounds ? (j % wrounds) : j) * rwr;
        for (int w = 0; w < rwr; ++w) dst[(wr + w) * s + t] = one;
    }
    if (verify) sums[blockIdx.x] = acc;      // exact integer-valued sum: order-independent
}

// ---- configuration ---------------------------------------------------------------------
struct Cfg {
    std::string mode = "read";       // read | write | mix | mixlaunch | wthenr
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
};

// A resolved geometry: what gets allocated and launched, with exact byte counts. `sched` IS the
// launch order inside one iteration, so the host never re-derives the interleaving at launch time
// and the reported byte counts cannot disagree with what was actually launched.
struct Plan {
    struct Step {
        int kind = 0;        // 0 = read pass, 1 = write pass
        int obj = 0;         // read-buffer index (reads)
        size_t off = 0;      // float4 offset into the write arena (writes)
        size_t n4 = 0;       // float4s this launch touches
        int blocks = 1;      // grid for THIS launch: a 4 KB write must not launch a 640-block grid
    };
    int blocks = 0, threads = 256, nbuf = 1;
    size_t stride = 0;               // threads per full-size launch = blocks*threads
    size_t rd_n4 = 0;                // float4s per read buffer
    size_t rounds = 0;               // grid passes per buffer (mix: rounds of the ratio loop)
    size_t wr_n4 = 0;                // float4s written per iteration = counted write traffic
    size_t wr_buf_n4 = 0;            // arena size; wr_n4 may exceed it (`mix` cycles it)
    size_t wrounds = 0;              // mix: write passes the arena holds before it wraps
    size_t wr_chunk_n4 = 0;          // float4s per write launch
    size_t chk_rd_n4 = 0;            // checksum target: float4s in the last read pass
    int rd_launches = 0, wr_launches = 0;
    double read_bytes = 0, write_bytes = 0, ratio_act = 0;
    std::string fit;                 // what the solver gave up; "-" means nothing
    std::vector<Step> sched;         // launch order for one iteration (empty for `mix`: one kernel)
};

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
        else if (a == "--selftest") c.selftest = 1;
        else if (a == "--selftest-geom") { c.selftest = 1; c.device_check = 0; }
        else if (a == "--dry-run") c.dry = 1;
        else if (a == "--help" || a == "-h") return false;
        else { fprintf(stderr, "unknown flag %s\n", a.c_str()); exit(2); }
    }
    if (c.mode != "read" && c.mode != "write" && c.mode != "mix" && c.mode != "mixlaunch" &&
        c.mode != "wthenr") { fprintf(stderr, "--mode must be read|write|mix|mixlaunch|wthenr\n"); exit(2); }
    if (c.pages != "both" && c.pages != "warm" && c.pages != "cold") {
        fprintf(stderr, "--pages must be both|warm|cold\n"); exit(2);
    }
    if (c.rrd < 0 || c.rwr < 0 || (c.rrd == 0 && c.rwr == 0)) {
        fprintf(stderr, "--ratio needs a positive side\n"); exit(2);
    }
    if ((c.mode == "mix" || c.mode == "mixlaunch") && (c.rrd == 0 || c.rwr == 0)) {
        fprintf(stderr, "%s needs a two-sided --ratio\n", c.mode.c_str()); exit(2);
    }
    if (c.mode == "wthenr" && c.overlap) c.nbuf = 1;     // same lines => one region
    c.iters = std::max(1, c.iters);
    c.threads = std::max(32, c.threads);
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
//     424-launch decode step already has plenty of dispatch limitation to study elsewhere.
Plan resolve(const Cfg& c) {
    const int nbuf = (c.mode == "mix" || c.mode == "wthenr") ? 1 : std::max(1, c.nbuf);
    const double total = (c.buf_mib > 0.0 ? c.buf_mib * nbuf : c.footprint_mib) * MI;
    double rb, wb;
    if (c.mode == "read") { rb = total; wb = 0.0; }
    else if (c.mode == "write") { rb = 0.0; wb = total; }
    else if (c.mode == "wthenr" && c.overlap) { rb = wb = total; }   // read back the same lines
    else { rb = total * (double)c.rrd / (double)(c.rrd + c.rwr); wb = total - rb; }
    const double arena = c.wr_arena_mib > 0.0 ? c.wr_arena_mib * MI : wb;
    const size_t wgran = (size_t)c.threads;      // float4 per write-launch unit: 4 KB at 256
    const int want = c.blocks > 0 ? c.blocks : (nbuf > 1 ? 448 : 640);
    auto dn = [](size_t v, size_t unit) { return unit ? (v / unit) * unit : 0; };

    std::vector<Plan> cands;
    for (int b = std::min(want, 4096); b >= 8; --b) {
        Plan q;
        q.blocks = b; q.threads = c.threads; q.nbuf = nbuf;
        const size_t s = q.stride = (size_t)b * (size_t)c.threads;
        size_t rd_n4 = 0, wr_chunk = 0, wr_total = 0, wr_arena = 0;
        int rdl = 0, wrl = 0;
        int wafter = 0;                       // 1 = write launch after each read launch
        std::vector<size_t> wr_group;         // per read launch: float4s to write after it
        if (c.mode == "mix") {                // one kernel: exact ratio, one read region
            q.rounds = (s * (size_t)c.rrd * 16) ? (size_t)(rb / (s * (size_t)c.rrd * 16)) : 0;
            rd_n4 = q.rounds * (size_t)c.rrd * s;
            rdl = q.rd_launches = q.rounds ? 1 : 0;
            wr_chunk = (size_t)c.rwr * s;
            wr_total = q.rounds * wr_chunk;
            wr_arena = std::min(wr_total, std::max(wr_chunk, dn((size_t)(arena / 16), wr_chunk)));
        } else if (c.mode == "read") {
            rd_n4 = dn((size_t)(rb / nbuf) / 16, s);
            rdl = q.rd_launches = rd_n4 ? nbuf : 0;
        } else if (c.mode == "write") {
            wr_chunk = std::max(wgran, dn((size_t)(wb / nbuf) / 16, wgran));
            wrl = wr_chunk ? nbuf : 0;
            wr_total = wr_chunk * (size_t)wrl;
            wr_arena = wr_total;
        } else if (c.mode == "mixlaunch") {    // the engine's shape: serialized launches, one stream
            rd_n4 = dn((size_t)(rb / nbuf) / 16, s);
            rdl = q.rd_launches = rd_n4 ? nbuf : 0;
            // Write launches are distributed across the read launches instead of rounded per
            // buffer: the whole iteration hits the requested ratio to within one 4 KB unit, so the
            // ratio error no longer grows with the ratio (per-buffer rounding made a 62:1 mix
            // silently become 63:1).
            const double target = (double)rd_n4 * (double)std::max(1, rdl) * (double)c.rwr
                                / (double)std::max(1L, c.rrd);
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
            wr_chunk = c.overlap ? rd_n4 : std::max(wgran, dn((size_t)(wb) / 16, wgran));
            wrl = wr_chunk ? 1 : 0;
            wr_total = wr_chunk;
            wr_arena = wr_chunk;               // the same arena is dirtied again each pass
        }
        q.rd_n4 = rd_n4;
        q.rd_launches = rdl;
        q.wr_launches = wrl;
        q.rounds = std::max(q.rounds, rd_n4 / s);
        q.wr_chunk_n4 = wr_chunk;
        q.wr_n4 = wr_total;
        q.wr_buf_n4 = wr_arena;
        q.wrounds = wr_chunk ? wr_arena / wr_chunk : 0;
        q.chk_rd_n4 = rd_n4;
        size_t woff = 0;
        auto push_wr = [&](size_t n4) {
            if (!n4) return;
            Plan::Step st; st.kind = 1; st.off = woff; st.n4 = n4;
            st.blocks = (int)std::min<size_t>(8192, std::max<size_t>(1, (n4 + wgran - 1) / wgran));
            q.sched.push_back(st); woff += n4;
        };
        auto push_rd = [&](size_t i) {
            if (!rd_n4) return;
            Plan::Step st; st.kind = 0; st.obj = (int)i; st.n4 = rd_n4; st.blocks = b;
            q.sched.push_back(st);
        };
        if (c.mode == "wthenr") { push_wr(wr_chunk); push_rd(0); }
        else if (c.mode == "write") { for (int i = 0; i < wrl; ++i) push_wr(wr_chunk); }
        else {
            for (int i = 0; i < rdl; ++i) {
                push_rd(i);
                if (wafter) push_wr(wr_group[i]);
            }
        }
        q.read_bytes = (double)rd_n4 * (double)std::max(1, rdl) * 16.0;
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
    if (best.rd_n4 == 0 && c.mode != "write") { fprintf(stderr, "read region smaller than one grid pass\n"); exit(2); }
    if (best.wr_n4 == 0 && c.mode != "read") { fprintf(stderr, "write region smaller than one launch\n"); exit(2); }
    if (c.mode == "mix" && c.nbuf > 1) best.fit = "nbuf_forced_1(mix_is_one_region)";
    if (best.wr_buf_n4 > 0 && best.wr_n4 > best.wr_buf_n4 && best.fit.empty()) best.fit = "write_arena_cycled";
    if (best.blocks < 80 && best.fit.empty()) best.fit = "blocks_below_80(ratio_too_steep_for_footprint)";
    return best;
}

// ---- device side -----------------------------------------------------------------------
struct Dev {
    std::vector<float4*> rd;      // read buffers (nbuf of them; for wthenr+overlap, wr aliases rd[0])
    float4* wr = nullptr;         // write arena
    double* sums = nullptr;       // per-block read checksums
};

inline float4* first_rd(const Dev& d) { return d.rd.empty() ? nullptr : d.rd[0]; }

Dev alloc_dev(const Cfg& c, const Plan& p, bool fill) {
    Dev d;
    const bool alias = (c.mode == "wthenr" && c.overlap);
    for (int i = 0; i < p.rd_launches; ++i) {
        float4* q = nullptr;
        CK(hipMalloc(&q, p.rd_n4 * 16));                       // hipMalloc, never managed
        d.rd.push_back(q);
    }
    if (alias) { d.wr = first_rd(d); }                         // read back the just-dirtied lines
    else if (p.wr_buf_n4) { CK(hipMalloc(&d.wr, p.wr_buf_n4 * 16)); }
    CK(hipMalloc(&d.sums, sizeof(double) * p.blocks));
    if (fill) {   // write_k is the fill: it stores 1.0f, which is what the checksum expects
        for (auto q : d.rd) write_k<<<p.blocks, p.threads>>>(q, p.rd_n4);
        if (d.wr && d.wr != first_rd(d)) write_k<<<p.blocks, p.threads>>>(d.wr, p.wr_buf_n4);
        CK(hipDeviceSynchronize());                            // this pass is what warms the pages
    }
    return d;
}

void free_dev(Dev& d) {
    for (auto q : d.rd) if (q) CK(hipFree(q));
    if (d.wr && d.wr != first_rd(d)) CK(hipFree(d.wr));
    if (d.sums) CK(hipFree(d.sums));
    d = Dev();
}

// One iteration of the mode's launch schedule, exactly as resolve() laid it out. Kept apart from
// the timing so the cold / warm / write-then-read variants all share one launch shape.
//
// Note on what `mixlaunch` measures: launches on one stream are serialized, so its read and write
// passes do NOT overlap in flight — that is the engine's own shape (one stream, 424 launches per
// token), and it is a different question from `mix`, where both streams are in flight inside one
// kernel. Rows carry --mode for exactly this reason; do not compare across them as if they were one
// experiment.
void launch_once(const Cfg& c, const Plan& p, Dev& d, bool reads, bool writes) {
    if (c.mode == "mix") {
        if (reads && writes)
            mix_k<<<p.blocks, p.threads>>>(first_rd(d), d.wr, p.rounds, (int)c.rrd, (int)c.rwr,
                                           (int)p.wrounds, c.verify ? d.sums : nullptr, c.verify);
        return;
    }
    for (const Plan::Step& st : p.sched) {
        if (st.kind == 0) {
            if (!reads) continue;
            read_k<<<st.blocks, p.threads>>>(d.rd[st.obj], st.n4,
                                             c.verify ? d.sums : nullptr, c.verify);
        } else {
            if (!writes) continue;
            write_k<<<st.blocks, p.threads>>>(d.wr + st.off, st.n4);
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

// Exact checksum of the last read pass: the buffers hold 1.0f, so every partial sum is an exact
// integer in double and the order of accumulation cannot matter.
bool readback(const Dev& d, const Plan& p, double* sum_out) {
    std::vector<double> h(p.blocks, 0.0);
    CK(hipMemcpy(h.data(), d.sums, sizeof(double) * p.blocks, hipMemcpyDeviceToHost));
    double s = 0.0;
    for (double v : h) s += v;
    *sum_out = s;
    return true;
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
    printf("MIXEDBW tag=%s mode=%s ratio=%ld:%ld ratio_act=%.3f pages=%s iters=%d blocks=%d "
           "threads=%d launches=%d footprint_mib=%.3f read_mib=%.3f write_mib=%.3f wrbuf_mib=%.3f "
           "rounds=%zu wrounds=%zu ms=%.4f gbs=%.1f rd_gbs=%.1f wr_gbs=%.1f "
           "vs_read_probe_pct=%.1f vs_copy_probe_pct=%.1f suspect=%d %s\n",
           c.tag.c_str(), c.mode.c_str(), c.rrd, c.rwr, p.ratio_act, pages, iters, p.blocks,
           p.threads, p.rd_launches + p.wr_launches, tot / MI, rb / MI, wb / MI,
           p.wr_buf_n4 * 16.0 / MI, p.rounds, p.wrounds, ms, gbs, rd_gbs, wr_gbs,
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
bool invariants(const Cfg& c, const Plan& p, std::string* why) {
    auto bad = [&](const std::string& s) { *why = s; return false; };
    if (p.blocks < 8) return bad("blocks_collapsed_below_8");
    if (c.mode == "read" && (p.rd_n4 == 0 || p.wr_n4 != 0)) return bad("read_side_empty");
    if (c.mode == "write" && p.wr_n4 == 0) return bad("write_side_empty");
    if (c.mode != "read" && c.mode != "write" && p.wr_n4 == 0) return bad("write_stream_dropped");
    if (p.wr_n4 > 0 && p.wr_buf_n4 == 0) return bad("no_write_arena");
    if (c.mode == "mix") {
        if (p.rounds < 1) return bad("zero_rounds");
        const double want = (double)c.rrd / (double)c.rwr;
        if (std::fabs(p.ratio_act - want) > 1e-9 * want) return bad("ratio_not_exact");
    }
    if (c.mode == "mixlaunch" && p.ratio_act <= 0.0) return bad("no_traffic");
    if (p.fit == "write_stream_dropped") return bad("write_stream_dropped");
    if (p.read_bytes + p.write_bytes < 1.0 * MI) return bad("footprint_below_1MiB");
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
        "  --verify 0|1 --suspect-gbs G --strict      checksum + >240 GB/s flag (see file header)\n"
        "  --tag S | --dry-run (print the geometry, touch no device) | --selftest-geom (CPU) |\n"
        "  --selftest (CPU geometry + a small untimed device checksum pass)\n");
}

void print_plan(const Cfg& c, const Plan& p) {
    std::string why;
    const bool ok = invariants(c, p, &why);
    printf("PLAN mode=%s blocks=%d threads=%d stride=%zu nbuf=%d rd_n4=%zu rd_launches=%d "
           "rd_mib=%.3f wr_n4=%zu wr_chunk_n4=%zu wr_launches=%d wr_mib=%.3f wrbuf_mib=%.3f "
           "rounds=%zu wrounds=%zu ratio_req=%ld:%ld ratio_act=%.4f footprint_mib=%.3f fit=%s\n",
           c.mode.c_str(), p.blocks, p.threads, p.stride, p.nbuf, p.rd_n4, p.rd_launches,
           p.read_bytes / MI, p.wr_n4, p.wr_chunk_n4, p.wr_launches, p.write_bytes / MI,
           p.wr_buf_n4 * 16.0 / MI, p.rounds, p.wrounds, c.rrd, c.rwr, p.ratio_act,
           (p.read_bytes + p.write_bytes) / MI, p.fit.empty() ? "-" : p.fit.c_str());
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
        Dev d = alloc_dev(c, p, false);                  // pages allocated and never touched
        const int it = 1;                                // one pass; a second pass would be warm
        const double ms = burst(c, p, d, rd, wr, it, false);   // no warm-up: that IS the measurement
        std::string extra = "checksum=skipped_cold";
        if (rd && c.verify) {
            double sum = 0.0;
            readback(d, p, &sum);
            char buf[128];
            snprintf(buf, sizeof buf, "raw_sum=%.0f fresh_zero_sum=%d", sum, sum == 0.0 ? 1 : 0);
            extra = buf;
        }
        row(c, p, "cold", it, ms, p.read_bytes, p.write_bytes, extra);
        free_dev(d);
    }

    if (do_warm) {
        Dev d = alloc_dev(c, p, true);                   // the fill pass is what warms the pages
        if (c.mode == "wthenr") {
            const double ms_w = burst(c, p, d, false, true, c.iters, true);
            const double ms_r = burst(c, p, d, true, false, c.iters, true);
            const double ms_p = burst(c, p, d, true, true, c.iters, true);
            char buf[256];
            snprintf(buf, sizeof buf,
                     "ms_write_alone=%.4f ms_read_alone=%.4f ms_pair=%.4f pair_credit_ms=%.4f "
                     "read_dirty_penalty_pct=%.1f pair_sync=%d overlap=%d",
                     ms_w, ms_r, ms_p, ms_w + ms_r - ms_p,
                     ms_r > 0 ? 100.0 * (ms_p - ms_w) / ms_r : 0.0, c.pair_sync, c.overlap);
            row(c, p, "warm", c.iters, ms_p, p.read_bytes, p.write_bytes, buf);
        } else {
            const double ms = burst(c, p, d, rd, wr, c.iters, true);
            std::string extra = "checksum=skipped";
            if (c.verify) {
                double sum = 0.0;
                double want = 0.0;
                if (c.mode == "write") {          // prove the writes landed: untimed read-back pass
                    read_k<<<p.blocks, p.threads>>>(d.wr, p.wr_buf_n4, d.sums, 1);
                    CK(hipDeviceSynchronize());
                    readback(d, p, &sum);
                    want = 4.0 * (double)p.wr_buf_n4;
                    extra = (sum == want) ? "checksum=ok_writes_landed"
                                          : (g_bad++, "checksum=FAIL_writes_missing");
                } else {
                    readback(d, p, &sum);
                    want = 4.0 * (double)p.chk_rd_n4;   // the last read pass covered one buffer
                    extra = (sum == want) ? "checksum=ok" : (g_bad++, "checksum=FAIL_reads_short");
                }
            }
            row(c, p, "warm", c.iters, ms, p.read_bytes, p.write_bytes, extra);
        }
        free_dev(d);
    }
    if (g_bad) return 3;
    if (g_suspect && c.strict) return 4;
    return 0;
}

// ---- selftest --------------------------------------------------------------------------
// `--selftest-geom` is the pure-CPU half: the geometry invariants over the sweep's own config
// table. It touches no device, so it can be run anywhere, and it is what catches a solver that
// quietly rounds a ratio or drops the write stream. `--selftest` adds a small untimed device pass
// that proves the checksums work — correctness only, it produces NO bandwidth number, which is
// what makes it a gate rather than a measurement.
int selftest(bool device) {
    int fails = 0;
    struct T { const char* mode; long rrd, rwr; double mib; int nbuf; } ts[] = {
        {"read", 20, 1, 128, 1}, {"read", 20, 1, 528, 48}, {"write", 20, 1, 128, 1},
        {"mix", 20, 1, 512, 1}, {"mix", 10, 1, 512, 1}, {"mix", 4, 1, 512, 1},
        {"mix", 1, 1, 512, 1}, {"mix", 1, 2, 128, 1}, {"mix", 62, 1, 528, 1},
        {"mix", 128, 1, 528, 1}, {"mix", 20, 1, 528, 48}, {"mixlaunch", 62, 1, 528, 48},
        {"mixlaunch", 1, 1, 256, 48}, {"mixlaunch", 20, 1, 528, 48}, {"wthenr", 20, 1, 256, 1},
    };
    for (const T& t : ts) {
        Cfg c; c.mode = t.mode; c.rrd = t.rrd; c.rwr = t.rwr; c.footprint_mib = t.mib;
        c.nbuf = t.nbuf; c.tag = "selftest";
        const Plan p = resolve(c);
        std::string why;
        const bool ok = invariants(c, p, &why);
        printf("SELFTEST geom mode=%s ratio=%ld:%ld mib=%.0f nbuf=%d blocks=%d rounds=%zu "
               "ratio_act=%.4f rd_mib=%.1f wr_mib=%.3f fit=%s ok=%d %s\n",
               t.mode, t.rrd, t.rwr, t.mib, t.nbuf, p.blocks, p.rounds, p.ratio_act,
               p.read_bytes / MI, p.write_bytes / MI, p.fit.empty() ? "-" : p.fit.c_str(),
               ok ? 1 : 0, ok ? "-" : why.c_str());
        if (!ok) ++fails;
    }
    if (!device) {
        printf("SELFTEST geom-only (no device touched) fails=%d\n", fails);
        fflush(stdout);
        return fails ? 1 : 0;
    }
    CK(hipInit(0));
    struct D { const char* mode; long rrd, rwr; double mib; int nbuf; } ds[] = {
        {"read", 20, 1, 64, 1}, {"mix", 20, 1, 64, 1}, {"mix", 1, 1, 64, 1}, {"mix", 62, 1, 64, 1},
        {"write", 20, 1, 64, 1}, {"wthenr", 20, 1, 64, 1}, {"mixlaunch", 62, 1, 264, 24},
    };
    for (const D& t : ds) {
        Cfg c; c.mode = t.mode; c.rrd = t.rrd; c.rwr = t.rwr; c.footprint_mib = t.mib;
        c.nbuf = t.nbuf; c.pages = "warm"; c.verify = 1; c.tag = "selftest";
        const Plan p = resolve(c);
        Dev d = alloc_dev(c, p, true);
        launch_once(c, p, d, c.mode != "write", c.mode != "read");
        CK(hipDeviceSynchronize());
        double sum = 0.0;
        double want = 4.0 * (double)p.chk_rd_n4;
        if (c.mode == "write") {
            read_k<<<p.blocks, p.threads>>>(d.wr, p.wr_buf_n4, d.sums, 1);
            CK(hipDeviceSynchronize());
            want = 4.0 * (double)p.wr_buf_n4;
        }
        readback(d, p, &sum);
        const bool ok = (sum == want);
        printf("SELFTEST device mode=%s ratio_act=%.3f blocks=%d sum=%.0f want=%.0f ok=%d\n",
               t.mode, p.ratio_act, p.blocks, sum, want, ok ? 1 : 0);
        if (!ok) ++fails;
        free_dev(d);
    }
    printf("SELFTEST %s fails=%d\n", fails ? "FAIL" : "ok", fails);
    fflush(stdout);
    return fails ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    Cfg c;
    if (!parse(argc, argv, c)) { usage(); return 0; }
    if (c.selftest) return selftest(c.device_check != 0);
    return run(c);
}
