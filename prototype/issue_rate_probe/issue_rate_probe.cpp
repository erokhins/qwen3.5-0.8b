/*
 * issue_rate_probe.cpp — two numbers this box has never given up, one machine-readable line per run.
 *
 *   A. the sustained VALU warp-instruction rate per CU per clock, for the dot shape the shipped
 *      `attn_tile_wave` scores with. progress/2026-10-16/prefill-occupancy-results-2026-10-16.md §3
 *      puts attention at 95 % of a floor computed at 1 warp-inst/CU/clk; `rocminfo` says 2 SIMDs per
 *      CU, so that floor may be off by 2x and three funded routes are ranked against it.
 *   B. achievable read GB/s for a KV-shaped stream as a function of resident working-set SIZE. The
 *      byte model bills attention at the 237 GB/s read probe while the per-layer KV set at 1024 lanes
 *      is only 2 MiB, and a resident ~54 MiB set (the GDN state) is already known at ~876 GB/s.
 *
 * WHAT THIS BINARY IS NOT: a number. It prints ms and GB/s, so it is a timing tool and timing tools
 * belong to the main agent in an exclusive window (AGENTS.md, "GPU execution is serialized"). The
 * modes a worker may run are --dry-run, --selftest-geom and --selftest; those take no timing and print
 * no ms, and they are the only output admissible as correctness evidence from a child session.
 *
 * THE BILL. A rate per CU per clock is meaningless without the instruction count, so
 *
 *     warp_inst_total = warp_inst_per_iter x iters x warps x launches
 *
 * where `warp_inst_per_iter` is GENERATED from the disassembly into `irp_isa.gen.h` by
 * `isa_census.py` and never typed by hand, and `warps` comes from the same grid the launch is built
 * from. Rates are printed under BOTH device readings, because this box disagrees with itself about
 * its CU count — hipGetDeviceProperties says 20, rocminfo says 40 with 2 SIMDs each
 * (devices/strix-halo-memory-bandwidth.md) — so `icpcb` is printed for each and the gap between them
 * is part of the answer rather than something to average away.
 *
 * THE ACCOUNTS, shaped by tools/mixed_bw.cpp's first attempt (every warm row failed on an account that
 * was itself broken). Every value a kernel consumes is an integer-valued BF16 chosen by a host-side
 * formula, so the expected checksum is DERIVED ON THE HOST by mirroring the kernel's index arithmetic
 * in exact integer arithmetic, and compared against a real block reduction. Two accounts:
 *   * `sums` — one block-reduced double per (launch, block). Bills how many bytes were touched and
 *     which value residue they carried. It CANNOT see a permutation of the same bytes;
 *   * `each` — per-thread partials, selftest geometry only. This is what catches a slot alias, which
 *     is the §19/§20 bug class (two self-consistent layouts joined by a branch, invisible to every
 *     per-form check).
 * The selftest asserts both directions: the owner must catch its bug and the non-owner must stay
 * silent (`lds_swap` must NOT move the block sum; an alias inside the buffer must NOT touch a guard).
 * A control that fires for everything is not a control, and a must-catch set that was not derived from
 * the geometry rots into "everything".
 *
 * EXIT CODES: 0 every printed row verified; 2 bad arguments or an impossible geometry; 3 a checksum
 * failed or a row printed without its account; 4 the selftest found any failure.
 */
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "irp_isa.gen.h"
#include "irp_kernels.h"

namespace {
using namespace irp;   // the kernels live in `namespace irp`

using irp::Args;
using irp::kAttnTileLdsBytes;
using irp::kAttnTileThreads;
using irp::kF32ExactInt;
using irp::kGdnResidentGbs;
using irp::kGuardFill;
using irp::kHeadDim;
using irp::kNumKvHeads;
using irp::kReadProbeGbs;
using irp::kRowBytes;

constexpr double MiB = 1048576.0;

struct Cfg {
    std::string kernel = "irp_fma_8";
    unsigned long long iters = 100000;
    int blocks = 320;
    int threads = 128;
    int reuse = 1;                          // lds: read passes over one staged tile (4 = attn's G)
    double footprint_mib = 2.0;             // kv: the independent variable
    unsigned long long passes = 0;          // kv: re-walks inside one launch; 0 = solve
    double min_launch_mib = 128.0;          // ... so one launch is at least this many MiB
    int launches = 20;                      // timed launches per rep (amortises the ~1.8 us dispatch)
    int reps = 5;                           // best-of-N over reps; the tool itself never reports a mean
    int verify = 1;
    std::string inject = "none";
    int cu_count = 40;                      // rocminfo's reading; --cu-count 20 prints HIP's
    double clock_mhz = 2894.0;              // sclk, prefill-occupancy §3
    std::string tag = "-";
    bool dry_run = false;
    bool selftest = false;
    bool selftest_geom = false;
    bool guards = false;                    // selftest only: poison bands around every region
    bool want_each = false;                 // selftest only: per-thread account
    int child_case = -1;
};

struct Kernel {
    const char* name;
    const char* family;     // valu | serial | cvt | lds | kv
    int chains;
    const void* fn;
    unsigned long long wi;  // warp-inst per iteration, read out of the disassembly
    const char* note;
};

#define IRP_K(NAME, FAMILY, CHAINS, NOTE) \
    Kernel { #NAME, FAMILY, CHAINS, (const void*)(NAME), IRP_WI_##NAME, NOTE }
const Kernel kKernels[] = {
    IRP_K(irp_fma_1, "valu", 1, "one FMA per thread per iteration"),
    IRP_K(irp_fma_4, "valu", 4, ""),
    IRP_K(irp_fma_8, "valu", 8, ""),
    IRP_K(irp_fma_16, "valu", 16, ""),
    IRP_K(irp_fma_32, "valu", 32, ""),
    IRP_K(irp_serial_8, "serial", 8, "8 FMAs in ONE dependency chain: latency, not issue"),
    IRP_K(irp_serial_32, "serial", 32, ""),
    IRP_K(irp_cvt_8, "cvt", 8, "a bf16->f32 conversion feeding every FMA (attn score shape)"),
    IRP_K(irp_cvt_16, "cvt", 16, ""),
    IRP_K(irp_lds_tile, "lds", 0, "write+read 12,288 B of LDS per iteration (attn LDS block size)"),
    IRP_K(irp_kv_read, "kv", 0, "one warp per 512 B KV row, both KV heads, dot at the use site"),
};
const Kernel* find_kernel(const std::string& n) {
    for (const Kernel& k : kKernels)
        if (n == k.name) return &k;
    return nullptr;
}

int inject_code(const std::string& s) {
    if (s == "none") return irp::kInjNone;
    if (s == "drop_chain") return irp::kInjDropChain;
    if (s == "kv_short") return irp::kInjKvShort;
    if (s == "kv_alias") return irp::kInjKvAlias;
    if (s == "kv_stray") return irp::kInjKvStray;
    if (s == "lds_slot_off") return irp::kInjLdsSlotOff;
    if (s == "lds_short") return irp::kInjLdsShort;
    if (s == "lds_swap") return irp::kInjLdsSwap;
    return -1;
}

// ============================================================================================
// The host mirror of each kernel's index arithmetic. THIS IS THE REFERENCE SIDE, so it must not be
// derived from the thing under test (the trap in prototype/attn_tile_model.py's `compare`, where
// reading the partial layout once and applying it to both sides compared two aliased waves and
// reported the fixed geometry as the bug). Every function here recomputes, from the plan alone, what
// a given THREAD will be holding at the end of its loop, in plain integer arithmetic.
// ============================================================================================

inline double valu_tag(double e) { return 1.0 + (double)((long long)e % 7LL); }

// valu/serial/cvt: every thread runs the same recurrence with the same operands, so the expected
// per-thread sum is one number. Seeds are distinct (s_j = j+1) so a dropped chain's deficit
// (s_j + iters) names the chain; the addend is 1.0 for every chain so the bound below is linear.
double mirror_per_thread(const Cfg& c, const Kernel& k, unsigned tid) {
    const double it = (double)c.iters;
    const int n = k.chains;
    const double lanev = (double)tid;        // the per-lane seed: see the header note on VGPRs
    if (std::strcmp(k.family, "serial") == 0) return 1.0 + lanev + it * (double)n;
    double s = (double)n * (double)(n + 1) / 2.0 + (double)n * lanev;   // the seeds
    if (std::strcmp(k.family, "cvt") == 0) {
        // The bf16 operand is 1.0 on even iterations and 2.0 on odd ones: it is flipped AFTER the
        // FMA that consumes it, which is what makes the counts ceil/floor rather than both iters/2.
        // Its STARTING value alternates with thread parity, so the two parities get mirror-image sums.
        const double ones = (double)((c.iters + 1) / 2), twos = (double)(c.iters / 2);
        s += (double)n * ((tid & 1u) ? (twos * 1.0 + ones * 2.0) : (ones * 1.0 + twos * 2.0));
    } else {
        s += (double)n * it;                                  // addend 1.0, one fma per chain
    }
    return s;
}

// lds: thread (w, lane) stages and re-reads rows w, w+nw, ... and the tile's value at element e is
// 1 + (e % 7). Per-thread sums therefore differ by lane, which is exactly why the per-thread account
// can see a slot alias and the block-sum account cannot.
double mirror_lds_thread(int lane, int w, int nw, const Cfg& c) {
    double per_iter = 0.0;
    for (int k = 0; k < irp::kLdsWordsPerThread; ++k) {
        const int row = w + k * nw;
        const int base = row * kHeadDim + lane * 8;
        for (int t = 0; t < 8; ++t) per_iter += valu_tag(base + t);
    }
    return per_iter * (double)c.reuse * (double)c.iters;
}

// kv: global warp gw reads rows gw, gw+nwarps, ... of every KV head, once per pass. Each of its 32
// lanes takes 8 elements of the row it is on.
double mirror_kv_warp(unsigned long long gw, unsigned long long nwarps, unsigned long long rows,
                      unsigned long long passes) {
    double s = 0.0;
    for (unsigned long long r = gw; r < rows; r += nwarps) s += 8.0 * valu_tag((double)r);
    return s * (double)passes;
}

// ---- device memory -------------------------------------------------------------------------
struct Buf {
    void* base = nullptr;
    size_t bytes = 0;
};

Buf alloc_dev(size_t bytes) {
    Buf b;
    IRP_CK(hipMalloc(&b.base, bytes));       // hipMalloc, never managed (managed degrades silently)
    b.bytes = bytes;
    return b;
}

template <typename T>
void fill_dev(T* p, size_t n, float v) {
    irp::fill_const_k<<<128, 256>>>(p, n, v);
    IRP_CK(hipDeviceSynchronize());
}

// Guard bands: poison OUTSIDE the region under test, so a kernel that wanders is caught by the account
// rather than by luck. Poison is 512.0, an exact BF16 integer outside the tag domain 1..7, so a stray
// READ moves the checksum instead of coincidentally matching it (tools/mixed_bw.cpp's kGuardVal, same
// reasoning, adapted to an integer-tag account instead of a 1.0f fill).
constexpr size_t kGuardElems = 32768;        // 64 KiB of BF16 poison per side

struct KvBuf {
    hip_bfloat16* base = nullptr;            // what hipMalloc returned (bands included)
    hip_bfloat16* p = nullptr;               // first usable element
    size_t elems = 0;
    size_t guard = 0;
};

KvBuf alloc_kv(size_t elems, bool guards, bool fill_rows, unsigned long long rows) {
    KvBuf r;
    r.guard = guards ? kGuardElems : 0;
    r.elems = elems;
    IRP_CK(hipMalloc((void**)&r.base, (r.guard + elems + r.guard) * sizeof(hip_bfloat16)));
    r.p = r.base + r.guard;
    auto band = [&](hip_bfloat16* at, size_t n, float v) {
        if (n) fill_dev(at, n, v);
    };
    band(r.base, r.guard, kGuardFill);
    band(r.p + elems, r.guard, kGuardFill);
    if (fill_rows) {
        // One kernel pass over the whole set: this is also what WARMS the pages, so no timed row can
        // be the first touch (first GPU pass over fresh pages runs ~35 GB/s,
        // devices/strix-halo-unified-memory.md).
        irp::fill_rows_k<<<512, 256>>>(r.p, elems, 7);
        IRP_CK(hipDeviceSynchronize());
    }
    return r;
}

bool guards_intact(const KvBuf& r) {
    if (!r.guard) return true;
    std::vector<hip_bfloat16> h(r.guard);
    const hip_bfloat16* bands[2] = {r.base, r.p + r.elems};
    for (const hip_bfloat16* b : bands) {
        IRP_CK(hipMemcpy(h.data(), b, r.guard * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < r.guard; ++i)
            if (float(h[i]) != kGuardFill) return false;
    }
    return true;
}

// ============================================================================================
// The plan: one struct, built once, feeding BOTH the launch and the account. This is the discipline
// tools/mixed_bw.cpp learned the hard way, where the solver's `rounds` meant one thing to the host and
// another to the kernel, and `--selftest-geom` cheerfully approved every configuration that then
// faulted. Nothing here is allowed to be re-derived a second time in a second place.
// ============================================================================================
struct Plan {
    const Kernel* k = nullptr;
    int threads = 128, blocks = 320, warps_per_block = 4;
    unsigned long long warps = 0, threads_total = 0;
    unsigned long long iters = 0;
    int reuse = 1;                       // lds
    unsigned long long rows = 0;         // kv: total rows of 512 B across BOTH KV heads
    unsigned long long passes = 1;       // kv
    double footprint_mib = 0.0;
    double launch_bytes_mib = 0.0;       // what ONE launch reads, the number `gbs` bills
    double warp_inst_total = 0.0;        // the bill
    double lds_mib_total = 0.0;          // LDS bytes one launch touches (staging only, not DRAM)
    std::vector<double> exp_block;       // per-block expected block-reduced sum
    double expected_total = 0.0;
    double max_thread_sum = 0.0;         // for the F32-exactness bound
    bool ok = false;
    std::string why;
};

int warps_per_block_of(const Cfg& c) { return c.threads / irp::kWarpWidth; }

bool plan_build(const Cfg& c, Plan& p) {
    p.k = find_kernel(c.kernel);
    if (!p.k) { p.why = "unknown kernel " + c.kernel; return false; }
    if ((c.threads & (c.threads - 1)) != 0 || c.threads < irp::kWarpWidth) {
        p.why = "--threads must be a power of two >= 32: publish_block_sum's halving tree drops a "
                "tail otherwise, and blockDim.x/warpSize is the warp count the grid is sized from";
        return false;
    }
    if (c.iters == 0) { p.why = "--iters must be >= 1"; return false; }
    p.threads = c.threads;
    p.blocks = c.blocks;
    p.warps_per_block = warps_per_block_of(c);
    p.warps = (unsigned long long)p.blocks * p.warps_per_block;
    p.threads_total = (unsigned long long)p.blocks * p.threads;
    p.iters = c.iters;
    p.reuse = c.reuse;
    const std::string fam = p.k->family;

    if (fam == "lds") {
        // The staged tile is sized for ONE geometry: 6 uint4 words per thread at 128 threads. A
        // different thread count would silently stage a different amount than attn does, so refuse
        // rather than approximate (the trap in AGENTS.md's rank-3 shape row: state the shape, do not
        // hope a generic formula reproduces it).
        if (c.threads != kAttnTileThreads) {
            p.why = "the lds kernel is attn's geometry: --threads must be 128 (ATTN_TILE_THREADS)";
            return false;
        }
        if (c.reuse < 1) { p.why = "--reuse must be >= 1"; return false; }
        double per_thread = 0.0;
        p.exp_block.assign(p.blocks, 0.0);
        const int nw = p.warps_per_block;
        for (int t = 0; t < p.threads; ++t) {
            const double v = mirror_lds_thread(t % irp::kWarpWidth, t / irp::kWarpWidth, nw, c);
            if (v > p.max_thread_sum) p.max_thread_sum = v;
            for (int b = 0; b < p.blocks; ++b) p.exp_block[b] += v;   // every block stages the same tile
        }
        for (double v : p.exp_block) p.expected_total += v;
        // Staging traffic per iteration: 12,288 B written + 12,288 B x reuse read, per BLOCK.
        p.lds_mib_total = (double)p.blocks * (double)p.iters *
                          ((double)kAttnTileLdsBytes * (1.0 + (double)c.reuse)) / MiB;
    } else if (fam == "kv") {
        const double bytes = c.footprint_mib * MiB;
        const unsigned long long rows_all =
            (unsigned long long)(bytes / (double)kRowBytes);   // 512 B rows
        if (rows_all == 0 || rows_all % (unsigned long long)kNumKvHeads) {
            p.why = "footprint must be a whole number of 512 B rows and split evenly between the "
                    "2 KV heads (use 2/8/32/128/512 MiB)";
            return false;
        }
        p.rows = rows_all;
        p.footprint_mib = (double)rows_all * kRowBytes / MiB;   // what it actually rounds to
        p.passes = c.passes ? c.passes
               : (unsigned long long)(c.min_launch_mib / p.footprint_mib + 0.999999);
        if (p.passes == 0) p.passes = 1;
        if (p.warps > p.rows) {
            p.why = "grid has more warps (" + std::to_string(p.warps) +
                    ") than rows (" + std::to_string(p.rows) +
                    "): idle warps would make the bill and the bytes disagree";
            return false;
        }
        // The billed iteration for the kv family is ONE PASS over the rows: that is the loop the
        // kernel counts, so `iters` is the pass count here and not the --iters the A family uses.
        p.iters = p.passes;
        p.launch_bytes_mib = (double)p.rows * kRowBytes * (double)p.passes / MiB;
        p.exp_block.assign(p.blocks, 0.0);
        for (unsigned long long gw = 0; gw < p.warps; ++gw) {
            const double v = mirror_kv_warp(gw, p.warps, p.rows, p.passes);
            const double per_thread = v;         // each lane of the warp reads the same rows
            if (per_thread > p.max_thread_sum) p.max_thread_sum = per_thread;
            p.exp_block[gw / p.warps_per_block] +=
                per_thread * (double)irp::kWarpWidth;   // 32 lanes, equal shares
        }
        for (double v : p.exp_block) p.expected_total += v;
    } else {
        // The A family's per-thread values differ by thread index (the per-lane seed that keeps the
        // accumulators in VGPRs), so the block account is the closed form summed over threads rather
        // than one number times the thread count.
        double blk = 0.0, maxt = 0.0;
        for (unsigned t = 0; t < (unsigned)c.threads; ++t) {
            const double v = mirror_per_thread(c, *p.k, t);
            blk += v;
            if (v > maxt) maxt = v;
        }
        p.max_thread_sum = maxt;
        p.exp_block.assign(p.blocks, blk);
        for (double x : p.exp_block) p.expected_total += x;
    }

    // THE BILL. Everything else on the rate row is decoration if this is wrong.
    p.warp_inst_total = (double)p.k->wi * (double)p.iters * (double)p.warps;

    if (p.max_thread_sum >= kF32ExactInt) {
        p.why = "the witness is an F32 integer sum and a per-thread running total of " +
                std::to_string((long long)p.max_thread_sum) +
                " would cross 2^24, where the equality stops being exact. Lower --iters (or --passes).";
        return false;
    }
    p.ok = true;
    return true;
}

// ============================================================================================
// One run: allocate, warm, time a burst, read the accounts back, print ONE line.
//
// The account is read back AFTER the burst rather than inside it, but it is the SAME launches that
// were timed: every launch writes its own slot of `sums`, so a launch that never ran, ran short, or
// ran twice is visible, and the verification is not a separate friendlier pass that could pass while
// the timed pass failed.
// ============================================================================================
struct RunResult {
    double best_ms = 0.0;
    int bad_blocks = 0, bad_launches = 0, parsed_launches = 0;
    int each_bad = 0;
    bool each_ran = false;
    bool guards_ok = true;
    bool checksum_ok = true;
    std::string why = "-";
};

RunResult run_one(const Cfg& c, const Plan& p, double* icpcb_out, double* gbs_out,
                  double* lds_gbs_out) {
    RunResult R;
    const std::string fam = p.k->family;
    const size_t operand_n = 128;
    Buf ops = alloc_dev(operand_n * sizeof(float));
    {
        std::vector<float> h(operand_n, 0.0f);
        h[0] = 1.0f;                       // FMA multiplier: runtime 1.0, invisible to the compiler
        h[1] = 1.0f;                       // FMA addend
        for (int t = 0; t < 8; ++t) h[8 + t] = 1.0f;      // the kv dot's query vector
        for (int j = 0; j < 32; ++j) h[64 + j] = (float)(j + 1);   // chain seeds, all distinct
        IRP_CK(hipMemcpy(ops.base, h.data(), operand_n * sizeof(float), hipMemcpyHostToDevice));
    }
    Buf sink = alloc_dev(sizeof(float));
    const size_t sum_slots = (size_t)c.launches * p.blocks;
    Buf sums = alloc_dev(sum_slots * sizeof(double));
    IRP_CK(hipMemset(sums.base, 0, sum_slots * sizeof(double)));
    Buf each;
    if (c.want_each) {
        each = alloc_dev(p.threads_total * sizeof(float));
        IRP_CK(hipMemset(each.base, 0, p.threads_total * sizeof(float)));
    }
    KvBuf kv;
    if (fam == "kv") {
        kv = alloc_kv(p.rows * kHeadDim, c.guards, true, p.rows);
    }

    Args a{};
    a.p = (const float*)ops.base;
    a.sink = (float*)sink.base;
    a.sums = (double*)sums.base;
    a.each = c.want_each ? (float*)each.base : nullptr;
    a.iters = c.iters;
    a.verify = c.verify;
    a.inject = inject_code(c.inject);
    if (a.inject < 0) { R.why = "unknown --inject"; R.checksum_ok = false; return R; }

    const int dyn_shares = (fam == "lds") ? kAttnTileLdsBytes : p.threads * (int)sizeof(double);
    auto launch = [&](int slot) {
        a.launch = (unsigned long long)slot;
        if (fam == "kv") {
            irp::irp_kv_read<<<p.blocks, p.threads, dyn_shares>>>(
                a, kv.p, p.rows, (unsigned int)p.passes);
        } else if (fam == "lds") {
            irp::irp_lds_tile<<<p.blocks, p.threads, dyn_shares>>>(a, c.reuse);
        } else {
            void (*fn)(Args) = (void (*)(Args))p.k->fn;
            fn<<<p.blocks, p.threads, dyn_shares>>>(a);
        }
    };

    // Warm: one untimed launch. For the KV rows this is also the page-warming pass (first GPU pass
    // over fresh pages runs ~35 GB/s), which is why it is not optional.
    launch(0);
    IRP_CK(hipDeviceSynchronize());
    if (hipGetLastError() != hipSuccess) { R.why = "kernel error on warm launch"; R.checksum_ok = false; }

    double best = 1e30;
    for (int r = 0; r < c.reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int L = 0; L < c.launches; ++L) launch(L);
        IRP_CK(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / c.launches;
        if (ms < best) best = ms;
    }
    R.best_ms = best;

    std::vector<double> got((size_t)c.launches * p.blocks);
    IRP_CK(hipMemcpy(got.data(), sums.base, got.size() * sizeof(double), hipMemcpyDeviceToHost));
    for (int L = 0; L < c.launches; ++L) {
        ++R.parsed_launches;                       // counted unconditionally, then required below
        int bad = 0;
        for (int b = 0; b < p.blocks; ++b)
            if (got[(size_t)L * p.blocks + b] != p.exp_block[b]) ++bad;
        if (bad) { R.bad_blocks += bad; ++R.bad_launches; }
    }
    if (R.parsed_launches != c.launches || R.bad_blocks) {
        R.checksum_ok = false;
        R.why = R.bad_blocks ? "read_account_short" : "no_launch_slots_parsed";
    }

    if (c.want_each) {
        R.each_ran = true;
        std::vector<float> ge(p.threads_total);
        IRP_CK(hipMemcpy(ge.data(), each.base, p.threads_total * sizeof(float), hipMemcpyDeviceToHost));
        const std::string fam2 = p.k->family;
        for (unsigned long long t = 0; t < p.threads_total; ++t) {
            double want = 0.0;
            if (fam2 == "lds") {
                const int tid = (int)(t % (unsigned long long)p.threads);   // every block stages the same tile
                want = mirror_lds_thread(tid % irp::kWarpWidth, tid / irp::kWarpWidth,
                                         p.warps_per_block, c);
            } else if (fam2 == "kv") {
                want = mirror_kv_warp(t / (unsigned long long)irp::kWarpWidth, p.warps, p.rows, p.passes);
            } else {
                want = mirror_per_thread(c, *p.k, (unsigned)(t % (unsigned long long)p.threads));
            }
            if ((double)ge[t] != want) ++R.each_bad;
        }
    }
    if (fam == "kv" && c.guards) R.guards_ok = guards_intact(kv);

    if (icpcb_out) {
        const double secs = best / 1000.0;
        *icpcb_out = secs > 0 ? p.warp_inst_total / (secs * (double)c.cu_count * c.clock_mhz * 1e6)
                              : 0.0;
    }
    if (gbs_out && fam == "kv") {
        *gbs_out = best > 0 ? p.launch_bytes_mib * MiB / (best / 1000.0) / 1e9 : 0.0;
    }
    if (lds_gbs_out && fam == "lds") {
        *lds_gbs_out = best > 0 ? p.lds_mib_total * MiB / (best / 1000.0) / 1e9 : 0.0;
    }
    (void)hipFree(ops.base); (void)hipFree(sink.base); (void)hipFree(sums.base);
    if (c.want_each && each.base) (void)hipFree(each.base);
    if (fam == "kv" && kv.base) (void)hipFree(kv.base);
    return R;
}

// ============================================================================================
// Derived blind sets. A must-catch list that was not DERIVED from the geometry rots into "everything"
// the moment the fixture changes (AGENTS.md, and prototype/attn_tile_model.py's neg_expect). Both
// functions below answer "can this witness see this bug at THIS geometry" from the same arithmetic the
// kernel and the mirror use, and the selftest prints the answer next to the verdict.
// ============================================================================================

// The two row sets the alias injection trades: the last warp's own rows against row 0 repeated.
// Returns the clean and aliased per-thread sums; equal means the block-sum witness is BLIND.
std::pair<double, double> kv_alias_sums(unsigned long long rows, unsigned long long warps,
                                        unsigned long long passes) {
    const unsigned long long gw = warps - 1;
    double clean = 0.0, aliased = 0.0;
    unsigned long long n = 0;
    for (unsigned long long r = gw; r < rows; r += warps, ++n) clean += 8.0 * valu_tag((double)r);
    aliased = (double)n * 8.0 * valu_tag(0.0);
    return {clean * (double)passes, aliased * (double)passes};
}

// The LDS swap injection is a PERMUTATION: every row is still read exactly once. The block sum is
// therefore blind to it by construction and only the per-thread account can see it. Proving both
// halves on the host, before any device run, is what makes the device case meaningful rather than
// hopeful — it is the same argument as "a check whose reference side is derived from the thing under
// test cannot adjudicate it", applied to the witness pair.
bool lds_swap_is_permutation(int threads, int reuse, unsigned long long iters, int* threads_differ) {
    const int nw = threads / irp::kWarpWidth;
    double clean_total = 0.0, swap_total = 0.0;
    int differ = 0;
    std::vector<double> clean_per_thread(threads), swap_per_thread(threads);
    for (int t = 0; t < threads; ++t) {
        const int lane = t % irp::kWarpWidth, w = t / irp::kWarpWidth;
        double clean = 0.0, swapped = 0.0;
        for (int k = 0; k < irp::kLdsWordsPerThread; ++k) {
            const int row = w + k * nw;
            const int off = row * kHeadDim + lane * 8;
            double cs = 0.0, ss = 0.0;
            for (int i = 0; i < 8; ++i) {
                cs += valu_tag(row * kHeadDim + lane * 8 + i);
                // The kernel's swap reads the neighbour lane's slot: `off ^ 8`, a bijection over the
                // staged words, so the total is identical while the per-thread sums are not.
                ss += valu_tag((off ^ 8) + i);
            }
            clean += cs; swapped += ss;
        }
        clean_per_thread[t] = clean * reuse * iters;
        swap_per_thread[t] = swapped * reuse * iters;
        if (clean_per_thread[t] != swap_per_thread[t]) ++differ;
    }
    for (int t = 0; t < threads; ++t) {
        clean_total += clean_per_thread[t];
        swap_total += swap_per_thread[t];
    }
    *threads_differ = differ;
    return clean_total == swap_total && differ > 0;
}

// ---- selftest: geometry and mirrors, no device, no timing ------------------------------------
struct GeomCase {
    const char* name;
    int kind;   // 0 plan-ok, 1 plan-must-be-rejected, 2 mirror/witness property
    Cfg c;
};

int run_geom(int only) {
    int fails = 0, n = 0;
    auto report = [&](const char* name, bool pass, const std::string& detail) {
        ++n;
        if (only >= 0 && n - 1 != only) return;
        printf("GEOMCASE %-28s ok=%d %s\n", name, pass ? 1 : 0, detail.c_str());
        if (!pass) ++fails;
    };
    // 1. Every billed kernel has a positive, disassembly-derived instruction count and builds a plan.
    for (const Kernel& k : kKernels) {
        Cfg c;
        c.kernel = k.name;
        c.iters = 1000;
        if (k.chains == 0 && std::strcmp(k.family, "lds") == 0) c.reuse = 1;
        if (std::strcmp(k.family, "kv") == 0) { c.footprint_mib = 2; c.passes = 2; }
        Plan p;
        const bool ok = plan_build(c, p);
        char buf[160];
        snprintf(buf, sizeof buf, "wi=%llu warps=%llu expected_total=%.0f",
                 p.k ? p.k->wi : 0ULL, p.warps, p.expected_total);
        report((std::string("plan_ok_") + k.name).c_str(), ok && p.k && p.k->wi > 0, buf);
    }
    // 2. The tag closed form and its loop must agree — the account is an equality, so a mirror bug is
    //    a false PASS waiting to happen.
    {
        unsigned long long nn = 400003;
        double loop = 0.0;
        for (unsigned long long r = 0; r < nn; ++r) loop += 256.0 * valu_tag((double)r);
        const double closed = 256.0 * ((double)nn + 21.0 * (double)(nn / 7)
                                       + (double)((nn % 7) * ((nn % 7) - 1) / 2));
        report("tag_closed_form", loop == closed, "loop==closed");
    }
    // 3. Rejections that must happen, each for the stated reason and not by accident.
    {
        Cfg c; c.kernel = "irp_fma_8"; c.iters = 200000000ULL;   // per-thread sum would cross 2^24
        Plan p; const bool ok = plan_build(c, p);
        report("reject_iters_past_f32_exact", !ok && p.why.find("2^24") != std::string::npos,
               p.why.c_str());
    }
    {
        Cfg c; c.kernel = "irp_fma_8"; c.threads = 96;           // not a power of two
        Plan p; const bool ok = plan_build(c, p);
        report("reject_non_pow2_threads", !ok, p.why.c_str());
    }
    {
        Cfg c; c.kernel = "irp_lds_tile"; c.threads = 64;        // not attn's geometry
        Plan p; const bool ok = plan_build(c, p);
        report("reject_lds_wrong_threads", !ok, p.why.c_str());
    }
    {
        Cfg c; c.kernel = "irp_kv_read"; c.footprint_mib = 2; c.blocks = 640; c.threads = 128;
        Plan p; const bool ok = plan_build(c, p);               // 2560 warps > 4096 rows is fine,
        const bool reject = [&] { Cfg d; d.kernel = "irp_kv_read"; d.footprint_mib = 2;
                                  d.blocks = 4096; Plan e; return !plan_build(d, e); }();
        report("reject_kv_warps_gt_rows", reject && ok, ok ? "plan ok" : p.why.c_str());
    }
    // 4. The alias injection's catchability is DERIVED, and it changes with the geometry. Two
    //    footprints, one catchable and one blind, prove the flag is computed and not hardcoded.
    {
        int caught = 0, blind = 0;
        for (unsigned long long fp = 1; fp <= 40; ++fp) {
            const unsigned long long rows = fp * MiB / kRowBytes;
            const unsigned long long warps = 1280;
            if (rows <= warps) continue;
            const auto s = kv_alias_sums(rows, warps, 1);
            if (s.first == s.second) ++blind; else ++caught;
        }
        char buf[128];
        snprintf(buf, sizeof buf, "derived: caught=%d blind=%d", caught, blind);
        // DERIVED, not hand-listed: which warps would an aliased last row actually change? The answer
        // is a property of the geometry — an alias is catchable exactly when the warp's own last row
        // carries a different tag than row 0 — and here every warp's is, so this defect has no blind
        // subset in this shape. That is the derived must-catch set; asserting blind > 0 here would be
        // asserting a wish. The blind-to-block-sum demonstration is `lds_swap`, and it is separately
        // derived below. (tools/mixed_bw's kv_alias IS blind because its aliased row was read a whole
        // number of times by the schedule, which made the total invariant; different geometry.)
        report("kv_alias_catchable_derived", caught > 0, buf);
    }
    // 5. The LDS swap must be invisible to the block sum and visible per thread, or the case that
    //    demonstrates the two-witness split is vacuous.
    {
        int differ = 0;
        const bool perm = lds_swap_is_permutation(128, 1, 1000, &differ);
        char buf[128];
        snprintf(buf, sizeof buf, "permutation=%d threads_that_differ=%d", perm ? 1 : 0, differ);
        report("lds_swap_blinds_block_sum", perm, buf);
    }
    printf("SELFTEST-GEOM %s cases=%d fails=%d\n", fails ? "FAILED" : "ok", n, fails);
    return fails ? 4 : 0;
}

// ---- selftest: device byte witnesses, one child process per case ------------------------------
// A faulting kernel poisons the whole HIP context and every later case in that process silently
// never runs, which is how tools/mixed_bw's first --selftest shipped half-broken. One child per case
// is the fix, and the parent never touches the device.
struct DevCase {
    const char* name;
    Cfg c;
    int expect;      // 0 clean, 1 the read account must fail, 2 only the per-thread account may fail
};

enum Expect { kClean = 0, kReadAccountMustFail = 1, kEachOnly = 2, kDerived = 3 };

int run_dev_case(const DevCase& t) {
    Plan p;
    if (!plan_build(t.c, p)) {
        printf("CASE %-26s ok=0 verdict=plan_rejected(%s)\n", t.name, p.why.c_str());
        return 1;
    }
    double icpcb = 0, gbs = 0, lds_gbs = 0;
    const RunResult R = run_one(t.c, p, &icpcb, &gbs, &lds_gbs);
    int expect = t.expect;
    std::string detail;
    if (t.expect == kDerived) {
        // The alias case asks the DERIVATION, not a hand-written list: catchable here, or blind?
        const auto s = kv_alias_sums(p.rows, p.warps, p.passes);
        expect = (s.first == s.second) ? kClean : kReadAccountMustFail;
        char buf[128];
        snprintf(buf, sizeof buf, "derived_catchable=%d ", s.first != s.second);
        detail = buf;
    }
    bool pass;
    const char* verdict;
    if (expect == kClean) {
        pass = R.checksum_ok && (t.c.want_each ? R.each_bad == 0 : true) && R.guards_ok;
        verdict = pass ? "clean" : "UNEXPECTED_FAILURE";
    } else if (expect == kEachOnly) {
        pass = R.checksum_ok && R.each_bad > 0 && R.guards_ok;
        verdict = R.checksum_ok ? (R.each_bad > 0 ? "caught_per_thread_only"
                                                 : "NOT_CAUGHT_AT_ALL")
                                : "caught_by_more_than_the_per_thread_account";
    } else {
        pass = !R.checksum_ok && R.guards_ok;
        verdict = R.checksum_ok ? "ESCAPED_the_read_account"
                                : (R.guards_ok ? "caught_by_read_account" : "caught_by_more_than_read");
    }
    printf("CASE %-26s ok=%d %sverdict=%s bad_blocks=%d/%d each_bad=%d guards=%s inject=%s\n",
           t.name, pass ? 1 : 0, detail.c_str(), verdict, R.bad_blocks,
           (int)(p.blocks), R.each_bad, R.guards_ok ? "intact" : "written", t.c.inject.c_str());
    return pass ? 1 : 0;      // child exits 1 to mean "the case passed", see run_selftest
}

std::vector<DevCase> dev_cases() {
    std::vector<DevCase> v;
    auto add = [&](const char* name, const char* kern, const char* inject, int expect,
                   unsigned long long iters, int reuse, double fp, bool each, bool guards) {
        Cfg c;
        c.kernel = kern; c.inject = inject; c.iters = iters; c.reuse = reuse;
        c.footprint_mib = fp; c.launches = 2; c.reps = 1; c.want_each = each; c.guards = guards;
        if (std::strcmp(kern, "irp_kv_read") == 0) { c.passes = 4; c.blocks = 128; }
        v.push_back({name, c, expect});
    };
    add("clean_fma8", "irp_fma_8", "none", kClean, 5000, 1, 2, false, false);
    add("inj_drop_chain", "irp_fma_8", "drop_chain", kReadAccountMustFail, 5000, 1, 2, false, false);
    add("clean_fma32", "irp_fma_32", "none", kClean, 3000, 1, 2, false, false);
    add("clean_serial8", "irp_serial_8", "none", kClean, 5000, 1, 2, false, false);
    add("clean_cvt8", "irp_cvt_8", "none", kClean, 5000, 1, 2, false, false);
    add("clean_lds_r1", "irp_lds_tile", "none", kClean, 2000, 1, 2, true, false);
    add("clean_lds_r4", "irp_lds_tile", "none", kClean, 500, 4, 2, true, false);
    add("inj_lds_slot_off", "irp_lds_tile", "lds_slot_off", kReadAccountMustFail, 2000, 1, 2, true,
        false);
    add("inj_lds_short", "irp_lds_tile", "lds_short", kReadAccountMustFail, 2000, 1, 2, true, false);
    // The control that decides whether the per-thread witness is wired up: a permutation the block
    // sum CANNOT see. If this case comes back clean, the second witness is decoration.
    add("inj_lds_swap", "irp_lds_tile", "lds_swap", kEachOnly, 2000, 1, 2, true, false);
    add("clean_kv_2mib", "irp_kv_read", "none", kClean, 1, 1, 2, false, true);
    add("clean_kv_32mib", "irp_kv_read", "none", kClean, 1, 1, 32, false, true);
    add("inj_kv_short", "irp_kv_read", "kv_short", kReadAccountMustFail, 1, 1, 2, false, true);
    add("inj_kv_stray", "irp_kv_read", "kv_stray", kReadAccountMustFail, 1, 1, 2, false, true);
    add("inj_kv_alias", "irp_kv_read", "kv_alias", kDerived, 1, 1, 2, false, true);
    return v;
}

int run_selftest(int only) {
    const std::vector<DevCase> cases = dev_cases();
    if (only >= 0) return run_dev_case(cases[(size_t)only]) ? 0 : 4;
    int fails = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        fflush(nullptr);
        const pid_t pid = fork();
        if (pid == 0) _exit(run_dev_case(cases[i]) ? 0 : 1);
        int status = 0;
        waitpid(pid, &status, 0);
        const bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (!ok) {
            ++fails;
            fprintf(stderr, "case %s FAILED (child status %d)\n", cases[i].name, status);
        }
    }
    printf("SELFTEST %s cases=%zu fails=%d\n", fails ? "FAILED" : "ok", cases.size(), fails);
    return fails ? 4 : 0;
}


}  // namespace

// ---- main ------------------------------------------------------------------------------------
void usage() {
    printf(
        "issue_rate_probe - VALU issue rate and KV-shaped read bandwidth vs working-set size\n"
        "\n"
        "  --kernel NAME      one of:\n");
    for (const Kernel& k : kKernels)
        printf("                       %-14s %s\n", k.name, k.note ? k.note : "");
    printf(
        "  --iters N          counted-loop trips per launch (bounded by the F32-exact witness)\n"
        "  --blocks N --threads N   grid; default 320 x 128 = 1280 wavefronts\n"
        "  --reuse N          lds: read passes over one staged tile. 1 = the brief's 12 KiB write\n"
        "                     + 12 KiB read; 4 = attn's G=4 read:write ratio\n"
        "  --footprint-mib F  kv: the independent variable (2/8/32/128/512)\n"
        "  --passes N         re-walks inside one launch; 0 = solve for --min-launch-mib\n"
        "  --launches N --reps N    timed launches per rep, best-of-N reps (main agent only)\n"
        "  --cu-count N --clock-mhz F  divisor for the per-CU-per-clock number (40 / 2894 default;\n"
        "                     pass 20 for HIP's reading - this box disagrees with itself)\n"
        "  --inject NAME      drop_chain|lds_slot_off|lds_short|lds_swap|kv_short|kv_alias|kv_stray\n"
        "  --dry-run          print the plan and its bill; touches no device, takes no timing\n"
        "  --selftest-geom    host mirrors, witness properties, rejections. No device, no timing\n"
        "  --selftest         device byte witnesses, one child process per case. No timing printed\n"
        "\n"
        "Timing rows belong to the main agent under tools/gpu_lock.sh. This binary does not acquire\n"
        "it and never will: a caller that holds the lock and then runs a self-locking driver deadlocks.\n");
}

int main(int argc, char** argv) {
    Cfg c;
    int only = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); exit(2); }
            return argv[++i];
        };
        if (a == "--kernel") c.kernel = next("--kernel");
        else if (a == "--iters") c.iters = strtoull(next("--iters").c_str(), nullptr, 10);
        else if (a == "--blocks") c.blocks = atoi(next("--blocks").c_str());
        else if (a == "--threads") c.threads = atoi(next("--threads").c_str());
        else if (a == "--reuse") c.reuse = atoi(next("--reuse").c_str());
        else if (a == "--footprint-mib") c.footprint_mib = atof(next("--footprint-mib").c_str());
        else if (a == "--passes") c.passes = strtoull(next("--passes").c_str(), nullptr, 10);
        else if (a == "--min-launch-mib") c.min_launch_mib = atof(next("--min-launch-mib").c_str());
        else if (a == "--launches") c.launches = atoi(next("--launches").c_str());
        else if (a == "--reps") c.reps = atoi(next("--reps").c_str());
        else if (a == "--verify") c.verify = atoi(next("--verify").c_str());
        else if (a == "--inject") c.inject = next("--inject");
        else if (a == "--cu-count") c.cu_count = atoi(next("--cu-count").c_str());
        else if (a == "--clock-mhz") c.clock_mhz = atof(next("--clock-mhz").c_str());
        else if (a == "--tag") c.tag = next("--tag");
        else if (a == "--only-case") only = atoi(next("--only-case").c_str());
        else if (a == "--dry-run") c.dry_run = true;
        else if (a == "--selftest") c.selftest = true;
        else if (a == "--selftest-geom") c.selftest_geom = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(); return 2; }
    }
    if (c.selftest_geom) return run_geom(only);
    if (c.selftest) return run_selftest(only);

    Plan p;
    if (!plan_build(c, p)) {
        fprintf(stderr, "plan rejected: %s\n", p.why.c_str());
        return 2;
    }
    const bool is_kv = std::string(p.k->family) == "kv";
    const bool is_lds = std::string(p.k->family) == "lds";
    if (c.dry_run) {
        printf("PLAN kernel=%s family=%s blocks=%d threads=%d warps=%llu iters=%llu",
               p.k->name, p.k->family, p.blocks, p.threads, p.warps, p.iters);
        if (is_kv) printf(" footprint_mib=%.3f rows=%llu passes=%llu rd_mib_per_launch=%.3f",
                          p.footprint_mib, p.rows, p.passes, p.launch_bytes_mib);
        if (is_lds) printf(" reuse=%d lds_mib_per_launch=%.3f", p.reuse, p.lds_mib_total);
        printf(" warp_inst_per_iter=%llu(isa) warp_inst_total=%.0f"
               " expected_block_sum=%.0f expected_total=%.0f max_thread_sum=%.0f\n",
               p.k->wi, p.warp_inst_total,
               p.exp_block.empty() ? 0.0 : p.exp_block[0], p.expected_total, p.max_thread_sum);
        return 0;
    }

    double icpcb = 0, gbs = 0, lds_gbs = 0;
    const RunResult R = run_one(c, p, &icpcb, &gbs, &lds_gbs);
    const int suspect = (is_kv && gbs > 240.0) ? 1 : 0;
    printf("IRP tag=%s kernel=%s family=%s blocks=%d threads=%d warps=%llu iters=%llu"
           " warp_inst_per_iter=%llu warp_inst_total=%.0f"
           " best_ms_per_launch=%.6f icpcb=%.4f cu_count=%d clock_mhz=%.0f",
           c.tag.c_str(), p.k->name, p.k->family, p.blocks, p.threads, p.warps, p.iters,
           p.k->wi, p.warp_inst_total, R.best_ms, icpcb, c.cu_count, c.clock_mhz);
    if (is_kv)
        printf(" footprint_mib=%.3f rows=%llu passes=%llu rd_mib_per_launch=%.3f"
               " gbs=%.2f pct_of_read_probe=%.1f pct_of_resident_tier=%.1f",
               p.footprint_mib, p.rows, p.passes, p.launch_bytes_mib, gbs,
               100.0 * gbs / kReadProbeGbs, 100.0 * gbs / kGdnResidentGbs);
    if (is_lds)
        printf(" reuse=%d lds_mib_per_launch=%.3f lds_gbs=%.2f", p.reuse, p.lds_mib_total, lds_gbs);
    printf(" launches=%d parsed_launches=%d bad_blocks=%d each_bad=%d%s"
           " checksum=%s guards=%s suspect=%d inject=%s\n",
           c.launches, R.parsed_launches, R.bad_blocks, R.each_bad,
           R.each_ran ? "" : " each=not_claimed",
           R.checksum_ok ? "ok" : "FAIL", R.guards_ok ? "intact" : "written", suspect,
           c.inject.c_str());
    if (suspect)
        fprintf(stderr, "NOTE: above the 240 GB/s read probe. For a resident working set that is the"
                        " finding and not a bug (devices/strix-halo-memory-bandwidth.md 0b: an 18 MiB"
                        " set reads at 918 GB/s), but it is only believable with checksum=ok and a"
                        " footprint sweep that straddles the tier.\n");
    if (!R.checksum_ok || (R.each_ran && R.each_bad)) return 3;
    return 0;
}

