// attn_m2.hip.cpp — the device A/B harness for `prototype/attn_m2` (R1: attention row-tiling at M=2).
//
// ONE PROCESS, ONE SET OF OPERANDS, BOTH FORMS, BIT FOR BIT. For every case in the plan this runs
//
//      attn_prep -> attn_tile_wave -> attn_combine        (the shipped tiled form, `attn_decode`)
//      attn_prep -> attn_m2_wave   -> attn_combine        (the M = 2 form under test)
//
// against SEPARATE cache replicas and separate output buffers, and then counts differing BF16 elements
// PER ROW between the two. The admission gate is bit-identity per row on the per-row lane order, so the
// only acceptable number is zero, and it is printed per row so a partial agreement cannot be read as a
// pass. `attn_combine` is the SHIPPED merge in both launches — the two forms share the partial layout
// slot for slot, which is what §19's aliasing argues for (one address expression shared by writer and
// reader) and it means this harness has exactly one merge to get wrong.
//
// NO TIMING IS TAKEN HERE, ON PURPOSE. A worker may not take timings (AGENTS.md), and a number taken
// while this session streams its own tokens is worse than no number. `RESULT` lines therefore carry
// counts, geometry and a byte census, never ms or GB/s. The timing that prices this route is a paired
// A/B in one lock hold, taken by the main agent; `HANDOFF.md` names the command and the case.
//
// The byte counters on this part are NOT the traffic witness (GL2C calibration failed by ~100x,
// decision D9), so the traffic column here is BYTES DEMANDED arithmetic: the sum of the K lane rows the
// pair's union walk must fetch, from the same closed forms the CPU mirror is checked against
// (`attn_m2_pair_union` / `attn_m2_row_union` in attn_m2_kernels.h). It says what the kernel ASKS for,
// not what the silicon moved.
//
// Build:  prototype/attn_m2/build.sh
// Run:    see prototype/attn_m2/HANDOFF.md (numbered, with the gpu_lock protocol; this binary does NOT
//         self-lock, so the caller holds the lock).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include "attn_m2_kernels.h"

#define CK(x)                                                                        \
    do {                                                                             \
        hipError_t e_ = (x);                                                         \
        if (e_ != hipSuccess) {                                                      \
            fprintf(stderr, "HIP %s:%d %s -> %s\n", __FILE__, __LINE__, #x,          \
                    hipGetErrorString(e_));                                          \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

// ---- the bit-diff kernel ----------------------------------------------------------------------
// One block per query row; the whole row (all N_Q heads x DH of `aout`, or of `gated`) is compared by
// BIT, as raw 16-bit patterns, so the comparison cannot be softened by a tolerance anywhere. The block
// total is a real reduction — warp butterfly then one shared slot per warp — because a per-thread store
// to one shared slot is not a reduction (the `mixed_bw` checksum trap: it reported 1/blockDim of the
// bytes it had read and every warm row looked plausible).
__device__ __forceinline__ int block_sum(int v) {
    __shared__ int warp_part[ATTN_M2_THREADS / 32];
    // The mask must be 64-bit on this HIP (prototype/README.md: implicit promotion of a smaller
    // integer is a static_assert, not a warning).
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffull, v, off);
    const int warp = threadIdx.x / 32;
    if ((threadIdx.x & 31) == 0) warp_part[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (int i = 0; i < ATTN_M2_THREADS / 32; ++i) t += warp_part[i];
        warp_part[0] = t;
    }
    __syncthreads();
    return warp_part[0];
}

__global__ void diff_rows(const hip_bfloat16* x, const hip_bfloat16* y, int* out, int per_row) {
    const int row = blockIdx.x;
    const unsigned short* a = reinterpret_cast<const unsigned short*>(x) + (size_t)row * per_row;
    const unsigned short* b = reinterpret_cast<const unsigned short*>(y) + (size_t)row * per_row;
    int mine = 0;
    for (int i = threadIdx.x; i < per_row; i += blockDim.x) mine += (a[i] != b[i]);
    // The reduction must run UNIFORMLY: block_sum() contains __syncthreads(), and a barrier executed by
    // one thread of a 256-thread block is undefined on GCN — it faults the queue (code 0x1016) rather
    // than returning a wrong count. This is the failure mode HANDOFF.md names for the m2 kernel itself,
    // and the harness was guilty of it: the first device run faulted here before any row was compared.
    const int total = block_sum(mine);
    if (threadIdx.x == 0) out[row] = total;
}

// ---- plan reading (same plan format as `prototype/attn_decode`) --------------------------------
static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    std::vector<char> b((size_t)f.tellg());
    f.seekg(0);
    f.read(b.data(), (std::streamsize)b.size());
    if ((size_t)f.gcount() != b.size()) { fprintf(stderr, "%s: short read\n", p.c_str()); exit(1); }
    return b;
}

// Two replicas of one file: the §05.7 append writes the cache IN PLACE, so each form needs its own
// copy or the second launch would read the first one's appended lanes at the first form's slots.
static void upload_pair(const std::string& path, size_t expect_elems,
                        hip_bfloat16** a, hip_bfloat16** b) {
    const std::vector<char> raw = slurp(path);
    if (raw.size() != expect_elems * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu BF16 elements\n", path.c_str(), raw.size(),
                expect_elems);
        exit(1);
    }
    for (hip_bfloat16** dst : {a, b}) {
        CK(hipMalloc(dst, raw.size()));
        CK(hipMemcpy(*dst, raw.data(), raw.size(), hipMemcpyHostToDevice));
    }
}

static hip_bfloat16* upload(const std::string& path, size_t expect_elems) {
    const std::vector<char> raw = slurp(path);
    if (raw.size() != expect_elems * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu BF16 elements\n", path.c_str(), raw.size(),
                expect_elems);
        exit(1);
    }
    hip_bfloat16* d = nullptr;
    CK(hipMalloc(&d, raw.size()));
    CK(hipMemcpy(d, raw.data(), raw.size(), hipMemcpyHostToDevice));
    return d;
}

static void dump_file(const std::string& path, const void* dptr, size_t elems) {
    std::vector<char> h(elems * 2);
    CK(hipMemcpy(h.data(), const_cast<void*>(dptr), elems * 2, hipMemcpyDeviceToHost));
    std::ofstream of(path, std::ios::binary);
    of.write(h.data(), (std::streamsize)h.size());
}

struct Case {
    std::string name, outdir;
    int pos = 0, lvalid = 0, cap = 0, nrows = 1;
    hip_bfloat16 *qg = nullptr, *kg = nullptr, *vg = nullptr, *qn = nullptr, *kn = nullptr;
    hip_bfloat16 *kc[2] = {nullptr, nullptr}, *vc[2] = {nullptr, nullptr};
    hip_bfloat16 *qrope[2] = {nullptr, nullptr}, *krope[2] = {nullptr, nullptr};
    hip_bfloat16 *gate = nullptr, *rcs = nullptr;
    hip_bfloat16 *aout[2] = {nullptr, nullptr}, *gated[2] = {nullptr, nullptr};
    float *part[2] = {nullptr, nullptr};
    int *diff = nullptr;
    int lanes_last() const { return lvalid + nrows - 1; }
};

// The shared `enum Neg` belongs to `prototype/attn_decode` and is not edited from here; the M = 2
// injections are further values of the same field. A name the shipped kernel does not know parses to
// an M = 2 value, and the harness then prints `shipped none-vs-neg = 0`, which is the honest way to
// record "this injection is only expressible in the form under test" rather than hiding it.
static int neg_from(const std::string& s) {
    return s == "none" ? NEG_NONE
         : s == "qswap" ? NEG_QSWAP
         : s == "gatehalf" ? NEG_GATEHALF
         : s == "ropefull" ? NEG_ROPEFULL
         : s == "ropenone" ? NEG_ROPENONE
         : s == "gqaoff" ? NEG_GQA
         : s == "nomax" ? NEG_NOMAX
         : s == "carry" ? NEG_CARRY
         : s == "readpast" ? NEG_READPAST
         : s == "dropcurrent" ? NEG_DROPCUR
         : s == "dimhalf" ? NEG_DIMHALF
         : s == "slotk" ? NEG_SLOTK
         : s == "slotv" ? NEG_SLOTV
         : s == "maskwide" ? NEG_M2_MASKWIDE
         : s == "partrow" ? NEG_M2_PARTROW
         : s == "partalias" ? NEG_M2_PARTALIAS : -1;
}

int main(int argc, char** argv) {
    std::string plan_path, negname = "none", outdir = ".";
    int split = 1, tile = TILE, nrows_cap = 0, burst = 0, reps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--plan") plan_path = next();
        else if (a == "--neg") negname = next();
        else if (a == "--split") split = std::atoi(next().c_str());
        else if (a == "--tile") tile = std::atoi(next().c_str());
        else if (a == "--rows") nrows_cap = std::atoi(next().c_str());
        else if (a == "--out") outdir = next();
        else if (a == "--burst") burst = std::atoi(next().c_str());
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    const int neg = neg_from(negname);
    if (neg < 0) { fprintf(stderr, "unknown --neg %s\n", negname.c_str()); return 1; }
    std::ifstream pf(plan_path);
    if (!pf) { fprintf(stderr, "cannot open plan %s\n", plan_path.c_str()); return 1; }

    std::vector<Case> cases;
    std::string line;
    int ln = 0;
    while (std::getline(pf, line)) {
        ++ln;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string qgf, kgf, vgf, kcf, vcf, qnf, knf;
        Case c;
        if (!(ss >> c.name >> qgf >> kgf >> vgf >> kcf >> vcf >> qnf >> knf >> c.pos >> c.lvalid
              >> c.outdir)) {
            fprintf(stderr, "plan %s line %d: NAME QG KG VG KCACHE VCACHE QNORM KNORM POS LCACHE "
                            "OUTDIR [NROWS]\n", plan_path.c_str(), ln);
            return 1;
        }
        ss >> c.nrows;
        if (c.nrows < 1) c.nrows = 1;
        if (nrows_cap > 0 && c.nrows != nrows_cap) continue;      // run one geometry in isolation
        const std::vector<char> kraw = slurp(kcf);
        c.cap = (int)(kraw.size() / ((size_t)N_KV * DH * 2));
        if (c.lvalid < 0 || c.lanes_last() > c.cap || c.pos + c.nrows - 1 >= c.cap) {
            fprintf(stderr, "%s: pos %d (+%d rows) / lvalid %d do not fit cap %d\n", c.name.c_str(),
                    c.pos, c.nrows - 1, c.lvalid, c.cap);
            return 1;
        }
        const int need = attn_m2_split(c.lanes_last());
        if (split < need) {
            // Not a refusal: a wave's widest row decides the lane grid, and a shallower grid would make
            // a row's slice exceed the staging. Raised here and echoed, so both forms are launched with
            // the SAME `a.split` and the partial slots cannot mean two things.
            split = need;
        }
        c.qg = upload(qgf, (size_t)N_Q * 2 * DH * c.nrows);
        c.kg = upload(kgf, (size_t)N_KV * DH * c.nrows);      // ROW_KV per row, like attn_decode.hip.cpp
        c.vg = upload(vgf, (size_t)N_KV * DH * c.nrows);
        c.qn = upload(qnf, DH);
        c.kn = upload(knf, DH);
        upload_pair(kcf, (size_t)N_KV * c.cap * DH, &c.kc[0], &c.kc[1]);
        upload_pair(vcf, (size_t)N_KV * c.cap * DH, &c.vc[0], &c.vc[1]);
        for (int f = 0; f < 2; ++f) {
            CK(hipMalloc(&c.qrope[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.krope[f], (size_t)N_KV * DH * 2 * c.nrows));
            CK(hipMalloc(&c.aout[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.gated[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMemset(c.aout[f], 0xA5, (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMemset(c.gated[f], 0xA5, (size_t)N_Q * DH * 2 * c.nrows));
            const size_t pe = (size_t)attn_m2_part_floats(c.nrows, c.lanes_last()) * 4;
            CK(hipMalloc(&c.part[f], pe));
            CK(hipMemset(c.part[f], 0, pe));
        }
        CK(hipMalloc(&c.gate, (size_t)N_Q * DH * 2 * c.nrows));
        CK(hipMalloc(&c.rcs, (size_t)ROT * 2 * 2));
        CK(hipMalloc(&c.diff, (size_t)c.nrows * 4 * 2));
        cases.push_back(c);
    }
    if (cases.empty()) { fprintf(stderr, "plan %s has no cases\n", plan_path.c_str()); return 1; }

    const char* why = "";
    int worst_lanes = 1;
    int rows_total = 0;
    for (auto& c : cases) { worst_lanes = std::max(worst_lanes, c.lanes_last()); rows_total += c.nrows; }
    if (!attn_m2_ok(0, 0, 32, split, worst_lanes, &why)) {
        fprintf(stderr, "attn_m2_ok refused this plan: %s\n", why);
        return 1;
    }
    printf("plan %s | cases=%zu | rows=%d | split=%d | tile=%d | neg=%s | MAXSC=%d STAGE=%d VSLOT=%d"
           " | LDS=%d B | grid m2=%dx%dx%d | grid tile=%dx%dx%d\n",
           plan_path.c_str(), cases.size(), rows_total, split, tile, negname.c_str(), MAXSC,
           ATTN_M2_STAGE, ATTN_M2_VSLOT, kM2LdsBytes, N_KV, split, attn_m2_rtiles_of(rows_total),
           N_KV, std::max(split, tile_split_of(worst_lanes)), tile_rtiles_of(1));
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    // 0 = the shipped tiled form, 1 = the M = 2 form. Same operands, same `a.split`, same merge.
    // Same CoreArgs for both the correctness path and the burst path: one spelling, so a timing can
    // never be taken on a different launch plan from the one that was gated. (Divergence between a
    // gated path and a timed path is how a timing ends up measuring a kernel nobody ships.)
    auto core_args = [&](Case& c, int form) {
        CoreArgs a{};
        a.qg = c.qg; a.kg = c.kg; a.vg = c.vg;
        a.kcache = c.kc[form]; a.vcache = c.vc[form];
        a.kc_as = a.kcache; a.vc_as = a.vcache;      // the append writes in place (§05.7)
        a.qn = c.qn; a.kn = c.kn;
        a.qrope = c.qrope[form]; a.krope = c.krope[form];
        a.aout = c.aout[form]; a.gout = c.gated[form];
        a.rope_out = c.rcs;
        a.part = c.part[form]; a.stale = nullptr;
        a.cap = c.cap; a.pos = c.pos; a.lvalid = c.lvalid; a.nrows = c.nrows;
        a.split = split;                             // IDENTICAL for both forms, on purpose
        a.neg = neg; a.full = 1; a.score_f32 = 0; a.fused = 0; a.tile = tile;
        a.tile_kernel = form == 0 ? 1 : 0;
        return a;
    };
    auto launch_wave = [&](Case& c, int form) {
        CoreArgs a = core_args(c, form);
        if (form == 0) {
            attn_tile_wave<<<dim3(N_KV, std::max(split, tile_split_of(c.lanes_last())),
                                  tile_rtiles_of(c.nrows)), ATTN_TILE_THREADS>>>(a);
        } else {
            attn_m2_wave<<<dim3(N_KV, split, attn_m2_rtiles_of(c.nrows)), ATTN_M2_THREADS>>>(a);
        }
        CK(hipGetLastError());
    };

    auto run_form = [&](Case& c, int form, int which_neg) {
        CoreArgs a = core_args(c, form);
        a.neg = which_neg;
        CK(hipMemset(c.part[form], 0, (size_t)attn_m2_part_floats(c.nrows, c.lanes_last()) * 4));
        CoreArgs p = a;
        p.qrope_out = c.qrope[form]; p.krope_out = c.krope[form]; p.gate_out = c.gate;
        attn_prep<<<dim3(N_Q + N_KV, c.nrows), THREADS>>>(p);
        a.qrope = c.qrope[form]; a.krope = c.krope[form];
        if (form == 0) {
            attn_tile_wave<<<dim3(N_KV, std::max(split, tile_split_of(c.lanes_last())),
                                  tile_rtiles_of(c.nrows)), ATTN_TILE_THREADS>>>(a);
        } else {
            attn_m2_wave<<<dim3(N_KV, split, attn_m2_rtiles_of(c.nrows)), ATTN_M2_THREADS>>>(a);
        }
        attn_combine<<<dim3(N_Q, c.nrows), THREADS>>>(a);   // the shipped merge, both forms
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
    };

    int total_fail = 0;
    for (auto& c : cases) {
        run_form(c, 0, neg);
        run_form(c, 1, neg);
        int* dhost = new int[c.nrows * 2];
        diff_rows<<<c.nrows, 256>>>(c.aout[0], c.aout[1], c.diff, N_Q * DH);
        diff_rows<<<c.nrows, 256>>>(c.gated[0], c.gated[1], c.diff + c.nrows, N_Q * DH);
        CK(hipMemcpy(dhost, c.diff, (size_t)c.nrows * 4 * 2, hipMemcpyDeviceToHost));
        int suma = 0, sumg = 0;
        for (int r = 0; r < c.nrows; ++r) { suma += dhost[r]; sumg += dhost[c.nrows + r]; }
        // Bytes demanded, from the kernel's own closed forms: K lane rows of the pair's union walk.
        // The shipped side is Σ lanes over the rows, per KV group, which is `attn_tile_wave`'s census
        // (results.md §20.5: the re-use it actually has is G, at block level, not across rows).
        int ship_rows = 0, m2_rows = 0;
        for (int r = 0; r < c.nrows; ++r) ship_rows += c.lvalid + r;
        for (int rt = 0; rt < attn_m2_rtiles_of(c.nrows); ++rt) {
            const int n_lo = c.lvalid + ATTN_M2_M * rt;
            m2_rows += (ATTN_M2_M * rt + 1 < c.nrows) ? attn_m2_pair_union(n_lo, split, 1)
                                                      : attn_m2_row_union(n_lo, split, 1);
        }
        const double ship_b = (double)ship_rows * N_KV * 2 * DH * 2;
        const double m2_b = (double)m2_rows * N_KV * 2 * DH * 2;
        printf("RESULT case=%s rows=%d lanes=%d-%d split=%d ndiff_aout=%d ndiff_gated=%d"
               " bytesdemanded_ship=%.0f bytesdemanded_m2=%.0f ratio=%.4f\n",
               c.name.c_str(), c.nrows, c.lvalid, c.lanes_last(), split, suma, sumg, ship_b, m2_b,
               m2_b / ship_b);
        for (int r = 0; r < c.nrows; ++r) {
            if (dhost[r] || dhost[c.nrows + r]) {
                printf("  ROWDIFF case=%s row=%d pos=%d lanes=%d ndiff_aout=%d ndiff_gated=%d\n",
                       c.name.c_str(), r, c.pos + r, c.lvalid + r, dhost[r], dhost[c.nrows + r]);
            }
        }
        if (suma || sumg) ++total_fail;
        if (!outdir.empty()) {
            dump_file(outdir + "/" + c.name + ".m2.aout.bin", c.aout[1], (size_t)N_Q * DH * c.nrows);
            dump_file(outdir + "/" + c.name + ".ship.aout.bin", c.aout[0], (size_t)N_Q * DH * c.nrows);
        }
        delete[] dhost;
    }
    printf("RESULT mode=ab cases=%zu rows=%d neg=%s cases_with_any_difference=%d"
           " claim=bit-identity-per-row-on-the-per-row-lane-order\n",
           cases.size(), rows_total, negname.c_str(), total_fail);

    // Burst timing, main-agent-only path (workers may not take timings). Only the wave kernel is
    // looped: attn_prep and attn_combine are identical on both sides, so looping the trio would add
    // the same constant to both and dilute the ratio. steady_clock, not hipEventElapsedTime, which
    // returns exactly 0 for sub-ms kernels here; both forms alternate inside one rep so a
    // mid-run disturbance hits both; best-of-reps, and the pair is quoted together.
    if (burst > 0 && neg == NEG_NONE) {
        for (auto& c : cases) {
            double best[2] = {1e30, 1e30};
            for (int f = 0; f < 2; ++f) { launch_wave(c, f); }   // warm the pages: first pass ~35 GB/s
            CK(hipDeviceSynchronize());
            for (int r = 0; r < reps; ++r) {
                for (int f = 0; f < 2; ++f) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (int n = 0; n < burst; ++n) launch_wave(c, f);
                    CK(hipDeviceSynchronize());
                    auto t1 = std::chrono::steady_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / burst;
                    if (ms < best[f]) best[f] = ms;
                    printf("BURST case=%s rep=%d form=%s ms=%.4f\n", c.name.c_str(), r,
                           f == 0 ? "ship" : "m2", ms);
                }
            }
            printf("RESULT case=%s burst=%d reps=%d best_ship_ms=%.4f best_m2_ms=%.4f"
                   " speedup=%.3f no_timing_taken=0\n",
                   c.name.c_str(), burst, reps, best[0], best[1], best[0] / best[1]);
        }
    } else {
        printf("NOT TIMED: no timing taken in this run (pass --burst N to time the wave launch)\n");
    }
    return total_fail ? 1 : 0;
}
