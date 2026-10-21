// attn_valu_slim.hip.cpp — the device A/B harness for `prototype/attn_valu_slim` (D20 item 4: the
// per-touch instruction bill of the shipped tiled attention wave).
//
// ONE PROCESS, ONE SET OF OPERANDS, BOTH FORMS, BIT FOR BIT. For every case in the plan this runs
//
//      attn_prep -> attn_tile_wave  -> attn_combine        (the shipped tiled form, `attn_decode`)
//      attn_prep -> attn_slimN_wave -> attn_combine        (the slim form, `--slim N`)
//
// against SEPARATE cache replicas and separate output buffers, and then counts differing BF16
// elements PER ROW between the two. The slim levers claim BIT identity, not a band, so the only
// acceptable number is zero, and it is printed per row so a partial agreement cannot be read as a
// pass. `attn_combine` is the SHIPPED merge in both launches, and the geometry (grid, threads, LDS,
// partial-slot expression) is identical on both sides by construction — the two forms differ in
// exactly the two spellings under test, which is what makes this a single-variable A/B.
//
// `--slim 1` runs both levers (unrolled score dot + guard-free fold); `--slim 2` and `--slim 3` run
// one each, so a regression can be attributed to a lever instead of to the pair. MODE 0 in the
// header is the shipped body copied verbatim; it exists for the ISA census, not as a launch target
// here — the shipping form is the control.
//
// NO TIMING IS TAKEN HERE BY THE AUTHOR. A worker may not take timings (AGENTS.md). `--burst N` is
// present and is the paired A/B the main agent runs under one lock hold; until then every `RESULT`
// line carries counts and geometry and never ms. See `HANDOFF.md` for the numbered device protocol,
// including the reason this binary must be run TWICE: MODE 1's fold is an inline `ds_bpermute_b32`
// whose waitcnt is written by hand, and a racy permute would show up as run-to-run disagreement
// rather than as a wrong-but-stable answer.
//
// Build:  prototype/attn_valu_slim/build.sh
// Run:    see progress/2026-10-16/attn_valu_slim-HANDOFF.md (this binary does NOT self-lock; the caller holds
//         the lock, and releases before invoking any self-locking gate driver).
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

#include "attn_valu_slim_kernels.h"

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
    __shared__ int warp_part[ATTN_TILE_THREADS / 32];
    // The mask must be 64-bit on this HIP (prototype/README.md: implicit promotion of a smaller
    // integer is a static_assert, not a warning).
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffull, v, off);
    const int warp = threadIdx.x / 32;
    if ((threadIdx.x & 31) == 0) warp_part[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (int i = 0; i < ATTN_TILE_THREADS / 32; ++i) t += warp_part[i];
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

// record "this injection is only expressible in the form under test" rather than hiding it.
// The shipped `enum Neg` belongs to `prototype/attn_decode` and is not edited from here. The M = 2
// injections (`maskwide`, `partrow`, `partalias`) are deliberately absent: this form changes no
// routing, no partial layout and no mask, so there is nothing for them to inject into, and a name the
// kernel does not know parses to -1 and is refused by name rather than silently meaning "none".
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
         : s == "slotv" ? NEG_SLOTV : -1;
}

int main(int argc, char** argv) {
    std::string plan_path, negname = "none", outdir = ".";
    int split = 1, tile = TILE, nrows_cap = 0, burst = 0, reps = 5, slim = 1;
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
        else if (a == "--slim") slim = atoi(next().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    // Host-only: no HIP call, no device, no lock. Walks the (lanes_last, split, full, nrows) space and
    // requires `attn_m2_ok`'s verdict to agree with what the kernel geometry would do, in both
    // directions (see `attn_m2_geometry_guard` in the header for why this exists).
    if (!attn_valu_slim_mode_ok(slim)) {
        fprintf(stderr, "--slim must be 1 (both levers), 2 (score-dot lever) or 3 (fold lever)\n");
        return 1;
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
        const int need = tile_split_of(c.lanes_last());
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
            const size_t pe = (size_t)tile_part_floats(c.nrows, c.lanes_last()) * 4;
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
    if (!attn_tile_ok(0, 0, 32, &why)) {
        fprintf(stderr, "attn_tile_ok refused this plan: %s\n", why);
        return 1;
    }
    printf("plan %s | cases=%zu | rows=%d | split=%d | tile=%d | neg=%s | slim=%d (mode %d)"
           " | MAXSC=%d TILE_L=%d TILE_G=%d LDS=%d B | grid=%dx%dx%d\n",
           plan_path.c_str(), cases.size(), rows_total, split, tile, negname.c_str(), slim,
           attn_valu_slim_mode_of(slim), MAXSC, ATTN_TILE_L, ATTN_TILE_G, kTileLdsBytes, N_KV,
           std::max(split, tile_split_of(worst_lanes)), tile_rtiles_of(1));
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
        a.tile_kernel = 1;   // both forms are the tiled route; the slim kernels do not branch on it
        return a;
    };
    // The one place either form is launched. Form 0 is always the shipped `attn_tile_wave`; form 1 is
    // the slim kernel `--slim` selects. Identical grid, threads, args and merge on both sides, so the
    // only difference between the two runs is the two spellings under test.
    auto launch_core = [&](Case& c, int form) {
        CoreArgs a = core_args(c, form);
        const dim3 grid(N_KV, std::max(split, tile_split_of(c.lanes_last())),
                        tile_rtiles_of(c.nrows));
        if (form == 0) attn_tile_wave<<<grid, kTileThreads>>>(a);
        else if (slim == 2) attn_slim2_wave<<<grid, kTileThreads>>>(a);
        else if (slim == 3) attn_slim3_wave<<<grid, kTileThreads>>>(a);
        else attn_slim1_wave<<<grid, kTileThreads>>>(a);
        CK(hipGetLastError());
    };
    auto launch_wave = [&](Case& c, int form) { launch_core(c, form); };

    auto run_form = [&](Case& c, int form, int which_neg) {
        CoreArgs a = core_args(c, form);
        a.neg = which_neg;
        CK(hipMemset(c.part[form], 0, (size_t)tile_part_floats(c.nrows, c.lanes_last()) * 4));
        CoreArgs p = a;
        p.qrope_out = c.qrope[form]; p.krope_out = c.krope[form]; p.gate_out = c.gate;
        attn_prep<<<dim3(N_Q + N_KV, c.nrows), THREADS>>>(p);
        a.qrope = c.qrope[form]; a.krope = c.krope[form];
        launch_core(c, form);
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
        // Bytes demanded: IDENTICAL on the two sides, and that is the point. The slim levers move no
        // traffic — same stage, same rows, same lane order — so any time the A/B shows is instruction
        // time, not bytes. The census is printed so a reader can check the identity rather than trust
        // the sentence: it is SUM(lanes) x per-KV-group K and V rows, `attn_tile_wave`'s own census
        // (results.md §20.5), which the slim forms inherit unchanged.
        int lanes_total = 0;
        for (int r = 0; r < c.nrows; ++r) lanes_total += c.lvalid + r;
        const double bytes_both = (double)lanes_total * N_KV * 2 * DH * 2;
        printf("RESULT case=%s rows=%d lanes=%d-%d split=%d slim=%d ndiff_aout=%d ndiff_gated=%d"
               " bytesdemanded_ship=%.0f bytesdemanded_slim=%.0f ratio=%.4f\n",
               c.name.c_str(), c.nrows, c.lvalid, c.lanes_last(), split, slim, suma, sumg,
               bytes_both, bytes_both, 1.0);
        for (int r = 0; r < c.nrows; ++r) {
            if (dhost[r] || dhost[c.nrows + r]) {
                printf("  ROWDIFF case=%s row=%d pos=%d lanes=%d ndiff_aout=%d ndiff_gated=%d\n",
                       c.name.c_str(), r, c.pos + r, c.lvalid + r, dhost[r], dhost[c.nrows + r]);
            }
        }
        if (suma || sumg) ++total_fail;
        if (!outdir.empty()) {
            dump_file(outdir + "/" + c.name + ".slim.aout.bin", c.aout[1], (size_t)N_Q * DH * c.nrows);
            dump_file(outdir + "/" + c.name + ".ship.aout.bin", c.aout[0], (size_t)N_Q * DH * c.nrows);
        }
        delete[] dhost;
    }
    printf("RESULT mode=ab cases=%zu rows=%d neg=%s slim=%d cases_with_any_difference=%d"
           " claim=bit-identity-per-row-on-the-per-row-lane-order\n",
           cases.size(), rows_total, negname.c_str(), slim, total_fail);

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
                           f == 0 ? "ship" : "slim", ms);
                }
            }
            printf("RESULT case=%s burst=%d reps=%d best_ship_ms=%.4f best_slim_ms=%.4f"
                   " speedup=%.3f no_timing_taken=0\n",
                   c.name.c_str(), burst, reps, best[0], best[1], best[0] / best[1]);
        }
    } else {
        printf("NOT TIMED: no timing taken in this run (pass --burst N to time the wave launch)\n");
    }
    return total_fail ? 1 : 0;
}
