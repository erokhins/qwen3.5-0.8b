// attn_m2.hip.cpp — the device A/B harness for `prototype/attn_m2` (R1: attention row-tiling at M=2),
// and of the two per-touch instruction levers on both the M = 2 body and the tile body.
//
// ONE PROCESS, ONE SET OF OPERANDS, EVERY FORM, BIT FOR BIT. For every case in the plan this runs
//
//      attn_prep -> <wave form> -> attn_combine
//
// once per active form, against SEPARATE cache replicas and separate output buffers, then counts
// differing BF16 elements PER ROW between the reference form (`ship`) and every other active form,
// plus m2-slim against m2-plain. The four forms:
//
//   ship      `attn_tile_wave`      — the shipped tiled form (`prototype/attn_decode`); the reference
//   m2        `attn_m2_wave`        — the M = 2 body as measured at 2.5402 ms (MODE 0)
//   m2slim    `attn_m2_slim_wave`   — the same M = 2 body, both levers on (--slim 1)
//   shipslim  `attn_slim1_wave`     — the TILE body with both levers, i.e. the form the engine ships
//                                     today, so m2-slim vs ship-slim is one ruler and not two windows
//
// `--slim 0` (the default) runs the two forms of record, ship + m2, which is the set every existing
// number in `results.md` was taken on. `--slim 1` adds `m2slim`, so the admission item — m2-slim
// against m2-plain AND m2-slim against ship — comes out of ONE process on ONE set of operands, which is
// the only way the second comparison is a single-variable claim (same cache contents, same `a.split`,
// same merge, same launch geometry as the form it is diffed against).
//
// The admission gate is bit-identity per row on the per-row lane order, so the only acceptable number
// is zero, and it is printed per row so a partial agreement cannot be read as a pass. `attn_combine` is
// the SHIPPED merge in every launch — the forms share the partial layout slot for slot, which is what
// §19's aliasing argues for (one address expression shared by writer and reader) and it means this
// harness has exactly one merge to get wrong.
//
// NO TIMING IS TAKEN HERE, ON PURPOSE. A worker may not take timings (AGENTS.md), and a number taken
// while this session streams its own tokens is worse than no number. `RESULT` lines therefore carry
// counts, geometry and a byte census, never ms or GB/s. The timing that prices this route is a paired
// A/B in one lock hold, taken by the main agent; `HANDOFF-slim.md` names the commands, and the burst
// below alternates EVERY active form inside one rep, so a four-form comparison is still one ruler.
//
// The byte counters on this part are NOT the traffic witness (GL2C calibration failed by ~100x,
// decision D9), so the traffic column here is BYTES DEMANDED arithmetic: the sum of the K lane rows the
// pair's union walk must fetch, from the same closed forms the CPU mirror is checked against
// (`attn_m2_pair_union` / `attn_m2_row_union` in attn_m2_kernels.h). It says what the kernel ASKS for,
// not what the silicon moved. A slim form demands EXACTLY what its unslimmed twin demands — the levers
// remove issue slots, not bytes — and the two pairs are printed so that identity is checked, not
// asserted.
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

// ---- the forms -----------------------------------------------------------------------------
// ONE index space for "which wave kernel", shared by the correctness path and the burst path, so a
// timing cannot be taken on a different launch plan from the one that was gated. `ship` is the
// reference: every other active form is bit-diffed against it, and `m2slim` is additionally diffed
// against `m2`, because THAT pair is what the two levers are claims about.
enum Form { F_SHIP = 0, F_M2 = 1, F_M2SLIM = 2, F_SHIPSlim = 3, NFORMS = 4 };
static const char* kFormName[] = {"ship", "m2", "m2slim", "shipslim"};
// The RESULT-key suffix per form. `ndiff_aout`/`ndiff_gated` keep their existing meaning (m2 vs ship)
// so every number already written down in `results.md` still names the same measurement.
static const char* kFormKey[] = {"", "", "_m2slim", "_shipslim"};

// An ordered set of forms, normalised to the declaration order above so neither the RESULT line nor the
// burst order depends on how an operator typed `--forms`.
struct FormSet {
    int f[NFORMS] = {0, 0, 0, 0};
    int n = 0;
    bool has(int form) const {
        for (int i = 0; i < n; ++i) if (f[i] == form) return true;
        return false;
    }
    void add(int form) { if (!has(form)) f[n++] = form; }
    std::string names() const {
        std::string s;
        for (int i = 0; i < n; ++i) { if (i) s += ","; s += kFormName[f[i]]; }
        return s;
    }
};

static FormSet forms_of(const std::string& spec, int slim) {
    // `ship` is ALWAYS active: it is the reference every other form is bit-diffed against, and a diff
    // against a form that never ran is worse than no diff. Tokens are matched EXACTLY, because a
    // substring test would make `--forms m2slim` silently add `m2` ("m2" is inside "m2slim") and
    // `--forms shipslim` silently add `ship` by accident rather than by this rule.
    FormSet fs;
    fs.add(F_SHIP);
    if (spec.empty()) {
        fs.add(F_M2);
        if (slim) fs.add(F_M2SLIM);
        return fs;
    }
    size_t p = 0;
    while (p < spec.size()) {
        const size_t c = spec.find(',', p);
        const std::string tok = spec.substr(p, c == std::string::npos ? std::string::npos : c - p);
        int found = -1;
        for (int i = 0; i < NFORMS; ++i) {
            if (tok == kFormName[i]) found = i;
        }
        if (found < 0) {
            fprintf(stderr, "--forms: no form \"%s\"; known: ship,m2,m2slim,shipslim\n", tok.c_str());
            exit(1);
        }
        fs.add(found);
        if (c == std::string::npos) break;
        p = c + 1;
    }
    return fs;
}

// One bit-diff pair to run. `key` is the RESULT-line suffix; the empty key is the m2-vs-ship pair the
// gate has always printed.
struct Pair { int a, b; const char* key; };

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

// One replica of one file per active form: the §05.7 append writes the cache IN PLACE, so each form
// needs its own copy or the second launch would read the first one's appended lanes at the first
// form's slots.
static void upload_forms(const std::string& path, size_t expect_elems, const FormSet& fs,
                         hip_bfloat16** dst) {
    const std::vector<char> raw = slurp(path);
    if (raw.size() != expect_elems * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu BF16 elements\n", path.c_str(), raw.size(),
                expect_elems);
        exit(1);
    }
    for (int i = 0; i < fs.n; ++i) {
        hip_bfloat16** d = dst + fs.f[i];
        CK(hipMalloc(d, raw.size()));
        CK(hipMemcpy(*d, raw.data(), raw.size(), hipMemcpyHostToDevice));
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
    hip_bfloat16 *kc[NFORMS] = {nullptr, nullptr, nullptr, nullptr};
    hip_bfloat16 *vc[NFORMS] = {nullptr, nullptr, nullptr, nullptr};
    hip_bfloat16 *qrope[NFORMS] = {nullptr, nullptr, nullptr, nullptr};
    hip_bfloat16 *krope[NFORMS] = {nullptr, nullptr, nullptr, nullptr};
    hip_bfloat16 *gate = nullptr, *rcs = nullptr;
    hip_bfloat16 *aout[NFORMS] = {nullptr, nullptr, nullptr, nullptr};
    hip_bfloat16 *gated[NFORMS] = {nullptr, nullptr, nullptr, nullptr};
    float *part[NFORMS] = {nullptr, nullptr, nullptr, nullptr};
    int *diff = nullptr;          // 2 x nrows ints per ACTIVE pair
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
    std::string plan_path, negname = "none", outdir = ".", formspec;
    int split = 1, tile = TILE, nrows_cap = 0, burst = 0, reps = 5, geom_guard = 0, slim = 0;
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
        else if (a == "--geom-guard") geom_guard = 1;
        else if (a == "--slim") slim = std::atoi(next().c_str());
        else if (a == "--forms") formspec = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    // Host-only: no HIP call, no device, no lock. Walks the (lanes_last, split, full, nrows) space and
    // requires `attn_m2_ok`'s verdict to agree with what the kernel geometry would do, in both
    // directions (see `attn_m2_geometry_guard` in the header for why this exists).
    if (geom_guard) return attn_m2_geometry_guard(1) ? 1 : 0;
    if (!attn_m2_slim_ok(slim)) {
        fprintf(stderr, "--slim must be 0 (no levers: ship + m2) or 1 (both levers: adds m2slim). "
                        "Single-lever attribution lives in prototype/attn_valu_slim --slim 2|3\n");
        return 1;
    }
    const FormSet fs = forms_of(formspec, slim);
    // The pairs to bit-diff: every active form against the reference, plus m2-slim against m2-plain when
    // both are present (that pair IS the lever claim). Built here, before any buffer is sized, so the
    // diff buffer cannot be sized by a guess made later.
    std::vector<Pair> pairs;
    for (int i = 0; i < fs.n; ++i) {
        if (fs.f[i] != F_SHIP) pairs.push_back({fs.f[i], F_SHIP, kFormKey[fs.f[i]]});
    }
    if (fs.has(F_M2) && fs.has(F_M2SLIM)) pairs.push_back({F_M2SLIM, F_M2, "_m2slim_vs_m2"});
    if (pairs.empty()) { fprintf(stderr, "--forms must name a form besides `ship`\n"); return 1; }
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
            // a row's slice exceed the staging. Raised here and echoed, so EVERY form is launched with
            // the SAME `a.split` and the partial slots cannot mean two things.
            split = need;
        }
        c.qg = upload(qgf, (size_t)N_Q * 2 * DH * c.nrows);
        c.kg = upload(kgf, (size_t)N_KV * DH * c.nrows);      // ROW_KV per row, like attn_decode.hip.cpp
        c.vg = upload(vgf, (size_t)N_KV * DH * c.nrows);
        c.qn = upload(qnf, DH);
        c.kn = upload(knf, DH);
        upload_forms(kcf, (size_t)N_KV * c.cap * DH, fs, c.kc);
        upload_forms(vcf, (size_t)N_KV * c.cap * DH, fs, c.vc);
        for (int fi = 0; fi < fs.n; ++fi) {
            const int f = fs.f[fi];
            CK(hipMalloc(&c.qrope[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.krope[f], (size_t)N_KV * DH * 2 * c.nrows));
            CK(hipMalloc(&c.aout[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMalloc(&c.gated[f], (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMemset(c.aout[f], 0xA5, (size_t)N_Q * DH * 2 * c.nrows));
            CK(hipMemset(c.gated[f], 0xA5, (size_t)N_Q * DH * 2 * c.nrows));
            // ONE footprint expression for every form: the partial layout is shared slot for slot, which
            // is what lets `attn_combine` stay one function (§20.1).
            const size_t pe = (size_t)attn_m2_part_floats(c.nrows, c.lanes_last()) * 4;
            CK(hipMalloc(&c.part[f], pe));
            CK(hipMemset(c.part[f], 0, pe));
        }
        CK(hipMalloc(&c.gate, (size_t)N_Q * DH * 2 * c.nrows));
        CK(hipMalloc(&c.rcs, (size_t)ROT * 2 * 2));
        CK(hipMalloc(&c.diff, (size_t)pairs.size() * c.nrows * 4 * 2));
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
    printf("plan %s | cases=%zu | rows=%d | split=%d | tile=%d | neg=%s | slim=%d | forms=%s"
           " | MAXSC=%d STAGE=%d VSLOT=%d | LDS=%d B | grid m2=%dx%dx%d | grid tile=%dx%dx%d\n",
           plan_path.c_str(), cases.size(), rows_total, split, tile, negname.c_str(), slim,
           fs.names().c_str(), MAXSC, ATTN_M2_STAGE, ATTN_M2_VSLOT, kM2LdsBytes, N_KV, split,
           attn_m2_rtiles_of(rows_total), N_KV, std::max(split, tile_split_of(worst_lanes)),
           tile_rtiles_of(1));
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    // ONE place builds a launch plan, used by the correctness path and the burst path alike, so a timing
    // can never be taken on a different launch plan from the one that was gated. (Divergence between a
    // gated path and a timed path is how a timing ends up measuring a kernel nobody ships.) Same
    // operands, same `a.split`, same merge, for every form.
    auto core_args = [&](Case& c, int form) {
        CoreArgs a{};
        const bool tile_form = (form == F_SHIP || form == F_SHIPSlim);
        a.qg = c.qg; a.kg = c.kg; a.vg = c.vg;
        a.kcache = c.kc[form]; a.vcache = c.vc[form];
        a.kc_as = a.kcache; a.vc_as = a.vcache;      // the append writes in place (§05.7)
        a.qn = c.qn; a.kn = c.kn;
        a.qrope = c.qrope[form]; a.krope = c.krope[form];
        a.aout = c.aout[form]; a.gout = c.gated[form];
        a.rope_out = c.rcs;
        a.part = c.part[form]; a.stale = nullptr;
        a.cap = c.cap; a.pos = c.pos; a.lvalid = c.lvalid; a.nrows = c.nrows;
        a.split = split;                             // IDENTICAL for every form, on purpose
        a.neg = neg; a.full = 1; a.score_f32 = 0; a.fused = 0; a.tile = tile;
        a.tile_kernel = tile_form ? 1 : 0;           // which CORE route a one-row wave would take; the
                                                     // slim tile kernels do not branch on it
        return a;
    };
    // ONE place launches. Grid and thread count belong to the form's FAMILY: the two tile forms share
    // `attn_tile_wave`'s launch, the two m2 forms share `attn_m2_wave`'s, and each slim form is launched
    // exactly as its unslimmed twin is — which is what keeps the A/B single-variable.
    auto launch_wave = [&](Case& c, int form) {
        CoreArgs a = core_args(c, form);
        const dim3 tgrid(N_KV, std::max(split, tile_split_of(c.lanes_last())),
                         tile_rtiles_of(c.nrows));
        const dim3 mgrid(N_KV, split, attn_m2_rtiles_of(c.nrows));
        switch (form) {
            case F_SHIP:     attn_tile_wave<<<tgrid, ATTN_TILE_THREADS>>>(a); break;
            case F_SHIPSlim: attn_slim1_wave<<<tgrid, ATTN_TILE_THREADS>>>(a); break;
            case F_M2:       attn_m2_wave<<<mgrid, ATTN_M2_THREADS>>>(a); break;
            case F_M2SLIM:   attn_m2_slim_wave<<<mgrid, ATTN_M2_THREADS>>>(a); break;
            default: fprintf(stderr, "no kernel for form %d\n", form); exit(1);
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
        launch_wave(c, form);
        attn_combine<<<dim3(N_Q, c.nrows), THREADS>>>(a);   // the shipped merge, every form
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
    };

    // Bytes demanded per form family, from the kernel's own closed forms: K lane rows of the pair's
    // union walk for the m2 forms, SUM(lanes) for the tile forms (results.md §20.5: the re-use the
    // shipped tile form actually has is G, at block level, not across rows). The slim forms are NOT a
    // separate census — they demand what their twins demand, and printing the same number twice is how
    // a reader checks that sentence instead of trusting it.
    auto bytes_demanded = [&](Case& c, bool m2_family) {
        if (!m2_family) {
            int lanes_total = 0;
            for (int r = 0; r < c.nrows; ++r) lanes_total += c.lvalid + r;
            return (double)lanes_total * N_KV * 2 * DH * 2;
        }
        int rows = 0;
        for (int rt = 0; rt < attn_m2_rtiles_of(c.nrows); ++rt) {
            const int n_lo = c.lvalid + ATTN_M2_M * rt;
            // `.total` — lane rows summed over lane blocks. The bound is `max_block`, and the refusal
            // predicate uses that one; conflating the two is results.md §2's blocked-widths bug.
            rows += (ATTN_M2_M * rt + 1 < c.nrows) ? attn_m2_pair_union(n_lo, split, 1).total
                                                   : attn_m2_row_union(n_lo, split, 1).total;
        }
        return (double)rows * N_KV * 2 * DH * 2;
    };

    int total_fail = 0;
    for (auto& c : cases) {
        for (int i = 0; i < fs.n; ++i) run_form(c, fs.f[i], neg);
        // One diff_rows pair per entry in `pairs`, laid out in `c.diff` as
        // [pair][aout|gated][row]. Sized from `pairs` above, so the layout and the allocation are the
        // same arithmetic.
        for (size_t p = 0; p < pairs.size(); ++p) {
            diff_rows<<<c.nrows, 256>>>(c.aout[pairs[p].a], c.aout[pairs[p].b],
                                        c.diff + p * c.nrows * 2, N_Q * DH);
            diff_rows<<<c.nrows, 256>>>(c.gated[pairs[p].a], c.gated[pairs[p].b],
                                        c.diff + p * c.nrows * 2 + c.nrows, N_Q * DH);
        }
        std::vector<int> dhost((size_t)pairs.size() * c.nrows * 2);
        CK(hipMemcpy(dhost.data(), c.diff, dhost.size() * sizeof(int), hipMemcpyDeviceToHost));
        const double b_ship = bytes_demanded(c, false);
        const double b_m2 = bytes_demanded(c, true);
        printf("RESULT case=%s rows=%d lanes=%d-%d split=%d slim=%d forms=%s",
               c.name.c_str(), c.nrows, c.lvalid, c.lanes_last(), split, slim, fs.names().c_str());
        for (size_t p = 0; p < pairs.size(); ++p) {
            int suma = 0, sumg = 0;
            for (int r = 0; r < c.nrows; ++r) {
                suma += dhost[p * c.nrows * 2 + r];
                sumg += dhost[p * c.nrows * 2 + c.nrows + r];
            }
            printf(" ndiff_aout%s=%d ndiff_gated%s=%d", pairs[p].key, suma, pairs[p].key, sumg);
            if (suma || sumg) {
                for (int r = 0; r < c.nrows; ++r) {
                    const int da = dhost[p * c.nrows * 2 + r], dg = dhost[p * c.nrows * 2 + c.nrows + r];
                    if (da || dg)
                        printf("\n  ROWDIFF pair=%s_vs%s case=%s row=%d pos=%d lanes=%d"
                               " ndiff_aout=%d ndiff_gated=%d",
                               kFormName[pairs[p].a], kFormName[pairs[p].b], c.name.c_str(), r,
                               c.pos + r, c.lvalid + r, da, dg);
                }
                ++total_fail;
            }
        }
        printf(" bytesdemanded_ship=%.0f bytesdemanded_m2=%.0f", b_ship, b_m2);
        if (fs.has(F_M2SLIM)) printf(" bytesdemanded_m2slim=%.0f", b_m2);
        if (fs.has(F_SHIPSlim)) printf(" bytesdemanded_shipslim=%.0f", b_ship);
        printf(" ratio=%.4f\n", b_m2 / b_ship);
        if (!outdir.empty()) {
            for (int i = 0; i < fs.n; ++i) {
                const int f = fs.f[i];
                dump_file(outdir + "/" + c.name + "." + kFormName[f] + ".aout.bin", c.aout[f],
                          (size_t)N_Q * DH * c.nrows);
            }
        }
    }
    printf("RESULT mode=ab cases=%zu rows=%d neg=%s slim=%d forms=%s cases_with_any_difference=%d"
           " claim=bit-identity-per-row-on-the-per-row-lane-order\n",
           cases.size(), rows_total, negname.c_str(), slim, fs.names().c_str(), total_fail);

    // Burst timing, main-agent-only path (workers may not take timings). Only the wave kernel is
    // looped: attn_prep and attn_combine are identical on every side, so looping the trio would add the
    // same constant to every side and dilute the ratio. steady_clock, not hipEventElapsedTime, which
    // returns exactly 0 for sub-ms kernels here. EVERY active form is timed inside one rep, so a
    // mid-run disturbance hits all of them and a three- or four-form comparison is one ruler rather
    // than two windows; best-of-reps, and the pair is quoted together.
    if (burst > 0 && neg == NEG_NONE) {
        for (auto& c : cases) {
            std::vector<double> best(fs.n, 1e30);
            for (int i = 0; i < fs.n; ++i) launch_wave(c, fs.f[i]);   // warm pages: first pass ~35 GB/s
            CK(hipDeviceSynchronize());
            for (int r = 0; r < reps; ++r) {
                for (int i = 0; i < fs.n; ++i) {
                    const int form = fs.f[i];
                    auto t0 = std::chrono::steady_clock::now();
                    for (int n = 0; n < burst; ++n) launch_wave(c, form);
                    CK(hipDeviceSynchronize());
                    auto t1 = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / burst;
                    if (ms < best[i]) best[i] = ms;
                    printf("BURST case=%s rep=%d form=%s ms=%.4f\n", c.name.c_str(), r,
                           kFormName[form], ms);
                }
            }
            printf("RESULT case=%s burst=%d reps=%d forms=%s", c.name.c_str(), burst, reps,
                   fs.names().c_str());
            for (int i = 0; i < fs.n; ++i)
                printf(" best_%s_ms=%.4f", kFormName[fs.f[i]], best[i]);
            printf(" no_timing_taken=0\n");
            // One line per ordered pair (later form over earlier, in the family order above), so the two
            // numbers the route is decided on — m2slim/m2 and m2slim/shipslim — are both here, read out
            // of one window, and a reader can see the pair rather than one side of it.
            for (int i = 0; i < fs.n; ++i)
                for (int j = i + 1; j < fs.n; ++j)
                    printf("SPEEDUP a=%s b=%s best_a_ms=%.4f best_b_ms=%.4f speedup_a_over_b=%.4f\n",
                           kFormName[fs.f[i]], kFormName[fs.f[j]], best[i], best[j],
                           best[j] / best[i]);
        }
    } else {
        printf("NOT TIMED: no timing taken in this run (pass --burst N to time the wave launch)\n");
    }
    return total_fail ? 1 : 0;
}
