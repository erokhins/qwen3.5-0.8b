/*
 * pack_ab.cpp — does the DISALLOWED lane of a straddling row move the live row? Isolated, on the
 * shipping kernel, at real magnitudes. See `prototype/mtp_verify/ISOLATED.md` for the question and
 * PART 1's source adjudication; this file is the measurement.
 *
 * WHY THIS SHAPE, AND WHY THERE IS NO COPIED ROUND IN IT.
 * The literal test the user asked for is "hold the live half of a packed f16x2 pair byte-fixed and change
 * only the dead half". Transcribing `wll_round<true>` into a probe TU to poke `p[L][2c]` would put a copy of
 * the shipping round between the question and the answer — and AGENTS' association row says a copy agrees
 * with itself, so a null there would be about the copy. The same two questions can be asked of the SHIPPED
 * kernel by mutating one side of the product instead of the other, and both mutations are exactly the ones
 * the datum uses (a different future token changes that lane's K row AND its V row):
 *
 *   ARM A — the DOT. Vary ONLY the disallowed lane's V rows; K byte-identical. Then the disallowed cell's
 *     weight is whatever the kernel wrote — and if that weight is exactly +0.0f (which IS the claim under
 *     test, `engine/attn_wide_llama_kernels.h:237,260-264`), the product is +0.0 for ANY V and the live row
 *     cannot move. Amplify V (x64, x4096, x1e6) and the test gets *more* sensitive, not less: a dead weight
 *     of 1e-5 times a 4096x V is a visible term. So a null here kills the pack as a carrier at every
 *     magnitude, with the kernel's own bits, no copy.
 *   ARM B — the MAX/SUM PATH. Vary ONLY the disallowed lane's K rows; V byte-identical. A K row is ONE
 *     column of the C tile (`acc[L] = wmma(kt, qt, acc[L])` with kt a KV position), so the only thing a K-row
 *     change can reach is that column's SCORE — the "pre-mask score" of the brief. If a masked score reached
 *     the row max or the rowsum (`:241,246,248,260-261,267`), a 4096x K row moves `mn` by 4096x and the row
 *     collapses — impossible to miss. A null here confirms PART 1's reading (mask at :237 precedes every
 *     reduction) on device, at real magnitudes.
 *   ARM C — BOTH, the engine's own order. K and V of the disallowed lane, taken from the perturbed-token
 *     dump = FIX-CROSSROW Result 5's datum replayed inside one process. This is the arm that says whether
 *     the dumped tiles are sufficient to reproduce the datum at all; if C is null, A and B are nulls about a
 *     rig that reproduces nothing, and that is reported as such.
 *
 * THREE THINGS MAKE A NULL HERE MEAN SOMETHING.
 * 1. FIDELITY CANARY (`fid`): with the inputs unmodified, this rig's launch must reproduce the engine's OWN
 *    dumped `pre_gate` tile BYTE-FOR-BYTE, all rows. If it does not, the rig's CoreArgs/geometry/strides are
 *    not the engine's and every other line is void. Printed first, and a failure aborts the run.
 * 2. MUST-FIRE CONTROL (`widen_*`): the engine's own host-side bound lie — `core.lvalid + 1`, which makes
 *    `lanes_of(a, row) = bound + 1` and lane `bound` LEGITIMATELY live for the adjudicated row (the same
 *    `widen1` lever `causal_bound.cpp` uses). With that, the identical mutation MUST move the row. A control
 *    that cannot fire is not a control (AGENTS, repeatedly), and here it also proves the mutation reaches the
 *    kernel through the buffer I wrote, not just the file I edited.
 * 3. REACH WITNESS: every case prints whether row `jbad` (and rows after it) moved. A mutation that moved
 *    nothing anywhere did not reach the model, and that is an unclassified run, not a null.
 *
 * WHAT IT DOES NOT TOUCH. Nothing in `engine/`; the kernels are the engine's own `__global__`s, included
 * read-only (same argument as `causal_bound.cpp:86`: a probe must not be accused of launching a paraphrase).
 * `attn_wide_wave` (knob 1) is run too, because Results 3/7 say both wide spellings leak identically and a
 * result that holds on one spelling only would be about the port, not the pair.
 *
 * NO TIMING. Correctness only: no ms, no burst, no best-of-N, no warm loop, no sweep. `--arm`/`--case` select;
 * output is raw BF16 tiles for `pack_ab_report.py`, which does every ULP comparison through the imported
 * `prototype/bf16_ulp.py` (never retyped here).
 *
 * Build:  bash prototype/mtp_verify/pack_ab/build.sh [outdir]
 * Run:    <outdir>/pack_ab --dump DIR [--pert-dir DIR2] [--seq 1 --layer 3] [--row 8] [--out DIR]
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "engine/kernels.h"
#include "prototype/attn_decode/attn_decode_kernels.h"
#include "engine/attn_slim_kernels.h"
#include "engine/attn_fattn_kernels.h"
#include "engine/attn_wide_kernels.h"
#include "engine/attn_wide_llama_kernels.h"

namespace {

using engine::kernels::attn_wide_llama_wave;
using engine::kernels::attn_wide_wave;
using engine::kernels::wide_rtiles_of;

// ---- bf16 in plain C++ (host side). Same round-to-nearest-even as causal_bound.cpp's helpers; every input
// this file WRITES is a real dumped value or an exact power-of-two scale of one, so no rounding path that
// could disagree with the device's is exercised.
uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}
float bf16_val(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// The scaled mutation: bf16(x * s), exact for s a power of two. Overflow to inf is possible by design in
// the `vinf` case and is part of what that case probes (it is labelled there, not passed off as the leak).
uint16_t bf16_scale(uint16_t b, float s) { return bf16_bits(bf16_val(b) * s); }

// ---- dump loading -------------------------------------------------------------------------------
// Names are `attnNNN_L<layer>_<tag>_r<rows>_lanes<lanes>.bin` (engine/kernels_attn.cpp:388). The dtype is
// inferred from the byte count, never assumed: reading a BF16 tile as F32 — or back — yields a plausible
// ndiff and a NaN cos (AGENTS' dtype row), so `expect_elems` is asserted against BOTH divisors and the file
// is rejected unless exactly one of them fits.
struct Dump {
    int rows = 0, lanes = 0, seq = 0, layer = 0;
    std::vector<uint16_t> qrope, krope, gate, kcache[N_KV], vcache[N_KV], pre_gate;

    // The dump's own name carries the geometry (`..._r<rows>_lanes<lanes>.bin`), so the geometry is READ,
    // never passed in: a caller that typed 128 next to a 640-row dump would price a different body.
    bool scan(const std::string& dir, const char* tag, std::string* out) const {
        char prefix[128];
        std::snprintf(prefix, sizeof prefix, "attn%03d_L%d_%s_r", seq, layer, tag);
        DIR* d = opendir(dir.c_str());
        if (!d) return false;
        bool found = false;
        struct dirent* de = nullptr;
        while ((de = readdir(d)) != nullptr) {
            const std::string n = de->d_name;
            if (n.rfind(prefix, 0) != 0) continue;
            if (n.size() < 4 || n.substr(n.size() - 4) != ".bin") continue;
            *out = dir + "/" + n;
            found = true;
            break;
        }
        closedir(d);
        return found;
    }
    static bool parse_geom(const std::string& name, int* rows, int* lanes) {
        const size_t lp = name.rfind("_lanes");
        const size_t rp = name.rfind("_r");
        if (lp == std::string::npos || rp == std::string::npos || rp >= lp) return false;
        *rows = std::atoi(name.c_str() + rp + 2);
        *lanes = std::atoi(name.c_str() + lp + 6);
        return *rows > 0 && *lanes > 0;
    }
    bool load(const std::string& dir, int want_seq, int want_layer, std::string* why) {
        seq = want_seq;
        layer = want_layer;
        std::string first;
        if (!scan(dir, "in_q_rope", &first)) {
            *why = "no " + dir + "/attn" + std::to_string(seq) + "_L" + std::to_string(layer)
                   + "_in_q_rope_r*_lanes*.bin";
            return false;
        }
        if (!parse_geom(first, &rows, &lanes)) {
            *why = "cannot read r<rows>_lanes<lanes> out of " + first;
            return false;
        }
        struct Need { const char* tag; std::vector<uint16_t>* dst; size_t elems; };
        Need need[] = {
            {"in_q_rope", &qrope, (size_t)rows * ROW_H},
            {"in_k_rope", &krope, (size_t)rows * ROW_KV},
            {"in_gate", &gate, (size_t)rows * ROW_H},
            {"in_kcache0", &kcache[0], (size_t)(lanes + 1) * DH},
            {"in_kcache1", &kcache[1], (size_t)(lanes + 1) * DH},
            {"in_vcache0", &vcache[0], (size_t)(lanes + 1) * DH},
            {"in_vcache1", &vcache[1], (size_t)(lanes + 1) * DH},
            {"pre_gate", &pre_gate, (size_t)rows * ROW_H},
        };
        for (auto& nd : need) {
            std::string tp;
            if (!scan(dir, nd.tag, &tp)) {
                *why = "no tile for " + std::string(nd.tag) + " in " + dir;
                return false;
            }
            int tr = 0, tl = 0;
            if (!parse_geom(tp, &tr, &tl) || tr != rows || tl != lanes) {
                *why = tp + " disagrees about the body geometry with the in_q_rope tile";
                return false;
            }
            FILE* f = std::fopen(tp.c_str(), "rb");
            if (!f) {
                *why = "cannot open " + tp;
                return false;
            }
            std::fseek(f, 0, SEEK_END);
            const long bytes = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            const bool bf16_ok = (size_t)bytes == nd.elems * 2;
            const bool f32_ok = (size_t)bytes == nd.elems * 4;
            if (bf16_ok == f32_ok) {   // neither, or both (elems == 0): refuse rather than guess
                std::fclose(f);
                *why = tp + " is " + std::to_string(bytes) + "B, matches neither "
                       + std::to_string(nd.elems * 2) + " (BF16) nor " + std::to_string(nd.elems * 4)
                       + " (F32) bytes for " + nd.tag;
                return false;
            }
            if (f32_ok) {   // a dump written F32 where this reader expects BF16 is a different instrument
                std::fclose(f);
                *why = tp + " is F32-shaped; this rig reads the BF16 tiles";
                return false;
            }
            nd.dst->resize(nd.elems);
            const size_t got = std::fread(nd.dst->data(), 2, nd.elems, f);
            std::fclose(f);
            if (got != nd.elems) {
                *why = std::string("short read on ") + nd.tag;
                return false;
            }
        }
        return true;
    }
};

// ---- device side --------------------------------------------------------------------------------
// One flat buffer per cache, `cap` lanes per KV group — the layout every core indexes it by
// (`a.kcache + (kv_read * a.cap + j) * DH`), so a mutation and a reload land where the kernel looks.
struct Dev {
    hip_bfloat16 *qrope = nullptr, *krope = nullptr, *qg = nullptr, *vg = nullptr, *qn = nullptr,
                 *kn = nullptr, *kg = nullptr, *aout = nullptr, *gout = nullptr;
    hip_bfloat16 *kcache = nullptr, *vcache = nullptr;
    int cap = 0, rows = 0;
};

#define CK(e)                                                                                         \
    do {                                                                                              \
        const hipError_t err_ = (e);                                                                   \
        if (err_ != hipSuccess) {                                                                      \
            std::printf("FATAL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(err_));              \
            std::exit(2);                                                                              \
        }                                                                                              \
    } while (0)

void upload(hip_bfloat16** dst, const std::vector<uint16_t>& src) {
    CK(hipMalloc(reinterpret_cast<void**>(dst), sizeof(uint16_t) * src.size()));
    CK(hipMemcpy(*dst, src.data(), sizeof(uint16_t) * src.size(), hipMemcpyHostToDevice));
}
// Fill one whole cache buffer (N_KV groups x cap lanes x DH) from the dumped per-group tiles, zeroing the
// slack lanes the dump does not carry (a dump is lanes+1 wide, this rig's cap is lanes+2).
void upload_cache(hip_bfloat16** dst, const std::vector<uint16_t> g[N_KV], int cap) {
    CK(hipMalloc(reinterpret_cast<void**>(dst), sizeof(uint16_t) * (size_t)N_KV * cap * DH));
    CK(hipMemset(*dst, 0, sizeof(uint16_t) * (size_t)N_KV * cap * DH));
    for (int kv = 0; kv < N_KV; ++kv) {
        CK(hipMemcpy(*dst + (size_t)kv * cap * DH, g[kv].data(),
                     sizeof(uint16_t) * g[kv].size(), hipMemcpyHostToDevice));
    }
}
void reload_cache(hip_bfloat16* dst, const std::vector<uint16_t> g[N_KV], int cap) {
    for (int kv = 0; kv < N_KV; ++kv) {
        CK(hipMemcpy(dst + (size_t)kv * cap * DH, g[kv].data(),
                     sizeof(uint16_t) * g[kv].size(), hipMemcpyHostToDevice));
    }
}

void zero_alloc(hip_bfloat16** dst, size_t elems) {
    CK(hipMalloc(reinterpret_cast<void**>(dst), sizeof(uint16_t) * elems));
    CK(hipMemset(*dst, 0, sizeof(uint16_t) * elems));
}

// The engine's own CoreArgs for this body, read off the run's ATTN_DUMP line (FIX-CROSSROW Result 0):
// `cap=130 pos=0 lvalid=1 neg=0 full=1 score_f32=0 fused=0 tile=8 tile_kernel=1 fattn_cover=0 split_eff=1`.
CoreArgs make_core(const Dev& d, int lvalid) {
    CoreArgs a{};
    a.qrope = d.qrope; a.krope = d.krope; a.qg = d.qg; a.kg = d.kg; a.vg = d.vg;
    a.qn = d.qn; a.kn = d.kn;
    a.kcache = d.kcache; a.vcache = d.vcache;
    a.kc_as = d.kcache; a.vc_as = d.vcache;
    a.aout = d.aout; a.gout = d.gout;
    a.cap = d.cap; a.pos = lvalid - 1; a.lvalid = lvalid; a.nrows = d.rows;
    a.split = 1; a.full = 1; a.neg = NEG_NONE; a.score_f32 = 0; a.fused = 0;
    a.tile = 8; a.tile_kernel = 1; a.fattn_cover = 0;
    return a;
}

bool launch_form(int knob, const CoreArgs& a, int rows) {
    if (knob == 2) {
        attn_wide_llama_wave<<<dim3(N_KV, 1, wide_rtiles_of(rows)), WIDE_THREADS>>>(a);
    } else {
        attn_wide_wave<<<dim3(N_KV, 1, wide_rtiles_of(rows)), WIDE_THREADS>>>(a);
    }
    const hipError_t le = hipGetLastError();
    const hipError_t se = hipDeviceSynchronize();
    if (le != hipSuccess || se != hipSuccess) {
        std::printf("LAUNCH FAILED knob=%d last=%s sync=%s\n", knob, hipGetErrorString(le),
                    hipGetErrorString(se));
        return false;
    }
    return true;
}

// ---- the mutations -----------------------------------------------------------------------------
// Which lane is the disallowed one, and why that one. Row `row` of a body whose first row may read
// `lvalid` lanes may read lanes_of(a,row) = lvalid+row (prototype/attn_decode/attn_decode_kernels.h:203).
// The first lane it may NOT read is therefore `lvalid + row`, and that is the lane this rig varies. For
// p128 and row 8 that is lane 9, bound 9 — the worked example in ISOLATED.md PART 1.
struct Mutation {
    const char* name;
    bool k_real, v_real;          // take lane jbad's row from the perturbed dump
    bool k_zero, v_zero;
    float k_scale, v_scale;       // exact power-of-two scale of the REAL row (1 = leave alone)
    bool widens_bound;            // the must-fire control: hand the kernel bound+1
    // A single-dim override applied AFTER the above: `v_dim >= 0` replaces ONE dim of the V row with
    // `v_dim_val`. This is the linear-response probe: if the disallowed lane's weight on this row is w,
    // writing V[jbad][v_dim] = X moves the row's dim-`v_dim` output by w.X/l, so the response is
    // proportional to X and its absence is w == 0 — with no bf16-boundary luck involved.
    int v_dim;
    float v_dim_val;
    const char* arm;
};
const Mutation kMutations[] = {
    // arm C — both, engine order, the datum
    {"real",       true,  true,  false, false, 1.f,     1.f,     false, -1, 0.f,    "C"},
    // arm B — K only: reaches the disallowed cell's pre-mask SCORE and nothing else
    {"kreal",      true,  false, false, false, 1.f,     1.f,     false, -1, 0.f,    "B"},
    {"kzero",      false, false, true,  false, 1.f,     1.f,     false, -1, 0.f,    "B"},
    {"kbig",       false, false, false, false, 64.f,    1.f,     false, -1, 0.f,    "B"},
    {"khuge",      false, false, false, false, 4096.f,  1.f,     false, -1, 0.f,    "B"},
    // arm A — V only: reaches the product the dead weight enters, and nothing upstream of it
    {"vreal",      false, true,  false, false, 1.f,     1.f,     false, -1, 0.f,    "A"},
    {"vzero",      false, false, false, true,  1.f,     1.f,     false, -1, 0.f,    "A"},
    // The linear-response ladder. If the dependence is a weight times V, response doubles at every step.
    {"vhalf",      false, false, false, false, 1.f,     0.25f,   false, -1, 0.f,    "A"},
    {"vsmall",     false, false, false, false, 1.f,     1e-4f,   false, -1, 0.f,    "A"},
    {"v2",         false, false, false, false, 1.f,     2.f,     false, -1, 0.f,    "A"},
    {"v4",         false, false, false, false, 1.f,     4.f,     false, -1, 0.f,    "A"},
    {"v16",        false, false, false, false, 1.f,     16.f,    false, -1, 0.f,    "A"},
    {"vbig",       false, false, false, false, 1.f,     64.f,    false, -1, 0.f,    "A"},
    {"vhuge",      false, false, false, false, 1.f,     4096.f,  false, -1, 0.f,    "A"},
    {"vinf",       false, false, false, false, 1.f,     1e8f,    false, -1, 0.f,    "A"},
    // Single-dim writes: the response of ONE output dim to ONE input dim, so a bf16-boundary luck cannot
    // hide it (dim 62 is one of the four the whole-row case moved, from the first run of this rig).
    // THE LADDER IS MONOTONE AND SPANS ZERO: a linear weight w on the disallowed lane makes out(dim62)
    // monotone in v_dim_val with slope w/l, so one non-monotone step kills the linear story outright.
    {"v62_m1e3",  false, false, false, false, 1.f,     1.f,     false, 62, -1000.f, "A"},
    {"v62_m100",  false, false, false, false, 1.f,     1.f,     false, 62, -100.f,  "A"},
    {"v62_m10",   false, false, false, false, 1.f,     1.f,     false, 62, -10.f,   "A"},
    {"v62_m1",    false, false, false, false, 1.f,     1.f,     false, 62, -1.f,    "A"},
    {"v62_m01",   false, false, false, false, 1.f,     1.f,     false, 62, -0.0992f,"A"},
    {"v62_zero",  false, false, false, false, 1.f,     1.f,     false, 62, 0.f,     "A"},
    {"v62_p1",    false, false, false, false, 1.f,     1.f,     false, 62, 1.f,     "A"},
    {"v62_p10",   false, false, false, false, 1.f,     1.f,     false, 62, 10.f,    "A"},
    {"v62_p100",  false, false, false, false, 1.f,     1.f,     false, 62, 100.f,   "A"},
    {"v62_p1e3",  false, false, false, false, 1.f,     1.f,     false, 62, 1000.f,  "A"},
    {"v62_p1e6",  false, false, false, false, 1.f,     1.f,     false, 62, 1e6f,    "A"},
    // The signed-zero discriminator, and the cheapest one available: if the dead product is 0 x V (a SIGNED
    // ZERO, which is what an exactly-zero weight times a finite V is), the response is magnitude-free and
    // survives all the way down to V = -1e-8. If instead the dead weight is some tiny non-zero w, the
    // product shrinks with |V| and the flip must DIE somewhere in this ladder. f16 subnormals bottom out at
    // 6e-8, so -1e-8 also says whether the staged V flushes.
    {"v62_m1e2",  false, false, false, false, 1.f,     1.f,     false, 62, -1e-2f,  "A"},
    {"v62_m1e4",  false, false, false, false, 1.f,     1.f,     false, 62, -1e-4f,  "A"},
    {"v62_m1e6",  false, false, false, false, 1.f,     1.f,     false, 62, -1e-6f,  "A"},
    {"v62_m1e8",  false, false, false, false, 1.f,     1.f,     false, 62, -1e-8f,  "A"},
    {"v62_m1e12", false, false, false, false, 1.f,     1.f,     false, 62, -1e-12f, "A"},
    // The methodological control: run the whole mutation machinery writing the SAME bytes. If this moves
    // the adjudicated row, the rig's own write path is the variable and no value-null below means anything.
    {"vident",    false, false, false, false, 1.f,     1.f,     false, -1, 0.f,    "A"},
    // the must-fire controls: the same mutations with the bound widened by one, which makes lane jbad
    // legitimately live for the adjudicated row. Identical buffers, one integer different.
    {"widen_real", true,  true,  false, false, 1.f,     1.f,     true,  -1, 0.f,    "CTRL"},
    {"widen_kbig", false, false, false, false, 64.f,    1.f,     true,  -1, 0.f,    "CTRL"},
    {"widen_vbig", false, false, false, false, 1.f,     64.f,    true,  -1, 0.f,    "CTRL"},
};

// Write lane jbad's rows into the device cache for both KV groups. `ref` is the loaded tile ([lane][DH]
// per group, lanes+1 rows), `pert` may be null (then `*_real` falls back to the scale mutation and says so).
void apply_mutation(const Dev& d, const Dump& ref, const Dump* pert, int jbad, const Mutation& m) {
    for (int kv = 0; kv < N_KV; ++kv) {
        for (int which = 0; which < 2; ++which) {
            std::vector<uint16_t> row(DH);
            const std::vector<uint16_t>& src = which ? ref.vcache[kv] : ref.kcache[kv];
            const bool use_real = which ? m.v_real : m.k_real;
            const bool use_zero = which ? m.v_zero : m.k_zero;
            const float scale = which ? m.v_scale : m.k_scale;
            const std::vector<uint16_t>* from = &src;
            if (use_real && pert) {
                from = which ? &pert->vcache[kv] : &pert->kcache[kv];
            }
            for (int dd = 0; dd < DH; ++dd) {
                if (use_zero) {
                    row[dd] = 0;
                } else if (use_real && !pert) {
                    row[dd] = src[dd];   // no perturbed dump: byte-identical, and the CASE line says so
                } else {
                    row[dd] = use_real ? (*from)[(size_t)jbad * DH + dd]
                                       : bf16_scale(src[(size_t)jbad * DH + dd], scale);
                }
            }
            hip_bfloat16* dst = (which ? d.vcache : d.kcache) + ((size_t)kv * d.cap + jbad) * DH;
            if (which && m.v_dim >= 0) row[m.v_dim] = bf16_bits(m.v_dim_val);
            CK(hipMemcpy(dst, row.data(), sizeof(uint16_t) * DH, hipMemcpyHostToDevice));
        }
    }
}

bool write_tile(const std::string& outdir, const std::string& tag, hip_bfloat16* dev, size_t elems) {
    std::vector<uint16_t> h(elems);
    CK(hipMemcpy(h.data(), dev, sizeof(uint16_t) * elems, hipMemcpyDeviceToHost));
    const std::string p = outdir + "/" + tag + ".bin";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) { std::printf("FATAL cannot write %s\n", p.c_str()); return false; }
    const size_t w = std::fwrite(h.data(), 2, elems, f);
    std::fclose(f);
    return w == elems;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dump_dir, pert_dir, outdir = "/tmp/packab/rig";
    int seq = 1, layer = 3, adjud_row = 8, knob = 2, flag_lvalid = 1, lane_arg = -1;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::printf("missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--dump") dump_dir = next("--dump");
        else if (a == "--pert-dir") pert_dir = next("--pert-dir");
        else if (a == "--out") outdir = next("--out");
        else if (a == "--seq") seq = std::atoi(next("--seq"));
        else if (a == "--layer") layer = std::atoi(next("--layer"));
        else if (a == "--row") adjud_row = std::atoi(next("--row"));
        else if (a == "--knob") knob = std::atoi(next("--knob"));
        else if (a == "--lvalid") flag_lvalid = std::atoi(next("--lvalid"));
        else if (a == "--lane") lane_arg = std::atoi(next("--lane"));
        else if (a == "--list") list = true;
        else { std::printf("unknown arg %s\n", a.c_str()); return 2; }
    }
    if (list) {
        for (const auto& m : kMutations)
            std::printf("%-11s arm=%-4s k{real=%d,zero=%d,scale=%g} v{real=%d,zero=%d,scale=%g,dim=%d,val=%g}"
                        " widen=%d\n", m.name, m.arm, m.k_real, m.k_zero, m.k_scale, m.v_real, m.v_zero,
                        m.v_scale, m.v_dim, m.v_dim_val, m.widens_bound);
        return 0;
    }
    if (dump_dir.empty() || (knob != 1 && knob != 2)) {
        std::printf("usage: pack_ab --dump DUMPDIR [--pert-dir DIR] [--out DIR] [--seq 1] [--layer 3]"
                    " [--row 8] [--knob 1|2]\n");
        return 2;
    }
    std::printf("mkdir %s\n", outdir.c_str());
    { std::string mk = "mkdir -p " + outdir; if (std::system(mk.c_str()) != 0) return 2; }

    Dump ref;
    std::string why;
    if (!ref.load(dump_dir, seq, layer, &why)) {
        std::printf("SETUP FAILED: %s\n", why.c_str());
        return 2;
    }
    Dump pert;
    bool have_pert = false;
    if (!pert_dir.empty()) {
        pert.seq = seq; pert.layer = layer;
        if (!pert.load(pert_dir, seq, layer, &why)) {
            std::printf("SETUP FAILED (pert): %s\n", why.c_str());
            return 2;
        }
        have_pert = (ref.qrope != pert.qrope);   // a pert dump identical to ref is a vacuous arm, said below
        if (pert.rows != ref.rows || pert.lanes != ref.lanes) {
            std::printf("SETUP FAILED: pert dump is r%d_lanes%d, ref is r%d_lanes%d — two bodies cannot be"
                        " compared by mutating one lane\n", pert.rows, pert.lanes, ref.rows, ref.lanes);
            return 2;
        }
    }
    // The body's own `lvalid` comes from the ATTN_DUMP line of the run that wrote the dump (p128
    // --steps 0 prints lvalid=1, FIX-CROSSROW Result 0), so it is a FLAG here, not a guess: a dump from a
    // later body must be run with the --lvalid that run printed.
    const int body_lvalid = flag_lvalid;
    const int bound = body_lvalid + adjud_row;       // lanes this row may read: 0..bound-1
    // The varied lane defaults to the row's FIRST dead lane. `--lane` overrides it, which is the reach
    // question: the engine datum (FIX-CROSSROW Result 1: reach exactly one lane, never two) claims only
    // the first dead lane reaches. Adjudicating row 7 while varying lane 9 — its SECOND dead lane — is the
    // test, at the sensitivity this rig already has, and it is the datum's own configuration.
    const int jbad = lane_arg >= 0 ? lane_arg : bound;
    std::printf("RIG lane_kind=%s (row bound=%d, lane-bound=%d)\n",
                lane_arg >= 0 ? "--lane override" : "first dead lane",
                bound, jbad - bound);
    if (jbad > ref.lanes) {
        std::printf("SETUP FAILED: row %d bound %d exceeds the dumped lanes %d\n", adjud_row, jbad,
                    ref.lanes);
        return 2;
    }
    const int cap = ref.lanes + 2;                   // the engine's own cap for this body (ATTN_DUMP cap=130)
    std::printf("RIG seq=%d layer=%d rows=%d lanes=%d cap=%d knob=%d core=%s\n", seq, layer, ref.rows,
                ref.lanes, cap, knob, knob == 2 ? "attn_wide_llama_wave" : "attn_wide_wave");
    std::printf("RIG adjudicated_row=%d bound=%d (odd=%d) disallowed_lane=%d pert_dump=%s\n", adjud_row,
                bound, bound & 1, jbad, have_pert ? "loaded" : "ABSENT(real-* arms degrade to byte-identical)");
    // The straddling pair, stated so a reader can check the parity claim without re-deriving it.
    {
        const int L = jbad >> 4, col = jbad & 15;
        std::printf("RIG straddle: L=%d col=%d held by half=%d c=%d -> pair (%d,%d) %s\n", L, col, col & 1,
                    col >> 1, 16 * L + (col & ~1), 16 * L + (col | 1),
                    (col & 1) ? "STRADDLES (one live, one masked)" : "does not straddle (both halves agree)");
    }
    std::printf("RIG note: --lvalid %d is read off the ATTN_DUMP line of the run that wrote the dump"
                " (lvalid=1 for p128 --steps 0); a dump from another body needs its own value.\n",
                body_lvalid);

    Dev d;
    d.rows = ref.rows;
    d.cap = cap;
    upload(&d.qrope, ref.qrope);
    upload(&d.krope, ref.krope);
    upload_cache(&d.kcache, ref.kcache, cap);
    upload_cache(&d.vcache, ref.vcache, cap);
    // aout is NOT pre-filled with the engine's own answer: a buffer pre-filled with the value the writes
    // store proves nothing (AGENTS' mixed_bw row). Fill with a sentinel the writes cannot produce, so a row
    // the kernel never published reads -1024 and cannot be mistaken for a match.
    {
        CK(hipMalloc(reinterpret_cast<void**>(&d.aout), sizeof(uint16_t) * ref.pre_gate.size()));
        CK(hipMalloc(reinterpret_cast<void**>(&d.gout), sizeof(uint16_t) * ref.pre_gate.size()));
        std::vector<uint16_t> sent(ref.pre_gate.size(), bf16_bits(-1024.f));
        CK(hipMemcpy(d.aout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(d.gout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice));
    }
    zero_alloc(&d.qg, (size_t)ref.rows * ROW_Q);
    zero_alloc(&d.vg, (size_t)ref.rows * ROW_KV);
    zero_alloc(&d.kg, (size_t)ref.rows * ROW_KV);
    zero_alloc(&d.qn, (size_t)ref.rows * ROW_H);
    zero_alloc(&d.kn, (size_t)ref.rows * ROW_KV);

    // ---- 1. the fidelity canary: unmodified dumped inputs must reproduce the engine's own pre_gate ----
    {
        CoreArgs a = make_core(d, body_lvalid);
        if (!launch_form(knob, a, ref.rows)) return 2;
        if (!write_tile(outdir, "fid", d.aout, ref.pre_gate.size())) return 2;
        std::vector<uint16_t> got(ref.pre_gate.size());
        CK(hipMemcpy(got.data(), d.aout, sizeof(uint16_t) * got.size(), hipMemcpyDeviceToHost));
        size_t diff = 0, first = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != ref.pre_gate[i]) { ++diff; if (!diff) first = i; }
        }
        std::printf("CANARY fid: %s ndiff_elems=%zu of %zu first_elem=%zu (row %zu, head %zu, dim %zu)\n",
                    diff == 0 ? "BYTE-IDENTICAL to the engine's dumped pre_gate" : "DIFFERS", diff,
                    got.size(), first, first / ROW_H, (first / DH) % N_Q, first % DH);
        if (diff != 0) {
            std::printf("ABORT: the rig does not reproduce the engine's own output, so no null below means"
                        " anything. Fix the CoreArgs/geometry before reading a CASE line.\n");
            return 3;
        }
    }

    // ---- 2. the arms -----------------------------------------------------------------------------
    int fired = 0, controls_red = 0;
    for (const auto& m : kMutations) {
        const bool degraded_real = (m.k_real || m.v_real) && !have_pert;
        // Run every case twice, back to back, in one process: determinism is a property of the arithmetic
        // path (FIX-CROSSROW Result 5 point 3), and a nondeterministic result changes what a null means.
        std::vector<uint16_t> two[2];
        bool ok = true;
        for (int rep = 0; rep < 2 && ok; ++rep) {
            // Reload the pristine caches every rep: the mutation is applied on top, so a case can never
            // inherit a previous case's bytes (which would make a null a fact about buffer order).
            reload_cache(d.kcache, ref.kcache, cap);
            reload_cache(d.vcache, ref.vcache, cap);
            apply_mutation(d, ref, have_pert ? &pert : nullptr, jbad, m);
            CoreArgs a = make_core(d, m.widens_bound ? body_lvalid + 1 : body_lvalid);
            ok = launch_form(knob, a, ref.rows);
            if (!ok) break;
            two[rep].resize(ref.pre_gate.size());
            CK(hipMemcpy(two[rep].data(), d.aout, sizeof(uint16_t) * two[rep].size(),
                         hipMemcpyDeviceToHost));
        }
        if (!ok) return 2;
        const bool deterministic = two[0] == two[1];
        const std::string tag = std::string(m.name) + "_L" + std::to_string(layer);
        if (!write_tile(outdir, tag, d.aout, ref.pre_gate.size())) return 2;

        // Byte-level reach counts, host side and per row: which rows moved, and in which half of the row
        // index space. The ULP sizing is NOT done here — `pack_ab_report.py` does it with bf16_ulp imported.
        int moved_below = 0, moved_at = 0, moved_above = 0, first_below = -1;
        size_t ndiff = 0;
        for (int r = 0; r < ref.rows; ++r) {
            size_t rd = 0;
            for (size_t i = (size_t)r * ROW_H; i < (size_t)(r + 1) * ROW_H; ++i) {
                if (two[0][i] != ref.pre_gate[i]) { ++rd; ++ndiff; }
            }
            if (rd == 0) continue;
            if (r < adjud_row) { ++moved_below; if (first_below < 0) first_below = r; }
            else if (r == adjud_row) moved_at = 1;
            else ++moved_above;
        }
        const bool control = std::strcmp(m.arm, "CTRL") == 0;
        const bool fired_here = moved_below > 0;
        if (control) {
            if (fired_here) ++controls_red;
            std::printf("CONTROL %s arm=%s bound_widened=1 rows_below_adjudicated=%d first_below=%d"
                        " ndiff=%zu -> %s, as it %s be\n", m.name, m.arm, moved_below, first_below, ndiff,
                        fired_here ? "Fired" : "DID NOT FIRE", fired_here ? "must" : "MUST");
        } else {
            if (fired_here) ++fired;
            std::printf("CASE %s arm=%s rows_below=%d first_below=%d row_at_moved=%d rows_above=%d"
                        " ndiff=%zu determinism=%s%s leak=%s\n", m.name, m.arm, moved_below, first_below,
                        moved_at, moved_above, ndiff, deterministic ? "identical" : "NONDETERMINISTIC",
                        degraded_real ? " real-data-degraded(no --pert-dir)" : "",
                        fired_here ? "MOVES the live row" : "null");
        }
        if (!deterministic) std::printf("  NOTE determinism: two launches of identical inputs differ\n");
    }
    std::printf("SUMMARY knob=%d cases=%zu leak_cases=%d controls_fired=%d/%d pert=%d\n", knob,
                sizeof(kMutations) / sizeof(kMutations[0]), fired, controls_red, 3, have_pert ? 1 : 0);
    if (controls_red < 3) {
        std::printf("SUMMARY: a MUST-FIRE control did not fire. Nothing in this run adjudicates anything;"
                    " the mutation did not reach the kernel or the geometry is not the engine's.\n");
        return 1;
    }
    std::printf("RESULT files in %s; size and ULP sizing: prototype/mtp_verify/pack_ab/report.py\n",
                outdir.c_str());
    return 0;
}
