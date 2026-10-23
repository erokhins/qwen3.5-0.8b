/*
 * causal_bound.cpp — which cache lanes did each query row ACTUALLY attend? A cell-resolution causality
 * probe for the engine's own attention cores, with no reference implementation and no tolerance.
 *
 * WHY THIS EXISTS. `prototype/mtp_verify/CROSSROW.md` measured that the wide attention block lets row `q`
 * attend lane `lanes_of(a,q)` — one lane PAST its own causal bound — whenever that bound is odd, and that
 * the defect hides inside every band in force (worst 1−cos 2.3e-4, ~9x inside CASCADE_BAND). Reading the
 * mask finds nothing: `engine/attn_wide_kernels.h:257-262` computes
 * `j = base + FATTN_MLANES*L + fattn_c_col(lane,c)` and masks on `j < rlanes`, per cell, per row, and that
 * text is correct. So a dump of `acc[L][c]` after that mask would be read through the expression under
 * suspicion, and a CPU mirror of the mask would agree with itself (AGENTS: "a check whose reference side
 * is derived from the thing under test cannot adjudicate it"). This probe adjudicates the mask's EFFECT
 * end to end instead: it asks the running kernel, per (row, lane) pair, "did you attend this lane", and
 * reads the answer out of the shipped epilogue's own output bits.
 *
 * HOW THE ANSWER IS READ OUT, and why it needs no band. Feed Q = 0 and K = 0. Every score is then exactly
 * +0.0, so after the causal mask a LIVE cell holds 0.0 and a MASKED cell holds -1e30f; the round's
 * exponentials are exactly 1.0 and exactly 0.0 (`expf(0-0) = 1`, `expf(-1e30f - 0) = 0`); the roundsum `l`
 * is exactly the integer count of live cells; the F16 P entries are exactly 1 and 0; and every V entry this
 * probe writes is exactly 0 or 1. The F16 accumulator therefore holds a small integer sum of exact 0/1
 * addends (<= 4032 addends of 1, and the rescale factor is exactly 1.0 after the first round), and
 *      aout[row][head][dim] = bf(fb( Σ_{j attended} V[j][dim] / |attended| )),
 * so      aout[row][head][dim] == 0   EXACTLY   <=>  this row contributed nothing through this dim.
 * Zero versus non-zero is not a threshold: there is nothing to widen, no band to tune, and bf16 output
 * resolution is irrelevant — which is why one probe reads the same way at 2 lanes and at 4,032.
 *
 * THE ENCODINGS — one question per output dim, DH = 256 questions per launch, `t` = this launch's base:
 *   indicator  V[j][d] = (j == t + d ? 1 : 0)   ->  aout[r][h][d] != 0  <=>  row r attended lane t+d.
 *              Swept over t = 0, 256, 512 ... this returns the EXACT attended set of every row.
 *   threshold  V[j][d] = (j >= t + d ? 1 : 0)   ->  aout[r][h][d] != 0  <=>  row r attended SOME lane >= t+d.
 *              Only the windows that bracket each row's own bound are launched (see `threshold_offsets`),
 *              because the assertion is about the bound, not about the whole cache.
 *
 * THE ASSERTION is §06.3's causal law: row `q` of a body whose first row may read `lvalid` lanes may read
 * lanes `0 .. causal_bound(lvalid,q) - 1`, and `causal_bound = lvalid + q`. The probe states that law
 * ITSELF (`causal_bound` below) rather than calling the engine's `lanes_of`, on purpose: `lanes_of` is the
 * function under suspicion — it is what carries the `NEG_READPAST`/`NEG_DROPCUR` injections, and an oracle
 * that called it could not fail. The engine's injections then form the derived MUST-CATCH set:
 *   dropcur   NEG_DROPCUR: the bound is one lane SHORT of causal      -> every row reads short
 *   readpast  NEG_READPAST: the causal mask is gone (bound = cap)     -> every row reads to the end
 *   widen1    the caller states bound B, the kernel is handed B+1 — the CROSSROW shape, one-lane wide,
 *             injected from the host so the probe needs no new seam in `engine/`
 * every one of them MUST turn the probe red, and `--must-catch` asserts that they do and prints
 * `CONTROL DID NOT FIRE` if one does not. The MUST-NOT-CATCH side is (a) the clean form — `attn_slim_wave`,
 * the core of `--attn-fattn-wide 0` — and (b) the parity split CROSSROW measured at engine level: pre-fix,
 * only odd-bound rows violate and only in the wide forms, which is what the `oddviol=`/`evenviol=` fields
 * are for. A control that cannot fail is called out in the same breath as the verdict.
 *
 * WHAT IT LAUNCHES. The engine's own `__global__` kernels, included read-only, on the grids
 * `engine/kernels_attn.cpp` builds: `attn_wide_llama_wave` (`--attn-fattn-wide 2` — THE SHIPPING DEFAULT,
 * `engine/decode_run.cpp:262`), `attn_wide_wave` (`--attn-fattn-wide 1`), and `attn_slim_wave` (the
 * `--attn-fattn-wide 0` core, the clean control; refused above MAXSC, where that route needs `part` plus
 * `attn_combine`, which this probe deliberately does not build — `crossrow_ladder.sh` is what covers
 * `--attn-fattn-wide 0` at streaming widths, on the engine, end to end). Nothing in `engine/` is edited.
 *
 * NO TIMING. Correctness only: no ms, no burst, no best-of-N, no warm loop. The only seconds printed are
 * whole-command wall, which AGENTS.md's device-run protocol requires.
 *
 * Build:  bash prototype/mtp_verify/build_causal_bound.sh [outdir]
 * Run:    <outdir>/causal_bound                      every form x case x encoding, clean path
 *         <outdir>/causal_bound --inject widen1     one injection
 *         <outdir>/causal_bound --must-catch        all three injections, and fail if one is not red
 *         <outdir>/causal_bound --form wide_llama --enc threshold
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// The two wide kernels live in `engine::kernels` (they are the engine's own `__global__`s, compiled here
// read-only so the probe cannot be accused of launching a paraphrase); `attn_slim_kernels.h` declares its
// kernel at global scope, the way that file was written.
using engine::kernels::attn_wide_llama_wave;
using engine::kernels::attn_wide_wave;

// ---- the oracle: §06.3's causal law, stated by the probe, NOT read from `lanes_of` -----------------
// Row `q` of a body whose first row may read `lvalid` cache lanes may read lanes 0..lvalid+q-1. Spelled
// here and nowhere else on the readout side, so that an injection or a widened mask shows up as a
// disagreement instead of as two copies of one function agreeing.
int causal_bound(int lvalid, int q) { return lvalid + q; }

// What the probe writes into `aout` before every launch. A dim the kernel never published would otherwise
// read 0 and be reported as "attended nothing" — §18.1's coverage failure wearing a green badge, so
// coverage is checked as a fact and not a reading. Every legitimate value is a mean of 0/1, i.e. in
// [0,1], and 1024.0f round-trips through BF16 exactly (a power of two), so the sentinel cannot be mimicked.
constexpr float kSentinel = 1024.0f;

// BF16 <-> bits in plain C++. This ROCm's `hip_bfloat16.h` exposes no host `__float2bfloat16`, and the
// probe needs to *build* its own inputs, so the round lives here. Round-to-nearest-even on the top 16
// bits; the probe only ever encodes 0.0, 1.0 and one power of two, every one of which is exact in BF16,
// so no rounding path is exercised by an input, and no NaN is ever written.
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

constexpr int kWindow = DH;          // one question per output dim: 256 thresholds per launch
constexpr int kBodyRows = 512;       // engine/decode_step.h:49 kMaxWaveRows — the engine's own body cap

// The slack the probe keeps past the last bound: with NEG_READPAST the kernel's bound becomes `cap`, and
// `cap` IS the cache stride, so every lane it can reach is a lane this probe allocated and filled.
constexpr int kCapSlack = 64;

enum class Form { kWideLlama, kWide, kSlim };
struct FormInfo {
    Form form;
    const char* name;
    const char* flag;
    bool shipping_default;
};
const FormInfo kForms[] = {
    {Form::kWideLlama, "wide_llama", "--attn-fattn-wide 2", true},
    {Form::kWide, "wide", "--attn-fattn-wide 1", false},
    {Form::kSlim, "slim", "--attn-fattn-wide 0 (attn_slim_wave)", false},
};
const char* form_name(Form f) {
    for (const auto& i : kForms) if (i.form == f) return i.name;
    return "?";
}

enum class Injection { kNone, kDropCur, kReadPast, kWiden1 };
struct InjectionInfo {
    Injection inj;
    const char* name;
    const char* what;
    bool must_be_red;
};
const InjectionInfo kInjections[] = {
    {Injection::kNone, "none", "the shipping path, no injection", false},
    {Injection::kDropCur, "dropcur", "NEG_DROPCUR: the bound is one lane SHORT of causal", true},
    {Injection::kReadPast, "readpast", "NEG_READPAST: the causal mask is gone, bound = cap", true},
    {Injection::kWiden1, "widen1", "the kernel is handed bound+1 while the caller states bound", true},
};
const InjectionInfo* find_injection(const char* name) {
    for (const auto& i : kInjections) if (std::strcmp(i.name, name) == 0) return &i;
    return nullptr;
}

// Set by `launch_form` to the function pointer the last launch submitted, and printed on every CASE line.
// The probe's own canary; see `ATTN_DUMP` in `engine/kernels_attn.cpp` for the engine's side of the pair.
const char* g_launched_kernel = "(nothing launched)";

enum class Enc { kIndicator, kThreshold, kPoison, kIdentity, kBoundary };
const char* enc_name(Enc e) {
    if (e == Enc::kIndicator) return "indicator";
    if (e == Enc::kThreshold) return "threshold";
    if (e == Enc::kPoison) return "poison";
    return e == Enc::kIdentity ? "identity" : "boundary";
}

// The poison and identity encodings ask one lane / one row at a time, so they are bounded to a small
// geometry. The identity readout also needs the dim index to fit in DH.
constexpr int kPoisonMaxLanes = 64;

// --dump "<form>,<nrows>,<lvalid>,<row>[,<limit>]": print the raw non-zero output cells of that one case.
bool g_dump = false;
std::string g_dump_form;
int g_dump_rows = 0, g_dump_lvalid = 0, g_dump_row = 0, g_dump_left = 60;

// ---- device buffers, grown on demand -----------------------------------------------------------------
struct Buffers {
    hip_bfloat16 *qrope = nullptr, *krope = nullptr, *kcache = nullptr, *vcache = nullptr;
    hip_bfloat16 *vg = nullptr, *qg = nullptr, *aout = nullptr, *gout = nullptr;
    int rows_cap = 0, lanes_cap = 0;

    void ensure(int rows, int lanes) {
        if (rows <= rows_cap && lanes <= lanes_cap) return;
        const int r = std::max(rows, rows_cap), c = std::max(lanes, lanes_cap);
        auto redo = [&](hip_bfloat16** p, size_t elems) {
            if (*p) hipFree(*p);
            *p = nullptr;
            const hipError_t e = hipMalloc(reinterpret_cast<void**>(p), sizeof(uint16_t) * elems);
            if (e != hipSuccess) { printf("FATAL hipMalloc: %s\n", hipGetErrorString(e)); std::exit(2); }
        };
        redo(&qrope, (size_t)ROW_H * r);
        redo(&krope, (size_t)ROW_KV * r);
        redo(&vg, (size_t)ROW_KV * r);
        redo(&qg, (size_t)ROW_Q * r);
        redo(&aout, (size_t)ROW_H * r);
        redo(&gout, (size_t)ROW_H * r);
        redo(&kcache, (size_t)N_KV * c * DH);
        redo(&vcache, (size_t)N_KV * c * DH);
        rows_cap = r;
        lanes_cap = c;
        // `finish_head` reads qg for the gate; a zero gate is a harmless 0.5 multiplier on gout, which
        // this probe does not read. Everything else is filled per launch.
        std::vector<uint16_t> zeros((size_t)ROW_Q * r, bf16_bits(0.f));
        if (hipMemcpy(qg, zeros.data(), sizeof(uint16_t) * zeros.size(), hipMemcpyHostToDevice)
            != hipSuccess) { printf("FATAL hipMemcpy qg\n"); std::exit(2); }
    }
};

// Threshold bases to launch for one case.
//   threshold mode: the assertion per row is `max attended == bound-1`, which needs exactly two answers
//     per row — threshold bound-1 must say yes and threshold bound must say no. All bounds live in
//     [lvalid, lvalid+nrows-1], so the bases walk that span, DH thresholds at a time: 1 launch for a body
//     of <= 254 rows, 3 for a 512-row body, at any context length.
//   indicator mode: the exact attended set of every row needs every lane below its bound asked about, so
//     the bases walk from 0; that is O(bound/DH) launches, which is why the exact-set mode is only run on
//     the small geometries (see `case_wants_indicator`).
std::vector<int> threshold_offsets(int lvalid, int nrows) {
    std::vector<int> off;
    const int lo = std::max(0, lvalid - 1);
    const int hi = lvalid + nrows - 1;
    for (int t = lo; t <= hi; t += kWindow) off.push_back(t);
    if (off.empty()) off.push_back(0);
    return off;
}

std::vector<int> indicator_offsets(int lvalid, int nrows) {
    std::vector<int> off;
    const int hi = lvalid + nrows - 1;                 // highest lane any row may attend
    for (int t = 0; t <= hi; t += kWindow) off.push_back(t);
    if (off.empty()) off.push_back(0);
    return off;
}

bool case_wants_indicator(Enc enc, int lvalid, int nrows) {
    if (enc == Enc::kPoison || enc == Enc::kIdentity || enc == Enc::kBoundary) {
        return lvalid + nrows <= kPoisonMaxLanes;
    }
    if (enc != Enc::kIndicator) return true;
    return lvalid + nrows - 1 <= 4 * kWindow;          // keep the exact-set sweep cheap
}

// V for one (encoding, base). Layout is the CACHE's own: [kv][lane][dim], because that is how every core
// indexes it — `a.vcache + (kv_read * a.cap + j) * DH` — and getting it wrong is not an error the kernel
// reports: it silently answers a different question per KV group (the first version of this probe wrote
// [lane][kv][dim] and read back "heads 4..7 of row 0 attended lane 33", which was the probe's layout bug,
// not the engine's).
void build_v(std::vector<uint16_t>& v, int lanes, Enc enc, int base) {
    for (int kv = 0; kv < N_KV; ++kv) {
        for (int j = 0; j < lanes; ++j) {
            uint16_t* p = &v[((size_t)kv * lanes + j) * DH];
            for (int d = 0; d < DH; ++d) {
                const int t = base + d;
                const bool one = (enc == Enc::kIndicator) ? (j == t) : (j >= t);
                p[d] = bf16_bits(one ? 1.f : 0.f);
            }
        }
    }
}

// One launch of one form on the launcher's own grid. Returns false if the launch faulted.
bool launch_form(Form form, const CoreArgs& core, int rows, int lanes_last, std::string* refusal) {
    // The canary on the probe's own side: which function pointer this launch submitted, and the grid it
    // went out on. Printed on every CASE line, because `form_ladder.sh` exists for the reason AGENTS' 2026
    // 10-17 row gives — an arm that silently is not the arm it is named for reads exactly like a null.
    extern const char* g_launched_kernel;
    if (form == Form::kSlim && lanes_last > MAXSC) {
        if (refusal) *refusal = std::string("lanes_last ") + std::to_string(lanes_last)
                             + " > MAXSC " + std::to_string(MAXSC)
                             + ": this route needs part + attn_combine, which the probe does not build";
        return false;
    }
    if (form == Form::kSlim) {
        g_launched_kernel = "attn_slim_wave";
        attn_slim_wave<<<dim3(N_KV, 1, tile_rtiles_of(rows)), ATTN_TILE_THREADS>>>(core);
    } else if (form == Form::kWide) {
        g_launched_kernel = "attn_wide_wave";
        attn_wide_wave<<<dim3(N_KV, 1, engine::kernels::wide_rtiles_of(rows)), WIDE_THREADS>>>(core);
    } else {
        g_launched_kernel = "attn_wide_llama_wave";
        attn_wide_llama_wave<<<dim3(N_KV, 1, engine::kernels::wide_rtiles_of(rows)), WIDE_THREADS>>>(core);
    }
    const hipError_t le = hipGetLastError();
    const hipError_t se = hipDeviceSynchronize();
    if (le != hipSuccess || se != hipSuccess) {
        printf("    LAUNCH FAILED form=%s: last=%s sync=%s\n", form_name(form),
               hipGetErrorString(le), hipGetErrorString(se));
        return false;
    }
    return true;
}

struct Case {
    int nrows;
    int lvalid;
    std::string why;
};

// The case list, DERIVED from the geometry rather than hand-picked (the repo's standing rule for a
// must-catch population, and CROSSROW's own blind-spot finding: a boundary that coincides with the
// geometry's own boundary cannot show a reach-1 leak, so every quantum gets its straddle).
//   WIDE_POS_PER_BLOCK = 32 query positions per block, WIDE_STAGE = 64 positions per round,
//   WIDE_POS_PER_WARP = 4 positions per wave, MAXSC = 512 one-pass ceiling, kBodyRows = 512 body cap.
std::vector<Case> derive_cases() {
    const int POS = WIDE_POS_PER_BLOCK;
    const int ST = WIDE_STAGE;
    char b[160];
    std::vector<Case> c;
    auto add = [&](int nrows, int lvalid, const std::string& why) { c.push_back(Case{nrows, lvalid, why}); };
    add(1, 1, "one row: the sole-bound case");
    add(2, 1, "the two-row body (the MTP verify pair)");
    add(WIDE_POS_PER_WARP, 1, "one wave's positions");
    for (int d = -1; d <= 1; ++d) {
        snprintf(b, sizeof b, "straddles a %d-position BLOCK boundary (%+d)", POS, d);
        add(POS + d, 1, b);
    }
    for (int d = -1; d <= 1; ++d) {
        snprintf(b, sizeof b, "straddles a %d-position ROUND boundary (%+d)", ST, d);
        add(ST + d, 1, b);
    }
    add(2 * ST + 2, 1, "a full round, then a partial one");
    add(4 * POS, 1, "128 rows: the p128 fixture width, four full blocks");
    add(4 * POS + 1, 1, "129 rows: one row past the block tail");
    add(MAXSC, 1, "512 rows: the whole one-pass ceiling in one body");
    // The engine chunks a prefill into bodies of at most kBodyRows rows, so these are the bodies a real
    // context actually issues, at the widths that matter: 640 lanes (CROSSROW's streaming reproduction)
    // and 4,032 lanes (the decode headline width).
    const int contexts[] = {MAXSC + 128, 4032};
    for (int ctx : contexts) {
        int lvalid = 1, left = ctx;
        while (left > 0) {
            const int nrows = std::min(left, kBodyRows);
            snprintf(b, sizeof b, "body of a %d-lane context: rows %d..%d%s", ctx, lvalid,
                     lvalid + nrows - 1, lvalid > MAXSC ? ", past MAXSC (streaming region)" : "");
            add(nrows, lvalid, b);
            lvalid += nrows;
            left -= nrows;
        }
    }
    return c;
}

struct Report {
    bool ran = false;
    bool refused = false;
    std::string why;
    int launches = 0;
    int past_bound_rows = 0;         // attended a lane >= its bound: THE causality failure
    int short_rows = 0;              // did not reach its bound: the narrowing failure
    int poison_wrong = 0;            // poison read by a cell that should not have read it
    int identity_wrong = 0;          // a row's scores came from another row's query
    int gate_wrong = 0;              // a row's gate came from another row's `qg`
    int wrong_l_rows = 0;            // the rowsum `l` is not the bound: a denominator-only leak
    int l_checked = 0;               // how many (row,head) readings the denominator check adjudicated
    int not_written = 0;             // cells the kernel never published (coverage)
    const char* kernel = "(none)";   // which kernel this case's launches went to (`g_launched_kernel`)
    int nonfinite = 0;
    int oddviol = 0, evenviol = 0;   // violations split by the parity of the row's bound
    int worst_row = -1, worst_observed = -1, worst_bound = -1;
    std::vector<std::string> findings;
    bool red() const {
        return past_bound_rows + short_rows + not_written + nonfinite + poison_wrong + wrong_l_rows
               + identity_wrong + gate_wrong > 0;
    }
};

void note(Report& r, const std::string& s) {
    if (r.findings.size() < 6) r.findings.push_back(s);
}

// ---- the poison encoding: WHICH CELL READ THIS LANE'S KEY? ------------------------------------------
// The two causal encodings ask the kernel what it attended; they read the mask's effect through the P*V
// product, and they are content-independent because Q = K = 0 makes every score identical. That is a real
// limit: a kernel whose mask is right but whose *operand* for one in-bound cell is fetched from the wrong
// cache lane attends no extra lane and still depends on a token past its bound — which is the other shape
// CROSSROW's datum could have, since CROSSROW perturbed tokens and read logits, not cells.
// So: make exactly one cache lane `j*` legible. Q's dim 0 is 16 everywhere, K[j*]'s dim 0 is 1 and every
// other K row is 0, so the engine's own round (`fattn_score_round` = bf(fb(fb(dot)·SCALE)), SCALE = 2^-4)
// gives the cell that read lane j* a score of exactly +1.0 and every other cell exactly 0.0. With V still
// the indicator, out[r][h][d] = (weight of the cell whose slot is d) / l, so:
//   * no cell read lane j*      -> every attended cell has the same weight -> the row's output is FLAT;
//   * some cell d* read lane j* -> that cell's weight is e times the others -> the peak IS d*, e.g.
//     measurable as out[d*]/out[other] ~ 2.718 and nothing else about it.
// The assertion per (row, j*): a read may happen only for j* < bound(row), and then only by cell j*.
// A read for j* >= bound is CROSSROW's leak, whichever stage caused it; a read by a cell other than j* is
// an operand fetched from the wrong lane; no read for an in-bound j* is a missing operand. All three print.
Report run_poison(Form form, Injection inj, const Case& cs, Buffers& buf) {
    Report rep;
    const int lanes_last = cs.lvalid + cs.nrows - 1;
    const int cap = lanes_last + kCapSlack;
    if (form == Form::kSlim && lanes_last > MAXSC) {
        rep.refused = true;
        rep.why = "lanes_last " + std::to_string(lanes_last) + " > MAXSC (part + attn_combine not built here)";
        return rep;
    }
    if (!case_wants_indicator(Enc::kPoison, cs.lvalid, cs.nrows)) {
        rep.refused = true;
        rep.why = std::string("poison sweep needs one launch per lane up to ") + std::to_string(lanes_last)
                  + ", over the " + std::to_string(kPoisonMaxLanes) + "-lane budget";
        return rep;
    }
    buf.ensure(cs.nrows, cap);

    const int bound_given_to_kernel = (inj == Injection::kWiden1) ? cs.lvalid + 1 : cs.lvalid;
    CoreArgs core{};
    core.qg = buf.qg; core.kcache = buf.kcache; core.vcache = buf.vcache; core.vg = buf.vg;
    core.qrope = buf.qrope; core.krope = buf.krope; core.aout = buf.aout; core.gout = buf.gout;
    core.kc_as = buf.kcache; core.vc_as = buf.vcache;
    core.cap = cap; core.pos = cs.lvalid - 1; core.lvalid = bound_given_to_kernel; core.nrows = cs.nrows;
    core.neg = (inj == Injection::kDropCur) ? NEG_DROPCUR
             : (inj == Injection::kReadPast) ? NEG_READPAST : NEG_NONE;
    core.full = 1; core.split = 1; core.fused = 0; core.score_f32 = 0; core.tile = TILE;
    core.tile_kernel = 1; core.fattn_cover = 0;

    // Q: dim 0 = 16 for every (row, head); everything else 0.
    std::vector<uint16_t> qrow((size_t)ROW_H * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r)
        for (int h = 0; h < N_Q; ++h) qrow[(size_t)ROW_H * r + (size_t)h * DH] = bf16_bits(16.f);
    hipMemcpy(buf.qrope, qrow.data(), sizeof(uint16_t) * qrow.size(), hipMemcpyHostToDevice);

    // V: the indicator at base 0, so output dim d reads out the weight of the cell whose slot is d.
    std::vector<uint16_t> vh((size_t)cap * N_KV * DH, bf16_bits(0.f));
    build_v(vh, cap, Enc::kIndicator, 0);
    hipMemcpy(buf.vcache, vh.data(), sizeof(uint16_t) * vh.size(), hipMemcpyHostToDevice);
    // The non-wide core reads its own new lane's V from `vg`; give it the same indicator rows.
    std::vector<uint16_t> staged((size_t)ROW_KV * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r) {
        const int j = core.pos + r;
        for (int kv = 0; kv < N_KV; ++kv) {
            uint16_t* p = &staged[(size_t)r * ROW_KV + (size_t)kv * DH];
            for (int d = 0; d < DH; ++d) p[d] = bf16_bits(j == d ? 1.f : 0.f);
        }
    }
    hipMemcpy(buf.vg, staged.data(), sizeof(uint16_t) * staged.size(), hipMemcpyHostToDevice);

    std::vector<uint16_t> kh((size_t)N_KV * cap * DH, bf16_bits(0.f));
    std::vector<uint16_t> khrow((size_t)ROW_KV * cs.nrows, bf16_bits(0.f));
    std::vector<uint16_t> sent((size_t)ROW_H * cs.nrows, bf16_bits(kSentinel));
    std::vector<uint16_t> host_aout((size_t)ROW_H * cs.nrows);
    const uint16_t sent_bits = bf16_bits(kSentinel);
    const int top_lane = std::min(cap - 1, lanes_last + 1);   // probe one lane past the last bound too

    for (int jstar = 0; jstar <= top_lane; ++jstar) {
        for (auto& x : kh) x = bf16_bits(0.f);
        for (int kv = 0; kv < N_KV; ++kv) kh[((size_t)kv * cap + jstar) * DH] = bf16_bits(1.f);
        for (auto& x : khrow) x = bf16_bits(0.f);
        for (int r = 0; r < cs.nrows; ++r) {
            if (core.pos + r != jstar) continue;               // the slim form's own-lane K, same lane law
            for (int kv = 0; kv < N_KV; ++kv) khrow[(size_t)r * ROW_KV + (size_t)kv * DH] = bf16_bits(1.f);
        }
        hipMemcpy(buf.kcache, kh.data(), sizeof(uint16_t) * kh.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.krope, khrow.data(), sizeof(uint16_t) * khrow.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.aout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.gout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);

        std::string refusal;
        if (!launch_form(form, core, cs.nrows, lanes_last, &refusal)) {
            rep.why = refusal.empty() ? "launch failed" : refusal;
            return rep;
        }
        rep.ran = true;
        rep.launches++;
        hipMemcpy(host_aout.data(), buf.aout, sizeof(uint16_t) * host_aout.size(), hipMemcpyDeviceToHost);

        for (int r = 0; r < cs.nrows; ++r) {
            const int bound = causal_bound(cs.lvalid, r);
            for (int h = 0; h < N_Q; ++h) {
                const uint16_t* rowp = &host_aout[(size_t)ROW_H * r + (size_t)h * DH];
                float mx = -1.f, mn = 1e30f;
                int dstar = -1, nz = 0, sent_cells = 0;
                for (int d = 0; d < DH; ++d) {
                    const uint16_t bits = rowp[d];
                    if (bits == sent_bits) { rep.not_written++; continue; }
                    const float x = bf16_val(bits);
                    if (!std::isfinite(x)) { rep.nonfinite++; continue; }
                    if (x <= 0.f) continue;
                    ++nz;
                    if (x > mx) { mx = x; dstar = d; }
                    if (x < mn) mn = x;
                }
                (void)sent_cells;
                if (nz == 0) { rep.not_written++; continue; }
                const bool flat = (mn * 1.5f > mx);            // every attended cell equally weighted
                const bool read_it = !flat;
                if (jstar >= bound) {
                    if (read_it) {
                        rep.past_bound_rows++; rep.poison_wrong++;
                        char b[224];
                        snprintf(b, sizeof b, "row %d (bound %d): cell %d read the KEY of lane %d, weight %.4fx the others",
                                 r, bound, dstar, jstar, mx / std::max(mn, 1e-30f));
                        note(rep, b);
                    }
                } else {
                    if (nz == 1) continue;    // a row with one attended cell cannot show a read: undecidable
                    if (!read_it) {
                        rep.short_rows++;
                        char b[192];
                        snprintf(b, sizeof b, "row %d (bound %d): in-bound lane %d was read by NO cell", r, bound, jstar);
                        note(rep, b);
                    } else if (dstar != jstar) {
                        rep.poison_wrong++;
                        char b[224];
                        snprintf(b, sizeof b, "row %d (bound %d): lane %d's key was read by cell %d, not by cell %d",
                                 r, bound, jstar, dstar, jstar);
                        note(rep, b);
                    }
                }
            }
        }
    }
    if (!rep.ran) { rep.refused = true; if (rep.why.empty()) rep.why = "no launch ran"; }
    return rep;
}

// ---- the identity encoding: WHICH ROW'S QUERY SCORED THIS ROW? -----------------------------------------
// Everything so far is row-invariant: Q is identical for every row (all zero, or one shared hot dim), so a
// kernel that scores row r with row r''s QUERY — a fragment map that pairs a C row with the wrong A row, an
// off-by-one in a staged Q row index, a Q tile loaded for the neighbouring row — answers all three exactly
// as it should. That candidate also fits CROSSROW's datum better than a mask leak does: "row q moves when
// token q+1 changes, row 0 immune, even q immune" is what a row-to-row operand mixup looks like from
// outside, because the quantity that leaks is the next row's query, not the next lane's K or V. And softmax
// does not absorb it: a row-CONSTANT score shift is eaten by the running max, so the fingerprint must be
// lane-VARYING, which is what the design below is.
//   Q[r][h][d] = 16 at d == r;  K[j][d] = 16 at d == j;  V = the indicator at base 0.
// The dot for cell (row r, lane j) is 256 exactly when j == r and 0 otherwise, and the engine's own round
// (`fattn_score_round` = bf(fb(fb(dot)·SCALE)), SCALE = 2^-4) turns those into scores of exactly +16 and
// exactly 0. Lane r is always live — bound = lvalid + r > r for lvalid >= 1 — so each row's softmax has one
// dominant term: out[r][h][r] = 1/bound and every other dim is exp(-16)/bound or flushed to zero. So THE
// ARGMAX DIM OF A ROW IS THE QUERY ROW THAT WAS USED, with a 10^7 margin and nothing fitted. The rowsum is
// readable from the same number (out = 1/l), which re-checks `l` under non-flat content.
// The gate rides along in the same launches: `qg` is +8 for one hot row and -8 for every other row, so
// gout/aout = bf(sigmoid(g)) reads out WHICH ROW'S GATE reached that row (0.9997 against 0.0003, again
// nothing fitted). `finish_head` is the only place either form reads `qg`, and it indexes it by `qrow`.
Report run_identity(Form form, Injection inj, const Case& cs, Buffers& buf) {
    Report rep;
    const int lanes_last = cs.lvalid + cs.nrows - 1;
    const int cap = lanes_last + kCapSlack;
    if (form == Form::kSlim && lanes_last > MAXSC) {
        rep.refused = true;
        rep.why = "lanes_last " + std::to_string(lanes_last) + " > MAXSC (part + attn_combine not built here)";
        return rep;
    }
    if (!case_wants_indicator(Enc::kIdentity, cs.lvalid, cs.nrows)) {
        rep.refused = true;
        rep.why = "identity sweep needs one launch per row up to " + std::to_string(cs.nrows)
                  + ", over the " + std::to_string(kPoisonMaxLanes) + "-row budget";
        return rep;
    }
    buf.ensure(cs.nrows, cap);

    const int bound_given_to_kernel = (inj == Injection::kWiden1) ? cs.lvalid + 1 : cs.lvalid;
    CoreArgs core{};
    core.qg = buf.qg; core.kcache = buf.kcache; core.vcache = buf.vcache; core.vg = buf.vg;
    core.qrope = buf.qrope; core.krope = buf.krope; core.aout = buf.aout; core.gout = buf.gout;
    core.kc_as = buf.kcache; core.vc_as = buf.vcache;
    core.cap = cap; core.pos = cs.lvalid - 1; core.lvalid = bound_given_to_kernel; core.nrows = cs.nrows;
    core.neg = (inj == Injection::kDropCur) ? NEG_DROPCUR
             : (inj == Injection::kReadPast) ? NEG_READPAST : NEG_NONE;
    core.full = 1; core.split = 1; core.fused = 0; core.score_f32 = 0; core.tile = TILE;
    core.tile_kernel = 1; core.fattn_cover = 0;

    std::vector<uint16_t> qh((size_t)ROW_H * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r)
        for (int h = 0; h < N_Q; ++h) qh[(size_t)ROW_H * r + (size_t)h * DH + r] = bf16_bits(16.f);
    hipMemcpy(buf.qrope, qh.data(), sizeof(uint16_t) * qh.size(), hipMemcpyHostToDevice);

    std::vector<uint16_t> kh((size_t)N_KV * cap * DH, bf16_bits(0.f));
    for (int j = 0; j < cap; ++j)
        for (int kv = 0; kv < N_KV; ++kv) kh[((size_t)kv * cap + j) * DH + j] = bf16_bits(16.f);
    hipMemcpy(buf.kcache, kh.data(), sizeof(uint16_t) * kh.size(), hipMemcpyHostToDevice);
    // The slim core's own-lane K comes from `krope`, indexed by the row it belongs to.
    std::vector<uint16_t> khrow((size_t)ROW_KV * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r) {
        const int j = core.pos + r;
        for (int kv = 0; kv < N_KV; ++kv) khrow[(size_t)r * ROW_KV + (size_t)kv * DH + j] = bf16_bits(16.f);
    }
    hipMemcpy(buf.krope, khrow.data(), sizeof(uint16_t) * khrow.size(), hipMemcpyHostToDevice);

    std::vector<uint16_t> vh((size_t)cap * N_KV * DH, bf16_bits(0.f));
    build_v(vh, cap, Enc::kIndicator, 0);
    hipMemcpy(buf.vcache, vh.data(), sizeof(uint16_t) * vh.size(), hipMemcpyHostToDevice);
    std::vector<uint16_t> staged((size_t)ROW_KV * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r) {
        const int j = core.pos + r;
        for (int kv = 0; kv < N_KV; ++kv) {
            uint16_t* p = &staged[(size_t)r * ROW_KV + (size_t)kv * DH];
            for (int d = 0; d < DH; ++d) p[d] = bf16_bits(j == d ? 1.f : 0.f);
        }
    }
    hipMemcpy(buf.vg, staged.data(), sizeof(uint16_t) * staged.size(), hipMemcpyHostToDevice);

    std::vector<uint16_t> sent((size_t)ROW_H * cs.nrows, bf16_bits(kSentinel));
    std::vector<uint16_t> host_aout((size_t)ROW_H * cs.nrows), host_gout((size_t)ROW_H * cs.nrows);
    const uint16_t sent_bits = bf16_bits(kSentinel);

    for (int hot = 0; hot < cs.nrows; ++hot) {
        // qg is the interleaved [q | gate] slab the projections write: ROW_Q per row, gate half at
        // g_off(head, NEG_NONE) = head*(2*DH) + DH. Read by `finish_head` alone, indexed by `qrow`.
        std::vector<uint16_t> qgh((size_t)ROW_Q * cs.nrows,
                                  bf16_bits(hot == -1 ? 0.f : -8.f));
        for (int h = 0; h < N_Q; ++h)
            for (int d = 0; d < DH; ++d)
                qgh[(size_t)ROW_Q * hot + (size_t)h * (2 * DH) + DH + d] = bf16_bits(8.f);
        hipMemcpy(buf.qg, qgh.data(), sizeof(uint16_t) * qgh.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.aout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.gout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);

        std::string refusal;
        if (!launch_form(form, core, cs.nrows, lanes_last, &refusal)) {
            rep.why = refusal.empty() ? "launch failed" : refusal;
            return rep;
        }
        rep.ran = true;
        rep.launches++;
        hipMemcpy(host_aout.data(), buf.aout, sizeof(uint16_t) * host_aout.size(), hipMemcpyDeviceToHost);
        hipMemcpy(host_gout.data(), buf.gout, sizeof(uint16_t) * host_gout.size(), hipMemcpyDeviceToHost);

        for (int r = 0; r < cs.nrows; ++r) {
            const int bound = causal_bound(cs.lvalid, r);
            for (int h = 0; h < N_Q; ++h) {
                const uint16_t* arow = &host_aout[(size_t)ROW_H * r + (size_t)h * DH];
                const uint16_t* grow = &host_gout[(size_t)ROW_H * r + (size_t)h * DH];
                float best = -1.f;
                int dbest = -1, nz = 0;
                for (int d = 0; d < DH; ++d) {
                    if (arow[d] == sent_bits) { rep.not_written++; continue; }
                    const float x = bf16_val(arow[d]);
                    if (!std::isfinite(x)) { rep.nonfinite++; continue; }
                    if (!(x > 0.f)) continue;
                    ++nz;
                    if (x > best) { best = x; dbest = d; }
                }
                ++rep.l_checked;
                if (dbest != r) {
                    rep.identity_wrong++;
                    if ((r % 2) == 0) rep.oddviol++; else rep.evenviol++;
                    char b[224];
                    snprintf(b, sizeof b,
                             "row %d (gate hot row %d): its scores came from ROW %d's query, not its own "
                             "(peak %.4g, %d dims above zero)", r, hot, dbest, best, nz);
                    note(rep, b);
                    continue;
                }
                // With the hot lane at +16 and every other lane at 0.0, the rowsum is 1 + (bound-1)*e^-16,
                // i.e. 1.0 to BF16, so the peak must read exactly 1.0. That is a scale check on the P*V
                // product and on the divide — a different check from the flat pass's `l == bound`, which is
                // where the rowsum is measurable (see the `read_l` comment in run()).
                if (!(std::fabs(best - 1.f) <= 0.01f)) {
                    rep.wrong_l_rows++;
                    char b[224];
                    snprintf(b, sizeof b, "row %d head %d: peak = %.6g, expected 1/rowsum = 1.000 (bound %d)",
                             r, h, best, bound);
                    note(rep, b);
                }
                const float av = bf16_val(arow[dbest]), gv = bf16_val(grow[dbest]);
                if (av <= 0.f || !std::isfinite(gv) || gv == bf16_val(sent_bits)) { rep.not_written++; continue; }
                const float s = gv / av;                       // = bf(sigmoid(gate used))
                const bool gate_is_hot = (s > 0.5f);
                if (gate_is_hot != (r == hot)) {
                    rep.gate_wrong++;
                    char b[224];
                    snprintf(b, sizeof b, "row %d (gate hot row %d): its gate was sigmoid(%.4f) = %.4f, "
                             "so it carries %s row's gate", r, hot, (s > 0.5f ? 8.f : -8.f), s,
                             gate_is_hot ? "the HOT" : "a COLD");
                    note(rep, b);
                }
            }
        }
    }
    if (!rep.ran) { rep.refused = true; if (rep.why.empty()) rep.why = "no launch ran"; }
    return rep;
}

// ---- the boundary encoding: DOES THE MASKED CELL'S SCORE REACH THE MAX OR THE ROWSUM? -------------------
// Everything before it asks a STRUCTURAL question — which lanes got weight, which cell read which key, whose
// query scored this row — and a structural answer is blind to the one failure mode that fits CROSSROW's
// numbers. The engine dump shows row q's attention output changing by ONE BF16 ULP in a handful of its 2048
// elements when lane `bound(q)` changes (`prototype/mtp_verify/FIX-CROSSROW.md`, Result 4), with every input
// that row is entitled to proved byte-identical. A change that small is not a weight appearing; it is the
// F32 rounding of the row changing. And softmax is scale-invariant: if a masked cell's SCORE reaches the
// running max `m`, every `exp(s - m)` and the rowsum scale by the same factor, the ratio is unchanged except
// in its last F32 bits, and a few outputs cross a BF16 rounding boundary. That is exactly the signature.
// So: give the masked boundary cell a score so large that the cancellation stops being harmless. Q is zero
// except at dim == bound(row), K is one-hot per lane, so score(row r, lane j) is 0 everywhere and exactly
// kBoundaryScore at the first lane PAST r's bound. If the mask protects the max and the rowsum, the answer is
// the flat answer again: 1/bound on dims 0..bound-1 and nothing beyond. If the masked score reaches `m`, the
// live weights become exp(0 - score) — flushed to zero by F16 — and the row publishes zeros or a wrong count,
// which every check in `run()` reads as a failure. With a flat probe this cannot be seen at all: the flat
// probe's masked cells score -1e30 and reach the max as -1e30, which is harmless by construction.
constexpr float kBoundaryScore = 40.f;      // exp(-40) underflows F16 to zero; 40 is exact in BF16
constexpr float kBoundaryQ = kBoundaryScore * 16.f;   // the round's SCALE is 1/16, so dot = score*16

Report run_boundary(Form form, Injection inj, const Case& cs, Buffers& buf) {
    Report rep;
    const int lanes_last = cs.lvalid + cs.nrows - 1;
    const int cap = lanes_last + kCapSlack;
    if (form == Form::kSlim && lanes_last > MAXSC) {
        rep.refused = true;
        rep.why = "lanes_last " + std::to_string(lanes_last) + " > MAXSC (part + attn_combine not built here)";
        return rep;
    }
    if (!case_wants_indicator(Enc::kBoundary, cs.lvalid, cs.nrows)) {
        rep.refused = true;
        rep.why = "boundary sweep needs one dim per lane up to " + std::to_string(lanes_last)
                  + ", over the " + std::to_string(kPoisonMaxLanes) + "-lane budget";
        return rep;
    }
    buf.ensure(cs.nrows, cap);

    const int bound_given_to_kernel = (inj == Injection::kWiden1) ? cs.lvalid + 1 : cs.lvalid;
    CoreArgs core{};
    core.qg = buf.qg; core.kcache = buf.kcache; core.vcache = buf.vcache; core.vg = buf.vg;
    core.qrope = buf.qrope; core.krope = buf.krope; core.aout = buf.aout; core.gout = buf.gout;
    core.kc_as = buf.kcache; core.vc_as = buf.vcache;
    core.cap = cap; core.pos = cs.lvalid - 1; core.lvalid = bound_given_to_kernel; core.nrows = cs.nrows;
    core.neg = (inj == Injection::kDropCur) ? NEG_DROPCUR
             : (inj == Injection::kReadPast) ? NEG_READPAST : NEG_NONE;
    core.full = 1; core.split = 1; core.fused = 0; core.score_f32 = 0; core.tile = TILE;
    core.tile_kernel = 1; core.fattn_cover = 0;

    // K: lane j is one-hot at dim j, so a dot picks out exactly one Q dim.
    std::vector<uint16_t> kh((size_t)N_KV * cap * DH, bf16_bits(0.f));
    for (int j = 0; j < cap; ++j)
        for (int kv = 0; kv < N_KV; ++kv) kh[((size_t)kv * cap + j) * DH + j] = bf16_bits(1.f);
    hipMemcpy(buf.kcache, kh.data(), sizeof(uint16_t) * kh.size(), hipMemcpyHostToDevice);
    std::vector<uint16_t> khrow((size_t)ROW_KV * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r) {
        const int j = core.pos + r;
        for (int kv = 0; kv < N_KV; ++kv) khrow[(size_t)r * ROW_KV + (size_t)kv * DH + j] = bf16_bits(1.f);
    }
    hipMemcpy(buf.krope, khrow.data(), sizeof(uint16_t) * khrow.size(), hipMemcpyHostToDevice);

    // Q: zero on every lane the row may attend, kBoundaryQ on its first forbidden lane and beyond.
    std::vector<uint16_t> qh((size_t)ROW_H * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r) {
        const int bound = causal_bound(cs.lvalid, r);
        for (int h = 0; h < N_Q; ++h)
            for (int d = bound; d < DH; ++d) qh[(size_t)ROW_H * r + (size_t)h * DH + d] = bf16_bits(kBoundaryQ);
    }
    hipMemcpy(buf.qrope, qh.data(), sizeof(uint16_t) * qh.size(), hipMemcpyHostToDevice);

    // V: the indicator at base 0, so out[r][h][d] is the weight of cell d divided by the rowsum.
    std::vector<uint16_t> vh((size_t)cap * N_KV * DH, bf16_bits(0.f));
    build_v(vh, cap, Enc::kIndicator, 0);
    hipMemcpy(buf.vcache, vh.data(), sizeof(uint16_t) * vh.size(), hipMemcpyHostToDevice);
    std::vector<uint16_t> staged((size_t)ROW_KV * cs.nrows, bf16_bits(0.f));
    for (int r = 0; r < cs.nrows; ++r) {
        const int j = core.pos + r;
        for (int kv = 0; kv < N_KV; ++kv) {
            uint16_t* p = &staged[(size_t)r * ROW_KV + (size_t)kv * DH];
            for (int d = 0; d < DH; ++d) p[d] = bf16_bits(j == d ? 1.f : 0.f);
        }
    }
    hipMemcpy(buf.vg, staged.data(), sizeof(uint16_t) * staged.size(), hipMemcpyHostToDevice);

    std::vector<uint16_t> sent((size_t)ROW_H * cs.nrows, bf16_bits(kSentinel));
    std::vector<uint16_t> host_aout((size_t)ROW_H * cs.nrows);
    const uint16_t sent_bits = bf16_bits(kSentinel);
    hipMemcpy(buf.aout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);
    hipMemcpy(buf.gout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);

    std::string refusal;
    if (!launch_form(form, core, cs.nrows, lanes_last, &refusal)) {
        rep.why = refusal.empty() ? "launch failed" : refusal;
        rep.refused = true;
        return rep;
    }
    rep.ran = true;
    rep.launches = 1;
    hipMemcpy(host_aout.data(), buf.aout, sizeof(uint16_t) * host_aout.size(), hipMemcpyDeviceToHost);

    for (int r = 0; r < cs.nrows; ++r) {
        const int bound = causal_bound(cs.lvalid, r);
        const bool odd = (bound % 2) == 1;
        for (int h = 0; h < N_Q; ++h) {
            const uint16_t* rowp = &host_aout[(size_t)ROW_H * r + (size_t)h * DH];
            int nz = 0, top = -1, beyond = 0;
            float v0 = 0.f;
            for (int d = 0; d < DH; ++d) {
                const uint16_t bits = rowp[d];
                if (bits == sent_bits) { rep.not_written++; continue; }
                const float x = bf16_val(bits);
                if (!std::isfinite(x)) { rep.nonfinite++; continue; }
                if (!(x > 0.f)) continue;
                ++nz;
                top = d;
                if (d >= bound) ++beyond;
                if (d == 0) v0 = x;
            }
            ++rep.l_checked;
            if (beyond > 0) {
                rep.past_bound_rows++;
                if (odd) rep.oddviol++; else rep.evenviol++;
                char b[224];
                snprintf(b, sizeof b, "row %d (bound %d): %d cells at or past the bound carry weight even though "
                         "the score there is %g and masked", r, bound, beyond, double(kBoundaryScore));
                note(rep, b);
                continue;
            }
            if (nz != bound) {
                rep.short_rows++;
                if (odd) rep.oddviol++; else rep.evenviol++;
                char b[260];
                snprintf(b, sizeof b, "row %d (bound %d): only %d of its own lanes carry weight — the rowsum "
                         "moved, because the masked lane %d's score is %g and %d cells vanished", r, bound, nz,
                         bound, double(kBoundaryScore), bound - nz);
                note(rep, b);
                continue;
            }
            // Same rowsum check as the flat pass: every live weight is exactly 1, so dim 0 reads 1/bound.
            const float implied = (v0 > 0.f) ? 1.f / v0 : -1.f;
            if (std::fabs(implied - (float)bound) > 0.4f) {
                rep.wrong_l_rows++;
                char b[224];
                snprintf(b, sizeof b, "row %d head %d: 1/out[0] = %.4f, bound %d — the rowsum is not the count "
                         "of live cells once a masked lane scores %g", r, h, implied, bound,
                         double(kBoundaryScore));
                note(rep, b);
            }
        }
    }
    return rep;
}

Report run(Form form, Enc enc, Injection inj, const Case& cs, Buffers& buf) {
    if (enc == Enc::kPoison) return run_poison(form, inj, cs, buf);
    if (enc == Enc::kIdentity) return run_identity(form, inj, cs, buf);
    if (enc == Enc::kBoundary) return run_boundary(form, inj, cs, buf);
    Report rep;
    const int lanes_last = cs.lvalid + cs.nrows - 1;
    const int cap = lanes_last + kCapSlack;
    if (form == Form::kSlim && lanes_last > MAXSC) {
        rep.refused = true;
        rep.why = "lanes_last " + std::to_string(lanes_last) + " > MAXSC " + std::to_string(MAXSC)
                  + " (part + attn_combine not built here)";
        return rep;
    }
    if (!case_wants_indicator(enc, cs.lvalid, cs.nrows)) {
        rep.refused = true;
        rep.why = std::string(enc_name(enc)) + " sweep would need " 
                  + std::to_string((lanes_last + kWindow) / kWindow) + " launches; skipped at this width";
        return rep;
    }
    buf.ensure(cs.nrows, cap);

    // Q = 0 and K = 0 -> every score is exactly +0.0, so every live cell's weight is exactly 1.
    std::vector<uint16_t> zero_rows(std::max((size_t)ROW_H * cs.nrows, (size_t)ROW_KV * cs.nrows),
                                        bf16_bits(0.f));
    std::vector<uint16_t> zero_cache((size_t)N_KV * cap * DH, bf16_bits(0.f));
    hipMemcpy(buf.qrope, zero_rows.data(), sizeof(uint16_t) * (size_t)ROW_H * cs.nrows,
              hipMemcpyHostToDevice);
    hipMemcpy(buf.krope, zero_rows.data(), sizeof(uint16_t) * (size_t)ROW_KV * cs.nrows,
              hipMemcpyHostToDevice);
    hipMemcpy(buf.kcache, zero_cache.data(), sizeof(uint16_t) * zero_cache.size(), hipMemcpyHostToDevice);

    // What the KERNEL's mask reads, and what the oracle expects, are deliberately different quantities
    // here: `widen1` is exactly the gap between them.
    const int bound_given_to_kernel = (inj == Injection::kWiden1) ? cs.lvalid + 1 : cs.lvalid;

    CoreArgs core{};
    core.qg = buf.qg;
    core.kcache = buf.kcache;
    core.vcache = buf.vcache;
    core.vg = buf.vg;
    core.qrope = buf.qrope;
    core.krope = buf.krope;
    core.aout = buf.aout;
    core.gout = buf.gout;
    core.kc_as = buf.kcache;
    core.vc_as = buf.vcache;
    core.qrope_out = nullptr;      // staged plan: this launch neither stages nor appends
    core.krope_out = nullptr;
    core.gate_out = nullptr;
    core.rope_out = nullptr;
    core.part = nullptr;
    core.stale = nullptr;
    core.cap = cap;
    core.pos = cs.lvalid - 1;                       // row 0 sits at cache position lvalid-1
    core.lvalid = bound_given_to_kernel;
    core.nrows = cs.nrows;
    core.neg = (inj == Injection::kDropCur) ? NEG_DROPCUR
             : (inj == Injection::kReadPast) ? NEG_READPAST : NEG_NONE;
    core.full = 1;
    core.split = 1;
    core.fused = 0;
    core.score_f32 = 0;
    core.tile = TILE;
    core.tile_kernel = 1;
    core.fattn_cover = 0;

    // Per (row, head): the topmost lane answered "yes", and (indicator mode) the attended set.
    std::vector<std::vector<int>> top(cs.nrows, std::vector<int>(N_Q, -1));
    std::vector<std::vector<int>> count(cs.nrows, std::vector<int>(N_Q, 0));
    std::vector<std::vector<std::vector<int>>> set_of(
        (enc == Enc::kIndicator) ? cs.nrows : 0, std::vector<std::vector<int>>(N_Q));
    // THE DENOMINATOR READOUT. Every test so far compares output cells against each other, so a leak that
    // inflates `l` without ever putting weight on a V entry — an extra term in the rowsum only — scales the
    // whole row and is invisible to a zero/non-zero or ratio test. The flat input fixes that: with every
    // score exactly 0, every live cell's weight is exactly 1, so with the indicator V the cell at dim 0 is
    // exactly 1/l and `l` is readable as 1/out[r][h][0]. BF16 gives 8 mantissa bits, so the reading
    // resolves a one-count error up to bound ~64, which is where the leak reproduces anyway (the MTP pair
    // is two rows). A denominator leak is exactly what CROSSROW's "argmax SAME, 1-cos ~1e-5" signature
    // looks like, because a scale change moves the logits a little and the attention direction not at all.
    std::vector<std::vector<float>> l_obs(cs.nrows, std::vector<float>(N_Q, 0.f));
    const bool read_l = (enc == Enc::kIndicator) && (cs.lvalid + cs.nrows - 1) <= 64;

    std::vector<int> bases = (enc == Enc::kIndicator) ? indicator_offsets(cs.lvalid, cs.nrows)
                                                      : threshold_offsets(cs.lvalid, cs.nrows);
    // --dump turns on a raw listing of the non-zero output cells of one small case. It exists because the
    // aggregate verdicts on the first run disagreed with the arithmetic in the header comment, and the only
    // honest way to find out which of the two is wrong is to look at the bits.
    const bool dumping = g_dump && cs.nrows == g_dump_rows && cs.lvalid == g_dump_lvalid
                         && form_name(form) == std::string(g_dump_form);
    std::vector<uint16_t> vh((size_t)cap * N_KV * DH, bf16_bits(0.f));
    std::vector<uint16_t> staged((size_t)ROW_KV * cs.nrows, bf16_bits(0.f));
    std::vector<uint16_t> sent((size_t)ROW_H * cs.nrows, bf16_bits(kSentinel));
    std::vector<uint16_t> host_aout((size_t)ROW_H * cs.nrows);
    const unsigned short sent_bits = bf16_bits(kSentinel);

    for (int base : bases) {
        build_v(vh, cap, enc, base);
        // The non-wide core reads its OWN new lane out of vg/krope at j == pos + row, so the staged copy
        // has to answer the same question the cache lane it stands for answers.
        for (int r = 0; r < cs.nrows; ++r) {
            const int j = core.pos + r;
            for (int kv = 0; kv < N_KV; ++kv) {
                uint16_t* p = &staged[(size_t)r * ROW_KV + (size_t)kv * DH];
                for (int d = 0; d < DH; ++d) {
                    const int t = base + d;
                    const bool one = (enc == Enc::kIndicator) ? (j == t) : (j >= t);
                    p[d] = bf16_bits(one ? 1.f : 0.f);
                }
            }
        }
        hipMemcpy(buf.vcache, vh.data(), sizeof(uint16_t) * vh.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.vg, staged.data(), sizeof(uint16_t) * staged.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.aout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);
        hipMemcpy(buf.gout, sent.data(), sizeof(uint16_t) * sent.size(), hipMemcpyHostToDevice);

        std::string refusal;
        if (!launch_form(form, core, cs.nrows, lanes_last, &refusal)) {
            rep.why = refusal.empty() ? "launch failed" : refusal;
            return rep;
        }
        rep.ran = true;
        rep.launches++;
        hipMemcpy(host_aout.data(), buf.aout, sizeof(uint16_t) * host_aout.size(),
                  hipMemcpyDeviceToHost);

        for (int r = 0; r < cs.nrows; ++r) {
            for (int h = 0; h < N_Q; ++h) {
                const uint16_t* rowp = &host_aout[(size_t)ROW_H * r + (size_t)h * DH];
                for (int d = 0; d < DH; ++d) {
                    const unsigned short bits = rowp[d];
                    if (bits == sent_bits) { rep.not_written++; continue; }
                    const float x = bf16_val(rowp[d]);
                    if (!std::isfinite(x)) { rep.nonfinite++; continue; }
                    if (x == 0.0f) continue;
                    if (dumping && (r == g_dump_row) && g_dump_left > 0) {
                        printf("    CELL r=%d h=%d base=%-4d d=%-4d lane=%-4d val=%.6g\n", r, h, base, d,
                               base + d, x);
                        --g_dump_left;
                    }
                    const int lane_queried = base + d;
                    if (read_l && base == 0 && d == 0) l_obs[r][h] = x;
                    if (enc == Enc::kThreshold) {
                        top[r][h] = std::max(top[r][h], lane_queried);
                    } else {
                        set_of[r][h].push_back(lane_queried);
                    }
                }
            }
        }
    }
    if (!rep.ran) { rep.refused = true; if (rep.why.empty()) rep.why = "no launch ran"; return rep; }

    // ---- adjudicate, per row, against the probe's own statement of the causal law.
    for (int r = 0; r < cs.nrows; ++r) {
        const int bound = causal_bound(cs.lvalid, r);      // lanes 0..bound-1 are this row's
        int observed = -1;
        int cnt = -1;
        for (int h = 0; h < N_Q; ++h) {
            if (enc == Enc::kThreshold) {
                observed = std::max(observed, top[r][h]);
            } else {
                observed = std::max(observed, set_of[r][h].empty() ? -1 : set_of[r][h].back());
                cnt = std::max<int>(cnt, (int)set_of[r][h].size());
            }
        }
        const bool odd = (bound % 2) == 1;
        if (observed >= bound) {
            rep.past_bound_rows++;
            if (odd) rep.oddviol++; else rep.evenviol++;
            rep.worst_row = r; rep.worst_observed = observed; rep.worst_bound = bound;
            char b[224];
            snprintf(b, sizeof b, "row %d: bound %d (lanes 0..%d), attended lane %d%s%s",
                     r, bound, bound - 1, observed,
                     enc == Enc::kIndicator ? ", attended " : "",
                     enc == Enc::kIndicator ? std::to_string(cnt).c_str() : "");
            note(rep, b);
        } else if (observed != bound - 1) {
            rep.short_rows++;
            if (odd) rep.oddviol++; else rep.evenviol++;
            char b[224];
            snprintf(b, sizeof b, "row %d: bound %d, topmost lane attended %d", r, bound, observed);
            note(rep, b);
        } else if (enc == Enc::kIndicator && cnt != bound) {
            rep.short_rows++;
            if (odd) rep.oddviol++; else rep.evenviol++;
            char b[224];
            snprintf(b, sizeof b, "row %d: bound %d but only %d lanes attended", r, bound, cnt);
            note(rep, b);
        }
        if (read_l) {
            for (int h = 0; h < N_Q; ++h) {
                const float v = l_obs[r][h];
                if (!(v > 0.f)) continue;                     // covered by not_written / short_rows
                ++rep.l_checked;
                const float implied = 1.f / v;                 // = l, exactly up to BF16 rounding
                const long l_whole = lroundf(implied);
                if (std::fabs(implied - (float)l_whole) > 0.35f) {
                    rep.wrong_l_rows++;
                    char b[224];
                    snprintf(b, sizeof b, "row %d head %d: 1/l = %.4f is not a whole count; bound %d",
                             r, h, implied, bound);
                    note(rep, b);
                } else if (l_whole != bound) {
                    rep.wrong_l_rows++;
                    if (odd) rep.oddviol++; else rep.evenviol++;
                    char b[224];
                    snprintf(b, sizeof b, "row %d head %d: rowsum l = %ld, bound %d -> the rowsum is "
                             "%s by %ld",
                             r, h, l_whole, bound, l_whole > bound ? "WIDER" : "NARROWER",
                             l_whole > bound ? l_whole - bound : bound - l_whole);
                    note(rep, b);
                }
            }
        }
    }
    // A check with an empty population cannot fail (AGENTS' gdn_collapse row): if the readout was asked for
    // and found nothing to read, that is the finding, not a pass.
    if (read_l && rep.l_checked == 0) {
        rep.wrong_l_rows++;
        note(rep, "denominator readout found no live cell: the l check adjudicated nothing");
    }
    return rep;
}

int print_case(Form form, Enc enc, Injection inj, const Case& cs, const Report& rep, int* total, int* failed) {
    (*total)++;
    const bool bad = rep.red();
    if (bad) (*failed)++;
    printf("CASE form=%-10s enc=%-9s inj=%-8s core=%-21s nrows=%-4d lvalid=%-5d launches=%-3d "
           "past=%-3d short=%-3d poisonwrong=%-3d idwrong=%-4d gatewrong=%-4d notwritten=%-4d "
           "nonfinite=%-3d lchecked=%-5d oddviol=%-3d evenviol=%-3d %s\n",
           form_name(form), enc_name(enc),
           (inj == Injection::kNone ? "none"
            : inj == Injection::kDropCur ? "dropcur"
            : inj == Injection::kReadPast ? "readpast" : "widen1"),
           rep.kernel,
           cs.nrows, cs.lvalid, rep.launches, rep.past_bound_rows, rep.short_rows, rep.poison_wrong,
           rep.identity_wrong, rep.gate_wrong, rep.not_written, rep.nonfinite, rep.l_checked,
           rep.oddviol, rep.evenviol,
           rep.refused ? "REFUSED" : (bad ? "FAIL" : "PASS"));
    if (rep.refused) printf("      refused: %s\n", rep.why.c_str());
    for (const auto& f : rep.findings) printf("      %s\n", f.c_str());
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* only_form = nullptr;
    const char* only_enc = nullptr;
    const char* inject_name = "none";
    bool must_catch = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* opt) -> const char* {
            if (i + 1 >= argc) { printf("FATAL: %s needs a value\n", opt); std::exit(2); }
            return argv[++i];
        };
        if (a == "--form") only_form = next("--form");
        else if (a == "--enc") only_enc = next("--enc");
        else if (a == "--inject") inject_name = next("--inject");
        else if (a == "--must-catch") must_catch = true;
        else if (a == "--dump") {
            const char* spec = next("--dump");
            char form[32];
            int nr = 0, lv = 0, rw = 0, lim = 60;
            const int n = sscanf(spec, "%31[^,],%d,%d,%d,%d", form, &nr, &lv, &rw, &lim);
            if (n < 4) { printf("FATAL: --dump wants form,nrows,lvalid,row[,limit]\n"); return 2; }
            g_dump = true; g_dump_form = form; g_dump_rows = nr; g_dump_lvalid = lv; g_dump_row = rw;
            g_dump_left = lim;
        }
        else { printf("FATAL: unknown argument %s\n", a.c_str()); return 2; }
    }

    printf("causal_bound: the engine's own attention cores, asked which lanes they attended.\n");
    printf("  geometry: N_Q=%d N_KV=%d GQA=%d DH=%d MAXSC=%d | WIDE_POS_PER_BLOCK=%d WIDE_STAGE=%d "
           "WIDE_POS_PER_WARP=%d | body cap kBodyRows=%d (engine/decode_step.h:49)\n",
           N_Q, N_KV, GQA, DH, MAXSC, WIDE_POS_PER_BLOCK, WIDE_STAGE, WIDE_POS_PER_WARP, kBodyRows);
    printf("  oracle: causal_bound(lvalid,q) = lvalid+q (probe's own statement of §06.3; NOT lanes_of)\n");
    printf("  shipping default form is wide_llama = --attn-fattn-wide 2 (engine/decode_run.cpp:262)\n");

    std::vector<Injection> injections;
    if (must_catch) {
        for (const auto& i : kInjections) if (i.must_be_red) injections.push_back(i.inj);
    } else {
        const InjectionInfo* ii = find_injection(inject_name);
        if (!ii) { printf("FATAL: unknown --inject %s\n", inject_name); return 2; }
        injections.push_back(ii->inj);
    }

    Buffers buf;
    const std::vector<Case> cases = derive_cases();
    if (cases.empty()) { printf("FATAL: derived case list is empty\n"); return 2; }
    int total = 0, failed = 0, controls_red = 0, controls_total = 0;
    std::vector<std::string> control_lines;

    for (Injection inj : injections) {
        const InjectionInfo* ii = nullptr;
        for (const auto& i : kInjections) if (i.inj == inj) ii = &i;
        printf("\n== injection %s: %s\n", ii->name, ii->what);
        for (const auto& fi : kForms) {
            if (only_form && std::strcmp(only_form, fi.name) != 0) continue;
            for (Enc enc : {Enc::kThreshold, Enc::kIndicator, Enc::kPoison, Enc::kIdentity,
                            Enc::kBoundary}) {
                if (only_enc) {
                    const bool want = (std::strcmp(only_enc, enc_name(enc)) == 0);
                    if (!want) continue;
                }
                int f_past = 0, f_short = 0, f_other = 0, ran = 0;
                for (const auto& cs : cases) {
                    Report rep = run(fi.form, enc, inj, cs, buf);
                    rep.kernel = g_launched_kernel;
                    print_case(fi.form, enc, inj, cs, rep, &total, &failed);
                    if (rep.ran) {
                        ran++;
                        f_past += rep.past_bound_rows;
                        f_short += rep.short_rows;
                        f_other += rep.not_written + rep.nonfinite + rep.poison_wrong + rep.identity_wrong
                      + rep.gate_wrong;
                    }
                }
                if (ran == 0) {
                    printf("form=%-10s enc=%-9s inj=%-8s NO CASE RAN -- unclassified, not a pass\n",
                           fi.name, enc_name(enc), ii->name);
                    failed++;
                    continue;
                }
                const bool red = (f_past + f_short + f_other) > 0;
                if (ii->must_be_red) {
                    controls_total++;
                    if (red) controls_red++;
                    char b[256];
                    snprintf(b, sizeof b, "control form=%s enc=%s inj=%s: past=%d short=%d other=%d -> %s",
                             fi.name, enc_name(enc), ii->name, f_past, f_short, f_other,
                             red ? "RED, as it must be" : "DID NOT FIRE -- the control is decoration");
                    control_lines.push_back(b);
                }
            }
        }
    }

    printf("\ncontrols:\n");
    for (const auto& l : control_lines) printf("  %s\n", l.c_str());
    if (controls_total > 0 && controls_red != controls_total) {
        printf("VERDICT: a must-catch control did not fire. The probe cannot see a one-lane bound change,\n"
               "so its PASS verdicts are not evidence. Fix the probe before trusting it.\n");
    }
    printf("RUN cases=%zu forms=%d encs=5 injections=%zu parsed=%d failures=%d\n",
           cases.size(), only_form ? 1 : (int)std::size(kForms), injections.size(), total, failed);
    printf("%s\n", (failed == 0 ? "CAUSAL_BOUND PASS" : "CAUSAL_BOUND FAIL"));
    return failed == 0 ? 0 : 1;
}
