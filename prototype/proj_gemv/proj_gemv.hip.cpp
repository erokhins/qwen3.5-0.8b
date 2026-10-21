/*
 * prototype/proj_gemv/proj_gemv.hip.cpp — the projection GEMV family, k = 1: one templated
 * row-per-wavefront BF16 GEMV instantiated over seven shapes, not seven kernels.
 *
 * The shapes (specification/01 §01.2/§01.3, sizes from `tools/weights_report.py`):
 *
 *   linear_attn.in_proj_qkv [6144,1024] x18   linear_attn.in_proj_z [2048,1024] x18
 *   linear_attn.in_proj_b   [  16,1024] x18   linear_attn.in_proj_a [  16,1024] x18
 *   linear_attn.out_proj    [1024,2048] x18   self_attn.q_proj      [4096,1024] x6
 *   self_attn.k_proj        [  512,1024] x6   self_attn.v_proj      [  512,1024] x6
 *   self_attn.o_proj        [ 1024,2048] x6
 *
 * Together 0.4677 GB/token = 29 % of the token's bytes (engine/performance-status.md §4), the
 * largest slice left after the MLP. Only two input widths appear — 1024 and 2048 — so the whole
 * family is `proj_gemv<1024>` and `proj_gemv<2048>`: the same skeleton as `mlp_gemv`'s
 * `mlp_down<IN>` (one wavefront per output row, x staged in shared, F32 accumulator, one BF16
 * rounding per dot; A-07, and §10.1's F1 line for what "reproduce" means here).
 *
 * LAUNCH PLAN. Every projection with the same input width and the same x can share one launch by
 * concatenating rows at load time, and the engine should: {in_proj_qkv, in_proj_z, in_proj_b,
 * in_proj_a} are all [*,1024] of the same normed x -> one [8224,1024] launch; {q_proj, k_proj,
 * v_proj} -> one [5120,1024] launch. out_proj/o_proj are [1024,2048] fed by a *different* x (the
 * gated norm output, §04.5/§05.6), so they cannot join, and nothing can be merged across a block
 * boundary because the residual add and the next norm sit in between. So: 2 launches per layer,
 * 48/token against 114 unfused — ~0.12 ms/token at the measured 1.85 us launch floor.
 * in_proj_b/in_proj_a are [16,1024] = 32 KiB per layer, i.e. launch cost rather than bandwidth,
 * which is why they are folded rather than timed as shapes of their own. The driver below is
 * plan-driven so the gate can exercise packed and unpacked forms with one code path.
 *
 * TOKEN BATCHING (P5). A prefill wave hands one projection a [T,IN] tile instead of one row, and
 * `proj_gemv_tokens` reads each weight row once per TPB-token tile rather than once per token; that
 * is the family's only bandwidth lever, because the narrow shapes are shape-limited (results.md:
 * `k_proj`/`v_proj` at 39-40 % of the read probe, `o_proj`/`out_proj` 46-49 %, `in_proj_a`/`b` ~5 %).
 * The engine issues that kernel at every width, one token included. §06.2's "one GEMM per
 * projection" is this sentence.
 *
 * Rounding: one F32 dot, rounded once to BF16 (RNE, the cast torch does). There is no epilogue to
 * get wrong here (unlike the MLP's three-rounding SwiGLU); what there is to get wrong is the row
 * LAYOUT — §05.1's per-head `[8 heads][q 256 | gate 256]` q_proj block and §04.1's `[q|k|v]`
 * split — which is why two of the injected bugs below are output-row permutations.
 *
 * Injected bugs, each exercised by run.py --negative with a declared must-catch / must-not-catch
 * location (the `mlp_gemv` convention: a negative that cannot fail is worse than none):
 *   tail       drop the last word of K (the off-by-one a `v < K/8 - 1` loop bound gives)
 *   half       accumulate half of K
 *   trunc      round toward zero instead of round-to-nearest-even
 *   rowswap    write each dot to the neighbouring output row (row/output swap)
 *   transpose  read the weight buffer transposed (rows/in reversed)
 *   qhalf      write q_proj rows in the contiguous-half [all q | all gate] layout §05.1 forbids
 *   kvswap     swap the k and v sections of a packed in_proj_qkv output (§04.1)
 * Plus two input-side negatives run by run.py with no kernel change: the pre-norm tensor, and the
 * other norm's weights, where the input-layernorm output belongs (see NEG_INPUTS in run.py).
 *
 * Build: prototype/proj_gemv/build.sh   (-O3, -Wl,-rpath,/opt/rocm/lib)
 * Run:   ./proj_gemv --mode once  --plan PLAN [--rows N] [--neg NAME] [--x-packed]
 *        ./proj_gemv --mode once  --plan PLAN --tokens T [--tpb N] [--kernel k1]
 *        ./proj_gemv --mode burst --plan PLAN [--rows N] [--iters N] [--tile N] [--write-out]
 *        ./proj_gemv --mode burst --plan PLAN --rows-list 2,4,8,16,32 [--tile N] [--tpb N]
 *        ./proj_gemv --mode burst --plan PLAN --variant v5-vx --rows-list 8,16,32 \
 *                                       --threads 512 --reps 3 --warm 2     # shape search
 *        ./proj_gemv --mode once  --plan PLAN --dump-plan                  # geometry, no device
 * Validate: prototype/proj_gemv/run.py
 *
 * ROWS/Block AND THREADS/BLOCK ARE TWO AXES. `--threads N` (alias `--tpr`) sets the block size
 * independently of `--rows`; the kernel walks a block's rows in trips of `threads/lanes_per_row`, so
 * the default (`--threads 0`, i.e. `rpb*lanes_per_row`) is one trip and is exactly the shape every
 * published number in `results.md` was taken with — nothing already measured moves. That welding
 * (`threads = rpb * ws`, in this file since its first commit 5d6405f) meant a published row could only
 * be reproduced at the one rows/block that implied its thread count, and "rows/block 32 at 512
 * threads" was not a thing the rig could ask for. Variant 0 and `--kernel k1` **refuse** a non-default
 * `--threads`: variant 0 is the kernel `engine/kernels_proj.cpp` compiles and its launcher emits
 * `kRowsPerBlock * warp_width()` threads, so timing any other shape would price something the engine
 * cannot run. `v1-unroll` and `v5-vx` are bit-identical to variant 0 and do take it.
 *
 * `--warm N` (default 2) runs the plan N untimed passes before each timed point. Freshly uploaded
 * pages run at ~35 GB/s on this part, and a config whose first timed rep had not been warmed reported
 * 17.497 ms where its own best rep was 3.544 ms — the fresh-page rate wearing the costume of a result.
 *
 * THE D1 VARIANT AXIS. `--variant` takes a name or index from `g_var` below: variant 0 is the shipping
 * `proj_gemv_tokens` and 1-6 are `proj_gemv_var` instantiations that differ from it in lanes-per-row
 * (LPW) and/or the width of the `x` staging load (VX). The table is the single source for the variant
 * name, its lanes-per-row, its default rows/block and whether it claims bit-identity, and the
 * launcher reads its thread count from the table rather than from a second spelling; `run.py` then
 * cross-checks this table against its own copy by running `--dump-plan`, which echoes
 * `label/lanes_per_row/rows_per_block/threads/exact` without touching the device. `--label NAME`
 * overrides only the printed name, so a sweep can give an arm a plan-specific label and still have the
 * geometry reported honestly. The variants are the decode (TPB=1) forms: the prefill tile widths stay
 * on variant 0 so a decode experiment cannot move a prefill number.
 *
 * `--reps N` bursts the same uploaded plan N times. `ms_per_plan` is the MIN of those reps, because
 * best-of-N is this repo's timing convention (`prototype/README.md`); `ms_min`/`ms_max`/`ratio` are
 * printed beside it and `pt` says what the spread means: `ok` at max/min ≤ 1.10, `provisional` above
 * it (the level at which performance-status.md §6 says a row cannot be adjudicated across windows),
 * and `void` above 2.00 — a point that still disagrees with itself by 2x after warming was not measured
 * by anything, so `ms_per_plan`, `gbs` and `probe_pct` print `VOID` rather than a plausible number.
 *
 * PLAN is a text file, one launch per line:
 *     NAME ROWS IN XFILE WFILE[,WFILE...] [OUTFILE] [VAR=name[:rows[:threads]]]
 * XFILE is raw little-endian BF16 with IN elements — or, with `--tokens T`, with T*IN elements, the
 * [T,IN] input tile one prefill wave presents (row-major, token-major). WFILEs are row-major BF16
 * [rows_i, IN] concatenated in the order given (ROWS must equal the sum); OUTFILE, if present,
 * receives the BF16 output — ROWS elements in the k=1 form, TOKENS*ROWS elements in the batched
 * one, laid out [token][row] — always in `once` mode, in `burst` mode only with --write-out.
 *
 * The optional 7th field picks the variant for THAT launch, overriding `--variant`, with an optional
 * rows-per-block of its own. That is what lets one plan carry a mixed geometry — the engine's 114
 * launches with only the 36 [16,1024] `in_proj_b`/`in_proj_a` jobs on a narrow split, the big shapes
 * untouched — over one weight upload, instead of one binary call per variant. `run.py --emit-plans`
 * writes `plan_token_{packed,unpacked,mixed}.txt` this way.
 *
 * --kernel picks which kernel the plan runs through: `tokens` (default) is the tile form the engine
 * uses at every width, `TPB = 1` included; `k1` is the original one-wavefront-per-row kernel, kept
 * as the batched gate's independent per-row reference. --tpb overrides the tile width the launcher
 * picks (4 at in=1024, 2 at in=2048, 1 when tokens == 1); it exists for the main agent's sweep, and
 * nothing in the gated correctness path uses a value other than the launcher's own.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "proj_gemv_kernels.h"

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------- host side
static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    std::vector<char> b((size_t)f.tellg());
    f.seekg(0);
    f.read(b.data(), (std::streamsize)b.size());
    if ((size_t)f.gcount() != b.size()) {
        fprintf(stderr, "%s: short read\n", p.c_str());
        exit(1);
    }
    return b;
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, sep))
        if (!t.empty()) out.push_back(t);
    return out;
}

struct Weight {                       // one launch's concatenated weight blob, `tile` replicas
    std::string key;
    std::vector<std::string> files;
    int in = 0, rows = 0;
    size_t bytes = 0;
    std::vector<hip_bfloat16*> rep;
};

struct Job {
    std::string name, xfile, outfile;
    int widx = 0, xidx = 0;
    int rows = 0, in = 0, tpb = 1, tokens = 1;
    size_t wbytes = 0, xbytes = 0;
    // Per-launch variant override (plan field 7: `VAR=name[:rows[:threads]]`), so ONE plan can carry
    // the mixed geometry the sweep needs — big shapes on the shipping split, 16-row shapes on a narrow
    // one — instead of the launcher applying one variant to every job. -1/0 means "the run's own
    // --variant/--rows/--threads". Without this the only way to gate a mixed plan is one binary call
    // per variant, which multiplies the gate's hip init and upload cost by the number of variants.
    int variant = -1;
    int rpb = 0;
    int threads = -1;
    hip_bfloat16* dout = nullptr;
    // Traffic accounting for one launch of this job: weight bytes once per TPB-token tile, and each
    // distinct x / output element once per tile pass. Same level of approximation the k=1 form used
    // (x is in fact re-staged by every row block of the launch, and was counted once there too); the
    // divisor that matters — weights per token-tile instead of per token — is exact.
    size_t bytes() const {
        const size_t tiles = (size_t)((tokens + tpb - 1) / tpb);
        return wbytes * tiles + xbytes * tiles + (size_t)tokens * rows * 2;
    }
};

struct Plan {
    std::vector<Weight> weights;
    std::vector<float*> xdev;              // widened F32 x, one per distinct x file
    std::vector<hip_bfloat16*> xdev_bf;    // the same x left as BF16
    std::vector<std::string> xpaths;
    int tokens = 1;                        // rows in each XFILE tile; set from --tokens
    std::vector<Job> jobs;

    int weight_index(const std::vector<std::string>& files, int in) {
        std::string key;
        int rows = 0;
        size_t bytes = 0;
        for (const auto& f : files) {
            std::ifstream g(f, std::ios::binary | std::ios::ate);
            if (!g) {
                fprintf(stderr, "cannot open %s\n", f.c_str());
                exit(1);
            }
            const size_t n = (size_t)g.tellg();
            if (n % ((size_t)in * 2)) {
                fprintf(stderr, "%s: %zu bytes is not a whole number of bf16 rows of in=%d\n",
                        f.c_str(), n, in);
                exit(1);
            }
            key += f + "+";
            rows += (int)(n / ((size_t)in * 2));
            bytes += n;
        }
        for (int i = 0; i < (int)weights.size(); ++i)
            if (weights[i].key == key) return i;
        weights.push_back(Weight{});
        Weight& w = weights.back();
        w.key = key;
        w.files = files;
        w.in = in;
        w.rows = rows;
        w.bytes = bytes;
        return (int)weights.size() - 1;
    }

    int x_index(const std::string& path, int in) {
        for (int i = 0; i < (int)xpaths.size(); ++i)
            if (xpaths[i] == path) return i;
        const size_t want = (size_t)tokens * in * 2;
        const std::vector<char> raw = slurp(path);
        if (raw.size() != want) {
            fprintf(stderr, "%s: %zu bytes, expected %zu (%d bf16 rows of in=%d)\n", path.c_str(),
                    raw.size(), want, tokens, in);            exit(1);
        }
        std::vector<float> host(raw.size() / 2);
        const hip_bfloat16* p = (const hip_bfloat16*)raw.data();
        for (size_t i = 0; i < host.size(); ++i) host[i] = float(p[i]);
        float* d = nullptr;
        hip_bfloat16* db = nullptr;
        CK(hipMalloc(&d, host.size() * sizeof(float)));
        CK(hipMemcpy(d, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
        CK(hipMalloc(&db, raw.size()));
        CK(hipMemcpy(db, raw.data(), raw.size(), hipMemcpyHostToDevice));
        xpaths.push_back(path);
        xdev.push_back(d);
        xdev_bf.push_back(db);
        return (int)xpaths.size() - 1;
    }
};

// One launch of the tile form. TPB is a template argument, so the widths are resolved by
// `with_variant` below rather than reached through a runtime switch: a wrong width must not silently
// become a different kernel.

/*
 * The D1 variant table. ONE row per candidate kernel, and `lpw`/`vx` are read from it by the
 * launcher, the geometry print and the RESULT line, so a thread-count claim can never be typed by
 * hand in two places (the trap AGENTS.md records for a geometry fact that lives in two headers).
 *
 * `lpw == 0` is variant 0: the shipping `proj_gemv_tokens`, kept as the control and as the
 * reference every other variant is diffed against. `exact == false` means the form reassociates
 * the dot and is therefore judged by the k=1 gate's bands (no large element above 1 ULP inside the
 * cancellation band, ndiff inside the band, cos/meanrel), NOT bit-exactly. No band is widened:
 * the bands are the ones `floor_probe.py` sized for K = 1024/2048 and variant 0 already meets.
 */
struct Variant {
    const char* name;
    int lpw;              // lanes per output row; 0 = the shipping kernel
    bool vx;              // stage x with 16-byte loads
    int rpb;              // default rows per block
    bool exact;           // bit-identical to variant 0?
    const char* claim;    // the single difference this variant isolates
};
static const Variant g_var[] = {
    {"v0-baseline", 0, false, 8, true,
     "shipping form: one wavefront per row, runtime word bound (the control)"},
    {"v1-unroll", 32, false, 8, true,
     "lm_head's inner loop: compile-time word bound with #pragma unroll, nothing else changed"},
    {"v2-lpw16", 16, false, 8, false,
     "16 lanes per row: 2 rows per wavefront, 8 words per lane at K=1024, 4-step ladder"},
    {"v3-lpw8", 8, false, 16, false,
     "8 lanes per row: 4 rows per wavefront, 16 words per lane at K=1024, 3-step ladder"},
    {"v4-lpw4", 4, false, 32, false,
     "4 lanes per row: 8 rows per wavefront, 32 words per lane at K=1024, 2-step ladder"},
    {"v5-vx", 32, true, 8, true,
     "variant 1 plus 16-byte x staging; isolates the staging loop's load width"},
    {"v6-lpw8-vx", 8, true, 16, false,
     "the combination: 8 lanes per row AND 16-byte x staging"},
};
static const int kNumVariants = (int)sizeof(g_var) / sizeof(g_var[0]);
static const int kDefaultVariant = 0;      // the shipping form; the gate and the sweep default here

// The variant's own name for `--label NAME`, or its table name when --label is absent. The sweep
// prints this verbatim on every timing row, so a stale binary is caught by a label that does not
// match the variant it was asked for (AGENTS.md: assert a behavioural canary before timing).
static std::string variant_label(int v, const std::string& override_label) {
    return override_label.empty() ? std::string(g_var[v].name) : override_label;
}

// Resolves (IN, TPB, variant, PACK_X) to a kernel function pointer ONCE, and the same pointer is
// used to launch and to ask the driver for occupancy, so those two can never disagree. `f` gets
// the kernel plus its lanes-per-row, which is what sets threads per block.
template <int IN, int TPB, typename F>
static void with_variant(int variant, bool packed, F&& f) {
    switch (variant) {
        case 0:
            if (packed) f(&proj_gemv_tokens<IN, TPB, true>, 32);
            else f(&proj_gemv_tokens<IN, TPB, false>, 32);
            return;
#define PROJ_CASE(V, LPW, VX)                                                       \
        case V:                                                                     \
            if (packed) f(&proj_gemv_var<IN, TPB, LPW, true, VX>, LPW);             \
            else f(&proj_gemv_var<IN, TPB, LPW, false, VX>, LPW);                   \
            return;
            PROJ_CASE(1, 32, false)
            PROJ_CASE(2, 16, false)
            PROJ_CASE(3, 8, false)
            PROJ_CASE(4, 4, false)
            PROJ_CASE(5, 32, true)
            PROJ_CASE(6, 8, true)
#undef PROJ_CASE
        default:
            fprintf(stderr, "internal: variant %d is not instantiated at TPB=%d\n", variant, TPB);
            exit(1);
    }
}

// The kernel pointer for the occupancy query. TPB here is the launcher's own tile width, and the
// occupancy number printed is for the widest staged tile (in=2048), as before.
static const void* variant_kernel(int in, int tpb, int variant, bool packed) {
    const void* out = nullptr;
    auto take = [&](auto k, int) { out = (const void*)k; };
    if (in == 1024) {
        if (tpb == 1) with_variant<1024, 1>(variant, packed, take);
        else if (tpb == 4) with_variant<1024, 4>(variant, packed, take);
    } else {
        if (tpb == 1) with_variant<2048, 1>(variant, packed, take);
        else if (tpb == 2) with_variant<2048, 2>(variant, packed, take);
    }
    return out;
}

static void upload(Weight& w, int copies) {
    char* host = nullptr;
    CK(hipHostMalloc(&host, w.bytes));
    size_t off = 0;
    for (const auto& fn : w.files) {
        std::vector<char> raw = slurp(fn);
        memcpy(host + off, raw.data(), raw.size());
        off += raw.size();
    }
    for (int r = 0; r < std::max(1, copies); ++r) {
        hip_bfloat16* d = nullptr;
        CK(hipMalloc(&d, w.bytes));
        CK(hipMemcpy(d, host, w.bytes, hipMemcpyHostToDevice));
        w.rep.push_back(d);
    }
    CK(hipHostFree(host));
}

// Fixed-format numbers for the result lines. A point that is judged VOID must not print a millisecond
// figure at all, so the values are formatted to strings and the VOID arm substitutes the word — a
// `%.4f` with a sentinel number in it would still read as a measurement to whatever parses it.
static std::string fmt_num(double v, const char* f) {
    char b[32];
    std::snprintf(b, sizeof b, f, v);
    return std::string(b);
}

int main(int argc, char** argv) {
    std::string mode = "once", plan_path, negname = "none", rows_list, kernel = "tokens";
    std::string variant_name = "", label_override;
    int iters = 30, rows_per_block = -1, tile = 1, tokens = 1, tpb_flag = 0, reps = 1;
    // threads_per_block: 0 = "the variant's own", which is rpb*lanes_per_row — the value every number
    // in results.md was taken with, so the default moves nothing. See `--threads` in the usage block.
    int threads_per_block = 0, warm = 2;
    // --burn SECONDS: sustained untimed launches of the real plan on the real weights, timed out by
    // wall clock rather than by a launch count. A 13-second timed set never gets the clock to boost:
    // progress/2026-10-21/performance-status-2026-10-21.md §36 measured 819/827/838 MHz idle against
    // 2432/2437/2423 MHz immediately after a rep, so an unburned window ranks its arms by when they
    // ran rather than by what they do. This is the burn the sweep runs before arm 1.
    double burn_s = 0.0;
    // --preburn SECONDS: the same sustained load, run INSIDE the timed process immediately before the
    // reps. The window that measured this burned once before arm 1: the clock read 1218 MHz before the
    // burn, 2718 after it, and 1749 by the end of a 44-second set — one burn boosts the queue and then it
    // decays while the table runs, so late arms get a faster box than early ones. Burning before every
    // burst is what makes each row start from the same place.
    double preburn_s = 0.0;
    bool x_packed = false, write_out = false, dump_plan = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--rows") rows_per_block = atoi(next().c_str());
        else if (a == "--rows-list") rows_list = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--reps") reps = atoi(next().c_str());
        else if (a == "--threads" || a == "--tpr") threads_per_block = atoi(next().c_str());
        else if (a == "--burn") burn_s = atof(next().c_str());
        else if (a == "--preburn") preburn_s = atof(next().c_str());
        else if (a == "--warm") warm = atoi(next().c_str());
        else if (a == "--tile") tile = atoi(next().c_str());
        else if (a == "--tokens") tokens = atoi(next().c_str());
        else if (a == "--tpb") tpb_flag = atoi(next().c_str());
        else if (a == "--kernel") kernel = next();
        else if (a == "--variant") variant_name = next();
        else if (a == "--label") label_override = next();
        else if (a == "--dump-plan") dump_plan = true;
        else if (a == "--neg") negname = next();
        else if (a == "--x-packed") x_packed = true;
        else if (a == "--write-out") write_out = true;
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }
    // `--variant` accepts the table name or its index; a name that matches nothing is an error, not
    // a silent fall-back to the shipping form, and `--kernel k1` (the per-row reference) only ever
    // runs variant 0 because the point of that path is that it is a DIFFERENT kernel.
    int variant = kDefaultVariant;
    if (!variant_name.empty()) {
        variant = -1;
        for (int i = 0; i < kNumVariants; ++i) {
            if (variant_name == g_var[i].name) { variant = i; break; }
        }
        if (variant < 0) {
            char digits = variant_name[0];
            if (digits >= '0' && digits <= '9') variant = atoi(variant_name.c_str());
        }
        if (variant < 0 || variant >= kNumVariants) {
            fprintf(stderr, "--variant must be one of");
            for (int i = 0; i < kNumVariants; ++i) fprintf(stderr, " %d=%s", i, g_var[i].name);
            fprintf(stderr, " (got %s)\n", variant_name.c_str());
            return 1;
        }
    }
    if (kernel == "k1" && variant != 0) {
        fprintf(stderr, "--kernel k1 is the per-row reference kernel and only runs variant 0;"
                        " drop --variant or drop --kernel k1\n");
        return 1;
    }
    const int lpw = g_var[variant].lpw;             // 0 for the shipping kernel
    const bool vx = g_var[variant].vx;
    if (rows_per_block < 0) rows_per_block = g_var[variant].rpb;
    if (plan_path.empty()) {
        fprintf(stderr, "usage: proj_gemv --mode once|burst --plan PLAN [options]\n");
        return 1;
    }
    if (tokens < 1) {
        fprintf(stderr, "--tokens takes at least 1 token (got %d)\n", tokens);
        return 1;
    }
    if (kernel != "tokens" && kernel != "k1") {
        fprintf(stderr, "--kernel must be `tokens` (the tile form the engine runs) or `k1`\n");
        return 1;
    }
    // The launcher's own tile width, and the engine's: shared memory is TPB*IN floats and 16 KiB is
    // the budget the gated MLP settled on, so 4 rows at in=1024 and 2 at in=2048. One token is a tile
    // of one — that is what makes `tokens == 1` the k=1 shape rather than a special case.
    auto tile_width = [&](int in) {
        if (tpb_flag > 0) return tpb_flag;
        if (tokens == 1) return 1;
        return in == 1024 ? 4 : 2;
    };
    const int neg = negname == "none" ? NEG_NONE : negname == "tail" ? NEG_TAIL
                  : negname == "half" ? NEG_HALF : negname == "trunc" ? NEG_TRUNC
                  : negname == "rowswap" ? NEG_ROWSWAP : negname == "transpose" ? NEG_TRANSPOSE
                  : negname == "qhalf" ? NEG_QHALF : negname == "kvswap" ? NEG_KVS : -1;
    if (neg < 0) {
        fprintf(stderr, "--neg must be none|tail|half|trunc|rowswap|transpose|qhalf|kvswap\n");
        return 1;
    }
    if (mode != "once" && mode != "burst" && mode != "burn") {
        fprintf(stderr, "--mode must be once|burst|burn\n");
        return 1;
    }

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    const int ws = prop.warpSize;
    const int ncu = prop.multiProcessorCount;
    if (ws != 32 && ws != 64) {
        fprintf(stderr, "warpSize %d unsupported by the row-dot reduction\n", ws);
        return 1;
    }
    // The variant kernel reduces with the width-limited __shfl_down_sync, whose segments are
    // 32-lane-aligned; the shipping kernel's reduce_row is wavefront-generic. Refuse rather than
    // launch a variant whose per-row segment could straddle a wavefront on a wave64 target.
    if (lpw > 0 && ws != 32) {
        fprintf(stderr, "variant %s needs warpSize 32 (this device reports %d); variant 0 runs on"
                        " either\n", g_var[variant].name, ws);
        return 1;
    }
    const std::string label = variant_label(variant, label_override);

    Plan plan;
    plan.tokens = tokens;      // how many rows each XFILE carries (x_index checks against it)
    {
        std::ifstream pf(plan_path);
        if (!pf) {
            fprintf(stderr, "cannot open plan %s\n", plan_path.c_str());
            return 1;
        }
        std::string line;
        int ln = 0;
        while (std::getline(pf, line)) {
            ++ln;
            if (line.empty() || line[0] == '#') continue;
            auto tok = split(line, ' ');
            if (tok.size() < 5) {
                fprintf(stderr, "plan %s line %d: NAME ROWS IN XFILE WFILES [OUTFILE]\n",
                        plan_path.c_str(), ln);
                return 1;
            }
            Job j;
            j.name = tok[0];
            j.rows = atoi(tok[1].c_str());
            j.in = atoi(tok[2].c_str());
            j.xfile = tok[3];
            auto wfiles = split(tok[4], ',');
            if (tok.size() > 5) j.outfile = tok[5];
            if (tok.size() > 6) {
                // VAR=name[:rows[:threads]] — `name` must be a variant in the table; a bad name is an
                // error, never a silent fall-back.
                std::string vv = tok[6];
                if (vv.compare(0, 4, "VAR=") != 0) {
                    fprintf(stderr, "plan line %d: field 7 must be VAR=name[:rows[:threads]],"
                                    " got %s\n", ln, tok[6].c_str());
                    return 1;
                }
                vv = vv.substr(4);
                const size_t colon = vv.find(':');
                const std::string vname = colon == std::string::npos ? vv : vv.substr(0, colon);
                if (colon != std::string::npos) {
                    const std::string rest = vv.substr(colon + 1);
                    const size_t colon2 = rest.find(':');
                    j.rpb = atoi(rest.substr(0, colon2).c_str());
                    if (colon2 != std::string::npos)
                        j.threads = atoi(rest.substr(colon2 + 1).c_str());
                }
                j.variant = -1;
                for (int i = 0; i < kNumVariants; ++i)
                    if (vname == g_var[i].name) { j.variant = i; break; }
                if (j.variant < 0) {
                    fprintf(stderr, "plan line %d: VAR=%s is not a variant the binary has\n",
                            ln, vname.c_str());
                    return 1;
                }
            }
            if (j.in != 1024 && j.in != 2048) {
                fprintf(stderr, "plan line %d: in=%d, only 1024 and 2048 are instantiated\n",
                        ln, j.in);
                return 1;
            }
            j.widx = plan.weight_index(wfiles, j.in);
            // `--dump-plan` never touches the device, so it does not stage x: the canary that reads
            // a variant's launch geometry must be runnable while another agent holds the GPU lock.
            j.xidx = dump_plan ? -1 : plan.x_index(j.xfile, j.in);
            j.tokens = tokens;
            j.tpb = tile_width(j.in);
            j.xbytes = (size_t)tokens * j.in * 2;
            // Traffic accounting: see Job::bytes(). `tile` replicas are cache-defeats, not extra
            // work, so they are NOT counted per launch — the reported GB/s is the same bytes a real
            // token moves.
            j.wbytes = plan.weights[j.widx].bytes;
            plan.jobs.push_back(j);
        }
    }
    if (plan.jobs.empty()) {
        fprintf(stderr, "plan %s has no jobs\n", plan_path.c_str());
        return 1;
    }
    // The engine's load path: pinned host copy, then one hipMemcpy per blob. Weights are
    // resident for the whole run; nothing here models a per-token H2D (devices/: H2D is
    // 76-82 GB/s and would dominate if a token had to wait for it).
    // `--dump-plan` prints geometry and leaves: it never touches the device, so the sweep's canary
    // can run while another agent holds the lock. Weight upload is what needs the GPU.
    if (!dump_plan) {
        for (auto& w : plan.weights) upload(w, (mode == "burst" || mode == "burn") ? tile : 1);
        for (auto& j : plan.jobs) CK(hipMalloc(&j.dout, (size_t)j.tokens * j.rows * 2));
    }

    double wbytes = 0;
    for (const auto& w : plan.weights) wbytes += w.bytes;
    // The launch-plan echo. The D1 sweep's stale-binary canary reads `variant=`, `lanes_per_row=`
    // and `launches=` from here, so a binary that did not get rebuilt cannot print a clean null:
    // every field is derived from the parsed flags AND from the variant table, never typed.
    printf("plan %s | jobs=%zu launches=%zu distinct weights=%zu (%.1f MiB) | warpSize=%d CUs=%d"
           " | variant=%d %s lanes_per_row=%d vx=%d exact=%d | rows/block=%d | x=%s | tile=%d"
           " | kernel=%s tokens=%d\n",
           plan_path.c_str(), plan.jobs.size(), plan.jobs.size(), plan.weights.size(),
           wbytes / 1048576.0, ws, ncu, variant, label.c_str(), lpw, (int)vx,
           (int)g_var[variant].exact, rows_per_block,
           x_packed ? "bf16 in shared" : "f32 in shared", tile, kernel.c_str(), tokens);
    printf("variant %s claim: %s\n", g_var[variant].name, g_var[variant].claim);
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());
    if (dump_plan) {
        // Per-launch geometry for the sweep and for a human checking that a plan is the plan it
        // thinks it is: one line per job with the launch shape the launcher will actually emit,
        // including any per-job variant the plan file asked for.
        int distinct_variants = 0;
        unsigned seen = 0;
        for (const auto& j : plan.jobs) {
            const int vidx = j.variant >= 0 ? j.variant : variant;
            if (!(seen & (1u << vidx))) { seen |= (1u << vidx); ++distinct_variants; }
            const int lpr = g_var[vidx].lpw > 0 ? g_var[vidx].lpw : ws;
            const int rpb = j.rpb > 0 ? j.rpb : rows_per_block;
            // The same three-level resolution launch_job uses: job tag, then --threads, then the
            // variant's own rpb*lanes. The device-free canary has to read the geometry the run would
            // actually emit, not the default it would have emitted before the flag existed.
            const int req = j.threads > 0 ? j.threads : threads_per_block;
            const int thr = req > 0 ? req : rpb * lpr;
            const int blocks = (j.rows + rpb - 1) / rpb;
            // trips: how many passes a block makes over its own rows at this shape (rpb divided by
            // the rows one wavefront-group walks at a time). 1 is the welded shape of every published
            // number here; >1 is what a --threads value below rpb*lanes buys.
            const int trips = (rpb + thr / lpr - 1) / (thr / lpr);
            printf("LAUNCH %-16s variant=%-11s rows=%5d in=%4d rpb=%2d blocks=%4d threads=%4d"
                   " trips=%d wbytes=%8zu waves=%4d\n", j.name.c_str(), g_var[vidx].name, j.rows,
                   j.in, rpb, blocks, thr, trips, plan.weights[j.widx].bytes,
                   blocks * (thr / ws));
        }
        printf("RESULT mode=dump-plan label=%s variant=%d launches=%zu distinct_variants=%d"
               " rows_per_block=%d lanes_per_row=%d vx=%d exact=%d threads=%d\n", label.c_str(),
               variant, plan.jobs.size(), distinct_variants, rows_per_block, lpw, (int)vx,
               (int)g_var[variant].exact,
               threads_per_block > 0 ? threads_per_block
                                     : rows_per_block * (lpw > 0 ? lpw : ws));
        return 0;
    }

    auto launch_job = [&](Job& j, int rep, int rpb_req) {
        const Weight& w = plan.weights[j.widx];
        if (w.rows != j.rows) {
            fprintf(stderr, "%s: plan says %d rows, weight files have %d\n", j.name.c_str(),
                    j.rows, w.rows);
            exit(1);
        }
        // Per-job variant wins over the run's, and the job's own rows/block wins over --rows.
        const int vidx = j.variant >= 0 ? j.variant : variant;
        const bool jvx = g_var[vidx].vx;
        const int jlpw = g_var[vidx].lpw;
        const int lpr = jlpw > 0 ? jlpw : ws;
        const int rpb = j.rpb > 0 ? j.rpb : rpb_req;
        // Threads per block: the job's own `VAR=name:rows:threads` tag, else `--threads`, else the
        // variant's own `rpb * lanes_per_row` — which is the value every published number here was
        // taken with (5d6405f already computed `threads = rpb * ws`), so the default moves nothing.
        const int req = j.threads > 0 ? j.threads : threads_per_block;
        const int threads = req > 0 ? req : rpb * lpr;
        // Variant 0 IS the kernel `engine/kernels_proj.cpp` compiles, and that launcher emits exactly
        // `kRowsPerBlock * warp_width()` threads. Handing it a different block size would time a
        // geometry the engine cannot produce, and the number could not be carried to the shipping path
        // — so refuse instead of quietly measuring something unportable. The prototype-only variants
        // (1-6, `proj_gemv_var`) take `--threads` freely, and v1/v5 are bit-identical to v0, so the
        // shape search loses nothing by running there. Same rule for `--kernel k1`, the gate's
        // independent one-row reference, which has no row loop to walk several trips with.
        if (req > 0 && threads != rpb * lpr && kernel == "k1") {
            fprintf(stderr,
                    "%s: --kernel k1 is the gate's independent one-row reference and has no row"
                    " loop; it takes exactly rpb*lanes = %d threads, not %d\n",
                    j.name.c_str(), rpb * lpr, req);
            exit(1);
        }
        if (req > 0 && threads != rpb * lpr && vidx == 0) {
            fprintf(stderr,
                    "%s: --threads %d at rows/block %d is a shape variant 0 (%s) cannot launch:"
                    " it is the kernel the engine compiles, and its launcher emits exactly"
                    " rpb*lanes = %d threads. Use --variant v1-unroll or v5-vx, both bit-identical"
                    " to it, for a free thread count.\n",
                    j.name.c_str(), req, rpb, g_var[0].name, rpb * lpr);
            exit(1);
        }
        const int blocks = (j.rows + rpb - 1) / rpb;
        if (threads > prop.maxThreadsPerBlock) {
            fprintf(stderr, "%s: rows/block %d at %d threads > max %d\n",
                    j.name.c_str(), rpb, threads, prop.maxThreadsPerBlock);
            exit(1);
        }
        if (threads < lpr) {
            fprintf(stderr, "%s: %d threads cannot hold one row of %d lanes\n",
                    j.name.c_str(), threads, lpr);
            exit(1);
        }
        // Every row's lanes must be one aligned segment of one wavefront, or the width-limited
        // shuffle reads a neighbouring row's partial. A tail block's dead rows are clamped inside
        // the kernel, so only this alignment is a hard contract on the launch shape.
        if ((threads % ws) != 0) {
            fprintf(stderr,
                    "%s: rows/block %d at %d lanes/row = %d threads is not a whole number of"
                    " %d-lane wavefronts; use a rows/block that makes rpb*lanes a multiple of %d\n",
                    j.name.c_str(), rpb, lpr, threads, ws, ws);
            exit(1);
        }
        const void* xp = x_packed ? (const void*)plan.xdev_bf[j.xidx]
                                  : (const void*)plan.xdev[j.xidx];
        hip_bfloat16* W = w.rep[(size_t)rep % w.rep.size()];
        // grid.x is the row blocks, exactly as the k=1 form; grid.y is the token tiles, and is 1
        // whenever there is one token or one tile, which is why the decode shape is unchanged.
        const int tpb = j.tpb;
        const dim3 grid_t((unsigned)blocks, (unsigned)((j.tokens + tpb - 1) / tpb));
        if (kernel == "k1") {
            if (j.tokens != 1) {
                fprintf(stderr, "%s: --kernel k1 is the one-row-per-launch form; drop --tokens\n",
                        j.name.c_str());
                exit(1);
            }
            if (j.in == 1024) {
                if (x_packed) proj_gemv<1024, true><<<blocks, threads>>>(W, xp, j.dout, j.rows,
                                                                         rpb, neg);
                else proj_gemv<1024, false><<<blocks, threads>>>(W, xp, j.dout, j.rows, rpb, neg);
            } else {
                if (x_packed) proj_gemv<2048, true><<<blocks, threads>>>(W, xp, j.dout, j.rows,
                                                                         rpb, neg);
                else proj_gemv<2048, false><<<blocks, threads>>>(W, xp, j.dout, j.rows, rpb, neg);
            }
            CK(hipGetLastError());
            return;
        }
        // The forms the engine could run, at every width. TPB=1 with grid.y=1 is the k=1 arithmetic.
        // TPB is a template argument, so only the widths the launcher actually picks are instantiated:
        // 1 (the decode shape) plus the tile width for that input width. Anything else is refused
        // rather than silently mapped onto a different width than the one asked for.
        //
        // The variants are the DECODE forms (TPB=1): one token per launch, which is what the D1
        // attribution costs. The prefill tile widths stay on variant 0, the kernel the prefill
        // campaign measured and shipped, so a decode experiment can never move a prefill number.
        if (vidx != 0 && tpb != 1) {
            fprintf(stderr,
                    "%s: variant %s is the decode (TPB=1) form; the token-tile widths stay on"
                    " variant 0 (drop --tokens/--tpb, or run --variant v0-baseline)\n",
                    j.name.c_str(), g_var[vidx].name);
            exit(1);
        }
        const bool packed = x_packed;
        (void)jvx;
        if (j.in == 1024) {
            if (tpb == 1) with_variant<1024, 1>(vidx, packed, [&](auto k, int) {
                k<<<grid_t, threads>>>(W, xp, j.dout, j.rows, j.tokens, rpb, neg); });
            else if (tpb == 4) with_variant<1024, 4>(vidx, packed, [&](auto k, int) {
                k<<<grid_t, threads>>>(W, xp, j.dout, j.rows, j.tokens, rpb, neg); });
            else {
                fprintf(stderr, "--tpb %d is not instantiated at in=1024 (1 or 4)\n", tpb);
                exit(1);
            }
        } else {
            if (tpb == 1) with_variant<2048, 1>(vidx, packed, [&](auto k, int) {
                k<<<grid_t, threads>>>(W, xp, j.dout, j.rows, j.tokens, rpb, neg); });
            else if (tpb == 2) with_variant<2048, 2>(vidx, packed, [&](auto k, int) {
                k<<<grid_t, threads>>>(W, xp, j.dout, j.rows, j.tokens, rpb, neg); });
            else {
                fprintf(stderr, "--tpb %d is not instantiated at in=2048 (1 or 2)\n", tpb);
                exit(1);
            }
        }
        CK(hipGetLastError());
    };

    auto write_outputs = [&]() {
        for (auto& j : plan.jobs) {
            if (j.outfile.empty()) continue;
            const size_t n = (size_t)j.tokens * j.rows;      // [token][row], the tile layout
            std::vector<hip_bfloat16> h(n);
            CK(hipMemcpy(h.data(), j.dout, n * 2, hipMemcpyDeviceToHost));
            std::ofstream of(j.outfile, std::ios::binary);
            of.write((const char*)h.data(), (std::streamsize)(n * 2));
        }
    };

    if (mode == "once") {
        int rep = 0;
        for (auto& j : plan.jobs) launch_job(j, rep++, rows_per_block);
        CK(hipDeviceSynchronize());
        write_outputs();
        printf("RESULT mode=once label=%s variant=%d jobs=%zu launches=%zu neg=%s"
               " rows_per_block=%d lanes_per_row=%d vx=%d x_packed=%d\n",
               label.c_str(), variant, plan.jobs.size(), plan.jobs.size(), negname.c_str(),
               rows_per_block, lpw, (int)vx, (int)x_packed);
        return 0;
    }

    // Sustained untimed load on THIS process's plan and weights, timed out by wall seconds (so it costs
    // the same on a faster machine). Shared by `--mode burn`, the burn that opens a window, and `--preburn`,
    // the burn immediately before each timed burst: one burn at the start of a 44-second set leaves the
    // queue decaying through it — the clock-controlled window read 1218 MHz before its burn, 2718 after,
    // and 1749 at the end — so every row needs its own. Returns (plans, launches, elapsed).
    auto burn_for = [&](double seconds, int rpb) {
        const auto bt = std::chrono::steady_clock::now();
        size_t plans = 0, launches = 0, rep = 0;
        double elapsed = 0.0;
        do {
            for (auto& j : plan.jobs) { launch_job(j, (int)rep++, rpb); ++launches; }
            CK(hipDeviceSynchronize());
            ++plans;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - bt).count();
        } while (elapsed < seconds);
        return std::make_tuple(plans, launches, elapsed);
    };

    if (mode == "burn") {
        // The window-opening burn: launch the plan until `--burn` wall seconds have passed, then report how
        // much work that was and nothing else. NO millisecond figure is printed here on purpose — a burn
        // number reads like a measurement and is not one; its whole job is to get the clock off its idle
        // floor (§36: 819/827/838 MHz idle, 2432/2437/2423 MHz right after a rep).
        if (burn_s <= 0.0) {
            fprintf(stderr, "--mode burn needs --burn SECONDS > 0\n");
            return 1;
        }
        auto br = burn_for(burn_s, rows_per_block);
        printf("RESULT mode=burn label=%s variant=%d rpb=%d plans=%zu launches=%zu wall_s=%.3f\n",
               label.c_str(), variant, rows_per_block, std::get<0>(br), std::get<1>(br), std::get<2>(br));
        return 0;
    }

    std::vector<int> cfgs;
    for (const auto& s : rows_list.empty() ? std::vector<std::string>{std::to_string(
             rows_per_block)} : split(rows_list, ','))
        cfgs.push_back(atoi(s.c_str()));

    double plan_bytes = 0;
    for (const auto& j : plan.jobs) plan_bytes += (double)j.bytes();

    for (int rpb : cfgs) {
        // Burst: submit the whole plan `iters` times, one sync at the end, wall-clock timed.
        // hipEventElapsedTime returns exactly 0 below ~1 ms on this iGPU (trap in
        // prototype/README.md), and a burst also keeps the host's ~1.85 us/launch submit cost
        // off the measured time. Repeating the whole plan — not one kernel — is what keeps the
        // working set cold: a real token touches each of these weights exactly once.
        auto burst = [&](int n) {
            hipEvent_t s, e;
            CK(hipEventCreate(&s));
            CK(hipEventCreate(&e));
            CK(hipEventRecord(s));
            const auto t0 = std::chrono::steady_clock::now();
            size_t counter = 0;
            for (int i = 0; i < n; ++i)
                for (auto& j : plan.jobs) launch_job(j, (int)(counter++), rpb);
            CK(hipEventRecord(e));
            CK(hipDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            float ev = 0.f;
            CK(hipEventElapsedTime(&ev, s, e));
            CK(hipEventDestroy(s));
            CK(hipEventDestroy(e));
            double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
            // Wall-clock is the number. The event pair is printed alongside as a cross-check
            // only: on this iGPU an event window around short work can read ~0 (trap #7), and a
            // min() of the two lets a broken timer masquerade as an infinitely fast kernel -
            // which is exactly how one shape here once reported 25433 GB/s.
            return std::pair<double, double>(wall, (double)ev);
        };
        // Per-arm burn, in this process, immediately before the timed reps: sustained load for
        // `--preburn` seconds on the same uploaded plan. Without it the clock decays across the set and a
        // late arm is measured on a faster box than an early one; with it every row starts boosted and the
        // LOWCLK gate has a chance of being met. The burst clock reads below are taken around this.
        size_t pre_plans = 0, pre_launches = 0;
        double pre_elapsed = 0.0;
        if (preburn_s > 0.0) {
            auto pr = burn_for(preburn_s, rpb);
            pre_plans = std::get<0>(pr); pre_launches = std::get<1>(pr); pre_elapsed = std::get<2>(pr);
        }
        // Shared bytes per block for the widest case (in=2048 at its tile width) and the occupancy
        // the kernel THIS variant resolved gets. With the tile form this is TPB*IN*(2 or 4) bytes,
        // which is exactly the quantity that bounds TPB, so it is printed rather than assumed. The
        // kernel handle comes from the same resolution the launch used (AGENTS.md: a geometry fact
        // held in two places is committed in one of them).
        const int occ_x = tile_width(2048) * 2048 * (x_packed ? 2 : 4);
        const int lpr = lpw > 0 ? lpw : ws;
        const int cfg_threads = threads_per_block > 0 ? threads_per_block : rpb * lpr;
        int occ = 0;
        const void* k = variant_kernel(2048, tile_width(2048), variant, x_packed);
        if (k == nullptr) {
            fprintf(stderr, "occupancy probe: no kernel for in=2048 tpb=%d variant=%d\n",
                    tile_width(2048), variant);
            exit(1);
        }
        CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, cfg_threads, 0));
        // Warm BEFORE timing, every timed point, and enough passes to matter: the weights are freshly
        // uploaded, and the first GPU pass over fresh pages runs ~35 GB/s against the ~237 this probe
        // measures (devices/strix-halo-unified-memory.md). One warm pass was not enough — a rep of one
        // config here read 17.497 ms where its own min read 3.544 ms, i.e. 26 GB/s, which is the
        // fresh-page rate wearing the costume of a measurement.
        for (int w = 0; w < std::max(0, warm); ++w) burst(1);
        // `--reps` bursts the same uploaded plan N times so the sweep can state a spread WITHOUT
        // re-uploading 466 MB per repeat (an upload is ~2 s and would dwarf a 90 ms burst). One
        // invocation, one upload, N independent timed bursts of `iters` plans each; the spread is
        // then the honest run-to-run number, not the within-burst average.
        const int nrep = std::max(1, reps);
        const int nit = std::max(1, iters);
        std::vector<double> rep_ms;
        std::pair<double, double> t(0.0, 0.0);
        for (int r = 0; r < nrep; ++r) {
            t = burst(nit);
            rep_ms.push_back(t.first / nit);
            printf("    rep %-3d %8.4f ms/plan\n", r, rep_ms.back());
        }
        const double ev_plan = t.second / nit;
        const double ms_min = [&]{ double m = rep_ms[0]; for (double x : rep_ms) m = std::min(m, x); return m; }();
        const double ms_max = [&]{ double m = rep_ms[0]; for (double x : rep_ms) m = std::max(m, x); return m; }();
        // Best-of-N is this repo's convention (prototype/README.md: timings are best-of-N on an
        // unpinned idle desktop), so the REPORTED number is the min — not the last rep, which is what
        // `ms_per_plan` used to be and how a 5x-slow rep ended up printed as the result.
        const double ms_plan = ms_min;
        const double ms_launch = ms_plan / (double)plan.jobs.size();
        const double gbs = plan_bytes / (ms_plan / 1e3) / 1e9;
        // A point is not allowed to report a millisecond when its own reps disagree that much: a
        // max/min above 2 means something else was on the device or the pages never warmed, and a
        // plausible-looking mean of a bimodal sample is worse than no number. `provisional` (1.10) is
        // the level at which performance-status.md §6 says a row cannot be adjudicated across windows.
        const double ratio = ms_min > 0 ? ms_max / ms_min : 0.0;
        const char* pt = ratio > 2.0 ? "void" : (ratio > 1.10 ? "provisional" : "ok");
        const bool voidpt = ratio > 2.0;
        const char* evnote = (ev_plan <= 0.0 || ev_plan > 1.5 * ms_plan || ev_plan < 0.67 * ms_plan)
                                 ? " (events disagree: treat as unmeasured)" : "";
        printf("  %-14s rows/block %-3d lanes/row %-3d threads %-5d %8s ms/plan %8s GB/s"
               " (%5s of the %.0f GB/s probe) | spread %.4f-%.4f ms over %zu reps (max/min %.3f)"
               " | warm %d | %s | events %.4f ms/plan%s | %.1f MiB/plan"
               " shared %d B/block, %d blocks/CU occupancy\n",
               label.c_str(), rpb, lpr, cfg_threads,
               voidpt ? "VOID" : fmt_num(ms_plan, "%.4f").c_str(),
               voidpt ? "VOID" : fmt_num(gbs, "%.1f").c_str(),
               voidpt ? "VOID" : fmt_num(100.0 * gbs / PROBE_GBS, "%.0f%%").c_str(), PROBE_GBS,
               ms_min, ms_max, rep_ms.size(), ratio, warm, pt, ev_plan, evnote,
               plan_bytes / 1048576.0, occ_x, occ);
        // One line per distinct (rows, in) in the plan: blocks and waves are the block-count
        // argument the frozen ledger's §5 item 5 asks about (progress/2026-10-16/performance-status-2026-10-16.md:240: 20 CUs, not the 40 rocminfo prints).
        for (size_t i = 0; i < plan.jobs.size(); ++i) {
            bool seen = false;
            for (size_t k2 = 0; k2 < i; ++k2)
                if (plan.jobs[k2].rows == plan.jobs[i].rows && plan.jobs[k2].in == plan.jobs[i].in)
                    seen = true;
            if (seen) continue;
            const Job& j = plan.jobs[i];
            const int blocks = (j.rows + rpb - 1) / rpb;
            printf("    shape [%6d,%4d] blocks=%5d waves=%5d waves/CU=%5.1f  %.2f MiB/launch"
                   "  %.1f us/launch at this rate\n",
                   j.rows, j.in, blocks, blocks * rpb, (double)blocks * rpb / ncu,
                   plan.weights[j.widx].bytes / 1048576.0, ms_launch * 1e3);
        }
        printf("RESULT mode=burst label=%s variant=%d rpb=%d lanes_per_row=%d vx=%d threads=%d"
               " jobs=%zu launches=%zu preburn_s=%.2f preburn_plans=%zu preburn_launches=%zu"
               " ms_per_plan=%s ms_min=%.6f ms_max=%.6f reps=%zu warm=%d"
               " ratio=%.3f pt=%s us_per_launch=%s gbs=%s probe_pct=%s x_packed=%d tile=%d neg=%s\n",
               label.c_str(), variant, rpb, lpw, (int)vx, cfg_threads, plan.jobs.size(),
               plan.jobs.size(), pre_elapsed, pre_plans, pre_launches, voidpt ? "VOID" : fmt_num(ms_plan, "%.6f").c_str(), ms_min, ms_max,
               rep_ms.size(), warm, ratio, pt,
               voidpt ? "VOID" : fmt_num(ms_launch * 1e3, "%.3f").c_str(),
               voidpt ? "VOID" : fmt_num(gbs, "%.2f").c_str(),
               voidpt ? "VOID" : fmt_num(100.0 * gbs / PROBE_GBS, "%.2f").c_str(),
               (int)x_packed, tile, negname.c_str());
    }
    if (write_out) {                     // replica 0 after timing: the timed config's own output
        for (auto& j : plan.jobs) launch_job(j, 0, rows_per_block);
        CK(hipDeviceSynchronize());
        write_outputs();
    }
    return 0;
}
