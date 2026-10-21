// Does hipBLASLt contain a faster kernel for THIS engine's matmul shapes than the one its default
// heuristic picks? The honest headroom on the library is 1.4-1.6x (devices/strix-halo-compute.md
// second-correction table, 33.2-37.7 TFLOP/s, against prototype/issue_rate_probe/results.md §1's
// 52.63 TFLOP/s hand-loop ceiling; decision log D17), and before anyone builds a hand WMMA GEMM to
// chase it, the cheap question is whether the library's own catalogue already has the win.
//
// THREE MODES, and the split is the point:
//   --list      host only. Prints the shape census read out of engine/*.cpp. Opens no device.
//   --verify    builds every descriptor and prints the whole algorithm enumeration with ZERO timing
//               and ZERO operand allocation, so the plan can be confirmed before anyone hands this
//               tool a GPU lock. hipblasLtCreate() is the only call that touches the device
//               (context + device properties); no matmul is launched and no operand buffer exists.
//   --time      the measurement run: correctness per algorithm FIRST, then paired timing. This is
//               the main agent's mode (AGENTS.md: no worker-owned timing), it is not fast, and it
//               does NOT take tools/gpu_lock.sh itself — HANDOFF.md §2 says who holds it.
//
// WHY hipBLASLt AND NOT hipBLAS. The engine's own call is `hipblasGemmEx` through
// `gemm_common::gemm_stored_weight` (engine/kernels_gemm.cpp:171, tools/gemm_common.h:119), which
// takes the kernel choice away from the caller: `HIPBLAS_GEMM_DEFAULT`, and that is all. hipBLASLt
// is the same vendor library with the choice exposed, so it is the instrument for asking "is the
// default leaving anything on the table". The engine's own call is timed too, as the per-shape
// `engine_call_ms` column, so a ratio can be attributed: hipBLASLt's rank-0 heuristic answer is not
// automatically the kernel hipBLAS picked.
//
// WHAT THIS BUILD CANNOT ENUMERATE — a fact about the library, not about this tool. The
// cuBLASLt-style enumeration path (AlgoGetIds / AlgoInit / AlgoCapGetAttribute /
// AlgoConfigGetAttribute / AlgoConfigSetAttribute) does not exist here: none of those symbols is in
// /opt/rocm/lib/libhipblaslt.so and none is declared in /opt/rocm/include/hipblaslt/*.h, so calling
// one is a link error, not a runtime refusal. The only algorithm entry point in the C API is
// `hipblasLtMatmulAlgoGetHeuristic`, and `hipblasLtMatmulAlgo_t` is 16 opaque bytes with no
// accessors — there is no tile/splitK/occupancy field to print. Enumeration does exist in the C++
// extension namespace: hipblaslt_ext::getAllAlgos + getIndexFromAlgo + getSolutionNameFromAlgo +
// getKernelNameFromAlgo + matmulIsAlgoSupported. That is what this tool uses, so the per-algorithm
// identity it can print is algo index, solution name, kernel name, workspace required, wavesCount
// and the raw 16 opaque bytes. README.md §3 carries the nm/grep evidence.
//
// CORRECTNESS BEFORE SPEED, twice, because a timing of a wrong kernel is worse than no timing:
//   1. every algorithm that will be timed is first run once and judged against a host dot, in double,
//      of the values the device was actually given, rounded to bf16 — the same reference and
//      comparator the repo already uses (gemm_common::compare_16bit_output, as at
//      tools/gemm_peak.cpp:261 and tools/prefill_mix.cpp:656). An algorithm outside the band prints
//      WRONG and is never timed; if the DEFAULT-ranked algorithm is outside it the whole shape is
//      refused, because then the harness rather than the kernel is what is broken.
//   2. once the winner is known, default and winner are re-run into separate buffers and compared to
//      each other, byte for byte and in ULP, before RESULT is printed.
// ALIGNMENT, said once and meant everywhere below: every buffer here is row-major [M][N] with
// ld = N, so element (t, j) lives at t*N + j on every side — device output, host reference, and the
// default-vs-winner pair alike. There is no index offset (this is not §10.5's pin pairing, where a
// decode sample at step s pairs with greedy[s+1]); the only index transform in the file is the
// column-major BLAS view of the same row-major buffer, and a wrong ld/OP_T pair cannot survive the
// reference band — tools/gemm_common.h:111-126 is where that pair is documented as silent when wrong.
//
// The reference dot costs nsamp*N*K, so it runs on --check-rows sampled rows of M (first and last
// always included): tools/prefill_mix.cpp:667's precedent for the same reason. The filler-survivor
// check ("beta=0 must have written every element", the tools/prefill_mix.cpp:656 discipline, with
// the 0xFF prefill it uses) reads all of D; --scan-filler 0 narrows it to the sampled rows and
// prints `scan=partial` so the weakened check is visible in the line that weakened it.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include "tools/gemm_common.h"

namespace {

// The shapes the engine actually issues. No square in here and nothing invented: one row per
// distinct (N, K) the library route reaches, with the call site that reaches it.
//
// The engine's one library form is `gemm_bf16(weight, x, out, m, n, k)` where m is the wave's token
// count, n the weight's row count and k its feature count, so a shape is (n, k) plus a wave width.
// Leaves that share a shape (k_proj/v_proj, gate_proj/up_proj, o_proj/out_proj) are ONE row here,
// because the tool times a shape and not a weight; `per_wave` keeps the census honest and its sum
// is 150 — the same 150-call 24-layer mix devices/strix-halo-compute.md's second correction prices,
// which is the witness that nothing in the family list was dropped (--list asserts it).
struct EngineShape {
    const char *name;     // what the engine calls the leaf(s)
    const char *family;   // the launch_stats family: proj | mlp_gateup | mlp_down
    const char *site;     // file:line of the engine call that issues it
    int n;                // output features = C rows = BLAS m
    int k;                // reduction
    int per_wave;         // how many such calls in one 24-layer wave
};

const EngineShape kShapes[] = {
    {"q_proj", "proj", "engine/decode_step.cpp:583", 4096, 1024, 6},
    {"k_proj+v_proj", "proj", "engine/decode_step.cpp:586,589", 512, 1024, 12},
    {"o_proj+out_proj", "proj", "engine/decode_step.cpp:810", 1024, 2048, 24},
    {"in_proj_qkv", "proj", "engine/decode_step.cpp:596", 6144, 1024, 18},
    {"in_proj_z", "proj", "engine/decode_step.cpp:599", 2048, 1024, 18},
    {"gate_proj+up_proj", "mlp_gateup",
     "engine/kernels_mlp.cpp:262,263 <- decode_step.cpp:748", 3584, 1024, 48},
    {"down_proj", "mlp_down", "engine/kernels_mlp.cpp:276 <- decode_step.cpp:762", 1024, 3584, 24},
};
// NOT here, with the reason said out loud:
//   in_proj_a / in_proj_b (N=16) — `proj_gemm` refuses them (engine/kernels_proj.cpp:134-141) and
//       they stay in the hand GEMV at every width (§7 W4: 12-22x the dispatch floor there).
//   lm_head (248320 x 1024) — not a library call at all. `lm_head_full` (engine/kernels_lmhead.cpp:25)
//       is a hand GEMV over an F32 activation and runs once per request, not once per wave row.
//       engine/performance-status.md §3 lists `lm_head` as a wave FAMILY, which is true of the hand
//       kernel; nothing routes it to hipBLAS, so sweeping it here would price a call not made.
//   a body of one row — `resolve_wave_gemm` sends M=1 to the hand GEMV whatever the run asked for
//       (engine/kernels_gemm.cpp:844-850), so M>=2 is the library's domain and --rows refuses 1.

struct Options {
    enum Mode { kList, kVerify, kTime, kSelftestLayout, kSelftestSys, kFloorProbe };
    Mode mode = kList;
    int requested = 32;           // heuristic request count, well above 1 on purpose
    int reps = 5;                 // timed reps per algorithm
    int warm = 2;                 // untimed calls per algorithm before the reps
    int inner = 1;                // launches per timed sample (burst; HANDOFF.md §3, sync floor)
    int max_algos = 0;            // 0 = every algorithm the build will give
    int check_rows = 16;          // sampled rows of M for the host reference dot
    bool scan_filler = true;      // read all of D looking for 0xFFFF filler survivors
    bool engine_ref = true;       // also time the engine's own hipblasGemmEx call per shape
    double max_ulp = 1.0;         // stated band, part 1: worst ULP off the rounded reference. NEVER
                                  // relaxed by flag default - see HANDOFF.md §5
    // Band, part 2: aggregate cosine over the sampled rows. There is deliberately NO constant here.
    // The only acceptable sources are a floor probe measured on this build (calibrate()) or an
    // explicit --cos-floor the operator typed (cos_floor_given). A bare inherited constant is what
    // refused all 7 shapes on the first device window.
    double cos_floor = 0.0;
    bool cos_floor_given = false;
    std::string band_provenance = "unresolved";
    double floor_margin = 1.0e-05;  // cos_floor = measured_worst_cos - this; stated on every line
    // Band, part 3: the cancellation half of the per-element test. Same discipline as the cos floor -
    // no inherited constant. cancel_rel is the |ref|/max|ref| at or below which a >1-ULP element is
    // treated as a cancellation; cancel_tol is the absolute tolerance (as a fraction of max|ref|) such
    // an element must satisfy instead. Both come from the calibration's license leg; an explicit
    // override is allowed for A/B and is named as an override wherever it prints.
    double cancel_rel = 0.0, cancel_tol = 0.0;
    bool cancel_rel_given = false, cancel_tol_given = false;
    double cancel_margin = 4.0;    // theta = measured worst legal offender x this; mlp_gemv used 3.7x
    double fullmag_rel = 1.0e-03;  // an offender above this fraction of max|ref| is NOT a cancellation
    int seeds = 4;                  // --floor-probe: operand fills per (shape, M)
    std::string calibration;        // floor_calibration.tsv; empty until resolve_band sets the path
    std::string calibration_out;    // --floor-probe writes the measured floors here
    std::string program = "gemm_algo_sweep";
    size_t ws_bytes = 128ull << 20;  // engine/kernels_gemm.cpp:60 pins 128 MiB; same here
    bool check_only = false;      // correctness pass, then stop: no timing at all (worker-safe)
    int algo_seq = -1;            // run exactly one candidate (implies --check-only): fault isolation
    bool trace_launch = false;    // print the full argument list of every launch, not just the warm one
    std::string algo_dump;        // --verify: write the captured algos here for --selftest-sys
    std::string algo_file;        // --selftest-sys: read them back
    std::vector<int> rows;        // wave widths (M) to enumerate
    std::vector<std::string> shapes;
};

std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t c = s.find(',', p);
        const std::string tok = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
        if (!tok.empty()) out.push_back(tok);
        if (c == std::string::npos) break;
        p = c + 1;
    }
    return out;
}

std::vector<int> parse_ints(const std::string &s) {
    std::vector<int> out;
    for (const auto &t : split_csv(s)) out.push_back(std::atoi(t.c_str()));
    return out;
}

void usage() {
    std::printf(
        "gemm_algo_sweep - enumerate hipBLASLt's algorithms at the engine's own matmul shapes\n"
        "\n"
        "  --list                shape census only; opens no device\n"
        "  --verify              descriptors + enumeration; no timing, no operand allocation\n"
        "  --time                correctness then timing; main agent's mode, lock held OUTSIDE\n"
        "  --rows 512,1024,4096  wave widths (M = tokens per call). 1 is refused: M<=1 is the\n"
        "                        hand GEMV by rule, not preference\n"
        "  --shapes q_proj,...   subset by the names --list prints (default: all)\n"
        "  --requested 32        algorithms to ask the heuristic for\n"
        "  --reps 5 --warm 2     timed reps / warm-up calls per algorithm\n"
        "  --inner 1             launches per timed sample; 4-8 is the honest setting below\n"
        "                        ~100 us, where the per-call sync floor is inside the number\n"
        "  --max-algos 0         cap the timed list per shape (0 = all enumerated)\n"
        "  --check-rows 16       rows of M dotted on the host (the dot costs nsamp*N*K)\n"
        "  --scan-filler 0|1     1 = read all of D for 0xFFFF survivors\n"
        "  --engine-ref 0|1      also time the engine's own hipblasGemmEx call, for attribution\n"
        "  --check-only          --time's correctness pass and nothing else: zero timing, so it is\n"
        "                        safe for a worker to run (no ms/tflops line exists in this mode)\n"
        "  --algo-seq N          exactly the candidate with seq N, implies --check-only. One process\n"
        "                        per candidate is the only real fault isolation: a faulting kernel\n"
        "                        poisons the context and every later launch in that process dies\n"
        "  --trace-launch 0|1    1 = print ARGS for every launch, not only the warm-up one\n"
        "  --algo-dump FILE      with --verify: save every enumerated algo (16 raw bytes as hex,\n"
        "                        state, support, workspace) so --selftest-sys can audit them offline\n"
        "  --algo-file FILE      with --selftest-sys: the file --algo-dump wrote\n"
        "  --selftest-layout     host-only: descriptor/buffer pairing invariants, no HIP call\n"
        "  --selftest-sys        host-only: rebuild every hipblasLtMatmul argument list from an\n"
        "                        --algo-dump capture and run the launch site's predicates on it\n"
        "  --max-ulp 1.0         the ULP half of the band. Never calibrated, never loosened by a\n"
        "                        default: a device leg past 1 ULP off the rounded reference is a bug\n"
        "  --cos-floor X         the aggregate half. There is NO built-in constant: without this flag\n"
        "                        the floor comes from floor_calibration.tsv, which only --floor-probe\n"
        "                        can write, and a missing row aborts before any device work. Passing\n"
        "                        this is an operator override and every line says so (README.md §5)\n"
        "  --floor-probe         measure the floor: per (shape, M) it sweeps operand seeds and three\n"
        "                        summation orders, prices three must-catch injections, and writes the\n"
        "                        calibration file. Needs a device; takes no timing\n"
        "  --seeds 4             operand fills per (shape, M) in --floor-probe\n"
        "  --floor-margin 1e-05  cos_floor = measured worst cos minus this\n"
        "  --cancel-margin 4.0   cancel_rel/tol = measured worst legal offender times this\n"
        "  --cancel-rel X        override the cancellation threshold (A/B only; named as an override)\n"
        "  --cancel-tol X        override the cancelled-element absolute tolerance\n"
        "  --fullmag-rel 1e-03   an offender above this share of max|ref| is a finding, not a "
        "cancellation\n"
        "  --calibration FILE    read floor_calibration.tsv from here (default: beside the binary)\n"
        "  --calibration-out FILE  where --floor-probe writes it\n"
        "  --workspace-mib 128   max workspace advertised to the heuristic\n"
        "\n"
        "No mode of this tool acquires tools/gpu_lock.sh.\n");
}

bool parse_args(int argc, char **argv, Options &o) {
    o.rows = {512, 1024, 4096};
    if (argc > 0 && argv[0]) o.program = argv[0];
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                std::printf("ARG_ERROR %s needs a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--list") o.mode = Options::kList;
        else if (a == "--verify") o.mode = Options::kVerify;
        else if (a == "--time") o.mode = Options::kTime;
        else if (a == "--rows") o.rows = parse_ints(next("--rows"));
        else if (a == "--shapes") o.shapes = split_csv(next("--shapes"));
        else if (a == "--requested") o.requested = std::atoi(next("--requested"));
        else if (a == "--reps") o.reps = std::atoi(next("--reps"));
        else if (a == "--warm") o.warm = std::atoi(next("--warm"));
        else if (a == "--inner") o.inner = std::atoi(next("--inner"));
        else if (a == "--max-algos") o.max_algos = std::atoi(next("--max-algos"));
        else if (a == "--check-rows") o.check_rows = std::atoi(next("--check-rows"));
        else if (a == "--scan-filler") o.scan_filler = std::atoi(next("--scan-filler")) != 0;
        else if (a == "--engine-ref") o.engine_ref = std::atoi(next("--engine-ref")) != 0;
        else if (a == "--check-only") o.check_only = true;
        else if (a == "--algo-seq") o.algo_seq = std::atoi(next("--algo-seq"));
        else if (a == "--trace-launch") o.trace_launch = std::atoi(next("--trace-launch")) != 0;
        else if (a == "--algo-dump") o.algo_dump = next("--algo-dump");
        else if (a == "--algo-file") o.algo_file = next("--algo-file");
        else if (a == "--selftest-layout") o.mode = Options::kSelftestLayout;
        else if (a == "--selftest-sys") o.mode = Options::kSelftestSys;
        else if (a == "--max-ulp") o.max_ulp = std::atof(next("--max-ulp"));
        else if (a == "--cos-floor") {
            o.cos_floor = std::atof(next("--cos-floor"));
            o.cos_floor_given = true;
        }
        else if (a == "--floor-margin") o.floor_margin = std::atof(next("--floor-margin"));
        else if (a == "--cancel-margin") o.cancel_margin = std::atof(next("--cancel-margin"));
        else if (a == "--cancel-rel") {
            o.cancel_rel = std::atof(next("--cancel-rel"));
            o.cancel_rel_given = true;
        } else if (a == "--cancel-tol") {
            o.cancel_tol = std::atof(next("--cancel-tol"));
            o.cancel_tol_given = true;
        } else if (a == "--fullmag-rel") o.fullmag_rel = std::atof(next("--fullmag-rel"));
        else if (a == "--seeds") o.seeds = std::atoi(next("--seeds"));
        else if (a == "--calibration") o.calibration = next("--calibration");
        else if (a == "--calibration-out") o.calibration_out = next("--calibration-out");
        else if (a == "--floor-probe") o.mode = Options::kFloorProbe;
        else if (a == "--workspace-mib")
            o.ws_bytes = (size_t)std::atoll(next("--workspace-mib")) << 20;
        else if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else {
            std::printf("ARG_ERROR unknown flag '%s'\n", a.c_str());
            usage();
            return false;
        }
    }
    // Value checks run before any mode early-return, so a flag that was typed is never silently
    // ignored because the chosen mode happens not to use it. `--requested 0` alone used to print a
    // clean --list and exit 0; a flag accepted and dropped is how a wrong run looks like a good one.
    if (o.requested < 1) { std::printf("ARG_ERROR --requested must be >= 1\n"); return false; }
    if (o.rows.empty()) { std::printf("ARG_ERROR --rows is empty\n"); return false; }
    for (int r : o.rows) {
        if (r < 2) {
            std::printf("ARG_ERROR --rows takes 2..N: M=1 is the hand GEMV whatever the run asked "
                        "for (engine/kernels_gemm.cpp:844-850), so the library has no one-row shape\n");
            return false;
        }
    }
    // A --shapes name that matches nothing would leave the whole sweep with zero problems and every
    // verdict line printed over an empty list - a check that can have no inputs cannot fail
    // (AGENTS.md, the gdn_collapse run.py lesson). Names are therefore checked against the table,
    // here, before any device exists.
    for (const auto &want : o.shapes) {
        bool known = false;
        for (const auto &s : kShapes)
            if (want == s.name) known = true;
        if (!known) {
            std::printf("ARG_ERROR --shapes names '%s', which is not a shape. Known:", want.c_str());
            for (const auto &s : kShapes) std::printf(" %s", s.name);
            std::printf("\n");
            return false;
        }
    }
    if (o.mode == Options::kList && !o.shapes.empty()) {
        std::printf("ARG_ERROR --shapes is not honoured by --list, which prints the whole census\n");
        return false;
    }
    if (o.mode == Options::kSelftestSys && !o.shapes.empty()) {
        std::printf("ARG_ERROR --shapes is not honoured by --selftest-sys, whose input is the "
                    "--algo-file capture\n");
        return false;
    }
    if (!o.algo_file.empty() && o.mode != Options::kSelftestSys) {
        std::printf("ARG_ERROR --algo-file is read only by --selftest-sys\n");
        return false;
    }
    if (!o.algo_dump.empty() && o.mode != Options::kVerify) {
        std::printf("ARG_ERROR --algo-dump is written only by --verify\n");
        return false;
    }
    if (o.algo_seq >= 0 && o.mode != Options::kTime) {
        std::printf("ARG_ERROR --algo-seq selects a candidate from a --time run\n");
        return false;
    }
    if (o.algo_seq >= 0) o.check_only = true;  // one candidate cannot answer a ratio question
    if (o.check_only && o.mode != Options::kTime) {
        std::printf("ARG_ERROR --check-only needs --time as well; it is the timing pass with the "
                    "timing removed, not a variant of --verify\n");
        return false;
    }
    if (o.mode == Options::kTime && !o.check_only) {
        if (o.reps < 2) { std::printf("ARG_ERROR --reps must be >= 2 (best-of-N needs a pair)\n"); return false; }
        if (o.inner < 1) { std::printf("ARG_ERROR --inner must be >= 1\n"); return false; }
        if (o.check_rows < 2) { std::printf("ARG_ERROR --check-rows must be >= 2 (first AND last)\n"); return false; }
    }
    // The band's two halves, checked wherever numerics will be judged. The ULP half is a bound and
    // stays one; the aggregate half has no built-in constant at all, so the only question here is
    // whether a typed override is even a number that could mean something.
    if (o.max_ulp <= 0.0) { std::printf("ARG_ERROR --max-ulp must be > 0\n"); return false; }
    if (o.cos_floor_given && !(o.cos_floor > 0.0 && o.cos_floor < 1.0)) {
        std::printf("ARG_ERROR --cos-floor must be in (0,1); got %g. 1.0 would demand bit-identity "
                    "with a double reference, which no bf16 chain can give\n", o.cos_floor);
        return false;
    }
    if (o.floor_margin <= 0.0 || o.floor_margin >= 0.01) {
        std::printf("ARG_ERROR --floor-margin must be in (0,0.01); got %g. It is the gap between the "
                    "measured floor and the floor, so a margin of 0 asks the next run to reproduce the "
                    "measured worst cos exactly\n", o.floor_margin);
        return false;
    }
    // The margin is what turns a measurement into a licence, so an enormous one is the same failure as
    // an inherited constant: a band that forgives everything while claiming to be measured. 10x is
    // already wider than any margin this repo has had to use (mlp_gemv: 3.7x).
    if (!(o.cancel_margin >= 1.0 && o.cancel_margin <= 10.0)) {
        std::printf("ARG_ERROR --cancel-margin must be in [1,10]; got %g. It multiplies the measured "
                    "worst legal offender, and above ~10 it is a widened band wearing a measurement's "
                    "clothes\n", o.cancel_margin);
        return false;
    }
    if (!(o.fullmag_rel > 0.0 && o.fullmag_rel < 1.0)) {
        std::printf("ARG_ERROR --fullmag-rel must be in (0,1); got %g. It is the share of max|ref| "
                    "above which a >1-ULP element cannot be called a cancellation\n", o.fullmag_rel);
        return false;
    }
    if (o.cancel_rel_given && !(o.cancel_rel > 0.0 && o.cancel_rel < 1.0)) {
        std::printf("ARG_ERROR --cancel-rel must be in (0,1); got %g\n", o.cancel_rel);
        return false;
    }
    if (o.cancel_tol_given && !(o.cancel_tol > 0.0)) {
        std::printf("ARG_ERROR --cancel-tol must be > 0 (a fraction of max|ref|); got %g\n",
                    o.cancel_tol);
        return false;
    }
    if (o.mode == Options::kFloorProbe) {
        if (o.seeds < 1) { std::printf("ARG_ERROR --seeds must be >= 1\n"); return false; }
        // The row_alias injection compares row s to row s+1, and the floor itself is a MINIMUM over
        // sampled rows; both are meaningless on one row.
        if (o.check_rows < 2) { std::printf("ARG_ERROR --floor-probe needs --check-rows >= 2\n"); return false; }
        // The probe's product IS the calibration file. A probe that measures and writes nothing leaves
        // the next run with no floor, which is the state this mode exists to end.
        if (o.calibration_out.empty()) {
            std::printf("ARG_ERROR --floor-probe needs --calibration-out FILE: its output is the file "
                        "every later --time run reads its cos floor from, e.g. --calibration-out "
                        "prototype/gemm_algo_sweep/floor_calibration.tsv\n");
            return false;
        }
    }
    if (!o.calibration_out.empty() && o.mode != Options::kFloorProbe) {
        std::printf("ARG_ERROR --calibration-out is written only by --floor-probe\n");
        return false;
    }
    if (!o.calibration.empty() && o.mode != Options::kTime) {
        std::printf("ARG_ERROR --calibration is read only by --time; --floor-probe WRITES the "
                    "calibration (--calibration-out)\n");
        return false;
    }
    if (o.mode == Options::kList) return true;
    if (o.mode == Options::kSelftestLayout || o.mode == Options::kSelftestSys) return true;
    return true;
}

// -----------------------------------------------------------------------------------------------
// The descriptor set for one (M, N, K) problem, in the engine's own form.
//
// The derivation is tools/gemm_common.h:111-126's and is not re-decided here: BLAS is column-major,
// the checkpoint stores the weight [n, k] row-major, and C_rm[m,n] = A_rm[m,k] * W_rm[n,k]^T means
// opA=T on W with lda=k, opB=N on A with ldb=k, and C/D of n rows with ldc=n. Getting the pair
// backwards does not error, it computes a different product — which is why the band in the header
// comment runs before a single millisecond is believed.
// -----------------------------------------------------------------------------------------------
struct Plan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    int m = 0, n = 0, k = 0;
    // The dims each layout was created WITH, recorded at creation. invariants() re-derives the operand
    // extents from THESE numbers rather than from the shape table, so the check reads the same facts
    // the launch does (the mixed_bw plan_bounds lesson: a host check that re-reads the write-up
    // instead of the struct proves nothing).
    int ra = 0, ca = 0, lda = 0, rb = 0, cb = 0, ldb = 0, rc = 0, cc = 0, ldc_ = 0;
    // The element types and compute type each object was created with, assigned at the creation site
    // rather than defaulted, so validate_args() checks what was built instead of what was intended.
    int dtA = 0, dtB = 0, dtC = 0, dtD = 0, compute = 0, scale = 0;
    // Column-major extent of a layout is ld * cols, in elements.
    size_t need_A() const { return (size_t)lda * (uint64_t)ca; }
    size_t need_B() const { return (size_t)ldb * (uint64_t)cb; }
    size_t need_C() const { return (size_t)ldc_ * (uint64_t)cc; }

    // Records the operand geometry. build() calls this before any descriptor is created and
    // --selftest-layout calls the very same function, so the host check reads the numbers the launch
    // will use instead of a second copy of the recipe. Names say which buffer each descriptor
    // describes: A is the WEIGHT, B is the ACTIVATION, because that inversion is what this file got
    // wrong once (README.md section 1a).
    void set_dims(int m_, int n_, int k_) {
        m = m_; n = n_; k = k_;
        ra = k; ca = n; lda = k;    // A = W, stored [n,k] row-major, OP_T
        rb = k; cb = m; ldb = k;    // B = X, the activation [m,k] row-major, OP_N
        rc = n; cc = m; ldc_ = n;   // C/D = column-major [n,m] view of row-major C_rm[m,n]
    }

    // Swap A and B behind each other's descriptor: the shape of the bug that shipped. An extent check
    // sees it exactly when n != m; when n == m the two buffers are the same size and only the
    // numerical band can see it, which is what HANDOFF.md section 4 says to read if LAYOUT is clean
    // but CHECK lines are WRONG.
    void swap_AB() { std::swap(ra, rb); std::swap(ca, cb); std::swap(lda, ldb); }

    // The forcing: each descriptor's required extent must equal the buffer actually passed for that
    // role. Returns false and fills `why` otherwise, so a bad pairing is a printed REFUSE taken
    // before a single kernel is launched, not an illegal access inside a locked window.
    bool invariants(size_t x_elems, size_t w_elems, size_t c_elems, std::string &why) const {
        why.clear();
        if (need_A() != w_elems)
            why += "A_desc=" + std::to_string(ra) + "x" + std::to_string(ca) + "_ld_" +
                   std::to_string(lda) + "_needs_" + std::to_string(need_A()) +
                   "_weight_elements_but_W_buffer_holds_" + std::to_string(w_elems) + "; ";
        if (need_B() != x_elems)
            why += "B_desc=" + std::to_string(rb) + "x" + std::to_string(cb) + "_ld_" +
                   std::to_string(ldb) + "_needs_" + std::to_string(need_B()) +
                   "_activation_elements_but_X_buffer_holds_" + std::to_string(x_elems) + "; ";
        if (need_C() != c_elems)
            why += "C_desc=" + std::to_string(rc) + "x" + std::to_string(cc) + "_ld_" +
                   std::to_string(ldc_) + "_needs_" + std::to_string(need_C()) +
                   "_output_elements_but_C_buffer_holds_" + std::to_string(c_elems) + "; ";
        return why.empty();
    }


    bool build(hipblasLtHandle_t lt, int m_, int n_, int k_, size_t ws_bytes, std::string &why) {
        (void)lt;
        set_dims(m_, n_, k_);
        compute = (int)HIPBLAS_COMPUTE_32F;
        scale = (int)HIP_R_32F;
        hipblasStatus_t s = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        if (s != HIPBLAS_STATUS_SUCCESS) { why = "MatmulDescCreate status=" + std::to_string((int)s); return false; }
        const int32_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
        // Column-major views of row-major buffers, exactly the pair gemm_stored_weight uses. Every
        // extent below comes out of set_dims(), the same writer --selftest-layout reads.
        s = hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, (uint64_t)ra, (uint64_t)ca, (int64_t)lda);
        dtA = (int)HIP_R_16BF;
        if (s != HIPBLAS_STATUS_SUCCESS) { why = "Adesc status=" + std::to_string((int)s); return false; }
        s = hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, (uint64_t)rb, (uint64_t)cb, (int64_t)ldb);
        dtB = (int)HIP_R_16BF;
        if (s != HIPBLAS_STATUS_SUCCESS) { why = "Bdesc status=" + std::to_string((int)s); return false; }
        s = hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, (uint64_t)rc, (uint64_t)cc, (int64_t)ldc_);
        dtC = (int)HIP_R_16BF;
        if (s != HIPBLAS_STATUS_SUCCESS) { why = "Cdesc status=" + std::to_string((int)s); return false; }
        s = hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, (uint64_t)rc, (uint64_t)cc, (int64_t)ldc_);
        dtD = (int)HIP_R_16BF;
        if (s != HIPBLAS_STATUS_SUCCESS) { why = "Ddesc status=" + std::to_string((int)s); return false; }
        s = hipblasLtMatmulPreferenceCreate(&pref);
        if (s != HIPBLAS_STATUS_SUCCESS) { why = "PreferenceCreate status=" + std::to_string((int)s); return false; }
        const uint64_t w = (uint64_t)ws_bytes;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &w, sizeof(w));
        return true;
    }
    void free() {
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (la) hipblasLtMatrixLayoutDestroy(la);
        if (lb) hipblasLtMatrixLayoutDestroy(lb);
        if (lc) hipblasLtMatrixLayoutDestroy(lc);
        if (ld) hipblasLtMatrixLayoutDestroy(ld);
        if (desc) hipblasLtMatmulDescDestroy(desc);
        pref = nullptr; la = lb = lc = ld = nullptr; desc = nullptr;
    }
};

// One candidate algorithm, with everything this build is willing to say about it.
struct AlgoEntry {
    hipblasLtMatmulAlgo_t algo{};
    int seq = 0;             // index in the printed list
    int heur_rank = -1;      // position in the heuristic answer, -1 = enumeration-only
    bool is_default = false; // its 16 bytes equal heuristic[0]'s
    int idx = -1;            // hipblaslt_ext::getIndexFromAlgo
    std::string solution, kernel;
    size_t ws = 0;
    size_t heur_ws = 0;
    float waves = 0.0f;
    int state = 0;
    bool supported = false;
    int support_status = -1;
};

bool same_algo(const hipblasLtMatmulAlgo_t &a, const hipblasLtMatmulAlgo_t &b) {
    return std::memcmp(a.data, b.data, 16) == 0;
}

std::string algo_bytes(const hipblasLtMatmulAlgo_t &a) {
    uint64_t w0 = 0, w1 = 0;
    std::memcpy(&w0, a.data, 8);
    std::memcpy(&w1, a.data + 8, 8);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%016llx:%016llx", (unsigned long long)w0,
                  (unsigned long long)w1);
    return buf;
}

struct Enumeration {
    std::vector<AlgoEntry> algos;      // heuristic first (in rank order), then enum-only
    int heur_status = -1;
    int heur_returned = 0;
    int enum_status = -1;
    size_t enum_returned = 0;
    int default_seq = -1;
};

// The heuristic ranking plus, if the build answers, the full catalogue. The heuristic list comes
// first and keeps its ranks: "the default" IS heuristic[0], and every other entry is described
// relative to it.
Enumeration enumerate(hipblasLtHandle_t lt, const Plan &plan, const Options &o,
                      const float &alpha, const float &beta) {
    Enumeration en;
    std::vector<hipblasLtMatmulHeuristicResult_t> heur((size_t)o.requested);
    int returned = 0;
    const hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(
        lt, plan.desc, plan.la, plan.lb, plan.lc, plan.ld, plan.pref, o.requested,
        heur.data(), &returned);
    en.heur_status = (int)hs;
    en.heur_returned = (hs == HIPBLAS_STATUS_SUCCESS) ? returned : 0;
    for (int i = 0; i < en.heur_returned; ++i) {
        AlgoEntry e;
        e.algo = heur[(size_t)i].algo;
        e.heur_rank = i;
        e.ws = heur[(size_t)i].workspaceSize;
        e.heur_ws = heur[(size_t)i].workspaceSize;
        e.waves = heur[(size_t)i].wavesCount;
        e.state = (int)heur[(size_t)i].state;
        en.algos.push_back(e);
    }

    // The full enumeration: absent from the C API on this build (README.md §3), present here.
    std::vector<hipblasLtMatmulHeuristicResult_t> all;
    const hipblasStatus_t as = hipblaslt_ext::getAllAlgos(
        lt, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF,
        HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F, all);
    en.enum_status = (int)as;
    en.enum_returned = all.size();
    for (const auto &h : all) {
        bool seen = false;
        for (const auto &e : en.algos)
            if (same_algo(e.algo, h.algo)) { seen = true; break; }
        if (seen) continue;
        AlgoEntry e;
        e.algo = h.algo;
        e.ws = h.workspaceSize;
        e.heur_ws = h.workspaceSize;
        e.waves = h.wavesCount;
        e.state = (int)h.state;
        en.algos.push_back(e);
    }
    if (!en.algos.empty() && en.algos[0].heur_rank == 0) {
        en.algos[0].is_default = true;
        en.default_seq = 0;
    }

    for (size_t i = 0; i < en.algos.size(); ++i) {
        AlgoEntry &e = en.algos[i];
        e.seq = (int)i;
        e.idx = hipblaslt_ext::getIndexFromAlgo(e.algo);
        e.solution = hipblaslt_ext::getSolutionNameFromAlgo(lt, e.algo);
        e.kernel = hipblaslt_ext::getKernelNameFromAlgo(lt, e.algo);
        if (e.solution.empty()) e.solution = "(unnamed)";
        if (e.kernel.empty()) e.kernel = "(unnamed)";
        // matmulIsAlgoSupported is problem-specific and may rewrite the algo struct with the
        // problem saved inside it; that is the documented flow, and the rewritten struct is what
        // gets launched below.
        size_t ws = 0;
        const hipblasStatus_t ss = hipblaslt_ext::matmulIsAlgoSupported(
            lt, plan.desc, &alpha, plan.la, plan.lb, &beta, plan.lc, plan.ld, e.algo, ws);
        e.support_status = (int)ss;
        e.supported = (ss == HIPBLAS_STATUS_SUCCESS);
        if (e.supported) e.ws = ws;
    }
    return en;
}

void print_enumeration(const EngineShape &sh, int m, const Enumeration &en, const Options &o) {
    std::printf("PLAN tool=gemm_algo_sweep shape=%s family=%s M=%d N=%d K=%d calls_per_wave=%d "
                "site=%s opA=T lda=%d opB=N ldb=%d ldc=%d dt=A:bf16/B:bf16/C:bf16/D:bf16 "
                "compute=32F scale=32F ws_max=%zu\n",
                sh.name, sh.family, m, sh.n, sh.k, sh.per_wave, sh.site, sh.k, sh.k, sh.n,
                o.ws_bytes);
    std::printf("HEUR shape=%s M=%d call=hipblasLtMatmulAlgoGetHeuristic status=%d requested=%d "
                "returned=%d\n", sh.name, m, en.heur_status, o.requested, en.heur_returned);
    std::printf("ENUM shape=%s M=%d call=hipblaslt_ext::getAllAlgos status=%d returned=%zu "
                "distinct_after_dedup=%zu\n", sh.name, m, en.enum_status, en.enum_returned,
                en.algos.size());
    for (const auto &e : en.algos) {
        std::printf("ALGO shape=%s M=%d seq=%d heur_rank=%d default=%d idx=%d ws=%zu heur_ws=%zu "
                    "waves=%.3f state=%d supported=%d support_status=%d data=%s solution=%s "
                    "kernel=%s\n",
                    sh.name, m, e.seq, e.heur_rank, e.is_default ? 1 : 0, e.idx, e.ws, e.heur_ws,
                    e.waves, e.state, e.supported ? 1 : 0, e.support_status,
                    algo_bytes(e.algo).c_str(), e.solution.c_str(), e.kernel.c_str());
    }
    int supported = 0, fit = 0;
    for (const auto &e : en.algos) {
        if (!e.supported) continue;
        ++supported;
        if (e.ws <= o.ws_bytes) ++fit;
    }
    std::printf("PLAN-SUMMARY shape=%s M=%d heuristic_returned=%d enumerated=%zu distinct=%zu "
                "supported=%d fits_workspace=%d default_seq=%d default_present_in_enum=%d "
                "(\"default\" = heuristic rank 0; a line carries default=1 iff its 16 opaque bytes "
                "equal heuristic[0]'s)\n",
                sh.name, m, en.heur_returned, en.enum_returned, en.algos.size(), supported, fit,
                en.default_seq, en.default_seq >= 0 ? 1 : 0);
}

// Sampled rows of M, first and last always included. tools/prefill_mix.cpp:667's precedent: the
// host dot costs nsamp*N*K, so it cannot run over all of M.
// -----------------------------------------------------------------------------------------------
// The host reference, spelled seven ways.
//
// Why more than one: the aggregate the band is built on (cos of a K-term dot against a device bf16
// chain) is a property of the SUMMATION ORDER, not of the arithmetic - the repo's F4 finding
// (prototype/mlp_gemv/results.md F4, prototype/glue/floor_probe.py). If the reference is moved by a
// legal reordering, then a band set tighter than that movement is measuring the reference. The probe
// mode measures exactly that, and these three orders are what it moves it by:
//   mode 0  serial i = 0..k-1           what every run has always used; unchanged, so old band
//                                       numbers stay reproducible and the probe's floor is the floor
//                                       of the reference the gate actually uses
//   mode 1  serial, reversed
//   mode 2  four strided partials, then folded - the shape a real SIMT kernel reduces with
//   mode 3  the first k/2 terms only    NOT a legal reordering: a must-catch injection, a kernel
//                                       that silently drops half the accumulation - the shape of a
//                                       two-way split-K whose second partial never gets added
//   mode 4  the first k-1 terms only     REPORTED, NOT a must-catch any more: see the RESOLUTION
//                                       line in probe_problem. At K=1024..3584 it moves cos by about
//                                       as much as bf16 output quantisation does, so it is below the
//                                       resolution of the aggregate that gates this op.
//   mode 5  serial, accumulated in F32  THE LICENSE LEG. HIPBLAS_COMPUTE_32F means the kernel is
//   mode 6  four strided F32 partials   entitled to accumulate in F32 rather than exactly; what that
//                                       licenses is what these two measure, and the cancellation band
//                                       is sized from them (NOT from modes 1-2, which the first probe
//                                       window showed move nothing: ndiff=0, worst_ulp=0).
//   mode 7  the first 7k/8 terms        the gentlest REACHABLE partial loss: one of eight split-K
//                                       partials dropped, which is what an eighth-partial defect
//                                       looks like (mode 3 is the two-partial version).
// -----------------------------------------------------------------------------------------------
void reference_row(const float *fx, const float *fw, int n, int k, int mode, double *out) {
    for (int j = 0; j < n; ++j) {
        const float *pw = fw + (size_t)j * k;
        double acc = 0.0;
        if (mode == 0) {
            for (int i = 0; i < k; ++i) acc += (double)fx[i] * (double)pw[i];
        } else if (mode == 1) {
            for (int i = k - 1; i >= 0; --i) acc += (double)fx[i] * (double)pw[i];
        } else if (mode == 2) {
            double p[4] = {0.0, 0.0, 0.0, 0.0};
            int i = 0;
            for (; i + 4 <= k; i += 4)
                for (int q = 0; q < 4; ++q)
                    p[q] += (double)fx[i + q] * (double)pw[i + q];
            for (; i < k; ++i) p[0] += (double)fx[i] * (double)pw[i];
            acc = ((p[0] + p[1]) + (p[2] + p[3]));
        } else if (mode == 4) {  // one term short: reported as below cos resolution at this K
            const int km1 = k - 1;
            for (int i = 0; i < km1; ++i) acc += (double)fx[i] * (double)pw[i];
        } else if (mode == 5) {  // F32 accumulation, serial: the license of a 32F compute type
            float a = 0.0f;
            for (int i = 0; i < k; ++i) a += (float)((double)fx[i] * (double)pw[i]);
            acc = a;
        } else if (mode == 6) {  // F32 accumulation in four strided partials, then folded
            float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int i = 0;
            for (; i + 4 <= k; i += 4)
                for (int q = 0; q < 4; ++q)
                    p[q] += (float)((double)fx[i + q] * (double)pw[i + q]);
            for (; i < k; ++i) p[0] += (float)((double)fx[i] * (double)pw[i]);
            acc = ((double)(p[0] + p[1]) + (double)(p[2] + p[3]));
        } else if (mode == 7) {  // seven of eight split-K partials
            const int seven = (k / 8) * 7;
            for (int i = 0; i < seven; ++i) acc += (double)fx[i] * (double)pw[i];
        } else {  // mode 3: half the chain, a must-catch injection and not a legal reordering
            const int half = k / 2;
            for (int i = 0; i < half; ++i) acc += (double)fx[i] * (double)pw[i];
        }
        out[j] = acc;
    }
}

std::vector<int> sample_rows(int m, int want) {
    const int c = std::max(1, std::min(m, want));
    std::vector<int> r((size_t)c);
    for (int i = 0; i < c; ++i)
        r[(size_t)i] = (c == 1) ? 0 : (int)((long long)i * (m - 1) / (c - 1));
    return r;
}

struct BandResult {
    size_t ndiff = 0;
    size_t n = 0;
    double worst_ulp = 0.0;      // max over EVERY element: printed always, no longer the predicate alone
    // The scale-aware split (prototype/mlp_gemv/run.py's gate, prototype/bf16_ulp.py:band_usage).
    size_t n_off = 0;            // elements more than the ULP bound off the rounded reference
    size_t n_off_in_band = 0;    // ... of which |ref|/max|ref| <= cancel_rel: cancellations
    size_t n_off_out_band = 0;   // ... the rest: these are the only ones that can fail the band
    double worst_off_rel = 0.0;  // largest |ref|/max|ref| among offenders = the tightest band that
                                 // these numbers would pass at, which is what mlp_gemv prints as
                                 // "tightest" and what sizes cancel_rel
    double worst_out_band_ulp = 0.0;  // worst ULP among offenders that did NOT cancel
    size_t off_index = 0;        // flat index (row-major within the sampled window) of that offender
    double off_rel = 0.0, off_ulp = 0.0, off_ref = 0.0, off_got = 0.0;
    double worst_in_band_excess = 0.0;  // max |delta|/max|ref| among in-band elements: cancel_tol's
    double worst_excess_rel = 0.0;      // same metric over every element, for the write-up
    double cos = 1.0;
    double meanrel = 0.0;
    size_t survivors = 0;
    size_t n_nonfinite = 0;   // elements of the device output that are NaN or inf: never a pass
    uint64_t hash = 0;
    bool ok = false;
    bool launched = false;   // false = never ran, so its cos is the 1.0 initialiser, not a measurement
    std::string why;
};

// -----------------------------------------------------------------------------------------------
// Why the per-element test is scale-aware rather than a bigger max-ULP.
//
// This file's own first probe window: the engine's own hipblasGemmEx call came back worst_ulp=2.0 on
// in_proj_qkv and 6.0 on gate_proj+up_proj, with meanrel still 1.4e-03 and cos identical to the
// correct leg to nine digits. One element in ~3e4, at a magnitude a flat bound cannot distinguish from
// a wrong product, so a flat `worst_ulp <= 1` refuses correct arithmetic and a flat `<= 6` would admit
// real bugs. mlp_gemv hit the same wall and measured its way out rather than widening: raw max ULP 24
// against "gated max ULP: 0 over tol", because every offender sat at ~1e-6 of the tensor max, and the
// fix was a band on rel = |ref| / max|ref| (CANCEL_REL 3e-5, sized against three measurements with a
// 3.7x margin - prototype/mlp_gemv/results.md, "How wide the cancellation band is"; the comparator
// form is prototype/bf16_ulp.py:band_usage, whose worst_off_rel is the same quantity this function
// calls worst_off_rel). F4 there goes further and drops the per-element rule entirely at K=3584 with a
// chained input; here the reference is a fresh double dot rather than a chained act, so the split
// below is the right strength: 1 ULP where an element still means something, an absolute tolerance
// where it does not.
//
// Judge one device output against the host reference. `raw` is the whole D buffer (row-major,
// ld = N); `rows` are the sampled token rows; `ref` holds one contiguous slice of n doubles per
// sampled row, in the order of `rows`. The aggregate metrics exist because max-ULP is blind to a
// whole-tensor 1-ULP shift and ndiff is meaningless for a long dot (AGENTS.md trap row; the
// prototype/mlp_gemv/floor_probe.py lesson), so a chain is judged by cos/meanrel as well.
// -----------------------------------------------------------------------------------------------
BandResult judge(const uint8_t *raw, size_t total_elems, const std::vector<int> &rows, int n,
                 const std::vector<double> &ref, const Options &o) {
    BandResult b;
    b.launched = true;
    // The scale the band is written against: max|ref| over the sampled window. Same definition as
    // rel_of() in prototype/mlp_gemv/floor_probe.py, so a cancel_rel measured here is comparable to
    // the 3e-5 measured there.
    double refmax = 0.0;
    for (size_t i = 0; i < (size_t)rows.size() * (size_t)n; ++i)
        refmax = std::max(refmax, std::fabs(ref[i]));
    if (!(refmax > 0.0)) refmax = 1.0;
    double meanrel_acc = 0.0;
    size_t flat = 0;  // index within the sampled window, so a printed index is findable again
    for (size_t s = 0; s < rows.size(); ++s) {
        const uint8_t *slice = raw + (size_t)rows[s] * (size_t)n * 2;
        const double *rslice = ref.data() + s * (size_t)n;
        const auto cmp = gemm_common::compare_16bit_output(slice, rslice, (size_t)n, true);
        b.ndiff += cmp.ndiff;
        b.n += (size_t)n;
        b.worst_ulp = std::max(b.worst_ulp, cmp.worst_ulp);
        double dot = 0, na = 0, nb = 0, rel = 0;
        bool row_nonfinite = false;
        for (int j = 0; j < n; ++j, ++flat) {
            uint16_t bits;
            std::memcpy(&bits, slice + (size_t)j * 2, 2);
            const double got = gemm_common::dec16(bits, true);
            const double want = rslice[j];
            if (!std::isfinite(got)) {
                // A NaN in the output must not be folded into the aggregate silently: NaN compares
                // false against every bound, and na > 0 is false for NaN, so the row cosine would
                // otherwise fall back to its 1.0 "no data" value and report agreement. The
                // beta_left_at_1_nan negative control caught exactly this: it printed cos=1.000000.
                ++b.n_nonfinite;
                row_nonfinite = true;
            }
            // The comparison that matters for a ULP count is against the ROUNDED reference: an exact
            // bf16 value 1 step away from the true dot is correct arithmetic, not a defect.
            const double rounded = gemm_common::dec16(gemm_common::enc16((float)want, true), true);
            const double abserr = std::fabs(got - rounded);
            dot += got * want;
            na += got * got;
            nb += want * want;
            if (std::fabs(want) > 1e-6) rel += std::fabs(got - want) / std::fabs(want);
            // ULP step at `rounded`, spelled exactly as compare_16bit_output spells it
            // (tools/gemm_common.h:150), so the two ULP numbers in this file cannot drift apart.
            const uint16_t wb = gemm_common::enc16((float)std::fabs(rounded), true);
            const double step =
                std::max(std::fabs((double)gemm_common::dec16((uint16_t)(wb + 1), true) -
                                   (double)gemm_common::dec16(wb, true)),
                         1e-30);
            const double ulp = abserr / step;
            const double relmag = std::fabs(rounded) / refmax;
            const double excess = abserr / refmax;
            b.worst_excess_rel = std::max(b.worst_excess_rel, excess);
            if (ulp > o.max_ulp) {
                ++b.n_off;
                b.worst_off_rel = std::max(b.worst_off_rel, relmag);
                if (relmag <= o.cancel_rel) {
                    ++b.n_off_in_band;  // a cancellation: the ULP denominator is meaningless here
                } else {
                    ++b.n_off_out_band;
                    if (ulp > b.worst_out_band_ulp) {
                        b.worst_out_band_ulp = ulp;
                        b.off_index = flat;
                        b.off_rel = relmag;
                        b.off_ulp = ulp;
                        b.off_ref = rounded;
                        b.off_got = got;
                    }
                }
            }
            // Every element inside the band contributes to its absolute tolerance, offender or not:
            // cancel_tol is "how far may a cancelled element sit from the rounded reference", and
            // measuring it only on the non-offenders would understate what a correct kernel does.
            if (relmag <= o.cancel_rel)
                b.worst_in_band_excess = std::max(b.worst_in_band_excess, excess);
        }
        const double row_cos =
            row_nonfinite ? 0.0 : ((na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 1.0);
        b.cos = std::min(b.cos, row_cos);
        meanrel_acc += rel / (double)n;
    }
    b.meanrel = meanrel_acc / (double)std::max<size_t>(1, rows.size());
    if (o.scan_filler) {
        for (size_t i = 0; i < total_elems; ++i) {
            uint16_t v;
            std::memcpy(&v, raw + i * 2, 2);
            b.survivors += (v == 0xFFFFu);
        }
    } else {
        for (int r : rows)
            for (int j = 0; j < n; ++j) {
                uint16_t v;
                std::memcpy(&v, raw + ((size_t)r * (size_t)n + (size_t)j) * 2, 2);
                b.survivors += (v == 0xFFFFu);
            }
    }
    b.hash = gemm_common::fnv1a(raw, total_elems * 2);
    // The predicates. worst_ulp is NOT one of them any more: an element that cancelled to nothing is
    // allowed to move many ULP of nothing, and saying so is the repo's measured band, not a widening -
    // every element that did not cancel still has to sit inside o.max_ulp, which stays 1.0.
    if (b.n_off_out_band)
        b.why += "ulp>" + std::to_string(o.max_ulp) + "_outside_cancellation_band(" +
                 std::to_string(b.n_off_out_band) + ",worst " + std::to_string(b.worst_out_band_ulp) +
                 "ULP_at_rel" + std::to_string(b.off_rel) + "_idx" + std::to_string(b.off_index) +
                 ") ";
    if (!(b.worst_in_band_excess <= o.cancel_tol))
        b.why += "cancelled_element_abs_error>cancel_tol(" + std::to_string(b.worst_in_band_excess) +
                 "x_max_ref) ";
    if (b.n_nonfinite)
        b.why += "nonfinite_output(" + std::to_string(b.n_nonfinite) +
                 ",elements_are_NaN_or_inf) ";
    if (b.survivors) b.why += "filler_survivors>0 ";
    if (!(b.cos >= o.cos_floor)) b.why += "cos<floor ";
    if (b.n && b.ndiff == b.n) b.why += "every_sampled_element_differs ";
    b.ok = b.why.empty();
    return b;
}

// One spelling of the scale-aware band metrics, shared by every line that reports a judgement, so a
// CHECK line and a CHECK-ENGINE line can be diffed mechanically instead of read. The raw max-ULP is
// printed alongside by every caller, because it is the number that first looked like a bug.
std::string band_str(const BandResult &b) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "n_off=%zu off_in_band=%zu off_out_band=%zu tightest_band=%.3e "
                  "worst_in_band_excess=%.3e nonfinite=%zu",
                  b.n_off, b.n_off_in_band, b.n_off_out_band, b.worst_off_rel,
                  b.worst_in_band_excess, b.n_nonfinite);
    // The offender's own coordinates, so a failure names its subject instead of its sync: which
    // element (flat index inside the sampled window), how far, and how large it is relative to the
    // tensor max - the three things that distinguish a cancellation from a wrong product.
    if (b.n_off_out_band) {
        char off[160];
        std::snprintf(off, sizeof(off),
                      " worst_offender=ulp%.2f@rel%.3e_idx%zu(ref=%.6g_got=%.6g)",
                      b.worst_out_band_ulp, b.off_rel, b.off_index, b.off_ref, b.off_got);
        return std::string(buf) + off;
    }
    return std::string(buf);
}

// -----------------------------------------------------------------------------------------------
// The complete argument list of one hipblasLtMatmul call, captured on the host at the moment the
// launch is built. Two callers use it: the launch site, which refuses to launch a list that fails a
// predicate, and --selftest-sys, which rebuilds the same lists from a captured enumeration and runs
// the same predicates with no device at all. One spelling of the checks, because a guard written
// separately from the thing it guards is decoration.
//
// Why this exists: --time faulted with the culprit narrowed only to "the warm-up sync", and LAYOUT
// ok=1 said the extents were fine. Naming the ALGO on the FAULT line cannot help when the algo is
// fine and some other argument is not, so every argument is recorded, printed on the FAULT line, and
// checkable offline.
// -----------------------------------------------------------------------------------------------
inline std::string ptr_str(const void *p) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)(uintptr_t)p);
    return std::string(buf);
}

inline std::string algo_hex(const hipblasLtMatmulAlgo_t &a) {
    const unsigned char *p = (const unsigned char *)&a;
    char buf[sizeof(a) * 2 + 1] = {0};
    for (size_t i = 0; i < sizeof(a); ++i) std::snprintf(buf + i * 2, 3, "%02x", p[i]);
    return std::string(buf);
}

inline bool algo_all_zero(const hipblasLtMatmulAlgo_t &a) {
    const unsigned char *p = (const unsigned char *)&a;
    for (size_t i = 0; i < sizeof(a); ++i)
        if (p[i]) return false;
    return true;
}

struct LaunchArgs {
    const char *phase = "";
    const char *shape = "";
    int m = 0, n = 0, k = 0;
    const void *A = nullptr;   // the operand in BLAS position A: must be the WEIGHT
    const void *B = nullptr;   // the operand in BLAS position B: must be the ACTIVATION
    const void *C = nullptr;
    void *D = nullptr;
    size_t a_elems = 0, b_elems = 0, c_elems = 0, d_elems = 0;  // what the buffers hold
    size_t need_A = 0, need_B = 0, need_C = 0;  // what the descriptors demand, from Plan
    float alpha = 1.0f, beta = 0.0f;
    const void *alpha_ptr = nullptr, *beta_ptr = nullptr;
    void *ws = nullptr;
    size_t ws_bytes = 0, ws_requested = 0;
    hipStream_t stream = nullptr;
    bool has_algo = false;
    hipblasLtMatmulAlgo_t algo{};
    int seq = -1, idx = -1, heur_rank = -1, state = -1, support_status = -1;
    // hipblasLtMatmulAlgo_t is 24 bytes here: uint8_t data[16] plus size_t max_workspace_bytes
    // (/opt/rocm/include/hipblaslt/hipblaslt.h:419-433). The header itself says the struct "can be
    // trivially serialized and later restored for use with the same version of the library", which is
    // what licenses --algo-dump / --selftest-sys. max_workspace_bytes travels with the algo, so it is
    // part of the argument list and gets checked like the rest.
    size_t algo_max_ws = 0;
    std::string solution;
    int dtA = 0, dtB = 0, dtC = 0, dtD = 0, compute = 0, scale = 0;
    bool desc_nonnull = false, layouts_nonnull = false;
    // True when the list was rebuilt on a host with no hipBLASLt objects in existence: the two
    // pointer-validity predicates about the descriptors themselves cannot be evaluated there, so they
    // report as skipped rather than as passing. The launch site leaves this false and really checks.
    bool host_only = false;
    bool inject_swap_AB = false;      // --selftest-sys must-catch injections
    bool inject_cd_split = false;
    bool inject_ws_small = false;
    bool inject_algo_zero = false;
    bool inject_state_bad = false;
    bool inject_null_D = false;
};

// Fills the predicates from a Plan plus the buffers actually being passed, so the launch site and the
// selftest cannot drift into checking two different things.
inline LaunchArgs make_args(const Plan &plan, const AlgoEntry *e, const char *phase,
                            const char *shape, const void *A, const void *B, const void *C,
                            const void *D, size_t a_elems, size_t b_elems, size_t c_elems,
                            size_t d_elems, const float *alpha_ptr, const float *beta_ptr,
                            const void *ws, size_t ws_bytes) {
    LaunchArgs g;
    g.phase = phase;
    g.shape = shape;
    g.m = plan.m; g.n = plan.n; g.k = plan.k;
    g.A = A; g.B = B; g.C = C; g.D = const_cast<void *>(D);
    g.a_elems = a_elems; g.b_elems = b_elems; g.c_elems = c_elems; g.d_elems = d_elems;
    g.need_A = plan.need_A(); g.need_B = plan.need_B(); g.need_C = plan.need_C();
    g.alpha_ptr = alpha_ptr; g.beta_ptr = beta_ptr;
    g.alpha = alpha_ptr ? *alpha_ptr : 0.0f;
    g.beta = beta_ptr ? *beta_ptr : 0.0f;
    g.ws = const_cast<void *>(ws); g.ws_bytes = ws_bytes;
    g.desc_nonnull = plan.desc != nullptr;
    g.layouts_nonnull = plan.la && plan.lb && plan.lc && plan.ld;
    g.dtA = plan.dtA; g.dtB = plan.dtB; g.dtC = plan.dtC; g.dtD = plan.dtD;
    g.compute = plan.compute; g.scale = plan.scale;
    if (e) {
        g.has_algo = true;
        g.algo = e->algo;
        g.seq = e->seq; g.idx = e->idx; g.heur_rank = e->heur_rank;
        g.state = e->state; g.support_status = e->support_status;
        g.ws_requested = e->ws;
        g.algo_max_ws = e->algo.max_workspace_bytes;
        g.solution = e->solution;
    }
    return g;
}

// The predicates, each named. `fails` gets "tag:detail", `passed` the tags that held - so a selftest
// can require that a predicate is reachable, not merely present: a check that never runs is
// indistinguishable from a check that always passes. Every one of these runs at the launch site too;
// --selftest-sys only adds the ability to run them with no device at all.
inline void validate_args(const LaunchArgs &gin, std::vector<std::string> &fails,
                          std::vector<std::string> &passed) {
    LaunchArgs g = gin;
    if (g.inject_swap_AB) { std::swap(g.A, g.B); std::swap(g.a_elems, g.b_elems); }
    // A different buffer, still 16-byte aligned: an offset of 2 bytes would also break alignment, and
    // then alignment_16B would appear to have caught a C/D split it has nothing to do with.
    if (g.inject_cd_split) g.D = (void *)((char *)g.D + 4096);
    if (g.inject_ws_small) {
        // Self-contained: an algorithm that asks for no workspace still has to be shown to be caught
        // when the allocation is too small, so the injection raises the request rather than only
        // shrinking the buffer.
        if (g.ws_requested == 0) g.ws_requested = g.ws_bytes + 1;
        else g.ws_bytes = g.ws_requested - 1;
    }
    if (g.inject_algo_zero) g.algo = hipblasLtMatmulAlgo_t{};
    if (g.inject_state_bad) g.state = (int)HIPBLAS_STATUS_NOT_SUPPORTED;
    if (g.inject_null_D) g.D = nullptr;
    fails.clear();
    passed.clear();
    auto check = [&](const char *tag, bool ok, const std::string &detail) {
        if (ok) passed.push_back(tag);
        else fails.push_back(std::string(tag) + ":" + detail);
    };
    if (g.host_only) {
        passed.push_back("desc_nonnull=skipped_host_only");
        passed.push_back("layouts_nonnull=skipped_host_only");
        // Same reason: with no descriptors in existence there is nothing to read the element types
        // from, so a host-built list must not claim the dtype contract either. The launch site leaves
        // host_only false and checks all three for real.
        passed.push_back("dtype_contract_bf16_in_f32_accumulate=skipped_host_only");
    } else {
        check("desc_nonnull", g.desc_nonnull, "matmul desc is null");
        check("layouts_nonnull", g.layouts_nonnull, "an A/B/C/D layout is null");
        check("dtype_contract_bf16_in_f32_accumulate",
              g.dtA == HIP_R_16BF && g.dtB == HIP_R_16BF && g.dtC == HIP_R_16BF &&
                  g.dtD == HIP_R_16BF && g.scale == HIP_R_32F,
              "dtypes A/B/C/D=" + std::to_string(g.dtA) + "/" + std::to_string(g.dtB) + "/" +
                  std::to_string(g.dtC) + "/" + std::to_string(g.dtD) + " scale=" +
                  std::to_string(g.scale) + " compute=" + std::to_string(g.compute));
    }
    check("A_extent_matches_desc", g.need_A == g.a_elems,
          "A desc (" + std::to_string(g.need_A) + " elems) behind a buffer holding " +
              std::to_string(g.a_elems));
    check("B_extent_matches_desc", g.need_B == g.b_elems,
          "B desc (" + std::to_string(g.need_B) + " elems) behind a buffer holding " +
              std::to_string(g.b_elems));
    check("C_extent_matches_desc", g.need_C == g.c_elems,
          "C desc (" + std::to_string(g.need_C) + " elems) behind a buffer holding " +
              std::to_string(g.c_elems));
    check("D_extent_matches_desc", g.need_C == g.d_elems,
          "D shares C's desc (" + std::to_string(g.need_C) + " elems) behind a buffer holding " +
              std::to_string(g.d_elems));
    check("ptrs_nonnull", g.A && g.B && g.C && g.D && g.alpha_ptr && g.beta_ptr,
          "A=" + ptr_str(g.A) + " B=" + ptr_str(g.B) + " C=" + ptr_str(g.C) + " D=" + ptr_str(g.D) +
              " alpha=" + ptr_str(g.alpha_ptr) + " beta=" + ptr_str(g.beta_ptr));
    check("alignment_16B", g.A && g.B && g.C && g.D && ((uintptr_t)g.A % 16) == 0 &&
                               ((uintptr_t)g.B % 16) == 0 && ((uintptr_t)g.C % 16) == 0 &&
                               ((uintptr_t)g.D % 16) == 0,
          "an operand pointer is not 16-byte aligned: A=" + ptr_str(g.A) + " B=" + ptr_str(g.B) +
              " C=" + ptr_str(g.C) + " D=" + ptr_str(g.D));
    // beta == 0: C is documented as unread, but here C and D are always the same buffer, so "where
    // the answer lands" stays one fact and a kernel that does read C reads something deliberate.
    // This is the invariant the operator named after the second fault.
    check("D_eq_C_when_beta_zero", !(g.beta == 0.0f && g.C != (const void *)g.D),
          "beta=0 but C=" + ptr_str(g.C) + " != D=" + ptr_str(g.D));
    check("scales_finite", std::isfinite(g.alpha) && std::isfinite(g.beta), "alpha/beta not finite");    if (g.ws_requested > 0)
        check("workspace_nonnull", g.ws != nullptr,
              "algo requests ws=" + std::to_string(g.ws_requested) + " and no workspace exists");
    else
        passed.push_back("workspace_nonnull");
    check("workspace_requested_fits", g.ws_requested <= g.ws_bytes,
          "algo requests ws=" + std::to_string(g.ws_requested) + ", allocated " +
              std::to_string(g.ws_bytes));
    if (g.has_algo && g.algo_max_ws != 0)
        check("algo_max_workspace_fits", g.algo_max_ws <= g.ws_bytes,
              "algo carries max_workspace_bytes=" + std::to_string(g.algo_max_ws) + ", allocated " +
                  std::to_string(g.ws_bytes));
    else
        passed.push_back("algo_max_workspace_fits");
    if (g.has_algo) {
        // state and support_status were recorded by the enumeration and, until this function existed,
        // printed but never enforced: a heuristic entry whose state was not SUCCESS still got launched.
        check("algo_state_success", g.state == (int)HIPBLAS_STATUS_SUCCESS,
              "enumeration state=" + std::to_string(g.state));
        check("algo_supported", g.support_status == (int)HIPBLAS_STATUS_SUCCESS,
              "matmulIsAlgoSupported status=" + std::to_string(g.support_status));
        check("algo_not_all_zero", !algo_all_zero(g.algo),
              "the 24 opaque algo bytes are all zero, i.e. an uninitialised hipblasLtMatmulAlgo_t");
    } else {
        passed.push_back("algo_state_success");
        passed.push_back("algo_supported");
        passed.push_back("algo_not_all_zero");
    }
}

inline void print_args(const char *tag, const LaunchArgs &g,
                       const std::vector<std::string> &passed) {
    std::string p;
    for (const auto &s : passed) p += s + ",";
    std::printf("%s phase=%s shape=%s M=%d N=%d K=%d desc=%d layouts=%d A=%s[ext=%zu] "
                "B=%s[ext=%zu] C=%s[ext=%zu] D=%s[ext=%zu] alpha=%.3g beta=%.3g ws=%s "
                "ws_bytes=%zu ws_requested=%zu algo_max_ws=%zu stream=%s algo_has=%d algo=%s seq=%d "
                "idx=%d "
                "heur_rank=%d state=%d support=%d solution=%s dtABCD=%d/%d/%d/%d compute=%d "
                "scale=%d checks=%s\n",
                tag, g.phase, g.shape, g.m, g.n, g.k, g.desc_nonnull ? 1 : 0,
                g.layouts_nonnull ? 1 : 0, ptr_str(g.A).c_str(), g.a_elems, ptr_str(g.B).c_str(),
                g.b_elems, ptr_str(g.C).c_str(), g.c_elems, ptr_str(g.D).c_str(), g.d_elems,
                g.alpha, g.beta, g.ws ? ptr_str(g.ws).c_str() : "null", g.ws_bytes, g.ws_requested,
                g.algo_max_ws, g.stream ? "user" : "default", g.has_algo ? 1 : 0,
                algo_hex(g.algo).c_str(),
                g.seq, g.idx, g.heur_rank, g.state, g.support_status, g.solution.c_str(), g.dtA,
                g.dtB, g.dtC, g.dtD, g.compute, g.scale, p.c_str());
}

// A device fault poisons the whole HIP context: every later launch in the same process fails, and
// fails quietly (AGENTS.md, the mixed_bw run_dev_case lesson - a faulting kernel made every case
// after it silently stop running). So there is nothing to continue to inside one process. What CAN
// be fixed is the anonymous death: CK prints `HIP_ERROR file:line`, which names the sync and not the
// candidate that was in flight, so a window that dies tells you nothing about which algo killed it.
// sync_named() names the subject and points at the one-process-per-candidate loop that can isolate
// it; the caller then exits 4.
bool sync_named(const char *phase, const EngineShape &sh, int m, int seq, int idx,
                const LaunchArgs *g = nullptr) {
    const hipError_t err = hipStreamSynchronize((hipStream_t)nullptr);
    if (err == hipSuccess) return true;
    std::printf("FAULT phase=%s shape=%s M=%d seq=%d idx=%d hip_error=\"%s\" "
                "note=\"the context is poisoned by this fault, so this process cannot continue; "
                "re-run this candidate alone with --algo-seq %d in its own process (HANDOFF.md)\"\n",
                phase, sh.name, m, seq, idx, hipGetErrorString(err), seq);
    // The whole argument list of the call that died, on the line before anything could have changed.
    // This is what makes the second fault diagnosable rather than merely localised: LAYOUT ok=1 told
    // us the extents were fine, so what is needed is every other argument, not a second guess.
    if (g) {
        std::vector<std::string> none, ok;
        validate_args(*g, none, ok);          // re-run here; `ok` is what the launch accepted
        print_args("FAULT-ARGS", *g, ok);
    }
    std::fflush(stdout);
    return false;
}

// -----------------------------------------------------------------------------------------------
// One (shape, M). Order is fixed and deliberate: enumerate -> allocate -> warm pages -> correctness
// pass per algorithm -> REFUSE or time -> default-vs-winner pair -> RESULT. RESULT is the last line
// of the block so a `grep RESULT` cannot be fooled by anything printed after it (AGENTS.md trap).
// -----------------------------------------------------------------------------------------------
bool time_problem(hipblasLtHandle_t lt, hipblasHandle_t hb, const EngineShape &sh, int m,
                  const Options &o, const float &alpha, const float &beta, void *ws,
                  double &ratio_out, bool &have_ratio) {
    const int n = sh.n, k = sh.k;
    const size_t na = (size_t)m * (size_t)k, nw = (size_t)n * (size_t)k, nc = (size_t)m * (size_t)n;

    Plan plan;
    std::string why;
    if (!plan.build(lt, m, n, k, o.ws_bytes, why)) {
        std::printf("REFUSE shape=%s M=%d reason=plan:%s\n", sh.name, m, why.c_str());
        plan.free();
        return false;
    }
    Enumeration en = enumerate(lt, plan, o, alpha, beta);
    print_enumeration(sh, m, en, o);

    // Candidates: supported for THIS problem, inside the workspace that exists, and inside the
    // --algo-seq / --max-algos window. Nothing is dropped silently any more: an algorithm that wants
    // more workspace than was allocated is a printed SKIP, because a silent drop here is exactly how
    // a split-K kernel that would have written past the buffer gets mistaken for a kernel that does
    // not exist. Counts always print, even when the lines are collapsed to keep the log readable.
    std::vector<AlgoEntry> cand;
    size_t skip_ws = 0, skip_unsup = 0, skip_cap = 0, skip_seq = 0, printed_skip = 0;
    for (const auto &e : en.algos) {
        const char *reason = nullptr;
        if (!e.supported) { reason = "not_supported"; ++skip_unsup; }
        else if (e.ws > o.ws_bytes) { reason = "workspace_exceeds_allocation"; ++skip_ws; }
        else if (o.algo_seq >= 0 && e.seq != o.algo_seq) { ++skip_seq; continue; }
        else if (o.max_algos > 0 && (int)cand.size() >= o.max_algos) { reason = "max_algos_cap"; ++skip_cap; }
        if (reason) {
            if (printed_skip < 40) {
                ++printed_skip;
                std::printf("SKIP shape=%s M=%d seq=%d idx=%d reason=%s ws=%zu allocated=%zu "
                            "heur_ws=%zu support_status=%d\n",
                            sh.name, m, e.seq, e.idx, reason, e.ws, o.ws_bytes, e.heur_ws,
                            e.support_status);
            }
            continue;
        }
        cand.push_back(e);
    }
    if (printed_skip < skip_ws + skip_unsup + skip_cap)
        std::printf("SKIP shape=%s M=%d reason=collapsed_lines total_skipped=%zu "
                    "(workspace=%zu unsupported=%zu cap=%zu) every one is in SKIP-SUMMARY\n",
                    sh.name, m, skip_ws + skip_unsup + skip_cap, skip_ws, skip_unsup, skip_cap);
    std::printf("SKIP-SUMMARY shape=%s M=%d candidates=%zu skipped_workspace=%zu "
                "skipped_unsupported=%zu skipped_cap=%zu skipped_algo_seq_filter=%zu allocated_ws=%zu\n",
                sh.name, m, cand.size(), skip_ws, skip_unsup, skip_cap, skip_seq, o.ws_bytes);

    int default_pos = -1;
    for (size_t i = 0; i < cand.size(); ++i)
        if (cand[i].is_default) default_pos = (int)i;
    const bool single = (o.algo_seq >= 0);
    if (cand.empty() || (default_pos < 0 && !single)) {
        std::printf("REFUSE shape=%s M=%d reason=no candidate carried both support and the "
                    "workspace (candidates=%zu default_in_candidates=%d heuristic_returned=%d "
                    "enumerated=%zu skipped_workspace=%zu skipped_unsupported=%zu)\n",
                    sh.name, m, cand.size(), default_pos >= 0, en.heur_returned, en.algos.size(),
                    skip_ws, skip_unsup);
        plan.free();
        return false;
    }
    if (single && default_pos < 0)
        std::printf("NOTE shape=%s M=%d mode=algo-seq seq=%d reason=default_not_in_set "
                    "-> no ratio is claimed, this is a single-candidate probe\n",
                    sh.name, m, o.algo_seq);

    // Operands. Same fill helper every other probe here uses: values exactly representable in both
    // 16-bit widths, so a difference between two runs can only be a different kernel, never a
    // different number.
    // hX/dX is the ACTIVATION tile [m,k]; hW/dW is the WEIGHT [n,k]. Named X and W and never A and B,
    // on purpose: gemm_stored_weight's own parameters are called A (holding the activation!) and W,
    // while in BLAS *operand position* A is the WEIGHT and B is the activation. Two conventions over
    // one buffer pair, and mixing them up is the bug that shipped in this file's first version
    // (README.md §1a): the activation buffer was passed behind plan.la, the weight's descriptor, so
    // the kernel read n*k elements out of an m*k buffer. Fault whenever n != m; a *silent wrong
    // product* whenever n == m.
    std::vector<uint8_t> hX(na * 2), hW(nw * 2);
    gemm_common::fill_16bit(hX.data(), na, true, 7919u + (unsigned)n);
    gemm_common::fill_16bit(hW.data(), nw, true, 104729u + (unsigned)k);
    std::vector<float> fw(nw);
    for (size_t i = 0; i < nw; ++i)
        fw[i] = gemm_common::dec16(((const uint16_t *)hW.data())[i], true);

    const std::vector<int> rows = sample_rows(m, o.check_rows);
    std::vector<double> ref((size_t)rows.size() * (size_t)n);
    {
        std::vector<float> fx((size_t)k);
        for (size_t s = 0; s < rows.size(); ++s) {
            const int t = rows[s];
            for (int i = 0; i < k; ++i)
                fx[(size_t)i] =
                    gemm_common::dec16(((const uint16_t *)hX.data())[(size_t)t * k + i], true);
            reference_row(fx.data(), fw.data(), n, k, 0, ref.data() + s * (size_t)n);
        }
    }

    void *dX = nullptr, *dW = nullptr, *dC = nullptr, *dC2 = nullptr;
    CK(hipMalloc(&dX, hX.size()));
    CK(hipMalloc(&dW, hW.size()));
    CK(hipMalloc(&dC, nc * 2));
    CK(hipMalloc(&dC2, nc * 2));
    CK(hipMemcpy(dX, hX.data(), hX.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));

    // The pairing check, taken on the host before a single kernel is launched. This is the guard that
    // turns "activation buffer behind the weight's descriptor" from an illegal memory access inside
    // a locked window into a printed REFUSE (README.md section 1a records the fault it replaces).
    {
        std::string lay;
        if (!plan.invariants(na, nw, nc, lay)) {
            std::printf("LAYOUT shape=%s M=%d ok=0 %s\n", sh.name, m, lay.c_str());
            std::printf("REFUSE shape=%s M=%d reason=layout_pairing -> %s nothing was launched for "
                        "this shape\n", sh.name, m, lay.c_str());
            CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
            plan.free();
            return false;
        }
        std::printf("LAYOUT shape=%s M=%d N=%d K=%d A=W[%dx%d,ld=%d]_need=%zu_buf=%zu_B=X[%dx%d,"
                    "ld=%d]_need=%zu_buf=%zu CD=[%dx%d,ld=%d]_need=%zu_buf=%zu ok=1\n",
                    sh.name, m, n, k, plan.ra, plan.ca, plan.lda, plan.need_A(), nw, plan.rb,
                    plan.cb, plan.ldb, plan.need_B(), na, plan.rc, plan.cc, plan.ldc_, plan.need_C(),
                    nc);
    }

    std::printf("COMPARE-MODE shape=%s M=%d reference=host double dot of the decoded bf16 inputs "
                "rounded to bf16 (gemm_common::compare_16bit_output, the comparator "
                "tools/gemm_peak.cpp:261 and tools/prefill_mix.cpp:656 use) | band=worst_ulp<=%.3f "
                "AND filler_survivors==0 AND cos>=%.9f | cos_floor_source=%s | rows_dotted=%zu_of_%d "
                "| scan=%s | alignment=row_major[t*N+j] on every side; the default-vs-winner "
                "comparison is the same tensor from two kernels at identical indices, no offset\n",
                sh.name, m, o.max_ulp, o.cos_floor, o.band_provenance.c_str(), rows.size(), m,
                o.scan_filler ? "full-D" : "partial");

    // One untimed default-algo call touches X, W, C and the workspace. AGENTS.md: the first GPU pass
    // over fresh pages runs ~35 GB/s, so a number taken before this is a page-fault measurement.
    //
    // THE OPERAND PAIRING, against the engine's working call as ground truth. gemm_stored_weight
    // issues hipblasGemmEx(OP_T, OP_N, n, m, k, alpha, W, dt, k, X, dt, k, beta, C, dt_out, n), so
    // BLAS-A is W with lda=k and BLAS-B is X with ldb=k, and that is exactly how plan.la (k rows, n
    // cols, ld=k) and plan.lb (k rows, m cols, ld=k) were built. So W goes first here, not dX.
    //
    // Every launch goes through here, so every launch has a recorded, validated argument list: the
    // list is built first, validate_args() runs on it, and a list that fails is refused with the
    // failing tags printed - no kernel is launched for a bad argument list, and the FAULT line for a
    // launch that faults anyway carries the same list.
    LaunchArgs last_args;
    std::string args_why;
    auto lt_run = [&](const AlgoEntry &e, void *dst, const char *phase) {
        args_why.clear();
        last_args = make_args(plan, &e, phase, sh.name, dW /*BLAS-A = weight, OP_T*/,
                              dX /*BLAS-B = activation, OP_N*/, dst /*C*/, dst /*D == C: beta is 0*/,
                              nw, na, nc, nc, &alpha, &beta, ws, o.ws_bytes);
        std::vector<std::string> fails, ok;
        validate_args(last_args, fails, ok);
        if (!fails.empty()) {
            for (const auto &f : fails) args_why += f + " ";
            std::printf("ARGS-REJECT phase=%s shape=%s M=%d seq=%d idx=%d solution=%s why=%s -> "
                        "no kernel was launched for this argument list\n",
                        phase, sh.name, m, e.seq, e.idx, e.solution.c_str(), args_why.c_str());
            print_args("ARGS-REJECT-LIST", last_args, ok);
            return (hipblasStatus_t)HIPBLAS_STATUS_INVALID_VALUE;
        }
        if (o.trace_launch || std::strcmp(phase, "warm") == 0) print_args("ARGS", last_args, ok);
        return hipblasLtMatmul(lt, plan.desc, &alpha, last_args.A, plan.la, last_args.B, plan.lb,
                               &beta, last_args.C, plan.lc, last_args.D, plan.ld, &e.algo, ws,
                               o.ws_bytes, (hipStream_t)nullptr);
    };
    if (single) {
        // A single-candidate probe warms with the candidate it is about to judge, not with a default
        // it is not running.
        const hipblasStatus_t s0 = lt_run(cand[0], dC, "warm-single");
        if (!sync_named("warm", sh, m, cand[0].seq, cand[0].idx, &last_args)) std::exit(4);
        if (s0 != HIPBLAS_STATUS_SUCCESS)
            std::printf("CHECK shape=%s M=%d seq=%d idx=%d launch_status=%d verdict=NOT-LAUNCHED\n",
                        sh.name, m, cand[0].seq, cand[0].idx, (int)s0);
    }
    {
        const int dpos = default_pos >= 0 ? default_pos : 0;
        const hipblasStatus_t s =
            single ? HIPBLAS_STATUS_SUCCESS : lt_run(cand[(size_t)dpos], dC, "warm");
        if (!sync_named("warm", sh, m, single ? -1 : cand[(size_t)dpos].seq,
                        single ? -1 : cand[(size_t)dpos].idx, &last_args))
            std::exit(4);
        if (s != HIPBLAS_STATUS_SUCCESS) {
            std::printf("REFUSE shape=%s M=%d reason=default algo failed to launch status=%d%s%s\n",
                        sh.name, m, (int)s, args_why.empty() ? " why=" : " ",
                        args_why.empty() ? "(launch was accepted, so the fault is inside the kernel)"
                                         : args_why.c_str());
            CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
            plan.free();
            return false;
        }
    }

    // ---- correctness pass. 0xFF filler first: beta=0 must erase all of it, and 0xFFFF is a bf16
    // NaN that no finite dot can produce, so a survivor is an unwritten element (AGENTS.md: pre-fill
    // with a value the writes never store). A kernel that READS C at beta=0 turns that NaN into
    // cos=nan and is reported WRONG, not worked around — see HANDOFF.md §4.
    std::vector<uint8_t> raw(nc * 2);
    std::vector<BandResult> band(cand.size());
    int band_ok = 0, not_launched = 0;
    for (size_t i = 0; i < cand.size(); ++i) {
        const AlgoEntry &e = cand[i];
        CK(hipMemset(dC, 0xFF, nc * 2));
        const hipblasStatus_t s = lt_run(e, dC, "check");
        if (!sync_named("check", sh, m, e.seq, e.idx, &last_args)) std::exit(4);
        if (s != HIPBLAS_STATUS_SUCCESS) {
            ++not_launched;
            std::printf("CHECK shape=%s M=%d seq=%d idx=%d launch_status=%d verdict=NOT-LAUNCHED\n",
                        sh.name, m, e.seq, e.idx, (int)s);
            continue;
        }
        CK(hipMemcpy(raw.data(), dC, raw.size(), hipMemcpyDeviceToHost));
        band[i] = judge(raw.data(), nc, rows, n, ref, o);
        std::printf("CHECK shape=%s M=%d seq=%d idx=%d ndiff=%zu/%zu worst_ulp=%.4f %s cos=%.9f "
                    "meanrel=%.3e survivors=%zu fnv=%016llx verdict=%s%s\n",
                    sh.name, m, e.seq, e.idx, band[i].ndiff, band[i].n, band[i].worst_ulp,
                    band_str(band[i]).c_str(), band[i].cos, band[i].meanrel, band[i].survivors,
                    (unsigned long long)band[i].hash, band[i].ok ? "ok" : "WRONG",
                    band[i].ok ? "" : (" " + band[i].why).c_str());
        if (band[i].ok) ++band_ok;
    }
    std::printf("CHECK-SUMMARY shape=%s M=%d candidates=%zu band_ok=%d band_wrong=%zu "
                "not_launched=%zu\n",
                sh.name, m, cand.size(), band_ok,
                cand.size() - (size_t)band_ok - (size_t)not_launched, (size_t)not_launched);

    // ---- the engine's own call, judged exactly like a candidate.
    //
    // This is the line that makes the band falsifiable instead of negotiable. hipBLAS'
    // HIPBLAS_GEMM_DEFAULT path is what engine/kernels_gemm.cpp:171 ships today, so:
    //   engine ok, algo WRONG   -> the algorithm is the suspect.
    //   engine WRONG too        -> no algorithm in the library could pass this gate, because the
    //                              reference and the floor are what are failing. That is exactly what
    //                              the first device window looked like: 12 of 12 WRONG, worst_ulp
    //                              exactly 1.0000, survivors 0, and one cos shared to the last digit
    //                              while ndiff wandered from 79 to 236.
    // It runs before the single/check_only returns below, so --check-only answers that question too.
    BandResult eng;
    bool eng_ran = false;
    if (hb) {
        CK(hipMemset(dC2, 0xFF, nc * 2));
        const hipblasStatus_t se =
            gemm_common::gemm_stored_weight(hb, m, n, k, HIP_R_16BF, dX, dW, dC2, HIP_R_16BF);
        if (!sync_named("engine-check", sh, m, -1, -1, nullptr)) std::exit(4);
        if (se != HIPBLAS_STATUS_SUCCESS) {
            std::printf("CHECK-ENGINE shape=%s M=%d launch_status=%d verdict=NOT-LAUNCHED\n",
                        sh.name, m, (int)se);
        } else {
            CK(hipMemcpy(raw.data(), dC2, raw.size(), hipMemcpyDeviceToHost));
            eng = judge(raw.data(), nc, rows, n, ref, o);
            eng_ran = true;
            std::printf("CHECK-ENGINE shape=%s M=%d call=hipblasGemmEx HIPBLAS_GEMM_DEFAULT "
                        "ndiff=%zu/%zu worst_ulp=%.4f %s cos=%.9f meanrel=%.3e survivors=%zu "
                        "fnv=%016llx "
                        "verdict=%s%s witness=\"the shipping path (engine/kernels_gemm.cpp:171); if "
                        "this is WRONG too, the band or the reference is the defect, not the "
                        "algorithm\"\n",
                        sh.name, m, eng.ndiff, eng.n, eng.worst_ulp, band_str(eng).c_str(), eng.cos,
                        eng.meanrel,
                        eng.survivors, (unsigned long long)eng.hash, eng.ok ? "ok" : "WRONG",
                        eng.ok ? "" : (" " + eng.why).c_str());
        }
    } else {
        std::printf("CHECK-ENGINE shape=%s M=%d verdict=SKIPPED reason=--engine-ref_0 -> this shape "
                    "has no known-correct witness, so a REFUSE here cannot be attributed to the "
                    "algorithm rather than to the band\n", sh.name, m);
    }

    // The constant-cos signature: several different kernels agreeing on the aggregate to the last bit
    // while their element counts differ. Distinct kernels do not do that to a measurement of
    // themselves; they do it to a measurement of something they all share, which here is the reference.
    // Printed always, including when clean, because a check that only speaks when it fires cannot be
    // told apart from a check that never ran.
    {
        std::vector<double> cs;
        size_t nlaunched = 0;
        size_t ndmin = (size_t)-1, ndmax = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            if (!band[i].launched) continue;
            ++nlaunched;
            cs.push_back(band[i].cos);
            ndmin = std::min(ndmin, band[i].ndiff);
            ndmax = std::max(ndmax, band[i].ndiff);
        }
        std::sort(cs.begin(), cs.end());
        const size_t distinct = std::unique(cs.begin(), cs.end()) - cs.begin();
        const double cs0 = cs.empty() ? 0.0 : cs[0];
        const bool sig = nlaunched >= 3 && distinct == 1 && ndmin != ndmax;
        std::printf("SIGNATURE shape=%s M=%d launched=%zu distinct_cos=%zu cos=%.12f ndiff_min=%zu "
                    "ndiff_max=%zu verdict=%s note=\"%s\"\n",
                    sh.name, m, nlaunched, distinct, cs0,
                    ndmin == (size_t)-1 ? 0 : ndmin, ndmax, sig ? "constant_cos_reference_limited"
                                                               : "no_constant_cos_signature",
                    sig
                        ? "every candidate shares one aggregate while disagreeing on how many elements "
                          "moved, which measures the reference and not the kernels; the floor for this "
                          "(shape, M) is --floor-probe's job and --cos-floor is the only override"
                        : "candidates do not share one aggregate, or ndiff does not vary, so nothing "
                          "here looks like reference noise");
    }

    // Which side of the harness a refusal belongs to, stated rather than left to be inferred from two
    // log files.
    const std::string attribution =
        eng_ran ? (eng.ok ? "attribution=engine_at_band_so_the_algorithm_is_the_suspect"
                          : "attribution=engine_call_also_outside_band_so_band_or_reference_is_suspect")
                : "attribution=no_engine_witness_so_unattributed";
    if (default_pos < 0) {
        std::printf("REFUSE shape=%s M=%d reason=no_default_ranked_algorithm -> nothing can be judged "
                    "against a default that does not exist\n", sh.name, m);
        CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
        plan.free();
        return false;
    }
    // A single-candidate probe (--algo-seq) reports and stops; it has no default to compare against.
    if (single) {
        CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
        plan.free();
        return band_ok > 0;
    }
    // --check-only is the worker-safe device path: it stops here, so no ms or TFLOP/s line exists in
    // that mode at all. This is what makes "did the fix work" answerable without a timing window.
    if (o.check_only) {
        if (!band[(size_t)default_pos].ok) {
            std::printf("REFUSE shape=%s M=%d mode=check-only reason=default-ranked algorithm not "
                        "at the band (%s) %s\n", sh.name, m,
                        band[(size_t)default_pos].why.c_str(), attribution.c_str());
            CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
            plan.free();
            return false;
        }
        std::printf("CHECK-ONLY shape=%s M=%d mode=check-only no_timing_taken=1 band_ok=%d_of=%zu "
                    "engine_verdict=%s cos_floor_source=%s\n", sh.name, m, band_ok, cand.size(),
                    eng_ran ? (eng.ok ? "ok" : "WRONG") : "skipped",
                    o.band_provenance.c_str());
        CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
        plan.free();
        return true;
    }
    // A default outside the band voids the shape: it is the harness or the pairing that is broken,
    // and timing the rest would launder that into a speed number.
    if (!band[(size_t)default_pos].ok) {
        std::printf("REFUSE shape=%s M=%d reason=default-ranked algorithm not at the band (%s) %s "
                    "-> nothing timed for this shape\n", sh.name, m,
                    band[(size_t)default_pos].why.c_str(), attribution.c_str());
        CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
        plan.free();
        return false;
    }

    // ---- the timed set: everything that survived the band, plus the engine's own call in the
    // rotation. Slot == timed.size() is the engine call, printed as ENGINE-CALL.
    std::vector<int> timed;
    for (size_t i = 0; i < cand.size(); ++i)
        if (band[i].ok) timed.push_back((int)i);
    const int nslot = (int)timed.size() + (o.engine_ref ? 1 : 0);
    const int engine_slot = o.engine_ref ? (int)timed.size() : -1;
    std::vector<double> best((size_t)nslot, -1.0);

    auto one = [&](int slot) {
        if (slot == engine_slot) {
            // hipblasGemmEx, not hipblasLtMatmul: same operands, the library's own default kernel.
            // gemm_common::gemm_stored_weight uses alpha=1, beta=0 internally, which is what is
            // recorded here (tools/gemm_common.h:122).
            last_args = make_args(plan, nullptr, "engine", sh.name, dW, dX, dC2, dC2, nw, na, nc,
                                  nc, &alpha, &beta, ws, o.ws_bytes);
            return gemm_common::gemm_stored_weight(hb, m, n, k, HIP_R_16BF, dX, dW, dC2,
                                                   HIP_R_16BF);
        }
        return lt_run(cand[(size_t)timed[(size_t)slot]], dC, "timed");
    };

    for (int i = 0; i < nslot; ++i) {           // warm-up per algorithm, untimed
        for (int w = 0; w < o.warm; ++w) (void)one(i);
        if (!sync_named("warm-loop", sh, m, i == engine_slot ? -1 : cand[(size_t)timed[(size_t)i]].seq,
                        i == engine_slot ? -1 : cand[(size_t)timed[(size_t)i]].idx, &last_args))
            std::exit(4);
    }
    // A launch that fails inside the timed loop would return instantly and look like a spectacular
    // kernel, so its status is checked and the whole shape is voided rather than re-ordered.
    bool launch_fail = false;
    int launch_fail_slot = -1;
    hipblasStatus_t launch_status = HIPBLAS_STATUS_SUCCESS;
    for (int rep = 1; rep <= o.reps && !launch_fail; ++rep) {
        // The rotation is the paired-measurement discipline: every slot is touched once per rep and
        // the order moves, so a disturbance landing mid-run hits all of them instead of permanently
        // favouring whoever ran first.
        for (int j = 0; j < nslot; ++j) {
            const int slot = (j + (rep - 1)) % nslot;
            const auto t0 = std::chrono::steady_clock::now();
            for (int q = 0; q < o.inner; ++q) {
                const hipblasStatus_t s = one(slot);
                if (s != HIPBLAS_STATUS_SUCCESS) {
                    launch_fail = true;
                    launch_fail_slot = slot;
                    launch_status = s;
                }
            }
            if (!sync_named("timed", sh, m,
                            slot == engine_slot ? -1 : cand[(size_t)timed[(size_t)slot]].seq,
                            slot == engine_slot ? -1
                                                : cand[(size_t)timed[(size_t)slot]].idx,
                            &last_args))
                std::exit(4);
            if (launch_fail) break;
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)o.inner;
            best[(size_t)slot] = (best[(size_t)slot] < 0) ? ms : std::min(best[(size_t)slot], ms);
        }
    }
    if (launch_fail) {
        std::printf("REFUSE shape=%s M=%d reason=launch failed inside the timed loop at slot %d "
                    "(seq %d) status=%d -> no RESULT for this shape\n",
                    sh.name, m, launch_fail_slot,
                    launch_fail_slot == engine_slot ? -1
                                                   : cand[(size_t)timed[(size_t)launch_fail_slot]].seq,
                    (int)launch_status);
        CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
        plan.free();
        return false;
    }

    const double flops = 2.0 * (double)m * (double)n * (double)k;
    double default_ms = -1.0, best_ms = -1.0, engine_ms = -1.0;
    int best_slot = -1;
    for (int i = 0; i < nslot; ++i) {
        const bool eng = (i == engine_slot);
        const AlgoEntry &e = eng ? cand[(size_t)default_pos] : cand[(size_t)timed[(size_t)i]];
        const double tf = best[(size_t)i] > 0 ? flops / (best[(size_t)i] * 1e9) : 0.0;
        std::printf("%s shape=%s M=%d seq=%d idx=%d ws=%zu best_ms=%.4f tflops=%.2f reps=%d "
                    "inner=%d solution=%s\n",
                    eng ? "ENGINE-CALL" : "TIME", sh.name, m, eng ? -1 : e.seq, e.idx,
                    eng ? (size_t)0 : e.ws, best[(size_t)i], tf, o.reps, o.inner,
                    eng ? "hipblasGemmEx HIPBLAS_GEMM_DEFAULT (engine/kernels_gemm.cpp:171)"
                        : e.solution.c_str());
        if (eng) { engine_ms = best[(size_t)i]; continue; }
        // `timed[i]` is the index into cand; `default_pos` is also an index into cand, NOT into
        // timed. Comparing the loop counter to it directly would silently never match once anything
        // has been dropped by the band, and default_ms would stay -1.
        if (timed[(size_t)i] == default_pos) default_ms = best[(size_t)i];
        if (best_ms < 0 || best[(size_t)i] < best_ms) { best_ms = best[(size_t)i]; best_slot = i; }
    }
    if (default_ms <= 0 || best_ms <= 0 || best_slot < 0) {
        std::printf("REFUSE shape=%s M=%d reason=no timed algorithm survived\n", sh.name, m);
        CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
        plan.free();
        return false;
    }
    const AlgoEntry &best_e = cand[(size_t)timed[(size_t)best_slot]];

    // ---- default vs winner, re-run into two buffers and compared at identical indices of the same
    // [M x N] row-major buffer. This is the check that the speed was won on the same product.
    {
        CK(hipMemset(dC, 0, nc * 2));
        CK(hipMemset(dC2, 0, nc * 2));
        const hipblasStatus_t s1 = lt_run(cand[(size_t)default_pos], dC, "pair");
        if (!sync_named("pair-default", sh, m, cand[(size_t)default_pos].seq,
                        cand[(size_t)default_pos].idx, &last_args))
            std::exit(4);
        const hipblasStatus_t s2 = lt_run(best_e, dC2, "pair");
        if (!sync_named("pair-winner", sh, m, best_e.seq, best_e.idx, &last_args)) std::exit(4);
        std::vector<uint8_t> d1(nc * 2), d2(nc * 2);
        CK(hipMemcpy(d1.data(), dC, d1.size(), hipMemcpyDeviceToHost));
        CK(hipMemcpy(d2.data(), dC2, d2.size(), hipMemcpyDeviceToHost));
        size_t diff = 0, nonfinite = 0;
        double worst_pair_ulp = 0.0;
        for (size_t i = 0; i < nc; ++i) {
            uint16_t b1, b2;
            std::memcpy(&b1, d1.data() + i * 2, 2);
            std::memcpy(&b2, d2.data() + i * 2, 2);
            if (b1 == b2) continue;
            ++diff;
            const double v1 = gemm_common::dec16(b1, true), v2 = gemm_common::dec16(b2, true);
            if (!std::isfinite(v1) || !std::isfinite(v2)) { ++nonfinite; continue; }
            const uint16_t wb = gemm_common::bits_of_bf16((float)std::fabs(v1));
            const double ulp = std::max(
                (double)(gemm_common::bf16_to_f32((uint16_t)(wb + 1)) -
                         gemm_common::bf16_to_f32(wb)), 1e-30);
            worst_pair_ulp = std::max(worst_pair_ulp, std::fabs(v1 - v2) / ulp);
        }
        const BandResult bb = judge(d2.data(), nc, rows, n, ref, o);
        std::printf("PAIR shape=%s M=%d default_seq=%d best_seq=%d diff_elements=%zu/%zu "
                    "worst_pair_ulp=%.4f nonfinite=%zu best_vs_ref_worst_ulp=%.4f "
                    "best_vs_ref_cos=%.9f launch=%d/%d verdict=%s | comparison: default output vs "
                    "winner output at identical index t*N+j of the same row-major [M x N] buffer; "
                    "the reference check is re-run on the winner over %zu sampled rows of %d\n",
                    sh.name, m, cand[(size_t)default_pos].seq, best_e.seq, diff, nc,
                    worst_pair_ulp, nonfinite, bb.worst_ulp, bb.cos, (int)s1, (int)s2,
                    (s1 == HIPBLAS_STATUS_SUCCESS && s2 == HIPBLAS_STATUS_SUCCESS &&
                     worst_pair_ulp <= o.max_ulp && bb.ok && nonfinite == 0)
                        ? "same_product" : "DIVERGENT",
                    rows.size(), m);
    }

    const double ratio = default_ms / best_ms;
    ratio_out = ratio;
    have_ratio = true;
    std::printf("RESULT shape=%s M=%d N=%d K=%d candidates=%zu band_ok=%d default_seq=%d "
                "default_ms=%.4f default_tflops=%.2f best_seq=%d best_ms=%.4f best_tflops=%.2f "
                "ratio=%.3f engine_call_ms=%.4f engine_vs_best=%.3f best_idx=%d best_solution=%s\n",
                sh.name, m, n, k, cand.size(), band_ok, cand[(size_t)default_pos].seq, default_ms,
                flops / (default_ms * 1e9), best_e.seq, best_ms, flops / (best_ms * 1e9), ratio,
                engine_ms, engine_ms > 0 ? engine_ms / best_ms : 0.0, best_e.idx,
                best_e.solution.c_str());

    CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
    plan.free();
    return true;
}

// Host-only: no HIP call is made on this path, which is what makes it runnable with no GPU window at
// all. It answers one question - does every descriptor ask for exactly the buffer it is paired with -
// and then proves the answer means something by re-injecting the bug. The must-catch set is DERIVED
// from the geometry (n != m is precisely when an extent check can see an A/B swap) rather than
// hand-listed, per the repo's negative-control rule; the n == m shapes are the derived
// must-not-catch set, printed as blind instead of quietly excluded, and the numerical band is what
// covers them (it REFUSEd k_proj at M=512 on exactly that case).
int selftest_layout(const Options &o) {
    int printed = 0, expected = 0, bad = 0, caught = 0, blind = 0;
    for (int m : o.rows) {
        for (const auto &sh : kShapes) {
            if (!o.shapes.empty()) {
                bool want = false;
                for (const auto &w : o.shapes)
                    if (w == sh.name) want = true;
                if (!want) continue;
            }
            ++expected;
            const size_t nx = (size_t)m * (size_t)sh.k, nw = (size_t)sh.n * (size_t)sh.k,
                         nc = (size_t)m * (size_t)sh.n;
            std::string why;
            Plan p;
            p.set_dims(m, sh.n, sh.k);
            const bool ok = p.invariants(nx, nw, nc, why);
            ++printed;
            std::printf("LAYOUT shape=%s M=%d N=%d K=%d A=W[%dx%d,ld=%d]_need=%zu_buf=%zu_B=X[%dx%d,"
                        "ld=%d]_need=%zu_buf=%zu CD=[%dx%d,ld=%d]_need=%zu_buf=%zu ok=%d%s\n",
                        sh.name, m, sh.n, sh.k, p.ra, p.ca, p.lda, p.need_A(), nw, p.rb, p.cb,
                        p.ldb, p.need_B(), nx, p.rc, p.cc, p.ldc_, p.need_C(), nc, ok ? 1 : 0,
                        ok ? "" : (" " + why).c_str());
            if (!ok) {
                ++bad;
                std::printf("LAYOUT-FAIL shape=%s M=%d reason=%s\n", sh.name, m, why.c_str());
            }
            // The injection: the same Plan with the two operand roles swapped behind their descriptors.
            Plan q;
            q.set_dims(m, sh.n, sh.k);
            q.swap_AB();
            std::string swhy;
            const bool swap_caught = !q.invariants(nx, nw, nc, swhy);
            if (sh.n != m) {
                if (swap_caught) ++caught;
                else {
                    ++bad;
                    std::printf("NEG-FAIL shape=%s M=%d reason=swapped_pairing_not_caught "
                                "must_catch_because=n_%d_ne_m_%d\n", sh.name, m, sh.n, m);
                }
            } else {
                ++blind;
                std::printf("NEG-BLIND shape=%s M=%d n_equals_m=%d reason=extents_identical "
                            "cover=numerical_band_only (CHECK verdict=WRONG; see HANDOFF.md section 4)\n",
                            sh.name, m, sh.n);
                if (swap_caught) {
                    ++bad;
                    std::printf("NEG-SURPRISE shape=%s M=%d reason=extent_check_claimed_a_catch_it_"
                                "cannot_make\n", sh.name, m);
                }
            }
        }
    }
    std::printf("SELFTEST-LAYOUT layout_lines_printed=%d expected=%d bad=%d swap_caught=%d "
                "swap_structurally_blind=%d "
                "rule=\"each descriptor's required extent must equal the buffer paired with it; the "
                "injected A/B swap must be caught wherever n != m and must NOT be claimed where "
                "n == m, where only the band can see it\"\n",
                printed, expected, bad, caught, blind);
    std::printf("SELFTEST-LAYOUT %s\n",
                (bad == 0 && printed == expected && expected > 0) ? "PASS" : "FAIL");
    return (bad == 0 && printed == expected && expected > 0) ? 0 : 1;
}

// -----------------------------------------------------------------------------------------------
// The capture --selftest-sys audits. --verify already enumerates every algorithm for every problem;
// writing the 16 opaque bytes plus the state, support status and workspace the library attached turns
// a runtime fault into something that can be re-examined on a host, with no lock and no second run.
// -----------------------------------------------------------------------------------------------
void algo_dump_line(FILE *f, const EngineShape &sh, int m, const AlgoEntry &e) {
    std::fprintf(f, "ALGODUMP shape=%s M=%d N=%d K=%d seq=%d idx=%d heur_rank=%d state=%d "
                   "support_status=%d supported=%d ws=%zu heur_ws=%zu data=%s solution=%s\n",
                 sh.name, m, sh.n, sh.k, e.seq, e.idx, e.heur_rank, e.state, e.support_status,
                 e.supported ? 1 : 0, e.ws, e.heur_ws, algo_hex(e.algo).c_str(),
                 e.solution.empty() ? "(unnamed)" : e.solution.c_str());
}

// Reads `key=value` out of a dumped line. The space-before-key test matters: looking for "ws=" inside
// "heur_ws=" is the kind of near-miss that makes a parser match nothing and report a clean bill.
std::string field(const std::string &line, const std::string &key, bool to_eol = false) {
    const std::string p = key + "=";
    size_t i = line.find(p);
    while (i != std::string::npos && i != 0 && line[i - 1] != ' ') i = line.find(p, i + 1);
    if (i == std::string::npos) return "";
    i += p.size();
    size_t j = to_eol ? line.size() : line.find(' ', i);
    if (j == std::string::npos) j = line.size();
    return line.substr(i, j - i);
}

bool hex_to_algo(const std::string &hex, hipblasLtMatmulAlgo_t &a) {
    if (hex.size() != sizeof(a) * 2) return false;
    unsigned char *p = (unsigned char *)&a;
    for (size_t i = 0; i < sizeof(a); ++i) {
        const std::string b = hex.substr(i * 2, 2);
        char *end = nullptr;
        const long v = std::strtol(b.c_str(), &end, 16);
        if (!end || *end) return false;
        p[i] = (unsigned char)v;
    }
    return true;
}

// -----------------------------------------------------------------------------------------------
// Band calibration, v2.
//
// v1 of this file carried `cos_floor` only, imported from another op's gate, and it refused 7 of 7
// shapes while missing by 1.8e-06. The first real probe window (5 shapes, 4 seeds, /tmp/
// gas_floor512.txt) then settled three things by measurement:
//
// 1. CONFLUENCE. On every shape probed, the minimum over seeds of the hipBLASLt default algo's cos,
//    of the engine's own hipblasGemmEx cos, and of the reference's own reordering cos are the same
//    number to all nine digits (q_proj 0.999997974322 on all three legs). The reason is visible in
//    the same log: every `ref_reorder` leg reports ndiff=0 and worst_ulp=0 - reordering the double
//    reference moves NOTHING, because all orders round to the same bf16 output. So the ~2.0e-06
//    cos deficit is not summation-order noise at all: it is the cost of storing the answer in bf16
//    at all (meanrel is 1.4e-03 throughout, and 1 - (1.4e-03)^2/2 is the observed 0.999998). All
//    three legs land there because all three quantise to bf16. That is what justifies calibrating
//    the floor instead of refusing forever - the floor is the format's resolution, not a tolerance.
//
// 2. A cos floor at this K cannot resolve a cut in the chain. one_term_short moves cos by
//    1.7e-06..9.2e-06 depending on shape and seed - within a factor of 4.6 of what bf16 output
//    rounding alone costs - so no floor can both admit a correct bf16 kernel and catch it. That is
//    a resolution limit of a K=1024..3584 dot, the same class of fact as mlp_gemv's F4 finding that
//    ndiff is blind to chains (prototype/mlp_gemv/results.md, "F4's aggregate bands"). v1's mistake
//    was a must-catch set built from chain arithmetic, which is a failure class NO algorithm swap
//    produces. v2 derives its injections from what an algorithm swap CAN break, below.
//
// 3. A flat max-ULP bound is the wrong instrument for a long dot. The device legs came back
//    worst_ulp=2.0 (in_proj_qkv) and 6.0 (gate_proj+up_proj) with meanrel 1.4e-03 and cos unchanged
//    - one element in ~3e4. mlp_gemv measured exactly this and priced it as a band on |ref|/max|ref|
//    rather than a wider ULP bound: raw max ULP 24 vs "gated max ULP 0 over tol", because every
//    offender was a cancellation at ~1e-6 of the tensor max (prototype/mlp_gemv/results.md, "How
//    wide the cancellation band is"; the form is `prototype/bf16_ulp.py:band_usage`). v2 does the
//    same, with the threshold measured here: ULP bound on elements that did not cancel, absolute
//    tolerance on elements that did. Note the exemption is NOT sized from the double reorders (they
//    move nothing); it is sized from the LICENSE leg, i.e. accumulating the same products in F32,
//    which is what HIPBLAS_COMPUTE_32F actually permits a kernel to do.
// -----------------------------------------------------------------------------------------------
struct CalRow {
    int version = 0;
    std::string shape, device;
    int m = 0, n = 0, k = 0, check_rows = 0, seeds = 0;
    double cos_floor = 0.0, margin = 0.0;
    double worst_algo_cos = 0.0, worst_engine_cos = 0.0, ref_reorder_worst_cos = 1.0;
    double license_worst_cos = 1.0, quant_cos_deficit = 0.0;
    // The two measured halves of the scale-aware band.
    double cancel_rel = 0.0, cancel_tol = 0.0;
    double license_worst_off_rel = 0.0, device_worst_off_rel = 0.0;
    double license_worst_excess = 0.0, device_worst_excess = 0.0;
    size_t license_n_off = 0, device_n_off = 0, device_n_off_in_band = 0;
    // The must-catch accounting, and the blind spot that is recorded rather than deleted.
    double mustcatch_max_cos = 0.0, one_term_cos = 0.0, one_term_ratio = 0.0;
    long ndiff_min = 0, ndiff_max = 0;
};

// The key binds check_rows on purpose: cos is the MINIMUM over sampled rows, so a floor measured with
// 8 rows is not the floor for a run that samples 16 (more rows can only lower it). Sharing one number
// across those two would be a silent tightening, and the tightening would be blamed on the kernel.
std::string cal_key(const std::string &shape, int m, int check_rows) {
    return shape + "|M=" + std::to_string(m) + "|check_rows=" + std::to_string(check_rows);
}

std::string cal_field(const std::string &s, const char *key) { return field(s, key); }
double cal_dbl(const std::string &s, const char *key) { return std::atof(cal_field(s, key).c_str()); }
long cal_int(const std::string &s, const char *key) { return std::atol(cal_field(s, key).c_str()); }

bool load_calibration(const std::string &path, std::vector<CalRow> &out, std::string &why) {
    FILE *f = std::fopen(path.c_str(), "r");
    if (!f) { why = "cannot_open " + path; return false; }
    char buf[8192];
    int total = 0;
    while (std::fgets(buf, sizeof(buf), f)) {
        std::string s(buf);
        if (s.rfind("CAL ", 0) != 0) continue;
        ++total;
        CalRow c;
        c.version = (int)cal_int(s, "version");
        if (c.version != 2) {
            why = "row " + std::to_string(total) + " is calibration format v" +
                  std::to_string(c.version) + ", this binary needs v2 (the scale-aware band needs "
                  "cancel_rel and cancel_tol, which v1 never measured) - re-run --floor-probe";
            out.clear();
            std::fclose(f);
            return false;
        }
        c.shape = cal_field(s, "shape");
        c.m = (int)cal_int(s, "M"); c.n = (int)cal_int(s, "N"); c.k = (int)cal_int(s, "K");
        c.check_rows = (int)cal_int(s, "check_rows"); c.seeds = (int)cal_int(s, "seeds");
        c.cos_floor = cal_dbl(s, "cos_floor"); c.margin = cal_dbl(s, "margin");
        c.worst_algo_cos = cal_dbl(s, "worst_algo_cos");
        c.worst_engine_cos = cal_dbl(s, "worst_engine_cos");
        c.ref_reorder_worst_cos = cal_dbl(s, "ref_reorder_worst_cos");
        c.license_worst_cos = cal_dbl(s, "license_worst_cos");
        c.quant_cos_deficit = cal_dbl(s, "quant_cos_deficit");
        c.cancel_rel = cal_dbl(s, "cancel_rel"); c.cancel_tol = cal_dbl(s, "cancel_tol");
        c.license_worst_off_rel = cal_dbl(s, "license_worst_off_rel");
        c.device_worst_off_rel = cal_dbl(s, "device_worst_off_rel");
        c.license_worst_excess = cal_dbl(s, "license_worst_excess");
        c.device_worst_excess = cal_dbl(s, "device_worst_excess");
        c.license_n_off = (size_t)cal_int(s, "license_n_off");
        c.device_n_off = (size_t)cal_int(s, "device_n_off");
        c.device_n_off_in_band = (size_t)cal_int(s, "device_n_off_in_band");
        c.mustcatch_max_cos = cal_dbl(s, "mustcatch_max_cos");
        c.one_term_cos = cal_dbl(s, "one_term_short_cos");
        c.one_term_ratio = cal_dbl(s, "one_term_over_quant");
        const std::string nd = cal_field(s, "ndiff_spread");
        const size_t dash = nd.find('-');
        if (dash != std::string::npos) {
            c.ndiff_min = std::atol(nd.substr(0, dash).c_str());
            c.ndiff_max = std::atol(nd.substr(dash + 1).c_str());
        }
        c.device = field(s, "device", true);
        // An unuseable row is refused, not skipped. Every one of these numbers is a gate threshold,
        // and a parsed-as-0.0 threshold is a check that cannot fail: cos >= 0 passes, |delta| <= 0
        // fails everything, cancel_rel <= 0 excuses nothing. All three are required to be in range.
        if (c.shape.empty() || !(c.cos_floor > 0.0 && c.cos_floor < 1.0) ||
            !(c.cancel_rel > 0.0 && c.cancel_rel < 1.0) || !(c.cancel_tol > 0.0)) {
            why = "row " + std::to_string(total) + " has no usable thresholds: " + s;
            out.clear();  // a half-parsed file must not be usable: the rows before the bad one would
                          // otherwise be judged as if the file were fine, and the missing-key count on
                          // the refusal line would understate what is actually unjudgable
            std::fclose(f);
            return false;
        }
        out.push_back(c);
    }
    std::fclose(f);
    if (out.empty()) { why = "no CAL lines in " + path; return false; }
    return true;
}

std::string cal_default_path(const std::string &program) {
    const size_t slash = program.rfind('/');
    return (slash == std::string::npos) ? std::string("floor_calibration.tsv")
                                        : program.substr(0, slash) + "/floor_calibration.tsv";
}

// The one place the band is chosen. The calibration is REQUIRED - it carries the cancellation-band
// thresholds as well as the cos floor, so an operator --cos-floor overrides one half of the band and
// never the other. Every source is named, because a band whose provenance is unknown cannot be
// argued about later.
bool resolve_band(const EngineShape &sh, int m, const Options &o, const std::vector<CalRow> &cal,
                  double &floor_out, std::string &prov_out, double &rel_out, double &tol_out) {
    const std::string key = cal_key(sh.name, m, o.check_rows);
    for (const auto &c : cal) {
        if (cal_key(c.shape, c.m, c.check_rows) != key) continue;
        floor_out = o.cos_floor_given ? o.cos_floor : c.cos_floor;
        // An override of the cancellation half is permitted for the same reason as the cos half - so
        // the band can be A/B'd - and it is named as an override in the same place, because a band
        // nobody can point at cannot be argued about later.
        rel_out = o.cancel_rel_given ? o.cancel_rel : c.cancel_rel;
        tol_out = o.cancel_tol_given ? o.cancel_tol : c.cancel_tol;
        prov_out = "measured on this build v" + std::to_string(c.version) + ": cos floor " +
                   std::to_string(c.cos_floor) + " = min over " + std::to_string(c.seeds) +
                   " seeds of {algo " + std::to_string(c.worst_algo_cos) + ", engine " +
                   std::to_string(c.worst_engine_cos) + "} minus margin " + std::to_string(c.margin) +
                   "; confluence ref_reorder=" + std::to_string(c.ref_reorder_worst_cos) +
                   " license=" + std::to_string(c.license_worst_cos) + " quantisation_deficit=" +
                   std::to_string(c.quant_cos_deficit) + " | cancellation band cancel_rel=" +
                   std::to_string(c.cancel_rel) + " cancel_tol=" + std::to_string(c.cancel_tol) +
                   " (license worst_off_rel " + std::to_string(c.license_worst_off_rel) +
                   ", device worst_off_rel " + std::to_string(c.device_worst_off_rel) +
                   (c.license_worst_off_rel > 0 || c.device_worst_off_rel > 0
                        ? ", source measured_worst_offender_x_margin)"
                        : ", source no_crossing_on_this_shape_so_nothing_is_exempt)");
        if (o.cos_floor_given)
            prov_out += " || cos floor OVERRIDDEN by --cos-floor " + std::to_string(o.cos_floor) +
                        ", typed explicitly, NOT a measurement of this build";
        if (o.cancel_rel_given || o.cancel_tol_given)
            prov_out += " || cancellation band OVERRIDDEN by " +
                        std::string(o.cancel_rel_given ? " --cancel-rel " + std::to_string(o.cancel_rel)
                                                       : "") +
                        std::string(o.cancel_tol_given ? " --cancel-tol " + std::to_string(o.cancel_tol)
                                                       : "") +
                        ", typed explicitly, NOT a measurement of this build";
        prov_out += " | check_rows=" + std::to_string(c.check_rows) + " device=\"" + c.device +
                    "\" source=" + o.calibration;
        return true;
    }
    floor_out = 0.0;
    rel_out = 0.0;
    tol_out = 0.0;
    prov_out = "no CAL row for " + key;
    return false;
}

void cal_write(FILE *f, const CalRow &c) {
    std::fprintf(f,
                 "CAL version=2 shape=%s M=%d N=%d K=%d check_rows=%d seeds=%d "
                 "worst_algo_cos=%.12f worst_engine_cos=%.12f ref_reorder_worst_cos=%.12f "
                 "license_worst_cos=%.12f quant_cos_deficit=%.3e ndiff_spread=%ld-%ld "
                 "margin=%.6e cos_floor=%.12f cancel_rel=%.6e cancel_tol=%.6e "
                 "license_worst_off_rel=%.3e license_n_off=%zu license_worst_excess=%.3e "
                 "device_worst_off_rel=%.3e device_n_off=%zu device_n_off_in_band=%zu "
                 "device_worst_excess=%.3e mustcatch_max_cos=%.9f one_term_short_cos=%.12f "
                 "one_term_over_quant=%.3f device=%s\n",
                 c.shape.c_str(), c.m, c.n, c.k, c.check_rows, c.seeds, c.worst_algo_cos,
                 c.worst_engine_cos, c.ref_reorder_worst_cos, c.license_worst_cos,
                 c.quant_cos_deficit, c.ndiff_min, c.ndiff_max, c.margin, c.cos_floor, c.cancel_rel,
                 c.cancel_tol, c.license_worst_off_rel, c.license_n_off, c.license_worst_excess,
                 c.device_worst_off_rel, c.device_n_off, c.device_n_off_in_band,
                 c.device_worst_excess, c.mustcatch_max_cos, c.one_term_cos, c.one_term_ratio,
                 c.device.c_str());
}

// Encodes a host-computed vector as bf16 and judges it with the SAME comparator that judges a kernel,
// so a must-catch number is never measured by a second, softer ruler.
BandResult judge_doubles(const std::vector<double> &got, size_t n_per_row, size_t n_rows,
                         const std::vector<double> &ref, Options o) {
    std::vector<uint8_t> raw(got.size() * 2);
    for (size_t i = 0; i < got.size(); ++i) {
        const uint16_t bits = gemm_common::enc16((float)got[i], true);
        std::memcpy(raw.data() + i * 2, &bits, 2);
    }
    std::vector<int> rows(n_rows);
    for (size_t i = 0; i < n_rows; ++i) rows[i] = (int)i;
    o.scan_filler = false;  // a host buffer holds no filler; that leg has nothing to see here
    return judge(raw.data(), got.size(), rows, (int)n_per_row, ref, o);
}

void ref_rows(const uint8_t *hX, const float *fw, const std::vector<int> &rows, int n, int k,
              int mode, double *ref) {
    std::vector<float> fx((size_t)k);
    for (size_t s = 0; s < rows.size(); ++s) {
        const int t = rows[s];
        for (int i = 0; i < k; ++i)
            fx[(size_t)i] = gemm_common::dec16(((const uint16_t *)hX)[(size_t)t * k + i], true);
        reference_row(fx.data(), fw, n, k, mode, ref + s * (size_t)n);
    }
}

// Reads the activation row back with a deliberately wrong leading dimension: the arithmetic a kernel
// does when BLAS-B's ld is not k. Real out-of-range reads are the FAULT class (README.md §1a), not a
// numerics case, so the index wraps instead; the point of the wrap is only to keep the modelled
// product inside the fixture, because any wrong stride at all turns the dot into a different dot.
void ref_rows_wrong_ld(const uint8_t *hX, size_t na, const float *fw, const std::vector<int> &rows,
                       int n, int k, long ld_extra, double *ref) {
    const uint16_t *h = (const uint16_t *)hX;
    std::vector<float> fx((size_t)k);
    for (size_t s = 0; s < rows.size(); ++s) {
        const long base = (long)rows[s] * ((long)k + ld_extra);
        for (int i = 0; i < k; ++i)
            fx[(size_t)i] = gemm_common::dec16(h[(size_t)(((base + i) % na + na) % na)], true);
        reference_row(fx.data(), fw, n, k, 0, ref + s * (size_t)n);
    }
}

// The three legs, in the order the probe runs them. Sizes are per-seed and small: one vector of
// nsamp*n doubles each, kept so the injections can be judged AFTER the band is sized (they need
// cancel_rel and cancel_tol to exist, and those come from the license and device legs).
struct LegSet {
    std::vector<uint8_t> dev_algo, dev_engine;       // device outputs, per seed
    std::vector<double> ref0, reorder[2], license[2];
    std::vector<BandResult> b_algo, b_engine, b_reorder[2], b_license[2];
    std::vector<double> cos_algo, cos_engine, cos_reorder, cos_license;
};

// -----------------------------------------------------------------------------------------------
// The derived must-catch set.
//
// v1 priced a cut in the accumulation chain and then refused to calibrate because a correct kernel
// and a one-term-short chain landed 7e-06 apart. The refusal was right and the set was wrong: no
// choice of hipBLASLt algorithm drops one term of a dot. Per the repo's rule that a negative-control
// set is DERIVED from what the object under test can actually break
// (prototype/attn_tile_model.py neg_expect / partalias), the object here is a library call whose
// arguments and workspace an algorithm swap can get wrong, so the set is:
//
//   ld_b_wrong            BLAS-B read with a leading dimension that is not k
//   operand_role_swapped  the two operands taken from the same buffer (this file's own fault-1 bug)
//   splitk_half_dropped   one of two split-K partials never added
//   splitk_eighth_dropped one of eight partials never added - the gentlest reachable partial loss
//   partial_slot_aliased  a split-K partial slot written by the wrong row (attn §20's class)
//   half_tile_stale       half a row stored, the other half left as the buffer was found
//   beta_left_at_1        D = alpha*prod + 1*C with C stale from a previous row
//   beta_left_at_1_nan    the same, with C still holding the 0xFF filler
//   alpha_doubled         alpha applied twice
//
// Each names the predicate that must catch it, exactly as --selftest-sys requires each argument
// injection to be caught by the predicate named for it ("some tag fired" is vacuous). alpha_doubled
// is the DERIVED BLIND CASE for the cos leg - cosine is scale-invariant by construction, so it
// catches nothing here and the ULP leg is what fires. Saying so out loud is the difference between a
// blind spot and a surprise.
// -----------------------------------------------------------------------------------------------
struct Injection {
    const char *name;
    int mode;             // reference_row mode, or -1/-2/-3/-4/-5 for the built ones below
    const char *expect;   // which predicate must catch it: "cos", "ulp", "nonfinite"
    const char *why;      // the algorithm-level defect it stands for
};
static const Injection kInj[] = {
    {"ld_b_wrong", -1, "cos", "BLAS-B leading dimension is not k"},
    {"operand_role_swapped", -2, "cos", "both operands from the weight buffer (this file's fault 1)"},
    {"splitk_half_dropped", 3, "cos", "one of two split-K partials never added"},
    {"splitk_eighth_dropped", 7, "cos", "one of eight split-K partials never added"},
    {"partial_slot_aliased", -3, "cos", "a split-K partial slot written by the wrong row"},
    {"half_tile_stale", -4, "cos", "half the tile stored, half left stale"},
    {"beta_left_at_1", -5, "cos", "beta left at 1 against a stale finite C"},
    {"beta_left_at_1_nan", -6, "nonfinite", "beta left at 1 against the 0xFF filler"},
    {"alpha_doubled", -7, "ulp", "alpha applied twice; cos is scale-invariant and cannot see it"},
};
static const int kInjCount = (int)(sizeof(kInj) / sizeof(kInj[0]));

void build_injection(const Injection &inj, const uint8_t *hX, size_t na, const float *fw,
                     const std::vector<int> &rows, int n, int k, const std::vector<double> &ref,
                     std::vector<double> &bad) {
    const size_t ns = rows.size();
    bad.assign((size_t)ns * (size_t)n, 0.0);
    switch (inj.mode) {
        case -1: ref_rows_wrong_ld(hX, na, fw, rows, n, k, 1, bad.data()); break;
        case -2: {
            // Both operands from W: bad[t][j] = sum_i W[t mod n, i] * W[j, i]. On n != m the real
            // swap reads out of the buffer and faults (the fault class, README.md §1a); the gram
            // product is the in-range stand-in, and it is the same defect seen by the band.
            for (size_t s = 0; s < ns; ++s)
                for (int j = 0; j < n; ++j) {
                    const float *pa = fw + (size_t)(rows[s] % (int)ns) * (size_t)k;
                    const float *pb = fw + (size_t)j * (size_t)k;
                    double acc = 0.0;
                    for (int i = 0; i < k; ++i) acc += (double)pa[i] * (double)pb[i];
                    bad[s * (size_t)n + j] = acc;
                }
            break;
        }
        case -3:  // whole row taken from the next sampled row
            for (size_t s = 0; s + 1 < ns; ++s)
                for (int j = 0; j < n; ++j)
                    bad[s * (size_t)n + j] = ref[(s + 1) * (size_t)n + j];
            for (int j = 0; j < n; ++j)
                bad[(ns - 1) * (size_t)n + j] = ref[(ns - 1) * (size_t)n + j];
            break;
        case -4:  // half the row stored, half taken from the next row
            for (size_t s = 0; s < ns; ++s) {
                const size_t u = (s + 1 < ns) ? s + 1 : s;
                for (int j = 0; j < n; ++j)
                    bad[s * (size_t)n + j] = (j < n / 2) ? ref[s * (size_t)n + j]
                                                        : ref[u * (size_t)n + j];
            }
            break;
        case -5:  // beta left at 1, C stale from another row
            for (size_t s = 0; s < ns; ++s) {
                const size_t u = (s + 1 < ns) ? s + 1 : s;
                for (int j = 0; j < n; ++j)
                    bad[s * (size_t)n + j] = ref[s * (size_t)n + j] + ref[u * (size_t)n + j];
            }
            break;
        case -6:  // beta left at 1 against the 0xFF filler, i.e. a bf16 NaN
            for (size_t i = 0; i < bad.size(); ++i)
                bad[i] = ref[i] + gemm_common::bf16_to_f32(0xFFFFu);
            break;
        case -7:
            for (size_t i = 0; i < bad.size(); ++i) bad[i] = 2.0 * ref[i];
            break;
        default:  // a chain cut, reachable as a dropped split-K partial
            ref_rows(hX, fw, rows, n, k, inj.mode, bad.data());
            break;
    }
}

// -----------------------------------------------------------------------------------------------
// The floor probe. Same shapes, same operands, same host reference as the gate; what it varies is the
// operand seed and the accumulation, and what it reports is how far the aggregates move when nothing
// is broken, how far a REACHABLE defect moves, and therefore where the band has to sit.
//
// Returns 0 = a CAL row was written, 1 = refused on purpose (PROBE-REFUSE printed, no row),
// 2 = could not run at all (no plan, no algorithm, launch failed).
// -----------------------------------------------------------------------------------------------
int probe_problem(hipblasLtHandle_t lt, hipblasHandle_t hb, const EngineShape &sh, int m, Options o,
                  const float &alpha, const float &beta, void *ws, FILE *calout,
                  const std::string &device_name) {
    const int n = sh.n, k = sh.k;
    const size_t na = (size_t)m * (size_t)k, nw = (size_t)n * (size_t)k, nc = (size_t)m * (size_t)n;
    Plan plan;
    std::string why;
    if (!plan.build(lt, m, n, k, o.ws_bytes, why)) {
        std::printf("PROBE-REFUSE shape=%s M=%d reason=plan:%s\n", sh.name, m, why.c_str());
        plan.free();
        return 2;
    }
    Enumeration en = enumerate(lt, plan, o, alpha, beta);
    const AlgoEntry *chosen = nullptr;
    for (const auto &e : en.algos)
        if (e.supported && e.ws <= o.ws_bytes) { chosen = &e; break; }
    if (!chosen) {
        std::printf("PROBE-REFUSE shape=%s M=%d reason=no_supported_algorithm_in_workspace\n", sh.name,
                    m);
        plan.free();
        return 2;
    }

    void *dX = nullptr, *dW = nullptr, *dC = nullptr, *dC2 = nullptr;
    std::vector<uint8_t> hX(na * 2), hW(nw * 2);
    std::vector<float> fw(nw);
    CK(hipMalloc(&dX, hX.size()));
    CK(hipMalloc(&dW, hW.size()));
    CK(hipMalloc(&dC, nc * 2));
    CK(hipMalloc(&dC2, nc * 2));
    const std::vector<int> rows = sample_rows(m, o.check_rows);
    const size_t ns = rows.size(), nsamp = ns * (size_t)n;

    CalRow c;
    c.shape = sh.name; c.m = m; c.n = n; c.k = k; c.check_rows = o.check_rows; c.seeds = o.seeds;
    c.margin = o.floor_margin; c.device = device_name; c.version = 2;

    std::vector<LegSet> seeds((size_t)o.seeds);
    bool failed = false;
    for (int s = 0; s < o.seeds && !failed; ++s) {
        LegSet &L = seeds[(size_t)s];
        // A different fill per seed, same generator: the spread reported is the spread of the
        // comparison, not of one lucky fixture.
        gemm_common::fill_16bit(hX.data(), na, true, 7919u + (unsigned)n + 1000003u * (unsigned)s);
        gemm_common::fill_16bit(hW.data(), nw, true, 104729u + (unsigned)k + 1000003u * (unsigned)s);
        for (size_t i = 0; i < nw; ++i)
            fw[i] = gemm_common::dec16(((const uint16_t *)hW.data())[i], true);
        CK(hipMemcpy(dX, hX.data(), hX.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));

        L.ref0.assign(nsamp, 0.0);
        ref_rows(hX.data(), fw.data(), rows, n, k, 0, L.ref0.data());
        for (int mode = 1; mode <= 2; ++mode) {
            L.reorder[mode - 1].assign(nsamp, 0.0);
            ref_rows(hX.data(), fw.data(), rows, n, k, mode, L.reorder[mode - 1].data());
        }
        for (int mode = 5; mode <= 6; ++mode) {
            L.license[mode - 5].assign(nsamp, 0.0);
            ref_rows(hX.data(), fw.data(), rows, n, k, mode, L.license[mode - 5].data());
        }

        // --- device legs. Filler in both buffers: a kernel that reads C at beta=0 turns the bf16 NaN
        // into a non-finite aggregate, which is a finding, not a pass.
        CK(hipMemset(dC, 0xFF, nc * 2));
        CK(hipMemset(dC2, 0xFF, nc * 2));
        const hipblasStatus_t sa =
            hipblasLtMatmul(lt, plan.desc, &alpha, dW, plan.la, dX, plan.lb, &beta, dC, plan.lc, dC,
                            plan.ld, &chosen->algo, ws, o.ws_bytes, (hipStream_t)nullptr);
        const hipblasStatus_t se = hb ? gemm_common::gemm_stored_weight(hb, m, n, k, HIP_R_16BF, dX,
                                                                        dW, dC2, HIP_R_16BF)
                                      : HIPBLAS_STATUS_NOT_SUPPORTED;
        if (!sync_named("probe", sh, m, chosen->seq, chosen->idx, nullptr)) std::exit(4);
        if (sa != HIPBLAS_STATUS_SUCCESS || se != HIPBLAS_STATUS_SUCCESS) {
            std::printf("PROBE-REFUSE shape=%s M=%d seed=%d reason=launch_failed algo=%d engine=%d\n",
                        sh.name, m, s, (int)sa, (int)se);
            failed = true;
            break;
        }
        L.dev_algo.resize(nc * 2);
        L.dev_engine.resize(nc * 2);
        CK(hipMemcpy(L.dev_algo.data(), dC, L.dev_algo.size(), hipMemcpyDeviceToHost));
        CK(hipMemcpy(L.dev_engine.data(), dC2, L.dev_engine.size(), hipMemcpyDeviceToHost));

        // Judge everything with the raw bound and NO cancellation band (cancel_rel 0 excuses nothing,
        // cancel_tol +inf excuses nothing), so these stats are the raw offenders that the band is
        // then sized against rather than pre-excused by it.
        Options raw = o;
        raw.cancel_rel = 0.0;
        raw.cancel_tol = 1e300;
        raw.scan_filler = false;  // the survivor count is per-device-buffer; counted once below
        L.b_algo = {judge(L.dev_algo.data(), nc, rows, n, L.ref0, raw)};
        L.b_engine = {hb ? judge(L.dev_engine.data(), nc, rows, n, L.ref0, raw) : BandResult{}};
        for (int q = 0; q < 2; ++q) {
            L.b_reorder[q] = {judge_doubles(L.reorder[q], (size_t)n, ns, L.ref0, raw)};
            L.b_license[q] = {judge_doubles(L.license[q], (size_t)n, ns, L.ref0, raw)};
        }
        const BandResult &ba = L.b_algo[0], &be = L.b_engine[0];
        std::printf("FLOOR shape=%s M=%d K=%d seed=%d leg=algo_cos algo_seq=%d ndiff=%zu/%zu "
                    "worst_ulp=%.4f %s cos=%.12f meanrel=%.3e\n", sh.name, m, k, s, chosen->seq,
                    ba.ndiff, ba.n, ba.worst_ulp, band_str(ba).c_str(), ba.cos, ba.meanrel);
        if (hb)
            std::printf("FLOOR shape=%s M=%d K=%d seed=%d leg=engine_cos call=hipblasGemmEx "
                        "ndiff=%zu/%zu worst_ulp=%.4f %s cos=%.12f meanrel=%.3e\n", sh.name, m, k, s,
                        be.ndiff, be.n, be.worst_ulp, band_str(be).c_str(), be.cos, be.meanrel);
        // A non-finite aggregate has to be caught before it is folded into anything: NaN compares false
        // against every bound, so it passes a ULP check for free.
        const bool nan_algo = !std::isfinite(ba.cos) || !std::isfinite(ba.worst_ulp);
        const bool nan_eng = hb && (!std::isfinite(be.cos) || !std::isfinite(be.worst_ulp));
        if (nan_algo || nan_eng) {
            std::printf("PROBE-REFUSE shape=%s M=%d seed=%d reason=nonfinite_aggregate "
                        "cos_algo=%.12f ulp_algo=%.4f cos_engine=%.12f ulp_engine=%.4f -> the device "
                        "output held a NaN or an inf (a kernel reading C at beta=0 is the usual "
                        "cause, HANDOFF.md §4); no floor can be measured from it\n", sh.name, m, s,
                        ba.cos, ba.worst_ulp, hb ? be.cos : 0.0, hb ? be.worst_ulp : 0.0);
            failed = true;
            break;
        }
        for (int q = 0; q < 2; ++q) {
            const char *ord[2] = {"reversed", "strided4_fold"};
            const char *lic[2] = {"f32_serial", "f32_strided4"};
            std::printf("FLOOR shape=%s M=%d K=%d seed=%d leg=ref_reorder order=%s ndiff=%zu/%zu "
                        "worst_ulp=%.4f n_off=%zu worst_off_rel=%.3e cos=%.12f\n", sh.name, m, k, s,
                        ord[q], L.b_reorder[q][0].ndiff, L.b_reorder[q][0].n,
                        L.b_reorder[q][0].worst_ulp, L.b_reorder[q][0].n_off,
                        L.b_reorder[q][0].worst_off_rel, L.b_reorder[q][0].cos);
            std::printf("FLOOR shape=%s M=%d K=%d seed=%d leg=license order=%s note=\"accumulate the "
                        "same products in F32, which is what HIPBLAS_COMPUTE_32F licenses\" "
                        "ndiff=%zu/%zu worst_ulp=%.4f n_off=%zu worst_off_rel=%.3e cos=%.12f\n",
                        sh.name, m, k, s, lic[q], L.b_license[q][0].ndiff, L.b_license[q][0].n,
                        L.b_license[q][0].worst_ulp, L.b_license[q][0].n_off,
                        L.b_license[q][0].worst_off_rel, L.b_license[q][0].cos);
        }
    }

    CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dC)); CK(hipFree(dC2));
    plan.free();
    if (failed) return 2;

    // ---- pass B: size the band from what a correct computation can do, then price the defects.
    // The exemption is cancellation-aware and never a widened bound: the ULP bound stays o.max_ulp
    // (1.0) for every element that did not cancel.
    for (auto &L : seeds) {
        c.worst_algo_cos = c.worst_algo_cos == 0.0 ? L.b_algo[0].cos
                                                  : std::min(c.worst_algo_cos, L.b_algo[0].cos);
        c.worst_engine_cos = c.worst_engine_cos == 0.0
                                 ? L.b_engine[0].cos
                                 : std::min(c.worst_engine_cos, L.b_engine[0].cos);
        for (int q = 0; q < 2; ++q) {
            c.ref_reorder_worst_cos = std::min(c.ref_reorder_worst_cos, L.b_reorder[q][0].cos);
            c.license_worst_cos = std::min(c.license_worst_cos, L.b_license[q][0].cos);
            c.license_worst_off_rel =
                std::max(c.license_worst_off_rel, L.b_license[q][0].worst_off_rel);
            c.license_n_off += L.b_license[q][0].n_off;
        }
        c.device_worst_off_rel = std::max({c.device_worst_off_rel, L.b_algo[0].worst_off_rel,
                                           hb ? L.b_engine[0].worst_off_rel : 0.0});
        c.device_n_off += L.b_algo[0].n_off + (hb ? L.b_engine[0].n_off : 0);
        c.ndiff_min = std::min(c.ndiff_min, (long)L.b_algo[0].ndiff);
        c.ndiff_max = std::max(c.ndiff_max, (long)L.b_algo[0].ndiff);
    }
    // The quantisation deficit: what bf16 output costs, which is exactly what all the legs agree on.
    c.quant_cos_deficit = 1.0 - c.worst_algo_cos;
    c.cos_floor = c.worst_algo_cos - o.floor_margin;

    // theta: the tightest band the LICENSE and DEVICE legs would pass at, times a stated margin -
    // the same arithmetic that took mlp_gemv's band from 1e-4 to 3e-5 with a 3.7x margin. Either leg can
    // supply it, and both are needed: on this build the LICENSE legs came back bit-identical to the
    // double reference (F32 accumulates a dot of bf16 products exactly at these K), so F32 reordering
    // alone would have sized a band of nothing and the single crossing element on in_proj_qkv - which the
    // engine's own call also produces, so it is legal arithmetic - would have been condemned. The device
    // legs supply what the license legs cannot, and the source is named rather than left as a number.
    const double worst_observed = std::max(c.license_worst_off_rel, c.device_worst_off_rel);
    const bool band_measured = worst_observed > 0.0;
    double theta = band_measured ? worst_observed * o.cancel_margin : 1e-12;
    const char *cancel_src = band_measured ? "measured_worst_offender_x_margin"
                                           : "no_crossing_on_this_shape_so_nothing_is_exempt";
    // cancel_tol is the latitude given to a CANCELLED element, so it can only be measured on the
    // elements that are inside the band - and the band is theta, which did not exist until the line
    // above. So the captured buffers are judged a second time with theta in place. This is host
    // arithmetic on data already read back; it launches nothing.
    //
    // Skipping this is not a rounding detail: measured against ALL elements, cancel_tol came out at
    // 1.9e-02 of the tensor max - one bf16 ULP on a full-magnitude element, times the margin - which
    // would have let a cancelled element drift by two hundredths of the tensor max and still pass. The
    // tolerance has to be the one the cancelled elements actually need.
    Options sized = o;
    sized.cancel_rel = theta;
    sized.cancel_tol = 1e300;  // this pass measures; the predicate is applied by the next run
    sized.scan_filler = false;
    double lic_excess = 0.0, dev_excess = 0.0;
    for (auto &L : seeds) {
        for (int q = 0; q < 2; ++q)
            lic_excess = std::max(
                lic_excess,
                judge_doubles(L.license[q], (size_t)n, ns, L.ref0, sized).worst_in_band_excess);
        dev_excess = std::max(
            dev_excess, judge(L.dev_algo.data(), nc, rows, n, L.ref0, sized).worst_in_band_excess);
        if (hb && !L.dev_engine.empty())
            dev_excess = std::max(
                dev_excess,
                judge(L.dev_engine.data(), nc, rows, n, L.ref0, sized).worst_in_band_excess);
    }
    c.license_worst_excess = lic_excess;
    c.device_worst_excess = dev_excess;
    double tol = std::max(lic_excess, dev_excess) * o.cancel_margin;
    if (!(tol > 0.0)) tol = 1e-12;

    std::printf("BAND-SIZE shape=%s M=%d K=%d ulp_bound=%.3f unchanged=1 license_worst_off_rel=%.3e "
                "device_worst_off_rel=%.3e license_n_off=%zu device_n_off=%zu margin=%.2f "
                "cancel_rel=%.6e cancel_rel_source=%s cancel_tol=%.6e "
                "in_band_excess{license=%.3e,device=%.3e} fullmag_reject=%.1e "
                "rule=\"elements with |ref|/max|ref| > cancel_rel keep the %.0f-ULP bound; elements at "
                "or below it are cancellations and are judged by |delta| <= cancel_tol x max|ref| "
                "instead - the form prototype/mlp_gemv/run.py gates with (CANCEL_REL 3e-5, sized by "
                "prototype/mlp_gemv/floor_probe.py with a 3.7x margin). source "
                "no_crossing_on_this_shape_so_nothing_is_exempt means no element of a CORRECT leg moved "
                "more than one ULP, so the band exempts nothing here and any future crossing on this "
                "shape is a real one\"\n",
                sh.name, m, k, o.max_ulp, c.license_worst_off_rel, c.device_worst_off_rel,
                c.license_n_off, c.device_n_off, o.cancel_margin, theta, cancel_src, tol,
                c.license_worst_excess, c.device_worst_excess, o.fullmag_rel, o.max_ulp);

    // The confluence witness, named. The three legs are different implementations of the same product
    // (a library algorithm, the engine's own call, and the reference reordered), so the question is not
    // "are they bit-identical" but "do they agree to within a hundredth of the deficit they all "
    // measure. Demanding exact equality was wrong and this file's own first v2 run proved it: q_proj
    // came back algo=...169280 against engine=...169179, one part in 1e10 apart, and a strict test "
    // called that disagreement.
    const double conf_max = std::max({c.worst_algo_cos, c.worst_engine_cos, c.ref_reorder_worst_cos});
    const double conf_min = std::min({c.worst_algo_cos, c.worst_engine_cos, c.ref_reorder_worst_cos});
    const double conf_spread = conf_max - conf_min;
    const bool confluence =
        c.quant_cos_deficit > 0 && conf_spread <= c.quant_cos_deficit / 100.0;
    std::printf("CONFLUENCE shape=%s M=%d K=%d legs={algo,engine,ref_reorder,license} agree=%d "
                "spread=%.3e spread_over_quantisation_deficit=%.3e threshold=1e-2 value=%.12f "
                "algo=%.12f engine=%.12f ref_reorder=%.12f quantisation_deficit=%.3e "
                "license_cos=%.12f verdict=%s note=\"the legs are a device kernel, the engine's own "
                "call, and the reference reordered; agreeing to within 1%% of the deficit they all "
                "measure means that deficit is bf16 output quantisation rather than any one of them, "
                "which is what licenses calibrating the floor instead of refusing\"\n", sh.name, m, k,
                confluence ? 1 : 0, conf_spread,
                c.quant_cos_deficit > 0 ? conf_spread / c.quant_cos_deficit : 0.0, conf_min,
                c.worst_algo_cos, c.worst_engine_cos, c.ref_reorder_worst_cos, c.quant_cos_deficit,
                c.license_worst_cos, confluence ? "confluent" : "legs_disagree");

    // ---- price the derived must-catch set with the band now in place.
    Options banded = o;
    banded.cancel_rel = theta;
    banded.cancel_tol = tol;
    banded.scan_filler = false;
    int neg_expected = 0, neg_caught = 0, neg_blind = 0, neg_fail = 0;
    bool neg_escaped = false;
    // The band is applied to injections but the cos floor must not be: an injection is judged by
    // "is its cos below where a correct kernel lands", which is a comparison against the measured
    // floor, not against a threshold that was derived from it.
    for (const auto &inj : kInj) {
        double worst_cos = -2.0;
        size_t sum_out_band = 0, sum_nonfinite = 0;
        bool any_nonfinite = false;
        double worst_ulp_seen = 0.0;
        for (auto &L : seeds) {
            std::vector<double> bad;
            build_injection(inj, hX.data(), na, fw.data(), rows, n, k, L.ref0, bad);
            const BandResult r = judge_doubles(bad, (size_t)n, ns, L.ref0, banded);
            worst_cos = std::max(worst_cos, r.cos);
            sum_out_band += r.n_off_out_band;
            sum_nonfinite += r.n_nonfinite;
            worst_ulp_seen = std::max(worst_ulp_seen, r.worst_out_band_ulp);
            // The witness is the element count, not the aggregate: a NaN propagates into the cosine and
            // judge() then reports that row as 0.0 rather than as NaN, so `cos is NaN` would have missed
            // it. n_nonfinite is what actually fired.
            if (r.n_nonfinite || !std::isfinite(r.cos)) any_nonfinite = true;
        }
        const bool cos_catches = (inj.expect[0] == 'c') && !(worst_cos >= c.cos_floor);
        const bool ulp_catches = (inj.expect[0] == 'u') && sum_out_band > 0;
        const bool nf_catches = (inj.expect[0] == 'n') && any_nonfinite;
        const bool caught = cos_catches || ulp_catches || nf_catches;
        if (std::strcmp(inj.name, "alpha_doubled") == 0) {
            // The derived blind case: cosine is scale-invariant, so this is the one injection the cos
            // leg must NOT be credited with. Printed as blind, not quietly omitted.
            std::printf("NEG-BLIND shape=%s M=%d injection=%s leg=cos reason=cosine_is_scale_invariant "
                        "cos=%.15f worst_ulp=%.1f out_of_band_offenders=%zu verdict=%s note=\"%s\"\n",
                        sh.name, m, inj.name, worst_cos, worst_ulp_seen, sum_out_band,
                        ulp_catches ? "covered_by_ulp_leg" : "NOT_COVERED", inj.why);
            ++neg_blind;
            if (!ulp_catches) { ++neg_fail; neg_escaped = true; }
            continue;
        }
        ++neg_expected;
        if (caught) {
            ++neg_caught;
            std::printf("NEG-CAUGHT shape=%s M=%d injection=%s by=%s cos=%.9f worst_ulp=%.1f "
                        "out_of_band_offenders=%zu nonfinite=%zu expected=%s defect=\"%s\"\n", sh.name, m,
                        inj.name,
                        cos_catches ? "cos_floor" : (ulp_catches ? "ulp_band" : "nonfinite_element"),
                        worst_cos, worst_ulp_seen, sum_out_band, sum_nonfinite, inj.expect, inj.why);
            if (cos_catches) c.mustcatch_max_cos = std::max(c.mustcatch_max_cos, worst_cos);
        } else {
            ++neg_fail;
            neg_escaped = true;
            std::printf("NEG-FAIL shape=%s M=%d injection=%s expected_predicate=%s cos=%.12f "
                        "worst_ulp=%.1f out_of_band_offenders=%zu nonfinite=%zu -> a defect this tool "
                        "can reach is not caught by the band it will be gated with\n", sh.name, m,
                        inj.name, inj.expect, worst_cos, worst_ulp_seen, sum_out_band, sum_nonfinite);
        }
    }

    // The recorded blind spot. Reported every run, never deleted, never counted as a must-catch:
    // a one-term-short chain is not reachable by any algorithm choice, AND it sits inside the same
    // bf16 quantisation the floor has to admit. Both halves of that sentence matter, so both print.
    {
        double worst_cos = -2.0, worst_ulp_seen = 0.0;
        size_t sum_out_band = 0;
        for (auto &L : seeds) {
            std::vector<double> bad;
            build_injection(kInj[0], hX.data(), na, fw.data(), rows, n, k, L.ref0, bad);  // unused
            std::vector<double> cut((size_t)ns * (size_t)n, 0.0);
            ref_rows(hX.data(), fw.data(), rows, n, k, 4, cut.data());
            const BandResult r = judge_doubles(cut, (size_t)n, ns, L.ref0, banded);
            worst_cos = std::max(worst_cos, r.cos);
            worst_ulp_seen = std::max(worst_ulp_seen, r.worst_out_band_ulp);
            sum_out_band += r.n_off_out_band;
        }
        c.one_term_cos = worst_cos;
        const double deficit = 1.0 - worst_cos;
        c.one_term_ratio = c.quant_cos_deficit > 0 ? deficit / c.quant_cos_deficit : 0.0;
        std::printf("RESOLUTION shape=%s M=%d K=%d injection=one_term_short cos=%.12f "
                    "cos_deficit=%.3e quantisation_deficit=%.3e ratio=%.2f ulp_out_of_band=%.1f "
                    "band_catches=%d verdict=below_cos_resolution_at_this_K note=\"PERMANENT: dropping "
                    "one term of a %d-term dot moves cos by only the printed multiple of what storing a "
                    "CORRECT answer in bf16 costs anyway - the same order of magnitude, and the worst "
                    "seed is the one printed, so no cos floor can both admit a correct kernel and catch "
                    "it on every seed. It is also unreachable by any hipBLASLt algorithm, which is the "
                    "other half of why it left the must-catch set. Recorded so nobody rediscovers the "
                    "blind spot and calls it a bug. Same class as prototype/mlp_gemv/results.md F4 "
                    "(ndiff is blind to chains)\"\n",
                    sh.name, m, k, worst_cos, deficit, c.quant_cos_deficit, c.one_term_ratio,
                    worst_ulp_seen, sum_out_band > 0 ? 1 : 0, k);
    }

    std::printf("FLOOR-RESULT shape=%s M=%d K=%d seeds=%d worst_algo_cos=%.12f worst_engine_cos=%.12f "
                "ref_reorder_worst_cos=%.12f license_worst_cos=%.12f mustcatch_max_cos=%.9f "
                "one_term_short_cos=%.12f ndiff_spread=%ld-%ld cos_floor_suggested=%.12f "
                "margin=%.6e cancel_rel=%.6e cancel_tol=%.6e "
                "note=\"probe mode reports aggregates and applies no band of its own; these are the "
                "numbers a later run defaults to\"\n", sh.name, m, k, o.seeds, c.worst_algo_cos,
                c.worst_engine_cos, c.ref_reorder_worst_cos, c.license_worst_cos,
                c.mustcatch_max_cos, c.one_term_cos, c.ndiff_min, c.ndiff_max, c.cos_floor, c.margin,
                theta, tol);

    bool refuse_written = false;
    if (neg_escaped) {
        std::printf("PROBE-REFUSE shape=%s M=%d reason=negative_control_escaped neg_expected=%d "
                    "neg_caught=%d neg_blind=%d neg_fail=%d -> a defect class this tool can reach is "
                    "not caught, so the band is wrong and must not be written\n", sh.name, m,
                    neg_expected, neg_caught, neg_blind, neg_fail);
        refuse_written = true;
    }
    if (!(c.cos_floor > c.mustcatch_max_cos)) {
        std::printf("PROBE-REFUSE shape=%s M=%d reason=band_would_not_discriminate "
                    "cos_floor=%.12f mustcatch_max_cos=%.12f -> the measured floor is looser than the "
                    "hardest REACHABLE injection, so a band here could not catch its own negative "
                    "controls\n", sh.name, m, c.cos_floor, c.mustcatch_max_cos);
        refuse_written = true;
    }
    // The full-magnitude tripwire: an element that did not cancel, moving by more than the ULP bound,
    // is not a cancellation and is not exempted by anything. This is the case worth writing up.
    if (c.device_worst_off_rel > o.fullmag_rel) {
        std::printf("NUMERICS-FINDING shape=%s M=%d K=%d worst_offender_rel=%.3e "
                    "fullmag_reject=%.1e device_n_off=%zu license_n_off=%zu "
                    "device_worst_excess_rel=%.3e -> at least one FULL-MAGNITUDE element is more than "
                    "%.0f ULP off the rounded reference on the shipping path. That is not a "
                    "cancellation and no band excuses it: the engine's own GEMM disagrees with a double "
                    "reference here. Write it up (results.md) before any timing of this shape is "
                    "quoted; no calibration is written\n",
                    sh.name, m, k, c.device_worst_off_rel, o.fullmag_rel, c.device_n_off,
                    c.license_n_off, c.device_worst_excess, o.max_ulp);
        refuse_written = true;
    }
    c.cancel_rel = theta;
    c.cancel_tol = tol;
    if (refuse_written) return 1;
    if (calout) {
        cal_write(calout, c);
        std::fflush(calout);
    }
    return 0;
}

// Rebuilds each captured algorithm's argument list on the host and runs the launch site's predicates
// on it, then re-injects the ways an argument list goes wrong and requires each to be caught. Exits
// nonzero on any violation. No HIP call is made here, which is what makes it runnable without a GPU
// window at all - the whole point of asking for it after a fault the on-device lines could not explain.
// -----------------------------------------------------------------------------------------------
int selftest_sys(const Options &o) {
    if (o.algo_file.empty()) {
        std::printf("SELFTEST-SYS INCOMPLETE reason=no_--algo-file cmd=\"%s --verify "
                    "--algo-dump /tmp/gas_algos.txt [--rows ...] then --selftest-sys --algo-file "
                    "/tmp/gas_algos.txt\"\n", "gemm_algo_sweep");
        return 3;
    }
    FILE *f = std::fopen(o.algo_file.c_str(), "r");
    if (!f) {
        std::printf("SELFTEST-SYS INCOMPLETE reason=cannot_open file=%s\n", o.algo_file.c_str());
        return 3;
    }
    std::vector<std::string> lines;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), f)) {
        std::string s(buf);
        if (s.rfind("ALGODUMP ", 0) != 0) continue;
        lines.push_back(s);
    }
    std::fclose(f);
    if (lines.empty()) {
        std::printf("SELFTEST-SYS INCOMPLETE reason=no_ALGODUMP_lines file=%s - a check over an empty "
                    "capture cannot pass, so nothing is claimed\n", o.algo_file.c_str());
        return 3;
    }

    // Synthetic but real addresses: 16-byte aligned, distinct, and sized from the shape, so the
    // pairing/alignment/extent predicates are exercised against numbers instead of against nulls.
    const void *kA = (const void *)(uintptr_t)0x1000;      // the weight buffer
    const void *kB = (const void *)(uintptr_t)0x200000;    // the activation buffer
    const void *kC = (const void *)(uintptr_t)0x400000;    // C and D, the same buffer
    void *kWS = (void *)(uintptr_t)0x800000;
    const float alpha = 1.0f, beta = 0.0f;

    int parsed = 0, checked = 0, bad = 0, neg_expected = 0, neg_caught = 0, problems = 0;
    std::string last_problem;
    for (const auto &line : lines) {
        const std::string name = field(line, "shape");
        const EngineShape *shp = nullptr;
        for (const auto &s : kShapes)
            if (std::string(s.name) == name) shp = &s;
        const int m = (int)std::atoi(field(line, "M").c_str());
        const std::string hex = field(line, "data");
        std::string sol = field(line, "solution", true);
        while (!sol.empty() && (sol.back() == '\n' || sol.back() == '\r' || sol.back() == ' '))
            sol.pop_back();
        if (!shp) {
            std::printf("UNMATCHED shape=%s reason=not_in_census - the capture is from a different "
                        "census than the binary reading it\n", name.c_str());
            ++bad;
            continue;
        }
        AlgoEntry e;
        e.seq = (int)std::atoi(field(line, "seq").c_str());
        e.idx = (int)std::atoi(field(line, "idx").c_str());
        e.heur_rank = (int)std::atoi(field(line, "heur_rank").c_str());
        e.state = (int)std::atoi(field(line, "state").c_str());
        e.support_status = (int)std::atoi(field(line, "support_status").c_str());
        e.supported = field(line, "supported") == "1";
        e.ws = (size_t)std::strtoull(field(line, "ws").c_str(), nullptr, 10);
        e.heur_ws = (size_t)std::strtoull(field(line, "heur_ws").c_str(), nullptr, 10);
        e.solution = sol;
        ++parsed;
        if (m < 2 || !hex_to_algo(hex, e.algo)) {
            std::printf("OPAQUE shape=%s M=%d seq=%d data=%s verdict=UNPARSEABLE reason=%s\n",
                        name.c_str(), m, e.seq, hex.c_str(),
                        m < 2 ? "bad_M" : ("hex_length_" + std::to_string(hex.size()) +
                                            "_want_" + std::to_string(sizeof(e.algo) * 2)).c_str());
            ++bad;
            continue;
        }

        Plan p;
        p.set_dims(m, shp->n, shp->k);
        const size_t nx = (size_t)m * (size_t)shp->k, nw = (size_t)shp->n * (size_t)shp->k,
                     nc = (size_t)m * (size_t)shp->n;
        const std::string problem = name + "/M=" + std::to_string(m);
        const bool first_of_problem = problem != last_problem;
        if (first_of_problem) {
            ++problems;
            last_problem = problem;
            std::string lwhy;
            const bool lok = p.invariants(nx, nw, nc, lwhy);
            std::printf("LAYOUT shape=%s M=%d N=%d K=%d A=W[%dx%d,ld=%d]_need=%zu_buf=%zu_B=X[%dx%d,"
                        "ld=%d]_need=%zu_buf=%zu CD=[%dx%d,ld=%d]_need=%zu_buf=%zu ok=%d%s\n",
                        shp->name, m, shp->n, shp->k, p.ra, p.ca, p.lda, p.need_A(), nw, p.rb, p.cb,
                        p.ldb, p.need_B(), nx, p.rc, p.cc, p.ldc_, p.need_C(), nc,
                        lok ? 1 : 0, lok ? "" : (" " + lwhy).c_str());
            if (!lok) ++bad;
        }

        LaunchArgs g = make_args(p, &e, "selftest", shp->name, kA, kB, kC, kC, nw, nx, nc, nc,
                                 &alpha, &beta, kWS, o.ws_bytes);
        g.host_only = true;
        std::vector<std::string> fails, ok;
        validate_args(g, fails, ok);
        std::printf("OPAQUE shape=%s M=%d seq=%d idx=%d heur_rank=%d all_zero=%d algo=%s state=%d "
                    "support_status=%d supported=%d ws_requested=%zu algo_max_ws=%zu "
                    "ws_allocated=%zu solution=%s\n",
                    shp->name, m, e.seq, e.idx, e.heur_rank, algo_all_zero(e.algo) ? 1 : 0,
                    algo_hex(e.algo).c_str(), e.state, e.support_status, e.supported ? 1 : 0,
                    e.ws, g.algo_max_ws, o.ws_bytes, e.solution.c_str());
        std::printf("PAIR shape=%s M=%d seq=%d A=W@%s[buf=%zu,desc=%zu] B=X@%s[buf=%zu,desc=%zu] "
                    "C=%s D=%s cd_equal=%d beta=%.3g scales=%s/%s alignment=16B_ok=%d "
                    "violations=%zu%s\n",
                    shp->name, m, e.seq, ptr_str(g.A).c_str(), g.a_elems, g.need_A,
                    ptr_str(g.B).c_str(), g.b_elems, g.need_B, ptr_str(g.C).c_str(),
                    ptr_str(g.D).c_str(), g.C == (const void *)g.D ? 1 : 0, g.beta,
                    ptr_str(g.alpha_ptr).c_str(), ptr_str(g.beta_ptr).c_str(),
                    ((uintptr_t)g.A % 16 == 0 && (uintptr_t)g.B % 16 == 0 &&
                     (uintptr_t)g.C % 16 == 0 && (uintptr_t)g.D % 16 == 0)
                        ? 1 : 0,
                    fails.size(), fails.empty() ? "" : (" why=" + fails[0]).c_str());
        ++checked;
        if (!fails.empty()) {
            std::printf("VIOLATION shape=%s M=%d seq=%d idx=%d why=", shp->name, m, e.seq, e.idx);
            for (const auto &xf : fails) std::printf("%s ", xf.c_str());
            std::printf("solution=%s\n", e.solution.c_str());
            ++bad;
        }

        // The injections. Each one is a way an argument list can be wrong, and each must be caught by
        // the predicate whose name says it: "some tag fired" is not evidence, because one unrelated
        // violation would then make every injection look caught. That is the vacuous-control failure
        // this repo has already been bitten by twice, so the expected tag is stated per injection.
        if (first_of_problem) {
            struct Injection {
                const char *tag;
                const char *expect_a;   // acceptable catching predicate
                const char *expect_b;   // second acceptable one, or null
            };
            static const Injection kInj[6] = {
                {"swap_AB", "A_extent_matches_desc", "B_extent_matches_desc"},
                {"cd_split", "D_eq_C_when_beta_zero", nullptr},
                {"ws_small", "workspace_requested_fits", nullptr},
                {"algo_zero", "algo_not_all_zero", nullptr},
                {"state_bad", "algo_state_success", nullptr},
                {"null_D", "ptrs_nonnull", nullptr},
            };
            for (int t = 0; t < 6; ++t) {
                LaunchArgs h = g;
                h.inject_swap_AB = (t == 0);
                h.inject_cd_split = (t == 1);
                h.inject_ws_small = (t == 2);
                h.inject_algo_zero = (t == 3);
                h.inject_state_bad = (t == 4);
                h.inject_null_D = (t == 5);
                std::vector<std::string> f2, ok2;
                validate_args(h, f2, ok2);
                // Which tags fired, by name.
                auto fired = [&](const char *want) {
                    for (const auto &xf : f2)
                        if (xf.rfind(std::string(want) + ":", 0) == 0) return true;
                    return false;
                };
                if (t == 0 && shp->n == m) {
                    // Derived must-not-catch: with n == m the two operand buffers are the same size,
                    // so no extent check can see the swap. Only the numerical band can, and claiming
                    // otherwise here would be the vacuous-control failure above.
                    std::printf("NEG-BLIND shape=%s M=%d injection=%s reason=n_equals_m_%d "
                                "cover=numerical_band_only\n", shp->name, m, kInj[t].tag, shp->n);
                    if (fired(kInj[t].expect_a) || (kInj[t].expect_b && fired(kInj[t].expect_b))) {
                        std::printf("NEG-SURPRISE shape=%s M=%d injection=%s why=extent_check_claimed_"
                                    "a_catch_it_cannot_make\n", shp->name, m, kInj[t].tag);
                        ++bad;
                    }
                    continue;
                }
                ++neg_expected;
                const bool by_right_tag =
                    fired(kInj[t].expect_a) || (kInj[t].expect_b && fired(kInj[t].expect_b));
                if (!by_right_tag) {
                    std::printf("NEG-FAIL shape=%s M=%d injection=%s expected_predicate=%s "
                                "fired=%zu", shp->name, m, kInj[t].tag, kInj[t].expect_a,
                                f2.size());
                    for (const auto &xf : f2) std::printf(" got=%s", xf.substr(0, xf.find(':')).c_str());
                    std::printf("\n");
                    ++bad;
                } else {
                    ++neg_caught;
                    const char *by = fired(kInj[t].expect_a) ? kInj[t].expect_a : kInj[t].expect_b;
                    std::printf("NEG-CAUGHT shape=%s M=%d injection=%s by=%s\n", shp->name, m,
                                kInj[t].tag, by);
                }
            }
        }
    }
    std::printf("SELFTEST-SYS lines_parsed=%d algos_checked=%d problems=%d bad=%d neg_expected=%d "
                "neg_caught=%d ws_bytes_assumed=%zu\n", parsed, checked, problems, bad, neg_expected,
                neg_caught, o.ws_bytes);
    std::printf("SKIPPED-HOST-ONLY tags=desc_nonnull,layouts_nonnull,dtype_contract_bf16_in_f32_"
                "accumulate reason=no hipBLASLt object exists in this mode; the launch site checks "
                "all three for real\n");
    const bool pass = bad == 0 && parsed == (int)lines.size() && checked == parsed &&
                      neg_caught == neg_expected && neg_expected > 0 && parsed > 0;
    std::printf("SELFTEST-SYS %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int print_list() {
    int total = 0;
    std::printf("# The matmuls the library route issues. Read out of engine/, not remembered.\n");
    std::printf("# form: C_rm[M,N] = X_rm[M,K] * W_rm[N,K]^T, bf16 in / f32 accumulate / bf16 C\n");
    std::printf("#       (engine/kernels_gemm.cpp:171 -> tools/gemm_common.h:119)\n");
    std::printf("%-18s %-11s %-6s %-6s %-9s %s\n", "shape", "family", "N", "K", "per_wave", "site");
    for (const auto &s : kShapes) {
        std::printf("%-18s %-11s %-6d %-6d %-9d %s\n", s.name, s.family, s.n, s.k, s.per_wave,
                    s.site);
        total += s.per_wave;
    }
    std::printf("TOTAL library matmuls per 24-layer wave = %d  (cross-check: "
                "devices/strix-halo-compute.md second correction prices a 150-call 24-layer mix)\n",
                total);
    std::printf("EXCLUDED in_proj_a/in_proj_b (N=16): engine/kernels_proj.cpp:134-141 refuses them "
                "to the hand GEMV\n");
    std::printf("EXCLUDED lm_head (248320x1024): hand GEMV at engine/kernels_lmhead.cpp:25, once "
                "per request; no route reaches hipBLAS with it\n");
    std::printf("EXCLUDED M=1: engine/kernels_gemm.cpp:844-850 sends a one-row body to the hand "
                "GEMV whatever the run asked for\n");
    std::printf("\nWave width: the shipped per-call cap is 512 rows (engine/decode_step.h:49 "
                "kMaxWaveRows; engine/decode_run.cpp:279 refuses more), so M=1024 and M=4096 are "
                "route widths, not today's shipped call shape. README.md section 2.\n");
    return total == 150 ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 2;
    if (o.mode == Options::kList) return print_list();
    // The two host-only selftests run before any HIP call exists, which is what lets them be run with
    // no GPU window at all: no hipblasLtCreate, no hipMalloc, no stream, no kernel.
    if (o.mode == Options::kSelftestLayout) return selftest_layout(o);
    if (o.mode == Options::kSelftestSys) return selftest_sys(o);

    // ---- where the band comes from, settled before anything touches the device.
    //
    // The calibration file is MANDATORY for any run that judges numerics, and that is stricter than v1,
    // which let --cos-floor stand alone. The reason is the cancellation half: cancel_rel and cancel_tol
    // are thresholds that only a measurement can supply, and a run that took a cos floor from the command
    // line would have taken them from nowhere. So --cos-floor now overrides ONE half of one threshold
    // against a loaded calibration, and every line says so.
    //
    // A missing calibration is a refusal before allocation, not a fallback: a run that silently judged
    // numerics against an unmeasured band would print a verdict that means nothing.
    std::vector<CalRow> cal;
    std::string cal_why;
    if (o.calibration.empty()) o.calibration = cal_default_path(o.program);
    if (o.mode == Options::kTime) {
        const bool loaded = load_calibration(o.calibration, cal, cal_why);
        std::vector<std::string> missing;
        for (int m : o.rows) {
            for (const auto &sh : kShapes) {
                if (!o.shapes.empty()) {
                    bool want = false;
                    for (const auto &w : o.shapes) if (w == sh.name) want = true;
                    if (!want) continue;
                }
                double f = 0.0, rl = 0.0, tl = 0.0;
                std::string p;
                if (!resolve_band(sh, m, o, cal, f, p, rl, tl))
                    missing.push_back(cal_key(sh.name, m, o.check_rows));
            }
        }
        if (!loaded || !missing.empty()) {
            std::printf("CALIBRATION-NEEDED file=%s why=%s missing=%zu of %zu problems "
                        "missing_keys=%s%s%s\n", o.calibration.c_str(),
                        loaded ? "some (shape, M, check_rows) keys absent" : cal_why.c_str(),
                        missing.size(), (size_t)o.rows.size() * (sizeof(kShapes) / sizeof(kShapes[0])),
                        missing.empty() ? "-"
                                        : (missing.size() > 6 ? missing[0] + ",..." : missing[0])
                                              .c_str(),
                        "", "");
            std::printf("CALIBRATION-NEEDED cmd=\"%s --floor-probe --calibration-out %s "
                        "--rows <the rows you intend to time> --check-rows %d\"\n",
                        o.program.c_str(), o.calibration.c_str(), o.check_rows);
            std::printf("CALIBRATION-NEEDED note=\"the probe takes no timing but does launch one call "
                        "per (shape, M, seed), so it needs the GPU lock like any device run. --cos-floor "
                        "does NOT substitute: it overrides the cosine half only, and the cancellation "
                        "band (cancel_rel, cancel_tol) still has to come from a measured CAL row\"\n");
            return 2;
        }
        std::printf("CALIBRATION mode=measured file=%s rows_loaded=%zu check_rows=%d margin=%.6e "
                    "cancel_margin=%.2f fullmag_reject=%.1e cos_override=%s note=\"the floor and the "
                    "cancellation band below are per (shape, M) and name their own provenance on every "
                    "COMPARE-MODE line\"\n", o.calibration.c_str(), cal.size(), o.check_rows,
                    o.floor_margin, o.cancel_margin, o.fullmag_rel,
                    o.cos_floor_given ? std::to_string(o.cos_floor).c_str() : "none");
    }

    int lt_version = 0, hb_version = 0;
    int device = 0, major = 0, minor = 0, cu = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&device) == hipSuccess &&
        hipGetDeviceProperties(&prop, device) == hipSuccess) {
        major = prop.major; minor = prop.minor; cu = prop.multiProcessorCount;
    }
    hipblasLtHandle_t lt = nullptr;
    const hipblasStatus_t cs = hipblasLtCreate(&lt);
    if (cs != HIPBLAS_STATUS_SUCCESS) {
        std::printf("FATAL hipblasLtCreate status=%d\n", (int)cs);
        return 2;
    }
    hipblasLtGetVersion(lt, &lt_version);
    hipblasGetVersion(nullptr, &hb_version);
    std::printf("ENV device=\"%s\" arch=gfx%d%d CUs=%d hipblaslt=%d.%d.%d hipblas=%d.%d.%d "
                "algo_path=\"C API: hipblasLtMatmulAlgoGetHeuristic only; C++ ext getAllAlgos for "
                "the full catalogue\"\n",
                prop.name[0] ? prop.name : "?", major, minor, cu, lt_version / 10000,
                lt_version / 100 % 100, lt_version % 100, hb_version / 10000,
                hb_version / 100 % 100, hb_version % 100);

    if (o.mode == Options::kVerify) {
        std::printf("MODE verify - descriptors + enumeration only. No operand buffer was allocated, "
                    "no matmul was launched and no timing was taken; hipblasLtCreate() is the only "
                    "call that touched the device (context + device properties).\n");
    }
    // The capture for --selftest-sys. Opened before the loop so a failure to open is reported once, on
    // stdout, and turns into a nonzero exit rather than a silently empty file.
    FILE *dump = nullptr;
    int dumped = 0;
    if (!o.algo_dump.empty() && o.mode == Options::kVerify) {
        dump = std::fopen(o.algo_dump.c_str(), "w");
        if (!dump)
            std::printf("FATAL cannot_open --algo-dump file=%s\n", o.algo_dump.c_str());
        else
            std::printf("ALGO-DUMP path=%s header=ALGODUMP\n", o.algo_dump.c_str());
    }

    void *ws = nullptr;
    hipblasHandle_t hb = nullptr;
    if (o.mode == Options::kTime || o.mode == Options::kFloorProbe) {
        CK(hipMalloc(&ws, o.ws_bytes));
        if (o.engine_ref) {
            // The engine's own route rebuilt here rather than linked out of the engine: one handle,
            // one pinned workspace, default stream (engine/kernels_gemm.cpp:115-145). The same 128 MiB
            // goes to hipBLASLt, which is sound because exactly one library is ever in flight.
            BK(hipblasCreate(&hb));
            BK(hipblasSetWorkspace(hb, ws, o.ws_bytes));
            BK(hipblasSetStream(hb, (hipStream_t)nullptr));
        }
    }

    const float alpha = 1.0f, beta = 0.0f;
    // ---- --floor-probe: measure the floor, write the file, stop. No timing line exists in this mode,
    // and the whole product is one CAL row per (shape, M).
    if (o.mode == Options::kFloorProbe) {
        std::printf("MODE floor-probe - one default-algo call and one engine call per (shape, M, seed), "
                    "plus host reorderings and injections. No timing was taken and no ms or TFLOP/s "
                    "line exists in this mode.\n");
        if (!o.engine_ref)
            std::printf("NOTE probe_engine_witness=absent reason=--engine-ref_0 -> the calibration will "
                        "carry worst_engine_cos=0 and lose the known-correct leg; that is a weaker "
                        "calibration, not a faster one\n");
        FILE *calout = std::fopen(o.calibration_out.c_str(), "w");
        if (!calout) {
            std::printf("FATAL cannot_open --calibration-out file=%s\n", o.calibration_out.c_str());
            if (hb) (void)hipblasDestroy(hb);
            if (ws) (void)hipFree(ws);
            if (lt) (void)hipblasLtDestroy(lt);
            return 2;
        }
        std::printf("CALIBRATION-OUT path=%s header=CAL check_rows=%d seeds=%d margin=%.6e "
                    "max_ulp=%.3f note=\"truncated on open: a probe that appends to a stale file can "
                    "serve a floor from a different build\"\n", o.calibration_out.c_str(),
                    o.check_rows, o.seeds, o.floor_margin, o.max_ulp);
        int pcount = 0, written = 0, refused = 0, could_not = 0;
        for (int m : o.rows) {
            for (const auto &sh : kShapes) {
                if (!o.shapes.empty()) {
                    bool want = false;
                    for (const auto &w : o.shapes) if (w == sh.name) want = true;
                    if (!want) continue;
                }
                ++pcount;
                const int r = probe_problem(lt, hb, sh, m, o, alpha, beta, ws, calout, prop.name);
                if (r == 0) ++written;
                else if (r == 1) ++refused;
                else ++could_not;
            }
        }
        std::fclose(calout);
        std::printf("FLOOR-SUMMARY problems=%d results=%d rows_written=%d refused=%d could_not_run=%d "
                    "path=%s rule=\"written+refused+could_not_run must equal problems, or some (shape, "
                    "M) was neither measured nor named\"\n", pcount, written + refused + could_not,
                    written, refused, could_not, o.calibration_out.c_str());
        const bool clean =
            (written + refused + could_not == pcount) && pcount > 0 && refused == 0 && could_not == 0;
        std::printf("FLOOR %s - %d CAL row(s) in %s; every later --time run reads its WHOLE band "
                    "(cos floor and cancellation thresholds) from this file and refuses without a row "
                    "for its own (shape, M, check_rows) (--calibration %s)\n", clean ? "ok" : "FAILED",
                    written, o.calibration_out.c_str(), o.calibration_out.c_str());
        if (hb) (void)hipblasDestroy(hb);
        if (ws) (void)hipFree(ws);
        if (lt) (void)hipblasLtDestroy(lt);
        return clean ? 0 : 1;
    }

    int problems = 0, refused = 0, heur_empty = 0, timed_problems = 0, cases_printed = 0;
    double min_ratio = 1e9, max_ratio = 0.0;
    for (int m : o.rows) {
        for (const auto &sh : kShapes) {
            if (!o.shapes.empty()) {
                bool want = false;
                for (const auto &w : o.shapes)
                    if (w == sh.name) want = true;
                if (!want) continue;
            }
            ++problems;
            if (o.mode == Options::kVerify) {
                Plan plan;
                std::string why;
                if (!plan.build(lt, m, sh.n, sh.k, o.ws_bytes, why)) {
                    std::printf("REFUSE shape=%s M=%d reason=plan:%s\n", sh.name, m, why.c_str());
                    std::printf("CASE mode=verify shape=%s M=%d N=%d K=%d status=refused "
                                "reason=plan:%s\n", sh.name, m, sh.n, sh.k, why.c_str());
                    ++cases_printed;
                    ++refused;
                    plan.free();
                    continue;
                }
                Enumeration en = enumerate(lt, plan, o, alpha, beta);
                print_enumeration(sh, m, en, o);
                if (dump) {
                    ++dumped;
                    for (const auto &e : en.algos) algo_dump_line(dump, sh, m, e);
                }
                if (en.heur_returned == 0) {
                    std::printf("PROBLEM shape=%s M=%d heuristic_returned=0 - the heuristic itself "
                                "refused this configuration (status %d)\n",
                                sh.name, m, en.heur_status);
                    ++heur_empty;
                }
                // One line per problem, itemised rather than counted. A bare `problems=21` is the
                // same defect shape as a gate that prints "1023 missing" and carries on: the reader
                // tabulates the number and never sees which 21 were meant. `cases_printed` is then
                // required to equal `problems` at the end, so the itemisation cannot be empty.
                size_t nsup = 0, wsfit = 0;
                int dseq = -1;
                for (const auto &e : en.algos) {
                    if (e.supported) ++nsup;
                    if (e.supported && e.ws <= o.ws_bytes) ++wsfit;
                    if (e.is_default) dseq = e.seq;
                }
                std::string reason = "-";
                if (en.heur_returned == 0) reason = "heuristic_returned_0_status_" + std::to_string(en.heur_status);
                else if (wsfit == 0) reason = "no_algo_fits_allocated_workspace";
                std::printf("CASE mode=verify shape=%s M=%d N=%d K=%d status=%s heuristic_status=%d "
                            "heuristic_returned=%d enumerated=%zu supported=%zu fits_workspace=%zu "
                            "default_seq=%d reason=%s\n",
                            sh.name, m, sh.n, sh.k, reason == "-" ? "ok" : "problem",
                            en.heur_status, en.heur_returned, en.algos.size(), nsup, wsfit, dseq,
                            reason.c_str());
                ++cases_printed;
                if (reason != "-") ++refused;   // a problem in verify mode = a shape --time cannot run
                plan.free();
            } else {
                // The per-problem band: the resolved floor and the sentence naming where it came from
                // travel with the problem, so a COMPARE-MODE line can never quote a floor whose source
                // belongs to a different (shape, M).
                Options po = o;
                if (!resolve_band(sh, m, o, cal, po.cos_floor, po.band_provenance, po.cancel_rel,
                                  po.cancel_tol)) {
                    std::printf("CASE mode=%s shape=%s M=%d N=%d K=%d status=refused "
                                "reason=calibration_missing(%s)\n",
                                o.check_only ? "check-only" : "time", sh.name, m, sh.n, sh.k,
                                cal_key(sh.name, m, o.check_rows).c_str());
                    ++cases_printed;
                    ++refused;
                    continue;
                }
                double ratio = 0.0;
                bool have = false;
                const bool ran = time_problem(lt, hb, sh, m, po, alpha, beta, ws, ratio, have);
                std::printf("CASE mode=%s shape=%s M=%d N=%d K=%d status=%s reason=%s%s\n",
                            o.check_only ? "check-only" : "time", sh.name, m, sh.n, sh.k,
                            ran ? "ok" : "refused",
                            ran ? "-" : "see the REFUSE/CHECK lines above for this shape",
                            (ran && have) ? (" ratio=" + std::to_string(ratio)).c_str() : "");
                ++cases_printed;
                if (!ran) {
                    ++refused;
                    continue;
                }
                ++timed_problems;
                min_ratio = std::min(min_ratio, ratio);
                max_ratio = std::max(max_ratio, ratio);
            }
        }
    }

    if (dump) {
        std::fclose(dump);
        dump = nullptr;
        std::printf("ALGO-DUMP done path=%s algos=%d\n", o.algo_dump.c_str(), dumped);
    } else if (!o.algo_dump.empty() && o.mode == Options::kVerify) {
        std::printf("FATAL --algo-dump was requested and the file could not be written, so the "
                    "capture this run exists to produce does not exist\n");
        if (hb) (void)hipblasDestroy(hb);
        if (ws) (void)hipFree(ws);
        if (lt) (void)hipblasLtDestroy(lt);
        return 2;
    }

    if (problems == 0) {
        // Same lesson as the --shapes name check: an empty sweep must not be able to print a verdict.
        std::printf("FATAL no shape matched (rows=%zu shapes=%zu) - nothing was enumerated\n",
                    o.rows.size(), o.shapes.size());
        if (hb) (void)hipblasDestroy(hb);
        if (ws) (void)hipFree(ws);
        if (lt) (void)hipblasLtDestroy(lt);
        return 2;
    }
    // The itemisation is what the counts claim to be. A summary that prints problems=21 while zero
    // CASE lines were emitted is the same failure as a gate counting 1023 missing rows and passing:
    // the number is real but nothing can be read from it, so the equality is enforced, not assumed.
    if (cases_printed != problems) {
        std::printf("FATAL cases_printed=%d but problems=%d - the per-problem itemisation is "
                    "incomplete, so no verdict below can be trusted\n", cases_printed, problems);
        if (hb) (void)hipblasDestroy(hb);
        if (ws) (void)hipFree(ws);
        if (lt) (void)hipblasLtDestroy(lt);
        return 3;
    }

    if (o.mode == Options::kVerify) {
        std::printf("VERIFY %s - problems=%d cases_printed=%d refused=%d heuristic_empty=%d "
                    "algo_dumped=%d. One CASE line per problem, itemised above; enumeration printed; "
                    "no device timing exists in this mode.\n",
                    (refused == 0 && heur_empty == 0) ? "ok" : "FAILED", problems, cases_printed,
                    refused, heur_empty, dumped);
        if (lt) hipblasLtDestroy(lt);
        return (refused == 0 && heur_empty == 0) ? 0 : 1;
    }

    if (o.check_only) {
        std::printf("CHECK-ONLY-SUMMARY problems=%d cases_printed=%d refused=%d band_passed=%d "
                    "no_timing_taken=1 - this mode stops after the band, so no ms or TFLOP/s line "
                    "exists to quote\n", problems, cases_printed, refused, timed_problems);
        if (hb) (void)hipblasDestroy(hb);
        if (ws) (void)hipFree(ws);
        if (lt) (void)hipblasLtDestroy(lt);
        return refused ? 1 : 0;
    }

    std::printf("SWEEP problems=%d cases_printed=%d refused=%d timed=%d min_ratio=%.3f max_ratio=%.3f "
                "rule=\"ratio = default-ranked best_ms / best best_ms, so 1.000 means the heuristic "
                "already picked the fastest enumerated kernel\"\n",
                problems, cases_printed, refused, timed_problems, timed_problems ? min_ratio : 0.0,
                timed_problems ? max_ratio : 0.0);
    if (timed_problems) {
        std::printf("VERDICT %s\n",
                    max_ratio < 1.10
                        ? "the library default is within 1.10x of the best enumerated algorithm at "
                          "every shape measured: a hand WMMA GEMM has to win on its own merits "
                          "(HANDOFF.md section 5)"
                        : "the default is leaving at least 10% on the table somewhere; read the "
                          "RESULT line(s) with ratio > 1.10");
    } else {
        std::printf("VERDICT nothing was timed - read the REFUSE lines\n");
    }
    if (hb) (void)hipblasDestroy(hb);
    if (ws) (void)hipFree(ws);
    if (lt) (void)hipblasLtDestroy(lt);
    return refused ? 1 : 0;
}
