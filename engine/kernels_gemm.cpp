// The hipBLAS route of route (c): one handle, one pinned workspace, and the one call form the whole
// engine uses for a wave-width matmul (§6 item 1 of progress/2026-10-13/prefill-route-spike-2026-10-13.md).
//
// TWO CALL PATHS LIVE HERE NOW, selected by `--gemm` at run time and not by a build define, so one
// binary can be A/Bed against itself (the `--attn-touch` precedent):
//
//   `--gemm library` (the shipped default) → `hipblasGemmEx`, `HIPBLAS_GEMM_DEFAULT`. The kernel that
//       runs is hipBLAS's own choice and the caller has no way to ask which it was or to pin it.
//   `--gemm lt` → `hipblasLtMatmul` with one algorithm fetched per (shape, M) at plan time. Funded by
//       D26: at the shipping width the same 150-call mix costs 18.67 ms here, 17.42 ms through the
//       hipBLASLt heuristic and 15.90 ms with a pinned table (`prototype/gemm_algo_sweep/results.md`
//       §3a), and on three of the seven shapes the Lt heuristic's rank-0 is already its own fastest
//       enumerated kernel while `hipblasGemmEx` is still 1.22-1.63x slower than it.
//
// Everything else about the matmul is deliberately identical between them — same operands, same
// column-major view of the same row-major buffers, same bf16-in/f32-accumulate/bf16-`C` contract, same
// 128 MiB workspace, same legacy default stream — because the only thing under test is which library
// picks the kernel. What differs is the accumulation ORDER, which is why `lt` is gated by the same
// cascade band as `library` and not by byte-identity (`engine/gate.py --gemm lt`).
//
// Why this is its own TU: hipBLAS is the only thing in the engine that links `-lhipblas`, and the
// handle is process state — keeping it in one file means "what allocates, when, on which stream" has
// one answer instead of four. The routed entry points that use it (`proj_gemm` in kernels_proj.cpp,
// `mlp_gateup_gemm`/`mlp_down_gemm` in kernels_mlp.cpp) live with the families they belong to.
//
// THE CALL FORM IS NOT REWRITTEN HERE. `gemm_common::gemm_stored_weight` (tools/gemm_common.h) is the
// authority for it — `OP_T` on the weight as the shard stores it, `ld=k`, 16-bit `C` with `ldc=n` —
// and it is shared rather than copied because getting an `OP_T`/`ld` pair backwards does not error, it
// computes a different product (spike §5 records two ways that went wrong while it was being derived).
// The same function is what `tools/gemm_peak --check` and `tools/prefill_mix --check` certify against
// a CPU reference, so the probe certifies the call the engine makes rather than a near-copy of it.
//
// NOTHING ALLOCATES PER CALL, and that is the point of `ensure_init`: one handle and one 128 MiB
// workspace, created once, on the default stream, so a wave's ordering against the hand kernels is
// the ordering the engine already had (every submission in a body — launches and `hipMemcpyAsync`
// alike — is on the legacy default stream, and so is every hipBLAS call). `tools/prefill_mix.cpp`
// pins its workspace for the same reason §3.1 gives: workspace size feeds split-K eligibility, so a
// route that left it to the library's default would be making kernel choice a function of what else
// the allocator had recently freed.
#include "kernels.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "engine/decode_step.h"   // kMaxWaveRows: a pin is keyed to the shipping wave width
#include "engine/state.h"
#include "tools/gemm_common.h"

#define ENGINE_HB(x)                                                                     \
    do {                                                                                 \
        const hipblasStatus_t s_ = (x);                                                   \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                               \
            throw std::runtime_error(std::string("hipblas: ") + std::to_string((int)s_) + \
                                     " at " __FILE__ + ":" + #x);                        \
        }                                                                                 \
    } while (0)

#define ENGINE_CK(x)                                                                \
    do {                                                                            \
        const hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                      \
            throw std::runtime_error(std::string("hip: ") + hipGetErrorString(e_) + \
                                     " at " __FILE__ + ":" + #x);                   \
        }                                                                            \
    } while (0)

// hipBLASLt answers with the same `hipblasStatus_t` enum, but a failure has to name the library whose
// call actually failed: "hipblas: 13" on a line that ran `hipblasLtMatmul` would send the next reader
// to the wrong .so.
#define ENGINE_LT(x)                                                                      \
    do {                                                                                  \
        const hipblasStatus_t s_ = (x);                                                    \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                                \
            throw std::runtime_error(std::string("hipblaslt: ") + std::to_string((int)s_) + \
                                     " at " __FILE__ + ":" + #x);                          \
        }                                                                                  \
    } while (0)

namespace engine::kernels {
namespace {

// 128 MiB, the size `tools/prefill_mix.cpp` pins and the size §3.1's determinism pass and §7's W1/W2
// window were measured with. Memory capacity is not a constraint for this model (AGENTS.md: the whole
// bf16 checkpoint is ~4 % of the pool), so this is sized to match the instrument that graded the
// route, not to save anything.
constexpr size_t kWorkspaceBytes = 128ull << 20;

struct GemmRoute {
    hipblasHandle_t handle = nullptr;
    void* workspace = nullptr;
    std::string description;
    // Which call path `gemm_bf16` dispatches to. Process state for the same reason the handle is:
    // one run runs one route, and `--gemm` is a run flag. Defaults to the shipped path so a driver
    // that never names the flag (or never calls `gemm_route_init`) gets the bits it got before.
    WaveGemm path = WaveGemm::kLibrary;
};

GemmRoute& route() {
    static GemmRoute r;
    return r;
}

// The workspace is `hipMalloc`ed, not managed: `devices/strix-halo-unified-memory.md` records
// `hipMallocManaged` degrading SILENTLY to ~52 GB/s near 32 GiB on this box, and a matmul reading
// split-K partials out of degraded pages is exactly the kind of finding that never gets printed.
std::string build_route() {
    GemmRoute& r = route();
    if (r.handle) return r.description;

    hipblasHandle_t handle = nullptr;
    ENGINE_HB(hipblasCreate(&handle));
    void* workspace = nullptr;
    try {
        ENGINE_CK(hipMalloc(&workspace, kWorkspaceBytes));
        ENGINE_HB(hipblasSetWorkspace(handle, workspace, kWorkspaceBytes));
        // The legacy default stream — the one `<<<...>>>` and `hipMemcpyAsync` in
        // `submit_wave_body` submit to. Said explicitly rather than left to the handle's default so
        // that "library calls cannot overtake the hand kernels around them, or be overtaken" is a line
        // in the code and not an assumption about hipBLAS.
        ENGINE_HB(hipblasSetStream(handle, (hipStream_t)nullptr));
    } catch (...) {
        if (workspace) (void)hipFree(workspace);
        (void)hipblasDestroy(handle);
        r.handle = nullptr;
        r.workspace = nullptr;
        throw;
    }
    r.handle = handle;
    r.workspace = workspace;
    std::ostringstream os;
    os << "route=hipblas handle=1 workspace_mib=" << (kWorkspaceBytes >> 20)
       << " dtype=bf16-in/bf16-C/f32-compute stream=default";
    r.description = os.str();
    return r.description;
}

// ---- the hipBLASLt path (`--gemm lt`) ---------------------------------------------------------------

// How many algorithms the heuristic is asked for per (shape, M). One would reproduce the library's own
// default; the wider answer is here so a shape whose measured winner is NOT rank-0 can be adopted by a
// pin row without a second selection path. The query is per (shape, M) at plan time, never per call.
constexpr int kHeuristicWidth = 8;

// The shape census the plan pass walks: the same seven distinct (N,K) leaves that
// `prototype/gemm_algo_sweep --list` books and exits 0 on, spelled with the engine's own constants
// rather than copied numbers. Two leaves sharing a shape are one row here because an Lt plan is keyed
// by (N, K, M) and not by weight name.
struct ShapeRow {
    const char* name;
    int n;
    int k;
};
constexpr ShapeRow kLtShapes[] = {
    {"q_proj", kQprojRows, kHidden},
    {"k_proj+v_proj", kKvProjRows, kHidden},
    {"o_proj+out_proj", kHidden, kQwidth},
    {"in_proj_qkv", kQkvRows, kHidden},
    {"in_proj_z", kValueDim, kHidden},
    {"gate_proj+up_proj", kIntermediate, kHidden},
    {"down_proj", kHidden, kIntermediate},
};

// A per-shape algorithm pin: the one way a measured winner that is not the heuristic's rank-0 gets
// adopted without touching a call site. `algo_hex` is the 48 hex characters of one
// `hipblasLtMatmulAlgo_t` — 24 bytes: `uint8_t data[16]` plus `size_t max_workspace_bytes` — in the
// exact form `prototype/gemm_algo_sweep/gemm_algo_sweep --verify --algo-dump` writes. The library's own
// header licenses that round trip ("can be trivially serialized and later restored for use with the
// same version of the library to save compute time", hipblaslt.h:419-433), and the same sentence is
// why a pin is provenance-bound: after a hipBLASLt version bump (printed on the `GEMM` line) a pin must
// be re-captured, not inherited.
struct AlgoPin {
    const char* name;
    int n;
    int k;
    int m;
    const char* algo_hex;
};

// ONE shape is pinned today, and the reason it is pinned is the reason the table exists at all: at the
// shipping width the heuristic's rank-0 — which is all an unpinned `lt` run gets — is SLOWER than the
// call this engine already makes through hipBLAS. Taking rank-0 there gives back part of the win
// `prototype/gemm_algo_sweep/results.md` §3c measured. This used to be three rows; the two others were
// warm-burst winners that lost in situ (§3d), and the table shrank to this while this paragraph kept
// claiming three. The test is therefore `default_ms` vs `engine_call_ms`, never `best_ms` vs either, and
// that section's lesson 2 is that reading the
// "best" column as though it were the "default" column is how `mlp_down` was shipped at 0.812x.
// Numbers are `prototype/gemm_algo_sweep/results.md` §1's `RESULT` lines at M=512 (the sweep's
// `--rows 512 --max-algos 12` window); the bytes are that same build's `--verify --algo-dump`
// `ALGODUMP` lines, matched to each winner by `best_seq`/`best_idx` — not hand-written.
//
//   shape              rank-0 ms  engine ms  winner ms   rank-0/engine   winner/engine
//   o_proj+out_proj      0.0871     0.0801     0.0766      1.087x slow     0.956x
//   in_proj_qkv          0.2000     0.1989     0.1862      1.006x slow     0.936x
//   down_proj            0.1439     0.1313     0.1169      1.096x slow     0.890x
//
// The other four shapes of `kLtShapes` need nothing and so get nothing, by the same measurement: their
// rank-0 already beats the engine's own call (q_proj 0.820x, `k_proj+v_proj` 0.616x, in_proj_z 0.736x,
// gate_proj+up_proj 0.861x of `engine_call_ms`), and a pin is provenance-bound — it freezes one kernel
// against one hipBLASLt version (these were captured on 10.4.1) — so a shape that owes nothing is not
// worth a re-capture obligation. `gate_proj+up_proj`'s winner IS a further 1.067x over its own rank-0
// and is deliberately NOT taken here: that is new speed, not a recovered regression, and it belongs in
// its own funded change with its own A/B.
//
// The table above is the pre-adjudication candidate list, kept because it is the evidence for the
// decision below and not the decision. `down_proj` is the row that survived in situ; `o_proj+out_proj`
// and `in_proj_qkv` did not, and cost `proj` 5.5 ms when tried.
//
// Two facts about what a pin may name, because both were assumed this session and only one of them is
// true. TRUE: a pin bypasses the heuristic query entirely — the predicates below are "valid hex, not
// all zero, `max_workspace_bytes <= kWorkspaceBytes`" and nothing else — so `kHeuristicWidth` does not
// bound what can be pinned, and a winner the engine's own 8-wide query would never return is still
// admissible. FALSE: that the 12-candidate sweeps so far priced the whole field. Deepening the sweep to
// 64 candidates finds kernels at ranks 16-55 a further 1.13-1.39x over rank-0 on four shapes
// (`progress/2026-10-16/prefill-decision-log-2026-10-16.md` D31), so the candidate list above is
// measured against a truncated field and every ratio in it is a floor, not an answer.
//
// Pins bind on (N, K, M), so they cover the FULL wave width and nothing else: a partial wave (640 lanes
// = 512 + 128 rows) or a prompt-shaped body (dec8 plans M=17) still takes the heuristic. That is the
// same state the 1.120x A/B was measured in, and `GEMM-LT` prints `source=` for every plan, so which
// kernel served which width is readable in any log rather than inferred from a table.
//
// One field of that log line to read carefully: on a `source=pin` line `ws=` is the algo record's own
// stamped `max_workspace_bytes`, and for a record that came back from the heuristic — which all three of
// these did — hipBLASLt stamps that field with the preference cap the query ran under (128 MiB here,
// the whole buffer) rather than with what the kernel needs. Re-running `--verify --algo-dump` shows it:
// every heuristic-ranked record in that capture carries 134217728 there, and the records `getAllAlgos`
// adds carry SIZE_MAX, while the real per-kernel demand lives in the separate `workspaceSize` the sweep
// prints as `ws=` and `hipblasLtMatmulAlgoCheck` fills — that is the field `skipped_workspace=15077` in
// §4 of `prototype/gemm_algo_sweep/results.md` counts. Two consequences, both wanted: `ws=134217728` on a pin line is a ceiling the library wrote and
// not a demand by the kernel (the launch passes the whole pinned buffer on every path anyway, see
// `gemm_bf16` below), and the refuse-if-bigger-than-the-buffer predicate above also refuses any
// enumerated record, so a pin can only ever name an algorithm the heuristic itself would have served.
// Which pins are in force for this run: "*" means all, an empty list means none. This is a
// measurement variable, so it is a runtime knob and not a rebuild — pin-vs-rank-0-vs-pin-subset has to
// be one window with one binary, and the sweep already proved that a burst column and the in-situ
// family do not move by the same factor, so the subset question can only be answered in situ.
std::vector<std::string> g_pin_allow{std::string("*")};

bool pin_allowed(const char* name) {
    for (const std::string& s : g_pin_allow) {
        if (s == "*" || s == name) return true;
    }
    return false;
}

const std::vector<AlgoPin>& algo_pins() {
    static const std::vector<AlgoPin> kPins = [] {
        std::vector<AlgoPin> v;
        // Pinned by measurement, and only this one. The other two candidates (`o_proj+out_proj`,
        // `in_proj_qkv`) were warm-burst winners — rank-0 slower than the engine's call, this algo
        // faster — and measured in situ they cost `proj` 5.5 ms of a 24.8 ms family (24.82 -> 30.35 ms,
        // one window, three reps, `prototype/gemm_algo_sweep/results.md` §3d). A burst winner can be an
        // in-situ loser, so a pin earns its place in situ or not at all. `down_proj` stayed: 7.81 ->
        // 6.78 ms against the library's 6.43 ms, the only shape where rank-0 was a regression.
        v.push_back({"down_proj", kHidden, kIntermediate, kMaxWaveRows,
                     "890300000000000000000000000000000000000800000000"});
        return v;
    }();
    return kPins;
}

// An all-zero `hipblasLtMatmulAlgo_t` is an uninitialised one, and launching it is the fault the sweep
// spent two device windows on. Same predicate the sweep's `validate_args` runs (`algo_not_all_zero`).
bool algo_all_zero(const hipblasLtMatmulAlgo_t& a) {
    const unsigned char* p = (const unsigned char*)&a;
    for (size_t i = 0; i < sizeof(a); ++i) {
        if (p[i] != 0) return false;
    }
    return true;
}

std::string algo_hex(const hipblasLtMatmulAlgo_t& a) {
    static const char* kHex = "0123456789abcdef";
    const unsigned char* p = (const unsigned char*)&a;
    std::string out;
    out.reserve(sizeof(a) * 2);
    for (size_t i = 0; i < sizeof(a); ++i) {
        out.push_back(kHex[p[i] >> 4]);
        out.push_back(kHex[p[i] & 0xF]);
    }
    return out;
}

// The inverse of `algo_hex`. Length is checked against the struct, not assumed: a truncated pin parses
// into a partly-zero algo, which is exactly the uninitialised record the predicate above exists to refuse.
bool algo_from_hex(const std::string& hex, hipblasLtMatmulAlgo_t& out) {
    if (hex.size() != sizeof(out) * 2) return false;
    unsigned char* p = (unsigned char*)&out;
    for (size_t i = 0; i < sizeof(out); ++i) {
        int byte = 0;
        for (int j = 0; j < 2; ++j) {
            const char c = hex[i * 2 + j];
            int d;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
            else return false;
            byte = byte * 16 + d;
        }
        p[i] = (unsigned char)byte;
    }
    return true;
}

// One (shape, M) problem: its descriptors and the one algorithm chosen for it. Built once, reused for
// every call of that width — the engine issues 150 library calls per 24-layer wave and a heuristic query
// per call would cost more than the win the route is chasing.
struct LtPlan {
    hipblasLtMatmulDesc_t desc = nullptr;
    // Names say which buffer each descriptor describes, in BLAS operand order, not in the engine's:
    // `la` is the WEIGHT (OP_T, lda=k) and `lb` is the ACTIVATION (OP_N, ldb=k). `gemm_stored_weight`
    // names its parameter `A` for the activation, and the two conventions describing the same two
    // buffers is what made the sweep fault twice (`prototype/gemm_algo_sweep/README.md` §1a).
    hipblasLtMatrixLayout_t la = nullptr;
    hipblasLtMatrixLayout_t lb = nullptr;
    hipblasLtMatrixLayout_t lc = nullptr;
    hipblasLtMatrixLayout_t ld = nullptr;   // D: same buffer as C, because beta is 0
    hipblasLtMatmulAlgo_t algo{};
    std::string provenance;                 // the `GEMM-LT` line, built once per key
};

struct LtRoute {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    std::map<std::tuple<int, int, int>, LtPlan> plans;   // keyed (n, k, m)
    std::string description;
};

LtRoute& lt_route() {
    static LtRoute r;
    return r;
}

std::string build_lt_route() {
    LtRoute& r = lt_route();
    if (r.handle) return r.description;

    hipblasLtHandle_t handle = nullptr;
    ENGINE_LT(hipblasLtCreate(&handle));
    void* workspace = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    try {
        // `hipMalloc`, not managed, for the reason `build_route` gives: managed degrades SILENTLY near
        // 32 GiB on this box, and a split-K partial landing in degraded pages is a finding that never
        // gets printed. Same 128 MiB as the hipBLAS route — workspace size feeds split-K eligibility,
        // so the two paths must offer the same kernel set for the A/B to be about selection alone.
        ENGINE_CK(hipMalloc(&workspace, kWorkspaceBytes));
        ENGINE_LT(hipblasLtMatmulPreferenceCreate(&pref));
        const uint64_t ws = kWorkspaceBytes;
        ENGINE_LT(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &ws, sizeof(ws)));
    } catch (...) {
        if (pref) (void)hipblasLtMatmulPreferenceDestroy(pref);
        if (workspace) (void)hipFree(workspace);
        (void)hipblasLtDestroy(handle);
        r.handle = nullptr;
        r.workspace = nullptr;
        r.pref = nullptr;
        throw;
    }
    r.handle = handle;
    r.workspace = workspace;
    r.pref = pref;

    int version = 0;
    std::string lt_version = "unknown";
    if (hipblasLtGetVersion(handle, &version) == HIPBLAS_STATUS_SUCCESS && version > 0) {
        lt_version = std::to_string(version / 10000) + "." + std::to_string(version / 100 % 100) + "."
                     + std::to_string(version % 100);
    }
    std::ostringstream os;
    // `pins=<in force>/<in the table>`, both counted by THIS process out of algo_pins(). Two numbers,
    // not one, because one is undecidable from the outside: `pins=1` is the correct line both for a
    // one-entry table asked for in full and for a three-entry table the run was never given, so a gate
    // judging `--gemm-pins all` would have to remember the table size to check it — and a remembered
    // number is the provenance rot D28 records for the band (prototype/gemm_algo_sweep/results.md §3d,
    // where a band was cited against a pin set the binary no longer used). With the denominator in the
    // line every spec is an equality: `all` means allowed == total, `none` means allowed == 0, and a
    // csv means allowed == its own length. See `gemm_pin_checks` in engine/gate.py.
    os << "route=hipblaslt handle=1 workspace_mib=" << (kWorkspaceBytes >> 20)
       << " version=" << lt_version << " algo_source=heuristic-rank-0-per-shape pins="
       << [] { size_t n = 0; for (const AlgoPin& p : algo_pins()) { if (pin_allowed(p.name)) ++n; } return n; }()
       << "/" << algo_pins().size()
       << " stream=default";
    r.description = os.str();
    return r.description;
}

// Build (or fetch) the plan for one (n, k, m). The descriptor set is the engine's own call form translated
// into Lt objects — the derivation is `tools/gemm_common.h:111-126` and is NOT re-decided here:
// C_rm[m,n] = X_rm[m,k] · W_rm[n,k]^T, BLAS is column-major, so m_lt = n, n_lt = m, k_lt = k, with A = W
// (OP_T, lda=k), B = X (OP_N, ldb=k) and C/D of n rows at ldc=n. A pair transcribed backwards does not
// error; it computes a different product, which is why the extents are written from the same three
// numbers the launch uses and nowhere else.
LtPlan& lt_plan(LtRoute& r, const char* name, int n, int k, int m, std::string& built_line) {
    const auto key = std::make_tuple(n, k, m);
    auto it = r.plans.find(key);
    if (it != r.plans.end()) {
        built_line.clear();
        return it->second;
    }

    LtPlan p;
    ENGINE_LT(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const int32_t op_a = HIPBLAS_OP_T, op_b = HIPBLAS_OP_N;
    ENGINE_LT(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    ENGINE_LT(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    const int ra = k, ca = n, lda = k;      // A = W, stored [n,k] row-major, OP_T
    const int rb = k, cb = m, ldb = k;      // B = X, the activation [m,k] row-major, OP_N
    const int rc = n, cc = m, ldc = n;      // C/D = the column-major view of row-major C[m,n]
    ENGINE_LT(hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16BF, (uint64_t)ra, (uint64_t)ca, (int64_t)lda));
    ENGINE_LT(hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16BF, (uint64_t)rb, (uint64_t)cb, (int64_t)ldb));
    ENGINE_LT(hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_16BF, (uint64_t)rc, (uint64_t)cc, (int64_t)ldc));
    ENGINE_LT(hipblasLtMatrixLayoutCreate(&p.ld, HIP_R_16BF, (uint64_t)rc, (uint64_t)cc, (int64_t)ldc));

    // The pin runs first: a shape with a measured winner uses it and says so, and a pin that cannot be
    // honoured is an error rather than a silent fall back to rank-0 (a log that says `source=pin` while
    // running rank-0 would be worse than having no pin at all).
    bool have_algo = false;
    std::string source = "heuristic-rank-0";
    size_t ws_requested = 0;
    float waves = 0.0f;
    int returned = 0, skipped_ws = 0, skipped_state = 0;
    for (const AlgoPin& pin : algo_pins()) {
        if (pin.n != n || pin.k != k || pin.m != m) continue;
        if (!pin_allowed(pin.name)) continue;  // filtered out by `--gemm-pins`; rank-0 stands
        hipblasLtMatmulAlgo_t pinned{};
        if (!algo_from_hex(pin.algo_hex, pinned) || algo_all_zero(pinned)
            || pinned.max_workspace_bytes > kWorkspaceBytes) {
            throw std::runtime_error(std::string("gemm lt: the pin for ") + pin.name + " (N="
                                     + std::to_string(n) + " K=" + std::to_string(k) + " M="
                                     + std::to_string(m) + ") is unusable — it must be "
                                     + std::to_string(sizeof(hipblasLtMatmulAlgo_t) * 2)
                                     + " hex chars, not all zero, and want at most "
                                     + std::to_string(kWorkspaceBytes >> 20)
                                     + " MiB of workspace");
        }
        p.algo = pinned;
        ws_requested = pinned.max_workspace_bytes;
        source = "pin";
        have_algo = true;
        break;
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> heur((size_t)kHeuristicWidth);
    if (!have_algo) {
        ENGINE_LT(hipblasLtMatmulAlgoGetHeuristic(r.handle, p.desc, p.la, p.lb, p.lc, p.ld, r.pref,
                                                 kHeuristicWidth, heur.data(), &returned));
        for (int i = 0; i < returned; ++i) {
            const hipblasLtMatmulHeuristicResult_t& h = heur[(size_t)i];
            // Every one of these three predicates was printed-but-unenforced somewhere in the sweep's
            // first draft, and the first two are the normal case rather than the error path: of 15 750
            // enumerated (algo, shape) entries 15 077 ask for more than this workspace
            // (`prototype/gemm_algo_sweep/results.md` §4), so a route that launched them all would be
            // writing past its own buffer 96 times in 100.
            if (h.state != HIPBLAS_STATUS_SUCCESS) {
                ++skipped_state;
                continue;
            }
            if (h.workspaceSize > kWorkspaceBytes || h.algo.max_workspace_bytes > kWorkspaceBytes
                || algo_all_zero(h.algo)) {
                ++skipped_ws;
                continue;
            }
            p.algo = h.algo;
            ws_requested = h.workspaceSize;
            waves = h.wavesCount;
            source = "heuristic-rank-" + std::to_string(i);
            have_algo = true;
            break;
        }
    }
    if (!have_algo) {
        std::ostringstream os;
        os << "gemm lt: no algorithm serves N=" << n << " K=" << k << " M=" << m
           << " within the " << (kWorkspaceBytes >> 20)
           << " MiB workspace budget (heuristic returned " << returned << ", skipped_workspace="
           << skipped_ws << ", skipped_not_success=" << skipped_state
           << "). Refusing rather than falling back to hipBLAS mid-run: a run whose route depends on"
              " which shapes happened to fit is not an A/B side";
        throw std::runtime_error(os.str());
    }

    std::ostringstream os;
    os << "GEMM-LT shape=" << (name ? name : "?") << " N=" << n << " K=" << k << " M=" << m
       << " source=" << source << " ws=" << ws_requested << " waves=" << waves
       << " skipped_ws=" << skipped_ws << " skipped_state=" << skipped_state << " heur_returned="
       << returned << " algo=" << algo_hex(p.algo) << "\n";
    p.provenance = os.str();
    built_line = p.provenance;
    auto inserted = r.plans.emplace(key, std::move(p));
    return inserted.first->second;
}

LtPlan& lt_plan_for(int m, int n, int k, std::string& built_line) {
    LtRoute& r = lt_route();
    // Named only for the log line; the key is the shape, so a call site that arrives here without a
    // census row (an odd body width, or a shape added later) still gets a plan and a printed one.
    const char* name = nullptr;
    for (const ShapeRow& s : kLtShapes) {
        if (s.n == n && s.k == k) {
            name = s.name;
            break;
        }
    }
    return lt_plan(r, name, n, k, m, built_line);
}

}  // namespace

// `--gemm-pins all|none|<csv of shape names>`. Throws on a name that is not in the pin table: a typo
// that silently meant "no pins" would price rank-0 twice and report the pin as a null.
void gemm_pins_select(const std::string& spec) {
    std::vector<std::string> v;
    if (spec == "all") {
        v.emplace_back("*");
    } else if (spec != "none") {
        size_t p = 0;
        while (p <= spec.size()) {
            size_t c = spec.find(',', p);
            const bool last = (c == std::string::npos);
            if (last) c = spec.size();
            if (c > p) v.push_back(spec.substr(p, c - p));
            if (last) break;
            p = c + 1;
        }
    }
    for (const std::string& name : v) {
        if (name == "*") continue;  // the "all" sentinel, not a shape name
        bool known = false;
        for (const AlgoPin& pin : algo_pins()) {
            if (pin.name == name) known = true;
        }
        if (!known) {
            throw std::runtime_error("gemm lt: --gemm-pins names '" + name
                                     + "', which is not in the pin table (use all, none, or a"
                                       " comma-separated list of pinned shape names)");
        }
    }
    g_pin_allow = v;
}

std::string gemm_route_init(WaveGemm want) {
    route().path = want;
    if (want == WaveGemm::kLibraryLt) return build_lt_route();
    if (want == WaveGemm::kLibrary) return build_route();
    // A hand run creates no library state at all — not a handle, not 128 MiB of workspace. Said out
    // loud because `gemm_route_init` used to mean "build hipBLAS", and a caller that passed `kHand` and
    // got a handle anyway would be allocating memory the pre-route binary never touched.
    return "route=none gemm=hand";
}

std::string gemm_route_plan(int rows) {
    if (route().path != WaveGemm::kLibraryLt) return std::string();
    LtRoute& r = lt_route();
    if (!r.handle) build_lt_route();
    std::string out;
    for (const ShapeRow& s : kLtShapes) {
        std::string line;
        lt_plan(r, s.name, s.n, s.k, rows, line);
        out += line;
    }
    return out;
}

bool wave_gemm_is_library(WaveGemm want, int rows) {
    const WaveGemm resolved = resolve_wave_gemm(want, rows);
    return resolved == WaveGemm::kLibrary || resolved == WaveGemm::kLibraryLt;
}

void gemm_route_shutdown() {
    GemmRoute& r = route();
    LtRoute& lr = lt_route();
    if (lr.handle) {
        // Plans first: each descriptor belongs to the handle being freed just below. Then the workspace,
        // which the handle holds a pointer into.
        for (auto& kv : lr.plans) {
            LtPlan& p = kv.second;
            if (p.ld) (void)hipblasLtMatrixLayoutDestroy(p.ld);
            if (p.lc) (void)hipblasLtMatrixLayoutDestroy(p.lc);
            if (p.lb) (void)hipblasLtMatrixLayoutDestroy(p.lb);
            if (p.la) (void)hipblasLtMatrixLayoutDestroy(p.la);
            if (p.desc) (void)hipblasLtMatmulDescDestroy(p.desc);
            p = LtPlan{};
        }
        lr.plans.clear();
        if (lr.pref) (void)hipblasLtMatmulPreferenceDestroy(lr.pref);
        (void)hipblasLtDestroy(lr.handle);
        (void)hipFree(lr.workspace);
        lr.handle = nullptr;
        lr.workspace = nullptr;
        lr.pref = nullptr;
        lr.description.clear();
    }
    if (!r.handle) return;
    // Handle first, then its workspace: the handle holds a pointer into the buffer being freed.
    (void)hipblasDestroy(r.handle);
    (void)hipFree(r.workspace);
    r.handle = nullptr;
    r.workspace = nullptr;
    r.description.clear();
}

WaveGemm parse_wave_gemm(const std::string& name) {
    if (name == "library") return WaveGemm::kLibrary;
    // `legacy` is a synonym for `library`, not a third route: the A/B being run is legacy-hipBLAS
    // against lt-hipBLASLt and one shell loop should be able to spell both sides. It never comes back
    // out — `wave_gemm_name` prints `library` — so a log stays comparable with the ones taken before
    // this synonym existed and `gate.py`'s route check still sees one spelling per route.
    if (name == "legacy") return WaveGemm::kLibrary;
    if (name == "lt") return WaveGemm::kLibraryLt;
    if (name == "hand") return WaveGemm::kHand;
    throw std::runtime_error("--gemm takes library|legacy|lt|hand, got '" + name + "'");
}

const char* wave_gemm_name(WaveGemm route) {
    switch (route) {
        case WaveGemm::kLibrary: return "library";
        case WaveGemm::kLibraryLt: return "lt";
        case WaveGemm::kHand: return "hand";
    }
    return "hand";
}

std::string gemm_route_version() {
    // hipBLAS's own version, asked of the LIBRARY rather than read out of a header macro: D-11 makes
    // this number pin provenance, and a macro from the header the binary was compiled against would
    // describe a build input, not the .so whose kernels actually produced these logits (the two can
    // disagree after a runtime-library swap, which is exactly the numerics change being guarded).
    // `handle` may be NULL, so this works on a `--gemm hand` run too — the version is recorded even on
    // the route that does not use it, because the pins are one corpus, not one corpus per route.
    // Format is hipBLAS's own: major * 10000 + minor * 100 + patch, printed as a.b.c.
    int version = 0;
    if (hipblasGetVersion(nullptr, &version) != HIPBLAS_STATUS_SUCCESS || version <= 0) {
        return "unknown";
    }
    return std::to_string(version / 10000) + "." + std::to_string(version / 100 % 100) + "."
           + std::to_string(version % 100);
}

WaveGemm resolve_wave_gemm(WaveGemm want, int rows) {
    // One row is the hand GEMV whatever the run asked for. This is not a preference: decode's
    // byte-identity to every shipped pin is the thing that makes route (c) landable, and a one-row
    // body is also the case the library loses (W4: 12–22× the dispatch floor at these N).
    if (rows <= 1) return WaveGemm::kHand;
    return want;
}

void gemm_bf16(const void* weight, const void* x, void* out, int m, int n, int k) {
    if (m < 1 || n < 1 || k < 1) {
        throw std::runtime_error("gemm_bf16: m=" + std::to_string(m) + " n=" + std::to_string(n) +
                                 " k=" + std::to_string(k) +
                                 "; every dimension of a matmul has to be at least 1");
    }
    if (!weight || !x || !out) {
        throw std::runtime_error("gemm_bf16: null operand (m=" + std::to_string(m) +
                                 " n=" + std::to_string(n) + " k=" + std::to_string(k) + ")");
    }
    if (route().path == WaveGemm::kLibraryLt) {
        LtRoute& lr = lt_route();
        // Lazy as well as at plan time, for the same reason the hipBLAS path is: a driver that runs a
        // body without calling `gemm_route_plan` still gets a working route, and a body of a width
        // nobody prewarmed (a prompt whose last wave is a remainder) builds and caches its own key. The
        // one line that says so is printed for it, in the same format, so an unplanned width is visible
        // in the log rather than inferred from a missing line.
        if (!lr.handle) build_lt_route();
        std::string built_line;
        LtPlan& p = lt_plan_for(m, n, k, built_line);
        // A plan built here was not built by `gemm_route_plan`, so nobody else is going to print it. Send
        // it to stdout in the same format, because an unplanned (shape, M) width must not be an invisible
        // event inside a timed window: it is the one library bookkeeping that CAN land mid-measurement.
        if (!built_line.empty()) std::printf("%s", built_line.c_str());
        const float alpha = 1.0f, beta = 0.0f;
        // Operand order is the sweep's, verified there against the engine's own working call
        // (`prototype/gemm_algo_sweep/README.md` §1a): BLAS-A is the WEIGHT (OP_T), BLAS-B is the
        // ACTIVATION, and C and D are the same buffer because beta is 0.
        ENGINE_LT(hipblasLtMatmul(lr.handle, p.desc, &alpha, weight, p.la, x, p.lb, &beta, out, p.lc,
                                  out, p.ld, &p.algo, lr.workspace, kWorkspaceBytes,
                                  (hipStream_t)nullptr));
        return;
    }
    GemmRoute& r = route();
    // Lazy as well as at startup: a driver that runs a body without calling `gemm_route_init` gets a
    // working handle rather than a null dereference, and the startup call is what makes a failure
    // visible before the first prefill rather than inside it.
    if (!r.handle) build_route();
    ENGINE_HB(gemm_common::gemm_stored_weight(r.handle, m, n, k, HIP_R_16BF, x, weight, out,
                                             HIP_R_16BF));
}

}  // namespace engine::kernels
