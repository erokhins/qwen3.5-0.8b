// The hipBLAS route of route (c): one handle, one pinned workspace, and the one call form the whole
// engine uses for a wave-width matmul (§6 item 1 of bench/prefill-route-spike-2026-10-13.md).
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

#include <sstream>
#include <stdexcept>
#include <string>

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

}  // namespace

std::string gemm_route_init() { return build_route(); }

void gemm_route_shutdown() {
    GemmRoute& r = route();
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
    if (name == "hand") return WaveGemm::kHand;
    throw std::runtime_error("--gemm takes library|hand, got '" + name + "'");
}

const char* wave_gemm_name(WaveGemm route) {
    return route == WaveGemm::kLibrary ? "library" : "hand";
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
    GemmRoute& r = route();
    // Lazy as well as at startup: a driver that runs a body without calling `gemm_route_init` gets a
    // working handle rather than a null dereference, and the startup call is what makes a failure
    // visible before the first prefill rather than inside it.
    if (!r.handle) build_route();
    ENGINE_HB(gemm_common::gemm_stored_weight(r.handle, m, n, k, HIP_R_16BF, x, weight, out,
                                             HIP_R_16BF));
}

}  // namespace engine::kernels
