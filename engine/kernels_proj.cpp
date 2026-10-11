// Launcher for prototype/proj_gemv: one templated BF16 GEMV for all nine projections (§04.1,
// §05.1). Its own TU — proj_gemv defines its own `enum Neg`, bf/fb/rnd and FULL_MASK.
#include "kernels.h"

#include <hip/hip_runtime.h>

#include "prototype/proj_gemv/proj_gemv_kernels.h"

namespace engine::kernels {
namespace {
// The kernel indexes rows as `blockIdx.x*rpb + threadIdx.x/warpSize`, so the block width must be
// rows-per-block × THIS device's warp width (32 on gfx1151). Querying it is what each prototype's
// host does before launching; hardcoding 32 here would silently halve the rows on a wave64 target.
int warp_width() {
    static int w = [] {
        int v = 32;
        (void)hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, 0);
        return v > 0 ? v : 32;
    }();
    return w;
}
}  // namespace

void proj_gemv(const void* weight, const float* x, uint16_t* out, int rows, int in) {
    constexpr int kRowsPerBlock = 8;                 // the prototype's default, and its gated width
    const int threads = kRowsPerBlock * warp_width();
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    // PACK_X = false — the configuration this op's 912-comparison gate ran (`run.py:402-406` adds
    // `--x-packed` only on the timing path, so `<IN,true>` was never compared against a pin), and
    // the configuration the measurement prefers: `results.md:196` times the packed form at
    // 3.269 ms / 60 % of probe against 2.991 ms / 66 % for this one over the 48-launch token plan,
    // and `:241` hands W5 "`PACK_X = false`, measured above" as the setting to take.
    //
    // The consequence decode_step must know: with PACK_X=false the kernel reads `x` as `const float*`
    // (`proj_gemv_kernels.h:107-110`), so the caller hands F32 activations, NOT the BF16 pointer
    // this launcher used to take. That is proj_gemv's own recommendation — "the engine should hand
    // this kernel the F32 norm output and skip the round-trip" (`results.md:196`) — but since glue
    // writes BF16 norms, ≈48 casts/token come back (≈0.089 ms at the measured 1.85 µs launch floor,
    // `tools/launch_overhead_hip.cpp`). Still the cheaper side: 0.089 ms of casts against a measured
    // +0.279 ms kernel penalty. Products and accumulation order are bit-identical in the two forms
    // (every BF16 is exact in F32), so this is a performance choice, not a numerics one — and the
    // cold-vs-cold timing on THIS launcher has never been run (results.md, experiment E2).
    //
    // `in` is 1024 or 2048 for all nine shapes in this checkpoint (§01.2/§01.3); anything else is
    // a bug in the caller, not a shape to instantiate.
    if (in == 1024) {
        ::proj_gemv<1024, false><<<blocks, threads>>>((const hip_bfloat16*)weight, x,
                                                     (hip_bfloat16*)out, rows, kRowsPerBlock,
                                                     NEG_NONE);
    } else if (in == 2048) {
        ::proj_gemv<2048, false><<<blocks, threads>>>((const hip_bfloat16*)weight, x,
                                                     (hip_bfloat16*)out, rows, kRowsPerBlock,
                                                     NEG_NONE);
    } else {
        // No launch: the caller asked for a shape the gated kernel does not have.
        // decode_step asserts instead; a silent wrong result here would be invisible.
    }
}

}  // namespace engine::kernels
