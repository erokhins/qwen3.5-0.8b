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

void proj_gemv(const void* weight, const uint16_t* x, uint16_t* out, int rows, int in) {
    constexpr int kRowsPerBlock = 8;                 // the prototype's default
    const int threads = kRowsPerBlock * warp_width();
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    // PACK_X = true: `x` stays BF16 in shared memory instead of being widened to F32. The
    // products and the accumulation order are bit-identical either way (every BF16 is exact in
    // F32), and it removes a cast per projection from the token.
    //
    // `in` is 1024 or 2048 for all nine shapes in this checkpoint (§01.2/§01.3); anything else is
    // a bug in the caller, not a shape to instantiate.
    if (in == 1024) {
        ::proj_gemv<1024, true><<<blocks, threads>>>((const hip_bfloat16*)weight, x,
                                                     (hip_bfloat16*)out, rows, kRowsPerBlock,
                                                     NEG_NONE);
    } else if (in == 2048) {
        ::proj_gemv<2048, true><<<blocks, threads>>>((const hip_bfloat16*)weight, x,
                                                     (hip_bfloat16*)out, rows, kRowsPerBlock,
                                                     NEG_NONE);
    } else {
        // No launch: the caller asked for a shape the gated kernel does not have.
        // decode_step asserts instead; a silent wrong result here would be invisible.
    }
}

}  // namespace engine::kernels
