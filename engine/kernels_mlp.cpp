// Launchers for prototype/mlp_gemv: the fused gate+up+SwiGLU dot and the down projection (§03.3).
// Its own TU — mlp_gemv defines `enum`-free NEG_* macros, HID, FF, FULL_MASK and its own bf/fb/rnd/
// reduce_row, all of which collide with proj_gemv's and lm_head_gemv's spellings if merged.
#include "kernels.h"

#include <hip/hip_runtime.h>

#include <stdexcept>

#include "prototype/mlp_gemv/mlp_gemv_kernels.h"

namespace engine::kernels {
namespace {
// Same warp-width rule as `proj_gemv`: the kernel maps rows as `blockIdx.x*rpb + threadIdx.x/ws`,
// so the block must be rows-per-block × THIS device's warp width. The host of the gated prototype
// reads it from `hipGetDeviceProperties` and refuses anything but 32 or 64; querying here is the
// engine doing the same thing rather than assuming gfx1151 forever.
int warp_width() {
    static int w = [] {
        int v = 32;
        (void)hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, 0);
        return v > 0 ? v : 32;
    }();
    return w;
}

// ROWS_PER_BLOCK — the default in `mlp_gemv.hip.cpp:97`, and the geometry the F1–F4 gate ran at
// (`run.py --layers all --fixture dec8 --no-time --tile 0 --negative`: `--tile 0` turns off the
// cache-tiling experiment path, leaving rows/block at its default). `prototype/mlp_gemv/results.md`
// sweeps 4→32 and finds warm rankings move by 4 % with a different winner, so the gated default is
// the honest constant until a cold sweep on the engine's own plan says otherwise.
constexpr int kRowsPerBlock = ROWS_PER_BLOCK;

void check(const char* what, int rows, int in) {
    if (rows <= 0 || in <= 0) {
        throw std::runtime_error(std::string("mlp: ") + what +
                                 " called with rows=" + std::to_string(rows) +
                                 " in=" + std::to_string(in));
    }
}
}  // namespace

void mlp_gateup(const uint16_t* w_gate, const uint16_t* w_up, const float* x, uint16_t* act,
                int rows, int in) {
    check("gateup", rows, in);
    const int threads = kRowsPerBlock * warp_width();
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    // `pre` (the raw gate/up dumps the gate's F1 stage compares) stays null: the engine never
    // needs the two intermediates, and writing them would put 2×3584 BF16 per layer on the token's
    // write traffic for a dump no gate here reads.
    if (in == HID) {
        ::mlp_gateup<HID><<<blocks, threads>>>((const hip_bfloat16*)w_gate,
                                               (const hip_bfloat16*)w_up, x,
                                               (hip_bfloat16*)act, nullptr, rows, kRowsPerBlock,
                                               NEG_NONE);
    } else if (in == FF) {
        // Instantiated by the gated host too (mlp_gemv.hip.cpp:249). Unused by this checkpoint's
        // MLP shapes — gate/up are [3584,1024] — but the kernel is one template line and a second
        // entry point would be a worse way to say "this shape exists".
        ::mlp_gateup<FF><<<blocks, threads>>>((const hip_bfloat16*)w_gate,
                                              (const hip_bfloat16*)w_up, x,
                                              (hip_bfloat16*)act, nullptr, rows, kRowsPerBlock,
                                              NEG_NONE);
    } else {
        throw std::runtime_error("mlp_gateup: input width " + std::to_string(in) +
                                 " is not one of the two shapes the gated kernel instantiates");
    }
}

void mlp_down(const uint16_t* w_down, const float* act, uint16_t* out, int rows, int in) {
    check("down", rows, in);
    const int threads = kRowsPerBlock * warp_width();
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    // The MLP's second dot is 3584 wide (FF), which is why `in` cannot be pinned to 1024/2048 like
    // proj_gemv's — §01.3: down_proj is [1024, 3584].
    if (in == FF) {
        ::mlp_down<FF><<<blocks, threads>>>((const hip_bfloat16*)w_down, act,
                                            (hip_bfloat16*)out, rows, kRowsPerBlock, NEG_NONE);
    } else if (in == HID) {
        ::mlp_down<HID><<<blocks, threads>>>((const hip_bfloat16*)w_down, act,
                                             (hip_bfloat16*)out, rows, kRowsPerBlock, NEG_NONE);
    } else {
        throw std::runtime_error("mlp_down: input width " + std::to_string(in) +
                                 " is not one of the two shapes the gated kernel instantiates");
    }
}

}  // namespace engine::kernels
