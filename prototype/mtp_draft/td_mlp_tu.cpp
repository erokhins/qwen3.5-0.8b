/*
 * prototype/mtp_draft/td_mlp_tu.cpp — the §03.3 MLP for the MTP block, compiled from
 * `prototype/mlp_gemv/mlp_gemv_kernels.h` and `engine/mlp_xbf16_kernels.h`.
 *
 * The MTP MLP is the TARGET MLP's shapes verbatim — gate_proj/up_proj [3584,1024], down_proj
 * [1024,3584] — so the gated pair applies with no new code, including the three-rounding SwiGLU
 * epilogue (`silu(gate)` rounded, then the product with the rounded `up`, then rounded), which
 * `mlp_gemv`'s F2 gate is what discriminates. Geometry is the engine's: `ROWS_PER_BLOCK = 8`,
 * `threads = rpb * warp`, one launch for gate+up+act (the 3584-wide intermediate never reaches memory)
 * and one for down.
 *
 * Both staging forms are instantiated so `--mode stagecheck` can require them to be bit-identical on
 * this block's operands rather than trusting `engine/mlp_xbf16_kernels.h`'s argument (which is sound in
 * general and cheap to re-prove here): `hip_bfloat16 -> float` is exact, so the staged row is the same
 * F32 values either way and the only difference is where the widening happens.
 */
#include "td_launchers.h"

#include <stdexcept>

#include "engine/mlp_xbf16_kernels.h"
#include "prototype/mlp_gemv/mlp_gemv_kernels.h"

namespace td {

namespace {
constexpr int kRpb = ROWS_PER_BLOCK;   // 8, the engine's kRowsPerBlock

template <int IN>
void launch_gateup(const uint16_t* wg, const uint16_t* wu, const void* x, uint16_t* act, int rows,
                   bool pack_x) {
    const int threads = kRpb * 32;
    const int blocks = (rows + kRpb - 1) / kRpb;
    if (pack_x)
        ::engine::mlpxbf16::mlp_gateup_xbf16<IN, MAP_INTERLEAVE><<<blocks, threads>>>(
            (const hip_bfloat16*)wg, (const hip_bfloat16*)wu, (const hip_bfloat16*)x,
            (hip_bfloat16*)act, nullptr, rows, kRpb, NEG_NONE);
    else
        ::mlp_gateup<IN, MAP_INTERLEAVE><<<blocks, threads>>>(
            (const hip_bfloat16*)wg, (const hip_bfloat16*)wu, (const float*)x,
            (hip_bfloat16*)act, nullptr, rows, kRpb, NEG_NONE);
    td_launches += 1;
}

template <int IN>
void launch_down(const uint16_t* wd, const void* act, uint16_t* out, int rows, bool pack_x) {
    const int threads = kRpb * 32;
    const int blocks = (rows + kRpb - 1) / kRpb;
    if (pack_x)
        ::engine::mlpxbf16::mlp_down_xbf16<IN><<<blocks, threads>>>(
            (const hip_bfloat16*)wd, (const hip_bfloat16*)act, (hip_bfloat16*)out, rows, kRpb,
            NEG_NONE);
    else
        ::mlp_down<IN><<<blocks, threads>>>(
            (const hip_bfloat16*)wd, (const float*)act, (hip_bfloat16*)out, rows, kRpb, NEG_NONE);
    td_launches += 1;
}
}  // namespace

void gateup(const uint16_t* wg, const uint16_t* wu, const uint16_t* x, uint16_t* act, int in) {
    if (in != 1024) throw std::runtime_error("mtp_draft gateup: input width " + std::to_string(in));
    launch_gateup<HID>(wg, wu, x, act, FF, true);
}

void gateup_f32(const uint16_t* wg, const uint16_t* wu, const float* x, uint16_t* act, int in) {
    if (in != 1024) throw std::runtime_error("mtp_draft gateup_f32: input width " + std::to_string(in));
    launch_gateup<HID>(wg, wu, x, act, FF, false);
}

void down(const uint16_t* wd, const uint16_t* act, uint16_t* out, int in) {
    if (in != 3584) throw std::runtime_error("mtp_draft down: input width " + std::to_string(in) +
                                             " (down_proj is [1024, 3584], §01.3)");
    launch_down<FF>(wd, act, out, HID, true);
}

void down_f32(const uint16_t* wd, const float* act, uint16_t* out, int in) {
    if (in != 3584) throw std::runtime_error("mtp_draft down_f32: input width " + std::to_string(in));
    launch_down<FF>(wd, act, out, HID, false);
}

}  // namespace td
