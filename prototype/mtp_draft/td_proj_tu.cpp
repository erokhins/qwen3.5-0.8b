/*
 * prototype/mtp_draft/td_proj_tu.cpp — the projection family for the MTP block, compiled from
 * `prototype/proj_gemv/proj_gemv_kernels.h` (the same source `engine/kernels_proj.cpp` instantiates).
 * Four of the block's five GEMVs are this family and none of them needs a new kernel:
 *
 *   mtp.fc            [1024, 2048]   IN=2048   (the same (out,in) the target's o_proj already has)
 *   …self_attn.q_proj [4096, 1024]   IN=1024   (8 heads x (256 q | 256 gate), §05.1's interleave)
 *   …self_attn.k_proj [ 512, 1024]   IN=1024
 *   …self_attn.v_proj [ 512, 1024]   IN=1024
 *   …self_attn.o_proj [1024, 2048]   IN=2048
 *
 * The batched-token form is used with `tokens = 1`, which is what `engine/kernels_proj.cpp` issues for
 * every width ("TPB = 1 included"), so the drafter's row is the engine's row launch shape rather than a
 * third shape nobody timed. Geometry: `kRowsPerBlock = 32` at `threads = rpb * warp`, `TPB = 1`.
 *
 * PACK_X is the staging choice: true keeps the staged operand row BF16, false widens it to F32 at the use
 * site. The engine's shipped calls pass false and its `proj_gemv_xbf16` caller passes true; the header
 * says the products are identical because every BF16 value is exact in F32, and this rig does not take
 * that on faith — `--mode stagecheck` re-runs one GEMV both ways and requires 0 differing elements, which
 * is what licenses the whole draft chain staying BF16 end to end (and so carrying no cast launch).
 */
#include "td_launchers.h"

#include <stdexcept>

#include "prototype/proj_gemv/proj_gemv_kernels.h"

namespace td {

namespace {
template <int IN>
void launch(const uint16_t* W, const void* x, uint16_t* out, int rows, int rpb, bool pack_x) {
    const dim3 grid((rows + rpb - 1) / rpb, 1);
    const int threads = rpb * 32;
    if (pack_x)
        ::proj_gemv_tokens<IN, 1, true><<<grid, threads>>>(
            (const hip_bfloat16*)W, x, (hip_bfloat16*)out, rows, 1, rpb, NEG_NONE);
    else
        ::proj_gemv_tokens<IN, 1, false><<<grid, threads>>>(
            (const hip_bfloat16*)W, x, (hip_bfloat16*)out, rows, 1, rpb, NEG_NONE);
    td_launches += 1;
}
}  // namespace

void gemv(const uint16_t* W, const uint16_t* x, uint16_t* out, int rows, int in, bool pack_x) {
    if (in == 2048) launch<2048>(W, x, out, rows, td_rpb_proj, pack_x);
    else if (in == 1024) launch<1024>(W, x, out, rows, td_rpb_proj, pack_x);
    else throw std::runtime_error("mtp_draft gemv: input width " + std::to_string(in) +
                                  " is not one of the two widths the block has");
}

void gemv_f32(const uint16_t* W, const float* x, uint16_t* out, int rows, int in) {
    if (in == 2048) launch<2048>(W, x, out, rows, td_rpb_proj, false);
    else if (in == 1024) launch<1024>(W, x, out, rows, td_rpb_proj, false);
    else throw std::runtime_error("mtp_draft gemv_f32: input width " + std::to_string(in));
}

}  // namespace td
