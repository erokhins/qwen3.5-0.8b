/*
 * engine/proj_group_kernels.h — the DECODE one-row projection launch that can cover SEVERAL weight
 * tensors at once, on the word loop and block geometry `prototype/proj_gemv/results.md` §D1-W4 won.
 *
 * WHY A NEW KERNEL WHEN `proj_gemv_tokens` IS ALREADY GATED. The prototype's winning arm differs from
 * the shipped form in two places, and only one of them is a shape:
 *
 *   1. the weight-word loop carries a COMPILE-TIME trip count with `#pragma unroll 4` (4 × 16 B of
 *      weight in flight), where the shipped `proj_gemv_tokens` lost its pragma when the injected-bug
 *      vocabulary made the word bound a runtime value (`proj_gemv_kernels.h`, the note above its word
 *      loop). Same words, same order, same tree — bit-identical, which is why the prototype's gate ran
 *      it as `exact=true`.
 *   2. the block walks its rows in TRIPS (`step = blockDim.x / lanes_per_row`), which is what lets
 *      rows/block and threads/block be two axes instead of one. The winning point of the swept plane
 *      is rows/block 8 at 1024 threads (§D1-W4: 2.4693 ms against the shipped shape's 3.2692 ms in the
 *      same window, same plan, 0 VOID / 0 LOWCLK, two interleaved passes agreeing to 0.1 %).
 *
 * and the plan lever needs the same kernel: folding four input projections into one launch (§D1-W4's
 * 48-launch plan, −11.7 % at identical geometry) means ONE launch has to read FOUR weight tensors and
 * write FOUR destinations. The weights are one `hipMalloc` per tensor (`engine/loader.h`), so they are
 * NOT adjacent, and the only way to cover them in one launch without copying 467 MB of weights into a
 * packed copy is to hand the kernel a segment list. That is this file: a row-index space that is the
 * concatenation of the segments, and a per-row lookup of which segment owns the row.
 *
 * WHAT IS AND IS NOT OWED HERE. The admission contract is bit-identity with the shipped kernel, so the
 * per-row arithmetic is transcribed from `proj_gemv_kernels.h` character for character: stage `x`
 * (one row, F32, PACK_X=false), fetch `uint4` weight words at `v = lane, lane+32, lane+64, …`,
 * accumulate eight `bf(p[j]) * xs[v*8+j]` products per word in `j` order, reduce with the
 * `warpSize`-wide `__shfl_down_sync` ladder, round once with `hip_bfloat16(f)`. Segment lookup, trip
 * loop and store masking touch none of that. The engine's admission is therefore a device byte-compare
 * of logits and sampled tokens against the same run with the flags off — not a band, and no band is
 * widened (`HANDOFF-D1-ENGINE.md`).
 *
 * ONE SHAPE ONLY: a body of ONE row, `x` = one row of F32 activations, `out` = one row of a
 * row-family tile. A wider body takes `proj_gemv_tokens` at `TPB > 1` (the prefill wave's token tiling
 * is a different lever and §D1-W4 did not touch it), which is why `decode_step.cpp` gates this route on
 * `n == 1` and why `ProjSeg` carries no output stride: with one row, `out[lrow]` IS the destination.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "engine/kernels.h"   // engine::kernels::ProjSeg / ProjSegs / kProjMaxSegs

namespace engine::projgroup {

// HIP takes the shuffle mask as 64 bits on this port (`prototype/README.md`), and this TU cannot
// borrow `FULL_MASK` from an op header: the ops define colliding macros and one TU per family is the
// rule (`engine/kernels.h`'s header note). Namespace scope + constexpr = internal linkage, so it is
// private to whichever TU includes this file.
constexpr unsigned long long kFullShuffleMask = 0xffffffffffffffffull;

/*
 * One launch over the concatenation of `segs.n` weight tensors, all of width `IN`, all reading the
 * same `x` row.
 *
 * THE ROW SPACE. Row `r` of the launch is row `r - seg.start` of segment `seg`, and segments are
 * ordered and contiguous by construction (`kernels_proj.cpp` fills `start` as a running sum), so the
 * lookup is a walk over at most `kProjMaxSegs` comparisons — once per row, not once per element, and
 * uniform across a wavefront because every lane of a wavefront works the same row.
 *
 * THE TRIP LOOP, and why it is a loop. `rpb` (rows/block) and `blockDim.x` are two axes here, which is
 * the split §D1-W4 needed to price them separately: `step = blockDim.x / warpSize` is how many rows one
 * block walks at a time, and `rpb` is how many it owns. At the winning geometry (rpb 8, threads 1024,
 * warp 32) `step` is 32, so one trip runs and the 24 wavefronts past row 8 of the block fall straight
 * out of the loop — that is the shape that was measured, idle wavefronts and all, and reproducing it is
 * the difference between porting a win and inventing one. Rows past the concatenation are CLAMPED and
 * their stores masked (the prototype's form), so a `total` that is not a multiple of `rpb` stays
 * in-bounds without an early `return` in a region where lanes could still be inside a shuffle.
 *
 * `if (ws == 32)` folds at compile time for this target and is what keeps the trip count compile-time:
 * clang refuses `#pragma unroll` on a stride bounded by a runtime `warpSize` (the prototype hit exactly
 * that — "loop not unrolled" — and it is why variant 1 had to be re-spelled over `it`). The else arm is
 * the shipped kernel's runtime walk, word for word, so a wave64 target still gets correct (un-unrolled)
 * arithmetic instead of a silently skipped tail; `decode_step.cpp` refuses this shape off a wave32
 * device anyway, and says so.
 */
template <int IN>
__global__ void proj_group_rows(const engine::kernels::ProjSegs segs, const float* __restrict__ x,
                                int rpb) {
    __shared__ float xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int step = (int)blockDim.x / ws;
    const int total = segs.total;

    for (int r = threadIdx.x / ws; r < rpb; r += step) {
        int row = blockIdx.x * rpb + r;
        const bool live = row < total;
        if (!live) row = total - 1;              // clamped: valid loads, masked store

        int si = 0;
        while (si + 1 < segs.n && row >= segs.s[si].start + segs.s[si].rows) ++si;
        const engine::kernels::ProjSeg& sg = segs.s[si];
        const int lrow = row - sg.start;

        const hip_bfloat16* __restrict__ W = (const hip_bfloat16*)sg.weight;
        const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)lrow * IN);

        float acc = 0.f;
        auto dot8 = [&](int v) {
            const uint4 u = wrow[v];             // ONE 16-B weight word per iteration
            const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += float(p[j]) * xs[v * 8 + j];
        };
        constexpr int WORDS = IN / 8;
        if (ws == 32) {
            constexpr int ITER = WORDS / 32;     // 4 at IN=1024, 8 at IN=2048
#pragma unroll 4
            for (int it = 0; it < ITER; ++it) dot8(lane + it * 32);
        } else {
            for (int v = lane; v < WORDS; v += ws) dot8(v);
        }

        float s = acc;
        for (int o = ws / 2; o; o >>= 1) s += __shfl_down_sync(kFullShuffleMask, s, o);
        if (lane == 0 && live) {
            ((hip_bfloat16*)sg.out)[lrow] = hip_bfloat16(s);
        }
    }
}

}  // namespace engine::projgroup
