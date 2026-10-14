/*
 * gdn_collapse_kernels.h — §04.4's recurrence, resident: ONE launch per GDN layer per wave body,
 * where the shipped sequential form (`prototype/gdn_decode/gdn_decode_kernels.h::rec_kernel`,
 * re-declared in `engine/kernels.h`) issues one launch per GDN layer per TOKEN.
 *
 * WHY THIS ROUTE EXISTS, in one line. `progress/2026-10-14/p9-phase1-2026-10-14.md` D-12 rejected the chunked scan
 * for launch-bound reasons and funded this one instead: the sequential work is 33.94 GFLOP per
 * 1024-lane wave against the chunked form's 54.03, because collapsing the recurrence does not have to
 * recompute anything — it only has to stop writing the state to memory and reading it back. The win
 * line was pre-registered there (`gdn_rec` ≤ 63 ms at 1024 lanes, honest band 40–80 ms), and
 * `progress/2026-10-14/p9-route-a-2026-10-14.md` carries the geometry arithmetic that was priced BEFORE this file
 * was written plus the device commands that decide whether the route lives.
 *
 * WHY IT IS BIT-IDENTICAL TO `rec_kernel` INSTEAD OF CLOSE TO IT. Two things could move a bit, and
 * both are pinned by construction rather than measured:
 *
 *   1 THE SERIAL DOT. §04.4's step-2/step-3 dots are ascending F32 sums that CANCEL (§04.4 measured 1
 *     element in 286 of `o` losing 84 % of its magnitude to that cancellation), so re-associating
 *     them is not a rounding choice, it is a different function. Here a thread still owns exactly one
 *     `d_v` column and walks `i` from 0 to 127 in the same two loops, with the same three expressions,
 *     in the same order, with the same F32 storage: the state is staged into LDS and read back as F32,
 *     which is an exact round trip. That is also why nothing here rounds the state — `rec_kernel` never
 *     rounds it either.
 *   2 THE NORM REDUCTION. k̃ = k·rsqrt(Σk² + eps) needs a block reduction per token, and a block
 *     reduction is a tree, so its shape is observable. Rather than re-derive a 32-lane equivalent of
 *     `rec_kernel`'s 128-thread tree — which would hang the identity on how this device fills an
 *     out-of-range `__shfl_down` source lane, an undocumented thing to bet the gate on — this kernel
 *     keeps `blockDim.x == 128` and calls the shipped `block_sum` verbatim, fed the same
 *     element-per-thread mapping (thread i squares k[i]). Same source, same width, same tree. The
 *     cost of that decision is paid where it is paid: three of the block's four wavefronts park at the
 *     norm barrier and the recurrence runs on warp 0 alone. See `kBlockDim`.
 *
 * WHAT ACTUALLY CHANGED, which is the whole point: the state strip lives in LDS for the duration of a
 * body, so `S` is read once and written once per (layer, body) instead of once per token, and the
 * launch count goes from 18 × tokens to 18. `o` is still one F32 value per (token, head, column).
 *
 * GEOMETRY, and why it is this shape. One block owns one head's 32 `d_v` columns:
 *      grid  = (d_v / kColumns, heads) = (4, 16) = 64 blocks        blockDim = 128
 *      LDS   = 128 rows × 32 columns of F32 state (16,384 B) + k̃/q̃ staging (1,024 B) + the tree's
 *              4 partials (16 B) = 17,424 B, which is 3 blocks per CU against the 65,536 B/CU the
 *              device reports (`devices/strix-halo-compute.md`): the ≥2-blocks/CU target is met with
 *              one block of headroom, and `prototype/gdn_collapse/collapse_price.py` asserts it from
 *              the compiler's number rather than from this sentence.
 *     A 64-column strip was the obvious alternative and it is impossible: 128 rows × 64 columns of F32
 *     is 32,768 B, which is the whole LDS budget for a CU before the k̃/q̃ staging is added, so it
 *     would buy exactly 1 block/CU and pay for it by reading q and k twice as often. `slot()` swizzles
 *     rather than pads: bank (c ^ (i & 31)) keeps a fixed-`i` row — the only access pattern with more
 *     than one lane asking at once — on 32 distinct banks without spending a byte of LDS on padding,
 *     which is what keeps three blocks per CU instead of two.
 *
 * WHAT THIS DOES NOT FIX, said plainly because it is the number a reader will want: the sequential
 * recurrence has 16 heads × 128 columns = 2,048 independent chains per (layer, body), which is
 * 2,048/5,120 = 40 % of this device's FP32 lanes. That is the whole serial width of §04.4's scan, not
 * a staging choice — no re-blocking makes more of it, and a wave's bodies are submitted one after the
 * other so they cannot share a launch. What this route removes is the per-token launch and the
 * per-token state round trip, which is where 18,432 launches and most of the `gdn_rec` family's
 * executed work actually went.
 *
 * FAIL-CLOSED NEGATIVES (all four wired into `prototype/gdn_decode/gdn_decode.hip.cpp`'s `--neg`,
 * each required to break bit-identity, i.e. each proves the guard it names):
 *     rc_state_bf16   the staged state is rounded to BF16 and back while staging — the cheap "the
 *                     state is big, halve it" mistake, and the one this route is permanently exposed to.
 *     rc_strip_shift  the strip's column base is taken from the block's y index instead of its x, so
 *                     two blocks own the same columns and four own none: the mapping bug this layout
 *                     is most exposed to, and nothing about the output's shape reveals it.
 *     rc_norm_half    the norm sums half of the squared values: a reduction of the right shape and the
 *                     wrong width, which is what a lane-count change silently produces.
 *     rc_i_desc       the step-2 dot walks `i` downwards — same terms, different association, the
 *                     §04.4 cancellation case, so this one is expected to break a small number of
 *                     columns with decision room rather than everything.
 */
#pragma once
#include <hip/hip_runtime.h>

#include "../prototype/gdn_decode/gdn_decode_kernels.h"   // block_sum, D, NH, Q_SCALE (and the 1e-6f the shipped kernel spells literally)

namespace gdn_rc {

inline constexpr int kD = D;                  // 128: d_k = d_v (§01.6)
inline constexpr int kNH = NH;                // 16 value heads
inline constexpr int kQkvRows = 6144;         // one token's q|k|v row of `u` (§07.2 slices it)
inline constexpr int kValueDim = 2048;        // one token's row of `o`
inline constexpr int kColumns = 32;           // d_v columns per block = one wavefront of lanes
inline constexpr int kBlockDim = 128;         // threads per block: 4 warps, deliberately (see above)
inline constexpr int kStrips = D / kColumns;  // 4 strips per head

// LDS slot for state row i, strip column c. Swizzled, not padded: see the geometry note.
__device__ __forceinline__ int slot(int i, int c) { return i * kColumns + (c ^ (i & (kColumns - 1))); }

// One GDN layer, one body. Touches exactly the addresses `rec_kernel` would have been handed, one row
// at a time; holds `S[h][:][strip]` in LDS for the body.
__global__ void __launch_bounds__(kBlockDim)
rec_collapse_kernel(const float* __restrict__ u,
                    const float* __restrict__ g,
                    const float* __restrict__ beta,
                    float* __restrict__ S,
                    float* __restrict__ o,
                    int tokens) {
    const int strip = blockIdx.x;          // which 32-column slice of d_v this block owns
    const int h = blockIdx.y;              // which of the 16 value heads
    const int tid = threadIdx.x;

    // The resident state strip, the two normalized rows, and block_sum's 4 partials. A
    // `red[tokens][32]` like rec_kernel's would cost 64 KiB of LDS this kernel has no room for, which
    // is one way of saying why the norm is the only thing here that still reduces per token.
    __shared__ float Sst[D * kColumns];
    __shared__ float ks[D];
    __shared__ float qs[D];
    __shared__ float red[kBlockDim / 32];

    // Column ownership: warp 0 (threads 0..31) drives the recurrence with one thread per d_v column,
    // which is what preserves the serial `i` order per column. Threads 32..127 exist for the norm tree
    // and never touch Sst, whose columns all belong to warp 0.
    float* Sh = S + (size_t)h * D * D;
#ifdef NEG_RC_STRIP_SHIFT
    const int cbase = (h % kStrips) * kColumns;
    (void)strip;      // the defect is that the block's own strip index is ignored; saying so keeps -Wall
#else
    const int cbase = strip * kColumns;
#endif

    // ---- stage the state strip: 128 rows x 32 columns. `S` is [d_k][d_v] row-major, so a row is 32
    // contiguous floats = one 128 B transaction, and it lands on its swizzled slot.
    //
    // The cells are striped over the threads, the threads are NOT the columns. This block has 128
    // threads and the strip has 32 columns, so `tid` is not a column index here: a loop that walked
    // `i` and used `tid` as the column asked `Sh` for 128 columns of a 32-column strip, which read (and
    // in the write-back, wrote) up to 96 floats past every head's state — 96 floats past the END of the
    // tensor for the last head, i.e. an illegal access — and it wrote up to 65 cells past the end of
    // `Sst`, which on this device does not fault, it just silently corrupts whatever the allocator put
    // after the state array (`ks`, `qs`, `block_sum`'s `red`). Both halves of that are why the strip is
    // covered by a strided cell loop instead: every one of the 4096 cells is staged exactly once, by
    // exactly one thread, and within a warp the 32 lanes share `i` and own `c = 0..31`, so the global
    // read is one contiguous 128 B row and the swizzled LDS write hits 32 distinct banks.
    for (int idx = tid; idx < D * kColumns; idx += kBlockDim) {
        const int i = idx / kColumns;
        const int c = idx - i * kColumns;      // exact: idx, i are ints and kColumns is 32
#ifdef NEG_RC_STATE_BF16
        // The same rounding helper the shipped kernels use, so the defect is "the state got rounded",
        // not "a different rounding than the engine's".
        Sst[slot(i, c)] = bf16_round(Sh[(size_t)i * D + cbase + c]);
#else
        Sst[slot(i, c)] = Sh[(size_t)i * D + cbase + c];
#endif
    }
    __syncthreads();

    for (int t = 0; t < tokens; ++t) {
        const float* row = u + (size_t)t * kQkvRows;
        const float kv = row[kValueDim + h * D + tid];
        const float qv = row[h * D + tid];

        // ---- k̃ and q̃: the shipped kernel's expressions at the shipped reduction width, AND its
        // association. `kn`/`qn` below are `rec_kernel`'s `float kn = kq * rsqrtf(..)` and
        // `float qn = qq * rsqrtf(..) * Q_SCALE` character for character, and that matters more than it
        // looks: hoisting the `Q_SCALE` factor out of the per-thread multiply — writing
        // `qs[tid] = qv * (rsqrtf(..) * Q_SCALE)` — is a different product tree, and on the device it
        // moved 93,522 of 131,072 `o` elements while leaving the STATE bit-exact, because only q̃
        // carries the extra factor. `block_sum` is bit-identical per (token, head) either way: every
        // thread still reduces its own element, and the four strips of a head each redo the same tree —
        // four times the reduction, zero extra memory traffic, which is the trade §04.7's carry made in
        // the opposite direction.
#ifdef NEG_RC_NORM_HALF
        const float ksq = tid < (D / 2) ? kv * kv : 0.0f;
        const float qsq = tid < (D / 2) ? qv * qv : 0.0f;
#else
        const float ksq = kv * kv;
        const float qsq = qv * qv;
#endif
        const float kn = kv * rsqrtf(block_sum(ksq, red) + 1e-6f);
        const float qn = qv * rsqrtf(block_sum(qsq, red) + 1e-6f) * Q_SCALE;
        ks[tid] = kn;
        qs[tid] = qn;
        __syncthreads();

        if (tid < kColumns) {
            // Warp 0's thread `tid` owns the strip's column `cbase + tid`: its `v` element and its `o`
            // element carry the same +cbase the state does, which is the one place the strip split
            // touches an address that is not the state's.
            const float gam = expf(g[(size_t)t * kNH + h]);
            const float bet = beta[(size_t)t * kNH + h];

            // step 2: Pt = (k̃ ⊙ S_prev)ᵀ — ascending i, F32, one accumulator per column.
            //
            // No `#pragma unroll` here, and that is a measured choice: unrolling both `i` loops 8-wide
            // is bit-identical (one accumulator, `i` ascending — an unroll is not a re-association) and
            // it changed nothing, `gdn_rec` 112.8 → 113.4 ms at 1024 lanes (`prototype/gdn_collapse/
            // results.md` §4). The loop is therefore not LDS-latency-bound at this occupancy, and a
            // pragma that buys nothing is a claim about the kernel that no longer holds.
            float r = 0.f;
#ifdef NEG_RC_I_DESC
            for (int i = D - 1; i >= 0; --i) r += Sst[slot(i, tid)] * ks[i];
#else
            for (int i = 0; i < D; ++i) r += Sst[slot(i, tid)] * ks[i];
#endif

            // step 3's delta
            const float dv = bet * (row[2 * kValueDim + h * D + cbase + tid] - gam * r);

            // step 3: S = gam·S + k̃ᵀdV then o = q̃ ⊙ (S_new)ᵀ, one ascending walk, and the cell is
            // written exactly once per token so the two forms' states stay comparable cell by cell.
            float acc = 0.f;
            for (int i = 0; i < D; ++i) {
                const int sl = slot(i, tid);
                float s = gam * Sst[sl];
                s += ks[i] * dv;
                Sst[sl] = s;
                acc += qs[i] * s;
            }
            o[(size_t)t * kValueDim + h * D + cbase + tid] = acc;
        }
        // No barrier here, and that is a claim worth stating rather than leaving to the reader: the
        // state columns are private to the lane that owns them, and `ks`/`qs` are fenced by the
        // `__syncthreads()` inside `block_sum` at the top of the NEXT iteration — which the idle warps
        // reach without waiting on warp 0's loop, because nothing between this line and that barrier
        // touches shared memory. Adding a barrier per token would cost one in ~1,200 instructions and
        // buy an invariant the loop does not need.
    }

    // ---- ONE write-back per (layer, body), in the layout the strip came in. This is the 512× cut in
    // state traffic that the route exists for, and it is unrounded: an F32 round trip is exact.
    __syncthreads();   // REQUIRED, and its position is the whole difference between a race and a
                       // barrier: warps 1..3 left the token loop as soon as the last `block_sum`
                       // returned, so without this they would read state cells warp 0 is still writing
                       // for the final token. One barrier per body instead of one per token is the
                       // saving; one per token would be the bug the previous line used to guard.
    // Same strided cell loop as the staging, for the same reason and in the same order — a row at a
    // time, 32 contiguous floats per warp, so the write-back is the staging with the arrows reversed.
    for (int idx = tid; idx < D * kColumns; idx += kBlockDim) {
        const int i = idx / kColumns;
        const int c = idx - i * kColumns;
        Sh[(size_t)i * D + cbase + c] = Sst[slot(i, c)];
    }
}

}  // namespace gdn_rc
