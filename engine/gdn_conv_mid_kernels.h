/*
 * gdn_conv_mid_kernels.h — derive the conv window as it stood after ROW 0 of a multi-row body, without
 * touching the shipped conv. Owned by the MTP accept round (`prototype/mtp_accept/ACCEPT-BUILD.md` §4
 * cond. 1, option (a) as authorised); the shipped path never calls it.
 *
 * WHY A DERIVATION AND NOT A PER-ROW CONV. The first V1 attempt ran `conv_kernel` once per row and
 * chained row 1's carried window through the checkpoint. It is red on device (`ACCEPT-BUILD.md` §4a):
 * whole rows of logits differ, which is a changed-work-split signature, not a rounding difference.
 * The authorised form is the opposite division of labour — the shipped batched conv runs EXACTLY as
 * it does today (so `u` and the live end window are bit-identical by construction, because they are the
 * same launch), and this kernel's only job is to materialise the ONE quantity the batched launch does
 * not leave anywhere addressable: the window after row 0. That is a data-movement claim, and
 * data-movement claims are what this file's sibling `gdn_conv_roll_kernels.h` exists to make honest.
 *
 * WHY IT IS BIT-IDENTICAL, stated as addresses rather than as reassurance. `conv_kernel`'s own
 * end-condition block writes `new_win[j] = conv_tap(raw, win, c0, tokens-(K-1)+j, ...)` for
 * j in 0..K-2, and `conv_tap`'s rule is index arithmetic with no dependence on the tile width:
 * `k < 0` reads `win[(k+K-1)*CH]` and `k >= 0` reads `raw[k*CH]`. Evaluate that block at `tokens = 1`
 * and the three writes are `win[1]`, `win[2]`, and the widening of `raw[0]` — which is precisely
 * "the window after this one row", because a window is the K-1 columns immediately preceding the row
 * about to be convolved. So this kernel calls the SAME `conv_tap`, from the SAME pinned header, with
 * the SAME arguments the shipped end condition would pass at `tokens = 1`, and writes them to a third
 * buffer instead of to the state window. There is no re-spelled arithmetic here to drift: one loop,
 * `conv_tap`, one store — which is the arrangement `gdn_rec_unroll4_kernels.h`'s rule asks for ("diff
 * this body against the source, do not improve it"), and here there is nothing to improve because the
 * source is a function call.
 *
 * WHY `raw` IS ROW 0 OF THE BODY'S TILE AND NOT A COPY. The launch is `conv_kernel`'s one-row geometry
 * (`grid.y = 1`), and `conv_tap` indexes a tile row as `raw[k*CH]` with k=0, so the caller hands the
 * tile base and the kernel reads row 0 of it — the same address the batched launch read for row 0,
 * which is what makes "after row 0" mean the row the accept round is about to keep.
 *
 * WHY THE LIVE WINDOW NEVER ENTERS. The caller runs this BEFORE the batched conv's scratch is copied
 * over the live window, and this kernel writes only to its own output. The live window is therefore
 * advanced by exactly one writer, the shipped one, on every round whether or not a checkpoint is being
 * taken — which is the constraint §4a recorded as a fact and this file enforces: for a 2-row body the
 * shipped path stores its end window into scratch and the engine copies that; nothing here reads it
 * back, and nothing here writes it.
 */
#pragma once

#include "prototype/gdn_decode/gdn_decode_kernels.h"  // conv_tap, conv_value, CH, K, CONV_VEC, hip_bfloat16

// The window after one row, into a caller-owned buffer. `raw` is the body's [n][CH] BF16 tile and this
// reads its row 0; `win` is the layer's carried [K-1][CH] F32 window as it stood BEFORE the batched
// conv of this body ran (that is what `tokens = 1` means here — this row has not been folded in yet);
// `mid_win` is [K-1][CH] F32 and is never read.
__global__ void conv_mid_window_kernel(const hip_bfloat16* __restrict__ raw,
                                       const float* __restrict__ win,
                                       float* __restrict__ mid_win) {
    // `conv_kernel`'s strip and lane-to-column map, restated from the same constants rather than taken
    // from it: a derivation that covered fewer channels than the roll it stands in for would look
    // cheaper while leaving half the checkpoint unwritten.
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * CONV_VEC;
    if (c0 >= CH) return;

    // The end condition at `tokens = 1`: `k = tokens - (K - 1) + j` is `1 - (K-1) + j`, and each write
    // goes through `conv_tap` exactly as `conv_kernel`'s does — same `win`, same `raw`, same
    // `zero_pad = false` (a carried row is the carried row, not zero padding: §08.2, and A-04's trap
    // about storing the POST-conv value into the window), same `channel_major = false`.
#pragma unroll
    for (int j = 0; j < K - 1; ++j) {
        float v[CONV_VEC];
        conv_tap(raw, win, c0, 1 - (K - 1) + j, 1, false, false, v);
        ConvColsF out{};
#pragma unroll
        for (int i = 0; i < CONV_VEC; ++i) out.f[i] = v[i];
        *reinterpret_cast<ConvColsF*>(mid_win + (size_t)j * CH + c0) = out;
    }
}
