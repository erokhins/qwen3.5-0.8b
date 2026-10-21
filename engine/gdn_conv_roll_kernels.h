/*
 * gdn_conv_roll_kernels.h — §04.2's conv with its own window roll folded in: route S1's `copy` family,
 * 18 of its 19 submissions per decode token, deleted at the kernel that is both the roll's writer and
 * the window's only reader.
 *
 * WHAT THE SHIPPED PATH DOES. `prototype/gdn_decode/gdn_decode_kernels.h::conv_kernel` is handed
 * `win` (the layer's [K-1][CH] F32 state window, read), a scratch row `new_win` (written), and the
 * engine then D2D-copies that scratch over the state window — `engine/decode_step.cpp`'s roll, one
 * `hipMemcpyAsync` per GDN layer per body, 18 per decode token, 73,728 B each way. The scratch and
 * the copy exist for ONE reason, and the shipping comment names it: `conv_kernel`'s `win` and
 * `new_win` are both `__restrict__`, and writing the new window over the buffer it reads would break
 * that promise (`decode_step.cpp`: "`new_win` cannot be the window itself — conv_kernel's `win`/
 * `new_win` are both `__restrict__`").
 *
 * WHY IT CAN FOLD ANYWAY. `conv_kernel`'s own end-condition comment already states the real safety
 * property, and it is about ADDRESSES and not about restrict: "every thread owns the same columns on
 * both sides of the write, so this is safe against its own reads whatever the relationship between
 * `win` and `new_win` is". At tokens = 1 the roll is literally
 *     win row 0 <- old win row 1,   win row 1 <- old win row 2,   win row 2 <- widen(raw row 0)
 * and a lane owns CONV_VEC consecutive channels of EVERY row it touches: the three tap rows it reads
 * and the three new rows it writes are the same `c0 .. c0+CONV_VEC-1`. No lane can read a column
 * another lane is writing, and no lane reads after it writes. What blocks is only the qualifier, so
 * this file's job is to be the entry point that does not carry the lie.
 *
 * WHY A SEPARATE KERNEL AND NOT THE SHIPPED ONE WITH ALIASED POINTERS. Passing one buffer through two
 * `__restrict__` parameters is not a warning to be ignored: it is the case the qualifier exists to
 * forbid, and the compiler is entitled to sink `conv_kernel`'s window loads below its window stores
 * because it was promised they cannot alias. `prototype/gdn_decode` is a pinned reference and its
 * qualifiers are part of what its gate tests, so it stays untouched and this file holds the
 * aliased-legal spelling instead — the same arrangement as `engine/gdn_rec_unroll4_kernels.h` and
 * `engine/gdn_collapse_kernels.h`, which are engine-owned respellings of prototype kernels for
 * exactly this reason.
 *
 * BIT-IDENTITY IS THE ADMISSION, NOT A HOPE, and here it is a data-movement claim rather than an
 * arithmetic one: nothing is recomputed differently. The three new window rows are the same three
 * values the copy would have copied — two of them are F32 values the copy moved byte for byte, and the
 * third is `conv_tap`'s own widening of the raw row, which the shipped path also produced in the same
 * kernel and then copied as F32. So the gate owes a byte-compare of the window and of `u`, not a band.
 * The one thing that IS re-spelled is the tap/dot body, and it is transcribed expression for
 * expression from `conv_tap` and `conv_value` (the `gdn_rec_unroll4` rule: "diff this body against
 * the source, do not improve it"), because a hoisted or re-associated constant is a different number
 * — `engine/gdn_collapse_kernels.h`'s q̃ note, where a re-associated product moved 93,522 of 131,072
 * outputs on device while a CPU mirror saw nothing. `prototype/glue_copy_fold/equiv_copy_fold.py`
 * diffs the two spellings' arithmetic textually as well as running a bit-exact CPU mirror, so a future
 * edit to either side is caught by both.
 *
 * ONE-ROW ONLY, AND THAT IS A GEOMETRY FACT, NOT A POLICY. At tokens > 1 `blockIdx.y` is the token and
 * every token-block claims the SAME channel strip, so the roll is issued by `last_token_only` and the
 * other rows' taps still read a window this launch has not finished writing. A wave therefore keeps
 * the shipped pair, which is also what `decode_step.cpp`'s folded guard says.
 */
#pragma once

#include "prototype/gdn_decode/gdn_decode_kernels.h"  // conv_kernel, conv_tap, conv_value, CH, K, CONV_VEC, ConvCols, ConvColsF, bf16_round

// §04.2 conv over one token, writing its output AND rolling the state window in the same launch.
// `win` is the state window itself, both operands; `raw` is the [1][CH] BF16 projection row and `u` the
// [1][CH] F32 output, exactly as `conv_kernel` takes them at tokens = 1.
__global__ void conv_roll_kernel(const hip_bfloat16* __restrict__ raw, float* win,
                                 const float* __restrict__ C, float* __restrict__ u) {
    // Same strip, same lane->column map as `conv_kernel`: 6 blocks of 256 lanes, CONV_VEC columns
    // each, so the launcher's geometry is shared and a folded launch cannot be cheaper by touching
    // fewer addresses.
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * CONV_VEC;
    if (c0 >= CH) return;

    // Every read this lane owes, taken BEFORE a single byte of the window is written, into registers
    // that outlive the writes. That is what makes the aliased window safe on its own merits rather
    // than by the compiler's good behaviour: after this block the kernel touches `win` only to store.
    //
    // Taps, oldest first — `conv_value`'s `k = t = 0`, so `k - (K-1) + j` is -3, -2, -1, 0 and the
    // first three are the carried window rows 0, 1, 2 (`conv_tap`: `win[k + K - 1]`), the last is
    // this token's own raw row. The roll needs exactly rows 1, 2 and that same raw row, so one set of
    // loads serves both halves; the values land in `tap` in the order the dot consumes them in.
    float tap[K][CONV_VEC];
#pragma unroll
    for (int j = 0; j < K - 1; ++j) {
        const ConvColsF w = *reinterpret_cast<const ConvColsF*>(win + (size_t)j * CH + c0);
#pragma unroll
        for (int i = 0; i < CONV_VEC; ++i) tap[j][i] = w.f[i];
    }
    {
        const ConvCols r = *reinterpret_cast<const ConvCols*>(raw + c0);
#pragma unroll
        for (int i = 0; i < CONV_VEC; ++i) tap[K - 1][i] = float(r.h[i]);  // the same widening as before
    }

    // The values: `conv_value`'s dot, transcription for transcription — K taps in `j` order, one BF16
    // round, then SiLU, then the second round (§04.2). Compare `conv_value`: the only textual
    // difference is that its taps arrived through `conv_tap` instead of these four loads, and at
    // tokens = 1 they are the same four addresses in the same order.
    ConvColsF out{};
#pragma unroll
    for (int i = 0; i < CONV_VEC; ++i) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < K; ++j) acc += tap[j][i] * C[(c0 + i) * K + j];
        const float rounded = bf16_round(acc);
        out.f[i] = bf16_round(rounded / (1.f + expf(-rounded)));
    }
    *reinterpret_cast<ConvColsF*>(u + c0) = out;

    // The end condition, from the registers above: the window after a one-token tile is its last K-1
    // raw columns, which at tokens = 1 is old rows 1 and 2 shifted down plus the row this token just
    // contributed. `conv_kernel` writes the same three rows through `new_win` and the engine then
    // copies them here; here they land in place, and every value is one this lane loaded itself.
#pragma unroll
    for (int j = 0; j < K - 1; ++j) {
        ConvColsF rolled{};
#pragma unroll
        for (int i = 0; i < CONV_VEC; ++i) rolled.f[i] = tap[j + 1][i];
        *reinterpret_cast<ConvColsF*>(win + (size_t)j * CH + c0) = rolled;
    }
}
