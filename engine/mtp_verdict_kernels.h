#pragma once
// §09 / M2 — the accept round's VERDICT: both rows' greedy argmax and the accept test, resolved ON
// DEVICE, so a round reads back four integers instead of a vocabulary.
//
// WHY THIS EXISTS. An accept round has to know, before it decides anything, whether the verifier's own
// prediction at row 0 equals the token it was asked to believe. The verifier's answer is a
// [kRows][kVocab] F32 tile, and the only shipped way to read that tile is `finish_token`'s: download
// the row (993,280 B each) and scan it on the host. Two rows, twice per round-trip, and the round is
// then on a per-step host sync for 2 MB of traffic to transport two comparisons. The plan's `c`
// column and D-b's whole argument are that this comparison must not cost a vocabulary-sized download,
// so it stops being a download: one block per row reduces the row to an id, row 0's id is compared
// against the candidate in the same launch, and the round reads back `kVerdictWords` ints (16 B).
//
// THE TIE RULE IS THE CONTRACT. `decode_step.cpp::argmax_greedy` is §03.4's only spelling and it is
// deliberately STRICT (`if (logits[i] > logits[best])`), so the FIRST, i.e. lowest, id wins a tie;
// `engine/kernels_lmhead.cpp` keeps refusing the fused `lm_head_gemv` argmax for exactly this reason,
// because `atomicMax` on a packed (float-key, index) word resolves ties to the HIGHEST id. This kernel
// therefore reduces on the pair (value, index) with the comparator
//
//     better(a, b)  <=>  a.value > b.value  ||  (a.value == b.value && a.index < b.index)
//
// which is `argmax_greedy`'s scan restated as an associative fold: the maximum value, and among the
// indices attaining it the least. Determinism is not luck here — the block reduction is a fixed
// shared-memory tree with no atomics and no dependence on arrival order, so the same tile gives the
// same id every run, which is what makes it legal for a pinned gate to be green twice.
//
// FLOAT EQUALITY, not tolerance: the fold compares the same F32 values the host scan compares, so
// "equal" means bit-equal here and `>` means bit-equal-and-greater there. No reassociation is involved
// (nothing is summed), which is why this kernel can be admitted by exact equality against the host rule
// rather than by a band.

#include <hip/hip_runtime.h>

#include <cstdint>

// The status-word layout now lives in `engine/kernels.h` (next to the launcher that consumes it), and
// NOT here, for a link rule rather than a taste: a header whose namespace contains a `__global__` may
// be included by exactly one translation unit — `engine/attn_fattn_kernels.h`'s note is the carried
// version of this trap, and this build hit it the moment a host file wanted the word offsets. So this
// header is the kernel and nothing else, and it includes `kernels.h` to get the integers.
#include "engine/kernels.h"

namespace engine::mtpverdict {

// The block size every reduction in this file uses, and the size its launchers are required to launch
// with: the shared trees below are sized from it, so a block of any other width would read uninitialised
// shared memory rather than fail loudly.
constexpr int kVerdictThreads = 256;

namespace detail {

// One candidate: a logit value and the index it came with. The layout is `engine::kernels::ArgmaxWord`
// itself, aliased rather than duplicated, so the partials a split leaves in global memory, the word a
// single row writes, and the struct the host reads back are one object with one spelling — the
// "two self-consistent layouts joined by a branch" failure (`AGENTS.md`) is unavailable by construction.
using Best = engine::kernels::ArgmaxWord;

inline __device__ __host__ bool better(const Best& a, const Best& b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
}

// The fold, spelled once so no reduction site can invent a tie rule of its own.
inline __device__ __host__ Best fold(const Best& a, const Best& b) {
    return better(a, b) ? a : b;
}

// The block half of every reduction here: a fixed shared-memory tree over `kVerdictThreads` entries.
// No atomics, so no arrival order and no run-to-run variation; the comparator does the tie-breaking that
// a `max` reduction would silently drop. The array must be block-shared and fully populated (see
// `clamp_index`: every thread contributes a real element, so there is no empty slot to define).
__device__ inline Best block_reduce(Best* shared) {
    __syncthreads();
    for (int stride = kVerdictThreads / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            shared[threadIdx.x] = fold(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

// A thread index past the end of a short strip re-reads its last element instead of contributing nothing.
// Re-folding an element the fold has already seen cannot change the result — `better(x, x)` is false and
// `x` can never beat the maximum it is part of — and this avoids an "I have no candidate" sentinel, which
// would have to be a value the logits row could legitimately contain.
__device__ inline int clamp_index(int i, int n) {
    return i < n ? i : n - 1;
}

}  // namespace detail

// One block per logits row: `blockIdx.x` is the row, `gridDim.x` is `nrows`, and row t starts at
// `logits + t * vocab`. The strided walk visits each thread's indices in increasing order and the
// fold below is tie-broken by the least index, so neither the block's own scan nor the tree that
// combines it can prefer a higher id.
__global__ void accept_verdict_kernel(const float* __restrict__ logits, int vocab, int candidate,
                                      int* __restrict__ out) {
    __shared__ detail::Best best[kVerdictThreads];

    const int row = (int)blockIdx.x;
    const float* row_logits = logits + (size_t)row * (size_t)vocab;

    // Strided walk: thread t owns t, t+T, t+2T, ... Strict `>` keeps the first index seen, and the
    // indices are seen in increasing order, so a tie inside one thread's own strip resolves to the
    // least index before the tree ever runs. (The `=` form of the declaration is not a style choice:
    // `detail::Best local{a, b}` parses as a function declaration, which is what this kernel's first
    // compile said and what the brace-equals form does not.)
    detail::Best local = {row_logits[threadIdx.x], (int)threadIdx.x};
    for (int i = (int)threadIdx.x + kVerdictThreads; i < vocab; i += kVerdictThreads) {
        const float v = row_logits[i];
        if (v > local.value) {   // strict, for the same reason `argmax_greedy` is
            local.value = v;
            local.index = i;
        }
    }
    best[threadIdx.x] = local;
    __syncthreads();

    // Fixed shared-memory tree. No atomics, so no arrival order and no run-to-run variation; the
    // comparator does the tie-breaking that a `max` reduction would silently drop.
    for (int stride = kVerdictThreads / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            const detail::Best a = best[threadIdx.x];
            const detail::Best b = best[threadIdx.x + stride];
            best[threadIdx.x] = detail::better(a, b) ? a : b;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int token = best[0].index;
        out[kernels::kVerdictRow0Token + row] = token;
        if (row == 0) {
            // THE COMPARISON, on device. One equality of two ints is the whole accept test, and it is
            // evaluated here so the round never has to bring a logits row home to ask it. `candidate`
            // arrives as a kernel argument rather than a buffer: uploading it would be a second device
            // operation to move four bytes, which is the mistake `embed_gather_rows`'s note records.
            out[kernels::kVerdictAccept] = (token == candidate) ? 1 : 0;
            out[kernels::kVerdictCandidate] = candidate;
        }
    }
}

// ---- the split argmax: a row is a grid, not a block -------------------------------------------------
//
// WHY THE SPLIT IS LEGAL AND WHY IT IS NOT A NEW NUMERICS QUESTION. There is no summation anywhere in this
// file, so nothing can reassociate: the only operation is `better(a, b)`, and over finite logits that is a
// TOTAL ORDER — the greatest value, and among the indices attaining it the least. A fold over a total order
// returns the same element whichever order the operands arrive in, so `argmax_greedy`'s left-to-right scan,
// this file's fixed block tree, a strip split across 160 blocks and a combine over 160 partials are four
// evaluations of one answer. That is the whole argument `ROUND-TAX.md` §1 offers for admitting the change
// with a byte-compare instead of a band, and (c) below is where it would break.
//
// (c) is NaN. Under a NaN `better` is false in both directions, so the order does start to matter — exactly
// as it already does for the one-block kernel and for the host scan this replaces. No logits row this engine
// produces contains a NaN (a NaN logit would already have failed `engine/gate.py`'s finite check), and this
// note exists so that if that ever stops being true, the file that has to change is named.
//
// WHY A SECOND KERNEL RATHER THAN AN ATOMIC OR A LAST-BLOCK TRICK: both of those make the surviving value
// depend on arrival order, which is precisely the dependence the argument above removes. `hipMalloc` inside a
// round is refused for a different reason (a synchronisation in the middle of the cost being measured), so
// the partials buffer is caller-owned and named in the launcher.
static_assert(kernels::kArgmaxChunks <= kVerdictThreads,
              "the combine kernel reduces one block's worth of partials; more chunks than threads would "
              "need a second level, which is a different kernel and a different argument");

// One strip of one row. `blockIdx.x` is the chunk and `blockIdx.y` the row, so a single row is a
// `kArgmaxChunks × 1` grid and a two-row tile is `kArgmaxChunks × 2` — one kernel for both shapes rather
// than a spelling per caller. `per_chunk` is the strip width in ELEMENTS and the launcher guarantees it is
// a multiple of 4, which is what makes the `float4` view legal: the row base is `vocab * 4` bytes in, and
// `kVocab % 4 == 0`, so every strip start is 16-byte aligned.//
// Vectorised loads, because the shape this replaced read 1.987 MB in 0.308 ms — 6.5 GB/s against this box's
// measured 237 GB/s ceiling (`ROUND-TAX.md` §2) — and one block per row left 39 of 40 CUs idle.
__global__ void argmax_partials_kernel(const float* __restrict__ logits, int vocab, int per_chunk,
                                       int nchunks, detail::Best* __restrict__ partials) {
    __shared__ detail::Best best[kVerdictThreads];

    const int row = (int)blockIdx.y;
    const float* strip = logits + (size_t)row * (size_t)vocab
                       + (size_t)blockIdx.x * (size_t)per_chunk;
    const float4* vec = reinterpret_cast<const float4*>(strip);
    const int nvec = per_chunk / 4;
    // The index a partial carries is the INDEX IN THE ROW, not in the strip: the combine compares partials
    // from different strips, and a strip-relative index would tie-break against the wrong id. Witnessed by
    // the byte compare: the first version of this line omitted `base`, and on the raw fixture it returned the
    // winning VALUE with the winning strip's local index — `1422` instead of `1422 + 31*1552 = 49534` — a
    // wrong proposal that `argmax_greedy` would never produce.
    const int base = (int)blockIdx.x * per_chunk;

    // Each thread owns vectors t, t+T, t+2T… of its strip and folds the four lanes of a vector in
    // increasing index order. `detail::fold` is the only comparison, so the walk cannot prefer a higher id
    // any more than the tree can.
    const int first = detail::clamp_index((int)threadIdx.x, nvec);
    float4 f = vec[first];
    detail::Best local = {f.x, base + first * 4};
    local = detail::fold(local, {f.y, base + first * 4 + 1});
    local = detail::fold(local, {f.z, base + first * 4 + 2});
    local = detail::fold(local, {f.w, base + first * 4 + 3});
    for (int i = first + (int)blockDim.x; i < nvec; i += (int)blockDim.x) {
        f = vec[i];
        const int j = base + i * 4;
        local = detail::fold(local,
                             detail::fold(detail::fold({f.x, j}, {f.y, j + 1}),
                                          detail::fold({f.z, j + 2}, {f.w, j + 3})));
    }
    best[threadIdx.x] = local;
    // `[row][chunk]`, spelled exactly as `row_argmax_combine_kernel` reads it: one address expression
    // shared by the writer and the reader, because §20 of `prototype/attn_decode/results.md` is this
    // repo's witness for what two self-consistent spellings of one partial layout do to a result.
    partials[(size_t)row * (size_t)nchunks + blockIdx.x] = detail::block_reduce(best);
}

// Combine one row's `nchunks` partials into its answer. `blockIdx.x` is the row, so `partials` is read as
// `[row][chunk]` — the same expression the partials kernel wrote, shared rather than restated (the
// partial-slot aliasing bug in `prototype/attn_decode/results.md` §20 is what that rule is for).
__global__ void row_argmax_combine_kernel(const detail::Best* __restrict__ partials, int nchunks,
                                          detail::Best* __restrict__ out) {
    __shared__ detail::Best best[kVerdictThreads];
    const int row = (int)blockIdx.x;
    const detail::Best* row_partials = partials + (size_t)row * (size_t)nchunks;
    best[threadIdx.x] = row_partials[detail::clamp_index((int)threadIdx.x, nchunks)];
    *out = detail::block_reduce(best);
}

}  // namespace engine::mtpverdict
