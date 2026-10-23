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

namespace detail {

struct Best {
    float value;
    int index;
};

inline __device__ __host__ bool better(const Best& a, const Best& b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
}

}  // namespace detail

// One block per logits row: `blockIdx.x` is the row, `gridDim.x` is `nrows`, and row t starts at
// `logits + t * vocab`. The strided walk visits each thread's indices in increasing order and the
// fold below is tie-broken by the least index, so neither the block's own scan nor the tree that
// combines it can prefer a higher id.
constexpr int kVerdictThreads = 256;

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

}  // namespace engine::mtpverdict
