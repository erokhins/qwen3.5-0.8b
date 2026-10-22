/*
 * prototype/mtp_draft/mtp_draft_kernels.h — the NEW device code of the MTP draft-block rig: the
 * embedding-row gather, the byte witness that prices the block's traffic instead of asserting it, and
 * the derived work census. Everything that already exists in this repo is INCLUDED, not rewritten; see
 * td_glue_tu.cpp / td_proj_tu.cpp / td_mlp_tu.cpp / td_attn_tu.cpp for the four families the block reuses
 * and DRAFT-HANDOVER.md §0 for the launch-by-launch map.
 *
 * WHY THESE THREE AND NO OTHERS. `prototype/mtp_headbatch/hb_kernels.h` states the rule this rig follows:
 * a rig may generalise a shipped kernel by #including it, and anything it cannot include must be small
 * enough that its whole arithmetic fits on one screen. The embedding gather has no shared header
 * (`engine/kernels_embed.cpp` is a TU with an anonymous namespace), so it is transcribed below; the byte
 * witness exists because the block is a TRAFFIC claim and AGENTS forbids pricing traffic off a plan.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#define TD_FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit (AGENTS trap table)

/*
 * ── 1. the embedding-row gather ──────────────────────────────────────────────────────────────────
 *
 * Transcribed from `engine/kernels_embed.cpp`'s `embed_gather_kernel`, which is itself the only other
 * kernel in the tree that is not lifted from a `prototype/<op>/` header. Same shape, same reason: there
 * is no arithmetic here at all, so there is nothing to band. Row `b` of the output is the kHidden BF16
 * values at `table + ids[b] * kHidden`, stored exactly as loaded, and a row is a whole number of uint2
 * chunks so the walk covers it exactly once with no remainder. The draft step needs exactly one row of
 * the tied table (§09.1: `e = E[y_next]`, shared table, `mtp_use_dedicated_embeddings: false`), and this
 * is one launch instead of a per-token `hipMemcpyAsync`.
 *
 * The static_assert is not decoration: a remainder would leave the tail of every row unwritten, which is
 * a silent wrong answer, and the same argument the engine's copy makes.
 */
#define TD_HIDDEN 1024
constexpr int TD_CHUNKS_PER_ROW = TD_HIDDEN / 4;   // uint2 (8 B) chunks in one BF16 row
static_assert(TD_HIDDEN % 4 == 0, "the gather's uint2 walk needs a row that is a whole number of uint2");

__global__ void td_gather_row(const hip_bfloat16* __restrict__ table, const int* __restrict__ ids,
                              hip_bfloat16* __restrict__ out) {
    const uint2* src = (const uint2*)(table + (size_t)ids[blockIdx.x] * TD_HIDDEN);
    uint2* dst = (uint2*)(out + (size_t)blockIdx.x * TD_HIDDEN);
    dst[threadIdx.x] = src[threadIdx.x];
}

/*
 * ── 2. the byte witness ──────────────────────────────────────────────────────────────────────────
 *
 * WHAT IT PRICES. The MTP block is 40,905,728 B of weights per draft step and the whole route is priced
 * off that number (prototype/mtp_gatezero/mtp_census.json). A transfer claim read off a byte count is a
 * claim about the loops the source contains, so this kernel reads the block's byte multiset once, through
 * the same segment list the launch plan is built from, and reports what it actually touched. AGENTS
 * (`tools/mixed_bw`, `prototype/gdn_walk`) is explicit about the two ways this kind of check lies: a
 * per-thread store to a per-block slot is not a block reduction (that bug made a read checksum report
 * 1/blockDim of its bytes while its GB/s looked plausible), and a check derived from the value the kernel
 * accumulates cannot fail. Both are handled: the reduction is a real 64-bit warp butterfly with one
 * publish per block, and `gate.py` recomputes the expected triple from the host copy of the blobs, never
 * from `cs`.
 *
 * THREE ACCUMULATORS, because they fail differently:
 *   n     uint2 elements read. Off by a constant on a wrong bound.
 *   raw   sum of the 8-byte values. Off on a wrong pointer or a wrong length.
 *   addr  sum of value * (2*flat_index + 1). `raw` is blind to a fetch delivered to the WRONG place and
 *         to a duplicated-and-dropped pair; the odd per-element weight is not, because it makes the
 *         contribution position-dependent. (Same trick hb_kernels.h uses for its fetch witness.)
 * All three wrap mod 2^64, and the host expectation wraps the same way (the host function is selftested
 * against a bignum walk in gate.py, because a witness whose expectation is wrong is worse than none).
 */
struct TdSeg {
    const uint2* base;   // segment start, 8-byte aligned (every MTP tensor is a multiple of 16 B)
    long n;              // uint2 elements in this segment
    long prefix;         // flat index of this segment's first element
};

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
td_read_witness(const TdSeg* __restrict__ segs, int nsegs, unsigned long long* out /* 3 per block */) {
    const long total = segs[nsegs - 1].prefix + segs[nsegs - 1].n;
    const long stride = (long)gridDim.x * THREADS;
    unsigned long long n = 0, raw = 0, addr = 0;
    int s = 0;
    for (long j = (long)blockIdx.x * THREADS + threadIdx.x; j < total; j += stride) {
        while (s + 1 < nsegs && j >= segs[s].prefix + segs[s].n) ++s;
        const uint2 v = segs[s].base[j - segs[s].prefix];
        const unsigned long long vraw = (unsigned long long)v.x + (unsigned long long)v.y;
        n += 1;
        raw += vraw;
        addr += vraw * (2ull * (unsigned long long)j + 1ull);
    }
    // A real block reduction: 64-bit butterfly inside each warp, one publish per warp to shared, then
    // warp 0 reduces the <= THREADS/32 partials and thread 0 publishes this block's triple.
    __shared__ unsigned long long sh[3][THREADS / 32];
    for (int k = 0; k < 3; ++k) {
        unsigned long long v = (k == 0) ? n : (k == 1) ? raw : addr;
        for (int o = warpSize / 2; o; o >>= 1)
            v += __shfl_down_sync(TD_FULL_MASK, v, o);
        if (threadIdx.x % warpSize == 0) sh[k][threadIdx.x / warpSize] = v;
    }
    __syncthreads();
    // Second ladder runs in warp 0 with EVERY lane of that warp live, which is what makes the full
    // participating-lane mask correct here rather than a lie (same discipline, and the same reason
    // written down, as prototype/glue/glue_kernels.h's block_reduce).
    if (threadIdx.x < (unsigned)warpSize) {
        for (int k = 0; k < 3; ++k) {
            unsigned long long v = threadIdx.x < THREADS / 32 ? sh[k][threadIdx.x] : 0ull;
            for (int o = warpSize / 2; o; o >>= 1)
                v += __shfl_down_sync(TD_FULL_MASK, v, o);
            if (threadIdx.x == 0) sh[k][0] = v;
            __syncwarp();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out[blockIdx.x * 3 + 0] = sh[0][0];
        out[blockIdx.x * 3 + 1] = sh[1][0];
        out[blockIdx.x * 3 + 2] = sh[2][0];
    }
}

/*
 * Poison filler. A guard band that was zero proves nothing, because a stray store of a zero is invisible;
 * every guard word in this rig is filled with a value no write in the rig can produce (gate.py is given
 * the constant and requires the guards to still hold it after the whole chain runs). Written as a pattern
 * that cannot arise from BF16 activations: BF16 pairs of 0x8A82 are NaN-ish sentinel values that no
 * arithmetic in the block emits, and the same 32-bit word repeats.
 */
#define TD_POISON 0x8A828A82u
__global__ void td_poison(uint* p, long n) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        p[i] = TD_POISON;
}

/*
 * Exact BF16 -> F32 widening, for the staging check only (`--mode stagecheck`). There is no arithmetic
 * here either: every BF16 value is representable in F32, so this is a placement of the widening, not a
 * value change — which is precisely the claim `engine/mlp_xbf16_kernels.h` makes for the shipping MLP and
 * the claim this rig re-proves on the block's own operands instead of inheriting.
 */
__global__ void td_widen(const hip_bfloat16* __restrict__ src, float* __restrict__ dst, long n) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) dst[i] = float(src[i]);
}

/*
 * The derived work census lives in `td_host.h` (`tdplan::`), so the host entry and the mode
 * implementation read the plan out of one declaration rather than two.
 */
