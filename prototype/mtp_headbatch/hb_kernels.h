/*
 * prototype/mtp_headbatch/hb_kernels.h — the tied logits GEMV serving M query rows from ONE read
 * of the 485 MiB table, plus the byte witness that proves "one read" instead of asserting it.
 *
 * WHY. P7 plan A (prototype/mtp_delta/D3-RESULT.md) prices `delta` — what verifying two query rows
 * per decode step costs over one — with a single swing term left standing as a plan rather than a
 * measurement: `lm_head_batched`, ±2.17 ms. The plan says the second candidate row rides the table
 * read the first row already pays. Nothing here measures that; this file is the instrument the main
 * agent measures it with. `engine/generation-limits.md` §1 puts the one-row pass at 2.1685 ms
 * quoted, 104 % of the plateau, so the question is not "is there headroom" but "does a second row
 * cost a second cold pass (R ~ 2.0) or nothing new (R ~ 1.0)".
 *
 * SHAPE. W is the tied `model.language_model.embed_tokens.weight`, BF16 [248320, 1024],
 * 508,559,360 B — byte-identical to what prototype/lm_head_gemv feeds itself. `h` is M hidden rows
 * of H=1024 F32 (engine form: F32 values that came from a BF16 tensor, exactly lm_head_gemv's
 * h.bin). `out` is M logits rows, row-major by QUERY row (`out[q * V + v]`), because the engine's
 * tail wants one contiguous D2H covering both candidates.
 *
 * ONE READ, BY CONSTRUCTION. One warp owns one vocab row and fetches that row's 2 KiB exactly
 * `chunks` uint4 loads. The M dots are computed from registers already in hand: `u` is loaded once
 * and used M times, against M staged copies of h. So the fetch multiset is independent of M, and
 * the witness counts fetches rather than trusting that reading.
 *
 * THE DOT IS THE SHIPPED ONE, CHARACTER FOR CHARACTER. This file #includes
 * ../lm_head_gemv/lm_head_gemv_kernels.h and does not re-edit it (that header's own rule: nothing
 * in it may be edited for another's convenience), so the M=1 arm of this rig is a generalisation of
 * the shipped kernel, not a copy of it. The per-output summation order — `v = lane; v < HID/8;
 * v += 32` then `j = 0..7`, then the 5-step warp butterfly — is transcribed unchanged, and the
 * staged row is indexed as `hs[q * HID + v * 8]`, which for q = 0 is the shipped `hs[v * 8]`.
 * Therefore bit-identity across M is OWED, not hoped for: strict FP does not reassociate, and no
 * scalar is hoisted or re-associated (AGENTS, `engine/gdn_collapse_kernels.h`: bit-identity is owed
 * to the source text's association, not to the algebra). `gate.py` asserts it: row q of M=2 must be
 * bit-identical to that row run alone, and to the shipped binary's own output.
 *
 * THE WITNESS (WIT=true), and why it is not arithmetic. The claim under test is a TRAFFIC claim, so
 * the check counts bytes the kernel actually fetched, per AGENTS (`tools/mixed_bw`, `prototype/
 * gdn_walk`): a traffic statement read off a plan is a claim about the loops the source contains.
 *   * per lane: `lcs` = address-weighted sum of the fetched bytes, `lc0/lc1` = raw byte sum, `ln` =
 *     fetch count. Weighted by an odd multiplier per vocab row so a fetch delivered to the WRONG row
 *     changes the aggregate too, not only a fetch that is missing or duplicated.
 *   * per block: a REAL reduction — 64-bit warp butterfly, then one publish per warp to
 *     `wsum/wcnt/wraw[wid]`, `__syncthreads`, then warp 0 reduces the <=32 warp partials and thread 0
 *     publishes `cs[blockIdx.x]`. A per-thread store to a per-block slot is not a block reduction
 *     (AGENTS, `tools/mixed_bw`'s `publish_block_sum`), and that bug made a read checksum report
 *     1/blockDim of the bytes it had read while its GB/s looked entirely plausible.
 *   * host: sums the per-block slots and compares against `hb_expect_*`, which is derived by walking
 *     the intended (row, chunk) range from the launch plan — never by reading back what the kernel
 *     accumulated. A check that passes by construction cannot fail.
 * `fetch_per_weight = total_fetches / (V * chunks)` is the number plan A needs: 1.000000 at every M
 * means one table read served all M rows. `--inject extra_pass` (two launches, one accumulation
 * window) drives it to 2.000000 and that is the calibration: the witness is required to detect
 * exactly the failure plan A rules out.
 *
 * INJECTIONS live behind `if constexpr (WIT)`, so the timed instantiation carries no injection
 * branch at all and both instantiations derive every address from the same HbPlan fields. `gate.py`
 * additionally asserts the two instantiations produce bit-identical logits on the clean case, which
 * is the witness that the instrument is not the subject.
 *
 * Guard bands: the host allocates PAD words of poison before and after the output region and PAD
 * floats after the h region, and `gate.py` requires them untouched and requires zero poison words
 * left inside the output — "every element written" is checked by bytes touched, not by the timer.
 */
#pragma once

// The shipped M=1 kernel comes from here, together with HID, ROWS_PER_BLOCK and FULL_MASK. Including
// it (rather than pasting the dot) is what makes "the batched head is the shipped head, widened"
// checkable rather than a claim.
#include "../lm_head_gemv/lm_head_gemv_kernels.h"

#include <cstring>

#define HB_MAX_RPB 32              // one warp per row, so blockDim.x <= 32 * HB_MAX_RPB = 1024
#define HB_POISON_WORD 0x7fc00001u  // a NaN payload: no logit can equal it, and NaN != NaN

// One struct holds every geometry fact. The launch and the host-side expectation are both built
// from it, so a writer and a reader cannot disagree about an address (AGENTS, `prototype/
// attn_tile_model.py` §20.1: two self-consistent layouts joined by a branch are invisible to both).
struct HbPlan {
    int M;                      // query rows served by one pass
    int V;                      // vocab rows = 248320
    int rpb;                    // vocab rows per block (lm_head_gemv's `--rows`: rows per BLOCK)
    int chunks;                 // uint4 fetches per vocab row; the intended value is HID / 8
    int grid;                   // blocks = ceil(V / rpb)
    int threads;                // 32 * rpb
    unsigned long long intended_chunks;  // V * (HID / 8): the fetches a correct pass must report
    unsigned long long out_elems;        // M * V
    unsigned long long weight_bytes;     // V * HID * 2
};

inline HbPlan hb_plan(int M, int V, int rpb) {
    HbPlan p;
    p.M = M;
    p.V = V;
    p.rpb = rpb;
    p.chunks = HID / 8;
    p.grid = (V + rpb - 1) / rpb;
    p.threads = 32 * rpb;
    p.intended_chunks = (unsigned long long)V * (unsigned)(HID / 8);
    p.out_elems = (unsigned long long)M * (unsigned long long)V;
    p.weight_bytes = (unsigned long long)V * (unsigned long long)HID * 2ull;
    return p;
}

// The host and the device must compute the same three aggregates. Both write them here, once, so a
// drift between them is a compile-visible bug rather than a wrong PASS.
__host__ __device__ __forceinline__ void hb_fetch(unsigned row, const uint4& u,
                                                  unsigned long long& weighted,
                                                  unsigned long long& raw,
                                                  unsigned long long& count) {
    unsigned long long lo, hi;
    memcpy(&lo, (const char*)&u, 8);
    memcpy(&hi, (const char*)&u + 8, 8);
    // Odd multiplier per vocab row: a byte delivered to the wrong row changes the aggregate.
    weighted += lo * (2ull * (unsigned long long)row + 1ull)
              + hi * (2ull * (unsigned long long)row + 3ull);
    raw += lo + hi;
    count += 1ull;
}

// Walks the INTENDED range (every vocab row, every chunk) over a host copy of the table. Note that
// this deliberately does not mirror the kernel's lane striding: the intended multiset is what the
// traffic claim is about, and mirroring the kernel would make the check pass by construction.
inline void hb_expect(const char* W, const HbPlan& p, unsigned long long* weighted,
                      unsigned long long* raw, unsigned long long* count) {
    unsigned long long wsum = 0, rsum = 0, n = 0;
    const char* base = W;
    for (int row = 0; row < p.V; ++row) {
        const char* wr = base + (size_t)row * HID * 2;
        for (int c = 0; c < HID / 8; ++c) {
            uint4 u;
            memcpy(&u, wr + (size_t)c * 16, 16);
            hb_fetch((unsigned)row, u, wsum, rsum, n);
        }
    }
    *weighted = wsum;
    *raw = rsum;
    *count = n;
}

/*
 * The kernel. M query rows, one vocab row per warp, one fetch per weight byte.
 *
 * dropq / extra_pass are injections and exist only in the WIT instantiation. `dropq == q` drops the
 * last chunk of query row q's dot — a per-QUERY-row bug, which is the shape the must-catch set needs
 * (it must move exactly one output row and leave the others bit-identical).
 */
template <int M, bool WIT, int CH = HID / 8>
__global__ void __launch_bounds__(HB_MAX_RPB * 32)
hb_head(const hip_bfloat16* __restrict__ W, const float* __restrict__ h,
        float* __restrict__ out, unsigned long long* __restrict__ cs,
        unsigned long long* __restrict__ cnt, unsigned long long* __restrict__ raw,
        int V, int rpb, int chunks, int dropq, int rowdup, int rowshift) {
    __shared__ float hs[M * HID];
    for (int i = threadIdx.x; i < M * HID; i += blockDim.x) hs[i] = h[i];
    __syncthreads();

    const int row = blockIdx.x * rpb + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;

    unsigned long long weighted = 0, rawsum = 0, n = 0;
    if (row < V) {
        const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * HID);
        float acc[M];
#pragma unroll
        for (int q = 0; q < M; ++q) acc[q] = 0.f;
        // The TIMED instantiation must have the shipped kernel's compile-time trip count (HID/8),
        // so the chunk count is a template parameter there and only the witness takes a runtime one
        // (that is where the over-read injection lives). WIT is constexpr, so this folds.
        const int nch = WIT ? chunks : CH;
#pragma unroll
        for (int v = lane; v < nch; v += 32) {
            uint4 u = wrow[v];
            if constexpr (WIT) hb_fetch((unsigned)row, u, weighted, rawsum, n);
            const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int q = 0; q < M; ++q) {
                // The injection is per-query-row so that one output row moves and its neighbours do
                // not; the timed instantiation never sees the branch.
                if constexpr (WIT) {
                    if (dropq == q && v >= HID / 8 - 32) continue;
                }
                int sq = q;
                if constexpr (WIT) {
                    // row_dup: every query row reads staged row `rowdup`  (the "row 1 rides row 0's
                    // data" bug). row_shift: a cyclic relabelling of the query rows.
                    if (rowdup >= 0) sq = rowdup;
                    else if (rowshift > 0) sq = (q + rowshift) % M;
                }
                int off = sq * HID + v * 8;
                if constexpr (WIT) {
                    // Only the over-read injection can exceed the staged tile; clamping keeps it a
                    // wrong-value-in-the-right-place bug rather than a shared-memory overrun that
                    // would poison the whole context instead of failing one case.
                    if (off > M * HID - 8) off = M * HID - 8;
                }
                const float* hp = &hs[off];
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[q] += float(p[j]) * hp[j];
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
#pragma unroll
            for (int q = 0; q < M; ++q) acc[q] += __shfl_down_sync(FULL_MASK, acc[q], o);
        }
        if (lane == 0) {
#pragma unroll
            for (int q = 0; q < M; ++q) out[(size_t)q * V + row] = acc[q];
        }
    }

    if constexpr (!WIT) return;

    // ---- block reduction of the witness (real: butterfly, publish, butterfly, one publisher) ----
    for (int o = 16; o; o >>= 1) {
        weighted += __shfl_down_sync(FULL_MASK, weighted, o);
        rawsum += __shfl_down_sync(FULL_MASK, rawsum, o);
        n += __shfl_down_sync(FULL_MASK, n, o);
    }
    __shared__ unsigned long long wsum[HB_MAX_RPB];
    __shared__ unsigned long long wraw[HB_MAX_RPB];
    __shared__ unsigned long long wcnt[HB_MAX_RPB];
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        wsum[wid] = weighted;
        wraw[wid] = rawsum;
        wcnt[wid] = n;
    }
    __syncthreads();
    if (wid == 0) {
        unsigned long long a = (lane < rpb) ? wsum[lane] : 0ull;
        unsigned long long b = (lane < rpb) ? wraw[lane] : 0ull;
        unsigned long long c = (lane < rpb) ? wcnt[lane] : 0ull;
        for (int o = 16; o; o >>= 1) {
            a += __shfl_down_sync(FULL_MASK, a, o);
            b += __shfl_down_sync(FULL_MASK, b, o);
            c += __shfl_down_sync(FULL_MASK, c, o);
        }
        if (lane == 0) {
            cs[blockIdx.x] = a;
            raw[blockIdx.x] = b;
            cnt[blockIdx.x] = c;
        }
    }
}

// The shipped M=1 kernel (lm_head_gemv_full) is instantiated by including its header above; the rig
// launches it directly as the baseline arm, so the baseline is the shipping code, not a paraphrase.
