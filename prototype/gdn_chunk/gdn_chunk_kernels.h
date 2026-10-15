/*
 * prototype/gdn_chunk/gdn_chunk_kernels.h — §04.7's chunked delta-rule scan, device code.
 *
 * The CPU contract this file is written to is `prototype/gdn_chunk/chunk_scan.py`, and the form here
 * is the form that file priced: `chunked_scan(..., mma=True)` — every matmul operand BF16, every
 * accumulator F32, with the two exceptions that file names (the `ut_inverse` substitution stays F32;
 * `nv` is a RESULT, not an operand). The budget those forms were admitted against is
 * `prototype/gdn_chunk/phase1.py`: stage meanrel <= 9.9e-3 (Phase 0 §2.1) and per-row
 * 1-cos <= 3.2e-4 (§2.3). C=64 is the chunk size that budget selects; C=16 is rejected there at
 * 1-cos 3.388e-4 (> 3.2e-4) and C=32 is inside the budget but not built, so the launcher accepts
 * 64 and refuses everything else with the reason printed.
 *
 * TWO LAUNCHES per (layer, body), which is what P9 is for (progress/2026-10-14/p9-phase1-2026-10-14.md §6:
 * gdn_rec is 18,432 launches and 33.2 ms of dispatch per 1024-token wave):
 *
 *   `gdn_chunk_prep`   grid (chunks, heads) — one block per (chunk, head). Embarrassingly parallel,
 *                      because everything in it is intra-chunk: normalise q/k, the two [C,C]
 *                      matrices (`ut`, `intra`), the unit-triangular solve, and the two
 *                      applications of its inverse that produce `nv` and `kcd`. No block reads or
 *                      writes the state.
 *   `gdn_chunk_carry`  grid (d_v/32, heads) — one block per (head, 32-column strip of d_v). The only
 *                      SERIAL part: it walks the chunk axis, keeps that state strip resident in
 *                      LDS, and so touches the [16][128][128] state ONCE per body instead of once
 *                      per token (§04.4 moves 38.65 GB of state per 1024-token wave; this moves
 *                      ~0.6 GB, and it is the state traffic, not the flops, that has to go).
 *
 * WHY IT SPLITS TWO WAYS AND NOT FOUR: the serial depth of the scan is the number of chunks, and
 * only the inter-chunk carry has to be serial. FLA's shape (invert 16x16 diagonal blocks, combine
 * off-diagonal blocks by matmul) splits `prep` further and is allowed by the same budget — §6 costs
 * the four-site plan at 144 launches per wave, still ~0.5 ms against the 33 ms this package exists
 * to remove — but it is not what this file does.
 *
 * ----------------------------------------------------------------------------------------------
 * DEVIATIONS FROM THE CPU MODEL, and whether each one can change bits. Read this before gating.
 *
 *   D1  (CHANGES BITS) F32 accumulation order inside every dot. A thread here walks d (or the chunk
 *       row index) serially; the CPU model asks torch for a bf16 matmul, whose association is
 *       torch's. This is exactly the class §10.1 exists for — a chained dot's legal ndiff is
 *       29-47 %, which is why phase1 judged the chunk forms with cos/meanrel BUDGETS and never with
 *       a bit-identity claim, and why no gate here may demand bit-identity with the mirror.
 *   D2  (CHANGES BITS, <= 1 operand-ULP) k~ is staged in LDS as BF16 rather than F32. For the two
 *       matmuls that take k~ as an operand that is precisely what the model rounds to anyway, so it
 *       is free. But three quantities are DERIVED from k~ — `k*beta` (ut's first operand),
 *       `k*beta*Gamma` (kcd's right-hand side) and `khat` — and the model derives them from the F32
 *       k~ and rounds once, where this file derives them from a BF16 k~ and so rounds twice.
 *       The line it serves is LDS: F32 k~ is 32 KiB, which with the F32 C x C matrix (16 KiB) is
 *       48 KiB, over the ~38.9 KiB per-block ceiling this engine has measured itself against
 *       (`prototype/attn_decode/results.md` §12). BF16 k~ is 16 KiB and the block is ~33 KiB. The
 *       alternative §6 offered — build the [C,C] matrices in row-strips of 16 rows — is the right
 *       next move if a profile says these operands are the problem rather than the indexing.
 *   D3  (NO BIT CHANGE) staging: q~/k~ and the state strip through LDS, `intra`/`nv`/`kcd` through
 *       global scratch instead of registers. Per-element values and per-dot sums unchanged.
 *   D4  (NO BIT CHANGE) the [C,C] matrices are evaluated over every (t,s) pair and the s > t half
 *       thrown away rather than never evaluated; `intra`'s upper half is stored as the zeros the
 *       carry then reads, which is `dec.tril()` in the model. Wasted flops, not wrong bits.
 *   D5  (NO BIT CHANGE) the intra-chunk cumsum is one thread walking 64 adds. It is a prefix sum;
 *       a block-wide scan would cost more synchronisation than the 64 adds it saves.
 *   D6  (CHANGES BITS — DEFECT, UNFIXED AS COMMITTED) `khat`'s boundary decay is formed at :263 as
 *       `gamC / gam[i]` — two F32 `expf`s divided — where the CPU contract forms it as ONE exp of a
 *       difference (chunk_scan.py:196, `k * (cum[..., -1:] - cum).exp()`). They are equal in exact
 *       arithmetic and not equal in F32: `cum` decreases (every gate is negative), so `gamC <= gam[i]`
 *       and the two underflow TOGETHER, once the within-chunk gate sum passes about -104 (-87.3 with
 *       denormals flushed). The quotient is then 0/0 = NaN — for EVERY row of that chunk, including
 *       the last one, whose true ratio is exactly 1.0. The real pinned gates reach -163 (p64) and
 *       -293 (p300). `khat` is read only by the state update, so the damage lands in the state the
 *       NEXT chunk inherits: chunk 0's rows are right and every row of every later chunk is NaN,
 *       which is exactly the Phase 1b device signature (first NaN output row 64 = start of chunk 1;
 *       an all-NaN logits row argmaxes to token 0 under a strict `>`, hence "first answer token 0
 *       vs pin 279"). `dec8` passes because its worst gate sum is -61, not because one chunk is
 *       safe: p64 is a one-chunk fixture and is already poisoned (114,688 NaN state elements) with no
 *       later chunk to show it. The fix is the model's expression, in place:
 *           a.kh[off] = hip_bfloat16(kq * kr * expf(cum[GC_C - 1] - cum[i]));
 *       It is NOT applied here because the decision this kernel belongs to is a kill
 *       (progress/2026-10-14/p9-phase1c-2026-10-14.md §1c.3) and it has not been run on a device. Reproduced and
 *       bisected on CPU by `prototype/gdn_chunk/kernel_mirror.py` (`finiteness`, `candidates`).
 *       The general trap, for AGENTS.md: rewriting a decay ratio as a quotient of two exps is NaN
 *       where the original was 0 or 1. Keep the subtraction inside the exp.
 *
 * ONE-ROW BODIES DO NOT COME HERE. `engine/decode_step.cpp` routes a one-row body to `rec_kernel`
 * whatever this file offers, so every decode bit stays the shipped sequential form (§04.4 is what
 * the decode pins are pinned to). The distinction that makes that rule load-bearing rather than
 * cosmetic is measured by `prototype/gdn_chunk/phase1.py onerow` and is worth stating exactly, because
 * the two halves of the state update behave differently:
 *
 *   STATE  a chunked C=64 body of ONE token leaves S bit-identical to the sequential recurrence, at
 *          every layer probed, from a zero state and from a carried one.
 *   OUTPUT `o` is NOT bit-identical — worst meanrel 2.66e-07, cos 1.00000000 — because the chunked
 *          form reads the just-written state as a BF16 matmul operand where §04.4's recurrence reads
 *          it as the F32 value it just computed. Same numbers, different last bits, and the
 *          difference is §04.4's read, not a bug in either form.
 *
 * So "state-identical, read-differs" is the honest description of what a one-row chunk would cost:
 * nothing on the carry, ~2.7e-07 on every decode step's output, compounding over the steps of a
 * generation. That is under every band we have, and it is still not allowed, because decode's claim
 * is byte-identity and a byte-identity claim is not traded away for a launch count. The dispatch rule
 * (`kernels.h:resolve_gdn_scan`) is therefore what decode bit-identity rests on — not this file's
 * arithmetic — which is why it lives in one function and why `gdn_chunk_prefill` throws if it is ever
 * reached with `n < 2`.
 *
 * INJECTED BUGS. Each must move output rows AND must leave the rows where the bug is not; both sets
 * are DERIVED from the geometry, per the trap in AGENTS.md's index (a hand-listed set rots into
 * "everything" the moment the fixture changes).
 *   GCK_CARRY_STALE   the inter-chunk carry is stale: chunk c reads the state as of chunk 0 instead
 *                    of chunk c-1. Cannot move: chunk 0's rows and chunk 0's state, every head, and
 *                    every one-chunk body — so the must-not-catch set is a function of NC, not a list.
 *   GCK_BOUND_HEAD    the chunk boundary is taken on the wrong head: from chunk 1 on, the state
 *                    UPDATE reads `kh`/`vn` from head (h+1)%NH. Cannot move: chunk 0's rows (the
 *                    update lands after them) or any one-chunk body. Moves: every row of chunk 1..
 *                    of head h — head-mixing, which a whole-state comparison localises no better
 *                    than a per-head one, so the witness is per-head and per-chunk.
 *   GCK_UT_NO_DECAY   `ut` is built without the intra-chunk decay, dec[t,s] = 1. Cannot move: row 0
 *                    of every chunk (its only s is t, where dec = exp(0) = 1 already) — the bug is
 *                    invisible on a C=1 chunk by construction.
 *   GCK_PAD_ROW       the ragged tail is not padded: the tail chunk treats rows at or past `n` as
 *                    real, repeating row n-1's beta and gate. Cannot move: every body whose n is a
 *                    multiple of GC_C. Moves: that body's final state and nothing else — no output
 *                    row moves, because padded rows write no output.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>

#define GC_C 64          // the chunk size the CPU budget selected; the launcher refuses any other
#define GC_D 128         // d_k = d_v = 128 (§00.2)
#define GC_NH 16         // h_v
#define GC_THREADS 128   // one thread per d index
#define GC_STRIP 32      // d_v columns per carry block: the state strip is 128 x 32 x 4 = 16 KiB
#define GC_QKV 6144      // in_proj_qkv's row width: q|k|v, §04.1
#define GC_VD 2048       // one of those three slices, and o's row width
#define GC_QSCALE 0.08838834764831845f   // 128^{-1/2}, §04.4
#define GC_EPS 1e-6f     // §00.2 rms_norm_eps, and inside the root (A-05)
#define GC_FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit

enum GdnChunkNeg {
    GCK_NONE = 0,
    GCK_CARRY_STALE,
    GCK_BOUND_HEAD,
    GCK_UT_NO_DECAY,
    GCK_PAD_ROW,
};

__device__ __forceinline__ float gc_bf(float x) { return float(hip_bfloat16(x)); }

// The same two-ladder tree `prototype/gdn_decode/gdn_decode_kernels.h` spells `block_sum`, copied
// under its own name for the reason stated at the top of that file: each op owns its helper
// spellings and merging two op headers into one TU is a compile error. The copy is deliberate — with
// the same tree and the same expression, the q~/k~ this kernel normalises are the SAME F32 values
// `rec_kernel` normalises, which is what makes the aligned-chunk agreement between the two forms an
// arithmetic question rather than an indexing accident.
__device__ __forceinline__ float gc_block_sum(float v, float* red) {
    v += __shfl_down_sync(GC_FULL_MASK, v, 16);
    v += __shfl_down_sync(GC_FULL_MASK, v, 8);
    v += __shfl_down_sync(GC_FULL_MASK, v, 4);
    v += __shfl_down_sync(GC_FULL_MASK, v, 2);
    v += __shfl_down_sync(GC_FULL_MASK, v, 1);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const int warps = blockDim.x >> 5;
    float t = (threadIdx.x < (unsigned)warps) ? red[threadIdx.x] : 0.f;
    t += __shfl_down_sync(GC_FULL_MASK, t, 16);
    t += __shfl_down_sync(GC_FULL_MASK, t, 8);
    t += __shfl_down_sync(GC_FULL_MASK, t, 4);
    t += __shfl_down_sync(GC_FULL_MASK, t, 2);
    t += __shfl_down_sync(GC_FULL_MASK, t, 1);
    __syncthreads();
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    const float out = red[0];
    __syncthreads();
    return out;
}

/*
 * Both kernels' inputs and body-scoped scratch. `u` is the conv's [n][6144] F32 tile sliced q|k|v at
 * 0/2048/4096 exactly as `submit_wave_body` slices it for `gdn_rec`; `g`/`beta` are [n][16]. The
 * staging arrays are per-request scratch reused per layer, like `g_u`/`g_o`: nothing in them survives
 * the body. Only `S` does.
 *
 * The staging DTYPES are the model's rounding points, not a storage preference: `kcd`, `qg`, `kh`
 * and `intra` are BF16 because each is a matmul OPERAND on the carry side (the model rounds every
 * operand, and each of these is used exactly once, so rounding at the store is the same value as
 * rounding at the load). `nv` is F32 because in `v_new = nv - kcd@S` it is a result being subtracted,
 * never a rounded operand — which is the asymmetry phase1's `nv` comment warns about.
 */
struct GdnChunkArgs {
    const float* u;              // [n][6144] post-conv q|k|v (§04.1)
    const float* g;              // [n][16]   §04.3 log decay
    const float* beta;           // [n][16]   §04.3
    float* S;                    // [16][128][128] F32 (§08.5). The carry alone touches it.
    float* o;                    // [n][2048] F32, §04.5's input
    float* nv;                   // [n][16][128] F32
    hip_bfloat16* kcd;           // [n][16][128]
    hip_bfloat16* qg;            // [n][16][128]  q~ * Gamma_t
    hip_bfloat16* kh;            // [n][16][128]  k~ * (Gamma_C / Gamma_t)
    hip_bfloat16* intra;         // [n][16][GC_C] row t of the intra-chunk attention matrix
    float* cdec;                 // [chunks][16] Gamma_C per (chunk, head)
    int n;                       // rows in this body
    int nc;                      // ceil(n / GC_C)
    int neg;                     // GdnChunkNeg
};

/* ---- prep: one block per (chunk, head) -------------------------------------------------------
 *
 * The intra-chunk half of §04.7, in `chunk_scan.py`'s order:
 *
 *   normalise  q~ = q * rsqrt(sum q^2 + eps) * Q_SCALE ;  k~ = k * rsqrt(sum k^2 + eps)   §04.4
 *   ut         (k~ beta) k~^T * dec      strict lower — the UT system's coefficients
 *   intra      (q~ k~^T) * dec           INCLUDING the diagonal: §04.7's note that a token writes
 *                                        state and reads it in the same step
 *   solve      U = (I + strict_tril(ut))^{-1}, ascending rows, F32
 *   apply      nv  = U (v beta)          F32 result
 *              kcd = U (k~ beta Gamma)   BF16: it is a carry-side operand
 *
 * `dec[t][s] = exp(cum_t - cum_s)` for s <= t, with `cum` a prefix sum of the raw gate over THIS
 * CHUNK ONLY (§04.7's local-index convention: a chunk does not know earlier chunks exist; the
 * carry is where chunk c-1's state arrives).
 *
 * Ragged tail (§04.7 A-08): a row at or past the body's `n` is q=k=v=0, g=0, beta=0. That zeroes its
 * ut row, its v and k products and its contribution to `cum`, so the padded rows neither move a real
 * row nor move the state — and `gamC` stays the real rows' total because the pads add 0 to the sum.
 */
__global__ void gdn_chunk_prep(GdnChunkArgs a) {
    __shared__ hip_bfloat16 kst[GC_C * GC_D];   // k~, staged BF16 — see D2        16.00 KiB
    __shared__ float m[GC_C][GC_C];             // ut, then U in place, F32       16.00 KiB
    __shared__ float urow[GC_C];                // the row being substituted       0.25 KiB
    __shared__ float cum[GC_C], bet[GC_C], gam[GC_C], qf[GC_C], kf[GC_C];
    __shared__ float red[32];                                                  // + 1.41 KiB
    const int h = blockIdx.y;
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = c * GC_C;        // this chunk's first token, in body coordinates
    const int nvalid = a.n - base;    // rows this chunk really owns; < GC_C only on the tail

    // ---- the chunk's gates and the intra-chunk cumsum (D5) ---------------------------------
    if (tid == 0) {
        float run = 0.f;
        for (int i = 0; i < GC_C; ++i) {
            const int t = base + i;
            // GCK_PAD_ROW is the "the tail was not padded" reading: it repeats the last real row's
            // beta and gate into the padding instead of using §04.7's zeros.
            const bool padrow = (i >= nvalid);
            const int src = (padrow && a.neg == GCK_PAD_ROW) ? a.n - 1 : t;
            const float gv = padrow ? ((a.neg == GCK_PAD_ROW) ? a.g[(size_t)src * GC_NH + h] : 0.f)
                                    : a.g[(size_t)t * GC_NH + h];
            const float bv = padrow ? ((a.neg == GCK_PAD_ROW) ? a.beta[(size_t)src * GC_NH + h] : 0.f)
                                    : a.beta[(size_t)t * GC_NH + h];
            run += gv;
            cum[i] = run;
            bet[i] = bv;
        }
    }
    __syncthreads();
    if (tid < GC_C) gam[tid] = expf(cum[tid]);
    __syncthreads();
    const float gamC = gam[GC_C - 1];     // §04.7's Gamma_C = exp(sum of the chunk's gates)
    if (tid == 0) a.cdec[(size_t)c * GC_NH + h] = gamC;

    // ---- normalise q/k, and write the carry side's two scaled operands ---------------------
    // One row per iteration with the whole 128-thread block: `rec_kernel`'s reduction shape, so the
    // normalised values are its values. `kf`/`qf` keep the rsqrt factors, which lets the [C,C] pass
    // rebuild any row from `u` without a second reduction.
    for (int i = 0; i < GC_C; ++i) {
        const int t = base + i;
        const bool live = (i < nvalid);
        const float kq = live ? a.u[(size_t)t * GC_QKV + GC_VD + (size_t)h * GC_D + tid] : 0.f;
        const float qq = live ? a.u[(size_t)t * GC_QKV + (size_t)h * GC_D + tid] : 0.f;
        const float kr = rsqrtf(gc_block_sum(kq * kq, red) + GC_EPS);
        __syncthreads();
        const float qr = rsqrtf(gc_block_sum(qq * qq, red) + GC_EPS);
        __syncthreads();
        kf[i] = kr;
        qf[i] = qr;
        kst[i * GC_D + tid] = hip_bfloat16(kq * kr);        // D2: rounded on the way into LDS
        // Written for EVERY row of the chunk, padding included, and the padded rows land on zero
        // because their q, k, beta and g are zero. That is not tidiness: the carry reads these arrays
        // for all GC_C rows of the tail chunk, and staging it leaves unwritten is uninitialized scratch
        // feeding the state update.
        const size_t off = ((size_t)(base + i) * GC_NH + h) * GC_D + tid;
        if (base + i < a.n) {
            a.qg[off] = hip_bfloat16(qq * qr * GC_QSCALE * gam[i]);
            a.kh[off] = hip_bfloat16(kq * kr * (gamC / gam[i]));
        }
    }
    __syncthreads();

    // ---- the two [C,C] matrices -------------------------------------------------------------
    // One (t,s) pair per thread, d walked serially. Warp-uniform t within a warp is not the point
    // here (t = p / GC_C spreads a warp over two rows), but every `kst` read is a broadcast-shaped
    // row read and every write is one element of `m` or of `intra`, so nothing races: each (t,s)
    // slot has exactly one writer.
    for (int p = tid; p < GC_C * GC_C; p += GC_THREADS) {
        const int t = p / GC_C, s = p % GC_C;
        if (s > t) {
            // The zero half of `intra`, which is `dec.tril()`; `ut`'s is never stored, because the
            // solve only ever reads column < row of `m`.
            if (base + t < a.n) a.intra[((size_t)(base + t) * GC_NH + h) * GC_C + s] = hip_bfloat16(0.f);
            continue;
        }
        const float bt = bet[t];
        const float dec = (a.neg == GCK_UT_NO_DECAY) ? 1.0f : expf(cum[t] - cum[s]);
        float accu = 0.f, acci = 0.f;
        // A padded row's q never gets read: there is no token there, so there is no row of `u` to
        // read, and the row's own products are zero by the padding rule regardless.
        const bool live_t = (base + t < a.n);
        for (int d = 0; d < GC_D; ++d) {
            const float ksd = float(kst[s * GC_D + d]);
            // ut's first operand is (k~ * beta) rounded, per `_mm(k * be, kt)`.
            accu += gc_bf(float(kst[t * GC_D + d]) * bt) * ksd;
            // `intra`'s is q~ rounded and UNSCALED by Gamma — the Gamma scaling belongs to the
            // carry-side `q~ Gamma` operand (phase1's `q[:, i] * gam_i`), not to this product.
            const float qv = live_t
                                 ? a.u[(size_t)(base + t) * GC_QKV + (size_t)h * GC_D + d] * qf[t]
                                       * GC_QSCALE
                                 : 0.f;
            acci += gc_bf(qv) * ksd;
        }
        if (s < t) m[t][s] = accu * dec;
        if (live_t) a.intra[((size_t)(base + t) * GC_NH + h) * GC_C + s] = hip_bfloat16(acci * dec);
    }
    __syncthreads();

    // U's row 0 before the substitution starts: `ut_inverse`'s i = 0 case is e_0 with nothing
    // subtracted, and row 0 of `m` was never written above (the strict lower of row 0 is empty), so
    // leaving it here is leaving the FIRST row of every subsequent row's sum uninitialized — which is
    // not a wrong bit, it is the whole solve.
    if (tid < GC_C) m[0][tid] = (tid == 0) ? 1.0f : 0.0f;
    __syncthreads();

    // ---- the unit-triangular solve ----------------------------------------------------------
    // U = (I + strict_tril(ut))^{-1}, ascending rows, IN PLACE: row t's `ut` entries are read into
    // `urow` before any of them is overwritten by U's row, and rows < t already hold U. F32
    // throughout, including the products — the model's `ut_inverse` deliberately does not apply the
    // matmul rounding there ("the C x C coefficients are scalar F32 products there, not a
    // matrix-unit tile"), so rounding here would be the deviation.
    for (int t = 1; t < GC_C; ++t) {
        if (tid < GC_C) {
            float acc = 0.f;
            for (int s = 0; s < t; ++s) acc += m[t][s] * m[s][tid];
            urow[tid] = (tid < t) ? -acc : ((tid == t) ? 1.0f : 0.0f);
        }
        __syncthreads();
        if (tid < GC_C) m[t][tid] = urow[tid];
        __syncthreads();
    }

    // ---- apply U to the two right-hand sides -----------------------------------------------
    // One thread per d column, chunk rows ascending. Both matmuls reduce over the ROW index s, so
    // U's element is uniform across the block and comes out of LDS as a broadcast; the two
    // right-hand sides are what differ (`v beta` from global, `k~ beta Gamma` from LDS).
    for (int t = 0; t < GC_C; ++t) {
        float accn = 0.f, acck = 0.f;
        for (int s = 0; s <= t; ++s) {
            const float us = gc_bf(m[t][s]);   // U is F32 in LDS and BF16 as a matmul operand: the
                                               // model rounds it here and only here
            if (us == 0.f) continue;
            const int ts = base + s;
            const bool live = (ts < a.n);
            const float vv = live ? a.u[(size_t)ts * GC_QKV + 2 * GC_VD + (size_t)h * GC_D + tid] : 0.f;
            accn += us * gc_bf(vv * bet[s]);
            acck += us * gc_bf(float(kst[s * GC_D + tid]) * bet[s] * gam[s]);
        }
        // Every row of the chunk is written, padding included, for the carry's sake (see the normalise
        // loop). The padded rows are zero: their U row is e_t, and their v, beta and k~ are all zero.
        if (base + t < a.n) {
            const size_t off = ((size_t)(base + t) * GC_NH + h) * GC_D + tid;
            a.nv[off] = accn;
            a.kcd[off] = hip_bfloat16(acck);
        }
    }
}

/* ---- carry: one block per (head, 32-column strip of d_v); the chunk axis is serial -----------
 *
 * Per chunk, in §04.7's order:
 *   v_new = nv - kcd @ S_in                    (S_in is BF16-rounded: it is an operand)
 *   out   = (q~ Gamma) @ S_in + intra @ v_new
 *   S     = Gamma_C * S_in + khat^T @ v_new    and S stays in LDS for the next chunk
 * The chunk loop is the only serial part of the scan, and the state strip never leaves LDS inside
 * it: that is where 38.65 GB per wave of state traffic goes, and where the launch count goes from
 * 512 per layer per body to 1.
 *
 * Two thread mappings, ~2048 MACs per thread each:
 *   v_new / out   thread = (chunk row t, strip column j) with the warp fixed on one t, so the
 *                 [GC_STRIP] reads of S are one contiguous 128 B window and `kcd`/`qg` are
 *                 broadcasts within the warp.
 *   state         thread = d_k index, all 32 strip columns in registers, `vn` read as a full-block
 *                 broadcast (every thread reads the same address for a given (t, j)).
 * Ragged tail: `o` is written only for rows the body owns. Padded rows contribute nothing to the
 * state because their `kh` is zero (their k~ is zero) and their `vn` is zero.
 */
__global__ void gdn_chunk_carry(GdnChunkArgs a) {
    __shared__ float S[GC_D][GC_STRIP];    // the resident state strip              16.00 KiB
    __shared__ float vn[GC_C][GC_STRIP];   // v_new, all rows, for the two later reads 8.00 KiB
    const int x = blockIdx.x;              // which 32-column strip of d_v
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int jl = tid & (GC_STRIP - 1);   // column inside the strip — DECLARED AND NEVER READ.
                                           // `hipcc -Wall` says so: "unused variable 'jl'". The
                                           // mapping this block's comment claims (thread = row t,
                                           // column j) is not the one coded: `acc[GC_STRIP]` means
                                           // every lane of a row-group computes all 32 columns of
                                           // `v_new` and of `out`, so the executed carry is 24.3x
                                           // the modelled one and the wave executes 730 GFLOP
                                           // against the 54.03 GFLOP Phase 0 priced. Wiring `jl`
                                           // in (scalar accumulator, one column per lane) is the
                                           // fix, and it drops 32x of the LDS instructions with it
                                           // (32 broadcast reads per row today, one would do).
                                           // progress/2026-10-14/p9-phase1c-2026-10-14.md §1c.2.
    const int tg = tid / GC_STRIP;         // row group: 4 groups of 32 threads
    const int tstep = GC_THREADS / GC_STRIP;
    float* Sh = a.S + (size_t)h * GC_D * GC_D;

    // The carried state, read ONCE for the whole body. F32 in here, BF16 only where it is an
    // operand — which is the model's `s_read` asymmetry, not an accident.
    for (int i = tid; i < GC_D; i += GC_THREADS) {
#pragma unroll
        for (int j = 0; j < GC_STRIP; ++j) S[i][j] = Sh[(size_t)i * GC_D + x * GC_STRIP + j];
    }
    __syncthreads();

    for (int c = 0; c < a.nc; ++c) {
        const int base = c * GC_C;
        const int nvalid = a.n - base;
        const float cdec = a.cdec[(size_t)c * GC_NH + h];
        // GCK_CARRY_STALE: re-read the state from global, which still holds the PRE-BODY state for
        // the whole body (the write-back is after this loop), i.e. the inter-chunk carry went stale.
        if (a.neg == GCK_CARRY_STALE && c > 0) {
            for (int i = tid; i < GC_D; i += GC_THREADS) {
#pragma unroll
                for (int j = 0; j < GC_STRIP; ++j) S[i][j] = Sh[(size_t)i * GC_D + x * GC_STRIP + j];
            }
            __syncthreads();
        }
        // GCK_BOUND_HEAD: the chunk boundary taken on the wrong head. Only the state UPDATE reads
        // through it, so no `out` row of any chunk moves and no chunk-0 state moves.
        const size_t hupd = (a.neg == GCK_BOUND_HEAD && c > 0) ? (size_t)((h + 1) % GC_NH) : (size_t)h;

        // ---- v_new[t][j] = nv[t][j] - sum_i BF16(kcd[t][i]) * BF16(S[i][j]) ----------------
        for (int t = tg; t < GC_C; t += tstep) {
            const size_t row = (size_t)(base + t) * GC_NH + h;
            float acc[GC_STRIP];
#pragma unroll
            for (int j = 0; j < GC_STRIP; ++j) acc[j] = 0.f;
            // A padded row has no staging to read: `prep` wrote none, because rows past the body do not
            // exist and a tail chunk of a short body can be mostly padding (`--wave-rows 5` makes every
            // body's tail chunk one). Reading the scratch there would be uninitialized values feeding
            // the state — the same mistake the zeroing of `vn` exists to prevent, one array over.
            if (t < nvalid) {
                for (int i = 0; i < GC_D; ++i) {
                    const float kc = gc_bf(float(a.kcd[row * GC_D + i]));
                    if (kc != 0.f) {
#pragma unroll
                        for (int j = 0; j < GC_STRIP; ++j) acc[j] += kc * gc_bf(S[i][j]);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < GC_STRIP; ++j) {
                vn[t][j] = (t < nvalid ? a.nv[row * GC_D + x * GC_STRIP + j] : 0.f) - acc[j];
            }
        }
        __syncthreads();

        // ---- out[t][j] = sum_i BF16(qg[t][i]) BF16(S[i][j]) + sum_{s<=t} BF16(intra[t][s]) vn[s][j]
        // Two accumulators added at the end, because the model adds two separate matmul results;
        // folding the two loops into one would reorder the F32 chain for no benefit (D1).
        for (int t = tg; t < GC_C; t += tstep) {
            if (t >= nvalid) continue;         // a padded row writes no output row (§04.7), and has no
                                               // `qg`/`intra` row to read — see the v_new loop
            const size_t row = (size_t)(base + t) * GC_NH + h;
            float acc[GC_STRIP];
#pragma unroll
            for (int j = 0; j < GC_STRIP; ++j) acc[j] = 0.f;
            for (int i = 0; i < GC_D; ++i) {
                const float qv = gc_bf(float(a.qg[row * GC_D + i]));
                if (qv != 0.f) {
#pragma unroll
                    for (int j = 0; j < GC_STRIP; ++j) acc[j] += qv * gc_bf(S[i][j]);
                }
            }
            for (int s = 0; s <= t; ++s) {
                const float iv = gc_bf(float(a.intra[row * GC_C + s]));
                if (iv != 0.f) {
#pragma unroll
                    for (int j = 0; j < GC_STRIP; ++j) acc[j] += iv * gc_bf(vn[s][j]);
                }
            }
#pragma unroll
            for (int j = 0; j < GC_STRIP; ++j) {
                a.o[(size_t)(base + t) * GC_VD + (size_t)h * GC_D + x * GC_STRIP + j] = acc[j];
            }
        }
        __syncthreads();

        // ---- S[i][j] = cdec * S[i][j] + sum_t BF16(khat[t][i]) * BF16(vn[t][j]) -------------
        for (int i = tid; i < GC_D; i += GC_THREADS) {
            float acc[GC_STRIP];
#pragma unroll
            for (int j = 0; j < GC_STRIP; ++j) acc[j] = 0.f;
            for (int t = 0; t < GC_C; ++t) {
                if (t >= nvalid) continue;   // no staging there, and `vn[t]` is already zero
                const float kv = gc_bf(float(a.kh[((size_t)(base + t) * GC_NH + hupd) * GC_D + i]));
                if (kv != 0.f) {
#pragma unroll
                    for (int j = 0; j < GC_STRIP; ++j) acc[j] += kv * gc_bf(vn[t][j]);
                }
            }
#pragma unroll
            for (int j = 0; j < GC_STRIP; ++j) S[i][j] = cdec * S[i][j] + acc[j];
        }
        __syncthreads();
    }

    // One write-back per body — the traffic this whole shape buys.
    for (int i = tid; i < GC_D; i += GC_THREADS) {
#pragma unroll
        for (int j = 0; j < GC_STRIP; ++j) Sh[(size_t)i * GC_D + x * GC_STRIP + j] = S[i][j];
    }
}
