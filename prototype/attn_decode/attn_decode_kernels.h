/*
 * prototype/attn_decode/attn_decode_kernels.h — the device code of attn_decode: S05 gated full-attention decode (k=1): de-interleave, per-head norm, partial rope, cache append, GQA attention, output gate.
 *
 * Moved verbatim out of attn_decode.hip.cpp (host main() and the gate's file I/O stayed
 * there) so that attn_decode and `engine/decode_step` compile THE SAME KERNEL SOURCE. An
 * engine TU includes exactly one of these op headers: the ops each spell their own
 * helpers (bf/fb/rnd/block_sum/reduce_row) and define colliding macros (D, H, HID,
 * EPS, SCALE, FULL_MASK), which is fine while they are in separate translation
 * units and is a compile error if they are merged into one.
 *
 * Gate for this file: prototype/attn_decode/ (its run.py/check.py drives the binary in
 * the same directory, which includes this header). Nothing here may be edited for
 * the engine's convenience — see the rule in prototype/README.md.
 */
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>

#define FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit
#define N_Q 8
#define N_KV 2
#define GQA (N_Q / N_KV)     // 4 query heads per KV head; kappa(i) = i/4 (§05.5)
#define DH 256               // head_dim
#define ROT 64               // d_rot — only the first ROT dims of a head rotate (§05.3)
#define HR (ROT / 2)         // rope pairs are (i, i+32)
#define THREADS 256          // one thread per head dim
#define MAXW 8               // THREADS/32
#define TILE 8               // cache lanes per online-softmax tile: one per warp
#define MAXSC 512            // shared score staging for --softmax full, lanes per block range
#define EPS 1e-6f            // §00.2
#define SCALE 0.0625f        // 1/sqrt(256)
#define PROBE_GBS 237.0      // prototype/lm_head_gemv's read ceiling, for the sanity check only

enum Neg {
    NEG_NONE = 0,
    NEG_QSWAP,     // q_proj halves swapped (§05.1)
    NEG_GATEHALF,  // gate read contiguously (§05.1)
    NEG_ROPEFULL,  // rope over all 256 dims (§05.3)
    NEG_ROPENONE,  // no rope
    NEG_GQA,       // kappa off by one (§05.5)
    NEG_NOMAX,     // online softmax without (max,sum) rescaling (A-17)
    NEG_CARRY,     // softmax state carried from the previous query (§07.3)
    NEG_READPAST,  // ignore lvalid: the causal mask is gone
    NEG_DROPCUR,   // ignore the appended lane
    NEG_DIMHALF,   // half of head_dim in the score dot
    NEG_SLOTK,     // wrong cache slot for keys (§05.7)
    NEG_SLOTV,     // wrong cache slot for values (§05.7)
};

__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }
// The rounding the reference does at every documented boundary: keep computing in F32, but round
// through BF16. Used instead of bf(fb(x)) so the rounding policy reads as one named operation.
__device__ __forceinline__ float rbf(float x) { return float(hip_bfloat16(x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = warpSize / 2; o; o >>= 1) v += __shfl_down_sync(FULL_MASK, v, o);
    return v;
}

// Block reduction of a per-thread partial; every thread leaves with the same F32 total.
__device__ __forceinline__ float block_sum(float v, float* shred, int nw) {
    v = warp_sum(v);
    if (threadIdx.x % warpSize == 0) shred[threadIdx.x / warpSize] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = shred[0];
        for (int i = 1; i < nw; ++i) t += shred[i];
        shred[0] = t;
    }
    __syncthreads();
    const float out = shred[0];
    __syncthreads();
    return out;
}

// §05.3's cos/sin: F32 from the position, rounded to BF16 (what the pins say; see the header).
__device__ __forceinline__ void rope_cs(float pos, int fd, float& c, float& s) {
    const float inv = 1.0f / powf(1e7f, (2.0f * (float)fd) / (float)ROT);
    const float ang = pos * inv;
    c = bf(fb(cosf(ang)));
    s = bf(fb(sinf(ang)));
}

/*
 * §05.2 + §05.3 for one head, one block of THREADS threads (one per dim). `src` is the head's 256
 * pre-norm values, `w` the zero-centred RMSNorm weight; `out` (shared) receives the head as
 * BF16-valued F32. Rope needs the partner dim, so the normed values are staged in `zs` first.
 */
__device__ void head_norm_rope(const hip_bfloat16* __restrict__ src,
                               const hip_bfloat16* __restrict__ w, float pos, int neg,
                               float* out, float* zs, float* shred, int nw) {
    const int d = threadIdx.x;
    const float x = bf(src[d]);
    const float r = rsqrtf(block_sum(x * x, shred, nw) * (1.0f / DH) + EPS);
    const float z = rbf((x * r) * (1.0f + bf(w[d])));   // one BF16 round, §05.2
    if (neg == NEG_ROPENONE) { out[d] = z; return; }
    zs[d] = z;
    __syncthreads();
    int part, fd, lowhalf;
    if (neg == NEG_ROPEFULL) {           // rotate_half over the whole head: pairs (i, i+128)
        lowhalf = DH / 2;
        part = (d < lowhalf) ? d + lowhalf : d - lowhalf;
        fd = d % HR;
    } else if (d < ROT) {                // §05.3: pairs (i, i+32) inside the first 64 dims
        lowhalf = HR;
        part = (d < HR) ? d + HR : d - HR;
        fd = d % HR;
    } else {
        out[d] = z;                      // the 192 unrotated dims pass through unchanged
        __syncthreads();
        return;
    }
    float c, s;
    rope_cs(pos, fd, c, s);
    const float rh = (d < lowhalf) ? -zs[part] : zs[part];
    out[d] = rbf(rbf(z * c) + rbf(rh * s));
    __syncthreads();
}

/* §05.1: where a head's q half and gate half live inside the 4096-row q_proj output. */
__device__ __forceinline__ size_t q_off(int h, int neg) {
    if (neg == NEG_QSWAP) return (size_t)h * (2 * DH) + DH;
    return (size_t)h * (2 * DH);
}
__device__ __forceinline__ size_t g_off(int h, int neg) {
    if (neg == NEG_QSWAP) return (size_t)h * (2 * DH);
    if (neg == NEG_GATEHALF)           // the wrong-but-plausible [all q | all gate] reading
        return (size_t)(N_Q * DH) + (size_t)h * DH;
    return (size_t)h * (2 * DH) + DH;
}
__device__ __forceinline__ int kv_of(int head, int neg) {
    // The mapping the ATTENTION reads through. The cache append keeps using the honest one so a
    // wrong kappa moves the attention output only, not the cache too.
    return (neg == NEG_GQA) ? ((head / GQA + 1) % N_KV) : (head / GQA);
}
__device__ __forceinline__ int kv_write_of(int head) { return head / GQA; }
__device__ __forceinline__ int slot_of(int pos, int cap, int neg, int which) {
    // §05.7 writes lane `pos`; the injections land on the next slot (in-bounds, still wrong).
    if (which == 0 && neg == NEG_SLOTK) return (pos + 1) % cap;
    if (which == 1 && neg == NEG_SLOTV) return (pos + 1) % cap;
    return pos;
}

struct CoreArgs {
    hip_bfloat16 *qg, *kg, *vg, *kcache, *vcache, *qn, *kn, *qrope, *krope;
    hip_bfloat16 *aout, *gout, *qrope_out, *krope_out, *gate_out, *kc_as, *vc_as, *rope_out;
    float *part, *stale;
    int cap, pos, lvalid, split, neg, full, score_f32, fused, tile;
    // P8 (§06.3): how many query rows one launch serves, and the POSITION OF ROW 0. Row `qrow` sits at
    // `pos + qrow` and may attend `lvalid + qrow` lanes; every operand the op reads or writes is
    // addressed as row `qrow` of its own layout (see the ROW_* strides). `nrows = 1` — which is what a
    // zero-initialised struct gives, and what every pre-P8 caller passes — is exactly the decode shape,
    // same blocks, same addresses, same bits.
    int nrows;
    // Phase 2: which wave form a launch is. 0 (what a zero-initialised struct gives, so every
    // pre-Phase-2 caller and every one-row decode step) is the shipped per-(row,head) `attn_core`;
    // 1 is `attn_tile_wave`, the G x M tiled form. `attn_combine` reads it too, because the two forms
    // lay their split partials out differently and a merge that guessed would read garbage.
    int tile_kernel;
};

// The op's own row layouts (P8, §06.3). A wave hands these kernels one buffer per stage with one row
// per query token, and the stride of every one of them is a constant of THIS op — q_proj's row is
// [8 heads][q 256 | gate 256], k/v_proj's row is [2][256], and q_rope/k_rope/gate/attn_out/gated are
// rows of the head block they were staged from. Deriving them here instead of taking them as arguments
// is the point: a stride that arrives from the caller can disagree with the kernel that indexes it,
// and the engine's scratch (`decode_step.h`) is laid out to these same numbers.
constexpr size_t ROW_Q = (size_t)N_Q * (2 * DH);
constexpr size_t ROW_KV = (size_t)N_KV * DH;
constexpr size_t ROW_H = (size_t)N_Q * DH;

// How many lane blocks a row of `lanes` lanes is cut into. `__device__ __host__` so the engine's
// launcher and this op's host harness price a split with ONE expression — `engine/kernels_attn.cpp`
// sizes and grid-depths a launch with it, and a second spelling of the arithmetic is a second
// authority that can drift.
__device__ __host__ inline int split_for_lanes(int lanes) { return (lanes + MAXSC - 1) / MAXSC; }

// `__device__ __host__` because the Phase 2 tiled form needs the same default the launchers use:
// `nrows = 0` is a zero-initialised struct, and both sides must read it as one row.
__device__ __host__ inline int nrows_of(const CoreArgs& a) { return a.nrows > 0 ? a.nrows : 1; }

// The lanes query row `qrow` may attend to: §06.3's causal mask for THAT row (the wave's rows advance
// in position order), plus the two ways an engine gets the mask wrong, kept as injections.
//
__device__ __forceinline__ int lanes_of(const CoreArgs& a, int qrow) {
    if (a.neg == NEG_READPAST) return a.cap;
    if (a.neg == NEG_DROPCUR) return a.pos + qrow;
    return a.lvalid + qrow;
}

// The blocks row `qrow` actually uses, out of the launch's `split`. A row the one-pass form can serve
// gets exactly one block and the whole row (§07.3's decision — it is the form that is bit-exact with
// the pins, and a per-block weight matrix normalised inside its own range cannot be re-merged into one
// row, which is why the split form is only ever reached by being forced); the blocks above it in that
// row's slice of the grid go idle. Any other row streams, across however many blocks the LAUNCH was
// given — `split` is the caller's pricing knob (`run.py --split 4` exists to price a split, and
// `engine/kernels_attn.cpp` picks it from the widest row), so honouring it per row rather than
// re-deriving it is what keeps a priced run the run that was priced.
//
// That is also what makes one wave able to mix one-pass rows under the staging with split rows above
// it: the decision is per row, and the grid is deep enough for the row that needs the most.
__device__ __forceinline__ int blocks_for_row(const CoreArgs& a, int qrow) {
    if (a.full && lanes_of(a, qrow) <= MAXSC) return 1;
    return a.split > 0 ? a.split : 1;
}

/*
 * Stage outputs: q_rope, k_rope, the gate vector and the §05.7 append. The staged plan does this in
 * `attn_prep`; the fused plan does it in each core block (per query head, and per KV head for the
 * append), which is what lets the engine do the whole attention layer in one launch.
 */
__device__ void stage_head(const CoreArgs& a, int head, int qrow, int row_pos, const float* sq,
                           const float* sk, const float* sv) {
    if (a.qrope_out == nullptr || blockIdx.y != 0) return;
    const int d = threadIdx.x, kv = kv_write_of(head);
    a.qrope_out[ROW_H * qrow + (size_t)head * DH + d] = fb(sq[d]);
    a.gate_out[ROW_H * qrow + (size_t)head * DH + d] = a.qg[ROW_Q * qrow + g_off(head, a.neg) + d];
    if (head % GQA == 0) {          // one block per KV head owns that head's append
        a.krope_out[ROW_KV * qrow + (size_t)kv * DH + d] = fb(sk[d]);
        a.kc_as[((size_t)kv * a.cap + slot_of(row_pos, a.cap, a.neg, 0)) * DH + d] = fb(sk[d]);
        a.vc_as[((size_t)kv * a.cap + slot_of(row_pos, a.cap, a.neg, 1)) * DH + d] = fb(sv[d]);
    }
}

/* §05.6 epilogue: normalise, round to BF16, gate with plain sigmoid. */
__device__ void finish_head(const CoreArgs& a, int head, int qrow, float l, float acc, int d) {
    const float av = bf(fb(acc / l));
    a.aout[ROW_H * qrow + (size_t)head * DH + d] = fb(av);
    const float g = bf(a.qg[ROW_Q * qrow + g_off(head, a.neg) + d]);
    a.gout[ROW_H * qrow + (size_t)head * DH + d] = fb(av * bf(fb(1.0f / (1.0f + expf(-g)))));
}

__global__ void attn_core(CoreArgs a) {
    __shared__ float sq[DH], sk[DH], sv[DH], zs[DH], red[MAXW];
    __shared__ float sc[(MAXSC < TILE) ? TILE : MAXSC];
    const int head = blockIdx.x;
    // P8: `blockIdx.z` selects the query ROW, `blockIdx.y` the lane block within that row. Before P8
    // there was one row, so `gridDim.z == 1`, `qrow` is 0 and every `* qrow` below vanishes.
    const int qrow = blockIdx.z;
    const int row_pos = a.pos + qrow;              // §06.3: row qrow sits at position pos+qrow
    const int nrow = lanes_of(a, qrow);
    const int s_row = blocks_for_row(a, qrow);    // of a.split lane blocks, how many this row uses
    if (blockIdx.y >= s_row) return;              // uniform in the block: no barrier has run yet
    const int kv = kv_of(head, a.neg);
    const int d = threadIdx.x;
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;
    const float pos = (float)row_pos;

    if (a.fused) {          // §05.2 + §05.3 + §05.7, recomputed per block instead of staged
        head_norm_rope(a.qg + ROW_Q * qrow + q_off(head, a.neg), a.qn, pos, a.neg, sq, zs, red, nw);
        head_norm_rope(a.kg + ROW_KV * qrow + (size_t)kv * DH, a.kn, pos, a.neg, sk, zs, red, nw);
        for (int i = d; i < DH; i += blockDim.x)
            sv[i] = bf(a.vg[ROW_KV * qrow + (size_t)kv * DH + i]);
        __syncthreads();
    } else {                // staged: prep already produced q_rope/k_rope and appended
        for (int i = d; i < DH; i += blockDim.x) {
            sq[i] = bf(a.qrope[ROW_H * qrow + (size_t)head * DH + i]);
            sk[i] = bf(a.krope[ROW_KV * qrow + (size_t)kv * DH + i]);
            sv[i] = bf(a.vg[ROW_KV * qrow + (size_t)kv * DH + i]);
        }
        __syncthreads();
    }
    stage_head(a, head, qrow, row_pos, sq, sk, sv);

    // q in registers: lane owns dims lane*8..lane*8+7 — the same 512-byte read a K row needs.
    float qreg[8];
    for (int t = 0; t < 8; ++t) qreg[t] = sq[lane * 8 + t];

    const int span = (nrow + s_row - 1) / s_row;
    const int blo = blockIdx.y * span;
    const int bhi = std::min(nrow, blo + span);
    const int nd = (a.neg == NEG_DIMHALF) ? 4 : 8;      // of the 8 BF16 a lane owns
    const int tile = a.tile;

    // Softmax state. Neutral is (m=0, l=0, acc=0): the first tile's rescale multiplies a zero
    // accumulator and a zero sum, so a finite seed is harmless and avoids -inf bookkeeping.
    float m = 0.f, l = 0.f, acc = 0.f;
    if (a.neg == NEG_CARRY && blockIdx.y == 0) {        // §07.3's own bug: a state that outlived
        m = a.stale[head * 2];                          // the query it belonged to
        l = a.stale[head * 2 + 1];
        acc = a.stale[N_Q * 2 + (size_t)head * DH + d];
    }

    // ---- pass 1: the score tile(s). Warp w scores lane base+w with a full 512-byte coalesced
    // read; the appended lane comes from shared, so the fused plan has no intra-kernel dependency
    // on its own cache write.
    if (s_row == 1 && a.full) {
        int ns = 0;
        for (int base = blo; base < bhi; base += tile) {
            const int nt = std::min(tile, bhi - base);
            __syncthreads();
            if (w < nt) {
                const int j = base + w;
                float dot = 0.f;
                if (j == row_pos) {
                    for (int t = 0; t < nd; ++t) dot += sq[lane * 8 + t] * sk[lane * 8 + t];
                } else {
                    const hip_bfloat16* kr = a.kcache + ((size_t)kv * a.cap + (size_t)j) * DH;
                    uint4 u = *reinterpret_cast<const uint4*>(kr + lane * 8);
                    const hip_bfloat16* k = reinterpret_cast<const hip_bfloat16*>(&u);
                    for (int t = 0; t < nd; ++t) dot += bf(k[t]) * qreg[t];
                }
                dot = warp_sum(dot);
                if (lane == 0) sc[ns + w] = a.score_f32 ? dot * SCALE : bf(fb(dot)) * SCALE;
            }
            ns += nt;
            __syncthreads();
        }
        // ---- pass 2 (reference form): max, then sum, then BF16 weights, then P·V.
        for (int t = 0; t < ns; ++t) m = fmaxf(m, sc[t]);
        for (int t = 0; t < ns; ++t) l += expf(sc[t] - m);
        for (int t = 0; t < ns; ++t) {
            const int j = blo + t;
            const float wt = bf(fb(expf(sc[t] - m) / l));
            const float v = (j == row_pos)
                                ? sv[d]
                                : bf(a.vcache[((size_t)kv * a.cap + (size_t)j) * DH + d]);
            acc += wt * v;
        }
        // The weights were already normalised and rounded (that IS the reference form: HF rounds
        // the softmax output to the activation dtype before P·V), so the epilogue must not divide
        // again. Hence the split restriction below: a per-block weight matrix normalised inside its
        // own range cannot be re-merged into one row.
        l = 1.0f;
    } else {
        for (int base = blo; base < bhi; base += tile) {
            const int nt = std::min(tile, bhi - base);
            __syncthreads();
            if (w < nt) {
                const int j = base + w;
                float dot = 0.f;
                if (j == row_pos) {
                    for (int t = 0; t < nd; ++t) dot += sq[lane * 8 + t] * sk[lane * 8 + t];
                } else {
                    const hip_bfloat16* kr = a.kcache + ((size_t)kv * a.cap + (size_t)j) * DH;
                    uint4 u = *reinterpret_cast<const uint4*>(kr + lane * 8);
                    const hip_bfloat16* k = reinterpret_cast<const hip_bfloat16*>(&u);
                    for (int t = 0; t < nd; ++t) dot += bf(k[t]) * qreg[t];
                }
                dot = warp_sum(dot);
                if (lane == 0) sc[w] = a.score_f32 ? dot * SCALE : bf(fb(dot)) * SCALE;
            }
            __syncthreads();
            // ---- (max,sum) update, computed identically by every thread so the block stays in
            // lockstep. This state belongs to THIS query row and dies with it (§07.3, A-17).
            float mt = -1e30f;
            for (int t = 0; t < nt; ++t) mt = fmaxf(mt, sc[t]);
            const float mn = (a.neg == NEG_NOMAX) ? mt : fmaxf(m, mt);
            const float rescale = (a.neg == NEG_NOMAX) ? 1.0f : expf(m - mn);
            float e[TILE], ls = 0.f;
            for (int t = 0; t < nt; ++t) { e[t] = expf(sc[t] - mn); ls += e[t]; }
            l = l * rescale + ls;
            acc = acc * rescale;
            for (int t = 0; t < nt; ++t) {
                const int j = base + t;
                const float v = (j == row_pos)
                                    ? sv[d]
                                    : bf(a.vcache[((size_t)kv * a.cap + (size_t)j) * DH + d]);
                acc += e[t] * v;
            }
            m = mn;
        }
    }

    if (s_row == 1) {
        finish_head(a, head, qrow, l, acc, d);
        if (a.neg == NEG_CARRY && blockIdx.y == 0) {    // leave the state the next query must NOT
            a.stale[head * 2] = m;
            a.stale[head * 2 + 1] = l;
            a.stale[N_Q * 2 + (size_t)head * DH + d] = acc;
        }
    } else {
        // Stride by the LAUNCH's split, not the row's: every row of a wave owns a contiguous
        // `N_Q * split` slab, which is what makes `part` allocatable from one expression
        // (`nrows * N_Q * split * (DH + 2)`, see `attn_part_floats()` in the engine launcher). A row
        // that used fewer blocks simply leaves the tail of its slab unwritten.
        float* p = a.part + (((size_t)qrow * N_Q + head) * a.split + blockIdx.y) * (DH + 2);
        p[0] = m;
        p[1] = l;
        p[2 + d] = acc;
    }
}

// ============================================================================================
// Phase 2 (bench/attn-wave-diagnosis-2026-10-14.md §4/§5/§9): the G x M tiled wave form.
//
// WHY THIS EXISTS. The shipped `attn_core` reads exactly 1.0 byte of K/V per flop: one K row and
// one V row per (query row, query head, KV lane) touch, so a 1024-token wave reads 25.79 GB of K/V
// to do 25.79 GFLOP, and at this box's measured 237 GB/s read ceiling that byte count alone prices
// the wave at 108.8 ms against 109.2 ms measured. §9's counter pass settled which candidate wall
// was binding (`SQ_INSTS_VALU` = 6.53 G warp-instructions = a ~20 ms issue floor, 19 % of the
// line), so the fix is traffic, and traffic falls only by RE-USING one K/V row across more than one
// touch: `G` = the GQA=4 query heads that share a KV head, `M` = query rows. Putting one K row in LDS
// once and feeding G x M = 16 score dots off it is the whole lever; what THIS version books is M = 4 of
// the 16, and the bullet below prices why, names the ~15 lines that close the rest, and says whether the
// win line depends on it.
//
// BIT-IDENTITY (§5's contract). This form preserves the five ordered reductions verbatim; it
// changes WHERE a K row comes from (LDS, instead of one global load per (row, head) pair) and not any
// summation order:
//   1. the score dot is 8 serial `dot += bf(k[t]) * q[t]` per thread, then the 5-step
//      `__shfl_down_sync` butterfly, then ONE BF16 round, then `x SCALE` — same threads-per-dot,
//      same per-thread chunk, same butterfly, same rounding site as `attn_core`;
//   2. `m` is `fmaxf` over the row's scores in index order (max is exactly associative, so it could
//      not move the bits even as a tree; it is still walked in index order here);
//   3. `l` is `+= expf(sc[t] - m)` accumulated serially in index order by one thread — no tree, no
//      tile partial sums, no rescaling;
//   4. `wt = bf(fb(expf(sc[t] - m) / l))` — the BF16 round of the NORMALISED weight, kept because
//      that round IS §05.5's reference form, and the epilogue still does not divide again;
//   5. `acc += wt * v` serial in `t`.
// For a row wider than the staging (§07.3's pre-authorized streaming route) pass 2 reproduces
// `attn_core`'s tile-of-`a.tile` loop verbatim, including the `expf(m - mn)` rescale algebra, so
// the streaming route here lands on the bits the shipped form already streams — it does not open a
// second band. And a row the shipped form serves one-pass is served one-pass here too (the window
// rule below), so no row changes route between the two forms: the tiled form's claim is bit-for-bit
// equality with the shipped form on BOTH routes, which is what `--tile-kernel` gates.
//
// THE GEOMETRY. block = (one KV group of G=4 query heads) x (M consecutive query rows) x (one lane
// window of <= MAXSC lanes); grid = dim3(N_KV, split_for_lanes(lanes_last), ceil(nrows/M)).
//   * The window is MAXSC lanes — the number `attn_core` stages — and that is not a tuning choice,
//     it is the bit-identity boundary: a row that fits a window is one-pass, a row wider than MAXSC
//     is the streaming form §07.3 already bands. Shrinking the window below MAXSC would push rows
//     the shipped form computes exactly INTO that band, which is a band decision for the parent and
//     not something to slide into a kernel. It is also why `M` cannot be 16: the staging is
//     G x M x MAXSC scores, so M=4 is 16 KiB, M=8 is 32 KiB and M=16 would be 64 KiB on a CU that
//     has 64 KiB in total. §5's budget arithmetic, halved by the exact-BF16 staging below.
//   * The causal mask is a PREFIX in every row (row `qrow` may attend lanes [0, lvalid+qrow)), so a
//     pair's score list is never hole-punched: the window is used up to a per-pair limit `lim` and
//     lanes past it are not read, not scored, not loaded. That is what makes tiling rows legal, and
//     it is why `NEG_READPAST`/`NEG_DROPCUR` still move exactly the rows that gained or lost a lane:
//     `lim` comes from `lanes_of` and from nowhere else.
//   * Empty is identity. A pair whose row cannot attend any lane in this window emits NOTHING: no
//     score, no max, no sum, no partial. The streaming route's neutral state is (m=0, l=0, acc=0)
//     and `attn_combine`'s merge of one partial is then the identity (`mn = fmaxf(0, m_s)`,
//     `r0 = expf(0 - mn)`, and `l`/`acc` are 0 so they contribute nothing). A neutral of m=-1e30
//     would NOT be an identity — `expf(-1e30 - m_s)` is 0 and would erase a row — so emitting
//     nothing for an empty pair is load-bearing, not cosmetic.
//   * THE K STAGE IS `L` SLOTS DEEP, SO THE SLOT INDEX IS SUB-BLOCK-RELATIVE (`rr - s_lo`), NOT THE
//     WINDOW-ABSOLUTE LANE `rr`. This is the one index in the kernel that a reading of the algorithm
//     cannot get wrong and a reading of the loop can: `s_k` holds `ATTN_TILE_L` rows and the sub-block
//     loop steps `s_lo` by `ATTN_TILE_L`, so indexing it by `rr` walks `(span-1)*DH*2` bytes for a
//     tile whose widest row spans `span` lanes — 141 KiB for a 277-lane window and 256 KiB for a
//     MAXSC-wide one, against a kernel that reserves 20,480 B of LDS in total, and it does that for
//     every sub-block after the first, i.e. for every row above 8 lanes. Out-of-range LDS does not
//     fault on this part; the writes land on the score staging next door and the scores that survive
//     are K bytes. That is what §13 read as "1,098,790 elements moved" and §14 identifies: a silent
//     numerics difference produced by an address, which is why `attn_tile_model.py` now prices the
//     stage's LDS reach from this loop's own expression instead of trusting the declared size.
//   * Pass 1 stages K, pass 2 reads V from global, and BOTH do it per PAIR — so the re-use this kernel
//     actually books is M = 4x, not G x M = 16x. Precisely: the four heads of a group read the
//     identical rows and write them into the identical LDS slots with the identical bytes (benign, and
//     probably cheap after the first, since a pair's rows are staged and consumed adjacently so L1
//     likely serves three of the four) — but an L1 hit is not something a traffic claim may rest on, so
//     `attn_tile_model.py` books all four and the pre-registered numbers are the pessimistic ones. The
//     lever that closes the gap is block-level staging: stage the group's rows once per sub-block and let
//     a pair substitute only its OWN appended lane (`j == a.pos + row`, the single lane on which pairs
//     differ, because it is that row's §05.7 append) from `krope`/`vg` at the use site. That is ~15 lines
//     in pass 1 plus a sub-block loop in pass 2, and it takes the predicted line from ~1 GB to ~0.4 GB.
//     Run it only if the device says bytes still bind: at M = 4 the traffic term is already ~4x under the
//     arithmetic term, so §7's 40 ms win line does not depend on it. Staging a whole window of V is not
//     an option at any M: 512 lanes x 512 B = 256 KiB. `L` sizes only the K stage, so it never changes
//     the traffic — it is the cheap knob if LDS turns out to bind.
//   * WHERE BIT-IDENTITY STOPS, at exactly one place: a row WIDER than MAXSC. The shipped form cuts such
//     a row into `split` EQUAL slices (`span = ceil(n/split)`, per row); this form owns fixed MAXSC-wide
//     windows, because a window is what a block shares a K load over. Both are §07.3's streaming form and
//     both sit inside its pre-authorized band, but the two groupings are not bit-identical to EACH OTHER
//     — different partial boundaries are a different F32 sum. So the bit-for-bit claim is exact for every
//     row the one-pass route serves (lanes <= MAXSC) and is a band claim above it. No pinned fixture here
//     reaches 512 lanes — p300 tops out at 277 — so neither the CPU mirror nor `run.py --wave` can
//     exercise it; the engine's 1024-token body 1 (lanes 513..1024) is the first place it exists. If an
//     engine gate step moves there, that is THIS, it is inside §07.3's band by construction, and it is a
//     band question for the parent rather than a bug to chase. `attn_tile_model.py` models the two
//     groupings separately for exactly this reason, so the mirror cannot quietly agree with itself.
//   * Work is handed out PAIR-FIRST (warp w takes pairs w, w+nw, ...) and a pair owns its K rows.
//     The shipped form is lane-first (warp w scores lane base+w), which cannot survive M>1: with
//     lane-first order a warp can only reach one row's K rows, so the loads would stay G x M-fold.
//     Two side benefits worth naming: a warp holds exactly ONE pair's query (8 registers) instead
//     of the G or G x M rows §5 priced at ~+32 VGPRs, and a pair's whole softmax state is private
//     to one warp, so the passes need one barrier between them instead of `attn_core`'s two per
//     tile. Every barrier is in a block-uniform loop; there is no barrier inside a pair loop.
//   * The re-use does not make the bytes vanish, it moves them from the DRAM path (237 GB/s) to the
//     LDS path: the ~1 GB of K this form asks global for is about 4 ms of DRAM against under 1 ms of
//     LDS at a conservative 32 B/clk/CU. §7's falsifiers are about that exchange.
//
// WHAT IT COSTS, priced not discovered. Compile-only census (`attn_tile_model.py isa`, which is
// `hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage -c` on the header):
//   attn_tile_wave  49 VGPR, 0 spills, 0 scratch, 16 waves/SIMD, LDS 20,480 B/block
//   attn_core       53 VGPR, 0 spills, 0 scratch, 16 waves/SIMD, LDS 38,944 B/block
//   * registers came in CHEAPER than §5 priced them: 49, not 53+32. The +32 was for holding G (or
//     G x M) query rows in registers; pair-first means a warp only ever holds ONE pair's row (8
//     floats), so register occupancy is unchanged at 16 waves/SIMD and registers are not the limit.
//   * LDS is 16 KiB of score staging (G x M x MAXSC BF16 scores) + L x DH x 2 = 4 KiB of K at L=8,
//     and the descriptor reserves exactly that 20,480 B — 3 blocks/CU, against 1 for `attn_core` if
//     §3's 38,944 B reservation really caps residency. Note the asymmetry, which is new evidence for
//     §3: this kernel's group segment equals the sum of its `__shared__` arrays exactly, while
//     `attn_core`'s in the same TU is 6.3x that sum, so the inflation is kernel-specific and not a
//     code-object or toolchain property. At M=2 the staging halves to 8 KiB (12 KiB total, 4
//     blocks/CU = thread-capped) for half the re-use; at M=8 it is 36 KiB and one block fits.
//     `kTileLdsBytes` is the number to look at before turning a knob.
//   * blocks: 512 rows x 2 KV groups / (4 rows x 4 heads) = 256 core blocks per body per layer at
//     body 0 against 4096 shipped. Fewer, fatter blocks IS the re-use, but it makes §2's
//     dispatch-order note live, so `blockIdx.z` is inverted below and the widest row tiles launch
//     first instead of trailing the queue.
//   * what it does NOT share: §05.2's norm and §05.3's rope are read per (row, head) pair from the
//     staged `q_rope` rather than shared across the group, because sharing them would need a barrier
//     inside the pair loop — the barriers the pair-first order exists to avoid. Being staged-plan
//     only, this kernel does no norm/rope work at all: `attn_prep` already did it once per row-head,
//     which is also why the tile form cannot be `fused` (see `attn_tile_ok`).
//
// THE EXACT-BF16 SCORE STAGING. §05.5's score is `bf(fb(dot)) * SCALE` and SCALE = 1/16, and
// multiplying a BF16 value by a power of two is EXACT (mantissa unchanged, exponent -4), so the
// score is itself a BF16 value: it stages as 2 bytes and reads back as the identical F32. That
// halves §5's staging budget and is why M=4 fits at all. The limit of the claim is the subnormal
// boundary — a score below ~1e-37 would lose bits, which no normalised q.k of this model produces —
// and `--score-f32` (a policy variant, not the gated route) is NOT expressible in this staging, so
// both launchers refuse that combination rather than silently rounding it.
// ============================================================================================

#define ATTN_TILE_G GQA                 // 4 query heads of one KV group per block
#ifndef ATTN_TILE_M
#define ATTN_TILE_M 4                   // query rows per block; staging = G*M*MAXSC scores
#endif
#ifndef ATTN_TILE_L
#define ATTN_TILE_L 8                   // K lanes resident at once (LDS knob, traffic-neutral)
#endif
#ifndef ATTN_TILE_THREADS
#define ATTN_TILE_THREADS 256
#endif
constexpr int kTileThreads = ATTN_TILE_THREADS;
constexpr int kTilePairs = ATTN_TILE_G * ATTN_TILE_M;    // pairs one block can hold
constexpr int kTileScoreElems = kTilePairs * MAXSC;      // BF16 scores staged
constexpr int kTileLdsBytes = (kTileScoreElems + ATTN_TILE_L * DH) * 2;
static_assert(ATTN_TILE_M >= 1, "M is the row tile");
static_assert(ATTN_TILE_L >= 1, "L is the K stage depth");
static_assert(kTileScoreElems <= 32768, "score staging past 64 KiB cannot be resident on this CU");

// One MAXSC-lane window per lane block — the SAME expression `attn_core`'s split uses, so the
// one-pass/streaming boundary cannot drift between the two forms (the bit-identity argument rests
// on it). `__device__ __host__` for the house reason: launcher and op price one number once.
__device__ __host__ inline int tile_split_of(int lanes_last) { return split_for_lanes(lanes_last); }
__device__ __host__ inline int tile_rtiles_of(int nrows) { return (nrows + ATTN_TILE_M - 1) / ATTN_TILE_M; }
// This form's `part` is [row][head in group][lane window][2 + DH] F32 — half the shipped form's
// footprint at the same lane ceiling, because the 4 heads of a group are the block, not the grid.
__device__ __host__ inline int tile_part_floats(int nrows, int lanes_last) {
    return nrows * ATTN_TILE_G * tile_split_of(lanes_last) * (DH + 2);
}

__device__ __forceinline__ int tile_kv_of(int kv, int neg) {
    // The group whose K/V this block READS. The §05.7 append (in `attn_prep`) keeps using the honest
    // group, so a wrong kappa moves the attention output only and not the cache too — the same
    // split `kv_of` draws for the shipped form. Here all four heads of a row tile move together,
    // because sharing the group is the whole point of the block; that is what makes `NEG_GQA` the
    // control for "did the shared K/V load get wired to the right group".
    return (neg == NEG_GQA) ? ((kv + 1) % N_KV) : kv;
}

// How many lane windows row `qrow` needs here. Same rule as `blocks_for_row` — one window iff the
// row fits the staging, whatever the grid's depth, which is what lets one launch mix exact rows with
// streaming ones — clamped by the windows the LAUNCH actually has, so a row can never be told to
// write partials no block will produce. The clamp is inert unless a negative widens `lanes_of` past
// the geometry the launcher priced (`NEG_READPAST` does, on purpose).
__device__ __forceinline__ int tile_blocks_for_row(const CoreArgs& a, int qrow) {
    const int n = lanes_of(a, qrow);
    if (a.full && n <= MAXSC) return 1;
    const int want = (n + MAXSC - 1) / MAXSC;
    const int have = a.split > 0 ? a.split : 1;
    return want < have ? want : have;
}

// How many lanes of THIS window row `row` may attend: the causal prefix, clipped to the window. A
// limit rather than a mask, which is what §05.5's mask being a prefix buys the tiling.
__device__ __forceinline__ int tile_lim(const CoreArgs& a, int row, int blo, int span) {
    return std::min(span, lanes_of(a, row) - blo);
}

// What this form may not be asked for, in one place so both launchers refuse the same list.
inline bool attn_tile_ok(int score_f32, int fused, int warp_size, const char** why) {
    if (warp_size != 32) { *why = "one warp covers a 256-wide head as 32 lanes x 8 dims"; return false; }
    if (score_f32) { *why = "the score staging is BF16, exact only for the pinned BF16 score"; return false; }
    if (fused) { *why = "a wave's §05.7 append must finish before any row reads the cache"; return false; }
    return true;
}

__device__ __forceinline__ void attn_tile_body(const CoreArgs a) {
    __shared__ hip_bfloat16 s_sc[kTilePairs][MAXSC];       // exact-BF16 score staging
    __shared__ hip_bfloat16 s_k[ATTN_TILE_L * DH];         // K rows, L lanes deep
    const int y = blockIdx.y;
    const int z = gridDim.z - 1 - blockIdx.z;   // widest row tiles first: §2's dispatch tail, priced
    const int kv_read = tile_kv_of(blockIdx.x, a.neg);     // group whose K/V this block reads
    const int kv_write = blockIdx.x;                       // group whose lanes this wave appends
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;
    const int nrows = nrows_of(a);
    const int r_lo = z * ATTN_TILE_M;
    const int r_hi = std::min(nrows, r_lo + ATTN_TILE_M);
    const int pairs = ATTN_TILE_G * (r_hi - r_lo);
    const int blo = y * MAXSC;
    const int windows = tile_split_of(a.lvalid + nrows - 1);
    if (y >= windows || r_lo >= nrows) return;             // before any barrier: uniform exit
    const int tile = a.tile > 0 ? a.tile : TILE;           // the streaming route's grouping

    // What this window can serve: the widest lane count among this block's rows, capped at the
    // window. Lanes above it are never loaded, so a masked causal tail costs no bandwidth.
    int n_max = 0;
    for (int r = r_lo; r < r_hi; ++r) n_max = std::max(n_max, lanes_of(a, r));
    const int span = std::max(0, std::min(n_max, blo + MAXSC) - blo);

    // ---- pass 1: stage K a sub-block of L lanes at a time and score every pair's lanes out of it.
    // Both inner loops are warp-partitioned by pair, so no barrier is inside either of them.
    for (int s_lo = 0; s_lo < span; s_lo += ATTN_TILE_L) {
        const int s_hi = std::min(span, s_lo + ATTN_TILE_L);
        for (int p = w; p < pairs; p += nw) {
            const int row = r_lo + p / ATTN_TILE_G;
            const int l_p = tile_lim(a, row, blo, span);
            for (int rr = s_lo; rr < s_hi && rr < l_p; ++rr) {
                const int j = blo + rr;
                const int slot = rr - s_lo;      // a SLOT in `s_k`, in [0, ATTN_TILE_L). Using `rr`
                                                // here indexes past the array: see the header note.
                // Lane `pos + row` is this wave's own §05.7 append for `row`; like `attn_core` the
                // self lane comes from the staged buffers, not from the cache slot it may not be in
                // yet. `NEG_SLOTK` lands on that slot and so shows up in the NEXT row's read side.
                const hip_bfloat16* ksrc =
                    (j == a.pos + row)
                        ? a.krope + (size_t)ROW_KV * row + (size_t)kv_write * DH
                        : a.kcache + ((size_t)kv_read * a.cap + (size_t)j) * DH;
                *reinterpret_cast<uint4*>(&s_k[slot * DH + lane * 8]) =
                    *reinterpret_cast<const uint4*>(ksrc + lane * 8);
            }
        }
        __syncthreads();                                   // K sub-block published
        for (int p = w; p < pairs; p += nw) {
            const int row = r_lo + p / ATTN_TILE_G, h = p % ATTN_TILE_G;
            const int l_p = tile_lim(a, row, blo, span);
            if (l_p <= s_lo) continue;                     // EMPTY is identity: emit nothing
            const int nd = (a.neg == NEG_DIMHALF) ? 4 : 8;  // of the 8 BF16 a lane owns
            float q[8];                                    // one query row, 8 dims per lane
            {
                const hip_bfloat16* qs = a.qrope + (size_t)ROW_H * row +
                                         (size_t)(ATTN_TILE_G * kv_write + h) * DH + lane * 8;
                uint4 u = *reinterpret_cast<const uint4*>(qs);
                const hip_bfloat16* qq = reinterpret_cast<const hip_bfloat16*>(&u);
                for (int t = 0; t < 8; ++t) q[t] = bf(qq[t]);
            }
            for (int rr = s_lo; rr < s_hi && rr < l_p; ++rr) {
                const hip_bfloat16* k = &s_k[(rr - s_lo) * DH + lane * 8];
                float dot = 0.f;
                for (int t = 0; t < nd; ++t) dot += bf(k[t]) * q[t];   // serial in t, §05.5
                dot = warp_sum(dot);                                   // 5-step butterfly
                // One BF16 round, then the scale. Exact in BF16 because SCALE is 2^-4, which is
                // what lets the staging be 2 bytes and still read back the identical F32.
                if (lane == 0) s_sc[p][rr] = fb(a.score_f32 ? dot * SCALE : bf(fb(dot)) * SCALE);
            }
        }
        __syncthreads();                                   // s_sc published for pass 2
    }

    // ---- pass 2: one pair per warp, in the shipped form's order. The state is warp-private, so
    // there is no cross-warp hand-off and no second barrier.
    for (int p = w; p < pairs; p += nw) {
        const int row = r_lo + p / ATTN_TILE_G, h = p % ATTN_TILE_G;
        const int l_p = tile_lim(a, row, blo, span);
        if (l_p <= 0) continue;                            // empty pair: emit nothing, see above
        const int head = ATTN_TILE_G * kv_write + h;
        const bool one_pass = (tile_blocks_for_row(a, row) == 1);
        const hip_bfloat16* sc = &s_sc[p][0];
        float m = 0.f, l = 0.f, acc[8];
        for (int u = 0; u < 8; ++u) acc[u] = 0.f;
        if (one_pass) {
            // The reference form, exactly `attn_core`'s one-pass branch: max, then sum-of-exp, then
            // the BF16-rounded normalised weight and P·V, each serial in index order.
            for (int t = 0; t < l_p; ++t) m = fmaxf(m, bf(sc[t]));
            for (int t = 0; t < l_p; ++t) l += expf(bf(sc[t]) - m);
            for (int t = 0; t < l_p; ++t) {
                const int j = blo + t;
                const hip_bfloat16* v =
                    (j == a.pos + row)
                        ? a.vg + (size_t)ROW_KV * row + (size_t)kv_write * DH + lane * 8
                        : a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH + lane * 8;
                const float wt = bf(fb(expf(bf(sc[t]) - m) / l));
                for (int u = 0; u < 8; ++u) acc[u] += wt * bf(v[u]);   // serial in t, §05.5
            }
            l = 1.0f;   // weights were normalised and rounded above; the epilogue must not divide
        } else {
            // The streaming form: `attn_core`'s tile-of-`tile` loop verbatim, so this route lands on
            // §07.3's already-banded bits instead of opening a third set. Only the window differs.
            for (int base = 0; base < l_p; base += tile) {
                const int nt = std::min(tile, l_p - base);
                float mt = -1e30f;
                for (int t = 0; t < nt; ++t) mt = fmaxf(mt, bf(sc[base + t]));
                const float mn = fmaxf(m, mt);
                const float rescale = expf(m - mn);
                float e[TILE], ls = 0.f;
                for (int t = 0; t < nt; ++t) { e[t] = expf(bf(sc[base + t]) - mn); ls += e[t]; }
                l = l * rescale + ls;
                for (int u = 0; u < 8; ++u) acc[u] *= rescale;
                for (int t = 0; t < nt; ++t) {
                    const int j = blo + base + t;
                    const hip_bfloat16* v =
                        (j == a.pos + row)
                            ? a.vg + (size_t)ROW_KV * row + (size_t)kv_write * DH + lane * 8
                            : a.vcache + ((size_t)kv_read * a.cap + (size_t)j) * DH + lane * 8;
                    for (int u = 0; u < 8; ++u) acc[u] += e[t] * bf(v[u]);
                }
                m = mn;
            }
        }

        // §05.6 epilogue, 8 dims per lane, same per-element arithmetic as `finish_head`.
        if (one_pass) {
            hip_bfloat16 o[8], g8[8];
            uint4 gu = *reinterpret_cast<const uint4*>(
                a.qg + (size_t)ROW_Q * row + g_off(head, a.neg) + lane * 8);
            const hip_bfloat16* gg = reinterpret_cast<const hip_bfloat16*>(&gu);
            for (int u = 0; u < 8; ++u) {
                const float av = bf(fb(acc[u] / l));
                o[u] = fb(av);
                const float g = bf(gg[u]);
                g8[u] = fb(av * bf(fb(1.0f / (1.0f + expf(-g)))));
            }
            *reinterpret_cast<uint4*>(a.aout + (size_t)ROW_H * row + (size_t)head * DH + lane * 8) =
                *reinterpret_cast<uint4*>(o);
            *reinterpret_cast<uint4*>(a.gout + (size_t)ROW_H * row + (size_t)head * DH + lane * 8) =
                *reinterpret_cast<uint4*>(g8);
        } else {
            // One partial per (row, head in group, window); `attn_combine`'s tile branch merges the
            // windows of the row. A row that needed none of a window wrote nothing to it.
            float* pp = a.part + (((size_t)row * ATTN_TILE_G + h) * windows + y) * (DH + 2);
            if (lane == 0) { pp[0] = m; pp[1] = l; }
            for (int u = 0; u < 8; ++u) pp[2 + lane * 8 + u] = acc[u];
        }
    }
}

// The wave entry point's second form. `a.tile_kernel` in the staged launch plan selects it; a
// zero-initialised CoreArgs keeps the shipped `attn_core`, which is what a one-row decode step is.
__global__ void attn_tile_wave(CoreArgs a) { attn_tile_body(a); }

// Merge the split partials of each query head with the same (max,sum) algebra, then §05.6.
__global__ void attn_combine(CoreArgs a) {
    const int d = threadIdx.x;
    if (a.tile_kernel) {
        // The tiled form's merge. Its partial slab is [row][head in group][window][2 + DH], so the
        // grid is (KV group, head in group) x row, and the row's own rule decides whether it needs a
        // merge at all: a row the staging fits was finished inside its core block, which is the same
        // argument that lets one launch mix exact rows with streaming ones. Every window below
        // `s_row` was written: a row with n lanes has lanes in window s iff s < ceil(n / MAXSC), and
        // that is exactly `s_row` — so this loop never reads a slot no core block filled.
        const int kv = blockIdx.x / ATTN_TILE_G, h = blockIdx.x % ATTN_TILE_G;
        const int qrow = blockIdx.y;
        const int s_row = tile_blocks_for_row(a, qrow);
        if (s_row == 1) return;
        const int windows = tile_split_of(a.lvalid + nrows_of(a) - 1);
        const float* p = a.part + (((size_t)qrow * ATTN_TILE_G + h) * windows) * (DH + 2);
        float m = 0.f, l = 0.f, acc = 0.f;
        for (int s = 0; s < s_row; ++s) {
            const float ms = p[(size_t)s * (DH + 2) + 0];
            const float ls = p[(size_t)s * (DH + 2) + 1];
            const float av = p[(size_t)s * (DH + 2) + 2 + d];
            const float mn = fmaxf(m, ms);
            const float r0 = expf(m - mn), r1 = expf(ms - mn);
            l = l * r0 + ls * r1;
            acc = acc * r0 + av * r1;
            m = mn;
        }
        const int head = ATTN_TILE_G * kv + h;
        const float av = bf(fb(acc / l));
        a.aout[ROW_H * qrow + (size_t)head * DH + d] = fb(av);
        const float g = bf(a.qg[ROW_Q * qrow + g_off(head, a.neg) + d]);
        a.gout[ROW_H * qrow + (size_t)head * DH + d] = fb(av * bf(fb(1.0f / (1.0f + expf(-g)))));
        return;
    }
    const int head = blockIdx.x;
    const int qrow = blockIdx.y;
    const int s_row = blocks_for_row(a, qrow);
    if (s_row == 1) return;      // that row's single block already finished it in `attn_core`
    const float* p = a.part + (((size_t)qrow * N_Q + head) * a.split) * (DH + 2);
    float m = 0.f, l = 0.f, acc = 0.f;
    for (int s = 0; s < s_row; ++s) {
        const float ms = p[(size_t)s * (DH + 2) + 0];
        const float ls = p[(size_t)s * (DH + 2) + 1];
        const float av = p[(size_t)s * (DH + 2) + 2 + d];
        const float mn = fmaxf(m, ms);
        const float r0 = expf(m - mn), r1 = expf(ms - mn);
        l = l * r0 + ls * r1;
        acc = acc * r0 + av * r1;
        m = mn;
    }
    finish_head(a, head, qrow, l, acc, d);
}

// prep (staged plan): blocks 0..7 do query heads, blocks 8..9 do KV heads (k rope + both appends).
// P8: `blockIdx.y` is the query row, so ONE launch of `gridDim.y` copies of this stages a whole wave
// and performs every row's §05.7 append at that row's own position — which is exactly why a wave needs
// a staged plan. In the fused form each core block appends only its own row's lane, and a wave's row t
// reads the lanes rows 0..t-1 appended; there is no grid-wide barrier to order that, so the appends go
// in a launch that has finished before any row reads the cache.
__global__ void attn_prep(CoreArgs a) {
    __shared__ float out[DH], zs[DH], red[MAXW];
    const int b = blockIdx.x, d = threadIdx.x;
    const int qrow = blockIdx.y;
    const int row_pos = a.pos + qrow;
    const int nw = blockDim.x / warpSize;
    if (b < N_Q) {
        head_norm_rope(a.qg + ROW_Q * qrow + q_off(b, a.neg), a.qn, (float)row_pos, a.neg, out, zs,
                       red, nw);
        a.qrope_out[ROW_H * qrow + (size_t)b * DH + d] = fb(out[d]);
        a.gate_out[ROW_H * qrow + (size_t)b * DH + d] = a.qg[ROW_Q * qrow + g_off(b, a.neg) + d];
    } else {
        const int kv = b - N_Q;
        head_norm_rope(a.kg + ROW_KV * qrow + (size_t)kv * DH, a.kn, (float)row_pos, a.neg, out, zs,
                       red, nw);
        a.krope_out[ROW_KV * qrow + (size_t)kv * DH + d] = fb(out[d]);
        a.kc_as[((size_t)kv * a.cap + slot_of(row_pos, a.cap, a.neg, 0)) * DH + d] = fb(out[d]);
        a.vc_as[((size_t)kv * a.cap + slot_of(row_pos, a.cap, a.neg, 1)) * DH + d] =
            a.vg[ROW_KV * qrow + (size_t)kv * DH + d];
    }
}

// The cos||sin table for one position, so the gate can prove the in-kernel transcendentals.
__global__ void rope_table(CoreArgs a) {
    const int d = threadIdx.x;
    if (d < ROT) {
        float c, s;
        // The reference table is cat([cos(32), cos(32)]): dims 32..63 repeat 0..31 (§05.3 pairs
        // (i, i+32)), so the frequency index is d % HR, not d.
        rope_cs((float)a.pos, d % HR, c, s);
        a.rope_out[d] = fb(c);
        a.rope_out[ROT + d] = fb(s);
    }
}
