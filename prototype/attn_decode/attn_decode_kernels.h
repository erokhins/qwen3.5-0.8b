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
};

/*
 * Stage outputs: q_rope, k_rope, the gate vector and the §05.7 append. The staged plan does this in
 * `attn_prep`; the fused plan does it in each core block (per query head, and per KV head for the
 * append), which is what lets the engine do the whole attention layer in one launch.
 */
__device__ void stage_head(const CoreArgs& a, int head, const float* sq, const float* sk,
                           const float* sv) {
    if (a.qrope_out == nullptr || blockIdx.y != 0) return;
    const int d = threadIdx.x, kv = kv_write_of(head);
    a.qrope_out[(size_t)head * DH + d] = fb(sq[d]);
    a.gate_out[(size_t)head * DH + d] = a.qg[g_off(head, a.neg) + d];
    if (head % GQA == 0) {          // one block per KV head owns that head's append
        a.krope_out[(size_t)kv * DH + d] = fb(sk[d]);
        a.kc_as[((size_t)kv * a.cap + slot_of(a.pos, a.cap, a.neg, 0)) * DH + d] = fb(sk[d]);
        a.vc_as[((size_t)kv * a.cap + slot_of(a.pos, a.cap, a.neg, 1)) * DH + d] = fb(sv[d]);
    }
}

/* §05.6 epilogue: normalise, round to BF16, gate with plain sigmoid. */
__device__ void finish_head(const CoreArgs& a, int head, float l, float acc, int d) {
    const float av = bf(fb(acc / l));
    a.aout[(size_t)head * DH + d] = fb(av);
    const float g = bf(a.qg[g_off(head, a.neg) + d]);
    a.gout[(size_t)head * DH + d] = fb(av * bf(fb(1.0f / (1.0f + expf(-g)))));
}

__global__ void attn_core(CoreArgs a) {
    __shared__ float sq[DH], sk[DH], sv[DH], zs[DH], red[MAXW];
    __shared__ float sc[(MAXSC < TILE) ? TILE : MAXSC];
    const int head = blockIdx.x;
    const int kv = kv_of(head, a.neg);
    const int d = threadIdx.x;
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;
    const float pos = (float)a.pos;

    if (a.fused) {          // §05.2 + §05.3 + §05.7, recomputed per block instead of staged
        head_norm_rope(a.qg + q_off(head, a.neg), a.qn, pos, a.neg, sq, zs, red, nw);
        head_norm_rope(a.kg + (size_t)kv * DH, a.kn, pos, a.neg, sk, zs, red, nw);
        for (int i = d; i < DH; i += blockDim.x) sv[i] = bf(a.vg[(size_t)kv * DH + i]);
        __syncthreads();
    } else {                // staged: prep already produced q_rope/k_rope and appended
        for (int i = d; i < DH; i += blockDim.x) {
            sq[i] = bf(a.qrope[(size_t)head * DH + i]);
            sk[i] = bf(a.krope[(size_t)kv * DH + i]);
            sv[i] = bf(a.vg[(size_t)kv * DH + i]);
        }
        __syncthreads();
    }
    stage_head(a, head, sq, sk, sv);

    // q in registers: lane owns dims lane*8..lane*8+7 — the same 512-byte read a K row needs.
    float qreg[8];
    for (int t = 0; t < 8; ++t) qreg[t] = sq[lane * 8 + t];

    const int nrow = (a.neg == NEG_READPAST) ? a.cap
                     : (a.neg == NEG_DROPCUR) ? a.pos : a.lvalid;
    const int span = (nrow + a.split - 1) / a.split;
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
    if (a.full) {
        int ns = 0;
        for (int base = blo; base < bhi; base += tile) {
            const int nt = std::min(tile, bhi - base);
            __syncthreads();
            if (w < nt) {
                const int j = base + w;
                float dot = 0.f;
                if (j == a.pos) {
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
            const float v = (j == a.pos)
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
                if (j == a.pos) {
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
                const float v = (j == a.pos)
                                    ? sv[d]
                                    : bf(a.vcache[((size_t)kv * a.cap + (size_t)j) * DH + d]);
                acc += e[t] * v;
            }
            m = mn;
        }
    }

    if (a.split == 1) {
        finish_head(a, head, l, acc, d);
        if (a.neg == NEG_CARRY && blockIdx.y == 0) {    // leave the state the next query must NOT
            a.stale[head * 2] = m;
            a.stale[head * 2 + 1] = l;
            a.stale[N_Q * 2 + (size_t)head * DH + d] = acc;
        }
    } else {
        float* p = a.part + ((size_t)head * a.split + blockIdx.y) * (DH + 2);
        p[0] = m;
        p[1] = l;
        p[2 + d] = acc;
    }
}

// Merge the split partials of each query head with the same (max,sum) algebra, then §05.6.
__global__ void attn_combine(CoreArgs a) {
    const int head = blockIdx.x, d = threadIdx.x;
    const float* p = a.part + (size_t)head * a.split * (DH + 2);
    float m = 0.f, l = 0.f, acc = 0.f;
    for (int s = 0; s < a.split; ++s) {
        const float ms = p[(size_t)s * (DH + 2) + 0];
        const float ls = p[(size_t)s * (DH + 2) + 1];
        const float av = p[(size_t)s * (DH + 2) + 2 + d];
        const float mn = fmaxf(m, ms);
        const float r0 = expf(m - mn), r1 = expf(ms - mn);
        l = l * r0 + ls * r1;
        acc = acc * r0 + av * r1;
        m = mn;
    }
    finish_head(a, head, l, acc, d);
}

// prep (staged plan): blocks 0..7 do query heads, blocks 8..9 do KV heads (k rope + both appends).
__global__ void attn_prep(CoreArgs a) {
    __shared__ float out[DH], zs[DH], red[MAXW];
    const int b = blockIdx.x, d = threadIdx.x;
    const int nw = blockDim.x / warpSize;
    if (b < N_Q) {
        head_norm_rope(a.qg + q_off(b, a.neg), a.qn, (float)a.pos, a.neg, out, zs, red, nw);
        a.qrope_out[(size_t)b * DH + d] = fb(out[d]);
        a.gate_out[(size_t)b * DH + d] = a.qg[g_off(b, a.neg) + d];
    } else {
        const int kv = b - N_Q;
        head_norm_rope(a.kg + (size_t)kv * DH, a.kn, (float)a.pos, a.neg, out, zs, red, nw);
        a.krope_out[(size_t)kv * DH + d] = fb(out[d]);
        a.kc_as[((size_t)kv * a.cap + slot_of(a.pos, a.cap, a.neg, 0)) * DH + d] = fb(out[d]);
        a.vc_as[((size_t)kv * a.cap + slot_of(a.pos, a.cap, a.neg, 1)) * DH + d] =
            a.vg[(size_t)kv * DH + d];
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
