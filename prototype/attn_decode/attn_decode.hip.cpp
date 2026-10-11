/*
 * prototype/attn_decode/attn_decode.hip.cpp — the §05 gated full-attention decode step (k = 1).
 *
 * Scope: everything between the projections and `o_proj`. One query row arrives as the pinned
 * projection outputs (q_proj [4096] = [8 heads][q 256 | gate 256], k_proj [512], v_proj [512]),
 * the request's KV cache and the position; out come `attn_out_pre_gate` and the gated operand
 * `o_proj` consumes. §05.1 de-interleave, §05.2 zero-centred per-head RMSNorm, §05.3 partial rope
 * (first ROT=64 dims only), §05.7 cache append, §05.5 GQA (8 query heads over 2 KV heads) with
 * online (max,sum) softmax, §05.6 output gate. `o_proj` is prototype/proj_gemv's.
 *
 * ROUNDING POLICY — measured against the pins, not chosen (results.md has the table):
 *   norm   z   = bf16((x * rsqrt(mean(x^2)+eps)) * (1+w))          one round     (§05.2)
 *   rope   c,s = bf16(cos/sin) ; z' = bf16(bf16(z_i*c) + bf16(rot*s))  three rounds (§05.3)
 *   score  s   = bf16(q.k) * SCALE                                 scores live in BF16 (§10.3)
 *   softmax      F32 exp/sum over the BF16 scores                                             (§05.5)
 *   a            = bf16(sum_j w_j v_j)                             one round
 *   gate out     = a * bf16(sigmoid(gate))                                                    (§05.6)
 * The BF16 rounding of cos/sin is not in §05.3's text; it is what the pins say, and rounding them
 * in F32 instead misses 52-97 elements of every 2048-element q_rope (speccheck's apply_rope does
 * exactly that). See results.md — this is the single finding that makes this op reproducible.
 *
 * TWO LAUNCH PLANS, one math:
 *   --launch staged  prep (norm+rope+append) -> core -> combine (only when split>1)
 *   --launch fused   the core recomputes its own head norm+rope and appends inline: 1 launch at
 *                    split 1, 2 above. run.py requires the two plans to agree bit-for-bit, so the
 *                    fused form cannot drift away from the gated one.
 * W5 wants the fused form: at cache lengths 18..277 this op moves ~0.6 MB per layer, so it is
 * launch-latency-bound, not bandwidth-bound.
 *
 * SOFTMAX. `--softmax online` (the engine form, §07.3) reduces the row in tiles with (max,sum)
 * rescaling; the accumulators belong to this query row and die with it (A-17). `--softmax full` is
 * the reference form: max first, sum second, weights rounded to BF16, then the P·V product. Both
 * are gated against the same pins and the gap between them is measured by floor_probe.py — a band
 * copied from a dot product is what §10.1 warns about.
 *
 * CACHE LENGTH IS THE TRAP. The kernel takes `lvalid` (lanes this query may attend to) and `cap`
 * (rows actually allocated). `readpast` and `dropcurrent` below are the two ways an engine gets
 * that wrong, and run.py exercises cache lengths 1..277 against pinned data.
 *
 * Injected bugs — each with a declared must-catch / must-NOT-catch stage (the mlp_gemv/proj_gemv
 * convention: a negative that cannot fail is worse than none, and one that fails everywhere is
 * measuring nothing):
 *   qswap       q_proj's two halves swapped (§05.1)
 *   gatehalf    gate read as the contiguous second half — the [all q | all gate] reading (§05.1)
 *   ropefull    rope over all 256 dims in (i, i+128) pairs instead of the first 64 (§05.3)
 *   ropenone    no rope at all
 *   gqaoff      kappa(i) off by one: query head i reads KV head (i/4 + 1) mod 2 (§05.5)
 *   nomax       (max,sum) state not rescaled when the max moves — the A-17 online-softmax bug
 *   carry       the row seeded from the PREVIOUS query's (max,sum,acc) (§07.3's own failure)
 *   readpast    attend to `cap` lanes instead of `lvalid` — the causal mask is gone
 *   dropcurrent attend to lanes < pos — the appended token forgotten
 *   dimhalf     the score dot accumulates 128 of the 256 head dims
 *   slotk       the key appended to the wrong cache slot (§05.7)
 *   slotv       the value appended to the wrong cache slot (§05.7)
 * `--score-f32` is a policy variant (skip the BF16 rounding of the score matrix), reported by
 * floor_probe.py rather than judged as a negative.
 *
 * Build: prototype/attn_decode/build.sh  (-O3, -Wl,-rpath,/opt/rocm/lib)
 * Run:   ./attn_decode --mode once  --plan PLAN [--launch staged|fused] [--split N]
 *                      [--softmax online|full] [--neg NAME] [--score-f32]
 *        ./attn_decode --mode burst --plan PLAN --iters N [--tile N] [--write-out] [same knobs]
 * Validate: prototype/attn_decode/run.py
 *
 * PLAN is a text file, one case per line:
 *   NAME QG KG VG KCACHE VCACHE QNORM KNORM POS LCACHE OUTDIR
 * QG/KG/VG are raw little-endian BF16 rows (4096/512/512 elements); KCACHE/VCACHE are raw BF16
 * [2][cap][256] with cap = filesize/(2*256*2), whose first LCACHE-1 lanes hold the existing cache;
 * QNORM/KNORM are 256 BF16. OUTDIR receives q_rope, k_rope, gate, rope_cs, attn_pre_gate, gated,
 * cache_keys, cache_values (raw BF16) in `once` mode, and in `burst` mode only with --write-out.
 * `rope_cs` is the 128-value cos||sin table built for POS: the gate uses it to prove the in-kernel
 * transcendentals match the reference table bit-for-bit instead of trusting them.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

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
        rope_cs((float)a.pos, d, c, s);
        a.rope_out[d] = fb(c);
        a.rope_out[ROT + d] = fb(s);
    }
}
// ------------------------------------------------------------------ host side
static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    std::vector<char> b((size_t)f.tellg());
    f.seekg(0);
    f.read(b.data(), (std::streamsize)b.size());
    if ((size_t)f.gcount() != b.size()) {
        fprintf(stderr, "%s: short read\n", p.c_str());
        exit(1);
    }
    return b;
}

static hip_bfloat16* upload(const std::string& path, size_t expect_elems) {    const std::vector<char> raw = slurp(path);
    if (raw.size() != expect_elems * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu BF16 elements\n", path.c_str(), raw.size(),
                expect_elems);
        exit(1);
    }
    hip_bfloat16* d = nullptr;
    CK(hipMalloc(&d, (size_t)raw.size()));
    CK(hipMemcpy(d, raw.data(), raw.size(), hipMemcpyHostToDevice));
    return d;
}

static void dump_file(const std::string& path, const void* dptr, size_t elems) {
    std::vector<char> h(elems * 2);
    CK(hipMemcpy(h.data(), const_cast<void*>(dptr), elems * 2, hipMemcpyDeviceToHost));
    std::ofstream of(path, std::ios::binary);
    of.write(h.data(), (std::streamsize)h.size());
}

// Shared across the whole plan: --neg carry reproduces §07.3's bug, which is one query's softmax
// state surviving into the NEXT query, so the buffer has to outlive a case.
static float* g_stale = nullptr;

struct Case {
    std::string name, outdir;
    int pos = 0, lvalid = 0, cap = 0;
    hip_bfloat16 *qg = nullptr, *kg = nullptr, *vg = nullptr, *qn = nullptr, *kn = nullptr;
    std::vector<hip_bfloat16*> kc, vc;          // cache replicas (--tile)
    hip_bfloat16 *qrope = nullptr, *krope = nullptr, *gate = nullptr, *aout = nullptr;
    hip_bfloat16 *gated = nullptr, *rcs = nullptr;
    float *part = nullptr, *stale = nullptr;

    // Bytes one launch moves, for the bandwidth sanity check only: at these cache lengths the op
    // is launch-latency-bound, so GB/s here measures how little it moves, not how well it moves it.
    size_t bytes(int split) const {
        const size_t rd = (size_t)4096 * 2 + (size_t)512 * 2 + (size_t)512 * 2
                          + (size_t)256 * 2 * 2 + (size_t)lvalid * 2048;
        const size_t wr = (size_t)(DH * N_Q) * 2 * 2 + (size_t)(DH * N_KV) * 2 * 2
                          + (size_t)ROT * 2 * 2 + (size_t)(DH * N_KV) * 2 * 2;
        return rd + wr + (split > 1 ? (size_t)N_Q * split * (DH + 2) * 4 : 0);
    }
};

int main(int argc, char** argv) {
    std::string mode = "once", plan_path, negname = "none", launch = "staged", softmax = "online";
    int split = 1, iters = 30, tile = 1, tile_rep = 1;
    bool score_f32 = false, write_out = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--launch") launch = next();
        else if (a == "--softmax") softmax = next();
        else if (a == "--split") split = atoi(next().c_str());
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--tile") tile_rep = atoi(next().c_str());
        else if (a == "--neg") negname = next();
        else if (a == "--score-f32") score_f32 = true;
        else if (a == "--write-out") write_out = true;
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }
    const int neg = negname == "none" ? NEG_NONE
                    : negname == "qswap" ? NEG_QSWAP
                    : negname == "gatehalf" ? NEG_GATEHALF
                    : negname == "ropefull" ? NEG_ROPEFULL
                    : negname == "ropenone" ? NEG_ROPENONE
                    : negname == "gqaoff" ? NEG_GQA
                    : negname == "nomax" ? NEG_NOMAX
                    : negname == "carry" ? NEG_CARRY
                    : negname == "readpast" ? NEG_READPAST
                    : negname == "dropcurrent" ? NEG_DROPCUR
                    : negname == "dimhalf" ? NEG_DIMHALF
                    : negname == "slotk" ? NEG_SLOTK
                    : negname == "slotv" ? NEG_SLOTV : -1;
    if (neg < 0) {
        fprintf(stderr,
                "--neg must be none|qswap|gatehalf|ropefull|ropenone|gqaoff|nomax|carry|readpast|"
                "dropcurrent|dimhalf|slotk|slotv\n");
        return 1;
    }
    if (softmax != "online" && softmax != "full") {
        fprintf(stderr, "--softmax must be online|full\n");
        return 1;
    }
    const bool full = softmax == "full";
    if (launch != "staged" && launch != "fused") {
        fprintf(stderr, "--launch must be staged|fused\n");
        return 1;
    }
    if (split < 1) {
        fprintf(stderr, "--split must be >= 1\n");
        return 1;
    }
    if (neg == NEG_CARRY && split != 1) {
        fprintf(stderr, "--neg carry is only defined at --split 1 (one state per query row)\n");
        return 1;
    }
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    const int nw = THREADS / prop.warpSize;
    if (prop.warpSize != 32 && prop.warpSize != 64) {
        fprintf(stderr, "warpSize %d unsupported (TILE assumes one warp per lane)\n",
                prop.warpSize);
        return 1;
    }
    tile = std::min(TILE, nw);

    CK(hipMalloc(&g_stale, ((size_t)N_Q * 2 + (size_t)N_Q * DH) * 4));
    CK(hipMemset(g_stale, 0, ((size_t)N_Q * 2 + (size_t)N_Q * DH) * 4));

    std::vector<Case> cases;
    {
        std::ifstream pf(plan_path);
        if (!pf) {
            fprintf(stderr, "cannot open plan %s\n", plan_path.c_str());
            return 1;
        }
        std::string line;
        int ln = 0;
        while (std::getline(pf, line)) {
            ++ln;
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string qgf, kgf, vgf, kcf, vcf, qnf, knf;
            Case c;
            if (!(ss >> c.name >> qgf >> kgf >> vgf >> kcf >> vcf >> qnf >> knf >> c.pos
                  >> c.lvalid >> c.outdir)) {
                fprintf(stderr, "plan %s line %d: NAME QG KG VG KCACHE VCACHE QNORM KNORM POS "
                                "LCACHE OUTDIR\n", plan_path.c_str(), ln);
                return 1;
            }
            const std::vector<char> kraw = slurp(kcf);
            if (kraw.size() % ((size_t)N_KV * DH * 2)) {
                fprintf(stderr, "%s: cache is not a whole [2][cap][256] BF16 block\n", kcf.c_str());
                return 1;
            }
            c.cap = (int)(kraw.size() / ((size_t)N_KV * DH * 2));
            if (c.lvalid < 0 || c.lvalid > c.cap || c.pos >= c.cap) {
                fprintf(stderr, "%s: pos %d / lvalid %d do not fit cap %d\n", c.name.c_str(),
                        c.pos, c.lvalid, c.cap);
                return 1;
            }
            if (full && (c.lvalid + split - 1) / split > MAXSC) {
                fprintf(stderr,
                        "%s: --softmax full stages %d lanes per block, MAXSC is %d (raise --split)\n",
                        c.name.c_str(), (c.lvalid + split - 1) / split, MAXSC);
                return 1;
            }
            c.qg = upload(qgf, 4096);
            c.kg = upload(kgf, 512);
            c.vg = upload(vgf, 512);
            c.qn = upload(qnf, DH);
            c.kn = upload(knf, DH);
            for (int r = 0; r < std::max(1, tile_rep); ++r) {
                hip_bfloat16 *kd = nullptr, *vd = nullptr;
                CK(hipMalloc(&kd, kraw.size()));
                CK(hipMalloc(&vd, kraw.size()));
                CK(hipMemcpy(kd, kraw.data(), kraw.size(), hipMemcpyHostToDevice));
                CK(hipMemcpy(vd, slurp(vcf).data(), kraw.size(), hipMemcpyHostToDevice));
                c.kc.push_back(kd);
                c.vc.push_back(vd);
            }
            CK(hipMalloc(&c.qrope, (size_t)N_Q * DH * 2));
            CK(hipMalloc(&c.krope, (size_t)N_KV * DH * 2));
            CK(hipMalloc(&c.gate, (size_t)N_Q * DH * 2));
            CK(hipMalloc(&c.aout, (size_t)N_Q * DH * 2));
            CK(hipMalloc(&c.gated, (size_t)N_Q * DH * 2));
            CK(hipMalloc(&c.rcs, (size_t)ROT * 2 * 2));
            c.stale = g_stale;               // ONE buffer for the whole plan: `carry` has to leak
            if (split > 1)                    // from case i into case i+1, not into itself
                CK(hipMalloc(&c.part, (size_t)N_Q * split * (DH + 2) * 4));
            cases.push_back(c);
        }
    }
    if (cases.empty()) {
        fprintf(stderr, "plan %s has no cases\n", plan_path.c_str());
        return 1;
    }

    const int launches = (launch == "staged" ? 1 : 0) + 1 + (split > 1 ? 1 : 0);
    printf("plan %s | cases=%zu | launch=%s (split %d) | softmax=%s | warps=%d tile=%d | neg=%s"
           " | score=%s | replicas=%d\n",
           plan_path.c_str(), cases.size(), launch.c_str(), split, softmax.c_str(), nw, tile,
           negname.c_str(), score_f32 ? "f32" : "bf16", std::max(1, tile_rep));
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    auto launch_case = [&](Case& c, int rep, bool stages) {
        CoreArgs a{};
        a.qg = c.qg; a.kg = c.kg; a.vg = c.vg;
        a.kcache = c.kc[(size_t)rep % c.kc.size()];
        a.vcache = c.vc[(size_t)rep % c.vc.size()];
        a.qn = c.qn; a.kn = c.kn;
        a.aout = c.aout; a.gout = c.gated;
        a.kc_as = a.kcache; a.vc_as = a.vcache;   // the append writes in place (§05.7)
        a.rope_out = c.rcs;
        a.part = c.part; a.stale = c.stale;
        a.cap = c.cap; a.pos = c.pos; a.lvalid = c.lvalid; a.split = split;
        a.neg = neg; a.full = full ? 1 : 0; a.score_f32 = score_f32 ? 1 : 0;
        a.fused = launch == "fused" ? 1 : 0; a.tile = tile;
        if (stages) rope_table<<<1, ROT>>>(a);   // stage check, not part of the timed path
        if (launch == "staged") {
            CoreArgs p = a;
            p.qrope_out = c.qrope; p.krope_out = c.krope; p.gate_out = c.gate;
            attn_prep<<<N_Q + N_KV, THREADS>>>(p);
            a.qrope = c.qrope; a.krope = c.krope;      // core reads what prep wrote
        } else {
            a.qrope_out = c.qrope; a.krope_out = c.krope; a.gate_out = c.gate;
        }
        a.aout = c.aout; a.gout = c.gated;
        attn_core<<<dim3(N_Q, split), THREADS>>>(a);
        if (split > 1) attn_combine<<<N_Q, THREADS>>>(a);
        CK(hipGetLastError());
    };

    auto write_case = [&](Case& c) {
        const std::string& o = c.outdir;
        dump_file(o + "/q_rope.bin", c.qrope, (size_t)N_Q * DH);
        dump_file(o + "/k_rope.bin", c.krope, (size_t)N_KV * DH);
        dump_file(o + "/gate_vec.bin", c.gate, (size_t)N_Q * DH);
        dump_file(o + "/rope_cs.bin", c.rcs, (size_t)ROT * 2);
        dump_file(o + "/attn_pre_gate.bin", c.aout, (size_t)N_Q * DH);
        dump_file(o + "/gated.bin", c.gated, (size_t)N_Q * DH);
        dump_file(o + "/cache_keys.bin", c.kc[0], (size_t)N_KV * c.cap * DH);
        dump_file(o + "/cache_values.bin", c.vc[0], (size_t)N_KV * c.cap * DH);
    };

    size_t plan_bytes = 0;
    for (auto& c : cases) plan_bytes += c.bytes(split);

    if (mode == "once") {
        int rep = 0;
        for (auto& c : cases) launch_case(c, rep++, true);
        CK(hipDeviceSynchronize());
        for (auto& c : cases) write_case(c);
        printf("RESULT mode=once cases=%zu launches=%d neg=%s split=%d launch=%s softmax=%s\n",
               cases.size(), launches, negname.c_str(), split, launch.c_str(), softmax.c_str());
        return 0;
    }

    // Burst: submit the whole plan `iters` times, one sync, wall-clock/N. hipEventElapsedTime
    // reads exactly 0 below ~1 ms on this iGPU (trap #7), so wall-clock is the number.
    auto burst = [&](int n) {
        const auto t0 = std::chrono::steady_clock::now();
        size_t counter = 0;
        for (int i = 0; i < n; ++i)
            for (auto& c : cases) launch_case(c, (int)counter++, false);
        CK(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    burst(1);                                     // warm: fresh pages run at ~35 GB/s
    const double ms = burst(std::max(1, iters)) / std::max(1, iters);
    const double gbs = (double)plan_bytes / (ms / 1e3) / 1e9;
    printf("RESULT mode=burst cases=%zu launches_per_case=%d ms_per_plan=%.6f us_per_case=%.3f"
           " gbs=%.2f probe_pct=%.2f neg=%s split=%d launch=%s softmax=%s bytes=%.1f KiB/plan\n",
           cases.size(), launches, ms, ms * 1e3 / cases.size(), gbs,
           100.0 * gbs / PROBE_GBS, negname.c_str(), split, launch.c_str(), softmax.c_str(),
           plan_bytes / 1024.0);
    if (write_out) {
        for (size_t i = 0; i < cases.size(); ++i) launch_case(cases[i], (int)i, true);
        CK(hipDeviceSynchronize());
        for (auto& c : cases) write_case(c);
    }
    return 0;
}
