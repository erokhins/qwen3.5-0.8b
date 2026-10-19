/*
 * gdn_rec_wf_kernels.h — bet G2: §04.4's looped recurrence spelled as ONE WAVEFRONT PER BLOCK, with
 * no shared memory, no `__syncthreads()`, and no cross-block reduction of any kind.
 *
 * THE SHAPE. `rec_loop_kernel` (the shipped looped body) runs `<<<NH=16, D=128>>>`: four wavefronts
 * per block, one state COLUMN per thread, and it buys the two row L2-norms with `block_sum_pair_f32`
 * — 18 shuffles, three barriers and an LDS round-trip of both normalised rows, per token, on the
 * dependency chain that separates token t from token t+1 (`prototype/gdn_walk/evidence-tokenbill-2026-
 * 10-19.txt`: 122 of 1,327 per-token slots, and the barriers are the part that forbids pipelining).
 * `rec_wf_kernel<COLS>` runs `<<<NH * (D / (32*COLS)), 32>>>`: ONE wavefront per block, and each lane
 * owns COLS state columns (4 by pre-registration, 2 as the split leg — see §"THE ONE FREE PARAMETER").
 * A row is 128 elements = 4 per lane, so every reduction in this kernel is a warp butterfly and there
 * is no barrier to pay: with 32 lanes, "the whole block" and "the whole wavefront" are the same set,
 * and `__shfl_xor_sync` delivers the sum to every lane in one instruction per level.
 *
 * WHY THE NORMS COST NOTHING HERE, which is BET.md §10's constraint 1 and the reason this geometry was
 * fundable at all. A lane loads its own 4 k and 4 q elements as two `float4`s (the same 512-byte rows
 * the shipped walk already reads — NOTHING is staged, zero new bytes), reduces the two sums of squares
 * with a 5-level xor butterfly, and keeps `kap` = rsqrtf(sk+eps) and `iot` = rsqrtf(sq+eps) in
 * registers. G1, the alternative way to get the barriers out, paid +32 KB/token/layer of normed copies
 * and gave most of its row win back in `gdn_gated_norm` and `glue` (§40). This pays 0 bytes.
 *
 * THE ALGEBRA IT COSTS, and it does cost some. With no shared memory there is no way to broadcast
 * `q̂[i]` into the SECOND walk, so the q-dot over the updated state is taken in closed form instead.
 * Writing `k̂ = κ·k` and `q̂ = Q_SCALE·ι·q` (both are scalar multiples of the raw rows — that is what an
 * L2 row norm is), the recurrence per token and head is:
 *
 *     r_j   = Σ_i S[i][j]·k̂_i           dv_j = β·(v_j − γ·r_j)
 *     S[i][j] ← γ·S[i][j] + k̂_i·dv_j    o_j  = Σ_i q̂_i·S_new[i][j]
 *
 * and the last line distributes:
 *
 *     o_j = Q_SCALE·ι · ( γ·Σ_i q_i S_old[i][j]  +  κ·(Σ_i k_i q_i)·dv_j )
 *
 * so ONE pass over the old state produces both `Σ k_i S[i][j]` and `Σ q_i S[i][j]`, the second walk
 * only has to write the state, and `q̂` never has to reach a lane that does not own column j. That is
 * the distributive law, not an approximation — but it is a different product tree from the shipped
 * accumulation, and `AGENTS.md`'s `q̃` row (from `prototype/gdn_collapse/results.md` §2) is the
 * precedent that a different product tree is a different number. Three named things therefore move,
 * and only these three: (1) the norm sums' association (a 4-per-lane xor butterfly is not
 * `block_sum_pair_f32`'s 4-warp fold), (2) the state dots' association (each lane accumulates its own
 * columns, and walk A interleaves the k-dot and the q-dot), (3) the `o` algebra above. BET.md §10
 * constraint 2 fixes the band for all three: the EXISTING §10.1 aggregates (cos ≥ 0.9995,
 * meanrel ≤ 5 %) through `engine/gate.py` on the pinned fixtures. No new band, no widening, no
 * `--ack-spill`. `prototype/gdn_walk/g2_mirror.py` is the CPU witness for the size of the move,
 * measured against a reorder-only floor so "re-association any rewrite pays" is separated from what
 * this geometry pays extra.
 *
 * THE ONE FREE PARAMETER, priced before measuring, because AGENTS.md asks for cycles-per-dependency at
 * realised occupancy rather than an instruction count. COLS trades wavefronts for per-wavefront work,
 * and the two sides of that trade do not move at the same speed:
 *
 *     COLS  blocks               per-token slots per wavefront (walks only, from the shipped census)
 *      4    16 × 32 threads      ~2·COLS·(2.75 + 5.6)/4 × 128 ≈ 2,700  (the pre-registered shape:
 *                                llama.cpp's, per §38's `--geom` census — 32 threads, LDS = 0)
 *      2    32 × 32 threads      ~1,350
 *      1    64 × 32 threads      ~  690, but then a lane owns ONE column and the ILP that is this
 *                                geometry's whole point is gone
 *
 * Occupancy is set by the GRID, not by the resource limits: at 40 CUs × 2 SIMDs (KFD `simd_count 80`,
 * `simd_per_cu 2`; `rocminfo` "Compute Unit 40"), the shipped `<<<16,128>>>` puts 64 wavefronts on 64
 * of the 80 SIMDs — one each — and COLS=4 puts 16 wavefronts on 16 of them. So the pre-registered form
 * has to run at a HIGH issue duty on a single wavefront per SIMD to pay for itself, and the split leg
 * exists so the window can say which of the two it lost to if it loses. `wf4` is what BET.md §10
 * pre-registered and therefore what the verdict is taken on; `wf2` is the same body at a finer split and
 * is reported as a sensitivity leg, not as a second bet.
 *
 * WHAT IT DOES NOT TOUCH. The state stays in global memory, streamed, at the same addresses the looped
 * kernel is handed (route A already measured LDS-resident state at +10.6 % and died on its kill line;
 * the chunked scan measured 5.9× worse; unroll pragmas changed nothing — BET.md §1 closes all three and
 * none is re-opened here). The per-token `<<<NH, D>>>` kernels are untouched, so a ONE-ROW body — every
 * decode step — is bit-identical by construction, and `resolve_gdn_loop` stays the only place that rule
 * lives. `--gdn-norm hoist` does not combine with this (the norms are already in registers) and the
 * launcher refuses the pair rather than run one form under the other flag's timing.
 *
 * THE TOKEN-TO-TOKEN CHAIN, which is the whole bet and is worth stating exactly. Lane `l` owns columns
 * {c0(l) … c0(l)+COLS−1} of one head's 128×128 state, and NOTHING ELSE EVER READS OR WRITES THOSE
 * COLUMNS — no barrier between token t's last store and token t+1's first load, because there is no
 * other reader. That is the same property `rec_loop_normed_kernel` relies on for its own zero-barrier
 * claim (engine/gdn_rec_hoist_kernels.h), and G1's lesson applies to the code shape, not just the text:
 * the row pointers are declared INSIDE the token loop with the `t` advance in the declaration, because a
 * "legal strength reduction" that hoists them silently walks token 0's rows forever (G1 shipped that
 * bug past every CPU check and the device gate caught it: layer 0's state fingerprint moved
 * n2=221.737 → n2=45.290).
 */
#pragma once

#include "prototype/gdn_decode/gdn_decode_kernels.h"   // D, NH, CH, Q_SCALE

// The row strides are the same identities `rec_loop_kernel` static_asserts, restated here because this
// kernel indexes them itself: `CH` on the conv's [n][6144] tile and `NH*D` on the [n][2048] output.
static_assert(CH == 3 * NH * D, "rec_wf's q|k|v row stride is CH; must be 3*NH*D = the engine's kQkvRows");
static_assert(NH * D == 2048, "rec_wf's o row stride is NH*D; must be the engine's kValueDim");

// One wavefront. Every reduction in this file is a warp butterfly, so a wider block would be a
// different reduction, not a faster one.
#define GDN_WF_LANES 32

/* The columns a lane owns, loaded and stored as ONE vector access per state row.
 *
 * Validity, both ways: the owned run starts at a column index that is a multiple of COLS (the segment
 * base is `lane*COLS + seg*32*COLS`), the row is D = 128 elements wide so the run never straddles a row,
 * and every base pointer is at least 16-byte aligned — `S` and `o` are hipMalloc'd tiles whose row
 * stride is 128 floats, and `h*D*D` and `t*(NH*D)` are both multiples of 128. A COLS outside {2,4}
 * fails the static_asserts below rather than issuing four scalar loads and pretending it vectorised. */
template <int COLS> struct GdnWfVec;

template <> struct GdnWfVec<4> {
    typedef float4 type;
    static __device__ __forceinline__ void get(const type* p, int i, float* out) {
        const float4 t = p[i];
        out[0] = t.x; out[1] = t.y; out[2] = t.z; out[3] = t.w;
    }
    static __device__ __forceinline__ void set(type* p, int i, const float* in) {
        float4 t; t.x = in[0]; t.y = in[1]; t.z = in[2]; t.w = in[3];
        p[i] = t;
    }
};

template <> struct GdnWfVec<2> {
    typedef float2 type;
    static __device__ __forceinline__ void get(const type* p, int i, float* out) {
        const float2 t = p[i];
        out[0] = t.x; out[1] = t.y;
    }
    static __device__ __forceinline__ void set(type* p, int i, const float* in) {
        float2 t; t.x = in[0]; t.y = in[1];
        p[i] = t;
    }
};

/* `v += __shfl_xor_sync(FULL_MASK, v, o)` for o = 16, 8, 4, 2, 1: every lane finishes holding the whole
 * 32-lane sum, which is what a broadcast reduction needs and what a `__shfl_down_sync` tree does not give
 * without a second pass. The association is therefore the halving fold on the XOR pairing — a different
 * tree from `block_sum_pair_f32`'s, which is one of the three named reasons `o` and `STATEFP` move. */
__device__ __forceinline__ float gdn_bfly_sum_f32(float v) {
    // The 64-bit mask is a HIP requirement on this toolchain, not a style choice — see AGENTS.md's HIP row
    // and `block_sum_pair_f32`, which passes the same `0xffffffffffffffffull`.
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 16);
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 8);
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 4);
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 2);
    v += __shfl_xor_sync(0xffffffffffffffffull, v, 1);
    return v;
}

/* One GDN layer, ONE launch, `tokens` token-steps — the same operand list, buffers and addresses as
 * `rec_loop_kernel`, so a diff between the two forms is a claim about the schedule and the arithmetic,
 * never about the access pattern. `COLS` is the columns-per-lane leg described in the header. */
template <int COLS>
__global__ void rec_wf_kernel(const float* __restrict__ qtile, const float* __restrict__ ktile,
                              const float* __restrict__ vtile, const float* __restrict__ gtile,
                              const float* __restrict__ betatile, float* __restrict__ S,
                              float* __restrict__ otile, int nheads, int tokens) {
    constexpr int LANES = GDN_WF_LANES;
    constexpr int GROUP = LANES * COLS;          // state columns ONE BLOCK covers
    constexpr int SEGS = D / GROUP;              // blocks per head
    static_assert(D % GROUP == 0, "a block's columns must tile the 128-column head exactly");
    static_assert(COLS == 4 || COLS == 2, "only the vectorised column widths are implemented");
    static_assert(D % COLS == 0, "the owned column run must not straddle a state row");

    const int h = blockIdx.x / SEGS;
    const int seg = blockIdx.x % SEGS;
    const int lane = threadIdx.x;
    typedef typename GdnWfVec<COLS>::type VecT;
    const int vstride = D / COLS;                // vector elements per state row
    const int v0 = (seg * GROUP + lane * COLS) / COLS;   // this lane's first vector column

    for (int t = 0; t < tokens; ++t) {
        // This token's rows, declared HERE with the `t` advance in the declaration — see the header:
        // hoisting these is the bug that walked every token with token 0's operands.
        const float* __restrict__ q = qtile + (size_t)t * CH + (size_t)h * D;
        const float* __restrict__ k = ktile + (size_t)t * CH + (size_t)h * D;
        const float* __restrict__ v = vtile + (size_t)t * CH + (size_t)h * D;
        const float* __restrict__ g = gtile + (size_t)t * NH;
        const float* __restrict__ beta = betatile + (size_t)t * NH;
        float* __restrict__ o = otile + (size_t)t * (NH * D) + (size_t)h * D;

        // ---- the row norms: 4 elements per lane, one butterfly each, no barrier and no LDS write.
        // The split is 4-per-lane whatever COLS is, because the norm needs all 128 elements of the row
        // and 32 lanes x 4 is the row. These are the SAME k and q rows the shipped walk reads; nothing
        // new is staged (BET.md §10 constraint 1: the answer is zero bytes).
        const float4 kv4 = reinterpret_cast<const float4*>(k)[lane];
        const float4 qv4 = reinterpret_cast<const float4*>(q)[lane];
        float sk = kv4.x * kv4.x;
        sk += kv4.y * kv4.y;
        sk += kv4.z * kv4.z;
        sk += kv4.w * kv4.w;
        float sq = qv4.x * qv4.x;
        sq += qv4.y * qv4.y;
        sq += qv4.z * qv4.z;
        sq += qv4.w * qv4.w;
        // Σ k_i q_i as well: it is the scalar that lets `o` skip the second walk (header, algebra §3).
        float kpq = kv4.x * qv4.x;
        kpq += kv4.y * qv4.y;
        kpq += kv4.z * qv4.z;
        kpq += kv4.w * qv4.w;
        const float kap = gdn_bfly_sum_f32(sk);           // → sk in every lane
        const float iot = gdn_bfly_sum_f32(sq);
        const float kpq_s = gdn_bfly_sum_f32(kpq);
        const float rs_k = rsqrtf(kap + 1e-6f);           // κ: k̂_i = k_i · κ, the shipped product order
        const float rs_q = rsqrtf(iot + 1e-6f);           // ι: q̂_i = q_i · ι · Q_SCALE
        const float gam = expf(g[h]), bet = beta[h];
        // The two scalars `o` is assembled from: `qsc` carries Q_SCALE·ι, `kf` carries κ·Σ k_i q_i.
        // Both must not carry ι — multiplying the bracket by ι twice is a second 1/‖q‖ and the CPU mirror
        // caught it (cos(o) 0.666): `gdn_walk/g2_mirror.py` is the witness for the shape written here.
        const float qsc = Q_SCALE * rs_q;
        const float kf = rs_k * kpq_s;

        float* __restrict__ Sh = S + (size_t)h * D * D;
        VecT* __restrict__ Shv = reinterpret_cast<VecT*>(Sh) + v0;

        // ---- walk A: one read of the old state, two dots. The k and q values arrive as UNIFORM loads
        // (every lane reads the same element of the same row), which is how a row reaches a lane that
        // does not own it once there is no shared memory to stage it in; whether they compile to
        // `s_load_b32` or to a per-lane broadcast load is a listing fact, reported by
        // `prototype/gdn_walk/walk_isa_diff.py --g2` and priced by the A/B, not asserted here.
        float rq[COLS], tq[COLS], sv[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) { rq[c] = 0.f; tq[c] = 0.f; }
        for (int i = 0; i < D; ++i) {
            GdnWfVec<COLS>::get(Shv, i * vstride, sv);
            const float kk = k[i];
            const float qq = q[i];
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                rq[c] += sv[c] * kk;
                tq[c] += sv[c] * qq;
            }
        }

        // ---- dv, in the shipped shape: r_j first, then β·(v − γ·r).
        float dv[COLS];
        float vv[COLS];
        GdnWfVec<COLS>::get(reinterpret_cast<const VecT*>(v) + v0, 0, vv);
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const float r_c = rs_k * rq[c];
            dv[c] = bet * (vv[c] - gam * r_c);
        }

        // ---- o, in closed form over the OLD state (see the header's algebra). No second dot, so no
        // `q̂` broadcast, which is the thing that made this geometry possible at all.
        float ov[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) ov[c] = qsc * (gam * tq[c] + kf * dv[c]);
        GdnWfVec<COLS>::set(reinterpret_cast<VecT*>(o) + v0, 0, ov);

        // ---- walk B: the state only. `rs_k * k[i]` is `k̂_i` in the shipped product order
        // (`kq * rsqrtf(sk + 1e-6f)`), and the update keeps the shipped two-op shape `γ·S + k̂·dv`.
        for (int i = 0; i < D; ++i) {
            GdnWfVec<COLS>::get(Shv, i * vstride, sv);
            const float kn = k[i] * rs_k;
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                float s0 = gam * sv[c];
                s0 += kn * dv[c];
                sv[c] = s0;
            }
            GdnWfVec<COLS>::set(Shv, i * vstride, sv);
        }
    }
}
