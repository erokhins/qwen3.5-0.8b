// attn_decode_wide_kernels.h — Route D3: why is decode attention 3.023 ms/step at 4,032 lanes when
// the KV it must read is worth 0.21 ms? This header holds the first candidate, and it changes no
// arithmetic.
//
// THE FINDING THIS KERNEL IS THE FIX FOR. `attn_core` declares 6,176 B of `__shared__` (sq 1024 +
// sk 1024 + sv 1024 + zs 1024 + red 24 + sc 2048 + pad 32, at THREADS = 256) but its code object
// reserves 38,944 B — 32,768 B more, which is exactly 128 B per thread. The excess is LLVM's
// promote-alloca-to-LDS for the two per-thread arrays whose subscripts are only ever compared
// against a RUNTIME trip count: `float qreg[8]` (read by `for (t = 0; t < nd; ++t)`, where
// `nd = (a.neg == NEG_DIMHALF) ? 4 : 8` is runtime because `a.neg` is a kernel argument) and the
// `uint4 u` whose bytes the same loop indexes as `k[t]`. Bisected on this box (compile-only,
// `census.sh`): the `k[t]` loop alone costs 16,384 B (38,944 -> 22,560), both `nd` loops cost the
// whole 32,768 B (38,944 -> 6,176), and `float e[TILE]` in the streaming branch costs nothing
// (unchanged at 38,944 when only its `nt` loop is rewritten).
//
// WHY THAT IS A DECODE PROBLEM AND NOT A CURIOSITY. This part has 64 KiB of LDS per CU
// (`lds_size_in_kb 64` in /sys/.../nodes/1/properties, per the AGENTS.md rule that LDS comes from
// KFD and never from a HIP capability query) and a 1,024-thread per-CU ceiling. At 38,944 B per
// block, `attn_core` is resident ONE block per CU — 8 of 32 possible wavefronts, 25 % of the
// machine's latency-hiding capacity — and the extra 32,768 B is bought with zero information: the
// arrays are 8 floats each, indexed 0..7, always. At 6,176 B the LDS ceiling is 10 blocks/CU and
// the thread ceiling is 4 blocks/CU, i.e. FOUR TIMES the residency, with VGPR 45 instead of 53
// (`census.sh`). At 4,032 lanes the shipped geometry launches N_KV x split = 2 x 4 = 8 blocks per
// layer, which is under one block per CU anyway; what the reclaim buys is the *tile loop* — 63
// tiles of 8 lanes walked by a block that has nothing to overlap with while a dependent LDS load,
// a `ds_bpermute_b32` butterfly and a barrier are in flight.
//
// THE CHANGE, and why it is bit-identical. The two loops keep the same trip set and the same order
// of additions; only the trip count becomes a literal and the runtime test moves inside as a guard:
//      for (t = 0; t < nd; ++t)      dot += bf(k[t]) * qreg[t];
//      for (t = 0; t < 8;  ++t) if (t < nd) dot += bf(k[t]) * qreg[t];
// `nd` is 4 or 8, so the two spellings perform the same FMAs, in the same serial order, on the same
// values, and the skipped iterations contribute nothing to `dot`. The `dot` chain is
// data-dependent, so LLVM may not reassociate it regardless, but the argument here is the
// trip-set identity, not the optimizer's mood. `gate.py` proves it on device: every stage byte
// equal to the frozen `attn_core` on every case, including the `dimhalf` injection the guard
// exists to preserve.
//
// PROVENANCE. The kernel below was `prototype/attn_decode/attn_decode_kernels.h`'s `attn_core`, copied
// verbatim except (a) the name and (b) the two loop spellings above. **The port has now happened**
// (2026-10-21, `PORT-D3-LEAN.md`): the frozen header carries the literal-trip spelling itself, so (b)
// is gone and the name is the only difference — `gate.py --check-source` re-derives that from the two
// files and fails if the copy drifts or if either header regresses to a runtime trip count.
// Consequence to read before using `--core lean`: the two arms now compile from byte-identical bodies,
// so gate.py §1 (lean vs ship, bit-identity) is a tautology from here on and this rig's ship-vs-lean
// A/B measures one kernel twice. What this directory still witnesses is (i) the code-object claim —
// `census.sh` prints both reservations from the same object file, which is where the 38,944-vs-6,176
// asymmetry was found — and (ii) the case population and derived negatives, which the engine gate
// inherited. The pre-port device evidence is `results.md` and it stands on its own.
#pragma once
#include "../attn_decode/attn_decode_kernels.h"

__global__ void attn_core_lean(CoreArgs a) {
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
                    for (int t = 0; t < 8; ++t) if (t < nd) dot += sq[lane * 8 + t] * sk[lane * 8 + t];   // D3: literal trip + guard, NOT `nd` — promote-alloca-to-LDS; see "D3 LDS" at the foot of this file
                } else {
                    const hip_bfloat16* kr = a.kcache + ((size_t)kv * a.cap + (size_t)j) * DH;
                    uint4 u = *reinterpret_cast<const uint4*>(kr + lane * 8);
                    const hip_bfloat16* k = reinterpret_cast<const hip_bfloat16*>(&u);
                    for (int t = 0; t < 8; ++t) if (t < nd) dot += bf(k[t]) * qreg[t];   // D3: literal trip + guard, NOT `nd` — promote-alloca-to-LDS; see "D3 LDS" at the foot of this file
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
                    for (int t = 0; t < 8; ++t) if (t < nd) dot += sq[lane * 8 + t] * sk[lane * 8 + t];   // D3: literal trip + guard, NOT `nd` — promote-alloca-to-LDS; see "D3 LDS" at the foot of this file
                } else {
                    const hip_bfloat16* kr = a.kcache + ((size_t)kv * a.cap + (size_t)j) * DH;
                    uint4 u = *reinterpret_cast<const uint4*>(kr + lane * 8);
                    const hip_bfloat16* k = reinterpret_cast<const hip_bfloat16*>(&u);
                    for (int t = 0; t < 8; ++t) if (t < nd) dot += bf(k[t]) * qreg[t];   // D3: literal trip + guard, NOT `nd` — promote-alloca-to-LDS; see "D3 LDS" at the foot of this file
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
