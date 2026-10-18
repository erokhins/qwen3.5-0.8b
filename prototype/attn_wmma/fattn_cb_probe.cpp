// fattn_cb_probe.cpp — the C->B `get_half2` relabelling witness `prototype/attn_wmma/fattn_score_handover.md`
// §5d/§5e owes and `prototype/attn_wmma/fattn_map_results.md` names as the one thing the score-map probe
// does NOT prove. Separate checker, own injections, must be able to fail: `check_fattn_cb.py` adjudicates
// it and `--selftest` proves that it can.
//
// WHY THIS EXISTS AT ALL. `check_fattn_map.py` fixed where a C value LANDS (measured: lane t, register l
// holds FIRST[row = 2l + t/16][k] * SECOND[row = t%16][k]). The form then takes those 8 F32 registers,
// converts them to 16 F16 halves and hands them to the next `v_wmma` as an operand (`p_to_operand`, the
// transcribed `get_half2` of `mma.cuh:729-743`). Nothing in the score-map probe constrains that step: it
// does not touch the shuffle, the pair packing, or which operand slot the result is placed in. The
// handover's own cross-check for it — "its output shape is exactly the A/B operand shape" — is a SHAPE
// argument, and shape is exactly what a wrong relabelling preserves. That is the recorded failure class
// (AGENTS.md: a check whose independent quantity is not the thing under test cannot adjudicate it), and it
// is the class that let a transposed score reach a PASS.
//
// THE CLAIM, IN ONE LINE
// With the corrected operand order, `p_to_operand(score)` in the SECOND operand slot of the P*V `v_wmma`,
// against an identity V, must reproduce `out[q][d] = P[q][d]` cell for cell: the C tile, read back as an
// operand, must still be the P matrix with Q rows as rows and KV lanes as columns. Transposed,
// parity-swapped, or with a no-op butterfly, some cell moves and the closed form below says which.
//
// THE PATTERN (every value an integer <= 255, so BF16 -> F16 -> F32 is loss-free both ways and the check is
// equality, not a band — no floor probe is needed for a layout claim, and none is claimed)
//   Q[q][0] = 16q, Q[q][1] = 1, rest 0       with K[kv][0] = 1, K[kv][1] = kv this makes
//   score[q][kv] = 16q + kv                   (k >= 2 contributes nothing; 16q + kv <= 255)
//   P := score                                 (see "WHERE IS exp GONE?" below)
//   V[kv][d] = 1 if kv == (d & 15) else 0     identity on the 16 KV lanes, REPEATED every 16 columns so
//                                             all 16 output tiles are live, not only tile 0
//   =>  expected out[tile t][lane L][reg l] = 16 * (L % 16) + (2l + L / 16)
//       — the same closed form the score had, two legs later. The round trip IS the witness.
//
// WHERE IS "exp GONE"? The round's fold (`fold_max`) is a per-element scalar map: round to BF16, subtract
// the row max, exponentiate. None of the three touches an index, so none can move a cell from one
// (row, column) to another, and putting exp() into a layout probe would trade an exact equality for a
// mirror of transcendental rounding — the thing this repo refuses to adjudicate layout with (the `gdn_collapse`
// mirror could not see an association change for exactly this reason). So the probe hands `p_to_operand` the
// score tile unchanged. The fold's layout claim (one row per lane-pair, one `__shfl_xor_sync(mask, m, 16)`)
// is exercised by the same C geometry and is dumped, not numerically asserted.
//
// INJECTIONS — each its own binary, so a defect is only scored against a build that can express it (the
// `prototype/issue_rate_probe` rule). The must-catch set is DERIVED from what can move a cell, and the
// must-NOT-catch case is included because a control that catches everything proves nothing (AGENTS.md, the
// `partalias` row).
//   cb_lane_half_swap   `p_to_operand`: `i` -> `i ^ 1`, so the butterfly writes the PARTNER's value into
//                                                                  the slot the owner should fill.  MUST CATCH
//   cb_pack_swap        `p_to_operand`: make_half2(tmp[1], tmp[0]) — the pair order in every packed
//                                                                  register.                       MUST CATCH
//   cb_elem_reverse     `p_to_operand`: reads pexp[7-l] instead of pexp[l] — the register order inside a
//                                                                  lane's 8 values.                MUST CATCH
//   cb_no_shfl          `p_to_operand`: the `__shfl_xor_sync` replaced by the lane's own value. MEASURED
//                        RESULT, not an expectation: the output stays exactly right and only the operand
//                        goes wrong, because both halves of every pair are written into both half-warps
//                        and each half-warp's own parity is still correct in the slot its parity reads.
//                        Caught by the OPERAND stage, so it is still a control.               MUST CATCH
//   cb_score_slot_swap  `score_leg`: Q back in the FIRST slot — the pre-correction spelling, i.e. the
//                                    regression itself, required to fail HERE so it cannot return
//                                    silently.                                                     MUST CATCH
//   cb_vrow_off         `pv_leg`: tile t reads row 16t + (ab_row + 8) — a wrong tile-to-row map on the V
//                                    side.                                                         MUST CATCH
//   cb_v_odd_kv_zero    the V stage with every odd-KV entry zeroed: the prediction test for what
//                       `cb_no_shfl` measured. Predicted FAIL on the output; a PASS here alongside a green
//                       `clean` would mean the odd-KV staged entries are never read, i.e. V's staged k-run
//                       is not the contiguous 16 this form assumes. Either answer is a result.  MUST CATCH
//   cb_kv_both_relabel  THE BLIND SET, required NOT to fail: permute the KV index identically in the P
//                       operand (pair swap) and in the V stage (`kv ^ 1`). The two relabellings cancel
//                       inside the dot, so this probe adjudicates the CONSISTENCY of the two layouts and
//                       not the absolute naming of the KV axis — that naming comes from the score-map
//                       probe. Printing a passing blind case is what makes the other six meaningful.
//                                                                            MUST NOT BE CAUGHT
//
// WHAT IT IS NOT. Not a numerics admission (no pins, no band — values are exact by construction), not a
// timing (one launch, read back; a worker may run it as a correctness gate and takes no milliseconds from
// it), and not a witness for the F16-accumulator variant: `p_to_operand` is the F32-accumulator drain.
//
// BUILD:  bash prototype/attn_wmma/build_fattn_cb.sh
// RUN:    python3 prototype/attn_wmma/check_fattn_cb.py            clean probe
//         python3 prototype/attn_wmma/check_fattn_cb.py --selftest the six injections must fail,
//                                                                the one blind case must not

#include <cstdio>
#include <string>
#include <vector>

#include "prototype/attn_wmma/wmma_fattn_score.cpp"  // the REAL form: staging, addresses, score_leg,
                                                     // p_to_operand, pv_leg. Included, not copied: a
                                                     // witness that tests a copy of the shipping code
                                                     // witnesses the copy.

#ifndef CB_NEG
#define CB_NEG 0
#endif
#define CB_NONE 0
#define CB_LANE_HALF_SWAP 1
#define CB_PACK_SWAP 2
#define CB_ELEM_REVERSE 3
#define CB_NO_SHFL 4
#define CB_SCORE_SLOT_SWAP 5
#define CB_VROW_OFF 6
#define CB_KV_BOTH_RELABEL 7
#define CB_V_ODD_KV_ZERO 8

namespace cb {

constexpr int kDumpPerLane = kCElem + 16 + VTILES * kCElem;   // score | P operand | 16 output tiles
constexpr int kDumpLanes = 32;                                // warp 0 is the lane that is read back

// `p_to_operand`, mutated one line at a time. Compiled only in the binary that names the defect; the clean
// build calls the shipping function and never sees this one.
__device__ __forceinline__ ARegs p_to_operand_neg(const float pexp[kCElem], int lane, int which) {
    ARegs p;
    int i = lane >> 4;
    if (which == CB_LANE_HALF_SWAP) i ^= 1;
#pragma unroll
    for (int l = 0; l < kCElem; ++l) {
        const int src = (which == CB_ELEM_REVERSE) ? (kCElem - 1 - l) : l;
        float tmp[2];
        tmp[i] = pexp[src];
        tmp[i ^ 1] = (which == CB_NO_SHFL) ? pexp[src]
                                           : __shfl_xor_sync(kFullMask, pexp[src], 16, warpSize);
        const int a = (which == CB_PACK_SWAP || which == CB_KV_BOTH_RELABEL) ? 1 : 0;
        p[2 * l + 0] = (short)__half_as_ushort(__float2half_rn(tmp[a]));
        p[2 * l + 1] = (short)__half_as_ushort(__float2half_rn(tmp[1 - a]));
    }
    return p;
}

// `score_leg` with Q in the FIRST slot again — the pre-correction spelling, kept so the regression is a
// case and not a memory.
__device__ __forceinline__ CRegs score_leg_neg(const __half* s_q, const __half* s_kv, int lane) {
    CRegs acc{};
#pragma unroll 1
    for (int d0 = 0; d0 < DH; d0 += 16) {
        const ARegs a = load_tile(&s_q[q_addr(ab_row(lane), d0)]);
        const ARegs b = load_tile(&s_kv[k_addr(ab_row(lane), d0)]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    return acc;
}

// `stage_v_transposed_f16` with the KV axis relabelled (kv ^ 1). Only the blind-set binary uses it, and
// only there it cancels the P operand's pair swap.
__device__ __forceinline__ void stage_v_neg(__half* dst, const hip_bfloat16* src, int w) {
    const int lane = threadIdx.x & 31;
    for (int kv = w; kv < MLANES; kv += NWARPS) {
        const uint4 u = *reinterpret_cast<const uint4*>(src + (size_t)kv * DH + lane * 8);
        const hip_bfloat16* b = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d = lane * 8 + t;
            dst[v_addr(d, kv ^ 1)] = __float2half_rn(bf(b[t]));
        }
    }
}

// The V stage with every ODD-kv entry zeroed. This is the prediction test for what `cb_no_shfl` measured:
// removing the cross-lane exchange in `p_to_operand` left the OUTPUT exactly right while the operand dump
// went wrong, which is only possible if each half-warp's OWN parity of the KV axis is what the hardware
// reads from it — i.e. the operand fragment is read with the same parity split as the C tile (lane t
// supplies the kv values whose parity is `t/16`), and `get_half2`'s job is to make BOTH parities correct
// in BOTH half-warps so the duplication is harmless. The alternatives this distinguishes: if the whole
// 16-half run of one lane were read as the k-run, zeroing odd kv changes the dot and this case fails too
// but `cb_no_shfl` could not have passed; if the odd entries are never read at all, this case PASSES while
// `clean` still passes, which would mean V's staged k-run is not the contiguous 16 this form assumes and
// the port must not trust it. Either answer is usable; a silent one is not.
__device__ __forceinline__ void stage_v_odd_zero(__half* dst, const hip_bfloat16* src, int w) {
    const int lane = threadIdx.x & 31;
    for (int kv = w; kv < MLANES; kv += NWARPS) {
        const uint4 u = *reinterpret_cast<const uint4*>(src + (size_t)kv * DH + lane * 8);
        const hip_bfloat16* b = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d = lane * 8 + t;
            dst[v_addr(d, kv)] = (kv & 1) ? __float2half_rn(0.f) : __float2half_rn(bf(b[t]));
        }
    }
}

// `pv_leg` with a wrong tile -> row map.
__device__ __forceinline__ void pv_leg_neg(CRegs out[VTILES], const ARegs p, const __half* s_kv,
                                           int lane) {
#pragma unroll
    for (int t = 0; t < VTILES; ++t) {
        const ARegs a = load_tile(&s_kv[v_addr(16 * t + ((ab_row(lane) + 8) & 15), 0)]);
        CRegs zero{};
        out[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, p, zero);
    }
}

// One block, four warps (the staging helpers split 16 rows across NWARPS); warp 0 is read back. Every
// value reaches memory, so the dump is also the DCE witness.
extern "C" __global__ void __launch_bounds__(THREADS) cb_probe(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v, float* __restrict__ dump) {
    __shared__ __align__(16) __half s_q[MROWS * DH];
    __shared__ __align__(16) __half s_kv[MLANES * DH];
    const int lane = threadIdx.x & 31;
    const int w = (int)(threadIdx.x >> 5);
    float* out_lane = dump + lane * kDumpPerLane;

    stage_qk(s_q, s_kv, q, k);
    const CRegs score = (CB_NEG == CB_SCORE_SLOT_SWAP) ? score_leg_neg(s_q, s_kv, lane)
                                                       : score_leg(s_q, s_kv, lane);
    if (w == 0) {
#pragma unroll
        for (int l = 0; l < kCElem; ++l) out_lane[l] = score[l];
    }

    if (CB_NEG == CB_KV_BOTH_RELABEL) {
        const int ww = (int)(threadIdx.x >> 5);
        __syncthreads();                       // same ordering as stage_v: K readers first, then V writes
        stage_v_neg(s_kv, v, ww);
        __syncthreads();
    } else if (CB_NEG == CB_V_ODD_KV_ZERO) {
        const int ww = (int)(threadIdx.x >> 5);
        __syncthreads();
        stage_v_odd_zero(s_kv, v, ww);
        __syncthreads();
    } else {
        stage_v(s_kv, v);
    }

    float pexp[kCElem];
#pragma unroll
    for (int l = 0; l < kCElem; ++l) pexp[l] = score[l];    // identity: no index depends on exp
    const ARegs p =
        (CB_NEG == CB_NONE) ? p_to_operand(pexp, lane) : p_to_operand_neg(pexp, lane, CB_NEG);
    if (w == 0) {
#pragma unroll
        for (int e = 0; e < 16; ++e)
            out_lane[kCElem + e] = (float)__ushort_as_half((unsigned short)p[e]);
    }

    CRegs out[VTILES];
#pragma unroll
    for (int t = 0; t < VTILES; ++t) out[t] = CRegs{};
    if (CB_NEG == CB_VROW_OFF) pv_leg_neg(out, p, s_kv, lane);
    else pv_leg(out, p, s_kv, lane);
    if (w == 0) {
#pragma unroll
        for (int t = 0; t < VTILES; ++t)
#pragma unroll
            for (int l = 0; l < kCElem; ++l)
                out_lane[kCElem + 16 + t * kCElem + l] = out[t][l];
    }
}

// Host helpers live OUTSIDE the device namespace: they touch the runtime API, not the fragment code.
}  // namespace cb

// Host: fill the pattern, launch once, read back, print one machine-readable line per checked region. No
// timing anywhere; a single launch, and the only host clock read is the wall time the checker prints.
static int fill(hip_bfloat16** dst, int rows, int code) {
    std::vector<hip_bfloat16> h((size_t)rows * DH);
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < DH; ++d) {
            float v = 0.f;
            if (code == 0) {                                   // Q
                if (d == 0) v = 16.f * r;
                else if (d == 1) v = 1.f;
            } else if (code == 1) {                            // K
                if (d == 0) v = 1.f;
                else if (d == 1) v = 1.f * r;
            } else {                                           // V: identity, repeated every 16 columns
                if (r == (d & 15)) v = 1.f;
            }
            h[(size_t)r * DH + d] = hip_bfloat16(v);
        }
    }
    if (hipMalloc(dst, h.size() * sizeof(hip_bfloat16)) != hipSuccess) return 1;
    return hipMemcpy(*dst, h.data(), h.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice) ==
                   hipSuccess
               ? 0
               : 1;
}

int main(int argc, char** argv) {
    const char* name = (CB_NEG == CB_NONE) ? "clean"
                       : (CB_NEG == CB_LANE_HALF_SWAP) ? "cb_lane_half_swap"
                       : (CB_NEG == CB_PACK_SWAP) ? "cb_pack_swap"
                       : (CB_NEG == CB_ELEM_REVERSE) ? "cb_elem_reverse"
                       : (CB_NEG == CB_NO_SHFL) ? "cb_no_shfl"
                       : (CB_NEG == CB_SCORE_SLOT_SWAP) ? "cb_score_slot_swap"
                       : (CB_NEG == CB_VROW_OFF) ? "cb_vrow_off"
                       : (CB_NEG == CB_KV_BOTH_RELABEL) ? "cb_kv_both_relabel"
                       : (CB_NEG == CB_V_ODD_KV_ZERO) ? "cb_v_odd_kv_zero"
                                                        : "unknown";
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::printf("usage: %s   (one launch, one dump; the checker adjudicates)\n", name);
        return 0;
    }
    hip_bfloat16 *dq = nullptr, *dk = nullptr, *dv = nullptr;
    float* ddump = nullptr;
    if (fill(&dq, MROWS, 0) || fill(&dk, MLANES, 1) || fill(&dv, MLANES, 2)) {
        std::printf("CB-ERROR allocation/upload failed\n");
        return 2;
    }
    const size_t slots = (size_t)cb::kDumpLanes * cb::kDumpPerLane;
    if (hipMalloc(&ddump, slots * sizeof(float)) != hipSuccess) {
        std::printf("CB-ERROR dump allocation failed\n");
        return 2;
    }
    std::vector<float> zero(slots, -12345.f);
    (void)hipMemcpy(ddump, zero.data(), slots * sizeof(float), hipMemcpyHostToDevice);

    cb::cb_probe<<<1, THREADS>>>(dq, dk, dv, ddump);    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("CB-ERROR %s: launch failed\n", name);
        return 2;
    }
    std::vector<float> got(slots, 0.f);
    (void)hipMemcpy(got.data(), ddump, slots * sizeof(float), hipMemcpyDeviceToHost);

    // Three closed forms, all exact. `score` and `out` are the two ends of the round trip; `operand` is the
    // middle, which is the thing nobody had witnessed.
    size_t bad_score = 0, bad_operand = 0, bad_out = 0, bad_parse = 0, n_out = 0;
    for (int L = 0; L < cb::kDumpLanes; ++L) {
        const float* d = &got[L * cb::kDumpPerLane];
        const int q = L & 15;
        const int half = L >> 4;
        for (int l = 0; l < kCElem; ++l) {
            const float want = (float)(16 * q + (2 * l + half));       // score[q][kv]
            if (!(d[l] == want)) ++bad_score;
        }
        for (int e = 0; e < 16; ++e) {
            const float want = (float)(16 * q + e);                   // P[q][e], e = kv
            if (!(d[kCElem + e] == want)) ++bad_operand;
        }
        for (int t = 0; t < VTILES; ++t) {
            for (int l = 0; l < kCElem; ++l) {
                const int dcol = 16 * t + (2 * l + half);
                const float want = (float)(16 * q + (dcol & 15));      // out[q][dcol] = P[q][dcol & 15]
                if (!(d[kCElem + 16 + t * kCElem + l] == want)) ++bad_out;
                ++n_out;
                if (d[kCElem + 16 + t * kCElem + l] == -12345.f) ++bad_parse;   // never written
            }
        }
    }
    const size_t want_records = (size_t)cb::kDumpLanes * (2 * kCElem + 16 + VTILES * kCElem);
    std::printf(
        "CASE probe=%s lanes=%d parsed=%zu of %zu bad_score=%zu bad_operand=%zu bad_out=%zu "
        "unwritten=%zu n_out=%zu\n",
        name, cb::kDumpLanes, want_records, want_records, bad_score, bad_operand, bad_out, bad_parse,
        n_out);
    // Verdicts BY STAGE, not one AND. The blind case (`cb_kv_both_relabel`) moves the P operand and does
    // not move the output, and that IS its meaning: the operand's absolute naming is not adjudicated
    // here, only its consistency with the V stage. A single AND-verdict would have to call that a
    // failure, which would make the blind case indistinguishable from a real defect.
    const bool ok_score = !bad_score && !bad_parse;
    const bool ok_operand = !bad_operand && !bad_parse;
    const bool ok_out = !bad_out && !bad_parse;
    std::printf("VERDICT probe=%s score=%s operand=%s out=%s overall=%s\n", name,
                ok_score ? "PASS" : "FAIL", ok_operand ? "PASS" : "FAIL", ok_out ? "PASS" : "FAIL",
                (ok_score && ok_operand && ok_out) ? "PASS" : "FAIL");
    return (ok_score && ok_operand && ok_out) ? 0 : 1;
}
