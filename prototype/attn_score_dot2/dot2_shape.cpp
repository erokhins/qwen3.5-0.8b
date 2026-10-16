// dot2_shape.cpp — COMPILE-ONLY census: what does a dot2-bf16 score dot cost against the shipped
// scalar score dot, for exactly the operand shape the attention kernel uses?
//
// One touch = one warp scoring one KV lane for one query head: each lane owns 8 of the 256 head dims,
// held as 4 packed BF16 pairs (that is what `attn_tile_body`'s `ds_load_b128` of `s_k` delivers).
//   scalar form (shipped): 8 bf16->f32 conversions + 8 serial FMA          (§05.5's ordered reduction)
//   dot2 form (candidate): 4 v_dot2_f32_bf16, no conversions (operands are already packed BF16)
//
// No device is involved: hipcc -c compiles, llvm-objdump disassembles. See isa_census.py for the tally.
#include <hip/hip_bfloat16.h>

typedef short i2 __attribute__((ext_vector_type(2)));   // the two bf16 halves the fdot2 builtin wants

extern "C" __global__ void score_scalar(const uint4 *__restrict__ kpk,
                                        const uint4 *__restrict__ qpk,
                                        float *__restrict__ out) {
    const int lane = threadIdx.x;
    const uint4 k = kpk[lane];
    const uint4 q = qpk[lane];
    const hip_bfloat16 *kb = reinterpret_cast<const hip_bfloat16 *>(&k);
    const hip_bfloat16 *qb = reinterpret_cast<const hip_bfloat16 *>(&q);
    float dot = 0.f;
#pragma unroll
    for (int t = 0; t < 8; ++t) dot += float(kb[t]) * float(qb[t]);
    out[lane] = dot;
}

extern "C" __global__ void score_dot2(const uint4 *__restrict__ kpk,
                                      const uint4 *__restrict__ qpk,
                                      float *__restrict__ out) {
    const int lane = threadIdx.x;
    const uint4 k = kpk[lane];
    const uint4 q = qpk[lane];
    const unsigned *kp = reinterpret_cast<const unsigned *>(&k);
    const unsigned *qp = reinterpret_cast<const unsigned *>(&q);
    float dot = 0.f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        i2 ka, qa;
        ka = *reinterpret_cast<const i2 *>(&kp[p]);
        qa = *reinterpret_cast<const i2 *>(&qp[p]);
        dot = __builtin_amdgcn_fdot2_f32_bf16(ka, qa, dot, 0);
    }
    out[lane] = dot;
}
