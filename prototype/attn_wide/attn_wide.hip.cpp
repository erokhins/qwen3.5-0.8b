/*
 * attn_wide.hip.cpp — the A12 wide-shape kernel's standalone build: a correctness driver, not a benchmark.
 *
 * WHAT IT CHECKS, and against what. For each prompt length it runs (a) `attn_wide_wave` — the A12 wide block —
 * and (b) a deliberately dumb reference: one (query position, head) per thread, the whole row's scores in one
 * pass, ONE max over the row, no streaming, no rescale chain, no WMMA, no LDS, no tiling of any kind. If the
 * wide block's row/head/dim mapping, its causal mask, its four-head GQA packing, its lane-pair reduction or its
 * normalisation were wrong, it cannot agree with that reference to 1e-4 and would not pretend to. The two sides
 * deliberately SHARE their rounding policy — the same BF16 score round, the same F16 rounding of P, and `l`
 * summed over the UNROUNDED exp with the ROUNDED P going into the output sum, exactly as the kernel does — so
 * the residual is the streaming form, not a format difference.
 *
 * The reference is two launches because a thread cannot hold a 512-element row in registers or in locals: a
 * `float s[512]` local is 2 KB of private segment per thread, which is how a correctness check becomes an
 * allocation failure. Kernel R1 writes the rounded weights and the denominator into a scratch row; kernel R2
 * does the output contraction with one thread per (position, head, dim) so the V reads are coalesced.
 *
 * WHAT IT IS NOT: a numerics admission. The pinned-dump judgement is
 * `prototype/attn_wide/a12_wide_mirror.py`, on the derived fixture population, through `bf16_ulp`. This binary
 * is the "does the index arithmetic compute attention" check.
 *
 * NO TIMING. Nothing here prints a millisecond figure, a bandwidth, a best-of-N or a warm loop, and there is no
 * burst: a worker's device run is correctness-only (AGENTS.md, "GPU execution is serialized"). The reference is
 * O(npos²) per row and is the slow part, which is fine precisely because nobody is timing it.
 *
 * Build:  bash prototype/attn_wide/build_wide.sh [outdir]
 * Run:    <outdir>/attn_wide            (one window, all cases, seconds)
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "attn_wide.h"

namespace {

// Reference pass 1: the row's rounded softmax weights and its denominator.
//   s_j = bf16(bf16(dot) * SCALE)            (same round as the kernel, over an F32 serial dot)
//   mx  = max_j s_j                          (ONE max over the whole row: no streaming, no rescale)
//   p_j = f16(exp(s_j - mx))                 (the same rounding the kernel's P operand gets)
//   l   = sum_j exp(s_j - mx)                (UNROUNDED, as in the kernel: `ls += e`, not `ls += f16(e)`)
__global__ void attn_wide_ref_p(const WideArgs a, float* pbuf, float* lbuf) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pos = idx / WIDE_NQ;
    const int head = idx % WIDE_NQ;
    if (pos >= a.npos) return;
    const int kv = head / WIDE_GQA;
    const hip_bfloat16* qr = a.q + ((size_t)pos * WIDE_NQ + head) * WIDE_DH;
    const hip_bfloat16* kb = a.k + (size_t)kv * a.cap * WIDE_DH;
    float* p = pbuf + (size_t)idx * a.npos;
    float mx = -1e30f;
    const int nrow = pos + 1;                       // causal: this position sees lanes 0..pos
    for (int j = 0; j < nrow; ++j) {
        float dot = 0.f;
        for (int d = 0; d < WIDE_DH; ++d) dot += float(qr[d]) * float(kb[(size_t)j * WIDE_DH + d]);
        const float s = wide_bf(wide_fb(wide_fb(dot) * WIDE_SCALE));
        p[j] = s;
        mx = fmaxf(mx, s);
    }
    float l = 0.f;
    for (int j = 0; j < nrow; ++j) {
        const float e = expf(p[j] - mx);
        l += e;
        p[j] = __half2float(__float2half_rn(e));
    }
    lbuf[idx] = l;
}

// Reference pass 2: out[pos,head,d] = (sum_j p_j v[j,d]) / l. One thread per (pos, head, dim) so a warp's 32
// threads read 32 consecutive dims of one KV row — coalesced — and the V tile stays L2-resident.
__global__ void attn_wide_ref_out(const WideArgs a, const float* pbuf, const float* lbuf) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;      // over npos * NQ * DH
    const int d = idx % WIDE_DH;
    const int rh = idx / WIDE_DH;
    const int pos = rh / WIDE_NQ;
    const int head = rh % WIDE_NQ;
    if (pos >= a.npos) return;
    const int kv = head / WIDE_GQA;
    const hip_bfloat16* vb = a.v + (size_t)kv * a.cap * WIDE_DH;
    const float* p = pbuf + (size_t)rh * a.npos;
    float acc = 0.f;
    for (int j = 0; j <= pos; ++j) acc += p[j] * float(vb[(size_t)j * WIDE_DH + d]);
    a.out[idx] = acc / lbuf[rh];
}

// A deterministic bf16 fill, ~U[-3, 3): with 256 dims the scaled dot has a standard deviation near 1, so the
// softmax has real spread and a max that matters. A flat distribution would make every weight ~1/n and hide a
// broken rescale chain, which is the defect class this has to be able to see.
struct Lcg {
    unsigned long long s = 88172645463325252ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long r = (s >> 33) & 0xffff;
        return 6.0f * ((float)r / 65536.0f) - 3.0f;
    }
};

struct Case { int npos; const char* why; };

int run_case(int npos, const char* why) {
    const size_t n = (size_t)npos * WIDE_NQ * WIDE_DH;
    const int cap = npos + 8;                      // slack lanes, poisoned below
    Lcg rng;
    std::vector<hip_bfloat16> hq(n), hk((size_t)WIDE_NKV * cap * WIDE_DH),
                                  hv((size_t)WIDE_NKV * cap * WIDE_DH);
    for (auto& x : hq) x = hip_bfloat16(rng.next());
    for (auto& x : hk) x = hip_bfloat16(rng.next());
    for (auto& x : hv) x = hip_bfloat16(rng.next());
    // Lanes beyond every row's causal prefix are POISON, not zeros: an over-read that lands on zeros looks
    // merely small, one that lands on 9000 is unmistakable — the same rule `prototype/attn_decode/reference.py`
    // states for its own slack cache, and the reason `hi`-clamped reads must still be masked.
    for (int h = 0; h < WIDE_NKV; ++h)
        for (int j = npos; j < cap; ++j)
            for (int d = 0; d < WIDE_DH; ++d) {
                hk[((size_t)h * cap + j) * WIDE_DH + d] = hip_bfloat16(9000.0f);
                hv[((size_t)h * cap + j) * WIDE_DH + d] = hip_bfloat16(9000.0f);
            }

    WideArgs a{};
    hipMalloc((void**)&a.q, sizeof(hip_bfloat16) * n);
    hipMalloc((void**)&a.k, sizeof(hip_bfloat16) * hk.size());
    hipMalloc((void**)&a.v, sizeof(hip_bfloat16) * hv.size());
    hipMalloc((void**)&a.out, sizeof(float) * n);
    hipMemcpy((void*)a.q, hq.data(), sizeof(hip_bfloat16) * n, hipMemcpyHostToDevice);
    hipMemcpy((void*)a.k, hk.data(), sizeof(hip_bfloat16) * hk.size(), hipMemcpyHostToDevice);
    hipMemcpy((void*)a.v, hv.data(), sizeof(hip_bfloat16) * hv.size(), hipMemcpyHostToDevice);
    a.npos = npos;
    a.cap = cap;

    float* pbuf = nullptr;
    float* lbuf = nullptr;
    const size_t nrows = (size_t)npos * WIDE_NQ;
    hipMalloc((void**)&pbuf, sizeof(float) * nrows * npos);
    hipMalloc((void**)&lbuf, sizeof(float) * nrows);

    WideArgs r = a;                                  // the reference reads the same three device inputs; only
    float* rbuf = nullptr;                           // its output buffer is separate
    hipMalloc((void**)&rbuf, sizeof(float) * n);
    r.out = rbuf;
    hipMemset(a.out, 0, sizeof(float) * n);
    hipMemset(rbuf, 0, sizeof(float) * n);

    const int rtiles = (npos + WIDE_POS_PER_BLOCK - 1) / WIDE_POS_PER_BLOCK;
    attn_wide_wave<<<dim3(WIDE_NKV, 1, rtiles), dim3(WIDE_THREADS)>>>(a);
    attn_wide_ref_p<<<(unsigned)((nrows + 127) / 128), dim3(128)>>>(a, pbuf, lbuf);
    const size_t outn = (size_t)npos * WIDE_NQ * WIDE_DH;
    attn_wide_ref_out<<<(unsigned)((outn + 255) / 256), dim3(256)>>>(r, pbuf, lbuf);
    const hipError_t err = hipDeviceSynchronize();

    std::vector<float> got(n), want(n);
    hipMemcpy(got.data(), a.out, sizeof(float) * n, hipMemcpyDeviceToHost);
    hipMemcpy(want.data(), rbuf, sizeof(float) * n, hipMemcpyDeviceToHost);

    // Per (position, head) row over DH. A wrong row/head/dim mapping moves ONE row's cosine, which a
    // whole-tensor aggregate can average away, so the worst row is what is reported and gated (the
    // `ndiff == nrows x width` lesson: read the shape of a disagreement, not only its size).
    double worst_cos = 1.0, worst_mr = 0.0;
    int argmove = 0, nonfinite = 0;
    for (int i = 0; i < npos * WIDE_NQ; ++i) {
        const float* x = got.data() + (size_t)i * WIDE_DH;
        const float* y = want.data() + (size_t)i * WIDE_DH;
        double dot = 0, nx = 0, ny = 0, num = 0, den = 0;
        int ag = 0, ar = 0;
        double bg = -1e30, br = -1e30;
        for (int d = 0; d < WIDE_DH; ++d) {
            if (!std::isfinite(x[d])) ++nonfinite;
            dot += (double)x[d] * y[d]; nx += (double)x[d] * x[d]; ny += (double)y[d] * y[d];
            num += std::fabs((double)x[d] - y[d]); den += std::fabs(y[d]);
            if (x[d] > bg) { bg = x[d]; ag = d; }
            if (y[d] > br) { br = y[d]; ar = d; }
        }
        const double cos = (nx > 0 && ny > 0) ? dot / (sqrt(nx) * sqrt(ny)) : 0.0;
        const double mr = den > 0 ? num / den : 0.0;
        if (cos < worst_cos) worst_cos = cos;
        if (mr > worst_mr) worst_mr = mr;
        if (ag != ar) ++argmove;
    }

    // The bands, and what they are for: far tighter than any plausible mapping error (a permuted row gives a
    // cosine of a few tenths; half a denominator gives meanrel ~0.5 and cosine cannot see that AT ALL, which is
    // why meanrel is gated here and cosine alone would not be), and far looser than the streaming-vs-one-pass
    // residual, which the CPU mirror bounds in the 1e-4 range.
    const bool pass = err == hipSuccess && nonfinite == 0 && worst_cos >= 0.9995 && worst_mr <= 2e-2
                      && argmove * 20 <= npos * WIDE_NQ;
    printf("  npos=%-5d %-26s blocks=%-5d worst_cos=%.7f worst_meanrel=%.3e argmax_moved=%d/%d nonfinite=%d %s\n",
           npos, why, WIDE_NKV * rtiles, worst_cos, worst_mr, argmove, npos * WIDE_NQ, nonfinite,
           pass ? "PASS" : "FAIL");
    if (err != hipSuccess) printf("    hipError: %s\n", hipGetErrorString(err));

    hipFree((void*)a.q); hipFree((void*)a.k); hipFree((void*)a.v); hipFree(a.out); hipFree(rbuf);
    hipFree(pbuf); hipFree(lbuf);
    return pass ? 0 : 1;
}

}  // namespace

int main() {
    // The case list is DERIVED from the geometry, not hand-picked (the repo's rule for a must-catch set):
    // 1 is a single position; POS-1 / POS / POS+1 straddle the block tail and its `pos_read` clamp;
    // STAGE-1 / STAGE / STAGE+1 straddle the round boundary, where a row's prefix gains a position mid-round;
    // 2*STAGE+2 needs a partial round AFTER a full one; and the long rows are the only cases where the rescale
    // chain runs at all. Every pinned fixture in `data/refs/` is under 300 lanes, so 300 is the widest case the
    // pins can corroborate and 512 is the widest this can run without an O(n²) reference becoming the topic.
    const int POS = WIDE_POS_PER_BLOCK, ST = WIDE_STAGE;
    const std::vector<Case> cases = {
        {1, "single position"},
        {POS - 1, "block tail, short"}, {POS, "exactly one block"}, {POS + 1, "spills into a second block"},
        {ST - 1, "one round, short"}, {ST, "exactly one round"}, {ST + 1, "a second round, 1 lane"},
        {2 * ST + 2, "full round then a partial one"},
        {300, "the pinned fixtures' longest cache"},
        {512, "the widest many-round case"},
    };
    printf("attn_wide correctness: warps=%d threads=%d rows=%d stage=%d qreg=%d pvf16=%d pos_per_block=%d\n",
           WIDE_WARPS, WIDE_THREADS, WIDE_ROWS, WIDE_STAGE, WIDE_QREG, WIDE_PVF16, WIDE_POS_PER_BLOCK);
    int fails = 0;
    for (const auto& c : cases) fails += run_case(c.npos, c.why);
    printf("%s: %d case(s), %d failed\n", fails ? "FAIL" : "ATTN_WIDE SELFTEST PASS", (int)cases.size(), fails);
    return fails ? 1 : 0;
}
