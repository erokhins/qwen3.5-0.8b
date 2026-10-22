/*
 * cause_probe.cpp — can the engine's OWN P*V roll shape move its F16 result when one dead k-term's V cell
 * flips +0 -> -0? See prototype/mtp_verify/CAUSE.md.
 *
 * WHY THIS EXISTS WHEN `wmma_sign` ALREADY SAYS "NO".
 * `wmma_sign` section 2 measured 64 hand-built dot positions and found no signed-zero sensitivity, controls
 * firing. But its tile differs from the shipping round in three ways that this probe exists to close, and all
 * three are properties of the round, not of one dot position:
 *
 *   1. C. In the firing case the block runs ONE masked round (`nfull = lo_bound / WIDE_STAGE = 1/64 = 0`),
 *      so `out.t[tau]` starts at +0 and the per-round rescale multiplies it by `expf(m - mn)` =
 *      `expf(-1e30f - finite)` = +0: the C operand of every one of the round's issues is +0. `wmma_sign`
 *      used C = 0x3ECD+e/4 (non-zero) everywhere, and its zero-C arm (`zsum`) had exactly two zero products
 *      in a cancelling arrangement. A +-0 addend can only matter through what the adder does to a sum that is
 *      itself built from signed zeros, so the C regime is not a detail.
 *   2. Dead-map width. For row 8 of the firing case, 55 of the row's 64 k-terms are masked, i.e. 55 of the
 *      products are +-0 with signs set by V. `wmma_sign` zeroed ONE slot in 16.
 *   3. Accumulation depth. The engine accumulates one D tuple through FOUR L groups x two issues = 8 chained
 *      `v_wmma_f16_16x16x16_f16` issues (`attn_wide_llama_kernels.h:336-340`), the tuple as both C and D of
 *      every one. `wmma_sign` chained two.
 *
 * So this file builds the engine's roll: 4 B tuples (the P side, one per 16-position group), 8 A tuples (the
 * `vlo`/`vhi` pair per group), one D tuple, 8 issues in the shipped order, and sweeps the three axes above.
 * The measure is one question per configuration: with every byte fixed except the SIGN of one dead V cell,
 * do the D bits differ? Must-fire controls per configuration say the slot feeds D and that a real 1-ULP
 * change at the accumulator is resolvable, so a null here means the same thing the null in `wmma_sign` means.
 *
 * Correctness-only device probe. No timing of any kind is taken or printed. Usage:
 *      ./tmp/mtp_verify/cause_probe/cause_probe [--arm engine|single] [--c zero|datum|small]
 *                                              [--ti 0|1] [--dump]
 */
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HWAVE 32          // one wave; the tile is a full 16x16 per operand tuple
#define NGR 4             // WIDE_LGROUPS: four 16-position groups per 64-position round

using Frag = short __attribute__((ext_vector_type(16)));

static_assert(sizeof(Frag) == 32, "one operand tuple is 8 VGPRs, 16 F16 halves");

// ---- f16 bit helpers (the same total-order law `wmma_sign` uses; +0 and -0 adjacent, negatives mirrored) --
static long f16_ulp_idx(uint16_t b) {
    return (b & 0x8000u) ? (long)((~(uint32_t)b) & 0xFFFFu) : (long)(b | 0x8000u);
}
static long f16_ulp_dist(uint16_t a, uint16_t b) {
    const long d = f16_ulp_idx(a) - f16_ulp_idx(b);
    return d < 0 ? -d : d;
}
static bool f16_is_nan(uint16_t b) { return (b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0; }

static void selftest() {
    struct { uint16_t a, b; long want; const char* what; } t[] = {
        {0x3ECD, 0x3ECC, 1, "datum value, one ULP down"},
        {0xBC01, 0xBC00, 1, "NEGATIVE adjacency"},
        {0x0000, 0x8000, 1, "+0 and -0 adjacent in the total order"},
        {0x0000, 0x3C00, 0x3C00, "+0 to 1.0"},
        {0x8000, 0xBC00, 0x3C00, "-0 to -1.0 mirrors"},
    };
    int bad = 0;
    for (auto& x : t) {
        const long got = f16_ulp_dist(x.a, x.b);
        if (got != x.want) { bad++; printf("SELFTEST FAIL %s: got %ld want %ld\n", x.what, got, x.want); }
    }
    if (bad) { printf("SELFTEST FAILED (%d)\n", bad); std::exit(2); }
}

// ---- the tile ---------------------------------------------------------------------------------------------
// Passed by GLOBAL POINTER: 4*(2+1)*32*16*2 B + C is ~13 KB, far over the 4 KB kernarg budget, and a by-value
// struct that size is how `wmma_sign` leaked NaN cells into a clean tile (its own §"rebuilt twice" record).
struct RollIn {
    unsigned short vlo[NGR][HWAVE][16];   // A operand of the op_sel-0 issue, per group  (the engine's `vlo`)
    unsigned short vhi[NGR][HWAVE][16];   // A operand of the op_sel-1 issue, per group  (the engine's `vhi`)
    unsigned short p  [NGR][HWAVE][16];   // B operand, shared by both issues            (the engine's `p[L]`)
    unsigned short c  [    HWAVE][16];    // the F16 accumulator as issued (C and D are the same tuple)
};
struct RollOut { unsigned short d[HWAVE][16]; };

// THE ENGINE'S ROLL, one tau's worth: `for L { wmma(vlo,p,D,0); wmma(vhi,p,D,1); }` — the shipped order and
// the shipped C/D aliasing (`attn_wide_llama_kernels.h:334-341`), 8 chained issues into one D tuple.
__global__ void k_roll(const RollIn* __restrict__ inp, RollOut* __restrict__ outp, int two_issue) {
    const RollIn in = *inp;
    const int t = threadIdx.x & (HWAVE - 1);
    Frag A[NGR], A2[NGR], B[NGR], D;
#pragma unroll
    for (int L = 0; L < NGR; ++L) {
#pragma unroll
        for (int e = 0; e < 16; ++e) A[L][e] = (short)in.vlo[L][t][e];
#pragma unroll
        for (int e = 0; e < 16; ++e) A2[L][e] = (short)in.vhi[L][t][e];
#pragma unroll
        for (int e = 0; e < 16; ++e) B[L][e] = (short)in.p[L][t][e];
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) D[e] = (short)in.c[t][e];
#pragma unroll
    for (int L = 0; L < NGR; ++L) {
        D = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A[L], B[L], D, 0);
        if (two_issue) D = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A2[L], B[L], D, 1);
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) outp->d[t][e] = (unsigned short)D[e];
}

// ---- host plumbing ----------------------------------------------------------------------------------------
static void* d_malloc(size_t n) {
    void* p = nullptr;
    if (hipMalloc(&p, n) != hipSuccess) { printf("hipMalloc(%zu) failed\n", n); std::exit(3); }
    return p;
}

static std::vector<unsigned short> run_roll(const RollIn& in, int two_issue) {
    RollIn* din = (RollIn*)d_malloc(sizeof(RollIn));
    RollOut* dout = (RollOut*)d_malloc(sizeof(RollOut));
    hipMemcpy(din, &in, sizeof(RollIn), hipMemcpyHostToDevice);
    hipMemset(dout, 0, sizeof(RollOut));
    k_roll<<<1, HWAVE>>>(din, dout, two_issue);
    if (hipGetLastError() != hipSuccess) { printf("LAUNCH FAIL k_roll\n"); std::exit(3); }
    if (hipDeviceSynchronize() != hipSuccess) { printf("SYNC FAIL k_roll\n"); std::exit(3); }
    std::vector<unsigned short> o(HWAVE * 16);
    hipMemcpy(o.data(), dout, sizeof(RollOut), hipMemcpyDeviceToHost);
    hipFree(din); hipFree(dout);
    return o;
}

// `nz` counts movers whose bits are NON-ZERO ON BOTH SIDES. The engine datum's mover is 0x3ECD -> 0x3ECC, a
// non-zero value crossing a rounding boundary. A D cell that is +0 in one arm and -0 in the other is a
// sign-of-zero flip: real in the bits, but it cannot move an output bit that is not itself zero, and
// `packab_bits.py` reports the real datum as "0 of them a +/-0 sign flip". Conflating the two is how a probe
// prints a leak that cannot be the one being hunted.
struct Diff { int ndiff = 0; long maxulp = 0; int nnan = 0, nnanx = 0; int lane = -1, elem = -1;
              int nz = 0; long nz_maxulp = 0; };
static Diff cmp(const std::vector<unsigned short>& x, const std::vector<unsigned short>& y) {
    Diff d;
    for (size_t i = 0; i < x.size(); ++i) {
        if (f16_is_nan(y[i])) d.nnan++;
        if (f16_is_nan(x[i])) d.nnanx++;
        if (x[i] != y[i]) {
            d.ndiff++;
            const long u = f16_ulp_dist(x[i], y[i]);
            if (u > d.maxulp) d.maxulp = u;
            if (d.lane < 0) { d.lane = (int)(i / 16); d.elem = (int)(i % 16); }
            if ((x[i] & 0x7FFFu) != 0 && (y[i] & 0x7FFFu) != 0) { d.nz++; if (u > d.nz_maxulp) d.nz_maxulp = u; }
        }
    }
    return d;
}

// ---- the tile content, in the engine's own image ------------------------------------------------------------
// The P side is a softmax weight row: ALWAYS non-negative (it is `f16(expf(acc - mn))`), the row's own max is
// 1.0, and for row 8 of the firing case exactly 9 of the 64 columns are live and 55 are the exact +0 that
// `expf(-1e30f - mn)` produces. The V side is mixed-sign O(0.01..1) with the magnitudes a real V stage holds.
// `dead_single` is `wmma_sign`'s regime (one masked slot in one group) so the two probes overlap.
enum CMap { C_ZERO = 0, C_DATUM, C_SMALL };
enum DMap { DM_ENGINE = 0, DM_SINGLE };

static const uint16_t W_LIVE[9] = {0x3C00u, 0x3900u, 0x3A80u, 0x3400u, 0x3B00u,
                                   0x2C00u, 0x3880u, 0x3000u, 0x3600u};   // exp(s-mn): 1.0 down to ~2e-3

static uint16_t v_bits(int L, int t, int e) {
    // Mixed signs, mixed exponents, never zero: what a staged V row of real attention state looks like.
    const uint16_t mag = (uint16_t)(0x3400u + (((L * 5 + t * 3 + e * 7) & 0x3F) << 2));
    const uint16_t sgn = (((L + t + e) & 1) ? 0x8000u : 0u);
    return (uint16_t)(mag | sgn);
}

static void fill(RollIn* in, int cmap, int dmap) {
    std::memset(in, 0, sizeof(*in));
    for (int L = 0; L < NGR; ++L) {
        for (int t = 0; t < HWAVE; ++t) {
            for (int e = 0; e < 16; ++e) {
                in->vlo[L][t][e] = v_bits(L, t, e);
                in->vhi[L][t][e] = v_bits(L ^ 3, t, e ^ 1);
                // P: the row's max is 1.0 and everything else is a smaller exp; masked cells are +0.
                uint16_t p = 0x0000u;
                if (dmap == DM_ENGINE) { if (L == 0 && e < 9) p = W_LIVE[e]; }
                else { if (e != 5) p = W_LIVE[e & 7]; }        // one masked column, wmma_sign's shape
                in->p[L][t][e] = p;
                unsigned short cv = 0x0000u;
                if (cmap == C_DATUM) cv = (unsigned short)(0x3ECDu + (e & 3));
                else if (cmap == C_SMALL) cv = (unsigned short)(0x3000u + ((t + e) & 0x3F));
                in->c[t][e] = cv;
            }
        }
    }
}

// ---- characterisation: what value does the hardware ACT as if a signed zero were? --------------------------
// One slot, one mover cell, a grid over (V-slot bits) x (P-slot bits). The question each row answers:
//   does the +-0 operand act as a fixed tiny value (the shift is then independent of the partner's magnitude),
//   or as a corrupted operand that the partner multiplies (shift proportional to the partner)?
// Printed, not asserted: this is the datum the cause statement is written from.
static void characterise(int cmap, int arm) {
    const uint16_t vs[] = {0x0000u, 0x8000u, 0x0001u, 0x8001u, 0x0002u, 0x8002u, 0x0400u, 0x8400u,
                           0x3C00u, 0xBC00u};
    const uint16_t ps[] = {0x0000u, 0x3400u, 0x3C00u, 0x4000u};
    const int L = 0, p = 9;                    // the datum's own geometry: group 0, column 9, the `vlo` tuple
    printf("\n== CHARACTERISE side=vlo L=%d p=%d arm=%s C=%s\n", L, p, arm ? "single" : "engine",
           cmap == C_ZERO ? "+0" : (cmap == C_DATUM ? "datum" : "small"));
    printf("%-14s", "V bits \\ P bits");
    for (unsigned short ps_ : ps) printf("  p=%04x    ", ps_);
    printf("\n");
    for (unsigned short v : vs) {
        std::printf("  V=%04x      ", v);
        for (unsigned short pp : ps) {
            RollIn in; fill(&in, cmap, arm);
            for (int t = 0; t < HWAVE; ++t) { in.vlo[L][t][p] = v; in.p[L][t][p] = pp; }
            const std::vector<unsigned short> d = run_roll(in, 1);
            // Two cells, both movers for this geometry in the sweep above (lane 0 elem 2 and elem 6), so a
            // one-cell accident cannot be read as the pattern.
            printf("  %04x/%04x", d[2], d[6]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int arm = 0, cmap = 0, ti = 2, dump = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--arm") && i + 1 < argc) arm = !strcmp(argv[++i], "single") ? 1 : 0;
        else if (!strcmp(argv[i], "--c") && i + 1 < argc) {
            const char* s = argv[++i];
            cmap = !strcmp(s, "datum") ? C_DATUM : (!strcmp(s, "small") ? C_SMALL : C_ZERO);
        } else if (!strcmp(argv[i], "--ti") && i + 1 < argc) ti = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump")) dump = 1;
        else if (!strcmp(argv[i], "--charac")) { characterise(cmap, arm); return 0; }
        else { printf("usage: %s [--arm engine|single] [--c zero|datum|small] [--ti 0|1] [--dump] [--charac]\n", argv[0]); return 2; }
    }
    selftest();

    int configs = 0, fired = 0, fired_nz = 0, ctl = 0, ctl_fail = 0;
    const char* armn = arm ? "single-dead" : "engine-dead";
    const char* cn = cmap == C_DATUM ? "C=datum" : (cmap == C_SMALL ? "C=small" : "C=+0");

    for (int two_issue = (ti == 2 ? 0 : ti); two_issue <= (ti == 2 ? 1 : ti); ++two_issue) {
        // FIDELITY: the clean roll must produce a non-zero, NaN-free D in this issue shape, or nothing below
        // adjudicates (a NaN the shape itself makes must not be read as an effect of a mutation).
        {
            RollIn b; fill(&b, cmap, arm);
            const std::vector<unsigned short> clean = run_roll(b, two_issue);
            int nonzero = 0, nnan = 0;
            for (unsigned short u : clean) { if (u != 0 && u != 0x8000u) nonzero++; if (f16_is_nan(u)) nnan++; }
            const bool ok = nonzero > 0 && nnan == 0;
            if (!ok) ctl_fail++;
            ctl++;
            printf("%-44s %s ti=%d nonzero_cells=%d/%zu nan=%d %s\n", "fid clean roll ran",
                   ok ? "MUST-FIRE " : "*** NOT SATISFIED ***", two_issue, nonzero, clean.size(), nnan,
                   ok ? "this issue shape is usable" : "*** THIS SHAPE IS NOT USABLE ***");
        }
        for (int side = 0; side < 2; ++side) {          // 0 = the `vlo` tuple, 1 = the `vhi` tuple
            for (int L = 0; L < NGR; ++L) {
                for (int p = 0; p < 16; ++p) {
                    configs++;
                    RollIn plus, minus, wired, ulp1;
                    fill(&plus, cmap, arm);
                    auto put = [&](RollIn* in, uint16_t vslot, uint16_t pslot) {
                        for (int t = 0; t < HWAVE; ++t) {
                            if (side) in->vhi[L][t][p] = vslot; else in->vlo[L][t][p] = vslot;
                            if (pslot != 0xFFFFu) in->p[L][t][p] = pslot;
                        }
                    };
                    // THE MEASURE: the tested V cell is +0 vs -0, every other byte identical. In the engine
                    // the dead cell's staged V is exactly this: arm A wrote V[kv][dim] with a value below the
                    // F16 minimum subnormal, so the stage holds 0x0000 in one arm and 0x8000 in the other.
                    put(&plus, 0x0000u, 0xFFFFu);
                    minus = plus; put(&minus, 0x8000u, 0xFFFFu);
                    // MUST-FIRE 1 (`slotwired`): make that k-term live on the P side with V = 1.0. If D does
                    // not move, this (L,p) slot does not feed the D tuple at all and the null would be vacuous.
                    wired = plus; put(&wired, 0x3C00u, 0x3C00u);
                    // MUST-FIRE 2 (`acc1ulp`): a real 1-ULP change at the accumulator input, applied as a live
                    // P cell one ULP off the wired one plus a 1-ULP-off V, i.e. a change of ~1 f16 ULP in one
                    // product. Informational in `wmma_sign` (it fired at 56 of 64 positions there); kept
                    // informational here for the same reason: one addend in 16 can be below the dot's resolution.
                    ulp1 = plus; put(&ulp1, 0x3C01u, 0x3C00u);

                    const std::vector<unsigned short> b_plus = run_roll(plus, two_issue);
                    const std::vector<unsigned short> b_minus = run_roll(minus, two_issue);
                    const std::vector<unsigned short> b_wired = run_roll(wired, two_issue);
                    const std::vector<unsigned short> b_ulp = run_roll(ulp1, two_issue);

                    const Diff dz = cmp(b_plus, b_minus);
                    const Diff dw = cmp(b_plus, b_wired);
                    const Diff du = cmp(b_plus, b_ulp);
                    char kind[96];
                    std::snprintf(kind, sizeof kind, "zerosign %s L=%d p=%02d ti=%d %s",
                                  side ? "vhi" : "vlo", L, p, two_issue, cn);
                    if (dw.ndiff == 0) { ctl_fail++; }
                    ctl++;
                    fired += (dz.ndiff > 0) ? 1 : 0;
                    fired_nz += (dz.nz > 0) ? 1 : 0;
                    printf("%-44s %s ndiff=%d/%dnz maxulp=%ld/%ldnz nan=%d/%d | wired=%d acc1ulp=%d %s\n",
                           kind, dz.nz ? "*** SIGN MOVED A NON-ZERO D CELL ***"
                                : (dz.ndiff ? "movers are all +/-0 flips   " : "no difference           "),
                           dz.ndiff, dz.nz, dz.maxulp, dz.nz_maxulp, dz.nnanx, dz.nnan, dw.ndiff, du.ndiff,
                           dw.ndiff ? "" : "<<< slotwired DID NOT FIRE: this null is vacuous");
                    if (dump && dz.nz) {
                        int shown = 0;
                        for (size_t i = 0; i < b_plus.size() && shown < 8; ++i) {
                            if (b_plus[i] == b_minus[i]) continue;
                            if ((b_plus[i] & 0x7FFFu) == 0 || (b_minus[i] & 0x7FFFu) == 0) continue;
                            shown++;
                            printf("    D lane=%2d elem=%2d  +0 arm 0x%04x  -0 arm 0x%04x  ulp=%ld\n",
                                   (int)(i / 16), (int)(i % 16), b_plus[i], b_minus[i],
                                   f16_ulp_dist(b_plus[i], b_minus[i]));
                        }
                    }
                }
            }
        }
    }
    const int want = 128 * (ti == 2 ? 2 : 1);
    printf("SUMMARY arm=%s %s two_issue=%s configs=%d (want %d) controls=%d control_failures=%d "
           "sign_moves_any_D=%d sign_moves_NONZERO_D=%d\n", armn, cn, ti == 2 ? "both" : (ti ? "1" : "0"),
           configs, want, ctl, ctl_fail, fired, fired_nz);
    if (configs != want) { printf("ABORT: configuration census is not 4 L x 16 positions x 2 tuples x issue shape(s)\n"); return 4; }
    if (ctl_fail) { printf("ABORT: %d MUST-FIRE control(s) did not fire; those nulls measure nothing\n", ctl_fail); return 1; }
    printf("VERDICT: %s\n", fired_nz ? "THE SHIPPING ROLL SHAPE MOVES A NON-ZERO D CELL ON THE SIGN OF A ZERO K-TERM"
          : (fired ? "only +/-0 sign flips in D - cannot move a non-zero output bit, so it is not the datum"
                   : "no signed-zero sensitivity in the shipping roll shape (null, controls fired)"));
    return 0;
}
