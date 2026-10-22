/*
 * wmma_sign.cpp — can the SIGN of a zero k-term change an F16 matrix-product result on this part?
 *
 * `prototype/mtp_verify/ISOLATED.md` arm A established the datum's carrier: the dead k-term of the P*V leg
 * (`engine/attn_wide_llama_kernels.h:338-339`) is FORMED, not skipped — writing one dim of the dead lane's V
 * row to f16 `-0.0` moves the live row's output by exactly 1 BF16 ULP, twelve decades of magnitude change
 * nothing, and only the sign bit gets through. What it explicitly did NOT name (`ISOLATED`, "What this names
 * ... and what it does not"): **where a `-0.0` addend changes a non-zero F16 sum**, which is not a property of
 * an IEEE add. This file answers that in two sections, both on real F16 hardware paths.
 *
 * SECTION 1 — the micro semantics. The spellings of "add a zero product to a non-zero accumulator", at
 * several accumulator values: `x + (+0)`, `x + (-0)`, `fma(+0, +v, x)`, `fma(+0, -v, x)`, `fma(+0, +0, x)`,
 * `fma(-0, +0, x)`, the non-fused `x + (0*v)`, and a PACKED (`v_pk_*`) pair, because the shipping round's F16
 * arithmetic goes through packed ops (the per-round rescale is 64 `v_pk_mul_f16`,
 * `attn_wide_llama_kernels.h:278-288`). This is the cheapest possible witness for "a -0 addend cannot move a
 * non-zero sum" — and if it CAN on this part, that is the instruction-level answer.
 *
 * SECTION 2 — the hand-built `v_wmma_f16_16x16x16_f16` tile the brief asks for. One warp, one wmma (and the
 * engine's two-issue `op_sel` 0-then-1 shape), C a known non-zero value, exactly ONE k-position holding a
 * signed zero with every other byte fixed, dumping the D bits.
 *
 * WHY THIS IS LAYOUT-FREE, and why that is legitimate rather than sloppy. The operand map is pinned, not
 * assumed: `attn_fattn_kernels.h:469` `fattn_ab_row(lane) = lane & 15` — "lane t holds the 16 contiguous
 * halves of row t%16, the run along the dot index" — and the C->P exchange
 * (`attn_fattn_kernels.h:527-537` + `attn_wide_llama_kernels.h:259-266`) puts C's row (t%16)'s 16 columns
 * contiguous into `p[L][0..15]`. So for BOTH operands, element index `p` of a lane's fragment IS dot index
 * `p`, uniformly across lanes. Applying a mutation at element `p` on every lane is therefore exactly "dot
 * position p", with no need to model which D element it lands on — and the pair (bound-1, bound) that the
 * odd-bound law turns on is a pair of adjacent p values (`FIX-CROSSROW` Result 7).
 *
 * The two sides of the product are tested separately, because the datum's sign comes from the V side while the
 * P side already holds `+0.0`:
 *   - B side (P)  : the brief's literal `(+w_live, +0.0)` vs `(+w_live, -0.0)`;
 *   - A side (V)  : the engine's actual configuration — `P_dead = +0.0`, `V_dead` of either sign, product
 *                   sign = XOR = sign(V_dead). This is the reproduction; the B side is the control on it.
 *
 * CONTROLS, and which half is which (both halves derived, neither typed):
 *   MUST-FIRE   `slotlive`  the tested slot restored to its own live base value -> D must differ.
 *                           Without this, "the zero's sign is invisible" would also be true of a dead register
 *                           slot that is not in the dot at all. (The first version substituted the F16 min
 *                           subnormal instead, whose product underflows against an O(1) accumulator, so it
 *                           could not fire and reported a false breakage: a control must be
 *                           magnitude-appropriate, not merely different.)
 *   MUST-FIRE   `liveulp`   the SAME slot between two adjacent live values -> D must differ, i.e. the rig
 *                           resolves one F16 ULP at a dot position.
 *   MUST-FIRE   `liveulp`   the SAME position moved by 1 F16 ULP between two non-zero values -> D must differ.
 *                           Without this, the rig could simply be blind to 1-ULP moves.
 *   MUST-FIRE   `infnan`    the engine's own hazard shape: P slot = +0 AND V slot = +-inf -> D must be NaN
 *                           (one side alone gives +-inf, not NaN; the first version tested one side alone).
 *                           That is the hazard recorded in `FIX-ZERO.md`, witnessed here.
 *   MUST-FIRE   `fid`       the clean tile's D must be non-zero and non-NaN: if the wmma did not run, every
 *                           null below is about an empty rig (AGENTS' empty-parse row).
 *   MEASURE     `zerosign`  +0 vs -0 at one position, everything else byte-fixed. Not presumed either way;
 *                           its must-catch partners are the three above, so a null here is a measured null.
 *
 * NOTHING IS TIMED. No ms, no burst, no best-of-N, no warm loop. Bits in, bits out.
 * No engine file is included; the operand/accumulator shapes are re-declared here with a pointer to the
 * header that owns each law (the engine's are in an anonymous namespace and not includable, and re-declaring a
 * shape next to a cite of its owner is cheaper than widening that header for a probe).
 *
 * Build:  bash prototype/mtp_verify/wmma_sign/build.sh [outdir]
 * Run:    <outdir>/wmma_sign                  # everything; exit 0 iff every control behaved
 *         <outdir>/wmma_sign --section 1      # micro semantics only
 *         <outdir>/wmma_sign --section 2      # the tile only
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// ---- the shapes this probe reads, and where each law lives --------------------------------------------
// `attn_fattn_kernels.h:184` (operand tile, 16 halves/lane) and `:192` (F16 accumulator tile, 16 halves/lane,
// filled by two `v_wmma` issues with op_sel 0/1). Sizes are static-asserted, so a relabelled shape is a
// compile error rather than a permuted row (the class of defect that once came back as all-NaN).
using Frag = short __attribute__((ext_vector_type(16)));
using FragC32 = float __attribute__((ext_vector_type(8)));
using H2 = _Float16 __attribute__((ext_vector_type(2)));   // the packed pair: `v_pk_*` territory
static_assert(sizeof(Frag) == 32, "one F16 operand / F16 accumulator tile = 8 VGPRs");
static_assert(sizeof(FragC32) == 32, "one F32 accumulator tile = 8 VGPRs");

#define HWAVE 32

// ---- f16 helpers (host) ------------------------------------------------------------------------------
static uint16_t f16_bits(float f) {
    __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
static float f16_val(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}
// Monotone (total-order) index for an IEEE binary16 pattern: negatives are the complement, positives the same
// pattern with the sign bit set, so `a < b` in this index IS `a` orders below `b` as a number and adjacent
// patterns are 1 apart (+0 and -0 are adjacent and therefore 1 apart, which is what the total order says).
// `AGENTS.md`'s BF16-ULP trap is this line done wrong on a `uint16` view (`np.where(x < 0, ...)` is never true
// for uint16, every negative collapses to +0.0, and the gate passes vacuously). The sign test here is
// `b & 0x8000` and `selftest_ulp()` below is its witness, including the NEGATIVE-side cases.
static long f16_ulp_idx(uint16_t b) {
    return (b & 0x8000u) ? (long)((~(uint32_t)b) & 0xFFFFu) : (long)(b | 0x8000u);
}
static long f16_ulp_dist(uint16_t a, uint16_t b) {
    const long d = f16_ulp_idx(a) - f16_ulp_idx(b);
    return d < 0 ? -d : d;
}
static bool f16_is_nan(uint16_t b) { return (b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0; }
static bool f16_is_inf(uint16_t b) { return (b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) == 0; }

static void selftest_ulp() {
    struct { uint16_t a, b; long want; const char* what; } t[] = {
        {0x3ECD, 0x3ECC, 1, "datum value, one ULP down"},
        {0x3C00, 0x3C01, 1, "1.0 to next-up"},
        {0xBC01, 0xBC00, 1, "NEGATIVE adjacency (the dead-branch trap)"},
        {0x0000, 0x8000, 1, "+0 and -0 are adjacent in the total order"},
        {0x0001, 0x0002, 1, "min subnormal to next-up"},
        {0x7BFF, 0x7C00, 1, "max normal to +inf"},
        {0x0000, 0x3C00, 0x3C00, "+0 to 1.0 is the exponent-shift distance"},
        {0x8000, 0xBC00, 0x3C00, "-0 to -1.0 is the SAME distance (mirror check)"},
    };
    int bad = 0;
    for (auto& x : t) {
        const long got = f16_ulp_dist(x.a, x.b);
        if (got != x.want) {
            bad++;
            printf("SELFTEST-ULP FAIL %s: dist(0x%04x,0x%04x)=%ld want %ld\n", x.what, x.a, x.b, got, x.want);
        }
    }
    // The property a dead sign branch destroys: the negative side must mirror the positive side exactly.
    if (f16_ulp_dist(0x3C00, 0x4000) != f16_ulp_dist(0xC000, 0xBC00)) {
        bad++; printf("SELFTEST-ULP FAIL: negative side does not mirror the positive side\n");
    }
    if (!f16_is_nan(0x7E00) || !f16_is_nan(0xFC01) || f16_is_nan(0x7C00) || !f16_is_inf(0x7C00)) {
        bad++; printf("SELFTEST-ULP FAIL: nan/inf predicates\n");
    }
    if (f16_val(0xBC00) != -1.0f || f16_bits(-0.0f) != 0x8000u) { bad++; printf("SELFTEST-ULP FAIL: signed decode/encode\n"); }
    printf("SELFTEST ulp/decode %s\n", bad ? "FAIL" : "PASS");
    if (bad) std::exit(2);
}

// ================================================================================================================
// SECTION 1 kernel — micro semantics of a signed zero in F16 arithmetic, all spellings.
// Operands come from GLOBAL memory, so nothing is constant-folded: a folded answer would be a fact about
// LLVM's folder, not about this part's adders.
//   out[0] x+(+0)   out[1] x+(-0)   out[2] fma(+0,+v,x)  out[3] fma(+0,-v,x)
//   out[4] fma(+0,+0,x)  out[5] fma(-0,+0,x)  out[6] x+(0*+v)  out[7] x+(0*-v)   (non-fused pair)
//   out[8] fma(+v,+0,x) mirror      out[9..10] packed H2 pair (v_pk_* territory)
enum : int { M_NOPS = 11 };
// Device-side bit movement. `std::memcpy` is `__host__` on this toolchain (the build refuses it inside a
// `__global__`), so every bit cast in a kernel is a `__builtin_bit_cast` instead.
__device__ __forceinline__ unsigned short hraw(__half h) { return __builtin_bit_cast(unsigned short, h); }
__device__ __forceinline__ __half hbits(unsigned short b) { return __builtin_bit_cast(__half, b); }

__global__ void k_micro(const unsigned short* __restrict__ in, unsigned short* __restrict__ out, int livez) {
    const int t = threadIdx.x & (HWAVE - 1);
    const unsigned short* p = in + (size_t)t * 8;
    const __half x = hbits(p[0]);
    const __half v = hbits(p[2]);
    // `livez` is the must-fire lever: with it set, the operand that is a constructed +0/-0 elsewhere in this
    // kernel is instead a LIVE value read from the buffer. A control that varies only the OTHER operand of a
    // zero product cannot fire (0 x any finite is a zero either way), and the first version of this probe
    // made exactly that mistake, so the slot that holds the zero is what the control has to change.
    __half pz = __float2half_rn(0.0f);      // built, not a literal, so the compiler cannot pre-decide
    if (livez) pz = hbits(p[1]);
    const __half nz = __hneg(pz);
    unsigned short r[M_NOPS];
    r[0] = hraw(__hadd(x, pz));
    r[1] = hraw(__hadd(x, nz));
    r[2] = hraw(__hfma(pz, v, x));
    r[3] = hraw(__hfma(pz, __hneg(v), x));
    r[4] = hraw(__hfma(pz, pz, x));
    r[5] = hraw(__hfma(nz, pz, x));
    r[6] = hraw(__hadd(x, __hmul(pz, v)));
    r[7] = hraw(__hadd(x, __hmul(pz, __hneg(v))));
    r[8] = hraw(__hfma(v, pz, x));
    {   // the packed spelling: two halves in one 32-bit register, the shape the rescale fold uses.
      // z=+0 on lane-parity 0 and z=-0 on parity 1 in the SAME instruction, so one op decides both.
      const __half z = (t & 1) ? nz : pz;
      const unsigned int ua = (unsigned int)p[4] | ((unsigned int)p[5] << 16);
      const unsigned int ub = (unsigned int)p[6] | ((unsigned int)p[7] << 16);
      const H2 a = __builtin_bit_cast(H2, ua);
      const H2 b = __builtin_bit_cast(H2, ub);
      const __half z2[2] = {z, z};
      const H2 zz = __builtin_bit_cast(H2, (unsigned int)(hraw(z2[0]) | (hraw(z2[1]) << 16)));
      const H2 m = a * zz;                 // v_pk_mul_f16
      const H2 d = m + b;                  // v_pk_add_f16
      const unsigned int ud = __builtin_bit_cast(unsigned int, d);
      r[9] = (unsigned short)(ud & 0xFFFFu);
      r[10] = (unsigned short)(ud >> 16);
    }
    for (int i = 0; i < M_NOPS; ++i) out[(size_t)t * M_NOPS + i] = r[i];
}

// ================================================================================================================
// SECTION 2 kernels — the hand-built tile. One warp, one launch per case.
// Passed to the kernel by GLOBAL POINTER, not by value: four 1 KB arrays is exactly the 4 KB default kernarg
// budget, and a by-value struct that size is the kind of thing whose truncation shows up as three NaN cells in
// a CLEAN tile rather than as an error. A shape effect the probe cannot see is not allowed to become a finding.
struct TileIn {
    unsigned short a[HWAVE][16];    // operand A, op_sel-0 half = the engine's `vlo`
    unsigned short a2[HWAVE][16];   // operand A, op_sel-1 half = the engine's `vhi` (a DIFFERENT tuple)
    unsigned short b[HWAVE][16];    // operand B = the engine's P side, shared by both issues
    unsigned short c[HWAVE][16];    // the F16 accumulator / C operand (same shape as D)
};
struct TileOut { unsigned short d[HWAVE][16]; };

__global__ void k_wmma_h(const TileIn* __restrict__ inp, TileOut* __restrict__ outp, int two_issue) {
    const TileIn in = *inp;
    const int t = threadIdx.x & (HWAVE - 1);
    Frag A, A2, B, C, D;
#pragma unroll
    for (int e = 0; e < 16; ++e) A[e] = (short)in.a[t][e];
#pragma unroll
    for (int e = 0; e < 16; ++e) A2[e] = (short)in.a2[t][e];
#pragma unroll
    for (int e = 0; e < 16; ++e) B[e] = (short)in.b[t][e];
#pragma unroll
    for (int e = 0; e < 16; ++e) C[e] = (short)in.c[t][e];
    if (two_issue) {
        // The engine's shape, read out of ITS OWN listing rather than guessed: the causal_bound code object
        // emits `v_wmma_f16_16x16x16_f16 v[57:64], v[105:112], v[96:103], v[57:64]` then
        // `v_wmma_f16_16x16x16_f16 v[57:64], v[113:120], v[96:103], v[57:64] op_sel:[0,0,1]` - TWO DIFFERENT
        // A tuples (vlo, vhi), the SAME B, the SAME tuple as both C and D. Reusing one A tuple for both issues
        // is NOT that shape and was tried: the clean tile then produced NaN by itself (6 cells), which is how
        // this arm got rebuilt before any conclusion was drawn from it.
        D = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A, B, C, 0);
        D = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A2, B, D, 1);
    } else {
        D = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A, B, C, 0);
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) outp->d[t][e] = (unsigned short)D[e];
}

__global__ void k_wmma_f32(const TileIn* __restrict__ inp, unsigned int* __restrict__ outp) {
    const TileIn in = *inp;
    const int t = threadIdx.x & (HWAVE - 1);
    Frag A, B; FragC32 C, D;
#pragma unroll
    for (int e = 0; e < 16; ++e) A[e] = (short)in.a[t][e];
#pragma unroll
    for (int e = 0; e < 16; ++e) B[e] = (short)in.b[t][e];
#pragma unroll
    for (int e = 0; e < 8; ++e) C[e] = (float)(short)in.c[t][2 * e];
    D = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, C);
#pragma unroll
    for (int e = 0; e < 8; ++e) outp[t * 8 + e] = __builtin_bit_cast(unsigned int, (float)D[e]);
}

// ---- launch plumbing ---------------------------------------------------------------------------------
static void* d_malloc(size_t n) {
    void* p = nullptr;
    if (hipMalloc(&p, n) != hipSuccess) { printf("hipMalloc(%zu) failed\n", n); std::exit(3); }
    return p;
}
// The base tiles. Every k-term is non-zero EXCEPT the one under test, which is what the engine's masked round
// actually looks like (one dead column of a 16-wide group). V gets mixed signs so a product sign is not
// confounded with a positive-only operand; the C operand is the datum's own value family (0x3ECD =
// 0.400390625, the element arm A watched) shifted per element, so a 1-ULP move at one element cannot hide
// behind an identical neighbour.
static void base_tiles(TileIn* in, uint16_t live) {
    for (int t = 0; t < HWAVE; ++t) {
        for (int e = 0; e < 16; ++e) {
            in->a[t][e] = (e % 3 == 0) ? (uint16_t)(live ^ 0x8000u) : (uint16_t)(live + (e & 3));
            // The op_sel-1 A tuple is a DIFFERENT set of V rows (dims 16..31 of the same 64-position round),
            // so it carries different values over the SAME k axis. Mirroring `a` here would make the two
            // issues compute the same product, which the engine never does.
            in->a2[t][e] = (e % 2 == 0) ? (uint16_t)(0x3400u + (e & 5)) : (uint16_t)(0xB400u - (e & 3));
            in->b[t][e] = (uint16_t)(0x3A00u + ((t + e) & 7));
        }
        for (int e = 0; e < 16; ++e) in->c[t][e] = (uint16_t)(0x3ECDu + (e & 3));
    }
}

typedef std::vector<unsigned short> Bits;

static Bits run_h(const TileIn& in, int two_issue) {
    TileIn* din = (TileIn*)d_malloc(sizeof(TileIn));
    TileOut* dout = (TileOut*)d_malloc(sizeof(TileOut));
    hipMemcpy(din, &in, sizeof(TileIn), hipMemcpyHostToDevice);
    hipMemset(dout, 0, sizeof(TileOut));
    k_wmma_h<<<1, HWAVE>>>(din, dout, two_issue);
    if (hipGetLastError() != hipSuccess) { printf("LAUNCH FAIL k_wmma_h\n"); std::exit(3); }
    if (hipDeviceSynchronize() != hipSuccess) { printf("SYNC FAIL k_wmma_h (device fault?)\n"); std::exit(3); }
    Bits d(HWAVE * 16);
    hipMemcpy(d.data(), dout, sizeof(TileOut), hipMemcpyDeviceToHost);
    hipFree(din); hipFree(dout);
    return d;
}

static std::vector<unsigned int> run_f32(const TileIn& in) {
    TileIn* din = (TileIn*)d_malloc(sizeof(TileIn));
    unsigned int* dout = (unsigned int*)d_malloc(HWAVE * 8 * 4);
    hipMemcpy(din, &in, sizeof(TileIn), hipMemcpyHostToDevice);
    hipMemset(dout, 0, HWAVE * 8 * 4);
    k_wmma_f32<<<1, HWAVE>>>(din, dout);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        printf("LAUNCH/SYNC FAIL k_wmma_f32\n"); std::exit(3);
    }
    std::vector<unsigned int> o(HWAVE * 8);
    hipMemcpy(o.data(), dout, HWAVE * 8 * 4, hipMemcpyDeviceToHost);
    hipFree(din); hipFree(dout);
    return o;
}

// Compare two D dumps bitwise: count, max ULP distance, NaN count in EACH side, first mover. The NaN count
// is kept on BOTH sides on purpose: a NaN in the reference arm would mean the shape itself, not the mutation,
// makes NaN, and reading only one side would blame the mutation for it.
struct Diff { int ndiff = 0; long maxulp = 0; int nnan = 0, nnanx = 0; int lane = -1, elem = -1; };
static Diff cmp(const Bits& x, const Bits& y) {
    Diff d;
    for (size_t i = 0; i < x.size(); ++i) {
        if (f16_is_nan(y[i])) d.nnan++;
        if (f16_is_nan(x[i])) d.nnanx++;
        if (x[i] != y[i]) {
            d.ndiff++;
            const long u = f16_ulp_dist(x[i], y[i]);
            if (u > d.maxulp) d.maxulp = u;
            if (d.lane < 0) { d.lane = (int)(i / 16); d.elem = (int)(i % 16); }
        }
    }
    return d;
}

// ================================================================================================================
int main(int argc, char** argv) {
    int section = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--section") && i + 1 < argc) section = atoi(argv[++i]);
        else { printf("usage: %s [--section 1|2]\n", argv[0]); return 2; }
    }
    selftest_ulp();

    int checks = 0, ctl_fail = 0;        // controls: must-fire / must-not-fire, exit code depends on these
    int meas = 0, leak = 0;              // measures: the question itself, reported not asserted
    auto ctl = [&](const char* kind, const char* must, const Diff& d, bool want_diff) {
        const bool fired = d.ndiff > 0 || d.nnan > 0;
        const bool ok = want_diff ? fired : !fired;
        checks++;
        if (!ok) ctl_fail++;
        printf("%-30s %-9s ndiff=%5d maxulp=%-6ld nan=%4d first=(%3d,%2d) %s\n", kind, must,
               d.ndiff, d.maxulp, d.nnan, d.lane, d.elem, ok ? "control ok" : "*** CONTROL NOT SATISFIED ***");
    };
    auto measure = [&](const char* kind, const Diff& d, int mode) {
        meas++;
        const bool differs = d.ndiff > 0;
        if (differs && mode == 1) leak++;
        printf("%-44s %-28s ndiff=%5d maxulp=%-6ld nan=%4d/%4d %s\n", kind, "MEASURE", d.ndiff, d.maxulp,
               d.nnanx, d.nnan,
               differs ? (mode == 1 ? "<== SIGN REACHES THE RESULT"
                       : mode == 2 ? "(informational: a live term replaced by a much smaller one)"
                                   : "(zero accumulator: IEEE's own signed-zero rule)")
                       : "no difference");
    };

    // ------------------------------------------------------------------ SECTION 1
    if (section == 0 || section == 1) {
        printf("\n== SECTION 1 - F16 micro semantics of a signed zero (operands from global; nothing folded)\n");
        // Accumulator values: the datum's own element, unity, a small finite, max normal, F16 min subnormal,
        // and BOTH zeros (the zeros are kept separate: a difference at x=+-0 is IEEE's own signed-zero rule,
        // not a leak, and the datum's element is non-zero).
        const uint16_t xs[] = {0x3ECDu, 0x3C00u, 0x2000u, 0x7BFFu, 0x0001u, 0x3CCDu};
        const uint16_t zs[] = {0x0000u, 0x8000u};
        const uint16_t vs[] = {0x3C00u, 0x3ECDu, 0x0001u, 0x7C00u, 0x7BFFu};
        unsigned short* hin = new unsigned short[HWAVE * 8];
        unsigned short* hout = new unsigned short[HWAVE * M_NOPS];
        unsigned short* din = (unsigned short*)d_malloc(HWAVE * 8 * 2);
        unsigned short* dout = (unsigned short*)d_malloc(HWAVE * M_NOPS * 2);
        int pairs = 0;
        for (int pass = 0; pass < 2; ++pass) {                 // pass 0 = non-zero accumulators, 1 = zeros
            const int nx = pass == 0 ? (int)(sizeof xs / sizeof xs[0]) : (int)(sizeof zs / sizeof zs[0]);
            const uint16_t* xv = pass == 0 ? xs : zs;
            for (int ix = 0; ix < nx; ++ix) {
                for (uint16_t v : vs) {
                    const uint16_t x = xv[ix];
                    for (int t = 0; t < HWAVE; ++t) {
                        hin[t * 8 + 0] = x;
                        hin[t * 8 + 1] = 0x0000u;              // unused slot, kept for stride stability
                        hin[t * 8 + 2] = v;
                        const H2 pa = {(_Float16)f16_val(x), (_Float16)f16_val(x)};
                        const H2 pb = {(_Float16)f16_val(v), (_Float16)f16_val(v)};
                        const unsigned int ua = __builtin_bit_cast(unsigned int, pa);
                        const unsigned int ub = __builtin_bit_cast(unsigned int, pb);
                        std::memcpy(&hin[t * 8 + 4], &ua, 4);
                        std::memcpy(&hin[t * 8 + 6], &ub, 4);
                    }
                    hipMemcpy(din, hin, HWAVE * 8 * 2, hipMemcpyHostToDevice);
                    hipMemset(dout, 0, HWAVE * M_NOPS * 2);
                    k_micro<<<1, HWAVE>>>(din, dout, 0);
                    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
                        printf("LAUNCH/SYNC FAIL k_micro\n"); return 3;
                    }
                    hipMemcpy(hout, dout, HWAVE * M_NOPS * 2, hipMemcpyDeviceToHost);
                    // `o` is a COPY of the first launch's results, not a pointer into `hout`: the must-fire
                    // below launches again into `hout`, and a pointer alias would make the two sides of the
                    // comparison the same bytes (the check-that-agrees-with-itself family, one level down).
                    static unsigned short first[M_NOPS];
                    std::memcpy(first, hout, sizeof first);
                    const unsigned short* o = first;            // lane 0: every lane holds the same x, v
                    printf("x=%04x v=%04x | x+0=%04x x-0=%04x | fma(+0,+v,x)=%04x fma(+0,-v,x)=%04x | "
                           "fma(+0,+0,x)=%04x fma(-0,+0,x)=%04x | x+(0*v)=%04x/%04x fma(v,+0,x)=%04x | pk z+0=%04x z-0=",
                           x, v, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], hout[9]);
                    printf("%04x\n", hout[(size_t)1 * M_NOPS + 9]);
                    char kind[96];
                    const int nz = pass == 0;
                    ++pairs;
                    std::snprintf(kind, sizeof kind, "sec1 add: x+0 vs x-0  x=%04x", x);
                    { unsigned short a[1] = {o[0]}, b[1] = {o[1]}; measure(kind, cmp(std::vector<unsigned short>(a, a+1), std::vector<unsigned short>(b, b+1)), nz); }
                    std::snprintf(kind, sizeof kind, "sec1 fma: (+0,+v,x) vs (+0,-v,x) x=%04x v=%04x", x, v);
                    { unsigned short a[1] = {o[2]}, b[1] = {o[3]}; measure(kind, cmp(std::vector<unsigned short>(a, a+1), std::vector<unsigned short>(b, b+1)), nz); }
                    std::snprintf(kind, sizeof kind, "sec1 fma: (+0,+0,x) vs (-0,+0,x) x=%04x", x);
                    { unsigned short a[1] = {o[4]}, b[1] = {o[5]}; measure(kind, cmp(std::vector<unsigned short>(a, a+1), std::vector<unsigned short>(b, b+1)), nz); }
                    std::snprintf(kind, sizeof kind, "sec1 nonfused: x+(0,+v) vs x+(0,-v) x=%04x v=%04x", x, v);
                    { unsigned short a[1] = {o[6]}, b[1] = {o[7]}; measure(kind, cmp(std::vector<unsigned short>(a, a+1), std::vector<unsigned short>(b, b+1)), nz); }
                    std::snprintf(kind, sizeof kind, "sec1 packed pk_mul+pk_add z=+0 vs z=-0 x=%04x v=%04x", x, v);
                    { unsigned short a[1] = {hout[9]}, b[1] = {hout[(size_t)1 * M_NOPS + 9]};
                      measure(kind, cmp(std::vector<unsigned short>(a, a+1), std::vector<unsigned short>(b, b+1)), nz); }
                    // MUST-FIRE inside section 1: replace the zero ADDEND by a live small value. If this does
                    // not move, section 1 measures nothing (a zero operand that is not in the expression at all).
                    if (x == 0x3ECD && v == 0x3C00) {
                        // MUST-FIRE: replace the ZERO ADDEND with a live small value (`livez`), not the other
                        // operand of the product.
                        for (int t = 0; t < HWAVE; ++t) hin[t * 8 + 1] = 0x2C01u;   // live, same order as x
                        hipMemcpy(din, hin, HWAVE * 8 * 2, hipMemcpyHostToDevice);
                        hipMemset(dout, 0, HWAVE * M_NOPS * 2);
                        k_micro<<<1, HWAVE>>>(din, dout, 1);
                        if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) { printf("LAUNCH FAIL\n"); return 3; }
                        hipMemcpy(hout, dout, HWAVE * M_NOPS * 2, hipMemcpyDeviceToHost);
                        unsigned short a[3] = {o[2], o[6], o[0]}, b[3] = {hout[2], hout[6], hout[0]};
                        ctl("sec1 addend_live_vs_zero", "MUST-FIRE",
                            cmp(std::vector<unsigned short>(a, a + 3), std::vector<unsigned short>(b, b + 3)), true);
                        // And the same must-fire on the packed path: the pk arms of the livez launch must move.
                        unsigned short pa[1] = {first[9]}, pb[1] = {hout[9]};
                        ctl("sec1 addend_live_vs_zero_pk", "MUST-FIRE",
                            cmp(std::vector<unsigned short>(pa, pa + 1), std::vector<unsigned short>(pb, pb + 1)), true);
                    }
                }
            }
        }
        printf("sec1 operand pairs=%d (non-zero accumulators and the two signed zeros reported separately)\n", pairs);
        hipFree(din); hipFree(dout); delete[] hin; delete[] hout;
    }

    // ------------------------------------------------------------------ SECTION 2
    if (section == 0 || section == 2) {
        printf("\n== SECTION 2 - hand-built v_wmma_f16_16x16x16_f16_w32: one signed zero at dot position p\n");
        const uint16_t live = 0x3A00u;
        // FIDELITY must-fire: the clean tile must produce a non-zero, non-NaN D in BOTH issue shapes, or
        // nothing below adjudicates. The two-issue arm is checked separately because a NaN that the shape
        // itself produces (before any mutation) must not be read as an effect of a mutation.
        for (int ti : {0, 1}) {
            TileIn base; base_tiles(&base, live);
            const Bits clean = run_h(base, ti);
            int nonzero = 0, nnan = 0;
            for (unsigned short u : clean) { if (u != 0 && u != 0x8000u) nonzero++; if (f16_is_nan(u)) nnan++; }
            checks++;
            const bool ok = nonzero > 0 && nnan == 0;
            if (!ok) ctl_fail++;
            printf("%-30s %-9s ti=%d nonzero_cells=%d/%zu nan=%d  %s\n", "fid clean tile ran", "MUST-FIRE", ti,
                   nonzero, clean.size(), nnan,
                   ok ? "control ok" : "*** CONTROL NOT SATISFIED - this issue shape is not usable ***");
        }
        int positions = 0;
        for (int two_issue : {0, 1}) {
            for (int side : {0, 1}) {                       // 0 = B (P side), 1 = A (V side)
                for (int p = 0; p < 16; ++p) {
                    positions++;
                    TileIn plus, livet, ulpb, sub;   // livet = slot restored to its live base value; sub = underfloor
                    base_tiles(&plus, live);
                    base_tiles(&livet, live); base_tiles(&ulpb, live); base_tiles(&sub, live);
                    auto put = [&](TileIn* in, uint16_t bits) {
                        // The k axis is shared by BOTH A tuples (the engine's vlo and vhi are the same 16 KV
                        // positions at different output dims), so a V-side mutation lands at element p of each.
                        for (int t = 0; t < HWAVE; ++t) {
                            if (side == 0) in->b[t][p] = bits;
                            else { in->a[t][p] = bits; in->a2[t][p] = bits; }
                        }
                    };
                    // `plus` and `minus` differ in EXACTLY one bit: the sign of the zero at dot position p.
                    put(&plus, 0x0000u);
                    TileIn minus = plus; put(&minus, 0x8000u);
                    // `live` restores the slot's own base value: magnitude-appropriate, so a MUST-FIRE that
                    // cannot fail for the wrong reason (the first version substituted the F16 min subnormal,
                    // whose product underflows against an O(1) accumulator, and it reported a false breakage).
                    put(&livet, (side == 0) ? (uint16_t)(0x3A00u + (p & 7)) : (uint16_t)((p % 3 == 0) ? (live ^ 0x8000u) : (uint16_t)(live + (p & 3))));
                    put(&ulpb, 0x3A01u);                                    // 1 ULP off `livet`'s slot
                    TileIn dbl = livet;
                    auto putd = [&](uint16_t bits) {
                        for (int t = 0; t < HWAVE; ++t) {
                            if (side == 0) dbl.b[t][p] = bits;
                            else { dbl.a[t][p] = bits; dbl.a2[t][p] = bits; }
                        }
                    };
                    putd((side == 0) ? 0x4000u : 0x3C00u);   // double the slot's contribution: above resolution
                    put(&sub, 0x0001u);                                     // F16 min subnormal: the underfloor
                    // The hazard the engine actually reaches: P_dead = +0 AND V_dead = +-inf. One side alone
                    // gives +-inf, not NaN, and the first version of this control tested one side alone.
                    TileIn haz = plus;
                    for (int t = 0; t < HWAVE; ++t) {
                        haz.a[t][p] = 0x7C00u; haz.a2[t][p] = 0x7C00u; haz.b[t][p] = 0x0000u;
                    }
                    TileIn haz2 = haz;
                    for (int t = 0; t < HWAVE; ++t) { haz2.a[t][p] = 0xFC00u; haz2.a2[t][p] = 0xFC00u; }
                    const Bits b_plus = run_h(plus, two_issue);
                    const Bits b_minus = run_h(minus, two_issue);
                    const Bits b_live = run_h(livet, two_issue);
                    const Bits b_ulpb = run_h(ulpb, two_issue);
                    const Bits b_double = run_h(dbl, two_issue);
                    const Bits b_sub = run_h(sub, two_issue);
                    const Bits b_haz = run_h(haz, two_issue);
                    const Bits b_haz2 = run_h(haz2, two_issue);
                    const char* tag = side == 0 ? "B(P)" : "A(V)";
                    char kind[96];
                    std::snprintf(kind, sizeof kind, "zerosign %s p=%02d ti=%d", tag, p, two_issue);
                    measure(kind, cmp(b_plus, b_minus), 1);
                    std::snprintf(kind, sizeof kind, "slotlive %s p=%02d ti=%d", tag, p, two_issue);
                    ctl(kind, "MUST-FIRE", cmp(b_plus, b_live), true);
                    std::snprintf(kind, sizeof kind, "liveulp %s p=%02d ti=%d", tag, p, two_issue);
                    // INFORMATIONAL, not a control: one of 16 addends moved by 1 F16 ULP is below what a
                    // 16-term F16 dot can resolve, so it fires at some dot positions and not others (measured:
                    // 56 of 64). The MUST-FIRE for "this slot feeds D" is `slotlive`; the strengthened version
                    // of a 1-ULP claim is `livedouble` below, which is above the dot's resolution.
                    measure(kind, cmp(b_live, b_ulpb), 2);
                    std::snprintf(kind, sizeof kind, "livedouble %s p=%02d ti=%d", tag, p, two_issue);
                    ctl(kind, "MUST-FIRE", cmp(b_live, b_double), true);
                    std::snprintf(kind, sizeof kind, "underfloor %s p=%02d ti=%d", tag, p, two_issue);
                    measure(kind, cmp(b_plus, b_sub), 2);
                    std::snprintf(kind, sizeof kind, "infnan  %s p=%02d ti=%d", tag, p, two_issue);
                    {
                        const Diff d1 = cmp(b_plus, b_haz);
                        const Diff d2 = cmp(b_plus, b_haz2);
                        checks++;
                        const bool ok = d1.nnan > 0 && d2.nnan > 0;
                        if (!ok) ctl_fail++;
                        printf("%s %-9s +inf: ndiff=%5d nan=%4d/%4d | -inf: ndiff=%5d nan=%4d/%4d  %s\n", kind, "MUST-FIRE",
                               d1.ndiff, d1.nnanx, d1.nnan, d2.ndiff, d2.nnanx, d2.nnan,
                               ok ? "control ok (P=+0 x V=+-inf = NaN, the recorded hazard)" : "*** CONTROL NOT SATISFIED ***");
                    }
                }
            }
        }
        printf("sec2 positions=%d (2 two_issue x 2 sides x 16 dot positions; must be 64)\n", positions);
        if (positions != 64) { printf("ABORT: the position census is not the geometry the probe claims\n"); return 4; }
        // ------------------------------------------------------------------ the zero-sum sensitivity reading
        // Is this rig even ABLE to see a D that differs only in the sign of a zero? Arrange products that
        // cancel exactly: A[e] = +1 for every e; B[e] = +1 on even e and -1 on odd e (so 8x(+1) + 8x(-1) = 0
        // exactly in F16); then zero out one adjacent pair (q even, q+1 odd) on the B side, which removes one
        // +1 and one -1 and leaves 7x(+1) + 7x(-1) = 0 plus two zero products whose signs are the pair's.
        // C is all zeros, so D = the sum of those two zero products: +0 in one arm, -0 in the other, under
        // IEEE. If the two arms are byte-identical the zero products are not reaching the adder at all -
        // which would be a finding about the hardware, not a blind spot, and is printed as such.
        {
            TileIn z1, z2;
            for (int t = 0; t < HWAVE; ++t) {
                for (int e = 0; e < 16; ++e) {
                    z1.a[t][e] = 0x3C00u;             // +1.0
                    z2.a[t][e] = 0x3C00u;
                    z1.a2[t][e] = 0x3C00u;
                    z2.a2[t][e] = 0x3C00u;
                    z1.b[t][e] = (e & 1) ? 0xBC00u : 0x3C00u;   // -1 on odd, +1 on even
                    z2.b[t][e] = z1.b[t][e];
                    z1.c[t][e] = 0x0000u;             // C = 0
                    z2.c[t][e] = 0x0000u;
                }
            }
            const int q = 4;
            for (int t = 0; t < HWAVE; ++t) { z1.b[t][q] = 0x0000u; z1.b[t][q + 1] = 0x0000u;
                                              z2.b[t][q] = 0x8000u; z2.b[t][q + 1] = 0x8000u; }
            const Bits z1b = run_h(z1, 0), z2b = run_h(z2, 0);
            int nz1 = 0, nz2 = 0, nzero1 = 0, nzero2 = 0;
            for (size_t i = 0; i < z1b.size(); ++i) {
                if (z1b[i] == 0x0000u) nzero1++;
                if (z1b[i] == 0x8000u) nz1++;
                if (z2b[i] == 0x8000u) nz2++;
                if (z2b[i] == 0x0000u) nzero2++;
            }
            const Diff dz = cmp(z1b, z2b);
            meas++;
            printf("%-44s %-28s ndiff=%5d | D bits: arm(+0)=%d plus-zero %d minus-zero | arm(-0)=%d plus-zero %d minus-zero -> %s\n",
                   "zsum cancelling products, pair +0 vs -0", "SENSITIVITY", dz.ndiff, nzero1, nz1, nzero2, nz2,
                   dz.ndiff ? "the rig CAN see a signed-zero D (the null above is not a blind spot)"
                            : "a signed-zero D is invisible here OR the zero terms are not formed");
        }
        // The F32-accumulator sibling, one reading: does a signed zero reach an F32 D the way it might an F16 one?
        {
            TileIn base; base_tiles(&base, live);
            for (int p : {4, 5}) {
                TileIn m1 = base, m2 = base;
                for (int t = 0; t < HWAVE; ++t) { m1.b[t][p] = 0x0000u; m2.b[t][p] = 0x8000u; }
                std::vector<unsigned int> d1 = run_f32(m1), d2 = run_f32(m2);
                int nd = 0;
                for (size_t i = 0; i < d1.size(); ++i) if (d1[i] != d2[i]) nd++;
                meas++;
                printf("%-44s %-28s ndiff=%d/%zu %s\n", "f32D zerosign B-side", "MEASURE", nd, d1.size(),
                       nd ? "<== SIGN REACHES THE RESULT" : "no difference");
            }
        }
    }

    printf("\nSUMMARY controls=%d control_failures=%d measures=%d sign_reaches_result=%d\n",
           checks, ctl_fail, meas, leak);
    printf("VERDICT: %s\n", ctl_fail ? "CONTROLS FAILED - nothing in this run adjudicates anything"
           : (leak ? "A SIGNED ZERO REACHES THE F16 RESULT (see the lines marked SIGN REACHES)"
                   : "NO SIGNED-ZERO SENSITIVITY at add level or in the hand-built tile (measured null, controls fired)"));
    return ctl_fail ? 1 : 0;
}
