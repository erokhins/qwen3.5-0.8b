/*
 * equiv_check.cpp — route S1 item (1)'s CPU proof that folding the widening into the consumer changes
 * no operand VALUE at any index, for every folded site. Host-only: no HIP call, no device, no timing.
 *
 * WHAT IS BEING CLAIMED, AND WHAT THIS FILE CAN AND CANNOT PROVE IT WITH.
 *
 * The fold deletes `kernels_cast.cpp`'s widening of a BF16 activation row into an F32 scratch tile and
 * makes the consumer stage the BF16 itself, widening each element in registers at the use site. The
 * claim that makes that free is: for every site, at every index, at every BF16 value, the float the
 * consumer now computes is BIT-IDENTICAL to the float the deleted launch wrote into the buffer it used
 * to read. This file checks exactly that, in three layers:
 *
 *   [A] the widening itself is exact. `float(hip_bfloat16)` — the conversion BOTH spellings call — must
 *       reproduce the F32 whose bits are the BF16's shifted up by 16, for all 65,536 bit patterns:
 *       a widening is a bit position, not a rounding. A truncating or RNE-to-F32 implementation fails
 *       here, and NaNs are checked as NaN-vs-NaN because a payload is not a value. This is the layer
 *       with real content in it, and it is exhaustive rather than sampled.
 *   [B] per site, per position in that site's tile, the two spellings agree bit for bit: the shipped
 *       path (a host transcription of `kernels_cast.cpp`'s kernel body, writing an F32 buffer the way
 *       the launch does, then reading index i the way the consumer does) against the folded path (the
 *       in-register widen expression the folded kernels use). Transcribing the shipped cast TU as a
 *       separate artifact is the point: if the two ever stop being the same function, [B] goes red.
 *   [C] the population the check covers equals the population the fold claims: six sites, 97 launches a
 *       token. A check whose site list can be empty cannot fail, so both totals are counted, printed,
 *       and asserted against the expected values before the exit code is decided.
 *
 * WHAT THIS FILE DOES NOT PROVE, stated because the tree has been bitten by exactly this: it cannot
 * witness the DEVICE codegen of the two spellings. Two host mirrors of one expression agree with each
 * other, which is not the same claim as a compiled kernel agreeing with a compiled kernel —
 * `AGENTS.md`'s "a hoisted scalar constant is a different number, and a CPU mirror is blind to it" row
 * is that trap, and a register convert inside a dot's expression tree is exactly the class of thing it
 * covers. Bit-identity of the OUTPUTS is owed to the device byte-compare: `engine/gate.py` on the pinned
 * fixtures with `QWEN_CAST_FOLD=1`, and `prototype/cast_fold/ab_item1.sh`'s `sampled.txt` diff at both
 * 128 and 4,032 lanes. This file narrows what that device pass can be wrong about; it does not replace it.
 *
 * build and run:
 *   hipcc -O2 -I. -Iengine prototype/cast_fold/equiv_check.cpp -o prototype/cast_fold/equiv_check
 *   prototype/cast_fold/equiv_check            # or --verbose for every site
 */
#include <hip/hip_bfloat16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_verbose = 0;
int g_failures = 0;

void fail(const std::string& what) {
    ++g_failures;
    if (g_failures <= 40) std::printf("  FAIL %s\n", what.c_str());
}

// ---- the two spellings, transcribed from the sources they stand in for -------------------------
// Transcribed from `engine/kernels_cast.cpp`'s `cast_bf16_to_f32_kernel`: `dst[i] = float(src[i])`,
// elementwise, one element per thread, nothing else. The thread/block geometry is not arithmetic and
// is deliberately not mirrored; what is mirrored is the conversion and the index.
void shipped_cast_bf16_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) {
        hip_bfloat16 b;
        std::memcpy(&b, &src[i], sizeof(b));
        dst[i] = float(b);
    }
}

// The folded consumer's expression: `float(xs[k])` where `xs` is a BF16 stage row. Same conversion,
// applied at the use site instead of in a separate pass over memory.
float folded_widen(const uint16_t* src, int i) {
    hip_bfloat16 b;
    std::memcpy(&b, &src[i], sizeof(b));
    return float(b);
}

uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}
bool is_nan(float f) { return f != f; }

// [A] exactness of the widening over the whole BF16 domain.
void check_widen_exact() {
    long long checked = 0, nan_cases = 0;
    for (uint32_t b = 0; b <= 0xFFFFu; ++b) {
        const uint16_t bits = (uint16_t)b;
        float got = folded_widen(&bits, 0);
        float shipped;
        {
            float buf;
            shipped_cast_bf16_to_f32(&bits, &buf, 1);
            shipped = buf;
        }
        if (bits_of(got) != bits_of(shipped)) {
            fail("widen differs from the shipped cast for bits 0x" +
                 std::to_string(b));
        }
        // The independent oracle: an exact widening moves no bits of the payload and no exponent
        // bias, so the F32 is the BF16 bit pattern shifted up 16. Anything that rounds would fail.
        const uint32_t oracle = b << 16;
        if (is_nan(got)) {
            if (is_nan(shipped)) {
                ++nan_cases;
            } else {
                fail("NaN input produced a non-NaN widening");
            }
        } else if (bits_of(got) != oracle) {
            fail("widen of bits 0x" + std::to_string(b) + " is not the bit-shift widening: got 0x" +
                 std::to_string(bits_of(got)) + " want 0x" + std::to_string(oracle));
        }
        ++checked;
    }
    std::printf("  [A] widening exactness: %lld of 65536 BF16 patterns checked (NaN-as-NaN: %lld)\n",
                checked, nan_cases);
    if (checked != 65536) fail("widening check ran over " + std::to_string(checked) + " patterns");
}

// The folded population. `launches` is the per-decode-token count `census_sites.py` derives from
// engine/decode_step.cpp and engine/state.h; it is repeated here as the number this file asserts it
// covered, and `check_equiv.py` re-checks both against the census, so neither file is trusted alone.
struct Site {
    const char* name;
    const char* consumer;
    int elems_per_call;    // elements one widening covers
    int launches_per_step;
};
const Site kSites[] = {
    {"norm_for_input_proj",     "proj_rows_xbf16 / proj_gemv_xbf16", 1024, 24},
    {"attn_gated_for_o_proj",   "proj_gemv_xbf16 (o_proj)",          2048,  6},
    {"gdn_out_for_out_proj",    "proj_gemv_xbf16 (out_proj)",        2048, 18},
    {"norm_for_mlp_gateup",     "mlp_gateup_xbf16",                  1024, 24},
    {"mlp_act_for_mlp_down",    "mlp_down_xbf16",                    3584, 24},
    {"final_norm_for_lm_head",  "lm_head_full_xbf16",                1024,  1},
};
constexpr int kSiteCount = (int)(sizeof kSites / sizeof kSites[0]);
constexpr int kExpectedSites = 6;
constexpr int kExpectedLaunches = 97;      // engine/generation-limits.md §1, [Q1]

// [B] per site: every position of the tile, every BF16 value through position 0.
void check_sites() {
    long long comparisons = 0;
    int launches_covered = 0;
    int sites_checked = 0;

    for (const Site& s : kSites) {
        ++sites_checked;
        launches_covered += s.launches_per_step;
        std::vector<uint16_t> tile((size_t)s.elems_per_call);
        std::vector<float> cast_tile((size_t)s.elems_per_call);

        // (i) the tile as a decode step would see it: a dense sweep of the BF16 domain, phase-shifted
        // per site so a copy/paste of one site's buffer cannot stand in for another's.
        for (int i = 0; i < s.elems_per_call; ++i) {
            tile[i] = (uint16_t)((uint32_t)i * 2654435761u + (uint32_t)(sites_checked * 7919));
        }
        shipped_cast_bf16_to_f32(tile.data(), cast_tile.data(), s.elems_per_call);
        for (int i = 0; i < s.elems_per_call; ++i) {
            if (bits_of(cast_tile[i]) != bits_of(folded_widen(tile.data(), i))) {
                fail(std::string(s.name) + " position " + std::to_string(i) +
                     ": the folded widen is not the value the cast buffer holds");
            }
            ++comparisons;
        }

        // (ii) every representable BF16 value, pushed through position 0 of this site's tile. This is
        // the exhaustive version: a value that widens differently (a subnormal, a NaN payload, a −0)
        // is caught here even if the sweep above never generated it.
        for (uint32_t b = 0; b <= 0xFFFFu; ++b) {
            tile[0] = (uint16_t)b;
            float cast_one;
            shipped_cast_bf16_to_f32(tile.data(), &cast_one, 1);
            const float folded_one = folded_widen(tile.data(), 0);
            const bool same = bits_of(cast_one) == bits_of(folded_one);
            const bool both_nan = is_nan(cast_one) && is_nan(folded_one);
            if (!same && !both_nan) {
                fail(std::string(s.name) + " with operand bits 0x" + std::to_string(b) +
                     ": cast gives 0x" + std::to_string(bits_of(cast_one)) +
                     ", folded widen gives 0x" + std::to_string(bits_of(folded_one)));
            }
            ++comparisons;
        }
        if (g_verbose) {
            std::printf("  [B] %-24s consumer=%-36s elems/call=%5d launches=%3d\n", s.name,
                        s.consumer, s.elems_per_call, s.launches_per_step);
        }
    }

    std::printf("  [B] per-site operand equality: %lld comparisons over %d sites\n", comparisons,
                sites_checked);
    // [C] — and a check over an empty list is not a check.
    if (sites_checked == 0) fail("no sites were checked");
    if (sites_checked != kExpectedSites) {
        fail("checked " + std::to_string(sites_checked) + " sites, the published foldable count is " +
             std::to_string(kExpectedSites));
    }
    if (launches_covered != kExpectedLaunches) {
        fail("the sites cover " + std::to_string(launches_covered) +
             " launches/token, the §07 cast bill is " + std::to_string(kExpectedLaunches));
    }
    std::printf("SITES_CHECKED=%d LAUNCHES_COVERED=%d COMPARISONS=%lld\n", sites_checked,
                launches_covered, comparisons);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--verbose")) g_verbose = 1;
    }
    std::printf("equiv_check: host-side equivalence for route S1 item (1)'s cast fold\n");
    check_widen_exact();
    check_sites();
    if (g_failures) {
        std::printf("EQUIV FAIL: %d failing checks\n", g_failures);
        return 1;
    }
    std::printf("EQUIV PASS: the folded widen is the value the deleted cast wrote, at every index and "
                "every BF16 value, for all %d folded sites (%d launches/token)\n",
                kSiteCount, kExpectedLaunches);
    return 0;
}
