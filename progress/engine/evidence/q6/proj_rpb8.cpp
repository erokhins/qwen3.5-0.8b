// Launcher for prototype/proj_gemv: one templated BF16 GEMV for all nine projections (§04.1,
// §05.1), in ONE form — the token-tiled kernel, at a tile width of 1 for a decode step.
// Its own TU — proj_gemv defines its own `enum Neg`, bf/fb/rnd and FULL_MASK.
#include "kernels.h"

#include <hip/hip_runtime.h>

#include <stdexcept>

#include "engine/launch_stats.h"
#include "prototype/proj_gemv/proj_gemv_kernels.h"

namespace engine::kernels {
namespace {
// The kernel indexes rows as `blockIdx.x*rpb + threadIdx.x/warpSize`, so the block width must be
// rows-per-block × THIS device's warp width (32 on gfx1151). Querying it is what each prototype's
// host does before launching; hardcoding 32 here would silently halve the rows on a wave64 target.
int warp_width() {
    static int w = [] {
        int v = 32;
        (void)hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, 0);
        return v > 0 ? v : 32;
    }();
    return w;
}

// The token tile of the batched form (P5). Shared memory is TPB*IN floats and 16 KiB/block is the
// budget `prototype/mlp_gemv` settled its gated batched pair on, so 4 rows at in=1024 and 2 at
// in=2048. A wave with ONE token uses TPB = 1: that is the k=1 launch shape, with the k=1 kernel's
// 4 or 8 KiB of shared rather than a tile nobody reads. One kernel body at two tile widths, which is
// the point — P4's MLP launcher branches between two different kernels and this one does not.
// TPB is unmeasured, like the MLP's: `prototype/proj_gemv/results.md` says what sweep it wants and
// why that sweep belongs to the main agent.
template <int IN>
constexpr int tpb_batch() {
    return IN == 1024 ? 4 : 2;
}

// One submission. grid.x is the row blocks — exactly the k=1 grid — and grid.y the token tiles, so a
// decode step still launches (rows/rpb, 1) and a 295-token wave launches (rows/rpb, 74).
template <int IN, int TPB>
void launch(const void* weight, const float* x, uint16_t* out, int rows, int tokens, int threads) {
    constexpr int kRowsPerBlock = 8;   // the prototype's default, and its gated width
    const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock,
                    (unsigned)((tokens + TPB - 1) / TPB));
    ::proj_gemv_tokens<IN, TPB, false><<<grid, threads>>>(
        (const hip_bfloat16*)weight, x, (hip_bfloat16*)out, rows, tokens, kRowsPerBlock, NEG_NONE);
    // Counted after the launch, and counted on BOTH widths on purpose: the failure
    // `engine/launch_stats.h` exists to catch is a wave that only ever took the one-token branch,
    // and that is invisible unless the one-token branch reports itself too.
    launchstats::proj().record(tokens);
}
}  // namespace

// `tokens` is P5's token-batched form (§06.2's "one GEMM per projection"): `x` is then a
// [tokens, in] tile and `out` a [tokens, rows] tile, and each weight row is read once per tile
// instead of once per token. `tokens == 1` issues the same kernel at TPB = 1, which is bit-for-bit
// the shape this launcher issued before P5 — `prototype/proj_gemv/run.py --tokens` requires the two
// to agree exactly (138 400 elements per fixture, 0 differing) and `gate.py` section (h) requires the
// wave to be the one that batched.
//
// PACK_X = false — the configuration this op's 136-comparison gate ran (`run.py` adds `--x-packed`
// only on the timing path, so `<IN, ..., true>` was never compared against a pin), and the
// configuration the measurement prefers: `results.md` times the packed form at 3.269 ms / 60 % of
// probe against 2.991 ms / 66 % for this one over the 48-launch token plan, and hands W5
// "`PACK_X = false`, measured above" as the setting to take. Re-measured in the 2026-10-11 clean
// window at rows/block 32: **unpacked 0.0528 ms/launch (65.1 % of the read probe) against packed
// 0.0593 ms/launch (58.0 %)**, unpacked ahead at every rows/block swept — the ordering holds under a
// clean window, so `false` is a measured choice and not an inherited one (`engine/results.md` E2,
// closed).
//
// The consequence decode_step must know: with PACK_X=false the kernel reads `x` as `const float*`, so
// the caller hands F32 activations, NOT the BF16 pointer this launcher used to take. That is
// proj_gemv's own recommendation — "the engine should hand this kernel the F32 norm output and skip
// the round-trip" — but since glue writes BF16 norms, ≈48 casts/token come back (≈0.089 ms at the
// measured 1.85 µs launch floor, `tools/launch_overhead_hip.cpp`). Still the cheaper side: 0.089 ms
// of casts against a measured +0.279 ms kernel penalty. Products and accumulation order are
// bit-identical in the two forms (every BF16 is exact in F32), so this is a performance choice, not a
// numerics one. What is still unmeasured is the A/B through THIS launcher on the engine's own buffer
// plan — a token-level timing, listed as the open half of E2 in `engine/results.md`.
//
// `in` is 1024 or 2048 for all nine shapes in this checkpoint (§01.2/§01.3); anything else is a bug in
// the caller, not a shape to instantiate.
void proj_gemv(const void* weight, const float* x, uint16_t* out, int rows, int in, int tokens) {
    if (rows <= 0 || tokens < 1) {
        throw std::runtime_error("proj_gemv: rows=" + std::to_string(rows) +
                                 " tokens=" + std::to_string(tokens) +
                                 "; a wave has at least one token in it");
    }
    const int threads = 8 * warp_width();
    if (in == 1024) {
        if (tokens == 1) {
            launch<1024, 1>(weight, x, out, rows, tokens, threads);
        } else {
            launch<1024, tpb_batch<1024>()>(weight, x, out, rows, tokens, threads);
        }
    } else if (in == 2048) {
        if (tokens == 1) {
            launch<2048, 1>(weight, x, out, rows, tokens, threads);
        } else {
            launch<2048, tpb_batch<2048>()>(weight, x, out, rows, tokens, threads);
        }
    } else {
        // No launch: the caller asked for a shape the gated kernel does not have.
        // decode_step asserts instead; a silent wrong result here would be invisible.
    }
}

}  // namespace engine::kernels
