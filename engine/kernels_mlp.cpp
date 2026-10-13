// Launchers for prototype/mlp_gemv: the fused gate+up+SwiGLU dot and the down projection (§03.3).
// Its own TU — mlp_gemv defines `enum`-free NEG_* macros, HID, FF, FULL_MASK and its own bf/fb/rnd/
// reduce_row, all of which collide with proj_gemv's and lm_head_gemv's spellings if merged.
#include "kernels.h"

#include <hip/hip_runtime.h>

#include <stdexcept>

#include "engine/launch_stats.h"
#include "prototype/mlp_gemv/mlp_gemv_kernels.h"

namespace engine::kernels {
namespace {
// Same warp-width rule as `proj_gemv`: the kernel maps rows as `blockIdx.x*rpb + threadIdx.x/ws`,
// so the block must be rows-per-block × THIS device's warp width. The host of the gated prototype
// reads it from `hipGetDeviceProperties` and refuses anything but 32 or 64; querying here is the
// engine doing the same thing rather than assuming gfx1151 forever.
int warp_width() {
    static int w = [] {
        int v = 32;
        (void)hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, 0);
        return v > 0 ? v : 32;
    }();
    return w;
}

// ROWS_PER_BLOCK — the default in `mlp_gemv.hip.cpp:97`, and the geometry the F1–F4 gate ran at
// (`run.py --layers all --fixture dec8 --no-time --tile 0 --negative`: `--tile 0` turns off the
// cache-tiling experiment path, leaving rows/block at its default). `prototype/mlp_gemv/results.md`
// sweeps 4→32 and finds warm rankings move by 4 % with a different winner, so the gated default is
// the honest constant until a cold sweep on the engine's own plan says otherwise.
constexpr int kRowsPerBlock = ROWS_PER_BLOCK;

void check(const char* what, int rows, int in) {
    if (rows <= 0 || in <= 0) {
        throw std::runtime_error(std::string("mlp: ") + what +
                                 " called with rows=" + std::to_string(rows) +
                                 " in=" + std::to_string(in));
    }
}

// The token tile of the batched forms (P4): shared memory is TPB*IN floats, and the gated pair is
// TPB=4 at IN=1024 (16 KiB) and TPB=2 at IN=3584 (28 KiB) — see `prototype/mlp_gemv/results.md`
// §P4. Any other width has no gated instantiation, so it is refused rather than guessed at.
template <int IN>
constexpr int tokens_per_block_for() {
    return IN == HID ? 4 : 2;
}

void check_tokens(const char* what, int tokens) {
    if (tokens < 1) {
        throw std::runtime_error(std::string("mlp: ") + what +
                                 " called with tokens=" + std::to_string(tokens) +
                                 "; a wave has at least one token in it");
    }
}

}  // namespace

MlpGateMap parse_mlp_gate_map(const std::string& name) {
    if (name == "auto") return MlpGateMap::kAuto;
    if (name == "interleave") return MlpGateMap::kInterleave;
    if (name == "contig") return MlpGateMap::kContig;
    throw std::runtime_error("--mlp-map takes auto|interleave|contig, got '" + name + "'");
}

const char* mlp_gate_map_name(MlpGateMap map) {
    if (map == MlpGateMap::kAuto) return "auto";
    return map == MlpGateMap::kContig ? "contig" : "interleave";
}

MlpGateMap resolve_mlp_gate_map(MlpGateMap want, int rows) {
    // An explicit walk wins outright: `--mlp-map` is the A/B instrument and a row-count default must
    // not leak into a run that named one side of it. Otherwise the body's row count decides, and the
    // split is the measured one in `kernels.h` — one row contiguous, a wave interleaved.
    if (want != MlpGateMap::kAuto) return want;
    return rows == 1 ? MlpGateMap::kContig : MlpGateMap::kInterleave;
}

// The kernel spells the two walks `MAP_INTERLEAVE`/`MAP_CONTIG` (`mlp_gemv_kernels.h:MlpMap`), and
// this is the one place the launcher's `MlpGateMap` becomes that. It is a TEMPLATE argument chosen by
// a host-side branch, so each walk compiles as if it were the only one: the two sides of the A/B are
// two instantiations, never one kernel with a test inside its row loop.
template <int IN, int MAP>
static void launch_gateup_single_kernel(const uint16_t* w_gate, const uint16_t* w_up, const float* x,
                                        uint16_t* act, int rows, int threads, int blocks) {
    ::mlp_gateup<IN, MAP><<<blocks, threads>>>((const hip_bfloat16*)w_gate,
                                               (const hip_bfloat16*)w_up, x, (hip_bfloat16*)act,
                                               nullptr, rows, kRowsPerBlock, NEG_NONE);
}

template <int IN>
static void launch_gateup_single(const uint16_t* w_gate, const uint16_t* w_up, const float* x,
                                 uint16_t* act, int rows, int threads, int blocks,
                                 MlpGateMap map) {
    if (map == MlpGateMap::kContig) {
        launch_gateup_single_kernel<IN, MAP_CONTIG>(w_gate, w_up, x, act, rows, threads, blocks);
    } else {
        launch_gateup_single_kernel<IN, MAP_INTERLEAVE>(w_gate, w_up, x, act, rows, threads,
                                                       blocks);
    }
}

// The batched launch: grid.x is the row blocks and grid.y the token tiles, so a weight row is read
// once per tile. Same per-output-element order as the single-token kernel, which is what makes the
// two interchangeable for a given (row, token) — gated bit for bit in `prototype/mlp_gemv/run.py
// --tokens p64`.
template <int IN>
static void launch_gateup_batched(const uint16_t* w_gate, const uint16_t* w_up, const float* x,
                                  uint16_t* act, int rows, int tokens, int threads,
                                  MlpGateMap map) {
    constexpr int TPB = tokens_per_block_for<IN>();
    const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock,
                    (unsigned)((tokens + TPB - 1) / TPB));
    // Same template-argument choice as the single-token form above.
    if (map == MlpGateMap::kContig) {
        ::mlp_gateup_tokens<IN, TPB, MAP_CONTIG><<<grid, threads>>>(
            (const hip_bfloat16*)w_gate, (const hip_bfloat16*)w_up, x, (hip_bfloat16*)act, nullptr,
            rows, tokens, kRowsPerBlock, NEG_NONE);
    } else {
        ::mlp_gateup_tokens<IN, TPB, MAP_INTERLEAVE><<<grid, threads>>>(
            (const hip_bfloat16*)w_gate, (const hip_bfloat16*)w_up, x, (hip_bfloat16*)act, nullptr,
            rows, tokens, kRowsPerBlock, NEG_NONE);
    }
    launchstats::mlp_gateup().record(tokens);
}

template <int IN>
static void launch_down_batched(const uint16_t* w_down, const float* act, uint16_t* out, int rows,
                                int tokens, int threads) {
    constexpr int TPB = tokens_per_block_for<IN>();
    const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock,
                    (unsigned)((tokens + TPB - 1) / TPB));
    ::mlp_down_tokens<IN, TPB><<<grid, threads>>>((const hip_bfloat16*)w_down, act,
                                                  (hip_bfloat16*)out, rows, tokens, kRowsPerBlock,
                                                  NEG_NONE);
    launchstats::mlp_down().record(tokens);
}

void mlp_gateup(const uint16_t* w_gate, const uint16_t* w_up, const float* x, uint16_t* act,
                int rows, int in, int tokens, MlpGateMap map) {
    if (map == MlpGateMap::kAuto) {
        // Would otherwise fall into the `else` branch below and launch `interleave` silently, i.e. an
        // A/B row priced with the wrong walk and nothing printed to say so.
        throw std::runtime_error("mlp_gateup: got kAuto, which is a request and not a walk; call "
                                 "resolve_mlp_gate_map(want, rows) at the call site");
    }
    check("gateup", rows, in);
    check_tokens("gateup", tokens);
    const int threads = kRowsPerBlock * warp_width();
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    // `pre` (the raw gate/up dumps the gate's F1 stage compares) stays null: the engine never
    // needs the two intermediates, and writing them would put 2×3584 BF16 per layer on the token's
    // write traffic for a dump no gate here reads.
    if (tokens > 1) {
        if (in == HID) {
            launch_gateup_batched<HID>(w_gate, w_up, x, act, rows, tokens, threads, map);
        } else if (in == FF) {
            launch_gateup_batched<FF>(w_gate, w_up, x, act, rows, tokens, threads, map);
        } else {
            throw std::runtime_error("mlp_gateup: input width " + std::to_string(in) +
                                     " is not one of the two shapes the gated kernel"
                                     " instantiates");
        }
        return;
    }
    if (in == HID) {
        launch_gateup_single<HID>(w_gate, w_up, x, act, rows, threads, blocks, map);
    } else if (in == FF) {
        // Instantiated by the gated host too (mlp_gemv.hip.cpp:249). Unused by this checkpoint's
        // MLP shapes — gate/up are [3584,1024] — but the kernel is one template line and a second
        // entry point would be a worse way to say "this shape exists".
        launch_gateup_single<FF>(w_gate, w_up, x, act, rows, threads, blocks, map);
    } else {
        throw std::runtime_error("mlp_gateup: input width " + std::to_string(in) +
                                 " is not one of the two shapes the gated kernel instantiates");
    }
    // The single-token path, counted as the one token it launched. Batched and single go through the
    // same counter on purpose: the failure `engine/launch_stats.h` exists to catch is a wave that
    // only ever took this branch, and that is invisible unless this branch is counted too.
    launchstats::mlp_gateup().record(1);
}

void mlp_down(const uint16_t* w_down, const float* act, uint16_t* out, int rows, int in,
              int tokens) {
    check("down", rows, in);
    check_tokens("down", tokens);
    const int threads = kRowsPerBlock * warp_width();
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    if (tokens > 1) {
        if (in == FF) {
            launch_down_batched<FF>(w_down, act, out, rows, tokens, threads);
        } else if (in == HID) {
            launch_down_batched<HID>(w_down, act, out, rows, tokens, threads);
        } else {
            throw std::runtime_error("mlp_down: input width " + std::to_string(in) +
                                     " is not one of the two shapes the gated kernel"
                                     " instantiates");
        }
        return;
    }
    // The MLP's second dot is 3584 wide (FF), which is why `in` cannot be pinned to 1024/2048 like
    // proj_gemv's — §01.3: down_proj is [1024, 3584].
    if (in == FF) {
        ::mlp_down<FF><<<blocks, threads>>>((const hip_bfloat16*)w_down, act,
                                            (hip_bfloat16*)out, rows, kRowsPerBlock, NEG_NONE);
    } else if (in == HID) {
        ::mlp_down<HID><<<blocks, threads>>>((const hip_bfloat16*)w_down, act,
                                             (hip_bfloat16*)out, rows, kRowsPerBlock, NEG_NONE);
    } else {
        throw std::runtime_error("mlp_down: input width " + std::to_string(in) +
                                 " is not one of the two shapes the gated kernel instantiates");
    }
    launchstats::mlp_down().record(1);   // see the note in `mlp_gateup`
}

}  // namespace engine::kernels
