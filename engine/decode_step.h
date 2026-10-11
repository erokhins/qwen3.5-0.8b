#pragma once
// One greedy decode step (§07.1), assembled only out of the launchers in `kernels.h`.
//
// The order of everything in `decode_step.cpp` is §07.1's, and the arithmetic inside each launch is
// the graduated kernel's — nothing here computes a value. What this file does own is the buffer plan,
// the norm/residual pairing, and the dtype seam (which activation has to be widened before which
// kernel will accept it), and those are the parts worth reading here rather than in the kernels.
//
// The §03.2 pairing, which is the one thing a reader can get quietly wrong: layer `l`'s mixer input
// norm has NO residual add in front of it (layer 0's is a norm-only call), while the add after the
// mixer and the add after the MLP are each fused with the NEXT norm — the post-attention norm
// carries the mixer's add, and the next layer's input norm carries the MLP's. That is `prototype/glue`'s
// gated pairing verbatim (`glue_pins.py:w_g1/w_g2/w_in`: call 2 of layer 23 uses `lm.norm.weight`),
// and it is what makes the whole token 1 + 24×2 = 49 glue launches with the final norm falling out
// of the last one for free.

#include <cstdint>
#include <vector>

#include "engine/state.h"

namespace engine {

class WeightStore;

// Per-step working set. Everything here is one token's scratch, reused by every layer of every step;
// nothing in it survives a call to `decode_step`. Sizes are the model's own constants from `state.h`,
// so a shape change cannot leave a short buffer behind. The layout is documented in decode_step.cpp
// next to the buffers' use, because which kernel wants which dtype is the reason each one exists.
struct DecodeScratch {
    // Residual ping-pong. NOT aliased into one buffer: `glue_add_norm`'s `s` and `s_out` are both
    // `__restrict__`, so writing the new residual over the one it is reading is a contract violation
    // even though the same thread touches the same index.
    uint16_t* s_a = nullptr;        // [1024]
    uint16_t* s_b = nullptr;        // [1024]
    uint16_t* norm = nullptr;       // [1024] the glue norm output, BF16 (what the pins carry)
    float* norm_f32 = nullptr;      // [1024] the same row widened, for proj_gemv / mlp_gateup
    uint16_t* embed = nullptr;      // [1024] E[y], the residual layer 0 starts from
    uint16_t* delta = nullptr;      // [1024] a block's or the MLP's residual addend
    uint16_t* block_out = nullptr;  // [2048] gated_out / gated, i.e. the 2048-wide proj operand
    float* block_out_f32 = nullptr;  // [2048] the same, widened, for out_proj / o_proj

    // GDN (§07.2). `raw`/`u` are 6144 wide because in_proj_qkv is [q|k|v]; `a`/`b`/`g`/`beta` are the
    // 16 value heads. `conv_new` is the rolled window — see the conv note in decode_step.cpp.
    uint16_t* g_raw = nullptr;      // [6144] in_proj_qkv out, BF16 as the GEMV wrote it
    float* g_raw_f32 = nullptr;     // [6144]
    uint16_t* g_z = nullptr;        // [2048] in_proj_z out, stays BF16 (§04.5 reads it as BF16)
    uint16_t* g_a = nullptr;        // [16]   in_proj_a out
    uint16_t* g_b = nullptr;        // [16]   in_proj_b out
    float* g_a_f32 = nullptr;       // [16]
    float* g_b_f32 = nullptr;       // [16]
    float* g_dt_bias = nullptr;     // [kNumGdnLayers][16]: dt_bias is BF16 in the store (§01.6
                                    // keeps only A_log and linear_attn.norm.weight F32) and
                                    // gates_kernel reads `const float*`, so it is widened ONCE,
                                    // by prepare_weight_adapters, per GDN slot.
    float* g_conv_w = nullptr;      // [kNumGdnLayers][kConvChannels][kConvKernel]: same reason,
                                    // same one-time widening (conv_kernel takes `const float* C`).
    float* g_g = nullptr;           // [16] decay log
    float* g_beta = nullptr;        // [16]
    float* g_u = nullptr;           // [6144] post-conv, q|k|v in one row (§07.2 slices it)
    float* g_o = nullptr;           // [2048] the recurrence's o, F32 (§04.5's input dtype)
    float* conv_new = nullptr;      // [kConvWindow][kConvChannels]

    // Attention (§07.3). The three stage buffers are required, not optional — `kernels_attn.cpp`
    // says why (the cache append lives behind the same null check as one of them).
    uint16_t* a_q = nullptr;        // [4096] q_proj out: [8 heads][q 256 | gate 256], §05.1
    uint16_t* a_k = nullptr;        // [512]
    uint16_t* a_v = nullptr;        // [512]
    uint16_t* a_q_rope = nullptr;   // [2048]
    uint16_t* a_k_rope = nullptr;   // [512]
    uint16_t* a_gate = nullptr;     // [2048] the de-interleaved output gate, §05.6
    uint16_t* a_pre = nullptr;      // [2048] attn_out_pre_gate, §05.5
    uint16_t* a_gated = nullptr;    // [2048] pre * σ(gate), o_proj's operand

    // MLP (§03.3). `act` is the F32 intermediate `down_proj` wants, `mlp_act` the BF16 one the
    // gate is pinned against; both exist because the two kernels disagree about the dtype.
    uint16_t* mlp_act = nullptr;    // [3584]
    float* mlp_act_f32 = nullptr;   // [3584]

    float* logits = nullptr;        // [kVocab] F32 — §03.4's full-logit form, see kernels_lmhead.cpp

    // Set by `prepare_weight_adapters`, which is where `g_dt_bias`/`g_conv_w` get filled. Zeroed by
    // `allocate_scratch`, so a forgotten call is a refusal rather than a read of uninitialised memory.
    bool adapters_ready = false;
};

DecodeScratch allocate_scratch();
void free_scratch(DecodeScratch& sc);

struct StepResult {
    int token = -1;            // §03.4 argmax; ties go to the LOWEST id (strict `>` on the host)
    float top_logit = 0.0f;    // the value at the argmax, for the log and the gate's report
};

// §07.5: greedy decoding stops at `<|im_end|>` (A-10), at end of context, or at the caller's
// budget. `decode_step` sets `stopped` for the first; the other two are the loop's business.
constexpr int kStopToken = 248046;

// One token through the whole stack: input norm → 24 blocks (GDN §07.2 or attention §07.3 by layer
// type, each followed by the MLP §03.3 and the fused add+norm) → the final norm falling out of the
// last glue call → lm_head → host argmax. Submits ~423 launches and syncs once, at the logits.
//
// `st` is mutated in place: its GDN state, conv window and KV cache advance, `prefix_len` increments,
// and `sampled_token`/`stopped` are set (§07.5). Throws std::runtime_error on a HIP failure, on a
// position the gated kernels cannot serve (attention's MAXSC, §07.3 in kernels_attn.cpp), or if
// `prepare_weight_adapters` has not been called on this scratch.
//
// `logits` may be null; when it is not, the full vocabulary comes back in it (≈1 MB D2H per token,
// which is the price §03.4's tie rule and this gate both accept). The step always syncs, because the
// argmax is taken on the host.
StepResult decode_step(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                       int token, std::vector<float>* logits = nullptr);

// Two BF16 weight rows that the graduated kernels want in F32, widened once per request instead of
// once per token: `linear_attn.conv1d.weight` and `linear_attn.dt_bias`.
//
// This is not a dtype conversion the model asked for, it is a dtype the two halves disagree about.
// §01.6 keeps exactly two tensors F32 per GDN layer — `A_log` and `linear_attn.norm.weight` — and
// `engine/loader_test.cpp` counts 36 such tensors, so the other two operands `gates_kernel` and
// `conv_kernel` take as `const float*` arrive BF16. Widening them per token would cost 18 launches
// and ~1.8 MB of traffic for 36 rows that never change, so they are widened into the scratch once,
// exactly as `state.h`'s conv-dtype note prices the same trade the other way. `decode_step` refuses
// to run before this is called.
void prepare_weight_adapters(const WeightStore& weights, DecodeScratch& sc);

// The launch plan of one token, by family, derived from the same §00.3 constants `decode_step`'s loop
// iterates over — not a hand-typed total. It is the engine's launch BILL (at the measured 1.85 µs
// dispatch floor, `tools/launch_overhead_hip.cpp`), which is what `engine/results.md`'s two deferrals
// (packed projections, a BF16-staged MLP) are priced against. `gate.py` checks these numbers against
// its own independent reading of §07, so the C++ and the Python cannot silently drift.
struct LaunchCounts {
    int glue = 0, proj = 0, gdn = 0, attn = 0, mlp = 0, cast = 0, other = 0;
    int total() const { return glue + proj + gdn + attn + mlp + cast + other; }
    std::string describe() const;
};
LaunchCounts planned_launch_counts();

// Cross-checks the shapes §07 gives this file against the checkpoint the store actually holds, so a
// width typed in `state.h` cannot disagree with the file silently. Returns "" when all match, else a
// message naming the first mismatch. Host-side only; `decode_run` calls it before the first step.
std::string check_weights(const WeightStore& weights);

}  // namespace engine
