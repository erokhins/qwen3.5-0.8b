#pragma once
// Per-request mutable state — §08.1's inventory, allocated exactly as §08.5 lays it out.
//
// Everything here is per REQUEST and mutable. Weights are read-only and belong to
// `engine::WeightStore` (`loader.h`); nothing in this file touches them.
//
// Sizes are derived from the §08.1 shapes, not copied from prose: `state_sizes.py` re-derives
// the same table against §00.5's rows and fails if these constants and that file ever disagree.
//
//   GDN recurrent S      [18][16][128][128] F32   1 048 576 B/layer   18.0 MiB/request
//   GDN conv window      [18][3][6144]    F32       73 728 B/layer    1.265 MiB/request
//   KV K and V per layer [6]×([cap][2][256] BF16)   2048 B/token/layer, 12 KiB/token total
//
// Two things in this struct did not match §08. Neither was corrected when this file landed —
// `27dbaed`'s comment claimed a spec correction in the same commit and there was none; the two items
// were only FLAGGED, in `state_sizes.py`'s FIXES list, which is where the arithmetic lives. One has
// since been corrected in the spec and one has not:
//   * the conv window is F32, not BF16. The values are BF16-exact (they are pre-conv `raw`
//     columns, §08.2), but `prototype/gdn_decode`'s conv kernel reads and writes the window as
//     F32 (`conv_kernel`'s `win`/`new_win` are `float*`), and paying 36 cast launches/token to
//     store 0.6 MiB instead of 1.27 MiB — of a pool this model uses 4 % of — would be buying a doc
//     line with the token's time. CORRECTED 2026-10-11: §08.1 and §00.5 now say F32 / 1.27 MiB,
//     naming the old cell, and `state_sizes.py` fails if either page stops saying it.
//   * the window and the KV are laid out head-major (`[head][slot][dim]`) rather than §08.5's
//     `[slots][2][256]`, because `prototype/attn_decode`'s core indexes
//     `(kv_head*cap + slot)*256`. Same bytes, different order; the kernel is the authority.
//     §08.5 is UNCHANGED: flagged, not corrected.
//
// `hipMalloc` only, never managed memory: managed degrades silently near the pool limit on this
// box while `hipMalloc` fails loudly (`devices/strix-halo-unified-memory.md`).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine {

// §00.0/§00.3 hyperparameters this struct needs; §00 is the source, these are asserted against
// the checkpoint by `decode_run --check-sizes` rather than trusted.
constexpr int kNumLayers = 24;
constexpr int kNumGdnLayers = 18;
constexpr int kNumAttnLayers = 6;
constexpr int kNumValueHeads = 16;      // h_v = h_k, no key→value repeat (§04 preamble)
constexpr int kDk = 128;
constexpr int kDv = 128;
constexpr int kConvChannels = 6144;
constexpr int kConvKernel = 4;
constexpr int kConvWindow = kConvKernel - 1;   // §08.2: the last k_conv−1 pre-conv columns
constexpr int kNumKvHeads = 2;
constexpr int kNumQueryHeads = 8;       // §00.3 num_attention_heads; GQA 8/2, §05.5
constexpr int kHeadDim = 256;
constexpr int kHidden = 1024;
constexpr int kIntermediate = 3584;     // §00.3 intermediate_size (mlp_gemv calls it FF)
constexpr int kVocab = 248320;

// Derived widths, so a projection's row count is never a typed-in number. Every one of these is a
// shape `engine/weights_reference.py` / `specimpl._check` also asserts against the checkpoint; the
// cross-check that they AGREE with the checkpoint is `check_weights()` in decode_step.cpp.
constexpr int kValueDim = kNumValueHeads * kDv;             // 2048: in_proj_z, gated_out, out_proj's `in`
constexpr int kQkvRows = 3 * kValueDim;                     // 6144 == kConvChannels, §04.1
constexpr int kQwidth = kNumQueryHeads * kHeadDim;          // 2048: attn block output, o_proj's `in`
constexpr int kQprojRows = 2 * kQwidth;                     // 4096: q AND gate per head, §05.1
constexpr int kKvProjRows = kNumKvHeads * kHeadDim;         // 512: k_proj and v_proj

// §03.5 / §00.3 routing: `full_attention` where l mod 4 == 3 (layers 3,7,11,15,19,23).
constexpr bool is_full_attention(int layer) { return layer % 4 == 3; }
// Which §08.1 slot a model layer owns (the state arrays are compacted: 18 GDN, 6 attention).
int gdn_slot(int layer);
int attn_slot(int layer);

struct StateSizes {
    size_t gdn_state_bytes = 0;     // all 18 layers
    size_t conv_window_bytes = 0;   // all 18 layers
    size_t kv_bytes = 0;            // both K and V, all 6 layers, for `cap` slots
    size_t per_token_bytes = 0;     // §00.5's 12 KiB/token line
    size_t total_bytes = 0;
    int64_t cap = 0;
    std::string describe() const;   // one line, for the log and the gate
};

// One request's §08.1 tuple. Plain data + device pointers; `decode_step` mutates it in place.
struct RequestState {
    // §08.5: one contiguous [16][128][128] F32 block per GDN layer, head-major, kept away from
    // the BF16 tensors so a cache line never mixes dtypes. One allocation, 18 equal strides, so
    // a whole-token batch of layers stays one DMA-friendly region (§08.5) and a per-layer call
    // is just pointer arithmetic.
    float* gdn_state = nullptr;         // [18][16][128][128]
    float* conv_window = nullptr;       // [18][3][6144], row 0 oldest (§07.2's roll order)
    uint16_t* kv_keys = nullptr;        // [6][2][cap][256] BF16 bits, post-k_norm post-rope
    uint16_t* kv_values = nullptr;      // [6][2][cap][256] BF16 bits, raw v
    int64_t cap = 0;                    // slots allocated per layer (§08.1 "slots")
    int prefix_len = 0;                 // P: §02.4, the 0-based position of the NEXT token is P
    int sampled_token = -1;
    bool stopped = false;               // §07.5: a step's argmax was the stop id (248046). Raw
                                        // match, any phase — the loop decides if that ends the run
                                        // (`decode_run.cpp`; §07.5 arms only during generation)

    StateSizes sizes() const;
    // §08.4 "new request": zero S and the conv window, P = 0. Zeroing is a hipMemset, once.
    void reset();
    // §08.6's invariants, checked on the host after a step (cheap: they are all scalars).
    // Returns "" when clean, else a message naming the invariant that broke.
    std::string check_invariants() const;
    // Byte offsets used by both the kernels and the gate's dumps — one place, so a layout change
    // cannot silently desync a reader.
    float* gdn_state_layer(int layer) const;
    float* conv_window_layer(int layer) const;
    uint16_t* kv_keys_layer(int layer) const;
    uint16_t* kv_values_layer(int layer) const;
};

// Allocate one request's tuple for a context of `cap` tokens (§08.4: allocate → zero → P=0).
// Throws std::runtime_error on a HIP failure or on a cap that cannot fit.
RequestState allocate_state(int64_t cap);
void free_state(RequestState& st);

// §08.1's arithmetic in one place, so `decode_run` can print it and `state_sizes.py` can
// compare against it. `cap` only scales the KV term.
StateSizes state_sizes(int64_t cap);

}  // namespace engine
