#include "state.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define ENGINE_CK(x)                                                                \
    do {                                                                            \
        hipError_t e_ = (x);                                                        \
        if (e_ != hipSuccess) {                                                     \
            throw std::runtime_error(std::string("hip: ") + hipGetErrorString(e_) + \
                                     " at " __FILE__ + ":" + #x);                   \
        }                                                                           \
    } while (0)

namespace engine {

namespace {
// §08.5: "keep S 64-B aligned and separate from BF16 tensors". hipMalloc's own alignment is far
// above that on this runtime, so the requirement is asserted rather than hand-padded — if it ever
// stopped being true, the state layout would have to be slab-allocated deliberately.
void expect_aligned(const void* p, size_t n, const char* what) {
    if ((reinterpret_cast<uintptr_t>(p) % 64) != 0) {
        throw std::runtime_error(std::string(what) + " came back from hipMalloc unaligned");
    }
    if (p == nullptr && n != 0) {
        throw std::runtime_error(std::string(what) + " is null");
    }
}
}  // namespace

int gdn_slot(int layer) {
    int n = 0;
    for (int i = 0; i < layer; ++i)
        if (!is_full_attention(i)) ++n;
    return n;
}

int attn_slot(int layer) {
    int n = 0;
    for (int i = 0; i < layer; ++i)
        if (is_full_attention(i)) ++n;
    return n;
}

StateSizes state_sizes(int64_t cap) {
    StateSizes s;
    s.cap = cap;
    s.gdn_state_bytes = (size_t)kNumGdnLayers * kNumValueHeads * kDk * kDv * sizeof(float);
    s.conv_window_bytes =
        (size_t)kNumGdnLayers * kConvWindow * kConvChannels * sizeof(float);
    const size_t kv_per_layer = (size_t)cap * kNumKvHeads * kHeadDim * sizeof(uint16_t);
    s.kv_bytes = 2 * (size_t)kNumAttnLayers * kv_per_layer;   // K and V
    s.per_token_bytes = 2 * (size_t)kNumAttnLayers * kNumKvHeads * kHeadDim * sizeof(uint16_t);
    s.total_bytes = s.gdn_state_bytes + s.conv_window_bytes + s.kv_bytes;
    return s;
}

StateSizes RequestState::sizes() const { return state_sizes(cap); }

std::string StateSizes::describe() const {
    char buf[320];
    snprintf(buf, sizeof(buf),
             "cap=%lld GDN state %.3f MiB + conv window %.3f MiB + KV %.3f MiB = %.3f MiB"
             " (KV %.0f B/token, constant %.3f MiB/request)",
             (long long)cap, gdn_state_bytes / 1048576.0, conv_window_bytes / 1048576.0,
             kv_bytes / 1048576.0, total_bytes / 1048576.0, (double)per_token_bytes,
             (total_bytes - kv_bytes) / 1048576.0);
    return buf;
}

float* RequestState::gdn_state_layer(int layer) const {
    return gdn_state + (size_t)gdn_slot(layer) * kNumValueHeads * kDk * kDv;
}

float* RequestState::conv_window_layer(int layer) const {
    return conv_window + (size_t)gdn_slot(layer) * kConvWindow * kConvChannels;
}

uint16_t* RequestState::kv_keys_layer(int layer) const {
    return kv_keys + (size_t)attn_slot(layer) * kNumKvHeads * cap * kHeadDim;
}

uint16_t* RequestState::kv_values_layer(int layer) const {
    return kv_values + (size_t)attn_slot(layer) * kNumKvHeads * cap * kHeadDim;
}

RequestState allocate_state(int64_t cap) {
    if (cap < 1) throw std::runtime_error("allocate_state: cap must be >= 1");
    const StateSizes s = state_sizes(cap);
    RequestState st;
    st.cap = cap;
    ENGINE_CK(hipMalloc((void**)&st.gdn_state, s.gdn_state_bytes));
    expect_aligned(st.gdn_state, s.gdn_state_bytes, "gdn_state");
    ENGINE_CK(hipMalloc((void**)&st.conv_window, s.conv_window_bytes));
    expect_aligned(st.conv_window, s.conv_window_bytes, "conv_window");
    ENGINE_CK(hipMalloc((void**)&st.kv_keys, s.kv_bytes / 2));
    expect_aligned(st.kv_keys, s.kv_bytes / 2, "kv_keys");
    ENGINE_CK(hipMalloc((void**)&st.kv_values, s.kv_bytes / 2));
    expect_aligned(st.kv_values, s.kv_bytes / 2, "kv_values");
    st.reset();   // §08.4: zero S and the conv window before anything reads them
    return st;
}

void free_state(RequestState& st) {
    // Return values ignored on purpose: a free cannot be handled past the log, and every path
    // here is teardown. (void) casts say that, rather than tripping hipFree's nodiscard.
    (void)hipFree(st.gdn_state);
    (void)hipFree(st.conv_window);
    (void)hipFree(st.kv_keys);
    (void)hipFree(st.kv_values);
    st = RequestState{};
}

void RequestState::reset() {
    const StateSizes s = sizes();
    ENGINE_CK(hipMemset(gdn_state, 0, s.gdn_state_bytes));
    ENGINE_CK(hipMemset(conv_window, 0, s.conv_window_bytes));
    // The KV cache is NOT zeroed: every lane is written by the append before it is read, and
    // §08.1's append-only rule means a lane is only ever written once. prefix_len is the
    // boundary the attention kernel respects (`lvalid`), so an unread lane cannot leak in.
    prefix_len = 0;
    sampled_token = -1;
    stopped = false;
}

std::string RequestState::check_invariants() const {
    const StateSizes s = sizes();
    if (prefix_len < 0) return "prefix_len is negative";
    if ((int64_t)prefix_len > cap) return "prefix_len exceeds the allocated slot count";
    // §08.6 invariant 4: "prefix_len equals the KV length after every step". The engine stores
    // exactly `cap` slots and fills 0..P-1, so the check is that P is inside the allocation and
    // that every attention layer advanced together (they do: one append per attention layer per
    // step, in decode_step). A lane written past P would be invariant 3 breaking; the append
    // writes slot `prefix_len` before P advances, which decode_step asserts below it.
    if (gdn_state == nullptr || conv_window == nullptr || kv_keys == nullptr ||
        kv_values == nullptr) {
        return "state pointers are null after allocation";
    }
    if (s.total_bytes == 0) return "computed state size is zero";
    return "";
}

}  // namespace engine
