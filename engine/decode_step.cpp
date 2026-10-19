// One greedy decode step (§07.1), in §07.1's order, built only from the launchers declared in
// `kernels.h`. Read `decode_step.h` first: it owns the buffer plan and the §03.2 norm pairing.
//
// What this file deliberately does NOT do:
//   * no arithmetic of its own — not one add, norm, gate or dot happens here. Every value a step
//     produces comes from a kernel that has its own gate against the pinned reference dumps.
//   * no host round-trip inside the token. §07.1 / `results.md`: "no host round-trip per op", and the
//     step is submitted as one stream of launches on the default stream with a single sync at the
//     logits. The one exception is deliberate and named: the conv-window roll-back copy (§08.2 note
//     below) and the logits D2H, both of which are memcpys, not waits. The body/finish split below
//     does not weaken this, it makes it structural: `submit_token_body` contains no readback at all,
//     which is what lets `engine/prefill.cpp` queue a whole wave of them before waiting.
//   * no dtype conversion written by hand. Every widening goes through `kernels_cast.cpp` so the
//     rounding is one named thing (RNE, same as torch's) instead of eighteen local casts.
//
// The wave body's launch plan is the per-token plan with the batchable families collapsed — glue and
// the MLP's two halves (P4), the nine projections (P5, §06.2's "one GEMM per projection"), and the
// three token-parallel GDN sites conv/gates/gated norm (P6, §06.2's "the conv is causal over the
// request start" plus §04.3/§04.5 being per-(token, head) elementwise).
// `planned_launch_counts()` below derives the n = 1 row of that table, and `gate.py` re-derives
// every row from §07 in Python and fails on a mismatch, so this comment and the code cannot drift
// apart quietly. Each collapsed family bills the SAME 49/114/48/18/18/18 launches for a body of 295
// tokens as for a body of one — one launch per op per body, whatever its width — which is why this
// table is the plan for both paths and `gate.py` section (h) can judge a wave against it unchanged.
// The GDN family is the one that does NOT collapse whole: of its four launches per layer, three are
// now per body and `rec_kernel` (§04.4's sequential scan) stays per token until P9 chunks it. Same
// for `other`'s 18 conv-window copies: per token at n = 1, per wave from P6 on, because the window
// is written once per wave whatever T_w is. The embedding lookup joined them at D30 4.2′ — a one-row
// body still pays the one row copy the table below bills, and a wider body pays two operations (one id
// upload and one gather, `kernels_embed.cpp`) instead of one 2 KiB copy per row. So this table is the
// plan for BOTH paths at n = 1 and the wave.s own shape at n > 1, and the wave half is what
// `stage_clock.cpp`'s `wave_plan_bill` derives rather than what this table asserts.
//
//   glue   1 norm-only (layer 0's input norm) + 2 × 24 fused add+norm = 49   (§03.2, = glue's own
//          gated 49-launch token plan; the final norm is the last of them, not an extra launch)
//   proj   GDN 4 in_proj + out_proj = 5 × 18;  attention q,k,v,o = 4 × 6     = 114
//   gdn    conv + gates + rec + gated_norm = 4 × 18                          =  72
//   attn   one fused §05 block per attention layer                           =   6
//   mlp    gateup + down = 2 × 24                                            =  48
//   cast   GDN 4/layer × 18 + attention 4/layer × 6 + the final norm         =  97
//   other  1 embedding row copy + 18 conv-window copies + 1 lm_head          =  20
//                                                                          ─────────
//                                                                          = 406
// At the measured 1.85 µs dispatch floor (`tools/launch_overhead_hip.cpp`) that is ≈0.75 ms/token of
// pure launch cost on top of ~7.5 ms of weight traffic. It is the engine's current bill, not a
// measurement: the two levers `engine/results.md` names — packed projections, and BF16-staged
// `mlp_gateup`/`mlp_down` (which would delete 48 of the 97 casts) — are what shrink it.
// `gate.py` re-derives every row of that table from §07 in Python and fails on a mismatch, so this
// comment and the code cannot drift apart quietly.
//
// THE STAGE-CLOCK MARKERS. Every submission site below is preceded by one `stageclock::note(family)`
// naming which row of the table above the launch that follows belongs to. It is a marker, not a
// barrier: with no clock armed it reads one null pointer and returns — no sync, no launch, no reorder,
// no readback — and with one armed it is where that pass drains the stream. That is why it PRECEDES the
// submission rather than following it: a sync taken after a note has already let the next launch into
// the queue, so it drains that launch too and charges it to the previous stage, shifting every boundary
// by one submission and being invisible about it on a step where most segments are one launch long.
// Same always-compiled, cheap-when-idle choice as `engine/launch_stats.h`'s counters, for the same
// reason: a probe that changes the shape of the thing it measures is not a probe, and the decision to
// measure belongs to the driver, not to this file.
//
// The GDN and MLP sites name their OWN family — `gdn_conv`/`gdn_gates`/`gdn_rec`/`gdn_gated_norm`,
// `mlp_gateup`/`mlp_down`, spelled as `engine/launch_stats.h` spells them — rather than the coarse `gdn`
// and `mlp` of the table above, because a wave's attribution has to separate the three GDN launches that
// batch (P6 made them cheap) from the recurrence that cannot (which is P9's entire subject), and the two
// halves of the MLP prefer opposite weight walks (Q10b). A clock measuring a STEP folds these names back
// to `gdn`/`mlp` in `note()` before it compares families (`stage_clock.h:coarse_stage`), so the two extra
// names cost `--time-stages` nothing: same segments, same sync count, same nine report rows it printed
// before the fine names existed. Nothing about a byte of output depends on which unit is being measured.
#include "engine/decode_step.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels.h"
#include "engine/loader.h"
#include "engine/stage_clock.h"

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

// ---- weight lookup ---------------------------------------------------------------------------
// The store is keyed on the §02.5 normalised name (`lm.layers.14.linear_attn.norm.weight`), so this
// file builds engine names, never checkpoint names — `prototype/weight_names.py` documents why the
// other spelling aborts the process from inside a `WeightStore::at()`. `at()` takes a string_view,
// so the stack buffer costs no allocation; a missing name throws and names the tensor.
template <size_t N>
std::string layer_name(int layer, const char* leaf) {
    char buf[N];
    std::snprintf(buf, sizeof buf, "lm.layers.%d.%s", layer, leaf);
    return std::string(buf);
}

const Weight& layer_weight(const WeightStore& ws, int layer, const char* leaf) {
    return ws.at(layer_name<160>(layer, leaf));
}

const void* layer_dev(const WeightStore& ws, int layer, const char* leaf) {
    return layer_weight(ws, layer, leaf).device;
}

// The leaves this file needs, by the §01.2/§01.4/§01.5 spelling of the checkpoint.
namespace leaf {
constexpr const char* kInputNorm = "input_layernorm.weight";
constexpr const char* kPostAttnNorm = "post_attention_layernorm.weight";
constexpr const char* kGateProj = "mlp.gate_proj.weight";
constexpr const char* kUpProj = "mlp.up_proj.weight";
constexpr const char* kDownProj = "mlp.down_proj.weight";
constexpr const char* kInProjQkv = "linear_attn.in_proj_qkv.weight";
constexpr const char* kInProjZ = "linear_attn.in_proj_z.weight";
constexpr const char* kInProjB = "linear_attn.in_proj_b.weight";
constexpr const char* kInProjA = "linear_attn.in_proj_a.weight";
constexpr const char* kConvW = "linear_attn.conv1d.weight";
constexpr const char* kALog = "linear_attn.A_log";
constexpr const char* kDtBias = "linear_attn.dt_bias";
constexpr const char* kGatedNorm = "linear_attn.norm.weight";
constexpr const char* kOutProj = "linear_attn.out_proj.weight";
constexpr const char* kQProj = "self_attn.q_proj.weight";
constexpr const char* kKProj = "self_attn.k_proj.weight";
constexpr const char* kVProj = "self_attn.v_proj.weight";
constexpr const char* kQNorm = "self_attn.q_norm.weight";
constexpr const char* kKNorm = "self_attn.k_norm.weight";
constexpr const char* kOProj = "self_attn.o_proj.weight";
}  // namespace leaf

constexpr const char* kFinalNorm = "lm.norm.weight";
constexpr const char* kEmbedding = "lm.embed_tokens.weight";
constexpr const char* kHead = "lm_head.weight";

template <typename T>
T* alloc(size_t n) {
    T* p = nullptr;
    ENGINE_CK(hipMalloc((void**)&p, n * sizeof(T)));
    return p;
}

// One expected weight: the leaf under `lm.layers.N.` (or a full name, for the two top-level entries),
// the shape and dtype the SHARD HEADER says it has, with the rank stated explicitly.
//
// The rank is a field and not a convention because of a real failure: an earlier version of this
// struct was `{leaf, d0, d1}` with "d1 == 0 means 1-D", which cannot express `conv1d.weight`. The
// tempting repair — fold the depth-1 middle dim away and compare element counts — is the one thing a
// shape checker must not do: a checker that squeezes a rank stops reporting what the store holds,
// which is the only job it has. The checkpoint has exactly 18 rank-3 tensors, all of them
// [6144,1,4] (`tools/weights_report.py`, and nothing else in 488 tensors is rank 3), so this one
// entry is the whole reason the field exists.
//
// Every dims below was READ from the header, not derived from §00: BF16 [248320,1024] embedding,
// [1024] norms, [3584,1024] gate/up, [1024,3584] down, [6144,1024] qkv, [2048,1024] z, [16,1024]
// a/b, [6144,1,4] conv (BF16), [16] dt_bias (BF16), [16] A_log (F32), [128] gated-norm weight (F32),
// [1024,2048] out_proj/o_proj, [4096,1024] q_proj, [512,1024] k/v_proj, [256] q_norm/k_norm. The 36
// F32 tensors are 18×`A_log`[16] + 18×`norm.weight`[128] and nothing else.
struct Expected {
    const char* name;
    int rank;
    int64_t d0, d1, d2;
    Dtype dtype;
};

// The comparison, in one place so the two callers cannot disagree about what a mismatch means: dtype
// AND rank AND dims, element by element. Nothing here trims, squeezes or broadcasts.
bool matches(const Weight& w, const Expected& e) {
    if (w.dtype != e.dtype || w.shape.size() != static_cast<size_t>(e.rank)) return false;
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    for (int i = 0; i < e.rank; ++i) {
        if (w.shape[i] != d[i]) return false;
    }
    return true;
}

std::string want(const std::string& name, const Weight& w, const Expected& e) {
    if (matches(w, e)) return "";
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    std::ostringstream os;
    os << name << ": expected " << dtype_name(e.dtype) << " rank-" << e.rank << " [";
    for (int i = 0; i < e.rank; ++i) os << (i ? "," : "") << d[i];
    // Both sides print their rank, because "expected [6144,4] but the store holds 6144x1x4" reads
    // like a formatting difference and is really the whole finding.
    os << "] but the store holds " << dtype_name(w.dtype) << " rank-" << w.shape.size() << " "
       << w.shape_string();
    return os.str();
}

// The two weights this file widens, declared once and asserted twice: `check_weights` compares the
// store against them, and `prepare_weight_adapters` refuses to flatten anything but that shape into
// the layout the kernels index. A rank-3 [6144,1,4] collapsing to conv_kernel's flat [6144][4] is a
// decision the engine makes, so it is stated where it is spent — the checker's job is to report the
// store, not to make it look convenient.
constexpr Expected kWantConvW{leaf::kConvW, 3, kConvChannels, 1, kConvKernel, Dtype::BF16};
constexpr Expected kWantDtBias{leaf::kDtBias, 1, kNumValueHeads, 0, 0, Dtype::BF16};

}  // namespace

// ---- scratch ---------------------------------------------------------------------------------
DecodeScratch allocate_scratch(int wave_rows) {
    if (wave_rows < 1) {
        throw std::runtime_error("allocate_scratch: wave_rows must be >= 1, got " +
                                 std::to_string(wave_rows));
    }
    DecodeScratch sc;
    sc.wave_rows = wave_rows;
    // The row family: [wave_rows, H] / [wave_rows, I]. A decode step uses row 0 of each and issues
    // exactly the launches it issued when these were one row wide; a prefill wave uses rows 0..n-1
    // and gets one launch per call instead of n of them. P4's part of the family is ~18 MB of scratch
    // per request at kMaxWaveRows — AGENTS.md is explicit that memory capacity is not the constraint
    // for this model, so the wave buys the batching with scratch rather than with a second code path.
    // P5 adds to that family below, not beside it.
    const int h = (int64_t)wave_rows * kHidden;
    const int i = (int64_t)wave_rows * kIntermediate;
    // P5's side of the same argument: the nine projections read a [wave_rows, in] tile and write a
    // [wave_rows, rows] tile, so their inputs and outputs are row-family buffers too. At
    // kMaxWaveRows = 512 that is ~31 MiB on top of P4's ~18 MiB of scratch per request — against a
    // 1.55 GB checkpoint and a ~34 GiB pool, which AGENTS.md is explicit is not the constraint for
    // this model. P6 adds the GDN mixer's own three (g_u the conv's output tile, g_o the
    // recurrence's, g_g/g_beta the gates), because its batched conv / gates / gated norm each read
    // and write one row per token in one launch; the recurrence's per-token stage is row t of those
    // tiles. conv_new and the a_* stage buffers stay as they are: the first is a K-1-row window,
    // which is not a row family, and the others belong to an attention block that is still one
    // token wide (P7/P8). See the two-family note in decode_step.h.
    const int qkv = (int64_t)wave_rows * kQkvRows;
    const int vdim = (int64_t)wave_rows * kValueDim;
    const int qprj = (int64_t)wave_rows * kQprojRows;
    const int kprj = (int64_t)wave_rows * kKvProjRows;
    const int vheads = (int64_t)wave_rows * kNumValueHeads;
    sc.s_a = alloc<uint16_t>(h);
    sc.s_b = alloc<uint16_t>(h);
    sc.norm = alloc<uint16_t>(h);
    sc.norm_f32 = alloc<float>(h);
    sc.embed = alloc<uint16_t>(h);
    sc.delta = alloc<uint16_t>(h);
    sc.block_out = alloc<uint16_t>(vdim);
    sc.block_out_f32 = alloc<float>(vdim);

    sc.g_raw = alloc<uint16_t>(qkv);
    sc.g_z = alloc<uint16_t>(vdim);
    sc.g_a = alloc<uint16_t>(vheads);
    sc.g_b = alloc<uint16_t>(vheads);
    sc.g_dt_bias = alloc<float>(kNumGdnLayers * kNumValueHeads);
    sc.g_conv_w = alloc<float>(kNumGdnLayers * kConvChannels * kConvKernel);
    sc.g_g = alloc<float>(vheads);
    sc.g_beta = alloc<float>(vheads);
    sc.g_u = alloc<float>(qkv);
    sc.g_o = alloc<float>(vdim);
    sc.conv_new = alloc<float>(kConvWindow * kConvChannels);

    // P9's chunked-scan staging (§04.7). Allocated unconditionally, like `a_part`: a `seq` run never
    // touches it, and a conditional allocation here would be a branch in the hot path and a code path
    // the default run cannot exercise. ~11 MiB at the 512-row ceiling — the same order as `a_part`'s
    // 33.8 MB, and AGENTS.md is explicit that memory capacity is not this model's constraint.
    sc.g_nv = alloc<float>((size_t)wave_rows * kNumValueHeads * kHeadDim);
    sc.g_kcd = alloc<uint16_t>((size_t)wave_rows * kNumValueHeads * kHeadDim);
    sc.g_qg = alloc<uint16_t>((size_t)wave_rows * kNumValueHeads * kHeadDim);
    sc.g_kh = alloc<uint16_t>((size_t)wave_rows * kNumValueHeads * kHeadDim);
    sc.g_intra = alloc<uint16_t>((size_t)wave_rows * kNumValueHeads * kernels::kGdnChunkTokens);
    sc.g_cdec = alloc<float>(((size_t)wave_rows + kernels::kGdnChunkTokens - 1) /
                             kernels::kGdnChunkTokens * kNumValueHeads);

    // Bet G1's norm staging (`--gdn-norm hoist`). Unconditional for the chunked staging's reason: a
    // branch here would be a path the default run never exercises. The row strides are `kValueDim` and
    // `2 * kNumValueHeads`, which are the strides the two kernels index (`NH * D` and `kGdnScalarRow`),
    // and `gdn_rec_span_hoisted` static_asserts the pair.
    sc.g_kn = alloc<float>(vdim);
    sc.g_qn = alloc<float>(vdim);
    sc.g_gsc = alloc<float>((size_t)wave_rows * 2 * kNumValueHeads);

    sc.a_q = alloc<uint16_t>(qprj);
    sc.a_k = alloc<uint16_t>(kprj);
    sc.a_v = alloc<uint16_t>(kprj);
    // P8 (§06.3): the attention stage and output tiles are wave-sized, because one attention site now
    // serves every row of the body. At n = 1 these are the same buffers, same contents, same order.
    sc.a_q_rope = alloc<uint16_t>((size_t)wave_rows * kNumQueryHeads * kHeadDim);
    sc.a_k_rope = alloc<uint16_t>((size_t)wave_rows * kKvProjRows);
    sc.a_gate = alloc<uint16_t>((size_t)wave_rows * kNumQueryHeads * kHeadDim);
    sc.a_pre = alloc<uint16_t>((size_t)wave_rows * kNumQueryHeads * kHeadDim);
    sc.a_gated = alloc<uint16_t>((size_t)wave_rows * kNumQueryHeads * kHeadDim);
    // The split-K online softmax's partial staging (§07.3, P7). One buffer per request, sized by the
    // attention launcher itself — it owns the head width and the split cap, and this file should not
    // have to. With P8 it is sized for a whole wave of rows, which is 258 MB at the lane ceiling
    // (32,768 lanes) instead of the 516 KiB one row needs: AGENTS.md is right about the budget (this
    // model uses 4 % of the pool), and a wave whose widest row is over the staging has nowhere else
    // to put `rows x 8 x split` partials.
    sc.a_part = alloc<float>(kernels::attn_part_floats(wave_rows));

    sc.mlp_act = alloc<uint16_t>(i);
    sc.mlp_act_f32 = alloc<float>(i);
    // Route (c)'s two raw dots, same row family and the same argument as `mlp_act`: 3.7 MiB each at
    // kMaxWaveRows, against a checkpoint of 1.55 GB and a pool this model uses 4 % of.
    sc.mlp_gate_raw = alloc<uint16_t>(i);
    sc.mlp_up_raw = alloc<uint16_t>(i);
    sc.logits = alloc<float>(kVocab);

    // The §02.3 gather's id staging (`decode_step.h` owns why it is pinned and why it is a ring).
    // Allocated unconditionally, like the chunked-scan staging above: a decode-only run never fills it,
    // and a conditional allocation here would be a branch in the body and a path the wide gates cannot
    // exercise.
    sc.token_ids_dev = alloc<int>((size_t)DecodeScratch::kEmbedIdSlots * wave_rows);
    ENGINE_CK(hipHostMalloc((void**)&sc.token_ids_pin,
                            (size_t)DecodeScratch::kEmbedIdSlots * wave_rows * sizeof(int),
                            hipHostMallocDefault));
    for (hipEvent_t& ev : sc.token_id_copied) {
        ENGINE_CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    }
    return sc;
}

void free_scratch(DecodeScratch& sc) {
    // Teardown: a hipFree cannot be handled past the log, hence the (void) casts (same convention
    // as `state.cpp`).
    (void)hipFree(sc.s_a);
    (void)hipFree(sc.s_b);
    (void)hipFree(sc.norm);
    (void)hipFree(sc.norm_f32);
    (void)hipFree(sc.embed);
    (void)hipFree(sc.delta);
    (void)hipFree(sc.block_out);
    (void)hipFree(sc.block_out_f32);
    (void)hipFree(sc.g_raw);
    (void)hipFree(sc.g_z);
    (void)hipFree(sc.g_a);
    (void)hipFree(sc.g_b);
    (void)hipFree(sc.g_dt_bias);
    (void)hipFree(sc.g_conv_w);
    (void)hipFree(sc.g_g);
    (void)hipFree(sc.g_beta);
    (void)hipFree(sc.g_u);
    (void)hipFree(sc.g_o);
    (void)hipFree(sc.conv_new);
    (void)hipFree(sc.g_nv);
    (void)hipFree(sc.g_kcd);
    (void)hipFree(sc.g_qg);
    (void)hipFree(sc.g_kh);
    (void)hipFree(sc.g_intra);
    (void)hipFree(sc.g_cdec);
    (void)hipFree(sc.g_kn);
    (void)hipFree(sc.g_qn);
    (void)hipFree(sc.g_gsc);
    (void)hipFree(sc.a_q);
    (void)hipFree(sc.a_k);
    (void)hipFree(sc.a_v);
    (void)hipFree(sc.a_q_rope);
    (void)hipFree(sc.a_k_rope);
    (void)hipFree(sc.a_gate);
    (void)hipFree(sc.a_pre);
    (void)hipFree(sc.a_gated);
    (void)hipFree(sc.a_part);
    (void)hipFree(sc.mlp_act);
    (void)hipFree(sc.mlp_act_f32);
    (void)hipFree(sc.mlp_gate_raw);
    (void)hipFree(sc.mlp_up_raw);
    (void)hipFree(sc.logits);
    // The §02.3 id staging. A hipFree/hipFreeHost/hipEventDestroy cannot be handled past the log, hence
    // the (void) casts (same convention as above and `state.cpp`). Resetting the ring cursor comes free
    // from `sc = DecodeScratch{}` below, which is also what nulls the events so a freed scratch cannot
    // be gathered into.
    (void)hipFree(sc.token_ids_dev);
    (void)hipFreeHost(sc.token_ids_pin);
    for (hipEvent_t& ev : sc.token_id_copied) (void)hipEventDestroy(ev);
    sc = DecodeScratch{};
}

void prepare_weight_adapters(const WeightStore& weights, DecodeScratch& sc) {
    // The shape contract of the two widenings, asserted where the flattened result is consumed.
    // `check_weights` already compares the store against these same two `Expected`s; the reason to
    // check again here is that THIS is the code that would read out of bounds if the shape were wrong,
    // and a second caller of `want()` costs nothing next to that.
    for (int layer = 0; layer < kNumLayers; ++layer) {
        if (is_full_attention(layer)) continue;
        const int slot = gdn_slot(layer);

        const Weight& dt = layer_weight(weights, layer, leaf::kDtBias);
        std::string bad = want(layer_name<160>(layer, leaf::kDtBias), dt, kWantDtBias);
        if (!bad.empty()) {
            throw std::runtime_error(bad +
                                     " — `gates_kernel` takes dt_bias as 16 contiguous F32, which "
                                     "this copies 1:1 after widening. If the checkpoint changed "
                                     "dtype, the widening is the thing to delete, not the check "
                                     "(§01.6 keeps exactly 36 tensors in F32: A_log and "
                                     "linear_attn.norm.weight). dt_bias is not one of them.");
        }
        kernels::cast_bf16_to_f32((const uint16_t*)dt.device,
                                  sc.g_dt_bias + (size_t)slot * kNumValueHeads, kNumValueHeads);

        const Weight& cw = layer_weight(weights, layer, leaf::kConvW);
        bad = want(layer_name<160>(layer, leaf::kConvW), cw, kWantConvW);
        if (!bad.empty()) {
            throw std::runtime_error(bad +
                                     " — conv_kernel indexes the weight as `C[c*K + j]`, i.e. the "
                                     "flat [6144][4] that the depth-1 middle dimension of the "
                                     "stored [6144,1,4] collapses to. A different rank is a "
                                     "different tensor, not a reshape, so this refuses rather "
                                     "than squeezing it into a shape that happens to fit.");
        }
        // Byte order untouched (§01): this is a dtype widening of a contiguous row, not a reorder.
        kernels::cast_bf16_to_f32((const uint16_t*)cw.device,
                                  sc.g_conv_w + (size_t)slot * kConvChannels * kConvKernel,
                                  kConvChannels * kConvKernel);
    }
    ENGINE_CK(hipDeviceSynchronize());
    sc.adapters_ready = true;
}

// ---- the body -------------------------------------------------------------------------------
// The §07.1 order, in §07.1's order, with no readback anywhere in it. `decode_step` at the bottom of
// this file is this function plus `finish_token`; `prefill_wave` is the only caller that separates
// them (a body per wave, then the finishes it wants).
//
// P4 made it a WAVE body: `n` consecutive tokens go through the stack layer by layer, and the two
// families whose arithmetic is proven row-batchable (glue and the MLP) are launched once per layer
// over the [n, H] tiles instead of once per layer per token. P5 moved the nine projections into the
// same shape, and P6 the GDN conv, gates and gated norm — with the recurrence left alone, because it
// is a scan over positions and cannot be launched wide (§04.4). `submit_token_body` below is this
// function with n = 1, which is what keeps one implementation of the order.
void submit_wave_body(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                      const int* tokens, int n) {
    if (!sc.adapters_ready) {
        throw std::runtime_error(
            "submit_wave_body: prepare_weight_adapters() has not been called on this scratch, so "
            "gates_kernel would read dt_bias and conv_kernel would read conv1d.weight as "
            "uninitialised memory.");
    }
    if (n < 1) {
        throw std::runtime_error("submit_wave_body: a wave has at least one token in it (n=" +
                                 std::to_string(n) + ")");
    }
    if (n > sc.wave_rows) {
        throw std::runtime_error("submit_wave_body: " + std::to_string(n) +
                                 " tokens will not fit this scratch's " +
                                 std::to_string(sc.wave_rows) +
                                 " rows. `prefill_wave` splits at that bound; a direct caller must");
    }
    const int pos0 = st.prefix_len;   // §02.4: P of the wave's FIRST token
    if (pos0 < 0 || (int64_t)pos0 + n > st.cap) {
        throw std::runtime_error("submit_wave_body: positions " + std::to_string(pos0) + ".." +
                                 std::to_string((int64_t)pos0 + n - 1) +
                                 " are not inside the allocated context (cap " +
                                 std::to_string(st.cap) + "); §07.5's end-of-context stop is the"
                                 " caller's, not this function's");
    }
    for (int t = 0; t < n; ++t) {
        if (tokens[t] < 0 || tokens[t] >= kVocab) {
            throw std::runtime_error("submit_wave_body: token " + std::to_string(tokens[t]) +
                                     " at wave row " + std::to_string(t) +
                                     " is outside [0, vocab)");
        }
    }

    // ---- THE ROW-COUNT SEAM for route (c), in one place. A body of one row is the hand GEMV —
    // always, whatever the run asked for, because "decode is byte-identical" is a property of the rule
    // and not of each call site remembering it — and a wider body takes hipBLAS for the ten fat shapes
    // when `sc.wave_gemm` says so. Same shape as `resolve_mlp_gate_map` below and `kernels_attn.cpp`'s
    // plan choice: the row count decides, and it is decided here because this is where `n` is.
    //
    // `project` is the one spelling of "this projection, whichever GEMM won", because the two take the
    // SAME tile in different widths: the hand GEMV is instantiated PACK_X=false and so reads the F32
    // widening (`x_f32`), while a library call takes a 16-bit `A` operand and reads what glue wrote
    // (`x_bf16`). Same numbers either way — a BF16 widened to F32 is exact — which is what makes the
    // two interchangeable at a call site rather than a numerics change dressed as one.
    //
    // What deliberately does NOT go through it: `in_proj_a`/`in_proj_b` (N=16, below the ridge at every
    // width, §3.2 and §7 W4) call `proj_gemv` directly at any width, and every mixer, glue and
    // `lm_head` site is untouched.
    //
    // The widenings this route leaves unread are guarded on THIS `lib_gemm`, at the sites named below.
    // The dead set is derived from "which call reads this destination", not from a remembered count:
    // every reader of a `norm_f32`/`block_out_f32`/`mlp_act_f32` tile in this body is either `project()`'s
    // hand branch, `mlp_gateup`/`mlp_down`'s hand branch, or the two 16-row `proj_gemv` calls that never
    // go through `project` — so a wide body on a library route writes 78 of its 96 casts to nothing, and
    // a one-row body writes all 96 because `resolve_wave_gemm` will not give it the library.
    // Deleting a submission is a change to the §07 launch bill, so the derivers moved in the same commit
    // this guard did: `planned_launch_counts()` stayed the per-token bill (the row rule keeps a decode
    // step at the hand count, so a guard on `lib_gemm` cannot reach it) and `stage_clock.cpp`'s
    // `wave_plan_bill` learned the route from `WaveShape`. D30 1.7' later moved the per-token bill from
    // 115 to 97 by deleting a widening NO route reads (`cast_bf16_to_f32_3`), which is the same plumbing
    // obligation one level up: both derivers moved with it, and neither number was edited to fit.
    const bool lib_gemm = kernels::wave_gemm_is_library(sc.wave_gemm, n);
    const auto project = [&](const void* weight, const uint16_t* x_bf16, const float* x_f32,
                             uint16_t* out, int rows, int in) {
        if (lib_gemm) {

            kernels::proj_gemm(weight, x_bf16, out, rows, in, n);
        } else {
            kernels::proj_gemv(weight, x_f32, out, rows, in, n);
        }
    };

    // ---- §02.3: x = E[y] for every row of the wave. One row of the tied table per row of the body
    // (§01.1: there is no second table, so `lm_head.weight` and this are the same bytes).
    //
    // THE ROW-COUNT RULE, the same one `resolve_wave_gemm` and `resolve_gdn_scan` apply and for the same
    // reason: it is a rule about the body's width, not a flag, and it is decided here because this is
    // where `n` is. A ONE-ROW body keeps the single `hipMemcpyAsync` it has always issued — a decode
    // step's submission bill cannot move, and §07's per-token plan (`planned_launch_counts()`, `gate.py`
    // section (d)) is that body's bill. A WIDER body issues one upload of the body's ids plus one gather
    // instead of `n` copies, because those copies were the wave's single biggest pile of submissions:
    // 4096 of 4240 `copy` operations, 2.05 µs each against an empty launch's 1.72 µs, i.e. 84 % of each
    // op was the submission and the same bytes as one operation cost 0.10 ms instead of 8.40 ms
    // (`tools/submission_probe.cpp`, `progress/2026-10-17/glue-cast-copy-bytes-census.md` §8). That is
    // D30 4.2′ and it is the answer to the question this comment used to leave open.
    //
    // What is deliberately NOT done: no per-token id upload (that would be the same bug one level down),
    // and no double-buffering of `sc.embed`. The gather writes the same rows the copies wrote, byte for
    // byte, and `kernels_embed.cpp` says why no bit can move.
    const uint16_t* table = (const uint16_t*)weights.at(kEmbedding).device;
    if (n == 1) {
        stageclock::note(stageclock::kCopy);
        ENGINE_CK(hipMemcpyAsync(sc.embed, table + (size_t)tokens[0] * kHidden,
                                 kHidden * sizeof(uint16_t), hipMemcpyDeviceToDevice));
    } else {
        // Slot `slot` of the staging ring (`decode_step.h` explains why it is pinned and why it is a
        // ring). Waiting on its event before writing it is what makes reuse safe rather than lucky; on a
        // real wave that event completed bodies ago, so the wait returns without blocking.
        const int slot = (int)(sc.token_id_bodies % DecodeScratch::kEmbedIdSlots);
        if (sc.token_id_bodies >= DecodeScratch::kEmbedIdSlots) {
            ENGINE_CK(hipEventSynchronize(sc.token_id_copied[slot]));
        }
        int* ids_pin = sc.token_ids_pin + (size_t)slot * sc.wave_rows;
        int* ids_dev = sc.token_ids_dev + (size_t)slot * sc.wave_rows;
        std::memcpy(ids_pin, tokens, (size_t)n * sizeof(int));
        stageclock::note(stageclock::kCopy);
        ENGINE_CK(hipMemcpyAsync(ids_dev, ids_pin, (size_t)n * sizeof(int), hipMemcpyHostToDevice));
        // Recorded after the upload and before the gather, so "this event completed" means exactly
        // "the device has read this slot", which is the only thing the next occupant of the slot needs.
        ENGINE_CK(hipEventRecord(sc.token_id_copied[slot], 0));
        ++sc.token_id_bodies;
        stageclock::note(stageclock::kCopy);
        kernels::embed_gather_rows(table, ids_dev, sc.embed, n);
    }

    // ---- §03.2's pairing, as glue's own gate pairs it (`prototype/glue/glue_pins.py`).
    // `s` is the residual as it stands; the write target alternates so `glue_add_norm`'s
    // `__restrict__ s` and `s_out` never alias. `sc.embed` is never a write target: after the first
    // call the ping-pong is entirely inside s_a/s_b. Each call now covers the WHOLE tile: one launch
    // per layer instead of one per layer per token, which is what `prototype/glue --batched` gates.
    const uint16_t* s = sc.embed;
    uint16_t* nxt = sc.s_a;
    const auto add_norm = [&](const uint16_t* delta, const void* norm_w) {
        stageclock::note(stageclock::kGlue);
        kernels::glue_add_norm_rows(s, delta, (const uint16_t*)norm_w, nxt, sc.norm, n);
        s = nxt;
        nxt = (nxt == sc.s_a) ? sc.s_b : sc.s_a;
    };

    // Layer 0's mixer norm has nothing to add before it — the norm-only form of the same kernel,
    // which is the job glue's gate calls `L00.n0` and pins against `rmsnorm_zc(s_in, w_in(0))`.
    stageclock::note(stageclock::kGlue);
    kernels::glue_add_norm_rows(s, nullptr,
                                (const uint16_t*)layer_dev(weights, 0, leaf::kInputNorm), nullptr,
                                sc.norm, n);

    for (int layer = 0; layer < kNumLayers; ++layer) {
        const bool attn_layer = is_full_attention(layer);

        // Every projection in this layer reads the SAME rows, and `proj_gemv` is instantiated with
        // PACK_X=false, so it reads F32 (`kernels.h`, and E2 in `engine/results.md`). One widening
        // for the tile, which is the part of "48 casts/token" that batching removes.
        //
        // The GDN half of this one stays at every width and the attention half goes on the route, because
        // their readers differ: a GDN layer's `in_proj_b`/`in_proj_a` call `proj_gemv` on `sc.norm_f32`
        // directly (§3.2, below the ridge at any width), while an attention layer's three input
        // projections all go through `project`, which reads the BF16 `sc.norm` whenever the library won.
        // So on a library route an attention layer's widening is a whole-tile store with no reader.
        // The polarity of the guard is the reader list, not the layer word: `!lib_gemm || !attn_layer`
        // casts for every hand body and for every GDN layer, and skips only the library body's attention
        // layers.
        if (!lib_gemm || !attn_layer) {
            stageclock::note(stageclock::kCast);
            kernels::cast_bf16_to_f32(sc.norm, sc.norm_f32, n * kHidden);
        }

        // ---- the mixer's INPUT projections: ONE launch each for the whole wave (P5, §06.2's "one
        // GEMM per projection"). Each reads the [n, H] norm tile and writes a [n, rows] tile that the
        // token loop below reads row t of. Hoisting them out of the token loop is all of P5, and at
        // n == 1 it is the same launch in the same place in the order. The 16-row b/a GEMVs stay
        // separate launches: `proj_gemv` has a packed multi-weight form and using it is the
        // ≈0.12 ms/token deferral `engine/results.md` names, which is a plan change, not this one.
        if (attn_layer) {
            // §05.1. q_proj's 4096 rows stay interleaved [q|gate] per head because that is the stored
            // layout (§10.5 measured the split reading at cos −0.14), and the block de-interleaves it.
            stageclock::note(stageclock::kProj);
            project(layer_dev(weights, layer, leaf::kQProj), sc.norm, sc.norm_f32, sc.a_q,
                    kQprojRows, kHidden);
            stageclock::note(stageclock::kProj);
            project(layer_dev(weights, layer, leaf::kKProj), sc.norm, sc.norm_f32, sc.a_k,
                    kKvProjRows, kHidden);
            stageclock::note(stageclock::kProj);
            project(layer_dev(weights, layer, leaf::kVProj), sc.norm, sc.norm_f32, sc.a_v,
                    kKvProjRows, kHidden);
        } else {
            // §04.1: four projections off the tile. The two fat ones (qkv at N=6144, z at N=2048) take
            // the route; the two 16-row ones are the §3.2 skinny pair and call `proj_gemv` directly at
            // any width, which is the carve-out stated as a call rather than as a comment.
            stageclock::note(stageclock::kProj);
            project(layer_dev(weights, layer, leaf::kInProjQkv), sc.norm, sc.norm_f32, sc.g_raw,
                    kQkvRows, kHidden);
            stageclock::note(stageclock::kProj);
            project(layer_dev(weights, layer, leaf::kInProjZ), sc.norm, sc.norm_f32, sc.g_z,
                    kValueDim, kHidden);
            stageclock::note(stageclock::kProj);
            kernels::proj_gemv(layer_dev(weights, layer, leaf::kInProjB), sc.norm_f32, sc.g_b,
                               kNumValueHeads, kHidden, n);
            stageclock::note(stageclock::kProj);
            kernels::proj_gemv(layer_dev(weights, layer, leaf::kInProjA), sc.norm_f32, sc.g_a,
                               kNumValueHeads, kHidden, n);
            // No widening here, and that is the whole of D30 1.7'. The conv and the gates used to read
            // a packed `g_raw|g_a|g_b` F32 copy of these three tiles because their kernels took
            // `const float*`; both now widen their own operands in-register, and because a BF16->F32
            // widening is exact the values are the same bits. The launch went with them at EVERY
            // width and on BOTH routes — no consumer ever reads the F32 tiles — which is what makes
            // this one a §07 launch-bill change, not a route guard like 4.1-a's.
        }

        // ---- the mixer. What has always ordered it is §08.2's ascending positions WITHIN a layer, and
        // that is still all any of it needs: nothing in a mixer needs another token's mixer output, which
        // is what lets the launches around it batch. What changed in P8 is which mixers still need a loop
        // at all: the causal KV append's ordering moved INTO the attention site (one `attn_prep` launch
        // stages every row's append at its own position, then one core launch reads them), so the only
        // per-token loop left is §04.4's recurrence, which is a scan and cannot be one launch (§06.2's
        // chunked alternative is §04.7, i.e. P9's conditional package). At n == 1 both branches are the
        // decode path's loop body: same launches, same order, same bits. What the mixers read is row t of
        // the tiles the launches around them filled (P5's projections, P6's conv and gates).
        //
        // ---- P6: §04.2's conv and §04.3's gates go ONCE per GDN layer for the whole wave. The conv
        // reads the [n, 6144] BF16 tile P5's `in_proj_qkv` just wrote and the layer's carried window,
        // and writes the wave's `u` tile plus the window's new contents — so the 3·6144·4 B window
        // copy that used to run 18 times per TOKEN now runs 18 times per WAVE (§2.4's end condition:
        // the window after a wave is raw[T−3..T−1], written once). The gates are the same story one
        // buffer smaller. Neither is allowed to read the recurrence's output, and neither does: the
        // conv is causal within the tile and the gates are per (token, head), which is what makes
        // hoisting them out of the token loop the same kind of edit P5 made with the projections.
        if (!attn_layer) {
            const int slot = gdn_slot(layer);
            // §04.2 / §07.2: the conv reads the OLD window and the roll appends the PRE-CONV raw, so
            // `new_win` cannot be the window itself — conv_kernel's `win`/`new_win` are both
            // `__restrict__`. It rolls into scratch and the layer's window is updated by one D2D
            // copy, one per layer per wave: 18 × 73 728 B for the whole wave instead of 18 × T_w of
            // them. Cheaper than double-buffering 1.27 MiB of state per request, which is the
            // alternative §08.2's "in-place ring update" wording invites; results.md carries it.
            stageclock::note(stageclock::kGdnConv);
            kernels::gdn_conv(sc.g_raw, st.conv_window_layer(layer),
                              sc.g_conv_w + (size_t)slot * kConvChannels * kConvKernel,
                              sc.g_u, sc.conv_new, n);
            stageclock::note(stageclock::kCopy);
            ENGINE_CK(hipMemcpyAsync(st.conv_window_layer(layer), sc.conv_new,
                                     (size_t)kConvWindow * kConvChannels * sizeof(float),
                                     hipMemcpyDeviceToDevice));
            // §04.3, all n rows at once. `A_log` is one of the 36 tensors the checkpoint stores F32
            // (§01.6), so it goes to the kernel as-is; dt_bias came through the adapter.
            stageclock::note(stageclock::kGdnGates);
            kernels::gdn_gates(sc.g_a, sc.g_b,
                               (const float*)layer_dev(weights, layer, leaf::kALog),
                               sc.g_dt_bias + (size_t)slot * kNumValueHeads,
                               sc.g_g, sc.g_beta, n);
        }
        // ---- the mixer. Attention (§05.3-§05.5) is now ONE site per attention layer for the whole
        // wave (P8, §06.3); the GDN recurrence (§04.4) stays one launch per token per GDN layer, because
        // it is a sequential scan and token t cannot start before t−1 finished writing S. What has not
        // changed is what each row computes: row t of an attention wave sits at position pos0+t and
        // attends pos0+t+1 lanes, exactly the per-token call this replaced, and `prototype/attn_decode`
        // §11 gates that a wave of rows is BIT-IDENTICAL to those rows launched one at a time. At
        // n == 1 this is the same single fused launch in the same place in the order.
        if (attn_layer) {
            kernels::AttnLayerArgs a{};
            a.q_proj_out = sc.a_q;              // [n][4096]: the tiles P5's three GEMVs just wrote
            a.k_proj_out = sc.a_k;              // [n][512]
            a.v_proj_out = sc.a_v;              // [n][512]
            a.q_norm = (const uint16_t*)layer_dev(weights, layer, leaf::kQNorm);
            a.k_norm = (const uint16_t*)layer_dev(weights, layer, leaf::kKNorm);
            a.k_cache = st.kv_keys_layer(layer);
            a.v_cache = st.kv_values_layer(layer);
            a.cap = (int)st.cap;
            a.pos = pos0;                       // row 0's position; row t's is pos0+t (§06.3)
            a.lvalid = pos0 + 1;                // and it attends pos0+t+1 lanes (§07.3's rule, per row)
            a.nrows = n;
            a.attn_out_pre_gate = sc.a_pre;     // [n][2048]
            a.gated = sc.a_gated;               // [n][2048]
            // The three stage buffers are mandatory (see kernels_attn.cpp): the §05.7 cache append
            // lives behind the same null check as one of them. For a wave, `attn_prep` writes all n
            // rows' appends in one launch that completes before any row reads the cache — which is the
            // only ordering a wave needs, and the reason a wave cannot stay on the fused plan.
            a.q_rope = sc.a_q_rope;
            a.k_rope = sc.a_k_rope;
            a.gate_out = sc.a_gate;
            // The split-K partials (§07.3): unused by rows that fit the one-pass staging, and required
            // by any that does not — the launcher refuses rather than lose a partial.
            a.part = sc.a_part;
            // Which wave core runs, from the run's own switch (`DecodeScratch::attn_tile`). A one-row
            // body ignores it: `attn_decode_layer` keeps the fused single-row form for `rows == 1`, so
            // a decode step is the shipped bits whatever this flag says — which is what lets the A/B
            // be taken inside one binary without a decode rung moving.
            a.wave_tile = sc.attn_tile;
            // And which SPELLING of that core's touch loop, from `DecodeScratch::attn_touch`. Same
            // one-row immunity for the same reason: `attn_decode_layer` only consults it on the tiled
            // branch it never takes for `rows == 1`. Carried as a separate flag rather than folded into
            // `--attn-tile` because the A/Bs are independent — `--attn-tile 0` must keep pricing the
            // per-(row,head) `attn_core` exactly as it did before this flag existed.
            a.wave_touch = sc.attn_touch;
            // And which FORM owns this body's streaming rows (`--attn-fattn`,
            // `kernels.h:AttnLayerArgs::wave_fattn`). Same riding reason; a one-row body never consults it
            // because the fused single-row form runs whatever it says, which is the property that keeps a
            // decode step bit-identical under the flag.
            a.wave_fattn = sc.attn_fattn;
            // And which SPELLING of that form's score roll (`--attn-fattn-roll`). Same riding reason and the
            // same one-row immunity: `attn_decode_layer` only reaches the WMMA branch on a wave, so a decode
            // step cannot be moved by it — which is the property that makes "the default side is inert" a
            // checkable claim instead of a hope (`prototype/attn_percost/A4-L1-impl.md`).
            a.fattn_roll = sc.attn_fattn_roll;
            stageclock::note(stageclock::kAttn);
            kernels::attn_decode_layer(a);
            // One widening over the whole [n][2048] tile, for the same reason P5 packed the casts
            // around the projections: it is elementwise, so a wider call is the same bits in fewer
            // submissions (and `cast` is not a launchstats family, so nothing has to re-derive it).
            // Guarded: the o_proj `project` call below is this tile's only reader, and it names
            // `a_gated` as the BF16 operand and this widening as the F32 one, so on a library route the
            // widening is written and never read.
            if (!lib_gemm) {
                stageclock::note(stageclock::kCast);
                kernels::cast_bf16_to_f32(sc.a_gated, sc.block_out_f32, (size_t)n * kQwidth);
            }
        } else {
            // The GDN mixer, and the one place the §04.4 scan FORM is chosen. `resolve_gdn_scan` is
            // the whole rule: a one-row body is `rec_kernel` whatever the run asked for, so a
            // generation step is the shipped sequential bits under either side of this A/B — the
            // same "a decode bit cannot move" property `--attn-tile` and `--gemm` carry, and for the
            // same reason (the decode pins are pinned through the sequential form).
            //
            // The chunked branch is ONE site per layer per body instead of n launches per layer, which
            // is the entire point of P9: at 1024 lanes and two 512-row bodies that is 36 sites and 72
            // launches where the sequential form issues 18,432 (§06.4 says count them, and the wave
            // report's `gdn_chunk` row does). The scan is still serial; what got chunked is the
            // intra-chunk work and the state traffic, not the dependency.
            //
            // `gdn_span` is resolved once here rather than inside the last branch, because that branch
            // is the only reader and `resolve_gdn_loop` is a pure function of (flag, rows) — asking it
            // on a collapse body arms nothing, and having the number named once is what lets the two
            // sequential branches below read as the two spans they are.
            const int gdn_span = kernels::resolve_gdn_loop(sc.gdn_loop, n);
            if (kernels::resolve_gdn_scan(sc.gdn_scan, n) == kernels::GdnScan::kCollapse) {
                // Route A (D-12): the sequential recurrence, resident. ONE launch per layer per body,
                // and the SAME arithmetic per step — see `engine/gdn_collapse_kernels.h` for why that is
                // a construction rather than a measurement, and `prototype/gdn_collapse/` for the CPU
                // mirror and the geometry arithmetic. Billed on the `gdn_rec` row, not on a row of its
                // own: the win line D-12 pre-registered is written against `gdn_rec`, and a route that
                // moves the milliseconds to a different row has not moved them.
                stageclock::note(stageclock::kGdnRec);      // one site, one launch, one row
                kernels::gdn_rec_collapse(sc.g_u, sc.g_g, sc.g_beta, st.gdn_state_layer(layer),
                                         sc.g_o, n);
            } else if (kernels::resolve_gdn_scan(sc.gdn_scan, n) == kernels::GdnScan::kChunk) {
                kernels::GdnChunkLayerArgs c{};
                c.u = sc.g_u;                 // [n][6144]; the kernel slices q|k|v itself (§04.1)
                c.g = sc.g_g;
                c.beta = sc.g_beta;
                c.S = st.gdn_state_layer(layer);
                c.o = sc.g_o;                 // [n][2048], §04.5's input, F32 as rec_kernel leaves it
                c.nv = sc.g_nv;
                c.kcd = sc.g_kcd;
                c.qg = sc.g_qg;
                c.kh = sc.g_kh;
                c.intra = sc.g_intra;
                c.cdec = sc.g_cdec;
                c.n = n;
                stageclock::note(stageclock::kGdnChunk);   // one SITE, two launches (prep then carry),
                                                           // counted like the attention site is
                kernels::gdn_chunk_prefill(c);
            } else if (gdn_span > 1) {
                // The sequential form, walked N token-steps per launch (`--gdn-loop`, D53/D54).
                // `resolve_gdn_loop` is the whole rule and it is shaped like `resolve_gdn_scan`'s on
                // purpose: a ONE-ROW body takes the per-token loop below whatever the flag says, so a
                // decode step is the shipped launch sequence BY CONSTRUCTION and not because a
                // measurement said it was flat. Everything here is a choice about SUBMISSIONS and
                // nothing else: `engine/gdn_rec_loop_kernels.h` is `rec_fused_kernel`'s body with the
                // token index moved inside it, admitted bit-identical at spans 0/1/3/16
                // (`bench/performance-status.md` §23), and `engine/gate.py` section (j) re-proves the
                // claim per fixture through this site.
                //
                // One launch per `span` token-steps. With `span >= n` — the shipping intent,
                // `--gdn-loop` at or above the body size — this is exactly ONE call per (layer, body),
                // which is the axis the item is on: 18 x 4,096 launches became 18 x 8 at 4,096 lanes.
                // A span below the body size means what it says (ceil(n/span) launches, the last one
                // short) because span-invariance is proved
                // (`prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp --selftest-loop`), and it is the named
                // fallback if a 512-iteration launch behaves badly in situ.
                //
                // The tile BASE plus a row count is handed over rather than row t: `g_u`/`g_g`/
                // `g_beta`/`g_o` are the same body tiles the collapse branch above is fed, and the
                // kernel walks the row strides itself, so both forms touch identical addresses.
                // Bet G1's second shape: the same spans, but the per-token k/q norms are a pre-pass and
                // the walk reads them (`--gdn-norm hoist`, `engine/gdn_rec_hoist_kernels.h`). It rides
                // INSIDE the span loop rather than hoisting one pre-pass over the whole body on purpose:
                // the pair stays adjacent in the stream, so the stage tiles are the freshest bytes the
                // walk can be handed, and a span's stage tiles are one span wide rather than a body's.
                // Two submissions per span, therefore, and the marker says 2 — `stage_clock.h` bills
                // launches, and a family row that counted one launch where two fell is the row the win
                // line is written on. It bills on the SAME `gdn_rec` row for route A's and the loop's
                // reason: a change that moves milliseconds to another row has not moved them, and here
                // the pre-pass's DRAM is part of what the change costs.
                const bool norm_hoist = sc.gdn_norm == kernels::GdnNorm::kHoist;
                // Bet G2's geometry (`--gdn-geom wf4|wf2`): the same spans, the SAME one-launch-per-span
                // bill, a different block. Checked BEFORE `norm_hoist` because the two are not
                // combinable — the wavefront form computes the norms in registers, so there is nothing for
                // the pre-pass to hoist — and `decode_run` refuses the pair at parse time rather than
                // letting this ordering quietly pick one of two flags the caller named.
                //
                // Bet G4's map (`--gdn-geom reg16`) rides the SAME branch for the SAME reasons and is
                // routed first: same span loop, same one `stageclock::note(kGdnRec)` per span, same operand
                // tiles, different block and different state residency. Routing it before `norm_hoist` is
                // not a preference — its norms are registers too, so `--gdn-norm hoist` has nothing to
                // hoist here and `decode_run` refuses that pair at parse.
                const bool reg16_geom = sc.gdn_geom == kernels::GdnGeom::kReg16;
                const bool wf_geom = sc.gdn_geom != kernels::GdnGeom::kBlock;
                for (int t = 0; t < n; t += gdn_span) {
                    const int rows = gdn_span < n - t ? gdn_span : n - t;   // the ragged tail
                    if (reg16_geom) {
                        stageclock::note(stageclock::kGdnRec);              // one site, one launch
                        kernels::gdn_rec_span_reg16(sc.g_u + (size_t)t * kQkvRows,
                                                    sc.g_g + (size_t)t * kNumValueHeads,
                                                    sc.g_beta + (size_t)t * kNumValueHeads,
                                                    st.gdn_state_layer(layer),
                                                    sc.g_o + (size_t)t * kValueDim,
                                                    rows, sc.gdn_spell);
                        continue;
                    }
                    if (wf_geom) {
                        stageclock::note(stageclock::kGdnRec);              // one site, one launch
                        kernels::gdn_rec_span_wf(sc.g_u + (size_t)t * kQkvRows,
                                                 sc.g_g + (size_t)t * kNumValueHeads,
                                                 sc.g_beta + (size_t)t * kNumValueHeads,
                                                 st.gdn_state_layer(layer),
                                                 sc.g_o + (size_t)t * kValueDim,
                                                 rows, sc.gdn_spell, sc.gdn_geom);
                        continue;
                    }
                    if (norm_hoist) {
                        stageclock::note(stageclock::kGdnRec, 2);           // one site, two launches
                        kernels::GdnNormStage nst{};
                        nst.kn = sc.g_kn + (size_t)t * kValueDim;
                        nst.qn = sc.g_qn + (size_t)t * kValueDim;
                        nst.gsc = sc.g_gsc + (size_t)t * 2 * kNumValueHeads;
                        kernels::gdn_rec_span_hoisted(sc.g_u + (size_t)t * kQkvRows,
                                                      sc.g_g + (size_t)t * kNumValueHeads,
                                                      sc.g_beta + (size_t)t * kNumValueHeads,
                                                      st.gdn_state_layer(layer),
                                                      sc.g_o + (size_t)t * kValueDim,
                                                      nst, rows, sc.gdn_spell);
                        continue;
                    }
                    stageclock::note(stageclock::kGdnRec);                  // one site, one launch
                    kernels::gdn_rec_span(sc.g_u + (size_t)t * kQkvRows,
                                          sc.g_g + (size_t)t * kNumValueHeads,
                                          sc.g_beta + (size_t)t * kNumValueHeads,
                                          st.gdn_state_layer(layer),
                                          sc.g_o + (size_t)t * kValueDim,
                                          rows,
                                          // The SPELLING, like the per-token call below. Only
                                          // `fused` has a looped body and `gdn_rec_span` refuses
                                          // anything else, rather than running a kernel this flag
                                          // did not name under another one's timing.
                                          sc.gdn_spell);
                }
            } else {
                for (int t = 0; t < n; ++t) {
                    // §04.4's recurrence. q, k, v are the three 2048-wide slices of row t of the conv's
                    // output tile (§04.1's [q|k|v]); rec_kernel indexes its own slice as [head][dim], so
                    // the pointers are just offsets into that row.
                    stageclock::note(stageclock::kGdnRec);
                    kernels::gdn_rec(sc.g_u + (size_t)t * kQkvRows,
                                     sc.g_u + (size_t)t * kQkvRows + kValueDim,
                                     sc.g_u + (size_t)t * kQkvRows + 2 * kValueDim,
                                     sc.g_g + (size_t)t * kNumValueHeads,
                                     sc.g_beta + (size_t)t * kNumValueHeads,
                                     st.gdn_state_layer(layer), sc.g_o + (size_t)t * kValueDim,
                                     // The SPELLING, from the run (`--gdn-spell`). It reaches the
                                     // launcher here and nowhere else, which is what keeps the two
                                     // forms one flag apart inside one binary; the stage-clock site is
                                     // above it, so both forms land on the same `gdn_rec` row of the
                                     // wave report and the A/B is a diff of one family line.
                                     sc.gdn_spell);
                }
            }
        }  // the mixer

        // ---- §04.5 over the whole tile, P6's third batched launch. `o` arrives F32 as the
        // recurrence left it (row t of the tile, one block per (token, head)), `z` is still BF16 (it
        // is an activation, §04.1), and the norm weight is one of the 36 F32 ones — the kernel rounds
        // it to BF16 itself, which is what the pins carry (`prototype/gdn_gated_norm/results.md` §1).
        // The widening that follows goes over `n * kValueDim` elements in one launch for the same
        // reason `cast_bf16_to_f32_3` went over the projections' tile in P5: it is elementwise, so a
        // wider call is the same bits in fewer submissions.
        if (!attn_layer) {
            stageclock::note(stageclock::kGdnGatedNorm);
            kernels::gdn_gated_norm(sc.g_o, sc.g_z,
                                    (const float*)layer_dev(weights, layer, leaf::kGatedNorm),
                                    sc.block_out, nullptr, n);
            // The GDN twin of the attention widening above, and guarded for the identical reason: the
            // out_proj `project` call below reads `sc.block_out` when the library won and this
            // widening only when the hand GEMV is in charge.
            if (!lib_gemm) {
                stageclock::note(stageclock::kCast);
                kernels::cast_bf16_to_f32(sc.block_out, sc.block_out_f32, n * kValueDim);
            }
        }

        // ---- the mixer's OUTPUT projection, one launch over the [n, 2048] tile the loop just
        // filled (P5). The two layer types write the same scratch and the same width — `kQwidth` and
        // `kValueDim` are both 2048 — so this is one call site, and it is the §03.2 addend that
        // `add_norm` below consumes for every row of the wave.
        stageclock::note(stageclock::kProj);
        // The BF16 operand is NOT one tile: the attention mixer leaves its [n,2048] output in
        // `a_gated` and the GDN gated norm leaves its [n,2048] output in `block_out`, and the one named
        // here is the one this layer wrote. The hand GEMV never noticed which of the two it was pointed
        // at, because it only ever read the shared F32 widening — so `block_out` was stale for every
        // attention layer, and a library call pointed at it computes the output projection from
        // whatever the last GDN layer left there (measured on dec8: logits cos 0.42 against the pins,
        // 37 failed rows, every one of them downstream of layer 1). Same numbers either way once the
        // right tile is named: `a_gated` and `block_out` are both [wave_rows,2048] BF16 at ld = 2048.
        // On a library route neither side widens into `block_out_f32` any more (the only reader of that
        // scratch is this call's hand branch), so it holds whatever the last hand body wrote — which is
        // legal for exactly one reason: nothing on that route reads it.
        project(layer_dev(weights, layer, attn_layer ? leaf::kOProj : leaf::kOutProj),
                attn_layer ? sc.a_gated : sc.block_out, sc.block_out_f32, sc.delta, kHidden,
                attn_layer ? kQwidth : kValueDim);


        // ---- §03.2 add #1: the mixer's output onto the residual, and the post-attention norm.
        add_norm(sc.delta, layer_dev(weights, layer, leaf::kPostAttnNorm));

        // ---- §03.3 MLP, batched over the wave's rows (P4). The two HAND halves want F32 input (there
        // is no BF16-staged form of `mlp_gateup`/`mlp_down` — the deferral in results.md), so the norm
        // rows and `act` are widened once for the tile rather than once per token. The library halves
        // read the BF16 `sc.norm` and `sc.mlp_act` directly, so both widenings are guarded on that
        // difference instead of being submitted and ignored.
        if (!lib_gemm) {
            stageclock::note(stageclock::kCast);
            kernels::cast_bf16_to_f32(sc.norm, sc.norm_f32, n * kHidden);
        }
        // Which of the two bit-identical gate/up walks this body runs is decided HERE, by the body's
        // row count, because the one-row and the many-row body measurably prefer opposite walks
        // (Q10b; the numbers and the honest statement about the mechanism are in `engine/results.md`
        // §Q10b and the rule itself lives in `kernels.h:resolve_mlp_gate_map`). This is the row-count
        // seam attention's plan already runs on — except that attention branches inside its launcher,
        // while the mapping is settled here, where `n` is already in scope, so the launcher stays
        // plan-blind. An explicit `--mlp-map` comes back unchanged, so the flag still pins one walk
        // for every body.
        const kernels::MlpGateMap gate_up_map = kernels::resolve_mlp_gate_map(sc.mlp_gate_map, n);
        stageclock::note(stageclock::kMlpGateup);
        if (lib_gemm) {
            // Two library dots and the epilogue they unfuse, counted as the one gateup site the §07 bill
            // prices (`kernels_mlp.cpp` says why the count is per site, not per kernel).
            kernels::mlp_gateup_gemm(layer_dev(weights, layer, leaf::kGateProj),
                                     layer_dev(weights, layer, leaf::kUpProj), sc.norm,
                                     sc.mlp_gate_raw, sc.mlp_up_raw, sc.mlp_act, kIntermediate,
                                     kHidden, n);
        } else {
            kernels::mlp_gateup((const uint16_t*)layer_dev(weights, layer, leaf::kGateProj),
                                (const uint16_t*)layer_dev(weights, layer, leaf::kUpProj),
                                sc.norm_f32, sc.mlp_act, kIntermediate, kHidden, n, gate_up_map);
        }
        if (!lib_gemm) {
            stageclock::note(stageclock::kCast);
            kernels::cast_bf16_to_f32(sc.mlp_act, sc.mlp_act_f32, n * kIntermediate);
        }
        stageclock::note(stageclock::kMlpDown);
        if (lib_gemm) {
            // `sc.mlp_act`, not the widening above: the library reads a 16-bit `A` operand, which is
            // why that widening is now guarded rather than submitted for nobody.
            kernels::mlp_down_gemm(layer_dev(weights, layer, leaf::kDownProj), sc.mlp_act, sc.delta,
                                   kHidden, kIntermediate, n);
        } else {
            kernels::mlp_down((const uint16_t*)layer_dev(weights, layer, leaf::kDownProj),
                              sc.mlp_act_f32, sc.delta, kHidden, kIntermediate, n);
        }

        // ---- §03.2 add #2: the MLP's output onto the residual, fused with the NEXT norm — the
        // next layer's input norm, or after layer 23 the final norm (§03.4). This is why the token
        // needs 49 glue launches and not 73, and why there is no separate final-norm call below.
        const void* next_norm = (layer + 1 < kNumLayers)
                                    ? layer_dev(weights, layer + 1, leaf::kInputNorm)
                                    : weights.at(kFinalNorm).device;
        add_norm(sc.delta, next_norm);
    }

    // The launches are all submitted and the final normed rows are sitting in `sc.norm` for whoever
    // wants to price them — row t is token t's, and `finish_token` takes the row it should read.
    // The position advances here, not in the tail, because the NEXT body reads `st.prefix_len` as
    // its own starting `pos` — that is what makes a wave of bodies legal to queue.
    st.prefix_len = pos0 + n;
    const std::string broken = st.check_invariants();
    if (!broken.empty()) {
        throw std::runtime_error("submit_wave_body: §08.6 invariant broken after the wave: " +
                                 broken);
    }
}

// One token is a wave of one. Same launches, same order, same everything: the decode path's
// byte-identity to before P4 is this line, not an argument.
void submit_token_body(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                       int token) {
    submit_wave_body(weights, st, sc, &token, 1);
}

// ---- the tail -------------------------------------------------------------------------------
// One `lm_head_full`, ONE blocking D2H (the sync of the token), and §03.4's argmax. Reads the row
// `submit_token_body` left in `sc.norm`; see the declaration in `decode_step.h` for why that
// ordering is a contract rather than an accident.
StepResult finish_token(const WeightStore& weights, RequestState& st, DecodeScratch& sc,
                        std::vector<float>* logits, int norm_row) {
    // ---- §03.4: logits = RMSnorm_final(x) · Eᵀ. `sc.norm` already holds RMSnorm_final(x): the
    // last glue call of the body wrote it as a side effect of the layer-23 residual add. Row 0 for
    // a decode step; a wave asks for row t of the tile its body left behind.
    if (norm_row < 0 || norm_row >= sc.wave_rows) {
        throw std::runtime_error("finish_token: row " + std::to_string(norm_row) +
                                 " is outside this scratch's " + std::to_string(sc.wave_rows) +
                                 " norm rows");
    }
    const uint16_t* norm_row_ptr = sc.norm + (size_t)norm_row * kHidden;
    // Row 0 of `norm_f32` is the scratch this widens into: by the time a row is priced the body is
    // over, so nothing else is reading it.
    stageclock::note(stageclock::kCast);
    kernels::cast_bf16_to_f32(norm_row_ptr, sc.norm_f32, kHidden);
    stageclock::note(stageclock::kLmHead);
    kernels::lm_head_full((const uint16_t*)weights.at(kHead).device, sc.norm_f32, sc.logits,
                          kVocab);

    // ---- §03.4's tie rule, on the host: ties go to the LOWEST id, which is a strict `>` scan.
    // One sync for the whole step, here and only here.
    stageclock::note(stageclock::kHostTail, 0);   // 0: a D2H and a host loop, not a launch
    std::vector<float> host_logits(kVocab);
    ENGINE_CK(hipMemcpy(host_logits.data(), sc.logits, kVocab * sizeof(float),
                        hipMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < kVocab; ++i) {
        if (host_logits[i] > host_logits[best]) best = i;   // strict: first (lowest) id wins a tie
    }
    const float host_logit = host_logits[best];   // read before the vector is moved out
    if (logits) *logits = std::move(host_logits);

    st.sampled_token = best;
    if (best == kStopToken) st.stopped = true;   // §07.5 / A-10 — the raw match, phase-blind: the
                                                 // caller owns whether it ends anything

    StepResult out;
    out.token = best;
    out.top_logit = host_logit;
    return out;
}

// ---- the step -------------------------------------------------------------------------------
// §07.1 is one order and it has one implementation: a step is its body and then its tail, which is
// the same two call sites, in the same order, that were one function before the wave needed them
// apart. Same launches, same sync count (one, in the tail).
StepResult decode_step(const WeightStore& weights, RequestState& st, DecodeScratch& sc, int token,
                       std::vector<float>* logits) {
    submit_token_body(weights, st, sc, token);
    return finish_token(weights, st, sc, logits);
}

std::string LaunchCounts::describe() const {
    // key=value pairs, because `gate.py` parses this line. The totals are printed too rather than
    // recomputed by whoever reads it.
    std::ostringstream os;
    os << "glue=" << glue << " proj=" << proj << " gdn=" << gdn << " attn=" << attn << " mlp=" << mlp
       << " cast=" << cast << " other=" << other << " total=" << total();
    return os.str();
}

LaunchCounts planned_launch_counts() {
    LaunchCounts c;
    c.glue = 1 + 2 * kNumLayers;
    c.proj = kNumGdnLayers * 5 + kNumAttnLayers * 4;
    c.gdn = kNumGdnLayers * 4;
    c.attn = kNumAttnLayers;
    c.mlp = 2 * kNumLayers;
    c.cast = 4 * kNumGdnLayers + 4 * kNumAttnLayers + 1;   // +1: the final norm before lm_head
    c.other = 1 + kNumGdnLayers + 1;
    return c;
}

std::string check_weights(const WeightStore& weights) {
    const Weight& emb = weights.at(kEmbedding);
    std::string bad = want(kEmbedding, emb, {kEmbedding, 2, kVocab, kHidden, 0, Dtype::BF16});
    if (!bad.empty()) return bad;
    // §01.1's tying, re-asserted where it is used: if `lm_head.weight` ever became its own tensor the
    // logits GEMV would read a different 485 MiB and nothing else here would notice.
    const Weight& head = weights.at(kHead);
    if (head.device != emb.device) {
        return "lm_head.weight is not aliased to lm.embed_tokens.weight — §01.1 says one blob";
    }
    bad = want(kFinalNorm, weights.at(kFinalNorm), {kFinalNorm, 1, kHidden, 0, 0, Dtype::BF16});
    if (!bad.empty()) return bad;

    // The nine weights a GDN block reads, and the six an attention block reads, as §01.2/§01.4 store
    // them. Two separate lists on purpose: the two blocks share only the norms and the MLP, and one
    // list with "overwrite the first six for attention layers" is how a check ends up verifying a
    // tensor the layer does not have.
    static const Expected kGdnBlock[] = {
        {leaf::kInProjQkv, 2, kQkvRows, kHidden, 0, Dtype::BF16},
        {leaf::kInProjZ, 2, kValueDim, kHidden, 0, Dtype::BF16},
        {leaf::kInProjB, 2, kNumValueHeads, kHidden, 0, Dtype::BF16},
        {leaf::kInProjA, 2, kNumValueHeads, kHidden, 0, Dtype::BF16},
        // The checkpoint's only rank-3 tensors: 18 of them, one per GDN layer, all [6144,1,4]. The
        // depth-1 middle dimension is Conv1d's and it is flattened where `conv_kernel` consumes it
        // (prepare_weight_adapters), NOT here.
        kWantConvW,
        {leaf::kALog, 1, kNumValueHeads, 0, 0, Dtype::F32},
        kWantDtBias,
        {leaf::kGatedNorm, 1, kDv, 0, 0, Dtype::F32},
        {leaf::kOutProj, 2, kHidden, kValueDim, 0, Dtype::BF16},
    };
    static const Expected kAttnBlock[] = {
        {leaf::kQProj, 2, kQprojRows, kHidden, 0, Dtype::BF16},
        {leaf::kKProj, 2, kKvProjRows, kHidden, 0, Dtype::BF16},
        {leaf::kVProj, 2, kKvProjRows, kHidden, 0, Dtype::BF16},
        {leaf::kQNorm, 1, kHeadDim, 0, 0, Dtype::BF16},
        {leaf::kKNorm, 1, kHeadDim, 0, 0, Dtype::BF16},
        {leaf::kOProj, 2, kHidden, kQwidth, 0, Dtype::BF16},
    };
    static const Expected kEveryLayer[] = {
        {leaf::kInputNorm, 1, kHidden, 0, 0, Dtype::BF16},
        {leaf::kPostAttnNorm, 1, kHidden, 0, 0, Dtype::BF16},
        {leaf::kGateProj, 2, kIntermediate, kHidden, 0, Dtype::BF16},
        {leaf::kUpProj, 2, kIntermediate, kHidden, 0, Dtype::BF16},
        {leaf::kDownProj, 2, kHidden, kIntermediate, 0, Dtype::BF16},
    };

    for (int layer = 0; layer < kNumLayers; ++layer) {
        const bool attn = is_full_attention(layer);
        const Expected* lists[] = {kEveryLayer, attn ? kAttnBlock : kGdnBlock};
        const size_t lens[] = {sizeof(kEveryLayer) / sizeof(Expected),
                               attn ? sizeof(kAttnBlock) / sizeof(Expected)
                                     : sizeof(kGdnBlock) / sizeof(Expected)};
        for (size_t li = 0; li < 2; ++li) {
            for (size_t i = 0; i < lens[li]; ++i) {
                const Expected& e = lists[li][i];
                const std::string name = layer_name<160>(layer, e.name);
                const std::string msg = want(name, layer_weight(weights, layer, e.name), e);
                if (!msg.empty()) return msg;
            }
        }
    }
    return "";
}

}  // namespace engine
