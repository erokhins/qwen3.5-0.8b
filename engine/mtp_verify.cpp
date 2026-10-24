// §09 / P7 — the batched 2-row verify body, implemented. `engine/mtp_verify.h` carries the contract:
// what the body is, what it is NOT (no accept loop, no output change), the rollback surface with its byte
// counts, and the finding about what a pre-body snapshot can and cannot restore.
//
// Everything here submits through the SAME `kernels.h` launchers and the SAME `submit_wave_body` the
// shipping path uses — no arithmetic is re-spelled, and there is no kernel in this file. The one kernel
// that is new to the engine, the batched head, lives in `engine/lm_head_batched_kernels.h` and is reached
// through `kernels::lm_head_batched_xbf16`.
#include "engine/mtp_verify.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels.h"
#include "engine/loader.h"
#include "engine/mtp_meter.h"   // the default-off per-phase round meter; that header owns the semantics

#define VERIFY_CK(x)                                                                \
    do {                                                                            \
        hipError_t e_ = (x);                                                        \
        if (e_ != hipSuccess) {                                                     \
            throw std::runtime_error(std::string("hip: ") + hipGetErrorString(e_) + \
                                     " at " __FILE__ + ":" + #x);                   \
        }                                                                           \
    } while (0)

namespace engine::mtpverify {

namespace {
constexpr const char* kHead = "lm_head.weight";   // the tied table, same name `decode_step.cpp` resolves
}  // namespace

Route parse_route(const std::string& name) {
    if (name == "hand") return Route::kHand;
    if (name == "amortised") return Route::kAmortised;
    throw std::runtime_error("--mtp-route takes `hand` (the forced hand GEMV + sequential scan, the"
                             " default and today's launch sequence byte for byte) or `amortised` (the"
                             " scratch's own resolved route, i.e. what a wide body runs everywhere else);"
                             " got '" + name + "'. Refusing rather than guessing, because this names"
                             " which side of the route A/B the run is"
                             " (`prototype/mtp_accept/ROW-SCALE.md`)");
}

const char* route_name(Route r) {
    switch (r) {
        case Route::kHand: return "hand";
        case Route::kAmortised: return "amortised";
    }
    return "invalid";
}

Snapshot allocate_snapshot(const RequestState& st) {
    const StateSizes s = st.sizes();
    Snapshot sn;
    sn.bytes_gdn = (long long)s.gdn_state_bytes;
    sn.bytes_conv = (long long)s.conv_window_bytes;
    VERIFY_CK(hipMalloc((void**)&sn.gdn, (size_t)sn.bytes_gdn));
    VERIFY_CK(hipMalloc((void**)&sn.conv, (size_t)sn.bytes_conv));
    VERIFY_CK(hipMalloc((void**)&sn.shadow_gdn, (size_t)sn.bytes_gdn));
    VERIFY_CK(hipMalloc((void**)&sn.shadow_conv, (size_t)sn.bytes_conv));
    return sn;
}

void free_snapshot(Snapshot& s) {
    for (float** p : {&s.gdn, &s.conv, &s.shadow_gdn, &s.shadow_conv}) {
        if (*p) (void)hipFree(*p);   // the `mtp_draft.cpp` teardown rule: HIP marks hipFree nodiscard,
                                     // and a teardown failure surfaces on the next check rather than
                                     // being swallowed here
        *p = nullptr;
    }
    s.bytes_gdn = s.bytes_conv = 0;
}

void take_snapshot(const RequestState& from, const Snapshot& to) {
    VERIFY_CK(hipMemcpy(to.gdn, from.gdn_state, (size_t)to.bytes_gdn, hipMemcpyDeviceToDevice));
    VERIFY_CK(hipMemcpy(to.conv, from.conv_window, (size_t)to.bytes_conv, hipMemcpyDeviceToDevice));
}

void restore_snapshot(const Snapshot& s) {
    VERIFY_CK(hipMemcpy(s.shadow_gdn, s.gdn, (size_t)s.bytes_gdn, hipMemcpyDeviceToDevice));
    VERIFY_CK(hipMemcpy(s.shadow_conv, s.conv, (size_t)s.bytes_conv, hipMemcpyDeviceToDevice));
}

RequestState shadow_state(const RequestState& target, const Snapshot& snap, int prefix_len) {
    if (prefix_len < 0 || prefix_len > target.cap) {
        throw std::runtime_error("mtp_verify: shadow prefix_len " + std::to_string(prefix_len) +
                                 " is outside this request's cap " + std::to_string(target.cap));
    }
    RequestState sh = target;                 // KV, cap and every other field are the target's
    sh.gdn_state = snap.shadow_gdn;           // the mixer memory is the snapshot's shadow side
    sh.conv_window = snap.shadow_conv;
    sh.prefix_len = prefix_len;               // §08.6 inv. 4: this IS the KV length the launchers read
    sh.sampled_token = -1;                    // no sample exists until the two argmaxes below
    sh.stopped = false;
    return sh;
}

namespace {
// One contiguous run of KV bytes that a capture covers: one (attention layer, K-or-V, KV head) triple.
// `RequestState`'s own accessors give the base and `state.h`'s head-major note gives the stride, so the
// witness reads the bytes the kernels read rather than a second spelling of the layout.
struct KvRegion {
    const char* base = nullptr;      // start of this (layer, tensor, head) block
    long long lane_bytes = 0;        // kHeadDim * sizeof(uint16_t)
    long long head_stride = 0;       // cap * lane_bytes
    std::string name;                // "layer 7 K head 0", so a diff line names a place
};

std::vector<KvRegion> kv_regions(const RequestState& st) {
    std::vector<KvRegion> out;
    const long long lane = (long long)kHeadDim * (long long)sizeof(uint16_t);
    for (int l = 0; l < kNumLayers; ++l) {
        if (!is_full_attention(l)) continue;
        for (int h = 0; h < kNumKvHeads; ++h) {
            const std::string head = " head " + std::to_string(h);
            out.push_back({(const char*)st.kv_keys_layer(l) + h * st.cap * lane, lane, st.cap * lane,
                           "layer " + std::to_string(l) + " K" + head});
            out.push_back({(const char*)st.kv_values_layer(l) + h * st.cap * lane, lane, st.cap * lane,
                           "layer " + std::to_string(l) + " V" + head});
        }
    }
    return out;
}

std::string region_diff(const char* what, const std::vector<char>& a, const std::vector<char>& b) {
    if (a.size() != b.size()) {
        return std::string(what) + ": " + std::to_string(a.size()) + " bytes vs " +
               std::to_string(b.size());
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "%s: byte %llu of %llu: 0x%02x vs 0x%02x", what,
                          (unsigned long long)i, (unsigned long long)a.size(),
                          (unsigned char)a[i], (unsigned char)b[i]);
            return buf;
        }
    }
    return "";
}
}  // namespace

StateBytes capture_state_bytes(const RequestState& st, int token) {
    const StateSizes s = st.sizes();
    StateBytes c;
    c.token = token;
    c.pos = st.prefix_len;
    c.kv_lanes = st.prefix_len;
    c.gdn.resize((size_t)s.gdn_state_bytes);
    c.conv.resize((size_t)s.conv_window_bytes);
    VERIFY_CK(hipMemcpy(c.gdn.data(), st.gdn_state, c.gdn.size(), hipMemcpyDeviceToHost));
    VERIFY_CK(hipMemcpy(c.conv.data(), st.conv_window, c.conv.size(), hipMemcpyDeviceToHost));
    const std::vector<KvRegion> regions = kv_regions(st);
    const long long per_region = (long long)c.kv_lanes * kHeadDim * (long long)sizeof(uint16_t);
    c.kv.resize((size_t)per_region * (size_t)regions.size());
    size_t off = 0;
    for (const KvRegion& r : regions) {
        VERIFY_CK(hipMemcpy(c.kv.data() + off, r.base, (size_t)per_region, hipMemcpyDeviceToHost));
        off += (size_t)per_region;
        c.kv_names.push_back(r.name);
    }
    return c;
}

// The KV difference, decoded: "which layer's K, which head, which lane, which element". A raw byte offset
// into a 221 KB concatenation is not an address anybody can act on, and the whole reason this witness exists
// is to name a place.
std::string kv_diff(const StateBytes& a, const StateBytes& b) {
    if (a.kv.size() != b.kv.size()) {
        return "kv: " + std::to_string(a.kv.size()) + " bytes vs " + std::to_string(b.kv.size());
    }
    const long long lane_bytes = (long long)kHeadDim * (long long)sizeof(uint16_t);
    const long long per_region = (long long)a.kv_lanes * lane_bytes;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        if (a.kv[i] == b.kv[i]) continue;
        const size_t region = per_region ? i / (size_t)per_region : 0;
        const long long within = per_region ? (long long)(i % (size_t)per_region) : 0;
        char buf[256];
        std::snprintf(buf, sizeof buf, "kv: %s lane %lld element %lld: 0x%02x vs 0x%02x",
                      region < a.kv_names.size() ? a.kv_names[region].c_str() : "unknown region",
                      within / lane_bytes, within % lane_bytes / (long long)sizeof(uint16_t),
                      (unsigned char)a.kv[i], (unsigned char)b.kv[i]);
        return buf;
    }
    return "";
}

std::string state_diff_summary(const StateBytes& a, const StateBytes& b, size_t max_places) {
    auto count_bytes = [](const std::vector<char>& x, const std::vector<char>& y) {
        const size_t n = x.size() < y.size() ? x.size() : y.size();
        size_t d = 0;
        for (size_t i = 0; i < n; ++i) d += x[i] != y[i] ? 1 : 0;
        return d;
    };
    const size_t gdn = count_bytes(a.gdn, b.gdn);
    const size_t conv = count_bytes(a.conv, b.conv);
    const size_t lanes = (size_t)a.kv_lanes;
    const long long lane_bytes = (long long)kHeadDim * (long long)sizeof(uint16_t);
    const size_t per_region = (size_t)lanes * (size_t)lane_bytes;
    size_t kv_bytes = 0;
    std::vector<std::string> places;
    for (size_t r = 0; r < a.kv_names.size() && r * per_region < a.kv.size(); ++r) {
        for (size_t lane = 0; lane < lanes; ++lane) {
            const size_t base = r * per_region + (size_t)lane * (size_t)lane_bytes;
            size_t d = 0;
            for (long long j = 0; j < lane_bytes && base + (size_t)j < a.kv.size(); ++j) {
                d += a.kv[base + (size_t)j] != b.kv[base + (size_t)j] ? 1 : 0;
            }
            if (d == 0) continue;
            kv_bytes += d;
            if (places.size() < max_places) {
                places.push_back(a.kv_names[r] + " lane " + std::to_string(lane) + " (" +
                                 std::to_string(d) + "B)");
            }
        }
    }
    std::string out = "gdn=" + std::to_string(gdn) + "B conv=" + std::to_string(conv) +
                      "B kv=" + std::to_string(kv_bytes) + "B";
    if (!places.empty()) {
        out += " places[";
        for (size_t i = 0; i < places.size(); ++i) out += (i ? ", " : "") + places[i];
        out += "]";
    }
    return out;
}

std::string first_mixer_diff(const StateBytes& a, const StateBytes& b) {
    std::string d = region_diff("gdn_state", a.gdn, b.gdn);
    if (!d.empty()) return d;
    return region_diff("conv_window", a.conv, b.conv);
}

std::string first_state_diff(const StateBytes& a, const StateBytes& b) {
    if (a.token != b.token) {
        return "token fed: " + std::to_string(a.token) + " vs " + std::to_string(b.token);
    }
    if (a.pos != b.pos) {
        return "position: " + std::to_string(a.pos) + " vs " + std::to_string(b.pos);
    }
    std::string d = first_mixer_diff(a, b);
    if (!d.empty()) return d;
    // A capture of a different number of lanes is a difference. Without this a witness that quietly
    // covered fewer lanes than its partner would read "identical over what it looked at", which is the
    // empty-parse failure class again, one level down.
    if (a.kv_lanes != b.kv_lanes) {
        return "kv lanes captured: " + std::to_string(a.kv_lanes) + " vs " + std::to_string(b.kv_lanes);
    }
    return kv_diff(a, b);
}

long long kv_lane_bytes(const RequestState& st) {
    return (long long)kv_regions(st).size() * (long long)kHeadDim * (long long)sizeof(uint16_t);
}

LaneStage make_lane_stage(const RequestState& st) {
    LaneStage ls;
    ls.bytes_per_lane = kv_lane_bytes(st);
    ls.lanes = kRows;      // one lane per body row: each row of a wave appends exactly one lane
    ls.bytes = ls.bytes_per_lane * ls.lanes;
    VERIFY_CK(hipMalloc((void**)&ls.stage, (size_t)ls.bytes));
    return ls;
}

void free_lane_stage(LaneStage& ls) {
    if (ls.stage) (void)hipFree(ls.stage);   // the nodiscard rule, as in `free_snapshot`
    ls.stage = nullptr;
    ls.bytes = 0;
}

// Copy one lane of KV between the live cache and the staging buffer. `to_cache = false` saves, `true`
// restores; both walk `kv_regions`, the same address set the guard's capture compares, so the bytes put back
// are the bytes the witness can check rather than a second spelling of the layout. `slot_of` (the appender's
// own index) returns `pos` for every non-injection route, so lane == position is not an assumption here.
void copy_kv_lane(char* stage, const RequestState& st, int lane, bool to_cache) {
    if (stage == nullptr) {
        throw std::runtime_error("mtp_verify: lane stage is not allocated");
    }
    if (lane < 0 || lane >= st.cap) {
        throw std::runtime_error("mtp_verify: KV lane " + std::to_string(lane) + " is outside cap " +
                                 std::to_string(st.cap));
    }
    long long off = 0;
    for (const KvRegion& r : kv_regions(st)) {
        char* cache = const_cast<char*>(r.base) + (long long)lane * r.lane_bytes;
        if (to_cache) {
            VERIFY_CK(hipMemcpy((void*)cache, stage + off, (size_t)r.lane_bytes,
                                hipMemcpyDeviceToDevice));
        } else {
            VERIFY_CK(hipMemcpy(stage + off, (const void*)cache, (size_t)r.lane_bytes,
                                hipMemcpyDeviceToDevice));
        }
        off += r.lane_bytes;
    }
    if (off != (long long)kv_lane_bytes(st)) {
        // A layout change that made the two figures disagree would otherwise restore a shifted cache.
        throw std::runtime_error("mtp_verify: lane copy covered " + std::to_string(off) + " bytes, stage is " +
                                 std::to_string(kv_lane_bytes(st)));
    }
}

void poison_kv_lane(const RequestState& st, int lane) {
    // The refusal is the safety of the whole trick: a lane below `prefix_len` is a lane some later row
    // reads, and a witness that changes the run it is watching is worse than no witness.
    if (lane < st.prefix_len) {
        throw std::runtime_error("mtp_verify: refusing to poison lane " + std::to_string(lane) +
                                 ", which is inside the live prefix (prefix_len=" +
                                 std::to_string(st.prefix_len) + "); §08.6 makes only lanes at or above"
                                 " prefix_len unread, and only an unread lane may be poisoned");
    }
    if (lane >= st.cap) {
        throw std::runtime_error("mtp_verify: refusing to poison lane " + std::to_string(lane) +
                                 ", at or above cap " + std::to_string(st.cap));
    }
    for (const KvRegion& r : kv_regions(st)) {
        VERIFY_CK(hipMemset((void*)(r.base + (long long)lane * r.lane_bytes), kPoisonByte,
                            (size_t)r.lane_bytes));
    }
}

PoisonReport poisoned_lane_report(const RequestState& st, int lane) {
    PoisonReport rep;
    rep.lane_bytes = (long long)kHeadDim * (long long)sizeof(uint16_t);
    std::vector<char> buf((size_t)rep.lane_bytes);
    for (const KvRegion& r : kv_regions(st)) {
        VERIFY_CK(hipMemcpy(buf.data(), r.base + (long long)lane * r.lane_bytes, buf.size(),
                            hipMemcpyDeviceToHost));
        long long same = 0;
        for (char c : buf) same += (unsigned char)c == kPoisonByte ? 1 : 0;
        rep.survivors += same;
        rep.regions++;
        if (same == (long long)buf.size()) ++rep.regions_unwritten;   // a whole region still poison
    }
    return rep;
}

long long kv_bytes_above_prefix(const RequestState& st) {
    // One lane of K and V per attention layer: the bytes row 1 wrote and the target will later overwrite.
    return (long long)kNumAttnLayers * kNumKvHeads * 2 * kHeadDim * (long long)sizeof(uint16_t);
}

Result verify_body(const WeightStore& weights, const RequestState& target, DecodeScratch& sc,
                   Route route, const Snapshot& snap, int anchor_pos, const int* tokens,
                   float* logits_dev,
                   LaneStage& lanes, StateBytes* body_reads, PoisonReport* tail_poison,
                   const DecodeScratch::GdnCheckpoint* ckpt, bool capture_end_state,
                   int* verdict_dev, kernels::ArgmaxWord* verdict_partials) {
    if (tokens == nullptr || logits_dev == nullptr) {
        throw std::runtime_error("mtp_verify: verify_body needs a token array and a logits tile");
    }
    if (anchor_pos < 0 || anchor_pos + kRows > target.cap) {
        throw std::runtime_error("mtp_verify: anchor " + std::to_string(anchor_pos) + " + " +
                                 std::to_string(kRows) + " rows is outside cap " +
                                 std::to_string(target.cap));
    }
    // The body may not invent positions: the shadow reads the KV the TARGET has actually built, so its
    // first row has to sit at or below the target's own `prefix_len`. A violation here means the driver
    // anchored the probe ahead of the state, which would price a body over an empty cache and call the
    // number a verify cost.
    if (anchor_pos > target.prefix_len) {
        throw std::runtime_error("mtp_verify: anchor " + std::to_string(anchor_pos) + " is ahead of the"
                                 " target's prefix_len " + std::to_string(target.prefix_len) +
                                 "; the verify body reads the populated KV, it does not imagine one");
    }
    if (sc.wave_rows < kRows) {
        throw std::runtime_error("mtp_verify: the scratch is " + std::to_string(sc.wave_rows) +
                                 " rows tall and the body needs " + std::to_string(kRows));
    }

    RequestState sh = shadow_state(target, snap, anchor_pos);

    // SEED THE SHADOW BEFORE ANYTHING ELSE. `shadow_state` repointed the mixer at the shadow buffers; it
    // copied nothing, so without this line the body would run on memory it never wrote — a zeroed or a stale
    // GDN state, either of which is a *different operand* rather than a worse rounding, and both of which
    // propagate into the K/V this body stages. This is also the copy a reject would perform, so the cost is
    // the one `ARMS.md` arm C prices, not an extra.
    // `shadow_seed` and `witness` in `engine/mtp_meter.h`. Both close on a call that ALREADY blocks today
    // (`hipMemcpy` returns to the host once the copy is done), so these two lines carry `sync_added=0`:
    // they cost the run nothing it was not paying and bound their own device work by construction.
    {
        const engine::mtpmeter::PhaseGuard seed(engine::mtpmeter::kShadowSeed);
        restore_snapshot(snap);
    }
    if (body_reads != nullptr) {
        const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
        // What the shadow actually holds at the instant the body starts, in bytes. Taken here rather than
        // asserted here, because a witness that re-derives the value it is checking for is decoration.
        *body_reads = capture_state_bytes(sh, tokens[0]);
    }

    // THE ROUTE FORCING — see the header's "WHY THE BODY FORCES…" and "WHY THAT FORCING IS A FLAG NOW"
    // paragraphs. Under `kHand` a 2-row body is forced to the hand GEMV and the sequential scan, which is
    // what makes row 0 the same kernel chain the decode step ran and keeps the pre-flag launch sequence
    // byte for byte. Under `kAmortised` nothing is overridden: the scratch already carries the route the
    // driver resolved, i.e. the one a wide body takes everywhere else. Restored on every exit path,
    // including a throw, or the driver's own following steps would silently run the verify arm's route.
    const kernels::WaveGemm gemm = sc.wave_gemm;
    const kernels::GdnScan scan = sc.gdn_scan;
    const DecodeScratch::GdnCheckpoint* saved_ckpt = sc.gdn_ckpt;
    if (route == Route::kHand) {
        sc.wave_gemm = kernels::WaveGemm::kHand;
        sc.gdn_scan = kernels::GdnScan::kSeq;
    } else if (ckpt != nullptr && sc.gdn_scan != kernels::GdnScan::kSeq) {
        // Loud, not silent: the checkpoint IS a one-token-step recurrence's end state, so an amortised body
        // on `collapse` would capture a state the round never wrote and hand the caller a plausible restore.
        // `decode_run.cpp` refuses this pair at the flag; this is the guard for any other caller.
        throw std::runtime_error(std::string("mtp_verify: an amortised body with a mid-round checkpoint")
                                 + " must run the sequential scan; the driver resolved gdn_scan="
                                 + kernels::gdn_scan_name(sc.gdn_scan));
    }
    // Arming the split, when the caller asked for it. Null here is today's launch sequence with one
    // null test per GDN layer, which is why the probe's flag-off byte-identity is unaffected.
    sc.gdn_ckpt = ckpt;
    // And arming the attention form. Not route-dependent, unlike the two above: a round's body is a
    // speculation round whatever weight walk it rides, and the wide wave block was priced at PREFILL row
    // counts only (`decode_step.h`'s `kMtpRoundAttnWide` carries the numbers and the reason a row-count
    // predicate would not do). Restored below with the other two, on every exit path.
    const int saved_wide = arm_mtp_round_attn_form(sc);

    Result r;
    // What is in force for THIS body, read off the scratch it was just configured from rather than
    // restated from the flag, so the driver's echo cannot be satisfied by a flag that never took.
    r.gemm_ran = sc.wave_gemm;
    r.scan_ran = sc.gdn_scan;
    r.attn_wide_ran = sc.attn_fattn_wide;
    try {
        // THE ANCHOR-LANE REVERT. A 2-row body re-stages row 0's own KV lane, and that lane is inside the
        // live prefix: the step that produced the fed token already wrote those bytes. It cannot simply be
        // told not to — `attn_prep` is the wave's appender and it appends EVERY row unconditionally
        // (`prototype/attn_decode/attn_decode_kernels.h`: `a.kc_as[((size_t)kv * a.cap + slot_of(row_pos,
        // a.cap, a.neg, 0)) * DH + d] = fb(out[d]);`, in the KV-head branch, with no test on `qrow`),
        // launched with `gridDim.y = rows` (`engine/kernels_attn.cpp`: `attn_prep<<<dim3(N_Q + N_KV, rows),
        // THREADS>>>`), and the staged core then reads the cache itself, so there is neither a per-row write
        // mask nor a separate write base (`engine/attn_fattn_kernels.h`'s `fattn_plan_ok` note: the cache
        // holds "every lane this kernel reads — including a row's own new lane"). Suppressing the write would
        // mean a new `CoreArgs` field read by the kernel every prefill wave runs, i.e. a shipping-kernel
        // change with its own device admission — not this increment.
        //
        // So the probe saves the lanes and puts them back, which delivers the property that matters: no reader
        // can observe a live cache that differs from the one the step wrote, because the repair is enqueued on
        // the same stream right after the body and before anything else is submitted. It is also the cheaper
        // copy the decision asked to be priced first: one lane PER ROW each way (24,576 B at 6 attention layers
        // x 2 KV heads x K and V x 256 elements x 2 bytes x 2 rows), not a prefix copy (47 MB at 4,032 lanes). That the
        // leak is confined to this lane is NOT assumed — it is what the guard's live before/after diff measures
        // lane by lane, and a body that wrote anywhere else still goes red.
        //
        // Row 1's lane (`anchor_pos + 1`) is reverted for the same reason as the others, even though §08.6
        // says a lane above the prefix is unread until the next append overwrites it: "the next step overwrites
        // it" is true of the SHIPPING append, but the next VERIFY round anchors on that same lane and reads it
        // through its own row 0 before that row's staging is what it scores. Measured, before this was fixed:
        // row 0 of the body at anchors 18/20/22/24 moved by 1−cos 4.4e-05..5.4e-05 between a run whose row-1
        // candidate was 25 and one whose was 777, while two runs with the same candidate were byte-identical
        // at every anchor — i.e. a real candidate-to-row-0 path through the live cache, deterministic and far
        // too big to be rounding. Reverting every row's lane closes it: after a round the cache holds only what
        // the round was given.
        {
            const engine::mtpmeter::PhaseGuard save(engine::mtpmeter::kLaneSave);
            for (int row = 0; row < lanes.lanes; ++row) {
                copy_kv_lane(lanes.stage + (long long)row * lanes.bytes_per_lane, target,
                             anchor_pos + row, /*to_cache=*/false);
            }
        }
        // One body, two rows, positions q and q+1, over the populated KV. This is the funded shape and it
        // is the shipping order — `submit_token_body` is this call with n = 1.
        //
        // The phase name splits on ONE thing the caller can vary: whether a mid-round checkpoint is armed.
        // `--mtp-midsplit 1` runs this body a second time with `ckpt` set and everything else identical,
        // so `body_with_checkpoint - body` is that checkpoint's price at one anchor in one window — which
        // is the only honest way to price it, because the capture itself is 2 submissions PER GDN LAYER
        // interleaved inside `submit_wave_body` and no boundary exists for it at this level.
        {
            const engine::mtpmeter::PhaseGuard body(ckpt != nullptr ? engine::mtpmeter::kBodyCkpt
                                                                    : engine::mtpmeter::kBody);
            submit_wave_body(weights, sh, sc, tokens, kRows);
        }
        if (tail_poison != nullptr) {
            const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
            // Read BEFORE the revert, which puts the tail lane back and would otherwise erase the only evidence
            // that row 1 appended to a lane at all. The witness is about where the body wrote, and the revert
            // is about what the live cache ends up holding; ordering them here is the whole point.
            *tail_poison = poisoned_lane_report(target, anchor_pos + kRows - 1);
        }
        {
            const engine::mtpmeter::PhaseGuard revert(engine::mtpmeter::kLaneRevert);
            for (int row = 0; row < lanes.lanes; ++row) {
                copy_kv_lane(lanes.stage + (long long)row * lanes.bytes_per_lane, target,
                             anchor_pos + row, /*to_cache=*/true);
            }
        }
        r.lane_bytes = lanes.bytes;
        if (capture_end_state) {
            // What the shadow holds when the body finishes: S_{q+2}. Compared against the same capture
            // from the unsplit body, this is the half of the split admission the logits cannot make —
            // two routes can agree on a row and disagree on the state they hand the next step.
            r.end_state.resize((size_t)snap.bytes_gdn + (size_t)snap.bytes_conv);
            const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
            VERIFY_CK(hipMemcpy(r.end_state.data(), sh.gdn_state, (size_t)snap.bytes_gdn,
                                hipMemcpyDeviceToHost));
            VERIFY_CK(hipMemcpy(r.end_state.data() + snap.bytes_gdn, sh.conv_window,
                                (size_t)snap.bytes_conv, hipMemcpyDeviceToHost));
        }
        if (ckpt != nullptr) {
            // The checkpoint itself, in bytes, so the driver can ask the decisive question: is this the
            // mixer state the shipping ONE-ROW step leaves after consuming the same token at the same
            // position? Nothing else can answer it, because nothing else knows what S_{p+1} is.
            r.mid_state.resize((size_t)snap.bytes_gdn + (size_t)snap.bytes_conv);
            const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
            VERIFY_CK(hipMemcpy(r.mid_state.data(), ckpt->mid_gdn, (size_t)snap.bytes_gdn,
                                hipMemcpyDeviceToHost));
            VERIFY_CK(hipMemcpy(r.mid_state.data() + snap.bytes_gdn, ckpt->mid_conv,
                                (size_t)snap.bytes_conv, hipMemcpyDeviceToHost));
        }
        // ONE pass over the 485 MiB table serving both norm rows, then ONE download of both logits rows.
        // `sc.norm`'s rows are contiguous at stride kHidden, which is the layout the kernel stages.
        //
        // `head` gets a `sync_added=1` close. Without it the head's ~2 ms would be charged to whatever
        // blocks next — today that is the logits D2H two lines below, which is exactly why `logits_d2h`
        // would otherwise read as the vocabulary download when most of it is the head. The sync costs
        // nothing the round was hiding: the D2H already waited for this launch on the same stream.
        {
            const engine::mtpmeter::PhaseGuard head(engine::mtpmeter::kHead);
            kernels::lm_head_batched_xbf16((const uint16_t*)weights.at(kHead).device, sc.norm,
                                           logits_dev, kVocab, kRows);
        }
        r.head_passes = 1;
        {
            const engine::mtpmeter::PhaseGuard d2h(engine::mtpmeter::kLogitsD2H);
            r.logits.assign((size_t)kRows * kVocab, 0.f);
            VERIFY_CK(hipMemcpy(r.logits.data(), logits_dev, (size_t)kRows * kVocab * sizeof(float),
                                hipMemcpyDeviceToHost));
        }
        r.d2h_count = 1;
        r.bytes_read_back = (long long)kRows * kVocab * sizeof(float);
        {
            const engine::mtpmeter::PhaseGuard argmax(engine::mtpmeter::kArgmax);
            for (int q = 0; q < kRows; ++q) {
                const float* row = r.logits.data() + (size_t)q * kVocab;
                r.token[q] = argmax_greedy(row, kVocab);   // §03.4's rule, the engine's only spelling
                r.top[q] = row[r.token[q]];
            }
        }
        if (verdict_dev != nullptr || verdict_partials != nullptr) {
            // Both or neither. The device verdict is a PAIR of launches now and the second one reads the
            // first's buffer, so half a witness would fault — and a witness that silently did not run would
            // be read as a witness that passed (`AGENTS.md`: a check whose input can be empty cannot fail).
            if (verdict_dev == nullptr || verdict_partials == nullptr) {
                throw std::runtime_error(
                    "mtp_verify: the device verdict needs its status word AND its partials tile");
            }
            // The device verdict, on the SAME tile the host argmaxes above just read. The admission is
            // `verdict[kRow0Token] == token[0] && verdict[kRow1Token] == token[1]` against §03.4's own
            // host spelling, tie rule included, on real logits from real content — not on a fixture an
            // argmax kernel could have been tuned to.
            const engine::mtpmeter::PhaseGuard witness(engine::mtpmeter::kWitness);
            kernels::mtp_accept_verdict(logits_dev, kVocab, kRows, tokens[0], verdict_dev,
                                        verdict_partials);
            VERIFY_CK(hipMemcpy(r.verdict, verdict_dev, sizeof(r.verdict), hipMemcpyDeviceToHost));
            r.verdict_bytes = (long long)sizeof(r.verdict);
        }
    } catch (...) {
        sc.wave_gemm = gemm;
        sc.gdn_scan = scan;
        sc.gdn_ckpt = saved_ckpt;
        sc.attn_fattn_wide = saved_wide;
        throw;
    }
    sc.wave_gemm = gemm;
    sc.gdn_scan = scan;
    sc.gdn_ckpt = saved_ckpt;
    sc.attn_fattn_wide = saved_wide;
    return r;
}

std::string restore_witness(const Snapshot& snap) {
    // Poison first. Without this the witness would compare a region against the bytes it was copied from
    // and could not fail — and a restore that skipped a region is exactly the bug worth catching, because
    // the state it leaves is plausible rather than obviously broken.
    VERIFY_CK(hipMemset(snap.shadow_gdn, 0xA5, (size_t)snap.bytes_gdn));
    VERIFY_CK(hipMemset(snap.shadow_conv, 0xA5, (size_t)snap.bytes_conv));
    restore_snapshot(snap);

    std::vector<float> a, b;
    a.resize((size_t)snap.bytes_gdn / sizeof(float));
    b.resize((size_t)snap.bytes_gdn / sizeof(float));
    VERIFY_CK(hipMemcpy(a.data(), snap.gdn, (size_t)snap.bytes_gdn, hipMemcpyDeviceToHost));
    VERIFY_CK(hipMemcpy(b.data(), snap.shadow_gdn, (size_t)snap.bytes_gdn, hipMemcpyDeviceToHost));
    if (std::memcmp(a.data(), b.data(), (size_t)snap.bytes_gdn) != 0) {
        size_t i = 0;
        const char* pa = (const char*)a.data();
        const char* pb = (const char*)b.data();
        while (i < a.size() * sizeof(float) && pa[i] == pb[i]) ++i;
        return "GDN state differs at byte " + std::to_string(i) + " of " +
               std::to_string(snap.bytes_gdn) + " after restore";
    }
    a.resize((size_t)snap.bytes_conv / sizeof(float));
    b.resize((size_t)snap.bytes_conv / sizeof(float));
    VERIFY_CK(hipMemcpy(a.data(), snap.conv, (size_t)snap.bytes_conv, hipMemcpyDeviceToHost));
    VERIFY_CK(hipMemcpy(b.data(), snap.shadow_conv, (size_t)snap.bytes_conv, hipMemcpyDeviceToHost));
    if (std::memcmp(a.data(), b.data(), (size_t)snap.bytes_conv) != 0) {
        size_t i = 0;
        const char* pa = (const char*)a.data();
        const char* pb = (const char*)b.data();
        while (i < a.size() * sizeof(float) && pa[i] == pb[i]) ++i;
        return "conv window differs at byte " + std::to_string(i) + " of " +
               std::to_string(snap.bytes_conv) + " after restore";
    }
    return "";
}

}  // namespace engine::mtpverify
