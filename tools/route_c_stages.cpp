// tools/route_c_stages.cpp — the route (c) audit's device data collector
// (progress/2026-10-13/plan-route-c-audit-2026-10-13.md, comparisons A1 / A3 / A4).
//
// Three independently selectable modes, all correctness-only:
//   stages  — feed a real prompt through the engine's OWN `submit_wave_body`, once per route, and
//             copy out every `DecodeScratch` tile after each captured body. That is A3's material:
//             real weights, real activations, real launch order, and no second implementation of
//             any of it.
//   floor   — A1: for each routed shape, run the hand kernel and the hipBLAS call on the SAME real
//             weight and the SAME real activation tile captured above, at M = 1, 88, 512, 1024, and
//             dump both outputs side by side.
//   inject  — A4: three wrong-but-plausible operand bugs re-created at those shapes with those tiles
//             (a stale tile — the bug this route actually shipped with — a row-offset, a swapped
//             weight pointer), dumped as outputs for the same statistics.
//
// WHY IT LINKS THE ENGINE INSTEAD OF RE-TYPING IT. Every question here is "how big is the delta
// between these two GEMMs on real data", and that is only worth asking if the two GEMMs are the ones
// the engine calls and the data is the data it feeds them. So this allocates with
// `allocate_scratch`/`allocate_state`, submits with `submit_wave_body`, and calls `proj_gemv` /
// `proj_gemm` / `mlp_gateup` / `mlp_gateup_gemm` / `mlp_down` / `mlp_down_gemm` exactly as
// `engine/decode_step.cpp` does. A local copy of the call form would certify a copy, not the route
// (AGENTS.md: do not duplicate logic).
//
// NO ULP IS COMPUTED HERE. The distances are `prototype/bf16_ulp.py`'s, because its selftest exists
// for the sign trap AGENTS.md records and a second hand-rolled comparator in C++ is exactly how that
// trap survives. This file only moves bits.
//
// build: tools/build.sh route_c_stages
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/decode_step.h"
#include "engine/kernels.h"
#include "engine/loader.h"
#include "engine/state.h"

#define CK(x)                                                                                 \
    do {                                                                                      \
        const hipError_t e_ = (x);                                                            \
        if (e_ != hipSuccess) {                                                               \
            std::printf("HIP_ERROR %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_));   \
            return 2;                                                                         \
        }                                                                                     \
    } while (0)

namespace {

using engine::kHidden;
using engine::kIntermediate;
using engine::kKvProjRows;
using engine::kNumValueHeads;
using engine::kQkvRows;
using engine::kQprojRows;
using engine::kQwidth;
using engine::kValueDim;

// Which layers a snapshot after a body actually holds. A body overwrites every row-family tile once
// per layer, so a copy taken after the body holds each tile's LAST writer: layer 23 for the
// attention and MLP tiles (`full_attention_interval` 4 → the attention layers are 3,7,…,23) and
// layer 22 for the GDN tiles (22 is the last linear-attention layer). Naming those two numbers is
// what makes the weight/activation pairing in `shape_rows` checkable rather than decorative.
constexpr int kLastAttnLayer = 23;
constexpr int kLastGdnLayer = 22;

// The widths A1 prices. `1` is the decode width the row-count rule never sends to the library (the
// floor there is the proof the rule costs nothing numerically), 512 is `kMaxWaveRows` — every
// shipped fixture's real body width — and 88 / 1024 are the plan's width-locality probes: 88 is not
// a multiple of any tile a GEMM kernel could be tiling by, and 1024 is above the engine's own
// ceiling, so a delta that grows there beyond what the 512 floor predicts is reading past a body
// boundary rather than reassociating.
constexpr int kFloorMs[] = {1, 88, 512, 1024};

std::string layer_name(int layer, const char* leaf) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "lm.layers.%d.%s", layer, leaf);
    return std::string(buf);
}

bool write_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::printf("CANNOT_WRITE %s\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char*>(data), (std::streamsize)bytes);
    return (bool)f;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) {
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

// ---------------------------------------------------------------- stage snapshot -------------
struct Stage {
    const char* name;
    const void* ptr;
    int cols;   // elements per row of the [wave_rows][cols] tile
    int elem;   // 2 = BF16, 4 = F32
};

std::vector<Stage> scratch_stages(const engine::DecodeScratch& sc) {
    auto bf = [](const char* n, const void* p, int cols) { return Stage{n, p, cols, 2}; };
    auto f3 = [](const char* n, const void* p, int cols) { return Stage{n, p, cols, 4}; };
    return {
        bf("s_a", sc.s_a, kHidden),                bf("s_b", sc.s_b, kHidden),
        // `norm` is the tile every projection in the body reads; after the last layer it is §03.4's
        // final norm, i.e. the row `lm_head` is priced on.
        bf("norm", sc.norm, kHidden),              f3("norm_f32", sc.norm_f32, kHidden),
        bf("embed", sc.embed, kHidden),            bf("delta", sc.delta, kHidden),
        // The two mixer tiles. `block_out` is the GDN gated norm's output and an attention layer
        // never writes it; `a_gated` is the attention mixer's output. The stale-operand bug this
        // audit re-creates in --inject is an o_proj call reading the former on layer 23.
        bf("block_out", sc.block_out, kValueDim),  f3("block_out_f32", sc.block_out_f32, kValueDim),
        bf("g_raw", sc.g_raw, kQkvRows),           f3("g_raw_f32", sc.g_raw_f32, kQkvRows),
        bf("g_z", sc.g_z, kValueDim),
        bf("g_a", sc.g_a, kNumValueHeads),         bf("g_b", sc.g_b, kNumValueHeads),
        f3("g_a_f32", sc.g_a_f32, kNumValueHeads), f3("g_b_f32", sc.g_b_f32, kNumValueHeads),
        f3("g_g", sc.g_g, kNumValueHeads),         f3("g_beta", sc.g_beta, kNumValueHeads),
        f3("g_u", sc.g_u, kQkvRows),               f3("g_o", sc.g_o, kValueDim),
        // Attention: the three projections, the rope'd q/k, the de-interleaved gate, the mixer
        // output and the gated mixer output (o_proj's real operand, §05.5/§05.6).
        bf("a_q", sc.a_q, kQprojRows),             bf("a_k", sc.a_k, kKvProjRows),
        bf("a_v", sc.a_v, kKvProjRows),            bf("a_q_rope", sc.a_q_rope, kQwidth),
        bf("a_k_rope", sc.a_k_rope, kKvProjRows),  bf("a_gate", sc.a_gate, kQwidth),
        bf("a_pre", sc.a_pre, kQwidth),            bf("a_gated", sc.a_gated, kQwidth),
        // MLP: the fused act, its widening, and the library route's two raw dots (§3.3's unfused
        // epilogue — they stay untouched on the hand side, which is itself worth seeing dumped).
        bf("mlp_act", sc.mlp_act, kIntermediate),  f3("mlp_act_f32", sc.mlp_act_f32, kIntermediate),
        bf("mlp_gate_raw", sc.mlp_gate_raw, kIntermediate),
        bf("mlp_up_raw", sc.mlp_up_raw, kIntermediate),
    };
}

// ------------------------------------------------------------------- A1 / A4 harness ---------
// One isolated routed shape: the two kernels that stand at that site, the real weight, and the
// scratch tile the real activation came from. The pairing is the engine's own — the captured tile is
// the body's last writer of that tile and `layer` is the layer that wrote it — so `o_proj` is tested
// with layer 23's weight on layer 23's gated mixer output and `out_proj` with layer 22's weight on
// layer 22's gated norm output.
//
// `kind` names which PAIR of kernels stands at the site, because they are not all one pair:
//   kProj   — `proj_gemv` (reads the F32 widening) vs `proj_gemm` (reads BF16). The projections.
//   kGateUp — `mlp_gateup` vs `mlp_gateup_gemm`, compared at `act`. Not a bare dot on either side:
//             the hand kernel fuses `silu(gate)·up` and the library route unfuses it into its own
//             epilogue (§3.3), so the two sides of the A/B at those two sites are the fused act and
//             the unfused one. `gate_proj`/`up_proj` are the same shapes through the kProj pair, so
//             a reader can separate the epilogue's share of the floor from the dot's.
//   kDown   — `mlp_down` (F32 act) vs `mlp_down_gemm` (BF16 act): a bare dot on both sides.
//
// `inject` names the A4 bug this row produces (empty for the legal pair). Each one is aimed at ONE
// site and is scored against the legal pair at that SAME site with the SAME operands, so a
// comparator that flags everything scores separation 1.0, not ≥10 — that is what keeps these
// negatives non-vacuous (AGENTS.md: an injected bug must appear only where the bug lives).
struct ShapeRow {
    const char* name;
    int kind;
    int layer;
    int n;
    int k;
    const char* leaf;    // the weight (or the gate weight, for kGateUp)
    const char* leaf_b;  // the up weight, for kGateUp
    const char* act;     // the captured stage the activation tile came from
    const char* inject;  // "" | "stale_tile" | "row_offset" | "swap_weight"
};

std::vector<ShapeRow> shape_rows() {
    auto proj = [&](const char* name, const char* leaf, int layer, int n, int k, const char* act) {
        return ShapeRow{name, 0, layer, n, k, leaf, "", act, ""};
    };
    return {
        proj("q_proj", "self_attn.q_proj.weight", kLastAttnLayer, kQprojRows, kHidden, "norm"),
        proj("k_proj", "self_attn.k_proj.weight", kLastAttnLayer, kKvProjRows, kHidden, "norm"),
        proj("v_proj", "self_attn.v_proj.weight", kLastAttnLayer, kKvProjRows, kHidden, "norm"),
        proj("o_proj", "self_attn.o_proj.weight", kLastAttnLayer, kHidden, kQwidth, "a_gated"),
        proj("in_proj_qkv", "linear_attn.in_proj_qkv.weight", kLastGdnLayer, kQkvRows, kHidden,
             "norm"),
        proj("in_proj_z", "linear_attn.in_proj_z.weight", kLastGdnLayer, kValueDim, kHidden, "norm"),
        proj("out_proj", "linear_attn.out_proj.weight", kLastGdnLayer, kHidden, kValueDim,
             "block_out"),
        ShapeRow{"gate_proj", 0, kLastAttnLayer, kIntermediate, kHidden, "mlp.gate_proj.weight", "",
                 "norm", ""},
        ShapeRow{"up_proj", 0, kLastAttnLayer, kIntermediate, kHidden, "mlp.up_proj.weight", "",
                 "norm", ""},
        ShapeRow{"down_proj", 2, kLastAttnLayer, kHidden, kIntermediate, "mlp.down_proj.weight", "",
                 "mlp_act", ""},
        // The fused/unfused act pair, as one row, because that is the unit the two sites share.
        ShapeRow{"gate_up_stage", 1, kLastAttnLayer, kIntermediate, kHidden, "mlp.gate_proj.weight",
                 "mlp.up_proj.weight", "norm", ""},
    };
}

std::vector<ShapeRow> inject_rows() {
    return {
        // 1. THE STALE TILE, re-created: `o_proj` pointed at `block_out`, which an attention layer
        // never writes, so it holds layer 22's gated norm output instead of this layer's mixer
        // output. engine/results.md: this is the bug that made dec8 read 37 FAIL / cos 0.42.
        ShapeRow{"inj_stale_o_proj", 0, kLastAttnLayer, kHidden, kQwidth, "self_attn.o_proj.weight",
                 "", "block_out", "stale_tile"},
        // 2. A ROW OFFSET: the activation tile's rows shifted by one, i.e. body row r dotted against
        // row r-1's activations. Same weight, same shape, one wrong base pointer.
        ShapeRow{"inj_rowoffset_q_proj", 0, kLastAttnLayer, kQprojRows, kHidden,
                 "self_attn.q_proj.weight", "", "norm", "row_offset"},
        // 3. A SWAPPED WEIGHT POINTER: k_proj's call pointed at v_proj's weight — same shape
        // ([512,1024]), same dtype, same layer, so nothing but the pointer is wrong.
        ShapeRow{"inj_swapw_k_proj", 0, kLastAttnLayer, kKvProjRows, kHidden,
                 "self_attn.k_proj.weight", "self_attn.v_proj.weight", "norm", "swap_weight"},
    };
}

struct DevBuf {
    void* p = nullptr;
    ~DevBuf() {
        if (p) (void)hipFree(p);
    }
    void** addr() { return &p; }
};

// One shape row at width M: writes <outdir>/<name>@<M>_hand.bin and _lib.bin. Returns 0 on success
// (including "skipped, not enough real activation rows", which prints a SKIP line rather than
// silently padding a tile).
int run_shape_row(const engine::WeightStore& store, const ShapeRow& row, int m,
                  const std::map<std::string, std::vector<uint16_t>>& acts, int avail,
                  const std::string& outdir) {
    const auto found = acts.find(row.act);
    if (found == acts.end()) {
        std::printf("MISSING_ACTIVATION %s for %s\n", row.act, row.name);
        return 1;
    }
    const std::vector<uint16_t>& tile = found->second;
    const int act_cols = (int)(tile.size() / (size_t)avail);
    if (act_cols != row.k) {
        std::printf("SHAPE_MISMATCH %s: activation tile %s is %d wide, shape says k=%d\n", row.name,
                    row.act, act_cols, row.k);
        return 1;
    }
    const int shift = std::string(row.inject) == "row_offset" ? 1 : 0;
    if (m + shift > avail) {
        std::printf("SKIP %s M=%d: only %d real activation rows captured (shift %d)\n", row.name, m,
                    avail, shift);
        return 0;
    }
    // The M×k BF16 tile and its exact F32 widening — exact because a BF16 widened to F32 loses
    // nothing, which is the claim the engine's own call sites rest on. A row-offset injection reads
    // row r+1 of the real tile for body row r; nothing is invented to fill a short tile.
    std::vector<uint16_t> xb((size_t)m * row.k);
    for (int r = 0; r < m; ++r) {
        std::memcpy(&xb[(size_t)r * row.k], &tile[(size_t)(r + shift) * act_cols],
                    (size_t)row.k * 2);
    }
    std::vector<float> xf(xb.size());
    for (size_t i = 0; i < xb.size(); ++i) {
        const uint32_t bits = (uint32_t)xb[i] << 16;   // BF16 payload = the float's top half
        std::memcpy(&xf[i], &bits, 4);
    }

    const void* w_a = store.at(layer_name(row.layer, row.leaf)).device;
    const void* w_b = row.leaf_b[0] ? store.at(layer_name(row.layer, row.leaf_b)).device : nullptr;
    const bool swap = std::string(row.inject) == "swap_weight";
    const void* weight = swap ? w_b : w_a;
    const size_t out_bytes = (size_t)m * row.n * 2;

    DevBuf dx, dx32, dout, dgate, dup;
    CK(hipMalloc(dx.addr(), xb.size() * 2));
    CK(hipMalloc(dx32.addr(), xf.size() * 4));
    CK(hipMalloc(dout.addr(), out_bytes));
    CK(hipMemcpy(dx.p, xb.data(), xb.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dx32.p, xf.data(), xf.size() * 4, hipMemcpyHostToDevice));
    void* dg = nullptr;
    void* du = nullptr;
    if (row.kind == 1) {
        CK(hipMalloc(&dg, out_bytes));
        CK(hipMalloc(&du, out_bytes));
    }

    // Poison the destination with a value no product of these operands stores, so an unwritten row
    // shows up as a survivor count in the audit instead of as a plausible-looking partial tensor
    // (`tools/prefill_mix.cpp --check`'s lesson, applied to a side buffer for the same reason).
    const std::vector<uint16_t> poison(out_bytes / 2, 0x7F7F);

    for (int side = 0; side < 2; ++side) {
        const bool lib = side == 1;
        CK(hipMemcpy(dout.p, poison.data(), out_bytes, hipMemcpyHostToDevice));
        if (row.kind == 1) {
            if (lib) {
                engine::kernels::mlp_gateup_gemm(w_a, w_b, dx.p, dg, du, (uint16_t*)dout.p, row.n,
                                                 row.k, m);
            } else {
                engine::kernels::mlp_gateup((const uint16_t*)w_a, (const uint16_t*)w_b,
                                            (const float*)dx32.p, (uint16_t*)dout.p, row.n,
                                            row.k, m, engine::kernels::MlpGateMap::kInterleave);
            }
        } else if (row.kind == 2) {
            if (lib) {
                engine::kernels::mlp_down_gemm(weight, dx.p, (uint16_t*)dout.p, row.n,
                                               row.k, m);
            } else {
                engine::kernels::mlp_down((const uint16_t*)weight, (const float*)dx32.p,
                                          (uint16_t*)dout.p, row.n, row.k, m);
            }
        } else if (lib) {
            engine::kernels::proj_gemm(weight, dx.p, (uint16_t*)dout.p, row.n, row.k, m);
        } else {
            engine::kernels::proj_gemv(weight, (const float*)dx32.p, (uint16_t*)dout.p,
                                       row.n, row.k, m);
        }
        CK(hipDeviceSynchronize());
        std::vector<uint16_t> got(out_bytes / 2);
        CK(hipMemcpy(got.data(), dout.p, out_bytes, hipMemcpyDeviceToHost));
        size_t survivors = 0;
        for (uint16_t v : got) survivors += (v == 0x7F7F);
        const std::string tag = lib ? "lib" : "hand";
        std::printf("FLOORSAMPLE %s M=%d side=%s n=%d k=%d bytes=%zu poison_survivors=%zu\n",
                    row.name, m, tag.c_str(), row.n, row.k, out_bytes, survivors);
        if (survivors) return 1;   // a kernel that wrote nothing must not reach the statistics
        if (!write_file(outdir + "/" + std::string(row.name) + "@" + std::to_string(m) + "_" +
                            tag + ".bin", got.data(), got.size() * 2)) {
            return 1;
        }
    }
    return 0;
}

struct Args {
    std::string model = "models/Qwen3.5-0.8B";
    std::string ids;
    std::string out = "tmp/route_c";
    std::string routes = "hand,library";
    std::string capture = "0";
    bool stages = true;
    bool floor = false;
    bool inject = false;
    int rows = engine::kMaxWaveRows;
};

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) throw std::runtime_error(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--model") a.model = next();
        else if (s == "--ids") a.ids = next();
        else if (s == "--out") a.out = next();
        else if (s == "--routes") a.routes = next();
        else if (s == "--capture") a.capture = next();
        else if (s == "--rows") a.rows = std::atoi(next().c_str());
        else if (s == "--stages") a.stages = next() == "1";
        else if (s == "--floor") a.floor = true;
        else if (s == "--inject") a.inject = true;
        else {
            std::printf("usage: route_c_stages --model D --ids CSV --out DIR [--rows N]"
                        " [--capture 0,1] [--routes hand,library] [--stages 1|0] [--floor]"
                        " [--inject]\n");
            return 2;
        }
    }
    std::vector<int> ids;
    for (const std::string& t : split_csv(a.ids)) ids.push_back(std::atoi(t.c_str()));
    std::vector<int> capture;
    for (const std::string& t : split_csv(a.capture)) capture.push_back(std::atoi(t.c_str()));
    if (ids.empty()) {
        std::printf("route_c_stages: --ids is empty\n");
        return 2;
    }
    if (a.rows < 1 || a.rows > engine::kMaxWaveRows) {
        std::printf("route_c_stages: --rows must be 1..%d (kMaxWaveRows)\n", engine::kMaxWaveRows);
        return 2;
    }
    if ((a.floor || a.inject) && !a.stages) {
        // The real activations come out of the stage pass, in this process, from the route the
        // shipped bits are made of. Asking for a floor without stages would mean feeding the
        // isolated GEMMs synthetic data, which is the thing this audit is supposed to avoid.
        std::printf("route_c_stages: --floor/--inject need --stages 1 in the same run\n");
        return 2;
    }
    std::system(("mkdir -p " + a.out + "/floor").c_str());

    engine::LoadStats stats;
    engine::WeightStore::Options o;
    o.model_dir = a.model;
    o.measure_steady_h2d = false;
    engine::WeightStore store = engine::WeightStore::load(o, stats);
    std::printf("LOADED tensors=%zu resident_mib=%.1f\n", stats.n_loaded,
                (double)store.resident_bytes() / (1024.0 * 1024.0));

    // The activation tiles A1/A4 read: stage name → rows, concatenated across captured bodies so a
    // run that captures two 512-row bodies can price M = 1024 on 1024 real rows.
    std::map<std::string, std::vector<uint16_t>> acts;
    int avail = 0;

    if (a.stages) {
        std::vector<std::string> stage_desc;
        const std::vector<std::string> routes = split_csv(a.routes);
        for (size_t ri = 0; ri < routes.size(); ++ri) {
            const std::string& rname = routes[ri];
            const engine::kernels::WaveGemm route = engine::kernels::parse_wave_gemm(rname);
            if (route == engine::kernels::WaveGemm::kLibrary) {
                std::printf("GEMM %s\n", engine::kernels::gemm_route_init().c_str());
            }
            engine::RequestState st = engine::allocate_state((int64_t)ids.size() + 1);
            engine::DecodeScratch sc = engine::allocate_scratch(a.rows);
            sc.wave_gemm = route;
            engine::prepare_weight_adapters(store, sc);

            const int bodies = (int)((ids.size() + a.rows - 1) / a.rows);
            for (int b = 0; b < bodies; ++b) {
                const int off = b * a.rows;
                const int n = (int)std::min<size_t>((size_t)a.rows, ids.size() - off);
                engine::submit_wave_body(store, st, sc, ids.data() + off, n);
                CK(hipDeviceSynchronize());
                if (std::find(capture.begin(), capture.end(), b) == capture.end()) continue;
                const std::string dir = a.out + "/" + rname + "/body" + std::to_string(b);
                std::system(("mkdir -p " + dir).c_str());
                for (const Stage& s : scratch_stages(sc)) {
                    if (!s.ptr) {
                        std::printf("STAGE_NULL %s\n", s.name);
                        return 1;
                    }
                    const size_t bytes = (size_t)n * s.cols * s.elem;
                    std::vector<char> host(bytes);
                    CK(hipMemcpy(host.data(), s.ptr, bytes, hipMemcpyDeviceToHost));
                    if (!write_file(dir + "/" + s.name + ".bin", host.data(), bytes)) return 1;
                    if (ri == 0 && s.elem == 2) {
                        // A1's activations come from the HAND side (that is the route the shipped bits
                        // are made of, so the floor is measured on the data that path feeds), and
                        // only the BF16 tiles: the harness widens them itself, so an F32 stage would
                        // enter `acts` with a row count derived from a 4-byte element.
                        std::vector<uint16_t>& v = acts[s.name];
                        v.resize(v.size() + bytes / 2);
                        std::memcpy(&v[v.size() - bytes / 2], host.data(), bytes);
                    }
                    if (ri == 0) {
                        std::ostringstream d;
                        d << "  {\"name\": \"" << s.name << "\", \"dtype\": "
                          << (s.elem == 2 ? "\"BF16\"" : "\"F32\"") << ", \"rows\": " << n
                          << ", \"cols\": " << s.cols << "}";
                        stage_desc.push_back(d.str());
                    }
                }
                if (ri == 0) avail += n;   // the hand route's rows only: `avail` is what every tile
                                           // width in `run_shape_row` is derived from, so counting
                                           // the library pass too would mis-derive all of them.
            }
            engine::free_scratch(sc);
            engine::free_state(st);
            if (route == engine::kernels::WaveGemm::kLibrary) {
                engine::kernels::gemm_route_shutdown();
            }
        }
        std::ostringstream manifest;
        // The route names are quoted; the body indices are numbers. Writing `"routes": [hand,library]`
        // because the string happens to be comma-separated is how a manifest becomes unparsable.
        std::string routes_json;
        for (size_t i = 0; i < split_csv(a.routes).size(); ++i) {
            routes_json += "\"" + split_csv(a.routes)[i] + "\""
                           + (i + 1 < split_csv(a.routes).size() ? "," : "");
        }
        manifest << "{\n \"prompt_tokens\": " << ids.size() << ",\n \"rows\": " << a.rows
                 << ",\n \"bodies\": " << (ids.size() + a.rows - 1) / a.rows << ",\n \"capture\": ["
                 << a.capture << "],\n \"routes\": [" << routes_json
                 << "],\n \"activation_rows\": " << avail << ",\n \"stages\": [\n";
        for (size_t i = 0; i < stage_desc.size(); ++i) {
            manifest << stage_desc[i] << (i + 1 < stage_desc.size() ? "," : "");
        }
        manifest << "\n ]\n}\n";
        const std::string doc = manifest.str();
        write_file(a.out + "/stages.json", doc.data(), doc.size());
        std::printf("STAGES out=%s activation_rows=%d stages=%zu\n", a.out.c_str(), avail,
                    acts.size());
    }

    if (a.floor) {
        for (int m : kFloorMs) {
            for (const ShapeRow& row : shape_rows()) {
                if (run_shape_row(store, row, m, acts, avail, a.out + "/floor")) return 1;
            }
        }
    }
    if (a.inject) {
        // Injected at every width the floor was priced at, so each injection's legal denominator is
        // the same shape's hand-vs-library pair at the SAME M. A row-offset at M = the full capture
        // would read one row past the tile, so that one is skipped out loud rather than padded.
        for (const ShapeRow& row : inject_rows()) {
            for (int m : kFloorMs) {
                if (run_shape_row(store, row, m, acts, avail, a.out + "/floor")) return 1;
            }
        }
    }
    return 0;
}
