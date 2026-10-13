// tools/prefill_mix.cpp — the measurement instrument for §7 W1/W2/W4 of
// bench/prefill-route-spike-2026-10-13.md. It produces rows; it does not produce verdicts.
//
// What it measures is the engine's own WAVE MATMUL MIX, not a square: the ten library shapes of §6
// item 1 at their real per-layer counts (6 self-attn layers, 18 linear-attn, 24 MLP), plus the
// skinny shapes W4 names, timed per call. A square cannot answer W1/W2 because those thresholds are
// defined on the mix — at M=64 weights are 95 % of everything the wave touches, and the N=16 pair
// sits below the arithmetic-intensity ridge at every width (§3.2).
//
// Shapes and per-layer counts come from `tools/prefill_gemm_shapes.py --json`, never from anything
// typed in here. That census is the source of shape truth; if it and this tool ever disagree, the
// census has to be wrong in a way `--list` reports, not silently divergent.
//
// Build:  tools/build.sh prefill_mix
// Spec:   .venv/bin/python tools/prefill_gemm_shapes.py --json > tools/build/prefill_gemm_shapes.json
// Run:    tools/build/prefill_mix [--spec PATH] MODE [options]
//
// Modes
//   --list              CPU only, no device call at all. Prints the census as this tool expanded it:
//                       per-group call counts, flops, weight bytes, and the consistency checks. Run
//                       it first; it is where a census/harness divergence shows up.
//   --check             CORRECTNESS ONLY, no timing: one small shape per family, C written 16-bit,
//                       compared bit-for-bit against a CPU dot of the same 16-bit inputs rounded the
//                       same way (the comparator is gemm_peak --check's). This is the mode that says
//                       whether the mix harness computes what it claims — §4.3's bf16-C bit-for-bit
//                       claim re-asked at the mix's own shapes and call form.
//   --mix-determinism   CORRECTNESS ONLY, no timing: three identical passes of the whole mix, output
//                       bytes hashed, plus a fresh handle given the same workspace, plus the beta==0
//                       filler check. §3.1 proved determinism per shape; the mix's call order, and
//                       the handle that has already seen 150 calls, are a different question.
//   --smoke             Builds each timing plan exactly as --mix/--skinny would (every allocation,
//                       every fill, the whole group expansion at the same Mmax) and prints what came
//                       out, without issuing a single GEMM. Correctness-only: it is how the setup of
//                       the timing modes gets proven without spending an exclusive window, and it is
//                       where to look when a row asks for more memory than expected.
//   --mix               W1 (bf16) / W2 (fp16): aggregate flops ÷ wall over a whole pass of the mix, at
//                       each --M, for one attn layer, one gdn layer and all 24 layers.
//   --skinny            W4: the per-call cost of in_proj_a/in_proj_b (N=16), the a|b-fused N=32
//                       variant §6 offers as the alternative, and k_proj/v_proj (N=512).
//
// Honesty rules this file follows (AGENTS.md):
//   * Timings come from steady_clock bursts only. `hipEventElapsedTime` returns exactly 0 for kernels
//     below ~1 ms on this iGPU — a broken timer, not a fast kernel.
//   * Warm pass before any timed pass: the first GPU pass over fresh pages runs ~35 GB/s here.
//   * Best-of-N with the spread printed, always.
//   * No thresholds, no verdicts on a timing row. W1/W2/W4's bands live in §7 and are applied outside
//     this tool; a harness that prints a verdict gets read as one even when it is not, and a rate
//     printed here is not a ledger row. Nothing this tool prints belongs in
//     bench/performance-status.md or devices/. The two correctness modes are the only ones a worker
//     may run, and neither can emit an ms figure because neither contains a timing loop.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gemm_common.h"

namespace {

using namespace gemm_common;

// -----------------------------------------------------------------------------------------------
// The census file: the only place this tool gets shapes and per-layer counts from.
// -----------------------------------------------------------------------------------------------

struct CensusRow {
    std::string leaf, scope, dtype;
    long long count = 0, n = 0, k = 0;
};

struct Census {
    std::string model;
    long long interval = 0;
    long long n_attn = 0, n_gdn = 0, n_all = 0;
    long long block_matmuls = 0, block_flops = 0;
    std::vector<long long> attn_layers;   // 0-based, from config.json's own full_attention_interval
    std::vector<CensusRow> rows;
};

bool skip_ws(const std::string &s, size_t &p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p;
    return p < s.size();
}

// Locate `"key"` followed by a colon; vpos ends up on the first character of the value.
bool find_key(const std::string &s, const std::string &key, size_t &vpos) {
    const std::string needle = "\"" + key + "\"";
    const size_t at = s.find(needle);
    if (at == std::string::npos) return false;
    size_t p = at + needle.size();
    if (!skip_ws(s, p) || s[p] != ':') return false;
    ++p;
    return skip_ws(s, p) ? (vpos = p, true) : false;
}

bool read_str(const std::string &s, size_t &p, std::string &out) {
    if (p >= s.size() || s[p] != '"') return false;
    const size_t close = s.find('"', p + 1);
    if (close == std::string::npos) return false;
    out = s.substr(p + 1, close - p - 1);
    p = close + 1;
    return true;
}

bool read_num(const std::string &s, size_t &p, long long &out) {
    const size_t start = p;
    while (p < s.size() && (isdigit((unsigned char)s[p]) || s[p] == '-' || s[p] == '+')) ++p;
    if (p == start) return false;
    out = strtoll(s.substr(start, p - start).c_str(), nullptr, 10);
    return true;
}

bool read_int_array(const std::string &s, size_t &p, std::vector<long long> &out) {
    if (p >= s.size() || s[p] != '[') return false;
    ++p;
    while (true) {
        if (!skip_ws(s, p)) return false;
        if (s[p] == ']') { ++p; return true; }
        long long v;
        if (!read_num(s, p, v)) return false;
        out.push_back(v);
        if (!skip_ws(s, p)) return false;
        if (s[p] == ',') { ++p; continue; }
        if (s[p] == ']') { ++p; return true; }
        return false;
    }
}

// One flat object: {"a":1,"b":"x"}. Values are kept as the raw token and converted by the caller, so
// an unexpected nested value is a reportable string rather than a silent zero.
bool read_flat_obj(const std::string &s, size_t &p,
                   std::vector<std::pair<std::string, std::string>> &out) {
    if (p >= s.size() || s[p] != '{') return false;
    ++p;
    while (true) {
        if (!skip_ws(s, p)) return false;
        if (s[p] == '}') { ++p; return true; }
        std::string key;
        if (!read_str(s, p, key)) return false;
        if (!skip_ws(s, p) || s[p] != ':') return false;
        ++p;                       // past the colon, onto the value
        if (!skip_ws(s, p)) return false;
        std::string val;
        if (s[p] == '"') {
            if (!read_str(s, p, val)) return false;
        } else {
            const size_t start = p;
            while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != '\n') ++p;
            val = s.substr(start, p - start);
            while (!val.empty() && (val.back() == ' ' || val.back() == '\r')) val.pop_back();
        }
        out.emplace_back(key, val);
        if (!skip_ws(s, p)) return false;
        if (s[p] == ',') { ++p; continue; }
        if (s[p] == '}') { ++p; return true; }
        return false;
    }
}

bool flat_get(const std::vector<std::pair<std::string, std::string>> &o, const std::string &key,
              std::string &out) {
    for (const auto &kv : o) if (kv.first == key) { out = kv.second; return true; }
    return false;
}

// A deliberately small reader for the ONE JSON file this tool may take shapes from, emitted by
// `tools/prefill_gemm_shapes.py --json`. It understands that file's structure and nothing else: flat
// records of string or integer values, plus the three nested places the emitter puts them (counts,
// attn_layers, matmuls). Strings are read to the closing quote and never unescaped, which is safe
// because the emitter writes ASCII (its one non-ASCII leaf name, the tied lm_head row, arrives as a
// \uXXXX escape and is filtered by scope before anything compares it). Anything unparseable is an
// error rather than a default: a shape that silently failed to load would make a mix row measure less
// than it claims, which is the failure mode this file exists to avoid.
bool load_census(const std::string &path, Census &c, std::string &err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open '" + path + "' — generate it with:\n  .venv/bin/python tools/prefill_gemm_shapes.py --json > " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();

    size_t p = 0;
    std::string tmp;
    if (!find_key(s, "schema", p) || !read_str(s, p, tmp)) { err = "no \"schema\" string in " + path; return false; }
    if (tmp != "prefill_gemm_shapes/1") { err = "unsupported census schema '" + tmp + "' in " + path; return false; }
    if (!find_key(s, "model", p) || !read_str(s, p, c.model)) { err = "no \"model\" in " + path; return false; }
    if (!find_key(s, "full_attention_interval", p) || !read_num(s, p, c.interval)) { err = "no \"full_attention_interval\" in " + path; return false; }
    if (!find_key(s, "counts", p)) { err = "no \"counts\" in " + path; return false; }
    {
        std::vector<std::pair<std::string, std::string>> cnt;
        if (!read_flat_obj(s, p, cnt)) { err = "cannot parse \"counts\" in " + path; return false; }
        std::string v;
        if (!flat_get(cnt, "attn", v)) { err = "counts has no \"attn\""; return false; }
        c.n_attn = strtoll(v.c_str(), nullptr, 10);
        if (!flat_get(cnt, "gdn", v)) { err = "counts has no \"gdn\""; return false; }
        c.n_gdn = strtoll(v.c_str(), nullptr, 10);
        if (!flat_get(cnt, "all", v)) { err = "counts has no \"all\""; return false; }
        c.n_all = strtoll(v.c_str(), nullptr, 10);
    }
    if (!find_key(s, "attn_layers", p) || !read_int_array(s, p, c.attn_layers)) { err = "no \"attn_layers\" in " + path; return false; }
    if (!find_key(s, "block_matmuls_per_wave", p) || !read_num(s, p, c.block_matmuls)) { err = "no \"block_matmuls_per_wave\" in " + path; return false; }
    if (!find_key(s, "block_flops_per_token", p) || !read_num(s, p, c.block_flops)) { err = "no \"block_flops_per_token\" in " + path; return false; }
    if (!find_key(s, "matmuls", p)) { err = "no \"matmuls\" in " + path; return false; }
    if (p >= s.size() || s[p] != '[') { err = "\"matmuls\" is not an array in " + path; return false; }
    ++p;
    while (true) {
        if (!skip_ws(s, p)) { err = "\"matmuls\" array is unterminated"; return false; }
        if (s[p] == ',') { ++p; continue; }
        if (s[p] == ']') { ++p; break; }
        if (s[p] != '{') { err = "unexpected character in the \"matmuls\" array"; return false; }
        std::vector<std::pair<std::string, std::string>> rec;
        if (!read_flat_obj(s, p, rec)) {
            err = "cannot parse a matmuls record at offset " + std::to_string(p) + ": ..." +
                  s.substr(p > 40 ? p - 40 : 0, 60) + "...";
            return false;
        }
        CensusRow r;
        std::string v;
        if (!flat_get(rec, "leaf", v)) { err = "record with no \"leaf\""; return false; }
        r.leaf = v;
        if (!flat_get(rec, "scope", v)) { err = r.leaf + ": no \"scope\""; return false; }
        r.scope = v;
        if (!flat_get(rec, "dtype", v)) { err = r.leaf + ": no \"dtype\""; return false; }
        r.dtype = v;
        if (!flat_get(rec, "count", v)) { err = r.leaf + ": no \"count\""; return false; }
        r.count = strtoll(v.c_str(), nullptr, 10);
        if (!flat_get(rec, "n", v)) { err = r.leaf + ": no \"n\""; return false; }
        r.n = strtoll(v.c_str(), nullptr, 10);
        if (!flat_get(rec, "k", v)) { err = r.leaf + ": no \"k\""; return false; }
        r.k = strtoll(v.c_str(), nullptr, 10);
        c.rows.push_back(r);
    }
    if (c.rows.empty()) { err = "\"matmuls\" is empty"; return false; }
    return true;
}

// The census's own arithmetic, re-derived here so the two can disagree visibly. Not a threshold: the
// harness refusing to measure a shape set it cannot account for.
bool census_consistent(const Census &c, std::vector<std::string> &problems) {
    if (c.n_attn + c.n_gdn != c.n_all)
        problems.push_back("counts do not add: attn " + std::to_string(c.n_attn) + " + gdn " +
                           std::to_string(c.n_gdn) + " != all " + std::to_string(c.n_all));
    if ((long long)c.attn_layers.size() != c.n_attn)
        problems.push_back("attn_layers lists " + std::to_string(c.attn_layers.size()) +
                           " layers but counts.attn says " + std::to_string(c.n_attn));
    long long matmuls = 0, flops = 0;
    for (const CensusRow &r : c.rows) {
        if (r.scope == "once") continue;
        matmuls += r.count;
        flops += 2LL * r.n * r.k * r.count;
    }
    if (matmuls != c.block_matmuls)
        problems.push_back("the rows sum to " + std::to_string(matmuls) + " block matmuls but the census carries " +
                           std::to_string(c.block_matmuls));
    if (flops != c.block_flops)
        problems.push_back("the rows sum to " + std::to_string(flops) + " flops/token but the census carries " +
                           std::to_string(c.block_flops));
    for (const CensusRow &r : c.rows) {
        if (r.scope == "once") continue;
        if (r.n <= 0 || r.k <= 0 || r.count <= 0)
            problems.push_back(r.leaf + ": non-positive n/k/count");
        if (r.dtype != "BF16")
            problems.push_back(r.leaf + ": shard dtype is " + r.dtype +
                               ", and this harness only speaks 16-bit in / 16-bit C");
    }
    return problems.empty();
}

const CensusRow *census_find(const Census &c, const std::string &leaf);

// -----------------------------------------------------------------------------------------------
// The route. This names WHICH matmuls are in the mix; it says nothing about their sizes or their
// per-layer counts, which come from the census. §6 item 1 of the spike memo.
// -----------------------------------------------------------------------------------------------

const char *kLibraryLeaves[] = {
    "self_attn.q_proj.weight",      "self_attn.k_proj.weight",  "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",      "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",  "mlp.gate_proj.weight",     "mlp.up_proj.weight",
    "mlp.down_proj.weight",
};
// W4's skinny pair: out of the mix by route (§6 item 2 keeps them in the hand kernel) but timed on
// their own because that is what W4 asks. The fused variant is §6's alternative if the pair turns out
// to be dispatch-bound: one N=32 call instead of two N=16 calls per layer.
const char *kSkinnyLeaves[] = {"linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"};

bool is_library_leaf(const std::string &leaf) {
    for (const char *l : kLibraryLeaves) if (leaf == l) return true;
    return false;
}

// The route names shapes; the census has to have them. A rename in the checkpoint or the census shows
// up here, as a named missing row, before anything allocates or measures.
bool route_complete(const Census &c, std::vector<std::string> &missing) {
    for (const char *leaf : kLibraryLeaves)
        if (!census_find(c, leaf)) missing.push_back(leaf);
    for (const char *leaf : kSkinnyLeaves)
        if (!census_find(c, leaf)) missing.push_back(leaf);
    return missing.empty();
}

const CensusRow *census_find(const Census &c, const std::string &leaf) {
    for (const CensusRow &r : c.rows) if (r.leaf == leaf) return &r;
    return nullptr;
}

// -----------------------------------------------------------------------------------------------
// The expanded plan: one Site per matmul call, grouped the ways §7 asks to measure them.
// -----------------------------------------------------------------------------------------------

struct Site {
    std::string name;   // printed identity, e.g. "L19 self_attn.q_proj"
    std::string leaf;
    long long layer = -1;
    int n = 0, k = 0;
    void *A = nullptr;  // shared per-K activation buffer (reads the same bytes a per-site one would)
    void *W = nullptr;  // [n,k], as the checkpoint stores it
    void *C = nullptr;  // [Mmax,n]
};

struct Group {
    std::string name;
    std::vector<size_t> sites;
};

struct DType {
    hipDataType dt;
    hipDataType dt_out;  // always its own 16-bit width: bf16 in / bf16 C, fp16 in / fp16 C (§4.3 —
                         // the crossed pair is a refusal, not a conversion)
    const char *name;
    bool is_bf;
};

struct Plan {
    std::vector<Site> sites;
    std::vector<Group> mix_groups;   // attn-layer, gdn-layer, all-24-layers
    std::vector<Group> w4_groups;    // the skinny/small-N rows plus the fused variant
    std::map<int, void *> A_by_k;
    size_t weight_bytes = 0, c_bytes = 0;
};

size_t add_site(Plan &plan, long long layer, const std::string &leaf, int n, int k) {
    Site s;
    char buf[160];
    std::snprintf(buf, sizeof buf, "L%-2lld %s", layer, leaf.c_str());
    s.name = buf;
    s.leaf = leaf;
    s.layer = layer;
    s.n = n;
    s.k = k;
    plan.sites.push_back(s);
    return plan.sites.size() - 1;
}

// Build the wave. Library sites come from every layer in the model's own order (the census's attn
// indices, everything else linear-attn), so the all-24 group is the call order a route would issue
// and the weight footprint is the real one: 150 distinct weight tensors, not 150 references to 10.
bool build_plan(const Census &c, int mmax, const DType &dt,
                const std::vector<std::string> &want_groups, bool with_w4, Plan &plan,
                std::string &err) {
    std::vector<std::string> missing;
    if (!route_complete(c, missing)) {
        err = "the census has no row for";
        for (const auto &m : missing) err += "\n  " + m;
        err += "\n— the mix cannot be expanded without it, and this tool will not guess a shape.";
        return false;
    }

    std::vector<bool> is_attn((size_t)c.n_all, false);
    for (long long L : c.attn_layers) is_attn[(size_t)L] = true;

    std::vector<std::vector<size_t>> layer_sites((size_t)c.n_all);
    for (const CensusRow &row : c.rows) {
        if (row.scope == "once" || !is_library_leaf(row.leaf)) continue;
        for (long long L = 0; L < c.n_all; ++L) {
            const bool applies = (row.scope == "all") || (row.scope == "attn" && is_attn[(size_t)L]) ||
                                 (row.scope == "gdn" && !is_attn[(size_t)L]);
            if (applies) layer_sites[(size_t)L].push_back(
                add_site(plan, L, row.leaf, (int)row.n, (int)row.k));
        }
    }

    std::vector<size_t> w4_a, w4_b, w4_ab, w4_k, w4_v;
    const CensusRow *ra = census_find(c, "linear_attn.in_proj_a.weight");
    const CensusRow *rb = census_find(c, "linear_attn.in_proj_b.weight");
    for (long long L = 0; L < c.n_all; ++L) {
        if (is_attn[(size_t)L]) continue;   // in_proj_a/b exist on the 18 linear layers only
        w4_a.push_back(add_site(plan, L, "linear_attn.in_proj_a", (int)ra->n, (int)ra->k));
        w4_b.push_back(add_site(plan, L, "linear_attn.in_proj_b", (int)rb->n, (int)rb->k));
        // The fused variant is built from the census's a-row, so a change to in_proj_a moves it too.
        w4_ab.push_back(add_site(plan, L, "in_proj_a|b FUSED", (int)(2 * ra->n), (int)ra->k));
    }
    const CensusRow *rk = census_find(c, "self_attn.k_proj.weight");
    const CensusRow *rv = census_find(c, "self_attn.v_proj.weight");
    for (long long L = 0; L < c.n_all; ++L) {
        if (!is_attn[(size_t)L]) continue;
        w4_k.push_back(add_site(plan, L, "self_attn.k_proj", (int)rk->n, (int)rk->k));
        w4_v.push_back(add_site(plan, L, "self_attn.v_proj", (int)rv->n, (int)rv->k));
    }

    const auto want = [&](const char *g) {
        for (const auto &w : want_groups) if (w == g) return true;
        return false;
    };
    long long first_gdn = -1;
    for (long long L = 0; L < c.n_all; ++L) if (!is_attn[(size_t)L]) { first_gdn = L; break; }
    if (c.attn_layers.empty() || first_gdn < 0) { err = "the census lists no attn or no gdn layer"; return false; }
    if (want("attn"))
        plan.mix_groups.push_back({"attn-layer", layer_sites[(size_t)c.attn_layers[0]]});
    if (want("gdn"))
        plan.mix_groups.push_back({"gdn-layer", layer_sites[(size_t)first_gdn]});
    if (want("all")) {
        std::vector<size_t> all;
        for (long long L = 0; L < c.n_all; ++L)
            for (size_t idx : layer_sites[(size_t)L]) all.push_back(idx);
        plan.mix_groups.push_back({"all-24-layers", all});
    }
    plan.w4_groups.push_back({"in_proj_a", w4_a});
    plan.w4_groups.push_back({"in_proj_b", w4_b});
    plan.w4_groups.push_back({"a|b fused N=32", w4_ab});
    plan.w4_groups.push_back({"k_proj", w4_k});
    plan.w4_groups.push_back({"v_proj", w4_v});
    if (!with_w4) plan.w4_groups.clear();

    // Allocate only what the built groups use: a --skinny run must not pay for a gigabyte of library
    // weights it will never read, and a --mix run has no use for the skinny tensors.
    std::vector<char> needed(plan.sites.size(), 0);
    for (const Group &g : plan.mix_groups) for (size_t i : g.sites) needed[i] = 1;
    for (const Group &g : plan.w4_groups) for (size_t i : g.sites) needed[i] = 1;

    const size_t esz = 2;   // bf16 and fp16 are the same width: that is the whole point of §4
    for (size_t i = 0; i < plan.sites.size(); ++i) {
        if (!needed[i]) continue;
        Site &s = plan.sites[i];
        CK(hipMalloc(&s.W, (size_t)s.n * s.k * esz));
        std::vector<uint8_t> host((size_t)s.n * s.k * esz);
        fill_16bit(host.data(), (size_t)s.n * s.k, dt.is_bf, (unsigned)(37 + 7 * (i % 23)));
        CK(hipMemcpy(s.W, host.data(), host.size(), hipMemcpyHostToDevice));
        CK(hipMalloc(&s.C, (size_t)mmax * s.n * esz));
        plan.weight_bytes += host.size();
        plan.c_bytes += (size_t)mmax * s.n * esz;
    }
    for (size_t i = 0; i < plan.sites.size(); ++i) {
        if (!needed[i]) continue;
        Site &s = plan.sites[i];
        auto it = plan.A_by_k.find(s.k);
        if (it == plan.A_by_k.end()) {
            void *A = nullptr;
            CK(hipMalloc(&A, (size_t)mmax * s.k * esz));
            std::vector<uint8_t> host((size_t)mmax * s.k * esz);
            fill_16bit(host.data(), (size_t)mmax * s.k, dt.is_bf, 17u);
            CK(hipMemcpy(A, host.data(), host.size(), hipMemcpyHostToDevice));
            it = plan.A_by_k.emplace(s.k, A).first;
        }
        s.A = it->second;
    }
    return true;
}

void free_plan(Plan &plan) {
    for (Site &s : plan.sites) {
        if (s.W) CK(hipFree(s.W));
        if (s.C) CK(hipFree(s.C));
        s.W = s.C = nullptr;
    }
    for (auto &kv : plan.A_by_k) if (kv.second) CK(hipFree(kv.second));
    plan.A_by_k.clear();
}

// The one call form every mode shares, so the correctness modes certify the call the timing modes
// make: OP_T on the weight as stored, ld=k, 16-bit C with ldc=n.
bool call_site(hipblasHandle_t h, const Site &s, int M, const DType &dt) {
    const hipblasStatus_t st = gemm_stored_weight(h, M, s.n, s.k, dt.dt, s.A, s.W, s.C, dt.dt_out);
    if (st != HIPBLAS_STATUS_SUCCESS) {
        std::printf("GEMM_REFUSED %s M=%d N=%d K=%d %s-in/%s-C status=%d\n", s.name.c_str(), M, s.n,
                    s.k, dt.name, dt.name, (int)st);
        std::printf("  A refused shape is not a wash band. The mix cannot be measured past it, so this\n"
                    "  run stops rather than printing a rate for a shorter mix than it claims.\n");
        return false;
    }
    return true;
}

// One pass of a group: every call issued back to back, one sync at the end — what a wave does.
// steady_clock only: hipEventElapsedTime returns exactly 0 for sub-millisecond work here.
double one_pass_ms(hipblasHandle_t h, const Plan &plan, const Group &g, int M, const DType &dt) {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t idx : g.sites)
        if (!call_site(h, plan.sites[idx], M, dt)) return -1.0;
    CK(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -----------------------------------------------------------------------------------------------
// --list : the census as this tool expanded it. No HIP call at all, so no lock is needed.
// -----------------------------------------------------------------------------------------------

int list_mode(const Census &c, const std::vector<int> &Ms) {
    std::printf("CENSUS %s  (schema prefill_gemm_shapes/1, read by tools/prefill_mix.cpp)\n",
                c.model.c_str());
    std::printf("  layers: %lld total = %lld self-attn + %lld linear-attn (full_attention_interval %lld)\n",
                c.n_all, c.n_attn, c.n_gdn, c.interval);
    std::printf("  attn layers, 0-based:");
    for (long long L : c.attn_layers) std::printf(" %lld", L);
    std::printf("\n  block matmuls per wave %lld   block flops per token %lld (%.4f GFLOP)\n",
                c.block_matmuls, c.block_flops, c.block_flops / 1e9);
    std::printf("  %-36s %-6s %6s %8s %6s  %s\n", "matmul", "scope", "count", "N", "K", "dtype");
    for (const CensusRow &r : c.rows)
        std::printf("  %-36s %-6s %6lld %8lld %6lld  %s%s\n", r.leaf.c_str(), r.scope.c_str(),
                    r.count, r.n, r.k, r.dtype.c_str(),
                    is_library_leaf(r.leaf) ? "  [library mix]"
                                            : (r.scope == "once" ? "  [not a wave matmul]"
                                                                  : "  [out of the mix: W4]"));

    std::vector<std::string> problems;
    if (!census_consistent(c, problems)) {
        std::printf("\nSPEC-FAIL — the census disagrees with itself. Fix the census; do not fix this tool.\n");
        for (const auto &p : problems) std::printf("  - %s\n", p.c_str());
        return 4;
    }
    std::printf("\nSPEC ok: the rows, the layer counts and the carried totals agree.\n");

    long long lib_calls = 0, out_calls = 0;
    for (const CensusRow &r : c.rows) {
        if (r.scope == "once") continue;
        if (is_library_leaf(r.leaf)) lib_calls += r.count;
        else out_calls += r.count;
    }
    std::printf("  the mix is %lld of the wave's %lld block matmuls; the %lld left out are in_proj_a and\n"
                "  in_proj_b, which §6 item 2 keeps in the hand kernel and --skinny measures alone.\n",
                lib_calls, c.block_matmuls, out_calls);
    for (int M : Ms) {
        double gflop_mix = 0, gflop_wave = 0;
        size_t wbytes = 0;
        for (const CensusRow &r : c.rows) {
            if (r.scope == "once") continue;
            const double f = 2.0 * M * r.n * r.k * r.count;
            gflop_wave += f;
            if (is_library_leaf(r.leaf)) {
                gflop_mix += f;
                wbytes += (size_t)r.n * r.k * 2 * r.count;
            }
        }
        std::printf("  M=%-5d mix %9.3f GFLOP (%.4f GFLOP/token)  mix weights %7.1f MiB   whole wave %9.3f GFLOP\n",
                    M, gflop_mix / 1e9, gflop_mix / M / 1e9, wbytes / 1048576.0, gflop_wave / 1e9);
    }
    const CensusRow *ra = census_find(c, "linear_attn.in_proj_a.weight");
    std::printf("  groups: attn-layer, gdn-layer, all-24-layers for --mix; in_proj_a, in_proj_b,\n"
                "  a|b fused (N=%lld), k_proj, v_proj for --skinny.\n", 2 * ra->n);
    return 0;
}

// -----------------------------------------------------------------------------------------------
// --check : correctness only. One small shape per family, C written 16-bit, compared with a CPU dot
// of the same 16-bit inputs rounded the same way. There is no timing loop in this function, which is
// what makes it safe for a worker to run (AGENTS.md).
// -----------------------------------------------------------------------------------------------

bool check_one(hipblasHandle_t h, const char *tag, int M, int n, int k, const DType &dt,
               unsigned salt, bool &any_bad) {
    const size_t na = (size_t)M * k, nw = (size_t)n * k, nc = (size_t)M * n;
    std::vector<uint8_t> ha(na * 2), hw(nw * 2);
    fill_16bit(ha.data(), na, dt.is_bf, salt);
    fill_16bit(hw.data(), nw, dt.is_bf, salt + 5u);
    // The reference is the dot of the values the device was actually given, in double, so the only
    // thing left to compare after the call is the output rounding. Decoded once into float because
    // the dot is the expensive part, not the decode.
    std::vector<float> fa(na), fw(nw);
    for (size_t i = 0; i < na; ++i) fa[i] = dec16(((const uint16_t *)ha.data())[i], dt.is_bf);
    for (size_t i = 0; i < nw; ++i) fw[i] = dec16(((const uint16_t *)hw.data())[i], dt.is_bf);
    std::vector<double> ref(nc);
    for (int m = 0; m < M; ++m)
        for (int nn = 0; nn < n; ++nn) {
            const float *pa = fa.data() + (size_t)m * k;
            const float *pw = fw.data() + (size_t)nn * k;
            double s = 0.0;
            for (int t = 0; t < k; ++t) s += (double)pa[t] * (double)pw[t];
            ref[(size_t)m * n + nn] = s;
        }
    void *dA, *dW, *dC;
    CK(hipMalloc(&dA, ha.size())); CK(hipMalloc(&dW, hw.size())); CK(hipMalloc(&dC, nc * 2));
    CK(hipMemcpy(dA, ha.data(), ha.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dW, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemset(dC, 0xFF, nc * 2));       // beta=0 must erase all of it; no 0xFFFF may survive
    Site s;
    s.name = tag;
    s.leaf = tag;
    s.n = n;
    s.k = k;
    s.A = dA;
    s.W = dW;
    s.C = dC;
    const bool accepted = call_site(h, s, M, dt);
    if (!accepted) {
        CK(hipFree(dA)); CK(hipFree(dW)); CK(hipFree(dC));
        any_bad = true;
        return false;
    }
    CK(hipDeviceSynchronize());
    std::vector<uint8_t> raw(nc * 2);
    CK(hipMemcpy(raw.data(), dC, raw.size(), hipMemcpyDeviceToHost));
    size_t survivors = 0;
    for (size_t i = 0; i < nc; ++i) survivors += (((const uint16_t *)raw.data())[i] == 0xFFFFu);
    const auto cmp = compare_16bit_output(raw.data(), ref.data(), nc, dt.is_bf);
    const bool good = cmp.worst_ulp <= 1.0 && survivors == 0;
    any_bad = any_bad || !good;
    std::printf("  %-5s %-24s M=%-3d N=%-6d K=%-5d bits differ %-5zu / %-8zu worst %7.3f ULP  %s%s\n",
                dt.name, tag, M, n, k, cmp.ndiff, nc, cmp.worst_ulp,
                good ? "ok" : "NOT AT THE ROUNDING POINT",
                survivors ? "  0xFFFF SURVIVED beta=0" : "");
    CK(hipFree(dA)); CK(hipFree(dW)); CK(hipFree(dC));
    return good;
}

bool check_mode(hipblasHandle_t h, const Census &c, const std::vector<DType> &dtypes, int M) {
    std::printf("MIXCHECK — the mix's own call form (OP_T on W as [n,k] ld=k, 16-bit C with ldc=n) at\n"
                "the census's shapes, against a CPU dot of the same 16-bit inputs rounded the same way.\n"
                "The question is the ROUNDING POINT, not whether the product is exact: a library dot\n"
                "reassociates, and §10.1 sizes what that legally costs. No timing loop exists in this\n"
                "mode, so no ms figure can be printed by it and none is.\n");
    std::printf("  %-5s %-24s %-7s %-9s %-7s %-24s %s\n", "dt", "family", "M", "N", "K",
                "bits differ", "worst distance");
    bool any_bad = false;
    for (const DType &dt : dtypes) {
        for (const char *leaf : kLibraryLeaves) {
            const CensusRow *r = census_find(c, leaf);
            check_one(h, leaf, M, (int)r->n, (int)r->k, dt, 11u + (unsigned)leaf[0] * 3u, any_bad);
        }
        for (const char *leaf : kSkinnyLeaves) {
            const CensusRow *r = census_find(c, leaf);
            check_one(h, leaf, M, (int)r->n, (int)r->k, dt, 13u + (unsigned)leaf[0] * 5u, any_bad);
        }
        const CensusRow *ra = census_find(c, "linear_attn.in_proj_a.weight");
        check_one(h, "in_proj_a|b FUSED", M, (int)(2 * ra->n), (int)ra->k, dt, 29u, any_bad);
    }
    std::printf("MIXCHECK %s\n", any_bad ? "FAIL" : "PASS");
    if (any_bad)
        std::printf("  A 16-bit C came back further than one ULP from the host rounding, or left filler\n"
                    "  behind. Route (c)'s \"no numerics change\" premise does not hold through this\n"
                    "  harness, and that belongs in a correctness run, not in a slow timing row.\n");
    else
        std::printf("  Every row is within one ULP of the host rounding — §4.3's bf16-C claim restated\n"
                    "  at the mix's own shapes and call form. No timing taken by this mode.\n");
    return !any_bad;
}

// -----------------------------------------------------------------------------------------------
// --mix-determinism : three identical passes of the mix, output bytes hashed. §3.1 answered this per
// shape; here the handle has already issued 150 calls before it picks the kernel for the 151st, and
// the call order is the mix's own. Correctness only, no timing loop.
// -----------------------------------------------------------------------------------------------

bool mix_determinism(hipblasHandle_t h, void *ws, size_t wsz, const Plan &plan, const Group &g,
                     int M, const DType &dt, int reps, uint64_t &combined) {
    std::vector<uint8_t> tmp;
    size_t cbytes = 0;
    for (size_t idx : g.sites) cbytes += (size_t)M * plan.sites[idx].n * 2;
    std::printf("  %-16s M=%-5d calls=%-4zu cbytes=%-10zu", g.name.c_str(), M, g.sites.size(), cbytes);

    // Pre-fill every C with a pattern no product of these benign inputs can store, once, before pass
    // 1. That is the beta==0 check — nothing stale may masquerade as a hit — and it costs one memset.
    for (size_t idx : g.sites)
        CK(hipMemset(plan.sites[idx].C, 0xFF, (size_t)M * plan.sites[idx].n * 2));

    auto hash_all = [&](uint64_t &hash, size_t *survivors) {
        uint64_t hv = 1469598103934665603ull;
        size_t ff = 0;
        for (size_t idx : g.sites) {
            const Site &s = plan.sites[idx];
            tmp.resize((size_t)M * s.n * 2);
            CK(hipMemcpy(tmp.data(), s.C, tmp.size(), hipMemcpyDeviceToHost));
            hv = fnv1a_from(hv, tmp.data(), tmp.size());
            for (size_t i = 0; i + 1 < tmp.size(); i += 2)
                ff += (((tmp[i] | (tmp[i + 1] << 8)) & 0xFFFFu) == 0xFFFFu);
        }
        hash = hv;
        if (survivors) *survivors = ff;
    };
    auto run_pass = [&](hipblasHandle_t hh) {
        for (size_t idx : g.sites)
            if (!call_site(hh, plan.sites[idx], M, dt)) return false;
        CK(hipDeviceSynchronize());
        return true;
    };

    bool ok = true;
    bool same_reps = true, same_fresh = true;
    uint64_t first = 0, fresh = 0;
    size_t survivors = 0;
    for (int i = 0; i < reps; ++i) {
        ok = ok && run_pass(h);
        uint64_t hv = 0;
        hash_all(hv, i == 0 ? &survivors : nullptr);
        if (i == 0) first = hv;
        else if (hv != first) same_reps = false;
    }
    // A fresh handle given the same pinned workspace: does kernel choice depend on the call history
    // of the handle that asks? §3.1 asked that per shape; this asks it of a handle that ran the mix.
    hipblasHandle_t h2 = nullptr;
    BK(hipblasCreate(&h2));
    if (ws) BK(hipblasSetWorkspace(h2, ws, wsz));
    const bool okf = run_pass(h2);
    hipblasDestroy(h2);
    hash_all(fresh, nullptr);
    if (!okf || fresh != first) same_fresh = false;

    const bool clean = ok && same_reps && same_fresh && survivors == 0;
    std::printf(" pass1=%016llx pass2..%d %s freshHandle %s beta0 %s  %s\n", (unsigned long long)first,
                reps, same_reps ? "same" : "DIFF", same_fresh ? "same" : "DIFF",
                survivors == 0 ? "clean" : "0xFF SURVIVED", clean ? "ok" : "NOT DETERMINISTIC");
    combined = fnv1a_from(combined, (const uint8_t *)&first, 8);
    combined = fnv1a_from(combined, (const uint8_t *)&fresh, 8);
    return clean;
}

// -----------------------------------------------------------------------------------------------
// --mix : W1 (bf16) / W2 (fp16). Aggregate flops ÷ wall over a whole pass of the mix.
// -----------------------------------------------------------------------------------------------

void mix_mode(hipblasHandle_t h, const Plan &plan, const std::vector<int> &Ms, const DType &dt,
              int iters, int warm) {
    std::printf("MIX %s in / f32 compute / %s C — OP_T weights-as-stored, one handle, workspace pinned.\n",
                dt.name, dt.name);
    std::printf("  flops ÷ wall for a WHOLE PASS of the group's calls, one sync at the end of the pass,\n"
                "  which is what a wave does. Rates are aggregate over the mix, not per kernel. No\n"
                "  threshold is applied here: §7 owns W1/W2's bands, and a row here is not a ledger row.\n");
    std::printf("  %-5s %-16s %6s %10s %9s %9s %9s %8s %10s %11s %13s\n", "M", "group", "calls",
                "gflop", "best_ms", "med_ms", "worst_ms", "spread", "TFLOP/s", "TFLOP/s med",
                "TFLOP/s worst");
    for (int M : Ms) {
        for (const Group &g : plan.mix_groups) {
            double gflop = 0.0;
            for (size_t idx : g.sites) gflop += 2.0 * M * plan.sites[idx].n * plan.sites[idx].k;
            // Warm: fresh pages run ~35 GB/s on this box, and the first pass also pays hipBLAS's own
            // solution search for shapes this handle has not seen. Neither belongs in a timed pass.
            for (int w = 0; w < warm; ++w)
                if (one_pass_ms(h, plan, g, M, dt) < 0.0) std::exit(3);
            std::vector<double> samples;
            for (int i = 0; i < iters; ++i) {
                const double ms = one_pass_ms(h, plan, g, M, dt);
                if (ms < 0.0) std::exit(3);
                samples.push_back(ms);
            }
            const Stat st = stat_of(samples);
            std::printf("  %-5d %-16s %6zu %10.3f %9.3f %9.3f %9.3f %7.1f%% %10.3f %11.3f %13.3f\n",
                        M, g.name.c_str(), g.sites.size(), gflop / 1e9, st.lo, st.med, st.hi,
                        st.lo > 0 ? 100.0 * (st.hi / st.lo - 1.0) : 0.0,
                        st.lo > 0 ? gflop / (st.lo * 1e-3) / 1e12 : 0.0,
                        st.med > 0 ? gflop / (st.med * 1e-3) / 1e12 : 0.0,
                        st.hi > 0 ? gflop / (st.hi * 1e-3) / 1e12 : 0.0);
            std::fflush(stdout);
        }
    }
}

// -----------------------------------------------------------------------------------------------
// --skinny : W4. The per-call cost of the shapes a square says nothing about.
//
// Two numbers per shape, because they answer different questions:
//   each-sync — every call followed by its own device sync, min/median/max over iters × calls. This
//               carries a sync per call, so it is "the engine waits for this projection", not a
//               dispatch cost.
//   burst     — the wave's real distinct calls issued back to back with ONE sync, divided by the number
//               of calls. THIS is the column comparable with the 1.85 µs launch floor of
//               devices/strix-halo-memory-bandwidth.md, which was itself a burst of launches with one
//               sync, and it is the column W4's bands are stated against.
// Both sweep every layer's own weight buffer, so the caches see the footprint a wave sees (18 distinct
// 32 KiB tensors for in_proj_a, not one buffer repeated).
// -----------------------------------------------------------------------------------------------

void skinny_mode(hipblasHandle_t h, const Plan &plan, const std::vector<int> &Ms, const DType &dt,
                 int iters, int burst) {
    std::printf("SKINNY %s in / f32 compute / %s C — OP_T weights-as-stored, best-of-%d, steady_clock.\n"
                "No threshold applied here: W4's bands are in §7 and read against the burst column,\n"
                "the one measured the way the 1.85 µs dispatch floor was.\n", dt.name, dt.name, iters);
    std::printf("  %-5s %-20s %5s %6s %7s %23s %23s\n", "M", "shape", "N", "K", "calls",
                "each-sync us min/med/max", "burst us/call min/med/max");
    for (int M : Ms) {
        for (const Group &g : plan.w4_groups) {
            const Site &s0 = plan.sites[g.sites.front()];
            int reps = burst / (int)g.sites.size();
            if (reps < 1) reps = 1;
            std::vector<double> each, burst_per_call;
            for (int it = 0; it < iters; ++it) {
                for (size_t idx : g.sites) {
                    const auto t0 = std::chrono::steady_clock::now();
                    if (!call_site(h, plan.sites[idx], M, dt)) std::exit(3);
                    CK(hipDeviceSynchronize());
                    const auto t1 = std::chrono::steady_clock::now();
                    each.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r)
                    for (size_t idx : g.sites)
                        if (!call_site(h, plan.sites[idx], M, dt)) std::exit(3);
                CK(hipDeviceSynchronize());
                const auto t1 = std::chrono::steady_clock::now();
                burst_per_call.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                                         (reps * (double)g.sites.size()));
            }
            const Stat e = stat_of(each), b = stat_of(burst_per_call);
            std::printf("  %-5d %-20s %5d %6d %7zu %7.3f %7.3f %7.3f %9.3f %7.3f %7.3f\n", M,
                        s0.leaf.c_str(), s0.n, s0.k, g.sites.size(), e.lo, e.med, e.hi, b.lo, b.med,
                        b.hi);
            std::fflush(stdout);
        }
    }
}

// --smoke : build each plan exactly as the timing modes would -- every allocation, every fill, the
// whole group expansion, at the same Mmax -- then print what came out and free it. No GEMMs and no
// timing loop, so a worker can prove the timing modes' setup works before anyone spends an exclusive
// window finding out. It is also where to look when a row asks for more memory than expected.
void smoke_mode(const Census &census, int mmax, const DType &dt,
                const std::vector<std::string> &want_groups) {
    for (int which = 0; which < 2; ++which) {
        const bool mix = (which == 0);
        Plan plan;
        std::string perr;
        if (!build_plan(census, mmax, dt, mix ? want_groups : std::vector<std::string>(), !mix, plan,
                        perr)) {
            std::printf("PLAN-FAIL %s\n", perr.c_str());
            std::exit(4);
        }
        std::printf("  %-5s %-9s sites=%-4zu weights %7.1f MiB  C %8.1f MiB  A %zu buffers\n", dt.name,
                    mix ? "mix-plan" : "w4-plan", plan.sites.size(), plan.weight_bytes / 1048576.0,
                    plan.c_bytes / 1048576.0, plan.A_by_k.size());
        const std::vector<Group> &groups = mix ? plan.mix_groups : plan.w4_groups;
        for (const Group &g : groups) {
            double gflop = 0.0;
            std::vector<char> seen((size_t)census.n_all, 0);
            for (size_t i : g.sites) {
                gflop += 2.0 * mmax * plan.sites[i].n * plan.sites[i].k;
                seen[(size_t)plan.sites[i].layer] = 1;
            }
            size_t layers = 0;
            for (char v : seen) layers += (v != 0);
            std::printf("      %-18s calls=%-4zu layers=%-3zu gflop@Mmax=%10.3f\n", g.name.c_str(),
                        g.sites.size(), layers, gflop / 1e9);
        }
        free_plan(plan);
    }
}

std::vector<std::string> split_list(const std::string &s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = s.find(',', p);
        out.push_back(s.substr(p, (q == std::string::npos ? s.size() : q) - p));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

void usage() {
    std::printf(
        "usage: tools/build/prefill_mix [--spec PATH] MODE [options]\n"
        "  MODE   --list | --check | --mix-determinism | --smoke | --mix | --skinny\n"
        "  --spec PATH      census JSON (default tools/build/prefill_gemm_shapes.json), from\n"
        "                   .venv/bin/python tools/prefill_gemm_shapes.py --json\n"
        "  --dtypes LIST    bf16, fp16, or both (default bf16 — §6 item 3 lands the route in bf16)\n"
        "  --M LIST         wave rows (default 64,512,1024,3072)\n"
        "  --iters N        best-of-N (default 5)\n"
        "  --warm N         untimed warm passes before each timed set (default 1)\n"
        "  --groups LIST    attn, gdn, all (default all three)\n"
        "  --check-m N      M for the --check rows (default 16)\n"
        "  --det-m N        M for --mix-determinism (default 64)\n"
        "  --burst N        calls per burst in --skinny (default 128)\n"
        "  --workspace-mib N hipBLAS workspace to pin (default 128)\n");
}

}  // namespace

int main(int argc, char **argv) {
    std::string spec = "tools/build/prefill_gemm_shapes.json";
    std::string mode;
    std::vector<std::string> want_groups = {"attn", "gdn", "all"};
    std::vector<int> Ms = {64, 512, 1024, 3072};
    std::vector<std::string> want_dtypes = {"bf16"};
    int iters = 5, warm = 1, check_m = 16, det_m = 64, burst = 128;
    size_t wsz = (size_t)128 << 20;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *flag) {
            if (i + 1 >= argc) { std::printf("%s needs a value\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (a == "--list" || a == "--check" || a == "--mix-determinism" || a == "--smoke" ||
            a == "--mix" || a == "--skinny") mode = a;
        else if (a == "--spec") spec = next("--spec");
        else if (a == "--dtypes") want_dtypes = split_list(next("--dtypes"));
        else if (a == "--M") {
            Ms.clear();
            for (const auto &t : split_list(next("--M"))) Ms.push_back(atoi(t.c_str()));
        } else if (a == "--groups") want_groups = split_list(next("--groups"));
        else if (a == "--iters") iters = atoi(next("--iters"));
        else if (a == "--warm") warm = atoi(next("--warm"));
        else if (a == "--check-m") check_m = atoi(next("--check-m"));
        else if (a == "--det-m") det_m = atoi(next("--det-m"));
        else if (a == "--burst") burst = atoi(next("--burst"));
        else if (a == "--workspace-mib") wsz = (size_t)atoi(next("--workspace-mib")) * 1048576u;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::printf("unknown argument: %s\n", a.c_str()); usage(); return 2; }
    }
    if (mode.empty()) { usage(); return 2; }

    Census census;
    std::string err;
    if (!load_census(spec, census, err)) {
        std::printf("SPEC-FAIL %s\n", err.c_str());
        return 4;
    }
    if (mode == "--list") return list_mode(census, Ms);
    std::vector<std::string> missing;
    if (!route_complete(census, missing)) {
        std::printf("SPEC-FAIL the route asks for shapes the census does not have:");
        for (const auto &m : missing) std::printf("\n  %s", m.c_str());
        std::printf("\n  Fix the census (tools/prefill_gemm_shapes.py), not this tool.\n");
        return 4;
    }

    std::vector<DType> dtypes;
    for (const auto &d : want_dtypes) {
        if (d == "bf16") dtypes.push_back({HIP_R_16BF, HIP_R_16BF, "bf16", true});
        else if (d == "fp16") dtypes.push_back({HIP_R_16F, HIP_R_16F, "fp16", false});
        else { std::printf("unknown dtype '%s' — bf16 or fp16\n", d.c_str()); return 2; }
    }

    CK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s  CUs=%d\n", prop.name, prop.multiProcessorCount);
    std::printf("census %s: %s, %lld layers (%lld attn / %lld gdn), %lld block matmuls per wave\n",
                spec.c_str(), census.model.c_str(), census.n_all, census.n_attn, census.n_gdn,
                census.block_matmuls);

    hipblasHandle_t h = nullptr;
    BK(hipblasCreate(&h));
    void *ws = nullptr;
    CK(hipMalloc(&ws, wsz));          // one workspace buffer, pinned for the whole process
    BK(hipblasSetWorkspace(h, ws, wsz));
    std::printf("hipBLAS workspace pinned: %zu MiB (the fresh handle in --mix-determinism gets the "
                "same buffer)\n", wsz >> 20);

    int rc = 0;
    if (mode == "--check") {
        for (const DType &dt : dtypes)
            if (!check_mode(h, census, {dt}, check_m)) rc = 1;
    } else if (mode == "--mix-determinism") {
        std::printf("MIXDET — three identical passes of the mix, every C hashed in call order (FNV-1a).\n"
                    "§3.1 proved determinism per shape; this asks the question of the mix's call order\n"
                    "and of a handle that has already issued the whole mix. No timing taken.\n");
        for (const DType &dt : dtypes) {
            Plan plan;
            std::string perr;
            if (!build_plan(census, det_m, dt, {"all"}, false, plan, perr)) {
                std::printf("PLAN-FAIL %s\n", perr.c_str());
                rc = 4;
                break;
            }
            uint64_t combined = 1469598103934665603ull;
            for (const Group &g : plan.mix_groups)
                if (!mix_determinism(h, ws, wsz, plan, g, det_m, dt, 3, combined)) rc = 1;
            std::printf("MIXDET-COMBINED %s %016llx  (identical across three separate PROCESSES says the\n"
                        "  selection does not depend on address layout or process history either)\n",
                        dt.name, (unsigned long long)combined);
            free_plan(plan);
        }
    } else if (mode == "--smoke") {
        const int mx = *std::max_element(Ms.begin(), Ms.end());
        std::printf("SMOKE — plan construction at Mmax=%d: every allocation and fill, no GEMMs and no\n"
                    "timing taken. This is what a --mix or --skinny run sets up before it measures.\n", mx);
        for (const DType &dt : dtypes) smoke_mode(census, mx, dt, want_groups);
    } else {
        const int mmax = *std::max_element(Ms.begin(), Ms.end());
        for (const DType &dt : dtypes) {
            Plan mix_plan, w4_plan;
            std::string perr;
            // --skinny builds only the skinny sites, --mix only the library ones: allocating both for
            // every mode would cost a gigabyte of weights a --skinny row never reads.
            if (mode == "--mix" && !build_plan(census, mmax, dt, want_groups, false, mix_plan, perr)) {
                std::printf("PLAN-FAIL %s\n", perr.c_str());
                rc = 4;
                break;
            }
            if (mode == "--skinny" && !build_plan(census, mmax, dt, {}, true, w4_plan, perr)) {
                std::printf("PLAN-FAIL %s\n", perr.c_str());
                rc = 4;
                break;
            }
            const Plan &shown = mode == "--mix" ? mix_plan : w4_plan;
            std::printf("allocated: weights %.1f MiB, C buffers %.1f MiB, distinct A buffers %zu, Mmax=%d\n",
                        shown.weight_bytes / 1048576.0, shown.c_bytes / 1048576.0,
                        shown.A_by_k.size(), mmax);
            if (mode == "--mix") mix_mode(h, mix_plan, Ms, dt, iters, warm);
            else skinny_mode(h, w4_plan, Ms, dt, iters, burst);
            free_plan(mix_plan);
            free_plan(w4_plan);
        }
    }
    if (ws) CK(hipFree(ws));
    BK(hipblasDestroy(h));
    return rc;
}
