// engine/loader_test.cpp — the gate for weight residency (progress/2026-10-15/build-queue.md W5, first half).
//
// Every claim it prints it also asserts; any failure exits non-zero. The checks:
//   * 335 text tensors loaded / 153 vision skipped — the G0 split (§10), re-derived from
//     the checkpoint here rather than trusted from a prompt, and compared against the
//     reference table `tools/weights_report.py` produces (via engine/weights_reference.py);
//   * per tensor: shape + dtype identical to that reference;
//   * per tensor: SHA-256 of the bytes on the device == SHA-256 of the bytes in the file
//     (this is what proves name map + offset + dtype + tying are right *simultaneously*);
//   * the name map is a bijection in both directions, with nothing unclaimed and nothing
//     double-mapped;
//   * `lm_head` is one allocation under two names, not two 485 MiB copies;
//   * honest memory accounting (`hipMemGetInfo`) and a timing for every phase.
//
// build: engine/build.sh   (hipcc -O3 -Wl,-rpath,/opt/rocm/lib)
// run:   ./engine/loader_test            # from the repository root
//        ./engine/loader_test --no-verify

#include "loader.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using engine::Dtype;
using engine::LoadStats;
using engine::normalize_name;
using engine::sha256_hex;
using engine::TensorPlan;
using engine::Weight;
using engine::WeightStore;

int g_checks = 0;
int g_failures = 0;

void ok(bool cond, const std::string& label) {
    ++g_checks;
    if (!cond) ++g_failures;
    std::printf("  [%s] %s\n", cond ? " OK " : "FAIL", label.c_str());
}

void note(const std::string& line) { std::printf("        %s\n", line.c_str()); }

void section(const std::string& title) { std::printf("\n== %s\n", title.c_str()); }

std::string fmt_gib(double bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}
std::string fmt_gb(double bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4f GB", bytes / 1e9);
    return buf;
}
std::string fmt_mib(double bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", bytes / (1024.0 * 1024.0));
    return buf;
}
std::string fmt_rate(double bytes, double ms) {
    char buf[64];
    const double gbs = ms > 0 ? (bytes / 1e9) / (ms / 1000.0) : 0.0;
    std::snprintf(buf, sizeof(buf), "%.1f GB/s", gbs);
    return buf;
}

// Runs a command and returns its stdout lines; *rc is the exit status.
std::vector<std::string> run(const std::string& cmd, int* rc) {
    std::vector<std::string> lines;
    FILE* p = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (!p) {
        *rc = -1;
        return lines;
    }
    char buf[4096];
    while (fgets(buf, sizeof(buf), p)) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        lines.push_back(s);
    }
    *rc = pclose(p);
    return lines;
}

struct RefTensor {
    std::string ckpt_name;
    std::string dtype;
    std::string shape;  // "6144x1024"
    long long nbytes = 0;
    std::string cat;    // text | vision | other
};

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += sep;
        s += v[i];
    }
    return s;
}

// Splits "a\tb\tc" into fields; tab-separated so no field can contain a tab.
std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t t = s.find('\t', start);
        out.push_back(s.substr(start, t == std::string::npos ? std::string::npos : t - start));
        if (t == std::string::npos) return out;
        start = t + 1;
    }
}

void copy_back(const Weight& w, std::vector<char>& buf) {
    buf.resize(w.nbytes);
    hipError_t rc = hipMemcpy(buf.data(), w.device, w.nbytes, hipMemcpyDeviceToHost);
    ++g_checks;
    if (rc != hipSuccess) {
        ++g_failures;
        std::printf("  [FAIL] hipMemcpy D2H of %s: %s\n", w.name.c_str(), hipGetErrorName(rc));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models/Qwen3.5-0.8B";
    bool verify = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-verify") verify = false;
        else if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else {
            std::printf("usage: %s [--model-dir DIR] [--no-verify]\n", argv[0]);
            return 2;
        }
    }

    std::printf("engine/loader_test — weight residency gate (W5 first half)\n");
    std::printf("model dir: %s\n", model_dir.c_str());
    std::printf("device byte verification: %s\n", verify ? "ON" : "OFF (--no-verify)");

    // -----------------------------------------------------------------------
    section("0. SHA-256 selftest (FIPS 180-1 vectors)");
    {
        const std::string abc = sha256_hex("abc", 3);
        ok(abc == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "sha256(\"abc\") matches the published vector");
        const std::string empty = sha256_hex("", 0);
        ok(empty == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "sha256(\"\") matches the published vector");
        const std::string long_in(1000000, 'a');
        ok(sha256_hex(long_in.data(), long_in.size()) ==
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
           "sha256(1e6 x 'a') matches the published vector");
    }

    // -----------------------------------------------------------------------
    section("1. reference table from tools/weights_report.py");
    std::vector<RefTensor> refs;
    std::set<std::string> ref_text, ref_vision;
    long long ref_text_bytes = 0;
    int ref_total = 0;
    {
        int rc = -1;
        const auto lines = run("python3 engine/weights_reference.py " + model_dir, &rc);
        ok(rc == 0 && !lines.empty(), "engine/weights_reference.py ran (exit " + std::to_string(rc) + ")");
        int n = 0;
        for (const std::string& line : lines) {
            if (line.compare(0, 2, "T\t") == 0) {
                std::vector<std::string> f = split_tab(line.substr(2));
                if (f.size() != 5) { ok(false, "malformed T line: " + line); continue; }
                RefTensor r;
                r.ckpt_name = f[0];
                r.dtype = f[1];
                r.shape = f[2];
                r.nbytes = std::strtoll(f[3].c_str(), nullptr, 10);
                r.cat = f[4];
                refs.push_back(r);
                if (r.cat == "text") {
                    ref_text.insert(r.ckpt_name);
                    ref_text_bytes += r.nbytes;
                } else if (r.cat == "vision") {
                    ref_vision.insert(r.ckpt_name);
                }
                ++n;
            } else if (line.compare(0, 2, "K\t") == 0) {
                std::vector<std::string> f = split_tab(line.substr(2));
                if (f.size() == 2 && f[0] == "total_tensors")
                    ref_total = (int)std::strtol(f[1].c_str(), nullptr, 10);
            }
        }
        ok(n == ref_total && n > 0,
           "reference table lists every tensor in the checkpoint (" + std::to_string(n) + " of " +
               std::to_string(ref_total) + ")");
        note("reference split: " + std::to_string(ref_text.size()) + " text / " +
             std::to_string(ref_vision.size()) + " vision / " + std::to_string(refs.size()) +
             " total, " + fmt_gb((double)ref_text_bytes) + " of it text");
        if (refs.empty()) {
            std::printf("\nRESULT: FAIL — no reference table, cannot gate anything\n");
            return 1;
        }
    }

    // -----------------------------------------------------------------------
    section("2. load (§02.5 names, hipMalloc + hipMemcpy, warm pass)");
    LoadStats st;
    std::unique_ptr<WeightStore> store;
    try {
        WeightStore::Options opts;
        opts.model_dir = model_dir;
        opts.verify_device_bytes = verify;
        store.reset(new WeightStore(WeightStore::load(opts, st)));
    } catch (const std::exception& e) {
        std::printf("  [FAIL] loader threw: %s\n", e.what());
        return 1;
    }
    ok(true, "load() returned without throwing");

    // -----------------------------------------------------------------------
    section("3. the G0 split: every tensor claimed, 335 loaded / 153 skipped");
    ok(st.n_loaded == 335, "loaded text tensors = 335 (loader counted " + std::to_string(st.n_loaded) + ")");
    ok(st.n_skipped_vision == 153, "skipped vision tensors = 153 (loader counted " +
                                      std::to_string(st.n_skipped_vision) + ")");
    ok((long long)st.n_loaded == (long long)ref_text.size(),
       "loader's load set == reference text set (" + std::to_string(ref_text.size()) + ")");
    ok((long long)st.n_skipped_vision == (long long)ref_vision.size(),
       "loader's skip set == reference vision set (" + std::to_string(ref_vision.size()) + ")");
    ok(st.n_loaded + st.n_skipped_vision == (size_t)ref_total,
       "loaded + skipped == tensors in the checkpoint (" + std::to_string(ref_total) +
           "); nothing was silently ignored");
    note("the split is what §10 gate G0 pins (335/153), not the 344/144 miscount");

    // -----------------------------------------------------------------------
    section("4. byte accounting against §01.7");
    ok(store->resident_bytes() == (size_t)ref_text_bytes,
       "resident bytes == reference text bytes (" + std::to_string(store->resident_bytes()) + ")");
    ok(store->allocation_count() == st.n_loaded,
       "one allocation per loaded tensor: " + std::to_string(store->allocation_count()) +
           " hipMallocs for " + std::to_string(st.n_loaded) + " loaded tensors");
    std::printf("        resident %s = %s = %.4f GiB\n", fmt_gib((double)store->resident_bytes()).c_str(),
                fmt_gb((double)store->resident_bytes()).c_str(), store->resident_bytes() / (1024.0 * 1024.0 * 1024.0));
    {
        const double spec_017_gb = 1.5457;  // specification/01 §01.7 text-only total
        const double got = store->resident_bytes() / 1e9;
        char d[64];
        std::snprintf(d, sizeof(d), "%.7f GB, delta %+.4f MB from §01.7", got, (got - spec_017_gb) * 1000.0);
        ok(std::abs(got - spec_017_gb) < 5e-5,
           std::string("resident total matches §01.7's 1.5457 GB (") + d + ")");
        int rc = -1;
        double report_gb = -1.0;
        for (const std::string& line : run("python3 tools/weights_report.py " + st.shard_paths[0], &rc)) {
            if (line.rfind("text-only totals", 0) == 0) {
                const size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) report_gb = std::strtod(line.c_str() + pos, nullptr);
            }
        }
        ok(rc == 0 && report_gb > 0, "tools/weights_report.py ran and printed its totals");
        ok(report_gb > 0 && std::abs(report_gb - got) < 1e-4,
           "tools/weights_report.py's own 'text-only totals' agrees with what is resident (" +
               std::to_string(report_gb) + " GB vs " + std::to_string(got) + " GB)");
    }

    // -----------------------------------------------------------------------
    section("5. per tensor: shape and dtype vs tools/weights_report.py");
    {
        int checked = 0, bad = 0;
        std::vector<std::string> examples;
        for (const RefTensor& r : refs) {
            if (r.cat != "text") continue;
            std::string norm;
            if (normalize_name(r.ckpt_name, norm) != TensorPlan::Load) {
                if (examples.size() < 5) examples.push_back("reference name not loadable: " + r.ckpt_name);
                ++bad;
                continue;
            }
            const Weight* w = store->find(norm);
            if (!w) {
                if (examples.size() < 5) examples.push_back("engine has no " + norm);
                ++bad;
                continue;
            }
            const bool same_dtype = std::string(engine::dtype_name(w->dtype)) == r.dtype;
            const bool same_shape = w->shape_string() == r.shape;
            const bool same_bytes = (long long)w->nbytes == r.nbytes &&
                                    w->nbytes == (size_t)w->numel() * engine::dtype_size(w->dtype);
            if (!(same_dtype && same_shape && same_bytes)) {
                ++bad;
                if (examples.size() < 5)
                    examples.push_back(norm + ": engine " + std::string(engine::dtype_name(w->dtype)) + " " +
                                       w->shape_string() + " " + std::to_string(w->nbytes) + "B vs ref " +
                                       r.dtype + " " + r.shape + " " + std::to_string(r.nbytes) + "B");
            }
            ++checked;
        }
        ok(bad == 0 && checked == (int)ref_text.size(),
           "shape + dtype + byte size match the reference for all " + std::to_string(checked) + " text tensors");
        for (const std::string& e : examples) note("mismatch: " + e);

        std::printf("        sample (engine's own view):\n");
        const char* samples[] = {"lm.embed_tokens.weight",
                                 "lm.layers.0.linear_attn.in_proj_qkv.weight",
                                 "lm.layers.0.linear_attn.A_log",
                                 "lm.layers.3.self_attn.q_proj.weight",
                                 "lm.layers.23.mlp.down_proj.weight",
                                 "lm.norm.weight",
                                 "mtp.fc.weight",
                                 "mtp.layer0.self_attn.q_norm.weight"};
        for (const char* s : samples) {
            const Weight* w = store->find(s);
            if (!w) { ok(false, std::string("sample name missing: ") + s); continue; }
            std::printf("          %-46s %-5s %-14s %s\n", s, engine::dtype_name(w->dtype),
                        w->shape_string().c_str(), fmt_mib((double)w->nbytes).c_str());
        }
        size_t n_f32 = 0, n_not_weight = 0, n_odd_size = 0;
        for (const auto& ce : store->ckpt_to_engine()) {
            const Weight& w = store->at(ce.second);
            if (w.dtype == Dtype::F32) ++n_f32;
            const std::string leaf = w.name.substr(w.name.rfind('.') + 1);
            if (leaf != "weight") ++n_not_weight;
            if (w.nbytes % 16) ++n_odd_size;
        }
        note("special cases: " + std::to_string(n_f32) + " F32 tensors (§01.6), " +
             std::to_string(n_not_weight) + " tensors whose leaf is not `.weight` "
             "(A_log, dt_bias), " +
             std::to_string(n_odd_size) + " with a size that is not a multiple of 16 B");
    }

    // -----------------------------------------------------------------------
    section("6. the name map, both directions");
    {
        const auto& fwd = store->ckpt_to_engine();
        ok(fwd.size() == st.n_loaded,
           "forward map (checkpoint -> engine) has " + std::to_string(fwd.size()) + " entries");
        std::set<std::string> norm_seen;
        int stray = 0, wrong = 0;
        for (const auto& ce : fwd) {
            if (!ref_text.count(ce.first)) ++stray;          // checkpoint tensor that is not in the ref text set
            if (!norm_seen.insert(ce.second).second) ++wrong;  // two names for one engine name
            std::string want;
            if (normalize_name(ce.first, want) != TensorPlan::Load || want != ce.second) ++wrong;
        }
        ok(stray == 0, "every checkpoint tensor the loader claims is a reference text tensor");
        ok(wrong == 0, "every engine name is produced by exactly one checkpoint name "
                       "and is what §02.5 says it should be");
        std::set<std::string> missing;
        for (const std::string& ck : ref_text) {
            std::string norm;
            if (normalize_name(ck, norm) != TensorPlan::Load || !store->find(norm)) missing.insert(ck);
        }
        ok(missing.empty(), "every reference text tensor is consumed by exactly one engine name" +
                               (missing.empty() ? "" : ("; missing: " + join(std::vector<std::string>(missing.begin(), missing.end()), ","))));
        ok(store->lookup_count() == st.n_loaded + 1,
           "engine name table = " + std::to_string(store->lookup_count()) + " names = " +
               std::to_string(st.n_loaded) + " checkpoint tensors + 1 tied alias");
        ok(store->find("model.visual.patch_embed.weight") == nullptr,
           "a skipped vision tensor is not reachable under any name");
        ok(store->find("lm.layers.99.mlp.up_proj.weight") == nullptr,
           "an invented name is not silently answered");
    }

    // -----------------------------------------------------------------------
    section("7. tied lm_head: one allocation, two names");
    {
        const Weight& emb = store->at("lm.embed_tokens.weight");
        const Weight& head = store->at("lm_head.weight");
        ok(head.is_alias() && head.storage_owner == "lm.embed_tokens.weight",
           "lm_head.weight is registered as an alias of lm.embed_tokens.weight");
        ok(head.device == emb.device, "same device pointer (" + std::to_string((uintptr_t)emb.device) + ")");
        ok(head.nbytes == emb.nbytes && head.dtype == emb.dtype && head.shape_string() == emb.shape_string(),
           "alias carries the embedding's shape/dtype (" + head.shape_string() + " " +
               engine::dtype_name(head.dtype) + ", " + fmt_mib((double)emb.nbytes) + ")");
        ok(head.checkpoint_name.empty(), "the alias names no checkpoint tensor (there is no lm_head.weight, §01.1)");
        ok(store->allocation_count() == st.n_loaded,
           "allocation count is " + std::to_string(store->allocation_count()) +
               ", so the head cost no second copy of " + fmt_mib((double)emb.nbytes));
        const size_t delta = st.free_before_bytes - st.free_after_upload_bytes;
        std::printf("        embedding/head table is %.1f MiB of the %s resident (%.1f%%)\n",
                    emb.nbytes / (1024.0 * 1024.0), fmt_gib((double)store->resident_bytes()).c_str(),
                    100.0 * emb.nbytes / (double)store->resident_bytes());
        ok(delta >= store->resident_bytes(),
           "device-visible bytes consumed since before-load (" + fmt_gib((double)delta) +
               ") cover the resident weights (" + fmt_gib((double)store->resident_bytes()) + ")");
    }

    // -----------------------------------------------------------------------
    section("8. device bytes == file bytes (SHA-256, copy-back at verification only)");
    if (!verify) {
        std::printf("        skipped by --no-verify; a normal load does not pay the D2H\n");
    } else {
        ok(st.device_verified && st.n_verified == st.n_loaded,
           "verified " + std::to_string(st.n_verified) + " of " + std::to_string(st.n_loaded) +
               " blobs (aliases hash nothing extra)");
        ok(st.verify_failures.empty(),
           "every resident tensor's SHA-256 equals the SHA-256 of its bytes in the file" +
               (st.verify_failures.empty()
                    ? std::string()
                    : (": " + join(st.verify_failures, "; " + std::to_string(st.verify_failures.size()) + " total"))));
    }

    // -----------------------------------------------------------------------
    section("9. the gate can fail: located injections");
    {
        // (a) a one-byte change must move the hash — the comparator is not degenerate.
        std::vector<char> buf;
        copy_back(store->at("lm.layers.5.mlp.gate_proj.weight"), buf);
        const std::string before = sha256_hex(buf.data(), buf.size());
        buf[12345] = (char)(buf[12345] ^ 0x01);
        const std::string after = sha256_hex(buf.data(), buf.size());
        ok(before != after, "flipping one bit of a resident tensor changes its SHA-256");
        copy_back(store->at("lm.layers.5.mlp.gate_proj.weight"), buf);

        // (b) the failure this gate exists for: a name mapped to the wrong blob. Shapes and
        // dtypes cannot see it — these two tensors are identical in both — only the bytes can.
        std::vector<char> buf2;
        copy_back(store->at("lm.layers.6.mlp.gate_proj.weight"), buf2);
        const Weight& a = store->at("lm.layers.5.mlp.gate_proj.weight");
        const Weight& b = store->at("lm.layers.6.mlp.gate_proj.weight");
        ok(a.shape_string() == b.shape_string() && a.dtype == b.dtype && a.nbytes == b.nbytes,
           "layers.5 and layers.6 gate_proj are indistinguishable by shape+dtype (" +
               a.shape_string() + " " + engine::dtype_name(a.dtype) + ")");
        const std::string ha = sha256_hex(buf.data(), buf.size()), hb = sha256_hex(buf2.data(), buf2.size());
        ok(ha != hb, "…but their bytes differ, so a swapped mapping is caught by the hash");
        ok(ha == before, "and re-reading the same device bytes hashes identically (no uninitialised slack)");

        // (c) the §02.5 rules, individually — including the refusal this loader must make.
        std::string n;
        ok(normalize_name("model.visual.blocks.0.attn.qkv.weight", n) == TensorPlan::SkipVision,
           "model.visual.* is skipped (§01.8)");
        ok(normalize_name("model.language_model.layers.7.self_attn.q_proj.weight", n) == TensorPlan::Load &&
               n == "lm.layers.7.self_attn.q_proj.weight",
           "model.language_model.* -> lm.* (" + n + ")");
        ok(normalize_name("mtp.layers.0.mlp.down_proj.weight", n) == TensorPlan::Load &&
               n == "mtp.layer0.mlp.down_proj.weight",
           "mtp.layers.0.* -> mtp.layer0.* (" + n + ")");
        ok(normalize_name("mtp.fc.weight", n) == TensorPlan::Load && n == "mtp.fc.weight",
           "mtp.* keeps its name (" + n + ")");
        ok(normalize_name("lm_head.weight", n) == TensorPlan::Unknown,
           "a checkpoint that carried lm_head.weight would be refused: the head is tied (§01.1)");
        ok(normalize_name("totally.unknown.weight", n) == TensorPlan::Unknown,
           "an unclassifiable name is Unknown, which load() turns into an error (gate G0)");
    }

    // -----------------------------------------------------------------------
    section("10. memory and timings");
    {
        size_t free_now = 0, total_now = 0;
        (void)hipMemGetInfo(&free_now, &total_now);
        const size_t stage_cost = st.free_before_bytes - st.free_after_stage_bytes;
        const size_t weight_cost = st.free_after_stage_bytes - st.free_after_upload_bytes;
        std::printf("        hipMemGetInfo total        %s\n", fmt_gib((double)total_now).c_str());
        std::printf("        free before load           %s\n", fmt_gib((double)st.free_before_bytes).c_str());
        std::printf("        free after pinned staging  %s  (-%s for the staging buffer)\n",
                    fmt_gib((double)st.free_after_stage_bytes).c_str(), fmt_gib((double)stage_cost).c_str());
        std::printf("        free after hipMalloc/H2D   %s  (-%s for the weights)\n",
                    fmt_gib((double)st.free_after_upload_bytes).c_str(), fmt_gib((double)weight_cost).c_str());
        std::printf("        free after warm pass       %s\n", fmt_gib((double)st.free_after_warm_bytes).c_str());
        std::printf("        free at end of gate        %s\n", fmt_gib((double)free_now).c_str());
        std::printf("        payload %s in %zu allocations; the pool gives up %s, the gap is "
                    "per-allocation rounding\n",
                    fmt_gib((double)store->resident_bytes()).c_str(), store->allocation_count(),
                    fmt_gib((double)weight_cost).c_str());
        ok(total_now > 0 && st.free_before_bytes > 0 && free_now > 0,
           "hipMemGetInfo reported a pool (" + fmt_gib((double)total_now) + " total, " +
               fmt_gib((double)st.free_before_bytes) +
               " free — this box exposes ~34 GiB of a ~110 GiB nominal carve-out)");
        ok(weight_cost >= store->resident_bytes(),
           "the pool charged at least the payload for it ("
               + fmt_gib((double)weight_cost) + " vs " + fmt_gib((double)store->resident_bytes()) + ")");

        const double total_ms =
            st.parse_ms + st.stage_alloc_ms + st.alloc_ms + st.stage_read_ms + st.h2d_ms + st.warm_ms;
        std::printf("        parse     %8.1f ms   %zu header bytes, %zu shard(s)\n", st.parse_ms,
                    st.header_bytes, st.shard_paths.size());
        std::printf("        staging   %8.1f ms   hipHostMalloc of the %s staging buffer\n",
                    st.stage_alloc_ms, fmt_mib((double)st.stage_bytes).c_str());
        std::printf("        alloc     %8.1f ms   %zu hipMallocs (reserve pages, no transfer)\n",
                    st.alloc_ms, store->allocation_count());
        std::printf("        stage     %8.1f ms   %s file -> pinned staging\n", st.stage_read_ms,
                    fmt_rate((double)st.uploaded_bytes, st.stage_read_ms).c_str());
        std::printf("        H2D cold  %8.1f ms   %s, destination pages fresh\n", st.h2d_ms,
                    fmt_rate((double)st.uploaded_bytes, st.h2d_ms).c_str());
        std::printf("        warm      %8.1f ms   %s first GPU read pass over %s\n", st.warm_ms,
                    fmt_rate((double)st.warm_bytes, st.warm_ms).c_str(),
                    fmt_gib((double)st.warm_bytes).c_str());
        std::printf("        H2D steady %7.1f ms   %s over %s with pages committed (devices doc: 76-82)\n",
                    st.steady_h2d_ms, fmt_rate((double)st.steady_bytes, st.steady_h2d_ms).c_str(),
                    fmt_gib((double)st.steady_bytes).c_str());
        std::printf("        cold load %7.1f ms   parse + staging + alloc + stage + H2D + warm (the steady"
                    " probe and verification are not part of it)\n",
                    total_ms);
        std::printf("        load() wall %6.1f ms   what load() actually took; the gap is the steady H2D "
                    "probe and bookkeeping\n",
                    st.total_ms);
        if (verify) {
            std::printf("        D2H       %8.1f ms   %s copy-back of %s (devices doc: 63-70)\n", st.d2h_ms,
                        fmt_rate((double)st.uploaded_bytes, st.d2h_ms).c_str(),
                        fmt_gib((double)st.uploaded_bytes).c_str());
            std::printf("        verify    %8.1f ms   D2H + SHA-256 of device and file bytes (one-time; "
                        "--no-verify skips it)\n", st.verify_ms);
        }

        ok(st.warm_checksum != 0, "warm pass actually read memory (checksum " +
                                     std::to_string(st.warm_checksum) + "; a zero would mean a no-op kernel)");
        ok(st.parse_ms > 0 && st.stage_read_ms > 0 && st.h2d_ms > 0 && st.warm_ms > 0 &&
               st.steady_h2d_ms > 0,
           "every phase reports non-zero time (a zero here is a broken timer, not a fast phase)");
        const auto rate = [](double bytes, double ms) { return ms > 0 ? bytes / 1e9 / (ms / 1000.0) : 0.0; };
        const double cold = rate(st.uploaded_bytes, st.h2d_ms);
        const double steady = rate(st.steady_bytes, st.steady_h2d_ms);
        const double warm = rate(st.warm_bytes, st.warm_ms);
        const double d2h = verify ? rate(st.uploaded_bytes, st.d2h_ms) : 1.0;
        ok(cold > 0 && warm > 0 && steady > 0 && d2h > 0 && cold < 240 && warm < 240 && steady < 240 &&
               d2h < 240,
           "no phase exceeds the measured 240 GB/s ceiling (H2D cold " + std::to_string(cold) +
               ", warm read " + std::to_string(warm) + ", H2D steady " + std::to_string(steady) +
               ", D2H " + std::to_string(d2h) +
               " GB/s) — a bigger number would mean skipped work");
        ok(steady >= cold,
           "steady H2D is at least as fast as cold H2D, which is what the first-touch penalty "
           "predicts ("
               + std::to_string(steady) + " vs " + std::to_string(cold) + " GB/s)");
        ok(st.total_ms >= total_ms,
           "the phase times do not exceed the wall clock load() took ("
               + std::to_string(total_ms) + " ms of phases vs " + std::to_string(st.total_ms) +
                   " ms wall)");
    }

    std::printf("\nRESULT: %s — %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks,
                g_failures);
    return g_failures ? 1 : 0;
}
