/*
 * prototype/gdn_gated_norm/gdn_gated_norm.hip.cpp — the §04.5 op's driver.
 *
 * Shape follows prototype/glue (the nearest neighbour: a tiny norm with an elementwise epilogue,
 * launch-cost rather than bandwidth, weights resolved through engine/loader.h rather than
 * hand-rolled H2D). The op's own CPU mirror gate is mirror_probe.py and needs no device; this
 * binary is the device half of the gate.
 *
 * Build: ./build.sh        CPU gate:  ../.. /.venv/bin/python run.py --mirror
 * Device gate: .venv/bin/python prototype/gdn_gated_norm/run.py --gate
 *
 * PLAN is a text file, one launch per line:
 *     NAME OFILE ZFILE WNAME OUTFILE [NORMFILE]
 * OFILE is raw little-endian F32, 2048 elements (the recurrence's output; the gate writes the
 * `gdn.o_pre_norm` pin, which is BF16-valued, widened). ZFILE is raw BF16, 2048 elements
 * (`in_proj_z.out`). WNAME is an ENGINE NORMALISED weight name resolved through the loader
 * (`lm.layers.N.linear_attn.norm.weight`, F32 [128] per §01.6). OUTFILE receives the 2048 BF16
 * results; NORMFILE, if given, receives BF16(n) — the pre-gate stage the negatives are localised
 * against. `-` skips a write.
 *
 * `--mode once` is the correctness path, and it reports on itself: every output buffer is prefilled
 * with a 0x0101 sentinel before the launch, and a `sentinel:` line prints for the first job, the last
 * job, and every job still holding the sentinel (with the first 16 bytes hashed before and after);
 * every resolved weight row is copied back and range-checked in a `weights:` line. Both exist because
 * the first run of this gate printed `RESULT mode=once jobs=144` and wrote files that were mostly +0
 * and NaN — which is the byte pattern untouched fresh pages give, so the log could not tell "the
 * launch did not write" from "the launch wrote zeros". The weight check is the same argument one level
 * up: `weights:` names the rows the store handed over and their magnitude, so a wrong-dtype read is
 * visible in one line.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "engine/loader.h"
#include "gdn_gated_norm_kernels.h"

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    std::vector<char> b((size_t)f.tellg());
    f.seekg(0);
    f.read(b.data(), (std::streamsize)b.size());
    if ((size_t)f.gcount() != b.size()) {
        fprintf(stderr, "%s: short read\n", p.c_str());
        exit(1);
    }
    return b;
}

static void spit(const std::string& p, const void* data, size_t nbytes) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char*)data, (std::streamsize)nbytes);
}

// A non-zero fill pattern for the output buffers, so "the launch never wrote here" is separable from
// "the launch wrote zeros". Fresh hipMalloc pages read back as +0, which is exactly what a silent
// write failure looks like in the output file, so 0x0000 is useless as a prefill. The first choice
// here was 0x3F3F (0.746) and it was wrong in a way worth keeping: 0.746 is a perfectly ordinary
// value for the `norm_pre_gate` stage, whose elements are O(1), so 252 norm buffers came back
// reporting "left 1/2048 (PARTIALLY WRITTEN)" when what had really happened is that one lane was
// written with a value equal to the prefill. A sentinel is only evidence if the kernel cannot write
// it: 0x0101 is a BF16 denormal (~1.8e-38), unreachable for either output stage from O(1) inputs,
// and still one byte value to memset.
static const uint16_t GN_SENTINEL = 0x0101;

static uint64_t fnv1a64(const void* data, size_t n) {
    const unsigned char* b = (const unsigned char*)data;
    uint64_t h = 14695981039346656037ull;   // FNV-1a 64-bit offset basis
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;           // FNV-1a 64-bit prime
    }
    return h;
}

// Elements still holding the sentinel after the launch: 0 means the buffer was written, n means it
// was not touched at all, and anything in between is a partially written (wrong-shape) launch.
// `first` receives the index of the first survivor (-1 if none): "1 of 2048 left" means different
// things when it names lane (0,0) of head 0 than when it names a scattered lane.
static size_t sentinel_left(const hip_bfloat16* host, size_t n, int* first = nullptr) {
    const uint16_t* bits = (const uint16_t*)host;
    size_t left = 0;
    for (size_t i = 0; i < n; ++i) {
        if (bits[i] == GN_SENTINEL) {
            if (first && *first < 0) *first = (int)i;
            ++left;
        }
    }
    return left;
}

struct Job {
    std::string name, ofile, zfile, wname, out, nout;
    const float* do_ = nullptr;
    const hip_bfloat16* dz = nullptr;
    const float* dw = nullptr;
    hip_bfloat16 *dout = nullptr, *dnorm = nullptr;
    uint64_t pre_hash = 0;   // fnv1a64 of the first 16 sentinel bytes, taken after the prefill
    size_t bytes() const {
        return (size_t)2048 * 4 + (size_t)2048 * 2 + 128 * 4 + (size_t)2048 * 2
               + (nout == "-" ? 0 : (size_t)2048 * 2);
    }
};

int main(int argc, char** argv) {
    std::string plan_path, negname = "none", model = "models/Qwen3.5-0.8B", mode = "once";
    int iters = 200, trials = 3;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--neg") negname = next();
        else if (a == "--model") model = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--trials") trials = atoi(next().c_str());
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }
    const int neg = negname == "none" ? GN_NONE : negname == "w_f32" ? GN_W_F32
                  : negname == "affine_uncast" ? GN_AFFINE_UNCAST
                  : negname == "single_cast" ? GN_SINGLE_CAST
                  : negname == "silu_rounded" ? GN_SILU_ROUNDED
                  : negname == "bf16acc" ? GN_BF16ACC : negname == "noeps" ? GN_NOEPS : -1;
    if (neg < 0) {
        fprintf(stderr, "--neg must be none|w_f32|affine_uncast|single_cast|silu_rounded"
                        "|bf16acc|noeps\n");
        return 1;
    }
    if (plan_path.empty() || (mode != "once" && mode != "burst")) {
        fprintf(stderr, "usage: gdn_gated_norm --mode once|burst --plan PLAN [--neg NAME]\n");
        return 1;
    }

    engine::WeightStore::Options lopts;
    lopts.model_dir = model;
    lopts.verify_device_bytes = false;
    lopts.measure_steady_h2d = false;
    engine::LoadStats lstats;
    engine::WeightStore store = engine::WeightStore::load(lopts, lstats);
    printf("loader: %zu tensors resident, %.3f GiB, warm %.1f ms\n", lstats.n_loaded,
           lstats.uploaded_bytes / 1073741824.0, lstats.warm_ms);
    if (lstats.warm_checksum == 0) {
        fprintf(stderr, "loader: warm pass read nothing (checksum 0)\n");
        return 1;
    }

    std::vector<Job> jobs;
    size_t w_rows = 0;
    float w_absmax = 0.f;
    {
        std::ifstream pf(plan_path);
        if (!pf) {
            fprintf(stderr, "cannot open plan %s\n", plan_path.c_str());
            return 1;
        }
        std::string line;
        int ln = 0;
        while (std::getline(pf, line)) {
            ++ln;
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ls(line);
            Job j;
            if (!(ls >> j.name >> j.ofile >> j.zfile >> j.wname >> j.out)) {
                fprintf(stderr, "plan %s line %d: NAME OFILE ZFILE WNAME OUTFILE [NORMFILE]\n",
                        plan_path.c_str(), ln);
                return 1;
            }
            j.nout = "-";
            ls >> j.nout;
            const std::vector<char> oraw = slurp(j.ofile), zraw = slurp(j.zfile);
            if (oraw.size() != 2048 * 4 || zraw.size() != 2048 * 2) {
                fprintf(stderr, "%s: o must be 2048 F32 and z 2048 BF16 (%zu / %zu bytes)\n",
                        j.name.c_str(), oraw.size(), zraw.size());
                return 1;
            }
            CK(hipMalloc((void**)&j.do_, oraw.size()));
            CK(hipMemcpy((void*)j.do_, oraw.data(), oraw.size(), hipMemcpyHostToDevice));
            CK(hipMalloc((void**)&j.dz, zraw.size()));
            CK(hipMemcpy((void*)j.dz, zraw.data(), zraw.size(), hipMemcpyHostToDevice));
            const engine::Weight& wt = store.at(j.wname);
            if (wt.dtype != engine::Dtype::F32 || wt.nbytes != 128 * 4 || wt.shape.size() != 1
                || wt.shape[0] != 128) {
                fprintf(stderr, "%s: %s is %s %s (%zu B); §01.6 says linear_attn.norm.weight is"
                                " F32 [128]\n", j.name.c_str(), j.wname.c_str(),
                        engine::dtype_name(wt.dtype), wt.shape_string().c_str(), wt.nbytes);
                return 1;
            }
            j.dw = (const float*)wt.device;
            // The guard above proves the store *called* this F32 and 512 bytes; this proves the
            // bytes are a §04.5 affine weight and not, say, denormal garbage reached through the
            // wrong dtype. A norm weight is O(1) and finite in every layer, so a value outside that
            // band means the pointer or the dtype is wrong, and it is worth saying here rather than
            // reading it back out of 49 000-ULP output files.
            std::vector<float> wbytes(128);
            CK(hipMemcpy(wbytes.data(), wt.device, 128 * 4, hipMemcpyDeviceToHost));
            for (const float v : wbytes) {
                w_absmax = std::max(w_absmax, std::abs(v));
                if (!(v == v) || std::abs(v) > 1e4f) {
                    fprintf(stderr, "%s: %s resolves to a non-finite or O(1e4) value (%g); a"
                                    " linear_attn.norm.weight row is O(1), so these are not the"
                                    " bytes this op wants\n", j.name.c_str(), j.wname.c_str(), v);
                    return 1;
                }
            }
            ++w_rows;
            CK(hipMalloc((void**)&j.dout, 2048 * 2));
            j.dnorm = nullptr;
            if (j.nout != "-") CK(hipMalloc((void**)&j.dnorm, 2048 * 2));
            jobs.push_back(j);
        }
    }
    if (jobs.empty()) {
        fprintf(stderr, "plan %s has no jobs\n", plan_path.c_str());
        return 1;
    }
    printf("plan %s | jobs=%zu | 16 blocks x 128 threads per job | neg=%s\n", plan_path.c_str(),
           jobs.size(), negname.c_str());
    printf("weights: %zu [128] F32 rows read back from the store, all finite, |value| <= %.4g"
           " (an affine row is O(1): a BF16 misread of these bytes lands at 1e-38, not here)\n",
           w_rows, w_absmax);
    if (neg != GN_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    auto launch = [&](Job& j) {
        gdn_gated_norm<<<GN_HEADS, GN_D_V>>>(j.do_, j.dz, j.dw, j.dout, j.dnorm, neg);
        CK(hipGetLastError());
    };

    if (mode == "once") {
        // Prefill every output with a non-zero sentinel before launching, then report per job what the
        // launch left behind. The first run of this gate reported a successful launch (`RESULT
        // mode=once jobs=144` printed, every negative plan too) and wrote files that were mostly +0
        // and NaN — which is indistinguishable from untouched fresh pages, the thing hipMalloc hands
        // back zero-filled. With a 0x3F3F prefill, "nothing was written" and "something wrong was
        // written" are different sentences on the first re-run instead of a second probe.
        for (auto& j : jobs) {
            CK(hipMemset(j.dout, GN_SENTINEL & 0xFF, 2048 * 2));   // both bytes, so 0x3F3F
            if (j.dnorm) CK(hipMemset(j.dnorm, GN_SENTINEL & 0xFF, 2048 * 2));
            std::vector<unsigned char> pre(16);
            CK(hipMemcpy(pre.data(), (void*)j.dout, 16, hipMemcpyDeviceToHost));
            j.pre_hash = fnv1a64(pre.data(), 16);
        }
        for (auto& j : jobs) launch(j);
        CK(hipDeviceSynchronize());
        std::vector<hip_bfloat16> host(2048);
        size_t unwritten = 0, partial = 0, norm_unwritten = 0, norm_partial = 0;
        const size_t last = jobs.size() - 1;
        for (size_t i = 0; i < jobs.size(); ++i) {
            Job& j = jobs[i];
            CK(hipMemcpy(host.data(), (void*)j.dout, 2048 * 2, hipMemcpyDeviceToHost));
            int first = -1;
            const size_t left = sentinel_left(host.data(), 2048, &first);
            const uint64_t post = fnv1a64(host.data(), 16);
            unwritten += (left == 2048);
            partial += (left > 0 && left < 2048);
            // Per-job line for every offender, plus the first and last job: "only the last job was
            // written" is an output-mapping bug and those two lines are what show it.
            if (left || i == 0 || i == last) {
                char where[48];
                if (first < 0) snprintf(where, sizeof where, "first@none");
                else snprintf(where, sizeof where, "first@%d (h%d,d%d)", first, first / GN_D_V,
                              first % GN_D_V);
                printf("sentinel: %s left %4zu/2048 %s  first16 fnv %016llx -> %016llx  %s\n",
                       j.name.c_str(), left, where, (unsigned long long)j.pre_hash,
                       (unsigned long long)post,
                       left == 2048 ? "NOT WRITTEN" : left ? "PARTIALLY WRITTEN" : "written");
            }
            spit(j.out, host.data(), 2048 * 2);
            if (j.nout != "-") {
                CK(hipMemcpy(host.data(), (void*)j.dnorm, 2048 * 2, hipMemcpyDeviceToHost));
                int nfirst = -1;
                const size_t nleft = sentinel_left(host.data(), 2048, &nfirst);
                norm_unwritten += (nleft == 2048);
                norm_partial += (nleft > 0 && nleft < 2048);
                if (nleft) {
                    char nwhere[48];
                    if (nfirst < 0) snprintf(nwhere, sizeof nwhere, "first@none");
                    else snprintf(nwhere, sizeof nwhere, "first@%d (h%d,d%d)", nfirst,
                                  nfirst / GN_D_V, nfirst % GN_D_V);
                    printf("sentinel: %s norm left %zu/2048 %s (%s)\n", j.name.c_str(), nleft,
                           nwhere, nleft == 2048 ? "NOT WRITTEN" : "PARTIALLY WRITTEN");
                }
                spit(j.nout, host.data(), 2048 * 2);
            }
        }
        // Both buffers get both counts. The first version printed only "norm buffers never written"
        // here while the per-job lines above were reporting partials, so the summary read "0 norm
        // buffers never written" next to 252 "PARTIALLY WRITTEN" lines and the two looked like a
        // contradiction when the summary was simply silent about the category it did not count.
        printf("write-back: gated %zu/%zu never written, %zu partial; norm %zu/%zu never written,"
               " %zu partial (prefill 0x%04x, unwritable by the kernel, so a survivor is an"
               " unwritten lane)\n",
               unwritten, jobs.size(), partial, norm_unwritten, jobs.size(), norm_partial,
               GN_SENTINEL);
        printf("RESULT mode=once jobs=%zu neg=%s unwritten=%zu partial=%zu norm_unwritten=%zu"
               " norm_partial=%zu\n", jobs.size(), negname.c_str(), unwritten, partial,
               norm_unwritten, norm_partial);
        return 0;
    }

    // Timing: wall-clock over a burst, never hipEventElapsedTime (rule 7 — sub-ms reads exactly 0
    // on this iGPU). The bytes here are ~14 KiB/launch, so this op is a launch-count item like
    // glue; GB/s is printed to show that, not as a claim.
    double bytes_token = 0;
    for (auto& j : jobs) bytes_token += j.bytes();
    auto burst = [&](int n) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i)
            for (auto& j : jobs) launch(j);
        CK(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    burst(1);
    double best = 1e30, worst = 0;
    for (int t = 0; t < std::max(1, trials); ++t) {
        const double ms = burst(iters) / iters;
        best = std::min(best, ms);
        worst = std::max(worst, ms);
    }
    printf("TIMING jobs=%zu best=%.4f ms/plan spread=%.1f%% us/launch=%.3f (launch floor 1.85)"
           " bytes=%.1f KiB/plan\n",
           jobs.size(), best, best > 0 ? 100.0 * (worst - best) / best : 0.0,
           best * 1000.0 / jobs.size(), bytes_token / jobs.size() / 1024.0);
    printf("RESULT mode=burst jobs=%zu iters=%d trials=%d ms_per_plan=%.6f us_per_launch=%.4f"
           " neg=%s\n", jobs.size(), iters, trials, best, best * 1000.0 / jobs.size(),
           negname.c_str());
    return 0;
}
