/*
 * prototype/glue/glue.hip.cpp — one launch for the two things that sit between the GEMVs:
 * the residual add (§03.2) and the zero-centred RMSNorm that follows it (§03.1).
 *
 *     s = s + delta            §03.2: a BF16 add, no bias, no gate, no per-layer scaling
 *     out = RMSnorm_zc(s) · (1 + w)    §03.1: F32 math, mean of SQUARES over 1024, eps 1e-6,
 *                                             affine (1+w) in F32, cast back ONCE
 *
 * Both in one launch is the whole reason this file exists as its own op. Per §03.2 a layer runs
 * 2 norms + 2 residual adds, so an unfused path costs 4 launches per layer (97/token including
 * the norm before layer 0, which has no add in front of it); fused it is 2 per layer (49/token).
 * At the measured 1.85 us/launch floor (bench/performance-status.md §1) that is 0.179 ms/token
 * unfused against 0.091 ms/token fused, and folding the pair into the GEMV that produces `delta`
 * (the `mlp_gateup` pattern) would take it to 0. The BYTES here are irrelevant by comparison —
 * see results.md; this is a launch-count item, and reporting it in GB/s would be reporting the
 * wrong thing.
 *
 * WHAT THERE IS TO GET WRONG, and each one has an injected bug or a Python-side falsification:
 *   1. the affine is `1 + w`, not `w` (§03.1 — "Gemma form"). `plainw`.
 *   2. the norm applies to the UPDATED `s`, not to `delta` and not to the old residual
 *      (§03.2's order: `x = residual + h` then `h = RMSnorm(x)`). `normdelta`, `noadd`.
 *   3. mean of SQUARES, not mean (§03.1). `mean`.
 *   4. the squared-sum accumulator must stay F32. Every other number in this path is BF16, so a
 *      BF16 accumulator looks free and is the one real accuracy risk here: `bf16acc` keeps the
 *      whole reduction in BF16, `bf16sub` rounds only this thread's partial before it joins the
 *      F32 ladder. results.md carries what each costs.
 *   5. NO GATING on this path. This checkpoint's blocks have no residual gate and no per-layer
 *      scale (§03.2 says so explicitly), so `delta` goes into the sum raw and `out` goes out raw.
 *      The gate-level falsification lives in run.py's `verify_rebuild()` (multiply by
 *      sigmoid(z) or by the §04.5/§05.6 block gate and the pins stop reproducing). Note that
 *      `mlp.gate_proj` DOES exist here (24 of them, §01.2) and is the SwiGLU gate inside the MLP
 *      — `prototype/mlp_gemv` already consumes it. It is not a gate on the residual stream and
 *      nothing in this file should read it.
 *
 * Rounding, fixed by §03.1/§03.2 and matched to `speccheck.specimpl.rmsnorm_zc` element order:
 * cast s to F32; square-sum in F32; `r = rsqrt(mean + eps)`; `out_f32 = (s_f32 * r) * (1 + w)`;
 * cast to BF16 once. The residual add rounds to BF16 before the norm sees it, because the
 * reference stores `x` as BF16 and the norm casts that stored value.
 *
 * WEIGHTS COME FROM THE ENGINE LOADER. `s` and `delta` are per-token activations and stay
 * ordinary buffers, but the 49 norm vectors are weights, so this binary links
 * `engine/loader.{h,cpp}` and resolves every `w` by its normalised §02.5 name
 * (`lm.layers.N.post_attention_layernorm.weight`, `lm.layers.N.input_layernorm.weight`,
 * `lm.norm.weight`) through `engine::WeightStore::at()`. No hand-rolled H2D for a weight
 * anywhere in this directory, and the loader's warm pass is what prototype/README.md rule 3
 * demands before timing. run.py asserts dtype BF16 and shape [1024] on every name it fetches.
 *
 * INJECTED BUGS (each also takes --scope, so the bug lands only on the call that can show it and
 * the must-NOT-catch claims mean something):
 *   plainw      affine is w instead of 1+w
 *   normdelta   normalise `delta` instead of the updated s
 *   noadd       skip the residual add entirely
 *   mean        rsqrt(mean(s) + eps) instead of rsqrt(mean(s^2) + eps)
 *   bf16acc     the squared-sum accumulator is BF16 end to end (per-thread adds AND the ladder)
 *   bf16sub     only this thread's partial is rounded to BF16 before the F32 ladder
 *
 * Build: prototype/glue/build.sh   (-O3, -Wl,-rpath,/opt/rocm/lib, links ../../engine/loader.cpp)
 * Run:   ./glue --mode once   --plan PLAN [--threads 256] [--neg NAME] [--scope 1|2|0] [--split]
 *        ./glue --mode burst  --plan PLAN [--threads 256] [--iters N] [--trials N]
 *                            [--chained] [--split]
 * Validate: prototype/glue/run.py
 *
 * PLAN is a text file, one launch per line:
 *     NAME SFILE DFILE WNAME SOUT NFOUT
 * SFILE/DFILE are raw little-endian BF16, 1024 elements; `-` for DFILE means "norm only" (the
 * norm in front of layer 0, which has no add). WNAME is an ENGINE NORMALISED weight name,
 * resolved through the loader, not a file. SOUT/NFOUT are output paths, `-` to skip. A job name
 * ending in `.g1` / `.g2` is call 1 / call 2 of a layer, which is what --scope matches on.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "engine/loader.h"

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit
#define H 1024                            // §00: the hidden size every §03.1 norm runs over
#define EPS 1e-6f                         // §03.1
#define PROBE_GBS 237.0                   // devices/strix-halo-memory-bandwidth.md read probe

enum Neg {
    NEG_NONE = 0,
    NEG_PLAINW,      // (1 + w) -> w
    NEG_NORMDELTA,   // norm the addend instead of the updated residual
    NEG_NOADD,       // skip s += delta
    NEG_MEAN,        // mean instead of mean-of-squares
    NEG_BF16ACC,     // squared-sum accumulator lives in BF16
    NEG_BF16SUB,     // one BF16 subpartial, F32 ladder
};

__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }

// "Store this into a BF16 slot": the RNE round-trip the reference does when it writes the
// residual back. Also what a BF16 accumulator does to a running sum, which is why the two
// BF16 negatives are written in terms of it.
__device__ __forceinline__ float bf_round(float f) { return float(hip_bfloat16(f)); }

// One-block reduction for a 1024-element row: intra-warp shuffle ladder, then warp partials
// through shared memory. `bf16_tree` is NEG_BF16ACC — the same ladder with the running value
// put back through a BF16 slot at every step.
__device__ __forceinline__ float block_reduce(float v, float* red, bool bf16_tree) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) {
        const float n = __shfl_down_sync(FULL_MASK, v, o);
        v = bf16_tree ? bf_round(v + n) : v + n;
    }
    if ((threadIdx.x & (ws - 1)) == 0) red[threadIdx.x / ws] = v;
    __syncthreads();
    // Second ladder runs in warp 0 with every lane of that warp live, which is what makes the
    // full participating-lane mask correct here rather than a lie.
    if (threadIdx.x < (unsigned)ws) {
        float x = threadIdx.x < blockDim.x / ws ? red[threadIdx.x] : 0.f;
        for (int o = ws / 2; o; o >>= 1) {
            const float n = __shfl_down_sync(FULL_MASK, x, o);
            x = bf16_tree ? bf_round(x + n) : x + n;
        }
        if (threadIdx.x == 0) red[0] = x;
    }
    __syncthreads();
    return red[0];
}

/*
 * The fused op. One block, whole row resident in registers (`v[]`), so `s` and `delta` are
 * each read once and the norm reads the values it already has instead of going back to memory.
 */
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
glue_add_norm(const hip_bfloat16* __restrict__ s, const hip_bfloat16* __restrict__ delta,
              const hip_bfloat16* __restrict__ w, hip_bfloat16* __restrict__ s_out,
              hip_bfloat16* __restrict__ n_out, int neg) {
    __shared__ float red[32];   // >= THREADS/warpSize warp partials (1024/32 is the most)
    constexpr int PER = H / THREADS;
    const bool norm_delta = (neg == NEG_NORMDELTA) && delta != nullptr;
    const bool bf16_acc = (neg == NEG_BF16ACC);

    float v[PER];
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;   // consecutive threads, consecutive lanes
        const float sv = bf(s[idx]);
        float m = sv;
        if (delta && neg != NEG_NOADD) {
            m = bf_round(sv + bf(delta[idx]));       // §03.2: the residual add, BF16
            if (s_out) s_out[idx] = hip_bfloat16(m);
        } else if (s_out) {
            s_out[idx] = s[idx];                     // norm-only: the residual is unchanged
        }
        v[i] = m;
        float t = norm_delta ? bf(delta[idx]) : m;
        if (neg != NEG_MEAN) t *= t;                 // §03.1: mean of SQUARES
        acc = bf16_acc ? bf_round(acc + t) : acc + t;
    }
    if (neg == NEG_BF16SUB) acc = bf_round(acc);
    const float sum = block_reduce(acc, red, bf16_acc);
    const float r = rsqrtf(sum * (1.0f / H) + EPS);

#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        const float wv = bf(w[idx]);
        const float aff = (neg == NEG_PLAINW) ? wv : (1.0f + wv);
        n_out[idx] = hip_bfloat16((v[i] * r) * aff);   // §03.1 order, one cast back
    }
}

/*
 * The unfused pair, for measuring what fusion is worth rather than asserting it: `residual_add`
 * writes the new residual, then `norm_only` reads it back from memory and normalises it. Same
 * arithmetic, twice the launches, one extra round trip through DRAM/L2 per row.
 */
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
residual_add(const hip_bfloat16* __restrict__ s, const hip_bfloat16* __restrict__ delta,
             hip_bfloat16* __restrict__ s_out) {
    constexpr int PER = H / THREADS;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        s_out[idx] = hip_bfloat16(bf_round(bf(s[idx]) + bf(delta[idx])));
    }
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
norm_only(const hip_bfloat16* __restrict__ s, const hip_bfloat16* __restrict__ w,
          hip_bfloat16* __restrict__ n_out) {
    __shared__ float red[32];
    constexpr int PER = H / THREADS;
    float v[PER];
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        v[i] = bf(s[idx]);
        acc += v[i] * v[i];
    }
    const float r = rsqrtf(block_reduce(acc, red, false) * (1.0f / H) + EPS);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * THREADS;
        n_out[idx] = hip_bfloat16((v[i] * r) * (1.0f + bf(w[idx])));
    }
}

// ---------------------------------------------------------------- host side
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

struct Job {
    std::string name, sfile, dfile, wname, sout, nout;
    int scope = 0;                 // 1 = first glue call of a layer, 2 = second, 0 = neither/both
    const hip_bfloat16 *ds = nullptr, *dd = nullptr, *dw = nullptr;
    hip_bfloat16 *dsout = nullptr, *dnout = nullptr;
    size_t bytes_fused = 0, bytes_split = 0;
    int launches = 0;
};

int main(int argc, char** argv) {
    std::string plan_path, negname = "none", model = "models/Qwen3.5-0.8B", threads_list;
    std::string mode = "once";
    int scope = 1, iters = 200, trials = 3, threads = 256;
    bool split = false, chained = false, loader_verify = false, quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--neg") negname = next();
        else if (a == "--scope") scope = atoi(next().c_str());
        else if (a == "--split") split = true;
        else if (a == "--chained") chained = true;
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--threads-list") threads_list = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--trials") trials = atoi(next().c_str());
        else if (a == "--model") model = next();
        else if (a == "--loader-verify") loader_verify = true;
        else if (a == "--quiet") quiet = true;
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }
    const int neg = negname == "none" ? NEG_NONE : negname == "plainw" ? NEG_PLAINW
                  : negname == "normdelta" ? NEG_NORMDELTA : negname == "noadd" ? NEG_NOADD
                  : negname == "mean" ? NEG_MEAN : negname == "bf16acc" ? NEG_BF16ACC
                  : negname == "bf16sub" ? NEG_BF16SUB : -1;
    if (neg < 0) {
        fprintf(stderr, "--neg must be none|plainw|normdelta|noadd|mean|bf16acc|bf16sub\n");
        return 1;
    }
    if (plan_path.empty() || (mode != "once" && mode != "burst")) {
        fprintf(stderr, "usage: glue --mode once|burst --plan PLAN [options]\n");
        return 1;
    }
    if (chained && mode != "burst") {
        fprintf(stderr, "--chained only makes sense for timing; correctness runs feed pins\n");
        return 1;
    }
    if (threads != 128 && threads != 256 && threads != 512 && threads != 1024) {
        fprintf(stderr, "--threads must be 128, 256, 512 or 1024 (H=1024 must split evenly)\n");
        return 1;
    }

    // ---- the engine loader is the only way a weight gets to the device here. It is also what
    // prototype/README.md rule 3 (warm the pages before timing) is satisfied by: load() does one
    // GPU read pass over every resident page before it returns.
    engine::WeightStore::Options lopts;
    lopts.model_dir = model;
    lopts.verify_device_bytes = loader_verify;      // ~1.55 GB D2H + hashing, opt-in only
    lopts.measure_steady_h2d = false;               // this consumer does not need that probe
    engine::LoadStats lstats;
    engine::WeightStore store = engine::WeightStore::load(lopts, lstats);
    if (!quiet)
        printf("loader: %zu tensors resident, %.3f GiB, warm %.1f ms (%.1f GB/s), total %.1f ms"
               "\n",
               lstats.n_loaded, lstats.uploaded_bytes / 1073741824.0, lstats.warm_ms,
               lstats.warm_ms > 0 ? lstats.warm_bytes / (1.0e6 * lstats.warm_ms) : 0.0,
               lstats.total_ms);
    if (lstats.warm_checksum == 0) {
        fprintf(stderr, "loader: warm pass read nothing (checksum 0)\n");
        return 1;
    }

    std::vector<std::string> ths;
    {
        std::stringstream ss(threads_list.empty() ? std::to_string(threads) : threads_list);
        std::string t;
        while (std::getline(ss, t, ','))
            if (!t.empty()) ths.push_back(t);
    }

    std::vector<Job> jobs;
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
            if (!(ls >> j.name >> j.sfile >> j.dfile >> j.wname >> j.sout >> j.nout)) {
                fprintf(stderr, "plan %s line %d: NAME SFILE DFILE WNAME SOUT NFOUT\n",
                        plan_path.c_str(), ln);
                return 1;
            }
            const size_t tail = j.name.rfind('.');
            const std::string suf = tail == std::string::npos ? "" : j.name.substr(tail);
            j.scope = suf == ".g1" ? 1 : suf == ".g2" ? 2 : 0;
            jobs.push_back(j);
        }
    }
    if (jobs.empty()) {
        fprintf(stderr, "plan %s has no jobs\n", plan_path.c_str());
        return 1;
    }

    // Activations: ordinary per-token buffers, hipMalloc + hipMemcpy, one upload for the run.
    std::vector<std::string> paths;
    std::vector<hip_bfloat16*> devs;
    auto act = [&](const std::string& p) -> hip_bfloat16* {
        for (size_t i = 0; i < paths.size(); ++i)
            if (paths[i] == p) return devs[i];
        std::vector<char> raw = slurp(p);
        if (raw.size() != H * 2) {
            fprintf(stderr, "%s: %zu bytes, expected %d bf16 elements\n", p.c_str(), raw.size(), H);
            exit(1);
        }
        hip_bfloat16* d = nullptr;
        CK(hipMalloc(&d, H * 2));
        CK(hipMemcpy(d, raw.data(), H * 2, hipMemcpyHostToDevice));
        paths.push_back(p);
        devs.push_back(d);
        return d;
    };

    std::set<std::string> wnames;
    size_t wbytes = 0;
    for (auto& j : jobs) {
        j.ds = act(j.sfile);
        j.dd = j.dfile == "-" ? nullptr : act(j.dfile);
        const engine::Weight& wt = store.at(j.wname);   // throws if the name is not the engine's
        if (wt.dtype != engine::Dtype::BF16 || wt.nbytes != H * 2 || wt.shape.size() != 1
            || wt.shape[0] != H) {
            fprintf(stderr, "%s: %s is %s %s (%zu B), expected a BF16 [%d] norm vector\n",
                    j.name.c_str(), j.wname.c_str(), engine::dtype_name(wt.dtype),
                    wt.shape_string().c_str(), wt.nbytes, H);
            return 1;
        }
        j.dw = (const hip_bfloat16*)wt.device;
        if (wnames.insert(j.wname).second) wbytes += wt.nbytes;
        CK(hipMalloc(&j.dsout, H * 2));
        CK(hipMalloc(&j.dnout, H * 2));
        // Traffic the kernel itself moves, per launch of this job.
        //   fused, with a delta: read s + delta + w, write s_out + norm_out  = 5 vectors
        //   fused, norm only:    read s + w, write norm_out                  = 3 vectors
        //   split, with a delta: add reads s + delta writes s_out (3), then
        //                        norm reads s_out + w writes norm_out (3)   = 6 vectors
        //   split, norm only:    norm reads s + w writes norm_out            = 3 vectors
        const bool writes_s = j.sout != "-";
        j.bytes_fused = (size_t)((j.dd ? 3 : 2) + (writes_s ? 1 : 0) + 1) * H * 2;
        j.bytes_split = (size_t)(j.dd ? 6 : 3) * H * 2;
        j.launches = split ? (j.dd ? 2 : 1) : 1;
    }
    if (!quiet)
        printf("plan %s | jobs=%zu | norm weights: %zu distinct, %.1f KiB, all via"
               " engine::WeightStore | neg=%s scope=%d split=%d\n",
               plan_path.c_str(), jobs.size(), wnames.size(), wbytes / 1024.0,
               negname.c_str(), scope, (int)split);
    if (neg != NEG_NONE && !quiet)
        printf("NEGATIVE TEST: injected bug = %s on jobs with scope %d (or all if scope 0)\n",
               negname.c_str(), scope);

    auto launch = [&](Job& j, const hip_bfloat16* s, int th) {
        const int eff = (scope == 0 || j.scope == scope) ? neg : NEG_NONE;
        hip_bfloat16* so = j.sout == "-" ? nullptr : j.dsout;
        if (split) {
            // The unfused pair reads what the add wrote, from memory — that round trip is part
            // of what fusion removes, so it must not be optimised away here.
            if (j.dd) {
                if (th == 128) residual_add<128><<<1, 128>>>(s, j.dd, j.dsout);
                else if (th == 256) residual_add<256><<<1, 256>>>(s, j.dd, j.dsout);
                else if (th == 512) residual_add<512><<<1, 512>>>(s, j.dd, j.dsout);
                else residual_add<1024><<<1, 1024>>>(s, j.dd, j.dsout);
            }
            const hip_bfloat16* src = j.dd ? j.dsout : s;
            if (th == 128) norm_only<128><<<1, 128>>>(src, j.dw, j.dnout);
            else if (th == 256) norm_only<256><<<1, 256>>>(src, j.dw, j.dnout);
            else if (th == 512) norm_only<512><<<1, 512>>>(src, j.dw, j.dnout);
            else norm_only<1024><<<1, 1024>>>(src, j.dw, j.dnout);
        } else if (th == 128) {
            glue_add_norm<128><<<1, 128>>>(s, j.dd, j.dw, so, j.dnout, eff);
        } else if (th == 512) {
            glue_add_norm<512><<<1, 512>>>(s, j.dd, j.dw, so, j.dnout, eff);
        } else if (th == 1024) {
            glue_add_norm<1024><<<1, 1024>>>(s, j.dd, j.dw, so, j.dnout, eff);
        } else {
            glue_add_norm<256><<<1, 256>>>(s, j.dd, j.dw, so, j.dnout, eff);
        }
        CK(hipGetLastError());
    };

    if (mode == "once") {
        for (auto& j : jobs) launch(j, j.ds, threads);
        CK(hipDeviceSynchronize());
        std::vector<hip_bfloat16> host(H);
        for (auto& j : jobs) {
            if (j.sout != "-") {
                CK(hipMemcpy(host.data(), j.dsout, H * 2, hipMemcpyDeviceToHost));
                spit(j.sout, host.data(), H * 2);
            }
            if (j.nout != "-") {
                CK(hipMemcpy(host.data(), j.dnout, H * 2, hipMemcpyDeviceToHost));
                spit(j.nout, host.data(), H * 2);
            }
        }
        printf("RESULT mode=once jobs=%zu neg=%s scope=%d threads=%d split=%d\n", jobs.size(),
               negname.c_str(), scope, threads, (int)split);
        return 0;
    }

    // Timing. Wall-clock over a burst, never hipEventElapsedTime: this whole op is
    // microseconds and the event timer returns exactly 0 below ~1 ms on this iGPU
    // (prototype/README.md rule 7). --chained makes job i read job i-1's output, which is what
    // a token actually does — the residual chain is serial, so a plan whose launches all read
    // their own input file and can overlap measures the submit path, not the token.
    double bytes_token = 0;
    int launches_token = 0;
    for (auto& j : jobs) {
        bytes_token += split ? (double)j.bytes_split : (double)j.bytes_fused;
        launches_token += j.launches;
    }
    const double us_per_launch_floor = 1.85;   // bench/performance-status.md §1

    for (const auto& ts : ths) {
        const int th = atoi(ts.c_str());
        auto burst = [&](int n) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) {
                // --chained: job k reads the residual as job k-1 left it. That is the real
                // dependency, and it is what makes the number a token rather than a submit test.
                const hip_bfloat16* cur = nullptr;
                for (size_t k = 0; k < jobs.size(); ++k) {
                    const hip_bfloat16* s = chained ? (cur ? cur : jobs[k].ds) : jobs[k].ds;
                    launch(jobs[k], s, th);
                    if (chained && jobs[k].sout != "-") cur = jobs[k].dsout;
                }
            }
            CK(hipDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        burst(1);                              // warm: fresh pages run ~7x slow (devices/)
        double best = 1e30, worst = 0;
        for (int t = 0; t < std::max(1, trials); ++t) {
            const double ms = burst(iters) / iters;
            best = std::min(best, ms);
            worst = std::max(worst, ms);
        }
        const double us = best * 1000.0 / launches_token;
        const double gbs = bytes_token / (best / 1000.0) / 1.0e9;
        printf("TIMING threads=%d split=%d chained=%d launches/token=%d best=%.4f ms/token"
               " spread=%.1f%% us/launch=%.3f (launch floor %.2f) bytes/token=%.3f MB"
               " roofline=%.4f ms @%.0f GB/s  [%.1f GB/s — not the finding, see results.md]\n",
               th, (int)split, (int)chained, launches_token, best,
               best > 0 ? 100.0 * (worst - best) / best : 0.0, us, us_per_launch_floor,
               bytes_token / 1.0e6, bytes_token / PROBE_GBS / 1.0e6, PROBE_GBS, gbs);
        printf("RESULT mode=burst threads=%d split=%d chained=%d iters=%d trials=%d"
               " launches=%d ms_per_token=%.6f us_per_launch=%.4f ms_per_launch_floor=%.4f\n",
               th, (int)split, (int)chained, iters, trials, launches_token, best, us,
               launches_token * us_per_launch_floor / 1000.0);
    }
    return 0;
}
