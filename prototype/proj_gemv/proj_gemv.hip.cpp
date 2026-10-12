/*
 * prototype/proj_gemv/proj_gemv.hip.cpp — the projection GEMV family, k = 1: one templated
 * row-per-wavefront BF16 GEMV instantiated over seven shapes, not seven kernels.
 *
 * The shapes (specification/01 §01.2/§01.3, sizes from `tools/weights_report.py`):
 *
 *   linear_attn.in_proj_qkv [6144,1024] x18   linear_attn.in_proj_z [2048,1024] x18
 *   linear_attn.in_proj_b   [  16,1024] x18   linear_attn.in_proj_a [  16,1024] x18
 *   linear_attn.out_proj    [1024,2048] x18   self_attn.q_proj      [4096,1024] x6
 *   self_attn.k_proj        [  512,1024] x6   self_attn.v_proj      [  512,1024] x6
 *   self_attn.o_proj        [ 1024,2048] x6
 *
 * Together 0.4677 GB/token = 29 % of the token's bytes (bench/performance-status.md §2), the
 * largest slice left after the MLP. Only two input widths appear — 1024 and 2048 — so the whole
 * family is `proj_gemv<1024>` and `proj_gemv<2048>`: the same skeleton as `mlp_gemv`'s
 * `mlp_down<IN>` (one wavefront per output row, x staged in shared, F32 accumulator, one BF16
 * rounding per dot; A-07, and §10.1's F1 line for what "reproduce" means here).
 *
 * LAUNCH PLAN. Every projection with the same input width and the same x can share one launch by
 * concatenating rows at load time, and the engine should: {in_proj_qkv, in_proj_z, in_proj_b,
 * in_proj_a} are all [*,1024] of the same normed x -> one [8224,1024] launch; {q_proj, k_proj,
 * v_proj} -> one [5120,1024] launch. out_proj/o_proj are [1024,2048] fed by a *different* x (the
 * gated norm output, §04.5/§05.6), so they cannot join, and nothing can be merged across a block
 * boundary because the residual add and the next norm sit in between. So: 2 launches per layer,
 * 48/token against 114 unfused — ~0.12 ms/token at the measured 1.85 us launch floor.
 * in_proj_b/in_proj_a are [16,1024] = 32 KiB per layer, i.e. launch cost rather than bandwidth,
 * which is why they are folded rather than timed as shapes of their own. The driver below is
 * plan-driven so the gate can exercise packed and unpacked forms with one code path.
 *
 * TOKEN BATCHING (P5). A prefill wave hands one projection a [T,IN] tile instead of one row, and
 * `proj_gemv_tokens` reads each weight row once per TPB-token tile rather than once per token; that
 * is the family's only bandwidth lever, because the narrow shapes are shape-limited (results.md:
 * `k_proj`/`v_proj` at 39-40 % of the read probe, `o_proj`/`out_proj` 46-49 %, `in_proj_a`/`b` ~5 %).
 * The engine issues that kernel at every width, one token included. §06.2's "one GEMM per
 * projection" is this sentence.
 *
 * Rounding: one F32 dot, rounded once to BF16 (RNE, the cast torch does). There is no epilogue to
 * get wrong here (unlike the MLP's three-rounding SwiGLU); what there is to get wrong is the row
 * LAYOUT — §05.1's per-head `[8 heads][q 256 | gate 256]` q_proj block and §04.1's `[q|k|v]`
 * split — which is why two of the injected bugs below are output-row permutations.
 *
 * Injected bugs, each exercised by run.py --negative with a declared must-catch / must-not-catch
 * location (the `mlp_gemv` convention: a negative that cannot fail is worse than none):
 *   tail       drop the last word of K (the off-by-one a `v < K/8 - 1` loop bound gives)
 *   half       accumulate half of K
 *   trunc      round toward zero instead of round-to-nearest-even
 *   rowswap    write each dot to the neighbouring output row (row/output swap)
 *   transpose  read the weight buffer transposed (rows/in reversed)
 *   qhalf      write q_proj rows in the contiguous-half [all q | all gate] layout §05.1 forbids
 *   kvswap     swap the k and v sections of a packed in_proj_qkv output (§04.1)
 * Plus two input-side negatives run by run.py with no kernel change: the pre-norm tensor, and the
 * other norm's weights, where the input-layernorm output belongs (see NEG_INPUTS in run.py).
 *
 * Build: prototype/proj_gemv/build.sh   (-O3, -Wl,-rpath,/opt/rocm/lib)
 * Run:   ./proj_gemv --mode once  --plan PLAN [--rows N] [--neg NAME] [--x-packed]
 *        ./proj_gemv --mode once  --plan PLAN --tokens T [--tpb N] [--kernel k1]
 *        ./proj_gemv --mode burst --plan PLAN [--rows N] [--iters N] [--tile N] [--write-out]
 *        ./proj_gemv --mode burst --plan PLAN --rows-list 2,4,8,16,32 [--tile N] [--tpb N]
 * Validate: prototype/proj_gemv/run.py
 *
 * PLAN is a text file, one launch per line:
 *     NAME ROWS IN XFILE WFILE[,WFILE...] [OUTFILE]
 * XFILE is raw little-endian BF16 with IN elements — or, with `--tokens T`, with T*IN elements, the
 * [T,IN] input tile one prefill wave presents (row-major, token-major). WFILEs are row-major BF16
 * [rows_i, IN] concatenated in the order given (ROWS must equal the sum); OUTFILE, if present,
 * receives the BF16 output — ROWS elements in the k=1 form, TOKENS*ROWS elements in the batched
 * one, laid out [token][row] — always in `once` mode, in `burst` mode only with --write-out.
 *
 * --kernel picks which kernel the plan runs through: `tokens` (default) is the tile form the engine
 * uses at every width, `TPB = 1` included; `k1` is the original one-wavefront-per-row kernel, kept
 * as the batched gate's independent per-row reference. --tpb overrides the tile width the launcher
 * picks (4 at in=1024, 2 at in=2048, 1 when tokens == 1); it exists for the main agent's sweep, and
 * nothing in the gated correctness path uses a value other than the launcher's own.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "proj_gemv_kernels.h"

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

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

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, sep))
        if (!t.empty()) out.push_back(t);
    return out;
}

struct Weight {                       // one launch's concatenated weight blob, `tile` replicas
    std::string key;
    std::vector<std::string> files;
    int in = 0, rows = 0;
    size_t bytes = 0;
    std::vector<hip_bfloat16*> rep;
};

struct Job {
    std::string name, xfile, outfile;
    int widx = 0, xidx = 0;
    int rows = 0, in = 0, tpb = 1, tokens = 1;
    size_t wbytes = 0, xbytes = 0;
    hip_bfloat16* dout = nullptr;
    // Traffic accounting for one launch of this job: weight bytes once per TPB-token tile, and each
    // distinct x / output element once per tile pass. Same level of approximation the k=1 form used
    // (x is in fact re-staged by every row block of the launch, and was counted once there too); the
    // divisor that matters — weights per token-tile instead of per token — is exact.
    size_t bytes() const {
        const size_t tiles = (size_t)((tokens + tpb - 1) / tpb);
        return wbytes * tiles + xbytes * tiles + (size_t)tokens * rows * 2;
    }
};

struct Plan {
    std::vector<Weight> weights;
    std::vector<float*> xdev;              // widened F32 x, one per distinct x file
    std::vector<hip_bfloat16*> xdev_bf;    // the same x left as BF16
    std::vector<std::string> xpaths;
    int tokens = 1;                        // rows in each XFILE tile; set from --tokens
    std::vector<Job> jobs;

    int weight_index(const std::vector<std::string>& files, int in) {
        std::string key;
        int rows = 0;
        size_t bytes = 0;
        for (const auto& f : files) {
            const size_t n = slurp(f).size();
            if (n % ((size_t)in * 2)) {
                fprintf(stderr, "%s: %zu bytes is not a whole number of bf16 rows of in=%d\n",
                        f.c_str(), n, in);
                exit(1);
            }
            key += f + "+";
            rows += (int)(n / ((size_t)in * 2));
            bytes += n;
        }
        for (int i = 0; i < (int)weights.size(); ++i)
            if (weights[i].key == key) return i;
        weights.push_back(Weight{});
        Weight& w = weights.back();
        w.key = key;
        w.files = files;
        w.in = in;
        w.rows = rows;
        w.bytes = bytes;
        return (int)weights.size() - 1;
    }

    int x_index(const std::string& path, int in) {
        for (int i = 0; i < (int)xpaths.size(); ++i)
            if (xpaths[i] == path) return i;
        const size_t want = (size_t)tokens * in * 2;
        const std::vector<char> raw = slurp(path);
        if (raw.size() != want) {
            fprintf(stderr, "%s: %zu bytes, expected %zu (%d bf16 rows of in=%d)\n", path.c_str(),
                    raw.size(), want, tokens, in);            exit(1);
        }
        std::vector<float> host(raw.size() / 2);
        const hip_bfloat16* p = (const hip_bfloat16*)raw.data();
        for (size_t i = 0; i < host.size(); ++i) host[i] = float(p[i]);
        float* d = nullptr;
        hip_bfloat16* db = nullptr;
        CK(hipMalloc(&d, host.size() * sizeof(float)));
        CK(hipMemcpy(d, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
        CK(hipMalloc(&db, raw.size()));
        CK(hipMemcpy(db, raw.data(), raw.size(), hipMemcpyHostToDevice));
        xpaths.push_back(path);
        xdev.push_back(d);
        xdev_bf.push_back(db);
        return (int)xpaths.size() - 1;
    }
};

// One launch of the tile form. TPB is a template argument, so this is spelled out per width rather
// than reached through a runtime switch: a wrong width must not silently become a different kernel.
template <int IN, int TPB>
static void launch_tiles(dim3 grid, int threads, const hip_bfloat16* W, const void* xp,
                         hip_bfloat16* out, const Job& j, int rpb, int neg, bool packed) {
    if (packed) {
        proj_gemv_tokens<IN, TPB, true><<<grid, threads>>>(W, xp, out, j.rows, j.tokens, rpb, neg);
    } else {
        proj_gemv_tokens<IN, TPB, false><<<grid, threads>>>(W, xp, out, j.rows, j.tokens, rpb, neg);
    }
}

static void upload(Weight& w, int copies) {
    char* host = nullptr;
    CK(hipHostMalloc(&host, w.bytes));
    size_t off = 0;
    for (const auto& fn : w.files) {
        std::vector<char> raw = slurp(fn);
        memcpy(host + off, raw.data(), raw.size());
        off += raw.size();
    }
    for (int r = 0; r < std::max(1, copies); ++r) {
        hip_bfloat16* d = nullptr;
        CK(hipMalloc(&d, w.bytes));
        CK(hipMemcpy(d, host, w.bytes, hipMemcpyHostToDevice));
        w.rep.push_back(d);
    }
    CK(hipHostFree(host));
}

int main(int argc, char** argv) {
    std::string mode = "once", plan_path, negname = "none", rows_list, kernel = "tokens";
    int iters = 30, rows_per_block = 8, tile = 1, tokens = 1, tpb_flag = 0;
    bool x_packed = false, write_out = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--plan") plan_path = next();
        else if (a == "--rows") rows_per_block = atoi(next().c_str());
        else if (a == "--rows-list") rows_list = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--tile") tile = atoi(next().c_str());
        else if (a == "--tokens") tokens = atoi(next().c_str());
        else if (a == "--tpb") tpb_flag = atoi(next().c_str());
        else if (a == "--kernel") kernel = next();
        else if (a == "--neg") negname = next();
        else if (a == "--x-packed") x_packed = true;
        else if (a == "--write-out") write_out = true;
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }
    if (plan_path.empty()) {
        fprintf(stderr, "usage: proj_gemv --mode once|burst --plan PLAN [options]\n");
        return 1;
    }
    if (tokens < 1) {
        fprintf(stderr, "--tokens takes at least 1 token (got %d)\n", tokens);
        return 1;
    }
    if (kernel != "tokens" && kernel != "k1") {
        fprintf(stderr, "--kernel must be `tokens` (the tile form the engine runs) or `k1`\n");
        return 1;
    }
    // The launcher's own tile width, and the engine's: shared memory is TPB*IN floats and 16 KiB is
    // the budget the gated MLP settled on, so 4 rows at in=1024 and 2 at in=2048. One token is a tile
    // of one — that is what makes `tokens == 1` the k=1 shape rather than a special case.
    auto tile_width = [&](int in) {
        if (tpb_flag > 0) return tpb_flag;
        if (tokens == 1) return 1;
        return in == 1024 ? 4 : 2;
    };
    const int neg = negname == "none" ? NEG_NONE : negname == "tail" ? NEG_TAIL
                  : negname == "half" ? NEG_HALF : negname == "trunc" ? NEG_TRUNC
                  : negname == "rowswap" ? NEG_ROWSWAP : negname == "transpose" ? NEG_TRANSPOSE
                  : negname == "qhalf" ? NEG_QHALF : negname == "kvswap" ? NEG_KVS : -1;
    if (neg < 0) {
        fprintf(stderr, "--neg must be none|tail|half|trunc|rowswap|transpose|qhalf|kvswap\n");
        return 1;
    }
    if (mode != "once" && mode != "burst") {
        fprintf(stderr, "--mode must be once|burst\n");
        return 1;
    }

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    const int ws = prop.warpSize;
    const int ncu = prop.multiProcessorCount;
    if (ws != 32 && ws != 64) {
        fprintf(stderr, "warpSize %d unsupported by the row-dot reduction\n", ws);
        return 1;
    }

    Plan plan;
    plan.tokens = tokens;      // how many rows each XFILE carries (x_index checks against it)
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
            auto tok = split(line, ' ');
            if (tok.size() < 5) {
                fprintf(stderr, "plan %s line %d: NAME ROWS IN XFILE WFILES [OUTFILE]\n",
                        plan_path.c_str(), ln);
                return 1;
            }
            Job j;
            j.name = tok[0];
            j.rows = atoi(tok[1].c_str());
            j.in = atoi(tok[2].c_str());
            j.xfile = tok[3];
            auto wfiles = split(tok[4], ',');
            if (tok.size() > 5) j.outfile = tok[5];
            if (j.in != 1024 && j.in != 2048) {
                fprintf(stderr, "plan line %d: in=%d, only 1024 and 2048 are instantiated\n",
                        ln, j.in);
                return 1;
            }
            j.widx = plan.weight_index(wfiles, j.in);
            j.xidx = plan.x_index(j.xfile, j.in);
            j.tokens = tokens;
            j.tpb = tile_width(j.in);
            j.xbytes = (size_t)tokens * j.in * 2;
            // Traffic accounting: see Job::bytes(). `tile` replicas are cache-defeats, not extra
            // work, so they are NOT counted per launch — the reported GB/s is the same bytes a real
            // token moves.
            j.wbytes = plan.weights[j.widx].bytes;
            plan.jobs.push_back(j);
        }
    }
    if (plan.jobs.empty()) {
        fprintf(stderr, "plan %s has no jobs\n", plan_path.c_str());
        return 1;
    }
    // The engine's load path: pinned host copy, then one hipMemcpy per blob. Weights are
    // resident for the whole run; nothing here models a per-token H2D (devices/: H2D is
    // 76-82 GB/s and would dominate if a token had to wait for it).
    for (auto& w : plan.weights) upload(w, mode == "burst" ? tile : 1);
    for (auto& j : plan.jobs) CK(hipMalloc(&j.dout, (size_t)j.tokens * j.rows * 2));

    double wbytes = 0;
    for (const auto& w : plan.weights) wbytes += w.bytes;
    printf("plan %s | jobs=%zu distinct weights=%zu (%.1f MiB) | warpSize=%d CUs=%d"
           " | rows/block=%d | x=%s | tile=%d | kernel=%s tokens=%d\n",
           plan_path.c_str(), plan.jobs.size(), plan.weights.size(), wbytes / 1048576.0,
           ws, ncu, rows_per_block, x_packed ? "bf16 in shared" : "f32 in shared", tile,
           kernel.c_str(), tokens);
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    auto launch_job = [&](Job& j, int rep, int rpb) {
        const Weight& w = plan.weights[j.widx];
        if (w.rows != j.rows) {
            fprintf(stderr, "%s: plan says %d rows, weight files have %d\n", j.name.c_str(),
                    j.rows, w.rows);
            exit(1);
        }
        const int threads = rpb * ws;
        const int blocks = (j.rows + rpb - 1) / rpb;
        if (threads > prop.maxThreadsPerBlock) {
            fprintf(stderr, "rows/block %d -> %d threads > max %d\n", rpb, threads,
                    prop.maxThreadsPerBlock);
            exit(1);
        }
        const void* xp = x_packed ? (const void*)plan.xdev_bf[j.xidx]
                                  : (const void*)plan.xdev[j.xidx];
        hip_bfloat16* W = w.rep[(size_t)rep % w.rep.size()];
        // grid.x is the row blocks, exactly as the k=1 form; grid.y is the token tiles, and is 1
        // whenever there is one token or one tile, which is why the decode shape is unchanged.
        const int tpb = j.tpb;
        const dim3 grid_t((unsigned)blocks, (unsigned)((j.tokens + tpb - 1) / tpb));
        if (kernel == "k1") {
            if (j.tokens != 1) {
                fprintf(stderr, "%s: --kernel k1 is the one-row-per-launch form; drop --tokens\n",
                        j.name.c_str());
                exit(1);
            }
            if (j.in == 1024) {
                if (x_packed) proj_gemv<1024, true><<<blocks, threads>>>(W, xp, j.dout, j.rows,
                                                                         rpb, neg);
                else proj_gemv<1024, false><<<blocks, threads>>>(W, xp, j.dout, j.rows, rpb, neg);
            } else {
                if (x_packed) proj_gemv<2048, true><<<blocks, threads>>>(W, xp, j.dout, j.rows,
                                                                         rpb, neg);
                else proj_gemv<2048, false><<<blocks, threads>>>(W, xp, j.dout, j.rows, rpb, neg);
            }
            CK(hipGetLastError());
            return;
        }
        // The form the engine runs, at every width. TPB=1 with grid.y=1 is the k=1 arithmetic.
        // TPB is a template argument, so only the widths the launcher actually picks are instantiated:
        // 1 (the decode shape) plus the tile width for that input width. Anything else is refused
        // rather than silently mapped onto a different width than the one asked for.
        if (j.in == 1024) {
            if (tpb == 1) launch_tiles<1024, 1>(grid_t, threads, W, xp, j.dout, j, rpb, neg, x_packed);
            else if (tpb == 4) launch_tiles<1024, 4>(grid_t, threads, W, xp, j.dout, j, rpb, neg, x_packed);
            else {
                fprintf(stderr, "--tpb %d is not instantiated at in=1024 (1 or 4)\n", tpb);
                exit(1);
            }
        } else {
            if (tpb == 1) launch_tiles<2048, 1>(grid_t, threads, W, xp, j.dout, j, rpb, neg, x_packed);
            else if (tpb == 2) launch_tiles<2048, 2>(grid_t, threads, W, xp, j.dout, j, rpb, neg, x_packed);
            else {
                fprintf(stderr, "--tpb %d is not instantiated at in=2048 (1 or 2)\n", tpb);
                exit(1);
            }
        }
        CK(hipGetLastError());
    };

    auto write_outputs = [&]() {
        for (auto& j : plan.jobs) {
            if (j.outfile.empty()) continue;
            const size_t n = (size_t)j.tokens * j.rows;      // [token][row], the tile layout
            std::vector<hip_bfloat16> h(n);
            CK(hipMemcpy(h.data(), j.dout, n * 2, hipMemcpyDeviceToHost));
            std::ofstream of(j.outfile, std::ios::binary);
            of.write((const char*)h.data(), (std::streamsize)(n * 2));
        }
    };

    if (mode == "once") {
        int rep = 0;
        for (auto& j : plan.jobs) launch_job(j, rep++, rows_per_block);
        CK(hipDeviceSynchronize());
        write_outputs();
        printf("RESULT mode=once jobs=%zu neg=%s rows_per_block=%d x_packed=%d\n",
               plan.jobs.size(), negname.c_str(), rows_per_block, (int)x_packed);
        return 0;
    }

    std::vector<int> cfgs;
    for (const auto& s : rows_list.empty() ? std::vector<std::string>{std::to_string(
             rows_per_block)} : split(rows_list, ','))
        cfgs.push_back(atoi(s.c_str()));

    double plan_bytes = 0;
    for (const auto& j : plan.jobs) plan_bytes += (double)j.bytes();

    for (int rpb : cfgs) {
        // Burst: submit the whole plan `iters` times, one sync at the end, wall-clock timed.
        // hipEventElapsedTime returns exactly 0 below ~1 ms on this iGPU (trap in
        // prototype/README.md), and a burst also keeps the host's ~1.85 us/launch submit cost
        // off the measured time. Repeating the whole plan — not one kernel — is what keeps the
        // working set cold: a real token touches each of these weights exactly once.
        auto burst = [&](int n) {
            hipEvent_t s, e;
            CK(hipEventCreate(&s));
            CK(hipEventCreate(&e));
            CK(hipEventRecord(s));
            const auto t0 = std::chrono::steady_clock::now();
            size_t counter = 0;
            for (int i = 0; i < n; ++i)
                for (auto& j : plan.jobs) launch_job(j, (int)(counter++), rpb);
            CK(hipEventRecord(e));
            CK(hipDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            float ev = 0.f;
            CK(hipEventElapsedTime(&ev, s, e));
            CK(hipEventDestroy(s));
            CK(hipEventDestroy(e));
            double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
            // Wall-clock is the number. The event pair is printed alongside as a cross-check
            // only: on this iGPU an event window around short work can read ~0 (trap #7), and a
            // min() of the two lets a broken timer masquerade as an infinitely fast kernel -
            // which is exactly how one shape here once reported 25433 GB/s.
            return std::pair<double, double>(wall, (double)ev);
        };
        // Shared bytes per block for the widest case (in=2048 at its tile width) and the occupancy the
        // instantiated kernel gets. With the tile form this is TPB*IN*(2 or 4) bytes, which is exactly
        // the quantity that bounds TPB, so it is printed rather than assumed.
        const int occ_x = tile_width(2048) * 2048 * (x_packed ? 2 : 4);
        int occ = 0;
        CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, x_packed ? (const void*)&proj_gemv_tokens<2048, 2, true>
                           : (const void*)&proj_gemv_tokens<2048, 2, false>, rpb * ws, 0));
        burst(1);                                      // warm: fresh pages run at ~35 GB/s
        const int reps = std::max(1, iters);
        const std::pair<double, double> t = burst(reps);
        const double ms_plan = t.first / reps;
        const double ev_plan = t.second / reps;
        const double ms_launch = ms_plan / (double)plan.jobs.size();
        const double gbs = plan_bytes / (ms_plan / 1e3) / 1e9;
        const char* evnote = (ev_plan <= 0.0 || ev_plan > 1.5 * ms_plan || ev_plan < 0.67 * ms_plan)
                                 ? " (events disagree: treat as unmeasured)" : "";
        printf("  rows/block %-3d %8.4f ms/plan %6.1f GB/s (%3.0f%% of the %.0f GB/s probe)"
               " | events %.4f ms/plan%s | %.1f MiB/plan, shared %d B/block, %d blocks/CU"
               " occupancy\n",
               rpb, ms_plan, gbs, 100.0 * gbs / PROBE_GBS, PROBE_GBS, ev_plan, evnote,
               plan_bytes / 1048576.0, occ_x, occ);
        // One line per distinct (rows, in) in the plan: blocks and waves are the block-count
        // argument the ledger's §5 item 5 asks about (20 CUs, not the 40 rocminfo prints).
        for (size_t i = 0; i < plan.jobs.size(); ++i) {
            bool seen = false;
            for (size_t k = 0; k < i; ++k)
                if (plan.jobs[k].rows == plan.jobs[i].rows && plan.jobs[k].in == plan.jobs[i].in)
                    seen = true;
            if (seen) continue;
            const Job& j = plan.jobs[i];
            const int blocks = (j.rows + rpb - 1) / rpb;
            printf("    shape [%6d,%4d] blocks=%5d waves=%5d waves/CU=%5.1f  %.2f MiB/launch"
                   "  %.1f us/launch at this rate\n",
                   j.rows, j.in, blocks, blocks * rpb, (double)blocks * rpb / ncu,
                   plan.weights[j.widx].bytes / 1048576.0, ms_launch * 1e3);
        }
        printf("RESULT mode=burst rpb=%d jobs=%zu launches=%zu ms_per_plan=%.6f"
               " ms_per_launch=%.6f gbs=%.2f probe_pct=%.2f x_packed=%d tile=%d neg=%s\n",
               rpb, plan.jobs.size(), plan.jobs.size(), ms_plan, ms_launch, gbs,
               100.0 * gbs / PROBE_GBS, (int)x_packed, tile, negname.c_str());
    }
    if (write_out) {                     // replica 0 after timing: the timed config's own output
        for (auto& j : plan.jobs) launch_job(j, 0, rows_per_block);
        CK(hipDeviceSynchronize());
        write_outputs();
    }
    return 0;
}
