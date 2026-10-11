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
 *        ./proj_gemv --mode burst --plan PLAN [--rows N] [--iters N] [--tile N] [--write-out]
 *        ./proj_gemv --mode burst --plan PLAN --rows-list 2,4,8,16,32 [--tile N]
 * Validate: prototype/proj_gemv/run.py
 *
 * PLAN is a text file, one launch per line:
 *     NAME ROWS IN XFILE WFILE[,WFILE...] [OUTFILE]
 * XFILE is raw little-endian BF16 with IN elements; WFILEs are row-major BF16 [rows_i, IN]
 * concatenated in the order given (ROWS must equal the sum); OUTFILE, if present, receives the
 * BF16 output of ROWS elements — always in `once` mode, in `burst` mode only with --write-out.
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
#define PROBE_GBS 237.0                   // devices/strix-halo-memory-bandwidth.md read probe

enum Neg {
    NEG_NONE = 0,
    NEG_TAIL,        // one word (8 bf16) of K dropped: the subtle reduction-length bug
    NEG_HALF,        // half of K accumulated
    NEG_TRUNC,       // truncate instead of round-to-nearest-even
    NEG_ROWSWAP,     // write out[row] to the neighbouring row
    NEG_TRANSPOSE,   // read W as if it were stored [in, rows]
    NEG_QHALF,       // q_proj rows in the wrong interleave (§05.1)
    NEG_KVS,         // in_proj_qkv k/v sections swapped (§04.1)
};

__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
// The staged x is either BF16 (packed) or F32 (widened); both convert to F32 the same way.
template <typename T> __device__ __forceinline__ float tof(T v) { return float(v); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }

// Truncating cast, NEG_TRUNC only: dropping the low 16 bits of the fp32 payload rounds toward
// zero, the classic way to get a BF16 conversion wrong.
__device__ __forceinline__ hip_bfloat16 fb_trunc(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    unsigned short top = (unsigned short)(u >> 16);
    hip_bfloat16 b;
    memcpy(&b, &top, 2);
    return b;
}

__device__ __forceinline__ hip_bfloat16 rnd(float f, int neg) {
    return neg == NEG_TRUNC ? fb_trunc(f) : fb(f);
}

// Full-wavefront reduction: the shift ladder starts at warpSize/2, so one body is correct on a
// wave32 (this part) and a wave64 target.
__device__ __forceinline__ float reduce_row(float acc) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    return acc;
}

// §05.1: q_proj's 4096 rows are [8 heads][q 256 | gate 256]. §10.5 measured that reading them as
// [all q 2048 | all gate 2048] scores cos -0.14 against the pinned q_proj output, so the layout
// cannot be gotten quietly wrong. NEG_QHALF *is* that reading: every dot is still computed
// correctly, its result is just written where the wrong reading expects it — the same set of
// wrong values an engine that de-interleaves wrongly would hand to §05.2.
__device__ __forceinline__ int qhalf_perm(int row) {
    const int H = 8, DH = 256;
    if (row >= 2 * H * DH) return row;              // past the q_proj region: k/v sections stay
    const int h = row / (2 * DH), j = row % (2 * DH);
    return j < DH ? h * DH + j : H * DH + h * DH + (j - DH);
}

// §04.1: raw = [q 2048 | k 2048 | v 2048]. NEG_KVS swaps the k and v sections of the output.
// Rows past 6144 — the z/b/a sections of a packed in_proj launch — are untouched, which is what
// makes this negative's must-not-catch claim testable instead of decorative.
__device__ __forceinline__ int kv_swap_perm(int row) {
    const int C = 2048;
    if (row >= 3 * C) return row;
    if (row < C) return row;
    return row < 2 * C ? row + C : row - C;
}

// Where this row's result goes. `rows` clamps the neighbour swap so an odd row count cannot
// write past the end of the output buffer.
__device__ __forceinline__ int perm_out_row(int row, int rows, int neg) {
    if (neg == NEG_ROWSWAP) return (row % 2 == 0) ? (row + 1 < rows ? row + 1 : row) : row - 1;
    if (neg == NEG_QHALF) return qhalf_perm(row);
    if (neg == NEG_KVS) return kv_swap_perm(row);
    return row;
}

// One wavefront per output row. PACK_X keeps the staged input as BF16 (2 KiB at in=1024) instead
// of unpacking it to F32 (4 KiB): identical products — every BF16 value is exact in F32 — half
// the shared-memory traffic and half the shared memory per block, which is what limits blocks/CU
// for the 2048-wide shapes. run.py's sweep measures both instead of guessing.
template <int IN, bool PACK_X>
__global__ void proj_gemv(const hip_bfloat16* __restrict__ W, const void* __restrict__ x,
                          hip_bfloat16* __restrict__ out, int rows, int rpb, int neg) {
    __shared__ typename std::conditional<PACK_X, hip_bfloat16, float>::type xs[IN];
    if (PACK_X) {
        const hip_bfloat16* src = (const hip_bfloat16*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = src[i];
    } else {
        const float* src = (const float*)x;
        for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = src[i];
    }
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    float acc = 0.f;
    if (neg == NEG_TRANSPOSE) {
        // Read the same bytes as [in, rows]: in-bounds for any shape (k*rows + row < in*rows)
        // and a completely different set of products.
        for (int k = lane; k < IN; k += ws) acc += bf(W[(size_t)k * rows + row]) * tof(xs[k]);
    } else {
        const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
        int words = IN / 8;
        if (neg == NEG_TAIL) words -= 1;
        else if (neg == NEG_HALF) words /= 2;
#pragma unroll
        for (int v = lane; v < words; v += ws) {
            uint4 u = wrow[v];
            const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += bf(p[j]) * tof(xs[v * 8 + j]);
        }
    }
    acc = reduce_row(acc);
    if (lane == 0) out[perm_out_row(row, rows, neg)] = rnd(acc, neg);
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
    int rows = 0, in = 0;
    size_t wbytes = 0, xbytes = 0;
    hip_bfloat16* dout = nullptr;
    size_t bytes() const { return wbytes + xbytes + (size_t)rows * 2; }
};

struct Plan {
    std::vector<Weight> weights;
    std::vector<float*> xdev;              // widened F32 x, one per distinct x file
    std::vector<hip_bfloat16*> xdev_bf;    // the same x left as BF16
    std::vector<std::string> xpaths;
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
        const std::vector<char> raw = slurp(path);
        if (raw.size() != (size_t)in * 2) {
            fprintf(stderr, "%s: %zu bytes, expected %d bf16 elements\n", path.c_str(),
                    raw.size(), in);
            exit(1);
        }
        std::vector<float> host(in);
        const hip_bfloat16* p = (const hip_bfloat16*)raw.data();
        for (int i = 0; i < in; ++i) host[i] = float(p[i]);
        float* d = nullptr;
        hip_bfloat16* db = nullptr;
        CK(hipMalloc(&d, (size_t)in * sizeof(float)));
        CK(hipMemcpy(d, host.data(), (size_t)in * sizeof(float), hipMemcpyHostToDevice));
        CK(hipMalloc(&db, (size_t)in * 2));
        CK(hipMemcpy(db, raw.data(), (size_t)in * 2, hipMemcpyHostToDevice));
        xpaths.push_back(path);
        xdev.push_back(d);
        xdev_bf.push_back(db);
        return (int)xpaths.size() - 1;
    }
};

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
    std::string mode = "once", plan_path, negname = "none", rows_list;
    int iters = 30, rows_per_block = 8, tile = 1;
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
            j.xbytes = (size_t)j.in * 2;
            // Traffic accounting: every launch streams its whole weight blob once, plus x in and
            // the result out. `tile` replicas are cache-defeats, not extra work, so they are NOT
            // counted per launch — the reported GB/s is the same bytes a real token moves.
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
    for (auto& j : plan.jobs) CK(hipMalloc(&j.dout, (size_t)j.rows * 2));

    double wbytes = 0;
    for (const auto& w : plan.weights) wbytes += w.bytes;
    printf("plan %s | jobs=%zu distinct weights=%zu (%.1f MiB) | warpSize=%d CUs=%d"
           " | rows/block=%d | x=%s | tile=%d\n",
           plan_path.c_str(), plan.jobs.size(), plan.weights.size(), wbytes / 1048576.0,
           ws, ncu, rows_per_block, x_packed ? "bf16 in shared" : "f32 in shared", tile);
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
    };

    auto write_outputs = [&]() {
        for (auto& j : plan.jobs) {
            if (j.outfile.empty()) continue;
            std::vector<hip_bfloat16> h(j.rows);
            CK(hipMemcpy(h.data(), j.dout, (size_t)j.rows * 2, hipMemcpyDeviceToHost));
            std::ofstream of(j.outfile, std::ios::binary);
            of.write((const char*)h.data(), (std::streamsize)j.rows * 2);
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
        const int occ_x = x_packed ? 2048 : 4096;      // shared bytes for in=2048, widest case
        int occ = 0;
        CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, x_packed ? (const void*)&proj_gemv<2048, true>
                           : (const void*)&proj_gemv<2048, false>, rpb * ws, 0));
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
