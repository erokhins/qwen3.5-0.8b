/*
 * prototype/mlp_gemv/mlp_gemv.hip.cpp — the second GEMV family: the §03.3 MLP, k = 1.
 *
 * Op (S03.3, decode step, one token):
 *     act  = bf16( silu(bf16(x . Wg^T)) * bf16(x . Wu^T) )      [3584]
 *     out  = bf16(act . Wd^T)                                   [1024]
 * The MLP is 0.5285 GB/token = 33 % of the decode bytes (§01.7, bench/performance-status.md
 * §2), the largest single family left after lm_head, and it is the first place the engine has
 * to fuse an epilogue: the launch floor here is 1.85 us/launch (tools/launch_overhead_hip.cpp)
 * and an unfused MLP pays 3 launches per layer instead of 2, so the shape of this kernel is
 * the shape of the four remaining GEMV families.
 *
 * Same skeleton as lm_head_gemv (one wavefront per output row, x staged in shared, F32
 * accumulation), parameterized on the input width instead of hardcoding H = 1024, because the
 * MLP has two different widths (gate/up: in 1024, down: in 3584) and GDN/attention
 * projections bring more. Templated on IN so shared memory is exactly the size used.
 *
 * Rounding policy, all three measured in ulp_probe.py against the pinned §03.3 operands:
 *   - each dot is F32 and rounds once to BF16 (the reference is a bf16 torch matmul);
 *   - silu is applied to the BF16 gate and rounds again, and the gate product rounds a
 *     third time. Computing it all in F32 instead — the literal A-07 reading — moves 27 %
 *     of `act` by 1 ULP, so the double rounding is not optional;
 *   - a reordered F32 sum costs ~1 element in 3584 by exactly 1 ULP, which is the floor the
 *     gate below is sized against.
 *
 * Two differences from lm_head_gemv on purpose:
 *   - it uses `warpSize` rather than a literal 32, so it is correct on a wave64 target too
 *     (the host refuses to run if warpSize is not 32 or 64);
 *   - it writes BF16 outputs, because that is what the reference produces and what the next
 *     op consumes; no fp32 logits array exists here.
 *
 * Build: prototype/mlp_gemv/build.sh
 * Run:   ./mlp_gemv --mode gateup --x x.bin --wg Wg.bin --wu Wu.bin --out act.bin [--pre pre.bin]
 *        ./mlp_gemv --mode down   --x act.bin --wd Wd.bin --out out.bin
 *        ./mlp_gemv --mode mlp    --x x.bin --act act.bin --wg Wg.bin --wu Wu.bin --wd Wd.bin \
 *                                 --out out.bin --actout act_out.bin
 *            the two kernels ALTERNATED in one burst, which is what a token does and what the
 *            cold bandwidth check has to measure (see the accounting note at the burst).
 * Validate: prototype/mlp_gemv/run.py
 *
 * Files are raw little-endian: x.bin BF16 (in = file size / 2, must be 1024 or 3584),
 * weights row-major BF16 [out x in], outputs BF16.
 */
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(x)                                                                    \
    do {                                                                         \
        hipError_t e_ = (x);                                                     \
        if (e_ != hipSuccess) {                                                  \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),    \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define HID 1024      // hidden size (§00)
#define FF 3584       // intermediate size (§00)
#define ROWS_PER_BLOCK 8
#define FULL_MASK 0xffffffffffffffffull   // HIP takes the shuffle mask as 64-bit

// Injected bugs, so run.py --negative can prove the gate rejects them (the G-neg convention
// in speccheck/gates/run_negative.py: a gate that cannot fail verifies nothing).
#define NEG_NONE 0
#define NEG_HALF 1     // accumulate half of K: a wrong reduction length
#define NEG_TRUNC 2    // round toward zero instead of round-to-nearest-even
#define NEG_F32ACT 3   // do the SwiGLU entirely in F32 (the literal A-07 reading)
#define NEG_SWAPROW 4  // write each finished row into its neighbour's slot (wrong output index)
#define NEG_TAIL 5     // drop the LAST word of K: an off-by-one in the reduction bound
#define NEG_PRESWAP 6  // stage gate into the up slot and vice versa in the --pre dump, leaving
                       // `act` correct — the bug class the F1/F2 split exists to detect

// Round-to-nearest-even, same as torch's cast to bf16. The product of two bf16 values must
// round the same way the reference's bf16 multiply does, or `act` drifts by 1 ULP.
// HIP has no `__float2bfloat16_rn` (CUDA-ism, see prototype/README.md): the bf16
// constructor is the RNE conversion, and the gate in run.py would notice truncation
// immediately (it would move ~half of `act` by 1 ULP instead of ~0).
__device__ __forceinline__ float bf(hip_bfloat16 b) { return float(b); }
__device__ __forceinline__ hip_bfloat16 fb(float f) { return hip_bfloat16(f); }

// Truncating cast, for NEG_TRUNC only: dropping the low 16 bits of the fp32 payload rounds
// toward zero, which is the classic way to get a bf16 conversion wrong.
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

// Full-wavefront reduction helper: the shift ladder starts at warpSize/2, so the same
// code is correct on a wave32 and a wave64 target (the host checks which one this is).
__device__ __forceinline__ float reduce_row(float acc) {
    const int ws = warpSize;
    for (int o = ws / 2; o; o >>= 1) acc += __shfl_down_sync(FULL_MASK, acc, o);
    return acc;
}

// One wavefront per intermediate row: both projections and the SwiGLU epilogue, so the
// gate/up pair costs one launch and the 3584-element intermediates never reach memory
// unless asked for (--pre, which exists so the gate can compare them).
template <int IN>
__global__ void mlp_gateup(const hip_bfloat16* __restrict__ Wg,
                           const hip_bfloat16* __restrict__ Wu, const float* __restrict__ x,
                           hip_bfloat16* __restrict__ act, hip_bfloat16* __restrict__ pre,
                           int rows, int rpb, int neg) {
    __shared__ float xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    // Two independent streams: the compiler interleaves the loads, which is what keeps
    // enough of them in flight to sit on the roofline at only 8 rows per block.
    const uint4* wg = reinterpret_cast<const uint4*>(Wg + (size_t)row * IN);
    const uint4* wu = reinterpret_cast<const uint4*>(Wu + (size_t)row * IN);
    // NEG_HALF: half of K. NEG_TAIL: the last word of K (8 of 1024 terms), which is the shape an
    // off-by-one in `v < words` actually takes — small enough that only a gate with a real
    // per-element rule notices, unlike NEG_HALF.
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
    float ag = 0.f, au = 0.f;
#pragma unroll
    for (int v = lane; v < words; v += ws) {
        uint4 ug = wg[v], uu = wu[v];
        const hip_bfloat16* pg = reinterpret_cast<const hip_bfloat16*>(&ug);
        const hip_bfloat16* pu = reinterpret_cast<const hip_bfloat16*>(&uu);
        const float* xp = &xs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            ag += bf(pg[j]) * xp[j];
            au += bf(pu[j]) * xp[j];
        }
    }
    ag = reduce_row(ag);
    au = reduce_row(au);
    if (lane != 0) return;
    hip_bfloat16 g = rnd(ag, neg), u = rnd(au, neg);
    // NEG_SWAPROW: store the finished row one slot over. rows is a multiple of 3584/1024, so the
    // neighbour always exists; what matters is that every other row is now somebody else's answer.
    const int orow = neg == NEG_SWAPROW ? (row ^ 1) : row;
    // silu in F32 on the rounded gate, rounded back to BF16, THEN multiplied with the rounded
    // `up` and rounded again — three roundings, because HF's F.silu returns BF16 and the
    // product is a BF16 tensor op. Doing the whole thing in F32 (NEG_F32ACT, and the literal
    // reading of A-07) differs from the reference in 27 % of `act` elements; ulp_probe.py
    // measures that and that this three-rounding form is bit-equal to the pin. HF's order is
    // x * (1/(1+exp(-x))); writing it as a division is algebraically identical but cost the
    // first version a 2 ULP outlier at a large element (a tie in the product's BF16 rounding).
    if (neg == NEG_F32ACT) {              // the wrong reading, for the negative test only
        float a = (ag * (1.0f / (1.0f + expf(-ag)))) * au;
        act[orow] = rnd(a, neg);
    } else {
        float gf = bf(g);
        float sg = bf(fb(gf * (1.0f / (1.0f + expf(-gf)))));
        act[orow] = rnd(sg * bf(u), neg);
    }
    if (pre) {
        // NEG_PRESWAP: the two dots and the epilogue are all correct — only the staging of the
        // raw dumps swapped, so `act` matches the reference and `gate_pre` is `up_pre`'s value.
        // A gate that judged `act` alone would sign this off; the F1/F2 split is what catches it.
        const bool sw = neg == NEG_PRESWAP;
        pre[orow] = sw ? u : g;
        pre[rows + orow] = sw ? g : u;
    }
}

template <int IN>
__global__ void mlp_down(const hip_bfloat16* __restrict__ W, const float* __restrict__ x,
                         hip_bfloat16* __restrict__ out, int rows, int rpb, int neg) {
    __shared__ float xs[IN];
    for (int i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int ws = warpSize;
    const int lane = threadIdx.x % ws;
    const int row = blockIdx.x * rpb + threadIdx.x / ws;
    if (row >= rows) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row * IN);
    float acc = 0.f;
    const int words = neg == NEG_HALF ? IN / 16 : (neg == NEG_TAIL ? IN / 8 - 1 : IN / 8);
#pragma unroll
    for (int v = lane; v < words; v += ws) {
        uint4 u = wrow[v];
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(&u);
        const float* xp = &xs[v * 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += bf(p[j]) * xp[j];
    }
    acc = reduce_row(acc);
    if (lane == 0) out[neg == NEG_SWAPROW ? (row ^ 1) : row] = rnd(acc, neg);
}

static std::vector<char> slurp(const std::string& p, size_t expect) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    std::vector<char> b(expect);
    f.read(b.data(), expect);
    if ((size_t)f.gcount() != expect) {
        fprintf(stderr, "%s: wanted %zu bytes, got %lld\n", p.c_str(), expect,
                (long long)f.gcount());
        exit(1);
    }
    return b;
}

static size_t filesize(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    return (size_t)f.tellg();
}

int main(int argc, char** argv) {
    std::string mode = "gateup", xpath, wgpath, wupath, wdpath, opath, ppath, apath, aopath,
            negname = "none";
    int iters = 20;
    int rows_per_block = ROWS_PER_BLOCK;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--mode") mode = next();
        else if (a == "--x") xpath = next();
        else if (a == "--wg") wgpath = next();
        else if (a == "--wu") wupath = next();
        else if (a == "--wd") wdpath = next();
        else if (a == "--out") opath = next();
        else if (a == "--pre") ppath = next();
        else if (a == "--act") apath = next();
        else if (a == "--actout") aopath = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--rows") rows_per_block = atoi(next().c_str());
        else if (a == "--neg") negname = next();
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    const int ws = prop.warpSize;
    if (ws != 32 && ws != 64) {
        fprintf(stderr, "warpSize %d unsupported by the row-dot reduction\n", ws);
        return 1;
    }
    if (rows_per_block * ws > prop.maxThreadsPerBlock) {
        fprintf(stderr, "rows/block %d x warpSize %d exceeds maxThreadsPerBlock %d\n",
                rows_per_block, ws, prop.maxThreadsPerBlock);
        return 1;
    }

    const bool gateup = mode == "gateup";
    const bool pair = mode == "mlp";   // gateup and down ALTERNATED, one layer of a real step
    const size_t xbytes = filesize(xpath);
    const int in = (int)(xbytes / 2);
    const size_t wbytes = (gateup || pair) ? filesize(wgpath) : filesize(wdpath);
    const int rows = (int)(wbytes / (in * 2));
    // For --mode mlp the down kernel is fed the reference act (in = 3584), exactly as F3 does,
    // so the pair measures the two dots and not whatever the epilogue just produced.
    size_t abytes = 0;
    int in2 = 0, rows2 = 0;
    if (pair) {
        abytes = filesize(apath);
        in2 = (int)(abytes / 2);
        rows2 = (int)(filesize(wdpath) / (in2 * 2));
    }
    printf("mode %s | x in=%d | rows=%d | warpSize=%d | rows/block=%d | weights %.2f MiB%s\n",
           mode.c_str(), in, rows, ws, rows_per_block, wbytes / 1048576.0,
           pair ? " (+down weights, pair burst)" : "");

    // x arrives as BF16 (the reference's own operands are BF16-valued). It is staged as F32
    // because every bf16 value is exact in fp32, so the products are identical and the
    // inner loop never converts x.
    auto xraw = slurp(xpath, xbytes);
    std::vector<float> xhost(in);
    {
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(xraw.data());
        for (int i = 0; i < in; ++i) xhost[i] = float(p[i]);
    }
    std::vector<float> ahost;
    if (pair) {
        auto araw = slurp(apath, abytes);
        ahost.resize(in2);
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(araw.data());
        for (int i = 0; i < in2; ++i) ahost[i] = float(p[i]);
    }

    // The engine's load path: pinned host copy, then one hipMemcpy per blob
    // (devices/strix-halo-unified-memory.md: hipMalloc + explicit copy, never managed).
    char* htmp = nullptr;
    CK(hipHostMalloc(&htmp, wbytes));
    hip_bfloat16 *dw = nullptr, *dw2 = nullptr, *dw3 = nullptr, *dout = nullptr, *dpre = nullptr,
                 *dact = nullptr;
    float* dx = nullptr;
    float* dx2 = nullptr;
    auto upload = [&](const std::string& path, hip_bfloat16** dst) {
        std::ifstream f(path, std::ios::binary);
        f.read(htmp, wbytes);
        CK(hipMalloc(dst, wbytes));
        CK(hipMemcpy(*dst, htmp, wbytes, hipMemcpyHostToDevice));
    };
    if (gateup || pair) {
        upload(wgpath, &dw);
        const size_t b2 = filesize(wupath);
        if (b2 != wbytes) {
            fprintf(stderr, "Wu is %zu bytes, Wg is %zu\n", b2, wbytes);
            return 1;
        }
        std::ifstream f(wupath, std::ios::binary);
        f.read(htmp, wbytes);
        CK(hipMalloc(&dw2, wbytes));
        CK(hipMemcpy(dw2, htmp, wbytes, hipMemcpyHostToDevice));
    } else {
        upload(wdpath, &dw);
    }
    if (pair) {
        const size_t b3 = filesize(wdpath);
        CK(hipHostFree(htmp));
        CK(hipHostMalloc(&htmp, b3));
        std::ifstream f(wdpath, std::ios::binary);
        f.read(htmp, b3);
        CK(hipMalloc(&dw3, b3));
        CK(hipMemcpy(dw3, htmp, b3, hipMemcpyHostToDevice));
        CK(hipMalloc(&dx2, (size_t)in2 * sizeof(float)));
        CK(hipMemcpy(dx2, ahost.data(), (size_t)in2 * sizeof(float), hipMemcpyHostToDevice));
        CK(hipMalloc(&dact, (size_t)rows * 2));
    }
    CK(hipMalloc(&dx, (size_t)in * sizeof(float)));
    CK(hipMemcpy(dx, xhost.data(), (size_t)in * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&dout, (size_t)(pair ? rows2 : rows) * 2));
    if (gateup && !ppath.empty()) CK(hipMalloc(&dpre, 2 * (size_t)rows * 2));

    const int neg = negname == "none" ? NEG_NONE : negname == "half" ? NEG_HALF
                  : negname == "trunc" ? NEG_TRUNC : negname == "f32act" ? NEG_F32ACT
                  : negname == "rowswap" ? NEG_SWAPROW : negname == "tail" ? NEG_TAIL
                  : negname == "preswap" ? NEG_PRESWAP : -1;
    if (neg < 0) {
        fprintf(stderr, "--neg must be none|half|trunc|f32act|rowswap|tail|preswap, got %s\n",
                negname.c_str());
        return 1;
    }
    if (neg != NEG_NONE) printf("NEGATIVE TEST: injected bug = %s\n", negname.c_str());

    const int threads = rows_per_block * ws;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    const int blocks2 = (rows2 + rows_per_block - 1) / rows_per_block;
    auto launch = [&]() {
        if (pair) {          // one layer of a real decode step: gate/up, then down, alternated
            if (in == HID)
                mlp_gateup<HID><<<blocks, threads>>>(dw, dw2, dx, dact, nullptr, rows,
                                                     rows_per_block, neg);
            else {
                fprintf(stderr, "mlp: in=%d not implemented\n", in);
                exit(1);
            }
            if (in2 == FF)
                mlp_down<FF><<<blocks2, threads>>>(dw3, dx2, dout, rows2, rows_per_block, neg);
            else {
                fprintf(stderr, "mlp: act in=%d not implemented\n", in2);
                exit(1);
            }
            return;
        }
        if (gateup) {
            if (in == HID)
                mlp_gateup<HID><<<blocks, threads>>>(dw, dw2, dx, dout, dpre, rows,
                                                     rows_per_block, neg);
            else if (in == FF)
                mlp_gateup<FF><<<blocks, threads>>>(dw, dw2, dx, dout, dpre, rows,
                                                    rows_per_block, neg);
            else {
                fprintf(stderr, "gateup: in=%d not implemented\n", in);
                exit(1);
            }
        } else {
            if (in == HID)
                mlp_down<HID><<<blocks, threads>>>(dw, dx, dout, rows, rows_per_block, neg);
            else if (in == FF)
                mlp_down<FF><<<blocks, threads>>>(dw, dx, dout, rows, rows_per_block, neg);
            else {
                fprintf(stderr, "down: in=%d not implemented\n", in);
                exit(1);
            }
        }
    };

    launch();  // first touch faults the pages in; timing starts after this
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    // Traffic accounting: weights dominate; x is read once per block but is 2-7 KiB, and
    // the intermediates are 2 KiB-7 KiB, so they are rounding error next to 22 MiB. For the pair
    // it is the sum of both kernels' traffic, and one burst iteration is one layer.
    const size_t wdbytes = pair ? filesize(wdpath) : 0;
    const double bytes = pair
        ? (double)(2 * wbytes + wdbytes) + xbytes + abytes + (double)rows * 2.0
          + (double)rows2 * 2.0
        : (double)wbytes * (gateup ? 2.0 : 1.0) + xbytes + (double)rows * 2.0
          + (gateup && !ppath.empty() ? 2.0 * rows * 2.0 : 0.0);

    double best = 1e30;
    auto wall0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        hipEvent_t s, e;
        CK(hipEventCreate(&s));
        CK(hipEventCreate(&e));
        CK(hipEventRecord(s));
        launch();
        CK(hipGetLastError());   // a failed launch inside the loop would otherwise time as 0 ms
        CK(hipEventRecord(e));
        CK(hipEventSynchronize(e));
        float ms = 0.f;
        CK(hipEventElapsedTime(&ms, s, e));
        // hipEventElapsedTime on this iGPU returns exactly 0 below roughly a millisecond of
        // work (the burst below shows the same for 50 launches), so a zero is "not measured",
        // not "instant". Folding that 0 into a min() is how a kernel gets reported at inf GB/s.
        if (ms > 0.f) best = std::min(best, (double)ms);
        CK(hipEventDestroy(s));
        CK(hipEventDestroy(e));
    }
    auto wall1 = std::chrono::steady_clock::now();
    double wall_us = std::chrono::duration<double, std::micro>(wall1 - wall0).count() / iters;
    printf("per-launch events: %s; loop wall %.1f us/launch\n",
           best < 1e29 ? (std::to_string(best) + " ms best").c_str()
                       : "unmeasured (the event timer has ~1 ms granularity here)",
           wall_us);

    // Burst timing: submit N launches, one sync at the end, timed with the wall clock and
    // with one event pair. These kernels are 20-130 us, and on this iGPU hipEventElapsedTime
    // returns exactly 0 for a single launch that short (it is fine at millisecond scale:
    // lm_head_gemv reads 2.105 ms with the same code), so per-launch events are useless here.
    // In a burst the host submits at ~1.8 us per launch (tools/launch_overhead_hip.cpp) while
    // the GPU is the bottleneck, so wall/N is the GPU time.
    {
        const int burst = std::max(iters, 50);
        hipEvent_t bs, be;
        CK(hipEventCreate(&bs));
        CK(hipEventCreate(&be));
        CK(hipEventRecord(bs));
        auto b0 = std::chrono::steady_clock::now();
        for (int i = 0; i < burst; ++i) launch();
        CK(hipGetLastError());
        CK(hipEventRecord(be));
        CK(hipDeviceSynchronize());
        auto b1 = std::chrono::steady_clock::now();
        float ev = 0.f;
        CK(hipEventElapsedTime(&ev, bs, be));
        CK(hipEventDestroy(bs));
        CK(hipEventDestroy(be));
        double wall_ms = std::chrono::duration<double, std::milli>(b1 - b0).count() / burst;
        double ev_ms = ev > 0.f ? ev / burst : 0.0;
        printf("burst %d: wall %.4f ms/launch -> %.1f GB/s | events %.4f ms/launch -> %.1f GB/s"
               "  (%.0f%% / %.0f%% of the 237 GB/s probe)\n", burst, wall_ms,
               bytes / (wall_ms / 1e3) / 1e9, ev_ms, ev_ms > 0 ? bytes / (ev_ms / 1e3) / 1e9 : 0.0,
               bytes / (wall_ms / 1e3) / 237e9 * 100.0,
               ev_ms > 0 ? bytes / (ev_ms / 1e3) / 237e9 * 100.0 : 0.0);
        if (ev_ms > 0 && ev_ms < best) best = ev_ms;
        if (wall_ms < best) best = wall_ms;
    }
    double gbs = bytes / (best / 1e3) / 1e9;
    printf("kernel %.4f ms -> %.1f GB/s (%.1f%% of the 237 GB/s probe)\n", best, gbs,
           gbs / 237.0 * 100.0);
    printf("RESULT mode=%s ms=%.4f gbs=%.1f rows=%d in=%d rows_per_block=%d\n", mode.c_str(),
           best, gbs, rows, in, rows_per_block);

    std::vector<hip_bfloat16> hout(pair ? rows2 : rows);
    CK(hipMemcpy(hout.data(), dout, (size_t)hout.size() * 2, hipMemcpyDeviceToHost));
    {
        FILE* f = fopen(opath.c_str(), "wb");
        fwrite(hout.data(), 2, hout.size(), f);
        fclose(f);
    }
    if (pair && !aopath.empty()) {
        std::vector<hip_bfloat16> hact(rows);
        CK(hipMemcpy(hact.data(), dact, (size_t)rows * 2, hipMemcpyDeviceToHost));
        FILE* f = fopen(aopath.c_str(), "wb");
        fwrite(hact.data(), 2, rows, f);
        fclose(f);
    }
    if (gateup && dpre) {
        std::vector<hip_bfloat16> hpre(2 * (size_t)rows);
        CK(hipMemcpy(hpre.data(), dpre, 2 * (size_t)rows * 2, hipMemcpyDeviceToHost));
        FILE* f = fopen(ppath.c_str(), "wb");
        fwrite(hpre.data(), 2, hpre.size(), f);
        fclose(f);
    }
    CK(hipFree(dw));
    if (dw2) CK(hipFree(dw2));
    if (dw3) CK(hipFree(dw3));
    if (dx2) CK(hipFree(dx2));
    if (dact) CK(hipFree(dact));
    CK(hipFree(dx));
    CK(hipFree(dout));
    if (dpre) CK(hipFree(dpre));
    CK(hipHostFree(htmp));
    return 0;
}
