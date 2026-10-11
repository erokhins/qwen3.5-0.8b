/*
 * prototype/lm_head_gemv/lm_head_gemv.hip.cpp — first real HIP kernel of the engine.
 *
 * Op: logits[v] = sum_k h[k] * W[v][k] for v in [0, V), with W the tied embedding/lm_head
 * table (bf16, [V=248320, H=1024], 508.6 MB) and h one normed hidden row. §01.7 says this
 * table is ~33 % of all decode bytes, and §07.4 flags a skinny argmax GEMV as a
 * first-class optimization rather than a micro one, so this is the right first kernel to
 * find out whether the HIP path can actually sit on the 237 GB/s the probe says is
 * available.
 *
 * Shapes are fixed by the model (H = 1024), which allows the simple thing:
 *   - one warp per vocab row. 32 lanes x 8 bf16 = 256 elements per step, so H = 1024 is
 *     four steps of 16 B loads; a warp's loads are 512 contiguous bytes, i.e. coalesced.
 *   - h (2 KiB bf16 / 4 KiB fp32) staged once into shared memory and read broadcast.
 *   - fp32 accumulation, bf16 inputs: matches what speccheck does and what §09.4 measured.
 *   - two output modes: `full` writes all V logits (1 MB, fp32); `fused` skips the logits
 *     entirely and folds the argmax into the kernel with one atomicMax per row, keyed by a
 *     sign-flipped float so integer max == float max. Greedy decoding never needs the
 *     logits themselves (A-17), so this is the mode a decode loop would use.
 *
 * Build: prototype/lm_head_gemv/build.sh
 * Run:   ./lm_head_gemv --w W.bin --h h.bin --out out.bin [--mode full|fused] [--iters N]
 * Validate: prototype/lm_head_gemv/run.py (does all of the above and gates the result).
 *
 * W.bin is raw row-major bf16 (V*H*2 bytes); h.bin is fp32 H floats; out.bin is fp32 V
 * floats (mode full) or two floats (mode fused: value, index).
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
#include "lm_head_gemv_kernels.h"

#define CK(x)                                                                    \
    do {                                                                         \
        hipError_t e_ = (x);                                                     \
        if (e_ != hipSuccess) {                                                  \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),    \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)


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

int main(int argc, char** argv) {
    std::string wpath, hpath, opath, mode = "full";
    int iters = 20;
    int rows_per_block = ROWS_PER_BLOCK;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--w") wpath = next();
        else if (a == "--h") hpath = next();
        else if (a == "--out") opath = next();
        else if (a == "--mode") mode = next();
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--rows") rows_per_block = atoi(next().c_str());
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }

    // V is implied by the weight file size: bf16, row-major, hidden = 1024.
    std::ifstream wf(wpath, std::ios::binary | std::ios::ate);
    size_t wbytes = (size_t)wf.tellg();
    int V = (int)(wbytes / (HID * 2));
    printf("W %zu MiB (%d rows x %d) | mode %s | rows/block %d\n", wbytes >> 20, V, HID,
           mode.c_str(), rows_per_block);

    auto hbuf = slurp(hpath, HID * sizeof(float));

    // The load path the engine will use: read shards into pinned host memory, then one
    // hipMemcpy per shard (devices/strix-halo-unified-memory.md, ~76-82 GB/s).
    char* hw = nullptr;
    CK(hipHostMalloc(&hw, wbytes));
    {
        std::ifstream f(wpath, std::ios::binary);
        f.read(hw, wbytes);
    }
    hip_bfloat16* dw = nullptr;
    CK(hipMalloc(&dw, wbytes));                       // > carve-out, lands in GTT
    float* dh = nullptr;
    CK(hipMalloc(&dh, HID * sizeof(float)));
    CK(hipMemcpy(dh, hbuf.data(), HID * sizeof(float), hipMemcpyHostToDevice));

    auto t0 = std::chrono::steady_clock::now();
    CK(hipMemcpy(dw, hw, wbytes, hipMemcpyHostToDevice));
    auto t1 = std::chrono::steady_clock::now();
    printf("weight upload %.1f ms (%.1f GB/s)\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count(),
           wbytes / std::chrono::duration<double>(t1 - t0).count() / 1e9);

    float* dout = nullptr;
    unsigned long long* dbest = nullptr;
    bool fused = mode == "fused";
    if (fused) {
        CK(hipMalloc(&dbest, sizeof(unsigned long long)));
        CK(hipMemset(dbest, 0, sizeof(unsigned long long)));
    } else {
        CK(hipMalloc(&dout, (size_t)V * sizeof(float)));
    }

    int threads = 32 * rows_per_block;
    int blocks = (V + rows_per_block - 1) / rows_per_block;
    auto launch = [&]() {
        if (fused)
            lm_head_gemv_fused<<<blocks, threads>>>(dw, dh, dbest, V, rows_per_block);
        else
            lm_head_gemv_full<<<blocks, threads>>>(dw, dh, dout, V, rows_per_block);
    };

    launch();  // first touch: pages fault in here, which is why timing starts after it
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    // Burst wall-clock, not hipEvents. The event path here returned exactly 0.000 ms for a
    // ~2 ms kernel (best of 40), which is the AGENTS trap taken further than it was written:
    // the floor is not "below ~1 ms" on this box. Same idiom as prototype/glue: submit a burst,
    // one sync, divide. best/worst over several bursts is the spread the ledger asks for.
    auto burst = [&](int n) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) launch();
        CK(hipDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / n;
    };
    const int nb = 5;
    const int per = std::max(1, iters / nb);
    burst(1);                        // warm the burst loop itself
    double best = 1e30, worst = 0.0;
    for (int b = 0; b < nb; ++b) {
        const double ms = burst(per);
        best = std::min(best, ms);
        worst = std::max(worst, ms);
    }
    printf("spread best %.4f ms / worst %.4f ms (%d bursts x %d)\n", best, worst, nb, per);
    double bytes = (double)wbytes + (fused ? 8.0 : (double)V * 4);
    printf("gemv %.3f ms -> %.1f GB/s (%.2f%% of the 237 GB/s probe)\n", best,
           bytes / (best / 1e3) / 1e9, bytes / (best / 1e3) / 237e9 * 100.0);
    printf("RESULT mode=%s ms=%.4f gbs=%.1f rows=%d h=%d rows_per_block=%d\n",
           mode.c_str(), best, bytes / (best / 1e3) / 1e9, V, HID, rows_per_block);

    if (fused) {
        unsigned long long key = 0;
        CK(hipMemcpy(&key, dbest, 8, hipMemcpyDeviceToHost));
        unsigned kb = (unsigned)(key >> 20);
        unsigned vb = (kb & 0x80000000u) ? (kb & 0x7fffffffu) : ~kb;
        float val;
        memcpy(&val, &vb, 4);
        unsigned idx = (unsigned)(key & 0xfffff);
        printf("argmax %u (%.4f)\n", idx, val);
        FILE* f = fopen(opath.c_str(), "wb");
        float outv[2] = {val, (float)idx};
        fwrite(outv, 4, 2, f);
        fclose(f);
    } else {
        std::vector<float> hout(V);
        CK(hipMemcpy(hout.data(), dout, (size_t)V * 4, hipMemcpyDeviceToHost));
        FILE* f = fopen(opath.c_str(), "wb");
        fwrite(hout.data(), 4, V, f);
        fclose(f);
    }
    CK(hipFree(dw));
    CK(hipFree(dh));
    if (dout) CK(hipFree(dout));
    if (dbest) CK(hipFree(dbest));
    CK(hipHostFree(hw));
    return 0;
}
