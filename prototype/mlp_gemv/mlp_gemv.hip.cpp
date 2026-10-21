/*
 * prototype/mlp_gemv/mlp_gemv.hip.cpp — the second GEMV family: the §03.3 MLP, k = 1.
 *
 * Op (S03.3, decode step, one token):
 *     act  = bf16( silu(bf16(x . Wg^T)) * bf16(x . Wu^T) )      [3584]
 *     out  = bf16(act . Wd^T)                                   [1024]
 * The MLP is 0.5285 GB/token = 33 % of the decode bytes (§01.7, progress/2026-10-16/performance-status-2026-10-16.md:80; §4 of
 * `engine/performance-status.md` carries today's row), the largest single family left after lm_head, and it is the first place the engine has
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
 *                   [--mapping interleave|contig]
 *            Q10: the gate/up weight walk. `interleave` (the default, and what has always shipped)
 *            fetches both streams' words side by side; `contig` walks the gate row end to end, then
 *            the up row. Same launch shape either way, and bit-identical output — the claim and its
 *            device proof are `MlpMap` in mlp_gemv_kernels.h plus run.py --mapping-identity.
 *            `down` has one weight stream, so it has no walk to choose and refuses the flag.
 *        ./mlp_gemv --mode down   --x act.bin --wd Wd.bin --out out.bin
 *        ./mlp_gemv --mode gateup --x xtile.bin --in 1024 --tokens T --wg Wg.bin --wu Wu.bin \
 *                                 --out act_tile.bin [--pre pre_tile.bin]
 *            P4: the token-batched form. --in gives the per-token width so the x file can hold T
 *            tokens ([T, in] BF16); outputs are [T, rows] (and --pre [T, 2, rows]). T=1 without
 *            --in is the shipped single-token kernel, same launch shape.
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
#include "mlp_gemv_kernels.h"

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

static size_t filesize(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    return (size_t)f.tellg();
}

// Q10's mapping switch, one line of it: `--mapping` picks which instantiation gets launched. The
// walk is a template argument, so each mapping compiles exactly as it would on its own (no runtime
// test in the row loop) and the two cannot drift into sharing code by accident. Everything else —
// grid, threads, rows/block, argument list — is identical between them by construction: both sides
// of the A/B are the same launch shape, which is what §Q9's pair held fixed.
static const char* map_name(int map) { return map == MAP_CONTIG ? "contig" : "interleave"; }

template <int IN>
static void launch_gateup(const hip_bfloat16* wg, const hip_bfloat16* wu, const float* dx,
                          hip_bfloat16* out, hip_bfloat16* pre, int rows, int rpb, int blocks,
                          int threads, int neg, int map) {
    if (map == MAP_CONTIG)
        mlp_gateup<IN, MAP_CONTIG><<<blocks, threads>>>(wg, wu, dx, out, pre, rows, rpb, neg);
    else
        mlp_gateup<IN, MAP_INTERLEAVE><<<blocks, threads>>>(wg, wu, dx, out, pre, rows, rpb, neg);
}

template <int IN, int TPB>
static void launch_gateup_tiles(const hip_bfloat16* wg, const hip_bfloat16* wu, const float* dx,
                                hip_bfloat16* out, hip_bfloat16* pre, int rows, int tokens, int rpb,
                                const dim3& grid, int threads, int neg, int map) {
    if (map == MAP_CONTIG)
        mlp_gateup_tokens<IN, TPB, MAP_CONTIG><<<grid, threads>>>(wg, wu, dx, out, pre, rows,
                                                                   tokens, rpb, neg);
    else
        mlp_gateup_tokens<IN, TPB, MAP_INTERLEAVE><<<grid, threads>>>(wg, wu, dx, out, pre, rows,
                                                                      tokens, rpb, neg);
}

int main(int argc, char** argv) {
    std::string mode = "gateup", xpath, wgpath, wupath, wdpath, opath, ppath, apath, aopath,
            negname = "none";
    int iters = 20;
    int rows_per_block = ROWS_PER_BLOCK;
    int tok_in = 0, tok_n = 1;      // P4: the token-batched form (--in gives the width, so the
                                    // x file can hold several tokens; without --in this is k=1)
    std::string mapping = "interleave";   // Q10: interleave | contig (see MlpMap in the kernel header)
    bool mapping_given = false;
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
        else if (a == "--in") tok_in = atoi(next().c_str());
        else if (a == "--tokens") tok_n = atoi(next().c_str());
        else if (a == "--neg") negname = next();
        else if (a == "--mapping") {
            mapping = next();
            mapping_given = true;
        }
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
    // Only gate/up has two weight streams, so only gate/up has a mapping to choose. `down` is one
    // stream by construction and says so by refusing the flag rather than pretending to switch.
    const int map = mapping == "interleave" ? MAP_INTERLEAVE : mapping == "contig" ? MAP_CONTIG : -1;
    if (map < 0) {
        fprintf(stderr, "--mapping must be interleave|contig, got %s\n", mapping.c_str());
        return 1;
    }
    if (mapping_given && mode == "down") {
        fprintf(stderr, "--mapping is the gate/up weight walk; --mode down has one weight stream"
                        " and no interleave to remove\n");
        return 1;
    }
    const size_t xbytes = filesize(xpath);
    // Without --in the x file is one token and everything is derived from its size, exactly as
    // before. With --in the file is a [tokens, in] tile and `--tokens` must agree with it.
    const int in = tok_in ? tok_in : (int)(xbytes / 2);
    const int tokens = tok_n;
    if (tokens < 1) {
        fprintf(stderr, "--tokens must be >= 1\n");
        return 1;
    }
    if (xbytes != (size_t)in * tokens * 2) {
        fprintf(stderr, "%s is %zu bytes; --in %d x --tokens %d x BF16 wants %zu\n",
                xpath.c_str(), xbytes, in, tokens, (size_t)in * tokens * 2);
        return 1;
    }
    if (tokens > 1 && pair) {
        fprintf(stderr, "--mode mlp (the alternating pair) is the k=1 burst; use --mode gateup"
                        " with --tokens\n");
        return 1;
    }
    // Shared memory, not registers, is what bounds the token tile: TPB*IN floats must stay inside
        // the 48 KiB static limit with room for two blocks per CU (64 KiB LDS on this part).
    const int tpb = in == HID ? 4 : 2;
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
    printf("mode %s | x in=%d | rows=%d | tokens=%d (tile %d) | warpSize=%d | rows/block=%d |"
           " mapping=%s | weights %.2f MiB%s\n",
           mode.c_str(), in, rows, tokens, tpb, ws, rows_per_block, map_name(map),
           wbytes / 1048576.0,
           pair ? " (+down weights, pair burst)" : "");

    // x arrives as BF16 (the reference's own operands are BF16-valued). It is staged as F32
    // because every bf16 value is exact in fp32, so the products are identical and the
    // inner loop never converts x.
    auto xraw = slurp(xpath, xbytes);
    std::vector<float> xhost((size_t)in * tokens);
    {
        const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(xraw.data());
        for (size_t i = 0; i < xhost.size(); ++i) xhost[i] = float(p[i]);
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
    CK(hipMalloc(&dx, (size_t)xhost.size() * sizeof(float)));
    CK(hipMemcpy(dx, xhost.data(), (size_t)xhost.size() * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&dout, (size_t)(pair ? rows2 : rows) * tokens * 2));
    if (gateup && !ppath.empty()) CK(hipMalloc(&dpre, 2 * (size_t)rows * tokens * 2));

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
    // tokens == 1 keeps the shipped single-token kernel and its exact launch shape — the decode
    // path's ledger number is not this package's to move. The batched form takes over above 1 and
    // is checked bit-for-bit against the single-token one by run.py --tokens.
    const dim3 grid_t((unsigned)blocks, (unsigned)((tokens + tpb - 1) / tpb));
    auto launch = [&]() {
        if (pair) {          // one layer of a real decode step: gate/up, then down, alternated
            if (in == HID)
                launch_gateup<HID>(dw, dw2, dx, dact, nullptr, rows, rows_per_block, blocks,
                                   threads, neg, map);
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
            if (tokens > 1) {
                if (in == HID)
                    launch_gateup_tiles<HID, 4>(dw, dw2, dx, dout, dpre, rows, tokens,
                                                rows_per_block, grid_t, threads, neg, map);
                else if (in == FF)
                    launch_gateup_tiles<FF, 2>(dw, dw2, dx, dout, dpre, rows, tokens,
                                               rows_per_block, grid_t, threads, neg, map);
                else {
                    fprintf(stderr, "gateup: in=%d not implemented\n", in);
                    exit(1);
                }
                return;
            }
            if (in == HID)
                launch_gateup<HID>(dw, dw2, dx, dout, dpre, rows, rows_per_block, blocks, threads,
                                   neg, map);
            else if (in == FF)
                launch_gateup<FF>(dw, dw2, dx, dout, dpre, rows, rows_per_block, blocks, threads,
                                  neg, map);
            else {
                fprintf(stderr, "gateup: in=%d not implemented\n", in);
                exit(1);
            }
        } else {
            if (tokens > 1) {
                if (in == HID)
                    mlp_down_tokens<HID, 4><<<grid_t, threads>>>(dw, dx, dout, rows, tokens,
                                                                 rows_per_block, neg);
                else if (in == FF)
                    mlp_down_tokens<FF, 2><<<grid_t, threads>>>(dw, dx, dout, rows, tokens,
                                                                rows_per_block, neg);
                else {
                    fprintf(stderr, "down: in=%d not implemented\n", in);
                    exit(1);
                }
                return;
            }
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
    // it is the sum of both kernels' traffic, and one burst iteration is one layer. For the
    // batched form the weights are counted ONCE for the whole tile — that is the whole claim of
    // the form, so the accounting must state it; x and the outputs scale with tokens.
    const size_t wdbytes = pair ? filesize(wdpath) : 0;
    const double bytes = pair
        ? (double)(2 * wbytes + wdbytes) + xbytes + abytes + (double)rows * 2.0
          + (double)rows2 * 2.0
        : (double)wbytes * (gateup ? 2.0 : 1.0) + xbytes
          + (double)rows * tokens * 2.0
          + (gateup && !ppath.empty() ? 2.0 * rows * tokens * 2.0 : 0.0);

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
    printf("RESULT mode=%s ms=%.4f gbs=%.1f rows=%d in=%d rows_per_block=%d tokens=%d tile=%d"
           " mapping=%s\n",
           mode.c_str(), best, gbs, rows, in, rows_per_block, tokens, tpb, map_name(map));

    std::vector<hip_bfloat16> hout((size_t)(pair ? rows2 : rows) * tokens);
    CK(hipMemcpy(hout.data(), dout, hout.size() * 2, hipMemcpyDeviceToHost));
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
        std::vector<hip_bfloat16> hpre((size_t)2 * rows * tokens);
        CK(hipMemcpy(hpre.data(), dpre, hpre.size() * 2, hipMemcpyDeviceToHost));
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
