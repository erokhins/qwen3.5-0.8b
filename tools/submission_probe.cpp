// tools/submission_probe.cpp — is a per-op *submission* cost on the critical path?
//
// Why this exists. The bytes-and-launches census (progress/2026-10-17, D30 4.2') found the `copy`
// family is 4096 2-KiB device-to-device row copies per wave, 2.56 us each, and that the family's whole
// byte bill is 0.16 ms against a 10.86 ms row. That reading makes the family a *submission* item, and
// the same question re-prices every launch-shaped row in D30 (glue, and the 73,728 gdn_rec submissions).
// The engine's own logs cannot answer it: WAVEGAP bounds the exposed submission cost only to +-22 ms.
// So: submit exactly the shape the engine submits, in one wall, and divide. Read as a two-sided
// prediction, not a hope (the D34 rule) - if a 2-KiD2D submission costs ~2 us eager, deleting these
// ops is worth ~7-10 ms of wave; if it costs <=0.5 us, the copy row is NOT submission cost and the
// gather kernel is dead before it is written.
//
// Timings on this box are main-agent-only (AGENTS.md): run blocking, zero subagents, under the GPU
// lock, and expect dispatch-bound rows to be bimodal - so every config is repeated and reported as a
// spread, never as a single best.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)
using clk = std::chrono::steady_clock;

__global__ void touch(uint16_t* dst, const uint16_t* src) {
    // one 2 KiB row per block, the shape of one E[y] row copy: 1024 uint16 = 2 KiB.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i];
}
__global__ void empty() {}

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    const int rows = 4096;                       // one wave at 4096 lanes: E[y] is one 2 KiB row/token
    const size_t row_bytes = 2 * 1024;
    const int blocks = 64, threads = 256;        // 2 KiB / (2 bytes) = 1024 threads = 4 blocks; keep it small
    uint16_t *src = nullptr, *dst = nullptr;
    CK(hipMalloc(&src, row_bytes * rows));
    CK(hipMalloc(&dst, row_bytes * rows));
    CK(hipMemset(src, 0x3c, row_bytes * rows));
    CK(hipMemset(dst, 0, row_bytes * rows));
    uint16_t* host = (uint16_t*)malloc(row_bytes);
    memset(host, 0x11, row_bytes);

    printf("submission_probe rows=%d row_bytes=%zu reps=%d\n", rows, row_bytes, reps);
    for (int r = 0; r < reps; ++r) {
        // warm the pages once per config (first GPU pass over fresh pages runs ~35 GB/s).
        CK(hipMemcpy(dst, src, row_bytes * rows, hipMemcpyDeviceToDevice));
        CK(hipDeviceSynchronize());

        // (1) the engine's own shape: n separate D2D copies, one per token, then ONE sync. This is
        //     what decode_step.cpp:466-471 does and what the gather kernel would delete.
        auto t0 = clk::now();
        for (int t = 0; t < rows; ++t)
            CK(hipMemcpyAsync(dst + (size_t)t * 1024, src + (size_t)t * 1024, row_bytes,
                              hipMemcpyDeviceToDevice));
        CK(hipDeviceSynchronize());
        double d2d = ms(t0, clk::now());

        // (2) the same bytes as ONE copy: the floor a gather kernel cannot beat, bytes-wise.
        auto t1 = clk::now();
        CK(hipMemcpyAsync(dst, src, row_bytes * rows, hipMemcpyDeviceToDevice));
        CK(hipDeviceSynchronize());
        double one = ms(t1, clk::now());

        // (3) a kernel launched once per row: the gather kernel's own submission shape, and a check
        //     that a real kernel per row is not cheaper than a memcpy per row.
        auto t2 = clk::now();
        for (int t = 0; t < rows; ++t)
            touch<<<1, 1024>>>(dst + (size_t)t * 1024, src + (size_t)t * 1024);
        CK(hipDeviceSynchronize());
        double kern = ms(t2, clk::now());

        // (4) empty launches: the dispatch floor the census used (repo figure 1.85 us).
        auto t3 = clk::now();
        for (int t = 0; t < rows; ++t) empty<<<1, 32>>>();
        CK(hipDeviceSynchronize());
        double floor_us = ms(t3, clk::now());

        // (5) one kernel doing all rows: what "delete the submissions" actually costs.
        auto t4 = clk::now();
        touch<<<rows / 4, 256>>>(dst, src);
        CK(hipDeviceSynchronize());
        double gathered = ms(t4, clk::now());

        printf("REP %d  n_d2d=%.3f ms (%.3f us/op)  one_d2d=%.3f ms  n_kern=%.3f ms (%.3f us/op)"
               "  empty=%.3f ms (%.3f us/op)  gathered=%.3f ms\n",
               r, d2d, d2d * 1000.0 / rows, one, kern, kern * 1000.0 / rows, floor_us,
               floor_us * 1000.0 / rows, gathered);
        fflush(stdout);
    }
    free(host);
    CK(hipFree(src));
    CK(hipFree(dst));
    return 0;
}
