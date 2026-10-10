// launch_overhead_hip.cpp -- cost of a kernel launch, isolated.
//
// Why this exists: the token roofline (1.565 GB / 237 GB/s = ~6.6 ms) assumes bandwidth is
// the only cost. It is not. A decode step is ~24 layers x ~6-8 kernels ~= 150-200 launches,
// and on a 20-CU part with no work to hide behind, launch cost adds directly to the token if
// it is not covered by overlap or graph capture. If launch cost x launches is a meaningful
// fraction of 6.6 ms, the engine needs HIP Graphs (or fused megakernels) before it can
// claim the roofline; if it is ~1% the projection is safe.
//
//     hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o launch_overhead_hip launch_overhead_hip.cpp
//
// Three measurements:
//   empty   : a kernel that does nothing  -> pure launch + completion cost
//   tiny    : one block writing one float -> same, plus negligible work
//   graph   : the same tiny launches captured into one HIP graph and replayed
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e=(x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

__global__ void empty_k() {}
__global__ void tiny_k(float* p) { if (threadIdx.x == 0) p[0] = 1.f; }

static float timeit(void (*fn)(void*), void* ctx, int iters) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    fn(ctx);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) fn(ctx);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, a, b));
    hipEventDestroy(a); hipEventDestroy(b);
    return ms / iters;
}

static float* d_p;

void run_empty(void*) { empty_k<<<1, 64>>>(); }
void run_tiny(void*) { tiny_k<<<1, 64>>>(d_p); }

struct GraphCtx { hipGraphExec_t exec; };
static hipStream_t g_stream = nullptr;
void run_graph(void* v) { CK(hipGraphLaunch(((GraphCtx*)v)->exec, g_stream)); }

static float timeit_graph(void (*fn)(void*), void* ctx, hipStream_t s, int iters) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    g_stream = s;
    fn(ctx);
    CK(hipStreamSynchronize(s));
    CK(hipEventRecord(a, s));
    for (int i = 0; i < iters; ++i) fn(ctx);
    CK(hipEventRecord(b, s));
    CK(hipEventSynchronize(b));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, a, b));
    hipEventDestroy(a); hipEventDestroy(b);
    return ms / iters;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 500;
    CK(hipFree(0));
    CK(hipMalloc(&d_p, sizeof(float)));

    float us_empty = timeit(run_empty, nullptr, n) * 1e3f;
    float us_tiny = timeit(run_tiny, nullptr, n) * 1e3f;
    printf("RESULT empty_launch_us=%.3f tiny_launch_us=%.3f\n", us_empty, us_tiny);

    // Same launches, but captured into a graph and replayed as one submission. Capture 64
    // of them so one replay stands for 64 kernel issues, like one transformer block.
    // The legacy default stream cannot be captured ("operation not permitted"), so use a
    // dedicated non-blocking stream for both capture and replay.
    const int K = 64;
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipGraph_t g;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < K; ++i) tiny_k<<<1, 64, 0, s>>>(d_p);
    CK(hipStreamEndCapture(s, &g));
    GraphCtx gc;
    CK(hipGraphInstantiate(&gc.exec, g, nullptr, nullptr, 0));
    float us_graph_rep = timeit_graph(run_graph, &gc, s, n) * 1e3f;
    printf("RESULT graph_replay_64_us=%.3f  per_kernel_us=%.3f  speedup=%.1fx\n",
           us_graph_rep, us_graph_rep / K, us_tiny / (us_graph_rep / K));
    printf("RESULT token_projection: 180 launches => %.2f ms eager, %.2f ms graph\n",
           us_tiny * 180 / 1e3, us_graph_rep / K * 180 / 1e3);
    return 0;
}
