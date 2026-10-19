// a6_rf_probe.inc — appended to a standalone TU that includes the QUARANTINED fattn header, making the A6 Step 3 probe.
//
// WHY NOT THE LAUNCHER TU. `engine/kernels_attn.cpp` references `engine::launchstats`, which lives in another
// translation unit, so linking it here would be a link exercise. The fattn kernel and the rungs are both DEFINED
// in `engine/attn_fattn_kernels.h`, so a TU that merely includes it carries them.
//
// WHAT IT IS. `hipOccupancyMaxActiveBlocksPerMultiprocessor` per kernel, plus `hipFuncGetAttributes` per kernel,
// and NOTHING ELSE. No kernel is launched, no memory is allocated, no timing is taken. `tools/kernel_occupancy_probe.cpp`
// and `prototype/attn_wmma/occupancy_fattn_probe.cpp` are the precedents that a driver capability/occupancy query
// is a device FACT and not a measurement; this file creates a context and submits nothing.
//
// WHY IT IS APPENDED RATHER THAN LINKED SEPARATELY. The bodies live in `engine/attn_fattn_kernels.h`'s anonymous
// namespace, so the ladder kernels have to be compiled in the same translation unit that includes it —
// `a6_rf_ladder.py` inserts them there and this file supplies the `main()` for that same TU. One TU also means the
// `--save-temps` listing carries every rung's note block, so the compile-side read and the driver-side read are
// two views of ONE object file rather than two builds.//
// THE STUBS ARE THE INSTRUMENT, and they are the reason the table can be read at all. The driver's LDS term is
// measured with trivial bodies that hold a fixed LDS and almost no registers:
//     0 B -> many,  8,192 -> 8,  12,288 -> 5,  20,480 -> 3,  32,768 -> 2,  65,536 -> 1
// which is `prototype/attn_percost/A5-GEOMETRY.md` §4.1's derived column read from the driver instead of from
// arithmetic. With those green, a REAL kernel that reports FEWER blocks than its own LDS term allows has been
// refused by the register file, and the gap is attributable rather than inferred. That is the question
// `prototype/attn_percost/A3-result.md` §2b left open (clang says 178 VGPR, the trace said 248, the part reports
// no register-file field) and that `prototype/attn_wmma/E1-result.md` §9 named as a precondition for any claim
// above 6 wf/CU.
//
// The `__launch_bounds__(96, N)` rungs are the second instrument, with a caveat this file states instead of
// hiding: a minBlocks hint makes the compiler *squeeze* VGPRs to satisfy the driver, so a rung printing N is
// partly self-fulfilling. What is not self-fulfilling is the cost of the squeeze — the VGPR count and scratch at
// each rung — so both are printed, and the honest register-file answer is the pair (stub term, real-kernel term)
// rather than the plateau of the ladder alone.

namespace engine {
namespace kernels {

// LDS-only stubs at the block size the fattn form uses. Bodies do nothing but touch their stage, so their register
// footprint is the few VGPRs of the address arithmetic and the LDS term is the only thing that can bind.
#define A6_LDS_STUB(NAME, BYTES)                                                    \
    __global__ void __launch_bounds__(FATTN_THREADS) NAME(float* o) {               \
        __shared__ __align__(16) char s[BYTES];                                     \
        s[threadIdx.x] = (char)threadIdx.x;                                         \
        __syncthreads();                                                            \
        o[threadIdx.x] = (float)s[threadIdx.x];                                     \
    }
// The 0-B control carries no stage at all: a zero-length `__shared__` array is not permitted in HIP device code,
// and its absence IS the control (the query must be able to say "many" when nothing binds).
__global__ void __launch_bounds__(FATTN_THREADS) a6_stub_0(float* o) { o[threadIdx.x] = (float)threadIdx.x; }
A6_LDS_STUB(a6_stub_8192, 8192)
A6_LDS_STUB(a6_stub_12288, 12288)
A6_LDS_STUB(a6_stub_20480, 20480)
A6_LDS_STUB(a6_stub_32768, 32768)
A6_LDS_STUB(a6_stub_65536, 65536)
#undef A6_LDS_STUB

}  // namespace kernels
}  // namespace engine

namespace {

struct A6Row {
    const char* name;
    const void* fn;
    int threads;
    int expect_blocks;      // -1 = no pre-registered expectation (a real kernel: that IS the question)
    const char* what;
};

int a6_report(const A6Row& r) {
    int blocks = -1;
    hipFuncAttributes at{};
    const bool have = hipFuncGetAttributes(&at, r.fn) == hipSuccess;
    const hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, r.fn, r.threads, 0);
    if (e != hipSuccess) {
        std::printf("ROW %-18s QUERY FAILED: %s\n", r.name, hipGetErrorString(e));
        return 1;
    }
    // dynamicLDS is 0 on purpose: every byte here is static shared, declared in the kernel body. Charging the
    // static figure again as dynamic double-counts it, which is the mistake recorded in
    // `prototype/attn_wmma/occupancy_fattn_probe.cpp`'s own header.
    const int warps = r.threads / 32;
    const int bad = (r.expect_blocks >= 0 && blocks != r.expect_blocks);
    std::printf("ROW %-18s threads=%-4d numRegs=%-4d scratch=%-4d lds=%-6d blocks/CU=%-3d wf/CU=%-3d %-9s %s\n",
                r.name, r.threads, have ? (int)at.numRegs : -1,
                have ? (int)(at.localSizeBytes) : -1, have ? (int)(at.sharedSizeBytes) : -1,
                blocks, blocks * warps,
                r.expect_blocks < 0 ? "" : (bad ? "MISMATCH" : "as-derived"), r.what);
    return bad ? 1 : 0;
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) { std::printf("hipSetDevice failed\n"); return 2; }
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) { std::printf("hipGetDeviceProperties failed\n"); return 2; }
    // Printed as the driver's own view, WITH the warning: on this SKU `multiProcessorCount` reports against a
    // halved SM count and `regsPerMultiprocessor` is the figure A5 §3.3 had to quote with its honesty caveat.
    // The authoritative CU/LDS facts are KFD's and rocminfo's (AGENTS' counter/occupancy trap row), and this
    // probe does not use the HIP numbers for anything except printing them beside the answer.
    std::printf("device %s | hip multiProcessorCount=%d (halved on this SKU; KFD+rocminfo say 40 CUs) "
                "hip sharedMemPerMultiprocessor=%d B hip regsPerMultiprocessor=%d\n",
                p.name, p.multiProcessorCount, (int)p.sharedMemPerMultiprocessor, (int)p.regsPerMultiprocessor);
    std::printf("No kernel is launched and no timing is taken: driver queries only. threads/warps come from the "
                "header of the side under test.\n\n");

    int bad = 0;
    const A6Row stubs[] = {
        {"a6_stub_0",     (const void*)&engine::kernels::a6_stub_0,     FATTN_THREADS, -1, "LDS term, unbounded"},
        {"a6_stub_8192",  (const void*)&engine::kernels::a6_stub_8192,  FATTN_THREADS,  8, "LDS 65536/8192"},
        {"a6_stub_12288", (const void*)&engine::kernels::a6_stub_12288, FATTN_THREADS,  5, "LDS term, the shipped attn_slim_wave row"},
        {"a6_stub_20480", (const void*)&engine::kernels::a6_stub_20480, FATTN_THREADS,  3, "LDS term, G1's figure"},
        {"a6_stub_32768", (const void*)&engine::kernels::a6_stub_32768, FATTN_THREADS,  2, "LDS term, shipped figure"},
        {"a6_stub_65536", (const void*)&engine::kernels::a6_stub_65536, FATTN_THREADS,  1, "LDS term, G0's figure"},
    };
    for (const A6Row& r : stubs) bad += a6_report(r);
    std::printf("\n");

    const A6Row real[] = {
        {"attn_fattn_wave", (const void*)&engine::kernels::attn_fattn_wave, FATTN_THREADS, -1,
         "THE QUESTION: this side's shipped body, no minBlocks hint"},
        {"a6_lb1", (const void*)&engine::kernels::a6_lb1, FATTN_THREADS, -1, "launch_bounds(,1)"},
        {"a6_lb2", (const void*)&engine::kernels::a6_lb2, FATTN_THREADS, -1, "launch_bounds(,2)"},
        {"a6_lb3", (const void*)&engine::kernels::a6_lb3, FATTN_THREADS, -1, "launch_bounds(,3)"},
        {"a6_lb4", (const void*)&engine::kernels::a6_lb4, FATTN_THREADS, -1, "launch_bounds(,4)"},
        {"a6_lb5", (const void*)&engine::kernels::a6_lb5, FATTN_THREADS, -1, "launch_bounds(,5)"},
        {"a6_lb6", (const void*)&engine::kernels::a6_lb6, FATTN_THREADS, -1, "launch_bounds(,6)"},
        {"a6_lb7", (const void*)&engine::kernels::a6_lb7, FATTN_THREADS, -1, "launch_bounds(,7)"},
        {"a6_lb8", (const void*)&engine::kernels::a6_lb8, FATTN_THREADS, -1, "launch_bounds(,8)"},
    };
    for (const A6Row& r : real) bad += a6_report(r);

    // CANARY: the stubs must reproduce the derived column, or the driver's answer to the real question is not a
    // fact about registers — it is a query that never arrived (AGENTS: a check whose input can silently be empty
    // cannot fail, so the check that can fail is run first and its failure aborts the reading).
    int b12 = 0, b20 = 0;
    const hipError_t c1 =
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&b12, (const void*)&engine::kernels::a6_stub_12288, FATTN_THREADS, 0);
    const hipError_t c2 =
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&b20, (const void*)&engine::kernels::a6_stub_20480, FATTN_THREADS, 0);
    if (c1 != hipSuccess || c2 != hipSuccess) { std::printf("CANARY FAILED: the control queries errored\n"); return 2; }
    std::printf("\nCANARY stubs reproduce A5 §4.1's derived LDS column: %s (12,288 B -> %d want 5; "
                "20,480 B -> %d want 3)\n", (b12 == 5 && b20 == 3) ? "PASS" : "FAIL", b12, b20);
    return bad ? 1 : 0;
}
