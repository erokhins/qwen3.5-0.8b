// device_query.cpp — authoritative device identity and the properties every
// CU-count argument in this repo depends on.
//
// Why this exists: devices/strix-halo-memory-bandwidth.md recorded "hipGetDeviceProperties
// reports 20 CUs" from an ad-hoc print in bw_probe_hip.cpp, while `rocminfo` reports
// 40 Compute Units for gfx1151. The MLP results.md and engine/performance-status.md both
// explain a 16% bandwidth gap with "N blocks across 40 CUs", i.e. an argument whose input
// the repo contradicts itself on. Numbers quoted in this repo need a tool behind them
// (AGENTS.md), so query them directly instead of arguing from whichever doc is open.
//
// build: hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o device_query device_query.cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
  return 1; } } while (0)

int main() {
// device_query build: hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o device_query tools/device_query.cpp/
  int n = 0;
  CK(hipGetDeviceCount(&n));
  printf("device count: %d\n", n);
  for (int d = 0; d < n; d++) {
    hipDeviceProp_t p{};
    CK(hipGetDeviceProperties(&p, d));
    printf("\n[device %d]\n", d);
    int mc = -1, mt = -1, ws = -1, l2 = -1, tb = -1, bpu = -1;
    CK(hipDeviceGetAttribute(&mc, hipDeviceAttributeMultiprocessorCount, d));
    CK(hipDeviceGetAttribute(&mt, hipDeviceAttributeMaxThreadsPerMultiProcessor, d));
    CK(hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, d));
    CK(hipDeviceGetAttribute(&tb, hipDeviceAttributeMaxThreadsPerBlock, d));
    // l2CacheSize is not in hipDeviceProp_t on this ROCm; the attribute is.
    hipError_t e = hipDeviceGetAttribute(&l2, hipDeviceAttributeL2CacheSize, d);
    if (e != hipSuccess) l2 = -1;
    hipDeviceGetAttribute(&bpu, hipDeviceAttributeMaxBlocksPerMultiProcessor, d);
    printf("  name                  %s\n", p.name);
    printf("  gcnArchName           %s\n", p.gcnArchName);
    printf("  multiProcessorCount   %d          <- the CU count (prop)\n", p.multiProcessorCount);
    printf("  attr MultiprocessorCount %d       <- the CU count (attribute)\n", mc);
    printf("  warpSize              %d          (prop %d)\n", ws, p.warpSize);
    printf("  maxThreadsPerMP       %d          (prop %d)\n", mt, p.maxThreadsPerMultiProcessor);
    printf("  maxThreadsPerBlock    %d          (prop %d)\n", tb, p.maxThreadsPerBlock);
    printf("  maxBlocksPerMP        %d\n", bpu);
    printf("  wavesPerMP            %d          = maxThreadsPerMP / warpSize\n", ws > 0 ? mt / ws : -1);
    printf("  L2CacheSize           %d bytes (%.1f MiB)\n", l2, l2 / 1048576.0);
    printf("  sharedMemPerBlock     %zu bytes\n", p.sharedMemPerBlock);
    printf("  memoryBusWidthBits    %d\n", p.memoryBusWidth);
    printf("  memoryClockRateKHz    %d\n", p.memoryClockRate);
    size_t free_b = 0, tot_b = 0;
    CK(hipMemGetInfo(&free_b, &tot_b));
    printf("  totalGlobalMem        %.1f GiB\n", p.totalGlobalMem / 1073741824.0);
    printf("  hipMemGetInfo         %.1f GiB total / %.1f GiB free\n",
           tot_b / 1073741824.0, free_b / 1073741824.0);
    // Concurrent-copy / async-engine count: relevant to whether a copy can overlap a
    // kernel at all, which the residency design assumes.
    printf("  asyncEngineCount      %d\n", p.asyncEngineCount);
    printf("  concurrentKernels     %d\n", p.concurrentKernels);
  }
  return 0;
}
