// p1c_query.cpp — P1c: can a HIP hardware residency hint address the 32 MiB tier on this part?
//
// WHY THIS FILE EXISTS. Route P1 (queue row P1c) has two possible mechanisms and they are not
// interchangeable:
//   (M1) a hardware residency HINT — hipStreamAttributeAccessPolicyWindow /
//        hipAccessPropertyPersisting, CUDA's carve-out idiom. On NVIDIA that carves the L2.
//   (M2) EXPLICIT PREFETCH — a real load issued early, so the bytes are already in the tier when
//        the compute reaches them.
// The hint is cheaper and more robust *if it can reach the tier at all*. `devices/
// strix-halo-memory-bandwidth.md` §"The size-bandwidth curve" measures the tier as ~32 MiB at
// 876.7 GB/s, and hipDeviceGetAttribute(L2CacheSize) is 2 MiB, so the hint may address only a
// tier 16x too small. This tool answers that from the part, not from a header, and prints a verdict.
//
// WHAT COUNTS AS EVIDENCE HERE. An attribute that exists is not an answer: a CUDA-carried enum can
// report 0 while the API "succeeds", and a value of 0 read as "no pressure" is the vacuous-zero
// failure AGENTS.md records for pmc counters. So three independent sources are printed side by side
// and the verdict is computed from them:
//   (a) HIP: hipDeviceProp_t fields and hipDeviceGetAttribute for the persisting-window trio.
//   (b) KFD: /sys/.../kfd/topology/nodes/<n>/caches/*/properties, which is the only place on this
//       box that names a level-3 cache at all (rocminfo has no notion of MALL).
//   (c) FUNCTIONAL: call hipStreamSetAttribute / hipDeviceSetCacheConfig / hipMemAdvise /
//       hipMemPrefetchAsync for real and print what the runtime returns, plus whether the stream
//       attribute round-trips through hipStreamGetAttribute. A hint that returns hipSuccess but
//       does not round-trip is not a hint.
//
// THE OTHER QUESTION THIS ANSWERS, which decides whether the rig's arm (b) is even implementable:
// `devices/strix-halo-memory-bandwidth.md` records hipDeviceProp_t::concurrentKernels == 1 ("the
// driver claims it will not run two kernels at once") and says "re-check before relying on it either
// way". A prefetch on a second stream is worthless if the second stream cannot be co-resident with
// the first, so this tool PROBES co-residency with a boolean, not a stopwatch: a spinner sets
// `spinning=1` for as long as it is resident, and a probe on the other stream reports whether it saw
// that flag set. No timing is taken or printed anywhere in this file; every output is a size, a
// count, or a yes/no.
//
// SAFETY. The spinner is bounded by an iteration cap, not by a deadline, so no configuration of this
// tool can wedge a queue (AGENTS: a device fault on a busy-polling queue presents as a hang). The
// functional probes run in a forked child, because one faulting launch poisons the context and every
// later check in the same process would silently never run (AGENTS: mixed_bw / gdn_walk).
//
// build: hipcc -O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib -o p1c_query p1c_query.cpp
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/wait.h>
#include <unistd.h>

#define CK(x)                                                                     \
    do {                                                                          \
        hipError_t e_ = (x);                                                      \
        if (e_ != hipSuccess) {                                                   \
            fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_),     \
                    __FILE__, __LINE__);                                          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static const char* E(hipError_t e) { return hipGetErrorString(e); }

// ---------------------------------------------------------------------------------------------
// KFD: the only source on this box that names the big tier. Printed as raw fields, plus a summary.
// ---------------------------------------------------------------------------------------------
struct KfdCache {
    int node;
    int index;
    long level;
    long size_kib;
    long line_bytes;
    long type;
};

static long field(const char* path, const char* name) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char key[128];
    long val = -1;
    while (fscanf(f, "%127s %ld", key, &val) == 2) {
        if (!strcmp(key, name)) { fclose(f); return val; }
    }
    fclose(f);
    return -1;
}

// Walks every KFD node, then every cache under each node. Returns the count filled into `out`.
// The node list is collected first: readdir() on a sysfs directory is not rewindable, and the first
// draft of this function reopened the directory mid-walk to keep going, which silently scanned one
// node twice and none of the others.
static int kfd_caches(KfdCache* out, int cap) {
    int nodes[16];
    int nn = 0;
    DIR* nd = opendir("/sys/devices/virtual/kfd/kfd/topology/nodes");
    if (!nd) { fprintf(stderr, "kfd: no topology directory\n"); return 0; }
    struct dirent* de;
    while ((de = readdir(nd))) {
        if (de->d_name[0] < '0' || de->d_name[0] > '9') continue;
        int node = atoi(de->d_name);
        char props[512];
        snprintf(props, sizeof props,
                 "/sys/devices/virtual/kfd/kfd/topology/nodes/%d/properties", node);
        if (field(props, "simd_count") <= 0) continue;   // CPU nodes own no GPU caches
        if (nn < 16) nodes[nn++] = node;
    }
    closedir(nd);

    int n = 0;
    for (int k = 0; k < nn; k++) {
        char dir[512];
        snprintf(dir, sizeof dir,
                 "/sys/devices/virtual/kfd/kfd/topology/nodes/%d/caches", nodes[k]);
        DIR* cs = opendir(dir);
        if (!cs) continue;
        struct dirent* ce;
        while ((ce = readdir(cs))) {
            if (ce->d_name[0] < '0' || ce->d_name[0] > '9') continue;
            int idx = atoi(ce->d_name);
            char p[640];
            snprintf(p, sizeof p, "%s/%d/properties", dir, idx);
            long lvl = field(p, "level"), sz = field(p, "size"), ln = field(p, "cache_line_size");
            long ty = field(p, "type");
            if (lvl < 0 || sz < 0) continue;
            if (n < cap) { out[n++] = {nodes[k], idx, lvl, sz, ln, ty}; }
        }
        closedir(cs);
    }
    return n;
}

static void print_kfd() {
    printf("\n== [b] KFD topology: /sys/devices/virtual/kfd/kfd/topology/nodes/*/caches/*\n");
    static KfdCache cs[128];
    int n = kfd_caches(cs, 128);
    if (n == 0) { printf("  NONE FOUND (cannot answer the verdict from KFD)\n"); return; }
    // Print only the non-L1 caches in full; L1 rows are per-SIMD duplicates (40+20+20+4 here).
    long max_l3 = -1, max_l2 = -1;
    int n_l1 = 0;
    for (int i = 0; i < n; i++) {
        if (cs[i].level == 1) { n_l1++; continue; }
        printf("  node=%d cache=%-3d level=%ld size=%-7ld KiB (%.1f MiB) line=%ld B type=%ld\n",
               cs[i].node, cs[i].index, cs[i].level, cs[i].size_kib, cs[i].size_kib / 1024.0,
               cs[i].line_bytes, cs[i].type);
        if (cs[i].level == 2 && cs[i].size_kib > max_l2) max_l2 = cs[i].size_kib;
        if (cs[i].level >= 3 && cs[i].size_kib > max_l3) max_l3 = cs[i].size_kib;
    }
    printf("  (%d level-1 caches, per-SIMD, omitted)\n", n_l1);
    printf("  KFD level-2 total      : %ld KiB (%.1f MiB)\n", max_l2, max_l2 / 1024.0);
    printf("  KFD level-3+ total     : %ld KiB (%.1f MiB)   <- the tier the bandwidth curve found\n",
           max_l3, max_l3 / 1024.0);
    char props[256];
    snprintf(props, sizeof props, "/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties");
    printf("  node1 lds_size_in_kb=%ld simd_count=%ld simd_per_cu=%ld max_waves_per_simd=%ld\n",
           field(props, "lds_size_in_kb"), field(props, "simd_count"), field(props, "simd_per_cu"),
           field(props, "max_waves_per_simd"));
}

// ---------------------------------------------------------------------------------------------
// [a] HIP device facts.
// ---------------------------------------------------------------------------------------------
static int hip_facts(long* persist_attr, long* l2_attr, long* persist_prop, long* window_prop,
                    long* l2_prop, int* concurrent) {
    printf("\n== [a] HIP device properties and attributes\n");
    int nd = 0;
    CK(hipGetDeviceCount(&nd));
    printf("  device count             %d\n", nd);
    for (int d = 0; d < nd; d++) {
        hipDeviceProp_t p{};
        CK(hipGetDeviceProperties(&p, d));
        printf("  [device %d] %-28s %s\n", d, p.name, p.gcnArchName);
        printf("    prop l2CacheSize              %d B (%.1f MiB)\n", p.l2CacheSize,
               p.l2CacheSize / 1048576.0);
        printf("    prop persistingL2CacheMaxSize %d B (%.3f MiB)\n", p.persistingL2CacheMaxSize,
               p.persistingL2CacheMaxSize / 1048576.0);
        printf("    prop accessPolicyMaxWindowSize %d B (%.3f MiB)\n",
               p.accessPolicyMaxWindowSize, p.accessPolicyMaxWindowSize / 1048576.0);
        printf("    prop concurrentKernels        %d   asyncEngineCount %d\n", p.concurrentKernels,
               p.asyncEngineCount);
        printf("    prop multiProcessorCount      %d   maxThreadsPerMP %d\n",
               p.multiProcessorCount, p.maxThreadsPerMultiProcessor);
        *persist_prop = p.persistingL2CacheMaxSize;
        *window_prop = p.accessPolicyMaxWindowSize;
        *l2_prop = p.l2CacheSize;
        *concurrent = p.concurrentKernels;
        int v;
        hipError_t e = hipDeviceGetAttribute(&v, hipDeviceAttributePersistingL2CacheMaxSize, d);
        printf("    attr PersistingL2CacheMaxSize ");
        if (e == hipSuccess) { printf("%d B (%.1f MiB)\n", v, v / 1048576.0); *persist_attr = v; }
        else { printf("UNSUPPORTED (%s)\n", E(e)); *persist_attr = -1; }
        e = hipDeviceGetAttribute(&v, hipDeviceAttributeL2CacheSize, d);
        printf("    attr L2CacheSize              ");
        if (e == hipSuccess) { printf("%d B (%.1f MiB)\n", v, v / 1048576.0); *l2_attr = v; }
        else { printf("UNSUPPORTED (%s)\n", E(e)); *l2_attr = -1; }
        // The CUDA enum cudaDevAttrMaxAccessPolicyWindowSize has no HIP spelling on this ROCm;
        // report the value the prop carries and name the enumerators that are absent, rather than
        // inventing an attribute or reading silence as a zero.
        printf("    absent on this ROCm: attr MaxAccessPolicyWindowSize, hipLimitPersisting"
               "L2CacheSize, hipCtxResetPersistingL2Cache, hipStreamRemoveAttribute\n");
        size_t lv = 0;
        hipError_t le = hipDeviceGetLimit(&lv, hipLimitMallocHeapSize);
        printf("    hipDeviceGetLimit(mallocHeap) %s  value=%zu B   (sanity: GetLimit works)\n",
               le == hipSuccess ? "ok" : E(le), lv);
        printf("    hipLimit_t has NO persisting-L2 enumerator on this ROCm (hipDeviceSetLimit"
               " cannot carve a persisting set at all)\n");
        break;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// [c] functional probes. Does the hint do anything, and can two streams be resident at once?
// ---------------------------------------------------------------------------------------------
struct Flags {
    volatile unsigned spinning;      // 1 for exactly the interval the spinner is resident
    volatile unsigned release;       // host -> device: the spinner may stop
    volatile unsigned probe_done;    // device -> host: the probe has finished its poll loop
    volatile unsigned saw;           // did the probe see spinning==1
    volatile unsigned long long polls;   // how many polls it took (a COUNT, not a duration)
    volatile unsigned release_seen;  // was release already set when the probe looked
};

// `spinning` is 1 for exactly the interval in which this kernel is resident. It is bounded by an
// iteration cap, never by a deadline, so no setting of `cap` can wedge a queue.
__global__ void spin_kernel(volatile unsigned* spinning, volatile unsigned* release,
                            unsigned long long cap) {
    *spinning = 1u;
    __threadfence();
    const unsigned* rel = (const unsigned*)release;
    unsigned long long i = 0;
    while (i < cap) {
        if (__atomic_load_n(rel, __ATOMIC_ACQUIRE)) break;
        ++i;
    }
    *spinning = 0u;
    __threadfence();
}

// Bounded poll of the partner kernel's liveness flag. Records saw=1 only if it observed the spinner
// resident, and records the poll COUNT at which it saw it. If the driver serialises the two streams the
// flag is already 0 by the time this runs and it records saw=0 after burning its cap. Booleans and a
// count, never a duration.
__global__ void probe_kernel(volatile unsigned* spinning, volatile unsigned* release,
                             unsigned long long cap, volatile unsigned* saw_out,
                             volatile unsigned long long* polls_out,
                             volatile unsigned* release_seen_out,
                             volatile unsigned* done_out) {
    const unsigned* spn = (const unsigned*)spinning;
    const unsigned* rel = (const unsigned*)release;
    unsigned long long i = 0;
    unsigned saw = 0;
    while (i < cap) {
        if (__atomic_load_n(spn, __ATOMIC_ACQUIRE)) { saw = 1; break; }
        ++i;
    }
    *saw_out = saw;
    *polls_out = i;
    *release_seen_out = __atomic_load_n(rel, __ATOMIC_ACQUIRE) ? 1u : 0u;
    __threadfence();
    *done_out = 1u;                 // the host waits for THIS before releasing the spinner
}

__global__ void touch_kernel(const float* p, size_t n, double* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    double acc = 0.0;
    for (size_t k = i; k < n; k += stride) acc += (double)p[k];
    __shared__ double sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

// Returns 0 when the runtime accepted the window, 1 when it refused. The count of accepted sizes is a
// verdict input, so it has to travel back to the parent; a hint that is merely mentioned in a header is
// not a hint.
static int probe_stream_hint(long window_bytes) {
    void* buf = nullptr;
    hipError_t a = hipMalloc(&buf, (size_t)window_bytes);
    if (a != hipSuccess) { printf("  hipMalloc(%ld B) failed: %s\n", window_bytes, E(a)); return 1; }
    hipStream_t s;
    hipStreamCreate(&s);
    hipStreamAttrValue v{};
    v.accessPolicyWindow.base_ptr = buf;
    v.accessPolicyWindow.num_bytes = (size_t)window_bytes;
    v.accessPolicyWindow.hitRatio = 1.0f;
    v.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
    v.accessPolicyWindow.missProp = hipAccessPropertyStreaming;
    hipError_t se = hipStreamSetAttribute(s, hipStreamAttributeAccessPolicyWindow, &v);
    hipStreamAttrValue got{};
    hipError_t ge = hipStreamGetAttribute(s, hipStreamAttributeAccessPolicyWindow, &got);
    printf("  hipStreamSetAttribute(window=%-9ld B persisting/streaming) -> %s\n", window_bytes,
           E(se));
    printf("    hipStreamGetAttribute -> %s  roundtrip_num_bytes=%zu roundtrip_hitRatio=%.3f\n",
           E(ge), (ge == hipSuccess) ? got.accessPolicyWindow.num_bytes : 0u,
           (ge == hipSuccess) ? (double)got.accessPolicyWindow.hitRatio : -1.0);
    // Does the runtime still run a kernel on a stream that carries the attribute?
    float* f = nullptr;
    hipMalloc((void**)&f, 4096);
    double* part = nullptr;
    hipMalloc((void**)&part, 64 * sizeof(double));
    hipError_t le = hipSuccess;
    touch_kernel<<<64, 256, 0, s>>>(f, 1024, part);
    le = hipStreamSynchronize(s);
    printf("    kernel on that stream -> %s\n", E(le));
    hipStreamDestroy(s);
    hipFree(f);
    hipFree(part);
    hipFree(buf);
    return (se == hipSuccess) ? 0 : 1;
}

static void probe_other_knobs() {
    hipFuncCache_t cfg = hipFuncCachePreferShared;
    printf("  hipDeviceSetCacheConfig(preferShared) -> %s\n", E(hipDeviceSetCacheConfig(cfg)));
    void* dm = nullptr;
    hipError_t a = hipMalloc(&dm, 1 << 20);
    if (a == hipSuccess) {
        printf("  hipMemAdviseSetAccessedBy on hipMalloc ptr -> %s\n",
               E(hipMemAdvise(dm, 1 << 20, hipMemAdviseSetAccessedBy, 0)));
        // The engine's buffers are hipMalloc, so this is the call form a real async prefetch would
        // have to use if the API accepted them. Reported, not assumed.
        printf("  hipMemPrefetchAsync on a hipMalloc ptr -> %s   (the engine's state/KV/weights are "
               "hipMalloc, so this is the only async spelling that could apply to them)\n",
               E(hipMemPrefetchAsync(dm, 1 << 20, 0, 0)));
        hipFree(dm);
    }
    void* mg = nullptr;
    a = hipMallocManaged(&mg, 1 << 20);
    printf("  hipMallocManaged(1 MiB) -> %s\n", E(a));
    if (a == hipSuccess) {
        printf("    hipMemPrefetchAsync(1 MiB, dev 0) -> %s ; sync -> %s\n",
               E(hipMemPrefetchAsync(mg, 1 << 20, 0, 0)), E(hipDeviceSynchronize()));
        printf("    NOTE devices/strix-halo-unified-memory.md: managed memory degrades SILENTLY to"
               " ~52 GB/s near ~32 GiB. Weight paths must not use it.\n");
        hipFree(mg);
    }
}

// Co-residency witness, in a forked child so a fault cannot blank the rest of the tool.
//
// THE PROTOCOL MATTERS, and the first draft got it wrong: it wrote `release = 1` immediately after
// launching the probe, so the spinner often finished before the probe ever ran and the probe reported
// "not observed" for a reason that had nothing to do with concurrency. The same tool printed "observed"
// on another run, which is how a race announced itself. The fix is to let the PROBE tell the host when it
// is done (`probe_done`) and to release the spinner only after that, so the release can never pre-empt
// the observation. Everything stays bounded: caps on the device, a bounded host poll.
static int co_residency_child(unsigned capA, unsigned capB, int trials, int* saw_out) {
    Flags* fh = nullptr;
    hipError_t he = hipHostMalloc((void**)&fh, sizeof(Flags), hipHostMallocMapped);
    if (he != hipSuccess) {
        printf("  hipHostMalloc(mapped) -> %s; retrying unmapped\n", E(he));
        he = hipHostMalloc((void**)&fh, sizeof(Flags), 0);
        if (he == hipSuccess) {
            void* dp = nullptr;
            hipError_t ge = hipHostGetDevicePointer(&dp, (void*)fh, 0);
            printf("  hipHostGetDevicePointer -> %s dev_ptr=%p\n", E(ge), dp);
        }
    }
    if (he != hipSuccess) {
        printf("  child: hipHostMalloc failed (%s) - cannot run the co-residency probe\n", E(he));
        return 3;
    }
    hipStream_t s1, s2;
    hipStreamCreate(&s1);
    hipStreamCreate(&s2);
    int observed = 0, completed = 0;
    for (int t = 0; t < trials; t++) {
        memset((void*)fh, 0, sizeof(Flags));
        spin_kernel<<<1, 1, 0, s1>>>(&fh->spinning, &fh->release, capA);
        if (hipGetLastError() != hipSuccess) { printf("  spin launch failed\n"); continue; }
        probe_kernel<<<1, 1, 0, s2>>>(&fh->spinning, &fh->release, capB, &fh->saw, &fh->polls,
                                      &fh->release_seen, &fh->probe_done);
        if (hipGetLastError() != hipSuccess) { printf("  probe launch failed\n"); continue; }
        // Bounded host poll on the mapped flag. No timer: a count of spins, so this prints no duration.
        unsigned long long wait = 0;
        while (!fh->probe_done && wait < 200000000ull) wait++;
        fh->release = 1u;
        hipError_t e2 = hipStreamSynchronize(s2);
        hipError_t e1 = hipStreamSynchronize(s1);
        if (e1 != hipSuccess || e2 != hipSuccess) {
            printf("  trial %d: spin %s probe %s\n", t, E(e1), E(e2));
            continue;
        }
        completed++;
        printf("  trial %d: probe saw the spinner resident = %u after %llu polls (cap %u);"
               " release was already set when the probe looked = %u\n",
               t, fh->saw, fh->polls, capB, fh->release_seen);
        if (fh->saw) observed++;
    }
    printf("  trials=%d completed=%d overlap_observed=%d\n", trials, completed, observed);
    // NEGATIVE CONTROL, and it is not optional: an instrument that can only answer YES has measured
    // nothing (AGENTS: a negative control with an empty must-catch set is not a control). Launching the
    // SAME two kernels on ONE stream serialises them by construction, so the probe must answer 0. If it
    // answers 1 there, the whole [d] section is decoration and the verdict says so.
    memset((void*)fh, 0, sizeof(Flags));
    spin_kernel<<<1, 1, 0, s1>>>(&fh->spinning, &fh->release, capA);
    probe_kernel<<<1, 1, 0, s1>>>(&fh->spinning, &fh->release, capB, &fh->saw, &fh->polls,
                                  &fh->release_seen, &fh->probe_done);
    fh->release = 1u;   // safe here: the probe cannot start before the spinner has finished
    (void)hipStreamSynchronize(s1);
    const int control_saw = (int)fh->saw;
    printf("  negative control, both kernels on ONE stream (must be 0): saw=%d after %llu polls\n",
           control_saw, fh->polls);
    printf("    (device caps: spinner %u register iterations, probe %u polls; host wait is a bounded"
           " spin count. None of these is a duration.)\n", capA, capB);
    if (control_saw != 0) {
        *saw_out = -3;      // the probe cannot say no, so its YES means nothing
        hipStreamDestroy(s1);
        hipStreamDestroy(s2);
        hipHostFree(fh);
        return 0;
    }
    *saw_out = (completed == trials && trials > 0)
                 ? ((observed == trials) ? 1 : (observed == 0 ? 0 : -2)) : -1;
    hipStreamDestroy(s1);
    hipStreamDestroy(s2);
    hipHostFree(fh);
    return 0;
}

// Process structure. Every HIP-touching section runs in a freshly forked process and the parent never
// initialises the runtime. Two independent reasons, both learned the hard way in this repo:
//   * mixed_bw / gdn_walk: one faulting launch poisons the context and every later case in the same
//     process silently never runs, so each case gets its own child;
//   * HSA/HIP does not survive fork() once the parent holds a live context. The first draft of this
//     tool ran [a]-[c] in the parent and forked only for [d]; the child's hipHostMalloc failed with an
//     unrelated-looking error. The failure was the fork, not the call. So the parent reads sysfs only
//     and every child is forked before anyone in it has touched the device.
// Values cross back on a pipe, which makes the verdict computed from data rather than asserted by a
// child's exit code, and makes "the child never got there" distinguishable from "the value is 0".
static int run_pipe_child(int (*fn)(int, void*), void* arg, unsigned char* out, size_t n_out) {
    int fds[2];
    if (pipe(fds) != 0) { fprintf(stderr, "pipe failed\n"); return -1; }
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        int rc = fn(fds[1], arg) & 0x7f;
        fflush(stdout);
        close(fds[1]);
        _exit(rc);
    }
    if (pid < 0) { fprintf(stderr, "fork failed\n"); return -1; }
    close(fds[1]);
    size_t got = 0;
    while (got < n_out) {
        ssize_t r = read(fds[0], out + got, n_out - got);
        if (r <= 0) break;
        got += (size_t)r;
    }
    close(fds[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {}
    if (WIFSIGNALED(status)) {
        printf("  CHILD KILLED by signal %d (a fault here poisons only this child)\n",
               WTERMSIG(status));
        return 2;
    }
    if (got != n_out) {
        printf("  CHILD returned %zu of %zu payload bytes (rc=%d): the verdict inputs did not "
               "arrive, and an absent value is not a zero\n", got, n_out, WEXITSTATUS(status));
        return 3;
    }
    return WEXITSTATUS(status);
}

struct Facts {
    long hint_cap;
    long l2;
    int concurrent;
    int windows_accepted;
};

static int facts_fn(int fd, void* p) {
    (void)p;
    long persist_attr = -1, l2_attr = -1, persist_prop = -1, window_prop = -1, l2_prop = -1;
    int concurrent = -1;
    if (hip_facts(&persist_attr, &l2_attr, &persist_prop, &window_prop, &l2_prop, &concurrent)) {
        return 1;
    }
    printf("\n== [c] functional probes (does the API answer, and does anything stick)\n");
    const long sizes[] = {1L << 20, 2L << 20, 8L << 20, 16L << 20, 32L << 20, 64L << 20};
    int accepted = 0;
    for (long s : sizes) { accepted += (probe_stream_hint(s) == 0) ? 1 : 0; }
    printf("  access-policy windows hipStreamSetAttribute actually accepted: %d of %d sizes tried\n",
           accepted, (int)(sizeof sizes / sizeof sizes[0]));
    probe_other_knobs();
    Facts f{};
    f.hint_cap = persist_attr > 0 ? persist_attr : (persist_prop > 0 ? persist_prop : window_prop);
    f.l2 = l2_attr > 0 ? l2_attr : l2_prop;
    f.concurrent = concurrent;
    f.windows_accepted = accepted;
    ssize_t w = write(fd, &f, sizeof f);
    return w == (ssize_t)sizeof f ? 0 : 4;
}

struct CoArgs { unsigned capA, capB, trials; };
static int co_fn(int fd, void* p) {
    CoArgs* a = (CoArgs*)p;
    int saw = -1;
    int rc = co_residency_child(a->capA, a->capB, (int)a->trials, &saw);
    ssize_t w = write(fd, &saw, sizeof saw);
    return (rc == 0 && w == (ssize_t)sizeof saw) ? 0 : (rc ? rc : 4);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    long tier_kib = 32768;   // the tier to beat; [b] prints what KFD actually says
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tier-kib") && i + 1 < argc) tier_kib = atol(argv[++i]);
    }

    printf("P1c - HIP residency hint vs the big tier, queried from the part\n");
    printf("(no timing is taken by this tool; every line is a size, a count, or a yes/no)\n");

    print_kfd();

    printf("\n== [a]+[c] HIP facts and functional probes (fresh process)\n");
    Facts f{-1, -1, -1, 0};
    int rc1 = run_pipe_child(facts_fn, nullptr, (unsigned char*)&f, sizeof f);
    if (rc1 != 0) printf("  facts child rc=%d\n", rc1);

    printf("\n== [d] can two streams be co-resident (arm (b) feasibility, fresh process)\n");
    CoArgs ca{20000000ull, 200000u, 5u};
    int saw = -1;
    int rc2 = run_pipe_child(co_fn, &ca, (unsigned char*)&saw, sizeof saw);
    if (rc2 != 0) printf("  co-residency child rc=%d\n", rc2);

    // ---- verdict, computed from the piped values, not hardcoded -----------------------------
    const long tier_bytes = tier_kib * 1024L;
    printf("\n== VERDICT\n");
    printf("  tier to beat            : %ld KiB (%.1f MiB) [KFD level 3, cross-checked against the"
           " size-bandwidth curve in devices/strix-halo-memory-bandwidth.md]\n", tier_kib,
           tier_kib / 1024.0);
    printf("  largest hint capacity   : %ld B (%.3f MiB)\n", f.hint_cap, f.hint_cap / 1048576.0);
    printf("  stream-settable windows : %d of 6 sizes accepted\n", f.windows_accepted);
    const bool hint_reaches_tier = f.hint_cap > 0 && f.hint_cap >= tier_bytes;
    if (!hint_reaches_tier) {
        printf("  VERDICT: (M1) HARDWARE RESIDENCY HINT IS THE WRONG TOOL HERE. The persisting/"
               "access-policy capacity this part reports is %.3f MiB against a %.1f MiB tier%s, so a"
               " HIP persisting window can address at most the per-GPC L2 and cannot pin anything into"
               " the MALL-class tier the fast reads come from.\n",
               f.hint_cap / 1048576.0, tier_kib / 1024.0,
               f.windows_accepted == 0 ? ", and the setter refuses every window size outright" : "");
        printf("  CONSEQUENCE: every P1 arm must be (M2) EXPLICIT PREFETCH, a real early load."
               " hipStreamSetAttribute, hipDeviceSetCacheConfig and any hipDeviceSetLimit carve-out"
               " are not levers for this engine and no gate should be written against them.\n");
    } else {
        printf("  VERDICT: a persisting window of %.1f MiB is addressable, so (M1) must be tested as"
               " an arm beside explicit prefetch.\n", f.hint_cap / 1048576.0);
    }
    printf("  concurrentKernels prop  : %d\n", f.concurrent);
    printf("  arm (b) implementable   : %s\n",
           saw == 1 ? "yes - co-residency observed in every trial with the serialized control reading"
                      " 0, so a second-stream prefetch can overlap a running kernel"
                    : (saw == 0 ? "no - the second stream never ran while the first was resident, so a"
                                  " prefetch must be interleaved INSIDE a phase kernel, not beside it"
                                : (saw == -2 ? "SPLIT across trials: re-run, do not average a boolean"
                                  : (saw == -3 ? "INSTRUMENT BROKEN: the serialized control answered"
                                                 " 1, so this probe cannot distinguish overlap from"
                                                 " serialization. Fix the probe, not the verdict."
                                                : "unknown, re-run [d]"))));
    printf("  (byte sizing for every target is in P1-ARITHMETIC.md; nothing here is a timing)\n");
    return 0;
}
