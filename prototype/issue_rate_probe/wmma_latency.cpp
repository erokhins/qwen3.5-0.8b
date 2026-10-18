// wmma_latency.cpp — the WMMA LATENCY leg that `prototype/attn_wmma/fattn_score_handover.md` §5e names,
// built in the directory that owns rate measurements. WRITTEN, NOT RUN: a worker takes no timing
// (AGENTS.md, "GPU execution is serialized"), so the main agent runs `run_wmma_latency_window.sh`
// blocking, under `tools/gpu_lock.sh`, with zero subagents active. NOTHING here has been executed on
// device, and no clock figure in this file is a measurement — the two denominators are hypotheses the
// leg exists to replace.
//
// WHAT IT MEASURES, AND WHY THE NUMBER IS NOT ALREADY IN THE LEDGER
// `prototype/issue_rate_probe/results.md` §1 has one WMMA row: `v_wmma_f32_16x16x16_f16` at **88.8 % of
// its class roof (52.63 of 59.36 TFLOP/s)**, taken at 32 waves/CU with four INDEPENDENT accumulator
// chains — deliberately un-latency-limited, which is correct for a roof and useless as a latency. The
// census currency every WMMA price in this tree is built from is instead **32 clocks of matrix-pipe time
// per instruction** (`fattn_census.py`'s `WMMA_CLK`: 4096 MAC / (32 lanes x 4 MAC/lane/clk)), and that is
// what makes `m16_fattn_score`'s 9.719 pipe-weighted slots/touch what it is. **32 clk is a class roof,
// not a measured latency, and nobody has timed a dependent WMMA on this part.** The recorded cost of
// pricing a serial kernel by its issue slots at an occupancy it never realises is
// `prototype/gdn_collapse/results.md` §4-§5: priced 11.4 ms, measured 112.04 ms, 9.8x, because the
// currency was wrong. A fattn round IS a serial kernel — the score tile feeds the fold feeds the P·V tile
// feeds the online (m, l, acc) rescale, and `prototype/attn_wmma/evidence/occupancy-fattn.txt` records the form at 16
// wavefronts/CU of the 32 the part allows, so four wavefronts per SIMD is what has to hide the latency.
//
// WHY BOTH ACCUMULATOR WIDTHS, IN ONE LEG
// The open question is whether the engine would use `v_wmma_f16_16x16x16_f16`. Two facts are already
// checkable from the toolchain and neither is a rate:
//   * The F16 accumulator does **not** save registers. rocwmma's gfx11 path declares
//     `ARegsT = VRegF32x8` / `CRegsT = AccRegF32x8` for `f16 x f16 -> f16`
//     (`/opt/rocm/core-10.0/include/rocwmma/internal/wmma_impl.hpp:320-360`) and this file's own listing
//     emits `v_wmma_f16_16x16x16_f16 v[1:8], v[9:16], v[17:24], v[1:8]` — an 8-VGPR D block with `C == D`,
//     the same width the F32 accumulator uses. The `.amdhsa` metadata agrees: `wl_f16_i{1,2,4,8}` cost
//     vgpr 34 / 33 / 51 / 92 against `wl_f32_i{1,2,4,8}` at 25 / 33 / 49 / 81. The builtin's fourth
//     argument is the D `op_sel` (rocwmma's `WmmaCtrlFlags`: `LOW = bits[15:0]`, `HIGH = bits[31:16]`,
//     `wmma_impl.hpp:129-135`), so the two forms are two views of one register block and the F16 form's
//     economy is on the DRAIN — P stays 16-bit through the fold instead of being widened to F32 and
//     re-narrowed by `p_to_operand` — not on the accumulator footprint. What it pays in precision is 10
//     mantissa bits per accumulate instead of 24, and whether that is free depends entirely on the rate,
//     which is what this leg measures.
//   * We have never emitted the F16 accumulator form at all in this repository: every WMMA row in
//     `prototype/issue_rate_probe/results.md` §1 and every kernel in `prototype/attn_wmma/` is the F32-accumulator form.
// So: same geometry, same window, both widths, four ILP depths each, and a loop-overhead baseline to
// subtract. A measured difference is the only honest basis for the decision.
//
// AND A NUMBER THIS REPLACES, NOT CONFIRMS
// `prototype/attn_wmma/fattn_census.py:99-100` carries `WMMA_CLK = 32` (class roof) and
// `WMMA_CLK_MEASURED = 36`. The 36 is the SUSTAINED rate converted (3.552 MAC/lane/clk from a throughput
// leg with independent chains), not a dependent chain: nothing in this tree has ever timed a WMMA whose own
// C is its predecessor's D. Do not read the ILP=1 row here as a check on the 36; it is a different
// quantity, and if it comes back near 32 the census is not vindicated, it is just lucky.
//
// "LATENCY", OPERATIONALLY
// One accumulator chain issued back to back — `acc = wmma(A, B, acc)` reading and writing the same acc,
// so instruction i+1 cannot start until i's D has landed. `--ilp k` interleaves k independent chains, so
// the SAME kernel sweeps from pure latency (k = 1) to the issue-limited roof (large k) and the result is
// the CURVE, not a number; the row where `clk_per_wmma` stops falling is the pipeline depth, and the k = 1
// row is the latency. A/B fragments are read into registers once, before the timed region, so the timed
// loop is wmma instructions and nothing else — no LDS, no global, no address arithmetic. `wl_empty_i1` is
// the same loop with the wmma replaced by one FADD, which is the loop-overhead baseline the wmma rows are
// read against; quoting a wmma row without subtracting the `wl_empty` row is quoting loop overhead.
//
// THE GEOMETRY, and why it is NOT the rate probe's
// The rate legs run 32 wavefronts/CU to saturate. This one defaults to low occupancy on purpose, because
// the shipping shape is low-occupancy: 160 blocks x 128 threads = 640 wavefronts over the KFD's 40 CUs =
// **16 wavefronts/CU**, the figure `prototype/attn_wmma/evidence/occupancy-fattn.txt` records for `m16_fattn_score`. That is
// where the GDN lesson bites — a serial kernel priced by issue slots at an occupancy it never realises was
// measured 9.8x wrong (`prototype/gdn_collapse/results.md` §4-§5). Every row prints wavefronts/CU and
// wavefronts/SIMD, the driver's blocks/CU for that exact kernel, VGPRs, LDS and private bytes, so a row
// that accidentally turned into a saturation test is visible as one. Read the driver column with the
// half-SKU trap in mind: `hipOccupancyMaxActiveBlocksPerMultiprocessor` answers per the HIP-visible CU (20
// on this SKU), not the KFD's 40 (AGENTS.md, the counter/occupancy row), which is why both the driver
// answer and the derived count are printed. `--clock-mhz` defaults to the divisor
// `prototype/issue_rate_probe/README.md` books, and the window prints
// the live `sclk` beside the rows: 2894 is a MAX clock and ordinary operation measured 2.206 GHz
// (`devices/strix-halo-compute.md`), so a clocks figure is only as good as its divisor.
//
// WHY THE BILL IS A SLOPE, NOT A DIVISION
// There is no single-point mode in this tool. A launch carries a fixed residue the leg is not about —
// dispatch (~2-6 us here, and bimodal: `bench/performance-status.md` §6), the fragment loads, the store,
// and first touch (`devices/strix-halo-unified-memory.md`: the first GPU pass over fresh pages runs ~35
// GB/s, 7x slow) — and none of it depends on `--iters`. Dividing one burst by its iteration count charges
// all of that residue to the instruction, which is the same mistake as the flat per-kernel ISA census that
// billed once-per-launch addressing overhead once per token (AGENTS.md, D60: flat +156 slots/token, true
// per-token −47). So every leg is timed twice, at `--iters-lo` and `--iters-hi`, and the bill is
// `(t_hi - t_lo) / (iters_hi - iters_lo)`, which cancels every per-launch constant by construction.
// The cancelled residue is printed as `fixed_us_per_launch` rather than assumed small, and the doubling is
// itself a witness: a leg that does not get slower with a longer chain is not measuring a chain, so a
// non-positive slope is REFUSED, not reported as a fast number.
//
// THE WITNESSES — a row is REFUSED, not printed, if any fails
//  1. CALIBRATED EXACTNESS, not an assumed fragment width. Before timing, the leg runs ONCE with
//     `iters = 1` and reads every thread's sum back: that is one wmma's contribution, measured, with no
//     assumption about how many elements a D fragment holds per lane (which is exactly the thing the map
//     probe found the transcription wrong about, so it is not assumed here). The timed run must then give
//     `iters x` that contribution, per thread, exactly, at BOTH timed chain lengths — so a chain the
//     compiler deleted, hoisted, shortened, or turned into a multiply is caught by the count, and a chain
//     that never ran gives 0 and is refused. Arithmetic stays exact by construction: A and B are all
//     `--scale` = 0.5, a power of two, so each element grows by exactly 16 x scale^2 = 4 per step; the row
//     is REFUSED once that value leaves the accumulator's exact range (2^24 for F32, 2048 for F16 — f16
//     integers are exact to 2048), rather than printing a rounded number as a latency. That F16 range is
//     what caps `--iters-hi` at 512 with the default scale, and the cap is enforced per leg, not assumed.
//     The calibration run also reports `elems_per_lane`, the measured number of D slots a lane owns, which
//     is deliberately not hard-coded: the map probe is the reason not to trust a transcription of it.
//  2. THE INSTRUCTION IS THERE. `build_wmma_latency.sh` fails unless the listing carries both
//     `v_wmma_f32_16x16x16_f16` and `v_wmma_f16_16x16x16_f16`, at least once per timed kernel.
//  3. THE ACCUMULATORS ARE IN REGISTERS. `local_spilled` must be 0, or the leg is measuring
//     private-memory latency: the recorded `#pragma unroll 1` failure moved 128 accumulator registers to
//     scratch (544 B/thread) while clang's own remark still printed "VGPRs Spill: 0"
//     (`prototype/attn_wmma/fattn_score_handover.md` §1). The row prints `local` whatever it is.
//  4. EVERY READ-BACK SLOT PARSED and every thread agreeing. A check whose input list can be empty cannot
//     fail (`prototype/gdn_collapse/run.py`), so all `blocks x threads` slots are required and compared
//     against each other as well as against the calibration.
//
// WHAT IT IS NOT. Not a throughput row (that exists, at 32 waves/CU). Not a numerics admission: the F16
// accumulator's rounding is only *bounded* here (exact-range refused), never judged against the pins. And
// per AGENTS.md no figure from this tool enters `bench/` or `prototype/issue_rate_probe/results.md` except through a main-agent
// window with the lock held and zero subagents.
//
// BUILD: bash prototype/issue_rate_probe/build_wmma_latency.sh
// RUN:   bash prototype/issue_rate_probe/run_wmma_latency_window.sh   (main agent only)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define IRP_CK(x)                                                                        \
    do {                                                                                 \
        const hipError_t e_ = (x);                                                       \
        if (e_ != hipSuccess) {                                                          \
            std::printf("IRP-ERROR %s at %s:%d\n", hipGetErrorString(e_), __FILE__,         \
                        __LINE__);                                                       \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

using h16 = _Float16 __attribute__((ext_vector_type(16)));   // 16 f16 = 8 VGPRs: one A/B operand tile
using f8 = float __attribute__((ext_vector_type(8)));        // F32 accumulator: 8 VGPRs
using u8 = unsigned __attribute__((ext_vector_type(8)));     // F16 accumulator: the same 8 VGPRs

namespace {

// The two accumulator widths, spelled once. The F16 builtin's fourth argument is the D `op_sel`
// (rocwmma: LOW = bits[15:0]); `false` is LOW, the value rocwmma itself passes on gfx11.
__device__ __forceinline__ f8 wmma_f32(const h16 a, const h16 b, const f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ u8 wmma_f16(const h16 a, const h16 b, const u8 c) {
    const h16 ch = *reinterpret_cast<const h16*>(&c);
    const h16 dh = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, ch, false);
    return *reinterpret_cast<const u8*>(&dh);
}

// Sum one accumulator into a double. Every element is summed, so the witness does not need to know how
// many of the D block's slots are meaningful — the calibration says, and `elems` is printed as a fact.
__device__ __forceinline__ double sum_acc(const f8 v) {
    double s = 0.0;
#pragma unroll
    for (int e = 0; e < 8; ++e) s += (double)v[e];
    return s;
}
__device__ __forceinline__ double sum_acc(const u8 v) {
    const h16 h = *reinterpret_cast<const h16*>(&v);
    double s = 0.0;
#pragma unroll
    for (int e = 0; e < 16; ++e) s += (double)h[e];
    return s;
}

// One leg. `ILP` independent chains, each `iters` deep, every step dependent on its own accumulator. A
// and B are read once, before the loop, so the timed body is the wmma stream plus loop overhead (which
// `wl_empty_i1` prices separately). `out` gets one double per thread.
#define IRP_LAT_LEG(name, ILP, ACC, STEP)                                        \
    extern "C" __global__ void __launch_bounds__(128) name(                      \
        const __half* __restrict__ frag, double* __restrict__ out, const int iters) { \
        const int lane = threadIdx.x & 31;                                       \
        const h16 A = *reinterpret_cast<const h16*>(frag + (size_t)lane * 16);   \
        const h16 B = *reinterpret_cast<const h16*>(frag + (size_t)(lane + 32) * 16); \
        ACC acc[ILP];                                                            \
        _Pragma("unroll")                                                        \
        for (int j = 0; j < (ILP); ++j) acc[j] = ACC{};                          \
        for (int i = 0; i < iters; ++i) {                                        \
            _Pragma("unroll")                                                    \
            for (int j = 0; j < (ILP); ++j) acc[j] = (STEP);                     \
        }                                                                        \
        double s = 0.0;                                                          \
        _Pragma("unroll")                                                        \
        for (int j = 0; j < (ILP); ++j) s += sum_acc(acc[j]);                    \
        out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;                   \
    }

IRP_LAT_LEG(wl_f32_i1, 1, f8, wmma_f32(A, B, acc[j]))
IRP_LAT_LEG(wl_f32_i2, 2, f8, wmma_f32(A, B, acc[j]))
IRP_LAT_LEG(wl_f32_i4, 4, f8, wmma_f32(A, B, acc[j]))
IRP_LAT_LEG(wl_f32_i8, 8, f8, wmma_f32(A, B, acc[j]))
IRP_LAT_LEG(wl_f16_i1, 1, u8, wmma_f16(A, B, acc[j]))
IRP_LAT_LEG(wl_f16_i2, 2, u8, wmma_f16(A, B, acc[j]))
IRP_LAT_LEG(wl_f16_i4, 4, u8, wmma_f16(A, B, acc[j]))
IRP_LAT_LEG(wl_f16_i8, 8, u8, wmma_f16(A, B, acc[j]))
// The loop-overhead baseline: identical shell, one FADD where the wmma was. Every wmma row is read as
// (wmma row - this row), so quoting a wmma row without it quotes loop overhead as matrix time.
IRP_LAT_LEG(wl_empty_i1, 1, f8, acc[j] + 1.0f)

struct Leg {
    const char* name;
    void (*fn)(const __half*, double*, int);
    int ilp;
    const char* acc;      // "f32" | "f16" | "none"
};

const Leg kLegs[] = {
    {"wl_f32_i1", &wl_f32_i1, 1, "f32"},   {"wl_f32_i2", &wl_f32_i2, 2, "f32"},
    {"wl_f32_i4", &wl_f32_i4, 4, "f32"},   {"wl_f32_i8", &wl_f32_i8, 8, "f32"},
    {"wl_f16_i1", &wl_f16_i1, 1, "f16"},   {"wl_f16_i2", &wl_f16_i2, 2, "f16"},
    {"wl_f16_i4", &wl_f16_i4, 4, "f16"},   {"wl_f16_i8", &wl_f16_i8, 8, "f16"},
    {"wl_empty_i1", &wl_empty_i1, 1, "none"},
};

struct Cfg {
    std::string only;
    int blocks = 160;             // 160 x 128 = 640 wavefronts = 16 per CU (KFD 40 CUs), the residency recorded in
    int threads = 128;            // 4 warps: the shipped block shape
    int iters_lo = 128;           // low point of the timed pair: wmmas per accumulator per launch
    int iters_hi = 512;           // high point. THE PAIR, NOT ONE COUNT: see the slope note below.
    int launches = 20;            // timed launches per rep (the rate probe's amortisation, same reason)
    int reps = 5;                 // best-of-N; the tool never prints a mean
    int cu_count = 40;            // KFD, not HIP: hipGetDeviceProperties reports 20 CUs on this SKU, half
                                  // the architecture, and maxThreadsPerMultiProcessor is off by the same
                                  // factor (AGENTS.md, the counter/occupancy row; the real figures come
                                  // from /sys/.../kfd/topology/nodes/1/properties via
                                  // tools/kernel_occupancy_probe.cpp).
    double clock_mhz = 2894.0;    // the divisor prototype/issue_rate_probe/README.md books (a MAX
                                  // clock; the window prints live sclk, and ordinary
                                  // operation measured 2.206 GHz per devices/strix-halo-compute.md)
    double scale = 0.5;           // A and B magnitude: 0.5 keeps every accumulator value an exact integer
    int simd_per_cu = 2;          // KFD `simd_per_cu 2`, `simd_count 80` over 40 CUs — NOT the 16 that
                                  // clang's -Rpass-analysis "Occupancy [waves/SIMD]" remark prints, which
                                  // is a constant for every kernel in the TU (AGENTS.md, same row).
};

}  // namespace

int main(int argc, char** argv) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::printf("IRP-ERROR %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--only") c.only = val("--only");
        else if (a == "--blocks") c.blocks = atoi(val("--blocks"));
        else if (a == "--threads") c.threads = atoi(val("--threads"));
        else if (a == "--iters-lo") c.iters_lo = atoi(val("--iters-lo"));
        else if (a == "--iters-hi") c.iters_hi = atoi(val("--iters-hi"));
        else if (a == "--launches") c.launches = atoi(val("--launches"));
        else if (a == "--reps") c.reps = atoi(val("--reps"));
        else if (a == "--cu-count") c.cu_count = atoi(val("--cu-count"));
        else if (a == "--clock-mhz") c.clock_mhz = atof(val("--clock-mhz"));
        else if (a == "--scale") c.scale = atof(val("--scale"));
        else if (a == "--simd-per-cu") c.simd_per_cu = atoi(val("--simd-per-cu"));
        else if (a == "--help") {
            std::printf("usage: wmma_latency [--only LEG] [--blocks 160] [--threads 128]\n"
                        "                    [--iters-lo 128 --iters-hi 512] [--launches 20] [--reps 5]\n"
                        "                    [--scale 0.5] [--cu-count 40 (KFD)] [--clock-mhz 2894]\n"
                        "                    [--simd-per-cu 2 (KFD)]\n"
                        "  legs: wl_f32_i{1,2,4,8} wl_f16_i{1,2,4,8} wl_empty_i1 (loop overhead)\n"
                        "  RAW rows only; conversion to a ledger row is a main-agent step.\n");
            return 0;
        } else {
            std::printf("IRP-ERROR unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (c.threads != 128) {
        std::printf("IRP-ERROR --threads must be 128: the leg's __launch_bounds__(128), the occupancy\n"
                    "  answer below and the wavefront arithmetic all assume that block. A silently\n"
                    "  different block size is two geometries reported as one leg.\n");
        return 2;
    }
    if (c.iters_hi <= c.iters_lo) {
        std::printf("IRP-ERROR --iters-hi (%d) must exceed --iters-lo (%d): the bill is the SLOPE between"
                    " the two points, and a zero-width pair has no slope. There is no single-point mode;"
                    " see the two-point rule in this file's header.\n", c.iters_hi, c.iters_lo);
        return 2;
    }
    // Exactness budget. With A = B = scale the per-step contribution per element is 16*scale^2 and every
    // value stays an exact integer; F32 accumulation of integers is exact below 2^24, F16 to 2048.
    const double per_step = 16.0 * c.scale * c.scale;
    const int warps = c.blocks * (c.threads / 32);
    const double waves_per_cu = (double)warps / (double)c.cu_count;
    const double wf_per_simd = waves_per_cu / (double)c.simd_per_cu;   // KFD view: 2 SIMDs per CU

    __half* dfrag = nullptr;
    double* dout = nullptr;
    const size_t frag_halves = (size_t)64 * 16;                 // one warp's A and B, in f16
    IRP_CK(hipMalloc(&dfrag, frag_halves * sizeof(__half)));
    {
        std::vector<__half> h(frag_halves, __float2half((float)c.scale));
        IRP_CK(hipMemcpy(dfrag, h.data(), frag_halves * sizeof(__half), hipMemcpyHostToDevice));
    }
    const size_t slots = (size_t)c.blocks * c.threads;
    IRP_CK(hipMalloc(&dout, slots * sizeof(double)));

    for (const Leg& leg : kLegs) {
        if (!c.only.empty() && c.only != leg.name) continue;
        const double budget = (std::strcmp(leg.acc, "f16") == 0) ? 2048.0 : 16777216.0;  // 2^24
        if (per_step * c.iters_hi > budget) {
            std::printf("REFUSED leg=%s per-element accumulator value %.1f at iters-hi exceeds the exact"
                        " %s range %.0f; lower --iters-hi or --scale (a rounded accumulator is not a"
                        " latency)\n", leg.name, per_step * c.iters_hi, leg.acc, budget);
            continue;
        }
        hipFuncAttributes attr{};
        if (hipFuncGetAttributes(&attr, (const void*)leg.fn) != hipSuccess) {
            std::printf("REFUSED leg=%s hipFuncGetAttributes failed\n", leg.name);
            continue;
        }
        int bpc = 0;
        IRP_CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&bpc, (const void*)leg.fn, c.threads, 0));
        if (attr.localSizeBytes != 0) {
            std::printf("REFUSED leg=%s local_spilled=%zu B/thread: the accumulators live in private\n"
                        "  memory, so this would measure scratch latency, not WMMA latency (the rolled-\n"
                        "  accumulator failure recorded in prototype/attn_wmma/fattn_score_handover.md §1)\n",
                        leg.name, (size_t)attr.localSizeBytes);
            continue;
        }

        // ---- witness 1, calibration: ONE wmma per accumulator, read back, so the expected timed value
        // is measured rather than derived from an assumed fragment width.
        leg.fn(dfrag, dout, 1);
        IRP_CK(hipDeviceSynchronize());
        std::vector<double> one(slots, -1.0);
        IRP_CK(hipMemcpy(one.data(), dout, slots * sizeof(double), hipMemcpyDeviceToHost));
        size_t parsed = 0, odd = 0;
        for (size_t i = 0; i < slots; ++i) {
            ++parsed;
            if (!(one[i] > 0.0)) ++odd;
            if (i && !(one[i] == one[0])) ++odd;
        }
        if (parsed != slots || odd || one[0] <= 0.0) {
            std::printf("REFUSED leg=%s calibration bad=%zu/%zu value=%.6g: one wmma per accumulator did\n"
                        "  not produce a uniform nonzero contribution, so the chain is not the chain the\n"
                        "  bill assumes (a witness that cannot fail is not a witness)\n",
                        leg.name, odd, parsed, one[0]);
            continue;
        }
        const double one_value = one[0];
        const double elems = one_value / ((double)leg.ilp * per_step);   // elements per lane, measured

        // ---- the two-point burst, which is the whole measurement.
        //
        // WHY TWO POINTS AND NEVER ONE. A launch carries a fixed residue the leg is not about: dispatch
        // (~2-6 us on this box, and bimodal — `bench/performance-status.md` §6), the A/B fragment loads,
        // the store, and the first-touch penalty (`devices/strix-halo-unified-memory.md`: the first GPU
        // pass over fresh pages runs ~35 GB/s, 7x slow). At `iters = 128` the timed body is a few
        // microseconds, so a single-point `ms / iters` would be dominated by exactly that residue and
        // would report dispatch as WMMA latency — the same class of error as the flat per-kernel ISA
        // census that bills once-per-launch overhead once per token (AGENTS.md, D60). Measuring the SAME
        // leg at two chain lengths and taking the slope cancels every per-launch constant, because none of
        // them depends on `iters`. The residue itself is printed (`fixed_us_per_launch`) so a reader can
        // see what was cancelled instead of trusting that it was small.
        //
        // The slope doubling is also the witness that the chain is real and latency-shaped: a leg whose
        // time does not grow with its own chain length is not measuring a chain, so `slope <= 0` is
        // refused rather than reported.
        const auto burst = [&](int iters, double* ms_out, size_t* bad_out) -> bool {
            leg.fn(dfrag, dout, iters);                       // warm launch, untimed
            if (hipDeviceSynchronize() != hipSuccess) return false;
            double best = 1e30;
            for (int r = 0; r < c.reps; ++r) {
                // steady_clock over a burst of launches: `hipEventElapsedTime` returns exactly 0.000 ms
                // at these sizes (bench/performance-status.md §6), and best-of-N because the box is not
                // pinned (prototype/README.md).
                const auto t0 = std::chrono::steady_clock::now();
                for (int L = 0; L < c.launches; ++L) leg.fn(dfrag, dout, iters);
                if (hipDeviceSynchronize() != hipSuccess) return false;
                const auto t1 = std::chrono::steady_clock::now();
                const double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count() / c.launches;
                if (ms < best) best = ms;
            }
            std::vector<double> got(slots, -1.0);
            if (hipMemcpy(got.data(), dout, slots * sizeof(double), hipMemcpyDeviceToHost) !=
                hipSuccess)
                return false;
            const double want = one_value * (double)iters;
            size_t bad = 0;
            for (size_t i = 0; i < slots; ++i) {
                if (!(got[i] > want * 0.9999 && got[i] < want * 1.0001)) ++bad;
            }
            *ms_out = best;
            *bad_out = bad;
            return bad == 0;
        };

        double ms_lo = 0.0, ms_hi = 0.0;
        size_t bad_lo = 0, bad_hi = 0;
        if (!burst(c.iters_lo, &ms_lo, &bad_lo) || !burst(c.iters_hi, &ms_hi, &bad_hi)) {
            std::printf("REFUSED leg=%s read-back bad=%zu@iters_lo %zu@iters_hi of %zu (calibrated step"
                        " %.6g): a timed run that is not `iters` x the calibrated step is not the chain"
                        " being billed, so no clock figure is printed\n",
                        leg.name, bad_lo, bad_hi, slots, one_value);
            continue;
        }

        const double slope_ms_per_iter = (ms_hi - ms_lo) / (double)(c.iters_hi - c.iters_lo);
        const double fixed_us_per_launch = (ms_lo - slope_ms_per_iter * c.iters_lo) * 1000.0;
        if (slope_ms_per_iter <= 0.0) {
            std::printf("REFUSED leg=%s slope=%.9f ms/iter between iters %d and %d: the leg did not get"
                        " slower with a longer chain, so it is not measuring a chain (a deleted or hoisted"
                        " accumulator, or a burst lost to the dispatch floor)\n",
                        leg.name, slope_ms_per_iter, c.iters_lo, c.iters_hi);
            continue;
        }

        // The bill. Per loop iteration every wavefront issues `ilp` wmmas, so the wall time of one
        // iteration is (wavefronts per SIMD) x ilp x clocks-per-instruction. Inverting that is the
        // per-instruction clock cost the census spends — which is the denominator `fattn_census.py`'s
        // WMMA_CLK = 32 assumed and this replaces.
        const double secs_per_iter = slope_ms_per_iter / 1000.0;
        const double clk_per_wmma =
            secs_per_iter * c.clock_mhz * 1e6 / (wf_per_simd * (double)leg.ilp);
        const double wmma_per_launch_hi = (double)warps * (double)c.iters_hi * (double)leg.ilp;
        const double tflops_hi = 2.0 * 4096.0 * wmma_per_launch_hi / (ms_hi / 1000.0) / 1e12;
        std::printf("WL kernel=%s acc=%s ilp=%d iters_lo=%d iters_hi=%d blocks=%d threads=%d"
                    " launches=%d reps=%d scale=%g | wavefronts_per_cu=%.2f wavefronts_per_simd=%.2f"
                    " driver_blocks_per_cu=%d vgpr=%d lds=%zu local=%zu elems_per_lane=%.4g"
                    " | ms_lo=%.6f ms_hi=%.6f fixed_us_per_launch=%.3f"
                    " clk_per_wmma_per_simd=%.3f Tflops_hi=%.3f wmma_per_launch=%.0f\n",
                    leg.name, leg.acc, leg.ilp, c.iters_lo, c.iters_hi, c.blocks, c.threads,
                    c.launches, c.reps, c.scale, waves_per_cu, wf_per_simd, bpc, attr.numRegs,
                    attr.sharedSizeBytes, (size_t)attr.localSizeBytes, elems, ms_lo, ms_hi,
                    fixed_us_per_launch, clk_per_wmma, tflops_hi, wmma_per_launch_hi);
        std::fflush(stdout);
    }
    return 0;
}
