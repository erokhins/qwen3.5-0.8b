# Measured dense-matrix (compute) ceiling — `gfx1151` / AMD Radeon 8060S

Every other number in `devices/` is about moving bytes. This file is about doing arithmetic, because the
prefill-throughput question is not bandwidth-bound and cannot be argued from the bandwidth probe.

Probe: `tools/gemm_peak.cpp` (`tools/build.sh gemm_peak`), which calls **hipBLAS** `hipblasGemmEx`
(bf16 in, f32 accumulate, `HIPBLAS_GEMM_DEFAULT`) — i.e. it measures the vendor's matrix-unit kernels, which
is the ceiling any hand-written GEMM is competing against, not a hand-written FMA loop. `--check` compares a
64×96×128 product against a CPU reference and printed `SELFCHECK ok maxrel=0.000e+00`, so the product being
timed is the product we think it is.

## The rows (2026-10-13, window `gemm-peak`, blocking, zero subagents, util 40 % at acquire)

Device as reported: `AMD Radeon 8060S Graphics`, **CUs=20**, `clock_MHz=2900`, `mem_clock_MHz=1000`,
`bus_width=256`. Best of 15 after a warm call, `steady_clock` (AGENTS.md: `hipEventElapsedTime` is not
trustworthy below ~1 ms, and three of these rows are near it).

| tag | M×N×K | GFLOP | best ms | **TFLOP/s** |
|---|---|---|---|---|
| `square_4096` | 4096×4096×4096 | 137.4 | 29.250 | **4.70** |
| `square_2048` | 2048×2048×2048 | 17.2 | 3.511 | **4.89** |
| `wide_8192` | 8192×4096×4096 | 274.9 | 58.950 | **4.66** |
| `engine_q_shape` | 1024×4096×1024 | 8.6 | 1.439 | **5.97** |
| `engine_gate_shape` | 1024×3584×1024 | 7.5 | 1.295 | **5.81** |
| `decode_gemv_shape` | 1×4096×1024 | 0.008 | 0.071 | **0.12** (118 GB/s of traffic) |

Three things to take from this, in order of how much they matter:

1. **Dense bf16 matrix work on this box runs at ~4.7 TFLOP/s on big squares and ~5.9 TFLOP/s at the shapes the
   engine actually issues.** The shaped rows being *faster* than the squares is not surprising — at `M=1024,
   K=1024` the whole working set is 6 MiB and the tile shapes are friendlier — but it means the relevant
   ceiling for prefill is the ~5.8–6.0 number, not the 4.7 one.
2. **The one-row shape is not compute at all**: 0.12 TFLOP/s with 118 GB/s of traffic is a launch-and-latency
   bound shape. This is the numeric statement of why decode is a bandwidth problem and prefill is a compute
   problem, and it is the same asymmetry the whole engine is built around.
3. Nothing here is a token rate, and the spec sheet's TFLOPS figure is deliberately not quoted: it is not a
   measurement, and this file only reports measured rows.

## What this says about a prefill target (arithmetic, not measurement)

A token through this model costs **~1.5–1.6 GFLOP** of dense matmul: 0.773 B matmul parameters (the 0.53 B of
block matrices plus the 0.243 B tied `lm_head`) times two flops per multiply-accumulate, with attention
negligible at these context lengths. Dividing the measured ceiling by that:

| sustained fraction of the shaped ~6 TFLOP/s ceiling | **prompt tok/s** |
|---|---|
| 100 % (unreachable) | ~3,700 |
| 70 % | ~2,600 |
| 60 % | ~2,240 |
| 50 % | ~1,870 |
| **today's engine (971 tok/s at a 1024-token prompt ⇒ 1.55 TFLOP/s)** | **~26 % of the ceiling** |

So the operator's ~5,000 prompt tok/s would require **8 TFLOP/s sustained, which is more than this GPU can do
in total** — the target is above the hardware, not merely above the current kernel, and no amount of tiling,
fusion or batching moves it. The defensible target is **~2,000–2,600 prompt tok/s** (50–70 % of the shaped
matrix peak), which is a **2–2.7× improvement** over the current 971 and is the thing worth funding: today's
prefill reaches about a quarter of what one hipBLAS call at the same shape reaches, and the gap is the usual
suspects (no MFMA tiling in the engine's projections, wave rows capped at 512 where `M=1024` is both faster
per flop and fewer weight passes, and per-row weight traffic across waves).

Two honest caveats. First, "1.6 GFLOP/token" is a parameter count times two, not a measurement of the engine's
own flop accounting; if the engine's real work per token is higher (padding, the MTP draft block, a wave grid
that rounds a prompt up), the achievable tok/s is proportionally lower, and a flop-accounting audit belongs in
the prefill package before anyone commits to a number. Second, the ceiling above is for *one* big GEMM; the
model's projections are 10–12 smaller products per layer with different shapes, and a real kernel plan has to
show what it reaches at *those* shapes — `engine_gate_shape` is one of them and it is fine, but `k/v_proj`
(N=256) is a shape this probe has not touched and is the one most likely to be bad.

## Correction (same day): "above the hardware" was too strong, and the CU count I printed was wrong

Two errors in the page above, both mine, both found by the operator asking for a spec cross-check instead of
taking the measurement at face value.

**1. `CUs=20` is a reporting artifact, not the device.** The probe printed `hipGetDeviceProperties`
`multiProcessorCount = 20`; `rocminfo` on the same box reports `gfx1151 … Compute Unit: 40, SIMDs per CU: 2`,
and AMD's spec for the Ryzen AI Max+ 395 says the Radeon 8060S has **40 CUs at up to 2.9 GHz** ([AMD product
page](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html),
[TechPowerUp](https://www.techpowerup.com/gpu-specs/radeon-8060s.c4270)). So HIP is enumerating this SKU at
half the physical CU count (or pairing them — `SIMDs per CU: 2`), and either way **20 must not be used for
any roofline arithmetic here**. Quote 40 CUs with `rocminfo` as the source.

**2. That changes what the measured ceiling means.** At 40 CUs × 2.9 GHz × 128 FP32 lanes the advertised dense
rate is **14.85 TFLOPS FP32, and 14.85 TFLOPS FP16 at 1:1** (TechPowerUp lists FP16 = FP32 for this part;
AMD's "up to 60 TFLOPS AI" is the 4× marketing number). My measured **4.7–6.0 TFLOP/s is therefore 32–40 % of
the advertised dense rate**, not the machine's limit. Three candidate reasons, none of them yet tested:

- **library immaturity on this SKU** — ROCm's `gfx1151` support is publicly reported as incomplete
  ([ROCm issue 6348](https://github.com/ROCm/ROCm/issues/6348),
  [rocm-libraries issue 4618](https://github.com/ROCm/rocm-libraries/issues/4618): "gfx1151 support is still
  broken in practice as of February 2026"), so hipBLAS may simply be picking poor kernels here;
- **sustained clocks** — this is a 120 W part whose iGPU shares that budget with 16 Zen 5 cores and a
  bandwidth-hungry LPDDR5x fabric; if a big GEMM sustains ~1.0–1.2 GHz rather than 2.9 GHz, 4.7 TFLOP/s is
  *exactly* what the silicon is allowed to do, and no kernel work fixes it;
- **bf16 specifically** — if the matrix units run bf16 below fp16 on RDNA3.5, an fp16 path would be faster at
  the same precision class.

**What this does to the prefill target.** It re-opens it. 5,000 prompt tok/s needs ~8 TFLOP/s sustained, which
is **54 % of the advertised dense rate** — that is not "above the hardware" as the paragraph above claimed; it
is "above what hipBLAS sustains on this box today", which is a much weaker statement and could well be a
driver problem rather than a wall. The honest position is: **5k is not excluded, ~2.0–2.6k is what a
well-tuned kernel has demonstrable room to reach, and the discriminator is a device window** that measures
(i) fp16 vs bf16 vs fp32 at the same shapes, (ii) the sustained SM clock during the GEMM, and (iii) what a
hand-tiled `rocwmma`/MFMA kernel reaches — `rocwmma.hpp` is installed on this box, so that test is available.
The claim that stands regardless: today's prefill at **1.55 TFLOP/s is ~10 % of even the FP32 vector rate and
~26 % of what one hipBLAS call at the same shape gets**, so the gap is structural (no matrix-unit tiling in
the engine's projections) whether or not 5k is ultimately reachable.

## Does precision buy compute on this part? Measured: yes, ~22–26 % — fp16 beats bf16

Same probe, `--dtypes bf16,fp16,fp32` (hipBLAS, f32 accumulate in all three, selftest green), window
`dtype2`, blocking, zero subagents, best of 25:

| shape | bf16 | **fp16** | fp32 | fp16 vs bf16 |
|---|---|---|---|---|
| `square_4096` | 4.758 | **5.976** | 2.952 | **+25.6 %** |
| `engine_q_shape` (1024×4096×1024) | 5.667 | **6.885** | 2.932 | **+21.5 %** |

Three things follow, and the first is the one that reframes the whole prefill argument.

1. **fp16 is faster than bf16 on this GPU's matrix units, at identical accumulate precision.** That is the
   opposite of the usual intuition — and the reason is that bf16 and fp16 are the *same width*, so nothing is
   saved in bandwidth; the difference is purely in how fast the matrix units chew each format. For a
   compute-bound path like prefill, choosing fp16 is not a precision sacrifice at all: fp16 has **10 mantissa
   bits against bf16's 7**, so at the magnitudes this model works at (weights ~0.01, logits ~±20) fp16 is
   strictly *more* accurate than the bf16 the checkpoint ships in. It is a narrower format too (5 exponent
   bits, max 65504), so the honest caveat is that a bf16→fp16 weight conversion needs a range check rather
   than an assumption — values outside fp16's range, and very small ones that would go subnormal, have to be
   counted before anyone calls the conversion exact.
2. **fp32 lands at ~2.93 TFLOP/s, essentially half the fp16 rate**, which is what you would expect if the fp32
   path is running the matrix units at half rate or falling back to the vector path. Either way it bounds what
   our current kernels could reach *without* changing formats: the engine does bf16→f32 conversions and scalar
   f32 FMAs, and 2.93 TFLOP/s is the ceiling of that style of code, not of the engine's tiling.
3. **This does not retire the structural gap.** At the engine's own shape one hipBLAS call gets 5.7 (bf16) to
   6.9 (fp16) TFLOP/s while the engine's whole prefill achieves 1.55 TFLOP/s in aggregate. Format choice is
   worth ~1.25×; matrix-unit tiling is worth ~4×. They are independent and the second is much bigger.

**Sustained clocks, because the spec sheet number was raised at me and deserved a direct answer.**
`/sys/class/drm/card1/device/pp_dpm_sclk` reads `0: 600Mhz 1: 2206Mhz * 2: 2900Mhz` — the GPU sits at
**2.206 GHz of a 2.9 GHz maximum** in ordinary operation, so ~24 % of the advertised peak is gone to clock
state before any kernel runs. `rocm-smi --showclkfrq` is useless here (`Clock [fclk] … exists but EMPTY!
Likely driver error!`), which is why this file quotes sysfs. So: my measured 4.76–6.89 TFLOP/s against the
14.85 TFLOPS advertised is ~32–46 %, and part of that shortfall is clock state rather than kernel quality.
5k prompt tok/s still needs ~8 TFLOP/s sustained, which is now **within sight of the fp16 rate at the engine's
own shape (6.89)** rather than science fiction — it needs a good tiling *and* fp16 *and* the clocks to stay up.
