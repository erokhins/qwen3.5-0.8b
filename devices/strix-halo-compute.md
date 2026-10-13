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
