# Strix Halo memory bandwidth, measured (2026-10-10)

Framework Desktop, AMD Ryzen AI Max+ 395. 128 GB LPDDR5x-8000, 256-bit, shared by the
CPU and the Radeon 8060S iGPU (`gfx1151`). ROCm 7.1 / HIP 7.15, clang 23.
**The two processors draw from one pool**, so CPU and GPU bandwidth numbers are
alternatives, not additive; measure with the machine otherwise idle.

Two probes, both in `tools/`:

| probe | what it does | command |
|---|---|---|
| `tools/bw_probe.py` | CPU-side 6 GiB read-only reduction + copy, thread sweep | `.venv/bin/python tools/bw_probe.py --threads 4,8,16,32` |
| `tools/bw_probe_hip.cpp` | iGPU grid-stride read + copy sweep over blocks × threads, plus H2D/D2H | `hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/bw_probe_hip tools/bw_probe_hip.cpp && /tmp/bw_probe_hip 1.0 4` |

## Results

| path | read | copy (r+w) | notes |
|---|---|---|---|
| **iGPU, 1 GiB** | **240 GB/s** (best; 233–240 across a 20–640 block × 128–1024 thread sweep) | **210 GB/s** | plateaus from ~80 blocks up; 1 GiB buffers |
| **iGPU, 4 GiB** | 235 GB/s | 203 GB/s | size barely matters — no cache to hide behind |
| iGPU H2D / D2H | 76–82 / 63–70 GB/s | – | any staging path through the host is bounded here |
| CPU, torch reduce | 75–117 GB/s (best ~117 at 16 threads) | 59–101 GB/s | run-to-run variance is large; re-measure before quoting |

Other facts the probe printed: `hipGetDeviceProperties` reports **20 CUs** and
**2 MiB L2** for the 8060S, and the visible device memory is a **110 GiB** carve-out,
so 4 GiB allocations succeed without `hipMallocManaged`.

### The CU count, settled — HIP says 20, `rocminfo` says 40, and the difference matters

`rocminfo` prints `Compute Unit: 40` for the gfx1151 agent, while `hipGetDeviceProperties`
and `hipDeviceGetAttribute(hipDeviceAttributeMultiprocessorCount)` both say **20**. The number
a kernel is sized against is the HIP one — that is what a launch grid is scheduled on — and any
occupancy argument written against 40 is off by a factor of two. Settled with
`tools/device_query.cpp` (built with the usual `-O3 -Wl,-rpath,/opt/rocm/lib`), which prints
both sources side by side rather than asking the reader to trust a transcript:

| fact | value | why it matters here |
|---|---|---|
| `multiProcessorCount` | **20** | the divisor in every "is the grid big enough" argument |
| `maxThreadsPerMultiProcessor` | 2048 | 20 × 2048 = **40960 resident threads**; `warpSize` 32 ⇒ 64 waves/CU |
| `maxBlocksPerMultiProcessor` | 64 | block-count ceiling per CU is not the binding limit at 256-thread blocks |
| `l2CacheSize` | **2.0 MiB** | why a 22 MB layer beats the tile trick and why a 42/84 MiB working set genuinely defeats the cache |
| `totalGlobalMem` / `hipMemGetInfo` | 110 GiB / **34.2 GiB free** | the two numbers in the unified-memory note: 110 GiB is the nominal carve-out, ~34 GiB is what is actually allocatable today |
| `asyncEngineCount` | 8 | copy engines exist, so H2D can overlap compute in principle |
| `concurrentKernels` | **1** | but the driver claims it will not run two kernels at once — relevant to any plan that overlaps a small op with a big GEMV; re-check before relying on it either way |

One consequence recorded here because it was a wrong argument in two other files: `down_proj`
at `rows/block = 8` issues 128 blocks × 256 threads = 32768 threads, which fits inside the
40960 resident-thread capacity, so its 12 % bandwidth shortfall versus `lm_head_gemv` is **not**
a block-count under-fill and remains unexplained. `prototype/mlp_gemv/results.md` and
`bench/performance-status.md` carried the 40-CU version of that claim.

## What this means for the decode roofline

Ceiling = bandwidth / bytes-per-token. §01.7 + §00.5 give **1.565 GB/token** for
bf16 Qwen3.5-0.8B single stream (weights 1.5457 + GDN state 18 MiB + conv 0.63 MiB).
At the measured iGPU read bandwidth of **237 GB/s**:

| configuration | GB/token | ceiling @237 GB/s |
|---|---|---|
| bf16 single stream | 1.565 | **151 tok/s** |
| bf16, 4k context | 1.616 | 147 |
| bf16, 32k context | 1.968 | 120 |
| 4-bit blocks, bf16 logits table | 1.170 | 202 |
| 4-bit everywhere | 0.977 | 243 |
| 3-bit everywhere | 0.838 | 283 |
| MTP k=1 at accept 0.58 | 1.312 | 181 |
| MTP k=2 at accept 0.41 | 1.461 | 162 |
| batch 4 / 8 / 16 (aggregate) | — | 575 / 1082 / 1936 tok/s (144 / 135 / 121 per request) |

Two corrections to what §07.4 previously assumed:

1. The old "210–215 GB/s attainable" figure is **closer to a copy/STREAM-style number**
   here — a pure read stream actually reaches ~237 GB/s, so the single-stream bf16
   ceiling is ~151 tok/s rather than 135–140.
2. A CPU-resident engine would have capped at ~50–75 tok/s (75–117 GB/s / 1.565), which
   is a further argument for the GPU path beyond the compute argument.

Caveats worth remembering: the read kernel is a float4 grid-stride sum, so it is a fair
but not necessarily optimal access pattern; the L2 is only 2 MiB, so nothing about a
1.5 GB weight stream is cache-friendly; and the iGPU shares the pool with the CPU, so a
Python driver thread touching memory during decode steals bandwidth from the kernel.
Re-run both probes after any change to the memory carve-out (`ConfigInfo`/`rmem`) or
BIOS memory settings.

## 0b. There *is* something cache-friendly: the decode state

`hipGetDeviceProperties` reports `l2CacheSize = 2 MiB` and that is the whole story for the
weight stream — but a working set of ~18 MiB (all 18 GDN recurrent states, touched every
decode step) behaves like it is cached. The same kernel and access pattern measured
**918 GB/s at an 18 MiB working set and 255 GB/s when tiled to 108 MiB**
(`prototype/gdn_decode`, see its results.md). Some larger cache the property query does not
report is holding it.

Practical consequences: state-carrying traffic is ~4× cheaper than a cold read predicts, and
any bandwidth result in the 240–950 GB/s range must be re-measured with the working set
tiled past ~64 MiB before it is called a bug or believed as a roofline.

## 0c. Kernel launch floor (1.8 µs) and why HIP Graphs do not fix it

`tools/launch_overhead_hip.cpp`, 1000–3000 iterations:

| measurement | result |
|---|---|
| empty kernel, back-to-back | **1.75–2.08 µs** |
| 1-block kernel writing one float | **1.85 µs** |
| same 64 launches captured into one graph, replayed | 114.05 µs = **1.78 µs/kernel (1.0×)** |

Graphs buy nothing, so the floor is the GPU's per-kernel dispatch cost rather than the host
submit path. A decode step is ~200 kernel issues, so the floor is **~0.36 ms on top of the
~6.5 ms of bandwidth** (5–6 %) — and unlike bandwidth, it is paid per *small* op, which is
where it would actually bite: a 0.09 GB attention projection cannot hide 1.8 µs of dispatch.
The lever is fusion (fewer, bigger kernels: fuse norms/gating into the GEMV epilogue), not
graph capture. That is a change to the usual advice, so the probe exists to re-check it on
any other ROCm part.
