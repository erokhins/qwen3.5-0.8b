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
