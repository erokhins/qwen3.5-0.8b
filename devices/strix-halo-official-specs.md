# The box as the manufacturer describes it — official specs, transcribed, and checked against this machine

Purpose: give a future session the **vendor's own numbers** for this machine, with the original text
transcribed so it survives link rot, and a **local command beside every claim** so it can be re-checked
without a network. It exists because three findings this week turned on a device fact
(the 40-CU count, the 237 GB/s read ceiling, the 2 MiB fast tier) and none of those had an external
anchor in this repo — `devices/` recorded what *we* measured, not what AMD and Framework publish, so
there was nothing to compare a suspicious measurement against.

Accessed 2026-10-16. `amd.com` refuses scripted retrieval (the fetch aborted twice, browser-free), so
§1 is a **transcription by the operator from the open official page**, kept verbatim for the fields that
matter here; the live page must be treated as the original and re-read if a claim is load-bearing.
Everything in §2–§5 was read or run by this session and is reproducible offline.

## 1. AMD Ryzen AI Max+ 395 — transcribed from the official product page

Source: <https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html>
(AMD lists the same part on the Ryzen AI Halo board page:
<https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html>).

Fields relevant to this repository, verbatim:

```
Name                       AMD Ryzen™ AI Max+ 395
Series                     Ryzen AI Max 300 Series
Form Factor                Laptops , Desktops
Former Codename            Strix Halo
Processor Architecture     16x Zen 5
# of CPU Cores             16          Multithreading (SMT)  Yes   # of Threads  32
Max. Boost Clock           Up to 5.1 GHz        Base Clock  3 GHz
L2 Cache                   16 MB               L3 Cache     64 MB
Default TDP                55W                 AMD Configurable TDP (cTDP)  45-120W
Processor Technology       TSMC 4nm FinFET (CPU cores and I/O die)   Package Die Count  3
CPU Socket                 FP11
Supported Extensions       … AVX512 … AVX2 … FMA3 …
Max. Operating Temperature (Tjmax)  100°C
PCI Express® Version       PCIe® 4.0           Native PCIe® Lanes (Total/Usable)  16 , 16
System Memory Type         256-bit LPDDR5x     Max. Memory  128 GB   Max Memory Speed  LPDDR5x-8000
Graphics Model             Radeon 8060S Graphics
Graphics Core Count        40                  Graphics Frequency  2900 MHz
FP16 AI Compute            Up to 60 TFLOPS
Overall TOPS               Up to 126 TOPS      NPU TOPS  Up to 50 TOPS
Product ID Tray            100-000001099
```

Two things a reader should not miss. The **memory ceiling is arithmetic**: 8000 MT/s × 256 bit ÷ 8 =
**256 GB/s, shared by CPU, GPU and NPU** — not a GPU-side figure. And **60 TFLOPS is filed under "AI
Engine Capabilities / FP16"**, not under graphics; see §5, because our own MFMA probe reaches about a
tenth of it and that gap is still unexplained.

## 2. The platform: Framework Desktop

Source: <https://frame.work/blog/introducing-the-framework-desktop> and
<https://frame.work/desktop?tab=specs>. The live spec page now sells the **Ryzen AI Max+ PRO 495**
configuration (192 GB LPDDR5x-**8533** → 273 GB/s, 3.1 GHz base / 5.2 GHz boost, 120 W sustained /
140 W boost package power, 400 W FlexATX PSU, 4.5 L, Realtek RTL8126 5GbE, 2× M.2 2280 **PCIe 4.0 x4**,
one open-ended PCIe x4 slot). The part shipped in *this* box is the launch SKU, the plain **Max+ 395**
with 128 GB LPDDR5x-8000 — so the current page's memory speed and TOPS are **not** this machine's.

Local identity, which is what to diff against when a number moves:

```
$ for f in sys_vendor product_name product_family board_name bios_version bios_date; \
    do printf "%-15s %s\n" "$f" "$(cat /sys/class/dmi/id/$f)"; done
sys_vendor      Framework
product_name    Desktop (AMD Ryzen AI Max 300 Series)
product_family  Desktop
board_name      FRANMFCP06
bios_version    03.02                    bios_date  07/22/2025
```

`BIOS 03.02 / 2025-07-22` matters: the GPU's usable memory and its clock policy are BIOS-visible
behaviour on this part, and Framework has shipped firmware changes to both. Record the BIOS version
above any number that could move with firmware.

## 3. Software stack as pinned on this machine

```
$ uname -r                        7.0.0-38-generic
$ cat /proc/cmdline | tr ' ' '\n' | grep amdgpu
amdgpu.gttsize=114688             # 112 GiB GTT window — this is why HIP reports ~110 GiB total
$ dpkg -l | grep rocm-core
amdrocm-core10.0-gfx1151          10.0.0-4
$ rocm-smi --showclocks | grep -E "sclk|mclk"
GPU[0]: sclk clock level: 1: (2896Mhz)     # reads 2884–2896 across windows
GPU[0]: mclk clock level: 2: (1000Mhz)     # idle floor; do not trust mclk readings taken idle
```

`amdgpu.gttsize=114688` is the answer to a question `devices/strix-halo-unified-memory.md` raises but
does not settle — where the ~110 GiB "device-visible pool" comes from. It is a **kernel module
parameter on this machine's command line**, not a hardware or BIOS property, so a different host (or a
kernel-command-line edit) changes every capacity ceiling quoted in that file. sysfs meanwhile reports
`mem_info_vram_total` = 2147483648 (2 GiB dedicated), which is a third number and not the pool we
allocate from.

## 4. Claim → local command → value

| vendor / spec claim | command | this machine | verdict |
|---|---|---|---|
| 16 Zen 5 cores, 32 threads | `lscpu` | `CPU(s): 32`, `Core(s) per socket: 16`, `Thread(s) per core: 2` | **matches** |
| up to 5.1 GHz boost | `lscpu \| grep MHz` | `CPU max MHz: 5187,5` | matches (5187 is a boost step above the rounded 5.1 GHz) |
| L2 16 MB, L3 64 MB | `lscpu \| grep cache` | `L2: 16 MiB (16 instances)`, `L3: 64 MiB (2 instances)` | **matches**; 2 L3 instances = the two CCDs AMD's page counts as 3 dies |
| 128 GB max memory | `grep MemTotal /proc/meminfo` | `129611480 kB` (123.6 GiB) | **consistent** — the rest is the GPU carve-out and reservations |
| 256-bit LPDDR5x-8000 → 256 GB/s shared | `tools/mem_bw`-class probe, §bandwidth note | **237–239 GB/s** pure read (93 % of the 256 GB/s fabric) | **matches in magnitude**, and it is why 237 GB/s is the repo's denominator |
| Radeon 8060S, 40 graphics cores, 2900 MHz | `rocminfo` | `Compute Unit: 40`, `Max Clock Freq.: 2900`, `SIMDs per CU: 2`, `Wavefront Size: 32`, `Max Waves Per CU: 32` | **matches** (HIP's `multiProcessorCount` = 20 is a driver bug — see `strix-halo-compute.md`) |
| PCIe 4.0, 16 lanes | `cat /sys/bus/pci/devices/0000:c1:00.0/current_link_{speed,width}` | `16.0 GT/s PCIe`, width `4` (one M.2 socket) | **matches** (Gen4, per-socket width 4 as Framework's spec says) |
| NPU present (50 TOPS XDNA 2) | `rocminfo` | agent `RyzenAI-npu5` enumerated | **matches**; unused by this repo |
| GPU cache sizes (not on AMD's page) | `rocminfo` | `L2: 2048 KB`, per-CU `L1: 32 KB` | see §5 — this is where our measurements and the published teardown numbers disagree |

## 5. Discrepancies worth keeping open, with both sides cited

These are recorded as **open**, not resolved; each has an experiment attached.

1. **60 TFLOPS FP16 (AMD) vs 5.667 TFLOP/s measured bf16 matmul.** → **RESOLVED in §7.** Short version:
   the 60 is the WMMA matrix roof (a real instruction this compiler can emit, 4× the vector roof), and
   **0 of the 55 gfx1151 code objects shipped with rocBLAS contain a matrix instruction** — so the gap
   is in the library stack, not the silicon and not marketing.
2. **The on-die cache hierarchy.** Our own probes say a **2 MiB** tier at ~1.4 TB/s and a **~54 MiB**
   tier at ~876 GB/s (`strix-halo-memory-bandwidth.md`, `strix-halo-mixed-traffic.md`). `rocminfo`
   agrees on 2 MiB of L2. Chips and Cheese, measuring the same part, reports a **4 MB GPU L2 at ~2.5 TB/s**
   and a **32 MB Infinity Cache (MALL) on the I/O die that the CPU cannot reach**
   (<https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix>). Our 54 MiB tier is bigger than
   any L2 either side claims, so MALL is the obvious candidate — and if it is MALL, the whole
   "does the layer's KV set stay resident" question has a 32 MB budget, not a 2 MiB one. Untested.
3. **Occupancy metrics that exceed the architecture.** `rocminfo` says **Max Waves Per CU = 32** (2 SIMDs
   × 16). A `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES` ratio taken from `rocprofv3` today came out at **41**, which
   is above that ceiling, so the ratio is *not* "wavefronts resident per CU" on this part and must not be
   quoted as such — the counter is summed over a set whose size I have not established. Recorded in
   `progress/2026-10-16/prefill-occupancy-results-2026-10-16.md` §3's correction.
4. **Clock state is part of every number.** AMD's 2900 MHz is a maximum; `rocm-smi` idles `mclk` at 1000 MHz
   and the GPU has been observed under load near 1.0–1.2 GHz during large GEMMs
   (`strix-halo-compute.md`). A measurement that does not state the clock window is not comparable to one
   that does.
5. **CPU-side reachability of the fabric.** Chips and Cheese measured the CPU at ~124 GB/s of reads
   (~175 GB/s read-modify-add) because **each CCD has a 32-byte-per-cycle link to the I/O die**, and notes
   ~123 ns DRAM latency versus 75–80 ns on desktop Zen 5. That matches this repo's CPU-side 75–117 GB/s
   ceiling and is the reason the engine must stay GPU-resident.

## 6. Re-verifying without a network

Every local number above comes from a command in this file. The vendor claims that cannot be re-checked
offline are transcribed in §1; if a decision is going to rest on one of them — the 256 GB/s fabric, the
40 CUs, the 60 TFLOPS FP16 — re-read the page rather than trusting this transcription, and note the
access date next to the claim you used.

## 7. The "60 TFLOPS FP16" question: marketing, or a gap in the stack?

Asked on 2026-10-16, because the repo had priced a funded route against "the MFMA peak of this part is
14.85 TFLOPS (1:1 with FP32), of which we realize 38 %" (`prototype/compute_bw/results.md` §10.1).
That sentence is wrong in both halves. Reproduce with **`tools/isa_probe.sh`** — compile and
disassemble only, no device, no GPU lock.

### 7.1 No MFMA on this part; WMMA instead

| instruction | gfx1151 | note |
|---|---|---|
| `v_mfma_f32_16x16x16f16`, `v_mfma_f32_16x16x16bf16_1k` | **absent** ("instruction not supported on this GPU") | MFMA is CDNA (Instinct); RDNA never had it |
| `v_wmma_f32_16x16x16_f16` | **present** | reachable via `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` with `ext_vector_type(16)` fp16 operands and an `ext_vector_type(8)` f32 accumulator; such a kernel compiles for `gfx1151` and `llvm-objdump` shows the instruction emitted |
| `v_dot2_f32_f16`, `v_dot2_f32_bf16` | present | 2 MACs/lane/clk |
| `v_pk_fma_f16` | present | 2 MACs/lane/clk — what rocBLAS's fp16 kernels use |
| `v_pk_fma_bf16` | **absent** | bf16 has no packed FMA on this part |
| `v_fma_mix_f32` | present | bf16→f32, **1 MAC/lane/clk** — what our own kernels and rocBLAS's bf16 kernels use |
| `v_dot4_i32_i8`, `v_dot8_i32_i4` | present | int8 = 4 MACs, int4 = 8 MACs per lane per clk; rocBLAS's int8 library uses `v_dot4_i32_i8` |
| `v_dot4_f32_fp8_fp8` / `_fp8_bf8` / `_bf8_bf8` | **absent** | **no FP8 on this ISA.** 8-bit here means INT8/INT4 integer ops, not FP8 |

Two traps cost an hour each, and both are now commented in `tools/isa_probe.sh`: the CDNA-style builtin
name (`__builtin_amdgcn_wmma_*_16f16_w32`) does not exist for gfx1151, so a mis-spell reads as "WMMA is
missing"; and Tensile's per-kernel `.co` files are CCO containers (`CCO\003`) that `llvm-objdump`
refuses, which also reads as absence. Scan the `.hsaco` libraries, not the `.co` files.

### 7.2 The vendor number is the matrix roof, and it is 4× the vector roof

```
vector FP32 : 40 CU × 128 FMA-lane/clk × 2 FLOP × 2.894 GHz = 14.82 TFLOP/s  (= the 14.85 AMD/TPU quote)
WMMA fp16   : 40 CU × 512 FLOP/clk/CU           × 2.894 GHz = 59.36 TFLOP/s  (= AMD's "up to 60 TFLOPS FP16 AI")
int8 dp4a   : 40 CU × 256 MAC/clk/CU × 2 × 2.894 GHz        = 59.37 TOPS     (same 4× multiplier, integer)
```

The 512 FLOP/clk/CU figure is AMD's own, published for RDNA3 WMMA (GPUOpen, *How to accelerate AI
applications on RDNA 3 using WMMA*), and `llm-tracker.info` derives the identical 59.392 TFLOPS for
this exact APU independently: *"RDNA3 has a theoretical 512 FP16 FLOPS/clock/CU. A Ryzen AI Max 395's
Radeon 8060S has 40 CUs at a max clock of 2.9GHz should have a peak 59.392 FP16 TFLOPS"*. So the vendor
figure is a real instruction-level roof — not sparsity, not fp16-accumulate, not GPU+NPU summed.

### 7.3 The gap is in the libraries, and it is nearly the whole gap

`tools/isa_probe.sh` §3 disassembles every gfx1151 code object shipped with ROCm 10.0:

```
matrix-instruction census across every gfx1151 code object shipped with rocBLAS:
  0 of 55 code objects contain a matrix instruction.
```

| rocBLAS gfx1151 library | arithmetic it actually executes | MACs/lane/clk |
|---|---|---|
| `BB_HPA` (bf16 in/out — **our path**) | `v_fmac_f32_e32` only | **1** |
| `BS_HPA` (bf16 in, f32 out) | `v_fmac_f32_e32` only | 1 |
| `HH` (fp16) | `v_pk_fma_f16` | 2 |
| `HS_HPA` (fp16 in, f32 out) | `v_fma_mix_f32` | 1 |
| `4xi8I_HPA` (int8) | `v_dot4_i32_i8` | 4 |

So the 4.696 TFLOP/s we measure through hipBLASLt was never a matrix measurement: it is a scalar-FMA
kernel judged against a scalar roof — **32 % of the vector roof, 7.9 % of the matrix roof**. Nothing in
this repo, and nothing in the shipped stack, has touched the matrix roof, because Tensile ships no WMMA
kernel for `gfx1151`. Independent evidence that this is an ecosystem gap rather than our local
misfortune, same silicon, from `llm-tracker.info` (Llama-2-7B Q4_0, pp512): Vulkan **881.71 tok/s**,
HIP **348.96**, HIP+FA 331.96, llama.cpp's own WMMA build 322.63, CPU 294.64 — with their note that
*"the HIP version performs far below what you'd expect in terms of tok/TFLOP efficiency for prompt
processing even vs other RDNA3 architectures"* (a `gfx1103` 780M was measured at 14.51 tok/TFLOP; at
that efficiency this box would do ~850). The llama.cpp side records the same ISA fact: `gfx1151`
builds must stub out the MFMA flash-attention kernel because *gfx1151 lacks MFMA*.

### 7.4 What follows

1. **Re-price the fast tier.** It was booked against 38 % of a 14.85 "MFMA" roof. Framed correctly,
   today's scoring runs at 1 MAC/lane/clk and a WMMA fp16 path has a **4×** arithmetic roof, with the
   score cost (0.246 ms at 1024 lanes) still untouched. Whether a hand-written WMMA kernel lands near
   that roof is unmeasured — at 41 % lane coverage it may be latency-bound rather than issue-bound —
   and `prototype/issue_rate_probe` now carries a WMMA leg to discriminate.
2. **The fast tier is fp16, not bf16.** WMMA here is fp16-only and `v_pk_fma_bf16` does not exist, so a
   matrix fast tier means fp16 inputs with f32 accumulate (10-bit mantissa, far narrower range than
   bf16) — a numerics band this repo has never admitted.
3. **8-bit means int8, and it is the same 4× multiplier with half the bytes.** `v_dot4_i32_i8` is the
   lever; FP8 does not exist on this ISA, so any plan written against FP8 must be restated as int8 plus
   scales.

Sources used in this section (external, accessed 2026-10-16): AMD product page for the Ryzen AI Max+ 395
(§1); GPUOpen *How to accelerate AI applications on RDNA 3 using WMMA* (512 FLOP/clk/CU, 16×16-only
tiles); `https://llm-tracker.info/_TOORG/Strix-Halo` (59.392 TFLOP derivation, HIP-vs-Vulkan pp512
table, MFMA-stub note); TechPowerUp's `gfx1151` entry (FP16 14.85 TFLOPS quoted 1:1 with FP32 — i.e. a
database that quotes only the vector rate); `hipfire.dev/learn/rdna-wmma` (independent write-up reaching
the same per-arch matrix-path table, incl. `gfx1151` WMMA half16 and no FP8 before RDNA4);
`ROCm/amd_matrix_instruction_calculator` (per-arch instruction layouts).
