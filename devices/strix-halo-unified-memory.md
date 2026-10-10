# Putting weights in RAM and streaming them from the iGPU (measured)

Short version: **there is no VRAM to reserve.** This iGPU has no local memory — every
device allocation is system RAM, reached through the GTT (Graphics Translation Table), and
it streams at the same ~240 GB/s whether it came from inside the 2 GiB "VRAM" reservation,
outside it, from `hipMallocManaged`, or from pinned host memory. `hipMalloc` over-commits
the reservation silently and hands back GTT pages. The only real limits are a
**~32–34 GiB total device-visible pool** and a **first-touch penalty**, both below.

Probes: `tools/bw_probe_hip.cpp`, `tools/mem_mode_probe_hip.cpp`. Reproduce with

    hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/mem_mode_probe tools/mem_mode_probe_hip.cpp
    /tmp/mem_mode_probe 4.0 5          # 4 GiB buffers, 5 reps
    /tmp/mem_mode_probe 2.0 1 1.565e9 many 24   # 24 x 2 GiB, granularity test

## What the driver actually does

    $ cat /sys/class/drm/card*/device/mem_info_vram_total   # the "VRAM" reservation
    2147483648                       # 2 GiB
    $ cat /sys/class/drm/card*/device/mem_info_gtt_total    # GTT = system RAM, GPU-mapped
    120259084288                     # ~112 GiB
    $ rocm-smi --showmeminfo vram    # same 2 GiB, ~1.97 GiB already used by the desktop
    $ /opt/rocm/bin/rocm-smi --showmeminfo vram | grep -i total
    # HIP-side equivalent (there is no CUDA; the project .venv torch is CPU-only, so ask
    # HIP from C++ or from a torch build with ROCm, not from this venv):
    #   hipGetDeviceProperties(...).totalGlobalMem  -> ~110 GiB (reservation + GTT merged)

Note on tooling: the project's `.venv` has a **CPU-only torch**, so nothing in Python can
see the GPU. Every GPU measurement here is a `hipcc` binary. Also, HIP binaries need
`-Wl,-rpath,/opt/rocm/lib` (or `LD_LIBRARY_PATH=/opt/rocm/lib`) or they die at load with
`libamdhip64.so.7: cannot open shared object file`.

## Measured, per allocation mode (4 GiB buffers, read = grid-stride float4 sum)

| mode | landed in | steady read | first GPU pass | CPU fill |
|---|---|---|---|---|
| `hipMalloc` 1 GiB (inside reservation) | **GTT** (+1.00 GiB gtt, +0.00 vram) | 237–240 GB/s | **~35 GB/s** | — |
| `hipMalloc` 4 GiB (over the reservation) | GTT (+4.00 GiB) | 239 GB/s | 219 GB/s | — |
| `hipMallocManaged` 4 GiB | plain RAM, not even counted as GTT | 239 GB/s | 235 GB/s | **32 GB/s** |
| `hipHostMalloc` 4 GiB, read from kernel | pinned host | 237 GB/s | 232 GB/s | 32 GB/s |
| `hipMemcpy` H2D / D2H | — | **76–82 / 63–70 GB/s** | — | — |

Three things this table is telling us:

1. **Placement does not matter for bandwidth.** 237–240 GB/s in every mode. The 2 GiB
   reservation is not a performance tier here; the compositor already occupies most of it,
   so even a 1 GiB `hipMalloc` lands in GTT and nobody notices.
2. **First touch is the trap.** The first GPU pass over freshly allocated pages ran at
   ~35 GB/s — 7× slower — because pages fault in during the kernel. Warm up after loading,
   and never time the first launch.
3. **`hipMallocManaged` is not free.** It reads fine and the CPU can fill it directly at
   ~32 GB/s (single-threaded volatile store; a real loader would beat that), which is
   tempting — see the 32 GiB warning below.

## The two real limits

> **Scope note.** This model uses ~4 % of the pool (`AGENTS.md`). Nothing in the current
> milestones depends on either limit; they are recorded because they would bind a larger
> target, not because the engine must handle them.

**Total device-visible memory is ~32–34 GiB per process, independent of granularity.**
| test | result |
|---|---|
| one `hipMalloc` of 32 GiB | ok, 231 GB/s |
| one `hipMalloc` of 40 GiB | **fails** (`hipErrorOutOfMemory`) |
| 8 × 8 GiB | **4th ok (32 GiB), 5th fails**, mean 222 GB/s |
| 24 × 2 GiB | **17 ok (34 GiB), 18th fails**, mean 227 GB/s |

`GPU_MAX_HEAP_SIZE=100` and `GPU_MAX_ALLOC_SIZE=…` did not change it. It is a total-pool
limit, not an allocation-size limit, and bandwidth is flat right up to the edge.

**[FUTURE] consequence, not a current requirement:** this model (1.55 GB) and even a 35B at 4-bit (~20 GB) fit
comfortably, but **bf16 35B-A3B (~66 GB) will not fit** in one process. M4 either
quantizes or raises the carve-out with the `rmem` module (`ConfigInfo=`) / BIOS memory
settings, which is a root change and must be re-measured with these probes afterwards.

**`hipMallocManaged` degrades silently near that limit.** At 4–24 GiB it hit 215–239 GB/s;
at 32 GiB it dropped to **51.6 GB/s without erroring**. `hipMalloc` fails loudly in the
same situation, which is the behaviour an engine wants. So: `hipMalloc` + explicit
`hipMemcpy` for the weight pool; managed only for small host-shared control buffers. This
one *does* apply today — it is an allocation-strategy rule, not a capacity question.

**The pool is system RAM, so it competes.** `free -g` on this box shows tens of GiB in
buff/cache; page-cache pressure and the desktop's own ~2 GiB of the reservation both come
out of what is left. A kernel that reads 34 GiB while the page cache is full will be slower
than the probe measured on an idle machine — re-measure after heavy I/O, and never benchmark
while something else is touching memory.

## The recipe the engine will use

```c
// 1. one slab for weights, one for state. Never per-tensor allocations in the decode
//    loop: allocation and page-faulting are the 35 GB/s regime, not the 240 GB/s one.
void* weights;                      // 1.5457 GB today (§01.7)
hipMalloc(&weights, WEIGHTS_BYTES);
void* state;                        // GDN 18.0 MiB + conv 0.63 MiB per request + KV (§00.5)
hipMalloc(&state, STATE_BYTES);

// 2. stage shard bytes in pinned memory, then one bulk copy per shard.
//    Measured 54-58 GB/s including the destination's first touch, 76-82 GB/s steady.
char* stage; hipHostMalloc(&stage, shard_bytes);   // read safetensors into `stage`
hipMemcpy(weights + offset, stage, shard_bytes, hipMemcpyHostToDevice);

// 3. warm the pages before measuring anything or serving a request.
launch_first_kernel(...);  hipDeviceSynchronize();  // this pass runs at ~35 GB/s
```

Checklist, in order, after any driver/BIOS/carve-out change:

1. `cat /sys/class/drm/card*/device/mem_info_{vram,gtt}_{total,used}` — record the split.
2. `/tmp/mem_mode_probe 4.0 5` — read must stay 235–240 GB/s in every mode; note the
   first-pass column, it is the warm-up requirement.
3. `/tmp/mem_mode_probe 2.0 1 1.565e9 many 24` — find the total-pool ceiling again before
   sizing any model against it.
4. `free -g` — the pool is system RAM: whatever the desktop holds is unavailable, and a
   Python driver thread touching memory during decode steals bandwidth from the kernel.

## What this does *not* buy

- It does not raise the roofline. 240 GB/s is the LPDDR5x ceiling here
  (`strix-halo-memory-bandwidth.md`), so ~151 tok/s bf16 single-stream regardless of
  where the bytes are registered.
- Host-pinned access being the same speed as GTT access means the choice between them is
  about control and eviction, not throughput. Zero-copy pointers are still the wrong tool
  for weight streaming: they keep the host in the loop per access.
- `hipHostMalloc` staging costs a copy at 76–82 GB/s. For this model that is ~10–20 ms of
  load time, which is irrelevant against a 200 ms prefill; it only matters for big models,
  where the alternative (managed + CPU fill at ~32 GB/s) is slower anyway.
