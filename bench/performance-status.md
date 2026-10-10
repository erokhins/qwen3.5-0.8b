# Performance status — what is measured, what is projected

Written after the second HIP kernel. The point of this file is the split: **nothing
end-to-end has been timed yet**, because there is no engine — there are three measured GPU
numbers and a roofline they are consistent with. Do not quote a tokens/s figure from this
project as measured until a row moves into the first table.

## 1. Measured on the GPU (real kernels, real reference data)

| what | traffic | time | bandwidth | vs 237 GB/s probe |
|---|---|---|---|---|
| `lm_head_gemv` bf16 GEMV, 248320 × 1024 | 485.0 MB | **2.105 ms** | 242.1 GB/s | **102 %** |
| `gdn_decode` recurrence, 18 layers, state warm | 54 MB (3 × 18 MiB) | **0.062 ms** | 918 GB/s | 387 % — cache-resident |
| same kernel, state tiled to 108 MiB (cold) | 324 MB | 1.344 ms | 255 GB/s | 108 % |
| kernel launch floor, empty/tiny kernel | — | **1.85 µs**/launch | — | graphs: **1.78 µs**, no gain |
| iGPU read probe (`tools/bw_probe_hip.cpp`) | 1 GiB | — | 237 GB/s best | reference |
| CPU read probe (`tools/bw_probe.py`) | 1 GiB | — | 75–117 GB/s | reference |

Two things in that table were surprises and both are now load-bearing:

- **HIP Graphs do not help on this part.** 64 captured launches replay in 114.05 µs =
  1.78 µs/kernel, against 1.85 µs eager. The limit is the GPU's own dispatch rate, not the
  host submit path, so the usual "wrap the decode step in a graph" optimization buys ~4 %
  here, not the 3–5× it buys on a discrete card. Fusion — fewer, bigger kernels — is the
  only real lever on launch cost. Probe: `tools/launch_overhead_hip.cpp`.
- **18 MiB of GDN state stays resident in the iGPU cache** across decode steps (918 GB/s
  warm vs 255 cold). The cache is not reported by `hipGetDeviceProperties` (it says 2 MiB
  L2), so this had to be found by tiling the working set.

## 2. Per-component budget: 32 % of a token has a real kernel

Weights re-derived from the safetensors headers, grouped as §01.7 (decimal GB, so
`1.5457 GB` text-path total checks out). Roofline column = bytes ÷ 237 GB/s.

| component | GB/token | roofline | measured | status |
|---|---|---|---|---|
| `lm_head` / `embed_tokens` (tied) | 0.5086 | 2.146 ms | **2.105 ms** | **kernel exists, 102 % of roofline** |
| MLP, 24 layers (gate/up/down) | 0.5285 | 2.230 ms | — | not written |
| GDN `in_proj`/`out_proj` + conv, 18 layers | 0.3796 | 1.602 ms | — | not written (recurrence is written) |
| full attention `qkv`/`o`, 6 layers | 0.0881 | 0.372 ms | — | not written |
| MTP draft layer (only when drafting) | 0.0409 | 0.173 ms | — | not written |
| norms and small tensors | ~0.005 | 0.021 ms | — | not written |
| GDN state traffic (18 layers, warm) | 0.054 | 0.228 ms cold | **0.062 ms** | **kernel exists** |
| **total per token** | **1.605** | **6.52 ms** | **2.17 ms measured** | |

So **0.5086 / 1.565 = 32 %** of the token's bytes are covered by a kernel that runs at
102 % of the measured roofline, and the recurrence (which touches no weights) is done.
The other 68 % is four GEMV families of the same shape class as the one that already works.

## 3. Projection, with the assumptions named

| assumption | value | risk |
|---|---|---|
| remaining GEMVs reach the probe bandwidth | ≥ 230 GB/s | **low** — same shape class as `lm_head_gemv`, which hit 242 |
| launch floor | +0.36 ms (200 launches × 1.8 µs), partly hidden by overlap | **medium** for the small ops: a 0.09 GB attention projection at 1.8 µs floor cannot overlap much |
| weights already resident (no H2D per token) | assumed | low — H2D is 76–82 GB/s, so it would dominate if violated |
| no fp32 upcast traffic for accumulation | assumed | **medium** — GDN state is F32 by spec; bf16 GEMV accumulates in registers, which is free |
| MTP drafting disabled | assumed | drafting adds 0.0409 GB + an extra logits read per accepted burst |

**Projected single-stream bf16 decode: 6.5–6.9 ms/token = ~145–153 tok/s.** §07.4's
headline 151 tok/s is the optimistic end of that range and survives; it is not contradicted
by anything measured so far, but it is not confirmed either — no kernel has been measured on
the MLP or attention shapes yet, and those are 39 % of the token.

## 4. What *is* measured end-to-end today (and is not the engine)

`speccheck/` runs the whole model in torch, CPU-only, bf16: **~24 tok/s** greedy, **~21
tok/s** with `--mtp 1` (accept 0.660, 1.24 tok/forward). That is the correctness oracle.
It is 6× below the GPU projection because the CPU path reads at 75–117 GB/s at best and
torch adds per-op overhead on ~200 ops/token. It is quoted here so the number is on the
record as *not* the engine's performance.

## 5. What has to happen before this file can say "measured"

1. MLP and GDN/attention projection GEMVs (4 families, 68 % of the token).
2. Fused per-layer epilogues (RMSNorm, gating, residual) so the small ops stop paying 1.8 µs
   each — the launch floor is 0.36 ms/token if eager and graphs will not fix it.
3. `engine/` consuming the graduated kernels, then a `bench/` run of greedy decode on the
   `testdata/` fixtures, then **the first real tok/s row in table 1**.
4. Re-measure MTP accept economics at GPU speed: bit-lossless drafting that loses on CPU may
   win once the drafter's 0.0409 GB is cheap relative to 1.565 GB.
