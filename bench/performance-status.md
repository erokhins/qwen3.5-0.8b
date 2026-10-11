# Performance status — what is measured, what is projected

Written after the second HIP kernel. The point of this file is the split: **nothing
end-to-end has been timed yet**, because there is no engine — there are three measured GPU
numbers and a roofline they are consistent with. Do not quote a tokens/s figure from this
project as measured until a row moves into the first table.

## 1. Measured on the GPU (real kernels, real reference data)

| what | traffic | time | bandwidth | vs 237 GB/s probe |
|---|---|---|---|---|
| `lm_head_gemv` bf16 GEMV, 248320 × 1024 | 485.0 MB | **2.105 ms** | 242.1 GB/s | **102 %** |
| `mlp_gemv` MLP, 24 layers, cold weights | 528.5 MB | **2.650 ms** | 199.4 GB/s | **84 %** |
| `gdn_decode` recurrence, 18 layers, state warm | 54 MB (3 × 18 MiB) | **0.062 ms** | 918 GB/s | 387 % — cache-resident |
| same kernel, state tiled to 108 MiB (cold) | 324 MB | 1.344 ms | 255 GB/s | 108 % |
| kernel launch floor, empty/tiny kernel | — | **1.85 µs**/launch | — | graphs: **1.78 µs**, no gain |
| iGPU read probe (`tools/bw_probe_hip.cpp`) | 1 GiB | — | 237 GB/s best | reference |
| CPU read probe (`tools/bw_probe.py`) | 1 GiB | — | 75–117 GB/s | reference |

Three things in that table were surprises and all are now load-bearing:

- **HIP Graphs do not help on this part.** 64 captured launches replay in 114.05 µs =
  1.78 µs/kernel, against 1.85 µs eager. The limit is the GPU's own dispatch rate, not the
  host submit path, so the usual "wrap the decode step in a graph" optimization buys ~4 %
  here, not the 3–5× it buys on a discrete card. Fusion — fewer, bigger kernels — is the
  only real lever on launch cost. Probe: `tools/launch_overhead_hip.cpp`.
- **18 MiB of GDN state stays resident in the iGPU cache** across decode steps (918 GB/s
  warm vs 255 cold). The cache is not reported by `hipGetDeviceProperties` (it says 2 MiB
  L2), so this had to be found by tiling the working set.
- **A per-layer MLP kernel runs at 84 % of the probe, not 102 %, and the row mapping is not
  why.** `mlp_gemv` cold is 199 GB/s and sweeping rows/block 4→32 moves it by 4 %; warm, the
  same sweep looks like a 60 % spread and the best config is a different one, because warm ranks
  configs by cache residency (`prototype/mlp_gemv/results.md` has both tables). The likely cause
  is block count: `down_proj` is 1024 rows, so rows/block 8 issues 128 blocks over 40 CUs where
  `lm_head_gemv` issued 31040. Left as a named gap rather than optimised: closing it is worth
  ~0.4 ms/token of the ~6.5 ms budget.

## 2. Per-component budget: 65 % of a token has a real kernel

Weights re-derived from the safetensors headers, grouped as §01.7 (decimal GB, so
`1.5457 GB` text-path total checks out). Roofline column = bytes ÷ 237 GB/s.

| component | GB/token | roofline | measured | status |
|---|---|---|---|---|
| `lm_head` / `embed_tokens` (tied) | 0.5086 | 2.146 ms | **2.105 ms** | **kernel exists, 102 % of roofline** |
| MLP, 24 layers (gate/up/down) | 0.5285 | 2.230 ms | **2.650 ms** | **kernel exists, 84 % of roofline** |
| GDN `in_proj`/`out_proj` + conv, 18 layers | 0.3796 | 1.602 ms | — | not written (recurrence is written) |
| full attention `qkv`/`o`, 6 layers | 0.0881 | 0.372 ms | — | not written |
| MTP draft layer (only when drafting) | 0.0409 | 0.173 ms | — | not written |
| norms and small tensors | ~0.005 | 0.021 ms | — | not written |
| GDN state traffic (18 layers, warm) | 0.054 | 0.228 ms cold | **0.062 ms** | **kernel exists** |
| **total per token** | **1.605** | **6.52 ms** | **4.82 ms measured** | |

So **1.0371 / 1.605 = 65 %** of the token's bytes are covered by a kernel that runs at 84–102 %
of the measured roofline, and the recurrence (which touches no weights) is done. What is left is
**one shape class** — `mlp_gemv`'s `mlp_down<IN>` instantiated at the GDN and attention projection
shapes — plus §05's attention itself.

## 3. Projection, with the assumptions named

| assumption | value | risk |
|---|---|---|
| remaining GEMVs reach the same fraction of the probe the MLP did | ~200 GB/s (84 %) | **medium** — `lm_head` hit 102 % with 31040 blocks, the MLP lands at 84 % with 128–448; the remaining projections are 3456/1536/6144-row shapes, i.e. the MLP's end of that range, not `lm_head`'s |
| launch floor | +0.36 ms (200 launches × 1.8 µs), partly hidden by overlap | **medium** for the small ops: a 0.09 GB attention projection at the 1.8 µs floor cannot overlap much |
| weights already resident (no H2D per token) | assumed | low — H2D is 76–82 GB/s, so it would dominate if violated |
| no fp32 upcast traffic for accumulation | assumed | low — now demonstrated: `mlp_gemv` accumulates in F32 registers and moves no extra bytes |
| MTP drafting disabled | assumed | drafting adds 0.0409 GB + an extra logits read per accepted burst |

**Projected single-stream bf16 decode: ~6.8–7.3 ms/token = ~137–147 tok/s.** That is a revision
*down* from the 145–153 this file used to carry, and the reason is the MLP row: the §07.4 roofline
assumes every byte stream arrives at probe speed, and the two GEMV families with enough rows to
fill 40 CUs disagree with each other by 18 %. 4.82 ms of the budget is now measured; the remaining
1.99 ms of roofline (GDN + attention projections, norms) projects to ~2.0–2.4 ms of real time.
§07.4's headline 151 tok/s is above the optimistic end of that range and should be treated as
conditional on the block-count gap closing.

## 4. What *is* measured end-to-end today (and is not the engine)

`speccheck/` runs the whole model in torch, CPU-only, bf16: **~24 tok/s** greedy, **~21
tok/s** with `--mtp 1` (accept 0.660, 1.24 tok/forward). That is the correctness oracle.
It is 6× below the GPU projection because the CPU path reads at 75–117 GB/s at best and
torch adds per-op overhead on ~200 ops/token. It is quoted here so the number is on the
record as *not* the engine's performance.

## 5. What has to happen before this file can say "measured"

1. The projection GEMVs — `mlp_gemv`'s shape-generic kernel instantiated at GDN `in_proj` /
   `out_proj` and the attention `qkv`/`o` shapes (38 % of the token), and §05's attention itself.
   MLP is done: `prototype/mlp_gemv` graduated 2026-08.
2. Fused per-layer epilogues (RMSNorm, gating, residual) so the small ops stop paying 1.8 µs
   each — the launch floor is 0.36 ms/token if eager and graphs will not fix it. `mlp_gateup`
   already shows the pattern: the epilogue lives inside the GEMV that produces its input.
3. `engine/` consuming the graduated kernels, then a `bench/` run of greedy decode on the
   `testdata/` fixtures, then **the first real tok/s row in table 1**.
4. Re-measure MTP accept economics at GPU speed: bit-lossless drafting that loses on CPU may
   win once the drafter's 0.0409 GB is cheap relative to 1.565 GB.
5. Close the block-count gap: `down_proj` at 128 blocks reaches 88 % of the probe. If keeping `x`
   packed in shared (halving LDS traffic) or splitting long rows across wavefronts does not fix
   it, say so here and accept 84 % as this box's number for small-GEMV shapes.
