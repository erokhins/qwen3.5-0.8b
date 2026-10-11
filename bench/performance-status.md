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
| `proj_gemv` all 9 projections × 24 layers, 48 launches, cold | 445.7 MB | **2.990 ms** † | 156.3 GB/s | **66 %** † |
| `proj_gemv` best config seen (rows/block 32, same conditions) | 445.7 MB | 2.785 ms † | 167.8 GB/s | 71 % † |
| `glue` residual+RMSNorm, fused, 49 launches/token | ~5 MB | **0.137 ms** † | ~37 GB/s | launch-bound, not bandwidth-bound |
| `attn_decode` §05 core, 6 layers, one launch each | ~0.4 MB at 25 lanes | ~0.011 ms (launch floor arithmetic, not timed) | — | bit-exact to 277 lanes |
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
- **A per-layer MLP kernel runs at 84 % of the probe, not 102 %, and the cause is not the row
  mapping and not block count.** `mlp_gemv` cold is 199 GB/s and sweeping rows/block 4→32 moves
  it by 4 %; warm, the same sweep looks like a 60 % spread and the best config is a different one,
  because warm ranks configs by cache residency (`prototype/mlp_gemv/results.md` has both tables).
  The obvious explanation was grid size — `down_proj` is 1024 rows, so rows/block 8 issues 128
  blocks where `lm_head_gemv` issued 31040 — but it does not survive the arithmetic: the device
  has **20 CUs** (`tools/device_query.cpp`; the 40 in `rocminfo` counts SIMDs, and this file used
  to say 40), i.e. 20 × 2048 = 40960 resident threads, and 128 blocks × 256 threads = 32768 fits
  in a single resident wave. So the 16 % gap is **unexplained**, and it is left as a named gap
  rather than optimised: closing it is worth ~0.4 ms/token of the ~6.5 ms budget.
- **Every number in this file is a measurement of a GPU that is not idle, and the published MLP
  figure is conditional on that.** The box runs `flash_serve`, which is the server generating this
  session's own tokens; it holds the iGPU at ~98 % utilisation from its polling loop whether or
  not it is producing tokens, so utilisation alone does not distinguish a clean timing from a
  contended one — in-flight concurrency does. The 2.650 ms below was taken with no other GPU work
  in flight and reproduced to ±0.9 %; the same command with two gate processes running spread
  2.61–3.59 ms and ranked the configs differently. Rule that follows: a number goes in this file
  only if it was taken under `tools/gpu_lock.sh` **and** its repeat spread is stated with it.

## 2. Per-component budget: 65 % of a token has a real kernel

Weights re-derived from the safetensors headers, grouped as §01.7 (decimal GB, so
`1.5457 GB` text-path total checks out). Roofline column = bytes ÷ 237 GB/s.

| component | GB/token | roofline | measured | status |
|---|---|---|---|---|
| `lm_head` / `embed_tokens` (tied) | 0.5086 | 2.146 ms | **2.105 ms** | **kernel exists, 102 % of roofline** |
| MLP, 24 layers (gate/up/down) | 0.5285 | 2.230 ms | **2.650 ms** | **kernel exists, 84 % of roofline** |
| GDN `in_proj`/`out_proj` + conv, 18 layers | 0.3796 | 1.602 ms | — † | **kernel exists** (`proj_gemv`), see † |
| full attention `qkv`/`o`, 6 layers | 0.0881 | 0.372 ms | — † | **kernel exists** (`proj_gemv`), see † |
| **the two above together, measured as one 48-launch plan** | 0.4677 | 1.973 ms | **2.990 ms** † | **66 % of the probe** |
| MTP draft layer (only when drafting) | 0.0409 | 0.173 ms | — | not written |
| norms and small tensors | ~0.005 | 0.021 ms | — | not written |
| GDN state traffic (18 layers, warm) | 0.054 | 0.228 ms cold | **0.062 ms** | **kernel exists** |
| **total per token** | **1.605** | **6.52 ms** | **7.81 ms measured** | |

So **1.5588 / 1.605 = 97 %** of the token's bytes are now covered by a kernel that has been
measured on this box — `lm_head`, the MLP, every projection, the GDN recurrence, and the norms.
What is left is §05's attention core, and that number is a standing warning about the whole
table: **attention's traffic is a function of context length, not of the model.** At the fixture's
25 lanes the KV read is ~0.4 MB/token, which is why `attn_decode` is launch-bound and costs
~11 µs. At a 4k context the same six layers read ~25 MB/token — an extra ~0.1 ms that this table
cannot see, because the fixtures are short. Any tok/s number taken at prefix-24 is therefore an
optimistic bound on long-context behaviour, not a prediction of it. Coverage stopped being the interesting number; the spread did —
**102 % / 84 % / 66 %** for `lm_head` / MLP / projections, and within the projections alone the
per-shape results run from 77 % of the probe (`in_proj_qkv` packed) down to 39 % (`v_proj`) and
5 % for the 16-row `in_proj_b`/`in_proj_a`. The wide shapes behave like `lm_head`; the narrow ones
behave like nothing measured before, and that is what moved the projection below.

† **The projection numbers are provisional.** `gpu_lock.sh util` read 9 % at acquire and 60 % at
release — something worked on the GPU mid-run without the lock — so they are floors, not
measurements, and a clean re-measure is outstanding (§5 item 1).

## 3. Projection, with the assumptions named

| assumption | value | risk |
|---|---|---|
| ~~remaining GEMVs reach the same fraction of the probe the MLP did~~ **measured** | projections land at **66 %**, not 84 % | no longer an assumption — the per-shape spread is 39–77 % of the probe and the weighted result is 66 %. The row-count theory is dead: `in_proj_qkv` has 6144 rows and reaches 77 %, `k_proj`/`v_proj` have 512 and reach ~40 %, while `lm_head` with 248320 rows reached 102 % |
| launch floor | +0.36 ms (200 launches × 1.8 µs), partly hidden by overlap | **medium** — the projections alone are 48 launches/token and the small ones are already launch-dominated: `in_proj_b`/`in_proj_a` cost 3.2 µs/launch to move 32 KiB, which is why they are folded into the packed launch |
| weights already resident (no H2D per token) | assumed | low — H2D is 76–82 GB/s, so it would dominate if violated |
| no fp32 upcast traffic for accumulation | assumed | low — demonstrated twice now: MLP and projections both accumulate in F32 registers and move no extra bytes |
| MTP drafting disabled | assumed | drafting adds 0.0409 GB + an extra logits read per accepted burst |

**Projected single-stream bf16 decode: ~7.9–8.3 ms/token = ~120–127 tok/s.** That is a second
revision *downward* (was 145–153, then 137–147), and the cause is the same both times: the §07.4
roofline assumes every byte stream arrives at probe speed, and each family measured so far arrives
slower than the one before it — 102 %, then 84 %, then 66 %. 7.81 ms of the token is now measured,
so only attention's core and the norms remain, and they are small in bytes (~0.03 ms of roofline)
but not in launches. §07.4's headline 151 tok/s is now roughly 25 % above the measured trajectory
and should be treated as wrong on this box unless the narrow-shape gap closes.

The two small ops confirm that framing rather than changing it: `glue` at 0.137 ms/token is 97 →
49 launches worth of saving (fusion bought ~0.079 ms/token, ~1 % of the budget, which is exactly
what a launch-count item should return), and `attn_decode` is ~11 µs/token of dispatch at short
context. Neither moves the projection. What moves it is the 66 %.

The one lever that is already visible: `rows/block` 32 measured 8 % better than the shipped default
of 8 on the projections (2.785 vs 2.990 ms), which is ~0.2 ms/token — worth more than anything else
known, and currently an open decision rather than a change, because it was measured on a contended
box.

## 4. What *is* measured end-to-end today (and is not the engine)

`speccheck/` runs the whole model in torch, CPU-only, bf16: **~24 tok/s** greedy, **~21
tok/s** with `--mtp 1` (accept 0.660, 1.24 tok/forward). That is the correctness oracle.
It is 6× below the GPU projection because the CPU path reads at 75–117 GB/s at best and
torch adds per-op overhead on ~200 ops/token. It is quoted here so the number is on the
record as *not* the engine's performance.

## 5. What has to happen before this file can say "measured"

1. **Re-measure the projections on a quiet box.** They are the third of the token and they came in
   at 66 % under contention; both the headline 2.990 ms and the `rows/block` 32 finding (+8 %) need
   a clean run with util near 0 at both ends of the lock before they are quoted anywhere else.
2. §05's attention core (RoPE, per-head norm, GQA read over the KV cache, output gate) and the
   norm/gating/residual glue. `mlp_gemv` and `proj_gemv` graduated 2026-08.
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
