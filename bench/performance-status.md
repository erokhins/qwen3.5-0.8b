# Performance status — what is measured, what is projected

Written after the second HIP kernel. The point of this file has always been the split between
what is measured and what is projected.

**Status as of 2026-10-12: the engine exists and has been timed.** `engine/build/decode_run` runs
the whole token on the GPU and measures **11.919 ms/step at a 25-lane context (83.9 tok/s)**, rising
to **12.614 ms at 327 lanes** — see §6's context-length table, which is the row this file was
waiting for. Sections 1–3 below are the *pre-engine* projection and are kept for provenance, but
three of their claims are now known to be wrong and are corrected in place: the projection of
**120–127 tok/s** (§3) is ~52 % above what the engine actually does; the claim that §05 attention
adds **~0.1 ms at a 4k context** (§2) is wrong by two orders of magnitude, because attention here is
dispatch-bound rather than bandwidth-bound and extrapolates to **+9.4 ms**; and "coverage stopped
being the interesting number" (§2) stopped being true the moment attention's context dependence
appeared. Read §6 for current numbers and §1–§3 for how each op behaves in isolation.

## 1. Measured on the GPU (real kernels, real reference data)

| what | traffic | time | bandwidth | vs 237 GB/s probe |
|---|---|---|---|---|
| `lm_head_gemv` bf16 GEMV, 248320 × 1024 | 485.0 MB | **2.105 ms** | 242.1 GB/s | **102 %** |
| `mlp_gemv` MLP, 24 layers, cold weights | 528.5 MB | **2.650 ms** | 199.4 GB/s | **84 %** |
| `proj_gemv` all 9 projections × 24 layers, 48 launches, cold | 445.7 MB | **2.990 ms** † → clean re-takes: **3.039 ms** (2026-10-11), **3.081 ms** (2026-10-12) | 156.3 → **150.1 GB/s** | **66 % → 63–64 %** — the clean number is the lower one |
| `proj_gemv` best config seen (rows/block 32, same conditions) | 445.7 MB | 2.785 ms † **never reproduced clean** | 167.8 GB/s | **71 % is a contended-box figure.** The clean E2 A/B at rows/block 32 gave unpacked **65.1 %** vs packed 58.0 % — see §6 |
| `glue` residual+RMSNorm, fused, 49 launches/token | ~0.5 MB | **0.1336–0.1360 ms** † — **superseded: §6 re-take is 0.1499 ms @512 t / 0.1851 @256, and the row is bimodal (spread up to 269 %)**, so quote it at 256 threads with the spread | 2.73–2.78 µs/launch | launch-bound: roofline is 0.0021 ms |
| same, unfused (48 adds + 49 norms = 97 launches) | ~0.5 MB | 0.2131–0.2155 ms † | 2.20–2.22 µs/launch | fusion buys 0.0795 ms/token |
| `attn_decode` §05 core, 6 layers, one launch each | ~0.4 MB at 25 lanes | ~0.011 ms (launch floor arithmetic, not timed) — **superseded**: measured in situ at **2.301 µs per lane per token**, i.e. 5.34 GB/s effective at 327 lanes | **5.34 GB/s** at 327 lanes | **2.3 % of the probe — dispatch-bound, not bandwidth-bound**; bit-exact to 277 lanes |
| `gdn_decode` recurrence, 18 layers, state warm | 54 MB (3 × 18 MiB) | **0.0646 ms** | 876 GB/s (derived) | 370 % — cache-resident |
| `gdn_gated_norm` §04.5, 18 layers × 8 dec8 steps = 144 launches | ~2.9 MB/plan (0.37 MB/token) | **0.4227 ms/plan** = 2.936 µs/launch | 6.6 GB/s | launch-bound: 0.0016 ms/token of roofline |
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

## 2. Per-component budget: every op now has a real kernel (title was written when it was 65 %)

Weights re-derived from the safetensors headers, grouped as §01.7 (decimal GB, so
`1.5457 GB` text-path total checks out). Roofline column = bytes ÷ 237 GB/s.

| component | GB/token | roofline | measured | status |
|---|---|---|---|---|
| `lm_head` / `embed_tokens` (tied) | 0.5086 | 2.146 ms | **2.105 ms** | **kernel exists, 102 % of roofline** |
| MLP, 24 layers (gate/up/down) | 0.5285 | 2.230 ms | **2.650 ms** | **kernel exists, 84 % of roofline** |
| GDN `in_proj`/`out_proj` + conv, 18 layers | 0.3796 | 1.602 ms | **not measured on its own** — see the split note below | kernel exists (`proj_gemv`); **81 % of the projection bytes** |
| full attention `qkv`/`o`, 6 layers | 0.0881 | 0.372 ms | **not measured on its own** — see the split note below | kernel exists (`proj_gemv`); **19 % of the projection bytes** |
| **the two above together, measured as one 48-launch plan** | 0.4677 | 1.973 ms | **3.081 ms** (2026-10-12 re-take; 3.039 ms on 2026-10-11) | **63–64 % of the probe** — §6 |
| MTP draft layer (only when drafting) | 0.0409 | 0.173 ms | — | not written (§09 MTP exists in the torch oracle only) |
| norms and small tensors | ~0.005 | 0.021 ms | **0.150 ms** (`glue` fused, 49 launches, 512 t) | **kernel exists** — 98 % launch overhead, 3 % roofline |
| GDN state traffic (18 layers, warm) | 0.054 | 0.228 ms cold | **0.0648 ms** | **kernel exists** — 883 GB/s, cache-resident |
| §04.5 gated norm (18 layers) | ~0.0004 | 0.0016 ms | **0.053 ms** | **kernel exists** — launch-bound, 18 × 2.94 µs |
| §05 attention core over the KV cache | context-dependent (~0.31 MB at 25 lanes) | ~0.001 ms at 25 lanes; ~0.11 ms at 4k | **2.014 µs per lane per token** (§6 fit) | **kernel exists — 6.10 GB/s effective, 2.6 % of the probe**; dispatch-bound |
| **sum of the separately-measured parts** | **1.605** | **6.52 ms roofline** | **7.81 ms summed** | **97 % of the token's bytes are covered by a measured kernel**; worst family 63 %, best 101 % |
| **what the engine actually costs** | **1.605 + KV** | — | **11.919 ms/token** at a 25-lane context (83.9 tok/s), **12.614 ms** at 327 lanes | **the sum is not the engine: +4.1 ms/token of launch structure and per-step sync**, of which 1.27 ms is priced in `engine/results.md` and ~2.5 ms is explicitly unattributed |

> **Correction, 2026-10-12.** This table is the sum of separately-measured op times, and the engine
> now measures **11.919 ms/token** end to end at a 25-lane context (§6). Two things follow. First,
> **4.1 ms/token is not accounted for by any op in this table** — 424 launches with a sync per step,
> 115 of them pure dtype casts, plus whatever the interleave costs; `engine/results.md` prices 1.27 ms
> of that as plan structure and leaves ~2.5 ms explicitly unattributed. Second, the summed figure is
> not an upper bound on anything: the engine's per-launch cost is paid in series, which the table's
> "measured" column cannot see because each op was timed in a burst of its own.

**Split note on the two projection rows (2026-10-12).** They carried `— †` because no run has ever
timed the GDN and attention projections as separate plans, and they still have not:
`prototype/proj_gemv` has no subset selector, so the only measured plan is the whole 48-launch set. Do
**not** fill those two rows by adding up the per-shape table in `prototype/proj_gemv/results.md` — that
sweep takes the best tile *per shape* (`PACK_inproj` 92.68 µs, `q_proj` 58.31, `o_proj` 36.11, `k_proj`
11.23 …) and its GDN+attention sum is **3.54 ms**, i.e. 15 % *worse* than the shipped 48-launch plan at
3.081 ms. That gap is a finding, not a rounding error: **best-config-per-shape is not the shipped
plan**, so per-shape numbers cannot be recombined into a plan total. Closing those two rows properly
needs a `--only gdn|attn` selector in `proj_gemv` and one clean timing run each — cheap, and queued in
`bench/plan-2026-10-12.md`.

So **1.5588 / 1.605 = 97 %** of the token's bytes are now covered by a kernel that has been
measured on this box — `lm_head`, the MLP, every projection, the GDN recurrence, and the norms.
What is left is §05's attention core, and that number is a standing warning about the whole
table: **attention's traffic is a function of context length, not of the model.** At the fixture's
25 lanes the KV read is ~0.4 MB/token, which is why `attn_decode` is launch-bound and costs
~11 µs. ~~At a 4k context the same six layers read ~25 MB/token — an extra ~0.1 ms that this table
cannot see, because the fixtures are short.~~ **Corrected 2026-10-12: that ~0.1 ms was traffic
arithmetic and it is wrong by two orders of magnitude.** Measured in situ, attention costs
**2.301 µs per lane per token** (§6), so a 4k context adds **≈9.4 ms/token**, not 0.1 ms — the engine
would fall from 83.9 tok/s to roughly 10. The bytes really are only ~25 MB (≈0.1 ms of roofline); the
cost is that the kernel reads them at 5.34 GB/s, i.e. the kernel is the problem, not the memory.
Any tok/s number taken at prefix-24 is therefore not merely an optimistic bound on long-context
behaviour, it is a number from a different regime. Coverage stopped being the interesting number; the spread did —
**102 % / 84 % / 66 %** for `lm_head` / MLP / projections, and within the projections alone the
per-shape results run from 77 % of the probe (`in_proj_qkv` packed) down to 39 % (`v_proj`) and
5 % for the 16-row `in_proj_b`/`in_proj_a`. The wide shapes behave like `lm_head`; the narrow ones
behave like nothing measured before, and that is what moved the projection below.

† **The projection and glue numbers below are the original contended ones.** Both were taken while
something else worked on the GPU without the lock — `glue`'s two windows read 94 %→75 % and 98 %→35 %
`util`, and an earlier one 17 %→60 % — so they are floors, not measurements. **§6 now supersedes both**
(2026-10-11 and 2026-10-12 clean re-takes: projections 3.039 → 3.081 ms at 63–64 %; glue fused 0.1496 ms
at 512 threads), and the 66 % these rows claim is 63–64 % measured clean. Rows in `glue`'s own sweep with 9–331 % spread are marked not-quotable in its `results.md`
rather than quietly dropped. One real knob came out of the contended data anyway, because it is a
*ratio*: µs/launch depends on block width — 2.776 / 3.06 / 3.75 / 5.30 µs at 1024 / 512 / 256 / 128
threads, which means the 1.85 µs launch floor from the probe is a floor for a narrow block only, and
per-launch cost at a useful block width is ~50 % above it.

## 3. Projection, with the assumptions named

| assumption | value | risk |
|---|---|---|
| ~~remaining GEMVs reach the same fraction of the probe the MLP did~~ **measured** | projections land at **66 %**, not 84 % | no longer an assumption — the per-shape spread is 39–77 % of the probe and the weighted result is 66 %. The row-count theory is dead: `in_proj_qkv` has 6144 rows and reaches 77 %, `k_proj`/`v_proj` have 512 and reach ~40 %, while `lm_head` with 248320 rows reached 102 % |
| launch floor | +0.36 ms (200 launches × 1.8 µs), partly hidden by overlap | **medium** — the projections alone are 48 launches/token and the small ones are already launch-dominated: `in_proj_b`/`in_proj_a` cost 3.2 µs/launch to move 32 KiB, which is why they are folded into the packed launch |
| weights already resident (no H2D per token) | assumed | low — H2D is 76–82 GB/s, so it would dominate if violated |
| no fp32 upcast traffic for accumulation | assumed | low — demonstrated twice now: MLP and projections both accumulate in F32 registers and move no extra bytes |
| MTP drafting disabled | assumed | drafting adds 0.0409 GB + an extra logits read per accepted burst |

**Projected single-stream bf16 decode: ~~7.9–8.3 ms/token = ~120–127 tok/s~~ → measured
11.919 ms/token = 83.9 tok/s at a 25-lane context (§6), so the projection was ~52 % high.**
That was a second revision *downward* at the time (was 145–153, then 137–147), and the cause is the
same both times and is now fully identified: the §07.4 roofline assumes every byte stream arrives at
probe speed, and each family measured so far arrives slower than the one before it — 102 %, then
84 %, then 66 %, and finally attention at **2.3 %**. 7.81 ms of the token is now measured in
isolation and the engine costs 11.919 ms, so the remaining error is not in the bytes at all: it is
launch structure (424 launches/token, 115 of them casts) and attention's context term. §07.4's
headline 151 tok/s is now roughly 80 % above the measured engine and should be treated as wrong on
this box.

The two small ops confirm that framing rather than changing it: `glue` at 0.137 ms/token is 97 →
49 launches worth of saving (fusion bought ~0.079 ms/token, ~1 % of the budget, which is exactly
what a launch-count item should return), and `attn_decode` is ~11 µs/token of dispatch at short
context. Neither moves the projection. What moves it is the 66 %.

The one lever that is already visible: ~~`rows/block` 32 measured 8 % better than the shipped default
of 8 on the projections (2.785 vs 2.990 ms), which is ~0.2 ms/token~~ **resolved by the clean E2 A/B in
§6: at rows/block 32, unpacked beats packed (0.0528 ms/launch = 65.1 % vs 0.0593 = 58.0 %), and the
2.785 ms / 71 % figure never reproduced on a clean box.** `PACK_X=false` is therefore the shipped and
confirmed choice, and the rows/block 32 gain is **not** ~0.2 ms/token — it is inside the noise of the
63–64 % ceiling. The narrow-shape deficit (39–49 % for `k_proj`/`v_proj`/`o_proj`/`out_proj`, 5 % for the
16-row `in_proj_a`/`in_proj_b`) is what caps the family, and it is a shape problem, not a tile problem.

## 4. What *is* measured end-to-end today (and is not the engine)

> **Superseded in part, 2026-10-12.** The engine now exists and is measured — see §6's engine rows
> (11.919 ms/token at 25 lanes, 83.9 tok/s). What follows is kept because the CPU oracle's numbers are
> still the correctness reference and the comparison is still informative.

`speccheck/` runs the whole model in torch, CPU-only, bf16: **~24 tok/s** greedy, **~21
tok/s** with `--mtp 1` (accept 0.660, 1.24 tok/forward). That is the correctness oracle.
It is 6× below the GPU projection because the CPU path reads at 75–117 GB/s at best and
torch adds per-op overhead on ~200 ops/token. It is quoted here so the number is on the
record as *not* the engine's performance.

## 5. What has to happen before this file can say "measured"

> **Done, 2026-10-12.** Item 0 below was executed as §6 ("Clean-window re-take"), with the engine rows
> added afterwards in the same regime. The numbers in §1 marked † are floors, not measurements, and stay
> marked; §6 supersedes them wherever it has a row.

0. **A quiescent GPU pass, run by the main agent with no subagents live.** Every device command run
   today — `mlp_gemv`, `proj_gemv`, `attn_decode`, `glue` gates and the cold timing sweeps — was run
   while other agents were active on the box, which the new serialization rule forbids. None of
   those numbers may be quoted until this pass re-takes them in one order:
   (a) `prototype/glue/run.py` full, (b) `prototype/proj_gemv/run.py --layers all --steps all
   --negative` + its cold sweep, (c) `prototype/mlp_gemv/run.py --layers all --fixture dec8
   --negative` + cold, (d) `prototype/attn_decode/run.py --long --negative`, (e) `engine/loader_test
   --no-verify` for the startup number, (f) the engine's own gate and the first tok/s row, (g) the
   position-64 pre-rope reduction attribution described below. Record
   `gpu_lock.sh util` at both ends of each and state the repeat spread; anything without both is
   provisional and says so.

   (g) **position-64 pre-rope reduction attribution.** `prototype/attn_decode`'s kernel is 1 element
   of `q_rope` off the pin at `p = 64` (2 at layer 11) and 0 off at the other 276 positions, while
   `speccheck/rope_table_probe.py` shows torch's F32 table and a plain glibc `cosf`/`sinf` reproduce
   the pin at 0-of-277 wrong — so the device table, not the reference, is the outlier, and no ±2 ULP
   nudge of any single frequency index's cos or sin explains the difference. Leading candidate is the
   kernel's **pre-rope F32 reduction**: the mean of squares in §05.2's RMSNorm, summed warp-tree here
   and by torch's reduction in the pins. To close it: dump `z` (post-`q_norm`, pre-rope) for
   `prefill064.L03` from the kernel and diff it element-by-element against `specimpl.rmsnorm_zc` on
   the same pinned `q_proj.out` row. If `z` differs, a graduated kernel has a reduction-order bug to
   fix; if `z` is identical, the residual is in the device's `cosf`/`sinf` and gets documented as a
   device-libm boundary instead. Either way the footnote in `prototype/attn_decode/results.md` §4
   ("unattributed") gets replaced by an answer.

   **Answered 2026-10-12, and it is the function.** The `z` dump turned out to be unnecessary: §05.3
   copies the dims past `ROT_DIM` straight out of `z`, so the gate's existing `q_rope` stage already
   carries a 192-dim-per-head witness for the norm. In that witness the device is clean at every
   attention layer at `p=64`, the device's own `rope_cs` dump differs from torch's table in exactly 2
   of 128 entries (`cos[3]` and its duplicate, one BF16 step low), and the CPU path handed that table
   reproduces the device's `q_rope` bit-for-bit at all six layers. `prototype/attn_decode/prerope_probe.py`
   also re-permutes §05.2's sum of squares on the pinned rows — including the kernel's exact warp tree
   — and finds it moves 0 elements at `p=64` while moving 20–207 elsewhere, so the reduction is
   exonerated twice over. Full numbers, the two corrections to `results.md` §4 that came out of it, and
   the reproduction commands: `prototype/attn_decode/results.md` §4. No timing was taken for this and
   nothing here is a bandwidth or latency figure.

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

## 6. Clean-window re-take (2026-10-11, all subagents stopped, measurements run in the foreground)

Method, because it is the point: every row below was measured while **no subagent was running and the
supervisor's own turn was blocked inside the measurement command**. A blocking tool call pauses token
generation, so `flash_serve` — which otherwise busy-polls at ~130 % CPU — was not emitting work during
the burst. This is the first set taken under the serialization rule, and it refutes what
`bench/review-2026-10-11.md` §4 claimed: a clean-enough window *is* reachable from inside the session,
you just have to be the one holding the turn. `gpu use` read 24–91 % across these runs and did **not**
predict the numbers, so util is recorded and then discounted.

| component | this window | repeat spread | ledger §1 | verdict |
|---|---|---|---|---|
| `lm_head_gemv` fused | **2.107–2.110 ms** = 241.0–241.3 GB/s (101.8 % of probe) | 0.15 % (5 bursts × 8, ×3 runs) | 2.105 ms / 242.1 GB/s | reproduces |
| MLP cold (24 layers, tile 6) | **2.605 ms** = 202.9 GB/s (86 % of probe) | 1 run + 1.7 % vs earlier run's 2.650 | 2.650 ms (84 %) | reproduces |
| Projections cold (48 launches) | **3.039 ms** = 153.8 GB/s (65 % of probe) | 1.6 % vs the §1 figure | 2.990 ms (66 %) | reproduces |
| glue fused, 49 launches | **0.1499 ms @512 t / 0.1851 @256 / 0.2563 @128** | **spread 2.7–269 %** — see below | 0.1336–0.1360 | **worse, and unstable** |
| glue unfused, 97 launches | 0.2455 ms @256 t | spread 2.7–4.2 % | 0.2131–0.2155 | worse ~15 % |
| `engine/loader_test --no-verify` | 50 checks PASS; wall **383.9 / 509.0 ms** | 2 runs, 32 % | 794 ms wall | PASS re-confirmed; wall depends on page-cache state, do not compare across runs |
| glue gate | 784 comparisons, 0 failures, both residual adds bit-exact | deterministic | 784 / 0 | reproduces |
| glue gate **at 1024 threads** (engine experiment E1) | 784 comparisons, 0 failures; `g1_n` gated 0 / raw 1, both adds bit-exact, all negatives caught | deterministic | — (never run before) | **PASS at the same floor as 256** — see E1 below |
| `proj_gemv` PACK_X A/B, cold, rows/block 32 (E2) | **unpacked 0.0528 ms/launch = 65.1 % of probe** vs **packed 0.0593 = 58.0 %** | one A/B pair, same window | 66 % / 60 % at rpb 8 | **unpacked wins at every rows/block** — see E2 below |
| `gdn_decode` recurrence, 18 layers warm (re-take of §1's disowned row) | **0.0646 ms** best, worst 0.0650 | **spread 0.5 %** (warm burst + `--trials` bursts) | 0.062 ms (never re-taken, `d81ff13`-dead timer) | **replaces it**; `rec` alone 0.0064 ms |
| `gdn_gated_norm` device gate (§04.5) | `DEVICE GATE PASS`; worst ndiff **1/2048**, max ULP 1, zeros 0 / NaN 0, all six negatives localise | deterministic | — (first pass ever) | **PASS** — see §04.5 row below |
| `gdn_gated_norm` burst | **0.4227 ms** per 144-job plan = **2.936 µs/launch** | **spread 1.4 %** | — (never timed) | ≈0.053 ms/token at 18 GDN layers |
| proj gate | 912 comparisons, 0 failures | deterministic | 912 / 0 | reproduces |
| **`engine/decode_step`, one token end to end — the first engine row** | **11.978 / 12.053 / 12.232 ms/step = 83.5 / 82.9 / 81.7 tok/s** | **spread 2.1 %** (3 runs × 81 steps) | 8.03 ms op-sum ≈ 124 tok/s | **the engine runs, and it is 1.49× the sum of its parts** — provenance below, gap attributed in `engine/results.md` |

Token sum from this window: 2.108 + 2.605 + 3.039 + 0.150 + 0.0646 (GDN state, re-taken) + 0.053
(§04.5 gated norm, 18 launches) + ~0.011 (attention core at fixture cache sizes) = **8.03 ms →
~124 tok/s**. The §3 projection of 120–127 tok/s stands.

### Two findings that came out of the re-take

**hipEvents are useless here at 2 ms, not just sub-millisecond.** `lm_head_gemv`'s event loop returned
`0.000 ms → inf GB/s` on every run today, including best-of-40 on a kernel that wall-clocks at 2.11 ms.
The AGENTS trap row says the event floor is "below ~1 ms"; it is at least that high, and the row is now
too narrow. Fixed in `d81ff13` by giving this prototype the same steady_clock burst the other four
already had — which also means `lm_head_gemv` was the only timing on this box that depended on the dead
path, and the historical 242.1 GB/s was evidently taken while events still worked (or by another path).

**Launch-bound numbers are bimodal on this box; bandwidth-bound ones are not.** The three GEMV
families repeat within ~2 % window-to-window, but glue's 49-launch fused path has a best-of-N spread
of 2.7 % at 256 threads and **158–269 % at 512**, i.e. the distribution has a tail several times the
best and "best of N" is picking a lucky mode rather than estimating a cost. Today's best (0.1499 @512)
is *worse* than the ledger's 0.1336 despite the cleaner window, so the historical figure was the tail,
not the machine. Practical consequences: quote glue at 256 threads (0.185 ms) rather than 512 until
someone characterises the bimodality, and treat per-launch µs figures — 2.545 @256 here vs the 1.85 µs
narrow-block floor — as dispatch contention for a shared queue, not kernel cost. It is ~1.7 % of the
token either way, so this is a measurement-integrity finding, not a budget one.

### The four rows that were owed, and what came back

**`gdn_decode`'s state row is now a re-take, not an inheritance.** Best **0.0646 ms** over 18 layers of
warm state, worst 0.0650, **spread 0.5 %**; `rec` alone (one layer) **0.0064 ms**. That replaces the
0.062 ms the `d81ff13` commit disowned — taken by the per-iteration `hipEventElapsedTime` loop that
cannot see sub-millisecond work at all — so the row finally has a timer, a `util` record and a spread
behind it. It is 4 % slower than the number it replaces, which is the direction a wrong timer predicts.
`check.py` re-ran clean in the same window: **F1–F4 still PASS**, `o` within 1 BF16 ULP of the pinned
dump on all 18 layers, with 3 layers holding 1-ULP ties. Bandwidth for the row is derived from the same
56.6 MB of traffic the old row used (876 GB/s = 370 % of the probe), so "above the ~240 GB/s package
ceiling" is still an open question this re-take does not settle — `check.py` still prints % of probe and
asserts nothing about it.

**E1 is closed: `glue` at 1024 threads passes with a legal error profile identical to 256.** 784
comparisons, 0 failures, the norm half at the published floor (`g1_n` gated 0 / raw 1) and both residual
adds bit-exact, and every negative still lands on its own stage. The point of the experiment was that
thread count changes `PER` and therefore the F32 summation order of the mean of squares, and a norm's
floor is 1 ULP with no cancellation band, so only `ndiff` could tell the widths apart — and it did not.
Consequence, stated the other way round because it is the useful direction: **`engine/kernels_glue.cpp`'s
`<256>` is a measured preference, not a numerical requirement.** Nothing about the engine's arithmetic
would break if it launched 1024; whether it should is now only the performance question glue's own §4
sweep raises (≈2.78 vs 3.75 µs/launch), with the bimodality caveat that file already carries.

**E2 is closed: `PACK_X=false` is the right constant.** Cold, on the prototype's 48-launch token plan, at
rows/block 32: **unpacked 0.0528 ms/launch = 65.1 % of the read probe** against **packed 0.0593 ms/launch
= 58.0 %** — unpacked wins at every rows/block swept, not just the gated 8. So the ≈48 cast launches the
unpacked form bills back (≈0.089 ms at the 1.85 µs floor) do not overturn the ordering, and
`kernels_proj.cpp` keeps its F32 `x` with a measurement instead of an inference. The half of E2 that
awaited `decode_step` — the same A/B through `engine/kernels_proj.cpp`'s own launcher — is still owed;
the engine has a launcher on that path now, so it is runnable.

**§04.5 now has a device row, and it is a PASS.** `prototype/gdn_gated_norm/run.py --gate` over 18 GDN
layers × 8 `dec8` steps: `gated_out` worst case **ndiff 1/2048** with **max ULP 1**, zeros 0 / NaN 0, and
all six injected negatives localising (each moves `gated_out`, the four epilogue ones leave
`norm_pre_gate` at 0/294912). Max ULP 1 is exactly the floor the CPU `neg_probe.py` predicted for a
128-term reduction done in a different order than torch's, so this is the op sitting on its own floor,
not a band that happens to be wide. Timing: **0.4227 ms per 144-launch plan = 2.936 µs/launch, spread
1.4 %** → **≈0.053 ms/token** for the 18 launches a real token makes. Worth 3.5 % of the token budget,
which makes the packed/3-in-1 launch question here a real later lever, not a rounding error.

The first two device attempts at this gate **failed 144/144 and the kernel was never at fault**: the
gate's own fixture writer emitted `z.to(torch.uint16)` on a float32 tensor, which converts instead of
reinterpreting (0.195 became the integer 0, −1.3 wrapped to 0xFFFF), so the kernel read +0 and NaN where
the pin had neither. The full symptom-by-symptom account is `prototype/gdn_gated_norm/results.md` §8 and
the trap is a row in `AGENTS.md`; recorded here because a PASS with that history needs its provenance
next to it, not three files away.

### The first engine row, and what it cost against the op-sum

Provenance, because this is the row §5 item 0(f) was asking for. Three runs of
`engine/build/decode_run` in the same foreground, zero-subagent window as every other row in this
section — blocking call, so `flash_serve` had nothing to emit for the duration — `gpu_lock.sh util`
**76 / 71 / 69 %**, recorded and then discounted exactly as §6's opening rule discounts 24–91 % for the
rows above it. Each run was `RESULT steps=81`: the 17-token `dec8` prompt fed **one token per decode
step** (§07.1 has no prefill yet — see the caveat below) plus 64 requested generation steps, and
generation ended when the engine emitted id **248046** at generation step 54. `ms_per_step` is
`decode_run`'s own `steady_clock` mean over the steps actually run — one sync per step, at the logits,
which is what a token server pays anyway. **This is a warm loop**: unlike the `14.631 ms/step` that
`engine/results.md` retired (a first pass over freshly allocated state, measured by a gate run that was
not there to measure), these steps run after the state and scratch have been through the pipeline.
Spread is 2.1 %, which is bandwidth-bound behaviour, and that is the expected shape: the step is
~1.6 GB of weight traffic, not a launch-bound microbench, so §6's bimodality warning does not apply to
it.

Two things this row says, and one it does not.

* **It replaces the projection's claim to be the nearest number to the engine.** §3 projected
  120–127 tok/s and §6's op-sum gives 8.03 ms ≈ 124 tok/s; the engine delivers 83.5. The projection is
  now what it always was — a sum of per-family cold bursts, each measured in its own tight loop — and
  the gap between it and the engine is **3.95 ms/token**, which is 0.94 GB of traffic at the 237 GB/s
  probe. Stated the other way round: the whole token streams at **134 GB/s = 57 % of the probe**,
  against the **200 GB/s = 84 %** the sum of its parts implies.
* **Part of that gap is the plan, not the machine.** ~1.27 ms of it is itemised in
  `engine/results.md` (the engine runs the 114-launch unpacked projection plan where the ledger row
  bills the 48-launch packed one; the 115 cast launches are in no ledger row at all), and another
  ~0.15 ms is the host tail behind the per-step sync. The remaining ~2.5 ms is **not attributable from
  any measurement in this file**, and says so.
* **What it does not say: anything about time-to-first-token.** There is no prefill, so all 17 prompt
  steps cost a full decode step each — 17 × 12 ms ≈ 0.2 s of prompt before the first generated token.
  A chunked prefill pass would remove most of that, so this row is a steady-state decode figure only,
  and any first-token number quoted from it would be ~17× wrong.

One open provenance item, flagged rather than hidden: these runs reported `ms_per_step` from a driver
that may or may not have written the per-step logits dumps (`--out`, 0.993 MB per step). Writing that
much to the page cache costs **0.254 ms/step** measured on the CPU, so if the dumps were on, ~2 % of
each figure is the driver, not the engine. The A/B (same three runs without `--out`) is in the hand-off
in `engine/results.md`.

### Engine decode cost vs context length (2026-10-12, foreground, zero subagents)

Provenance, same rules as the rows above: three `engine/build/decode_run` runs per point, blocking
in a foreground tool call with **zero subagents active**, `gpu_lock.sh util` **92 / 91 %** at acquire
and release and discounted on the same grounds as every other row here. `ms_per_step` is
`decode_run`'s own `steady_clock` mean over all steps actually run (prompt fed one token per step —
§06 prefill does not exist yet — plus the requested generation steps, one sync per step at the
logits). Spread is (max−min)/min over the three runs.

| fixture / prompt | lanes at the last step | `ms_per_step` best of 3 | all three runs | spread | tok/s |
|---|---|---|---|---|---|
| `dec8` (17-token prompt, 8 steps) | 25 | **11.919** | 11.919 / 11.966 / 12.045 | 1.06 % | 83.9 |
| `dec64t` (27-token prompt, 64 steps) | 91 | **12.187** | 12.187 / 12.304 / 12.487 | 2.46 % | 82.1 |
| `dec64t` (27-token prompt, 300 steps) | 327 | **12.614** | 12.614 / 12.838 / 12.904 | 2.30 % | 79.3 |

Reproduce:

```bash
tools/gpu_lock.sh acquire <tag> && tools/gpu_lock.sh util
PT=$(.venv/bin/python -c "import json;print(','.join(map(str,json.load(open('data/refs/dec64t/manifest.json'))['prefill_ids'])))")
for i in 1 2 3; do mkdir -p /tmp/psw/$i; ./engine/build/decode_run \
  --model models/Qwen3.5-0.8B --ids "$PT" --steps 300 --out /tmp/psw/$i; done
tools/gpu_lock.sh release <tag>
```

**The finding is the slope, not the levels.** Everything the engine does is context-independent except
the §05 attention read over the KV cache, so the difference between these rows *is* the attention
kernel's cost: **(12.614 − 11.919) / (327 − 25) = 2.301 µs per lane per token**, i.e. 0.384 µs per
lane per attention layer. At 2048 B of KV per lane per layer (§00.5: 2 kv-heads × 256 × 2 B × k and v)
that is **5.34 GB/s effective — 2.3 % of the 237 GB/s probe**. For the whole 300-step run the engine
reads 3.83 MiB of KV per token, which the probe says should cost ≈16 µs; it costs ≈700 µs.

Two consequences, stated as arithmetic and not as measurements:

1. **Attention is dispatch/parallelism-bound, not bandwidth-bound**, by a factor of ~40. §07.4's
   correction that "skinny argmax is bandwidth-critical" was already shown wrong for `lm_head`; this is
   the same lesson on the other hot kernel. The fix is the block-parallel form (more blocks over the lane
   axis, online `(max, sum)` merge per §07.3), not a faster load.
2. **Linear extrapolation puts attention at +9.4 ms/token at a 4096-lane context** — which would take the
   engine from 83.9 to roughly 10 tok/s. That number is an **extrapolation from a 25–327 lane window, not
   a measurement**, and it must not be quoted as one. Measuring it is currently blocked by `MAXSC = 512`
   in `engine/kernels_attn.cpp`, which throws rather than truncating — so the split-block work that was
   filed under "capability, later" is now on the performance critical path.

Next measurement owed (main agent, before any attention rewrite): `prototype/attn_decode` timed in
isolation at 64 / 256 / 512 / 1024 lanes, so the rewrite is priced against a curve rather than against
this single slope.

### Per-component re-take, whole board (2026-10-12, zero subagents, blocking calls)

Every op family re-measured in one sitting, to see which of the §6 numbers from 2026-10-11 survive.
`gpu_lock.sh util` read 93 % at acquire and release, discounted as always; the numbers below were
taken in the foreground with nothing else running, which is the condition that makes them repeat.

| component | 2026-10-11 | 2026-10-12 best of 3 | spread | verdict |
|---|---|---|---|---|
| `lm_head_gemv` | 2.107–2.110 ms = 241.0–241.3 GB/s | **2.1266 / 2.1274 / 2.1321 ms** = 238.5–239.1 GB/s | 0.26 % | reproduces within **0.8 %**; still ~101 % of probe |
| MLP cold, 24 layers | 2.605 ms = 202.9 GB/s (86 %) | **2.594 / 2.597 ms** = 203.5–203.7 GB/s (86 %) | 0.1 % | reproduces, marginally better |
| projections, 48 launches | 3.039 ms = 153.8 GB/s (65 %) | **3.081 / 3.114 ms** = 150.1–151.7 GB/s (**63–64 %**) | 1.1 % | same story, 2.4 % slower; the 66 % of §1 is not achievable |
| `glue` fused, 49 launches @512 t | 0.1499 ms | **0.1496 ms** (3.7 %) | 3.7 % | identical. **@256 t: 0.1904 ms but spread 62 %**, @128 t: 0.2576 ms, spread 83 % — the bimodality is still there and is a property of narrow blocks, not of the 10-11 window |
| `glue` unfused, 97 launches @512 t | 0.2455 ms | **0.2228 / 0.2228 / 0.2257 ms** | 1.3 % | fusion still buys **0.073–0.076 ms/token** |
| `gdn_decode` recurrence, 18 layers | 0.0646 best / 0.0650 worst (0.5 %) | **0.0648 ms** best, worst 0.0691, 20 launches × 3 trials | **6.6 %** | same level, worse tail than the 10-11 re-take — quote 0.065 ± 6 % |
| `gdn_gated_norm` §04.5, 144-job plan | 0.4227 ms = 2.936 µs/launch (1.4 %) | **0.4260 ms** = 2.958 µs/launch | **0.3 %** | reproduces; gate PASS in the same run |
| engine, context slope (below) | 3 points | **5 points**, `R² = 0.9475` | — | slope tightened from 2.30 to **2.01 µs per lane-token** |

Commands, group 1 (caller holds the lock — these three do **not** self-lock):

```bash
tools/gpu_lock.sh acquire <tag>
.venv/bin/python prototype/lm_head_gemv/run.py --iters 8
.venv/bin/python prototype/mlp_gemv/run.py --layers all --fixture dec8
.venv/bin/python prototype/gdn_decode/check.py --iters 20
tools/gpu_lock.sh release <tag>
```

Group 2 (**do not hold the lock** — `glue`, `proj_gemv` and `gdn_gated_norm` acquire it themselves and
will park until it frees; this cost a stalled run and a stale-holder entry today):

```bash
.venv/bin/python prototype/glue/run.py --time          # sweeps 128/256/512 t, fused and unfused
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps all --time
.venv/bin/python prototype/gdn_gated_norm/run.py --gate --time
```

**Attention, five points instead of two.** Same engine, one variable (context), `dec64t` prompt,
`decode_run --steps {8,64,128,250,300}` giving 25 / 91 / 155 / 277 / 327 lanes at the last step:

| lanes | best of 3 `ms_per_step` | all three runs | spread |
|---|---|---|---|
| 25 | 11.919 | 11.919 / 11.966 / 12.045 | 1.06 % |
| 91 | 12.187 | 12.187 / 12.304 / 12.487 | 2.46 % |
| 155 | 12.241 | 12.241 / 12.247 / 12.455 | 1.75 % |
| 277 | 12.420 | 12.420 / 12.658 / 12.687 | 2.15 % |
| 327 | 12.614 | 12.614 / 12.838 / 12.904 | 2.30 % |

Least-squares over the five points: **slope 2.014 µs per lane-token, intercept 11.924 ms,
`R² = 0.9475`, max residual 0.080 ms.** That is **6.10 GB/s effective per attention layer against the
237 GB/s probe**. Extrapolating the fit (still an extrapolation, still not a measurement):
**+2.06 ms/token at 1024 lanes (71.5 tok/s)** and **+8.25 ms/token at 4096 lanes (49.6 tok/s)**. The
4k figure is the reason the block-parallel attention rewrite is a performance item rather than a
capability item, and `MAXSC = 512` is the reason it cannot yet be measured directly.

Board-level conclusion, stated once: **no op family is more than 2.5 % off its previous clean-window
number**, so the ledger's per-op rows are stable; the engine's 11.92 ms/token is not explained by any
single op being slower than the board says, and the two named residues (4.1 ms of launch structure
per §2's correction, and attention's context term) are where the work is.

### Cold weight-restore (2026-10-12, zero subagents, foreground, lock `main-cold-restore`)

| measurement | best of 5 | median | worst | spread | source |
|---|---|---|---|---|---|
| cold weight-restore: text weights made resident in a fresh process (device pool empty, host page cache warm) | **454.4 ms** | 510.2 ms | 804.4 ms | **77.0 %** | `tools/cold_restore.sh --runs 5` → `engine/build/loader_test --no-verify` |

`gpu util` read 86 % at acquire and 16 % at release; discounted per the standing rule, the gate on
validity is zero subagents plus a blocking foreground call.

Phases of the best run: cold H2D **85.2 ms**, pinned staging **59.4 ms**, first GPU read pass over the
fresh pages **8.8 ms**; the remaining ~300 ms is header parse, device slab allocation and file → pinned
staging. The 77 % spread is one outlier (run 2: 804 ms, with its warm read pass at 42 ms instead of
~8.5 ms), which is the page-fault behaviour `devices/strix-halo-unified-memory.md` already describes for
a first pass over fresh pages.

What this number is for: it is a **once-per-process** cost, not a per-token one — 454 ms against an
11.92 ms/token steady state, so a cold start is ~38 tokens' worth of generation and no spread in this
row changes anything downstream. It closes the last `cold weight-restore` debt from
`bench/gate-audit.md`, and it is the number that bounds any future "restart the engine per request"
idea: at 454 ms it is affordable exactly zero times inside a conversation.

### Long-context decode and time-to-first-token (2026-10-13, zero subagents, foreground, locks `main-ctx-final` and `main-ttft3`)

**This is the row that `MAXSC = 512` blocked, and it is also the first correction to it.** P7 removed the
lane ceiling, so the 4k point is now measurable instead of extrapolated. Two things had to be fixed in
method before any number could be quoted:

- **`RESULT ms_per_step` stopped being a decode cost in P2.** It averages every token the run priced, and
  since P2/P4–P6 the prompt is a wave with amortised launches, so the mean fell to ~2 ms/token and says
  nothing about decode. The decode rows below are therefore **differential**: two runs that differ only in
  `--steps` (32 vs 64), same prompt, same prefill in both, so `(ms_total_B − ms_total_A) / 32` is the cost
  of a generation step at that context. `tools/context_sweep.py`.
- The old extrapolation in this file (**+9.4 ms/token at 4096 lanes, "~10 tok/s", flagged there as an
  extrapolation from a 25–327 lane window and not to be quoted as a measurement**) is now refuted by a
  measurement. It over-predicted by ~2.6×.

| lanes at last step | marginal ms/token, best of 3 | all three | spread | tok/s |
|---|---|---|---|---|
| 128 | **12.863** | 12.863 / 12.966 / 13.051 | 1.47 % | 77.7 |
| 512 | **13.942** | 13.942 / 13.975 / 14.469 | 3.78 % | 71.7 |
| 1024 | **13.793** | 13.793 / 13.861 / 14.348 | 4.02 % | 72.5 |
| 2048 | **13.634** | 13.634 / 13.882 / 13.969 | 2.46 % | 73.3 |
| 4032 | **16.311** | 16.311 / 16.635 / 17.089 | 4.77 % | 61.3 |

`gpu util` 94 % at acquire, 46 % at release; discounted as everywhere else here. Prompts are repeated
prose truncated to length and checked stop-id-free — the lane count is what matters for this row, not the
tokens' meaning, and that is stated rather than hidden.

**Reading it.** The context term is **(16.311 − 12.863) / (4032 − 128) = 0.89 µs per lane per token**, not
2.3 µs: the ≤327-lane slope was dominated by fixed launch cost and did not extrapolate. The curve is flat
from 512 to 2048 and only climbs past 2k, which is consistent with P7's split-K path parallelising the
lane axis above the 512-lane staging — the same "attention is dispatch-bound, not bandwidth-bound" finding
as before, now with the fix in place and still only at 61 tok/s at 4k, so the attention read remains the
hot path. 4k single-stream decode is **viable, not fast**.

| prompt lanes | ms to first sampled token, best of 3 | all three | spread | prompt tok/s |
|---|---|---|---|---|
| 64 | **89.4** | 89.4 / 89.9 / 90.5 | 1.31 % | 716 |
| 256 | **347.5** | 347.5 / 347.6 / 348.0 | 0.13 % | 737 |
| 1024 | **1610.2** | 1610.2 / 1614.5 / 1620.3 | 0.63 % | 636 |
| 3072 | **5407.1** | 5407.1 / 5441.5 / 5450.2 | 0.80 % | 568 |

`tools/prefill_sweep.py`, `ms_total` with `--steps 1` (prompt wave plus one priced step, weight load
excluded — the 454.4 ms cold restore above is a separate, once-per-process cost). Before P2 the prompt was
fed one token per decode step, so 3072 tokens would have cost roughly 3072 × 12.9 ms ≈ **40 s** of TTFT;
**5.4 s is ~7× better**, and that is the derived comparison, not a measured old-path row. What it is still
not: good. A 3k-token chat prompt waits five and a half seconds, because prefill throughput is ~570–740
tok/s while the per-token GDN recurrence and per-token attention remain serial — P9's chunked scan is the
lever, and this table is what prices it.

**And a new ceiling to mind.** P7's split is capped at `kAttnMaxSplit = 8`, so the engine's lane ceiling is
now **512 × 8 = 4096** and it refuses fail-closed past it: the first attempt at this row died with
`attn_decode_layer: lvalid=4097 needs split=9`. The boundary is therefore **total session lanes** — prompt
*plus* everything generated — so a 4096-token prompt cannot produce even one token, and "4k support" as
shipped means a window that the whole exchange fits inside, not 4k of prompt *plus* 4k of generation.
Raising it means raising `attn_part_floats()` scratch alongside the constant, and
`engine/gate.py::lane_ceiling()` reads both rather than trusting a number; P8b priced that and left the
constant at 8 with the arithmetic in `bench/plan-prefill-2026-10-12.md` §3.1.

Reproduce:

```bash
tools/gpu_lock.sh acquire <tag> && tools/gpu_lock.sh util
.venv/bin/python tools/context_sweep.py --lanes 128,512,1024,2048,4032 --steps 32 --runs 3
.venv/bin/python tools/prefill_sweep.py  --lanes 64,256,1024,3072 --runs 3
tools/gpu_lock.sh release <tag>
```

### P8's payoff: attention leaves the token loop (2026-10-13, zero subagents, foreground, locks `main-p8b-verify` then `main-p8b-timing`)

P8b landed the engine half of P8 (`f4a1095..88fd712`): one attention site per attention layer per body,
with the plan chosen by row count — P7's fused single launch for one row, `attn_prep` + `attn_core` for a
wave. The worker took no timings by rule; these are mine, same two harnesses and same protocol as the two
tables directly above, so they are comparable row-for-row. First, the identity check, on device, my own
run of all eight fixtures after rebuilding from HEAD: `dec8 168`, `dec64f 650`, `dec64t 647`,
`dec300f 1156`, `mt32f 391`, `p128 197`, `wide520f 327`, all 0 FAIL, and `chunk64b` at 372 PASS with
**exactly its 3 announced D-9 red lines** and nothing else. Every count is the pre-P8 count **+7**, which is
section `(h)`'s new attention checks — the numbers moved because the gate got stricter, not because the
engine did. `gpu util` 7 % at acquire, 58 % at release; discounted as usual.

**Time to first sampled token, before → after** (`tools/prefill_sweep.py`, `--steps 1`, weight load excluded):

| prompt lanes | before (P7 tree) | **after (P8b)** | Δ | spread after | prompt tok/s before → after |
|---|---|---|---|---|---|
| 64 | 89.4 | **82.7** | −7.5 % | 1.91 % | 716 → 774 |
| 256 | 347.5 | **276.5** | −20.4 % | 0.96 % | 737 → 926 |
| 1024 | 1610.2 | **1124.3** | −30.2 % | 1.87 % | 636 → 911 |
| 3072 | 5407.1 | **4116.9** | −23.9 % | 0.66 % | 568 → 746 |

This is the first change in the project that moved a wall-clock number the user would feel, and the shape is
the expected one: the gain grows with the prompt, because the thing removed is per-query launches, and the
spreads stayed under 2 % so the effect is far outside noise. A 3072-token prompt now waits 4.1 s instead of
5.4 s. Against the pre-prefill era (~40 s for the same prompt, derived above) the whole sequence is roughly
**10×**.

**Decode cost did not move, and I am not going to pretend otherwise** (`tools/context_sweep.py`, differential):

| lanes at last step | before | after | spread after |
|---|---|---|---|
| 128 | 12.863 | **12.949** | 1.77 % |
| 512 | 13.942 | **14.137** | 0.63 % |
| 1024 | 13.793 | **13.903** | 3.12 % |
| 2048 | 13.634 | **13.360** | 5.43 % |
| 4032 | 16.311 | **14.773** (median; best 12.692, worst 17.489) | **37.79 %** |

Four of five points sit inside run-to-run noise, which is the correct result: decode is one row, so it takes
P7's fused path and P8 deliberately did not change its route. The fifth point is worth stating rather than
smoothing — at 4032 lanes the spread blew out from 4.8 % to 37.8 % across three runs, and I quote the
**median**, not the best. That is the dispatch-bound bimodality this file already warns about, showing up at
the widest lane count where the split path runs eight blocks per site; the honest summary is *"4k decode is
~13–17 ms/token and we do not yet know which"*, and a re-measure belongs on the list.

Two debts from this pass, both written up by the worker rather than glossed: the mixed one-pass + streaming
wave grid is priced by engine-vs-engine self-consistency, **not** by a reference pin — pinning it needs a
>512-token-prompt fixture with its own D-2 allowance, and P8b declined to invent one; and `kAttnMaxSplit`
stays 8 with the scratch arithmetic (129 KiB per wave-row, ~64.5 MiB per request at `kMaxWaveRows`) recorded
in `bench/plan-prefill-2026-10-12.md` §3.1 instead of a silent constant bump.

### Q6: the decode step is not launch-bound, and the width change that follows from that (2026-10-13, zero subagents, foreground, lock `q6-rpb-ab`)

Q6's worker built the launch bill for one decode step from the counters — **424 launches per token**, priced at the block width each site actually uses, ≈**1.6 ms of a 12.949 ms token** — and the conclusion it forces is that **launch overhead is not the gap**. The full table is in `engine/results.md`; the shape is: `lm_head` 2.127 ms measured against a 2.146 ms roofline (at ceiling, no lever), MLP 2.594 against 2.230 (86 % traffic-bound), projections 3.54–3.60 against 1.973 — **the single named inefficiency, ~+1.6 ms, and it is a *shape* problem** (`in_proj_a/b` run at 5 % of probe, `k/v_proj` 40 %, `o/out_proj` 46–49 %), glue 0.190 measured against 0.184 derived (pure dispatch, the launch-cost curve agrees within 3 %), casts and D2D copies ~0.35–0.49 ms of dispatch that is the only overhead still unclaimed by a family. Sum of named parts leaves a **≈3.4–3.6 ms residue** — same residue as §6's older "zero-context intercept 11.924 ms vs 8.4 ms of parts", two framings of one unknown.

The one change it shipped is `kRowsPerBlock` 8 → **32** in the projection launcher (`bc61dce`, threads follow the one constant so they cannot drift). Bit-identical by construction — a row is owned end-to-end by one wavefront — and proven: byte-identical across all 40 output files of `dec8` at 64 steps, identical counters, identical gate counts on all eight fixtures, `margin_delta` x1.0000 everywhere including the two tightest lanes on the tree (`dec300f` step 86, `wide520f` step 17). I priced it on device, both binaries in one locked window, differential decode and TTFT, best-of-3 each. **Decode, medians** (medians rather than bests because the spreads are larger than the effect):

| lanes | rows/block 8 | **rows/block 32** | Δ | spread (old / new) |
|---|---|---|---|---|
| 128 | 12.915 | **12.728** | −1.4 % | 3.84 % / 0.38 % |
| 512 | 13.959 | **13.784** | −1.3 % | 18.92 % / 4.49 % |
| 1024 | 13.934 | **13.670** | −1.9 % | 4.64 % / 3.02 % |
| 2048 | 14.157 | **13.781** | −2.7 % | 14.28 % / 5.39 % |
| 4032 | 15.620 | 16.502 (see below) | +5.6 % | 7.07 % / 9.19 % |

Read honestly: at ≤2048 lanes the gain is **1.3–2.7 %, same sign at all four points but inside or barely outside the run-to-run spread**, so it is real but small — consistent with the attribution, which predicted a fraction of a millisecond, not a breakthrough. The worker's derived hope of −0.50 ms/token did not materialise.

**TTFT moved clearly, and that is where the change pays** (spreads ≤1.8 %, so these are far outside noise):

| prompt lanes | rows/block 8 | **rows/block 32** | Δ | spread new | prompt tok/s old → new |
|---|---|---|---|---|---|
| 64 | 84.1 | **79.4** | −5.6 % | 0.34 % | 761 → 806 |
| 256 | 281.1 | **258.6** | −8.0 % | 0.60 % | 911 → 990 |
| 1024 | 1140.4 | **1054.6** | −7.5 % | 0.86 % | 898 → 971 |
| 3072 | 4099.7 | **3841.2** | −6.3 % | 1.80 % | 749 → 800 |

Which is the expected asymmetry, and a good illustration of why the attribution came first: a wider block only helps when there are many rows to fill it, so the wave path eats the win and the one-row decode path does not. **Cumulative prompt speed for a 3072-token context is now 1406 → 800…749 tok/s era-wise: 5407 ms at P7, 4117 after P8, 3841 after Q6.**

**The 4032-lane point, re-measured at 5 runs** because its spread had gone to 37.8 %: `15.179 / 15.159 / 14.514 / 15.050 / 12.989` → **median 15.050 ms (66.6 tok/s)**, spread 16.86 %. So the earlier "we do not yet know which" resolves to *~15 ms, ±10 %, with one occasionally-fast run at 12.99 ms* — the bimodality is still there, and it is not resolved by more repetitions. What resolves it is a stage-clock instrument inside the step (Q6.1), not a bigger N: the residue is ~3.5 ms and the only honest statement left is that we cannot yet say where it is.

### Q6.1: the residue is inside the kernels — stage clock, one window (2026-10-13, zero subagents, foreground, lock `q61-stage`)

`decode_run --time-stages` (`8b81970`, spec `e90f62f`) replays the same fed tokens at three submission
granularities and proves each pass did the same work (`STAGEPARITY … logits_bitidentical=1`,
`plan_match=1`, `launches_per_step=424 = plan_per_step`), so the three numbers below are one step measured
three ways, not three different runs. Medians of three runs, `idle_sync_us` from the instrument's own
calibration line (0.098 µs at 128, 0.108–0.164 at 4032):

| | burst (pipelined, production shape) | stage (323 waits/step) | kernel (per-launch waits) | exposed = stage − burst |
|---|---|---|---|---|
| 128 lanes | **12.399** (12.342 / 12.399 / 12.544) | 14.399 | 15.037 | **1.880** (1.855 / 1.880 / 2.088) |
| 4032 lanes | **15.378** (15.324 / 15.378 / 15.400) | 17.219 | 18.335 | **1.819** (1.786 / 1.819 / 1.819*) |

\*third 4032 run's stage gap 1.786, median 1.819. Against the pre-registered bands — story A predicted
`exposed ≤ 0.2 ms`, story B predicted `1.1–1.7 ms + N·s` — the measurement lands on **story B**, and
comfortably: `D + N·s ≈ 1.63 + 323×0.098 µs ≈ 1.66 ms`, and the observed gap is **1.82–1.88 ms**, i.e.
`≈ D + N·s` with nothing left over. Serialising the step costs exactly the dispatch bill and not a µs more,
which means the pipelined step was **already hiding every launch behind execution**. There is no
submission-side prize: no fusion-for-launch-count, no graph capture, no wider block. Also inside the ±5 %
sanity gate the spec set (§6's decode was 12.7–13.0 ms, this window's burst is 12.40; §6's 4032 median was
15.05, here 15.38).

**So the residue is in-situ kernel cost, and the isolated bursts understated it.**
`Σ exec_in_situ = stage − (D + N·s) = 14.399 − 1.66 ≈ 12.74 ms/token` against the ~9.4–9.6 ms the named
family bursts predicted: **~3.2 ms/step, ~25 %, was invisible to a burst.** Per family, in-situ (128 lanes,
median of 3) against the isolated-burst number Q6's table carried:

| family | in-situ ms | burst / derived | Δ | in-situ vs its own roofline |
|---|---|---|---|---|
| mlp | **4.264** | 2.594 | **+1.67** | 4.264 vs 2.230 → **52 %** (burst claimed 86 %) |
| proj | **4.225** | 3.54–3.60 | +0.63 | 4.225 vs 1.973 → 47 % |
| cast | 0.918 | never measured (0.43 dispatch) | +0.49 | 0.918 vs 0.007 traffic |
| host tail | 0.718 | 0.14 (CPU) | +0.58 | — |
| glue | 0.678 | 0.190 | +0.49 | — |
| gdn | 0.641 | ~0.12 (rec+norm) | +0.4 | 0.641 vs 0.228 cold |
| attn | 0.486 | 0.114 (derived) | +0.37 | — |
| copy | 0.208 | never (0.06 dispatch) | +0.15 | — |
| lm_head | **2.200** | 2.127 | **+0.07** | 2.200 vs 2.146 → at ceiling, and it *agrees* |

`lm_head` — one fat launch, streaming 485 MiB — moves by 3 % between burst and step, which is the control
case: the instrument is not inflating everything, the *small* kernels are. The candidate explanation is the
one this file already has a row for in miniature ("warm-burst understates in-step", GDN rec): a burst repeats
one kernel against the same buffers, so its ramp-up overlaps the next repeat and its DRAM pages stay hot,
while a real step interleaves 424 launches across ~1.5 GB of weights. That is now the standing reason a
burst number may not be quoted as a step cost, and it is the number to plan against: **the engine runs at
~52 % of achievable bandwidth on MLP and ~47 % on projections inside a real step**, not 86 %. `engine/results.md`
§Q7 has since decided that candidate in that direction, with the arithmetic in `tools/instep_family_budget.py`.

**Two things retired on the way.** The 4032-lane "bimodality" was mostly an artefact of the differential
harness: measured directly, step means are 15.324 / 15.378 / 15.400 (**0.5 % across runs**) even though
individual steps inside a run still jitter 11 %. The earlier 16.9 % spread came from differencing two runs
per point, which amplifies noise; the honest figure for 4k decode is **15.35 ms (65.2 tok/s)**. And the
one reporting bug claimed against the instrument did not survive its own logs: `syncs_per_step` was said to
print 323 for `stage` but **42 for `kernel`**, where the spec says ~425 waits. It prints **425** — all nine
kernel-granularity `TOTAL` rows of that window say so (`tmp/q61/stage128_*.log`, `tmp/q61/stage4032_*.log`,
`tmp/q61/on*.log`; re-checkable with `tools/instep_family_budget.py`, which prints the field per log), and
`stage`'s 323 is segments rather than launches, as designed. The field was right; **this sentence was the
bug**, and it is corrected here rather than left standing. The parity and token checks were clean, so the
numbers above stand.

Reproduce: `engine/results.md` §Q6.1 command block, one locked window `q61-stage`, three runs at 128 lanes
(`--steps 16`) and three at 4032 (`--steps 32`), prompts from `context_sweep.build_prompt`.

### Q7 M1: the locality story died in its own pre-registered test (2026-10-13, zero subagents, foreground, lock `q7-window`)

Q7 established the residue is not dispatch (Q6.1), not launch ramp (a 42.4 µs/launch ramp would bill a
12.74 ms step 18.0 ms), and not a wrong roofline (all three recompute from `engine/state.h` within 0.2 %),
and nominated **(b) fresh-window locality**: the MLP burst re-walks a 132 MB window 6×, the step walks
528 MB once and never returns. **M1** was the decisive test, written before the run: grow the burst's window
at fixed kernel geometry (`--tile 6 / 12 / 24`, i.e. 132 → 264 → 528 MB, `--iters 30`, `--layers all`,
`--mode all`, same kernel, same grid, same 11.01 MB per launch) and watch the per-token MLP figure. Story (b)
predicted ≥ 3.6 ms — the in-step row is 4.26 ms; the flat answer that refutes it was set at < 2.8 ms.

| burst window | per-token MLP (best of 30 iters) | vs probe | pre-registered verdict |
|---|---|---|---|
| 132 MB (`--tile 6`, the committed 2.594 row) | **2.630 ms** | 201 GB/s (85 %) | reproduces the ledger's 2.594 within 1.4 % ✓ |
| 264 MB (`--tile 12`) | **2.457 ms** | 215 GB/s (91 %) | |
| **528 MB (`--tile 24`, same size as one token's real MLP traffic)** | **2.408 ms** | 219 GB/s (93 %) | (b) needed ≥ 3.6 → **refuted** |

Growing the window 4× made the MLP **faster**, not slower — it walked *toward* the roofline, monotonically.
So warmth is not what the step is missing, and a cold 528 MB window is not the step's problem: at exactly the
traffic size of one real token, the same kernels reach **93 % of probe** in isolation and **52 %** inside the
step. Each run also re-gated correctness (`act ndiff 1 out ndiff 0`, `PASS: the §03.3 MLP reproduces the
reference operands`), so this is a green run per row, not a bare timing.

That leaves a fourth candidate that neither (a) nor (b) covers, and it is now the cheapest thing on the board
to test: **the step's traffic is not pure read.** A burst reads a weight buffer and writes a small activation;
a real step reads 1.5 GB of weights *while* writing activations, GDN state (18.66 MiB/request), KV, and the
~115 cast outputs — so every weight read can be competing with write-back of dirty lines the step just
dirtied. Read-only probes would never show that, and it fits the shape of the deficit: **+3.85 µs/MB on MLP
and +4.87 µs/MB on proj is roughly a doubling**, which is what mixed read/write traffic on LPDDR5x commonly
costs, not what a kernel defect looks like. If it holds, the conclusion is uncomfortable but useful: the
"52 % of a 237 GB/s read roofline" is a **mis-stated ceiling**, the real per-step ceiling is a mixed-traffic
number, and the way to go faster is to move fewer bytes, not to write better kernels. **Q8** is that probe.

Not run, and why: **M2** (re-take of the 128-lane stage rows) would reproduce numbers already in this file
from `q61-stage`, and nothing shipped to A/B against, so citing them is correct and re-making them is the
budget sink `AGENTS.md` warns about. **M3** (packed projections, proj 114 → 48) is held: it predicts
0.1–0.25 ms, needs a `gate.py::launch_plan` and §07 table change, and buying ~2 % of a 12.4 ms step with a
gate edit is a bad trade until the ceiling question is settled. **M4** (captured-step check) needs a stream
handle at the launch sites, which the engine does not have (default stream, so capture is impossible today);
Q6.1 already priced the same question from the other side, and the graph result agreed with the drain test.

### Q8: the memory system is not the excuse either (2026-10-13, zero subagents, foreground, lock `q8b-sweep`)

The full 26-row sweep, all twelve pre-registered expectations, and the two instrument bugs that ate the first
attempt live in `devices/strix-halo-mixed-traffic.md`; the four numbers that matter for the plan are these.
At the engine's own read:write ratio the box runs at **99.1 % of its read probe** (235.0 GB/s at 62:1), and
the ratio curve is flat — 96–99 % at every ratio ≥ 10:1, 87.2 % at 1:1. Reading freshly dirtied lines is
**free** (`E_same` pair 4.8228 ms vs 4.9137 ms written-then-read separately, −1.8 %), so write-back pressure
is not a mechanism on this box. And a single row that replays the whole step's traffic budget — 1530.6 MiB
read against 24.7 MiB written into a small re-dirtied arena, 96 serialized launches — achieves **218.9 GB/s**,
which prices one token's weight traffic at **7.06 ms**, against the engine's 12.4 ms.

So the three candidate excuses are now each dead on their own measurement: dispatch (Q6.1: exposed cost is
exactly the dispatch bill, nothing more), warmth and window size (Q7 M1: a 4× larger cold window makes MLP
*faster*), and traffic mix / write-back (this section). What remains is pointed at by one comparison:
`C_mixlaunch_62` is the MLP's own geometry — 48 × 11.01 MiB read once, 96 serialized launches, 62:1 — and it
runs at **86.6 % of the read probe**, while the same kernels in a real step run at **52 %**. The gap is not
the memory system's floor, it is the engine failing to reach a floor the box demonstrably offers. Splitting
one kernel into 96 serialized launches costs 9 % (expectation 6), an order of magnitude too small to be the
answer. One row tripped the trap flag and is excluded (`D_20_arena2m`, 102.8 %, `suspect=1`), and the three
cold-page rows are void by their own rule (`unverifiable_cold_pages`, exit 3).
