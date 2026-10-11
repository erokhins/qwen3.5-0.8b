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
| `glue` residual+RMSNorm, fused, 49 launches/token | ~0.5 MB | **0.1336–0.1360 ms** † | 2.73–2.78 µs/launch | launch-bound: roofline is 0.0021 ms |
| same, unfused (48 adds + 49 norms = 97 launches) | ~0.5 MB | 0.2131–0.2155 ms † | 2.20–2.22 µs/launch | fusion buys 0.0795 ms/token |
| `attn_decode` §05 core, 6 layers, one launch each | ~0.4 MB at 25 lanes | ~0.011 ms (launch floor arithmetic, not timed) | — | bit-exact to 277 lanes |
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
| GDN state traffic (18 layers, warm) | 0.054 | 0.228 ms cold | **0.0646 ms** | **kernel exists** |
| §04.5 gated norm (18 layers) | ~0.0004 | 0.0016 ms | **0.053 ms** | **kernel exists** — launch-bound, 18 × 2.94 µs |
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

† **The projection and glue numbers are provisional.** Both were taken while something else worked on
the GPU without the lock — `glue`'s two windows read 94 %→75 % and 98 %→35 % `util`, and an earlier
one 17 %→60 % — so they are floors, not measurements, and the quiescent pass in §5 item 0 re-takes
them. Rows in `glue`'s own sweep with 9–331 % spread are marked not-quotable in its `results.md`
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
