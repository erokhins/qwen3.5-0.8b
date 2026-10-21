# `prototype/glue` — the residual add + zero-centred RMSNorm, fused into one launch

Status: **gate PASS — 784 comparisons, 0 failures** (`prototype/glue/run.py`, built by
`prototype/glue/build.sh`). Binary: `prototype/glue/glue` (ignored by exact path in
`.gitignore`). Spec: §03.1 (the two zero-centred norms) + §03.2 (one add per sub-block).

```bash
./prototype/glue/build.sh
.venv/bin/python prototype/glue/run.py --negative --loader-verify   # the gate, ~2 min
.venv/bin/python prototype/glue/floor_probe.py                      # the band sizing
.venv/bin/python prototype/glue/run.py --time --iters 300           # timings (takes the lock)
```

## 0. What this op is, and how it must be read

Per layer the decoder runs 2 RMSNorms and 2 residual adds (§03.2). This file does
`s = s + delta; out = rms(s)·(1+w)` in **one** launch — the pair that must be fused is
(add the block's output, then the norm that feeds the MLP) and (add the MLP's output, then the
norm that feeds the next block). Plus one norm-only launch per step in front of layer 0, which
has no add before it.

**This is a launch-count item, not a bandwidth item.** One token's glue moves
**0.498 MB** (fused) — that is **0.0021 ms** of roofline at the 237 GB/s read probe, i.e. ~0.03 %
of the ~7.8 ms token budget. What it actually costs is **0.137 ms/token**, ~65× its roofline,
because all of it is dispatch and single-block latency. Every GB/s figure for this op is therefore
meaningless (~2–4 GB/s, and the working set is cache-resident anyway); the numbers that matter are
**launches/token** and **µs/launch against the measured 1.85 µs floor**
(`engine/performance-status.md` §4). `run.py` prints GB/s only to label it as not the finding.

Note the queue's estimate (~0.005 GB/token for W4) is ~10× this: measured traffic is 0.498 MB/token
fused, 0.596 MB/token unfused. The launch argument gets stronger, not weaker.

## 1. Launch accounting (the point of the op)

| form | launches/token | at the 1.85 µs floor | measured, window A / window B (see §4) |
|---|---|---|---|
| unfused: 48 adds + 49 norms | **97** | 0.1795 ms | 0.2155 / 0.2131 ms (2.222 / 2.197 µs/launch) |
| **fused: 24×2 + 1 norm-only** | **49** | **0.0907 ms** | **0.1360 / 0.1336 ms (2.776 / 2.727 µs/launch)** |
| folded into the GEMV that produces `delta` (the `mlp_gateup` pattern) | 0 | 0 | — |

Fusion here halves the launches and buys **0.0795 ms/token** in window A and **0.0795 ms/token** in
window B (two independent gated windows agreeing to 1.8 % on the fused form and 1.1 % on the
unfused one), ≈ **1.0 % of the 7.81 ms measured token**. Folding the pair into the producer's epilogue would remove the whole
remaining 0.136 ms — that is the W5 lever, not this file's.

Each launch is one block over a 1024-element row. The kernel's own latency is ~0.9–1.0 µs above
the 1.85 µs dispatch floor (2.776 µs/launch at 1024 threads); block width matters and is swept in
§4. The residual dependency chain costs nothing extra: chained (each launch reads the previous
launch's output, as a real token does) and unchained agree to 0.7 % — 0.1360 vs 0.1369 ms/token.

## 2. Gate — what is compared against what

No new dumps were needed and none were made. The pins are the ones already in `data/refs/dec8/`:
`decode{N}.L{ii}.s_in` / `.s_out` (residual either side of a block, **all 24 layers × all 8
steps**), the two addends (`...out_proj.out` for GDN layers, `...o_proj.out` for the 6 attention
layers, `...mlp.out`), `...mlp.in` (which *is* glue call 1's norm output), and `final_hidden`.
(The names given in the work order — `transformer.h.N.glue.s_in` — do not exist; `glue_pins.py`
uses the real ones and `final_pair()` **finds** which step the unprefixed `final_hidden` /
`pre_final_hidden` belong to by bit-comparison instead of assuming one, because §10.5's
last-writer-wins trap makes that pairing exactly the thing that has already cost this project time.
It is `decode7`, asserted unique.)

Per layer, two launches, exactly the two an engine issues:

| observable | fed with | compared against | criterion |
|---|---|---|---|
| `g1_s` | `s_in` + the block's output | the reference add | **bit-exact** (a BF16 add has no reordering freedom) |
| `g1_n` | same | **the pin `mlp.in`** | band + aggregates |
| `g2_s` | that updated `s`, + `mlp.out` | **the pin `s_out`** | **bit-exact** |
| `g2_n` | same | the spec mirror, and **the pin `final_hidden`** on the step those bare pins pair with | band + aggregates |
| `n0_n` | `L00.s_in`, no add | the spec mirror | band + aggregates |
| `xstage`, `n0_xstage` | feed `g2_n` / `n0_n` into the next layer's first projection | **that projection's pinned output** | band + aggregates |

The `xstage` rows are what give call 2's norm dump-level bite: `g2_n` at layer L has no pin of its
own (except at L23), but it is the tensor layer L+1's projections consume, so the pinned
projection output is a real dump to gate it against.

**Stage R runs first** (1673 checks, 0 failures): torch's own BF16 arithmetic must reproduce
`mlp.in`, `s_out` and `L.s_out == (L+1).s_in` **bit-exactly** from the pinned inputs, and the
plausible misreadings must not — plain `w` instead of `1+w`, normalising the addend, normalising
the pre-add residual, this layer's *input* norm used for call 1, this layer's *post-attention* norm
used for call 2, and any sigmoid factor on the norm output (§03.2: no gate, no bias, no per-layer
scale on this path). All refused.

**Result: 784 comparisons, 0 failures.** 382 of the 384 norm outputs are **bit-exact against the
pinned dumps**; the two that are not (`decode3.L23 g1_n`, `decode7.L04 g1_n`) each differ in one
element by 1 ULP with `n_bad` 0 — HF's own reduction order vs ours, the arithmetic-sensitivity
floor, not an error. The two residual adds are bit-exact in all 384 cases (0 differing elements
summed over 393,216). `--split` re-runs the same plan as separate add+norm kernels and produces the
same bits (17 outputs × 1024 elements, 0 differing), so the launch saving was not bought with an
arithmetic change.

## 3. Floor probe, and the BF16-partial question

`floor_probe.py`, sample size in the header line: **24 layers × 8 steps × 2 glue calls = 384 norms
× 1024 elements = 393,216 compared values**, reference = `speccheck.specimpl.rmsnorm_zc`.

| variant | ndiff | max ULP | >1 ULP |
|---|---|---|---|
| numpy F32 sum | 4 / 393,216 (0.0010 %) | 1 | 0 |
| `math.fsum` (exactly rounded) | 3 / 393,216 (0.0008 %) | 1 | 0 |
| the kernel's own order (4 strided/lane, shuffle ladder, 8-partial tree) | 3 / 393,216 (0.0008 %) | 1 | 0 |
| 128-block fold | 2 / 393,216 (0.0005 %) | 1 | 0 |
| `1/sqrt` instead of `rsqrt` | 3 / 393,216 (0.0008 %) | 1 | 0 |
| **`bf16sub`** — one BF16 subpartial per thread, F32 ladder | **8,078 / 393,216 (2.05 %)** | **1** | 0 |
| **`bf16acc`** — the whole reduction in a BF16 slot | **392,947 / 393,216 (99.93 %)** | **45** | 392,947 |

**A norm's floor is not a dot's floor.** Reordering a mean of 1024 non-negative squares moves no
element more than 1 ULP in this sample — every element of the output is scaled by the same `r`, so
a ~1e-7 relative change in `r` can only flip outputs sitting on a BF16 rounding boundary. The
consequence is that **the cancellation band is carrying nothing here**: `CANCEL_REL = 3e-5` (the
convention, kept) excuses 0 of 393,216 values and the tightest band these numbers pass at is 0 —
infinite margin. The criterion doing the work is `ndiff ≤ max(2 % of n, 1)`, not the ULP bound.
Do not copy the dot gates' band into a norm and assume it is what makes them pass.

**The BF16 accumulator is the one real accuracy risk in this op, and it is invisible to a ULP
criterion.** A BF16 subpartial (`bf16sub`) moves 2.05 % of the output elements and *not one of
them by more than 1 ULP* — meanrel 3.4e-4. It is caught by `ndiff` and by nothing else. A BF16
running accumulator (`bf16acc`, the worst form: a BF16 slot at every add) is a disaster: 99.93 %
of elements differ, up to 45 ULP, meanrel ~1. So "the squared-sum must be F32" is not stylistic.
Caveat worth keeping: on the case set the kernel-level `bf16sub` negative lands at 247/6144 = 4.0 %
against a 2 % limit — caught, but with only ~2× margin. The measured legal floor (3 flips in
393,216) would support tightening `NDIFF_PCT_MAX` from 2 % to ~0.5 %, which would give `bf16sub` an
8× margin; not done here, because that needs the negative suite re-run.

## 4. Timings — **provisional**

Under `tools/gpu_lock.sh`. Window A (tag `glue threads sweep (worker W4)`) had `util` **94 %** at
acquire and **75 %** at release; window B below had **98 %** → **35 %**. This box's GPU is never
idle (`flash_serve` polls it), so a quiet window may not exist; **every number here is provisional**
and each row's spread is printed with it. Burst = the 49-launch token plan × 300 iterations, one
sync, wall-clock/N, best of 5 trials; `hipEventElapsedTime` is not used (it reads exactly 0 below
~1 ms here). Bytes/token 0.498 MB (fused), 0.596 MB (unfused); roofline 0.0021 / 0.0025 ms.

**Window A** (10 launches in the window):

| form | threads | ms/token | spread | µs/launch | quotable? |
|---|---|---|---|---|---|
| fused, chained | 1024 | **0.1360** | 3.2 % | **2.776** | yes |
| fused, unchained | 1024 | 0.1369 | 0.7 % | 2.793 | yes (best repeat) || fused, chained | 512 | 0.1497 | 34 % | 3.055 | no — contention |
| fused, chained | 256 | 0.1836 | 3.2 % | 3.747 | yes |
| fused, chained | 128 | 0.2595 | 3.3 % | 5.295 | yes |
| unfused pair, chained | 1024 | **0.2155** | 3.0 % | 2.222 | yes |
| unfused pair, chained | 512 | 0.2269 | 33 % | 2.339 | no — contention |
| unfused pair, chained | 256 | 0.2470 | 6.6 % | 2.547 | marginal |
| unfused pair, chained | 128 | 0.2950 | 331 % | 3.041 | **no** — contention |
| fused, unchained | 256 | 0.1866 | 278 % | 3.808 | **no** — contention |
| fused, unchained | 512 | 0.1535 | 9.3 % | 3.133 | marginal |

**Window B** (tag `glue results recheck (worker W4 revive)`, util **98 %** at acquire, **35 %** at
release) — the cleanest numbers in this file, every row under a 6 % spread:

| form | threads | ms/token | spread | µs/launch |
|---|---|---|---|---|
| fused, chained | 1024 | **0.1336** | 5.7 % | 2.727 |
| fused, unchained | 1024 | 0.1344 | 3.4 % | 2.743 |
| unfused pair, chained | 1024 | **0.2131** | 3.5 % | 2.197 |

Window A's headline rows reproduce in window B to 1.8 % (fused) and 1.1 % (unfused), so the *ratio*
and the fusion saving are solid even though no window had a low `util` at both ends.

Rows at 9–331 % spread are other work landing in the window and are **not quotable**; a third
window (tag `glue timing (run.py)`, util 17 %→60 %) reproduced the 512-thread fused token number
to within 3 % (0.1494 ms/token, 5.2 % spread).

Block width is a real config knob: 1024 threads > 512 > 256 > 128 (2.78 / 3.06 / 3.75 / 5.30
µs/launch) — with one block per 1024-element row, fewer elements per thread and a shallower
per-thread loop wins, and the reduction latency dominates the arithmetic. `run.py`'s default is
still 256 (the config the correctness gate was run at, including all negatives); the timing
section sweeps all four. Re-run the gate at 1024 before W5 hard-codes a width.

## 5. Negatives (each scoped so the must-not-catch claim is testable)

Case set: layers 0, 3, 23 × steps decode0, decode7. Severity is summed over the case set.

| injection | scope | must-catch | caught | severity | must-not | result |
|---|---|---|---|---|---|---|
| `plainw` (affine is `w`) | call 1 | `g1_n` | `g1_n` | ndiff 6140/6144, max 33,377 ULP, meanrel 9.6e-1 | `g1_s,g2_s,g2_n,n0_n` | all clean |
| `plainw` | call 2 | `g2_n` | `g2_n` | ndiff 6137/6144, max 32,576 ULP, meanrel 7.1e-1 | `g1_s,g1_n,g2_s,n0_n` | all clean |
| `normdelta` (norm the addend) | call 1 | `g1_n` | `g1_n` | ndiff 5589/6144, max 198 ULP, meanrel 1.8 | `g1_s,g2_s,g2_n` | all clean |
| `noadd` (skip the add) | call 1 | `g1_s,g1_n` | both | `g1_s` 6129/6144; `g1_n` max 32,912 ULP | `g2_s,g2_n,n0_n` | all clean |
| `noadd` | call 2 | `g2_s,g2_n` | both | `g2_s` 6125/6144; `g2_n` max 32,974 ULP | `g1_s,g1_n,n0_n` | all clean |
| `mean` (mean, not mean-of-squares) | call 1 | `g1_n` | `g1_n` | ndiff 6144/6144, max 49,355 ULP, meanrel 1.4 | `g1_s,g2_s` | all clean |
| `bf16acc` (BF16 accumulator) | call 1 | `g1_n` | `g1_n` | ndiff 720/6144 (11.7 %), max 1 ULP, meanrel 8.8e-4 | `g1_s,g2_s,g2_n` | all clean |
| `bf16sub` (one BF16 subpartial) | call 1 | `g1_n` | `g1_n` | ndiff 247/6144 (4.0 %), max 1 ULP, meanrel 3.4e-4 | `g1_s,g2_s,g2_n` | all clean |

8 injections, 0 escaped, 0 leaked into a must-not-catch observable. The scoped injection is what
makes the must-not-catch column mean anything: with the bug in call 1, call 2's outputs are
genuinely untouched because each call is fed its own reference input (stage R proves that input is
the reference's own, bit for bit).

## 6. `engine/loader.h` as the consumer of the norm weights — feedback

**It worked, with no workaround.** The kernel binary links `engine/loader.cpp` and resolves every
one of the 49 norm vectors through `engine::WeightStore::at()` by its normalised §02.5 name
(`lm.layers.N.post_attention_layernorm.weight`, `lm.layers.N.input_layernorm.weight`,
`lm.norm.weight`); there is no hand-rolled H2D for a weight in this directory. Each fetched weight
is asserted `dtype BF16 && shape [1024] && nbytes 2048` before use — 49 distinct names, 98.0 KiB —
and one gate run used `--loader-verify`, i.e. the loader's own copy-back + SHA-256 of all 335
resident blobs against the file, so the device bytes the gate trusted are the file's bytes. The
loader's warm pass is also what prototype/README.md rule 3 (warm before timing) is satisfied by;
`warm_checksum != 0` is asserted rather than assumed.

Friction, in the order it showed up:

1. **No partial load.** The loader loads all 335 tensors (1.440 GiB, 0.34–1.03 s wall here) to give
   one consumer 98 KiB. Fine for the engine, which wants everything resident; annoying for a
   prototype — the gate spawns ~11 processes and pays the full load each time (~11 s of the gate's
   runtime). A `Options::only = {prefix list}` (or a `load_matching()`) would cut this gate's wall
   time in half and would not weaken anything, because the same `at()`/shape/dtype contract applies.
2. `Options::measure_steady_h2d` defaults to `true`; a consumer that just wants weights has to
   remember to switch it off. Defaulting it the other way (or moving it to a separate probe call)
   would be friendlier.
3. `WeightStore` has a private default constructor, which is right, but means the only spelling is
   `auto store = WeightStore::load(opts, stats)` — worth knowing before you declare one.
4. Everything else was a pleasure: `at()` throwing on an unknown name caught a name-mapping bug of
   *mine* immediately; `Weight::checkpoint_name` made the engine-name → checkpoint-name direction
   trivial for the torch-side mirrors (`glue_pins.ckpt_w`); `LoadStats` is a good instrument
   (the loader's warm pass read 1.440 GiB at 176–179 GB/s, matching `engine/results.md`).
5. One number to correct over there: `LoadStats::warm_bytes/warm_ms` is bytes-per-millisecond, so a
   consumer computing GB/s must divide by 1e6·ms, not 1e9·(ms/1000) — my first print claimed
   125,107 GB/s. Fixed in `glue.hip.cpp`; mentioned because the same arithmetic is easy to repeat.

## 7. Not done / open

* **Provisional timings.** No window had a low `util` at both ends (94 %→75 % and 17 %→60 %); the
  ledger should not take a glue row until a window repeats 0.136 ms/token with a small spread.
* `NDIFF_PCT_MAX` stays at the inherited 2 %; §3 explains why ~0.5 % is better-measured here and
  what re-run it needs.
* `run.py`'s default block width is 256, the sweep's best is 1024 (§4).
* No `engine/` change: this op consumes the loader, it does not modify it.

## 8. P4: `blockIdx.x` is the row, so a wave batched the pair (§08.2)

Prefill (§08.2) does the same op over a `[T, H]` tile, and the wave wants one launch per call
instead of `T` of them. The kernel gained a row index and nothing else:

```cpp
const size_t row = (size_t)blockIdx.x * H;
s += row;  if (delta) delta += row;  if (s_out) s_out += row;  n_out += row;
```

The reason this is allowed at all is that §03.1's reduction is **row-local** — one block, one
1024-element row, no partial ever crosses a row boundary — so `<<<T, THREADS>>>` is obliged to
produce the same bits as `T` launches of `<<<1, THREADS>>>`, and at `gridDim.x == 1` the pointers
above are the pointers the kernel formed before the row index existed. That is the decode
path's protection by construction, not by argument; `--rows 1` is still the default and the
shipped rows=1 gate is unchanged. `residual_add`/`norm_only` took the same index so the
fused-vs-unfused equivalence (§2) still holds at any row count. The norm weight `w` is
deliberately not row-indexed: one norm vector serves the tile.

New gate stage, `run.py --batched p64` (also `p300`), three checks per layer, all of them on
device:

1. **stage R for tiles** (`pre_rebuild`): torch's own adds must land on the pinned
   `L{ll}.s_out`, `rmsnorm_zc(s_mid, w_g1)` must BE the pinned `L{ll}.mlp.in`, layer L's
   `s_out` must be layer L+1's `s_in` across the whole chunk, and `pre_final_hidden` /
   `final_hidden` must reproduce from layer 23. A prefill fixture has one phase, so §10.5's
   last-writer-wins trap cannot bite here — said out loud because the decode fixture's `final_*`
   pins are exactly the trap. Misreadings (plain `w`, norming the addend, norming the pre-add
   residual, the wrong layer's norm) must still fail.
2. **the tile launch vs the pins**, judged **per row** rather than over the flattened tile: the
   cancellation band is relative to the tensor max, and a flattened 47 104-element "row" would
   let a loud token hide a quiet one. Adds are required bit-exact; norms get the same
   `floor_probe.py` band as §2/§3.
3. **bit-identity vs one launch per row** — the check the engine's margins actually depend on.
   Same plan, same inputs, built by one function (`_tile_lines`) so the two shapes cannot drift
   apart, then a hard `!= 0` on the element count.

Measured (`.venv/bin/python prototype/glue/run.py --fixture dec8 --layers 0,3,23 --steps decode0
--batched p64`, wall 3.3 s, `gpu_lock util` 93 % — correctness only, no timing taken, and the
tile stage takes no `ms` figure anywhere):

| observable (46-row tile, layers 0/3/23) | vs pins | vs one launch per row |
|---|---|---|
| `g1_s`, `g2_s` (the two residual adds) | 0 / 47 104 differ | 0 |
| `g1_n`, `g2_n`, `n0_n` (the three norms) | worst row ≤ 2 / 1024 differ, max 1 ULP, cos 1.00000000, meanrel ≤ 4.4e-07 | 0 |
| all six outputs together | — | **612 352 elements compared, 0 differ** |

So the multi-row launch is bit-identical to the per-token one, and prefill glue does not move the
legal route that `engine/route_allowance.py` priced. Falsification, because a stage that cannot
fail verifies nothing: `--neg plainw --scope 1` on the tile plan moved `g1_n` and left
`g1_s`/`g2_s`/`g2_n` alone, exactly as it does at rows=1.

`--rows` also multiplies the burst's byte accounting (the `--time` path is unchanged in shape and
still main-agent-only, engine/performance-status.md §6).
