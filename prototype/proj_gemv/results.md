# `proj_gemv` — one BF16 GEMV for all seven projection shapes (§04.1, §04.5, §05.1, §05.6), k = 1

**Status: the correctness gate passes.** One templated kernel (`proj_gemv<IN, PACK_X>`)
matches every pinned projection output of every layer, at every pinned decode step — 912
comparisons, 0 failures. **The cold timing is provisional** (utilisation was 9 % at lock
acquire and 60 % at release; someone else was working on the box without the lock).

```
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps all --negative   # THE AUTHORITATIVE RUN (~40 s, no GPU)
.venv/bin/python prototype/proj_gemv/floor_probe.py --layers all --steps all      # sizes the tolerances (~15 s)
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps decode0 --time --sweep   # cold timing, takes gpu_lock
```

`run.py` also runs `tools/check_projection_pin_semantics.py` first: that checker independently
proves each pin means what its name says, so the gate cannot pass by comparing against a
mislabelled fixture.

## What runs

The seven pinned projections, in the two packings §04.1 / §05.1 ask for (`--time` prints this):

```
packed in_proj_qkv+z+b+a [8224,1024] x18   in_proj_qkv [6144,1024] x18   in_proj_z [2048,1024] x18
in_proj_b [16,1024] x18   in_proj_a [16,1024] x18   out_proj [1024,2048] x18
packed q+k+v [5120,1024] x6   q_proj [4096,1024] x6   k_proj [512,1024] x6   v_proj [512,1024] x6
o_proj [1024,2048] x6
```

| | | | | spec § | | | |
|---|---|---|---|---|---|---|---|
| GDN | `in_proj_qkv` [6144,1024] §4.1 (q 2048 \| k 512 \| v 512 row order — NOT channel-first) | `in_proj_z` [2048,1024] | `in_proj_b` [16,1024] β | `in_proj_a` [16,1024] −A_log |
| attn | `q_proj` [4096,1024] §5.1 per-head `[8][q 256 \| gate 256]` | `k_proj`/`v_proj` [512,1024] | `o_proj` [1024,2048] | `out_proj` [1024,2048] |

One work item = one output row: `gridDim = M`, 512 threads, each thread owns a strided set of
K positions (`K / (512 × PACK_X)`, PACK_X = 2 at K = 1024, 1 at K = 2048), block reduction with
warp shuffles (§07.3 item 1, no `__syncthreads` chain), one BF16 store per row. The templated
form is what makes it one kernel family rather than seven copies:

- `IN` ∈ {1024, 2048} — K is a template parameter so the unroll and the register accumulator are
  compile-time; a `switch` on `k` picks the instantiation and an unsupported K **fails loudly**.
- `PACK_X` — what dtype `x` is staged as in shared memory: BF16 as the norm left it (2 KiB at
  K = 1024) or unpacked to F32 (4 KiB). The products are identical — every BF16 value is exact in
  F32 — so this is purely a shared-memory/occupancy question, and it is measured rather than
  guessed. Half the blocks' shared footprint, **and it still never wins**; see "Performance".
- `NEG_*` — the located negatives, compiled in because a negative must be exactly reproducible.

A `--plan` file turns N logical projections into N jobs in **one launch** (row section + row count
+ which weight buffer + which input section). That is the engine-shaped part: 48 launches per
token instead of 114, with the same per-row kernel underneath. `b`/`a` (16 rows each) exist in the
plan only to prove the fold costs nothing; §"Performance" shows why folding them is right.

## Where the operands come from — proven before any kernel runs

The pins are **outputs**. Every projection but `out_proj`/`o_proj` consumes `input_layernorm(x)`,
which is not pinned (§00.3 rule 1 — every intermediate is pinned, but norm outputs are).
`operands.py` rebuilds each operand and **proves** it reproduces the pin bit-for-bit with a torch
BF16 matmul before the GPU is allowed anywhere near it:

| operand | rebuilt from | proof |
|---|---|---|
| input projections (all 7 K=1024 shapes) | `rmsnorm_zc(L{ii}.s_in, input_layernorm.weight)` — the §03.5 zero-centred form | `torch.matmul` on the rebuilt x equals the pin **exactly** |
| `out_proj` (GDN) | `L{ii}.gdn.gated_out` | same, exact |
| `o_proj` (attention) | `L{ii}.attn.attn_out_pre_gate × sigmoid(gate)`, gate = the **per-head second half** of `q_proj.out`, `[8][q 256 \| gate 256]` | same, exact |

Three things this step earned rather than assumed:

1. **`s_in` is pre-norm.** Feeding the raw `s_in` pin (un-normalised) is caught, feeding the
   post-attention norm output is caught, and feeding `rmsnorm` without the `(1+w)` affine is
   caught. All three are separate negative rows in the table below, so "verify the norm usage" is
   now enforced by the gate instead of remembered in prose.
2. **`o_proj`'s input is the gated attention output, not the raw attention output.** §05.6's
   `o_proj(attn(gate))` is vague; the pin that survives is `attn_out_pre_gate × sigmoid(gate)`
   (§05.2 gating). `tools/check_projection_pin_semantics.py` check E now says the same thing,
   and `attn.out` (the tensor `attn_io_hook` wanted) is not pinned at all.
3. **The q split is interleaved.** `[q half | gate half]` contiguously is wrong — it fails with
   max ULP 65535 / cos −0.14. `--neg qhalf` and the contiguous-half operand negative keep it that way.

## The gate

Per comparison, `got` = kernel output, `ref` = the pin. `prototype/bf16_ulp.py` (with its
`selftest()`, run at startup) does the BF16 ULP comparison; the metrics are §10.1's.

| check | threshold | where it comes from |
|---|---|---|
| large elements (`\|ref\| > 3e-5 × max\|ref\|`) | **0** may exceed 1 BF16 ULP | §10.1 F1 |
| `ndiff` | ≤ max(2 % of elements, 1 element) | §10.1 F3, re-measured for K=1024/2048 here |
| `cos` / `meanrel` | ≥ 0.9999 / ≤ 1e-2 | §10.1 F4, sized by `floor_probe.py`'s perturbation survey |
| speccheck cross-check | same bands | torch CPU `Weights.matmul`, pinned to HF by W-C |

`CANCEL_REL = 3e-5` is the band `mlp_gemv` re-set from measurement; this op re-measured it for
K = 1024/2048 and the worst offender anywhere in 912 comparisons sits at **4.4e-6 of the tensor
max — 6.8× of slack**. The band is printed with its slack at the end of every run so nobody has
to trust it.

24 layers × 8 decode steps of `dec8` (full pins: 18 GDN + 6 attention layers; `prefix24dec8`
agrees shape-for-shape and is accepted for `out_proj`/`o_proj` if `dec8` is absent):

| output | cases | out of | max ndiff | cases with any diff | raw max ULP | over-tolerance | min cos | max meanrel |
|---|---|---|---|---|---|---|---|---|
| `in_proj_qkv` | 144 | 6144 | 12 | 87 | 193 | **0** | 0.99999996 | 5.1e-06 |
| `in_proj_z` | 144 | 2048 | 2 | 33 | 40 | **0** | 0.99999999 | 3.8e-06 |
| `in_proj_b` | 144 | 16 | 0 | 0 | 0 | **0** | 1.00000000 | 0 |
| `in_proj_a` | 144 | 16 | 0 | 0 | 0 | **0** | 1.00000000 | 0 |
| `out_proj` | 144 | 1024 | 3 | 36 | 3 | **0** | 0.99999993 | 1.8e-05 |
| `q_proj` | 48 | 4096 | 4 | 20 | 14 | **0** | 1.00000000 | 1.0e-06 |
| `k_proj` | 48 | 512 | 1 | 3 | 1 | **0** | 0.99999998 | 1.3e-05 |
| `v_proj` | 48 | 512 | 1 | 4 | 2 | **0** | 1.00000000 | 1.7e-07 |
| `o_proj` | 48 | 1024 | 2 | 16 | 2 | **0** | 0.99999994 | 1.5e-05 |

The two 16-row outputs (`in_proj_b`, `in_proj_a`) are **bit-exact** everywhere: 16 rows of K=1024
happen to reduce to exactly the same value here. A reviewer should not read the `ndiff ≥ 1`
allowance as proof the reduction order is right — on those shapes the aggregate checks are
vacuous (they are 0/16 diffs and cos 1.0), which is exactly the trap `mlp_gemv`'s F1 review
found. They stay in the table because 0/16 with a 1-element allowance is also the floor derived
from `floor_probe.py` (0 of 9216 reorderings move), not a guess.

`§10.5` guard: the only reference-valued intermediate this op could confuse with the output is the
pin itself, and the operand rebuild (above) fails loudly if `got` is asked to equal `ref`.

## Why the tolerances are what they are — `floor_probe.py` (K = 1024, K = 2048)

```
attn K=1024  worst 17 ULP over 983,040 reorderings  |  >1 ULP 13 (0.00132 %)  |  worst element at 1.76e-07 of max
attn K=2048  worst  2 ULP over 196,608              |  >1 ULP  1 (0.00051 %)  |  1.36e-05
gdn  K=1024  worst 193 ULP over 4,737,024          |  >1 ULP 105 (0.00222 %) |  7.63e-40
gdn  K=2048  worst  3 ULP over 589,824             |  >1 ULP  4 (0.00068 %)  |  6.28e-07
```

Four legal reorderings of the reference's own F32 sum (blocked-128, einsum, F64-exact, and a
faithful emulation of **this kernel's** order: 8-element strided serial sums, 16-lane butterfly,
F32 accumulate then BF16 round). K=1024 is worse than K=3584's 106 ULP because the K=1024 tensors
have a far larger dynamic range: `in_proj_qkv` cancels to **7.6e-40** of its own max somewhere in
18 layers × 8 steps × 6144 rows, and at that element BF16 spacing is enormous — 193 ULP is a
handful of F32 ulps. `kernel-order` is inside the same envelope as the other three, i.e. this
kernel's order is not special.

The perturbation half is what sizes the aggregates: **a single 1-ULP flip in the input** — exactly
what a legal BF16 norm upstream produces — moves an output by up to **29,331 ULP** at an element
1.2e-5 of the tensor max. So a chained projection output cannot be gated per element; the
aggregates are the gate (same conclusion as §10.1 F4), and they leave ~27× of margin on this
injection.

## The negatives — each must be caught where declared and nowhere else

Eleven rejections, **0 false passes, 0 caught-out-of-scope** (`--negative`, every layer/step):

| injection | must be caught in | must NOT be caught in |
|---|---|---|
| `--neg tail` — drop the last K word | all 7 GDN-stage outputs of the layer | — |
| `--neg half` — accumulate half of K | all | — |
| `--neg trunc` — truncate instead of RNE | all | — |
| `--neg rowswap` — write each dot to the neighbouring row | all | — |
| `--neg transpose` — read the weight transposed | all | — |
| `--neg qhalf` — `q_proj` rows in contiguous-half layout | **`q_proj` only** | `k_proj`, `v_proj` |
| `--neg kvswap` — swap the k/v sections of a packed `in_proj_qkv` | **`in_proj_qkv.k`, `.v`** | `.q`, `in_proj_z`, `_b`, `_a` |
| wrong operand: raw `s_in` | the 6 input projections | `out_proj` (its own operand is untouched) |
| wrong operand: post-attention norm output | the 6 input projections | `out_proj` |
| wrong operand: norm without the `(1+w)` affine | the 6 input projections | `out_proj` |
| wrong operand: contiguous-half gate on `o_proj` | **`o_proj`** | `q_proj`, `k_proj`, `v_proj` |

## Performance — **provisional**

Taken under `tools/gpu_lock.sh` (tag `proj_gemv timing (run.py)`), `gpu_lock.sh util` = **9 % at
acquire, 60 % at release**: an unlocked process was using the GPU during the run, so treat every
number here as a floor, not a published measurement. Wall-clock/N in a burst; `hipEventElapsedTime`
is printed beside every line as a cross-check and **reads 0 or negative here** (trap #7), which is
why the min() of the two was removed — a broken timer had been reporting 25433 GB/s for one shape.

Cold per shape (weights from every real layer that has the shape, `decode0`, `tile` = replicas deep
to defeat the 2 MiB L2 — bytes are always the real single-launch traffic):

| launch | shape | ×layers | tile | µs/launch | GB/s | % of 237 GB/s probe |
|---|---|---|---|---|---|---|
| `PACK_inproj` (qkv+z+b+a) | [8224,1024] | 18 | 1 | 92.68 | **181.9** | 77 % |
| `in_proj_qkv` | [6144,1024] | 18 | 1 | 75.05 | 167.9 | 71 % |
| `PACK_qkv` (q+k+v) | [5120,1024] | 6 | 2 | 66.92 | 156.9 | 66 % |
| `q_proj` | [4096,1024] | 6 | 2 | 58.31 | 144.0 | 61 % |
| `o_proj` | [1024,2048] | 6 | 4 | 36.11 | 116.3 | 49 % |
| `in_proj_z` | [2048,1024] | 18 | 2 | 37.73 | 111.3 | 47 % |
| `out_proj` | [1024,2048] | 18 | 2 | 38.54 | 109.0 | 46 % |
| `k_proj` | [512,1024] | 6 | 16 | 11.23 | 93.7 | 40 % |
| `v_proj` | [512,1024] | 6 | 16 | 11.40 | 92.3 | 39 % |
| `in_proj_b` | [16,1024] | 18 | 1 | 3.18 | 11.0 | 5 % |
| `in_proj_a` | [16,1024] | 18 | 1 | 3.23 | 10.8 | 5 % |

**The whole token's projections: 2.9895 ms, 156.3 GB/s, 66 % of the probe** (48 launches, 445.7
MiB of weights + the activations; roofline at 237 GB/s would be 1.973 ms). Against §07.4's budget
(10 ms/token total, of which the projections are the largest single block after `lm_head`), that
is 30 % of the token budget spent on projections where the roofline says 20 %.

`rows/block` sweep on the token plan (this is the "what was tried" the ledger asks for):

| rows/block | 2 | 4 | 8 (default) | 16 | 32 |
|---|---|---|---|---|---|
| `x` staged as F32 in shared (default) | 3.393 ms (58 %) | 3.191 (62 %) | 2.991 (66 %) | 2.946 (67 %) | **2.785 (71 %)** |
| `PACK_X` — `x` kept BF16 in shared | 3.677 (54 %) | 3.489 (57 %) | 3.269 (60 %) | 3.347 (59 %) | 3.173 (62 %) |

Reading:

- **Keeping `x` BF16 in shared loses.** It halves the shared bytes per block (4 KiB → 2 KiB at
  K = 1024, 8 → 4 KiB at K = 2048) but the occupancy report says blocks/CU was never the binding
  limit here, and it costs an extra BF16→F32 conversion per element inside the inner loop. Net:
  identical on the K = 1024 input projections (`in_proj_qkv` 167.9 vs 170.4 GB/s) and clearly worse
  on the K = 2048 output projections (`o_proj` 116.3 → 80.9, `out_proj` 109.0 → 80.5 GB/s). The
  engine should hand this kernel the F32 norm output and skip the round-trip; the packed
  instantiation stays in the binary so the claim stays checkable.
- **rows/block 32 is the best setting found**, +8 % over the shipped default of 8 (512 threads →
  64 warps/block, 2 blocks/CU). Not changed in this commit: the default is what every number above
  was measured with, and moving it is a separate, measurable decision for W5.
- **Small-row shapes are block-starved, not bandwidth-limited.** `k_proj`/`v_proj` (512 rows)
  launch 64 blocks against 20 CUs × 8 resident blocks — 0.4 of one occupancy wave — and reach
  39–40 % of the probe. The 16-row `in_proj_b/a` are pure launch cost: **3.2 µs to read 32 KiB**,
  i.e. ~1.7 µs of queueing plus a kernel that finishes before the queue drains. That is the
  measurement behind §04.1's suggestion: folding `b` and `a` into the `in_proj_qkv` launch removes
  36 launches and ~0.11 ms/token for 576 KiB of extra rows in a launch that is already 1.3 ms.
  The packed line confirms it from the other side — `PACK_inproj` runs at 181.9 GB/s, *above*
  `in_proj_qkv` alone at 167.9, because more rows means more blocks.
- **The gap to the probe is not occupancy on the big shapes.** The token plan runs 4112 blocks
  (waves/CU 411) for `PACK_inproj`, and still lands at 71 % at best. `mlp_gemv`'s one-wave probe
  says the copy ceiling itself is 84 % at 64 blocks × 512 threads, so ~13 % is this kernel's
  reduction (512 threads → 16-wide butterfly + 8-element strided serial sums per thread, 2 BF16
  loads + 1 F32 op per element). Closing it means the §07.3 item 1 experiment: fewer threads per
  row and more in-flight rows per block, i.e. the multi-row-per-block variant this kernel does not
  have yet. **This op does not claim that number; it is recorded as W5's starting gap.**
- Numbers here are one burst of the real token plan with the weights of every layer, so the same
  weight traffic as a real token, not the same tensor. Absolute ms moves a few % run to run;
  ratios against the probe are the stable form.

## What graduates into `engine/` (input to W5's launch plan)

Per decode token, this op supports: **48 launches** (24 packed input projections + 24 output
projections) covering 445.7 MiB of weights, measured **2.99 ms at 156 GB/s (66 % of probe,
provisional)**. The unfused 114-launch form would pay ~66 extra launches at the 1.8 µs floor
`mlp_gemv` measured on this box, i.e. **~3.1 ms** — that 3.1 is arithmetic on a measured launch
cost, not a second measurement. Folding `in_proj_b`/`in_proj_a` — which §04.1 suggests and this op
now measures — is worth ~0.11 ms of that. The kernel graduates as `engine::proj_gemv<IN, PACK_X>`
with the plan struct as its interface; what does **not** graduate is any assumption that the
launch count is the bottleneck. It isn't: at rows/block 32 the same 48 launches run in 2.79 ms,
and the remaining ~0.8 ms to roofline is reduction efficiency, which is a kernel change, not a
plan change. The two plan-level knobs W5 can take from here are the `rows/block` 32 setting and
handing the kernel the F32 norm output (`PACK_X = false`, measured above). W5 also inherits §07.4's open question — projections at 30 % of a 10 ms/token budget
only stay inside the budget if `lm_head` (~485 MiB/token) and the MLP (~353 MiB/token) are at the
probe too, which `mlp_gemv` measured at 50–62 % and `lm_head_gemv` at 52 %.

## P5 — the token-batched form: `proj_gemv_tokens`, and what it is gated on

A prefill wave hands a projection a `[T, IN]` tile instead of one row. `proj_gemv_tokens<IN, TPB,
PACK_X>` gives each block `rpb` rows **and** a tile of `TPB` tokens, so a weight row is read once per
tile instead of once per token — the family's only bandwidth lever, since the narrow shapes are
shape-limited (`k_proj`/`v_proj` 39–40 % of the probe at 64 blocks, `in_proj_a`/`b` at 5 %, the table
above). `TPB` is bounded by shared memory, not registers: the launcher picks 4 at `IN = 1024` and 2 at
`IN = 2048`, both 16 KiB/block, the same budget `mlp_gemv` settled on. **`TPB` is not measured** —
`--tpb` exists so the main agent's timing pass can sweep it (engine/performance-status.md §6), and no
duration appears anywhere in this section.

One kernel, not two. `engine/kernels_proj.cpp` instantiates the tile form at **every** width,
`tokens == 1` included, where `TPB = 1` and `gridDim.y = 1` are the k=1 launch shape and the k=1
arithmetic — one staged row, one accumulator, same shared footprint. That is deliberate: P4's MLP
launcher keeps a `tokens > 1` branch and a single-token branch, i.e. two kernels to keep honest, and
P5 was briefed not to repeat that. `proj_gemv<IN, PACK_X>` (the original) stays in the header because
the gate needs an independent per-row reference to diff the tile against (`--kernel k1`) and because
the `--x-packed` timing sweep still runs through it; the engine never launches it.

### `run.py --tokens <p64|p300>` — the batched stage

Same four rules as the k=1 gate, judged **per row** (`worst_over_rows`, worst row printed per block),
against the prefill pins `L{ll}.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,q_proj,k_proj,v_proj,out_proj,
o_proj}.out`, after re-running the pairing proof on the tiles (`prefill_verify`: the rebuilt operand
reproduces every row of every pin bit-exactly with torch's own matmul, and each wrong norm fails to —
0 failures on both fixtures, both block types). Three claims, in order of how much the engine needs them:

| claim | `p64` (T = 46) | `p300` (T = 277) |
|---|---|---|
| **bit-identity vs the k=1 kernel** (tile row *t* must BE the k=1 kernel's answer for token *t*, sampled tokens `0,1,T/2,T−2,T−1`, every projection of the layer) | **138 400 elements compared, 0 differ** (4 layers) | **138 400 elements compared, 0 differ** (4 layers) |
| the pins, per row: worst row of every block | `cos ≥ 0.99999997`, `meanrel ≤ 8.56e−06`, `ndiff ≤ 0.293 %` on the wide shapes, **zero large elements over tolerance** anywhere | `cos ≥ 0.99999998`, `meanrel ≤ 9.92e−06`, `ndiff ≤ 0.391 %` wide, **6.25 % = 1 element of 16** on `in_proj_a`/`b` at 1 ULP (the granularity trap above), 0 large over tolerance |
| the identity check can fail: `--neg tail` on the tile run only | 408 843 / 425 408 (L00) and 258–266 k of 282 624 (attn layers) tile elements moved | 2 466 286 / 2 561 696, and 1.56–1.60 M / 1 701 888 |

The ULP outliers are the same population the k=1 gate already excuses — a handful of elements per row, at
`|ref|/max` of 1e−6 or less (cancellation, `floor_probe.py`), the raw maximum reaching 193 ULP at such
elements while **no large element anywhere exceeds the band** — so **batching moved nothing**: the batched rows
are the k=1 bits, and the k=1 bits are what the pins reproduce. The per-row pin numbers above are therefore the
*kernel-vs-reference* distance, not a batching distance; the batching distance is the 0 in the first row.

### rows = 1 stayed green, and stayed green through the new kernel

`run.py` now launches the tile form by default at `tokens = 1`, so the existing 136-comparison gate and
all eleven negatives ran **through `proj_gemv_tokens<IN, 1, PACK_X>`** — the instantiation the engine
uses for decode — and passed unchanged (`136 comparisons, 0 failures`; every negative caught where
declared and nowhere else). `--kernel k1` reproduces the old path: the two are the same arithmetic, and
the diff above (138 400 elements, 0 differ) is the proof rather than an argument.

**Not taken:** any timing. The commands the main agent's clean-window pass needs are
`run.py --time` (unchanged, k=1 shapes) and, for the batched family, `--mode burst --plan … --tokens 295
--tpb {1,2,4}` with the plan of §"Performance"; `Job::bytes()` already counts weight bytes per
*TILE* rather than per token, which is the divisor that matters.

## Traps found here (added to the index)

- **`min(wall, event)` is a trap on this iGPU.** `hipEventElapsedTime` returns 0 — or, once, −1.9e6
  ms — around short work. Taking the minimum silently turned one shape into "25433 GB/s". Wall-clock
  is the number; the event value is printed as a cross-check and labelled unmeasured when it
  disagrees.
- **The `o_proj` pin is the gated attention output's projection**, not `attn_out`'s (§05.6's
  `o_proj(attn(gate))`). Reconstructing from the raw attention output is off by ~5 % ULP-mean and
  would have looked "close"; the gate catches it as a wrong operand.
- **`ndiff ≥ 1` is vacuous on a 16-row output.** `in_proj_b/a` are bit-exact; a 1-element allowance
  on 16 elements is 6 % of the tensor. The floor probe, not the allowance, is what makes those two
  rows mean something.
- **A gate that runs the previous binary passes vacuously.** `run.py` rebuilt only when `proj_gemv`
  did not exist, so a source edit that failed to compile left the OLD binary in place and the k=1
  stage printed PASS against a kernel that was no longer in the tree — caught mid-P5 only because the
  new stage then failed on `unknown arg --tokens`. `run.py` now rebuilds whenever either source is
  newer than the binary.

## D1 — the decode variants (added 2026-10-21). **PERF ROWS UNMEASURED: `<main-agent window>`**

`progress/2026-10-21/decode-attribution-2026-10-21.md` bills the projection family **4.234 ms/token in
situ for 466.7 MB** against a **1.973 ms** byte floor, with the lines for this route fixed before any
kernel at **WIN ≤ 2.60 ms** (≥ 76 % of the 237 GB/s probe) and **KILL ≥ 3.05 ms** (< 9 % of the step).
The variant axis, the plan shapes, the gate and the sweep are built; **no millisecond figure in this
section was taken by the worker who built them** — AGENTS.md reserves timing for an exclusive
main-agent window, and `sweep_decode.sh` is the one-command handoff. Read the rows below together with
the two published numbers above (`2.9895 ms` at rows/block 8, `2.785 ms` at 32), which sit *between*
the lines and were taken with another process on the GPU.

Whole-token plan `plan_token_packed.txt`: 48 launches, 463.3 MiB of distinct weights (the bytes
behind the 1.973 ms floor), one row per variant at its own default geometry.

| variant | lanes/row | rows/blk | threads | launches | ms/plan | GB/s | % probe | spread (min..max of 3 reps) |
|---|---|---|---|---|---|---|---|---|
| `v0-baseline` — the shipping `proj_gemv_tokens` | 32 | 8 | 256 | 48 | `<main-agent window>` | | | |
| `v1-unroll` — compile-time trip count, `#pragma unroll 4` | 32 | 8 | 256 | 48 | `<main-agent window>` | | | |
| `v2-lpw16` — 2 rows per wavefront | 16 | 8 | 128 | 48 | `<main-agent window>` | | | |
| `v3-lpw8` — 4 rows per wavefront | 8 | 16 | 128 | 48 | `<main-agent window>` | | | |
| `v4-lpw4` — 8 rows per wavefront | 4 | 32 | 128 | 48 | `<main-agent window>` | | | |
| `v5-vx` — `v1` + `uint4` staging of `x` | 32 | 8 | 256 | 48 | `<main-agent window>` | | | |
| `v6-lpw8-vx` | 8 | 16 | 128 | 48 | `<main-agent window>` | | | |

Launch-plan lever: identical bytes, different launch counts. `unpacked` is the **114** launches
`engine/decode_step.h` issues today (18 GDN layers × {qkv,z,b,a,out} = 90, 6 attention layers ×
{q,k,v,o} = 24); `mixed` is those 114 with **only** the 36 [16,1024] `in_proj_b`/`in_proj_a` jobs
tagged `VAR=v3-lpw8:32`, so the small shapes are priced alone and every big shape runs as shipped.

| variant | plan | launches | ms/plan | GB/s | % probe | what the row decides |
|---|---|---|---|---|---|---|
| `v0-baseline` | packed | 48 | `<main-agent window>` | | | the folded plan, re-taken in a clean window |
| `v0-baseline` | unpacked | 114 | `<main-agent window>` | | | the 66-launch bill: 66 × 1.85 µs ≈ 0.12 ms, or ≈ 0.21 ms at the 3.2 µs the small shapes actually cost |
| `v0-baseline` | mixed | 114 | `<main-agent window>` | | | whether the narrow form alone recovers the ~0.11 ms those 36 launches cost |
| best of the packed arms | packed | 48 | `<main-agent window>` | | | **the row the WIN/KILL verdict is read from** |

Rows/block is swept per variant inside the same arm list (`--rows-list 8,16,32`), so the rows/block
8-vs-32 finding is re-taken in the window that measures the variants rather than compared across
windows — the trap AGENTS.md records for a ruler re-taken in a different window.

### Settled without a window (CPU gate, no timing taken)

```
.venv/bin/python prototype/proj_gemv/run.py --layers all --steps decode0 --negative
  PASS: 798 pin comparisons, 684 cross-variant comparisons over 1,219,968 elements,
        49 injected-bug cases (7 bugs × 7 variants) rejected where declared and nowhere else,
        4 wrong-operand cases unchanged, 0 failures.
```

Every one of the 24 layers, every projection block (`in_proj_qkv`'s q/k/v sections, `in_proj_z`,
`in_proj_b`, `in_proj_a`, `out_proj` / `q_proj`,`k_proj`,`v_proj`,`o_proj`), every variant, each
against its own §10.5 pin, with the criteria and constants of this file **unchanged — no band was
widened for any variant**. Two results matter more than a ms row:

* **`v1-unroll` and `v5-vx` are bit-identical to the shipping kernel** across all 1.22 M compared
  elements. Either could be adopted without a numerics argument, because neither can move the legal
  route by a bit. `v5-vx` is bit-identical for a reason worth writing down: widening a *copy* is not a
  change to a sum. `v1-unroll` is, because the compile-time trip count visits the same words in the
  same order in each lane.
* **The four `LPW < 32` forms are band-judged, and that is the honest label rather than a hedge.**
  Reducing a K = 1024 dot over 8 lanes instead of 32 changes the parentheses of the sum, and §F1
  already established that merely reordering the reference's own F32 sum moves ~1 element in 1e5 by
  > 1 ULP at cancelling elements. The gate's own line for L18 `in_proj_z` names the instrument: the
  **shipping** kernel sits 40 ULP from the pin at one element of 2048 whose `|ref|/max` is 1.9e-08,
  with `over-tol large: 0`. A raw `maxulp <= 1` rule between variants therefore fails on correct code
  — it did, on its first run: 11 cases, 2 to 384 ULP, every one at a cancelling element — so the
  cross-variant check applies exactly the band-relative four criteria the pins get, and nothing else.

### Corrections this section makes to the one above

* "the multi-row-per-block variant this kernel does not have yet": it has one, and the axis turned out
  to be lanes-per-row alone, not an independent rows-per-warp — at `LPW = 32` a row already owns a
  whole wavefront, so the only way to put more rows in flight per group of lanes is to give each row
  fewer lanes, which puts 32/`LPW` rows in every wavefront for free. "Fewer threads per row" and
  "more in-flight rows per block" are one dial here, not two.
* "vector width (bf16 128-bit loads)", listed as an untried lever in the D1 attribution, was already
  true of the weight walk: both this kernel and `lm_head` read weights as `uint4`, 16 B per lane per
  step, and always have. The un-vectorised side was the cooperative `x` staging loop, which moved one
  element per lane-step with an `i % IN` / `i / IN` index split — that is what `v5-vx`/`v6-lpw8-vx`
  price, and the staging bill scales with **blocks** (1028 blocks × 1024 scalar loads at rpb 8 for
  `PACK_inproj`, i.e. ~1.3 staging loads per weight load), which is why its expected gain shrinks as
  rows/block grows.
* The unroll was not merely absent: on the runtime-bounded word loop clang **refused** it
  (`loop not unrolled: the optimizer was unable to perform the requested transformation`, first build
  of this file), so an experiment that added `#pragma unroll` while keeping the runtime bound would
  have measured a no-op and looked like a null result. The trip count has to be compile-time, which
  the negatives now avoid by skipping tail iterations instead of shrinking the bound.

### Explicitly not priced, so it is not mistaken for covered

* **split-K — ruled out, not left unfinished** (accepted by the supervisor on 2026-10-21, on the
  arithmetic below rather than by omission). Block count is its only prize on these shapes and
  `rows/block` already reaches it (`k_proj` at rpb 2 is 256 blocks, not 64), so a split buys a knob this
  sweep already sweeps; `SPLITK = 2` would additionally cut a [512,1024] row to 2 words per lane, a
  *shorter* per-thread dependent chain, which is the wrong direction for a latency-bound one-row GEMV,
  and it costs a second launch plus a partial buffer plus a deterministic merge. The small shapes'
  actual measured pathology is launch cost (~3.2 µs for 32 KiB), and the `mixed` arm prices that
  directly. Re-open only if a `mixed`/`unpacked` row shows those launches costing more than that after
  the narrow geometry.
* **hipBLASLt at `M = 1` — unpriced, and blocked on a numerics-authority decision (a user call), not on
  a harness**, so a later pass should not re-litigate it with a cheaper probe. The engine keeps
  `M = 1` off the library path deliberately: `resolve_wave_gemm` **forces the hand GEMV at one row**
  (`engine/gate.py:769`, which is why `gate.py` carries a separately measured `row_1cos_max_lt` band for
  the hand↔hipBLASLt pairing at all), and `engine/kernels_gemm.cpp:19` states the rule that band serves
  — a library-chosen kernel differs in **accumulation order**, so it "is gated by the same cascade band
  as `library` and not by byte-identity". Routing `M = 1` through hipBLASLt would move the decode
  projection out of the byte-identity guarantee it holds today (`engine/kernels_proj.cpp:70`: `tokens ==
  1` is "bit-for-bit the shape this launcher issued before P5", which `run.py --tokens` requires to agree
  exactly) and under a reassociation band. A ceiling number that cannot be admitted is not a cheap
  number. Separately, the tree's only hipBLASLt harness refuses `M <= 1` on purpose
  (`gemm_algo_sweep.cpp:186,298`: "`M<=1` is the hand GEMV whatever the run asked for") and belongs to
  the prefill campaign with its own gates and calibration tables; it was not touched by this work.

## D1's first window (2026-10-21): what it read, and the three rig defects it proved

All millisecond figures in this section were taken by the **main agent** in one blocking
`tools/gpu_lock.sh` hold with zero subagents active, `util` 9 %→79 %, phase-2 wall **25 s** — not by
the worker who built the rig. 12 arms, 0 canary failures, admission 798 comparisons / 0 failures.
Whole-token packed plan, 48 launches, 463.3 MiB, `--iters 30 --reps 3`, one invocation per arm and one
weight upload per invocation. `ms` columns are `ms_per_plan` **as the rig reported it at the time**,
which was the *last* rep — see defect 2.

| variant (packed plan) | rpb 8 | rpb 16 | rpb 32 | verdict on the row |
|---|---|---|---|---|
| `v0-baseline` — the shipping kernel | 3.547 (55.6 %) | **17.497** (min 3.544) | 11.462 (min 7.186) | two of three points were self-contradictory; see defect 2 |
| `v1-unroll` | 3.548 | 3.577 | **3.272** | the unroll/compile-time-trip-count form, bit-identical to `v0` |
| `v5-vx` — `v1` + vector `x` staging | 3.410 | 3.395 | **3.268** (60.3 %) | **best arm of the window**, bit-identical to `v0` |
| `v2-lpw16` | 4.21–4.65 | | | refuted |
| `v3-lpw8` | 6.58–6.97 | | | refuted |
| `v4-lpw4` | 6.74–7.52 | | | refuted |
| `v6-lpw8-vx` | 6.47–6.74 | | | refuted |
| `v0-baseline` unpacked (114 launches) | 4.135 | 4.154 | 3.829 | the plan pair, above |
| `v0-baseline` mixed (114, 36 small jobs narrow) | 4.321 | 4.294 | 4.022 | narrower-is-better-for-the-small-shapes did not survive contact |

**Where that leaves the pre-registered lines.** Best arm `v5-vx` at 3.268 ms is **above the 3.05 ms KILL
line**, so on its face the route is dead. It cannot be read that way, because the control did not
reproduce: `v0` rpb 8 read **3.547 against the published 2.9895, a +18.6 % offset on the arm that is
byte-for-byte the shipped kernel at the shipped geometry**. `engine/performance-status.md` §6 is the
rule for exactly this: a line cannot be adjudicated across an offset of that size, so the corrected
estimate for the best arm is 3.268 / 1.186 ≈ **2.76 ms, which is between the lines — report, do not
decide**. No WIN/KILL call comes out of this window, and none should be quoted from it.

The lanes-per-row axis is settled regardless of that offset, because its spread (4.2–7.5 ms) is 20×
the size of it: at 25–47 % of the probe, the narrow-split rows are not slower by a margin, they are on
the other side of the mechanism. 8 lanes per row leaves 16 words in a lane's dependent chain and the
load-level parallelism that `lm_head_gemv`'s four independent 16-B loads per lane provide never
materialises. **Not being chased further** (the main agent's call, 2026-10-21): `v2`,`v3`,`v4`,`v6` are
closed, and the shape search continues on the two bit-identical forms and on the launch plan.

### The sub-finding that is banked regardless of how the shape search ends

Inside one window at identical geometry: **`v1` packed 3.272 ms vs `v1` unpacked 3.902 ms = −16.2 %
from folding 114 launches into 48**, with the same bytes and the same kernels. My CPU prediction was
−4…−7 %, computed as 66 launches × the ledger's 1.85 µs launch floor (≈ 0.12 ms) or × the 3.2 µs the
small shapes cost (≈ 0.21 ms). The measured difference is **0.63 ms for 66 launches ≈ 9.5 µs per
launch**, 3–5× either figure in `engine/performance-status.md`. Two readings, both recorded because
they imply different remedies: either the per-launch cost on this iGPU is materially above the ledger's
1.85 µs whenever launches are small and dependent (which makes every launch-count reduction in the
engine worth more than booked), or the unpacked line is paying something structural — the extra 36
`in_proj_b`/`in_proj_a` launches are the known 3.2 µs-for-32-KiB cases and the packed fold hides all of
them, which is the ~0.11 ms §4.1's suggestion already priced. The re-take re-prices the pair on the
shipped kernel at the reproduced geometry, so the number that goes into the engine is measured on what
ships.

### Defect 1 — threads/block and rows/block were one number, and the published row for "512 threads" was not what the prose said

`proj_gemv.hip.cpp` computed `const int threads = rpb * lpr;`, so `v0` ran 256 threads at rpb 8 and
1024 at rpb 32, and the thread count could not be asked for independently. Fixed: `--threads N` (alias
`--tpr`), whole wavefronts, bounded by `maxThreadsPerBlock`, default 0 = the variant's own `rpb*lanes`
so **every existing number stays the same number**. `--dump-plan` now echoes the resolved `threads` and
a `trips` field (how many passes a block makes over its own rows), and the plan field grew to
`VAR=name[:rows[:threads]]` so one upload can still carry a mixed geometry.

Two consequences worth more than the flag:

* **Variant 0 refuses a non-default thread count, and that refusal is deliberate.** Variant 0 *is* the
  kernel `engine/kernels_proj.cpp` compiles, and that launcher emits `kRowsPerBlock * warp_width()`
  threads; timing a block size the engine cannot emit would price something unportable. The decoupled
  axis therefore runs on `v1-unroll`/`v5-vx` — which are bit-identical to `v0` (72 pin + 63
  cross-point comparisons at `rpb 32 @ 512 threads` = two trips per block, 0 differing elements,
  re-run 2026-10-21), so a difference on that axis is geometry and nothing else.
* **The published sweep never held threads fixed.** The first window read `results.md:34` ("One work
  item = one output row: `gridDim = M`, 512 threads"), and `:207`/`:220`, as a record of the geometry
  behind the table at `:195`, and concluded that rpb 32 at 512 threads was unreachable and the published
  best might live there. It is not what produced those numbers. Every revision of this launcher —
  5d6405f (the commit that created the op), aa1b350 (the commit that published 2.9895 and 2.785), and
  HEAD — computes `threads = rpb * ws`, and the kernel has had 32 lanes per row since 5d6405f
  (`row = blockIdx.x*rpb + threadIdx.x/ws`). So the `rows/block` column *implied* the thread count:
  **rpb 2 → 64, 4 → 128, 8 → 256, 16 → 512, 32 → 1024 threads.** The 512-thread point in the published
  sweep is rpb 16 at **2.946 ms**, and 2.785 is rpb 32 at 1024 threads. `results.md:34` describes spec
  §07.3's form (512 threads, one row, `K/(512×PACK_X)` per thread) and has been inconsistent with the
  kernel since the commit that wrote it. Corrected control arms therefore reproduce each published row
  at the geometry that row actually used: rpb 8 @256 vs 2.9895, rpb 16 @512 vs 2.946, rpb 32 @1024 vs
  2.785 — and the attractive-but-unpublished "rpb 8/32 @ 512 threads" is priced as an ordinary arm
  (A4/A5), labelled report-only, because there is no published number there to reproduce.

### Defect 2 — a rep ran 5× slow and `ms_per_plan` reported it

rpb 16 read min 3.544 / max **17.497 ms**; rpb 32 read min 7.186 / max 11.462. 17.5 ms over 463 MiB is
**26 GB/s**, which is not a slow kernel — it is the first GPU pass over freshly uploaded pages, measured
at ~35 GB/s in `devices/strix-halo-unified-memory.md`, and it was the number the line printed because
`ms_per_plan` was the **last** rep. Four changes, all in the launcher so no caller has to remember them:

* every timed point is warmed first — `--warm N` whole-plan passes, default 2, before the first timed
  rep (one pass was not enough);
* `ms_per_plan` is now `ms_min`, i.e. best-of-N, which is this repo's convention
  (`prototype/README.md`: "timings are best-of-N");
* `ratio=ms_max/ms_min` is printed on the RESULT line with `pt=ok|provisional|void`, and the sweep
  recomputes it rather than trusting the launcher — one check is not two;
* a point with `ratio > 2` prints **`VOID` in the ms, GB/s and %-of-probe columns** and no millisecond
  figure at all, so a bimodal sample cannot be tabulated. `provisional` (>1.10) is the level at which
  `engine/performance-status.md` §6 says a row cannot be compared across windows, and it is labelled
  rather than suppressed — a provisional row is still the best number this window has.

Note what the offset and the void together imply. Every arm sits ~18.6 % above its published counterpart,
including arms whose geometry is bit-for-bit the published one (which is why the offset cannot be a
geometry mismatch), and two points disagree with themselves by 1.6–4.9×. That is a window-level effect
(79 % util at release, fresh pages, dispatch-bound plan), not a property of the kernels — and it is
exactly what the A1–A3 ±3 % control canaries exist to catch before a shape conclusion is drawn.

### Defect 3 — the gated set was not the timed set

Admission judged each variant at its own default rows/block (`--rows 0`) while the sweep timed rpb
{8,16,32}: a case list about different bytes than the ones measured, which is the D59 trap named in
AGENTS.md. Now one file, `gate_grid.txt`, lists the points as `variant[:rpb][@threads]`; phase 1 runs
`run.py --grid-file` over exactly those points, phase 2 times exactly those points, and the run ends
with `gated N, timed M, N==M: yes`. If the counts differ the script prints the diff between the two
lists and exits non-zero rather than letting any row be read as adjudicable. Plan shape is deliberately
not part of a point — packed and unpacked emit the same per-projection output, which is what the gate
judges — so the plan pair is one gated point timed against two plan files, not a secret second
geometry. Verified without a timing: the re-take grid is 15 distinct points, admission prints
`GATED 15 point(s) … 1710 comparisons, 0 failures` (15 points × 114 projection blocks), and the
canonical gate is unchanged at 6384 comparisons / 0 failures with no band widened.

### The re-take, in one command

```bash
bash prototype/proj_gemv/sweep_decode.sh 30 3 decode0     # one hold; ~15 points, warm + 3 reps each
NO_TIMING=1 bash prototype/proj_gemv/sweep_decode.sh      # phases 0-1 only: grid + admission, no timing
bash prototype/proj_gemv/sweep_decode.sh selftest         # canary + VOID parsing, no GPU, no lock
```

---

## §D1-W2 — the DVFS window (main agent, 2026-10-21): the ranking was partly the clock

Window facts, so the numbers below carry their provenance: main agent, blocking, foreground, **zero
subagents active**, one `tools/gpu_lock.sh` hold, `util` 9 → 30 % at acquire/release and discounted per
`engine/performance-status.md` §6. Phase-2 GPU work was **13 s** for 15 points × 3 reps. Admission:
`GATED 15 point(s)`, **1710 comparisons, 0 failures**, `gated 15, timed 15, N==M: yes`, 0 canary
failures. The three fixes from the first window all held: no bimodal rep printed as a result (internal
spread 1–2 %, no `VOID` from self-disagreement), and the thread axis is real (the launcher echoed the
requested `threads`, so `rpb32@512` ran as asked).

### The numbers, as taken

| arm | variant | plan | rows/block | threads | ms/plan | published | off published | clk MHz at the row |
|---|---|---|---|---|---|---|---|---|
| A1 | `v0-baseline` | packed | 8 | 256 (default) | 3.5377 | 2.9895 | **+18.3 %** | *(main agent to fill)* |
| A2 | `v0-baseline` | packed | 16 | 512 (default) | 3.5289 | 2.946 | **+19.8 %** | *(main agent to fill)* |
| A3 | `v0-baseline` | packed | 32 | 1024 (default) | 3.2778 | 2.785 | **+17.7 %** | *(main agent to fill)* |
| best search arm | `v1-unroll` | packed | 8 | 1024 | **2.4845** | — | report-only, 79.4 % of probe | *(main agent to fill)* |

The shipped kernel's best in-window figure is A3's 3.2778 ms; the best search arm is 2.4845 ms, i.e.
**−24.2 %** relative to it. The WIN line is `≤ 2.60 ms` and the KILL line `≥ 3.05 ms`, both unchanged by this
section, and 2.4845 ms sits below WIN. **This window does not bank that WIN**, for two reasons that are
properties of the window rather than of the shapes:

### Why the clock moves these numbers at all: sclk moves, mclk does not

Recorded here, next to the blank cells it explains, because it is a finding about D1 and not about the rig.
What moves between windows is **sclk**; `mclk` reads **1000 MHz throughout** and LPDDR5x bandwidth does not
move with sclk. So the sensitivity of a kernel here is decided by what it is waiting on:

* a **bandwidth-bound** kernel is clock-insensitive — the evidence is the main agent's own cross-window
  reproductions: `lm_head` comes back at **99.6 % of probe** window to window, and the 4,032-lane decode
  rows repeat to **0.1 %**, across windows whose sclk differed;
* a **latency- or issue-bound** kernel is strongly clock-sensitive, because its cost is a count of cycles and
  cycles are what the clock just changed. `proj` at ~60 % of probe is this kind.

Two consequences. First, **the published `proj` rows missing by 18–20 % in these windows is most likely a
sustained-clock difference, not a wrong old number** — which is also why the controls miss by the same amount
in the same direction while agreeing with each other. Second, **every `proj` figure has to be quoted with its
clock**: a large part of what looks like a slow kernel here is the clock it happened to run at, and an
improvement quoted without one cannot be compared with the row it replaces, in either direction. The rig now
carries the clock on every row and a normalised `ms × clk / 2400` reading column for cross-window work; the
win/kill call stays on raw ms from a row whose clock passed the floor.

1. **The controls miss together, in one direction.** Three geometries of one kernel, three different
   thread counts, all ~18 % above their published rows while agreeing with each other to 1–2 %. A kernel
   cannot be 18 % slower in three shapes and 1 % self-consistent by accident; the offset belongs to the
   window. That is also what makes `−24.2 %` an upper bound rather than a measurement: the shipping-kernel
   rows ran first and the winning arm ran later, so part of that gap is the window getting faster.
2. **A 13-second timed set is shorter than the clock's ramp to boost.** §36 of
   `progress/2026-10-21/performance-status-2026-10-21.md` measured this part at **819 / 827 / 838 MHz idle
   and 2432 / 2437 / 2423 MHz immediately after a rep** — a range that swallows an 18 % drift without
   needing any other explanation. The arm table ran monotonically faster down its own order, which is that
   ramp read as a ranking.

So: **no WIN/KILL adjudication from this window.** What it does establish is the confound and its fix.

### What changed in the rig because of it

1. **A clock burn before arm 1.** `proj_gemv --mode burn --burn SECONDS` launches the real plan on the
   real weights until *wall* seconds have passed (wall-timed out, so it costs the same on a faster
   machine) and prints `plans`/`launches` and deliberately **no millisecond figure** — a burn number reads
   like a measurement and is not one. The sweep runs `BURN=5` s by default and prints the clock before the
   burn and after it.
2. **A clock reading on every row, with a floor.** `rocm-smi --showclocks` (73 ms per read; sysfs
   `pp_dpm_sclk` is unreadable without root on this box) is read immediately before and after each burst,
   printed in a `CLK` column, and a row taken below `CLK_MIN` (default **2300 MHz**) prints `LOWCLK` and is
   excluded from every pool, minimum and verdict. `rocm-smi` is the *only* readable clock source here and it
   jitters by ±50 MHz between adjacent reads, so a `LOWCLK` label is a statement about the sensor and the
   floor, not a precise clock history; a window where no reading parses at all now fails loudly instead of
   reporting every row as `clk=?`.
3. **Interleaved, twice, in two different orders, min across passes.** The points are flattened and put in
   a **fixed pseudo-random order per pass** (seed printed, each order recorded in
   `$OUT/arm_order_p<N>.txt`), the whole set is run `PASSES=2` times, and each point's figure is the **min
   across passes** over usable rows only. Two ramp verdicts are printed either way, because a null is the
   result: *pass over pass* (is pass 2 still faster?), and a **paired position statistic** — each arm's
   residual off its own two-pass mean, correlated with how far its position moved between the two orders
   (`r` printed with the verdict). A position effect that survives that is a clock, not a ranking.
4. **The plan pair is driven by the variant that owns the axis.** Both C arms returned `rc=1` because they
   asked `v0-baseline` for `--threads 1024`, and `v0` is the kernel `engine/kernels_proj.cpp` compiles and
   refuses a shape the engine cannot emit. The pair now runs at the **winning search arm's** variant, rows
   and threads, so it is `v1-unroll` or `v5-vx` — both bit-identical to `v0` — and still isolates the launch
   plan. Banked from the first window and unchanged: packed 3.272 vs unpacked 3.902 at identical geometry,
   **−16.2 % from the 48-launch plan alone**.

### Four rig bugs the offline checks caught before this rig could lie

None of these needed a timing to find, and all four would have corrupted the re-take silently, which is why
they are recorded beside the window that prompted them.

* **Locale-dependent numerics.** This box's shell is `LC_NUMERIC=nl_NL.UTF-8`, where `awk` reads
  `"3.550000"` as the integer **3**. Under that locale every pooled minimum, control offset and ramp ratio
  in the summary was meaningless (min across passes became 3 vs 3 vs 2 vs 2). Fixed with `export LC_ALL=C`
  in `sweep_decode.sh`; the first window escaped it only because that shell happened to be a C locale.
* **Tab is IFS whitespace.** The per-row record file was tab-separated, so `while IFS=$'\t' read` collapsed
  runs of tabs and an empty `want` field (a search arm has no published number) shifted every column after
  it — a canned row read its usable-row count as its published target and reported a false +20 % control
  miss. The record is `|`-separated now, and the selftest asserts a single-pass arm with empty `want`/`tol`
  pools correctly.
* **The launcher refused its own new mode.** `--mode burn` was rejected by the mode whitelist (`once|burst`)
  that predates it, so the burn would have exited 1 and left the clock unbuilt while the sweep carried on and
  printed a full table. Caught by running the burn against a real plan file rather than reading the code. The
  whitelist now names `burn`, and `--mode burn` without `--burn SECONDS` exits non-zero instead of idling.
* **The first position metric could not mean anything.** It compared the mean ms of the first half of
  pass 1 against the second half — but with one order shared by both passes, a point's position is
  identical in both, so the comparison measured which *geometries* happened to land early. The stub run
  produced a confident "ramp survived interleaving, 4.6 %" out of nothing. Fixed by drawing an independent
  order per pass and correlating each arm's own residual with its own position offset, which is
  arm-identity-free; the selftest now feeds it a synthetic monotone ramp and refuses to pass if it fails to
  notice.

The selftest also drives the pooling rules with a canned record file — min across passes, a control
judged on the pooled min, `LOWCLK` rows excluded from the pool, a point with no usable row printing
`NO-USABLE-ROW`, and the winner being the best packed **search** arm rather than the fastest row — and
`stub_result.sh` prints canned `RESULT` lines for the whole sweep, so the interleaved order, the clock
column, `VOID`, `LOWCLK`, the pooling, the winner and the plan pair can all be run end to end without a GPU
and without a millisecond figure being produced:

```bash
BIN=$PWD/prototype/proj_gemv/stub_result.sh CLK_TOOL=$PWD/prototype/proj_gemv/stub_result.sh \
  SKIP_GATE=1 OUT=/tmp/proj_gemv/dryrun bash prototype/proj_gemv/sweep_decode.sh 1 1 decode0
STUB_CLK=800 …        # same command with the clock under the floor: every row LOWCLK, nothing adjudicable
```

### The re-take, in one command

```bash
bash prototype/proj_gemv/sweep_decode.sh 30 3 decode0    # burn, gate, 15 points x 2 passes interleaved
NO_TIMING=1 bash prototype/proj_gemv/sweep_decode.sh     # grid + admission only, nothing measured
bash prototype/proj_gemv/sweep_decode.sh selftest        # canary, VOID, clock parse, pooling; no GPU
```

Knobs, all defaults shown: `BURN=5` (s), `PASSES=2`, `CLK_MIN=2300` (MHz), `CLK_TOOL=rocm-smi`,
`SEED=d1-interleave-2`, `WARM=2`, `ITERS=30`, `REPS=3`. A row is a measurement only if its canary passed,
it disagrees with itself by at most 2× (`VOID` otherwise), and its clock was at or above `CLK_MIN`
(`LOWCLK` otherwise). A window is adjudicable only if the three controls reproduce their published rows
within ±3 % **on the pooled min**, `gated N == timed M`, and both ramp verdicts come back quiet — after
which the best `v1-unroll`/`v5-vx` arm's pooled min is compared with the unchanged lines:
**WIN ≤ 2.60 ms (≥76 % of probe)**, **KILL ≥ 3.05 ms (<9 % of step)**, and between them the number is
reported rather than decided.

## §D1-W3 — the clock-controlled window (main agent, 2026-10-21): the gate held, and that is the result

Window facts: main agent, blocking, foreground, **zero subagents active**, one `tools/gpu_lock.sh` hold,
`util` 9 → 41 %. Admission green (`GATED 15 point(s)`, `gated 15 == timed 15`). Clock trace for the window:
**1218 MHz before the burn, 2718 MHz immediately after it, 1749 MHz at the end of the 44-second timed
set** — 71 reads parsed, 0 unreadable.

**The rig refused to adjudicate, and it was right to: 33 of 34 rows printed `LOWCLK`.** One arm landed
usable — `v1-unroll` packed rpb 8 @512 threads at **2.882023 ms** — and the plan pair ran at 1674 / 1721 MHz,
giving packed 2.883941 vs unpacked 3.278635 = **−12.0 %**, which is marked `LOWCLK` and **not booked**. No
line was adjudicated on a ruler that was moving under it. The winner-selection logic is also confirmed
working: it picked `v1-unroll rpb8 t512`, the one shape the window could measure.

The trace is the finding. A single burn before arm 1 boosts the part to 2718 MHz and then the queue decays
to 1749 MHz over the set, so the burn fixed the first arm and nobody else's. That is why the burn moved
inside the process, per arm (§D1-W4 below).

### Why the clock moves this number at all: sclk moves, mclk does not

`mclk` read **1000 MHz throughout** every window, while `sclk` swung from 1218 to 2718 and back to 1749.
LPDDR5x bandwidth therefore does not move with the clock that is moving, and that splits the kernels here
into two classes:

* **Bandwidth-bound work is clock-insensitive.** The main agent's cross-window reproductions are the
  evidence: `lm_head` comes back at **99.6 % of probe** from window to window, and the 4,032-lane decode
  rows repeat to **0.1 %**, in windows whose sclk differed by tens of percent. Those numbers are stable
  because the thing they are limited by is not the thing that changed.
* **Latency- and issue-bound work is strongly clock-sensitive.** `proj` at ~60 % of probe is in this class:
  it is waiting on dependent work per token, so its cost is a count of cycles and cycles are what the clock
  just changed.

Two consequences, both of which change how this table must be read:

1. **The published `proj` rows missing by 18–20 % in these windows is most likely a sustained-clock
   difference, not a wrong old number.** The old rows were taken at whatever boost the box held during
   them; the new rows are taken in a window where the clock spent its time near 1.7–1.8 GHz. The controls
   all missing by the same amount in the same direction is exactly what that looks like.
2. **Any `proj` figure must be quoted with its clock, always.** A large part of what reads as "a slow kernel"
   here is the clock it happened to run at, and a proj improvement quoted without its clock is not
   comparable to anything, including the row it is meant to replace. The `clk` column of the table above
   stays blank for the main agent to fill from `$OUT/pass_rows.tsv`, whose ninth field is the adjudicating
   read for that row.

### Clock pinning: a dead end on this box, recorded so nobody retries it

Tried in the window above and unavailable to uid 1000:

* `/sys/.../power_dpm_force_performance_level` reads `auto`; writing it needs root.
* `rocm-smi --setperflevel high` goes through sudo, and there is **no passwordless sudo** here
  (`sudo -n` fails).

So the clock cannot be pinned from this account and **the rig is the control**: burn per arm, read the clock
on every row, refuse rows under the floor, and keep the window short. Anyone tempted to retry pinning should
note it needs a root action outside this session, not a flag.

## §D1-W4 — what the rig does about it, and how to read the two clock columns

1. **The burn moved inside the process, per arm.** `proj_gemv --preburn SECONDS` (sweep default
   `ARM_BURN=1.5`) runs the same sustained load as `--mode burn`, through the same code, immediately before
   the warm passes and the reps of **that** burst, on the plan that is about to be timed. Each row's
   `RESULT` line carries `preburn_s`/`preburn_plans`/`preburn_launches`, and the sweep's canary **fails the
   row** if the burn did not actually run for the requested time, so a launcher that quietly ignored
   `--preburn` cannot produce a plausible-looking table. The window-opening `BURN=5` stays, for the clock
   evidence at the edges, but it is the per-arm burn that is load-bearing now.
2. **The adjudicating clock is the read taken after the burst**, not before it. With the burn inside the
   process the before-read is the decayed clock the burn was about to lift (1218 MHz in the trace above), so
   judging `LOWCLK` on it would mark nearly every row unusable by construction; the after-read is the sample
   closest to the reps. Both are printed — the note on each row carries both — because the gap between them
   is itself the measurement of how far the queue decays between arms. The floor stays at 2300 MHz,
   unchanged.
3. **A normalised column exists, and it does not adjudicate.** `NORM` on a row and `NORMMIN` in the summary
   are `ms × clk / CLK_NORM` with `CLK_NORM=2400` printed in the column header, so a row taken at 1750 MHz
   and one taken at 2450 MHz can be compared, and a cross-window comparison has a defensible form instead of
   none. The WIN/KILL call is made on **raw `ms` from a row whose clock passed the floor** — a normalised
   figure is an inference about a clock the row did not run at, which is how a corrected number starts
   pretending to be a measurement. `LOWCLK` rows are excluded from `NORMMIN` exactly as from `MIN`; the
   selftest's canned file carries a LOWCLK row whose normalised value would look like the fastest arm in
   the table, and asserts it does not appear.
4. **If `LOWCLK` rows persist after the per-arm burn, that is not a rig failure — it is the answer.** It
   would mean this part cannot hold boost through a ~100 ms burst no matter what load precedes it, that any
   `proj` improvement then has to be quoted as a clock-paired statement (`at X MHz, ms …`, or the normalised
   figure with its divisor), and that the WIN line of 2.60 ms — set against published rows taken at a
   sustained boost — may need the user's attention as a band question rather than a speed question. Say which
   of those happened; do not quietly report the normalised column as if it were the raw one.

Everything the second window required is still in force: two independent fixed pseudo-random orders
(`PASSES=2`), `gated == timed`, the plan pair driven by the winning search arm, and the unchanged
**WIN ≤ 2.60 ms / KILL ≥ 3.05 ms** lines.

## §D1-W4 results — the window that decided it (main agent, 2026-10-21)

One `tools/gpu_lock.sh` hold, blocking, zero subagents, `util` 9 %→77 % (recorded, then discounted);
`bash prototype/proj_gemv/sweep_decode.sh 30 3 decode0`, admission 1710 comparisons / 0 failures,
`gated 15, timed 15 distinct points, N==M: yes`, **34 rows timed over 2 passes, 0 VOID, 0 LOWCLK**.
Clock: 1220 MHz before the window-opening burn, 2720 after it, 2408 at the end of the timed set,
71 reads parsed / 0 unreadable. Two independent interleaved orders, mean pass1 3.0853 vs pass2
3.0819 ms = **ratio 0.9989**, so there is no monotone ramp across the set; the paired-position
statistic fired at **r = +0.460** (later positions marginally slower within an arm), which is why the
quotable figure is the min across passes and not the first reading.

Rows below are **raw ms from rows that passed the 2300 MHz floor** — not the normalised column — and
every arm is the same 48-launch packed plan unless the row says otherwise.

| arm | variant | rpb | threads | min of 2 passes | % of probe | pass2/pass1 |
|---|---|---|---|---|---|---|
| A1 control | v0-baseline | 8 | 256 | 3.5333 | 59.0 | 0.999 |
| A2 control | v0-baseline | 16 | 512 | 3.5270 | 59.1 | 0.998 |
| A3 control | v0-baseline | 32 | 1024 | **3.2692** | 60.2 | 1.002 |
| B | v1-unroll | 8 | 512 | 2.8783 | 68.4 | 1.000 |
| **B** | **v1-unroll** | **8** | **1024** | **2.4693** | **79.4** | 0.986 |
| B | v1-unroll | 16 | 1024 | 2.8928 | 67.7 | 1.000 |
| B | v1-unroll | 32 | 1024 | 3.2852 | 60.0 | 0.992 |
| B | v5-vx (bit-identical) | 8 | 1024 | 2.5013 | 78.3 | 1.005 |
| **D packed** | v1-unroll | 8 | 1024 | **2.4942** | **79.1** | clock 2606 MHz |
| **D unpacked** | v1-unroll | 8 | 1024 | **2.8244** | 69.8 | clock 2625 MHz |

**Verdict against the lines fixed before any kernel existed: WIN line met.** 2.4693 ms against
**≤ 2.60 ms**, by a margin of 5 %, on an arm that is **bit-identical to the shipped kernel** (so no band
is in play). The two levers are separable and roughly multiplicative:

- **launch shape, −24.5 %**: the shipped kernel's best row in this same window is 3.2692 ms, the winning
  arm is 2.4693. The mechanism is lanes per row, not rows per block: at K = 1024, 128 lanes per row is
  8 bf16 per lane = **exactly one 128-bit load per lane**, where the shipped one-warp-per-row shape does
  four narrower loads and a four-times-longer serial FMA chain per lane. Rows/block is close to
  irrelevant on top of that (rpb 8/16/32 at 1024 threads: 2.469 / 2.893 / 3.285 — the *narrow* end wins,
  which is the opposite of what the published rows/block sweep implied).
- **plan packing, −11.7 %**: identical kernel, identical geometry, same window, ~2600 MHz on both sides:
  packed 48 launches 2.4942 vs unpacked 114 launches 2.8244. Against the engine's shipped combination
  (114 launches at the shipped shape, 4.135 ms in the first window) the two together are about **−40 % of
  the family**.

Two things this window does **not** claim. The published control rows do not reproduce: A1/A2/A3 land
18.2 / 19.8 / 17.4 % above 2.9895 / 2.946 / 2.785 at 0.1 % internal spread, so absolute ms from this
rig are not comparable to any pre-clock-control row and the win rests entirely on paired within-window
evidence — which is the form §6 of the live ledger trusts anyway. And 79.4 % of probe is not the end of
the road: `lm_head`, inside a real decode step and at whatever clock that step runs at, sustains
238.7 GB/s, so the remaining 21 % between 187 and 237 GB/s is still shape, not machine.
