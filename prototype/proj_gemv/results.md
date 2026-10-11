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
