# `mlp_gemv` — §03.3 MLP decode path (k = 1) in HIP

Status: **gate PASS** — 192 layer×step cases (all 24 layers × 8 decode steps of `dec8`), plus
the three negative injections, plus the cold bandwidth sweep. Binary: `mlp_gemv` (built by
`build.sh`), gate: `run.py`, floor measurement: `floor_probe.py`, rounding-policy probe:
`ulp_probe.py`. Authoritative run: `run.py --layers all --fixture dec8 --iters 60 --rows-sweep
4,8,16,32 --tile 6 --negative`.

This is the second third of the token's bytes: the MLP streams 24 × 22.02 MiB = **0.5285 GB per
token** (§03 / `bench/performance-status.md`), second only to `lm_head` (0.5085 GB) and well
ahead of the GDN state traffic.

## What runs

Two kernels per layer, one row (k = 1 decode), weights resident, `x` staged once in shared:

| kernel | shape (out × in) | weights/launch | what it fuses |
|---|---|---|---|
| `mlp_gateup<1024>` | gate 3584×1024 **and** up 3584×1024 | 14.68 MiB | two dots + the whole SwiGLU epilogue + optional raw `gate_pre`/`up_pre` dump |
| `mlp_down<3584>` | down 1024×3584 | 7.34 MiB | dot + BF16 rounding |

One wavefront per output row, `uint4` (8 × BF16) per lane, F32 accumulator, wave-shuffle
reduction, `warpSize` used everywhere so wave32 (this part) and wave64 (Navi) both work. Row
loop is `#pragma unroll`-free and index-strided: lane `l` takes words `l, l+ws, …`.

`mlp_gateup` is one kernel rather than two because the epilogue needs both dots' results: as
written it costs 24 launches per token that a split version would pay, and §07.4's launch floor
(1.85 µs × ~200 launches/token) says fusion is the lever, not a micro-optimisation.

## The gate (F1–F4), and what each case isolates

| case | input | isolates |
|---|---|---|
| **F1** `gate_pre`, `up_pre` | reference `in` | the two dots, independently |
| **F2** `act` | reference `in` | the SwiGLU rounding policy, nothing upstream |
| **F3** `out` | reference `act` | `down_proj` alone — the most the dot may move |
| **F4** `out` | **kernel's** `act` | the chain a real token runs (F3 ⊂ F4) |
| cross-check | — | `speccheck.layers.mlp` vs the pins (bit-identical, 0 ULP) |

Result over 192 cases (`run.py --layers all --fixture dec8`):

| output | cases | max ndiff | cases with any diff | raw max ULP | **gated max ULP** | min cos | max meanrel |
|---|---|---|---|---|---|---|---|
| F1 `gate_pre` | 192 | 4 / 3584 | 61 | 24 | **0 over tol** | 0.99999996 | 6.1e-6 |
| F1 `up_pre`   | 192 | 3 / 3584 | 84 | 8  | **0 over tol** | 0.99999999 | 3.4e-6 |
| F2 `act`      | 192 | 5 / 3584 | 101 | 24 | **0 over tol** | 0.99999998 | 4.5e-6 |
| F3 `out`      | 192 | 3 / 1024 | 48  | 2  | **0 over tol** | 0.99999993 | 1.6e-5 |
| F4 `out`      | 192 | 86 / 1024 | 85 | 182 | 52 (aggregate-gated) | 0.99999963 | 1.3e-4 |

"gated max ULP" is the distance of the worst element **large enough for a ULP to mean
something** (|ref| > 1e-4 of the tensor max). "raw max ULP" includes elements that cancelled to
~1e-6 of the max; the gap between the two columns is the whole subject of the next section.

The criteria, in words: no element above the cancellation band may exceed 1 ULP (2 for `act`),
≤2 % of elements may differ at all, and the §10.1-class aggregates (cos ≥ 0.9999,
meanrel ≤ 1e-2) must hold. F4 is judged on aggregates only, for the reason measured below.

## Why the per-element rule is scale-aware — `floor_probe.py`

A first reading of "reproduce the reference" is *max 1 ULP*. That bound is not achievable by a
correct kernel, and the probe is what says so rather than an argument:

- Reorder the reference's **own** F32 summation (blocked 128 + pairwise fold, and a plain
  sequential `einsum`, and an F64-exact sum rounded once) and compare to the pin: about **1
  element in 100 000 moves by more than 1 ULP** — `gate_pre` worst 20 ULP at 4 of 688 128
  elements, `out` worst 7 ULP at 10 of 589 824. Every one of those elements sits at
  |ref| ≈ 1e-6 of the tensor max, and the *exactly rounded* sum shows the same 4–16 ULP there.
  So the distance is the reference's own cancellation, not the kernel's error. Forbidding it
  forbids correctness.
- Inject a legal 1-ULP difference into `act` (one representable BF16 step, sign-aware) and
  re-run `down_proj`. Sized survey: **192 layer × step cases × 8 fixed trials = 1 536 perturbed
  dots, 1 572 864 output elements, `--perturb 1`.** All 192 cases produced at least one element
  more than 1 ULP from the pin; 7 483 elements did (0.48 % of them), the farthest being
  **28 665 ULP at |ref|/max = 1.9e-5** (L23 decode2). With 3 flips per trial: 16 021 offenders,
  worst 28 621 ULP. Note where the offenders sit: only **850 of the 7 483 are inside the 1e-4
  band** — the largest one is at 1.3e-1 of the tensor max, an element that is 13 % of the output
  and not a cancellation at all. So this is not F4 being judged loosely, it is F4 being unjudgeable
  per element: one legal 1-ULP flip anywhere in a 3584-term dot can move a *large* output element
  by tens of thousands of ULP. Hence cos (≥ 0.9999) and meanrel (≤ 1e-3) there, and nothing else.
  The earlier version of this bullet quoted "up to 142 ULP", which was the worst over an unstated
  1–4 trials per case: the loop stopped at each case's first positive trial (and its `break` was
  dead code, so the coverage looked wider than it was). Fixed-size trials and N on the line moved
  the quoted worst by ~200×, which is the whole reason to report N.

The probe had to be fixed twice before it could be believed, and both bugs are the kind that
manufactures a false conclusion: broadcasting a `(rows,)` block sum onto an `(rows, nblocks)`
accumulator (returns the dot times a power of two), and folding an odd number of block columns
(`in = 3584` gives 28 blocks → 14 → 7 → and a naive half-fold silently drops one). A floor
probe that lies is worse than no floor probe: with the broken version the reference "looked"
imprecise by 31721 ULP and the honest response would have been to loosen the real gate. The
docstring of `blocked_sum` claims that remainder class is handled, and it still had no assert —
`n % block != 0` would have dropped a K tail and printed it as the reference being imprecise,
which is the same failure mode one level down. It asserts now (`in = 1024` and `3584` are both
multiples of 128, so no published number came from a truncated sum).

## The SwiGLU rounding policy — `ulp_probe.py`

§03.3 pins HF's actual dtype path, and A-07's "accumulate in F32" is about the **dot**, not the
epilogue. What the reference does, measured element by element against the pins:

```
act = BF16( BF16( silu_F32( BF16(gate) ) ) × BF16(up) )      # three roundings
```

Computing the epilogue entirely in F32 — the tempting, "mathematically prettier" reading — moves
**27 % of `act` by 1 ULP**. The kernel therefore rounds gate and up to BF16, does silu in F32 on
the rounded value, rounds that to BF16, multiplies by the rounded `up`, and rounds again. Two
further details cost real debugging time:

- HF's order is `x * (1/(1+exp(-x)))`. Writing the algebraically identical `x / (1+exp(-x))`
  cost a 2 ULP outlier at a *large* element (a tie in the product's BF16 rounding one step
  downstream).
- An F32 `exp` that differs from libm's by 1 ULP flips the BF16 silu result roughly once per
  few thousand elements, and a 1-ULP BF16 input to the following multiply can round 2 ULP
  away. Hence `act`'s 2 ULP band: measured 0–5 of 3584 elements differ, never beyond 2 ULP at a
  large element. Matching libm bit-for-bit is not the deliverable; `--negative f32act` shows the
  gate still catches the policy that matters.

The `act` policy numbers above were originally printed by a comparator that was itself broken —
`ulp_probe.py` carried its own BF16 ordering with the dead `< 0` branch described in the trap
below. It now imports `prototype/bf16_ulp.py` and runs `selftest()` first, like `run.py` and
`floor_probe.py` do. Re-running it with the working comparator leaves every ULP column
unchanged (all-f32 `act`: 978/984/963/1001 of 3584 = 26.9–27.9 % at **max 1 ULP**, `>1 ULP: 0`,
for layers 0/3/11/23 of `decode0`; the three recompute rows are still BIT-EQUAL), because the
broken key is correct for same-sign pairs and this probe only ever compares a value against a
neighbour of itself. What did change: the old `cos` column was an F32 dot and printed things like
`1.00000048` for tensors that are bit-identical; the shared comparator does the aggregates in
F64 and prints exactly `1.00000000`. **No number in §03.3 or in this file needed correcting** —
but until the comparator was real, none of those ULP columns were evidence of anything.

## Negative tests — `run.py --negative`

The `speccheck/gates/run_negative.py` convention: inject plausible wrong kernels and require the
gate to refuse each **where the bug is and nowhere else**.

| injection | the bug | caught at |
|---|---|---|
| `half` | accumulate half of K (wrong reduction length) | `gate_pre`, `act`, `out` |
| `trunc` | round toward zero instead of RNE | `gate_pre`, `act`, `out` |
| `f32act` | SwiGLU entirely in F32 (the literal A-07 reading) | `act`, `out` — and **must not** move `gate_pre` |

`f32act` is the one a naive "max ULP at any element" rule would have missed and the one that
matters most, because it is the difference between matching HF and matching arithmetic. The
first version of this table was vacuous: the negative runs reused the previous layer's weight
files, so every injection "was caught" everywhere including where it was impossible. Now each
injection declares where it may and may not show up.

## Performance

Two facts dominate every number here, both already in `AGENTS.md`: the measured read roofline of
this box is 237 GB/s, and anything above it is cache residency, not speed.

`hipEventElapsedTime` returns **exactly 0** for kernels below ~1 ms on this iGPU, so all timing
here is burst wall-clock (submit N launches, one `hipDeviceSynchronize`).

Warm (same weights re-read `--iters` times — cache-resident, **not** a budget number):

| config | gateup | down |
|---|---|---|
| rows/block 4 | 0.0331 ms, 443.6 GB/s | 0.0194 ms, 379.2 GB/s |
| rows/block 8 (default) | 0.0315 ms, 466.6 GB/s | 0.0129 ms, 571.0 GB/s |
| rows/block 32 | 0.0291 ms, 504.2 GB/s | 0.0113 ms, 652.6 GB/s |

That table is a cache-ranking, and the ranking it gives is not the one that matters: warm says
rows/block 32 is 60 % better than rows/block 4. Cold — each weight replicated 6× so consecutive
touches of the same rows are 84 MiB / 42 MiB apart, far past the ~20 MB iGPU cache — ranks the
same configs within 4 % of each other:

| cold config | gateup | down | per-token MLP (24 layers) |
|---|---|---|---|
| rows/block 4 | 200.0 GB/s (84 %) | 191.0 GB/s (81 %) | 2.686 ms |
| **rows/block 8** | 195.2 GB/s (82 %) | 208.7 GB/s (88 %) | **2.650 ms** ← best |
| rows/block 16 | 189.6 GB/s (80 %) | 211.5 GB/s (89 %) | 2.692 ms |
| rows/block 32 | 192.6 GB/s (81 %) | 207.6 GB/s (88 %) | 2.679 ms |

**Per-token MLP = 2.650 ms = 199.4 GB/s = 84 % of the 237 GB/s probe**, against a 2.230 ms
roofline. Replica 0's output still matches the reference at every config, which is what makes
the 6× replication legal as a cache-defeat rather than a different program.

Repeatability, because it nearly produced a wrong number here: three clean back-to-back runs give
2.686 / 2.672 / 2.663 ms. With **two gate processes on the GPU at once** the same command reports
3.374 ms (66 % of probe) — the cold check is a bandwidth measurement and it cannot share the
memory controller. Every number in this file was taken with the box to itself (`pgrep -f run.py`
before each run); the config ranking above is stable across clean runs, while the identity of the
best config is not (rows/block 4, 8 and 16 each won a clean run, within 0.9 % of each other).
Treat the honest statement as "any of rows/block 4–16, 2.65–2.69 ms" rather than a winner.

The remaining 16 % is not a missed row mapping — `rows/block` does not move it — and is
consistent with `down_proj` issuing only 128 blocks (1024 rows / 8) across 40 CUs, versus
`lm_head_gemv`'s 31040 blocks. Candidate follow-ups, in the order I would try them: keep `x`
packed in shared as BF16 instead of unpacked F32 (halves the LDS read to global ratio, which is
2:1 for `down`), split long rows across two wavefronts, or fold the residual add in. Not
measured, recorded as candidates; `bench/performance-status.md` §5 item 5 owns the follow-up.

Launch cost, for the ledger: 2 launches per layer × 24 = 48/token ≈ 0.089 ms at the 1.85 µs
floor, 3.4 % of this op's time. A split gate/up/down version would pay 72.

## What graduates into `engine/`

- `mlp_gateup` (fused epilogue) and `mlp_down`, `rows/block = 8`, F32 accumulator, `warpSize`
  general.
- The epilogue rounding policy above, as a rule and not an implementation detail.
- `prototype/bf16_ulp.py` as the shared comparator — every future BF16 gate should import it and
  run its `selftest` (see the trap below).
- The scale-aware comparison idea: any future gate that compares BF16 against a reference will
  meet the same cancellation floor and will need the same band, chosen from its own floor probe
  rather than from a guess.

## Traps found while building this

- **A ULP comparator written on a `uint16` view has a dead branch.** The usual
  `np.where(x < 0, -x, x | 0x8000)` never takes the first branch — a `uint16` array is never
  negative — so every negative float collapses to +0.0 and the metric silently reports
  "everything is within 1 ULP" for the whole negative half of the range. Correct:
  `np.where(bits & 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)`. `prototype/bf16_ulp.py` has a
  selftest that prints next-neighbour distances against the torch ground truth; run it before
  trusting any ULP number. **This op shipped that bug** — `ulp_probe.py` had its own copy of the
  comparator, so every ULP figure it printed came from a dead branch. Its numbers survived the fix
  (see above), but only because it compares same-sign neighbours; a probe that compares across
  zero would have produced the opposite conclusion and the gate would have been loosened to match.
- **`hipEventElapsedTime` is 0 below ~1 ms on this iGPU.** Sub-millisecond kernels timed with
  events report zero, which looks like an infinitely fast kernel rather than a broken timer.
- **Pinning `act` means hooking `down_proj`'s *input*, not its output.** The first version read
  the hook's `output` argument and pinned `down_proj(act)` under the name `act`; every
  self-check still passed (`out` reproduced, and `act` reproduced something the reference really
  had computed), and the bug would have gated a kernel against the wrong tensor. It surfaced only
  when a `--mode down` reference run made F3 impossible to satisfy.
- **Replicating a weight to defeat the cache has to alternate the kernels that touch it** (and
  count the same bytes the kernel reports). The first version launched 6 replicas of `gateup`
  back to back, so replicas 1–5 re-read replica 0's out-of-cache-but-still-cached bytes and
  reported 344 GB/s — above the probe, i.e. impossible, and caught only by the rule that says so.
- **Sweeps and negative tests must load their own weights.** Both the config sweep and the
  negative tests inherited the last layer's weight files from the main loop; the sweep would have
  ranked configs against the wrong matrix and the negatives were vacuous. Cheap to check, ugly
  to notice.
