# `mlp_gemv` — §03.3 MLP decode path (k = 1) in HIP

Status: **gate PASS** — 192 layer×step cases (all 24 layers × 8 decode steps of `dec8`), plus
the six negative injections, plus the cold bandwidth sweep. Binary: `mlp_gemv` (built by
`build.sh`), gate: `run.py`, floor measurement: `floor_probe.py`, rounding-policy probe:
`ulp_probe.py`. Authoritative run: `run.py --layers all --fixture dec8 --iters 60 --rows-sweep
4,8,16,32 --tile 6 --negative`.

F4's aggregate bands were re-set from measurement since the first version of this file: meanrel
1e-3 → **5e-3** and **ndiff dropped** (see “F4's aggregate bands”). The six negative injections were
re-run after the band moved; all six are still rejected where they are declared.

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
something** (|ref| > 3e-5 of the tensor max, see the next section for why that is the number).
"raw max ULP" includes elements that cancelled to ~1e-6 of the max; the gap between the two
columns is the whole subject of the next section.

The criteria, in words: no element above the cancellation band may exceed 1 ULP (2 for `act`),
≤2 % of elements may differ at all, and the §10.1-class aggregates (cos ≥ 0.9999,
meanrel ≤ 1e-2) must hold. F4 is judged on cos and meanrel only — no per-element rule, and no
ndiff rule either — for the reasons measured below.

## How wide the cancellation band is, and how it got set

The band was `1e-4` because the first version needed *some* number and nothing printed one. It is
now **3e-5**, sized against three measurements — the first two from `floor_probe.py`, the third
printed by `run.py` at the end of every run:

| what | worst >1-ULP offender, as a fraction of that tensor's max |
|---|---|
| reference's own reordered sums, `gate_pre` (192 cases × 3 reorderings) | 1.1e-6 |
| reference's own reordered sums, `out` | 6.1e-6 |
| this kernel, F1/F2/F3, all 192 cases | 8.2e-6 (`up_pre`; gate_pre 8.8e-7, act 9.7e-7, F3 out 2.1e-6) |

A band has to clear all three or it forbids correct arithmetic: 3e-5 clears the worst by 3.7×,
where 1e-4 cleared it by 12×. Not 1e-5 deliberately — that would leave the whole gate sitting
1.2× above the worst element ever measured, so one unlucky cancellation on a new fixture would
read as a kernel bug. Both probes now *report* their margin instead of trusting the constant:
`floor_probe.py` prints band usage (elements inside the band and how many >1-ULP offenders the
band excuses — **7 of 7** for `gate_pre`, **10 of 10** for `out`, i.e. the band exempts exactly
the cancellations and nothing else), and `run.py` prints per output the count of elements inside
the band and the tightest band those results would pass at:

```
    gate_pre                0       24        7   8.8e-07      [gated | raw | in-band | tightest]
    up_pre                  0        8       10   8.2e-06
    act                     0       24       70   9.7e-07
    out(ref-act)            0        2        6   2.1e-06
    out(kernel-act)         0      182        6   -            (aggregate-only: no band to size)
```

### F4's aggregate bands: measured, and then moved

F4 has no per-element rule, so its aggregates *are* the gate. They used to be cos ≥ 0.9999,
meanrel ≤ 1e-3, ndiff ≤ 15 %, and none of those three had a measurement behind it. Nothing in the
repo computed cos / meanrel / ndiff of a perturbed `out` against the pin — which is exactly what F4
compares — so `floor_probe.py` does it, over the same 192 layer × step cases × 8 fixed trials
(1 536 perturbed dots per survey, seed 7) at `--perturb 1` and `--perturb 5`; five flips is what F2
measures as legal per layer-step. Measured worst of a **legal** chain, next to what this kernel
actually produces over the same 192 cases:

| `act` → `out` chain (192 cases × 8 trials = 1 536 dots) | worst cos | worst meanrel | worst ndiff |
|---|---|---|---|
| 1 legal 1-ULP flip | 0.99999661 | 9.12e-4 | 29.1 % |
| 5 legal 1-ULP flips | 0.99999615 | **2.01e-3** | **46.8 %** |
| this kernel's real chain, 192 cases | 0.99999963 | 1.25e-4 | 8.4 % |

**The bands are now cos ≥ 0.9999 and meanrel ≤ 5e-3, and ndiff is gone.** Each is set off that
table, and the sample size sits next to it in §10.1 for the same reason:

- **meanrel ≤ 5e-3** — 2.5× over the 2.01e-3 measured worst. The cushion is that wide because
  2.01e-3 is a *sample* worst from 1 536 perturbations, not an upper bound: a different summation
  order, different layers, or a fixture with more cancellation can land higher, and a band that a
  correct kernel can cross is worse than no band, because the first time it fires someone loosens
  it without measuring. If this band is ever tightened, the thing to change is the survey size, not
  the multiplier.
- **cos ≥ 0.9999** — unchanged; the measured legal worst leaves 26× of slack in 1 − cos, and it is
  the strongest of the three criteria it used to be judged by.
- **ndiff: removed, not widened.** A single legal 1-ULP flip puts 29.1 % of `out` at a different
  BF16 value, five flips put 46.8 %, so the 15 % band would have failed correct kernels — this
  kernel survived it at 8.4 % only because its `act` flips land on benign elements, which is luck,
  not something the band bounded. A band wide enough to admit 47 % (~50 %) states nothing that cos
  and meanrel do not already state, so the criterion was dropped rather than inflated. Its role —
  noticing that the chain moved — is carried by the two that remain.

What the surviving band can and cannot detect is worth stating plainly, because "aggregate-only"
hides it: **F4 is a gross-error net with a floor of ~5e-3 in meanrel.** Four of the six negative
injections put the chain far above that floor (`out(chain)` meanrel against the pin, layer 0 /
`decode0`): `trunc` 1.04e-2, `tail` 9.83e-2, `half` 7.42e-1, `rowswap` 1.37e+0 — so the net does
still catch gross errors. The one that does not is `f32act`, whose chain signature is **4.14e-3,
below the floor**: a 1-ULP epilogue rounding change is precisely the legal-sized perturbation F4
refuses to judge element-wise. `f32act` is rejected at F2 `act` instead — 40.4 % of `act` differs
against a 2 % rule, a 20× margin — which is also where the bug lives, so the gate as a whole loses
nothing; but nobody should claim F4's meanrel is what catches it. The whole gate was re-run after
the band moved (`--layers all --fixture dec8 --no-time --tile 0 --negative`, PASS, all six rejected
where declared and nowhere else).

The alternative formulation — judge `down(kernel act)` against `down(ref act)` under the same down
kernel, cancelling the arithmetic sensitivity instead of gating around it — is still the cleaner
shape and remains open. The bands above make it unnecessary for a kernel that behaves like this one.

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
  by tens of thousands of ULP. Hence cos (≥ 0.9999) and meanrel (≤ 5e-3) there, and nothing else —
  ndiff was the third rule and is gone for the same reason (a legal chain measures 29–47 %; see the
  F4 section above).
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
gate to refuse each **where the bug is and nowhere else**. Five separable observations now take
part, because three were too coarse to localise anything:

| injection | the bug | must be caught at | must NOT move |
|---|---|---|---|
| `half` | accumulate half of K (wrong reduction length) | `gate_pre`, `up_pre`, `act`, `out(down)` | — |
| `tail` | drop the **last word** of K: 8 of 1024 terms, the off-by-one a real kernel ships with | same four | — |
| `trunc` | round toward zero instead of RNE | same four | — |
| `rowswap` | every dot right, each finished row written into its neighbour's slot | same four | — |
| `preswap` | gate staged into the `up` slot of the raw dump: `act` correct, each dump the other's value | `gate_pre`, `up_pre` | `act`, `out(down)`, `out(chain)` |
| `f32act` | SwiGLU entirely in F32 (the literal A-07 reading) | `act`, `out(chain)` | `gate_pre`, `up_pre`, `out(down)` |

`out(down)` is `mlp_down` carrying the injection, fed the **reference** act — F3's isolation, so a
`mlp_down` bug cannot hide behind propagation and an upstream bug cannot fake one; the old version
fed the injected act and so could not tell the two apart. `out(chain)` is the F4 form (injected
act), which is how an epilogue bug reaches the returned value. `preswap` is the interesting one:
it is invisible to any gate that looks only at `act`, and it is the exact reason F1 and F2 are
different cases. `--mode down` skips the two injections that live in the epilogue rather than
reporting them as failures — a run that never launched the buggy kernel cannot catch the bug.

The verdict comes from `judge_neg()`, which asks whether an observation violates the **per-element**
criteria (over-tolerance ULP outside the cancellation band, ndiff > 2 %, cos < 0.9999,
meanrel > 1e-2) rather than F4's looser aggregate bands: the question at `out(chain)` is "did the
bug reach the value the engine returns", not "would F4's band catch it". Those columns are worth
recording, because they say how far each bug sits from the line (`out(chain)` against the pin,
layer 0 / `decode0`):

| injection | over-tol elements | ndiff | cos | meanrel |
|---|---|---|---|---|
| `half` | 1022 / 1024 | 100.0 % | 0.75670431 | 7.42e-1 |
| `tail` | 971 / 1024 | 98.2 % | 0.99502812 | 9.83e-2 |
| `trunc` | 656 / 1024 | 90.8 % | 0.99998553 | 1.04e-2 |
| `rowswap` | 1021 / 1024 | 100.0 % | −0.02176629 | 1.37e+0 |
| `preswap` | 0 | 0.6 % | 0.99999998 | 9.45e-6 |
| `f32act` | 303 / 1024 | 64.2 % | 0.99999113 | 4.14e-3 |
| clean chain (for reference) | 0 | 0.6 % | 0.99999998 | 9.45e-6 |

`preswap` is the `must NOT move` case working: its `out(chain)` row is the clean chain's numbers,
byte for byte, because a staging swap in the raw dump cannot reach a dot. `f32act` is the row to
read carefully: 303 over-tolerance elements prove the epilogue bug does reach `out`, but its
meanrel (4.14e-3) sits **under** F4's own 5e-3 floor and its cos clears 0.9999, so F4 alone would
not have rejected it — F2 `act` (40.4 % of elements differ, against a 2 % rule) is what does. That
is the honest form of "the gate catches `f32act`", and §10.1 says it the same way.

`f32act` is the one a naive "max ULP at any element" rule would have missed and the one that
matters most, because it is the difference between matching HF and matching arithmetic. The
first version of this table was vacuous: the negative runs reused the previous layer's weight
files, so every injection "was caught" everywhere including where it was impossible. Now each
injection declares where it may and may not show up, and the declarations are mode-aware.

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

> **That headline is provisional and due for a clean re-measure.** Two method faults were found in
> the sweep that produced it, both now fixed: it burst `gateup` 30× and then `down` 30×, which is
> not what a token does — gateup's 84 MiB stream is what evicts the down weights and the other way
> round — and it validated only replica 0, the one slice even a kernel that read too little would
> certainly touch. The cold check now runs `--mode mlp`, which **alternates** the two kernels
> inside one burst (one burst iteration = one layer, bytes = both kernels' traffic) and validates
> the **last** replica's `act` and `out`. A first run of the fixed method gave **2.594 ms/token at
> rows/block 8 (203.8 GB/s, 86 % of the probe; last replica: act 1 element, out 0 elements
> differing)**, but the box was not quiet for it — `tools/gpu_lock.sh util` read 93 % when the run
> started and 25 % when it ended, with `flash_serve` resident throughout — so that is a sanity
> check that the fixed method runs and lands near the old number, **not** a replacement for the
> published figure. The headline stays 2.650 ms until someone re-measures on an idle box; the fix
> can only move a number like this down (alternating evicts more, never less), and 2.650 → 2.594 is
> inside the 2.66–2.69 spread three clean runs of the *old* method already showed.

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
measured, recorded as candidates. The frozen ledger's §5 item 5 (progress/2026-10-16/performance-status-2026-10-16.md:240) owned the follow-up; the live `bench/performance-status.md` §4 now carries its answer — at the 528 MB window and the `contig` walk the same kernels reach 98 % of probe, so 84 % was never this box's ceiling for the shape.

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
  The second version kept the burst per-kernel (all `gateup`, then all `down`), which is still a
  different machine from the one a token runs on: `--mode mlp` is the fix. It also checks the
  **last** replica rather than replica 0, because replica 0 is the slice even a short read hits.
- **Sweeps and negative tests must load their own weights.** Both the config sweep and the
  negative tests inherited the last layer's weight files from the main loop; the sweep would have
  ranked configs against the wrong matrix and the negatives were vacuous. Cheap to check, ugly
  to notice.

## P4: the token-batched forms (`mlp_gateup_tokens` / `mlp_down_tokens`)

Decode is one token; a prefill wave is `T` of them (§08.2), and the MLP is the biggest weight
family in the model — 22 MiB per layer against a **2 MiB L2** (measured: `hipDeviceProp` on this
part reads `l2CacheSize = 2 MiB`, `sharedMemPerBlock = 64 KiB`, `warpSize = 32`, 20 CUs). A
per-token launch therefore pays that 22 MiB from DRAM for every token it processes. The batched
kernels load each weight row **once per tile of `TPB` tokens** instead of once per token, which is
the only reason they exist.

The tile is `TPB` token rows of `x` staged in shared memory and an accumulator array per lane:

```cpp
for (int v = lane; v < words; v += ws) {
    uint4 ug = wg[v], uu = wu[v];          // once per row, for the whole tile
    for (int t = 0; t < TPB; ++t) {
        const float* xp = &xs[t][v * 8];
        for (int j = 0; j < 8; ++j) { ag[t] += bf(pg[j]) * xp[j]; au[t] += bf(pu[j]) * xp[j]; }
    }
}
```

Three things worth keeping, because each is a constraint someone could break:

* **The per-output-element order is untouched.** Same `v = lane; v += ws` walk over the same
  `uint4` words, same 8-element inner loop, same `reduce_row` ladder, same three roundings. Only
  the accumulator became an array. So output element `(t, row)` is *obliged* to be the bits the
  k=1 kernel already produced — which is what makes this safe to land at all, and `--tokens` checks
  it rather than assuming it.
* **Shared memory, not registers, is the bound.** The tile is `TPB*IN` floats, so the two shapes
  want different tiles: `TPB=4` (16 KiB) at `IN=1024`, `TPB=2` (28 KiB) at `IN=3584`. Both stay
  under the 48 KiB static limit and leave two blocks resident per CU. The old `NEG_HALF` /
  `NEG_TAIL` tricks still work unchanged because `words` stayed a runtime value and the loop shape
  did not move — no unroll pragma on the word loop, for the same reason (asking the compiler to
  unroll a runtime trip count only earns a `-Wpass-failed` warning; whether it rolls that loop is
  not observable in the result).
* `tokens == 1` keeps the shipped `mlp_gateup`/`mlp_down` kernels and their exact launch shape, so
  decode's ledger number is not this package's to move.

### Gate: `run.py --tokens p64`

Two claims, and the first is the one the engine's margins depend on.

1. **bit-identity vs the k=1 kernel.** For sampled tokens (`0, 1, T/2, T-2, T-1`) the single-token
   kernel's `act` and `out` must BE the batched kernel's row for that token, bit for bit.
2. **the pins**, F1/F2/F3/F4 over the whole `[T, dim]` tile with the rules unchanged, judged **per
   row** (flattening would move the cancellation band's reference max and let a loud token hide a
   quiet one), plus a stage-R on the tile: `bf16(silu(gate_pre)) * up_pre` under the reference's
   three roundings must reproduce the pinned `act` (it does, 0 elements differing; the
   one-rounding reading moves 26 % of it, which is `NEG_F32ACT`'s number).

Measured (`.venv/bin/python prototype/mlp_gemv/run.py --fixture dec8 --layers 0,3 --no-time
--tile 0 --negative --tokens p64`, wall 9 s, `gpu_lock util` 93 % — correctness only, **no timing
taken**; the cold-bandwidth section was suppressed with `--tile 0` on purpose):

| | L00 worst row | L03 worst row | rule |
|---|---|---|---|
| stage R `act` (pin self-consistency) | 0 / 3584 differ | 0 / 3584 | bit-exact |
| F1 `gate_pre` / `up_pre` | 3, 2 / 3584 | 1, 2 / 3584 | ≤ 2 %, none > 1 ULP over the band |
| F2 `act` | 4 / 3584, max 2 ULP | 3 / 3584, max 2 ULP | ≤ 2 %, none > 2 ULP |
| F3 `out` from the reference `act` | 1 / 1024 | 2 / 1024 | ≤ 2 %, none > 1 ULP over the band |
| F4 `out` from the kernel `act` | cos 0.99999995 / meanrel 4.5e-05 | cos 0.99999986 / 8.2e-05 | aggregates only, as always |
| **bit-identity vs k=1** (`act`+`out`, 5 tokens) | **23 040 elements, 0 differ** | **23 040, 0 differ** | exactly 0 |

The six `--negative` injections (`half`, `tail`, `trunc`, `rowswap`, `preswap`, `f32act`) are still
caught where the bug lives and nowhere else, at `rows=1`, unchanged from §"Negative tests" above.

What this does **not** claim: that the batched form is fast. The weight-traffic argument is
arithmetic (22 MiB per layer ÷ `TPB` instead of × 1), the x/L2 traffic is the same shape as the
per-token path's, and whether the GPU turns the first into time is a measurement — and a measurement
of a kernel this wide needs the box to itself, which is bench/performance-status.md §6's to take,
not a worker's.

## Q10: the gate/up weight walk, spelled twice — `--mapping interleave|contig`

§Q9 of `devices/strix-halo-mixed-traffic.md` pinned the MLP's in-step deficit inside this kernel: the
engine's own per-thread weight mapping reads at **0.470** of a contiguous sweep over the same bytes on
the same grid (0.472 with the step's writes, 0.629 at the `down` shape), with transaction coalescing
measured at 100 % on both sides and LDS staging exonerated (`--stage 0` ⇒ `nostage/eng = 1.098`). What
that leaves is the shape of the walk, and the unusual thing about this kernel's walk is that it
interleaves **two** weight streams: one row loop that fetches `wg[v]` and `wu[v]` side by side, so a
warp's live addresses alternate between two separate 7.2 MiB allocations every instruction. Two
contiguous passes over the same bytes is the cheap thing to try, and it is the shape `--mapping contig`
selects.

| mapping | what one warp does with one row | emitted weight loads, `IN=1024`/gfx1151 |
|---|---|---|
| `interleave` — default, exactly what has always shipped | one loop, `words = 128` iterations at `IN=1024` ÷ wave64; each iteration fetches `wg[v]` **and** `wu[v]` | 8 × `global_load_b128` off **two** live address pairs, alternating: `v4,v2,v4,v2,v4,v2,v4,v2` |
| `contig` | the gate row end to end, then the up row end to end — same `v = lane; v < words; v += ws` stride, twice | 8 × `global_load_b128` off **one** live pair, re-based between the two walks: `v17,v17,v17,v17,v17,v17,v17,v17` |

### Why this is the free kind of change, and what had to be proven anyway

Inside one stream nothing moved: same word walk, same 8-element inner loop, same `reduce_row` ladder,
same three SwiGLU roundings, same stores. An output element belongs to exactly one stream — `gate_pre[i]`
comes from `gate`'s row *i* alone — so permuting *when* a warp touches the two streams cannot reach an
element's accumulation order, and the epilogue reads `gate_pre[i]`/`up_pre[i]` after the block barrier,
which does not care which of the two walks wrote them first. That is the argument. It is also exactly the
kind of argument that hides a bug (a mis-set bound in the second walk would look like this and still
"match" if the comparison were sloppy), so both halves are checked rather than trusted:

* **The switch is not inert.** `mlp_gateup` is templated on the mapping (`mlp_gateup<IN, MAP>`,
  `mlp_gateup_tokens<IN, TPB, MAP>`) and the host launches one instantiation or the other, so the two
  mappings are two pieces of machine code. The table above is the witness, taken from the compiler:

  ```bash
  mkdir -p tmp/q10/isa && cd tmp/q10/isa
  hipcc -O3 -Wno-unused-command-line-argument --offload-arch=gfx1151 --save-temps \
        -I"$REPO" -I"$REPO/prototype/mlp_gemv" -c "$REPO/prototype/mlp_gemv/mlp_gemv.hip.cpp" -o x.o
  # then, per instantiation, the address pairs feeding the weight loads, in program order:
  awk '/^_Z10mlp_gateupILi1024ELi0EE/,/^\t\.size/' \
      mlp_gemv.hip-hip-amdgcn-amd-amdhsa-gfx1151.s | grep -o 'global_load_b128.*v\[[0-9]*:[0-9]*\]'
  ```

  Two live bases alternating versus one base used eight times is the structural difference between "both
  streams side by side" and "one stream, then the other", and it is visible without timing anything.
* **The two walks agree bit for bit** — `run.py --mapping-identity`, a stage that runs both mappings (and
  both the k=1 and the batched form) on the same operands and requires **0 differing elements** in `act`,
  in `gate_pre`/`up_pre`, in `out`, and in the batched `act`. Zero is the rule: a mapping change that
  moves one bit is a different kernel, not a faster one. Since the outputs are identical *by design*, the
  identity stage is not the proof that the switch does something — the ISA dump above is; the stage is the
  proof that the switch is free.

### Device evidence (correctness only, no timing taken)

`.venv/bin/python prototype/mlp_gemv/run.py --fixture dec8 --step decode0 --layers all --no-time --tile 0
--mapping-identity` — wall 19.7 s, `gpu_lock util` 93 % at acquire (the serving engine; discounted per
`AGENTS.md`), **no timing taken** (the harness's by-product `ms` lines were ignored, and `--tile 0`
suppressed the cold sweep):

| per layer, all 24 | rule | result |
|---|---|---|
| `act` interleave vs contig | 0 / 3584 differ | 0 at every layer |
| `gate_pre`+`up_pre` interleave vs contig | 0 / 7168 differ | 0 at every layer |
| `out` (chain) interleave vs contig | 0 / 1024 differ | 0 at every layer |
| `act`, batched kernel under each mapping | 0 / 7168 differ | 0 at every layer |
| the shipped gate, judged under each mapping | F1/F2/F3/F4 unchanged | `PASS` |

Same window, same discipline:

```bash
.venv/bin/python prototype/mlp_gemv/run.py --fixture dec8 --no-time --tile 0 --mapping contig --negative
.venv/bin/python prototype/mlp_gemv/run.py --fixture dec8 --no-time --tile 0 --tokens p64 --mapping contig
```

wall 10.6 s and 13.8 s. All six injections still localise to where the bug lives and nowhere else
(`half`/`tail`/`rowswap`/`trunc` → `gate_pre,up_pre,act,out(down),out(chain)`; `preswap` → the two raw
dumps only; `f32act` → `act,out(chain)`), so the contig walk did not smooth over anything the interleaved
walk's negatives could reach. Under `--mapping contig` the batched form is still bit-identical to k=1:
23 040 elements compared per layer, **0 differ**.

`--mode down` refuses `--mapping` rather than ignoring it: `down` is one weight stream, so it has no
mapping to choose, and a flag that silently did nothing on half the shapes would make a null A/B
indistinguishable from a mis-typed one. `run.py` enforces the same rule.

What this does **not** claim: that `contig` is faster. Every number above is a comparison of bits, and the
warm table under §Performance is a cache ranking that this change cannot be judged with. The A/B rows,
their thresholds and their null bands are pre-registered in `engine/results.md` §Q10 and belong to the
main agent's exclusive window (`bench/performance-status.md` §6).
