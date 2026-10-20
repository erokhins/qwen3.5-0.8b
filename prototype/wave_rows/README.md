# Wave-body width: 512 is at the optimum, and the residency A12 asked for does not pay (measured 2026-10-20)

Two legs, one question each, both run by the main agent inside one `tools/gpu_lock.sh` hold apiece
(`TAG=r1-wavewidth`, `TAG=r1b-wide-residency`), blocking, zero subagents, `util` 10–92 % at acquire and
discounted per `bench/performance-status.md` §6. Timing only: **no numerics admission was run for any arm
here**, and neither leg ships anything.

## Why this was funded

`bench/performance-status.md` §56 measured the *downward* half of the body-width curve with
`--wave-rows`: at fixed total lanes, halving the body costs **+10.8 %**, quartering **+29 %**,
eighthing **+86 %**. Convex, so a body **bigger** than 512 looked cheap — but `--wave-rows` is capped at
`kMaxWaveRows`, so the interesting direction is not reachable by a flag. It needs three builds of one
source line, which is what `r1_wavewidth.sh` makes.

The same constant is also A12's stated precondition: the wide block emits `2 × ceil(rows/32)` workgroups,
so 512 rows gives 32 on a 40-CU machine and the rig that printed +25.1 % ran at 2 blocks/CU × 40 CUs.
`kMaxWaveRows = 2048` yields `2 × ceil(2048/32) = 128` workgroups — 3.2 per CU asked against an LDS
bound of 2 — i.e. **exactly the rig's 2 blocks/CU = 16 wavefronts/CU, inside the engine**. That is
`r1b_wide_residency.sh`.

## Instrument

`kMaxWaveRows` ∈ {512, 1024, 2048}, three binaries from otherwise identical HEAD, built with
`engine/build.sh`, source restored after each build (the tree is clean; the scripts run the saved
binaries from `tmp/`). Four arms, and the fourth is the one that makes the others readable:

| arm | binary | `--wave-rows` | what it separates |
|---|---|---|---|
| `w512` | 512 | 512 | the shipping configuration |
| `w1024` | 1024 | 1024 | a 1,024-row body |
| `w2048` | 2048 | 2048 | a 2,048-row body |
| `w2048c` | 2048 | **512** | **the cost of the constant alone** — same plan, same launches, same geometry as `w512`; only the scratch the binary allocates is bigger |

Four canaries, each read off the run's own lines before its number is kept: `NUMERICS wave_rows_cap=`
(which binary — the one a stale build cannot fake, since a build agrees with itself), `PREFILL wave_rows=`
(which body it actually ran), and in R1b `NUMERICS attn_fattn_wide=` plus `ATTN_WIDE grid=(…)` — the last
one being the residency claim itself, `(2,1,16)` at 512 rows and `(2,1,64)` at 2,048. Every rep asserted
all of them; none aborted.

## R1 — a bigger body is not cheaper. 512 is at or near the optimum.

Best of 3, `RESULT ms_total`, `--steps 1`, shipping attention. Ranges are `[min, max]` over the reps.

| arm | 4,096 lanes | vs ship | 16,384 lanes | vs ship |
|---|---|---|---|---|
| `w512` (ships) | 505.9 ms `[505.9, 517.9]` | — | 2,763.4 ms `[2763.4, 2808.5]` | — |
| `w1024` | 504.8 ms `[504.8, 511.8]` | −0.20 % | 2,778.1 ms `[2778.1, 2824.9]` | +0.54 % |
| `w2048` | 524.3 ms `[524.3, 614.9]` | **+3.76 %** | 2,840.3 ms `[2840.3, 2855.6]` | **+2.79 %** |
| `w2048c` | 515.3 ms `[515.3, 523.2]` | +1.98 % | 2,788.6 ms `[2788.6, 2861.2]` | +0.90 % |

* **1,024 rows is neutral** and its ranges sit inside the shipping ranges — no win.
* **2,048 rows is a real regression**: every `w2048` rep is slower than every `w512` rep at both widths
  (min 2,840.3 against max 2,808.5 at 16k; 524.3 against 517.9 at 4k). At 4,096 one rep read 614.9 ms,
  which is why `best-of-N` is the statistic here; dropping it does not change the verdict.
* The cost curve is **asymmetric**, and that is the useful shape of the answer: halving the body costs
  +10.8 %, doubling costs +2.8 %. §56's LEG B measured the steep side and extrapolated the flat one; the
  flat side is mildly negative, not positive. **`kMaxWaveRows = 512` is where the shipping form wants to
  stay.**
* `w2048c` is a hint, not a finding, and is recorded as one: the same plan built with a 4× bigger ceiling
  — and therefore a `part` scratch of 1.03 GiB instead of 258 MiB per scratch instance
  (`attn_part_floats` × 4, `engine/kernels_attn.cpp:73` states the bytes) — read **+1.98 %** at 4,096 and
  **+0.90 %** at 16,384, with overlapping ranges at 16k and near-disjoint at 4k. Three reps do not
  establish it. What it does mean is that `kAttnMaxSplit`'s comment — "the ONLY thing the raise costs is
  the scratch named below" — has an unmeasured time side, and the next person who raises one of these
  constants should price the build, not just the bytes.

## R1b — and with that body the wide shape still does not move, so the residency premise is falsified

Best of 3 at 16,384, best of 2 at 32,000. `wide0`/`wide1` = `--attn-fattn-wide 0|1`; the grid column is
the canary, i.e. the actual workgroup count the wide launch submitted.

| arm | grid (16,384 run) | wgs | 16,384 lanes | vs ship | 32,000 lanes | vs ship |
|---|---|---|---|---|---|---|
| `w512/512/wide0` (ships) | — | — | 2,762.9 ms `[2762.9, 2781.3]` | — | 8,055.9 ms | — |
| `w512/512/wide1` | `(2,1,16)` | 32 | 2,737.8 ms `[2737.8, 2762.3]` | **−0.91 %** | 7,907.5 ms | **−1.84 %** |
| `w2048/2048/wide0` | — | — | 2,787.8 ms `[2787.8, 2830.8]` | +0.90 % | 8,050.0 ms | −0.07 % |
| `w2048/2048/wide1` | `(2,1,64)` | **128** | 2,804.8 ms `[2804.8, 2854.1]` | +1.52 % | 7,987.5 ms | −0.85 % |

Read the fourth row against the second, because that is the whole bet: the wide kernel with **four times
the block population — the rig's own 2 blocks/CU and 16 wavefronts/CU, reached inside the engine** — is
**+2.45 % slower at 16,384 and +1.01 % slower at 32,000** than the same kernel at 32 workgroups, and no
better than the shipping form at either width. Residency was not what the shape was missing.

Consequences, in the order they matter:

1. **A12 is closed on all three axes**, each killed by a measurement rather than by taste: coverage
   (it owns every row already, §56), rate at engine geometry (ratio 1.0001 over 31 paired bodies, §56),
   and now rate at the rig's residency (this leg). The gap between the rig's +25.1 % and the engine's
   ~0 % belongs to the rig's *workload* — one kernel, no `attn_prep`, no combine, no §05.6 epilogue, and
   one uniform pass over the triangle — and it does not transfer in any launch shape we can build.
2. **The `kMaxWaveRows` lever is closed for the shipping form**, which also removes the "free
   precondition" argument: the precondition costs +2.8 % of the wave by itself.
3. `--attn-fattn-wide` at the default body is confirmed as a small real win at long context
   (**−0.91 % at 16,384**, ranges just disjoint; **−1.84 % at 32,000**, ranges disjoint) and it remains
   unshippable for the reason `A12-ADMISSION.md` records — the `smoke` cascade cosine — while
   `--attn-split-cap 8` buys −1.14 % of the same thing with no numerics risk (§56).

## What these legs cannot say

No numerics: a wider body is not bit-identical (the library GEMM route is explicitly width-dependent and
band-judged), so R1's +2.8 % is not a numerics verdict and nothing here was admitted. No 32,768-lane row:
a 32,768-token prompt plus one generation step is 32,769 lanes, above the engine's lane ceiling
`MAXSC × kAttnMaxSplit = 32,768`, so the engine refuses it — by design, §39 — and the long row here is
32,000 lanes instead. Two reps at 32,000 and three elsewhere, `best-of-N`, one lock per leg, so a
width-to-width comparison across the two legs is not a paired window; within a leg every arm is paired
and interleaved.

## Files

| file | what it is |
|---|---|
| `r1_wavewidth.sh` | R1: four arms × three builds of one constant, canaries, best-of-3 |
| `r1b_wide_residency.sh` | R1b: shipping vs wide at the default body and at a 2,048-row body, grid canary |
| `evidence/rows_4096.txt`, `evidence/rows_16384.txt` | R1 per-rep `<binary> <wave_rows> <ms_total>` |
| `evidence/wide_16384.txt`, `evidence/wide_32000.txt` | R1b per-rep `<arm> <ms_total>` |

Main-agent-only by the standing rule: a worker may read these and re-run nothing here, since both legs
take timings.
