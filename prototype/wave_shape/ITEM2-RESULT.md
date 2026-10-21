# Item (2) result: shape and split-K are not the lever — and the rig says so at the edge of its own resolution

Main-agent authored, from the two timed windows taken 2026-10-21 with zero subagents active. Rig
`prototype/wave_shape/` (worker-built); bands pre-registered in `ITEM2-HANDOVER.md` §5 **before** any
timing. Nothing here is averaged across windows, because the pre-registration forbids it and the second
window independently refuses to be fitted.

## 1. What the two windows did

| window | command | outcome |
|---|---|---|
| 1 | `REPS=5 ITERS=3 sweep.sh` → `fit_shape.py --expect 37` | **ADJUDICATION-FAILED** — noise floor 0.3071 ms ≥ WIN band 0.30 ms; pre-reg rule 2 says re-take, not average |
| 2 | `REPS=9 ITERS=3 sweep.sh` (100 s wall, util 89 %/32 %) | **REFUSE bimodal** — `down\|two4\|f32` repeats read 0.0285 vs 0.0347 ms across the two passes (21.8 % > 5 % guard). A launch-bound row on this box is bimodal across windows; `AGENTS.md` records this as a property of the machine, not of the rig |

Two harness defects surfaced on the way and both were fixed by refusing, not by relaxing: `sweep.sh`'s
`CELLS-PRINTED` trailer counted *distinct cells* while `fit_shape.py`'s count check compares *records*, so
a complete 54-record file refused itself until the trailer counted records (`prototype/wave_shape/sweep.sh`,
comment at the trailer). And `down\|ship\|f32` prints `NO-TIMING reason=no_--time_flag` during the pre-timing
smoke gate, which is correct behaviour for a smoke pass and must not be read as a missing arm.

**Verdict on the matrix: NOT ADJUDICABLE at the pre-set band.** I am not moving the band. What that costs
is stated in §4.

## 2. What window 1 still says, because its shape effects are 100–1000× its noise

The **iso-population diagonal** holds total workgroups *and* total bytes constant and moves only the split
between launches and blocks-per-launch. At `pop = 1024`:

| L | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| G | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
| ms | **7.20** | 10.77 | 19.10 | 37.56 | 74.67 | 150.21 | **311.37** |

**43× at equal bytes and equal total workgroups.** That is not launch cost (1008 extra launches × 3.10 µs
= 3.1 ms, not 304 ms) and it is not noise. The mechanism is per-block fetch capacity: at `G=1` a single
block must pull 1461 MiB / 1024 ≈ **1.43 MiB** with its own 256 threads and nothing else in flight, and one
block cannot hold enough outstanding loads to approach the plateau. The curve is a **plateau then a cliff**
(`pop = 4096`: 6.98 / 6.91 / 7.33 ms at L=16/32/64, then 11.03 → 76.85 ms), and the fixed-`G` rows confirm
where the plateau ends: spreads of 0.13–0.45 ms at `G ≥ 32` against 30–270 ms at `G ≤ 4`.

**The engine lives on the plateau and never approaches the cliff.** Its smallest real launches are one-row
norms (1 block) but they read *kilobytes*, not megabytes — the cliff needs a fat-per-block launch, which is
a shape the engine's geometry forbids: the only way to fatten a launch is to merge across layers, and D2
proved that illegal (layer *i*'s `act` must finish before its `down`, and `down` feeds the residual layer
*i+1* reads). So the largest effect this rig found is **not reachable** from the engine, and this is the
answer to "why not make launches fewer and fatter": the bytes-per-launch that would make fatter launches
fast are exactly the bytes a 24-layer residual chain cannot be rearranged to provide.

## 3. Split-K on the one-row `down_proj`: KILL

D2's note that a one-row `down_proj` is **1,024 output rows over 128 blocks** (its `K` is 3,584 — the
worker corrected my brief's conflation of the two) was the candidate. Shipped form reads **0.0179 ms** per
call; every alternative is slower:

| arm | ms | gain | launches | stage_kib | vs ship |
|---|---|---|---|---|---|
| `intra1` (wpr 1, same split) | 0.0179 | −0.0001 | 1 | 1,792 | ndiff 0 |
| `intra2` / `intra4` / `intra8` | 0.0204 / 0.0251 / 0.0246 | −0.0026 / −0.0072 / −0.0068 | 1 | 3,584 / 7,168 / 14,336 | ndiff ≤ 2, ≤ 48 ULP |
| `two2` … `two14` (two-pass) | 0.0228 … 0.0369 | −0.0049 … −0.0190 | 2 | 3,584 … 25,088 | ndiff ≤ 2 |

**Best split-K gain = 0.0000 ms** against a required 0.15 ms, so this is a KILL on its own sub-band even
though the matrix above did not adjudicate. Two reasons, both visible in the same row set: the split
multiplies activation re-staging (`stage_kib` 1,792 → 14,336 KiB against only 7,168 KiB of weights — each
extra block re-stages the activation row), and `intra1 == ship` at `ndiff 0` says the shipped form already
gets the parallelism it can use. At 24 calls/step the *loss* would have been 0.06–0.46 ms; the gain is nil.
Honest caveat on scale: a one-call `down_proj` is 7 MiB, so these micro-arms are L2-warm (0.0179 ms over
7 MiB implies ~391 GB/s), which makes the *absolute* times unrepresentative of the DRAM regime — but the
staging penalty that decides the verdict is independent of that, and every sign agrees in both windows.

## 4. Consequences I will actually carry

1. **`mlp` at 70 % and `gdn` at 30 % of their byte floors are not explained by launch count or launch shape.**
   Count was priced by deletion (3.10 µs/launch, item 1); shape is priced here and lands on the plateau.
   The remaining explanations are per-family and different: for `mlp` the M=1 GEMV shape itself (library algo
   choice at 3,584 rows, and hipBLASLt heuristic bounds of ~1.17× from `gemm_algo_sweep`), for `gdn` the
   per-token serial chain plus the fact that **47.4 % of its traffic is writes**, which is not a read stream
   at all and therefore never belonged in a bandwidth-floor comparison.
2. **"Fewer, fatter launches" is closed as a route**, not deferred. §2 is the reason: the cliff is on the far
   side of a merge the residual chain forbids.
3. **One instrument lesson to keep**: a noise floor defined as the *largest absolute* within-arm spread is set
   by the biggest arm, so a 311 ms pathological cell can make a 7 ms comparison unadjudicable. The fix is to
   stratify the floor by regime (or use relative spread) **before** the next matrix is pre-registered — not to
   widen `--max-spread` after a refusal, which is the move this file exists to make unattractive.
4. **Still funded, still untried**: `glue` + `copy` = 68 launches (was ~116 before the cast fold took its 97)
   at 3.10 µs ≈ **0.21 ms**, on the same mechanism item (1) proved. That is below item (1)'s WIN line and
   above its KILL line, so it is a route with an honest ~2 % ceiling and no more.
