# P1 — cross-phase prefetch into the 32 MiB tier: what CPU can settle, and what it cannot

Route row: `progress/2026-10-21/generation-idea-queue-2026-10-21.md` P1, with sub-rows P1a (GDN state),
P1b (per-layer KV), P1c (does the residency API exist). This file is the arithmetic half of the package;
`P1-HANDOVER.md` is the live status, `sweep.sh` holds the pre-registered WIN/KILL lines, and
`fit_prefetch.py` is the only reader allowed to turn timed rows into a verdict.

**Nothing in this file is measured here.** Every rate, stage cost and penalty is quoted with a `[Qn]` tag
and its source; every byte count is *derived* by `size_targets.py`, which runs
`tools/step_traffic_model.py` and reads `engine/state.h` rather than retyping a shape. No millisecond
figure in this directory is a measurement — the timed matrix belongs to the main agent.

## 0. The quoted inputs

| tag | value | source |
|---|---|---|
| `[Q1]` | plateau read **237 GB/s** (copy 210) | `devices/strix-halo-memory-bandwidth.md` results table |
| `[Q2]` | plateau copy **210 GB/s** | same |
| `[Q3]` | fast tier read at 32 MiB **876.7 GB/s**, cliff to **239.4** by 64 MiB | `devices/strix-halo-memory-bandwidth.md` §"The size–bandwidth curve, measured" |
| `[Q4]` | fast tier **copy** at 32 MiB **325.7 GB/s** | same (the copy column is not the read column) |
| `[Q5]` | tier capacity **32 MiB** = 33 554 432 B | `p1c_query` reading KFD: node1 cache 85, **level 3, size 32768 KiB, line 64 B** |
| `[Q6]` | per-GPC L2 **2 MiB** = 2 097 152 B | `p1c_query`: KFD cache 84 level 2 size 2048 KiB; HIP `attr L2CacheSize` agrees |
| `[Q7]` | step at 4,032 lanes **10.9357 ms** | `engine/generation-limits.md` operating point |
| `[Q8]` | prize bar **0.30 ms** | queue P1 preamble ("the bar both of this week's wins cleared") |
| `[Q9]` | GDN state-coldness penalty **+0.070 ms, saturating at exactly 32 MiB** | `prototype/gdn_insitu/D4-DIAGNOSIS.md` §7, `cold_0MiB -> 32 -> 512 -> 1536` row |
| `[Q10]` | `mlp` gap decomposition, **memory-state term ~0.06 ms** | `engine/generation-limits.md` §2 (`mlp_insitu/D2-DIAGNOSIS.md` §5) |
| `[Q11]` | a launch costs **3.10 µs** in production | `engine/generation-limits.md` §4 (cast-fold deletion) |

Stage costs and byte floors come from `engine/generation-limits.md` §1, which states they were taken
before the cast fold; they are used as quoted.

## 1. P1c: the residency hint is the wrong tool, twice over

`p1c_query` (built here, run on device, **no timing taken**) answers the mechanism question from three
independent sources rather than from a header:

| fact | value | why it decides something |
|---|---|---|
| `attr PersistingL2CacheMaxSize` | **2 097 152 B (2.000 MiB)** | identical to `attr L2CacheSize`: the persisting carve-out *is* the L2, not the tier |
| `prop accessPolicyMaxWindowSize` | **0** | the runtime advertises no window it will accept |
| `hipStreamSetAttribute(accessPolicyWindow)` | **`invalid argument` at 1, 2, 8, 16, 32 and 64 MiB — 0 of 6 accepted** | the setter does not merely size-limit the hint, it refuses it |
| `hipLimit_t` | **no `PersistingL2CacheSize` enumerator**; no `hipCtxResetPersistingL2Cache`; no `hipStreamRemoveAttribute` | CUDA's carve-out idiom has no spelling on this ROCm |
| KFD caches | level 2 = 2048 KiB (line 128 B), **level 3 = 32768 KiB (line 64 B)** | KFD is the only source that names the tier at all; `rocminfo` has no notion of MALL |
| `hipMemPrefetchAsync` on a `hipMalloc` pointer | returns `no error` | accepted, but see the caution below |
| `prop concurrentKernels` | **1** | and the probe below shows it is not the whole story |

**Finding, unhedged: hardware residency hints cannot address the tier this engine would need.** The
persisting capacity is 2.000 MiB against a 32.0 MiB tier and the setter refuses every window size, so
every P1 arm must be **explicit prefetch** — a real load issued early. No gate should be written against
`hipStreamSetAttribute`, `hipDeviceSetCacheConfig` or a `hipDeviceSetLimit` carve-out.

Two notes a later reader would otherwise trip on:

* `hipMemPrefetchAsync` returns `no error` on a `hipMalloc` buffer. Accepted is not performed: nothing in
  this package claims it moves a byte, and AGENTS.md records that managed memory degrades **silently**
  toward ~52 GB/s near ~32 GiB, which is why the engine's weight path is `hipMalloc` + explicit copy.
  If anyone later wants this call as a mechanism, the burden is a witness that it changed a residency
  state, not a success code.
* `concurrentKernels == 1` ("the driver claims it will not run two kernels at once",
  `devices/strix-halo-memory-bandwidth.md`, which itself says "re-check before relying on it either
  way"). **Re-checked**: a bounded spinner sets a flag for exactly the interval it is resident, and a
  probe on a second stream reports whether it saw that flag set. Five trials, five observations of real
  overlap, and a negative control that runs both kernels on one stream and correctly reports 0. A
  second-stream prefetch is therefore implementable, and the device property is not a usable answer to
  that question. All of it is booleans and poll counts; no duration is measured or printed.
  The first draft of this probe released the spinner immediately after launching the probe and flipped
  between runs; that was a race, not a flaky machine, and the protocol now releases only after the probe
  signals completion.

## 2. The idle fetch capacity of each candidate window

Derived by `size_targets.py` from `[Q1]` and the quoted stage lines. "idle" is `cost − floor`; "free
bytes" is that idle time at the plateau rate.

| family | cost ms `[Q]` | floor ms | idle ms | free fetch |
|---|---|---|---|---|
| `mlp` | 3.3628 | 2.340 | 1.0228 | 242.40 MB |
| `proj` | 2.7578 | 2.069 | 0.6888 | 163.25 MB |
| `lm_head` | 2.1685 | 2.254 | **0** | **0 B** — at/over the wall (104 % of floor), nothing can ride along |
| `attn` | 1.6121 | 0.209 | 1.4031 | 332.53 MB |
| `gdn` | 0.6458 | 0.190 | 0.4558 | 108.02 MB |
| `glue` | 0.5593 | ~0 | 0.5593 | 132.55 MB |
| `copy` | 0.2141 | ~0 | 0.2141 | 50.74 MB |
| `host_tail` | 0.7163 | — | 0.7163 | 169.76 MB |

Idle stage time sums to **5.0602 ms of a 10.9357 ms step**. The queue's own figures reproduce
(attn ≈ 330 MB, host_tail ≈ 170 MB), which is the cross-check that this table is reading the same ledger.
**Bandwidth is not P1's constraint**; capacity and residency are, exactly as the queue row says. The row
that matters most is `lm_head`: a fifth of the step is at the wall, so a prefetch riding beside it
displaces useful traffic byte for byte.

## 3. Ceiling per target

`size_targets.py` prints this table; the formulas are

```
ceiling_transfer(bytes) = min(tier_bytes, bytes) * (1/[Q1] - 1/[Q3])
ceiling_latency(family) = cost(family) - floor(family)
```

The first says: if the only thing a prefetch changes is *which tier the bytes come from at the use site*,
the saving is the transfer-rate difference on the bytes that actually fit. The second bounds the part a
transfer model cannot see, which is exposed DRAM latency in a family running below the wall.

| target | bytes (derived) | MiB | fits 32 MiB tier | transfer ceiling | latency ceiling |
|---|---|---|---|---|---|
| GDN state, all 18 layers | 18.87 MB | 18.000 | yes | **0.0581 ms** | 0.4558 ms (`gdn`) |
| GDN state, one layer | 1.05 MB | 1.000 | yes | 0.0032 ms | 0.4558 ms |
| KV, all 6 attn layers @ 4,032 lanes | 49.55 MB | 47.250 | **no** | 0.1033 ms (tier-bounded) | **1.4031 ms** (`attn`) |
| KV, one attn layer @ 4,032 lanes | 8.26 MB | 7.875 | yes | 0.0254 ms | 1.4031 ms |
| one GDN layer's weight block | 43.07 MB | 41.076 | **no** | 0.1033 ms | n/a (`mlp`+`proj`, see below) |
| one attn layer's weight block | 36.71 MB | 35.014 | **no** | 0.1033 ms | n/a |

Three things fall out of the byte column that no prose summary of P1 has said:

1. **No layer's weight block fits the tier.** One GDN layer is 41.08 MiB and one attention layer is
   35.01 MiB against a 32.0 MiB tier, so "prefetch the next layer's weights" cannot stage a layer whole.
   The most it can stage is 32/41 ≈ 78 % of a GDN layer, and the part it does stage has to survive the
   part it did not.
2. **The weight target's prize is already measured, and it is small.** `[Q10]` decomposes the `mlp`
   gap and assigns **~0.06 ms** to memory state. That is the empirical ceiling on what warming a
   bandwidth-bound family's weights buys, and it is not new money for P1 — it was measured in the `mlp`
   route and spent there.
3. **The state target's prize is also already measured.** `[Q9]`'s +0.070 ms cold penalty, saturating at
   exactly the tier size, sits directly beside the 0.0581 ms this transfer model predicts for 18.0 MiB.
   Two instruments, same order: that is the strongest available evidence that the rig's model of the tier
   is the right model, and it also says P1a's realistic prize is ~0.06–0.07 ms, not more.

## 4. Does the staged target survive the traffic that sits in between?

A staged slice survives only if `staged + intervening ≤ 32 MiB`, where "intervening" is the cold traffic
that really runs between the prefetch window and the use site. Derived from the same byte census:

| staged target | staging window | intervening traffic | total | verdict |
|---|---|---|---|---|
| all 6 KV slices (49.55 MB) | layer *l−1*'s MLP | 22.03 MB | 71.58 MB | **evicted by arithmetic** |
| one KV slice (8.26 MB) | layer *l−1*'s MLP | 22.03 MB | 30.29 MB | **fits, with 3.11 MiB to spare** |
| one KV slice (8.26 MB) | layer *l*'s own q/k/v proj | 14.68 MB | 22.94 MB | **fits, with 10.13 MiB to spare** |
| all 18 states (18.87 MB) | its own layer's `in_proj` | 21.04 MB | 39.91 MB | **evicted by arithmetic** |
| one state (1.05 MB) | its own layer's `in_proj` | 21.04 MB | 22.09 MB | fits |
| one layer's weights (43.07 MB) | this layer's MLP | 22.03 MB | 65.11 MB | **evicted by arithmetic** |

This is the sharpest result in the package, and it reshapes P1 rather than confirming it:

* **P1b is the only engine-shaped route that survives on arithmetic**, and only in its per-layer form:
  stage *one* attention layer's KV slice (7.875 MiB) in the window immediately before that layer's
  attention, where the intervening traffic is that layer's own q/k/v projections (14.68 MB). It fits with
  about 9 MiB of slack. Staging all six slices, or staging a slice four layers early, does not fit and
  will be evicted no matter what the prefetch does.
* **P1a's whole-state form does not survive.** Staging the entire 18.0 MiB state and then reading 21 MB of
  `in_proj` weights before using it puts 39.91 MB through a 32 MiB tier. The per-layer form (1 MiB state,
  staged in its own layer) fits — which matches how the `gdn` family actually launches, 18 times per step.
* **The weight target is arithmetically dead**: it does not fit, and its prize was already measured at
  `[Q10]`.

The rig's `--intervene-mib` sweep is the measurement of this boundary: where `cold − polluted` collapses
toward zero is the amount of traffic the tier actually survives, which should land near the arithmetic if
the model is right (and if it does not, the arithmetic is what is wrong).

## 5. Aggregate: can P1 reach the bar without a timing?

| component | ms | status |
|---|---|---|
| GDN state, transfer ceiling | 0.0581 | same order as the measured `[Q9]` +0.070 ms |
| KV, transfer ceiling (tier-bounded) | 0.1033 | unmeasured; the latency upside is larger but unbounded by this model |
| next-layer weights | ~0.06 | already measured as `[Q10]`, **not new money** |
| **new money, transfer model** | **≈ 0.161** | **below the 0.30 ms bar `[Q8]`** |
| upside if a staged slice also removes exposed latency | up to 1.4031 (`attn`) / 0.4558 (`gdn`) | this is the only term that can clear the bar, and only the rig can say where it lands |

So the honest summary before a millisecond is spent:

* On the transfer model alone, **P1 cannot reach the bar** (0.161 ms against 0.30 ms), and half of what it
  would win is already inside the shipped numbers via `[Q9]`/`[Q10]`.
* The queue's 0.2–0.6 ms envelope is reachable **only** through the latency term, because a prefetch that
  hides a DRAM round trip in a family running at 13 % of its floor buys more than the transfer-rate
  difference. That term is real but unpriced, and `attn`'s non-floor time (1.4031 ms) is its ceiling, not
  its estimate.
* Therefore the rig is not being asked "is prefetch fast". It is being asked: **does a staged slice
  survive to the use site** (`cold − polluted`), and **how much intervening traffic kills it**
  (the `--intervene-mib` cliff). If the answer is "no" or "almost none", P1 is dead and the answer is
  cheap — see the KILL lines in `sweep.sh`.

## 6. What the rig adds that this arithmetic cannot

`p1_rig.hip.cpp` runs three arms per stratum, and the arm that wins must win for a residency reason, so the
arms differ only in what sits between the prefetch and the use:

```
cold      evict(P)                                    then USE(T)
warm      evict(P)   PREFETCH(T)                     then USE(T)
polluted  evict(P)   PREFETCH(T)   intervene(Q)      then USE(T)
```

**Only USE(T) is timed.** The evictor, the prefetch and the intervening stream run before the timer starts
and are synced outside it. The first draft timed the whole sequence and compared arms, which cannot work:
a prefetch is an *extra* read — the consumer's read still happens afterwards — so the warm arm moves
strictly more bytes than the cold arm and `cold - warm` is negative whatever the cache does. Timed this
way a prefetch can only win by making the consumer's read come from the tier, which is the claim under
test, and no subtraction of a separately timed setup is involved anywhere (the move that made the D4
attribution's "wall share" unreadable, `prototype/gdn_insitu/D4-DIAGNOSIS.md` §0(i)). The consequence is a
sub-ms timed quantity, 0.03–0.2 ms for the engine's real slices, so the rig prints `samples=`, `reps=`,
`iters=` and a within-row `spread_rel=` on every row, the fitter refuses a row whose own min and max differ
by more than 30 %, and the arms' medians are what get compared.

Because the intervening stream is untimed, `cold - polluted` is a clean residency reading: Q's transfer
cost is not inside it, so if `polluted ≈ cold` the cause is that Q evicted the staged bytes, not that Q
costed time.

Every walk publishes an exact per-block double checksum of the bytes it actually read, the host
re-derives the expected partials by replaying the same index arithmetic over the plan it built the launch
from, guard bands hold a bit pattern no kernel stores, and the region after every target is a 1e30
sentinel so an over-read cannot hide. Four injections are shipped as must-catch cases — wrong base,
over-read past the sentinel, the mixed_bw per-thread-store "reduction", and a wrong content fill — and
`--selftest` runs every case in its own forked child. Current state: `SELFTEST ok cases=16 parsed=16
children_bad=0 caught=5/5 clean_red=0`, with `worst_rel = 0.000e+00` on the clean cases (the device and
host sums agree bit-for-bit, so the witness is exact rather than tolerance-based in practice). The canary
`gate`/`sweep` read alongside the numbers is `pre_rd=-1` on the cold arm (it issued no prefetch),
`pre_rd=1` on warm, `inter_rd=1` on polluted — a stale binary would make the three arms one code path and
that field is what shows it.

## 7. Corrections this file forces on how P1 is written elsewhere

Not edits to other documents — those are the main agent's to make. Statements this package would want
corrected where they appear:

1. The queue's P1b line says "per-layer KV at 4k is 8.3 MiB". In decimal MB that is **7.875 MiB**
   (8 257 536 B); the 8.3 figure is MB, not MiB. The same MB/MiB mixing makes `attn`'s "49.55 MiB"
   actually 47.25 MiB and the state's "18.9 MB" exactly 18.0 MiB. Nothing downstream breaks, but a
   tier-fitting argument is exactly where a 4 % unit error shows up.
2. "`concurrentKernels = 1`" is not an answer to "can a second stream overlap a running kernel" on this
   part; the probe in `p1c_query` says they do overlap, with a serialized control that reads 0.
3. P1's "only ~32 MiB can ever be resident" should say plainly that **no single layer's weight block fits
   in it** (41.08 MiB and 35.01 MiB derived), which retires the weight-staging variant without a device
   run.
