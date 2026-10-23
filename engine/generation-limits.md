# Why generation costs what it costs — every part of a decode step, with the bound that fixes it

**What this document is.** One self-contained answer to "why can't generation be faster?", per part of a
decode step, on this machine, for `Qwen/Qwen3.5-0.8B` text-only. Every claim names the measurement that
established it and the width it was taken at. Every "cannot be faster" is a bound with arithmetic behind
it, and every "could still be faster" says what is unpriced. Written 2026-10-21 at `c9d3b3f` (which already
includes the cast fold described in §1 and §7); the routes behind it are D2, D4 and S1, and each retired
number is named at its own `results.md`.

**Operating point.** Decode is reported at **4,032 lanes** (4,000 prompt + 32 steps), shorter widths beside
it, per the standing rule. Headline: **10.9357 ms/token = 91.4 tok/s** at 4,032, **9.7855 ms = 102.2 tok/s**
at 128. Prefill: 11,070 tok/s at 4,096. Both from paired windows (`engine/performance-status.md` §1).

## 0. The three facts that set every floor

1. **The machine reads ~237–240 GB/s.** Measured copy 210, mixed-read probes to ~240
   (`devices/strix-halo-memory-bandwidth.md`). A byte on this box has a price; almost everything below is
   a consequence of how many bytes a token must move.
2. **A token must move ~1.50 GB of weights.** Census per step (derived from `engine/state.h` +
   `tools/step_traffic_model.py`, `prototype/launch_price/census.py`, 128-lane context):
   **1532.3 MB read / 24.4 MB written**, at 340 launches and 305 stage-clock segments before the cast fold
   of §7 and **243 / 184** after it. Divided by the
   plateau that is **6.47 ms of unavoidable weight traffic** inside a 10.936 ms step. Everything above
   6.47 ms is the sum of everything that is not a pure weight read.
3. **16-bit quantization is not on the table.** It would roughly halve fact 2, which is why so many rows
   below end at "would need int8/int4, explicitly out of scope" (`AGENTS.md`, numeric-format rule:
   quantization is a band change and is a different argument from a format change).

**Weights are ~4 % of the device pool** (1.5457 GB of text weights), so nothing here is a capacity
problem. Generation is a bandwidth and latency story, end to end.

## 1. Per-family ledger: cost, floor, and what the gap is

Costs are stage-clock family lines at 4,032 lanes from a paired window (see §4 for what a stage line is
worth). **They were taken before the cast fold**, so `cast` still shows the 0.7775 ms it no longer costs and
the others carry their pre-fold values; they summed to an 11.175 ms step and the same step now measures
10.9357 ms. Floors are bytes ÷ 237 GB/s. Efficiency = floor ÷ cost.

| family | launches | reads/step | ms/step at 4k | floor | eff | the bound that fixes it |
|---|---|---|---|---|---|---|
| `mlp` | 48 | 528.8 MiB | **3.3628** | 2.340 ms | 70 % | **not memory state** (only ~0.06 ms there), **not launch count** (3.10 us measured), **not launch shape** (item 2: split-K on the one-row `down_proj` loses at every setting, best gain 0.0000 ms). What is left is the M=1 GEMV shape itself and hipBLASLt's algo choice at 3,584 rows, bounded ~1.17x by `gemm_algo_sweep` |
| `proj` | 48 | 467.6 MiB | **2.7578** | 2.069 ms | 75 % | hipBLASLt heuristic picks per-shape algos; D1 already shipped the packed projection launch (−11 %). Remaining 25 % is library algo choice at M=512, swept and bounded at ~1.17× |
| `lm_head` | 1 | 509.6 MiB | **2.1685** | 2.254 ms | **104 %** | **AT THE WALL, cannot be faster.** 246 GB/s demanded against a ~237–240 plateau — it is reading the 486 MiB tied embedding as fast as this machine can read anything. Only quantization moves it |
| `attn` | 6 | 49.55 MiB (unique KV, 4k) | **1.6121** | 0.209 ms | 13 % | **latency at occupancy**, and the fix already landed: a spelling change took `attn_core` from 38,944 B to 6,176 B reserved LDS, residency 1 → 4 blocks/CU, −46 % on this line (§3). Remaining headroom is large in ratio terms but is a serial per-tile walk at 4,032 lanes with 8 query heads |
| `cast` | **97 → 0** | 0.371 MiB | 0.7775 → **0** | ~0.00 ms | — | **DELETED, 2026-10-21.** Widening BF16→F32 is exact, so all 6 sites fold into their consumers and the step saves **0.3008 ms** — which is also the answer to whether a launch costs anything: **3.10 us per launch**, a production cost, not an instrument artefact |
| `gdn` | 72 | 22.6 r + 20.4 w MiB | 0.6458 | 0.190 ms | 30 % | **not the chain** (the recurrence kernel alone is 0.1243 of a 0.2790 shipped-shape rig row); ~0.07 ms is L2 residency of the 18.9 MiB state, saturating at 32 MiB; 47.4 % of its traffic is **writes**, so it is not a read stream |
| `glue` | 49 | 0.407 MiB | 0.5593 | ~0 | 0 % | pure per-launch cost (49 × 11.16 us), same question as `cast` |
| `host_tail` | 0 | — | 0.7163 | — | — | CPU work per step: one blocking D2H of the whole 993 KB logits row plus §03.4's argmax (`decode_step.cpp:1169`), sampling and bookkeeping. **On the critical path, not hidden**: greedy needs the token id before the next step can be submitted, so this is serial per token. Addressable part is bounded by what §07.4 already measured for fusing argmax (~1 %), not by the whole 0.72 ms — see queue P2 |
| `copy` | 19 | 1.327 MiB r + 1.327 w | 0.2141 | ~0 | 0 % | pure per-launch cost (19 × 10.7 us); candidate work is folding the roll-back into the conv ring |

**Read the table as a budget, not as a wish list.** Three families are effectively finished: `lm_head` at
104 % of the wall, `proj` at 75 % with a swept library bound, and `mlp`, where memory state, launch count and
launch shape have all now been measured and priced out (§2, item 2). One is deleted. Two tails remain: the 68
`glue`/`copy` launches worth ~0.21 ms at the measured 3.10 us each, and `gdn`, whose 30 % of floor is the one
remaining gap with no mechanism named yet.

## 2. Why the two big families are near their floors and cannot be moved much

**`lm_head` — 2.1685 ms, 1 launch, 246 GB/s.** The output projection is a GEMV against the tied 486 MiB
embedding table; for one token there is no reuse to find, every weight byte is touched exactly once, and
it is already streaming faster than the measured copy plateau. There is no kernel change that makes a
single-pass read of 509 MiB faster than the machine reads. **The only lever is fewer bytes** — int8/int4 on
the embedding — which is out of scope by standing decision. Quota: this row is 19.4 % of the step and
essentially non-negotiable.

**`mlp` — 3.3628 ms, and the "in-situ penalty" that turned out to be nothing.** The route was built on
3.388 in-situ against a 2.275 standalone burst. Provenance killed it: the burst row is one layer's tensor
**replicated 24× and concatenated** (`mlp_gemv.hip.cpp:166-180,219`) — 2 launches at 86,016 rows — while
the engine ships 48 launches at 3,584 rows. With a shape-matched arm the published floor reproduces at
+1.0 % and the 1.09 ms gap decomposes as **launch shape 0.7565 + drain 0.2709 (instrument, §4) + pipelining
0.0484 + memory state ~0.06**. The fat shape cannot be harvested: layer *i*'s `act` must exist before its
`down`, and `down` feeds the residual layer *i+1* reads, so 24 layers' MLPs cannot become 2 launches.
That is why the last candidate for `mlp` was parallelism rather than memory state — and parallelism is now
measured too: split-K on the one-row `down_proj` **loses at every setting tested** (best gain 0.0000 ms
against 0.15 ms required, because a split re-stages the activation row 1,792 → 14,336 KiB against 7,168 KiB
of weights). What survives is the M=1 GEMV shape itself and hipBLASLt's algorithm choice at 3,584 rows,
bounded at ~1.17× by `prototype/gemm_algo_sweep` (`mlp_insitu/D2-DIAGNOSIS.md` §5, `wave_shape/ITEM2-RESULT.md` §3).

## 3. Why attention got faster today, and why the short-context line did not

`attn_core` declared 6,176 B of `__shared__` and **reserved 38,944 B**, because LLVM promote-alloca-to-LDS
moves `qreg[8]` and the K-row `uint4` into LDS when the loop bound is the runtime `nd`. At 64 KiB LDS/CU
that is 1 resident block per CU — 8 of 32 wavefronts — and it is why `attn` cost 3.023 ms at 4,032 lanes
against a 0.209 ms unique-KV floor. The fix was a spelling: `for (t=0;t<8;++t) if (t<nd)`, byte-identical
across 126 stage comparisons and 21 cases, residency 1 → 4 blocks/CU, **−46 % on the line, −9.9 % on the
step** (`prototype/attn_decode_wide/results.md`). Bound classification, established by arithmetic not
intuition: traffic was 27.7 % of the plateau (not a wall), issue was 5.9× under its own roof (not a wall),
so it was **latency at occupancy** — a compiler artifact wearing a hardware story.

**Why 128 lanes did not move, and will not.** At 128 lanes the launch is 8 blocks on 40 CUs: one round at
either residency, so neither residency nor population levers can bind, and the unique-KV floor is 0.007 ms
of a 0.496 ms line. Measured after the port: 0.4805 → 0.4649 ms (flat within spread). **The short-context
line is not the tile walk**; it is the launch/prep structure around it, which is a different route.

Also learned the hard way: the `>MAXSC` streaming route still misses the pinned reference's flat floors by
`n_bad` counts at 524/648/1,032/4,104 lanes while cos and meanrel are 3–4 decades inside them — two
instruments (flat §10.1 floors vs the wide fixtures' priced route allowance) disagree, and until that is
ruled the `--split 18` geometry (worth ~0.36 ms) is held, not spent.

## 4. What a stage-clock number is worth — and one it is not worth at all

The stage clock closes a segment per family by **synchronizing** (`engine/stage_clock.cpp:32`). Measured at
the engine's own operating point — engine launch pattern, 340 launches, same grids, the step's real
1461.3 MiB, only the sync count varying: **1 sync 9.4798 ms, 305 syncs 11.0932, 340 syncs 11.2855**, i.e.
**5.29 us per sync**, linear to 4 %. The rig's 1.61 ms for 305 syncs independently reproduces the
stage-total-vs-burst gap (14.2057 vs 12.4036 = 1.80 ms) to within 10 %.

**And the shipping path pays none of it.** `decode_run.cpp:340` defaults `time_stages = false`, `:2496`
engages the clock only under the flag, and `decode_step.cpp:423`'s sync is one-time adapter prep. So:

* Per-family stage rows are correct as an **ordering** and as byte-traffic attribution, and inflated by
  ~5.3 us per segment as a **wall** figure. The `segments × 7.2 us` subtraction used in the attribution is
  the cost of the ruler, not of the engine.
* Every per-segment "drain" prize this week inherited that inflation: `mlp` 0.2709 ms, `gdn` ~0.19–0.21 ms.
  They are withdrawn as prizes, not renegotiated.
* What is **not** settled is the marginal cost of a *launch* in the pipelined shape. The rig's slope says
  3.95 us but was measured at `body=tiny` (nothing to overlap behind), and its merge gain (13.08 us/launch)
  moves workgroups-per-launch along with launch count (grid 271 vs 1918) — a confound, not a result. The
  small-op families suggested the residual was real: `cast` at 97 × 7.78 us minus ~5.3 us of instrument is
  ~3.8 us/launch. **Item (1) settled it by deletion**: folding all 97 `cast` launches out of the step moved
  the wall **−0.3008 ms at 4,032 lanes = 3.10 us per launch**, so the residual is real and the 68 tiny
  launches left in `glue`/`copy` (49 + 19) are a funded mechanism, not a speculation. The *shape* question the fold left
  open — is a launch expensive because of its count or because a one-row launch has no work to hide behind
  — was answered by item (2) and is now closed: the engine sits on the plateau of a plateau-then-cliff curve
  (`prototype/wave_shape/ITEM2-RESULT.md` §2), and the cliff is behind a cross-layer merge the residual
  chain forbids.

## 5. Things that look like headroom and are not

* **"Fewer, fatter launches" — closed, not deferred.** It looks free: fewer launches, more work each. Item 2
  measured the price of the fat end of that curve (7.20 ms at 16 launches of 64 blocks vs 311.37 ms at 1,024
  launches of 1 block, equal bytes and equal total workgroups) and the engine is on the flat side of it; the
  only way to build a fat launch here is a cross-layer merge, which layer *i*'s `act` → its `down` → the
  residual that layer *i+1* reads forbids outright (§2, `wave_shape/ITEM2-RESULT.md` §2).
* **Split-K for the one-row GEMVs**: measured, loses at 2, 4, 7, 8 and 14 ways (`intra1 == ship` at `ndiff 0`,
  every gain negative). It is not a latency problem in that shape.
* **Sub-ms "obvious" wins already taken**: the F32 seam, the packed-cast deletion, the conv coalesced read,
  the looped GDN launch, the projection packed flip (D1, −11 % at 4k), G4 `reg16` GDN wave form, A13 wide
  attention (+39.5 % at 16k prefill). Each has its own record in `progress/`.
* **MTP speculative decoding**: bit-lossless but **slower** on this box (~21 vs ~24 tok/s on torch CPU,
  accept 0.66 at k=1). It is a GPU-path bet, not a win today.
* **Deeper lane splits in attention** (`--split 18`) — **retired twice over.** The ~0.36 ms figure quoted here and in §7 was never a numerics-gated prize: the geometry has never existed on the shipping path (`gate.py` has no `--split`; `decode_run --attn-split-cap` is `min(split, cap)`, so the knob can only make a cut *shallower*). And in the rig that does have the knob, at the wide geometry, **18 is already the optimum**: 0.129 ms/case vs 0.141 at 16, **0.175 at 24 (worse)**, 0.140 at 32, 0.141 at 48. Best possible delta ≈ 0.063 ms/step, and the default sits on it (`progress/2026-10-21/generation-idea-queue-2026-10-21.md` P4).
* **Cross-phase prefetch — measured 2026-10-21, real, and 10× under the bar.** The idea was to stage phase *i+1*'s bytes into the ~32 MiB fast tier while phase *i* runs, since `attn` uses ~31 GB/s of 237 and the host tail leaves the GPU idle. Both the mechanism and its ceiling are now measured (`prototype/prefetch_tier/`): a prefetch makes the use-site read **+0.027 to +0.029 ms faster**, adjudicable wherever adjudicable — but it survives only while *staged + intervening* fits the tier. For the 18 MiB GDN state the engine reading (warmed, then deliberately polluted) is **+0.0278 ms at 7 MiB of intervention and −0.0001 to +0.0008 ms at 21, 43 and 64 MiB**; a 7.875 MiB KV slice still survives 21 MiB (+0.0138). Best engine-shaped extrapolation **+0.0278 ms/step against a pre-registered 0.30 ms bar**. Hardware residency hints are not an alternative: `PersistingL2CacheMaxSize` = 2,097,152 B, byte-identical to `L2CacheSize` (the carve-out is the 2 MiB per-GPC L2, not the tier), `accessPolicyMaxWindowSize` = 0, and an access-policy window is rejected at every size from 1 to 64 MiB.
* **Bigger tiles / WMMA in decode attention**: the decode step is one row, and the wave forms only route
  prefill (`kernels_attn.cpp:627`: `if (rows == 1) attn_core<<<...>>>`). Decode attention is a GEMV-shaped
  read, not a matmul; matrix units do not help a one-row walk.
* **Anything memory-capacity shaped**: weights are 4 % of the pool. Paging, eviction, layer reload,
  offload — all irrelevant to this model at this size, and none of them would move a byte faster than §0 fact 1.

## 6. The honest summary, in three sentences

About **6.5 ms of the 10.936 ms step is weight bytes this machine can only read once at ~237 GB/s**, and
`lm_head` (2.17 ms, 104 % of the wall) plus `proj` (2.76 ms, 75 %) are already at or near that line, so
they cannot be made faster without quantization, which is out of scope. Attention spent months at 3.0 ms
because a compiler artifact cut its occupancy to a quarter and that has now been fixed for −46 % on that
line and −9.9 % on the step. The launch tail is now spent rather than pending: item 1 deleted 97 `cast`
launches at a measured **3.10 us each** (−0.3008 ms) and the copy fold deleted 19 more (−0.036 to −0.065 ms
across two paired windows), leaving 224 launches per step with the 49 `glue` launches priced at 10.9× their
prize to fold. Since then every remaining candidate has been measured and come in short of the bar —
split-K loses at every setting, per-launch shape is a plateau the engine already sits on, vocabulary pruning
survives at 100 % so nothing is prunable, deeper lane splits are worth ~0.063 ms at best, and cross-phase
prefetch is worth **+0.0278 ms/step** where it works at all. **Nothing structural is left above the bar**
short of quantization (out of scope) or multi-request batching (a server scope decision, not a kernel).

## 7. Open, with what would settle each

| open item | the arm that settles it | pre-registered reading |
|---|---|---|
| **(1) the small-op tail** — **RESOLVED 2026-10-21**: all 97 `cast` launches folded into their consumers, `dec8` + `wide4096f` green with the fold on, bit-identical at both widths | done (`c9d3b3f`, `prototype/cast_fold/`) | **WIN: −0.3008 ms at 4,032 = 3.10 us per launch**, above the pre-registered 0.24 ms line → the mechanism is proven, and the 68 launches left in `glue`/`copy` (49 + 19) are worth ~0.21 ms at that rate, which is **below** that WIN line, so it is funded as a mechanism and small as a prize |
| **(2) per-launch parallelism** — **RESOLVED 2026-10-21**: (L,G) matrix at fixed bytes with an iso-population diagonal + a real two-pass split-K on the one-row `down_proj` (`prototype/wave_shape/`) | matrix **NOT ADJUDICABLE** (noise floor 0.3071 ms ≥ 0.30 band; re-take then refused bimodal at 21.8 %) — **split-K KILLED** (every arm loses, best gain 0.0000 vs 0.15 ms required) | shape is a plateau-then-cliff and the engine is on the plateau: 7.20 → 311.37 ms at equal bytes/workgroups only as G falls to 1, i.e. **1.43 MiB per block**, a shape the 24-layer residual chain cannot build. "Fewer, fatter launches" is closed, not deferred |
| **`glue`/`copy` fold** — **RESOLVED 2026-10-21**: 19 of the 68 fold; the 49 `glue` launches cost 394 MB/token of extra reads to ride a consumer, 10.9× their 0.152 ms prize | done (`b2cffa4`, `c118d12`, `775e376`; `prototype/glue_copy_fold/`) | two paired windows both cleared the scaled WIN line (0.10 ms × 19/68 = 0.0279): **−0.0361 and −0.0645 ms**, bit-identical `sampled.txt` and all 86 STATEFP rows at 128 and 4,032 |
| **cross-phase prefetch (P1)** — **RESOLVED 2026-10-21**: three arms per target with an *intervening-traffic* arm, because "is a warm loop fast" was never the question | done (`9109f2f`, `prototype/prefetch_tier/`) | mechanism real (+0.027 to +0.029 ms per staged read), cliff-bounded (18 MiB state survives 7 MiB of intervention, is gone by 21), best **+0.0278 ms/step** vs a 0.30 ms bar → **no engine change funded**; and HIP residency hints cannot address the tier at all (P1c) |
| **> `MAXSC` numerics** — **RESOLVED 2026-10-21, and the question was mis-posed**: on the shipping `--gemm lt` route a priced allowance cannot bind (`cascade_floor` takes `max(cap, 2e-3 / 3 %)`, derived caps sit 31–73× below), so a third instrument has been judging those rows all along; the format also records no **route**, so cross-route pricing reddens `hand` | `prototype/maxsc_allowance/MAXSC-HANDOVER.md`; the two offending allowances withdrawn (`e588807`) | it gated `--split 18`, which turns out to be unbuilt *and* worth ~0.063 ms — see §5. What is left is a gate-integrity fix, not a numerics ruling: give the allowance format a `route` field and refuse a mismatch (queue P11) |

*Corrections log: this document exists partly because eight confident numbers were wrong — the `mlp`
"1.11 ms in-situ penalty" (shape mismatch), `X ≈ 0.64` for attention's walk (real value 2.82), the
per-segment drain prizes (instrument cost), a `--split 1` probe that produced plausible numbers from
an out-of-bounds LDS run, a gate-zero slope that printed "S1 IS REAL (3.95 us/launch)" from a regime with
nothing to overlap behind it (the honest number came from deleting launches instead: 3.10), a noise
floor defined as the largest *absolute* within-arm spread, which let one 311 ms cell make a 7 ms comparison
unadjudicable, a **per-stratum** bimodality cap that no arm on this rig could pass (the warm arm of a
1,152-workgroup single read repeats 2.0–2.5 % apart at REPS=5 *and* at REPS=12 alike, so it discarded
almost every stratum — now each comparison is judged against the spread of the two arms it uses, cap
unmoved), and a `--split 18` prize of 0.36 ms that was both unbuilt and 6× smaller than claimed. Each is recorded at its own `results.md`, and the pattern is consistent: a measurement is
only worth the instrument that took it.*
