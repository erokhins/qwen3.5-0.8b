# Route S1 — step-level launch and segment economy (opened 2026-10-21)

**Status: opened by the user, 2026-10-21, in place of adding D2's and D4's prizes.** It replaces nothing
in the frozen route tables; `progress/2026-10-21/decode-attribution-2026-10-21.md` §154-157 stays exactly
as written and is the record of what those routes claimed. This file is the live brief for the one route
they both pointed at.

## 1. Why this route exists instead of D2 + D4

Two routes died the same death on the same day, independently:

- **D2 (mlp)** — `prototype/mlp_insitu/D2-DIAGNOSIS.md` §5: the "1.11 ms in-situ penalty" was a shape
  mismatch (a 2-launch 86,016-row burst against the engine's 48-launch 3,584-row walk). Decomposed:
  `T_geo` **+0.7565 ms** launch shape, `T_serial` **+0.2709 ms** per-segment drain, `T_pipe` +0.0484 ms,
  **~0.06 ms left for memory state**. The fat shape is not harvestable: layer *i*'s `act` precedes its
  `down`, and `down` feeds the residual the next layer reads.
- **D4 (gdn)** — `prototype/gdn_insitu/D4-DIAGNOSIS.md` §7: the recurrence chain is **under half** the
  line (`only_rec` 0.1243 of a 0.2790 shipped-shape row). Merging 18 launches into 1 at identical bytes
  buys **0.12 ms**, and ~0.19-0.21 ms of the engine's 0.63 ms gdn line is per-segment drain.

Both said: the money is in **how many launches and stage-clock segments a step emits**, not in what any
one of them reads. So the prizes must not be added — **-0.80 ms (D2) and -0.23 ms (D4) are one lever
wearing two family names.** S1 is that lever, stated once, with lines set where a step is actually
measured: the **wall**, at the headline width.

## 2. The step's current structure, from today's window (quoted, not re-taken)

`engine/performance-status.md` §1 and `prototype/attn_decode_wide/results.md` §engine window, 4,032 lanes
on the tree at `e02c568`: **11.1750 ms/token** wall (89.5 tok/s), and `decode_run` echoes
`LAUNCH glue=49 proj=48 gdn=72 attn=6 mlp=48 cast=97 other=20 total=340`, `plan_per_step=340`,
`syncs_per_step=305`. Per-family stage readings from the same window: `mlp 3.3628`, `proj 2.7578`,
`attn 1.6121`, `lm_head 2.1685`, `cast 0.7775`, `gdn 0.6458`, `glue 0.5593`, `host_tail 0.7163`,
`copy 0.2141`; the serialized stage `TOTAL` is 14.2057 against a burst of 12.4036 on the base arm.

Two facts that bound this route before any code exists, and they pull in opposite directions:

- **For:** the serialized shape costs 1.80 ms more than the burst shape for identical work, and that gap
  is ~305 syncs = **5.9 us each** — the same constant the ledger bills per segment (5.64-7.2 us). If any
  of that is exposed on the production wall, launch count is worth real milliseconds.
- **Against:** the shipping path is the *pipelined* one. `tools/context_sweep.py` and the stage clock
  both drain per segment by construction, so a large part of `T_serial` may be an **instrument cost of
  measuring**, never paid by a serving engine. D2's `T_pipe` (+0.0484 ms for pipelining 8 tokens) is a
  hint in that direction. If launches are free when nothing drains between them, S1 collapses to deleting
  *work*, which is D5, not a new route.

## 3. Gate zero — the cheap arm that decides whether S1 exists at all

**Measure the wall price of a launch in the pipelined shape, at the engine's own population.** One arm:
replay the step's launch/segment pattern (340 launches / 305 segments, plus a merged variant) with a
negligible kernel, pipelined, burst-timed, and read milliseconds per step. This must run **before** any
engine code is written.

| reading of gate zero | consequence |
|---|---|
| marginal wall cost per launch >= **3 us** at 340 launches | S1 is real: the prize is (launches removable) x (price), and the candidate list below is funded |
| between **1 and 3 us** | S1 is marginal: fund only merges that also delete work, i.e. fold it into D5 |
| below **1 us** | **S1 IS DEAD.** Launches are free when pipelined; every T_geo/T_serial figure in D2 and D4 is an instrument artefact, and the only remaining prize is deleting work (D5). Say so and close the route. |

Rig discipline, non-negotiable, because this is the exact class of thing that produced two false
conclusions this month: bytes-touched verification with guard bands, must-land sets derived from what an
arm actually launches, an empty expectation FAILS rather than passing vacuously, a no-launch arm prints
NO-TIMING and never `0.0001`, every parser aborts on an empty or non-numeric parse, `LC_ALL=C`, one
`tools/gpu_lock.sh` hold acquired and released inside one shell block, and **no millisecond figure from
any worker device run**. The launch count an arm emitted must be printed by the arm, not assumed.

## 4. Pre-registered lines, on the step wall at the headline width

Set now, before the first number. Decode is reported at 4,032 lanes with 128 beside it
(`AGENTS.md`, user 2026-10-21). Baselines are today's paired window: **11.1750 ms/token at 4,032**,
**9.9182 at 128**. Lines are on a *paired* window (base = HEAD build from a scratch worktree, same hold),
never on absolutes from different windows.

| arm at 4,032 lanes | line |
|---|---|
| **WIN** | step wall <= **10.55 ms/token** (-0.62 ms, -5.6 %; 95.0 tok/s) |
| **KILL** | step wall >= **11.00 ms/token** (-0.18 ms) |
| between | neither, reported as neither |

At 128 lanes there is **no line**: D3 established that short widths are one round at either residency and
D2/D4 both came out flat there; a short-context row is reported beside the headline, not instead of it,
and a flat 128 row is a predicted PASS.

## 5. What is in the candidate list, and what is explicitly not

**In**, ordered by (launches removed) x (work removed), all subject to gate zero:

1. **The 97 `cast` launches** — D5's own first candidate: the GEMV input is widened BF16->F32 by a
   separate cast; doing the widen inside the GEMV removes both launches and bytes. Largest single
   population in the step.
2. **The 19 `copy` launches** — D5's second candidate, `copy`'s roll-back folded into the conv ring.
3. **The 49 `glue` launches** — folding the norm+gate epilogues, D5's third candidate.
4. Any legal merge inside a single layer that D2/D4 did not already test. D2 proved cross-layer MLP merges
   are impossible; nobody has proved the 48 is minimal *within* a layer.

**Not in this route**, each for a stated reason: cross-layer MLP merging (forbidden by the residual
dependency, D2 §5); the `--split 18` attention geometry (held separately pending the > `MAXSC` numerics
ruling, and its value dropped when the walk halved); per-family memory-state work (D2 leaves ~0.06 ms,
D4 leaves +0.070 ms of L2 residency — both below this route's resolution); anything prefill, MoE, vision
or a second model.

## 6. Numerics posture

Every candidate here moves the shipping path's arithmetic inputs or their order, so each needs the engine
gate green *before* its timing is quoted — the pattern that worked on the D3 port: numerics first inside
the window, `NO-TIMING` if red, and a `sampled.txt`-style decision check at **both** widths including the
> `MAXSC` streaming shape, because byte-identity on short fixtures is not a witness at width
(`AGENTS.md`). `spell_unroll4` from D4 is the model: bit-identical at equal bytes, so no band is in play
(`AGENTS.md`, numeric-format rule) and any win is pure.

---

## 7. Gate zero's answer (2026-10-21, main agent): the sync half of S1 does not exist in the shipping path

`prototype/launch_price/s1_sweep.sh --run`, blocking, zero subagents, one `tools/gpu_lock.sh` hold, util
55 % at release; fitter `fit_slope.py` printed `FIT ok` on 24 arms, every arm `verify=ok` with
`count_match=1` on its own printed launch and sync counts. Correctness gate `check_s1.sh` was green
before the rig was timed.

**The decisive arm is a single-variable sync sweep at the engine's own operating point** — engine launch
pattern, 340 launches, same grids, the step's real 1461.3 MiB, identical work, one variable:

| syncs | ms/step | marginal |
|---|---|---|
| 1 | **9.4798** | — |
| 305 | 11.0932 | **+1.6134 ms** = 5.29 us per sync |
| 340 | 11.2855 | +0.1923 ms for 35 more = 5.5 us per sync |

Linear to within 4 % across three points, so this is a real, well-measured cost of **5.3 us per sync** —
and it independently reproduces the gap I measured today between the serialized stage total (14.2057) and
the burst step (12.4036): 1.80 ms, against the rig's 1.61 ms for 305 syncs. Two different instruments
agreeing to 10 % on the size of the thing.

**And the shipping path never pays it.** `engine/decode_run.cpp:325` is `bool time_stages = false` and
`:2403` is `if (o.time_stages) { time_stages_report(...); }`; the only per-segment `hipDeviceSynchronize`
in the engine is `engine/stage_clock.cpp:32`, and the one at `engine/decode_step.cpp:423` is a one-time
adapter-prep sync guarded by `sc.adapters_ready`. So `syncs_per_step=305` is a property of the
**instrument**: without `--time-stages` the step is S≈1. That gives gate zero's verdict on the sync half:

- **S1's sync leg is closed.** Every per-segment drain term in this week's route work is the stage clock
  billing for its own ruler: D2's `T_serial` +0.2709 ms, D4's ~0.19-0.21 ms of the gdn line, and the
  ledger's `segments x 7.2 us` subtraction. The ledger's per-family stage rows remain correct *as an
  ordering of families*, but any prize computed as "segments x us_per_segment" is pricing the measurement,
  not the engine. This is the same class as the D2 shape mismatch and the `spantest` no-op: an attribute of
  the instrument read as an attribute of the machine.
- **S1's launch leg is unpriced, not refuted** — and the fitter's headline `3.9473 us/launch -> S1 IS REAL`
  must not be read as funding it, for two named reasons. The slope ladder is measured at body=tiny
  (5.3 MiB, 340 launches, nothing to overlap behind), which is precisely the regime the brief warned
  would overstate; and the merge gain (13.08 us per launch removed) compares 340 launches at
  `grid_max=271` against 48 at `grid_max=1918`, so it moves workgroups-per-launch along with launch count —
  the A12 rule, read workgroups on both sides. The one clean comparison at real bytes and equal grid shape
  is not in the matrix.
- **Therefore the WIN/KILL lines in §4 are withdrawn untested.** They were lines for a mechanism that
  turns out not to be in the shipping path, and leaving them in place would invite someone to chase 0.62 ms
  that has no cause behind it.

**What this leaves, honestly.** The remaining decode prize is *work deletion* on the wall, which is D5's
half of the argument and was never a segment-count story: the 97 `cast` launches, `copy`'s roll-back, the
`glue` epilogues. Plus one unpriced candidate that D2 surfaced and this gate did not settle: a one-token
`down_proj` emits 1,024 rows over 128 blocks on 40 CUs, and D4's `comb_all` vs `split8` gap (0.1839 ->
0.0644 at equal bytes) says there is something in per-launch parallelism. Pricing that needs a launch-count
comparison that holds workgroups-per-launch fixed — a different arm from anything run today.
