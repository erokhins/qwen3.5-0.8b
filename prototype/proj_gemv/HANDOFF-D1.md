# HANDOFF D1 — decode projections: run the **clock-controlled re-take**

For the main agent. **Blocking, foreground, zero subagents active.** One GPU lock hold, taken and released
by the script. The worker who wrote this rig takes no timing, so every `ms` cell below is yours to fill and
nothing here has been measured by it.

Pre-registered lines, **unchanged**: **WIN** = whole-token projections at **≤ 2.60 ms** per decode step
(≥ 76 % of the GEMV probe); **KILL** = **≥ 3.05 ms** (< 9 % of the step, nothing left to win); between the
lines, report the number and do not decide.

## Where the campaign stands

* **Window 1** (`results.md`, the first-window section) fixed three rig defects: rows/block and
  threads/block were one number, a 5×-slow rep was printed as the result, and the gated set was not the
  timed set. It banked one sub-finding that survives everything since: **packed 3.272 vs unpacked 3.902 at
  identical geometry = −16.2 % from the 48-launch plan alone**.
* **Window 2 — the DVFS window** (`results.md` §D1-W2) was clean on all three fixes and still could not be
  adjudicated. Its three shipped-kernel controls came in **+18.3 / +19.8 / +17.7 %** above their published
  rows while agreeing with one another to 1–2 %, and the arms ran monotonically faster down the table over
  13 s of GPU work. `progress/2026-10-21/performance-status-2026-10-21.md` §36 has this part at
  **819/827/838 MHz idle and 2432/2437/2423 MHz right after a rep**, so a 13 s window is shorter than the
  ramp to boost. Its best arm (`v1-unroll` packed rpb 8 @1024, **2.4845 ms**, 79.4 % of probe) is under the
  WIN line and is **not booked**: same window, same controls 18 % off.

The rig now (a) burns the clock before arm 1 **and again inside every arm**, (b) reads the clock on every row
and refuses to adjudicate a row taken under the floor, (c) runs the arms twice in two different fixed
pseudo-random orders and reports the min across passes, and (d) drives the plan pair from the variant that
can actually hold the winning shape.

* **Window 3 — the clock-controlled window** (`results.md` §D1-W3) is the one that proved the gate works.
  One burn before arm 1 lifted sclk 1218 → 2718 MHz and it decayed to 1749 MHz across the 44-second set, so
  **33 of 34 rows printed `LOWCLK` and the run refused to adjudicate.** One arm landed usable
  (`v1-unroll` packed rpb 8 @512, 2.882023 ms) and the plan pair ran at 1674/1721 MHz (packed 2.883941 vs
  unpacked 3.278635 = **−12.0 %**, marked `LOWCLK`, not booked). Nothing was decided on a moving ruler, which
  was the point. The burn therefore moved **inside** the process, per arm.

**Clock pinning is off the table on this box** — do not retry it. `power_dpm_force_performance_level` reads
`auto` and writing it needs root; `rocm-smi --setperflevel high` goes through sudo and there is no
passwordless sudo here (uid 1000, `sudo -n` fails). The rig *is* the control: burn per arm, read the clock on
every row, refuse rows under the floor, keep the window short.

## The one command

```bash
bash prototype/proj_gemv/sweep_decode.sh 30 3 decode0
```

Everything has a default: `ARM_BURN=1.5` wall-seconds of in-process burn immediately before each arm's reps
· `BURN=5` the window-opening burn · `PASSES=2` · `CLK_MIN=2300` MHz · `CLK_NORM=2400` the divisor printed
on the normalised column · `CLK_TOOL=rocm-smi` · `SEED=d1-interleave-2` (the orders are printed and recorded
either way) · `WARM=2` · `ITERS=30` · `REPS=3` · `OUT=/tmp/proj_gemv/sweep`.

**Budget the hold at ~2 minutes.** The timed set is ~95–110 s: the previous window's ~44 s plus 34 rows ×
1.5 s of per-arm burn, which is the load-bearing addition. If you need it shorter, `ARM_BURN=1.0` saves ~15 s
and `PASSES=1` halves it — but `ARM_BURN=0` puts you straight back in window 3's situation, where almost
every row is unusable.

Two runs that are not measurements, and when to use them:

```bash
NO_TIMING=1 bash prototype/proj_gemv/sweep_decode.sh   # phases 0-1: plans + admission, ~10 s, nothing timed
bash prototype/proj_gemv/sweep_decode.sh selftest      # canary, VOID, clock parse, pooling, no GPU, no lock
```

And a full dry run of the timed path with **no GPU at all**, against a stub that prints canned `RESULT`
lines (`BIN=` is the only override needed):

```bash
BIN=$PWD/prototype/proj_gemv/stub_result.sh CLK_TOOL=$PWD/prototype/proj_gemv/stub_result.sh \
  SKIP_GATE=1 OUT=/tmp/proj_gemv/dryrun bash prototype/proj_gemv/sweep_decode.sh 1 1 decode0
STUB_CLK=800 \
BIN=$PWD/prototype/proj_gemv/stub_result.sh CLK_TOOL=$PWD/prototype/proj_gemv/stub_result.sh \
  SKIP_GATE=1 OUT=/tmp/proj_gemv/dryrun_lowclk bash prototype/proj_gemv/sweep_decode.sh 1 1 decode0
```

The second of those is the LOWCLK rehearsal: every row under the floor, every point
`NO-USABLE-ROW`, phase 3 skipped, non-zero exit. It exits non-zero **on purpose** — the stub's numbers are
fixed arithmetic, not measurements, so its controls miss by design.

## Read it in this order

1. **The burn and the clock edges.** The window-opening burn prints
   `RESULT mode=burn … plans=N launches=M wall_s=5.00x` — and no millisecond figure, deliberately, because a
   burn number is not a measurement. Every row's note carries `burn <s>/<plans> plans`: the per-arm burn
   inside that arm's own process, and the sweep **fails the row** if the launcher did not actually burn for
   the requested time. The log records sclk before the opening burn, after it, and at the end.
2. **`GATED 15 point(s)` … and `gated 15, timed 15 distinct points, N==M: yes`.** Otherwise nothing here
   is adjudicable.
3. **`clk` on every row; `LOWCLK` for anything under `CLK_MIN`.** The adjudicating read is the one taken
   **after** the burst — with the burn inside the process, the before-read is the decayed clock the burn was
   about to lift, and judging on it would mark nearly every row unusable by construction. Both reads are in
   the row's note, because the gap between them is how far the queue decays between arms. A `LOWCLK` row
   keeps its millisecond figure for the ramp analysis and is excluded from every pool, minimum and verdict.
   `CLK-UNKNOWN` means the probe did not parse; a window with zero parsed reads fails loudly at the end.
   **If most rows are still `LOWCLK` after the per-arm burn, that is the answer, not a rig fault**: it means
   this part cannot hold boost through a ~100 ms burst regardless of the load in front of it, and then any
   `proj` number has to be quoted as a clock-paired statement and the WIN line becomes a band question for
   the user. Say which happened; do not quietly substitute the normalised column.
4. **`NORM` / `NORMMIN`, and what they are for.** They are `ms × clk / CLK_NORM` with the divisor printed in
   the header (`NORM@2400`), so a row at 1750 MHz and a row at 2450 MHz can be read against each other and a
   cross-window comparison has a defensible form. They are a **reading** column: the WIN/KILL call is made on
   raw `ms` from a row whose clock passed the floor, never on a normalised figure. `LOWCLK` rows are excluded
   from `NORMMIN` exactly as from `MIN`.
5. **The two ramp verdicts.** *Pass over pass*: if pass 2 is still ≥ 2 % faster, the clock was climbing
   across the whole set and only the min is quotable. *Position (paired)*: each arm's residual off its own
   two-pass mean, correlated with how far its position moved between the two orders — `r ≤ -0.30` means late
   positions are faster **within the same arm**, i.e. the ramp survived interleaving and the ranking is
   clock-contaminated: raise `BURN`, keep the arm list, re-take. Both quiet ⇒ the ranking is not the clock.
6. **A1–A3, the shipped-kernel controls, judged on the pooled min at ±3 %** (`CTRL` column):

   | tag | arm | threads the published sweep actually used | published ms | your min across passes | clk MHz |
   |---|---|---|---|---|---|
   | A1 | `v0-baseline` packed rpb 8 | 256 (its own `rpb*32`) | 2.9895 | *(fill in)* | *(fill in)* |
   | A2 | `v0-baseline` packed rpb 16 | 512 | 2.946 | *(fill in)* | *(fill in)* |
   | A3 | `v0-baseline` packed rpb 32 | 1024 | 2.785 | *(fill in)* | *(fill in)* |

   Those thread counts are not a guess: every revision of the launcher from `5d6405f` to `HEAD` computes
   `threads = rpb * 32`, so those are the only geometries that can serve as controls. If any control misses
   ±3 % on the pooled min, the window says nothing about the variants — report the miss and stop.
7. **B arms**: `v1-unroll` and `v5-vx`, packed, rows/block {8,16,32} × threads {512,1024}. The point is the
   shape, and `MIN` is the min across the two passes over usable rows. Against the unchanged lines: under
   2.60 is WIN, over 3.05 is KILL, between is reported and not decided. (`A4`/`A5` are the report-only rows
   from the previous spec and now duplicate two `B-v1-t512` geometries exactly; they cost two extra uploads
   per pass and print identical numbers. They stay because the spec listed them — drop them from `ARMS` if
   you want the ~10 s back.)
8. **Phase 3, the plan pair** — printed as `PLAN PAIR: packed … vs unpacked … = ±x %`. It runs at the
   winning search arm's shape with that arm's own variant, both bit-identical to `v0`, so it isolates the
   launch plan. This is the sub-finding to carry into the engine whatever the lines say: it is a plan change,
   not an arithmetic change.

## Things not to do

* Do not move the WIN/KILL lines, add arms, or widen a band to reach a verdict.
* Do not read `v2-lpw16`, `v3-lpw8`, `v4-lpw4`, `v6-lpw8-vx`: window 1 measured them at 25–47 % of probe and
   they are refuted.
* Do not chase split-K: it is dominated, because rows/block already buys block count and shortening the
   per-thread dependent chain is the wrong direction for a latency-bound one-row GEMV.
* hipBLASLt at `M=1` is not a rig gap. It is blocked on a numerics-authority decision about moving decode off
  bit-identity-by-seam into a reassociation band, and needs the user's words.
* Do not book the two window-2 numbers (2.4845 ms best arm, −24.2 % vs the same window's shipping best) as
  a win. The gap is an upper bound: part of it is the window getting faster while the table ran.

## If it goes wrong

| symptom | meaning | action |
|---|---|---|
| `NO-TIMING: admission printed no point count` | gate output shape changed | fix the parser; never skip the gate |
| `gated 15, timed 12 … N==M: NO` | a point did not run | fix that point and re-take |
| `RUN-FAILED` on an arm | launcher exited (window 2: `v0` + non-default `--threads`) | the row names the variant and the rc |
| `VOID` on a point | max/min > 2× after warming | quote `ms_min..ms_max`, do not book it |
| `LOWCLK` on most rows | the burn did not hold the clock up | raise `ARM_BURN` (and `BURN`), same arm list, re-take |
| `CANARY-FAIL … no per-arm burn` | the launcher ignored `--preburn` | the row is not a measurement; check `preburn_s` in `$OUT/burst_*.log` |
| `clock reads: 0 parsed` | `rocm-smi` output moved | fix `parse_sclk`; `selftest` re-runs the same parse |
| control off ±3 % on the pooled min | window drift, not a slow kernel | report the window as un-adjudicable |
| position verdict fires | ramp survived interleaving | raise `BURN`, re-take; do not book the ranking |

Artifacts left in `$OUT`: `gate_grid.txt`, `plan_token_{packed,unpacked,mixed}.txt`, `arm_order_p1.txt`,
`arm_order_p2.txt`, `pass_rows.tsv` (one line per timed row, `|`-separated), `summary.tsv`, `burn.log`, and
one `burst_p<pass>_<tag>_r<rpb>_t<threads>.log` per row. The `pt` field each row echoes is the launcher's own
self-verdict; `verdict()` in the script recomputes it from `ms_min`/`ms_max` rather than trusting it, so a
stale binary cannot mark itself `ok`.
