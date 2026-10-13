# Strix Halo mixed read/write bandwidth — is 237 GB/s even the right ceiling for a decode step?

Q8. Framework Desktop, Radeon 8060S iGPU (`gfx1151`), 128 GB LPDDR5x-8000. Read this with
`devices/strix-halo-memory-bandwidth.md` (the pure-read roofline this file exists to question) and
`devices/strix-halo-unified-memory.md` (the page-warmth trap, which is why cold rows here are their
own rows and never a mean).

## The question

Every roofline in this repo divides a byte count by **237 GB/s**, which is what
`tools/bw_probe_hip.cpp` gets when it streams a 1 GiB buffer with a grid-stride **read**. But a
decode step is not a read: it streams ~1.5 GB of weights while writing activations, 115 cast
outputs, the KV append and the GDN state. Q7 killed the fresh-window story for the ~3-6 ms residue
between the step and the roofline (its M1 row: the same MLP kernels at a 528 MB cold window run at
93 % of the probe in isolation and 52 % inside the step), which left exactly one traffic-shaped
candidate standing:

> **the step's write traffic is taxing its read traffic, so 237 GB/s is the wrong denominator.**

This file settles it. Two instruments, both CPU-verifiable before any GPU is touched:

| piece | what it does |
|---|---|
| `tools/mixed_bw.cpp` | achieved bandwidth as a function of (read:write mix) × (footprint/access pattern), one machine-readable `MIXEDBW` line per run |
| `tools/step_traffic_model.py` | the step's real traffic budget from `engine/state.h`, the ceiling that implies, and the byte-reduction ladder |

`tools/mixed_bw.cpp` is a timing tool: it is the main agent's to run, under `tools/gpu_lock.sh`,
blocking. `--dry-run` and `--help` touch no device and are the CPU-side verification; `--selftest-geom`
(CPU) and `--selftest` (untimed device cases, ~2 s wall for all 20) are correctness gates that print
no ms and no GB/s, and those two are what a worker is allowed to run.

## What one token actually moves (computed, not measured)

From `engine/state.h` shapes via `python3 tools/step_traffic_model.py`, n = 1, 128 lanes:

| item | direction | MB/token | note |
|---|---|---|---|
| MLP weights (24 layers) | read | 528.48 | 33.9 % of everything |
| `lm_head` row | read | 508.56 | 32.7 %, one row of the tied 248320×1024 BF16 table |
| projection weights (24 layers) | read | 466.75 | 29.9 % |
| GDN recurrent state | read | 18.87 | 18 layers × [16][128][128] F32 |
| GDN recurrent state | **write** | 18.87 | `rec_kernel` writes back what it read |
| conv window (read + `conv_new` + roll-back copy) | both | 5.32 | 18 layers, F32 |
| `conv1d` weight, read as F32 | read | 1.77 | the step pre-widens BF16 `[6144,1,4]` into `g_conv_w` |
| casts, 115 launches | both | 1.78 | 0.59 read BF16 + 1.19 written F32 |
| logits | write | 0.99 | + 0.99 read back by the D2H copy |
| KV read at 128 lanes | read | 1.57 | 12 KiB/lane-token; this is the only row that grows |
| everything else (tiles, glue, norms, stage buffers) | both | ~2.5 | |
| **total read** | | **1532.5** | |
| **total write** | | **24.85** | |

**The step's own ratio is 61.7 : 1, not 20 : 1.** Writes are **1.60 %** of the bytes the step moves.
That single fact does most of the work here, and it is arithmetic from the checkpoint, not an
opinion:

* at the byte-weighted (no-cliff) model — `1/BW = f_read/237 + f_write/BW_write` — even charging
  writes a stingy 105 GB/s gives a ceiling of **6.70 ms/token**, against reads-only at 6.47 ms;
* the engine costs **12.399 ms/token** (`bench/performance-status.md`, Q6.1 stage medians at 128
  lanes), so it sits at **~53 % of the ceiling** no matter which of those two denominators is used;
* for write traffic to eat the residue on its own, those 24.85 MB would have to cost **5.9 ms**,
  i.e. **4.2 GB/s** — a write would have to be **~57× more expensive than a read**. No memory system
  does that to a 64 B-granular write stream, and the already-measured 1:1 copy rate (203–210 GB/s,
  `devices/strix-halo-memory-bandwidth.md`) is a direct refutation: if writes were that expensive,
  copy traffic could not run at 88 % of the read rate.

So the question the probe answers is not "does a mix cost something" (it costs ~1 %, by byte
weighting) but **"is there a cliff"** — some write share or access pattern where this box's
controller falls off. The one number the probe must answer directly:

> at 62 : 1 and the step's geometry, does the box sustain **≥ 200 GB/s** (story dead, 237 stays the
> denominator) or **≈ 113 GB/s** (story alive: 12.4 ms would then be 90 % of the real ceiling)?

## The probe

`tools/mixed_bw.cpp`, built with `tools/build.sh` (`-O3 -Wl,-rpath,/opt/rocm/lib`). One line per
run; `ratio_act` is the ratio it *actually* moved, because a silently rounded ratio is how a mix
probe lies.

| mode | shape | what it can prove |
|---|---|---|
| `read` / `write` | pure stream over `--footprint-mib` | the two ends of the curve; `write` has never been measured on this box |
| `mix` | rrd reads per rwr writes **inside one kernel**, so both streams are in flight together | whether mixed traffic has a cliff at all |
| `mixlaunch` | nbuf read launches with proportional write launches interleaved on one stream | the engine's own shape: 424 serialized launches, 48 × 11.01 MB |
| `wthenr` | a write pass immediately followed by a read pass, timed as a pair **and** separately | whether reading while dirty lines drain costs extra (the step's order of business) |

Geometry is `--nbuf N --buf-mib B` (the MLP's 48 × 11.01 MB walked once) vs one big region;
`--pages cold|warm|both` keeps the first-GPU-pass-over-fresh-pages effect (~35 GB/s, 7× slow) in its
own row instead of averaging it into a warm number.

Three self-checks, because "a bandwidth number above ~240 GB/s is evidence of skipped work" is a
repo rule and a probe that skips work silently is the worst possible instrument for this question.
Every read pass carries a byte account: the source buffers are filled with 1.0f, each block reduces
its per-thread partials in **double** (exact, integer-valued) and the total must equal
`4.0 × float4s-read`; a short or skipped read is `checksum=FAIL_reads_short` and a non-zero exit.
The write side is verified by an untimed read-back against an arena pre-filled with a value the write
stream never stores, so a missing write is visible instead of merely unwitnessed
(`checksum=FAIL_writes_missing`). `checksum=ok` is the only passing value: a row prints
`ok_writes_aliased` when its write side cannot be proven (`wthenr --overlap 1` reads back the lines it
just dirtied) and `unverifiable_cold_pages` when proving it would have warmed the pages being
measured — and in the cold case the process exits non-zero too, because rule 11 says no conclusion may
be drawn from such a row. A cold **read** row additionally prints `fresh_zero_sum=1`, the one freshness
witness cold pages allow (the bytes read were all zero); a non-zero sum means the pages had been used
before and the row must be thrown out. A cold **write** row overwrites that evidence, so it prints
`fresh_alloc_untouched=1` (structural: this process issued no access to those pages before the timed
launch) rather than inventing a measurement. Any row above `--suspect-gbs` (default 240) carries
`suspect=1` with a note on stderr (`--strict` makes it fail). `--selftest` is the same accounts turned
into a gate: one child process per configuration, a poison guard band around every region, one untimed
pass, and four injected bugs, each of which must be caught by exactly the account that owns it — a read
one round short and a skipped final write (caught by the read account and the write account
respectively, and by neither the other), and one float4 straying over each edge, which only the guard
bands can see and which is their negative control.

`mixlaunch` serializes its launches (one stream: the engine's own shape); `mix` overlaps them in
one kernel. Those are two different questions and the rows carry `--mode` so they are never
compared as if they were one experiment.

## Pre-registered sweep, and what is expected — written before the run

Expectations are stated here first, with the number, so the result is a verdict and not a
re-reading of the evidence. All commands run under the lock, blocking, with **zero subagents
active**; `util` is recorded and then discounted (identical clean runs read 24–91 %).

```bash
tools/build.sh mixed_bw                                  # -O3 + rpath, nothing else
tools/gpu_lock.sh status                                # must show no holder
tools/gpu_lock.sh acquire q8-mixedbw
tools/gpu_lock.sh util                                  # record it for the ledger
mkdir -p tmp/q8
# Every MIXEDBW line goes to stdout, so tee the whole sweep into the log the pricing step reads:
{

# 0a. the geometry self-check — CPU only, touches no device, run it anywhere
tools/build/mixed_bw --selftest-geom

# 0b. correctness gate for the tool itself — correctness only, produces no bandwidth number
tools/build/mixed_bw --selftest

# A. the two ends of the curve, one big warm region
tools/build/mixed_bw --mode read  --footprint-mib 528 --pages warm --iters 8 --tag A_read_512m
tools/build/mixed_bw --mode write --footprint-mib 528 --pages warm --iters 8 --tag A_write_512m

# B. the ratio curve in one kernel, warm, 528 MiB footprint
for r in 1 4 10 20 32 62 128; do
    tools/build/mixed_bw --mode mix --ratio $r --footprint-mib 528 --pages warm --iters 8 \
        --tag B_mix_${r}
done

# C. geometry: the same curve over the MLP's own shape — 48 separate buffers walked once per iteration
tools/build/mixed_bw --mode read --nbuf 48 --buf-mib 11.01 --pages warm --iters 8 --tag C_read_48x11
for r in 1 20 62 128; do
    tools/build/mixed_bw --mode mixlaunch --ratio $r --nbuf 48 --buf-mib 11.01 --pages warm \
        --iters 8 --tag C_mixlaunch_${r}
done

# D. the write arena: one-shot writes vs the ~2 MB arena the engine really dirties
tools/build/mixed_bw --mode mix --ratio 62 --footprint-mib 528 --pages warm --iters 8 --tag D_62_oneshot
tools/build/mixed_bw --mode mix --ratio 62 --footprint-mib 528 --wr-arena-mib 2 --pages warm \
    --iters 8 --tag D_62_arena2m
tools/build/mixed_bw --mode mix --ratio 20 --footprint-mib 528 --wr-arena-mib 2 --pages warm \
    --iters 8 --tag D_20_arena2m

# E. writes immediately before reads, which is the step's order of business
tools/build/mixed_bw --mode wthenr --footprint-mib 512 --pages warm --iters 8 --tag E_same
tools/build/mixed_bw --mode wthenr --overlap 0 --ratio 62 --footprint-mib 528 --pages warm \
    --iters 8 --tag E_62_async
tools/build/mixed_bw --mode wthenr --overlap 0 --ratio 62 --footprint-mib 528 --pair-sync 1 \
    --pages warm --iters 8 --tag E_62_synced

# F. cold pages: their own rows, documenting the trap in this probe's own numbers
tools/build/mixed_bw --mode read --footprint-mib 528 --pages cold --tag F_cold_read
tools/build/mixed_bw --mode mix --ratio 62 --footprint-mib 528 --pages cold --tag F_cold_mix62
tools/build/mixed_bw --mode write --footprint-mib 528 --pages cold --tag F_cold_write

tools/gpu_lock.sh release q8-mixedbw
} 2> tmp/q8/mixed_bw.err | tee tmp/q8/mixed_bw.log

# and finally, with the log in hand:
tools/step_traffic_model.py --probe-file tmp/q8/mixed_bw.log --lanes 128
```

The stderr side (device banner, `NOTE:` lines for any suspect row) belongs in
`tmp/q8/mixed_bw.err`, which is why the pricing step reads the stdout log only: `MIXEDBW` is the
parseable prefix and a suspect row must stay visible, not be silently consumed.

### Expected, before the run

| # | row | expected | if instead |
|---|---|---|---|
| 1 | `A_read_512m` warm | **225–240 GB/s**, `checksum=ok`, `suspect=0` | a miss here means the instrument is wrong, not the box: re-check before believing any other row |
| 2 | `A_write_512m` warm | **≥ 200 GB/s**. Pure write has no committed number on this box; 1:1 copy is 203–210, and a write rate far below that would itself be the finding | < 100 GB/s: the write story is alive on its own terms and the rest of the sweep matters much more |
| 3 | `B_mix_*` ratio curve | **monotone** toward `read`, and flat: ≥ 225 GB/s at every ratio ≥ 10:1; ~210 at 1:1 (the committed copy number) | any row ≥ 30 % below its neighbours is a cliff, and the ceiling must be restated at that ratio |
| 4 | the decisive ratio 62:1 | **≈ 230 GB/s**, because writes are 1.6 % of bytes. Byte-weighting predicts 236; the expectation is deliberately a bit worse than the model | ≤ 130 GB/s at 62:1 ⇒ mixed traffic is the missing overhead and 237 was the wrong denominator |
| 5 | `C_read_48x11` vs `A_read_512m` | within **±5 %** — Q7's M1 already showed a 528 MB window re-walked runs at 93 % of the probe, so "walked once in 48 buffers" should not be much worse warm | > 15 % slower: the many-buffer walk is its own effect and belongs in the ledger separately |
| 6 | `C_mixlaunch_62` | near `C_read_48x11`, ≥ 95 % of it | a large gap is the serialized-launch story, which is a different finding and must be named as such |
| 7 | `D_62_arena2m` vs `D_62_oneshot` | **faster or equal** — a small arena dirtied 260 times is cache-resident and never reaches DRAM, which is why the engine's writes are cheaper than 1.6 % would suggest. `fit=write_arena_cycled` on both | slower: the dirty-line eviction path costs something, and `wthenr` decides whether that is it |
| 8 | `E_same` (`wthenr` overlap) | read-after-write on the same lines: pair ≈ 1.9× the write-alone time, `read_dirty_penalty_pct` within ±10 % of zero | a large positive penalty is write-back pressure, the mechanism the traffic-mix story actually needs |
| 9 | `E_62_synced` vs `E_62_async` | within a few %: a sync between the passes cannot buy much if writes are cheap | a big difference says the queue drain, not the bytes, is what costs |
| 10 | every `--pages cold` row | **~35 GB/s** (`devices/strix-halo-unified-memory.md`), `fresh_zero_sum=1` | `fresh_zero_sum=0` means the pages were not fresh: discard the row |
| 11 | every `checksum` | `ok` | `FAIL` means the probe skipped work: no conclusion may be drawn from that run |
| 12 | `suspect` | `0` everywhere at 528 MiB (it dwarfs the 18 MiB cache) | `suspect=1` at this footprint is the >240 GB/s trap, not a new record |

### Row corrections after the first attempt

The first attempt (`q8-mixedbw`, 13 of 23 rows printed, util 92 %→25 %) produced **no usable
number**: every warm row failed its checksum, and every `mix` row with ratio > 1 faulted. Both were
instrument bugs, not box behaviour, and both are fixed in `tools/mixed_bw.cpp`:

1. `mix_k`'s loop bound was the solver's *grid-pass* count while `rounds` was meant to be the
   ratio-loop trip count. rd_n4 was already `rounds*rrd*stride`, so the kernel read **`rrd` times
   past the end of its own buffer** at any ratio above 1:1 — 62× at 62:1 — and faulted. Minimal repro
   (pre-fix binary, 64 MiB, `--iters 1`): ratio 1 exits 3, ratios 2/4/62 each take
   `HSA_STATUS_ERROR_MEMORY_FAULT`. It was ratio-dependent, not footprint-dependent.
   `--selftest-geom` called all of those geometries `ok=1` because it checked the ratio and the
   allocation but never the indexing the kernel runs; `plan_bounds()` now re-derives the kernel's
   index arithmetic from the same `Plan` and `invariants()` rejects it, and `--selftest` (device)
   proves the bytes instead of the arithmetic. One `--selftest` run now covers every configuration
   that faulted — mix at 4/10/20/32/62/128:1 at 528 MiB, both `--wr-arena-mib 2` shapes, the
   `mixlaunch` 48×11.01 MiB geometry, and a cold write — each in its own child process (a device
   fault poisons the context, so one process could only ever report the first dead configuration).
2. The read checksum was `sums[blockIdx.x] = per_thread_acc` — a per-thread store wearing a block
   reduction's clothes. Every thread of a block wrote the same address and the surviving value was
   one thread's partial, i.e. **1/blockDim of the bytes actually read** (measured pre-fix:
   `sum=65520 want=16773120`, exactly 1/256). That is why *every* warm row failed and why no number
   from that run may be quoted. It is now a real shared-memory block reduction, exact as before, and
   it runs inside the timed burst (`--verify 1`), so a row cannot be printed without having proved
   what it touched. The write side got the symmetric fix: the arena is pre-filled with a value the
   write stream never stores, so the untimed read-back proves the writes **landed** rather than
   finding a buffer that was already full of 1.0f.
3. There was no third defect. **`--mode write --pages cold` did not fault.** The `mixed_bw.cpp:417`
   lines belong to the *warm* `mix` rows and `:421` to `F_cold_mix62`; `tmp/q8/mixed_bw.err` has 11
   `Memory Fault Error` lines — one from `--selftest`, nine from `B_mix_{4,10,20,32,62,128}` and the
   three `D_*`, one from `F_cold_mix62` — and nothing after the cold-write banner, whose row is the
   last line of `tmp/q8/mixed_bw.log`. What that row genuinely lacked was evidence, not a crash; see
   the `F_cold_write` correction below.

The expectations table above is **not** edited: a run that failed does not get to rewrite its test.
What follows is the corrected command list, and why each line changed.

| row | correction | why |
|---|---|---|
| `0a` | command unchanged | `--selftest-geom` still passes all 25 configurations, and it is now *allowed* to be cheap: the bounds model is in it, and the device witness in `0b` is what carries the load |
| `0b` | command unchanged; now forks one child per device case and prints byte accounts | it is the gate for this whole sweep. If `--selftest` does not exit 0, no row below is worth tabulating |
| `A_*`, `B_*`, `C_*`, `D_*` | **commands unchanged** | the geometry was right; the tool was wrong. Every one of these rows must now print `checksum=ok`; anything else means the instrument broke again, not that the box changed |
| `E_same` | command unchanged; expect `checksum=ok_writes_aliased` | reading back the just-dirtied lines makes the write side *physically* unverifiable (the region was pre-filled with the same bytes the write stores). The field says so instead of printing a bare `ok`; the read side is verified |
| `E_62_async`, `E_62_synced` | commands unchanged; both now print `checksum=ok rd_sum=… wr_landed=exact` | the worry that they "did nothing because there is no arena" is not the case: `--overlap 0` gives the write and the read **separate** regions (write 8.4 MiB into one, read a different 517.5 MiB), and `--pair-sync 1` inserts a real `hipDeviceSynchronize()` between the two launches. Note what that sync is: one whole-queue drain between the passes, not a per-launch fence — the engine never syncs per launch, so expectation 9 reads this row as "drain cost", not as an engine shape |
| `F_cold_read` | command unchanged; expect `checksum=unverifiable_cold_pages fresh_zero_sum=1` and **exit 3** | verifying cold bytes would warm them, so the row is unverifiable *by construction* and now exits non-zero rather than printing `checksum=skipped_cold` for a reader to walk past. `fresh_zero_sum=1` stays the admissible freshness witness; `fresh_zero_sum=0` still means throw the row out (rule 10) |
| `F_cold_mix62` | command unchanged | it faulted for defect 1 and never printed. Same unverifiable-cold treatment as `F_cold_read` |
| `F_cold_write` | command unchanged; expect `checksum=unverifiable_cold_pages fresh_zero_sum=not_observable_write_pass fresh_alloc_untouched=1`, **exit 3** | a write pass overwrites the only evidence of freshness, so expectation 10's `fresh_zero_sum=1` is measurable only on rows that *read*. This row's freshness claim is structural — `alloc_dev(fill=false)` issues no access to those pages before the timed launch, so the flag is `fresh_alloc_untouched=1` — and its admissible conclusion is limited to "the first GPU pass over fresh pages costs this much for a write stream" |
| **`G_step_1p5g_61p7`** (new) | `tools/build/mixed_bw --mode mixlaunch --ratio 617:10 --nbuf 48 --buf-mib 32.45 --pages warm --iters 8 --tag G_step_1p5g_61p7` | the **step's absolute traffic** in its own launch shape: 1531.9 MiB read + 24.83 MiB written at 61.70:1, i.e. the traffic model's totals rather than a footprint scaled to the MLP's 528 MiB. Ratio 617:10 is the step's 61.7:1 written as integers so `ratio_act` cannot drift |
| **`G_step_1p5g_61p7_arena2m`** (new) | `tools/build/mixed_bw --mode mix --ratio 62 --footprint-mib 1557 --wr-arena-mib 2 --pages warm --iters 8 --tag G_step_1p5g_61p7_arena2m` | the same budget with the engine's write *shape*: 1530.6 MiB read while a 2 MiB arena is re-dirtied ~250 times (`fit=write_arena_cycled`), which is closer to 424 launches over a GDN-state-sized arena than any one-shot write. This is the row that would show dirty-line pressure from writes the cache never lets reach DRAM |
| **`G_mlpshape_21p5`** (new) | `tools/build/mixed_bw --mode mixlaunch --ratio 43:2 --nbuf 48 --buf-mib 11.01 --pages warm --iters 8 --tag G_mlpshape_21p5` | the MLP read shape at the step's **absolute write volume** (504.0 MiB read + 23.44 MiB written = 21.50:1). With `C_mixlaunch_62` it brackets the same geometry at two write shares 3× apart; if the box is flat across 21.5→62:1 it is flat, and if it is not, this pair says by how much at the geometry the step actually walks |

and after the sweep, gate the rows themselves rather than the exit codes a `tee` pipeline swallows:

```bash
# every warm row must read ok (E_same: ok_writes_aliased); every cold row unverifiable_cold_pages
grep '^MIXEDBW' tmp/q8/mixed_bw.log | sed 's/.*tag=\([^ ]*\).*checksum=\([^ ]*\).*/\2  \1/' | sort
# and the pricing step, unchanged:
tools/step_traffic_model.py --probe-file tmp/q8/mixed_bw.log --lanes 128
```

**Which rows decide the question now.** `A_read_512m` first (if it is not 225–240 GB/s the
instrument is wrong and nothing below it means anything), then four rows:
`C_mixlaunch_62` (the step's ratio in the engine's serialized-launch shape), `B_mix_62` (the same
ratio with both streams genuinely in flight inside one kernel — the only row that can show read/write
interference as opposed to serialization), `G_step_1p5g_61p7` (the step's absolute bytes) and
`G_step_1p5g_61p7_arena2m` (the step's write shape, one small arena re-dirtied). Expectations 4
and 6 are read off the first two; the G rows are what the pre-registered list was missing.

**Is `C_mixlaunch_62` the step's ratio row?** For the ratio and the launch shape, yes: it runs
62.002:1 over 48 × 11.01 MiB read once, one stream, 96 launches — which is the step's ratio against
the weight family that dominates it, and expectation 4's decisive number is read from it. It is not
the step's *write* row: per iteration it writes 8.38 MiB of fresh arena where the step writes ~24.6 MB
against 1.5 GB read, and the step's writes re-dirty a small arena rather than streaming a new one.
So: yes, this question hinges on that one number — but only together with `B_mix_62` (overlap,
not serialization) and the two G rows (absolute volume, arena cycling). If `C_mixlaunch_62` and
`G_step_1p5g_61p7` land within a few % of each other and of `C_read_48x11`, the traffic-mix story is
dead on those three numbers and nothing else in this file needs re-reading.

**The prediction, in one sentence: the curve is flat.** Reads dominate the step so completely
(61.7 : 1) that mixed traffic can only move the ceiling from 6.47 ms to ~6.7 ms, so the expected
outcome is that 237 GB/s was the right denominator after all, the traffic-mix story dies next to
Q7's M1, and the residual stays where §Q6 left it: 424 launches of dispatch and inter-launch gaps,
plus whatever the instrument itself adds. A cliff between 130 and 200 GB/s at 62:1 is the one
result that would change the plan's shape, and it is what rows B_62, C_mixlaunch_62 and E_* test.

## Results

_Not yet taken: these rows need a GPU pass with zero subagents active, taken by the main agent._

_The first attempt (2026-10-13, tag `q8-mixedbw`, blocking, zero subagents, util 92 %→25 %) printed
13 of 23 rows and yielded **no** number: every warm row failed a checksum that was itself broken, and
every `mix` row with ratio > 1 faulted. Both instrument defects are diagnosed and fixed above (§"Row
corrections after the first attempt"); the raw lines are in `tmp/q8/mixed_bw.log` and its stderr in
`tmp/q8/mixed_bw.err`, kept as evidence about the instrument and not as data — by rule 11 none of
those 13 rows may be tabulated, quoted, or used to pre-judge the re-run._

| tag | mode | ratio_act | pages | ms | GB/s | %of 237 | suspect | checksum |
|---|---|---|---|---|---|---|---|---|
| (fill from `MIXEDBW` lines; `tools/step_traffic_model.py --probe-file` renders the priced table) | | | | | | | | |

## Consequences for the plan

Written before the run so the conclusion cannot drift into it:

1. **If the curve is flat (expected):** the traffic-mix explanation is retired next to Q7's M1.
   The roofline stays `bytes / 237 GB/s`; the ~5.8 ms residue stays attributed to launch count and
   inter-launch gaps, which is what the byte-reduction ladder below independently says.
2. **If there is a cliff at the step's ratio:** every roofline in `bench/performance-status.md`
   gets re-based on the measured mixed rate at 62:1, and the plan's #1 priority becomes making the
   step's traffic *more* read-pure (fold the casts, stop widening to F32, write the conv window in
   place) rather than merely fewer bytes.
3. **Either way, the byte-reduction ladder says byte reduction is not where the milliseconds are.**
   Ranked from `tools/step_traffic_model.py`, with `ms` priced at the no-cliff ceiling:

| MB/token | ms/token at ceiling | launches | opportunity |
|---|---|---|---|
| 37.75 | 0.160 | 0 | GDN state read+write — **not reducible**: it is the recurrence, and F32 is what the reference pins. Warm it already runs at ~876 GB/s, cache-resident |
| 3.98 | 0.017 | 18 | conv window read + `conv_new` + roll-back copy — kill the copy with a ring/in-place window (§08.2's alternative). The prize is 18 launches |
| 1.99 | 0.008 | 0 | logits write + D2H — a device argmax deletes the D2H read but must keep §02.5's lowest-index tie rule |
| 1.78 | 0.008 | 115 | the cast round-trip — BF16 staging in mlp/proj deletes ~48 of them. The bytes are noise; the **launches** are the prize |
| 1.77 | 0.008 | 0 | `conv1d` weight read as F32 instead of BF16 — convert in-kernel instead of pre-widening |
| 1.57 | 0.007 | 0 | KV read at 128 lanes — the only row that grows; at 4032 lanes it is 49.5 MB/token, and the fix there is the attention kernel (~13.7 GB/s against its own bytes), not fewer bytes |
| 1.51 | 0.006 | 0 | activation tiles read as F32 by the GEMVs — BF16 staging halves them |
| **1504** | **6.35** | 0 | **the weights themselves: 96.5 % of the read budget.** The only byte lever of the right size is weight dtype (M3 — and for speed, not for fitting), which is out of scope here |

   Everything except the weights is ~46 MB/token = **0.21 ms** at the ceiling, against a ~5.8 ms
   residue. And the largest non-weight item, the 37.75 MB of GDN state, is not reducible at all.
   Byte reduction cannot close this gap at this model size; only the rate or the launch count can.
