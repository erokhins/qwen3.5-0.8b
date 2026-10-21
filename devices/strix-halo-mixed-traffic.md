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
* the engine costs **12.399 ms/token** (frozen ledger progress/2026-10-16/performance-status-2026-10-16.md:666, Q6.1 stage medians at 128
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
} 2> progress/devices/evidence/q8/mixed_bw.err | tee progress/devices/evidence/q8/mixed_bw.log

# and finally, with the log in hand:
tools/step_traffic_model.py --probe-file progress/devices/evidence/q8/mixed_bw.log --lanes 128
```

The stderr side (device banner, `NOTE:` lines for any suspect row) belongs in
`progress/devices/evidence/q8/mixed_bw.err`, which is why the pricing step reads the stdout log only: `MIXEDBW` is the
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
   lines belong to the *warm* `mix` rows and `:421` to `F_cold_mix62`; `progress/devices/evidence/q8/mixed_bw.err` has 11
   `Memory Fault Error` lines — one from `--selftest`, nine from `B_mix_{4,10,20,32,62,128}` and the
   three `D_*`, one from `F_cold_mix62` — and nothing after the cold-write banner, whose row is the
   last line of `progress/devices/evidence/q8/mixed_bw.log`. What that row genuinely lacked was evidence, not a crash; see
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
| **`G_step_1p5g_61p7_arena2m`** (new) | `tools/build/mixed_bw --mode mix --ratio 62 --footprint-mib 1557 --wr-arena-mib 2 --pages warm --iters 8 --tag G_step_1p5g_61p7_arena2m` | the same budget with the engine's write *shape*: 1530.6 MiB read while a 2.469 MiB arena is re-dirtied 10 times (`wr_mib=24.688`, `fit=write_arena_cycled`), which is closer to the step re-dirtying a GDN-state-sized arena than any one-shot write of 24.7 MiB. Expect `wr_landed=arena_slots_only_cycled`: the account proves every arena slot holds a written value, but a slot written ten times is indistinguishable from one written once, and the row says so rather than rounding the claim up. This is the row that would show dirty-line pressure from writes the cache never lets reach DRAM |
| **`G_mlpshape_21p5`** (new) | `tools/build/mixed_bw --mode mixlaunch --ratio 43:2 --nbuf 48 --buf-mib 11.01 --pages warm --iters 8 --tag G_mlpshape_21p5` | the MLP read shape at the step's **absolute write volume** (504.0 MiB read + 23.44 MiB written = 21.50:1). With `C_mixlaunch_62` it brackets the same geometry at two write shares 3× apart; if the box is flat across 21.5→62:1 it is flat, and if it is not, this pair says by how much at the geometry the step actually walks |

and after the sweep, gate the rows themselves rather than the exit codes a `tee` pipeline swallows:

```bash
# every warm row must read ok (E_same: ok_writes_aliased); every cold row unverifiable_cold_pages
grep '^MIXEDBW' progress/devices/evidence/q8/mixed_bw.log | sed 's/.*tag=\([^ ]*\).*checksum=\([^ ]*\).*/\2  \1/' | sort
# and the pricing step, unchanged:
tools/step_traffic_model.py --probe-file progress/devices/evidence/q8/mixed_bw.log --lanes 128
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
corrections after the first attempt"); the raw lines are in `progress/devices/evidence/q8/mixed_bw.log` and its stderr in
`progress/devices/evidence/q8/mixed_bw.err`, kept as evidence about the instrument and not as data — by rule 11 none of
those 13 rows may be tabulated, quoted, or used to pre-judge the re-run._

| tag | mode | ratio_act | pages | ms | GB/s | %of 237 | suspect | checksum |
|---|---|---|---|---|---|---|---|---|
| A_read_512m | read | 0.000 | warm | 2.3274 | 237.7 | 100.3 | 0 | ok |
| A_write_512m | write | 0.000 | warm | 2.5863 | 214.1 | 90.3 | 0 | ok |
| B_mix_1 | mix | 1.000 | warm | 2.6776 | 206.6 | 87.2 | 0 | ok |
| B_mix_4 | mix | 4.000 | warm | 2.6013 | 212.6 | 89.7 | 0 | ok |
| B_mix_10 | mix | 10.000 | warm | 2.4305 | 227.6 | 96.1 | 0 | ok |
| B_mix_20 | mix | 20.000 | warm | 2.3939 | 231.2 | 97.6 | 0 | ok |
| B_mix_32 | mix | 32.000 | warm | 2.5222 | 219.5 | 92.6 | 0 | ok |
| B_mix_62 | mix | 62.000 | warm | 2.3546 | 235.0 | 99.1 | 0 | ok |
| B_mix_128 | mix | 128.046 | warm | 2.3450 | 235.7 | 99.4 | 0 | ok |
| C_read_48x11 | read | 0.000 | warm | 2.4656 | 224.4 | 94.7 | 0 | ok |
| C_mixlaunch_1 | mixlaunch | 1.000 | warm | 2.7217 | 203.4 | 85.8 | 0 | ok |
| C_mixlaunch_20 | mixlaunch | 19.999 | warm | 2.8088 | 197.1 | 83.2 | 0 | ok |
| C_mixlaunch_62 | mixlaunch | 62.002 | warm | 2.6977 | 205.3 | 86.6 | 0 | ok |
| C_mixlaunch_128 | mixlaunch | 128.046 | warm | 2.6726 | 207.1 | 87.4 | 0 | ok |
| D_62_oneshot | mix | 62.000 | warm | 2.3484 | 235.6 | 99.4 | 0 | ok |
| D_62_arena2m | mix | 62.000 | warm | 2.3616 | 234.3 | 98.8 | 0 | ok |
| D_20_arena2m | mix | 20.000 | warm | 2.2714 | 243.7 | 102.8 | **1** | ok |
| E_same | wthenr | 1.000 | warm | 4.8228 | 221.8 | 93.6 | 0 | ok_writes_aliased |
| E_62_async | wthenr | 61.762 | warm | 2.3348 | 236.2 | 99.7 | 0 | ok |
| E_62_synced | wthenr | 61.762 | warm | 2.3495 | 234.7 | 99.0 | 0 | ok |
| F_cold_read | read | 0.000 | cold | 2.8529 | 193.9 | 81.8 | 0 | unverifiable_cold_pages |
| F_cold_mix62 | mix | 62.000 | cold | 2.7981 | 197.7 | 83.4 | 0 | unverifiable_cold_pages |
| F_cold_write | write | 0.000 | cold | 2.9308 | 188.9 | 79.7 | 0 | unverifiable_cold_pages |
| G_step_1p5g_61p7 | mixlaunch | 61.699 | warm | 8.8869 | 183.7 | 77.5 | 0 | ok |
| G_step_1p5g_61p7_arena2m | mix | 62.000 | warm | 7.4509 | 218.9 | 92.4 | 0 | ok |
| G_mlpshape_21p5 | mixlaunch | 21.500 | warm | 2.8508 | 194.0 | 81.9 | 0 | ok |

Taken 2026-10-13 in window `q8b-sweep`: blocking, zero subagents, `util` 34 % at acquire and 31 % at
release (recorded, then discounted as everywhere else here), `--selftest` green first (`SELFTEST ok
fails=0`, including all four anti-vacuity injections), 26 of 26 rows printed, raw lines in
`progress/devices/evidence/q8/mixed_bw2.log`.

**The prediction held: the curve is flat, so the traffic-mix story is dead.** Walking the pre-registered
expectations one at a time, as written before the run: (1) `A_read_512m` **100.3 %** — the instrument agrees
with the committed read probe, so it is not lying about anything else either; (2) `A_write_512m` 90.3 %
(214 GB/s), above the ≥ 200 GB/s floor; (3) the `B_mix` curve is monotone toward read and flat where the
step lives — **96–99 % of the read probe at every ratio ≥ 10:1**, and 87.2 % at 1:1, right on the ~210 GB/s
the 1:1 copy number predicts; (4) the decisive 62:1 row came in at **235.0 GB/s (99.1 %)** against the ~230
expected; (5) `C_read_48x11` is 94.7 % versus `A_read_512m`'s 100.3 %, i.e. **5.6 % outside** the ±5 % band —
48 separate 11 MiB buffers walked once is genuinely a little worse than one big region, and that is recorded
as a real (small) geometry effect, not smoothed; (6) `C_mixlaunch_62` reached **91.4 % of `C_read_48x11`
against a ≥ 95 % bar**, so splitting one kernel into 96 serialized launches costs about **9 %** — a named
finding, and Q6.1 already showed the same shape from the other side; (7) a 2 MiB arena re-dirtied 260×
matched the one-shot write (98.8 % vs 99.4 %) as expected; (8) read-after-write on freshly dirtied lines was
**free**: `E_same`'s pair took 4.8228 ms against 2.5863 + 2.3274 = 4.9137 ms written then read separately,
−1.8 %, well inside ±10 %, which kills write-back pressure as a mechanism; (9) `E_62_synced` vs `E_62_async`
differ by 0.6 %; (10) the three `--pages cold` rows are **void by their own rule** — they printed 79.7–83.4 %
with `checksum=unverifiable_cold_pages` (exit 3), so this probe does not reproduce the 35 GB/s figure
`strix-halo-unified-memory.md` records from a different tool, and neither number is used here; (11) every
warm checksum is `ok`; (12) one row tripped the >240 GB/s trap — `D_20_arena2m` at **102.8 % with
`suspect=1`**, flagged and excluded, exactly the small-arena-was-cache-resident case the flag exists for.

Two rows are worth pulling out because they are the tightest apples-to-apples comparison this project has
managed against the engine's own numbers. `C_mixlaunch_62` is the MLP's geometry — 48 × 11.01 MiB read once,
96 serialized launches, the step's 62:1 ratio — and it runs at **86.6 % of the read probe**, where the same
kernels inside a real decode step run at **52 %** (§Q6.1). And `G_step_1p5g_61p7_arena2m` is the whole step's
traffic budget — 1530.6 MiB read against 24.7 MiB written into a small re-dirtied arena — at **218.9 GB/s
(92.4 %)**, which prices one token's weight traffic at **7.06 ms**, not 12.4 ms. So the ceiling question is
answered in the boring direction: **237 GB/s was approximately the right denominator after all**, and the
remaining ~5.3 ms is neither dispatch (Q6.1), nor warmth (Q7 M1), nor traffic mix or write-back (this file).


## Consequences for the plan

Written before the run so the conclusion cannot drift into it — and item 1 is what happened:

1. **Struck — the curve was flat.** At the step's own ratio and geometry the box delivers 92–99 % of its
   read-probe rate, so the roofline stays `bytes / 237 GB/s` (conservatively `bytes / 219 GB/s`, the
   step-budget row), and the traffic-mix explanation is retired next to Q7's M1. What that leaves is
   narrower and more specific than "launch count": the engine reads MLP-shaped traffic at **52 % of what the
   identical geometry achieves in a probe**, and the one measured gap that does exist — 9 % for serialized
   launches versus one kernel, §expectation 6 — is an order of magnitude too small to be it.
2. **If there is a cliff at the step's ratio:** every roofline in `engine/performance-status.md`
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

## Q9: the two surviving candidates — pre-registered 2026-10-13, before any Q9 device run

Q8 left one comparison standing. `C_mixlaunch_62` — 48 × 11.01 MiB read once, 96 serialized launches,
62:1 — runs at **86.6 % of the read probe**, while the same MLP kernels inside a real step run at **52 %**
(4.264 ms/token against a 2.230 ms roofline), and Q7 M1 showed those same kernels at the same 528 MB
footprint reaching **93 %** in isolation. The box offers the floor; the engine does not reach it. Two
mechanisms can still do that, and their fixes are opposite:

* **(A) intra-kernel mapping** — *which thread touches which byte* inside `mlp_gateup`/`mlp_down`,
  including the row's LDS staging and its 8-rows-per-block wavefront shape. Fix: change a kernel.
* **(B) inter-launch locality** — *what else was touched between* two MLP launches: in a step every MLP
  launch is separated from the next by several other families' reads and writes, and no MLP blob is
  revisited inside a token. Fix: change the step's layout, or re-declare the ceiling.

`tools/mixed_bw` gained one instrument per mechanism, each holding every other variable fixed:

| new | holds fixed | only free variable |
|---|---|---|
| `--pattern engine-mlp` (`--engine-in 1024\|3584`, `--stage 0\|1`) | bytes, regions, grid, threads, launch count, serialized order, checksum | which thread reads which byte — the engine's real per-thread index arithmetic, with the dot products, the SwiGLU epilogue and the output store removed |
| `--mode interleave --interleave blocked\|roundrobin` (`--revisits`, `--dummy-mib`, `--dummy-nbuf`) | the **multiset** of launches and the total bytes, proved equal by `--selftest`'s order anti-vacuity | the order in which the regions are touched |

### CPU-side verdict on (A), written before any device run: the weight reads are already fully coalesced

Every GEMV family reads weights the same way (`prototype/mlp_gemv/mlp_gemv_kernels.h`, which
`engine/kernels_mlp.cpp` includes verbatim, and `prototype/proj_gemv/proj_gemv_kernels.h`): one wavefront
per weight row (`row = blockIdx.x*rpb + threadIdx.x/warpSize`), and inside the row
`for (v = lane; v < words; v += warpSize) wrow[v]` over `uint4` words.

| family | weight row | uint4 words/row | weight bytes per warp instruction | 128-B lines touched | lines fully consumed | stride, next iteration | verdict |
|---|---|---|---|---|---|---|---|
| `mlp_gateup<1024>` | 2 KiB, **two streams** (gate + up) | 128 | 32×16×2 = 1024 B | 8 | 8/8 = 100 % | +512 B per lane, still whole lines | coalesced |
| `mlp_down<3584>` | 7 KiB, one stream | 448 | 32×16 = 512 B | 4 | 4/4 = 100 % | +512 B | coalesced |
| projections `q/k/v/o/gate/up/down` | row-major `uint4` walk, same loop | IN/8 | 512 B | 4 | 4/4 = 100 % | +512 B | coalesced |
| `lm_head` (248 320 rows) | row-major `uint4` walk | IN/8 | 512 B | 4 | 4/4 = 100 % | +512 B | coalesced, and measured at ceiling |

So (A) in the form it was posed — a per-thread stride on the weight read — is **dead on the host**: there
is no strided weight read and no partially-used 128-byte transaction anywhere in the engine's GEMV path.
Q7 M1 covers most of what is left: the 93 % burst runs *the same kernel source* at the same rpb, so the
staging, the two-stream interleave and the wavefront shape are not new to the burst. What the burst never
reproduced is the step's x/act traffic and the spacing between launches — which is why the pattern rows are
still worth running (a fall there would be the surprise), and why (B) is the stronger prior.

### The pattern geometry is the engine's, not an average

A gate blob, an up blob and a down blob are each 3584×1024×2 = 1024×3584×2 = **7 MiB exactly**, so the
probe's regions can be the engine's blobs rather than an averaged 11.01 MiB launch:

| row | launches | per launch | grid vs rows | engine truth |
|---|---|---|---|---|
| `--nbuf 48 --buf-mib 7 --pattern engine-mlp --engine-in 1024 --threads 256 --blocks 448` | 24, each pairing two 7 MiB regions (gate + up) | 2 × 7 MiB, 3584 rows of 2 KiB | 448×256 ⇒ 3584 wavefronts = **exactly one grid pass** | `mlp_gateup`, 24 launches ✓ |
| `--nbuf 24 --buf-mib 7 --pattern engine-mlp --engine-in 3584 --threads 256 --blocks 128` | 24 | 7 MiB, 1024 rows of 7 KiB | 128×256 ⇒ 1024 wavefronts = **one pass** | `mlp_down`, 24 launches ✓ |
| the two together | 48 | 504 MiB | — | one token's whole MLP weight traffic |

`--stage 1` (the default with `--pattern engine-mlp`) reproduces the engine's activation staging too: the
kernels declare `__shared__ float xs[IN]`, stage one row, `__syncthreads()`, then re-read `&xs[v*8]` —
32 B of LDS per 16 B of weight, every launch. The probe stages the same width and element type, so the LDS
footprint (4 KiB gate/up, 14 KiB down) and the bank pattern match the engine's rather than approximating
them. The staged buffer is pre-filled with `0.0f`, so it adds exactly 0.0 to a checksum that must stay
exact: traffic, not evidence. The row prints `stage_mib=` and `stage_witness=zero_by_construction`, and
`--stage 0` is the row that separates the staging from everything else.

### Thresholds — written down now, applied to the run unchanged

Every comparison is a **ratio of same-window rows**, never a ratio against a historical number: absolute
%of-probe moves a few % between windows, so a threshold against a stale row could be flipped by the window
instead of by the mechanism. Every row must print `checksum=ok` and `suspect=0`, or it is void (rule 11).

**(A) mapping** — same bytes, same launch count, same grid, the read mapping the only difference. Each
pair was `--dry-run`-checked to bill **identical** byte counts and launch counts (printed by
`--selftest`'s mapping anti-vacuity, which also rejects a plan where `--pattern engine-mlp` failed to
reach its launches — the vacuous form of this instrument, invisible to the byte account because both
readers touch the same bytes):

| pair | engine side | contiguous twin | both sides bill |
|---|---|---|---|
| gate/up, pure read | `--mode read --pattern engine-mlp --engine-in 1024 --nbuf 48 --buf-mib 7 --threads 256 --blocks 448` | `--mode read --engine-in 1024 --nbuf 24 --buf-mib 14 --threads 256 --blocks 448` | 336.000 MiB, 24 launches, 448 blocks |
| gate/up, step's write stream | `--mode mixlaunch --ratio 62 --pattern engine-mlp --engine-in 1024 --nbuf 48 --buf-mib 7 --threads 256 --blocks 448` | `--mode mixlaunch --ratio 62 --engine-in 1024 --nbuf 24 --buf-mib 14 --threads 256 --blocks 448` | 330.000 MiB read + 5.324 MiB written, 24 + 24 launches |
| down, step's write stream | `--mode mixlaunch --ratio 62 --pattern engine-mlp --engine-in 3584 --nbuf 24 --buf-mib 7 --threads 256 --blocks 128` | `--mode mixlaunch --ratio 62 --engine-in 3584 --nbuf 24 --buf-mib 7 --threads 256 --blocks 128` | 164.391 MiB read + 2.652 MiB written, 24 + 24 launches |

`--engine-in` is passed to the contiguous twins on purpose: the row width is geometry (it decides the
whole-row rounding of a region), while `--pattern` alone decides which kernel runs. Without it the two
sides round differently and the comparison is not byte-clean.

* **(A) is refuted per family if `gbs(engine) ≥ 0.95 × gbs(twin)`, confirmed if `≤ 0.75 ×`, partial in
  between**; a partial reports the share of the 86.6 → 52 gap it accounts for and is not rounded up. The
  bytes-weighted score over the three pairs (336 : 330 : 164 MiB) is quoted alongside, never instead of
  the per-family verdict — gate/up and down are different launches and one may refute while the other
  confirms.
* inside (A), attribution: `--stage 0` on the engine side removes the engine's `__shared__ float xs[IN]`
  staging and its `&xs[v*8]` re-reads. If `gbs(stage1) ≤ 0.90 × gbs(stage0)` the fall is LDS staging, a
  one-kernel fix with a known shape; if not, the fall belongs to the two-stream row interleave or the row
  tail, which is a different fix.
* `C_mixlaunch_62` itself is re-taken in the same window as a reference only (it is the average launch
  shape, 11.01 MiB, not one of the engine's launches, so it is not an (A) denominator).

**(B) locality** — identical launch multiset, two orders. The stand-ins stand in for the other families and
are sized to the step: ~424 launches/token minus the MLP's 48 leaves 376, and 376 × 2.73 MiB ≈ 1026 MiB ≈
the step's 1530.6 MiB read budget minus the MLP's 504 MiB.

* `B_blk` = `--mode interleave --nbuf 48 --buf-mib 11.01 --revisits 1 --dummy-nbuf 376 --dummy-mib 2.73
  --interleave blocked --threads 256 --blocks 448` — all stand-ins first, then the 48 MLP-shaped buffers back to
  back: the best locality the MLP can have.
* `B_rr` = the same row with `--interleave roundrobin` — pass-major over the MLP buffers with the stand-ins
  spread evenly between them, ≈7.8 non-MLP launches between consecutive MLP launches, the step's spacing.
* **(B) confirmed if `gbs(B_rr) ≤ 0.75 × gbs(B_blk)`; refuted if `≥ 0.92 ×`; partial between.** The bound
  is not arbitrary: fully explaining the step needs an order penalty of ~0.60, so a fall under a quarter
  cannot be the mechanism, and a quarter-or-more is well outside this box's ±5 % step reproducibility.
  `--selftest` proves the two orders launch one multiset of the same sizes (`same_multiset ok=1`) and that
  dropping a stand-in breaks that equality (`dropped_dummy_caught=1`), so the only difference a row can
  show is the order.
* `B_rr_nostandins` (same row, `--dummy-nbuf 1`) separates "the stand-ins' own bytes compete for the
  controller" from "the stand-ins evict what the MLP just read": if the fall survives removing 375 of 376
  stand-ins, it was contention, not locality, and (B) is re-opened as a controller question rather than
  closed as an L2/TLB one.

**(A) and (B) together — the row that ends the argument**

`R_decisive` = `--mode interleave --nbuf 72 --buf-mib 7 --pattern engine-mlp --engine-in 3584 --revisits 1
--dummy-nbuf 376 --dummy-mib 2.73 --interleave roundrobin --threads 256 --blocks 128`, judged **against
`A_read_512m` re-taken in the same window** (not against the historical 100.3 %). Why this shape: 72
launches × 7 MiB is the MLP's 504 MiB in launches whose row width (7 KiB) and grid (128 blocks ⇒ 1024
wavefronts ⇒ 1024 rows, one grid pass) are `mlp_down`'s exactly, with the step's other families' bytes
spread between them. It cannot carry the gate/up two-stream shape — `interleave` keeps one region per
launch so that the two orders are the same multiset — and that shape is tested by the (A) rows instead.

* **≤ 62 % of that probe → (A)+(B) reproduce the step's 52 %.** Nothing further is owed on the
  memory-system side; the work becomes per-mechanism as attributed by the (A) and (B) rows above.
* **≥ 80 % → neither mechanism explains the step**, and Q9's conclusion reads: the residue is per-launch
  engine work (epilogue, waiting, co-tenancy with the step's own writes), not DRAM-side mapping or
  locality; the next instrument is in-situ per-launch stall attribution, not a kernel change.
* between → report the fraction of the gap closed and leave the residue named and open.

If either confirms, the prize is bounded by arithmetic Q6.1 already carries: the MLP is **4.264 ms/token
in-situ against a 2.594 ms burst and a 2.230 ms roofline**, so the whole MLP gap is **1.67–2.03 ms/token,
13–16 % of a 12.4 ms token**. A mechanism whose measured rate ratio is `f` explains
`(1/f − 1) × 2.594 ms`, capped at that gap: `f = 0.75` buys **≈0.86 ms/token** (about half the gap), and
`f = 0.60` buys the whole 1.67 ms — which is why 0.75 is the confirm line and not something softer. That
number, not a percentage, is what the next step gets scheduled against. A confirmed (A) is a kernel edit
inside `engine/kernels_mlp.cpp`, which includes `prototype/mlp_gemv/mlp_gemv_kernels.h` verbatim, and it
must come with that gate re-run; a confirmed (B) is a step-layout change (group the MLP launches, or stop
reading 504 MiB of weights per token), never a kernel edit.

### The pre-registered command list (main agent: one lock window, blocking, zero subagents)

Every row is `--pages warm` — a cold row is void by rule 11 and exits non-zero, and cold is not what the
step is. `--blocks` is a cap: the solver takes the largest grid within 0.5 % of the best fit, so the
resolved `blocks=` in the row is the number to compare, and each pair resolves to the same one.

```bash
tag="q9-mapping-locality"; tools/gpu_lock.sh acquire "$tag"; tools/gpu_lock.sh util
tools/build/mixed_bw --selftest            # gate: must print `SELFTEST ok fails=0` before any timing
mkdir -p tmp/q9
r() { tools/build/mixed_bw "$@" | tee -a progress/devices/evidence/q9/rows.log; }

# references, same window (the two committed rows the sections above quote)
r --tag A_read_512m     --mode read --footprint-mib 512 --threads 256 --pages warm --iters 8
r --tag C_mixlaunch_62  --mode mixlaunch --ratio 62 --nbuf 48 --buf-mib 11.01 --threads 256 --pages warm --iters 8

# (A) mapping: three engine/contiguous pairs, each byte-identical on both sides
r --tag A_gu_eng        --mode read --pattern engine-mlp --engine-in 1024 --nbuf 48 --buf-mib 7 \
                          --threads 256 --blocks 448 --pages warm --iters 8
r --tag A_gu_plain      --mode read --engine-in 1024 --nbuf 24 --buf-mib 14 \
                          --threads 256 --blocks 448 --pages warm --iters 8
r --tag A_gu_eng_nostage --mode read --pattern engine-mlp --engine-in 1024 --stage 0 --nbuf 48 --buf-mib 7 \
                          --threads 256 --blocks 448 --pages warm --iters 8
r --tag A_mgu_eng       --mode mixlaunch --ratio 62 --pattern engine-mlp --engine-in 1024 \
                          --nbuf 48 --buf-mib 7 --threads 256 --blocks 448 --pages warm --iters 8
r --tag A_mgu_plain     --mode mixlaunch --ratio 62 --engine-in 1024 --nbuf 24 --buf-mib 14 \
                          --threads 256 --blocks 448 --pages warm --iters 8
r --tag A_mdn_eng       --mode mixlaunch --ratio 62 --pattern engine-mlp --engine-in 3584 \
                          --nbuf 24 --buf-mib 7 --threads 256 --blocks 128 --pages warm --iters 8
r --tag A_mdn_plain     --mode mixlaunch --ratio 62 --engine-in 3584 --nbuf 24 --buf-mib 7 \
                          --threads 256 --blocks 128 --pages warm --iters 8

# (B) order: one launch multiset (1548.688 MiB, 424 read launches), two orders
r --tag B_blk           --mode interleave --nbuf 48 --buf-mib 11.01 --revisits 1 \
                          --dummy-nbuf 376 --dummy-mib 2.73 --interleave blocked --threads 256 --pages warm --iters 5
r --tag B_rr            --mode interleave --nbuf 48 --buf-mib 11.01 --revisits 1 \
                          --dummy-nbuf 376 --dummy-mib 2.73 --interleave roundrobin --threads 256 --pages warm --iters 5
r --tag B_rr_nostandins --mode interleave --nbuf 48 --buf-mib 11.01 --revisits 1 \
                          --dummy-nbuf 1 --dummy-mib 2.73 --interleave roundrobin --threads 256 --pages warm --iters 5

# the row that ends the argument: engine mapping + the step's spacing + the step's read budget
r --tag R_decisive      --mode interleave --nbuf 72 --buf-mib 7 --pattern engine-mlp --engine-in 3584 \
                          --revisits 1 --dummy-nbuf 376 --dummy-mib 2.73 --interleave roundrobin \
                          --threads 256 --pages warm --iters 5
tools/gpu_lock.sh util; tools/gpu_lock.sh release "$tag"
```

Verdict arithmetic to apply to those rows, in this order: `gbs(A_*) ratio` per pair → (A) per family;
`gbs(B_rr)/gbs(B_blk)` → (B); then `gbs(R_decisive)/gbs(A_read_512m)` → the 62 % / 80 % call above. If
(B) is confirmed, `B_rr_nostandins` decides whether the finding is "eviction" (re-opened as a layout
question) or "controller contention" (re-opened as a Q8 refinement), and the two are different next steps.

Nothing here changes engine code, and nothing here is a number: the rows, the ratios and the two
exit-criterion lines were written before the run, and the thresholds are unchanged by whatever it prints.

## Q9 Results — taken 2026-10-13, window `q9-mapping-locality` (blocking, zero subagents, util 91 %→27 %)

`--selftest` ran first in the same window and printed `SELFTEST ok fails=0` over **65 device cases**,
including the byte-accounting injections for both new modes. 13 of 13 rows printed, all `checksum=ok`,
no row flagged `suspect`. Raw lines: `progress/devices/evidence/q9/rows.log`. Thresholds below are the ones written in
§"Thresholds" before the run, applied unchanged.

| tag | what it is | GB/s | % of read probe |
|---|---|---|---|
| `A_read_512m` | reference: contiguous sweep | 235.0 | 99.2 |
| `C_mixlaunch_62` | reference: the MLP's geometry, contiguous pattern | 213.0 | 89.9 |
| `A_gu_eng` | **gate/up bytes, the engine's own per-thread mapping** | **103.3** | **43.6** |
| `A_gu_plain` | same bytes, same grid, contiguous | 219.7 | 92.7 |
| `A_gu_eng_nostage` | engine mapping, LDS staging off | 113.4 | 47.8 |
| `A_mgu_eng` / `A_mgu_plain` | gate/up with the step's 62:1 writes | **100.6** / 212.9 | 42.5 / 89.8 |
| `A_mdn_eng` / `A_mdn_plain` | `down` shape (3584), engine vs contiguous | **127.1** / 201.9 | 53.6 / 85.2 |
| `B_blk` | 424 read launches, one launch multiset, **blocked** order | 183.3 | 77.3 |
| `B_rr` | **identical multiset, round-robin order** | 184.2 | 77.7 |
| `B_rr_nostandins` | round-robin with no stand-in traffic between | 207.0 | 87.4 |
| `R_decisive` | engine mapping + the step's spacing + the step's read budget | **140.5** | **59.3** |

**(A) confirmed, per family, at every pair.** `gbs(eng)/gbs(plain)` = **0.470** (gate/up), **0.472**
(gate/up with writes), **0.629** (`down`) — all three inside the ≤ 0.70 "confirmed" band, none in the partial
band, and the byte counts are identical on both sides of each pair by construction (336.0 MiB read-shape pair
checked as `same_bytes_as_contiguous` in `--selftest`), so this cannot be a byte difference wearing a
costume. Attribution inside (A): staging is *not* the fall — `nostage/eng = 1.098`, outside the ≤ 0.90 band
that would have blamed LDS — so what costs the factor of two is the **two-stream row interleave and/or the row
tail**, which is a kernel-internal question and the cheapest kind of thing to fix.

**(B) refuted.** Identical launch multiset, two orders: `B_rr / B_blk = 1.005`. Reordering 424 read launches
across a 1549 MiB footprint, with or without 376 stand-in launches interleaved, changes nothing. Inter-kernel
locality is out, and with it the eviction-versus-contention branch (`B_rr_nostandins`, 87.4 % vs 77.7 %, is
recorded but decides nothing since its premise never fired).

**The decisive row lands on mapping, not on the two combined.** `R_decisive / A_read_512m = 0.598`: engine
mapping, the step's spacing and the step's whole read budget together reproduce ~60 % of probe, and the
engine's MLP in-step figure from §Q6.1 is **52 %**. So the multi-package residue — the ~3.2 ms that Q6.1
located inside the kernels, that Q7 M1 refused to blame on warmth and Q8 refused to blame on traffic mix — is
**the weight-walk itself**: same bytes, same grid, same coalescing at the transaction level (the CPU verdict
above stands: 4/4 and 8/4 of every 128 B transaction used), and it arrives at half the rate. Contiguous
streaming is not free on this box; a strided two-stream walk with a row tail is a memory-system pattern, and
the pattern is the bottleneck.

**What this funds.** MLP is 4.264 ms of a 12.4 ms step against a 2.230 ms roofline; the probe says the gap is
reachable and kernel-internal. The next package is a mapping fix — the candidate named by the attribution is
the two-stream interleave in `mlp_gateup`, i.e. read `gate` and `up` as two contiguous passes (or one
contiguous pass over a co-allocated pair) instead of interleaving rows between them — with the usual proof
standard: byte-identical outputs, `margin_delta` ×1.0000, gate counts unchanged, and a measured A/B in one
locked window against the 4.264 ms in-step row it is meant to move.

**Dated correction (2026-10-17, `62b73e9`).** The `cast` family in this document's rows is **97 launches per
token, not 115**: the packed `g_raw│g_a│g_b` widening lost its only reader when `conv_kernel`/`gates_kernel`
began widening their BF16 operands in-register. The **per-launch figures in the rows stand** — they were
measured per launch and are unaffected — but the family byte totals move to **0.371 MB read / 0.741 MB written
per token** (from 0.593 / 1.186, i.e. −0.222 read and −0.445 written), and `G_step`'s read:write denominator
moves **61.7 : 1 → 62.8 : 1**. Rows L45 and L249-250 were left exactly as taken; this note is the correction,
because a measured number and the date it was taken on are both part of the measurement.
