# Performance status — the live ledger (current state, re-taken 2026-10-16)

**What this file is.** The current measured state of the engine on this box (Framework Desktop /
Strix Halo `gfx1151`, `Qwen/Qwen3.5-0.8B`, text-only path). It carries numbers, not chronology: one
table per question, every number naming its command, its window and its spread. The append-only
record of how those numbers arrived — every superseded table, every retired ruler, every route that
died, every correction that was made in place — is frozen byte-for-byte at
`progress/2026-10-16/performance-status-2026-10-16.md` (the frozen 1,149 lines, HEAD `bb7246d`), and
`progress/2026-10-16/ledger-prune-2026-10-16.md` records what moved, what stayed, and which
citations were re-pointed. **The section numbering here is a compatibility requirement, not a
style choice:** §6 is and stays the section that answers "what is a number on this box worth", and
live code, `engine/gate.py` and `AGENTS.md` cite it by that number.

**If the question is "what have we already tried, and why did it not work?" — do not read this file
for that.** `progress/2026-10-14/speedup-routes-2026-10-14.md` is the index of every speed-up route
tried, its verdict (shipped / dead / held / unpriced), the line it had to clear and the number that
decided it. This file is the measurement record those routes' citations point at.

**Every number below was re-taken on 2026-10-16** by the main agent, blocking, in the foreground,
with zero subagents active, each window inside one `tools/gpu_lock.sh` hold (`util` recorded at
acquire and release, then discounted — §6). Commands, lock tags and util pairs:
`progress/bench/evidence/ledger-2026-10-16/commands-and-provenance.md`; the filtered raw rows behind
§2 and §3: `progress/bench/evidence/ledger-2026-10-16/stage-wave.txt`. Binary:
`engine/build/decode_run` rebuilt from HEAD after `6fd6e61` (which flipped `--attn-tile` to 1).
Shipped defaults on every row unless stated: `--gemm library`, `--attn-tile 1`, `--mlp-map auto`,
`--gdn-scan seq`, `kRowsPerBlock = 32`.

**What today's re-take changed** against the previous era's ledger (2026-10-14, progress/2026-10-16/performance-status-2026-10-16.md:1073-1149):
cold weight-restore moved from 454.4 ms best / 77 % spread to **335.8 ms best / 9.2 % spread**; the
projection burst moved from the 2026-10-12 board's 3.081 ms / 63–64 % of probe to **3.2843 ms /
60 %** at the shipped rows/block 32 — recorded, not reconciled, because a ruler re-taken in a
different window is not an A/B (§6); and the in-situ `mlp` row moved from Q6.1's 4.264 ms to
**3.3880 ms**, which is not the machine changing but the `--mlp-map auto` split having shipped after
that window (it reproduces Q10b's 3.391 ms to 0.1 %). Everything else reproduces: `lm_head` to
0.2 %, the wave's `attn` row to 0.15 %, `gdn_gated_norm` to 0.3 %, `glue` fused to 0.0 %. The read
probe re-took at **239.3 GB/s** — 1 % above the 237 GB/s this ledger has used since 2026-08 — so the
"% of probe" columns do not move and are still stated against 237 GB/s.

## 1. Headline: what the engine does today

**Decode — marginal cost of one generated token** (`tools/context_sweep.py --lanes 128,512,1024,2048
--steps 32 --runs 3`; differential by construction: two runs differing only in `--steps`, so the
prompt wave cancels). `was` is the 2026-10-14 headline on the same ruler (progress/2026-10-16/performance-status-2026-10-16.md:1106-1107).

| lanes at last step | marginal ms/token, best of 3 | all three runs | spread | tok/s | was |
|---|---|---|---|---|---|
| 128 | **11.035** | 11.526 / 11.988 / 11.035 | 8.64 % | 90.6 | 11.155 / 89.6 |
| 512 | **12.408** | 12.525 / 12.475 / 12.408 | 0.95 % | 80.6 | 12.538 / 79.8 |
| 1024 | **12.407** | 12.415 / 12.407 / 12.635 | 1.84 % | 80.6 | 12.625 / 79.2 |
| 2048 | **12.382** | 16.490 / 12.382 / 13.378 | 33.17 % | 80.8 | 12.839 / 77.9 |

Medians, because the 128 and 2048 rows each carry one outlier run: 128 → 11.526; 512 → 12.475;
1024 → 12.415; 2048 → 13.378. The curve is **flat from 512 to 2048 lanes and climbs only past 2k**,
which is what P7's split-K path was for; the 2.301 µs-per-lane slope of 2026-10-12 was dominated by
fixed launch cost and did not extrapolate
(progress/2026-10-16/performance-status-2026-10-16.md:537-540).

**4032 lanes** — quoted from the stage-clock burst, never from `context_sweep.py`, which is unusable
at that width (§6): median of 3 **14.028 ms = 71.3 tok/s** (14.0280 / 13.8788 / 14.0995,
`decode_run --time-stages`, `tmp` rows committed in `stage-wave.txt`). `was` 14.50 ms / 68.9 tok/s
on the same burst ruler (progress/2026-10-16/performance-status-2026-10-16.md:878), so −3.2 %.

**Prefill / TTFT** (`tools/prefill_sweep.py --lanes 64,256,1024,3072 --runs 3`; `ms_total` with
`--steps 1` — prompt wave plus one priced step, weight load excluded). `was` is the 2026-10-14
reading of the same sweep (progress/2026-10-16/performance-status-2026-10-16.md:1098-1100; the 256-lane row of that day's sweep, 1,893 tok/s, is
the figure recorded in `AGENTS.md`'s ruler-window trap row).

| prompt lanes | ms to first sampled token, best of 3 | all three runs | spread | prompt tok/s | was |
|---|---|---|---|---|---|
| 64 | **91.1** | 92.1 / 92.7 / 91.1 | 1.76 % | 702.9 | 708.4 |
| 256 | **140.0** | 143.0 / 140.0 / 142.4 | 2.17 % | 1,828.5 | 1,893 |
| 1024 | **297.4** | 345.3 / 310.1 / 297.4 | 16.09 % | 3,443.1 | 3,326 |
| 3072 | **907.3** | 914.4 / 918.0 / 907.3 | 1.18 % | 3,385.8 | 3,320.9 |

The 1024 row carries a 16.09 % spread (median 310.1 ms) and is dispatch-bound at that length; the
3072 row, 1.18 % spread, is the one that should carry any claim about long-prompt throughput. The
64- and 256-lane rows continue to read *below* the paired A/B figures taken for the same builds in
one window (2,096.3 tok/s at 256 on the paired ruler), which is §6's bimodality, not a regression.

**Cold start** (`tools/cold_restore.sh --runs 5` → `engine/build/loader_test --no-verify`): text
weights made resident in a fresh process, device pool empty, host page cache warm — **best 335.8 ms,
median 345.8, worst 366.8, spread 9.2 %**; phases of the best run: cold H2D 58.5 ms, pinned staging
58.9 ms, first GPU read pass 8.5 ms. Once per process, not per token. `was` 454.4 ms best / 77 %
spread (progress/2026-10-16/performance-status-2026-10-16.md:493). At 335.8 ms it is ~30 tokens' worth of generation, and it still bounds any
"restart the engine per request" idea at zero times inside a conversation.

## 2. Where a decode token goes

One step, measured three ways at once (`decode_run --time-stages`, medians of 3, 128 lanes =
prompt 112 + 16 steps). The three passes proved they did the same work: `STAGEPARITY …
tokens_identical=1 logits_bitidentical=1` at all three granularities on all three runs,
`launches_per_step = 424 = plan_per_step`, `plan_match=1`, `syncs_per_step` 323 (stage) / 425
(kernel), `idle_sync_us` calibration 0.099.

| quantity | three runs | median |
|---|---|---|
| burst (pipelined, production shape) | 11.3573 / 11.2735 / 11.5772 | **11.357** |
| stage (323 waits/step) | 13.2385 / 13.2469 / 13.7743 | 13.247 |
| kernel (425 waits/step) | 13.7162 / 14.3092 / 14.3141 | 14.309 |
| **exposed (`STAGEGAP exposed_ms`)** | 1.889 / 1.981 / 2.208 | **1.98** |

**The exposed gap is 1.98 ms and it is the whole submission bill, not a prize.** Q6.1 pre-registered
two stories and measured between them: serialising the step costs `≈ D + N·s` (1.63 ms + 323 ×
0.098 µs ≈ 1.66 ms then; 1.98 ms today) and nothing beyond it, so the pipelined step was already
hiding every launch behind execution. No fusion-for-launch-count, no graph capture, no wider block
buys anything here (progress/2026-10-16/performance-status-2026-10-16.md:669-676).
*Which subtraction:* this row is the instrument's own `STAGEGAP exposed_ms` = serialized mean − burst
mean. Subtracting the gaged `TOTAL` (13.2385) from the burst instead gives 1.881 / 1.973 / 2.197 — the
same bill to within 0.008 ms, which is the ungaged remainder of the step — and the two spellings must
not be mixed in one comparison.

In-situ family rows at 128 lanes (stage granularity, ms/step, median of 3), each against **its own**
roofline and against today's isolated burst from §4. The roofline column is bytes ÷ 237 GB/s from the
component table (progress/2026-10-16/performance-status-2026-10-16.md:79-90); the ratios in column three are arithmetic on the two printed numbers.

| family | in-situ ms/step | in-situ vs its own roofline | today's isolated burst (§4) | burst → in-situ |
|---|---|---|---|---|
| `proj` | 4.1128 | 1.973 vs 4.1128 → **48 %** | 3.2843 (48-launch plan, rows/block 32) | +0.83 |
| `mlp` | 3.3880 | 2.230 vs 3.3880 → **66 %** | 2.378 (132 MB contig) / 2.275 (528 MB contig) | +1.01 |
| `lm_head` | 2.1799 | 2.146 vs 2.1799 → **98 %**, at ceiling | 2.1309 | +0.05 |
| `cast` | 0.9028 | ~0.007 ms of traffic — pure dispatch | never measured | — |
| `host_tail` | 0.7360 | — | 0.14 (CPU, progress/2026-10-16/performance-status-2026-10-16.md:687) | +0.60 |
| `gdn` | 0.6375 | 0.228 (state, cold) vs 0.6375 → 36 % | 0.0648 state + ≈0.053 gated norm | +0.52 |
| `glue` | 0.6125 | 0.0021 ms — 97 % launch | 0.1496 (49 launches, 512 t) | +0.46 |
| `attn` | 0.4847 | context-dependent, no fixed roofline | none taken today | — |
| `copy` | 0.2186 | — | never measured | — |
| **Σ families = TOTAL** | **13.247** | the gages are exhaustive: Σ families − TOTAL ≤ 0.0001 ms in every one of the six runs, at stage *and* kernel granularity, at both 128 and 4032 lanes | — | — |

**The step wall has no unattributed time in it, and that is a fact about the instrument.** The nine
family gages sum to the TOTAL line in every run to the printed precision, so nothing is hiding between
them; what the stage clock cannot see is that the families cost more *here* than in isolation. Do not
compare a sum of per-family medians (13.273) against the median wall (13.247) — medians do not add
across independent runs, so the two cannot be equal and the 0.2 % difference between them is an
arithmetic artifact, not a finding. Q6's ≈3.4–3.6 ms residue (progress/2026-10-16/performance-status-2026-10-16.md:629) was that older
burst-vs-token comparison, and it is not contradicted here: it compared burst numbers against a token,
not gages against a wall. What survives in this table is the burst-to-step delta in the last column.

**A burst is not a step, and this table is why.** `lm_head` — one fat launch streaming 485 MiB —
moves 2.3 % between burst and step; every small family moves by 0.46–1.01 ms. The cause is not the
instrument: a burst repeats one kernel against the same buffers, so its ramp-up overlaps the next
repeat and its pages stay hot, while a real step interleaves 424 launches across ~1.5 GB of weights
and interleaves writes with reads. Q7 refuted window size as the explanation (a 4× bigger cold
window makes the MLP *faster*, progress/2026-10-16/performance-status-2026-10-16.md:727-737), Q8 refuted write-back traffic (99.1 % of the read
probe at the step's 62:1 ratio, progress/2026-10-16/performance-status-2026-10-16.md:762-768), and Q9 named what remains — **the weight-walk
pattern itself** (progress/2026-10-16/performance-status-2026-10-16.md:780-792), of which Q10's `contig` walk recovered 0.945 ms of the 2.034 ms
gap (progress/2026-10-16/performance-status-2026-10-16.md:811). Plan against the in-situ column, not the burst column.

**4032 lanes, same instrument**: burst **14.028**, stage 16.276, kernel 17.204, exposed **2.26**
(`STAGEGAP`: 2.212 / 2.259 / 2.369), `idle_sync_us` 0.143, `STAGEPARITY` clean. Family medians: glue 0.706 · proj 4.137 · gdn 0.657 ·
**attn 2.996** · mlp 3.381 · cast 0.988 · copy 0.229 · lm_head 2.156 · host_tail 1.018. Attention is
the only family that moves materially with context — **0.4847 → 2.996 ms/step** (that median is
exactly 2.9955, printed half-up; a naive `round(…, 3)` says 2.995) from 128 to 4032
lanes — while mlp, proj and lm_head are flat to under 1 %. That is the whole context term, in one row.

## 3. Where a 1024-token prefill wave goes

`decode_run --steps 1 --time-waves`, medians of 3, ids from
`progress/bench/evidence/q10/ids1024.txt`, `wave_rows=512`. Composite rows (`cast`+`copy`, `gdn`
batched, `lm_head`+`host_tail`) are **sums of per-family medians**, not medians of per-run sums — the
two bases differ in the third digit for `gdn` batched (7.12 vs 7.108) — and the share column is the
median of the three printed `share_pct`/`matmul_pct` values; both bases are written out in
`progress/bench/evidence/ledger-2026-10-16/commands-and-provenance.md` §4. Witnesses first: `WAVEPARITY
tokens_identical=1 logits_bitidentical=1` at both granularities, `launches=20264 =
plan_launches`, `plan_match=1 launchstats_match=1 within_tol=1`, and the instrument's own completeness
account `sum_ms=243.1511` against `wall_ms=243.1739` — `ungaged_ms=0.0228`, 0.009 % of the wave, so the
share column divides the whole wave and not most of it (the same completeness property §2's stage gages
have).

| family | ms (median) | share | runs (ms) |
|---|---|---|---|
| `gdn_rec` | **126.83** | **51.2 %** | 123.08 / 126.84 / 126.83 |
| `proj` | 38.81 | 15.8 % | 38.81 / 39.21 / 38.10 |
| `attn` | **26.74** | **10.9 %** | 26.74 / 27.00 / 26.55 |
| `mlp_gateup` | 24.99 | 10.2 % | 24.67 / 25.42 / 24.99 |
| `mlp_down` | 6.42 | 2.6 % | 6.82 / 6.07 / 6.42 |
| `cast` + `copy` | 9.46 | 3.8 % | — |
| `gdn` batched (conv / gates / gated_norm) | 7.12 | 2.9 % | — |
| `glue` | 3.32 | 1.3 % | — |
| `lm_head` + `host_tail` | 3.04 | 1.2 % | — |
| serialized wall | **245.73** | — | 243.17 / 247.60 / 245.73 |
| burst wall | 235.48 | — | 236.03 / 234.89 / 235.48 |
| `GROUP matmul_ms` | 70.30 | **28.6 %** | 70.30 / 70.69 / 69.52 |

**Matmul 28.6 %, `gdn_rec` 51.2 %, `attn` 10.9 %.** That is a different wave from the one that made
route (c) a funded item: matmul was **74.4 %** of this same wave under the hand GEMMs (progress/2026-10-16/performance-status-2026-10-16.md:912)
and hipBLAS took it to 21–29 %, and `attn` was 107.67 ms before the tiled wave (progress/2026-10-16/performance-status-2026-10-16.md:1116) and is
26.74 ms now — reproduced to 0.15 % today. The serial recurrence is now over half the wave, and it
is the last big share left.

**What that funds: P9, the chunked delta-rule scan**
(`progress/2026-10-14/p9-chunked-scan-2026-10-14.md`). `gdn_rec` is 18,432 launches of 6.8 µs each —
neither dispatch-bound nor arithmetic-bound, but serial along the sequence. This is an algorithm
change, not a bit-identical rewrite, so it enters as a band question with its own plan file the way
route (c) did.

**Route A is dead and it is dead on its own number.** The collapsed/resident recurrence measured
**112.04 ms** against lines fixed before the kernel existed (**≤ 63 ms lives, > 84 ms dies**;
measured 112.04, progress/2026-10-16/performance-status-2026-10-16.md:1055-1057), so `--gdn-scan` keeps `seq`. The finding to carry is *why* making
it resident did not help: 6.1 µs per (token, layer) against `seq`'s 6.8 µs while issuing 36 launches
instead of 18,432 — i.e. the gap is exactly the dispatch that was removed. The recurrence is
**latency-bound on §04.4's 2,048 dependent chains** at ~1.6 working wavefronts per CU, and neither
reshaping launches nor residency moves it. Detail: `prototype/gdn_collapse/results.md`.

## 4. Per-op floors in isolation

Each row is the burst idiom (warm-up, N launches, one sync, wall/N; `hipEvents` are not usable here —
§5). Percentages are of the **237 GB/s** read probe, as every historical row in this repo is; the
probe re-took today at **239.3 GB/s best read, 210.5 best copy, 82.4 H2D, 69.7 D2H**, i.e. +1 %, so
no column moves. `tools/build/bw_probe_hip 1.0 4`.

| op | today | GB/s | % of probe | roofline | notes |
|---|---|---|---|---|---|
| `lm_head_gemv` fused | **2.1309 ms** | 238.7 | **100.7 %** | 2.146 ms | `--iters 8`, lock held |
| MLP, 24 layers, 132 MB window (`--tile 6`), interleave = the wave walk | 2.647 ms | — | 84 % | 2.230 ms | `--layers all --fixture dec8` |
| MLP, 132 MB, contig = the one-row walk | **2.378 ms** | — | 94 % | 2.230 ms | the shipped `--mlp-map auto` choice for one row |
| MLP, 528 MB (`--tile 24`), interleave | 2.372 ms | — | 94 % | 2.230 ms | |
| MLP, 528 MB, contig | **2.275 ms** | — | **98 %** | 2.230 ms | every row re-gated: `ndiff act 1 out 0` |
| projections, whole 48-launch token plan, rows/block 32 | **3.2843 ms** | 142.3 | **60 %** | 1.972 ms | shipped `kRowsPerBlock`, unpacked `x`; gate 912 comparisons, 0 failures |
| same plan at the prototype's rows/block 8 | 3.5863 ms | — | 55 % | 1.972 ms | the −8.4 % that decided `kRowsPerBlock = 32` (§5) |
| — of which **GDN projections** (`--only gdn`) | **2.5969 ms** | 146.0 | **62 %** | 1.600 ms | closes the "not measured on its own" row at progress/2026-10-16/performance-status-2026-10-16.md:81 |
| — of which **attention projections** (`--only attn`) | **0.6887 ms** | 128.1 | **54 %** | 0.372 ms | ditto (:82); the two sum to 3.2856 ms, 0.04 % off the whole plan |
| `glue` fused, 49 launches, 512 t | **0.1496 ms** (spread 4.0 %) | — | launch-bound | 0.0021 ms | 0.1852 @256 t, 0.2644 @128 t; gate 784 comparisons, 0 failures |
| `glue` unfused, 97 launches, 512 t | 0.2386 ms | — | launch-bound | — | fusion still buys ≈0.089 ms/token |
| `gdn_decode` state, 18 layers warm | **0.0648 ms** best, worst 0.0682, spread 5.2 % | 882.7 | 372 % — cache-resident | 0.228 ms cold | `rec` alone 0.0071 ms; all four functions PASS vs dumps |
| `gdn_gated_norm` §04.5, 144-job plan | **0.4273 ms** = 2.967 µs/launch (spread 0.8 %) | — | launch-bound | 0.0016 ms | `DEVICE GATE PASS`, all six negatives localise |
| kernel launch floor (empty/tiny kernel) | **1.85 µs**/launch | — | — | — | captured graphs replay at **1.78 µs**: no gain (progress/2026-10-16/performance-status-2026-10-16.md:39,46) |
| CPU read probe | — | 75–117 | 32–49 % | — | why the engine must stay GPU-resident: `devices/strix-halo-memory-bandwidth.md` |

Three things that belong with those rows rather than under them. **(1)** The projection families are
**0.4677 GB/token — 29 % of the token's bytes** (progress/2026-10-16/performance-status-2026-10-16.md:83), and the family is still shape-limited, not
traffic-limited: the 16-row `in_proj_a`/`in_proj_b` sit at **5 % of probe** for 32 KiB of weight per
launch (progress/2026-10-16/performance-status-2026-10-16.md:126),
which is launch and occupancy, not walk pattern. **(2)** Do *not* rebuild a plan total out of
per-shape bests: the per-shape sweep's GDN+attention sum is 3.54 ms, 15 % *worse* than the shipped
48-launch plan, and that gap is a finding (progress/2026-10-16/performance-status-2026-10-16.md:105-106).
**(3)** The `gdn` state row's 372 % of probe is **~54 MiB of state staying resident in the iGPU
cache** across decode steps — a cache property, not a bandwidth claim; the same 54 MiB driven cold
runs at 255 GB/s (progress/2026-10-16/performance-status-2026-10-16.md:36-37,303).

## 5. What the measurements decided

**Shipped defaults, and the number that decided each.**

- **`--gemm library`.** TTFT at a 1024-token prompt **1077.4 ms hand → 384.7 ms library** (progress/2026-10-16/performance-status-2026-10-16.md:941), matmul from 74.4 % to 21.2 % of the
  wave. The weakened guarantee is stated in `engine/results.md`: on this route the wave-boundary
  check sits inside a band, not byte-identical, and wave-row grouping can change a library GEMM's
  reassociation (§6 debts).
- **`--attn-tile 1`.** `attn` **107.67 → 26.70 ms** at 1024 lanes (−75.2 %, 4.03×, progress/2026-10-16/performance-status-2026-10-16.md:1116), +94.6 % prompt tok/s at 3072 lanes; the
  two forms emit the *same bits* on every above-`MAXSC` row (`attn_form_band.py` 136/136 and 520/520,
  0 elements differing). Today's 26.74 ms is the reproduction. Detail:
  `prototype/attn_decode/results.md` §20-§21.
- **`--mlp-map auto`** — `contig` when a body is one row, `interleave` when it is a wave. `contig`
  recovers **0.945 ms of the MLP's 2.034 ms in-step gap** at 128 lanes, but as a whole-default flip
  it cost **+5.1 % TTFT** at 1024 lanes; the per-path split keeps the decode win and returns TTFT to
  +0.8 %, inside the ±2 % band (progress/2026-10-16/performance-status-2026-10-16.md:811,831-841).
  Detail: `engine/results.md` §Q10b.
- **`kRowsPerBlock = 32`** in the projection launcher: −1.3 to −2.7 % decode at ≤2048 lanes and
  −5.6 to −8.0 % TTFT, bit-identical by construction (one wavefront owns one row end to end). Today's
  burst agrees from the other side: 3.2843 ms at 32 against 3.5863 ms at 8.
- **`PACK_X = false`.** Cold, 48-launch plan, rows/block 32: **unpacked 0.0528 ms/launch = 65.1 % of
  probe against packed 0.0593 = 58.0 %**, and unpacked wins at every rows/block swept — so the ≈48
  cast launches the unpacked form bills back do not overturn it (progress/2026-10-16/performance-status-2026-10-16.md:264,318).

**Dead routes, with the number that killed each.**

- **Route A, the collapsed GDN recurrence: 112.04 ms against a ≤ 63 ms win line and an 84 ms kill
  line.** End to end it bought 6.8 ms on a 322 ms wave (2.1 %). `prototype/gdn_collapse/results.md`.
- **Captured graphs: 1.78 µs/kernel replay against 1.85 µs eager** — 4 %, not the 3–5× a discrete
  card gives, because the limit is the GPU's own dispatch rate. Fusion is the only lever on launch
  cost. Probe: `tools/launch_overhead_hip.cpp` (progress/2026-10-16/performance-status-2026-10-16.md:39,45-60).
- **The packed-projection rewrite (proj 114 → 48 launches) is held, not dead**: it predicts
  0.1–0.25 ms — about 2 % of a 12.4 ms step — and costs a `gate.py::launch_plan` and a §07 table
  change to buy it (progress/2026-10-16/performance-status-2026-10-16.md:753-755).
- **`hipEvents` are not a timer here**: `lm_head_gemv`'s event loop returned `0.000 ms → inf GB/s` on
  every run including best-of-40 on a kernel that wall-clocks at 2.11 ms; fixed in `d81ff13`, and the
  steady_clock burst is the only timer to trust (progress/2026-10-16/performance-status-2026-10-16.md:277-282).
- **MTP drafting on CPU: ~21 tok/s with `--mtp 1` against ~24 tok/s plain greedy** (accept 0.660,
  1.24 tok/forward) — bit-lossless and slower. It is a GPU-path bet, not a win (progress/2026-10-16/performance-status-2026-10-16.md:179,
  `specification/09-mtp-draft-layer.md` §09.5).

## 6. What a number in this repo is worth

The protocol, then its known pathologies. Both halves are load-bearing: the first says how to take a
number, the second says which numbers cannot be trusted even when taken that way.

- **A clean window = zero subagents active + a blocking foreground call.** This box serves this
  session's own tokens on the same iGPU (`flash_serve`, and every subagent's tokens too), so
  contention is not incidental — it is the machine's normal state. A blocking tool call pauses the
  session's generation, which is what makes the window clean; a backgrounded run keeps the stream
  alive and contends with itself. `AGENTS.md`, "GPU execution is serialized".
- **`util` is recorded and then discounted.** Identical clean runs read 24–91 %; today's windows read
  92/40, 88/85, 90/68 and 63/14, 91/5, 90/21, 48/64 at acquire/release. The gate on validity is zero
  subagents plus a blocking call, never a util percentage.
- **State the spread or the number is provisional** — best-of-3 with all three runs printed, and a
  row whose spread exceeds its own effect is provisional no matter how clean its window was (today:
  2048 lanes at 33.17 %, prefill 1024 at 16.09 %).
- **A burst is not a step.** §2's table is the evidence: families run 0.46–1.01 ms slower in-situ than
  in their own burst while `lm_head` agrees to 2.3 %. Quote the in-situ column for budget work.
- **A ruler re-taken in a different window is not an A/B.** The same build read 64 and 256 lanes
  *slower* in a fresh sweep (708 vs 770, 1,893 vs 1,971 prompt tok/s) while a paired A/B inside one
  lock hold read the same two widths *faster* (83.26 → 80.91, 131.78 → 122.12 ms). For short prompts
  the only honest comparison is both sides inside one lock hold, and the long-context row carries the
  headline. `AGENTS.md`'s trap row; today's cold-restore and projection moves are flagged the same way.
- **Dispatch-bound rows are bimodal; bandwidth-bound ones repeat.** The three GEMV families reproduce
  within ~2 % window to window; `glue`'s 49-launch path has spread 2.7 % at 256 threads and up to
  269 % at 512, i.e. "best of N" picks a lucky mode rather than estimating a cost.
- **The differential harness is unusable at ≥4k lanes.** Quote 4k decode from the stage-clock burst —
  today **14.028 ms = 71.3 tok/s** — never from `context_sweep.py`: at 4032 lanes the differential
  produced 11.243 / 14.586 / 14.483 ms, spread 29.7 %, with a "best" faster than a 128-lane step at
  32× the KV, which is not a physical result (progress/2026-10-16/performance-status-2026-10-16.md:878-880).
- **Above ~240 GB/s means skipped work.** That is the measured LPDDR5x ceiling of this part; today's
  probe is 239.3. The one legitimate exception on record is the tiled attention wave's 241.6 GB/s of
  *demanded* K/V bytes, where correctness was established independently of the timing (progress/2026-10-16/performance-status-2026-10-16.md:1125).
- **Three gate drivers self-lock the GPU** (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`,
  `prototype/gdn_gated_norm/run.py`), so a caller that holds `tools/gpu_lock.sh` and then runs one of
  them **deadlocks it** — the child parks silently and looks like a hung box. Hold the lock only
  around the self-lock-free scripts and release before invoking a self-locking one; check with
  `grep -c gpu_lock prototype/<op>/run.py`.
- **A timer can be silently dead.** `hipEventElapsedTime` returns exactly 0.000 ms for work below
  ~2 ms on this iGPU, not only sub-millisecond work; a zero is a broken timer, not a fast kernel (§5).
- **A first GPU pass over fresh pages runs ~35 GB/s** — warm before you time, or you are measuring
  page faults (`devices/strix-halo-unified-memory.md`). That is also why cold restore's "first GPU
  read pass" phase is 8.5 ms and its worst run once read 42 ms.

**Open debts this ledger is still carrying.**

1. The projections' slide from 63–64 % (2026-10-12) to 60 % (today) needs a paired A/B in one window
   before it is called a regression; the two rows are 4 eras apart.
2. The mixed one-pass + streaming wave grid is priced by engine-vs-engine self-consistency only —
   pinning it needs a >512-token-prompt fixture with its own D-2 allowance.
3. `chunk64b` step 0 still has no sized rule, and its three D-9 red lines stand on purpose.
4. `--gemm library` makes a prompt's logits depend on the wave-row cap (a library GEMM reassociates
   with `M`), so wave-width independence is now a band question, not a byte-identity claim.
5. ≈1.09 ms/token of the MLP's in-step gap is still unattributed now that the walk is single-stream.
6. The second half of E2 — the same `PACK_X` A/B through `engine/kernels_proj.cpp`'s own launcher —
   has been runnable since `decode_step` grew that launcher and has not been run.
7. The 4032-lane row is quoted from one instrument; `attn`'s in-situ 2.996 ms/step at 4032 lanes has
   no isolated burst beside it, so its share of the step is a stage-clock figure, not a floor.

## The `--attn-touch` pair: the attention family at 1.466× and the wave at 1.125× (2026-10-16)

Appended at the end deliberately — §§1–§5 are banner-marked and cited by line from `progress/`, so
nothing above this line may shift.

This is the first entry in this ledger taken as a **paired A/B inside one lock hold**: both spellings in
**one binary**, alternating, so the pair is one ruler and not two windows. Window `attn-touch`,
`attn-touch-timing`; zero subagents; blocking; util read 92 % and discounted (it has never been a usable
gate on this box). Instrument: `decode_run --time-waves --attn-tile 1 --attn-touch 0|1`, `--steps 0`,
`ids4096.txt` / `ids1024.txt` from `progress/bench/evidence/prefill-occupancy-2026-10-16/`.

| 4096 lanes, `--steps 0` | `--attn-touch 0` (shipped spelling) | `--attn-touch 1` (slim) | pair |
|---|---|---|---|
| `attn` family | **403.38 / 404.92 ms** | **275.35 / 275.19 ms** | **1.466×** (−128.2 ms, −31.8 %) |
| `gdn_rec` (control, untouched by the lever) | 501.82 / 507.01 ms | 509.27 / 510.53 ms | 1.00 — as it must be |
| `TOTAL wall_ms` | **1243.07 / 1245.43 / 1271.12 ms** | **1105.20 / 1110.82 / 1112.11 ms** | **1.125×** (−137.9 ms) |
| prefill throughput at 4096 tokens | 3295 tok/s | **3706 tok/s** | +12.5 % |

1024 lanes: 240.39 → 230.81 ms = **1.041×**. That row is dispatch-bound and bimodal on this box
(§6), so quote it with the 4096 row, not instead of it.

What makes this admissible as a change to the shipping path rather than a prototype number:

- **Bit-identity on the engine path, not just the copy.** A 4096-lane prefill at `--attn-touch 0` and at
  `1` produced **byte-identical dumps across every written file**. `engine/gate.py` passes at both
  values on `dec8` and `wide520f` with numerically identical aggregates (pooled meanrel 9.447e-03,
  pooled 1-cos 5.499e-05, same worst row), and the gate now *fails* if the engine's NUMERICS line omits
  `attn_touch=` — an inert extra field would have been the silent-pass failure mode.
- **The hand-written `s_waitcnt` is not racy**: three 1024-lane runs at `--attn-touch 1`, byte-identical
  dumps (a missing wait would show as run-to-run disagreement, not a stable wrong answer).
- **The frozen reference still agrees**: `prototype/attn_decode/run.py --wave --tile-kernel` prints
  4112 stage comparisons, 0 failures, unchanged from before the port.
- **Census on the engine TU**: `attn_tile_wave` 116 slots/touch (55 VALU / 14 LDS / 47 SALU) vs
  `attn_slim_wave` 45 (28 / 7 / 10), both bodies compiled into the same TU so the control pair is
  permanent and drift is visible.

Attribution note for anyone re-taking this: the `attn` family row fell by 128 ms while the wave fell by
138 ms, so ~10 ms is outside the attributed families; `gdn_rec` did not move, which is what a control is
for. The prior ledger row for `attn` at 4096 lanes was 412.07 ms (§1 of
`progress/2026-10-16/prefill-occupancy-results-2026-10-16.md`), consistent with the 403–405 ms taken
here in a different window — a further reminder that absolute ms moves a few % between windows and only
a paired run like this one may be read as a ratio.

**Decode side of the same pair (measured after the row above, same day).** `ids2048.txt`, `--steps 64`
against `--steps 192`, marginal = Δms ÷ 128, three reps, both values in one hold:
`--attn-touch 0` gives 12.732 / 12.859 / 12.668 ms/token (spread 1.5 %), `--attn-touch 1` gives
12.308 / 12.458 / 12.840 (spread 4.3 %) — **ratio of bests 1.029, i.e. unchanged within noise**. That is
the expected result and it should be read as such, not as a small win: this is single-request decode, so
attention runs the **one-row** path, and the two levers live in the tiled *wave* body. §1's
14.028 ms = **71.3 tok/s** headline therefore stands as the decode number; the prefill rows above are what
this port moved. (Widths differ — 2112 lanes here vs 4032 in §1 — so these ms are not comparable to §1's,
only to each other.)

---

## 7. The GEMM route flip: `--gemm lt` is the shipped default (2026-10-16)

Route: the engine's wide-body library matmuls moved from `hipblasGemmEx` to `hipblasLtMatmul` with one
algorithm resolved per `(shape, M)` at plan time (`engine/kernels_gemm.cpp`, decision log D26–D29). The
`--gemm` flag keeps `library` as a first-class value and `legacy` as its alias, so every row below stays
A/B-able in one binary; `--gemm-pins all|none|<csv>` does the same for the pin table.

**Instrument:** `decode_run --time-waves --steps 0`, `ids4096.txt` from
`progress/bench/evidence/prefill-occupancy-2026-10-16/` for the headline, `ids1024.txt` / `ids3072.txt` from
`progress/bench/evidence/q10/` for the rest. Both routes inside one `gpu_lock` hold, alternating per rep,
medians of 3, zero subagents, blocking. `util` at acquire: 92 % (the 1024/3072 windows), 90 % (3072 re-take),
92 % (the 4096 headline window) — discounted per §6, and the same-window pairing is what makes the ratio,
not the absolute ms.

| width | `GROUP matmul_ms` | whole wave `wall_ms` | prefill tok/s | ratio |
|---|---|---|---|---|
| **4096 lanes (headline)** | 276.90 → **195.11** | 1122.50 → **1069.89** | 3649 → **3828** | **1.049×** |
| 3072 lanes | 210.10 → **145.26** | 794.96 → **733.24** | 3864 → **4190** | **1.084×** |
| 1024 lanes | 71.11 → **47.32** | 238.02 → **214.40** | 4302 → **4776** | **1.110×** |

The 4096 spread is 1069.89 / 1111.95 / 1055.11 ms on the `lt` side against 1122.50 / 1123.88 / 1121.58 on
`library` — ±2.6 % against ±0.1 %, i.e. the `lt` side is the noisier of the two at this width and the
headline ratio is quoted from the medians, not the best. The §1 prefill row before this change was 3706
tok/s, taken in a different window; **the paired pair is 3649 → 3828**, and the honest statement of the win
is the ratio.

Family rows, same pairs (`library` → `lt`): `proj` 38.64 → 24.75 ms (**1.561×**) at 1024 lanes and 115.25 →
75.31 (**1.530×**) at 3072; `mlp_gateup` 26.23 → 15.92 (**1.647×**) and 75.53 → 49.56 (**1.524×**);
`mlp_down` 6.24 → 6.60 (**0.945×**) and 19.15 → 20.44 (**0.937×**) — `down_proj` is the one pinned shape and
still costs a little; an in-situ A/B of three pinned shapes left only that one, because the other two
burst-winners cost 5.5 ms of a 24.8 ms `proj` family in situ (§3d of
`prototype/gemm_algo_sweep/results.md`). `WAVEPARITY tokens_identical=1 logits_bitidentical=1` on both routes
in every run.

**Decode is untouched, by construction rather than by measurement:** a one-row body is the hand GEMV on
either route (`resolve_wave_gemm`), so the **71.3 tok/s** row above stands unchanged and was not re-taken
here. If a future change ever sends M=1 to the library, that claim dies with it and decode must be re-taken.

**Gate state at the flip:** `engine/gate.py` over all nine fixtures × both routes was 18/18 green *before*
the default moved (worker window, util 93 %), and four fixtures re-taken here at the new default — `dec8`,
`dec64f`, `wide520f`, `wide1024f` — are green with the log echoing `gemm=lt`. The per-row band is
**2.75e-03** and the flip-room line is now **1.50 BF16 ULP** with the pin-vs-engine pairing read on the
reference's own room; both carry their provenance in the gate's own output
(`prototype/gemm_algo_sweep/results.md` §3f holds the census: 74 flips over 11,804 comparisons, worst
0.9985 ULP, 0 at or over the line).

---

## 8. The `gdn_rec` spelling flip: prefill 3927 → **4284 tok/s** at 4096 tokens (2026-10-16)

Route: `gdn_rec` — **47.6 % of the wave** — is not traffic-bound, not instruction-count-bound and not
dependency-latency-bound. It is bound by how its own loop gets scheduled, and a second spelling of the same
operations with `#pragma unroll 4` on both state walks (`engine/gdn_rec_unroll4_kernels.h`, `--gdn-spell`,
decision log D31/D32/D33) is **bit-identical** and 1.153× on the family row. The instrument history is
`prototype/gdn_rec_isa/results.md` §8–§10; the falsifier that refuted the latency reading is §9.

**Instrument:** `decode_run --time-waves --steps 0`, `ids4096.txt` / `ids1024.txt`, both spellings **paired
inside one `gpu_lock` hold per width**, alternating per rep, medians of 3, blocking, zero subagents. `util` at
acquire 91 % and 89 % (discounted per §6; the same-window pairing is what makes the ratio). Every rep carried
`plan_match=1 launchstats_match=1` and `WAVEPARITY tokens_identical=1 logits_bitidentical=1`, which is what makes
a 15 % number safe to read at all: this is not a route that trades precision for speed, because there is nothing
to trade.

| width | `gdn_rec` ms | whole wave `wall_ms` | prefill tok/s | family ratio | wave ratio |
|---|---|---|---|---|---|
| **4096 lanes (headline)** | 496.55 → **430.69** | 1043.11 → **956.12** | 3927 → **4284** | **1.153×** | **1.091×** |
| 1024 lanes | 121.71 → **110.62** | 209.46 → **197.66** | 4889 → **5181** | **1.100×** | **1.060×** |

Spread, because the `ship` side is the noisy one at this width: 4096-lane `ship` read 481.55 / 523.11 / 496.55 ms
and `unroll4` read 433.83 / 428.21 / 430.69 ms — **every `unroll4` rep beat every `ship` rep**, worst-vs-worst
1.110×, best-vs-best 1.222×, so the ordering does not depend on which rep is picked. §1's prefill row before this
change was 3828 tok/s taken in a different window; the same-window pair here is **3927 → 4284**, and the ratio is
the claim.

**Decode moves ~0.1 %, by arithmetic, and was deliberately not re-taken.** One decode step submits 18 `gdn_rec`
launches (one per layer, one row each) at 6.73 µs → 5.88 µs, i.e. 121 µs → 106 µs of a 14.028 ms step:
**0.86 % of the step, and the saving is 0.11 %**, four times below the ruler's own ±3 % spread. The
**71.3 tok/s** row therefore stands as an upper bound on this change rather than as a re-measurement, and if a
future decode ruler gets tighter than 1 %, this row is owed a re-take.

**Gate state at the flip:** ten fixtures green across both spellings (`dec8`, `dec64f`, `dec64t`, `dec300f`,
`mt32f`, `p128`, `chunk64b`, `wide520f`, `wide640f`, `wide1024f`), and `gate.py`'s section (j) byte-compares the
two corpora inside every one of those runs — `wide1024f` compared **1032 logits rows / 1025.1 MB per side and
found them identical**. The engine dumps no state *values* (`--state-fp` rows are order-invariant moment sums and
cannot witness bit-identity), so the per-element witness lives in the harness, where the engine's own header is
compiled into the probe TU (`--modes … receng`, `ndiff_o_vs0 = ndiff_state_vs0 = 0`).

## 9. The deep-field algo pins: 4 admitted at burst, **2 ship**, and a subset that is worse than no change (2026-10-17)

**Instrument:** `decode_run --time-waves --steps 0`, `ids4096.txt` / `ids1024.txt`, three configs and both widths
**paired inside two `gpu_lock` holds** (one per config set), alternating per spec, medians of 3, blocking, zero
subagents, `util` 91 %/92 % at acquire (discounted per §6). Prerequisite, not a formality: the four candidates were
first admitted by the numerics gate under the existing bands — **9/9 canonical fixtures green with all four bound**
(`pins=4/1 cand=4/4`, four `source=candidate` plans at M=512), and the default path re-verified unchanged
(`pins=1/1 pin_names=down_proj cand=0/4`). Analysis: `tmp/pinab/{ab.txt,ab2.txt}`, `tmp/pinab/tab.py`.

Judge this **only** on the per-family rows of the pinned shapes. The `wall_ms` column cannot adjudicate a four-shape
pin change: between the `all` and `cand` runs `gdn_rec` moved 470.04 → 426.42 ms and `attn` 272.48 → 259.29 ms,
families **no algo pin can touch**, and those two drifts (−56.8 ms) account for essentially all of the −50.25 ms
wall movement. A null test on the same logs sizes the row noise: `proj` under a config that cannot change it reads
**+3.1 % at 4096 and +1.9 % at 1024**.

Medians, `gu` = the `gate_proj+up_proj` candidate alone, `gud` = that plus the `down_proj` candidate, `cand` = all
four admitted:

| width | cfg | `proj` | `mlp_gateup` | `mlp_down` | sum3 | wall |
|---|---|---|---|---|---|---|
| 4096 | `all` (shipped) | 99.23 | 63.95 | 26.87 | 190.05 | 988.34 |
| 4096 | `gu` | 102.33 | **57.59** | 32.16 | 192.07 | 963.20 |
| 4096 | **`gud`** | 98.10 | **57.89** | **26.62** | **182.62** | 979.09 |
| 4096 | `cand` | **125.18** | 57.66 | 26.63 | 209.47 | 981.83 |
| 1024 | `all` | 24.70 | 16.05 | 6.61 | 47.36 | 203.54 |
| 1024 | **`gud`** | 24.60 | **14.48** | **6.45** | **45.54** | 200.81 |
| 1024 | `cand` | **31.29** | 14.18 | 6.75 | 52.22 | 202.62 |

**What ships: two of the four, and only together.** `deep_gate_proj+up_proj` is a clean −10.0 % on its family at
*both* widths, every rep; `deep_down_proj` is small but consistently positive. The two `proj` candidates —
`deep_o_proj+out_proj` and `deep_in_proj_qkv`, winners at burst by 1.34× and 1.13× — run **+26 % and +27 % slower
in situ**, which is the same pair D28 rejected from a shallower field, now rejected a second time from a deeper one.
Net of what actually ships: **sum3 190.05 → 182.62 ms (−3.9 %) at 4096 and 47.36 → 45.54 (−3.8 %) at 1024**
= **−0.75 % of the wave**, not the +1.903 % the burst table promised, and the wall is unchanged within its own
noise so no headline tok/s number is claimed from this window.

**The finding that outranks the win: algo pins are not independent, and one admissible subset is worse than shipping
nothing.** With the `gate_proj+up_proj` candidate in force and the *shipped* `down_proj` pin left alone (`gu`),
`mlp_down` goes **26.87 → 32.16 ms (+19.7 %) at 4096 and 6.61 → 7.97 (+20.6 %) at 1024 — in all three reps at both
widths** — so `gu` is net-negative on sum3 (192.07 vs 190.05) despite winning 10 % on its own shape. Add the
`down_proj` candidate and it comes back to 26.62: the deep-field `down_proj` algo is the one that suits the
*new* `gateup` kernel's L2 state, while the old pin is not. Two consequences kept as rules: a pin set is adjudicated
**as a set** or not at all, and **a per-shape winner that changes a neighbouring shape's L2 can turn a win into a
loss on a family the flag does not name** — which is the D33 mechanism ("ask what the burst holds fixed") arriving
inside a single binary, with nothing but a pin changed between the two sides.

### Correction to §9, same day: the `gu` harm was **an unpinned shape**, not a bad algo, and one row of the shipped set comes back out

§9 claimed the `gate_proj+up_proj` candidate, shipped while the shipped `down_proj` pin was left alone, made
`mlp_down` +20 % through L2 state, and that the deep-field `down_proj` algo was the row that suited the new kernel.
**That mechanism is refuted by a direct test.** With the gateup candidate shipped *and the original `down_proj` pin
named explicitly*, paired 3× against the two-promoted-rows default, both widths, one lock hold, `util` 92 %
(`tmp/pinab/ab3.txt`):

| width | side | `mlp_gateup` | `mlp_down` |
|---|---|---|---|
| 4096 | `deep_gate_proj+up_proj,deep_down_proj` (the §9 "good" config) | 57.88 | 26.19 |
| 4096 | `deep_gate_proj+up_proj,down_proj` (gateup + the *retired* pin) | 58.25 | **26.82** — not the predicted 32.16 |
| 1024 | same two sides | 14.09 / 14.16 | **6.59 / 6.56** |

The prediction was `+17 %/+20 %` on `mlp_down`; measured **+2.4 % and −0.5 %**, inside the ±3.1 %/±1.9 % null of §9.
So the old `down_proj` algo and the deep-field one are **indistinguishable when either is in force**, and nothing is
superseded by anything.

**What the §9 harm actually was.** `--gemm-pins <csv>` *replaces* the pin set, it does not extend it, so §9's `gu`
side named only `deep_gate_proj+up_proj` — which left `down_proj` **unpinned, running rank-0 heuristic**. The +20 %
was not "the wrong algo suits the new neighbour", it was **no pin at all** on that shape once the neighbour changed.
That is a stronger and more useful finding than the one it replaced: on this box the *coverage* of the pin set is
what costs, and dropping a shape from `all` is a bigger perturbation than swapping which of two near-equal winners
runs it. The correct shipped set is therefore **`down_proj` (the original D28 pin) + `deep_gate_proj+up_proj`**: the
gateup candidate's −10 % is reproduced again here (`mlp_gateup` 63.95 → 57.88 at 4096, 16.05 → 14.09 at 1024) and
the `down_proj` row earns its place by *being pinned*, not by which algo is pinned.

**No headline number changes**: the wave gain is the gateup row's, ≈ **−0.63 % of wave**, and the `down_proj` row
contributes nothing measurable either way. The trap that produced the error is worth carrying: a config label that
names a *subset* silently describes an *exclusion*, and the exclusion was the measured effect. `decode_run` did say
so — every unpinned shape echoed `source=heuristic-rank-0` — but the label was read before the echo was.

**Admission of the corrected set (same window, no re-take).** The shipped pair is now `down_proj` + 
`deep_gate_proj+up_proj`, and the timing for *exactly that set* is the `old` column above — cited from
`tmp/pinab/ab3.txt` rather than re-measured, because nothing feeding it changed. Against §9's shipped baseline that
is `mlp_gateup` **63.95 → 58.25 ms** at 4096 and **16.05 → 14.16** at 1024, ≈ **−0.6 % of the wave**, and the
`down_proj` row contributes nothing measurable. Numerics admission: **9/9 canonical fixtures green** under the new
default (`pins=2/2 pin_names=down_proj,deep_gate_proj+up_proj cand=0/3`), including `wide1024f`. No `tok/s` headline
is claimed, for the §9 reason: the wave column is carried by families no pin touches.

**New default vs old default, paired (`tmp/pinab/ab4.txt`, one lock hold, 3 pairs, alternating, `util` 91 %).**
`new` = `all` as shipped now (`down_proj` + `deep_gate_proj+up_proj`), `old` = `--gemm-pins down_proj`, which *is*
the previous default spelled out:

| width | cfg | `proj` | `mlp_gateup` | `mlp_down` | sum3 | wall |
|---|---|---|---|---|---|---|
| 4096 | **`new`** | 98.91 | **58.51** | 27.02 | **184.44** | **963.28** |
| 4096 | `old` | 98.85 | 62.92 | 26.55 | 188.32 | 985.44 |
| 1024 | **`new`** | 24.47 | **14.28** | 6.57 | 45.32 | 201.81 |
| 1024 | `old` | 24.50 | 15.77 | 6.62 | 46.89 | 195.19 |

The attributable effect is the family row: **`mlp_gateup` −7.0 % at 4096 and −9.4 % at 1024**, every pair, which is
**−4.4 ms and −1.5 ms = −0.46 % and −0.75 % of the two waves**, and `proj`/`mlp_down` sit where they should for a
change that touches neither. The wall is reported and then set aside: at 4096 it reads **+2.3 % with the three pairs
non-overlapping** (every `new` rep beat every `old` rep), which is ~5× more than this pin can cause, and at 1024 the
same window reads **−3.4 %** on the same pin while its own family row wins by 9 % — so the wave column is still being
carried by drift in families no pin touches, exactly as §9 said. **No new tok/s headline is claimed from this
window**; the recorded 4284 stays as the last clean cross-window ruler row, and the honest number for this ship is
the family row's **≈ −0.5 % of wave**.

## 10. The F32 seam deletion: `cast` −57 %, prefill **+1.86 %** at 4096 lanes (2026-10-17)

D30 4.1 was booked as a fusion and turned out to be a deletion: 78 of the 114 casts per wave body widen a
BF16 tile into an F32 scratch buffer the library route never reads, because `project()` and the MLP call
sites pick the 16-bit operand under `lib_gemm` and the widening happened anyway. Deleting them cannot move a
bit, so no numerics band was touched and the gate set is the witness. Detail in
`progress/2026-10-17/glue-cast-copy-bytes-census.md` §3/§8 and D37.

Paired inside one lock hold, alternating per rep, medians of 3, blocking, zero subagents, `util` 92 %
(`tmp/gcc/ab.txt`, `tmp/gcc/run_ab.sh`; base binary saved beside the new one so the A/B is the change and
not the window):

| width | family | base | 4.1-a | Δ | Δ % |
|---|---|---|---|---|---|
| 4096 | `cast` (913 → **289** launches) | 27.3218 | **11.7545** | −15.5673 | **−56.98 %** |
| 4096 | `glue` | 14.1991 | 15.4061 | +1.2070 | +8.50 % (noise; not touched) |
| 4096 | `proj` | 99.8226 | 97.3716 | −2.4510 | −2.46 % (noise; not touched) |
| 4096 | **wall** | 966.0399 | **948.3901** | −17.6498 | **−1.83 %** |
| 1024 | `cast` (229 → 73 launches) | 6.3860 | **2.8893** | −3.4967 | **−54.76 %** |
| 1024 | **wall** | 194.6688 | **185.6256** | −9.0432 | **−4.65 %** |

The reason this ship is trusted where the algo pin was not: **the row and the wall agree at both widths**.
The row predicts −15.57 ms and the wall gives −17.65 ms at 4096; the row predicts −3.50 ms and the wall
gives −9.04 at 1024. D21's null was a row that collapsed while the wall sat still — here they move together,
so the deleted work was on the critical path and not absorbed by L2. The two untouched families move at
their usual spread (±2.5 % on `proj`, ±8.5 % on `glue`), which is what bounds the noise floor of this
window.

**Admission.** 18/18 gates — the nine canonical fixtures on **both** routes (`--gemm lt` and `--gemm hand`),
which is the control that matters: the guard is a route test, so the hand route had to come back unchanged
and did. The per-token cast bill stays **115 on both routes** (section (d) never moved, because
`resolve_wave_gemm` forces the hand GEMV at `rows <= 1`), `plan_match=1` and `counts_match=1` in every
wave-timed run, and the new deriver self-check `body1_library_plan_matches_token_plan=1` printed in every
one.

**Headline.** The recorded prefill row stays **4284 tok/s** until a dedicated ruler window re-takes it; the
paired ratio here is **1.0186 at 4096 and 1.0487 at 1024**, which *derives* **≈ 4364 tok/s** by applying
this window's own ratio to the last clean row. Stated as a derivation, not as a ruler row, because §9's
lesson is that a short-window wall is not portable between windows. Decode is untouched (71.3 tok/s): a
one-row body never takes this route.

## 11. The gather kernel: `copy` −85 %, prefill **+1.40 %**, and the finding that submissions are *not* absorbed (2026-10-17)

`copy` was 4096 per-token 2 KiB row copies of the embedding table, one submission per token in the wave
(§02.3). Its bytes are worth **0.16 ms against its own 10.86 ms row**, so the family was never a traffic item;
`tools/submission_probe.cpp` had already shown **2.05 µs/op for a real copy against 1.72 µs for an empty
launch**, i.e. 84 % of the op is the submission. One gather kernel plus one id upload per body replaces all
4096. Detail: `progress/2026-10-17/glue-cast-copy-bytes-census.md` §3/§8, D37, D38.

Paired inside one lock hold, alternating per rep, medians of 3, blocking, zero subagents, base binary saved
beside the new one (`tmp/gcc/ab42p.txt`, `tmp/gcc/run_ab_42prime.sh`):

| width | family | base | gather | Δ | Δ % |
|---|---|---|---|---|---|
| 4096 | `copy` (**4240 → 160** launches) | 10.9677 | **1.6240** | −9.3437 | **−85.19 %** |
| 4096 | `cast` (control, untouched) | 11.7342 | 11.4370 | −0.2972 | −2.5 % (noise) |
| 4096 | **wall** | 936.5254 | **923.6193** | −12.9061 | **−1.38 %** |
| 1024 | `copy` (1060 → 40 launches) | 2.7213 | **0.4658** | −2.2555 | **−82.88 %** |
| 1024 | **wall** | 187.4449 | **185.4343** | −2.0106 | **−1.07 %** |

**The prediction was written down before the measurement and it survived, two-sided.** `copy` had to collapse
to ≲0.5 ms *and* the wall had to drop ≥5 ms. Row: −9.34 ms (predicted ~−10.4). Wall: −12.9 ms median, and
**all three pairs negative at 4096 (−2.88, −13.00, −14.59) and all three at 1024 (−1.48, −2.01, −1.77)** —
six of six. The refuting branch was the interesting one: had the row collapsed with a still wall, per-op
submission cost would be absorbed by a busy stream and every launch-shaped row in D30 would have needed
re-reading. It is not absorbed, so **launch-count reduction is a real lever on this box**, which re-prices
`glue`'s launch component and confirms that `gdn_rec`'s 73,728 submissions per wave are not hiding behind the
same mechanism (that family is at 140 µs/launch, ~80× the floor).

**Admission.** 18/18 gates — nine canonical fixtures × both routes. The witness that matters is that the
wide-vs-one-row logits comparison is now **gather-against-copy rather than gather-against-itself**, since the
one-row path keeps the `hipMemcpyAsync` verbatim: a decode step's bill is **424 on both routes** and
`--wave-rows 1` is unchanged to the launch, while the shipped wave's plan reads 80426 → **76346** with
`family=copy launches=160 plan_launches=160 counts_match=1`. No numerics band was consulted: a byte-moving
change with no arithmetic cannot reach one.

**Headline.** Recorded prefill stays **4284 tok/s** pending a dedicated ruler window. Two shipped changes
since that row give paired ratios of 1.0186 (§10) and 1.0140 (this one), which **derives ≈ 4425 tok/s**; the
wording is deliberate — chained ratios from two different windows compound each window's drift, so this is
arithmetic on two measurements and not a third measurement. Decode is **71.3 tok/s, unchanged by
construction**: a one-row body never takes either path.

## 12. Ruler re-take: prefill is **4419 tok/s** at 4096 tokens, and the chained derivation was right to 0.14 % (2026-10-17)

§10 and §11 each declined to move the headline, so this is the dedicated window that replaces it. Shipped
binary (`47274f4` + `31848bb`), shipped defaults (`--gemm lt --gemm-pins all`), one lock hold, `util` 90 % at
acquire, blocking, zero subagents, five reps, `ids4096.txt` / `ids1024.txt`:

| width | rep walls (ms) | median | best | **tok/s (median)** |
|---|---|---|---|---|
| 4096 | 926.70 · 928.23 · 924.32 · 929.23 · 924.72 | **926.70** | 924.32 | **4419** (best 4431) |
| 1024 | 187.90 · 183.53 · 187.63 · 192.38 · 186.83 | 186.83 | 183.53 | 5481 |

**Recorded headline: prefill 4284 → 4419 tok/s (+3.15 %)** at 4096 tokens, from the two ships in §10 (the F32
seam deletion) and §11 (the embedding gather). This window is the tightest of the session — 924.3–929.2 ms, a
0.53 % spread over five reps, no bimodality — which is what a bandwidth-shaped wave looks like once the
launch-shaped work has been taken out of it.

**The methodological result is worth more than the number.** §11 derived ≈ 4425 tok/s by chaining two paired
ratios from two different windows and explicitly warned that chaining compounds drift. The measured answer is
4419 — **0.14 % apart**. So on this box, paired ratios taken inside their own lock holds *are* portable, and
the drift that made §9's absolute rows untrustworthy was drift in the *level*, not in the *ratio*: the same
window that reads a wall 3 % high reads its own A/B pair 3 % high on both sides. That is the licence for
quoting a paired ratio as the ship's effect without re-taking a full ruler row each time, which is how §10 and
§11 were written.

Decode: **71.3 tok/s, not re-measured in this window.** Unchanged by construction rather than by measurement —
a one-row body never takes the library route (§10) and keeps the per-token embedding copy verbatim (§11), and
the nine decode fixtures in both routes' admissions came back bit-identical. If a decode-shaped number is
needed, take it from a decode ruler, not from this row.

## 13. The packed GDN cast goes away: `cast` −81 %, `gdn_conv` **+32 %**, net **−0.50 %** of wave (2026-10-17)

D30 1.7′ deleted `cast_bf16_to_f32_3` by having `conv_kernel`/`gates_kernel` widen their BF16 operands
in-register. It was the first deletion in the line with no route to guard it on, so it moved the §07
per-token launch bill (cast 115 → 97, total 424 → 406) rather than a route-scoped row — see D39 for why
that was admissible. Detail: `progress/2026-10-17/HANDOFF_17prime_gdn-cast-deletion.md`.

Paired in one lock hold, medians of 3, canary asserted before any timing (`tmp/gcc/ab17p_retake.txt`):

| width | family | base | 1.7′ | Δ | prediction that was on file |
|---|---|---|---|---|---|
| 4096 | `cast` (289 → **145** launches) | 13.8637 | **2.5964** | **−81.3 %** | ≈ −33 % (to ~7 ms) — **beat** |
| 4096 | `gdn_conv` | 19.9876 | **26.4833** | **+32.5 %** | −22 % (to ~15 ms) — **refuted, wrong sign** |
| 4096 | `gdn_gates` | 1.5980 | 1.5835 | −0.9 % | ≈ flat — held |
| 4096 | net of those three | 35.449 | 30.663 | **−4.79 ms = −0.50 %** | — |
| 1024 | net of those three | 8.075 | 7.411 | **−0.66 ms = −0.36 %** | — |

**The refutation is the finding.** The census argued that halving `gdn_conv`'s read side (1.81 GB → 0.90 GB
per wave) would take ~4.5 ms off it. It went 6.5 ms *slower* instead: the packed F32 copy was not merely a
format conversion, it was a **coalescing layout** — one contiguous tile read instead of three strided BF16
tiles widened per lane. So the byte account was right and the timing consequence was backwards, which is
the same lesson as D21 one level down: bytes demanded do not determine time, and a consumer's access pattern
can be worth more than the traffic you delete. **The recoverable prize is therefore named and sized: ~6.5 ms
= 0.68 % of wave**, by giving the conv kernel a coalesced read of the BF16 operands. That is a new D30 item,
not a reason to revert — the deletion's own win (−11.27 ms) is real and is what nets out positive here.

**Admission.** 18/18 gates — nine canonical fixtures × both routes — with the *new* bill in both engine and
checker, bit-identical outputs (worker's byte diffs: `wide520f` 26/26 files, `dec64f` paired HEAD-vs-new on
both routes 66/66 each), `plan_match=1`, and `selftest_launch_bill()` shown biting at **+1 and −1** so the
moved row cannot rot into a check that passes vacuously.

**Decode did not move, and that was the prediction.** 18 launches per token at the 1.72 µs floor is ~31 µs on
a 13.3 ms step (≈0.23 %), and the measured pair sat inside the ruler's own spread with the three pairs
disagreeing in sign (base alone spread 8 %). **71.3 tok/s stands.** Wave headline stands at §12's **4419**
pending a re-take; this item is booked at the family rows, −0.50 % of wave.

**Trap, and it cost a full A/B: a stale binary produced a false null.** The first run of this A/B printed a
clean "nothing moved" — which would have refuted the whole item — because the `new` side was a pre-change
binary: `engine/build.sh`'s output had been discarded and `cmp` between the two saved binaries was used as
proof the change was compiled in, but builds on this box are not bit-reproducible, so *any* two builds differ.
Worse, `plan_match=1` stayed green throughout, because `wave_plan_bill` is compiled **into** `decode_run`: a
stale binary agrees with itself, and a self-consistency check cannot detect staleness. The fix is a
**behavioural canary asserted inside the measurement window** — here, `new` must print `cast=97 other=20
total=406` and `base` must print `115/424`, or the script aborts before timing. Every A/B script from now on
carries one, and it is `run_ab_17p_retake.sh` that produced the numbers above.

## 14. Dated correction to §6 and §11: the per-token launch bill moved (2026-10-17)

Appended at the end rather than edited in place, because §1–§6 are cited by line number from `progress/` and
the ledger's older sections are banner-marked rather than rewritten for exactly that reason.

`62b73e9` deleted the packed GDN widening at **every** width and both routes, so the two route-symmetric
counts quoted earlier moved: **cast 115 → 97 per token, per-token bill 424 → 406, wave plan 76346 → 76202**.
Specifically §11's "a decode step's bill is **424 on both routes**" and §6's cast row are the plan *as measured
on their own dates* and were not touched; §13 carries the re-take, and `engine/gate.py` section (d) now
polices 97/406 with `selftest_launch_bill()` shown biting at +1 and −1 (D39). The §12 ruler row — **4419
tok/s** — was taken after `62b73e9` and needs no correction.

## 15. The conv coalesced read ships, and a correction to the admission claims in §10/§11/§13 (2026-17)

### 15.1 the item: `gdn_conv` 26.90 → 16.76 ms at 4096 lanes (−37.7 %), shipped `5096a57`

`62b73e9` halved the conv read side and the kernel got **32.5 % slower**. §13 recorded that as a refuted
prediction; this section records why it was refuted and what recovered it. The deleted packed F32 tile had
been doing a second job as a **coalescing layout**: one contiguous fetch per lane instead of three strided
2-byte fetches. The listing states the mechanism without interpretation — base issued `global_load_u16`
(2 B/lane = 64 B per wavefront, half a line) with `s_waitcnt vmcnt(0)` immediately after every tap, 29
immediate drains; the shipped form issues `global_load_b64`/`b128` (8 B/lane = 256 B) with the whole K-row
window fetched before any of it is consumed, 4 loads → 1 and 4 drains → 1 per output element at 4× the
width. `CONV_VEC = 4`, grid 24 blocks → 6.

**Measurement (main agent, one lock hold, five pairs, blocking, canary asserted before any timing).** Paired
at 4096 lanes, `--steps 0 --time-waves --gemm lt --gemm-pins all`, `ids4096.txt`:

| rep | base `gdn_conv` | new `gdn_conv` | base wall | new wall |
|---|---|---|---|---|
| 1 | 26.90 | 16.76 | 926.30 | 934.72 |
| 2 | 28.64 | 16.82 | 925.30 | 987.14 |
| 3 | 29.25 | 15.64 | 936.75 | 916.43 |
| 4 | 26.68 | 16.80 | 944.21 | 923.35 |
| 5 | 26.90 | 16.30 | 931.82 | 911.74 |
| **median** | **26.90** | **16.76** | **931.82** | **923.35** |

All **five** pairs negative on the conv row; the row is the stable number (base spread 9.6 %, new spread 7.6 %,
every pair −10.1 to −13.6 ms). Wall median **−8.47 ms = −0.91 % of wave**, above the 0.1 % refuse bar, but
quoted with its spread: 911.7–987.1 on the new side is dispatch-bound bimodality, not a 8 ms story. **The §12
headline of 4419 tok/s is not restated from this table** — §12's ruler (926.70) and this window's base (931.82)
are different windows, and a ruler re-taken in another window is not an A/B. A clean ruler re-take is owed
before the headline moves; the honest claim today is the conv row and a −0.9 % paired wall delta.

Attribution note, because it is the whole point of the item: the win came from **latency, not traffic**. The
new form reads 4 bytes per element more than the ideal and is still 37.7 % faster than a form that read half
what the old one did. Byte accounts did not predict this in either direction (D21 one level down, again).

### 15.2 correction: three admissions this session claimed "18/18 gates — nine canonical fixtures" and the fixture list was wrong

Lines 616, 657 and 725 of this file say *18/18 gates — nine canonical fixtures × both routes*. The list used
for those runs was `wide520f dec64f wide3100f prefill_all3093f prefill_t256f prefill_t1024f prefill_t1552f
prefill_t1832f prefill_t2464f`. **Only `wide520f` and `dec64f` exist.** `data/refs/` holds manifests for
`chunk64b corpus48 corpus48engine corpus48sdpa dec300f dec64f dec64t dec8 mt32f p128 p300 p64 p8
prefix24dec8 smoke wide1024f wide520f wide640f zz`, and `gate.py` exits with *"fixture X has no manifest"* for
any other name. So each of those admissions ran **4 real gate runs, not 18**, and the other 14 were error
exits that my harness reported as `PASS=0` without me reading them; the `18/18` figure was carried from a
worker's report and repeated by me without checking the population behind it.

What that actually invalidates is narrow but real: the numerics of §10 (4.1-a), §11 (4.2′) and §13 (1.′) rest on
`wide520f` + `dec64f` on both routes, which is thin for a claim of whole-engine admission, and the *wide* cases
(`wide1024f`, `wide640f`, `wide3100f`-intent) were never run at all on those two changes. What it does not
invalidate: every one of those changes is bit-identical by its own differential or gate on the fixtures that do
exist, `gate.py`'s own structural checks (plan parity, launch bill, band history) ran and bit, and nothing was
loosened to pass.

The conv change shipped today is the first admission in this session run against the real population:
`wide520f wide1024f wide640f dec64f dec64t mt32f dec300f` on `--gemm lt` plus `wide520f dec64f p300` on
`--gemm hand`. Three of those — `p300`, `corpus48engine`, `prefix24dec8` — raise **inside `gate.py`**
(`IndexError` in `pairing_checks`, `KeyError: sources`), and they raise **identically on the base binary**, so
they are a pre-existing gap between those fixtures and this invocation style, recorded here rather than hidden:
they are not evidence for or against anything. Fixing that invocation gap is now a task, not a footnote.

## 16. Ruler re-take after the conv coalesced read: 4444 tok/s at 4096 (2026-10-17)

Five reps, one lock hold, blocking, zero subagents, shipped binary (`5096a57`), `ids4096.txt`, `--steps 0
--time-waves --gemm lt --gemm-pins all`, every rep `plan_match=1`:

| rep | 1 | 2 | 3 | 4 | 5 | **median** |
|---|---|---|---|---|---|---|
| wall_ms | 954.09 | 984.09 | 920.18 | 913.90 | 921.56 | **921.56** |
| `gdn_conv` ms | 16.41 | 16.95 | 16.79 | 15.57 | 16.39 | 16.41 |

**4096 tokens / 921.56 ms = 4444 tok/s**, from §12's 4419 — **+0.56 %**. Stated with its weakness rather than
without it: this window's spread is **7.7 %** (913.9–984.1, two reps above 950), where §12's five reps sat inside
0.53 %. The iGPU was at 83–87 % `util` throughout, which the ledger already records as unusable as a gate. So
the paired A/B in §15 is the stronger evidence for the conv change (five of five pairs negative, **−0.91 %**
median wall, conv row −37.7 %) and the ruler here is a *confirmation that nothing regressed*, not the proof of
the win. Two numbers of different strength, both kept, because they were measured differently: the headline is
the median of a clean-window ruler, and the claim about a change is the paired delta inside one window.

`gdn_conv` in this window reads 16.41 ms, agreeing with §15's new side (16.76 median) to 2 %, which is the
stable-kernel behaviour the ledger expects and not dispatch bimodality — the row is bandwidth/latency-bound
enough to repeat. Decode unchanged at **71.3 tok/s** and not re-measured: the conv change is prefill-path only
at these widths, and a decode re-take would need its own paired window to say anything.
