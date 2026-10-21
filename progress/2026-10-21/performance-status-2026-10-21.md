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

**Every number below was re-taken by the main agent**, blocking, in the foreground (§1's decode and prefill rows on **2026-10-19** per §36; the rest on 2026-10-16),
with zero subagents active, each window inside one `tools/gpu_lock.sh` hold (`util` recorded at
acquire and release, then discounted — §6). Commands, lock tags and util pairs:
`progress/bench/evidence/ledger-2026-10-16/commands-and-provenance.md`; the filtered raw rows behind
§2 and §3: `progress/bench/evidence/ledger-2026-10-16/stage-wave.txt`. Binary: `engine/build/decode_run`
at HEAD (`2ac956d` for the §36 re-take, which is what §1's two tables now carry). Shipped defaults unless stated: `--gemm lt`, `--gemm-pins all`, `--attn-tile 1`,
`--mlp-map auto`, `--gdn-scan seq`, `--gdn-spell fused`, `--gdn-loop 512`, `kRowsPerBlock = 32`.

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
prompt wave cancels). `was` is the previous banner on the same ruler (§25's window, taken 2026-10-17);
the 2026-10-14 row behind that sits at (progress/2026-10-16/performance-status-2026-10-16.md:1106-1107).

| lanes at last step | marginal ms/token, best of 3 | all three runs | spread | tok/s | was |
|---|---|---|---|---|---|
| 128 | **11.475** | 11.475 / 11.660 / 11.703 | 1.99 % | 87.1 | 11.480 / 87.1 |
| 512 | **12.928** | 13.115 / 13.078 / 12.928 | 1.45 % | 77.4 | 12.764 / 78.3 |
| 1024 | **11.922** | 12.564 / 12.490 / 11.922 | 5.39 % | 83.9 | 12.128 / 82.5 |
| 2048 | **12.558** | 12.897 / 12.861 / 12.558 | 2.70 % | 79.6 | 12.487 / 80.1 |

Medians, because the spreads above are 1.4–5.4 % and one run per row can sit at either end: 128 →
11.660; 512 → 13.078; 1024 → 12.490; 2048 → 12.861. Nothing moved beyond its own spread: §30–§33 is wave-only.
which is what P7's split-K path was for; the 2.301 µs-per-lane slope of 2026-10-12 was dominated by
fixed launch cost and did not extrapolate (progress/2026-10-16/performance-status-2026-10-16.md:537-540).

**4032 lanes** — quoted from the stage-clock burst, never from `context_sweep.py`, which is unusable
at that width (§6): median of 3 **14.028 ms = 71.3 tok/s** (14.0280 / 13.8788 / 14.0995,
`decode_run --time-stages`, `tmp` rows committed in `stage-wave.txt`). `was` 14.50 ms / 68.9 tok/s
on the same burst ruler (progress/2026-10-16/performance-status-2026-10-16.md:878), so −3.2 %.

**Prefill / TTFT** (`tools/prefill_sweep.py --lanes 64,256,1024,3072 --runs 3`; `ms_total` with
`--steps 1` — prompt wave plus one priced step, weight load excluded). `was` is the previous banner
(§25's window, taken 2026-10-17); the 2026-10-14 reading of the same sweep sits at
progress/2026-10-16/performance-status-2026-10-16.md:1098-1100; the 256-lane row of that day's sweep, 1,893 tok/s, is
the figure recorded in `AGENTS.md`'s ruler-window trap row).

| prompt lanes | ms to first sampled token, best of 3 | all three runs | spread | prompt tok/s | was |
|---|---|---|---|---|---|
| 64 | **103.0** | 103.0 / 103.6 / 104.0 | 0.92 % | 621.2 | 104.7 / 611.2 |
| 256 | **121.7** | 121.7 / 124.4 / 122.9 | 2.22 % | 2,103.7 | 122.8 / 2,084.6 |
| 1024 | **221.3** | 222.6 / 223.8 / 221.3 | 1.15 % | 4,627.7 | 228.3 / 4,486.2 |
| 3072 | **520.8** | 528.8 / 528.2 / 520.8 | 1.55 % | **5,898.8** | 615.8 / 4,988.5 |

The row that moved is **3072**: 4,988.5 → **5,898.8 prompt tok/s, +18.3 %** at a 1.55 % spread, and it is
still the row that should carry any claim about long-prompt throughput (1024: +3.2 % at 1.15 %). The
64-lane row still reads *below* the paired A/B figures taken for the same builds in one window
(2,096.3 tok/s at 256 on the paired ruler), which is §6's bimodality, not a regression.

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

## 17. The lane ceiling moves to 4608, and it is not a speed item (2026-10-17)

`kAttnMaxSplit` 8 → 9 (`0f5dca8`), so the attention stage plans up to `512 × 9 = 4608` lanes where it threw
above 4096. **No timing claim is made for it, deliberately.** Three reps at 4096 lanes in one window read
**914.12 / 919.70 / 920.17 ms** with `plan_match=1`, median 919.70 against §16's 921.56 — a 0.2 % move inside
this box's noise, and the change only enlarges the partial buffer (32.25 → 36.28 MiB per wave), so there is
nothing to bank. The §16 headline of **4444 tok/s** stands; re-measuring a ruler to record a no-op is not a
result.

What this row *is* for: the geometry is now admissible. `wide4096f` — 4096 prompt + 8 generated = **4104
lanes**, which the launcher refused outright until today — PLANS and PASSES, with **all 4096 pinned rows
judged** inside floors already in force (worst cos deficit 4.3696e-04, worst meanrel 3.5902 %), and
`wide4096p` / `wide1024f` / `dec64f` stay green. Bit-identity below the old ceiling was proved twice over: the
CPU census in `tools/attn_ceiling_proof.py` (the cap reaches the allocation and the throw and nothing else;
three writers, the reader and the mirror share one address expression that never names the cap), and a direct
old-binary-vs-new comparison at 1024 lanes, **ndiff 0 in all four pairings**.

One instrument trap recorded here because it nearly produced a false finding: the first bit-identity A/B looked
like a mismatch, and the mismatch was mine — `md5sum dir/logits*.bin | md5sum` hashes the *path text* in
md5sum's output, so two different `--out` directories hash differently no matter what the tensors contain.
Element-level comparison is the only honest form of this check; an aggregate hash over a glob of two different
directories is not a comparison at all.

## 18. `rec2c` burst screen: 1.0268 against a 1.02 kill line, which is not a verdict (2026-10-17)

D42 class 4 named one thing worth measuring out of the two `gdn_rec` closures: `rec2c`, the fused-pair
prologue with compiler-lowered shuffles, bit-identical (`ndiff = 0`), above the refuse bar on paper
(predicted −0.93 % of wave) and never adjudicated in situ. Screened on the existing harness, one lock hold,
blocking, `--volume --burst 40 --reps 7`, four cases:

| case | `rec0` med ms/seq | `rec2c` med | ratio | pairs favouring `rec2c` |
|---|---|---|---|---|
| t128-carried-d0.002 | 0.8408 | 0.8152 | 1.0314 | 6/7 |
| t128-carried-d0.4 | 0.8411 | 0.8228 | 1.0222 | 5/7 |
| t128-zeros-d0.002 | 0.8317 | 0.8256 | **1.0074** | 4/7 |
| t128-zeros-d0.4 | 0.8493 | 0.8083 | 1.0507 | 5/7 |

**Median of case-medians 1.0268; best case 1.0507; worst case 1.0074.** An earlier 5-rep window read
1.0208 best-of and the harness itself called that case `INCONCLUSIVE`, so the honest summary is: the fused
prologue is *probably* a small win, and one of four cases sits under the kill line with the pair-count
coin-flipping there. That is not an admission, and no engine change follows from it yet.

The reason it cannot be adjudicated from this screen at all: the harness compares `rec0` against `rec2c`,
both pre-`unroll4` spellings, while the shipped kernel is the D33 unroll4 form. The fused prologue has never
existed **merged with** the unroll4 walks, and a merge can silently give the unroll back through register
pressure — which is precisely the class of thing a burst cannot see because the merged kernel is not in the
binary the burst ran. The gate in front of any port is a compile-only census of the merged listing, not another
burst window. Recorded here so the 1.0268 is not later quoted as a win: it is a screen of a kernel the engine
does not run.

## 19. `rec2c4` — the merged fused-prologue-on-unroll4 form, screened against the shipping spelling (2026-10-17)

D46's gate was a compile-only census; it passed (§`prototype/gdn_rec_isa/results.md` 11: merged form bills
**1,374** slots/token-step against shipping **1,406**, same 59 VGPRs, zero spill, loop B unchanged at 16×8).
Then the correctness gates ran on device and are green: `--selftest-lane cases=80660 bad=0`, `--selftest-mirror
cases=524304 bad=0`, volume `VOLUME_OK` for all three MODEs with identical byte accounts, and admission
**bit-identical** — `ndiff_o_vs0=0 ndiff_state_vs0=0` on all four cases. So the merged kernel is exact, which was
the open question in §18.

The burst, `--modes receng,rec2c4 --volume --burst 40 --reps 7`, both sides in **one binary alternating inside
every rep**, canary read at build time (`CANARY OK`, both bills pinned) before any launch:

| case | `receng` med ms/seq | `rec2c4` med | `receng/rec2c4` | pairs favouring `rec2c4` |
|---|---|---|---|---|
| t128-carried-d0.002 | 0.7100 | 0.7166 | **0.9908** | 4/7 |
| t128-carried-d0.4 | 0.7369 | 0.6930 | 1.0633 | 6/7 |
| t128-zeros-d0.002 | 0.7339 | 0.7012 | 1.0466 | 7/7 |
| t128-zeros-d0.4 | 0.7368 | 0.7110 | 1.0363 | 5/7 |

**Median of case-medians 1.0415, mean 1.0343, worst case 0.9908.** Three of four cases clear D25's 1.02 kill
line and the median clears it by more than the census predicted (1.018–1.019 from −32 slots at elasticity 0.8),
which is the §10 direction — on this kernel a spelling change has previously returned ~2.4× its instruction delta.
The spread is honest: `t128-carried-d0.002` is *adverse*, at 0.9908 with 4/7 pairs, so this is a small win with
one case going the wrong way, not a uniform one.

**What this does not decide.** A burst on a 128-token synthetic sequence is a screen, not the verdict: the claim
being priced is −0.85 % of a 4096-lane wave's `gdn_rec` family, and that is an in-situ number from
`decode_run --time-waves`, not a harness ratio. §18's warning still stands against anyone tempted to quote
1.0415 as the win — the instrument here is `receng` vs `rec2c4` inside one harness binary at `t=128`. The next
step is the engine port behind `--gdn-spell fused`, default unchanged, and a paired in-situ A/B inside one lock
hold; the family row decides.

## 20. `--gdn-spell fused` — paired in-situ A/B, and why the headline did not move (2026-10-17)

The engine port (§19's merged form) was measured with both sides **inside one lock hold, alternating every rep**,
canaries read in the same breath as the numbers (`CANARY 1 OK` on the engine TU with both bills pinned;
`gdn_spell=fused` echoed by **every** side's own log; `plan_match=1 launchstats_match=1` on every pass; an
unknown spelling refused, which is what proves the flag is not inert).

**The `gdn_rec` family row is the claim. Medians of 3:**

| width | `unroll4` ms | `fused` ms | paired median ratio | reps favouring `fused` | verdict vs 1.02 |
|---|---|---|---|---|---|
| **4096 lanes** | 455.83 | 428.22 | **1.0645** (reps 1.0200 / 1.0725 / 1.0645) | **3/3** | **clears** |
| 1024 lanes | 108.87 | 109.20 | 0.9909 (reps 0.9640 / 1.0252 / 0.9909) | 1/3 | does not clear |
| decode, ms/token | 12.7302 | 12.6237 | **1.0025** (1.0147 / 0.9950 / 1.0025) | 2/3 | flat |

Decode used the ledger's own instrument (`ids2048.txt`, `--steps 64` vs `--steps 192`, marginal = Δms ÷ 128),
both step counts inside the same hold; 12.62 ms/token is 79.2 tok/s at 2112 lanes and 12.73 is 78.6 — the two
sides are inside the spread this file documents for that instrument (1.5–4.3 %), so decode is **unchanged**, not
"slightly better".

**And the ruler did not move.** Re-taken with `fused` as the default, same box, zero subagents, 3 reps:
**919.17 / 923.36 / 925.22 ms → median 923.36 ms = 4,436 tok/s**, against §17's 921.56 ms = **4,444 tok/s**.
So the headline is **unchanged**, and it would be dishonest to quote the paired −2.96 % of wall (947.52 → 919.49
ms inside that one window) as a new headline. The reconciliation is the window, exactly as AGENTS warns: the
`unroll4` side of the paired window ran ~2.8 % *slower* than the historical ruler median, and today's `fused`
number (919–925) is about what this box reads at 4096 lanes in any window. Cross-window on the family row the
change is **442.55 → 430.89 ms = −2.6 %**, not the −6.5 % the paired window showed.

**What that means for the decision (D51).** The change is kept and the default moved: the paired instrument is
the correct one for attributing a delta, it favoured `fused` 3/3 at the width that carries the headline, and
−2.6 % cross-window still clears the refuse bar by ~26×. But **no headline improvement is claimed today** —
the honest statement is "the family row got ~3–6 % cheaper depending on which comparison you trust, and the
end-to-end number is the same 4,440 ± 20 tok/s it has been all session". Prefill stays **4,444 tok/s** as the
quoted headline until a ruler re-take in a clean window places it lower.

## 21. llama.cpp prefills this model **1.7–2.3× faster than we do** — measured in one hold (2026-10-17)

D48 run for real. `llama.cpp` @ `797de63`, built here `GGML_HIP=ON` / `gfx1151` only, model
`models/gguf/Qwen3.5-0.8B.f16.gguf` (f16, **unquantised**, 752.4 M params, arch read at load as
`qwen35 0.8B F16`, ROCm backend, all 24 layers offloaded). Every number below is from **one lock hold**, with
our own `--time-waves` re-taken in that same window, per §16's rule. `-b 2048` pinned so `-ub` is the only
thing moving; `kMaxWaveRows = 512` means **`-ub 512` is the matched row**.

| `-ub` | `-fa` | `-t` | pp512 | pp1024 | pp4096 | llama/ours at 4096 |
|---|---|---|---|---|---|---|
| 128 | on | 32 | 4,427 | 4,165 | 3,707 | **0.84×** |
| **512** | on | 32 | 9,627 | 9,487 | 8,290 | **1.87×** |
| 512 | off | 32 | 9,331 | 9,569 | 8,999 | **2.03×** |
| 2048 | on | 32 | 9,637 | 8,230 | 7,624 | **1.72×** |
| 512 | on | **4** | 9,678 | 10,054 | **10,273** | **2.32×** |

Ours, same window: **pp512 91.99 ms = 5,566 tok/s · pp1024 185.40 ms = 5,523 · pp4096 924.62 ms = 4,430**.
The user's estimate was ~9k and the prediction was right: llama.cpp does **8,290–10,273 tok/s** at 4096 tokens
against our 4,430.

Three things the table says beyond the headline:

- **Shape of the curve, not just height.** Ours loses 20 % from 512 to 4096 tokens (5,566 → 4,430); llama.cpp
  loses 14 % (9,627 → 8,290) and *gains* with `-t 4`. So a per-token-cost term of ours grows with context where
  theirs barely does — which points at the per-token serial scan and the quadratic attention term, not at the
  GEMMs, exactly as D47's roof arithmetic predicted.
- **`-ub` is their lever and it is enormous.** At `-ub 128` llama.cpp falls to 3,707 tok/s and is **slower than
  us** at 4096 tokens (0.84×). Any "they are faster" claim without its batch setting is meaningless, and ours at
  `-ub 512` is not their best case.
- **`-t 4` beating `-t 32` (10,273 vs 8,290, +24 %) is unexplained and is recorded as such.** Four CPU threads
  on a 32-thread box being the fastest setting smells like oversubscription hurting them, and it is the kind of
  row that must not be quietly dropped because it flatters the comparison.

**Ceiling cross-check (D47).** 10,273 tok/s × 1.0947 GFLOP/token = **11.25 TFLOP/s demanded = 48 %** of the
measured 23.3 TFLOP/s mix roof. We are at **4.85 TFLOP/s = 21 %**. So llama.cpp neither exceeds the roof nor
contradicts the ~21,000 tok/s bound — it *corroborates* it, and it demonstrates that well over 2× our current
rate is attainable on this silicon at this model by something other than GEMM work.

**Not comparable, stated rather than glossed.** llama-bench feeds synthetic token ids, we feed `ids4096.txt`;
both compute prompt-only (`-n 0` / `--steps 0`) and last-token logits only; llama.cpp's number includes its own
scheduler and graph build per batch and excludes load, as does ours; and their `-ub` is a tunable policy where
our 512 is a compile-time ceiling (`kMaxWaveRows`, raisable only with the lane-ceiling work of D45).

## 22. Where the 1.87× actually is — per-kernel trace of both engines at 4096 tokens (2026-10-17)

`rocprofv3 --kernel-trace --memory-copy-trace` on both engines in one hold (`tmp/gcc/run_d52_trace.sh`).
Traces inflate, and they inflate **in proportion to launch count**, which is the exact axis under study — so the
absolute traced ms are used as **upper bounds for llama.cpp and structural evidence only**, and our own family
ms come untraced from `--time-waves`. Pass counts inferred from arithmetic that agrees on both sides
(18 GDN layers × 8 chunks of 512 = 144; 6 attn layers × 8 = 48): llama.cpp's trace holds warmup + timed
(÷2), ours holds three passes (÷3).

| | **llama.cpp** | **ours** | |
|---|---|---|---|
| launches per token | **1.93** | **18.76** | 9.7× |
| GDN recurrence launches / layer / 4096 tok | **8** (512 tokens per launch) | **4,096** (one token per launch) | **512×** |
| attention launches / pass | 138 counted | 96 counted | **not identical** — this row read "48 \| 48 \| identical", which was inferred from `6 × 8`; §26 counts both from traces |
| GEMM launches / pass | 1,464 | 1,200 | comparable |
| GDN family time | **≤ 47.4 ms** | **428.2 ms** | we are **≥ 9.0× slower** |
| attention family time | **≤ 52.5 ms** | **263.2 ms** | we are **≥ 5.0× slower** |
| GEMM time | ~176 ms (traced/2) | ~154 ms (untraced §20) | comparable — both near the roof, as D47 said |

Two mechanisms, and they are *different* in kind:

- **GDN is a launch-structure gap, not a math gap.** llama.cpp runs `gated_delta_net_cuda<128,…>` **once per
  512-token chunk per layer** and walks the tokens *inside* the kernel. We relaunch per token — 73,728 launches
  for a 4096-token prompt, each carrying a 1.72 µs submission (≈**127 ms**, 13.8 % of the wave) plus its own
  prologue and drain. Looping N tokens inside one launch changes no arithmetic and no address: the recurrence is
  already sequential, so the same expression tree runs in the same order and the result is **bit-identical by
  construction** (the same reasoning D33 accepted for `#pragma unroll 4`). This is **not** P9's chunked
  reformulation — it keeps the scan, it removes the relaunching.
- **Attention is a kernel-shape gap at identical launch counts.** Same 48 launches, ≥5× the time. Theirs is
  `flash_attn_ext_f16<256, 256, 16, 4, …>` — a 256×256-tiled flash kernel; ours is `attn_slim_wave` at
  `MAXSC = 512` lanes per block. That is queue item **2.6 with external evidence attached**, and it is now the
  second-largest item on the board: 263 ms against ≤52 ms.

**Explicitly not claimed:** that llama.cpp's traced times equal its untraced times (they are upper bounds, which
makes both ≥ ratios conservative); that their `-t 4` setting was used here (it was the matched `-t 32` row); and
any per-family *rate* comparison for GEMMs — D47 already put both engines near the same roof, and the trace
agrees, which is what makes the two rows above worth acting on.

## 23. The D53 screen: one launch per body beats 128 launches per token by **1.58–1.64×**, bit-identically (2026-10-17)

`recloop` = the shipping `rec_fused_kernel` body, token index inside the kernel, same grid `<<<16,128>>>`, same
operands, same state buffer. Correctness ran first, on CPU and then on device, and all of it cleared:

* **Addressing** — `--selftest-loop`: host mirror of the looped mapping vs the per-token mirror, **39,444,480
  elements bit-compared** across spans {1, 2, 3, whole}, `bad=0`, `clean_mismatch=0`, and five injected loop
  defects (token index pinned, `o` stride halved, state restarted per token, `beta` pinned, `g` pinned) each
  caught **exactly** where their geometry predicts, with `wrong_catch=0` and `defects_unreachable=0`.
* **Arithmetic as text** — `check_recloop_transcription.py`: 33 core lines identical across `recloop`,
  `rec2c4` and the engine's `rec_fused_kernel`, including the `#pragma unroll 4` lines; its own `--selftest`
  catches all four mutations (hoisted `Q_SCALE`, reordered serial adds, dropped pragma, dropped store).
* **On device** — admission at spans **0, 1, 3, 16**: **4/4 cases bit-identical** every time
  (`ndiff_o_vs0=0 ndiff_state_vs0=0`, reruns clean), plus `VOLUME_OK` at spans 0 and 3 with the per-token-step
  byte account unchanged. So the §19 reassociation risk did not materialise: looping the token index moved no value.

The screen (`--modes rec2c4,recloop`, same binary, alternating inside every rep, build-time canaries on the
census bill and the resource pass — 75 VGPR / 40 SGPR / **0 spill**, occupancy unchanged at 16 waves/SIMD):

| span | case | per-token ms/seq | looped ms/seq | ratio |
|---|---|---|---|---|
| whole body (1 launch vs 128) | carried-d0.002 | 0.7342 | 0.4487 | **1.6363** |
| | carried-d0.4 | 0.6970 | 0.4402 | **1.5834** |
| | zeros-d0.002 | 0.6986 | 0.4387 | **1.5924** |
| | zeros-d0.4 | 0.6993 | 0.4383 | **1.5955** |
| 32 tokens/launch (4 launches) | four cases | 0.6955–0.7326 | 0.4608–0.4850 | **1.4713–1.5553** |

**All eight cases, 7/7 pairs each.** Per-token cost falls from 5.43–5.74 µs to **3.43–3.51 µs**, i.e. if the wave
behaves the same way, `gdn_rec` goes from 428 ms toward ~270 ms — **~17 % of the wave** — which is well outside
anything this file has recorded as a single-kernel win.

Two cautions that the ratio does not dissolve. First, this is a **128-token harness shape**: the real call will
walk 512 tokens in one launch, and the census says the loop's own addressing costs **+11.4 % slots** (1,374 →
1,530 — `salu.mov` 7→82, `loop.salu` 57→79, from recomputing `t*CH` per iteration), a cost the screen here
overwhelms but which may read differently at 512 iterations. Second, §10 is the standing precedent that a burst
winner can be an in-situ loser, so the claim is due from the paired **family row**, not from this table. Per the
screen's own bands (`>1.10` → port and adjudicate in situ), the port is authorised; the addressing-spelling
variant was deliberately **not** prototyped first, because the screen already answers the question it would have
answered, at a ratio far above the band.

## 24. The ported token-loop launch in situ: `gdn_rec` **449 → 266 ms**, a **1.63× family row**, bit-identical (2026-10-17)

Gate first, on the fixture the worker picked and for the reason it gave: `wide520f` is **520 prompt tokens = a
full 512 body plus a ragged 8**, the two cases the tail logic and `wave_plan_bill`'s derivation exist for. With
`--gdn-loop 512` section (j) runs a fourth side and **370 checks pass, 0 failures**, byte-comparing **524 logits
rows / 520.5 MB** of the looped body against `ship`. Parse-level refusals all fire (`--gdn-loop nope`,
`--gdn-loop 0`, and both inert pairs with `--gdn-scan collapse` and `--gdn-spell unroll4`), and the default side
echoes `gdn_loop=1`.

Then the paired in-situ A/B, medians of 3 reps per width, both sides inside one lock hold, alternating per rep:

| shape | base `gdn_rec` | looped | ratio | counted launches |
|---|---|---|---|---|
| 4096 lanes, 3 timed reps | 437.60 / 452.19 / 449.07 | 273.47 / 267.68 / 266.03 | **1.6002 / 1.6893 / 1.6881** | 73,728 → **144** |
| 1024 lanes, 3 reps | 108.55 / 115.07 / 109.34 | 68.51 / 66.59 / 67.16 | **1.5845 / 1.7281 / 1.6280** | 18,432 → **36** |

**Every one of the seven pairs favours the looped form; median across all pairs 1.6280, min 1.5845.** The
1024-lane worry recorded in §23 — two bodies instead of eight, a smaller share of a shorter wave — did not
materialise: 1024 wins as hard as 4096, so the prize is not a body-count effect and does not need to be reported
as one. Per-launch cost goes **5.94 µs → 1.90 ms per 512-token launch**, i.e. 3.70 µs/token against 5.94.

Canaries, all asserted before any timing: the flag is live and refuses garbage (a pre-port binary aborts
there); the **counted** `family=gdn_rec … launches=` falls **73,728 → 144** — behavioural, not the plan bill, per
the `17p` lesson; and **1,024/1,024 prefill logits rows are byte-identical** between the two forms, which is the
claim in its strongest form: no band was consulted because no value moved. Controls: `attn` issues **48 launches
on both sides** (the untouched family), and decode is **flat — median ratio 0.9999** over three paired
one-row-per-step reps, which is what the one-row-routes-to-per-token construction predicts and is now also a
measurement rather than only an argument.

**Attribution and its limits.** The family row drops ~183 ms of a ~923 ms wave, so the wall is due to fall by
roughly a fifth — but this section deliberately does not print a wall number: D51's discipline says a paired
window is an attribution, and the headline comes from its own clean-window re-take (§25). The known cost the
harness flagged is still on the books — the loop bills **+156 slots/token-step** (1,530 vs 1,374) from recomputing
`t*CH` — and it was paid here and still came out 1.63×, which is what makes the address-spelling variant
(increment six row pointers instead) a follow-up rather than a prerequisite.

## 25. The looped launch ships: ruler **763.49 ms = 5,364 tok/s**, and §1's banner re-taken (2026-10-17)

Default moved `--gdn-loop 1 → 512` on §24's evidence (engine option, `DecodeScratch`, `WaveShape` and
`gate.py`'s default in one commit, per the `--attn-tile` rule that a gate and a binary must not disagree).
`wide520f` then passed the whole gate with the looped form on every pinned path: **370 checks, 0 failures**,
with the new echo assertion reading `gdn_loop=512` on the main run and section (j) still a real A/B (its
spelling sides are pinned to span 1 because `ship`/`unroll4` have no looped body).

**The ruler** — `ids4096.txt`, `--steps 0 --time-waves --gemm lt --gemm-pins all --gdn-loop 512`, three reps,
one hold, blocking, zero subagents: **763.49 / 762.24 / 781.95 ms → median 763.49 ms**, against §16's
921.56 ms. **4,444 → 5,364 tok/s, +20.7 %** — the largest single move in this ledger, and the family row that
earned it is §24's (`gdn_rec` 449.07 → 266.03 ms). `gdn_rec` is now 34.9–36.5 % of the serialized wave against
46.4 % before; `attn` is now the largest single family at 34.0–34.9 %.

**Short widths, because a win that only exists at 4,096 lanes would be a different kind of result.** Paired
inside one hold, same binary, alternating per reps, claim = the `gdn_rec` family row, wall `TOTAL` reported
separately (medians of 3):

| lanes | `gdn_rec` span 1 → 512 | family ratio | counted launches | wall span 1 → 512 | wall move |
|---|---|---|---|---|---|
| 64 | 6.674 → 4.241 ms | **1.574×** | 1,152 → 18 | 21.30 → 18.75 ms | **+12.0 %** |
| 256 | 27.602 → 16.828 ms | **1.640×** | 4,608 → 18 | 49.54 → 39.14 ms | **+21.0 %** |
| 1024 | 109.400 → 67.007 ms | **1.633×** | 18,432 → 36 | 185.798 → 144.649 ms | **+22.2 %** |

So the lever is elastic at every width the engine serves, and the 64-lane row — where §1's old banner seemed to
show a 15 % *regression* — is flat-to-better in a paired window. That apparent regression was window noise and
this section says so with the pair beside it.

**A null worth recording, because it was nearly a false refutation.** The first version of this short-width A/B
generated ids space-separated; `--ids` takes **comma**-separated ids, so `parse_ids` read exactly one token and
both sides priced a one-token wave — 12.99 vs 12.93 ms at every "width", which reads as "the loop does nothing
below 1,024 lanes" and would have closed the wrong conclusion. Two guards came out of it: the instrument now
prints `launches=` next to the family `ms` (18 launches at 64 lanes is the loop; 1,152 is not), and any sweep
whose `tokens=` field disagrees with the width it names is a broken sweep, not a result. The script is
`tmp/gcc/run_smallwidth_ab.sh` with the trap in its header comment.

**§1's banner** — re-taken same day, same window order as its own commands, because it had gone stale by three
shipped items (`--attn-tile 1`, the conv coalesced read, and now the loop):

| prompt lanes | ms, best of 3 | all runs | spread | prompt tok/s | was (§1, 2026-10-16) |
|---|---|---|---|---|---|
| 64 | 104.7 | 104.9 / 104.7 / 115.9 | 10.72 % | 611.2 | 702.9 |
| 256 | **122.8** | 124.6 / 124.6 / 122.8 | 1.49 % | **2,084.6** | 1,828.5 |
| 1024 | **228.3** | 229.9 / 232.0 / 228.3 | 1.64 % | **4,486.2** | 3,443.1 |
| 3072 | **615.8** | 623.2 / 618.3 / 615.8 | 1.20 % | **4,988.5** | 3,385.8 |

Decode is **where it was**: 128 lanes 11.480 ms = 87.1 tok/s, 512 → 78.3, 1024 → 82.5, 2048 → 80.1 (§1's rows
were 90.6 / 80.6 / 80.6 / 80.8), i.e. every row inside its own historical spread, which is what §24's flat
decode pair (median 0.9999) predicted — a one-row body takes the per-token path whatever the flag says. The
64-lane prefill row still reads below the paired number (611 vs the 12-lane-scale paired ruler) and carries a
10.72 % spread: §6's dispatch bimodality, unchanged.

**What this does to the comparison instrument.** At matched `-ub 512` llama.cpp read 8,290 tok/s against our
4,430 (§21/§22, ratio 1.87×). At today's 5,364 the same llama.cpp number is **1.55×**, and the §22 census said
the two remaining mechanisms were launch count (GDN — now done) and the attention tile (still open: 48 launches
both sides, our 263 ms against their ≤52 ms). That is now the only ≥1.5× row left in the attribution table,
which is why it is item 2.6 and next.

## 26. Attention attributed: no matrix instruction anywhere in the engine, and the gap is **4.6×**, not 14× (2026-10-18)

Item 2.6 (attention, the last family with a ≥1.5× row from the llama.cpp comparison) was attributed before
any kernel was written, and the attribution corrected the item's own size twice. Census artifacts:
`prototype/attn_wmma/evidence/`, window script `prototype/attn_wmma/run_attn_attrib_window.sh`.

**Instruction class — counted out of GPU code carved from the shipping binary.** `tools/carve_codeobjects.py`
extracts the nine code objects from `decode_run`'s `.hip_fatbin` (the canary requires each object to name
`gfx1151` in its *own* ELF flags, because the "0 of the readable 55" census once scanned the wrong
population), and `tools/kernel_isa_census.py` refuses to run if a requested kernel is missing or empty.
Result across **all 40 kernels of the engine's own objects: 0 matrix instructions, 0 dot-class**. The shipping
`attn_slim_wave` is 1,983 instructions with 130 f32-FMA-class and no `v_mfma_*`/`v_wmma_*` at all. Every matmul
that reaches the matrix unit on this box reaches it through hipBLASLt, whose tuned kernels sit in containers
`llvm-objdump` cannot read — so D56's "the GEMMs run at 97–102 % of roof" is partly a statement about *who
wrote the kernel*, and 3.3 (hand WMMA GEMM) means writing from scratch to approach a rate the library already
sustains.

**The ceiling was the wrong one, so D56's 14× is retracted (see D57).** 22.3 TFLOP/s is a *matrix* rate; this
kernel can only occupy a scalar rung. Against the box's measured sustained `v_fma_f32` rate of **7.30 TFLOP/s**
(`prototype/issue_rate_probe`), attention's 1.59 TFLOP/s is **4.6× low**, not 14×. Two things fall out of the
ladder 1.59 → 7.30 (scalar) → 23.28 (`dot2_bf16`) → 52.63 (WMMA): llama.cpp's **≥7.9 TFLOP/s is above the best
scalar rate this machine has ever sustained**, so parity is a matrix-class question no matter what our own
disassembly says; and it is only 15 % of the WMMA rate, so the class is not exotic — the obstacle is
amortization.

**Occupancy — the first real LDS number for this box.** KFD `lds_size_in_kb 64`, `simd_per_cu 2`,
`simd_count 80`; `tools/kernel_occupancy_probe.cpp` (host-side, launches nothing, with a positive control that
caught LLVM deleting an unused LDS object) gives `attn_slim_wave` **5 blocks/CU = 20 of a possible 32
wavefronts = 62.5 %**, LDS and VGPR binding at the same point (12,288 B and 47→48 VGPR). Headroom is therefore
**1.6×, and only if LDS ≤ 8,192 B *and* VGPR ≤ 32 both hold**. Related ledger defect, since it was cited as
evidence: clang's `-Rpass-analysis=kernel-resource-usage` prints `Occupancy [waves/SIMD]: 16` for *every*
kernel in a TU regardless of resources — it is the machine maximum and cannot fail, so the
`attn_valu_slim` "Lever 0" claim that leaned on it is unsupported (the supported version is the sentence
before it).

### The device window (`run_attn_attrib_window.sh`, one hold, canary per leg)

**W1 — the shipped touch pair, measured in situ rather than borrowed from a harness.** `--attn-touch 0`
(frozen `attn_tile_wave`) against `--attn-touch 1` (shipping `attn_slim_wave`): **1,024/1,024 prefill logits
rows byte-identical**, `launches=` **identical** both sides (48 at 4,096, 12 at 1,024 — the pair changes the
body, not the submission), and `attn` family **403.06 → 264.71 ms at 4,096 lanes (1.5227×)** and
**27.07 → 17.27 ms at 1,024 (1.5672×)**, best-of-3 inside the hold. That is the §20 slim lever finally priced
against the ruler width instead of a probe.

**W2 — a denominator collision settled, and it settled the favourable way.** Two numbers 1.2 % apart had been
used interchangeably as *the* per-touch figure: 116 (static pass-1 slots) and 114.6 (measured whole-kernel VALU
per touch). The counter leg calibrates first: the `--attn-touch 0` side reproduces the published
`attn_tile_wave` figure at **2,790.5 M against 2,888 M recorded = −3.4 %**, inside the ±5 % band, so the
counter is believed. It then reads `attn_slim_wave` at **2,265.7 M VALU per pass = 89.9 per touch**, which is
**model A (whole-kernel) by a factor of 3.2** over model B (pass-1-only). Pass 1 is therefore the *high* branch,
≈61 % of the kernel's VALU — every remedy priced in D56's range keeps its upper end. Same run, first-ever
per-kernel split of the family: `attn_prep` 62.1 M and `attn_combine` 21.0 M per pass against the core's
2,265.7 M, i.e. **90 of 138 dispatches and 3.4 % of the family's VALU instructions**, so the un-censused
satellites are not where the time is (instructions are not wall, but the bound is one-sided).

**W4 — a ledger row corrected from inference to count.** §22's "attention launches / pass 48 vs 48, identical"
was arithmetic (`6 layers × 8`), not a measurement. Counting dispatch instances, deduped by `Dispatch_Id`:
**ours 414 traced = 138 per pass** (48 `attn_prep` + 48 `attn_slim_wave` + 42 `attn_combine`; pass count
derived from `attn_prep` against its 48 known sites), and **llama.cpp 96 `flash_attn_ext_f16` instances**,
which is 48 per pass over the two prompt passes that run made — so the row was right about *their* count and
wrong about ours: we submit **138 to their 48**, because our attention is three kernels and theirs is one fused
kernel. Submission cost of the difference is ~0.15 ms of a 763.49 ms wave, so the *conclusion* "not a
launch-count item" stands; the *row* did not, and §22 now carries the counted numbers.

**W3 and W5.** Realized occupancy was **not** established: the leg prints its calibration rules and refuses the
already-retracted `SQ_WAVE_CYCLES/SQ_BUSY_CYCLES` ratio, and it did not produce a number that satisfies them,
so the 20-of-32 figure stays a static cap rather than a measurement. W5 is a refutation, not a measurement:
4,609 lanes is **refused by name** (`kAttnMaxSplit = 9`) while 4,096 runs, so raising that ceiling cannot move
the ruler width at all — it is a long-context enabler and leaves this queue.

**Where the item stands after this.** `attn` is ~265 ms of 763.49 ms and it is a **mix-and-stall** problem
(5.6 issue slots per useful MAC × ~60 % of its own code's ceiling at 62.5 % occupancy), whose parity target
sits above the scalar roof: the matrix class subsumes ~27 of a touch's 45 pass-1 slots, a **2.4× cut of pass 1**,
not a 5× cut of `attn`. Priced honestly that is **+5 % to +12 % of the wave**, contingent on a compile-only
falsifier (an M = 16 feeder census at ≤19 slots/touch) that costs a day and no device time. Full adjudication
in D57, including the retraction of D56's rate framing.

**Addendum, same window — R6's wall split, answered from the trace already collected (no new device run).**
`prototype/attn_wmma/r6_wall_split.py` sums per-kernel durations out of the W4 engine trace (deduped by
`Dispatch_Id`, pass count derived from `attn_prep` against its 48 known sites). Against a family total of
279.38 ms/pass as the profiler sees it: **`attn_slim_wave` 270.77 ms (96.9 %), `attn_combine` 4.86 ms (1.7 %),
`attn_prep` 3.74 ms (1.3 %)**. So the 90 satellite dispatches §26 counted are **3.0 % of the family's wall** —
R6 closes on the one-sided bound the instruction census suggested, and the item does not exist. Traced durations
run above the stage clock's 264.71 ms for the same family (the inflation §22 already names, proportional to
launch count), so quote the **ratio**, not the 279 ms. One kernel is the whole problem, which is what makes
D57's compile-only feeder gate the right next instrument rather than a second sweep of the family.

## 27. The M = 16 geometry tax, measured (D58's deciding number), and why it prices item 2.6 out

Harness `prototype/attn_m2/attn_m16.hip.cpp` (worker `a15fd29e`), burst `burst_m16.sh`, window
`m16tax`, **run blocking with zero subagents**, `util` read 92 % before the gate and discounted. Admission
first (`gate_m16.py device` → **PASS 40/40**, 1.3 s wall, no timing). Every pair below is `shipslim`
(`attn_slim1_wave`, the kernel the engine ships, re-compiled in the same TU, `ndiff_aout_shipslim=0`)
against `m16` (the 16-row re-tile, shipped VALU spell, no WMMA), **launched alternately inside one rep**,
40 launches × 5 reps, best-of — so it is one ruler, not two windows. Log: `tmp/gcc/m16tax_burst.txt`.

| fixture (rows @ lanes) | `shipslim` best ms | `m16` best ms | **tax** | traffic ratio m16/ship | blocks ship→m16 |
|---|---|---|---|---|---|
| 64 @ 4096 (widest grid this harness builds) | 1.4659 | 3.1055 | **+112 %** | 0.30 | 1,024 → 256 |
| 64 @ 2048 | 0.7664 | 2.5117 | +228 % | — | — |
| 64 @ 1024 | 0.3494 | 2.1102 | +504 % | — | — |
| 16 @ 4096 | 0.4433 | 3.0593 | +590 % | 0.30 | 256 → 64 |
| 16 @ 1024 | 0.1437 | 2.0479 | +1,326 % | 0.28 | 96 → 24 |
| 4 @ 512 (`full=0`, streaming forced) | 0.0988 | 0.3819 | +287 % | 0.27 | 16 → 16 |

**The re-tile demands 0.28–0.30 of the bytes and is 2.1× to 14.3× slower, at bit-identical output, in
every fixture this harness can build.** Two things make that a verdict rather than a bad day:

- **The traffic win is worth nothing here, and that is already measured.** D57's instruction-cut falsifier
  made `attn` **1.521× faster with the byte census unchanged**, which is the proof that this kernel is not
  traffic-bound at these widths. A re-tile whose only lever is demanded bytes is pulling on a rope that is
  not attached to anything.
- **The cost is registers, and the probe kernels understated it.** The full-body VALU re-tile needs
  **134 VGPR / 12,800 B** against the shipped **47 / 12,288** — LDS still allows 5 blocks/CU but the
  register file caps it at **3 (12 wavefronts against 20, −40 %)**. The 16-row live set is 4 `acc[8]` per
  warp; `GRP=8` is refused by the accumulator `static_assert`, and `GRP=2` restores parity only by buying
  the same 8 touches/lane-row the shipped M = 2 already gets, i.e. **no re-use at all**. So re-use and
  occupancy are *coupled* in this spell: you may have one.

**Why this prices the WMMA route and is not a proxy for it.** D58's `m16_bf16gq` at **30 VGPR** is not a
counterevidence: it is a score-tile probe that carries no softmax/rescale state and no `P·V`. A block that
does the whole attention body at M = 16 must hold the score fragments **plus** the `P·V` fragments over
`DH = 256` (**128 VGPR/lane on its own**, D30 §2), so the real WMMA block's register bill is *above* this
re-tile's 134, not below it. **The measured tax is therefore an under-estimate of what a WMMA M = 16 block
would pay**, and the cheapest form of the geometry is already +112 % at its best fixture against a route
whose whole prize was −15 % to −17 % of `attn`.

**Two things this does NOT establish, stated so nobody has to rediscover them.**
1. The engine's real wave is **512 rows** and this harness tops out at 64, so the tax *at shipping
   geometry* is not measured — only bounded by the trend (16 rows → 6.92×, 64 rows @ 4096 → 2.12×, which
   is grid starvation: `blocks_m16` falls 4× and the part has 40 CUs to fill). Re-open condition, cheaply
   stated: a 512-row fixture showing the tax below ~1.15×, which requires the re-tile to be traffic-bound,
   which D57 already disproved.
2. `burst_m16.sh` includes **`short-64`** (`64 rows @ lanes 64–127`, `split=1`, `full=0`), which came back
   **`ndiff_aout_m16 = 131072` — every element differs** — and the harness correctly printed
   **`NO-TIMING: admission is red`** and refused a millisecond figure. The worker's report claimed bit-
   identity at spans 0/1/3/16 (true, and the device gate is 40/40) and did not mention this case. Signature
   matters: `ndiff == nrows × 2048` exactly is a whole-tensor difference, which reads as a changed work
   split at `split=1` rather than a wrong value — but it is unresolved, it is on the M = 16 form, and
   item 2.6's closure does not depend on it.

**Decision.** Item **2.6 is closed**, not deferred: the feeder is cheap (D58), the launch count was never
the wall (§22), the kernel is not traffic-bound (D57), and the geometry that a matrix tile requires costs
**+112 % at best** in registers and occupancy (this section). Every ingredient of the route has now been
measured and each one fails; there is no untested lever left to fund. 3.3 and 2.4 stay closed as before.
The wave's attention share (34.0–34.9 %, `attn` 264.71 ms of 763.49 ms at 4096 lanes) stays where it is,
and the engine headline is unchanged by this section: **prefill 5,364 tok/s, decode 71.3**.

## 28. §27's conclusion is overturned by the missing side of its own experiment: the tax is neither registers nor traffic (2026-10-18)

§27 booked the M = 16 re-tile at **+112 % at best** and attributed the cost to register pressure
(**134 VGPR → 3 blocks/CU**). The harness could already answer that — `attn_m16_grp2` is the same source
built `-DATTN_M16_GRP=2` (**96 VGPR, 10,496 B → 5 blocks/CU, shipped parity**) — and §27 timed only the
`GRP=4` binary. So the attribution was made against the one shape the census had already called
unoccupiable, and that number was then used to close item 2.6.

Retaken in a fresh hold (`m16grp2`, lock held, `burst_m16.sh` with `BIN` pointed at the parity build,
admission inside the same run — `ndiff_aout_m16 = 0` on every case timed, `short-64` still red and still
refused). Log: `prototype/attn_m2/evidence_burst_grp2.txt`; tabulated by `tmp/gcc/tab_m16_grp.py`.

| fixture | shipped ms | **GRP4** m16 (3 blk/CU, traffic 0.28–0.30) | **GRP2** m16 (**5 blk/CU parity**, traffic 0.50–0.55) | GRP2/GRP4 |
|---|---|---|---|---|
| 64 @ 4096 | 1.4659 | 3.1055 (2.12×) | 3.1135 (2.12×) | 1.003 |
| 64 @ 2048 | 0.7664 | 2.5117 (3.28×) | 2.6101 (3.41×) | 1.039 |
| 64 @ 1024 | 0.3494 | 2.1102 (6.04×) | 2.0459 (5.86×) | 0.970 |
| 16 @ 4096 | 0.4418 | 3.0593 (6.92×) | 3.0546 (6.91×) | 0.998 |
| 16 @ 1024 | 0.1437 | 2.0479 (14.25×) | 2.0028 (13.94×) | 0.978 |
| 4 @ 512 (streaming) | 0.0988 | 0.3819 (3.87×) | 0.3628 (3.67×) | 0.950 |

**Going from 3 to 5 blocks/CU, while giving up half the traffic advantage, changes the time by at most
5 %.** So the mechanism §27 named is not the mechanism, and neither is traffic:

- **Registers: refuted.** A 66 % occupancy increase buys nothing, so this kernel was never occupancy-limited.
- **Traffic: refuted.** GRP2 demands ~1.8× the bytes of GRP4 and costs the same — D57's "demanded bytes are
  not DRAM bytes" appearing a third time from a different direction.
- **What the numbers actually say: serial-round latency at low interleave.** The m16 wall tracks **lanes and
  only lanes** (4096 → ~3.05 ms, 2048 → ~2.6, 1024 → ~2.05, 512 → ~0.36) and is **flat in rows**: 64 rows
  @ 4096 and 16 rows @ 4096 cost the same 3.05–3.11 ms, while the shipped form's cost spans 15× across the
  same set. Halving traffic, halving live pairs, and going from 128 to 512 blocks all fail to move it. That
  is the `gdn_collapse` signature read from the other side: price a serial kernel in **cycles-per-dependency
  at its realised interleave**, because a tile makes each block *longer* at the same time as it makes each
  block *fewer* — the shipped form is cheap precisely because it is short and volume hides its latency
  behind 20 wavefronts/CU.

**What that does and does not license.** It does **not** rehabilitate the re-tile on these fixtures, where it
is 2.1× to 14× slower and the mechanism is now understood rather than guessed. It **does** remove the two
claims that made the closure read as a law — "the register file refuses it" and "+112 % in its cheapest
form" — and one was flatly false: **llama.cpp runs a 256×256-tiled attention kernel on this same silicon**
at ≤52.5 ms for the family where we spend 264.71 ms, so the machine demonstrably holds tiles far bigger
than 16×16. The difference between the two designs is not register capacity but **what is inside a round**:
theirs pipelines matrix ops with independent accumulators; ours walks a dependent dot → 5-step butterfly
fold → rescale chain per live pair, 16 of them per warp.

**Item 2.6 re-opens as unmeasured**, with a specific instrument, because the two censuses have never been
combined and neither geometry was ever timed at the engine's own size:

1. the engine's wave is **512 rows**, which would issue 8× the blocks of the largest fixture here (where 16
   rows put just 128 blocks on a 40-CU part, leaving the latency nowhere to hide); a **512-row × 4096-lane
   fixture** in this harness is the first measurement that could be read as a tax;
2. nobody has timed a tile whose **score comes from a matrix op**, which is the term that removes the
   dependent fold — D58 censused it at **1.98 slots/touch** and §27 measured the tile with the VALU fold
   still in it. Two halves of one question, measured in two runs that never met.

## §29. The WMMA attention round, measured on the engine path: 1.3665× of the `attn` family at 4096 lanes — under the 1.5× bar, and everything the bar was not designed to see

Device window, main agent, blocking, zero subagents, one `gpu_lock` hold (`util` 6 %–92 % seen across it, and the
paired sides were taken inside one hold as §13 requires). `prototype/attn_wmma/run_fattn_burst_window.sh`,
`--attn-fattn 0` vs `1` paired best-of-3, canaries green in the same breath as the `ms`:
**CANARY A** the knob exists and refuses garbage; **CANARY B** banner `form=fattn16 vgpr=256 lds=16384
vtiles=16 wf=4 fattn_rows=512/512` **agreeing with the engine listing's own note block** (the stale-`new`-side
guard); **CANARY C** a covered row's logits differ byte-wise between the sides (so neither side is a stale copy);
**CANARY D** every dump byte-identical at 64 lanes (route (A)'s inertness asserted, not trusted).

| width | `attn` family, shipped | `attn` family, WMMA round | ratio |
|---|---|---|---|
| T=1024 | 17.4761 ms (12 launches) | 14.6424 ms | **1.1935×** |
| T=4096 | 262.3243 ms (48 launches) | 191.9748 ms | **1.3665×** |

Covered share printed with both rows: **512/512 = 100 % of rows in the widest wave** — so unlike route (A) at
short context, this is not a partial-coverage ratio diluted by uncovered work. Against D62's bar the claim
**FAILS** (1.3665 < 1.5), and it is recorded as a failure, not as a near-miss to be re-quoted.

What the bar was not designed to see, and what therefore has to be said out loud: `attn` is 34.0–34.9 % of the
wave, so −26.8 % of the family is roughly **−9 % of prefill** (5,364 → ~5,900 tok/s if the flag were defaulted
on), which is larger than any single item shipped since the GDN launch batching. The bar was set at 1.5× because
the funding argument (D62) priced the round body at 3.29–7.16 slots/touch against our measured 89.9, and a
~1.37× realised says the **round shape is right and something around it is costing more than the census saw** —
the two named candidates are both things the census could not price: **4 wavefronts/CU** (five times below the
12–20 band §28 measured as irrelevant, so this is untested territory, printed as `wf_per_cu=4` for exactly that
reason) and **49 spilled VGPRs / 72 B scratch** acknowledged under D63.

Two things this window also had to fix before it could measure anything, both on the shipping path and both found
by gates rather than by inspection: `gate.py` carried an **unterminated string literal** at line 6859 that the C++
build cannot see, so every `gate.py` invocation was a `SyntaxError`; and the launcher's plan guard read
`c.fused` — the **one-row** core args, whose `fused` field the wave launcher never writes — so `--attn-fattn 1`
threw "fused plan" on every wave. The guard now sits directly above the launch and reads `core`, the struct the
launch dereferences; and a wave with **no streaming rows allocates no partials at all**, so the form is skipped
and says so (`ATTN_FATTN_SKIP rows=0/N`) instead of dereferencing a null `part`. With those, `dec8` at
`--attn-fattn 1` passes the full engine gate, and admission **section 1 is green on dec8/dec64f/dec300f on both
flag values**. Section 2 (the covered-row band) is still blocked: `prototype/attn_decode/floor_probe.py` parses
every dump directory name as an integer and dies on `wave000-031`, so the cos/meanrel band for streaming rows is
still a borrowed number, and **adoption cannot be proposed until that is fixed**. The WMMA latency legs never
ran at all — `wmma_latency` core-dumps on its first leg, and its own row-count check reported 0-of-15 as the red
result it is, so the f16-accumulator question is still unanswered.

### §29 correction, same day: the 1.3665× was measured on a form whose covered rows are ALL NaN, and CANARY C was green *because* of that

[CORRECTION IN PLACE — see the section above before quoting anything from it.]

The acceptance review (`tmp/gcc/fattn_band_analysis.md`, worker `80316334`) found that every row the WMMA form
owns (rows 512..1023 of `wide1024f`) is entirely non-finite: direct dump census `nan = 248320/248320` per row,
the engine's own echo goes `prefill 511 ... top 29.66` then `prefill 512 ... sampled 0 top nan`, and the fault
trace puts the first NaN in the output of the first full-attention layer (`layer_types = [L,L,L,F]x6`, kv row 0
finite, kv rows 1-5 and GDN state rows 3.. NaN). The 560 gate failures decompose exactly as 20 token-id +
24 logits + 512 per-row + 4 state-invariant, and the control is unambiguous: **with the flag off, zero NaN, rows
0..511 bit-identical to the flag-on side, max 1-cos 1.969e-04, max meanrel 3.390e-02** — so every failing class
belongs to this form and none is pre-existing. Decision room is not the excuse: min top1-top2 gap 3.875 (31 BF16
ULP) at row 519, median 11.3125, **0 of 512 rows under the thin-room line**. NaN is not band-able, so the
question the user funded (is the precision cost small?) cannot be asked yet, and the flip and the merge are off
the table until the NaN is gone.

Two things about my own instrumentation, because they are the reason this survived as long as it did.
**CANARY C was green *because* of the NaN**: it asserted the two sides differ bit-wise so a stale side could not
pass, and a NaN row differs from a finite row beautifully. A difference witness proves non-staleness and nothing
about validity — a form that computes garbage passes it. And three gate checks could not see it either:
`gate.py:3409,3414` judge rows against caps with `>`, and `nan > cap` is False, so section (i) printed
**PASS, "1024 rows judged ... inside the floors", over 512 NaN rows**; the b2 side at `gate.py:8196` does not
forward `--attn-fattn`, so the "identical conditioning" comparison silently ran the frozen form; and section (j)'s
"1025.1 MB bit-identical" was vacuous over non-finite data. Every one of those is the same trap in a new coat: a
check whose predicate cannot be false for the failure it is aimed at.

## §30. The WMMA streaming form ships: 1.3955× of `attn`, 1.1407× of the wave, no new band needed (2026-10-18)

The NaN is gone and the form is the engine's default (`--attn-fattn 1`, `decode_step.h` and `gate.py` moved in
one commit). What was measured, all of it main-agent, blocking, zero subagents, each pair inside a single
`gpu_lock` hold:

| instrument | frozen (`attn-fattn 0`) | WMMA default | ratio |
|---|---|---|---|
| `attn` family, 4096 lanes | 263.3832 ms (48 launches) | 188.7432 ms | **1.3955×** |
| `attn` family, 1024 lanes | 17.4584 ms (12 launches) | 14.4516 ms | 1.2081× |
| whole wave, 4096 lanes, best-of-5 medians | 856.435 ms | **750.824 ms** | **1.1407× (+14.1 %)** |

Claim (iii)'s 1.5× bar is still **not met** and is recorded as failed — but the user adopted the win on the
grounds that a diluted whole-family ratio of ~1.37–1.40 is worth ~14 % of prefill and the numerics came in
**inside the bands already in force**, so no band was widened: `wide1024f` PASSes the full gate on **both** flag
values, with 512 streaming rows judged 0 outside the cos cap, 0 outside the meanrel cap, 0 non-finite, 0 argmax
moved, worst 1−cos 1.99e-04 against the 2.0e-3 cap and meanrel 3.39 % against 5 %, and rows 0..511 bit-identical
between forms. Boundary row 512 (first streaming row): cos deficit 2.8301e-05.

**Honest caveat on the absolute headline.** My frozen side reads **856 ms** where §25's ledger median is
**763.49 ms** — this window ran ~12 % hotter than the one that booked the old headline, so the two sides of
*this* table are only quotable as a **ratio**. By ratio the clean-window expectation is ~669 ms (**≈6,100
tok/s**); by this window's own absolute median it is 750.8 ms (**5,455 tok/s**). The canonical headline re-take
in a clean window is owed before the banner number is rewritten, and §25's 5,364 stands until then.

Also on the record from this window: `prototype/attn_wmma/fattn_ship_census.py`'s derived-geometry canary now
**fails** on `ds_load_b128 sites 68 > derived 39 + 2`, because the NaN fix publishes all 256 dims per row where
the model expected the broken epilogue's traffic. That is the census being right to be suspicious and its model
being stale, in that order — the expectation must be **re-derived from the new source's staging loops**, never
raised to meet the listing. Until that is done the census is a red gate and the form's per-touch instruction
figure is stale. The remaining gap to 1.5× still has two unpriced candidates, both scheduling rather than
arithmetic: **4 wavefronts/CU** (five times below the 12–20 band §28 measured) and **49 spilled VGPRs**.

## §31. Bet 1: un-staging Q doubles residency and takes the `attn` family to 1.647× — the bar is cleared, and residency was real (2026-10-18)

`tmp/gcc/fattn_unstage_q.patch` applied: the block stops staging Q, `group_segment_fixed_size` goes 16,384 →
**8,192 B**, and the driver's answer goes **4 → 8 wavefronts/CU** (read from each side's own `wf_per_cu` banner,
not asserted). Costs measured at compile time: **+2 spilled VGPRs, +4 B scratch**, −32 B of LDS reads, +16 B of
global reads for a tile that is now L1/L2-resident. Timing: three sides interleaved in one lock hold, `sleep 10`
before every run, 4096 lanes:

| side | `attn` family median | wave total median |
|---|---|---|
| frozen (`--attn-fattn 0`) | 264.58 ms | 872.14 ms |
| staged Q (the merged form, 4 wf/CU) | 192.85 ms | 760.05 ms |
| **Bet 1, 8 wf/CU** | **160.65 ms** | **735.51 ms** |

So residency was **not** the free variable §28 made it look like: doubling it from 4 to 8 wavefronts cut the
`attn` family a further **1.20×**, which puts the whole change at **1.647× over frozen** — past D62's 1.5× bar,
which the staged form missed at 1.3955×. Wave total moves 872.14 → 735.51 ms, **+18.6 % prefill** at today's
sustained clock (~2.45 GHz), i.e. **5,569 tok/s** measured in this window against the same window's frozen
4,696. One bet1 rep printed a 1615 ms total (an outside interference, visible and rejected by the median of 3);
its `attn` figure was normal, so the stage-level numbers stand while the total is taken from the median.

Correctness: `wide1024f --scope full` **PASSES** on the Bet-1 build with the default ON, and the three boundary
rows report the same cos deficits as the staged form (511: 3.7518e-05, 512: 2.8301e-05, 1023: 2.7208e-05).
**Known-red gate on this commit:** `fattn_ship_census.py` now refuses the listing
(`lds_read_B 1056 vs derived 1088`, `global_read_B 64 vs derived 32`) — correctly, because Q moved from an LDS
stage to a global read and the model still derives the Q operand's source from the old spelling. The fix is to
teach `derive()` to read that source from the source text; the constant may not be raised to match the listing.
Bet 2 (3 warps/block sharing one K/V stage, 6 wf/CU at zero spill) is now the test of whether 8 is still too
few: if 6 wf/CU with no spill at all beats 8 wf/CU with 50 spilled, the two suspects are separable and spill is
the cheaper of the two.

## §32. Bet 2 compiled exactly as designed and is numerically dead — but it proved the spill is avoidable (2026-10-18)

`tmp/gcc/fattn_bet2.patch` (96 threads = 3 warps/block, each owning a 16-row block of one head, one shared K/V
stage): **the compile-time target was met** — `vgpr_count 178, vgpr_spill_count 0, private_segment 0,
group_segment 32768` ⇒ **6 wf/CU with zero spill and zero scratch**, driver-confirmed. That alone retires a
belief: **the 50 spilled VGPRs Bet 1 carries are not intrinsic to the round** — widening the block clears them
(50→0 spilled, 72→0 B scratch).

But the form **cannot be admitted**: every covered row emits non-finite logits (248320/248320). The bisect is
informative rather than mysterious — the same patch at `FATTN_WARPS=1` is finite and reproduces Bet 1's
`1−cos = 2.104e-05` exactly, while `WARPS=2` and `3` are all-NaN. So the per-warp Q re-stage, per-warp
transpose staging and per-warp publish are individually correct, and the defect lives in what only appears when
warps share the stage: coordination across warps, not geometry. That is the third NaN on this kernel in one day,
and each time the signature was "whole rows non-finite", which is a missing/aliased write, not a wrong value.

**Interim read of the programme:** residency is real (4→8 wf/CU = +20 % of the family, §31); spill is avoidable
(here); whether *spill-free at 6* beats *spilled at 8* is still unmeasured, and it is the discriminator between
"interleave is the wall" and "spill traffic is the wall". Fixing the multi-warp coordination is the next
experiments' gate, and no timing of a non-finite form is permitted (the burst refuses `SPILL`, and admission
section (i) now refuses non-finite — both by design, and both earned).

## §33. The discriminator: spill-free at 6 wf/CU BEATS spilled at 8 — the round is now 1.985× over frozen (2026-10-18)

Bet 2's multi-warp NaN was a one-loop defect: the patch filled each warp's own Q stage with the shared-buffer
stride (`for (qr = w; qr < 16; qr += nw)`), so warp *w* staged slots *w, w+3, …* of a stage its score leg reads
in full — ~2/3 of every warp's Q read unwritten, `live` true so nothing masked it, and the NaN path ran out
through `attn_combine`. Correct loop: stage every slot of one's own stage once. Note why the earlier bisect
missed it: at `WARPS=1` the strided loop **is** the correct loop, so the probe that "proved the per-warp Q
re-stage correct" was structurally blind to the only thing that differed at `WARPS>1`.

Discriminator, three sides interleaved in one lock hold, `sleep 10` before every run, medians of 3, each side's
`wf_per_cu` read from its own banner:

| side | residency | spill | `attn` family | wave total |
|---|---|---|---|---|
| frozen | 20 wf/CU (VALU wave) | 0 | 263.12 ms | 869.13 ms |
| Bet 1 (staged→un-staged Q) | **8 wf/CU** | **50 regs / 72 B** | 161.31 ms | 751.69 ms |
| **Bet 2 (3 warps/block)** | **6 wf/CU** | **0 / 0** | **132.57 ms** | **715.69 ms** |

So **spill traffic was real, and it was worth more than the interleave it bought**: fewer resident wavefronts
with no scratch at all beat more resident wavefronts with 50 spilled registers, by **1.217×** on the family.
Cumulative: **1.985× on the `attn` family** and **+21.4 % on the whole wave** (869.13 → 715.69 ms) in this
window — against D62's 1.5× bar, which the staged form missed and both widened forms clear. One bet2 rep printed
a 1408 ms total (the same intermittent interference seen in §31); its `attn` figure was normal and the median
rejects it. Correctness: `wide1024f --scope full` **PASSES** on the Bet-2 build with the default on — 512
streaming rows, 0 outside either cap, 0 non-finite, boundary row 512 cos deficit 2.8301e-05, identical to Bet 1.

**Known-red gate on this commit:** `fattn_ship_census.py` now **crashes** on the geometry evaluator
(`cannot evaluate 'FATTN_WARPS'`) because its expression model does not know the new warp count. That is a model
gap on the census's side, not a finding about the kernel — but it is a red gate, recorded as such, and the fix is
to teach `geometry()` about `FATTN_WARPS` and re-derive per-warp traffic from the source, not to bypass it.
Next lever, now that spill is separated: **push residency further inside a zero-spill geometry** — the live
question is whether anything at 8+ wf/CU exists that does not pay for it in scratch.

## §34. The tip did not build: §33's commit carried the kernel half of Bet 2 and not the launcher half (2026-10-19)

`5b82c56` shipped `engine/attn_fattn_kernels.h` at `FATTN_WARPS 3` while the header that **defines** the
launcher's row-block rule — `prototype/attn_decode/attn_decode_kernels.h`, `FATTN_ROWS_PER_BLOCK` — was still
`16` and sat **uncommitted in the working tree**. So HEAD failed to compile:

```
engine/attn_fattn_kernels.h:211:15: error: static assertion failed due to requirement '16 == 3 * 16'
```

That assert is the one written against the other failure of §20.1 of
`prototype/attn_decode/results.md` ("a row neither kernel owns is invisible in both"), and it is what caught
this — one level up, at the commit boundary rather than on device. The fact it
guards is a **two-header** fact (`fattn_rtiles_of` counts blocks in one header, the kernel carries rows in the
other, and the launcher's grid is built from the first), so no single-file review of the diff could have seen
the missing half; only a build of the committed tree can. Fixed by amending the same topic (`2ac956d`), which
is what amending is for; `backup-bet2-5b82c56` keeps the pre-amend ref.

**Every number in §33 stands.** The three-sided discriminator and the `wide1024f --scope full` admission were
both run from a tree carrying both halves — it is the *record* that was half-committed, not the measurement.
The rule this installs is cheap: **build the committed tree before the commit is called landed.** Not the
working tree — the working tree is precisely where the missing half hides. `engine/build.sh` is **17.4 s**
(measured, this session, both binaries), and the one-TU form is cheaper still:

```
worktree=$(mktemp -d); git worktree add "$worktree" HEAD --detach -q
git worktree remove "$worktree" --force   # or: hipcc -fsyntax-only -I. -Iengine engine/kernels_attn.cpp
```

This is the same failure family as the two `--attn-tile`-default rows in the traps index (a geometry fact that
lives in two files must move in one commit) plus a new half: **moving it in one commit is not enough if only
one of the two files is staged.** The gate for that is a build, and the build is 17 s.

## §35. The next lever is not a geometry search, it is one compile: the residency ladder, priced (2026-10-19)

§34's last line asked whether an 8+ wf/CU zero-spill shape exists. `prototype/attn_wmma/fattn_regfile_ladder.md`
answers it on paper, compile-only and driver-only, and the answer has three parts.

1. **The Q-staged family is LDS-bound and its ceiling is 7, not 8+.** `LDS(block) = 8192·(NW+1)` gives
   `wf/CU = NW·floor(8/(NW+1))` = 4, 4, 6, 4, 5, 6, **7** for NW = 1…7 — which reproduces every skeleton row §4
   of `tmp/gcc/fattn_residency.md` measured, so its interpolation is safe. The shipped NW=3 at 6 wf/CU is
   LDS-bound at 178 VGPR, so **no register cut moves residency in the shipping geometry**; the only rung that
   family still has is NW=7 at the full 64 KiB.
2. **The handover mis-attributes the 178 VGPR.** Un-staging Q costs +2 spilled registers (Bet 1's own table:
   48 → 50 spills), so the −128 VGPR and all of the spill relief came from the **widening**, not from a stage.
   Which means the register demand of the *3-warp body with Q un-staged* — the one shape that reaches 8 blocks/CU
   of LDS headroom — **has never been compiled**: V6 is the skeleton's version of it and §4 forbids carrying a
   spill claim from the skeleton, whose control undercuts the real body's demand by ~100 registers.
3. **The register file per CU is unmeasured on this part**, and every rung above 8 wf/CU is model until it is.
   The usual AMD per-SIMD figure is contradicted by the one driver answer that could test it (§4's V6: 8
   blocks/CU at 3 warps × 256 VGPR = 24 wavefronts, needing 3× that file), and `rocminfo` prints no register
   size here — after two retirements of HIP's own numbers on this SKU, a third cannot be assumed.

So the queue's next bet is a **port and a read**: re-establish Bet 1's un-stage on the shipped 3-warp body and
read its note block. (Written first as "one transient patch", then disproved by `patch --dry-run` on a clean
snapshot: 4 of 5 hunks of `engine/attn_fattn_kernels.h` and 1 of 1 of `engine/kernels_attn.cpp` fail, because
Bet 2 rewrote the very hunks the un-stage touched.) `spill = 0` at ≲190 VGPR opens an 8+ zero-spill rung worth a
device A/B; spill > 0 closes the residency axis at 6 and moves the money to the instruction/traffic gap of §26.
No number in this section is a timing, and none of it predicts one — §31 and §33 together are the proof that
residency and speed are not the same axis on this kernel.

## §36. The banner re-taken on the shipped WMMA form, with the clock on the record (2026-10-19)

§33 closed with two honest gaps: the headline was quoted against an *inferred* "~2.45 GHz", and decode was
"unchanged by construction, unmeasured". Both are closed. Main agent, blocking, foreground, **zero subagents
active**, two `tools/gpu_lock.sh` holds (`bench36-ruler`, `bench36-sweeps`), `util` read at acquire and release
and discounted per §6 (it read 89→65 % and 90→35 % across the two holds, which is exactly the range §6 says to
ignore). Scripts: `tools/bench36_ruler.sh`, `tools/bench36_sweeps.sh`. Logs:
`progress/bench/evidence/retake-2026-10-19/`.

**The clock, measured rather than assumed.** `rocm-smi --showclocks` read **819 / 827 / 838 MHz idle before each
ruler rep and 2432 / 2437 / 2423 MHz immediately after it**, so the sustained clock under load is **≈2.43 GHz**
— the §31/§33 "~2.45 GHz" was 0.8 % high but was not a fiction. `mclk` read 1000 MHz at the window edge and is
recorded as read, not interpreted.

**The ruler** (§25's form: `ids4096.txt`, `--steps 0 --time-waves --gemm lt --gemm-pins all --gdn-loop 512`,
shipped defaults so `--attn-fattn 1`, three reps, `sleep 10` before each, one hold):

| rep | `RESULT ms_total` | wave `wall_ms` | launches | plan |
|---|---|---|---|---|
| 1 | 705.863 | 626.6192 | 2618 = 2618 | `plan_match=1` |
| 2 | 708.141 | 627.7535 | 2618 = 2618 | `plan_match=1` |
| 3 | 714.855 | 629.5035 | 2618 = 2618 | `plan_match=1` |
| **median** | **708.14 ms = 5,784 tok/s** | **627.75 ms** | — | spread 1.27 % / 0.46 % |

Against §25's 763.49 ms that is **+7.8 %**; against §33's 715.69 ms on this same binary it is **+1.07 %**, which
is window noise and not a win — the honest reading is that §33's number reproduces. The 4,096-lane headline
this ledger carries is therefore **5,784 tok/s at ≈2.43 GHz**.

**Prefill and decode: §1's two tables are updated in place**, with `was` carrying the §25-window take. The short
version: 3072 lanes **4,988.5 → 5,898.8 prompt tok/s (+18.3 %, 1.55 % spread)**, 1024 **+3.2 %**, and every
decode row inside its own spread (128: 11.475 vs 11.480; 512: 12.928 vs 12.764; 1024: 11.922 vs 12.128; 2048:
12.558 vs 12.487 ms/token). **Decode did not move, and that is now a measurement rather than an argument** —
the WMMA round is wave-only by construction, and it left the decode table where it found it. The 3072 row is
the one that carries the round, as §1 has always insisted long-context rows should.

**Correctness of the committed binary, re-confirmed here rather than inherited:** `.venv/bin/python
engine/gate.py --fixture wide1024f` at defaults (`--scope full`, `--attn-tile 1`, `--attn-fattn 1`) prints
**ENGINE GATE PASS** in 66 s on the tree built from `2ac956d` — the commit that §34 records as having been
incomplete. It is complete now, it builds in 17.4 s, and the form it ships is what the gate judged.

One editorial note, because it is the reason this section is appended rather than folded in: `progress/` cites
this file by line at **92, 618, 659 and 1184**, so §1's tables were rewritten with **same-line substitutions
only** and the file's line count is byte-identical to before (1,621 both ways, and the four cited lines verified
character-for-character). A banner that rots four archive citations to update itself would be a bad trade.

## §37. Nothing else moved: HEAD vs `origin/master`, family by family, in one window (2026-10-19)

The round that took `attn` from 263 to 132 ms is 28 commits deep, and a before/after on the **total** would
hide the regression such a change actually risks: something *outside* the changed kernel — the one-pass
`attn_slim_wave` rows, the split/combine launches, casts, or the launcher-side row-block rule that §34 shows
was already half-committed once. So this compares **families**, paired inside one `tools/gpu_lock.sh` hold
(`regress-origin`, main agent, blocking, zero subagents), baseline built in a scratch worktree, medians of 3,
alternating sides per rep. Baseline = `origin/master` = `9303c13`, strictly 28 behind. Harness:
`tools/regress_vs_origin.sh`; logs: `progress/bench/evidence/regress-2026-10-19/`.

| family | 4096 base | 4096 HEAD | ratio | 1024 base | 1024 HEAD | ratio |
|---|---|---|---|---|---|---|
| `attn` | 265.198 | **132.849** | **1.996×** | 17.440 | **12.181** | **1.432×** |
| `gdn_rec` | 275.998 | 266.236 | 1.037 | 69.018 | 69.140 | 0.998 |
| `proj` | 95.724 | 94.516 | 1.013 | 24.281 | 23.191 | 1.047 |
| `mlp_gateup` | 56.629 | 57.693 | 0.982 | 14.000 | 14.173 | 0.988 |
| `mlp_down` | 26.736 | 27.382 | 0.976 | 6.876 | 6.813 | 1.009 |
| `gdn_conv` | 16.156 | 16.681 | 0.969 | 3.961 | 3.985 | 0.994 |
| `glue` | 15.946 | 15.198 | 1.049 | 3.833 | 3.753 | 1.021 |
| `gdn_gated_norm` | 9.252 | 9.056 | 1.022 | 2.321 | 2.370 | 0.979 |
| `gdn_gates`, `cast`, `copy`, `lm_head`, `host_tail` | — | — | all 0.95–1.05 | — | — | all 0.94–1.02 |
| **wave `wall_ms`** | **770.248** | **628.451** | **1.226×** | 147.204 | 140.164 | 1.050× |
| **`RESULT ms_total`** | **869.299** | **708.544** | **1.227×** | 240.290 | 237.185 | 1.013× |

**Verdict: no regression, and the instrument's own noise floor is now measured.** Every family the round did
not touch lands inside **±3.1 %** of the baseline — `mlp_gateup` −1.8 %, `mlp_down` −2.4 %, `gdn_conv` −3.1 %,
`copy` −5.0 % on an absolute 0.09 ms — and none of those four files changed in the 28 commits. That is what
sets the floor: **on this box, in a paired window, a sub-3 % move on a sub-30 ms row is not claimable as
either a regression or a win**, which is a number this ledger had implied but never stated. What did move is
the thing that was supposed to: `attn` **1.996×**, matching §33's 1.985× taken in a different window and a
different harness, and the wave **1.226×** at 4,096 lanes against **1.050×** at 1,024 — the smaller number is
not a weaker kernel, it is a smaller denominator, because at 1,024 lanes `attn` is 8.7 % of the wave where at
4,096 it is 21.2 %.

**Cross-window corroboration, which is the part worth trusting.** The HEAD side reads `RESULT` **708.544 ms**
against §36's independently-taken **708.14 ms** (0.06 % apart), and the *baseline* side reads **869.299**
against §33's frozen **869.13** (0.02 % apart). Two windows that share neither clock nor day agree to a
twentieth of a percent on both sides, so the 1.227× is not a window artifact. At 4,096 lanes that is
**5,781 tok/s** on this instrument against §36's 5,784.

**Numerics, at the same cost as a footnote.** The baseline binary run through **its own** `gate.py` (HEAD's
gate passes `--attn-fattn`, which the baseline cannot parse) prints **ENGINE GATE PASS** on both `wide520f`
and `wide1024f`. That is not a claim about HEAD — §36 already re-admitted HEAD on `wide1024f` — it is the
check that the fixtures are not HEAD-specific, i.e. that the baseline is a baseline and not a straw man.

**Three canaries, and one of them earned its keep immediately.** A: the baseline must reject `--attn-fattn`
and the shipping binary must accept it. B: the shipping log must echo `ATTN_FATTN`. C: the `attn` family must
differ by more than 5 %, or the window is discarded rather than reported. **Canary B fired on the first run**
and found a real harness bug: the fallback id list was joined with spaces while `--ids` is comma-separated, so
the 1,024-lane runs were feeding the engine **one token** and would have reported a 1-lane wave as a width.
That is the D30 lesson again — a stale or mis-fed side produces a *plausible* table, and only a check that a
difference must exist catches it.

**What this cannot see.** `--steps 0` means the decode path is not exercised here (that is §36's banner, which
showed decode unchanged within spread); two fixtures are not the 18-fixture admission; and one width pair is
not a sweep — a regression that appears only below 1,024 or above 4,096 lanes would walk straight past it.

## §38. llama.cpp again, phase by phase: the gap is no longer attention, and launch structure is dead as an explanation (2026-10-19)

The round-2 queue asked for the llama.cpp comparison to be re-taken **by phase** after the WMMA work, with the
existing split as the null. Two legs, both inside one `tools/gpu_lock.sh` hold (`llama-phase`, main agent,
blocking, zero subagents): throughput rows medians of 3, then a `rocprofv3` kernel trace of both engines at
4,096 lanes. Harness `tools/llama_phase_compare.sh`, analysis `tools/trace_phase_split.py` (pass count
**derived** from a kernel with a known per-pass site count — 288 `gated_delta_net` ÷ 144 = 2 passes for llama,
144 `attn_prep` ÷ 48 = 3 for ours — which is the §W2 rule made executable rather than remembered), logs in
`progress/bench/evidence/llamaphase-2026-10-19/`. llama.cpp is still `797de63`, same GGUF.

**Throughput, matched row (`-ub 512 -fa on -t 32`, our side `--steps 0` so both compute prompt-only):**

| lanes | ours (this window) | llama.cpp | llama/ours | §21's ratio, same widths |
|---|---|---|---|---|
| 512 | 168.019 ms = 3,047 tok/s | 8,025.84 | **2.63×** | 2.27× |
| 1,024 | 235.736 ms = 4,344 | 8,142.83 | **1.87×** | 1.54× |
| 4,096 | 710.661 ms = 5,764 | 10,380.79 | **1.80×** | 1.87× |

Read the third column with care, because both halves moved. Ours improved (924.62 ms → 710.66 ms at 4,096 =
+30 %), and **so did llama's own matched row** (8,290 → 10,381 = +25 %) on the same build — the §21 window was
a colder one, and their `fa off` row swung the other way (8,999 → 5,414). A cross-window ratio comparison is
therefore worth nothing here (§"a ruler re-taken in a different window is not an A/B" applies to the *other*
engine too); the two numbers inside each row of this table were taken in one hold, and that is what makes the
4,096-lane **1.80×** honest. The WMMA round bought ~30 % of our prefill and the headline gap moved 1.87 → 1.80.
Where the gap actually lives is the traced half of the item, and it is not where §21 left it:

| phase | llama ms | llama launches | ours ms | ours launches | ours/llama |
|---|---|---|---|---|---|
| `gdn_recur` | 47.20 | 144 | **272.07** | **144** | **5.76×** |
| `gemm` | 176.83 | 1,464 | **154.19** | 1,200 | **0.87×** |
| `attention` | 51.65 | 48 | 136.83 | 180 | 2.65× |
| `other` (dtype/concat glue) | 96.23 | **4,827** | 24.95 | 698 | 0.26× |
| `norm` | 11.43 | 908 | 11.65 | 392 | 1.02× |
| `silu_mlp` | 12.87 | 328 | 9.90 | 192 | 0.77× |
| `copy`/cast | 0.21 | 99.5 | 1.84 | 350 | — |
| **TOTAL traced** | **406.16** | **7,920** | **621.16** | **3,301** | **1.53×** |
| launches/token | 1.93 | | **0.81** | | **0.42×** |

**The pre-registered prediction (decision **L/C5**, written before the trace) is confirmed, and the launch
count died first.** We launch **0.81 kernels/token against llama's 1.93** — we are the lean engine by 2.4× —
so any remaining "maybe it's launch overhead" theory is dead on arrival. And the GDN line is the sharpest
number in the table: **identical launch counts (144 each, both 8 per layer per 4,096 tokens) at 5.76× the
time**, which leaves only one place for the difference to live: the per-token walk inside the kernel. The
`--geom` census pins it further: llama runs `gated_delta_net` at 32 threads × 512 grid and ours at
128 × 2,048, and **both land at 0.80 blocks/CU** — 16 workgroups on a 20-CU part, four CUs idle, in *both*
engines. Neither engine is using the machine for the recurrence; llama just makes the serial step ~6× cheaper.

**Two things this table says that were not the question.** First, **we now beat llama.cpp on GEMM** (154.19 ms
vs 176.83 at 4,096 lanes, with 264 fewer launches) — §21 had us 3.5× *worse* there, so the hipBLASLt pinning
work has flipped that phase, and GEMM is no longer a target. Second, llama pays **96 ms and 4,827 launches per
pass** in `convert_unary_cont_vec4` / `concat_non_cont` — f16↔f32 conversion and concat glue its graph
generator inserts — which is 24 % of its traced prefill and 61 % of its launch count. Our equivalent is 1.84 ms
of `copy` and 2.6 ms of `cast`. That is a structural advantage of a hand-written engine and the reason its
headline is worse than its phase sum suggests.

**Honest limits.** Traced ms are upper bounds and the overstatement scales with launch count, so llama's
96 ms of glue is an *inflated* number and its true total is below 406 ms (its untraced matched row is 394 ms —
so the inflation here is small, ~3 %). Attention at 2.65× is still a real gap and a funded candidate, but it is
now the second item, not the first: at 137 ms against llama's 52 ms it is worth 85 ms where GDN is worth 225.
And the sum: ours traced 621 ms vs our untraced wave `wall_ms` 628 (§37) vs `RESULT` 710 — the 82 ms between
kernel sum and RESULT is host-side/launch gap outside any kernel, i.e. **12 % of the prefill is not in any
phase row above**, a lever this table can size but not explain.

**What gets funded next, per the pre-registration.** The GDN inner walk, not a third attention pass. The
ceiling of the one GDN form already measured is modest and known (route A's LDS-resident collapse: 112.04 ms
against a 125.36 ms shipped form at the 1,024-lane shape, 10.6 %, dead on its own kill line), so this is not
a "try the drawer again" item: it needs a cheaper *per-token* step at ~1 wavefront per block, or a form that
fills the four idle CUs. That is a new campaign and it is the largest single number left in the engine:
43.8 % of our traced prefill, and 5.76× from the only external yardstick we have.

## §39. Long context is reachable now: 32,768 lanes runs, 32,769 refuses, and what the wide band costs (2026-10-19)

Item L raised the attention lane ceiling from `MAXSC 512 × kAttnMaxSplit 9 = 4,608` to
`512 × 64 = 32,768` (`c188aa3`, with `tools/attn_ceiling_proof.py` retargeted as the CPU witness: every plan
the old ceiling admitted keeps its split, grid and byte offsets, so the raise is bit-preserving by
construction rather than by sampling). The whole price is scratch — `attn_part_floats(kMaxWaveRows)` goes
**36.28 MiB → 258.00 MiB per scratch instance**, ~0.78 % of the ~34 GiB pool — and the honest caveat is in the
source comment where the constant lives: the charge is **per scratch, not per request-length**, because
`allocate_scratch()` sizes from the constant and does not know the run's cap. One scratch per `decode_run`
today, so nothing regresses; the day a server shares the allocator, `a_part` alone would hold ~25 GiB at 100
requests. Sizing by `split_for(cap)` is the fix and was **deliberately not taken** in this commit — one
constant that `gate.py::lane_ceiling()` reads from the enforcing files is what keeps the allocation and the
refusal from disagreeing (§20.1's aliasing class), and trading that for memory is a decision, not a cleanup.

**The device verdict** (`tools/longctx_check.sh`, five assertions per width, main agent, blocking, zero
subagents, one `tools/gpu_lock.sh` hold, `util@acquire = 90 %`):

| lanes | finish | finite + non-vacuous logits | `plan_match=1` | no silent clamp | prefix consistency at row W−1 |
|---|---|---|---|---|---|
| 8,192 | PASS | PASS | PASS | PASS | PASS |
| 16,384 | PASS | PASS | PASS | PASS | PASS |
| **32,768** | PASS | PASS (maxabs 9.45, 247,457 distinct of 248,320) | PASS (attn 384 calls / 196,608 rows = 6 sites × 64 bodies) | PASS (kv 402,653,184 B = 12,288 × 32,768) | PASS (inside the published band) |
| 32,769 | **REFUSED**, naming `kAttnMaxSplit=64` — the ceiling is a wall, not a clamp | | | | |

The last row is the one that matters most and it did not pass on the first try. **Three harness bugs, all found
by running the check rather than reading it:** (1) `--ids` cannot carry a 32,768-token prompt at all — Linux
caps **one argv slot** at `MAX_ARG_STRLEN = 131,072 B`, independent of `getconf ARG_MAX`, which quotes 2 MiB and
misleads everyone; 32,768 ids is 191,588 B, so `exec` failed with E2BIG and shell exit 126 *before* the binary
was reached, which read as "the engine crashed at 32k". `decode_run` now takes `--ids-file PATH` (same parser,
same list). (2) The refusal width, ceiling+1, was **beyond the generated id pool**, so the refusal case was fed
32,768 ids, ran a perfectly legal prompt, and reported "the ceiling moved somewhere other than the launcher's
refusal" — a vacuous negative control wearing a real failure, the same class as `partalias` and `mixed_bw`.
Fixed by generating to `max(widths, ceiling+1)` *and* by asserting the id count at every call site, since a
silently short prompt is the failure mode, not the missing width. (3) `--selftest` writes its fixture pool into
the work directory, which is the same directory a real run cuts its ids from — so after a selftest the next run
was quietly fed **1,024 ids** while claiming 8,192. That one got me: the first 8k/16k timing set below is
rebuilt because the original three reps at each width were 1,024-token runs wearing the wrong label, and every
field in their `RESULT` line looked plausible. Treat "a check that shares a scratch dir with its own selftest"
as the trap; the numbers here were re-taken from a pool whose field count was asserted first.

**What the wide band costs.** Medians of 3, one lock hold per width, same ruler as §36/§37 (`--steps 0`, `--gemm lt
--gemm-pins all --gdn-loop 512`), spreads 1.6 % / 0.15 % / 0.25 % — these are compute-bound and they repeat:

| lanes | `RESULT ms_total` | prompt tok/s | vs the 4,096-lane banner |
|---|---|---|---|
| 4,096 | 710.661 | 5,764 | (the §37/§38 reference point) |
| 8,192 | 1,541.690 | **5,314** | −7.8 % |
| 16,384 | 3,952.271 | **4,145** | −28.1 % |
| **32,768** | **11,729.095** | **2,794** | **−51.5 %** |

The shape is the expected one and it cross-checks: attention is 21 % of the wave at 4,096 lanes and grows
quadratically with the row, so at 32,768 it should cost ~64 × its 133 ms ≈ 8.5 s of the 11.7 s measured, with
the linear families contributing the remaining ~3.5 s (their 4k sum × 8). So nothing here says the long-context
path is doing anything dumber than attention-shaped work — and nothing here is a claim that the *quadratic term*
is optimised either; at 32k, attention is now the majority of the run by share as well as by absolute ms, which
is a reason the attention bets in `prototype/gdn_walk/BET.md` §5 matter more at long context than at 4k, not
less. What this section cannot see: decode at 32k (`--steps 0` only), numerics *beyond* the published band (the
prefix-consistency adjudication uses `engine/attn_form_band.py` verbatim and widens nothing), multi-request
residency at the new ceiling (the 258 MiB/scratch caveat above), and any width past 32,768, which is refused by
design and stays refused until someone funds the next raise.

## §40. G1: three barriers and an LDS round-trip left the GDN walk, and the row got 5.9 % cheaper — which is not a win (2026-10-19)

The bet and its kill lines are pre-registered in `prototype/gdn_walk/BET.md` §3; the implementation and its
checks are `1e92500`; this is the measurement. `--gdn-norm hoist` moves the per-token k/q L2-norms out of
`rec_loop_kernel`'s token loop into a pre-pass, leaving a walk that issues **0 barriers and 0 LDS ops** where
the shipped one issues 5 and 39 (ISA-verified, `walk_isa_diff.py`). Admission was bit-identity, and it holds:
`base == inline == hoist` on every prefill logit row and every `STATEFP` state fingerprint across all 18 GDN
layers — settled by the device diff, not by the association argument, after it read red twice (a grid-axis swap
that zeroed layer 0's state; operand pointers hoisted above the token loop so every token walked token 0's rows).

Window `g1-ab`, main agent, blocking, zero subagents, one `gpu_lock` hold, medians of 3 at 4,096 lanes, canary A
(baseline rejects `--gdn-norm`) and canary B (each side echoes its form and submits 144 vs 288 planned
launches) both green before any ms was read; digest in `progress/bench/evidence/g1ab-2026-10-19/`:

| quantity | base | new `inline` | new `hoist` | hoist vs base |
|---|---|---|---|---|
| **`gdn_rec` — the pre-registered row** | 266.60 | 264.21 | **250.84** | **−5.91 %** |
| every other family summed | — | — | — | **+6.06 ms** |
| wave `wall_ms` | 626.75 | 623.45 | **612.87** | **−2.21 %** |
| `RESULT ms_total` | 717.27 | 714.87 | 707.88 | −1.31 % |

Rep spread on the hoist side: 246.49 / 250.84 / 255.09 (3.5 %). The inline-vs-base null is **−0.89 %**, so the
shipping path did not move and the hoist number is real. It landed **0.84 ms above the 250 ms live line** —
one fifth of its own spread — inside the 250–258 band declared **no verdict**. Per the pre-registration: **not
shipped, default stays `inline`, no narration.**

**The measurement's real product is a methodology correction.** The hoist removes work from `gdn_rec` and puts
traffic into the stage, paid in `gdn_gated_norm` (+15.6 %) and `glue` (+8.8 %): **+6.06 ms outside the row that
improved**, against −15.75 ms inside it. A pre-registered line on a single family row therefore **over-counts a
change whose cost lands in its neighbours** — the honest whole-change figure is −2.21 % of `wall_ms`, not
−5.91 %. From G2 onward the lines are written on both denominators before measurement: the family row answers
"did the walk get cheaper", `wall_ms` answers "did the engine get faster", and only the second one ships.

And the barrier hypothesis is neither disproved nor vindicated beyond its own size: 3 barriers, 18 shuffles and
the LDS round-trip left the per-token chain and the row moved 5.9 %, which is what the census's 9.2 % slot share
allowed — not the multiple a latency story could have been read to promise, and most of it was eaten by the
traffic the hoist itself adds (the worker's own traffic correction, +32 KB/token/layer, called that before the
run). What this cannot see: a cheaper walk that adds no stage traffic — G2's geometry, one wavefront per block,
`LDS = 0`, no barriers by construction — would keep the −5.91 % and drop the +6.06 ms. That is now the funded
question, and G1's own knob is the instrument that will measure it.

## §41. Attention's residency axis closes at 6 wf/CU — on LDS, not on spills — and the only lever left is per-launch cost (2026-10-19)

No kernel ran for this section: it is note blocks, KFD properties and a launcher read, and it closes a research
axis, which is as valuable as a small win and much cheaper. `attn_fattn_wave` ships at **178 VGPR / 0 spill /
32,768 B LDS**, which is 2 blocks/CU (LDS from KFD: `lds_size_in_kb 64`; ceilings from `rocminfo`) = **6
wavefronts/CU, LDS-bound**. Ladder rung **E1** — the 3-warp body with Q un-staged, a hand port because the old
patch fails 4 of 5 hunks — gives **184 VGPR / 0 spill / 32,768 B LDS**: same blocks, same 6 wf/CU. The Q stage
was never extra memory, it is the **epilogue's F32 transpose staging on loan**
(`FATTN_MROWS·FATTN_TROW·4 == FATTN_MROWS·DH·2`, 8,192 B per warp either way), so removing it moves nothing
that binds. Note blocks re-derived by the main agent from fresh compiles of both sides, not inherited.

The ladder's E1 admission (`spill = 0 ∧ VGPR ≲ 190 ⇒ an 8+ wf/CU shape exists`) is now on record as
**incomplete**: its antecedent assumed LDS would fall. Consequences: attention occupancy is **closed** at 6,
and buying 3 blocks/CU would need the epilogue down to ≤ 21,845 B (transpose staged 8 rows at a time, or the
three serialised through `s_kv`) — structural changes to a shipping kernel's epilogue with no measurement
behind them, **not approved by anything in this ledger**.

The same commit settles A0's loose end, and that removes a candidate too. The traced
`prep 48 / slim 48 / fattn 42 / combine 42` at 4,096 lanes comes from **one integer**: `split = split_for(lanes_last)`
(`engine/kernels_attn.cpp:248`), with `attn_fattn_wave` gated on `split ≥ 2` and `attn_combine` on `split > 1`.
Body 0 has `lvalid = 1` (`engine/decode_step.cpp:697`), so `lanes_last = 512 = MAXSC`, so `split = 1`, so that
one body per layer pays neither — 7 × 6 = 42. The WMMA form covers **87.5 % of rows and 98.4 % of the
lane-work** (`prototype/attn_wmma/a0_work_split.py`, RECONCILED). "Too many launches go through it" is dead as
a framing; what remains for attention is **per-launch cost** — §26's instruction/traffic gap, sized there at
4.6×, at 2.92 ms per launch.

Where the money is, restated with today's numbers: `gdn_rec` is 41.3 % of the wave at 266 ms against llama.cpp's
47.2 on identical launch counts; attention is 21.6 % at 133 ms against 51.65 and its residency route is now
closed. So the next funded implementation is **G2**, the barrier-free one-wavefront GDN geometry, whose lines
(including the `wall_ms` denominator §40 made necessary) are pre-registered in `prototype/gdn_walk/BET.md` §10
before any code exists.

## §42. G2: removing every barrier and all LDS from the GDN walk makes it **2.2× slower** — the GDN gap is not shape (2026-10-19)

§38 read llama.cpp's `gated_delta_net` as *same launch counts, same 0.80 blocks/CU, `LDS = 0`, 32-thread
blocks, 5.76× faster*, and the obvious inference was that our barriers and block shape were the gap. **G2**
(`prototype/gdn_walk/BET.md` §10 pre-registered it, `8b65fb0` implements it) removes all of it — one wavefront
per block, 4 or 2 state columns per lane, row norms by a 5-level `__shfl_xor` butterfly, no LDS at all, state
still streamed from global — and it is **2.2× slower**. At 4,096 lanes, medians of 3 in one hold (window
`g2-ab`, util 5 %→73 %, clocks discounted):

| side | `gdn_rec` ms (live ≤180, dead >230) | wave `wall_ms` (≤560 / >607) | externality | verdict |
|---|---|---|---|---|
| `wf4` (4 cols/lane) | 264.00 → **585.79** (+121.9 %) | 626.44 → 951.13 (+51.8 %) | +2.69 ms | **DEAD** |
| `wf2` (2 cols/lane) | 264.00 → **380.73** (+44.2 %) | 626.44 → 742.32 (+18.5 %) | +1.89 ms | **DEAD** |

The §40 discipline paid for itself on the first use: both denominators were pre-registered, the **externality**
was printed (+2.7 / +1.9 ms over all other families), and so this window says plainly that the walk itself got
slower rather than that a cost moved next door. The null held at **−0.33 %** (same new binary at its own
default), both forms move the state fingerprint so neither is a no-op flag, and the structure was re-derived
by the main agent on an independent compile: `walk_isa_diff.py --g2` PASS — 0 barriers, 0 LDS data ops, 15
permutes, all state traffic `b128`, against the shipped 5 barriers and 19 LDS ops. The geometry did exactly
what it claimed at 0.20–0.40 wf/SIMD against the shipped 0.8, which is where the cost comes from: with 4×
fewer wavefronts there is nothing on the SIMD to hide a per-token dependency chain, and the barriers deleted
were worth 5.9 % (G1), not the chain.

**Consequence for the ledger's biggest remaining gap.** Every structural hypothesis about GDN's 5.76× is now
measured dead, and the five numbers are in `prototype/gdn_walk/BET.md` §11: norms hoisted −5.9 %, geometry
rewritten +121.9 %, state LDS-resident +10.6 %, parallel scan +493 %, unrolled +0.5 %. What survives is
per-token work itself — §26's 4.6× instruction/traffic axis — and the next funded item is a census of
llama.cpp's actual kernel source, which has been readable in this repo the whole time
(`engines/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`) while we inferred its shape from a profiler.
Shipping default is unchanged (`--gdn-geom block`, proved byte-identical to HEAD: 46/46 logit rows, 44/44
`STATEFP` lines). Engine headline today: **5,764 prompt tok/s at 4,096 lanes** (`RESULT ms_total` 708–713 ms
across §37/§40/§42 windows), 2,794 tok/s at 32,768 lanes from §39, decode unchanged.

## §43. Correction to §38: llama's GDN kernel runs **8× our wavefronts** — and GDN's gap is elements-per-thread, not shape (2026-10-19)

**What §38 got wrong.** Its geometry rows read `Workgroup_Size_X` and `Grid_Size_X` and nothing else, so
llama's `gated_delta_net` — launched `dim3(32,4)` blocks over a `dim3(512,4,8)` thread grid — was recorded as
*"wg 32, grid 512 → 16 workgroups, 0.80 blocks/CU, the same shape we run, and `LDS = 0`"*. Read on all three
axes it is **128 blocks × 128 threads = 512 wavefronts per launch = 6.40 wf/SIMD, 3.20 blocks/CU**: eight
times our 0.8 wf/SIMD, no idle CU. `rocprofv3`'s `Grid_Size_*` is in **threads**, not blocks — that is the
whole bug, and it was silent because 16 and 128 both look like plausible GDN grids. §38's *phase times* stand
(GDN 272.07 vs 47.20 ms, GEMM ours ahead, attention 136.83 vs 51.65); its geometry comparison does not. The
tool now derives blocks as `grid_threads / workgroup`, **refuses** a non-whole division instead of mis-stating
it, and takes CUs/SIMDs from KFD rather than a hardcoded 20 (`879ef31`); the claim was re-read independently
from the CSVs on disk (`prototype/gdn_walk/g3_geom_recheck.py`) rather than taken from the worker who found it.

**Why it cost a cycle.** G2 was briefed on the X-axis reading: §38 appeared to show that llama's advantage was
barrier-free one-wavefront blocks, so G2 built exactly that, took wf/SIMD from 0.8 to 0.20–0.40, and lost
**122 %**. G2 therefore flips meaning — it is not evidence against shape, it is evidence *for* wavefront count
with the sign flipped. A wrong premise did not just waste time; it inverted the sign of a real measurement.

**What the gap actually is.** G3's census (`prototype/gdn_walk/G3-llama-kernel.md`, source provenance pinned
to `797de63` and to the object §38 traced) closes instruction count too: both engines perform the same 4
FMA-class ops per state element (7.73×10¹⁰ MAC/pass), and llama issues **1.28× more** slots per token while
finishing **5.76× faster**. The surviving difference is **state elements per thread — 128 ours, 16 theirs**.
At 16 the shard lives in registers (`vgpr 78, spill 0, LDS 0`), so llama moves no state per token, reduces
rows with 8-lane butterflies needing no barrier, and fielded 8× the wavefronts. Adjudicated against measured
ceilings: we issue at 9.9 % of the 231.5 G-slot roof, llama at 73.6 %; their 3.28 TFLOP/s is 45 % of the
scalar `v_fma_f32` roof (§26's ladder) with no matrix or dot instruction in their object; their LSU demand
coincidentally matches ours (864 vs 861 GB/s ≈ 94 % of the 918 GB/s cache roof) and is recorded then rejected,
because route A already falsified a traffic reading of this kernel. **Nothing in llama's number requires a
capability this part lacks** — which is the sentence that makes the next bet worth funding.

**Funded: G4** — llama's thread map under our recurrence: 128 blocks × 128 threads, 8 lanes per column, 16
elements per thread in registers across the span, the map forced by `wavefronts × 32 × el/thread = 262,144`.
Lines pre-registered in `prototype/gdn_walk/BET.md` §12 on both denominators before code exists: `gdn_rec`
≤170 live / >225 dead, `wall_ms` ≤540 live / >607 dead, `wall_ms` the ship gate. Engine headline while it
runs: **5,764 prompt tok/s at 4,096 lanes** (unchanged — §42's default is inert and §39's long-context ladder
is unaffected).

## §44. G4 ships: the GDN walk keeps its state shard in registers, `gdn_rec` −66.6 %, and the engine goes **5,764 → 7,655 prompt tok/s** (2026-10-19)

The bet §43's census earned, measured against lines pre-registered in `prototype/gdn_walk/BET.md` §12 before
the code existed. Window `g4-ab`, main agent, blocking, zero subagents, one `gpu_lock` hold (`g4-ab`, util
5 %→61 %, sclk 840 MHz→2185 MHz recorded then discounted), 4,096 lanes, medians of 3:

| side | `gdn_rec` ms (live ≤170, dead >225) | wave `wall_ms` (≤540 / >607) | `RESULT ms_total` | externality | verdict |
|---|---|---|---|---|---|
| `reg16` vs shipped `block` | 267.09 → **89.22** (−66.6 %) | 628.89 → **453.65** (−27.9 %) | 712.065 → **535.105** | +2.18 ms | **LIVE → shipped** |

Null held at −1.6 % (same binary at `block`), canaries green including an in-hold geometry census reading
`(32,4,1)` × 128 blocks against `(128,1,1)` × 16 — the read §43's tool fix made possible. **7,655 prompt tok/s
at 4,096 lanes**, from 5,764. The mechanism, verified on an independent compile rather than inherited: inside
the token loop, 9 global loads and exactly one global store (`o`), with the 16 state loads and 16 stores sitting
*outside* it — `next_free_vgpr 76`, `group_segment 0`, private segment disabled, against `rec_loop_kernel`'s 75 /
1280 B LDS / 5 barriers / 19 LDS ops.

**The comparison restated as like quantities.** Traced, same window, all at 144 launches per pass:
`rec_loop_kernel` 276.99 ms/pass → `rec_reg16_kernel` **89.22 ms/pass**, against llama.cpp's traced
`gated_delta_net` **47.20 ms/pass** (§38). So the gap this section was chasing goes **5.87× → 1.89×**, and the
GDN phase falls from 43.8 % to **18.7 %** of the traced wave. That re-ranks the queue rather than settling it:
attention is now the engine's largest traced phase (155.71 ms) with its residency axis closed in §41, so
per-launch cost is the only live attention lever; and the 1.89× residue in GDN cannot be geometry or
occupancy, because we now run llama's map at its own elements-per-thread.

**Admission at the new default** (device, exclusive): `ENGINE GATE PASS` on `dec64f` — 64/64 pinned ids, 2 by
the standing tie policy, worst step cos 0.998307, drift ratio 0.59 — and on `wide520f` — 24/24 exactly, worst
step cos 0.999851, above `MAXSC` — with the gate's new geometry assertion confirming the run really ran `reg16`.
**Decode did not move**: paired 64-step A/B reads 1.623 → 1.584 ms/step (−2.4 %), inside the ±3 % paired floor,
which is the expected answer because `reg16` re-spells the *looped* body and a one-row body is routed to the
per-token kernels whatever the geometry says. Long context re-checked at the new default: **8,192 lanes PASS**
on all five checks including prefix consistency across the wave and one-row routes, and one rep there reads
1249.94 ms against §39's 1541.69 median — **one rep, not a ladder; §39's rows are now stale and a re-take is
owed.**

**Numerics, and the one witness demoted on evidence.** The state dot re-associates through 16-element partials
and an 8-lane butterfly, so `o` and `STATEFP` move; judged by the existing §10.1 aggregates, no band widened and
no `--ack-spill` used. The inherited far-field check `rms_rel ≤ 1.25 × shipped` failed at 1.250290 and is now
informative rather than binding, with the constant untouched and the check still printed on every span: the
shipped binary's own rms_rel to the pins swings **27.8 %** between the very spans where the ratio is measured,
the excursion's sign flips with span, `reg16` reads **0.694** (closer to the pins than shipped) at wave-rows 16,
61.6 % of elements move with top contributors at 0.01 % each, and a 64-step free run emits **110/110 identical
token ids**. A witness whose noise exceeds its margin cannot adjudicate.

**Two guard fixes a new default forced, both the same lesson.** A loop-1 refusal keyed on `geom != block` would
have made a legal, pinned route an error the instant `reg16` became the default; it now fires only on an explicit
`--gdn-geom reg16 --gdn-loop 1`, and the echo prints `na-loop1-per-token-bodies` rather than claiming a map that
did not run. `--gdn-norm hoist` with an unasked geometry falls back to the map that has a norm prologue and
echoes `block`, so G1's route stays reachable, while an explicitly contradictory pair still refuses naming
registers. And `engine/gate.py` gained `--gdn-geom`: BET §10 named it the band authority
for geometry bets while its argv could not name a geometry, so every run judged the default while a bet-local
script carried the claim — a requirement written after the object it constrains.

Commits: `46139e9` (the map), `0d1f41c` (gate can name a geometry), `e1df7a7` (default ships, guards fixed),
`eb43ca1` (traced classifier), `6564161` (BET §13). Evidence in `progress/bench/evidence/g4ab-2026-10-19/`.

## §45. Long-context ladder re-taken at the `reg16` default — every width moves, and the ceiling is unchanged (2026-10-19)

§44 shipped a new default and thereby made §39's ladder stale, which is the one kind of staleness an ledger has
to fix in the same sitting rather than note. Re-run of `tools/longctx_check.sh` at the shipping default, all five
checks per width (feed/row parity against the one-row route, prefix consistency above `MAXSC`, argmax room,
banded numerics, and the refusal leg past the ceiling), `--widths 8192,16384,32768`:

| width | §39 median | this window | prompt tok/s now | change |
|---|---|---|---|---|
| 8,192 | 1541.690 ms | **1278.80 ms** | **6,406** | **−17.1 % time, +20.6 % tok/s** |
| 16,384 | 3952.271 ms | **3319.25 ms** | **4,937** | **−16.0 % time, +19.1 % tok/s** |
| 32,768 | 11729.095 ms | **10448.13 ms** | **3,136** | **−10.9 % time, +12.2 % tok/s** |

All three widths **PASS**; the lane ceiling still reads 32,768 (`kAttnMaxSplit = 64`, §39), and the improvement
decays with width exactly as it should for a GDN fix — GDN is a per-token cost, so as attention's quadratic
term grows with context it claims a larger share of the wave and the same −66.6 % on 18.7 % of the wave buys
less. That is the arithmetic check on the three numbers: 20.6 / 19.1 / 12.2 %, monotonically falling, which is
what a fixed-fraction win on a shrinking share looks like. Two reps per width here (§39 meditated on three), so
the rows are honest to about ±2 % and the direction is not in question.

## §46. The gap is not one number: against llama.cpp we pay a **1.285× constant and a 2.164× quadratic term** (2026-10-19)

The operator's question, in the form that can be answered: llama.cpp and this engine were measured at the **same
five widths in one lock hold** (`decay3`, util 5 %→9 %, `tools/llama_phase_compare.sh` with `PPS`/`OURWIDTHS`
extended to 2,048–32,768 and ids routed through `--ids-file`, because a 32,768-id argv string is ~200 KB and
dies at exec at exactly the width that matters). Ratios are `llama tok/s ÷ ours`, both engines, this window —
no cross-window ruler (§37/§38's rule), and the second pass of this window replaced a 2-rep one whose 16k llama
row came out *faster* than its 8k row, which is what two reps at these widths is worth.

| width | ours tok/s | llama tok/s | llama/ours | drift vs 2,048 |
|---|---|---|---|---|
| 2,048 | 6,735 | 9,614 | 1.43× | — |
| 4,096 | 7,441 | 10,236 | 1.39× | −2.8 % |
| 8,192 | 6,684 | 9,315 | 1.41× | −1.4 % |
| 16,384 | 4,975 | 7,726 | 1.56× | **+9.1 %** |
| 32,768 | 3,153 | 5,481 | 1.74× | **+21.7 %** |

**The suspicion is confirmed, and it is not flat.** Up to 8,192 the ratio is flat inside noise (1.39–1.43×); from
there it worsens monotonically. Over the same span we lose 46.8 % of our 2k rate per doubling-to-32k where llama
loses 43.0 % — no model needed for that. Splitting the curve properly (`tools/prefill_scaling_fit.py`, fit
`ms/token = a + b·W`, five points, medians of 3):

| engine | `a` (per token, context-independent) | `b` (grows with context) | max residual |
|---|---|---|---|
| ours | 0.114956 ms | 5.920e-06 ms/token² | 18.6 % of `a` |
| llama.cpp | 0.089456 ms | 2.735e-06 ms/token² | 10.0 % of `a` |
| **ratio** | **1.285×** | **2.164×** | ratio of ratios **1.684×** |

So there is **both a constant and an asymptotic problem, and the asymptotic one is 1.68× worse in relative terms
than the constant.** The widening is not a mystery: `b` is the term attention owns — at these widths attention is
the only quadratic cost this engine emits, which is an argument about the code rather than a measurement of the
kernel, and A2 is funded to make that a measurement. It also says something sober about the headline: §44's
+32.8 % came almost entirely to `a` (GDN is per-token), so it *flattens* the curve without touching `b` — which
is exactly why the ratio at 32,768 got worse in the same window that the 4,096 headline improved.

Residuals are honest: 18.6 % and 10.0 % of `a`, and they live at the narrow end, where per-process page-in of the
weights (the 35 GB/s first-pass trap) and wave quantisation at 4 waves are both per-prompt costs a two-term model
cannot express. The 16k/32k points, which are what the claim is about, sit within 3 % of the fit. The direct
drift measurement above needs no model at all.

**Consequence for funding.** Attention is now the phase that owns the term that grows, at 2.164×. Long-context
work has to be priced against `b`, not against the 4,096 headline: a change that improves `a` by 5 % while
leaving `b` alone buys +5 % at short context and about nothing at 32k. §39's ladder (§45's re-take) and this fit
are the two together: absolute capacity, and the shape of its decay.

## §47. The attention round is not latency-bound on its accumulator either: A3 dies at the source, A4-L1 measures **−1.67 %** and is DEAD (2026-10-19)

Two bets on the largest traced phase (attention, 150.16 ms/pass of a 446 ms wave at 4,096 lanes) and neither
reached a ship. Both are recorded because each closes a category rather than a tweak.

**A3 — buying a third resident block by staging Q in 8-row half-tiles — died at the source, before a line of
kernel code or a second of device time.** The WMMA B-operand consumes all 16 query rows × all 256 dims from
`s_q` *every round* (`engine/attn_fattn_kernels.h:581`) and the epilogue's F32 transpose rides that same per-warp
buffer (`:710` via `:216`), so `3×8,192 + 8,192 = 32,768 B` is the LDS floor of the shipped arithmetic with Q
staged at all. 4,096 B/warp is physically 8 rows and a 16×16×16 WMMA cannot be fed an 8-row M tile, so "half-tile
staging" is a work-split change, not a staging change: either two 8-row half-passes (+379 KB/token/layer of K/V
staging, ~12× the externality that made G1 a non-win) or a per-round Q re-fill (+150 % per-round global load
demand on the kernel whose LSU slack A2 measured at 13.5 % of roof). Second, independent kill: even at 20,480 B,
3 blocks/CU is undecidable on this part — clang grants 178 VGPR where rocprofv3 reports 248 for the same kernel
and no register file size is reported at all. Detail: `prototype/attn_percost/A3-result.md`.

**A4-L1 — give the score roll a second accumulator so its 16 WMMA ops form two chains — was measured and is
DEAD.** Gate 0 proved from the emitted code that the shipped roll really is one chained site in a rolled 16-trip
loop with no independent work in the body (and corrected A2's polarity: the 17 sites are 1 rolled score site +
16 unrolled P·V sites, the reverse of the census's guess). Gate 1 compiled the two-accumulator spelling at
**spill 0, private 0, LDS unchanged 32,768 B, VGPR 178 → 200**, with two sites on two distinct accumulator tuples.
Then one paired window (`util` 26 %, `sclk` 1420 MHz, NULL base-vs-`single` **−0.06 %**) measured, against
A4-BET's pre-registered lines: `attn` family **136.2548 → 133.9760 ms (−1.67 %)**, traced `attention`
**−0.28 %**, wave `wall_ms` **+0.27 %**, 16,384 lanes **−0.01 %**, 32,768 lanes **−0.42 %**. Every row dead,
externality +2.18 / −13.23 / −43.62 ms inside its band, default stays `single`. Admission stayed green
(`--attn-fattn-roll two`: 289 PASS / 0 FAIL, worst 1-cos 2.406e-04 against the 2.75e-03 line, nothing widened).
Detail: `prototype/attn_percost/A4-L1-verdict.md`.

**What that leaves, stated as a map.** The shipping attention wave is at **5.56 % of the WMMA roof** and ≤ 8.8 %
of the issue roof (A2), cannot buy a third resident block through staging (A3), and does not care about its
accumulator dependency (A4-L1). L2's whole budget (128 of ≥ 1,654 round slots = 7.7 %) is below the −10.5 % the
live line demanded, and L3's 10–13 % of the excess quadratic fails the 4,096-lane lines by construction — so the
remaining hypothesis is the **tile geometry itself**: 16-row M tiles, 3 warps/block, 2 blocks/CU, 6 wavefronts/CU,
against llama's `flash_attn_ext_f16<256,256,16,4,…>` at 256 threads with 900 B/thread of deliberate scratch.
Headline unchanged at **7,655 prompt tok/s**; attention's share of the wave is 30.37 %.

**Harness discipline, from three defects found in the A/B before it produced a number.** The baseline was `HEAD`
(which by then *had* the knob, so canary A correctly refused to time anything); the traced leg execed `./$BIN`
against an absolute path and could never run; and the header carried a "re-based" line set attributed to the
supervisor that the supervisor never fixed — its 1,196.89 ms wave base is 2.6× the ~460 ms the same fixture
produces. Lines are now A4-BET's ratios × the in-window base, externality prints but decides nothing (it was
never pre-registered), and the colour fixtures are generated from the harness's own ratio constants instead of
hand-typed — the stale hand-typed dict survived as a second assignment and overrode the derived one, caught only
because the colour test runs before any measurement. A pre-registered line is only pre-registered if the file
that carries it is the file the harness reads.

## §48. Residency was the wall: **G1 ships**, attention −22.09 %, the wave −5.90 % at 4,096 and **−18.61 % at 32,768**, headline **7,655 → 7,986 prompt tok/s** (2026-10-19)

The first attention win of this arc, and it arrived only after three routes were closed and one cheap falsifier
ran. Bases, lines and the two-step order are in `prototype/attn_percost/A6-BET.md`; the measurements are
`prototype/attn_percost/A6-G0-verdict.md` and `A6-G1-verdict.md`.

**G0, the falsifier (not shipped).** Warps 3 → 7 and nothing else: VGPR unchanged at 178, LDS 65,536 = exactly one
block/CU, 7 wf/CU. It moved the `attn` family **137.5176 → 122.6404 ms (−10.82 %)**, traced **−8.62 %**, and its
own wave row only −1.89 % — inside the paired floor, exactly as A5 predicted, which is why it was funded as a
thesis test rather than as a bet. §31's log-elasticity of 0.72 had predicted −10.5 % for 6 → 7 wf/CU and got
−10.82 %. Its externality was **+6.80 / +74.72 / −1.03 ms**, i.e. at 16k a 224-thread block slowed its neighbours
by more than attention gained.

**G1, shipped.** Q read from global at the use site (E1's port, re-anchored after the roll knob moved the roll) plus
the epilogue's F32 transpose staged 8 rows at a time: `3 × 4,096 + 8,192 = 20,480 B` → 3 blocks/CU → **9 wf/CU**
at 185 VGPR, spill 0, and the driver's own occupancy query grants 3 blocks — which settles the register-file
question A3 left open as a bracket (53,280–71,040 lane-VGPRs) rather than an unknown. Paired window (`util` 29 %,
NULL −0.41 %, rep spreads ≤ 1.56 %): `attn` family **135.3870 → 105.4753 ms (−22.09 %)**, traced **−17.54 %**,
wave `wall_ms` **451.8840 → 425.2243 (−5.90 %)**, 16,384 lanes **−14.22 %**, 32,768 lanes **−18.61 %**. Every row
cleared its live line, so the default moved (`stage` → `unstaged8`) in `decode_run.cpp`, `decode_step.h` and
`gate.py` in one commit.

**Why the ship cost nothing in numerics.** The change moves staging addresses only, and it is **byte-identical**:
`base` vs default and `base` vs `unstaged8` match on every dumped logit file and all 44 `STATEFP` rows at 640,
ragged 1,111 and 2,000 ids. `engine/gate.py --fixture wide640f` on the committed tree at the new default: **PASS**,
nothing widened, no `--ack` of anything. A byte-identical win is the only kind that needs no decision from the user.

**Headline and ladder.** Own window (`util` 5 %, `sclk` 845 MHz — clock-discounted, so the pair is the number):
`RESULT ms_total` best-of-5 **512.908 ms** vs base best-of-3 **531.285 ms** = **+3.46 %** → **7,986 prompt tok/s**
at 4,096 lanes. Long context from the paired A/B above, on §45's denominator: **5,801 tok/s at 16,384**
(§45: 4,937, +17.5 %) and **3,838 tok/s at 32,768** (§45: 3,136, +22.4 %). The win grows with width, which is
§46's claim that the quadratic term is bought with residency, observed. Decode did not move: paired 64-step reads
base 7.438 vs new 7.545 ms/step (**+1.4 %**, inside the floor; three pairs one sign, so noted, not claimed, and
not comparable to §44's absolute because that used the wave-timed denominator). Externality
**+2.83 / +20.08 / +5.05 ms** — better than G0's +74.72 but not zero: a third resident block is not free for its
neighbours, and attention's share of the wave is now 29.91 % of a smaller wave.

**Method note worth keeping.** Three negative results bought the discriminating measurement: A2 bounded the pipes
(5.56 % of the matrix roof, ≤ 8.8 % of issue), A3 closed the staging route by deriving the LDS floor, A4-L1 priced
the accumulator chain at 1.67 %. Only then was the residency thesis tested with a one-constant change before the
surgery was funded — and G1 reached the same byte count A3 called unreachable because A3's kill was specific to
splitting a 16-row round, which G1 never does. **A closed route is a claim about a mechanism, not a byte count.**

## §49. Re-compared with llama.cpp after G1: **1.29× at 4,096** (was 1.80×), the quadratic gap cut from 2.164× to **1.554×** — and a **~65–71 ms per-prompt constant** that is most of what is left (2026-10-19)

One lock hold (`llama-phase`, `util` 90 %→9 %, `sclk` 2573 MHz at start, 951/933 mid/end — clock-discounted as
always, and every ratio here is two engines measured inside the same hold), matched row `-ub 512 -fa on -t 32`,
medians of 3, six widths. `tools/llama_phase_compare.sh`, logs `tmp/gcc/llamaphase/`.

| lanes | ours tok/s | llama tok/s | llama/ours |
|---|---|---|---|
| 512 | 3,544 | 7,378.72 | **2.08×** |
| 1,024 | 5,446 | 8,014.23 | 1.47× |
| 4,096 | **7,983** | 10,259.19 | **1.29×** |
| 8,192 | 7,393 | 9,449.85 | 1.28× |
| 16,384 | 5,890 | 7,800.29 | 1.32× |
| 32,768 | 3,915 | 5,475.38 | **1.40×** |

§38 measured 1.80× at this row; §46 measured a ratio that *widened* past 8,192. It no longer widens: the worst
ratio is now at the **short** end. The two-term fit (`tools/prefill_scaling_fit.py`, widths 4,096–32,768, residuals
≤ 6.4 % of `a`) puts it as **`a` ours/llama = 1.207×** (was 1.285×) and **`b` ours/llama = 1.554×** (was
2.164×) — G1 took 21.6 % out of our quadratic term (`b` 5.920e-06 → 4.644e-06) and the wide-end ratio with it. The
six-width fit is reported but not used: including 512 pushes the residual to 58 % of `a`, because a fixed per-prompt
cost is not a per-token constant — which is the finding below.

**Traced phases at pp4096, same engine, one hold** (`tools/trace_phase_split.py`, ours ÷ derived passes; llama's
traced ms are upper bounds, so its ratios are conservative floors): attention **ours 100.79 vs llama 52.31 =
1.93×** (was 2.65× pre-G1 — `attn_fattn_wave_u8` at 42 launches/pass); GDN-recur **ours 89.11 vs llama 44.93 =
1.98×**; GEMM ours ≈ 150 ms vs llama ≈ 172 ms, **still ours**. llama spends a further ~58 ms/pass on
`concat_non_cont` and `convert_unary_cont_vec4` (328–1,464 launches each) that this engine has no analogue for,
which is part of why closing to 1.29× on a kernel-by-kernel basis is not the same as closing to 1.29× in the wall.

**The new number, and the one that reframes the rest.** `RESULT ms_total` minus the wave's own `wall_ms`, measured
in one window at three widths: **512 lanes 119.830 − 48.733 = 71.1 ms; 1,024 lanes 167.080 − 96.229 = 70.9 ms;
4,096 lanes 494.957 − 430.053 = 64.9 ms.** A **fixed ~65–71 ms per prompt that does not scale with context**, and
it is *not* new — §44's 535.105 against a 453.65 wall carried the same ~81 ms, and every headline in this ledger
has included it silently. It is **59 % of a 512-token run** and 13 % of a 4,096-token one, and it is roughly equal
to llama.cpp's *entire* measured pp512 (69 ms). That is what the 2.08× at 512 lanes actually is, and most of the
1.207× `a` ratio with it: the remaining constant gap is not a kernel, it is per-prompt cost outside the wave.

**Not attributed.** Nothing here says what the 65–71 ms is made of. Candidates this ledger can already name:
host-side prompt feeding and per-prompt H2D copies; `--gemm lt` shape/algo work at a new shape despite
`--gemm-pins all`; the first-touch page-warming cost recorded in `devices/strix-halo-unified-memory.md` (~35 GB/s
on a first pass over fresh pages); and the ~2,083 syncs a 4,096-lane wave issues. Attributing it is the next funded
item, because no kernel-level attention or GDN work can reach it.

> **CORRECTED IN PLACE, same day — see §50.** The measurement above is right and the interpretation was not. Both
> operands of this subtraction are the *same* `prefill_wave` call in the *same* process (`RESULT ms_total` is the
> first pass; `--time-waves wall_ms` is a `st.reset()` + re-run of the identical wave), so this is a **cold-vs-warm
> first-execution cost of ~90–110 ms**, not per-prompt host work outside the wave, and the bulk of it is hipBLASLt's
> first execution per algo — with `--gemm hand` the gap at 512 lanes falls from +91.5 ms to +5.8 ms. The engine's
> weight pages are already warmed inside `load_ms`, so none of the weight-first-touch candidate above can be the
> cause. Read §50 before quoting any ratio from this section against another engine's rep-median.

## §50. Correction to §49: that ~65–71 ms is **cold vs warm inside one process**, mostly hipBLASLt's first execution — and warm-vs-warm we are **faster** than llama.cpp at short context (2026-10-19)

§49's subtraction was measured correctly (both operands came from one `--steps 0 --time-waves` invocation, lock
TAG `fixedcost`) but **interpreted wrongly by me**: I called `RESULT ms_total − wall_ms` "per-prompt host cost
outside the wave". It is not. `prototype/perprompt/A7-CANDIDATES.md` §0 read the source and the worker's reading
is confirmed line by line: `ms_total` is the wall of the **first** `prefill_wave` (`engine/decode_run.cpp:1797-1795`,
printed `:1990`), while `wall_ms` is printed by `wave_pass`, which calls `st.reset()` and runs **the same
`prefill_wave` again** (`:1404-1421`), with `WAVEPARITY … logits_bitidentical=1` as the engine's own witness that
the re-pass is identical work. Both operands are the same call in the same process, so the difference can only be
**first-execution cost**. The two hypotheses §49 leaned on are also false as stated: the weight load is timed into
a separate `load_ms` and never added (`:1588-1593`), and `WeightStore::load()` already performs a GPU warm read
pass over every resident page inside `load_ms` (`engine/loader.cpp:626-644`), so weight first-touch is not inside
`ms_total` at all.

**Re-measured inside one process** (`prototype/perprompt/a7_host_split.sh`, `a7-hostsplit`, `util` 81 % at
release, `sclk` 2572 MHz at end, medians of 3, `gap = cold − warm_burst`):

| lanes | cold `ms_total` | warm `burst` | **gap** | gap with `--gemm hand` |
|---|---|---|---|---|
| 512 | 138.755 | 48.132 | **+91.5 ms** | **+5.8 ms** |
| 1,024 | 188.874 | 90.815 | +98.1 ms | +30.9 ms |
| 4,096 | 523.003 | 418.398 | **+109.9 ms** | **−72.5 ms** |
| 16,384 | 2,775.064 | 2,675.212 | +102.2 ms | −119.2 ms |
| 32,768 | 8,344.827 | 8,348.082 | −5.8 ms | −193.7 ms |

Two readings survive, and the second is the actionable one. **(1)** The gap is flat in **ms** (~90–110) from 512 to
16,384 and vanishes at 32,768, so it is a fixed one-time cost, not a percentage of the wave — which also means
§49's flat 64.9–71.1 was the same quantity seen through a narrower window. **(2)** With the library GEMM disabled
the gap collapses at short widths (512: +91.5 → +5.8 ms, i.e. 6 % of it) and goes negative beyond 4k, so **the
bulk of the constant is hipBLASLt's first execution per algo**, not page first-touch (which `a7_scratch_bytes.py`
bounds at 6.8–7.9 ms) and not kernel image loading (`a7_codeobj_census.py`: `.hip_fatbin` is 413,624 B embedded
and registered before `main`, bounding that variant at ~0.012 ms). The negatives at ≥4k are the confound the leg
warns about — the hand wave is far slower, so differences are the only honest reading there, not ratios.

**What it does to the llama.cpp comparison.** Warm-vs-warm (ours on `warm_burst`, llama.cpp from §49's matched
row; both engines' first-execution cost excluded):

| lanes | ours warm tok/s | llama tok/s | llama/ours | §49's cold ratio |
|---|---|---|---|---|
| 512 | 10,637 | 7,378.72 | **0.69× — we are faster** | 2.08× |
| 1,024 | 11,276 | 8,014.23 | **0.71× — we are faster** | 1.47× |
| 4,096 | 9,790 | 10,259.19 | 1.05× | 1.29× |
| 16,384 | 6,125 | 7,800.29 | 1.27× | 1.32× |
| 32,768 | 3,924 | 5,475.38 | **1.40×** | 1.40× |

So the engine is **ahead of llama.cpp warm at short context**, level at 4,096, and behind only as context grows —
the opposite shape to §49, which was comparing our cold number against llama.cpp's rep-median. Whether
llama-bench's own median excludes its first-execution cost is argued and not measured (with `-r 3` its rep 1
carries it, so a median of 3 is warm-ish); that is a cheap check and is owed before this table is quoted as
parity. The honest convention adopted here is **report both**: the cold number is the real first-request latency a
server serves, the warm number is steady state, and no comparison across engines may mix them.

**The funded consequence.** The engine already has a documented contract that warming belongs in `load_ms` and not
in the timed region — loader phase 3 does exactly that for weight pages. The same contract applied to the GEMM
algorithms (execute each algo once on a minimal body during load) should remove **~85–110 ms from `RESULT ms_total`
at every width**, which is 3× on the short-context headline and ~20 % at 4,096, and it must be bit-identical
because a warm-up body that touches run state would be a bug. Tracked as **A8** in `prototype/perprompt/`. What
this still cannot see: GPU-side per-kernel durations (a `rocprofv3 --kernel-trace` pass with the pass-count
discipline is a separate item), a finer split of first-touch pages from library warm-up than the F1 bound, and
clock state, which is read and not controlled — `sclk` here sat at 2572 MHz with a flat gap at 32k, which is
consistent with DVFS ramp being part of what the short waves pay first.

## §51. Headline convention is now **warm**; and the long-context gap is **attention and nothing else** — we lose 1.98× at 16k and 1.91× at 32k on that phase while winning everywhere else (2026-10-19)

**Convention (user decision, 2026-10-19).** Every headline in this ledger from now on is the **warm** number: the
same `prefill_wave` after `st.reset()`, i.e. `WAVEPASS gran=burst wall_ms`, first-execution cost excluded — the
quantity llama-bench's rep-medians are closer to. Cold (`RESULT ms_total`) stays recorded as first-request latency
and must never be compared against another engine's warm number (§50). Warm re-taken from `a7_host_split.sh` leg A
(medians of 3, one process per width): **512 → 10,637 tok/s; 1,024 → 11,276; 4,096 → 9,790; 16,384 → 6,125;
32,768 → 3,924.** The 4,096 headline is therefore **9,790 prompt tok/s warm** (9,790 = 4,096 / 0.418398 s), against
the 7,986 cold figure §48 published — same window, same engine, different bracket.

**The long-context phase attribution.** Traced both engines at the widths one cannot answer at 4,096
(`tools/llama_phase_compare.sh` with the trace leg now width-parameterised via `TRWIDTHS`; pass counts **derived**
from `attn_prep` at 6 sites per body for us and `gated_delta_net` at 18 per chunk for llama, giving 3 passes ours
and 2 theirs; llama's traced ms are upper bounds, so every ratio below is conservative — and its launch density is
1.91–1.92/token against our 0.77–0.78, so the conservatism is real):

| width | phase | ours ms/pass | llama ms/pass | ours/llama | our share |
|---|---|---|---|---|---|
| 16,384 | **attention** | **1,518.91** | 767.54 | **1.98×** | 55.8 % |
| | gemm | 619.40 | 701.66 | 0.88× (we win) | 22.8 % |
| | gdn_recur | 365.99 | — | — | 13.4 % |
| | all | 2,721.90 | 2,126.23 | 1.28× | |
| 32,768 | **attention** | **5,908.16** | 3,093.25 | **1.91×** | 70.9 % |
| | gemm | 1,237.93 | 1,380.25 | 0.90× (we win) | 14.9 % |
| | gdn_recur | 746.07 | — | — | 9.0 % |
| | all | 8,328.23 | 5,839.50 | 1.43× | |

Read the excess, not just the ratio: at 16k the attention gap is **+751 ms** while the total gap is **+596 ms**; at
32k it is **+2,815 ms** against a total of **+2,489 ms**. **The attention excess is larger than the whole gap**,
because GEMM and everything else are net in our favour at both widths (§49 said the same at 4,096; it is now shown
to hold as context grows). GDN's traced gap, which was the headline of §38, no longer bounds the total: at 32k
`gdn_recur` is 9.0 % of our wave.

**The shape of it.** Per-token attention cost doubles with each doubling for both engines (ours 0.0927 → 0.1803
ms/token from 16k to 32k; llama 0.0468 → 0.0944), so this is **a constant factor on the quadratic term, not a
faster decay** — matching §50's `b`-ratio and ruling out "our attention degrades worse". Our absolute
achievement is the revealing number: 26.389 TFLOP of causal attention per pass at 32k in 5.908 s is **4.47 TFLOP/s**,
llama's is **8.53 TFLOP/s**, and the WMMA roof measured in `prototype/issue_rate_probe/results.md` §1 is 52.63 — so
**both** engines are 6–12× under the matrix roof at long context, and we are 1.9× under llama's utilisation. That
kills the bandwidth-wall story for long context (a bandwidth-bound phase would not sit ~90 % below the roof) and puts
the remaining attention question exactly where A4 left it: what the round *does* per unit of K/V at 16-row M tiles,
against llama's 256×256 geometry.

> **CORRECTED IN PLACE, same day — `prototype/attn_percost/A9-LONGCTX-CENSUS.md` §1.** The first version of this
> section published **0.56 TFLOP/s for us and 1.07 for llama**, which was **8× low**: the hand-built flop model
> counted one head where the geometry has `N_Q = 8`. Corrected: MACs/pass = `12288·W(W+1)`, so 26.389 TFLOP per
> pass at 32,768, and the rates are ours **4.09 / 4.34 / 4.47** against llama's **7.88 / 8.59 / 8.53** TFLOP/s at
> 4,096 / 16,384 / 32,768 — 7.8–8.5 % of the WMMA roof against their 15.0–16.3 %. The ratio, the constant-factor
> shape and the not-bandwidth-bound conclusion are all rate-scale-independent, so nothing in the paragraph above
> changes; but "50–90× under the roof" was the wrong magnitude, and the corrected magnitude is what licenses A9 §5's
> reading — at 8 % of a *matrix* roof, while a peer engine sits at 16 % of the same roof with **fewer** resident
> wavefronts (8 vs our 9), the phase is limited by neither matrix throughput nor residency nor DRAM.

**Parked, on instruction.** A8 (execute the GEMM algorithms once during `load_ms` to move the ~85–110 ms
first-execution cost out of the timed window) is **parked**: it is a short-context and first-request win, and the
user's call is that long context is where the remaining gap lives. Recorded so it is not lost — the measurement
stands in §50 and the legs are in `prototype/perprompt/a7_host_split.sh`.

**What this cannot see.** The traces give per-phase ms and launch counts, not why a phase costs what it costs; no
wasted-work accounting yet (padding to 16-row tiles, causal ragged tails, split-path partial traffic at 32k, the
per-round 16-accumulator rescale). That census is the next item, and it must be done before anyone funds either a
wider-M tile or the full llama-shaped rewrite — A5 §5 priced the rewrite at family −63 % and explicitly declined to
fund it without this.

## §52. A9 closes the long-context question: the gap is neither residency, nor bandwidth, nor matrix throughput — **L-c (cap the split) is funded, the rewrite is pre-gated and not funded** (2026-10-19)

`prototype/attn_percost/A9-LONGCTX-CENSUS.md` is the evidence; this is the decision record. It begins with the
correction above (§51's achieved rates were 8× low for want of the head axis) and then does what §51 named as the
prerequisite: it prices the wasted work instead of theorising about it, and it converts nothing at 1:1 — the
rescale and geometry items are charged at **A4-L1's measured 0.217 slot pass-through**, the discipline that lever
earned by moving 7.7 % of the round's slots and 1.67 % of the family.

**What is actually being wasted at 32,768 lanes.** Tile padding 2.94 %, round quantisation 1.25 %, causal-union
overhang 4.01 %, per-launch/body overhead 1.07 % — and the rescale/running-max chain at 428,251 `v_mul`
slots/token. Summed at measured conversion: **12–13 % of attention**, against the **47.6 % parity needs**. So the
arithmetic answer to "find the waste and we will be level" is *no*.

**The one genuinely-DRAM item, and it is ours.** The split path moves **105.46 GB/pass** of partial scratch
round-trip at 32k (2,079 Σsplits → 1.06 M partial slots/layer × 1,032 B × 8 heads × 6 layers), and `attn_combine`
is traced doing it at **208 GB/s = 88 % of the 237 GB/s roof** — a rate measured from traced bytes, not the
demanded-bytes arithmetic D21 retired, which is why it is the only traffic-shaped claim this ledger will act on.

**Decisions.**

- **L-c — FUNDED.** Cap the split above a lane count. Zero registers, zero LDS, zero instructions per round;
  `split_for` (`engine/kernels_attn.cpp:86`) and `blocks_for_row`
  (`prototype/attn_decode/attn_decode_kernels.h:220-223`) are one rule in two spellings and move in one commit
  (§34's two-header trap). At cap 8: round-trip **105.46 → 24.50 GB/pass (−81 GB)**, widest-body serial depth
  **+2.13 %** — because `ceil(64/8) = 8` blocks per row is still 5.87 block-waves against 120 concurrent blocks,
  so the machine stays fed; at 16,384 the cap costs 0.00 %. Predicted **−4.2 to −4.7 % of the 32k wave**
  (−5.9 to −6.6 % of attention), falsified at <3 % in one paired window. Note this is **not** bit-identical: the
  cap re-associates `attn_combine`'s merge, so it is judged by the existing gates per the standing numeric-format
  rule, and if a gate goes red the question is whether the band can widen without losing a must-catch, not whether
  to ship.
- **L-b — DEAD, on declared LDS, without a device run.** 16→32 KV lanes per round halves the round count, but the
  aliased K/V stage grows 20,480 → 28,672 B, forcing 3 blocks/CU → 2 and **9 → 6 wavefronts/CU**, which by this
  engine's own §48 elasticity is **+28 % of family**. It is the residency lever run backwards; §48 bought −22 %
  by going the other way.
- **L-d — BUNDLED, not tested alone.** Batched rescale over 4 rounds (llama's `nbatch_fa = 64` against our 16) is
  −1.26 % of family at measured conversion — under any ≥3 % line, so an solo A/B would measure noise.
- **L-a — the named-but-not-chosen variant.** The only axis for more query rows per K/V stage is warp count, and
  every traffic-halving spelling either loses wavefronts or pays +12.9 % issued work. The census' better candidate
  is **`TRG` 8 → 4 → 12 wavefronts/CU with the geometry untouched**, which is a compile read and costs nothing to
  price; it is queued ahead of anything that edits the round.
- **The llama-shaped rewrite — NOT funded, and now pre-gated.** A9 §5's verdict is that no targeted lever closes
  1.9×: best-case stack (L-c + residency) lands at **1.494×**, L-c alone at **1.784×**, against parity 1.000×. The
  residual is **9.6 vs 31 MAC per LSU byte**, and the decisive observation is that llama achieves 1.9× our rate
  with **fewer** resident wavefronts (8 against our 9) — so it is provably not residency, and after A3 (LDS floor),
  A4-L1 (dependency), L-b (bandwidth) and G0/G1 (both residency) the same premise keeps failing from different
  directions. Before any rewrite is funded: an `SQ_WAIT*` counter pass on the shipped kernel at 32k (which stall
  class dominates), and a **pre-registered abort line — a prototype must clear 6.5 TFLOP/s to be worth a second
  week of engineering**, written down before it is built so a number, not fatigue, ends it.

**Honest bottom line.** After L-c we expect to remain ~1.5× behind llama.cpp at 32,768 lanes, with GEMM and GDN
going our way. Closing the last factor is a scope decision about rewriting the attention geometry (~1.1k lines in
this engine's terms), and it is the user's to make, not mine to spend days on. Cold/warm conventions from §50–§51
still apply: every figure here is warm or trace-based, never a cold number against a peer's warm one.

## §53. L-c measured and **DISPLACED**: the split cap takes −4.61 % off attention and only −2.38 % off the 32k wave — the last targeted attention lever with a measured mechanism is now spent (2026-10-19)

`prototype/attn_percost/A9-LC-verdict.md` is the record; `A9-LC-impl.md` the implementation. Window:
`prototype/attn_percost/run_ab_a9lc.sh`, one `gpu_lock` hold (`util` 91 % at acquire, 55 % at release), base pinned
to the pre-change sha `8b0a3b4`, `NULL` arm **−0.75 % family / −0.40 % wall** so the window is paired, deciding-family
rep spreads 1.49 % (base) and 0.15 % (cap). The knob is `--attn-split-cap`, default inert, and its inertness was
witnessed rather than assumed: committed binary with no flag vs `8b0a3b4` at 4,608 lanes gives **byte-identical
`logits4607.bin` and `sampled.txt`** with identical geometry lines (`vgpr=185 lds=20480 wf_per_cu=9`) and
`SPLITCAP split=9 cap=0 span=512`.

| pre-registered line at 32,768 lanes | base | cap 8 | ratio | line | result |
|---|---|---|---|---|---|
| LINE 1 — `attn` family ms | 5,961.62 | 5,686.51 | **0.9539 (−4.61 %)** | ≤ 0.9700 | **cleared** |
| LINE 2 — wave `wall_ms` | 8,607.94 | 8,403.26 | **0.9762 (−2.38 %)** | ≤ 0.9700 | **missed** |

**DISPLACED: not shipped.** The verdict is not a rep artifact — per-rep paired wall is **−2.90 %** and **−1.85 %**,
both individually short of the −3.0 % line — and the cap genuinely bound (the capped arm printed
`span 576 > MAXSC 512` at 4,608 lanes, so it was not the same geometry timed twice). A9 §4.2's claim of
**−4.2…−4.7 % of the 32k wave** is retired; measured −2.38 %.

Three things this buys that are worth more than the millisecond count:

1. **The mechanism survives; the conversion factor does not.** Removing 77 % of a 105 GB/pass partial round-trip
   moved the phase it lives in by −4.61 %, so §52's reason for funding L-c — `attn_combine` traced at 208 GB/s =
   88 % of the roof, a *measured* rate rather than a modelled demand — is **not** falsified. But even a traced rate
   times traced bytes overstates the payoff by **~1.35×**. That is a calibration point for every future
   traffic-shaped estimate in this ledger, sitting alongside D21's much harsher finding (demanded bytes are not DRAM
   bytes at all).
2. **An externality line printed beside two lines is what kept this honest in both directions.** The ~70 ms between
   the two lines was blamed on `gdn_rec` (758.52 → 837.98 ms), but per-rep `gdn_rec` spans **730–886 ms** across
   the four runs — 7.7 % and 12.1 % rep-to-rep, intervals overlapping. So the gap is recorded as **unattributed**,
   not "GDN got slower": a family whose own spread is ±10 % cannot adjudicate a 70 ms claim from a median of two.
   Had only LINE 1 existed, this same window would have been reported as a −4.61 % win.
3. **The lever was inert where the headline is read.** `split_for(4,096) = 8`, so cap 8 changes nothing at or below
   4,608 lanes; even a SHIP would not have moved the §51 warm headline, which stands unchanged at **9,790 prompt
   tok/s at 4,096 lanes** (6,125 at 16,384; 3,924 at 32,768). The remaining long-context distance to llama.cpp is
   still **1.91× on the attention phase at 32,768**.

**Where that leaves the plan.** §52 funded L-c as the one targeted lever whose mechanism was measured rather than
theorised, and it is now spent: it produced a real phase-level win that does not reach the wave. Combined with A3
(LDS floor), A4-L1 (accumulator dependency), G0/G1 (residency, which shipped and helped), L-b (bandwidth, killed on
declared LDS) and now L-c, every axis except the round's *work per operand byte* has been falsified from its own
direction. That is precisely the residual A9 §5 named — **9.6 vs 31 MAC per LSU byte** — and the only route to it is
the geometry rewrite, which stays **pre-gated and unfunded**: an `SQ_WAIT*` stall census on the shipped kernel at
32k first, and a prototype that must clear **6.5 TFLOP/s** to earn the build. It is a scope decision for the user
(§51/§52), not a queue item.

## §54. A11: the big cache tier is measured (~32 MiB at 877 GB/s — the ">240 GB/s means skipped work" rule needed an exception), the stall census neither confirms nor falsifies, and **the geometry rewrite becomes the main line** (2026-10-19)

Two device legs, existing tools only, one `gpu_lock` hold each (`util` 91 % / 88 % at acquire), foreground and
blocking, zero subagents. The user's instruction with it: *"don't invest heavily in new
implementations/prototypes — changing geometry to llama.cpp is our best bet."* So this census reused
`tools/build/bw_probe_hip` and `rocprofv3` rather than building anything, and its output is a **direction**, not a
new instrument line.

**Leg A closes open item 2 of `devices/strix-halo-official-specs.md`** — the question "is the ~54 MiB tier MALL?"
was recorded as *Untested* and is now answered. Warm read bandwidth falls off a **cliff between 32 and 64 MiB**:
467 → 640 → 784 → **876.7 GB/s at 32 MiB** → 239.4 at 64 MiB → 237.2 at 1 GiB (full table in
`devices/strix-halo-memory-bandwidth.md`, "The size–bandwidth curve, measured"). Three consequences:

1. The big tier exists, the GPU reaches it, and its effective budget is **~32 MiB** — matching the Chips and Cheese
   32 MB MALL figure and our own earlier "~54 MiB at ~876 GB/s" inference in *rate* while tightening its *size*.
2. **The standing trap "a bandwidth number above ~240 GB/s is evidence of skipped work" is only valid for
   DRAM-resident working sets.** A warm ≤32 MiB read stream legitimately reaches 876.7 GB/s, 3.7× the DRAM ceiling.
   The rule is amended in place at the source doc rather than left to be re-broken.
3. **The copy column is not the read column** (887 GB/s copy at 16 MiB but 325.7 at 32 MiB while read holds 876.7),
   so any "keep it in the big tier" idea must be judged on copy, not read, when the buffer is written as well as
   read — which the attention split partials are, and which is consistent with §53's finding that shrinking their
   traffic by 77 % bought only −4.61 % of the family.

**Legs B/C, the counter census, and its honest verdict.** One `rocprofv3 --pmc` pass at 32,768 lanes, filtered by
kernel name (`attn_fattn_wave_u8` 378 dispatches — prefill only, decode kernels excluded by name):

| kernel | TCP_REQ | L1 miss % | note |
|---|---|---|---|
| `attn_fattn_wave_u8` | 3.17e10 | **26.8 %** | L1 absorbs ~73 % of the shipping attention kernel's traffic |
| `attn_combine` | 4.38e8 | **73.7 %** | the DRAM one, exactly as A9 §2.3 and §53 said |
| `attn_prep` | 5.18e7 | 68.2 % | |
| `attn_slim_wave` | 1.27e7 | 53.6 % | |

Stall composition for the shipping kernel, as shares of `SQ_WAIT_ANY`: **`INST_ANY` 16.3 % (of which
`INST_LDS` 13.9 %), `BARRIER` 11.2 %, `IFETCH` 0.4 %, and **`other` 72.1 %** — i.e. the overwhelming majority of
waiting is *not* LDS-instruction availability, *not* barrier, *not* instruction fetch, which is consistent with
memory-return/dependency waiting, the class operand reuse reduces. **But that is as far as it goes, and two things
must be said plainly:**

* **Two counters read exactly 0.0 across a whole run** — `DRAM_REQ_PER_CLIGRP` and `TCP_BACK_PRESSURE`. Those are
  **vacuous, not clean**: a counter that never fires cannot say "no back-pressure" any more than an empty list can
  say "no violations". Anyone using `rocprofv3` on this part must first witness a counter reading non-zero on a run
  where it must fire. Recorded as a trap because it fails in the direction of a plausible negative result.
* **Counter semantics on this part are unverified** — the same ledger already records `SQ_WAVE_CYCLES/SQ_BUSY_CYCLES`
  reading 41 against a 32-wave-per-CU architectural ceiling. So the 72 % "other" share cannot be *claimed* as
  proof of the operand-reuse thesis; it fails to falsify it and fails to confirm it.

Therefore **pre-gate 1 of §52/A10 resolves to "not falsified"**, and the decisive gate becomes the timing one: a
deep-stage prototype must clear **6.5 TFLOP/s** at 32,768 lanes (vs today's 4.47), which is a measurement rather
than an interpretation. One derived number worth keeping: the fattn kernel's 26.8 % miss rate on 3.17e10 requests
is ~8.5e9 missed lines ≈ **1.09 TB over the ~8.3 s prefill ≈ 131 GB/s**, *below* the 237 GB/s DRAM ceiling — so this
pass cannot even say whether the big tier absorbs the misses. And it refines a claim in `A9-LONGCTX-CENSUS.md` §3
that §A10 repeated: Q's per-block working set is 24,576 B, but at **3 blocks/CU that is 73,728 B, above the 32 KB
L1**, so the Q re-read is *not* wholly an L1 phenomenon — the deeper tiers are load-bearing for G1's trade, which
is a hidden dependency worth remembering before any change that streams more K/V through them.

**Direction, recorded as the user's decision.** The targeted-lever space is exhausted (§53) and the census does not
point away from the geometry explanation, so **the llama-shaped geometry becomes the funded main line** — starting
at R1 of `prototype/attn_percost/A10-REWRITE-BRIEF.md` (256 threads / 8 warps / 16 query rows per wave = 128 rows/block, per A10 §6's R1 correction), because R2
(Q in registers) is impossible before it. Instruments are reused, not rebuilt; no further side prototypes. The
abort line stays: **below 6.5 TFLOP/s at 32k, the geometry route is dead and gets recorded as dead**, and short
context (warm 10,637 tok/s at 512 lanes, 9,790 at 4,096) is a non-regression line, not collateral.

## §55. The pad A/B settles which resource binds `attn_fattn_wave_u8`: **LDS, at 3 blocks/CU — so 248 VGPR does not bind, and a register saving cannot buy residency** (2026-10-20)

§54's rewrite decision needed one thing no counter could give: whether the shipping kernel is resident at 3
blocks/CU (LDS-bound, as the driver's occupancy query said and §53 recorded) or at 2 (register-bound, because
a dispatch reserves 248 VGPR/thread and 3 x 96 x 248 = 71,424 looked impossible against the 65,536/CU this
ledger has been using). A11's attribution (`prototype/attn_percost/A11-attribution.md` §6) named the cheap
measurement; it is now run, and it answers both halves.

The instrument is a compile-time pad, `FATTN_LDS_PAD_HALVES` in `engine/attn_fattn_kernels.h` (0 in every
shipping build, so the default is untouched): 6,144 extra halves appended to the aliased K/V stage take
declared LDS 20,480 → 32,768 while the kernel's work stays byte-identical, because no index the kernel can
produce reaches the pad. Both arms compile to the same `next_free_vgpr 241` → 248 allocated, so the single
variable is the LDS bound: 3 blocks/CU allowed versus 2. First attempt at the pad used a second `__shared__`
array with a store under a condition the compiler could not prove false — the allocation was eliminated and
both arms read 20,480 B, which is how a null nearly got reported as a finding.

One `gpu_lock` hold (TAG=a11pad), blocking, zero subagents, arms interleaved per rep, `tools/prefill_sweep.py`
with `--runs 1` per invocation and best-of-3 taken across reps; full table in
`prototype/attn_percost/evidence/a11_pad_ab.txt`:

| lanes | BASE best | PAD best | PAD cost |
|---|---|---|---|
| 3,072 | 387.3 ms (7,931.7 tok/s) | 400.3 ms (7,674.8 tok/s) | **+3.36 %** |
| 16,384 | 2,773.8 ms (5,906.6 tok/s) | 2,896.5 ms (5,612.2 tok/s) | **+4.42 %** |

Every PAD rep is slower than every BASE rep at both widths — the per-rep ranges are disjoint (16,384:
2,773.8–2,802.5 against 2,896.5–2,913.2), so this is a shift, not a spread. Three things follow, and they
point the same way:

1. **The LDS bound is live at 3 blocks/CU**, so the kernel really runs 3 blocks = 9 wavefronts/CU. §53's
   witness stands; the field it was read from stands with it.
2. **The 248 VGPR a dispatch reserves is not the residency binder.** 3 blocks at 248/thread needs 71,424
   registers, so the per-CU vector register file on this part is *not* 65,536: it is at least 71,424, most
   plausibly 65,536 **per SIMD** (the `256 threads × 256 VGPR ⇒ 1 block/CU` arithmetic in
   `prototype/attn_percost/A10-RF-LADDER.md` §2 divided the per-SIMD file by a per-CU thread count). The
   per-thread ceiling of 256 is unaffected, and with it A10 step 2's source kill: the kernel now uses 241 of
   those 256, which is a tighter place to add Q residency than the 185 the note block reported.
3. **A register saving cannot buy residency on this geometry** — LDS binds first. That was `A10-F16-IMPL.md`
   D2's conclusion, but D2 reached it with the wrong baseline (185) and it is now carried by a measurement
   instead: the f16 accumulator's 64 VGPR/lane buys headroom under the per-thread ceiling (241 → ~177, which
   is what makes a partial Q residency imaginable at all) and buys nothing in wavefronts. Conversely **LDS
   reductions are the residency lever**, and G1's +3.01 % for 6 → 9 wf/CU is corroborated from the other
   direction by the 3.4–4.4 % this pad costs for 9 → 6.

Headline unchanged, as this section changed no shipping code: warm **10,637 tok/s at 512 lanes**, **9,790 at
4,096**, and no attention number moved — the pad exists only behind a compile-time macro that no build
defines. What moved is that the rewrite's residency premise is now measured rather than assumed: R1 gives up
an LDS-bound residency it currently has, which is the shape of A11's verdict that R1 costs 7.5–9.1 %.
The same window ran the arm §54's rewrite decision actually turned on, at equal residency: the padded
unstaged8 build (LDS 32,768, Q re-read from global every round) against the shipping build with
`--attn-fattn-qstage stage` (LDS 32,768, Q staged, zero Q global traffic). Same declared LDS, same 248 VGPR
on both arms, so only the Q delivery channel differs. At 16,384 lanes, 3 interleaved reps
(`prototype/attn_percost/run_a11_qstage_ab.py`): Q-from-global best **2,911.6 ms** (2,911.6/2,914.8/2,921.5),
Q-from-LDS best **3,289.9 ms** (3,289.9/3,311.2/3,342.3) — the LDS-fed arm is **12.99 % slower**, ranges
disjoint. A11's BET asked for ≥ +25 % to keep R2 alive; what was measured is negative, against the ≤ 12 % its
three independent bounds predicted. So the Q channel is not a small win the attribution overweighted: at the
resource that binds, feeding Q from LDS is a regression, which is also why `unstaged8` shipped as the default
and why G1's +3.01 % must be read as an LDS/eligibility effect rather than as Q-traffic savings. Consequence
for A10's queue: **R2 is measured dead, R3's 5.2 % and R1's residency give-up stand, and the only lever this
window corroborates is reducing LDS per block, since that moves residency in both directions it was tested.**
Headline numbers are unchanged by all of it — warm 10,637 tok/s at 512 and 9,790 at 4,096 — because neither
arm is the shipping build.

## §56. Why A12's 25 % never reaches the wave: the wide launch gets 32 workgroups on a 40-CU machine, and the ~2 % it does deliver is the split round-trip (2026-10-20)

The tip before this, `6b857d2`, reported wide=1 at −0.69 % against the rig's 1.2507× and asked the right
question: *"the question is coverage, not speed: which rows at 16,384 lanes does the wide kernel actually
own"*. Four legs in four `gpu_lock` holds (TAGs `a12-why-census` / `-trace` / `-wall` / `-cap`, blocking,
zero subagents, `util` 89–92 % at acquire and discounted per §6) answer it, and the answer is neither the
coverage hypothesis nor a diluted rate win. Evidence, commands and tables:
`prototype/attn_wide/A12-WHY-NOT-FASTER.md` and `prototype/attn_wide/evidence/`.

1. **Coverage is total, so the hypothesis is refuted.** The engine's own `ATTN_WIDE` echo reads
   `wide_rows=512/512 grid=(2,1,16)` at **all 32** distinct geometries of a 16,384-lane prefill — the wide
   block owns every row of every body, exactly as its header says it must. What does not change with
   context is the **grid**: `2 × ceil(wave_rows/32)` workgroups, fixed at 32 by `kMaxWaveRows` in
   `engine/decode_step.h`, while the shipping launch it replaces grows on its `split` axis from 176
   workgroups on average to **2,816** at the last body.
2. **The rate win is absent, not diluted.** Paired per-body from rocprofv3 dispatch timestamps
   (`tools/prefill_kernel_trace.py`, one prefill per arm): the wide core is **1,476.05 ms** against the
   shipping core's **1,473.15 ms** — **+0.20 %** — and the ratio per body stays inside 0.937–1.122 with no
   trend, so the widest body (the shape the BET priced) reads **1.0210**, not 1.25. The attention
   *family* is −5.1 %, but every ms of that is `attn_combine` (71.58 ms) and the inert `attn_slim_wave`
   (13.42 ms), which the wide route does not launch.
3. **The mechanism is residency the body cannot fund, measured in the engine rather than argued from the
   rig.** The rig ran the wide kernel at 2 blocks/CU × 40 CUs = **16 wavefronts/CU** in one 2,048-workgroup
   launch; the engine hands the same kernel **32 workgroups**, so it hosts at most 1 block/CU on 32 of 40
   CUs — **8 wavefronts/CU with 8 CUs idle, below the 9 the form it replaces sustains**. Sweeping
   `--wave-rows` at fixed total lanes (same work, only the launch population moves) turns that into a
   curve: ratio wide/shipping **0.9785** at 32 wgs, **1.0867** at 16, **1.5582** at 8, **2.0080** at 4.
   The shipping arm does not track its own block count; the wide arm tracks nothing else.
4. **Correction to the row before this one.** With 5 interleaved reps at 16,384 the wide arm is
   **−2.03 % best / −1.76 % median** (2,753/2,778 against 2,810/2,829, ranges **disjoint**: wide0 min
   2,810 > wide1 max 2,789) and **−1.96 %** at 4,096. So the wide route is a real ~2 %, roughly 3× what
   `6b857d2` recorded — and still ~12× short of the rig's 25 %, which is a fact about a grid depth the
   engine never issues. Same class as the `gemm_algo_sweep` row: *price the geometry the engine emits.*
5. **And ~78 % of that 2 % is not the shape's at all.** Since the whole engine win is the skipped partial
   round-trip, `--attn-split-cap 8` on the **shipping** kernel reproduces **−1.14 %** of the wide arm's
   **−1.46 %** at 16,384 (best of 3, canary `SPLITCAP` line per run; cap 4 is +1.60 % and cap 2 +5.31 %, so
   the cap over-binds below 8). A knob that moves no format, no band and no route buys most of what the wide
   form buys — while the wide form still owes a `smoke` cosine (`A12-ADMISSION.md`) and moves every row at
   or below `MAXSC` off the bit-exact one-pass route.

**State of the bet.** A12's default stays `--attn-fattn-wide 0` and its BET verdict stands as recorded
(NOT CLEARED at 6.5 TFLOP/s). What this window adds is the requirement the shape would have to be met:
80 blocks in flight, which in the engine means either `kMaxWaveRows` ≥ 1,280 or a lane split in the wide
form — the one thing `engine/attn_wide_kernels.h` states it does not have (`split_used=0` on its own echo).
Neither is funded here. **Headlines unchanged: warm 10,637 tok/s at 512 lanes, 9,790 at 4,096**, because
nothing shipped; the honest long-context statement is now *attention is 56 % of a 16,384-lane wave and the
wide shape cannot move it, but the split's partial traffic is worth ~1.1 % and is already a knob.*

## §57. `kMaxWaveRows` is at its optimum at 512, and the residency A12 asked for does not pay — both closed in one session (2026-10-20)

§56 left a precondition standing: the wide shape needs ~80 workgroups in flight, which a 512-row body
cannot emit, and §56's LEG B showed a convex body-cost curve that made a *bigger* body look cheap. Two
legs priced it (`prototype/wave_rows/README.md`, `TAG=r1-wavewidth` and `TAG=r1b-wide-residency`, one
lock each, blocking, zero subagents, canaries asserted per run): three builds of one source line
(`kMaxWaveRows` ∈ 512/1024/2048) with a fourth arm — the 2048 build run at `--wave-rows 512` — that
isolates the constant from the plan.

**R1: the shipping body is right where it should be.** Best of 3, `RESULT ms_total`: at 16,384 lanes
1,024 rows is **+0.54 %** (ranges overlap the shipping ranges), 2,048 rows is **+2.79 %** with every rep
slower than every shipping rep (min 2,840.3 against max 2,808.5 ms); at 4,096 lanes 2,048 rows is
**+3.76 %**, disjoint. The curve is asymmetric, and that is the answer §56 could only extrapolate:
halving the body costs **+10.8 %**, doubling costs **+2.8 %**. The `w2048c` arm is recorded as a hint and
labelled as one — same plan, same launches, only 4× the `part` scratch (258 MiB → 1.03 GiB per scratch
instance, the bytes stated in `engine/kernels_attn.cpp`) read **+1.98 %** at 4,096 and **+0.90 %** at
16,384 on 3 reps, overlapping at 16k. So `kAttnMaxSplit`'s "the ONLY thing the raise costs is the scratch
named below" has an unmeasured time side; price the build, not just the bytes.

**R1b: the residency premise is falsified, and with it A12.** A 2,048-row body gives the wide launch
`2 × ceil(2048/32) = 128` workgroups — asserted from the run's own `ATTN_WIDE grid=(2,1,64)` — which is
3.2 blocks/CU asked against an LDS bound of 2, i.e. the rig's exact **2 blocks/CU = 16 wavefronts/CU**,
inside the engine. The wide kernel with that population is **+2.45 % slower at 16,384 and +1.01 % slower
at 32,000 than the same kernel at 32 workgroups**, and no better than the shipping form at either width
(16,384: wide@128 wgs 2,804.8 ms against wide@32 wgs 2,737.8 against shipping 2,762.9; 32,000:
7,987.5 / 7,907.5 / 8,055.9). So the rig's +25.1 % was neither coverage nor block population: it is a
property of the rig's own workload — one kernel, no `attn_prep`, no combine, no §05.6 epilogue, one
uniform pass over the triangle — and it does not transfer in any launch shape the engine can emit.
**A12 is closed on all three axes**, each by a measurement: coverage (§56), rate at engine geometry (§56),
rate at the rig's residency (this section).

Also confirmed as a side effect, and still unshipped: `--attn-fattn-wide` at the default body is
**−0.91 % at 16,384** (just-disjoint ranges) and **−1.84 % at 32,000** (disjoint), which is the launch
elimination of §56 and ~78 % of which `--attn-split-cap 8` already buys without a numerics risk.
Neither leg ran numerics — a wider body is not bit-identical and the wide form still owes `smoke` — so
nothing here is admitted, and **headlines stay warm 10,637 tok/s at 512 and 9,790 at 4,096**. What these
two legs buy for the queue is a clean field: the geometry axis is empty, so the only remaining account of
llama's 1.9× is the per-wavefront one (`attn_fattn_wave` issues 17 `v_wmma` sites where their kernel
issues 2,592, `A2-census` §2), which is R2's job to price before anyone funds R3.

## §58. R2: the gap is amortization, not ILP — 108 / 111 / 52 non-matrix issue slots per matrix slot, and the ratio closes the gap arithmetically (CPU-only, 2026-10-20)

`prototype/attn_percost/R2-amortization-spec.md`, `evidence/r2_round_bills.txt`. Same tool as A2 §2
(`g3_isa_bill.py` finds each kernel's widest backward branch, `a2_round_bill.py` histograms it), run now on
all three kernels from carved code objects — ours out of `engine/build/decode_run`, theirs out of the llama
build. No device, no timing.

**First, the ILP framing is dead on arrival and was closed by numbers already on file**: §26's probe
sustains 32.4 clk per WMMA per SIMD with **one** independent accumulator chain (33.9 at four), so a single
chain already fills the matrix pipe's 32-cycle acceptance interval, and `A3-wmma-accumulator-form.md` found
four chains plus four dependent LDS loads *degrade* 8.5× on one load port. Accumulator ILP is not the
missing quantity, so the only ISA question left is **how many non-matrix slots a wavefront spends per matrix
slot**: shipping `attn_fattn_wave_u8` **108.3** (17 of 1,858), wide `attn_wide_wave` **110.9** (128 of
14,325), llama `flash_attn_ext_f16<256,256,16,4>` **51.9** (256 of 13,537). Their head-dim instance must be
selected explicitly — the carve holds eight instantiations and the bare-symbol match `<64,64,…>` bills at
72.4, which would have reported their advantage as 1.5× instead of 2.09×.

**The arithmetic closes.** llama's amortization is **2.09× better**; their measured rate on this family is
**1.9× ours** (A2 §3). Nothing else is needed to account for the gap. And the number explains the thing this
session could not otherwise explain: **A12 moved the ratio the wrong way** — 7.5× the matrix work per round,
8.2× the non-matrix bill, 108 → 111 — so there was never an amortization win for the engine to inherit from
the rig's +25.1 %, which is consistent with §56 (coverage), §57 (residency) and the paired per-body ratio of
1.0001.

Per matrix slot, the buckets that differ: `salu` 15.8/21.8 against their 3.7, `move` **19.8** (shipping's
`v_mov_b32`×241; the wide kernel already fixed this at 2.9), `delay` 10.6/18.9 against 4.5, `select_mask`
7.5/9.6 against 2.6, `fmul` 8.5 (the 128-slot accumulator rescale) against 0.5. **Already at their level:**
`int_vec` address arithmetic (19.1 vs 18.1) and `s_waitcnt` (4.2 vs 4.2) — so neither memory-wait behaviour
nor index arithmetic is the difference, a guess this census kills before anyone funds it.

**The spec R3 must satisfy: ≤ 52 non-matrix slots per matrix slot**, reachable only two ways. (a) Shrink the
non-matrix bill at fixed matrix work: −52 % at 17 matrix slots, more than `move`+`salu`+`delay` combined —
not credible, and it retroactively explains why A4-L1's −41 % cut of the rescale measured **−1.67 %, dead**
(7 % of a quantity that had to fall 52 %). (b) Widen the matrix work per round **while holding the non-matrix
bill flat** — A12's direction with its violated constraint made explicit, and it names the work list: at 64
matrix slots and today's shipping bill the ratio would be **29, better than llama**; to get there from A12's
wide round, `int_vec` 3,836 → ~2,300, `salu` 2,789 → ~600, `delay` 2,420 → ~600.

Falsifiable, and stated as a gate on future work: any candidate kernel billing > ~70 non-matrix slots per
matrix slot **cannot** reach llama's rate, and this census says so before it is written. Not established:
causation. Both kernels are stall-bound in wall-clock terms (A2: ≤ 8.8 % issue slots, 5.6 % matrix pipe; A11:
65–74 % "waiting"), llama's stall profile was never instrumented, so the ratio is a serialised-work proxy
whose mechanism is inferred — the residency alternative that would have explained the gap differently is
what §57 R1b killed. **Headlines stay warm 10,637 tok/s at 512 and 9,790 at 4,096** — R2 ships no code and
moves no default.

## §59. **A13 lands: attention is at llama's rate.** Warm −10.4 % at 512, −14.9 % at 4,096, −27.4 % at 16,384 and −35.7 % at 32,000 lanes; gate 18/18 with no band moved (2026-10-20)

`prototype/attn_wide/A13-RESULT.md` (instruments `a13_byte_identity.py`, `a13_longctx_identity.sh`,
`a13_identity_metrics.py`, `a13_ab_timing.sh`, `a13_warm_headline.sh`; evidence `evidence/a13_*`). Form:
`--attn-fattn-wide 2`, `engine/attn_wide_llama_kernels.h`, **default still 0**. Every number below is
main-agent, blocking, zero subagents, one lock per leg, canaries (`attn_fattn_wide=`, `form=`, `tokens=`,
and the required-to-move `scratch=` field) asserted on every run.

**Warm headline (§51 convention, `WAVEPASS gran=burst wall_ms`, median of 3).** knob 0 → knob 2:
512 48→43 ms (**10,667 → 11,907 tok/s**); 1,024 94→87 (**10,894 → 11,770**); 4,096 422→359
(**9,706 → 11,409**); 16,384 2,674→1,942 (**6,127 → 8,437**); 32,000 7,960→5,116 (**4,020 → 6,255**). The
knob-0 column reproduces §51's published figures within 0.3 % / 0.9 % / 0.03 % at 512 / 4,096 / 16,384, so
the baseline is the same instrument. Cold moved alike (16,384: 2,782→2,032 ms, `[2782,2855]` vs
`[2032,2094]`, disjoint; 32,000: 7,957→5,112, disjoint), while knob 1 (A12) stayed at its −1.0…−1.6 %.

**Against §51's attribution, which is the whole point of this section.** §51 recorded our attention excess
as **+751 ms/pass at 16k** against a total gap of +596 ms, and **+2,815 ms at 32,768** against +2,489 ms.
A13 removes **−732 ms at 16,384** and **−2,844 ms at 32,000** warm — 97.5 % and ~100 % of that named excess
(widths differ slightly at 32k, so read the second as ±5 %). Against llama's own traced totals this now
reads **8.7 % faster at 16k** (1,942 vs their 2,126.23 ms) and **12.4 % faster at 32k** (5,116 vs their
5,839.50 taken at 32,768 lanes, so if anything understated) — cross-window against §51's llama figures, not
a paired A/B, and labelled as such per §57. GEMM was already 0.88–0.90× in our favour, so **the attention
question is now closed and any remaining long-context difference is not attention.**

**§58's model predicted the size from the other side.** The ported round bills **20.56** non-matrix slots
per matrix slot against A12's **24.05** (round-scoped; the mandated widest-span instrument reads 110.91 →
67.09 but encloses both round instantiations plus the epilogue, so the round-scoped figure is the quotable
one). The worker's flagged price — spilled VGPRs 163→920, private segment 176→720 B/thread — is paid for
many times over at every width, and `scratch=` is exactly the behavioural canary §57's D30 rule asks for.

**Correctness, and the wrinkle that mattered.** Byte-identical to knob 1 across the five pinned geometries
(1,960 files, 0 differing) **and** bit-identical at 4,096 lanes including all 4,099 free-run decisions. At
16k/32k it is **not** bit-identical: **1−cos 1.41e-05 and 6.43e-05**, argmax SAME, top-5 SAME, and the
**entire 16,385 / 31,826-token greedy continuation agrees token for token**; all 32 wave lines carry
identical `grid`/`wide_rows`/`lanes_last`/`split`, so coverage is unchanged and this is re-association, not
skipped work. Gate over the derived 18-fixture population: **knob 2 18/18 PASS, knob 1 18/18, knob 0
(ships) 17/18** — the red is `chunk64b`, pre-existing per `A12-ADMISSION.md`. **No band was widened**; the
user's pre-authorisation was not spent. The only gate edit is knob admission itself: `choices` gains `"2"`,
the echo map becomes an explicit dict `{"0":fattn,"1":wide,"2":wide_llama}` so a fourth form cannot fall
into an `else` and be judged as the shipper, and `_is_wide_side` treats `wide_llama` as wide, routing it
onto the already-adopted wide-scoped pooled line (measured 2.028e-04 against the 2.500e-04 line, 19 % inside).

Two instruments died and were replaced in the making, both worth the rows in `AGENTS.md`: `--ids-file`
splits on **commas**, so a space-joined fixture read as one token and a whole window of "16k" numbers came
out as 23 ms until the `form=` canary aborted it; and a logits dump read as BF16 when it is F32 yields NaN
for every aggregate while still printing a confident `ndiff`. **Headlines stay as §51 published until the
default decision is made** — knob 2 is admitted and one line from becoming the shipper, and the precedent
when A12 passed at ~1 % was that this call is the user's. What changed is the size of what is on the table:
27 % of the 16k wave, not 1 %.

## §60. The default flips to the A13 form: **18/18 on the default admission, `chunk64b` among them**, and the published warm headline moves to 11,070 tok/s at 4,096 and 8,542 at 16,384 (2026-10-20)

`prototype/attn_wide/A13-RESULT.md` §"State and what is left"; instruments
`a13_default_admission.sh` and `a13_default_warm_headline.sh`, both of which pass **no form flags** — that is
the entire point of them, since after a default flip an explicit-flag run proves nothing about what ships.
`decode_run.cpp`'s option struct and `gate.py`'s `--attn-fattn-wide` default moved in **one commit**, because
the half-flipped pair is the failure this file already records (the gate silently judging a kernel the engine
does not ship); `AttnLayerArgs`' struct default deliberately stays 0, which is not a missed half but the
library default that only reaches callers leaving the field unset — the standalone rigs, whose baselines were
all taken on the inert form. `decode_run` assigns the field unconditionally, and the flip is asserted per run
by each run's own `attn_fattn_wide=wide_llama` / `form=wide128llama` echo, not inferred from a flag.

**Default admission: 18 PASS, 0 FAIL, 0 NO-VERDICT over the derived 18-fixture population.** Among them is
`chunk64b`, the red the shipping path has carried since before A12 — a batched-vs-one-row token flip with
2.925 BF16 ULP of room against a 1.50 flip line, witnessed in `A12-ADMISSION.md` against a pre-change binary
as pre-existing. **The flip did not inherit that red, it removed it**: the A13 form passes the same check.
That is reported as a fact of this admission, not as an explanation, and the old red stays on the record as
what knob 0 does.

**Warm headline as shipped (§51 convention, medians of 3, one process per width, no flags):** 512 →
**11,636 tok/s** (`[44.31, 44.61]` ms); 1,024 → **12,047** (`[85.09, 86.16]`); 4,096 → **11,070**
(`[361.8, 373.5]`); 16,384 → **8,542** (`[1894.3, 1922.7]`); 32,000 → **6,154** (`[5158, 5295]`). Against
§51's published figures that is **+9.4 % at 512, +6.8 % at 1,024, +13.1 % at 4,096, +39.5 % at 16,384**, and
~+57 % at 32k against its 3,924 (published at 32,768 lanes). §51's table stands as its own record and is not
edited; **this section is the headline from here.** §59's cross-window note still applies to any comparison
against llama's traced totals — ours is now the faster of the two at both 16k and 32k on those numbers.

What the flip does *not* settle: the `smoke` wide-scoped margin is unchanged (measured 2.028e-04 against the
2.500e-04 line, so the wide forms have ~19 % headroom and the next wide-form change will find it tight), and
§51's phase-attribution table now records an attention row (1,518.91 ms/pass at 16k) that the engine no
longer produces — re-tracing it is a separate piece of work, not a bookkeeping fix.

---

**Frozen 2026-10-21: this is the ledger exactly as it stood at the moment of the 2026-10-21 prune.**
The frozen part is the **2,828** lines above this marker, taken at HEAD `2bdea89`, byte-identical to
the file in `bench/` it was pruned from — a whole-file copy rather than an excerpt precisely so
that every `path:NNN` line cite and every `§label` in `progress/` still resolves at the same
coordinates. It contains §1–§6 as they stood on 2026-10-16 plus §7 through §60, the append-only
record of the sixteen days of rounds that followed (the GEMM flip, the F32-seam and cast deletions,
the conv coalesced read, the looped GDN launch, the WMMA attention rounds, G1/G2/G4, the residency
and per-cost attributions, A12/A13 and the default flip). Two things are deliberately **not**
edited here: the numbers that later rounds superseded, and §1's banner, which §60 overtook and did
not rewrite. The live ledger is now `engine/performance-status.md` and carries only current state.
What decided this: `progress/2026-10-21/ledger-prune-2026-10-21.md`.
