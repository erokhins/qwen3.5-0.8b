# Performance status — the live ledger (current state, pruned 2026-10-21)

**What this file is.** The current measured state of the engine on this box (Framework Desktop /
Strix Halo `gfx1151`, `Qwen/Qwen3.5-0.8B`, text-only path). It carries numbers, not chronology: one
table per question, every number naming its command, its window and its spread, and every number
that a later round overtook is named as overtaken rather than quietly kept.

**It moved here on 2026-10-21** — out of `bench/`, where it had lived since the project began — and it
was cut back to current state at the same time. The append-only record of how those numbers arrived — every
superseded table, every retired ruler, every route that died, every correction made in place — is
frozen byte-for-byte in two snapshots:

| file | what it holds |
|---|---|
| `progress/2026-10-21/performance-status-2026-10-21.md` | the ledger as it stood at HEAD `2bdea89`: 2,828 lines, §1–§6 of the 2026-10-16 era plus §7–§60, the sixteen days of rounds after that prune (the GEMM algo flip, the F32-seam and packed-cast deletions, the conv coalesced read, the looped GDN launch, the WMMA attention rounds, G1/G2/G4, the residency and per-cost attributions, A12/A13 and the default flip) |
| `progress/2026-10-16/performance-status-2026-10-16.md` | everything before that prune (the frozen 1,149 lines, HEAD `bb7246d`) |
| `progress/2026-10-14/speedup-routes-2026-10-14.md` | the index of every speed-up route tried, its verdict, the line it had to clear and the number that decided it |

**If the question is "what have we already tried, and why did it not work?" — do not read this file
for that.** Read the routes index in the row above. This file is the measurement record those
routes' citations point at.

**How to read a `§` in this file.** §1–§6 are sections of *this* file. Any number §7 or higher is a
section of the frozen snapshot, `progress/2026-10-21/performance-status-2026-10-21.md`; where a
number below is quoted from one, that path is written out in the sentence or its footnote, and a bare
`§N` elsewhere means the same file (`AGENTS.md`'s citation rule says a `§label` must name its file, and
this file is the one place the alias is load-bearing enough to be declared instead of repeated).

**The section numbering here is a compatibility requirement, not a style choice.** §6 is and stays
the section that answers "what is a number on this box worth", and live code, `engine/gate.py` and
`AGENTS.md` cite it by that number. §1–§5 keep their subjects for the same reason; what changed is
that the tables under them are current or say which later section overtook them.

**Shipped defaults, read out of the option struct rather than off a banner** (`engine/decode_run.cpp`
`Options`, `engine/decode_step.h`, `engine/kernels_proj.cpp`): `--gemm lt` with `--gemm-pins all`,
`--attn-tile 1`, `--attn-touch 1`, `--attn-fattn 1` with `--attn-fattn-qstage unstaged8` and
`--attn-fattn-wide 2` (A13, the wide llama-ported round), `--attn-split-cap` off, `--gdn-scan seq`,
`--gdn-spell fused`, `--gdn-geom reg16` (G4), `--gdn-loop 512`, `--gdn-norm inline`, `--mlp-map auto`, `--proj-plan packed --proj-shape unroll8` (D1; pre-D1 is `split`+`shipped`), `kRowsPerBlock = 32`, `wave_rows = kMaxWaveRows = 512`, lane ceiling
`MAXSC × kAttnMaxSplit = 512 × 64 = 32,768` (§39 of the frozen ledger).

**The headline convention is warm** (§51): a cold process pays hipBLASLt's first-execution cost and
a per-process weight restore, and warm-vs-warm is the comparison that means something for a serving
engine (§50). Cold numbers are still taken and still labelled.

## 1. Headline: what the engine does today

**Prefill, warm, as shipped** (`prototype/attn_wide/a13_default_warm_headline.sh`, which passes **no
form flags** on purpose: after a default flip an explicit-flag run proves nothing about what ships;
`WAVEPASS gran=burst wall_ms`, median of 3, one process per width, one `tools/gpu_lock.sh` hold,
zero subagents). Taken 2026-10-20 on the tree that carries the A13 default, with the form echoed as
`attn_fattn_wide=wide_llama` / `form=wide128llama` by every run.

| prompt lanes | wall ms | warm prompt tok/s |
|---|---|---|
| 512 | 44.31 / 44.61 | **11,636** |
| 1,024 | 85.09 / 86.16 | **12,047** |
| 4,096 | 361.8 / 373.5 | **11,070** |
| 16,384 | 1,894.3 / 1,922.7 | **8,542** |
| 32,000 | 5,158 / 5,295 | **6,154** |

All five rows are `progress/2026-10-21/performance-status-2026-10-21.md` §60, the default flip of
2026-10-20, which says so itself: "§51's table stands as its own record and is not edited; this
section is the headline from here."

Against §51's published figures that is **+9.4 % at 512, +6.8 % at 1,024, +13.1 % at 4,096,
+39.5 % at 16,384** and about **+57 % at 32k**. Against llama.cpp's traced totals on the same model
this engine is now the faster of the two at long context — **8.7 % at 16k, 12.4 % at 32k** — which
is a cross-window comparison, not a paired A/B, and is quoted as such (§57's rule, §59's note).

**Decode — marginal cost of one generated token** (`tools/context_sweep.py --lanes 96,128,512,1024,2048,4096
--steps 32 --runs 3`; differential by construction: two runs differing only in `--steps`, so the prompt wave
cancels). **The headline decode row is the 4k one** (user, 2026-10-21): report decode at 4,032 lanes and quote
shorter widths beside it, not instead — a short-context row hides the KV term, where most of the remaining headroom
sits; at 4k it comes from the stage-clock burst, never `context_sweep.py` (§6). **Rows below are the 2026-10-21
window on the D1 tree** (projection flip, §5; on G4 §44 / G1 §48 / A13 §60); 96 lanes reads 9.619 ms = 104.0 tok/s,
and 512's 11.36 % spread is §6's dispatch bimodality, not a regression. Paired, the step moved −13.0 % at 128 and −11.0 % at 4,032 (`proj` 4.1946 → 2.7284 and 4.177 → 2.745 ms/step, 7/7 reps separating).

| lanes at last step | marginal ms/token, best of 3 | spread | tok/s |
|---|---|---|---|
| **4,032** | **12.680** (stage-clock burst, never `context_sweep.py` — §6; the sweep's own 4,096 row reads 12.556, 2.81 %) | 0.70 % | **78.9** |
| 128 | 9.876 | 0.43 % | 101.3 |
| 512 | 10.609 | 11.36 % | 94.3 |
| 1,024 | 11.028 | 3.98 % | 90.7 |
| 2,048 | 10.929 | 7.20 % | 91.5 |

**Cold start** (`tools/cold_restore.sh --runs 5` → `engine/build/loader_test --no-verify`,
2026-10-16): text weights made resident in a fresh process, device pool empty, host page cache warm —
**best 335.8 ms, median 345.8, worst 366.8, spread 9.2 %**; phases of the best run: cold H2D 58.5 ms,
pinned staging 58.9 ms, first GPU read pass 8.5 ms. Once per process, not per token; at 335.8 ms it is
~30 tokens' worth of generation and it bounds any "restart the engine per request" idea at zero inside
a conversation.

**Numerics state that constrains the next change.** The wide attention forms sit **2.028e-04 against
a 2.500e-04 pooled line** — about 19 % headroom (§59, §60), so the next wide-form change has less
room than it looks like. `chunk64b`, the batched-vs-one-row token flip the shipping path carried from
before A12, is **green on the A13 default** (§60's default admission, 18 PASS / 0 FAIL over the
derived 18-fixture population). Byte-identity claims carry a width: the A13 form is bit-identical to
the frozen form on the five pinned geometries and at 4,096 lanes, and at 16k/32k it differs by
1−cos 1.4e-05 / 6.4e-05 with argmax, top-5 and the whole greedy continuation unchanged.

## 2. Where a decode token goes

**The newest full decode attribution is frozen at §2 (2026-10-16) and §36 (2026-10-19) of
`progress/2026-10-21/performance-status-2026-10-21.md`, and it is superseded**: `gdn_rec` fell 66.6 %
under G4 (§44) and the attention families fell 22.09 % under G1 (§48) and again 14.9 % at 4,096 warm
under A13 (§59). A re-take is owed and it is one command:

```
tools/gpu_lock.sh acquire <tag>              # zero subagents, blocking call, record util at both ends
engine/build/decode_run --time-stages ...    # the exact argv of the last window is recorded at
                                             # progress/bench/evidence/ledger-2026-10-16/commands-and-provenance.md §3
tools/gpu_lock.sh release <tag>
```
and §6 is the rule the window has to satisfy.

What is *method* rather than a number, and survives every re-take:

- **The family gages are exhaustive.** The nine `STAGE*` families sum to the `TOTAL` line to the
  printed precision in every run (`Σ − TOTAL ≤ 0.0001 ms`, at stage and kernel granularity, at both
  128 and 4,032 lanes), so there is no unattributed time in the step wall — what the gage cannot see
  is that families cost more in-situ than in isolation.
- **A burst is not a step.** `lm_head` — one fat launch streaming 485 MiB — moves 2.3 % between
  burst and step; every small family moves 0.46–1.01 ms, because a burst repeats one kernel against
  hot pages while a step interleaves 424 launches across ~1.5 GB of weights. **Quote the in-situ
  column for budget work.** Refuted explanations are on the record: window size (Q7), write-back
  traffic (Q9's predecessor Q8), and the residual that survives is the weight-walk pattern itself
  (Q10's `contig` recovered 0.945 ms of a 2.034 ms gap).
- **The exposed submission gap is the whole launch bill, not a prize.** Serialising the step costs
  ≈ the measured `STAGEGAP exposed_ms` and nothing beyond it (1.98 ms in the 2026-10-16 window), so
  no fusion-for-launch-count, graph capture or wider block buys anything there. Captured graphs
  replay at 1.78 µs against 1.85 µs eager for the same reason.
- **Do not compare a sum of per-family medians against a median wall.** Medians do not add across
  independent runs; the 0.2 % between them is an arithmetic artifact, not a finding.

## 3. Where a prefill wave goes

**Superseded, and named as such.** The attribution this repo has for a prefill wave —
`gdn_rec` 51.2 %, `proj` 15.8 %, `attn` 10.9 %, matmul 28.6 % of a 1,024-lane wave (`progress/2026-10-21/performance-status-2026-10-21.md` §3)
and §51's phase table at 16k — predates G4, G1 and A13. §60 says it in as many words: the phase table
"records an attention row (1,518.91 ms/pass at 16k) that the engine no longer produces", and
re-tracing it is a piece of work, not a bookkeeping fix. The instruments are unchanged and the commands
are recorded: `decode_run --steps 1 --time-waves` at §4 of
`progress/bench/evidence/ledger-2026-10-16/commands-and-provenance.md`.

What is still true as a claim about the shape of the wave, with its number behind it:

- **The serial recurrence, not matmul, is the big share.** Matmul went from 74.4 % of the same wave
  under the hand GEMMs to 21–29 % under hipBLASLt, and `gdn_rec` is over half of what is left. That
  is why the funded campaigns after the GEMM flip were GDN-shaped (§23–§25, G1/G2/G4).
- **Attention is the only family that moves materially with context.** 0.4847 → 2.996 ms/step from
  128 to 4,032 lanes in the frozen window, while `mlp`, `proj` and `lm_head` were flat to under 1 %.
  After A13 the long-context attention gap is closed against llama.cpp (§59), so the remaining
  long-context difference is not attention.
- **`launches_per_step = plan_per_step`, and the instruments agree.** `STAGEPARITY` /
  `WAVEPARITY tokens_identical=1 logits_bitidentical=1`, `plan_match=1`, and the wave's own
  completeness account (`ungaged_ms` 0.009 % of the wave) — the share column divides the whole wave,
  not most of it.

## 4. Per-op floors in isolation

Each row is the burst idiom (warm-up, N launches, one sync, wall/N; `hipEvents` are not usable here —
§6). Percentages are of the **237 GB/s** read convention every historical row in this repo uses; the
probe re-took 2026-10-16 at **239.3 GB/s best read, 210.5 best copy, 82.4 H2D, 69.7 D2H**, i.e. +1 %,
so no column moves (`tools/build/bw_probe_hip 1.0 4`).

**Rows below are the latest published floors for families that have not been rewritten since.** Two
families have been rewritten — the GDN recurrence (G4, `reg16`) and the attention wave (WMMA, G1,
A13) — so their floors are **not** in this table; they are in the snapshot at §33/§44/§48/§59, and a
re-take is owed before any of them is used as a floor.

| op | value | % of probe | notes |
|---|---|---|---|
| `lm_head_gemv` fused | **2.1309 ms** (238.7 GB/s) | **100.7 %** — at ceiling | `--iters 8`, lock held; reproduces window to window to 0.2 % |
| MLP, 528 MB, `contig` (the one-row walk) | **2.275 ms** | 98 % | roofline 2.230 ms; the shipped `--mlp-map auto` choice for one row |
| MLP, 132 MB, `interleave` (the wave walk) | 2.647 ms | 84 % | `--layers all --fixture dec8` |
| projections, whole 48-launch token plan, rows/block 8 at 1,024 threads with a compile-time word walk | **2.4693 ms** (187.3 GB/s, 79.4 % of probe) | 79 % | roofline 1.972 ms; clock-controlled window (per-arm burn, 0 VOID / 0 LOWCLK), and the *shape label* moved with the number — the pre-D1 default was rows/block 32 at 3.2843 ms (60 %) |
| `glue` fused, 49 launches, 512 t | **0.1496 ms** (spread 4.0 %) | launch-bound | fusion still buys ≈0.089 ms/token; unfused 0.2386 ms |
| `gdn_gated_norm` §04.5, 144-job plan | **0.4273 ms** = 2.967 µs/launch (spread 0.8 %) | launch-bound | `DEVICE GATE PASS`, all six negatives localise |
| kernel launch floor (empty/tiny kernel) | **1.85 µs**/launch | — | captured graphs replay at 1.78 µs: no gain |
| CPU read probe | 75–117 GB/s | 32–49 % | why the engine must stay GPU-resident: `devices/strix-halo-memory-bandwidth.md` |
| big cache tier (the exception the 240 GB/s rule needs) | ~32 MiB at **877 GB/s** | — | §54; a *demanded*-bytes rate above 240 GB/s is L1/L2 absorbing reuse, not a broken probe — see §6 |

Three facts that belong with those rows rather than under them: **(1)** the projection families are
**0.4677 GB/token — 29 % of the token's bytes**, and the family is shape-limited, not
traffic-limited (the 16-row `in_proj_a`/`in_proj_b` sit at 5 % of probe for 32 KiB of weight per
launch, which is launch and occupancy); **(2)** do *not* rebuild a plan total out of per-shape
bests — the per-shape sweep's GDN+attention sum was 15 % *worse* than the shipped 48-launch plan,
and that gap is a finding, not noise; **(3)** a small working set can read far above DRAM ceiling
because it is cache-resident (the 54 MiB of GDN state read 372 % of probe warm and 255 GB/s cold),
so a "% of probe" over 100 % needs a cache claim attached before it means anything.

## 5. What the measurements decided

**Shipped defaults, and the number that decided each** (the banner above lists them; this is the
*evidence* column, and the ones that postdate the snapshot are named with their section):

- **`--gemm lt`** (hipBLASLt with admitted algo pins): TTFT at a 1,024-token prompt **1077.4 ms hand
  → 384.7 ms library**, matmul from 74.4 % to 21.2 % of the wave. The weakened guarantee is on the
  record: on this route the wave-boundary check sits inside a band, not byte-identical, and wave-row
  grouping can change a library GEMM's reassociation.
- **`--attn-tile 1`, `--attn-touch 1`, `--attn-fattn 1` + `qstage unstaged8`, `--attn-fattn-wide 2`**:
  the attention chain 107.67 → 26.70 ms at 1,024 lanes for the tiled wave (4.03×, bit-identical),
  then the WMMA rounds at 1.3955× / 1.647× / 1.985× of the `attn` family, G1 at −22.09 %, and A13
  at −14.9 % at 4,096 warm — each admitted on the existing bands with **no band widened**.
- **`--gdn-geom reg16`** (G4): the walk keeps its state shard in registers, `gdn_rec` −66.6 %, and the
  engine went 5,764 → 7,655 prompt tok/s on that one change.
- **`--gdn-loop 512`**: one launch per body instead of 128 per token, 1.58–1.64× bit-identically, and
  in situ `gdn_rec` 449 → 266 ms.
- **`--mlp-map auto`**: `contig` recovers 0.945 ms of the MLP's 2.034 ms in-step gap at 128 lanes, but
  as a whole-default flip it cost +5.1 % TTFT at 1,024 lanes; the per-path split keeps the decode win
  and returns TTFT to +0.8 %, inside the ±2 % band.
- **`kRowsPerBlock = 32`**: −1.3 to −2.7 % decode at ≤2,048 lanes and −5.6 to −8.0 % TTFT,
  bit-identical by construction (one wavefront owns one row end to end).
- **`PACK_X = false`**: unpacked 65.1 % of probe against packed 58.0 % at the 48-launch plan, and it
  wins at every rows/block swept, so the ≈48 cast launches the unpacked form bills back do not
  overturn it.

**Dead routes, with the number that killed each — these bind, so do not re-litigate them without new
evidence.**

- **Route A, the collapsed GDN recurrence: 112.04 ms against a ≤ 63 ms win line and an 84 ms kill
  line.** It bought 6.8 ms on a 322 ms wave (2.1 %). The reason is the finding, not just the number:
  6.1 µs per (token, layer) against `seq`'s 6.8 µs while issuing 36 launches instead of 18,432 — the
  recurrence is **latency-bound on §04.4's 2,048 dependent chains at ~1.6 working wavefronts per CU**,
  so neither reshaping launches nor residency moves it. `prototype/gdn_collapse/results.md`.
- **Captured graphs: 1.78 µs/kernel replay against 1.85 µs eager** — 4 %, because the limit is the
  GPU's own dispatch rate. Fusion is the only lever on launch cost. `tools/launch_overhead_hip.cpp`.
- **`hipEvents` are not a timer here**: `hipEventElapsedTime` returns exactly 0.000 ms for work below
  ~2 ms, including best-of-40 on a kernel that wall-clocks at 2.11 ms. The steady_clock burst is the
  only timer to trust (fixed in `d81ff13`).
- **MTP drafting on CPU: ~21 tok/s with `--mtp 1` against ~24 tok/s plain greedy** (accept 0.660,
  1.24 tok/forward) — bit-lossless and slower. It is a GPU-path bet, not a win.
- **A2's register axis and the L-c split cap are both closed.** §55 settled which resource binds the
  wide kernel — **LDS, at 3 blocks/CU**, so a register saving cannot buy residency — and L-c took
  −4.61 % off attention but only −2.38 % off the 32k wave, which was the last targeted attention
  lever with a measured mechanism behind it. A13 then closed the gap by porting llama's round (§58's slot model
  predicted the size from the other side: 20.56 non-matrix slots per matrix slot against A12's 24.05).

**Open, and what would close each.**

1. **The decode banner is stale** (§1's rows predate G4/G1/A13) and so is the phase attribution
   (§3). Close: one clean window with the commands in §2 and §3.
2. **The wide-form band has ~19 % headroom** (2.028e-04 against 2.500e-04). Close: the next wide-form
   change either widens it with a witnessed reason or is admitted inside it.
3. **`--gemm library` makes a prompt's logits depend on the wave-row cap** (a library GEMM
   reassociates with `M`), so wave-width independence is a band question, not a byte-identity claim.
4. **The mixed one-pass + streaming wave grid is priced by engine-vs-engine self-consistency only**;
   pinning it needs a >512-token-prompt fixture with its own allowance.
5. The 4,032-lane row is quoted from one instrument, with no isolated `attn` burst beside it.

## 6. What a number in this repo is worth

The protocol, then its known pathologies. Both halves are load-bearing: the first says how to take a
number, the second says which numbers cannot be trusted even when taken that way.

- **A clean window = zero subagents active + a blocking foreground call.** This box serves this
  session's own tokens on the same iGPU (`flash_serve`, and every subagent's tokens too), so
  contention is not incidental — it is the machine's normal state. A blocking tool call pauses the
  session's generation, which is what makes the window clean; a backgrounded run keeps the stream
  alive and contends with itself. `AGENTS.md`, "GPU execution is serialized".
- **Sub-3 % on a sub-30 ms row is nothing has moved, in either direction.** Four families a round
  never edited came in at −1.8 %/−2.4 %/−3.1 %/−5 %-of-0.09 ms in a paired window, so that is what
  this instrument cannot resolve (§37). Quote a pair, state the spread, and let the long-context row
  carry the headline.
- **A ruler re-taken in a different window is not an A/B.** The same build read 64 and 256 lanes
  *slower* in a fresh sweep while a paired A/B inside one lock hold read the same two widths *faster*.
  For short prompts the only honest comparison is both sides inside one lock hold.
- **Warm versus cold is part of the number.** Headline rows are warm (§51); a cold run pays
  hipBLASLt's first execution and the weight restore (§50), and a warm-vs-cold difference was already
  mistaken for a per-prompt constant once.
- **`util` is recorded and then discounted.** Identical clean runs read 24–91 %. The gate on validity
  is zero subagents plus a blocking call, never a `util` percentage.
- **State the spread or the number is provisional** — best-of-3 with all runs printed, and a row whose
  spread exceeds its own effect is provisional no matter how clean its window was.
- **A burst is not a step.** §2 says why; quote the in-situ column for budget work.
- **Dispatch-bound rows are bimodal; bandwidth-bound ones repeat.** The three GEMV families reproduce
  within ~2 % window to window; `glue`'s 49-launch path has spread 2.7 % at 256 threads and up to
  269 % at 512, i.e. "best of N" picks a lucky mode rather than estimating a cost.
- **A rate must be compared against the ceiling its own instruction class can occupy.** Scalar
  `v_fma_f32` 7.30 → `v_dot2_f32_bf16` 23.28 → WMMA 52.63 TFLOP/s (`prototype/issue_rate_probe/results.md`
  §1). Judging a scalar kernel against a matrix roof, or a kernel against a library's tuned kernel,
  both produce a gap that is not there. Check who wrote the kernel you are comparing against: the
  engine's own kernels had zero matrix or dot-class instructions until the WMMA round.
- **Above ~240 GB/s of DRAM traffic means skipped work** — that is the measured LPDDR5x ceiling of
  this part. The legitimate exceptions are named, not inferred: a *demanded* census above the ceiling
  means L1/L2 absorbed the reuse (§54, and §54's A11 row: ~32 MiB at 877 GB/s is the big tier), and a
  bytes-demanded ÷ time figure near the plateau is not evidence of a bandwidth wall — a bit-identical
  instruction cut made a kernel 1.521× faster with the census unchanged, which is only possible if the
  census was never DRAM traffic.
- **A first GPU pass over fresh pages runs ~35 GB/s** — warm before you time, or you are measuring
  page faults (`devices/strix-halo-unified-memory.md`).
- **A timer can be silently dead.** `hipEventElapsedTime` returning 0.000 ms is a broken timer, not a
  fast kernel (§5).
- **A counter that reads exactly 0 is vacuous, not clean**, and a check whose parsed list can be empty
  cannot fail: witness a counter non-zero on a run where it must fire before quoting its zero
  (`AGENTS.md`'s counter rows).
- **Three gate drivers self-lock the GPU** (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`,
  `prototype/gdn_gated_norm/run.py`), so a caller that holds `tools/gpu_lock.sh` and then runs one of
  them **deadlocks it**. Hold the lock only around the self-lock-free scripts and release before
  invoking a self-locking one; check with `grep -c gpu_lock prototype/<op>/run.py`.
- **A/B harnesses assert a behavioural canary before they time anything** — a field the change is
  required to move, read in the same breath as the measurement. A stale binary agreed with itself and
  reported a clean null for a real change; every self-consistency check compiled into the binary under
  test is blind to exactly that.
- **The differential harness is unusable at ≥4k lanes.** Quote 4k decode from the stage-clock burst,
  never from `context_sweep.py`: at 4,032 lanes the differential produced a spread of 29.7 % with a
  "best" faster than a 128-lane step at 32× the KV, which is not a physical result.
