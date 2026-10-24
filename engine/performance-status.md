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
sits; at 4k it comes from the stage-clock burst, never `context_sweep.py` (§6). **The 512–2,048 rows below are the 2026-10-21
window on the D1 tree** (projection flip, §5; on G4 §44 / G1 §48 / A13 §60); 96 lanes reads 9.619 ms = 104.0 tok/s,
and 512's 11.36 % spread is §6's dispatch bimodality, not a regression. Paired, the step moved −13.0 % at 128 and −11.0 % at 4,032 (`proj` 4.1946 → 2.7284 and 4.177 → 2.745 ms/step, 7/7 reps separating).

| lanes at last step | marginal ms/token, best of 3 | spread | tok/s |
|---|---|---|---|
| **4,032** | **10.9357** (cast-fold paired window, 2026-10-21: stage-clock burst, never `context_sweep.py` — §6; the base arm in the SAME window read 11.2365 = 89.0 tok/s, so the step moved −0.3008 ms = −2.7 % by deleting all 97 per-token `cast` launches, which prices the per-launch residual at 3.10 µs against the ~3.8 µs predicted and thereby funds the `glue`/`copy` tail; the D3 row it overtakes read 11.175 = 89.5, and base-vs-ledger across windows is §6's ~2 % drift) | 0.34 % | **91.4** |
| 128 | 9.7855 (same window; base 9.9782 = 100.2, so −1.93 %, and NO LINE was set here — D3/S1 §4 put short widths out of reach of this route in advance, so a move at 128 is a bonus, not the adjudicating row) | 1.64 % | 102.2 |
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

## §7 MTP speculation is the shipping default — the measured state (2026-10-24)

> **Partly overtaken the same day by §8.** The prefill cost recorded below was removed by scoping the GDN span to
> the round's own bodies, and the decode row below was overtaken by a higher number for the same reason. What
> stands from this section: N=2's basis, the break-even accept of p ≈ 0.40, the pathological-fixture result, and
> the admission state of the gates.

Since `e75e192` (+ `1a9d8c9`, `04dd3de`, `aceee97`) the default path of `engine/build/decode_run` is
`--mtp-accept 1 --mtp-k 2 --mtp-draft 1` at N=2, chosen by measurement (`K2-RESULT.md` §7: k=1 +9.75…+10.38 %,
k=2 +16.53/+17.76 % against a pre-registered kill line of round/step ≥ 1.95 that k=2 did not hit). The plain
per-step path is now an explicit instrument: `--mtp-accept 0 --mtp-draft 0`, named on every harness that prices
a per-step cost (61 files, 120 pins). Decision record: `prototype/mtp_accept/DEFAULT-ON.md`.

**Decode, wall tok/s, `mtp_wall_ab.sh`, one lock hold, 5 reps, medians.** Every number below is
`prototype/mtp_accept/mtp_wall_ab.sh` after `0c7ad84` (base arm must name the plain path — unpinned it inherited
the new default and would have timed the accept loop against itself; the MTPSTAT canary aborted instead).

| fixture | lanes | base ms/step | round ms | round/step | accept arm vs base |
|---|---|---|---|---|---|
| templated (`tmp/llama_input/ids_long_chat.csv`) | 4,152 | 11.098 | 20.392 | 1.837 | **+16.50 %** |
| raw prose, long (`tmp/llama_input/ids_long.csv`) | 4,142 | — | — | 1.84 | +16.09 % (carried, `7e80400`) |
| raw prose, pathological (`tmp/verify_body_scale/ids4032.csv`) | 4,032 | 11.295 | 19.438 | 1.721 | **−27.37 %** |

The third row is not a defect and not noise: that fixture has always been the worst accept case in the tree
(`a ≈ 0.10`). Speculation is bought back only by accept, so a default of k=2 has a break-even — with a round
costing 1.72–1.84 × a step, `tok/round = 1 + p₁ + p₂` must clear that, i.e. **p ≈ 0.40** at k=2. Below it the
default is a real loss, and the loss grows with the round cost. Adaptive depth (drop to k=1, or off, when a
running accept estimate falls under the break-even) is therefore the next honest item on the MTP track, not a
refinement.

**Prefill is the default's open cost — carried, measured here, and not accepted.** `--mtp-accept` pins the GDN
launch span to 1 for **every** body including prefill, and the span-1 GDN scan costs 449.07 ms vs 266.03 ms at
4,096 lanes (`engine/decode_run.cpp` comment). `DECWALL prefill_ms`, medians of 3, one lock hold:

| lanes | plain (`--mtp-accept 0 --gdn-loop 512`) | default (forced `gdn_span=1`) | cost |
|---|---|---|---|
| 1,024 | 178.5–189.4 ms → **5,508 tok/s** | 249.9–263.2 ms → 4,096 tok/s | **+37 %** |
| 4,032 | 441.4–450.7 ms → **9,091 tok/s** | 761.0–771.8 ms → 5,257 tok/s | **+72 %** |

Decode is unaffected by the span (a 1-row body already resolved to span 1 under the old default), so the MTP
decode ratios stay clean — but the prefill headline above **replaces** the pre-MTP one until the span
requirement is scoped to the round's own bodies (the `arm_mtp_round_attn_form` pattern at
`engine/decode_step.h:441`). That scoping needs its own admission run, because the round's input state would
then come from a span-512 prefill while every MTP number in this ledger was taken at span 1 throughout.

**Admission state at the new default.** `engine/gate.py --fixture dec8` → `ENGINE GATE PASS`; `--selftest` →
PASS; gate section (m) at the **default** k (18 fixtures, 92 rounds, 188 tokens) shows the same two admitted
forks as the pre-flip record — no new bit divergence. Full-population losslessness re-taken at the default:
49/49 corpus lines byte-identical to the plain stream, 743 rounds, 1,617 tokens, 2.176 tok/round, identical
counters to the explicit `--mtp-k 2` run (`m3_loop_rate.sh`, `LOSSLESS=1`). `speccheck/run_all.py` 26 gates
0 failed; op gates glue 784/0, proj_gemv 952/0, `gdn_gated_norm` DEVICE GATE PASS.

## §8 The GDN span is scoped to the round — §7's prefill cost is retired, and decode moved (2026-10-24)

§7 recorded the shipping default paying **+37 % at 1,024 lanes and +72 % at 4,032** because `--mtp-accept`
pinned the GDN span to 1 for every body. That requirement belongs to the *round*, not the run: it exists because
the mid-round checkpoint restores a one-token-step recurrence's end state. It is now enforced where it is owed —
`engine/decode_step.h::kMtpRoundGdnSpan = 1` is taken when the round's checkpoint is armed and `sc.gdn_loop`
otherwise (the same arming `arm_mtp_round_attn_form` keys on), the run-wide resolution and the CLI-level refusal
are gone, and the mode line echoes both spans:
`MTP mode=accept k=2 draft=1 gdn_span_round=1 gdn_span_other=512 forced_by=default`.

**Prefill, three arms in one lock hold, medians of 3** (`prototype/mtp_accept/scoped_prefill_ab.sh`; arms =
scoped default / plain `--mtp-accept 0 --gdn-loop 512` / `oldpin` = `--gdn-loop 1`, i.e. what the default paid
before this change):

| lanes | scoped default | plain | oldpin (pre-scoping default) | scoped vs plain |
|---|---|---|---|---|
| 1,024 | 165.6–166.7 ms → **6,147 tok/s** | 165.3–167.7 ms → 6,120 tok/s | 246.2–249.0 ms → 4,135 tok/s | **±0 %** |
| 4,032 | 442.5–456.4 ms → **9,004 tok/s** | 443.6–451.1 ms → 9,017 tok/s | 763.8–773.9 ms → 5,227 tok/s | **±1 %** |

The default now costs prefill nothing — §7's regression table is retired, and the pre-MTP prefill headline is
back (9,004 tok/s at 4,032 against §7's 5,257). `oldpin` is kept in the table so the size of what was removed
stays visible.

**Decode moved, and not for the reason one would guess.** Paired A/B at templated 4,152 lanes inside one window,
`mtp_wall_ab.sh` with `ACCEPT_EXTRA` empty (shipping spans) versus `GDN=1` (the pre-scoping whole-run pin),
5 reps and 2 reps respectively, same fixture, same lock discipline:

| spans | base ms/step | round ms | round/step | accepted / rounds | emitted | vs base |
|---|---|---|---|---|---|---|
| `gdn_span_round=1 other=512` (ships) | 11.088 | 20.280 | 1.829 | **135 / 255** | 578 | **+23.44 %** |
| whole-run span 1 (pre-scoping) | 11.044 | 20.428 | 1.850 | 117 / 255 | 548 | +15.72 % |

Both reps of each arm printed identical accept counters, so this is not window noise: scoping the span raised the
**per-round acceptance on this fixture** from 117/255 to 135/255 (0.459 → 0.529) as well as removing the prefill
tax. **The mechanism is not established and is not claimed here.** What is measured is that the span used by the
non-round bodies changes the state the round starts from, and with it which proposals the draft makes — the round's
own span is 1 in both arms. Two candidate readings, both unverified: the span-512 prefill leaves a GDN state the
draft block is better conditioned on (span 512 is also what the pinned HF reference path used, so its state is the
closer match to every pin in this repo), or the two spans simply produce different states and one happens to suit
this fixture. Deciding between them needs a span-1-vs-512 accept ladder on more than one fixture, which is a
measurement, not a re-reading of this one. Either way losslessness is what adjudicates it, and it holds.

**Current headline: decode +23.44 % over our own plain path at 4,152 lanes with the shipping default** (llama.cpp's
best MTP configuration measures +13.6 % on the same prompt), and prefill unchanged from the pre-MTP ledger.
Losslessness holds at the scoped spans: `m3_loop_rate.sh` full population — lines compared 49, identical 49,
flipped 0, kill-line verdict 0.6703 vs 0.4568 (worker's admission; my own prior-default run had 743 rounds,
1,617 tokens, 2.176 tok/round). Gate (m) asserts the echoed span pair on 18/18 fixtures; `default_refusals.sh`
25 ok / 0 FAIL, with the two span refusals now positive cases and `--mtp-midsplit`'s sibling refusal kept because
its unsplit arm arms no checkpoint.

Still open, unchanged by this: the pathological 4,032 fixture is still a loss at k=2 (break-even accept p ≈ 0.40),
so adaptive depth remains the next item; `k=3` unimplemented/ungated; `kArgmaxChunks` unswept; gate (m) depth
bounded at 8 because `--mtp-accept` refuses `--replay`.

## §9 Two §8 claims retracted, and the honest decode band (2026-10-24)

§8 made two claims that do not survive measurement. Both are retracted here; the numbers that refute them are
mine, taken in clean exclusive windows after §8 was committed.

**(1) "Scoping the span raised the accept rate" — false.** It changed acceptance non-systematically. Same-fixture
paired counts: templated 4,152 accepts **135/255** at shipping spans vs **117/255** at the whole-run pin (+18),
but raw long 4,142 accepts **102/255** vs **106/255** (−4). Deterministic within a fixture, opposite in sign
across fixtures — so the span used by non-round bodies changes *which* proposals the draft makes, with no
direction to claim. The +23.44 % at templated width is therefore not a gain the scoping delivered; it is that
fixture's luck of which rounds flipped, and the scoping's honest benefit is the prefill tax alone.

**(2) §7's −27.37 % row was never a decode measurement.** `tmp/verify_body_scale/ids4032.csv` ends
`stop=1 ended=stop_token` with `rounds=3 plain=1 accepted=0 emitted=5` — it stops at the end-of-turn token after
**five** emitted tokens, so its `tok_per_s` is fixed cost divided by five tokens and never describes a decode
loop. §7's break-even argument built on that row is retired with it.

**The valid picture at the shipping default** (`--mtp-accept 1 --mtp-k 2`, `gdn_span_round=1 gdn_span_other=512`),
each row a paired A/B in one lock hold, medians:

| population | lanes | a | round/step | accept vs base |
|---|---|---|---|---|
| corpus, 12 cases, natural width | ~0.3k | 0.5269 | 1.971 | **+17.87 %** |
| templated long prose | 4,152 | 0.529 | 1.829 | **+23.44 %** |
| raw long prose | 4,142 | 0.400 | 1.841 | **+12.88 %** |
| corpus case that fits 4k (1 case) | 4,032 | 0.0323 | 1.889 | **−28.28 %** |

**Honest headline: +12.9 % to +23.4 % across the valid long-prose fixtures, +17.9 % on a 12-case population,
against llama.cpp's best +13.6 % — parity to clear win.** The cost case is also now properly attributed: the loss
appears at **near-zero accept** (a = 0.03 → −28 %), not at "pathological width". Two consequences: the default is
only safe if acceptance is not near zero, so **adaptive depth is funded** (drop to k=1, then off, on a running
accept estimate); and `corpus_wall_ab.py --ctx 4032` is nearly unusable as a population — the corpus prompts are
short (narrowest 304 lanes), so asking for 12 cases at 4k yields **1** case, which the harness prints and which a
reader would otherwise tabulate as twelve.

Harness bug found while making these numbers, fixed in `9e711c7`: `corpus_wall_ab.py` indexed `per_rep` with a
position taken from a *sorted* copy of the ratios, so the "medians" table could pair one rep's ratio with
another's ms and tok/s; at an even `--reps` it raised `ValueError` after finishing all device work. Also: the
harness line `parity needs a* = round/step - 1` is the **k=1** identity (`tok/round = 1 + a`). At k=2 parity needs
`p₁ + p₂ = round/step − 1`, so comparing the printed `a` with `a*` across k is the same nested-vs-marginal
confusion that the llama comparison already documented. Quote the adopt line, not `a*`, for k > 1.
