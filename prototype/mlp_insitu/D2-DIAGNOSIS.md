# D2 — diagnosing the MLP in-situ penalty

**Status: instrument built, nothing measured.** This is a diagnosis package, not an optimization and
not a result. It contains: the three candidate causes restated so each is adjudicable; the rig
(`mlp_insitu.hip.cpp`) that runs the shipping kernel under one-variable-changed arms; a source-derived
traffic census (`census.py`); the sweep script the main agent runs (`ab_d2_sweep.sh`); and a
pre-registered prediction with a number in it. **No millisecond figure in this file is measured here.**
Every one is quoted, and labelled.

Quoted, from the ledger and from this route's brief (do not re-take any of these):

| quantity | value | source |
|---|---|---|
| `mlp` isolated burst, 528 MB, `contig` | **2.275 ms/token** (98 % of probe) | `engine/performance-status.md` §4 |
| `mlp` in-situ, stage granularity, 128 lanes | **3.3880 ms/step** | `progress/2026-10-21/performance-status-2026-10-21.md` §118 row / attribution §4 table |
| `mlp` in-situ, stage granularity, 4,032 lanes, today | **3.3628 ms/step**, 48 launches, 70.06 µs/launch | main agent, this week's window (brief) |
| step at 4,032 lanes today | **11.1750 ms/token**, `mlp` the largest family | main agent, this week's window (brief) |
| read probe denominator | **237 GB/s** | `devices/strix-halo-memory-bandwidth.md` |
| pre-registered lines | WIN ≤ **2.90 ms**, KILL ≥ **3.20 ms** | `progress/2026-10-21/decode-attribution-2026-10-21.md` §5, D2 row |
| per-segment constant the ledger already subtracts from a stage row | **7.2 µs × segments**, i.e. 0.346 ms for 48 segments | attribution §4 table, "− segments × 7.2 µs" |
| rig standalone thin-shape walk, 504.000 MiB, 48 launches | **2.9523 ms/token = 179 GB/s demanded** | main agent, first window of this rig (abort row R0) |
| same window's read-size control | **510 MiB @ 225.5 GB/s**, **1535 MiB @ 235.9 GB/s** | main agent, first window (`mixed_bw` rows) |

### 0(iv) The published floor and the shipped walk are not the same launch shape, and that is a term

The first window aborted on the reproduction gate, and the reason is a spelling difference that the
phrase "same 528 MB, same `contig` map" had been hiding. `prototype/mlp_gemv/run.py --layers all
--tile 24 --mapping contig` (provenance: `progress/bench/evidence/ledger-2026-10-16/
commands-and-provenance.md` §6) builds `Wg_t.bin` as **one layer's tensor replicated 24× and
concatenated**, and `mlp_gemv.hip.cpp` then derives the row count from the file size
(`rows = wbytes / (in*2)`, line 219). So the burst that produced 2.275 ms is not 48 launches. Read
both spellings side by side:

| | the published 2.275 ms burst | the engine, and this rig's `--geometry thin` |
|---|---|---|
| tool | `mlp_gemv --mode mlp --tile 24`, `run.py` divides the iteration by 24 | `engine/kernels_mlp.cpp`, one layer per launch |
| launches per 24-layer token | **2** (gateup, down) | **48** (gateup, down × 24 layers) |
| gateup rows / grid | **86,016 / 10,752 blocks** | **3,584 / 448 blocks** |
| down rows / grid | **24,576 / 3,072 blocks** | **1,024 / 128 blocks** |
| threads, rows/block, kernel, mapping | 256, 8, `mlp_gateup`/`mlp_down`, `contig` | 256, 8, same templates, `contig` |
| weight allocations | 3 × 168 MiB (`hipMalloc` per file) | 3 × 168 MiB in `blob3`; 72 × 7 MiB as shipped |
| bytes per token | 504.000 MiB | 504.000 MiB |

Same bytes through the same kernel at 1/24 the launches and 24× the work per launch. The rig can be
*either* shape (`--geometry thin|fat`), and `--token-burst N` separates the other pipelining claim (the
published burst submits 30 iterations between one sync; the shipped measure drains every token). So the
budget the route was funded on is now written as three terms instead of one:

```
   1.09 ms  QUOTED: stage-row in-situ (3.3628) - published burst (2.275)
 =  T_geo   rig's own thin floor minus its fat floor   <- launch granularity, not memory state
 +  T_serial 48 drains vs one burst                   <- the instrument's shape (ledger: 0.346 ms)
 +  P     R7 - R0(thin)                               <- what D2's three candidates are actually about
```

**The consequence, stated not worked around.** With R0 = 2.9523 measured in the abort window, P against
the shipped-shape floor is **0.323 ms** — *below* the pre-registered 0.35 ms NO-DIAGNOSIS floor. If the
fat arm reproduces 2.275 and P stays under 0.35, the honest result is that **there is no ~1 ms
memory-state penalty at the shipped launch geometry to decompose**, that the ledger's 1.11 ms is mostly
launch shape plus the drain term, and that the −0.80 ms D2 prize must be re-cut before any code is
funded. The 2.275 target is *not* moved: canary C2 now measures the rig's fat arm against it, so if the
fat arm also misses, that is a NO-RESULT about the rig, and it is visible.

## 0. What the penalty is, and what the instrument that found it can and cannot see

`mlp` costs 3.3628 ms/step inside the step and 2.275 ms/token alone: a **1.09 ms penalty on a kernel
that is at 98 % of the read probe by itself.** Two facts about the measurement change what the three
candidate causes can even mean, and a third question is cheaper than all of them.

**(i) The family row comes from the SERIALIZED stage pass.** `engine/stage_clock.h`: *"family X's `ms`
is what family X took when nothing else was in the queue"*. `note()` closes a segment whenever the
family changes (`stage_clock.cpp`: `if (open_ && (mode_ == Mode::kKernel || s != stage_))
close_segment();`), and inside one layer the MLP's two launches are separated from each other by the
`cast` note — so **each of the 48 MLP launches is its own drained segment**, at ~70 µs each. The
`STAGE ... segments_per_step=` field is the witness and the main agent should print it once to confirm
the expected 48.

Consequence: candidate (a) in its literal form — *other families' reads and fills sharing the fabric at
the same time* — is **excluded by the instrument**, not measured by it. Nothing is in flight beside a
drained family. What a drained family can still inherit is **state**: cache-tier contents, page tables,
and lines left dirty by the step's 24.409 MB/token of writes (`census.py`). So the (a) arm has to be a
*carry-over* arm with a dose, plus a separate concurrent arm that is honest about being a bound.

**(ii) A part of the 1.09 ms is already named, by the ledger itself.** attribution §4 subtracts
`segments × 7.2 µs` from every stage row and gets `mlp` 3.388 → **3.04 ms** of wall share against a
2.244 ms floor: gap **0.80 ms** — which is exactly the prize D2 was funded with. So the honest budget is

```
   1.09 ms  burst -> stage-row penalty
 = 0.35 ms  the drain/dispatch shape of the row itself (48 segments x 7.2 us), NOT a memory mechanism
 + 0.74 ms  the memory-state residue, which is what D2's prize actually buys
```

The rig measures the first line directly (`--drain each` vs `--drain burst` with no other traffic at
all) and that measurement is a **pre-registered cross-check**: it must come out near 48 × 7.2 µs =
0.346 ms. If it lands far from that, either the rig's drain shape is not the stage clock's or the
ledger's 7.2 µs constant is wrong for this family — and either answer is worth more than the
attribution, because the whole per-family prize table is built on that constant.

**(iii) The first fork is cheap and it is not among the three candidates.** Is the penalty *fixed per
launch* or *proportional to bytes*?

| model | gateup (14.7 MiB weights, roofline 61.9 µs) | down (7.34 MiB, roofline 30.9 µs) | what it points at |
|---|---|---|---|
| measured mean today | — | — | 3.3628 / 48 = **70.06 µs/launch** |
| fixed ≈ 23.6 µs per launch | ≈ 85.5 µs | ≈ 54.5 µs | drain, dispatch, empty-queue ramp — a **step-shape** answer, not a memory one |
| proportional, ×1.489 | ≈ 92.2 µs | ≈ 46.0 µs | bytes-proportional: contention, page walk, writeback — a **memory** answer |

Two rows (`--kernels gateup` and `--kernels down`, each at both drain settings) decide this before any
of (a)/(b)/(c) is spent, and if it comes out fixed-per-launch the three memory candidates are largely
moot for this route line.

## 1. The three causes, the arm that adjudicates each, and the reading that rules it out

Notation: every term is a difference of two rows **printed in the same window** (a ratio against a
historical row is not a ratio — `devices/strix-halo-mixed-traffic.md` rule). **`P` = reproduced penalty
= `R7 − R0`**, where `R0` is the rig's **thin** (48-launch, shipped-shape) standalone floor — the floor
changed meaning in §0(iv) and the formula did not. Shares `s_x = T_x / P`. Two terms live *outside* P
because they are shape, not memory: **`T_geo = R0 − R0f`** (thin vs the published fat shape) and
**`T_pipe = R0 − R0b`** (a drained token vs 8 tokens between syncs). **A term is CONFIRMED at `s ≥ 0.25`, REFUTED at `s ≤ 0.10`, PARTIAL
between; the residue `1 − Σs` is printed, never absorbed.** Doses are sized from `census.py`'s
`STEPMIX` line, which is parsed from `tools/step_traffic_model.py`: non-MLP traffic is **957 MiB read and
23 MiB written per token**, and that 957 splits by *placement*, which matters for the arms — **471 MiB is
per-layer traffic** (attention + GDN + norms, `read_mib_per_layer=19.627 × 24`) and **486 MiB is `lm_head`,
issued once per step** (`head_mib=485.947`). A dose arm can only place the per-layer part around every
launch, so the two get separate terms rather than one mislabeled 957.

### (a) interleaving with the rest of the step's traffic

Restated for a drained family: *what the MLP inherits from the ~957 MiB read and 23 MiB written that
ran immediately before it* — cache-tier contents evicted, page tables cold, dirty lines still to write
back. True concurrency is a separate sub-arm and is a bound, not the shipped shape (§0(i)).

* **Arm A-dose** — `--drain each --layout shipped72 --if-read-mib {118, 236, 471} --if-when before` (rows
  R4a/R4b/R4c). The interferer is submitted and drained *outside* the timed window, so the row is the MLP's
  own time as a function of carry-over state, and three doses make it a **slope**, not a single difference.
  Estimator: **T_readcarry = R4c − R3**, the per-layer 471 MiB. Plus R9 (`--mlp 0` at the full dose) to
  prove the interferer alone does not cost what is being attributed to it.
* **Arm A-head** — the `lm_head`'s 486 MiB is the step's single largest read and it is placed **once per
  step**, so no per-layer dose arm carries it. It is present only in R7 (via `--head-mib 485.947`), and its
  carry-over is **T_headcarry = R7 − R4c − T_writecarry**. Naming it is not pedantry: left unnamed it would
  hide 486 MiB of the step's traffic in the residue and make candidate (a) look unexplained.
* **Arm A-conc** — `--if-when concurrent --drain burst` (two non-default streams; the legacy default
  stream implicit-syncs and would serialize this into a null). Estimator: cross-term
  `t(A,B) − t(A) − t(B)`, which needs all three rows.
* **Ruled out if** `T_readcarry + T_headcarry ≤ 0.10 × P` **and** the cross-term `≤ 0.10 × P`. **Confirmed if**
  either carry term `≥ 0.25 × P`.
* **What was already done, and what it did not settle.** Q9's `B_blk`/`B_rr` rows refuted launch
  **order** (1.005, identical launch multiset, two orders) — this rig does **not** re-test order. Its
  `B_rr_nostandins` row (87.4 % vs 77.7 %) is the only measurement in the tree that hints at this
  candidate's cost, and it lumps all bytes together so it cannot give a marginal slope. That slope is
  the new quantity.
* **A free bound from the ledger.** From 128 → 4,032 lanes the step adds ~45.6 MB of KV read and `mlp`
  moves ~−0.025 ms (3.3880 → 3.3628, inside window drift). That bounds the marginal carry-over cost at
  ≲ 0.7 µs/MiB *in that range*; linearly extrapolated to 957 MiB it would still be ~0.65 ms, so it does
  **not** refute (a) — it says the dose arm is needed and that a slope even half that large is decisive.

### (b) the DRAM page walk over 24 layers' disjoint weight ranges

The published burst is **not** the engine's placement. `prototype/mlp_gemv/run.py --tile 24` replicates
one layer's three tensors and concatenates them, so the 2.275 ms row walks **three** contiguous regions
(168 + 168 + 168 MiB) inside a ~504 MiB touched set. The engine
(`engine/loader.cpp`: *one `hipMalloc` per checkpoint tensor*, 335 of them) gives `mlp` **72 separate
7 MiB blobs scattered through a 1.5457 GB resident set** that is walked once per token. Same bytes,
same kernel, same `contig` walk — different region count, different span, different translation load.
That asymmetry is invisible in the phrase "same 528 MB, same `contig` map", and it is the largest
unexamined difference between the two numbers.

* **Arm B-region** — `--layout blob3` → `--layout blob72`: 3 regions → 72, adjacent, bytes identical.
  Isolates **region count / a walk chopped into pieces**. `T_region = R2 − R1`.
* **Arm B-span** — `--layout blob72` → `--layout shipped72 --gap-mib 42`: the same 72 blobs separated
  by warm, allocated-but-never-read stand-ins for the other ~1 GiB, so the span goes 504 MiB → ~1.5 GiB
  with **zero additional bytes touched in the window**. Isolates **translation reach + DRAM locality of
  a scattered working set** from traffic, which is the whole point of separating it from (a).
  `T_span = R3 − R2`, with a `--gap-mib {12, 42}` dose on top.
* **Size control, same window, already written**: `mixed_bw --mode read --footprint-mib 1536`. If a
  contiguous 1.5 GiB sweep still reads ~235 GB/s, footprint *size* is dead as an explanation and only
  *placement* can carry (b). This row is why the span arm is interpreted at all.
* **Ruled out if** both `T_region ≤ 0.10 × P` and `T_span ≤ 0.10 × P`. **Confirmed if** either
  `≥ 0.25 × P`. Partial reports the share of the 0.74 ms residue closed, not a rounded-up story.

### (c) the `act`/`delta` writes landing in the same window as the weight reads

Arithmetic first, from `census.py` (parsed, not typed): the MLP writes **0.221 MB/token** — 7 KiB of
`act` plus 2 KiB of `delta` per layer — against 504 MiB read, i.e. **0.0419 %**. For those writes to
explain 1.09 ms a write would have to cost **4.95 GB/s, ~48× a read**, which is the same shape of
refutation Q8 already used against the whole-step write story (it needed 4.11 GB/s and got dismissed on
the measured 210 GB/s copy rate). There is also a shipping detail that makes it weaker still: `act` and
`delta` are one **reused** scratch per output (`decode_step.cpp` writes `sc.mlp_act`, `sc.delta` every
layer), so those 9 KiB are re-written in place 24× a token and essentially never leave the cache.

* **Arm C-own** — `--writes shared` (as shipped) → `--writes spread --write-mul {1,16}`: identical
  stores, identical byte count, destinations moved 1 and 16 MiB apart per layer so the lines cannot
  survive between layers. That is the *upper bound* on what the MLP's own stores can cost;
  `T_ownwrites = R6b − R1`. If even the spread form is ≈ `shared`, (c) as posed is dead whatever the
  arithmetic says.
* **Arm C-writeback** — `--if-write-mib 23 --if-when before`: the step's *other* 23 MiB of writes
  issued and then **drained**, so the kernels are done but the lines are dirty when the MLP segment
  opens. This is the only route by which write traffic reaches a drained number, and Q8's `mixlaunch`
  rows did not test it (they put the writes in flight *concurrently*, at 62:1, and got 89.9 % of probe).
  `T_writecarry = R5 − R3`.
* **Ruled out if** `T_ownwrites ≤ 0.10 × P` **and** `T_writecarry ≤ 0.10 × P`. **Confirmed if** either
  `≥ 0.25 × P`. A confirmed C-writeback is a *step-layout* consequence (move the GDN state writes off
  the MLP's window), never a kernel edit, so it must not be confused with C-own.

### The term none of the three is, and it is on the record

`T_serial = R1 − R0` (drain each vs drain burst, MLP alone, no traffic at all). It is not a memory
mechanism and no kernel edit removes it, but it is ~1/3 of the penalty being diagnosed and the ledger
already bills it at 0.346 ms. **If `T_serial ≥ 0.55 × P`, the D2 line as written — a
stage-granularity family row — is not winnable by any change to the kernel or the memory system, and
the route must be re-written against the burst wall before code is funded.** That is a legitimate,
useful, pre-registered outcome of this window.

## 2. The rig, and what is deliberately not in it

`prototype/mlp_insitu/`, prototype-only; `engine/` untouched, nothing in `progress/` touched, no gate
threshold moved, no commit made.

| file | what it is |
|---|---|
| `mlp_insitu.hip.cpp` | the rig. `#include`s `../mlp_gemv/mlp_gemv_kernels.h` — the **same header `engine/kernels_mlp.cpp` includes** — so there is no second kernel to drift. Every arm is a flag. |
| `census.py` | the traffic census, parsed out of `mlp_gemv_kernels.h`, `kernels_mlp.cpp`, `decode_step.cpp`, `state.h` and `tools/step_traffic_model.py`. Refuses an unlabelled >240 GB/s figure; every regex must match its expected record count or it dies. |
| `ab_d2_sweep.sh` | the main agent's window: one `tools/gpu_lock.sh` hold, blocking, canaries inside the window, empty-parse guards, `NO-TIMING:` on any red. |
| `build.sh` | `-O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib`. |

**Existing instruments, listed before adding one** (AGENTS.md: *before writing a census tool, list the
census tools*), and what this rig does that they cannot:

* `tools/mixed_bw.cpp` — the traffic-**curve** tool. It runs a *copy* of the engine's read mapping with
  the maths removed, so it can price a pattern but not a kernel; it cannot place blobs at shipped
  spacing, cannot put a store in the window, cannot drain between them, and its `B_blk`/`B_rr` rows
  already refuted launch **order** (ratio 1.005) — which is why nothing here re-tests order. Its
  `--pattern engine-mlp` is a byte-identical *read* pattern; here the object is the kernel with its dot,
  epilogue and store. **Its rows stay in the main agent's window** as the same-window normalization.
* `tools/step_traffic_model.py` — the byte census from `engine/state.h`. This rig *consumes* it via
  `census.py`; the interferer sizes are never typed.
* `prototype/mlp_gemv/` — the F1–F4 numerics gate over this exact header, 192 cases, with the band
  history and the traps already paid for. **This rig neither re-runs it nor re-bands it.** Its own
  witness is deliberately narrower and exact — see below.
* `tools/bw_probe_hip.cpp` — the quoted 237 GB/s denominator.

**The rig's witnesses, and their declared blindness.** The rig needs to prove only that *its geometry and
its buffers are right*, and it does that without touching any band:

* **exact-pre** — feed `x = one-hot(j)`. Every dot is then exactly one product, `bf(W[i][j]) × 1.0`, so
  each output slot must equal a known weight word **bit for bit**: no tolerance, no summation-order
  question, no `expf` on the host, nothing to re-band. Catches a wrong word, wrong row, dropped word,
  wrong output slot.
* **landing** — every `act`/`delta`/`pre` slot written (buffers pre-set to 0xFFFF, a pattern no store in
  this kernel emits), plus a poison guard band around every region.
* **if-read / if-write** — the interferer's block-reduced double checksum equals `4.0 × float4s billed`,
  and the write arena is read back against a value the writer never stores. One sums slot per launch, so
  a launch that never ran or ran twice cannot hide.
* The must-catch sets are **derived from the geometry**, and `--selftest-geom` refuses an injection whose
  must-catch set is empty, or whose derived set does not match: `NEG_TAIL` may be caught only at
  `j == in-1`, `NEG_HALF` only at `j >= in/2`, `NEG_SWAPROW` by exact-only (a permutation writes every
  slot, so `landing` is blind by construction), and `NEG_F32ACT` is a declared **blindness control** —
  it is invisible to this rig on purpose, and its catcher is `prototype/mlp_gemv`'s F2 gate. A harness
  that catches everything proves nothing, in both directions.
* Every device case runs in its **own forked child**, because one faulting kernel poisons the context
  and every later case in that process silently never ran (`mixed_bw`'s lesson).
* Geometry is re-derived on the host from the same fields the launch is built from (`bounds_for`:
  `row = blockIdx.x*rpb + threadIdx.x/ws`, `v = lane; v < IN/8; v += ws`), and `--selftest-geom` checks
  the re-derivation lands exactly on the blob end — the `mixed_bw` "the solver and the kernel meant two
  different things by `rounds`" lesson.
* Cold pages are refused, not averaged (`--pages cold` exits 2). `hipMalloc` only. No ms is printed
  without `--time`; `--selftest` prints no ms and no GB/s.

**No decision check, on purpose.** Nothing here can move a token: `engine/` is untouched, the rig is a
separate binary over synthetic weight bytes, and `sampled.txt` parity is meaningless for it. The
canaries in the window are instead (1) the burst reproduction, (2) the rig's own byte witnesses on every
timed row, (3) the drain-cost cross-check against the ledger's 7.2 µs constant, (4) `verify=ok` enforced
on every row before its ms is used.

### 2.5 Four things this rig's own gate caught before it could print a confident wrong answer

All four were found by *running* the untimed gate, not by reading the code, and each is a class this file
already records (a check that cannot fail, or a harness that parses nothing):

1. **fork-after-runtime-init reads as OOM.** `--selftest` forks a child per case, but the parent's CPU
   geometry pass called `hipDeviceGetAttribute` for the warp width *before* forking. Result: 76 children,
   76 `hip error out of memory`, `children_bad=0` — no real memory limit anywhere, and the message points
   at a `hipMalloc` line that was not the cause. Fix: the warp width is cached and *no HIP call may sit on
   a path reachable before the fork*; `bill()` is device-free by construction.
2. **a permutation control needs a pattern that varies along the permutation.** `NEG_SWAPROW` (`row ^ 1`)
   scored 14 false clean passes: the tag was `tag(j)`, identical in every row, so permuting rows permuted
   equal numbers and `exact` could not see it. Now `tag_bits(row, j)` — BF16 `1 + k/256`, exactly
   representable, distinct per (row, column), never the `0xFFFF` unwritten sentinel.
3. **a negative control whose drop is never scheduled cannot fire.** `NEG_SHORT_READ` meant "read one float4
   short" as `last = n4-1` for threads with `t < gridDim.x` — but a grid-stride thread only visits index
   `n4-1` when `t ≡ n4-1 (mod stride)`, and at `stride = 65536` that thread was outside the selected set, so
   **zero** float4s were dropped and the account matched exactly. Now the injection drops a whole grid
   stride (`n4 - stride`): one round short for every thread, invisible to every witness except the byte
   account — which is the property the control exists to test.
4. **a read cannot break a write guard, and expecting it to made a good control look broken.** `stray_read`
   was declared to break the poison band; a stray read only *poisons the checksum* (it read `NaN`) and the
   band stays intact. The expectation was wrong, not the kernel: the band is a **write** witness and the
   account is the only catcher for a short or stray read. The case's own `why=` says so now.

Corollary kept from the first version: the must-catch / must-not-catch sets stay *derived*
(`--selftest-geom` refuses a `NEG_TAIL`/`NEG_HALF` injection whose derived catch set is empty), and
`NEG_F32ACT` remains a declared **blindness control** — invisible to this rig on purpose, because its
catcher is `prototype/mlp_gemv`'s F2 gate, which this rig neither re-runs nor re-bands.

### 2.6 What the first window's abort found — three defects, all in the rig, none in the kernel

The window aborted on its own admission and printed `NO-TIMING`, which is the instrument working. Three
things it printed on the way out were the rig's own bugs:

1. **`verify=okFAIL_delta_unwritten` on a gateup-only arm.** `delta` is `down_proj`'s output; a row that
   never launches `down_proj` cannot land one, so the *must-land set* was hand-listed instead of derived
   from the arm's launch list. Fixed by `must_land_mask(o)`: `act` iff the arm runs gate/up, `delta` iff
   it runs down, and `verify_pass` now refuses a pass with `checked == 0` (`FAIL_nothing_expected`), so a
   row cannot pass by expecting nothing. `--selftest-geom` asserts the derived set is non-empty and
   correct for all three `--kernels` values. The pass string is also no longer self-contradicting: it was
   literally printing `okFAIL_…`, which a `grep verify=ok` scores as a pass.
2. **`R0` did not reproduce the published burst — and the reproduction gate is the only reason that is
   cheap.** See §0(iv): the published 2.275 ms is a 2-launch shape, `R0` was the shipped 48-launch shape,
   and the two differ by `T_geo`. Fixed by making the shape a first-class arm (`--geometry thin|fat`,
   with `fat` refusing `blob72`/`shipped72` and `--writes spread` because its contiguity premise is
   false there), plus `--token-burst N` to test the pipelining claim separately instead of arguing about
   it. The rig's own geometry check caught a second bug while being written: `--blocks-cap 448` was
   silently capping the fat grid back to 448, i.e. the fat arm would have been a thin arm wearing a label.
   A third surfaced on device: `act`/`delta` were sized for `thin`, so the fat launch wrote past `act`
   and the landing witness read 165 KB beyond the allocation — which it reported as a `memcpy` error for
   `delta` and, worse, **passed vacuously for `act`** because the bytes beyond a 7 KB buffer happened not
   to be the unwritten sentinel. Buffer sizes now follow the geometry and `verify_pass` refuses any arm
   whose expected landing set is larger than the buffer that has to hold it.
3. **`--mlp 0` printed `0.0001 ms`.** A control whose timed window launches nothing has no milliseconds;
   this part's timer returns ~0 below a millisecond (AGENTS.md), and a reader tabulates 0.0001 as
   "essentially free" and it silently enters a subtraction. Such a row now prints
   `NO-TIMING … no_launch=1 … reason=timed_window_launches_nothing` and the sweep records it as a witness
   row, not a number. `T_concurrent` consequently dropped its `− t(R9)` term and is stated as the plain
   `R8 − R0` bound it always was.

Each fix has a witness: `--selftest-geom` (derived must-land set, fat/thin byte equality, fat grids
10,752/3,072, the refusals) and `--selftest` (76 device cases, unchanged and still green after the
per-layer `exact_expect` change, so the fat arm's 86,016-slot landing is checked by the same one-hot
witness as the thin one).

## 3. The prediction, stated so this window can refute it

With `P = R7 − R0` (the rig's own reproduced penalty, against the **thin/shipped** floor — see §0(iv))
and the shares as defined in §1:

1. **Reproduction gate, re-anchored but NOT loosened.** C2 is now `R0f` (the rig's **fat** arm, the
   published 2-launch shape) within **±3 %** of the quoted 2.275 ms. The number being hit and the
   tolerance are exactly the pre-registered ones; what changed is that the arm being compared is the same
   *shape* as the published row rather than a different one. `R0` is no longer expected to reach 2.275 —
   the gap is `T_geo`, printed as its own term. C3 is unchanged at **P ≥ 0.35 ms**; if it trips, the
   reading is the §0(iv) consequence (no ~1 ms memory-state penalty at shipped geometry), which is
   reported as the result and not routed around. Both gates now run **before** the matrix is paid for.
2. **Drain cross-check:** `T_serial = R1 − R0` must land in **0.25–0.45 ms** (the ledger's 48 × 7.2 µs =
   0.346 ms ± reproducibility). Outside that band, the prize table built on that constant is the finding.
3. **The prediction proper:** the residue is **traffic-state, not placement**:
   `T_readcarry + T_writecarry + T_headcarry ≥ 0.40 × P`, with `T_span ≤ 0.15 × P`.
4. **Refutation lines:** `T_serial ≥ 0.55 × P` ⇒ the D2 line is an instrument artefact of the stage row
   and must be re-cut against the burst wall before code is funded. `T_span ≥ 0.25 × P` ⇒ it is
   placement/page-walk, and the fix is a step-layout change (group the MLP launches / co-resident
   weights), not a kernel. `T_ownwrites ≥ 0.25 × P` ⇒ (c) as posed after all, against the 48× arithmetic.
5. **What the route's own lines mean today.** KILL is 3.20 ms and the row is 3.3628: the route is
   **0.16 ms from KILL and 0.46 ms from WIN**, and 0.16 ms is 3.4 µs/launch — under the per-segment
   constant the ledger subtracts. So the D2 verdict cannot be read off one re-take of the family row; it
   has to come from the attribution above. Say which rows were compared, and quote spreads.

## 4. Handover — the exact command sequence (main agent, one lock hold, blocking, zero subagents)

Nothing here runs in the background and nothing here is run by a worker with `--time`. `mlp_insitu` does
**not** self-lock (checked: `grep -c gpu_lock prototype/mlp_insitu/*.py` is 0), so wrapping it in one
outer hold is safe; `tools/mixed_bw` does not self-lock either, and neither does `mlp_gemv/run.py` (the
three self-locking drivers in this tree are `glue`, `proj_gemv` and `gdn_gated_norm`, and none is called
here — checked with `grep -c gpu_lock`).

```bash
# 0. CPU-only, no lock, no device: build + the two CPU gates.
bash prototype/mlp_insitu/build.sh
prototype/mlp_insitu/mlp_insitu --selftest-geom          # must print SELFTEST-GEOM ok fails=0
.venv/bin/python prototype/mlp_insitu/census.py --emit-stepspec

# 1. Untimed device witnesses (worker-allowed, <60 s, PASS/FAIL, prints no ms).
tag="d2-selftest"; tools/gpu_lock.sh acquire "$tag"; tools/gpu_lock.sh util
prototype/mlp_insitu/mlp_insitu --selftest; echo "rc=$?"
prototype/mlp_insitu/mlp_insitu --geometry fat --reps 1 --iters 1                 # NO-TIMING, verify=ok
prototype/mlp_insitu/mlp_insitu --kernels gateup --reps 1 --iters 1               # VERIFY land=act-no-delta
prototype/mlp_insitu/mlp_insitu --mlp 0 --if-read-mib 8 --reps 1 --iters 1        # no_launch=1, no number
tools/gpu_lock.sh util; tools/gpu_lock.sh release "$tag"

# 2. THE 20-SECOND REPRODUCTION CHECK, first, before the matrix is paid for. One lock hold, blocking.
bash prototype/mlp_insitu/check_r0.sh
#    R0 (thin, 48 launches) / R0f (fat, the published shape) / R0b (thin, 8 tokens per sync) /
#    R1 (drain each). It prints T_geo, T_pipe, T_serial, and it refuses to pass if R0f is >3 % off 2.275.

# 3. THE window, only if step 2 passed. One hold, blocking, nothing backgrounded, zero subagents.
bash prototype/mlp_insitu/ab_d2_sweep.sh          # REPS=3 ITERS=6 ROUNDS=2 by default
```

The window script is the executable form of §1–§3; §5 is what it prints and how it decides. It runs the
same four-row floor gate itself before spending a row on anything else, so step 2 is a fast preview of
gates it would otherwise re-take. Its canaries: **C1** the CPU gates pass before the lock is taken;
**C2** row `R0f` — the rig's fat arm, the published *shape* — reproduces the quoted 2.275 ms within
±3 % (the target is the same one pre-registered; only the arm being compared is now shape-matched);
**C3** `P = R7 − R0 ≥ 0.35 ms`, unchanged, evaluated in the floor gate so a trip costs six rows and not
thirty; **C4** every row must print `verify=ok` before its `ms` is used; **C5** every parsed field is
checked numeric and the row count must equal the number of rows the script asked for. `ROUNDS ≥ 2` is
enforced, because a single-pass matrix cannot show drift.

### 4a. What has already been run here — correctness only, **no timing taken**

| command | result |
|---|---|
| `bash prototype/mlp_insitu/build.sh` | builds `-O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib` |
| `prototype/mlp_insitu/mlp_insitu --selftest-geom` | `SELFTEST-GEOM ok fails=0`, `cases=76` |
| `prototype/mlp_insitu/mlp_insitu --print-cases` | `CASE-ROWS total=76`; `census.py` prints `CASES rig=76 derived_here=76 agree=1` |
| `.venv/bin/python prototype/mlp_insitu/census.py` | census below; every regex matched its expected record count |
| `prototype/mlp_insitu/mlp_insitu --selftest` (one `tools/gpu_lock.sh` hold, blocking) | **`SELFTEST ok fails=0 cases=76 children_bad=0`**, rc 0, 6.8 s wall, `util` 25 % at acquire / 6 % at release — **no timing taken** (the gate prints no `ms` and no GB/s by construction) |
| five untimed arm shapes in one lock hold: `blob3 burst`, `shipped72` + full step dose, `--if-when concurrent`, `--writes spread --write-mul 16`, `--mlp 0`, each `--reps 1 --iters 1` | **`verify=ok` on all five**, `NO-TIMING` printed, block wall 1.26 s — **no timing taken** |
| `bash prototype/mlp_insitu/ab_d2_sweep.sh` against a **stub** rig (`RIG=/tmp/stub_mlp_insitu.sh`) | plumbing only: dose parsing, row-count assertion, canaries, report block, lock release all exercised end to end. The stub prints fabricated numbers; no figure from that run means anything. |

Census figures the interferer doses are sized from (parsed, never typed): `mlp` reads **504.000 MiB =
528.482 MB**/token and writes **0.221 MB**; whole step **1532.311 MB read / 24.409 MB written, 62.8 : 1**;
non-MLP **1003.485 MB read / 24.188 MB written**; `STEPMIX read_mib_per_layer=19.627
write_mib_per_layer=0.961 head_mib=485.947 layers=24`.

`.gitignore` gained one line, `prototype/mlp_insitu/mlp_insitu` — the binary is named after its directory,
so the path is written in full (the recorded trap: a `prototype/**/mlp_insitu` pattern matches the
directory and hides the sources). It is appended at the end, so no `.gitignore:NN` citation moves.

### 4b. Re-run after the three fixes — correctness only, **no timing taken**

| command | result |
|---|---|
| `--selftest-geom` | `SELFTEST-GEOM ok fails=0`, with the new checks: fat bills identical read/write bytes to thin (504.000 / 0.211 MiB), `thin = 48 launches`, `fat = 2`, fat grids **10,752 / 3,072**, thin grids 448/128, fat refused on `blob72` and on `--writes spread`, and the derived must-land set non-empty and correct for all three `--kernels` values |
| `--selftest` (one lock hold) | **`SELFTEST ok fails=0 cases=76 children_bad=0`** — unchanged after the per-layer `exact_expect` change, so the witness still adjudicates both geometries — **no timing taken** |
| eight untimed arms, one lock hold, 7.96 s total: `fat both`, `fat gateup`, `thin gateup`, `thin down`, `thin both + --writes spread --write-mul 16`, `--token-burst 8`, `--mlp 0`, `--selftest` | every MLP arm **`verify=ok`**; `VERIFY land=` lines show the derived set (`act-no-delta` for gateup-only, `+delta` only for down-only, `act+delta` for both) and the geometry-sized element counts (`86016,24576` fat vs `3584,1024` thin); `--mlp 0` prints **`no_launch=1` and no number**. **no timing taken** |
| `--geometry fat --layout blob72` | refuses: `ABORT: --geometry fat requires --layout blob3 …` — the contiguity premise is checked, not assumed |
| `ab_d2_sweep.sh` / `check_r0.sh` against a **stub** rig | plumbing re-exercised after the edits. With the first window's real numbers fed to the stub, the sweep aborts in the floor gate on C3 (`P = 0.3233 ms < 0.35`) and prints the §0(iv) reading rather than a share table. Fabricated numbers; nothing measured. |

Two rig bugs found by the fat arm on device are worth keeping in the same list as §2.5: a geometry that
changes buffer sizes while the witness keeps reading the old size **passes vacuously** (the fat `act`
check read past a 7 KB buffer into warm-filled bytes that were not the unwritten sentinel), and a
`--blocks-cap` silently applied to a shape it was never meant for would have made the fat arm a thin arm
wearing a label. Both are now asserted in `--selftest-geom` rather than noticed later.

## 5. (written after the window) the attribution table

*Intentionally empty until the main agent's window lands.* On red, this section says
`NO-DIAGNOSIS` and which gate went red; on green, it gets one row per named term, its two source rows,
its share of `P`, and the residue, with the pre-registered lines quoted unchanged.

---

## 5. The window's answer (2026-10-21, main agent): the premise is refuted, and the term that is left is geometry

Taken by the main agent, blocking, zero subagents, one `tools/gpu_lock.sh` hold (`d2-check-r0`), util
89 % at acquire, discounted per `AGENTS.md`. Command: `bash prototype/mlp_insitu/check_r0.sh` (median of
3 reps x 6 tokens per row). The full matrix (`ab_d2_sweep.sh`) was **not** run, and the reason is
arithmetic rather than impatience: see the residue line below.

| row | arm | median ms | what it isolates |
|---|---|---|---|
| R0 | `blob3 --drain burst`, thin (48 launches) | 3.0545 | the shipped MLP shape alone |
| R0f | same, **fat** (2 launches) | **2.2980** | the published floor, shape-matched |
| R0b | thin, `--token-burst 8` | 3.0061 | whether pipelining tokens matters |
| R1 | thin, `--drain each` | 3.3254 | per-segment drain |

**C2 PASS: R0f reproduces the quoted 2.275 ms at +1.0 %.** So the published floor is real and
reproducible — but only at a shape the engine never emits. Per
`progress/bench/evidence/ledger-2026-10-16/commands-and-provenance.md` §6 → `mlp_gemv.hip.cpp:166-180,219`,
that row is one layer's tensor **replicated 24x and concatenated**: 2 launches, gateup 86,016 rows /
10,752 blocks, down 24,576 / 3,072. The engine's shipped walk is 48 launches at 3,584 / 1,024 rows
(448 / 128 blocks). The route's "1.11 ms penalty on a kernel at 98 % alone" has been comparing **two
different shapes**, not one kernel alone versus in situ.

The budget, with the gap named as its own term instead of being attributed to memory state:

| term | value | share of the quoted 1.0878 ms |
|---|---|---|
| `T_geo = R0 - R0f` | **+0.7565 ms** | **69.5 %** — launch shape |
| `T_serial = R1 - R0` | **+0.2709 ms** | 24.9 % — per-segment drain; inside the pre-registered 0.25-0.45 band, and an independent cross-check of the ledger's `segments x 7.2 us` = 0.346 ms |
| `T_pipe = R0 - R0b` | +0.0484 ms | 4.4 % — token pipelining is not the mechanism |
| residue for memory state | **~0.06 ms** | 5.5 % |

**Consequence, stated as the route's own verdict.** `KILL >= 3.20 ms` and `WIN <= 2.90 ms` were lines for
a kernel-and-memory prize; ~0.06 ms is available from memory state at all, which is under the rig's own
resolution and 13x under the route's -0.80 ms prize. So: **D2 as written is dead, and no code should be
funded against it.** Spending the matrix to decompose the residue would have been paying for a term that
cannot exceed 0.06 ms, which is why it was not run.

**Where the 0.7565 ms actually is, and it is not a free prize.** The fat shape merges 24 layers' MLP into
two launches, which the decode step cannot do: layer *i*'s `act` must exist before its `down`, and `down`
feeds the residual that layer *i+1* reads. So the dependency chain forbids harvesting T_geo by merging.
What it does say is where to look instead: at the shipped shape `down_proj` runs 1,024 rows across 128
blocks on 40 CUs — roughly three blocks per CU — so a one-token MLP is **under-parallelised per launch**,
and that is a split-K / row-mapping question, not a memory-state one. That is a new route line, not a
re-labelled D2, and it needs its own pre-registered lines before anything is written.

Two honest caveats on the fat arm. Its unique bytes are 1/24 of its demanded bytes (one layer replicated),
so if replication bought L2 residency it would read *above* the plateau; it reads 219 GB/s demanded, at
the plateau, so no cache advantage is detectable and the comparison stands. And it allocates weights as
three blobs, so it is a bound, not a shippable configuration.
