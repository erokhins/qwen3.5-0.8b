# `prototype/attn_m2` — results

Status: **CPU gates green; device pass not run.** Every device cell below is `NOT MEASURED` and stays
that way until someone with an exclusive box fills it (`HANDOFF.md`). A worker may not take timings
(AGENTS.md), so this file deliberately contains no device number — not an estimate, not a "should be".

---

## §1 CPU gates — reproducible, no device, no lock

Each line is the command and the outcome it prints. Re-run them; do not take this table on trust.

| command | outcome |
|---|---|
| `.venv/bin/python prototype/attn_m2/m2_model.py selftest` | 17 checks, 0 failures |
| `.venv/bin/python prototype/attn_m2/m2_model.py compare --anchor wide1024f` | 13 cases, **0 differing elements in every case**, 0 poison reads, 0 provenance violations, 0 partial-slot aliases |
| `.venv/bin/python prototype/attn_m2/m2_model.py negative --anchor wide1024f` | 9 injections × 13 cases, 0 failures against derived must-catch/must-not-catch sets |
| `.venv/bin/python prototype/attn_m2/m2_model.py traffic` | byte model; reproduces §3.4's floors 27.2 / 108.8 / 435.0 and §6's M = 2 prizes 13.6 / 49.9 / 194.6 ms |
| `.venv/bin/python prototype/attn_m2/run.py plans` | 8 device plans written, no device touched |

### §1.1 Bit-identity per case (the admission gate)

`compare` prints, per case: lane range, split, elements differing between this form and the shipped
`attn_tile_wave` mirror, K+V lane rows fetched by each form, the ratio, poison reads and provenance
failures. The claim is **zero**, not "within band" — the funding decision required bit-identity per row on
the per-row lane order (D5 item 3), and a row that changed at all would mean the design is wrong.

| case | lanes | split | forms differ | K+V ship | K+V m2 | ratio |
|---|---|---|---|---|---|---|
| `wave000-063.L03` (pinned) | 1–64 | 1 | 0 of 131,072 | 8,320 | 4,480 | 0.538 |
| `wave064-127.L03` | 65–128 | 1 | 0 of 131,072 | 24,704 | 12,672 | 0.513 |
| `wave128-191.L03` | 129–192 | 1 | 0 of 131,072 | 41,088 | 20,864 | 0.508 |
| `wave192-255.L03` | 193–256 | 1 | 0 of 131,072 | 57,472 | 29,056 | 0.506 |
| `wave256-276.L03` | 257–277 | 1 | 0 of 43,008 | 22,428 | 11,872 | 0.529 |
| `m2.pair-at-maxsc` | 512–515 | 2 | 0 of 8,192 | 8,216 | 5,216 | **0.635** |
| `m2.maxsc-minus-1` | 511–514 | 2 | 0 of 8,192 | 8,200 | 4,120 | 0.502 |
| `m2.pair-just-above` | 513–520 | 2 | 0 of 16,384 | 16,528 | 8,304 | 0.502 |
| `m2.straddles-block-2` | 1024–1027 | 3 | 0 of 8,192 | 16,408 | 8,488 | 0.517 |
| `m2.straddles-block-3` | 1536–1539 | 4 | 0 of 8,192 | 24,600 | 12,908 | 0.525 |
| `m2.pair-widest-1024` | 1021–1024 | 2 | 0 of 8,192 | 16,360 | 8,200 | 0.501 |
| `m2.pair-shortest` | 1–8 | 1 | 0 of 16,384 | 144 | 112 | **0.778** |
| `decode7.L03(wide)` — real HF row, 1032 lanes | 1032 | 3 | 0 of 2,048 | 4,128 | 4,132 | **1.001** |

The last row is the route's limit stated plainly: **a single-row wave has no partner to share with and
costs ~0.2 %** (the self lane is staged twice). M = 2 is a wave/prefill route; decode steps keep
`attn_core`. The two bolded ratios are the two geometries where sharing pays least (§1.4).

### §1.2 The negatives, and how their sets are derived

| injection | kind | what it is | must-catch derived from |
|---|---|---|---|
| `readpast` | form-neutral | causal mask gone | rows already at `cap` are blind (cannot gain a lane) |
| `dropcurrent` | form-neutral | §05.7 append forgotten | everything moves |
| `dimhalf` | form-neutral | dot over 128 of 256 dims | a 1-lane row is blind |
| `gqaoff`, `gqa_head0` | **self-independent** | group reads the neighbour's K/V | `gqa_reach`: lanes whose bytes differ between KV groups |
| `maskwide` | **m2, range-adjudicated** | a row attends the pair's union | per (row, lane block): does the union extend past that row's own slice |
| `partrow` | m2 | the two rows of a pair write one split-partial slot | streaming rows only — a one-pass row writes no partial, so there is no slot to steal |
| `partalias` | m2 | §19's bug re-injected (slot keyed by head-in-group) | streaming rows, heads 0..G−1 only |
| `headcarry` | m2, mirror-only | row 1 seeded from row 0's `(max,sum)` | a pair's second row only — a carry cannot cross a block, and the first row has no predecessor inside its own block |

`maskwide` is adjudicated on **which lanes each row attended**, not on output bits: a widened row can
cancel to the same BF16 result (the chaos floor every bf16 gate on this box fights), so "the output moved"
is not a property of the mask. What is asserted exactly: the attended-range set equals the derived set; no
row whose lanes did *not* change may move (that is a mask leak); and every range-moved row moves at least
one head (the injection reaches the arithmetic).

`slotk`/`slotv` stay device-only: they perturb the cache in `attn_prep`, which this mirror does not model.

### §1.3 Two findings from writing the controls

1. **`gqaoff`/`gqa_head0` cannot be form-neutral bit-identity checks, and the reason is worth keeping.**
   The shipped mirror rewrites the self operand's KV group along with the group's cache reads
   (`attn_tile_model.wave_out`, the `gqaoff` branch before `self_k = ...`). The shipped kernel stages the
   self row from `kv_write` — the block's own group (`attn_decode_kernels.h:593,628,656`) — and so does
   this kernel. So the two *mirrors* disagree under those injections while the two *kernels* agree. The
   gate therefore asserts that each form's wrong lands on its own derived row set, and that stage/cache
   provenance is undisturbed, and says why the stronger check is unavailable rather than dropping it.
   This is a real inconsistency in the shipped mirror (its gqa must-catch is wider than the kernel's);
   it is recorded here, not fixed, because `prototype/attn_decode` is frozen and its own negative gate
   only ever asserted "moved", which the wider set still satisfies.
2. **A zero-filled synthetic cache makes KV-group controls vacuous.** With `wide_case`'s zeros, a wave's
   first row attends exactly one non-zero lane — its own self lane — so a group swap cannot reach it and
   a hand-written must-catch set scores 0 of everything while looking clean. `m2_model.wide_plan`
   populates the whole cache by cycling the pinned prefill cache, and `gqa_reach` derives reachability
   from the operand bytes. Same lesson as `partalias` over pinned spans, one level over: derive the case
   list from what can make the control fire, including the fixture.

### §1.4 What the byte model says, and where it stops paying

From `m2_model.py traffic` (arithmetic on the head geometry over the 237 GB/s ceiling; quoted, not
measured). Per-block census, so these are fetched lane rows with multiplicity — the mistake §20.5 was
about is booking a re-use the loop does not have:

* 1024 lanes → 0.5024 of M = 1's K+V bytes; 2048 → 0.5012; 4096 → 0.5006. The floor halves.
* Ratio by shape: 0.778 (lanes 1–8) · 0.538 (1–64) · 0.525 (512–575, split 2) · 0.502 (513–576) ·
  0.512 (1024–1087, split 3). Short pairs and straddling pairs pay the union residue.
* V is re-used less than K whenever the two spans differ (each row's chunk *k* is anchored at its own
  `blo`, so the staged unions overlap).
* The arithmetic floor (3.53 / 14.1 / 56.5 ms, §3.4) sits 7.7× *under* the byte floor and M = 2 does not
  move it at all — two rows still do two rows' worth of dots. That is where the M ≈ 7.7 crossover comes
  from, and why M = 2 is the funded step rather than M = 4 or 8.

### §1.5 Geometry facts the gates established

* Pair union worst case over every legal `(split, n)`: **519 lanes** against `ATTN_M2_STAGE = 520`. The
  bound is `span_max + (split − 1)` and it is met.
* A `split` shallower than `split_for_lanes` is refused: `attn_m2_ok` and the mirror both flag `too_wide`
  (3/3 probes at n = 513/1025/4097, split 1).
* Two independent spellings of the union agree for all 96 `(split, n, full)` combinations priced; the
  device-side rules are transcribed character for character from the shipped header
  (`m2_lanes_of`/`m2_blocks_for_row`), and the resulting lane cut equals `attn_tile_model.row_bounds`'
  cut block for block.
* Compile census, `hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage -c`:
  `attn_m2_wave` VGPR 50, SGPR 64, ScratchSize 0 B/lane, **LDS 22,656 B/block**, 256 threads. The
  compiler's `Occupancy [waves/SIMD] 16` is register-limited and ignores LDS; the LDS-derived figure is
  65536/22656 = **2 blocks/CU = 16 wavefronts**, against the shipped `attn_tile_wave`'s 12,288 B × 4
  warps → 5 blocks = 20 wavefronts (§20's table). M = 2 buys bytes and spends latency-hiding.

---

## §2 Device A/B — bit-identity

Window `attn-m2` (lock held, zero subagents, blocking, util read 91 % — util is not a gate on this box),
2026-10-16. **Before any row was compared the harness faulted**: `diff_rows` called `block_sum()`, which
contains `__syncthreads()`, from inside `if (threadIdx.x == 0)` — a barrier executed by one thread of a
256-thread block is undefined on GCN and aborts the queue with code `0x1016` rather than returning a
wrong count. Fixed (reduce uniformly, then store from thread 0); it is the same tripwire the header
names for the m2 kernel itself, and it is why "the CPU mirror is green" bought nothing here.

| plan | rows | lanes | split | ndiff `aout` | ndiff `gated` | bytes demanded ship | m2 | ratio |
|---|---|---|---|---|---|---|---|---|
| `plan-pair-at-maxsc` | 4 | 512–515 | 2 | **0** | **0** | 4,206,592 | 2,629,632 | 0.6251 |
| `plan-maxsc-minus-1` | 4 | 511–514 | 2 | **0** | **0** | 4,198,400 | 2,101,248 | 0.5005 |
| `plan-pair-just-above` | 8 | 513–520 | 2 | **0** | **0** | 8,462,336 | 4,235,264 | 0.5005 |
| `plan-pair-shortest` | 8 | 1–8 | 1 | **0** | **0** | 73,728 | 40,960 | 0.5556 |
| `plan-straddles-block-2` | 4 | 1024–1027 | 3 | REFUSED | REFUSED | — | — | — |
| `plan-straddles-block-3` | 4 | 1536–1539 | 4 | REFUSED | REFUSED | — | — | — |
| `plan-pair-widest-1024` | 4 | 1021–1024 | 2 | REFUSED | REFUSED | — | — | — |
| `plan-tail` (real HF row 1032) | 1 | 1032 | 3 | REFUSED | REFUSED | — | — | — |

Bit-identity holds on the four plans that run — including the MAXSC/MAXSC+1 straddle, which is the case
M = 2 exists to prove. The bytes-demanded ratio reproduces the byte model (~0.50 for a full pair).

**The four refusals are a bug, not a limit, and they are the important finding of this window.**
`attn_m2_ok` calls `attn_m2_pair_union(lanes_last - 1, split, 0)`, but that function's first argument is
a **row index** `n_lo` — which is how the harness itself calls it (`n_lo = c.lvalid + ATTN_M2_M * rt`).
Passing a lane count makes the closed form derive each row's lane span from a row number, so every plan
above ~520 lanes is refused with *“the union of a pair's slices exceeds ATTN_M2_STAGE”*. At 1032 lanes
and `split = 3` a slice is ~344 lanes and the union should be ~346, well inside the 520 bound. The
consequence is that **the widths the route is funded for — 1024 and 4096 — cannot be run at all**, which
is also why §4 is negative only over the short end.

## §3 Device negatives

Run with `run.py negative --neg X`, which iterates every plan; the summary line prints the **last** plan
only, and with the §2 refusals the only plans that execute are the short ones. That makes eight of the
nine injections **vacuous on device**: `partrow`, `partalias` and friends can only move a *streaming*
row, and a plan whose rows are 1–8 lanes has no streaming row, so their must-catch set is empty and
“nothing moved” proves nothing. This is the AGENTS.md trap (“a negative control with an empty must-catch
set is not a control”) arriving through plan selection rather than through the injection itself.

| injection | observed (short plans only) | admissible? |
|---|---|---|
| `maskwide` | moved 1 case | yes — the one injection whose geometry survives a short plan |
| `partrow`, `partalias`, `readpast`, `dropcurrent`, `dimhalf`, `gqaoff`, `slotk`, `slotv` | moved 0 cases | **no** — empty derived must-catch set at these widths |

Re-run after the refusal fix, when the > MAXSC plans execute; the CPU mirror's derived sets (§1.2) stand
until then.

## §4 Timing and traffic — measured, and negative where measurable

Paired inside one `attn-m2` lock hold, zero subagents, blocking, best-of-40 launches × 5 reps,
`steady_clock` (not `hipEventElapsedTime`), both forms alternating within one rep so a disturbance hits
both, one warm pass first because the first pass over fresh pages runs ~35 GB/s. Only the wave launch is
looped — `attn_prep` and `attn_combine` are identical on both sides, so timing the trio would add the
same constant to both and dilute the ratio.

| case | lanes | ship best | m2 best | speedup |
|---|---|---|---|---|
| `pair-at-maxsc` | 512–515 | 0.2953 ms | 0.3167 ms | **0.933** |
| `maxsc-minus-1` | 511–514 | 0.2949 ms | 0.3240 ms | **0.910** |
| `pair-just-above` | 513–520 | 0.1348 ms | 0.1495 ms | **0.902** |
| `pair-shortest` | 1–8 | 0.0085 ms | 0.0094 ms | **0.906** |

**M = 2 is 7–10 % slower than the shipped form at every width this harness can currently run**, with the
bytes demanded down ~2×. That is the occupancy term from §1.5 (16 wavefronts/CU against 20) arriving as
predicted and being, at these widths, larger than the traffic saving. It is **not** a verdict on the
funded claim, because the claim lives at 1024 and 4096 lanes, where the byte model books 13.6 ms and
194.3 ms and where §2's refusal bug currently blocks the run. Spread across reps was under ~3 %; the
pair is quoted, not a single number from each side.

Realized blocks/CU and wavefronts per CU from a profiler: NOT MEASURED (the compile census says
VGPR 50, SGPR 64, scratch 0, LDS 22,656 B → 2 blocks/CU).

## §5 Decision rule for shipping this

1. Any non-zero `ndiff` on §2 → not admissible, fix the kernel (do not touch the gate).
2. Bit-identity green but the paired timing flat or negative → the occupancy cost is the binding term;
   the named lever is aliasing `s_k`/`s_v` (24 wavefronts) or a 2-row/4-warp block, and the byte model in
   §1.4 says the prize is real only at ≥ 2048 lanes.
3. Timing wins at 4096 but loses at ≤ 1024 → ship it behind the same kind of width gate the engine already
   uses for the tile form, and say so by lane count, not by benchmark-after-the-fact.
