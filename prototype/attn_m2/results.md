# `prototype/attn_m2` — results

Status: **bit-identity green on device (§2), and every device number here belongs to the main agent** — one
window was run under lock `attn-m2` on 2026-10-16 and its numbers are in §2/§3/§4. What that window could
not reach, it could not reach because of a launcher bug (§2) rather than a limit (§6), and the four plans it
blocked are still `NOT MEASURED`, along with the above-the-knee waves §7 says are the only ones that can
price the funded claim. A worker may not take timings (AGENTS.md), so nothing new in this file is a device
number — the corrections are to mechanism and to what the existing numbers mean.

---

## §1 CPU gates — reproducible, no device, no lock

Each line is the command and the outcome it prints. Re-run them; do not take this table on trust.

| command | outcome |
|---|---|
| `.venv/bin/python prototype/attn_m2/m2_model.py selftest` | 23 checks, 0 failures |
| `.venv/bin/python prototype/attn_m2/m2_model.py guard` | the refusal guard (§6): 13,744 geometries, 0 failures |
| `.venv/bin/python prototype/attn_m2/m2_model.py compare --anchor wide1024f` | 13 cases, **0 differing elements in every case**, 0 poison reads, 0 provenance violations, 0 partial-slot aliases |
| `.venv/bin/python prototype/attn_m2/m2_model.py negative --anchor wide1024f` | 9 injections × 13 cases, 0 failures against derived must-catch/must-not-catch sets |
| `.venv/bin/python prototype/attn_m2/m2_model.py traffic` | byte model; reproduces the floors the funding doc quotes (`progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md` §3.4: 27.2 / 108.8 / 435.0 ms) and its §6 M = 2 prizes 13.6 / 49.9 / 194.6 ms |
| `.venv/bin/python prototype/attn_m2/m2_model.py regime` | which regime each wave was measured in, and the crossover (§7) |
| `./prototype/attn_m2/attn_m2 --geom-guard` | the C++ spelling of the guard: same 13,744 geometries, same numbers, host-only |
| `.venv/bin/python prototype/attn_m2/run.py plans` | 12 device plans written, no device touched |
| `.venv/bin/python prototype/attn_m2/run.py negmatrix` | which plan proves which injection, derived (§3) |
| `.venv/bin/python prototype/attn_m2/run.py selftest` | the driver's own verdict logic against `stub_harness.py`: 12 cases, 0 failures (§3) |

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
measured). Per-block census, so these are fetched lane rows with multiplicity — the mistake §20.5 of
`prototype/attn_decode/results.md` was
about is booking a re-use the loop does not have:

* 1024 lanes → 0.5024 of M = 1's K+V bytes; 2048 → 0.5012; 4096 → 0.5006. The floor halves.
* Ratio by shape: 0.778 (lanes 1–8) · 0.538 (1–64) · 0.525 (512–575, split 2) · 0.502 (513–576) ·
  0.512 (1024–1087, split 3). Short pairs and straddling pairs pay the union residue.
* V is re-used less than K whenever the two spans differ (each row's chunk *k* is anchored at its own
  `blo`, so the staged unions overlap).
* The arithmetic floor (3.53 / 14.1 / 56.5 ms, `ceiling-based-route-ranking-2026-10-16.md` §3.4) sits 7.7×
  *under* the byte floor and M = 2 does not
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

**The four refusals are a bug, not a limit, and they are the important finding of this window — with a
mechanism different from the one first written here.** `attn_m2_ok` tested
`attn_m2_pair_union(lanes_last - 1, split, 0) > ATTN_M2_STAGE`, and that function returns lane rows
**summed over the pair's lane blocks** (`total`), whereas `ATTN_M2_STAGE` bounds the **widest single lane
block** (`max_block`). A 1024-lane wave sums four ~344-lane blocks to ~1032, so every plan above ~520
*total* lanes was refused — `plan-straddles-block-2`, `plan-straddles-block-3`, `plan-pair-widest-1024`,
`plan-tail` — while every short plan passed, `pair-just-above` at exactly 520 by luck. The first diagnosis
in the brief and in this section, that `n_lo` is a row index fed a lane count, is wrong in that detail:
`n_lo` has always been the lower row's lane count, and the harness's `n_lo = c.lvalid + ATTN_M2_M * rt` is
the same quantity (`decision log D18` item 3 carries the same wrong detail; it is left as written because
journals are not rewritten). The window's arithmetic was right — at 1032 lanes with `split = 3` the slice
is 344 and the union 344 of a 520 bound — and it is now checked instead of asserted (§6). Consequence
unchanged: **the widths the route is funded for, 1024 and 4096, did not run**, so §4 is negative only over
the short end and the four REFUSED rows above have to be re-run.

## §3 Device negatives — the window's controls were vacuous, for two reasons, and the driver is rebuilt

The window ran `run.py negative --neg X`, which iterated every plan and printed **one summary line for the
last plan only**. With §2's refusals the only plans that executed were the short ones, so eight of the nine
injections reported "0 moved". Two independent defects made that number meaningless; both are now
structurally impossible rather than watched for.

1. **Empty derived must-catch sets could be run at all.** `partrow` and `partalias` alias a SPLIT partial,
   which only a *streaming* row writes; a plan whose rows are 1–8 lanes has no streaming row, so the set is
   empty and "nothing moved" proves nothing. `run.py` now computes `m2_model.neg_expect` per injection per
   plan and runs an injection **only on plans whose derived set is non-empty**; if no plan qualifies it
   stops with an assertion before touching the device — an empty list cannot print PASS. `run.py negmatrix`
   prints the whole matrix, derived: at 1–8 lanes `partrow`/`partalias` score 0 and are excluded; a lone
   1032-lane row's `maskwide` set is 0, so `plan-tail` is excluded from that injection; all nine have at
   least one plan.
2. **The baseline was wrong.** "Moved" was measured as *form vs form* under
   the injection. For a **form-neutral** injection — `readpast`, `dropcurrent`, `dimhalf`, `gqaoff` — both
   forms do the same wrong thing, so they agree and the count is **zero however far the bug spread**. Those
   four could never have registered on device at any geometry. Negative mode now runs each plan **twice**
   (`--neg none`, then `--neg X`) and compares each form's injected output against **that same form's clean
   output**, per `(row, head)`, from the dumps the harness already writes.
3. **…and the fix for defect 2 brought in a third: the forms-agree assertion was applied to every class.**
   Found on device in the next window. `if disagree: problems.append("forms disagree")` fired for the three
   `NEG_M2_*` ids as well — and there the shipped form is the **control group**: it does not implement
   `maskwide`/`partrow`/`partalias`, it is *required* not to move (which the driver already asserted as
   `moved_ship == 0`), so the two forms disagreeing is the definition of the injection reaching the
   arithmetic. Those three could therefore never print PASS; the device read
   `must=24 moved_m2=24 moved_ship=0 blind_hits=0 verdict=FAIL` over a control that had worked. The
   `mixed_bw` trap one level over — a check that fails for every configuration catches nothing — and the
   same shape as §20.6's `partalias` lesson: an assertion must be scoped to the geometry it can adjudicate.

   The assertion now runs only for the classes named in `run.py:FORMS_AGREE_NEG` (the four shared ones plus
   the `attn_prep` cache pair, where both forms really do the same wrong thing and §1.2's claim is what is
   being re-asserted). For the m2-only class the positive assertions are `moved_ship == 0` and — because a
   control group that is not clean proves nothing — that the **clean** run of the same plan is
   bit-identical, which negative mode had never checked and now requires.

Per plan the driver prints one line with its own verdict, and the two new fields say which rule was applied:

```
NEGVERDICT plan=pair-just-above neg=partrow class=m2-only rows=8 lanes=513-520 split=2 must=64 \
           moved_m2=64 moved_ship=0 blind_hits=0 forms_agree=no \
           forms_check=control-group-must-not-move verdict=PASS
```

`forms_agree=no` beside `verdict=PASS` is the correct answer for `class=m2-only` and was impossible before
this fix; for `class=shared` a PASS still requires `forms_agree=yes`. A plan the launcher refuses prints
`verdict=NO-LAUNCH` and counts as a **failure**, never a skip — the
one check that would have surfaced §2's bug inside the window itself.

**How the driver itself is checked, without a device — now committed.** `run.py selftest` drives the real
driver end-to-end against `stub_harness.py` through `--bin`: a stub with the same CLI, the same `RESULT case=`
line and the same dump layout, which moves `(row, head)` blocks exactly as its `ATTN_M2_STUB` instruction
says and has no attention, no lanes and no geometry of its own. Twelve cases, 0 failures:

| case | instruction | the verdict the driver must produce |
|---|---|---|
| `maskwide` moves exactly the derived rows, ship does not move | `m2=1,2,3` | **PASS** — the case that could not exist before the fix |
| `partrow` moves an all-rows derived set | `m2=all` | **PASS** |
| the clean run is already not bit-identical | `m2=all;clean_diff=1` | FAIL — "CLEAN run is not bit-identical", no control group |
| the m2 form moves nothing | `m2=none;clean_diff=1` | FAIL — must-catch rows did not move |
| `maskwide` moves a derived-blind row | `m2=all` on `pair-at-maxsc` | FAIL — `blind_hits=8` |
| the shipped form moves under an m2-only id | `m2=all;ship=all` | FAIL — "shipped form moved" |
| form-neutral injection, both forms move alike | `m2=all;ship=all` | PASS |
| cache injection (`slotk`), both forms move | `m2=all;ship=all` | PASS — and it witnesses a third latent bug in the same dispatch: the `DEVICE_ONLY` branch never bound `missing`, so `--neg slotk` died with a `NameError` instead of printing a verdict |
| form-neutral injection, the two forms' wrongs differ | `m2=all;ship=all;form_delta=1` | FAIL — "forms disagree under a shared injection" |
| `readpast` moves nothing | (no movement) | FAIL — must-catch rows did not move |
| the launcher refuses, no `RESULT` line | `no_result=1` | FAIL — `verdict=NO-LAUNCH` |
| `partrow` on the 1–8-lane plan | `m2=all` | ASSERTION FAIL before any binary runs — EMPTY derived set |

The middle rows are what stop the fix being "delete the agreement assertion": the form-neutral case with
`form_delta=1` fails **only** on that assertion, and the m2-only case with `clean_diff=1` fails **only** on
the control-run identity that replaced it. The expectation lives in the selftest, not in the stub — where a
spec says `must`, the selftest expands it into that plan's derived rows and tells the stub to move exactly
those — so the stub stays a bit-mover with no opinion and what the table adjudicates is the driver's rule.
Whether the derived sets are *right* is a different question, witnessed by `m2_model.py negative` against
the Python mirror; the stub cannot see geometry and does not try to answer it. In none of these can an empty
list print PASS.

| injection | window's observation | admissible then? | what the re-run must show |
|---|---|---|---|
| `maskwide` | moved 1 case | yes — the one a short plan can reach | m2 moves exactly the widened rows, ship moves 0 heads, and the two forms disagree **by design** |
| `partrow`, `partalias` | moved 0 cases | **no** — empty derived set at 1–8 lanes | must move on `straddles-block-2/3`, `pair-widest-1024`, `tail` |
| `readpast`, `dropcurrent`, `dimhalf`, `gqaoff` | moved 0 cases | **no** — wrong baseline, not wrong geometry | each form moves its own derived set against its own clean output |
| `slotk`, `slotv` | moved 0 cases | **no** — wrong baseline, and no mirror model | both forms move: the shared stage must not launder a cache perturbation |

Until that re-run, the CPU mirror's derived sets (§1.2) are the only evidence, and §2's re-run is a
prerequisite for most of them.

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
bytes demanded down ~2×. The first reading here — "the occupancy term from §1.5 arriving as predicted" —
does not survive §7: all four of these waves are **16–32 blocks**, and this box's read bandwidth only
reaches its 233–240 GB/s plateau **from ~80 blocks up**, so in every one of them M = 2's halved grid
cancels its halved bytes by construction. What the numbers say is narrower than they looked: below the
bandwidth knee this route loses ~9 %, and §7 shows where it should win. It is **not** a verdict on the
funded claim, which lives at 1024 and 4096 lanes on waves of tens of rows, where the byte model books
13.6 ms and 194.3 ms and where §2's refusal bug blocked the run. Spread across reps was under ~3 %; the
pair is quoted, not a single number from each side.

Realized blocks/CU and wavefronts per CU from a profiler: NOT MEASURED (the compile census says
VGPR 50, SGPR 64, scratch 0, LDS 22,656 B → 2 blocks/CU).

## §5 Decision rule for shipping this

1. Any non-zero `ndiff` on §2 → not admissible, fix the kernel (do not touch the gate).
2. Bit-identity green but the paired timing flat or negative → **first ask which regime was
   measured** (§7). At 16–32 blocks the grid is below the bandwidth plateau, where M = 2's halved grid
   cancels its halved bytes by construction and the result is not evidence about the route. Only a loss
   measured above the knee is a loss; the named levers (aliasing `s_k`/`s_v`, a 2-row/4-warp block) move
   occupancy, and §7 argues occupancy is not the binding term.
3. Timing wins at 4096 but loses at ≤ 1024 → ship it behind the same kind of width gate the engine already
   uses for the tile form, and say so by lane count, not by benchmark-after-the-fact.

---

## §6 The refusal guard — the bound is now something that runs

`attn_m2_ok` refuses on a prediction made from `lanes_last` alone. Before this fix that prediction used the
wrong aggregate, and the failure was silent: four of eight funded plans simply did not launch. A comment
saying "the refusal is implied by the split check" is a note, not a forcing, so the bound is now walked.
Two spellings, same numbers, no device:

```
.venv/bin/python prototype/attn_m2/m2_model.py guard        # CPU mirror, no build needed
./prototype/attn_m2/attn_m2 --geom-guard                    # the C++ closed forms, host-only
```

Both print: **13,744 geometries walked** (split 1..8 × full 0/1 × nrows 1/2/3/4/8/17 over a lane census
dense below 64, dense over the MAXSC boundary, every multiple of MAXSC ± 2, and a sweep to the engine's
4096-lane ceiling), and four assertions:

| assertion | result |
|---|---|
| **BOUND** — no legal launch stages a union wider than `ATTN_M2_STAGE` or a slice wider than MAXSC | worst single-block union **518 of 520** (at `lanes_last=4094 split=8 full=0`), worst row slice **512 of 512** |
| **NO PHANTOM** — the launcher refuses no launch the kernel would run happily | **0 phantom refusals** |
| **NO LEAK** — the launcher accepts no launch the kernel would have to poison | **0 leaks** |
| **REAL PLANS** — the 12 geometries this op actually runs are accepted | all 12 ok, read out of the header's own table |

Three things worth stating plainly:

* **The bound is not exceeded at any real width.** The headroom is 2 lanes (518 of 520) and the derivation
  `span_max + (split − 1) = 512 + 7 = 519` is met, so `ATTN_M2_STAGE = 520` is honest, not generous. That
  answers the question the brief asked: there is **no real width where M = 2's union bound fails**, so M = 2
  is not limited by staging.
* Geometries with a grid shallower than `split_for_lanes` do exceed the bound (e.g. `split=7` at 4096 lanes
  gives a 586-lane slice) and are correctly refused — but by the **split** check. The guard asserts which
  reason fired, so a union refusal can never again be the accidental answer for a too-shallow grid.
* The `M2_REAL` table lives in the header and the Python guard **parses it out of the header text**, so
  there is one table, not two that can drift. A plan that starts being refused fails the guard, in seconds,
  before it can cost a device window.

`m2_model.py selftest` now carries these as six of its checks, so the guard cannot be skipped by running
only `compare`.

## §7 Where M = 2 pays: a block-count crossover, not a lane-count one

`.venv/bin/python prototype/attn_m2/m2_model.py regime` prints this section's arithmetic from the census,
with §4's ms numbers as quoted inputs.

**The measured plateau.** `devices/strix-halo-memory-bandwidth.md`: a grid-stride read sweep over 20–640
blocks × 128–1024 threads reaches **233–240 GB/s and plateaus from ~80 blocks up**. Below ~80 blocks the
achieved bandwidth scales with the grid, because the grid is what produces outstanding loads.

**What M = 2 does to the grid.** `blocks_ship = N_KV × split × nrows`, `blocks_m2 = N_KV × split ×
⌈nrows/2⌉`. M = 2 halves the grid at the same time as it halves the bytes. Below the knee those two halves
cancel — that is not a defect in the kernel, it is arithmetic on a device fact.

| wave | blocks ship → m2 | above the knee? | achieved GB/s ship → m2 | traffic share of ship's time | bytes ratio β | speedup |
|---|---|---|---|---|---|---|
| `pair-at-maxsc` | 16 → 8 | **no** | 14.2 → 8.4 | 6.0 % | 0.635 | 0.932 |
| `maxsc-minus-1` | 16 → 8 | **no** | 14.2 → 6.5 | 6.0 % | 0.502 | 0.910 |
| `pair-just-above` | 32 → 16 | **no** | 62.8 → 28.4 | 26.5 % | 0.502 | 0.902 |
| `pair-shortest` | 16 → 8 | **no** | 8.7 → 6.1 | 3.7 % | 0.778 | 0.904 |

All four are 2.5–5× below the knee and 4–27× above their own DRAM floor, so none of them is a wave whose
time *is* its traffic — which is the only regime where the funded saving exists. Note also that the loss is
~0.90 in all four while β ranges 0.50–0.78 and traffic share ranges 3.7–26.5 %: the loss does not track the
traffic term at all, which is what the plateau explanation predicts.

**Above the knee** (census, floor at 237 GB/s — **NOT MEASURED**):

| wave | blocks ship → m2 | ship MB | floor ship | floor m2 | β | speedup if both sit at the plateau |
|---|---|---|---|---|---|---|
| 64 rows at 1021–1084 lanes (split 3) | 384 → 192 | 138.0 | 0.582 ms | 0.298 ms | 0.512 | **1.95** |
| 64 rows at 2045–2108 lanes (split 5) | 640 → 320 | 272.2 | 1.148 ms | 0.591 ms | 0.515 | **1.94** |
| 64 rows at 4033–4096 lanes (split 8) | 1024 → 512 | 532.7 | 2.248 ms | 1.125 ms | 0.500 | **2.00** |
| 16 rows at 4081–4096 lanes (split 8) | 256 → 128 | 134.0 | 0.565 ms | 0.283 ms | 0.500 | **2.00** |

**The crossover, stated as a condition.** M = 2 pays where `blocks_m2 ≥ ~80` *and* the shipped wave is at
the plateau, i.e. **40 rows at split 2, 20 rows at split 4, 10 rows at split 8**. The engine's own prefill
waves are 64 rows, so at every funded width the engine is above the knee and the prize arithmetic is intact
there — **but it is unmeasured**, and §4's numbers neither support nor refute it. Nothing in this section is
a measurement of `attn_m2_wave`; it is the census plus the box's published plateau, and the way to falsify
it is the burst at the four waves above.

### §7.1 Do the two named levers change the crossover? No.

* **Aliasing `s_k`/`s_v`.** LDS map: `s_sc` 8,320 B + `s_k` 4,096 + `s_v` 8,192 + `s_kself`/`s_vself`
  2,048 = **22,656 B**. `s_v` is the larger stage, so an alias costs `s_k`: 22,656 → **18,560 B** →
  3 blocks/CU → **24 wavefronts/CU** instead of 16. Legal in this two-pass structure (pass 1 stages K, pass
  2 stages V, barrier between), and it is worth measuring *only if* a burst taken above the knee still
  shows a loss.
* **A 2-row/4-warp block.** LDS per block does not depend on warp count — `s_sc` is sized by
  `M × G × STAGE`, not by warps — so blocks/CU stays 2 and wavefronts/CU **halves to 8**. This lever is
  strictly worse than what it is meant to fix. Worth recording because it was proposed as a fix.
* The lever that would actually move occupancy is not `s_k`/`s_v` at all but **`s_sc` (37 % of the block's
  LDS)**: keeping each warp's 520 union scores in registers instead of shared memory takes LDS to 14,336 B
  → 4 blocks/CU → 32 wavefronts, the device's `Max Waves Per CU`. Its price is ~9–17 VGPR (50 → ~60), which
  the register census would have to re-adjudicate:
  `hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage -c attn_m2.hip.cpp`. Not
  implemented — no measurement yet asks for it.
* Why occupancy is probably not the binding term at all: the plateau sweep hit 233–240 GB/s with
  **128-thread** blocks, i.e. ~8 wavefronts/CU at ~80 blocks over 40 CUs (`devices/strix-halo-compute.md`:
  40 CUs, `rocminfo`), which is *below* this kernel's 16. So 16 wavefronts should be enough to reach the
  plateau, and the loss in §4 is explained by the block count being under the knee without invoking LDS.