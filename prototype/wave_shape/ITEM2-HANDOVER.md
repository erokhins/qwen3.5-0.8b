# Item 2 — pricing launch **shape**, separately from launch **count**: rig handover

Route S1 item (2). Prototype only: nothing outside `prototype/wave_shape/` was touched, nothing is
committed, and this builder took **no timings** (see §7 for the device holds that were taken and what
they printed).

---

## 0. The defect this rig exists to repair

S1 gate zero quoted `13.08 us/launch` from a comparison of **340 launches at `grid_max=271`** against
**48 launches at `grid_max=1918`**. Launch count and workgroups-per-launch moved together, so the
number described neither. Item (1) already priced **count**, cleanly, by deletion: **3.10 µs/launch**
[Q1]. The open question is the *shape* term:

> Is a launch expensive because it is a launch, or because a one-row launch does not carry enough
> blocks / wavefronts to hide latency?

This rig answers it by varying `L` (launches per step) and `G` (workgroups per launch) **independently
at fixed total bytes**, and by giving the one-row `down_proj` shape a real split-K arm.

---

## 1. Source facts, reproduced from source (not assumed)

| fact | value | where it comes from |
|---|---|---|
| one-token `down_proj` output rows | **1,024** (`kHidden`) | `engine/decode_step.cpp` → `kernels::mlp_down(..., kHidden, kIntermediate, n)` |
| its reduction length `K` | **3,584** (`kIntermediate`) | same call |
| rows per block, shipped form | **8** | `ROWS_PER_BLOCK` in the shipped `mlp_down` |
| ⇒ grid of the shipped one-row launch | **128 blocks × 256 threads**, one warp per row, 448 `uint4` words per row | 1,024 / 8 |
| is the shape "3,584 rows"? | **No.** 3,584 is `down_proj`'s `K` and `gate/up_proj`'s row count | the two numbers were conflated upstream |
| shipping activation dtype | **F32** (`cast_fold = false` is the default); the BF16 twin is carried as a pair case | `engine/` fold flag; `pair_f32_eq_bf16act` |
| machine | **40 CUs** (`simd_count 80 / simd_per_cu 2`), wavefront **32**, 16 waves/SIMD, 64 KiB LDS | KFD sysfs; HIP reports **20** CUs on this SKU (AGENTS trap), so every `wavefronts_per_cu` here uses 40 |

`--print-device` prints both the KFD and the HIP numbers side by side with that note attached.

---

## 2. Quoted figures (labelled, not measured here)

| id | value | source |
|---|---|---|
| [Q1] | 3.10 µs/launch | cast-fold deletion, `prototype/launch_price/S1-GATEZERO.md` |
| [Q2] | step read census 1,532.311 MB, write census 24.409 MB | `engine/performance-status.md` census rows |
| [Q3] | decode step 10.9357 ms/token at 4,032 lanes | `engine/performance-status.md` §1 |
| [Q4] | `down_proj` = 1,024 rows over 128 blocks | §1 above |
| [Q5] | 40 CUs, wave 32, 64 KiB LDS, 16 waves/SIMD | KFD, §1 above |

The rig's fixed traffic is **1,464.0000 MiB read + 22.8750 MiB write** = one decode step's census to
**+0.183 %** on the read and **−1.732 %** on the write ([Q2]). Both gaps are printed on `PLAN_TOTALS`
(`dev_read_pct`, `dev_write_pct`) rather than being assumed away, and both spellings of the plan —
`wave_shape --print-plan` and `plan_table.py` — derive them from the same quoted MB constants with the
same word rounding, and agree line for line: `PLAN-DIFF ok lines_compared=205 mismatches=0`.

---

## 3. Arm family A — the `(L, G)` matrix at fixed bytes

Every cell reads `191,889,408` words and writes `2,998,272`, identical in **every** cell. A cell is
accepted only if `L*G` divides both totals exactly, so there is no rounding anywhere; **99** cells are
reachable of the 192 listed (both totals are `2^14·3·61` words, which is why e.g. `L=48 G=24` is not).
Per block the booked work is `u_r = tot_read/(L·G)` read words and `u_w = tot_write/(L·G)` write words.

Read the matrix three ways, and the point of the design is that these are *different* axes:

* **fixed `L`, varying `G`** — the **shape term**, equal bytes and equal launch count. This is item (2).
* **fixed `G`, varying `L`** — the count term at fixed bytes (a cross-check on [Q1], which came from a
  different mechanism: deletion inside the engine).
* **iso-population diagonals**, constant `L·G` — does the machine care how a fixed workgroup population
  is spread over launches at all. The diagonals include `G = 1, 2, 4` (`1024×1`, `512×2`, `256×4`), the
  one-block-per-launch end, which is what a one-row launch actually looks like.

**Every cell's bytes are witnessed on device**, not booked from the argv: an exact read checksum per
block, a per-`(launch, block)` record (read sum + count, write sum + count, launch tag), a write-landing
read-back checked at the owner's offsets, and poison guard bands on every region. Negative controls are
**derived from the geometry** (`block_short`, `range_short`, `row_double`, `alias_write`, `lie_write`,
`tag_const`) plus one control that is **provably blind** by construction (`permute_reads`: a sum is
invariant to permutation, so it must *not* be caught and the rig asserts that it isn't).

---

## 4. Arm family B — split-K for the one-row `down_proj` shape

| arm | launches | grid | warps | what it is |
|---|---|---|---|---|
| `down\|ship\|f32` | 1 | 128×1 | 1,024 | transcription of the shipped form (8 rows/block, 1 warp/row) |
| `down\|intra{1,2,4,8}` | **1** | 128·S ×1 | 1,024·S | K split **inside one launch** (wpr warps per row) — pure shape change, **zero extra launches** |
| `down\|two{2,4,7,8,14}` | **2** | 128×S | 1,024·S + 32 | K split across launches with **real two-pass F32 partials** + one reduce launch. No atomics. |

Bit-level claims, all from `--selftest` (device against a CPU mirror that models the arm's own reduction
order, and device against the shipped device output):

* `pair_intra1_eq_ship` **EQUAL, ndiff=0** — the degenerate split reproduces the shipped form bit for
  bit, which is also the witness that `down_ship` is a faithful transcription.
* `pair_intra4_eq_two4`, `pair_intra8_eq_two8` **EQUAL, ndiff=0** — the *launch structure* is
  numerically free: same association ⇒ same bits, whether the split rides `gridDim` or a second launch.
* `pair_f32_eq_bf16act` **EQUAL, ndiff=0** — on a BF16-representable activation (which is what the
  engine actually has: `sc.mlp_act` is BF16 and the F32 path only widens it) the cast-fold twin is
  bit-identical, which is the claim `mlp_xbf16_kernels.h` makes.
* **Saying it plainly: `S > 1` cannot be bit-identical to the shipped form, because split-K
  reassociates the dot.** Measured on this fixture, device vs device:

  | arm | rows of 1,024 that differ from shipped | max BF16 ULP | rows over 1 ULP | mean rel. diff |
  |---|---|---|---|---|
  | `intra2` / `two2` | 2 | 4 | 1 | < 1e-6 |
  | `intra4` / `two4` | 2 | 48 | 1 | < 1e-6 |
  | `intra8` / `two8` | 1 | 16 | 1 | < 1e-6 |
  | `two14` | 2 | 8 | 1 | < 1e-6 |

  The largest move (48 ULP) sits at a row whose dot nearly cancels — that is the cancellation band the
  `mlp_gemv` floor probe already documented, not a large error: the aggregate `meanrel` is below 1e-6 in
  every case. Whether 1–2 rows of 1,024 moving by ≤48 ULP is acceptable is the **gates'** question, which
  the standing numeric-format rule makes the decision procedure; it is not decided here.

**Two costs the timing has to weigh, both printed on every down line:**

* the **launch tax** of the two-pass form is **+1 launch** (the split rides `gridDim.y`, the reduce is
  one more launch), i.e. **3.10 µs** [Q1] = 0.0031 ms of a 10.9357 ms step [Q3]. *Correction to an
  earlier draft of this note: the tax is one extra launch, not `S` extra.*
* the **staging tax** of *any* split: every block stages the whole activation row, so demanded staging
  scales with the split — `stage_kib` 1,792 KiB shipped → 3,584 / 7,168 / **14,336** KiB at `S`=2/4/8,
  against **7,168 KiB** of weights for the whole call. A split can therefore lose on traffic while
  winning on latency; `fit_shape.py` prints `stage_kib` next to every gain and refuses to let the two
  families be compared as equal-footprint items. (Per-block declared LDS stays ≤ 14 KiB, well under the
  64 KiB/CU ceiling [Q5], so every arm here is reachable.)

---

## 5. PRE-REGISTRATION — written before any timing was taken

Bands, per the brief, unchanged:

* **WIN** — a shape change worth **≥ 0.30 ms on a 4,032-lane step**, measured at **equal bytes and equal
  launch count** (i.e. the spread across a fixed-`L` row of the matrix, or a down arm's gain over
  `down|ship|f32` at launch count 1). 0.30 ms is 2.7 % of the quoted step [Q3].
* **KILL** — the iso-population diagonal is **flat within the measured noise floor** *and* the best
  split-K gain is **< 0.15 ms**.
* Anything else is **INCONCLUSIVE**, which is a result and not a failure.

Definitions fixed in advance, so nobody has to improvise them after seeing a number:

1. **noise floor** = the largest within-arm repeat spread across two full passes taken **inside one lock
   hold** (`sweep.sh` runs exactly two passes for this reason). "Flat" means a diagonal's spread ≤ that
   floor, per diagonal, all of them.
2. **ADJUDICATION-FAILED** is reported (not "flat") when the noise floor alone is ≥ 0.30 ms: then the
   window cannot carry either band and the matrix is re-taken, not averaged.
3. A gain must beat its own tax to count: the two-pass gain is read against +1 launch = 3.10 µs [Q1].
4. **Attribution rule.** If a fixed-`L` row's winner is the cell with the **fewest wavefronts per CU**,
   that is reported as a *population* effect (less parallelism is faster here), not as evidence about
   per-launch cost; conversely if the winner is the cell with the most. The fitter prints
   `wf/cu` on every row so this is read, not remembered.
5. No cell may be quoted unless its `verify=ok` is on the line; the rig prints `NO-TIMING` instead, and
   the fitter refuses a file containing one.

---

## 6. The device pass — main agent only, in this order

```bash
export LC_ALL=C                       # 1. correctness gate first (already green; ~16 s, no timing)
./prototype/wave_shape/check_wave_shape.sh

# 2. the timed matrix: one blocking call, one lock hold, two passes, never backgrounded,
#    zero subagents active. REPS/ITERS are best-of; the defaults are 5x3.
REPS=5 ITERS=3 ./prototype/wave_shape/sweep.sh          # ~2-4 min; writes prototype/wave_shape/out/

# 3. the only reader allowed to state a verdict (37 = 27 distinct matrix cells + 10 down arms;
#    sweep.sh prints the number it used in its last line — use that if the cell list is edited)
python3 prototype/wave_shape/fit_shape.py \
        prototype/wave_shape/out/cells.txt prototype/wave_shape/out/downs.txt \
        --expect 37 --json prototype/wave_shape/out/fit.json
```

`sweep.sh` prints `CELLS-PRINTED=` / `DOWNS-PRINTED=` trailers, so `--expect 37` is the count of
*distinct* cells (27 matrix + 10 down arms; every one of them divides both pinned totals, checked
against the plan rather than assumed). Before timing anything it runs four gates: `--selftest` green; a behavioural canary on
`down|ship` (`mirror=fma`, `frozen_fma=0`, `grid=128x1`, `vs_ship_ndiff=0`) because `engine/build.sh` is
not bit-reproducible here and a stale binary would time the wrong shape while every in-binary parity
field agreed with itself; a `RESULT`-line structural smoke at `reps=1 iters=1` (a shape check on the
printer, not a measurement); and a booked-footprint assert on every timed row.

If it comes back **INCONCLUSIVE**, the pre-planned levers, in order: raise `ITERS` (dispatch-bound rows
on this box are bimodal, [AGENTS]); add fixed-`L` cells at `L=48` (the axis with the most reachable
`G`); for split-K keep `S∈{4,7}` and drop the tails, since `S=2` is the least parallel arm and `S=14`
pays 112 KiB of partial traffic per call.

---

## 7. Correctness state right now (what the builder ran)

`./prototype/wave_shape/check_wave_shape.sh` → **`CHECK-WAVE-SHAPE ok checks=34 pass=34 fail=0`**,
wall 16 s, `gpu use: 16 %` at acquire, **no timing taken**. Sections: A build+geometry, B CLI refusals,
C the fitter's refusals *and* its two accepting verdicts, D device witnesses.

`./prototype/wave_shape/wave_shape --selftest` →
**`SELFTEST ok cases=33 parsed=33 children_bad=0 clean=6/6 caught=6/6 blind=1/1 down_clean=11/11
down_caught=5/5 pairs_equal=4/4`**, wall 5 s, one lock hold, **no timing taken**.

Builder's device holds, all correctness-only, one at a time, lock acquired and released in the same
blocking block: the `--selftest` runs above (5 s each), five extreme/iso-pop cells
(`1024×1`, `512×2`, `256×4`, `16×256`, `48×1024`) plus `1024×8` (2 s), three untimed arm probes
(`128×8`, `down ship`, `down two --split 4`), and the `check_wave_shape.sh` run (16 s). No burst, no
best-of-N, no `ms` figure was produced by any of them, and nothing here may enter a timing doc.

Files in `prototype/wave_shape/`: `wave_shape.hip.cpp` (the rig), `build.sh`, `sweep.sh` (**main agent
only**), `fit_shape.py` (the only verdict-writer), `plan_table.py` (independent re-derivation of the
plan, diffed against the binary), `check_result_line.py` (RESULT-line structural canary),
`check_wave_shape.sh` (this gate), `out/` (92 KB of fixture and result text — scratch), and this file.
Nothing is committed and nothing here is git-ignored, so before committing: add the built binary by
**exact path** `prototype/wave_shape/wave_shape` (a `prototype/**/wave_shape`-style line would match this
directory and hide every source file in it — the trap recorded in `.gitignore`), and drop `out/`.

---

## 8. What this rig does **not** cover

* The matrix read stream is a strided `uint64` walk over a flat buffer, **not** a GEMV. It prices
  dispatch, occupancy and per-launch overhead at a given shape; it does not price cache-resident weight
  reuse. A shape term measured here is a **floor** on what the engine can gain, not the gain.
* Every cell is one launch shape. The engine's step interleaves ~340 launches of *different* shapes;
  item (1)'s deletion experiment is the number for that mixture, and this rig deliberately isolates one
  axis instead of reproducing it.
* `down_ship` is a **transcription** of the shipped kernel (this prototype may not edit
  `prototype/mlp_gemv`, and `#include`-ing the engine would make the rig judge the engine by itself).
  `pair_intra1_eq_ship` is the witness that the transcription has not drifted; if the engine's
  `mlp_down` changes, re-transcribe.
* The guard bands, records and read checksums are deterministic (no atomics), which is why a contended
  window cannot corrupt them — but a contended window *can* corrupt the timing, and that is the fitter's
  bimodality refusal's job, not the witnesses'.
* The iso-population diagonal is only defined where two `(L,G)` pairs share `L·G`; the divisor structure
  of the pinned totals (`2^14·3·61`) means a few interesting pops have a single decomposition and are
  skipped rather than interpolated. The shipped sweep list carries **two full diagonals** — pop = 1,024
  (`1024×1, 512×2, 256×4, 128×8, 64×16, 32×32, 16×64`) and pop = 4,096 (`1024×4 … 16×256`) — seven
  decompositions each, and fixed-`L` rows of 3–8 widths for the shape term.

---

## 9. Traps this rig found (candidates for the AGENTS table)

1. **A warp reduction whose shuffle offsets start at 32 doubles every value on a wavefront-32 part**,
   even though the shuffle *mask* is genuinely 64-bit — mask and offsets are separate facts. Every matrix
   record came out at exactly 2× the booked sum. `wave_shape.hip.cpp: warp_red_u64`.
2. **A record expectation and a content expectation that use different index origins make one witness
   always green.** The kernel indexed written values from the *launch* base; the host expected the
   *block* base. Per-block records then agreed with themselves, content agreed with itself, and the two
   disagreed with each other. Share one address expression, or diff the two spellings.
   `wave_shape.hip.cpp: ws_touch` / `mverify`.
3. **A must-catch set must be sized to the output quantum.** Dropping one word of a 448-word dot moves
   the row by ~0.2 % of itself, which is below a BF16 output quantum (1/128 of an octave), so the control
   reported `NOT-CAUGHT` while the injection worked perfectly. Truncating the injected slice to a quarter
   fires on geometry rather than on luck. `wave_shape.hip.cpp: DN_TRUNC`.
4. **A negative control's expectation must be the shipped plan's value; predicting the injected value
   erases the control.** A `tag_const` special case inside the expected value made 360k changed bytes read
   as unchanged. `wave_shape.hip.cpp: mverify`.
5. **A buffer sized with the wrong element width overflows the heap and blames the next process.** The
   F32 partial vector was declared `vector<unsigned short>`; the `memcpy` of `n*4` bytes into `n*2`
   produced `malloc(): invalid size (unsorted)` in a later forked child, three cases after the culprit.
6. **`kfd_source=` printed mojibake** because `DevFacts::src` held a `const char*` into a loop-local
   `std::string`. A field that a checker parses is a field that must outlive the line.
7. **A backslash-newline inside a single-quoted `printf` format is not a line continuation** — the fitter's
   fixture rows were being cut in half, and its `REFUSE empty` was a correct diagnosis of garbage input.
   One physical line per record format. `check_wave_shape.sh`.
8. **A mirror must model the arm's dtype, not the formula.** The BF16-activation arm's CPU mirror read the
   F32 fixture, so 419 of 1,024 rows "failed" — the mirror's bug, not the kernel's. The engine's
   activation *is* BF16, so the fixture must be BF16-representable for the cast-fold twin to mean anything.
