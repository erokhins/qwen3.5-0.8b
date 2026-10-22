# ISOLATED — the user's one-word test: does a packed pair's dead half move the live half?

Read `CROSSROW.md` (datum) and `FIX-CROSSROW.md` Results 0, 2, 5, 6, 7 and "What is left" first. This file
is the checkpoint for a small, cheap isolation, not a fix. **No engine file, gate, floor, pin, ledger, limits,
README or progress doc was touched.** Nothing is committed. No timing is taken or printed anywhere; the only
seconds are whole-command wall, reported per the device-run protocol.

Two claims have to be separated before anything is built, because they have been conflated:

* **the datum** — perturbing token `p` moves row `p−1`, only when `bound(p−1) = p` is odd, deterministically,
  by ≤ 1 BF16 ULP in a handful of the row's 2048 output elements (`FIX-CROSSROW` Results 1, 5). Real,
  reproduced, localised INTO the wide core by an inert dump flag.
* **the carrier** — two candidates remain, and both were built on the parity clue:
  (i) the **f16x2 P-pair** (`FIX-CROSSROW` Result 7), one live cell and one masked cell in one 32-bit word;
  (ii) **a disallowed cell's score reaching the row's `max`/`rowsum`** before the mask zeroes it — the shape
  that would produce exactly this magnitude, since softmax is scale-invariant and a last-bit change of `m`
  or `l` survives only as rounding.

This file answers (ii) from the shipped source first, then tests (i) and (ii) on a rig with real magnitudes.

---

## PART 1 — the source adjudication: NO, a disallowed cell cannot reach `mn` or `l`

The question in one sentence: **can a lane with `live == false` still influence `mn` (or the summed
denominator) for a row whose own bound excludes it?** Answer: **no, in the shipped text it cannot**, and the
reason is an ordering, not a value. All lines are `engine/attn_wide_llama_kernels.h` (knob 2, the shipping
default; knob 1 `engine/attn_wide_kernels.h:257-262` is the same law).

The masked round, in order:

| line | text | what it establishes |
|---|---|---|
| `:229` | `float rm = -1e30f;` | the round-local max starts below every finite score |
| `:235` | `const int j = base + FATTN_MLANES * L + fattn_c_col(lane, c);` | the cell's TRUE lane index |
| `:236` | `const bool live = row_ok && j < rlanes;` | the per-row bound, per cell |
| `:237` | `acc[L][c] = live ? wll_score_round(acc[L][c]) : -1e30f;` | **THE ORDERING. The accumulator cell is overwritten BEFORE anything reads it** |
| `:241` | `rm = fmaxf(rm, acc[L][c]);` | the max reads the ALREADY-MASKED cell — a disallowed cell contributes exactly `-1e30f`, which cannot raise a max that already holds `-1e30f` or any finite value |
| `:246` | `rm = fmaxf(rm, __shfl_xor_sync(FULL_MASK, rm, 16, warpSize));` | the butterfly pairs lane `t` with `t^16`. `fattn_c_row(lane) = lane & 15` (`engine/attn_fattn_kernels.h:453`) makes `t` and `t^16` the **same row-slot**, and `rlanes` comes from that same `rslot` (`:382` `rslot = fattn_c_row(lane)`, `:383` `rrow`, `:385` `row_ok`, `:391` `rlanes`), so both halves of the pair agree on the bound. It is a within-row max, not a cross-row one |
| `:247` | `const float mn = fmaxf(m, rm);` | `m` is this row-slot's own carried max from earlier rounds — never another row's |
| `:248` | `const float rescale = expf(m - mn);` | pair-uniform, per-row |
| `:260` | `const float e = expf(acc[L][c] - mn);` | reads the ALREADY-MASKED cell. A masked cell gives `expf(-1e30f - mn)`; `mn` is finite for any `row_ok` row (see the bound-argument below), so this is **exactly `+0.0f`** |
| `:261` | `ls += e;` | adding `+0.0f` to a running F32 sum is bit-exact — the denominator cannot move |
| `:267` | `l = l * rescale + ls;` | same argument |
| `:428` | `l += __shfl_xor_sync(FULL_MASK, l, 16, warpSize);` | again `t` with `t^16` — one row-slot |
| `:458` | `if (qrow_pub < nrows) { … finish_head(…) … }` | the publish guard: the only genuinely all-masked row-slots (`row_ok == false`, `rlanes = 0` forced at `:391`, which DO get `e = expf(-1e30 − (−1e30)) = 1` and a garbage `l`) are confined to their own registers and never reach memory |

**Why `mn` is finite for every real row in a masked round** (the one place the argument could fail): the
masked loop starts at `nfull * WIDE_STAGE` with `nfull = lo_bound / WIDE_STAGE` (`:379-380`), so
`base ≤ lo_bound ≤ rlanes` for every row-slot of the block. If `base < rlanes`, column `base` is live, and
column `base` is held by lane `rslot` (`fattn_c_col(lane,0) = lane>>4`) of that same row-slot, so that
row-slot's `rm` is at least that finite score. If `base == rlanes` (possible when `lo_bound` is an exact
multiple of 64), the row has nothing new to attend this round, `rm` stays `-1e30f`, and `mn = fmaxf(m, −1e30f)`
is the finite `m` carried from the full round before it — `rescale = expf(0) = 1`, `ls = 0`, row unchanged.
The only way `mn = -1e30f` is a row-slot past `nrows`, which the publish guard at `:458` blocks.

### The worked example the brief asked for: `q = 8`, bound 9, the p128 body

Geometry of the shipping prefill body (`FIX-CROSSROW` Result 0: `rows=128`, `lanes_last=128`, `lvalid=1`,
`split=1`, grid `(2,1,4)`): `WIDE_POS_PER_BLOCK = 32`, `WIDE_POS_PER_WARP = 4`, `GQA = 4`, `WIDE_STAGE = 64`,
`WIDE_LGROUPS = 4`, `FATTN_MLANES = 16`.

* Row 8 sits in the `rt` block covering positions 0..31 (`row0 = 0`), in the warp with `warp_row0 = 8`
  (`w = 2`), and in every row-slot whose `rrow = 8 + rslot/4 = 8`, i.e. `rslot ∈ {0,1,2,3}` — **lanes
  {0,1,2,3} ∪ {16,17,18,19}**.
* Rounds: `hi = lanes_of(a, 31) = 32`, `lo_bound = lanes_of(a, 0) = 1`, `nfull = 1/64 = 0` → exactly ONE
  masked round at `base = 0`. Cell lane index `j = 16L + 2c + (lane >> 4)`, `L ∈ 0..3`, `c ∈ 0..7` → columns
  0..63, of which **`live` is `j < 9`**.
* Lane 0 (low half of each pair) holds `j = 0,2,4,6,8,10,12,14` at `c = 0..7` → live for `c ≤ 4`.
  Lane 16 (high half) holds `j = 1,3,5,7,9,11,13,15` → live for `c ≤ 3`. **Lane 9's column is 9 = `2·4+1`,
  held by lane 16 at `c = 4`.**
* The max: `rm(lane 0) = max(s0,s2,s4,s6,s8, −1e30…)` and `rm(lane 16) = max(s1,s3,s5,s7, −1e30…)`; the
  butterfly at `:246` gives `mn = max(s0..s8)`. **`s9` — the future token's score — is never an operand of
  `mn`.** Its weight is `expf(-1e30f - mn) = +0.0f` exactly, so `ls` gets an exact `+0.0f` addend.
* **But `:263-264`**: the pair written at `p[0][8], p[0][9]` is one 32-bit f16x2 word, and for `c = 4` it is
  `(f16(exp(s8 − mn)), f16(+0.0f))` on lane 0 and the same word in the mirror order on lane 16. **Live
  column 8 and masked column 9 in one register.** That is Result 7's straddle, and bound 9 is odd.
* The parity check that makes this the right example: bound 8 (row 7) → live 0..7 → pairs `(0,1)(2,3)(4,5)(6,7)`
  fully live and `(8,9)` fully masked — **no straddle**. Odd bound ⇒ exactly one straddling pair per row per
  L-group; even bound ⇒ none. Same dependence set as Result 5.

**So Part 1's verdict: candidate (ii) is refuted by the shipped text — the mask is applied to the
accumulator cell (`:237`) before the max (`:241`), before the butterfly (`:246`), and before the exp/rowsum
(`:260-261`), and the butterflies are within-row. The remaining candidate is the pack, `:263-264`.**
Two honest limits on that verdict, both of which the rig below exists to close:

1. It is a *reading*. The two load-bearing sub-claims are device facts I have not measured: that
   `expf(-1e30f − mn)` is bit-exact `+0.0f` on this part for the `mn` values a real row produces, and that an
   `expf` underflow adds exactly nothing to `ls`. **Arm B measures them** and carries a must-catch that
   proves the rig would see it otherwise.
2. Part 1 says nothing about whether the *pack* can do it. `+0.0 × V` is inert in any sane lowering of
   `v_wmma_f16`, but "sane" is a claim about the hardware, and the datum is real. **Arm A measures it.**

## PART 2 — the rig (design; results appended below as they land)

`prototype/mtp_verify/pack_ab/`, three separable arms so each candidate can be implicated alone. Real
magnitudes come from a `--dump-attn-rows` capture of the p128 body, not from tidy numbers — `FIX-CROSSROW`
Result 6's 345 synthetic cases saw nothing precisely because a probe-built magnitude cannot express a
1e−7 relativity.

* **A (dot only)** — the engine's own score roll and pack, then the disallowed half of the straddling pair
  overridden to (0x0000 / its own unmasked value / a deliberately large finite), live half byte-fixed; then
  the engine's own V stage and P×V leg. Report: does the live row's output last F32 bit move, ULP, and is it
  deterministic across repeats. Witness: an arm that varies the LIVE half must move it.
* **B (max/sum path)** — the masked round's max-and-rescale on a real row's score vector, with the
  disallowed lane's **pre-mask score** varied over (real / +40 / −1e30 / huge). Report: bits of `mn`,
  `rescale`, `ls`, `l`, and the packed word. Must-catch: the hypothesis's own shape — mask applied *after*
  the max — must turn it red, or the arm is decoration.
* **C (both, engine order)** — the engine's own `attn_wide_llama_wave` launched on the real dumped tiles,
  twice, with only the disallowed lane's K and V rows differing. This is a replay, not a paraphrase, and it
  is the arm that says whether the datum reproduces outside the engine at all.

---

## PART 3 — the measurements, and what they name

Rig: `prototype/mtp_verify/pack_ab/{pack_ab.cpp,build.sh,report.py}` → `tmp/mtp_verify/pack_ab/pack_ab`
(device, correctness-only) and `report.py` (CPU, sizes every tile through the imported
`prototype/bf16_ulp.py`). Inputs are the engine's own `--dump-attn-rows` tiles from
`p128 --steps 0` (`seq=1 layer=3`, and `seq=2 layer=7` for the second layer), so every operand magnitude is
the engine's; nothing is tidied. `--attn-fattn-wide 1` was dumped separately for the knob-1 side.

**The shape, and why there is no copied round in it.** Transcribing `wll_round<true>` to poke `p[L][2c]`
would put a copy between the question and the answer (AGENTS' association row: a copy agrees with itself).
The same two questions are answerable of the SHIPPED kernel by moving the other side of the product:

* **arm B = vary only the disallowed lane's K row.** A K row is one column of the C tile, so the only thing it
  can reach is that column's **pre-mask score** — i.e. exactly the max/sum hypothesis of PART 1.
* **arm A = vary only the disallowed lane's V row.** Then the disallowed cell's weight is whatever the kernel
  wrote, untouched; if it is exactly `+0.0f` the product is `+0.0` for ANY V and the row cannot move.
  Amplifying V makes the test more sensitive, not less, so a null is not a resolution artefact.
* **arm C = vary both** — the datum, replayed.

### Falsifiability first: three things that make a null here mean something

1. **`CANARY fid` — BYTE-IDENTICAL** to the engine's own dumped `pre_gate`, all 128 rows, on every
   configuration run (`layer 3`, `layer 7`, `--knob 2`, and `--knob 1` against its own dump). The rig's
   `CoreArgs` (taken field-for-field from the run's `ATTN_DUMP` line: `cap=130 pos=0 lvalid=1 neg=0 full=1
   score_f32=0 fused=0 tile=8 tile_kernel=1 fattn_cover=0 split_eff=1`), strides and grid are therefore the
   engine's, and a null below is about the arithmetic and not about the rig. The canary aborts the run on any
   difference (`rc=3`), so no CASE line can be read without it having passed.
2. **`CONTROL widen_*` — 3/3 FIRED.** `core.lvalid + 1` (the same host-side bound lie `causal_bound.cpp`
   calls `widen1`) makes the varied lane legitimately live; the identical mutation then moves **1 743 / 2 043
   / 2 048 of the adjudicated row's 2 048 elements** (`widen_real` / `widen_kbig` / `widen_vbig`), max
   31 057 / 32 406 / 32 828 BF16 ULP, adjudicated-row `1−cos` 4.507e-04 / 2.638e-01 / 3.440e-01 —
   231 092 / 260 602 / 256 742 elements across the whole tile. Same buffers, same mutation, one integer
   different — so the mutation demonstrably reaches the kernel through the buffer this rig wrote. When
   `--lane` names a lane that even a widened bound cannot reach, the control correctly refuses and the rig
   says *"a MUST-FIRE control did not fire; nothing in this run adjudicates anything"* rather than printing a
   null.
3. **`vident` — the write path itself is inert**: running the whole mutation machinery while writing the
   SAME bytes leaves every row of the tile byte-identical (`ndiff=0`, `rows_above=0`). And **reach witness**:
   every real mutation moves 115–127 rows above the adjudicated row, so the perturbation reached the model.
   `aout` is pre-filled with a `-1024.0f` sentinel, not with the engine's answer (`AGENTS`' `mixed_bw` row).

### Arm B — the max/sum path: NULL, and null at a 4096× amplifier

Adjudicated row byte-identical in every configuration tried, for all four K mutations:

| K mutation of the disallowed lane | row 8 (L3) | row 8 (L7) | row 4 / 6 / 7 | knob 1 |
|---|---|---|---|---|
| `kreal` (the perturbed dump's real K row) | identical | identical | identical | identical |
| `kzero` (K row = 0) | identical | identical | identical | identical |
| `kbig` (K row × 64) | identical | identical | identical | identical |
| `khuge` (K row × 4096) | identical | identical | identical | identical |

A score that reached `mn` at ×4096 would collapse the row (the live max would move by four orders of
magnitude and every weight with it); nothing moved by one bit. **PART 1's reading is confirmed on device:
`engine/attn_wide_llama_kernels.h:237` overwrites the cell before `:241`'s max, `:246`'s butterfly pairs two
halves of ONE row-slot, and `:260-261` sees the already-masked cell. The mask-to-`-inf`-before-the-reduction
fix is therefore aimed at a path that is already clean — B did NOT fire, so that fix is not the one.**

### Arm C — the datum reproduces, outside the engine, in one process

Varying lane 9's K **and** V (the perturbed dump) moves row 8's `pre_gate`: **4 of 2048 elements, each by
exactly 1 BF16 ULP, `1−cos 2.708e-08`, `meanrel 1.238e-05`, heads {4,5}**, deterministic (two launches of
identical inputs byte-identical). That is FIX-CROSSROW Result 5's datum (row 8, a handful of elements,
≤ 1 BF16 ULP) rebuilt from dumped operands with no engine in the loop.

### Arm A — the dot is NOT inert: it carries the sign of a zero

`vreal` (V from the perturbed dump) reproduces row 8's move **exactly** — same 4 elements, same ULPs, same
`1−cos` as arm C — and `kreal` contributes nothing, so the carrier is the **V operand of the dead lane**.

The response is not a weight times V. Writing ONE dim of the disallowed V row (`V9[62]`, whose own live
output element is `+0.400390625 = 0x3ecd`) and probing that one element:

| `V9[62]` | −1e-12 | −1e-8 | −1e-6 | −1e-4 | −1e-2 | −0.0992 | −1 | −10 | −100 | −1000 | 0 | +1 | +10 | +100 | +1000 | +1e6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| output (h4, dim 62) | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecc` | `0x3ecd` | `0x3ecd` | `0x3ecd` | `0x3ecd` | `0x3ecd` | NaN |

Twelve decades of magnitude, one answer; **the sign bit is the only thing that gets through**. At
`−1e-12` the staged value is f16 `-0.0` (below f16's minimum subnormal, `5.96e-8`), and a signed zero is
still enough to move a live output by one BF16 ULP; at `+1e6` the staged V overflows F16 to `inf`, the dead
term becomes `0 × inf = NaN`, and the NaN appears in the live row — so the dead k-term is genuinely formed
and genuinely accumulated, not skipped. Consistently: all movers move **down** by exactly one BF16 ULP (a
uniform shrink, which is what a rounding-level perturbation does, not an attended lane).

### The dependence set this rig measures — CROSSROW's law, reproduced, with a mechanism

Verified at the file level with `report.py` (`bf16_ulp`), not with the rig's own counters:

| adjudicated row (bound) | lane varied | where that lane sits | moves the row? |
|---|---|---|---|
| 4 (bound 5, odd) | 5 | first dead, **straddles** | **YES** — 10 elements ≤ 1 ULP, `1−cos 1.2e-07` |
| 6 (bound 7, odd) | 7 | first dead, **straddles** | **YES** — 15 elements ≤ 1 ULP, `1−cos 1.26e-07` |
| 8 (bound 9, odd) | 9 | first dead, **straddles** | **YES** — 6 elements ≤ 1 ULP (`vreal`: 4) |
| 7 (bound 8, even) | 8 | first dead, **does not straddle** | no (`vident`-grade identical) |
| 7 | 9 | second dead | no |
| 8 | 10 | second dead | no |
| 6 | 9 | third dead | no |
| 0 | 9 | ninth dead | no (and its control correctly refuses) |
| 8 | 5 | **live** for rows ≥ 5 | YES — from ONE dim of that lane's V alone, row 8's dim-62 element moves **111 BF16 ULP** (`v62_m1`; `vzero` on the whole row: 1 982 elements, max 31 489 ULP), and rows 5, 6, 7 legitimately, **and row 4**, for which lane 5 is its own first dead lane (bound 5, odd → straddles), exactly as the law predicts |

So both of CROSSROW's laws hold *and* are now explained: **reach is exactly one lane, and only when the bound
is odd** — because only then does the masked cell share its packed `f16x2` P register with a live cell
(Result 7's clue, confirmed rather than clued). The single-dim control on a live lane (writing one dim of a
live lane's V moves the row by 111 ULP where a dead lane's whole row moves it by 1) shows the rig would have
seen a dependence in the even-bound or second-dead rows had one existed at this sensitivity.

Both wide spellings behave identically (`--knob 2 attn_wide_llama_wave`, `--knob 1 attn_wide_wave`); the
rig's knob-1 run is byte-identical to knob 1's own dump, and knob 1's and knob 2's dumped `pre_gate` tiles are
byte-identical to each other (`cmp` clean), so the carrier is what the two share — the packed f16x2 P operand
and the `v_wmma_f16` P×V leg — which is also why knob 2 owes knob 1 byte identity and gets it.

### What this names, and what it does not

**Named**: the mechanism is NOT in the row's max/sum path, and it is not an attended lane. It is the dead
k-term of the P×V leg: `p = +0.0` for a masked cell, `0 × V = ±0`, and the **sign** of that zero changes the
packed pair's F16 accumulation by one F16 ULP, which surfaces on a live output element as one BF16 ULP
whenever that element sits near a BF16 rounding boundary. It is deterministic, it is in the shipping form,
and it reproduces on real magnitudes.

**Not named**: where a `-0.0` addend changes a *non-zero* F16 sum — that is not a property of an IEEE add, so
the instruction-level reason is still open. The next measurement is small and does not need the engine: one
hand-built `v_wmma_f16_16x16x16_f16_w32` tile whose C operand is a known non-zero value and whose k-pair is
`(+w_live, +0.0)` versus `(+w_live, -0.0)`, dumping the D bits; if the D bits differ, the pair is summed in
one hardware step and its rounding is the leak, and the ISA listing of `attn_wide_wave`'s P×V body says which
instruction to read.

**Fix directions, for the main agent — NOT implemented** (shipping-path change):
* The mask-to-`-inf`-before-the-reduction fix that this brief pre-authorised **is not the fix** — arm B is
  null with a 4096× amplifier, so the reduction is already clean. Do not spend the hot path on it.
* What the evidence supports is making the dead k-term's *sign* not reach the accumulator: forcing the
  staged V to `+0.0` for disallowed lanes (a per-lane store cost in the V stage, per round), or forcing the
  packed P pair to a signed-positive zero before the wmma (`e + 0.0f` on the masked select, which is a
  `v_add` per cell on the tail body only), or, cheapest of all, accepting it and recording it — because
  arm B's null plus arm A's sign-only null-at-`+0` means the observable is bounded by one F16 ULP of the
  accumulator, i.e. the `CASCADE_BAND` it hides inside is doing its job.
* `--attn-fattn-wide 0` remains the clean off switch (CROSSROW §2), unchanged by anything here.

### Honest loose ends

* The rig's own `row_at_moved` counter disagreed once with the bytes on disk (`--row 7`, `vzero`: the C++
  printed 1 in an earlier build, the file is byte-identical, and a later build prints 0). **Every number in
  PART 3 is taken from the files, through `report.py`/`bf16_ulp`, not from that counter**, and the counter is
  labelled in the source as a live readout of rep 0. Nothing in the conclusions depends on it.
* `real`/`vreal`/`kreal` are only meaningful for the row whose own bound is the perturbed position (row 8 ↔
  position 9); for rows 4, 6, 7 those cases write byte-identical data and read as `vident` does. The
  synthetic mutations are what adjudicate those rows.
* One row per layer at one body width (128 rows, `split=1`, `lanes_last=128`). The streaming region (rows
  above `MAXSC`, `split=2`) is untouched by this rig, and FIX-CROSSROW's arm 9 is still outstanding.

### Device runs (correctness-only; no timing taken; `tools/gpu_lock.sh` held; wall = whole command)

| command | wall | `util` at acquire |
|---|---|---|
| `decode_run --ids-file ref.csv --steps 0 --prefill-logits all --dump-attn-rows` + same on `pert9.csv` | 2 s | 91 % |
| `pack_ab --dump dump-ref --pert-dir dump-pert9 --row 8` (13 cases) | 1 s | 92 % |
| same, 22 cases | 1 s | 0 % |
| same, 30 cases | 0 s | 91 % |
| `decode_run … --attn-fattn-wide 1` ×2 (knob-1 dump pair) | 3 s | 91 % |
| `pack_ab --row {7,6,4}`, `--seq 2 --layer 7 --row 8` | <1 s each | 91 % |
| `pack_ab --row R --lane L` reach battery (5 configs) | <1 s each | 1 % |
| `pack_ab` signed-zero ladder (35 cases) + `--probe 4:62` report | <1 s | 0 % |

`util` on this box reads high because the session serves its own tokens on the same iGPU; every figure above
is a whole-command wall, and **no kernel timing, burst, best-of-N or sweep appears anywhere in this file**.
