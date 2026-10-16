# `prototype/attn_score_dot2` — pricing an arithmetic-class change to the score loop

**Status: priced, STOPPED at the kill line. No kernel was written.** D19 funded this as the partner lever to
M-tiling ("`M`-tiling plus an arithmetic-class change to the score loop",
`progress/2026-10-16/prefill-decision-log-2026-10-16.md` D19). The price says the lever is too small to be
that partner, and says something more useful about what the "arithmetic wall" actually is.

---

## 0. The verdict, in the form the brief asked for

> **What fraction of the m2 form's 2.5402 ms residual at 4096 lanes could a 3–4× faster score dot remove?**

**3.9 % – 10.5 %**, i.e. **0.100 – 0.236 ms** of 2.5402 ms. The kill line in the brief is ~15 % of the wave,
so: **STOP**. For scale, D19's pair target (~1.25 ms per wave at 4096) needs **1.29 ms** removed; the best
framing of the dot2 removes 18 % of that.

Two independent framings bracket it, and they are 2.7× apart rather than one number because the kernel is
**not** issue-saturated — it sustains ~0.92 warp-inst per CU per clock of a 2-wide roof (§5.3):

| framing | at 4096 lanes, 64 rows | as % of the m2 wave |
|---|---|---|
| **(A) MAC-roof** — only the multiply-adds move, `3.65 → 11.64` T-MAC/s (both measured, not modelled) | 0.100 ms | **3.9 %** |
| **(B) instruction share** — the dot2 deletes 12 of the 114.6 VALU warp-inst a (row, lane, head) touch costs | ≤ 10.5 % of the wave (0.236 ms at the achieved issue rate) | **10.5 % upper bound** |

(B) is an upper bound in a precise sense: a kernel that achieves 106.1 G warp-inst/s of a 231.5 G/s roof
(45.8 %) is waiting on something other than issue slots for more than half its time, and a freed slot that
was not the critical path saves nothing. (A) is the floor because it prices only the FMAs.

**And the reason it is small is structural, not a modelling choice:** the score dot is **14 % of a touch's
instruction stream** (16 of 114.6 warp-inst). A dot2 that made it free would buy 14 %, not 50 %.

---

## 1. Inputs, and where each one came from

Every number below is measured elsewhere or is arithmetic on measured numbers. `price.py` re-derives all of
it and prints this list; nothing in the price is a guess wearing a fact.

| input | value | source |
|---|---|---|
| `attn_tile_wave` VALU warp-inst, 1024-lane prefill | 2,888 M | `progress/2026-10-16/prefill-occupancy-results-2026-10-16.md` §3 (`rocprofv3 --pmc SQ_INSTS_VALU`) |
| touches in that census | 25,190,400 | Σ_lanes 524,800 × 6 attn layers × 8 heads (geometry of `attn_decode_kernels.h`) |
| **VALU warp-inst per (row, lane, head) touch** | **114.6** | division of the two rows above; §3 of that doc prints "~115" |
| achieved VALU rate | 106.1 G/s = 45.8 % of roof | 2,888 M / 27.23 ms (same §3 row) |
| issue roof | 2.3152e11 warp-inst/s | 2 warp-inst/CU/clk × 40 CU × 2.894 GHz (D16 item 3; `rocminfo` 2 SIMDs/CU) |
| scalar `v_fma_f32` sustained | 3.65 T-MAC/s (7.30 TFLOP/s) | `prototype/issue_rate_probe/results.md` §1 |
| `v_dot2_f32_bf16` sustained | 11.64 T-MAC/s (23.28 TFLOP/s) = **3.19×** scalar | same |
| `v_wmma` sustained | 26.31 T-MAC/s (52.63 TFLOP/s) | same (named in §5) |
| measured waves (ship / m2) | see table in `price.py` | D19's window `attn-m2-2` table, all four rows |

Note the coincidence that makes the census trustworthy: the shipped wave's *whole* VALU stream, priced at
the rate the same measurement produced, predicts the measured wave times to within 7 %
(2.2495 ms predicted vs 2.4163 ms measured at 4096; 0.5825 vs 0.5991 at 1024). Instruction count explains
most of the time. It also bounds the error the wrong way round: the m2 form's per-touch count **cannot be
materially lower** than the shipped form's — the arithmetic per (row, lane, head) touch is identical, and what
the pair shares is loads, which are not VALU — and any excess bookkeeping makes the denominator bigger, which
makes the dot2's share *smaller*: at 20 % more instructions per touch the bound is 8.7 %, not 10.5 %. The one
open case — that m2 is somehow much cheaper per touch — is exactly what HANDOFF step 5 measures, with an exact
threshold (under 80 warp-inst/touch flips the verdict).

## 2. The score loop, counted out of the disassembly

CPU-only, no device: `hipcc -O3 --offload-arch=gfx1151 --save-temps -c` on a two-line TU that includes
`prototype/attn_decode/attn_decode_kernels.h` (HANDOFF steps 1–2), then read the GCN `.s`. Line numbers
below are **kernel-relative** inside `_Z14attn_tile_wave8CoreArgs` of that `.s`.

### 2.1 The dot itself: 16 ops, 13 issue slots

```
693  ds_load_b128 v[39:42], v25 offset:4096        ; the staged K row: 8 bf16 in 4 regs (LDS pipe)
699  v_lshlrev_b32_e32 v44, 16, v40              ┐
700  v_lshlrev_b32_e32 v46, 16, v42              │
701  v_and_b32_e32     v42, 0xffff0000, v42      │
702  v_lshlrev_b32_e32 v45, 16, v41              │  8 bf16→f32 conversions, ONE instruction each:
704  … v_dual_and_b32  v41, 0xffff0000, v41      │  low half << 16, high half & 0xffff0000,
705  … v_dual_and_b32  v40, 0xffff0000, v40      │  the duals pack a conversion beside an FMA
706  v_lshlrev_b32_e32 v43, 16, v39              │
708  … v_dual_and_b32  v39, 0xffff0000, v39      ┘
704,705,708,709  v_dual_fmac_f32 ×4             ┐
709  … v_dual_add_nc_u32 v25, 0x200, v25         │  (the pointer bump, not part of the dot)
711–714          v_fmac_f32_e32 ×4              ┘  8 FMAs into one accumulator, serial in t (§05.5)
715  s_cbranch_scc0 .LBB1_70                      ; loop back edge
```
16 ops of dot in 13 issue slots — four `v_dual_*` packets each pair a conversion with an FMA — plus one
`ds_load` and one pointer bump (`v_dual_add_nc_u32 v25, 0x200, v25`, the `+= DH` on the stage pointer). This is
the part a dot2 replaces, and it replaces it with 4 slots (§2.2).

### 2.2 `v_dot2_f32_bf16`: 4 ops, 0 conversions — compile-proved, not asserted

`dot2_shape.cpp` is a two-kernel toy with the same shapes as the real loop (extern "C", so the names survive
mangling). Compiled the same way (HANDOFF step 2), `isa_census.py` reports:

```
score_scalar  whole kernel: ops=43 VALU=34 …  v_lshlrev_b32_e32=9, v_and_b32_e32=7, v_mov_b16_e32=8, v_fmac=5+v_dual
score_dot2    whole kernel: ops=13 VALU=6  …   v_dot2_f32_bf16=4, conversions=0, fma=0
```
The scalar toy converts *both* operands (the real kernel hoists `q` into F32 registers once per row-head),
hence 34 vs the 16 quoted in §2.1; the load-bearing row is the dot2 kernel: **exactly four
`v_dot2_f32_bf16` and zero conversion instructions** — the packing is free, the halves stay in place, and
`q` needs 4 registers instead of 8. That VGPR saving is not a lever here: M = 2 is LDS-limited
(22,656 B → 2 blocks/CU), not register-limited.

### 2.3 The reduction is bigger than the dot it reduces

One `__shfl_down_sync` step of `warp_sum`, as compiled (kernel-relative 320–330; five steps, same shape):

```
320  s_cmp_eq_u32 exec_lo, -1            ; "is exec the mask?" guard — the compiler cannot prove it is
321  v_cndmask_b32_e64 v11, 0, 8, vcc_lo ; participation select
323  v_add_lshl_u32 v10, v11, v12, 2     ; LDS address for the permute
324  ds_bpermute_b32 v10, v10, v9        ; the shuffle itself, on the LDS pipe
325  s_cbranch_scc0 .LBB1_26             ; skip when exec is full
328  v_cmp_gt_u32_e32 vcc_lo, 28, v12    ; the mask again, in VALU
330  v_add_f32_e32 v9, v9, v10           ; accumulate
```
**4 VALU + 1 LDS + 2 SALU per step × 5 steps = 20 VALU, 5 LDS, 10 SALU per touch.** The reduction costs
more issue slots (25) than the dot (13) whose result it produces. That ratio is the finding: the score
*dot* is not where the score's cost is.

### 2.4 Per-touch census, and the honest label on each line

| part | VALU warp-inst / touch | share of 114.6 | how known |
|---|---|---|---|
| score dot (8 conversions + 8 FMAs) | 16 | 14.0 % | counted, §2.1 |
| score reduction (5 shuffle steps + guards) | 20 | 17.4 % | counted, §2.3 |
| score BF16 round, ×SCALE, addressing | ~6 | 5.2 % | counted, the tail of the same loop |
| **score loop total** | **42** | **36.6 %** | |
| everything else: `expf`, V conversions, 8 PV FMAs, rescale, LDS address math | 72.6 | 63.4 % | **by subtraction** from the measured 114.6; its composition is the opcode mix `isa_census.py --ops` prints, not a per-touch count |

`isa_census.py` is the tool for the static side (op mix, per-loop grouping by fingerprint rather than by the
compiler's `BBn_m` numbering, `v_dual` counted as 2 ops in 1 slot). Loop attribution there is approximate on
this shape — the butterfly is unrolled under guards and LLVM splits it across blocks — which is why the
per-touch figures above are quoted from windows with line numbers, and the script labels its own static
counts as presence checks.

## 3. The price, all four funded waves

```
                          (A) MAC-roof          (B) instruction bound
wave                 ship / m2 ms   Δ ms   % m2     ≤ % of wave   ≤ ms (at achieved rate)
wave64-at-1024    0.5991 / 0.6511   0.026   4.0 %       10.5 %        0.061
wave64-at-2048    1.2753 / 1.3546   0.051   3.8 %       10.5 %        0.120
wave64-at-4096    2.4163 / 2.5402   0.100   3.9 %       10.5 %        0.236
wave16-at-4096    0.7166 / 0.7667   0.025   3.3 %       10.5 %        0.059
```
The full arithmetic, including MACs by part (score dot, PV dot, rope/norm in `attn_prep`, the §05.6
epilogue) and which ceiling each part runs at, is `price.py`'s output, captured in `price.output.txt`.
Two structural lines from it:

* **The PV dot cannot be a dot2.** `__builtin_amdgcn_fdot2_f32_bf16` returns a *single* f32: it folds two
  products into **one** accumulator. `acc[u] += wt * bf(v[u])` needs eight accumulators, one per output dim,
  and shares `wt` across them instead of sharing an accumulator. The shape is wrong, not inconvenient. So
  the largest single arithmetic item in the wave (16 of 114.6, same size as the score dot) is out of reach
  of this instruction class.
* MACs are not where the wave lives. At 4096 lanes both dots together are 1,065 M MAC, which at the
  *scalar* sustained rate costs 0.292 ms of the measured 2.4163 ms. The other ~2.1 ms is instructions that
  are not multiply-adds.

## 4. Forensics: the "score cost = 0.246 ms at 1024 lanes" claim

The literal figure is **not in the tree** (grep over `prototype/`, `progress/`, `bench/`, `specification/`).
Two readings reproduce its neighbourhood, and neither is the score dot. The first:

```
538,880 touches × 114.6 warp-inst / 2.3152e11 warp-inst/s  =  0.267 ms
```
i.e. **the entire per-touch instruction bill of that wave at the 2-warp-inst/CU/clk roof**, 8 % above the
quoted figure. The score dot's real cost in the same wave is 0.0378 ms at the measured sustained scalar rate
(138.0 M MAC / 3.65 T-MAC/s), and the whole score *loop* is 0.098 ms at the roof / 0.213 ms at the achieved
rate. So the figure overstated the prize by ~7× if it was quoted as the dot's cost, and by ~3× if as the
loop's.

The nearest ancestor that *is* in the tree is the 2026-10-14 pre-tile census
(`progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md`, §2 near line 116): "the ISA runs **~1,030
warp-instructions per touch** in the one-pass form, of which the score-tile loop is 190 and the softmax+PV loop
~840". That form has been replaced — tiling moved pass 2 from "every warp of the block executes it
redundantly, so each touch pays `expf` and a divide **8 times**" to "exactly one warp does a head's pass 2"
(the note at `attn_decode_kernels.h` line 429 says exactly that) — and the shipped touch census is 114.6, a 9×
cut that came almost entirely out of the softmax side. Two things follow, and both matter more than the 0.246
figure itself:

* **190 of 1,030 = 18.4 %** — even in the pre-tile form, with a different metric (all pipes, counted statically
  off loop bodies, not the counter's VALU-only stream, so do not compare it to 114.6 directly), the score loop
  was under a fifth of the touch. The score dot has never been the instruction pile, in either form.
* Reading 0.246 ms off *that* census gives 190 × 538,880 / (4 inst/CU/clk × 40 CU × 2.894 GHz) = **0.221 ms** —
  the same neighbourhood, using the retired 4-wide issue assumption D16 later corrected to 2. So the figure is
  underdetermined, and there are two plausible ancestries near it. What is **not** plausible under any reading is
  that it is the score dot's cost: that is 0.0378 ms at a measured sustained rate, 6× smaller than the smallest
  reconstruction.

**Lesson worth keeping: a cost attributed to a loop must name its denominator, its form, and its metric.** All
three of the retired inputs behind 0.246 were individually defensible when written; the number survived them
by being quoted without any of the three.

## 5. What the arithmetic wall actually is (this is the part D19 needs)

D19's framing — traffic wall ≈ arithmetic wall, so pair M-tiling with "an arithmetic-class change to the
score loop" — is right that a byte-only lever dead-ends, and wrong about the shape of the other wall.
Restating it from §2/§3:

1. **Both dots together are 27.9 % of the instruction stream** (14.0 % score + 13.9 % PV). That is the
   ceiling on what *any* arithmetic-class change to the dots can save, and it is an upper bound reached only
   if the kernel were issue-saturated. So the 2.42 → ~1.25 ms target is **not reachable by an arithmetic-class
   change to the dots**, not by dot2 (≤10.5 %), not by WMMA (≤27.9 % before its own layout costs).
2. **The larger part of "the arithmetic wall" is the non-dot tax**: the shuffle's mask guards (17.4 %),
   `expf` (~14 ops, ~12 % — `v_exp_f32_e32` + `v_ldexp_f32` + the argument reduction and clamps around them,
   counted at kernel-relative 633–649), the IEEE divide (`v_div_scale_f32`/`v_rcp_f32`/`v_div_fmas_f32`/
   `v_div_fixup_f32`, ~6 ops per site; once per touch on the one-pass route, once per (row, head) in the
   epilogue on the streaming route these waves take), the BF16 round-trips, and LDS address arithmetic. The
   static census agrees in kind if not in scale: `expf`/divide/rndne opcodes are 7.1 % of `attn_tile_wave`'s
   static instruction stream (`isa_census.output.txt`; static share and dynamic per-touch share are different
   metrics and are not interchangeable — the dynamic number is bigger because both live inside loops). The
   pass-2 loop is the kernel's second-largest body by VALU content (loop `BB1_86`, 47 VALU per iteration, vs
   the score loop's 24 in the same approximate attribution).
3. **The half that is not instructions at all is latency.** Two readings agree on ~0.92 warp-inst per CU per
   clock, from independent directions: the occupancy doc's own "issue utilisation" column is 95 %/92 % of its
   1-inst/CU/clk floor, and `price.py` gets 106.1 G/s of the 231.5 G/s roof = 45.8 % = 0.92/CU/clk from the
   counter census divided by the measured ms. More than half the issue capacity is idle, so there is a second
   wall under the instruction count: the per-lane dependency chain — staged `ds_load_b128` → 8 serial FMAs →
   5 permutes (each an LDS round trip) → round → store. A dot2 shortens the FMA segment (8 dependent FMAs → 4
   steps) and nothing else. This is the shape of the mistake `gdn_collapse` recorded — an instruction-count
   price came in 9.8× below measurement because "the cost is latency per token at ~realised occupancy, not
   instructions per token" (its §4–§5); its occupancy numbers are its own and are *not* imported here, and the
   retired "wavefronts per CU against a max of 64" claim in the occupancy doc stays retired. The honest
   statement is the one above: 46 % of the issue roof, whatever the resident-wavefront count turns out to be.
4. The lever this analysis *does* expose, without funding it: `v_wmma_f32_16x16x16_f16` at 52.63 TFLOP/s
   covers **both** dots (D19 says the same, and adds that fp16 does not halve bytes because it is also
   16-bit). Its ceiling is the 27.9 % of line 1, and it pays only if a wave can feed WMMA-shaped operands
   without a transpose tax — which is a price, not a hope, and belongs in its own brief.

## 6. THE GATE PLAN for step 2 (written, not executed — step 2 is unfunded)

If a dot2 is ever built, these are the checks that decide it. Nothing here needs a kernel to *prepare*: steps
1 and 2 of HANDOFF step 6 are CPU-only and would be finished before a `.hip.cpp` exists.

### 6.1 What a dot2 changes numerically — and it is one thing only

Operands do not change: bf16→f32 is exact, and `v_dot2_f32_bf16` consumes the same packed bf16 pairs that
the conversions read. The **only** change is the association of the 8-term in-thread partial sum: serial in
`t` (§05.5's ordered reduction) becomes four two-term products folded into the accumulator, and the exact
internal order of `fdot2` (`acc + (a0*b0 + a1*b1)` vs `(a0*b0 + a1*b1) + acc`) is a hardware fact to be
*measured*, not assumed from the builtin's name. The 5-step butterfly, the single BF16 round of the score,
`× SCALE`, and all of pass 2 are untouched.

### 6.2 Which checks go red, and which do not

| check | what it holds today | after a score-loop dot2 |
|---|---|---|
| `prototype/attn_decode/run.py` vs the HF pins | `attn_pre_gate`/`gated` inside the **2e-4** band (`attn_decode/results.md` §5, 1.4× margin); copy stages (`cache_keys`, `cache_values`, `v_append`, `gate_vec`) bit-exact | expected to stay **green** — see §6.3. If red, that is the D13 question, and the band is the one to argue about |
| `prototype/attn_m2/m2_model.py compare` + `run.py ab` | **bit-identity** of the m2 form against the shipped mirror, 0 differing elements | **red by construction** unless the mirror models the dot2 order too. See §6.5 — this is the expensive one |
| `prototype/attn_tile_model.py` mirror + `--neg-wide` | derived must-catch / must-not-catch sets | must-catch magnitudes must be re-derived against the new baseline, §6.4 |
| `prototype/attn_m2/run.py` NEGVERDICT | per-class scoping (fixed this session): forms-agree for form-neutral + device-only classes; `moved_ship == 0` for m2-only classes | unaffected in logic; `dimhalf`'s "accumulates 128 of 256 dims" stays a must-catch, and must still move *beyond* the dot2's own movement |
| `engine/gate.py` numerics section (prefill row coverage + floors) | floors from `route_allowance.json` | only if the engine adopts it; coverage check first (`cmd_flag`), never after |
| `speccheck/` | torch CPU reference vs pinned dumps | **unaffected** — it never runs the kernel. The pins stay the pins |

### 6.3 The existing floor that may make the whole question free

`prototype/attn_decode/results.md` §5 **F1** already probed re-associating *this exact dot*: "score dot: the
kernel's warp tree (8 dims/lane + 5 shuffle steps) and a pairwise 256-sum, vs torch's matmul order",
15,592 scores → 108,544 output elements, result **0 scores flip, 0 elements differ**, with the recorded
conclusion "Reordering this op's reference arithmetic needs no band at all — the products are BF16-exact and
the sums are 256 terms at magnitudes where F32 reassociation stays far inside a BF16 step."

A dot2's re-association is *narrower* than F1's pairwise-256 case, so if F1 holds, D13's rule 1 applies and
no new band and no user decision is needed at all. **But F1's sample is 15,592 scores from the pinned spans
(one prompt, ≤1024 lanes)**, and AGENTS.md's `mlp_gemv` lesson says reassociation damage is rare and
cancellation-local: for a long dot, re-ordering the reference's own F32 sum already moves ~1 element in 1e5
by more than 1 ULP, always at an element that cancelled to ~1e-6 of the tensor max. 0-of-15,592 cannot see a
1-in-1e5 event. So F1 is a strong prior, not a clearance.

### 6.4 The floor probe that would size it, and the metric rule

New `prototype/attn_score_dot2/floor_probe.py`, on the pattern of `prototype/mlp_gemv/floor_probe.py` (the
same instrument `gdn_collapse` reused), over the pinned `q_rope`/`k_rope` dumps, at a sample size ≥ 1e6
scores — which means the wide anchors that `attn_tile_model.py --neg-wide` already defaults to
(`wide640f`, `wide1024f`) plus the p300 cycled fixture, not just the pinned spans:

* reproduce the dot2 association on CPU (pack the thread's 8 dims into 4 bf16 pairs, sum each pair in F32,
  then the identical 5-step butterfly), and report **ndiff after the BF16 round**, **max ULP plus a
  cancellation band** sized by `|ref| / max|ref|`, and **cos / meanrel on the chained outputs**
  (`attn_pre_gate`, `gated`);
* the metric rule from AGENTS.md, because getting it wrong here is what made `mlp_gemv`'s first gate vacuous:
  **max-ULP bands are blind to whole-tensor 1-ULP moves, and ndiff is blind to chains.** ndiff is admissible
  only on the stage-local near-exact op (the rounded score, floor 1 ULP, no cancellation band); the chained
  output is judged by cos/meanrel aggregates;
* the band, if one is needed at all, is then **the probe's own movement**, not taste — and it must still catch
  every injection in the **derived** must-catch set. Concretely: add one `--dot2` association switch to
  `prototype/attn_tile_model.py` and `prototype/attn_m2/m2_model.py` (both sides of every comparison, never
  one — the §20.3 lesson is that a reference derived from the thing under test cannot adjudicate it), then
  require for every derived case that the injection's movement exceeds the dot2's own movement by ≥10×, and
  that the **must-not-catch set stays bit-exact**. Do *not* widen the blind-row check: a widened "moved"
  predicate makes `maskwide` invisible on the rows it must not touch, and §20.6's `partalias` episode is what
  a control that cannot fire costs. The case list must include the wide anchors and the tile-boundary rows of
  exactly 512/513/514/515 lanes, since that is the geometry the derived sets need.

### 6.5 The cost that is not about size, and is the real reason to decline

M = 2's admission to the tree is **bit-identity** — `m2_model.py compare` reports 0 differing elements
against the shipped mirror, and the device gate holds it at every funded width including the 1032-lane pinned
HF row. A score-loop dot2 makes that claim false by construction: the shipping pair would be (m2 + dot2), and
"identical to the shipped form" would have to be replaced by "inside band X versus the shipped form", with the
mirror carrying the dot2's association. That is spending the strongest numerics guarantee in this tree — a
bit-level one, on the op whose whole §05.5 contract is an *ordered* reduction — to buy at most 10.5 % of a
wave, on a form that is currently 2× off the memory plateau for other reasons.

So the recommendation has two halves, and the second does not depend on the first: **do not build the dot2,
and if an arithmetic-class change is wanted anyway, pick the one that does not spend bit-identity to buy less
than a sixth of the target** — i.e. price WMMA over both dots (ceiling 27.9 %) rather than dot2 over one.

## 7. Files here, and what was deliberately not done

| file | what it is |
|---|---|
| `price.py` | the price. Constants carry their source file; `--isa-dir` re-checks the ISA claims |
| `price.output.txt` | its captured output (CPU-only; no timing, no device) |
| `isa_census.py` | static op-mix / per-loop counter for a GCN `.s`; `--ops` regex, `--loop-has` fingerprint |
| `isa_census.output.txt` | census over both kernel TUs and the toy |
| `dot2_shape.cpp` | the two-kernel compile-proof of §2.2 (`extern "C"`, no main, never run) |
| `isa_tu.cpp`, `isa_tu_m2.cpp` | two-line TUs so either kernel header is compilable standalone for `--save-temps` |

**Not done, on purpose:** no `attn_score_dot2.hip.cpp`, no `build.sh`, no device run, no timing, no
`gpu_lock`, nothing committed. Step 2 of the brief is conditional on the price clearing ~15 % and it did not.
The gate plan above is the deliverable for the case where someone overrules the price; HANDOFF.md carries the
commands, including the one measurement that would falsify this price (the m2 form's own warp-inst/touch
census — HANDOFF step 5).
