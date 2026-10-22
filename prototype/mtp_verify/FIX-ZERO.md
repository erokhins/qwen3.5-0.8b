# FIX-ZERO — pin the instruction-level cause of the signed-zero leak, fix it minimally, make the probe permanent

Checkpoint written BEFORE the analysis, per the brief. Read `CROSSROW.md` (the datum),
`FIX-CROSSROW.md` Results 0/2/5/6/7 + "What is left", and `ISOLATED.md` (the arm A/B/C isolation, commit
`316f08c`) first; this file is the work log for the follow-up, not a substitute for those.

## What is already established (carried in, not re-derived here)

* The datum: perturbing lane `bound(q)` moves row `q`'s attention output by <= 1 BF16 ULP in a handful of
  its 2048 elements, deterministically, only when `bound(q)` is odd (`FIX-CROSSROW` Results 1/5).
* Arm B (a disallowed cell's score reaching the row `max`/`rowsum`) is a **decisive NULL** at a 4096x
  amplifier (`ISOLATED` arm B). `engine/attn_wide_llama_kernels.h:237` overwrites the accumulator cell with
  `-1e30f` before the max at `:241`, the `t^16` butterfly at `:246` pairs two halves of ONE row-slot, and the
  exp/rowsum at `:260-261` reads the already-masked cell. **The reduction path is not touched by this fix,
  and the mask-to-`-inf` idea is dead.**
* Arm A: the dead k-term of the P*V leg (`:338-339`,
  `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(v, p[L], out.t[tau], op_sel)`) is **formed, not skipped**.
  Writing one dim of the dead lane's V row to `-1e-12` (f16 `-0.0`) versus `0` moves the live row's element
  by exactly 1 BF16 ULP; twelve decades of magnitude make no difference, only the sign bit does
  (`ISOLATED`, arm A table). At `+1e6` the staged V overflows F16 to `inf`, `0 x inf = NaN`, and the NaN
  lands in the LIVE row — recorded under "Hazard" below, not fixed.
* Open at the time of writing: *why* a `-0.0` addend changes a non-zero F16 sum at all, since IEEE says it
  cannot. `ISOLATED` names this as the one remaining instruction-level question.

## Plan (the three funded steps)

1. **Pin the instruction behaviour.** `prototype/mtp_verify/wmma_sign/` — a hand-built
   `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` tile, C a known non-zero value, one k-position holding a
   signed zero with everything else byte-fixed, dumping the D bits. Both operand spellings are tested, because
   the engine's product is `V x P` and the *sign* in the datum comes from the V side while the P side already
   holds `+0.0`. If D differs, the instruction-level cause is pinned and the ISA listing names the
   instruction. If D does NOT differ, the sign effect comes from elsewhere in the staging path and that is
   what gets reported instead - no fix ships on a premise the probe refuted.
2. **Minimal fix** at the store/select where the mask is applied, only if step 1 supports it; verified against
   the emitted code, not the source text; proved three ways (arm A V-ladder null + `widen1` controls still
   firing / CROSSROW per-row sweep clean at 128 and >= 640 rows / `engine/gate.py --fixture dec8` + one wide
   fixture still PASS with the ndiff/ULP delta counted).
3. **Permanent gate**: a geometry-derived per-row perturbation sweep driver
    (`prototype/mtp_verify/zero_gate.py`) with must-catch = one-lane widening + signed-zero injection,
    must-not-catch = even-bound rows + the clean forms, and a selftest proving both directions.

## Constraints honoured

No timings anywhere (cost is the main agent's to measure). Correctness-only device checks, each <= ~60 s,
one at a time, `tools/gpu_lock.sh` held by the caller, wall + `util` reported, "no timing taken". No edit to
gate thresholds / floors / pins, `engine/performance-status.md`, `engine/generation-limits.md`, `README.md`,
`progress/`. Nothing is committed. `LC_ALL=C`.

## State

- [x] read `ISOLATED.md`, `FIX-CROSSROW.md`, the wide round (`attn_wide_llama_kernels.h:225-340`), the operand
      map (`attn_fattn_kernels.h:453-469`), `pack_ab/`
- [x] step 1 probe built — `prototype/mtp_verify/wmma_sign/{wmma_sign.cpp,build.sh}`
- [x] step 1 run on device — **the answer is a measured NULL, controls fired** (§Step 1)
- [x] the datum re-read at bit level — `prototype/mtp_verify/packab_bits.py` (§The datum's bits)
- [ ] step 2 fix — **NOT WRITTEN: step 1 does not support the prescribed one.** Escalated as a decision;
      see §"What this leaves open" for the three options and why the prescribed edit cannot be the cause-slayer.
- [ ] step 2 three-way proof — not applicable with no fix; instead: **no engine file was touched** (see
      §"Tree state"), so every pinned gate that was green before this pass is green after it by construction,
      and `engine/gate.py` was deliberately not re-run for it (AGENTS: do not re-make a result that nothing
      could have changed).
- [x] step 3 gate + selftest — `prototype/mtp_verify/zero_gate.py`, green on the shipping default form at 128
      and 640 rows, green on `--attn-fattn-wide 0` with **no** known leak, and its selftest fails in both
      directions on demand (§Step 3)
- [x] hazard recorded (below), and now witnessed on device (`infnan` control, 64/64)

## Step 3 — the permanent gate: `prototype/mtp_verify/zero_gate.py`

The check that did not exist when the defect did. Its ability to fail is the deliverable, so every count in it
is asserted and its must-not-catch side is tested as hard as its must-catch side.

**Why this hid.** The numerics band that covers this output is ~2e-3 relative and the effect is **one BF16 ULP**
(1−cos measured 2.7e-08 for the same move in `ISOLATED` arm C), i.e. the leak sits about five orders of
magnitude below the tolerance it has to pass. A band that wide was bought for a different purpose — bf16
chaos across forms (0.8776 sdpa-vs-eager token agreement, `ASSUMPTIONS` A-18) — and nothing in the tree asked
the question the leak does fail: *is this row byte-identical when every operand it is entitled to is
byte-identical?* That question is what this gate asks, per row, at the attention output.

Geometry is derived, not typed: `WIDE_POS_PER_BLOCK` is resolved through its own macro chain
(`(WIDE_WARPS * WIDE_POS_PER_WARP)` = 32, and the resolver had to cross headers and strip a comment containing
a `/` to read it at all), the row count comes from the fixture manifest, the attention layer from the smallest
`L` in the dump dir, the row width from `attn_dump_diff.ROW_FLOATS`, and ULP distances from
`bf16_ulp.monotone_key`. The derived position set is asserted non-empty and asserted to cover both bound
parities; a position at or beyond the first tile's row count is **excluded by name** rather than silently
adjudicated against a tile that cannot contain its straddle row.

The rule table (see the module docstring): rows after the straddle row are exempt and must move (reach
witness, else the run is unclassified); the straddle row with an **even** bound must not move; with an **odd**
bound it may move **<= 1 BF16 ULP** and no NaN, which is recorded as `known-leak` — carried, not waived, so a
worsening (more ULP, a NaN, or any row further back) goes red; every row before the straddle row must not move.

Three arms, each its own invocation so every device command stays small and blocking:
`sweep` (engine, one `decode_run --dump-attn-rows` per position), `sinfzero` (the signed-zero injection
through `pack_ab`, expectations **derived from the value the rig prints** against the F16 range — so the
`0 x inf` arm is a must-fire in its own right instead of a typed exception), `widen` (reuses
`causal_bound --must-catch`'s `widen1` host-side bound lie instead of re-deriving an injection).

Measured, this session (device, correctness-only, no timing):

| command | wall | result |
|---|---|---|
| `zero_gate.py plan --fixture p128` / `--fixture wide640f` | <1 s | `WIDE_POS_PER_BLOCK=32`, 4 / 20 rtiles, both parities covered |
| `zero_gate.py run --fixture p128` | 15 s | GREEN: sweep cases=8 violations=0 **known_leaks=2** (rows 2, 98 — odd bounds — 1 cell each, 1 ULP); sinfzero cases=16 failures=0; widen red=39 green=0 |
| `zero_gate.py run --fixture wide640f --arms sweep --budget 5` | 20 s | GREEN: cases=5 known_leaks=2 (rows 2, 34 — 6 and 5 cells, 1 ULP) on a 512-row first tile |
| `zero_gate.py run --fixture p128 --arms sweep --attn-fattn-wide 0` | 8 s | GREEN, **known_leaks=0** — the slim off-switch is byte-clean, so the gate is not firing spuriously |
| `zero_gate.py run --fixture p128 --arms sweep --attn-fattn-wide 1` | 8 s | GREEN, known_leaks=1 — knob 1 carries the same leak as knob 2, as `FIX-CROSSROW` Result 3 said |
| `zero_gate.py selftest --ref-dump /tmp/zero_gate/p128/zg-ref` | <1 s | cases=6 failures=0: clean copy reads clean; 1-ULP straddle move reads `known`; 2-ULP straddle move, even-bound move, a row-further-back move, and a NaN all read VIOLATION |

Slow option left to the main agent: `--fixture wide4096f` (each body there costs an order of magnitude more,
so that window is not a sub-minute worker run).

## Step 1 — the answer: the sign of a zero k-term does NOT reach the F16 result

`prototype/mtp_verify/wmma_sign/` (device, correctness-only, `--section 1|2`). Final run:
`SUMMARY controls=194 control_failures=0 measures=195 sign_reaches_result=0`, verdict *"NO SIGNED-ZERO
SENSITIVITY at add level or in the hand-built tile (measured null, controls fired)"*.

**Section 1 — the add level.** Four spellings of "add a zero product to a non-zero F16 accumulator", at six
accumulator values × five multiplicands, operands from global so nothing folds:
`x+(+0)` vs `x+(-0)`, `fma(+0,+v,x)` vs `fma(+0,-v,x)`, `fma(+0,+0,x)` vs `fma(-0,+0,x)`, the non-fused pair
`x+(0*±v)`, and the packed `v_pk_mul_f16` + `v_pk_add_f16` form. **200 measures, and every single one is
identical** except the six taken with `x = -0.0` itself, where a difference is IEEE's own rule
(`-0 + +0 = +0`, `-0 + -0 = -0`) and not a leak. So the open question `ISOLATED` left — *where does a `-0.0`
addend change a non-zero F16 sum* — has the answer **nowhere, on this part, in any of these spellings**. The
emitted instructions are the ordinary ones (`v_add_f16_e32 v1.l, 0, v1.h`, `v_fma_f16 v2.l, 0, v2.h, v1.h`,
`v_fmamk_f16 v2.h, v2.h, 0x8000, v1.h` — the `fmamk` literal is the compiler's folded `-0.0` multiplicand).

**Section 2 — the hand-built `v_wmma_f16_16x16x16_f16_w32` tile.** 64 dot positions = 2 issue shapes ×
2 operand sides × 16 element indices, one signed zero at a time with every other byte fixed, `C` a known
non-zero value (`0x3ECD+e/4`, the datum's own value family), plus the F32-accumulator sibling:
**all 64 `zerosign` measures: no difference**, F32-D likewise. The dot position is real and live —
`slotlive` (the slot restored to its own live value) fired **64/64**, and `livedouble` fired 64/64.

Why element index `p` IS dot position `p`, so no fragment model was needed: `attn_fattn_kernels.h:469`
(`fattn_ab_row(lane) = lane & 15`, "lane t holds the 16 contiguous halves of row t%16, the run along the dot
index") plus the C->P exchange (`:527-537`, `attn_wide_llama_kernels.h:259-266`), which writes C's row's 16
columns contiguously into `p[L][0..15]`.

**The two-issue shape had to be rebuilt twice before it was admissible, and both failures are worth the
record**, because each produced a confident wrong result first:

1. Reusing ONE A tuple for both `op_sel` issues (`wmma(A,B,C,0)` then `wmma(A,B,D,1)`) is **not the engine's
   shape**. The engine's own code object (`tmp/mtp_verify/causal_bound-…-gfx1151.s:24250-24268`) emits
   `v_wmma_f16_16x16x16_f16 v[57:64], v[105:112], v[96:103], v[57:64]` then
   `v_wmma_f16_16x16x16_f16 v[57:64], v[113:120], v[96:103], v[57:64] op_sel:[0,0,1]` — **two different A
   tuples** (`vlo`, `vhi`), the same B, the same tuple as both C and D. With one A tuple reused, the *clean*
   tile produced NaN by itself and 14 `zerosign` cases "found" 1-ULP sign effects; with two A tuples the clean
   tile is NaN-free and every `zerosign` is null. The probe's own `fid` control (clean tile, both issue
   shapes, must be non-zero and NaN-free) is what caught it.
2. A `TileIn` of four 1 KB arrays **passed by value** is exactly the 4 KB kernarg budget, and it leaked three
   NaN cells into a clean tile too. Now passed by global pointer.
   Both are the same failure family this file already knows: a rig effect nobody can see becomes a finding.

**Two further measured facts from the same run:**
* `underfloor`: replacing a live k-term by the F16 min subnormal changes D at only 8 of 64 dot positions, and
  `liveulp` (one live term moved by 1 F16 ULP) fires at 56 of 64 — a 1-ULP change at one of 16 addends is
  frequently below what a 16-term F16 dot resolves. Priced here so nobody re-derives it as a surprise.
* `zsum`: with `C = 0` and products arranged to cancel exactly (8x+1, 8x-1) and the straddling pair replaced
  by two zeros, **both the all-`+0` and the all-`-0` arm publish `+0` in all 512 D cells**. That is IEEE's own
  rule (`(+0) + (-0) = +0` under round-to-nearest), and it closes the door on "the row's output element is
  exactly zero and its sign is what flips" — that route is not reachable by feeding signed zeros.

## The datum's bits, re-read (this is what the null has to be squared with)

`prototype/mtp_verify/packab_bits.py` diffs two `pack_ab` output tiles and prints the **bits** of the movers,
through `prototype/bf16_ulp.py`'s `monotone_key` (imported, not retyped). On the surviving p128 dump
(`/tmp/cr_fix/dump-refwide`, rig `fid` canary BYTE-IDENTICAL, `widen_*` controls 3/3 fired):

| comparison (V of lane 9, dim 62) | row 7 (bound 8, even) | row 8 (bound 9, odd) |
|---|---|---|
| `v62_zero` -> `v62_m1e12` (staged `+0` -> `-0`) | 0 movers | **1 mover: head 4 dim 62 `0x3ecd -> 0x3ecc`, 1 BF16 ULP** |

`pack_ab`'s own CASE lines agree across the whole ladder: every negative value (`-1e-12`, `-1e-8`, `-1e-6`,
`-1e-4`, `-1e-2`, `-0.0992`, `-1`, `-10`, `-100`, `-1000`) gives `row_at_moved=1`, every non-negative one
(`+0`, `+1`, `+10`, `+100`, `+1000`) gives `row_at_moved=0`, and `+1e6` (the `0 x inf` case) is the one arm
that also moves rows *below* the adjudicated row. So:

* the datum is **real** and reproduces outside the engine, with the canary green;
* the mover is a **non-zero** value (`0x3ecd = +0.400390625`), one BF16 ULP, i.e. the F16 accumulator
  underneath it moved by about one F16 ULP and crossed a BF16 rounding boundary. It is **not** a `+0`/`-0`
  sign flip (the probe counts those explicitly and found none).

## What this leaves open — and why the prescribed fix was not written

`ISOLATED`'s sign reading and this probe's null are in direct tension, and the tension is now localised:

* In the shipped kernel, flipping the staged V cell of a disallowed lane from `+0` to `-0` moves the live
  row's non-zero output element by one BF16 ULP. Whatever the hardware does with that k-term, its sign
  survives.
* In a hand-built `v_wmma_f16_16x16x16_f16` — single issue, and the engine's faithful two-issue
  `op_sel:[0,0,1]` shape, both operand sides, 64 dot positions, with controls that prove the slot is live and
  that the rig resolves 1-ULP changes — **the sign of a zero addend never survives**.

One of the two models of the engine's k-term is therefore wrong, and the one that can be wrong is mine: the
chain from `vcache` cell to `A` fragment to `D` cell is the part neither measurement can see, because both
read the same shipped round from outside. Naming the expression needs the kernel's own intermediates on real
content (the copied-round option `FIX-CROSSROW` §"What is left" already weighed and distrusted, or a
`QWEN_WIDE_DEBUG`-style instrument inside the shipping header), and that is a larger, different change than
the funded one.

**Why the funded edit is not written.** Step 2's prescription was "force the masked cell's staged P value to a
positive zero". Three facts kill it as *the* fix: (a) the masked P cell is **already** f16 `+0.0`
(`expf(-1e30f - mn)` then `__float2half_rn`, `attn_wide_llama_kernels.h:260-264`; `FIX-CROSSROW` Result 2 and
`ISOLATED` arm B both confirmed the reduction path is clean at a 4096x amplifier), so there is nothing to
force; (b) the sign that demonstrably arrives comes from the **V** side, and a product's sign is the XOR of
its operands' signs, so forcing P positive cannot remove it; (c) the instruction-level premise — that a `-0`
adend can move a non-zero F16 accumulator — is what section 1 and section 2 together measured false. Shipping
it would put a hot-path change into the shipping kernels that cannot be the cause-slayer, and would let the
defect be recorded as fixed while it is not.

## Hazard, recorded not fixed: `0 x inf` forms NaN in the live row

`ISOLATED` arm A reached it at `V_dead = +1e6`: the staged V is F16, `+1e6` overflows to `inf`, the dead
k-term is `0 x inf = NaN`, and the NaN appears in the **live** row's output. Unreachable today — the fp16
range census (`tools/fp16_range_probe.py`, decision D14, recorded in `AGENTS.md` "Numeric format") measures
activation peaks of 82.5 against F16's 65504, a 794x margin — so no engine change is warranted. It becomes a
live defect for any model whose attention output or V magnitude can exceed 65504/… in F16, i.e. two orders of
magnitude of headroom above anything this checkpoint produces. Recorded here so the next person does not
rediscover it through a model with bigger activations.

## Hazard, recorded not fixed: `0 x inf` forms NaN in the live row

`ISOLATED` arm A reached it at `V_dead = +1e6`: the staged V is F16, `+1e6` overflows to `inf`, the dead
k-term is `0 x inf = NaN`, and the NaN appears in the **live** row's output. Now witnessed twice more:
`wmma_sign --section 2`'s `infnan` control (64/64: P slot `+0` and V slot `+-inf` -> NaN in D, and note the
first version of that control tested one operand side alone, which gives `+-inf`, not NaN), and `pack_ab`'s
`v62_p1e6` case, which is the only arm in the ladder that moves rows *below* the adjudicated row.

Unreachable today — the fp16 range census (`tools/fp16_range_probe.py`, decision D14, recorded in `AGENTS.md`
"Numeric format") measures activation peaks of 82.5 against F16's 65504, a 794x margin — so no engine change
is warranted. It becomes a live defect for any model whose V magnitudes can approach F16 overflow. Recorded
here so the next person does not rediscover it through a model with bigger activations.

## Tree state

**No engine file, gate, floor, pin, ledger, limits, README or progress doc was touched, and nothing is
committed.** `git status --short` shows exactly four untracked additions, all under `prototype/mtp_verify/`:
`FIX-ZERO.md` (this file), `wmma_sign/` (the step-1 probe + its build script), `zero_gate.py` (the step-3
gate) and `packab_bits.py` (the bit-level tile diff). Because no engine byte changed, the pinned gates cannot
have moved; `engine/gate.py` was therefore not re-run, per the budget rule about re-making a result nothing
could have changed.

## Device runs for this pass (correctness-only; **no timing taken**; `tools/gpu_lock.sh` held by the caller; wall = whole command)

| command | wall | `util` at acquire |
|---|---|---|
| `wmma_sign --section 1` (200 measures, 2 controls) | 1 s | 33 % |
| `wmma_sign --section 2` (194 controls, 195 measures, 64 dot positions) | 1 s | 91 % |
| `pack_ab --dump /tmp/cr_fix/dump-refwide --row 8 --out /tmp/packab/r6` (36 tiles, canary + 3 widen controls + 16 v62 cases) | <1 s | 89 % |
| earlier iterations of the same two, as the controls were repaired | <1 s each | 33-92 % |

`util` reads high because this session serves its own tokens on the same iGPU. Every figure above is a
whole-command wall; no kernel timing, burst, best-of-N or sweep appears anywhere in this file.

`packab_bits.py` and the `report.py` diffs are CPU-only.
