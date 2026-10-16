# `prototype/attn_valu_slim` — counting the per-touch instruction bill, before and after

D20 item 4 funded this route (`progress/2026-10-16/prefill-decision-log-2026-10-16.md`):

> "the causal-mask guards cost 20 VALU instructions per touch — more than the dot they protect — and
> the bf16→f32 conversions 8 more. Together 24 % of the stream, in control flow and format handling,
> neither of which needs a numerics decision to touch: applying the causal bound as a loop limit
> instead of a per-element guard is bit-identity preserving by construction."

D19 asked for it to be counted in the disassembly, not modelled. It is counted, in
`isa_census.output.txt` (regenerate: `.venv/bin/python prototype/attn_valu_slim/isa_census.py`).

**Headline.** The 20 VALU are real but they are not causal-mask guards, and the 8 conversions are not
removable at all. What *is* removable is a different pile, about the same size: **−27 VALU warp-inst
per (row, lane, head) touch, −7 LDS, −37 SALU** (55 → 28 VALU on the pass-1 touch loop), for
**≤ 23.6 % of the wave** by D20's own framing B — **≤ 0.60 ms of the m2 form's 2.5402 ms at 4096
lanes**, with a floor of zero unless a paired device A/B says otherwise. No numerics decision is
required: both levers claim bit-identity, and `touch_probe.py` passes on the CPU for both.

---

## 1. R1 — the named guard is not a causal mask, and the shipped form has no per-element causal guard

`attn_tile_body` (`prototype/attn_decode/attn_decode_kernels.h:546`, copied verbatim here as MODE 0)
bounds its score walk with:

```c
const int ns = bhi - blo;                     // lanes this block owns of this row
for (int s_lo = 0; s_lo < ns; s_lo += ATTN_TILE_L) { ...
    for (int rr = s_lo; rr < s_hi; ++rr) {    // <- the causal bound, already a loop limit
```

`[blo, bhi)` is the row's own causal slice (`lanes_of` → `span` → `blocks_for_row`, the shipped
form's rule), so **there is no per-element causal guard in the form the 114.6 census was taken on,
and nothing to convert into a loop bound.** The form that *does* carry a per-element bound guard is
m2 (`prototype/attn_m2/attn_m2_kernels.h`, `if (j < g.blo || j >= g.bhi) continue;`), because there
the walk is over the union of a row *pair*'s slices and each warp must skip lanes outside its own
row. That guard is 3 `v_cmp`/`v_cndmask`-class ops and it is the only honest referent for D20's
sentence — worth ≈ 3 of the 20, not 20.

The other 17–20 are the per-step machinery of `warp_sum`'s `__shfl_down_sync`, which D20 re-labelled
(the figure comes from `prototype/attn_score_dot2/results.md` §2.3, whose own heading says "the
reduction is bigger than the dot it reduces" and whose ops are `s_cmp_eq_u32 exec_lo, -1`,
`v_cndmask_b32_e64 v11, 0, 8, vcc_lo`, `v_add_lshl_u32` — shuffle-lowering, not masking). The census
below separates them so the two cannot be conflated again.

## 2. What one touch costs before, and where each instruction lives

`isa_census.py` (§5 of its output prints every rule and its example; **each rule is selftested
against its own example at startup** — the first version's space-escape was broken and four rules
matched nothing while printing a tidy table, which is `prototype/gdn_collapse/run.py`'s bug one layer
up). The touch is the depth-2 pass-1 loop, LLVM's never-taken `%Flow` blocks excluded, the rolled
score-dot child counted `nd = 8` times:

| form | slots/touch | VALU/touch | LDS/touch | SALU/touch |
|---|---|---|---|---|
| `attn_tile_wave` (shipped) | 116 | 55 | 14 | 47 |
| `attn_slim0_wave` (shipped body, copied) | **116** | **55** | **14** | **47** |
| `attn_slim3_wave` (fold lever only) | 96 | 43 | 14 | 39 |
| `attn_slim2_wave` (dot lever only) | 65 | 40 | 7 | 18 |
| `attn_slim1_wave` (both levers) | **45** | **28** | **7** | **10** |
| `attn_m2_wave` (m2 form, as it stands) | 124 | 55 | 14 | 55 |

The second row is the control: **the copy is instruction-for-instruction identical to the shipped
kernel** (whole-kernel static too: 1454 ops / 1416 slots / 908 VALU / 17 LDS / 475 SALU, and VGPR 47
/ LDS 12,288 B for both), so every delta below is the lever and not the copy.

The last row is the transfer argument: **m2's touch is 55 VALU/touch too**, with the same composition
(`dot.q-indexed` 8, `dot.lds-load` 8, `fold.exec-guard` 16, `fold.addr-math` 4) and 8 *more* SALU
than the ship form (its per-element pair-walk guard). The two levers apply to it unchanged; they
were implemented here, not there, because the A/B has to be single-variable and the ship form is what
the census baseline belongs to.

Category detail (instructions per touch; `*` marks D20's two named items):

| category | shipped | slim1 | what it is |
|---|---|---|---|
| `*dot.bf16-conv` | 9 | 8 | widen BF16 to F32: `v_lshlrev_b32 …,16,` / `v_and_b32 …,0xffff0000,` (+`v_mov_b16` on the unrolled side) |
| `dot.fma` | 8 | 8 | the eight products, serial in `t`, one accumulator |
| `dot.lds-load` | 8 | **1** | the staged K row: eight `ds_load_u16` → one `ds_load_b128` |
| `dot.q-indexed` | 8 | **0** | `v_movrels_b32_e32` — `q[t]` read through VGPR-relative addressing because `t` was a runtime index |
| `*fold.exec-guard` | 16 | 2 | `s_cmp_eq_u32 exec_lo, -1` + `s_cbranch_scc0` per shuffle step ("is exec the mask?") |
| `*fold.range-select` | 7 | **0** | `v_cmp_gt_u32_e32 vcc_lo, 24/28/30,` / `v_cndmask_b32_e64 v, 0, δ,` |
| `*fold.addr-math` | 4 | **0** | `v_add_lshl_u32` / `v_add_co_ci_u32_e64` — the permute address, per touch |
| `fold.permute` | 5 | 5 | the five LDS permutes themselves — **not removable** |
| `fold.add` | 4 | 4 | the five adds (one rides a `v_dual` slot) |
| `score.bf16-round*` | 3 | 3 | `fb(bf(fb(dot)) * SCALE)`, NaN-tested, under divergent `if (lane == 0)` |
| `wait.lgkmcnt` | 13 | 6 | one per rolled child iteration in the shipped form |

## 3. The two levers, and their honest labels

**Lever 0 — the score dot's trip count is runtime, and that is what the dot costs.** `nd` is
`(a.neg == NEG_DIMHALF) ? 4 : 8`, a runtime value, so LLVM keeps the dot *rolled*: per element a
scalar `ds_load_u16` of the staged K, a `v_movrels_b32_e32` for the dynamically indexed `q[]`, one
conversion, one FMA, and loop bookkeeping with an `s_waitcnt lgkmcnt(0)`. The PV loop two lines away,
whose trip count is the literal 8, compiles as one `ds_load_b128` plus eight unrolled convert+FMA
pairs that dual-issue. The slim spelling gives the score dot that shape and moves `NEG_DIMHALF` into
a second copy of the loop behind a uniform branch, so the injection keeps its exact 4-term semantics
without its trip count deciding the hot path's shape. Cost: the loop is emitted twice (+111 static
ops), SGPRs 52 → 54, **VGPRs unchanged at 47, LDS unchanged at 12,288 B, occupancy unchanged at 16
waves/SIMD** — so unlike dot2 this lever cannot cost occupancy, which is the binding constraint on
this kernel (D19: 2 blocks/CU is what m2 pays 22,656 B of LDS for).

**Lever 1 — the fold's guards are loop-invariant and were hoisted.** Four of `warp_sum`'s five steps
lower to `ds_bpermute_b32` plus a participation select, a permute-address add, a range compare, and
an `exec == mask` guard with a slow-path branch. Every one of those four quantities is a function of
`lane` and the delta only, so they are computed once before the pass-1 loops and the permute is
spelled once as inline `ds_bpermute_b32`. The clamp is LLVM's own (`δ → 0` out of range, i.e. the
lane reads itself), which is why the claim is "bit-identical in every lane" rather than the weaker
"bit-identical at lane 0, which is all the consumer reads". The `s_waitcnt lgkmcnt(0)` is written
inside the asm block, because `SIInsertWaitcnts` tracks the instructions it recognises and an inline
DS op is not one of them.

**What neither lever touches, stated plainly:**

* **The conversions stay.** D20's "the bf16→f32 conversions 8 more" cannot be removed by any
  spelling that keeps the same values: two BF16 operands cannot multiply without being widened, and
  the only instructions that widen *for free* are `v_dot2_f32_bf16` (rejected in D20 precisely
  because it would spend m2's bit-identity admission) and WMMA. The unrolled form spends about the
  same number of widening ops (9 → 8 in the census: a low half now needs `v_mov_b16` + `lshlrev`
  where a scalar `u16` load needed a single shift) and gets them into fewer *slots* by dual-issuing
  them with the FMAs they feed — visible in the listing as
  `v_dual_fmac_f32 v2, v11, v3 :: v_dual_and_b32 v3, 0xffff0000, v37`. Any report that said "8
  conversions removed" would be false.
* **The five permutes and the five adds stay.** They are the reduction.
* **Pass 2 is untouched** — the V conversions, the eight PV FMAs, `expf`, the rescale. That is most
  of the 72.6-of-114.6 that dot2 counted "by subtraction", and it is why the whole-kernel delta is
  much smaller than the per-touch delta: whole-kernel static ops go 1454 → 1565 (up, because the
  score loop is now emitted twice unrolled), while the per-touch VALU bill goes 55 → 28.
* **The `ds_swizzle_b32` at delta 16 stays**, in both MODEs: there the compiler already emits an
  immediate permute with no address math and no guard, so hand-rolling could only lose.

## 4. The price, in D20's units and by D20's method

Inputs, each from the tree: `attn_tile_wave` = 114.6 VALU warp-inst per touch
(`prototype/attn_score_dot2/results.md` §2.4, from `rocprofv3 SQ_INSTS_VALU`); wave times from D19's
`attn-m2-2` table; method = framing B (instruction share × measured wave time).

| lever | Δ VALU/touch | share of 114.6 | ≤ ms, m2 2.5402 @ 4096 | ≤ ms, m2 0.6511 @ 1024 |
|---|---|---|---|---|
| fold only | −12 | 10.5 % | 0.266 | 0.068 |
| dot only | −15 | 13.1 % | 0.332 | 0.085 |
| **both** | **−27** | **23.6 %** | **0.598** | **0.153** |

They compose additively (−12 and −15 individually, −27 together), which is the check that the two are
independent levers and not one lever counted twice.

Three honest qualifications, in the order a reader will ask for them:

1. **This is an upper bound, same as dot2's framing B.** The kernel sustains ~45.8 % of the 2-wide
   issue roof (D20 §0), so a freed slot that was not on the critical path saves nothing. The floor of
   the honest bracket is **0 ms**.
2. **But the evidence in the tree leans toward the upper half of the bracket, not the lower.** dot2's
   own calibration: pricing the shipped wave's whole VALU stream at the measured achieved rate
   predicts the measured wave times to within 7 % (2.2495 predicted vs 2.4163 ms at 4096). And D19
   has m2 at 4096 running 105 GB/s against a 220 GB/s plateau, i.e. not traffic-bound. Both say the
   time is substantially instruction time, which is what this lever removes.
3. **The units do not line up exactly and the gap is not hidden.** 114.6 is a *dynamic VALU-op*
   counter over the whole kernel; −27 is a *static* count over the pass-1 touch loop, and a `v_dual`
   packet is one slot for two VALU ops. The slot view is more favourable than the op view (116 → 45
   slots/touch is −61 %, vs −49 % on VALU ops), so the quoted 23.6 % uses the less favourable of the
   two, deliberately. The number that settles it is a paired device `--burst` A/B, which a worker may
   not take: `HANDOFF.md` step 6.

Against D20's ~15 % kill line for dot2: the pair clears it at the bound (23.6 %), the dot lever alone
almost does (13.1 %), the fold lever alone does not (10.5 %) — **the fold lever alone buys exactly
what dot2 would have bought, and costs no bit-identity admission to do it.** That is the most useful
single sentence in this file.

## 5. Risks this file does not retire

* **The inline `ds_bpermute_b32`.** If the hand-written `s_waitcnt` is wrong or insufficient the
  failure mode is *nondeterminism*, not a wrong-but-stable number. Hence `HANDOFF.md` requires the
  clean A/B twice and compares the two runs.
* **Uniformity.** The guards that were removed are what made the shuffle safe under divergence. The
  loops they sit in are warp-uniform (`s_lo`/`s_hi`/`ns` are per-block values, and the early `return`
  is uniform), and the shipped `__syncthreads()`-discipline comment in `attn_tile_body` says the same.
  But this is an argument, not a proof; the device A/B is the proof, on every plan and every
  injection, including the multi-row waves where the geometry is worst.
* **A second copy of the loop is real code growth.** +111 static instructions, +2 SGPRs. If a future
  width makes the kernel I-cache- or SGPR-limited, this is where it came from.
* **Nothing here was run on a device.** No `ndiff` has been observed yet; §2's numbers are all static
  from `hipcc -O3 --offload-arch=gfx1151`, and `touch_probe.py` is CPU-only and explicitly cannot see
  FMA contraction (`AGENTS.md`: "a hoisted scalar constant is a different number, and a CPU mirror is
  blind to it").

## 6. Evidence files

| file | what it is |
|---|---|
| `attn_valu_slim_kernels.h` | one body, four MODEs (0 shipped copy / 1 both / 2 dot / 3 fold) |
| `attn_valu_slim.hip.cpp`, `build.sh` | the device A/B harness (builds; never run by the author) |
| `isa_census.py`, `isa_census.output.txt` | the census + its selftested rule table |
| `touch_probe.py` | CPU bit-identity for both levers, with derived must-catch / must-not-catch sets |
| `HANDOFF.md` | the device commands, for the main agent, with the lock protocol |

---

## 7. Device pass (main agent, 2026-10-16) — bit-identical, and **1.52–1.57× on the shipped form**

Three windows, each blocking with zero subagents active, `util` recorded and then discounted (it read
91–93 % while the box was exclusive — it has never once been a usable gate on this machine).

**Control re-taken before anything else:** `attn_slim0_wave` is instruction-identical to
`attn_tile_wave` (VGPR 47, SGPR 52, LDS 12,288 B, whole-kernel 1454/1416/908/17/10/475), and
`touch_probe.py` prints `PROBE summary PASS` — including negative controls that catch 2002 of 2009
fold must-catches and 1000/1000 dot must-catches with no must-not-catch violation.

| check | result |
|---|---|
| bit-identity, `--slim 1`, all 12 plans | **`ndiff_aout=0 ndiff_gated=0` on all 12**, `cases_with_any_difference=0` |
| run 1 vs run 2 diff (the hand-written `s_waitcnt` race check) | **byte-identical logs** — the inline `ds_bpermute_b32` is not racy as written |
| injections, 12 combinations × 3 geometries (`pair-at-maxsc`, `straddles-block-3`, `wave64-at-4096`) = 36 runs | **0 disagreements**, including `dimhalf` — the injection that makes `nd` runtime and motivated lever 0 |

Full 13-injection × 12-plan matrix was NOT run; 36 of 156 cells, chosen to cover the boundary geometry,
the funded width, and the real pinned row.

### 7.1 The timing, paired inside one lock hold

`--burst N --reps 5`, `steady_clock`, both forms alternating within each rep, warm pass first, best-of.

| wave | ship | slim | **speedup** | ship demanded GB/s | slim demanded GB/s |
|---|---|---|---|---|---|
| `wave64-at-1024` | 0.5557 ms | 0.3617 ms | **1.536** | 248.3 | 381.5 |
| `wave64-at-2048` | 1.2100 | 0.7822 | **1.547** | 225.0 | 348.0 |
| `wave64-at-4096` | 2.3446 | 1.5416 | **1.521** | 227.2 | 345.6 |
| `wave16-at-4096` | 0.7068 | 0.4497 | **1.572** | 189.6 | 298.0 |

This is a bigger result than §4 priced, and it breaks a reading the repo has held for two days. §4
predicted ≤ 23.6 % **on the m2 form** and explicitly said a flat ship-side result would be consistent
with D19. Instead the ship form — the one D19 called "at the memory plateau" — got **1.52×**.

**The byte-wall reading is falsified by these four rows.** The demanded-bytes census is unchanged by
these levers (no traffic is removed), so the slim side of the 4096 row demands 532.7 MB in 1.5416 ms =
**345 GB/s**, and the 1024 row demands **381 GB/s** — both *above* the 237 GB/s DRAM read ceiling. A
kernel cannot demand more from DRAM than DRAM can deliver, so the census never described DRAM traffic:
L1/L2 absorb most of the reuse, and `bytes-demanded ÷ time` landing near the plateau at 225–248 GB/s was
coincidence. The wall was instructions. The evidence is the change itself: −61 % of issue slots per
touch bought −34 % of time, which is what an instruction-bound kernel does and what a bandwidth-bound
kernel cannot do.

Consequences, in order of what they cost to act on:

1. **This lever ships ahead of everything else today.** Bit-identical, no numerics band, no occupancy
   cost (VGPR 47→47, LDS unchanged), and it is a win on the *shipping* form, so the engine port is a
   same-shape transplant into `engine/kernels_attn.cpp`. Attn is 412.07 ms of a 1307.75 ms wave at 4096
   lanes; 1.52× there is on the order of 140 ms of the wave.
2. **D19 needs correcting, not just annotating.** "Two walls of equal height" was inferred from m2
   halving bytes without moving time; the correct inference is that m2's residual was *instruction and
   latency* bound and the shipped form was too — the byte wall was never at the surface. Recorded as
   D21.
3. **M = 2 reopens on different terms.** Its traffic halving was measured real; what it lacked was a
   low enough arithmetic wall to expose. With the touch bill at 45 slots instead of 116, the pair is
   worth re-measuring, and the order matters: port the levers first, re-test M = 2 second.
