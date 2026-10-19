# A4-L1 — Gate 0 (is the shipped score roll a dependent chain?) and Gate 1 (two accumulators, compile-and-read)

Written 2026-10-19 on tip `9a19cf5`, against `prototype/attn_percost/A4-BET.md` (the gates) and
`A4-CANDIDATES.md` §1/§1b/§2-L1, `A2-census.md` §2, `A3-result.md` §2b. **CPU-only: no device was touched, no
gate was run, no timing is taken or quoted as mine.** Every ISA fact below is a *read* with a listing address;
every ratio is *derived* with its formula; every number from another file is *quoted* with its path.

## 0. Verdicts

* **Gate 0: `L1 PREMISE HOLDS`.** The shipped score roll emits as **one `v_wmma` site inside a rolled loop of
  16 trips, whose destination tuple is also its accumulator operand** — a true loop-carried recurrence on a
  single 8-VGPR accumulator, with no independent matrix work and no software pipelining anywhere in the body.
  Evidence: §1.
* **Gate 1: `L1 SURVIVES THE COMPILE` — both admission lines pass and the two chains are really there.**
  `vgpr_spill_count 0` **and** `private_segment_fixed_size 0` on the variant (so §33's spilled-at-8 precedent does
  not kill it), `group_segment_fixed_size` **unchanged at 32,768 B**, `vgpr_count` **178 → 200** (+22), and the
  emitted roll loop now holds **2 `v_wmma` sites on 2 distinct accumulator tuples**. Evidence: §2, §3.
* Neither gate killed L1, so **Gate 2 (the device A/B) is unblocked but not run** — this worker touched no
  device. What it would need is in §5.

One correction to the premise as written, up front because it changes what L1 can buy: A4-BET says the roll is
"16-long dependent WMMA chain", which the emitted code confirms, but A2 §2's parenthetical guessed the 17 sites
decompose as *"16 unrolled score tiles plus one rolled P·V loop"*. **The polarity is reversed**: the score leg
is the rolled one (1 site × 16 trips) and the P·V leg is the fully-unrolled one (16 sites, 16 distinct
accumulator tuples). §1.1 and §1.2 witness that from the register operands and the loop body, and
`a2_round_bill.py`'s histogram of the 11-instruction loop body is the third witness. A2 §2 already labelled that sentence an inference
("an inference from the source's `FATTN_KTILES`/`FATTN_VTILES` loops against the object's count"), so the count
of 17 was right and only the attribution was backwards.

## 0b. Listing provenance, and why a compile-only census build is admissible here

`g3_isa_bill.py`'s docstring demands the code object carved from the shipping binary rather than a
`--save-temps` re-build, so the substitution is witnessed rather than assumed:

```
$ bash prototype/attn_wmma/fattn_census_build.sh /tmp/a4l1            # hipcc -O3 --save-temps -c engine/kernels_attn.cpp, 1.44 s, launches nothing
```

* `attn_fattn_wave` in this listing starts at `0xda00`; the same function in A2's carved shipping object
  (`tmp/a2census/ours/co_335872.s`, read: `grep` of the `FUNC` lines) starts at `0x10e00`. Base delta `0x3400`,
  so A2's cited round span `0x11520..0x13970` maps onto `0xe120..0x10570` here — the same span
  (`0x13970-0x11520 = 0x2450 = 0x10570-0xe120`, read).
* Addresses stripped, the two bodies are **byte-identical**: 2,175 instructions each,
  `diff -q` clean (`/tmp/a4l1/ship_body.txt` vs `/tmp/a4l1/carved_body.txt`, read). So every address cited below
  describes the shipped binary, in two coordinate systems: add `0x3400` for A2's carved object.
* Note block of the shipped side, read with `prototype/attn_wmma/fattn_residency_read.py --asm
  /tmp/a4l1/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s --kernel attn_fattn_wave --threads 96 --warps 3`:
  `vgpr_count 178`, `vgpr_spill_count 0`, `sgpr_count 24`, `sgpr_spill_count 0`,
  `private_segment_fixed_size 0`, `group_segment_fixed_size 32768`, derived **2 blocks/CU = 6 wf/CU, LDS-bound**.
  This reproduces `A3-result.md` §0 and §33's banner exactly (quoted), which is what makes §1's ISA admissible
  as a statement about the shipping path.

## 1. GATE 0 — the score roll is a chained single accumulator, not a pipelined one

### 1.1 The 17 matrix sites and their accumulator tuples (read)

`grep -c v_wmma` on the whole TU = 17, all inside `attn_fattn_wave` (read). Every one of the 17 has
**destination tuple == 4th operand (the accumulator in)** — i.e. none of them was renamed into a fresh tuple.
The decomposition:

| where | address (this listing) | dst tuple | operands (A, B) | reading |
|---|---|---|---|---|
| **score roll**, 1 site, inside a rolled loop | `0xe648` | `v[129:136]` | `v[159:166]`, `v[167:174]` | the loop-carried accumulator |
| P·V leg, 16 sites, fully unrolled | `0xf684`, `0xf6ec`, `0xf754`, `0xf7bc`, `0xf824`, `0xf85c`, `0xf878`, `0xf894`, `0xf8b0`, `0xf8cc`, `0xf8e8`, `0xf904`, `0xf920`, `0xf93c`, `0xf958`, `0xf974` | 16 **distinct** tuples: `v[121:128] v[105:112] v[57:64] v[33:40] v[25:32] v[17:24] v[9:16] v[1:8] v[113:120] v[97:104] v[89:96] v[81:88] v[73:80] v[65:72] v[49:56] v[41:48]` | all read `v[168:175]` (V tile) and `v[129:136]` (P) | the 16 `out.t[]` accumulators, 128 VGPR/lane (quoted: `engine/attn_fattn_kernels.h:189`) |

16 distinct destination tuples for the P·V sites = `FattnO t[FATTN_VTILES]`, and their accumulator inputs are
their own destinations — so the P·V accumulators are independent *of each other within a round* and carried
across rounds. The score roll has **one** tuple, `v[129:136]`, and it is the same tuple the P operand later
occupies (read: `v[129:136]` appears as the P·V B-operand), which is the register allocator reusing the dead
score accumulator.

### 1.2 The roll is rolled in the emitted code, and it is a 16-trip loop on that tuple (read)

`engine/attn_fattn_kernels.h:576-583` is `fattn_CRegs acc{};` + `#pragma unroll 1` over `t < FATTN_KTILES`.
The emitted block, verbatim from `/tmp/a4l1/ship-objdump.s` (addresses are absolute in this listing;
subtract `0xda00` for function-relative, add `0x3400` for A2's carved object):

```
0xE5E0  v_mov_b32_e32   v129, 0                     ; acc{} — the accumulator is zeroed
0xE5E4  s_movk_i32      s4, 0xfe00                  ; trip counter = -512
0xE5E8  s_waitcnt       lgkmcnt(0)
0xE5EC  s_barrier                                   ; barrier 3 (`:560`, K published)
0xE5F0  buffer_gl0_inv
0xE5F8  v_mov_b32_e32   v130, v129                  ; acc[1..7] = 0  (through 0xE610)
0xE614  v_add_nc_u32_e32 v163, s4, v150             ; <-- LOOP HEADER = function+0xc14
0xE618  v_add_nc_u32_e32 v171, s4, v155             ; second operand's tile address
0xE61C  s_add_i32       s4, s4, 32                  ; 32 B = one 16-half k-tile
0xE620  ds_load_b128    v[159:162], v163 offset:25088
0xE628  ds_load_b128    v[163:166], v163 offset:25104
0xE630  ds_load_b128    v[167:170], v171 offset:512
0xE638  ds_load_b128    v[171:174], v171 offset:528
0xE640  s_cmp_eq_u32    s4, 0                       ; -512 + 16*32 = 0  => 16 trips
0xE644  s_waitcnt       lgkmcnt(0)                  ; this trip's LDS fetch, waited on
0xE648  v_wmma_f32_16x16x16_f16 v[129:136], v[159:166], v[167:174], v[129:136]
0xE650  s_cbranch_scc0  -> <attn_fattn_wave+0xc14>   ; back edge to 0xE614
0xE654  v_dual_mov_b32 ...                          ; falls through into the fold/exp at `:588+`
```

The four facts that decide the gate:

1. **One matrix site.** The loop body `0xE614..0xE654` contains exactly one `v_wmma`. `a2_round_bill.py`
   (the existing span histogram — reused, not re-derived) over that span:
   `# loop E614..E650: 11 instructions — wmma 1 (9.1 %), lds 4 (36.4 %), wait 1, salu 3, int_vec 2;
   matrix-class slots 1 of 11 = 9.1 %; non-matrix slots per matrix slot = 10.00` (read).
   This is the third witness that the loop was **not** unrolled: 11 instructions cannot hold 16 tiles.
2. **The accumulator is the loop's carried value and there is exactly one of it.** Destination `v[129:136]`
   **is** the 4th operand; the tuple is written only by this site inside the loop and is zeroed once in the
   preheader (`0xE5E0..0xE610`). Two accumulators would require two tuples both live across the back edge, and
   there is one.
3. **Nothing independent is inside the loop to hide the latency.** Of the 10 non-matrix slots, 4 are the operand
   LDS loads this very WMMA consumes, 1 is the `s_waitcnt` that waits for them, and 5 are loop control
   (2 address adds, `s_add_i32`, `s_cmp_eq_u32`, `s_cbranch`). There is no second matrix op, no other tile's
   load, no float work between trips. Contrast §1.4: the P·V leg *does* have independent work interleaved.
4. **Trip count is 16, matching `FATTN_KTILES = DH/16 = 16`** (`:166`, quoted): `s4 = 0xfe00 = -512`,
   `+32` per trip (32 B = 16 halves), exit at `s4 == 0` (read).

So the runtime stream in the score leg is 16 × (LDS fetch → `v_wmma` on the same accumulator), and the 16-deep
WMMA→WMMA dependency is real in the emitted code. **LLVM has not software-pipelined, unrolled, or replicated the
accumulator.** Gate 0 passes.

### 1.3 How I distinguished "chained" from "already pipelined", and what the gate cannot see

The discriminator is **register identity across the back edge**, not schedule shape: a pipelined or
multi-accumulator form must present ≥2 distinct accumulator tuples live across the loop's back edge (or, in
straight-line code, a WMMA whose accumulator operand was written by a *non-predecessor* WMMA). What the listing
shows instead is one tuple (`v[129:136]`) written and read by the same site, one site in the body, and the
back edge immediately after it. A `v_wmma` with `dst == acc-in` cannot have its successor overlapped with it by
any transformation short of adding a second accumulator, so no scheduling interpretation of the same listing
could show "already pipelined". Method notes: the brief named `prototype/gdn_rec_isa/token_bill.py`'s loop-annotation method, and its traps
(function labels vs `.LBB` "in Loop" annotations, `v_fmac_f32_e32` not `v_fma_f32`) are why the loop structure
here was read off LLVM's/objdump's own annotations rather than a heuristic. It could not be run as-is: it is
hard-wired to GDN's two state walks, and against this listing it exits 1 with
`.LBB6_65: 33 state accesses/trip does not divide D=128` (read: `token_bill.py --asm
/tmp/a4l1/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s --kernels engine::kernels::attn_fattn_wave`; note the
symbol is `engine::kernels::attn_fattn_wave` in the `.s`, which is itself a trap for anyone passing
`attn_fattn_wave`). It does not apply to this kernel, so the pieces that *do* apply were used instead: The
reused pieces that *do* apply are `g3_isa_bill.py` (loop spans, and it found this round loop:
`loop 0xe120..0x10570 span=9292 B per-iteration ISA: {total 1653, lds 158, barrier 11, fma 16, exp 9, store 33}`,
read) and `a2_round_bill.py` (histogram of a given span, which reproduces A2 §2's round bill exactly:
1,654 instructions, wmma 17, fmul 145, lds 146, barrier 11, 96 non-matrix slots per matrix slot, read). The
register-identity question is the one thing neither can ask, so §1.1/§1.2 read it directly off the disassembly
rather than adding a third census tool.

**What Gate 0 does NOT decide, stated so §2 is not over-read:**

* It shows the chain exists, not its price. The per-trip stall is the WMMA result latency plus the operand LDS
  fetch (both exposed, §1.2), and L1 splits the chain 16 → 2×8 but does **not** remove the per-trip LDS fetch
  wait — it just gives two tiles per trip to pay it with. How much that buys depends on which of the two is the
  bigger term, and nothing in this tree has it: `A4-CANDIDATES.md` §4 says outright that no `rocprofv3`
  VALU-busy / `SQ_WAIT_*` pass on this kernel exists (quoted). AGENTS' row "price a serial kernel in
  cycles-per-dependency at its realised occupancy, or do not price it" is the reason this file prints **no**
  predicted ms.
* At 6 wf/CU = **3 wf/SIMD resident** (read, §0b) another wavefront can cover a stall, so chain length and
  residency are not separable here — the same coupling A2 §5 admitted for C6 vs C7 (quoted). L1 spends
  registers to shorten the chain; A3 §2b is why the register side cannot be adjudicated from CPU facts on this
  part (clang 178 vs rocprofv3 248, no RF size reported), so Gate 1's admission is the note block, not occupancy
  arithmetic.
* The score leg is one leg of a round that also contains 16 P·V sites, a 128-multiply rescale (`:620-622`,
  quoted) and 4 barriers (read: `barrier=11` in the round span). A chain halving inside an 11-slot loop cannot
  be priced from the round's 1,654 slots, so this file does not try.

### 1.4 A sibling observation that must not be mistaken for this gate's finding (read)

The 16 P·V sites are independent in their **accumulators** but not free-running in the emitted schedule: each is
immediately preceded by its own 2 `ds_load_b128` + `s_waitcnt lgkmcnt(0)` refilling the **same** tuple
`v[168:175]`, e.g. `0xF864/0xF86C loads → 0xF874 wait → 0xF878 v_wmma v[9:16] → 0xF880/0xF884 loads → 0xF890
wait → 0xF894 v_wmma v[1:8]` (read; each `ds_load_b128` is 8 bytes, hence the +8 stride). What does overlap is the **rescale**: the `v_mul_f32_e32` block runs
between the loads and the wait (`0xF840..0xF854` multiplies, `0xF858` wait, `0xF85c` WMMA, read), i.e. LLVM
already interleaves 128 of the round's float slots with the V fetch. So "the P·V leg is 16 independent chains
so the rescale is the only serial dependency" (`:648-650`, quoted) is true of the accumulator graph and
incomplete as a statement about the emitted schedule. Recorded because a reader who takes it as "the round
already interleaves its matrix work" would conclude Gate 0 was EMPTY; the score leg has no such filler (§1.2,
point 3), and that asymmetry is precisely what L1 targets.

## 2. GATE 1 — the two-accumulator spelling, and the note blocks side by side

### 2.1 What was built, and how it is quarantined

`prototype/attn_percost/a4_l1_two_acc.py` carries the variant as **one exact text replacement** of the shipped
roll (`engine/attn_fattn_kernels.h:576-583`) and refuses to write anything unless its anchor occurs exactly once.
`prototype/attn_percost/a4_l1_build.sh` builds both sides from **copies** of the engine TU into `/tmp` —
`base/` with the header untouched, `new/` with the replacement applied — same flags, same include order, so the
only difference between the two listings is the patch. Nothing in `engine/` was written to
(`git status --porcelain` is empty for `engine/`; the only files this task added are listed in §6), there is no
runtime knob, and nothing was launched: `hipcc -c` twice, 3.2 s total (read).

The spelling (even k-tiles into `acc0`, odd into `acc1`, both loops under the same `#pragma unroll 1`, then
`acc[i] = acc0[i] + acc1[i]` once after the roll), with both tiles of a pair fetched before either matrix op
issues — that last part is what makes it a latency lever rather than a register shuffle.

**Canary, per AGENTS' "a paired A/B against a stale binary reads as a clean null" row.** Both sides are compiled
from copies, so the base side must be *shown* to be the shipped code before any delta means anything:

```
$ python3 prototype/attn_percost/a4_l1_canary.py /tmp/a4l1v/base/objdump.s /tmp/a4l1/ship-objdump.s
attn_fattn_wave: … base = 2175 insns | … ship = 2175 insns
IDENTICAL (addresses stripped): 2175 instructions agree
```

and `new` differs from `base` (2,203 vs 2,175 instructions) — i.e. the harness moved the code when, and only
when, the patch was applied.

### 2.2 The note blocks, side by side (read: `fattn_residency_read.py`, then `g3_isa_bill.py` for the metadata)

| note-block field | shipped (§0b) | `base` (this harness) | **`new` (two accumulators)** | gate |
|---|---|---|---|---|
| `vgpr_count` | 178 | 178 | **200** (+22) | report the delta, no occupancy arithmetic |
| `vgpr_spill_count` | 0 | 0 | **0** | **required 0** — passes |
| `private_segment_fixed_size` | 0 | 0 | **0** | **required 0** — passes (§33: spilled-at-8 lost to spill-free-at-6) |
| `sgpr_count` / `sgpr_spill_count` | 24 / 0 | 24 / 0 | **24 / 0** | unchanged |
| `group_segment_fixed_size` | 32,768 | 32,768 | **32,768 (unchanged)** | **required unchanged** — the accumulator is registers, as A4-BET demanded |
| derived blocks/CU, binding limit | 2, LDS | 2, LDS | **2, LDS** | unchanged |
| kernel instructions | 2,175 | 2,175 | **2,203** | +28 static |
| widest loop (the round) | 1,653 slots | 1,653 | **1,689 slots** | +36 static slots/round (+2.2 %) |
| `v_wmma` sites in the kernel | 17 | 17 | **18** | +1, inside the roll loop |

The register delta is accounted for, approximately: `acc1` is +8 VGPR/lane, and a two-tile trip stages four
operand tuples (`k0,q0,k1,q1` = +16 against the shipped two) — +24 expected against **+22 measured** (read), the
rest being allocator reuse (`v168:175` stays the single V-operand tuple in the P·V leg, §1.4). Not derived to the
last register, and not needed.

**What the +22 does and cannot do to residency, stated without pretending to an RF answer.** A3-result §2b is
quoted and binding here: the register file per CU is **unreported on this part** (clang 178 vs rocprofv3 248 for
the same kernel), so the note block is the admission and no occupancy arithmetic is offered. The one honest
boundary: at 2 blocks/CU of 96 threads the register demand is `2 x 96 x 200 = 38,400` VGPR/CU against
`2 x 96 x 178 = 34,176` shipped (derived), so +22 cannot be what costs the second block unless the RF sits
between those two numbers — and A3 §2b already showed 3 blocks of 96 was never arithmetically reachable
(51,264-71,424). The LDS limit is untouched and still binds at 2 blocks/CU.

## 3. GATE 1 — the ISA witness: two chains really exist (and the base side proves the check can fail)

`prototype/attn_percost/a4_acc_chain.py` reads the accumulator identity of every matrix site (why it exists and
why it is not a fourth census tool: its docstring). Run on both sides with the count it requires:

```
$ python3 prototype/attn_percost/a4_acc_chain.py /tmp/a4l1v/base/objdump.s --require-chains 1
# matrix loop 0xe614..0xe650: 1 site(s), 1 distinct accumulator tuple(s): [(129, 136)]
#   ONE accumulator: every site in the loop chains on the same tuple it writes …
# REQUIRE-CHAINS 1: OK (got 1)

$ python3 prototype/attn_percost/a4_acc_chain.py /tmp/a4l1v/new/objdump.s --require-chains 2
  site 0xe698  dst=(137,144)  A=(168,175)  B=(184,191)  acc-in=(137,144)   dst==acc-in: YES   loop: 0xe640..0xe6ac
  site 0xe6a4  dst=(129,136)  A=(176,183)  B=(192,199)  acc-in=(129,136)   dst==acc-in: YES   loop: 0xe640..0xe6ac
# matrix loop 0xe640..0xe6ac: 2 site(s), 2 distinct accumulator tuples: [(137,144), (129,136)]
# REQUIRE-CHAINS 2: OK (got 2)
```

The two sides are the control pair: the same checker called by the same command on the unpatched build reads
**1** and on the patched build reads **2**, so the witness is not a reading that passes whatever it is shown
(the empty-parse/decoration failure class in AGENTS). The emitted variant loop, verbatim (read):

```
0xE5DC  v_mov_b32_e32 v129, 0                      ; acc0 = 0
0xE5E4  s_mov_b32     s4, -2                       ; trip counter — 8 trips
0xE5F0  … v130..v136 = v129                        ; acc0{} = 0 (v[129:136])
0xE60C  … v137..v144 = v129                        ; acc1{} = 0 (v[137:144])
0xE640  v_add_nc_u32_e32 v196, v167, v157          ; <-- LOOP HEADER
0xE644  ds_load_b128 v[168:171], v167 offset:24576   }  tile 2t   (K)
0xE64C  ds_load_b128 v[172:175], v167 offset:24592   }
0xE654  ds_load_b128 v[176:179], v167 offset:24608   }  tile 2t+1 (K)
0xE65C  ds_load_b128 v[180:183], v167 offset:24624   }
0xE664  ds_load_b128 v[184:187], v196                }  tile 2t   (Q)
0xE66C  ds_load_b128 v[188:191], v196 offset:16      }
0xE674  ds_load_b128 v[192:195], v196 offset:32      }  tile 2t+1 (Q)
0xE67C  ds_load_b128 v[196:199], v196 offset:48      }
0xE684  v_add_nc_u32_e32 v167, 64, v167            ; +64 B = two k-tiles
0xE688  s_add_i32     s4, s4, 2
0xE690  s_cmp_gt_u32  s4, 13                       ; 8 trips x 2 tiles = 16 = FATTN_KTILES
0xE694  s_waitcnt     lgkmcnt(2)                   ; wait for 6 of the 8 — tile 2t's operands
0xE698  v_wmma_f32_16x16x16_f16 v[137:144], v[168:175], v[184:191], v[137:144]
0xE6A0  s_waitcnt     lgkmcnt(0)
0xE6A4  v_wmma_f32_16x16x16_f16 v[129:136], v[176:183], v[192:199], v[129:136]
0xE6AC  s_cbranch_scc0 -> <attn_fattn_wave+0xc40>   ; back edge to 0xE640
```

and the post-roll sum is **one** pass of 8 adds, not per trip: whole-kernel `fadd` sites go **16 → 24** (read,
`g3_isa_bill.py` class counts on the two sides), and the two sites land interleaved into the fold at
`0xE6E0 v_add_f32_e32 v129, v137, v129` and `0xE7F0 v_add_f32_e32 v129, v138, v130` — after the back edge at
`0xE6AC`, i.e. outside the loop (read).

Why this is the right witness rather than a diff of source: the compiler could have folded `acc0`/`acc1` back
into one chain (renaming one tuple, or sinking the add inside the loop) and the C++ text would have looked
identical either way. It did not. It also did the thing the lever was *about*: **eight `ds_load_b128` are issued
before the first WMMA and the first wait is `lgkmcnt(2)`, not `lgkmcnt(0)`**, so one LDS round-trip covers both
tiles and the first `v_wmma` starts as soon as its own four loads land. Per roll, derived from the two loop
readings above — same denominator on both sides, slots per (round, warp):

| | shipped | variant |
|---|---|---|
| trips | 16 (`s4: 0xfe00 +32 → 0`) | 8 (`s4: -2 +2 → >13`) |
| body slots per trip | 11 (1 WMMA, 4 LDS, 1 wait) | 18 (2 WMMA, 8 LDS, 2 waits) |
| **roll slots per round** | **176** | **144 + 8 adds = 152** (−13.6 %) |
| LDS ops per roll | 64 | 64 (unchanged) |
| **serial LDS round-trips per roll** | **16** | **8** |
| **WMMA→WMMA chain depth** | **16** | **8** (two chains of 8, interleaved) |

That table is the honest form of the lever: same matrix work, same LDS bytes, half the exposed LDS round-trips
and half the chain depth, for +22 VGPR and +36 static slots in the round.

## 4. Numerics — what moved, sized on the pinned data (CPU only)

**The consequence, stated before any measurement, as A4-BET requires:** the shipped dot is a fixed ascending F32
sum of the 16 k-tile partials (`:576-583`, and Gate 0 §1.2 confirmed it in the ISA), so `acc0`/`acc1` **is** a
re-association. Bit-identity with the shipped form is therefore not on offer, and A4-BET already decided the
judge: the existing §10.1 aggregates through `engine/gate.py` (quoted: G4 attention **cos ≥ 0.999,
meanrel ≤ 5 %**; the G3i-class isolated-per-layer row **cos ≥ 0.9999, meanrel ≤ 1 %**), under the delegated
numeric-format rule — existing bands, no widening, no `--ack-spill`. One citation correction: A4-BET writes the
band as "cos ≥ 0.9995"; `specification/10-reference-dumps-and-gates.md:38` (G4) is **cos ≥ 0.999** and
`:39`'s 0.9995 is the G5 logits row. Both sides of that correction are looser than 0.9995 in the row that
governs attention, so the admission does not get harder.

**A cheap CPU mirror for the round math does exist, so it was used instead of building one:**
`prototype/attn_decode/floor_probe.py` is the instrument §10.1's F1 floor was taken with, and
`prototype/attn_percost/a4_l1_reassoc.py` imports its `rows()`, `a_from_scores()` and `bf16_ulp.report()` and
feeds it the two associations. Same case population as F1 (48 decode cases from `dec8` + 5 long-cache rows from
`p300`), same ULP comparator, selftest run first:

```
$ .venv/bin/python prototype/attn_percost/a4_l1_reassoc.py
53 cases (48 decode from dec8 + 5 long-cache rows from p300) — the same population F1's floor was taken on
L1 re-association of the score roll, one accumulator vs two, output form `full`:
    15592 scores: 0 round to a different BF16 (0.00000%)
    over 108544 output elements: 0 differ (0.00000%), 0 LARGE, worst 0 ULP, cos >= 1.00000000, meanrel <= 0.000e+00
L1 re-association … output form `online`: the same, 0 / 0 / cos 1.00000000
```

For comparison, the mirror's own recorded F1 row (quoted, `prototype/attn_decode/results.md:209`): two *other*
legal orders of the same products move **0 of 15,592 scores and 0 of 108,544 output elements**.

Why zero is the expected answer rather than a lucky one (derived): the re-associated dot differs from the
ascending one by F32 rounding, order **1e-7** relative, while `fattn_score_round` (`:145-150`, quoted) rounds
the score through **BF16**, whose granularity is order **8e-3** relative — a flip needs the value to sit within
~1e-7 of a BF16 boundary, i.e. probability ~7.5e-6 per element, **~0.1 expected flips over 15,592 scores**.
So the sizing says: the §10.1 bands are not remotely stressed by this re-association, and the interesting
question is the opposite one — whether `o` stays bit-identical, which would let the admission keep its
bit-level section instead of leaning on the band.

**What that sizing does NOT license, and the device A/B must not skip:** the CPU mirror cannot see (i) the matrix
unit's intra-tile sum order — the 16 products inside one `v_wmma` are added in an order no ISA exposes, and my
mirror uses numpy's order for them; (ii) the fact that the shipping path rounds the score **three** times
(`fattn_score_round`) where the mirror rounds once; (iii) that a single moved BF16 in a generated prefix is
amplified by chaos over a long generation (AGENTS: greedy token equality decays with depth on any bf16
implementation, cos floor 0.8776 sdpa-vs-eager, so long-generation token match is not a gate). So the honest
statement is: **the re-association's floor on the pinned data is 0 BF16 flips and 0 differing output elements in
the only CPU mirror this tree has for the **WMMA round's** score arithmetic; the device admission is still what
says whether `o` is bit-identical or band-governed.** No device was touched here and no new mirror was built, and
the other CPU mirror in the tree — `prototype/attn_tile_model.py`, the CPU mirror and CPU-only gate for
`attn_tile_wave` — mirrors the **VALU** tiled wave's reductions, not a WMMA tile-partial sum, so it was not the
instrument for this question and was not modified. No device was touched here.

## 5. Handover to the main agent — what a funded Gate 2 would need

Both gates passed, so nothing is closed by this file. What follows is the harness contract, not a recommendation
— A4-BET already set the live/dead lines and they are unchanged by anything here (quoted: `attn` untraced
133.00 ms base, live ≤ 119, dead > 129; traced 155.71/≤140/>151; `wall_ms` 453.65/≤428/>440; 16,384-lane
3,319.25/≤3,050/>3,220; 32,768-lane 10,448.13/≤9,400/>10,135).

1. **The change to ship if it wins** is the one replacement in `a4_l1_two_acc.py`'s `NEW`, applied to
   `engine/attn_fattn_kernels.h:576-583`. It is a source change, not a knob: unlike §30's `--attn-fattn`, there
   is nothing to default off, so a Gate 2 that goes red means reverting a source edit.
2. **The behavioural canary inside the A/B window is mandatory** (AGENTS' stale-binary row, and the reason
   `a4_l1_canary.py` exists). Read `vgpr_count` from the *built* binaries' own note blocks in the same window
   and abort unless base reads **178** and new reads **200** — a note-block read inside the window is the only
   signal that survives `engine/build.sh` not being bit-reproducible. Second canary: `group_segment_fixed_size`
   must read 32,768 on **both** sides, and the `v_wmma` site count 17 vs 18 (`a4_acc_chain.py` on the shipped
   listing of each build). Any of the three reading equal means the A/B timed one binary twice.
3. **`launches` identical both sides, externality printed, verdict parser colour-tested, `SUMMARIZE=1`**
   (A4-BET's harness list); the wide legs through `--ids-file` (§39's `MAX_ARG_STRLEN` trap, which silently drops
   the widest point a check exists to test); and both sides of every pair measured **inside one `gpu_lock.sh`
   hold**, because AGENTS' "a ruler re-taken in a different window is not an A/B" row bites exactly the two wide
   rows that are ship gates.
4. **Lock protocol, already checked rather than warned about vaguely**: `prototype/attn_wmma/run_fattn_burst_window.sh`
   **self-acquires** `tools/gpu_lock.sh` (`grep -n gpu_lock` → acquire at `:101`, release at `:249`, and its own
   header at `:13-15` says it takes the lock because it *is* the window and calls nothing that self-locks). So a
   caller must **not** hold the lock around it — that is AGENTS' silent-deadlock row, and `grep -c gpu_lock`
   returns 7 for this script (read). This worker ran none of it.
5. **The numerics leg is not optional and is cheap**: `attn_fattn_wave`'s admission already carries a
   magnitude-sensitive check (`prototype/attn_wmma/fattn_rowsum_witness.py`'s point — cosine alone is blind to a
   scale error), so run the covered-row band on both sides and report whether `o` is **bit-identical** or
   band-passing. §4 above says the CPU mirror expects bit-identity at the pins; if the device disagrees, that
   is the finding, not an obstacle.
6. **One thing to re-check before spending the window**: §1.4's asymmetry says the P·V leg already overlaps its
   float work with the V fetch, so the round's remaining exposed latency may sit in the **barrier chain**
   (4 `s_barrier`/round, `barrier=11` in the round span, read) rather than in the score roll. If Gate 2 comes
   back unresolved-between-lines, the next cheap CPU step is a `rocprofv3` VALU-busy/`SQ_WAIT_*` pass
   (A4-CANDIDATES §4's first bullet), not another roll-shaped tweak; if Gate 2 comes back **dead**, then per
   A4-BET's closing paragraph the conclusion is a costed rewrite of the WMMA body against llama's tile
   geometry, and L2 (rescale batching, 7.7 % of a round's slots) is the diagnostic that distinguishes
   slot-bound from chain-bound — not a funded bet either.

## 6. Files this task added (all under `prototype/attn_percost/`; nothing else touched)

* `A4-L1-gates.md` — this file.
* `a4_l1_two_acc.py` — the variant, as a single anchored text replacement (refuses if the anchor is not unique).
* `a4_l1_build.sh` — base/new compile-only builds into `/tmp`, engine never written.
* `a4_acc_chain.py` — accumulator-identity read over matrix sites (`--require-chains` makes it a gate).
* `a4_l1_canary.py` — instruction-stream equality check between two listings.
* `a4_l1_reassoc.py` — imports `prototype/attn_decode/floor_probe.py` and sizes the re-association on the pins.

No `engine/`, `bench/`, `progress/`, gate or threshold file was edited; nothing was committed; no device
command was run, so there is no `util` reading and **no timing of any kind in this file**.
