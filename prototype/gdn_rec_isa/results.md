# `gdn_rec` — what the shipping seq recurrence actually issues

CPU-only ISA census of `prototype/gdn_decode/gdn_decode_kernels.h::rec_kernel`, the kernel behind
the **biggest decode-kernel share on disk**: `gdn_rec` = **126.83 ms, 51.2 %** of the 245.7 ms wave
at 1024 lanes, **~509 ms, 46 %** of the ~1,105 ms wave at 4096 lanes
(`bench/performance-status.md`, "Headline tables after the tiled attention wave ships": 18,432 and
73,728 launches, ~6.8–6.9 µs each). The shape of the question is the one
`prototype/attn_valu_slim/results.md` asked of attention, and `prototype/m2_launch`'s D21 rule is
the one that constrains the answer: **census first, do not price a rate as the wall.** The short
answer is that attention's pile is here, it is ~10× smaller in relative terms, and the family runs
far enough below its own issue roof that instruction surgery cannot be the wall (§4).

No device was touched. Every number here is either from a listing or already on disk.

## 1. The unit — a "token-step", and why attention's amortisation analogy breaks

One **token-step** = one `rec_kernel` launch = one (token, GDN layer):
`rec_kernel<<<NH=16, D=128>>>` (`engine/kernels_gdn.cpp:40`), wave32 (`.amdhsa_wavefront_size32 1`),
so 16 blocks × 4 wavefronts = **64 wavefronts**, and the family is 18 launches per token (18 GDN
layers). The grid is not "16 wavefronts over 40 CUs" as the earlier read put it — it is 16 *blocks*,
but the conclusion survives: 64 wavefronts is ≤1 per SIMD on this part either way (§3). (Corrected
while the harness was written: this paragraph carried a line cite into `engine/kernels_gdn.cpp` past
the end of that file, and the `rec_kernel<<<NH, D>>>` launch is at
`engine/kernels_gdn.cpp:40`. A stale coordinate still resolves to *some* line, which is why
`tools/doc_ref_check.py` is what caught it and not a reader.)

The launch is **one token deep**. The per-`j` state walk is a real loop with a runtime trip count;
the per-token loop is not a loop at all — the next token is the next launch. So:

* attention's lever worked because a per-touch cost amortised over `L` touches and the prologue was
  once per ~2,368 slots;
* here the prologue is once per token-step against **nothing** — it is 323 of 1,523 slots (21 %)
  and it repeats 18 times per token;
* and no lever can fix that within one launch: the recurrence inside a head is serial
  (`s = s*decay + k*v`, then `o` needs `s`), so there is no second token to pipeline against.
  Cross-token pipelining is what `prototype/gdn_collapse`'s scan route is for.

**Control, both directions, before any delta means anything:** a copy of the shipped kernel
(`rec0_kernel`, its own `red[32]`, same 1,152 B LDS, same launch geometry) censuses **IDENTICAL** to
`rec_kernel` (1,523 vs 1,523 slots, category for category), and the copy compiled in a *different
TU* (`prototype/gdn_rec_isa/isa_tu.cpp` vs the engine TU) is also identical. Had that row failed,
none of §2–§4 would be usable.

## 2. The bill, per token-step, per wavefront

Two hot loops, both compile-time unrolled by LLVM (trip counts derived in the script from state
accesses per trip and cross-checked against each loop's own induction bound):

| region | static lines | slots | trips | slots/token-step |
|---|---|---|---|---|
| prologue (q/k/v loads, gate+beta, both block reductions, `expf`, `rsqrtf`) | 382 | 323 | 1 | **323** |
| `loopA_state_dot` (`.LBB2_20`: state load + `dv` dot) | 73 | 48 | 8 | **384** |
| `loopB_state_update` (`.LBB2_22`: state read-modify-write + `o` dot) | 65 | 51 | 16 | **816** |
| **total** | | | | **1,523 slots** = 97,472 per launch |

Category rollup (76 rules, every one self-tested against its own example line; `s_waitcnt` /
`s_delay_alu` / `s_nop` are *bookkeeping lines* and are excluded from the slot count, then reported
separately — 482 of them):

| category | slots | share of the 1,523 | what it is |
|---|---|---|---|
| `encode.bookkeeping` | (482 lines) | excluded from slots | waits/delay/nop |
| `state.fma` + `state.mul` | 443 | 29.1 % | the recurrence itself |
| `state.load` | 261 | 17.1 % | 16 + 8 `global_load_b32` per trip |
| `state.store` | 129 | 8.5 % | the state writeback |
| `addr.math` | 127 | 8.3 % | `v_add_lshl`/`v_add3`/`v_accvgpr` addressing |
| **`xlane.exec-guard`** | **127** | **8.3 %** | see §3 |
| `loop.salu` | 101 | 6.6 % | loop counters/branches (×trips) |
| `state.coef-lds-load` | 96 | 6.3 % | `ds_read_b128` for a/b |
| `vmem.clause` | 64 | 4.2 % | `dual_pudl_b32` clause openers |
| `xlane.add` / `permute` / `range-select` / `addr-math` / `lds-round-trip` | 22 / 20 / 7 / 9 / 9 | 4.4 % | the reduction's real work |
| `barrier.sync` + `invalidate` + `drain` | 23 | 1.5 % | 11 `s_barrier` + 11 `buffer_gl0_inv` + 1 drain |
| `norm.expf` / `rsqrt` / `lane-id` / `warp-count` | 28 | 1.8 % | 18 `expf` (polynomial incl.) + root |
| `valu.mov` / `salu.*` | 50 | 3.3 % | |
| unclassified | 7 | 0.5 % | two `s_mov_b32 s4,0` (kernarg), 4 `s_cbranch_execz`, `s_endpgm` |

**Per-element cost of the state walk**, which is what attention's per-touch table was: loop A does
16 state elements per trip in 48 slots — **3 slots/element**, of which 1 is the load, 1 the FMA and
~1 the addressing (and RDNA3 `v_dual_*` halves the slot count where LLVM found a pair: 12 of loop
A's 48 slots are duals carrying two ops each). Loop B: 8 elements in 51 slots — 6.4/element,
because it carries load + FMA + `*g` + store + the `o` FMA + `beta*(...)` + a mid-loop wait. That
is ~1.15 slots above the bare `load+FMA+store` floor per element. There is no per-touch score-style
overhead to find, because there is no score here.

**Divergence from `v_cmp_*` exec masks: none.** All 20 shuffles are over full waves; the 20
`s_trap 2` sites are LLVM's divergent-shuffle fallbacks and are *never executed* (86 static lines,
reported and excluded). No `v_permlane16/8/4/1/2`, no `s_but_or_b64`, no `s_savepc`/`s_rrsc` — those
appear only for genuinely divergent shuffles.

## 3. The named classes — what is present, what is absent

**PRESENT, class 1 — the cross-lane spill (the attention pathology, same instruction names).**
The prologue's two `block_sum` calls compile each of 20 `__shfl_down_sync` into 10–13 lines. The
bookkeeping, per shuffle: `s_mov_b32 exec_lo/hi == -1`, `s_cmp_lg_u32`, `s_cbranch_scc1`, and the
`%Flow` convergence restore (`s_and_not1_b32 exec_lo`, `s_cbranch_vccnz`) — LLVM's divergent-shuffle
guard, which a uniform-wave source cannot switch off. Plus the range-select that zeroes the delta
when the source lane leaves the wave (CUDA's out-of-range rule), the byte-address math, and — for
every delta except 16 — an LDS round trip through `ds_bpermute_b32` at `4*src`. Totals:
**127 exec-guard + 22 add + 20 permute + 9 range-select + 9 addr-math + 9 lds-round-trip + 7
lane-id/reduce + 23 barrier = 226 slots, 14.8 % of the token-step bill**, all of it in the 323-slot
prologue. Attention's equivalent was 29 % of the per-touch bill (exec-mask 16.2 % + lane-boundary
7.6 %) at 224 slots per touch of ~844 — same op names, 1/40th of the exposure, and no ~5.5
instructions per element of score-side arithmetic to hide it in.

**PRESENT, class 2 — barriers: 11 per token-step for two reductions.** `block_sum` contains 4
`__syncthreads()` (post-write, pre-publish, publish, read) and the kernel adds 3, each lowering to
`s_waitcnt vmcnt(0) lgkmcnt(0)` + `s_barrier` + `s_waitcnt lgkmcnt(0)` + `buffer_gl0_inv`. Four of
those 11 are pure "protect `red[]` for the next call" and 6 of them are the same rendezvous done
twice because the two reductions are sequential calls rather than one pass over shared memory.

**PRESENT, class 3 (small) — one over-full drain.** `loopB`'s mid-loop `s_waitcnt vmcnt(0)` (16 per
token-step) drains *all* outstanding global work where `vmcnt(8)` would do — the 8 state stores of
this trip must land before the next trip's 8 loads issue (WAR on the same addresses). A hand-written
renaming of the state buffers would let the trips overlap; the census cannot say whether the stall
is real, only that the wait is 8 slots stricter than the data requires. Loop A's `vmcnt(15…0)` are
correctly staggered (one per load, consumed in order) and are not a target.

**ABSENT — runtime trip counts.** `D=128` is `#define D 128`, so both state loops are
compile-time unrolled (8 and 16 copies, induction bound folded: `s_add_i32 s2,s2,8` +
`s_cmpk_eq_i32 s2,0x80`). There is no dynamic trip count to hoist and no scalar-boundary recompute per
element. Loop `.salu` is 101 slots/step and is unroll bookkeeping, not per-element overhead.

**ABSENT — `v_movrels`.** Zero occurrences in the listing. The dynamic `state[(j<<8)+tid]` indexing
did not become a register-array spill; LLVM kept it in `v_add_lshl_u32`/`v_add3_u32_fwd` address
math (127 slots, `addr.math`, 8.3 % — which is the *cost* of not having movrels, and it is 3×
attention's 2.4 %).

**ABSENT — redundant bf16 ↔ f32 conversions.** The whole recurrence is F32; the listing has no
`v_cvt_*` at all. (The engine's bf16 traffic is in the projection kernels, not here.)

**ABSENT — a per-step SALU pile hoistable out of a token loop.** The prologue's 18 `expf` calls,
12 `s_*` coefficient setups and the `rsqrtf` are per-token-by-nature: `g` and `beta` are that
token's inputs. There is no token loop inside the launch to hoist them out of — which is the §1
point in ISA form, and why "hoist per-step SALU out of the token loop" is not available here while
"hoist per-touch SALU out of the inner loop" *was* available in attention.

**NOT a pathology, recorded so no one re-finds it: occupancy.** 33 VGPR / 25 SGPR / 1,152 B LDS, no
spills, 16 waves/SIMD theoretical occupancy. The kernel is register-light; nothing is limited by
residency. The grid is the limit, not the kernel — 16 blocks of 128 threads on a 40-CU part (§4).

## 4. Pricing the pile, D21-style: and why it does not fund

Machine facts from `devices/strix-halo-compute.md`: 40 CUs, 2.9 GHz max / 2.206 GHz observed under
load. SIMDs per CU is printed as 2 by `rocminfo` in that file and used as 4 (128 FP32 lanes/CU) by
its roofline section; the conclusion is the same under both readings and both are stated.

* Issue floor per launch: one wavefront per SIMD means the launch cannot go faster than one
  wavefront's own bill: **1,523 cycles ≈ 0.69 µs at 2.206 GHz** (64 wavefronts fill ≤64 SIMDs at
  80, or ≤64 of 160 at the 4-SIMD reading — either way ≤1 per SIMD, so no second stream to fill
  stalls).
* Measured per launch: **6.8–6.9 µs**. So the family runs at **~10 % of its own issue roof**, and
  ~90 % of its wall clock is stalls on the per-token dependency chain (24 exposed global round trips
  per token-step: 8 in `loopA`, 16 in `loopB`, with 2 wavefronts/SIMD to interleave them with).
* The two levers I built to price the cut: MODE 1 (hand-spelled permutes, compiler's own out-of-range
  clamp, shipped barrier structure) = **1,488 slots, −2.3 %**; MODE 2 (the two reductions fused into
  one pass over shared memory, hand-spelled, 11 barriers → 5) = **1,391 slots, −8.7 %**
  (97,472 → 89,024 slots/launch); MODE 2's control `rec2c_kernel` (fused, compiler-lowered
  shuffles) = **1,492, −2.0 %**, which shows the cut is mostly the guard removal, not the fusion.
* **Ceiling if the kernel were issue-bound:** 8.7 % of 509 ms ≈ **44 ms** at 4096 lanes (11 ms at
  1024). It is not issue-bound. **Ceiling under the census's own stall model:** the removed work is
  prologue-side (167 of the 323 prologue slots, 6 of 11 barriers, 119 of 127 exec-guards), so
  best case ~2,000 cycles of the ~15,000-cycle launch → **~13 % of the family ≈ 66 ms at 4096
  lanes**, and the realistic band sits between that and the ~5 ms the pure-slot view allows, because
  most removed SALU issues in the shadow of an LDS or VMEM wait.
* **Assumptions:** one issue slot per SIMD per cycle at 2.206 GHz; 64 wavefronts do not cooperate
  across blocks; the ledger's per-launch time is whole-kernel wall, not launch overhead.
* **Falsification (the cheap one first):** burst MODE 0 vs MODE 2, same buffers, one lock hold. My
  model predicts **< 1.02×** — instruction cuts at 10 % roof. If MODE 2 comes back > 1.10×, the
  prologue chain is on the critical path and the norms belong outside the launch (MODE 3, unwritten).
  The expensive falsifier, if anyone wants the traffic question again: `rocprofv3` `SQ_WAIT_ANY` vs
  `SQ_BUSY` on the two modes.
* **Bit-identity vs band:** MODEs 1–2 preserve every value's addition tree exactly (same 5+5 shuffle
  steps in the same order per value, same `beta*(…)` fold, same barriers before each `red[]` reuse),
  so no numerics band moves and no gate changes. The only liberty is the discarded lanes' values,
  which never reach lane 0's tree. This is a formatting-level rewrite of the reduction, not new
  arithmetic — unlike the collapse route, which *re-associated* the dot and needed its own §7.2
  discrimination run.
* **Verdict: not funded.** The honest read of the census is that `gdn_rec` is latency-bound at the
  geometry, not slot-bound, and the xlane spill — the thing this census was hunting — is 15 % of a
  bill that is 10 % of the time. The levers are written, bit-identity-clean, and cheap to test in
  one lock hold; they should be tested *as a falsifier of this model*, not as a win. What the census
  does add is the reason D22's launch arithmetic came out the way it did: cutting per-token
  *instructions* by ~4× in the collapse form bought 10 % for exactly this reason, and no instruction
  cut will do better than the ~10 % roof ratio allows. If `gdn_rec` is to move, it moves by raising
  work-per-launch across an axis the recurrence permits (more tokens per launch, i.e. the scan
  route) — not by slimming the instruction stream.

## 5. Files, commands, limits

`isa_census.py` (76 self-tested rules; imports `attn_valu_slim`'s wrapper over
`attn_score_dot2`'s classifier — nothing copied), `gdn_rec_isa_kernels.h` (MODE 0 control + MODEs
1/2/2c; MODE 3 specified but unwritten), `isa_tu.cpp`, `isa_census.output.txt`/`.s`,
`rec_kernel.ship.s` — and, since §6, `gdn_rec_isa.hip.cpp` + `build.sh` (the device A/B harness) with
`isa_census.harness.output.txt` (the census re-taken on the harness TU). Commands, device steps and what
is *not* built (MODE 3, and the `rocprofv3` counter pass): `HANDOFF.md`. Limits of this write-up: it is
a static census — `expf`'s real length, LDS bank behaviour of the 512 B-stride `ds_read_b128`, and
whether any wait is actually exposed are all device questions, listed in `HANDOFF.md` §4.

## 6. The harness that makes the falsifier runnable (added after the verdict, not before it)

`results.md` §4 ended with a falsifier no one could run, because `HANDOFF.md` named the commands and
`prototype/gdn_rec_isa/` had no `.hip.cpp`. `gdn_rec_isa.hip.cpp` is that harness, built on
`prototype/gdn_collapse/gdn_collapse_check.hip.cpp`'s shape (one process, one set of operands, one state
buffer and one `o` buffer per MODE, guard bands and an unwritten-element sentinel on everything) with
`prototype/attn_m2/attn_m2.hip.cpp`'s burst shape (every active MODE timed *inside* each rep, so the
falsifier is one measurement window rather than four). Commands and expected output shape:
`HANDOFF.md` §3. No timing was taken by whoever wrote it — the `--burst` path is present and unused.

**What it judges, and how little room it leaves.** MODE 0 (`rec0_kernel`) is the reference; MODEs 1, 2
and 2c must reproduce it **bit for bit on `o` and on the final state over T = 128 tokens**, four cases
(two state memories × zeros/carried state, four seeds). Zero is the only acceptable `ndiff_*_vs0`. No
band, no cos, no max-ULP: a band here would be a way of not checking the claim, and the claim (§4's
"bit-identity vs band" bullet) is that these MODEs re-spell a reduction rather than re-associate one.
Two extras the census did not ask for but the tree's own traps insist on:

* **every case runs twice** (`--repeat 2`). MODEs 1 and 2 cross lanes with inline
  `ds_bpermute_b32`/`ds_swizzle_b32` whose `s_waitcnt lgkmcnt(0)` is written by hand, and per
  `prototype/attn_valu_slim` a *racy* permute does not produce a stable wrong answer — it produces
  run-to-run disagreement. `rerun_ndiff_*` is that check; a non-zero value indicts the wait, not the
  arithmetic.
* **a CPU reference is printed but is not the gate.** The host mirror transcribes the shipped tree,
  including the FMA sites the listing proves the device fuses (`v_fmac_f32_e32` in loop A at
  `rec_kernel.ship.s:243-282` and loop B at `:346-383`, `v_fma_f32 v1, -v6, v1, v5` at `:315`), and keeps
  `qn = (qq*rsqrt)*Q_SCALE`'s association — the bug class `prototype/gdn_collapse/results.md` §2
  documents. It cannot be bit-identical to the device, because the device computes `rsqrtf` as
  `v_rsq_f32_e32` (`rec_kernel.ship.s:177`) and `expf` as `v_exp_f32_e32` + `v_ldexp_f32` (`:304-307`),
  approximations of a different kind from glibc's. So `ndiff_*_vs_cpu` is non-zero for MODE 0 too, MODE
  0's line is echoed as `CPU-FLOOR`, and a MODE passes by matching MODE 0. The mirror's real job is the
  one thing MODE-vs-MODE structurally cannot do: a fixture bug shared by *all* the MODEs — a q|k|v row
  stride off, heads indexed the other way, the state transposed — leaves every MODE agreeing with every
  other, and shows instead as `maxrel_*_vs_cpu` of order 1 instead of order 1e-6.

**Two CPU-only selftests, because a gate nobody can run should still be half-checkable on a CPU.**
`--selftest-lane` (80,660 checks, `bad=0`) re-derives the hand fold's source-lane rule and the
compiler's on the host and requires them equal for all 32 lanes × 5 deltas and for both stages of the
shipped `block_sum`, with a must-catch control: the xor-butterfly rule and the "a lane that falls off the
wave contributes nothing" rule each have to *move* lanes, and they do (62,000 each), so the agreement is
not vacuous. `--selftest-mirror` (524,304 checks, `bad=0`) witnesses the mirror's addresses with three
exact identities at `g = 0` exactly — `beta = 0` leaves the state bit-identical, `beta = 1` from a zero
state yields the outer product `ks ⊗ v`, and `o` is `qs · state` — which is what says the CPU column is
measuring the recurrence and not a private invention of it.

**The census row: the harness work did not move MODE 0.** `isa_census.harness.output.txt` is
`isa_census.py` run over the *harness* TU (`SAVE_TEMPS=1 ./build.sh` leaves `tmp/gdn_isa/gdn_rec_isa.s`),
so the listing is the code the binary will actually execute, built with the `-Xcompiler -mfma` the host
mirror needs:

| listing | `rec_kernel` | `rec0_kernel` | `rec1_kernel` | `rec2_kernel` | `rec2c_kernel` |
|---|---|---|---|---|---|
| engine TU (`isa_census.output.txt`) | 1,523 | 1,523 | 1,488 | 1,391 | 1,492 |
| harness TU (`isa_census.harness.output.txt`) | 1,523 | 1,523 | 1,488 | 1,391 | 1,492 |

Both control rows read IDENTICAL (shipped vs the MODE-0 copy, engine TU vs harness TU) and the category
tallies agree slot for slot with §2, so the −2.3 %/−8.7 %/−2.0 % deltas §4 priced are the deltas the
binary will run, and nothing in the harness work reached device codegen.

**What the thresholds decide.** The burst prints, per case, `best_<mode>_ms` (ms per *token-step* launch,
best-of-`reps`, state restored before each timed loop so every MODE walks the trajectory that was gated)
and a `FALSIFIER` line per lever with `rec0_over_mode` and its band:

| `rec0_over_rec2` | what it means, in §4's words |
|---|---|
| **> 1.10** | the latency reading is refuted: the prologue chain *is* on the critical path and the barrier/guard cut is a lever, not a formatting change. MODE 3 (the norms out of the launch, one pre-pass per token for all 18 layers) becomes the funded next step, and it is still unwritten. |
| **< 1.02** | the instruction route is dead for this family, as §4 predicted: `gdn_rec` moves by raising work per launch (the scan route), not by slimming the instruction stream. Nothing else in this directory needs re-opening. |
| **between** | inconclusive. Re-take on the long-memory cases (the ones whose state resembles a real wave), and if it persists the `rocprofv3` counter pass in `HANDOFF.md` §4 is the tie-breaker. |

And what a bit-identity failure would mean per MODE, since that is the question this harness answers
first: `rec1` differs → the hand fold is not the tree the compiler emits (the lane selftest says it is at
the rule level, so look at the asm and the wait); `rec2c` differs while `rec2` does not → the *fusion* is
wrong (shared barrier structure, the two `red[]` arrays, the dropped "protect `red[0]`" barriers), since
2c differs from MODE 0 only by fusion with compiler shuffles; both differ → the spelling is common to
both; `rec2c` clean and `rec2` dirty → the hand spelling alone. `rerun_ndiff_*` non-zero on any of them →
the hand-written `s_waitcnt` is racy, which is a defect in the MODE, not a band to negotiate. A MODE that
moves a bit is not a slower lever, it is a numerics change, and this harness exists to say so before
anyone times it.

## §7 Device falsifier (main agent, 2026-10-16): D23's verdict stands, and one of its premises needed replacing

Window `gdn_rec_isa_ab`, blocking, zero subagents, util 92 % (discounted). CPU selftests first:
`--selftest-lane` 80,660 cases bad=0 with its must-catch control firing (62,000 lanes moved), and
`--selftest-mirror` 524,304 cases bad=0. Then admission on device, 4 cases × 4 MODEs, T=128:

| MODE | vs MODE 0 | rerun (same process, twice) | verdict |
|---|---|---|---|
| `rec0` (shipped copy) | 0 / 0 | 0 / 0 | BIT_IDENTICAL |
| `rec1` (hand-spelled fold) | **262,144 / 262,144 — every element** | **1,408 and 80,512 o elements differ between runs** | DIFFERS, **non-deterministic** |
| `rec2` (fused pair, hand-spelled) | **262,144 / 262,144** | **14,080 in the carried case** | DIFFERS, **non-deterministic** |
| `rec2c` (fused pair, compiler-lowered) | **0 / 0** | 0 / 0 | BIT_IDENTICAL |

Two things follow, and only one of them is about `gdn_rec`.

**1. The hand-spelled cross-lane spellings in MODEs 1 and 2 are broken, not merely re-associated.**
Every element of `o` differs (262,144 of 262,144), `maxrel_o_vs_cpu` reaches 5.4e+05 to 1.2e+06 against a
MODE-0 floor of 1.9e-01, and the same MODE disagrees with itself across two runs — which is the signature
this repo uses for a racy hand `ds_bpermute_b32`, not for a different addition tree. `rec2c`, the *same*
fused-pair idea lowered by the compiler instead of by hand, is bit-identical. So the technique is not the
problem — `attn_valu_slim`'s hand-fold survives bit-identity and a three-run race check on the shipping
path — these two spellings are wrong, and the harness caught them before any number was taken from them
because the gate was bit-identity and "no timing if admission failed" was enforced in the binary.

**2. The falsifier ran on the one valid lever, and it landed in the kill band: 1.0159×.**
`rec0` 0.8088 ms vs `rec2c` 0.7962 ms, burst 40 × 5 reps, same buffers, one window, per-launch 6.319 µs
(the census modelled 6.8–6.9 µs, so the model was not wrong about the launch cost either). `rec2c` removes
2.0 % of slots, so measured elasticity is ≈ **0.8** — slots cut → time cut, nearly proportionally.

That elasticity is *much* more issue-sensitive than D23's roof argument implied, and it is the honest
correction to make here: "runs at 10 % of its issue roof" predicted instruction cuts would be near-worthless,
and the direct measurement says a slot cut does translate at roughly 0.8×. It does not change the verdict,
because the arithmetic lands in the same place from the other direction: the best available lever in this
family is MODE 2 at −8.7 % of slots, which at elasticity 0.8 is ≈ 7 % of the family ≈ 36 ms of a
1105 ms wave, and that MODE is the one that is broken. **Repairing two hand spellings to collect a
predicted ~3 % of the wave is not funded**, and the falsifier's own threshold agrees: to refute the
latency reading you needed > 1.10×, and the only working spelling returned 1.016×.
