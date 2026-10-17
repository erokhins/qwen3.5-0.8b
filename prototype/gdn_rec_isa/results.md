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
`isa_census.harness.output.txt` (the census re-taken on the harness TU). Since §8–§9 the directory also
holds `traffic_census.py` (+ `traffic_census.output.txt`, its captured run) for the state-traffic census and
the ISA volume admission, and the four dependency probes `recx_indep/unchain/shape/both_kernel`. Commands,
device steps and what is *not* built (MODE 3, and the `rocprofv3` counter pass): `HANDOFF.md`. Limits of
this write-up: it is a static census — `expf`'s real length, LDS bank behaviour of the 512 B-stride
`ds_read_b128`, and whether any wait is actually exposed are all device questions, listed in `HANDOFF.md` §4.

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

## §8 Traffic census: the state is 99 % of the demand, and the demand falsifies the traffic model

D30's queue item 1.6 was "is the state dtype costing us — cheap census first, before anybody writes a
16-bit kernel". `traffic_census.py` is that census. It is CPU-only and touches no device: the shapes come
out of `engine/state_sizes.py` (which is itself the gate that re-derives `state.h` against §00.5, so a spec
edit that renames a shape fails the census instead of quietly disagreeing with it), the dtype out of the
declared field in `engine/state.h`, the geometry out of the launch site, and the number of state passes out
of `rec_kernel`'s own loop body. Full output is kept beside it as `traffic_census.output.txt`; run it with
`--selftest-all` (9 checks, `bad=0`) before believing any line below.

**What is in memory, with the witness for each row.**

| claim | value | where it comes from |
|---|---|---|
| GDN state dtype | **F32**, one `float` per element | `engine/state.h:88` — `float* gdn_state` |
| one layer's state | 16 heads × 128 × 128 × 4 B = **1.000 MiB** | `kNumValueHeads`, `kDk`, `kDv` from `engine/state_sizes.py` |
| per request, 18 GDN layers | **18.000 MiB** | same, × `kNumGdnLayers` |
| conv window | **F32**, `float* conv_window` (`engine/state.h:89`) = **1.266 MiB**/request | same. The brief's "conv state 0.63 MiB" is the **retired** §00.5 BF16 cell; `state_sizes.py`'s FIXES list and `state.h`'s header comment both record the correction to F32 / 1.27 MiB |
| launch | `rec_kernel<<<NH=16, D=128>>>`, 64 wavefronts, **one (token, layer) per launch** | `engine/kernels_gdn.cpp:40` |
| state passes per launch | **2 read + 1 write** | `prototype/gdn_decode/gdn_decode_kernels.h:179`'s two `for (i = 0; i < D; ++i)` walks; independently confirmed in machine code (§9's volume check reads 256 state loads and 128 state stores per token-step, i.e. exactly 2 MiB read and 1 MiB written) |
| token loop | a **host** loop around the launch | `engine/decode_step.cpp:752` → call at `:757` |

**Is the state resident across tokens, or re-read?** **Re-read, every token — and it cannot be anything
else in this geometry.** The shipped `seq` route launches once per (token, layer), so there is no token loop
inside the kernel for a value to be resident in; the next token is the next launch, and it loads the state
from global again. It is not a staging choice either: one block owns one head's 128 × 128 F32 strip =
64 KiB, which *is* the CU's whole LDS budget, so the shipped form could not hold the state across tokens if
it wanted to. The engine does own a resident form (`--gdn-scan collapse`,
`engine/gdn_collapse_kernels.h`) — which matters because it is already the direct witness below.

**The census.** Bytes demanded per token-step (one launch, one layer), then scaled:

| scope | read | written | total |
|---|---|---|---|
| one token-step (1 layer) | 2.024 MiB | 1.008 MiB | 3.031 MiB |
| — of which state | 2.000 MiB | 1.000 MiB | 3.000 MiB |
| — of which `q\|k\|v`, `g`, `beta`, `o` | 0.024 MiB | 0.008 MiB | 0.031 MiB |
| one token (18 layers) | 36.424 MiB | 18.141 MiB | 54.565 MiB |
| one 4096-lane wave (73,728 launches) | 156.440 GB | 77.913 GB | **234.354 GB** |

At the measured `gdn_rec` family time for that wave (534.54 ms, D30's table) the demand is **438.4 GB/s**,
of which 433.9 GB/s is state; the operands are 1.03 % of the bill. Per launch that is 7.25 µs against the
ledger's 6.8–6.9 µs model and D25's measured 6.319 µs.

**The D21 caveat, applied rather than recited.** That 438.4 GB/s is *demanded*, not DRAM, and it is
**above** the 240 GB/s read plateau the part can reach at 1 GiB
(`devices/strix-halo-memory-bandwidth.md`). A demand figure that exceeds the machine's ceiling is not a
bandwidth wall — it is a units confession, and it *falsifies* the traffic model rather than supporting it.
Two independent numbers say the same thing. The identical access pattern with the state warm measures
**876 GB/s** at an 18 MiB working set (`prototype/gdn_decode/results.md`, Performance), so this kernel asks
for 50 % of what the same shape is measured delivering — traffic cannot be what bounds it. And the traffic
has already been removed once, in situ: `prototype/gdn_collapse/results.md` §5's route A keeps the state in
LDS and cuts the round trip from 38.7 GB to 75 MB per 1024-lane wave (516×) for **125.36 → 112.04 ms,
1.119×** of the family. Attribute all 13.32 ms of that to traffic and a 2× cut is still worth at most half;
attribute it the way the launch arithmetic says to — route A also replaced 18,432 launches with 36, and the
empty-kernel dispatch floor alone is 1.75–2.08 µs — and traffic bought close to nothing. Either reading
leaves the state dtype out of it.

**What a 16-bit state would buy, priced rather than hoped.** It halves the *state* term (99.0 % of the
demand), so the wave's demand falls from 234.4 GB to ~119.6 GB and the demand rate from 438 to ~224 GB/s —
i.e. from above the package ceiling to about it, which was never the binding quantity. The cost side is the
instruction bill: the recurrence consumes the state as F32 inside FMA chains, so a 16-bit state means a
conversion per state access — 2 × D reads + 1 × D writes = 384 conversions against a 1,523-slot bill
(**+25.2 % slots**), and D25's measured slot elasticity on this very kernel is ~0.8, i.e. **roughly +108 ms**
against a traffic term bounded at ~13 ms. Packing two halves into one 32-bit access does not dodge it: a
thread owns one `d_v` column `j` and walks `i`, so its pair would live in another thread — packing the state
means re-cutting the thread mapping, which is a different kernel, not a dtype flag. And it is already named
as a defect here: `engine/gdn_collapse_kernels.h`'s fail-closed negative `rc_state_bf16` is "*the cheap the
state is big, halve it* mistake". The numerics question cannot even be posed with today's pins: §10.1 G3
gates `state_new`, but the `decode*.state_fp.npy` dumps are fingerprint rows (`tools/reference_dump.py` →
`tools/state_fp.py`), not values, so D14's fp16 range census (peak 82.5, 794× margin) covers
activations/logits/weights and says nothing about the state's dynamic range — under D13 the gates decide,
and here they could not go either way. **Recommendation: do not fund it.**

**Verdict.** `gdn_rec` is **not traffic-bound**: the demand rate exceeds what the part can move (so the
model is wrong, per D21), the same access pattern is measured at twice the demanded rate when warm, and
removing essentially all of the traffic in situ bought 1.119×. Nor is it instruction-bound (D25: 1.0159× for
a −2.0 % bill). Launch count is out too (route A cut 18,396 launches for 13 ms). What is left is the
dependency chain — still the live reading, still unmeasured, and now measurable: §9.

Limits of this census, stated in full because the repo asks for them: demanded bytes are not DRAM bytes and
no counter pass was taken, so the absorbed fraction is unknown; the 876/255 GB/s rows are decode-step
measurements of the same access pattern rather than of this wave, and the 108 MiB row comes from a build
that no longer exists; route A's 1.119× bundles dispatch and traffic and the split above is argued, not
measured; the conv window belongs to `gdn_conv` (20.71 ms, D30 1.7) and is listed here only for the
per-request arithmetic. The 16-bit instruction cost is an F32↔F16 conversion count derived from the access
pattern, not a compiled variant — if anyone ever funds it, the number to re-take is that slot delta.

## §9 Dependency-break probes: the falsifier for the latency reading, and what it will cost to read correctly

§4's verdict rested on a latency reading — ~1.6 *working* wavefronts per CU, so a dependent-load chain has
nothing to overlap with — and D25's falsifier closed the *instruction* route without ever testing it: a slot
cut cannot refute a latency model, it can only shorten the stream. The instrument that can is a form whose
**dependency structure is broken while its volume is held**. Four new MODEs do that, all in
`gdn_rec_isa_kernels.h`, all launched through the harness's single `launch_mode` so a timing can never be
taken on a different launch plan from the one that was gated. None of them is a lever, and none touches
`engine/`.

| MODE | kernel | what it breaks | value status | gate |
|---|---|---|---|---|
| `rec3` | `recx_indep_kernel` | **cross-token**: reads `S_in`, writes `S_out`, so no launch waits on its predecessor's store and loop B's in-loop `vmcnt(0)` drain disappears | meaningless by design | volume only |
| `rec4` | `recx_unchain_kernel` | **intra-token**: same buffer, same addresses, same load/store mix, but each 128-link accumulator chain becomes four 32-link ones | re-associated | existing §10.1 GDN band (cos ≥ 0.999, meanrel ≤ 5 %) |
| `rec5` | `recx_both_kernel` | both at once | meaningless by design | volume only |
| `rec6` | `recx_shape_kernel` | **nothing — it is `rec4`'s control**: the same unroll-4 loop spelling with the partials folded back into one accumulator in MODE 0's own order | bit-identical to `rec0`, expected | bit-identity vs `rec0` |

`rec6` exists because the census caught `rec4` being **cheaper**, not merely shallower: four bodies per trip
fold away index arithmetic MODE 0's one-per-trip spelling leaves behind, and `traffic_census.py
--volume-check` prices that at **−160 slots (−10.5 %)**. With D25's elasticity at ~0.8, up to 8.4 % of any
`rec4` speedup would be explainable by its instruction bill alone and could not be attributed to the chain
at all. `rec6` keeps the shape, gives back the chain, and is bit-identical, which turns the pair into a
difference that means something: `rec6 − rec0` is the shape/bill effect, `rec4 − rec6` is the chain at
matched shape (residual −43 slots, ~2.3 % by elasticity). `#pragma unroll 4` is required on both loops of
both kernels for this to hold: without it LLVM fully unrolls `rec6`'s loop A (141 static loads against 29)
and the two are no longer comparable — the census's walk fingerprint refused that listing outright, which is
the right way for a geometry change to fail.

**Volume is enforced in two independent halves, and a `--burst` refuses to run unless the device half
passes.** The device half (`--volume`) writes a sentinel the recurrence cannot produce, then checks
per element *which addresses stopped being it*: a split probe's read buffer must still be byte-identical to
the state seeded into it, which proves the write pass MOVED rather than vanished; guard bands on every buffer
catch a write outside either; and the per-launch byte account is re-derived on the host from the constants
the launch is built from (the `mixed_bw` `plan_bounds` rule — a kernel may not certify its own geometry).
That half cannot separate reads from writes when a MODE uses one buffer, so the instruction half reads the
listing the binary actually runs and admits four columns — state-walk loads/stores **and** whole-token-step
loads/stores:

| kernel | slots | state ld/st | step ld/st | read | write | verdict |
|---|---|---|---|---|---|---|
| `rec_kernel` | 1,523 | 256 / 128 | 261 / 129 | 2.000 MiB | 1.000 MiB | reference |
| `rec0_kernel` | 1,523 | 256 / 128 | 261 / 129 | 2.000 MiB | 1.000 MiB | same |
| `rec3` `recx_indep` | 1,522 | 256 / 128 | 261 / 129 | 2.000 MiB | 1.000 MiB | same |
| `rec4` `recx_unchain` | 1,363 | 256 / 128 | 261 / 129 | 2.000 MiB | 1.000 MiB | same |
| `rec6` `recx_shape` | 1,406 | 256 / 128 | 261 / 129 | 2.000 MiB | 1.000 MiB | same |
| `rec5` `recx_both` | 1,473 | 256 / 128 | 261 / 129 | 2.000 MiB | 1.000 MiB | same |

Every probe moves the same 2 MiB read and 1 MiB written per launch as the shipped kernel; only the slot
column differs, and `--volume-check` prints each delta with its elasticity consequence so a reader cannot
miss it. The admission's own must-catch control is `--selftest-volume` (3 cases, `bad=0`), and its two
mutations are derived from the admission's blind spots rather than hand-listed: deleting a **state-walk
store** must be caught (it is — the census refuses the listing, since a per-trip access count that stops
dividing D is no longer the two-walk form), and deleting a **prologue operand load** must be caught by the
*step* columns, which is the witness that those columns are load-bearing and not decoration (it is: state
columns stay 256/128 while step loads go 261 → 260), with the unmutated listing required to pass.

**Commands.** CPU-only and safe for anyone:

```bash
SAVE_TEMPS=1 ./prototype/gdn_rec_isa/build.sh
.venv/bin/python prototype/gdn_rec_isa/traffic_census.py              # §8's census
.venv/bin/python prototype/gdn_rec_isa/traffic_census.py --selftest-all   # 6 + 3 checks
.venv/bin/python prototype/gdn_rec_isa/traffic_census.py --volume-check   # the table above
```

Device — main agent only, blocking, zero subagents, `tools/gpu_lock.sh` acquired; the burst takes timings and
nothing here has one:

```bash
tools/gpu_lock.sh acquire gdn_rec_isa_probe
./prototype/gdn_rec_isa/gdn_rec_isa --modes rec0,rec3,rec4,rec5,rec6 --volume   # admission first
./prototype/gdn_rec_isa/gdn_rec_isa --modes rec0,rec3,rec4,rec5,rec6 --burst 40 --reps 5
tools/gpu_lock.sh release gdn_rec_isa_probe
```

**How to read the result.** `dependency_band` classifies `rec0_over_mode` at the same >1.10 / <1.02 bounds
as the instruction falsifier, but they mean the opposite there, which is why the two functions are separate:
here **>1.10 means the chain binds** — fund D30 1.4 (occupancy) and 1.5 (chain shortening); **<1.02 means the
chain does not bind at this granularity** and `gdn_rec` needs a model D23's census has not named, which makes
the `rocprofv3` pass (`HANDOFF.md` §4) the next step rather than another lever. Three specific readings to
get right:

* **Read `rec4` against `rec6`, not against `rec0`.** The burst prints `DIFF … rec6_over_rec4` when both are
  present precisely so the loop-shape effect cancels; the residual −43 slots is worth ~2.3 % by elasticity and
  must be subtracted before calling the remainder the chain.
* **`rec4`'s own ceiling is small and calculable**: shortening two 128-link chains to 32 at ~4-cycle FMA
  latency saves ~2 × 96 links ≈ 0.09–0.17 µs of a 6.3–6.9 µs launch, i.e. **1.015–1.027×**. A much larger
  number is not the accumulator chain — look for the store drain or the dispatch gap instead. `rec3` is the
  probe whose payoff could be large, because what it removes is the cross-launch round trip plus the in-loop
  drain, and that is also where the wave's launch-to-launch gap lives.
* **A `rec6` that is not bit-identical is a finding, not a nuisance.** Its addition tree is MODE 0's term for
  term; if the device says otherwise, LLVM re-associated across the spelling change — the same class as D25's
  hoisted-`q̃` bug — and `rec4`'s numbers are then uninterpretable until it is understood. A probe whose volume
  columns differ cannot be read at all: that is the `mixed_bw` failure wearing a fast costume, and the binary
  already refuses the timing.

**What has been run, and what has not.** The admission path is a deterministic correctness gate, so it was
run: `gdn_rec_isa --modes rec0,rec3,rec4,rec5,rec6 --volume` and then the same set without `--burst`, inside
`tools/gpu_lock.sh acquire gdn_rec_isa_probe_worker` → `release`, lock free before and after, `util` 93 %
(discounted — this box serves its own tokens), wall **0.74 s** for the 4-case × 5-MODE admission (20/20
verdict lines `BIT_IDENTICAL` or `ADMITTED_BY_POLICY`, `admission=PASS`) and **0.27 s** for the volume pass
(`verdict=VOLUME_OK`). **No timing was taken**: the binary's own `NOT TIMED … no timing taken in this run` line
closes every case, and `--burst` for `rec3`–`rec6` is still unrun — that command, with the lock and a zero-
subagent window, is handed to the main agent above.

What the device said:

| MODE | gate | device verdict (all four cases) |
|---|---|---|
| `rec0` | bit-identity vs itself | `BIT_IDENTICAL`, `ndiff_*_vs0 = 0`, `rerun_ndiff_* = 0` |
| `rec6` | bit-identity vs `rec0` | **`BIT_IDENTICAL`: `ndiff_o_vs0 = 0`, `ndiff_state_vs0 = 0`** in 4/4 cases — the unroll-4 spelling does not re-associate, so `rec6` is a valid control and `rec4 − rec6` is a legitimate difference |
| `rec4` | existing §10.1 GDN band | admitted: `cos_o_vs0 = cos_state_vs0 = 1.000000`, `maxrel10` 8.99e-08…5.83e-07 — the four-partial re-association is ~1e-7, three orders inside the band it is judged by, and 217k–235k elements *are* moved, which is why bit-identity was never the right question for it |
| `rec3`, `rec5` | volume only | admitted by policy, values meaningless as designed (`cos_o_vs0` 0.15–0.75); `unwritten_o = 0`, `guard = 0`, `rerun_ndiff_* = 0` |
| all five | `--volume` | `VOLUME_OK`, `account_same_as_rec0 = 1`; the two split probes report `readside_bits_differing_vs_seeded = 0`, i.e. the read buffer really was left untouched, and `split = 1` only where the MODE claims it |

So the probes are admissible and the control is bit-identical; the only thing standing between this directory
and a statement about `gdn_rec`'s critical path is one burst window in the main agent's hands.

## 10. The burst, measured (main agent, 2026-10-16) — and it does not read the way §9 expected

Two windows, one `gpu_lock` hold each, blocking, zero subagents, `util` 92 % (discounted per
`bench/performance-status.md` §6), `--modes rec0,rec3,rec4,rec5,rec6 --volume --burst 40 --reps 5`, all four
cases. `BURST-LINES printed=100 want=100 ok=1` in both windows, `admission=PASS`, `volume=OK`. Logs:
`tmp/gdn_burst_w1.log`, `tmp/gdn_burst_w2.log`.

Best-of-5 of 40-bursts, `ms_per_sequence` at T = 128 (one value per case; the four cases agree to 1 %),
`rec0` = 0.8051–0.8141 ms/sequence = **6.29–6.36 µs per launch**, identical in both windows to 0.5 %:

| MODE | best ms/seq | `rec0_over_mode` | reading |
|---|---|---|---|
| `rec0` | 0.8045–0.8141 | 1.0000 | reference |
| `rec3` cross-token broken | 0.8723–0.8866 | **0.918–0.923** | **8.4 % SLOWER** — the dependency it removes is not what this kernel waits on |
| `rec4` intra-token chain 128→32 | 0.7443–0.7533 | 1.073–1.089 | faster, but −160 slots predicts 8.4 % of that from the bill alone (D25 elasticity) |
| `rec5` both | 0.8205–0.8335 | 0.977–0.982 | 2.2 % slower |
| `rec6` `rec4`'s shape, chain **intact**, bit-identical | 0.6984–0.7012 | **1.149–1.158** | **the fastest MODE in the harness, and it is bit-identical to `rec0`** |

**Three readings, in order of what they cost the plan.**

1. **The latency reading is refuted, on both chains.** Breaking the cross-token dependency (`rec3`) makes the
   kernel **8.4 % slower**, and at matched loop shape `rec6_over_rec4` = **0.933–0.941**, i.e. splitting the
   serial accumulator into four 32-link chains *costs* 6.5 %. Nothing that shortens a dependency chain earns
   anything here, which closes D30 items 1.3, 1.4 and 1.5 — no occupancy work, no chain re-association, and
   the `rocprofv3` pass is no longer the next step either.
2. **`rec6` is a bit-identical 1.15× on the shipped recurrence.** `ndiff_o_vs0=0 ndiff_state_vs0=0` in 4/4
   cases, and its `ndiff_*_vs_cpu` equals `rec0`'s digit for digit (216728 / 180870), which is the strongest
   witness the harness can give: same values, same addition tree, different loop spelling. It removes 117
   slots (−7.7 %), and D25's elasticity of 0.8 predicts **6.1 %**; it delivers **15 %**. So the kernel is not
   instruction-count-bound and not dependency-latency-bound on the recurrence — it is bound by **how its own
   loop is scheduled**, and an unroll-4 spelling that folds address arithmetic out of the critical path buys
   about twice what its instruction delta buys. §9's `band=` field already said this in its own vocabulary:
   `REFUTES-the-latency-reading`.
3. **D25's "instruction route is closed" needs restating, not extending.** What D25 closed was *fusion and
   hand-folded reductions* — the axis it could price. The axis that pays is **spelling**: same operations,
   same order, same values, unrolled so the compiler stops leaving address arithmetic inside the serial chain.

**What this does not yet claim, and the one thing that decides it.** This is a burst at the harness geometry
(one `16×128` grid per token-step, which *is* the engine's launch geometry, `engine/kernels_gdn.cpp:40`) over a
**1 MiB single-layer state that stays resident across the 128-token burst**. The engine walks 18 layers per
token, so its L2 working set and its interleaving with every other family are different, and D28 is precisely
the precedent that a burst winner can be an in-situ loser. The claim is therefore: **a bit-identical rec
kernel worth ~1.15× at launch level**, whose prize at 4096 lanes would be 534.54 → ~465 ms = **~70 ms ≈ 6.3 %
of the whole wave** if it survives in situ, and ~0 if it does not. Port and adjudicate in situ before any of
it enters `bench/performance-status.md`.

Instrument note, because the first three attempts at this window timed nothing: `--volume` was overloaded as
both "run the volume check" and "volume-only mode", so the documented `--volume --burst 40 --reps 5` printed
`VOLUME_OK`, exited 0, and timed **zero** launches — the `gdn_collapse/run.py` class of defect (a driver whose
parse matches nothing still prints PASS). Fixed: `--volume` alone is the no-timing pass, `--volume --burst`
checks then times, and the burst now prints `BURST-LINES printed=N want=cases×reps×modes` and **fails** if the
matrix was not fully timed.
