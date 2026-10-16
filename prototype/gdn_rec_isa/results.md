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
`rec_kernel<<<NH=16, D=128>>>` (`engine/kernels_gdn.cpp:295`), wave32 (`.amdhsa_wavefront_size32 1`),
so 16 blocks × 4 wavefronts = **64 wavefronts**, and the family is 18 launches per token (18 GDN
layers). The grid is not "16 wavefronts over 40 CUs" as the earlier read put it — it is 16 *blocks*,
but the conclusion survives: 64 wavefronts is ≤1 per SIMD on this part either way (§3).

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
`rec_kernel.ship.s`. Commands, device steps and what is *not* built (no `.hip.cpp` harness yet):
`HANDOFF.md`. Limits of this write-up: it is a static census — `expf`'s real length, LDS bank
behaviour of the 512 B-stride `ds_read_b128`, and whether any wait is actually exposed are all
device questions, listed in `HANDOFF.md` §4.
