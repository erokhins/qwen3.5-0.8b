# A3 — DEAD AT THE SOURCE: 32,768 B is the LDS floor of the shipped arithmetic, and 20,480 B buys residency with the resource the bet is trying to free

Written 2026-10-19 on tip `b75b4b3`, against `prototype/attn_percost/A3-BET.md` (the pre-registration) and
`prototype/attn_percost/A2-census.md` §2/§5/§6 (the premise). **No engine, kernel, launcher or gate file was
touched. No device was touched. No timing was taken, by anyone, in this file** — every `ms` below is quoted
and cited, every byte figure is derived and its formula is printed beside it. The decision not to build was
taken at the source and confirmed by the supervisor before any code existed; §5 states what would reopen it.

The finding in one sentence: **`s_q`'s live range is the whole round loop, a 16x16x16 WMMA's M dimension is
fixed at 16 rows, and the epilogue transposes through the same per-warp buffer — so `3 x 8,192 + 8,192 =
32,768 B` is the floor of the shipped arithmetic with Q staged, and the only two ways to declare 20,480 B
spend either the round's dependent chain (rounds, WMMA, rescale, barriers and K/V staging all x2) or the LSU
feed that supplies it (Q re-read from global every round).** A3's prize is 1.36x from residency; both routes
pay with the exact quantity A2 §5 named as where the remaining ~1.6x lives (C7, the round's chain). That is a
refuted premise, not a risky result.

## 0. Milestone 0, as far as it goes: the shipped note block, read not asserted

A3-BET's Milestone 0 asked for the note block of the variant beside the shipped one. There is no variant, so
this is the **left column only** — taken this session, compile-only, and it reproduces §33's banner and A2 §1's
row exactly, which is what makes the rest of this file's arithmetic legal rather than inherited.

```
$ bash prototype/attn_wmma/fattn_census_build.sh /tmp/a3m0            # hipcc -O3 --save-temps -c engine/kernels_attn.cpp — CPU only, launches nothing
$ python3 prototype/attn_wmma/fattn_residency_read.py --asm kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s \
        --kernel attn_fattn_wave --threads 96 --warps 3
kernel                 : attn_fattn_wave
  vgpr_count               178
  vgpr_spill_count         0
  sgpr_count               24
  sgpr_spill_count         0
  private_segment_fixed_size 0
  group_segment_fixed_size 32768
  block threads            96
  block warps              3
device facts:
  LDS per CU             : 65536 B  [KFD nodes/1/properties lds_size_in_kb 64]
  work items per CU      : 1024     [rocminfo 'Max Work-item Per CU']
  waves per CU           : 32       [rocminfo 'Max Waves Per CU']
  register file per CU   : UNKNOWN on this part (fattn_regfile_ladder.md §3) — not applied
derived blocks/CU        : 2   (LDS=2  threads=10  wavefronts=10)
  binding limit(s)       : LDS
derived wf/CU            : 6
```

The two admission lines A3-BET Milestone 0 set for the **new** side — `group_segment_fixed_size <= 21,845` and
`vgpr_spill_count = 0` — were never reached, because no legal spelling of "half-tile Q staging" can declare
`<= 21,845 B` while the round loop still consumes 16 query rows of Q per warp (§1). `21,845 = 65,536 / 3` is
the third-block boundary and it is below the floor.

## 1. The LDS floor, derived, with a path:line on every step

1. **The second WMMA operand consumes the whole 16-row Q stage, every round.**
   `engine/attn_fattn_kernels.h:581`:
   `fattn_load_tile(&s_q[w * FATTN_MROWS * DH + fattn_q_addr(fattn_ab_row(lane), 16 * t)])`, inside
   `for (int t = 0; t < FATTN_KTILES; ++t)` (`:577-578`, `FATTN_KTILES = DH/16 = 16`, `:166`) inside the round loop
   `for (int base = lo_all; base < hi_all; base += FATTN_MLANES)` (`:549`). `fattn_ab_row(lane) = lane & 15`
   (`:283`), so every lane indexes one of **16 query rows** and the `t` loop covers **all 256 dims**, every
   round, from LDS. The stage is written once per launch (`:505-509`) and read by every round, so its live
   range is the entire round loop: `FATTN_MROWS * DH * 2 = 16 * 256 * 2 = 8,192 B` **per warp**, ×`FATTN_WARPS
   = 3` (`:204`) = 24,576 B per block.
2. **The epilogue stands on that same per-warp buffer.** `:710`:
   `fattn_transpose_out(s_q + w * FATTN_MROWS * DH, out, ...)`, and the size is asserted to be the same object
   twice over — `:216` `static_assert(FATTN_MROWS * FATTN_TROW * 4 == FATTN_MROWS * DH * 2)`
   (16 x 128 F32 == 16 x 256 F16 == 8,192 B) and `:234` the publish-coverage assert over `FATTN_PBLK`. So
   halving `s_q` halves the epilogue's staging whether or not anyone wants it to.
3. **`s_kv` is one 8,192 B block-shared stage** — `:457` `__shared__ __align__(16) __half s_kv[FATTN_MLANES * DH]`
   with `FATTN_MLANES = 16` the WMMA N dimension, asserted at `:220` and totalled at `:223`
   (`FATTN_WARPS * FATTN_MROWS * DH * 2 + FATTN_MLANES * DH * 2 == 32768`).
4. **Therefore the floor is 32,768 B, not 32,768 B of "current usage".** With Q staged there is nothing to
   shave: the two operands of the round and the epilogue's F32 transpose already occupy disjoint bytes only at
   different *times*, and the disjointness is what `s_kv`'s three-layout aliasing (and its fourth barrier,
   `:675`) pays for. `E1-result.md` §4 reaches the identical floor from the opposite direction —
   "the block's LDS floor with this epilogue is `3 x 8,192 + 8,192`, Q staged or not" — and E1's own measured
   note block is the witness: un-staging Q entirely left `group_segment_fixed_size` at **32,768**
   (`prototype/attn_wmma/E1-result.md` §1).

**Corollary, and it is the sentence that kills the bet.** `A3-BET.md:13`'s "stage Q in 8-row half-tiles"
means a 4,096 B per-warp stage, and 4,096 B is physically **8 rows x 256 dims**. A `v_wmma_f32_16x16x16_f16_w32`
tile has M = 16 by construction (`:169` `static_assert(FATTN_MROWS == 16 && FATTN_MLANES == 16)`), and the
measured fragment map (`:15` and `prototype/attn_wmma/fattn_map_results.md`) puts the C tile's row at
`lane % 16` — one warp cannot own 32 rows and cannot own 8 without half its C rows being junk. So
**"half-tile staging" is not a staging change at all: it is a work-split change.** Everything downstream of
that corollary follows.

## 2. The two reachable routes, priced and rejected

Bit-identity was never the obstacle in either — per row, the dot's tile order, the fold, the deferred rowsum
butterfly and the publish set all survive both shapes. The obstacle is what each one costs.

### H1 — "two 8-row half-passes" (the literal reading of A3-BET's sentence)

Stage rows `0..7`, run the round loop, transpose them out; then rows `8..15`. `out.t`'s live range stays inside
one half-pass, so no register cost is expected (the accumulator is 16 tiles x 8 F32 = 128 VGPR either way), and
`s_q` can be declared 4,096 B per warp — **20,480 B, the bet's number, and probably spill-free.** The price:

| quantity per (block, launch) | shipped | H1 | source of the shipped figure |
|---|---|---|---|
| rounds of the block's union range | R | **~2R** | range is per-row-block; two row-halves each sweep it |
| K/V bytes staged from global | `staged` | **2 x `staged`** | A2 §3: `staged` = 1,552,416,768 B/layer/pass at 4,096 lanes |
| `v_wmma` issues per useful row | 16 score + 16 PV | **x2** (each tile carries 8 live rows of 16) | §1's corollary |
| accumulator rescale (128 F32 mults/lane/round, `:620-622`) | per round | **x2** | same round-doubling |
| barriers | 4/round + 2 | **~x2** | `:560,633,646,675` per round, `:526,536` per launch |
| Q bytes demanded from global | 8,192 B/warp once | 8,192 B/warp once (unchanged) | each row-half is staged once per launch |

**Traffic, in the currency the brief asked for:** `1,552,416,768 B / 4,096 tokens = ` **379 KB/token/layer**
of additional K/V staging demand, i.e. **+2.27 MB/token/pass** summed over the 6 attention layers
(A2 §3's own per-pass figure 9,314,500,608 B / 4,096 = 2.27 MB/token, doubled). The comparison the brief named
is G1: its hoist added **+32 KB/token/layer** of stage traffic, which landed as **+6.06 ms outside the row it
improved** and turned a −5.91 % family row into −2.21 % of `wall_ms` (`progress/2026-10-21/performance-status-2026-10-21.md` §40).
H1's added demand is **~12x G1's in bytes**, and unlike G1's it lands inside the family the bet is about, so
the externality print would have to be read as the primary evidence, not as a footnote.

**Why that is fatal rather than merely expensive.** A2 §2 bounded the round at 5.65 % matrix-pipe and
<= 8.8 % issue-slot occupancy, so a 2x of both still fits under the roofs — but A2 §2's own conclusion is that
**84-94 % of the round's wall is neither**, and that the residual ~1.6x is C7, the round's dependent chain.
H1 doubles that chain per useful row (twice the trips, twice the barriers, twice the rescale) to buy 1.5x the
wavefronts. §31's elasticity (4->8 wf/CU = 1.647x, log-elasticity 0.72) predicts 1.36x on the residency leg;
nothing in this ledger prices a *chain doubling*, and G2 (§42) is the precedent for what happens when a
geometry change lengthens the per-token chain — 2.2x slower, with the externality printed to prove it was the
kernel and not the neighbourhood.

### H2 — 16 rows x 128 dims, re-filled inside the round (the "second pass over Q in global" the brief predicted)

The stage holds all 16 rows and **half of DH** (16 x 128 x 2 = 4,096 B). The score leg splits at `t = 8` with a
Q re-fill between, accumulating into the *same* `acc` in ascending `t` order, so the tiles consume the same
values in the same order and bits cannot move. Rounds, WMMA count, rescale and barriers per round are
unchanged — this is the route that does not lengthen the chain. The price is the feed:

* per (block, round): Q demanded from global = 3 warps x 8,192 B = **24,576 B**, against K/V's
  2 fills x 16 lanes x 256 dims x 2 B = **16,384 B** — i.e. **+150 % on this kernel's per-round global load
  demand** (shipped: Q is demanded once per launch, 24,576 B/block, not once per round);
* per (block, launch) that is **32x Q** at the shape that matters: at 4,096 lanes the widest body has
  `split = 8` (A2 §4: body *i* has `split = i+1`), so `span = ceil(4096 / 8) = 512` lanes and
  `512 / FATTN_MLANES = 32` rounds — 24,576 B x 32 = 786 KB per block launch instead of 24.6 KB;
* plus two extra LDS-publish barriers per round (the stage is rewritten between the two half-legs), and a
  second half-score leg whose operands are only half of the dot.

A2 §3's D21 caveat cuts both ways here and must be said rather than picked: demanded bytes are not DRAM bytes
(a block's whole Q tile is 24,576 B, so most of this is L1/L2-served), but §27's D57 falsifier — an
instruction-cut falsifier that made `attn` **1.521x faster with the byte census unchanged** — is on record
saying this kernel is not traffic-bound *at these widths*, which is an argument about **demanded bytes**, not
about LSU issue slots and load latency added inside a dependent round. A2 §2 already counted the round at
**96 non-matrix issue slots per matrix slot** with 146 LDS ops in the body; H2 adds to exactly that column
while A2 §5's own sentence says the bet should spend *idle* issue slots, not add to the feed of the chain.

### The sentence in A2 §5 that was wrong, corrected in place

A2 §5: "the half-tile spelling is the cheap way to reach that byte count **without touching the epilogue's
transpose** or re-spilling Q into registers (V6's refuted variant)." The first clause is false in **both**
routes: the epilogue's F32 transpose staging *is* the per-warp Q stage (`:710`, `:216`), so any 8-row Q scheme
drags the transpose with it — it would have to stage 8 rows at a time (4 passes instead of 2, exactly what
`E1-result.md` §4 foresaw) in order for the block to reach 20,480 B. The clause is corrected in place in
`A2-census.md` §5 with a dated marker; nothing else in A2 changed, and A2's numbers all survive (the census
never depended on the sentence).

## 2b. The second kill: even at 20,480 B, three blocks/CU is not arithmetically decidable here

Recorded because it makes the call unambiguous rather than close, and because it is the reason the residency
counter was made mandatory in A3-BET's item 3.

* clang's note block (above, and re-taken this session): `vgpr_count 178`.
* rocprofv3's `VGPR_Count` for the same kernel: **248** (A2 §1's table, and A2 §6 lists the discrepancy as
  unreconciled).
* At 3 blocks/CU of 96 threads the register demand is `3 x 96 x VGPR` = **51,264 VGPR/CU at 178** and
  **71,424 VGPR/CU at 248**. If the register file is 64K-class (65,536) the first fits and the second does
  not — and the part reports **no register file size at all**:
  `/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties` carries `simd_count 80`, `simd_per_cu 2`,
  `lds_size_in_kb 64`, `max_waves_per_simd 16`, `max_slots_scratch_cu 32`, and no register-file field;
  `rocminfo`'s only ceiling here is `Max Waves Per CU: 32` (`fattn_regfile_ladder.md` §3 is the write-up).

So A3 stacked two unknowns — does 20,480 B cost the chain (decidable at the source, and it does), and does the
register file even allow the third block (not decidable on this part with the facts on record). Killing the bet
before the counter is legitimate precisely because the second one cannot be settled by the cheap half of the
experiment.

## 3. What would have made this route viable — and why it is not expressible here

The route needed a form whose **per-warp Q live range is genuinely 8 rows**, i.e. one that fills 16-row WMMA
tiles from two 8-row halves **inside one round**, with no second round loop and no second K/V stage. That is
not expressible with this instruction and this fragment map:

* the tile is 16x16x16 with the C tile's row at `lane % 16` (`:169`, `:280`, `fattn_map_results.md`), so 16
  query rows per tile per warp is the only legal M; two warps cannot co-own one tile, and one warp cannot own
  32 rows;
* splitting M and issuing two half-live tiles per (round, k-tile) is H1's cost with the doubling moved inside
  the round instead of outside it — same 2x of matrix slots, rescale and LDS traffic, plus the accumulator
  problem: both row-halves' `out` would be live across the same round, which is 256 VGPR of accumulator and a
  guaranteed spill (§33's precedent);
* splitting the dot instead (H2) keeps M = 16 honest but moves the cost to the feed, because Q is *live for
  every round* — that is not a property of the staging loop, it is a property of the dot.

The only other shape that reaches 20,480 B without either penalty is noted for completeness and **not
recommended or implemented**: un-stage Q entirely (`E1`, already ported at `prototype/attn_wmma/evidence/fattn_unstage_q.diff`,
measured **184 VGPR / 0 spill / 0 scratch** on the real body) and stage the epilogue's transpose 8 rows at a
time, giving `3 x 4,096 + 8,192 = 20,480`. Its LDS arithmetic is the bet's, but its mechanism is the one
§41 closed — Q demanded from global at the use site, every round — so it inherits H2's feed cost (all 8,192
B/warp/round, not half of it) and it needs the same RF answer above at 184 rather than 178 VGPR. It is a
different bet, not A3, and it is named here only so that nobody spends a window rediscovering it.

## 4. What this cannot see

* **The residency axis is now closed twice over**: on LDS arithmetic for every Q-staged shape (§1, and the
  ladder's `wf/CU(NW) = NW x floor(8/(NW+1))` ceiling of 7 wf/CU at `fattn_regfile_ladder.md` §1), and
  undecidably on register facts for anything above 2 blocks of 96 threads (§2b). **Any future residency claim
  on this part must settle the register-file question first** — the cheap form is a `__launch_bounds__(96, N)`
  ladder over `prototype/attn_wmma/occupancy_fattn_probe.cpp` (compile N = 1..8, which forces a register
  budget per rung, and read each rung's `hipOccupancyMaxActiveBlocksPerMultiprocessor`), cross-checked against
  one `rocprofv3` pass reading `VGPR_Count` and `SQ_LEVEL_WAVES_FILLING` on the real kernel, deduplicated per
  `(Dispatch_Id, Counter_Name)` per AGENTS' rocprofv3 trap row. Named, not run: it is driver-and-device work
  and it belongs to the main agent's window, and the LDS finding in §1 makes it moot for A3 specifically.
* **No timing of any kind is in this file**, and none of §44/§45/§46's figures are re-taken here; they are
  quoted as bases (A3-BET's table) and are untouched by it.
* **The dead lines in A3-BET's table are therefore never tested.** That is not a hole in this finding: the
  lines adjudicated a build, and the build is not constructible without spending the chain or the feed. The
  six rows stay as written for whoever reopens the axis (§5), and the A/B harness the brief asked for
  (`run_ab_a3.sh`, which would have lived in this directory) was deliberately **not** written — a harness for a
  form that cannot be built is the kind of tool that later gets run against the wrong thing.
* **`attn_fattn_wave`'s shipped behaviour did not move**: no kernel, launcher, `decode_run` or `gate.py` file
  was edited, so the default path, its plan-bill row and its NUMERICS echo are byte-for-byte what `b75b4b3`
  had. Nothing needs reverting.
* What this finding would need to be **wrong**: a wave32 WMMA shape on gfx1151 with a legal M < 16 that still
  fills a 16x16x16 tile — nothing like it is on record anywhere in this tree (every WMMA rate and census here
  is `v_wmma_f32_16x16x16_*`, `prototype/attn_wmma/results.md` §2, `prototype/issue_rate_probe/results.md` §1),
  — or a proof that the round loop can sweep two row-halves with one K/V stage fill, which is the
  accumulator-live-range wall in §3, not a coding gap.

## 5. What reopens this, at what price

One measured fact would reopen it, and it is the same one §28 asked for in a different context: the price of
**one K/V stage fill plus one round** at the engine's own geometry, read against the price of one extra round.
If a paired burst showed the round's cost is dominated by the K/V stage fill rather than by its own chain (i.e.
H1's doubled rounds cost much less than 2x), the byte-count objection dissolves and H1 becomes worth a window;
if it shows the chain dominates — which is what A2 §2's two bounds and §28's flat-in-rows signature both predict
— then A3 stays dead and the residual ~1.6x belongs to `A4-CANDIDATES.md`'s inventory instead. That burst is
main-agent device work, costs one hold at 4,096 lanes with the existing window scripts
(`prototype/attn_wmma/run_fattn_burst_window.sh`, `prototype/attn_m2/burst_m16.sh`), and does not need any of
A3's code to run.
