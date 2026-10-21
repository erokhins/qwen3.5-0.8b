# A5 — the geometry, costed: three re-tiles, one mechanism, and the stop rule that goes with them

Written 2026-10-19 on tip `cbadeca`, against `prototype/attn_percost/A4-BET.md`'s closing instruction (*"the
honest conclusion is that the shipping WMMA body needs to be rewritten against llama's tile geometry, and that
comes to me as a costed rewrite proposal — not as a tweak smuggled in under a familiar flag name"*).

**This is a proposal, not an implementation. No engine file, kernel, launcher or gate file was touched. No
device was touched. No timing was taken by anyone in this file** — every `ms` is quoted with its § or is a
pre-registered *prediction* (a quoted ratio applied to a quoted in-window base, labelled `pred.`); every byte
count is a sum printed beside it, every ratio is arithmetic on quoted numbers. Labelled throughout: **read**
(a `path:line`), **derived** (the formula is shown), **quoted** (a § of the ledger or of an `A*-result.md`).

Citation conventions, so that no cite resolves only in a reader's memory (AGENTS' citation-hygiene section):
**bare `§N`** means §N of `engine/performance-status.md`; **bare `:NNN`** means
`engine/attn_fattn_kernels.h:NNN` at tip `cbadeca` (line numbers in that file moved when the
`--attn-fattn-roll` knob landed, which is why `prototype/attn_percost/A3-result.md`'s cites of it read differently); a section of
another prototype document is always written with its path. `§0`–`§7` and the Appendices are this file. The two
mechanical halves of that claim are checked, CPU-only, by `prototype/attn_percost/a5_check_cites.py` (every
cited path exists; every bare `:NNN` resolves inside `engine/attn_fattn_kernels.h`).

## 0. Recommendation, in one line, and the table it belongs to

**FUND-ONE-ONLY.** Fund **G1** — the E1 un-stage (`prototype/attn_wmma/evidence/fattn_unstage_q.diff`, already
ported, zero-spill at 184 VGPR) **plus the 8-row epilogue staging `prototype/attn_wmma/E1-result.md` §4 already named** — because it
is the only candidate whose residency arithmetic clears the ship gate, and compile it **together with G0** as
the free falsifier for the whole geometry thesis. Do not fund G2 until G1's note block and the driver's
occupancy answer exist. Do not fund G3 at all; it is dead on its own budget and is written up that way in §5.

| id | geometry (M / warps / threads) | LDS per block, as a sum | blocks/CU → **wf/CU** | predicted family / wave move under §31's elasticity | verdict here |
|---|---|---|---|---|---|
| shipped | 16 rows/warp, 3 warps, 96 thr, 1 aliased K/V stage, Q staged per warp | `3×8,192 (Q) + 8,192 (K/V) = 32,768` (read `:223`) | 2 → **6** | — | baseline |
| **G0** | 16 rows/warp, **7 warps**, 224 thr, everything else shipped | `7×8,192 + 8,192 = 65,536` | 1 → **7** | family **−10.5 %**, wave **−3.2 %** | **dead as a standalone bet** (wave line is −5.65 %); fund as the *falsifier* only — costs one define in two headers |
| **G1** | 16 rows/warp, 3 warps, 96 thr, **Q un-staged (global at the use site)**, epilogue staged **8 rows at a time** | `3×4,096 (transpose) + 8,192 (K/V) = 20,480` | 3 → **9** | family **−25.3 %**, wave **−7.7 %** | **FUND** (gate first: one compile + one driver query, §4) |
| **G2** | as G1 but the three transposes **serialised through the one dead K/V stage** | `8,192 (K/V, reused by the epilogue) = 8,192` | 8 by LDS → **24**, but register-bound and undecidable on CPU facts | not claimed beyond 12 wf/CU (**−39 % family, −11.9 % wave** at 12) | **GATED** behind G1's note block + the RF probe; §20.1 hazard, §6 |
| G3 | round widened to 32 KV lanes (second K/V buffer) | `3×8,192 + 2×16,384 = 57,344` | 1 → **3** | family **+64.7 %**, wave **+19.6 %** (loses 6→3 residency to save 149 slots per 2 rounds) | **DEAD ON ARRIVAL**, §5 |

The one sentence that decides the table: **8 wavefronts/CU is the floor the wave ship gate requires** (§4.3), so
a geometry that cannot get past 7 has no budget, and the only fact that has ever held this kernel at 6 is the
**per-warp 8,192 B stage** — which is Q today and the epilogue's transpose tomorrow (`prototype/attn_wmma/E1-result.md` §4: *"the
residency axis is the epilogue's, not Q's"*).

## 1. What this proposal is allowed to assume (the closed list, with its receipts)

Nine categories are closed and none is re-opened here. Everything in this table is quoted.

| closed | the receipt |
|---|---|
| instruction class | both engines issue `v_wmma_f32_16x16x16_f16`; ours 5.56 %, llama's 15.2 % of the same 52.63 TFLOP/s roof — `prototype/attn_percost/A2-census.md` §2, `prototype/issue_rate_probe/results.md` §1 |
| issue slots | round body ≥ 1,654 slots/wavefront-round, **≤ 8.8 %** of the 231.5 G-slot/s roof — `prototype/attn_percost/A2-census.md` §2 |
| matrix pipe | **5.65 %** busy — same § |
| tile waste | 4.2–7.3 %, 1.07× at most — `prototype/attn_percost/A2-census.md` §2 C2 |
| traffic as the rate gap | staged 67 GB/s = 7 % of the 918 GB/s cache roof; D57's falsifier (a bit-identical instruction cut made `attn` **1.521×** faster with the byte census unchanged) — `prototype/attn_percost/A2-census.md` §3, §28 |
| launch structure / coverage | 48/48/42/42 reconciles from one integer; 87.5 % of rows, 98.4 % of lane-work — §41, `prototype/attn_wmma/E1-result.md` §6 |
| **the accumulator chain** | **A4-L1 measured**: 16-long chain halved, spill-free, `attn` family **−1.67 %**, wave **+0.27 %** — `prototype/attn_percost/A4-L1-verdict.md`, §47 |
| L2 (rescale batching), L3 (split round-trip) | 128 of ≥ 1,654 slots = 7.7 % budget; 10–13 % of the excess quadratic — `prototype/attn_percost/A4-L1-verdict.md` §"Why L2 and L3 are not funded" |
| the Q-staged residency family | `wf/CU(NW) = NW·floor(8/(NW+1))`, ceiling **7 wf/CU at NW=7** — `prototype/attn_wmma/fattn_regfile_ladder.md` §1 |

Two premises this proposal adds, both derived, because they change what is worth funding:

**P1 — the residency contrast with llama is 8 vs 6, not 8 vs 6-by-a-different-reason, and llama's "LDS 0" is a
report artifact.** A2 §1 transcribed the RDNA config row at
`engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:164` (written `fattn-mma-f16.cuh:NNN` below) as
`occupancy 1`; the field is **2** — the row, with the macro's own argument order
(`fattn-mma-f16.cuh:26`) is
`(256, 256, 64, 256, 2, 64, 128, 128, 64, 1, true)` = `nthreads 256, occupancy 2, nbatch_fa 64, nbatch_K2 128,
nbatch_V2 128, nbatch_combine 64, nstages_target 1, Q_in_reg true` (**read**). But their shared memory is
**dynamic**, so `LDS_Block_Size = 0` in the trace measures nothing; their own launcher sizes it
(`fattn-mma-f16.cuh:2033-2043`, **read**): with those numbers, `nbytes_shared_KV_1stage = 64 × 132 × 4 = 33,792 B`,
`nbytes_shared_mask = 16 × 36 × 4 = 2,304 B`, and `Q_in_reg` makes the total a **max**, not a sum:
`max(33,792, 33,792 + 2,304) = 36,096 B per block of 256 threads` (**derived**).
`floor(65,536 / 36,096) = 1` block/CU, so **1 × 8 warps = 8 wf/CU** — the number A2 stated, arrived at through
LDS, not through `launch_bounds`. Consequence, and it is the spine of this proposal:

```
LDS per resident wavefront   ours (shipped) : 32,768 / 3  = 10,922.7 B
                             llama         : 36,096 / 8  =  4,512.0 B   → 2.42× cheaper per wavefront
```

**P2 — their round is wider, not deeper.** `nbatch_fa` is documented at `fattn-mma-f16.cuh:13` as *"Number of KV
rows per softmax rescaling"* (**read**), and their row is 64 against our `FATTN_MLANES = 16`
(`engine/attn_fattn_kernels.h:165`, **read**), i.e. **4× fewer rescale-and-barrier events per KV lane** — while
`nstages_target = 1` is documented at `fattn-mma-f16.cuh:17` as *"1 == always load data synchronously"*
(**read**), so whatever llama's advantage is, **it is not software prefetch at this config row.** Any proposal
of ours that prices itself on prefetch is pricing itself on
something the fast kernel demonstrably does not do here.

## 2. The load-bearing question: why would a geometry change shorten stall when halving the WMMA chain did not?

Stated plainly, because if the answer is "it would not", the correct recommendation is to stop.

**2.1 What A4-L1's null actually excludes.** The shipped round costs **11.4× its own issue floor**
(**derived** from quoted numbers: per pass the family needs `568,512 block-rounds × 3 warps × 1,654 slots =
2.82e9` slots against `0.13883 s × 231.5e9 = 3.21e10` offered = 8.8 %, so wall/issue = 1/0.088 = 11.4 —
A2 §2's figures). A 16-long WMMA chain is *one link set* inside that 11.4×. Halving it moved the family 1.67 %.
So the chain is not the 11.4×, and the two candidate explanations for the 11.4× are:

* **(H-1) too few independent chains to cover any stall.** 6 wf/CU on 2 SIMDs = **3 wavefronts per SIMD**
  (**derived** from `simd_per_cu 2`, KFD, read in `prototype/attn_wmma/fattn_regfile_ladder.md` §0). Stated in one unit: a
  block-round needs `3 warps × 1,654 = 4,962` issue slots (**derived**, A2 §2's round bill) and the CU offers 2
  slots/cycle, so **2,481 cycles minimum**; the observed wall is **28,270 cycles per block-round per CU**
  (**derived**: `2.894e9 × 0.13883 = 401.8e6 cycles` ÷ `568,512/40 = 14,213 block-rounds/CU`). The CU therefore
  leaves **91 % of its issue slots idle with six wavefronts resident** — the same fact as A2's "≤ 8.8 %", read
  from the round side — which is the signature of "not enough independent work between two dependent events",
  and the reason the lever is more resident wavefronts rather than a shorter chain inside each of the same three.
* **(H-2) the per-round dependency is not the WMMA but the memory it waits for** — two serialized global
  round-trips inside every round (K fill at `:575`, V fill at `:692`, separated by barriers 3–5 because the
  stage is aliased, read `:581,691,704,733`), plus four block barriers per round.

**2.2 Why a re-tile can act on that and a re-association cannot.** A re-association shortens one link set inside
one wavefront. A re-tile changes **how many wavefronts the scheduler can put between two stalls** — the only
quantity on this engine with a measured elasticity: §31 moved **4 → 8 wf/CU and the family went 1.647×**, which
is a log-elasticity of **0.72** (quoted, `prototype/attn_percost/A2-census.md` §2 C6). Nothing else on the closed list has a measured
elasticity at all, and every other category is closed by a *measurement*, not an inference.

So the mechanism claim of this proposal is exactly one sentence, and it is falsifiable rather than rhetorical:

> the round is long because each SIMD holds three wavefronts and they stall together; a geometry that holds more
> wavefronts per SIMD buys overlap, and the only fact that has ever kept this kernel at three is a byte count —
> the per-warp 8,192 B stage.

**2.3 The honest counter-measurement, which is why this is a falsifier and not a bet.** §28 ran the *same axis*
from the other side and found **elasticity ≈ 0**: `attn_m16` GRP4 (3 blocks/CU) → GRP2 (5 blocks/CU, a 66 %
occupancy increase, half the traffic advantage given back) moved the time by **at most 5 %** across six
fixtures (quoted, §28). Two measurements of the same axis, one 0.72, one ~0. The difference is not noise:
§31's change *also* moved Q from LDS to global and halved the LDS reads, and it doubled a small number (4→8) at
512-row engine geometry, whereas §28's fixture was 64 and 16 rows on a kernel whose wall was flat in rows.
**These two cannot both be right about this geometry, and the cheapest way to find out is not another argument —
it is a shape whose residency is decidable before it runs.** That is what G0 is for, and it is why §0 funds the
gate rather than the ship.

**2.4 What each candidate does to the round's dependent sequence, per useful row per 16 KV lanes.** The chain
itself is *identical* in all four columns; that is the point, and it is what makes this proposal compatible with
A4-L1's null rather than in tension with it.

| per useful row per round | shipped | G0 | G1 | G2 |
|---|---|---|---|---|
| chained score WMMA links (`FATTN_KTILES = 16`, read `:166`, roll at `:636`) | 16 | 16 | 16 | 16 |
| independent P·V WMMA (`FATTN_VTILES = 16`, read `:167`, unrolled leg) | 16 | 16 | 16 | 16 |
| block barriers | 4 (`:581,691,704,733`, read) | 4, but the whole CU is in the one block's barrier | 4 | 4 |
| **exposed global round-trips inside the round** | 2 (K `:575`, V `:692`, aliased) | 2 | 2 **+ the Q operand's global load is now in the chain head** (16 loads/lane-round move from `ds_load_b128` to `global_load_b128`, `prototype/attn_wmma/E1-result.md` §5, read) | as G1 |
| accumulator rescale | 128 F32 mults/lane-round (`:680`, read) | same | same | same |
| fully-masked union rounds per block | 48-row union | **112-row union** (up to +7 rounds on ≥ 32, §3.1) | 48-row union | 48-row union |
| **resident wavefronts per SIMD** | **3** | **3.5** | **4.5** | **≤ 12 claimed** |

Only the last row is a mechanism the ledger has measured. That is the whole argument, and §7 states what happens
if it is wrong.

## 3. The candidates, as numbers

Conventions. `DH = 256` (`prototype/attn_decode/attn_decode_kernels.h:26`, quoted from
`prototype/attn_wmma/fattn_regfile_ladder.md` §0). A WMMA is 16×16×16 and **M = 16 rows per warp is not movable**
(`prototype/attn_percost/A3-result.md` §1 corollary — an 8-row stage is physically 8 rows and the C tile's row is `lane % 16`), so
*none* of these changes M: "re-tiling" here means warps-per-block, the stage plan, and the epilogue's byte
allocation. LDS per CU = 65,536 B (KFD `lds_size_in_kb 64`, quoted `prototype/attn_wmma/fattn_regfile_ladder.md` §0); ceilings
`Max Waves Per CU 32`, `Max Work-item Per CU 1,024`, `simd_per_cu 2` → 40 CU / 80 SIMD (same §). **HIP's
capability answers and clang's `Occupancy [waves/SIMD]` remark are not used anywhere in this file** (AGENTS'
counter/occupancy trap row).

### 3.1 G0 — seven warps, Q staged, shipped epilogue: the ceiling of the Q-staged family, as a falsifier

* **Geometry.** `FATTN_WARPS 3 → 7`, `FATTN_THREADS 96 → 224` (`engine/attn_fattn_kernels.h:204-205`, read),
  and with it `FATTN_WARPS_PER_BLOCK 3 → 7` (`prototype/attn_decode/attn_decode_kernels.h:543`, read), because
  the two are one fact asserted at `:211` — and §34 is the precedent for what happens when they move in
  separate commits (the tip did not build). M stays 16/warp; the block now owns **112 rows of one head**;
  `fattn_rtiles_of(rows)` = `ceil(512/112) = 5` instead of 11 (`prototype/attn_decode/attn_decode_kernels.h:561`, read), so **grid.z changes** — the one
  candidate here that does, and the reason its A/B canary must be the grid+LDS and not the plan bill alone.
* **LDS per block, as a sum.** Q: 7 warps × 16 rows × 256 dims × 2 B = `7 × 8,192 = 57,344`; K/V stage:
  16 lanes × 256 dims × 2 B = `8,192`; transpose rides the per-warp Q stage as shipped (`:216`, `:768`, read).
  **Total 65,536 B = the entire CU budget.** Per resident wavefront: `65,536 / 7 = 9,362.3 B`.
* **Residency.** `floor(65,536/65,536) = 1` block/CU → **7 wf/CU**. Threads `224 ≤ 1,024` ✓; waves `7 ≤ 32` ✓;
  registers `1 × 224 × ~184 = 41,216 VGPR/CU`, under any plausible file. **This is the only candidate whose
  residency is decidable without the register-file question** — and the driver has already confirmed the shape:
  the ladder's §0 canary row records **"1 at 64 KiB"** as a driver-confirmed stub answer (`prototype/attn_wmma/occupancy_fattn_probe.cpp`,
  `prototype/attn_wmma/evidence/occupancy-fattn.txt`, quoted).
* **Where it comes from.** Not invented here: `prototype/attn_wmma/fattn_regfile_ladder.md` §1 derives
  `wf/CU(NW) = NW·floor(8/(NW+1))` = **7 at NW=7** and names its price in the same line — *"224 threads,
  `LDS = 65536` = the entire budget, one block/CU, and a 112-row union the round loop pays for"* (quoted).
* **Traffic — it goes down, not up.** Staged K/V bytes scale with block-rounds, which scale with `rtiles`:
  `94,752 × 5/11 = 43,069` block-rounds/layer → `43,069 × 16,384 B = 705.6 MB/layer` →
  **172 KB/token/layer against 379 today (−207 KB/token/layer)** (**derived** from A2 §2's round count and A2 §3's
  per-fill size). Offset, and it is real: a 112-row union has a wider `hi_all − lo_all` than a 48-row union, and
  with rows differing in `lanes_of` by ≤ 111 the extra is ≤ 7 fully-masked rounds on ≥ 32 → **≤ +22 % rounds
  against −55 % blocks** (**derived**; the ladder's "112-row union" warning, quantified rather than waved at).
* **Why it is called dead in this document anyway.** §4.3 derives the ship gate: the wave line needs
  **−18.6 % of the family**, which needs **8 wf/CU** under §31's elasticity. G0 buys 7 → family −10.5 %, wave
  **−3.2 %**, i.e. at the dead line by the ledger's own convention. Its extra risk is structural, not marginal:
  at **1 block/CU the four barriers per round have no second resident block to overlap with**, so the very
  quantity it buys (wavefronts) arrives barrier-coupled — the §20.1-of-occupancy shape. **Its value is as a
  falsifier**: one define in two headers, one compile, zero numerics risk, and if the wave moves at all on a
  6→7 step then the elasticity is real and G1 is priced by measurement rather than by extrapolation.

### 3.2 G1 — the recommended one: Q out of LDS, epilogue staged 8 rows at a time (3 blocks/CU)

* **Geometry.** Warps/threads unchanged (3 / 96). Two changes: **(a) Q stops being staged** — the score leg's
  SECOND operand is read from global at the use site, which is exactly `prototype/attn_wmma/evidence/fattn_unstage_q.diff`
  (258 lines, `git apply --check` clean, `fattn_load_q_tile` with *"the same `__float2half_rn(fattn_bf(x))`
  conversion `fattn_stage_row` applied on the way in"*, `prototype/attn_wmma/E1-result.md` §5, read) — measured on the real body at
  **184 VGPR / 0 spill / 0 private** (`prototype/attn_wmma/E1-result.md` §1, read). **(b) the epilogue's F32 transpose stages 8 rows
  at a time instead of 16**, i.e. 4 passes instead of 2 — the first of the two routes `prototype/attn_wmma/E1-result.md` §4 names
  (*"stage the epilogue 8 rows at a time instead of 16: `3 × 4,096 + 8,192 = 20,480 B` → 3 blocks/CU = 9 wf/CU,
  at the cost of 4 transpose passes instead of 2"*, quoted).
* **LDS per block, as a sum.** Transpose stage: 8 rows × `FATTN_TROW = DH/2 = 128` F32 × 4 B = `4,096 B` per
  warp × 3 = `12,288`; K/V aliased stage `16 lanes × 256 × 2 = 8,192`; Q stage: `0`.
  **Total 20,480 B.** Per resident wavefront: `20,480 / 3 = 6,826.7 B` — versus 10,922.7 shipped, i.e. 1.60×
  cheaper per wavefront, and the same order as llama's 4,512 (P1).
* **Residency.** `floor(65,536 / 20,480) = 3` blocks/CU → **9 wf/CU (4.5/SIMD)**. Register check, stated with
  its honest limit: 3 blocks × 96 threads × 184 VGPR = **52,992 VGPR/CU**, which fits a 64K-class file (65,536)
  with 19 % of the file spare, and fits anything larger. It would **not** fit at 248 VGPR (`71,424 > 65,536`) — and the
  trace has reported 248 for the shipped kernel while the object says 178 (`prototype/attn_percost/A3-result.md` §2b, quoted), and
  **no RF size is reported on this part at all** (same §). So the CPU answer is "3 blocks if the grant is 184,
  2 blocks if it is 248", and §4.2 is the cheap witness that settles it instead of guessing.
* **Round sequence.** Identical to shipped except the Q operand's 16 loads per lane-round move from `ds` to
  `global` (`prototype/attn_wmma/E1-result.md` §5's ISA note, read). Barriers, rescale, tile order, fold alignment, publish set all
  unchanged.
* **Bit-identity.** Preserved by argument of the same shape E1 gave for its own port: same 16 source bytes per
  tile, same `bf16 → f32 → f16 RN` conversion, same clamp, same order, and the epilogue change moves *staging
  addresses in LDS only*, never the published values. **Plus one argument this file adds and E1 did not need**:
  a row's round boundaries do not move, because `lo_all` is the union minimum and every covered row of a body
  shares `blo = blockIdx.y · span` (`:360-362`, read) — so a row's slice is partitioned into the same 16-lane
  rounds in every geometry here, and extra union rounds are exact no-ops (masked cells give `sc = −1e30`,
  `rm = max(m, −1e30) = m`, `rescale = expf(0) = 1`, `e = 0`). **Therefore no numerics band re-opens** — but
  this is an argument, not a measurement, and the witness is named in §4.2, not here.
* **Traffic, in the currency A3 used.** Q demanded from global per block-round = `3 warps × 8,192 B = 24,576 B`
  (vs 0 today) → per layer `94,752 block-rounds × 24,576 = 2.329e9 B` → **568 KB/token/layer**, i.e.
  **1.50× the +379 KB/token/layer magnitude A3 priced H1 at**, and per pass `13.97 GB` against A2 §3's
  `staged 9.31 GB` (**derived**). Rate: `13.97e9 / 0.13883 s = 100.6 GB/s` of additional demanded traffic on
  top of the shipped `staged 67 GB/s` → **167 GB/s total = 18 % of the 918 GB/s cache roof** (**derived**; the
  roof and its caveat are A2 §3's and §6's). Which ceiling binds first, said rather than picked: **not DRAM** —
  the unique Q of a launch is `3,080 blocks × 24,576 B = 75.7 MB/layer` (derived) and one block's own Q is
  24,576 B, re-read once per round (`~32×` at the widest body, A3's arithmetic), so the hot Q per CU at G1's
  3 blocks is `3 × 24,576 = 73,728 B` — L1/L2 material by construction; **the first ceiling is VMEM
  queue depth and L1/L2 latency inside the score leg**, and no counter in this tree has ever measured it (§4.2).
  The counter-evidence is on the record and is the reason this is fundable: **§31 measured this exact un-stage as
  a 1.20× family win when it doubled residency**, and **§28 measured a 1.8× demand increase as time-neutral**.

### 3.3 G2 — G1 with the three transposes serialised through the one dead K/V stage: 8 blocks/CU on LDS, and the register file decides the rest

* **Geometry.** As G1 (Q from global, 3 warps / 96 threads), but the epilogue does **not** get a per-warp stage:
  the three row-blocks transpose through `s_kv` one after another, behind block barriers. This is `prototype/attn_wmma/E1-result.md`
  §4's second named route: *"serialise the three transposes through the one `s_kv` stage: 8,192 B → 8 blocks/CU
  on LDS, but that puts the epilogue behind block-wide barriers per warp and would then be register-file-bound
  at an unknown file (§3's canary is a precondition for even claiming it)"* (quoted).
* **LDS per block, as a sum.** K/V aliased stage `8,192`, **reused** as the transpose stage after the round loop
  is dead; Q `0`; own transpose stage `0`. **Total 8,192 B.** Per resident wavefront `8,192 / 3 = 2,730.7 B`.
* **Residency.** `floor(65,536/8,192) = 8` blocks/CU → **24 wf/CU on LDS alone** (threads `8 × 96 = 768 ≤ 1,024`
  ✓; waves 24 ≤ 32 ✓). **The register file almost certainly refuses it and that is the point**: 8 blocks × 96 ×
  184 = **141,312 VGPR/CU**, which exceeds the 131,072 that llama's own `__launch_bounds__(256, 2)` at 256 VGPR
  implies (**derived** from A2 §1's traced 256 VGPR and the row read in P1) while sitting under the 196,608 that
  the driver *granted* V6 (`prototype/attn_wmma/fattn_regfile_ladder.md` §3, quoted). **No CPU fact on this part can arbitrate
  that**, so no claim above 9 wf/CU is made anywhere in this document without the probe.
* **Why it is gated and not funded.** Two independent reasons, both cheap to state. (i) Its payoff is not
  claimed: extrapolating §31's 0.72 elasticity to 24 wf/CU predicts family −63 %, which is a number nobody
  should quote from an elasticity measured over 4→8. At the largest step this file will name, **12 wf/CU, the
  prediction is family −39.3 %, wave −11.9 %** (derived, §4.3). (ii) It is the one candidate that walks into
  §20.1: the file's own note says the per-warp staging is *"per warp on purpose … three warps transposing
  through `s_kv` would write one another's rows"* (`:376-386`, `:761-767`, read), and §20.1 is the bug where two
  self-consistent layouts joined by a branch passed every per-form check and still returned half of every
  streaming row as the other group's answer. A serialised epilogue is legal, but it needs a new publish-set
  proof and a new coverage assert beside `FATTN_PBLK` (`:234`, read). **Fund the proof-of-layout on CPU first;
  the device only after.**

### 3.4 G3 — widen the round to 32 KV lanes: written up so that it is not re-proposed

The one candidate class that does *not* touch residency: two 16-lane sub-tiles per round, one fold/rescale per
32 lanes instead of two. Its LDS: a second 32-lane K and V stage → `3 × 8,192 (Q) + 2 × 16,384 = 57,344 B` →
**1 block/CU → 3 wf/CU**. What it removes per 32 KV lanes is slots: 128 rescale multiplies + ~16 `l`-updates +
17 fold/cvt + 4 barrier slots (A2 §2's histogram for the rescale, `fmul 145 — of which 128 are the accumulator
rescale`, quoted) ≈ **149 of ≥ 1,654 slots per 16 lanes**, i.e. the same 7.7 %-of-a-round family that
A4-L1-verdict already called below the bar for L2, now paid for with a **halving of residency** (6 → 3 wf/CU,
predicted family time **×1.647, i.e. +64.7 %**, by §31's elasticity applied symmetrically to a halving — the
same doubling it measured, inverted; derived §4.3). **DEAD ON ARRIVAL.** The verdict does not depend on that
extrapolation: under §28's measured null (occupancy elasticity ≈ 0) the residency loss costs nothing and the
family keeps only the 3.9–7.7 % slot saving, which is under the pre-registered line either way. And the variant
that keeps LDS at 32,768 by doing two sequential 16-lane fills inside one round keeps the barriers (they guard
the aliased stage per fill), so what it saves is only the rescale and the fold — 3.9 % of slots on a machine at
8.8 % slot utilisation.

## 4. Residency arithmetic, its honest limits, and the cheapest witness per candidate

### 4.1 The arithmetic, once, with the source of every divisor

```
LDS/CU                = 65,536 B      KFD nodes/1/properties lds_size_in_kb 64        [read, ladder §0]
CU count, SIMD/CU     = 40, 2         rocminfo / KFD simd_count 80, simd_per_cu 2      [read, ladder §0]
waves/CU ceiling      = 32            rocminfo 'Max Waves Per CU'                       [read, ladder §0]
work-items/CU ceiling = 1,024         rocminfo 'Max Work-item Per CU'                   [read, ladder §0]
blocks/CU(LDS_block)  = floor(65,536 / LDS_block)                                       [derived]
wf/CU                 = warps_per_block × blocks/CU, then min(32, floor(1024/threads))  [derived]
```

Never used: `hipGetDeviceProperties` (halved SM count on this SKU), clang's `Occupancy [waves/SIMD]` remark (a
constant), `hipOccupancyMaxActiveBlocksPerMultiprocessor` **as a prediction** — the last is used only as the
driver's own *answer* to be compared against the arithmetic, which is what §31/§33 did with the engine's
`wf_per_cu=` banner (`engine/kernels_attn.cpp:143-180`, read).

| shape | LDS_block | by LDS | by waves | by threads | binding | **wf/CU** |
|---|---|---|---|---|---|---|
| shipped | 32,768 | 2 | 10 | 10 | LDS | 6 |
| G0 | 65,536 | 1 | 4 | 4 | LDS | 7 |
| G1 | 20,480 | 3 | 10 | 10 | LDS **if the grant is 184** | 9 |
| G2 | 8,192 | 8 | 10 | 10 | **unknown register file** | ≤ 24, unclaimed |
| G3 | 57,344 | 1 | 10 | 10 | LDS | 3 |

### 4.2 The register-file limit, and the cheapest thing that settles each candidate

The limit, restated because every candidate above 9 wf/CU inherits it: clang's note block says **178** VGPR for
the shipped kernel, `rocprofv3`'s `VGPR_Count` says **248** for the same kernel, and **the part reports no
register-file field at all** (`prototype/attn_percost/A3-result.md` §2b; `prototype/attn_wmma/fattn_regfile_ladder.md` §3, all quoted). A3's own conclusion
stands and is not re-litigated: *"Any future residency claim on this part must settle the register-file question
first."* This document therefore proposes nothing above 9 wf/CU on CPU facts, and every candidate below carries
its witness:

| candidate | cheapest witness that settles residency, in order of cost |
|---|---|
| **G0** | **compile only**: `bash prototype/attn_wmma/fattn_census_build.sh <dir>` then `python3 prototype/attn_wmma/fattn_residency_read.py --asm … --kernel attn_fattn_wave --threads 224 --warps 7`. Admission: `group_segment_fixed_size == 65,536`, `vgpr_spill_count == 0`, `private_segment_fixed_size == 0`, derived blocks/CU == 1. No register question at all. Driver confirmation already on file for the shape ("1 at 64 KiB", ladder §0 canary). |
| **G1** | (1) same compile read with `--threads 96 --warps 3`: `group_segment_fixed_size == 20,480`, spill 0, `vgpr_count ≤ 190` (E1's own admission bound, quoted). (2) **no device yet**: `bash prototype/attn_wmma/build_occupancy_fattn.sh` extended with a `__launch_bounds__(96, N)` ladder for N = 1..8 on the *new* body — one `hipOccupancyMaxActiveBlocksPerMultiprocessor` per kernel, **no kernel launched, no timing** (`prototype/attn_wmma/occupancy_fattn_probe.cpp` header, read). The ladder rung at which the driver's answer stops rising **is** the register-file answer, and it is the experiment `prototype/attn_percost/A3-result.md` §4 named and nobody ran. (3) **then the device**, correctness-only: one `decode_run --attn-fattn …` step reading the `wf_per_cu=` and `scratch=` fields the launcher already prints (`kernels_attn.cpp:143-180`, read) — that is how §31 and §33 read residency, and it is under the worker time box by construction. (4) **the counter, once**: one `rocprofv3` pass on the real kernel reading `SQ_LEVEL_WAVES_FILLING` and `VGPR_Count`, deduped per `(Dispatch_Id, Counter_Name)`, matched on the kernel-name **prefix** (`attn_fattn_wave(CoreArgs)`), pass count **derived** from a kernel with known per-pass sites (`attn_prep` = 48 at 4,096 lanes) exactly as `prototype/attn_wmma/w2_reanalysis.py` does — all of that is the AGENTS rocprofv3 trap row applied, and it is main-agent window work. Bonus from the same pass: `SQ_WAIT_*` / VALU-busy, the one quantity A2 §6 and A4-CANDIDATES §4 both name as the missing adjudicator of C7 and which nothing in this tree has. |
| **G2** | G1's (1)+(2) **plus** a CPU layout proof first: a new coverage assert beside `FATTN_PBLK` and a re-derived publish set for the serialised epilogue, checked by `prototype/attn_wmma/fattn_ship_census.py` (which is the tool that refuses a listing that disagrees with the source's geometry, §33). Device witness as G1, with `SQ_LEVEL_WAVES_FILLING` **mandatory** — claiming > 9 wf/CU on this part without it is the exact thing A3 §2b forbids. |
| **G3** | none — it is dead before a witness is owed (§5). |

### 4.3 The bar, and the number it requires

The convention is A4-BET's and §47's: **dead lines at base − 3 %** (§37's paired-window floor — *"a dead line
inside the noise floor is unfalsifiable"*), live lines as quoted ratios from A4-BET's table, and `wall_ms` plus
the 16k/32k rows are **ship gates**. Attention's share of the wave is **30.37 %** (§47, quoted).

```
wave live line (A4-BET)      428 / 453.65 = 0.9435  → −5.65 %                     [derived, quoted inputs]
family move the wave needs   5.65 % / 0.3037        = 18.6 %                        [derived]
residency ratio that buys it factor = 1/(1−0.186) = 1.228
                             ratio = 1.228^(1/0.72) = 1.330   → 6 × 1.33 = 8 wf/CU  [derived]
```

Predictions under §31's elasticity (log-elasticity 0.72, quoted), family base **136.2548 ms** and wave base
**446.2689 ms** in-window (`prototype/attn_percost/A4-L1-verdict.md`, quoted), with the honest note that these are in-window bases and
the harness must re-derive ratios inside its own window (AGENTS' re-based-line-set defect, §47):

| candidate | wf/CU | family factor | family ms (pred.) | family line −10.5 % | wave move | wave line −5.65 % |
|---|---|---|---|---|---|---|
| G0 | 7 | 1.117 | 122.0 | passes | −3.2 % | **fails** |
| **G1** | 9 | 1.339 | 101.8 | passes | **−7.7 %** | **passes** |
| G2 (at 12, the most this file claims) | 12 | 1.648 | 82.7 | passes | −11.9 % | passes |
| G3 | 3 | 0.607 | 224.5 | fails | +19.6 % | fails |

The wide rows: under the same elasticity the family factor is a property of wavefronts, not of context width,
and attention's share of the wave **grows** with context (§46: `b` is the quadratic term and attention owns it),
so 16k/32k should move **at least** as much as 4k. That is the reverse of A3-BET's framing ("the wide rows carry
the verdict"), and it should be checked rather than assumed — if the wide rows do *not* move while 4k does, the
change is a short-context artifact and the bet is not shipped.

## 5. The dead-on-arrival filter, applied to my own candidates before any of them was recommended

The bar is the existing one, not a new one: **live ≤ −10 % on the `attn` family with `wall_ms` and the 16k/32k
rows as ship gates; dead > −3 %** (§47, §37's floor, A4-BET's ratios). A candidate whose *entire budget* is
under that bar is called dead here the way `prototype/attn_percost/A4-L1-verdict.md` called L2 and L3 dead — arithmetic, not taste.

| candidate | entire budget, best case | bar | called |
|---|---|---|---|
| G0, 7 warps, 7 wf/CU | family −10.5 %, **wave −3.2 %** | wave ship gate needs −5.65 % | **dead as a bet**, funded only as the falsifier (§3.1) |
| G3, 32-lane round | slots saved ≈ 149/2 rounds = **3.9–7.7 % of a round's slots** on a machine at **8.8 % slot utilisation**, against residency −50 % (family ×1.647 extrapolated, ×1.0 under §28's null) | same as L2's already-dead 7.7 % budget | **dead on arrival under either hypothesis** |
| K/V un-aliased on its own (2 stages, Q still staged) | LDS `3×8,192 + 2×8,192 = 40,960` → 1 block/CU → **3 wf/CU**; removes one exposed global round-trip and one barrier per round | loses 6→3 residency to save 2 slots + 1 latency of 28,270 | **dead**: it is G3's mistake with a smaller bill |
| Q-staged re-tile at NW = 4/5/6 | `wf/CU(NW) = NW·floor(8/(NW+1))` = **4 / 5 / 6** (ladder §1, quoted) — no gain at all over the shipped 6 | zero residency delta | **dead by the ceiling formula**; the family's ceiling is 7, at NW=7 = G0 |
| round-widening *without* a second stage (two sequential fills, one fold per 32 lanes) | barriers unchanged (they guard the aliased stage per fill); only rescale+fold halve → **3.9 % of slots** | below L2's already-dead budget | **dead** |
| **llama-shaped rewrite** (256 threads, `occupancy 1-2`, Q in registers with deliberate scratch, `nbatch_fa = 64`) | their whole family is **51.65 ms traced against our 138.84** (A2 §1, quoted) → parity would be family −63 %, wave **−19 %** — far above the bar | passes the bar handily | **NOT dead, and not funded by this document**: it is the escalation if G1's residency lands and the time does not move, and it needs its own proposal (§7) |

Nothing in the funded set is below the bar: **G1's predicted −25.3 % family / −7.7 % wave clears every line with
margin**, and the only reason it is a *gate* rather than a straight build order is that the quantity it buys is
measured two contradictory ways in this ledger (§2.3) and the register-file fact is still missing.

## 6. What each candidate gives up

| | given up | how bad, and what witnesses it |
|---|---|---|
| **G1 / G2** (G0 keeps Q staged) | The header's stated precondition. `engine/attn_fattn_kernels.h` says *"There is deliberately no global-Q loader here any more"* and *"the Q stage is not a lever, it is a precondition"* (the `FattnQ`/Bet-1/V6 note block, read). G1/G2 reverse that decision **in the open**, with the reversal written into the header, not left to a diff. | documented reversal; E1's port already renames the one spelling `fattn_q_addr → fattn_q_row` so the §20.1 one-address-expression rule moves *with* the object rather than being forked beside it (`prototype/attn_wmma/E1-result.md` §5, read) |
| **G1 / G2** | Bit-identity is owed as an **argument**, and this time it is E1's argument plus the round-alignment argument in §3.2 — not a measurement (G0's is the same argument with the union-range extension in §3.1). **No band re-opens** (unlike L1/L2, which re-associated the dot and had to be judged by the §10.1 cos/meanrel bands), but only if `prototype/attn_wmma/run_fattn_admission.sh` §1 and `engine/gate.py` return `ndiff = 0` against the shipped form. | `prototype/attn_percost/A4-BET.md` "Numerics"; `prototype/attn_wmma/E1-result.md` §5 ("bit-identity is an argument, not a measurement") |
| **G0** | 1 block/CU ⇒ the four barriers per round have no second resident block to overlap with; grid.z changes `11 → 5`, so the launcher's `fattn_rtiles_of` and the kernel's `FATTN_WARPS` must move **in one commit** or the tip does not build — §34 is that exact precedent (`static assertion failed due to requirement '16 == 3 * 16'`) | the assert at `:211` is the compile-time guard; `engine/build.sh` on the *committed* tree is the real guard (AGENTS' two-header trap) |
| **G1** | 2 extra epilogue transpose passes (4 instead of 2) and their barriers, once per launch not per round; Q's 16 operand loads per lane-round move from the DS queue into the VMEM queue, i.e. the round's **head** gets longer even as the SIMD gets more chains — the honest risk is that A3's H2 feed objection is right and §31's measured win was mostly the residency doubling | §4.2's counter set; `SQ_WAIT_LGKM`/L1 pressure from the same `rocprofv3` pass |
| **G2** | additionally: the epilogue's **per-warp independence**, which is documented as deliberate (`:761-767`, read) and is the exact shape of §20.1's aliasing bug — three warps publishing into one stage with a branch between two layouts. This is the highest-review-risk item in the file and it is why the layout proof precedes the device. | new coverage assert beside `FATTN_PBLK` (`:234`), `prototype/attn_wmma/fattn_ship_census.py`, and `prototype/attn_tile_model.py`'s derived must-catch/must-not-catch sets extended to the serialised publish set |
| **all A/B sides** | **plan-bill / launch-count invariance is kept and must be asserted, not assumed.** G1/G2 keep the grid `(N_Q, split, rtiles)` and the launch counts (48/48/42/42, §41) exactly; **G0 does not** (grid.z 11→5). Because every A/B here assumes launch invariance, the behavioural canary inside the window must be a field the change is *required* to move — here `group_segment_fixed_size` (32,768 → 20,480/8,192/65,536) read out of the **linked binary** by `prototype/attn_percost/a4_l1_note_read.py`, plus the launcher's printed `q=stage → q=global` and `wf_per_cu=` fields. A canary compiled into the binary under test is blind to a stale binary (AGENTS, 2026-10-17; `prototype/attn_wmma/E1-result.md` §5 makes the same point about its own banner). | `prototype/attn_percost/a4_l1_canary.py` / `prototype/attn_percost/run_ab_a4l1.sh` are the reference harness shape, including the three §47 harness defects it now refuses |

## 7. The stop rule, and what would have made this document say "stop" instead

The brief asked me to say plainly if I could not articulate the mechanism. I can (§2.2), but it is one-legged —
it rests entirely on wavefronts-per-SIMD being the binding variable — and this ledger contains a direct
measurement of that variable doing nothing (§28's GRP4→GRP2, ≤ 5 %). So the honest structure of the
recommendation is a **falsifier with a pre-registered stop**, not a bet:

1. **Compile G1 and G0 together** (§4.2's reads). If G1's note block cannot reach `20,480 B` at **spill 0 and
   private 0**, G1 is dead on §33's precedent (spilled-at-8 lost to spill-free-at-6) and nothing further is built.
2. **Run the `__launch_bounds__(96, N)` occupancy ladder** — driver answer, no kernel launched. If it refuses
   3 blocks at 20,480 B, the register file has answered the question A3 §2b left open, the residency axis is
   closed for the third time, and **this is where attention funding stops** with a fact instead of a wish.
3. **One paired window** for G0 and G1 (both sides inside one hold, medians of 3, `--ids-file` for the wide
   legs, canary as §6's, verdict parser colour-tested, `SUMMARIZE=1` — all of §47's harness discipline).
   * **Wave moves ≥ −5.65 % and the family ≥ −10 %** → residency is real at engine geometry, ship G1, and the
     llama-shaped rewrite earns its own costed proposal with a *measured* elasticity instead of §31's.
   * **Residency confirmed on counters (`wf_per_cu` 9) but the wave moves inside ±3 %** → §28 was right about
     this kernel family and §31's elasticity does not transfer. **Then the recommendation is DO NOT FUND
     ATTENTION, and this document's successor should be a `b`-term-only discussion of the split path (L3, wide
     context) or nothing.** Three routes have already died to reach that sentence; a fourth null should end the
     line of work rather than fund a fifth.
   * **Anything between** → unresolved, not shipped, in `prototype/attn_percost/A4-BET.md`'s own words ("Anything
     between a live and a dead line is **unresolved and not shipped**", with G1 as the precedent it cites).

The reason this is the right thing to fund and not another round of arguments: the whole gate costs **two
compiles and two driver queries** before any device minute, it re-uses a port that already exists and was
measured spill-free, and it is the *only* experiment in the tree that decides which of the ledger's two
contradictory occupancy measurements describes the shipping kernel.

## Appendix A — labelling ledger, and the two census corrections this file found while reading

Corrections found while pricing (both are in the direction of "the contrast is less flattering to us than A2 §1
recorded, for a different reason than A2 gave"):

1. **llama's config row says `occupancy 2`, not 1** (`fattn-mma-f16.cuh:164` read against the macro at
   `fattn-mma-f16.cuh:26`);
   A2 §1's transcription said 1. The effective residency A2 stated (8 wf/CU) survives, but it comes from the
   block's own **dynamic** shared size (`fattn-mma-f16.cuh:2033-2043` → 36,096 B derived), which caps it at 1
   block/CU on this
   part's 64 KiB — not from `__launch_bounds__`.
2. **llama's traced `LDS_Block_Size = 0` is a report artifact of dynamic shared allocation**, not an absence of
   staging. Their derived LDS/wavefront (4,512 B) is **2.42× cheaper than ours** (10,922.7 B), which is the
   quantitative form of "the geometry is the problem" and is the single number this proposal is built to move.

Number-by-number provenance, for the reviewer who wants to check a byte:

| figure | value | label | source |
|---|---|---|---|
| shipped LDS/block | 32,768 B | read | `engine/attn_fattn_kernels.h:223` (the assert is the sum) |
| per-warp Q stage | 8,192 B | derived | `16 rows × 256 dims × 2 B`, decl `:473` |
| K/V stage | 8,192 B | derived | `16 lanes × 256 dims × 2 B`, decl `:478` |
| G0 LDS/block | 65,536 B | derived | `7 × 8,192 + 8,192` |
| G1 LDS/block | 20,480 B | derived | `3 × (8 × 128 × 4) + 8,192`; route named in `prototype/attn_wmma/E1-result.md` §4 |
| G2 LDS/block | 8,192 B | derived | `prototype/attn_wmma/E1-result.md` §4's second route |
| G3 LDS/block | 57,344 B | derived | `3 × 8,192 + 2 × (32 × 256 × 2)` |
| LDS/wavefront, ours vs llama | 10,922.7 vs 4,512 B | derived | `32,768/3`; `36,096/8` |
| blocks/CU, all | floor(65,536/LDS) | derived | KFD 64 KiB (ladder §0) |
| round count G1's Q demand | 94,752 block-rounds/layer | quoted | `prototype/attn_percost/A2-census.md` §2 |
| G1 extra Q demand | 568 KB/token/layer; +100.6 GB/s; 18 % of cache roof | derived | §3.2 |
| H1's comparable magnitude | +379 KB/token/layer | quoted | `prototype/attn_percost/A3-result.md` §2 H1 |
| staged K/V, shipped | 379 KB/token/layer, 67 GB/s | quoted | `prototype/attn_percost/A2-census.md` §3 |
| G0 staged K/V | 172 KB/token/layer, 30.5 GB/s | derived | §3.1 |
| matrix pipe / issue roof use | 5.65 % / ≤ 8.8 % | quoted | `prototype/attn_percost/A2-census.md` §2 |
| round/wave issue ratio | 11.4× | derived | `1/0.088`, from the two rows above |
| wall cycles per block-round | 28,270 | derived | §2.1 |
| residency elasticity | 0.72 (4→8 = 1.647×) | quoted | §31 via `prototype/attn_percost/A2-census.md` §2 C6 |
| attention share of wave | 30.37 % | quoted | §47 |
| required family move for the wave line | 18.6 % → 8 wf/CU | derived | §4.3 |
| shipped note block | 178 VGPR / 0 spill / 0 private / 32,768 B | quoted | `prototype/attn_percost/A3-result.md` §0, `prototype/attn_wmma/E1-result.md` §1 |
| E1 (Q un-staged) | 184 VGPR / 0 spill / 32,768 B | quoted | `prototype/attn_wmma/E1-result.md` §1 |
| llama geometry | 256 thr, 8 warps, 256 VGPR, 900 B/thr scratch, 36,096 B dyn LDS, 1.6 blocks/CU issued, 51.65 ms traced | quoted + derived | `prototype/attn_percost/A2-census.md` §1; P1 above |
| llama `v_wmma` sites | 2,592 vs our 17 | quoted | `prototype/attn_percost/A2-census.md` §2, `prototype/attn_percost/A4-CANDIDATES.md` §1b |

## Appendix B — the commands, all of which are compile-only or driver-only, and what a worker may not run

```
# 1. compile-and-read (CPU only, launches nothing) — the same recipe A3 §0 used
bash prototype/attn_wmma/fattn_census_build.sh /tmp/a5g1
python3 prototype/attn_wmma/fattn_residency_read.py \
        --asm /tmp/a5g1/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s \
        --kernel attn_fattn_wave --threads 96  --warps 3      # G1: expect group 20,480 / spill 0
python3 prototype/attn_wmma/fattn_residency_read.py \
        --asm /tmp/a5g0/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s \
        --kernel attn_fattn_wave --threads 224 --warps 7      # G0: expect group 65,536 / spill 0
python3 prototype/attn_wmma/fattn_ship_census.py              # the existing spill gate, unchanged
# 2. driver answer, no kernel launched (A3 §4's named-but-never-run ladder)
bash prototype/attn_wmma/build_occupancy_fattn.sh            # extend with __launch_bounds__(96,N), N=1..8
# 3. canary on the LINKED binary (the stale-binary guard, AGENTS 2026-10-17)
python3 prototype/attn_percost/a4_l1_note_read.py <decode_run> --kernel attn_fattn_wave
# 4. numerics witness (bit-identity, no new band)
bash prototype/attn_wmma/run_fattn_admission.sh              # §1 is the one-pass-row byte-identity leg
# 5. main agent only, one hold, zero subagents, blocking: the paired window + one rocprofv3 pass
#    (SQ_LEVEL_WAVES_FILLING, VGPR_Count, SQ_WAIT_*; dedup per (Dispatch_Id, Counter_Name);
#     pass divisor derived from attn_prep's 48 sites; --ids-file for the 16k/32k legs)
```

A worker must **not** run step 5, must not take a single `ms`, and must not write into
`engine/performance-status.md` or `progress/`. Nothing in this file was measured on a device by its author.
