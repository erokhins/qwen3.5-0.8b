# A10 — the attention geometry rewrite: what it is, what it costs, and what would make it the wrong call

**Status: the funded main line (§54), with R1+R2 source-killed on step 2 — see `A10-RF-LADDER.md`.** Nothing here is
built. §52 of `bench/performance-status.md` pre-gated it (stall census first, prototype must clear **6.5 TFLOP/s**),
§54 made it the main line, and A10 step 2's compile-only register ladder then found that **Q in registers costs 128
VGPR per lane, not 64, so on the atom this engine ships R2 cannot be honoured at all**: the accumulator and Q
together want the whole 256-VGPR budget, and clang satisfies that by re-issuing the Q loads — handing back the very
traffic R2 exists to delete (`A10-RF-LADDER.md` §2, §6).
The only shape that fits is llama's, which needs the f16-accumulating atom — a numerics-format decision, not a
queue item. This file exists so the arithmetic behind that decision can be read, questioned and attacked instead
of being re-derived from memory each time.

Evidence it rests on: §51 (long-context phase attribution, warm convention) and §53 (L-c measured DISPLACED) of
`bench/performance-status.md`; `A9-LONGCTX-CENSUS.md` (traffic, waste, levers, verdict); `A3-result.md` (the LDS
floor); `A4-L1-verdict.md` (instruction cuts convert at ~0.217); `A6-G1-impl.md` / §48 (residency, shipped);
`A9-LC-verdict.md` (the last targeted lever spent). Source claims are cited `path:line` and were read, not recalled.

---

## 1. Glossary — this machine's real numbers, not generic GPU text

Read from KFD (`/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties`: `simd_count 80`, `simd_per_cu 2`,
`cu_per_simd_array 10`, `lds_size_in_kb 64`).

| term | what it is on this part (gfx1151) |
|---|---|
| **CU** (compute unit; NVIDIA says SM) | **40** = 80 SIMDs ÷ 2 per CU. Everything parallel is placed on one of 40. |
| **SIMD** | **2 per CU**. A CU issues **2 instructions per cycle**, each from a different wave. This is why "2 SIMDs per CU" sets the ceiling on per-cycle issue. |
| **lane** | one 32-bit execution lane. A register's cost is per lane. |
| **wavefront** (AMD) = **warp** (NVIDIA) | **32 threads locked in lockstep**: one instruction fetch, one issue, 32 data elements. The unit the scheduler actually schedules. A kernel's "threads per block" is therefore really "waves per block × 32". |
| **block** (AMD: workgroup) | a group of threads that shares its slice of LDS and can barrier among itself; placed on a CU, many coexist per CU. |
| **LDS** (local data share) | a fast scratchpad **per CU**, **64 KiB total**, divided among the blocks resident on it. Slower than registers, far faster than DRAM. Most of this file is about what fits in it. |
| **register file** | **65,536 lane-VGPRs per CU**, divided among resident waves. A6's ladder brackets it: 3 blocks × 96 threads × 185 VGPR = 53,280 fits, 4 blocks would be 71,040 and does not. |
| **wf/CU** (waves per CU) | **how many wavefronts are resident on a CU at once** — the latency-hiding budget. A wave waiting on a load is not ready; the CU runs another resident wave instead. More resident waves = more chance one is ready. Bound by whichever of LDS / registers / waves-per-CU (32 = 16 per SIMD × 2) / threads-per-CU binds first. |
| **wave16** | this architecture can run 16-lane waves, which is why 16-wide tiles are natural here. Our shipping block uses 32-lane waves (`FATTN_THREADS 96` = 3 waves exactly, asserted at `engine/attn_fattn_kernels.h:265`). |

⚠️ **Two counters lie on this SKU and one more is fake.** `hipGetDeviceProperties` reports 20 CUs and
`maxThreadsPerMultiProcessor = 2048` (= 64 waves/CU against an architectural 32) — a halved SM count. And clang's
`-Rpass-analysis=kernel-resource-usage` prints `Occupancy [waves/SIMD]: 16` as a **constant**, for a 38,944 B-LDS
kernel and a 0 B one alike. So every occupancy claim here comes from a probe (`tools/kernel_occupancy_probe.cpp`,
KFD, `rocminfo`) or from a launch-bounds ladder — never from a query. This is a recorded trap, not a theoretical worry.

### "Row" is overloaded three ways in this repo — the disambiguation

| term | means | where |
|---|---|---|
| **wave row / "lane"** | **one query position (token) of the prompt.** "4,096 lanes" = a 4,096-token prompt. Prefill cuts the prompt into **bodies of `kMaxWaveRows = 512`** (`engine/decode_step.h:51`), so 4k = 8 sequential bodies and 32k = 64. Every headline in this ledger is on this axis. | ledger, `decode_step.h` |
| **attention query row** | a token again, but specifically **one row of the causal score matrix**: row *r* attends KV positions 0..*r*. Our block owns **48** = 3 warps × `FATTN_MROWS 16` (`attn_fattn_kernels.h:187`, `:267`). | `attn_fattn_kernels.h` |
| **"KV lane"** (`FATTN_MLANES 16`) | **columns, not rows** — KV positions staged in one round (`:188`). | `attn_fattn_kernels.h` |

"48 rows × 16 lanes per round" thus means: a block holds 48 query positions and, each round, brings a window of 16
KV positions into LDS, multiplies, updates softmax, advances. A query row near the end of a 32k prompt streams
32,768 KV positions = **2,048 rounds** of 16.

---

## 2. The operation and its shapes

Per attention layer: `Q(W×8×256) · Kᵀ → scores(8, W×W, causal) → softmax → ·V → O(W×8×256)`, with **8 query heads,
2 KV heads (GQA ratio 4), head_dim 256, 6 attention layers** (`engine/state.h:67`, `models/Qwen3.5-0.8B/config.json`).
The work is quadratic: **MACs/pass = 12288·W(W+1)**, i.e. **26.389 TFLOP per 32,768-lane pass**. At 512 lanes the
phase is noise; at 32,768 it is **70.9 % of our wave** (§51).

Measured rates (traces, pass counts *derived* not typed; llama's traced ms are upper bounds, so every ratio is
conservative — and its launch density is 1.91/token against our 0.77):

| | ours | llama.cpp | WMMA roof |
|---|---|---|---|
| 4,096 lanes | 4.09 TFLOP/s (7.8 %) | 7.88 (15.0 %) | 52.63 TFLOP/s (`prototype/issue_rate_probe/results.md` §1) |
| 16,384 | 4.34 (8.3 %) | 8.59 (16.3 %) | |
| 32,768 | **4.47 (8.5 %)** | **8.53 (16.2 %)** | |

---

## 3. The two geometries, side by side

| | **ours — `attn_fattn_wave_u8`** (shipped, `--attn-fattn-qstage unstaged8`) | **llama.cpp — `fattn_mix_raw`** (ROCm) |
|---|---|---|
| threads / block | **96** = 3 waves (`:235`) | **256** = 8 waves |
| query rows / block | **48** = 3 × 16 (`:187`, `:267`) | **64** = `ncols1 16 × ncols2 4` (`template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu:10`) |
| query rows / warp | **16** (forced by the atom) | **16 owned, 8 distinct** — 4 query groups of 16, each co-owned by 2 KV-split warps (`np = 2`, `fattn-mma-f16.cuh:609`, `:1229`); settled below |
| KV positions per stage/round | **16** (`FATTN_MLANES`, `:188`) | **64** (`nbatch_fa`, `fattn-mma-f16.cuh:13`, `:164`) |
| stage bytes | K/V aliased: 16×512 B ×2 = **8,192 B**; block LDS **20,480 B** (`:280`, `:287-291`) | **33,792 B resident, K and V aliased** (row stride `nbatch_K2+4 = 132` half2 → 64×132×4; `:1251-1252`, sized at `:2033`). 67,584 B is the *demanded* K+V per round, not the stage. Whole dynamic LDS demand **36,096 B** (`:2041-2043`) |
| **Q operand** | **re-read from global every round** (`:398-410`, called from the score leg `:815`) | **`Q_in_reg = true`** — registers for the whole work unit (`fattn-mma-f16.cuh:18`, `:665`); its shared tile is aliased, not added (`:2041-2043`) |
| softmax rescale | every **16** KV positions → 428,251 `v_mul` slots/token at 32k | every **64** (`nbatch_fa`) |
| registers | **185 VGPR, spill 0** | **256 VGPR + 900 B/thread scratch**, traced (`A2-census.md:78`); the "358" this row carried is a mis-cite of that line, which reads 256, and A2 §6 already warns a traced `VGPR_Count` is not an object `.vgpr_count` |
| blocks/CU → **wf/CU** | **3 → 9** | **1 → 8** |
| per-round MACs per LSU byte | **9.6** | **≈31** |
| achieved at 32k | 4.47 TFLOP/s | 8.53 TFLOP/s |

Per-round arithmetic behind the last two rows, from A9 §5.1:

* **ours**: 48 rows × 16 lanes × 512 MAC = 393,216 MAC, against 16,384 B staged K/V **+ 24,576 B of Q re-read**
  → 393,216 / 40,960 = **9.6 MAC per byte**.
* **llama**: 64 rows × 64 lanes × 512 MAC = 2,097,152 MAC against 67,584 B of K/V and Q fetched **once ever**
  → 2,097,152 / 67,584 = **≈31 MAC per byte**.

**3.2× the matrix work per byte that crosses the LSU. That single number is the 1.91×.**

### What is established about llama's shape (read from the vendored source, 2026-10-19)

**Verified.** For our head size the RDNA config case is `DKQ=256, DV=256, ncols=64` → **`nthreads 256, occupancy 2,
`nbatch_fa 64, nbatch_K2 128, nbatch_V2 128, nbatch_combine 64, nstages_target 1, Q_in_reg true`**
(`engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:164`, selected by `amd_wmma_available(cc)` at `:245`).
`Q_in_reg` and `nbatch_fa = 64` are therefore facts, and `nstages_target = 1` means **their staging is synchronous —
there is no double-buffering to copy**. The block is 256 threads and its own LDS demand leaves **1 block/CU**
(§5's 8 wf/CU figure).

**SETTLED, 2026-10-19, by a source read of the vendored tree (`engines/llama.cpp` @ `797de63` — the same commit §21
built and traced): the warps split along KV, in pairs, sharing one query set. The branch this question was open on,
however, comes out the expensive way: Q-in-registers is per warp, so R2 still costs 64 VGPR per lane.** The whole
mapping hangs on one constant, `np` = how many warps share one query column.

| quantity | value for our shape (`DKQ = DV = 256`, `ncols = 64`) | where |
|---|---|---|
| block threads / warps | 256 / 8 | config row `:164`; `nwarps = nthreads / warp_size` `:1879` |
| `cols_per_warp` = `T_B_KQ::I` | 16 | `:607`, `:1227`; type `tile<16, 8, half2>` `:1120`, host mirror `:340` |
| **`np`** | `nwarps × cols_per_warp / ncols` = 8×16/64 = **2** | `:609`, `:1229` |
| KV axis | `threadIdx.y % np` picks a 16-row slice of the 64-position stage; the KQ loop steps `np × T_A_KQ::I` = 32, and the same `%np` indexes the VKQ reduction | `:667-668`, softmax `:744`, `:1025` |
| query axis | `threadIdx.y / np` ∈ [0,4) picks which 16-row query group — **4 query sets × 2 warps = 8 waves, 64 distinct rows** | `:691`, `:748`, `:814`, `:1322` |
| what one "row" is | `j = jc/ncols2` is a sequence position, `c = jc%ncols2` a GQA head of **one** KV group | `:1297-1300` |
| the two partials | combined through the shared tile once per work unit, not per round | `:1503` (`np > 1` combine), `:1588-1610` |

Four things follow, and three of them reprice the route.

1. **A corrected-R1 block is not llama's block.** Llama covers 64 *distinct* query rows with 8 waves — 8 distinct rows
   per wave, bought by KV-splitting with 2× redundancy — while corrected R1 (8 waves × 16 **distinct** rows = 128
   rows/block) has no KV split and doubles the rows served per staged tile. Per-round arithmetic on that shape, Q
   resident and a 64-position aliased K+V stage: 128 × 64 × 512 = 4,194,304 MAC against 64 positions × 2 tensors ×
   512 B = 65,536 B demanded → **64 MAC per LSU byte**, against llama's ≈31 and our 9.6. So the 6.5 TFLOP/s abort line
   asks for about half of what this shape models, and it is not a parity bar with llama but a bar well below one. The
   cost side, to be priced at the gate rather than waved off: 128 rows of one KV group is 32 positions × 4 heads, so
   the causal-union overhang (A9 §2.2, 4.01 % today) grows with positions per block, and a 32-position row-set wastes
   more of a 64-position stage's tail than llama's 16.
2. **Q-in-registers is per warp and duplicated across the KV pair, not shared per block — and it costs 128 VGPR
   per lane, not the 64 this brief first priced it at.** `j0 = (threadIdx.y/np) × cols_per_warp`, then
   `Q_B[k] = load_ldmatrix(tile_Q + j0·stride + k)` across all `DKQ/2` (`:1321-1327`): **16 operand tuples** of 16
   query rows × 16 dims each. On this part one 16×16 F16 operand tuple is **8 VGPR per lane**, read three ways in
   `A10-RF-LADDER.md` §1 — the `.vgpr_count` slope of a hold probe (12 → 36 → 68 for 1 → 4 → 8 fragments), the
   shipping kernel's own `v_wmma_f32_16x16x16_f16 v[129:136], v[159:166], v[167:174], v[129:136]` operand spans,
   and the engine's declared `fattn_ARegs` (`engine/attn_fattn_kernels.h:184`). 16 × 8 = **128**. The packing
   argument that gives 64 (128 halves per lane, two per VGPR) fails because the tuple is per 16 dims of K, not per
   32. The two warps of a pair load the same 16 rows, so the cheap branch this question could have landed on does
   not exist — and **this is the number that kills R1+R2 as funded**, item 3.
3. **The accumulator is a format choice, not an atom limit — and with item 2 it decides whether the route is
   buildable at all.** §4 point 2 says a 16-row × 256-dim P·V accumulator costs 128 f32 per lane and that M
   therefore cannot grow. That is a property of *our* atom (`v_wmma_f32_16x16x16_f16`), not of the machine: llama's
   RDNA3 path declares `T_C_VKQ = tile<16,16,half2>` (`:1125`) and its mma is
   `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` issued twice with `opsel` 0 and 1 (`mma.cuh:1556-1566`) — an
   **f16-accumulating** WMMA covering 16 queries × 32 dims per 8-VGPR tile, 8 tiles = **64 VGPR/lane**, rescaled in
   `half2` (`:1470-1478`) and held in f16 across the whole KV sweep. Put the two together against the ceiling,
   **256 VGPR/thread** (65,536 lane-VGPRs per CU ÷ 256 threads at 1 block/CU — and RDNA's own per-thread maximum,
   which no code object in this tree exceeds):

   | one wave, mid-round | f32 atom (ours) | f16 atom (llama's) |
   |---|---|---|
   | P·V accumulator, 16 rows × 256 dims | 128 | 64 |
   | Q resident (item 2) | 128 | 128 |
   | a 64-position round's scores | 32 | 32 |
   | P operands, staging temps, addressing | ≥ 20 | ≥ 20 |
   | **against 256, naive simultaneous sum** | **≥ 308 — impossible** | **≈ 244 — 12 to spare** |
   | **against 256, measured in isolation (`A10-RF-LADDER.md` §6)** | **219 VGPR, 0 spill** | **155 VGPR, 0 spill** |

   **R2 cannot be honoured on the atom this engine ships.** The naive sum is 308 against a 256 ceiling, and clang's
   response is not to overflow but to decline: it re-issues the Q loads rather than keep 16 tuples live, which is
   precisely the traffic R2 was funded to delete (`A10-RF-LADDER.md` §6 reading 2). That is a source kill in the
   sense that matters — not "the compiler errors", but "the compiler silently gives back the thing you went to
   remove". On the f16 atom the same probe measures **155 VGPR at 0 spill**, so the shape fits with ~100 VGPR of
   room instead of none, and llama spends that room deliberately (`A2-census.md:78`: 256 VGPR, 900 B/thread scratch,
   1.9× our rate regardless) — which moves point 6's "what is on the critical path" from a side note to the open
   question. Two catches
   come with the f16 atom and neither is arithmetic: carrying the accumulator in f16 across 32,768 positions is a
   *format* change on the shipping path, admitted by the standing numeric-format rule **only if the gates stay
   green** and never by assumption; and `prototype/issue_rate_probe/results.md` §1 measured **only**
   `v_wmma_f32_16x16x16_f16` at 52.63 TFLOP/s, so the f16 form's rate on gfx1151 is unknown and may be worse. The
   cheap probe that settles the rate is named in `A10-RF-LADDER.md` §4.
4. **A 64-position stage is a ~34 KiB stage, not a 64 KiB one, so R3's premise is half wrong.** With `nstages = 0` on
   this part (`:366` + `:372`; `cp_async_available` is false, which confirms R3's synchronous-staging assumption) the
   pointers collapse: `tile_K = Q_in_reg ? tile_Q : …` and `tile_V = nstages > 1 ? … : tile_K` (`:1251-1252`), so Q, K
   and V share one buffer and V is re-loaded into it after K is consumed. The declared demand is at `:2033-2043`: KV
   stage 64 × 132 × 4 = **33,792 B**; mask `ncols1 × (nbatch_fa/2 + 4) × 4` = 2,304 B; combine
   `nwarps·cols_per_warp × (nbatch_combine+4) × 4` = 34,816 B; total `max(combine, max(Q, KV+mask))` = **36,096 B**.
   R3 said "LDS is what R1 gives up"; what R1 gives up is 20,480 B and what R3 spends is ~34 KiB, leaving ~29 KiB of
   the 64 KiB unspent rather than exhausted. It does not buy a second block/CU (2 × 36,096 > 65,536), so 8 wf/CU
   stands — but after R1 the binding resource is the **register file**, not LDS.

**R5 is not an idea to import, it is llama's arrangement**: `j = jc/ncols2`, `c = jc%ncols2`, `zt_gqa·ncols2 + c`
bounded by `gqa_ratio` (`:1297-1300`) makes a block's 64 columns 16 positions × 4 heads of **one** KV group, so one
staged tile already serves 4 query heads; R1's shape keeps that at 32 positions × 4 heads = 128 rows.

Two provenance defects surfaced by this read, both in this file rather than in the source: the registers row above
cited "358 VGPR" against `A2-census.md:78`, which reads 256; and A2 recorded this config's `occupancy` as 1 where
`:164` spells 2 (consumed as `__launch_bounds__(nthreads, occupancy)` at `:1797`). The second is not cosmetic — 256
traced VGPR is what a `launch_bounds(256, 1)` budget produces, and whether clang honours a min-blocks argument on
gfx1151 at all is precisely what the step-2 compile ladder exists to answer. What the read does **not** change: the
atom stays 16×16×16, staging stays synchronous, the 6.5 TFLOP/s abort line and the short-context non-regression lines
stand, and the two settled findings (residency is not the gap; traffic cuts do not convert) are untouched.

The traffic table behind it (A9 §3, derived from these constants):

| | ours | llama |
|---|---|---|
| K/V demanded per causal touch | 23.23 B | 16.53 B |
| **Q demanded per causal touch** | **34.85 B** (898.11 GB/pass at 32k = `DH×2 / lanes_per_round` = 512/16) | 0.03 B |
| total demanded per touch | **58.08 B** | ≈16.6 B |
| implied demand at 32k | **253 GB/s — above the 237 GB/s DRAM roof**, so *provably* cache-served (32 KB L1/CU, 2 MiB L2) | 138 GB/s |

Which is why the Q re-read is described as *"free in bytes, expensive in queue slots"*: the LSU and VMEM issue
slots spent fetching it are real, and no wavefront count pays them back.

> **MEASURED REFINEMENT, same day — §54 of `bench/performance-status.md`.** "Provably cache-served" is right; "from
> L1" was not checked and is only partly true. Counters on the shipped kernel at 32k give **TCP_REQ 3.17e10 with a
> 26.8 % L1 miss rate** — L1 absorbs ~73 % of its traffic, not all of it — and Q's own footprint is **24,576 B per
> block × 3 blocks/CU = 73,728 B, above the 32 KB L1**, so a meaningful share of the re-read comes from the deeper
> tiers. Those tiers are now measured too: **876.7 GB/s at 32 MiB, 239.4 GB/s at 64 MiB**
> (`devices/strix-halo-memory-bandwidth.md`, "The size–bandwidth curve, measured"), so the load that G1 pushed out
> of LDS lands in a ~32 MiB tier running ~3.7× DRAM speed — **G1's −22 % has an unstated dependency on that tier**,
> and a change that streams more K/V through it could regress G1 without touching a line of Q code.

---

## 4. Why our geometry is what it is — the constraint chain

Each step was a correct local decision. Each one forecloses the next. This is the part that explains why nobody
simply "used bigger tiles".

1. **The only tensor atom on this part is `v_wmma_f32_16x16x16_f16`.** There is no MFMA here, so M and N are
   quantised to 16: `static_assert(FATTN_MROWS == 16 && FATTN_MLANES == 16, "the WMMA tile is 16x16x16…")`
   (`:192`), and one C tile is 8 f32 per lane (`fattn_CRegs`, `sizeof == 32`, `:196`).
2. **A 16-row × 256-dim P·V accumulator costs each lane 128 f32** (16×256/32). That is **128 of our 185 VGPR**.
   M cannot grow: at 32 rows the accumulator alone wants 256 registers/lane. **(Repriced by the §3 source read, not
   retracted: 128 f32 is the cost of *our* f32-accumulating atom. Llama's RDNA3 path packs the same accumulator as
   f16 halves — 64 VGPR/lane — see "SETTLED" item 3 of §3, including the two catches that come with it.)**
3. **Q-in-LDS and residency are mutually exclusive on this part.** Full Q staging needs 3 warps × 8,192 B × 2
   halves + K/V 8,192 B ≈ **57 KB/block** → 1 block/CU. A3 established that **20,480 B is the LDS floor of the
   shipped arithmetic** — `attn_fattn_kernels.h:581` consumes all 16 rows × 256 dims every round and `:710`/`:216`
   put the epilogue transpose on the same per-warp buffer — and priced both rescue routes and rejected them
   (+379 KB/token/layer of K/V staging, or +150 % per-round load demand).
4. **So G1 un-staged Q** (§48, **shipped, −22.09 % family, byte-identical**): Q comes from global once per round.
   Its price is the 898 GB/pass Q re-read and its purchase is 6 → 9 wf/CU. Seen from the traffic side, G1 traded
   operand reuse for residency — the right trade at the time, and the reason the traffic asymmetry is now ours.
5. **A 16-lane round means the softmax running max and rescale fire every 16 positions**: 428,251 `v_mul`
   slots/token at 32k.
6. **Register-resident Q was already tried here and rejected on measurement**, not theory: 4 warps with Q in
   registers gave `.vgpr_count 264 / .vgpr_spill_count 115 / .private_segment_fixed_size 724` — it spilled harder
   than the staging it replaced, and scratch traffic lands on the same dependent chain §28 identified
   (`attn_fattn_kernels.h:55-59`). The same comment notes llama's 358 VGPR is "not a number to admire from a
   distance": **they spill too.**

Point 6 is the one most often misread, and correcting it changed what this brief funds. **Rows per wave is 16 on
both sides** — the atom forces it here (`attn_fattn_kernels.h:192`), and llama's own config for our shape takes it
through 4 warps × 16 rows rather than 8 warps × 8. So the difference is **not** a smaller accumulator per lane.
What differs is what llama *does with the rest of the machine*: a **64-position KV stage** instead of 16, Q held in
registers for the whole sweep, the softmax rescale batched over those 64 positions — and a **willingness to spill**:
their kernel reports 358 VGPR *with* spill and is still 1.9× faster, which means scratch traffic on their side sits
somewhere harmless, whereas our rejected attempt (A above) spilled into the same dependent chain §28 identified.
R1 is therefore not "cut the accumulator", it is **buy 256 VGPR/thread by making the block one 8-wave block per CU**,
and that is a compile question before it is anything else.

---

## 5. What has already been falsified as "the cause"

| axis | test | result |
|---|---|---|
| LDS capacity | A3: any route to 20,480 B | **dead at the source** — 20,480 B is the floor; both rescues priced and rejected |
| accumulator dependency | A4-L1: two accumulator chains, no spill | **−1.67 % family** — real, inside the paired floor |
| residency | G0 falsifier (−10.82 % family) → **G1 shipped −22.09 %** | **harvested** — 9 wf/CU today |
| bandwidth | L-b: 16→32 KV lanes/round halves rounds | **dead on declared LDS**: 9→6 wf/CU = **+28 % family** by §48's own elasticity |
| instructions per round | A4-L1 again; L-d batched rescale | converts at **~0.217** of budget; L-d is −1.26 %, bundle-only |
| traffic (modelled demand) | D21 / §20.7: a 2× traffic cut moved the wave **0.7 %** | demanded ≠ DRAM bytes |
| traffic (traced rate) | **L-c: remove 77 % of a 105 GB/pass round-trip** | **§53: −4.61 % family, −2.38 % wave → DISPLACED**; rate×bytes overstates by **~1.35×** |
| launch count / splits | §38/§41 | refuted, already spent |

**Every axis except work-per-operand-byte is closed from its own direction.** That residual — 9.6 vs 31 MAC/byte —
is the geometry, and the targeted-lever space was bounded at **1.494× best-case stack** before L-c was spent
(A9 §5), which §53 then confirmed with the cheapest of those levers.

---

## 6. The rewrite, itemised

Not "tune the kernel": adopt the **deep-stage geometry as a second form**, routed by width. Each item lists its
dependency, because order is the whole difficulty.

| # | change | why it is required | cost / constraint | depends on |
|---|---|---|---|---|
| **R1** | **8 waves per block (256 threads), 16 query rows per wave → 128 query rows per block** | cuts the LDS cost per served row by 8/3 with nothing else changed, and at 1 block/CU of 256 threads the register file hands out **256 VGPR per thread**, which is what makes R2
   possible (**measured correction: 256/thread is also RDNA's own per-thread ceiling, and the shipped body sits at
   185 — so R1 buys +29 VGPR/lane, not +128; `A10-RF-LADDER.md` §0, §3**) | LDS ~64 KiB → **1 block/CU → 8 wf/CU** (the residency §48 bought is deliberately given up here); **rows per wave cannot go below 16** — see the note below | — |
| **R2** | **Q in registers for the whole KV sweep** | deletes the 898 GB/pass re-read ≈ 60 % of our demanded bytes, and the VMEM/LSU slots behind it | only fits **after** R1: 16 rows/warp costs 128 accumulator VGPR and Q **128** more (corrected: not ~64 —
   `A10-RF-LADDER.md` §1), which needs the 256/thread that only 1 block/CU of 256 threads provides — and even then
   **cannot be honoured on the shipped f32 atom**, because clang re-issues the Q loads instead of overflowing
   (`A10-RF-LADDER.md` §6). It fits only on the f16-accumulating atom of llama's RDNA path, at 155 VGPR measured. Locally rejected before (variant A: spill 115) *because* it was attempted at 3 waves/block, where the budget is ~217/thread. Llama pays spill here too (256 VGPR + 900 B/thread scratch, traced) and wins anyway — so the question is *what* the spilled bytes are on the critical path, not whether spilling is allowed. **Repriced:** the 128 is our f32 atom; an f16-accumulating one prices that accumulator at 64 (§3 SETTLED item 3) | R1 |
| **R3** | **KV stage 16 → 64 positions**, V kept `[dim][lane]` as today (`fattn_v_addr`) | cuts rounds/token ~4× (1,115 → ~280 at 32k) and raises MAC/byte toward 31 | LDS is what R1 gives up; **this is the reverse of L-b**, which is dead *because* it tried deep stages at 3 blocks/CU. **Not double-buffered** — llama's RDNA case asks for `nstages_target = 1` (`fattn-mma-f16.cuh:164`), so staging stays synchronous and no async copy machinery is needed (there is no `cp.async` equivalent here anyway). **Repriced:** llama aliases K and V in one 33,792 B buffer (total dynamic LDS 36,096 B), so a 64-position stage is a ~34 KiB stage, not a 65 KiB one — §3 SETTLED item 4 | R1 |
| **R4** | **softmax rescale batched over 64 positions** (their `nbatch_fa`) | cuts the 428k-slot rescale/max chain ~4× | falls out of R3; L-d alone is −1.26 %, worth taking in context only | R3 |
| **R5** | **GQA packing in the row index**: 64 rows = 16 positions × 4 heads of one KV group, so one staged tile serves 4 query heads | today each head column re-stages its group's K/V (up to 4×) | an **index** change in the launcher + mask logic, **not** a layout change; no producer touched | R1, R3 |
| **R6** | **epilogue / O transpose in 8-row groups** | matches R1's per-warp row count | **mostly already built** — G1's `FATTN_TRG 8` machinery (`:242-264`) | R1 |
| **R7** | **route by width**: keep the current form for short prompts, deep-stage for long | our current shape is *why* we beat llama warm at short context (10,637 vs 7,378 tok/s at 512 lanes); a 64-row tile pads a 64-token prompt to a full tile and 1 block/CU idles the machine at low batch | a knob + a routing threshold + a gate case **at** the threshold | R1–R6 |
| **R8** | numerics adjudication | a 64-lane stage re-associates the softmax merge → **not bit-identical** | judged by existing bands (`CASCADE_BAND`, §10.1 of `specification/10-reference-dumps-and-gates.md`); standing rule: a red gate asks whether the band can widen **without losing a must-catch**, never a silent loosening | all |

Explicitly **not** part of it: changing the tensor atom (there is no wider atom), and producer-side K/V swizzling —
see §8.

> **CORRECTION TO R1, made before any work was spent on it.** The first version of this table said "8 query rows per
> warp, 64 rows per block". **That is impossible on this part**: the tensor atom is 16×16×16 *per wave*, and the
> repo already asserts it — `static_assert(FATTN_MROWS == 16 && FATTN_MLANES == 16, "the WMMA tile is 16x16x16: MROWS
> below 16 cannot build an A operand…")` (`engine/attn_fattn_kernels.h:192`). A wave cannot own 8 query rows and
> still build an A operand. The corrected shape keeps **16 rows per wave** and raises the **wave count to 8**,
> i.e. 128 query rows per block, which preserves the accumulator cost per lane (still 128 f32) instead of halving
> it — and that is also why **R2 is priced harder than the first version implied**: at 256 threads and 1 block/CU,
> 256 VGPR/thread has to hold 128 accumulator + Q (~64 f16-packed) + scores + addressing, which is tight but not
> obviously impossible, and it is exactly the register picture llama's *358-with-spill* kernel is sitting in. R1 is
> therefore a **compile question first**, and it is answered by a compile-only ladder (A6's method: build the
> variant, read `.vgpr_count` / `.vgpr_spill_count` / `group_segment_fixed_size`, no kernel launched) before any
> device window is asked for.

---

## 7. Plumbing that must move in the same commit

This is what "not just the kernel" actually means — plumbing, not redesign, but each item has already caused a real
bug when it was left behind:

1. **The launcher mirror.** `FATTN_ROWS_PER_BLOCK` has a second spelling in
   `prototype/attn_decode/attn_decode_kernels.h`. This class of bug bit twice already: two self-consistent layouts
   joined by a branch were invisible to both sides, and half of every streaming row came out as the other KV
   group's answer (§20.1 of `prototype/attn_decode/results.md`). One shared address expression.
2. **Launch accounting and its canaries.** `fattn_rtiles_of` counts blocks of 48 rows; `LaunchCounts::describe()` /
   `wave_plan_bill` are compiled into `decode_run` and are what A/B harnesses use as the behavioural canary. Row
   counts change → every expected number must be **re-derived, never retyped**. Note also that a counter pass
   derives its pass count from `attn_prep` per-body site counts.
3. **The gate's *derived* case lists.** `engine/gate.py::lane_ceiling()` reads `MAXSC × kAttnMaxSplit` from source
   and is unaffected, but the wide anchors, the synthetic tile-boundary wave (rows of exactly 512/513/514/515
   lanes) and the must-not-catch blind sets are all derived from the current row geometry. The repo's own rule:
   must-catch/must-not-catch sets must be **derived from the geometry**, because a hand-listed set rots into
   "everything" — and a control derived from 48-row geometry cannot see a 64-row bug. `partalias` scored 0/1,024
   for exactly this reason before the case list was derived.
4. **The routing knob trio.** Default moved in `engine/decode_run.cpp`, `engine/decode_step.h`, `engine/gate.py` in
   **one commit** (the `--attn-tile` trap: otherwise the gate silently judges a different binary than the engine
   ships), plus a gate case **at** the routing width, because "a route no fixture reaches has no tests, including
   tests of the gate".
5. **Byte-identity/inertness witness for the untouched default**, as L-c did: the default must be byte-identical to
   the pre-change binary, witnessed on device, not argued.

---

## 8. What does **not** change — the "does the rest of prefill move?" question

**No.** The interface attention exposes does not change, and internal tiling is invisible across it:

* the wave/body loop (`prefill_wave`, 512-row bodies) — a geometry change is *inside* one phase of a body;
* the tensors: attention consumes `q_rope`/`k_rope`/V + the gate vector and writes `O`, which feeds the `o_proj`
  GEMM. The rope/glue producers and GEMMs keep their contract. **GDN (45 of 48 layers), GEMMs, glue, KV append:
  untouched interfaces.**
* **KV cache layout** (2 KV heads × 256 dims/token) — R5 is an index change, not a layout change;
* split partial sizing `rows × N_Q × kAttnMaxSplit × (DH+2)` — keyed to split, not rows-per-block;
* the **decode path** (different kernels entirely);
* numerics **bands** — no new band, per the standing numeric-format rule.

The one variant that **would** reach outside attention, and which I would not fund: having rope/glue write K/V
**pre-swizzled** so the deep stage is a straight copy. That changes the glue writer and a layout every 32k fixture
exercises, for a saving the census cannot price; V is already staged transposed.

Because the wave is a serial stream of launches, attention running 1 block/CU does not starve GEMM/GDN (different
launches) — but the §53 result is a caution here: we could not attribute a ~70 ms neighbour move at 32k because
`gdn_rec`'s own rep spread is ±8–12 %. Cross-family attribution at 32k needs paired per-rep neighbour measurement
with enough reps to resolve 3 %, not a printed externality sum.

---

## 9. Scope, sequence, and the abort line

* **Size: ~1.1k lines** (round body + launcher/mirror/gate plumbing) and **2–3 weeks** at this repo's cadence: one
  pre-registered BET, paired windows inside one lock hold, falsifiers, negative controls derived from geometry, at
  512 / 4k / 16k / 32k, plus a routing decision. The measurement discipline, not the C++, is the schedule.
* **Pre-gate 1 — the stall census: RUN, outcome "not falsified" (§54).** An `SQ_WAIT*` counter pass on the shipped
  kernel at 32k, with the pass-count discipline (dedup by `(Dispatch_Id, Counter_Name)`, derive the pass count from a
  kernel whose per-site count is known). It asks one question: is the binding stall the LSU/VMEM queue that R2/R3
  relieve, or something they would sit next to? **Measured: 72.1 % of the shipping kernel's waiting is none of
  LDS-instruction (13.9 %), barrier (11.2 %) or ifetch (0.4 %)** — consistent with memory-return waiting, and that is
  the strongest statement the data supports, because `DRAM_REQ_PER_CLIGRP` and `TCP_BACK_PRESSURE` both read exactly
  0.0 (vacuous) and counter semantics on this part are unverified. It does not point away from the operand-reuse
  explanation, and it cannot confirm it.
* **Pre-gate 2 — the abort line, written before the build: a prototype must clear 6.5 TFLOP/s at 32k**
  (≈ +45 % over today's 4.47) to earn a second week. A number ends it, not fatigue.
* **Step 2 outcome, compile-only, before any build or device window (`A10-RF-LADDER.md`):** the funded geometry —
  R1 + R2 with the atom this engine ships — **cannot be honoured**: 128 accumulator + 128 Q + 32 scores + addressing
  is ≥ 308 VGPR against a 256 ceiling, and clang's response is to re-issue the Q loads, which is the traffic R2
  exists to delete. The one shape that fits is llama's, on the f16-accumulating atom (**155 VGPR measured in
  isolation, 0 spill**), which makes **R8 a precondition rather than a downstream item** and leaves that atom's rate
  on gfx1151 unmeasured. So the next number owed is not a 32k run: it is `v_wmma_f16_16x16x16_f16`'s issue rate, and
  the legs for it are **already written** in `prototype/issue_rate_probe/wmma_latency.cpp` (`wl_f16_i{1,2,4,8}`) with
  no binary on disk — one build and one short hold, because a slow f16 atom retires the whole route without spending
  a week on it.
* **Success criterion, stated in advance:** parity is attention phase 5,908 → 3,093 ms at 32k (**−47.6 %**), i.e.
  the wave's 1.43× gap at 32k closes to ~1.0×; the long-context warm figures this shows up in today are 6,125
  tok/s at 16k and 3,924 at 32k against llama's 7,800 / 5,475 (§50/§51).
* **Non-regression line:** warm 512-lane throughput (10,637 tok/s) and 4,096 (9,790) must not fall, which is what
  R7 routing exists to protect. If routing cannot protect them, the rewrite has bought long context with short
  context and that must be reported, not averaged.
* **What we give up if we do nothing:** the long-context gap stays at ~1.4–1.9× depending on width, while short
  context stays ahead. That is a coherent position for a model whose realistic deployment is short-to-medium
  context, and it is a legitimate answer, not a defeat.

## 10. Honest uncertainties

1. **llama is not a ceiling.** At 16 % of the matrix roof they are not near a hardware wall either; matching them
   is a target, not a proof of optimality.
2. **The mechanism is inferred, not measured.** "9.6 vs 31 MAC per LSU byte" is an arithmetic property of the two
   source forms, consistent with four falsifications, but no counter on this box measures LSU queue pressure
   directly. The `SQ_WAIT` census is the closest available proxy, and it could say something unexpected.
3. **Occupancy above 9 wf/CU is extrapolation.** Every elasticity used for residency is anchored on §48's measured
   6→9 pair; claims past that are labelled extrapolations in A9 §4 and could easily be optimistic.
4. **The rewrite may make the short-context path worse even with routing**, because 1 block/CU changes behaviour
   under low occupancy in ways the pinned fixtures (all one-pass prompts) do not exercise.
5. **Instruction-class ceiling.** The part's WMMA atom is 16×16×16 and RDNA's matrix throughput here is what the
   52.63 TFLOP/s probe measured; a deep-stage rewrite buys operand reuse, not a wider atom.
