# A10 — the attention geometry rewrite: what it is, what it costs, and what would make it the wrong call

**Status: decision brief, not an approved plan.** Nothing here is funded or started. §52 of
`bench/performance-status.md` pre-gates it (stall census first, prototype must clear **6.5 TFLOP/s**), and it is the
user's scope decision. This file exists so the arithmetic behind that decision can be read, questioned and attacked
instead of being re-derived from memory each time.

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
| query rows / warp | **16** | **8** |
| KV positions per stage/round | **16** (`FATTN_MLANES`, `:188`) | **64** (`nbatch_fa`, `fattn-mma-f16.cuh:13`, `:164`) |
| stage bytes | K/V aliased: 16×512 B ×2 = **8,192 B**; block LDS **20,480 B** (`:280`, `:287-291`) | K+V **67,584 B** (row stride `nbatch_K2+4 = 132` half2), LDS ~64 KiB |
| **Q operand** | **re-read from global every round** (`:398-410`, called from the score leg `:815`) | **`Q_in_reg = true`** — registers for the whole work unit (`fattn-mma-f16.cuh:18`, `:665`); its shared tile is aliased, not added (`:2041-2043`) |
| softmax rescale | every **16** KV positions → 428,251 `v_mul` slots/token at 32k | every **64** (`nbatch_fa`) |
| registers | **185 VGPR, spill 0** | 358 VGPR **with spill** (`A2-census.md:78`) |
| blocks/CU → **wf/CU** | **3 → 9** | **1 → 8** |
| per-round MACs per LSU byte | **9.6** | **≈31** |
| achieved at 32k | 4.47 TFLOP/s | 8.53 TFLOP/s |

Per-round arithmetic behind the last two rows, from A9 §5.1:

* **ours**: 48 rows × 16 lanes × 512 MAC = 393,216 MAC, against 16,384 B staged K/V **+ 24,576 B of Q re-read**
  → 393,216 / 40,960 = **9.6 MAC per byte**.
* **llama**: 64 rows × 64 lanes × 512 MAC = 2,097,152 MAC against 67,584 B of K/V and Q fetched **once ever**
  → 2,097,152 / 67,584 = **≈31 MAC per byte**.

**3.2× the matrix work per byte that crosses the LSU. That single number is the 1.91×.**

The traffic table behind it (A9 §3, derived from these constants):

| | ours | llama |
|---|---|---|
| K/V demanded per causal touch | 23.23 B | 16.53 B |
| **Q demanded per causal touch** | **34.85 B** (898.11 GB/pass at 32k = `DH×2 / lanes_per_round` = 512/16) | 0.03 B |
| total demanded per touch | **58.08 B** | ≈16.6 B |
| implied demand at 32k | **253 GB/s — above the 237 GB/s DRAM roof**, so *provably* cache-served (32 KB L1/CU, 2 MiB L2) | 138 GB/s |

Which is why the Q re-read is described as *"free in bytes, expensive in queue slots"*: the data arrives from L1,
but the LSU and VMEM issue slots spent fetching it are real, and no wavefront count pays them back.

---

## 4. Why our geometry is what it is — the constraint chain

Each step was a correct local decision. Each one forecloses the next. This is the part that explains why nobody
simply "used bigger tiles".

1. **The only tensor atom on this part is `v_wmma_f32_16x16x16_f16`.** There is no MFMA here, so M and N are
   quantised to 16: `static_assert(FATTN_MROWS == 16 && FATTN_MLANES == 16, "the WMMA tile is 16x16x16…")`
   (`:192`), and one C tile is 8 f32 per lane (`fattn_CRegs`, `sizeof == 32`, `:196`).
2. **A 16-row × 256-dim P·V accumulator costs each lane 128 f32** (16×256/32). That is **128 of our 185 VGPR**.
   M cannot grow: at 32 rows the accumulator alone wants 256 registers/lane.
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

Point 6 is the one most often misread. llama reaches register-resident Q because it spreads 64 rows over **8 warps
= 8 rows/warp**, so their per-lane accumulator slice is ~64 f32 where ours is 128 — Q fits in the space that is
left, at 1 block/CU where the register file hands out **65,536/256 = 256 VGPR per thread**. Register-resident Q is
therefore a **consequence** of more-warps-fewer-rows-per-warp, not an independent wish that can be adopted first.

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
| **R1** | **256 threads (8 waves) per block, 8 query rows per warp, 64 rows per block** | the only way to cut the per-lane accumulator from 128 f32 to ~64 f32, which is what makes everything after it affordable | LDS ~64 KiB → **1 block/CU → 8 wf/CU** (the residency we bought in §48 is deliberately given up here) | — |
| **R2** | **Q in registers for the whole KV sweep** | deletes the 898 GB/pass re-read ≈ 60 % of our demanded bytes, and the VMEM/LSU slots behind it | only fits **after** R1 (8 rows/warp + 256 VGPR/thread at 1 block/CU). Locally rejected before (A: spill 115) *because* it was attempted at 16 rows/warp | R1 |
| **R3** | **KV stage 16 → 64 positions, double-buffered**, V kept `[dim][lane]` as today (`fattn_v_addr`) | cuts rounds/token ~4× (1,115 → ~280 at 32k) and raises MAC/byte toward 31 | LDS is what R1 gives up; **this is the reverse of L-b**, which is dead *because* it tried deep stages at 3 blocks/CU. Deep stages only make sense at 1 block/CU | R1 |
| **R4** | **softmax rescale batched over 64 positions** (their `nbatch_fa`) | cuts the 428k-slot rescale/max chain ~4× | falls out of R3; L-d alone is −1.26 %, worth taking in context only | R3 |
| **R5** | **GQA packing in the row index**: 64 rows = 16 positions × 4 heads of one KV group, so one staged tile serves 4 query heads | today each head column re-stages its group's K/V (up to 4×) | an **index** change in the launcher + mask logic, **not** a layout change; no producer touched | R1, R3 |
| **R6** | **epilogue / O transpose in 8-row groups** | matches R1's per-warp row count | **mostly already built** — G1's `FATTN_TRG 8` machinery (`:242-264`) | R1 |
| **R7** | **route by width**: keep the current form for short prompts, deep-stage for long | our current shape is *why* we beat llama warm at short context (10,637 vs 7,378 tok/s at 512 lanes); a 64-row tile pads a 64-token prompt to a full tile and 1 block/CU idles the machine at low batch | a knob + a routing threshold + a gate case **at** the threshold | R1–R6 |
| **R8** | numerics adjudication | a 64-lane stage re-associates the softmax merge → **not bit-identical** | judged by existing bands (`CASCADE_BAND`, §10.1 of `specification/10-reference-dumps-and-gates.md`); standing rule: a red gate asks whether the band can widen **without losing a must-catch**, never a silent loosening | all |

Explicitly **not** part of it: changing the tensor atom (there is no wider atom), and producer-side K/V swizzling —
see §8.

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
* **Pre-gate 1 — the stall census (cheap, device, main agent).** An `SQ_WAIT*` counter pass on the shipped kernel at
  32k, with the pass-count discipline (dedup by `(Dispatch_Id, Counter_Name)`, derive the pass count from a kernel
  whose per-site count is known). It asks one question: is the binding stall the LSU/VMEM queue that R2/R3 relieve,
  or something they would sit next to? **This should run before any rewrite work is funded.**
* **Pre-gate 2 — the abort line, written before the build: a prototype must clear 6.5 TFLOP/s at 32k**
  (≈ +45 % over today's 4.47) to earn a second week. A number ends it, not fatigue.
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
