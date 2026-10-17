# D30 item 2.4 — price a WMMA path for `attn`, decide it before anyone writes a hot kernel

## VERDICT: **funded no.**

The single number that refutes it: **at the arithmetic share this repo already measured
(27.9 % of the touch — D30 §2.4's own ceiling line, from D20), and this kernel's own measured
slot→time elasticity (`time ∝ slots^0.39–0.44`, D21/D22), the entire WMMA prize is −9.6 % of
`attn` after paying the feeder cost I measured, against a geometry tax already measured in-kind
at 7–29 % for the same class of re-tile (`prototype/attn_m2/results.md` §2, §8.4).** It does not
clear the 15 % kill band; at the central case it does not clear zero.

No device run was made. No timing was taken. Everything below is source facts, static ISA from
`-O3 --offload-arch=gfx1151` compile-only builds, and already-measured ledger numbers.

---

## 1. Denominator, named before any share is quoted (D32/D33 rule)

Every share below is stated against one of exactly two denominators, and which one is named at
every use:

| denominator | value | source |
|---|---|---|
| **pass-1 touch bill** (one `(row, lane, head)` score) | **45 issue slots** = 28 VALU + 7 LDS + 10 SALU | `prototype/attn_valu_slim/results.md`, reproduced against the engine build by this session (`prototype/attn_wmma/isa_census.engine.output.txt` §3: `attn_slim_wave8CoreArgs … slots/touch=45`, `attn_tile_wave8CoreArgs … 116`) |
| **whole-kernel static bill**, `attn_slim_wave8CoreArgs` | 1565 ops / 1524 slots / 965 VALU / 24 LDS / 10 VMEM / 519 SALU | same census, engine listing |
| **attn's share of the wave** | 257.56 ms of 956.12 ms = **26.94 %** | D33, decision log line 783 |

A whole-kernel *dynamic* per-touch total is **not** established by me (§6, `region_census.py` is
half-finished and its calibration currently fails). The price below therefore does not depend on
it: it uses D20's measured **27.9 % of the touch** arithmetic share, which is the number D30's own
2.4 row already adopted as this route's ceiling.

---

## 2. Geometry: the route cannot be substituted, only re-tiled

| fact | value | path:line |
|---|---|---|
| `ATTN_TILE_M` | **1**, one query row per block | `engine/attn_slim_kernels.h` tile geometry block; `:164` `static_assert` ties `blockDim.x / warpSize == ATTN_TILE_G` |
| `GQA`, `DH` | 4 heads per KV group, 256 dims | `engine/attn_slim_kernels.h:153-164` |
| block threads, warps | 128 threads = 4 warps | `ATTN_TILE_THREADS` |
| one shipped score touch | 32 lanes × 8 dims = **256 MACs**, 45 slots | `attn_slim_dot<8>` / `attn_slim_dot<4>` |
| **WMMA tile** | 16×16×16 = **4096 MACs = 16 touch-equivalents** | `v_wmma_f32_16x16x16_bf16` |

So `M = 16` is not available inside the shipped loop at all. Filling it means either 16 query rows
× 1 head or 4 rows × 4 heads (GQA), and either way the block, grid, lane-slice union, and score
staging all change. That is a **re-tile, not an instruction substitution** — which is the exact
class this repo has already measured once, and measured negative (§5).

**PV accumulator wall.** PV output is `DH = 256` dims. A WMMA D fragment is 8 f32 per lane, so one
warp keeping all 256 dims needs 16 fragments = **128 VGPR/lane**. The shipped kernel is at
**47 VGPR** total. N-splitting across warps is mandatory, which further changes the block shape.

---

## 3. The feeder probe (item 2) — compile-only census, the thing this item was sent to settle

`prototype/attn_wmma/wmma_feeder_probe.cpp` + `build.sh` + `feeder_census.py`. Built with
`hipcc -O3 --offload-arch=gfx1151 --save-temps -c`. **Never launched.** A and B are staged in LDS in
the engine's real layouts (row-major bf16, `ldm = DH = 256` for K/V rows, `ldm = MAXSC = 512` for a
score row), fed through `nvidia::wmma::` bf16 `load_matrix_sync`/`wmma_sync`/`store_matrix_sync`.
Marginal cost is differenced between the 1-tile and 16-tile variants of the *same* kernel, so
prologue/epilogue cancel.

| kernel | tiles | executed instr | WMMA | lane-movement ops | notable |
|---|---|---|---|---|---|
| `wm_score_tile` | 1 | 155 | 1 | 24 | 6 `ds_load_b128`, 6 LDS stores |
| `wm_score_k256` | 16 | 358 | 16 | 144 (136 `ds_swizzle_b32` + 8 `dpp`) | 32 `ds_load_b128` |
| `wm_pv_tile` | 1 | 160 | 1 | 24 | 9 of 12 LDS loads are `ds_load_u16` |
| `wm_pv_n256` | 16 | 858 | 16 | 324 (196 `ds_swizzle_b32` + 128 `dpp`) | **129 `ds_load_u16`**, 66 LDS stores |
| `wm_raw_floor` | 1 | 122 | 1 | **0** | hand-fed, contiguous map assumed |
| `wm_raw_floor_k256` | 16 | 220 | 16 | **0** | 64 `ds_load_b128` |

**Marginal cost per 16×16×16 tile (Δ ÷ 15):**

| leg | slots/tile | composition | slots per **touch-equivalent** | % of the shipped 45-slot touch bill |
|---|---|---|---|---|
| score, map-aware (rocwmma) | **13.5** | 1 WMMA + 2 `ds_load_b128` + **8 lane-movement** + ~2.5 wait/address | **0.84** | **1.9 %** |
| PV, map-aware, fed from a `s_sc` row | **53.6** | 1 WMMA + 8 element-wise `ds_load_u16` + **20 lane-movement** + store | **3.35** | **7.4 %** |
| score, hand-fed raw builtin | **6.5** | 1 WMMA + 4 `ds_load_b128` + 0 lane-movement | **0.41** | 0.9 % |

**Did the compiler have to move lanes? Yes.** 8 lane-movement ops per score tile, 20 per PV tile.
The WMMA operand map is **not** the engine's row-major 8-dims-per-lane slice, so `rocwmma` inserts a
`ds_swizzle_b32`/`dpp` cascade to redistribute. Two consequences, both measured, not inferred:

- The lane-movement tax is **~7 slots per tile** (13.5 map-aware vs 6.5 hand-fed). So it is real,
  and it is about half the score-leg feed. **But at 0.84 slots per touch it is 1.9 % of the
  shipped touch bill — it cannot refute the route.** The question `prototype/issue_rate_probe`
  §5.1 left open ("no shipping WMMA kernel may assume a hand-rolled fragment layout") therefore
  resolves as: the assumption is unnecessary *and* not worth the fight, because paying the
  map-aware feed still retires the score dot for ~2 % of what it costs today.
- The **PV leg is materially worse and for a layout reason**: feeding A from `s_sc`'s 1024-byte
  row lowers to **element-wise `ds_load_u16`** (8 per tile) instead of `b128` feeds, plus 20
  lane-movement ops. That is 4× the score-leg feed and it is a fact about *this* staging choice,
  not about WMMA — a P tile re-staged into a fragment-friendly layout would pay less, but the
  re-stage is work this route is supposed to be removing.

VGPR/LDS from the build metadata: `wm_score_tile` 28 VGPR, `wm_pv_tile` 29, `wm_score_k256`
**116**. The shipped kernel is 47 VGPR / 12,288 B LDS.

---

## 4. Geometry tax, priced in the repo's own currency (LDS → blocks/CU → wavefronts/CU)

`prototype/attn_m2/results.md` §1.5/§2 establishes the currency and the shipped baseline:
`attn_tile_wave` at 12,288 B × 4 warps → 65536/12288 = **5 blocks/CU = 20 wavefronts/CU**;
m2 at 22,656 B → 2 blocks/CU = 16 wavefronts.

| candidate shape | LDS/block | blocks/CU | VGPR constraint | wavefronts/CU |
|---|---|---|---|---|
| shipped (M=1) | 12,288 B | 5 | 47 → not binding | **20** |
| 4 rows × 4 heads, scores staged whole-row (16×512 bf16 = 16 KB + K + V) | ~24 KB | **2** | must stay ≲ 96 to hold 2 blocks at 8 warps | 16–32 depending on warps/block |
| 16 rows × 1 head, whole-row scores | 32 KB + K + V | **1** | D-fragment N-split needed | collapse |
| 4×4 with flash-style **tile-local** scores (16×16 f32) | ~9 KB | **6** | +accumulator pressure | ≥20, no loss |

The last row is the only geometry that does not lose wavefronts, and it requires abandoning the
shipped two-pass structure (whole-row scores exist so pass 2 can re-read them), which is a deeper
restructure than this item covers. Note the honest correction to my earlier checkpoint: m2's loss
mechanism was partly *block count below the ~80-block bandwidth knee*
(`prototype/attn_m2/results.md:291-296`), and at the funded ruler the WMMA grid (256–1024 blocks)
stays above that knee — so I do **not** claim m2's number transfers wholesale. What transfers is
the LDS→blocks/CU term, which §5 shows is enough on its own.

**Cycles-per-dependency at realised occupancy (D32/D33 rule, not an instruction count).**
Per wave at the funded 64-row / 4096-lane basis: 2 × 64 × 4096 × 256 × 8 ≈ 1.07e9 MACs. The matrix
unit's whole-device rate is 4 MACs/lane/clk × 64 lanes/CU × 40 CU × 2.9 GHz = 2.96e13 MACs/s
(roof basis: `devices/strix-halo-compute.md`; measured 52.63 TFLOP/s = 88.8 % of roof, so the
roof basis is the conservative one). That is **≈ 36 µs of matrix-unit time against 257.56 ms of
`attn` — the matrix unit is ~99.99 % idle for this workload.** The dependency chain is 16 dependent
WMMA per score tile (K=256), ≈ 256 clk; with 16–20 wavefronts/CU resident (2–2.5 per SIMD on the
matrix path) that chain is throughput-bound, not latency-bound. **So unlike `gdn_collapse` route A
(1.6 working wavefronts/CU, 9.8× over its instruction-count price), the latency-hiding premise
holds here** — and it is the one part of the price where I can say the naive model is not the
failure mode. The failure mode is the ceiling and the geometry, which is what the kill arithmetic
below is built from.

---

## 5. Kill band, then the arithmetic against it

**Kill band: the route must be shown to beat `attn` by more than ~15 %.** Justification, all three
from measured repo outcomes, not taste:
- `prototype/attn_m2/results.md` §2 + §8.4: an M=2 re-tile, **bit-identical** with bytes demanded
  halved, measured **7–10 % slower** at every width taken, and priced at **15–29 % slower** at the
  funded ruler. A geometry change of exactly this kind has one prior, and it is negative.
- `prototype/gdn_collapse/results.md` §4–§5: an instruction-count model missed by **9.8×** because
  it priced instructions instead of latency at realised occupancy.
- D32/D33: slot elasticity is not stable on this box (a −7.7 % slot delta bought +15 % elsewhere),
  so any share→time mapping must use *this kernel's measured* elasticity, not a linear one.

**The arithmetic.** D21/D22 measured this kernel's own elasticity: 116 → 45 slots/touch bought
1.466–1.521×, i.e. `time ∝ slots^0.39–0.44`. Applying it to a retirable fraction *f* of the touch
stream, net of the measured feed cost (0.84 + 3.35 = 4.2 slots/touch ≈ 5.5 % of the stream):

| retirable share f | basis | f_net after feed | attn time change | after 7–29 % geometry tax |
|---|---|---|---|---|
| **0.279** | D20's measured dots-only share, adopted by D30 §2.4 as this route's ceiling | 0.224 | **−9.6 %** | **−2 % to −33 %… i.e. straddles zero** |
| ~0.42 | dots + cross-lane fold + bf16 round/store (my static estimate, **not census-verified**) | ~0.37 | −16 % | −5 % to −38 % |
| ≥0.50 | would require fold + staging + address work all retiring | ~0.45 | −21 % | −8 % to −43 % |

At the **measured** denominator the route buys **−9.6 %** gross and cannot be shown to beat the
7 %-floor geometry tax, let alone 15 %. The only rows that clear the band require the
matrix-addressable share to be ≥ ~50 % of the touch stream — **more than D20 measured and more than
D30 §2.4 booked as this route's ceiling**, and precisely the quantity the whole-kernel census
(§6) failed to certify.

So the route falls on the kill side of its own band, and it falls there **at the ceiling the parent
row already adopted**, which is the strongest form of the refutation: even granting D30's stated
premise (bf16 present, no format change needed for *operands*), and even after measuring the feeder
cost as cheap, the arithmetic does not reach the band.

---

## 6. Two costs the last shipped win did not have to pay

**(a) The PV/streaming leg is a format change, not a re-association.** The funded wide-wave route
accumulates PV with **f32** weights:

```
engine/attn_slim_kernels.h:307   for (int u = 0; u < 8; ++u) acc[u] += e[slot] * bf(v[u]);
```

`e[]` is `float e[TILE]` (`:298`). WMMA's P operand must be bf16. Replacing this line with a WMMA
tile therefore changes the **operand format** of the softmax weights on the shipping path, not just
the summation order. Per the standing numeric-format rule that is admissible if the gates stay
green — but it needs a floor probe first, and the AGENTS trap row ("max-ULP bands are blind to
whole-tensor 1-ULP error") says the band has to be sized by a probe, not by taste. That is a real
bill this item has not yet paid, and it exists **only** for the PV leg.

**(b) Bit-identity is forfeited on both legs.** The 1.52× slim lever that got shipped was
bit-identical (ndiff 0), and that property is what made it free. A WMMA score dot does **not**
reproduce §05.5's serial reduction order, and the repo's own trap row is explicit that a
re-associated product tree is "a different number" (the GDN q̃ case moved **93,522 of 131,072 `o`
elements** from a hoisted scalar). Every wide-wave row with decision room could flip. The
instruments exist (floor probe + cos/meanrel), but "bit-identity preserved" is off the table, and
that is exactly the property D22 was bought on.

---

## 7. Script status in `prototype/attn_wmma/` (honest)

- **`region_census.py` — HALF-FINISHED, LAST EDITS UNVERIFIED.** Do not cite its output as a
  denominator. It now reports **125.5 slots/touch vs the published 45, `SOFT-CALIBRATION-FAIL`**,
  after two late changes that were never re-run clean: the `dot` multiplier 1.0 → 0.5 (both
  `attn_slim_dot<8>` and `<4>` inline on the same source lines, only one executes) and a
  `WIDE`/`SHORT` pass-2 route split. Both edits are defensible in isolation, the roll-up they
  produce is not, and the calibration failure is left visible rather than tuned away, per
  instruction not to re-green a census that cannot move the verdict.
- **`explore_loops.py` — dead end:** LLVM fuses pass 1 and pass 2 into single back-edge blobs
  (one region of 837 instructions / 192 blocks), so loop-graph attribution cannot separate them.
- **`isa_census_attn.py` — superseded:** first source-line attempt, replaced by `region_census.py`.
- **`list_wmma_builtins.py` — dead end:** this toolchain's `clang -cc1` has no `-dump-builtins`;
  `rocwmma` headers were read instead.
- Kept and working: `wmma_feeder_probe.cpp`, `build.sh`, `feeder_census.py`, `rw_test.cpp`,
  `feeder_census.output.txt`, and `isa_census.engine.output.txt` (the §1 reproduction of the
  shipped 45/116 bills, produced by running `prototype/attn_valu_slim/isa_census.py` against the
  engine listing — it is the witness for the denominators, and it is the one number in this
  document that is a direct re-measurement of the shipped binary rather than a citation). Build
  output lives in `tmp/attn_wmma_price/` (scratch).

---

## 8. What would reopen this, and why it does not change the verdict

If the parent disagrees with the kill, the reopen lever is **not** a device measurement — it is a
number: **the matrix-addressable share of the per-touch stream must be shown to be ≥ ~50 %, not the
27.9 % D20 measured.** Two commands price it without writing a WMMA kernel:

```bash
# 1. finish the census the honest way: split WIDE-route pass 2 and re-derive the dot multiplier
python3 prototype/attn_wmma/region_census.py --calib \
    --out prototype/attn_wmma/region_census.output.txt   # must land ~45 / ~116 pass-1 first

# 2. price the geometry directly on existing code: the M-row ladder, no WMMA involved
prototype/attn_m2/run.py --rows 4    # and --rows 16; main-agent, exclusive lock, paired A/B
```

Row 1 tests the only assumption that can lift the prize into the band; row 2 prices the geometry tax
in one lock hold on binaries that already exist. If (1) stays at 27.9 % **or** (2) shows M=4 already
costing ≥ 10 %, the verdict here is confirmed and no WMMA kernel should be written. Neither can
rescue the route via the feeder: I measured the map-aware feed at **0.84 slots/touch (1.9 % of the
shipped 45)**, and the hand-fed floor that `issue_rate_probe` §5.1 could not certify would only move
that to 0.41 — **0.43 slots/touch of upside, which cannot close a 5-point arithmetic gap.**

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Delivered exactly the two requested items and nothing else: the compile-only WMMA feeder probe (prototype/attn_wmma/wmma_feeder_probe.cpp + build.sh + feeder_census.py) and a single one-of-three verdict (funded no). Explicitly dropped the whole-kernel census roll-up and the full packing table as prerequisites, per the narrowed brief; region_census.py is left half-finished with its calibration failure recorded instead of tuned."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Handoff names the refuting number (-9.6% of attn at D20's measured 27.9% share x this kernel's measured slots^0.4 elasticity, net of the measured 4.2 slots/touch feed) against a geometry tax measured in-kind at 7-29% (prototype/attn_m2/results.md §2, §8.4), plus the full feeder census table with per-tile marginal costs, lane-movement counts, VGPR/LDS metadata, and the reopen commands."
    }
  ],
  "changedFiles": [
    "/home/erokhins/.pi/agent/sessions/--home-erokhins-erokhins_projects-qwen3.5-0.8b--/subagent-artifacts/outputs/a3e0693f-8448-4aa1-99b4-2b62b7064aa0/prototype/attn_wmma/HANDOFF_price.md",
    "prototype/attn_wmma/wmma_feeder_probe.cpp",
    "prototype/attn_wmma/build.sh",
    "prototype/attn_wmma/feeder_census.py",
    "prototype/attn_wmma/feeder_census.output.txt",
    "prototype/attn_wmma/rw_test.cpp",
    "prototype/attn_wmma/region_census.py",
    "prototype/attn_wmma/explore_loops.py",
    "prototype/attn_wmma/isa_census_attn.py",
    "prototype/attn_wmma/list_wmma_builtins.py"
  ],
  "testsAddedOrUpdated": [
    "prototype/attn_wmma/build.sh (compile-only, --offload-arch=gfx1151 --save-temps -c; never launched)",
    "prototype/attn_wmma/feeder_census.py (ISA census over the generated .s; output in feeder_census.output.txt)"
  ],
  "commandsRun": [
    {
      "command": "bash prototype/attn_wmma/build.sh",
      "result": "passed",
      "summary": "hipcc -O3 --offload-arch=gfx1151 --save-temps -c; compiles, emits 6 kernels incl. v_wmma_f32_16x16x16_bf16; no launch"
    },
    {
      "command": "python3 prototype/attn_wmma/feeder_census.py tmp/attn_wmma_price/wmma_feeder_probe-hip-amdgcn-amd-amdhsa-gfx1151.s",
      "result": "passed",
      "summary": "per-kernel slot/lane-movement/LDS census: score 13.5 slots/tile (8 lane-movement), PV 53.6 (20, 129 ds_load_u16), hand-fed floor 6.5 (0 lane-movement)"
    },
    {
      "command": "python3 prototype/attn_wmma/region_census.py --calib",
      "result": "failed",
      "summary": "reports 125.5 slots/touch vs published 45 -> SOFT-CALIBRATION-FAIL; left failing and labelled unverified per instruction not to re-green it"
    },
    {
      "command": "device runs",
      "result": "not-run",
      "summary": "no device run made, no timing taken, no gpu_lock touched; verdict priced from source, static ISA, and ledger facts"
    }
  ],
  "validationOutput": [
    "wm_score_tile 155 instr / 1 WMMA / 24 lane-movement; wm_score_k256 358 / 16 / 144 -> marginal 13.5 slots per 16x16x16 tile, 8 lane-movement per tile",
    "wm_pv_tile 160 / 1 / 24; wm_pv_n256 858 / 16 / 324 with 129 ds_load_u16 -> marginal 53.6 slots per tile when A is fed from s_sc's 1024-byte row",
    "wm_raw_floor 122 / 1 / 0 lane-movement; wm_raw_floor_k256 220 / 16 / 0 with 64 ds_load_b128 -> marginal 6.5 slots per tile (map unproven)",
    "vgpr_count: wm_score_tile 28, wm_pv_tile 29, wm_score_k256 116 vs shipped attn_slim_wave 47 / 12288 B LDS",
    "engine listing group_segment_fixed_size = 12288 for attn_tile_wave and attn_slim_wave; num_vgpr 47",
    "ceiling basis: 2 x 64 x 4096 x 256 x 8 = 1.07e9 MACs / 2.96e13 MACs/s = ~36 us of matrix-unit time vs 257.56 ms attn (matrix ~99.99% idle)"
  ],
  "residualRisks": [
    "The verdict's leverage sits on the retirable-share denominator. At D20's measured 27.9% the route is at -9.6% gross and cannot beat a 7% geometry-tax floor; clearing 15% requires ~50%, which is above the ceiling D30 §2.4 itself booked. If a corrected whole-kernel census showed >=50%, the verdict would need reopening (commands in §8).",
    "The hand-rolled 6.5-slot/tile floor assumes a contiguous fragment map that has never been proven on device; issue_rate_probe §5.1 forbids assuming it. It does not affect the verdict (0.43 slots/touch of upside), but nobody should treat wm_raw_floor as a legal kernel today.",
    "The PV feed cost (53.6 slots/tile) is a fact about feeding A from s_sc's ldm=512 row, not about WMMA intrinsically; a fragment-friendly P re-stage could lower it. It cannot rise enough to rescue the route, only to make a funded attempt worse than priced.",
    "The matrix-unit roof basis (4 MACs/lane/clk) is a roof, not a sustained rate for this chain shape; the measured 88.8%-of-roof WMMA number is the anchor. This only makes the arithmetic prize estimate slightly pessimistic, which is the safe direction for a kill.",
    "region_census.py is left half-finished and its calibration currently fails; its output must not be cited as a denominator. Three scripts (explore_loops.py, isa_census_attn.py, list_wmma_builtins.py) are dead ends and each is labelled as such in §7 and in-file."
  ],
  "noStagedFiles": true,
  "diffSummary": "New compile-only WMMA feeder probe plus its ISA census script, and the final D30-2.4 pricing handoff ending in 'funded no'; census scripts left labelled as half-finished or dead ends.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "Verdict: funded no. The feeder probe (the item this task existed to settle) came back CHEAP - the map-aware score feed is 0.84 slots per touch-equivalent, 1.9% of the shipped 45-slot touch bill, and yes the compiler does move lanes (8 ds_swizzle/dpp per score tile, 20 per PV tile, plus per-element ds_load_u16 when the PV A operand is fed from s_sc's 1024-byte row). So issue_rate_probe §5.1's open question is closed as non-fatal. The route dies instead on the ceiling and the geometry: at D20's measured 27.9% arithmetic share and this kernel's own measured elasticity (time proportional to slots^0.39-0.44) the whole matrix prize is -9.6% of attn net of feed cost, while the only measured precedent for the re-tile WMMA forces (M=2, bit-identical, bytes halved) landed 7-10% slower and was priced 15-29% slower at the funded ruler. That is inside the ~15% kill band and at the central case inside zero. Two further costs the slim win did not pay: the streaming PV route accumulates with f32 e[slot] (engine/attn_slim_kernels.h:307), so WMMA there is a P-operand FORMAT change needing a floor probe, and bit-identity is forfeited on both legs (the repo's own trap row shows a re-associated product tree moving 93,522 of 131,072 elements). No device run was made and no timing taken; no engine file, bench doc, or gate was modified; nothing staged."
}
```<|im_end|>
<|im_start|><|im_end|>
<|im_start|>user
<|im_start|>assistant<think>
I have written the handoff and the acceptance report. But wait — I should double check one thing: did I actually verify that the handoff file was written? The write tool call succeeded presumably. Let me verify the file exists and is complete. Also I claimed in the acceptance report that the three dead-end scripts are "labelled as such in §7 and in-file" — I need to verify they actually have the one-line labels in-file, since the brief required "Leave three dead-end scripts in place with one line each saying why they are dead." Let me check and fix if needed.

Also I should verify region_census.py header says it's half-finished (brief requirement: "its header must say so").

Let me check those quickly and fix.
</think>

<tool_call>
<function=bash>