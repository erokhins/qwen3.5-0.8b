# A11 — attribution of the `attn_fattn_wave_u8` window at 32,768 lanes: where the time actually is

**CPU-only.** Source + static ISA + three published paired-window measurements. No timing taken here, no
correctness gate, no `rocprofv3`, no `gpu_lock`, no HIP launch. Stall counters are *not* re-litigated: §54 of
`bench/performance-status.md` establishes they are unattributable on this part (`DRAM_REQ_PER_CLIGRP` and
`TCP_BACK_PRESSURE` read exactly 0.0 = vacuous; `SQ_WAVE_CYCLES/SQ_BUSY_CYCLES` reads above the architectural
ceiling), so this file prices **cycles-per-dependency at the realised occupancy** and never instructions per token.

Deliverable question: **can deleting the Q re-read + deepening the KV stage (A10's R2 + R3) recover 45 % of the
window?** Answer below: **no — and not by a small margin.**

---

## BET (pre-registered, A11-ATT)

**I bet that the Q-delivery channel — the whole of what R2 attacks — is worth ≤ 12 % of the 32,768-lane attention
window, and the round-fixed work that a 4× deeper stage divides is worth ≤ 3 %.**

**The single falsifying number:** a paired equal-LDS `qstage` A/B at 32,768 lanes — arm A = shipped `unstaged8`
padded with a dummy `__shared__` array to `LDS_Block_Size = 32,768` so both arms run 2 blocks/CU and 6 wf/CU;
arm B = `--attn-fattn-qstage stage` (Q fed from LDS, zero Q global traffic per round). **If the LDS-fed arm wins
≥ 25 % of the traced attention phase at equal residency, the bet is falsified and R2's premise is live.**

**Does this attribution already falsify the bet? No — and not because the data is thin, but because three
independent instruments land inside the bet:** G1's own row decomposed by every residency exponent on record gives
**4.5–9.9 %**; a latency-class bound on the Q returns gives **2.6–8.7 %**; and even if every Q return were a DRAM
miss it could not exceed **18 %** (§4). The equal-residency A/B has never been run, so the number is owed — but the
route's own premise is already bounded 3× below its bar.

**Verdict: DO NOT FUND** A10's R2+R3 as the route to 6.5 TFLOP/s. One cheap measurement (§6) would reopen it; it
is not the rewrite.

---

## 0. Two corrections to the premise before any arithmetic

1. **The premise's window/rate pair is not reproducible tree-wide.** `119.69 ms` / `4.33 TFLOP/s` appear nowhere
   on disk. The nearest recorded rows are 118.548 ms (grid 371,712) and 120.586 ms (grid 380,160) per 6-layer pass.
   The window used here is taken from the A11 pmc dispatch timestamps themselves: **378 dispatches of
   `engine::kernels::attn_fattn_wave_u8`, Σ(End−Start) = 5,598.220 ms/pass** at 32,768 lanes
   (`tmp/a11/pmc2/a32k_counter_collection.csv`), which agrees with A9 §2.5's traced 5,583.87 ms to **+0.26 %**.
   Widest single body (grid 540,672 = 5,632 blocks, split 64): 169.559 ms over six layers = **28.26 ms/layer**;
   smallest body 4.621 ms. Useful work is 26.389 TFLOP/pass ⇒ **kernel-only rate 4.714 TFLOP/s = 8.96 % of the
   52.63 TFLOP/s WMMA roof** (`prototype/issue_rate_probe/results.md` §1).
2. **"+45 %" is a throughput delta; the window delta is smaller.** Clearing 6.5 TFLOP/s from the kernel-only 4.714
   needs 26.389/6.5 = **4,059.8 ms = −27.5 % of the window**. On the phase rate A9 §1.2 books (4.466 TFLOP/s,
   5,908.16 ms) it is **−31.3 %**. Both are quoted below; neither is 45 %.

---

## 1. Denominators — every one printed, both clock divisors

| quantity | value | source |
|---|---|---|
| window, kernel-only, 32,768-lane pass | **5,598.220 ms** (378 dispatches) | pmc dispatch timestamps; A9 §2.5 traced 5,583.87 ms (+0.26 %) |
| block-rounds / token (6 layers) | 1,115.24 | A9 §2.4 (self-tested geometry mirror) |
| wavefront-rounds / pass (×3 warps ×32,768 tok) | **1.0963e8** | arithmetic on the row above |
| independent cross-check | 1,097,712 blocks/pass (Σ grid/96, one row per dispatch) ⇒ **33.29 rounds/block** vs the source rule `ceil(span/16)` ≈ 32 | pmc grids; `engine/attn_fattn_kernels.h:771` |
| SIMDs | 80 (`simd_per_cu 2`, 40 CU) | KFD, A10 §1 |
| clock divisors | **2,894 MHz** (booked MAX, `issue_rate_probe/README.md`) and **2,206 MHz** (ordinary). Every cycles↔ms figure below names which. | ledger convention |
| **SIMD-cycles available per wavefront-round** | **11,822** @2894 · **9,012** @2206 | window × SIMDs × f ÷ wavefront-rounds |

**Two occupancy readings, kept separate on purpose.** The shipped banner/§48 reading is `185 VGPR, LDS 20,480 B`
⇒ 3 blocks/CU ⇒ **9 wf/CU = 4.5 wf/SIMD**. The briefed A11-occupancy reading (driver `VGPR_Count = 248`, which is
also what the pmc CSV reports for the widest body) gives the register file as the binder: 3×96×248 = 71,424 >
65,536 ⇒ **2 blocks/CU = 6 wf/CU = 3 wf/SIMD**. Neither file (`A11-occupancy.md`, `A11-counter-availability.md`)
exists on disk, so both are carried. **The percentages in §2–§4 are per-wavefront-round aggregates and do not
depend on which is right**; only the round wall and the per-return price do, and both are given at 4.5 and at 3.

---

## 2. The ISA bill per wavefront-round

Region-split with `prototype/gdn_rec_isa/token_bill.py`'s block/loop machinery over
`kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s` (regenerated compile-only by
`bash prototype/attn_wmma/fattn_census_build.sh /tmp/a11census`). Round loop = `.LBB8_69` (depth 1: 57 blocks, 703
slots once through), with three subloops: the **score roll** `.LBB8_73` (53 slots/trip × **16** trips), the **K
stage fill** `.LBB8_71` and the **transposed-V fill** `.LBB8_169` (36 and 39 slots/trip × 16 trips spread over 3
warps ⇒ **5.33** trips per wavefront).

| | slots/wavefront-round | | mnemonic | per round |
|---|---|---|---|---|
| work slots (census convention) | **1,876** | | `v_wmma_f32_16x16x16_f16` | **32** |
| encode-only issue slots (`s_waitcnt` 81 + `s_delay_alu` 166) | 247 | | `ds_load_b128` (K/V wmma operands) | **64** |
| **total issue slots** | **2,123** | | `ds_store_b128` / `ds_store_b16[_d16_hi]` | 4.3 / 34.7 |
| VALU | 1,441 (67.9 %) | | `global_load_b128` (32 of them Q) | 40.7 |
| SALU | 494 | | `ds_bpermute_b32` / `v_exp_f32_e32` | 8 / 8 |
| LDS | 111 (5.2 %) | | `v_cvt_f16_f32_e64` / `v_mul_f32_e32` | 341 / 144 |
| VMEM | 41 | | `s_barrier` | **4** |
| WMMA | 32 | | | |

**The roll's shape is the whole Q story** (`.LBB8_73`, one trip):
`2 × global_load_b128` (the Q tile) ‖ `2 × ds_load_b128` (the K tile) → `s_waitcnt vmcnt(1)` → bf16 unpack
(8 `v_and_b32` + 8 `v_lshlrev_b32` + 8 `v_mov_b16`) → `s_waitcnt vmcnt(0)` → 16 `v_cvt_f16_f32_e64` →
`s_waitcnt lgkmcnt(0)` → `v_wmma`. **No software pipelining**: each of the 16 trips fully exposes its own Q return,
and the Q unpack is 640 of the round's 2,123 slots (30 %) *plus* 16 of the round's 26.7 exposed memory-return sites
(60 %). That is precisely R2's target mass: **672 slots = 31.7 % of the issue bill**.

**Round-fixed vs per-KV-position** (what a deeper stage can and cannot divide): per-KV-position work — score roll
(848), stage fills (400), P·V leg (236), exp/max/P-pack (~280) — is **1,765 of 2,123 slots = 83 %**. Round-fixed is
17 %, and only **111 slots (5.2 %)** of it are actually removable by a 4× deeper round: the 128-multiply rescale
(`engine/attn_fattn_kernels.h:879-881`, A9 §2.4's 7.7 %-of-budget item), 3 of the 4 barriers, loop control. A
stage of any depth still needs one barrier set per round and one softmax fold per cell.

---

## 3. Where the window goes (shares of the 11,822 SIMD-cycles per wavefront-round @2894; ×1.312 @2206)

These are **per-resource occupancies, not additive shares of time** — which is the finding: no resource is
saturated, so the window is *waiting*, and the attribution is the list of what it is not waiting for.

| category | value | share | method | confidence |
|---|---|---|---|---|
| **WMMA issue** | 32 × 34.08 = **1,091 clk** | **9.2 %** (12.1 % @2206) | 32 sites counted in the ISA; 34.080 clk/wmma/SIMD from `prototype/issue_rate_probe/evidence/wmma-latency.txt` `wl_f32_i1` (saturated reciprocal throughput at 8 wf/SIMD, `--clock-mhz 2894`) | **high** — cross-checked from the other end: kernel-only rate is 8.96 % of the same roof. Because the probe rate is a *saturated* number and the kernel runs 3–4.5 wf/SIMD, 9.2 % is a floor. |
| **instruction issue (all classes)** | 2,123 slots | **18.0 %** (23.6 % @2206) | one issue slot per instruction per SIMD (RDNA has one VALU issue slot/SIMD); 2,123/11,822 | **high**. Consequence: **~82 % of issue capacity is unused ⇒ the kernel is not instruction-issue-bound.** Consistent with §54's `INST_ANY` 16.3 %, which cannot be claimed and is not claimed. |
| **LDS issue + service** | 111 issue slots (5.2 % of the bill); service 313 → ~1,050 clk/wf-round | **5.3 % → 19.2 % of the LDS pipe** at 3 blocks/CU | ideal service (b128 = 4 clk, b16 = 1) = 313; ×16 conflict on the 64 operand `ds_load_b128` = 19.2 %. Saturation needs an **86× multiplier** ⇒ LDS is not the wall | **medium** — the conflict factor is modelled from the address arithmetic, not measured (§5) |
| **VMEM / LSU / cache service** | Q demands **321 GB/s** at the LSU (1.80 TB/pass), total demand 2.39 TB/pass = 428 GB/s | **2.9 %** of L1 return bandwidth (40 CU × 128 B × 2.894 GHz); Q's DRAM-class share ≈ 98 GB/s = **41 % of the 237 GB/s DRAM ceiling** | 32 `global_load_b128`/round × 512 B × wavefront-rounds; L1 absorbs 73 % (§54's 26.8 % miss rate); §54's own DRAM census is 131 GB/s | **medium** — bandwidth demand is not service time; the honest statement is that **no memory pipe is saturated** |
| **VALU / softmax** | 1,441 slots, of which the Q bf16→f16 unpack is 640 | ≤ 12.2 % of the window **if** issue-bound (it is not) | ISA census | high for the count, low as a time share |
| **barriers** | 4 per round | **unattributable** | §54's `BARRIER` 11.2 % cannot be claimed on this part | none — named, not priced |
| **exposed dependency latency (the residual)** | ≥ 65–74 % of the window is waiting with nothing ready | — | 26.7 exposed memory-return sites per round (16 Q + 10.7 stage fill); round wall = 53,200 cycles @4.5 wf/SIMD (35,467 @3) ⇒ **filling the round with returns alone costs 689 ns per return (460 ns at 3 wf/SIMD)** — above any credible single-return latency on this part | **bounded, not attributed.** The named owner is queue/service occupancy (A9 §5.1's "VMEM/LSU queue slots… the one resource this repo has never counted"), not instruction supply |

---

## 4. What deleting the Q re-read is worth — two independent instruments, same band

1. **G1's row, decomposed by the residency elasticity.** §48 measured `stage` (Q in LDS, 6 wf/CU, LDS 32,768) →
   `unstaged8` (Q re-read from global, 9 wf/CU, LDS 20,480) as **−18.61 % of the wave at 32,768, byte-identical**
   (ratio 0.8139). Residency alone predicts (9/6)^−e: **0.779** at e=0.616, **0.747** at e=0.72 (the §31 value,
   independently validated at G0: predicted −10.5 %, measured −10.82 %), **0.741** at e=0.74. The Q channel is the
   residual: **+4.5 % / +9.0 % / +9.9 %**. Circularity to declare: e=0.616 was fitted to G1's *4,096* family row
   (1.5^−0.616 = 0.7791 = that ratio exactly), so the 32k residual under it is a real residual but the exponent is
   from the same experiment; e=0.72 is not.
2. **Latency-class bound.** The re-read is ~1,100× redundant over a 24 KB/block working set (A9 §5.1, refined by
   §54: 3 blocks ⇒ 73,728 B > the 32 KB L1, so the deep tiers are load-bearing) — i.e. L1/L2-class returns, not
   DRAM. 16 exposed Q returns × (30–100 ns) = 1,390–4,640 cycles = **2.6–8.7 % of the round wall**. If *every* Q
   return were instead a DRAM miss (contradicted by §54's 131 GB/s census against a 237 GB/s ceiling) the ceiling is
   **18 %**.

R2 does better than the LDS-fed arm by exactly one thing: the LDS round-trip it avoids. That was measured — A4-L1's
two-accumulator roll batched a pair's shared-tile operand loads and halved the accumulator chain and bought
**−1.67 % of family / −0.28 % of the traced phase at 4,096** on the pre-G1 Q-in-LDS form (A9 §2.4's reading: slot →
time pass-through **≈ 0.217**). So **R2's ceiling ≈ 4.5–9.9 % (channel) + ~3 % (beating the LDS feed) ⇒ ≲ 12 % of
the window**, which is the bet above.

**R3's ceiling**, at the same bar: it divides the 111 round-fixed slots (≤ 0.9 % of the window even at 1:1
pass-through; ≤ 0.2 % at 0.217), and its real lever is traffic-per-MAC (9.6 → 31 MAC/byte). Traffic deletion is the
*measured* worst converter on this kernel: §53's L-c removed **77 % of a 105 GB/pass DRAM-certified round-trip and
bought −4.61 % of the family, −2.38 % of the wave**, and D21/§20.7 already showed a demanded-bytes ÷ time figure
near the DRAM plateau is not a bandwidth wall. And R3 at 3 warps is L-b, which is dead on residency: 9→6 wf/CU costs
**+28–35 %** by §48's own elasticity.

**Net against the bar.** The funded shape is R1+R2+R3. R1 gives up residency the ledger has measured to convert:
9→8 wf/CU = **+7.5–9.1 %**, so the geometry alone must deliver **32.6–33.5 %** of the window to reach 6.5 TFLOP/s.
What R2+R3 can delete, priced by every instrument above, is **≲ 10–14 %**. The route is short by a factor of ~3.

---

## 5. The named category R2/R3 do not touch: bank-aliased LDS operand addressing

`fattn_kv_addr(kv,d) = kv*DH + d` (`engine/attn_fattn_kernels.h:226`) makes the per-lane stride of every
`ds_load_b128` that feeds a K wmma operand **512 B** (DH×2), and `fattn_v_addr(d,kv) = d*FATTN_MLANES + kv`
(`:227`, used at `:902`) makes the per-lane stride of every V-transpose `ds_store_b16` **256 B** — both whole
multiples of the **128 B bank group**, so `(byte/4) % 32` is lane-independent and a wavefront's 32 lanes land in
one bank group on every wmma operand read and every V-transpose store. The ISA confirms the shapes
(`ds_store_b16 v130, v132 offset:12288` … `+32 t`, i.e. per-lane 256 B, per-`t` 32 B). llama.cpp pads for exactly
this: its stage row stride is `nbatch_K2 + 4 = 132` half2, not 128 (A10 §3). **Neither R2 nor R3 changes either
expression** — a deeper stage keeps the same row stride, and Q-in-registers removes a *global* read, since the
shipping form does not read Q from LDS at all.

Bound, so this is not a second revolution: LDS-pipe occupancy runs **5.3 % (no conflict) → 19.2 % (16-way) →
saturating only at 86×**, so a pad (512→528 B per staged row, ~256 B extra per stage, no residency cost, addresses
only ⇒ byte-identical, free gate) is a **few-percent lever, not 45 %**. It is nonetheless the only untouched
category with a measured-shaped mechanism and a zero-risk witness.

---

## 6. The one cheap measurement owed (not the rewrite)

**Equal-residency Q-channel ablation.** Two arms, one `gpu_lock` hold, 32,768 lanes, traced:
`A` = shipped `unstaged8` + a dummy `__shared__` array padding `LDS_Block_Size` to **32,768 B**; `B` =
`--attn-fattn-qstage stage`. Both then run 2 blocks/CU, **6 wf/CU**, so the elasticity transfer disappears and the
Q channel is read directly. **Canary (behavioural, in-window, per the D30 lesson):** both banners must print
`lds=32768` and the same wavefronts-per-CU field, with the `q=` field differing; outputs must be byte-identical
(G1 already proved this pair is), which makes it gate-free. **Cost:** one scratch variant + one paired window;
correctness-only is not enough here, so it is a main-agent timing pass. **Adjudication:** ≥ 25 % for the LDS-fed
arm ⇒ the bet is falsified and R2's premise is live (re-price the route); ≤ 10 % ⇒ R2 cannot carry a rewrite and
the unattributed residual must be attacked as queue/barrier cost, not operand cost.

---

## 7. Commands run, files changed, and what could not be answered

**Commands (all CPU-only).** `bash prototype/attn_wmma/fattn_census_build.sh /tmp/a11census` (compile-only
`hipcc -c --save-temps`, ~3.3 s, emits `kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s`);
`python3 /tmp/a11att/model.py --json /tmp/a11att/model.json`; `python3 /tmp/a11att/bet.py`;
`python3 /tmp/a11att/final.py`; `python3 /tmp/a11att/dumploop.py --loop BB8_73 --out /tmp/a11att/roll.txt` (and
`--loop BB8_71`); plus read-only greps of `bench/performance-status.md`, `A9-LONGCTX-CENSUS.md`,
`A10-REWRITE-BRIEF.md`, `A4-L1-verdict.md`, `A6-G1-verdict.md`, `issue_rate_probe/evidence/wmma-latency.txt` and
the pmc CSV `tmp/a11/pmc2/a32k_counter_collection.csv`. **No timing taken, no device run, no `gpu_lock`.**
All analysis scripts are scratch in `/tmp/a11att/` and import the existing census machinery
(`prototype/gdn_rec_isa/token_bill.py`) rather than adding a fourth census tool. **Files changed: this file only.**
No census denominators were retyped: `blocks/pass` is counted once per dispatch (the pmc CSV emits ~9 rows per
`Dispatch_Id`, one per counter — naive summing inflates it 9×), and wavefront-rounds come from A9 §2.4's
self-tested 1,115.24 with the grid count as an independent cross-check.

**Could not be answered from CPU evidence, and what would answer it:**
1. **The conflict factor of the bank-aliased strides** (§5) — needs a device A/B of a padded `fattn_v_addr`;
   modelled as a band (5.3–19.2 % of the LDS pipe), not measured.
2. **Per-tier return latencies on gfx1151** (L1 vs the ~32 MiB tier) — the Q-channel latency bound in §4 uses
   30–100 ns as a class argument; a measured L1/L2 latency would tighten it to a number.
3. **Whether the unattributed ≥ 65 % is queue-slot occupancy** — A9 §5.1 names VMEM/LSU queue slots as the one
   resource this repo has never counted, and no counter on this part can see it (both candidate counters read
   exactly 0). §6's ablation bounds the Q share of it; the remainder stays unattributed.
4. **`A11-occupancy.md` and `A11-counter-availability.md` do not exist on the filesystem.** Their briefed figures
   (1.59 active wf/CU, LDS-bound 2 blocks/CU, 32.06 clk/wmma) are used as briefed and flagged; the independent
   on-disk probe value is **34.080 clk/wmma/SIMD**, used here, and the two occupancy readings are priced in
   parallel in §1 rather than reconciled, because the shipped `.vgpr_count` (185, §48) and the pmc
   `VGPR_Count` (248) disagree and the register-file binder is not knowable from the CSV.
5. **The premise's 119.69 ms / 4.33 TFLOP/s pair** could not be reproduced or traced to a table; the window used
   here is the pmc-measured 5,598.220 ms, which A9 §2.5 independently confirms to 0.26 %.
