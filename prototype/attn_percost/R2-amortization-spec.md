# R2 — the missing quantity is not ILP, it is amortization: 108 / 111 / 52 non-matrix slots per matrix slot (CPU-only census, 2026-10-20)

`evidence/r2_round_bills.txt`. CPU-only: listings carved out of the shipping binary and out of the llama
build, disassembled, class-histogrammed inside each kernel's widest backward branch by
`prototype/attn_percost/a2_round_bill.py` (`g3_isa_bill.py` finds the span; A2 §2 did this for our shipping
kernel only). No device, no timing. This is the item the queue called **R2**: turn A2's
"17 `v_wmma` sites against their 2,592" into a spec that says what a rewrite must achieve. It does, and
the spec is not the one the question assumed.

## 1. The ILP framing is closed before it starts, by a measurement already on file

`prototype/issue_rate_probe/results.md` §: `v_wmma_f32_16x16x16_f16` sustains **32.4 clk per WMMA per SIMD
at one independent chain (i1)** and 33.9 at four (i4) — i.e. **a single accumulator chain already fills the
matrix pipe's 32-cycle acceptance interval**, and `prototype/issue_rate_probe/A3-wmma-accumulator-form.md`
additionally found that four chains *together with four dependent LDS loads* degrade 8.5× because they
serialise on one load port. So "more independent accumulators per wavefront" is not the missing quantity:
at 4.5 wavefronts per SIMD (9 wf/CU ÷ 2 SIMD/CU) there is already ≥1 chain per SIMD ready to issue, and the
pipe accepts one every 32 cycles whether the chain is dependent or not.

What is left is the only question the ISA can answer: **how many non-matrix issue slots does a wavefront
spend per matrix slot** — the amortization A2 §2 named and did not compare across engines.

## 2. The three bills, one method, one tool

Span = each kernel's widest backward branch (their KV/round loop, our round loop); `v_wmma`/`v_mfma` counted
as matrix slots, everything else as non-matrix. Every WMMA does the same 4,096 MACs in all three, so the
ratio is also "issue slots per unit of matrix work".

| kernel | round slots | matrix slots | **non-matrix per matrix** | measured rate vs 52.63 TFLOP/s roof |
|---|---|---|---|---|
| ours, shipping `attn_fattn_wave_u8` (3 wf, M16 tile) | 1,858 | 17 | **108.3** | 4.5 TFLOP/s = 8.6 % |
| ours, wide `attn_wide_wave` (8 wf, 32 pos × 4 heads) | 14,325 | 128 | **110.9** | (rig figure only; in-engine ratio 1.0001) |
| llama `flash_attn_ext_f16<256,256,16,4>` (8 wf, `nbatch_fa=64`) | 13,537 | 256 | **51.9** | 7.99 TFLOP/s = 15.2 % |

Two readings, and the second is the one that survives:

* **The gap is fully accounted for at the instruction level.** llama's amortization is **2.09× better**
  than ours and their measured rate on this model's attention family is **1.9× ours** (A2 §3: 51.65 ms vs
  our 98.9 ms). Nothing else needs to be invoked to explain it.
* **A12 changed the ratio in the wrong direction: 108 → 111.** Its round does 7.5× the matrix work of the
  shipping round, but its non-matrix bill grew 8.2× (1,841 → 14,197 slots). That is the whole
  rig-vs-engine discrepancy in one number, and it is consistent with everything §56/§57 established:
  coverage was never the gap, residency is not the gap (adding 4× block population moved nothing), and the
  shape that "won" +25 % on the rig had *not* improved amortization at all, so there was nothing for the
  engine to inherit.

The instance matters and is now pinned: this model's `head_dim` is 256, and the carve contains eight
`flash_attn_ext_f16` instantiations (64/80/96/112/128/256/512/576×512). **The `<64,64,…>` one that a bare
symbol match finds first is not ours** — billing it gives 9,398 slots / 128 matrix slots = **72.4**, which
would have reported llama's advantage as 1.5× instead of 2.09×. Any future bill here must select
`flash_attn_ext_f16ILi256`, whose own metadata (256 threads, `private_segment_fixed_size = 900`) matches
the traced figures A2 §2 quoted.

## 3. Where the slots actually go, normalised per matrix slot

| bucket | ours ship | ours wide | llama | verdict |
|---|---|---|---|---|
| `int_vec` (address arithmetic) | 19.1 | **30.0** | 18.1 | **not the difference** — shipping already matches llama; the wide kernel is +65 % worse here |
| `salu` | 15.8 | **21.8** | 3.7 | **our largest single leak** (4–6×) |
| `move` | **19.8** | 2.9 | 2.7 | shipping's second leak (`v_mov_b32` 241 of 1,858); wide fixed it, so the fix is known |
| `delay` (`s_delay_alu`) | 10.6 | **18.9** | 4.5 | compiler-inserted, so a symptom of dependency shape: wide is 4.2× worse |
| `lds` | **12.2** | 3.6 | 5.2 | shipping's staging pattern (`ds_store_2addr_b32` 129) |
| `fmul` | **8.5** | 2.8 | 0.5 | the 128-slot accumulator rescale of A2 §2 / A4-L1; wide already 3× cheaper |
| `select_mask` | 7.5 | 9.6 | 2.6 | 2.9–3.7× |
| `cvt` | 3.4 | 4.4 | 1.3 | 2.5–3.4× |
| `fma32` | 1.0 | 8.7 | 2.7 | wide's score/normalise FMA mass |
| `wait` (`s_waitcnt`) | 4.2 | 2.4 | 4.2 | equal — nobody wins here |
| `exp` | 0.53 | 0.76 | 0.29 | a rounding error in all three |

So: **address arithmetic and memory-wait behaviour are already at llama's level.** The deficit is
scalar/control overhead (`salu`), register shuffling (`move`), the rescale and mask/select traffic, and —
for the wide kernel specifically — a dependency shape the compiler papers over with `s_delay_alu`.

## 4. The spec, in the only two forms it can take

Target: **≤ 52 non-matrix slots per matrix slot** (llama's, measured identically). Given that, either:

* **(a) hold the round's matrix work and shrink the non-matrix bill.** At 17 matrix slots the round must
  fall from 1,841 non-matrix slots to ≤ 884, i.e. **−52 %** — which is more than `move` (368) + `salu`
  (269) + `delay` (180) combined. Not credible inside the current form; and A4-L1's −41 % cut of the
  accumulator rescale alone already measured **−1.67 %, dead**, exactly as this arithmetic predicts
  (removing 128 of 1,841 is 7 % of a quantity that must fall 52 %).
* **(b) hold the non-matrix bill and widen the matrix work per round** — the direction A12 took, with the
  constraint A12 violated made explicit: **the non-matrix bill must not scale with the round's width.**
  At 64 matrix slots per round and today's shipping non-matrix bill the ratio would be 29, i.e. better than
  llama's; at A12's actual scaling (14,197 for 128) it is 111. Concretely the three buckets that scaled
  with width in A12 are the work list: `int_vec` 3,836 → ~2,300 (index arithmetic that is being redone
  per lane of the widened dimension instead of hoisted per round), `salu` 2,789 → ~600, `delay` 2,420 →
  ~600. **That is R3's brief, stated as an ISA target rather than as a design.**

## 5. What is falsifiable here, and what is not

Falsifiable and cheap: any candidate kernel — rewritten or restructured — whose round bills at > ~70
non-matrix slots per matrix slot **cannot** reach llama's rate, and this census will have said so before
it is written. Conversely a kernel that bills ≤ 52 should land near 15 % of roof; if it does not, this
model of the gap is wrong and §58 goes back on the record.

Not established by this document: that the ratio *causes* the rate rather than co-varying with it. A2 §2
showed the machine is far from issue-saturated overall (≤ 8.8 % of issue slots, 5.6 % matrix pipe), so
both kernels are stall-bound in wall-clock terms; the ratio is a per-wavefront serialised-work proxy, and
the two engines' stall profiles were not compared here (A11's `--pmc` pass covers ours: 65–74 % "waiting",
nothing saturated; llama's was never instrumented). The honest summary is that the ratio is the only
quantity found so far that closes the gap arithmetically — 2.09 for the ratio against 1.9 measured — while
the mechanism by which it bites (per-wavefront serial work under a stall-bound schedule) is inferred, and
the residency alternative that would have explained it differently was already killed in §57 R1b.
