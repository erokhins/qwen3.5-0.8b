# `fattn_map_probe` — §5d's fragment-map probe, RUN, and the map **does not hold as transcribed**

Device gate, correctness only, one launch per case, **no timing taken** (no millisecond figure appears
here). Appended at the end of `prototype/attn_wmma/` deliberately; nothing above or elsewhere moved.

Base `5666adc`. Run 2026-10-18 by a worker under `tools/gpu_lock.sh` (util read 37–91 % across the
window; the serving engine busy-polls the same iGPU, so `util` is recorded and discounted, per AGENTS.md).

## 0. The answer, in one line

The composition `wmma(Q_tile, K_tile, acc)` + "lane t holds `C[t%16][2l + t/16]`", which
`wmma_fattn_score.cpp`'s header states and every leg of the census prices, **is false on this part**.
The true law, measured, is:

> **With both operands staged row-major with the 16-half run along k — the `mma.cuh:552,560` spelling the
> form uses, unchanged — lane t's l-th C value is
> `sum_k FIRST[row = 2l + t/16][k] * SECOND[row = t%16][k]`.**
> i.e. the **first** wmma operand's row index lands in the C tile's **column** and the **second**
> operand's row index lands in its **row**: the two operand slots are exchanged relative to the
> transcription. Equivalently: `v_wmma_f32_16x16x16_f16` computes `D = SECOND · FIRSTᵀ` under the
> transcription's own operand-layout sentence.

`v_wmma_f32_16x16x16_f16` is *not* doing something alien — the tile really is a 16-deep accumulating dot
(`P2` below shows all 256 `head_dim`s contributing exactly once and the accumulator carrying across the 16
K-steps). What is wrong is only which operand is which.

## 1. What was built, and how to run it

| file | what it is |
|---|---|
| `prototype/attn_wmma/fattn_map_probe.cpp` | one warp, one launch, four coordinate-encoding patterns; stores every lane's 8 raw C values and that lane's own id. Takes `--inject=` (store-side permutations) and `--swap=` (exchange the two wmma operand slots) |
| `prototype/attn_wmma/build_map_probe.sh` | links it into `tmp/attn_wmma_map/fattn_map_probe`; **fails the build** unless the listing carries ≥ 17 `v_wmma_f32_16x16x16_f16` lines (the DCE witness: 1 + 16 tiles) |
| `prototype/attn_wmma/check_fattn_map.py` | the checker: builds if needed, launches once per case in its own subprocess, classifies the observed table against six candidate maps, and carries the injection self-tests. **Does not self-lock** (`grep -c gpu_lock` = 0) |
| `prototype/attn_wmma/evidence/fattn-map-probe.txt` | the recorded output of both commands below, with the `util` line |
| `prototype/attn_wmma/evidence/dump-as-transcribed.txt`, `…-swapped-operands.txt` | the raw per-lane dumps |

```bash
bash prototype/attn_wmma/build_map_probe.sh
python3 prototype/attn_wmma/check_fattn_map.py                      # PASS 26/26, two launches, 0.3 s wall
python3 prototype/attn_wmma/check_fattn_map.py --selftest           # PASS 12/12, three launches, 0.3 s
```

## 2. Why the patterns can falsify at all

§5d suggested `Q[q][d] = q`, `K[kv][d] = kv`. That pattern is **useless**: it yields `C[q][kv] = 256·q·kv`,
symmetric in `(q, kv)`, so a transposed tile satisfies it exactly and the probe would have passed on the
very error it was built to catch. Every pattern here instead makes a cell's value *name its own
coordinates*, all of them exact in f16 and f32 (integer products below 2²⁴), so a relabelling is a wrong
number rather than a plausible sum. The self-test proves this property rather than claiming it: it shows
the symmetric pattern has movement **0** against the transposed hypothesis, and the coordinate pattern
**240** of 256.

| pattern | operands (f16) | expected cell value | what it may adjudicate |
|---|---|---|---|
| **P1** | `Q1[q][0]=q, Q1[q][1]=1`; `K1[kv][0]=16, K1[kv][1]=kv`; one tile | `16q + kv` | orientation (injective and asymmetric); row-weight on the first operand, column-weight on the second |
| **P2** | `Q2[q][d]=q+1`, `K2[kv][d]=kv+1`, all 256 dims, 16 tiles accumulated | `256(q+1)(kv+1)` | **coverage and accumulation only** — symmetric on purpose, so it cannot see an orientation error |
| **P3** | `Q3 = Q1`; `K3[kv][0]=1, K3[kv][1]=16kv`; one tile | `16kv + q` | orientation, with the axis weights exchanged → an operand-side error moves P1 and P3 in opposite directions |
| **P4** | `Vt[d][0]=d, Vt[d][1]=1` at row stride 16 (the transposed-V stage); `Pm[q][0]=1, Pm[q][1]=16q` (q-major) | `d + 16q` | the **port's own spellings** (`v_addr`, P q-major, `C[dim][row]` drain) with the roles exchanged |

Candidate maps the checker classifies against, and the **derived** movement of each against the
transcription (cells out of 256 that would read differently — computed from the map pair and the pattern,
never hand-listed): `parity-swap` 256, `transposed` 240, `contig8-per-halfrow` 224 (the assumption that
disqualified D58's `m16_bf16frag`), `flat-run` 254, `adjacent-lane-swap` 256. Every candidate is visible;
none is an empty must-catch set.

## 3. The result

```
[as-transcribed]   P1 → all 256 cells match 'transposed'   (240 of 256 disagree with 'claimed-rdna3')
                   P3 → all 256 cells match 'transposed'   (240 of 256 disagree)
                   P4 → all 256 cells match 'transposed'   (240 of 256 disagree)
                   P2 → matches (as it must; it is symmetric and claims nothing)
[swapped-operands] P1, P3, P4 → all 256 cells match 'claimed-rdna3', exactly one candidate fits
```

Raw dump, first three lanes, P1 (`v = 16·(row of Q) + (row of K)`):

```
LANE  0 p1=0   32  64  96 128 160 192 224     →  rows 0,2,4,…,14 at kv 0
LANE  1 p1=1   33  65  97 129 161 193 225     →  rows 0,2,4,…,14 at kv 1
LANE 16 p1=16  48  80 112 144 176 208 240     →  rows 1,3,5,…,15 at kv 0
```

The two orders agree with each other from two directions (P1/P3/P4 each fit the transposed law in one
order and the claimed law in the other), which is what makes the answer a law and not a fixture artefact.
The three store-side injections (`lane-half-swap`, `reg-reverse`, `lane-rot`) are each caught
(256/256 cells disagree, and none is classified as the claimed map), so the comparator is not vacuous.

## 4. The fix, and what it costs

The fix is **two operand swaps**, not a redesign — and note that the *drain* the transcription describes
("lane t holds `C[t%16][2l + t/16]`") is exactly what you get **once the operands are in the exchanged
slots**, so the staging addresses, the epilogue shape and the fold's row grouping all survive:

* **score leg**: call `wmma(K_tile, Q_tile, acc)` — K in the first slot, Q in the second — and the score
  tile then reads `lane t → S[q = t%16][kv = 2l + t/16]`, as `wmma_fattn_score.cpp` assumes.
* **P·V leg**: call `wmma(P_operand, Vt_tile, acc)` — P first, the transposed V second — giving
  `lane t → O[d = t%16][q = 2l + t/16]`.
* **`p_to_operand` (the transcribed `get_half2`) still does the right exchange.** It pairs lane t with
  lane `t^16`, and under the measured law those two lanes hold the two column parities **of the same row
  of the second operand's axis**, so interleaving own/partner still yields the 16 contiguous values of one
  row — which is what the first operand slot wants. This is a *derivation*, not a re-derivation: see §5.
* Consequence for the fold (change (2)): one cross-lane op per row per tile is unaffected — the parity
  partner is still `t^16`.
* Consequence for claim (ii): the census counts instructions, and swapping two arguments moves none of
  them. **9.719 pipe-weighted slots/touch and the 4-blocks/CU answer carry over unchanged**, because the
  staging loops, the LDS budget and the instruction mix are untouched by the exchange.

## 5. What this probe does NOT witness — read before any port

1. **The `get_half2` C→B relabelling is still unwitnessed.** It moves values across lanes, so no
   single-warp pattern can test it; §4's claim about it is a derivation from the measured law and the
   pattern P4 result, not a measurement. The first corrected port must witness it directly — the same
   construction as P4, but with the operand produced from a C tile instead of from a host-built array.
   Until then the P·V leg rests on an argument.
2. **The two readings are indistinguishable and the report does not pretend otherwise.** "The C map is
   transposed" and "the operand slots are exchanged" predict identical tables for every possible pattern;
   the only observable is which staged buffer lands where. What a port needs is the law in §0, and it is
   stated in those role-free terms on purpose. The transcription of `mma.cuh:207` may be perfectly faithful
   to `mma.cuh` — what is falsified is the *composition* the handover built from `mma.cuh:195,207` and
   `mma.cuh:552,560`.
3. Nothing here prices anything. No `ms`, no burst, no occupancy query was run against the corrected form
   (the exchange moves no instruction, so `fattn_census.py` and `occupancy_fattn_probe.cpp` remain valid
   inputs, but they describe the *uncorrected* listing until someone re-runs them).
4. `SQ_LDS_BANK_CONFLICT` for the transposed-V store, and the realised-occupancy counter check, are still
   unrun (§5d's other two items).

## 6. Standing consequence

`prototype/issue_rate_probe/results.md` §5.1 already carries the rule "**no shipping WMMA kernel may
assume a hand-rolled fragment layout** — use the builtin's fragment abstraction, or pay the per-(lane,slot)
probe the tool names." This probe is that payment, made the cheap way (one launch, four patterns), and it
came back **red**: the hand-rolled layout in `wmma_fattn_score.cpp` is wrong in exactly the way the rule
existed to prevent, and `m16_fattn_score` would have shipped a transposed score matrix had the map not been
executed against. Any engine port must be built on §0's law, must keep `check_fattn_map.py` as its
admission's first gate, and must add the `get_half2` witness §5 names.
