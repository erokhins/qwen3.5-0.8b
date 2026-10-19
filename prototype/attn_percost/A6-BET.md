# A6 — the residency thesis gets a falsifier before its payoff: G0 (7 warps) first, G1 (20,480 B, 9 wf/CU) only if it moves

Pre-registered before code, 2026-10-19, tip `7ff27bf`. Source: `prototype/attn_percost/A5-GEOMETRY.md` §0/§3.1/§3.2.
The worker recommended one candidate and asked for a gate; the gate and the order are mine.

## Why there is a falsifier at all

A5 §2 argues attention's stall is the number of **resident wavefronts** between two stalls — the only quantity on
this engine with a measured elasticity (§31: 4→8 wf/CU = 1.647×, log-elasticity 0.72). G1 is the payoff but it is
also the surgery: Q stops being staged, the epilogue's transpose goes from 2 passes to 4, and ~130 lines of the
round's staging change. G0 buys **7 wf/CU from 6** by changing the warp count and nothing else, predicting
**family −10.5 %, wave −3.2 %**. If the wave does not care about a 17 % residency increase, then it will not care
about 9 wf/CU either, and G1 must not be built. That is the whole point: **spend one small kernel to test the
thesis before spending the big one.**

## Lines and denominators (A4-BET's ratios, bases taken in-window, dead at base − 3 %)

| row | live | dead | note |
|---|---|---|---|
| `attn` family ms/pass | **≤ 0.8947 × base** | **> 0.9699 × base** | the falsifier row: G0 predicts −10.5 % here |
| `attention` traced ms/pass | ≤ 0.8991 × base | > 0.9697 × base | must agree in direction with the family row |
| wave `wall_ms` @4,096 | ≤ 0.9435 × base | > 0.9699 × base | ship gate |
| 16,384-lane wave ms | ≤ 0.9189 × base | > 0.9701 × base | ship gate |
| 32,768-lane wave ms | ≤ 0.8997 × base | > 0.9700 × base | ship gate |

**G0's falsifier rule, stated now.** The `attn` **family** row decides G0, because G0's predicted wave move
(−3.2 %) sits *inside* the paired floor and therefore cannot adjudicate anything — a row that cannot move by more
than the noise is not a gate, and pretending otherwise is how §37's floor got written. So:

- family improves **≥ 10 %** → residency thesis holds → **fund G1**.
- family improvement **< 3 %** (inside the floor) → **thesis falsified, G1 is not funded**, and attention's
  remaining route is the llama-shaped rewrite priced in A5 §5, which comes to the user as a scope decision, not
  as a tweak.
- between 3 % and 10 % → **UNRESOLVED**: G1 is funded only if the traced `attention` row independently improves by
  ≥ 5 %, and that condition is recorded here rather than discovered later.

**G1's own bar, pre-registered now so the payoff cannot be re-sized after the fact:** same five rows, same ratios.
A5 predicts −25.3 % family / −7.7 % wave, which clears every line with margin; if it lands inside them, it is a
win, and if it lands between live and dead it is unresolved and not shipped (G1's precedent, §47).

## Numerics — declared in advance, because the two halves differ

- **G0 (warp count) re-splits the round**: `for (kv = w; kv < FATTN_MLANES; kv += nw)` means each warp accumulates
  different lanes and the cross-warp fold sums in a different order. **Bit-identity is not expected and is not
  required**; the existing §10.1 aggregates govern (cos ≥ 0.9995, meanrel ≤ 5 %), same delegated-format rule A4
  used, nothing widened.
- **G1 (Q un-staged + 8-row epilogue) must be bit-identical.** Same 16 source bytes per tile, same
  `bf16 → f32 → f16 RN` conversion, same clamp, same order, and the epilogue change moves staging addresses only;
  row round-boundaries do not move because every covered row of a body shares `blo = blockIdx.y · span`
  (`attn_fattn_kernels.h:360-362`). A5 calls this "an argument, not a measurement" and it is right to: if logits
  or `STATEFP` move at all, the change is not what this bet thinks it is and it stops.

## Milestones that gate device time

- **G0-M0 (compile-and-read).** 7-warp build: `group_segment_fixed_size == 65,536`, `vgpr_spill_count == 0`,
  `private_segment_fixed_size == 0`, derived blocks/CU == 1 → 7 wf/CU. Read off the note block, never asserted in
  prose. Launch counts and grid must otherwise match shipped except `blockDim`.
- **G0-M0b (the register-file answer without a timing).** A `__launch_bounds__(96, N)` occupancy ladder
  (`prototype/attn_wmma/build_occupancy_fattn.sh` + `occupancy_fattn_probe.cpp`, driver query only, no kernel
  launched). This is the experiment A3-result §4 named and nobody ran; it is also G1's gate: **if the driver
  refuses 3 blocks at ≤ 190 VGPR, G1 is dead without any device A/B.**
- **G1-M0 (compile-and-read).** `group_segment_fixed_size == 20,480`, spill 0, `vgpr_count ≤ 190`.
- **G1-M1 (bit-identity device gate)** on ≥ 2 shapes including a ragged tail, logits + `STATEFP` byte-identical to
  the shipped default, plus default-inertness of whatever knob carries it.

## Standing harness requirements (both A/Bs)

Baseline pinned to a pre-change sha with `BASEDIR` derived from it (not `HEAD` — A4's defect); an in-window
behavioural canary read from the **built** binaries (note block + `wf_per_cu=`, since `engine/build.sh` is not
bit-reproducible); numerics legs before the hold; wide legs through `--ids-file`; `LC_ALL=C`; every parse asserted
non-empty; the verdict parser colour-tested against fixtures **generated from the same ratio constants**; and the
externality printed as information only. Line sets live in this file; a harness that overrides them is wrong, per
§47's harness lesson.
