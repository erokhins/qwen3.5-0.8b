# D3 — main agent's measurement window (2026-10-21)

Taken by the main agent, blocking, zero subagents active, one `tools/gpu_lock.sh` hold
(`d3-timing`), util 23 % at acquire / 53 % at release, discounted per `AGENTS.md`. Every arm is
`--mode burst` (warm-up + wall-clock/N over one sync), interleaved so clock drift shows as
disagreement between reps. No number here is a ledger row: these are prototype-plan figures, not
the engine's decode step.

## Canary, read before any ms

```
_Z9attn_core8CoreArgs         lds= 38944 B  next_free_vgpr= 53
_Z14attn_core_lean8CoreArgs   lds=  6176 B  next_free_vgpr= 45
attn_core      38944 B ->  1 blocks/CU by LDS -> 1 block(s)/CU resident
attn_core_lean  6176 B -> 10 blocks/CU by LDS -> 4 block(s)/CU resident
variant is the frozen kernel: 151 lines, 5 permitted differences
RESULT d3gate cases=21 bitdiff_stages=0 pinned_off_floor=8 negative_failures=0
```

Two distinct code objects, so the A/B is not one kernel printed twice. `cases=21` with 0 bit
differences across 126 stage comparisons and 0 negative-control failures — the candidate is
byte-identical to the frozen core on 1…4,104 lanes including the 511/512/513/514/515 boundary.
Device facts read from KFD (`lds_size_in_kb 64`, `simd_per_cu 2`), not from HIP's halved report.

## A/B #1 — one variable (the core), wide width, traffic held identical

`wide4096f.decode7.L03`: 4,104 lanes, split 9, 72 blocks. `ms_per_plan` over three reps each.

| core | reps (ms) | median | vs ship |
|---|---|---|---|
| `ship` | 0.3219 / 0.3190 / 0.3201 | **0.3201** | — |
| `lean` | 0.1868 / 0.1666 / 0.1753 | **0.1753** | **−45.2 %** |

Reps agree within 1.5 % within an arm and the arms never overlap, so the 45 % is not drift. The
`kv_demanded_gbs` column the rig prints is a demanded-bytes figure, not a DRAM rate: it rises 210 →
384 GB/s *because the kernel got faster at identical traffic*, which is exactly the reading `AGENTS.md`
warns about — it is above the 237 GB/s plateau, so demanded bytes were never the wall.

## A/B #2 — the population lever (lane split), `lean` held

| arm | ms | vs ship |
|---|---|---|
| `lean` split 9 (default) | 0.1839 | −42.6 % |
| **`lean` split 18** | **0.1073** | **−66.5 %** |
| `lean` split 36 | 0.1095 | −65.8 % |

18 is better than 36, so the population axis saturates at 144 blocks (3.6 per CU) rather than
improving without bound — consistent with §1's rounds model and against "more blocks always better".

## A/B #3 — the same pair at the short widths

Nine cases, 18…511 lanes, 8 blocks per launch on 40 CUs: `ship` 0.3107 ms, `lean` 0.2893 ms,
**−6.9 %**. Nearly flat, which is what §4.3 predicted and is a *pass*, not a disappointment: at 8
blocks/CU one round at either residency, so neither lever can bind. It also means the 128-lane
`attn` line (0.496 ms) is not the tile walk and will not be moved by this route.

## What X comes out as, and the projection it implies

§4 defined X as the share of the engine's 3.023 ms `attn` line that is the core's tile walk, and gave
`X = 2 × (3.023 − lean)`. That form assumed the A/B would be taken in engine units; it was taken in
prototype units, so the honest statement is the ratio instead: the core body of one layer at the
route's width costs 0.3201 ms isolated, and six layers' worth is 1.92 ms of the 3.023 ms line, i.e.
**X ≈ 0.64** — above §4.2's 0.65 threshold for the pair to clear WIN, and inside the band where
"`lean` alone" is a measurement rather than a candidate.

Projected engine prize, stated as arithmetic so the engine A/B can falsify it: `lean` alone saves
0.452 × 1.92 ≈ 0.87 ms/step at 4k → 12.68 → ≈11.81 ms/token ≈ **84.7 tok/s at 4,032 lanes**;
`lean` + split 18 saves ≈1.28 ms → ≈11.40 ms ≈ **87.7 tok/s**. Both are projections from an isolated
per-layer body and will not survive intact — the in-situ family has prep, combine and launch
structure around the core — so they are the prediction the engine window is obliged to contradict or
confirm, not a result.

## Decisions taken on this evidence

1. **Port `lean` to the engine.** Byte-identical, five permitted spelling differences, no band in
   play, and it raises residency from 1 to 4 blocks/CU on the shipping path. Complication the
   handoff did not name: `attn_core` is *shared* — `engine/kernels_attn.cpp` includes
   `prototype/attn_decode/attn_decode_kernels.h`, the same header `prototype/attn_decode/gate.py`
   proves provenance against — so the port edits a file two gates check and must move both in one
   commit or `--check-source` will fail by construction.
2. **Hold the split lever.** `--split 18` is a geometry change, not a spelling: it re-windows the
   streaming merge, and §5's open item (two instruments disagreeing about the >`MAXSC` pins) is
   unresolved. It is worth ~0.4 ms more than `lean` alone and it needs the numerics ruling first.
3. **Do not touch the 128-lane line here.** Confirmed out of reach of both levers.
