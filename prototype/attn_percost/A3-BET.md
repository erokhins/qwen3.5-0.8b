# A3 — buy the third resident block: stage `s_q` in half-tiles so `attn_fattn_wave` reaches 3 blocks/CU

Pre-registered **before any code exists**, per this tree's rule. Funded by the main agent on
`prototype/attn_percost/A2-census.md`; approved as structural work on a shipping epilogue that §41 explicitly
declined to fund *for want of a price* — A2 is the price. Written 2026-10-19, tip `fbb94d9`.

## The premise, in one paragraph

A2 measured that the attention wave is **neither matrix-bound (5.56 % of the 52.63 TFLOP/s WMMA roof) nor
issue-bound (≤ 8.8 % of the slot roof)** at 6 wavefronts/CU = 3 wf/SIMD, which means 84–94 % of its wall is
stall with spare issue slots. The one scarce resource is resident wavefronts, and the binding limit on them is
**LDS**: 32,768 B/block → 2 blocks/CU. `s_q` (8,192 B/warp) is the epilogue's transpose staging on loan, and
E1 (§41) proved un-staging Q does not move it. The lever left is to **stage Q in 8-row half-tiles** instead of
whole tiles: `s_q` 3 × 4,096 = 12,288 B beside `s_kv`'s 8,192 B = **20,480 B/block ≤ 21,845 B → 3 blocks/CU = 9
wf/CU = 4.5 wf/SIMD**. Expected from the one elasticity this ledger has measured on this same axis (§31's Bet 1,
4 → 8 wf/CU = 1.647×, log-elasticity 0.72): 6 → 9 wf/CU predicts **≈ 1.36×** on the attention family.

## Constraint: this is a data-movement change, so bit-identity is the numerics gate

Unlike G4, nothing about the arithmetic may change — the same values reach the same WMMA tiles in the same
order, only the staging loop is split into halves. So **prefill logits and `STATEFP` must be bit-identical to
shipped** on the gate shapes. If they are not, the change moved arithmetic and is not A3; it is reported, not
re-banded. No new band, no widening, no `--ack-spill` (§10 rule 2, unchanged).

## Pre-admission, compile-side, BEFORE any device time is spent (Milestone 0)

1. `group_segment_fixed_size` ≤ **21,845 B** for the modified kernel, and it must be read off the note block,
   not asserted in prose (E1's lesson: `spill=0 ∧ VGPR≲190` was satisfied and the LDS leg silently failed).
2. `vgpr_spill_count = 0` and `private_segment_fixed_size = 0`. Non-negotiable: §33 is the precedent where the
   8-wf/CU shape spilled and died.
3. **The residency claim must be witnessed, not assumed.** `rocminfo` does not report the register file per CU
   on this part (A2 §6), and clang says `vgpr_count 178` where rocprofv3's `VGPR_Count` says 248 for the same
   kernel, so 3 blocks/CU is *not* arithmetically guaranteed even at 20,480 B. The decisive witness is on
   device: an occupancy counter (e.g. `SQ_LEVEL_WAVES_FILLING`, deduplicated per `(Dispatch_Id, Counter_Name)`
   per AGENTS' rocprofv3 trap row) must rise by ≈ 1.5× against shipped. **If residency does not rise, A3 is dead
   regardless of what the timing does** — that stops an LDS change from being shipped on a timing move that came
   from somewhere else.

## Lines, both denominators plus the pair that actually carries the verdict

Bases from the §44/§45/§46 windows: `attn` untraced family 133.00 ms, traced 155.71 ms, wave `wall_ms` 453.65
ms at 4,096 lanes; fitted `b` = 5.920e-06 ms/token²; wide rows 3,319.25 ms (16,384) and 10,448.13 ms (32,768).
**All dead lines are `base − 3 %`, the §37/§42 paired-window noise floor**, because a dead line inside the noise
floor is a coin toss and would make the verdict un-falsifiable:

| metric | base | live | dead |
|---|---|---|---|
| `attn` family, untraced ms/pass | 133.00 | **≤ 106** | **> 129** |
| `attn` phase, traced ms/pass | 155.71 | **≤ 125** | **> 151** |
| wave `wall_ms` at 4,096 | 453.65 | **≤ 428** | **> 440** |
| fitted `b` (ms/token²) | 5.920e-06 | **≤ 4.3e-06** | **≥ 5.62e-06** |
| 16,384-lane wave ms | 3,319.25 | **≤ 3,050** | **> 3,220** |
| 32,768-lane wave ms | 10,448.13 | **≤ 9,400** | **> 10,135** |

Every row must be measured in one window per side; **the wide pair and `wall_ms` are the ship gate** — the census's
own argument is that A3's value lives in `b`, so a win visible only at 4,096 is not the bet. Anything between a
row's live and dead line is unresolved and **not shipped** (G1's precedent holds). The worker's proposal asked
for live ≤ 435 / dead > 445 on `wall_ms`; I moved them, because 445 is within the ±3 % floor of a 453.65 base
and would have made "dead" unreachable and "live" reachable on noise.

## Harness requirements named, not dropped

- The fitted `b` needs our wide rows through `--ids-file` (§39's `MAX_ARG_STRLEN` trap silently drops the widest
  point otherwise); `tools/llama_phase_compare.sh` now does this and `tools/prefill_scaling_fit.py` refuses to
  fit through holes.
- In-window geometry canary: `prototype/attn_percost/a2_trace_rows.py` reading `LDS_Block_Size` (**20,480 new vs
  32,768 shipped**) with identical grid and launch counts. `trace_phase_split.py --geom` cannot serve here — it
  has no LDS column and keys on a kernel-name substring.
- Launch counts must be identical both sides; if they move, the A/B is not comparing one change.

## What A3 cannot see, and what follows if it dies

If residency rises and the timing does not move, then stall at 3 wf/SIMD was not the binding stall and the
attention gap is inside the wave's own dependency structure — at which point the remaining candidate is the one
A2's §5 leaves standing and nothing here can price: the useful-vs-issued ratio of the hand-written WMMA body
itself (17 sites, vs llama's 2,592, at a tile shape no one has compared). That is a rewrite conversation, not a
tweak, and it comes to me with numbers rather than adjectives.
