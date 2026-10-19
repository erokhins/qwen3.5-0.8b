# A4 — L1 funded, falsification-first: is the shipped `#pragma unroll 1` score roll a latency chain or just an issue dependency?

Pre-registered before code, 2026-10-19, tip `a05bb97`. Source: `prototype/attn_percost/A4-CANDIDATES.md` §2
L1. Funded by the main agent; the worker's inventory did the pricing, this file sets the gates and the lines.

## The claim under test

`engine/attn_fattn_kernels.h:577-578` rolls all 16 k-tiles of the score leg into **one** accumulator under
`#pragma unroll 1` — a 16-long **dependent** WMMA chain per round per warp. A2 measured the matrix pipe at
**5.6 % busy** and the issue roof at ≤ 8.8 % used, so the round is neither; what it can be is
**latency-bound on that chain**. L1 gives the roll a second accumulator (`acc0`/`acc1`, summed once after the
roll) so two chains progress at once. llama does the aggressive version of this on purpose (`nbatch_fa = 64`,
`Q_in_reg`, 256 VGPR and **900 B/thread of deliberate scratch** at occupancy 1; 2,592 `v_wmma` sites against
our 17) — it pays in registers and spill to buy independent matrix work.

## Gate 0 — the cheap falsifier, and it runs first (CPU only)

If LLVM already software-pipelines that roll, the lever is **empty** and nothing downstream is funded. Decide it
from the emitted code, not from the source's intent: reuse `prototype/gdn_rec_isa/token_bill.py`'s loop-annotation
method (its AGENTS-recorded traps are exactly why: function labels, `.LBB` "in Loop" annotations, and
`v_fmac_f32_e32` rather than `v_fma_f32`) to ask whether the 16 WMMA sites in the round body are **chained on the
same accumulator register** or interleaved across two or more. **A pass at this gate is a prerequisite, not a
suggestion** — the last two bets on this loop (§27 explained by §28, `attn_valu_slim`'s Lever 0) died from
explaining something that had already been measured wrong.

## Gate 1 — compile-and-read, before any device time

Build the two-accumulator variant behind a compile-time switch and read the note block with
`prototype/attn_wmma/fattn_ship_census.py` + `fattn_residency_read.py`:

- `vgpr_spill_count = 0` **and** `private_segment_fixed_size = 0`. §33 is the precedent that prices this:
  spilled-at-8 lost to spill-free-at-6. If the second accumulator spills, L1 is dead and says so.
- Report `vgpr_count` before/after against the shipped 178, with the honest caveat that the register file per CU
  is unreported on this part (A3-result §2b) — the note block is the admission, no occupancy arithmetic.
- LDS must not move (it should not: the accumulator is registers).

## Numerics — flagged, not assumed

The 16-tile dot is a fixed ascending F32 sum today (`:577`'s whole point), so two accumulators **re-associate
it** and `o` moves. Judged by the existing §10.1 aggregates (`cos ≥ 0.9995`, `meanrel ≤ 5 %`) through
`engine/gate.py`, which the delegated-format rule already covers: an existing band, no widening, no
`--ack-spill`. Bit-identity is explicitly **not** the gate here, unlike A3 — and that is a decision stated in
advance, not discovered afterwards.

## Gate 2 — the device A/B, funded only if gates 0 and 1 pass

Bases: `attn` untraced family **133.00 ms**, traced **155.71 ms**, wave `wall_ms` **453.65 ms** at 4,096 lanes;
wide rows **3,319.25 ms** (16,384) and **10,448.13 ms** (32,768) from §45. Dead lines at **base − 3 %**
(§37's paired-window floor), the same convention as A3 and for the same reason — a dead line inside the noise
floor is unfalsifiable. L1 has no measured elasticity behind it (A3 at least had §31's), so the live line is set
deliberately, not hopefully: this must be worth a real slice of the wave, not a rounding.

| metric | base | live | dead |
|---|---|---|---|
| `attn` family, untraced ms/pass | 133.00 | **≤ 119** (−10 %) | **> 129** |
| `attn` phase, traced ms/pass | 155.71 | **≤ 140** | **> 151** |
| wave `wall_ms` at 4,096 | 453.65 | **≤ 428** | **> 440** |
| 16,384-lane wave ms | 3,319.25 | **≤ 3,050** | **> 3,220** |
| 32,768-lane wave ms | 10,448.13 | **≤ 9,400** | **> 10,135** |

Both `wall_ms` and the wide pair are ship gates: attention is 33 % of the wave at 4,096 and a bigger share
every doubling, so a change that owns the round's latency must show up where the quadratic lives. Anything
between a live and a dead line is **unresolved and not shipped** (G1's precedent). Harness requirements: the
wide legs through `--ids-file` (§39), `launches` identical both sides, externality printed, verdict parser
colour-tested, `SUMMARIZE=1`.

## What is closed, so this list is not re-litigated

Residency (A3, on LDS arithmetic **and** unresolvable RF facts), instruction class (A2: both sides are
matrix-class; ours 5.56 % of roof, llama's 15.2 %), launch count/coverage (§41: one integer explains them,
98.4 % of lane-work), and the split path as a major term (A2 §4: ~10–13 % of the excess quadratic — real, and
L3-sized, but not the 1.6×). If Gate 0 kills L1 and L2/L3 are priced as small, then the honest conclusion is
that the shipping WMMA body needs to be rewritten against llama's tile geometry, and that comes to me as a costed
rewrite proposal — not as a tweak smuggled in under a familiar flag name.
