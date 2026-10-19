# A4-L1 verdict: DEAD — a second accumulator chain moves the attention family by 1.67 %, inside the paired floor

Measured by the main agent, 2026-10-19 12:59–13:05Z, one `gpu_lock.sh` hold (`a4l1-ab`), `util` 26 % at acquire,
`sclk` 1420 MHz at start, zero subagents active, blocking, `LC_ALL=C`. Harness
`prototype/attn_percost/run_ab_a4l1.sh` at `e1d932c`, baseline pinned to the pre-knob `47d390e`, logs in
`tmp/attn_percost/a4l1ab/`. Lines are A4-BET.md's ratios applied to the in-window base (see the harness header
for why, and for the fabricated "re-based" set this replaces).

## The verdict

| row | base (in-window) | `two` | move | live line | dead line | colour |
|---|---|---|---|---|---|---|
| `attn` family, untraced ms/pass | 136.2548 | 133.9760 | **−1.67 %** | ≤ 121.91 | > 132.15 | **DEAD** |
| `attention` phase, traced ms/pass | 150.16 | 149.74 | −0.28 % | ≤ 135.01 | > 145.61 | **DEAD** |
| wave `wall_ms` @4,096 | 446.2689 | 447.4837 | **+0.27 %** | ≤ 421.05 | > 432.84 | **DEAD** (ship gate) |
| wave ms @16,384 | 3,271.73 | 3,271.33 | −0.01 % | ≤ 3,006.39 | > 3,173.91 | **DEAD** |
| wave ms @32,768 | 10,540.20 | 10,496.40 | −0.42 % | ≤ 9,483.02 | > 10,223.99 | **DEAD** |

Externality (sum of `two − base` over every family other than `attn`, informational, no pre-registered line):
**+2.18 / −13.23 / −43.62 ms** at 4,096 / 16,384 / 32,768, inside its ±3 % band. **VERDICT: DEAD.** The default
stays `single`; `two` is not shipped, and the knob stays in the tree as an inert, documented instrument.

## What was established before the timing, and what the timing therefore means

- **The chain was real.** Gate 0 read the shipped roll as one `v_wmma` site whose destination tuple is also its
  accumulator operand, in a rolled 16-trip loop with no independent work in the body.
- **The fix was real and it ran.** Canary B read `178/0/0/32768` for base and for the new default, and
  `200/0/0/32768` for `attn_fattn_wave_two`, off the linked binaries in this window; canary C read the launched
  kernel's own `vgpr` per side; numerics leg 2 showed `two` moves `STATEFP`, so the second chain is executing,
  not being folded away.
- **The window was paired.** NULL: base vs `single` differ by **−0.06 %** on the `attn` family (tolerance 2 %),
  and base == `single` bit-for-bit on every prefill logit row and every `STATEFP` line outside the hold.
- Admission: `engine/gate.py --fixture wide640f --attn-fattn-roll two` — **289 PASS / 0 FAIL**, worst per-row
  1-cos 2.406e-04 against the 2.75e-03 line, nothing widened (worker run at this same engine tree, cited rather
  than re-made).

**So the hypothesis was right and the hypothesis was not the bottleneck.** A 16-long dependent WMMA chain is
there — and cutting it in half, free of spill, buys 1.67 % of one family and nothing at all at the wave. Together
with A2 (matrix pipe 5.6 % busy, issue roof ≤ 8.8 % used) and A3 (LDS floor 32,768 B, so residency cannot be
bought by staging), the residue is neither the matrix pipes, nor the issue slots, nor the accumulator dependency,
nor the split path. What is left standing is the **shape**: 16-row M tiles, 3 warps/block, 2 blocks/CU, 6
wavefronts/CU, and a per-round `LDS → WMMA → fold → rescale` sequence that no re-association shortens.

## Why L2 and L3 are not funded against these lines (arithmetic, not taste)

- **L2** (batch the accumulator rescale over 4 rounds, llama's `nbatch_fa=64`): its entire budget is **128 of
  ≥ 1,654 slots in the round = 7.7 %**, and the live line demands **−10.5 % of the family row**. Even deleting
  the rescale outright cannot reach the bar this bet pre-registered. Below the bar, so not funded.
- **L3** (split/combine scratch round-trip): A2 sized it at **10–13 % of the excess quadratic**, which is ~8 ms
  of a 136 ms `attn` row and ~1.8 % of the wave at 4,096 — it fails the 4,096-lane lines by construction even
  if it is delivered perfectly. It stays on the books as a wide-context item, not as an A4.

## What this cannot see

It cannot say what the stall *is*, only that it is not the accumulator dependency. It cannot separate "6 wf/CU is
too few" from "the round's dependent sequence is too long", because A3 closed the only cheap route to more blocks
and this bet touched only one link of the sequence. And it cannot price the geometry rewrite — the only route left
with room — without the LDS and register arithmetic that A3-result.md shows is partly unanswerable on this part
(no register file size reported, clang 178 vs rocprofv3 248 for the same kernel). That gap is what the next
proposal has to confront before any kernel is written.
