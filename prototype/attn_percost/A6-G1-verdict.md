# A6-G1 verdict: **LIVE on all five rows, byte-identical, shipped** — attention −22.09 %, wave −5.90 % at 4,096 lanes, and **−18.61 % at 32,768**

Window: main agent, 2026-10-19 15:34Z, one `gpu_lock.sh` hold (`a6g1-ab`), zero subagents, blocking, `LC_ALL=C`,
`util` 29 % at acquire, `sclk` 1561 MHz at start. Harness `prototype/attn_percost/run_ab_a6g1.sh`, three sides in
one window (`base` = worktree pinned at `ed658fe`, `stage` = new binary with the knob unset, `u8` = same binary +
`unstaged8`). Logs `tmp/attn_percost/a6g1ab/`. Lines: `prototype/attn_percost/A6-BET.md`.

| row | base | `unstaged8` | ratio | line | colour |
|---|---|---|---|---|---|
| `attn` family, untraced | 135.3870 ms | **105.4753** | 0.7791 | live ≤ 0.8947, dead > 0.9699 | **LIVE** |
| `attention` phase, traced | 147.07 | 121.27 | 0.8246 | live ≤ 0.8991 | **LIVE** (direction agrees) |
| wave `wall_ms` @4,096 | 451.8840 | 425.2243 | 0.9410 | live ≤ 0.9435 | **LIVE** (ship gate) |
| wave ms @16,384 | 3,292.48 | 2,824.39 | 0.8578 | live ≤ 0.9189 | **LIVE** (ship gate) |
| wave ms @32,768 | 10,491.30 | 8,538.82 | 0.8139 | live ≤ 0.8997 | **LIVE** (ship gate) |

**Pairing and canaries.** NULL (new default vs base binary): family **−0.41 %**, wall @4,096 +0.22 %, 16k −0.23 %,
32k +0.01 % → paired. Rep spreads: base 1.56 %, stage 0.94 %, u8 0.77 % (tolerance 3 %). Canary A read the note
blocks off the **linked** binaries in the window — `178 0 0 32768` base, `178 0 0 32768` new default,
`185 0 0 20480` for `unstaged8`; the banner read `q=unstaged8 … lds=20480 wf_per_cu=9` on the rows that ran it.

**Numerics: no negotiation, because nothing moved.** `base` vs default and `base` vs `unstaged8` are **byte-identical**
on 640, 1,111 (ragged tail) and 2,000 ids — every dumped logit file and all 44 `STATEFP` rows
(`tmp/attn_percost/a6g1ab/bits.log`, and independently green in the worker's legs). After shipping the default,
`engine/gate.py --fixture wide640f` on the committed tree: **ENGINE GATE PASS**, nothing widened.

## What shipped and what it is worth

Default flipped `stage` → `unstaged8` in `engine/decode_run.cpp`, `engine/decode_step.h` and `engine/gate.py`
together (the G4 pattern), so the shipping engine now runs **Q out of LDS + an 8-row epilogue = 20,480 B/block =
3 blocks/CU = 9 wavefronts/CU** at 185 VGPR with no spill.

* **Headline (own window, `util` 5 % at acquire, `sclk` 845 MHz — clock-discounted, so trust the pair):**
  `RESULT ms_total` best-of-5 **512.908 ms** vs base best-of-3 **531.285 ms** in the same window = **+3.46 %**,
  i.e. **7,986 prompt tok/s** at 4,096 lanes against §44's 7,655.
* **Long context, from the paired A/B above (the same denominator §45's ladder used):** 16,384 lanes
  2,824.39 ms → **5,801 tok/s** (§45: 4,937, **+17.5 %**); 32,768 lanes 8,538.82 ms → **3,838 tok/s**
  (§45: 3,136, **+22.4 %**). The win grows with context exactly as §46's quadratic-term argument predicted,
  because residency is what the quadratic term is buying.
* **Decode did not move.** Paired 64-step A/B (`p64` fixture): base best 7.438 vs new best 7.545 ms/step
  (**+1.4 %**, inside the ±3 % floor). Honest footnote: all three pairs ran the same sign, so it is noted rather
  than claimed; and the absolute here is *not* §44's 1.584 ms/step, which came from the wave-timed path — a
  different denominator, comparable only within its own window.
* **Externality +2.83 / +20.08 / +5.05 ms.** Better than G0's +74.72 at 16k but still positive: a third resident
  block costs the neighbourhood something. Informational, decides nothing, and worth a look if attention is
  touched again.

## Why this one landed when four earlier routes did not

A2 said the wave runs at 5.56 % of the matrix roof and ≤ 8.8 % of the issue roof; A3 said LDS could not be cut
without paying in the round's dependent chain or its LSU feed; A4-L1 said the accumulator dependency is worth
1.67 %. G1 buys residency a *third* way — take Q out of LDS (E1's port) and halve the epilogue's loaned transpose
buffer — which A3's kill did not cover because it never proposed splitting a 16-row round. G0 then proved, on one
axis and for the price of a constant, that this family responds to residency at about §31's predicted elasticity.
The sequence is the point: three negative results bought the discriminating measurement, and the falsifier ran
before the surgery so the surgery could be cancelled cheaply if it was wrong.
