# A12 — the wide-shape attention kernel, funded under a relaxed gate policy (2026-10-20)

Funded by the user in these words: *"for other scenarios these are good protections, but for this one we
should reject them, i.e. let's accept register spilling + low occupancy + f16 precision for this, and
implement it."* This file is the record of what was relaxed, for whom, and what would make the relaxation
wrong. **A10 stays recorded as DO NOT FUND** (`A11-attribution.md`); A12 is not a retry of A10's arithmetic, it
is the one premise A11 could not bound — llama's ~1.9× attention rate — attacked at the shape the ledger says
produces it.

## 1. What is relaxed, and where the protections stay

| policy | status for the SHIPPING path | status for the A12 wide kernel | witness that the shipping path is untouched |
|---|---|---|---|
| spill gate (`fattn_ship_census.py` refuses any spill) | **unchanged** | a **budget, not a taboo**: ≤ 1,024 B/thread private scratch allowed, and the paired window must show the spilled traffic is smaller than the traffic the wider shape removes (llama's traced figure is 900 B/thread, `A2-census.md:78`) | the census keeps failing on every kernel it fails on today; A12's kernel is a separate entry with its own bound |
| occupancy floor (3 blocks/CU, 9 wf/CU) | **unchanged** | **1 block/CU at 8 waves is allowed** (= 8 wf/CU, which is *llama's* residency, not a new low) | ledger §55 measured the elasticity in both directions, so the cost of giving residency up is a number we already carry (3.4–4.4 % per third of residency) |
| numerics (f32 accumulator, bf16-calibrated bands) | **unchanged** | **f16 accumulator permitted** — already built and inert as `--attn-fattn-pvf16` (`A10-PVF16.md`); argmax stable and no token flipped on the probe fixture, no band widened | default is byte-identical by `cmp` on the logits dumps, already witnessed |
| no new band, no loosening of a check that catches a real bug | **unchanged and binding here too** | a wide shape whose *outputs* are wrong is caught by the same §10.1 aggregates + must-catch injections; nothing in this relaxation touches them | the must-catch set is geometry-derived and stays in force for the wide kernel too |

What is *not* relaxed: the default trio (wide kernel is never the default until it wins a paired window), the
main-agent-owns-measurements rule, and the **ABORT LINE: 6.5 TFLOP/s at 32k** from the A10 kickoff.

## 2. The BET (pre-registered before anything is built)

**Bet:** a 128-query-row × 64-position-staged block (8 waves, 256 threads, f16 accumulator, Q converted once
per block and held in registers, spill within budget) clears **6.5 TFLOP/s at 32,768 lanes**, i.e. ≈ +38 %
over today's kernel-only 4.714.

**Falsifier:** a paired window at 32,768 lanes (via `--ids-file`, since 32k ids in one argv field trips
`MAX_ARG_STRLEN` — AGENTS.md) against a pinned pre-change sha, with the behavioural canary read first. If the
wide kernel does not reach the abort line, **A12 is recorded dead the same way A10 was**, and the relaxed
policies go back to being shipping-path protections with nothing to show for them. The ≥3 % falsifier rule from
the kickoff still applies on the deciding row.

**The premise being tested, stated so it can be killed cheaply:** A11 found *nothing saturated* (issue 18 %,
LDS 5–19 %, no memory pipe full, 65–74 % waiting). If A12's wide shape moves the rate at all, it is because
**LLC/L2 latency on re-fetched operands** is the real wall even though the LSU pipe is idle; if it does not move
the rate, the latency-bound reading wins and no shape change will help. That is the single number this bet
buys, and it is worth running even though the prior is not favourable.

## 3. Sequencing (cheapest first, no engine plumbing until it wins)

1. **Compile-only ladder, re-read with the corrected instrument.** `a11_read_vgpr.py` reads
   `.amdhsa_next_free_vgpr` (the reservation the driver uses); re-price llama's RDNA3 kernel and our wide-shape
   skeleton with it, because `A2-census.md`'s `256 VGPR / 900 B scratch` predates the note-block discovery.
2. **Standalone prototype in `prototype/attn_wide/`** — no `engine/` plumbing, no knobs, no gate changes. Same
   discipline as `prototype/attn_decode`: correctness against the pinned reference first, then one timing.
3. **Device window (main agent only)** at 32,768 lanes, pinned pre-change sha, canary, `--ids-file`.
4. Only a win graduates: knob default-inert, one-commit plumbing (launcher mirror + `FATTN_ROWS_PER_BLOCK`,
   launch bill, derived gate case lists, default trio, inertness witness).

## 4. Honest priors, written down before spending

- The reachable mass A11 could bound (Q channel ≤12 %, round-fixed work 5.2 %) is **well short** of +38 %. The
  case for A12 rests on the unbounded part: per-byte operand re-use, which A11 measured only as *pipe
  utilisation*, not as *latency cost*.
- Our shipping kernel today reserves 241 of 256 registers with zero spill, so the wide shape will spill or
  will not fit; that is precisely what the relaxation is for, and it is also why an earlier attempt at this
  shape was source-killed rather than built.
- If A12 lands between "faster" and "the abort line", the kickoff's rule applies: **DISPLACED, not shipped**,
  and reported as a trade by width rather than averaged away.

## 5. The deciding window, run 2026-10-20 (main agent, `run_a12_window.sh`, one lock, blocking, zero subagents)

32,768 lanes, best-of-3, reps interleaved, canary asserted before any timing, correctness pre-flight green
(coverage `COVERAGE OK`; 19 derived-edge rows, worst cos 0.9999255, argmax moved 2/152, inside the calibrated
band 0.9995 / 2e-02). Per-rep ranges are **disjoint**: wide 734.82 / 738.70 / 744.85 ms versus
`attn_fattn_wave_u8` 919.06 / 919.06 / 933.19 ms.

| form | best ms | TFLOP/s | blocks/CU | wf/CU | LDS | regs / scratch |
|---|---|---|---|---|---|---|
| `attn_wide_wave` (128 rows, staged K/V, f16 acc, Q in registers) | **734.82** | **5.985** | 2 | 16 | 32,768 | 256 / 156 B |
| `attn_fattn_wave_u8` (shipping) | 919.06 | 4.786 | 3 | 9 | 20,480 | 241 / 0 B |

**Verdict against the pre-registered line: NOT CLEARED.** The abort line is **6.5 TFLOP/s** and the wide shape
reads **5.985** — a 7.8 % miss. The A10/A12 thesis ("geometry closes the gap to llama's attention rate") is
therefore recorded dead at the widest shape the relaxed policy admits, which is the outcome the BET said to
record rather than renegotiate.

**The relaxation nonetheless bought a real win:** **+25.1 %** kernel throughput at the deciding width
(1.2507× wall), at 16 wf/CU and 2 blocks/CU where the shipping form has 9. Operand re-use moved the rate even
though A11 found no saturated pipe — which is the mechanistic answer the BET was bought to get, and it says the
wall is latency on re-fetched operands, not LSU issue. Honest caveats, all printed by the rig: kernel-only on
both sides (`attn_prep`/`attn_combine`/append unpriced, which flatters the baseline); the shipping arm is inert
on 512 of 32,768 rows; and numerics admission remains `a12_wide_mirror.py`'s and `gate.py`'s, not this
window's. The +25 % is a shipping decision (route by width) that the abort line does not authorise by itself.
