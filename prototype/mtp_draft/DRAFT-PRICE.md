# DRAFT-PRICE — the MTP draft block, priced before it is measured

Pre-registered 2026-10-24 by `prototype/mtp_draft/price_block.py` (rerun that file; do not edit the
numbers below by hand — every one of them is arithmetic on a named input, and a band chosen after the
measurement is a rationalisation).

Target: the §09.2 draft block at its real shapes — 15 tensors, **40,905,728 B** (39.0107 MiB),
`prototype/mtp_gatezero/mtp_census.json`. One draft step is **14 kernel launches**, counted at the launch
sites in `td_modes.cpp` and asserted against the plan at run time (D2's "~18 launches" was never a
census; this one is).

## 1. The prediction, from bytes and published efficiencies

Inputs, each with the file that measured it: the LPDDR5x plateau is **237 GB/s**
(`devices/strix-halo-memory-bandwidth.md`); `prototype/proj_gemv/results.md` sustains **75 %** of it and
`prototype/mlp_gemv/results.md` **70 %**; per-launch overhead is **3.10 µs**
(`engine/performance-status.md`); a decode step is **10.933 ms**, which is `HEADBATCH-RESULT.md`'s own
arithmetic (`delta = 1.4847 ms = 0.1358 of a step`).

| stage | bytes | rate applied | predicted |
|---|---|---|---|
| fc + o_proj + q + k + v (the proj-class GEMVs) | 18,874,368 | 237 GB/s × 0.75 | **0.1062 ms** |
| gate + up + down (the MLP) | 22,020,096 | 237 GB/s × 0.70 | **0.1327 ms** |
| the 7 weight vectors of the norms | 11,264 (0.03 %) | negligible, read once | — |
| **block-only, one draft step** | **40,905,728** | — | **0.2389 ms** = 2.19 % of a step |
| launch floor: 14 × 3.10 µs | — | — | 0.0434 ms |
| **block + launch overhead** | — | — | **0.2823 ms** |

The head is deliberately absent: `prototype/mtp_headbatch/HEADBATCH-RESULT.md` measured R(M=2) = 1.0019,
so the ~485 MiB tied table is read once for the target and the draft together and does **not** scale with
draft depth. That is the whole reason the block is now the marginal cost of a round.

Whether the 0.0434 ms launch floor is *exposed* or hidden behind device work is an experiment, not an
assumption: `run_locked.sh`'s `empty` arm measures 14 no-op submissions in the same lock hold as the
block arms, so the overhead is attributed inside one window (a ruler re-taken in a different window is not
an A/B).

## 2. What draft depth k is worth, at the carried accept

`speedup(k) = a·(k+1) / (1 + k·c)` with `c` = block cost as a fraction of a step — the form
`HEADBATCH-RESULT.md` itself uses — and `a = 0.7329` carried on the engine's bits (its kill line 0.257).

| k | round cost (block+launch) | projected | note |
|---|---|---|---|
| 1 | 0.2823 ms | **1.4289×** | vs the **1.247×** already promised |
| 2 | 0.5646 ms | 2.0907× | optimistic: uses the k=1 accept for every drafted position |
| 4 | 1.1293 ms | 3.3214× | same caveat, squared — depth is priced by accept, not by this table |

**Sensitivity: +0.1 ms on a k=1 round moves the headline by −0.0126×.** Deep draft is cheap in bytes and
expensive in accepts; `k` is only worth funding after the engine's own accept-by-depth curve exists (D4).

Break-evens, arithmetic not taste:
- k=1 pays at all only if `a > 0.5`; at `a = 0.7329` the block could cost **5.093 ms** before a round
  stops being a win outright.
- the block could cost **1.918 ms** before the projection falls below the 1.247× the plan already
  promises.

## 3. PRE-REGISTERED BANDS — fixed before any millisecond is read

Judged on the k=1 **block-only** arm of `prototype/mtp_draft/run_locked.sh`, which is admitted by the full
numeric gate (21 observables against imported bands + 9 derived negative controls).

| band | line | meaning |
|---|---|---|
| **WIN** | ≤ **0.2986 ms** (1.25 × census) | slack for staging and any exposed launch cost; still ≥ 1.421× projected → fund graduation into `engine/` |
| **NARROW** | ≤ **0.5973 ms** (2.5 × census) | legal-but-slow: the block is above what its bytes demand, so the next spend is a staging/launch pass, not a deeper draft. No `k>1` funding. |
| **KILL** | > **1.9183 ms** | drops the projection under the promised 1.247×. And because the census says 0.239 ms, landing there is not a slow kernel — it means the design re-reads weights per step (a route/staging defect), so the fix is architectural and this line is where the work stops. |

Two additional pre-registered kills, both about the design rather than the kernel:

1. **Superlinearity in depth.** If `measured(k=2) / measured(k=1) > 2.35`, the second round is re-reading
   something the first should have kept (cache or routing defect). The census is per-step by construction,
   so a linear block is the expectation, not a hope.
2. **Admission level.** An arm admitted only structurally (k=4 has no pinned golden: `dec8`'s golden is two
   steps) may not be quoted as a headline, and no ms is printed for any arm whose admission was red —
   `run_locked.sh` prints `NO-TIMING: admission is red` and exits instead.

## 4. The measurement table — filled by the main agent, blank by design

`run_locked.sh` (one lock hold, blocking, two interleaved passes, canary-asserted) is the only authorised
source for these cells. Nothing in this file is a measurement.

| arm | admission | pass 1 | pass 2 | min | verdict vs band |
|---|---|---|---|---|---|
| block-k1 | numeric | — | — | — | — |
| block-k2 | numeric | — | — | — | ratio k2/k1 = — |
| block-k4 | structural only | — | — | — | not a headline |
| empty (14 no-op launches) | structural | — | — | — | exposed launch µs vs 3.10 |

## 5. What this rig can and cannot tell the ledger

- It can: the block's per-step byte cost, whether the block's launches are exposed, and whether depth is
  linear in cost.
- It cannot: the accept rate (D4's job), the verify body's cost (the 2-row batched target pass,
  `MTP-IMPL.md` §1, ~0.12 ms/round of rollback surface excluded here), or any number for the 4k headline
  step — the step used here is HEADBATCH's 10.933 ms, and the ledger's headline decode row is the 4,032-lane
  one, so a headline claim must be re-based on that step before it is quoted.
