# Generation optimization queue — every avenue, sized and with the test that settles it

Companion to `standing-rules-2026-10-21.md` and `engine/generation-limits.md`. Rule 2 of the standing rules
says do not stop until every idea is measured; this file is the population, so "I think we are done" can be
checked against a list instead of against memory. Update the STATUS column, never delete a row.

Measured at 4,032 lanes; step is **10.9357 ms = 91.4 tok/s** at the time of writing (`775e376`, 224
launches/step after both folds). A prize is only worth an engine change if it clears ~0.30 ms (2.7 %, the
bar both of this week's wins cleared) — anything smaller needs to be nearly free to implement.

## Tier facts this queue depends on

- Copy plateau **~237–240 GB/s**. Fast tier: **~32 MiB at 876.7 GB/s, cliff to 239.4 by 64 MiB**
  (`devices/strix-halo-memory-bandwidth.md` §"The size–bandwidth curve"); per-GPC L2 is only **2 MiB**, so
  the warm tier everyone means here is MALL, not L2.
- Per-token bytes: 1532.3 MB read (of which `lm_head` 509.6, `mlp` 528.8, `proj` 467.6 MiB).
- Latency-bound families (the only place hiding latency buys anything): `attn` at 13 % of its byte floor
  (1.6121 ms vs 0.209), `gdn` at 30 % (0.6458 vs 0.190). Families at or near the wall cannot be helped by
  prefetch at all — there is no idle fetch capacity in them.

| # | idea | mechanism | est. prize | STATUS | the test that settles it |
|---|---|---|---|---|---|
| P1 | **Cross-phase prefetch into the 32 MiB tier** (user, 2026-10-21) | Issue explicit prefetch of phase *i+1*'s inputs while phase *i* runs, so the compute finds bytes in the 877 GB/s tier instead of cold DRAM. Only works where bandwidth is idle: `attn` uses ~31 GB/s of 237 during 1.61 ms (~330 MB of unused fetch time), `gdn` similar, and **the 0.7163 ms host tail leaves the GPU idle entirely** (0.72 ms × 237 GB/s ≈ 170 MB of free DRAM time). Capacity, not bandwidth, is the limit: only ~32 MiB survives, and it must not be evicted before use | 0.2–0.6 ms (2–5 %) if it works; 0 if the tier evicts before the use site | **OPEN — top of queue** | a rig with three arms per target: cold baseline / warmed-by-explicit-load on a second stream during a chosen idle window / warmed then polluted. Reports whether the warm bytes are *still resident* at the use site, not just whether a warm-loop is faster |
| P1a | ↳ GDN state warmed (concrete instance, already measured at one end) | D4 measured the state-coldness penalty at **+0.070 ms and its saturation at exactly 32 MiB** — the tier size. Warming the 18.9 MiB state during `attn` is the same mechanism with both ends already priced | ~0.07 ms | **OPEN** | P1's rig, target = the state buffer, idle window = `attn` |
| P1b | ↳ next-layer KV warmed before its `attn` | `attn` is latency-bound (13 % of floor), and per-layer KV at 4k is 8.3 MiB — comfortably inside the tier, so it can be resident *before* the walk starts rather than streamed cold | unknown; the family's 1.4 ms of non-floor time is the upper bound on what any latency fix can touch | **OPEN** | same rig, target = one layer's KV slice, warm issued during that layer's MLP |
| P1c | ↳ does the HIP residency API even exist on this part | `hipDeviceGetAttribute(hipDeviceAttributePersistingL2CacheMaxSize)` and `hipStreamSetAttribute(accessPolicyWindow)` are in the headers and used **nowhere** in `engine/` or `prototype/`. If the persisting tier maps to the 2 MiB L2 rather than the 32 MiB MALL, hardware residency hints are the wrong tool and only explicit prefetch applies | decides P1's implementation, not its prize | **OPEN — cheapest item here** | a 20-line device query + one gated A/B; do it first because it tells us which of two mechanisms is real |
| P2 | **Device-side argmax / smaller logits D2H** | `finish_token` (`decode_step.cpp:1169`) does one blocking D2H of the whole 993 KB logits row and a CPU §03.4 argmax; the token id is a data dependency for the next step, so **`host_tail` 0.7163 ms is on the critical path**, not hidden. Prior art: `kernels_lmhead.cpp:38-42` already rejected a *fused* device argmax because `atomicMax` breaks ties to the highest id while §03.4 wants the lowest, and §07.4 measured fusing at only ~1 % — so the addressable part is ~0.1 ms, not the whole 0.72 | ~0.1 ms, plus unblocking P1's use of the idle host window | **OPEN** | a warp-reduction argmax with the lowest-id tie rule (exact, no band in play: integer index over identical F32s), then prove `sampled.txt` byte-identical and time it |
| P3 | ~~Bound-based vocab pruning in `lm_head`~~ | Cauchy–Schwarz bound `\`logit_r ≤ \|W_r\|·\|x\|\`` is **exact**, so pruning on it would need no band — but measured on the pinned dumps it prunes nothing: `\|x\|·max\|W\| = 145.704` against a largest observed logit of 26.625, so the norm-ordered walk never terminates early and **survival is 100 % on all 9 pinned rows** | **0** (was hoped up to ~1 ms) | **CLOSED 2026-10-21** (`prototype/vocab_prune/RESULTS.md`, one CPU run, no device time) | done: `prototype/vocab_prune/survival_study.py` — refuses to print a rate unless an (x, logits) pair reproduces the reference |
| P4 | `--split 18` deeper lane splits in attention | already-implemented geometry, worth ~0.36 ms | 0.36 ms | **IN FLIGHT** (`c16bb38c` pricing the `>MAXSC` allowances) | see `prototype/maxsc_allowance/MAXSC-HANDOVER.md` |
| P5 | `mlp`/`proj` GEMV at 70–75 % of floor | count and shape both measured out (items 1–2); what is left is the library's algo choice and the GEMV inner form itself (vector width, loads-per-thread, cache policy on the weight stream) | ~0.3–1.1 ms in principle, but every prior attempt at this class landed at ~1.17× | **OPEN, low prior** | a per-kernel achieved-bandwidth probe across block-size and per-thread work before writing any kernel |
| P6 | BF16 GDN state instead of F32 | halves ~21 MB of the 43 MB/step state traffic | ~0.05–0.09 ms | **OPEN, numerics-gated** | standing rule: adopt if gates stay green; must-catch derived from geometry, and state is on the recurrent path so a band probe comes first |
| P7 | MTP draft path on GPU | recorded as slower on torch CPU (~21 vs ~24 tok/s, accept 0.66 at k=1) and explicitly "a GPU-path bet, not a win today" — **never measured on device** | unknown by construction | **OPEN, unmeasured** | needs a device burst of draft+verify vs plain greedy; large implementation, so it is queued behind everything cheaper |
| P8 | multi-request decode batching | Answered from source, not guessed: `decode_step.h:411-424` shows the engine **already** runs an `n`-row body layer-major — one batched MLP, batched projections, one glue over the `[n, H]` tile, bit-identical to the per-token forms — but the rows are `n` tokens of **one sequence sharing one `RequestState`**. B concurrent requests would need B independent states, per-request conv rings, per-request KV ranges for `attn`, and B sampled ids per step. The weight read (6.5 ms of 10.9) then amortizes across B tokens, so throughput scales while latency does not | largest lever in the repo for *throughput* (several-fold), zero for single-stream latency | **OPEN — a scope decision, not a route** | user decision first: this turns the engine into a concurrent-request server (the shape `AGENTS.md`'s 100-request arithmetic anticipates). If approved, the design step is one document, not a kernel | 
| P9 | glue/copy remainder | 49 `glue` launches: priced at 10.9× the prize to fold (394 MB/token extra reads) | 0 | **CLOSED** (`c118d12`) | — |
| P10 | split-K one-row GEMVs; "fewer, fatter launches"; per-segment drain prizes; cross-layer MLP merge | all measured and closed: `wave_shape/ITEM2-RESULT.md`, `mlp_insitu/D2-DIAGNOSIS.md`, gate zero §7 | 0 | **CLOSED** | — |

## How to read the top of this list

P1 is now the best cost-to-information ratio in the queue and it is the user's: two families are demonstrably
latency-bound rather than bandwidth-bound, there are three named windows where the machine has idle fetch
capacity, and one end of the mechanism is already priced (+0.07 ms, saturating at exactly the tier size).
P1c costs minutes and decides which mechanism is real. P3, previously the biggest hoped-for prize, is
closed: the exact bound prunes 0 % of the vocabulary (`prototype/vocab_prune/RESULTS.md`). P8 may be the
largest lever of all and we do not yet know whether it is even reachable in the current design.
