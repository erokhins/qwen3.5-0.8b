# Beat-MTP programme (door b funded, 2026-10-23) — and the ranking the numbers give it

## Objective, stated so it can be failed

**Our engine's speculative decode must beat llama.cpp's best MTP result on this box.**
llama.cpp's best is **+13.6 %** tok/s at 4,152 lanes (k=2, `evidence/llama_mtp_ab-2026-10-23.txt`).
Bar, pre-registered before any number: **≥ +14 % tok/s at 4,000 lanes over our own base**, greedy,
DECWALL, medians inside a single lock hold. The standing adopt floor (+10 %) still applies; this bar is
the *programme* target, i.e. what "beat MTP" means in writing.

The door was chosen after the route gate failed (round/step 1.964 at 4k against the pre-registered
1.55), so the programme's work is to make a verify row cheap rather than to keep re-pricing k.

## The ranking is NOT the one the source-read implied — say so before spending on it

`LLAMA-HOW.md` ranked the mechanisms llama.cpp uses: (1) the recurrence walked inside one launch with
kernel-emitted snapshots, (2) multi-query-row attention over one streamed K/V pass, (3) round-level
costs paid once. Ranked by *llama.cpp's* architecture that order is right. Ranked by **our** numbers it
inverts, and the inversion is large.

Per-family cost of ONE plain decode step at 4,032 lanes, measured this session
(`--time-stages`, `--gdn-loop 1`, step 12.07 ms, `plan_match=1`, one lock hold):

| family | ms/step | extra row costs, from `row_scale.sh` long-run marginals |
|---|---|---|
| `mlp` | 3.162 | +0.52 |
| `proj` | 2.764 | +1.54 (hand GEMVs, one launch per row; the amortised route buys ~2 ms of this back for the whole body) |
| `lm_head` | 2.298 | **flat** (2.14 ms for 1 row or 64 — `ROW-SCALE.md`) |
| `attn` | 1.621 | **+1.6 to +2.8** (per-row KV re-scan; ramp-average reads +2.79) |
| `host_tail` | 1.020 | — |
| `glue` | 0.663 | +0.09 |
| `gdn` | 0.539 | **+0.11 to +0.4** (state traffic is 36 MiB/step ≈ 0.2 ms at the measured rate) |

Now the sum. Measured cost of a k=1 round over a plain step, 4,032 lanes, on the amortised route:
**22.235 − 11.319 = 10.916 ms**. The kernel marginals above add up to **4.3–5.5 ms**. So:

> **~5.5–6.6 ms of the round's marginal — half of it — is not kernel work at all.** It is the round's own
> machinery: the 19.27 MiB mid-checkpoint taken, the 19.27 MiB shadow re-seeded, the lane save/revert
> over three rows, the verdict kernel, the **blocking** logits D2H, and the sync bubbles between them.
> And the mechanism I ranked first yesterday — GDN snapshot planes for the recurrence — is worth
> **0.1–0.4 ms per row** on our engine, i.e. close to nothing as a speed item.

Reading a mechanism off a reference implementation and assuming it prices the same here is the same
error as the WMMA sentence I had to correct in my own report, one level over. Hence the re-ordering.

## Funded order (b), with per-phase kill lines set now

| phase | work | expected | **kill if it buys <** |
|---|---|---|---|
| **B0** | meter the round: per-phase ms of the verify body *and* of the accept loop's bookkeeping, printed by the engine, default-off, non-comparable with `--time-stages` | makes B1/B2 arithmetic into measurement | no bar — it is the instrument the others are judged with |
| **B1** | delete the round's byte traffic: snapshot planes so rollback selects a slot instead of copying 19.27 MiB out and back; on-device verdict with **async** readback instead of the blocking D2H; lane revert by index | 3–6 ms/round at 4k | **1.0 ms/round** at 4,000 lanes |
| **B2** | attention that amortises rows at long context: several query rows per block over one streamed K/V pass, per-row causal bounds (llama's `KV_max`-style truncation), on the decode path rather than the prefill wide form | 1.6–2.8 ms/row at 4k | **1.0 ms/round** at 4,000 lanes |
| **B3** | `proj` row amortisation beyond what hipBLASLt already buys (per-row hand GEMVs are +1.54 ms/row) | ≤ 1.5 ms/row | 0.5 ms/round |
| **not funded for speed** | GDN multi-token walk / snapshot planes **as a recurrence-speed item** | 0.1–0.4 ms/row | — it may still come back inside B1, which is where its value actually is |
| **B4** | k=2 accept loop + M3 flip, on whatever the above delivers, with the M3 bands unchanged (kill mean accepted < 0.40/round) | the k=2 step that is currently refused | — |

Two things this plan deliberately does **not** do: it does not re-draw the failed route gate to make the
amortising route pass (the failure stands: 1.964 against 1.55); and it does not touch
`engine/performance-status.md` while `--mtp-accept` is default-off.

## What has to be true at the end

A k=1 round at 4k under ~13 ms (from 21.4–23.4 today) with accept ~0.66/round, i.e. **~1.05–1.1
steps per emitted token**, which is the +14 % bar. If B1+B2 land and the round is still >15 ms, the
programme is closed on measurement and the honest answer is that speculation does not pay for this model
at 4k on this box — a conclusion worth having, not a failure to avoid.
