# Row scale: what an extra row actually costs, and why that reopens speculation at k = 2

Measured 2026-10-23 by `prototype/mtp_accept/row_scale.sh`, evidence in
`evidence/row_scale-2026-10-23.txt`. This **corrects** §2 of `prototype/mtp_llama/LLAMA-MTP-RESULT.md` and
the D3 row of `AGENTS.md`: the mechanism named there ("our width-2 wave re-pays the weight walk and the KV
pass") is wrong, and the correction is the whole value of this file.

---

## 1. The instrument, and why it needed no new code

A round is a body of `k+1` rows. `--mtp-accept` prices such a body at **2.085 steps** at 4k
(`verify_body_scale.sh`), but `engine/mtp_verify.cpp` forces `WaveGemm::kHand` + `GdnScan::kSeq` on it so
that row 0 stays bit-comparable to the one-row step. Nobody had priced the *same body* on the amortising
route the engine uses everywhere else. It is already instrumented: **a prefill pass chunked with
`--wave-rows N` is a sequence of N-row bodies through the amortising route**, and `--time-waves` splits it
by family. So: same `submit_wave_body`, rows ∈ {1,2,3,4,8,16}, twice — once over an empty-ish cache (pure
weight-walk) and once as a 4,032-token pass (KV scan live).

Caveats kept visible: the wave clock **serializes** the stream, so family `ms` are attribution-only (their
sums exceed the burst wall by the `exposed_ms` shown per row); in the long case a body at position *p* sees
*p* lanes, so that column mixes body cost with the causal ramp — the **slope** is what is quoted.

## 2. The measurement

Cost per body, `burst_wall / bodies`, ms:

| rows | short KV (weight walk) | 4k KV (long) | long, as a multiple of the 4k step (11.22 ms) |
|---|---|---|---|
| 1 | 9.18 | 7.47 | 0.67 |
| 2 | **11.48** | **12.25** | **1.09** |
| 3 | 11.80 | — | — |
| 4 | 11.67 | 12.41 | 1.11 |
| 8 | 13.13 | 12.68 | 1.13 |
| 16 | 12.85 | — | — |

Per family, the two jumps that matter, 1 row → 2 rows:

| family | short KV | 4k KV | what it is |
|---|---|---|---|
| `proj` | 2.63 → 4.21 (+1.59) | 2.73 → 4.27 (+1.54) | the projections switch from the one-row hand GEMV to the tiled/library form |
| `attn` | 0.11 → 0.67 (+0.55) | 1.41 → 4.20 (+2.79) | the wide attention form: at 4k the second row re-reads the KV scan |
| `mlp_gateup` / `mlp_down` | 2.29 → 2.47 / 1.01 → 1.27 | 2.28 → 2.54 / 1.05 → 1.31 | ~0.2 ms each |
| `gdn_rec` | 0.30 → 0.39 | 0.32 → 0.43 | the recurrence, which is serial per row by construction |
| `lm_head` | 2.12 → 2.13 | one launch per pass, flat | the tied head reads its 485 MiB once for any row count |

**And beyond two rows there is nothing left to pay.** 2 → 8 rows costs +1.2 ms short and **+0.43 ms at 4k
in total**, with `attn` per body frozen at 4.2–4.3 ms whether the body carries 2 rows or 8: the wide form
stages the K/V tiles once and covers the query positions inside the same tiles. Rows 3 and 4 are, to within
the noise of this instrument, free.

## 3. What this says, and what it corrects

1. **The geometry was never the problem.** A 3-row body over a populated 4k cache costs **1.09–1.13 of a
   decode step** on the amortising route. llama.cpp's own verify — the reference implementation this repo
   has been holding up as the bar — costs **1.26 (2 rows) to 1.50 (5 rows)**. Our engine is *cheaper per
   row* than llama.cpp; it just isn't allowed to say so on the accept path.
2. **So the 2.085-step round is a route cost, not a machine cost.** The forced `kHand`+`kSeq` exists for one
   reason — row 0 must reproduce the shipping step bit for bit — and that reason is in tension with the only
   thing that makes speculation pay. §2 of the llama.cpp doc and the D3 row of `AGENTS.md` attributed the 2×
   to a width-2 wave re-reading weights; `row_scale.sh` disproves that: on the amortising route the same
   wave re-reads almost nothing. What remains true is the empirical half — D3's plan-A bill of 0.1505 step
   for a 2-row body is far below any measured round — but the *explanation* is now the route, and it is
   fixable rather than structural.
3. **The prize, sized.** Take a 3-row (k = 2) round at 4k on the amortising route: body ≈ 12.4 ms, plus the
   per-round bookkeeping the engine already pays (checkpoint+restore ~19.3 MiB each way, verdict kernel,
   readback) and two draft passes at ~0.44 ms. That is **≈ 1.3–1.4 steps**, against tokens-per-round of
   `1 + a₁ + a₁a₂`. At the measured corpus rate a₁ ≈ 0.66 and llama.cpp's conditional second-position rate
   (0.52) as the stand-in for a₂, that is 1.96 tokens for 1.35 steps ≈ **+45 %**. With a₁ alone at 0.66 it is
   1.66 / 1.35 ≈ **+23 %**. Both are far above the standing adopt line, which is why this is worth doing
   before any other decode route.

## 4. Pre-registered lines, written before any k = 2 number exists

- **Route gate.** The accept round, amortising route, must reach **≤ 1.55 steps at 4,000 lanes** (today
  2.085). If it does not, the route work is insufficient and the row argument dies with it — no k=2 loop is
  built on top of a round that still costs two steps.
- **Accept-rate kill.** Mean accepted tokens per k=2 round on the corpus population **< 0.40** kills the
  route: at that rate even a 1.35-step round returns less than one token per step. Derived, not fitted —
  parity at a 1.35-step round needs 0.35, and 0.40 is that with the smallest margin this repo treats as real.
- **Adopt line, unchanged:** **≥ +10 % tok/s at 4,000 lanes** vs base, medians of 5 in one lock hold, on
  `DECWALL`, with the fixture and the alignment named beside the number (`AGENTS.md`'s index-alignment row).
- **Numerics condition.** If the amortising route makes row 0 non-bit-identical to the one-row step, that is
  the already-adjudicated "two legal bf16 forms" case, not a new band: the near-tie flip admission
  (`m3_loop_rate.py`'s `flip_witness`) is the instrument and must be re-run over the corpus population. If
  instead the hand route gets a token-tiled form that stays bit-identical, nothing is loosened at all and
  that is the preferred outcome — it is also why the route work comes before the loop work.
