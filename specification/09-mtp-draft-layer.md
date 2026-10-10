# 09 — MTP block as speculative drafter

One MTP layer (`mtp_num_hidden_layers: 1`), geometry §01.5: a `full_attention`
block (§05) with its own KV, two pre-norms, a fusion `fc`, and its own final norm.
HF-ref has **no** MTP class for `qwen3_5`, so this section is specified from the
checkpoint tensors plus vLLM-ref `qwen3_5_mtp.py` `[REF]`; everything marked A-xx
is falsifiable by accept-rate (§10 G8).

## 09.1 Inputs of a draft step

Given, after a verified target step:

- `y_next ∈ [0,V)`: the token the target just emitted (the token whose successor
  we want to draft) — embedded with the **shared** table
  (`mtp_use_dedicated_embeddings: false`), so `e = E[y_next]`;
- `h ∈ ℝ^{1024}`: the target hidden state **after** `RMSnorm_final` — i.e. the vector
  the `lm_head` sees (§03.4). A-12 originally said *before* (`x_24`, §03.5) and was
  **falsified by accept rate**: 0.642 post-norm vs 0.232 pre-norm, 9 prompts, k=1
  (`speccheck/gates/run_mtp.py`). The drafter eats the normed hidden.
- `p`: position of the drafted token (= `prefix_len` of the target at that point).

## 09.2 Forward

```
e' = RMSnorm_{w_pre_fc_norm_embedding}( e )
h' = RMSnorm_{w_pre_fc_norm_hidden}( h )
u  ( e' ⊕ h' ) · W_fcᵀ                       ∈ ℝ^{2048} → ℝ^{1024}
then one §05 block at position p on input u   (own K/V, own q_norm/k_norm, own gate)
then the §03.2 MLP sub-block
then x_d = RMSnorm_{mtp.norm}( · )
draft logits = x_d · Eᵀ                        (tied table, same as §03.4)
d = argmax
```

`(e' ⊕ h')` is concatenation in the order *embedding first, hidden second*
`[REF: torch.cat([inputs_embeds, hidden_states])]`, matching `W_fc ∈ ℝ^{1024×2048}`.
All three MTP norms are the zero-centered form (§03.1).

## 09.3 Drafting more than one token

`L_mtp = 1`, so multi-token drafting reuses the same block iteratively: feed the
draft's own `x_d` as `h` and the previous draft token as `y_next` at position
`p+1` `[REF]` (vLLM indexes `layers[spec_step_idx % num_mtp_layers]`). Each draft
step appends to the MTP KV; a rejected draft must roll that KV back by truncating
to the verified length (§08.5).

## 09.4 Verification rule (greedy)

With greedy target sampling, accept a draft token iff it equals the target's
argmax at that position; on rejection, the target's argmax wins and the MTP KV is
truncated. The engine may draft `k` tokens per target step; the accept path costs
one target forward per token, the win is skipping MTP-free work — i.e. the gain
exists only if a drafted step is cheaper than a real step (§07.4 roofline: a real
step is ~1.55 GB of traffic, a draft step is `40.9 MB + E row + E for logits`).
Because the draft also streams the tied table for its logits, draft cost ≈
`0.55 GB/token` — the speedup budget is thin, which is why MTP is M3 and not M0.

**A batched verify is not bit-lossless in bf16.** The rule above assumes the target's
verification computes what it would have computed alone; one forward over
`[t, d1..dk]` is a batched GEMM where the sequential path is a GEMV, and bf16 rounds
those differently, so a near-tie argmax can flip inside the verify itself. Measured
with `verify="steps"` (draft block fed through the ordinary decode path) the output is
token-identical to plain greedy and the state is bit-identical to a sequential replay;
with `verify="batch"` agreement came out 1.000 on the gate prompts but is not
*guaranteed*. An engine that must be exactly lossless has to make its verify path reuse
the decode kernels bit for bit; one that accepts the A-18 chaos floor may batch.

## 09.5 Gate G8 (measured; this is what discharges A-12/A-13)

There is no HF reference for this block, so `speccheck/gates/run_mtp.py` checks three
things, in order of strength. Numbers: k=1, 24 tokens, 9 prompts (6 corpus + 3 fixture
prompts), torch CPU bf16.

1. **G8a — the loop is exactly right (binding, bit-exact).** With `verify="steps"`, the
   speculative output equals plain greedy token for token, *and* the target state after
   the run is bit-identical to a sequential replay (`torch.equal` on every GDN `S`,
   conv window and KV entry, plus `prefix_len`). That is what tests positions, state
   rollback and MTP KV truncation; it caught an off-by-one in the consumed-prefix
   definition and, before that, a loop that re-fed the first generated token forever.
2. **G8b — batched verify stays honest (binding, floored).** Agreement of the batched
   verify output with plain greedy, measured 1.0000 against the A-18 floor 0.8776.
3. **Accept rate (binding, decisive between readings of §09).**

| reading of §09 | accept `accepted/drafted` | tokens per target forward |
|---|---|---|
| as corrected: post-final-norm `h`, `fc = [emb ‖ hidden]` | **0.642** (86/134) | 1.15 |
| A-12 as originally written (pre-final-norm `h`) | 0.232 (41/177) | 0.68 |
| A-13 alternative (`fc = [hidden ‖ emb]`) | 0.014 (3/213) | 0.50 |

A wrong `fc` order collapses to ~0, which is what makes this gate worth running: it
cannot be passed by an accident of tuning.

**Speed, reported not gated** (`./bench.sh --engine speccheck --mtp K`, §10 tests):

| engine | decode tok/s | accept | tokens/forward |
|---|---|---|---|
| speccheck | 23.3 | – | 1.00 |
| speccheck + mtp1 | 18.3 | 0.582 | 1.11 |
| speccheck + mtp2 | 15.8 | 0.408 | 1.03 |

Roofline break-even at k=1 needs tokens/forward > (1.55+0.55)/1.55 = 1.36, i.e.
accept > 0.36, and the measured 0.58 clears that. torch CPU nevertheless *loses* 21 %:
the draft pays a full 248k-row tied `lm_head` as a slow bf16 GEMV, and this
implementation pays an extra target forward to roll state back on rejection. Both are
avoidable in the engine — the draft `lm_head` is bandwidth work like the target's, and
a per-position state checkpoint (~19 MB) is cheaper than a re-forward — so the honest
conclusion is **not "MTP does not help"** but "MTP does not help *in Python*, while the
roofline says ~1.15× at k=1 if the draft is written as a kernel". Re-measure after the
first real `mtp_layer` kernel exists.
