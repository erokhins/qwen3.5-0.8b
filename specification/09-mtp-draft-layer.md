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
- `h ∈ ℝ^{1024}`: the target hidden state **before** `RMSnorm_final`
  (i.e. `x_24`, §03.5) — A-12;
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

## 09.5 What must be pinned before trusting §09

There is no HF reference for this block, so §10 G8 requires, in order:

1. load `mtp.*` and confirm shapes §01.5 (mechanical);
2. single-step check: feed the *target's* `h` and `y_next` from a real greedy
   trace, compare `d` against the true next token — a correct MTP should agree on
   a high fraction of easy continuations (A-12 makes or breaks this);
3. accept-rate measurement over a corpus: `accepted / drafted`, reported together
   with tokens/s with drafting off. If accept ≈ 0, A-12 (pre- vs post-final-norm)
   is the first suspect, then the `fc` concat order.
