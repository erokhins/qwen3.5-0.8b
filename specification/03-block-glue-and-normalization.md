# 03 — Normalization, decoder skeleton, logits head

## 03.1 RMSNorm (zero-centered / “Gemma” form) — `Qwen3_5RMSNorm`

Used by `input_layernorm`, `post_attention_layernorm`, the final
`model.language_model.norm`, `q_norm`/`k_norm`, and every MTP norm. The affine is
`1 + w`, and `w` is initialized/stored near zero `[REF]`:

```
RMSnorm_w,ε(z) = ( z · rsqrt( mean(z², last axis) + ε ) · (1 + w) )  → cast back
```

Order fixed by HF-ref: cast input to F32 → normalize in F32 → multiply by
`(1 + w)` **in F32** → cast back to the input dtype `[REF]`. `ε = 1e-6`.
Weights: `input_layernorm`, `post_attention_layernorm`, final norm: `[1024]`;
`q_norm`, `k_norm`: `[256]`; all BF16.

Do not confuse with §03-below gated norm (§04.5), which uses a **plain** `w`.

## 03.2 Decoder-layer skeleton (all 24 layers)

```
residual = x
h = RMSnorm(x)                                  # input_layernorm
h = (linear_attn(h)  if layer_types[l] = linear_attention   else  attn(h))
x = residual + h                                # BF16 add
residual = x
h = RMSnorm(x)                                  # post_attention_layernorm
x = residual + mlp(h)                           # BF16 add
```

Pre-norm only, one residual add per sub-block, no bias anywhere, no per-layer
scaling, no hyperconnections/Gated-Residual (Flash-Next-only, absent here).

## 03.3 MLP (identical in every layer and in MTP)

`d_ff = 3584`, all three matrices bias-free (§01.2):

```
mlp(x) = ( silu(x·W_gateᵀ) ⊙ (x·W_upᵀ) ) · W_downᵀ
```

`silu` elementwise in the activation dtype; every dot accumulates in F32 and rounds once
(engine policy, A-07). The epilogue is stricter than that sentence suggests, and the difference
is measured, not theoretical (`prototype/mlp_gemv/ulp_probe.py`): HF's dtype path is **three**
roundings, not one —

```
g     = BF16( F32 dot(x, W_gate) )        u = BF16( F32 dot(x, W_up) )
act   = BF16( BF16( g·(1/(1+exp(-g))) ) × u )     # silu in F32 on the rounded gate, then ×u
out   = BF16( F32 dot(act, W_down) )
```

Doing the whole epilogue in F32 — the natural kernel, and the tempting literal reading of A-07,
which is about accumulation and not about this — moves **27 % of `act` elements** by 1 ULP away
from the reference. Two smaller facts belong here too: HF's order is `x·(1/(1+exp(-x)))` and the
algebraically identical division form costs a 2 ULP outlier at a *large* element; and an F32
`exp` differing from libm's by 1 ULP flips roughly one BF16 `act` element per few thousand,
which is why §10.1's F2 line carries a 2 ULP band instead of 1. All of it is arithmetic
sensitivity of a specific implementation; matching it is what "greedy parity" means here.

## 03.4 Logits head (tied)

```
logits[t] = RMSnorm_final(x_final[t]) · Eᵀ      ∈ ℝ^{248320}
```

`E = model.language_model.embed_tokens.weight` is reused as the output matrix;
there is no `lm_head.weight` (§01.1). Greedy decoding takes
`argmax_v logits[t][v]`; ties broken by lowest id (engine policy). The engine may
compute logits only for the last position during decode (§07.4) — an argmax
over a streamed F32 reduction of the 2048-lane-by-248320 table is equivalent.

## 03.5 Block routing summary

```
x0 = E[input_ids]                                    §02.3
for l = 0..23:   x_{l+1} = skeleton_l(x_l)           mixers §04 / §05
x_out = x_24 ; logits = logits_head(x_out)           §03.1 + §03.4
```

`tools/reference_dump.py` pins `inputs_embeds`, every `L{ll}.s_in`/`L{ll}.s_out`,
`pre_final_hidden`, `final_hidden` and `logits_last` for this chain (§10).
