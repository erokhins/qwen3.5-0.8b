# 04 — Gated DeltaNet block (`linear_attention`, 18 layers)

Layers `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22`. Weights §01.2, geometry
§00.3. Notation: input `x ∈ ℝ^{T×1024}` (the post-`input_layernorm` block input,
§03.2); per value head `η ∈ 0..15`; `d_k = d_v = 128`; `h_k = h_v = 16` so there
is **no** key→value head repeat (ratio 1) `[REF]`. No RoPE here — positionality
comes from the conv only.

## 04.1 Projections (§04 chain step 1)

```
raw  = x · W_qkvᵀ   ∈ ℝ^{T×6144}      W_qkv ∈ ℝ^{6144×1024}
z    = x · W_zᵀ     ∈ ℝ^{T×2048}      viewed as [16][128]
b_raw= x · W_bᵀ     ∈ ℝ^{T×16}        per value head
a_raw= x · W_aᵀ     ∈ ℝ^{T×16}        per value head
```

`raw` channel layout: `q = raw[:, 0:2048]`, `k = raw[:, 2048:4096]`,
`v = raw[:, 4096:6144]`; each view is `[T][16][128]` head-major.

## 04.2 Causal depthwise conv (kernel 4)

`C ∈ ℝ^{6144×4}` (stored `[6144,1,4]`, tap axis last), **no bias**, activation
`SiLU` (`hidden_act`):

```
u[t] = silu( raw[t] + Σ_{j=0..3} C[:, j] ⊙ raw[t-(3-j)] )     raw[<0] = 0
```

i.e. tap `j` multiplies the token `3-j` positions earlier; the last tap `j=3`
multiplies the current token. Split `u` into `q, k, v` with §01/§04.1 layout.
Incremental state: the last `k_conv-1 = 3` columns of `raw` (**pre**-conv) —
see §08.2 `[REF: HF caches pre-conv mixed_qkv]`.

## 04.3 Decay and write gates

Per value head `η`, computed in F32:

```
β[t,η] = σ(b_raw[t,η])
g[t,η] = −exp(A_log[η]) · softplus(a_raw[t,η] + dt_bias[η])      (≤ 0)
γ[t,η] = exp(g[t,η]) ∈ (0,1]
```

`A_log` is stored F32; `dt_bias` BF16 — see A-02. `a_raw` is F32-cast before the
addition `[REF]`.

## 04.4 Recurrent step (normative definition)

L2 normalization, F32, epsilon **inside** the root:

```
q̃[t,η] = q[t,η] · rsqrt(‖q[t,η]‖² + 1e-6) · 128^{-1/2}
k̃[t,η] = k[t,η] · rsqrt(‖k[t,η]‖² + 1e-6)
```

State `S_η ∈ F32^{128×128}` (`S[i, j]`, `i` over `d_k`, `j` over `d_v`), zero at
request start. Per token, in this order:

```
S ← γ[t,η] · S
r   ∈ ℝ^{128}:  r[j] = Σ_i S[i,j] · k̃[i]          (r = Sᵀ k̃)
Δv  = β[t,η] · (v[t,η] − r)
S   ← S + k̃ ⊗ Δv                                   (S[i,:] += k̃[i]·Δv[:])
o[t,η] = Σ_i S[i,:] · q̃[i]                          (o = q̃ᵀ S)
```

`o ∈ ℝ^{T×16×128}`. Notes:

- The `128^{-1/2}` factor on `q̃` **is** applied by HF-ref (and FLA), always, and
  is *not* applied to `k̃` `[REF]`. Because §04.5 normalizes `o` per head, any
  uniform scale on `o` is erased in exact arithmetic; the factor still changes
  F32 rounding, and it is what the reference computes — keep it (A-05).
- Decaying `S` before retrieving is HF-ref order; folding the decay into `r`
  (`r = γ·Sᵀk̃`, `S ← γ·S + k̃⊗Δv`) is algebraically identical (§06 note in the
  old project's spec 06).

## 04.5 Gated RMSNorm and output projection

Per token and head, F32 statistics over the 128 value dims, `ε = 1e-6`, plain
affine (this norm is **not** zero-centered, unlike §03.1):

```
ô[t,η] = ( o[t,η] · rsqrt(mean(o[t,η]²) + ε) ) ⊙ w_norm
ô[t,η] = ô[t,η] ⊙ silu(z[t,η])
y      = concat_η(ô) · W_outᵀ  ∈ ℝ^{T×1024}
```

Cast order fixed by HF-ref `[REF]` (A-01): normalize in F32 → cast the normalized
value back to the activation dtype → multiply by `w_norm` → multiply by
`silu(z)` in F32 → cast back → `out_proj`. `w_norm` shape `[128]`, shared by all
16 heads. `z` is consumed as `[16][128]` aligned with the heads of `o`.

## 04.6 State produced

Per request per GDN layer: `S ∈ F32^{16×128×128}` and the conv window
`∈ BF16^{6144×3}`. Sizes and lifecycle §08.

## 04.7 Chunked prefill equivalence

Any prefill algorithm is correct iff, for every `t`, it returns the `o[t]` and
final `S` of §04.4 applied token-by-token. The standard Gated-DeltaNet chunk
form (chunk 64, log-decay prefix sums, UT/WY transform of the delta updates,
intra + inter-chunk read split) is HF-ref's prefill path `[REF]`; vLLM-ref and
the engine may differ in chunk size or fusion strategy. Gate: chunked vs recurrent
must agree on random short sequences before any chunked kernel is trusted
(`specification/10` G3).

HF-ref pads the sequence to a multiple of the chunk size with zeros for `q,k,v,β,g`
and drops the padding afterwards `[REF]`; zero-`β` padding means those positions
contribute no state update, which is the property the engine must preserve when
prefilling ragged batches.

## 04.8 Reference operand names

`tools/reference_dump.py` pins, per GDN layer and pass:
`mixed_qkv_pre_conv`, `q_post_conv`, `k_post_conv`, `v_post_conv`, `g_decay_log`,
`beta`, `o_pre_norm`, `gated_out`, `state_new`, `conv_state` — mapping §04.1–§04.5.
