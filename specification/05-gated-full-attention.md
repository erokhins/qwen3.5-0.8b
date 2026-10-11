# 05 — Gated GQA block (`full_attention`, 6 layers)

Layers `3,7,11,15,19,23` plus the MTP layer (§09). Weights §01.3; `n_q = 8`,
`n_kv = 2`, `d_head = 256`, GQA group `g = 4`, `d_rot = 64`, scale `s = 1/16`.
Input `x ∈ ℝ^{T×1024}` = post-`input_layernorm` (§03.2). No attention bias, no
dropout.

## 05.1 Projection and gate split

```
qg     = x · W_qᵀ ∈ ℝ^{T×4096}
qg[t]  viewed as [8][2×256]:   q[t,h] = qg[t][h][0:256],  gate[t,h] = qg[t][h][256:512]
gate[t] = concat_h gate[t,h] ∈ ℝ^{2048}
k      = x · W_kᵀ ∈ ℝ^{T×512}   → [T][2][256]
v      = x · W_vᵀ ∈ ℝ^{T×512}   → [T][2][256]
```

The interleaved `[q | gate]` per head (not `[all q | all gate]`) is HF-ref:
`chunk(q_proj(x).view(T, −1, head_dim*2), 2, dim=−1)` `[REF]`. Getting this wrong
is silent and catastrophic, so it is gated explicitly (§10 G4).

## 05.2 Head normalization

`q_norm`, `k_norm` are §03.1 zero-centered RMSNorms with weight `[256]`, applied
per head over `d_head` (not over the flattened 2048) `[REF]`, F32 math, cast back.

```
q̂[t,h] = RMS_c(q[t,h]) ⊙ (1 + w_qnorm)
k̂[t,j] = RMS_c(k[t,j]) ⊙ (1 + w_knorm)
```

## 05.3 Partial rope

Inverse frequencies (§00.3): `f_i = 1e7^{-(2i)/64}`, `i = 0..31`, F32. For
position `p`:

```
c[p,i] = cos(p·f_i),  s[p,i] = sin(p·f_i)          i = 0..31
c[p]   = concat(c[p], c[p]) ∈ ℝ^64,  s[p] = concat(s[p], s[p]) ∈ ℝ^64
```

Applied to the **first 64** dims of each head, remaining 192 dims pass through
unchanged `[REF]`:

```
rot(z)[0:64]  = z[0:64] ⊙ c[p] + rotate_half(z[0:64]) ⊙ s[p]
rot(z)[64:]   = z[64:]
rotate_half(a)[i] = −a[i+32] (i<32),  a[i−32] (i≥32)
```

so pairs are `(i, i+32)` for `i = 0..31`. Both `q̂` and `k̂` are rotated; `v` is
not. Positions come from §02.4.

**Rounding policy — measured, and not stated here until now.** The table enters the rotate as BF16,
and each product is rounded before the add:

```
c[p], s[p] ∈ BF16        (evaluate in F32, cast to BF16 — HF's rotary embedding returns cos.to(dtype))
rot(z)[d]  = bf16( bf16(z[d]·c[p,d]) + bf16(rotate_half(z)[d]·s[p,d]) )      d < 64
```

Keeping the table in F32 — which is what `rot*c + rotate_half(rot)*s` does by type promotion, and
what `speccheck::apply_rope` did until this was corrected — misses **52–97 of every 2048
`attn.q_rope` elements** on `dec8` (cos 0.99983, meanrel 0.0174: comfortably inside §10 G4's
threshold, which is why no gate ever failed on it). The BF16 policy reproduces `attn.q_rope` and
`attn.k_rope` bit-for-bit: 0 wrong elements over 277 positions × 6 attention layers × 512 rope
elements each (`speccheck/rope_table_probe.py`). The evidence was a prototype kernel matching the
pins the oracle could not match, so the pins — not taste — are the source of this rule.

Two boundaries the same probe measured, both of which constrain any implementation of this section:

* the dumps carry **F32 libm artifacts, not correctly-rounded values**: evaluating the same angle in
  F64 and rounding down misses 3 of 277 positions (180, 217, 241) that the F32 path gets exactly.
  "More precision" in the table is therefore not a route to matching the pins.
* a device libm's F32 `cosf`/`sinf` need not agree with this box's CPU libm: the gfx1151 table
  written by `prototype/attn_decode` differs from torch's (and from a plain glibc `cosf`) in 2 of
  128 entries at `p = 64` and in 0 of 128 at the other 276 positions. See §07.3 and
  `prototype/attn_decode/results.md` for what that residual costs.

## 05.4 mrope is inert for text-only

`mrope_interleaved: true`, `mrope_section: [11,11,10]`. HF-ref builds `cos/sin`
from three position streams `(t,h,w)` and overwrites interleaved slots
(`t` keeps pairs `0,3,6,…`; `h` takes `1,4,7,…`; `w` takes `2,5,8,…`) before the
`concat(·,·)` of §05.3 `[REF]`. With §02.4 the three streams are identical, so
the composition collapses to plain rope on `d_rot = 64` dims. The engine may
therefore implement plain rope and nothing else (A-06). If image input is ever
added, this section must be re-specified.

## 05.5 Attention

Let `κ(j) = ⌊j/4⌋` map a query head to its KV head. For query position `t`
(`t` global, i.e. offset by the cached prefix length during decode):

```
z[t,i,j] = ( q̂rot[t,i] · k̂rot[κ(i),j] ) · s        over all cached j ≤ t
w[t,i,j] = softmax_j( z[t,i,·] ⊕ mask )             computed in F32, then cast
a[t,i]   = Σ_j w[t,i,j] · v[κ(i), j]               ∈ ℝ^256
```

`mask` is causal; padding positions are excluded when a batch is padded. Softmax
accumulates over all valid `j` (§08.3 note on long context). `a` is laid out back
to `[T][2048]` head-major.

## 05.6 Output gate and projection

```
ô[t] = concat_i a[t,i] ⊙ σ(gate[t])     (elementwise, 2048 lanes, head-major)
out  = ô · W_oᵀ                          ∈ ℝ^{T×1024}
```

The gate is applied to the attention output **before** `o_proj`, in the
activation dtype, with plain `σ` (no `output_gate_type` key exists in this
checkpoint; HF-ref hardcodes `torch.sigmoid`) `[REF]`.

## 05.7 KV cache write

`k̂rot` and `v` (post rope, post `k_norm`) are appended to the request's cache:
`K^{(l)}_j, V^{(l)}_j ∈ F32?` — no: stored in the activation dtype (BF16),
`[2][256]` per token per layer `[REF]`. §08.1 sizes. The engine must never cache
pre-rope keys.

## 05.8 Reference operand names

Pinned per attention layer: `in` (normed input), `q_rope`, `k_rope`, `v`,
`attn_out_pre_gate`, `out` (post gate + `o_proj`).
