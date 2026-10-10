# 06 — Prefill pass

Whole-request evaluation of `x0 → logits` for `T` tokens with fresh state.

## 06.1 Sequence of operations

```
input_ids ∈ ℤ^{1×T}   (rendered per §02.2)
p[t] = t              (t = 0..T−1)
x = E[input_ids]                                   §02.3
for l = 0..23:                                     §03.2
    x = block_l(x)        where block = §04 (18×) or §05 (6×)
logits = §03.4(x)
```

State initialization at request start (§08 for the buffers):

| state | initial value |
|---|---|
| GDN `S^{(l)}` (18 × `[16][128][128]`) | 0 |
| GDN conv window (18 × `[6144][3]`) | 0 |
| `K^{(l)}, V^{(l)}` (6 layers) | empty, length 0 |

## 06.2 GDN prefill (§04 within a pass)

`raw`, `z`, `b_raw`, `a_raw` are computed for all `T` tokens at once (one GEMM
per projection). The conv is causal over the *request start*: `raw[t<0] = 0`,
which is exactly the zero conv window of §06.1 — the engine may either pad
`raw` with 3 zero columns or use a chunked conv that carries the window (§04.2).

The scan is the recurrence of §04.4, or the chunk form of §04.7 (chunk 64,
zero-β padding to a multiple of 64, HF-ref choice `[REF]`). At the end of prefill
the state buffers hold `S` after token `T−1` and the conv window holds
`raw[T−3 .. T−1]` (§08.2).

## 06.3 Attention prefill (§05 within a pass)

`q̂rot, k̂rot, v` for all `T` positions, appended to `K^{(l)}, V^{(l)}`, then
causal attention over the whole prefix. Because the prefix is empty at prefill,
`j` ranges over `0..t` only.

## 06.4 Cost and what it implies for the engine

Per prefill token the arithmetic is ~`2·(Σ W params read)`; the dominant terms
per layer are `in_proj_qkv` (6.3 M MACs), `out_proj` (2.1 M), MLP (22.0 M) and,
for attention layers, `q/k/v/o` (4.2 + 1.0 + 1.0 + 2.1 M) plus the attention
score work (`O(T²)`). Attention layers are only 6/24 and their scores are
`8` heads over `256` dims, so prefill on this model is dominated by the MLP +
GDN GEMMs and is **compute-bound** for `T ≳ 128` — unlike decode (§07.5).

Long-context note: the cost of §04 is `O(T)` and its memory is `O(1)` per request,
while §05 costs `O(T²)` time and `O(T)` memory over 6 layers. At 262144 context
the KV cache is 3.07 GiB per request (§00.5), which on this box is a scheduling
decision, not a capacity limit (§08.3).

## 06.5 What prefill must produce for decode

After prefill the engine holds, per request:

- `x_T`: nothing — decode restarts from `E[sampled_token]`, so no hidden state
  crosses the prefill/decode boundary except the §08 state and `prefix_len = T`.
- sampled token `y0 = argmax logits[T−1]`.

This is a real constraint: the decode step (§07) must be expressible with only
`y_{t−1}`, the per-layer state and `prefix_len` — which it is, because both mixers
are incremental (§04.6, §05.7).
