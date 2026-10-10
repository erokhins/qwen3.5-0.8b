# 07 — Decode step

Single-token, incremental. Inputs: previous sampled token `y ∈ [0,V)`, current
`prefix_len = P`, and the request state of §08.

## 07.1 Order of operations

```
p    = P                                       (§02.4; 0-based position of the new token)
x    = E[y]                                    ∈ ℝ^{1024}           §02.3
for l = 0..23:
    if linear_attention: x = §07.2 step_l(x)
    else:                x = §07.3 step_l(x)
    x ← residual adds per §03.2
logits = RMSnorm_final(x) · Eᵀ                 ∈ ℝ^{V}             §03.4
y'     = argmax_v logits[v]                    (ties: lowest id)
P     ← P + 1
```

Both mixers consume `x ∈ ℝ^{1×1024}`; every norm, gate and add is the same
formula as prefill with `T = 1`.

## 07.2 GDN decode step (§04 with T = 1)

```
raw  = y_row · W_qkvᵀ ∈ ℝ^{6144};  z = ·W_zᵀ;  b_raw, a_raw ∈ ℝ^{16}
u    = silu( raw + Σ_{j=0..2} C[:, j] ⊙ convwin[3−1−j] )        # window has 3 cols
convwin ← roll(convwin, raw)                                   # drop oldest, append raw
per head η:
    β, γ from b_raw, a_raw                                      §04.3
    q̃, k̃ L2-normalized (k̃ over d_k=128)                        §04.4
    r = S_ηᵀ k̃ ;  Δv = β(v − r) ;  S_η ← γ·S_η + k̃⊗Δv ;  o_η = q̃ᵀS_η
ô  = RMSNormGated(o, z)                                         §04.5
out = concat_η(ô) · W_outᵀ
```

The conv window is updated from **pre-conv** `raw` values (§04.2), so the window
write and the `u` computation must use the old window contents — an in-place
ring update, not a write-then-read.

## 07.3 Attention decode step (§05 with T = 1)

```
q, gate, k, v                                              §05.1
q̂ = q_norm per head ; k̂ = k_norm per head                  §05.2
q̂rot, k̂rot = rope(·, p)                                    §05.3
append k̂rot, v to K^{(l)}, V^{(l)}  (slot P)
for i = 0..7:  w = softmax_j( (q̂rot[i] · K^{(l)}[κ(i), j]) · s ⊕ mask )
               a[i] = Σ_j w[j] · V^{(l)}[κ(i), j]
out = ( concat_i a[i] ⊙ σ(gate) ) · W_oᵀ                    §05.6
```

`mask` allows `j ≤ P`. During decode the softmax is over `P+1` lanes. The engine may
compute it as one `softmax` or as a blocked loop with `(max, sum)` rescaling —
*online* softmax, the reduction a flash-decoding / split-K kernel uses — and §10 G6
pins that the two agree.

**`(max, sum)` state is per query row, never per request.** Every decode step has a
new query, so its softmax is a new reduction: an accumulator carried from the
previous step belongs to a query that no longer exists. Implemented that way by
mistake here, decode step 1 produced attention with `cos 0.876` to the reference and
matched the *previous* prefix instead (`0.9946`), then diverged token-wise. The only
defensible reading of "running accumulators" is the within-row one, and §10 G6 tests
exactly that (A-17).

## 07.4 Cost model (this box)

Bytes read per token: all block weights `1.5457 − 0.5086 = 1.0371 GB` minus what
is not touched — every block weight is touched, the embedding contributes one row
(2 KB) — plus the tied logits matmul, which streams the whole `0.5086 GB` table.
So ≈ **1.55 GB/token** (§01.7). At the measured attainable bandwidth of
~210–215 GB/s (§00.5 sources) the roofline is ~**135–140 tok/s**. Two
consequences:

- the logits table is ~33 % of decode traffic; a skinny `argmax`-only GEMV that
  never materializes `logits` is a first-class optimization, not a micro-optimization
  (cf. the `lm_head_gemv` prototype in the old project);
- below ~130 tok/s the engine is losing time in scheduling or kernel
  overheads, not memory — which is exactly the regime M0–M2 wants to be in.

**Whose bandwidth is it?** 210–215 GB/s is what the platform can attain, which the iGPU
is expected to get close to; a CPU-resident path does not get it all. Measured with
`tools/bw_probe.py` (6 GiB read-only reduction, far above the 32 MiB L3): best
**~117 GB/s at 16 threads**, 75–117 GB/s across thread counts and runs — run-to-run
variance is large enough that it must be re-measured before being quoted. So the same
1.55 GB/token arithmetic caps a *CPU-resident* single-stream bf16 decode at roughly
**50–75 tok/s**, and the 135–140 figure only holds if the HIP kernels actually pull the
platform bandwidth. The first `gdn_decode`/`lm_head_gemv` kernel must therefore ship with
a HIP version of this probe; until then treat 137 tok/s as the GPU-side ceiling and
~70 tok/s as the CPU-side one.

Same arithmetic, other levers at 215 GB/s (bytes/token in parentheses): 4-bit blocks
with a bf16 logits table → 184 tok/s (1.17 GB); 4-bit everywhere → 220 (0.98); MTP at
k=1, accept 0.58 → 161 (1.34 GB per round / 1.58 tokens, §09.5); batch 8 → ~980 tok/s
aggregate at 123 tok/s per request (weights amortized, state traffic grows with B).

## 07.5 Stop / continuation policy

Greedy only in M0–M2: stop at `248046` (`<|im_end|>`, A-10), at end of context, or
at the caller's token budget. The engine appends the sampled token to the
transcript and advances `P`.

## 07.6 Reference fixtures

`tools/reference_dump.py --decode N` pins, per step `s`: `decode{s}.input_ids`,
`decode{s}.logits_last`, plus the same per-layer operand names as prefill
(`L{ll}.gdn.*`, `L{ll}.attn.*`, `L{ll}.s_in`, `L{ll}.s_out`) tagged `step=decode{s}`,
and the post-step cache contents (`decode{s}.L{ll}.*`) used by §10 G5/G7.
