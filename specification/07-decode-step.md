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

`mask` allows `j ≤ P`. During decode the softmax is over `P+1` lanes.

**Which form the engine uses — corrected, and the old claim is named rather than hidden.** This
paragraph used to say that computing the softmax as one `softmax` or as a blocked loop with `(max,
sum)` rescaling (*online* softmax, the reduction a flash-decoding / split-K kernel uses) was pinned
by §10 G6 to "the two agree". **They do not agree bit-for-bit, and no tile size makes them.** §05.5
rounds the *normalised* weight row to the activation dtype ("computed in F32, then cast") before
`a = Σ w·v`, and a streaming reduction cannot round a weight it has not divided yet. Measured on
the §05 decode pins — 6 attention layers × 8 decode steps of `dec8`, `prototype/attn_decode/run.py
--long`:

| decode softmax form | `attn_out_pre_gate` vs the pins | large elements (>1 ULP) | cos | meanrel |
|---|---|---|---|---|
| one `softmax`, weights cast to BF16 | **0 of 98 304 elements differ** | 0 | 1.00000000 | 0 |
| blocked `(max, sum)` | 25 745 differ (26.2 %) | 6 098 (6.2 %) | ≥ 0.99999134 | ≤ 1.8e-3 |

Over all 53 cases including the long-cache rows the streaming form moves 25.9 % of elements at cos
0.99999651, meanrel 1.58e-3 (`prototype/attn_decode/floor_probe.py`, F2) — either way it lands
outside the band this op's gate uses, while the one-pass form is bit-exact.

**Decision: the engine's decode path uses the one-pass form.** Bit-exactness against the pins is the
strongest gate available in this project, and the second pass is cheap here: one query row's scores
are `(P+1) × 8 heads × 4 B` = 8.9 KiB at the longest pinned cache (277 lanes), which is one block's
shared allocation and a rounding error against this box's 2 MiB L2 (`devices/`).

The blocked form is not banned — it is the shape a long-context path has to take (W7 split-K /
flash-decoding, where a row no longer fits one block), and it stays available in the oracle
(`specimpl.attention(softmax="online")`) and in the prototype (`--softmax online`). When that path
is built it must be gated on a **cos/meanrel band** — §10 G6's thresholds are already of that shape
— and not on agreement with the pins, which it cannot reach.

What G6 really pins is unchanged and was never the thing that failed: the `(max, sum)` pair is
reduction state of a single query row and is **never carried into the next decode step** (A-17).
Carrying it produced attention cos 0.876 after one step, matching the *previous* prefix instead of
the current one.

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

- the logits table is ~33 % of decode traffic. **Corrected after measurement**
  (`prototype/lm_head_gemv/results.md`): on the GPU a fused `argmax`-only GEMV is worth
  ~1 %, because writing 1 MB of logits is nothing next to reading 485 MB — the table's
  cost is the *read*, which no output form avoids. Fusing `argmax` is still worth doing,
  for the reasons in §09/§07.5: greedy never needs the logits (A-17), and it removes a 1 MB
  host copy plus a host-side argmax per token. On the *CPU* path the fused form remains a
  first-class optimization, since there the logits materialization is real compute.
- below ~130 tok/s the engine is losing time in scheduling or kernel
  overheads, not memory — which is exactly the regime M0–M2 wants to be in.
  (Measured ceiling is ~151 tok/s, see below: this threshold stays deliberately below
  the roofline so missing it means a real bug, not ordinary inefficiency.)

**Measured, both sides** (`devices/strix-halo-memory-bandwidth.md`; probes in `tools/`).
The iGPU and the CPU share one LPDDR5x pool, so these are alternatives, not additive:

| path | pure read | copy (r+w) | 1.565 GB/token ceiling |
|---|---|---|---|
| iGPU (`tools/bw_probe_hip.cpp`) | **237 GB/s** (233–240 over a block×thread sweep; 235 at 4 GiB) | 210 GB/s | **~151 tok/s** |
| CPU (`tools/bw_probe.py`) | 75–117 GB/s | 59–101 GB/s | ~50–75 tok/s |
| iGPU H2D / D2H | 76–82 / 63–70 GB/s | — | staging paths are bounded here |

So the single-stream bf16 ceiling is **~151 tok/s**, not the 135–140 originally estimated:
the old 210–215 GB/s figure behaves like a copy/STREAM number, while the access pattern
a decode step actually has (stream weights once, read-only) reaches ~237 GB/s. The device
reports 20 CUs and a 2 MiB L2, so nothing about a 1.5 GB weight stream is
cache-friendly and the roofline really is bandwidth. A CPU-resident engine would have
capped near 50–75 tok/s, which is a second argument for the GPU path on top of the
compute one.

Two measured footnotes, both from `prototype/` kernels rather than from the probe:

- **Not everything is uncached.** A ~18 MiB working set — all 18 GDN recurrent states, the
  only state a decode step touches — behaves as if cached: the same kernel does 918 GB/s at
  18 MiB and 255 GB/s at 108 MiB. State traffic therefore costs 0.06 ms warm (0.22 ms if it
  were cold), not a surprise. See `devices/strix-halo-memory-bandwidth.md` §0b.
- **Kernel dispatch adds a floor of ~0.36 ms/token.** A kernel costs ~1.8 µs to launch even
  if it does nothing, and **capturing the launches into a HIP graph does not reduce it**
  (1.78 µs/kernel replayed, 1.0×) — the limit is GPU dispatch, not host submission. At ~200
  kernel issues per decode step that is 6.5 ms of bandwidth + ~0.36 ms of dispatch, so the
  honest band is **145–153 tok/s** and the lever is fusing the small ops into the GEMV
  epilogues, not graph capture. Probe: `tools/launch_overhead_hip.cpp`.

Same arithmetic, other levers at 237 GB/s (bytes/token in parentheses): 4-bit blocks with
a bf16 logits table → 202 tok/s (1.17 GB); 4-bit everywhere → 243 (0.98); 3-bit → 283
(0.84); MTP k=1 at accept 0.58 → 181 (1.31); MTP k=2 at accept 0.41 → 162 (1.46); batch
8 → ~1080 tok/s aggregate at ~135 per request (weights amortized, state traffic grows
with B); 32k context costs ~30 tok/s (KV adds 0.4 GB/token).

## 07.5 Stop / continuation policy

Greedy only in M0–M2: stop at `248046` (`<|im_end|>`, A-10), at end of context, or
at the caller's token budget. The engine appends the sampled token to the
transcript and advances `P`.

## 07.6 Reference fixtures

`tools/reference_dump.py --decode N` pins, per step `s`: `decode{s}.input_ids`,
`decode{s}.logits_last`, plus the same per-layer operand names as prefill
(`L{ll}.gdn.*`, `L{ll}.attn.*`, `L{ll}.s_in`, `L{ll}.s_out`) tagged `step=decode{s}`,
and the post-step cache contents (`decode{s}.L{ll}.*`) used by §10 G5/G7.
