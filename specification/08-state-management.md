# 08 — State management

Per-request mutable state, its size, and its lifecycle. Weights are read-only and
not part of this inventory.

## 08.1 Inventory

Per request, with `L_lin = 18`, `L_full = 6`, `L_mtp = 1`, `n_kv = 2`,
`d_head = 256`, `h_v = 16`, `d_k = d_v = 128`, `conv_ch = 6144`, `k_conv = 4`:

| state | shape | dtype | size | count |
|---|---|---|---|---|
| GDN recurrent `S^{(l)}` | `[16][128][128]` | F32 | 1.0 MiB | 18 → **18.0 MiB** |
| GDN conv window | `[6144][3]` | BF16 | 36 KiB | 18 → **0.63 MiB** |
| KV `K^{(l)}, V^{(l)}` | `[P][2][256]` each | BF16 | 4 KiB/token/layer | 6 → **12 KiB/token** |
| MTP KV (only while drafting, §09) | `[P][2][256]` | BF16 | 4 KiB/token | 1 → **4 KiB/token** |
| scalars | `prefix_len`, sampled token, stop flags | — | — | 1 |

Constant per request ≈ 18.6 MiB; linear part 16 KiB/token. At the full
`262144` context: 3.07 GiB KV + 18.6 MiB ≈ **3.09 GiB per request**. On this box
(125 GiB usable, 1.55 GB resident weights) hundreds of concurrent long requests
fit — capacity is not the constraint, prefix reuse and scheduling are.

## 08.2 Semantics of each buffer

GDN recurrent `S^{(l)}`: defined by §04.4; the *only* memory a GDN layer has of
the past. Written wholesale at prefill end, updated in place per token at decode.
Must stay F32 (`mamba_ssm_dtype: float32`): a BF16 `S` loses ~3 decimal digits on
a quantity that is repeatedly decayed and re-written, and §10 G3/G5 would fail.

GDN conv window: the last `k_conv − 1 = 3` **pre-conv** `raw` columns (§04.2).
Roll order matters (§07.2): read the window, then write `raw`. Zero-initialized
at request start, which is what makes prefill and decode agree at the boundary.

KV: post-`k_norm`, post-rope keys and raw values (§05.7), indexed by global
position `0..P`. Appends only. Never store pre-rope keys: the rope position is
fixed at append time, which is what allows a cached prefix to be reused (§08.3).

## 08.3 Prefix reuse

Because §05 caches post-rope `K`, a prefix cache hit requires the cached prefix
to have been computed with the **same** `prefix_len` alignment, and because §04
state is a single matrix per layer, reuse is at GDN-segment granularity: a shared
prefix can be reused only if the cached `S`/conv window corresponds to exactly the
shared token span. Practical rule for the engine: state the cache key as
`(shared prefix token ids)` and store the whole §08.1 tuple under it — partial
reuse of GDN state without the matching span is silently wrong (the state is not
decomposable), so the granularity of reuse is the whole prefix or nothing.

## 08.4 Lifecycle

```
new request   → allocate §08.1 tuple; zero S and conv window; P = 0
prefill       → write S, conv window, KV[0..T−1]; P = T          §06
decode step   → update S, conv window in place; append KV[P]; P += 1   §07
preempt       → snapshot the tuple (contiguous copy) to host storage
resume        → restore tuple verbatim; no recompute needed
finish        → keep or drop the prefix entry (§08.3) per policy
```

No state is derived from the transcript at decode time, so preemption is a pure
memcpy of 18.6 MiB + `16 KiB × P`.

## 08.5 Layout requirements (engine)

- `S` per layer: one contiguous `[16][128][128]` F32 block, head-major — the
  recurrence touches it once per token and it must stay one DMA-friendly region.
- conv windows of all 18 layers in one slab (`[18][6144][3]`) so a prefill can
  write them with one strided pass.
- KV: per layer `[slots][2][256]` with slot indices from an allocator; a slot is
  never rewritten except by a discarded draft (§09).
- Alignment: keep `S` 64-B aligned and separate from BF16 tensors; mixing dtypes
  in one cache line hurts both the F32 scan and BF16 GEMV.

## 08.6 Invariants to assert in tests

1. After prefill of `T` tokens then `D` decode steps, the state equals the state
   after prefill of `T + D` tokens fed as one pass (§10 G5).
2. Zero-initialized conv window + prefill of `T` tokens ⇒ window equals
   `raw[max(0,T−3)..T−1]`.
3. `S` and conv window are the only GDN writes; `K,V` appends never overwrite a
   position `< P`.
4. `prefix_len` equals the KV length after every step.
