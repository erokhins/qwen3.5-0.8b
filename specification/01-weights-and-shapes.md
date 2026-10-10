# 01 — Weights and shapes

Source: safetensors header of `models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors`,
reported by `tools/weights_report.py` (`[MEASURED]`). 488 tensors, 1.7469 GB.
Shapes are as stored: a linear weight `W ∈ ℝ^{out×in}`, so the forward map is
`y = x · Wᵀ`. No bias tensors exist in the text path.

Naming: the checkpoint prefixes the language tower with `model.language_model.`
because the outer class is `Qwen3_5ForConditionalGeneration`. Names below are
written with that prefix in full; the engine may normalize it (§02.5, A-09).

## 01.1 Token embedding (shared)

| name | dtype | shape | bytes |
|---|---|---|---|
| `model.language_model.embed_tokens.weight` | BF16 | `[248320, 1024]` | 508.56 MB |

`lm_head.weight` does **not** exist: `tie_word_embeddings: true`, so the logits
head reuses this matrix (§03.4). This single tensor is 32.9 % of the shard.

## 01.2 `linear_attention` block (GDN, §04) — layers 0,1,2,4,…,22

14 tensors, 43.11 MB each, 18 layers, 0.7760 GB total.

| name (after `model.language_model.layers.{l}.`) | dtype | shape | role |
|---|---|---|---|
| `input_layernorm.weight` | BF16 | `[1024]` | zero-centered RMSNorm (§03.1) |
| `linear_attn.in_proj_qkv.weight` | BF16 | `[6144, 1024]` | mixed qkv (§04.1) |
| `linear_attn.in_proj_z.weight` | BF16 | `[2048, 1024]` | output gate (§04.5) |
| `linear_attn.in_proj_b.weight` | BF16 | `[16, 1024]` | β pre-activation, per value head |
| `linear_attn.in_proj_a.weight` | BF16 | `[16, 1024]` | decay pre-activation, per value head |
| `linear_attn.conv1d.weight` | BF16 | `[6144, 1, 4]` | causal depthwise conv, tap axis last |
| `linear_attn.A_log` | **F32** | `[16]` | per value head (§04.3) |
| `linear_attn.dt_bias` | BF16 | `[16]` | per value head (§04.3) |
| `linear_attn.norm.weight` | **F32** | `[128]` | gated RMSNorm, shared across heads (§04.5) |
| `linear_attn.out_proj.weight` | BF16 | `[1024, 2048]` | output projection |
| `post_attention_layernorm.weight` | BF16 | `[1024]` | zero-centered RMSNorm |
| `mlp.gate_proj.weight` | BF16 | `[3584, 1024]` | §03.3 |
| `mlp.up_proj.weight` | BF16 | `[3584, 1024]` | §03.3 |
| `mlp.down_proj.weight` | BF16 | `[1024, 3584]` | §03.3 |

There is no `linear_attn.conv1d.bias`, no `in_proj_*.bias`, and no per-head
weight copy: `linear_attn.norm.weight` is one `[128]` vector applied to every
value head.

## 01.3 `full_attention` block (§05) — layers 3,7,11,15,19,23

11 tensors, 36.71 MB each, 6 layers, 0.2202 GB total.

| name (after `model.language_model.layers.{l}.`) | dtype | shape | role |
|---|---|---|---|
| `input_layernorm.weight` | BF16 | `[1024]` | |
| `self_attn.q_proj.weight` | BF16 | `[4096, 1024]` | q **and** output gate, interleaved per head (§05.1) |
| `self_attn.k_proj.weight` | BF16 | `[512, 1024]` | 2 KV heads × 256 |
| `self_attn.v_proj.weight` | BF16 | `[512, 1024]` | |
| `self_attn.q_norm.weight` | BF16 | `[256]` | zero-centered RMSNorm over `d_head` |
| `self_attn.k_norm.weight` | BF16 | `[256]` | |
| `self_attn.o_proj.weight` | BF16 | `[1024, 2048]` | applied after the output gate |
| `post_attention_layernorm.weight` | BF16 | `[1024]` | |
| `mlp.{gate_proj,up_proj,down_proj}.weight` | BF16 | as §01.2 | |

The `q_proj` row block is `[8 heads][2 × 256]`: rows `h·512 … h·512+255` are
`q` of head `h`, rows `h·512+256 … h·512+511` are its gate (§05.1) `[REF]`.

## 01.4 Final norm

| name | dtype | shape |
|---|---|---|
| `model.language_model.norm.weight` | BF16 | `[1024]` |

## 01.5 MTP block (§09) — `mtp.*`

15 tensors, 40.91 MB total.

| name | dtype | shape |
|---|---|---|
| `mtp.pre_fc_norm_embedding.weight` | BF16 | `[1024]` |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | `[1024]` |
| `mtp.fc.weight` | BF16 | `[1024, 2048]` |
| `mtp.layers.0.input_layernorm.weight` | BF16 | `[1024]` |
| `mtp.layers.0.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight` | BF16 | as §01.3 |
| `mtp.layers.0.{post_attention_layernorm}.weight` | BF16 | `[1024]` |
| `mtp.layers.0.mlp.{gate_proj,up_proj,down_proj}.weight` | BF16 | as §01.2 |
| `mtp.norm.weight` | BF16 | `[1024]` |

The MTP layer is a `full_attention` block: same geometry as §01.3, no GDN, and
no dedicated embedding (`mtp_use_dedicated_embeddings: false`).

## 01.6 The 36 F32 tensors

Exactly `linear_attn.A_log` (16 elements) and `linear_attn.norm.weight`
(128 elements) in each of the 18 GDN layers: 10 KB total.

HF-ref loads the whole checkpoint into one dtype, so with
`dtype=bfloat16` these two tensors are rounded to BF16 at load `[REF]`. The
engine keeps them in F32 and documents the difference in A-02.

## 01.7 Byte accounting

| group | bytes |
|---|---|
| 18 GDN layers | 0.7760 GB |
| 6 full-attention layers | 0.2202 GB |
| shared embedding (tied) | 0.5086 GB |
| MTP block | 0.0409 GB |
| `model.language_model.norm` | 2 KB |
| **text-only total** | **1.5457 GB** |
| `model.visual.*` (not loaded) | 0.2012 GB |
| shard total | 1.7469 GB |

Decode traffic implication (used by §07.5): per token the engine reads every
text weight **except** the embedding rows it does not touch — one embedding row
for the input token, but the whole tied table again for the logits (`logits =
h · Eᵀ`). First-order bytes/token ≈ `1.5457 − 0.5086 + 0.5086 + 0.002 ≈ 1.55 GB`.

## 01.8 Not loaded

`model.visual.{patch_embed,blocks.0..11,merger,pos_embed}` (144 tensors,
201.19 MB) and `preprocessor_config.json` / `video_preprocessor_config.json`.
Any loader must accept and skip them (A-09).
