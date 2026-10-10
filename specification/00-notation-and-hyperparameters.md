# 00 — Notation and hyperparameters

## 00.1 Symbols

| symbol | meaning |
|---|---|
| `t`, `T` | token index inside a request, request length (tokens) |
| `d` | model width `d_model` |
| `ℝ^{a×b}` | real matrix; a vector in `ℝ^a` is a row vector |
| `x · Wᵀ` | linear map with `W ∈ ℝ^{out×in}` (checkpoint layout: rows = outputs) |
| `⊙`, `⊘` | elementwise multiply / divide, broadcasting over the last axis |
| `σ(x)` | sigmoid `1/(1+e^{-x})` |
| `silu(x)` | `x·σ(x)` |
| `softplus(x)` | `log(1+e^x)` (linear above `x > 20` is acceptable, see A-03) |
| `RMS_c(x)` | `x · rsqrt(mean(x², last axis) + ε)` with `ε = rms_norm_eps` |
| `concat_j(·)` | concatenation along the last axis, `j` ascending |
| `⌈x⌉` | ceiling |
| `S^{(l)}` | GDN recurrent state of layer `l` (§04) |
| `K^{(l)}, V^{(l)}` | key/value cache of a full-attention layer `l` (§05) |
| `γ` | per-head scalar decay factor `exp(g)` (§04.3) |

Superscript `(l)` is dropped inside a section when only one layer is meant.

## 00.2 Constants read from `config.json`

Top level: `model_type: qwen3_5`, `architectures: [Qwen3_5ForConditionalGeneration]`,
`tie_word_embeddings: true`, `image_token_id: 248056`, plus
`vision_config` / `vision_start_token_id` / `vision_end_token_id` (ignored, §00.6).

`text_config` (`model_type: qwen3_5_text`):

| key | value | use |
|---|---|---|
| `hidden_size` | 1024 | `d` |
| `num_hidden_layers` | 24 | `L` |
| `layer_types` | 24 entries, see §00.3 | block routing |
| `full_attention_interval` | 4 | `full_attention` where `l mod 4 == 3` |
| `num_attention_heads` | 8 | `n_q` |
| `num_key_value_heads` | 2 | `n_kv` |
| `head_dim` | 256 | `d_head` |
| `intermediate_size` | 3584 | `d_ff` |
| `vocab_size` | 248320 | `V` |
| `rms_norm_eps` | 1e-6 | `ε` for every norm (block, head, gated) |
| `hidden_act` | `silu` | MLP and GDN gate activation |
| `attention_bias` | false | no bias on `q,k,v,o` |
| `attention_dropout` | 0.0 | none |
| `attn_output_gate` | true | attention output gate (§05.1) |
| `linear_num_key_heads` | 16 | `h_k` |
| `linear_num_value_heads` | 16 | `h_v` |
| `linear_key_head_dim` | 128 | `d_k` |
| `linear_value_head_dim` | 128 | `d_v` |
| `linear_conv_kernel_dim` | 4 | `k_conv` |
| `mamba_ssm_dtype` | `float32` | GDN state/scan precision |
| `max_position_embeddings` | 262144 | context |
| `mtp_num_hidden_layers` | 1 | `L_mtp` (§09) |
| `mtp_use_dedicated_embeddings` | false | MTP shares `E` (§09.1) |
| `tie_word_embeddings` | true | logits head = `E` (§03.4) |
| `rope_parameters.rope_type` | `default` | plain inverse frequencies |
| `rope_parameters.rope_theta` | 1e7 | `θ` |
| `rope_parameters.partial_rotary_factor` | 0.25 | `p_rot` |
| `rope_parameters.mrope_interleaved` | true | §05.4 |
| `rope_parameters.mrope_section` | [11, 11, 10] | §05.4 (text-only: inert, see A-06) |

Not present in this checkpoint (Flash-Next/3.8-27B keys, deliberately absent):
`hc_count`, `hc_lowrank`, `indexer_*`, `ngram_*`, `ple_*`, `output_gate_type`,
`output_router_logits`, `num_experts*`, `shared_expert_intermediate_size`,
`mlp_only_layers: []`.

## 00.3 Derived sizes

| name | formula | value |
|---|---|---|
| GDN key width | `d_k·h_k` | 2048 |
| GDN value width | `d_v·h_v` | 2048 |
| GDN conv channels | `2·d_k·h_k + d_v·h_v` | 6144 |
| GDN gate width | `d_v·h_v` | 2048 |
| GDN head ratio | `h_v / h_k` | 1 (no key repeat) |
| attn q width | `n_q·d_head` | 2048 |
| attn q_proj width | `n_q·d_head·2` (gate) | 4096 |
| attn kv width | `n_kv·d_head` | 512 |
| GQA group | `n_q / n_kv` | 4 |
| rotary width | `d_head·p_rot` | 64 |
| rotary pairs | `rotary width / 2` | 32 |
| mrope pairs per section | `[11, 11, 10]` | sums to 32 |
| attention scale | `d_head^{-0.5}` | 1/16 |
| GDN l2 q scale | `d_k^{-0.5}` | 128^{-0.5} (§04.4, A-05) |

Layer routing (`layer_types`, 0-based) — `[MEASURED]` from the checkpoint's
tensor names, not just the config list:

- `linear_attention` (GDN, §04) at
  `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22` → 18 layers
- `full_attention` (§05) at `3,7,11,15,19,23` → 6 layers

Inverse frequencies (used by both attention block and MTP): for `i = 0..31`,
`f_i = θ^{-(2i)/64}`; `θ = 1e7`. Frequencies are computed in F32 `[REF]`.

## 00.4 Precision classes

| class | contents |
|---|---|
| `W_bf16` | all checkpoint weights except the 36 F32 tensors of §01.6 |
| `X_bf16` | activations carried between blocks and through residuals |
| `F32-only` | inside `RMS_c` (variance and the affine multiply), softmax, the GDN scan and its state, the GDN gate/decay computation |

HF-ref casts back to the input dtype after each norm and after each GEMM; the
engine must declare per-op whether it fuses (see A-01 for norm cast order).

## 00.5 State sizes per token / per request

`B` = bytes. See §08 for the full inventory and lifecycle.

| item | formula | value |
|---|---|---|
| KV cache per token | `2·n_kv·d_head·L_full·2 B` | 12288 B (12 KiB) |
| KV cache at 262144 ctx | above × 262144 | 3.07 GiB |
| GDN state per request | `L_lin·h_v·d_k·d_v·4 B` | 18.0 MiB |
| GDN conv state per request | `L_lin·6144·(k_conv−1)·2 B` | 0.63 MiB |
| MTP KV per token | `2·n_kv·d_head·L_mtp·2 B` | 2048 B |
| resident text weights | §01.7 | 1.5457 GB (bf16) |

## 00.6 Explicitly out of scope constants

`vision_config` (`depth 12`, `hidden_size 768`, `out_hidden_size 1024`,
`patch_size 16`, `spatial_merge_size 2`, `temporal_patch_size 2`) and the
vision special ids exist in the checkpoint. The engine does not load them and
does not accept image/video input; the chat template must be evaluated only for
text-only messages (§02.3).
