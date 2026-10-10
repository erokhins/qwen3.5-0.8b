# qwen3.5-0.8b

A small, deliberately simple decode engine plus the small engine model it runs:
`Qwen/Qwen3.5-0.8B`, text-only path. The directory is named after that model;
checkpoints go in `models/<modelname>/`, so a later engine model (e.g. the
MoE-backed `Qwen3.6-35B-A3B`) becomes a sibling directory rather than a fork.
If that happens, this directory's name is kept for continuity.

This project is the "shrink the target, keep the engine honest" successor to
`../qwen3.8-flash-next-old`. That project targeted Qwen3.8-Flash-Next
(125B-A6B + n-gram PLE + QSA + Gated Residual) and stalled on a
cross-architecture **accept-rate ceiling**: token acceptance stayed at chance
(`thinking/p3d-9..13`), and the divergence was localized to the three
lab-only subsystems, not to a fixable kernel bug. Here we keep the parts of
that engine that are *shared with the whole Qwen3.5/3.6/3.8 family* and drop
the parts that exist in Flash-Next only.

## Hardware baseline

Framework Desktop, AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`), 128 GB
LPDDR5x-8000 unified memory, single NUMA node.

| quantity | value | consequence |
|---|---|---|
| OS-visible RAM | ~125.1 GiB | whole checkpoint stays resident; no streaming tricks needed |
| attainable memory bandwidth | ~210–215 GB/s | decode speed ≈ bytes read per token ÷ bandwidth |
| measured facts | see `../qwen3.8-flash-next-old/devices/framework-desktop-ryzen-ai-max-395/` | reuse, do not re-measure |

Decode ceiling for the target is bandwidth-bound, not FLOP-bound: text-only
weights ≈ 1.3 GB bf16 → ~150 tok/s theoretical, ~0.5 GB at Q4 → far above what
the runtime can schedule (i.e. at 0.8B we become **kernel/latency-bound**, which
is exactly the regime we want for bring-up).

## Target model — verified facts

From `Qwen/Qwen3.5-0.8B` (Apache-2.0, published 2026-02-28, one shard
`model.safetensors-00001-of-00001.safetensors`, 1.747 GB bf16, 488 tensors,
`model.safetensors.index.json` present). All numbers below read from the
repo's `config.json` / index, not from a blog post.

Language tower (`text_config`, `model_type: qwen3_5_text`):

| field | value |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 24 (18 `linear_attention`, 6 `full_attention`) |
| layout | `full_attention_interval: 4` → full attention at layer idx 3,7,…,23 |
| head_dim | 256 |
| attention heads | 8 Q / 2 KV (GQA), `attn_output_gate: true` |
| RoPE | `rope_theta 1e7`, `partial_rotary_factor: 0.25`, `mrope_interleaved: true`, sections `[11,11,10]` |
| linear attention (Gated DeltaNet) | 16 V heads / 16 QK heads, head dim 128, `linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: float32` |
| MLP | SwiGLU, `intermediate_size: 3584`, no bias |
| embeddings | `vocab_size: 248320`, `tie_word_embeddings: true` (~0.51 GB of the shard is `embed_tokens`) |
| MTP | `mtp_num_hidden_layers: 1` (one full-attention draft layer, shared embeddings) |
| context | `max_position_embeddings: 262144` |

Relevant tensor names (from `model.safetensors.index.json`):

```
model.language_model.embed_tokens.weight
model.language_model.layers.{0..23}.input_layernorm.weight
model.language_model.layers.{linear}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,conv1d,A_log,dt_bias,norm,out_proj}
model.language_model.layers.{full}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}
model.language_model.layers.*.mlp.{gate_proj,up_proj,down_proj}
model.language_model.layers.*.post_attention_layernorm.weight
model.language_model.norm.weight
mtp.{fc,norm,pre_fc_norm_embedding,pre_fc_norm_hidden}.weight
mtp.layers.0.{self_attn…,mlp…,input_layernorm,post_attention_layernorm}
model.visual.{patch_embed,blocks,merger,pos_embed}…        # skipped: text-only
```

Checkpoints are downloaded to `models/<modelname>/` and never committed.

```sh
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

## Why this model, and what is out of scope

Config diff across the family (key-presence check, not prose):

| subsystem | config keys | Flash-Next | 3.5-0.8B/4B/9B | 3.8-27B | here |
|---|---|---|---|---|---|
| Gated DeltaNet + short conv | `linear_*`, `linear_conv_kernel_dim` | ✅ | ✅ | ✅ | **implement** |
| Gated GQA attention | `attn_output_gate`, `output_gate_type` | ✅ (QSA-wrapped) | ✅ | ✅ | **implement** |
| MTP draft layer | `mtp_num_hidden_layers` | ✅ | ✅ | ✅ | implement (optional path) |
| MoE + shared expert | `num_experts*`, `shared_expert_*` | ✅ 512/10+1 | — | — | later (`models/Qwen3.6-35B-A3B`) |
| Gated Residual / hyperconnections | `hc_count`, `hc_lowrank` | ✅ | — | — | **not implemented** |
| QSA micro-block indexer | `indexer_*` | ✅ | — | — | **not implemented** |
| N-gram PLE embeddings | `ngram_*`, `ple_*` | ✅ (51B params) | — | — | **not implemented** |
| Vision encoder | `vision_config` | ✅ | ✅ | ✅ | **not implemented** (text-only) |

Non-goals: vision/multimodal, batched multi-user serving, vLLM API parity,
long-context YaRN experiments, and any model whose forward pass needs a
subsystem not listed as "implement" above.

## Layout

```
README.md
models/<modelname>/          # downloaded checkpoints (gitignored, read-only)
                             # i.e. qwen3.5-0.8b/models/Qwen3.5-0.8B/
specification/               # math-only, library-agnostic forward-pass spec (≤200 lines each)
engine/                      # the engine: kernels, residency, scheduler, state pools, drivers
prototype/                   # one subdir per op: standalone kernel + reference dump + gate
engines/                     # cloned reference engines for cross-checking (gitignored, read-only)
thinking/                    # gitignored per-chunk design notes (cross-session memory)
devices/                     # measured platform facts
```

Conventions inherited from the old project:

- `specification/` is math + exact tensor shapes only, no library calls. Any
  behavior derived from a reference implementation is tagged `[REF]` or
  `[ASSUMPTION A-xx]` and registered in `specification/ASSUMPTIONS.md`.
- Every `prototype/<op>/` ships with a reference-data tool, a test, and a
  `results.md`. Code graduates from `prototype/` into `engine/` only after its
  gate passes.
- Start a chunk by reading the relevant `thinking/*.md`; those notes hold
  on-hardware-verified facts and are the handoff memory. Do not re-derive them.
- Never time anything against an `-O0` build (pin `-O2`; see
  `../qwen3.8-flash-next-old/thinking/p3d-4-slow-tests.md`).

## Milestones

- **M0 — spec.** `specification/` for the text-only forward pass of
  Qwen3.5-0.8B: notation/hyperparams, weights-and-shapes, tokenization +
  embedding, GDN layer, gated GQA layer, MLP, prefill, decode step, state
  management, MTP. Plus a dumper that pins HF-transformers reference tensors.
- **M1 — per-op parity.** each op matched against the reference dumps within a
  documented tolerance, wired as a `ctest` gate suite.
- **M2 — end-to-end greedy parity.** identical greedy tokens on a pinned prompt
  set, single-stream decode, reported in tok/s on this box.
- **M3 — useful.** quantized blobs (reuse the GGUF-import lessons), MTP
  speculative decode, minimal HTTP/serving shim.
- **M4 — scale-up (only after M2).** add the MoE block and a second engine
  model at `models/Qwen3.6-35B-A3B`, which needs no subsystem beyond what
  M0–M3 already implements.

## Status

M0 in progress:

- checkpoint downloaded to `models/Qwen3.5-0.8B/` (1.747 GB, 488 tensors,
  gitignored); `tools/weights_report.py` prints the authoritative shapes and
  byte accounting straight from the safetensors header.
- `specification/` written: index + 00–10 + `ASSUMPTIONS.md`, math-only, every
  shape verified against the checkpoint and every non-obvious behavior tagged
  `[REF]`/`[ASSUMPTION A-xx]` against the HF `qwen3_5` reference.
- `tools/reference_dump.py` works: transformers 5.19 CPU reference, eager
  attention, pins 297 tensors per run into `data/refs/<tag>/` (`smoke` fixture:
  17-token chat prompt + 2 greedy decode steps, greedy continuation `The capital
  of`). This is the oracle for the §10 gates.
- `.venv/` has torch 2.14 CPU + transformers 5.19 (gitignored).

Next: G0/G1 (loader + shape gate), then per-op kernels under `prototype/`
starting with `gdn_decode` and `lm_head_gemv`, gated by §10 G3/G5.
