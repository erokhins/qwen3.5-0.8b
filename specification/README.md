# Specification index

Math-only, library-agnostic definition of the **Qwen3.5-0.8B text-only forward
pass** (prefill + decode + MTP draft). No library calls, no kernel names, no
engine code. Every tensor shape is exact.

Ground truth for this directory, in precedence order:

1. `models/Qwen3.5-0.8B/config.json` + the safetensors header (`tools/weights_report.py`).
2. `transformers 5.19.0`, `transformers/models/qwen3_5/modeling_qwen3_5.py`
   (the checkpoint's own reference implementation; hereafter **HF-ref**).
3. `engines/vllm/vllm/model_executor/models/qwen3_5{,_mtp}.py` and
   `layers/mamba/gdn/qwen_gdn_linear_attn.py` (second opinion; hereafter **vLLM-ref**).
4. Verified notes in `../qwen3.8-flash-next-old/specification/` — same GDN and
   gated-attention family, different head geometry and none of the
   Flash-Next-only subsystems.

## Tag conventions

- `[REF]` — behavior not stated in `config.json`, derived from reading a
  reference implementation. Name the source and file:line in `ASSUMPTIONS.md`.
- `[ASSUMPTION A-xx]` — a decision the engine makes where references disagree
  or are silent; registered in `ASSUMPTIONS.md` with source, risk and the test
  that would falsify it.
- `[MEASURED]` — number obtained by running on this box or on the checkpoint.

## Files

| file | contents |
|---|---|
| `00-notation-and-hyperparameters.md` | symbols, all constants, derived sizes, dtype policy |
| `01-weights-and-shapes.md` | every checkpoint tensor: name, shape, dtype, role |
| `02-tokenization-and-embedding.md` | tokenizer, chat template, special ids, embedding lookup, positions |
| `03-block-glue-and-normalization.md` | the two RMSNorm variants, decoder-layer skeleton, logits head |
| `04-gdn-linear-attention.md` | Gated DeltaNet block (18 layers) |
| `05-gated-full-attention.md` | gated GQA block with partial mrope (6 layers) |
| `06-prefill-pass.md` | whole-request pass, including state initialization |
| `07-decode-step.md` | single-token step, incremental state updates |
| `08-state-management.md` | state inventory, sizes, lifecycle, reuse, preemption |
| `09-mtp-draft-layer.md` | MTP block used as speculative drafter |
| `10-reference-dumps-and-gates.md` | what `tools/reference_dump.py` pins and the parity gates |
| `ASSUMPTIONS.md` | every `[REF]`/`[ASSUMPTION]` with source, risk, falsifying test |

## Scope

In: text-only prefill and decode, greedy sampling, the recurrent/KV state model,
MTP drafter math.

Out (no spec, no code): vision tower (`model.visual.*`, 201 MB of the shard),
batched/paged multi-request scheduling, YaRN/long-context extension, sampling
beyond greedy, and the three Flash-Next-only subsystems (Gated Residual,
QSA indexer, n-gram PLE) which this checkpoint does not contain.

## Numerical policy

Arithmetic is defined over real numbers here; precision classes are engine
decisions recorded per-op. Defaults fixed by `config.json`/HF-ref:

- GDN recurrent state and its scan: **F32** (`mamba_ssm_dtype: float32`).
- GDN `A_log`, GDN output `norm.weight`: stored **F32** in the checkpoint (36 tensors).
- Everything else: **BF16** weights, BF16 activations, F32 reduction inside
  norms and softmax `[REF]`.
