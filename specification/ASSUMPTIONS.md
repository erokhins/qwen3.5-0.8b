# ASSUMPTIONS

Every `[REF]`/`[ASSUMPTION]` used by `specification/`. Sources: HF-ref =
`transformers 5.19.0` `models/qwen3_5/modeling_qwen3_5.py`; vLLM-ref =
`../qwen3.8-flash-next-old/engines/vllm/vllm/model_executor/models/qwen3_5*.py`;
ckpt = this checkpoint's `config.json` + safetensors header.

| id | statement | source | risk | falsified by |
|---|---|---|---|---|
| **A-01** | Norm cast order: F32 normalize → multiply affine in F32 → cast back to activation dtype (§03.1); gated norm normalizes in F32, casts the normalized value back, then applies `w_norm` and `silu(z)` in F32 (§04.5) | HF-ref `Qwen3_5RMSNorm.forward`, `Qwen3_5RMSNormGated.forward` | low — rounding only | G2/G3 maxrel blows past 2 % when fused differently |
| **A-02** | `linear_attn.A_log` and `linear_attn.norm.weight` are stored F32 (36 tensors, §01.6); HF-ref rounds them to the model dtype at load, the engine keeps them F32 | ckpt header; HF `from_pretrained(dtype=bf16)` behavior | low | G3 with both variants; A/B dump diff |
| **A-03** | `softplus(x) = log1p(e^x)`, computed in F32, may fall back to `x` for `x > 20` (§04.3) | standard, HF uses `F.softplus` default β=1, threshold=20 | low | G3 at `p300` |
| **A-04** | GDN conv window stores **pre-conv** `raw` values, 3 columns, zero-initialized (§04.2, §07.2, §08.2) | HF-ref `causal_conv1d_update` (`cat([conv_state, x])` before conv) | medium — silent if wrong | G5 (incremental == batch) fails immediately |
| **A-05** | GDN applies `d_k^{-1/2}` to `q̃` always and nothing to `k̃`; `l2norm` uses `rsqrt(Σx² + 1e-6)` with ε **inside** the root (§04.4) | HF-ref `torch_*_gated_delta_rule` (`scaling = d^-0.5`), `l2norm` | medium | G3; a no-scale engine still passes cos but drifts |
| **A-06** | Text-only ⇒ mrope (`mrope_interleaved`, `mrope_section [11,11,10]`) collapses to plain rope on the first 64 head dims (§05.4, §02.4); three position streams are identical | HF-ref `Qwen3_5TextRotaryEmbedding.forward` expands to 3 identical streams for text | low for text-only, high if vision added | G4; vision input (out of scope) |
| **A-07** | Dot-product/MLP reductions accumulate F32; activations stay BF16 between ops (§03.3, §05.5) | engine policy (HF uses torch defaults) | low | G2/G9 A/B |
| **A-08** | Prefill chunk form with zero-`β` padding to a multiple of 64 is exactly equivalent to the recurrence for real tokens (§04.7, §06.2) | HF-ref `torch_chunk_gated_delta_rule` padding | medium | G3 self-check (chunk vs recurrent) |
| **A-09** | Loader name map `model.language_model.* → lm.*`, skips `model.visual.*` (144 tensors) and accepts `model.safetensors.index.json` with its unusual shard filename (§02.5, §01.8) | ckpt index | low | G0 |
| **A-10** | EOS: `tokenizer.eos_token = <|im_end|> (248046)` while `text_config.eos_token_id = 248044` (); the engine stops on `248046` and treats `248044` as legacy end-of-text (also accepted as stop) (§02.1) | `tokenizer_config.json` vs `config.json` | medium — wrong stop truncates or rambles | G7 token-for-token on `corpus512` |
| **A-11** | Chat rendering is re-tokenized from the rendered string with special tokens preserved, `add_special_tokens = false` (§02.2) | `tokenizer_config.add_bos_token = false`, template output | medium — id-sequence mismatch kills parity | `manifest.chat_template_render_*` vs engine renderer |
| **A-12** | MTP consumes the target hidden state **before** the final norm (§09.1) | vLLM-ref drafter wiring (drafter owns `mtp.norm`); no HF-ref exists for `qwen3_5` MTP | **high** — no authoritative reference | §09.5 step 2/3: try pre-norm vs post-norm `h`, whichever gives non-chance accept rate |
| **A-13** | MTP `fc` concat order is `[embedding ‖ hidden]` (§09.2) | vLLM-ref `torch.cat([inputs_embeds, hidden_states])` | medium | §09.5 step 2 |
| **A-14** | Attention output gate is plain `σ` applied before `o_proj`; no `output_gate_type` key exists here, so no SiLU variant (§05.6) | ckpt (key absent), HF-ref hardcodes `torch.sigmoid` | low | G4 |
| **A-15** | `q_proj` interleaves `[q ‖ gate]` **per head** (§05.1) | HF-ref `chunk(view(T,−1,head_dim*2), 2, −1)` | **high** if misread | G4 negative test (G4 requires it to fail when de-interleaved globally) |
| **A-16** | MTP is out of scope for M0–M2 correctness (G7 measured without drafting); HF-ref cannot run it at all | HF has no `qwen3_5` MTP class | — | §09.5 only |
