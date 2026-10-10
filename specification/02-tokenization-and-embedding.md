# 02 — Tokenization, chat rendering, embedding, positions

## 02.1 Tokenizer

`tokenizer_class: Qwen2Tokenizer`, BPE, `model_max_length: 262144`,
`add_bos_token: false`, no `bos`/`unk` token, `pad_token = <|im_end|>`. `E` has
`V = 248320` rows, 26 of them added special tokens at ids `248044 … 248069`
`[MEASURED]`.

| token | id | role |
|---|---|---|
| `<|endoftext|>` | 248044 | legacy end-of-text; equals `text_config.eos_token_id` (A-10) |
| `<|im_start|>` | 248045 | turn start |
| `<|im_end|>` | 248046 | turn end; `tokenizer.eos_token` |
| `<|im_start|>` | 248068 | thinking-block start |
| `<|im_end|>` | 248069 | thinking-block end |

Vision/tool ids `248047 … 248067` exist in the table and are out of scope (§00.6).
Stop condition for the engine: emit until `248046` (see A-10 for the `248044`
ambiguity).

## 02.2 Chat rendering (text-only subset of `chat_template.jinja`)

For messages `m` with text-only `content`, `add_generation_prompt = true`,
thinking **off**:

```
for m in messages:                     # concatenated with no extra separators
    "<|im_start|>" + role(m) + "\n" + content(m) + "<|im_end|>\n"
then, if add_generation_prompt:
    "<|im_start|>assistant\n"
     + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")
```

`enable_thinking` is exactly this trailing branch `[MEASURED: chat_template.jinja
lines ~149-152]`; the `preserve_thinking`/`reasoning_content` branches only matter
when a prior assistant turn carries reasoning and are out of scope for M0–M2.
The engine stores the rendered string and re-tokenizes it (A-11): the
tokenization of the rendered string must be reproduced token-exactly, including
that special tokens are not merged with adjacent text.

`tools/reference_dump.py --message` writes the two renderings (thinking on/off)
into `manifest.json` (`chat_template_render_probe`, `chat_template_render_think`)
as the pinned fixtures.

## 02.3 Embedding

```
x0[t] = E[ input_ids[t] ]      ∈ ℝ^{1024},  BF16 rows, no scaling, no offset
```

No embedding ×√d scaling, no dropout, no learned position embedding.

## 02.4 Positions

Text-only: `p[t] = t + prefix_len` (0-based, one stream). HF-ref expands positions
to four streams (`text, temporal, height, width`) and uses `position_ids[0]` as
the text stream while the rotary module consumes the three vision streams
`[REF]`; for pure text all four are `arange + prefix_len`, so the engine needs one
stream (A-06). Position values are integers, converted to F32 only inside the
rope product (§05.3). Maximum context `262144`; no YaRN (§00 scope).

## 02.5 Name normalization for the loader

The checkpoint prefixes the language tower with `model.language_model.` and the
MTP block with `mtp.`; the vision subtree is `model.visual.*`. The loader maps

```
model.language_model.*  → lm.*
mtp.layers.0.*          → mtp.layer0.*
model.visual.*          → skipped (A-09)
```

and must fail loudly on any tensor it neither loads nor skips (§10 G0).
