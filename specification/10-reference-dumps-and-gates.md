# 10 — Reference dumps and parity gates

## 10.0 Fixtures

`tools/reference_dump.py` (project venv: torch CPU + transformers 5.19 with
`qwen3_5`) writes `data/refs/<tag>/` (gitignored): one F32 `.npy` per pinned
tensor + `manifest.json` (config, prompt ids, chat renderings, special ids,
greedy continuation, per-pin shape/dtype/step/note). Reference model runs
BF16, eager attention, single sequence, greedy.

The pinned names are exactly the ones referenced by §02–§07, e.g.
`inputs_embeds`, `L07.attn.q_rope`, `L04.gdn.o_pre_norm`, `final_hidden`,
`logits_last`, `decode0.logits_last`.

Canonical fixture set (to be regenerated and pinned by commit hash of the
checkpoint dir + tool version):

| tag | content |
|---|---|
| `smoke` | one chat-rendered prompt, 2 decode steps (bring-up) |
| `p8`, `p64`, `p300` | raw prompts of those lengths: chunk-boundary coverage (8 < 64 = chunk size, 64, 300 ⇒ padding to 320) |
| `prefix24+dec8` | prefill 24 then 8 decode steps ⇒ §08.6 invariant 1 |
| `corpus512` | 512 short prompts for accept-rate and throughput reporting |

## 10.1 Gates

Tolerances are relative-to-reference-F32 unless noted; `cos` = cosine similarity
over the flattened tensor, `maxrel` = max |Δ| / (max |ref| + 1e-30). Engine
outputs BF16 unless the spec pins F32.

| gate | statement | threshold |
|---|---|---|
| **G0 loader** | every checkpoint tensor loaded or explicitly skipped (`model.visual.*` = 153); count = 488, loaded = 335 | exact |
| **G1 shapes** | every pinned reference tensor matches §01–§05 shapes | exact |
| **G2 block glue** | `L{ll}.s_out` for all 24 layers, prefill | cos ≥ 0.999, meanrel ≤ 5 % |
| **G3 GDN scan** | `L{ll}.gdn.o_pre_norm` + `state_new` for all 18 GDN layers, at `p8/p64/p300`; plus chunk-vs-recurrent self-check inside the engine | cos ≥ 0.999, meanrel ≤ 5 %; chunk==recurrent within 1e-5 F32 |
| **G4 attention** | `L{ll}.attn.q_rope`, `k_rope`, `attn_out_pre_gate`, `out` for all 6 layers. A `q`/`gate` de-interleave error (§05.1) must fail here with cos ≪ 1 — the gate is required to actually discriminate | cos ≥ 0.999, meanrel ≤ 5 %; de-interleave-negative test must fail |
| **G5 incremental == batch** | prefill `T` then decode `D` equals one prefill of `T+D` (states, logits, sampled ids) | sampled ids exact; logits cos ≥ 0.9995 |
| **G6 running-softmax** | if running `(max,sum)` softmax accumulators are used (§07.3), tokens identical to full-prefix softmax | sampled ids exact on `corpus512` |
| **G7 end-to-end greedy** | token-for-token match with HF-ref on `smoke`, `p8`, `p64`, `p300`, `prefix24+dec8`, then ≥ 95 % prefix agreement on `corpus512` before any perf claim | exact / 95 % |
| **G8 MTP** | §09.5 sequence: shapes, single-step agreement, measured accept rate | reported, no hard bound |
| **G9 perf** | decode tok/s on this box vs the §07.4 roofline (~135 tok/s); `-O2` build only | report; investigate below 60 % |

## 10.2 Metrics and current drift

`cos` = cosine similarity over the flattened tensor; `meanrel` = mean|Δ| /
mean|ref|; `maxrel` = max|Δ| / max|ref| is reported but not gating (one bf16 tail
element in the gated norm dominates it). The binding correctness signal is G7
(exact token match) plus cos ≥ 0.999 everywhere; the meanrel ceiling exists to
catch a regression in drift, not to assert bit fidelity. With the recurrent prefill
of §04.4 against HF's chunked form, measured drift on the `smoke` fixture is
meanrel 8e-6 at layer 0 rising to 3.3 % at layer 21 (`cos` stays ≥ 0.9992).
**Tighten meanrel to 1 % when the chunked GDN prefill (§04.7) lands**, and record
the change here and in `engine/gates/run_gates.py`.

## 10.3 Rules

- Every gate is a checked-in binary/script run by `ctest` (or a `run_all.sh` per
  `prototype/<op>/`), never a manual eyeball.
- A gate that passes on `smoke` only is **not** passing: `p64`/`p300` are required
  because of the chunk-64 padding in §04.7.
- Reference dumps are regenerated whenever the tool, transformers version, or
  checkpoint changes; the `manifest.json` fields (`transformers`, `torch`,
  `dtype`) are part of the fixture identity.
- Never trust byte values from ad-hoc dumps in the old project
  (`../qwen3.8-flash-next-old/prototype/*/data`): shapes only, per its own
  `REFERENCE.md`.
