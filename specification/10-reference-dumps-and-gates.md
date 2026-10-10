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
| **G3 GDN scan** | `L{ll}.gdn.o_pre_norm` + `state_new` for all 18 GDN layers, at `p8/p64/p300`; plus chunk-vs-recurrent self-check inside the engine | cos ≥ 0.999, meanrel ≤ 5 %; chunk==recurrent cos ≥ 0.9999 |
| **G3i isolated per-layer** | feed the reference's own `L{ll}.s_in` into one engine layer, compare every pinned tensor of that layer (`run_isolated.py`). This is the gate a kernel is signed off against — no upstream cascade can enter | cos ≥ 0.9999, meanrel ≤ 1 % |
| **G4 attention** | `L{ll}.attn.q_rope`, `k_rope`, `attn_out_pre_gate`, `out` for all 6 layers. A `q`/`gate` de-interleave error (§05.1) must fail here with cos ≪ 1 — the gate is required to actually discriminate | cos ≥ 0.999, meanrel ≤ 5 %; de-interleave-negative test must fail |
| **G5 incremental == batch** | prefill `T` then decode `D` equals one prefill of `T+D` (states, logits, sampled ids) | sampled ids exact; logits cos ≥ 0.9995 |
| **G6 running-softmax** | if running `(max,sum)` softmax accumulators are used (§07.3), tokens identical to full-prefix softmax | sampled ids exact on `corpus512` |
| **G7 end-to-end greedy** | token-for-token match with HF-ref on `smoke`, `p8`, `p64`, `p300`, `prefix24+dec8`, then ≥ 95 % prefix agreement on `corpus512` before any perf claim | exact / 95 % |
| **G8 MTP** | §09.5 sequence: shapes, single-step agreement, measured accept rate | reported, no hard bound |
| **G9 perf** | decode tok/s on this box vs the §07.4 roofline (~135 tok/s); `-O2` build only | report; investigate below 60 % |

## 10.2 Metrics, drift, and what drift does *not* mean

`cos` = cosine similarity over the flattened tensor; `meanrel` = mean|Δ| /
mean|ref|; `maxrel` = max|Δ| / max|ref| is reported but not gating (one bf16 tail
element in the gated norm dominates it).

Measured on the four fixtures, both GDN prefill forms, all tensors:

| fixture | worst cascade `cos` | worst cascade `meanrel` | worst **isolated** `cos` | worst isolated `meanrel` |
|---|---|---|---|---|
| `smoke` (17 tok) | 0.99928 | 3.3 % | 0.99997 | 0.53 % |
| `p64` (46) | 0.99946 | 2.4 % | 0.99997 | 0.53 % |
| `p300` (277, 5 chunks) | 0.99978 | 1.6 % | 0.99995 | 0.76 % |
| `dec8` (17+8) | 0.99923 | 3.3 % | 0.99997 | 0.53 % |

Two things follow, and both were verified rather than assumed.

1. **Per-op math is exact.** With the reference's own layer input, the raw
   projections (`S04.1`, `S05.1`) reproduce it *bit-exactly* (`meanrel = 0`), and
   the worst residual is the attention output before the output gate at 5–8 ‰ —
   F32 reduction-order noise in softmax/scan/norm. So the 1.5–3.3 % cascade
   numbers are **not** a wrong formula; they are bf16 rounding amplified by 23
   further layers of residual stream + norms.
2. **Cascade drift is a floor, not a knob.** Three plausible fixes were each
   implemented and measured against the reference, and none moved it: chunked GDN
   prefill (`S04.7`), attention scores kept in the activation dtype instead of
   F32, and F32 accumulation in the causal conv. Each was kept because it is the
   reference-faithful or kernel-realistic choice, not because it improved the
   numbers. The earlier prediction that the chunked prefill would let `meanrel`
   drop to 1 % was therefore **wrong**, and is corrected here.

Consequence for gate design: the cascade ceiling stays at `cos ≥ 0.999, meanrel ≤
5 %` and its job is regression detection plus exact-token agreement (G7); the
binding *correctness* gate for a single op is **G3i**, which is tight
(`cos ≥ 0.9999, meanrel ≤ 1 %`) precisely because the cascade is removed. A kernel
that fails G3i cannot be excused by "cascade noise".

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
