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
| **G6 online softmax** | decode attention reduced block-wise with `(max, sum)` rescaling (§07.3) instead of one `softmax` — accumulators are per-row reduction state, never carried between steps | sampled ids exact; decode logits cos ≥ 0.9999 (bf16 cascade floor, §10.2) |
| **G7 end-to-end greedy** | token-for-token match with HF-ref on every short fixture (`smoke`, `p8`, `p64`, `p300`, `dec8`, `prefix24dec8`); on the corpus, prefix agreement must be **at least the reference's own self-consistency floor** (§10.4, A-18) | exact on short fixtures; ≥ floor (measured 0.8776) on corpus |
| **G8 MTP** | §09.5: bit-exact loop with sequential verify + accept rate across the three readings of §09 | accept ≥ 0.30 and the §09 reading strictly best; speculative output and target state bit-identical to plain greedy when verify reuses the decode path |
| **G-neg negative controls** | `speccheck/gates/run_negative.py` injects three plausible bugs (wrong `q`/`gate` de-interleave, conv window written before it is read, GDN state not persisted) and asserts the pins *stop* matching, and still match when clean | each injection must drop its canary below cos 0.999 |
| **G9 perf** | decode tok/s on this box vs the §07.4 roofline (~135 tok/s); `-O2` build only | report; investigate below 60 % |
| **F1–F4 MLP decode path** | `prototype/mlp_gemv/run.py`, against the MLP pins (`decode{N}.L{ii}.mlp.{in,gate_pre,up_pre,act,out}`). F1 `gate_pre`/`up_pre` (two independent dots), F2 `act` (the SwiGLU rounding policy), F3 `out` from the *reference* `act` (isolates `down_proj` from everything upstream), F4 `out` from the *kernel's* `act` (the chain a real token runs), plus a cross-check of `speccheck.layers.mlp` against the same pins | no element above 1e-4 of the tensor max may exceed 1 BF16 ULP (2 for `act`); ≤2 % of elements differ; G3i-class cos ≥ 0.9999, meanrel ≤ 1e-2. F4 is aggregate-only (cos ≥ 0.9999, meanrel ≤ 1e-3) — see below |

The MLP gate is the one place where a per-element ULP bound had to be derived rather than
assumed, and `prototype/mlp_gemv/floor_probe.py` is where it comes from: reordering the
reference's **own** F32 summation moves about 1 element in 100 000 by more than 1 ULP, always at
an element that cancelled to ~1e-6 of the tensor max (where one ULP is a large fraction of the
value), and the exactly-rounded sum disagrees there by 4–16 ULP. It also measures the other half:
a legal 1-ULP difference in `act` — the most a correct kernel with a different summation order may
produce — produces an `out` element more than 1 ULP away from the pin in **every one of the 192
layer × step cases measured** (192 cases × 8 fixed trials = 1 536 perturbed dots; 7 483 offending
elements; worst 28 665 ULP). Most of those offenders are not small elements — the largest sits at
0.13 of the tensor max — so judging F4 per-element would judge the sensitivity of a 3584-term dot,
not the kernel. `prototype/mlp_gemv/floor_probe.py --layers all` prints all of this, with the trial
count on the line, because a worst-of-N is not a bound until N is next to it.

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

## 10.4 Greedy chaos: why corpus-level token equality is not a gate

The original requirement — “≥ 95 % prefix agreement on the corpus before any perf
claim” — is not achievable by *any* implementation, and saying so is better than
quietly moving the goalposts later. Measured on 49 plain-text prompts × 24 greedy
tokens (`data/corpus.txt`, `tools/corpus_dump.py`):

| comparison | mean prefix agreement |
|---|---|
| HF-ref sdpa vs HF-ref eager (**the reference against itself**, same weights, same bf16) | **0.8776** |
| speccheck vs HF-ref eager | **0.9124** |

Divergence starts after ~4–8 tokens and then the continuations are different texts,
not different renderings of the same text: a near-tie argmax flips, and greedy
decoding has no mechanism to recover. Consequences, all implemented:

1. G7 is exact-match on short fixtures (where the sampled ids still agree) and
   *relative* on the corpus: `run_g7.py --corpus corpus48 --floor corpus48sdpa`
   requires speccheck to be at least as faithful to the oracle as a legal rounding
   variant of the oracle is.
2. `run_gates.py` stops comparing decode tensors at the first token flip
   (`NOTE token flip after decodeN`) — beyond that the two sides are feeding each
   other different tokens, so the tensors are not comparable.
3. For the HIP engine, token equality on long generations is therefore **not** the
   acceptance test; G3i (per-op), G5 (state invariant) and short-fixture G7 are.
   Corpus agreement vs the floor is the regression tripwire.

## 10.5 Rules

- **Bare dump names are last-writer-wins across phases.** `reference_dump.py` pins
  `input_ids`, `pre_final_hidden`, `logits_last`, … with the same names in prefill *and*
  every decode step, so a bare `*.npy` holds whichever phase ran last while the
  `decodeN.`-prefixed copies survive separately. Consequence, and it cost a debugging
  session: pairing `pre_final_hidden.npy` with `logits_last.npy` does **not** reproduce the
  logits (cos 0.25) in a fixture that also contains decode steps. Anyone consuming a dump
  as "the input that produced that output" must either take a `decodeN.`-prefixed pair or
  re-run the reference live and self-check against `logits_last` first
  (`prototype/lm_head_gemv/run.py::normed_row_from_reference` does exactly this).
  Fixing the tool to namespace prefill names would invalidate every pinned fixture, so the
  trap is documented rather than removed.
- Every gate is a checked-in binary/script run by `ctest` (or a `run_all.sh` per
  `prototype/<op>/`), never a manual eyeball.
- A gate that passes on `smoke` only is **not** passing: `p64`/`p300` are required
  because of the chunk-64 padding in §04.7.
- Reference dumps are regenerated whenever the tool, transformers version, or
  checkpoint changes; the `manifest.json` fields (`transformers`, `torch`,
  `dtype`) are part of the fixture identity. **`tools/dump_fixtures.sh`** is the single source
  of the exact command line per fixture (including `--no-mlp-io` where a fixture carries no
  decode step), and **`tools/check_dump_reproduction.py`** re-verifies a fresh dump against the
  pinned one with zero tolerance — bit-identical BF16, byte-identical ids. Use it after any
  re-dump: it is what stands between a `transformers` bump and silently different numerics, and
  it already caught a fixture whose prompt had been generated as a *continuation* rather than a
  completion (so its 24-token prefix was not the one §08.6 invariant 1 needs).
- **The MLP pins exist as of `prototype/mlp_gemv`, with two limits.**
  `decode{N}.L{ii}.mlp.{in,gate_pre,up_pre,act,out}` are captured by module hooks, so they exist
  only for decode steps (k=1) — a prefill still pins only `s_in`/`s_out` per layer — and `act` is
  `down_proj`'s **input**, i.e. the SwiGLU product. A kernel that batches rows through the MLP
  has no fixture until the hook is widened, and the first version of that hook read
  `down_proj`'s *output*, pinned it under the name `act`, and passed every self-check while
  pointing the gate at the wrong tensor (§03 / `prototype/mlp_gemv/results.md`).
- **The projection operands are pinned as of W1, and unlike the MLP pins they exist for prefill
  too.** Nine modules, `decode{N}.L{ii}.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,
  q_proj,k_proj,v_proj,o_proj}.out` (unprefixed for the prefill pass), because what already existed
  was a step downstream in every interesting case: `gdn.mixed_qkv_pre_conv` is `in_proj_qkv`
  reshaped for the conv, `gdn.o_pre_norm` is after the gated RMSNorm, `attn.q_rope` after `q_norm`
  and rope, and `attn.attn_out_pre_gate` is **before** `o_proj`. Gating a projection kernel against
  a derivative of its output is how a transposed weight or a wrong head interleave survives a
  review, so the direct pins came first.
  Two facts about them are measured, not assumed (`/tmp/verify_proj_pins.py` at the time, rebuilt
  from the pins plus checkpoint weights; worth folding into the dumper's self-check):
  - `q_proj` is interleaved **per head** as `[8 heads][q 256 | gate 256]`. Reconstructing `q_rope`
    from a contiguous-half split scores cos **−0.14** against the pin, so the layout cannot be
    gotten quietly wrong (§05.1).
  - the `self_attn` block-output pin the dumper intended (`L{ii}.attn.out`) **does not exist**:
    `self_attn` is called with keyword arguments, so a positional-input hook sees nothing. Until
    that hook is fixed, `o_proj.out` is the only pin of the attention block's output — and it is
    the residual addend, which is what §03's residual chain needs anyway.
- Never trust byte values from ad-hoc dumps in the old project
  (`../qwen3.8-flash-next-old/prototype/*/data`): shapes only, per its own
  `REFERENCE.md`.
