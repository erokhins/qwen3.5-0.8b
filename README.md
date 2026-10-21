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

## Scope: one model, and it is small

**The only target is `Qwen/Qwen3.5-0.8B`, text-only.** See `AGENTS.md` for the full scope
statement and the traps index. The part that keeps getting confused: the whole bf16
checkpoint is **1.5457 GB ≈ 4 % of the device-visible memory pool**, so nothing in this
project needs streaming weights, eviction, quantization-to-fit, or any capacity
engineering. MoE / 35B-A3B / vision appear only as `[FUTURE]` device observations and are
not scheduled work.

## Hardware baseline

Framework Desktop, AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`), 128 GB
LPDDR5x-8000 unified memory, single NUMA node.

| quantity | value | consequence |
|---|---|---|
| OS-visible RAM | ~125.1 GiB | whole checkpoint stays resident; no streaming tricks needed |
| **measured iGPU read bandwidth** | **~237–240 GB/s** (copy 210, CPU-side only 75–117) | `devices/strix-halo-memory-bandwidth.md`; decode speed ≈ bytes read per token ÷ bandwidth |
| device-visible memory pool | ~34 GiB (system RAM via GTT, no real VRAM) | `devices/strix-halo-unified-memory.md` |
| measured facts | `devices/` (this project) and `../qwen3.8-flash-next-old/devices/framework-desktop-ryzen-ai-max-395/` | reuse, do not re-measure |

Decode ceiling for the target is bandwidth-bound, not FLOP-bound: text-only
weights **1.5457 GB** bf16 + 18.7 MiB/request state ≈ 1.565 GB per token ÷ the
measured 237 GB/s → **~151 tok/s** theoretical (§07.4, confirmed by
`prototype/lm_head_gemv` at 242 GB/s); ~1.0 GB at Q4 → ~240 tok/s. Either way the
0.8B case is memory-roofline-bound, and today's torch path (23 tok/s, CPU, and it is the
correctness oracle rather than a prototype engine) is 6× below it — which is exactly the
gap the engine exists to close.

**The engine runs end to end.** `engine/performance-status.md` is the honest ledger: prefill **12,047 warm prompt tok/s
at 1,024 lanes** (11,070 at 4,096, 8,542 at 16,384 — the 2026-10-20 warm banner) and decode **101.3 tok/s marginal at 128
lanes** (2026-10-21 window, D1 tree) are device numbers; §1 is the live table, §2–§6 say where the time goes
and what a number is worth, and superseded eras are frozen in the two `progress/<date>/performance-status-<date>.md` snapshots. Carry the lane count and the ledger row with any tokens/s figure.

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
speccheck/                   # independent Python implementation of that spec + all gates (NOT the engine)
speccheck/chat.sh            # interactive chat on that path, with prefill/decode stats
testdata/                    # engine-agnostic speed tests (long summarise, long poem) + schema
bench/                       # one data file: `bench/band-history.jsonl`, the band record the gate reads and appends to (the ledger moved to `engine/` and the audits to `progress/`)
progress/                    # archive: dated one-shot docs in <YYYY-MM-DD>/ dirs, plus
                             # progress/<owner>/evidence/ for the committed evidence files
engine/                      # the engine: kernels, residency, scheduler, state pools, drivers, and `engine/performance-status.md`, the live ledger
prototype/                   # one subdir per op: standalone kernel + reference dump + gate
engines/                     # cloned reference engines for cross-checking (gitignored, read-only)
data/                        # reference dumps, fixtures, bench output (gitignored, regenerable)
devices/                     # measured platform facts
```

Conventions inherited from the old project:

- `specification/` is math + exact tensor shapes only, no library calls. Any
  behavior derived from a reference implementation is tagged `[REF]` or
  `[ASSUMPTION A-xx]` and registered in `specification/ASSUMPTIONS.md`.
- Every `prototype/<op>/` ships with a reference-data tool, a test, and a
  `results.md`. Code graduates from `prototype/` into `engine/` only after its
  gate passes.
- Start a chunk by reading `progress/` (latest date folder first) and the
  `results.md` journals, plus the current handoff note
  `progress/2026-10-14/HANDOFF-2026-10-14.md`. Those documents hold
  on-hardware-verified facts and are the handoff memory. Do not re-derive them.
  `progress/README.md` is the index and the rule for what goes there.
- Never time anything against an `-O0` build (pin `-O2`; see
  `../qwen3.8-flash-next-old/thinking/p3d-4-slow-tests.md`).

## Milestones

- **M0 — spec.** `specification/` for the text-only forward pass of
  Qwen3.5-0.8B: notation/hyperparams, weights-and-shapes, tokenization +
  embedding, GDN layer, gated GQA layer, MLP, prefill, decode step, state
  management, MTP. Plus a dumper that pins HF-transformers reference tensors.
- **M1 — per-op parity.** each op matched against the reference dumps within a
  documented tolerance, wired as a `ctest` gate suite.
  **Met** — every op family has its own scripted device gate against the pinned dumps
  (`prototype/<op>/run.py`), not a `ctest` suite; the bands and their floors are recorded per op.
- **M2 — end-to-end greedy parity.** identical greedy tokens on a pinned prompt
  set, single-stream decode, reported in tok/s on this box.
  **Met 2026-10-12** — `engine/gate.py`: `dec8` 8/8 ids exact (94 checks), `dec64t` 0 fails over 546
  checks with 64/64 ids exact and 64/64 identical on a free run; **83.9 tok/s** (11.919 ms/token at a
  25-lane context, progress/2026-10-16/performance-status-2026-10-16.md:15; the live headline is §1 of `engine/performance-status.md`).
- **M3 — useful.** quantized blobs (reuse the GGUF-import lessons), MTP
  speculative decode, minimal HTTP/serving shim.
  **Descoped 2026-10-12 by scope ruling** — quantization, MTP and any serving shim are dropped for
  this project; the work that replaced them is `progress/2026-10-12/plan-2026-10-12.md` (fixture breadth, prefill §06,
  single-stream performance at a 4k context).
- **M4 — scale-up (only after M2).** add the MoE block and a second engine
  model at `models/Qwen3.6-35B-A3B`, which needs no subsystem beyond what
  M0–M3 already implements. **[FUTURE]** — not scheduled; see `AGENTS.md` scope note.

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
- `speccheck/specimpl.py`: spec implemented from scratch in torch-CPU (not by
  calling transformers) and `speccheck/gates/`: **all §10 gates pass** on four
  fixtures (`smoke`, `p64`, `p300`=277 tok/5 chunks, `dec8`), with both the
  recurrent (S04.4) and chunked (S04.7) GDN prefill. `engine/` is reserved for the
  real HIP engine; `speccheck/` is the independent implementation that checks the
  specification and will gate the kernels.
- Bugs the gates caught (not code reading): the G0 tensor split (335 loaded /
  153 vision, not 344/144); the GDN recurrent state not persisted across calls,
  so decode restarted from zeros while prefill matched to 1e-5; MLP gate product
  in F32 instead of the activation dtype.
- `speccheck/gates/run_isolated.py` (G3i) feeds the reference's own layer input into
  one engine layer: projections come out **bit-exact**, worst residual 5–8 ‰.
  That is what proves the 1.5–3.3 % end-to-end drift is bf16 cascade amplification
  rather than a wrong formula — three candidate fixes were implemented and
  measured, none moved it (§10.2).
- Gate set completed: G6 (blocked online softmax, A-17), G7 as an explicit
  token-equality gate (short fixtures) plus a corpus gate measured against the
  reference's own self-consistency floor (A-18: HF-sdpa vs HF-eager = 0.8776,
  speccheck = 0.9124 over 49 prompts × 24 tokens), A-11/A-10 tokenizer+render gate,
  and G-neg (`run_negative.py`) which injects three plausible bugs and asserts the
  pins stop matching — a gate that cannot fail verifies nothing.
  `speccheck/run_all.py` runs all 24 gate invocations; currently **24/24 PASS**.
- Two spec bugs found by the new gates: §02 had the wrong literals for 248068/248069
  (`<|im_start|>`/`<|im_end|>` are distinct tokens from `<|im_start|>`), and §07.3’s
  “running accumulators” wording invited a per-request reading that is wrong (A-17).
- Stop rules implemented (A-10): greedy stops after emitting 248046, accepting
  248044 as a legacy stop.

- **Speed harness.** `./bench.sh` runs `testdata/` (one prefill-dominated test: summarise
  a ~2200-token article; one decode-dominated test: write a long poem) against
  `speccheck` and the HuggingFace reference in separate processes with the same thread
  count, splitting prefill from per-token decode, and checks the two engines consumed
  identical prompt ids. Median of 2, 8 threads: prefill 674 tok/s (speccheck) vs 946
  (HF), decode 23.0 vs 22.6 tok/s, i.e. ~35 GB/s of weight reads against the ~210-215
  GB/s `specification/07.4` assumes. That gap is the engine's job; these numbers are
  the baseline `G9` replaces.

- **MTP (§09) implemented and measured.** `speccheck/mtp.py` runs the draft layer and a
  greedy speculative loop; `run_mtp.py` (gate G8) shows the loop is bit-lossless when
  verification reuses the decode path, and **falsified A-12**: the block takes the
  hidden state *after* the final norm (accept 0.642 vs 0.232 pre-norm; a swapped `fc`
  concat order collapses to 0.014, which is what makes the gate meaningful). On torch
  CPU drafting still *loses* — 23.3 → 18.3 tok/s at k=1, 15.8 at k=2 — because the draft
  pays a 248k-row tied `lm_head` as a slow bf16 GEMV and rejection costs a re-forward.
  The roofline says k=1 should win ~1.15× at accept 0.58, so the conclusion is “not in
  Python”, not “MTP does not help”: re-measure once `mtp_layer` is a kernel.
  `./bench.sh --engine speccheck --mtp 1` prints both rows.
- Gate suite is **26/26 PASS** via `speccheck/run_all.py`.

- **Three HIP kernels exist and each is gated against the pins.** `prototype/lm_head_gemv` (tied
  logits GEMV, 242.1 GB/s = 102 % of the measured read probe), `prototype/gdn_decode` (S04.2–S04.4
  decode step, all four functions, bit-exact conv), `prototype/mlp_gemv` (S03.3 gate+up+SwiGLU and
  down, F1–F4 over 192 layer×step cases, 2.650 ms per token cold = 84 % of the probe).
  `engine/performance-status.md` §4 is the ledger's per-op floor table: 4.82 ms of the ~6.5 ms/token
  budget was this line's pre-engine-era figure, and it is the place that says which numbers are measured and which are projected.

Next: the projection GEMVs — `mlp_gemv`'s shape-generic kernel instantiated at the GDN and
attention projection shapes (38 % of the token) — and the §05 gated full-attention decode step,
then `engine/` (safetensors → `hipMalloc` + `hipMemcpy`, warm pages) so the first real tokens/s
row can go into the ledger. `prototype/gdn_decode`, `prototype/lm_head_gemv` are done.