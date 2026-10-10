# Agent notes — qwen3.5-0.8b

## SCOPE: one model, and it is small. Read this before sizing anything.

The only target of this repository is **`Qwen/Qwen3.5-0.8B`, text-only path**
(prefill + decode + MTP draft), running on the Framework Desktop (Strix Halo,
`gfx1151`). There is no second model in flight, and memory capacity is **not** a
constraint for this one:

| footprint item | size | source |
|---|---|---|
| text weights, bf16 | **1.5457 GB** (335 tensors; 485 MiB of it is the tied `lm_head`/embedding table) | §01.7 |
| MTP draft block | 39 MiB | §01 |
| per-request state | GDN 18.0 MiB + conv 0.63 MiB = **18.66 MiB**, + KV 12 KiB/token | §00.5 |
| 100 concurrent requests @ 4k ctx | 100 × 69 MiB + weights ≈ **8.3 GB** | arithmetic on the rows above |
| device-visible memory pool (measured ceiling) | ~34 GiB | `devices/strix-halo-unified-memory.md` |

So the whole bf16 checkpoint is **~4 % of the device-visible pool**, and even a
hypothetical 100-request server sits under a quarter of it. Therefore, for this model:

- **no** paging, streaming, eviction policy, or layer-by-layer reload;
- **no** quantization *to fit* (quantization for speed is M3 and is a different argument);
- **no** multi-model residency, no weights-on-disk fallback path, no allocation-churn
  optimization;
- **no** work on MoE, the vision tower, or any 35B/125B target. **M4 is not scheduled.**

Anything in this repo that discusses bigger models is a recorded device observation, not a
requirement. Such passages are tagged **[FUTURE]** and currently live in
`devices/strix-halo-unified-memory.md` (§"the two real limits") and the README's naming
note. Do not implement a [FUTURE] item without an explicit request.

## Traps and limitations index (detail lives in the linked file, not here)

| trap / limitation | where it is written up |
|---|---|
| bare dump names are last-writer-wins across phases, so `pre_final_hidden.npy` + `logits_last.npy` is not a valid input/output pair (cos 0.25) | `specification/10-reference-dumps-and-gates.md` §10.5, `prototype/lm_head_gemv/results.md` |
| a bandwidth number > ~240 GB/s means skipped work, not a fast kernel | `prototype/README.md`, `prototype/lm_head_gemv/results.md` |
| first GPU pass over fresh pages runs ~35 GB/s (7× slow) — warm before timing | `devices/strix-halo-unified-memory.md` |
| `hipMallocManaged` degrades **silently** to ~52 GB/s near ~32 GiB; `hipMalloc` fails loudly — use `hipMalloc` + explicit `hipMemcpy` for weights | `devices/strix-halo-unified-memory.md` |
| device-visible pool is ~32–34 GiB total (granularity-independent); irrelevant for this model at 4 % usage | `devices/strix-halo-unified-memory.md` (tagged [FUTURE]) |
| CPU-side bandwidth is only 75–117 GB/s, so a CPU-resident decode would cap at ~50–75 tok/s — the engine must be GPU-resident | `devices/strix-halo-memory-bandwidth.md` |
| HIP build needs `-Wl,-rpath,/opt/rocm/lib`; `.venv` torch is CPU-only; CUDA bf16/bitcast intrinsics differ; `__shfl_down_sync` mask is 64-bit | `prototype/README.md` |
| torch refuses `bf16.numpy()` — dump bits via `view(torch.uint16)` | `prototype/README.md` |
| MTP drafting is bit-lossless but **slower** than plain greedy on torch CPU (~21 vs ~24 tok/s, accept 0.66 @ k=1) — it is a GPU-path bet, not a win today | `specification/09-mtp-draft-layer.md` §09.5, `speccheck/README.md` |
| greedy token equality decays with depth on any bf16 implementation (chaos floor 0.8776 sdpa-vs-eager), so long-generation token match is not a gate | `specification/ASSUMPTIONS.md` A-18, §10.4 |
| §07.4's "skinny argmax is bandwidth-critical" was wrong for the GPU (fused gains ~1 %) — corrected in place | `specification/07-decode-step.md` §07.4, `prototype/lm_head_gemv/results.md` |
| timings are best-of-N on an unpinned idle desktop; absolute ms moves a few %, ratios vs the probe are the stable number | `prototype/README.md` |
| a `.gitignore` line like `prototype/**/<op>` (meant for the built binary) matches the op **directory** too, so the whole `prototype/<op>/` sits untracked and only `README.md` gets committed — ignore the binary by exact path `prototype/<op>/<op>`, and check `git status --ignored` after adding an op | `.gitignore` (comment block above the two binary lines) |

## Standing rules

- **Verify against the reference implementation at every step.** `speccheck/` is the
  independent checker (torch CPU, gated against the pinned HF dumps in `data/refs/`);
  `engine/` is the deliverable; `prototype/<op>/` graduates into `engine/` only after its
  own gate passes. Never loosen a gate to make it pass.
- Reference dumps are pinned files. A fresh HF run must reproduce them; if it does not,
  suspect the fixture pairing before the kernel (`prototype/lm_head_gemv/results.md`
  documents one such trap: the bare dump names are decode-step values).
- **A kernel number above ~240 GB/s is evidence of skipped work**, not of a good kernel —
  that is the measured LPDDR5x ceiling of this box (`devices/`).
- Never benchmark an `-O0` build. HIP builds use `-O3` via `prototype/*/build.sh`.
- Device measurements belong in `devices/` and must be reproducible with a tool in
  `tools/`. Numbers quoted in `specification/` without a probe behind them are assumptions
  and need an `A-xx` tag.
- Local commits are amended, not stacked, while unpushed.
