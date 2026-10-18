# HANDOFF — D48/D49: llama.cpp as a prefill comparison instrument

Base: `b066120`. **Who runs what:** S1 and S3 are worker work (CPU only). **S2 is main-agent work** — it is a
device measurement, and per `AGENTS.md` the main agent owns every timing, blocking, with zero subagents
active. Do not run device commands in S1. Do not touch `bench/`, `specification/`, or `engine/`.

## Why this item exists

The user expects llama.cpp to prefill `Qwen/Qwen3.5-0.8B` on this box at roughly 9k tok/s against our **4,444**
(§16), and wants the *reason*, not the number. Arithmetic already recorded (D47/D48):

- 9k tok/s demands **9.85 TFLOP/s** = 44 % of the measured mix roof, and our `proj+mlp` families already run at
  **97–102 %** of that roof → **the gap cannot be in the matmuls**. It must live in the 80.8 % that is not matmul.
- “9k” is arithmetically identical to “`gdn_rec` → 0”: that family is **442.55 ms of the 920.18 ms wave**, and
  deleting it yields 477.6 ms = **8,575 tok/s**.
- llama.cpp formulates the linear-attention layers as **generic ggml ops** (`src/models/delta-net-base.cpp`:
  `ggml_mul_mat` ×10, `ggml_transpose` ×13, `ggml_mul` ×11, `ggml_exp` ×6, `ggml_pad` ×5, **`ggml_tri` ×4**,
  `ggml_sum_rows`) — i.e. the chunked, matmul-shaped delta rule. `GGML_OP_GATED_DELTA_NET` exists in
  `ggml/include/ggml.h:583` but has **no CUDA implementation** (open upstream issue #22967; Metal #20342).
  That is the same axis as our item **1.2 / P9**, whose 5.9× loss D46 showed was a thread-mapping defect.
- Counter-evidence on record: public Strix Halo llama.cpp reports put prefill in the **hundreds** of prompt
  tok/s for Qwen3-class models on this hardware. The measurement decides; either result is a useful outcome.

## Facts already established (don't re-derive)

| fact | evidence |
|---|---|
| clone is at `engines/llama.cpp`, HEAD `797de63`, gitignored | this session |
| arch supported: `LLM_ARCH_QWEN35`, `LLM_ARCH_QWEN35MOE` | `src/llama-arch.cpp:41,1231` |
| per-op perf printing lives in `src/llama-context.cpp` (`llama_perf_*`); `common/arg.cpp` has no `--profile-ops` in this tree — **find the actual switch or the build flag** | grep this session |
| **no `cmake` on PATH** (not in /usr/bin, ~/.local/bin, or `.venv/bin`) | blocker for S1 step 1 |
| ROCm at `/opt/rocm`, `hipcc` and `rocminfo` present at `/opt/rocm/bin`; `/opt/rocm/.info/version` empty → get version another way | this session |
| no `.gguf` anywhere locally; `huggingface_hub` 2.2.0 importable; `huggingface.co` reachable (HTTP 200) | this session |
| disk free 2.3 TB | `df` |

## S1 — worker, CPU only

1. **Unblock the build.** Install/locate `cmake` (`.venv/bin/pip install cmake` is acceptable and keeps it out of
   the system), record the ROCm version via `hipcc --version` / `rocminfo | grep gfx1151`, and confirm the ROCm
   toolchain can target `gfx1151` (our own prototypes build with `--offload-arch=gfx1151`, so codegen exists).
2. **Get a comparable model file.** Requirement: **f16/bf16, unquantised** — a `Q4`/`Q8` GGUF makes the
   comparison meaningless (different bytes, different arithmetic). Candidate `AaryanK/Qwen3.5-0.8B-GGUF`
   (verify its dtypes and that the arch loads), else convert from `models/Qwen3.5-0.8B` with
   `convert_hf_to_gguf.py`. Record the file, its dtype, and its size.
3. **Find the per-op profiling switch** for this tree (build flag or CLI arg), and the `llama-bench` flags that
   control the knobs that can fake a win: `-ub` ∈ {128,256,512,1024,2048}, `-fa` on/off, `-ngl 99`, `-t`.
4. **Hand back exact commands for S2**, ready to paste, including `tools/prefill_sampler.sh TAG -- …` wrapping,
   the per-op profile invocation at 512/1024/4096 prompt tokens, and the mapping table from ggml op names →
   our `--time-waves` families. **Do not run them.**

## S2 — main agent, device, blocking, one lock hold

`llama-bench -pp 512,1024,4096` at the settings above (knobs swept, every quoted number naming its settings),
per-op profile at the same three widths, plus our own `decode_run --time-waves` re-take **inside the same lock
hold** for any ratio that will be quoted (§16's ruler lesson: a re-take in a different window is not an A/B).
CPU % and VRAM come from `tools/prefill_sampler.sh`; note its header's warning that the **GPU-busy column
saturates on this box** (the session engine busy-polls the iGPU) and is a sanity check only.

## S3 — verdict, and the mandatory part

Deliver a ratio table `llama/ours` per comparable family, with an explicit **not-comparable** column (llama.cpp
scheduler/graph-build overhead, `-ub` as a policy choice, model load, our absent vision/MoE path). **Every row
where llama.cpp is ≥1.5× faster becomes an investigation item**, each needing: (a) the named kernel/library call
in its backend, (b) that one op isolated at **our** geometry in both engines (same shapes, dtype, entropic fill),
(c) attribution to exactly one of **traffic / launch count / occupancy / fusion** with a falsifier. “Their kernel
is better” is not a finding.

**If the delta is real and sits in GDN**, D46's closure of 1.2 must be re-opened with this as the external
existence proof — the untested combination is *chunked GDN expressed as library GEMMs* rather than P9's
hand-written scan kernel (which re-executed its priced arithmetic 13.5× over with a mis-mapped thread layout,
`p9-route-a`/`p9-phase1c`). **Fallback if the build or the arch refuses:** prototype the chunked GDN prefill
ourselves against the pinned pins — that was always the target; llama.cpp is only the existence proof.
