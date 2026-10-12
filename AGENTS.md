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
| **max-ULP bands are blind to whole-tensor 1-ULP error, and ndiff is blind to chains** — a BF16 subpartial in a norm moves 2.05 % of elements by ≤1 ULP and passes any max-ULP bound, while a chained dot's *legal* ndiff is 29–47 % so ndiff there is meaningless. Use ndiff for stage-local near-exact ops (norms, residual adds, rope: their floor is 1 ULP with **no** cancellation band), and cos/meanrel aggregates for chains | `prototype/glue/floor_probe.py`, `prototype/mlp_gemv/results.md` (F4) |
| §07.4's "skinny argmax is bandwidth-critical" was wrong for the GPU (fused gains ~1 %) — corrected in place | `specification/07-decode-step.md` §07.4, `prototype/lm_head_gemv/results.md` |
| timings are best-of-N on an unpinned idle desktop; absolute ms moves a few %, ratios vs the probe are the stable number | `prototype/README.md` |
| a GPU number taken while the main agent is streaming (or via a backgrounded run) is contended by the session's own tokens — run the command blocking, with zero subagents | `bench/performance-status.md` §6 |
| `hipEventElapsedTime` can return exactly 0.000 ms for a 2.1 ms kernel, not only sub-ms ones; steady_clock burst is the only timer to trust | `bench/performance-status.md` §6, `d81ff13` |
| a poll loop that `pgrep -f`s a script named in its own command line matches **itself** and never breaks — wait on a sentinel the job writes (log line, `touch` marker) instead | `bench/commit-review-2026-10-11.md` |
| a `.gitignore` line like `prototype/**/<op>` (meant for the built binary) matches the op **directory** too, so the whole `prototype/<op>/` sits untracked and only `README.md` gets committed — ignore the binary by exact path `prototype/<op>/<op>`, and check `git status --ignored` after adding an op | `.gitignore` (comment block above the two binary lines) |
| a BF16 ULP comparator written on a `uint16` view has a **dead branch**: `np.where(x < 0, …)` is never true for `uint16`, so every negative value collapses to +0.0 and the gate passes vacuously. Correct: `np.where(bits & 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)` | `prototype/bf16_ulp.py` (has a selftest — run it), `prototype/mlp_gemv/results.md` |
| `hipEventElapsedTime` returns **exactly 0** for kernels below ~1 ms on this iGPU. Sub-millisecond timings must come from a burst (N launches, one sync, wall-clock/N); a zero is a broken timer, not a fast kernel | `prototype/mlp_gemv/results.md` |
| "max 1 ULP against the dump" is **not** an achievable bound for a long dot: reordering the reference's own F32 sum already moves ~1 element in 1e5 by more than 1 ULP, always at an element that cancelled to ~1e-6 of the tensor max. A BF16 gate needs a cancellation band and aggregate metrics, both sized by a floor probe, not by taste | `prototype/mlp_gemv/floor_probe.py`, §10.1 (F1–F4) |
| a forward hook that reads the hook's `output` argument instead of `input` can pin *the answer* under the name of an intermediate — the MLP's `act` is `down_proj`'s **input**, and the wrong version reproduced real reference tensors and passed every self-check | §10.5, `prototype/mlp_gemv/results.md` |
| a negative test that loads another layer's weights "catches" every injection no matter what the kernel does — vacuous. Injected bugs must also be required to appear **only** where the bug lives | `prototype/mlp_gemv/run.py --negative`, §10.1 |
| **never print chat-template special tokens literally** — in agent replies, commit messages, gate output, logs or reports. Every agent on this box is served by the checkpoint itself, whose stop list is its own eos id (`248046`), so a literal end-of-turn token in an assistant message ends that message where it stands: if it lands in prose the tail is silently dropped, and if it lands inside a tool call the call is lost and the session restarts on the same call, looping. Refer to these tokens **by id** (248046 end-of-turn/eos, 248068/248069 the think-boundary specials — the latter two are not stop tokens); code already uses `engine::kStopToken` and needs no literal. Two agents were cut off this way on 2026-10-11, one of them a worker brief that lost its second half. Same rule as a trap for the engine: a prompt fixture legitimately *contains* 248046, so the §07.5 stop check is generation-phase only and must never be applied while feeding a prompt | this row, §07.5, `engine/decode_run.cpp` |
| A shape claim written as a guess is a bug with a compile-time guarantee it will be wrong eventually: `check_weights` claimed `linear_attn.conv1d.weight` was BF16 `[6144,4]` when the shard holds BF16 `[6144,1,4]` — all 18 rank-3 tensors in the checkpoint — because the `Expected` struct had no rank field and could not express otherwise. It aborted the first device run before any kernel launched. Correct: state the rank, and put a rank-collapsing contract (`[6144,1,4]` → the flat `[6144][4]` a kernel indexes) at the use site, never in the checker | `engine/weights_contract.py` (re-reads every claim from the shard header on CPU), `engine/results.md` |
| **three gate drivers acquire `tools/gpu_lock.sh` themselves** (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`), so a caller that holds the lock and then runs one of them **deadlocks it** — the child parks silently with no CPU or GPU work, which looks like a hung box. Correct protocol: hold the lock only around the scripts that do *not* self-lock (`lm_head_gemv`, `mlp_gemv`, `attn_decode`, `gdn_decode`, `decode_run`), and release before invoking a self-locking one. Check with `grep -c gpu_lock prototype/<op>/run.py`; a stale holder shows as `STALE (holder pid dead)` in `gpu_lock.sh status` | this row, `bench/performance-status.md` §6 (2026-10-12 re-take) |
| a torch CPU baseline line printed *after* the device `RESULT` line will swallow it if you pipe through `tail -1`; `lm_head_gemv/run.py` prints the GPU `RESULT` first and the CPU baseline last, so grep for `RESULT`, not for `ms` | `prototype/lm_head_gemv/run.py`, this session |
| `Tensor.to(torch.uint16)` casts numerically; `.view(torch.uint16)` bitcasts — a fixture writer that confuses them writes garbage that looks like unwritten memory (float 0.195 → the integer 0, −1.3 → 0xFFFF, read back as BF16 +0 and NaN; a red device gate, a green CPU mirror, and a kernel that was never at fault) | `prototype/gdn_gated_norm/results.md` §8, `prototype/gdn_gated_norm/run.py` (`write_bf16_raw`, `verify_cases`, `--selftest`) |

## GPU execution is serialized and belongs to the main agent

The inference engine that generates tokens for **this session and every subagent** (`flash_serve`,
`/models/qwen38-flash-next-v2.hgn`) runs on this same GPU. So GPU work is not just a scarce device,
it is the thing we are sitting on: any kernel run concurrent with another agent's work — or with our
own serving — is contaminated, and a contaminated number is worse than no number.

Therefore, as a hard rule:

- **Subagents may run short correctness-only device gates.** A worker may execute a device check that
  answers PASS/FAIL — a gate, a negative test, a parity diff, `--selftest`, one step of a loop — as long
  as **all** of these hold:
  - **under ~60 s wall.** If it does not fit, stop and hand the command to the main agent instead; a
    long worker-held device run is what starves everyone else's tokens.
  - **correctness only, never timing.** No bursts, no best-of-N, no warm loops, no sweeps, no `ms`
    figure in its report. Contention cannot corrupt a deterministic result (these gates have no
    atomics), but it corrupts every timing, and a contaminated number is worse than no number.
  - **no number from a worker device run enters a timing doc.** `bench/performance-status.md` and any
    `results.md` perf row are main-agent-only, because only the main agent runs with the box provably
    exclusive.
  - **it reports what it ran**: the exact command line, wall time, `tools/gpu_lock.sh util` at the time,
    and an explicit "no timing taken". An unattributed device result is not a result.
  - **the one-at-a-time rule still binds it**: it must not launch a device run while another subagent is
    active. Same cap as everything else.
- **The main agent still owns every measurement**: any timing, anything that enters the ledger, and any
  gate that cannot finish inside a minute — run while **zero subagents are active**, blocking, never
  backgrounded. The reason for the split is cost: three of today's five device bugs each needed a
  stop/brief/resume round trip purely because the worker could not run its own one-second gate.
- `tools/gpu_lock.sh` still applies (acquire/release by TAG, record `util`), but it is now the second
  line of defence, not the first: the first is that nothing else is running at all.
- **One subagent at a time, and it must be STOPPED for the main agent to measure.** An idle-but-alive
  child still streams from the same endpoint, so "zero subagents active" means stopped/paused, not
  merely quiet between steps. The working cycle is therefore: brief one worker → it does all code,
  analysis and CPU-side verification → **stop it** → main agent runs the device commands it handed over,
  in the foreground → resume it with the results. Budget for a worker's device-dependent steps being
  deferred by construction.
- **Division of labour.** The main agent steers: briefs, sequencing, gate-policy decisions, the device
  pass, and commit hygiene. Subagents do the actual work — kernels, gates, references, review, docs —
  including CPU test runs. The main agent does not implement while a worker could.
- **Run device commands *blocking*, never backgrounded.** A foreground bash call suspends this session's
  token generation, so `flash_serve` — which busy-polls the iGPU and is the source of the interference —
  has nothing to emit for the duration. Backgrounded runs (`&`, `nohup`, `bg_wait`) keep the stream alive
  and contend with themselves. Measured: with the turn blocked and zero subagents, the three GEMV
  families repeated within ~2 % of ledger figures (`bench/performance-status.md` §6). Record `util`
  anyway, then discount it: identical clean runs read 24–91 %.
- **Bandwidth-bound timings repeat; dispatch-bound ones do not.** Launch-bound measurements (glue's
  49-launch path) are bimodal on this box — best-of-N spreads of 158–269 % at 512 threads, and a best
  that beat the contended historical number while being *worse*. Quote those at 256 threads and state
  the spread, or characterise the bimodality before trusting a per-launch µs figure.
- Consequence for planning: a worker's report will often be "gate written, not yet run". That is an
  acceptable state — schedule a GPU pass after the workers drain, run the commands they handed over,
  and feed failures back.

## Standing rules

- **Budget verification to what a change can affect.** `speccheck/run_all.py` is 26 gates and ~20 min;
  run it **only** after touching `speccheck/specimpl.py` or a gate file. For everything else
  (`prototype/*` plan writers, bench docs, loaders) `run_all.py --quick` (11 gates) plus the op's own
  CPU probe is the correct level. Three workers today expired their budget re-running a suite that was
  already green and could not have been affected by their edit — the failure mode is silent, because
  re-verifying always looks responsible. If a result is already on disk and nothing that feeds it
  changed, cite it instead of re-making it.

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
- **Commit by topic, and keep commits small.** One commit = one change that stands on its own
  and can be described in one sentence: a kernel plus its gate and its `results.md`; a tool or
  probe; a dumper change plus the fixtures it re-pins; a specification correction; a
  ledger/status update; the traps that op discovered. Prefer several such commits over one that
  spans the whole session — a reader should be able to revert one topic without touching
  another, and the subject line should say which. Amending is for finishing a topic's own
  work-in-progress, not for folding a new topic into the tip. If work already landed as one
  lump, `git reset --soft` and re-commit it in topical pieces; nothing here is pushed, so
  splitting is free. A backup ref (`git branch backup-<something>`) before rewriting history is
  cheap and worth it.
