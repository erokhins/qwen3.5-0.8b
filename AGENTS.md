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
| a negative test that loads another layer's weights "catches" every injection no matter what the kernel does — vacuous. Injected bugs must also be required to appear **only** where the bug lives. Two derived exceptions found while gating the tiled attention wave: a **score-only** injection cannot move a row with exactly ONE lane (softmax of a singleton is 1.0 whatever the score arrives as), and a mask-widening injection cannot move a row that already attended the whole buffer. Must-catch/must-not-catch sets must therefore be DERIVED from the geometry, not hand-listed — a hand-listed set rots into "everything" the moment the fixture changes. Corollary for CPU mirrors: the mirror of §05.5's own form is NOT bit-identical to the pins (0.034 % of elements, ≤ 10 ULP) because it sums `l` serially where torch's softmax does not, so the witness for "the operands are right" is `reference.attend` reproducing the pins at ndiff 0, and the mirror itself is judged by cos/meanrel | `prototype/attn_tile_model.py` (`neg_expect`, `pin_witness`), `prototype/mlp_gemv/run.py --negative`, §10.1 |
| **never print chat-template special tokens literally** — in agent replies, commit messages, gate output, logs or reports. Every agent on this box is served by the checkpoint itself, whose stop list is its own eos id (`248046`), so a literal end-of-turn token in an assistant message ends that message where it stands: if it lands in prose the tail is silently dropped, and if it lands inside a tool call the call is lost and the session restarts on the same call, looping. Refer to these tokens **by id** (248046 end-of-turn/eos, 248068/248069 the think-boundary specials — the latter two are not stop tokens); code already uses `engine::kStopToken` and needs no literal. Two agents were cut off this way on 2026-10-11, one of them a worker brief that lost its second half. Same rule as a trap for the engine: a prompt fixture legitimately *contains* 248046, so the §07.5 stop check is generation-phase only and must never be applied while feeding a prompt | this row, §07.5, `engine/decode_run.cpp` |
| A shape claim written as a guess is a bug with a compile-time guarantee it will be wrong eventually: `check_weights` claimed `linear_attn.conv1d.weight` was BF16 `[6144,4]` when the shard holds BF16 `[6144,1,4]` — all 18 rank-3 tensors in the checkpoint — because the `Expected` struct had no rank field and could not express otherwise. It aborted the first device run before any kernel launched. Correct: state the rank, and put a rank-collapsing contract (`[6144,1,4]` → the flat `[6144][4]` a kernel indexes) at the use site, never in the checker | `engine/weights_contract.py` (re-reads every claim from the shard header on CPU), `engine/results.md` |
| **three gate drivers acquire `tools/gpu_lock.sh` themselves** (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`), so a caller that holds the lock and then runs one of them **deadlocks it** — the child parks silently with no CPU or GPU work, which looks like a hung box. Correct protocol: hold the lock only around the scripts that do *not* self-lock (`lm_head_gemv`, `mlp_gemv`, `attn_decode`, `gdn_decode`, `decode_run`), and release before invoking a self-locking one. Check with `grep -c gpu_lock prototype/<op>/run.py`; a stale holder shows as `STALE (holder pid dead)` in `gpu_lock.sh status` | this row, `bench/performance-status.md` §6 (2026-10-12 re-take) |
| a torch CPU baseline line printed *after* the device `RESULT` line will swallow it if you pipe through `tail -1`; `lm_head_gemv/run.py` prints the GPU `RESULT` first and the CPU baseline last, so grep for `RESULT`, not for `ms` | `prototype/lm_head_gemv/run.py`, this session |
| **a per-thread store to a per-block checksum slot is not a block reduction.** `sums[blockIdx.x] = per_thread_acc` looks like the partial and is not one: every thread of the block writes the same address and the survivor is one thread's total, so `tools/mixed_bw`'s read checksum reported exactly 1/blockDim of the bytes it had read (measured: `sum=65520 want=16773120` at 256 threads) and **every** warm row of the first Q8 sweep failed while its ms/GB/s looked entirely plausible. A gate that fails for every configuration catches nothing, and a reader tabulates the number anyway. Correct: reduce across the block (dynamic shared memory, one double per thread, or a warp butterfly with the 64-bit mask), and keep the expected value derived from the plan rather than from the kernel so the check cannot pass by construction. Same lesson one layer over: a write side verified by reading back a buffer that was pre-filled with the value the writes store proves nothing either — pre-fill it with a value writes never store. | `tools/mixed_bw.cpp` (`publish_block_sum`, `verify_accounts`), `devices/strix-halo-mixed-traffic.md` §"Row corrections after the first attempt" |
| **a CPU-only geometry selftest is not a witness for the kernel.** `--selftest-geom` printed `ratio_act=62.0000 ok=1` for exactly the configurations whose `mix` kernel then read 62x past its own buffer and faulted (`HSA_STATUS_ERROR_MEMORY_FAULT`), because the solver's `rounds` meant "grid passes per buffer" while the kernel's meant "trips of the ratio loop" and `rd_n4` was already `rounds*rrd*stride`. Correct, twice over: re-derive the kernel's index arithmetic on the host from the same struct the launch is built from (`plan_bounds` + `invariants`), and make the device selftest verify **bytes touched** (exact read checksum, write-landing read-back, poison guard bands around every region) rather than arithmetic. Related trap: run each device case in its own forked child — one faulting kernel poisons the context and every case after it silently never ran, which is how the first `--selftest` shipped half-broken. | `tools/mixed_bw.cpp` (`plan_bounds`, `SELFTEST` in the file header, `run_dev_case`), `devices/strix-halo-mixed-traffic.md` §"Row corrections after the first attempt" |
| a free run compared against the wrong index of the reference looks like total divergence: §10.5 pins `argmax(decode{s}.logits_last) == greedy_continuation[s+1]`, so a `kind=decode` sample at step *s* pairs with `greedy[s+1]`, never `greedy[s]` — aligned on `[0:]` a 128-token prose free run read 0/128 and a 64-step one read 1/64, both false. The session REPL is the other index: its first `gen` output is the sample already pending from the last fed token, so `gen[i]` pairs with `greedy[i]`. State which alignment a comparison used, always | `engine/results.md` (Q2 section, free run + `feed`/`gen` parity), §10.5, `engine/chat.py` |
| `reference_dump.py --verify` compared every file in a fixture directory, so any fixture that has been **priced** (`engine/route_allowance.py` writes `route_allowance.json` beside the pins) could never reproduce: `--verify` exited 1 on 826 byte-identical tensors. A carried file is named and skipped, not silently ignored — the file's own provenance check lives in `engine/gate.py:load_allowance` | `tools/reference_dump.py` (`CARRIED_NOT_PINNED`), this session |
| `Tensor.to(torch.uint16)` casts numerically; `.view(torch.uint16)` bitcasts — a fixture writer that confuses them writes garbage that looks like unwritten memory (float 0.195 → the integer 0, −1.3 → 0xFFFF, read back as BF16 +0 and NaN; a red device gate, a green CPU mirror, and a kernel that was never at fault) | `prototype/gdn_gated_norm/results.md` §8, `prototype/gdn_gated_norm/run.py` (`write_bf16_raw`, `verify_cases`, `--selftest`) |
| **a device fault on a wave-only path presents as a hang, not a crash.** A staged launch plan has READ pointers as well as write pointers — `attn_core`'s staged branch reads `CoreArgs::qrope`/`krope` at `ROW_H * qrow` — and if only the write half is set, the row dereferences a null base while the host sits in a stream sync busy-polling a faulted queue: ~10 cores, zero logits files, no exit, for as long as nobody looks at the dump. The dump names both: `kernel: attn_core(CoreArgs)` and `faulting addr = ROW_H * qrow * 2`, so `0x1000`/`0x3000` are rows 1 and 3. It also means a path that exists only for `rows > 1` is invisible to every one-row gate the tree has | `engine/results.md` (“Q5 / P8”), `engine/kernels_attn.cpp`’s launch-plan note |
| **a requirement written after the object it constrains is a note, not a forcing — and an aggregate verdict that does not name its denominator cannot fail.** `gate.py` raised `--prefill-logits all` on the statement *after* it assembled the argv, so the engine ran `last` and wrote 1 of 1024 pinned rows while the gate’s own note claimed `all`; the row section then counted `1023 missing` and carried on into `check(not outside, “every judged prefill row is inside the floors”)`, which passed over the one row that existed. A device run therefore reported an instrument gap as a numerics result. Correct, twice over: decide such a requirement before the argv exists and assert it on the argv (`cmd_flag`), not on the variable the forcing mutates; and write the verdict as `total == n_rows and not outside`, aborting a coverage gap before any numerics section runs. Then witness it by stubbing the predicate to always pass — if no case changes colour, the check was decoration | `engine/gate.py` (`prefill_row_coverage`, `cmd_flag`, `selftest_prefill_rows`), `prototype/attn_decode/results.md` §18 |
| **a check whose independent variable is not the variable under test cannot adjudicate it — label and record the result, never delete it.** The §17 `--attn-tile 1` run reported 26 failures headed by section (h)’s batched-vs-carried wave-width comparisons, which hold the attention form identical on every side (`run_batched_ab` passes `--attn-tile` to all of them) and vary `--wave-rows`. “Run both sides against the same form” was therefore already true and could not help; what was missing was attribution. `--scope form` defers those output comparisons: they still run, print as `[DEFERRED-PASS]`/`[DEFERRED-FAIL]`, and land in `wave_carry_deferred.json` beside the dumps so “does it fail identically on both forms?” is a diff of two files rather than a reading of two logs — while `--scope full`, the default, counts them again | `engine/gate.py` (`SCOPES`, `defer`, `WAVE_CARRY_REASON`), `prototype/attn_decode/results.md` §18.2 |
| **a hoisted scalar constant is a different number, and a CPU mirror is blind to it.** Writing `q̃ = qv * (rsqrtf(s) * SCALE)` where the shipped kernel computes `(qv * rsqrtf(s)) * SCALE` is a different product tree: on device it moved **93,522 of 131,072 `o` elements while leaving the state bit-exact** (only `q̃` carried the factor). The CPU mirror of the same recurrence could not see it, because the mirror reproduces the *formula* and two sides that share a formula agree — what bit-identity is owed is the **source text's association**, not the algebra. When a form claims bit-identity with a shipped kernel, transcribe the shipped expressions character for character, and remember §04's caveat "a mirror cannot see FMA contraction and the device intrinsics" covers association too | `engine/gdn_collapse_kernels.h` (q̃), `prototype/gdn_collapse/results.md` §2 |
| **an instruction-count cost model assumes issue slots the machine does not have at low occupancy.** Route A was priced bottom-up at 11.4 ms from 1,370 instructions/token/useful-wavefront — one issue slot per SIMD per cycle — and measured **112.04 ms, 9.8× above it**, because the grid realises ~1.6 *working* wavefronts per CU (one warp per block drives the recurrence, three park at the norm barrier), so nothing is available to schedule while a lane waits on a dependent load and the cost is latency per token, not instructions per token. Same class as the chunked scan (priced ≥1.5×, measured 5.9×). Corollary that makes it testable rather than rhetorical: `#pragma unroll 8` on the two serial `i` loops was bit-identical and changed 112.8 → 113.4 ms, so the loop was not latency-bound *inside* the loop — the stall is the whole per-token dependency chain at 40 % lane coverage. **Price a serial kernel in cycles-per-dependency at its realised occupancy, or do not price it** | `prototype/gdn_collapse/results.md` §4–§5, `prototype/gdn_collapse/collapse_price.py` |
| **a gate driver whose parser silently matches nothing reports PASS.** `prototype/gdn_collapse/run.py`'s first version had a regex that matched zero `CASE` lines, so its per-case assertions ran over an empty list and the driver printed PASS over a matrix it had not read. Any driver that parses another program's output must require the number of parsed records to equal the number of records printed (and a fixed matrix to have its expected case count) — a check whose input list can be empty cannot fail | `prototype/gdn_collapse/run.py` (`CASE_RE`, `run()`), `prototype/gdn_collapse/results.md` §1 |

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

## Subagent run deadlines: pass 2 h explicitly, every launch

`pi-subagents` has **no project-scoped timeout setting**. A run deadline resolves: per-call
`options.timeoutMs` (alias `maxRuntimeMs`) → the agent's own frontmatter `timeoutMs` →
the *user-global* `~/.pi/agent/extensions/subagent/config.json` `timeoutMs` → built-in 30 min.
Project `.pi/settings.json` `subagents.agentOverrides.<name>` cannot carry it (`applyBuiltinOverride`
in `pi-subagents/src/agents/agents.js` has no `timeoutMs` branch), and the only project-scoped
*enforced* default is a `.pi/agents/worker.md` that **replaces** the builtin worker's prompt —
not worth it. (This repo has no `.pi/` directory at all today.) So the lever here is the call:

```js
subagent({ agent: "worker", task: "...",
           options: { timeoutMs: 7200000, checkpointBeforeDeadlineMs: 600000 } })
```

- **`timeoutMs: 7200000` (2 h) on every worker launch in this repo.** Omit it and you get
  whatever the previous session guessed — lately 90 min, which is what has been killing workers.
- `checkpointBeforeDeadlineMs: 600000` (async single-agent runs only) steers the child to
  checkpoint and stop 10 min early, so a long task hands back changed files, build/test state
  and remaining work instead of being killed mid-tool. Expiring the deadline is terminal: the
  child session is disposed, its open tool call aborted, acceptance unevaluated.

Why, from this repo's own artifacts (`~/.pi/agent/sessions/--home-erokhins-erokhins_projects-qwen3.5-0.8b--/subagent-artifacts`,
73 recorded runs): **21 died on the run deadline** — 16 at 90 min, 2 at 80, 2 at 30, 1 at 120 —
while *successful* workers ran up to 161 min, seven of them finishing at ≥80 min. 90 min sits
inside the productive distribution, not above it. The killed runs were still emitting tool
results at the deadline (one lost its report one line after its own gate printed `SELFTEST PASS`,
the 120-min one died just after printing its `RESULT` lines), and their longest inter-event gaps
were 7–18 min of model generation or one long device `bash` — this box serves its own tokens on
the same GPU, so a `high`-thinking turn on an 80k-token context costs minutes. Task length, not
a wedge, which is why the deadline (not `toolTimeoutMs`) is the right knob.

Two honest caveats: the one run that already carried 120 min died at it too, so a 161-min-shaped
task must be split or resumed rather than out-waited; and 12 of the 21 deaths also read "Child
failure followed session compaction and agent settlement", a parent-side compaction effect that
no child deadline fixes.

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

## Execution policy: keep moving, report at milestones (user, 2026-10-14)

- **Do not stop between milestones to ask.** Once a funded package has its decision, execute through the queue:
  dispatch, verify on device, commit, take the next item. Stop only for a decision that is genuinely the user's
  (a new band wider than an accepted one, a scope change, or a destructive action) or when the queue is empty.
- **After every milestone, report status and the tokens/s state.** A milestone = a landed and verified change to
  engine behaviour, a gate/admission decision, or a measurement that changes a plan. The report must include, in
  plain numbers: what was measured or shipped, and the current honest headline — decode tok/s at 128 lanes and
  prefill tok/s at 1024 lanes — with the commit that earned the number. If the headline did not move, say so
  explicitly; an unchanged number reported honestly is more useful than silence.
- **Fix-first over speed, pre-authorised (the branch-2 rule).** If a correctness defect is proved on the shipping
  path — e.g. the engine disagreeing with the pinned HF reference above `MAXSC` at a row with decision room to
  spare — **it outranks all pending speed work** (P9, attention tiling, the 5k target) until fixed or explicitly
  waived by the user in those words. Do not queue it, do not "bank the win first", and do not soften a gate to
  route around it. Report which rows fail, how many, and their decision room before choosing the remedy.
