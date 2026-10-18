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
| a poll loop that `pgrep -f`s a script named in its own command line matches **itself** and never breaks — wait on a sentinel the job writes (log line, `touch` marker) instead | `progress/2026-10-11/commit-review-2026-10-11.md` |
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
| **two self-consistent layouts joined by a branch are invisible to both sides.** `attn_tile_wave` keyed its split partial by head-**in-group** while `attn_core` keyed by the FULL head, and `attn_combine` had one branch per layout: each form agreed with itself, every per-form check passed, and above `MAXSC` two lane blocks of one row wrote the same `2+DH` floats, so half of every streaming row came out as the other KV group's answer bit for bit (§19's room-13.25 rows). Diagnosed as an address, not arithmetic — the mirror's own first straddling print showed `cos 0.449` on ONE head-pair, not a gradient. Corollary that costs more: a route no fixture reaches has **no tests, including tests of the gate** — the same mirror had been mis-normalising the streaming merge since §12 (returned `(m,acc)`, caller divided by 1) and nothing could notice, because every pinned prompt is under 512 lanes. Fix both ways round: one address expression shared by writer and reader, and make the unreachable route reachable in a gate before trusting any claim about it | `prototype/attn_decode/results.md` §20.1–§20.3, `prototype/attn_tile_model.py` (`PartBuf`, `merge_partials`), `engine/kernels_attn.cpp` ("`tile_kernel` must not decide an address") |
| **a negative control with an empty must-catch set is not a control.** `partalias` (re-injecting the aliased partial slot) scored a clean **0 of 1,024** when the gate ran only the pinned spans — a one-pass row writes no partial, so it has no slot to steal, and every pinned prompt is one-pass. The control became meaningful only when the wide anchors and a synthetic tile-boundary wave (rows of exactly 512/513/514/515 lanes) became mandatory cases of the gate. Same lesson as `mlp_gemv` and `mixed_bw`, one layer over: after writing any control, ask what geometry can make it fire, and DERIVE the case list from that — then check that the must-not-catch set is also derived (a row at or below `MAXSC` structurally cannot be hit, which is a fact about the geometry, not a wish) | `prototype/attn_tile_model.py` (`wide_case`, `--neg-wide`, the derived blind sets), `prototype/attn_decode/results.md` §20.6 |
| **a traffic claim is a property of the staging loops the source contains, not of the tile shape you booked.** §12 claimed G×M re-use of K while its loop was `for (p = w; p < pairs; p += nw)` — one fetch per (row,head) **pair**, i.e. G identical fetches per lane row and a real re-use of **1x**, not the 4x booked — and it declared no `s_v` at all, so pass 2 re-read V from global at the use site. The honest lever was G (one block per row, both K and V staged per block), and the counterfactual is worth quoting: hoisting K and leaving V where it was buys 1.6x, not 2x, because V is half the bytes. Read the re-use factor out of the header (`stage_copies` looks for the loop shape) instead of asserting it, and check declared LDS against reach (`lds` FAILs an undeclared V stage rather than trusting the write-up) | `prototype/attn_tile_model.py` (`stage_copies`, `part_layout`, `lds`), `prototype/attn_decode/results.md` §20.5, §20.7 |
| **a check whose reference side is derived from the thing under test cannot adjudicate it.** The first straddling `compare` read the partial layout once, out of the tiled header, and applied it to BOTH sides — so it compared two aliased waves and reported the fixed geometry as the one that moved 512 rows. The frozen form must be modelled with its own spelling always. This is the sibling of the `--scope form` row above: there the independent variable was wrong, here the *reference* was. Related and cheaper to fix: `gate.py`'s `--attn-tile` default and `decode_run.cpp`'s must move in one commit, or the gate silently judges a different binary than the one the engine ships (the NUMERICS echo cross-check is what catches it) | `prototype/attn_tile_model.py` (`compare`), `engine/gate.py` (`--attn-tile` help) |
| **a ruler re-taken in a different window is not an A/B.** `tools/prefill_sweep.py` under the new default read 64 and 256 lanes *slower* than the ledger's previous table (708 vs 770, 1,893 vs 1,971) while a paired A/B inside one lock hold read the same two widths *faster* (83.26→80.91, 131.78→122.12 ms). The dispatch-bound bimodality in `bench/performance-status.md` §6 swallows 5–10 % at short prompts, so a short-prompt row only means anything when both sides were measured in one window — quote the pair, state the spread, and let the long-context row carry the headline (1024 lanes: 385.15→307.46 ms; 3072: 1812.08→931.13 ms) | `bench/performance-status.md` ("Headline tables after the tiled attention wave ships") |
| **a hoisted scalar constant is a different number, and a CPU mirror is blind to it.** Writing `q̃ = qv * (rsqrtf(s) * SCALE)` where the shipped kernel computes `(qv * rsqrtf(s)) * SCALE` is a different product tree: on device it moved **93,522 of 131,072 `o` elements while leaving the state bit-exact** (only `q̃` carried the factor). The CPU mirror of the same recurrence could not see it, because the mirror reproduces the *formula* and two sides that share a formula agree — what bit-identity is owed is the **source text's association**, not the algebra. When a form claims bit-identity with a shipped kernel, transcribe the shipped expressions character for character, and remember §04's caveat "a mirror cannot see FMA contraction and the device intrinsics" covers association too | `engine/gdn_collapse_kernels.h` (q̃), `prototype/gdn_collapse/results.md` §2 |
| **an instruction-count cost model assumes issue slots the machine does not have at low occupancy.** Route A was priced bottom-up at 11.4 ms from 1,370 instructions/token/useful-wavefront — one issue slot per SIMD per cycle — and measured **112.04 ms, 9.8× above it**, because the grid realises ~1.6 *working* wavefronts per CU (one warp per block drives the recurrence, three park at the norm barrier), so nothing is available to schedule while a lane waits on a dependent load and the cost is latency per token, not instructions per token. Same class as the chunked scan (priced ≥1.5×, measured 5.9×). Corollary that makes it testable rather than rhetorical: `#pragma unroll 8` on the two serial `i` loops was bit-identical and changed 112.8 → 113.4 ms, so the loop was not latency-bound *inside* the loop — the stall is the whole per-token dependency chain at 40 % lane coverage. **Price a serial kernel in cycles-per-dependency at its realised occupancy, or do not price it** | `prototype/gdn_collapse/results.md` §4–§5, `prototype/gdn_collapse/collapse_price.py` |
| **a gate driver whose parser silently matches nothing reports PASS.** `prototype/gdn_collapse/run.py`'s first version had a regex that matched zero `CASE` lines, so its per-case assertions ran over an empty list and the driver printed PASS over a matrix it had not read. Any driver that parses another program's output must require the number of parsed records to equal the number of records printed (and a fixed matrix to have its expected case count) — a check whose input list can be empty cannot fail | `prototype/gdn_collapse/run.py` (`CASE_RE`, `run()`), `prototype/gdn_collapse/results.md` §1 |
| **a report's "bit-identical everywhere" is a claim about the case list its author ran, and a timing harness often carries a wider list than the gate.** `burst_m16.sh` includes a case (`short-64`: 64 rows @ lanes 64–127, `split=1`, `full=0`) that returns `ndiff = 131072` — every element — while the M = 16 form's own gate (spans 0/1/3/16) printed 40/40 PASS and the handover said `ndiff = 0` "everywhere". The two lists were not the same list. Correct: have the timing harness re-run the admission set and refuse to print a millisecond figure if it goes red (the harness did exactly that: `NO-TIMING: admission is red`), and when reading a report, ask which cases were compared — then look for the signature: `ndiff == nrows × width` **exactly** is a whole-tensor difference, i.e. a changed work split or a wrong index, not a wrong value | `prototype/attn_m2/burst_m16.sh`, `prototype/attn_m2/gate_m16.py`, §27 of `bench/performance-status.md` (D59) |
| **an admission over a fixture list typed from memory is not an admission.** Three gate admissions this session claimed `18/18 — nine canonical fixtures × both routes`; **seven of those nine names do not exist** in `data/refs/`, `gate.py` exits with `fixture X has no manifest`, and a harness doing `grep -c PASS` prints `PASS=0` for them — which reads like a red result about the engine and is actually an error exit from the harness. The total looked plausible because it was arithmetic on a list nobody enumerated. Correct: derive the population (`for d in data/refs/*/; do [ -f $d/manifest.json ] && …`), assert `cases == |population|`, and treat any `PASS=0` as an unclassified run until its own tail has been read. Companion case, cheap to disprove and worth 60 s: an **existing** fixture can still be unadjudicable by a given invocation style (`p300` → `IndexError` in `pairing_checks`, `corpus48engine` → `KeyError: 'sources'`) — run the **base** binary, see the identical crash, and then report it as a pre-existing invocation gap instead of dropping the case | `bench/performance-status.md` §15.2, D40, this row |

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
not worth it. (The only `.pi/` in this repo is `engine/.pi/subagents.json` — harness-written
plugin state, now gitignored — and there is no `.pi/agents/`.) So the lever here is the call:

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
- **Commit messages: a short header, then a body that explains.** Header ≤ 72 chars, in the form
  `area: what the commit does` (`engine:`, `prototype/gdn_decode:`, `tools:`, `docs:`, `bench:`,
  `specification:`), one clause, no trailing punctuation — `git log --oneline` is the only view most
  readers get. Then `\n\n` and a body in short paragraphs saying **what** changed and **why** it was
  needed, including the number or the failure that forced it and a pointer to the file that holds the
  evidence (`prototype/<op>/results.md` §n, a `bench/` doc). A one-line message is only correct for a
  commit whose whole content is obvious from its diff — a typo, a renamed file. The 405 history
  commits were rewritten to this shape on 2026-10-16 by `tools/commit_msg_rewrite.py`, which refuses
  a set that loses a word or leaves a `TODO-HEADER` sentinel; do not hand-edit a `results.md` line
  number into a message that a later rewrap would move.

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

## Citation hygiene (added 2026-10-16; census in `progress/2026-10-16/archive-review-2026-10-16.md`)

Appended at the end of this file deliberately: `progress/2026-10-16/cleanup-audit-2026-10-16.md:179`
cites `AGENTS.md:172`, and anything inserted above it would silently re-point that cite while
still resolving as a line. That is the trap this section describes.

- **A line-form citation (`path:138,314`) is a claim about a file's line count, not just its
  identity.** 96 of the 101 such citations in this tree live in `progress/`, which rule 2 freezes,
  so any edit that adds or removes a line in a cited file silently rots them — the cite still
  resolves to *a* line, which is why no checker calls it broken. The guard for touching a cited
  file is `wc -l` before and after, and journals take same-line substitutions only. If a section
  must go, re-point the archive's cites first, and for `bench/performance-status.md` that needs
  the user's explicit words: the ledger's §1–§5 are banner-marked instead of deleted for exactly
  this reason. Check with `tools/doc_ref_check.py` (paths + anchors + `§labels`) and
  `tools/citation_census.py --anchors` and `tools/citation_rot_check.py` (the text at a `path:NNN` coordinate must match `HEAD` — the one drift the other two are blind to).
- **A `§label` citation that names no file resolves only in a reader's memory.** `engine/gate.py`
  carried 17 `§A1…§A4` cites into an audit document without ever naming it — the band they
  justify is enforced from those rows. Write `§N of <path>`, never a bare `§N`, when the target is
  in another file. Witness: `LABEL-UNNAMED` in `tools/doc_ref_check.py` (91 → 74 when the
  seventeen were fixed).
- **A checker's clean bill measures the population it scanned.** `tools/doc_ref_check.py` read
  only `*.md` when it was written, so every citation inside `engine/gate.py`, a kernel header or a
  writer script was invisible to it; and of the 96 paths it first called missing, 75 are scratch
  directories recorded as history in `progress/` and are not defects at all. Before trusting any
  count, ask what was walked and what was filtered — and when you extend a checker, expect a
  *larger* number, not a smaller one.

## Numeric format: delegated, with a gate-shaped test (user, 2026-10-16)

Appended at the end deliberately, for the reason stated in the section above it: `progress/2026-10-16/cleanup-audit-2026-10-16.md:179` cites `AGENTS.md:172`, so inserting anywhere earlier would silently re-point it.

The question on the table was whether switching a hot path to fp16 needed a new numerics band, i.e.
a user decision. It does not, and the standing rule now is:

- **Any 16-bit or 32-bit arithmetic may be adopted if the existing gates stay green.** No new band,
  no new approval round: bands already bought are wide enough, and a green gate set *is* the decision
  procedure.
- **If a gate goes red, ask whether the check can be widened without materially weakening it.** Here
  that has a concrete test, not a mood: a widened band must still catch every injection in the
  must-catch set that the negative controls derive from geometry (see the traps table — hand-listed
  must-catch sets rot into "everything" and prove nothing). If it still catches them, adopt it and
  record the new band beside what witnesses it.
- **Only a change that neither passes nor can be widened is a user decision.** Do not pre-emptively
  park work behind a band question that the gates would have answered.

Why the gates are the right instrument for fp16 specifically: fp16 is 1+5+10 against bf16's 1+8+7,
so inside fp16's range it is *more* precise than the format every band is calibrated to. The only
new failure modes are overflow above 65504 and flush-to-zero below ~5.96e-08, and those are a range
census, not a band negotiation — `tools/fp16_range_probe.py` over every pinned dump answers it in one
pass (activations peak 82.5, logits 35.5, weights 0.17, nothing over 65504; decision log D14).

Two things this rule does **not** cover, both still stop-and-ask: **int8/int4 quantisation** (its error
is orders of magnitude larger than a format change and would need bands materially wider than today),
and **widening a band that a pinned gate depends on for its discriminating power** — if the only way
to make a switch pass is to loosen a check that currently catches a real bug, that is the second
bullet's failure case, and it is reported, not routed around.

## Two provenance rules from a retracted headline (user-visible, 2026-10-16)

Appended at the end for the standing line-citation reason; the entry it belongs to is decision log
**D17** in `progress/2026-10-16/prefill-decision-log-2026-10-16.md`.

- **Check a number's file for a correction section before letting it size a route.** A WMMA route was
  priced at 11.2× the engine's rate against 4.696 TFLOP/s, which `devices/strix-halo-compute.md`
  had already retired in a section titled *"Second correction, same day: every rate above is an
  F32-`C` rate, and it is 6–8× below the same call"*. The engine was on the fast form all along, so
  the true prize was 1.4–1.6×. This repo corrects itself in place and marks the dead figure; quoting
  across that mark is the failure, not the arithmetic.
- **A census must name the files it could read.** "0 of the 55 `gfx1151` code objects contain a matrix
  instruction" was taken over the `*.hsaco` fallback libraries that `llvm-objdump` parses, and missed
  the tuned kernels in CCO containers it refuses — and the engine's measured 33–37.7 TFLOP/s is
  *above* the 2-MAC vector roof of 29.63, which proves the tuned path is a matrix path with no
  disassembly at all. "0 of 55" and "0 of the readable 55" read the same and mean different things;
  write the second form until the first is actually true.
- **A bytes-demanded ÷ time figure near the DRAM plateau is not evidence of a bandwidth wall.** Three
  days of prefill planning (D8, D9, D16, D19) read `attn` as bandwidth-bound because its demanded census
  divided by its measured time landed at 225–248 GB/s against a 237 GB/s ceiling. A bit-identical
  instruction cut (116 issue slots per touch → 45) then made it **1.521× faster with the census
  unchanged** — which puts the demanded rate at 345–381 GB/s, *above* the ceiling the part cannot
  exceed. Demanded bytes are not DRAM bytes: L1/L2 absorb the reuse. The falsification is cheap and
  should be run before a traffic route is funded — cut instructions or traffic and see which one moves
  time, and check whether the implied rate is even achievable. Related and subtler: a model fitted to a
  measurement cannot falsify itself (the byte model fit three widths to within 5 % and still described
  nothing) — see D21.
- **a sweep timed at a width the shipping path never issues is a fact about the library, not the
  engine.** `engine/decode_step.h:49` is `kMaxWaveRows = 512`, so a wave is at most 512 rows and a
  4096-lane prefill is **eight sequential 512-row waves**; the engine never issues an M = 1024 or
  M = 4096 GEMM. The first read of `prototype/gemm_algo_sweep` priced the win at the M = 4096 rows
  (`q_proj`: engine 1.7459 ms vs library 1.0534 ms, 1.66×) and the real prize at M = 512 is 1.17×
  — 2.0 % of the wave, not 5 %. Before aggregating any per-shape sweep, derive the call width the
  engine can actually emit and aggregate at that width; keep the other widths only as a bound on a
  future geometry, labelled as such. | `prototype/gemm_algo_sweep/results.md` §3a |

## A subagent "needs attention (no observed activity for 300s)" notice on this box is usually the child's own compaction, not a wedge

Appended at the end for the line-citation reason stated at the top of the citation-hygiene section.

Both workers launched in the 2026-10-16 prefill session were reported stalled at exactly 300 s of no
observed activity, and **both were mid-work**: `tools/subagent_stall_check.py` over their event logs
reads 3 `compaction_start(reason=threshold)` events, 2 attention notices, and
`attention_directly_after_compaction=2` — every notice was preceded immediately by a compaction. The
mechanism is this file's own premise applied one level over: the children are served by the same
checkpoint on the same iGPU, so a threshold compaction is a multi-minute generation with no tool call
inside it, which is precisely what the stall detector cannot see past.

Consequences for the parent, in order:

- **Read `events.jsonl` before interrupting.** A `compaction_start` on the line above the notice means
  the child is alive and the correct action is to wait, not to `interrupt`, `stop`, or re-brief. An
  interrupt aimed at a compacting child destroys work that was about to resume.
- **Expect the notice after any long CPU analysis**, since long analysis is what fills the child's
  context. A worker that goes quiet after its third file read is normal here; one that goes quiet
  mid-device-run is not, and that distinction is what the event log's last real tool result shows.
- **Tell the worker to checkpoint *before* the compaction, not after.** The useful version of this
  arrived by accident on 2026-10-16: steering a child to append its partial tables to its output path
  as its next tool call means the work that survives a compaction is on disk rather than in a window
  that has just been summarised. `a3e0693f` lost real numbers that way and its checkpoint was
  reconstructing from memory of a table it had already printed.
- **The 300 s constant is the detector's, not the machine's.** Two notices at the same interval, on two
  different runs, in unrelated phases, is a signature of a timer; look for a periodic cause before
  diagnosing the child.

## A paired A/B against a stale binary reads as a clean null — assert a behavioural canary inside the window (2026-10-17)

Appended at the end for the line-citation reason stated in the citation section.

`engine/build.sh` is not bit-reproducible on this box, so `cmp` between two saved binaries proves only that
they are two builds, **not** that either carries the change under test. The first D30 1.7′ A/B therefore timed
a pre-change binary as the `new` side and reported "nothing moved" — a null that would have refuted an item worth
−0.5 % of wave — while every parity field in the run stayed green:

- `plan_match=1` cannot detect staleness, because `wave_plan_bill` is compiled *into* `decode_run`: a stale
  binary agrees with itself. Any self-consistency check that lives inside the binary under test is blind to the
  one failure mode that matters here.
- Discarding the build log (`>/dev/null 2>&1`) removed the only other signal, and this session had already been
  bitten by discarded output once before (§"a gate driver whose parser silently matches nothing reports PASS").

**The rule: every A/B script asserts a behavioural canary before it times anything** — a field the change is
*required* to move, read in the same breath as the measurement, aborting the script if it does not read as
expected on both sides. In §13 that was the per-token bill line (`new` must print `cast=97 other=20 total=406`,
`base` must print `115/424`); for a change that moves no plan field, pick a launcher counter or a dump hash.
Reference implementation: `tmp/gcc/run_ab_17p_retake.sh`. The general form is the same class as every other
empty-parse failure logged in this file — a check whose input can silently fail to arrive must be shown to have
arrived, in the window that produces the number.

## Counter, occupancy and census traps from the attention attribution (2026-10-18)

Appended at the end for the standing line-citation reason stated above (three prior sections did the same).

- **A HIP capability query on this SKU reports against a halved SM count, so its numbers are 2× off the
  architecture.** `hipGetDeviceProperties` gives `sharedMemPerMultiprocessor = 65536` *and*
  `multiProcessorCount = 20` *and* `maxThreadsPerMultiProcessor = 2048` (= 64 wavefronts, against an
  architectural max of 32). Read **LDS from KFD** (`/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties`
  → `lds_size_in_kb 64`, `simd_per_cu 2`, `simd_count 80`) and CU/SIMD counts from `rocminfo`, never from HIP.
  Evidence: `tools/kernel_occupancy_probe.cpp`, `prototype/attn_wmma/evidence/{device_facts,occupancy}.txt`.
- **clang's `-Rpass-analysis=kernel-resource-usage` line `Occupancy [waves/SIMD]: 16` is a constant, not a
  measurement.** It prints 16 for every kernel in the TU regardless of LDS or VGPR — `attn_core` at 38,944 B
  per block and `attn_combine` at 0 B alike. Any claim that rests on "occupancy unchanged at 16 waves/SIMD" is
  resting on a check that cannot fail; the real bound is LDS-bound (`64 KiB / 12,288 B = 5` blocks/CU) and comes
  from the probe above. `prototype/attn_valu_slim/results.md` §"Lever 0" is the claim this row retires.
- **A `rocprofv3` pmc CSV row is per (Dispatch_Id, Counter_Name), and `--time-waves` replays the prompt wave
  three times** (the run, the burst pass, the stage pass). A raw `sum(Counter_Value)` is therefore 3× a pass and
  reads plausible while being wrong by exactly that factor; and matching on a bare kernel name fails, because
  `Kernel_Name` carries the signature (`attn_slim_wave(CoreArgs)`). Both mistakes made a correctly-collected
  counter set look like "no rows, counter uncalibrated". Correct: dedup by `(Dispatch_Id, Counter_Name)`, match
  on the name prefix, and divide by a pass count **derived** from a kernel whose per-pass site count is known
  (`attn_prep` = 48 sites at 4,096 lanes), as `prototype/attn_wmma/w2_reanalysis.py` does.
- **A rate must be compared against the ceiling its own instruction class can occupy.** Judging a kernel that
  issues only scalar `v_fma_f32` against a matrix-unit roof (22.3 TFLOP/s here) overstated one gap by 3× and
  nearly funded a route on the wrong premise. The ladder in `bench/performance-status.md` §26 is the fix:
  scalar `v_fma_f32` 7.30 → `v_dot2_f32_bf16` 23.28 → WMMA 52.63 TFLOP/s, measured in
  `prototype/issue_rate_probe/results.md` §1. Also check **who wrote** the kernel you are comparing against:
  the engine's own 40 kernels contain **zero** matrix or dot-class instructions, so every matrix rate in this
  ledger belongs to hipBLASLt, whose tuned kernels are unreadable CCO containers.
