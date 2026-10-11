# Engine (HIP)

The deliverable: C++/HIP for `gfx1151`, gated by `../speccheck/`. The Python
implementation in `../speccheck/` validates the specification and is **not** the engine.

What exists now is **W5's first half: weight residency** (`bench/build-queue.md` W5) plus **W5b's
token loop**: per-request state, one launcher per graduated kernel, and `decode_step`, which runs one
token through all of it. `engine/decode_run.cpp` drives it and `engine/gate.py` gates it against the
pinned `dec8` fixture. **Neither has been executed** — see *Hand-off*: the gate is written, the device
pass is not. No token has come out of this engine yet.

## `loader.{h,cpp}` — what it guarantees

Loads `models/Qwen3.5-0.8B` (shard taken from `model.safetensors.index.json`, per A-09)
and leaves the text path resident on the device.

* **§02.5 name normalisation, and nothing else.** `model.language_model.* → lm.*`,
  `mtp.layers.0.* → mtp.layer0.*`, `mtp.*` unchanged, `model.visual.*` skipped. A name
  that fits none of those is an error, never a silent skip (§10 gate G0). 335 loaded /
  153 skipped / 488 total.
* **One `hipMalloc` + one `hipMemcpy` per tensor.** Never `hipMallocManaged`: it degrades
  silently to ~52 GB/s near the pool limit on this box, `hipMalloc` fails loudly
  (`devices/strix-halo-unified-memory.md`). Bytes are uploaded verbatim — no transpose, no
  reshape, no dtype conversion; `shape()` is as stored, so `y = x·Wᵀ` (§01 preamble).
* **The tied head is one allocation under two names.** The checkpoint has no
  `lm_head.weight` (§01.1), so `lm_head.weight` is an alias of `lm.embed_tokens.weight`
  — same device pointer, no second 485 MiB copy. A checkpoint that actually carried
  `lm_head.weight` is refused rather than silently duplicated.
* **Every resident page is warmed** by one GPU read pass before the load returns, because
  the first GPU touch of fresh pages is ~35 GB/s instead of ~240 (same doc).
* **Lookup by normalised name** with shape/dtype/bytes: `at()`, `find()` (nullptr, never a
  wrong tensor), `ckpt_to_engine()` for the reverse direction, `allocation_count()`,
  `resident_bytes()`.
* **Verification is opt-in.** `Options::verify_device_bytes` copies every blob back once
  and SHA-256s it against the file — ~1.55 GB of D2H, so a normal load does not pay it.

## What this is NOT yet

* No `prefill.{h,cpp}`. A prompt is fed to `decode_step` one token at a time — which is what
  `gate.py` does and what a chunked prefill would have to agree with, but it is 17 separate calls
  where the reference does one chunked forward, so prefill is slow and its GDN layers run the
  recurrence where the reference ran the chunked form (§10.5). The token gate does not care; a
  prefill-throughput gate will.
* No MTP drafting, no batching, no sampler other than greedy, no KV eviction (§08.7).
* Slab allocation, multi-shard streaming, and loading are unchanged from the per-tensor
  pattern described above (fine at 4 % of the pool; see `results.md` for what it costs).

## `state.{h,cpp}` and `kernels.h` — W5b, compiled, **never run on a device**

* `state.h` declares `RequestState` (§08.1's inventory: GDN `S` 18.0 MiB F32, conv window
  [18][3][6144], KV for the 6 full-attention layers at 12288 B/token) plus `allocate` /
  `free` / `reset` / `check_invariants`. `state.cpp` `hipMalloc`s them on a 512-B boundary and
  implements §08.6's invariants. One departure from §08.1's table is now the spec's own reading and
  one is still only the engine's: **the conv window is F32, not the BF16 §08.1 used to print** (the
  graduated `gdn_decode` conv kernel reads and writes it as F32; 36 cast launches/token is the wrong
  price for 0.6 MiB of a pool this model uses 4 % of) — §08.1 and §00.5 were corrected to say so on
  2026-10-11 — and **KV is head-major `[head][slot][dim]`** because that is the layout the graduated
  `attn_decode` kernel indexes, which §08.5 still does not say. `state_sizes.py` prints the first as
  FIXED (and greps the spec pages to keep it true) and the second as an open item.
* `kernels.h` declares **one launcher per graduated kernel**. The kernels themselves are
  `prototype/<op>/<op>_kernels.h` — the device code of each graduated op, moved **verbatim** out of
  the binary its own gate signed off on (`tools/` has no checker for this; `git show` the extraction
  commits `5dbfb9a..cc24680`, whose diffs are a pure move plus one `#include`). `decode_step` must
  call these, never a copy.
* Each launcher family is its own TU (`kernels_gdn.cpp`, `kernels_glue.cpp`, `kernels_proj.cpp`,
  `kernels_mlp.cpp`, `kernels_attn.cpp`, `kernels_lmhead.cpp`, `kernels_cast.cpp`)
  because the ops each define colliding macros (`D`, `H`, `HID`, `EPS`, `SCALE`, `FULL_MASK`) — one
  family per TU is a compile-time requirement, not a style choice.
* **All ten TUs exist and compile clean** (`hipcc -O3`, zero stderr): `state`, `loader`,
  `loader_test`, the seven `kernels_*`, `decode_step`, `decode_run`. `engine/build.sh` links both
  binaries into `engine/build/`. **Not one of them has been executed**: the state buffers have never
  been `hipMalloc`'d outside `loader_test`'s own scratch, no launcher has been launched, and
  `gate.py` has never seen a logits file. "Compiles clean, never run" is still the state of the work.

## `decode_step.{h,cpp}` — one token, §07.1's order

`decode_step(weights, state, scratch, token, &logits)` embeds the token, runs 24 blocks (GDN §07.2 /
attention §07.3 by `layer_types`, each followed by the MLP §03.3), takes the final norm, launches
`lm_head_full`, and argmaxes on the host — strict `>`, so ties go to the **lowest** id (§03.4; the
fused-argmax kernel resolves them the other way, which is why the engine keeps the full logits).
It syncs once, at the logits. It contains **no model arithmetic**: every value comes from a kernel
that has its own gate.

Two things to know before reading it:

* **The norm pairing is §03.2's, and the final norm is not a launch.** Layer 0's mixer norm uses the
  glue kernel's norm-only form; after that each layer is mixer → glue(add + post-attention norm) →
  MLP → glue(add + the *next* norm), and layer 23's second glue call takes the final norm as a side
  effect of the residual add. 49 glue launches per token, which is the plan `prototype/glue/run.py`
  pins and `bench/gate-audit.md`'s §07 audit demanded; `gate.py` re-derives it in Python.
* **The conv window is rolled into scratch and copied back** (one D2D memcpy per GDN layer). The
  roll-back is the documented cost of `conv_kernel`'s `__restrict__` `win`/`new_win`, sized in the
  comment where it is spent.

## Hand-off — read this if you did not write it

Nothing below marked **device** has ever been executed. No buffer in `state.cpp` has been
allocated, no launcher has been launched, and the §07.1 block order exists only as comments in
`kernels.h`. "Compiles clean, never run" is the state of the work, not an oversight.

### CPU-only — safe to run while anything else is running

```bash
.venv/bin/python engine/state_sizes.py
# PASS. 18.000 MiB GDN S + 1.266 MiB F32 conv window = 19.2656 MiB constant per request;
# 12288 B/token KV = 2048 B/token/layer (14 KiB/token including MTP KV); 3.000 GiB at 262144 ctx.
# It prints three FIXED lines: §08.1's conv dtype, its KV and MTP-KV cells, and §00.5's full-context
# row are all corrected in `specification/`, and this script greps those pages for the corrected
# figures — edit a doc row back and the size check fails. Only §08.5's KV layout stays open.

.venv/bin/python prototype/gdn_gated_norm/mirror_probe.py dec8
# THE LOAD-BEARING CPU RESULT OF THIS PHASE: 144 cases (18 GDN layers x 8 decode steps),
# 294912 BF16 elements of gdn.gated_out. TWO rows are bit-exact (ndiff 0): the kernel's POLICY and
# speccheck.specimpl.rmsnorm_gated called the way speccheck calls it, i.e. with the norm weight in
# BF16 as `Weights` stores it. The probe asserts exactly that pair and nothing else ties. The
# A-01-as-written reading is refused, and how far depends only on the weight dtype it is handed:
# BF16-rounded-then-widened-to-F32 74860, raw F32 checkpoint weight 105683. (An earlier revision of
# this README called the oracle itself wrong on the strength of that 105683; it had been handed the
# raw F32 weight, which speccheck never does. Retracted in fc444d0, asserted here since.) See
# prototype/gdn_gated_norm/results.md §1.

.venv/bin/python prototype/gdn_gated_norm/run.py --mirror
# the same gate over all four fixtures (dec8 decode, prefix24dec8 decode, smoke decode, p300
# prefill = 10.2 M elements), under two seconds on this box. Ends "MIRROR GATE: PASS".

`.venv/bin/python engine/gate.py --selftest`
# Proves the token gate DISCRIMINATES, with no engine and no device: it re-runs gate.py in
# subprocesses against a CPU stand-in with three bugs injected — a wrong sampled id at a named step,
# a swapped top-two logits, and 5 % logits noise that leaves the argmax alone — and requires each to
# fail at the step it was injected at. Ends "SELFTEST PASS". `--dry-run` is the same stand-in without
# the injections, and is what exercises sections (a)–(d)'s parsing before a GPU window exists.

`.venv/bin/python engine/gate.py --no-engine`
# Sections (0), (c) and — with --selftest/--dry-run — the whole judging half of the token gate, with
# no binary and no device: it proves the dec8 fixture pairs with itself (the §10.5 trap, as 26
# assertions), runs engine/state_sizes.py against the sizes it will later compare with decode_run's,
# and requires three injected bugs to be caught at their own steps. Runs in about a second, and is
# the part of the gate a worker is allowed to run. Ends "ENGINE GATE PASS" with a note that (a),(b)
# were skipped (or, under --dry-run, that the stand-in stood in).

.venv/bin/python speccheck/run_all.py
# CPU torch. Expected green as of the rope revert d06b4e3. Any failure is worth reporting; note
# that this suite's rmsnorm_gated is bit-exact against the pins only when the weight arrives BF16,
# so the A-02 knob (--keep-f32-scalars) moves it off bit-exactness by dtype alone.
```

Compile-only HIP (compiling is not running; allowed while the GPU is busy):

```bash
bash engine/build.sh                                   # -O3: engine/build/{loader_test,decode_run}
for TU in state loader loader_test kernels_gdn kernels_glue kernels_proj kernels_mlp \
          kernels_attn kernels_lmhead kernels_cast decode_step decode_run; do \
  hipcc -O3 -Wno-unused-command-line-argument -I. -Iengine -c engine/$TU.cpp -o /tmp/$TU.o; done
# Re-verified on the current tree with these exact commands: twelve TUs, zero stderr (no warnings),
# and `engine/build.sh` links both binaries. That is the whole extent of the compile claim — neither
# binary has been executed.
bash prototype/gdn_gated_norm/build.sh                 # kernel + host driver + engine/loader.cpp
for OP in gdn_decode attn_decode glue proj_gemv mlp_gemv lm_head_gemv; do \
  bash prototype/$OP/build.sh; done                    # all six still build after the extraction
```

Two of the gate's four sections need no device at all, and one launches no kernels but still moves
bytes — the difference matters here, because a run that touches the pool while something else is
being timed contaminates that timing:

```bash
.venv/bin/python engine/gate.py --selftest      # no binary, no device, no allocation: four runs of
.venv/bin/python engine/gate.py --no-engine     # this file against CPU stand-ins. Safe anywhere.
./engine/build/decode_run --sizes-only          # NO kernel launches, but it loads the checkpoint:
                                                # 2.4 GB of hipMalloc + H2D + the warm read pass.
                                                # Run it in a GPU window, not beside a timing.
```

### Device — operator only, in a quiescent GPU pass (zero subagents, then `tools/gpu_lock.sh`)

```bash
tools/gpu_lock.sh acquire "engine W5b GPU pass" && tools/gpu_lock.sh util

# 1. residency gate must not have regressed
./engine/loader_test --no-verify
# expect: 335 tensors / 1.5457 GB resident, tied lm_head aliased not copied, PASS, ~7 s

# 2. §04.5 device gate (18 layers x 8 dec8 steps) + its six located negatives
.venv/bin/python prototype/gdn_gated_norm/run.py --gate          # add --time for the burst row
# PASSES as of the 2026-10-11 clean window: "DEVICE GATE PASS", gated_out worst case ndiff 1/2048 at
# max ULP 1 (= the reduction-order floor neg_probe.py predicted before the run), zeros 0 / NaN 0, all six
# negatives localising (w_f32/affine_uncast/single_cast/silu_rounded at norm_pre_gate 0/294912), burst
# 0.4227 ms per 144-job plan = 2.936 us/launch at spread 1.4 %. A LARGE ndiff here (not 1/2048) means the
# kernel's per-head reduction or its cast order is wrong on device, NOT that the pins moved. History:
# the first two runs failed on the gate's own fixture writer, not the kernel —
# prototype/gdn_gated_norm/results.md §7-§9, and the timing is bench/performance-status.md §6.

tools/gpu_lock.sh release "engine W5b GPU pass"
```

```bash
# E1 — CLOSED 2026-10-11, in a clean window. glue's gate at 1024 threads, the width the engine does
# NOT launch: PASS with a legal error profile identical to 256 (g1_n gated 0 / raw 1, adds bit-exact,
# 784/0, every negative caught at its own stage). So <256 in `kernels_glue.cpp` is a measured
# preference, not a numerical requirement. The numbers are `bench/performance-status.md` §6; to
# reproduce:
tools/gpu_lock.sh acquire "glue gate at 1024 (E1)" && tools/gpu_lock.sh util
.venv/bin/python prototype/glue/run.py --threads 1024
.venv/bin/python prototype/glue/run.py --threads 1024 --negative
tools/gpu_lock.sh release "glue gate at 1024 (E1)"
# Why it was run at all: 1024 changes the F32 summation order of the mean of squares, and ndiff is the
# only criterion that can see it, so "it passes" was never the result — "it passes AT THE SAME FLOOR"
# was. Re-running it is only worth another GPU pass if `prototype/glue/glue_pins.py` changes.

# E2 — CLOSED for the prototype half, 2026-10-11. Cold, on the prototype's 48-launch token plan, at
# rows/block 32: unpacked 0.0528 ms/launch = 65.1 % of the read probe against packed 0.0593 = 58.0 %,
# and unpacked won at every rows/block swept. So `PACK_X=false` in `kernels_proj.cpp` is a measurement
# and the ≈48 cast launches it bills back do not overturn the ordering. Numbers: §6 of the ledger.
tools/gpu_lock.sh acquire "proj_gemv PACK_X A/B (E2)" && tools/gpu_lock.sh util
.venv/bin/python prototype/proj_gemv/run.py --time --sweep
tools/gpu_lock.sh release "proj_gemv PACK_X A/B (E2)"
# The half still owed is NOT a measurement: running the same A/B through `engine/kernels_proj.cpp`
# needs a PACK_X switch in that launcher, which hard-codes the gated `false`. That is deferral #1 in
# `results.md`, and it is a code change with a timing attached, not a command waiting to be typed.
```

```bash
# 3. THE TOKEN GATE — the first time any engine code except loader_test touches the device.
#    One command; it builds nothing, runs engine/build/decode_run, and judges the result.
tools/gpu_lock.sh acquire "engine decode_step token gate (W5b)" && tools/gpu_lock.sh util
./engine/build.sh && .venv/bin/python engine/gate.py --build
tools/gpu_lock.sh release "engine decode_step token gate (W5b)"
# Expected on a first clean run:
#   (0) fixture pairing  — 26 PASS lines, no FAIL. These are the reference's own invariants
#       (`decode{s}.input_ids == greedy_continuation[s]`,
#        `argmax(decode{s}.logits_last) == greedy_continuation[s+1]`), so they cannot fail because of
#       the engine. If one does, the fixture moved: stop and re-read §10.5, do not touch the kernel.
#   (a) sampled ids      — the 9 samples are exactly
#                          [760, 6511, 314, 9338, 369, 2972, 57590, 159034, 271]
#   (b) logits           — per step, cos >= 0.9995 (§10.1 G5) and meanrel <= 5 % (§10.2), and the
#                          argmax must equal the pin's. ndiff/max-ULP lines are printed for the record.
#   (c) state sizes      — engine's bytes == state_sizes.py's, byte-exact, at cap 26.
#   (d) launch plan      — glue=49 proj=114 gdn=72 attn=6 mlp=48 cast=115 other=20 total=424
# If it FAILS, in this order:
#   1. Read the (a) lines first. A wrong id at step 0 (760 expected) means the prompt path, not the
#      decode path; a wrong id at step k>0 with step 0 right means the state did not survive a step.
#   2. `./engine/build/decode_run --ids 248045,846,... --steps 1 --out tmp/g` reproduces one step
#      with every intermediate still on the device; the per-step lines name the position and token.
#   3. A `(b)` failure with a correct `(a)` is a band problem, not a token problem, and the band is
#      §10's, not this gate's to loosen. Report it; do not edit the constant.
#   4. `(c)`/`(d)` failing while `(a)`/`(b)` pass means the state struct or the block plan drifted
#      from §00/§07 — decode_run prints both sides of each comparison.
```

## What the next worker owes W5b

Nothing on the code side — the loop, the driver and the gate exist. What is left is measurement and
the two named deferrals:

1. **Run the device pass above**, in a quiescent window, and land the result in `results.md` and
   `bench/performance-status.md` §6 (including the first ms/token figure, which the ledger has never
   had for the engine).
2. `prefill.{h,cpp}` — token-at-a-time through `decode_step` is already what `gate.py` does, so this
   is a chunked/parallel prefill and nothing else. It is a throughput item, not a correctness one.
3. The two levers `results.md` defers: packed projections (merges the 16-row `in_proj_a`/`in_proj_b`
   launches and several other small dots) and BF16-staged `mlp_gateup`/`mlp_down` (would remove 48 of
   the 115 casts/token). Both are launch-count and traffic wins; neither is a correctness item, and
   `gate.py`'s (d) section is what has to be updated if the plan changes.
4. `attn_decode` is called `fused=1, full=1, split=1, score_f32=0, tile=min(TILE, threads/warp)`, and
   its `MAXSC` caps `lvalid` at 512 — the dec8 fixture is far inside (max 25). Beyond 512 the engine
   needs the split-block form §07.3 describes; `kernels_attn.cpp` throws rather than truncating.

## Build and run the gate

```bash
./engine/build.sh                    # hipcc -O3 -Wl,-rpath,/opt/rocm/lib
tools/gpu_lock.sh acquire "engine loader_test residency gate"
tools/gpu_lock.sh util               # record it; numbers are provisional unless low
./engine/loader_test                 # full gate, with the SHA-256 verification
./engine/loader_test --no-verify     # skip the copy-back + hashing (~7.6 s)
tools/gpu_lock.sh release "engine loader_test residency gate"
```

`loader_test` prints and asserts every claim; any failure is a non-zero exit. It expects to
be run from the repository root (it calls `tools/weights_report.py` and
`engine/weights_reference.py` by relative path). Timings in `engine/results.md`.
