# Engine (HIP)

The deliverable: C++/HIP for `gfx1151`, gated by `../speccheck/`. The Python
implementation in `../speccheck/` validates the specification and is **not** the engine.

What exists now is **W5's first half: weight residency** (`bench/build-queue.md` W5) plus the
**first pieces of W5b**: per-request state and the launchers that will drive it. There is no token
loop on disk yet, so nothing here produces a token — see *Hand-off* below for the exact line.

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

* No token loop and no `decode_step()` on disk — the launcher layer below it is, the loop is not.
* No kernel calls from a host loop, no prefill, no sampling, no `gate.py`.
* Slab allocation, multi-shard streaming, and loading are unchanged from the per-tensor
  pattern described above (fine at 4 % of the pool; see `results.md` for what it costs).

## `state.{h,cpp}` and `kernels.h` — W5b, compiled, **never run on a device**

* `state.h` declares `RequestState` (§08.1's inventory: GDN `S` 18.0 MiB F32, conv window
  [18][3][6144], KV for the 6 full-attention layers at 12288 B/token) plus `allocate` /
  `free` / `reset` / `check_invariants`. `state.cpp` `hipMalloc`s them on a 512-B boundary and
  implements §08.6's invariants. Two deliberate departures from §08.1's table, both open:
  **the conv window is F32, not the table's BF16** (the graduated `gdn_decode` conv kernel reads
  and writes it as F32; 36 cast launches/token is the wrong price for 0.6 MiB of a pool this model
  uses 4 % of), and **KV is head-major `[head][slot][dim]`** because that is the layout the
  graduated `attn_decode` kernel indexes. `state_sizes.py` prints both and flags the doc slips.
* `kernels.h` declares **one launcher per graduated kernel**. The kernels themselves are
  `prototype/<op>/<op>_kernels.h` — the device code of each graduated op, moved **verbatim** out of
  the binary its own gate signed off on (`tools/` has no checker for this; `git show` the extraction
  commits `5dbfb9a..cc24680`, whose diffs are a pure move plus one `#include`). `decode_step` must
  call these, never a copy.
* Each launcher family is its own TU (`kernels_gdn.cpp`, `kernels_glue.cpp`, `kernels_proj.cpp`)
  because the ops each define colliding macros (`D`, `H`, `HID`, `EPS`, `SCALE`, `FULL_MASK`) — one
  family per TU is a compile-time requirement, not a style choice.
* **Written so far:** `state.cpp`, `kernels_gdn.cpp`, `kernels_glue.cpp`, `kernels_proj.cpp` — all
  four compile clean (`hipcc -O3`, zero warnings). **Not written:** `kernels_mlp.cpp`,
  `kernels_attn.cpp`, `kernels_lmhead.cpp`, `kernels_cast.cpp`, `decode_step.{h,cpp}`,
  `prefill.{h,cpp}`, `decode_run.cpp`, `gate.py`. So `kernels.h` currently declares launchers with
  no definitions and nothing links; `engine/build.sh` still builds only `loader_test`.

## Hand-off — read this if you did not write it

Nothing below marked **device** has ever been executed. No buffer in `state.cpp` has been
allocated, no launcher has been launched, and the §07.1 block order exists only as comments in
`kernels.h`. "Compiles clean, never run" is the state of the work, not an oversight.

### CPU-only — safe to run while anything else is running

```bash
.venv/bin/python engine/state_sizes.py
# PASS. 18.000 MiB GDN S + 1.266 MiB F32 conv window = 19.2656 MiB constant per request;
# 12288 B/token KV; prints three OPEN doc slips (§08.1's 4 KiB/token/layer cell contradicts its
# own 12 KiB/token total; §08.1's BF16 conv cell; §00.5's 3.07 GiB is 3.00 GiB from the shapes).

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

.venv/bin/python speccheck/run_all.py
# CPU torch. Expected green as of the rope revert d06b4e3. Any failure is worth reporting; note
# that this suite's rmsnorm_gated is bit-exact against the pins only when the weight arrives BF16,
# so the A-02 knob (--keep-f32-scalars) moves it off bit-exactness by dtype alone.
```

Compile-only HIP (compiling is not running; allowed while the GPU is busy):

```bash
bash engine/build.sh                                   # loader_test only, today
for TU in state kernels_gdn kernels_glue kernels_proj; do \
  hipcc -O3 -Wno-unused-command-line-argument -I. -Iengine -c engine/$TU.cpp -o /tmp/$TU.o; done
bash prototype/gdn_gated_norm/build.sh                 # kernel + host driver + engine/loader.cpp
for OP in gdn_decode attn_decode glue proj_gemv mlp_gemv lm_head_gemv; do \
  bash prototype/$OP/build.sh; done                    # all six still build after the extraction
```

### Device — operator only, in a quiescent GPU pass (zero subagents, then `tools/gpu_lock.sh`)

```bash
tools/gpu_lock.sh acquire "engine W5b GPU pass" && tools/gpu_lock.sh util

# 1. residency gate must not have regressed
./engine/loader_test --no-verify
# expect: 335 tensors / 1.5457 GB resident, tied lm_head aliased not copied, PASS, ~7 s

# 2. §04.5 device gate (18 layers x 8 dec8 steps) + its six located negatives
.venv/bin/python prototype/gdn_gated_norm/run.py --gate          # add --time for the burst row
# expect: "DEVICE GATE PASS"; gated_out within 1 BF16 ULP of the pins with ndiff ~0 (the CPU
# mirror is bit-exact, so a large ndiff here means the kernel's per-head reduction or its cast
# order is wrong on device, NOT that the pins moved). Each negative must move gated_out and leave
# norm_pre_gate alone; w_f32/affine_uncast/single_cast/silu_rounded with norm_pre_gate 0/294912.

tools/gpu_lock.sh release "engine W5b GPU pass"
```

```text
3. NOT RUNNABLE YET — the token gate. Expected result once decode_run.cpp + gate.py exist:
     prefill_ids (17 tokens, data/refs/dec8/manifest.json) fed token-at-a-time, then 8 decode
     steps; the 9 sampled ids must equal greedy_continuation
     [760, 6511, 314, 9338, 369, 2972, 57590, 159034, 271]
     — prefill's last sample is 760, the 8 decode steps give the rest. Decode-step logits compare
     against decode{s}.logits_last.npy through prototype/bf16_ulp.py (report(n>ulp 1, cos >=
     0.9999)); the pins are the ONLY authority, and bare logits_last.npy is the PREFILL output
     (argmax 760), not the last decode step's — see §10.5's last-writer-wins trap.
     Do not add engine/decode_run to .gitignore as a directory pattern; exact path, the trap in
     .gitignore's comment block is a live one.
```

### What the next worker still owes W5b

In dependency order: `kernels_cast.cpp` (engine's own `cast_*`, declared already), then
`kernels_mlp.cpp` / `kernels_attn.cpp` / `kernels_lmhead.cpp` (one TU each, macros collide), then
`decode_step.{h,cpp}` in §07.1's order (§07.2 for the 18 linear layers, §07.3 for the 6 attention
ones, residual adds per §03.2, final norm-only glue, `lm_head_full` + host argmax with a strict `>`
so ties go to the **lowest** id per §03.4), then `prefill.{h,cpp}` (token-at-a-time through
`decode_step`, no chunked variant), then `decode_run.cpp` and `gate.py`. `attn_decode` must be
called `full=1, split=1, fused=1, score_f32=0, tile=8` and its `MAXSC` caps `lvalid` at 512, which
the dec8 fixture is far inside.

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
