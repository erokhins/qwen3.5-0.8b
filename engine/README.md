# Engine (HIP)

The deliverable: C++/HIP for `gfx1151`, gated by `../speccheck/`. The Python
implementation in `../speccheck/` validates the specification and is **not** the engine.

What exists now is **W5's first half: weight residency** (`bench/build-queue.md` W5).
There is no token loop, so nothing here produces a token.

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

* No token loop and no `decode_step()` — the kernels are still landing in `prototype/`.
* **No per-request state buffers**: no GDN `S` (18.0 MiB), no conv window (0.63 MiB), no
  KV cache (§08 sizes). W5b owns those.
* No kernel calls, no policy the spec has not fixed, no prefill, no sampling.
* Slab allocation, multi-shard streaming, and loading are unchanged from the per-tensor
  pattern described above (fine at 4 % of the pool; see `results.md` for what it costs).

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
