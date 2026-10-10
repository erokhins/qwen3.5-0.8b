# prototype/ — one directory per op, before it graduates to `engine/`

Convention (inherited from the old project and stated in `../AGENT.md` terms): each op
gets its kernel, its own gate script, and a `results.md`. **An op graduates to `engine/`
only when its gate passes**, and a gate is a script, never an eyeball. Current contents:

| dir | op | status |
|---|---|---|
| `lm_head_gemv/` | tied logits GEMV, bf16, 248320 × 1024 | **passes** §10.1 logits class at 242.1 GB/s (102 % of the read probe) — `lm_head_gemv/results.md` |
| `gdn_decode/` | S04.2–S04.4 decode step: conv, gates, gated-delta recurrence, 18 layers | **passes** all four functions (bit-exact conv, 1e-7 vs the spec recurrence) — `gdn_decode/results.md` |

Next in line: the gated full-attention decode step (§05), then the MLP GEMVs; then wiring
`engine/` to call these instead of simulating them.

## HIP on ROCm 7.1 / gfx1151 — the things that bit us

Saved here so the next kernel does not rediscover them.

- **Build:** `hipcc -O3 -Wl,-rpath,/opt/rocm/lib -o <bin> <src>.cpp`. Without the rpath the
  binary dies at load with `libamdhip64.so.7: cannot open shared object file`. `-O3` (or at
  worst `-O2`) is mandatory — timing an unoptimized build is banned by `../AGENT.md`.
- **CUDA-isms that do not exist.** `__nv_bfloat16` → `hip_bfloat16`; `__bfloat162float(x)` →
  `float(x)`; `__float_as_uint`/`__uint_as_float` → `memcpy` through a local.
- **`__shfl_down_sync` takes a 64-bit mask** in HIP (wave32 *and* wave64 targets), so
  `0xffffffffu` fails a `static_assert`. Use `0xffffffffffffffffull`.
- **The `.venv` torch is CPU-only.** No Python can touch the GPU here; every GPU number
  comes from a compiled binary. Python's job is preparing inputs and checking results.
- **torch refuses `bf16.numpy()`.** Dump raw bits with `t.view(torch.uint16).numpy().tofile(f)`
  and read them back as `uint16`. Converting *values* (`t.float().numpy()`) is different
  from bit-viewing (`from_numpy(fp32).view(torch.bfloat16)`) — the latter doubles the
  element count and produces a shape error far from its cause.

## Measurement hygiene (rules, not suggestions)

1. **Never trust a bandwidth number above ~240 GB/s** (`devices/strix-halo-memory-bandwidth.md`
   is the measured ceiling). It has two innocent explanations and one bug: (a) the working
   set is small enough to stay resident in the iGPU cache — check by tiling it past any
   plausible cache and re-measuring, which is how `gdn_decode`'s 918 GB/s at 18 MiB became a
   believable 255 GB/s at 108 MiB; (b) the clock boosted; (c) **the kernel skipped work**,
   which is what a `--rows` sweep of `lm_head_gemv` was doing at 710 GB/s. Absent (a) or (b)
   demonstrated, treat it as (c).
2. **Correctness-check every config, not just the default.** `run.py` compares each config's
   full logits against the reference (`ndiff`, `max|Δ|`). A fast wrong kernel is the most
   expensive kind of progress.
3. **Warm the pages before timing.** First GPU pass over freshly allocated pages runs at
   ~35 GB/s instead of 240 (`devices/strix-halo-unified-memory.md`). One untimed launch,
   then best-of-N.
4. **Take the input from the reference implementation and prove it.** `run.py` runs a live
   HF prefill, hooks the final norm, and self-checks that the row reproduces the pinned
   `logits_last` before using it — see the dump-name trap in `results.md` and §10.5.
5. **Report against the probe, not in the abstract.** Every kernel result quotes GB/s *and*
   the percentage of the 237 GB/s reference read, so a slow kernel is visibly slow rather
   than superficially plausible.
6. **Expect a floor from the reference itself.** GDN's `o_pre_norm` cannot be reproduced
   better than meanrel 1.45e-3 from the pinned HF dumps no matter how exact the kernel is
   (§04.4 records why). Gate tight against a mirror of the spec block *and* loosely against
   the dump; a single threshold would have made either a false FAIL or a false PASS.
7. **Check inputs for NaNs before blaming the kernel** — `check.py` prints it, because a
   shared-memory overrun once produced NaNs in one launch configuration and silence in
   another.

## Known limitations of what exists so far

- `lm_head_gemv` handles exactly one shape (`H = 1024`, one row at a time, `k = 1` batch).
  Batched logits and any other hidden size are unimplemented, not "parameterized later".
- No `engine/` code consumes a kernel yet; the prototypes are standalone binaries plus
  Python gates.
- Timing is best-of-N on an otherwise idle desktop; there is no clock pinning, so absolute
  milliseconds move by a few percent between runs. Ratios against the probe are the stable
  number.
