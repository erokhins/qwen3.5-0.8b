# P9 — chunked GDN delta-rule scan: spike and band pre-registration (2026-10-14)

**Funded by measurement, not narrative.** With route (c) landed, matmul is 21.2 % of a 1024-token prefill wave
and `gdn_rec` is **38.2 %** (125.4 ms, 18,432 launches, 6.8 µs each) — past the 40 %-non-matmul threshold this
repo pre-registered for funding P9. See `bench/performance-status.md` §"Wave-path attribution" and §"Headline
tables after the route (c) flip".

**What P9 is.** The gated delta-rule recurrence is serial along the sequence: state `S_t` depends on `S_{t-1}`
through a rank-1 update with a decay gate, so today `gdn_rec` is one tiny launch per step per head — which is why
it is 18k launches of small work rather than a few large ones. A **chunked scan** rewrites that as intra-chunk
work that is associative (matmul-shaped) plus a small serial carry across chunks, the standard chunked-scan trick.
It is the only known route to making the recurrence cheap; it is **not bit-identical** to the sequential form, so
it enters as a band question.

## Phase 0 — what this spike must decide, before any HIP is written

All CPU-only, all in `.venv` torch (CPU-only build, which is fine here), against the pinned dumps in `data/refs/`:

1. **Is the chunked form actually what the engine needs?** Derive it on paper and implement it as a torch
   reference. Then compare chunked-vs-sequential on real weights for the fixture prompts and report the error
   **the way §10.1 does**: per-stage `meanrel` and `cos`, a ULP histogram with the cancellation band, and the
   decision-margin-bucketed token agreement — not a single max-ULP number, which §10.1 (F1–F4) proves is
   meaningless for a long dot.
2. **Cost model, from arithmetic, not hope.** Chunked scan trades serial steps for `O(T·C + T²/C)` intra-chunk work
   at chunk size `C`. At the engine's real shapes (`head_dim = 128`, 16 v-heads, `K = 2048` in, 512-row bodies) and
   the measured rates in `devices/strix-halo-compute.md` (~33 TFLOP/s matrix, ~2.9 TFLOP/s vector), state the
   speedup the chunked form could buy at `C = 16/32/64` and whether matmul growth (it becomes matrix work, which
   the library route can already run fast) eats it. **If the model says the win is under ~1.5× of `gdn_rec`,
   say so and kill it here** — 125 ms of a 328 ms wave is not worth an unbounded band for a rounding error.
3. **Sized band, proposed not adopted.** Give the numbers for a band (per-stage `meanrel`, per-row `1-cos`, flip
   room) that the chunked form would need, and how far it sits from the cascade band D-11 just admitted
   (`meanrel ≤ 3e-2`, row `1-cos ≤ 2e-3`, flip room `< 0.10`). If it needs more than that, the decision is the
   user's, not the implementer's.
4. **The attention half, sized in passing.** `attn` is 33.3 % (109.2 ms, 12 launches at 9.1 ms each at
   `wave_rows=512`). Measure on CPU what that per-launch cost implies in FLOPs vs the measured rate — is
   attention arithmetic-bound, or is it the same serial-scan shape as GDN in disguise? One paragraph, one number.

## Hard rules for this spike

- **CPU only. No GPU.** `AGENTS.md` reserves all device execution for the main agent; this spike must not touch
  the iGPU, not even for a correctness check. Hand the parent the exact command lines for anything device-side.
- **No gate, band, pin, or `speccheck/specimpl.py` edits.** Bands are proposed in this file's findings and priced
  by the parent. `tools/band_history.py --check` must stay green.
- **The dumps are the reference.** Never validate the chunked form against the sequential engine alone — that
  proves self-consistency, not correctness. Against `data/refs/`, with the existing gates' comparison code.
- Commit per piece; the write-up is `progress/2026-10-14/p9-chunked-scan-2026-10-14.md` with the verdict on line 1: **fund**,
  **kill**, or **fund-with-a-wider-band-than-D-11** (the last one is the parent's and the user's call).

## Out of scope until Phase 0 reports

Any HIP kernel, any change to `engine/kernels_gdn.cpp`, any band adoption, and any claim about tokens/s.
