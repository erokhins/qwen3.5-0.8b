# Route (c) audit: is the hand↔library delta reassociation, or a bug? (2026-10-14)

D-11 accepts the wave-width dependence **conditional** on this audit. The standard being met is not "the
aggregates look small" — cos 1.28e-04 and meanrel 1.5 % are consistent with both a benign reassociation and a
wrong-operand bug that happens to be in a small place. The standard is **element-level, per-step, and bounded
against a measured floor.**

## The four comparisons, in order of power

**A1 — Isolated per-shape reassociation floor (the reference bound).** For each of the ten library shapes at the
widths the engine actually uses (`M = 1, 88, 512, 1024`), run hand kernel vs `hipblasGemmEx` on the **same real
weight tensor and the same real activation**, and record: fraction of outputs differing, ULP histogram
(0 / 1 / 2 / 3 / >3), max ULP, and the *magnitude of the element where the max occurs*. This is the floor: it is
what a legal reassociation can do at that shape with no network around it. Everything downstream is judged
against it. A library-vs-hand delta inside a network that **exceeds** its own isolated floor is a bug — it cannot
be explained by the route.

**A2 — Per-step element diff on logits, all fixtures.** `--prefill-logits all` under both paths, then compare
step by step with `prototype/bf16_ulp.py`'s selftested comparator (**never** a hand-rolled `uint16` compare: the
sign trap in `AGENTS.md` makes a naive one silently pass). Report per step: `% differing`, ULP histogram, max ULP,
worst element's index and its value, and whether the worst elements sit at **near-cancellation** (|value| ≪ tensor
max) — the rounding signature — or at **large magnitude** — the bug signature. Expected pattern, to be
pre-registered before the run: small bounded delta at the last prefill row, near-zero at the first generation
steps (whose `lm_head` is hand on both sides, so they differ only by carried state), widening with chaos thereafter.

**A3 — Per-stage element diff at one fixed step.** The engine already dumps intermediates (`STAGEDUMP` /
`stagedump` path). Diff every stage tensor hand-vs-library: norms, rope'd q/k, `o_proj` out, gate/up/act/down, GDN
gates and state, `block_out`. Rounding appears as **small deltas at every matmul stage, accumulating**; a bug
appears as **one stage moving far more than its A1 floor** with clean stages around it. This is the comparison that
found the stale `block_out` operand once already, and it is the one an aggregate cannot make.

**A4 — Negative control, required.** Inject at least two wrong-but-plausible bugs — a row-offset in one operand,
a swapped weight pointer, a stale tile (the actual `block_out` bug, re-created) — and show the A1–A3 statistics
**catch** them by orders of magnitude. A comparator that cannot distinguish a reassociation from a row offset is
not a check. Record the separation: ratio between the injected-bug delta and the legal delta.

## Discriminators being used (stated so they can be argued with)

- **Locality in the tensor**: rounding is spread across elements whose values cancelled; bugs concentrate on whole
  rows, whole columns, or one contiguous slice.
- **Locality in the stage graph**: rounding touches every matmul stage a little; a bug is one stage, sharply.
- **Locality in width**: if a delta depends on `M` beyond what A1's per-width floor predicts, the width dependence
  is not the library's summation order but something reading past a body boundary — which would also be an
  out-of-bounds read candidate.
- **Growth across steps**: chaos-driven growth is smooth and starts where state drifts; a bug jumps at the step
  where the wrong value is first consumed.

## Bands (pre-registered, and they are ceilings not targets)

- A1: report, do not band — it *is* the bound.
- A2 last-prefill-row: **no element beyond its own shape's A1 max ULP**, and `% > A1 max` at or below the A1
  fraction. Any element beyond is a stop-and-report.
- A3: every stage within **2× its A1 floor** (2× for accumulation through the block, chosen before the run and
  justified after, not tuned to the result).
- A4: injected bugs must exceed the legal delta by **≥10×**; if they don't, the audit is too weak and gets
  strengthened rather than the bugs being dropped.

## What changes if it passes

`--gemm library` becomes the default, D-2 allowances get re-derived from the A1/A2 numbers, the cross-wave-width
gate is replaced by the band check, and provenance headers gain `wave_rows` and hipBLAS version.

## What changes if it fails

D-11 is void, the route stays off, and A3's worst stage becomes the bug report. A 2.8× win that cannot be proved
benign is not a win; it is an unbounded error with good marketing.
