# P7 to shipping: the accept loop, the stage family, and the bands priced in advance

Written 2026-10-22 by the main agent, **before any MTP timing exists**. Everything numeric here is either a
measurement already in the repo or a line drawn in advance; nothing in this file may be re-drawn after a number
arrives. Why: three routes in this repo were nearly funded on post-hoc arithmetic, and `AGENTS.md` is blunt that
a band widened after a refusal is not a band.

## Where P7 actually stands (all measured, none shipped)

| quantity | value | source |
|---|---|---|
| draft block cost `c_full` | 0.2237 of a step | `prototype/mtp_gatezero/` |
| draft block, measured | 0.4371 ms (NARROW), depth-linear (×1.75 k2, ×3.44 k4) | `prototype/mtp_draft/DRAFT-RESULT.md` |
| batched head | R(M=2)=1.0019, R(M=4)=0.0128 over | `prototype/mtp_headbatch/` |
| `delta` (head+block, plan A) | 0.1358 of a step | `prototype/mtp_delta/D3-RESULT.md` |
| accept rate on **engine bits** | `a = 615/926 = 0.6641` | `prototype/mtp_accept/ACCEPT-RESULT.md` |
| binding kill line | `a_needed = 0.4568` (plan A **+ rollback**) | same |
| verify body (2-row) | built, VERIFY GATE PASS 164/0, engine gate RED on undeclared census | `engine/mtp_verify.{h,cpp}` |
| projection | **1.348×** if plan A holds | arithmetic on the rows above |

Baseline for every ratio below, taken by the main agent in one lock hold (4,000 lanes, 32 steps, medians of 3):
burst **10.6689 ms/token = 93.7 tok/s**, stage TOTAL 11.6086 ms, `launches_per_step=224 plan_match=1`.
Cross-window drift is §6's ~2 %, so the pair is quoted together, never against the ledger row.

## What is missing, and it is two things

1. **The accept loop.** Nothing emits two tokens per step today. `--mtp-draft` prices a proposal; `--mtp-verify`
   probes a run; neither is a decoder.
2. **An MTP stage family.** `decode_run` *refuses* `--mtp-draft 1` with `--time-stages` because "the draft
   belongs to no family of §07's launch bill". That refusal is correct and must be satisfied, not bypassed:
   charging draft submissions to whatever segment preceded them would bill the wrong stage and print a
   plausible wrong tok/s. Same remedy that closed the draft-census red: **declare the launches in the plan**.

## The two open engineering decisions, named before code

**D-a — how a rejected draft is rolled back.** Per-step state is GDN 18.0 MiB + conv 0.63 MiB + 12 KiB/token KV.
Options, with the arithmetic that decides between them:
- *snapshot/restore*: copy 18.66 MiB per step = **~0.079 ms of pure bandwidth** at the 237 GB/s plateau, plus
  the same again to restore on reject. This is what `rho ≈ 0.12 ms/round` was priced from, and it is already
  inside the kill line's denominator — it is not a surprise, it is a budget.
- *recompute*: replay the accepted prefix through the recurrence. Costs a full prefill-body per rejected round
  (~9× more at any realistic accept rate) — dead on the same arithmetic that killed the pessimistic-delta shape.
Choice: **snapshot/restore, and measure it**, because if `rho` reads materially above 0.12 ms the accept
economics move and the kill line has to be re-derived *from the plan*, not from the observed win.

**D-b — where acceptance is decided.** The draft token must be compared against the verifier's argmax every
step. If that comparison happens on the host it costs a per-step sync, and the repo already knows the exposed
sync leg is ~1.98 ms of a step at 128 lanes. Decision: **argmax on device, comparison on device, one status
word read back**, which makes device-side argmax (queue P2, ~0.1 ms) a *dependency* of MTP rather than an
optimisation. If P2 cannot be made cheap, the fallback is a fused accept-and-sample epilogue in the lm_head
kernel, and that is a design change, not a tuning knob.

## Numerics bands — pre-registered, and the shape of the argument

Speculative acceptance is only legitimate if a rejected token cannot change what the model would otherwise
have emitted. That gives exactly three requirements, and they are judged against what already exists:

1. **Losslessness by construction, not by agreement.** A token is accepted iff the verifier's own argmax equals
   the draft's. No tolerance is involved in the accept decision itself.
2. **The verify body's row 0 must be the shipped single-row path unchanged**, so that "would the shipping path
   have emitted this token" is answered by using the shipping code, not by negotiating between two
   implementations. Bit-identity there must be a *consequence* of using the unchanged path. (Row 0's K/V
   contamination is already solved by the 24,576 B lane save/restore — that stands.)
3. **Everything else is judged by the bands that already exist** — `CASCADE_BAND` and the imported imported
   floors in `engine/gate.py` — because the standing numeric-format rule says a green gate set *is* the
   decision procedure, and no new band is needed unless a gate goes red. If one does: widen only if the
   geometry-derived must-catch set still turns red under every injection, and record the new band beside what
   witnesses it. **Never** widen to route around a red.

Known, and stated rather than discovered later: the two-row body's row 0 carries the form gap against the
one-row kernel (argmax SAME, sub-band magnitudes), and this engine's wide-vs-one-row divergence means
"bit-identical verify" is false as a matter of fact. It also now carries the named F16-D signed-zero
dependence (`devices/strix-halo-wmma-signed-zero.md`), which is ≤1 BF16 ULP on odd bounds and gated as a
budget in `prototype/mtp_verify/zero_gate.py`.

## Timing protocol, fixed in advance

Main agent only, blocking, zero subagents, one lock hold per arm, `tools/gpu_lock.sh` acquired and released
inside a single shell block, util recorded at both ends and then discounted.

- widths: **128** and **4,000** lanes (the headline width is the 4k one, per the standing decode rule),
  median of 3, burst idiom (`hipEventElapsedTime` reads exactly 0 for sub-ms kernels and lies for 2.1 ms ones).
- arms: flag-off baseline, flag-on k=1, flag-on k=2 — same fixture, same window, **behavioural canary asserted
  before any timing is taken** (the canary here is free: the `MTPDRAFT`/verify census lines and
  `plan_match=1`; a stale binary that prints no census is caught, because a census line is what changed).
- report: tok/s **and** mean accepted length and reject distribution. A tok/s number without the accept length
  is unauditable — it cannot be checked against `(1+a)/(1+c+rho)`.
- pre-registered win/kill, stated per width because a fixed per-token saving is a smaller fraction of a 4k step:
  **adopt at ≥ +10 % tok/s at 4,000 lanes with gates green**; **kill if measured `a` on the shipping path falls
  below `0.4568`**, which is the line already drawn and is not movable by the implementation.

## Deliberately out of scope for this package

MTP depth beyond k=2 (block cost is depth-linear and k=1 is where `a` was measured), multi-request batching
(a scope decision, not a kernel route), and any quantisation. Draft-layer numerics come from the pinned §09
reference and are not renegotiated here.
