**VERDICT — `fund`.** The chunked delta-rule scan clears the plan's kill threshold (~1.5× of `gdn_rec`) by
arithmetic, not hope: priced at the engine's own **worst measured** matmul rate (its hand GEMMs, 0.8–0.9 TFLOP/s)
the chunked form is **1.96× faster at C=64, 2.56× at C=32, 2.99× at C=16**, and the rate needed for 1.5×
(0.42–0.65 TFLOP/s) is 50–76 % of that hand rate and **9–13 % of the measured hipBLAS `M=64` wave-mix rate
(4.80 TFLOP/s)**. On the error side it needs **no band wider than D-11**: against `data/refs/` the chunked form
in F32 is *closer to the pinned dumps than today's sequential form is* (state `meanrel` 1.6e-7 vs 5.2e-7; at the
width the `o_pre_norm` pins actually carry, 58 elements of 10.2 M differ at all and none of them falls outside
the §10.1 cancellation band), and even a model of the device shape — every matmul operand rounded to BF16, F32
accumulator, which is what MFMA does — costs **per-stage `meanrel` ≤ 9.9e-3 and per-logits-row `1-cos` ≤ 3.2e-4**,
i.e. 3× and 6× inside D-11's admitted `meanrel ≤ 3e-2` / `1-cos ≤ 2e-3`. Token flips: **0 of 417 rows with
≥ 0.10 of top-1/top-2 room** at C=64 and for every BF16-operand model (1 of 417 at C=16/C=32, and that row lands
on HF's own pinned token — §2.3). No product decision is made here; §5 proposes numbers for the parent to price.

Plan: `progress/2026-10-14/plan-p9-chunked-scan-2026-10-14.md`. Nothing here touched `engine/gate.py`,
`engine/route_allowance.py`, `tools/band_history.py`, a pin, or `speccheck/specimpl.py`, and `--check` is green
(§6). **No device execution**: this spike is CPU-only per `AGENTS.md`; §6 hands the device command lines to the
main agent. Every number below is a CPU measurement or arithmetic on numbers already measured in `bench/` and
`devices/`.

Artifacts (committed; `tmp/` is gitignored and regenerable):

| file | what it is |
|---|---|
| `prototype/gdn_chunk/chunk_scan.py` | the derivation, written into the module docstring, plus the torch CPU reference (`chunked_scan`, `recurrent_scan`, `qk_normalized`), a `--selftest`, and seven injected bugs |
| `prototype/gdn_chunk/spike.py` | the four deliverables as subcommands: `stage` / `carry` / `tokens` / `cost` |
| `tmp/p9/*.out` | verbatim output of the §6 commands |

Four findings contradict something already written down; they are in §7, not absorbed into the narrative.

---

## 0. Apparatus, and the one thing that had to be true before any of this meant anything

```bash
.venv/bin/python prototype/bf16_ulp.py                          # selftest: 0/15 ordering failures, passed
.venv/bin/python prototype/gdn_chunk/chunk_scan.py --selftest    # 0 failures (§1.4)
```

**The reference is the dump, and the dump is already a chunked scan.** `tools/reference_dump.py` patches
`specimpl.gdn_chunk_scan` so prefill pins come from `torch_chunk_gated_delta_rule` (chunk 64) and decode pins
from `torch_recurrent_gated_delta_rule`. So `L**.gdn.o_pre_norm.npy` / `state_new.npy` **are** a chunked
delta-rule scan's output, produced by code that shares no line with mine. Chunk-vs-pin is therefore an external
check, not a form-vs-form proxy — which is exactly what was missing: `speccheck/gates/run_chunkcheck.py` (G3)
checks chunk-vs-recurrent plus a cos against the pins, i.e. the weak half, the comparison this plan forbids.

Pins used (18 GDN layers each; unprefixed files = prefill phase): `p8` T=8, `p64` T=46, `p300` T=277, plus the
free-run fixtures' token streams (`dec300f` 295-tok prompt / 129 pinned tokens, `wide520f` 500/25, `chunk64b`
120/31, `mt32f` 232/33, `p128` 128/9, `dec64f` 17/65) and all 49 `data/refs/corpus48` prompts with their pinned
continuations. `p300` carries the stage tables (T ≫ C, so every chunk size is multi-chunk); `p64` is in the logs
as the ragged-tail case (T=46 < C=64 → the §04.7 padding path).

**Pin widths are measured, not assumed** (`progress/bench/evidence/p9/probe_pinwidth.py`): `gdn.o_pre_norm` is stored float32 but
**every element is BF16-representable** — HF casts its output back to the activation dtype. `gdn.state_new` is
genuine F32; `gdn.gated_out` and `out_proj.out` are BF16 dtype. Consequence, and it is the most load-bearing
fact in this document: an F32 candidate compared to the `o_pre_norm` pin *in F32* scores the pin's own
quantization (`meanrel` ~1.4e-3 — identical for the sequential and for every chunked form), which is not
candidate error. Per-element comparisons at that stage are therefore done **in BF16 ULPs**; `meanrel` against
that pin is quoted only with that reading attached.

---

## 1. Deliverable 1 — the chunked form, derived, and validated against the dumps

### 1.1 The algebra (full derivation is in `chunk_scan.py`'s docstring; this is the summary)

§04.4's step, per head, with `S ∈ R^{d_k×d_v}`, `k_t` L2-normalised, `β_t` the write gate, `α_t = exp(g_t)` the
decay gate. Measured on the pins: `max g = -0.0000`, i.e. **some heads never decay** (`exp(g) = 1.0000/token`),
so no chunk may be treated as independent and the inter-chunk carry is load-bearing:

    S̃_t = α_t S_{t-1}                  v̂_t = β_t (v_t − S̃_tᵀ k_t)                S_t = S̃_t + k_t v̂_tᵀ
    o_t = q̃_tᵀ S_t                     Γ_t = Π_{j≤t} α_j  (within a chunk, reset to the incoming state)

Inside one chunk, unrolling the rank-1 updates makes the state linear in the chunk's incoming state `S_in`. With
`γ_i = Σ_{j≤i} log α_j`, `Γ_i = exp(γ_i)`, `M_ij = exp(γ_i − γ_j)` for `j ≤ i` (0 above the diagonal) and the
forward-decayed keys `k̃_i = k_i`, `k̂_i = k_i exp(γ_C − γ_i)`:

    intra_ij = (q_i·k_j) β_j M_ij                    (kept INCLUDING i=j, see below)
    ut_ij    = (β_i (k_i·k_j)) M_ij  for i>j, 1 on the diagonal      ← unit lower triangular
    dv       = ut⁻¹ (β ⊙ v) − ut⁻¹ (βΓ ⊙ k̂) S_in                     ← the carry read, folded in
    o        = (Γ ⊙ q) S_in + intra · dv                              (the diagonal KEPT)
    S_C      = Γ_C S_in + k̂ᵀ dv                                       (the only thing chunk i+1 needs)

`ut⁻¹` is applied by forward substitution, blocked (each squaring doubles the exact prefix), batched over
(chunk, head). The `ut` matrix is unit-triangular by construction so its inverse needs no pivot and its diagonal
is exactly 1 — the reason this system is stable at C=16…64 with no condition-number story.

**The diagonal of `intra` must be kept.** §04.4 reads the state *after* the rank-1 write, so a token reads its own
update; §04.7's chunked form reads the decayed pre-write state and adds the diagonal term `(q_i·k_i) β_i M_ii dv_i`
separately. Dropping the diagonal is not a rounding choice, it drops the `v_i` term of the delta rule entirely.
Keeping it makes the two forms the same quantity up to the association of one sum, which is what §7.1 measures.

Three further clauses, each pinned by a bug in `--selftest` (§1.4): the carry read must be the **decayed**
pre-write state (reading before/after the decay costs 2.6e-2/6.2e-1 `meanrel`); the pairwise decay `M_ij` is not
the chunk-total decay `Γ`; and the ragged tail pads with `β = 0, g = 0` (§04.7, A-08), which must not move any
real row — asserted against the sequential form for C ∈ {1,2,3,5,16,32,64}.

Chunk sizes are not interchangeable with the reference (whose chunk is 64): C=16/32 are engine choices, so all
three are validated against the pins rather than assumed equivalent.

### 1.2 Why a device-shaped variant is priced here too

The matrix units take 16-bit operands with an F32 accumulator, so `mma=True` rounds every matmul **operand** to
BF16 and keeps F32 accumulation: `dv = bf16(ut⁻¹)·bf16(β⊙v) − …`, `S_C` decay with `bf16(k̂)ᵀ bf16(dv)`, the
pairwise decay factors themselves rounded at their use site. It is a model of a kernel's operand width, **not** a
kernel: no tile schedule, no MFMA instruction shape, no shared-memory staging, and it is pessimistic in one
known way (a real kernel can keep `ut⁻¹` in F32, and FP16 is measured at the same rate with a shorter mantissa —
`devices/strix-halo-compute.md`). It is included because the band question is about **operand width**, not about
chunking: §2 shows the chunking itself is free and the operands are what cost.

### 1.3 Validation against the dumps

`spike.py stage --fixtures p64,p300 --hist --negatives`; worst of the 18 GDN layers; `band` = 3e-5 of the tensor
max (§10.1's cancellation band, sized by `prototype/mlp_gemv/floor_probe.py`); ULP columns compare at the width
the pin carries. Full text in `progress/bench/evidence/p9/stage.out`.

vs **PINS** (p300, T=277):

| form | `o_pre_norm` cos / meanrel | `state_new` cos / meanrel | `gated_out` meanrel | `out_proj_out` cos / meanrel |
|---|---|---|---|---|
| sequential §04.4 (today's engine) | 0.99999846 / 1.42e-3 | 1.00000000 / 5.23e-7 | 2.30e-6 | 0.99999995 / 1.94e-5 |
| chunked C=16, F32 operands | 0.99999846 / 1.42e-3 | 1.00000000 / 2.57e-7 | 2.20e-6 | 0.99999997 / 1.57e-5 |
| chunked C=32, F32 operands | 0.99999846 / 1.42e-3 | 1.00000000 / 2.08e-7 | 8.83e-7 | 0.99999997 / 1.31e-5 |
| chunked C=64, F32 operands | 0.99999846 / 1.42e-3 | 1.00000000 / 1.60e-7 | 2.01e-6 | 0.99999998 / 1.10e-5 |
| chunked C=16, BF16 operands | 0.99999548 / 3.21e-3 | 0.99999648 / 2.93e-3 | 3.99e-3 | 0.99997443 / 6.70e-3 |
| chunked C=32, BF16 operands | 0.99999589 / 3.37e-3 | 0.99999660 / 3.08e-3 | 4.30e-3 | 0.99996921 / 7.40e-3 |
| chunked C=64, BF16 operands | 0.99999437 / 3.97e-3 | 0.99999647 / 3.06e-3 | 4.86e-3 | 0.99995679 / 8.81e-3 |

Two readings of that table, both important:

* **The `o_pre_norm` column is dominated by the pin's own BF16 width.** `meanrel` 1.42e-3 is what an *exactly
  correct* F32 answer scores against that pin (verified per layer: rounding my F32 result to BF16 and comparing
  again drops it to 3.8e-10 … 6.7e-9, with 3–104 differing elements out of 94 k–567 k and `maxULP` 1–6). So no
  form can be graded on `o_pre_norm` `meanrel` in F32; the honest instrument there is the ULP-with-band column.
* **The chunked form is closer to the dumps than the sequential form**, on every stage that is graded in true F32
  (`state_new` 1.60e-7 vs 5.23e-7 — 3.3× closer at C=64; `out_proj_out` 1.10e-5 vs 1.94e-5 — 1.8× closer). That
  is not a coincidence: the dumps are chunked output. A route change to a chunked prefill therefore moves the
  engine *towards* its reference, which reframes the band question — the band is not a licence to be worse, it is
  permission to stop paying a 1.4e-3/5e-7 gap that exists today (§7.1).

The `mma` rows are the ones a real device kernel will be compared on. Their worst stage number is
`out_proj_out` `meanrel` 8.81e-3 (C=64) — 3.4× inside D-11's `3e-2`, with `cos` 0.99995679 against D-11's
stage `cos ≥ 0.9998`.

### 1.4 The injected bugs, and where each one is visible

A negative that catches everything proves nothing (`prototype/mlp_gemv/results.md`), so the bugs are required to
appear **where they live**. On p300 (5 chunks of C=64), per-chunk `max|Δo|` against the correct chunked scan:

| injected bug | per-chunk max&#124;Δo&#124; (chunks 0…4) | state meanrel |
|---|---|---|
| `no_inter_chunk_read` (chunk i starts from zero) | 0.0e+00 1.9e-01 4.0e-01 5.9e-01 7.0e-01 | 2.08e+00 |
| `causality_upper` (`ut` built upper) | non-finite — caught outright | 2.79e+02 |
| `no_pairwise_decay` (`M_ij` → 1) | 5.3e-02 3.9e-02 3.3e-02 3.4e-02 1.6e-02 | 6.56e-02 |
| `no_chunk_decay` (`Γ_C` → 1) | 0.0e+00 0.0e+00 2.8e-02 3.2e-02 3.8e-02 | 5.37e-01 |
| `drop_last_chunk` (ragged tail) | 0.0e+00 0.0e+00 0.0e+00 0.0e+00 3.3e-01 | 5.59e-01 |
| `read_before_decay` (sequential form) | — (state meanrel 4.37e-02) | |
| `read_after_write` (sequential form) | — (state meanrel 6.45e-01) | |

Each is invisible where the bug is absent and loud where it is present: the `no_chunk_decay` row is exactly zero
for the first two chunks (a zero `S_in` makes the chunk-total decay unobservable) and the `drop_last_chunk` row
is zero until the last. `--selftest` additionally asserts the *legal* chunk-vs-sequential delta as a floor — it
measures 5.44e-7 `meanrel` on synthetic data and every injected bug must exceed 100× it, so the negatives are
calibrated by measurement rather than by taste. Note one boundary: at `T=46` with `C=64` there is exactly one
chunk, so a carry bug is **unobservable**; `spike.py` says so instead of printing a vacuous pass.

---

## 2. Deliverable 2 — the error report, in §10.1's shape

### 2.1 Per-stage aggregates

Above (§1.3), for four stages × seven forms × three sides (vs pins / vs sequential / vs the pinned `o` pushed
through this file's §04.5 code). The third side is the control: it matches the pinned `gated_out`/`out_proj_out`
to `meanrel` 1e-7 or better, so every `gated`/`out` number in §1.3 belongs to the scan and nowhere else.

### 2.2 BF16 ULP histogram with the cancellation band, and how much the band excuses

p300, all 18 layers pooled into one array per stage (`n` = elements), band = rel-magnitude < 3e-5 of the tensor
max, `ULP_MAX` = 1. Cells are `total[/off-band]`, the band column printed only for buckets that can offend.

```
seq       o_pre_norm  n=10211328 band  7.3%  >1ULP     216  off-band      0  worst off-band 0 ULP
          [0-0] 10207734 [1-1] 3378 [2-4] 133/0 [5-16] 59/0 [17-128] 17/0 [129-4096] 5/0 [4097-max] 2/0
seq       state_new   n= 4718592 band 14.2%  >1ULP      75  off-band      0  worst off-band 0 ULP
chunk64   o_pre_norm  n=10211328 band  7.3%  >1ULP      58  off-band      0  worst off-band 0 ULP
          [0-0] 10210324 [1-1] 946 [2-4] 38/0 [5-16] 12/0 [17-128] 7/0 [129-4096] 0/0 [4097-max] 1/0
chunk64   state_new   n= 4718592 band 14.2%  >1ULP      20  off-band      0  worst off-band 0 ULP
chunk64mma o_pre_norm n=10211328 band  7.3%  >1ULP 2014793  off-band 1577797  worst off-band 29185 ULP
          [0-0] 4706130 [1-1] 3490405 [2-4] 1346099/1171352 [5-16] 485330/338942 [17-128] 159028/65154
          [129-4096] 16095/1828 [4097-max] 8241/521
chunk64mma state_new   n= 4718592 band 14.2%  >1ULP  776138  off-band  515117  worst off-band 29791 ULP
```

What that says, in order:

* **F32 chunked, against the dumps: 58 elements of 10.2 M differ by more than 1 ULP, and every one of them is a
  cancelled element** (off-band count 0, i.e. the band excuses all of them; worst offender sits at 2.3e-9 of the
  tensor max). Same for `state_new`: 20 elements, none off-band. This is as close to a bit-exact result as a
  re-associated scan can produce, and it is *better* than today's sequential form on the same instrument
  (216 elements, again none off-band).
* **BF16-operand chunked: the max-ULP instrument stops working, and the band is what fails, not the kernel.**
  2.01 M of 10.2 M elements move by more than 1 ULP and 1.58 M of them are *outside* the band (15.5 % of the
  tensor); the worst off-band element is 29,185 ULP away, and the largest-magnitude offending element sits at
  6.6e-1 of the tensor max. That is not a bug — it is what rounding a 128-term dot's operands does when the
  result cancels: `meanrel` on the same tensors is 3.97e-3 and `cos` 0.99999437 (§1.3). This is exactly the trap `AGENTS.md` records as "max-ULP bands are blind to whole-tensor
  1-ULP error, and ndiff is blind to chains", arriving from the opposite direction: a per-element ULP gate at
  this stage would reject a form whose aggregate error is 8× under D-11. **A chunked-GDN gate must be aggregate
  (`meanrel`/`cos`) plus decision-level, not per-element ULP**, and that is a statement about the instrument, not
  a request for a looser number. The 3e-5 band is still quoted, with the number it excuses printed next to it
  (15.5 % of elements), so the band stays falsifiable.
* `bf16_ulp.py` is used for every comparison here (`ulp_dist`, `band_usage`), after running its own selftest; no
  hand-rolled `uint16` comparator appears anywhere (the dead-branch trap in `AGENTS.md` is why).

### 2.3 Decision-margin-bucketed token agreement (the A2 shape)

Two runs, both with **replay**: the chunked side is fed exactly the tokens the sequential side generated, so
every row is comparable and a difference cannot be blamed on a diverged token stream. Row 0 is the last prefill
row (it decides the first generated token); row *i* is the logits after feeding token *i−1*. `room` is the
**smaller** of the two sides' top1−top2 gaps, as in `progress/2026-10-14/route-c-audit-2026-10-14.md` §A2.

Long prompts (`dec300f, wide520f, chunk64b, mt32f, p128, dec64f`, 16 steps, 95 rows per form; the sequential
side reproduces 75/89 of the pinned continuation tokens on the same anchor):

| form | flips | row `1-cos` min / mean / max | pinned continuation | rows ≥ 0.10 room, flipped |
|---|---|---|---|---|
| chunked C=16 F32 | 1 / 95 | 2.23e-5 / 5.95e-5 / 2.62e-4 | **76/89** | 1 of 93 |
| chunked C=32 F32 | 1 / 95 | 2.15e-5 / 5.70e-5 / 1.74e-4 | **76/89** | 1 of 94 |
| chunked C=64 F32 | 1 / 95 | 2.04e-5 / 5.77e-5 / 2.06e-4 | **76/89** | 0 of 92 |
| chunked C=16 BF16 ops | 0 / 95 | 2.51e-5 / 6.34e-5 / 3.15e-4 | 75/89 | 0 of 92 |
| chunked C=32 BF16 ops | 1 / 95 | 2.38e-5 / 6.15e-5 / 1.75e-4 | **76/89** | 0 of 92 |
| chunked C=64 BF16 ops | 0 / 95 | 2.12e-5 / 6.46e-5 / 2.42e-4 | 75/89 | 0 of 93 |

Corpus (`corpus48`, all 49 prompts, 6 steps, 343 rows per form; sequential side reproduces 294/294):

| form | flips | row `1-cos` min / mean / max | pinned continuation | rows ≥ 0.10 room, flipped |
|---|---|---|---|---|
| chunked C=16 F32 | 1 / 343 | 1.56e-5 / 5.56e-5 / 3.12e-4 | 293/294 | 0 of 324 |
| chunked C=32 F32 | 1 / 343 | 1.56e-5 / 5.56e-5 / 3.12e-4 | 293/294 | 0 of 324 |
| chunked C=64 F32 | 1 / 343 | 1.56e-5 / 5.56e-5 / 3.12e-4 | 293/294 | 0 of 324 |
| chunked C=64 BF16 ops | 4 / 343 | 1.84e-5 / 6.60e-5 / 4.17e-4 | 291/294 | 0 of 324 |

Every flip, with the pinned HF continuation as the adjudicator — because "a decision changed" and "a decision
changed towards the reference" are different findings, and D-11's flip clause cannot tell them apart:

* `chunk64b` row 1: sequential `20619` → chunked `8337`; **pinned token 8337**. At C=16/C=32 this row reads as a
  flip at room 0.125 (outside the room D-11 admits); at C=64 the room is 0.062 (inside). The chunked side is the
  one that agrees with HF.
* `corpus48[32, 5010]` row 3: sequential `303` → chunked `854`; pinned `303`, room **0.000** (an exact top-1/top-2
  tie). The BF16-operand model adds three more, all at room 0.000.
* Every remaining flip in both runs sits in the `room < 0.02` bucket. **No flip was ever observed at
  room ≥ 0.10 except the single `chunk64b` row above.**

Aggregate over both runs: **438 decision rows per chunk size** (95 long-prompt + 343 corpus), of which ~417 have
≥ 0.10 of room. F32 forms: 2 flips each; BF16-operand C=64: 4; BF16-operand C=16: 0. Rows with ≥ 0.10 room that
flipped: **0 for C=64 and for every BF16-operand model; 1 for C=16 and 1 for C=32 — the same `chunk64b` row,read
differently by each side's gap.** The comparison the A2 audit set as the shape to hit ("0 of 1,566 rows with
≥ 0.10 room flipped") is met at C=64 and missed by exactly one row at C=16/C=32 — on a row where the chunked side
is the one that matches HF's pinned token.

### 2.4 The initial-state and end-condition clauses (§04.7), against the pins

`spike.py carry --fixtures p64,p300` — cut each fixture's pinned operands at token counts that are **not**
multiples of any chunk size in use (92, 100, 177, …), run segment 1 from the zero state, feed its end state into
segment 2, and compare the concatenation and the final state against the pins:

```
carry: 45/45 splits PASS
  o_cos worst 1.000e+00   s_cos worst 1.000e+00   o_meanrel worst 1.453e-03   s_meanrel worst 2.483e-07
  a zero-initialised second segment would score state meanrel 6.013e-01 — the carry is doing real work, not
  being excused
```

The last line is the non-vacuity control: the same comparison with the carry dropped scores 6e-1, so the PASS is
not the band excusing a wrong state. The `o_meanrel` 1.453e-3 is the pin-width floor of §0, not carry error
(`s_meanrel` 2.5e-7 is the honest number). This matters for the engine concretely: prefill already runs in
`wave_rows` bodies and requests resume mid-sequence, so a chunked kernel must carry a state in and hand one out
at an arbitrary token count — that path is now checked against the dumps, not assumed.

---

## 3. Deliverable 3 — cost model at the engine's real shapes, priced only at measured rates

Shapes: 512-row bodies × 2 = a 1024-token wave, H=16 v-heads, `d_k = d_v = 128`, 18 GDN layers. Baseline is the
measured `gdn_rec`: **125.4 ms, 18,432 launches, 6.8 µs each** (`bench/performance-status.md`, route-(c) table).

Sequential §04.4 arithmetic per token-head-layer: decay `D²` + read `2D²` + delta `3D` + write `2D²` + read
`2D²` = **115,072 flops** → **33.94 GFLOP per wave**, which against the measured 125.4 ms is **0.271 TFLOP/s
effective**. Two cross-checks on why that is neither a flops story nor a DRAM story:

* implied state traffic `2·D²·4 B` per token-head = **38.65 GB per wave**; at the measured 240 GB/s DRAM ceiling
  that is 161 ms — *more* than the 125.4 ms measured, so the state is being served above DRAM rate (1 MiB per
  layer against the 2 MiB L2 `hipGetDeviceProperties` reports) and the op is per-token-latency-shaped.
* pure dispatch is 18,432 × 1.8 µs (the measured eager-launch floor) = **33 ms of the 125.4 ms** — a quarter of
  the line is launch overhead, which the chunked form removes whatever its arithmetic does.

Chunked, per chunk-head: intra `10C²D + ⅔C³` (UT system, intra attention, the inverse applied to two `[C,128]`
right-hand sides, intra·dv, blocked inversion) + inter `6CD² + D²` (the serial carry: `k̂S`, `q̃S`, `k̂ᵀdv`, state
decay). Per token-head-layer, and the resulting wave:

| C | flops/token-head-layer | vs sequential | GFLOP/wave | serial depth per body | state traffic/wave | launch sites (FLA-shaped) |
|---|---|---|---|---|---|---|
| 16 | 119,978 | **+4.3 %** | 35.38 | 32 chunk-steps (was 512 token-steps) | 2,416 MB | 144 (0.3 ms dispatch) |
| 32 | 140,458 | **+22.1 %** | 41.42 | 16 | 1,208 MB | 144 |
| 64 | 183,210 | **+59.2 %** | 54.03 | 8 | 604 MB | 144 |

Priced at measured rates only (`devices/strix-halo-compute.md`'s two corrections honoured: **no F32-C rate is
used anywhere** — 16-bit C is what the library numbers mean), including the 0.3 ms dispatch:

| rate TF/s (what it is) | C=16 | C=32 | C=64 |
|---|---|---|---|
| 0.271 — today's `gdn_rec` effective rate | 130.8 ms (**0.96×**, kill) | 153.1 (**0.82×**) | 199.6 (**0.63×**) |
| **0.85 — the engine's own hand GEMMs** | **41.9 (2.99×)** | **49.0 (2.56×)** | **63.8 (1.96×)** |
| 4.80 — hipBLAS wave mix at `M=64` | 7.6 (16.4×) | 8.9 (14.1×) | 11.5 (10.9×) |
| 18.84 — hipBLAS wave mix at `M=512` | 2.1 (58.7×) | 2.5 (51.0×) | 3.1 (40.1×) |
| 33.0 — hipBLAS at `gate_proj`'s own shape | 1.3 | 1.5 | 1.9 |

and the requirement in rate terms:

| | 1.0× | **1.5× (the kill line)** | 2.0× |
|---|---|---|---|
| C=16 needs | 0.282 TF/s (33 % of the hand rate, 6 % of the M=64 mix) | 0.423 (50 %, 9 %) | 0.564 (66 %, 12 %) |
| C=32 needs | 0.330 (39 %, 7 %) | 0.495 (58 %, 10 %) | 0.661 (78 %, 14 %) |
| C=64 needs | 0.431 (51 %, 9 %) | 0.646 (76 %, 13 %) | 0.862 (101 %, 18 %) |

**Reading.** The 1.5× bar is cleared as soon as the chunked kernel is *half as good as the engine's worst
existing GEMM*, and by a wide margin as soon as it reaches the library route the tree now has. It is not cleared
if the chunked form is written on the same per-token vector path as today's scan (0.271 TF/s row: 0.63–0.96×,
i.e. **the win comes from changing the shape of the work, not from the chunk algebra**).

**Can the now-landed hipBLAS route absorb the matmul growth cheaply?** Yes, and the growth is not the interesting
term. Intra-chunk work is a batched small-K product (`C×128 by 128×C`, batch = H × chunks = 128…2,304), which is
where the library route is weakest relative to itself — measured `M=64` wave-mix 4.80 TF/s versus 18.84 at
`M=512`, a 3.9× shape penalty, and `W9`'s `M=64` body regression is the same lesson from the engine side. Even
so, priced at that worst-case 4.80 TF/s the whole chunked wave is 7.6–11.5 ms against the 125.4 ms it replaces.
The three things that actually decide the outcome are: (a) the flop growth is only +4 % at C=16 and +22 % at C=32,
so a library call per chunk-step does not have to be efficient to win; (b) **launch count falls from 18,432 to
~144–1,152**, which is worth 33 ms on its own; (c) the serial carry shrinks from 512 steps to 8–32, but at C=64
the parallelism left in a step is one head-tile per step per (layer, body) — 8 × 2 × 16 = 256 tile-steps against
20 CUs, ~13 waves, which is ample. C=16 is the safe middle: +4 % flops, 32-step carry, 2,416 MB of state traffic
still 16× below today's.

Two honest caveats on the model. It prices flops, not a schedule: no tiling, no shared-memory staging, no
register-pressure model, and the state-traffic row is an upper bound (some of it is L2-resident today, which is
why the measured time is under the DRAM floor). And it assumes a kernel shaped like FLA's chunked scan (four
sites per layer-body). A phase-1 kernel should re-price this table with `--time-waves` rather than argue with it
(§6).

---

## 4. Deliverable 4 — `attn`, sized in passing

**One number: `attn` runs at 0.236 TFLOP/s effective**, which is arithmetic-shaped work running ~80× below the
measured matrix rate and ~3.6× below the engine's own hand GEMMs — so attention is **neither** arithmetic-bound
in any measurable sense **nor** the same serial-scan shape as GDN.

Arithmetic: 6 attention layers × 2 bodies of 512 rows, causal triangle, `N_Q=8` heads, `d_head=256`, scores +
PV = 4 MACs per (dim, lane, head): body 0 contributes `Σ_{t=1..512} t` lanes and body 1 `Σ_t (512+t)`, so
**25.79 GFLOP per 1024-token wave** over the family's 12 recorded sites (note `engine/launch_stats.h`: the attn
family counts *sites*, one per layer per body; a site submits 1–3 kernels by lane depth, so the real per-kernel
number is 3–9 ms, not 9.1). 25.79 GFLOP / 109.2 ms = **0.236 TFLOP/s**. Against the measured menu: 30.3 ms at
the hand-GEMM rate, 5.4 ms at the `M=64` mix, 1.4 ms at `M=512`. The triangle assumption is stated because it
matters by exactly 2×: if the kernel scores the full rectangle instead of the causal triangle the rate is
0.47 TFLOP/s and the conclusions are unchanged.

Is it a serial scan in disguise? No. Attention has no cross-row carry — rows are independent given the KV, the
causal mask only shortens the work of early rows — and its 12 sites at 9.1 ms are nothing like dispatch-bound
(12 × 1.8 µs is 22 µs). So the GDN lever (chunked scan) does not transfer, and neither does its band question:
there is no reassociation of a long serial carry to price. What attention has in common with chunked GDN is the
*mechanism of the deficit*: matrix-shaped work executed on a vector path. Its own lever is tiling/MFMA
(flash-style blocking over the KV, which is also what makes a chunked GDN fast), and it is the same 20 CUs that
both levers compete for. It also scales the other way: attention's cost grows `O(T²)` in context (4,096-token
context makes this arithmetic 47.9× larger) while GDN is linear, so the relative shares shift towards attention
with longer prompts even with `gdn_rec` fixed. One paragraph, one number, and the number says the next funded
attention item is a kernel-shape item, not a numerics item.

---

## 5. The band, **proposed and not adopted**

Reference point: **D-11** as admitted in `progress/2026-10-14/route-c-audit-2026-10-14.md` — per-stage `meanrel ≤ 3e-2`, per
stage `cos ≥ 0.9998`, per logits row `1-cos ≤ 2e-3`, and flips admitted only below 0.10 of top1−top2 room.

**Proposed band for a chunked GDN prefill (any C in {16, 32, 64}, BF16 matmul operands):**

| quantity | proposed | worst measured here | headroom vs D-11 |
|---|---|---|---|
| per-stage `meanrel` (`o`, `state`, `gated`, `out`) | ≤ 1.2e-2 | 9.87e-3 | 2.5× tighter than 3e-2 |
| per-stage `cos` | ≥ 0.9995 | 0.99994523 | tighter than 0.9998 |
| per logits row `1-cos` | ≤ 5e-4 | 3.15e-4 | 4× tighter than 2e-3 |
token flips at room ≥ 0.10 | 0 | 0 for C=64 and for every BF16-operand model; 1 row at C=16/C=32 whose chunked token matches the pin | same rule, no widening |
| pinned-continuation agreement | ≥ the sequential form's | 76/89 vs 75/89 (long), 293/294 vs 294/294 (corpus) | equal or better |

So: **no band wider than D-11 is needed** — the proposal above is strictly tighter than D-11 on every axis the
cascade band measures. That is the answer to the plan's third question, and it is the reason the verdict is
`fund` rather than `fund-with-a-wider-band-than-D-11`.

Two clauses that the parent has to decide, both about instruments rather than numbers:

1. **Per-element ULP cannot gate this stage.** §2.2: with BF16 operands, 15.5 % of `o_pre_norm` elements sit
   beyond 1 ULP outside the 3e-5 band while `meanrel` is 8× inside D-11. A chunked-GDN gate must be aggregate +
   decision-level. If the parent wants a per-element clause, the band it needs is not `3e-5` and must be sized by
   a floor probe at the chunked dot lengths (this repo's rule: bands are sized by a probe, not by taste).
2. **The flip clause needs an adjudicator.** D-11 counts a flip when the two implementations disagree at large
   room. Here the one ≥ 0.10 disagreement is the chunked form matching HF's pinned token while the current engine
   does not. Counted against the current engine it is a violation; counted against the reference it is a
   correction. Propose: a flip counts against the chunked form only if the **sequential** side matches the pinned
   continuation and the chunked side does not. Under that rule the count is 0 for C=64 and 0 for C=16/C=32 too
   (the corpus48 flip at room 0.000 is inside the admitted band either way).

Not requested, not assumed: nothing here adopts a band, edits a gate, touches `engine/route_allowance.py` /
`tools/band_history.py` / a pin / `speccheck/specimpl.py`, or claims a tokens/s figure.

---

## 6. What to run next, and by whom

CPU (either agent, cheap, already green):

```bash
.venv/bin/python prototype/bf16_ulp.py                        # selftest first (the ULP instrument)
.venv/bin/python prototype/gdn_chunk/chunk_scan.py --selftest
.venv/bin/python prototype/gdn_chunk/spike.py stage --fixtures p64,p300 --hist --negatives
.venv/bin/python prototype/gdn_chunk/spike.py carry --fixtures p64,p300
.venv/bin/python prototype/gdn_chunk/spike.py tokens --fixtures dec300f,wide520f,chunk64b,mt32f,p128,dec64f --steps 16 --mma
.venv/bin/python prototype/gdn_chunk/spike.py tokens --fixtures corpus: --corpus-n 49 --steps 6
.venv/bin/python prototype/gdn_chunk/spike.py cost
.venv/bin/python tools/band_history.py --check                # green; unchanged by this spike
```

Device (**main agent only**, zero subagents, blocking, per `AGENTS.md`; none of this was run here, and no number
from this spike may enter a timing doc):

```bash
tools/gpu_lock.sh acquire "p9 phase1 gdn_rec baseline"; tools/gpu_lock.sh status
# the baseline the model above is priced against, re-taken with the current tree (same invocation as progress/bench/evidence/w9.sh):
timeout 1200 ./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --time-waves
timeout 1200 ./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --time-waves --gemm library
tools/gpu_lock.sh release "p9 phase1 gdn_rec baseline"
```

Record `tools/gpu_lock.sh status` (util) at acquire, run blocking with zero subagents, and treat the resulting
`gdn_rec` share as the number the §3 table must be re-priced against — those `ms` figures belong in
`bench/performance-status.md`, which is main-agent-only, and not in this document.

Phase 1, gated on the parent funding this: the HIP chunked kernel for `engine/kernels_gdn.cpp` **prefill only**
(decode stays on the recurrent kernel — §2.4 shows the handoff is exact), the four-site launch plan, and a gate
that compares against `data/refs` prefill GDN pins in the §5 shape. Out of scope until then, per the plan: any
claim about tokens/s, and any band adoption.

---

## 7. Four things this spike found that contradict what is written down

1. **§04.4's note mis-attributes the 1.45e-3 gap.** `specification/04-gdn-linear-attention.md` says our recurrent
   scan "agrees with `state_new` to 2e-8 but with `o_pre_norm` only to 1.45e-3 — HF's fused read accumulates
   differently". Measured here: `o_pre_norm` pins are **BF16-valued**, and an exactly-correct F32 result scores
   ~1.4e-3 against them for that reason alone (rounding the candidate to BF16 first drops it to 4e-7 … 7e-10).
   The remaining sequential-vs-chunk difference is real but two orders smaller than the note implies. Worth a
   correction in §04.4 so the next reader does not chase a fused-read difference that is mostly the dump's width.
2. **The dumps are the chunked form, so chunking moves the engine closer to its reference.** §1.3: `state_new`
   1.60e-7 (chunked) vs 5.23e-7 (sequential) against the pins, `out_proj_out` 1.10e-5 vs 1.94e-5, and pinned
   continuation 76/89 vs 75/89. The band question is therefore not "how much worse may it be"; on these
   fixtures it is better, and the band is what the *device kernel* will need, not what the algebra needs.
3. **G3 is weaker than it looks.** `speccheck/gates/run_chunkcheck.py` compares the chunked scan against the
   recurrent one and then takes a `cos` against the pins. The first half is self-consistency between two
   implementations of the same equation, which is what this plan explicitly forbids as the primary evidence, and
   the `cos` half cannot see a whole-tensor 1-ULP shift (§2.2). A chunked-prefill gate needs the §5 shape
   (`meanrel` + `cos` + decision buckets), and `carry`'s non-multiple-of-C cut (§2.4) needs to exist at all —
   today no gate covers a mid-sequence state handoff.
4. **`o_pre_norm` is BF16-valued and `state_new` is F32.** Any future per-element check on `o_pre_norm` must
   compare in BF16 ULPs (as `route_c_audit.py` does), because the pin carries 8 bits of mantissa; comparing an
   F32 candidate to it elementwise measures the reference's rounding. `spike.py` records the width of each pin
   in its own header line for exactly this reason.
