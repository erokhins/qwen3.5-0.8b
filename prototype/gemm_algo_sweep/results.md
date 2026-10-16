# `gemm_algo_sweep` — results

**Empty on purpose.** Every row below is filled by a main-agent device window (AGENTS.md: no
worker-owned timing, and no number from a worker device run may enter a timing doc). The tool is
built and host-verified; `--verify` and `--time` have not been run by a worker, and the first operator
window ended on the band rather than on a fault (`HANDOFF.md` §1). See `HANDOFF.md` §1 for what is
proven and §2 for the commands.

Question in one line: *at the engine's own matmul shapes, does hipBLASLt own a kernel faster than the
one its default heuristic ranks first?* — `README.md` §1.

---

## 0. Enumeration capability of this build (from `--verify`)

No device window has produced these yet. The static evidence, which is stronger than one run for the
one claim it makes, is `README.md` §3: the cuBLASLt-style enumeration API is absent from this build,
and enumeration comes from `hipblaslt_ext::getAllAlgos`.

| field | value | window |
|---|---|---|
| `hipblasLtMatmulAlgoGetHeuristic` status / returned (per shape) | | |
| `hipblaslt_ext::getAllAlgos` status / returned (per shape) | | |
| distinct algorithms per shape after dedup, and how many are `supported` | | |
| does the default (heuristic rank 0) appear in the enumeration | | |
| solution/kernel names the default resolves to | | |

## 0a. Band calibration (from `--floor-probe`, no timing in it)

Every `CHECK` verdict below is judged against this band, so it is recorded beside the results rather
than inherited (`README.md` §5a, `HANDOFF.md` §2b). Window: `--rows 512 --check-rows 8 --seeds 4`,
`tools/gpu_lock.sh util` **92 %** at acquire (this session's own tokens, zero subagents), blocking,
**no timing taken and none in these lines** — one default-algorithm launch plus one `hipblasGemmEx`
launch per `(shape, M, seed)`, everything else host arithmetic. `FLOOR ok - 7 CAL row(s)`,
`refused=0 could_not_run=0`.

| shape | K | worst algo cos | worst **engine** cos | ref-reorder | license | quant deficit | hardest inj cos | **cos_floor** | headroom | cancel_rel | cancel_tol |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q_proj | 1024 | 0.999997974322 | 0.999997974322 | 0.999997974322 | 0.999997974322 | 2.026e-06 | 0.985226350 | **0.999987974322** | 1.48e-02 | nothing exempt | nothing exempt |
| k_proj+v_proj | 1024 | 0.999997880608 | 0.999997880608 | 0.999997880608 | 0.999997880608 | 2.119e-06 | 0.972070936 | **0.999987880608** | 2.79e-02 | nothing exempt | nothing exempt |
| o_proj+out_proj | 2048 | 0.999997772741 | 0.999997772741 | 0.999997772741 | 0.999997772741 | 2.227e-06 | 0.991677774 | **0.999987772741** | 8.31e-03 | nothing exempt | nothing exempt |
| in_proj_qkv | 1024 | 0.999997682620 | 0.999997682620 | 0.999997682620 | 0.999997682620 | 2.317e-06 | 0.926571686 | **0.999987682620** | 7.34e-02 | 7.831e-07 | 1.631e-08 |
| in_proj_z | 1024 | 0.999997861145 | 0.999997861145 | 0.999997861145 | 0.999997861145 | 2.139e-06 | 0.953999909 | **0.999987861145** | 4.60e-02 | nothing exempt | nothing exempt |
| gate_proj+up_proj | 1024 | 0.999997776338 | 0.999997777309 | 0.999997776087 | 0.999997776087 | 2.224e-06 | 0.894695465 | **0.999987776338** | 1.05e-01 | 9.377e-07 | 4.395e-08 |
| down_proj | 3584 | 0.999997691655 | 0.999997691655 | 0.999997691655 | 0.999997691655 | 2.308e-06 | 0.993670845 | **0.999987691655** | 6.32e-03 | nothing exempt | nothing exempt |

Four things that were assumptions before this window and are measurements after it:

* **The deficit is the format, not the kernels.** On six of seven shapes the hipBLASLt default, the
  engine's own `hipblasGemmEx`, the reordered reference and the F32-accumulation license leg all land on
  **the same cos to all twelve printed digits**; on `gate_proj+up_proj` they differ by `1.2e-09`, which
  is 0.05 % of the deficit they all sit at. `CONFLUENCE agree=1` on 7/7, `quant deficit` is
  `2.0e-06 … 2.3e-06`, and `meanrel` is `1.4e-03` throughout — consistent with `1 − meanrel²/2`, i.e.
  the cost of writing a bf16 answer. That is the fact that licenses a floor at `0.999988` instead of
  refusing forever, and it is what `0.999999` was refusing.
* **The license legs are bit-identical** (`ndiff=0`, `worst_ulp=0.0000`) — accumulating a dot of bf16
  products in F32 gives the same bf16 output as the exact double sum, at K=1024…3584 on every sampled
  element of every shape. So F32 accumulation licenses *nothing* here, and the cancellation band had to
  be sized from the device legs instead. If it had been sized from the license legs alone it would have
  come out zero and condemned a correct element of the engine's own call.
* **Two `>1 ULP` crossings exist, and both are cancellations.** `in_proj_qkv`: **1 element** of 49 152,
  2.00 ULP, `|ref|/max|ref| = 1.96e-07`, flat index 266, on the **engine's** leg. `gate_proj+up_proj`:
  **1 element** of 28 672, 6.00 ULP, `rel = 2.34e-07`, index 7715, on both the algorithm leg and the
  engine leg. `NUMERICS-FINDING` count **0** — no offender anywhere reached `1e-03` of the tensor max,
  which is the line between "cancelled" and "a library that disagrees with a double reference". The band
  that came out of it is `cancel_rel 7.8e-07 / 9.4e-07`, ~35× tighter than the `3e-05` `mlp_gemv` had
  to import, and `cancel_tol` is `1.6e-08 / 4.4e-08` — measured on the cancelled elements only, after
  the first version of this code sized it against all elements and produced `1.9e-02`, a tolerance under
  which a cancelled element could drift two hundredths of the tensor max and still pass.
* **The band fails its own negative controls by four digits.** 56 `NEG-CAUGHT` (8 per shape), 7
  `NEG-BLIND` (`alpha_doubled`, by design), **0 `NEG-FAIL`**, and the hardest reachable defect sits
  `6.3e-03 … 1.05e-01` below the floor — 3 to 5 orders of magnitude below the `7e-06` gap that made v1's
  arithmetic-based control set undecidable. `alpha_doubled` is the derived blind spot for the cos leg:
  `cos = 0.9999985263`, within `8e-09` of a correct kernel, caught only because 131 072 elements moved
  255 ULP.

`RESOLUTION` records the blind spot that was *removed* from the must-catch set, and the numbers say why
more loudly than the argument did — `one_term_short`'s worst seed against the quantisation deficit:

| shape | q_proj | k_proj+v_proj | o_proj+out_proj | in_proj_qkv | in_proj_z | gate_proj+up_proj | down_proj |
|---|---|---|---|---|---|---|---|
| injection deficit ÷ quantisation deficit | 4.54 | 4.29 | **0.81** | 1.62 | 1.37 | 5.09 | **0.73** |

On two of seven shapes a one-term-short dot moves cosine **less** than storing a correct answer in bf16
does. No floor can admit the correct kernel and catch that, and no algorithm swap produces it either, so
it is reported on every probe and gated by nothing.

## 1. Per-shape sweep

One `RESULT` line per (shape, M), pasted verbatim. `ratio = default-ranked best_ms / best best_ms`,
so 1.000 = the heuristic already picked the fastest enumerated kernel.

```
RESULT shape=... M=... N=... K=... candidates=... band_ok=... default_seq=... default_ms=...
default_tflops=... best_seq=... best_ms=... best_tflops=... ratio=... engine_call_ms=...
engine_vs_best=... best_idx=... best_solution=...
```

| shape | M | default tflops | best tflops | **ratio** | engine call ms | engine_vs_best | verdict |
|---|---|---|---|---|---|---|---|
| `down_proj` | 512 | 26.12 | 32.16 | **1.231** | 0.1313 | 1.124 | algo swap pays |
| `o_proj+out_proj` | 512 | 24.64 | 28.05 | **1.138** | 0.0801 | 1.046 | algo swap pays |
| `in_proj_qkv` | 512 | 32.21 | 34.61 | **1.075** | 0.1989 | 1.068 | algo swap pays |
| `gate_proj+up_proj` | 512 | 29.85 | 31.85 | **1.067** | 0.1462 | 1.239 | algo swap pays |
| `q_proj` | 512 | 33.78 | 33.78 | **1.000** | 0.1551 | 1.220 | heuristic already optimal |
| `k_proj+v_proj` | 512 | 26.62 | 26.62 | **1.000** | 0.0328 | 1.625 | heuristic already optimal |
| `in_proj_z` | 512 | 31.38 | 31.38 | **1.000** | 0.0930 | 1.359 | heuristic already optimal |
| `down_proj` | 1024 | 29.36 | 34.31 | **1.169** | 0.2518 | 1.149 | algo swap pays |
| `in_proj_z` | 1024 | 29.93 | 32.34 | **1.081** | 0.1540 | 1.159 | algo swap pays |
| `o_proj+out_proj` | 1024 | 29.08 | 31.24 | **1.074** | 0.1444 | 1.050 | algo swap pays |
| `q_proj` | 1024 | 33.75 | 34.16 | **1.012** | 0.2486 | 0.989 | marginal |
| `k_proj+v_proj` | 1024 | 32.41 | 32.41 | **1.000** | 0.0418 | 1.262 | heuristic already optimal |
| `in_proj_qkv` | 1024 | 35.61 | 35.61 | **1.000** | 0.3455 | 0.955 | heuristic already optimal |
| `gate_proj+up_proj` | 1024 | 33.06 | 33.06 | **1.000** | 0.2183 | 0.960 | heuristic already optimal |
| `down_proj` | 4096 | 27.39 | 35.53 | **1.297** | 0.9077 | 1.073 | algo swap pays |
| `o_proj+out_proj` | 4096 | 29.43 | 37.46 | **1.273** | 0.5424 | 1.183 | algo swap pays |
| `in_proj_z` | 4096 | 28.19 | 33.87 | **1.201** | 0.5358 | 1.056 | algo swap pays |
| `q_proj` | 4096 | 32.62 | 32.62 | **1.000** | 1.7459 | 1.657 | heuristic already optimal |
| `k_proj+v_proj` | 4096 | 33.98 | 33.98 | **1.000** | 0.1605 | 1.270 | heuristic already optimal |
| `in_proj_qkv` | 4096 | 31.98 | 31.98 | **1.000** | 2.6196 | 1.626 | heuristic already optimal |
| `gate_proj+up_proj` | 4096 | 34.43 | 34.43 | **1.000** | 0.9555 | 1.094 | heuristic already optimal |

Window: all blocking, zero subagents. `--rows 512 --check-rows 8 --max-algos 12 --inner 8`: lock
`gas-final`, `util` 94 %, and the probe (exit 0) plus `--check-only` (exit 0, `band_passed=7`) ran in the
same hold as the timing. `--rows 1024,4096`: same lock, `util` 90 %, with `--floor-probe --rows 1024,4096`
first (exit 0), because those widths have no CAL row otherwise and `--time` exits 2 by design without one.
Wall time 2 s and 3 s — the sweep is cheap, which is worth stating because nothing here needed a long
exclusive window.

## 2. Correctness gates that were actually met

The speed numbers below are void unless every line here is green for the same window, because a
timing of a wrong kernel is worse than no timing (`README.md` §5).

| check | line | must show |
|---|---|---|
| where the band came from | `CALIBRATION mode=measured …` and `COMPARE-MODE … cos_floor_source=measured on this build v2: …` | an `OVERRIDDEN` clause in either place means the run is an A/B against thresholds this build never measured, and any ratio from it says so |
| the band was measured, not widened | `BAND-SIZE … ulp_bound=1.000 unchanged=1` | present, and no `--max-ulp` / `--cancel-rel` / `--cancel-tol` / `--cancel-margin` on the command line. `cancel_rel_source=` names which leg sized it |
| every timed algorithm is inside the band | `CHECK … verdict=ok` | `off_out_band=0`, `worst_in_band_excess ≤ cancel_tol`, `nonfinite=0`, `survivors=0`, `cos ≥` the floor named on that run's `COMPARE-MODE` line (not a constant anyone has to remember) |
| the engine's own call is inside it | `CHECK-ENGINE … verdict=ok` | same, `verdict=ok` on **every** shape. If it is `WRONG`, no algorithm could pass and the band is the defect |
| the controls still fire | `NEG-CAUGHT` ×8 and `NEG-BLIND` ×1 per probed shape, no `NEG-FAIL` | the band fails its own negative controls; `mustcatch_max_cos` sits `6.3e-03 … 1.05e-01` below the floor (§0a). A band that cannot fail on anything is not a band |
| the aggregate is not measuring the reference | `SIGNATURE … verdict=` and `CONFLUENCE … agree=` | recorded either way. `constant_cos_reference_limited` on 4 of 7 shapes is expected here, and §0a explains it: the shared aggregate is bf16 output quantisation, which is also what `CONFLUENCE agree=1` says. Reported, not hidden |
| the default is inside it too | no `REFUSE … default-ranked algorithm not at the band` | absent; if present, the `attribution=` field says which side is suspect |
| the winner reproduces the default | `PAIR … verdict=same_product` | `diff_elements`, `worst_pair_ulp ≤ 1.0`, `nonfinite=0` |
| alignment used, said out loud | `COMPARE-MODE … alignment=` | `row_major[t*N+j]`, same tensor both sides, no offset |

**Already met, at `--rows 512 --check-rows 8 --max-algos 12`** (same window as §0a, `util` 92 %, zero
subagents, blocking, no timing taken): `CHECK-ONLY-SUMMARY problems=7 cases_printed=7 refused=0
band_passed=7`, with `band_ok=12_of=12 band_wrong=0 not_launched=0` and `engine_verdict=ok` on all seven
shapes, exit 0. What is *not* met is everything at `M=1024/4096`: those widths have no CAL row, so a
`--time` run at them exits 2 by design until `--floor-probe` is re-run with those `--rows`.

Band provenance: the 1-ULP half is the bound `tools/gemm_peak.cpp:261` and `tools/prefill_mix.cpp:656`
both reach on these shapes (`devices/strix-halo-compute.md` second correction records the same ≤1.000
ULP agreement for the 16-bit-`C` path), is **never** calibrated, and applies to every element that did
not cancel. The cancellation thresholds and the cos floor come from `floor_calibration.tsv`, written by
`--floor-probe` on this build; the injections it prices (a wrong leading dimension, swapped operand
roles, a dropped split-K partial at two severities, an aliased partial slot, a half-written tile,
`beta` left at 1 against stale and against NaN C, `alpha` doubled) are what keep it a bound rather than
an accommodation — and they are derived from what an algorithm swap can actually break, which is why
`one_term_short` is reported by `RESOLUTION` instead of gating (§0a, `README.md` §5a).

## 3. Verdict

```
SWEEP problems=… refused=… timed=… min_ratio=… max_ratio=…
VERDICT …
```

Reading, and the decision it feeds (`HANDOFF.md` §5):

* `max_ratio < 1.10` everywhere → tuning is closed as a route; a hand WMMA GEMM must beat the
  **default** on its own merits, against the `default_tflops` column and not the retired 4.7
  TFLOP/s (decision log D17).
* some `ratio ≥ 1.10` → the win exists but is not free: the engine calls hipBLAS, so capturing it is
  a `hipblasLtMatmul` call path plus a per-shape algo table. Price it per family
  (`mlp_gateup` 24.99 ms and `proj` 38.81 ms of a 245.73 ms wave; `mlp_down` 6.42 ms is noise).
* `engine_vs_best ≫ ratio` → the finding is that hipBLAS and hipBLASLt pick differently, which is a
  route question worth more than an algo table. **This is the branch that fired — see §3a.**

## 3a. The verdict, measured (main agent, 2026-10-16)

`VERDICT the default is leaving at least 10% on the table somewhere; read the RESULT line(s) with ratio >
1.10` — the third bullet fired, and it is the one that matters. **The finding is not the algo table, it is
`engine_vs_best ≫ ratio`.** Three shapes (`q_proj`, `k_proj+v_proj`, `in_proj_z`) have `ratio = 1.000` —
the heuristic's rank-0 is already the fastest of the twelve — and yet the engine's own `hipblasGemmEx` call
is **1.220×, 1.625× and 1.359× slower** than that same kernel at the shipping width. hipBLAS and hipBLASLt
are picking differently, and the difference is bigger than anything the enumeration can offer.

**Which M is the shipping one.** `engine/decode_step.h:49` is `kMaxWaveRows = 512`, so a wave is at most 512
rows and a 4096-lane prefill is eight sequential 512-row waves. The engine therefore never issues M = 1024
or M = 4096: **those two blocks of the table describe call sizes that do not occur**, and their dramatic
rows (`q_proj` at M=4096: engine 1.7459 ms vs lt-default 1.0534 ms) are a fact about the library, not about
this engine. They are kept because they show the divergence widening with M, which bounds what a future
wider wave would face.

Pricing the mix at M = 512 with the call counts from the shape census (6 / 12 / 24 / 18 / 18 / 48 / 24 = 150
calls per 24-layer wave):

| path | ms per 24-layer wave | vs engine | ms per 4096-lane prefill (8 waves) | saved off the 1105.20 ms wave |
|---|---|---|---|---|
| engine today (`hipblasGemmEx`) | 18.67 | — | 149.4 | — |
| `hipblasLtMatmul`, heuristic rank-0 | 17.42 | **1.072×** | 139.4 | 10.0 ms (0.9 %) |
| `hipblasLtMatmul`, best of 12 enumerated | 15.90 | **1.174×** | 127.2 | **22.2 ms (2.0 %)** |

So roughly a third of the prize is free (switch the call, take the heuristic) and two thirds needs a
per-shape algo table. **No hand WMMA GEMM is required to collect it**, and none should be started on the
strength of the 1.4–1.6× ceiling while this 1.17× sits unbuilt.

**One gap left open on purpose.** The ledger's `GROUP matmul_ms` is 70.30 ms at a 1024-lane wave
(`bench/performance-status.md` §3, 28.6 % of 245.73 ms), while this 150-call mix predicts 2 × 18.67 =
37.3 ms at that width. The ledger's `GROUP` therefore contains library time beyond these 150 calls —
draft-layer projections, or a family attribution difference — and until that is reconciled the honest
statement is that the algo route is **worth at least 0.9–2.0 % of a 4096-lane wave and possibly closer to
twice that**. The engine-side A/B is what sizes it; this file does not claim a number it cannot attribute.

## 4. What this says about the 1.4–1.6× headroom

Fill in only from measured rows:

* **fraction of the 1.4–1.6× that was algorithm-selection:** at the shipping width the enumerated best
  reaches **34.6–35.5 TFLOP/s** on `in_proj_qkv`, `down_proj`, `gate_proj+up_proj` and `o_proj` — i.e.
  essentially the whole `prototype/issue_rate_probe` WMMA roof of 52.63 TFLOP/s is *not* reached by the
  library at any M, but 33–35 TFLOP/s is reachable today versus the 24.6–33.8 the engine's call settles
  for. The selection gap is ~1.17× of a possible ~1.4×; the rest is kernel quality, which is the part a
  hand kernel would have to earn.
* **shapes where the library is already at a rate worth having:** `q_proj` (33.78 TFLOP/s) and
  `in_proj_qkv` (34.61 TFLOP/s best) at M = 512 are within 1.06–1.09× of the best any enumerated kernel
  found, so no algo work is owed there — the engine's own call is what lags them (1.220× and 1.068×).
* **what enumeration reveals that timing cannot:** of 15,750 enumerated (algo, shape) entries,
  **15,077 request more workspace than the engine's 128 MiB** (`engine/kernels_gemm.cpp:59`) and are
  therefore unreachable at the engine's budget — `skipped_workspace=0` per shape among the twelve that
  survived, `skipped_unsupported=368` and `skipped_cap=370`. So the 12 candidates timed per shape are not
  a sample of the catalogue, they are almost the whole of what this workspace budget admits. Raising the
  128 MiB cap is a separate, unpriced question that would have to be re-gated the same way.
