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

## 3b. Reconciling that gap: what `matmul_ms = 70.30 ms` is actually made of (2026-10-17)

§3a left the gap open on purpose. This closes the part that enumeration and the ledger's own
evidence can close, and names the rest precisely.

**The group is three families, not one kind of call.** `stage_clock.cpp:454` defines
`matmul_ms = share({kProj, kMlpGateup, kMlpDown})`, and §3's three family medians — 38.81, 24.99,
6.42 — sum to 70.22 against the printed 70.30 (the group is the median of per-run sums, so the
0.08 ms is rounding across runs, not a missing family). Nothing else is in the group. So the gap is
not a fourth family hiding inside `matmul`; it is that a family's segment measures **everything**
launched under that family's note, and the sweep prices only library calls.

**Sites are not calls.** `--time-waves` counts launches per note, and `stage-wave.txt` records
`family=proj launches=228`, `family=mlp_gateup launches=48`, `family=mlp_down launches=48` — 324
sites for the 1024-lane wave (`--wave-rows 512`, two bodies). The same file's
`TOTAL … launches=20264 plan_launches=20264 plan_match=1 launchstats_match=1` means the engine's
independent structural bill agreed with every family count, and that bill
(`decode_step.h:300`, `stage_clock.cpp:86,117,118`) has no draft or MTP entry at all. Decomposing
the 324 sites against the source:

| family | sites/wave | **library calls/wave** | non-library launches inside the same segment |
|---|---|---|---|
| `proj` | 228 | **156** | 72 — `in_proj_a`/`in_proj_b` at `N=16`. `proj_gemm` refuses `N < 512` (`kernels_proj.cpp:135-142`, `kSmallestLibraryRows`) and the two sites call the hand `proj_gemv` directly at any width (`decode_step.cpp:532,535`) |
| `mlp_gateup` | 48 | **96** | 48 — the `silu(gate)*up` epilogue kernel: `mlp_gateup_gemm` is three GPU submissions counted as **one** site (`kernels_mlp.cpp:253-268`), so its third launch carries no family note of its own |
| `mlp_down` | 48 | **48** | 0 |
| **total** | **324** | **300 = 2 × 150** | **120** |

So the census was complete and there is nothing to add to it: no draft-layer projections (no MTP site
exists in `engine/*.cpp` outside `loader.cpp:366-376`, which only maps draft weight *names*), no
`lm_head` GEMM (that is a hand `gemv_bf16` on its own family, `kernels_lmhead.cpp:25`), no second
library path. The 1024-lane wave issues exactly 300 library GEMMs and 120 non-library launches
inside `matmul_ms`.

**The enumeration the gap needed** — shape | N | K | M | calls/wave | family | file:line. Every row is
`M = 512` because `kMaxWaveRows = 512` (`decode_step.h:49`): a 1024-lane wave is two 512-row bodies.
`engine-call ms` is §3's `engine_call_ms` for that shape at that `M`; `family ms/wave` is that number
× calls/wave.

| shape | N | K | M | calls/wave | family | call site | engine-call ms | family ms/wave |
|---|---|---|---|---|---|---|---|---|
| `q_proj` | 4096 | 1024 | 512 | 12 | `proj` | `decode_step.cpp:513` | 0.1551 | 1.861 |
| `k_proj`, `v_proj` (two weights, one shape) | 512 | 1024 | 512 | 24 | `proj` | `decode_step.cpp:516,519` | 0.0328 | 0.787 |
| `o_proj` (attn) / `out_proj` (GDN) — same shape, one site | 1024 | 2048 | 512 | 48 | `proj` | `decode_step.cpp:721` | 0.0801 | 3.845 |
| `in_proj_qkv` | 6144 | 1024 | 512 | 36 | `proj` | `decode_step.cpp:526` | 0.1989 | 7.160 |
| `in_proj_z` | 2048 | 1024 | 512 | 36 | `proj` | `decode_step.cpp:529` | 0.0930 | 3.348 |
| `in_proj_b` | 16 | 1024 | 512 | 36 | `proj` **(hand)** | `decode_step.cpp:532` | — | below |
| `in_proj_a` | 16 | 1024 | 512 | 36 | `proj` **(hand)** | `decode_step.cpp:535` | — | below |
| `gate_proj`, `up_proj` (two weights, one shape) | 3584 | 1024 | 512 | 96 | `mlp_gateup` | `kernels_mlp.cpp:262-263` | 0.1462 | 14.035 |
| `down_proj` | 1024 | 3584 | 512 | 48 | `mlp_down` | `kernels_mlp.cpp:276` | 0.1313 | 6.302 |
| **library total** | | | | **300** | | | | **37.338** |

Against the ledger: `proj` 38.81 vs 17.002 predicted, `mlp_gateup` 24.99 vs 14.035, `mlp_down`
6.42 vs 6.302 — residuals **21.81 / 10.96 / 0.12 over the published medians**, total **+32.88 ms**, which is the ~33 §3a
asked about. Per call that is `proj` **2.28×**, `mlp_gateup` **1.78×**, `mlp_down` **1.02×**.

| part of the 32.88 ms | ms | basis |
|---|---|---|
| 48 `silu·mul` epilogues, swept into `mlp_gateup` | **≥ 2.2** | traffic floor: 11.0 MB moved per call (3 × `512×3584×2` read + 3.67 MB write) × 48 = 529 MB at the 237 GB/s read plateau |
| 72 hand `in_proj_a`/`in_proj_b` launches, swept into `proj` | **≥ 0.7**, plausibly 1–4 | each re-reads the 512×1024 F32 activation tile (2.10 MB) that `qkv`/`z` already touched: 151 MB at the plateau. The hand kernel runs 16 rows on 32-wide blocks, so it is occupancy-poor at wave width and the floor understates it |
| the same 300 library calls, in situ vs warm burst | **≈ 26–30** | the residual once those two are subtracted: 1.55× on `mlp_gateup`, 1.02× on `mlp_down`, ~2.2× on `proj` |

Two candidate explanations are **excluded by the evidence**, which is worth recording because both
looked likely:

- **not measurement serialization.** `stage-wave.txt`'s own `WAVEGAP` line reports
  `exposed_ms = 7.1437` of 243.17 ms — `ratio=1.030` — so the per-family numbers are within 3 % of
  the pipelined wall and the stage clock is not draining between notes.
- **not cold weight traffic, even though the burst is cache-hot.** The wave reads each weight once
  (1.94 GiB of weight per wave); a burst re-reads one weight 40 times. That floor is
  `weight bytes ÷ 237 GB/s` per call: ≤ 4.1 ms/wave for `proj`, ≤ 3.1 for `mlp_gateup`, 1.6 for
  `mlp_down`. It cannot be the mechanism for `mlp_down`, which has the largest cold-read floor
  relative to its residual and a residual of 0.12 ms.

**What remains unaccounted, and what would account for it.** About 26–30 ms is the same 300 calls
costing 1.5–2.2× their `engine_call_ms` in the wave, and the split of that between activation traffic
the burst never pays and plain interleaving/dispatch effects **cannot be derived from anything on
disk**: the wave clock attributes at family granularity, the sweep measures at shape granularity, and
the two do not meet. Ledger §2 prices the same in-situ effect on the hand path at 1.25× (`proj`
4.1128 in situ vs 3.2843 isolated) and 1.42× (`mlp` 3.388 vs 2.378) — right direction, right order,
not enough to cover 2.28×.

Two measurements close it, and neither needs a new tool:

1. **`decode_run --time-waves` at `--gemm library` and `--gemm lt` inside one lock hold** (§7). Both
   numbers come from the same instrument as the 70.30, so their difference is in situ and needs no
   ruler translation. This is the number the question was asked for: the algo lever is 1.072× of
   *something*, and the two candidate denominators price it at 4.8 ms (of 70.30) or 2.5 ms (of 37.34)
   out of a 245.73 ms wave — 2.0 % vs 1.0 %.
2. If the attribution itself is wanted: split the `proj` note at the `proj_gemm`/`proj_gemv` seam —
   one extra note in `kernels_proj.cpp:128-143` and one line in `stage_clock.cpp`'s plan — retake §3
   in one window, and the hand share of `proj` stops being a bound and becomes a number. Same
   instrument, no new tool, and it retires §4's hand-projection question at the same time.


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

## 3c. The engine A/B, measured (`--gemm lt` vs `--gemm library`, main agent, 2026-10-16)

**The prize is bigger than §3a priced, and the projection was the thing that was wrong.** §3a said
1.174× on the 150-call mix and hedged that the ledger's `matmul` group carries more than that mix. It does,
and the lt path captures all of it. Both routes measured in **one** lock hold (`gemm-lt-ab`, `util` 92 %,
zero subagents, blocking), alternating `library`/`lt` inside each rep, medians of 3, `--steps 1
--time-waves` so the family rows are the same instrument as ledger §3.

| width | family | `library` median ms | `lt` median ms | **ratio** | the three runs on each side |
|---|---|---|---|---|---|
| **1024 lanes** | `GROUP matmul` | 71.28 | **49.58** | **1.437×** | 72.20/71.28/70.67 · 49.01/49.86/49.58 |
| | `proj` | 38.99 | 25.63 | **1.521×** | 39.98/38.75/38.99 · 25.62/25.87/25.63 |
| | `mlp_gateup` | 25.71 | 16.10 | **1.597×** | 25.87/25.71/25.34 · 15.62/16.10/16.13 |
| | `mlp_down` | 6.35 | 7.82 | **0.812×** ⚠ | 6.35/6.81/6.34 · 7.77/7.89/7.82 |
| | **whole wave** | 240.96 | **215.14** | **1.120×** | 241.82/240.96/238.93 · 215.21/215.14/219.73 |
| **3072 lanes** | `GROUP matmul` | 211.72 | **149.98** | **1.412×** | 210.52/211.72/213.10 · 149.98/151.21/149.19 |
| | `proj` | 116.98 | 76.65 | **1.526×** | |
| | `mlp_gateup` | 75.18 | 48.50 | **1.550×** | |
| | `mlp_down` | 19.16 | 24.56 | **0.780×** ⚠ | |
| | **whole wave** | 804.85 | **743.35** | **1.083×** | 797.77/804.85/812.51 · 737.12/743.35/749.44 |

Throughput at those widths: **4250 → 4760 prefill tok/s at 1024 tokens (+12.0 %)** and **3817 → 4133 tok/s
at 3072 tokens (+8.3 %)**. `WAVEPARITY tokens_identical=1 logits_bitidentical=1` on both routes in all six
runs (it compares burst against stage within a route; cross-route bit-identity is not owed — §7c of
`HANDOFF.md`).

Three things this says that the burst sweep could not:

1. **§3a's mix under-priced the win by ~4×** because the mix timed 150 calls at M = 512 in a tight loop,
   while the engine's `proj` family is 228 launches including the hand `in_proj_a/b` GEMVs and the epilogues
   that ledger §3 sweeps into these groups. The `proj` and `mlp_gateup` ratios (1.52×, 1.60×) are not the
   burst ratios of any single shape — they include launches that are not GEMMs at all. **A per-call burst
   table cannot project a family-level wave number**; this one was off by 4×, in the flattering direction.
2. **`mlp_down` regresses, 0.78–0.81×, and the sweep predicted it and I ignored the sign.** For
   `down_proj` at M = 512, lt rank-0 is `default_ms=0.1439` versus the engine call's `0.1313` — the ratio
   1.231 that made `down_proj` the headline "algo swap pays" row belongs to **best-of-12**, not to the
   default heuristic. Shipping rank-0 therefore loses here, and the fix is exactly what `algo_pins()` exists
   for: pin `down_proj`'s `best_seq=1` algo (0.1169 ms, 35.53 TFLOP/s) instead of taking the heuristic. A
   sweep column that says "best" must be read against the default column before any of it is called a win.
3. **The band question is the only thing between this and the default.** The gate's section (h) — batched
   vs one-row, which under `lt` is hand-GEMV vs hipBLASLt — reads worst per-row `1-cos 2.204e-03` on
   `dec64f` against a 2e-03 band calibrated for **hipBLAS-vs-hand**, a different pairing, while every
   pin-facing section passes and `lt`'s pooled numbers are *better* than `library`'s. That is a
   band-re derivation, not a bug, and it must be derived from a measured spread rather than by editing a
   constant. Until it is, the default stays `library` and a red gate stays red.
