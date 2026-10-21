# `prototype/gemm_algo_sweep/` — is hipBLASLt's default kernel the fastest one it owns?

Question this tool answers, in one line: **at the matmul shapes this engine actually issues, does
hipBLASLt contain an algorithm that beats the one its default heuristic ranks first?** If yes, the
1.4–1.6× headroom between the library and a hand WMMA loop
(`devices/strix-halo-compute.md` second-correction table 33.2–37.7 TFLOP/s against
`prototype/issue_rate_probe/results.md` §1's 52.63 TFLOP/s; decision log D17) is partly *available
without writing a kernel*. If no, a hand GEMM has to win on its own merits and the tuning route is
closed.

Matmul is 28.6 % of a 1024-lane prefill wave (`engine/performance-status.md` §3, `GROUP matmul_ms`
70.30 ms of 245.73 ms), so a 1.2× on the library is worth ~6 % of the wave — and it costs a tool run
rather than a kernel campaign.

Built by `bash prototype/gemm_algo_sweep/build.sh` → `gemm_algo_sweep` (gitignored binary; the
`.gitignore` line is the exact path, per the trap row about ignore patterns matching the directory).

---

## 1. The call form being swept

One form, one place: `engine/kernels_gemm.cpp:156 gemm_bf16()` →
`gemm_common::gemm_stored_weight()` (`tools/gemm_common.h:119`) → `hipblasGemmEx(..., HIP_R_16BF,
..., HIP_R_16BF out, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT)`.

So: **bf16 operands, f32 accumulate, bf16 `C`** — the 16-bit `C` being the whole reason the engine
reaches 33–38 TFLOP/s instead of the 4.7–6.0 that `devices/strix-halo-compute.md` first printed
(that file's second correction). The tool books the identical form: `OP_T` on the weight as the
shard stores it with `lda = k`, `OP_N` on the activation with `ldb = k`, `ldc = n`, `alpha = 1`,
`beta = 0`, one 128 MiB workspace on the default stream — the same size `engine/kernels_gemm.cpp:60`
pins, because workspace size feeds split-K eligibility and a route that left it to the library's
default would be making kernel choice a function of what the allocator last freed.

```
C_rm[m,n] = X_rm[m,k] · W_rm[n,k]^T     BLAS is column-major, so:  m_blas=n, n_blas=m,
                                       A = W with OP_T (lda=k), B = X with OP_N (ldb=k), ldc=n
```

Why the engine's own call is timed *inside* the sweep (`ENGINE-CALL`, `engine_call_ms`): hipBLAS
takes the kernel choice away from the caller (`HIPBLAS_GEMM_DEFAULT` and that is all), so hipBLASLt's
rank-0 answer is **not** automatically the kernel the engine runs. Without that column a ratio could
not be attributed to the engine at all.

### 1a. The operand pairing, and the fault it caused

The form above is easy to read and easy to get backwards, because two naming conventions describe the
same two buffers. `gemm_stored_weight(h, m, n, k, dt, A, W, C, dt_out)` names its parameters after
the *engine's* roles — its `A` is the activation. In **BLAS operand position** the first operand is
the one carrying `OP_T`, which is the **weight**. So the function's `A` is BLAS's `B`.

The first version of this file passed the activation buffer behind `plan.la`, the weight's descriptor.
The compiler cannot see it: both are `const void *`. What happens depends on the shape:

| shape | buffers | what happened |
|---|---|---|
| `n != m` (6 of 7 families) | the A descriptor wants `n·k` elements from an `m·k` buffer | **illegal memory access** on the first launch, whatever the algorithm |
| `n == m` (`k_proj` at M=512) | the two buffers are the same size | no fault — a silently **different product**, caught by the band (`REFUSE … worst_ulp>1`) |

That split is the whole diagnostic, and it is what made the fault cheap to localise. The rule is
**fault ⟺ n ≠ m**: checked on all 7 shapes at M=512, then re-checked at M=1024 and M=4096, where the
non-faulting set moved to exactly the shapes whose `n` equalled the new `M` (`down_proj` and
`o_proj+out_proj` at 1024, `q_proj` at 4096). Ten observations, one rule, no guessing required.

Two guards exist so it cannot come back silently:

* **names**: the buffers are `hX`/`dX` (activation) and `hW`/`dW` (weight), never `A`/`B`. A name
  that means one thing in `gemm_common.h` and another in BLAS is how the bug was written.
* **`--selftest-layout`**: host-only, no HIP call. It re-derives each descriptor's required extent
  from the same `Plan::set_dims()` the launch is built from and requires that extent to equal the
  buffer paired with it, then re-injects the A/B swap and requires the check to catch it on the 17
  problems where `n != m` and to *not* claim a catch on the 4 where `n == m` — an extent check
  structurally cannot see a swap between equal-sized buffers. A must-catch set with no derived
  must-not-catch set is the vacuous control this repo has been bitten by twice already.

For the `n == m` case the only witness left is the numerical band, and that is not a gap to
apologise for: it is the check that caught the bug on `k_proj` on device. `--selftest-sys`
(section 4a) covers everything checkable with no device at all.

## 2. The shapes, and where each came from

`./gemm_algo_sweep --list` prints this table out of the same constants the binary uses, and exits
non-zero if the per-wave count stops summing to 150. Shapes were also cross-read from the shard
header (`engine/weights_contract.header_shapes` over `models/Qwen3.5-0.8B`), which agrees with every
row. Leaves that share a shape are one row here, because the tool times a shape and not a weight.

| shape | family | N | K | calls / 24-layer wave | site it came from |
|---|---|---|---|---|---|
| `q_proj` | proj | 4096 | 1024 | 6 | `engine/decode_step.cpp:583` (`kQprojRows`, `kHidden`) |
| `k_proj` + `v_proj` | proj | 512 | 1024 | 12 | `engine/decode_step.cpp:586,589` (`kKvProjRows`) |
| `o_proj` + `out_proj` | proj | 1024 | 2048 | 24 | `engine/decode_step.cpp:810` (`kHidden`, `kQwidth`/`kValueDim`) |
| `in_proj_qkv` | proj | 6144 | 1024 | 18 | `engine/decode_step.cpp:596` (`kQkvRows`) |
| `in_proj_z` | proj | 2048 | 1024 | 18 | `engine/decode_step.cpp:599` (`kValueDim`) |
| `gate_proj` + `up_proj` | mlp_gateup | 3584 | 1024 | 48 | `engine/kernels_mlp.cpp:262,263` ← `decode_step.cpp:748` |
| `down_proj` | mlp_down | 1024 | 3584 | 24 | `engine/kernels_mlp.cpp:276` ← `decode_step.cpp:762` |
| | | | | **150** | matches the "150-call 24-layer wave mix" priced in `devices/strix-halo-compute.md` |

Family attribution is the one `engine/performance-status.md` §3 prints (proj 38.81 ms, `mlp_gateup`
24.99 ms, `mlp_down` 6.42 ms of a 245.73 ms wave).

**Deliberately not swept, with the reason** — the brief listed `lm_head` among the families and it is
not a library call:

| excluded | why |
|---|---|
| `in_proj_a` / `in_proj_b`, N=16 | `proj_gemm` **refuses** them (`engine/kernels_proj.cpp:134-141`); the engine guard itself records why (12–22× the dispatch floor at that N), so they stay in the hand GEMV at every width |
| `lm_head`, 248320×1024 | `lm_head_full` (`engine/kernels_lmhead.cpp:25`) is a hand GEMV over an **F32** activation, run once per request, not per wave row. `engine/performance-status.md` §3 lists it as a wave *family*, which is true of the hand kernel; no route hands it to hipBLAS, so sweeping it would price a call the engine does not make |
| a body of one row (M=1) | `resolve_wave_gemm` sends M≤1 to the hand GEMV whatever the run asked for (`engine/kernels_gemm.cpp:844-850`); `--rows` refuses 1 rather than timing a shape the library never sees |

### Wave widths (the M axis), and an honest correction

`--rows` defaults to **512, 1024, 4096**. The brief asked for 1024 and 4096, and the first of the
three is the one that matters today: **the shipped per-call cap is 512 rows** —
`engine/decode_step.h:49 kMaxWaveRows = 512`, and `engine/decode_run.cpp:279` refuses a larger
`--wave-rows`. A 1024-lane prompt is therefore two 512-row bodies, and 4096 lanes is `512 × 8`
(`engine/results.md` ceiling note). So:

* **M = 512 is the shipped call shape** and carries today's headline. Left in because the brief's
  "wave widths that carry the headline" are prompt widths, and the two are not the same number.
* **M = 1024 and 4096 are route widths** — the shapes a wider wave would issue, and the ones
  `tools/prefill_mix.cpp` already prices (it runs M = 64 / 512 / 1024 / 3072). Reading them is the
  cheap way to know whether a wider-wave route is *also* better-tuned by the library before someone
  raises `kMaxWaveRows`.

## 3. What this build can and cannot enumerate

**The cuBLASLt-style enumeration API does not exist here.** Not a runtime refusal — the symbols are
absent, so calling them is a link error:

```
$ nm -D --defined-only /opt/rocm/lib/libhipblaslt.so | grep " T hipblasLtMatmulAlgo"
00000000002d47c0 T hipblasLtMatmulAlgoGetHeuristic          # <-- the ONLY Algo* entry point

$ nm -D --defined-only /opt/rocm/lib/libhipblaslt.so | grep -c \
    "AlgoGetIds\|AlgoInit\|AlgoCapGetAttribute\|AlgoConfigSetAttribute"
0
$ grep -c "hipblasLtMatmulAlgoGetIds\|hipblasLtMatmulAlgoInit\|hipblasLtMatmulAlgoCapGet" \
    /opt/rocm/include/hipblaslt/*.h | grep -v ":0"
                                                  (no output: absent from every header too)
```

Consequence, stated plainly: **there are no tile / split-K / stages / occupancy fields to print.**
`hipblasLtMatmulAlgo_t` is 24 opaque bytes — `uint8_t data[16]` plus `size_t max_workspace_bytes`
(`/opt/rocm/include/hipblaslt/hipblaslt.h:419-433`) — with no accessors, and
`AlgoCapGetAttribute`/`AlgoConfigGetAttribute` are not there to ask. A version of this tool that
printed `tile=… splitk=…` would be printing fiction.

The same header note is what licenses `--algo-dump`: the struct "can be trivially serialized and
later restored for use with the same version of the library to save compute time". A captured
algorithm is therefore a legitimate offline artifact, which is how `--selftest-sys` audits real algo
bytes with no device in sight.

Enumeration *does* exist, in the C++ extension namespace, and the tool uses it:

| call | what it contributes |
|---|---|
| `hipblasLtMatmulAlgoGetHeuristic(…, requested=32, …)` | the ranking the library would use, in its own order; **rank 0 is "the default"** |
| `hipblaslt_ext::getAllAlgos(handle, GEMM, OP_T, OP_N, bf16×4, COMPUTE_32F, out)` | the full catalogue for the type combination (the enumeration path this build does expose) |
| `hipblaslt_ext::matmulIsAlgoSupported(desc, layouts, algo, &ws)` | whether a catalogue entry supports *this problem*, and the workspace it needs |
| `hipblaslt_ext::getIndexFromAlgo` / `getSolutionNameFromAlgo` / `getKernelNameFromAlgo` | the identity: solution index, solution name, kernel name |

So the per-algorithm line carries everything the API will actually say — `seq`, `heur_rank`,
`default`, `idx`, `ws`, `heur_ws`, `waves`, `state`, `supported`, `support_status`, the 24 raw bytes,
solution name and kernel name — and nothing it will not.

`ldd -r ./gemm_algo_sweep` reports **no undefined symbols**, which is the CPU-side proof that all
five `hipblaslt_ext` entry points and every C-API call resolve in `/opt/rocm/lib/libhipblaslt.so.1`.

## 4. Modes, and what each touches

| mode | device work | prints |
|---|---|---|
| `--list` | none (no HIP call at all) | the shape census above |
| `--selftest-layout` | **none** — returns in `main` before any HIP call | `LAYOUT` per problem, `NEG-BLIND` for the `n == m` set, `SELFTEST-LAYOUT PASS/FAIL` |
| `--selftest-sys --algo-file F` | **none** | `LAYOUT`, `OPAQUE`, `PAIR` per captured algorithm, `NEG-CAUGHT`/`NEG-BLIND`/`NEG-FAIL`, `VIOLATION`, `SELFTEST-SYS PASS/FAIL`. Exit 3 means the input was incomplete, which is deliberately not a pass |
| `--verify` | `hipblasLtCreate` only: context + device properties. **No operand buffer is allocated, no matmul is launched, no timing is taken** | `ENV`, `PLAN`, `HEUR`, `ENUM`, one `ALGO` line per algorithm, `SKIP`, one `CASE` line per problem, `PLAN-SUMMARY`, `VERIFY ok` |
| `--verify --algo-dump F` | as `--verify` | also writes one `ALGODUMP` line per algorithm, for `--selftest-sys` |
| `--time --check-only` | operands, workspace, one correctness launch per algorithm plus one engine call — **no timing at all, so no `ms` or TFLOP/s line exists to misquote** | `LAYOUT`, `ARGS`, `CHECK`, `SKIP`, `CHECK-SUMMARY`, `CHECK-ENGINE`, `SIGNATURE`, `CASE`, `CHECK-ONLY-SUMMARY` |
| `--time` | everything: operands, workspace, correctness runs, timed runs | `CALIBRATION`, `PLAN*`, `LAYOUT`, `COMPARE-MODE`, `CHECK`, `SKIP`, `CHECK-SUMMARY`, `CHECK-ENGINE`, `SIGNATURE`, `TIME`, `ENGINE-CALL`, `PAIR`, `RESULT`, `CASE`, then `SWEEP` + `VERDICT` |
| `--time --algo-seq N` | one candidate only, implies `--check-only` | the lines for that one algorithm, in a process that can be sacrificed |
| `--floor-probe --calibration-out F` | one default-algo call and one engine call per (shape, M, seed) — **no timing**, but it is a device run and needs the lock like any other | `FLOOR` per leg, `FLOOR-RESULT` per problem, `PROBE-REFUSE` when it declines to calibrate, `FLOOR-SUMMARY`, `FLOOR ok/FAILED`; writes `CAL` rows to `F` |


### 4a. The argument predicates

`hipblasLtMatmul` takes sixteen arguments and reports a bad one by faulting inside the launch, whose
only on-host symptom is a failed `hipStreamSynchronize` — which is how two `--time` windows died
naming nothing but "the warm-up sync". The tool now builds the whole argument list as a `LaunchArgs`
struct before launching, runs `validate_args()` on it, and **refuses to launch a list that fails**
(`ARGS-REJECT`, then a `REFUSE`, no kernel submitted). The same function is what `--selftest-sys`
runs, so the checks cannot drift from the thing they guard, and a `FAULT` line now carries the entire
list (`FAULT-ARGS`) instead of leaving the next reader to guess.

| tag | what it forbids | where it can be evaluated |
|---|---|---|
| `A_extent_matches_desc`, `B_extent_matches_desc`, `C_extent_matches_desc`, `D_extent_matches_desc` | a descriptor paired with a buffer that is not the one it describes | host and device |
| `D_eq_C_when_beta_zero` | `beta = 0` with `C != D`: "where the answer lands" stops being one fact, and a kernel that reads `C` anyway reads something arbitrary | host and device |
| `ptrs_nonnull`, `alignment_16B` | a null or misaligned operand / scale pointer | host (synthetic addresses) and device (real) |
| `scales_finite` | non-finite `alpha`/`beta` | host and device |
| `workspace_nonnull`, `workspace_requested_fits` | an algo asking for more workspace than exists | host and device |
| `algo_max_workspace_fits` | the algo's own `max_workspace_bytes` exceeding the allocation | host and device |
| `algo_state_success`, `algo_supported` | launching an entry the enumeration marked not-SUCCESS. Both were **printed but never enforced** before this existed | host and device |
| `algo_not_all_zero` | an all-zero `hipblasLtMatmulAlgo_t`, i.e. an uninitialised one | host and device |
| `desc_nonnull`, `layouts_nonnull`, `dtype_contract_bf16_in_f32_accumulate` | a null object, or dtype drift away from bf16-in / f32-accumulate / bf16-out | **launch site only** — a host-built list has no descriptors and says so (`SKIPPED-HOST-ONLY`) instead of pretending to pass |

`--selftest-sys` re-applies six injections and requires each to be caught **by the predicate whose
name says it** — `cd_split` by `D_eq_C_when_beta_zero`, `state_bad` by `algo_state_success`, `null_D`
by `ptrs_nonnull`, and so on. "Some tag fired" is not evidence: one unrelated violation would make
every injection look caught, which is the vacuous-control failure in the opposite direction.

`RESULT` is the **last** line of each shape's block, deliberately — AGENTS.md's trap row about a
baseline printed after the `RESULT` line swallowing it under `tail -1`. Grep for `RESULT`, not `ms`.

The heuristic is asked for 32 by default (`--requested`), i.e. well above 1, and
`--max-algos 0` (the default) keeps everything the catalogue returns.

## 5. The stated band, the comparison, and the alignment

Printed verbatim at run time by `COMPARE-MODE` so no reader has to trust this file:

* **Reference**, computed the way the repo computes references elsewhere: the dot in **double** of
  the *values the device was actually given* (bf16-decoded operands), then rounded to bf16 —
  `gemm_common::compare_16bit_output`, the same comparator `tools/gemm_peak.cpp:261` and
  `tools/prefill_mix.cpp:656` certify against. The reference dot costs `nsamp·N·K`, so it runs on
  `--check-rows` sampled rows of M with first and last always included; that is
  `tools/prefill_mix.cpp:667`'s precedent for the same arithmetic.
* **Band, part 1 (a bound, never widened)**: **no element may sit more than 1.0 bf16 ULP from the
  rounded reference — unless that element cancelled**, in which case the ULP denominator is meaningless
  and the element is judged by an absolute tolerance instead: `|Δ| ≤ cancel_tol × max|ref|`. The split
  line is `cancel_rel`: an element with `|ref| / max|ref| ≤ cancel_rel` is a cancellation. Both numbers
  come from the calibration (§5a); `1.0` ULP itself is **not** calibrated and is the bound
  `gemm_peak --check` and `prefill_mix --check` already meet on these very shapes. What did change from
  v1 is that `worst_ulp` alone is no longer the predicate: this tool's own first probe window found the
  engine's shipping call 2 ULP off at `in_proj_qkv` and 6 ULP off at `gate_proj+up_proj` **on one
  element per leg**, at `rel ≈ 2e-07` of the tensor max — a cancellation, not a wrong product. Flat
  `≤ 1` refused correct arithmetic; flat `≤ 6` would admit real bugs. The repo priced this same wall
  before and measured its way out: `prototype/mlp_gemv/results.md`, "How wide the cancellation band
  is" (raw max ULP 24 vs gated 0 over tol), with the comparator form in `prototype/bf16_ulp.py`.
  **AND zero `0xFFFF` filler survivors** (D is prefilled with `0xFF`, a bf16 NaN no finite dot can
  store, so a survivor is an unwritten element: the "pre-fill with a value writes never store" rule).
  **AND zero non-finite outputs** — a NaN compares false against every bound, and an earlier draft of
  `judge` reported a NaN row as `cos = 1.0`; the `beta_left_at_1_nan` negative control caught that
  (below), and `nonfinite=` is printed on every judgement line.
* **Band, part 2 (an aggregate, and therefore measured — see §5a)**: **cos ≥ the floor this build was
  calibrated to**, over each sampled row. There is **no constant in the binary**, and `--cos-floor`
  overrides only this half: a `--time` run still refuses without a calibration, because part 1's two
  thresholds have no other source. `COMPARE-MODE` prints `cos_floor_source=` naming the provenance.
  cos and meanrel are reported alongside the per-element test because max-ULP is blind to a
  whole-tensor 1-ULP shift and ndiff is meaningless for a long dot (AGENTS.md trap row) — a chain is
  judged by aggregates as well as by its worst surviving element.

* **Ordering.** Every algorithm that will be timed is first run once and judged. An algorithm outside
  the band prints `WRONG` and is **never timed**. If the **default-ranked** algorithm is outside it,
  the whole shape prints `REFUSE` and nothing is timed for it — that failure means the harness or the
  pairing is wrong, and timing the rest would launder it into a speed number.
* **Requirement 4, the winner-vs-default check.** After the winner is known, both are re-run into
  separate buffers and compared by `PAIR`: `diff_elements` over the whole `M×N` buffer,
  `worst_pair_ulp` between the two device outputs, and the reference band re-run on the winner.
  `verdict=same_product` requires `worst_pair_ulp ≤ band` **and** the winner inside the band **and**
  no non-finite pair.
* **Alignment** — said because this repo has been bitten twice by comparing the wrong index of a
  reference. Every buffer here is **row-major `[M][N]` with `ld = N`**, so element *(t, j)* lives at
  **`t*N + j`** on every side: device output, host reference slice, and the default-vs-winner pair.
  The pair comparison is *the same tensor produced by two kernels*, compared at identical indices —
  there is no `s` vs `s+1` pairing here (that is `specification/10-reference-dumps-and-gates.md` §10.5's pin/`greedy_continuation` convention, which
  does not apply), and the only index transform in the file is the column-major BLAS view of the same
  row-major buffer. A wrong `ld`/`OP_T` pair cannot survive the reference band; that is what makes
  the band the real check on the layout, since a wrong pair computes a *different product* silently
  (`tools/gemm_common.h:111-126`).

## 5a. Where the band comes from, and what keeps it honest

The floor this tool shipped with was `0.999999`, imported from another op's gate. On the first device
window it refused **7 of 7 shapes** while missing by `1.8e-06`, with `worst_ulp` sitting exactly at
`1.0000`, `survivors=0`, `ndiff` wandering 79 → 236 across the twelve candidates, and **one `cos`
shared by all twelve to the last digit**. Distinct kernels do not agree that precisely about themselves
while disagreeing about how many elements moved; they agree about the thing they all share. That is the
signature of a band measuring its own reference, and the answer is a measurement, not a loosening.

`--floor-probe` is that measurement, modelled on `prototype/glue/floor_probe.py`. For each `(shape, M)`
it sweeps `--seeds` operand fills and prices four legs, then writes one `CAL` row per shape carrying
**the whole band** — the cos floor *and* the two cancellation thresholds.

| leg | what it is | what it licenses |
|---|---|---|
| **device** | the default-ranked hipBLASLt algorithm **and `hipblasGemmEx` — the engine's own shipping call** (`engine/kernels_gemm.cpp:171`) | what a correct kernel on this box actually does. If the *engine* leg is outside the floor too, no algorithm in the library could ever pass: the band or the reference is the defect |
| **reference** | three legal summation orders of the same dot (serial, reversed, strided-4-then-fold) against the mode-0 reference the gate uses | whether the reference moves by itself. Measured: **it does not** — `ndiff=0`, `worst_ulp=0.0000` on every shape and seed |
| **license** | the same products accumulated in **F32**, serial and in four strided partials — what `HIPBLAS_COMPUTE_32F` entitles a kernel to do | the cancellation band's first candidate size. Measured here: also bit-identical (`ndiff=0`), because a dot of bf16 products accumulates exactly in F32 at these K — so on this box the device legs, not the license legs, are what needed exempting |
| **must-catch** | defects an *algorithm swap can actually produce*, encoded to bf16 and judged by the same `compare_16bit_output` a kernel's output goes through | if `cos_floor ≤ mustcatch_max_cos` the probe **refuses to write the row** (`PROBE-REFUSE reason=band_would_not_discriminate`): a floor that cannot catch its own negative controls is not a floor |

### The confluence, and why it is the argument for calibrating at all

`CONFLUENCE` is printed per shape. The device cos, the engine cos and the reordered-reference cos land on
the same number — `q_proj 0.999997974322` on all three — with a spread of `≤1.2e-09`, i.e. **below 1 % of
the deficit they all sit at** (the test is relative for exactly that reason; demanding bit-equality
called a `1e-10` spread "disagreement"). The deficit is `1.83e-06 … 2.32e-06` across the seven shapes,
and `meanrel` is `1.4e-03` throughout — and `1 − 1.4e-03²/2` *is* `0.999998`. So the deficit is not
summation-order noise and not a kernel defect: **it is what storing the answer in bf16 costs.** A floor
below it would be asking a bf16 kernel to be better than its output format. That is the measurement that
licenses calibrating instead of refusing forever.

### The must-catch set is derived from the object, not from arithmetic

v1 priced "drop a term of the dot" and then refused to calibrate, because a one-term-short chain landed
7e-06 from a correct kernel. The refusal was right and **the set was wrong**: no choice of hipBLASLt
algorithm drops one term of a dot. The set is now derived from what an algorithm swap can get wrong
(same discipline as `prototype/attn_tile_model.py`'s derived `neg_expect`), and each entry names the
predicate that must catch it — "some tag fired" is vacuous:

| injection | the defect it stands for | caught by | cos: q_proj, worst of 4 seeds |
|---|---|---|---|
| `ld_b_wrong` | BLAS-B read with a leading dimension that is not `k` | cos | 0.528 |
| `operand_role_swapped` | both operands from the weight buffer — **this file's own fault-1 bug** | cos | 0.546 |
| `splitk_half_dropped` | one of two split-K partials never added | cos | 0.217 |
| `splitk_eighth_dropped` | one of eight partials never added | cos | 0.941 |
| `partial_slot_aliased` | a partial slot written by the wrong row (attn §20's class) | cos | 0.952 |
| `half_tile_stale` | half the tile stored, half left as found | cos | 0.703 |
| `beta_left_at_1` | `D = α·prod + 1·C` against a stale C | cos | **0.985** = `mustcatch_max_cos` |
| `beta_left_at_1_nan` | the same against the `0xFF` filler | non-finite element | 0.000 |
| `alpha_doubled` | α applied twice | **the ULP leg only** | 0.9999985263 — see below |

`alpha_doubled` is the **derived blind set** for the cosine leg, printed as `NEG-BLIND` rather than
quietly omitted: cosine is scale-invariant, so doubling the product leaves `cos` within `8e-09` of a
correct kernel's value while moving **all 131 072 sampled elements by 255 ULP**. Naming it is what makes
it a blind spot on the record instead of a surprise.

And `one_term_short` left the set but not the log. `RESOLUTION` prints it every probe: at `q_proj` its
worst seed moves cos to `0.999990799880`, a deficit `4.54×` what bf16 output rounding costs anyway —
same order of magnitude, and the *worst* of four seeds is what prints (seed 1 gives `0.9945`, seed 3
gives the above, because the dropped term's size relative to its dot varies). No cos floor can admit a
correct kernel on every seed and still catch it. It is also unreachable by any algorithm. Both halves
are printed with a `PERMANENT` note, so nobody rediscovers the blind spot and files it as a bug — the
same class as `prototype/mlp_gemv/results.md`'s F4 finding that ndiff is blind to chains.

### Rules the probe enforces rather than asserts

* `cos_floor = min(worst device cos over seeds) − --floor-margin` (default `1e-05`, printed). Never a
  bare constant. `--cos-floor` overrides the cosine half only, and prints
  `cos_floor_source=… OVERRIDDEN … NOT a measurement of this build`.
* **`cancel_rel` / `cancel_tol` are the measured worst offender of a *legal* leg × `--cancel-margin`
  (default 4)** — exactly how `prototype/mlp_gemv` moved its band from `1e-4` to `3e-5` with a stated
  3.7× margin. Measured: `in_proj_qkv 7.83e-07`, `gate_proj+up_proj 9.38e-07` (both from a *device*
  leg; the license legs crossed nothing), and five shapes at "no crossing on this shape, nothing is
  exempt" — `BAND-SIZE` prints `cancel_rel_source=` so the difference is visible rather than implied by
  a number. These are ~35× tighter than `mlp_gemv`'s imported `3e-5`, which is what measuring buys.
* **A full-magnitude offender is a finding, not a cancellation.** If the worst >1-ULP element sits above
  `--fullmag-rel` (default `1e-3` of the tensor max) the probe prints `NUMERICS-FINDING` and writes no
  row. That did **not** fire: the two crossings in the census were `2.00 ULP at rel 1.96e-07, index 266`
  (`in_proj_qkv`, engine leg) and `6.00 ULP at rel 2.34e-07, index 7715` (`gate_proj+up_proj`, both the
  algo and the engine leg), **one element per leg** out of 49 152 and 28 672. The offender's index,
  relative magnitude and count are on the line (`worst_offender=ulp…@rel…_idx…`), because a crossing that
  cannot name its element is not attributable.
* A non-finite aggregate is refused (`reason=nonfinite_aggregate`) before it can be folded into a floor —
  NaN compares false against every bound.
* A `--time` run with no calibration, or one missing a `(shape, M, check_rows)` key, prints
  `CALIBRATION-NEEDED` with the exact probe command and exits 2 **before any HIP call**. A v1 file is
  refused by version, not by luck: it never measured a cancellation band, so any default for it would be
  a threshold nobody measured.
* The key binds `check_rows`, because cos is a **minimum over sampled rows**: a floor measured with 8
  rows is not the floor for a run that samples 16, and sharing one number across the two would be a
  silent tightening blamed on the kernel.
* `FLOOR-SUMMARY` requires `rows_written + refused + could_not_run == problems`, so a probe that measured
  nothing cannot report a clean sweep.
* Every injection must be caught by the predicate named for it, or the probe refuses with
  `reason=negative_control_escaped`. Two real bugs were caught this way on the first v2 run, both in the
  harness: a NaN row reporting `cos = 1.0` (because `na > 0` is false for NaN, so the row cosine fell
  back to its "no data" value), and a confluence test demanding bit-equality that flagged a `1e-10`
  spread as disagreement. The first was a fail-open.

And the two standing witnesses, printed on every judging run rather than only when they fire:

* **`CHECK-ENGINE`** — the engine's own call, judged exactly like a candidate, in `--check-only` too.
  A `REFUSE` now carries `attribution=engine_at_band_so_the_algorithm_is_the_suspect` or
  `attribution=engine_call_also_outside_band_so_band_or_reference_is_suspect`, so "is the band wrong?"
  is answered by the log instead of argued about.
* **`SIGNATURE`** — printed always. `verdict=constant_cos_reference_limited` when ≥3 launched candidates
  share one cos bit-for-bit while `ndiff` varies. On this build it fires on four of seven shapes, and
  confluence above explains why: the shared aggregate is the output format, not the algorithm. It is a
  diagnostic, never an automatic pass — the floor still only moves by way of a probe or a typed override.

## 6. Timing discipline

`steady_clock` only (AGENTS.md: `hipEventElapsedTime` returns exactly 0 below ~1 ms, and several of
these shapes are near it). Per shape: warm the pages once (the first GPU pass over fresh pages runs
~35 GB/s), warm up each algorithm `--warm` times, then `--reps` reps in which **every algorithm is
touched once per rep and the order rotates**, so a disturbance landing mid-run hits all of them
instead of permanently favouring whoever ran first — the paired-measurement rule
(the AGENTS.md trap row "a ruler re-taken in a different window is not an A/B"). Each sample
is `--inner` launches with one sync, divided, so `--inner 1` measures one call including its sync
floor (~10 µs). For the skinny shapes at M=512 (`k_proj`, `in_proj_z`) that floor is a real fraction
of the number: use `--inner 8` there and quote the pair, not the absolute ms.

`ratio` is **default-ranked ms ÷ best ms**, so 1.000 means the heuristic already picked the fastest
enumerated kernel. `engine_vs_best` is engine-call ms ÷ best ms — the same ratio against the call the
engine actually makes.

Not a worker mode, and does not self-lock. See `HANDOFF.md`.

## 7. Files

| file | what it is |
|---|---|
| `gemm_algo_sweep.cpp` | the tool (single TU; no kernels of its own — every matrix product here is the library's) |
| `build.sh` | `hipcc -O3 -Wl,-rpath,/opt/rocm/lib -lhipblaslt -lhipblas` |
| `make_selftest_fixtures.py` | writes `/tmp/fx_ok.txt` and `/tmp/fx_bad.txt`, the hand-authored `--selftest-sys` inputs: 4 legal algorithm records and 5 records carrying exactly one defect each. Hand-authored, **not** a device capture — the real input is `--verify --algo-dump`. It asserts the algo field is 48 hex chars (`sizeof(hipblasLtMatmulAlgo_t)` = 24 bytes), because a fixture whose length is wrong is rejected by the tool and proves nothing. Also writes `/tmp/cal_full.tsv` / `cal_partial.tsv` / `cal_badrow.tsv`, the calibration-file fixtures: complete, missing exactly one key, and carrying one row whose `cos_floor` parses to 0 — the three states the band resolver must distinguish |
| `cpu_check.sh` | 17 host-only invocations: the five pre-existing selftest/`--list` regressions plus every band and probe validation path. All of them return **before** any HIP call, so this whole script is runnable with no GPU window and no lock |
| `floor_calibration.tsv` | written by `--floor-probe`, read by `--time`: one `CAL` row per `(shape, M, check_rows)` carrying the measured worst cos, the engine's cos, the reference-reordering cos, the hardest injection cos, the ndiff spread, the margin and the resulting `cos_floor` + device name. Absent until the first probe; `--time` refuses to judge numerics without it |
| `README.md` | this file: shapes + capabilities + the stated band |
| `HANDOFF.md` | exact commands, lock protocol, failure patterns, the decision rule |
| `results.md` | skeleton; every number in it comes from a main-agent device window |
