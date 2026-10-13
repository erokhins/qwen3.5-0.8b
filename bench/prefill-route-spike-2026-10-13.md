# Prefill compute route spike — 2026-10-13

**What this is.** The spike `bench/plan-prefill-2026-10-12.md` §"Funded: prefill compute" asked for:
does the matrix-unit path work on `gfx1151` at all, is the vendor library a usable prefill backend, is
the bf16→fp16 conversion trustworthy for *this* checkpoint, and therefore which of the two funded routes
should be built. It answers those four questions and pre-registers the measurement list that decides the
rest.

**What it is not.** It is not a timing document. No TFLOP/s, no ms, no token rate was measured here, and
nothing in it may be copied into `bench/performance-status.md`. The dense-matrix rates it divides by
(bf16 **5.667**, fp16 **6.885**, fp32 **2.932** TFLOP/s at the engine's own `1024×4096×1024`, and the
2.206 GHz-of-2.9 sustained clock) are **quoted from `devices/strix-halo-compute.md`**, which measured
them in an exclusive window. Where this file does arithmetic on those quoted rates it says so. Every
device run behind it was correctness-only, under a minute, lock held, and reported `util` (§"How to
re-run").

---

## 1. Answer in one place

| question | answer |
|---|---|
| Do matrix units work on `gfx1151`? | **Yes.** `rocwmma::mma_sync` compiles to real `v_wmma_*` instructions for `gfx1151` — including a native bf16 one — and returns correct results against a CPU reference at three sizes. The public `gfx1151` breakage report does not reproduce at this size on this stack. |
| Is the vendor library usable as a prefill backend? | **Yes, with two carve-outs.** Determinism is clean at every engine shape; the epilogue bill is one line item, not a fatal one; the two `N=16` GDN projections and the one-row `lm_head` should stay out of it. |
| Is bf16→fp16 trustworthy for this checkpoint? | **Yes for the GEMM weights, and the reason matters:** the loss is entirely in fp16's subnormal band, and the largest absolute change any weight suffers is **3.0e-08**, about 1e-5 of a bf16 quantum at |w|≈1. It is **not** a licence to run activations in fp16 — that is a 24→11-bit mantissa cut and a separate decision. |
| Which route? | **Hybrid, library-first, and land it in bf16 first.** hipBLAS `GemmEx` for the ten fat shapes with the weight used as stored; hand kernels keep `in_proj_a/b`, one-row `lm_head`, and every epilogue. Hand-tiled WMMA stays an option, not a plan — see §6 and the gate in §7 W3. |

The single most useful fact for sequencing is in §4.3: **the library can write a 16-bit `C` at exactly the
rounding point today's pins use, in bf16, bit-for-bit.** So route (c) can be landed with *no* numerics
change at all, and the fp16 half of route (a) becomes a separable, separately-priced step instead of a
prerequisite.

---

## 2. Matrix units on `gfx1151`

Probe `tools/mfma_probe.cpp` (runtime) and `tools/mfma_isa.sh` (compile-only ISA evidence). Logs:
`logs/mfma_isa-2026-10-13.log`, `logs/mfma_probe-2026-10-13.log`.

### 2.1 What the compiler emitted

`hipcc -O3 --offload-arch=gfx1151`, target line `amdgcn-amd-amdhsa--gfx1151`, notes target
`amdgcn.target: amdgcn-amd-amdhsa--gfx1151`:

| kernel | instrs | `v_wmma` | `v_mfma` | `v_fma` | verdict |
|---|---|---|---|---|---|
| `wmma_gemm<fp16, f32 acc>` | 219 | 1 | 0 | 0 | MATRIX UNIT (`v_wmma_f32_16x16x16_f16`) |
| `wmma_gemm<bf16, f32 acc>` | 219 | 1 | 0 | 0 | MATRIX UNIT (`v_wmma_f32_16x16x16_bf16`) |
| `wmma_gemm<fp16, fp16 acc>` | 247 | 1 | 0 | 0 | MATRIX UNIT (`v_wmma_f16_16x16x16_f16`) |
| `fma_gemm<fp16, f32>` (control) | 113 | 0 | 0 | 1 | scalar/vector FMA only |

One `v_wmma` per kernel because each block computes exactly one 16×16×16 tile; the control kernel is in
the same translation unit so the table shows what a lowered-to-FMA kernel looks like next to one that was
not. Three things follow:

1. **`gfx1151` has WMMA and the compiler uses it.** Not lowered to scalar FMA — which is the failure mode
   a runtime PASS alone cannot exclude, and the reason `mfma_isa.sh` exists.
2. **bf16 has its own matrix instruction** (`v_wmma_f32_16x16x16_bf16`), not an emulation through fp16.
   This is the machine-readable version of why fp16's measured edge over bf16 (+21.5 % at the engine's
   shape, `devices/strix-halo-compute.md`) is a rate difference and not "bf16 has no hardware".
3. `v_mfma_*` never appears: RDNA 3/3.5 exposes `v_wmma_*` at the wave level. Anything written against
   CDNA `v_mfma_*` names is writing against the wrong ISA.

### 2.2 What the device returned

`C[m,n] = A[m,k] · W[n,k]^T`, weight row-major `[n,k]` used **as stored** (as WMMA's col-major B
operand, `ldm = k`), versus a double-precision CPU reference on the same 16-bit inputs:

| size | fp16 in / f32 acc | bf16 in / f32 acc | fp16 in / fp16 acc |
|---|---|---|---|
| 16×16×16 | PASS cos 1.000000000, meanrel 6.2e-08 | PASS cos 1.000000000, meanrel 1.4e-08 | PASS meanrel 3.5e-04 |
| 64×96×128 | PASS cos 1.000000000, meanrel 3.8e-07 | PASS cos 1.000000000, meanrel 6.3e-08 | PASS meanrel 1.0e-03 |
| 128×512×1024 | PASS cos 1.000000000, meanrel 3.2e-06 | PASS cos 1.000000000, meanrel 5.5e-07 | PASS meanrel 2.9e-03 |

All aggregates, per the §10.1 lesson that max-ULP and maxrel are blind to a long dot's cancellation.
The 128×512×1024 case is a `k_proj`-sized product with an engine-shaped `K`. Correctness only: 0.87 s
wall, `util` 90 % at acquire (contended, which cannot corrupt a deterministic comparison), **no timing
taken and no rate printed by the probe at all.**

Reading:

- **The weights-as-stored layout is right**, and it is the layout both routes want: no transpose copy of
  1.5 GB of weights anywhere. §5.2 shows the same fact for hipBLAS by computation.
- **fp32 accumulation is where the accuracy lives**: f32-accumulate outputs sit at `meanrel` 1e-8…3e-6.
- **The fp16 accumulator works but is not a candidate**: 3e-4…2.9e-3 relative against its *own*
  fp16-accumulated reference, and worse on cancelled elements (maxrel 4.7e+01 at `K=1024`). D-3 pins F32
  accumulation anyway; the line exists to show the instruction is there and what it costs.

### 2.3 The limits of this evidence, stated before anyone extends them

A one-tile-per-warp kernel with no blocking, no shared-memory staging, no K-pipelining and no skirting. It
proves the instruction path, the operand layout, and that the reported `gfx1151` WMMA breakage
([rocm-libraries 4618](https://github.com/ROCm/rocm-libraries/issues/4618), cited in the plan) does not
reproduce for a minimal `mma_sync` on ROCm 7.15 / ROCwmma 2.2.1. It says **nothing** about whether a
hand-tiled kernel can reach the vendor's rate, nothing about Tensile's kernel selection on this part, and
nothing about throughput. §7 W3 is the gate that would answer the rate question, and it is deliberately
*not* on the critical path.

Three traps found while writing the probe, because they are the same traps the repo already documents:
reading a 2-byte accumulator into a `float` buffer is a bitcast-with-wrong-stride and produced a false
FAIL of magnitude 5e2 on a kernel that was fine (`tools/mfma_probe.cpp`, in the comment where the fix
lives); judging an fp16 accumulator against an f32 reference measures accumulation noise with the wrong
yardstick; and `--save-temps=<dir>` writes into the CWD anyway on this ROCm, so `mfma_isa.sh` runs in a
scratch dir.

---

## 3. The vendor library as a prefill backend

All of §3 is `tools/gemm_peak.cpp` — the file `devices/strix-halo-compute.md` was measured with, now with
two correctness modes (`--check`, `--determinism`) that print no timing at all. Logs:
`logs/gemm_peak_check-2026-10-13.log`, `logs/gemm_peak_det-2026-10-13.log`.

### 3.1 (i) Determinism: clean at every shape this model issues

Same product, three calls on one handle, then a second handle given the **same** 128 MiB workspace, then
a third given **none**, then a check that the `beta==0` call erased a 0xFF filler that no write could
produce by accident. Output hashed with FNV-1a over the whole buffer.

Shapes are the engine's own set at wave rows 64/512/1024/3072, in **both** layouts (`W[k,n]` pre-transposed
and `W[n,k]` as the checkpoint stores it), for bf16 and fp16 in / f32 accumulate / f32 C: 44 rows.

- 3 repeats on one handle: byte-identical everywhere.
- Second handle, same workspace: byte-identical everywhere — so the answer is not "kernel choice depends
  on handle history" at these shapes.
- Second handle, **no** workspace: byte-identical everywhere here too. Not a general result: workspace
  size does feed split-K eligibility, so a route that pins it in one place and not another is not owed
  this outcome. The probe says *pin the workspace anyway*.
- `beta==0` overwrote the whole buffer at every row (no 0xFF survived), so a stale output cannot masquerade
  as a hit.
- Across **three separate processes**: `DET-COMBINED bf16 b30c25c0e7fd147a`, `fp16 48ad8cbb1e7ae419`,
  identical in all three.
- The hashes are sensitive in the ways they should be: the two layouts of the same shape hash differently,
  and `lm_head@1` is the one row where the bf16 and fp16 runs disagree — which is informative rather than
  worrying, because the fill values are multiples of 1/256 and therefore exactly representable in both
  widths, so the two dtypes are multiplying the same numbers and a hash difference there can only be a
  different kernel being chosen.

**No split-K nondeterminism was found.** There is no atomic-reduction smearing output bytes at these
shapes. Caveats, in the interest not hedging: one algorithm (`HIPBLAS_GEMM_DEFAULT`), one GPU, f32 `C`;
hipBLASLt's algorithm search was not exercised, and an explicit algo choice would need the same question
asked again.

### 3.2 (ii) Shape coverage — every matmul a wave performs, and what it is bound by

`tools/prefill_gemm_shapes.py` reads the shapes from the shard header via `engine/weights_contract.py`,
the layer split from `config.json`, and the constants from `engine/state.h`. Full output:
`logs/prefill_gemm_shapes-2026-10-13.log`. Two corrections come out of it, and both matter.

**Correction 1 — the brief's "skinny `k/v_proj`, N=256" is wrong, and the real skinny pair is worse.**
`k_proj` and `v_proj` are `[512,1024]`: `kKvProjRows = kNumKvHeads × kHeadDim = 2 × 256`
(`engine/state.h`). The genuinely skinny matmuls are `linear_attn.in_proj_a` and `in_proj_b` at
**`N=16`**, 18 layers each — **36 library calls per wave for 0.6 GFLOP** at `M=512`, at **0.065
bytes/flop against a ridge of 0.035**. They sit below the ridge at every wave width: each call reads
~1.05 MiB of shared activation to produce 16 KiB at `M=512`, and at `M=64` its whole ~162 KiB of traffic
is ~0.7 µs of stream time against a 1.85 µs launch floor, so the launch costs more than the shape does. A
square probe says nothing about them; a library route must either fuse `a|b` into one `N=32` call or leave
them in the hand kernel.

**Correction 2 — the flop divisor for prefill is ~1.0 GFLOP/token, not 1.6.** The per-token block matmul
set is **0.9952 GFLOP/token** (186 matmuls). The ~1.5–1.6 GFLOP/token figure that every ceiling in
`devices/strix-halo-compute.md` and the plan divides by includes the full `lm_head` per token
(0.5086 GFLOP for one priced row); prefill prices **exactly one** logits row per request (§06.5), so at a
1024-token prompt that row contributes 0.0005 GFLOP/token. Consequences, arithmetic only, on the quoted
6.885 fp16 rate:

| | with the 1.6 divisor | with this one |
|---|---|---|
| pure-matmul ceiling at the engine's own shape | ~3,700 tok/s | **~6,900 tok/s** |
| what 971 tok/s implies about today's block-matmul rate | 1.55 TFLOP/s | **0.97 TFLOP/s** |
| sustained rate a 5,000 prompt tok/s target needs | ~8 TFLOP/s (above the hardware) | **~5.0 TFLOP/s** |

That does not make 5k achievable — the non-matmul half of a wave is not free, and the plan's conclusion
that the defensible band is ~2,000–2,600 still stands as a *delivery* target — but it does retire the
"5k needs more TFLOP/s than this GPU can produce in total" argument. The gap to attack is that the engine
currently converts ~0.97 of the ~6.9 available, and the pre-registered thresholds below are stated in
these units so that a future ledger row cannot quietly switch divisors.

**The table.** Ridge = quoted rate ÷ measured 240 GB/s = 28.7 flops/byte; byte columns are the *minimum*
traffic (each operand touched once), so a badly blocked kernel can only be worse.

| matmul | ×/layer | N | K | bytes/flop @M=512 | bound @M=512 | bytes/flop @M=1024 | bound |
|---|---|---|---|---|---|---|---|
| `q_proj` | 6 | 4096 | 1024 | 0.0032 | compute | 0.0022 | compute |
| `k_proj`, `v_proj` | 6 each | 512 | 1024 | 0.0049 | compute | 0.0039 | compute |
| `o_proj` | 6 | 1024 | 2048 | 0.0034 | compute | 0.0024 | compute |
| `in_proj_qkv` | 18 | 6144 | 1024 | 0.0031 | compute | 0.0021 | compute |
| `in_proj_z` | 18 | 2048 | 1024 | 0.0034 | compute | 0.0024 | compute |
| **`in_proj_a`, `in_proj_b`** | 18 each | **16** | 1024 | **0.0654** | **BANDWIDTH** | **0.0645** | **BANDWIDTH** |
| `out_proj` | 18 | 1024 | 2048 | 0.0034 | compute | 0.0024 | compute |
| `gate_proj`, `up_proj` | 24 each | 3584 | 1024 | 0.0032 | compute | 0.0022 | compute |
| `down_proj` | 24 | 1024 | 3584 | 0.0032 | compute | 0.0022 | compute |
| `lm_head` (tied) | 1 | 248320 | 1024 | 1.001 | BANDWIDTH (one row) | — | — |
| **wave total** | 186 | | | 0.0043 (232 f/B) | compute | 0.0028 (353 f/B) | compute |

Wave totals (flops; weight bytes; activation r+w bytes): `M=64`: 64.2 GFLOP, 1.40 GiB, 83 MiB — weights
are **95 %** of everything touched. `M=512`: 510 GFLOP, 1.40 GiB, 661 MiB — 68 %. `M=1024`: 1020 GFLOP,
1.40 GiB, 1.29 GiB — 52 %. `M=3072`: 3058 GFLOP, 1.40 GiB, 3.87 GiB — weights down to **27 %**. So one
pass over the weights per wave is the whole game at small `M`, and `M=64` is already compute-bound by
weight traffic alone — which is why the plan's "raise the wave cap above 512" (`kMaxWaveRows = 512` today,
`engine/state.h`) helps per-flop and per-weight-pass at the same time.

For orientation only, arithmetic on the quoted fp16 rate and nowhere near a measurement: 74 ms of matmul
for a 512-row wave, 148 ms for a 1024-row wave.

### 3.3 (iii) Epilogues: where rope, `silu`×`up`, gating and the residual add would live

Today, and what a library call changes (`logs/prefill_gemm_shapes-2026-10-13.log` has the table):

| epilogue | lives today | extra bytes per wave if the matmul is a call |
|---|---|---|
| `silu(gate)·up` | **fused inside `mlp_gateup`** | 42 MiB @M=64 · 336 MiB @M=512 · **672 MiB @M=1024** · 1.97 GiB @M=3072 |
| attention gate de-interleave | `attn_prep`, already a separate pass reading the projection output | 0 |
| q/k head-norm + rope | `attn_prep` | 0 |
| GDN gated RMSNorm (§04.5) | own kernel already | 0 |
| residual add + next RMSNorm | `glue_add_norm`, already a pass | 0 |

Only the MLP is actually unfused, because only the MLP is fused today. `mlp_gateup` writes `act` once; a
library gate+up writes both halves, and a `silu`-mul kernel reads both and writes `act` — 5 tiles of
`M×3584×2 B` per layer instead of 1. Two notes that keep this in proportion:

- Concatenating the weights into one `[7168,1024]` gate|up merges two calls into one but does **not**
  change that byte count. It buys launches, not traffic.
- The extra traffic scales with `M`, exactly as the MLP's own flops do, so it is **≈3.7 %** of the MLP's
  matmul time at any wave width (arithmetic on the quoted fp16 rate and the measured 240 GB/s: 28,672 B of
  extra traffic per layer per row against 22,020,096 flops per layer per row), and while the MLP is
  compute-bound part of it hides behind compute. It is a line item, not a veto. If the library's own rate
  turns out to be mediocre the ratio worsens — which is why W6 prices it instead of assuming it.

What the library route gives back: `proj_gemv` runs `PACK_X=false` and therefore needs **F32** activations,
which is most of why 115 cast launches per token exist at all (`engine/kernels_proj.cpp`, `engine/results.md`
E2). A library call taking a 16-bit `A` operand reads what `glue` wrote, so those casts go away for the
projections — a launch/dispatch saving at the 1.85 µs measured floor, not a traffic one. And the
anti-cost, which §4.3 turns out not to cost anything: the call must write a **16-bit `C`**, because every
shipped pin rounds a projection's product to a 16-bit width where the reference does. An F32 `C` would
both double the activation traffic above and move a rounding, and a moved rounding is a route change
(D-3/D-5), not a tuning knob.

---

## 4. The fp16 range audit

`tools/fp16_range_audit.py`, over every tensor in the checkpoint (488) plus the pinned activation fixture
`data/refs/p300` (484 arrays after carrying two non-activation families). Full output:
`logs/fp16_range_audit-2026-10-13.log`. `--selftest` passes, and it is the part that makes the rest
trustworthy: it proves the cast under audit rounds half-to-even and does not flush subnormals, because a
converter that did either would make the subnormal and inexact counts read low and the verdict would pass
vacuously. (`logs/*.log` is gitignored, so the named logs are local run artifacts; §8 regenerates any of
them in one line.)

### 4.1 The conversion itself, proved by exhaustion

All 65 536 bf16 bit patterns through the audited cast:

```
exact 8960 / 65536   inexact 56576
of the inexact set: 28672 overflow to Inf, 1792 inside the subnormal band,
                    0 in fp16's normal range, 26112 flushed to zero
finite patterns > 65504: 28672; the cast sends 28672 of them to Inf
```

**`0 in fp16's normal range` is the claim the plan asked to be verified rather than assumed**: every bf16
value with an exponent inside fp16's normal range converts exactly — bf16 has 7 mantissa bits, fp16 has
10, and the exponent range is a superset there. The check is by census, not by argument. The entire risk
surface of the conversion is therefore two things: the subnormal band below 2^-14, and the ceiling at
65504.

### 4.2 What this checkpoint actually contains

| bucket | tensors | values | max abs | → Inf | inexact in normal range | rounded in subnormal band | flushed to zero | largest absolute change any value suffers |
|---|---|---|---|---|---|---|---|---|
| **text-gemm** (every GEMM weight, incl. tied embed) | 187 | 751,894,528 | **0.582031** | **0** | **0** | 251,008 | 15,889 | **2.98e-08** |
| text-aux (norms, `A_log`, `dt_bias`, `conv1d`) | 133 | 498,496 | 13.0625 | 0 | 1001 — **all from F32 sources** | 471 | 59 | 4.88e-04 |
| mtp | 15 | 20,452,864 | 6.125 | 0 | 0 | 9,536 | 56 | 2.98e-08 |
| visual (not used by this engine) | 153 | 100,592,896 | 23.625 | 0 | 0 | 30,600 | 146 | 2.98e-08 |
| activations, `data/refs/p300` | 484 | 252,910,720 | **82.5** | **0** | 4,237,738 — all F32 sources | 1,504,085 | 19,201 | 3.4e-03 |

No NaN and no Inf anywhere in the checkpoint or the fixture. `zero` counts and −0 counts are in the log.
The activation row is the `p300` prefill fixture; the same audit over `wide520f` and `chunk64b` puts the
activation max at 37.25 (`decode*.L15.attn.v.npy`) instead of 82.5, and nowhere higher — so the range
headroom on the activation side is ~800× on the widest sample looked at, not a one-fixture accident.

**Verdict, in the words of the question.** bf16→fp16 is **not** bit-lossless, and it does not need to be:
0.036 % of the GEMM weights change, all of them inside fp16's subnormal band, and **the largest absolute
change any weight suffers is 2.98e-08**. A bf16 weight near |w|=1 has a quantum of ~4e-3, so the
conversion error is ~5 orders of magnitude below the weight's own quantum and cannot move a rounded
result. Range is not a risk at all: max |weight| is 0.582, **1.1e5× below 65504**, so nothing overflows
and nothing needs a scale factor. `text-gemm` is the route verdict because it is what a prefill GEMM
multiplies by.

What has to be handled, precisely:

1. **F32 tensors are a different question and must not be swept in.** `input_layernorm`,
   `linear_attn.norm`, `A_log`, `dt_bias` are F32 in the shard (§01.6), and 24 mantissa bits do not fit in
   11 — the 1001 normal-range inexact values in `text-aux` are that width change, nothing more. None of
   them is a GEMM operand. A conversion pass must be scoped to the rank-2 BF16 tensors, which is exactly
   how `group_of` buckets them.
2. **fp16 for the activations/residual stream is not what this audit licenses.** The engine carries F32
   activations today (`PACK_X=false`). Range is fine — max |activation| in the fixture is 82.5, 794× below
   65504, top values in `L04.in_proj_qkv.out` / `gdn.mixed_qkv_pre_conv` — but F32→fp16 is a 24→11-bit cut
   whose measured absolute damage is 3.4e-03, three orders of magnitude above the weight-side number.
   That is a route change needing its own band and margins, not a conversion.
3. **Keep F32 accumulation.** The fp16-accumulator line in §2.2 is the cost of not doing so.
4. **The two non-activation families in a fixture directory are carried, not audited**: `*_fp.npy` FNV
   fingerprint arrays and the float32-token-id `input_ids` arrays. On the first run they produced
   `absmax=248069` and nine phantom overflows; the filter is now explicit and named in the tool, and the
   activation max is 82.5, not 248069.

### 4.3 The rounding-point fact that changes the sequencing

`tools/build/gemm_peak --check`, `logs/gemm_peak_check-2026-10-13.log`:

```
CDTYPE MATRIX (C[m,n] = A[m,k] W[n,k]^T, W as [n,k], OP_T; 64x96x128)
  A/B    C      status     bits differ  worst |device - host-rounded| in ULP of the 16-bit format
  bf16   bf16   accepted   0            0
  bf16   f32    accepted   n/a          F32 output has no 16-bit rounding to compare
  fp16   fp16   accepted   12           1
  fp16   f32    accepted   n/a          F32 output has no 16-bit rounding to compare
```

hipBLAS will write a **bf16 `C` bit-for-bit at the same place the engine rounds** (0 of 6144 differ), and an
**fp16 `C`** within one ULP (12 of 6144 differ, from accumulation noise crossing a rounding boundary, not
from a different rounding rule). So neither route is forced into an F32 output plus a cast pass. The first
draft of this check asked for C=fp16 with A/B=bf16 and got status 7, which is *that pair* being
unsupported, not fp16 `C` being unavailable — the axis had to be walked the way the routes use it.

Consequence: **route (c) can be landed in bf16 with no numerics change and no band re-sizing**, and fp16
becomes a second, separately-priced step that shares the same plumbing.

---

## 5. Layout: the weight as stored, verified by computation

Both routes want to multiply by the shard's `[n,k]` weight without a transpose copy.

- WMMA: `[n,k]` row-major read as a col-major B operand with `ldm=k` — verified in §2.2 at three sizes.
- hipBLAS: `hipblasGemmEx(h, OP_T, OP_N, n, m, k, α, W, dt, k, A, dt, k, β, C, dt_out, n, COMPUTE_32F,
  DEFAULT)` — derived in the comment at `gemm_stored_weight`, and checked: `LAYOUT CROSS-CHECK ok: OP_T(W
  as [n,k], ld=k) against the same CPU reference at max|delta| 0.000e+00; 0 of 6144 outputs differ in their
  bits from the OP_N result`.

Getting this wrong is silent — it computes a different product, not an error. Two live examples from
writing the check: an `OP_N` call with `ldb=k` on a `[n,k]` buffer is an index swap, not "the same product"
(an early draft contained it and would have failed its own reference for a reason of its own inventing),
and a use-after-free of the activation buffer made the OP_T result look wrong by 4.2e+01 until the free
was moved. Both comments are in the file.

---

## 6. Route recommendation

**Hybrid. Vendor library for the fat shapes, hand kernels for everything else, landed in bf16 first.**

1. **Library** (hipBLAS `GemmEx`, one handle created at init, workspace pinned via `hipblasSetWorkspace`,
   `OP_T` with the weight as stored, 16-bit `C`): `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`,
   `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`. Ten shapes, **150 of the wave's 186**
   matmul calls, and 99.9 % of the wave's matmul flops between them.
2. **Hand kernels keep**: `in_proj_a`/`in_proj_b` (`N=16`, below the ridge at every width; at `M=64` a
   call's whole ~162 KiB of traffic is ~0.7 µs of stream time against a 1.85 µs launch floor, so the call
   is dispatch-bound before anything else gets a vote; if they go to the library at all, fuse them into
   one `N=32` call), `lm_head` at one row (a GEMV already at **102 % of the stream roofline**, 2.105 ms
   for 485 MB, `bench/performance-status.md`: there is nothing left for a library to win there), every
   mixer and glue kernel, and the `silu`-mul if W6 says the unfuse earns its keep.
2b. **The in-situ spread is the best argument for the split.** Today's hand projections run at **66 % of
   the probe**, and within them from **77 %** (`in_proj_qkv`) down to **39 %** (`v_proj`) and **5 %** for
   the 16-row `in_proj_a`/`in_proj_b` (`bench/performance-status.md`, "the wide shapes behave like
   `lm_head`; the narrow ones behave like nothing measured before"). The library's upside is concentrated
   in exactly the shapes the hand kernel is worst at — and it is zero for the ones already at roofline.
3. **bf16 first, fp16 second.** Land the library route with bf16 in / bf16 `C` — no numerics change at
   all, per §4.3 — then re-measure the same path in fp16 and let that measurement, not the +21.5 % on
   record, decide whether to pay the band re-sizing. The audit says fp16 is safe on the weight side and
   fp16 also has *more* mantissa bits than bf16, so the case is good; what it is not is free, since every
   band in the ledger is sized against bf16 pins.
4. **Hand-tiled WMMA: keep as an option, do not fund now.** The instruction path and the layout are proven
   (§2), so the door is open and the public breakage report does not reproduce here. But the vendor's
   kernels already reach 5.667/6.885 TFLOP/s at this exact shape on this exact box, and the engine today
   converts about 0.97 of that. The first factor of 4–6 is available by calling the library well; a
   hand-tiled kernel competes for the remainder, on 20 CUs, across 13 shapes, with its own tuning and its
   own gates. §7 W3 is the pre-registered gate that would change this mind: if hipBLAS at our shapes comes
   back in the dead band, hand-tiling becomes the only way up.

**Gate churn the implementation inherits** (not decisions, consequences to plan for): a library GEMM
reassociates the dot, so token-level bit-identity against the pinned dumps is gone for the projections —
per A-18 the chaos floor already makes long-generation token equality a non-gate, and per §10.1 the legal
ndiff for a chained dot is 29–47 %, so ndiff is the wrong metric there; the gate metric for a library
route is cos/meanrel aggregates plus margins sized by a floor probe against the dumps, as §10.1 does for
the BF16 subpartials. `engine/route_allowance.py`'s allowances need re-deriving for the ten affected
matmuls, and the fp16 step needs them again.

---

## 7. Pre-registered measurement list

Thresholds are fixed **before** the window, in the units defined in §3.2 (block-matmul flops = 0.9952
GFLOP/token; `lm_head` priced once per request). Every run: main agent only, blocking, zero subagents,
best-of-N with the spread printed, `tools/gpu_lock.sh util` at acquire, **and the sustained clock recorded
with the row** — the recorded window sustained 2.206 GHz of 2.9, i.e. 76 %, and every one of these rates
is clock-bound. A result inside a wash band means the route is not funded further; that is what
pre-registering is for.

| # | measure | shapes / dtype / route | "real" | "wash" | "dead" |
|---|---|---|---|---|---|
| **W1** | hipBLAS rate at the engine's own shape mix, **bf16, no numerics change** | the ten library shapes of §6 item 1, at `M=64/512/1024/3072`; bf16 in, f32 compute, **bf16 C**; `OP_T` weights-as-stored; one handle, 128 MiB workspace; aggregate flops ÷ wall over one layer's shape mix at `M=1024` | ≥ **3.4** TFLOP/s (= 60 % of the recorded 5.667 at this shape) | 2.3–3.4 | < **2.3** |
| **W2** | same runs in **fp16** (route (a) as a delta on W1, not a separate route) | identical, fp16 in / fp16 `C` | aggregate gain over W1 ≥ **+10 %**, and engine TTFT improves ≥ 5 % after the band work | +3…+10 % | < +3 % |
| **W3** | hand-tiled WMMA prototype, only if W1 lands wash-or-dead | `q_proj` 1024×4096×1024 and `gate` 1024×3584×1024, fp16 in / f32 acc, one production-quality tiled kernel | ≥ **85 %** of hipBLAS at the same shape and dtype | 60–85 % | < 60 % → hand-tiling is not fundable |
| **W4** | the skinny pair and the other two small-`N` shapes | `in_proj_a`/`in_proj_b` (16×1024), `k_proj`/`v_proj` (512×1024), `M=64/512/1024` | library per-call time ≤ **2×** the 1.85 µs dispatch floor | 2–4× | > 4× → stays in the hand kernel (or fuses `a|b` into `N=32`) |
| **W5** | call-count and cast accounting | launches per wave before/after: 186 matmul calls minus the 115 F32-widening casts per token that go away | net launch count per token **falls** | ±5 % | rises |
| **W6** | the `silu` unfuse bill | `silu`-mul kernel alone at `M=512/1024` (bytes known: 4 tiles × 24 layers), against the fused `mlp_gateup` | added wall ≤ **5 %** of the wave's matmul wall | 5–12 % | > 12 % → keep MLP in the hand kernel |
| **W7** | clock accounting | `pp_dpm_sclk` sampled during every W1/W2 window, same protocol as `devices/strix-halo-compute.md` | — every quoted row states its clock — | — | a row without a clock is not a row |
| **W8** | engine-level determinism of the library route | run the same 300-token prefill twice with the route enabled; compare `logits` bytes and the greedy continuation | byte-identical | — | any difference → the workspace/handle is not actually pinned; §3.1's shape-level PASS does not cover the engine's call order |
| **W9** | the end-to-end number the route exists for | engine TTFT at a 1024-token prompt, route enabled, `--wave-rows` at `kMaxWaveRows`, then with the cap raised to 1024 | ≤ **400 ms** (2,560 tok/s ≈ 2.6× today) | 400–800 ms | > 800 ms |

W9 is the decision; W1–W8 are its diagnostics. The two must be reported together, because W1 in the real
band with W9 in the wash band means the matmul was never the binding constraint, and the next thing to
price is then the non-matmul half of the wave.

---

## 8. How to re-run everything

```bash
# CPU / compile only — no device, no lock
.venv/bin/python tools/fp16_range_audit.py --selftest
.venv/bin/python tools/fp16_range_audit.py --refs data/refs/p300 \
    > logs/fp16_range_audit-2026-10-13.log
.venv/bin/python tools/prefill_gemm_shapes.py > logs/prefill_gemm_shapes-2026-10-13.log
tools/mfma_isa.sh tools/mfma_probe.cpp > logs/mfma_isa-2026-10-13.log   # compile only

# device — correctness only, each under a second; one at a time, lock held, util recorded
tools/gpu_lock.sh acquire prefill-spike && tools/gpu_lock.sh util
tools/build.sh gemm_peak mfma_probe
tools/build/gemm_peak  --check                       > logs/gemm_peak_check-2026-10-13.log
tools/build/gemm_peak  --determinism --dtypes bf16,fp16 > logs/gemm_peak_det-2026-10-13.log
tools/build/mfma_probe                               > logs/mfma_probe-2026-10-13.log
tools/gpu_lock.sh release prefill-spike
```

`--check` and `--determinism` print no timing by construction, so a worker can run them (AGENTS.md):
`--check` is under a second, `--determinism` about eight. Nothing in this file is a measurement of speed;
the timing windows are §7's, and they belong to the main agent.

## §7 results — W1, W2, W4 (main agent, window `prefill-Wclean`, blocking, zero subagents, clock 1857–2401 MHz)

Everything below is **16-bit `C`**, and per the correction in `devices/strix-halo-compute.md` none of it is
comparable with the `gemm_peak` rows in §4.3 or §7's own band arithmetic. `--check` and `--mix-determinism`
were re-certified in this same window before any timing was taken (PASS both dtypes; recorded hashes
reproduced exactly).

| M | W1 bf16 mix, TFLOP/s (best / med / worst, spread) | W2 fp16 mix | fp16 vs bf16 |
|---|---|---|---|
| 64 | 4.80 / 4.75 / 4.63 (3.6 %) | 4.90 / 4.68 / 4.42 | **+2.2 %** |
| 512 | 18.84 / 18.56 / 18.41 (2.3 %) | 19.06 / 18.39 / 18.23 | **+1.2 %** |
| 1024 | 22.31 / 21.82 / 19.15 (16.5 %) | 17.82 / 12.90 / 11.62 (53.3 %) | **−20.1 %** |
| 3072 | 23.26 / 23.01 / 19.51 (19.2 %) | 24.12 / 24.05 / 21.25 | **+3.7 %** |

**W1 — passes its letter, and its basis is void.** The band was "≥ 3.4 TFLOP/s real, 2.3–3.4 wash, < 2.3 dead",
written as 60 % of the then-recorded 5.667 TFLOP/s. Measured on the mix at `M=1024` it is **22.3**, which clears
the "real" bar 6.6×. That is a **unit mismatch, not a verdict** — the anchor was an F32-`C` rate — so I am
recording W1 as *satisfied and re-anchored*: the route is real, and the decision now belongs entirely to **W9**
(engine TTFT ≤ 400 ms), which was always the row the rest were diagnostics for. I am not retro-fitting a new
percentage bar after the fact; the honest statement is that a threshold anchored on a crippled call form cannot
grade a fast one.

**W2 — dead, per the pre-registered bands.** The bar was "aggregate gain ≥ +10 % and ≥ 5 % engine TTFT after
the band work; +3…+10 % wash; < +3 % dead". Measured: **+2.2 / +1.2 / −20.1 / +3.7 %** across the four `M`
values — never at or above the +3 % floor except once, and *negative* at `M=1024`, where the fp16 mix also
carried a 53 % spread against bf16's 16.5 %. So **the fp16 half of route (a) is not funded further**, and it
takes its premise with it: the "+21–26 % for fp16" that motivated it was itself an F32-`C` artifact. The
fp16 *audit* (§4) still stands as a fact about the checkpoint — it is simply no longer buying anything, so the
band re-sizing it implied is off the table. Route (a) collapses into route (c): **get the library path in, in
bf16.**

**W4 — dead for the skinny shapes, which confirms §6's carve-out rather than testing it.** Against the
1.85 µs dispatch floor, burst times per call: `in_proj_a`/`in_proj_b` (N=16) **22.2–23.6 µs** = 12–13× the
floor, the fused `a|b` (N=32) **22.8–24.4 µs** — no better than the pair, so the fusion §6 floated buys nothing
at library rates — and `k_proj`/`v_proj` (N=512) **28.6–40.2 µs** = 15–22×. The band was ≤2× real, 2–4× wash,
>4× dead: all five are **dead**. The small shapes stay in hand kernels, and it is worth naming why: at `N=16`
the whole call is ~162 KiB of traffic, so a library kernel's tile machinery never pays for itself and the call
is dispatch- and launch-bound before arithmetic enters. One caveat kept on purpose: each-sync `max` values on
these rows run to 165–55,828 µs, which is scheduling noise on a busy box, and no verdict here reads them.

**Net route after measurement.** hipBLAS for the ten fat shapes in **bf16 with a bf16 `C`** (W1 real, §6's list),
hand kernels for `in_proj_a/b`, `k/v_proj` and one-row `lm_head` (W4 dead, §6's carve-out upheld), **no fp16**
(W2 dead), and **W9 as the only remaining gate that can fund or kill this**. W3 (hand-tiled WMMA) is not
triggered — it was conditional on W1 landing wash-or-dead, and W1 did not.
