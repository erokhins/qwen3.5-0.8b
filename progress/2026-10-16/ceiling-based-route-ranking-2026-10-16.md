# Ceiling-based route ranking — prefill, rebuilt on both of D16's facts

CPU-side analysis, 2026-10-16. **No device work, no timings taken here, nothing committed.** Every
number is quoted from an on-disk measurement or is arithmetic on quoted numbers, with the file and
section named at the point of use. Where a number needed by the argument does not exist on disk it is
written as `NOT MEASURED: <what>`. Companion documents: `prefill-decision-log-2026-10-16.md`
(D13–D16), `prefill-occupancy-results-2026-10-16.md` (family attribution),
`bench/performance-status.md` (the ruler), `prototype/issue_rate_probe/results.md` (the ceilings).

The deliverable is §5 (the re-ranked route table) and §6 (the one route to fund, and the cheap
measurement that would falsify the ranking). §1 states the correction to fact B that forced the
re-ranking to be done on arithmetic rather than on the decision log's headline; §4 states what I
assumed about which of A and B binds where, which is the load-bearing judgement.

---

## 1. The two facts, and the one number in fact B that does not survive its own provenance check

**Fact A (D16 item 3) — stands, and I re-derived it independently.** `attn` issues 2,984 M VALU
warp-instructions for a 1024-lane wave (`prefill-occupancy-results-2026-10-16.md` §3). Against
2 issue slots × 40 CU × 2.894 GHz (rocminfo: SIMDs per CU = 2, `strix-halo-official-specs.md` §4):
`2,984e6 / (40 × 2 × 2.894e9) = 12.9 ms`, against 27.23 ms measured → **47.3 %**, matching D16's ~47 %.

**Fact B (D16 items 1–2) — the instruction ceilings stand; the baseline they are compared against
does not.** The probe's headline is "WMMA 52.63 TFLOP/s = **11.2× the engine's 4.696 TFLOP/s through
hipBLASLt**" (`prototype/issue_rate_probe/results.md` §2). Two problems with 4.696 as *the engine's
rate*:

1. **4.696 is a retired number.** `devices/strix-halo-compute.md`, §"Second correction … it is 6–8×
   below the same call", retires "dense bf16 runs at 4.7–6.0 TFLOP/s" outright: those rows came from
   `tools/gemm_peak.cpp` hard-coding `HIPBLAS_R_32F` for `C`. With a 16-bit `C` — which is what the
   engine actually asks for — the same calls sustain **33.1–33.7 TFLOP/s** at the engine's own shapes,
   and the 24-layer wave mix sustains **18.84 / 22.31 / 23.26 TFLOP/s at M = 512 / 1024 / 3072**.
2. **The engine is verifiably on the 16-bit-`C` path.** `engine/kernels_gemm.cpp:171-172` calls
   `gemm_stored_weight(..., HIP_R_16BF, ..., HIP_R_16BF)` and `kernels_gemm.cpp:101` describes the
   route as `dtype=bf16-in/bf16-C/f32-compute`. All of `proj`, `mlp_gateup`, `mlp_down` go through that
   one function (`kernels_mlp.cpp:262-263,276`, `kernels_proj.cpp:146`).

So the comparison that matters is not WMMA against 4.696 but WMMA against **what the engine's matmul
families actually deliver in situ**, which is computable from the wave attribution. FLOP counts come
from the checkpoint shapes in `engine/state.h:41-56` (d=1024, d_ff=3584, vocab 248,320, 18 GDN +
6 attention layers, q 4096 rows / kv 512 rows / o_proj 2048→1024, in_proj_qkv 6144 rows, in_proj_z
2048, in_proj_a/b 16, out_proj 2048→1024):

| quantity | derivation | value |
|---|---|---|
| proj FLOP per token | 2 × Σ(in×out) over all 114 matrices = 2 × 233,373,696 | **0.4667 GFLOP/token**; weights 0.4667 GB — reproduces the frozen ledger's 0.4677 GB/token, `progress/2026-10-16/performance-status-2026-10-16.md`:79-90 |
| mlp FLOP per token | 2 × 24 × 3 × 1024 × 3584 | **0.5285 GFLOP/token**; weights 0.5285 GB — same check ✓ |
| wave matmul FLOP at 1024 lanes | (0.4667 + 0.5285) × 1024 | **1,019.1 GFLOP** — independent agreement: `devices/strix-halo-compute.md` quotes "a 1024-token wave's block matmuls are 1017.9 GFLOP" |
| engine in-situ matmul rate, 1024 / 2048 / 4096 | 1019.1/72.55, 2038.2/144.08, 4076.4/283.17 (`GROUP matmul_ms`, window1 + window4) | **14.05 / 14.14 / 14.39 TFLOP/s** |

**Consequence, and it is the first surprise: the shipped engine's matmul families are at ~14 TFLOP/s,
not 4.696.** The honest multiple for WMMA over the engine's matmul path is **3.7×** (52.63/14.05), and
over the library's own same-shape rate it is **1.6×** (52.63/33.6). D16 item 1's 11.2× is an artifact
of comparing a measured matrix leg against a retired F32-epilogue number, and the route index should
stop citing 4.696.

**And a harder consequence that I cannot resolve on paper.** Per-family, the same attribution implies:

| family | FLOP at 1024 lanes | measured (window1) | achieved rate | against the 1-MAC vector roof 14.82 |
|---|---|---|---|---|
| `proj` | 477.9 GFLOP | 39.7239 ms | **12.03 TFLOP/s** | 81 % |
| `mlp_gateup` | 360.8 GFLOP | 26.5184 ms | **13.61 TFLOP/s** | 92 % |
| `mlp_down` | 180.4 GFLOP | 6.3106 ms | **28.59 TFLOP/s** | **193 %** |

`mlp_down` at 28.59 TFLOP/s is **above the 1-MAC roof (14.82)**, above the probe's best sustained
non-WMMA leg (`dot2` bf16, 23.28, itself 78.6 % of *its* 2-MAC roof), and at 96.6 % of the 2-MAC roof
(29.6). It cannot be executing `v_fmac_f32_e32`, which is exactly what
`strix-halo-official-specs.md` §7.3 attributes to the `BB_HPA` (bf16-in/bf16-out) kernels the engine
calls. Either the ISA census is blind to the kernels that actually run — and that file records its own
blind spot: *"Tensile's per-kernel `.co` files are CCO containers (`CCO\003`) that `llvm-objdump`
refuses, which also reads as absence"* (§7.1 trap note) — or one of the numbers above is wrong.
**NOT MEASURED: whether the engine's `--gemm library` kernels issue matrix instructions** (no
matrix-op counter has ever been asked for; window2's pass A requested only
`SQ_INSTS SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_VMEM SQ_WAVES`). This is the most consequential open fact
in the tree right now, because it decides whether "nothing in the stack issues a matrix instruction" is
a capability gap (fund a WMMA kernel) or a misreading (pick a better library algo instead).

---

## 2. Method: a family cannot be bound by both walls, so both are priced

For each family: `binding = max(byte_floor, arith_floor)`, and
`prize from an arithmetic-class swap = measured − max(byte_floor, FLOP/new_ceiling)`. That formula *is*
the answer to "which binds": a swap can only pay down to whichever floor is higher, so if
`byte_floor > FLOP/7.30` no arithmetic class on this ISA can help the family at all.

**Ceilings — measured sustained rates, not roofs (`prototype/issue_rate_probe/results.md` §1):**

| class | sustained | share of its own roof |
|---|---|---|
| scalar `v_fma_f32` | **7.30 TFLOP/s** | 49.3 % of 14.82 |
| `v_pk_fma_f16` | 17.66 | 59.6 % of 29.6 |
| `v_dot2_f32_f16` / `_bf16` | 22.59 / **23.28** | 76.2 / 78.6 % of 29.6 |
| `v_dot4_i32_i8` | 28.75 TOPS | 48.5 % of 59.4 |
| `v_dot8_i32_i4` | **55.96 TOPS** | 47.2 % of 118.8 |
| `v_wmma_f32_16x16x16_f16` | **52.63 TFLOP/s** | 88.8 % of 59.36 |
| (library, 16-bit `C`, engine's own shapes) | **18.84 (M=512) / 22.31 (M=1024) / 23.26 (M=3072)**; single calls 33.1–37.7 | `devices/strix-halo-compute.md`, window `prefill-Wclean` |

The library row is included because it is the only ceiling a route can adopt without writing a kernel,
and because the engine's own gemms are already above the probe's scalar leg — pricing a swap against
7.30 alone would overstate every prize in §5.

**Byte denominator: 237 GB/s** (`bench/performance-status.md` §4; retook 239.3, +1 %, no column moves).
The byte side is priced per family, not in aggregate; §4 gives the aggregate sanity check.

---

## 3. Family table: measured, share, both predictions, and which is lower

In-situ ms and shares: 1024 and 2048 from
`progress/bench/evidence/prefill-occupancy-2026-10-16/window1-timing.txt`, 4096 from `window4-4096.txt`;
summarised in `prefill-occupancy-results-2026-10-16.md` §1. All are `--time-waves` **serialized
(stage) wall** shares, with `plan_match=1` and `ungaged_ms ≤ 0.009 %` on every row.

At 4096 the `WAVE` rows print only four families; the rest are recovered from the `GROUP` line's
percentages × `wall_ms=1307.7549` (marked **derived**). The recovery closes: the eleven rows sum to
1307.69 against a 1307.75 wall, i.e. to within the instrument's own `ungaged_ms = 0.098`.

| family | 1024 ms (%) | 2048 ms (%) | 4096 ms (%) | shape |
|---|---|---|---|---|
| `gdn_rec` | 128.31 (50.85) | 259.74 (46.87) | 520.26 (39.78) | recurrence, serial in T |
| `attn` | 27.23 (10.79) | 104.32 (18.83) | 412.07 (31.51) | two 256-dim dots per (row,lane,head) |
| `proj` | 39.72 (15.74) | 76.66 (13.83) | 155.73 (11.91) | 114 GEMMs / body, M=512 |
| `mlp_gateup` | 26.52 (10.51) | 53.68 (9.69) | 101.94 (7.79) | 2 × 24 GEMMs / body |
| `mlp_down` | 6.31 (2.50) | 13.73 (2.48) | 25.50 (1.95) **derived** | 24 GEMMs / body |
| `cast` | 7.32 (2.90) | 15.09 (2.72) | 31.52 (2.41) **derived** | elementwise bf16↔f32 |
| `gdn` batched (conv / gates / gated_norm) | 7.22 (2.86) | 14.84 (2.68) | 30.86 (2.36) **derived** | stencil + elementwise |
| `copy` (1 embed row + conv-window copies per token) | 2.67 (1.06) | 5.32 (0.96) | 10.72 (0.82) **derived** | gather, ~2.5 KB/token |
| `glue` | 3.80 (1.50) | 7.31 (1.32) | 15.69 (1.20) **derived** | norms, small tensors |
| `lm_head` | 2.26 (0.89) | 2.25 (0.41) | 2.22 (0.17) **derived** | GEMV, M=1 |
| `host_tail` | 0.95 (0.38) | 1.13 (0.20) | 1.18 (0.09) **derived** | host |
| **wall (serialized / burst)** | 252.33 / 248.04 | 554.13 / 548.56 | 1307.75 / 1285.57 | |
| `exposed_ms` (serialized − burst) | 4.29 | 5.57 | 22.19 | 1.7 % of the 4096 wave |

### 3.1 `proj` — arithmetic-bound, 5.6× above its own byte wall

* **Measured.** 39.72 / 76.66 / 155.73 ms; 15.74 / 13.83 / 11.91 %. `launches = 114 × bodies`
  (228 / 456 / 912), `ms_per_launch = 0.168–0.174`.
* **Byte side.** weights 0.4667 GB × bodies + outputs 0.4067 GB/token + `x` re-read 0.1447 GB/body
  (each of the 114 launches re-reads `x[512×k]`; Σk over the 114 = 141,312) = 1.64 / 3.28 / 6.56 GB →
  **byte floor 6.9 / 13.8 / 27.7 ms**. Measured is **5.6–5.8× above it** → not byte-bound.
* **Arithmetic side.** 477.9 / 955.9 / 1911.8 GFLOP. Floors: scalar 7.30 → 65.5 / 131 / 262 ms;
  dot2 23.28 → 20.5 / 41.1 / 82.1; WMMA 52.63 → **9.1 / 18.2 / 36.3**; library same-M mix (18.84) →
  25.4 / 50.7 / 101.5. **The scalar-FMA floor (65.5 ms) is 1.65× ABOVE the measurement**, so this
  family is not executing the class the probe's floor leg prices; it is at 12.03 TFLOP/s — 81 % of the
  1-MAC roof and 64 % of the library's same-M rate.
* **Swap prize** (`measured − max(byte_floor, FLOP/ceiling)`): library-parity, no kernel →
  **14.3 / 25.9 / 54.2 ms**; dot2 → **19.2 / 35.6 / 73.6 ms**; WMMA → **30.6 / 58.5 / 119.4 ms**.
* **Numerics cost, D13 class.** The output already comes from a bf16-in / f32-compute / bf16-out
  library GEMM whose reduction order the engine neither controls nor claims — ledger §6 debt 4 says so
  in words: *"`--gemm library` makes a prompt's logits depend on the wave-row cap (a library GEMM
  reassociates with `M`), so wave-width independence is now a band question, not a byte-identity
  claim"*. An algo/route choice inside the same library is therefore **the same error class it already
  carries → D13 case 1, gate decides**. A hand dot2 or WMMA kernel *adds* an association change on a
  path that has no bit-identity claim to break, and WMMA adds the fp16 format question (D14 census:
  activations peak 82.5, margin 794×, zero values over 65504, 0.0085 % flushed to zero) →
  **D13 case 2, gate decides; the band was already widened by debt 4.**
* **GEMM-shaped?** Yes, all 114. Caveat: 32 of them (`in_proj_a`, `in_proj_b` — 16 rows × 1024) are
  shape-starved, 5 % of probe for 32 KiB/launch
  (`progress/2026-10-16/performance-status-2026-10-16.md`:126). No tile of any shape fixes a 16-row
  matrix; they are launch-bound and hold 4.4 % of the family's FLOPs.

### 3.2 `mlp_gateup` — arithmetic-bound, and the largest gap to a *measured* ceiling in the tree

* **Measured.** 26.52 / 53.68 / 101.94 ms; 10.51 / 9.69 / 7.79 %.
* **Byte side.** weights 0.705 GB/body + gate/up outputs + `x` re-read = 0.86 / 1.72 / 3.45 GB →
  **floor 3.6 / 7.3 / 14.6 ms**; measured is 7.3× above → not byte-bound.
* **Arithmetic side.** 360.8 / 721.5 / 1443.1 GFLOP. Floors: scalar → 49.4 / 98.8 / 197.7 (again
  **above** the measurement); dot2 → 15.5 / 31.0 / 62.0; WMMA → **6.9 / 13.7 / 27.4**;
  **hipBLAS measured at exactly this shape and M (`gate_proj` 3584×1024, M=512, 33.6 TFLOP/s) →
  10.7 / 21.5 / 43.0 ms.**
* **Swap prize.** library-parity **15.8 / 32.2 / 58.9 ms**; dot2 **11.0 / 22.7 / 39.9 ms**;
  WMMA **19.6 / 40.0 / 74.5 ms**.
* **Numerics cost.** Identical to `proj`: same library, same 16-bit-`C` path, same band status.
* **GEMM-shaped?** Perfectly — 512×1024 → 2×3584 per layer. The best-shaped family in the engine for a
  16×16×16 tile, and the place where the library's own measured 33.6 says the machine will go with no
  hand-written matrix code at all.

### 3.3 `mlp_down` — arithmetic-bound, but already ≈at the library's rate

* **Measured.** 6.31 / 13.73 / 25.50 ms (4096 derived); 2.50 / 2.48 / 1.95 %.
* **Byte side.** 0.41 / 0.82 / 1.66 GB → floor 1.75 / 3.46 / 6.98 ms; measured is 3.6–4.0× above.
* **Arithmetic side.** 180.4 / 360.8 / 721.6 GFLOP → **achieved 28.59 / 26.27 / 28.29 TFLOP/s**.
  Floors: scalar 24.7 / 49.4 / 98.8 (above measurement); dot2 7.75 / 15.5 / 31.0; WMMA 3.4 / 6.9 /
  13.7; library `down_proj` at M=1024 measured 33.7 TFLOP/s → 5.35 / 10.7 / 21.4 ms.
* **Swap prize.** library-parity **0.96 / 3.0 / 4.1 ms**; dot2 **−1.4 / −1.8 / −5.5 ms — a loss**;
  WMMA **2.9 / 6.8 / 11.8 ms**.
* **The point.** `mlp_down` is at ~85 % of what one `hipblas` call at its own shape achieves, and is
  *above* dot2's best sustained rate. **A dot2 swap would make this family slower**, and a WMMA kernel
  must sustain > 29 TFLOP/s in situ (≈55 % of the probe's WMMA leg, at the engine's occupancy, in the
  wave's walk order) merely to tie. This is the concrete counter-example to "dot2 is the cheapest big
  lever" (D16 item 2): 4.1× is 4.1× over 4.696, and `mlp_down` already runs at 6.1× that.
* **Numerics cost.** Same class as above; nothing to add.
* **GEMM-shaped?** Yes — 512×3584 → 1024.

### 3.4 `attn` — byte-bound by 7.7×, and by nothing else

* **Measured.** 27.23 / 104.32 / 412.07 ms; 10.79 / 18.83 / 31.51 %.
* **Byte side.** `Σ_lanes = T(T+1)/2` × 2 KiB/layer × 6 layers (D2) = 6.45 / 25.78 / 103.10 GB →
  **floor 27.2 / 108.8 / 435.0 ms**. Measured/floor = 1.001 / 0.959 / 0.947, i.e. effective
  demanded-byte rate 236.8 / 247.1 / 250.2 GB/s (`prefill-occupancy-results` §2). **At the byte wall.**
* **Arithmetic side.** Per (row, lane, layer): 8 q-heads × 256 dim × 2 FLOP for scores + the same for
  PV = 8,192 FLOP. Totals 25.79 / 103.1 / 412.4 GFLOP → achieved **0.947 / 0.988 / 1.00 TFLOP/s**.
  Floors: scalar 3.53 / 14.1 / 56.5; dot2 1.11 / 4.43 / 17.7; WMMA 0.49 / 1.96 / 7.84;
  dot8-i4 0.46 / 1.84 / 7.37.
* **Which is lower: arithmetic, by 7.7× at every width** (27.2 vs 3.53; 108.8 vs 14.1; 435 vs 56.5 —
  the ratio is constant because both bills are quadratic). The byte floor binds; only traffic moves it.
* **Swap prize: zero at `ATTN_TILE_M == 1`** (`prototype/attn_decode/attn_decode_kernels.h:404-422`):
  27.23 − max(27.2, 3.53) = **0.03 ms** at 1024. No arithmetic class on this ISA — not dot2, not WMMA,
  not int4 — buys `attn` a millisecond while one query row walks the K/V buffer alone. Fact A's "47 % of
  a 2-wide issue" is fully consistent with this: an instruction stream 7.7× cheaper than bytes is an
  instruction stream that is not the wall.
* **Where fact B re-enters — the useful new arithmetic.** With M query rows sharing one K/V walk the
  byte floor becomes `435/M` at 4096 while arithmetic is unchanged, so the crossover
  `M* = byte_floor(M=1) / arith_floor = 27.2/3.53 = 108.8/14.1 = 435/56.5 = 7.7` —
  **width-independent**, because both bills scale with `Σ_lanes` (the D4b "scaling cannot separate the
  walls" observation, now priced with the corrected ceiling). Hence:

| M (rows per K/V walk) | 1024 best-case → prize | 2048 best-case → prize | 4096 best-case → prize | wall at the best case |
|---|---|---|---|---|
| 2 | 13.6 → **13.6 ms** (5.4 % of wave) | 54.4 → **49.9** (9.0 %) | 217.5 → **194.6** (14.9 %) | bytes |
| 4 | 6.8 → **20.4** (8.1 %) | 27.2 → **77.1** (13.9 %) | 108.8 → **303.3** (23.2 %) | bytes |
| 8 | 3.53 → **23.7** (9.4 %) | 14.1 → **90.2** (16.3 %) | 56.5 → **355.6** (27.2 %) | scalar FMA (just) |
| 16 | 3.53 → 23.7 | 14.1 → 90.2 | 56.5 → 355.6 | scalar FMA |
| 16, on dot2 | 1.70 → 25.5 | 6.80 → 97.5 | 27.2 → **384.9** (29.4 %) | bytes again |
| 32, on dot2 | — | — | 17.7 → 394.4 (30.2 %) | dot2's own floor (17.72 ms) |

  So: **dot2 is worth 2.1 ms at 4096 after M=8 tiling and 29.3 ms after M=16; WMMA on the score path is
  worth ~0 until M ≈ 32.** On dot2, bytes stop being the wall at 4096 once M > 24.5 (435/24.5 = 17.7),
  so a deep tile on dot2 saturates at 17.7 ms and a matrix-class score kernel (7.84 ms) is what would
  make M beyond that worth anything. The traffic lever is not merely ranked above instruction surgery,
  as D16 says — on this family it is the *only* lever at M ≤ 7, and the second lever is dot2 at
  M ≥ 16, not WMMA.
  Reachable M is a hardware question the repo has data for but has not answered: today's form is
  `M=1`, VGPR 48, LDS 12,288 B of 64 KiB (`prefill-occupancy-results` §3), so M = 8 looks plausible and
  M = 32 does not. **NOT MEASURED: the largest M that fits the accumulator + staged-lane budget.**
* **Numerics cost of M-tiling itself: zero, if admitted the way `--attn-tile 1` was.** D5 item 3 fixes
  the rule (bit identity on the per-row lane order; a row tile that changes a row's reduction order is
  not admissible as a speed change), and the precedent is `--attn-tile 1` itself: `attn_form_band.py`
  136/136 and 520/520 with 0 elements differing (ledger §5).
* **Numerics cost of a dot2/WMMA score dot: a bit-identity → band conversion.** The score dot's per-row
  summation order is what §05.5's ordered reduction pins. A dot2 keeps bf16 and f32 accumulate and
  changes only association (D16 item 2); a WMMA changes association *and* format (fp16 operands, f32
  accumulate; D14's census is green at 794×, and if it goes red the suspect order is the 92,793
  flush-to-zero activations first, association second — the `q̃` lesson, `prototype/gdn_collapse/results.md` §2).
  Classification: **D13 case 2** — widen only if the geometry-derived must-catch set still fires, and
  that set exists and is current (`prototype/attn_tile_model.py`'s `neg_expect` / `wide_case`, the
  512/513/514/515-lane synthetic wave rows, `partalias`), with the recorded caveat that a score-only
  injection structurally cannot move a one-lane row, so the set must stay derived.
* **GEMM-shaped?** Yes, and it becomes *more* so with M: scores are `Q[M·8 × 256] × Kᵀ[256 × L]`, PV is
  `P[M·8 × L] × V[L × 256]`. M-tiling is what creates the 16-deep operand a matrix tile needs, which is
  why the traffic route sits *on* the path to the matrix route rather than competing with it. D5 item 4's
  "MFMA subsumes M-tiling" is true in the limit and wrong as a sequence: there is nothing to tile until
  M exists.

### 3.5 `gdn_rec` — 40–51 % of the wave, and untouched by both facts

* **Measured.** 128.31 / 259.74 / 520.26 ms; **50.85 / 46.87 / 39.78 %**; `launches = 18 × T`,
  6.96–7.06 µs each.
* **Arithmetic side.** 33.94 GFLOP per 1024-lane wave (`prototype/gdn_chunk/kernel_price.py:31`,
  `prototype/gdn_collapse/collapse_price.py:39`) → achieved **0.265 TFLOP/s** (the repo states 0.271 at
  its own 125.4–127.0 ms rows). That is **3.6 % of the slowest measured ceiling (scalar 7.30)** and
  0.5 % of WMMA. Floors: scalar 4.65 / 9.3 / 18.6 ms; dot2 1.46 ms at 1024; dot8 0.61 ms.
* **Byte side.** The repo's own state-traffic account is 0.054 GB/token (frozen ledger :86 — "GDN state
  traffic (18 layers, warm) 0.054 GB, roofline 0.228 ms cold"), i.e. 55.3 / 110.6 / 221.2 GB per wave →
  byte floor **233 / 467 / 933 ms**, **1.8× the measured time**; effective 431 / 426 / 425 GB/s across
  three widths, the same above-DRAM signature as the decode state row (882.7 GB/s, 372 % of probe,
  ledger §4). The narrower account (read+write of the 18 MiB F32 state = 37.7 MB/token) gives 300 GB/s.
  Either way: above the probe, and *below* its own byte floor is not a state a family can be in — so the
  byte reading must be treated as a coincidence of two linear models, not as a wall.
* **The tie-breaker is already measured, and it says neither.** Route A
  (`prototype/gdn_collapse/results.md`) made the state **resident in LDS** and removed 18,396 of 18,432
  launches: **112.04 ms against seq's ~127 — a 12 % move**. A form that removes essentially all per-token
  state traffic and all per-token dispatch is neither byte-bound nor dispatch-bound. The repo's verdict
  stands: **latency-bound on §04.4's 2,048 dependent chains at ~1.6 working wavefronts per CU**, with the
  corroborating null result that `#pragma unroll 8` on the two serial loops was bit-identical and moved
  112.8 → 113.4 ms (AGENTS.md trap row; `prototype/gdn_collapse/results.md` §4–§5).
* **Swap prize: ≈0.** Even at dot8-i4's 55.96 TOPS the family's whole arithmetic is 0.6 ms of a
  128–520 ms family. The standing instruction for this family is the AGENTS.md one: *price a serial
  kernel in cycles-per-dependency at its realised occupancy, or do not price it.*
* **Numerics cost.** Moot. For completeness: no 16×16×16 tile applies — the state update `S_t = f(S_{t-1})`
  is serial in T with no 16-deep reduction across the batch, and the only reduction (read-out
  `o = S·q`, 128-long per output element) is per-(head, d_v) with M = 1. `gdn_rec` meets the matrix unit
  only by being reformulated chunk-wise (P9's delta-rule scan, which *is* GEMM-shaped) — and P9 is
  already priced and measured 5.9× worse than its bottom-up price (`prototype/gdn_chunk/`,
  `bench/performance-status.md` §3 "Route A is dead").
* **One prize I must name and explicitly not fund: halving the state width (F32 → 16-bit).** If the
  425–431 GB/s fit were a real wall, halving state bytes would take 520 ms → ~260 ms at 4096 (20 % of the
  wave). Route A's −12 % argues hard against it; **NOT MEASURED: a 16-bit GDN state variant**. And this
  is the one format change in the repo with a *genuine* band problem — a recurrent accumulator over T
  steps losing 7–16 mantissa bits, judged against §04.4's per-row budget, which is already 0.3 % inside
  a stage line. Ranked last among format ideas for that reason, and it would be a user decision, not a
  gate decision.

### 3.6 `glue`, `gdn_conv`/`gates`/`gated_norm`, `cast`, `copy`/embed, `lm_head`, `host_tail`

| family | ms at 1024 / 4096 | its own roofline | bound by | arithmetic-swap prize | shape |
|---|---|---|---|---|---|
| `glue` | 3.80 / 15.69 | **0.0021 ms** (49 launches, ledger §4) | launch + interleave | **0** — norms and small tensors: no reduction to accelerate | elementwise |
| `gdn_conv` | 4.44 / ~18 (linear; the split is not printed at 4096) | byte floor 1.91 ms at 1024 (453 MB of conv traffic) | bytes at 2.3× floor; arithmetic is 0.90 GFLOP = 0.12 ms | **0** | 4-tap depthwise stencil, not a matmul |
| `gdn_gated_norm` | 2.37 / ~9.5 (linear; split not printed) | 0.0016 ms; **burst 0.4273 ms** (ledger §4) | interleave: in-situ is 5.5× its own burst — 66 µs/launch against a 2.967 µs burst launch | **0** | elementwise + norm |
| `cast` | 7.32 / 31.52 | ~0.007 ms of traffic; 229 launches at 1024 | dispatch | **0** — these casts exist because the vector path and the gemm path disagree on width; dot2/WMMA take 16-bit operands and still need them | elementwise |
| `copy` = **embed** gather + conv-window copies | 2.67 / 10.72 | ~2.5 KB/token; `launches=1060` at 1024 = 1,024 embed rows + 36 window copies (`engine/decode_step.cpp:37,469`) | 100 % dispatch: 1,060 × 1.85 µs = 1.96 ms | **0** — a gather, no arithmetic at all | gather |
| `lm_head` | 2.26 / 2.22 | **2.146 ms** (485 MiB); burst 2.1309 ms = **100.7 % of probe** (ledger §4) | bytes, at the ceiling | **0**, and unfundable in principle: **M = 1**, so a 16×16×16 tile does not exist for it. Its 508.6 MFLOP is 0.07 ms of arithmetic against 2.26 ms of weight stream | GEMV |
| `host_tail` | 0.95 / 1.18 | CPU | host | 0 | — |

**Caution that applies to the whole bottom half of this table.** The family attribution is the
*serialized* (stage) wall. `exposed_ms` — serialized minus burst — is 4.29 ms at 1024 and **22.19 ms at
4096 (1.7 %)**, so production submission already hides launch structure behind execution. A fusion-style
route does not get to bank `glue`+`cast`+`copy`+`gated_norm` (≈ 67 ms at 4096, 5.1 %); its ceiling is
nearer `exposed_ms`, and the ledger has already refuted launch-count work once (*"Captured graphs:
1.78 µs/kernel replay against 1.85 µs eager — 4 %, not the 3–5× a discrete card gives, because the limit
is the GPU's own dispatch rate"*, ledger §5).

### 3.7 The integer future, priced, and the GEMM/recurrence split

**Integer-class floors for the GEMM families** — raw-dot bounds only. `issue_rate_probe` §2 states its
integer legs model no dequant or scale work, and int8/int4 are a D13 exclusion, so these are ceilings on
a route that is not allowed to be funded; they are shown because the brief asked for the dot8-i4 column.
Both columns count MACs×2, so they are directly comparable with the TFLOP/s rows above.

| family | dot8-i4 at 55.96 TOPS (1024 / 2048 / 4096 ms) | dot4-i8 at 28.75 TOPS | vs what the family already achieves |
|---|---|---|---|
| `proj` | 8.5 / 17.1 / 34.2 | 16.6 / 33.3 / 66.6 | 39.72 / 76.66 / 155.73 today |
| `mlp_gateup` | 6.4 / 12.8 / 25.8 | 12.6 / 25.1 / 50.2 | 26.52 / 53.68 / 101.94 today |
| `mlp_down` | 3.2 / 6.5 / 12.9 | **6.28 / 12.6 / 25.1** | **6.31 / 13.73 / 25.50 today** |

The `mlp_down` row is the finding: `v_dot4_i32_i8`'s *sustained* 28.75 TOPS would land within 0.5 % of
what that family already does in bf16 — **an int8 GEMM would buy roughly nothing in prefill**, before any
dequant cost, and its prize is instead half the weight bytes on the decode GEMV path (out of scope for
this pass, and still a user decision under D13).

**GEMM-shaped vs not — does a 16×16×16 tile exist for the family at all?**

| GEMM-shaped, matrix-tileable | not tileable, and why |
|---|---|
| `mlp_gateup` (512×1024 → 2×3584); `mlp_down` (512×3584 → 1024); `proj`'s 82 wide matrices; `attn` scores `Q[M·8×256] × Kᵀ[256×L]` and PV `P[M·8×L] × V[L×256]` — tileable only once M ≥ 16 rows/lanes exist | `proj`'s 32 narrow matrices (`in_proj_a/b`, 16 rows) — too thin for any tile, launch-bound at 5 % of probe; `lm_head` — **M = 1**, the tile's 16-deep dimension cannot be filled without batching 16 requests' logits; `gdn_rec` — recurrence, `S_t = f(S_{t-1})` is serial in T and its one 128-long reduction is per-(head, d_v) with M = 1; `gdn_conv` — 4-tap stencil; `glue`, `gdn_gates`, `gdn_gated_norm`, `cast` — elementwise/norms, no reduction; `copy`/embed — a gather |

---

## 4. What binds where — the load-bearing judgement, stated explicitly

1. **`attn` (31.51 % of a 4096 wave): A binds, B does not** — at every width and at every
   M ≤ 7. Bytes are 7.7× the scalar-arithmetic floor; the traffic/arithmetic crossover is at
   M ≈ 7.7 and is width-independent. Dot2 enters at M ≥ 16 (worth 29.3 ms at 4096), WMMA not until
   M ≈ 32.
2. **`proj` + `mlp_gateup` + `mlp_down` (21.65 % of a 4096 wave): B binds, A does not** — measured at
   3.6–7.3× their byte floors — **but B is already partly spent.** These three families run at
   12.0 / 13.6 / 28.6 TFLOP/s, i.e. 81 / 92 / 193 % of the 1-MAC vector roof and 64 / 40 / 85 % of what
   the library itself measures at their own shapes. The remaining headroom is **1.6×** (to the library's
   33.6) or **3.7×** (to WMMA's 52.63) — *on 21.65 % of the wave*, not 11.2× on it.
3. **`gdn_rec` + `gdn` batched (42.14 % of a 4096 wave): neither binds.** 0.265 TFLOP/s is 3.6 % of the
   slowest measured ceiling, and a form that removed all per-token state traffic and 18,396 launches
   bought 12 %. Latency on the per-token dependency chain at ~1.6 working wavefronts per CU. **Neither
   fact A nor fact B reaches 42 % of the wave**, which is why rebuilding the route index on both facts
   can only ever move ~58 % of it. Anyone asked "what does the matrix ceiling buy us?" should be told
   this number first.
4. **`glue` / `cast` / `copy` / norms (6.8 % of a 4096 wave): dispatch and interleave bind**, and only
   `exposed_ms` (1.7 % of the wave) of it is reachable from the submission side.
5. **Aggregate check.** Demanded bytes per wave: attn 103.10 + `gdn_rec` 221.18 + proj 6.56 + mlp 4.39 +
   `lm_head` 0.51 = **335.7 GB at 4096** against a 1307.75 ms wall = **257 GB/s**; the same ratio at
   1024 is 65.1 GB / 252.33 ms = **258 GB/s**. So the wave *in aggregate* sits at the fabric's measured
   ceiling, and **96.6 % of its demanded bytes are two families** (`attn` 30.7 %, `gdn_rec` 65.9 %).
   The excess over the 237 GB/s probe must be coming from the on-die tier (consistent with `attn`'s
   250.2 GB/s and `gdn_rec`'s 425 GB/s effective rates) — or one of these byte models overcounts;
   `NOT MEASURED: a traffic counter that can adjudicate it`, and window 2 already retired
   `GL2C_EA_RDREQ_64B` for failing to reproduce a 485 MiB known workload
   (`prefill-occupancy-results` §4).

---

## 5. Re-ranked route table

Prize = `measured − max(byte_floor, FLOP/new_ceiling)`, summed over the families the route touches
(§3). "Risk" weighs the numerics admission (D13 class) and the engineering unknowns, weighted by this
repo's own failure history for that class of change. Prizes are ms of the serialized wave, and the
parenthesised figure is that as a share of the wave at that width.

| rank | route | prize 1024 | prize 2048 | prize 4096 | numerics class (D13) | engineering risk |
|---|---|---|---|---|---|---|
| **1** | **R1 — attention row-tiling: M query rows share one K/V walk, diagonal masked, per-row lane order bit-identical.** M=2 → M=8 | 13.6 (5.4 %) → 23.7 (9.4 %) | 49.9 (9.0 %) → 90.2 (16.3 %) | **194.6 (14.9 %) → 355.6 (27.2 %)** | **case 1** — gates stay green, bit-identity required (D5 item 3; precedent `--attn-tile 1`, `attn_form_band.py` 136/136) | moderate-high. Known bug class is *address*, not arithmetic (the aliased partial slot, `attn_decode/results.md` §20.1); the derived must-catch machinery already exists and must be extended, not written |
| **2** | **R2 — close the matmul gap to the library's own measured rate**, no hand kernel: algo/algo-selection (hipBLAS vs hipBLASLt), op/transit layout, one fused `gate+up` call, `x` staging so the 114 launches stop re-reading it | 31.1 (12.3 %) | 61.1 (11.0 %) | **117.2 (9.0 %)** | **case 1** — same library, same dtypes, same 16-bit-`C` path; the band it touches (debt 4) is already widened | low-moderate, **but the gap is unverified in situ**: `NOT MEASURED: a burst of the shipped proj/mlp plan at rows=512 in the wave's own walk order`. Part of the 12.3 % may be the ledger's own "a burst is not a step" effect (§2: families +0.46–1.01 ms) |
| **3** | **R3 — hand-written WMMA gemm** (fp16 operands, f32 accumulate) for `proj` + `mlp_gateup` (+`mlp_down`) on 512-row bodies. Stated as an increment over R2's target: full prize vs today is 53.1 / 105.4 / 205.7 ms | +22.0 over R2 | +44.2 over R2 | **+88.5 over R2** | case 2 — fp16 range already censused green at 794× (D14); band already loose on this path | **high, and two of its own preconditions are unmet.** `issue_rate_probe/results.md` §5.1: the WMMA **fragment layout is not proven** (`tile SKIPPED … NOT DECIDABLE`); §5.2: **all 7 negative controls of the harness fail**, so the harness a WMMA gate would borrow is not honest yet. Must beat hipBLAS's 18.84–33.7, not 4.696 — `mlp_down` at 28.6 says a WMMA kernel sustaining < 29 TFLOP/s in situ is a **regression**, and `issue_rate_probe` §5 of its own doc says price in cycles-per-dependency at realised occupancy, not instructions |
| **4** | **R4 — `v_dot2_f32_bf16` in hand kernels**: the attn score/PV dots *after* deep M-tiling; any hand reduction with pairable products | 0 today; 0.13 after M=8 | 0 today; 0.5 after M=8 | **0 today; 2.1 after M=8; 29.3 after M=16** | case 2 on the score path: a bit-identity → band conversion of §05.5's ordered reduction, widened only if the derived must-catch set still fires | low technically (no format change, f32 accumulate), **but it is sequenced behind R1 by arithmetic, not by taste**: at M=1 its prize is 0.03 ms |
| **5** | **R5 — raise the lane ceiling** (`kAttnMaxSplit` 8 → 40, `a_part` ~68 → ~338 MiB at `wave_rows` 512, D1) so the >4096 regime exists at all | — | — | makes R1's prize realisable past 4096, where attn is the whole quadratic | needs its own band admission: the streaming merge's association depends on `split` (D1, D5 item 5, ledger §6 debt 4) | moderate; memory is a non-issue per AGENTS.md scope (338 MiB against a 34 GiB pool) |
| **6** | **R6 — fusion/launch leftovers**: `copy`'s 1,060 embed-row memcpys into one gather kernel, `glue`/`cast` fusion | ≤ 4.3 | ≤ 5.6 | **≤ 22.2** (`exposed_ms` is the ceiling, not the 63.8 ms of family ms) | case 1 (bit-identical by construction) | low, already refuted once for launch-count (captured graphs, 4 %) |
| **7** | **R7 — `gdn_rec` by arithmetic class or state width** | **≈ 0** | ≈ 0 | ≈ 0 by route A's −12 %; a 16-bit state variant could in principle be worth ~260 ms at 4096 but `NOT MEASURED`, and it is a genuine band problem (recurrent accumulator × T steps) | state width would be a **user decision**, not a gate decision | do not fund; the lever here is algorithmic (P9, already 5.9× worse than priced) |
| **8** | **R8 — integer paths (`v_dot4_i32_i8` 28.75 TOPS, `v_dot8_i32_i4` 55.96 TOPS)** | **~0 for prefill** | ~0 | ~0: `dot4_i8`'s *sustained* 28.75 TOPS is 1.005× `mlp_down`'s current 28.59 TFLOP/s, and neither leg models dequant/scale work (`issue_rate_probe` §2) | **needs a user decision** — D13 explicitly excludes int8/int4; the error class changes by orders of magnitude, not by a mantissa count | its real prize is half the weight bytes on the **decode** GEMV path, which this pass is out of scope for |
| **9** | **R9 — `lm_head` arithmetic** | 0 | 0 | 0 — M = 1, so no 16×16×16 tile exists for it; already at 100.7 % of probe | n/a | dead |

Two things this ranking deliberately refuses to do, for the record: it does not fund anything that
depends on the WMMA fragment layout being what the builtin's documentation implies (§5.1 of the probe's
own doc says it is not decidable from what has been measured), and it does not count a millisecond of
`gdn_rec` as available from any instruction change, because the one experiment that could tell us
(route A) measured a 12 % return on removing the whole traffic and dispatch bill.

---

## 6. The one route I would fund next, and the cheapest measurement that would falsify it

**Fund R1 — attention row-tiling, M = 2 first, with bit-identity on the per-row lane order required as
the admission gate** (D5 item 3 unchanged). Why this one and not the flashier fact-B routes:

1. **It is the largest prize at the width that carries the headline.** 194.6 ms of a 1307.75 ms wave at
   M=2, rising to 355.6 ms at M=8. §6 of the ledger is explicit that the long-context row carries any
   claim, and D5 says the user's workload is the big file.
2. **It needs no decision from the user and no new band.** Bit-identity on the per-row lane order is
   already the repo's rule for this kernel and already has a working instrument (`attn_form_band.py`),
   plus derived-from-geometry must-catch sets (`attn_tile_model.py`'s wide anchors and the
   512/513/514/515-lane synthetic wave rows) that were built precisely because a hand-listed set rots.
3. **None of its preconditions are unproven.** R3's are, in its own tool's documentation.
4. **It is on the path to the matrix route rather than in competition with it.** M rows sharing one walk
   is what creates the 16-deep operand a WMMA tile needs; the reverse is not true — there is nothing to
   tile for a matrix path while M = 1.
5. **Fact A and fact B agree here rather than pointing apart.** That is worth saying, because the brief
   framed them as opposing: A says cut traffic, B says the arithmetic class is cheap. Both are
   satisfied by the same first step, and B's price for the *next* step is now measured too — dot2 at
   M ≥ 16 (29.3 ms at 4096), WMMA not until M ≈ 32.

**Cheapest device measurement that would falsify this ranking** (and it is not about R1 — R1's mechanism
is the best-evidenced thing in the file, three widths fitting a byte model to 0.1–5 %):

> **A burst of the shipped `proj` plan and `mlp_gateup`/`mlp_down` plan at rows = 512, in the wave's own
> walk order** — one `tools/gpu_lock.sh` hold, blocking, ~2 minutes, no numerics, no new code beyond a
> row-count flag on `prototype/proj_gemv` / `prototype/mlp_gemv`. `NOT MEASURED` today: every existing
> burst row for those families is a decode-shape one-row plan (ledger §4), so nobody has ever seen these
> GEMMs outside the step.
>
> * If the burst reproduces the library's 18.84 / 33.6 / 33.7 TFLOP/s, then R2's 31.1 ms at 1024
>   (12.3 % of the wave) is real, costs no new numerics class, and **should be funded before R1 at
>   ≤ 2048 lanes** — R2's rank moves to 1 for short prompts and R1 stays 1 for long ones.
> * If the burst reproduces today's in-situ 12.0 / 13.6 TFLOP/s, then the gap is the step-context
>   effect the ledger already documents, R2 and R3 both lose their premise, and R1's lead widens to the
>   whole table.

Two cheaper premise-checks, one of which needs no device at all:

* **Compile-only (no device, minutes):** build a candidate tile at M = 8 and read VGPR/LDS/SGPR from the
  compiler's kernel resource report. Today's form is VGPR 48, LDS 12,288 B (`prefill-occupancy-results`
  §3). **NOT MEASURED: the largest M that fits the accumulator + staged-lane budget at ≥ 2 blocks/CU.**
  If only M = 2 fits, R1's prize is 194.6 ms rather than 355.6 and R2's 117.2 ms is much closer to it
  than this ranking allows.
* **CPU-side, from rows already on disk:** the `irp_kv_read` rows of `issue_rate_probe` §4 still have no
  GB/s conversion — the file says so itself ("no GB/s figure is claimed for these rows"). Converting
  them answers whether a per-layer 8 MiB KV set at 4096 lanes can be driven above 237 GB/s, which is
  D4's still-unbuilt `kv_sweep` question. A fast answer there does not break R1 (demanding fewer bytes
  still means less time), but it would open a *competing* route — get tier relief without reducing
  traffic — that this ranking does not currently carry.

---

## 7. Assumptions, and the NOT-MEASURED list

Assumptions I made that a reviewer should attack first, in order of how much they move §5:

1. **Wave FLOP counts from `engine/state.h` shapes are the families' real arithmetic.** Cross-checked two
   ways (weights-vs-bytes to the frozen ledger's 0.4677/0.5285 GB/token; total matmul 1019.1 vs the
   `devices/strix-halo-compute.md` 1017.9 GFLOP), so this is firm.
2. **`mlp_down` at 4096 lanes = 25.50 ms**, recovered from `matmul_pct=21.65` minus the two printed
   families. The recovery closes to `ungaged_ms`, but it is a derived row, not a printed one.
3. **The 237 GB/s denominator applies to wave-side traffic**, which the `attn` fit supports
   (236.8–250.2 GB/s) and `gdn_rec` does not (425 GB/s effective, cache tier).
4. **The library's `prefill-Wclean` rows are comparable to in-situ family rates.** They are 16-bit-`C`
   like the engine's calls, same dtype, same shapes — but they are bursts and the family rows are stage
   wall. That is exactly what R2's falsifying burst exists to settle.
5. **`Σ_lanes = T(T+1)/2` causal traffic for attention** (D2), verified at three widths to 0.1–5 %.

`NOT MEASURED`, collected: whether the engine's `--gemm library` kernels issue matrix instructions;
a burst of the shipped proj/mlp plan at rows=512 in walk order; the largest attention M that fits the
accumulator/LDS budget; a 16-bit GDN-state variant; GB/s for the `irp_kv_read` legs; a traffic counter
that can adjudicate wave-side bytes (`GL2C_EA_RDREQ_64B` is retired for this); `hipBLASLt` vs `hipBLAS`
algo selection at the engine's shapes (the engine calls `hipblasMatmul`, and every library row on disk
comes from `hipblas`/`rocblas_gemm_ex`); a per-family split of the `gdn` batched row at 4096;
`attn`'s isolated burst at any width (ledger §6 debt 7).

---

## 8. The three biggest surprises, in plain numbers

1. **The engine's matmul families already run at ~14 TFLOP/s, so WMMA is 3.7× above them, not 11.2×.**
   `GROUP matmul_ms` is 72.55 ms for 1,019.1 GFLOP at 1024 lanes = **14.05 TFLOP/s**, and `mlp_down`
   alone is **28.59 TFLOP/s** (180.4 GFLOP / 6.3106 ms) — 1.93× the 1-MAC vector roof of 14.82 and
   *above* dot2-bf16's best sustained 23.28. The probe's baseline of 4.696 TFLOP/s is the number
   `devices/strix-halo-compute.md` retired as an F32-`C` artifact, and the engine is verifiably on the
   16-bit-`C` path (`engine/kernels_gemm.cpp:171`). Consequence for the plan: **dot2-bf16 would make
   `mlp_down` 23 % slower** (its dot2 floor is 7.75 ms against 6.31 ms measured), and a WMMA kernel
   has to beat hipBLAS's 18.84–33.7, not 4.696.
2. **42 % of the wave is reachable by neither fact.** `gdn_rec` is 50.85 % of a 1024 wave and 39.78 % of
   a 4096 wave at **0.265 TFLOP/s** — 3.6 % of the slowest measured ceiling, 0.5 % of WMMA, and its own
   byte floor (0.054 GB/token of state traffic = 233 ms at 1024) is 1.8× *above* its measured time. Route
   A's resident-state form, which removed all per-token state traffic and 18,396 of 18,432 launches,
   bought **12 %**. No instruction swap and no traffic cut is available on it; it needs to become
   GEMM-shaped (P9), and that is already priced at 5.9× worse than predicted.
3. **Attention's arithmetic class is irrelevant by 7.7×, and stays irrelevant until the row tile
   reaches M ≈ 8.** Byte floor vs scalar-arithmetic floor: 27.2 vs 3.53 ms (1024), 108.8 vs 14.1
   (2048), 435 vs 56.5 (4096). Swapping scalar FMA → dot2 → WMMA buys **0.03 / 0.00 / 0.00 ms** at
   today's M=1, **2.1 ms** at M=8 (4096), 29.3 ms at M=16, and ~0 for WMMA until M ≈ 32 — while the
   traffic lever itself is worth **194.6 ms (M=2) to 355.6 ms (M=8)** on a 1307.75 ms wave. The
   crossover M is 7.7 at *every* width, because both bills are quadratic in T — which is the D4b
   "scaling cannot separate the walls" observation, finally priced with a measured ceiling beside it.

---

## 9. Status of this ranking after the work that followed it (2026-10-16, main agent)

Appended deliberately, so §5 and §6 stay exactly as written and every `§3.4`/`§6` cite from
`prototype/attn_m2/` still resolves. The wave has moved since §5 was booked — `--gemm lt` shipped and the attn
instruction surgery shipped — and two of this file's ranks turned out to be wrong in ways worth recording
next to the originals rather than over them.

Current measured state at 4096 lanes: `wall_ms = 1111.95`, `gdn_rec` **48.07 %**, `attn` **25.70 %**, matmul
group **17.92 %**, small-kernel tail ≈ 8 %. Full family table and disposition of every lever:
`prefill-decision-log-2026-10-16.md` **D30**, updated by **D31**.

| §5 rank | route | what actually happened |
|---|---|---|
| 1 | **R1 attention row-tiling (M = 2)** | **RANK WAS WRONG, but the next thing was not.** M = 2 is bit-identical and **slower** — 7–10 % before the levers (D18), **5.5–13.4 % slower after them** (D24), so traffic halving collected nothing twice over. The prize booked here (194.6 ms at 4096) never existed: it came from a byte model that D21 of `progress/2026-10-16/prefill-decision-log-2026-10-16.md` later falsified. What *did* ship out of the same family is instruction surgery on the tile touch — 116 → 45 issue slots, **attn 1.466×, whole wave 1.125×** — which this file priced at zero because it was not a traffic route. |
| 2 | **R2 close the matmul gap to the library's own rate, no hand kernel** | **SHIPPED, and it was the best of the file.** `--gemm lt` is the default (`aaefe01`, D26–D29): GROUP matmul **276.90 → 195.11 ms (1.419×)** and the wave **1.049×** at 4096, 1.110× at 1024. §6's cheap falsifier — burst the shipped plan at rows = 512 in walk order — is what found it, and §6's own conditional ("R2's rank moves to 1 for short prompts") came true: at 1024 lanes it *was* rank 1. The booked prize of 117.2 ms at 4096 was ~82 ms of real GROUP delta plus a remainder that was the ledger's burst-vs-step effect, which §2 already warned about. |
| 3 | **R3 hand-written WMMA GEMM** | **DEPRIORITISED, and by arithmetic rather than by its own unmet preconditions.** Those are still unmet (fragment layout undecidable, all 7 negative controls of the probe harness failing), and on top of that R2 ate most of its prize: the library now runs **33.8–35.9 TFLOP/s** at the shipping width against a measured WMMA roof of 52.63, so the remaining increment is ≈ 1.47× of a much smaller number, not +88.5 ms. Do not fund before D31's 3.7 and the `gdn_rec` items. |
| 4 | R4 `v_dot2_f32_bf16` | unchanged; still sequenced behind M-tiling, which is now parked, so effectively dormant. |
| 5 | R5 raise the lane ceiling | unchanged, still blocked on the numerics admission. It is also the only route that could re-open R1, since traffic that never reaches DRAM is what killed M = 2. |
| 6 | R6 fusion/launch leftovers | unchanged, and the ceiling is `exposed_ms` as written; `WAVEGAP ratio=1.030` says there is almost nothing exposed to overlap. |
| 7 | **R7 `gdn_rec` by arithmetic class or state width** | **RANK WAS TOO LOW, and it is now the head of the queue.** The instruction half is closed (D23/D25: 1.0159× on the fused-reduction MODE, kill band 1.0159 — funded no), but this file's own note — "a 16-bit state variant could in principle be worth ~260 ms at 4096" — is the largest live number on the board, and the reason it is not being dismissed is D25's real finding: the family is **latency-bound at ~1.6 working wavefronts per CU**, which no route in §5 addressed. Open now: the state-dtype/traffic census and the dependency-break falsifier (D30 §1 items 1.6 and 1.3). |
| 8 | R8 integer paths | unchanged, out of authority (int8/int4 is a user decision). |
| 9 | R9 `lm_head` | unchanged, dead, M = 1. |

**One new route this file does not contain, and where it ranks:**

| new | **D31 3.7 — select beyond the shallow heuristic answer** | **mid-table.** `ratio` to rank-0 at 64 candidates is 1.39 `down_proj` / 1.34 `o_proj+out_proj` / 1.21 `gate_proj+up_proj` / 1.13 `in_proj_qkv`, with the winners sitting at ranks 16–55 where the engine's 8-wide query cannot see them, and a pin is not bound by that width. Priced at **≈ 24 ms of a 4096-lane prefill (2.2 %) as a burst ceiling**, below `gdn_rec` because D28 proved a burst winner can be an in-situ loser (two of three were), and the same window showed 7.5 % window noise on a 0.12 ms call. Its sibling, raising the workspace budget, is **closed negative** (D31): the apparent +8.8 % at 12 candidates was the truncation, not the workspace.

The honest summary of §5 from here: **the file was right that the matmul gap was cheap and wrong that
attention's gap was traffic**, and the two errors are the same error — pricing a family from a byte model
instead of asking which wall is lower with the instructions counted. That is why D30 asks nine questions per
family now, and why the biggest thing left is the one question §5 never asked of `gdn_rec`: not "how many
instructions" but "what is it waiting on".
