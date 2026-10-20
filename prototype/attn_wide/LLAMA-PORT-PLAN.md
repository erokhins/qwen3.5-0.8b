# A13 — port llama.cpp's attention structure onto our wide form (funded 2026-10-20: "do as they did; fast and numerically sound; near-band numerics are fine")

Source read this session: `engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh` (2,175 lines) plus
`fattn-common.cuh`, and the actual compiled instance for this model,
`flash_attn_ext_f16<DKQ=256, DV=256, ncols1=16, ncols2=4>` — which is our shape exactly (`head_dim` 256,
GQA 4). All line numbers below were read in this pass, not recalled.

**What §58 already established, and why this is a patch and not a rewrite.** Their round bills at **51.9**
non-matrix issue slots per matrix slot, our wide form at **110.9**, our shipping form at **108.3**
(`R2-amortization-spec.md`). Their whole advantage is that number, and 2.09× against a measured 1.9× rate
gap is a closed account. Our wide form (`engine/attn_wide_kernels.h`, 440 lines) is *already* a partial
transliteration of theirs — WMMA scores, F16 P·V accumulator, fragment-local softmax, one `__shfl_xor`,
Q held in registers, `#pragma unroll` everywhere. So the work is not "adopt WMMA" (done) or "widen the
round" (done, §57 killed the residency half of that argument). **The work is the six things they do that we
do not, each of which maps onto a bucket §58 named.** The ISA canary for this whole plan is one number:
**the ported round must bill at ≤ ~70, target ≤ 55.** If it does not, the port failed and no device time
should be spent on it.

## The six divergences, in the order their ISA cost predicts

### 1. Template-parameter LDS strides + XOR swizzle, so no address is ever computed at run time

`load_ldmatrix_swizzled<stride_tile_K>` (`fattn-mma-f16.cuh:672`, `load_ldmatrix_trans_swizzled` at `:1032`);
`stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4` (`:1247`), i.e. **power-of-two when swizzled**, with the
swizzle folded into the lane index rather than into the address. Every LDS address in their round is
`base + const`, which is why their `ds_load_b128`/`ds_load_2addr_b32` cost almost no integer work.

**Ours:** `engine/attn_wide_kernels.h:120-123` deliberately runs an **unpadded, non-power-of-two stride**
("A11 §5 measured the unpadded stride and priced a pad at a few-percent lever, deliberately not taken"), and
`fattn_v_addr`-style layout math is recomputed at the use site. **This is the leading candidate for our
`int_vec` 30.0 per WMMA against their 18.1 and `salu` 21.8 against their 3.7.** Note the A11 decision was
made about *padding for bank conflicts*; the swizzle is a different lever and was never tried.

### 2. Stage K and V once per round into one region, and **prefetch the next K tile in the middle of the current round**

`nstages > 1`: the first K tile is `cp_async`'d before the loop (`:1310-1325`), and inside each iteration the
**next** K tile (and mask tile) is issued *after* the current score tile has been consumed and before P·V
runs (`:980-996`). So global latency for round *n+1* hides behind the matrix work of round *n*. Their round
bill shows they pay for this as ordinary issue slots (`lds` 9.8 %, `wait` 7.9 %) and win.

**Ours:** five `__syncthreads()` per round (`attn_wide_kernels.h:225, 312, 322, 342, 367`) and K/V staged at
the top of each round: publish K → barrier → score → barrier → publish V → barrier → P·V → barrier. Every
round therefore presents the global load latency of *this* round on the critical path, and A11 measured our
kernel at **65–74 % "waiting" with no pipe saturated**. Their structure is the direct answer to that
measurement. On RDNA there is no `cp.async`; the equivalent is issuing the next tile's `global_load`s into
registers early and committing them to LDS later (their `cp-async.cuh` is 57 lines and its AMD path is
exactly that).

### 3. Packed half rescale of the accumulator

`fattn-mma-f16.cuh:925-941`: when the VKQ accumulator is `half2` they multiply with `KQ_max_scale_h2`
(a `half2` broadcast) over `VKQ_C[i].x[l]`, and their round bill shows **448 `v_pk_mul_f16`** — two
accumulators per instruction. They also zero the scale branchlessly with
`*((uint32_t *)&KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD` (`:903`).

**Ours:** `attn_wide_kernels.h:298-302` broadcasts a scalar `_Float16 rh = rescale` and multiplies
element-wise; if the compiler does not pair those into `v_pk_mul_f16` we are paying 2× their rescale cost
for the same math. This is cheap to check and cheap to fix (`__half2` storage, one multiply per pair).

### 4. Bounds and masking specialized out of the steady-state loop

Their loop calls the iteration twice — `last_iter=false` with `oob_check=false` for every full tile, then a
specialized `last_iter=true` call with `k_VKQ_sup` clamping the live lanes (`:1350-1388`). Inside the
steady-state body there is no per-cell bounds test at all; the mask, when present, is an LDS `half` tile
**added** (`+= __half2float(tile_mask[...])`, the `RDNA3` branch at `:818-823`), not a select.

**Ours:** `attn_wide_kernels.h:251-252` computes a per-cell liveness predicate (`j < rlanes`) each round, and
our round bills `select_mask` at 9.6 per WMMA against their 2.6. Making the full-tile body predicate-free and
leaving the comparison to the tail round is a mechanical change with the same shape as theirs.

### 5. Fold the softmax scale into Q once, at staging

`:1268-1272` — `tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y)`. The score loop then
contains no scale multiply for any of the ~2,000 rounds.

**Ours:** the scale is a kernel constant (`SCALE`, `attn_decode_kernels.h`) — this plan must verify at the use
site whether it is applied at Q staging or per score element, and fold it if it is not. (Verification was not
finished this pass; treat as a 10-minute check, not a design item.)

### 6. Q-in-registers and V-chunking: already ours, keep

`Q_in_reg` (`:1298-1303`), V loaded in `i0_start` chunks that re-use one LDS slice
(`:998-1050`). Our wide form already keeps Q in registers (our own note at `:205` says the shipping form
re-reads Q at 34.85 B per causal touch, 898 GB per pass at 32,768 lanes) and stages 64 positions per round.
**No action — recorded so a reader does not "port" what is already ported.**

## Numerics: what the user authorized, and what still has to be proved

The user's words: *"if it is almost passing the gates because we changed numerics — it is still fine by me."*
Recorded as the band authorization for this item. Two parts of the port change numerics: the score/P path in
f16 (already true of A12, which is red on `smoke` at pooled 1-cos **2.028e-04** against a **2e-04** band),
and item 3 (packed f16 rescale) if the accumulator's rescale path changes association.

The authorization does **not** remove the repo's own test of a widened band: the new band must still catch
**every injection in the must-catch set derived from geometry** (`prototype/attn_tile_model.py`'s
`wide_case`/`neg_expect` discipline, `mlp_gemv --negative`). So the admission step for A13 is: (a) re-take
the A12 numerics suite at the new band, (b) re-run the derived negative controls and show they still fire,
(c) record the band beside what witnesses it. If the only way to pass is to loosen a check that currently
catches a real bug, that is reported, not routed around — the standing rule survived the authorization.

## Sequencing, and who runs what

1. **New prototype header, shipping path untouched**: `engine/attn_wide_llama_kernels.h` + a knob
   (`--attn-fattn-wide 2`) so the existing A12 gate rig can judge both forms in one run. Worker.
2. **Items 1, 2, 4 first** (addressing, staging/prefetch, bounds specialization) — they are the ones §58's
   bucket table prices. Worker.
3. **ISA canary before any timing**: carve the built binary, bill the round, require ≤ ~70 and report the
   per-bucket deltas against §58's table. CPU-only, so a worker does it (`tools/carve_codeobjects.py` +
   `g3_isa_bill.py` + `a2_round_bill.py`, all existing).
4. **Device**: correctness gate by the worker if it fits under a minute, otherwise handed to the main agent;
   **all timing by the main agent, blocking, zero subagents, one lock**. Compare paired against
   `--attn-fattn-wide 0` and `1` at 16,384 and 32,000 lanes in a single window (the §57 lesson: a ruler taken
   in another window is not an A/B).
5. Then, and only then, the numerics band question, then the default.

**Prize, restated so it stays honest**: attention is ~40 % of the wave at 16k and their family rate is 1.9×
ours, so a faithful port is worth on the order of **−15…−20 % of the prefill wave at long context**, not a
2× engine speedup. §58's ratio is the gate on that claim: a port that bills > ~70 cannot reach it, and the
ISA census will have said so before a single millisecond is spent.
