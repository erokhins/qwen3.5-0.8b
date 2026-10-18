# llama.cpp's tiled MMA attention, dissected on this box — the price sheet item 2.6 asked for

Analysis-only. **No timing was taken and no device measurement made here**: every number is (a) read
out of a built artifact on disk (`engines/llama.cpp/build-hip/bin/…`, carved and disassembled on the
CPU), (b) read out of the §22 trace of llama.cpp in `bench/performance-status.md`, already on disk
(`tmp/d52/llama_pp4096/pp4096_kernel_trace.csv`, host-side reuse), (c) quoted from
`engines/llama.cpp/ggml/src/ggml-cuda/`, or (d) arithmetic on those. The one host-side driver query
is disclosed in §0.3. Nothing under `engine/`, `bench/`, `tools/` or `engines/` was modified; scripts
live in `tmp/gcc/fattn/`.

Why it exists: §22 of `bench/performance-status.md` measured llama.cpp's attention family at
**≤ 52.5 ms** against our **263.2 ms** at matched conditioning — "we are ≥ 5.0× slower" — and named
that gap queue item 2.6 with external evidence. The precondition for funding 2.6 was that their path
first become a *priceable design*. This is that price sheet, in the order asked: what class their
attention runs on here (§1), what the design is (§2), how it maps onto our geometry and what the
honest multiplier is (§3), what to build first (§4). Tool findings are §5, because one of them
retires a constant that the last three price estimates in this repo leaned on.

## 0. Provenance, and the canaries

### 0.1 The artifact and the instantiation

The HIP build does **not** emit `libggml-cuda.so`; the file is
`engines/llama.cpp/build-hip/bin/libggml-hip.so.0.27.0` (finding F1). `tools/carve_codeobjects.py`
yields **140** `gfx1151` code objects into `tmp/gcc/fattn/co/`; the one that defines the traced
instantiation is **`co_22355968.co`**, from the TU
`fattn-mma-f16-instance-ncols1_16-ncols2_4.cu`.

**Canary 1 (object identity) — PASS.** The object carries `Machine: EM_AMDGPU`, `Flags: 0x4a`, and the
amdhsa note `target: amdgcn-amd-amdhsa--gfx1151` with **no** `wavefrontsize64` suffix (a wave32
object), and its symbol demangles to
`void flash_attn_ext_f16<256, 256, 16, 4, false, false, false>(char const*, … )`
(raw note block `tmp/gcc/fattn/meta_notes.txt:3205`; the `Kernel_Name` field of the §22 trace in
`bench/performance-status.md` carries the
same demangled text, so trace and object name the same function — the template arguments are read
from the trace, not inferred).

Those arguments are this model's head geometry: `DKQ = DV = 256` is `head_dim 256`, and
`ncols1 = 16, ncols2 = 4` with `ncols = ncols1·ncols2 = 64`, where `ncols2` is the GQA group packed
into one tile and this model's GQA group is `n_q / n_kv = 8 / 2 = 4`
(`specification/00-notation-and-hyperparameters.md:38-40,81`). **The configuration being priced is the
one llama.cpp picks for our exact head geometry on this exact GPU**, which is what makes the anatomy
transferable rather than merely interesting.

### 0.2 Driver-reported launch shape, from the trace already on disk

`python3 tmp/gcc/fattn/trace_row.py tmp/d52/llama_pp4096/pp4096_kernel_trace.csv 'void flash_attn_ext_f16<256, 256, 16, 4'`

```
instances: 96
  LDS_Block_Size 0   Scratch_Size 900  VGPR_Count 256  Accum_VGPR_Count 0  SGPR_Count 128
  Workgroup 32 x 8 x 1 (256 threads)        Grid 2048 x 8 x 1
distinct (gridx,gridy,gridz,wgx,wgy) shapes: 1
traced durations us: min 244.7 med 1091.1 max 3080.0 sum 104.976 ms; /2 pass = 52.488 ms
```

* `Grid_Size_*` is in **threads**, not blocks — calibrated against our own trace in the same window
  (`tmp/gcc/fattn/trace_calib*.py`: our `rec_fused_kernel` reports `Grid_Size_X = 2048` for a
  16-block × 128-thread launch). So their launch is `gridDim = (64, 1)`: **64 workgroups of 256
  threads**, 1.6 blocks per CU on 40 CUs — a thin grid, not a saturated one.
* Σ/2 = **52.488 ms/pass** matches the "≤ 52.5 ms" of §22 in `bench/performance-status.md`, to the last
  digit: at that conditioning the attention family *is* this one kernel, 48 launches/pass (§26 W4 of
  `bench/performance-status.md` counts 138/pass across the family, so the other 90 are auxiliaries). The
  ÷2 is §22's own convention (the trace holds warmup + timed).
* `LDS_Block_Size = 0` does **not** mean no LDS: their LDS is dynamic (`extern __shared__ half2
  tile_Q[]`, `fattn-mma-f16.cuh:1250`) and the code object's `group_segment_fixed_size` is likewise 0,
  so **this kernel's LDS footprint is invisible in every trace column we have** and must be derived
  from source constants (§2.4). Finding F3.
* `VGPR_Count 256` and `Scratch_Size 900` agree with the code object's own metadata (§1.1/§2.5), which
  is the cross-check that two independent sources say the same thing. The one disagreement is
  `SGPR_Count`: 128 traced vs 104 in the ELF (driver granularity).

### 0.3 The one host-side driver query (disclosure)

`tools/gpu_lock.sh acquire fattn-anatomy-regfile`, then `./tmp/gcc/fattn/regfile_probe`: wall **0.21 s**
(build 0.7 s), `tools/gpu_lock.sh util` **89 %** before / **69 %** after, **no timing taken**, no
kernel launched, no allocation, one worker active. It calls only `hipSetDevice`,
`hipGetDeviceProperties`, `hipFuncGetAttributes`, `hipOccupancyMaxActiveBlocksPerMultiprocessor` — the
same host-side calculator `tools/kernel_occupancy_probe.cpp` uses, whose header is this repo's
precedent for treating that as not a device measurement. Output `tmp/gcc/fattn/regfile_probe.txt`.

* **Good:** confirms `MaxRegistersPerMultiprocessor = 196608`,
  `MaxSharedMemoryPerMultiprocessor = 65536`, `MaxThreadsPerMultiProcessor = 2048` in HIP's halved-CU
  view (the halving is `prototype/attn_wmma/evidence/device_facts.txt`'s own finding), and confirms the
  shape question that matters: **a 256-thread block with 36,096 B of dynamic LDS returns 1 block/CU =
  8 wavefronts/CU.** That is the driver-authority witness for §2.4.
* **Failed, and its rows must not be used:** the register-pressure stubs all returned `numRegs = 5`,
  so the probe's own canary prints `FAIL (registers not visible)` — an empty-asm VGPR fence does not
  create live ranges. **No register-file budget is claimed from this probe**; §3.4 gets it from driver
  outputs already on disk.

## 1. The instruction class their attention actually runs on this box

### 1.1 The census

`bash tmp/gcc/fattn/disasm.sh` → `tmp/gcc/fattn/fattn_mma.s` (171k lines), then
`python3 tools/kernel_isa_census.py` over the carved objects (`tmp/gcc/fattn/census_fattn_all.txt`).
Traced kernel: 19,512 instructions; **MATRIX 384 = 192 `v_wmma_f32_16x16x16_f16` + 192
`v_wmma_f16_16x16x16_f16`**; **`v_mfma_*` 0**; **`v_dot*` 0** (the census's DOT bucket is 864
`v_fma_mix_f32` — a bucket-name collision, finding F5); FMA32 833; **LDS 2,084** (10.7 % of the
kernel — this is a fragment-staging kernel as much as a matmul kernel); VMEM 402; SMEM 0; SALU 4,872.

So: **they run on the RDNA3 WMMA tensor pipes, f16 operands, in two accumulator widths** —
`…_f32_16x16x16_f16` for the score product and `…_f16_16x16x16_f16` for P·V, i.e. **the P·V
accumulator is f16** (§2.3). Our side for contrast: §26 of `bench/performance-status.md` counted
**zero matrix instructions in all 40 of our kernels**, so the class difference is total, not partial.

### 1.2 What a round is, read out of the loop structure

Back-edges must be read from llvm's `<symbol+0xNN>` annotation, not the printed displacement, which is
an unsigned imm16 (a −20 branch prints as `65516`); the first version of `tmp/gcc/fattn/loop_census.py`
found **0** backward edges for exactly that reason (finding F2). With the annotation,
`tmp/gcc/fattn/round_body.py` finds exactly **three** loop bodies containing exactly 64 matrix
instructions, sized 1,683 / 1,706 / 1,790 instructions. That closes the source reading:

* `flash_attn_ext_f16_process_tile` is instantiated three times — the `needs_fixup` arms at `:1939`
  and `:1944`, and the fix-up pass at `:1988`;
* each has one rolled KV loop (`for (; kb0 < kb0_stop-1; ++kb0)`, `:1352`, `:1372`) **plus a peeled
  last iteration** (`:1363-1369`, `:1380-1389`) ⇒ 2 round copies each ⇒ 6, and
  384 static matrix ops ÷ 6 = **64 per round**, exactly what the fragment loops predict (§2.2).

Per round, per warp (the 1,683-instruction body; SIMT, so this is per-warp):

```
v_mov_b16_e32 249   s_waitcnt 182  v_mov_b16_e64 136  ds_load_2addr_b32 128  s_delay_alu 116
v_mov_b32_e32 65    ds_load_b128 64  v_pk_mul_f16 64   v_cndmask_b32_e64 48   v_cndmask_b32_e32 47
v_wmma_f32_16x16x16_f16 32          v_cvt_f16_f32_e64 32      v_wmma_f16_16x16x16_f16 32
v_exp_f32_e32 17  v_ldexp_f32 17  v_rndne_f32_e32 17  v_cvt_i32_f32_e32 17  v_mul_f32 15
ds_bpermute_b32 17  v_cmp_ngt_f32_e32 17  v_fma_f32 17  v_sub_f32_e32 18
scratch_load_b32 18  scratch_load_b128 18  global_load_b128 16  ds_store_b128 16  ds_load_u16_d16 16
v_max3_f32 8   v_max_f32_e32 2   (plus ~150 address ops)
```

Every line reconciles with the source: 16 `global_load_b128` × 16 B × 32 lanes × 8 warps = 65,536 B =
one K tile **plus** one V tile per block-round (§2.4 sizes both at 33,792 B with padding); 64
`v_pk_mul_f16` = the 8 VKQ tiles × 8 half2 of the rescale at `:938-941`; 17 `expf`'s (16 scores + the
one `KQ_max_scale` at `:902`); 8 `v_max3_f32` + 2 `v_max_f32_e32` for the 16-value row max; and 36
`scratch_load_*` per round, which is §2.5's spill traffic arriving in the hot loop.

### 1.3 Cross-lane ops per softmax row — the direct measure against our 5-step butterfly

Whole kernel: **`ds_bpermute_b32` × 111 and nothing else** — no `permlane_*`, no `ds_swizzle`. On this
ISA every `__shfl_*_sync` lowers to `ds_bpermute_b32`, so a census filtering for `permlane`/`shfl` and
missing `ds_bpermute` silently reports zero cross-lane work; `tools/kernel_isa_census.py:37` does
include it, and so does `prototype/attn_valu_slim/isa_census.py`, which is why our own fold counts
have been right.

**State the denominator, because there are two and they differ by 16×.** Per round body (per warp) the
17 cross-lane ops cover 512 scores (32 KV rows × 16 Q columns, §2.2) = **0.033 cross-lane ops per
(query, head) × KV-lane score**; per softmax row per round it is **1**. The split, source-exact:

* **1 per row per round — the fold.** `fattn-mma-f16.cuh:861-873`:
  ```c++
  #elif defined(AMD_WMMA_AVAILABLE)
      // Values per KQ column are spread across 2 threads:
      constexpr int offset_first = 16;
      constexpr int offset_last  = 16;
  …
      for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
          KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
      }
  ```
  `offset_first == offset_last` ⇒ the loop body runs **exactly once**. Why one suffices is a
  fragment-map fact (`mma.cuh:183-215`): a `tile<16,16,float>` holds `ne = I*J/32 = 8` values per lane
  at `get_i(l) = threadIdx.x % 16` and, for the C tile on RDNA3, `get_j(l) = 2*l + threadIdx.x/16` —
  **a lane's whole row of scores sits in its own registers**, so a column needs exactly one partner
  lane. The other 8 maxes are serial VALU (`fmaxf` at `:846`, lowered to 8 `v_max3_f32`).
* **16 per round — the price of the P fragment, not of the softmax.** `mma.cuh:729-743`, the RDNA3
  `get_half2`, does one `__shfl_xor_sync(0xFFFFFFFF, tile_float.x[l], 16, WARP_SIZE)` **per element**
  (`mma.cuh:739`), called once per KQ C tile at `fattn-mma-f16.cuh:972-977`
  (`B[k] = get_half2(KQ_C[k])`, `nbatch_fa/(np*2*T_B_VKQ::J) = 2` tiles) ⇒ 2 × 8 = 16. This is the f32
  C fragment being re-laid into the half2 B fragment P·V needs. **Their fold is one op per row; they
  then spend 16 ops per round on fragment geometry.**

Ours, same unit: D61 of `progress/2026-10-16/prefill-decision-log-2026-10-16.md` names the wall as a
dependent `dot → 5-step butterfly fold → rescale` per live pair, 16 live pairs per warp, nothing to
interleave; `prototype/attn_valu_slim/results.md` §2 bills the fold at `fold.exec-guard` 16 +
`fold.addr-math` 4 slots per touch plus 5 `fold.permute` slots which it marks "**not removable**" — **5 of
them cross-lane ops, one per butterfly step**. So the comparison is 5 cross-lane ops per score and one
fold per score against their 0.033 per score and one per row per 32 KV rows. That is the largest
structural difference between the two shapes, and it is not the matrix instruction.

## 2. The design, with cites

All paths are under `engines/llama.cpp/ggml/src/ggml-cuda/`.

### 2.1 How the configuration is chosen, and what `256, 256, 16, 4` means

The template arguments are `<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse>`, and
the trace says `<256, 256, 16, 4, false, false, false>` — softcap off, K/V distinct, sparse off.
`ncols = ncols1·ncols2 = 64` (`:606`) and the config table is keyed on `ncols`, not on `ncols1` — the
third macro argument (`fattn-mma-f16.cuh:26`) — which is why the matching row is

```c++
164:    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 2,  64, 128, 128,  64, 1, true);
```

selected by the `amd_wmma_available(cc)` branch (`:245`, `ggml_cuda_fattn_mma_get_config_rdna`), because
`vendors/hip.h` maps `__GFX11__` → `RDNA3` and `__gfx1151__` → `RDNA3_5`, defining
`AMD_WMMA_AVAILABLE` and **not** `AMD_MFMA_AVAILABLE`. Fields against their declarations (`:11-19`):

| field | value | meaning, from the source's own comment |
|---|---|---|
| `nthreads` | 256 | threads/block ⇒ **8 warps of 32** (trace: workgroup 32×8) |
| `occupancy` | 2 | the `__launch_bounds__` second argument (`:1797`) — unachievable here, §2.5 |
| `nbatch_fa` | 64 | "Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators" (`:13`) — **one rescale per 64 KV rows**, not per lane |
| `nbatch_K2` | 128 | "Number of K half2 values in direction of DKQ to load in parallel" (`:14`) = DKQ/2 ⇒ **one K stage is a whole 256-dim row**; with `swz = false` it sets `stride_tile_K = nbatch_K2 + 4 = 132` (`:617`, `:1246`) |
| `nbatch_V2` | 128 | same for V ⇒ the `i0_start` loop at `:1001` (`i0_start += 2*nbatch_V2`) runs once over DV |
| `nbatch_combine` | 64 | width of the epilogue combine; sizes `nbytes_shared_combine` |
| `nstages_target` | 1 | but `ggml_cuda_fattn_mma_get_nstages` = `cp_async_available(cc) && ncols2 >= 2 ? target : 0` (`:365-366`), and HIP has no `CP_ASYNC_AVAILABLE` ⇒ **nstages = 0** |
| `Q_in_reg` | true | **Q lives in registers for the whole kernel** (`:1255`, filled once at `:1326`) |

Two consequences that matter for us: the device-side `ggml_cuda_fattn_mma_get_swizzled` is
`#ifdef TURING_MMA_AVAILABLE … #else return false` (`:353-360`), so on HIP **`swz = false`** and every tile
stride is `nbatch_* + 4` (`:617`, `:1246`); and `nstages = 0` makes `tile_V` **alias** `tile_K` (`:1252`,
`half2 * tile_V = nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;`).

### 2.2 Tile shapes, and the fragment map that was an assumption for us and a citation for them

`fattn-mma-f16.cuh:1119-1126`:

```c++
1119: template<int DKQ, int ncols> struct mma_tile_sizes {
1120:     using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
1121:     using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
1122:     using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
1123:     using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
1124:     using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
1125:     using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
```

D58 of the decision log recorded that our `m16_bf16frag` probe needed an **assumed** fragment map;
`mma.cuh:183-215` is a published, shipping map for this tile on this architecture (`ne = I*J/32` at `:183`,
`get_i = threadIdx.x % 16` at `:195`, C-tile `get_j = 2*l + threadIdx.x/16` at `:207`, A/B `get_j = l` at
`:210`), and the per-lane fragment payload it implies is asserted rather than guessed: `sizeof(t.x) == 16`
for `tile<16,8>` (`mma.cuh:835`, four 4-byte half2 registers) and `sizeof(t.x) == 32` for `tile<16,16>`
(`mma.cuh:867`).

Block geometry, `:605-609`:

```c++
605:    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();   // 32 on this object
606:    constexpr int  ncols           = ncols1 * ncols2;                      // 64
607:    constexpr int  cols_per_warp   = T_B_KQ::I;                            // 16
608:    constexpr int  cols_per_thread = get_cols_per_thread();                // 1  ("AMD has a single column per thread", :332-337)
609:    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // 8*16/64 = 2
```

A block covers **16 query positions × 4 GQA heads = 64 columns** and **64 KV rows per round**, with
`np = 2` warps sharing each column group: each warp owns 16 Q columns × 32 KV rows = **512 scores per
warp-round**, each lane owning one column and 16 of its KV values (2 tiles × `ne` = 8).

Per-round matrix work, which is what §1.2's 64 matrix ops are:

* score: `for i_KQ_00 in nbatch_fa step np·T_A_KQ::I` (2, `:667`) × `for k_KQ_0 in DKQ/2 step
  T_A_KQ::J` (16, `:670`) = **32 `v_wmma_f32_16x16x16_f16`** (the call sites at `:674`, `:679`, `:682`)
  accumulating into 2 `KQ_C` tiles at depth 16;
* P·V: `for i_VKQ_0 in DV step T_A_VKQ::I` (8, `:1021`) × `for k00 in nbatch_fa/2 step np·T_A_VKQ::J`
  (2, `:1024`) = 16 sites × 2 instructions because `T_A_VKQ` is the wide `tile<32,8>` = **32
  `v_wmma_f16_16x16x16_f16`** into 8 accumulators at depth 4 (`:1030`).
* MAC check: (32 + 32) × 16·16·16 = 262,144 = 512 scores × 512 MACs ✓.

### 2.3 Accumulator registers per lane — the direct answer to D30's question

D30 asked whether "16 f32 accumulators over DH=256 = 128 VGPR/lane" is affordable. Their answer, from
the declarations:

| what | source | tiles | per tile per lane | VGPR/lane |
|---|---|---|---|---|
| `KQ_C` (scores, f32) | `:624` `T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)]` | 2 | 8 f32 | **16** |
| `VKQ_C` (O, **half2**) | `:1125`, rescale loop bound `(DV/2)/T_C_VKQ::J` at `:938` | 8 | 8 half2 | **64** |
| `Q_B` (Q resident) | `:1255` `T_B_KQ Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)]`, filled at `:1326` | 16 | 4 half2 | **64** |
| named live total | | | | **144** |

**They hold the same 128 accumulator values per lane that D30 costed at 128 f32, for 64 VGPR, because
the C tile is half2** — and the rescale is then `v_pk_mul_f16` (`:936-941`, 64 per round, measured in
§1.2). That is the escape from the 128-VGPR bill, and it is bought with f16 accumulation of O (§4.3).

### 2.4 LDS bytes per block, what is staged, and the occupancy that follows

Derived from the host-side sizing at `:2031-2041` with `swz = false`, `V_is_K_view = false`,
`nstages = 0 ≤ 1` (`sizeof(half2) = 4`):

| region | formula (line) | value |
|---|---|---|
| KV, 1 stage | `nbatch_fa · max(stride_K, stride_V)` = 64 × 132 (`:2033`) | 33,792 B |
| mask | `ncols1 · (nbatch_fa/2 + 4)` = 16 × 36 half2-sized slots (`:2036`) — 4 B per (query row, KV pair) | 2,304 B |
| Q | `ncols · (DKQ/2 + 4)` = 64 × 132 (`:2035`) | 33,792 B |
| epilogue combine | `nwarps · cols_per_warp · (nbatch_combine + 4)` = 8·16·68 (`:2037`) | 34,816 B |
| **total** | `max(combine, Q_in_reg ? max(Q, KV+mask) : …)` (`:2041`) | **36,096 B** |

The K tile holds one full 64-row × 256-dim K stage and V aliases the same bytes (`:1252`), which is
why 16 `global_load_b128` per warp-round (65,536 B per block-round) covers both. **65,536 / 36,096 = 1.8
⇒ one block per CU, 8 wavefronts/CU = 4 per SIMD = 25 % of the 32-wavefront cap**, against our shipped `attn_slim_wave` at **5 blocks = 20 wavefronts/CU = 62.5 %**
(`prototype/attn_wmma/evidence/occupancy.txt`). Witnessed by the driver for the shape (§0.3: 256
threads + 36,096 B dynamic LDS → 1 block/CU). **Derived, not traced** — F3 says why the trace cannot
confirm it.

### 2.5 Launch bounds, and the register posture that follows from an unachievable target

`:1797` is `__launch_bounds__(…get_nthreads(…), …get_occupancy(…))` = `__launch_bounds__(256, 2)`. The
code object metadata says `vgpr_count 256`, **`vgpr_spill_count 358`**, `sgpr_count 104` with 0 SGPR
spills, `private_segment_fixed_size 900`, `max_flat_workgroup_size 256`
(`tmp/gcc/fattn/meta_notes.txt:3205-3213`). Two readings:

* 256 VGPR is the **architectural per-thread ceiling**, so the 358 spilled registers are not the
  occupancy target being honoured — they are the ceiling being hit. A fattn-shaped tile at DH = 256
  genuinely does not fit in one wavefront's register budget.
* The spill traffic is *in* the round: 36 `scratch_load_*` per round body (§1.2), 900 B private per
  thread = 230 KB per resident block. Anyone copying this design copies a scratch-using inner loop.

### 2.6 One round's dependent chain, and where the ILP comes from

Per warp, per round, the source-shaped chain:

```
stage K (+V/mask) global → LDS      :636-661 (plain copies + __syncthreads; nstages = 0 ⇒ no cp.async)
score  = 32 WMMA, 2 chains deep 16  :667-684     ── accumulator-chain depth is the long pole
row max = 8 serial fmaxf/lane       :846         (8 × v_max3_f32 + 2 v_max_f32_e32)
        + 1 shfl per row            :871-872
P       = 16 expf, mutually independent          :888   (17 expf bodies incl. :902)
        + 16 shfl + 8 packs (C → B fragment)     mma.cuh:739, :972-977
rescale = 64 v_pk_mul_f16, all independent       :936-941   (once per 64 KV rows, not per lane)
O       = 32 WMMA into 8 independent chains      :1021-1041
```

**Independent accumulator chains per warp: 2 (KQ, depth 16) + 8 (VKQ, depth 4) = 10.** With 4 warps per
SIMD that is 40 independent chains feeding one matrix pipe per SIMD, so the round is throughput- rather
than latency-shaped: at the WMMA occupancy cost of 32 clk at the roof / 36.0 clk at this box's measured
sustained rate (`prototype/attn_wmma/m16_census.output.txt:91`), 4 warps × 64 WMMA = 256 WMMA × 32 clk
= 8,192 matrix-pipe cycles per SIMD-round against 4 × 1,619 ≈ 6,476 non-matrix issue slots (at one slot
per cycle per SIMD) — the two pipes within 27 % of each other, which is what a balanced tile looks like.

Two things the design does **not** have, so nobody has to wait for them: no async copy and no TMA
(nstages = 0 on HIP), and no `v_mfma_*` (RDNA3 has none). **The absence of cp.async on our HIP path is
not a blocker for copying this** — the shipped design being priced already runs without it.

## 3. Onto our geometry, and the honest 5×

Our facts as the ledger holds them: `DH = 256`, GQA `G = 4`, `MAXSC = 512` lanes/block, 40 CUs, 2
SIMDs/CU, 64 KiB LDS/CU, `max_waves_per_simd = 16` (⇒ 32 wavefronts/CU real, not HIP's 64);
`attn_slim_wave` at 47 VGPR / 12,288 B LDS / 5 blocks / 20 wf/CU
(`prototype/attn_wmma/evidence/occupancy.txt`), **1,983 instructions with 130 f32-FMA-class and zero
matrix instructions** (`bench/performance-status.md:1184`), and **89.9 warp-instructions
per touch** measured (§26 W2 of `bench/performance-status.md`), 116 static slots/touch
(`prototype/attn_valu_slim/results.md` §2).

A "touch" is one `(row, lane, head)` score (`prototype/attn_valu_slim/results.md` §2; 8 MACs/lane
because the 256-dim dot is spread across 32 lanes), which is the same unit as their score: one
(query, head) × one KV lane, 512 MACs. Both currencies below divide by the same thing.

### 3.1 Currency A — issue slots per touch

| | warp-inst per touch | of which matrix | of which cross-lane |
|---|---|---|---|
| ours, shipped, measured (§26 W2 of `bench/performance-status.md`) | **89.9** | 0 | 5 |
| theirs, steady-state round body (§1.2), raw | **3.29** | 0.125 | 0.033 |
| theirs, same body with a WMMA billed at its pipe time (32 clk, `m16_census.output.txt:91`) | **7.16** | 4.0 clk of matrix pipe | 0.033 |

Cut factor **27× raw, 12.6× with the matrix instruction billed honestly**. (AGENTS.md's "always ask a
slot figure which denominator it divided by" applies in full: a raw count bills a `v_wmma` and a
`v_fma_f32` at one slot each, and they are not the same purchase.)

Now through the one elasticity this repo has actually measured. AGENTS.md records the datum: the
bit-identical cut of **116 → 45 issue slots per touch** made `attn` **1.521× faster**, so
time ∝ slots^α with α = ln 1.521 / ln(116/45) = **0.44**. Applying it to `attn` = 264.71 ms of a
763.49 ms wave (§26 of `bench/performance-status.md`):

| instruction cut | predicted `attn` cut | `attn` becomes | wave becomes | prefill gain |
|---|---|---|---|---|
| 2.58× — **the datum itself** (116 → 45) | 1.52× (measured) | 174 ms | 673 ms | **+13 %** |
| 2.37× — D57's dot-only price (pass-1 45 → 19) | 1.46× | 181 ms | 680 ms | **+12 %** |
| 12.6× — pipe-weighted copy of their round | 3.05× | 87 ms | 586 ms | **+30 %** |
| 27× — raw instruction count | 4.28× | 62 ms | 561 ms | **+36 %** |
| their measured ratio ≥ 5.04× (§22 of `bench/performance-status.md`) | 5.04× | 52.5 ms | 551 ms | **+39 %** (ceiling) |

The model's second row **reproduces the +5 % to +12 % D57 booked** for the dot-only substitution, which
is the witness that α = 0.44 is the right bridge and not a fitted free parameter. Three honest caveats
printed where they belong: α was measured over a 2.58× range and is being extrapolated to 12–27×; the
datum's x-axis is the *static* slots/touch census (116 → 45) while this section's ratio is the *dynamic*
counter-calibrated warp-inst/touch (89.9 → 3.29/7.16), so the two scales are the same quantity measured
two ways and the exponent inherits that mismatch; and D57 priced the matrix class at **+5 % to +12 % of
the wave** because it counted only the ~27 of 45 pass-1 slots a WMMA can subsume, whereas this anatomy
says the replaceable object is the whole round — the fold (1 per row vs our 5 per score), the per-lane
rescale (theirs once per 64 KV rows), the Q re-read (they hold Q in registers), and the GQA×4 packing.
**That is a change to D57's price and it is the user's call, not mine**; §4.5 flags it.

### 3.2 Currency B — cycles per dependent chain at realized occupancy

Method: `tools/kernel_occupancy_probe.cpp`'s host-side calculator (driver-authority), plus the
32-clk/WMMA pipe cost from `prototype/attn_wmma/m16_census.output.txt:91`.

| | wavefronts/CU | independent accumulator chains per warp | critical chain per round | binding resource |
|---|---|---|---|---|
| theirs | **8** of 32 (25 %, LDS-bound at 36,096 B/block) | **10** (2 KQ + 8 VKQ) | 16 WMMA deep (KQ) ≈ 16 × 32 clk, then 8 serial `fmaxf` + 1 `ds_bpermute` + 16-way-independent `expf`, then 4 WMMA deep × 8 | matrix pipe ≈ 8,192 clk/SIMD-round vs ≈ 6,476 non-matrix issue slots — balanced |
| ours, shipped | **20** of 32 (62.5 %, LDS-bound at 12,288 B/block) | 1 shared rescale over 16 live pairs | dot (8 dependent FMAs) → **5-step butterfly** → rescale, per live pair, nothing to interleave (D61) | one pipe, all ALU/LDS, 89.9 slots/touch |

What this says, and it is the finding: **their win is not occupancy — they have a quarter of ours and
half the interleave, and still run ≥ 5× faster.** The GRP2 parity result of §28 in
`bench/performance-status.md` ("+66 % wavefronts moved time by ≤ 5 %") already said occupancy is not the
axis; this is the external witness from the other direction. The axis is what D61 named: a per-score
dependent chain with a 5-step cross-lane fold in it, versus a round whose chain is short (one fold op per
row per 32 KV rows) and whose 10 accumulator chains are software-pipelined by the fragment layout.

What currency B **cannot** be priced with today, stated as a gap: the repo has the sustained *rate* of
`v_wmma_f32_16x16x16_f16` (52.63 TFLOP/s, `prototype/issue_rate_probe/results.md` §1) and a derived
occupancy cost (32 clk at the roof), but never a dependent-chain *latency* for it. Until a WMMA latency
leg exists in `prototype/issue_rate_probe`, currency B is a balance argument, not a cycle count. That
leg is compile-plus-one-burst work and belongs to the main agent.

### 3.3 What we can and cannot copy on `gfx1151`

Can, now, with evidence:

* the class: `v_wmma_f32_16x16x16_f16` / `v_wmma_f16_16x16x16_f16` are present, used by a shipped object
  on this box, and sustain 52.63 TFLOP/s here. Our D58 `m16_bf16gq` already compiles WMMA forms at
  **1.98 slots/touch, 30 VGPR, 11,264 B, 5 blocks/CU**
  (`prototype/attn_wmma/m16_census.output.txt:78,127`).
* the fragment map: `mma.cuh:183-215` — D58's "assumed" map is now a citation plus a shipped witness.
* no async copy needed (§2.6).
* GQA `ncols2 = 4` is our `G = 4` and DKQ = DV = 256 is our DH: **same head geometry**, so the tile shape
  transfers without re-deriving anything.
* register posture: a 256-VGPR × 8-wavefront block is placeable here (§3.4), so our 128-VGPR accumulator
  worry is not a placement wall at 128 threads either.

Cannot copy, or must be paid for:

* **their block geometry is not ours.** A 16-query × 4-head block with a 64-KV-row round and 36 KB of
  LDS is a different block from `MAXSC = 512` lanes at 12,288 B; copying it moves every address
  expression in the wave path — §20.1 of `prototype/attn_decode/results.md` (two self-consistent
  layouts joined by a branch are invisible to both sides) is the failure mode to pre-empt, and its
  lesson stands: one address expression shared by writer and reader.
* **the C→B fragment conversion is 16 cross-lane ops per round** (`mma.cuh:739`) plus ~385
  `v_mov_b16_*` packing ops — 23 % of their round is fragment relabelling. Not incidental overhead we
  can engineer away; it is what RDNA3's f32-C → half2-A relayout costs.
* **their spill posture** (358 spilled VGPRs, 900 B/thread private, 36 scratch loads per round) is a
  consequence of DH = 256 at 256 threads with Q resident. We inherit it if we copy the same shape at the
  same width; §4.2 is the one place we may do better.
* **`expf` is libdevice's accurate form here** (`v_exp_f32_e32` + `v_ldexp_f32` + `v_cvt_i32_f32` +
  `v_rndne_f32`, ~5 ISA ops, 17 per round). Per the "a hoisted scalar constant is a different number"
  trap, any CPU mirror of a copied round must transcribe the shipped expressions, including
  `FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)` (`fattn-common.cuh:19`) — a decimal approximation of 3·ln2, so a
  mirror that uses the exact constant differs from the kernel by ~1e-4 on every P.

### 3.4 A register-file note that retires an inference this repo has been carrying

`tools/kernel_occupancy_probe.cpp`'s last line prints "47 VGPR (granularity 8) allows %d waves/SIMD"
with the value computed as `512 / 48` — a hard-coded 512-lane-slot-per-SIMD file, never a query. The
driver disagrees with that constant in two places already on disk:

* `attn_slim_wave` returns **5 blocks/CU** from the occupancy calculator
  (`prototype/attn_wmma/evidence/occupancy.txt`). A 512-lane-slot file would cap a 47-VGPR, 128-thread
  block at 2 blocks/CU, so 5 was never reachable by registers; the 5 is the **LDS** bound (65,536 /
  12,288); the note "the two caps are independent … 47 VGPR allows 10 waves/SIMD" in §26 of
  `bench/performance-status.md` attributes the coincidence to the wrong cap.
* `m16_bf16rt` returned **3 blocks/CU** at 235 VGPR × 128 threads
  (`prototype/attn_wmma/evidence/occupancy-m16.txt`) = 90,240 lane-VGPR per CU, which a 32,768-slot
  (512 × 32 × 2) file cannot hold; and HIP reports `MaxRegistersPerMultiprocessor = 196,608`
  (`device_facts.txt`, re-confirmed in §0.3).

Practical consequence, and it is decision-relevant: at our 128-thread block and 5 blocks/CU the register
budget permits ≈ 196,608/(5·128) ≈ 307 lane-regs/thread, so **D30's "16 f32 accumulators = 128
VGPR/lane" is not an occupancy wall at our geometry** — the binding cap is LDS, exactly as today, and
`m16_bf16gq` at 30 VGPR was already LDS-bound. It **is** an ISA-ceiling question at their 256-thread
shape (§2.5). Those are different objections and the repo has been mixing them.

### 3.5 The honest answer to "why are they 5× faster"

1. **27× fewer warp-instructions per score** (89.9 → 3.29, measured from their rolled round body),
   which becomes **12.6×** once a `v_wmma` is billed at its 32-clk pipe time. Through our own measured
   elasticity that is **3.0–4.2×**, and their observed ratio is **≥ 5.04×** — so instruction count
   accounts for essentially the whole gap, and the residual runs the right way (their 52.5 ms is a traced
   upper bound per §22 of `bench/performance-status.md`).
2. The instruction count falls for four separable reasons, only the first of which is the matrix pipe:
   (a) both 256-dim dots retire on the tensor pipes, 0.125 WMMA per score instead of 16 VALU FMA slots;
   (b) **the softmax fold is one cross-lane op per row per round instead of a 5-step butterfly per
   score** (§1.3) — exactly the chain D61 named; (c) **10 independent accumulator chains per warp** give
   the round its own ILP instead of relying on occupancy to hide it; (d) one rescale per 64 KV rows
   instead of per live pair (`nbatch_fa`, `:13`, and the `v_pk_mul_f16` block at `:936-941`).
3. What the 5× is **not**: not occupancy (8 wf/CU vs our 20); not the matrix roof (they run at
   ~8.8 TFLOP/s ≈ 17 % of this box's measured 52.63 TFLOP/s WMMA rate and ~60 % of the theoretical
   scalar `v_fma_f32` roof — **derived arithmetic** on §22's ms in `bench/performance-status.md` and the
   head census at `specification/00-notation-and-hyperparameters.md:38-40`, not a measurement, and it must
   be re-derived before any of it enters `bench/`); not launch count (1 kernel vs our 3 is worth ~0.15 ms
   per §26 W4 of `bench/performance-status.md`); not async copy or TMA (they have neither on HIP).
4. Therefore a copy lands at **their** distance from the roofs, not at the roofs. Their round spends
   18 % of its instructions on `s_waitcnt`/`s_delay_alu` and 2.1 % on spill loads: they are
   mix-and-stall-limited too, just from a far better shape. Any claim of the form "matrix pipes ⇒
   52 TFLOP/s" is not what this object demonstrates.

## 4. Funding recommendation for 2.6

### 4.1 Which instrument first: the matrix-op-score tile, not the 512-row fixture

The two candidate instruments §28 of `bench/performance-status.md` offers are (i) a 512-row × 4096-lane
fixture and (ii) a tile whose score comes from a matrix op (D58's `m16_bf16gq`, 1.98 slots/touch,
census-only, never timed).

**Build (ii) first**, and D61's own data is the reason, not preference:

* The fixture cannot decide the question. D61 measured the tile's cost as **row-flat and
  block-count-flat** — "64 rows @ 4096 and 16 rows @ 4096 cost the same 3.11 vs 3.05 ms, and 128 blocks
  vs 512 blocks cost the same" — i.e. the tile's wall was already shown not to respond to more rows or
  more blocks in the range tested. A 512-row fixture mostly adds rows and blocks to a shape whose answer
  to that variable is measured; it does not change anything inside the round, which is the only term the
  verdict turns on.
* The missing term is the one D61 named: **a score produced by a matrix op, which is precisely the term
  that deletes the dependent fold**. Only (ii) changes it, so only (ii) can falsify the design.
* (i) is still required, as the **harness that times and admits the new form**, not as the instrument that
  decides whether to build it: D61 asked for both halves "in one measurement", and §20.6 of
  `prototype/attn_decode/results.md` ("a route no fixture reaches has no tests, including tests of the
  gate") means a new streaming path must be reachable in a gate before it ships — and at 512 rows the
  shipped form is at its best K/V re-use (§20.5 of `prototype/attn_decode/results.md`), which makes the
  wide fixture the right place to prove a win rather than the right place to look for one. Sequence: (ii)
  decides, (i) admits and times.

### 4.2 The smallest falsifiable version

Extend `prototype/attn_wmma/` with one form, `m16_fattn_score`, which changes the M = 16 probe's round
in exactly the three ways this anatomy says matter and nothing else:

1. score tile from `v_wmma_*_16x16x16` using the transcribed map (`mma.cuh:183-215`), not an assumed
   one;
2. **the fold becomes 1 cross-lane op per row per tile** (the `offset_first == offset_last` shape of
   `:861-872`), replacing the 5-step butterfly;
3. **≥ 8 independent accumulator chains** for P·V (the 8-tile `VKQ_C` shape of `:1021-1041`), with P
   converted to 16 bits as they do it (`:972-977`).

Accumulator width: **f32 first.** Their half2 accumulator is a 256-VGPR-ceiling escape for a 256-thread
block (§2.3, §2.5), and per §3.4 that ceiling does not bind at our 128-thread, 5-block shape — so step 1
can be built with **no numerics decision at all** beyond the 16-bit operands the standing rule already
permits.

Note what §3.1 already implies for D57's funding bar: their round is **7.16 pipe-weighted slots per
touch**, against the **≤ 19 slots/touch** D57 set as the bar to fund this route at all. The bar is not the
hard part; D58's control ("the same geometry with the shipped VALU fold is 59.86, worse than the shipped
45") is — the tile has to pay its own geometry tax, and (iii) below is what measures it.

Falsifiable claim, stated so it can fail: with A staged and the fold deleted, the form must (i) hold
≥ 4 blocks/CU per the occupancy probe, and (ii) print ≤ 20 pipe-weighted slots/touch (a
`m16_census.py`-style census, so D57's ≤ 19 bar is met), **and** (iii) on the wall at 4096 lanes cut the
`attn` family by **≥ 1.5×** — the number D57 already paid with a 2.4× instruction cut. (i)+(ii) are
CPU-only; (iii) is one timed row taken by the main agent in a zero-subagent window, reusing
`prototype/attn_wmma/run_attn_attrib_window.sh`. If it holds the census and fails (iii), the matrix
class and the round shape are both exonerated as levers, D61's wall is attributed to something else, and
2.6 closes on evidence instead of on budget. If it wins, the full tile gets funded with §3.1's band
(+30 % ±) as the prize and §4.4's gate work becomes mandatory.

Two things to do in the same window because they are cheap and they are the missing inputs: a **WMMA
latency leg** in `prototype/issue_rate_probe` (§3.2's gap), and — if the fixture is to be built at all —
the 512-row × 4096-lane fixture itself, with its must-catch set derived (§4.4).

### 4.3 What the numerics forfeit, and which step forces it

Per the standing numeric-format rule in `AGENTS.md` (any 16- or 32-bit arithmetic may be adopted if the
existing gates stay green; a red gate asks whether the check can be widened without materially weakening
it, judged by whether the widened band still catches every geometry-derived injection), and per
`tools/fp16_range_probe.py` (census: activations peak 82.5, logits 35.5, weights 0.17, nothing near
65,504; the two failure modes are overflow above 65,504 and flush-to-zero below 5.96e-08):

* **Operands (step 1): no forfeit.** bf16 values inside f16's range are exactly representable in f16
  (7 stored mantissa bits ⊆ 10) and the range census puts every K/Q value far inside it. The score
  products are exact in f32 on both sides, so what changes is the **summation order** — a
  cancellation-band question, not a format question (§10.1 F1–F4 of
  `specification/10-reference-dumps-and-gates.md`: a long dot's legal ndiff is 29–47 % and max-ULP bands
  are blind to whole-tensor 1-ULP error, so this leg is judged by cos/meanrel plus a floor probe, never
  by ULP).
* **P in 16 bits (step 1 too): a small, bounded forfeit.** Their own code already flushes P below
  `SOFTMAX_FTZ_THRESHOLD = -20.0f` (`fattn-common.cuh:11`, applied at `:891`), i.e. they discard weights
  < 2e-9 of the row max before any 16-bit question, and f16's own floor is 5.96e-8. Both are orders of
  magnitude below what the gates decide on.
* **O accumulated in f16 (step 2, theirs, not required for ours): the only forfeit worth arguing
  about**, because the accumulator carries error across the whole KV span rather than one score, and
  because it is what the 256-VGPR ceiling forced on them. Build with f32 accumulators first (§4.2) and
  revisit only if the register probe shows the f32 version losing blocks/CU. **If the only way to make
  an f16-O variant green is to loosen a check that currently catches a real injection, that is the
  standing rule's failure case and it is reported, not routed around.**
* `logit_softcap` is `false` in the traced instantiation, so the `tanhf` path at `:728` is not in the
  priced design and needs no band.

### 4.4 Gate work that comes with it

* **A section-(j)-style spelling A/B.** `engine/gate.py:8120` is the existing pattern — section (j),
  "the gdn_rec spellings and the token-loop launch — bit-identical, all run in this gate (§04.4, D32, D50,
  D54)" — whose comment states the property to copy: "the sides are launches of one build and the claim
  under test … is checked rather than assumed". Both spellings run inside one gate process and the
  bit-identity/near-identity claim is asserted in that process, with deferrals written to a sidecar so "is
  this a form finding or a wave finding?" is a file diff (`SCOPES`/`--scope form`,
  `wave_carry_deferred.json`, `WAVE_CARRY_REASON`). The new form needs the same shape, not a new mechanism
  — and `gate.py --attn-tile`'s default and `decode_run.cpp`'s must move
  in **one commit** or the gate silently judges a different binary than the engine ships (§20.3 of
  `prototype/attn_decode/results.md`, the trap row).
* **A floor probe** in the `prototype/mlp_gemv/floor_probe.py` mould, sized for the two new near-exact
  operations: the bf16→f16 operand conversion, whose floor is exactly 0 so anything nonzero is a bug, and
  the reordered score dot, whose legal ndiff is the cancellation band and which must therefore be judged
  by `pin_witness`-style witnesses plus cos/meanrel. A max-ULP gate on the dot is decoration.
* **A must-catch set derived from the tile geometry** (`prototype/attn_tile_model.py`'s `neg_expect`
  rule — hand-listed sets rot into "everything"), injected at the new structure's own seams: one of the 8
  O accumulator tiles (must move exactly its output columns and no other); one lane-half of the C→B
  parity exchange (must move exactly one KV parity of one row — **and this is the injection the current
  butterfly form cannot see**, which is the point); the single fold shuffle (must move the whole row);
  `FATTN_KQ_MAX_OFFSET` (must move every P by ~1e-4, must **not** move the argmax); one Q-in-register
  fragment (must move a whole column across every round). Derived must-**not**-catch: a row with exactly
  one live lane (softmax of a singleton is 1.0 whatever arrives, §10.1 of
  `specification/10-reference-dumps-and-gates.md`), and any partial-slot injection at lanes ≤ `MAXSC`
  (structurally unreachable). The case list is generated from the tile geometry and the
  driver asserts `cases == |population|` (D40's lesson: an admission typed from memory is not an
  admission), with the wide anchors and tile-boundary wave rows mandatory (§20.6 of
  `prototype/attn_decode/results.md`).
* **Coverage before numerics**: whatever the new form dumps, the verdict must be
  `total == n_rows and not outside`, aborting a coverage gap before any numerics section runs (the trap
  row §17/§18 of `prototype/attn_decode/results.md` was written for).

### 4.5 What I would stop and ask about rather than proceed on

The price band above for the round route (**+30 % to +39 % of prefill**) is **an order of magnitude larger
than the +5 % to +12 % D57 booked**, and the
reason is a scope change, not a re-estimate: D57 priced replacing the dot, and this anatomy says
replacing the round is what buys the 5×. Choosing to fund the round rather than the dot is a decision
about how much of the attention shape gets rewritten (it puts back the address-aliasing risk that §20.1 of
`prototype/attn_decode/results.md` records), so it is the user's call. Everything in §4.2 up to and including the census is inside the
existing authority; the wall claim (iii) is the first thing that spends a decision.

## 5. Tool and artifact findings (nothing in `tools/` or `engines/` was modified)

* **F1** — the HIP build's library is `engines/llama.cpp/build-hip/bin/libggml-hip.so.0.27.0`, not
  `libggml-cuda.so`. A census that looks for the CUDA name finds nothing and reports zero code objects,
  which reads as "no matrix instructions anywhere" rather than "wrong file".
* **F2** — `llvm-objdump` prints AMDGPU branch displacements as **unsigned** imm16, so a backward branch
  looks like a large forward one; the only reliable target is the `<symbol+0xNN>` annotation after the
  encoding. `tmp/gcc/fattn/loop_census.py` and `round_body.py` use the annotation; any future loop census
  here must too, or it will find zero loops and report that nothing is in a loop.
* **F3** — a dynamic-LDS kernel's LDS footprint is invisible both in the `rocprofv3` kernel trace
  (`LDS_Block_Size 0`) and in the code object (`group_segment_fixed_size 0`). Only the source formulas
  give it. Pricing that depends on a dynamic-LDS tile's occupancy must say which of the three it used;
  §2.4 and §0.3 do.
* **F4** — mnemonic classes must be matched by prefix: the disassembler emits `v_exp_f32_e32`,
  `v_max_f32_e32`, `v_cvt_f16_f32_e64`. My first class regex (`v_exp_f32`) matched nothing and printed
  `exp = 0` in every round body while `v_div_scale_f32` matched elsewhere — a plausible-looking census
  over an empty match set, the same failure class as `prototype/gdn_collapse/run.py`'s zero-record parser.
* **F5** — `tools/kernel_isa_census.py`'s DOT bucket matched 864 instructions in this kernel, all
  `v_fma_mix_f32`. Our own kernels' DOT rows are real `v_dot2_*`. Two different instruction classes share
  one bucket name, so a cross-engine "DOT: 864 vs 130" row would be meaningless. Reported, not fixed.
* **F6** — `tools/kernel_occupancy_probe.cpp` computes "waves/SIMD allowed by VGPR" as a hard-coded
  `512 / 48` and prints it as a conclusion; §3.4 shows the driver's own returned block counts and its
  `MaxRegistersPerMultiprocessor` contradict the 512. The tool is right about LDS and wrong about
  registers, and its canary only tests LDS.
* **F7** — `hipFuncGetAttributes().numRegs` plus a VGPR-pressure stub is not a usable register probe on
  this driver: every burn level returned `numRegs = 5` (§0.3's failed canary). An empty-asm VGPR fence
  does not create a live range. Any future register probe must use a real instruction per value and must
  keep the canary that proves the registers are visible — this is the A/B-canary rule from AGENTS.md
  applied to a compile-time probe.
* **F8** — `tools/gpu_lock.sh acquire` binds to the calling shell's pid, and each `bash` tool call is its
  own shell, so a lock acquired in one call reads `STALE (holder pid dead)` by the next. The acquire /
  use / release pattern has to live inside a single command line. This session's `fattn-anatomy-regfile`
  record was released at the end and `tools/gpu_lock.sh status` now reads `free`.

## 6. Reproduction

```bash
# §0.1 carve the HIP library's code objects (host-side, no device) and find the defining object
python3 tools/carve_codeobjects.py engines/llama.cpp/build-hip/bin/libggml-hip.so.0.27.0 tmp/gcc/fattn/co
python3 tmp/gcc/fattn/find_symbol.py tmp/gcc/fattn/co _ZL18flash_attn_ext_f16ILi256ELi256ELi16ELi4
# §0.2 the traced launch shape, from the §22 trace already on disk (no device)
python3 tmp/gcc/fattn/trace_row.py tmp/d52/llama_pp4096/pp4096_kernel_trace.csv 'void flash_attn_ext_f16<256, 256, 16, 4'
# §1.1/§1.2 identity canaries + disassembly + ISA census + per-round mnemonic census
bash tmp/gcc/fattn/disasm.sh
python3 tools/kernel_isa_census.py tmp/gcc/fattn/fattn_mma.s
python3 tmp/gcc/fattn/round_body.py tmp/gcc/fattn/fattn_mma.s 'flash_attn_ext_f16ILi256ELi256ELi16ELi4'
# §0.3 the one host-side driver query (device context: no kernel launched, no timing taken)
hipcc -O2 -x hip tmp/gcc/fattn/regfile_probe.cpp -o tmp/gcc/fattn/regfile_probe && ./tmp/gcc/fattn/regfile_probe
```

`disasm.sh` prints the two canaries (§0.1) before it disassembles, so the section-1 census cannot be run
against an object that is not the traced one.

Claims I could **not** verify, listed so nobody inherits them as facts: the 36,096 B LDS figure is
source-derived, not driver- or trace-confirmed (F3); the register-file arithmetic in §3.4 rests on driver
outputs already on disk, not on the stub probe (§0.3); `v_wmma` dependent-chain latency is unmeasured
(§3.2); and the ~8.8 TFLOP/s their traced time implies is arithmetic on §22's ms, not a measurement.
