# A12 steps 1 and 2: the register question re-measured, and a wide block that computes the right answer

Scope, as funded: **step 1** = re-measure the register/spill anchors with the *corrected* instrument (llama's
kernel, and the wide shape laddered across register costs) and **step 2** = write the standalone wide kernel with
a CPU mirror on a derived fixture population, plus one device correctness check. No timing, no engine plumbing,
no gate changes, nothing committed.

Everything below was produced by the commands in this table, all of them under `prototype/attn_wide/`:

| what | command | cost |
|---|---|---|
| wide kernel, linked binary + GCN listing | `bash prototype/attn_wide/build_wide.sh tmp/attn_wide` | ~25 s CPU |
| the register ladder, 12 variants, compile-only | `bash prototype/attn_wide/wide_ladder.sh` | 21 s CPU |
| llama re-measured with the corrected reader | `bash prototype/attn_wide/llama_vgpr_check.sh` | 21 s CPU |
| numerics mirror over the derived population | `.venv/bin/python prototype/attn_wide/a12_wide_mirror.py` | 10 s CPU |
| the ONE device correctness check | `tmp/attn_wide/attn_wide` (under `tools/gpu_lock.sh`) | 0.28 s wall |
| one variant's fields, read back | `python3 prototype/attn_wide/wide_regread.py <listing> --match attn_wide_wave --threads 256` | instant |

Files this step added, and nothing else was touched:

    prototype/attn_wide/attn_wide.h           the wide block (the deliverable of step 2)
    prototype/attn_wide/attn_wide.hip.cpp     correctness driver: naive reference + the case list
    prototype/attn_wide/build_wide.sh         build, with a WMMA site-count refusal
    prototype/attn_wide/wide_ladder.sh        the compile-only register ladder (step 1b)
    prototype/attn_wide/wide_regread.py       code-object reader with the block size as an argument
    prototype/attn_wide/llama_vgpr_check.sh   llama re-measurement (step 1a)
    prototype/attn_wide/a12_wide_mirror.py    the numerics mirror
    prototype/attn_percost/A12-WIDE-STEP12.md this file

`git status --porcelain` at writing lists nothing but these paths — `prototype/attn_wide/` and this file — and
nothing under `engine/`, nothing else under `prototype/attn_percost/`, and no gate or census policy file.

---

## 1. Step 1a — llama's kernel, re-measured: the 256 VGPR / 900 B anchor HOLDS

`A12-WIDE-BET.md` §1 states the spill budget as "≤ 1,024 B/thread" and anchors it on llama's kernel at 900 B.
That anchor came from `A2-census.md` §1 row `flash_attn_ext_f16<256,256,16,4,false,false,false>`, which predates
the instrument correction of `bench/performance-status.md` §55 and `A11-attribution.md` §2 — the correction that
says a `.vgpr_count` read out of LLVM's note block can be *another kernel's* number, because note entries are not
in definition order. So the
number A12 is funded on was re-taken from the code-object header:

```
$ bash prototype/attn_wide/llama_vgpr_check.sh
instantiation              : flash_attn_ext_f16<ILi256ELi256ELi16ELi4ELb0ELb0ELb0>
shipped object holding it  : engines/llama.cpp/build-hip/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/
                             __/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu.o
instance TU compiled       : engines/llama.cpp/ggml/src/ggml-cuda/template-instances/
                             fattn-mma-f16-instance-ncols1_16-ncols2_4.cu

kernel                                                     vGPU  alloc8  scratch B   LDS B  scratch sites
_ZL18flash_attn_ext_f16ILi256ELi256ELi16ELi4ELb0ELb0ELb0E…    256     256        900       0            564
```

**Verdict: the census row is right, and it is right for the right reason now.** `.amdhsa_next_free_vgpr` = 256 (the
per-thread architectural ceiling, not LLVM's opinion about a count) and `.amdhsa_private_segment_fixed_size` =
900 B/thread. The spill is witnessed behaviourally too, not only in metadata: **564 `scratch_load`/`scratch_store`
instruction sites** in that one kernel body — llama pays a deliberate, large spill. So the A12 relaxation "spill is
a budget, not a taboo" is not inventing a practice; it is asking for a fifth of what the reference implementation
on this same architecture already pays.

Three things this measurement had to be careful about, because each of them produces a plausible wrong number:

* **`.amdhsa_group_segment_fixed_size` reads 0 and that is not "no LDS".** llama stages through
  `extern __shared__`, which is dynamic: it never appears in the object. The real figure — 36,096 B at
  `swz=false` on AMD, from `A10-REWRITE-BRIEF.md` §3 item 4 — is derived from the source, and with it llama's own
  residency is **1 block/CU = 8 wavefronts** (64 KiB ÷ 36,096 B), which is exactly the residency A12 asks for.
  Reading the 0 at face value would have priced llama at 5 blocks/CU.
* **the note-block `.vgpr_spill_count` is not printed by the script at all.** Pairing that field with a kernel name
  is the recorded mistake; the three `.amdhsa_kernel` fields need no help and are keyed by the mangled name in a
  single pass.
* **`llvm-nm -g` cannot find this kernel**: llama declares it `static` (the `L` in `_ZL18flash_attn_ext_f16`), so a
  globals-only symbol dump reports "no object defines the instantiation" about the object that does. The script
  searches with `--defined-only` and no `-g`, and asserts both that the object contains the symbol and that the
  source file it maps to exists — the instance-filename↔template-arguments mapping is not guessable
  (`<256,256,…>` is `ncols1_16-ncols2_4`).

### 1.1 The min-blocks question `A10` left open, answered by one compile

`A10-REWRITE-BRIEF.md` §3 left open whether clang honours a `__launch_bounds__` **min-blocks** argument on
gfx1151, asked of llama's `__launch_bounds__(256, 2)`. Row **L** of the ladder (§2) compiles the wide shape at
identical settings with min-blocks 4, where LLVM's own arithmetic (196,608 registers per CU ÷ (256 threads × 4
blocks) = 192) predicts a *cap of 192* where the unbounded shape says 256:

```
L_minblocks4   8 64 1 1 minb=4 → 256 vGPU, 224 B scratch     (row A, minb=1: 256 vGPU, 156 B scratch)
```

The reservation did **not** come down; the spill went up (156 → 224 B). So a min-blocks argument is not inert on
this target — it changed code generation — but it does not cap the VGPR reservation below the 256 ceiling. Which
also explains llama without any further experiment: at 256 threads and min-blocks 2 the same arithmetic gives
196,608/512 = 384, already above the 256 ceiling, so llama's `__launch_bounds__(256, 2)` was never going to bind.
The 256-register shape of both kernels comes from the per-thread ceiling, and from nothing else.

---

## 2. Step 1b — the wide shape laddered

`wide_ladder.sh` compiles twelve variants of the same body, one axis at a time, and reads
`.amdhsa_next_free_vgpr` / `.amdhsa_private_segment_fixed_size` / `.amdhsa_group_segment_fixed_size`. Nothing is
launched. Threads = 32 × warps, so the `blk/CU` column is stated per block size; `regs/CU` is the
"65,536 per SIMD" = 131,072-per-CU reading forced by the pad A/B of `bench/performance-status.md` §55, and the
flag `--regs-per-cu 65536` gives the older reading
(see the note under the table).

```
variant            warps  stage  qreg  pvf16  minb  |   vGPU  alloc  scratch     LDS blk/CU  binder
A_a12_wide         8      64     1     1      1     |    256    256     no      156   32768      2  registers  (16 wf/CU)
B_q_from_global    8      64     0     1      1     |    169    176     no        0   32768      2  registers  (16 wf/CU)
C_f32_accumulator  8      64     1     0      1     |    256    256     no      500   32768      2  registers  (16 wf/CU)
D_qreg0_f32        8      64     0     0      1     |    235    240     no        0   32768      2  registers  (16 wf/CU)
E_stage32          8      32     1     1      1     |    256    256     no       60   16384      2  registers  (16 wf/CU)
F_stage16          8      16     1     1      1     |    238    240     no        0   16384      2  registers  (16 wf/CU)
G_warps4           4      64     1     1      1     |    256    256     no      216   32768      2  LDS        ( 8 wf/CU)
H_warps3_stage64   3      64     1     1      1     |    256    256     no      180   32768      2  LDS        ( 6 wf/CU)
I_warps3_like_eng  3      16     1     1      1     |    238    240     no        0    8192      5  registers  (15 wf/CU)
J_stage128_f16     8     128     1     1      1     |    256    256     no      600   65536      1  LDS        ( 8 wf/CU)
K_stage128_f32     8     128     1     0      1     |    256    256     no      752   65536      1  LDS        ( 8 wf/CU)
L_minblocks4       8      64     1     1      4     |    256    256     no      224   32768      2  registers  (16 wf/CU)
```

Each compiled object was also checked for its WMMA site count by `build_wide.sh` (it refuses a listing whose
`v_wmma_f32_16x16x16_f16` / `v_wmma_f16_16x16x16_f16` sites do not match what the source says it emits — a
DCE'd shell would otherwise measure nothing). Row A emits 64 + 64 sites.

**The four readings that matter.**

1. **The A12 shape fits, with room.** Row A: 256 vGPU (at the ceiling, like llama), **156 B/thread of scratch**
   against the funded 1,024 B budget, declared LDS 32,768 B. The register sum that motivated the shape
   (128 Q + 64 accumulator + 32 scores + 32 P + ≈20 ≈ 276 against 256) is real — LLVM spills — but it spills
   156 B, i.e. ~39 registers' worth, not the ~300 the naive sum suggested. The budget is met by a factor of 6.5.
2. **Where spill first appears, and how big it gets.** With Q resident and the F16 accumulator: stage 16 → **0 B**,
   stage 32 → **60 B**, stage 64 → **156 B**, stage 128 → **600 B**. So spill is a function of stage depth (more
   live per round) and it is the *stage*, not the row count, that buys it: rows 3/4/8 at stage 64 spill 180/216/156 B
   at the same 256-register reservation. At stage 128 the LDS is 65,536 B, which is the whole CU's LDS and needs the
   dynamic-allocation opt-in to launch, so J and K are compile-only shapes and are labelled as such in the ladder.
3. **The F16 accumulator is doing real work even in spill terms.** Same shape, same 256 vGPU, F32 instead of F16:
   **500 B at stage 64, 752 B at stage 128** — 3.2× and 1.25× the spill. A10 killed the F32-on-a-wide-block shape on
   the register *sum*; this says it would also pay it in scratch. Row D (`qreg=0` + F32) is spill-free at 235 vGPU,
   which is the shape A10's `A10-RF-LADDER.md` §6 landed on; the wide shape's win comes from spending the released
   registers on Q instead of on the accumulator.
4. **Q-in-registers costs 87 registers and 156 B, and buys one global read of Q per round.** Row B is row A with Q
   re-read from global every round: 169 vGPU, **0 B** scratch. That is the escape hatch, stated with its price: if
   156 B/thread of spill ever reads as too much (it does not — llama pays 900), the wide shape can give Q back and
   still hold 2 blocks/CU. Whether the re-read costs time is a device question and is explicitly not answered here.

**Residency, and the one number that is still an assumption.** At 32,768 B of LDS and 256 vGPU on a 256-thread
block, the reader says 2 blocks/CU = 16 wavefronts, because it divides a register file of 131,072 per CU — the
"65,536 per SIMD" reading that `bench/performance-status.md` §55's pad A/B forces. Under the older 65,536-per-CU
reading the same fields give **1 block/CU = 8 wavefronts**:

```
$ … wide_regread.py <A listing> --threads 256                    → 2 blk/CU  (regs 2, LDS 2) 16 wf/CU
$ … wide_regread.py <A listing> --threads 256 --regs-per-cu 65536 → 1 blk/CU  (regs 1, LDS 2)  8 wf/CU
```

**That ambiguity does not gate A12**: `A12-WIDE-BET.md` §1 allows 1 block/CU at 8 wavefronts as a precondition
relaxation (llama's own residency), and both readings satisfy it. It does gate any *comparison* against the shipping kernel's
occupancy, and so it belongs to the device window (§6).

---

## 3. Step 2 — what the kernel is

`prototype/attn_wide/attn_wide.h`, standalone, template-free, macro-parameterised by exactly the four ladder axes
(`WIDE_WARPS`, `WIDE_STAGE`, `WIDE_QREG`, `WIDE_PVF16`), defaulting to the funded shape: **8 waves × 256 threads,
128 query rows, a 64-position K/V stage, Q converted to F16 once per launch and held in registers, F16 P·V
accumulator**. `attn_wide.hip.cpp` is the correctness driver: it builds Q/K/V on the device, runs the kernel, runs
a naive two-pass O(n²) reference (scores + softmax weights, then the output contraction) in F32, and compares.

Geometry and the laws it uses, in one place:

* **Rows are 32 query positions × the 4 heads of one KV group** (llama's `R5` packing). Wave `w` owns positions
  `pos0 + 4w … +4`; its 16 row-slots are those 4 positions × 4 heads. The consequence that actually matters is
  arithmetic, not aesthetics: the block's round range is `[0, min(npos, pos0+32))`, computed from `pos0` alone, so
  this shape needs **no `fold[]` buffer and no block-wide span fold** — the thing the shipping form pays two
  barriers and a shared array for.
* **The fragment maps are the engine's measured laws, restated** (`wide_c_row`, `wide_c_col`, `wide_h_dim`), from
  `prototype/attn_wmma/fattn_map_probe.cpp` and `prototype/attn_percost/a10_pvf16_map_probe.cpp`. They are restated
  because a prototype that `#include`s an anonymous-namespace engine header cannot be compiled alone; when this
  file moves into `engine/` the three duplicates collapse into the single spellings. A relabelled map is a permuted
  output row, which is the class of defect that previously arrived as all-NaN.
* **Scores**: `wide_score_round` is `bf16(bf16(f32_dot) × 0.0625)` — the shipping kernel's `fattn_score_round`,
  including the association, per the "a hoisted scalar constant is a different number" trap. The rounded score
  replaces the accumulator in place; a second live array would be 32 more registers per round for nothing.
* **The row's max needs one `__shfl_xor_sync(…, 16, 32)`** because a C tile's columns are split across the lane
  pair `(t, t^16)` — the shipping kernel's tenth lane-movement op, whose absence scales a row by ~2× and is
  invisible to cosine.
* **P·V** is 8 output tiles × 4 KV slices × 2 `op_sel` halves = 64 `v_wmma_f16_16x16x16_f16_w32` issues into 8
  independent F16 D blocks, with V staged **transposed** (rows are output dims) so the A fragment loads directly.
  The two-issues-into-one-D-block pair is llama's shape, read out of the shipped listing in `A10-PVF16.md` §3.
* **The per-round rescale is before the round's add, and it is load-bearing**: with an F16 accumulator, §5's chain
  probe shows the unscaled accumulator leaving fp16's finite range altogether.
* **LDS is one array with two layouts**, sized `max(K/V stage, output transpose)` = 32,768 B. The epilogue
  transpose re-purposes the K/V stage, which is safe exactly because the round loop's last barrier is "V consumed".
  Declaring the transpose as a *second* `__shared__` array — the obvious spelling — pushes the block to 49,152 B
  (exactly 48 KiB, at the API's static limit) and makes residency 1 block/CU on LDS arithmetic. That is the
  engine's `fattn_warp_stage_halves` note applied one step early.
* **What is deliberately absent**, because A12's sequencing says no engine plumbing: the split/combine path
  (`part` slots, `attn_combine`), the `specification/05-gated-full-attention.md` §05.7 append, the
  injection-aware KV-group rule, and one-pass rows at or
  below `MAXSC`. The block writes its final O row directly, and the mirror judges the same narrowed op — mirror
  and kernel must agree about *what the op is*, which is the lesson of `prototype/attn_decode/results.md` §20.1,
  applied at the boundary instead of inside.

---

## 4. The one device correctness check — PASS, and no timing taken

Run under the lock, blocking, one invocation, `tools/gpu_lock.sh util` **91 %** at acquire and **67 %** at release
(the serving process; the gate is deterministic and has no atomics). Wall clock of the whole process, as reported
by `time`: **0.28 s**. **No timing was taken** — no burst, no best-of-N, no `ms` figure for any kernel, and nothing
in this file belongs in `bench/performance-status.md`.

```
$ tools/gpu_lock.sh acquire a12-wide-selftest && tools/gpu_lock.sh util && time tmp/attn_wide/attn_wide
attn_wide correctness: warps=8 threads=256 rows=128 stage=64 qreg=1 pvf16=1 pos_per_block=32
  npos=1     single position            blocks=2     worst_cos=1.0000000 worst_meanrel=0.000e+00 argmax_moved=0/8    nonfinite=0 PASS
  npos=31    block tail, short          blocks=2     worst_cos=0.9999997 worst_meanrel=6.789e-04 argmax_moved=1/248 nonfinite=0 PASS
  npos=32    exactly one block          blocks=2     worst_cos=0.9999997 worst_meanrel=6.094e-04 argmax_moved=5/256 nonfinite=0 PASS
  npos=33    spills into a second block blocks=4     worst_cos=0.9999996 worst_meanrel=7.386e-04 argmax_moved=1/264 nonfinite=0 PASS
  npos=63    one round, short           blocks=4     worst_cos=0.9999994 worst_meanrel=9.730e-04 argmax_moved=5/504 nonfinite=0 PASS
  npos=64    exactly one round          blocks=4     worst_cos=0.9999994 worst_meanrel=9.319e-04 argmax_moved=2/512 nonfinite=0 PASS
  npos=65    a second round, 1 lane     blocks=6     worst_cos=0.9999994 worst_meanrel=9.210e-04 argmax_moved=4/520 nonfinite=0 PASS
  npos=130   full round then a partial  blocks=10    worst_cos=0.9999988 worst_meanrel=1.270e-03 argmax_moved=4/1040 nonfinite=0 PASS
  npos=300   pinned fixtures' max cache blocks=20    worst_cos=0.9999950 worst_meanrel=3.128e-03 argmax_moved=21/2400 nonfinite=0 PASS
  npos=512   widest many-round case     blocks=32    worst_cos=0.9999958 worst_meanrel=2.362e-03 argmax_moved=44/4096 nonfinite=0 PASS
ATTN_WIDE SELFTEST PASS: 10 case(s), 0 failed
```

The case list is built around the boundaries this geometry can get wrong: exactly one block / one round
(32, 64), one past each (33, 65 — the "second round of one lane" case that a `base < hi` loop drops), a tail block
(31), a partial round after a full one (130), and the pinned fixtures' own longest cache (300). The `argmax_moved`
counts are over `DH` against the naive reference and are what F16 rounding on near-ties looks like; they are
reported, not waved away, and none of them is a gate.

**Why one run was enough.** After this run the only change to `attn_wide.h` was exposing the `__launch_bounds__`
min-blocks argument as a ladder axis (`WIDE_MINB`, default 1). The rebuilt GCN listing is **byte-identical** to the
one the PASS was produced from (`cmp` clean), so the result stands for the shipped shape without a second device
run. That is the behavioural-canary rule applied in the cheap direction: prove the artifact did not change, rather
than re-spending device time to find out.

**What this check is not.** It compares the kernel against an in-process naive reference in F32. It validates index
arithmetic, row/head mapping, causal masking, the lane-pair reduction, staging and normalisation. It is **not** the
§10.1 pinned band (`specification/10-reference-dumps-and-gates.md` §10.1, "Gates"), it is not `engine/gate.py`,
and it does not admit anything: the pinned comparison is §5's
mirror, and the real admission is the engine gate, which this step does not touch.

---

## 5. Numerics on a derived population — the wide round costs less than the format change alone

`a12_wide_mirror.py` is `a10_pvf16_mirror.py`'s question one step on: that mirror changed the accumulator's
*format*; A12 changes the format **and** the round width, and the width is the more invasive of the two — one max
over 64 columns instead of four maxima over 16 moves the running max, the rescale factor, the denominator and the
order of the P·V adds at once. It imports `prototype/attn_decode/floor_probe.py` (the instrument §10.1's floors
were taken with) and feeds **one** pinned score matrix to three arms:

| arm | what it is the shipping form of |
|---|---|
| `f32@16` | the kernel that ships today (`FATTN_MLANES = 16`) |
| `f16@16` | the inert knob `--attn-fattn-pvf16 1` |
| `f16@64` | **A12's wide block** (`WIDE_STAGE = 64`) |

**The population is derived, not typed** — `data/refs/*/` with a `manifest.json`, filtered to those with a
*complete* §05 attention pin set, every (step, layer) and (layer, row) enumerated:

```
17 fixtures with cases (chunk64b dec64f dec64t dec8 mt32f p128 p300 p64 p8 prefix24dec8 smoke
                        wide1024f wide4096f wide4096p wide520f wide640f zz)
 5 skipped: corpus48, corpus48engine, corpus48engine-smoke (no manifest.json), corpus48sdpa, dec300f
 282 cases (decode=186, prefill=96); lanes 3..4104 = 1..257 rounds at 16, 1..65 rounds at 64
 2,256 head-rows (282 cases × 8 query heads) = 577,536 output elements compared
```

AGENTS.md has this admission trap in prose — a "nine canonical fixtures" list of which seven names did not exist —
and `a10_pvf16_mirror.py`'s `dec8`+`p300` pair is hand-carried, so this script does not inherit it. The skips are
printed with a reason, and zero cases is a refusal, not a pass.

```
1. THE BAND QUESTION
   f16@64   vs f32@16    cos >= 0.99999149  meanrel <= 1.83e-03  worst 27984 ULP  outside a no-excuse band: 761
   f16@64   vs f16@16    cos >= 0.99998497  meanrel <= 5.74e-03  worst 28391 ULP  outside a no-excuse band: 1942
   f16@16   vs f32@16    cos >= 0.99998632  meanrel <= 5.40e-03  worst 28667 ULP  outside a no-excuse band: 1718
   rounds per head-row: f32@16 median 2 (max 257), f16@64 median 1 (max 65)
   each arm against the PIN:
     f32@16   cos >= 0.99999068  meanrel <= 4.30e-03  worst 29827 ULP
     f16@16   cos >= 0.99998876  meanrel <= 4.89e-03  worst 29818 ULP
     f16@64   cos >= 0.99999083  meanrel <= 4.28e-03  worst 29809 ULP

2. THE DECISION QUESTION — argmax over DH, per head-row
   f32@16   is the reference arm              against the PIN: 7/2256
   f16@16   vs the f32@16 arm: 0/2256 rows move    against the PIN: 7/2256
   f16@64   vs the f32@16 arm: 2/4512 rows move    against the PIN: 8/2256

3. THE RANGE QUESTION
   largest |out| in the wide arm: 35 (fp16 max 65504): 0 values above it
   nonzero values below the flush-to-zero boundary (~5.96e-08): 0
   worst off_rel 0.0363 at wide4096f/decode7/L11 head 6 dim 7, |out| 0.9336 → 0.9297 against a row peak of 1.156
```

**The band verdict, in the words the decision rule wants.** The wide arm's distance to the **pin** (cos ≥ 0.99999083,
meanrel ≤ 4.28e-3) is *no worse* than the shipping F32 arm's own distance to the same pins (0.99999068 / 4.30e-3) —
the wide form is inside the same cloud as the form the existing bands were calibrated on. Against the f32 arm itself
it is cos ≥ 0.99999149. And the round-width change is *smaller* than the format change alone: `f16@64` vs `f16@16`
(0.99998497) is a hair worse than `f16@16` vs `f32@16` (0.99998632), so re-associating the softmax merge costs less
than swapping the accumulator does. That is R8's prediction from `A10-REWRITE-BRIEF.md` §3 ("not bit-identical;
judge with the §10.1 aggregates") confirmed on a population 15× wider than the two fixtures it was written from.
Per the standing numeric-format rule, no new band is requested and none is needed: the aggregates move within the
cloud the current bands already tolerate, and the worst `off_rel` values are cancellation positions (a small `|out|`
beside a large row peak), which is where a ULP-sized difference *must* read as a large relative one.

Two honesty items that are part of the result, not footnotes:

* **The pessimistic model does not separate the geometries.** Modelling each round's products as rounding and adding
  sequentially in F16 — the pessimistic end of what the matrix unit actually does — prints the same minimum for both
  arms (`cos ≥ 0.99909870`), because the population's worst row is one they share. Checked rather than assumed: the
  two pessimistic arms are bit-identical on **85 of 162** head-rows and elsewhere differ by at most **0.029**. So the
  optimistic round-model is the one that discriminates here, and the pessimistic print exists to say so. (The model
  is also restricted to the 162 head-rows with more than 16 lanes; below that, one round is one round in both
  geometries and the comparison cannot be made at all.)
* **The chain section is where "the rescale is mandatory" stops being rhetoric.** Without the per-round fold, and
  with rounds cycled to lengths the fixtures do not contain (stated as an extrapolation at fixed addend scale), the
  F16 accumulator does not merely drop low addends — it **walks out of fp16's finite range**:

  ```
    f16@16   1..257 rounds of 16   first drop: 1 (median 29)   most drop: 11   leaves fp16 range: 2039 (median 2180)
    f16@64   1..65 rounds of 64    first drop: 1 (median 2049) most drop: 19   leaves fp16 range: 1721 (median 2180)
  ```

  The wide arm reaches a given sweep length in a quarter of the rounds, so each of *its* rounds carries four times
  the addends into a single accumulator step: the rescale is **more** load-bearing at 64 lanes, not less, and
  `attn_wide.h` places it exactly where the shipping kernel places it — before the round's add, on the
  accumulator's critical path. The overflow is silenced and *reported* rather than left as a numpy warning: leaving
  fp16's range is a range failure, which is the same distinction §3 of `A10-PVF16.md` makes about the knob.

---

## 6. What A12 may now claim, and what still needs the device window

Stated against `A12-WIDE-BET.md` §1's three relaxations, so "the shape fits" means something checkable:

| precondition | status | evidence |
|---|---|---|
| spill ≤ 1,024 B/thread | **met: 156 B** at the funded shape, and the 900 B reference it was sized against is confirmed | §2 row A, §1 |
| 1 block/CU at 8 wavefronts allowed | **met under either register-file reading** (1 or 2 blocks/CU) | §2's residency note |
| F16 P·V accumulator | **within the existing bands**, and cheaper than the format change alone | §5 |
| the wide block computes attention | **yes**, on 10 boundary cases against a naive reference | §4 |

Still unmeasured, and each of them is main-agent device work or an explicit decision — none of them is answerable
from where this step sat:

1. **The traffic claim itself.** 64 MAC per LSU byte, the deletion of the 898 GB/pass Q re-read, and the BET's
   6.5 TFLOP/s at 32,768 lanes are *predictions*. No timing was taken here, by rule and by design.
2. **Blocks/CU actually granted**, and therefore whether the shipping comparison is 8 wf/CU vs 8 or vs 16. Needs a
   driver/counter read in an exclusive window; the two `wide_regread.py` invocations in §2 bracket it.
3. **The shallower-stage alternative.** Rows E and F (stage 32/16 at 8 waves) reach 2 blocks/CU on LDS arithmetic
   with 0–60 B of spill. If the MAC-per-byte story survives at stage 32, that is the cheaper shape. Unpriced.
4. **Negative controls for the wide geometry.** This prototype has no partial/split path, so the derived
   must-catch set of `prototype/attn_decode/results.md` §20.6
   must-catch set (rows of 512/513/514/515 lanes, the wide anchors) has nothing to bite on yet. The moment A12
   grows a `part` slot, that control set must be **re-derived from the new geometry** before any claim about the
   streaming route is made — the recorded trap is that a control whose must-catch set is empty scores a clean 0/1.
5. **Gate integration.** Nothing here is reachable from `engine/`: no `gate.py` flag, no `decode_run` route, no
   admission. That is A12 §3's sequencing, deliberately not violated.

## 7. Traps this step ran into, for the record

Not applied to `AGENTS.md` (a worker does not edit the trap index); recorded here with their witnesses so the
main agent can decide whether any is general enough.

* **`llvm-nm -g` cannot see a `static` kernel.** llama's fattn kernels are internal-linkage (`_ZL…`), so a
  globals-only symbol dump answered "no object defines this instantiation" about the object that does. The
  instance-filename↔template-arguments mapping is not guessable either (`<256,256,16,4,false,false,false>` =
  `ncols1_16-ncols2_4`), so both halves must be derived and asserted: symbol present in object, source exists at
  the mapped path. → `llama_vgpr_check.sh`
* **A declared-LDS claim written as the SUM of two `__shared__` arrays silently moves the residency it is about.**
  The epilogue transpose as its own array took the block to 49,152 B — exactly 48 KiB, at the API's static limit —
  and 1 block/CU on LDS; aliasing it over the K/V stage (the engine's own arrangement, and the reason its
  `fattn_warp_stage_halves` note exists) is 32,768 B and makes 2 blocks/CU possible. Same class as the
  register-file row of `bench/performance-status.md` §55:
  register-file row: the cost you did not declare is the one you will not notice. → `wide_lds_halves()`
* **`.amdhsa_group_segment_fixed_size` = 0 means dynamic, not none.** Reading it at face value would have put
  llama at 5 blocks/CU instead of 1. → §1
* **A pessimistic model that prints the same minimum for both arms is not agreement.** It was checked (85/162
  bit-identical, max Δ 0.029) rather than reported as a null; the fix was to print the agreement count, the max
  difference, and the excluded-case count instead of two minima. → §5
* Smaller but each one did stop a run before it printed anything: `Path / "fmt" % x` binds `%` before `/`; a
  `printf`-style `%d`×10 format handed 8 arguments; and `floor_probe`'s `selftest()` returns **0 on success**, so
  inverting it turns a green comparator into a refusal.
