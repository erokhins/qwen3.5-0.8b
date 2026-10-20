# A13 worker notes — porting llama's attention structure onto our wide form

Written incrementally by the A13 worker (2026-10-20/21). CPU-only so far: no device run, **no timing taken**.

## 0. Baseline, taken in this window before any edit

Carve + disassemble of the binary that was already in `engine/build/decode_run` (HEAD `8a729d9`, built
2026-10-20 21:24, i.e. before this work):

```
python3 tools/carve_codeobjects.py engine/build/decode_run tmp/a13/base --target gfx1151
  -> CANARY PASS: 9 code objects, 108 kernel symbols; attn_wide_wave is in co_352256.co (4 matching syms)
/opt/rocm/llvm/bin/llvm-objdump -d tmp/a13/base/co_352256.co > tmp/a13/base/k.s
python3 prototype/gdn_walk/g3_isa_bill.py tmp/a13/base/k.s tmp/a13/base/co_352256.co attn_wide_wave
  -> metadata: vgpr_count 256, vgpr_spill_count 163, private_segment_fixed_size 176 B,
     group_segment_fixed_size 32768, wavefront_size 32
  -> widest backward branch 0x2a9c0..0x3ef70, 14,324 insns, 128 v_wmma
python3 prototype/attn_percost/a2_round_bill.py tmp/a13/base/k.s 2a9c0 3ef70
  -> 110.9 non-matrix slots per matrix slot (reproduces `prototype/attn_percost/evidence/r2_round_bills.txt`)
```

Metadata for the same kernel, for the spill watch: **256 VGPR reserved, 163 spilled, 176 B scratch/thread.**
Any added register live-range (the item-2 prefetch buffer) shows up here, so it is re-read after the build.

## 1. Measurement finding: the span `g3_isa_bill.py` calls the round is NOT the round, for our kernels

`g3_isa_bill.py` reports the **widest backward branch**. For `attn_wide_wave` that branch is
`0x3ef6c: s_cbranch_vccnz -> attn_wide_wave+0xc0`, and `+0xc0` (0x2a9c0) is an **`s_endpgm`** — it is the
early-exit jump (`if (row0 >= nrows) return; if (hi <= 0) return;`), not a loop latch. The "span" therefore
covers the whole kernel body, epilogue included. Enumerating every backward branch (scratch
`tmp/a13/loops.py`, which resolves branch targets from objdump's `<sym+0xNN>` annotation):

| span | insns | what it holds | bill |
|---|---|---|---|
| `2a9c0..3ef6c` (widest; §58/R2 used this) | 14,325 | round **and** the epilogue pass loop (960 `v_div_*`, 188 `global_store_d16_hi_b16`, 9 barriers) | **110.9** |
| `30878..3ef8c` | 10,283 | the epilogue pass loop alone: 0 wmma, 9 barriers, 960 divide-class, 188 stores | — |
| `2bdcc..30844` | 3,206 | **the round loop**: 128 wmma, 4 barriers (= the round's four `__syncthreads`), 265 LDS, 0 divides | **24.05** |

The same is true on the other side: `attn_fattn_wave_u8` is 2,198 instructions and §58's billed span is 1,857
(85 % of the kernel, 33 output stores inside); llama's `flash_attn_ext_f16<256,256,16,4>` is 19,512 and
§58's span is 13,536 (69 %, 129 output stores and 644 divide-class slots inside). So **all three §58 bills
include each kernel's once-per-launch epilogue, in different proportions** — for our wide form the epilogue
is 72 % of the billed span. Restricting llama to a loop with one round's shape (`d5e04..d8774`, 1,683 insns
against 64 wmma, 6 barriers) reads **≈26**, not 52.

Consequence, stated plainly because it changes what the canary can prove: **on a span that is actually one
round, our wide form already bills ~24 non-matrix slots per matrix slot, and the "2.09× amortization gap"
of §58 is at least largely a span-selection artifact.** The mandated canary (widest backward branch) is
reported below anyway, next to the round-scoped bill, and the epilogue mass (~11.2 k of the 14.3 k slots) is
NOT what items 1/2/4 can move, so the mandated number cannot fall anywhere near ≤ 70 without touching the
§05.6 epilogue, which this task forbids.

## 2. Item-by-item verification against the real code (before writing anything)

- **Item 3 (packed half rescale) is already paid for.** `v_pk_mul_f16 = 64` inside the round
  (`2bdcc..30844`), which is exactly 8 accumulator tiles × 16 halves / 2 per instruction — the ideal count.
  `attn_wide_kernels.h`'s `fattn_H2Regs v; v[e] = v[e] * rh` already gets paired by LLVM. Nothing to port;
  this is presumably why item 3 was not in the assigned set.
- **Item 5: the scale is applied per score element today.** `attn_wide_kernels.h`'s mask/max step calls
  `fattn_score_round` (`engine/attn_fattn_kernels.h:431`), which is `bf(fb(fb(dot) * SCALE))` — one
  `v_mul_f32` per score element, 32 per lane per round. **It is not folded into Q.** Folding is authorized
  and, for `SCALE = 0.0625f = 2^-4` (`prototype/attn_decode/attn_decode_kernels.h:34`), it is *exact*:
  scaling by a power of two commutes with rounding, so `f16(bf(q)·2⁻⁴) = f16(bf(q))·2⁻⁴`, every WMMA
  product is exactly scaled, the F32 accumulator's sum is exactly scaled, and
  `bf(fb(dot·2⁻⁴)) = bf(fb(dot))·2⁻⁴`. The only divergence is F16 underflow of a Q element whose
  `|bf(q)|·2⁻⁴` falls into the subnormal range (< 2⁻¹⁴ normal, losing bits below ~2⁻¹⁷), which cannot
  change a dot by more than ~1e-8 of a typical term. So the fold moves no band.
- **Item 1's premise is wrong about llama on this hardware, and the mechanism is bank conflicts, not
  address arithmetic.** `load_ldmatrix_swizzled` is called unconditionally, but the *decision* variable is
  CUDA-only: `ggml_cuda_fattn_mma_get_swizzled` (`fattn-mma-f16.cuh:354`) returns true only under
  `TURING_MMA_AVAILABLE`, so **on RDNA3 `swz = false` and `stride_tile_K = nbatch_K2 + 4`** — llama pads the
  stage row by 4 half2 (16 B) rather than swizzling. `mma.cuh:792` says so itself: `if (stride_bytes % 32 !=
  0) return offset; // Equivalent to padding with 16 bytes.` That matches what A11 §5 recorded
  (`prototype/attn_percost/A11-attribution.md` §5: our `fattn_kv_addr` row stride is 512 B and
  `wide_v_addr`'s is 128 B, both whole multiples of the 128 B bank group, so a wavefront's 32 lanes land in
  one bank group on every wmma operand read; llama pads for exactly this; **A11 bounded the fix at a few
  percent**, because measured LDS-pipe occupancy runs 5.3 % conflict-free → 19.2 % at 16-way). So the plan's
  sentence "the swizzle is a different lever and was never tried" is not right: pad and XOR swizzle are the
  same lever, and it is the LDS-conflict lever, whose value the ISA-slot canary cannot see at all.
  What *is* portable for the slot bill is the template-stride half of item 1 (per-lane hoisted base, one XOR
  per lane hoisted out of the roll, every tile address = `base + const`).
- **Item 2's `nstages > 1` path is dead code on this device.** `CP_ASYNC_AVAILABLE` is
  `!defined(GGML_USE_HIP) && __CUDA_ARCH__ >= 800` (`common.cuh:302`), and
  `ggml_cuda_fattn_mma_get_nstages` returns 0 when it is undefined (`fattn-mma-f16.cuh:369-377`). So llama on
  RDNA3 loads K and V **synchronously inside the round**, exactly as we do, and the cited
  `:1310-1325` preload / `:980-996` mid-round prefetch are not in the binary this repo billed. The port is
  therefore a port of the *idea* (issue next round's global loads early, commit late), which the brief
  explicitly pre-authorises ("on RDNA emulate cp.async with early global_load into registers + late LDS
  commit"). Cost that has to be said out loud: our stage is ONE 32 KiB region with V aliased over K, so a
  next-round K tile has nowhere to go in LDS without doubling declared LDS (32 KiB → 64 KiB, which would
  move the residency claim pinned by `wide_lds_halves()*2 == 32768`); the register form is the only option
  that keeps the LDS claim, and it costs 32 VGPR/lane (64 halves in flight) on a kernel already spilling 163.
- **Item 4 is the only assigned item whose target bucket is big in the round.** Round-scoped bucket table
  (`a2_round_bill.py` on `2bdcc..30844`, per matrix slot = /128): int_vec 5.27, delay 4.13, salu 3.98,
  select_mask 3.31, lds 2.07, wait 1.24, cvt 0.89, move 0.49, fmul 0.51, fma32 0.48, exp 0.26. The per-cell
  predicate costs `v_cmp_lt_i32 32 + v_cndmask_b32_e64 64 + v_cmp_eq/ne_u32 128` ≈ 200 of the round's 3,206
  slots (6 %), plus the exec-mask bookkeeping (`s_and_saveexec_b32 96`, `s_and_not1_saveexec_b32 64`,
  `s_cbranch_execz 32`) that divergent per-cell code forces. Specializing it into a tail body is a real, if
  bounded, cut. **Note the numerical argument that makes it legal**: `lanes_of(a, qrow) = lvalid + qrow`
  advances by exactly 1 per query row, and a wide block spans 32 positions, so the block's causal spread is
  31 lanes < `WIDE_STAGE = 64` — therefore the number of rounds needing a mask is **at most 2**, and every
  earlier round is fully live for every row of the block. That is a stronger version of llama's single
  `last_iter`.

## 3. Design decisions taken (before code)

- New form is selected by **`--attn-fattn-wide 2`** (new value on the existing knob; default stays 0),
  kernel symbol `attn_wide_llama_wave` / body `attn_wide_llama_body` in
  `engine/attn_wide_llama_kernels.h`. `wide_echo` gets a form field so the canary line distinguishes them;
  the existing `form=wide128` line for knob 1 stays byte-identical.
- Item 4's two bodies (full / tail) are two instantiations of one `template <bool kMasked>` inlined body,
  as llama instantiates `flash_attn_ext_f16_iter` twice. Register arrays reach it as structs by reference
  (`WllQ`, `FattnO16`), never as array parameters — the `pexp[8]`/`out[16]` spill trap.
- The tail body keeps the **select** form of the mask (`live ? score : -1e30f`), not an added mask value.
  Adding a precomputed sentinel costs `cmp + cndmask + fadd` per cell where the select costs `cmp + cndmask`,
  i.e. the add form is one slot per cell *more* expensive here, and the tail runs at most 2 rounds per block.
  This is a deliberate departure from the plan's "mask as an added value" sentence, with the reason above.
- LDS stays 32,768 B: the swizzle is a permutation inside a row, so declared LDS, the residency claim and
  `wide_echo`'s `lds=` field are untouched.

## 4. What actually landed (code, this change)

One new header, four plumbed files, nothing else:

- **`engine/attn_wide_llama_kernels.h`** (new, ~430 lines, included after `attn_wide_kernels.h` and only from
  `engine/kernels_attn.cpp`). Contents, in file order: the swizzle law (`wll_swa`, `wll_k_addr`, `wll_v_addr`,
  five `static_assert`s), the scale-folded Q loader `wll_load_q_tile` + scale-free `wll_score_round`, the
  pipelined K loaders `wll_load_k`/`wll_store_k` with `WLL_KTRIP = WIDE_STAGE / WIDE_WARPS = 8`, the
  `WllQ` register struct, `template <bool kMasked> wll_round(...)` (the round, one spelling),
  `attn_wide_llama_body`, and the `__launch_bounds__(WIDE_THREADS, 1)` kernel.
- **`engine/kernels_attn.cpp`**: the include; one branch at the wide launch site selecting
  `attn_wide_llama_wave` when `a.fattn_wide == 2` on the SAME grid (`wide_rtiles_of(rows)`, unchanged);
  `wide_echo` gained a `form` argument and prints `form=wide128llama` for 2 and `form=wide128` for 1 — knob
  1's line is otherwise byte-identical so every A12 harness grep still reads; the `--attn-fattn-wide`
  + `--attn-fattn 0` refusal text now names both non-zero values.
- **`engine/decode_run.cpp`**: `--attn-fattn-wide` parses `0|1|2` (throws on anything else, same wording
  pattern as before, now describing 2); the NUMERICS echo prints `attn_fattn_wide=wide_llama` for 2, `wide`
  for 1, `fattn` for 0; usage string and help text updated.
- **`engine/kernels.h`**: `AttnLayerArgs::fattn_wide` documented for value 2. The field was already `int`, and
  `decode_step.{h,cpp}` already forwards it as an int, so neither needed a change.
- **Nothing** in `engine/gate.py`, no numerics band, no admission list, no default moved. `gate.py` still
  accepts only `--attn-fattn-wide 0|1` (its argparse `choices`), so **the gate cannot price knob 2 and was
  deliberately not edited** — see §6 for what that means for the main agent's device pass.

Four checks worth naming, because each is a place the port could have been silently wrong:

1. **The swizzle is one law with a hoisted restatement, and the debug build compares them.** `wll_swa` is the
   layout's only statement of record; the round roll uses per-lane hoisted bases (`kb + 16L*STRIDE + (16t ^ kx)`)
   because the row a lane reads is always `ab + 16L` and `16L ≡ 0 (mod 8)`, so `row & XOR = ab & XOR` for the
   whole roll. That identity is what makes the swizzle cost one XOR per lane instead of 96, and it is also a
   second spelling — exactly the pair §20.1 says must not diverge. So `#ifdef QWEN_WIDE_DEBUG` the round
   compares hoisted vs `wll_*_addr` at **every** address the roll uses and prints a line on any mismatch.
   `hipcc -fsyntax-only -DQWEN_WIDE_DEBUG -I. -Iengine engine/kernels_attn.cpp` compiles it; the release build
   carries none of it. (Run the device version if a debug build is ever wanted — the host pass only proves it
   type-checks.)
2. **Granularity is 16 halves (32 B), not 8 B, and that is load-bearing, not taste.** `fattn_load_tile` reads a
   16-half tuple and the K writer fills one `uint4` per lane; a finer XOR would swap the two 8-half halves of
   a tuple by a row-dependent amount, i.e. permute the dot's k dimension — a wrong answer that no max-ULP band
   would catch. At 32 B the writer's `uint4` store stays 16-B aligned and wide, and the reader's
   `ds_load_b128` count is unchanged at 512 (`a13_round_bill.txt` §4).
3. **The pipelined K lands only after the round's V has been consumed.** Barrier order per round is A (this K
   published) → B (K consumed) → V staged → C (V published) → P·V → D (V consumed) → commit the prefetched
   tile. K and V alias one 32 KiB buffer (unchanged from knob 1), so a commit before D would overwrite bytes
   another wave is still reading — §20.1 with a one-instruction window instead of a whole layout.
4. **Item 4's licence is arithmetic, not hope.** A wide block spans `WIDE_POS_PER_BLOCK = 32` query positions
   and `lanes_of(a, q) = lvalid + q` advances by exactly 1 per row, so a block's causal spread is 31 lanes <
   `WIDE_STAGE = 64`. Hence `nfull = lanes_of(a, row0) / WIDE_STAGE` full rounds and **at most two** tail
   rounds, and a full round is live for every row-slot of the block by definition (`base + 64 ≤ lo_bound`).
   Rows past `nrows` are no longer forced to −1e30f in a full round; their accumulator holds finite garbage
   (the Q row read is still clamped), it is confined to that lane's registers because both butterflies pair
   `(t, t^16)` = one row-slot, and the publish guard keeps it out of memory.

## 5. Device witness (correctness only, no timing) — commands the main agent can re-run

`engine/gate.py` cannot be pointed at knob 2 (§6), so the witness is a direct two-run byte-compare:
one binary, one window, one variable, `--steps 0 --prefill-logits all`, every output file hashed and compared.

    # tmp/a13/ab_identity.sh — one fixture pair, GPU lock acquired/released by the script
    # tmp/a13/ab_all.sh      — the five-geometry sweep below
    # tmp/a13/promptlen.py   # extracts a fixture's pinned prefill_ids into an --ids-file
    # tmp/a13/dirdiff.py     # sha256 every file in both --out dirs, exit 0 only if all bytes match
    python3 tmp/a13/promptlen.py wide520f tmp/a13/ids.txt
    IDS=tmp/a13/ids.txt TAG=a13_identity bash tmp/a13/ab_identity.sh

Result, five geometries (8 / 128 / 295 / 500 / 1,024 rows — the last one emits two bodies because
`kMaxWaveRows = 512`, and both bodies printed the right form on both sides):

    p8 9/9 · p128 129/129 · dec300f 296/296 · wide520f 501/501 · wide1024f 1025/1025
    1,960 files compared, 0 differing — BYTE-IDENTICAL on every geometry

Each pair also printed its own canary fields: `attn_fattn_wide=wide` + `form=wide128` + `scratch=176` on the
knob-1 side, `attn_fattn_wide=wide_llama` + `form=wide128llama` + `scratch=720` on the knob-2 side,
`lds=32768` both. Wall 23 s for all ten runs; `tools/gpu_lock.sh util` at start 24–31 % (the box serves this
session's own tokens, so it never reads 0); lock acquired/released per pair; **no timing taken and none of
these wall figures is a measurement**.

This is the check that the byte-identity claim is actually about: a swizzle that aliased a tile, or a scale
fold that lost a bit, would still sit inside §10.1's cos bands at these lengths and would have been tabled as
a win. It did not need a band, and it did not need the gate.

## 6. Where the plan (or a previous note) was wrong about our code, or about llama's

1. **Item 1's premise — "llama swizzles" — is CUDA-only, and llama's RDNA3 answer is a pad.**
   `ggml_cuda_fattn_mma_get_swizzled` is gated on `TURING_MMA_AVAILABLE`, which requires `!GGML_USE_HIP`; on
   HIP the stride is `nbatch_K2 + 4`, and `mma.cuh:792` annotates that pad with "Equivalent to padding with
   16 bytes". So the plan's swizzle is *our* choice, not a port, and pad-vs-XOR are the same lever. A11 §5 had
   already priced the aliasing this fixes at a few percent of the window; what actually landed (and what
   §1 of the bill measures) is the template-stride + hoisted-base half of item 1, which took `int_vec` per
   round from 674 to 601 and `salu` from 509 to 288.
2. **Item 2's `cp.async` does not exist on this device in llama's code either.** `CP_ASYNC_AVAILABLE` requires
   `!GGML_USE_HIP`, and `ggml_cuda_fattn_mma_get_nstages` returns 0 without it, so llama stages K and V
   synchronously inside the round on RDNA3 exactly as knob 1 does. What was portable is the *position* of the
   load — after the score tile is consumed, before P·V — and the register form of it, which is also the only
   form that keeps the residency claim (an LDS double-buffer for K would need 64 KiB against the 32 KiB that
   `wide_lds_halves()` makes a residency argument, and `wide_echo`'s `lds=` field is a published number).
3. **Item 5 was already half-done, and the other half is exact.** `SCALE = 0.0625f = 2⁻⁴`
   (`prototype/attn_decode/attn_decode_kernels.h:34`), so folding it into Q commutes with both the F16 round
   and the F32 WMMA sum; the score, max, `exp` argument and rowsum are bit-identical, and the only new failure
   mode is a Q element whose `bf(q)` is under ~2⁻¹⁰ landing in the F16 subnormal range with fewer than bf16's
   7 bits left. Five geometries of pinned prompts did not exercise a single differing element, which is
   consistent with that being a tail event rather than a route — but "did not fire in five fixtures" is not
   "cannot fire"; the claim on the record is "exact except for F16-subnormal Q".
4. **Item 3 needed no port and the plan should say so.** LLVM already pairs the per-round accumulator rescale
   into `v_pk_mul_f16` — 64 per round in both knob 1 and knob 2, the ideal count for 128 halves/lane.
5. **Item 4's "mask as an added value" is a pessimization for us.** Our only mask is the causal bound, one
   compare off a per-lane scalar; the add-a-sentinel form llama uses exists because llama carries a real
   padding mask in LDS. The tail body keeps `live ? score : -1e30f`, which is `cmp + cndmask` per cell
   instead of `cmp + cndmask + fadd`.
6. **§58's "widest backward branch" is not the round for either of our kernels**, and that changes how big the
   gap looks: knob 1's widest span holds 14,324 of 14,403 instructions and all of the epilogue; knob 2's holds
   17,431 of 17,610 and both round instantiations. §58's own per-bucket targets (`int_vec 3,836 → ~2,300`)
   are quoted from that span. The round-scoped numbers are in the bill and are the ones to compare against
   llama's round; the widest-span numbers are the ones the mandated canary is written in. Both are reported.
7. **`prototype/attn_wide/run.py` does not exist**, so the brief's `grep -c gpu_lock prototype/attn_wide/run.py`
   has no target. A12's engine-side device work is `prototype/attn_wide/a12_engine_admission.py` (CPU-only; it
   prints `gate.py` command lines) and `engine/gate.py --fixture <f>`. `gate.py`'s `--attn-fattn-wide` is
   `choices=("0","1")`, so the §17-style admission of knob 2 needs a gate edit and that is a gate-policy
   decision, not this worker's to make.

## 4b. Did adding a second wide kernel to the TU move knob 1's machine code? No, and it is checkable

The header is included into the same TU as `attn_wide_wave`, which is a form whose codegen is mid-admission
(`prototype/attn_wide/A12-ADMISSION.md`), so "I only added a new function" is a claim that needs a witness
rather than an assertion. Same tool, two code objects, one mnemonic histogram of the whole function
(`tmp/a13/funcdiff.py BASE.S NEW.S attn_wide_wave`, comparing this window's pre-edit carve in
`tmp/a13/base/` against the post-edit carve in `tmp/a13/new/`):

    mnemonic                            base     new   delta
    s_code_end                           123       0    -123
    s_nop                                  11      38     +27
    TOTAL                               14499   14403     -96

Every work bucket is identical (`int_vec` 5,539 / `salu` 3,101 / `delay` 2,420 / `select_mask` 1,269 /
divide-class 960 / `lds` 491 / `wmma` 128 / `gload` 98 / `barrier` 13, both sides), and the only lines that
moved are `s_nop` scheduling padding and the `s_code_end` filler the slicer attributes to whichever function
sits last in the section — in the new build the llama kernel follows it, so that filler belongs to the new
function instead. So knob 1's code is byte-for-byte the same work, and the §58 figure of 110.9 reproduces
exactly from the new build (110.91, `a13_round_bill.txt` §1) — which is the stronger form of the same point:
the baseline re-measured inside the changed build agrees with the published number.

Knob 0 is untouched by a different and cheaper argument: the only new code on the launch path is a branch on
`a.fattn_wide == 2`, and `wide_echo`'s knob-1 line is the same text with the same fields (`form=wide128`),
which the device runs above printed on every knob-1 side.

## 5b. The witness is committed as `prototype/attn_wide/a13_byte_identity.py`, and it caught one of its own bugs

The scripts above live in `tmp/` and are gitignored, so the check that adjudicates this change would have
evaporated with the window. It is promoted to `prototype/attn_wide/a13_byte_identity.py` (fixtures as
arguments, defaults to the five below; GPU lock acquired/released per pair; the `NUMERICS attn_fattn_wide=` and
`ATTN_WIDE form=` fields are asserted per side, not just printed, so a side that silently ran the default
cannot be tabled as the other side). Re-running the whole list from the committed location:

    # python3 prototype/attn_wide/a13_byte_identity.py        (gpu use: 89 % at start; no timing taken)
    p8          rows=    8 files=    9 knob1=wide128/wide       knob2=wide128llama/wide_llama  BYTE-IDENTICAL
    p128        rows=  128 files=  129 knob1=wide128/wide       knob2=wide128llama/wide_llama  BYTE-IDENTICAL
    dec300f     rows=  295 files=  296 knob1=wide128/wide       knob2=wide128llama/wide_llama  BYTE-IDENTICAL
    wide520f    rows=  500 files=  501 knob1=wide128/wide       knob2=wide128llama/wide_llama  BYTE-IDENTICAL
    wide1024f   rows= 1024 files= 1025 knob1=wide128/wide       knob2=wide128llama/wide_llama  BYTE-IDENTICAL
    A13 BYTE-IDENTICAL: PASS                                     (1,960 files, 0 differing; 22 s wall total)

Its first version compared the two dumps by ABSOLUTE path, so every pair differed by its own directory and it
printed `files=0` plus 18 `DIFFERS` lines for a 9-file fixture. That is the same failure family as the
empty-parse gate drivers, and worth recording because the direction of the error was the good one: a
comparison that cannot match anything FAILED rather than passing over an empty list. Fixed to compare names
relative to each directory before any result was read from it.
