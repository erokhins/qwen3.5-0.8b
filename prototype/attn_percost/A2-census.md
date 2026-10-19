# A2 — attention's demanded work, its per-launch cost, and what `b` is made of (2026-10-19)

Task: finish the census a previous worker got through the tools of. It is a census, not an implementation:
**no engine, kernel or gate file was touched, no device work was run, and no timing here is mine.** Every ms is
quoted (§38/§41/§44/§45/§46, cited at the line) or labelled *derived* and given its formula. The two tools are
fixed in place (`a2_work_table.py`, `a2_trace_rows.py`, plus one small histogram helper `a2_round_bill.py`);
their defects and their fixes are listed in §5 of `a2_work_table.py`'s own docstring and in the notes below.

Two tools' worth of warnings first, because both changed numbers a reader would have cited:

1. **The unit bug the brief named is real and is fixed.** Section 3 used to print
   `2.971 TFLOP/s = 5644434225721.43 % of the WMMA roof`. The roof was held in SI (52.63e12) but the
   percentage divided a TFLOP/s rate by `52.63`. Correct: **5.56 %**. `pct_of()` now computes every
   percentage-of-a-roof and **refuses above 1000 %** (`--selftest-pct` witnesses both halves: it refuses the
   1e12 slip *and* still prints llama's legal 109 % of the scalar roof).
2. **Two more outputs could not have been right, and both were the K/V side.** (a) Section 4's byte columns
   were per-*layer* while the time was per-*pass*, understating every GB/s by the 6 attention layers; (b) the
   no-reuse demand was `visits × N_KV × …`, which omits the query-head axis — 8 heads each fetch their KV
   group from their own block. Both fixed, and the columns are now named apart: **`demand1x`** (one fetch per
   (row, head, lane) — what a kernel with no staging would demand), **`staged`** (what this kernel's own loops
   fetch, read out of the round count), **`uniq`** (each visible lane read once: the DRAM lower bound). The
   predecessor's phrase "unique K+V" was `demand1x`, which is **not** unique; the section-5 percentage that
   the brief asked me to check inherits that, and §4 below settles it.

## 1. Provenance and geometry, on three axes, both sides

`rocprofv3`'s `Grid_Size_*` are **threads per dimension**, not blocks; blocks = grid_threads / workgroup, and
reading only `_X` is what mis-read llama's launch in §38 and got fixed in §43. CUs and SIMDs come from KFD
(`simd_count 80`, `simd_per_cu 2` → 40 CU), never from `hipGetDeviceProperties`, and never from clang's
`Occupancy [waves/SIMD]` line, which is a constant (AGENTS.md, "Counter, occupancy and census traps").
`tools/trace_phase_split.py --geom` is the phase tool and it is fixed, but its `--geom` table aggregates by
geometry and **does not print the kernel name**, and it drops `Accum_VGPR_Count`, `SGPR_Count` and
`Scratch_Size` — which are exactly the columns a residency claim needs. `a2_trace_rows.py` is that one job
done over one CSV; it refuses a kernel matched 0 times and a non-whole block division, and prints the pass
divisor it derived.

**Ours.** `engines/` at HEAD `2cc9ed6`; traced window `tmp/gdn_walk/g4ab/trace_reg16` (§44's `g4-ab` hold),
passes derived from `attn_prep` 48 seen ÷ 48 sites = 1.

| kernel | workgroup | grid (threads) | blocks | wf | wf/SIMD issued | VGPR | AGPR | LDS | scratch | ms/pass |
|---|---|---|---|---|---|---|---|---|---|---|
| `attn_fattn_wave` (split 2…8, 7 rows) | (96,1,1) | (768, *split*, 11) | 176…704 (3,080 tot) | 528…2,112 | 6.6…26.4 | 248 | 0 | 32,768 B | 0 | 138.84 |
| `attn_combine` | (256,1,1) | (2048,512,1) | 4,096 | 32,768 | 409.6 | 16 | 0 | 0 | 0 | 6.15 |
| `attn_slim_wave` | (128,1,1) | (256,1,512) | 1,024 | 4,096 | 51.2 | 48 | 0 | 12,288 B | 0 | 6.04 |
| `attn_prep` | (256,1,1) | (2560,512,1) | 5,120 | 40,960 | 512 | 32 | 0 | 2,560 B | 0 | 4.69 |

Family **155.71 ms traced** = §44's attention phase; §41's **untraced** family row is **133 ms**, so tracing
inflates this family by ~17 % — quote ratios, per §22/§26's standing caveat. The grid decodes exactly:
`768/96 = 8` query heads, `grid.y = split`, `grid.z = 11` row-tiles of `FATTN_WARPS_PER_BLOCK × FATTN_MROWS =
48` rows (512/48 → 11). **The block counts section 2 derives from the headers reproduce the traced grid in
every row**, which is the census's internal cross-check, and `covered = 7/8` of the rows and 98.44 % of the
lane-work reproduces §41's independently-derived 87.5 %/98.4 % from geometry alone.

Resident occupancy: `s_q` 3×8,192 B + `s_kv` 8,192 B = **32,768 B/block**, KFD `lds_size_in_kb 64` → **2
blocks/CU = 6 wf/CU = 3.0 wf/SIMD** (`prototype/attn_wmma/fattn_residency_read.py`, which also reports the
compile side: `.vgpr_count 178`, 0 spill, 0 private segment). Note the discrepancy and what it does not
change: the trace's `VGPR_Count` column says **248** for the same kernel while the object's own note says
**178** — an unexplained grant/rounding difference, unresolved here, and the block count is 2 on either
number, so nothing in this census rests on it. The `196,608` in `engine/attn_fattn_kernels.h:736-747` is
LLVM's occupancy *hint* budget (196,608/1024 = 192 VGPR/thread without `__launch_bounds__`), not a measured
register file — the register file per CU is **unknown on this part** (`fattn_regfile_ladder.md` §3), and the
first version of this tool quoted 196,608 as if it were one and concluded "registers do not bind at 3 blocks".
LDS is the binding limit; §41's closure stands.

**llama.cpp.** Tree `engines/llama.cpp` at `797de63b2fa7396dcc8caca76834c699ac84e1c3` (tag `b11716`), clean,
commit date 2026-10-17 13:36; the traced CSV is §38's own (`tmp/gcc/llamaphase/llama_pp4096`, passes derived
`gated_delta_net` 288 ÷ 144 = 2). The traced instantiation
`flash_attn_ext_f16<256, 256, 16, 4, false, false, false>` maps onto the template at
`ggml/src/ggml-cuda/fattn-mma-f16.cuh:1798` — `<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
use_sparse>` — i.e. DKQ = DV = 256, ncols1 = 16, ncols2 = 4, so `ncols = 64`, and the RDNA config row for
that shape (`fattn-mma-f16.cuh:164`) is `nthreads 256, occupancy 1, nbatch_fa 64, nbatch_K2 128,
nbatch_V2 128, nbatch_combine 64, nstages_target 1, Q_in_reg true`, with
`__launch_bounds__(nthreads, occupancy)` at `:1797`. Config selection goes through `amd_wmma_available(cc)`
(`:244-246`), i.e. the WMMA path. Traced geometry:

| kernel | workgroup | grid (threads) | blocks | wf | wf/SIMD issued | blocks/CU | VGPR | AGPR | LDS | scratch | ms/pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `flash_attn_ext_f16<256,256,16,4,false,false,false>` | (32,8,1) = 256 thr | (2048,8,1) | 64 | 512 | 6.40 | 1.60 | 256 | 0 | **0** | **900 B/thr** | 51.65 |

So: **both engines cover the machine** (ours 704 blocks/launch at 2 resident per CU; theirs 64 blocks at 1
resident per CU by their own `occupancy 1`), and the residency figures that matter are **6 wf/CU (3.0/SIMD)
for us against 8 wf/CU (4.0/SIMD) for them** — they declare 1 block/CU of 256 threads, spend every register
the budget allows (256) and then spill 900 B/thread *deliberately*, which is the shape of a kernel that
bought registers with scratch rather than with LDS. Our 32 KiB of LDS per block is what caps us at 2 blocks.

## 2. The work table and the ceilings — the adjudication

`a2_work_table.py` section 1 derives the demand from geometry: a (row, lane, head) touch is 2·DH MAC = 4·DH
FLOP, so 4,096 lanes cost **0.4124 TFLOP/pass** (6 attention layers), 8,390,656 row-lane visits, and the
visits grow as W²/2 (3.44e10 MAC/layer at 4,096 → 5.50e11 at 16,384 → 2.20e12 at 32,768). That is §46's `b`,
and attention is the only term in this engine that has it.

Verbatim, corrected section 3 (roof convention: `prototype/issue_rate_probe/results.md` §1's T·FLOP/s column
counts an FMA as 2 FLOP — 3.65 T·MAC/s = 7.30 TFLOP/s — the same convention as the demand above):

```
# demanded math per pass at 4096 lanes: 0.4124 TFLOP whole-family, 0.4060 TFLOP on the rows fattn owns
#   fattn row / covered rows  138.83 ms ->  2.924 TFLOP/s =  5.56 % of the WMMA roof |  40.1 % of the scalar roof |  12.6 % of dot2
#   untraced family           133.00 ms ->  3.101 TFLOP/s =  5.89 % of the WMMA roof |  42.5 % of the scalar roof |  13.3 % of dot2
#   frozen VALU form          263.12 ms ->  1.567 TFLOP/s =  2.98 % of the WMMA roof |  21.5 % of the scalar roof |   6.7 % of dot2
#   FLOORS for the same math: WMMA roof 7.84 ms | 50 % of WMMA 15.67 ms | 25 % 31.34 ms | dot2 17.72 ms | scalar 56.50 ms
# llama.cpp's flash_attn_ext_f16 does the SAME math in 51.65 ms traced (S38) = 7.985 TFLOP/s
#   = 15.2 % of the WMMA roof and 109.4 % of the SCALAR roof, so it cannot be a scalar kernel.
```

The predecessor's note that **instruction class is not the gap** is **confirmed, with an ISA witness rather
than an inference**: in the carved engine object, `attn_fattn_wave` contains 17 `v_wmma_f32_16x16x16_f16`
sites; in the carved llama object, 1,584 `v_wmma_f32_16x16x16_f16` + 1,008 `v_wmma_f16_16x16x16_f16`. Same
instruction, same rung of the ladder (52.63 TFLOP/s, itself 88.8 % of its roof). §26's "parity is a
matrix-class question no matter what our own disassembly says" is now closed by *having reached* the class:
what is left is amortization, and llama's 109.4 % of the scalar roof independently rules out any "we are just
waiting on a slow instruction class" story for them.

Two derived bounds, both assumption-free (formula given, no occupancy guess):

* **Matrix pipe: 5.6 % busy.** Demand 2.062e11 MAC/pass ÷ (26.315e12 MAC/s × 0.13883 s) = 5.65 %.
* **Issue slots: ≤ 8.8 % busy.** The round body costs **≥ 1,654 instructions per wavefront-round** (the
  widest backward branch of the carved kernel, `g3_isa_bill.py` span 0x11520..0x13970, histogrammed by
  `a2_round_bill.py`: wmma 17, fmul 145 — of which 128 are the accumulator rescale,
  `engine/attn_fattn_kernels.h:620-623` — lds 146, barrier 11, exp 9, cvt 41, global stores 33, salu 248,
  moves 339, masks 125, delay 160, waits 71 → **96 non-matrix issue slots per matrix slot**). With 94,752
  block-rounds/layer × 6 = 568,512 block-rounds/pass × 3 wavefronts × 1,654 slots = 2.82e9 slots against
  0.13883 s × 231.5e9 slots/s = 3.21e10 available = 8.8 % (a *lower* bound on utilisation is implied, since
  1,654 is a lower bound per round).

The 17 `v_wmma` sites are consistent with 16 unrolled score tiles plus one rolled P·V loop — an inference
from the source's `FATTN_KTILES`/`FATTN_VTILES` loops against the object's count, not a decoded address map.

**So the WMMA wave is neither matrix-bound nor issue-bound: 84–94 % of its wall is neither.** That single fact
does most of the adjudication below, and it is the attention sibling of what G3 found for GDN and of the
AGENTS.md row "price a serial kernel in cycles-per-dependency at its realised occupancy, or do not price it".

| # | candidate for the 2.16× (§46's `b` ratio) | magnitude from this census | verdict |
|---|---|---|---|
| C1 | instruction class (matrix vs scalar) | both sides `v_wmma_f32_16x16x16_f16`; ours 5.56 %, theirs 15.2 % of the same roof | **dead — already closed.** §26's premise paid off; nothing further to buy here |
| C2 | useful-vs-issued cells (tile waste) | 4.2–7.3 % waste at every body of every width (§2 table) | **dead — 1.07× at most** |
| C3 | staging / K/V traffic | staged demand 67 GB/s = 7 % of the 918 GB/s cache roof, 41× the unique bytes and L1/L2-served; no DRAM wall | **dead as the rate gap**; see §3 |
| C4 | the split/combine path | 19 % of the bytes this kernel fetches; 10–15 % of the excess `b` at wide context (§4) | **alive but small — cannot hold 2.16×** |
| C5 | per-element instruction mix (non-matrix slots per round) | 96 non-matrix slots per matrix slot, but the machine only needs 8.8 % of its slots | **dead as priced per-element**; it is a *chain*, see C6 |
| C6 | resident wavefronts (latency hiding) at 6 wf/CU, LDS-bound | ours 3.0 wf/SIMD resident vs their 4.0 = **1.33×**; measured elasticity on this exact axis is §31's Bet 1 (4→8 wf/CU bought 1.647× of the family, log-elasticity 0.72) → 6→9 predicts **1.36×** | **alive, 1.33–1.36× — the largest single measured piece** |
| C7 | the round's dependent chain (4 `__syncthreads`/round, LDS round-trip, the 128-multiply rescale every 16 lanes) | the round costs 11–34× its own issue floor and 16–49× its matrix maturation; llama's own config batches the same rescale over `nbatch_fa = 64` KV rows, ours every 16 → **4× the rescale traffic**, and rescale is 128 of the round's ≥1,654 slots (7.7 %) | **alive, and the remainder lives here** — but it is not priceable per-slot (C5), only as chain length |
| C8 | tail / wave quantisation | ours 704 blocks ÷ 40 CU at 2 resident = 8.8 block-waves (~11 % tail); theirs 64 ÷ 40 at 1 = 1.6 waves (~37 % quantisation) | **dead — favours us** |

No single candidate has 2.16× in it. **1.33× of it (C6) is residency, and the other ~1.6× is the round's
dependent chain (C7)**; the two multiply to ≈2.16 on the traced family ratio 3.0× only up to the ~1.4× of
tracing inflation and body-0 routing, which is exactly the honesty limit of this table: **C6 and C7 cannot be
separated by arithmetic**, and §31+§33 already showed they interact (spilled-at-8 lost to spill-free-at-6).

## 3. K/V traffic census (and the D21 caveat, read from the geometry side)

Section 4, corrected, at 4,096 lanes. `staged` is read out of the loop shape, not booked:
`for (int slot = w; slot < FATTN_MLANES; slot += nw)` and its V twin (`engine/attn_fattn_kernels.h:554,634`)
fill 16 lanes per (block, round) cooperatively across the block's 3 warps, so staged bytes = rounds × 16
lanes × DH × 2 B × (K,V), summed over the kernel's own rounds:

```
#   per LAYER: demand1x 68,736,253,952 B | staged 1,552,416,768 B | uniq 37,748,736 B
#   per PASS: demand1x 412,417,523,712 B | staged  9,314,500,608 B | uniq    226,492,416 B
#   staging cuts LSU-level demand 44.3x below the no-reuse figure — the loop shape books 48x (rows
#   per block sharing one s_kv) ... The staged column is still 41.1x the unique bytes
#   traced fattn at 138.83 ms: staged 67 GB/s (7 % of the 918 GB/s cache roof) | uniq 1.63 GB/s | demand1x 2971 GB/s
```

The re-use factor is **44.3× against the 48× the loop shape books** — the §20.5 trap method applied honestly,
reading the loop and then checking the booked number against the derived one; the residual is edge rounds and
the partly-masked tail row-tile. What the staging does *not* do: (i) the 4 query heads of a KV group are 4
separate blocks, so cross-head re-use in global is zero; (ii) each body re-reads its whole prefix once per
48-row tile, which is why `staged` is 41× `uniq`.

D21 now bites the other way. `uniq` K/V is 0.23 GB/pass = 12× the 18 MiB working set the 918 GB/s roof was
measured with, and `staged` is 41× that footprint again — so nearly all of the staged demand is L1/L2-served
and the GB/s columns are a census, not a wall; meanwhile `demand1x` at 2,971 GB/s exceeds the 237 GB/s DRAM
roof by construction. **Bytes demanded ≠ bytes fetched ≠ bytes from DRAM**, and a traffic route cannot be
funded off any of the three without saying which.

## 4. The split/combine scaling, and its verdict

Partial slot = (2 + DH) F32 = 1,032 B per (row, head, split-block); `split = split_for(lanes_last)`
(`engine/kernels_attn.cpp:86`) with `lanes_last = lvalid + 511`, so body *i* has split *i+1* and body 0 has 1
and is not split at all. Per body the write is 512 rows × 8 heads × split × 1,032 B. Independent arithmetic
at 4,096 lanes: Σ splits over bodies 1..7 = 2+…+8 = 35 → 512 × 8 × 35 × 1,032 = 147,947,520 B/layer × 6 =
**887,685,120 B written per pass**, and `attn_combine` reads exactly that back and writes the merged row
(4,096 rows × 8 × 256 × 4 B × 6 = 201,326,592 B).

The predecessor's "≈1.6 % of unique K+V" **does not survive checking**, and neither does dismissing the path
on it:

* against `demand1x` it is **0.4 %** (the predecessor's 1.6 % came from per-layer bytes over a per-layer
  demand that omitted the head axis — two errors that partly cancelled);
* against what this kernel actually fetches from global (`staged`) it is **19 %**;
* against the *unique* K/V bytes it must read at least once it is **7.8×** — the scratch traffic is nearly
  eight times the size of the KV cache it attends.

All three ratios are **constant in W** (7.26 → 8.06 against `uniq`, 0.204 → 0.176 against `staged`), because
both sides are rows × lanes/MAXSC-shaped. So the split path is *part of* the quadratic term, not a separate
blow-up — and it is **differential** against llama, whose `flash_attn_ext_f16` keeps the online softmax in
registers (`Q_in_reg true`, `nbatch_fa 64`) and writes its O tile once, paying no such round-trip.
Time sizing (derived: bytes ÷ 237 GB/s, *not* a timing, and an upper bound since most of it is L2-served):
the round-trip costs 8 ms/pass at 4,096 lanes (5.3 % of the 155.71 ms family) and **452 ms/pass at 32,768**.
Against §46's excess quadratic term — (b_ours − b_llama) × W² = (5.920 − 2.735)e-6 × 32,768² = **3,420 ms**
— that is **13 %**, and the alternative route to the same figure (extrapolating `attn_combine`'s traced 6.15 ms
by the split sum 2,016/35 = 57.6×) gives 354 ms = **10 %**. So: **the split path is roughly a tenth of the
excess `b`, not the whole of it.** Stated plainly because it is a negative result: the wide-context route is
real, it is ~10–13 % of the gap, and killing a candidate is worth the same as funding one.
One routing detail the first version got backwards, and the trace settles it by instance count
(`attn_prep` 48 / `attn_slim_wave` 48 / `attn_fattn_wave` 42 / `attn_combine` **42** per pass, §41's one
integer): `attn_combine` is gated on `split > 1`, so it runs on the 7 split bodies per layer and the merged
row is written for the 3,584 covered rows only (176,160,768 B/pass); body 0 is finalised by `attn_slim_wave`
and never touches the scratch.

## 5. What `b` is made of, and the one bet A3

§46 measured our quadratic term at **2.164× llama's**. From the table in §2: **no single candidate contains
2.16×.** The pieces with measured magnitude are C6 residency (1.33× on the resident-wavefront ratio, 1.36×
on this engine's own measured elasticity) and C7 the round's dependent chain (the remainder, ≈1.6×, which is
the only place left once 84–94 % of the time is neither matrix-pipe nor issue-slot). C1 is closed (same
instruction), C2 is 1.07×, C3 has no wall in it, C4/§4 is ~10–13 % of the excess, C8 favours us.

**A3: buy the third resident block on the LDS axis — stage each warp's Q in 8-row half-tiles so `s_q` is
3 × 4,096 = 12,288 B beside `s_kv`'s 8,192 B = 20,480 B/block ≤ 21,845 B, which is 3 blocks/CU = 9 wf/CU =
4.5 wf/SIMD.** §41 closed this axis on exactly this number and named the price ("the epilogue down to
≤ 21,845 B … not approved by anything in this ledger"); this is the request to fund it. **[CORRECTED IN PLACE
2026-10-19, A3-result.md §2: the sentence this clause carried — that the half-tile spelling reaches that byte
count "without touching the epilogue's transpose or re-spilling Q into registers" — is false, and the byte
count is not reachable that way at all. `engine/attn_fattn_kernels.h:710` and `:216` put the epilogue's F32
transpose on the SAME per-warp buffer the Q stage occupies, and `:581` has the second WMMA operand consume all
16 query rows x all 256 dims from that buffer every round, so 3 x 8,192 + 8,192 = 32,768 B is the floor of the
shipped
arithmetic and any 8-row scheme drags the epilogue with it (4 transpose passes, not 2). What remains of this
paragraph — the elasticity, the 20,480 B target, the C6/C7 split — stands; what does not is the claim that the
byte count was cheap. A3 is dead at the source: see `prototype/attn_percost/A3-result.md`.**

Why this and not the other live candidate: the round's extra LDS
re-reads are ~146 slots against a round body of ≥1,654 — the bet spends the resource the kernel is sitting
idle on (8.8 % issue utilisation) to buy the resource it is short of (wavefronts per SIMD). And it is the only
axis on this engine with a measured elasticity: §31's Bet 1 moved 4→8 wf/CU and the family went **1.647×**.

Both denominators, pre-registered in BET/G3 form, medians of 3 in one hold, three sides (base / new-off /
new-on), main-agent device work:

| metric | base (§44 window unless noted) | live | dead | why |
|---|---|---|---|---|
| `attn` family traced ms/pass | 155.71 | **≤ 125** (1.25×) | **> 148** (−5 %) | elasticity predicts 1.36× → 114 ms; 125 is inside the pessimistic end and still a 30 ms wave win |
| `attn` family untraced ms (§41) | 133 | ≤ 106 | > 126 | same ratio on the ruler that ships |
| wave `wall_ms` at 4,096 | 453.65 | ≤ 435 | > 445 | row −30 ms → 424 ms; allows ~10 ms externality (3× G1's measured +6 ms) |
| **fitted `b`, `tools/prefill_scaling_fit.py`, 2,048–32,768** | **5.920e-06 ms/token²** | **≤ 4.3e-06** | **≥ 5.62e-06** | the term the bet is about; llama's is 2.735e-06, so "live" is ~a third of the way to parity |
| 16,384 / 32,768 wave ms (`tools/longctx_check.sh`, §45 rows) | 3,319.25 / 10,448.13 | ≤ 3,050 / ≤ 9,400 | > 3,270 / > 10,100 | the wide-context denominator; at 32k attention is the majority of the run, so a residency win must show here to be believed |

The `b` row and the 16k/32k rows are the denominators that matter: at 4,096 lanes only 7/8 of the rows even
reach `attn_fattn_wave`, and a change to the resident-block count is a fraction-of-family effect everywhere, so
**the wide rows carry the verdict and the 4,096 row is a regression guard.** Harness: everything needed exists
— `tools/prefill_scaling_fit.py`, `tools/longctx_check.sh`, the burst scripts — with two named requirements,
not new tools: (a) the fit must be run at 32,768 via `--ids-file`, since a 32,768-id argv field is 191,588 B
and dies at `exec` with E2BIG (§39's `MAX_ARG_STRLEN` trap, which silently drops the widest point the fit
needs); (b) the in-window canary is `prototype/attn_percost/a2_trace_rows.py`, which must read
**`LDS_Block_Size` 20,480 on the new side and 32,768 on the base**, same grid rows and same launch counts
(48 `attn_prep` / 42 `attn_fattn_wave` per pass) — `tools/trace_phase_split.py --geom` cannot serve as this
canary because it prints neither the kernel name nor `LDS_Block_Size`. Compile-side pre-admission before any
device time: `prototype/attn_wmma/fattn_ship_census.py` refuses a non-zero spill and
`fattn_residency_read.py` must print derived blocks/CU = 3. **Kill line:** if 3 blocks/CU is only reachable by
spilling (any `private_segment_fixed_size > 0`), the bet is dead on §33's measured precedent (spilled-at-8
lost to spill-free-at-6), and the answer is §41's other spelling (serialise the three epilogues through
`s_kv`) or nothing.

Cheaper follow-up, explicitly *not* A3: batch the accumulator rescale over 4 rounds (llama's `nbatch_fa = 64`
vs our 16), which removes 96 of a round's ≥1,654 slots. At 8.8 % issue utilisation, cutting issue slots is
not where the money is — that is exactly what the §2 bounds say — so this is worth a day only as a probe of C7,
not a funded bet.

## 6. What this census cannot see

* **It cannot attribute C6 and C7.** They move together in any legal geometry, and the census has no counter
  evidence: a `rocprofv3` VALU-busy / SIM-clock pass (or `SQ_WAIT_*`) would say directly what share of the
  84–94 % is barrier, LDS latency and dependent-chain. That is main-agent device work and it is the check to
  run before spending on A3 if the premise is to be confirmed rather than bracketed.
* **Traced ms are upper bounds** (§38's caveat, ~17 % on this family, §41's 133 ms untraced row against
  §44's 155.71 ms traced). Both sides are traced in their own windows and the §38/§46 same-window rule is
  respected for the *throughput* ratios; the traced attention ratio 2.69–3.02× and the fitted `b` ratio
  2.164× are **not** the same measurement and should not be quoted as if they were.
* **The 918 GB/s cache roof was taken with a float4 grid-stride reader** on an 18 MiB working set, not with
  this kernel's LDS-staging pattern, so "7 % of the cache roof" is indicative, not a bound.
* **The 1,654-slot round body is a static count from one backward branch**, from `g3_isa_bill.py`'s widest
  span; the PV leg is a rolled inner loop inside it, so the true per-round stream is somewhat larger. It is
  used only as a lower bound (hence "≤ 8.8 % utilisation"), and AGENTS.md's loop-bill traps (clang's function
  labels, `.LBB` annotations, `v_fmac_f32_e32` vs `v_fma_f32`) are why the histogram reuses `g3_isa_bill.py`'s
  span rather than re-deriving one.
* **`VGPR_Count` 248 (trace) vs `.vgpr_count` 178 (object)** is unreconciled, and the per-CU register file is
  unknown on this part, so no register-derived block count is claimed anywhere here.
* **llama's `-ub 512` is a matched choice, not a charitable one** (G3 §5): at `-ub 2048`+ their RDNA path
  could take a different config row entirely, and nothing in the ledger has measured that.
* **Body 0 (rows ≤ MAXSC) is out of scope of all of this**: 12.5 % of rows, 1.6 % of lane-work, and it runs
  `attn_slim_wave` by design (`engine/attn_fattn_kernels.h` COVERAGE), so any family denominator mixes two
  kernels — the census separates them in section 3 rather than dividing one by the other.

## 7. Artifacts and how to re-run (all CPU-only; no device, no timing by me)

```
python3 prototype/attn_percost/a2_work_table.py                      # the tables quoted above
python3 prototype/attn_percost/a2_work_table.py --selftest-pct       # the unit guard, both directions
python3 prototype/attn_percost/a2_trace_rows.py tmp/gdn_walk/g4ab/trace_reg16/prefill_kernel_trace.csv \
        --calibrate attn_prep --per-pass 48 --match attn
python3 prototype/attn_percost/a2_trace_rows.py tmp/gcc/llamaphase/llama_pp4096/pp4096_kernel_trace.csv \
        --calibrate gated_delta_net --per-pass 144 --match flash_attn
python3 prototype/gdn_walk/g3_isa_bill.py tmp/a2census/ours/co_335872.s tmp/a2census/ours/co_335872.co attn_fattn_wave
python3 prototype/attn_percost/a2_round_bill.py tmp/a2census/ours/co_335872.s 11520 13970
python3 prototype/attn_wmma/fattn_residency_read.py --asm tmp/a2w/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s \
        --kernel attn_fattn_wave --threads 96 --warps 3
```

Reused, not reinvented: `g3_isa_bill.py` (loop spans), `fattn_ship_census.py`/`fattn_residency_read.py`
(note-block read + residency arithmetic), `trace_phase_split.py`'s pass-derivation rule, `attn_tile_model.py`'s
read-the-loop method, `carve_codeobjects.py` inputs. `a2_round_bill.py` exists only to histogram a span that
`g3_isa_bill.py` already located, because that tool's class list has no matrix rung and cannot see `v_wmma`.
