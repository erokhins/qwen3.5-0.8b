# A9 — why our attention costs ~1.9× llama.cpp's at LONG context: flop model, wasted work, traffic, levers (2026-10-19)

Task: `worker`, CPU-only, tip `6d0512f`. **No device work, no timing taken by me.** Every `ms` is quoted from
`progress/2026-10-21/performance-status-2026-10-21.md` §51/§50/§48/§46/§49 or is a *derived* ratio of quoted numbers with its formula
shown; every byte/slot count is a sum printed beside it or comes from the CPU mirror
`prototype/attn_percost/a9_waste_model.py` (this file's only new tool), which reproduces A2's two published
figures before it is allowed to publish its own (§2.0). Labelling follows A5's convention: **read** =
`path:line`, **derived** = formula shown, **quoted** = the § or file it came from.

Written in increments and appended as each part was finished, so a window compaction cannot lose a table.

---

## 0. The geometry, read from the source (every quantity with its `path:line`)

| quantity | value | source |
|---|---|---|
| model layers | 24, of which **6** are full attention (`layer % 4 == 3`) | `models/Qwen3.5-0.8B/config.json` `num_hidden_layers:24`, `full_attention_interval:4`; `engine/state.h:67` `is_full_attention(int layer) { return layer % 4 == 3; }` (quoted in `engine/decode_step.cpp:566` `attn_layer = is_full_attention(layer)`) |
| query heads `N_Q` | **8** | `prototype/attn_decode/attn_decode_kernels.h:23` |
| KV heads `N_KV` | **2** | `:24` |
| GQA share `GQA = N_Q/N_KV` | **4** query heads per KV head | `:25` |
| head dim `DH` | **256** | `:26` (and `config.json` `head_dim:256`) |
| wave rows per body `kMaxWaveRows` / `MAXSC` | **512** | `engine/decode_step.h:51`; `prototype/attn_decode/attn_decode_kernels.h:32` (`MAXSC`) |
| body *i* geometry | `pos0 = 512 i`, `lvalid = pos0 + 1`, `nrows = 512`, `lanes_last = lvalid + nrows − 1 = 512(i+1)` | `engine/decode_step.cpp:686-688` (`a.pos = pos0`, `a.lvalid = pos0 + 1`, `a.nrows = n`); `engine/kernels_attn.cpp:239` (`lanes_last = a.lvalid + rows - 1`) |
| per-row causal extent `lanes_of(row) = lvalid + row` | row 0 sees `lvalid` lanes | `prototype/attn_decode/attn_decode_kernels.h:203` |
| lane blocks a row uses | `blocks_for_row = 1` if `full && lanes_of ≤ 512` else `a.split` | `prototype/attn_decode/attn_decode_kernels.h:220-223` |
| launch `split` | `split_for(lanes_last) = ceil(lanes_last / 512)` ⇒ **body i has split i+1**, body 0 is not split | `engine/kernels_attn.cpp:86`, `:258` |
| who streams | `fattn_covers = fattn_cover && row < nrows && blocks_for_row > 1` ⇒ **body 0 belongs to `attn_slim_wave`**, every body `i ≥ 1` is fully covered by the WMMA form | `prototype/attn_decode/attn_decode_kernels.h:547-554` |
| fattn grid | `(N_Q=8, split, fattn_rtiles_of(512))`, `FATTN_ROWS_PER_BLOCK = FATTN_WARPS_PER_BLOCK × 16 = 48` ⇒ **rtiles = ceil(512/48) = 11** | `engine/kernels_attn.cpp:494` (`const dim3 fattn_grid(N_Q, split, fattn_rtiles_of(rows))`); `prototype/attn_decode/attn_decode_kernels.h:543-544`, `:561` |
| rows per block / warp | 48 rows per block, **16 rows per warp** (`FATTN_MROWS`), one head per block | `engine/attn_fattn_kernels.h:187`, `:234`; `prototype/attn_decode/attn_decode_kernels.h:543-544` |
| KV lanes per round `FATTN_MLANES` | **16** | `engine/attn_fattn_kernels.h:188` |
| round loop bound | `for (base = lo_all; base < hi_all; base += FATTN_MLANES)` — the **block union** of the three warps' slices | `engine/attn_fattn_kernels.h:771` |
| per-row slice | `span = ceil(nrow / s_row)`, `blo = blockIdx.y · span`, `bhi = min(blo+span, nrow)` | `engine/attn_fattn_kernels.h:476-480` |
| causal mask inside a round | `live = covered && j >= blo && j < bhi`, else `sc = −1e30` | `engine/attn_fattn_kernels.h:846-848` |
| score tiles / PV tiles per round | `FATTN_KTILES = FATTN_VTILES = DH/16 = 16` | `engine/attn_fattn_kernels.h:189-190` |
| accumulator rescale | 16 tiles × 8 F32 = **128 F32 mults per lane per round** | `engine/attn_fattn_kernels.h:879-881` |
| LDS (shipped `_u8` default) | `3 × 4,096 (transpose) + 8,192 (K/V) = 20,480 B` → 3 blocks/CU → 9 wf/CU at 185 VGPR, spill 0 | `engine/attn_fattn_kernels.h:655-660`, `:667`; progress/2026-10-21/performance-status-2026-10-21.md §48 of `engine/performance-status.md` |
| split partial slot | `(2 + DH)` F32 = **1,032 B** per (row, head, lane-block) | `engine/attn_fattn_kernels.h:489-491` (`fattn_part_slot`), stride `(DH + 2)` |
| roof | WMMA `v_wmma_f32_16x16x16_f16` = **52.63 TFLOP/s** (1 MAC counted as 2 FLOP) | `prototype/issue_rate_probe/results.md` §1 row `irp_wmma_4` |

**The causal work factor, stated as a formula rather than as a number.** For a prompt of `W` lanes prefilled
from an empty cache, row `r` (0-based) attends `r+1` lanes, so the number of (row, lane) causal touches per
head is `T(W) = W(W+1)/2`. Each touch does `2·DH` MAC (score dot over `DH` dims + `DH` MAC of the P·V sum), so

```
MAC/pass  = L_attn · N_Q · 2·DH · T(W)  = 6 · 8 · 512 · W(W+1)/2  = 12288 · W(W+1)
FLOP/pass = 2 · MAC/pass             = 24576 · W(W+1)          [the probe's 2-FLOP-per-MAC convention]
```

This is A2's model with its two corrections kept (`prototype/attn_percost/A2-census.md` §0 correction 2b: the
query-head axis must be in the demand; §2: "Demand 2.062e11 MAC/pass" at 4,096). Cross-check against prototype/attn_percost/A2-census.md §2:
`W=4096 → 12288 × 4096 × 4097 = 2.0621e11 MAC/pass` — the number A2 prints, to five digits. ✓

---

## 1. The flop model and the achieved rates (traces, no new measurement)

### 1.1 The traced numbers, re-derived with the pass-count discipline

All three traces are the ones already on disk from progress/2026-10-21/performance-status-2026-10-21.md §51's hold (`tmp/gcc/llamaphase/`), read through
`tools/trace_phase_split.py`, which derives the pass divisor from a kernel of known per-pass site count and
never sums raw rows. Commands and derived divisors:

```
python3 tools/trace_phase_split.py tmp/gcc/llamaphase/our_pp16384/pp16384_kernel_trace.csv \
        --engine ours  --calibrate attn_prep        --per-pass 6  --tokens 16384
python3 tools/trace_phase_split.py tmp/gcc/llamaphase/our_pp32768/pp32768_kernel_trace.csv \
        --engine ours  --calibrate attn_prep        --per-pass 6  --tokens 32768
python3 tools/trace_phase_split.py tmp/gcc/llamaphase/llama_pp16384/pp16384_kernel_trace.csv \
        --engine llama --calibrate gated_delta_net  --per-pass 18 --tokens 16384
python3 tools/trace_phase_split.py tmp/gcc/llamaphase/llama_pp32768/pp32768_kernel_trace.csv \
        --engine llama --calibrate gated_delta_net  --per-pass 18 --tokens 32768
```

`attn_prep` is 6 sites **per body** (one per attention layer), not per prompt-pass, so the tool's divisor
`6/pass` yields a **per-body-pass** unit: at 16,384 lanes it derives 96 = 3 wave replays × 32 bodies.
Multiplying its `ms/pass` by the body count `W/512` gives the whole-prompt-pass figure progress/2026-10-21/performance-status-2026-10-21.md §51 prints — this is
the reconciliation the progress/2026-10-21/performance-status-2026-10-21.md §51 note asks for, stated once:

| width | tool's attention ms (per body-pass) | × bodies | whole-prompt ms/pass | progress/2026-10-21/performance-status-2026-10-21.md §51's row |
|---|---|---|---|---|
| 16,384 ours | 47.47 | ×32 | **1,519.0** | 1,518.91 ✓ |
| 32,768 ours | 92.32 | ×64 | **5,908.5** | 5,908.16 ✓ |
| 16,384 llama | 23.99 | ×32 | **767.7** | 767.54 ✓ |
| 32,768 llama | 48.33 | ×64 | **3,093.1** | 3,093.25 ✓ |

Kernels behind our attention column at 32,768 (traced ms per body-pass / launches):
`attn_fattn_wave_u8` **87.25** at 5.9/pass, `attn_combine` **4.07** at 5.9/pass, the remainder ≈ 1.0 ms is
`attn_prep` + `attn_slim_wave`'s early-out — i.e. **94.5 % of the attention phase is one kernel**, and
`attn_combine` is 4.4 %. The `_u8` name confirms the traced binary is the shipped `--attn-fattn-qstage
unstaged8` default (progress/2026-10-21/performance-status-2026-10-21.md §48), so no stale-binary risk on this read (AGENTS' 2026-10-17 canary row: the canary here
is the kernel *name*, which only the shipped spelling carries).

### 1.2 Achieved rates — and a correction to progress/2026-10-21/performance-status-2026-10-21.md §51's absolute rate

progress/2026-10-21/performance-status-2026-10-21.md §51 printed: "3.3 TFLOP of causal attention per pass at 32k in 5.908 s is **0.56 TFLOP/s**, llama's is **1.07**".
That numerator is 8× under §0's model, and the factor is exactly `N_Q = 8` (it is prototype/attn_percost/A2-census.md §0's correction-2b bug
reintroduced: the head axis missing from the demand). `0.4124 TFLOP/pass at 4096` (prototype/attn_percost/A2-census.md §2, head axis included)
× `(32768/4096)² = 64` is **26.39 TFLOP/pass**, and `26.39 / 8 = 3.30` — the number progress/2026-10-21/performance-status-2026-10-21.md §51 printed. The ratio
ours/llama is unaffected (both sides carry the same 8×), so **progress/2026-10-21/performance-status-2026-10-21.md §51's attribution survives**, but the absolute
utilisation does not read the same way when it is right, so here it is right:

| width | FLOP/pass (derived, §0) | ours ms/pass (quoted from progress/2026-10-21/performance-status-2026-10-21.md §51 and §49) | ours TFLOP/s | % of WMMA roof | llama ms/pass | llama TFLOP/s | % of roof | ours/llama |
|---|---|---|---|---|---|---|---|---|
| 4,096 | 0.4124 TFLOP | 100.79 (§49's traced pair, same hold) | **4.09** | **7.78 %** | 52.31 | **7.88** | 14.98 % | 1.93× |
| 16,384 | 6.596 TFLOP | 1,518.91 | **4.343** | **8.25 %** | 767.54 | **8.594** | 16.33 % | 1.98× |
| 32,768 | 26.389 TFLOP | 5,908.16 | **4.466** | **8.49 %** | 3,093.25 | **8.531** | 16.21 % | 1.91× |

Arithmetic, one row shown: `24576 × 32768 × 32769 = 2.6389e13 FLOP`; `2.6389e13 / 5.90816 s = 4.466e12 FLOP/s`;
`4.466 / 52.63 = 8.49 %`.

**What the correction changes, and what it does not.** It does **not** reopen the bandwidth question: 8.5 % of
the matrix roof still leaves the phase ~91 % under it, and progress/2026-10-21/performance-status-2026-10-21.md §51's falsifier (a bandwidth-bound phase cannot sit
that far under the roof) stands — indeed the AGENTS row "a bytes-demanded ÷ time figure near the DRAM plateau
is not evidence of a bandwidth wall" and decision **D21** ("a model fitted to a measurement cannot falsify
itself") are why §3 below refuses to conclude anything from a GB/s figure. What it does change: llama is at
**16 % of the roof, not 2 %**, so the remaining gap is *two kernels that both fail to fill the matrix pipe,
one of them twice as hard* — not "llama found the fast path and we did not". Anyone quoting progress/2026-10-21/performance-status-2026-10-21.md §51's 0.56/1.07
should quote these instead and say which model they used; I cannot edit `bench/` (rule), so this file is the
correction's home until the ledger is updated.

Note on the 4,096 row: it uses progress/2026-10-21/performance-status-2026-10-21.md §49's **traced** attention pair (ours 100.79 vs
llama 52.31 ms/pass, one hold, post-G1), not §47's untraced 150.16 ms family, because mixing a traced and an
untraced number across widths is the "ruler re-taken in a different window" trap
(progress/2026-10-21/performance-status-2026-10-21.md §46/§48). Two checks say the row is sound: its 1.93× is exactly the phase ratio §49
printed, and its 4.09 TFLOP/s sits between the pre-G1 rate A2 derived (2.924 TFLOP/s,
`prototype/attn_percost/A2-census.md` §2) and the post-G1 16k rate below it — i.e. the same 8× correction applied
to A2's number lands where this one does.


## 2. Wasted-work accounting at 32,768 lanes

### 2.0 The mirror, and what adjudicates it

`prototype/attn_percost/a9_waste_model.py` walks the launch geometry exactly as the device does — the same
`lanes_of` / `blocks_for_row` / `fattn_covers` / `span, blo, bhi` / block-union round loop the table in §0 cites
— and sums, per attention layer, the issued and the useful matrix work, the staged bytes, the duplicate loads,
and the rescale rounds. It is adjudicated before it is trusted (`--selftest`, all four must be exact):

```
  block-rounds/layer : mirror 94,752            prototype/attn_percost/A2-census.md §2 94,752            delta +0
  staged B/layer     : mirror 1,552,416,768     prototype/attn_percost/A2-census.md §3 1,552,416,768     delta +0
  demand1x B/layer   : mirror 68,736,253,952    prototype/attn_percost/A2-census.md §3 68,736,253,952    delta +0
  partial B/pass     : mirror 887,685,120       prototype/attn_percost/A2-census.md §4 887,685,120       delta +0
  MAC/pass           : mirror 2.0621e+11        prototype/attn_percost/A2-census.md §2 2.062e+11
SELFTEST PASS
```

Two independent cross-checks from the traces, same hold, no re-collection
(`prototype/attn_percost/a9_trace_kernels.py`, which puts the trace on a whole-prompt-pass basis):
`attn_prep 6.00/pass · attn_slim_wave 6.00 · attn_fattn_wave_u8 5.91 · attn_combine 5.91` at 32,768 — the
5.91 is 6 × 63/64, which is `attn_decode_layer`'s `ATTN_FATTN_SKIP` for body 0 (a one-pass wave allocates no
streaming partials, `engine/kernels_attn.cpp:354`) reproduced by arithmetic nobody typed in; and the per-kernel
sum reproduces progress/2026-10-21/performance-status-2026-10-21.md §51's attention row exactly (5,583.87 + 260.76 + 32.12 + 31.42 = **5,908.16 ms/pass**).

At 32,768 lanes one prompt pass is **64 bodies** (`W/512`), of which body 0 is one-pass (`split = 1`, owned by
`attn_slim_wave`) and bodies 1..63 stream with `split = i+1`. Per layer: **182,952 fattn blocks**
(8 heads × Σ splits 2,079 × 11 row-tiles), **6,090,680 block-rounds**, i.e. per pass 1,097,712 blocks and
36,544,080 block-rounds = **1,115.24 block-rounds per token** = 3,345.7 wavefront-rounds/token over all six
layers. Issued MAC/pass **1.437e13** against useful causal MAC **1.319e13** (the model's §0 causal figure is
1.31945e13; the 0.02 % difference is body 0, which is `attn_slim_wave`'s and never enters this kernel):

```
issued/useful = 1.0893   ->  8.20 % of every matrix slot this kernel issues is not causal work
   pad (row-slots out of range / not this kernel's)  2.94 %   = 4.23e11 MAC/pass = 0.84 TFLOP/pass
   round quantisation ceil(union/16)                1.25 %   = 1.80e11 MAC/pass = 0.36 TFLOP/pass
   block-union overhang (48 rows, one range)        4.01 %   = 5.76e11 MAC/pass = 1.15 TFLOP/pass
   total wasted matrix work                                     2.36 TFLOP/pass against 26.39 useful
```

| item | what it is | magnitude at 32,768 | bill per token | can a lever remove it? |
|---|---|---|---|---|
| **(a1)** last-16-row-tile padding | 512 rows in 48-row blocks = 11 rtiles = 528 row-slots; rtile 10's third warp holds rows 512..527, all past `nrows`, and still runs every round of the union (`engine/attn_fattn_kernels.h:467` returns the row with `covered=false`, and only a *block*-wide `!any_mask` returns at `:757`) | 8,593,280 of 292,352,640 row-slot-rounds = **2.94 % of issued MAC** | 12.9 MFLOP/token of pure padding | only by a 48-agnostic split of 512 (512 = 4×48 + 32: no warp count divides it evenly; 4 warps → 64-row blocks → 8 rtiles exactly, see §4 L-a) |
| **(a2)** ragged causal tail (round quantisation) | the union width is not a multiple of `FATTN_MLANES = 16`, so the last round stages and computes lanes no row of the block needs (`:771`, `:778`) | **1.25 % of issued MAC** | 11.0 MFLOP/token | no — it is the price of a 16-wide C tile, and the C tile is 16 wide because the WMMA is (`:192` static_assert) |
| **(b1)** union overhang | rows of a 48-row block differ in `lanes_of` by up to 47 and in `span` by `ceil(47/s_row)`, so the shared round range `hi_all − lo_all` is wider than any one row's slice (`:475-480`, `:749-757`) | **4.01 % of issued MAC** | 35.1 MFLOP/token | only by shrinking the block (fewer warps = less overhang, more stage traffic — the same trade in reverse, §4 L-a) |
| **(b2)** K/V staged past the diagonal | `if (j >= hi_all) j = hi_all - 1;` — a real global load of a duplicate row, K at `:778`, V at `:895` | 1,232,280 duplicate lane loads = **1.26 GB/pass** | 38.5 KB/token, ≈ 5.3 ms/pass at the 237 GB/s roof | yes, but it is 0.09 % of the phase's demanded traffic — worth nothing |
| **(c)** split path | Σ splits 2,079 ⇒ 1,064,448 partial slots/layer × 1,032 B × 8 heads × 6 layers = **52.73 GB written + 52.73 GB read back + 1.59 GB merged rows** at 32,768 | **107.05 GB/pass** of scratch traffic | 3.27 MB/token | capped split, §4 L-c — and this is the one item whose *traced* rate is at the DRAM roof: `attn_combine` traced **260.76 ms/pass** moving 54.31 GB = **208 GB/s = 88 % of the 237 GB/s roof** (`devices/strix-halo-memory-bandwidth.md:62`) |
| **(d)** per-round rescale + running max | 128 `v_mul_f32`/lane-round (`:879-881`) + 1 butterfly + `fmax` + `expf` for the max (`:848-858`) + 16 `expf` and 16 butterflies for the P exchange (`:868-875`) | 109,632,240 rescale lane-rounds/pass = **428,251 `v_mul` slots/token**; 7.7 % of A2's ≥1,654-slot round bill | 428,251 slots/token | batched over 4 rounds (llama's `nbatch_fa = 64`), §4 L-d1 — charged at A4-L1's *measured* pass-through, see §4 |
| **(e)** per-body / per-launch structure | bodies = W/512 = **64** at 32k; 4 attention launches per body per layer (`attn_prep`, `attn_slim_wave`, `attn_fattn_wave_u8`, `attn_combine`) | traced `attn_prep` 31.42 + `attn_slim_wave` 32.12 = **63.5 ms/pass = 1.07 %** of the phase; `attn_combine` 260.76 ms = 4.4 % (counted in (c)) | 1.94 µs/token of launch-shaped cost | launch structure is already closed as a variable (progress/2026-10-21/performance-status-2026-10-21.md §41, progress/2026-10-21/performance-status-2026-10-21.md §29); the two 16-lane fills and four barriers per round are kernel shape, not launch |

**(c) in the currency progress/2026-10-21/performance-status-2026-10-21.md §46 invented.** Post-G1 the attention quadratic is
`b_attn = 0.1803 ms/token ÷ 32,768 = 5.50e-06 ms/token²` and llama's is `0.0944/32768 = 2.88e-06`, so the
excess quadratic at 32,768 is `(5.50−2.88)e-06 × 32768² = 2,813 ms` — which is progress/2026-10-21/performance-status-2026-10-21.md §51's measured +2,815 ms gap,
derived rather than subtracted ✓. Against it: the split round-trip's cost is 54.31 GB read at a traced
208 GB/s (260.76 ms) plus 52.73 GB written at the same rate (~253 ms of `attn_fattn_wave_u8`'s time) =
**453–514 ms, i.e. 7.7–8.7 % of attention and 16–18 % of the excess quadratic.** That is larger than prototype/attn_percost/A2-census.md §4's
10–13 %, for the reason A2 could not know: progress/2026-10-21/performance-status-2026-10-21.md §48 shipped G1 and shrank the denominator (the excess was 3,420 ms
pre-G1), so the split path is a *bigger* share of a *smaller* gap. A2's byte figure reproduces exactly
(`--selftest` line 4), so nothing here contradicts it — the base moved.

**(a)/(b) as a ceiling.** All of (a1)+(a2)+(b1) removed is 8.20 % of the round's matrix work, and A4-L1's
measured conversion says a slot cut of budget *B* % delivers ≈0.217·*B* % of family (1.67 % delivered for a
7.7 % budget, progress/2026-10-21/performance-status-2026-10-21.md §47) — i.e. **≈1.8 % of the attention family**, not 8.2 %. Anyone pricing a padding-removal
lever off the 8.2 % is quoting an upper bound on a quantity that has never converted at better than one-fifth.


### 2.1 (a) The causal ragged tail and the last row-tile

`512 % 48 != 0`: with `FATTN_WARPS = 3` (`engine/attn_fattn_kernels.h:234`) a block owns 48 query rows and a
body owns 512, so `fattn_rtiles_of(512, 48) = 11` row-tiles = **528 row-slots for 512 rows**. In the last
row-tile (`rt = 10`) the third warp's rows are 512..527 — entirely past `nrows` — and `fattn_covers` returns
`covered=false` for them (`engine/attn_fattn_kernels.h:467`), while the round loop runs the union
(`:749-757`), so that warp executes every `ds_store`, `barrier`, WMMA, rescale and `expf` of every round and
throws all of it away. The loop only returns early when *no* row in the block is inside `mask_hi`
(`:756`), which never happens while one warp is out of range and the other two are in it.

Measured by the mirror at 32,768 lanes: **8,593,280 of 292,352,640 issued row-slot-rounds = 2.94 % of all
issued matrix work is a row-slot that cannot produce an output** (per layer; ×6 layers per pass). This is
size-independent at the top of the range — every body has the same 512-row shape — and it is also present at
4,096 (3.26 %). It is removable *only* by making the row-tile divide 512: 4 warps (64 rows, 8 row-tiles) or
8 warps (64 rows, 8 tiles) do; 5/6/7 warps do not. So (a) is not an independent lever, it is a side-effect of
L-a's warp count, and §4 shows the two directions of that trade fight each other.

### 2.2 (b) K/V staged past the diagonal, and the union overhang

Two different wastes, both from sharing one K/V stage across 48 rows:

1. **Round quantisation — 1.25 % of issued MAC.** The union `[lo_all, hi_all)` is rounded up to
   `FATTN_MLANES = 16` (`:771`), so the last round stages lanes no row of the block needs.
2. **Union overhang — 4.01 % of issued MAC.** The 48 rows of a block have 48 different causal `hi` values
   (spread ≤ 47 lanes at `FATTN_RSTRIDE = 48`) *plus* a per-row split span of `ceil(47/s_row)`
   (`:475-480`), so the shared range exceeds each row's own slice by up to ~48 lanes. Every lane of that
   overhang is a full 16×16 score WMMA **and** a full 16×16 PV WMMA.
3. **Literal duplicate loads.** `if (j >= hi_all) j = hi_all - 1;` is a real global load of a duplicate K row
   at `:778` and V at `:895`: **1,232,280 lane loads = 1.2619 GB/pass** at 32,768 (4,155,520 lanes if
   counted per head-column). At the 237 GB/s ceiling that is 5.3 ms/pass — 0.09 % of the phase. It is listed
   because the brief asked for it, not because it is recoverable.

Total geometry waste at 32,768: **issued MAC/pass 1.43697e13 against useful 1.31913e13, i.e.
issued/useful = 1.0893, and 2.357 TFLOP/pass is issued and discarded** (pad 0.845 + quant 0.359 + union 1.152).
At §1.2's traced rate (4.467 TFLOP/s) that discarded work would be worth **528 ms of the 5,908 ms phase (8.9 %)**
— an *upper* bound,
because a fully-masked round is cheaper than a live one (no P exchange, no rescale). The mirror charges
`fully-masked rounds = 0` at every width: no round is dead for the whole block, so the overhang is always
paying the P-exchange cost too.

### 2.3 (c) The split path: 107 GB/pass of scratch that the M=16 geometry creates

`split_for(lanes) = ceil(lanes / MAXSC)` with `MAXSC = 512` (`engine/kernels_attn.cpp:86`, `prototype/attn_decode/attn_decode_kernels.h:32`),
so at 32,768 lanes the split saturates at `kAttnMaxSplit = 64` (`engine/kernels_attn.cpp:82`) from body 2 on,
and Σ splits over the 63 fattn-eligible bodies is 2,079. That produces **1,064,448 partial slots per layer**,
each `(DH + 2) = 1,032` floats (`engine/kernels_attn.cpp:204-206`):

| quantity | at 32,768 lanes | at 16,384 | at 4,096 |
|---|---|---|---|
| partial slots, all heads, one pass | 8,515,584 | 2,158,592 | 143,360 |
| partial bytes **written** (inside `attn_fattn_wave_u8`) | 52.728 GB | 13.366 GB | 0.888 GB |
| partial bytes **read back** by `attn_combine` | 52.728 GB | 13.366 GB | 0.888 GB |
| merged rows written (the only non-waste part) | 1.585 GB | 0.780 GB | 0.176 GB |
| **split round-trip total** | **107.042 GB/pass** | 27.512 GB | 1.952 GB |

Why it is **107 GB and not 33 GB**: the partial layout (`engine/kernels_attn.cpp:204-206`) carries `m` and `l` **as floats**, because
"(D)'s error was pricing the buffer, not its content: the epilogue needs 16 rows × FATTN_TROW floats"
(cited verbatim at `:97`). So each slot is 1,024 floats of accumulator + 8 floats of `(m, l)` = 1,032 — and
the accumulator alone is 8× what a bf16 partial would cost. `attn_part_floats`
(`engine/kernels_attn.cpp:204`) has no width parameter, so no caller can make it narrower.

The traced witness that this is *real DRAM time* and not just an address count: `attn_combine` is
**260.76 ms/pass** at 32,768 (`a9_trace_kernels.py`) moving 54.31 GB (read 52.728 + merged write 1.585) =
**208 GB/s = 88 % of the measured 237 GB/s ceiling** (`devices/strix-halo-memory-bandwidth.md:62`). No other
part of this phase is at the roof (§3.2: the whole phase demands ≈1.5 TB/s — 598.7 GB of K/V plus 898.1 GB
of Q in 5.908 s — which is above what the part can even fetch). Charging the
write side at the same achieved rate puts the full round-trip at **453–514 ms/pass = 7.7–8.7 % of attention =
5.5–6.2 % of the wave** at 32,768. This is the one item in the census whose removal has a mechanism that does
not depend on the machine being issue-slot-bound.

### 2.4 (d) The per-round rescale and running-max chain

Per round each warp: the running-max fold + `fmax` + `expf` for the new max (`:848-858`), **128 `v_mul_f32`** to
rescale the live accumulators (`:879-881`, one per accumulator element = 16 tiles × 8 floats), then the P
leg's 16 `expf` + 16 butterfly shuffles to pack P through LDS (`:868-875`; A2 bills this as the round's SHUFFLE class in
`prototype/attn_percost/A2-census.md` §1).

At 32,768: **1,115.24 block-rounds/token → 3,345.71 wavefront-rounds/token** across the six layers, so the
rescale alone is **428,251 `v_mul_f32` slots/token**. A2 bills the whole round at ≥1,654 slots/wavefront
(`prototype/attn_percost/A2-census.md:117`), so this is 7.7 % of the round's slots — and A4-L1 *measured* what
a cut there is worth: removing the whole rescale (7.7 % of budget) delivered **−1.67 % of the family** at
4,096 (`prototype/attn_percost/A4-L1-verdict.md` §4), i.e. a slot→time pass-through of **0.217**, not 1.0.
Long context does not change the *share* (the bill scales exactly with the useful work: rounds/token grows
linearly in W like the causal MAC does); it changes what a family-% is worth in wave-% terms (70.9 % vs
30.0 % of the wave), so the same −1.67 % of family is **−1.19 % of the wave at 32,768** instead of −0.50 % at
4,096.

### 2.5 (e) Per-launch and per-body overhead at 32k, and the one-pass body

`kMaxWaveRows = 512` (`engine/decode_step.h:51`) makes a 32,768-token prefill **64 sequential bodies**; the
attention kernels launch 4× per attention layer per body (prep + slim + fattn + combine), so **1,536 attention
launches per pass**, and the trace confirms the count (5.91–6.00 per body-pass per kernel). What they cost:

| kernel | ms/pass at 32,768 | % of attention | note |
|---|---|---|---|
| `attn_fattn_wave_u8` | 5,583.87 | 94.5 % | the covered rows |
| `attn_combine` | 260.76 | 4.4 % | counted in (c) |
| `attn_slim_wave` | 32.12 | 0.54 % | real at body 0, early-out elsewhere (`engine/kernels_attn.cpp:453-457`) |
| `attn_prep` | 31.42 | 0.53 % | rope/norm/append, O(W) and not attention's fault |

So the body structure and the launch structure together are **≈63.5 ms = 1.07 % of the phase**, and
`--attn-launch 1` already closed launch *structure* as a variable (progress/2026-10-21/performance-status-2026-10-21.md §38: it "changes everything about launch
structure and none of it about total instructions"). Body 0's rows — 512 of 32,768, 1.6 % of the tokens — are
done by the non-WMMA form; its cost is inside the 32.12 ms line, so the F1 asymmetry costs at most 0.5 % of
this phase and cannot be the 1.9×.

### 2.6 The whole accounting, one line each

| source of the gap | ms/pass at 32,768 | % of attention | status |
|---|---|---|---|
| split scratch round-trip (c) | 453–514 (derived: 54.31 GB read at the traced 208 GB/s + 52.73 GB written at the same rate) | 7.7–8.7 % | removable by L-c, partially |
| geometry waste (a)+(b): pad + quant + union | 528 ms **if slots converted 1:1**; **≈106 ms at A4-L1's measured 0.217 pass-through** | 8.9 % ceiling / **1.8 % expected** | bounded by the round's shape; only 48-agnostic row tiles remove it |
| rescale/max chain (d) | **≈99 ms** from A4-L1's measured −1.67 % of family (the raw slot arithmetic says 61 ms — the gap is chain effect) | 1.7 % measured at 4k | batchable, L-d |
| per-launch / per-body (e) | 63.5 traced | 1.07 % | closed by progress/2026-10-21/performance-status-2026-10-21.md §38/41 |
| duplicate K/V lane loads (b3) | 5.3 if DRAM, 0 as L-served | 0.09 % | not removable |
| **sum on the expected-value basis** (split at the measured rate + geometry/rescale at A4's pass-through + launch/body) | **≈730–790** | **12–13 %** | every basis here is measured or A4-calibrated |
| sum if *every* ceiling converted 1:1 (not a prediction) | ≈1,150 | 19.4 % | shown only to bound the arithmetic |
| **what parity needs** | **2,815** (progress/2026-10-21/performance-status-2026-10-21.md §51's measured gap) | **47.6 %** | |

Why this sums to 12–13 % while §5 stacks the levers to ≈22 %: this table counts only **wasted work**.
The largest single lever in §4.1 (the residency rung, ≈16 % of attention) is not waste — it is the phase
waiting on dependency chains with too few resident wavefronts to hide them, which is why it appears in
§5's stack and cannot be itemised in this table.

## 3. Traffic: our M=16 tile against llama's tiles — and the D21 caveat that decides how much it is worth

### 3.1 What each side stages, and what one stage is shared by

| | ours, `attn_fattn_wave_u8` | llama.cpp `fattn_mix_raw` (ROCm, RDNA) |
|---|---|---|
| query rows per block | **48** = `FATTN_WARPS 3 × FATTN_MROWS 16` (`engine/attn_fattn_kernels.h:234,:187`) | **64** = `ncols1 16 × ncols2 4` (`engines/llama.cpp/ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu:10`) |
| KV lanes per stage | **16** (`:188`) | **64** = `nbatch_fa` (`engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:164`, `:13`) |
| stage bytes per round | 16,384 (K 16×512 + V 16×512) | 67,584 (K+V bf16, row stride `nbatch_K2 + 4 = 132` half2 → 64 × 132 × 4 = 33,792 B per tensor, `engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:2031-2033`) |
| **stage bytes per (row × lane) touch** | **21.33 B** (16,384 / 768) | **16.50 B** (67,584 / 4,096) |
| **at 32,768: staged K/V per pass** | **598.74 GB = 23.23 B/touch** | **426.20 GB = 16.53 B/touch** |
| Q operand | re-read from global **every round** (post-G1: `:398-410`, called from the score leg at `:815`) = **898.11 GB/pass = 34.85 B/touch** | `Q_in_reg`: Q is kept **in registers for the whole work unit** (`engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:18`, `:665`) and its shared tile is aliased rather than added (`:2041-2043`), so it is fetched **once** per work unit (64 rows × 256 dims × 2 B = 32,768 B): `512·W / (W²/2)` = **0.03 B/touch** at 32,768 (**derived**) |
| **total demanded per touch** | **58.08 B** | **≈16.6 B** |

The brief's framing was "16 rows per K/V stage vs 256" — the **256 is llama's `DKQ + DV`** (256 + 256 = 512
halfs per row), not their query-row count; their row count is 64 and their lane tile is 64. Corrected, the
**headline of this table is that we demand 3.5× llama's bytes per causal touch, and 60 % of the excess is Q,**
not K/V. The K/V difference alone is 1.405×, and its cause is **reuse depth**: 48 of our rows share a 16-lane
stage, their 64 share a 64-lane stage, and a stage amortises over (rows × lanes) while costing (lanes ×
stride), so theirs is geometrically 1.29× cheaper per touch (16.50 vs 21.33 B) and 1.405× once our round
quantisation is charged. The much bigger item is Q: our re-read Q at 898.11 GB/pass is **1.5× our own K/V
staging and 2.1× llama's entire demanded K/V traffic**. Its size is set by the round width, not the row count:
Q bytes per touch = `DH × 2 / lanes_per_round` = 512/16 = 32 B for us, against 512/64 = 8 B for a 64-lane
round and ≈0 for llama, who never re-read it. And it exists because G1 un-staged Q to buy the residency G1
then bought (progress/2026-10-21/performance-status-2026-10-21.md §48, `engine/attn_fattn_kernels.h:242-264`) — the same trade seen from
the traffic side: Q-in-LDS and 9 wavefronts/CU are mutually exclusive on this part, since 3 blocks × 3 warps ×
8,192 B = 73,728 B against the 65,536 B of LDS the part has (`:287-291`).

### 3.2 The D21 caveat, applied to our own number

Demanded ≠ DRAM, and at this width the gap is not a factor, it is three orders of magnitude:

- unique K/V **data** per pass at 32,768 = **0.201 GB** (6 layers × 2 tensors × 512 B/token × 32,768 tokens) —
  a 0.85 ms read at the ceiling;
- the *minimum* any per-head-block kernel must demand (each of the 8 head columns streams its group's prefix
  once) = **1.611 GB/pass**;
- we stage **598.74 GB/pass = 2,974× the unique data and 372× the minimum**; llama stages 426.2 GB = 265× the
  minimum. (prototype/attn_percost/A2-census.md §3's `uniq` column was 41× at 4,096 because it divided by the minimum-demand row, not by the
  unique-data row; both are printed now so the two cannot be confused again.)
- the whole phase demands ≈1,497 GB in 5.908 s = **253 GB/s** (K/V 598.7 + Q 898.1), which is *above* the
  measured 237 GB/s DRAM ceiling (`devices/strix-halo-memory-bandwidth.md:62`) — so a large share is
  **provably** cache-served, on a part with **32 KB L1 per CU and 2 MiB GPU L2**
  (`devices/strix-halo-official-specs.md:108`, `devices/strix-halo-memory-bandwidth.md:25,:85`). llama's
  demanded rate is 426.2 GB / 3.093 s = **138 GB/s**, under the roof: theirs *could* be DRAM traffic and ours
  cannot be. That is the direction the D21 caveat cuts here, and it is why the Q re-read is affordable in
  bytes and expensive in queue slots.

And the decision-relevant evidence that traffic is not the wall:

1. **progress/2026-10-21/performance-status-2026-10-21.md §51's own roof test**: the causal work at 32,768 is 26.39 TFLOP/pass; at the measured WMMA rate of
   52.63 TFLOP/s (`prototype/issue_rate_probe/results.md:15`) that is 501 ms against **5,908 ms measured** —
   8.5 % of the roof. Neither engine is where the wall is (llama 16.2 %).
2. **D21's falsification, already run on this exact kernel family** (progress/2026-10-21/performance-status-2026-10-21.md §20.7 of
   `prototype/attn_decode/results.md`, quoted by `prototype/attn_percost/A2-census.md` §3): *"[e1]'s 2× traffic
   reduction, measured, moved the wave by 0.7 %."* A bit-identical instruction cut moved the same kernel 1.52×
   (D21). Traffic cuts do not convert here; instruction cuts do; and A4-L1 showed even instruction cuts convert
   at only ~0.22 of budget.
3. **The counterfactual price of L-a's traffic saving** (model, §4): halving our stage traffic by widening the
   block *raises* issued matrix work by 12.9 % — we would pay real issue slots to buy traffic the machine is
   not short of.

**The one place traffic IS DRAM**: the split round-trip (§2.3) runs at 208 GB/s measured, i.e. 88 % of the
roof. That is why L-c is the only traffic-shaped lever this census is willing to fund, and why its prediction
is quoted in bytes-and-rate rather than in slots.

## 4. Candidate levers, priced

Denominators: at 32,768 lanes attention is **70.9 % of the wave** (`a9_trace_kernels.py`: 5,908.16 of
8,328.23 ms traced), so **1 % of attention = 0.709 % of the 32k family/wave row**; at 16,384 it is 55.8 % →
1 % of attention = 0.558 % of the wave. Two conversion rules, both measured on this box, are used for every
prediction below and nothing else is:

- **residency**: progress/2026-10-21/performance-status-2026-10-21.md §48's own datapoint — G1 moved 6 → 9 wavefronts/CU and delivered −22.09 % of family
  (`progress/2026-10-21/performance-status-2026-10-21.md` §48, `prototype/attn_wmma/E1-result.md` §6), so
  `time factor = (wf_new/wf_old)^(−0.616)`, where 0.616 = ln(1/0.7791)/ln(9/6) is §48’s own two points
  (6 → 9 wavefronts/CU for family ×0.7791). The independent anchor in progress/2026-10-21/performance-status-2026-10-21.md §31
  (4 → 8 wf/CU, family ×0.607, A2-census.md §2 C6) gives −0.72; the printed predictions use 0.616 and the
  difference between the two exponents is ≈2 % of family at 12 wf/CU. Both are **extrapolations beyond the
  measured pair** above 9 wavefronts/CU and are labelled as such.
- **slot budget**: A4-L1 removed 128 of ≥1,654 slots per round (7.7 %) and delivered **−1.67 % of family**
  (`prototype/attn_percost/A4-L1-verdict.md` §4) → a pass-through of **0.217**, charged symmetrically (an
  *increase* in issued work costs +0.217 per 1 %).

| lever | source site | LDS / register cost | predicted effect at 32,768 | verdict |
|---|---|---|---|---|
| **L-a** more query rows per K/V stage | `FATTN_WARPS` (`engine/attn_fattn_kernels.h:234`) — rows/stage = warps×16; the launcher's mirror `FATTN_ROWS_PER_BLOCK` (`prototype/attn_decode/attn_decode_kernels.h:544`) **must move in the same commit** (progress/2026-10-21/performance-status-2026-10-21.md §34's two-header trap, `AGENTS.md` table) | +4,096 B per extra warp (post-G1 transpose stage, `:247-264`) | see the variant table | **partially fundable, aimed at the wrong resource** |
| **L-b** more KV lanes per round | `FATTN_MLANES` (`:188`); the aliased K/V stage is `FATTN_MLANES*DH*2` (static_assert at `:280`) | +8,192 B per doubling (16→32: 20,480 → 28,672 B) | **rounds/token 1,115 → 565 (−49 %), rescale slots 428k → 217k, BUT LDS forces 3 blocks → 2, so 9 → 6 wf/CU = +28 % family by progress/2026-10-21/performance-status-2026-10-21.md §48's elasticity**, plus issued/useful 1.0893 → 1.1040 | **DEAD without a device run** — it is the residency lever in reverse |
| **L-c** cap the split above a lane count | `split_for` (`engine/kernels_attn.cpp:86`) and `blocks_for_row` (`prototype/attn_decode/attn_decode_kernels.h:220-223`) — one rule, two spellings, one commit | **zero** registers, zero LDS; `kAttnMaxSplit` stays 64 (it is the lane ceiling, not the split cap) | at 32k, cap 8: partial round-trip **105.46 → 24.50 GB/pass (−81 GB)**, widest-body serial depth **+2.13 % only** (1,504 → 1,536 rounds); at the traced 208 GB/s that is **≈350–390 ms = 5.9–6.6 % of attention = 4.2–4.7 % of the 32k wave** | **the one lever with a mechanism that does not depend on an unproven theory** |
| **L-d** cut instructions per round beyond A4 | the rescale (`:879-881`) batched over 4 rounds — llama's `nbatch_fa = 64`, documented as *"Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators"* (`engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:13`) vs our 16 | +24 VGPR (3 extra live C tiles) → ~209/thread; 4 blocks × 96 × 209 = 60,192 ≤ 65,536, so 3 blocks/CU survives | budget 96 of 1,654 slots = 5.8 % → **−1.26 % of family** at A4's measured 0.217 | **below the 3 % falsification threshold on its own** — bundle it, do not test it alone |

### 4.1 L-a in detail: the axis that exists is the warp count, and it trades waste for traffic

M=32/64 per warp is not a knob: the C tile's row is `lane % 16` (`:187`, `:192`), so "more rows per warp"
means more C tiles — and the live O accumulator is already **128 VGPR/lane** at M=16
(`prototype/attn_wmma/fattn_regfile_ladder.md` §0, derived from `:189`), so doubling it doubles the largest
single register consumer in the kernel. progress/2026-10-21/performance-status-2026-10-21.md §33 ("spill-free at 6 wf/CU BEATS spilled
at 8") is the measured reason that trade loses here. The
knob that exists is `FATTN_WARPS`, which raises rows/stage *and* wavefronts/block together. Post-G1 ladder
(`a9_waste_model.py --variants`, LDS read from the `__shared__` declarations at `:665` + `:247-264`; 64 KiB/CU
from KFD, per the `AGENTS.md` counter trap):

```
warps mlanes   LDS B wf/CU rtiles staged GB/pass  Q GB/pass rescale slots/tok issued/useful  prediction (progress/2026-10-21/performance-status-2026-10-21.md §48 elasticity + A4 pass-through)
    3     16  20480     9     11          598.7      898.1           428,251        1.0893  (shipped)
    4     16  24576     8      8          442.0      884.1           421,552        1.0723  +7.2 % family  (residency LOSE: 8 wf/CU)
    5     16  28672    10      7          390.3      975.6           465,216        1.1834  −4.5 % family
    6     16  32768    12      6          338.1     1014.2           483,609        1.2301  −13.9 % family, if 12 wf/CU is reachable
    7     16  36864     7      5          286.3     1002.2           477,871        1.2155  +19.7 % family (1 block/CU — the progress/2026-10-21/performance-status-2026-10-21.md §48 G0 failure, post-G1)
    3     32  28672     6     11          606.8      455.1           217,005        1.1040  +14.6 % family  (L-b, dead)
    3     64  45056     3     11          626.6      235.0           112,039        1.1400  +65.2 % family  (L-b, dead)
```

Read the third column against the last: **every variant that halves our stage traffic either loses wavefronts
or pays ≥ 12 % more issued matrix work.** At `warps=6`, K/V staging falls 44 % (598.7 → 338.1 GB/pass) and Q
demand *rises* 13 %, `rtiles` 11 → 6 but the 96-row union spans the ragged tail far worse
(issued/useful 1.0893 → 1.2301, i.e. +12.9 % of issued work discarded = **+2.8 % of family** at A4's
pass-through), and the wavefront-round bill rises 13 % because the rescale is charged **per warp per round**
(§2.4) — widening the block adds warps without shortening anyone's round. So the traffic L-a buys is bought
with the currency A4 proved is scarce, in a phase §3 shows is not short of traffic.

The residual case for L-a is residency only (9 → 12 wf/CU at `warps=6`, predicted −16.3 % family,
offset by the +2.8 % slot cost → **−13.9 % net**), and it is gated on two facts nobody has established.
(i) **the register file per CU**: 2 blocks × 192 threads × 185 VGPR = 71,040, and the per-CU VGPR file is
recorded as UNKNOWN on this part — clang's note block says 185 while `rocprofv3` reports `VGPR_Count` 248
for the same kernel, and the part exposes no register-file field (`prototype/attn_wmma/fattn_regfile_ladder.md`
§3, `prototype/attn_percost/A3-result.md` §2b, restated at `prototype/attn_percost/A5-GEOMETRY.md` §4.2:
*"Any future residency claim on this part must settle the register-file question first."*) If the file is
64K-class the driver will squeeze VGPR and spill, and progress/2026-10-21/performance-status-2026-10-21.md §33 measured that losing
that trade (spill-free at 6 wf/CU beating spilled at 8) costs more than the extra wavefronts buy.
(ii) that extrapolating progress/2026-10-21/performance-status-2026-10-21.md §48's 6→9 elasticity to 12 holds at all. **Both are
answerable without a timing** — the compile read of `prototype/attn_wmma/fattn_census_build.sh
--arg __launch_bounds__=192,2` + `prototype/attn_wmma/fattn_residency_read.py`, listed as the cheapest
residency witness in `prototype/attn_percost/A5-GEOMETRY.md` §4.2. There is also a cheaper way
to buy the same wavefronts that does *not* touch rows-per-stage: **TRG 8 → 4** (half the epilogue stage again,
`:247`) gives LDS 14,336 → 4 blocks/CU → **12 wf/CU with `warps=3`, the 48-row union, the stage traffic and the
issued-work ratio all unchanged**; it costs only 8 transpose passes instead of 4 in a once-per-launch epilogue.
That is the residency play this census would actually fund, and it is not on the brief's list — named here
because the brief asked for better candidates if found.

**L-a falsifier (progress/2026-10-21/performance-status-2026-10-21.md §36 paired window, ≥ 3 %)**: `tools/prefill_sweep.py --lanes 32768
--runs 3` per arm (both arms inside one
`tools/gpu_lock.sh` hold, ≥ 2 reps each, canary per the 2026-10-17 rule (the `--attn-fattn` banner's `warps=` /
`lds=` fields must differ between arms, or the arms are the same binary). **Live line: −3.0 % of the family at
32,768** (= −4.2 % of attention). **Falsifier fires if** the residency field in the banner reads 12 wavefronts
and family Δ > −1 % — that combination means residency is not the binding resource, which retires the progress/2026-10-21/performance-status-2026-10-21.md §48
elasticity as a predictor for > 9 and sends the model back.

### 4.2 L-c in detail: the cap, and the cheaper spelling of the same byte

The model's cap table at 32,768 (`a9_waste_model.py --caps --cap-width 32768`):

```
 cap  round-trip GB/pass  widest-body blocks  block waves  rounds/block  widest-body serial depth  vs uncapped
  64           105.457                 5632        46.93            32                     1504       0.00 %   (shipped)
  32            78.674                 2816        23.47            64                     1536       2.13 %
  16            45.805                 1408        11.73           128                     1536       2.13 %
   8            24.500                  704         5.87           256                     1536       2.13 %
   4            12.630                  352         2.93           512                     1536       2.13 %
   2             6.391                  176         1.47          1024                     2048      36.17 %   (too serial)
```

Cap 8 removes 77 % of a 105 GB/pass round-trip and costs the widest body **2.13 %** more serial depth,
because `ceil(64/8) = 8` blocks per row at 4,096 lanes each is still 5.87 block-waves deep against 120
concurrent blocks (40 CUs × 3 blocks/CU) — the machine stays fed. At 16,384 the same cap costs **0.00 %**
(its widest body is 32-split, so a cap of 8 leaves 4 waves of 128 blocks: still fed). At 4,096 a cap of 8
does bite (`--caps --cap-width 4096` is 0.888 GB/pass total, so there is nothing there to win and the wave
count drops: run the cap ≥ 16 at short widths).

Cost: **zero registers, zero LDS, zero instructions per round.** The mechanism is DRAM bytes at a measured
rate, which is why this is the one lever whose prediction is not filtered through the 0.217 pass-through.
Expected: **−4.2 to −4.7 % of the 32k wave**, −5.9 to −6.6 % of attention.

Two honest caveats, both pre-registered:
1. Changing the split **re-associates `attn_combine`'s merge**, so this is not a bit-identity change and cannot
   be judged by one. Per `AGENTS.md` ("numeric format: delegated") a 16/32-bit-level reassociation is admissible
   if the existing gates stay green: the §10.1 cos/meanrel bands of
   `specification/10-reference-dumps-and-gates.md`, enforced by `engine/gate.py`, with the derived must-catch set, the derived must-catch set
   of `prototype/attn_tile_model.py`, and `prototype/attn_wmma/run_fattn_admission.sh`. **It must still catch
   every injection in that set** — that is the test, not a mood.
2. The serial-depth penalty is real for *one-pass-shaped* bodies: `--caps` computes it on the widest body only.
   A cap also lowers the block count of every *middle* body, and middle bodies are where the machine is
   thinnest; the wave column above (5.87 at cap 8) is the honest guard, and it is > 1 by a wide margin.

**Cheaper spelling of the same byte, named for the record**: the partial is 1,032 floats because the accumulator
is F32 (`engine/kernels_attn.cpp:204-206`), so a **bf16/fp16 partial store** halves-to-quarters the same
105 GB with *no* change to block geometry, split count or `attn_combine`'s association — it changes the
partial's *format*, which `AGENTS.md`'s delegated-format rule explicitly allows if the gates stay green, and it
is cheaper to reason about than a cap. Both attack item (c); the cap is bigger, the format change is safer.

**L-c falsifier (progress/2026-10-21/performance-status-2026-10-21.md §36 paired window, ≥ 3 %)**: admission first —
`tools/longctx_check.sh --widths 16384,32768` (numerics only; it prints no timing, which is the point) and
`prototype/attn_wmma/run_fattn_admission.sh`; then the timed A/B in the shape of
`prototype/attn_percost/run_ab_a4l1.sh` (cap 8 vs shipped, both arms inside one `tools/gpu_lock.sh` hold,
≥ 2 reps), with `prototype/attn_wmma/run_attn_attrib_window.sh` for the phase split. Canary = the
`split=`/`blocks=` fields in the `ATTN_FATTN` banner **must differ between arms** (the 2026-10-17
stale-binary rule), and `tools/attn_ceiling_proof.py` must still exit 0.
**Live line: −3.0 % family at 32,768 with admission green. Falsifier fires if** the byte model is honoured
(the scratch bytes actually drop 77 %, verifiable from the banner's `split=`/`blocks=` fields) **and** family
Δ > −1 % — that
combination says the partial round-trip is hidden behind compute after all, and retires §2.3's 88 %-of-roof
reading for this kernel.

**L-b falsifier (no window owed).** L-b dies on declared bytes, not on a measurement, so its falsifier is the
compile read, not a burst: build with `FATTN_MLANES = 32` and read `group_segment_fixed_size` with
`prototype/attn_wmma/fattn_census_build.sh` + `prototype/attn_wmma/fattn_residency_read.py`. The lever is alive
only if that read is ≤ 20,480 B (i.e. the aliasing assumption in `:280-282` is wrong and a 32-lane round costs
what a 16-lane one costs). If it reads 28,672 B as the static_assert arithmetic says, the residency loss is
arithmetic and no progress/2026-10-21/performance-status-2026-10-21.md §36 window can rescue it; A5's own write-up of the same candidate reached the same verdict
with a *second* K/V buffer (`prototype/attn_percost/A5-GEOMETRY.md` §3.4: `57,344 B → 3 wf/CU → family
+64.7 %`), so the cheaper spelling this census modelled (aliased, 6 wf/CU) is the *best case* for L-b and it is
still net-negative.

**L-d falsifier (only ever inside a bundle).** Its predicted −1.26 % is under any ≥ 3 % live line, so a solo
window cannot adjudicate it and would be a wasted hold. If it ships inside an L-a or L-c change, the canary is
the ISA slot count of the rescale loop actually falling (`prototype/attn_percost/a2_round_bill.py`, which is how
the 1,654-slot bill was made) — and if the bundle's family result is ≤ −1 % while the round bill fell, that is
a second independent measurement that A4's 0.217 pass-through is itself too generous, which is a finding worth
recording even though the lever is dead.

### 4.3 L-d, and why it cannot be tested alone

The batched rescale (one max+rescale decision per 4 rounds, as llama does at `nbatch_fa = 64`) has a budget of
96 of 1,654 slots → **−1.26 % of family**, which is below the ≥ 3 % threshold at 32,768 and far below it at
4,096 (where family is 30 % of the wave → −0.38 %). Long context does **not** enlarge it: the rescale bill
scales exactly with the useful work (§2.4), so its *share* is width-invariant; only its wave-weight changes
(×2.36 from 4k to 32k). It is worth doing **inside** an L-a or L-c change that already forces a re-timing, and
worth a solo window never. Register cost +24 VGPR is affordable at 3 blocks/CU only if the RF question above
comes back 64K-class or better (60,192 ≤ 65,536 ✓, but the driver's actual grant decides, per prototype/attn_percost/A3-result.md §2b).

## 5. Verdict: does anything short of the llama-shaped rewrite close 1.91×?

**No. On the numbers above, nothing in the targeted-lever space does, and the census can now say by how much.**

Parity at 32,768 means our attention phase going from **5,908.16 ms/pass to 3,093.25** — **−47.6 %** of the phase
(`progress/2026-10-21/performance-status-2026-10-21.md` §51's traced table). Everything this census can point at and name adds up to:

| item | best honest case | basis |
|---|---|---|
| L-c split cap at 8 (−77 % of a 105 GB/pass round-trip) | **−5.9 to −6.6 % of attention** (−4.2 to −4.7 % of the 32k wave) | bytes × the traced 208 GB/s, the only lever whose rate is measured |
| L-a residency only, `TRG 8→4` (3 → 4 blocks/CU, 9 → 12 wf/CU) | **−16.3 % of attention** if the RF allows it; **−8 %** if the progress/2026-10-21/performance-status-2026-10-21.md §48 elasticity is already overstretched | progress/2026-10-21/performance-status-2026-10-21.md §48's 6→9 = −22 % extrapolated (prototype/attn_percost/A2-census.md §2 C6's progress/2026-10-21/performance-status-2026-10-21.md §31 anchor gives −21 %); unverified past 9 |
| L-a as a traffic lever (`warps=6`) | **+2.8 %** (net of the +12.9 % issued-work penalty at A4's 0.217) | model, §4.1 |
| geometry waste (a1+a2+b1) removed entirely | **−1.8 %** | 8.20 % × 0.217 pass-through |
| L-d batched rescale | **−1.3 %** | 5.8 % × 0.217 |
| launch/body structure | **−1.1 %** already spent (progress/2026-10-21/performance-status-2026-10-21.md §38/§41) | traced |
| L-b | **+28 %** — do not | progress/2026-10-21/performance-status-2026-10-21.md §48's elasticity, §4 |

Stack the two levers that have a mechanism at their optimistic ends — L-c at −6.6 % and the residency rung at
−16.3 %, multiplied because they act on different resources — and attention goes 5,908 → 4,620 ms/pass, which
moves the phase gap from **1.910× to 1.494×**: 1,282 of the 2,815 excess milliseconds, **45 % of the gap**, bought
with a register-file fact nobody has established plus a re-associated merge that has to clear the gates. L-c alone
— the only lever whose mechanism is measured rather than extrapolated — gives 5,908 → 5,518 ms = **1.784×**.
Take the residency rung at half its extrapolated elasticity, which is what its own progress/2026-10-21/performance-status-2026-10-21.md §36 falsifier is likely to
allow, and the stack is −13.9 % → **1.64×**. Every one of those outcomes still ends between 1.49× and 1.78×
where parity is 1.000×, so the answer to the brief's question is: **the targeted levers are worth between a
quarter and a half of the gap, and no combination of them reaches parity.**

### 5.1 Where the other four-fifths is, named mechanically

Both engines run the same instruction class on the same part. Correcting progress/2026-10-21/performance-status-2026-10-21.md §51's rate arithmetic (§1's note: its
TFLOP count is per KV-group, missing the `N_Q = 8` head axis — an 8× factor in the numerator only), the
long-context attention rates are:

| | ours at 32,768 | llama.cpp at 32,768 |
|---|---|---|
| causal FLOP/pass | 26.389 TFLOP (§0, derived from `models/Qwen3.5-0.8B/config.json` + `engine/state.h:67`) | same work |
| traced time | 5,908.16 ms | 3,093.25 ms |
| **achieved** | **4.47 TFLOP/s = 8.5 % of the 52.63 WMMA roof** | **8.53 TFLOP/s = 16.2 %** |
| resident wavefronts/CU | **9** (`engine/attn_fattn_kernels.h:287-291`, post-G1) | **8** (prototype/attn_percost/A5-GEOMETRY.md §3.1: 256 threads × 1 block/CU) |

That last row is the finding that reframes the item: **llama achieves 1.9× our rate with *fewer* resident
wavefronts than we now have.** So the 1.9× is not residency — progress/2026-10-21/performance-status-2026-10-21.md §48 already harvested the residency we were
missing — and it is not bandwidth (§3: we demand 3.5× llama's bytes and still do half the rate; and a
bandwidth-bound phase cannot sit 91 % under a roof it is not touching). What is left is **the work each side
gets per operand touch**, and it can be stated in one ratio:

- ours, per block-round: 48 rows × 16 lanes × 512 MAC = 393,216 MAC, against 16,384 B of staged K/V
  **plus 24,576 B of Q re-read from global** (`:398-410`, called from the score leg at `:815`) →
  **9.6 MAC per LSU byte**;
- llama, per KV tile: 64 rows × 64 lanes × 512 MAC = 2,097,152 MAC against 67,584 B of K/V and a Q tile that
  is fetched **once per work unit** into registers + a deliberate 900 B/thread scratch
  (`engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:18`, `:665`; the scratch figure is A2's trace row,
  `prototype/attn_percost/A2-census.md:78`) → **≈31 MAC per LSU byte**.

**3.2× the matrix work per byte that crosses the LSU.** That single number is what the 1.9× is made of, and both
of its causes are structural, not tunable here: our M tile is 16 because the only WMMA this part exposes is
`n_wmma_f32_16_16_16` (`:187`, `:192`, prototype/attn_percost/A3-result.md §1), and our Q lives in global because putting it in LDS costs exactly the
residency G1 bought (`engine/attn_fattn_kernels.h:242-264`, progress/2026-10-21/performance-status-2026-10-21.md §48). The Q re-read is ~1,100× redundant per layer at 32k (898 GB/pass against
0.805 GB of unique Q) — free only because L1 is 32 KB/CU and the block's Q working set is 24 KB, and paid for in
VMEM/LSU queue slots, which is the one resource this repo has never counted (prototype/attn_percost/A5-GEOMETRY.md §3.6 asked for exactly this: a
`SQ_WAIT*`-class `rocprofv3` pass over `attn_fattn_wave_u8`).

### 5.2 What to fund, in order

1. **L-c (split cap), now.** Zero register/LDS cost, one rule to move (`split_for` at
   `engine/kernels_attn.cpp:86` + `blocks_for_row` at `prototype/attn_decode/attn_decode_kernels.h:220-223`,
   same commit — progress/2026-10-21/performance-status-2026-10-21.md §34's two-header trap), a mechanism measured at 88 % of the DRAM roof, worth **−3.9 % of the
   32k wave**. Its risk is numerics policy, not performance: a different split re-associates the merge, so it is
   judged by the §10.1 bands of `specification/10-reference-dumps-and-gates.md` as enforced by `engine/gate.py` and `prototype/attn_wmma/run_fattn_admission.sh`, and it must still
   catch every derived must-catch injection (the `partalias` lesson: derive the case list from the geometry).
   The narrower-partial variant (§4.2) attacks the same byte with a smaller blast radius and should be costed
   beside it.
2. **The two free compile reads, before any geometry work** (`prototype/attn_percost/A5-GEOMETRY.md` §4.2's
   cheapest-witness table): the register-file question via
   `prototype/attn_wmma/fattn_census_build.sh` + `prototype/attn_wmma/fattn_residency_read.py`, and the
   `TRG 8→4` census. These
   decide whether 12 wf/CU is a real configuration or a spill, at zero device cost.
3. **A counter pass, not a kernel**: prototype/attn_percost/A5-GEOMETRY.md §3.6's `SQ_WAIT*`-class `rocprofv3` collection on
   `attn_fattn_wave_u8` at 16,384. §1 showed the round is not issue-slot-bound (8.3 % utilisation) and A4 showed
   it is not chain-bound; the only unmeasured candidate is the stall for the LSU/VMEM queue implied by §5.1's
   9.6-vs-31 MAC-per-byte. If that counter cannot name it, no model in this file can be improved further.
4. **Only then** the llama-shaped geometry. prototype/attn_percost/A5-GEOMETRY.md §5 priced it at family −63 % if it reaches parity and declined to
   fund it without this census. What this census adds: the parity target is **16.2 % of the WMMA roof** (llama's
   measured long-context rate), the gap mechanism is MAC-per-LSU-byte, and the required geometry is 64 Q rows ×
   64 KV lanes per stage with the GQA group inside the tile and Q never returning to memory — which on this part
   means 256-thread blocks, deliberate 900 B/thread scratch for Q, and a register budget that must be checked
   against the same file the residency ladder reads. Its progress/2026-10-21/performance-status-2026-10-21.md §36 falsifier, pre-registered: a paired window at
   16,384 and 32,768, both arms in one `tools/gpu_lock.sh` hold, ≥ 2 reps, ≥ 3 % family live line, canary = the
   banner's geometry fields must differ between arms; **abort line: if a prototype's traced rate does not clear
   6.5 TFLOP/s (76 % of llama's) on the covered rows, the geometry was not the reason and the item dies.**

### 5.3 What this census cannot see

- **No timing was taken here.** Every millisecond figure is progress/2026-10-21/performance-status-2026-10-21.md §51's / progress/2026-10-21/performance-status-2026-10-21.md §49's / progress/2026-10-21/performance-status-2026-10-21.md §47's / A4-L1's, re-derived or
  re-binned on CPU; per `AGENTS.md` a worker may not put a number in a timing doc, and this file is a census,
  not a ledger row. progress/2026-10-21/performance-status-2026-10-21.md §51's rates are corrected in §1 but `engine/performance-status.md` was **not** edited — that
  is the main agent's call, and the correction needs the same line-citation care the ledger requires.
- **Two extrapolations carry most of the lever predictions**: the progress/2026-10-21/performance-status-2026-10-21.md §48 residency elasticity beyond 9 wf/CU (the
  12-wf numbers in §4.1 are predictions, not measurements) and A4-L1's single 0.217 pass-through measured once,
  at 4,096, on a 7.7 % budget. If either is wrong the levers move; the 1.9× conclusion does not, because it is
  an arithmetic sum of ceilings, and the ceiling sum is a fifth of the gap.
- The split-path write-side cost (§2.3) assumes the write runs at the rate `attn_combine` was measured reading.
  If writes are cheaper there, L-c shrinks toward its measured read half (−4.4 % of attention).
- The mirror's issued-work model charges every issued row-slot × lane × 2·DH as one MAC and cannot see FMA
  contraction or the `s_waitcnt` placement the device emits (prototype/attn_percost/A3-result.md §2's standing caveat about CPU mirrors). It is
  adjudicated against A2's four figures (§2.1), which is the strongest witness available without a device.
- §2.3's 88 %-of-roof reading is `attn_combine`'s, on progress/2026-10-21/performance-status-2026-10-21.md §51's traces, at 32,768 and 16,384 only. The short-width
  split path (≤ 4,096) is 1.95 GB/pass and stays noise.


---

## 6. Artifacts, and how to re-run everything here (all CPU-only; no device, no timing by me)

| artifact | what it is | how to run |
|---|---|---|
| `prototype/attn_percost/a9_waste_model.py` | the launch-geometry mirror: blocks, rounds, issued/useful MAC, staged and duplicate bytes, partial traffic, rescale rounds. Defaults `warps=3, mlanes=16` are the shipping geometry. | `--selftest` (the four A2 figures must be exact), `--widths 4096 16384 32768`, `--caps --cap-width 32768`, `--traffic`, `--variants` |
| `prototype/attn_percost/a9_trace_kernels.py` | re-bases §51's traces onto a whole-prompt-pass basis and derives the pass count from `attn_prep`'s 6 sites/body (refuses if that division is not exact) | `python3 prototype/attn_percost/a9_trace_kernels.py 16384 32768` (reads `tmp/gcc/llamaphase/our_pp*/pp*_kernel_trace.csv`) |
| `prototype/attn_percost/a9_check_cites.py` | prints the source line behind every `path:line` cite in this file, so a wrong line is visible instead of plausible (`tools/doc_ref_check.py` checks paths and §labels; this checks the line numbers) | `python3 prototype/attn_percost/a9_check_cites.py prototype/attn_percost/A9-LONGCTX-CENSUS.md` |

Current state of the three checks (re-run at the time of writing): `--selftest` → **SELFTEST PASS** (4/4 exact);
`a9_trace_kernels.py` → attention total **5,908.16** at 32,768, matching §51's row to the cent;
`a9_check_cites.py` → **0 citations point outside their file**, none landing on a blank line.

**Nothing in this file is a timing.** The `ms` figures are §51's, §49's, §47's and A4-L1's; the derived ones
(`208 GB/s`, `453–514 ms`, `528 ms`, `389 ms`) are arithmetic on those plus geometry sums, and each shows its
formula. Landing any of them in `engine/performance-status.md` is a main-agent decision requiring a paired window.
