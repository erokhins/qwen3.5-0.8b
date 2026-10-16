# `prototype/attn_m2` — row-tiled attention at M = 2 (route R1)

One block walks K and V **once** for **two** adjacent query rows instead of twice, with the causal mask
applied per row so each row attends exactly its own prefix. Funded by
`progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md` §6 and decision D5 item 3
(bit-identity required), against the diagnosis in §3.4 of that doc: `attn`'s byte floor is 7.7× *under* its
measured
time at every width, so the lever is traffic, not arithmetic — and arithmetic routes on this family are
worth only 1.4–1.6× (D16/D17).

**Status: bit-identity is green on device (including the MAXSC straddle) and bytes demanded falls to ~0.50
as modeled; the widths the prize is booked at are still `NOT MEASURED`.** One device window ran
(`prototype/attn_m2/results.md` §2–§4): four plans were blocked by a launcher bug since fixed and guarded
(§6), the controls it ran were vacuous for two reasons also fixed (§3), and the timings it took were all on
waves 2.5–5× **below this box's bandwidth plateau**, where M = 2's halved grid cancels its halved bytes by
construction (§7). No timing exists for this form at a width that can price it.

## The one-sentence design

Share the **fetches**, keep the **arithmetic private**. Two rows of a pair have `n` and `n+1` lanes, so
their per-row lane slices for a given lane block differ (`[512,1024)` vs `[513,1025)` at n=1024, split 2),
but the **union** of those slices is bounded by `span_max + (split − 1) ≤ MAXSC + 7`, so the union can be
staged once and each row can still reduce its own lanes, in its own order, into its own partial slot. The
union bound is `ATTN_M2_STAGE = MAXSC + ATTN_M2_MAXSPLIT = 520`, checked by `attn_m2_ok` and by the CPU
mirror for every legal `(split, n)` — worst observed 519.

## Why it is worth anything

From `m2_model.py traffic`, which reproduces the funding doc's own numbers from an independent census:

| width | M=1 byte floor | M=2 byte floor | `attn` measured (quoted §3.4 of the funding doc) | prize | doc §6 prize |
|---|---|---|---|---|---|
| 1024 lanes | 27.2 ms | 13.7 ms | 27.23 ms | 13.6 ms | 13.6 ms |
| 2048 lanes | 108.8 ms | 54.5 ms | 104.32 ms | 49.8 ms | 49.9 ms |
| 4096 lanes | 435.0 ms | 217.8 ms | 412.07 ms | 194.3 ms | 194.6 ms |

Quoted denominators: 237 GB/s read ceiling (`devices/strix-halo-memory-bandwidth.md`), 6 attention
layers (§05), `Σ_lanes` causal (§06.3). **None of these are measurements of this kernel.** The byte
counters on this part are not the witness (GL2C calibration failed by ~100×, decision D9) — the argument
is bytes-demanded arithmetic plus a timing, and `HANDOFF.md` says which of each.

## Why it might not pay: the honest costs

* **Occupancy.** `attn_m2_wave`: VGPR 50, SGPR 64, Scratch 0, **LDS 22,656 B/block** with 256 threads
  (8 warps). 65536/22656 = **2 blocks/CU = 16 wavefronts**, against the shipped `attn_tile_wave`'s
  12,288 B × 4 warps → 5 blocks = 20 wavefronts. That was the predicted cost and it was reported before any
  device number existed — but the first device numbers (0.90× on four short waves) are not evidence for it:
  all four were 16–32 blocks, below the ~80-block grid size at which this box's read bandwidth reaches its
  233–240 GB/s plateau, and M = 2 halves the grid as well as the bytes, so below the knee they cancel by
  arithmetic rather than by occupancy (§7). The plateau sweep reached 233–240 GB/s with 128-thread blocks,
  i.e. ~8 wavefronts/CU at the knee, which is *below* this kernel's 16 — so aliasing `s_k`/`s_v`
  (22,656 → 18,560 B → 24 wavefronts) is banked, not scheduled, and a 2-row/4-warp block is worse than the
  thing it was proposed to fix (LDS is sized by `M × G × STAGE`, not by warps, so wavefronts would fall to 8).
* **A single-row wave gains nothing and pays ~0.2 %.** The `decode7` case in `compare` shows ratio 1.001:
  a lone row has no partner to share with, and its self lane is staged twice. M = 2 is a **wave/prefill**
  route; decode steps should keep `attn_core`.
* **Short pairs share little**: 0.778 at lanes 1–8, 0.538 at 1–64, 0.502 at 513–520 (`traffic`). The
  wave's first pairs are the short ones.
* **A straddling pair shares least** (`pair-at-maxsc`: 0.635). The one-pass row walks its whole row in
  lane block 0 while its partner walks one slice; the partner's later slices are shared with nobody.
* **V is re-used slightly less than K** whenever the two spans differ, because the pair's chunk *k* are
  anchored at their own `blo` and the staged unions overlap. Printed, not hidden.
* **The self lane is fetched twice** (into the union stage from the cache, and into a dedicated
  `s_kself`/`s_vself` slot from `krope`/`vg`) — ~2 lane rows per block, ~0.4 % at a 512-lane block. It is
  that way on purpose: the shipped form reads the self lane from the stage, and keeping that memory the
  same is what preserves `NEG_SLOTK`/`NEG_SLOTV`'s bite.

## What is frozen and what is new

`prototype/attn_decode/attn_decode_kernels.h` is not touched. This header includes it, reuses `CoreArgs`,
`attn_prep`, **`attn_combine`** (the shipped merge — one address expression shared by writer and reader,
which is what §20.1 argues for), and the geometry rules, transcribed once into `__device__ __host__`
`m2_lanes_of`/`m2_blocks_for_row` (the shipped pair are `__device__`-only, and this file's census and
launcher refusal must run on the host). The tie to the shipped spelling is a CPU check
(`m2_model.py selftest`, which compares the two function bodies textually and the resulting lane cuts
block for block against `attn_tile_model.row_bounds`, itself device-witnessed).

Block shape: `grid (N_KV, split, ceil(nrows/2))`, 256 threads = 8 warps = **one warp per
(row, head-in-group)** pair; 4 warps of row 0 and 4 of row 1 share one K walk. `attn_m2_ok` refuses
`score_f32`, `fused`, a `split` shallower than `split_for_lanes`, and any union above `ATTN_M2_STAGE`.

## CPU gates (all green, all re-runnable, no device)

```
.venv/bin/python prototype/attn_m2/m2_model.py selftest                        # 23 checks
.venv/bin/python prototype/attn_m2/m2_model.py guard                            # the refusal bound (§6)
./prototype/attn_m2/attn_m2 --geom-guard                                        # same bound, C++ side
.venv/bin/python prototype/attn_m2/m2_model.py compare --anchor wide1024f       # 13 cases, 0 diffs
.venv/bin/python prototype/attn_m2/m2_model.py negative --anchor wide1024f      # 9 injections
.venv/bin/python prototype/attn_m2/m2_model.py traffic                          # the byte model
.venv/bin/python prototype/attn_m2/m2_model.py regime                           # the crossover (§7)
.venv/bin/python prototype/attn_m2/run.py negmatrix                             # plan x control matrix
.venv/bin/python prototype/attn_m2/run.py plans                                 # 12 plans, no device
```

`guard` and `--geom-guard` are the same assertion written twice — CPU mirror and C++ closed forms — and they
print identical numbers (13,744 geometries, worst single-block union 518 of 520, 0 phantom refusals, 0 leaks,
16 real plan geometries). `attn_m2_ok` refused four funded plans once by comparing a per-block bound against
a sum over blocks; a comment saying that check was implied is what failed, so the bound now walks the space
in both spellings and asserts *which* check fired (§6).

`compare` is the admission gate: **0 differing elements out of the whole tensor, per case**, on 5 pinned
spans, 7 built boundary waves and one real HF-pinned decode row at 1032 lanes. `negative` runs 9
injections against must-catch/must-not-catch sets **derived from the geometry and from the operands**
(`gqa_reach`), never hand-listed. Three of them exist only because M = 2 makes new failures possible:
`maskwide` (a row attends the pair's union — the bug the shared walk invites), `partrow` (the two rows of
a pair share a split-partial slot: §19's class at a new address), `headcarry` (row 1 seeded from row 0's
`(max,sum)`: §07.3/A-17's state leak in the first shape M = 1 could not have).

Two findings came out of writing those controls, both recorded in `results.md` and in the code comments:

1. **`gqaoff`/`gqa_head0` cannot be form-neutral checks.** The shipped *mirror* moves a row's self operand
   along with the group's cache reads; the shipped *kernel* stages the self row from `kv_write`
   (`attn_decode_kernels.h:593,628,656`), as this kernel does. The two mirrors therefore cannot agree bit
   for bit under those injections even though the two kernels agree. The assertion used instead is that
   both forms' wrongs land on exactly their own derived row sets, and that stage/cache provenance is
   undisturbed.
2. **A synthetic wave whose cache is zeros makes KV-group controls vacuous.** `wide_case`'s zeros leave a
   wave's first row attending exactly one non-zero lane — its own — so a group swap cannot reach it and a
   must-catch set that ignores this scores 0 and looks clean. `m2_model.wide_plan` therefore fills the
   whole cache by cycling the pinned prefill cache, and `gqa_reach` derives reachability from the bytes.

## Files

| file | what |
|---|---|
| `attn_m2_kernels.h` | the kernel: geometry, two-pass walk, staging bound, injections, launcher refusal |
| `attn_m2.hip.cpp` | device A/B harness: both forms, one process, per-row bit-diff, byte census, **no timer** |
| `m2_model.py` | the CPU mirror and every CPU gate |
| `run.py` | device plan writer + A/B driver. **Does not self-lock** `tools/gpu_lock.sh` |
| `build.sh` | `-O3`, `--offload-arch=gfx1151`, `-Wl,-rpath,/opt/rocm/lib` |
| `results.md` | skeleton; device columns say `NOT MEASURED` until the device window runs |
| `HANDOFF.md` | the numbered device commands, and which case proves what |
