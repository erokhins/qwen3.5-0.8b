# Prefill optimisation: decisions taken without asking, and the arithmetic behind them

Main agent's log for the pass the user opened on 2026-10-15 ("measure real load during the prefill
of a big file, analyse how the GPU is occupied, then plan experiments"). Scope set by the user in
the same session: **prefill only**; decode findings are notes for later, not funded work; prototype
before implementing is allowed; no approval round trips — decide, log, proceed.

Companion documents: `progress/2026-10-16/prefill-occupancy-2026-10-16.md` (the worker's instrument
plan and command list), `bench/performance-status.md` (the ruler),
`progress/2026-10-14/speedup-routes-2026-10-14.md` (the route index this log amends).

---

## D1 — The requested 20k-token prefill cannot be measured as asked, and that is a finding

`engine/kernels_attn.cpp:60` sets `kAttnMaxSplit = 8` and the staging is `MAXSC = 512`
(`prototype/attn_decode/attn_decode_kernels.h:32`), so the lane ceiling is **4096 tokens**;
`engine/prefill.cpp:49-57` says a prompt past it refuses rather than overrunning. A 20,480-token
prompt is therefore refused today. Measured proof (refusal text, and the wave behaviour right up to
the ceiling) is in the device pass this log feeds.

Decision: profile at the ceiling (4096) and at 1024/2048 for the scaling exponent **first**, and
raise the ceiling only after that, because the ceiling raise is a numerics admission (the streaming
merge's association depends on `split`) and should not be paid for before the profile says a
long-context route is worth anything. Cost of the raise, for the record: `a_part` is
`[wave_rows][8][split][258]` F32 (`engine/decode_step.h:155`), so split 8 → 40 moves it from ~68 MiB
to ~338 MiB at `wave_rows = 512`, against a box where `flash_serve` leaves ~4 GiB of the VRAM
aperture free (`mem_info_vram_used` 1.75 GiB of 2 GiB visible, GTT-backed beyond that).

## D2 — The KV cache is not big; the traffic bill is re-reading

`kNumKvHeads = 2`, `kHeadDim = 256`, bf16, 6 attention layers (`engine/state.h:41-52`) → **2 KiB per
token per attention layer, 12 KiB per token overall**, which matches the AGENTS.md figure. So

| context | KV per attn layer | KV total (all 6) | causal K/V traffic if every row sweeps its own window | re-sweeps of the resident set |
|---|---|---|---|---|
| 1,024 | 2 MiB | 12 MiB | 6.45 GB | 528× |
| 4,096 | 8 MiB | 48 MiB | 103 GB | 2,148× |
| 20,480 | 40 MiB | 240 MiB | 2,577 GB | 10,737× |

Traffic model: `Σ_lanes = T(T+1)/2` × 2 KiB × 6 layers, divided by the 237 GB/s read probe. It
reproduces the two existing ledger points: 27.2 ms modelled vs **26.74 ms measured** `attn` at 1024
lanes (ledger §3), and a 904 ms wave at 3072 vs **907.3 ms measured** (ledger §1). A model that fits
two independent widths to well under 1 % is worth pre-registering against, so it is written here
before the measurement: `attn` at 4096 lanes should be ≈ **435 ms** (33 % of the wave) and the wave
should land near 1.3 s.

## D3 — Why "just keep the layer's KV in cache" is not free, and what tier it is

The user's point — one layer's KV is 2–40 MiB, so it should be loadable once per layer and the traffic
concern evaporates — is arithmetically right about size. It runs into two facts.

1. **Reuse distance, not size, decides it.** Within one attention launch the reuse distance is one
   sweep of that layer's window, so a 2 MiB set is *not* automatically kept: the tiled form is one
   query row per block (`ATTN_TILE_M == 1`, `prototype/attn_decode/attn_decode_kernels.h:404-422`)
   with `L = 8` lanes staged per sub-block, and each block streams its own copy of the window from
   global. The 4× the tile flip bought was G = 4 heads sharing a staged row, not rows sharing one.
2. **The tier that would serve it is real, and this repo has already measured it**: `gdn_decode`'s
   state reads at **882.7 GB/s = 372 % of probe** because ~54 MiB stays cache-resident (ledger §4),
   and `devices/strix-halo-mixed-traffic.md:381` records the same warm state at ~876 GB/s. So the
   box offers ~3.7× DRAM for a working set of at least ~54 MiB, which covers a layer's KV at every
   context in D2 — including 20k's 40 MiB.

So the fast tier is big enough and the user's idea is sound; the open question is why `attn` is
nevertheless sitting at the DRAM rate (241 GB/s of demanded bytes) with a 2 MiB-per-layer working
set. Do not argue that from paper — measure it (D4).

## D4 — Three candidate walls for `attn`, and one instrument that separates them without counters

| candidate wall | mechanism | what it predicts | the lever that works | the lever that does nothing |
|---|---|---|---|---|
| **W1 DRAM byte wall** | every query row sweeps its window from memory | `attn` ∝ Σ_lanes at ~240 GB/s at any working-set size | row tiling M (one K/V walk serves M rows), fast-tier residency | anything instruction-side |
| **W2 instruction-issue wall** | the score/PV dots are VALU work, one lane-instruction per FMA | `attn` ≈ its own VALU issue floor regardless of where bytes come from | **MFMA-backed scoring** (bf16 5.667 TFLOP/s, `devices/strix-halo-compute.md`) | M-tiling, cache residency |
| **W3 latency/occupancy wall** | too few wavefronts per CU to hide dependent loads | `attn` flat against both bytes and instructions | more parallelism per row (split, wider blocks) | M-tiling |

W2 is not a guess: `progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md` §9 counted
`SQ_INSTS_VALU = 6.528 G` warp-instructions for `attn_core` at a 1024-lane wave and derived an
**issue floor ≈ 20 ms** (4 inst/CU/clk × 40 CUs × ~2 GHz). Back then the measured time was 109 ms, so
the conclusion "instructions are not binding" was correct *for that form*. The shipped tiled form
measures **26.74 ms** — the tile removed redundant K/V *fetches*, not VALU work, so the same
instruction count now sits near ~75 % of that same 20 ms floor. If that is right, D2's traffic model
matching the measurement to 1.7 % is a coincidence of two quadratics, and traffic-side work (M-tiling,
fast-tier residency) buys nothing while MFMA buys a lot. This is the same failure mode that killed
D-12: matrix-shaped work on a vector path (`progress/2026-10-14/speedup-routes-2026-10-14.md` §4.1,
730 GFLOP executed with no `v_mfma` in the binary).

**The separating instrument** (`prototype/kv_sweep/`, built by a worker, timed by the main agent):
read a KV-shaped buffer with `attn_tile_wave`'s access pattern over a resident working set of
2…256 MiB. If a 2–8 MiB set can be driven at ~876 GB/s, then `attn`'s 241 GB/s cannot be a bandwidth
wall and W2/W3 own the field. If even 2 MiB will not go past ~240 GB/s, W1 is real and the traffic
levers are the funded ones. A direct counter pass on `attn_tile_wave` (`SQ_INSTS_VALU`, `SQ_INSTS_MFMA`,
`SQ_INSTS_VMEM`, `SQ_WAVES`) is the same question asked head-on, if `rocprofv3 --pmc` is permitted to a
non-root user.

### D4b — The same four walls, priced (script `tmp/occ/walls.py`, all anchors from this repo)

| context | DRAM byte wall | fast-tier byte wall (880 GB/s) | VALU issue floor (§9's 6.528 G warp-inst @1024 → 20 ms, scaled by Σ_lanes) | same FLOPs on the matrix unit (5.667 TFLOP/s bf16) | linear + `gdn_rec` |
|---|---|---|---|---|---|
| 1,024 | 27.2 ms | 7.3 ms | 20.0 ms | 4.6 ms | 220.0 ms |
| 4,096 | 435 ms | 117 ms | 320 ms | 72.8 ms | 880.0 ms |
| 20,480 | 10,874 ms | 2,929 ms | 7,993 ms | 1,819 ms | 4,400 ms |

Measured today: **26.74 ms** at 1024 lanes. Three things fall out of this table, and they are the
reason the pass must measure rather than argue.

- The byte wall and the issue wall **both scale with Σ_lanes**, so a scaling-exponent measurement
  cannot separate them. Only (a) can-a-small-KV-shaped-working-set-go-faster-than-240-GB/s and
  (b) a direct `SQ_INSTS_VALU` count can. That is why `prototype/kv_sweep` is the funded instrument.
- At 1024 lanes the shipped form is already within ~25 % of the VALU issue floor derived from the
  *pre-tile* census, and the matrix unit could do the same arithmetic in **4.6 ms**. If that floor is
  real, M-tiling and cache residency are both dead at 1k and the prize is instruction-side.
- §9's census implies ~8,300 lane-ops issued per (row, lane, head) against a ~512-lane-op minimum for
  the two 256-dim dots — a ~16× tax of conversion, LDS addressing, masking and reduction. That is a
  fourth candidate (**W4, issued-instruction count**), cheaper than an MFMA rewrite and untouched by
  either traffic lever, so `kv_sweep` measures it directly with a second pattern that adds the
  bf16↔f32 round trip to an otherwise identical load+reduce.

## D5 — What is funded, in order, and the kill lines

1. **Profile the ceiling** — `attn` share at 1024/2048/4096 via `--time-waves`, device-busy coverage
   and the gap distribution via the kernel trace, and the `gpu_busy_percent` baseline-delta sanity
   check via the occupancy recorder. Funds everything else; costs ~1 min of device.
2. **`prototype/kv_sweep`** — decides W1 vs W2/W3. Kill line: if it shows a fast tier for small
   working sets *and* `attn` is at its issue floor, no traffic-side attention work is funded at all.
3. **Row-tiled attention (M query rows, one K/V walk, diagonal masked)** — funded only if W1. Bit
   identity is required on the per-row lane order, the way `--attn-tile 1` was admitted; a row tile
   that changes a row's reduction order is not admissible as a speed change.
4. **MFMA-backed scoring** — funded only if W2, and then it subsumes 3 (a row tile scored on the
   matrix unit takes both the traffic cut and the issue cut).
5. **Raise the lane ceiling** to make a real 20k prefill runnable — scheduled after 1–4 say the
   long-context regime is worth the numerics admission, and it needs its own band discussion
   (ledger §6 debt 4: wave-width independence is already a band question under `--gemm library`).

Long-context honesty for the record: at 1024 lanes `attn` is 11 % of the wave, so items 3–4 are
**long-context routes** and will not move the 1k headline (where `gdn_rec` is 51 % and route §6.1 of
the routes index is still the biggest single prize). The case for doing them anyway is that the
user's use case is a big file, and D2 says a big file is where the quadratic lives.

## D6 — Decode: notes only, explicitly not funded in this pass

- Ledger §2: `attn` in-situ is **2.996 ms/step at 4032 lanes** against 6 × 4032 × 2 KiB = 48 MiB of
  K/V per step, i.e. 0.21 ms at the read probe — **~14× its own roofline**. A row tile cannot help
  (M = 1 in decode), so the gap is W2 or W3, and it is bigger than anything prefill has found at
  that width. It also has no isolated burst row beside it (ledger debt 7), so the first action there
  is an instrument, not a kernel.
- Decode-side routes already parked: §6.2 fp16 on the matrix path (needs a user band decision),
  §6.3 weight quantisation (out of scope without an explicit request), §6.4 MTP on GPU (never
  priced on device).

## D7 — Instrument caveats found while setting this up

- `gpu_busy_percent` is a device-wide counter and `flash_serve` busy-polls the same iGPU
  (`tools/gpu_lock.sh` header: it holds ~98 % "utilisation" while emitting nothing), so a busy%
  sample is only meaningful as a **delta against a baseline window with no engine running**. The
  recorder was built to take that baseline itself and to say so if the counter is saturated.
- `rocprofv3` (v1.3.5) is present; kernel tracing works without root, `--pmc` availability is being
  tested — if it fails, D4's verdict must come from the microprototype instead.
- The engine's own `--time-waves` instrument is the cheapest family attribution and reconciles to
  0.009 % of the wave (ledger §3), so it, not the profiler, is the attribution source of record.

## D13 — numeric format: the user delegated the decision, with a test (2026-10-16, late)

Asked whether the WMMA fast tier's fp16 operands were "a user band decision". The user's answer,
verbatim in substance: **any 16-bit or even 32-bit arithmetic may be used if the gates still pass,
or if we cannot significantly widen the checks in the gates** — and the standing rule to apply
judgement and proceed rather than stop repeats.

So the decision rule for a datatype change in this repo is now:

1. Does the existing gate set stay green? If yes, adopt; no new band is needed, because the bands
   already bought are wide enough to cover it.
2. If a gate goes red, ask whether the check could be widened *without materially weakening it* —
   which for this repo has a concrete meaning: a band widened to admit an fp16 path must still catch
   the injected bugs that the negative controls derive (AGENTS.md: must-catch sets are derived from
   geometry, not hand-listed). If it still catches them, adopt and record the new band and what
   witnesses it.
3. Only a change that neither passes nor can be widened stays a user decision.

What that costs here is nothing, because of an asymmetry worth writing down: **fp16 has a narrower
range than bf16 but a wider mantissa** (1+5+10 vs 1+8+7), so inside fp16's range fp16 is strictly
*more* precise than the format every gate is calibrated to. The only ways the switch can make a
number worse are overflow above 65504 and flush-to-zero below ~5.96e-08. That makes the feasibility
question a range census, not a band negotiation.

## D14 — the fp16 range census: it fits, with ~800x margin

`tools/fp16_range_probe.py` over **every** pinned dump in `data/refs` (evidence:
`progress/bench/evidence/prefill-occupancy-2026-10-16/F_fp16_range_all_dumps.txt`), 22 098 tensors,
1.11 billion values, comparing on the BF16 bit pattern so no tensor is widened to f32:

| role | tensors | values | max abs | values over 65504 | below fp16 min normal | flushed to zero |
|---|---|---|---|---|---|---|
| activations | 18 332 | 597 778 432 | **82.5** | **0** | 4.92 % | 92 793 (0.0085 %) |
| logits | 406 | 513 525 760 | 35.5 | 0 | 0.0006 % | 0 |
| weights | 16 | 352 256 | 0.1738 | 0 | 0.23 % | 0 |
| fingerprint rows | 1 140 | 63 840 | 3.96e6 | 4 413 | — | — |
| token ids | 404 | 732 | 2.48e5 | 34 | — | — |

The last two rows are excluded by role, and that exclusion is the finding, not a detail: the first
version of the probe flagged **276 "overflowing" tensors**, and every one of them was either a
`*_fp.npy` fingerprint quadruple (whose first and fourth columns are magnitude-as-bits and run to
~1e6 by construction) or `input_ids`, which is an index tensor held as float32. A range census over
metadata is a census of nothing. The real margin on activations is 65504 / 82.5 = **794x**.

Two bugs found in my own instrument on the way, both recorded in the tool's header because both are
the repo's usual shape: (a) sign-stripping a bf16 pattern is `AND 0x7FFF`, **not** `abs()` of an
int16 view — 0xC190 is -12.5 and its int16 absolute value, 0x3E80, is the pattern for +2.5, which
made 351 tensors look like they overflowed; (b) the role filter had to be written before the verdict
was trusted, since a check whose population includes metadata cannot adjudicate a value question.

**Consequences, decided under D13 and not queued for approval:**

- The fp16 fast tier is unblocked. Ledger §6.2 and the route index's "needs a user band decision"
  note are superseded for the fp16 case: the gate set is the decision procedure, and the census says
  the range cannot be what breaks it.
- bf16 → fp16 for **weights** is exact in mantissa (7 bits into 10) and their max abs is 0.17, so a
  weight conversion cannot overflow at all; it is a free upgrade in precision, worth checking
  because it costs nothing to test.
- 4.92 % of activation values are below fp16 min normal and stay representable only as subnormals;
  92 797 values (0.0085 %) flush to zero. If a gate goes red on the switch, that flush is the first
  suspect and the second is an FMA-contraction/association change (the `q̃` lesson in
  `prototype/gdn_collapse/results.md` §2), not the mantissa.
- The census is fixture-bounded: it is a claim about pinned values, and the guard is to re-run
  `tools/fp16_range_probe.py` whenever a new fixture is pinned. A prompt that produced a hidden state
  above 65504 would show up there first.

## D15 — the probe's priority order flipped: the matrix ceiling is now the top number, not the issue floor

Written when the `issue_rate_probe` worker checkpointed mid-task (first pass died at its 2 h
deadline; resumed as run 56987460), because the reason it exists changed underneath it.

Order in the original brief (D8): (a) sustained VALU warp-inst/CU/clk on the attn dot shape, (b) KV
read GB/s against resident working-set size, (c) arithmetic-path rates as an afterthought. Order
after `tools/isa_probe.sh` and D13/D14:

1. **What a hand-written WMMA leg sustains, as percent of the 59.36 TFLOP/s roof.** This is now the
   single number the fast tier is gated on. Reason: the 4× is an instruction-level certainty
   (`v_wmma_f32_16x16x16_f16` exists, compiles, and 0 of 55 shipped `gfx1151` code objects use any
   matrix instruction), so the question is no longer "is there headroom" but "does a kernel we can
   write reach it at our occupancy". A kernel that sustains 2× the scalar rate makes the fast tier
   the biggest prefill lever since the tiled wave; one that sustains 1.1× means the 41 % lane
   coverage from §03 is the binding constraint and instruction choice is not.
2. **Where dot2 / dot4-i8 / dot8-i4 / `v_pk_fma_f16` sit between the 1-MAC floor and the 4× roof** —
   same run, and it prices the int8 future (which stays a stop-and-ask for bands: D13 exclusion) and
   the no-WMMA fallback for shapes too thin for a 16×16 tile.
3. The original two questions, demoted. They still decide the M-tiling route (D8 stands), but they no
   longer gate the biggest candidate win.

Two things the worker is told explicitly, because both are repo lessons rather than local ones: an
arithmetic leg whose measured op is not the op in the billed span is worse than a missing leg (so
`irp_lds_tile`, whose LDS reads LLVM peels out of the loop, is left **loudly unbilled** rather than
billed at the staging loop), and the WMMA lane layout gets a witness before any WMMA timing is
trusted — `--wmma-check` must print the ones / cadd / feederA / feederB results, since a wrong lane
map reproduces plausible numbers on a symmetric fixture.

## D16 — the matrix ceiling measured: 52.6 TFLOP/s, and `attn` was never at an issue floor

Device window 2026-10-16 ~02:20–02:50, `tools/gpu_lock.sh` held, zero subagents, blocking, best-of
20 launches × 5 reps. Table and caveats: `prototype/issue_rate_probe/results.md`; raw rows:
`progress/bench/evidence/issue-rate-probe-2026-10-16/G_raw_rows.txt`.

1. **`v_wmma_f32_16x16x16_f16` sustains 52.63 TFLOP/s = 88.8 % of the 59.36 roof**, against the
   engine's 4.696 TFLOP/s through hipBLASLt: **11.2×**. The 60 TFLOPS is therefore settled on
   device, not just on paper — the instruction reaches its roof in a loop and nothing in the shipped
   stack issues one (0 of 55 `gfx1151` code objects, `tools/isa_probe.sh`).
2. **`v_dot2_f32_bf16` sustains 23.28 TFLOP/s = 4.1× our current library path**, and it is a *bf16*
   instruction: no format change, no range question, only a reduction-association change (which still
   has to survive the §05.5 ordered-reduction logic, so it is a numerics change of a smaller kind).
   Cheapest big lever found so far, and it does not need the fp16 discussion at all.
3. **The 1-wide issue floor was wrong.** `rocminfo` says 2 SIMDs per CU, and the probe sustains
   0.493 (scalar) to 3.552 (WMMA) MAC/lane/clk against roofs of 1 to 4. Re-read against a 2-wide
   issue, `attn_tile_wave` is at **~47 %**, not the 92–95 % of §03 — so D8's discriminator resolved
   in favour of **bytes**, and M-tiling goes back above instruction-count surgery. §03's sentence
   stands as written history; this is the correction, and the reason it survived is recorded in the
   results doc §3: the earlier pass had no measured arithmetic ceiling beside its instruction count.
4. **One number that must not be quoted.** The probe's own `pct_of_ceiling` claimed 172 % and 248 %
   because its `mac_per_lane_clk` column was warp-instructions-per-CU-per-clock wearing a MAC label.
   Caught because a value above a roof is impossible by construction, which is the same tripwire that
   caught `41 wavefronts per CU` (D10). The corrected arithmetic lives in
   `prototype/issue_rate_probe/tabulate.py`; the binary's columns are flagged dead in results.md §0.
5. **Not yet licensed by this measurement**: any int8 route (integer legs are raw-dot ceilings at
   47–48 % of roof, no dequant cost modelled, and bands are a D13 exclusion), and any WMMA kernel
   that hand-rolls a fragment layout — the on-device `--wmma-check` proved the accumulating dot and
   the D==16+C wiring but its feeder probe was too degenerate to recover the k-slot pairing
   (`tile SKIPPED … NOT DECIDABLE`), so a real kernel uses the builtin's fragment abstraction.

Next, in order: (i) route the dot2-bf16 finding into the attention score loop, where the ordered
reduction already exists and the format is unchanged; (ii) prototype a WMMA score path through the
fragment abstraction with the layout question closed properly; (iii) re-take the byte-vs-issue call
for M-tiling now that the issue wall is half as tall as claimed.

## D17 — D16's headline retracted: the matrix prize over the engine is 1.4–1.6×, not 11.2×

Written the same night, from `progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md`. The
measurement in D16 item 1 is sound — 52.63 TFLOP/s out of a hand-written `v_wmma` loop, 88.8 % of
roof — and the comparison built on it was wrong in both of its inputs.

1. **The baseline was retired.** D16 compared the WMMA loop against 4.696 TFLOP/s. That number is the
   **F32-`C`** form of `rocblas_gemm_ex`; `devices/strix-halo-compute.md` has a section headed
   *"Second correction, same day: every rate above is an F32-`C` rate, and it is 6–8× below the same
   call"*, and the engine passes `HIP_R_16BF` for its output, so the engine is on the fast form:
   **33.2–37.7 TFLOP/s** at its own shapes, 22.3–23.3 across the 24-layer mix at `M ≥ 512`. Prize:
   **1.4–1.6×** on GEMM-shaped families, not an order of magnitude.
2. **"Nothing in the shipped stack issues a matrix instruction" was false.** The census behind it
   covered the `TensileLibrary_Type_*_fallback_*.hsaco` files because `llvm-objdump` can read those,
   and did not cover the tuned kernels, which are CCO containers and lazy `.dat` files it refuses.
   The arithmetic proves the tuned path is a matrix path without needing any disassembly: 33–37.7
   TFLOP/s is **above** the 2-MAC vector roof of 29.63 TFLOP/s, so no 1-MAC or 2-MAC kernel can be
   producing it. A census over the files you can read is not a census over the files that run.
3. **`dot2` is not a universal win.** At 23.28 TFLOP/s it is *below* `mlp_down`'s measured 28.59
   TFLOP/s in situ, so swapping that family to `dot2` would be a ~23 % regression. The arithmetic-class
   levers apply to our own scalar kernels and to families whose rate is far below any ceiling, not to
   the library's GEMMs, which are already near or above the vector roofs.

Guards taken, in the shape of a rule rather than a feeling: **before a number is allowed to size a
route, open the file it came from and look for a correction section** — this repo corrects itself in
place and marks the retired figure, and I quoted across that mark. And **a census must state which
files it could read**, since "0 of 55" and "0 of the readable 55" read identically and mean different
things. Both are in `AGENTS.md` with this entry.

What survives D16 unchanged: item 3 (the 1-wide issue floor does not exist, so `attn` sits near 47 %
of a 2-wide issue and traffic is the front wall), the corrected `pct_of_ceiling` trap, and item 5's
ban on hand-rolling WMMA fragment layouts before the layout is proven. Funded next, per
`ceiling-based-route-ranking-2026-10-16.md` §6: **R1, attention row-tiling at M = 2**, bit-identity on
the per-row lane order as the admission gate — the largest prize that needs no user decision
(194.6 ms of a 1307.75 ms wave at `M = 2`, 355.6 ms at `M = 8`), chosen over the fact-B routes precisely
because fact B turned out to be 1.4–1.6× and not 11×.

## D18 — D17's funded route ran: M = 2 is bit-identical and 7–10 % slower at every width measurable today

Window `attn-m2`, `prototype/attn_m2/results.md` §2–§4. Three findings, one of them a defect in my own
brief's assumption that the prototype could price itself at the funded widths.

1. **Bit-identity holds on device** — `ndiff = 0` for both outputs on every row of every plan that runs,
   including the MAXSC/MAXSC+1 straddle that M = 2 exists to prove, and bytes-demanded falls to ~0.50 as
   the traffic model predicted. The route is *correct*.
2. **It is slower where it can run**: ship 0.2953 / 0.2949 / 0.1348 / 0.0085 ms against m2 0.3167 /
   0.3240 / 0.1495 / 0.0094 ms — speedups **0.933, 0.910, 0.902, 0.906**, paired in one lock hold,
   best-of-40×5. The occupancy cost the worker named up front (16 wavefronts/CU vs the shipped form's 20)
   is bigger than the traffic saving at ≤ 520 lanes.
3. **The funded widths could not be measured at all.** `attn_m2_ok` feeds a *lane count* to
   `attn_m2_pair_union`, whose first parameter is a *row index*, so every plan above ~520 lanes is
   refused as exceeding the stage bound — including `plan-tail` at 1032 lanes and both straddling plans.
   At 1032 lanes with `split = 3` the true union is ~346 of a 520 bound, so the refusals are wrong. Until
   that is fixed the 13.6 ms / 194.3 ms prize booked at 1024 and 4096 lanes is unmeasured, and finding 2
   must not be read as the route's verdict.

Also recorded rather than smoothed over: with those refusals the only plans that execute are short ones,
so **eight of nine device negative controls had an empty derived must-catch set** (a streaming row cannot
exist at 1–8 lanes) and proved nothing; only `maskwide` fired and was admissible. And the harness itself
faulted before its first comparison — `diff_rows` called a `__syncthreads()`-containing reduction from
inside `if (threadIdx.x == 0)`, which aborts the queue on GCN instead of miscounting. A green CPU mirror
was worth nothing until that was fixed.

## D19 — two measured facts say `attn` sits where two equal walls meet, so no single-lever route can move it

Window `attn-m2-2`. With the refusal fixed (see below) the four above-the-knee waves ran, and the result
is not "M = 2 is slow" — it is that **dividing bytes by measured ms on both sides reveals a wall the byte
model cannot see.**

| wave (64 rows unless noted) | lanes | ship ms | m2 ms | speedup | bytes m2/ship | **ship GB/s** | **m2 GB/s** |
|---|---|---|---|---|---|---|---|
| `wave64-at-1024` | 1021–1084 | 0.5991 | 0.6511 | 0.920 | 0.512 | **230.3** | 108.5 |
| `wave64-at-2048` | 2045–2108 | 1.2753 | 1.3546 | 0.941 | 0.515 | **213.4** | 103.5 |
| `wave64-at-4096` | 4033–4096 | 2.4163 | 2.5402 | 0.951 | 0.500 | **220.5** | 104.9 |
| `wave16-at-4096` | 4081–4096 | 0.7166 | 0.7667 | 0.935 | 0.500 | **187.0** | 87.4 |

Read the last three columns together. The shipped form runs **at the memory plateau** (187–230 GB/s
against a 237 GB/s ceiling), so it is traffic-bound, exactly as the byte model said. M = 2 really does
halve the traffic — the two rows' lane sets are *nested prefixes*, so their union is just the wider one,
and the census halving is real, not an accounting trick. **And the time did not move.** So the m2 form
runs at ~105 GB/s not because the fabric is slow but because, with the bytes halved, what is left is the
*arithmetic and latency* of the same total work — the pair still computes both rows' scores over their
own prefixes.

Therefore: at 4096 lanes the traffic wall and the arithmetic wall are **roughly the same height** (ship's
traffic wall ≈ 2.42 ms; m2's residual ≈ 2.54 ms). A route that cuts only bytes hits the arithmetic wall;
D16's conclusion that "the byte wall is back in front" was right about which wall is *at* the surface and
wrong that cutting it helps on its own. This is why single-lever routes in this index keep disappointing:
the cheapest pair is what matters — **M-tiling (built, bit-identical) plus an arithmetic-class change to
the score loop.** If both land, `attn` at 4096 goes toward ~1.2–1.3 ms per wave instead of 2.42; if only
one lands, it goes the wrong way, which is now measured four times.

Note for the arithmetic side: fp16 does **not** halve bytes — it is also 16-bit — so WMMA cuts only the
arithmetic wall. Only 8-bit staging (int8, with scales, still a stop-and-ask) cuts the traffic wall by
half on its own.

**M = 2 status: parked, not failed.** Bit-identity is green on device at every width including the
1032-lane pinned HF row and 64-row waves; the traffic halving is real; the form is ~2× off the plateau
for reasons that are now attributable rather than mysterious. It reopens the moment an arithmetic-class
score loop exists to pair with it.

### The two defects that blocked the above, and what they cost

1. **`attn_m2_pair_union` returns a sum over lane blocks while `ATTN_M2_STAGE` bounds the widest single
   block** — a unit mismatch, not the row-index/lane-count confusion I diagnosed from the outside. The
   fix is an `M2Union {total, max_block, max_slice, over_read}` struct so the two aggregates cannot be
   conflated, plus `attn_m2_geometry_guard`: 13,744 geometries walked two ways (Python mirror and the C++
   launcher, identical numbers) asserting bound / no-phantom / no-leak / real-plans. Answer to the
   question I asked: **the stage bound is not exceeded at any real width** — worst legal single-block
   union 518 of 520, 0 phantom refusals, all 16 real plan geometries accepted.
2. **`NEGVERDICT` appends a `forms disagree` problem unconditionally**, including for injections that
   exist only in the m2 form (`maskwide`, `partrow`, `partalias`), where the shipped form is *required*
   not to move and disagreement is the definition of success. Those three can therefore never print PASS
   — the `mixed_bw` failure mode again: a check that fails for every configuration catches nothing. The
   same run also proved the earlier negative suite was doubly vacuous for a reason neither of us had seen:
   "moved" was measured form-vs-form, which is structurally blind for the form-neutral injections
   (`readpast`, `dropcurrent`, `dimhalf`, `gqaoff`) — both forms do the same wrong thing, so they could
   never register at *any* geometry. Negative mode now compares each form against **its own** clean
   output per `(row, head)`.

## D20 — the score dot is priced out: 14 % of the instruction stream, and both dots together cap any arithmetic-class change at 27.9 %

`prototype/attn_score_dot2/` — priced bottom-up from the **disassembly**, and stopped before writing a
kernel, which is what the brief asked for and what the branch-2 rule makes cheap.

1. **The prize is 3.9–10.5 % of the m2 form's 2.5402 ms residual (0.100–0.236 ms) at 4096 lanes**, against
   the ~15 % kill line I set. Not marginal-by-a-hair: structurally bounded.
2. **Why: the QK dot is 16 of the 114.6 VALU warp-instructions one (row, lane, head) touch costs — 14 %.**
   Eight one-instruction bf16→f32 conversions plus eight FMAs (kernel-relative lines 693–715), and
   `dot2_shape.cpp` proves a dot2 leg would be 4 × `v_dot2_f32_bf16` with zero conversions. **The PV dot
   cannot be a dot2 at all** — it has one accumulator, so there is nothing for the second lane of the
   packed instruction to multiply.
3. **Both dots together are 27.9 % of the stream, and that is the ceiling on any arithmetic-class change
   to this kernel — WMMA included.** So D19's hopeful "m2 + arithmetic-class ⇒ 2.42 → ~1.25 ms" is
   unreachable by instruction *class*. The arithmetic wall is real (D19 proved that) and it is made of
   something other than matrix-shaped work.
4. **What it is made of, and therefore what to fund: the causal-mask guards cost 20 VALU instructions per
   touch — more than the dot they protect — and the bf16→f32 conversions 8 more.** Together 24 % of the
   stream, in control flow and format handling, neither of which needs a numerics decision to touch:
   applying the causal bound as a loop limit instead of a per-element guard is bit-identity preserving by
   construction. That is the next route, and it must be counted in the disassembly before and after, not
   modelled.
5. **A retired number of mine:** the "score cost 0.246 ms at 1024 lanes" that I had written into the specs
   discussion of the fast tier is not in the tree as a score-only figure — it reproduces as the *whole*
   touch bill at the 2-instruction roof (0.267 ms), so it overstated a score-only prize by ~7×. The
   nearest real ancestor (190 of ~1,030 warp-inst/touch, 2026-10-14) already said the score loop was
   under a fifth of the touch, which is the same story the disassembly tells now.
6. **The decisive argument was not size.** dot2 would spend M = 2's bit-identity admission — the one thing
   that makes this route need no user decision — to buy ≤ 10.5 %. Trading a free gate for a small number
   is a bad trade even when the number is real. The gate plan is written anyway (§6) in case a future
   width changes the arithmetic: association is all a dot2 changes, and the F1 floor may make the band
   question free.

**Also in this entry:** the `NEGVERDICT` scoping defect is fixed and witnessed without a device —
`prototype/attn_m2/stub_harness.py` stands in for the binary and `run.py selftest` runs 12/12, including
the case the device proved impossible before (`must=24 moved_m2=24 moved_ship=0 forms_agree=no
verdict=PASS`). It also caught a latent `NameError` that killed `--neg slotk` on device, and it added the
missing control group: a verdict now FAILs if the *clean* run is not bit-identical, because without a
clean control no injection verdict means anything.

## D21 — D19 is falsified by a bit-identical change: `attn` was never bandwidth-bound, and the byte wall was a units coincidence

`prototype/attn_valu_slim/results.md` §7. A source-only change — unroll the score dot whose trip count is
runtime, hand-fold the warp reduce — cuts a touch from 116 issue slots to 45 and moves the **shipped**
attention form from 2.3446 ms to 1.5416 ms at 4096 lanes (**1.521×**, and 1.52–1.57× at every width
measured), with `ndiff = 0` on all 12 plans, run-to-run-identical logs, and 36 injection runs showing no
disagreement. No numerics decision, no occupancy cost.

The falsifying arithmetic is one division. These levers remove **no traffic**, so the bytes-demanded
census is unchanged; at 1.5416 ms the same 532.7 MB becomes **345 GB/s demanded**, and at 1024 lanes
**381 GB/s**. Both are above this part's measured 237 GB/s DRAM read ceiling, which a kernel cannot
exceed. Therefore the census never described DRAM traffic — L1/L2 absorb most of the reuse — and
`bytes-demanded ÷ time` sitting at 225–248 GB/s was coincidence, not measurement.

That retires the load-bearing claim of D8/D9/D16/D19:

- **D16 item 3** ("the byte wall is back in front, so traffic reduction outranks instruction surgery") —
  wrong: instruction surgery outranked everything and was worth 1.52×.
- **D19** ("two walls of equal height, so no single lever moves attn") — wrong in its premise. m2 halved
  traffic and gained nothing because the traffic was not the wall; m2's residual was instruction and
  latency bound, as was the shipped form. The "equal height" was two measurements of the same wall from
  two directions.
- **The byte model's 5 % fit across three widths** in `prefill-occupancy-results-2026-10-16.md` stands as
  arithmetic — it fits — and is still not evidence, because the quantity it fits is demanded bytes over
  time, which is a restatement of the time, not an independent constraint. That is the trap in general
  form: *a model that reproduces the measurement it was fitted to cannot falsify anything.*

What changes in the plan:

1. **The levers port to `engine/kernels_attn.cpp` now**, ahead of every route in the index: attn is
   412.07 ms of a 1307.75 ms wave at 4096 lanes, and 1.52× there is roughly 140 ms of the wave. Gate
   path: `prototype/attn_decode`'s existing pinned gates first, then the engine, then a full-wave
   re-measure for the ledger.
2. **Traffic routes go back in the queue behind instruction routes**, with M = 2 re-tested *after* the
   levers land — its halving is real and may pay against a lower instruction wall.
3. **`tools/` needs a cheap guard for this class of error**: any claim of the form "family X is
   bandwidth-bound" must show demanded-vs-achievable, not demanded-vs-elapsed, or must show the
   instruction-cut falsification directly. Two of the last three plan reversals in this log came from
   reading a rate as a wall.

## D22 — D21's lever is on the shipping path: attn 403.38 → 275.19 ms, the wave 1243.07 → 1105.20 ms

`engine/attn_slim_kernels.h` + `--attn-touch`, gates green at both values, witness in
`bench/performance-status.md` (appended section) and `engine/HANDOFF-attn-touch.md`. Numbers taken paired
inside one lock hold with both spellings in **one binary** — the runtime knob was the right call over the
build-time define the worker proposed, because a define would have made this a two-window comparison, and
two windows is how the last three plan reversals in this log were produced.

What happened to the plan, in one line each:

1. **D21 was right and it paid**: −128.2 ms of the `attn` family at 4096 lanes (**1.466×**), −137.9 ms of
   the whole wave (**1.125×**), 3295 → **3706 prefill tok/s** at 4096 tokens.
2. **`gdn_rec` did not move** (501.82/507.01 vs 509.27/510.53 ms) — the control behaved as a control
   should, which is worth recording because it is the first time in this sequence that a lever's
   non-effect was also gated rather than assumed.
3. **Decode is unchanged, and that is a fact about the route, not a disappointment**: single-request decode
   runs the one-row attention path; the levers are in the wave body. Best-of-3 ratio 1.029 with 4.3 %
   spread. The ledger's 71.3 tok/s stands.
4. **The traffic routes are now unblocked in the right order.** D19's M = 2 halving was real and bought
   nothing because it was measured against an instruction wall that was 2.6× taller than it needed to be.
   With the touch at 45 slots instead of 116, re-testing M = 2 against `--attn-touch 1` is the obvious
   next measurement, and it is cheap: both forms already exist, and `prototype/attn_m2`'s gates are green.
5. **What the day's reversals have in common**: every one of them came from reading a *rate* as a *wall* —
   41 wavefronts/CU, 237 GB/s demanded, 4.696 TFLOP/s. The guard that actually works is not a smarter
   model but a cheaper experiment: change one thing and see which side moves.

## D23 — `gdn_rec` got its instructions counted, and the answer is a funded **no**

`prototype/gdn_rec_isa/` (census + four MODE kernels + two control rows, no device run needed for the
verdict). The method that paid off on attention was pointed at the biggest family in the wave — ~509 ms
of 1105 ms at 4096 lanes, 52 % at 1024 — which had been argued about with rates for weeks and never
counted.

What the count says: **1,523 issue slots per wavefront (97,472 per launch)**, prologue 323 / loop A 384 /
loop B 816; the cross-lane class is a **226-slot spill (14.8 %, of which 127 are exec-guards)**; 11
barriers; one over-full `vmcnt(0)` store drain. What is *absent*, and that is half the finding: no runtime
trip counts (both hot loops are already unrolled), **zero `v_movrels`**, no bf16↔f32 conversions, and no
per-step SALU to hoist — there is no token loop to hoist out of, because one launch is one token.

The arithmetic that decides it: the measured per-launch time is 6.8–6.9 µs against a 0.69 µs issue floor,
i.e. **the family runs at ~10 % of its own issue roof**. That inverts the attention case — `attn` was near
its roof, which is why cutting 61 % of its slots bought 31 % of its time. Here the same surgery is capped
between ~1 % (slot view) and ~13 % (best-case critical-path view) of a 509 ms family. **Not funded.**
This is the D22 guard applied on the way *in* rather than after a reversal: check whether the rate is even
achievable before funding the cut.

Both controls passed before any delta was read — the MODE-0 copy censuses identically to the shipped
`rec_kernel` (1523 vs 1523 slots/token-step) and the engine TU matches the prototype TU — so the null is a
measurement and not a census artifact. The written levers (MODE 1 −2.3 %, MODE 2 −8.7 %, MODE 2c −2.0 % of
slots, all bit-identity-clean) stay in the tree as a **falsifier of this model**, not as a win: MODE 0 vs
MODE 2 in one burst returning > 1.10× refutes the latency reading, < 1.02× kills the instruction route for
this family. Given the harness does not exist yet and the family's own numbers cap the prize, the queue
takes the attention re-test first — `M = 2`'s traffic halving against a touch bill that is now 45 slots
instead of 116, where the harnesses already exist and the family is still 275 ms.

## D24 — the pair measured: the levers transfer to M = 2 (1.47–1.49×) and the traffic halving still buys nothing

`prototype/attn_m2/results.md` §6. Four forms in one process, alternating inside each rep. Admission green
first: 12 plans × 3 pairs, `ndiff = 0` everywhere, with the levers *called* from
`attn_valu_slim_kernels.h` rather than copied, so there is no second spelling to drift and the selftest
enforces that.

**The levers generalize** — 1.469–1.494× on the m2 body, the same as 1.496–1.543× on the tile body. That
is a second, independent confirmation of D21: for this kernel the wall is the instruction bill under
either row-tiling.

**The traffic halving does not pay, now twice measured** — m2-slim is 5.5–13.4 % slower than ship-slim at
every width with the bytes-demanded ratio still ~0.50. Pre-port it lost to a tall instruction wall; post-
port, with the bill at 45 slots, it loses by about the same margin. **M = 2 stays parked**, and the
traffic-reduction family loses its last funded member for attention at these widths.

Honest bookkeeping on the prediction: the worker wrote "not to pay, 1.15–1.29× slower". Right direction,
~2× pessimistic magnitude (measured 1.055–1.134), and the headline width landed at 1.055 inside the
ambiguous band of its own falsifier. Recorded rather than rounded: if anyone reuses that elasticity model,
it is biased pessimistic; and if `kAttnMaxSplit` is ever raised, the only untested regime left for this
route is traffic that genuinely reaches DRAM above 4096 lanes.

## D25 — `gdn_rec`'s falsifier ran: 1.016× on the one valid lever, and the two hand-spelled levers are broken rather than slow

`prototype/gdn_rec_isa/results.md` §7. D23 left a falsifier nobody could run; the harness now exists and
the answer is in the kill band: `rec0` 0.8088 ms vs `rec2c` 0.7962 ms = **1.0159×** (refute needed > 1.10×,
kill threshold < 1.02×), burst 40 × 5 reps in one window, per-launch 6.319 µs against the census's modelled
6.8–6.9 µs.

Two things worth more than the verdict:

1. **MODEs 1 and 2 do not compute the right answer, and the gate is what caught it.** Every element of `o`
   differs (262,144/262,144), `maxrel` vs the CPU reference reaches 1.2e+06 against MODE 0's 1.9e-01 floor,
   and each disagrees **with itself** across two runs (1,408 / 80,512 / 14,080 elements) — the repo's own
   signature for a racy hand-written cross-lane permute, not for a re-associated reduction. `rec2c`, the
   same fused-pair idea lowered by the compiler rather than by hand, is bit-identical, so the technique is
   fine (`attn_valu_slim`'s hand-fold ships) and these spellings are wrong. Admission was bit-identity and
   the binary refuses to time when admission fails, so no number was ever taken from a broken MODE.
2. **D23's premise was wrong even though its verdict survives.** Measured elasticity from −2.0 % slots to
   −1.6 % time is **≈ 0.8**: slot cuts *do* translate in this family, which "10 % of the issue roof ⇒ worth
   ~1 %" did not predict. The verdict survives because the arithmetic closes from the other side: the
   biggest lever available is MODE 2 at −8.7 % slots ≈ 7 % of the family ≈ 36 ms of a 1105 ms wave, and
   that MODE is the broken one. **Repairing hand spellings to collect a predicted ~3 % of the wave is not
   funded.** Recorded as a correction, not deleted, because a roof argument and a measured elasticity
   disagreeing is exactly the kind of thing that produced this session's three reversals.

## D26 — the GEMM gap is the engine's own call, not the library's kernels: 1.17× available at the shipping width, no hand WMMA needed

`prototype/gemm_algo_sweep/results.md` §3a, §4. The cheap question before building a hand WMMA GEMM — is
hipBLASLt's default heuristic leaving the 1.4–1.6× on the table? — answered something better than expected.

**Measured, all at the shipping width M = 512** (`engine/decode_step.h:49` `kMaxWaveRows = 512`, so a wave is
at most 512 rows and a 4096-lane prefill is eight such waves; the engine never issues M = 1024 or 4096):

| path | ms / 24-layer wave | vs engine | saved off the 1105.20 ms wave |
|---|---|---|---|
| engine today (`hipblasGemmEx`) | 18.67 | — | — |
| `hipblasLtMatmul`, heuristic rank-0 | 17.42 | 1.072× | 10.0 ms (0.9 %) |
| `hipblasLtMatmul`, best of 12 enumerated | 15.90 | **1.174×** | **22.2 ms (2.0 %)** |

The decisive rows are the ones where `ratio = 1.000`: on `q_proj`, `k_proj+v_proj` and `in_proj_z` the
heuristic's rank-0 **is** the fastest enumerated kernel, yet the engine's own call is 1.220×, 1.625× and
1.359× slower than it. hipBLAS and hipBLASLt choose differently, and that difference is larger than anything
the enumeration offers. A third of the prize is a call-path change; two thirds needs a per-shape algo table.
**No hand WMMA GEMM should be started on the strength of the 1.4–1.6× ceiling while this 1.17× sits unbuilt.**

Three things recorded alongside, because each is a trap of its own class:

1. **The band was wrong, not the library** — and the correction came from evidence, not preference. Every
   candidate initially failed `cos ≥ 0.999999` by 1.8e-06 with `worst_ulp` exactly at the bound. The
   mechanism turned out stronger than "reference noise": reordering the double reference moves *nothing*
   (`ndiff=0`), F32 accumulation moves nothing (a dot of bf16 products is bit-identical to the exact sum
   here), and the whole deficit is **bf16 output quantisation**, `1 − meanrel²/2`. Floors are now measured
   per `(shape, M, check_rows)` by `--floor-probe` with provenance printed, and there is no built-in
   default: a missing CAL row exits 2 before any HIP call.
2. **A must-catch set must be derived from what the object under test can break.** The one-term-short
   1024-chain injection is *below the resolution of any cos floor at this K* — its worst-seed deficit is
   0.73×–5.1× the quantisation deficit, i.e. on two shapes the injected bug moved cosine **less than a
   correct answer does**. It was replaced by nine algo-level injections (wrong `ld`, swapped operand roles,
   two split-K partial losses, aliased partial slot, half-written tile, `beta` left at 1 against stale and
   against NaN `C`, doubled `alpha`), which sit 3–5 orders of magnitude below the floor. `one_term_short`
   is still reported, by `RESOLUTION`, with the reason written down.
3. **A device leg at 2 ULP was a cancellation, and F1 predicted it before I looked.** `in_proj_qkv` crossed
   at one element in 49,152 with `|ref|/max|ref| = 1.96e-07`; `gate_proj+up_proj` at one in 28,672 with
   2.34e-07, on the engine leg as well as the algo leg. The bound stayed at 1 ULP and the exemption became
   cancellation-aware and symmetric. Had it been a fat cluster at full magnitude it would have been a
   library defect on the shipping path; it was not, and the check distinguished the two.

**Open, named:** the ledger's `GROUP matmul_ms` = 70.30 ms at a 1024-lane wave versus this mix's predicted
37.3 ms — so the ledger's `GROUP` carries library time beyond these 150 calls. Until reconciled, the algo
route is worth **at least 0.9–2.0 % of a 4096-lane wave and possibly near twice that**, and the engine-side
A/B is what sizes it.

## D27 — `--gemm lt` measured on the engine: whole-wave 1.120× at 1024 lanes, and my mix projection was 4× low

`prototype/gemm_algo_sweep/results.md` §3c. Both routes in one lock hold, alternating per rep, medians of 3:

| width | `GROUP matmul` | whole wave | prefill tok/s |
|---|---|---|---|
| 1024 lanes | 71.28 → **49.58 ms (1.437×)** | 240.96 → **215.14 ms (1.120×)** | 4250 → **4760 (+12.0 %)** |
| 3072 lanes | 211.72 → **149.98 ms (1.412×)** | 804.85 → **743.35 ms (1.083×)** | 3817 → **4133 (+8.3 %)** |

This is the largest prefill win since the engine exists — larger than the attention slim (1.125× of the wave)
— and it cost no new kernel: `hipblasLtMatmul` with the heuristic rank-0 cached per `(shape, M)`, default
still `library`, `legacy` accepted as a synonym so a shell loop can spell both sides of the pair.

Two lessons, both mine:

1. **A per-call burst table cannot project a family-level wave number; mine was 4× low.** §3a priced 1.174×
   from 150 calls timed at M = 512 in a tight loop; the engine's `proj` family is 228 launches including the
   hand `in_proj_a/b` GEMVs and epilogues that §3 sweeps into these groups, and the measured family ratios
   (proj 1.521×, gateup 1.597×) belong to no single shape in the table. The error was in the flattering
   direction, which is why it survived review.
2. **`mlp_down` regresses (0.78–0.81×) and the sweep had already told me so.** For `down_proj` at M=512, lt
   rank-0 is 0.1439 ms against the engine's 0.1313 ms; the 1.231× I called a win belongs to **best-of-12**,
   not the default. Reading a "best" column as though it were the default column is the mistake, and the fix
   is `algo_pins()` — pin `down_proj`'s `best_seq=1` (0.1169 ms) rather than take the heuristic. **Rule:
   a sweep column labelled "best" must be read against the "default" column before anything is called a win.**

**Default stays `library` until one band question is answered.** Gate section (h) (batched vs one-row, which
under `lt` is hand-GEMV vs hipBLASLt) reads worst per-row `1-cos 2.204e-03` on `dec64f` against a 2e-03 band
that was measured for the **hipBLAS-vs-hand** pairing — a different independent variable — while every
pin-facing section passes and lt's pooled numbers are better than library's (meanrel 1.429e-02 vs
1.433e-02). The remedy is to re-derive that band from a measured spread for the new pairing and re-run the
must-catch witnesses, not to edit the constant. A red gate does not get flipped by a big number next to it.
