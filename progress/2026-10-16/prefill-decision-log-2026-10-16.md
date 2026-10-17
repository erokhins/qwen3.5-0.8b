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

## D28 — pins adjudicated in situ: keep only `down_proj`; final `--gemm lt` is +11.0 % prefill at 1024 and +8.4 % at 3072, and one red row holds the default

`prototype/gemm_algo_sweep/results.md` §3d. Added `--gemm-pins all|none|<csv>` as a runtime knob (main-agent
edit, because a pin subset is only adjudicable in situ and a rebuild per subset would put each side in a
different window). Four configs in one hold: rank-0 218.85 ms, down_proj-only **215.32 ms**, all-three-pins
218.69 ms against library 239.43 ms.

**The two extra pins cost 5.5 ms of a 24.8 ms `proj` family** (25.22 → 30.35 ms) while having measured
4–6 % better in a warm burst. Same lesson as D27 one layer down, now in the painful direction: a warm-burst
per-call ranking does not transfer to in-situ family time. Pin table is one entry; `algo_pins()` without an
in-situ A/B would have shipped a regression dressed as an optimization.

Final: 1024 lanes GROUP 1.503×, whole wave 238.02 → 214.40 ms (**+11.0 %**, 4302 → 4776 tok/s); 3072 lanes
GROUP 1.446×, wave 794.96 → 733.24 ms (**+8.4 %**, 3864 → 4190 tok/s); parity 1/1 everywhere.

**Gate state: 9 fixtures × both routes — all green except `wide520f --gemm lt`.** Not a band overrun
(per-row 1-cos 1.077e-03 vs a 2.75e-03 band) but a token flip above the coin-toss rule: batched-vs-carry
`1 RED: prefill row 180: 13 vs 248046 with room 0.124807 (own gaps 0.182308 / 0.124807)` where room < 0.10 is
a coin toss and this row sits just above it; `library` does not flip it. **Default stays `library`** — a
borderline row that only the candidate route flips is exactly the thing not to ship past on the strength of
the headline next to it.

Provenance error of mine, recorded: the 2.75e-03 band was derived by the worker **on the three-pin algo set**
(the band text names `algo_pins()`), and I then trimmed the table to one. It still passes on the final set,
but the band must be re-derived on the set the binary actually ships before it is cited again.

## D29 — `--gemm lt` becomes the shipped default: 1.049× on the 4096-lane headline, and the flip-room line retires an absolute threshold for a unit

Headline pair, one window, `--time-waves --steps 0`, `ids4096`, medians of 3: **1122.50 → 1069.89 ms,
3649 → 3828 prefill tok/s (1.049×)**, `GROUP matmul` 276.90 → 195.11 ms (1.419×). Whole-wave ratios elsewhere:
**1.110× at 1024 lanes** (4302→4776 tok/s) and **1.084× at 3072** (3864→4190). The `lt` side is the noisier of
the two at 4096 (±2.6 % against ±0.1 %), so the ratio is quoted from medians. Decode unchanged at 71.3 tok/s
**by construction** — a one-row body is the hand GEMV on either route — and was deliberately not re-taken; if
M=1 ever goes to the library, that claim dies and decode must be re-measured.

Gate: 18/18 green on both fixtures×routes before the move (`wide520f --gemm lt` included, row 180 admitted at
0.9985 ULP), four fixtures re-taken at the new default green with `gemm=lt` echoed. `bench/performance-status.md`
§7 is the ledger row.

The thing worth keeping from the last mile is that **the blocker was a unit error in the gate, not a numerics
problem in the route.** `flip_room_max = 0.10` in logits is 0.80 ULP at logit 20 and 6.4 ULP at logit 1: it
excused thin ties at small magnitudes and condemned a one-step tie at a large one. Retired for
`flip_room_max_ulp = 1.5` (census: worst flip in 11,804 comparisons is 0.9985 ULP), and the attempt to serve
both pairings with it exposed a second confusion — a pin-vs-engine comparison must read the **reference's**
room, because the engine's own near-tie is not evidence about whether the reference decided; `--break-runnerup`
places its candidate exactly one ULP above the pin's max, so a symmetric min-over-sides reading admitted a
licensed negative control at any line ≥ 1.0 while the row needing admission sat at 0.9985. A 0.0015 ULP
squeeze, which is what a forced-to-be-shared threshold looks like from the inside.

Three rules out of it, all in the code and the ledger rather than only here: **a threshold must be printed
with the margin that justified it** (`flip_room_line(pairing)` prints value, observed max, ratio and which
pairing owns it); **an unmeasured threshold must say it is unmeasured** — the pin-vs-engine line has zero real
admissions in 18 runs and states that instead of borrowing the cross-width margin; and **an injection that
does nothing must refuse rather than pass** (`--break-ab-*` without `--dry-run` now exits 1; nine earlier
device runs had been silently vacuous).


---

# D30 — the explicit prefill inventory: every lever considered, with its disposition (2026-10-16)

Written because the session kept discovering routes in the order it stumbled into them, which is not the same
as having processed them. This is the closed list, and the method used to make it closed: **for each family of
the wave, ask nine questions** — is it bytes-bound, instruction-bound, latency-bound, launch/dispatch-bound,
fusible with a neighbour, improvable in precision-choice (16/32-bit), improvable in geometry (tile/width/split),
improvable in library selection, or blocked behind a numerics admission? Every answer below carries its
disposition and, for the ones not yet measured, the cheapest falsifier that would decide it.

**Basis, measured after the `--gemm lt` flip** (`--time-waves --steps 0`, `ids4096`, one rep of the median
window, `wall_ms=1111.95`, `plan_match=1 launchstats_match=1`):

| family | ms | % of wave |
|---|---|---|
| `gdn_rec` | 534.54 | **48.07** |
| `attn` | 285.80 | **25.70** |
| `proj` | 105.37 | 9.48 |
| `mlp_gateup` | 66.74 | 6.00 |
| `cast` | 29.59 | 2.66 |
| `mlp_down` | 27.15 | 2.44 |
| `gdn_conv` | 20.71 | 1.86 |
| `glue` | 14.48 | 1.30 |
| `copy` | 12.35 | 1.11 |
| `gdn_gated_norm` | 9.88 | 0.89 |
| `lm_head` | 2.52 | 0.23 |
| `gdn_gates` | 1.85 | 0.17 |
| `host_tail` | 0.87 | 0.08 |

So two families are 73.8 % of the wave, and anything that does not touch `gdn_rec`, `attn`, the matmul group
(17.9 %), or the small-kernel tail (8.0 %) cannot move the headline at all.

## 1. `gdn_rec` — 534.54 ms, 48.07 %

| # | lever | disposition | evidence / first falsifier |
|---|---|---|---|
| 1.1 | instruction surgery (unroll, hand-folded reductions, spill removal) | **CLOSED, funded no** | D23 census: 1523 slots/token-step at ~10 % of its own issue roof; D25 falsifier `rec0`/`rec2c` = **1.0159×**, elasticity ≈ 0.8, largest lever (−8.7 % slots) worth ~7 % of the family and that MODE is broken. Refuting needed > 1.10× |
| 1.2 | chunked scan (P9, §04.7) | **CLOSED for now, measured loss** | priced ≥ 1.5×, measured 5.9× worse. It chunks intra-chunk work, not the dependency. Only re-opens if 1.4 or 1.5 lands first |
| 1.3 | **latency of the per-token dependency chain** (the reading D25 leaves standing) | **UNTESTED — biggest open item on the box** | ~1.6 *working* wavefronts per CU (one warp per block drives the recurrence, three park at the norm barrier). Falsifier: a harness variant that breaks the cross-token dependency (each token reads an independent starting state — numerically meaningless, timing-only) and compares against MODE 0. If time collapses, latency binds and 1.4/1.5 are worth funding; if it does not, the family is bound by something the census has not named and the whole family needs a different model |
| 1.4 | raise working wavefronts per CU (more warps cooperating per token, or split the state across blocks and combine) | **UNTESTED, blocked behind 1.3** | only funded if 1.3 says latency binds. Cost is real: cross-block combine adds a barrier and a partial buffer |
| 1.5 | shorten the chain itself (re-associate the recurrence so two dependent steps overlap) | **UNTESTED, numerics-gated** | any re-association must pass the existing bands; the state is F32 with bf16 casts and A-18 says depth decays token match anyway, so the band question is answerable with the existing floors |
| 1.6 | **state dtype / traffic**: is the GDN state F32 in memory? If so, a 16-bit state halves the largest per-token traffic figure in the engine | **UNTESTED, cheap census first** | per-request state is 18.0 MiB + 0.63 MiB conv (`§00.5`), i.e. the recurrence moves a large buffer per token. Census: dtype as laid out in `engine/` + bytes moved per launch ÷ measured time. Then D13/D14 authority applies unchanged (16/32-bit is adoptable if gates stay green; fp16 is *more* precise than bf16 in mantissa and the range census peaked at 82.5) |
| 1.7 | fuse `gdn_conv` + `gdn_gates` + `gdn_gated_norm` (20.71 + 1.85 + 9.88 = 32.4 ms) into the rec launch, or into each other | **UNTESTED** | 32 ms of separate kernels plus their state round-trips; the ceiling is small (~3 %) but the traffic saved may exceed the kernel time, which is exactly the mistake D21 warns about — census the round-trips before funding |
| 1.8 | better algo/library route | **N/A** | hand kernel, no library call |

## 2. `attn` — 285.80 ms, 25.70 %

| # | lever | disposition | evidence / first falsifier |
|---|---|---|---|
| 2.1 | instruction surgery on the tile touch | **SHIPPED** | D21/D22: 116 → 45 slots per touch, family 1.466×, whole wave 1.125× |
| 2.2 | M=2 row tiling (traffic halving) | **PARKED, measured twice** | D18/D24: bit-identical, bytes halved, **5.5–13.4 % slower** both before and after the slim levers. Re-opens only above 4096 lanes, which needs 2.5 |
| 2.3 | further scalar instruction cuts on the 45-slot touch | **CLOSED by the same census** | 45 slots is the post-slim number; the remaining content is 8 WMMAs-worth of arithmetic plus the softmax; D20 priced arithmetic-class changes at ≤ 10.5 %, below the 15 % kill line |
| 2.4 | **`v_wmma_f32_16x16x16_w32` for the score and PV dots** | **UNTESTED, it is *not* 2.2 — and its stated premise was false** | the matrix-unit form replaces the score dot's whole 16-slot loop with one instruction per 16×16×16 tile across 16 rows at once — an instruction lever wearing a traffic costume, which is the direction D21 said wins. **Correction made while preparing it: the row required fp16 operands, and it does not.** `v_wmma_f32_16x16x16_bf16` is present on gfx1151 (`tools/wmma_operand_types.sh`, `devices/strix-halo-official-specs.md` §7.1), so this route needs no format change and no D13/D14 band decision — the operands stay bf16 exactly as the engine holds them, and every plan that costed the conversion was pricing a premise nobody had checked. Ceiling is the remaining arithmetic share, ~28 % of the touch, and `attn` is **26.94 %** of the wave (257.56 ms of 956.12 ms) since the `gdn_rec` flip, which is what ranks this second |
| 2.5 | raise `kAttnMaxSplit` / lane ceiling above 4096 | **BLOCKED, needs numerics admission** | D4/D5. It is also the only route that could re-open 2.2 (traffic that actually reaches DRAM). Unblocking needs an admitted band for > MAXSC straddles, not a constant |
| 2.6 | occupancy / wave shape of the tiled kernel | **UNTESTED cheaply** | the tile form's realised occupancy is not recorded; one `--time-waves` run with a couple of block-size variants costs minutes and would say whether 2.4 has a scheduling headroom to spend |

## 3. matmul group — `proj` + `mlp_gateup` + `mlp_down` = 199.26 ms, 17.92 %

| # | lever | disposition | evidence / first falsifier |
|---|---|---|---|
| 3.1 | library algorithm selection (`hipblasLtMatmul`, per-`(shape,M)` algo, one pin) | **SHIPPED** | D26–D29: group 1.419× at 4096 lanes, whole wave 1.049× |
| 3.2 | **raise the 128 MiB workspace budget** | **UNTESTED — cheapest open matmul item** | **15,077 of 15,750 enumerated algos need more than 128 MiB** (`algo_max_workspace_fits`). Memory is not a constraint for this model (whole checkpoint is 4 % of the pool), so a 512 MiB budget may admit strictly better algos. `gemm_algo_sweep --workspace-mib 512` answers it in one window; the engine change is one constant |
| 3.3 | hand WMMA GEMM to beat the library | **DEPRIORITISED, not closed** | after 3.1 the library runs at 33–38 TFLOP/s against a 52.63 measured WMMA roof → ~1.2–1.4× left, and it must be earned with a real kernel. Re-rank only if 3.2 says the library is out of algos to give |
| 3.4 | move the 72 hand `in_proj_a`/`in_proj_b` GEMVs into the library route | **UNTESTED** | they are `proj_gemm`-refused today and sit inside the 105.37 ms `proj` family; first measure is the burst `M=512` rate of those two shapes against what the hand GEMV achieves in situ (the lesson of §3d applies: adjudicate in situ) |
| 3.5 | pin more shapes, chosen **in situ** | **OPEN, small** | only `down_proj` is pinned and `mlp_down` still costs 0.945×; the burst-priced pins lost in situ, so the candidates must be ranked in situ before pinning. Worth ~1 ms |
| 3.6 | `silu·mul` epilogue fused into the gate/up GEMM | **UNTESTED, small** | ≥ 2.2 ms of `mlp_gateup` is the separate epilogue (Job-1 reconciliation); hipBLASLt has no two-matrix `silu(a)·b` epilogue, so this is a custom kernel — 6.00 % family, ~2 % ceiling |

## 4. the small-kernel tail — `cast` 29.59 + `copy` 12.35 + `glue` 14.48 + `gdn_gated_norm` 9.88 + `gdn_conv` (in 1.7) ≈ 66 ms, 6.0 %

| # | lever | disposition | evidence / first falsifier |
|---|---|---|---|
| 4.1 | fuse `cast` work into the producing kernels | **UNTESTED** | 29.59 ms of pure conversion. Census first: which casts, whose output, and whether the producer can write the target dtype without an extra pass. Ceiling ~2.7 % if all of it vanished |
| 4.2 | eliminate `copy` (12.35 ms) by in-place staging where legal | **UNTESTED** | needs the same census discipline: what is copied, why, and what aliasing rule forbids in-place |
| 4.3 | reduce `glue` launch count (49 launches/body) | **CLOSED for value** | dispatch-bound and bimodal on this box (`bench/performance-status.md` §6); 14.48 ms of 1112 with a ceiling well under the noise floor of every measurement in this log |
| 4.4 | overlap independent bodies on separate streams | **CLOSED, measured small** | `WAVEGAP exposed_ms=7.14, ratio=1.030` — there is almost nothing exposed to overlap |

## 5. out of my authority, listed so it is not mistaken for an oversight

| # | lever | status |
|---|---|---|
| 5.1 | **int8/int4 quantisation** of weights or activations | **STOP-AND-ASK** — error is orders of magnitude above any adopted band; the standing rule requires the user's words (`AGENTS.md`, numeric-format section) |
| 5.2 | MTP drafting as a prefill/decode accelerator | in scope for decode only, measured slower on CPU (spec §09.5), and no engine path exists; not a prefill lever |
| 5.3 | anything for models larger than this checkpoint | tagged **[FUTURE]**, not scheduled (M4 explicitly not scheduled) |

## 6. the queue this implies, in order, with the first measurement for each

1. **3.2 workspace budget** — `gemm_algo_sweep --workspace-mib 512` at M=512, paired against the 128 MiB
   result already in §1. Minutes, no new code. *Main agent.*
2. **1.6 `gdn_rec` state dtype + traffic census** — CPU census of the layout and bytes-per-launch, to say what
   fraction of 534 ms is state movement. *Worker.*
3. **1.3 dependency-break falsifier** — decides whether `gdn_rec` is latency-bound and therefore whether 1.4
   and 1.5 get funded. This is the single highest-value open measurement on the box: the family is 48 % of the
   wave and every existing model of it is an inference rather than a measurement. *Worker builds, main agent
   measures.*
4. **2.4 WMMA attention touch** — priced on CPU against the 45-slot body before any device time, then one
   A/B. *Worker.*
5. **2.6 attn tile occupancy** — one `--time-waves` window with block-size variants; tells 2.4 whether it has
   scheduling headroom. *Main agent.*
6. **1.7 / 4.1 / 4.2 fusion censuses** — census only; no fusion gets funded without a bytes figure first, per
   D21. *Worker.*
7. **3.4 / 3.5** — the two small matmul items, ranked in situ. *Worker builds, main agent measures.*
8. **3.3 hand WMMA GEMM** — deliberately last, and only if 3.2 comes back empty.

Items 1.2, 2.1, 2.2, 2.3, 3.1, 4.3, 4.4 are closed with evidence on the record. 2.5 is blocked on a numerics
admission, and 5.1–5.3 are out of authority or out of scope. **If items 1–7 all come back negative, then the
prefill wave at 4096 lanes has no remaining route above the noise except raising the lane ceiling (2.5) and
quantisation (5.1)** — which is the honest statement of where this ends, and the reason 1.3 matters more than
anything else on the list.

**Queue status, updated as items close.** Item 1 (3.2, workspace budget) **closed negative — see D31**, and its
replacement is **3.7, select beyond the shallow heuristic answer** (≈ 2.2 % ceiling, burst-priced, in-situ
adjudication required), inserted at position 7b below 3.4/3.5. Item 2 (1.6, state dtype/traffic) and item 3
(1.3, the dependency-break falsifier) are **closed by D32**: not traffic-bound, not latency-bound on either
chain, so 1.4 and 1.5 close with them and **1.9, loop spelling at bit-identity, is the new head of the queue**
at a ~6.3 % ceiling — **measured and shipped, see D33**: 1.153× on the family in situ, prefill 3927 → 4284 tok/s.
Item 4 (2.4, the matrix-path attention) is the head of the queue now, re-ranked by D33 because `attn` is the
largest non-recurrence family at 26.94 %, and its fp16-conversion premise was found false while preparing it.

# D31 — D30 item 3.2 comes back **negative**, and the thing it found instead is candidate depth (2026-10-16)

**Question asked:** 15,077 of 15,750 enumerated algos exceed the engine's 128 MiB workspace budget
(`prototype/gemm_algo_sweep/results.md` §4), and memory is not a constraint for this model — does raising the
budget admit better kernels? **Answer: no, and the apparent yes was an artifact.**

Two budgets, one lock hold each, `--rows 512 --check-rows 8 --inner 8`, `util` 63 % then 92 %, best_tflops per
shape:

| shape | 12 candidates |  | 64 candidates |  |
|---|---|---|---|---|
| | 128 MiB | 512 MiB | 128 MiB | 512 MiB |
| `q_proj` | 35.44 | 34.73 | 32.96 | 33.29 |
| `k_proj+v_proj` | 26.64 | 27.11 | 26.93 | 27.14 |
| `o_proj+out_proj` | 28.26 | 28.38 | **33.62** | **34.83** |
| `in_proj_qkv` | 35.61 | 35.94 | 35.37 | 35.02 |
| `in_proj_z` | 32.48 | 30.22 | 30.01 | 31.64 |
| `gate_proj+up_proj` | 32.04 | **34.86** | **35.07** | **35.39** |
| `down_proj` | 34.58 | 31.21 | **35.79** | **36.14** |

At 12 candidates the 512 MiB column looked like a **+8.8 % win on `gate_proj+up_proj`** (0.1173 → 0.1078 ms).
At 64 candidates the same shape reaches 35.07 **at the original 128 MiB**, so the gain was the candidate list
being too short, not the workspace being too small. Same for `down_proj` and `o_proj+out_proj`. Four shapes move
by less than the window noise in either direction. **`kWorkspaceBytes` stays at 128 MiB** — the budget is not the
lever, and the free memory stays unspent.

**The trap, spelled, because it is the same one as D27's "best column vs default column": a `best` read off a
truncated candidate list measures the truncation, not the field.** `--max-algos 12` was the setting the earlier
§1 sweep used, so every ratio in §1 — and the pin table built from it — is a floor on the available headroom and
not a measurement of it.

That gives a new queue item, and it is bigger than the one that closed:

**3.7 (new): select beyond the shallow heuristic answer.** At 128 MiB with 64 candidates, `ratio` (rank-0 vs
best) is **1.387 `down_proj`, 1.407/1.336 `o_proj+out_proj`, 1.207 `gate_proj+up_proj`, 1.133 `in_proj_qkv`**
— winners sitting at ranks 16–55, which is where the engine's `kHeuristicWidth = 8` cannot see them. Worth
noting against the code: a **pin is not bound by that width** (it is validated only as "valid hex, non-zero,
workspace fits", `engine/kernels_gemm.cpp:531-551`), so adopting a deep winner needs no new selection path —
only an algo capture at a deep query and an in-situ adjudication. The prize, priced on the per-wave call counts
`--list` books (24/48/24/18 calls) and the 64-candidate `default_ms` vs `best_ms`: ≈ **3.0 ms per 150-call wave,
≈ 24 ms of a 4096-lane prefill ≈ 2.2 %** — a ceiling, taken at burst rates, and D28's precedent is that
two of three burst winners were in-situ *losers*. Ranked below the `gdn_rec` items for exactly that reason.

Two instrument defects fell out of the same window, both recorded rather than patched around silently:

- **`--calibration-out` replaces `floor_calibration.tsv` instead of merging by key.** A later
  `--floor-probe --rows 1024,4096` therefore deleted the M=512 CAL rows, and this session's first two sweep
  attempts exited 2 with `CALIBRATION-NEEDED … missing=7 of 7` for want of them. Recovered by probing M=512
  into its own file and concatenating (21 rows now, backup in `tmp/wsjob/floor_calibration.before.tsv`); the
  writer should merge, and a sweep that cannot find its keys should say which prior probe deleted them.
- **A stale comment claimed three pins above a table holding one.** The prose was written before the in-situ
  adjudication shrank the table and survived it. Fixed in the same commit as this entry, with the two
  true/false claims about pin admissibility written down where the next reader will trip on them.

**Window noise witness, kept because it sizes every claim above:** `q_proj` best_ms was 0.1212 ms in the
12-candidate window and 0.1303 ms in the 64-candidate window, same budget, same shape, minutes apart — **7.5 %
on a 0.12 ms call**. Any single-shape burst claim below that is not a finding; it is why 3.7 must be adjudicated
in situ and not adopted from this table.

# D32 — **the latency reading is refuted**, and what `gdn_rec` is actually bound by is a bit-identical 1.15× (2026-10-16)

D30 item 1.3 was written as "the single highest-value open measurement on the box", because half the wave was
modelled as latency-bound at ~1.6 working wavefronts per CU on an inference from a census. It is measured now
(`prototype/gdn_rec_isa/results.md` §10; probes and their volume admission in §9, census in §8), and the answer
is no on both chains, with something better attached.

Best-of-5 of 40-bursts at the engine's own launch geometry (`16×128` per token-step), two windows, `rec0` =
6.29–6.36 µs per launch, windows agreeing to 0.5 %:

| probe | what it breaks | `rec0 / probe` |
|---|---|---|
| `rec3` | cross-token dependency (each token reads an independent state) | **0.918–0.923 — 8.4 % slower** |
| `rec4` | intra-token chain, 128 links → 4 × 32 | 1.073–1.089 (−160 slots explains ~8.4 % of it) |
| `rec5` | both | 0.977–0.982 |
| `rec6` | **nothing** — `rec4`'s loop shape, chain intact, −117 slots | **1.149–1.158** |

At matched loop shape `rec6_over_rec4` = **0.933–0.941**: splitting the serial accumulator costs 6.5 %. So
**neither dependency chain is what this kernel waits on**, and D30 items **1.3, 1.4 and 1.5 are closed** — no
occupancy work, no re-association, and the counters pass is no longer next either. Item **1.6 (16-bit state) is
closed too**, priced rather than waved away: it halves a term that is not the constraint (438.4 GB/s demanded
against a 240 plateau falsifies the traffic model per D21, and route A already removed essentially all of that
traffic for 1.119×) while costing 384 conversions against 1,523 slots, and its numerics question cannot be
posed with today's pins at all, because `state_fp.npy` rows are fingerprints rather than values.

**What it opens is item 1.9, and it is bit-identical.** `rec6` differs from the shipped kernel in loop *spelling*
only: same operations, same order, same values — `ndiff_o_vs0 = ndiff_state_vs0 = 0` in 4/4 cases and
`ndiff_*_vs_cpu` equal to `rec0`'s digit for digit. It removes 117 slots (−7.7 %), which at D25's measured
elasticity of 0.8 predicts **6.1 %**; it delivers **15 %**. That gap is the finding: the family is not
instruction-count-bound and not dependency-latency-bound, it is bound by **how its own loop gets scheduled**, and
unroll-4 spelling that folds address arithmetic out of the serial chain buys about twice what its instruction
delta buys. **D25 does not get extended, it gets restated**: what D25 closed was fusion and hand-folded
reductions — the axis it could price — and the axis that pays is spelling, which it never probed because a
spelling change looks like a no-op to an instruction census.

Prize arithmetic, stated as a ceiling and not a win: `gdn_rec` is 534.54 ms of a 1111.95 ms wave, so 1.15× at
launch level is **≈ 70 ms ≈ 6.3 % of the whole wave**, prefill 3828 → ~4080 tok/s if it survives. **If it
survives.** This is a burst over a 1 MiB single-layer state that stays resident for 128 tokens, and the engine
walks 18 layers per token with everything else competing for L2; D28 is exactly the precedent that a burst
winner can be an in-situ loser, and it is the reason nothing here enters `bench/performance-status.md` until an
in-situ A/B says so. A run that does not reproduce at wave level costs the plan the port and nothing else.

Instrument defect found before any number was read off it, and fixed: `--volume` was overloaded as both "run the
volume check" and "volume-only mode", so the documented `--volume --burst 40 --reps 5` printed `VOLUME_OK`,
exited 0 and timed **nothing**. Three windows were burned on that before it was caught. The burst now prints
`BURST-LINES printed=N want=cases×reps×modes` and fails when the matrix is incomplete — the fourth instance
this session of a driver printing PASS over an empty parse (`gdn_collapse/run.py`, `gate.py`'s prefill-row
count, `mixed_bw`'s block sum, and now this), which is why every new driver in this repo is now expected to
assert its own record count rather than print a verdict.

# D33 — the spelling survives the wave: `gdn_rec` **1.153×** in situ, prefill **3927 → 4284 tok/s**, and a burst winner is not necessarily an in-situ loser (2026-10-16)

`bench/performance-status.md` §7 is the ledger row; this is the decision and the correction to the reasoning
that surrounded it.

Paired inside one lock hold per width, alternating, medians of 3, blocking, zero subagents: **4096 lanes**
`gdn_rec` 496.55 → **430.69 ms (1.153×)**, whole wave 1043.11 → **956.12 ms (1.091×)**, prefill
**3927 → 4284 tok/s**; **1024 lanes** 1.100× on the family and 1.060× on the wave. Every `unroll4` rep beat
every `ship` rep (worst-vs-worst 1.110×), and all twelve runs carried `plan_match=1 launchstats_match=1
WAVEPARITY tokens_identical=1 logits_bitidentical=1`. Admission: ten fixtures green across both spellings, with
`gate.py` section (j) byte-comparing the two logits corpora inside every run (`wide1024f`: 1032 rows,
**1025.1 MB per side, identical**).

**The correction, stated as a rule and not as luck.** D32 said the burst claim was "explicitly not a win until
an in-situ A/B says so", citing D28's precedent that burst winners can be in-situ losers — and it was right to
say that, but the session's working inference had quietly become *burst numbers do not transfer*. Here the
launch-level 1.149–1.158× reproduced as 1.153× at family level, because the burst ran the engine's own launch
geometry (one `16×128` grid per token-step, 73,728 of them per 4096-lane wave), and D28's burst-vs-wave
divergence came from a different source: **library call ordering and L2 sharing across families, not from the
kernel's own micro-schedule.** The general form is therefore not "never trust a burst" but **ask what the burst
holds fixed**: a probe that keeps the shipped launch geometry and byte account transfers; a probe that changes
which calls surround it does not. Two precedents with different mechanisms were being quoted as one.

**What moved, and what it did not.** `gdn_rec` fell from 48.07 % to **44.79 %** of the wave and `attn` is now the
largest remaining non-recurrence family at **26.94 %** (257.56 ms of 956.12 ms), which is what re-ranks the
WMMA-attention question (D30 item 2.4) to second place. Decode moved **~0.1 %** by arithmetic — 18 launches per
step at 6.73 → 5.88 µs is 0.11 % of a 14.028 ms step, four times under the ruler's spread — so the 71.3 tok/s
row stands as an upper bound with the reason written down, and it is owed a re-take if the decode ruler ever
gets tighter than 1 %.

Two plumbing decisions, both forced by the flip rather than chosen for style. `gate.py`'s `--gdn-spell` default
moved in the same commit as the engine's, because a gate whose default disagrees with the shipped binary silently
judges a build nobody ships (the trap is already written up for `--attn-tile`). And the collapse-pair refusal is
now keyed on the flag having been **given**, not on the value differing from `ship`: with `unroll4` as the
default, a bare `--gdn-scan collapse` would otherwise have become an error the user never asked for — the same
class as `gate.py`'s inert-knob rule, one layer over. A default must not criminalise a legal run, and the echo
says `gdn_spell=inert-under-collapse` rather than naming a spelling that did not run.

**Residual, honestly.** The win is a schedule effect and the census cannot say *why* 117 fewer slots bought 15 %
rather than the 6.1 % D25's elasticity predicts — plausible candidates are address arithmetic leaving the serial
chain, or the unroll letting loads issue ahead of the accumulator they feed, and neither is measured. That gap
matters because it is the reason D30 item 1.9 generalises: if schedule is the axis, then the shipped
`rec_kernel`'s 1.15× is evidence that **other serial kernels may have the same free win in their spelling**, and
`prototype/gdn_rec_isa` now has the harness pattern (bit-identity gate + volume admission + bill-matched control)
to look for it cheaply. Also fixed in passing, found while plumbing the knob: `engine/gate.py --help` raised
`ValueError` from argparse's `%`-formatting in `--gemm`'s help text — the flag reference of the main gate driver
was unreachable since the lt flip, which is what the second commit says.

---

# D34 — **D30 item 2.4 (matrix-path attention) is a funded no**, and it dies at the ceiling this log itself had already adopted (2026-10-16)

`attn` is the largest non-recurrence family at **26.94 %** of the post-flip wave (257.56 ms of 956.12 ms), which
is why 2.4 was ranked second after `gdn_rec`. It is closed without a device measurement and without a kernel, and
the evidence is `prototype/attn_wmma/results.md` §1–§6.

**The verdict in one number.** At the arithmetic share this log already measured — D20's **27.9 %** of the touch
is the two dots' FMA work, the figure D30's own 2.4 row booked as the route's ceiling — and this kernel's own
measured slot→time elasticity (`time ∝ slots^0.39–0.44`, D21/D22), the whole WMMA prize after paying the feeder
cost is **−9.6 % of `attn`**, against a geometry tax whose floor was measured in kind at **7 %** and whose prior
for exactly this class of re-tile is negative. It does not clear the 15 % kill band; at the central case it does
not clear **zero**.

**Why the ceiling is the whole argument: the matrix unit is ~99.99 % idle for this workload.** One wave demands
≈1.07e9 MACs; the whole-device matrix roof is 2.96e13 MACs/s, so the matrix work is **≈36 µs against 257.56 ms**
of `attn`. That single ratio is why a faster arithmetic unit is not a prize here — the route can only win by
*retiring slots from the stream that is already the bottleneck*, and the retirable share is capped at 27.9 %. Any
row of the price table that clears the band needs the matrix-addressable share to be ≥ ~50 %, i.e. **more than was
measured and more than the row itself booked**.

**What the feeder probe settled, which `issue_rate_probe` §5.1 had explicitly refused to assume.**
`prototype/attn_wmma/wmma_feeder_probe.cpp` is compile-only — built with `-O3 --offload-arch=gfx1151
--save-temps -c`, **never launched**, no timing, no lock — and stages A/B in the engine's own row-major bf16
layouts. Marginal cost is differenced between the 1-tile and 16-tile variants of the same kernel, so prologue and
epilogue cancel:

| leg | slots per 16×16×16 tile | per touch-equivalent | vs the shipped 45-slot touch bill |
|---|---|---|---|
| score, map-aware (`rocwmma`) | **13.5** | 0.84 | **1.9 %** |
| PV, map-aware, fed from an `s_sc` row | **53.6** | 3.35 | **7.4 %** |
| score, hand-fed raw builtin (floor) | **6.5** | 0.41 | 0.9 % |

**Did the compiler have to move lanes? Yes** — 8 lane-movement ops per score tile, 20 per PV tile
(`ds_swizzle_b32` + `dpp`), because the WMMA operand map is not the engine's row-major 8-dims-per-lane slice. The
tax is real (~7 slots/tile) and it is **irrelevant**, because even paying it the score leg retires the dot for ~2 %
of what the dot costs today. That is the honest shape of the answer: the one unknown that could have rescued the
route came back *favourable*, and the route still dies. Two findings outlive the verdict: the **PV leg's 4× feed
cost is a layout fact, not a WMMA fact** (feeding A from `s_sc`'s 1024-byte row lowers to element-wise
`ds_load_u16`, 8 per tile), and this is the machinery a hand-written GEMM (3.3) would want — where, unlike here,
the arithmetic share is near-total, so the ceiling that killed 2.4 does not apply.

**Two source-forced costs no earlier plan for this route contained.** WMMA `16×16×16` needs **M = 16**, and the
shipped tile is `ATTN_TILE_M = 1` with a `static_assert` tying `blockDim.x / warpSize` to `ATTN_TILE_G`
(`engine/attn_slim_kernels.h:164`) — so this was never an instruction substitution, only a re-tile. And PV output
is `DH = 256`, so one warp keeping a whole row in D fragments is **16 fragments = 128 VGPR/lane against the
shipped kernel's 47 total**; N-splitting is mandatory, which changes the block shape again. The one geometry that
costs no wavefronts/CU (4×4 packing with flash-style tile-local scores, ~9 KB LDS) requires abandoning the
two-pass structure that exists so pass 2 can re-read whole-row scores — a deeper restructure than any item here.

**And a numerics cost the bf16 finding did not remove.** D33 corrected 2.4's premise — the *score* dot's bf16
operands need no conversion, since `v_wmma_f32_16x16x16_bf16` is present. The **streaming PV leg is still a
format change**, because it accumulates with **f32** weights `e[slot]` (`engine/attn_slim_kernels.h:307`) while
WMMA's P operand is bf16. That leg would need a floor probe and possibly a band decision, so 2.4 was never
format-free after all — only half of it was.

**Method note, because this is the third route closed by a ceiling rather than a measurement.** D31's workspace
budget, D32's dependency reading and now 2.4 were each refuted by asking what the prize could be *at most*, given
a number already in the ledger, before spending a device window. Here that cost one compile-only probe; the
alternative was implementing a re-tile to discover a −9.6 % against a 7 % tax. Related hygiene, applied at
admission: `prototype/attn_wmma/region_census.py` was handed over printing a plausible roll-up that was
**exactly 0.0** for `attn_tile_wave` against a published 116 while **exiting 0** — the fifth instance this session
of a check whose input can come back empty and therefore cannot fail (after `gdn_census/run.py`, `mixed_bw`, the
prefill-row count and the `gdn_rec_isa` `--volume` parse). It now aborts on both a zero roll-up and any calibration
miss. Its roll-up is unused for exactly this reason, and the file is committed labelled half-finished rather than
quietly trusted: the static bills and the 48-example rule selftest are the parts worth keeping.

---

# D35 — **D30 item 3.7: the deep algo field ships 2 pins of 4, at −0.75 % of wave, and the admissible subset that is worse than nothing** (2026-10-17)

Numbers and the paired instrument in `bench/performance-status.md` §9; the deriver and its admission rule in
`prototype/gemm_algo_sweep/deep_pins.py`; the mechanism in `engine/kernels_gemm.cpp`'s `PinSet`. Decision taken and
logged here rather than asked, per the standing rule.

**The burst ceiling was real and did not transfer.** `deep_pins.py` admitted 4 of 7 shapes at **1.903 % of wave**
warm-burst, against a per-shape cross-window noise floor it measures rather than assumes. In situ, paired, medians
of 3, both widths: `gate_proj+up_proj` **−10.0 % on its family at both widths, every rep**; the two `proj`
candidates `deep_o_proj+out_proj` and `deep_in_proj_qkv` **+26 % and +27 % slower** — the identical pair D28
rejected from an 8-deep field, now rejected again from a 64-deep one, which is the third independent confirmation
that a warm-burst per-call winner on this family is not an in-situ prediction. What ships is 2 of 4, worth
**−0.75 % of the wave** (sum3 190.05 → 182.62 ms at 4096), not the 1.9 %. No tok/s headline is claimed from this
window: `wall_ms` cannot resolve it, because between the two configs `gdn_rec` moved 470 → 426 ms and `attn`
272 → 259 ms — families **no algo pin can touch** — and those drifts carry the entire wall column. The null test on
the same logs (`proj` under a config that cannot change it: **+3.1 % / +1.9 %**) is what says which rows are
readable at all, and it is written into the ledger so a later reader does not quote the wall.

**The finding that outranks the win.** With only the `gateup` candidate in force and the *shipped* `down_proj` pin
untouched, `mlp_down` measured **+19.7 % at 4096 and +20.6 % at 1024, in all three reps at both widths** — a family
the flag does not name, moved by nothing but which kernel ran before it. That subset is **net-negative** (sum3
192.07 vs 190.05) even though its own shape won 10 %; adding the `down_proj` candidate restores `mlp_down` to 26.62,
so the deep-field `down_proj` algo is the one that suits the new `gateup` kernel's L2 state and the old pin is not.
Two rules follow, both now repo policy: **a pin set is adjudicated as a set or not at all** — per-shape promotion
from per-shape measurements can ship a loss — and the D33 instruction *"ask what the burst holds fixed"* now applies
**inside one binary**: two configs differing by one compiled-in row are not independent measurements of two shapes.

**Mechanism, because the adjudication was only possible with it.** Candidates are compiled in and unreachable by
default (`--gemm-pins all` means every *shipped* row and can never silently acquire an unadjudicated one), promotion
is a one-word `PinSet::kCandidate` → `kShipped` edit, `--gemm-heur` widens the pool at runtime without changing the
default pick (measured: rank-0 identical at width 8 and 64), and a candidate is addressed by its 24-byte record
rather than a rank, because the deep list is a **partly different list** — rank-0 agrees at both widths, rank 2
does not. Two refusals were needed as real fixes, not features: two pins in force for one `(N,K,M)` had previously
been resolved by table order, which would have made row order a measurement variable; and a workspace stamp above
the 128 MiB cap is now a refusal with its reason, since D31 closed raising the cap.

---

# D36 — **D35's shipped pair was wrong: the `down_proj` row is not superseded, and a pin set's coverage is what costs** (2026-10-17)

D35 shipped two pins and retired the original `down_proj` pin on an L2-state story. I stated a falsifier for it
before measuring, as the worker required: with the gateup candidate shipped and the *retired* pin named, `mlp_down`
should come back at ≈ 32.16 ms (4096) and ≈ 7.97 ms (1024), i.e. +17 %/+20 %. **It measured 26.82 and 6.56 —
+2.4 % and −0.5 %, both inside §9's own null.** Numbers, instrument and the corrected table in
`bench/performance-status.md` §9's same-day correction; the shipped set is now `down_proj` (the original D28 pin,
restored) **+** `deep_gate_proj+up_proj`, and the wave gain is the gateup row's alone at ≈ **−0.63 %**.

**The cause was a config label, and it is the kind of error a gate cannot catch.** `--gemm-pins <csv>` *replaces*
the admissible pin set; it does not extend the shipped one. So §9's `gu` side, which named only
`deep_gate_proj+up_proj`, did not hold "the gateup candidate plus the shipped `down_proj` pin" — it held the gateup
candidate and **nothing on `down_proj`**, which ran rank-0. The +20 % I attributed to a bad algo's interaction was a
shape with no pin at all. The binary said this out loud every time (`source=heuristic-rank-0` on every unpinned
shape in the `GEMM-LT` echoes, which is also what the worker's own §5 quote shows); I read the label and not the
echo, and the label was mine.

Two things follow, and both are stronger than what they replace. **Pin coverage is the expensive axis, not algo
identity**: on this family, once a neighbour changes, *removing* a shape from `all` costs ~20 % while swapping
between two near-equal winners costs nothing measurable — which is also why the retirement is not merely unsupported
but pointless: `deep_down_proj` is a tie against the incumbent, and a tie is exactly what `deep_pins.py`'s 0.1 %
materiality floor exists to refuse, since every pin is a re-capture obligation on the next library bump. And the
worker's verdict machinery turns out to have been load-bearing in the way I wanted: it enforces sets, and I stated
plainly in the handoff that **sets are not physics** — the falsifier existed precisely to catch this and it did.

**Trap, both directions.** For the main agent: a subset spec silently means an exclusion, so any A/B whose spec names
a subset must be read from the per-shape `source=` echoes before its family rows, and a config described in prose has
to state what is *not* pinned. For the deriver: `superseded` is a claim about an interaction, and an interaction
verdict must name the config that produced it — a verdict keyed to "shape + algo" without the co-named pin set can
record a coverage effect as an algo judgement, which is exactly how D35 got it backwards.
