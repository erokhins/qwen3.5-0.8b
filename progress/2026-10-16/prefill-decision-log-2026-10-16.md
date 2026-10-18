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

`engine/kernels_attn.cpp:65` sets `kAttnMaxSplit = 8` and the staging is `MAXSC = 512`
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
| 2.5 | raise `kAttnMaxSplit` / lane ceiling above 4096 (user, 2026-10-17) — see D41: **FUNDED, dump-first.** The admission is still required and still has to be sized, but the blocking step was the wide pinned dumps, and generating them is now approved. The one line the raise cannot be: a constant. | D4/D5. It is also the only route that could re-open 2.2 (traffic that actually reaches DRAM). Unblocking needs an admitted band for > MAXSC straddles, not a constant |
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

# D37 — **the bytes-vs-launches cut re-frames three D30 items at once, and the first one ships the same day** (2026-10-17)

Source: `progress/2026-10-17/glue-cast-copy-bytes-census.md` (worker, CPU-only), `tools/submission_probe.cpp`
and `tmp/gcc/ab.txt` (main agent, device). Detail lives there; what follows is the disposition.

**The framing that did the work.** Before pricing any fusion, ask of each family whether it is bound by
*bytes* or by *submissions*. That question was not in D30, and it moved three items at a stroke:

| item | D30 framed it as | the cut says | disposition |
|---|---|---|---|
| **4.1 `cast`** | a fusion | 29.25 µs/launch = **17× the dispatch floor**, 271.9 GB/s demanded | **funded as a deletion, shipped today** (`26cd5d6`, ledger §10) |
| **4.2 `copy`** | in-place staging to save bytes | 2.56 µs/launch = **1.4× floor**; whole family's bytes are 0.16 ms against a 10.86 ms row | **funded no as written**; redirected to `4.2′` |
| **4.3/`glue`** | fusion | 35.77 µs/launch but only 5 % floor-share and 116 GB/s — neither | **funded no** (D30's disposition stands, its stated reason does not) |
| **1.7 GDN trio** | fuse conv+gates+gated_norm into `rec` | `gdn_conv` is at **90.5 % of the copy probe** — at the wall, so fusing cannot recover its bytes | **funded no as framed**; `1.7′` (BF16 `g_raw`) stays funded pending §5-B |

**4.1 shipped: −57 % on the `cast` row, +1.86 % paired prefill.** 78 of 114 casts per wave body widened a
tile the library route never reads. The dead set was **derived from the readers, not from the census's line
numbers**, and that mattered: `norm_f32` is dead for attention layers only, because `in_proj_b`/`in_proj_a`
call `proj_gemv` directly at **every** width, so 18 of 24 mixer-input casts per body are live. The wall and
the row agreed at both widths (−17.65 ms measured against −15.57 ms predicted), which is the D21 test run in
the other direction — the deleted work was on the critical path, not L2-absorbed.

**A new device fact, and it re-prices a class.** `tools/submission_probe.cpp` submits the engine's own shape
(4096 separate 2 KiB D2D copies): **2.05 µs/op against an empty launch's 1.72 µs**, so a copy's bytes are
~0.33 µs and **its submission is 84 % of the op**; the identical work as one operation costs **0.10 ms
instead of 8.40 ms**. The repo's carried dispatch floor of 1.85 µs is re-taken at **1.72 µs** — the first
timing figure this box has produced that is not bimodal. Consequence: **4.2′ is funded at ~8.3 ms ≈ 0.87 % of
wave, inside the census's own predicted band** — the first D30 item whose ceiling a pre-measurement confirmed
*before* anything was built. What it does not settle is the engine half (an idle CPU submits at its own rate;
a busy stream may absorb submissions), so §5-C's paired A/B on a real gather kernel remains the gate, with
predictions numeric in both directions now.

**Two corrections to my own record, one of them the census correcting me.** D30 4.2's bytes framing is refuted
above. And the census's §5-A.3 claim that `gate.py`'s `wave_expect()` "must stop assuming a wave body owes 114
casts" **misread its own docstring**: that `49/24/24/114` is `proj` per body (5×18 + 4×6), and section (h)
cannot price `cast` at all because `engine/launch_stats.h` has no cast counter — a check over a family with no
counter cannot fail. So the item needed one bill deriver, not two. This is the same failure class I have been
logging all session, now caught in a *cost estimate* rather than a result: a check that structurally cannot
fail was being counted as plumbing that had to move.

**Next.** `4.2′` (one gather kernel replacing 4096 submissions, ~0.87 %, gated by its own §5-C A/B), then
`1.7′`. `2.6` (attn tile occupancy) and `3.3` (hand WMMA GEMM) are untouched; `2.5` needs a numerics admission
and `5.1` is out of authority.

# D38 — **per-op submission cost is *not* absorbed by a busy stream, so launch count is a real lever and D30 gets re-read** (2026-10-17)

Source: `47274f4`, `bench/performance-status.md` §11, `tmp/gcc/ab42p.txt`.

The submission probe said a 2 KiB copy is 84 % submission, but it ran with an idle CPU, where the submit rate
is by definition the binding rate. The open question was whether the same cost is exposed when the device is
busy — and the census wrote the answer as a **two-sided test**, not a hope: row collapses to ≲0.5 ms **and**
wall drops ≥5 ms, or the submissions were absorbed and every launch-shaped row in D30 needs re-reading.

**It is exposed.** Six of six pairs negative across two widths, wall −1.38 % at 4096 (−12.9 ms) for a row that
fell 9.34 ms, and the wall fell *more* than the row at both widths. So three things follow:

1. **Launch count is a lever in its own right on this box**, not a proxy for bytes. The empty-launch floor is
   1.72 µs and a real 2 KiB copy is 2.05 µs; at 4096 submissions per wave that is ~7 ms of wave that no amount
   of bandwidth can buy back.
2. **`glue`'s launch component gets re-read upward.** D30 4.3 was refused as a fusion and that stands, but its
   launch part is now priced by the same arithmetic rather than waved away as "dispatch-bound and bimodal" —
   392 submissions per wave is ~0.67 ms of floor, small but real and measurable.
3. **`gdn_rec` is the counterexample that keeps this honest.** Its 73,728 submissions per wave are *not*
   submission-shaped at 140 µs/launch (≈80× the floor), so this finding does not open a big new item there.
   A lever being real is not the same as every family being vulnerable to it, and the same ratio that made 4.2′
   fundable is what keeps `gdn_rec` out of scope.

**Method, second confirmation.** This is the D34 pattern working twice in two days on the same box: price the
mechanism with a small probe *before* building, state the refuting result as a finding, and let the build
follow the prediction. 4.2′ is the second D30 item whose ceiling a pre-measurement confirmed (4.1 was the
first, via the census's own bytes). Both times the pre-measurement also *shrank* the work: 4.1 needed one bill
deriver where the census priced two, and 4.2′ needed no `gate.py` change at all.

# D39 — **the per-token launch bill is an engine contract, not a specification claim, so a gate's expectation may move with the engine — under four conditions** (2026-10-17)

Source: `62b73e9`, `bench/performance-status.md` §13, `progress/2026-10-17/HANDOFF_17prime_gdn-cast-deletion.md`.

The worker hit the branch my brief had reserved as a stop-and-tell: 1.7′'s deletion is **route-blind and
width-blind**, so it removes a launch from §07's *per-token* bill (cast 115 → 97, total 424 → 406), which
`engine/gate.py` section (d) polices. Editing a checker's expectation to match a kernel inverts the direction
of proof, so it stopped instead of doing it. That was the right call and the decision was mine, not the user's,
because I checked where the number lives: **`grep` finds neither 115 nor 424 anywhere in `specification/`** —
§07 names operations, not dtype adapters. The bill is a derived formula in `gate.py` plus prose in
`engine/results.md`. A contract the checker derives from the engine's own structure is engine-internal, so
4.1-a's rule applied: engine and deriver in one commit.

**The four conditions, because "the gate moved" is only safe if they hold.** (1) the check stays an exact
integer equality *and* is re-proved to bite in both directions — `selftest_launch_bill()` now catches the cast
re-added (98 vs 97), a different cast wrongly dropped (96 vs 97), and a row the engine never printed, the last
being the case a `dict.get()` turns into a silent None; (2) `specification/`, `progress/`, `bench/` and
`devices/` were off-limits to the worker — journals and recorded measurements are re-pointed or annotated by the
main agent, not silently rewritten; (3) the live count claims in `engine/results.md`/`README`/`step_traffic_model.py`
move in their own commits, marking the booked opportunity that was taken rather than editing it out; (4) the
ship is contingent on the main agent's A/B, with the refuse-and-revert price stated in advance.

**Condition 4 is what nearly mattered.** The census predicted `gdn_conv` at ≈15 ms; it came back **+32 % slower**
(19.99 → 26.48 ms), because the packed F32 tile was a coalescing layout and not merely a format conversion. Net
of the three affected families the item is **−0.50 % of wave**, which clears `MIN_PCT_OF_WAVE`, so it ships —
but the refutation is booked as a named follow-up worth ~0.68 % of wave, and it is the third time this session
that a bytes-demanded argument predicted a timing consequence wrongly (D21, the census's `region_census`
calibration, and now this).

**Two provenance notes, one of them mine.** The handoff asserted "nothing in `engine/gate.py` was touched" while
the working tree carried 80 changed lines there including the bill formula — the state claim was stale, not the
work, and it was caught by `git status` before any of it was trusted. And the first A/B measured a **stale binary
as a clean null**: `plan_match` stayed green because the bill deriver is compiled into the same binary, so the
self-consistency check could not see staleness. A behavioural canary asserted inside the window is now mandatory
in these A/Bs; see the trap in `AGENTS.md`.

## D40 — the conv coalesced read ships; and an admission-honesty correction (2026-10-17)

- **Decision.** `CONV_VEC=4` ships as the conv spelling (`5096a57`): four consecutive channels per lane in one aligned 8-byte fetch, K-row window staged before consumption. `gdn_conv` 26.90 → 16.76 ms at 4096 lanes, five of five pairs negative, wall −0.91 % paired.
- **Why it was winnable at all.** §13's refuted prediction was that halving the conv read should speed it up; it slowed it down because the packed F32 tile was also a coalescing layout. The fix is therefore about *access shape*, and the ISA said so before anything was built: strided 2-byte loads with a drain after each tap, versus one 8-byte load with the window batched. This is the first D30 item sized by a refuted prediction instead of a byte census, and it is the second time in two days that a byte account pointed the wrong way (D21).
- **What is refused.** No headline restatement. §12's 4419 tok/s stands untouched until a clean ruler re-take in one window; this section's own base (931.82 ms) is a different window from §12's (926.70 ms), so the only number carried forward is the paired delta and the conv row.
- **Correction carried, not buried.** §10/§11/§13 claimed 18/18 gate admissions over nine fixtures; seven of those nine fixture names do not exist, so those three admissions were 4 real runs each. Recorded in §15.2 with the true population, what it does and does not invalidate, and the two gate-internal crashes on fixtures that do exist. The lesson generalises past this repo: a fixture list typed from memory is not a gate, and `grep -c PASS` over an error message is 0 — which reads as a failure of the code under test rather than of the harness, and so is invisible unless the harness asserts its own case count.

## D41 — item 2.5 unblocked: wide pinned reference dumps are approved, and a caveat I had stated wrong (2026-10-17)

- **Approved by the user.** Generating new pinned reference dumps at wide prompt lengths — the thing 2.5 was
  waiting on — is funded, and the numerics band that a > MAXSC straddle needs is **my call when we get there**,
  not a separate approval round. Item 2.5's row flips from BLOCKED to dump-first accordingly.
- **What I said wrong.** I justified the new bands by citing A-18, "greedy token equality decays with depth, so
  long-generation token match is not a gate". That is a *generation* statement and this is prefill: a long prompt
  is fed once and every row's logits are read from the same forward pass, so there is no generation depth for
  chaos to accumulate over, and the obvious move — take a long prompt, dump it — is exactly right. The caveat was
  inherited from the note rather than re-derived for the case in front of me, which is the same failure as
  quoting a number across a correction mark: the citation was real, the applicability was not.
- **The two caveats that do survive, and they are different animals.**
  1. **Wave carry, not chaos.** A multi-wave prefill legitimately is not bit-identical to a single-wave run of
     the same prompt, because the conv window carries across interior wave boundaries and reads rows an earlier
     wave wrote. `gate.py` already prints this on `dec64f` — "logits inside the cascade band rather than
     byte-identical … a WEAKER claim" — and the same will bind every > MAXSC dump taken at `--wave-rows` below
     prompt length. So the band is a **cascade band**, sized by comparing a one-wave reference against the
     multi-wave shipping geometry, and it has to be derived on the straddle geometry, not imported from the
     existing small-prompt bands.
  2. **Decision margin at depth, which is the legitimate descendant of A-18.** Deep rows frequently have a razor
     thin top-2 margin, so *argmax* agreement with the reference at depth is a weak test whatever the pass type.
     The tree's existing answer is decision-room-conditioned floors — rows with room must match, rows without
     are reported as ties — and `--prefill-logits all` plus the `prefill_row_coverage` forcing in `gate.py` are
     the instruments. So the band is per-row-conditioned, not global. Neither caveat is a blocker; both are
     sizing inputs.
- **Sizing, because `--prefill-logits all` is not free.** All-row logits are `rows × 151936 × 4 B` in F32: ~622 MB
  at 1024 lanes, ~2.5 GB at 4096. `--logits-bits bf16` halves that and is the first lever, not a compromise,
  since the pins are already read as BF16 downstream. Plan, cheapest first: re-dump the existing `wide640f` /
  `wide1024f` prompts with `--prefill-logits all` (pins rows above MAXSC immediately, no new prompt
  engineering), then a 4096-lane dump, then 8192, and 20k last and only if the cheaper widths do not settle the
  straddle question — a 20k eager reference pass materialises ~13 GB of attention per layer on CPU, so it is a
  deliberate step rather than a default.
- **Not implied by this approval, still stop-and-ask:** 5.1 int8/int4 (orders-of-magnitude error class, R8), and
  any change of scope beyond `Qwen3.5-0.8B` text path.

## D42 — queue item: audit the rejections, not just the open items (user, 2026-10-17)

Requested by the user after the `prefix24dec8` episode: **find the paths that were rejected because of this
class of error and re-evaluate them.** It is a queue item rather than a footnote because the error was not in a
kernel — it was in the *comparison*, and a comparison that misjudges one candidate has probably misjudged
others. `prefix24dec8` read as 19 shipping-path failures when the truth was one already-admitted 1.00-ULP flip
at step 2 whose free-run echo was then billed nineteen times. Every closed-negative on the ledger carries that
same possible defect: the instrument, not the candidate, produced the "no".

**Method — census, then a cheap named falsifier, in that order.** A rejection is re-opened only if (i) its
recorded evidence falls into one of the classes below, and (ii) someone can name a falsifier that is cheap to
run (a CPU re-analysis, a paired A/B inside one lock hold, a re-derived population). No candidate gets an
expensive re-sweep on (i) alone — that is how budgets die re-verifying things that were already green.

**The classes to search for, each already demonstrated in this repo:**

1. **Comparison conditioning invalid** — free-run compared against pins conditioned on different fed tokens
   (today), or any verdict drawn from a token-equality chain where an earlier step was admitted by tie, slack
   or room. Search the record for rejections whose evidence reads "tokens differ" or "diverges from step N on".
2. **A canary that cannot fail, or a stale binary.** AGENTS: a paired A/B against a stale binary reads as a
   clean null, because the bill deriver is compiled into the binary under test. Any "no difference measured"
   predating that rule (`bf1802f`) is suspect until re-taken with an in-window canary.
3. **A population that did not exist.** §15.2: seven of nine fixture names behind three admissions were
   phantoms. Any numerics or coverage claim citing `wide3100f` / `prefill_t*f` is re-adjudicated against the
   real population, and thin coverage (`wide520f` + `dec64f` only) is re-run at `wide1024f` / `wide640f` too —
   both of which pin every prefill row today, which also retires the old "no ground truth above MAXSC" limit.
4. **A price built from instruction slots at low occupancy.** Route A was priced at 11.4 ms and measured
   112.04 ms (9.8×) because the model assumed issue slots the machine does not have at ~1.6 realised wavefronts
   per CU; the chunked scan mispriced the same way (≥1.5× priced, 5.9× measured). **1.1** (instruction surgery,
   "CLOSED, funded no" on a 1523-slot census) and **1.2** (chunked scan, "measured loss") are the named
   candidates: both closed on price, and both price models are the discredited kind.
5. **A traffic claim taken near the plateau.** D21: `attn` looked bandwidth-bound at 225–248 GB/s against a 237
   ceiling, then a bit-identical instruction cut made it 1.521× faster with the byte census unchanged. So **2.2**
   (M=2 row tiling, parked on bytes-halved-but-slower) and **2.4** (WMMA, closed on a matrix-idle census) are
   re-read under "demanded bytes are not DRAM bytes" — and 2.4's census must state whether it read CCO
   containers or only the parseable `*.hsaco` set, since "0 of 55" and "0 of the readable 55" are not the same
   claim.
6. **A launch cost dismissed as absorbed.** **4.3** (`glue` launch count, "CLOSED for value, dispatch-bound and
   bimodal") is already contradicted by today's numbers: submission floor 1.72 µs/op, 4.2′ shipped −85 % on
   `copy` launches for −1.38 % wall, D38 re-priced `glue`'s launch component at ~0.67 ms per wave. **This one
   re-opens on the spot rather than waiting** — it needs only a paired A/B over a `glue` launch-count reduction
   inside one lock hold.

**Queue position.** After the gate-conditioning fix lands — it is the instrument this audit reads through — and
**before 2.6 and 3.3**, on the argument that re-opening a closed item is cheaper than opening a new one. Order
inside the audit: 4.3 (already proved by inference), then the class-4 candidates by re-pricing with no build,
then 2.2 / 2.4 under class 5, then a full re-adjudication of class 3 once `wide4096f` exists.

**What this item must not become.** Not a re-run of green gates (cite what is on disk), and not a licence to
widen a band: the delegated rule still binds, and a widened check must catch every injection in the must-catch
set derived from geometry, or the original rejection stands.

## D43 — retraction: the `prefix24dec8` "shipping-path correctness defect" was a comparison artifact (2026-10-17)

Commit `99fa8a7` says `prefix24dec8` "FAILS on the shipping `--gemm lt` route … at a row with 22 BF16 ULP of
room", and that framing went into a commit message on the strength of a gate log I misread twice (first as 0
failures, because the failure lines are indented and I grepped `^[FAIL]`; then as 19 real ones). It is
retracted: teacher-forced with the pin's own fed tokens, `--gemm lt` reproduces **every** pinned top-1, worst
cos deficit 1.2427e-04 against the 2.0000e-03 floor in force, and the 22-ULP row is among them. The one thing
the route moves is step 2, whose pin margin is exactly 1.00 ULP and which the gate already admits; the other 18
were the free-run echo of that admitted flip, compared against pins conditioned on a different fed token. The
instrument was fixed in `2ea4826` (section (b2) plus the ECHO bucket), and the handoff carries the blind spot
that fix cannot cover. Recorded here because a retraction that lives only in a chat log is not a retraction.

## D44 — D42 audit opens: 4.3 re-closed by arithmetic, and the >MAXSC evidence arrived unbidden (2026-10-17)

- **4.3 (`glue` launch count) is re-closed, and the correction is mine.** D42 said it re-opened on the spot
  because the submission floor and the shipped `copy` reduction contradicted the old closure. The arithmetic
  says the opposite: `glue` is 392 launches for 15.115 ms at 4096 lanes, 38.6 µs per launch against a
  **1.72 µs** submission floor, so deleting *every* glue launch — not a thing anyone can do — recovers
  392 × 1.72 µs = **0.674 ms = 0.073 %** of the 920 ms wave, and a realistic halving is **0.037 %**. Both sit
  under the `MIN_PCT_OF_WAVE = 0.1 %` refuse bar. `copy` was different in kind: 4240 launches at 2.6 µs each,
  i.e. launch-dominated, which is why cutting it moved 1 % of wave. **Lesson for the rest of the audit: the
  launch-floor bound is `launches × 1.72 µs`, and it must be computed before a launch-count item is re-opened,
  not after.**
- **The >MAXSC numerics admission has data now, and it says no new band is needed.** `wide4096p` (4096 lanes,
  `--decode 0`, one prompt family with `wide640f`/`wide1024f`) PASSes on `--gemm lt`: **4096 rows judged, worst
  cos deficit 4.3696e-04, worst meanrel 3.5902 %**, all inside floors already in force. D41 hedged that a
  cascade band for >MAXSC straddles would have to be derived first; at this width the existing floors hold with
  4×–8× margin, so 2.5 can proceed on evidence rather than on a borrowed band. What this does *not* settle: the
  decode-bearing side (4104 lanes) is still unjudged, because the ceiling refuses the plan.
- **And I fell into the same trap a third time in one day, caught before it landed.** My first read of the
  `wide4096f` gate run was "only 2 failures, both structural, so the rows passed". Zero rows had been judged:
  the launcher threw on the 4104-lane plan, `decode_run` exited 1, and the entire row section never ran — a
  `FAIL` count of 2 was read as "everything else passed". Third instance today of *a check that silently did
  not run reporting as a verdict* (after the phantom fixture names and the indented `[FAIL]` grep), which makes
  it a pattern rather than a mistake: **a gate log must be read for how many checks executed, not only for how
  many failed.** Worth a row in `AGENTS.md` when the audit next touches it.

## D45 — 2.5 ships at the minimum raise; the long-context target stays unfunded (2026-10-17)

- **Decision.** Ship `kAttnMaxSplit = 9` (ceiling 4608), not a larger number. The evidence covers exactly what
  is pinned — 4096 lanes of rows judged inside existing floors, plus the 4104-lane decode-bearing geometry —
  and the band **4105..4608 is legal but unpinned**, which is stated as a capacity bound rather than smuggled in
  as an admitted numerics claim. Going to 40 for the 20480-lane target would cost 161 MiB of partials
  (10.94 % of the checkpoint, affordable) but would admit 15 kB of width no reference has ever touched, and
  that is not this item's evidence to spend.
- **`kSessionDefaultCap` stays 4096.** It used to sit on the ceiling and now sits below it, so a session hits its
  own cap first and `--cap` is how a session reaches the wider geometry. Raising it costs +6 MiB of KV per
  request and no gate asserts it. That is a product default, and product defaults are not a mechanical
  follow-on from a constant someone else widened.
- **What this unlocks for the D42 audit.** Class 3 (re-adjudicate claims made on a population that did not
  exist) can now run at real width: `wide4096f` and `wide4096p` exist, are reproducible from
  `tools/dump_fixtures.sh`, and pass. Claims like "no ground truth above MAXSC" are dead in writing as well as
  in fact.
- **Next in queue** is class 4: re-price 1.1 (instruction surgery) and 1.2 (chunked scan) in cycles-per-
  dependency at realised occupancy rather than instruction slots, per the Route A lesson — and both re-pricings
  must be doable on recorded numbers with no build, because `gdn_rec` is 47.34 % of the wave and a wrong price
  there has already cost this project one 9.8× miss.

## D46 — D42 class 4: both `gdn_rec` closures stand, but neither for the reason recorded (2026-10-17)

- **Verdict.** 1.1 (instruction surgery) and 1.2 (chunked scan, P9) both stay closed. The audit's value is in
  the *reasons*: D23 closed 1.1 on a roof ratio computed against the wrong machine (it presumed one SIMD per
  wavefront; there are 64 wavefronts for 80 SIMDs on ≤16 busy CUs, so the floor is 2,812 cycles and the family
  sits at 16 %, not 10 %), and then closed on the **floor** of its own band while its top of band — ~66 ms —
  was later realised almost exactly (−65.86 ms) by loop spelling, a lever the census did not contain. So the
  census was not conservative, it was wrong in a direction that already cost time.
- **Measured elasticity of the axis is ~0.8** (D25's MODE probes), not ~0: on the axis that removes work, slots
  *do* translate. That is the correction, and it is why the residual named sub-slice had to be screened at all
  (§18) instead of being dismissed by the old ratio.
- **1.2's "measured loss" measured an implementation, not chunking** — a thread-mapping defect — but the
  re-price still cannot resurrect it: nothing in the recorded numbers supports a differently-shaped chunking
  reaching the bar, so it stays closed on new grounds rather than the old 5.9×.
- **The one live thread** is `rec2c` (§18): screen median 1.0268 vs a 1.02 kill line, one case below the line,
  and decisively — it is fused-prologue-vs-pre-unroll4, a kernel the engine no longer runs. It is **not
  admitted and not ported**; the gate in front of it is a compile-only census of the merged
  fused-prologue-plus-unroll4 listing, because the failure mode (the merge quietly returns the unroll through
  register pressure) is invisible to any burst run against a binary that lacks it.
- **What class 4 bought overall:** two closures made durable (resting on measured elasticity and a failed
  hand-schedule rather than on a roof ratio), one mispriced claim retired, and the queue cleared to class 5
  without a re-open that would have burned a device budget on a kernel shape nobody had built.

## D47 — queue item: the absolute ceiling, and the distance to it (user, 2026-10-17)

The user asked what the theoretical operation count is from the model spec, what the maximum prefill speed on
this GPU would be at known fp16/bf16 matrix rates, and how far off it we are. First pass done at the desk
(`tools/prefill_ceiling_bound.py`, CPU-only, every input cited in its header); the rigorous version is the
queue item.

- **FLOPs per prefill token, derived from the spec: 1.603 GFLOP.** Body matmuls 0.994 (from the engine's own
  `(N,K)` table: `q_proj 4096×1024`, `k/v 512×1024`, `o/out 1024×2048`, `in_proj_qkv 6144×1024`,
  `in_proj_z 2048×1024`, `gate/up 3584×1024 ×2`, `down 1024×3584`), `lm_head` **per prefill row** 0.509 (the
  tied 248320×1024 table — and the launch bill confirms one `lm_head` per token body, so this is real work in
  the timed path, not an artifact of the gate's all-rows mode), causal-average full attention at 4096 ctx 0.101.
- **The independent cross-check lands**: `devices/strix-halo-compute.md` records the whole prefill achieving
  1.55 TFLOP/s at 967 tok/s, which implies **1.60 GFLOP/token** — the spec-derived count agrees to the third
  digit from a measurement taken before this item existed.
- **Engine today demands 7.14 TFLOP/s** (4454 tok/s × 1.603 GFLOP). Against the **measured achievable mix**
  (22.3–23.3 TFLOP/s at `M ≥ 512`, window `prefill-Wclean`) the ceiling is **~13,900–14,500 tok/s** and we are
  at **30.6–32.0 %** of it. Against the best single measured shape (37.7 TFLOP/s) it is **~23,500 tok/s** and
  we are at **18.9 %**. Weight traffic is nowhere near binding for prefill: a 512-row wave reads the weights once
  in ~6.5 ms, which bounds at ~78,500 tok/s.
- **Decode's wall is bytes, and it is much closer.** 1.5457 GB of bf16 weights per token at the measured
  ~237 GB/s is **~153 tok/s** as an absolute bound; today's **71.3 tok/s is ~46 %** of that, which is the same
  arithmetic as D21's 75.9 GB/s aggregate read against the 164–168 GB/s plateau.
- **The honest reading of "a ceiling we won't break": the matrix roof is not the binding one.** 47 % of the wave
  is `gdn_rec`, a serial per-token scan that issues no matmul at all, and its own measured behaviour (unroll8
  changing nothing; 1.6 working wavefronts per CU) says it is latency-bound, not op-bound. So ~14k tok/s is what
  the matrix units would allow **if every non-matmul cost went to zero** — a bound, not a target. The item's
  real output is the per-family split: ops roof, bytes roof, measured, per family, so "how close are we" is
  answered per family rather than in aggregate.
- **Queued work.** (1) the clean dense-matrix peak probe at 16-bit `C`, which `devices/strix-halo-compute.md`
  explicitly says is still owed — without it the 37.7 figure is a shape we use, not a peak; (2) the per-family
  ops/bytes/measured table; (3) confirming `lm_head`'s per-row cost is the shipping intent rather than an
  artefact — if only the last row needed logits, 0.509 of 1.603 GFLOP/token, **32 % of the arithmetic**, would
  vanish, which makes it the largest single lever this census has found and squarely a design question, not an
  optimisation.

### D47 amendment — the per-row `lm_head` assumption was wrong, and the correction is worth more than the original

`prefill.cpp:85/91` prices `finish_token` for **every row only in `--prefill-logits all` mode**; the shipping
default `kLast` prices row `n−1` alone, and the family table proves it: `lm_head launches=1` for a whole
4096-token prefill, 2.197 ms. So the timed-path count is **1.0947 GFLOP/token**, not 1.603 — and the engine
demands **4.87 TFLOP/s**, i.e. **20.9–21.9 %** of the measured mix roof rather than 32 %. The 2.197 ms single
launch is itself a clean number: 485 MiB read in 2.197 ms = **232 GB/s**, i.e. `lm_head` is at the bandwidth
ceiling exactly and has nothing left to give.

Correcting that unblocked the actually useful result, which is per-family rather than aggregate:

| family group | ms | FLOP | achieved | vs mix roof 22.3–23.3 |
|---|---|---|---|---|
| `proj` + `mlp_gateup` + `mlp_down` | 179.24 | 4.072 TFLOP | **22.72 TFLOP/s** | **97–102 %** |
| `attn` | 263.20 | 0.412 TFLOP | 1.57 TFLOP/s | 6.7 % |
| `gdn_rec` | 442.55 | no matmul | — | not a matrix question |

**The GEMM path is already at the achievable matrix roof** — 97–102 % of the rate `prefill-Wclean` measured for
this exact wave mix, which means every GEMM-side item on this project (algo pins, workspace, hand-WMMA) is
competing for a prize that the roof says is ~0, and the ledger's habit of pricing those items at 1–2 % of wave
should be read against that. `attn` sits at 6.7 % of the roof, but it is a streaming small-M shape rather than
a GEMM, so the roof is the wrong yardstick there too. And `gdn_rec`, at 48.1 % of wall, is not a matrix
question at all.

So the answer to "what ceiling can we not break" splits in two: **prefill ~21,000 tok/s** if every non-matmul
cost vanished (and ~34,400 at the single-shape peak), with the hard part being that 80.8 % of the wave is not
matmul; **decode ~153 tok/s** from bytes (6.5 ms of weights per token at 237 GB/s), **106 tok/s** at the
realistic 164 GB/s plateau, against today's 71.3 — so decode is at **47 % of the hard bound and 67 % of the
realistic one**, and that is the tightest wall this project has.

## D48 — queue head: run llama.cpp on this model as a comparison instrument (user, 2026-10-17)

The user's ask: measure llama.cpp prefill on `Qwen/Qwen3.5-0.8B` on this box, expecting it to be much faster
(“like 9k”), and **find out why** — CPU/GPU usage during prefill, sources, and a profiling division comparable
to our `--time-waves` table. Cloning into `engines/` is authorised. Priority: **head of queue**, ahead of
class 5, 2.6 and 3.3, because it is an instrument rather than an optimisation.

**Arithmetic before the measurement, so the measurement is aimed at something.**

- 9,000 tok/s × 1.0947 GFLOP/token = **9.85 TFLOP/s demanded**, which is 44 % of the measured mix roof — and
  our GEMM families already run at **97–102 %** of that roof (D47 amendment). So **no comparison can be won in
  the matmuls**: any real gap lives in the 80.8 % of the wave that is not matmul.
- The user's 9k is arithmetically the *same sentence* as “`gdn_rec` → 0”: that family is **442.55 ms of the
  920.18 ms wave**, and deleting it gives 477.6 ms = **8,575 tok/s**. The comparison's target is therefore one
  number, not a whole engine.
- Why llama.cpp might have it: `GGML_OP_GATED_DELTA_NET` has **no fused chunked prefill kernel on CUDA** (open
  issue ggml-org/llama.cpp#22967) and none on Metal (#20342); the Qwen3-Next GDN path was written with “only
  existing ggml functions … so it can work on all backends”, i.e. the prompt path is expressed as generic
  ops — which is a parallel, matmul-shaped formulation of the recurrence rather than our per-token serial one.
  That is the same axis as **1.2 (chunked scan)**, which D46 left closed on the grounds that its 5.9× “measured
  loss” was actually a thread-mapping defect. If llama.cpp is genuinely ~2× on this model, that closure has to
  be re-opened with a named external implementation as evidence rather than a discredited price model.
- Counter-evidence to hold alongside, so this is not a confirmation hunt: public Strix Halo llama.cpp reports
  put prefill in the **hundreds** of prompt tok/s for Qwen3-class models on this hardware (one measures 216 vs
  406 tok/s across a lazy-mode switch), so ~9k for a 0.8B is plausible-but-not-given. Either outcome is useful:
  if llama.cpp is *slower*, our engine's number stands up against an outside implementation and the queue stays
  as-is; if it is faster, the gap is now localised to one family with an external working example.

**Stages.** *S1 (subagent, CPU-only)*: clone to `engines/llama.cpp`, read the tree for Qwen3.5/`qwen3_next`
model support and the GDN op path (file + line for how prompt processing is formulated), find the ROCm/gfx1151
build recipe and a bf16/f16 (not quantised) GGUF, and hand back exact build + bench commands plus the
per-op profiling switch (`--profile-ops`) so S2 produces a table in our units. *S2 (main agent, device,
blocking, zero subagents)*: build, run `llama-bench -pp 512,1024,4096` at documented `-b/-ub/-fa/-ngl/-t`,
capture the per-op profile plus **CPU % and GPU busy during prefill** (`pidstat` + `rocm-smi`, with this repo's
standing caveat that `rocm-smi` util is not a contention gate), and record conditions so the comparison cannot
credit the GPU for CPU work. *S3 (verdict)*: which structural choice explains the delta, and — only if the
delta is real — what it does to D46's closure of 1.2.

**Fallback**, if the arch is unsupported or the build will not fly on ROCm 10: price the same axis directly by
prototyping the chunked (matmul-shaped) GDN prefill against the pinned pins. That is the real target anyway;
llama.cpp is only the existence proof.

### D48 addendum — is llama.cpp's formulation the same thing as our item 1.2? Yes, and the gap between our price and our measurement is the whole question

Asked directly whether the llama.cpp difference is the chunked scan. It is the same reformulation, and the
records say so: `progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md:285` describes P9 as “the
delta-rule scan reformulated chunk-wise, which **is GEMM-shaped**”. Two things then have to be reconciled:

- **Our bottom-up price of the chunked form: 63.6 ms** against a serial `gdn_rec` of **127.0 ms** at 1024 lanes
  (`p9-route-a-2026-10-14.md`), i.e. ~2× on the family — which, scaled to the 4096-lane wave, is roughly the
  8,500 tok/s the user expects.
- **Our measurement: 1,953.7 ms wave vs 331.2 ms** (`p9-phase1c` §Q2), `gdn_rec` 127 → ~1,750 ms, i.e. **13.8×
  worse**, which the same record attributes to a **thread-mapping defect** (the carry kernel's header claimed
  `thread = (chunk row t, strip column j)`; what ran fed a different element-per-thread mapping) and to the
  kernel **re-executing the priced arithmetic 13.5× over** (`p9-route-a`: “13.5× the priced arithmetic”).

A ~30× disagreement between the price and the measurement of one idea, with the defect already named, means
neither number is currently trustworthy about chunking. **The specific untested combination** is chunked GDN
expressed as *library GEMM calls* — llama.cpp emits `ggml_mul_mat` + `ggml_tri` and therefore rides its tuned
matmul path, whereas P9 hand-wrote the chunk arithmetic into a HIP scan kernel and never touched the tensor
cores. On this box that distinction is worth naming precisely, because D47 measured our own GEMM families at
**97–102 % of the achievable matrix roof**: an expression that lands there inherits a rate we have already
proven we can hit, and an expression that hand-rolls the same arithmetic inherits our occupancy problem
(2,048 independent chains in the recurrent form, `p9-route-a`).

## D49 — decide-and-record instead of waiting for approval; D48 extended to a per-family timing comparison (user, 2026-10-17)

- **Standing, reaffirmed by the user:** do not stop for approvals — use judgement, act, and **record the
  decision** here. What still stops the session is only what is genuinely the user's: a numerics band wider than
  an accepted one, a scope change, a destructive action, int8/int4 quantisation (out of authority per the
  numeric-format rule), and queue reordering that drops funded work.
- **D48 extended (was: measure prefill tok/s and find out why).** S2 now owes a **per-family timing table for
  llama.cpp**, not one headline number: per-op timings from its own profiler (`--profile-ops` /
  `llama_perf_context_print`) at 512 / 1024 / 4096 prompt tokens, mapped onto our `--time-waves` families
  (`proj`, `mlp_*`, `attn`, `gdn_conv`, `gdn_rec`, gates/norm, `cast`, `copy`, `glue`, `lm_head`, `host_tail`),
  with **`llama/ours` ratio per comparable row** and an explicit *not-comparable* column for things that cannot
  be ratioed (its scheduler and graph-build overhead, `-ub` as a policy knob, model load, our engine having no
  vision/MoE path in scope here).
- **The mandatory part: every row where llama.cpp is ≥1.5× faster becomes an investigation item**, each with a
  named method rather than a vibe: (a) read the source for that op's backend path and name the kernel or
  library call; (b) isolate that one op at **our** geometry in both engines (same shapes, dtype, entropic
  fill, one lock hold); (c) attribute the delta to exactly one of **traffic / launch count / occupancy /
  fusion**, and give a falsifier for the attribution. This is the class-1 lesson applied to an outside
  implementation: “their kernel is better” is not a finding, the mechanism is.
- **Knob discipline, so a default cannot masquerade as a win:** sweep `-ub` ∈ {128, 256, 512, 1024, 2048},
  `-fa` on/off, `-ngl 99` vs partial offload, `-t` ∈ {4, 8, all}, and record which setting produced any claim —
  with our own numbers taken **in the same lock hold** where a ratio is being quoted (§16's ruler lesson: a
  re-take in a different window is not an A/B).
- **New tool, tested: `tools/prefill_sampler.sh TAG -- cmd…`** records `ts_s, cpu%, mem_avail, gpu_use, vram`
  around a run on either engine. Honest limits, written into the tool: the **GPU-busy column saturates on this
  box** (the engine serving this session busy-polls the iGPU — 100 % while doing nothing but serving tokens),
  so it is a sanity check that the process touched the device, never a contention gate; CPU % and VRAM are the
  columns to read. It exits 1 if it collected zero samples, so a dead sampler cannot present as a clean run.

## D50 — `rec2c` re-opens and clears its gate: the merged fused+unroll4 form is exact and screens above the kill line (2026-10-17)

- **D46's gate passed on the listing**: the merge kept both halves (1,374 vs 1,406 slots, 59 VGPRs unchanged,
  loop B 16×8 unchanged, all state-walk categories identical), so the silent-failure branch — *fused but slower
  because the merge handed the unroll back* — is disproved before any device time was spent. That ordering was
  the point of gating on a census.
- **And it is exact**: first device run of the merged kernel, `ndiff = 0` on both `o` and the state across all
  four cases, with the volume account unchanged. Bit-identity is now *measured*, not inherited from two
  transcriptions.
- **Burst median 1.0415** (mean 1.0343, worst case 0.9908 at 4/7 pairs) against a 1.02 kill line and a census
  prediction of ~1.018. So the axis is still giving back more than its instruction delta, consistent with D25's
  −2.0 %-probe returning 1.0159 and with the unroll4 change beating its price by 2.4×.
- **Decision (mine, recorded here, not waited on): port it as a knob, do not flip a default.** New spelling
  behind `--gdn-spell fused`, `kUnroll4` stays the shipping default, gate section (j) becomes a three-way
  byte-compare, then a paired in-situ A/B inside one lock hold decides the default. This is the D32 → D33 order
  repeated deliberately: a knob ships as a knob first, and the family row — not a harness ratio — is the
  instrument.
- **Cost of being wrong is bounded and known**: if the in-situ A/B is flat or negative, `rec2c` closes for good
  with a clean audit trail — census exact, bit-identity measured, screen positive, in-situ null — and the ~0.85 %
  prediction goes down as a measured null rather than as an unpriced idea.

## D51 — `fused` becomes the shipping default, and the headline is reported as unchanged (2026-10-17)

- **Decision (mine, recorded not waited for): move the default to `--gdn-spell fused`.** The instrument that
  decides was run as D50 promised: paired, both sides inside one lock hold, alternating every rep, canaries read
  in the same window (engine-TU ISA canary with both bills pinned; every side echoing its own spelling; unknown
  spelling refused so an inert flag cannot masquerade as a side). **4096 lanes: family ratio 1.0645, 3/3 reps,
  worst rep exactly 1.0200.** Decode flat (1.0025, 2/3). 1024 lanes flat-to-adverse (0.9909, 1/3). Bit-identity
  re-proved per fixture through the shipping launch path, 239 gate checks 0 failures.
- **But the headline did not move, and it is reported that way.** Ruler re-taken on the new default: median
  **923.36 ms = 4,436 tok/s** vs §17’s **921.56 ms = 4,444**. The paired window’s `unroll4` side was ~2.8 %
  slower than the historical median, so the paired −2.96 % of wall is real *as an attribution* and false *as a
  headline*. Cross-window the family row is −2.6 %, not −6.5 %. **Prefill stays quoted at 4,444 tok/s** until a
  clean-window ruler places it elsewhere; decode stays 71.3.
- **Cost of the flip, named**: gate section (j) now runs 3 device runs per fixture instead of 2 (+50 % of that
  section). If that becomes the thing that makes people skip the gate, the remedy is a narrower fixture for (j),
  never a dropped side.
- **Follow-ups this task surfaced and did not silently absorb.** (F1) **18 line-citations rotted** by the port —
  nine in files the port touched (`engine/kernels_gdn.cpp:40,32` etc.), and the worker proved they were already
  semantically wrong at HEAD (`:40` claims a `rec_kernel` launch that lived at `:64`). Remedy owed: re-point
  them at the **symbol** (`kernels_gdn.cpp::gdn_rec`), which is rot-proof, rather than at new line numbers.
  (F5) `DecodeScratch::gdn_spell` still struct-defaults to `kShip` while `decode_run` defaults to `fused` — inert
  today because the only caller always assigns, and worth a one-line change if a second scratch consumer ever
  appears. (F2/F3) The gate’s three-way compare needed a *new* denominator check and an argparse extension — the
  outline as written would have let a missing side pass and refused the very spelling it was added for.

## D52 — the comparison came back: llama.cpp is 1.7–2.3× faster at prefill, and two of my premises were wrong (2026-10-17)

- **The gap is real and large** (§21): 8,290–10,273 tok/s vs our 4,430 at 4096 tokens, same lock hold, same
  width class, unquantised f16. The user predicted ~9k; the measured matched row is 1.87× and their best swept
  row is 2.32×.
- **My premise about *how* they do it was wrong, and it is retracted.** D48 recorded that llama.cpp has no fused
  GDN kernel and expresses the prompt path as generic ops with a triangular term. In **this** build the node
  census prints `GATED_DELT → ROCm0` (324 nodes) and **no `TRI`/`SOLVE_TRI` node exists in the `qwen35` graph at
  all**; `SSM_CONV` is on the GPU too, and the only CPU node is the embedding `GET_ROWS`. The upstream issue text
  and my own path-mistaken grep were describing an absent backend, not this one. So the gap is **not** "they
  chunk GDN as GEMMs and we hand-roll a scan" — at least part of it is "their fused GDN kernel runs faster than
  our serial scan", which is a different investigation with a different remedy.
- **What that does to item 1.2 (chunked scan): nothing yet, and that is the discipline.** An external
  implementation being faster is not evidence for *chunking*; it is evidence that *something* in our
  non-matmul 80 % is ~2× off. Attribution comes first: per-kernel shares for both engines at 4096 lanes
  (`rocprofv3 --kernel-trace`, S1's §6.2 card), then each ≥1.5× family row gets its mechanism named —
  traffic / launch count / occupancy / fusion — with a falsifier, per D49. Re-opening 1.2 on "llama.cpp is fast"
  would be exactly the D42 failure mode this audit was created to stop, one layer over.
- **The ceiling is intact and now externally corroborated**: their best demands 48 % of the measured mix roof,
  we demand 21 %. The ~21,000 tok/s bound from D47 still holds, and 2× our rate is no longer hypothetical.
- **Decision: prefill investigation is now the head of the queue**, ahead of class 5, 2.6 and 3.3. Two named
  candidates, both already visible in our own family table: `gdn_rec` at 46–48 % of the wave (their GDN path
  hides its per-token cost far better) and `attn` at 263 ms / 6.7 % of the matrix roof with quadratic growth.
  The `-t 4 > -t 32` anomaly is recorded as unexplained, not dropped.

## D53 — what the trace licenses: batch the GDN *launch*, which is not the thing P9 was (2026-10-17)

§22 says the GDN gap is 512× in launch count and ≥9× in time, and that attention is ≥5× at **equal** launch
counts. Two decisions follow, and the first deliberately does **not** re-open 1.2:

- **New item, ahead of everything: one launch per N tokens for `gdn_rec`, recurrence unchanged.** Today the
  engine submits 73,728 kernels for a 4096-token prompt (18 layers × 4,096 tokens); the external existence
  proof walks 512 tokens inside a single launch. Looping the token index *inside* the kernel removes 4,095 of
  every 4,096 launches per layer — and because each token's arithmetic is the same expression tree in the same
  order, and addresses do not move (same grid `<<<16,128>>>`, same operands), the output is **bit-identical by
  construction**, so no band is requested and no gate is asked to move. The prize is bounded below by the
  submission floor alone: 73,728 × 1.72 µs ≈ **127 ms ≈ 13.8 % of the wave**, before any saving on the
  per-launch prologue/drain that §19's fused prologue just made cheaper per launch and therefore relatively less
  painful. **This is distinct from 1.2/P9**, which reformulated the recurrence into chunked matmul math, changed
  the association, and measured 13.8× worse for reasons D46 attributed to a thread-mapping defect.
- **Attention (item 2.6) is promoted by evidence, not by taste**: same launch count as an outside kernel that is
  ≥5× faster, which retires "it is launch overhead" as an explanation for that row and leaves tile shape and
  occupancy as the surviving candidates.
- **What would falsify the GDN item cheaply:** a prototype MODE that loops N tokens per launch and is *not*
  bit-identical to the per-token form (that would mean the loop crossed an association boundary, which the §19
  `q̃` lesson says is possible via compiler contraction), or a paired wave A/B in which the `gdn_rec` family row
  does not move when the launch count does. Both are one prototype run and one A/B.

## D54 — D53 screens at 1.58–1.64×, bit-identically; port it as `--gdn-loop`, knob first (2026-10-17)

- **Result (§23):** one launch per body beats 128 per-token launches by **1.5834–1.6363** (whole-body span) and
  **1.4713–1.5553** at span 32, **8/8 cases at 7/7 pairs**, with device admission bit-identical at spans
  0/1/3/16. Per-token cost 5.4–5.7 µs → **3.4–3.5 µs**. If it holds in situ, `gdn_rec` ≈ 428 → ~270 ms, about
  **17 % of the wave** — the largest single lever this project has measured, and it comes from the axis the
  llama.cpp comparison exposed (§22), not from anything on the old queue.
- **Decision (mine, recorded): port it behind `--gdn-loop N`, default `1`.** Same order as D32→D33→D50: knob
  first, paired in-situ A/B on the `gdn_rec` family row decides, and the headline is re-taken in a separate
  clean window before it moves. A one-row body **must** keep the per-token path whatever the flag says, which is
  what makes decode unchanged by construction rather than by measurement (§6 item 3 of the handoff).
- **Two sub-decisions taken inside that.** (a) The **addressing variant** (increment six row pointers instead of
  recomputing `t*CH`, worth −156 slots) was **not** prototyped before this window — the screen already returned
  1.58–1.64× with the naive addressing, and guessing a mitigation before measuring the thing it mitigates is the
  pattern this project keeps paying for. It stays available as the named follow-up if the in-situ ratio lands
  under the screen's. (b) **Span policy:** `--gdn-loop` accepts an explicit N but the engine will pin it to the
  body size, because span-invariance is *proven* (§3's mirror) while partial spans serve a preemption policy we
  do not have. Free knob today, honest default, no invented requirement.
- **What is still owed before this is a headline:** the paired in-situ A/B at 4096 **and** 1024 lanes (the 1024
  shape has 2 bodies, not 8, so a win at 4096 with a loss at 1024 would be a body-count effect and must be
  reported that way), decode confirmed flat, gate section (j) extended to the looped side, and a clean-window
  ruler re-take. `plan_match` is explicitly not the canary — the counted `family=gdn_rec launches=` drop (~512×)
  and a byte-compare of every prefill logits row are.
- **Queue effect:** 2.6 (attention tile) stays second — §22 gives it ≥5× at equal launch counts — but D53/D54 is
  first by a wide margin, and class 5 / 3.3 wait behind both.

## D55 — the looped launch is the default; prefill is **5,364 tok/s**; attention is the last ≥1.5× row (2026-10-17)

- **Shipped:** `--gdn-loop 512` (D54's knob, §24's family row 1.60–1.69 with 1,024/1,024 prefill logits rows
  byte-identical, gate 370/0 on `wide520f` with the looped form on every pinned path). Ruler **921.56 → 763.49 ms
  = 4,444 → 5,364 tok/s (+20.7 %)**, §1's banner re-taken in the same window, short widths paired at
  1.574/1.640/1.633 family ratio. This is the first item this project shipped that was found by *comparison
  against an external engine* rather than by the internal queue — the D52/D53 chain (llama.cpp launch census →
  512× GDN launch gap → walk the token index in-kernel) is what produced it, and it is worth 17–22 % of the wave
  where the internal queue's best open item was priced at a few percent.
- **Consequence for queue order, decided and recorded:** **2.6 attention tile is next and is the only ≥1.5× row
  left standing.** §22 counted both engines at 48 attention launches, so this is not a launch-count item —
  ours is 263 ms of the wave against their ≤52 ms at the same launch structure, and after §25 `attn` is the
  largest family in the wave (34.0–34.9 %). The §25 wall number also re-prices everything downstream: 1.0947
  GFLOP/token against 763.49 ms demands **1.43 TFLOP/s**, i.e. the engine now sits at ~30 % of the mix roof
  against D47's 21 %, so the remaining non-matmul time is concentrated in two families, not spread thin.
- **Method rules that paid for themselves this round, recorded because they were load-bearing:** (a) *a paired
  window, always, for a short-prompt row* — §1's 64-lane row looked like a 15 % regression across windows and
  was flat-to-better inside one hold; (b) *a sweep must print the count it claims* — the first short-width A/B
  fed space-separated ids to a comma-separated parser, priced a one-token wave at every width, and read as
  "the loop does nothing below 1,024 lanes"; `launches=` beside the family `ms` is now part of that instrument;
  (c) *the default flip changes refusal order* — a looped default made `--gdn-spell unroll4` alone an
  unrunnable pair, so that refusal now tests the effective span rather than `--given`, or the kernel would
  throw after a 1.55 GB load.
- **Not done, explicitly still open:** the pointer-increment address spelling (−156 slots, bit-neutral, now a
  follow-up rather than a prerequisite); D42 class 5 (2.2 M=2 tiling, 2.4 WMMA under "demanded bytes ≠ DRAM
  bytes"); class 3 re-adjudication via `wide4096f`; 3.3 hand WMMA GEMM, which carries D47's "GEMMs are at the
  roof" prior and is now cheap to falsify against a 763 ms wall; and the 18 rotted line-cites from D51 F1.

## D56 — attention is 9 % of the FLOPs and 34 % of the wall: **1.57 TFLOP/s, 7 % of the mix roof** (2026-10-17) — **rate framing retracted by D57: the roof must be the scalar one, so the gap is 4.6× not 14×**

Priced with `tools/prefill_ceiling_bound.py` (D47's instrument) against §25's wall, because item 2.6 was
promoted on an external comparison and an external comparison does not tell you the size of the prize:

- spec FLOPs, full attention at 4,096 ctx: **0.1007 GFLOP/token × 4,096 = 412.5 GFLOP** of the wave's
  4,483 GFLOP — **9.2 % of the arithmetic**;
- measured time: `attn` is **34.0–34.9 %** of the serialized wave (§25's ruler rows), i.e. **≈260 ms** of
  763.49 ms;
- rate: 412.5 GFLOP / 0.260 s = **1.59 TFLOP/s = 7.1 % of the 22.3 TFLOP/s mix roof**. The GEMM families
  (D47) run at **97–102 % of that same roof**, so this is not "the machine is slow here", it is one family
  running **fourteen times** below the rate the machine demonstrably sustains elsewhere in the same wave;
- the external instrument corroborates the size: llama.cpp's whole attention block at the same width was
  **≤52.5 ms** (§22), which is **7.9 TFLOP/s ≈ 35 % of roof** — i.e. the parity target is a 5× on this
  family, and it is reachable by *efficiency*, not by a different algorithm.

**Why the prize is worth re-opening a route D42 closed.** At the roof this family could not exceed ~19 ms, so
the honest band is what a hand-tiled WMMA kernel can actually reach: llama parity (52.5 ms) takes the wave to
**≈556 ms = 7,370 tok/s**; a more cautious 20 % of roof (92 ms) still takes it to **≈595 ms = 6,884 tok/s**.
Both are larger than everything else on the queue combined, which is the whole reason 2.4 (WMMA attention)
gets re-opened despite its earlier "funded no" — that verdict was priced when `gdn_rec` was the big family and
when the queue believed attention was bandwidth-bound, and §25 changed both beliefs at once.

**Division, and what would falsify it.** Worker brief is the D53 attribution pattern, not a greenfield kernel:
name the shipping kernel and its instruction mix (does `attn_core`/`attn_tile_wave` issue *any* matrix
instruction, or is `1.59 TFLOP/s` an FMA-path rate? `tools/isa_probe.sh` answers this on the shipping object,
which is exactly how the GEMM census got corrected in D47's second bullet); isolate the same arithmetic at our
own geometry (512-row waves, 6 layers, the head geometry in `engine/kernels.h`); attribute the gap to traffic,
occupancy or instruction class with a falsifier per D49. **The falsifier that would kill this item quickly**:
if the shipping kernel already issues matrix instructions and the 1.59 TFLOP/s is what the tile shape can
sustain at 40 % lane coverage, then the fix is occupancy/tiling, not WMMA, and 3.3 (hand WMMA GEMM) stays
closed under D47's roof prior instead of reopening.

## D57 — attention attributed and re-priced: **4.6× against the right roof**, 2.4 stays no, 2.6 re-opens at +5–12 % of the wave (2026-10-17)

§26 is the evidence; this is the adjudication, and it starts with a correction to **D56**, which I wrote the
day before and which was wrong in its headline number.

- **D56's "fourteen times below the rate the machine sustains elsewhere" is retracted.** 1.59 TFLOP/s was
  compared against the 22.3 TFLOP/s *mix roof*, which is a matrix rate, while `tools/kernel_isa_census.py`
  shows the shipping `attn_slim_wave` issues **no matrix and no dot-class instruction at all** — nor does any of
  the **40 kernels** in the engine's own code objects. Against the measured sustained scalar `v_fma_f32` rate
  (**7.30 TFLOP/s**) the gap is **4.6×**. The size of the prize did not change; the premise that priced it did,
  and this project has now been bitten twice by comparing a kernel against a ceiling its instruction class
  cannot occupy.
- **Corollary that cuts the other way and is the reason 2.6 stays open:** llama.cpp's **≥7.9 TFLOP/s is above
  the best scalar rate this box has ever measured**, so parity cannot be reached on any VALU path. The
  comparison is real evidence about the *class* — and only about the class: the matrix instruction subsumes
  ~27 of a touch's 45 pass-1 slots, a **2.4× cut of pass 1**, not a 5× cut of `attn`.
- **2.4 (WMMA score inside the current geometry): funded no, and now for a structural reason.** `ATTN_TILE_M ==
  1` is `static_assert`ed, so a 16×16 WMMA operand cannot be expressed without re-tiling; and granting the
  warp-fold term D30 excluded (the 11 slots/touch that a matrix tile genuinely absorbs) still lands inside the
  kill band after the measured geometry tax. D56 did not change `attn`'s 260 ms — only its share and the rate
  framing — so none of D30's inputs moved.
- **3.3 (hand WMMA GEMM): stays closed under D47, strengthened.** The census proves the engine's own objects
  contain no matrix instruction, so 3.3 is not "use the instruction we already emit elsewhere"; it is writing a
  GEMM from scratch to approach a rate hipBLASLt demonstrably sustains at 97–102 % of roof.
- **2.6 (re-tiled attention at M ≥ 16): re-opened, priced at +5 % to +12 % of the wave, gated on a compile-only
  falsifier.** Bottom-up: dot (16 slots) + fold (11) replaced by a measured-13.5-slots-per-tile feed takes a
  pass-1 touch 45 → ≈19 slots. Whether that is real depends on fragment redistribution inside the *engine's*
  block, which no probe has ever measured — so the gate is **`wmma_feeder_probe` extended with an M = 16
  score+fold+store variant, censused: ≤ 19 slots/touch → fund it; ≥ 30 → D30's verdict stands.** No device
  time is needed to answer it, and bit-identity is forfeited on this route (PV's P operand must become bf16),
  so a floor probe and section (j) history come with it if it is funded.
- **The denominator collision §2.4.2 settled favourably.** W2's calibrated counter read `attn_slim_wave` at
  **2,265.7 M VALU/pass** against model A's 2,207 M (whole-kernel) and model B's 705 M (pass-1 only) — model A
  by 3.2× — so **pass 1 is ~61 % of the kernel's VALU** and every remedy above keeps its upper price. One
  instrument lesson came free: the leg's own parser declared "no rows, uncalibrated" on a counter set that had
  collected correctly, because it summed without deduping by `(Dispatch_Id, Counter_Name)` and did not know
  `--time-waves` replays the wave three times. That trap is in `AGENTS.md`.
- **New item, from the same census: R6.** `attn_prep` + `attn_combine` are **90 of the family's 138 dispatches**
  and had never been censused; the counter says **3.4 % of the family's VALU**, which is small, but instructions
  are not wall and their launch/staging shape is unmeasured. Filed as cheap-to-check, not funded.
- **Ledger corrections made in place:** §22's "attention launches / pass 48 vs 48, identical" was inference
  (`6 × 8`) — counted, it is **ours 138 per pass against their 48**, because our attention is three kernels and
  theirs one fused kernel; the *conclusion* (not a launch-count item, 0.15 ms of submission difference)
  survives, the row did not. And `prototype/attn_valu_slim`'s "occupancy unchanged at 16 waves/SIMD" cited a
  clang remark that prints 16 for every kernel; the supported figure is **5 blocks / 20 wavefronts per CU
  (62.5 %)**, from KFD LDS 64 KiB, which is also the first LDS-per-CU number this repo ever recorded.

## D58. The M = 16 feeder falsifier clears, the census's occupancy claim does not, and 2.6 turns on a tax measurement rather than a probe

**Question.** D57 funded item 2.6's *gate* but not the route: re-tile the score half at `M >= 16` and put
the dot and fold on `v_wmma_f32_16x16x16_bf16`, fund it only if the feeder plus fold-return plus store
stays inside ~19 static slots per `(row, lane, head)` touch. `>= 30` and D30's funded-no stands.

**What the census says** (worker `b1a13913`, `prototype/attn_wmma/m16_census.output.txt`, compile-only;
its calibrations re-derives the three published figures — 45 shipped pass-1 slots, 116 pass-1 static,
13.5 feeder slots per tile — and passes, so the slot currency is the shipped one):

| spelling | slots/touch | what it costs |
|---|---|---|
| `m16_bf16gq` — Q from global, K staged, **rolled** K-step loop | **1.98** | 30 VGPR, 11,264 B LDS |
| `m16_roll` — Q+K staged, rolled | 2.10 | 19,456 B LDS |
| `m16_bf16rt` — staged, unrolled | 1.73 | 19,456 B LDS |
| **the same M = 16 geometry with the shipped VALU dot + fold, no WMMA** | **59.86** | — |
| shipped `attn_slim_wave`, pass-1 leg | **45** | 47 VGPR, 12,288 B LDS |

Two findings, and the second is the one that changes the argument:

1. **The feeder is not the wall** — `<= 19` clears by an order of magnitude: feed 0.82 + fold/return 0.90
   per touch against a 45-slot leg it replaces.
2. **The M = 16 re-tile is a debit the matrix form has to pay for.** The control — same geometry, shipped
   VALU, no WMMA — is **59.86**, *worse* than the shipped 45, because at M = 16 no per-touch parallelism
   is left to exploit. So "tile it and the matrix comes for free" is false, and the +5–12 % D57 booked is
   a net of a real cost, not a gross win.
3. D57's assumed **18-slot per-touch residue does not survive** the tile: at M = 16 the WMMA loop runs 16
   iterations per **256** touches, so an 18/touch residue would bill ~4,600 slots against a measured
   round of 507. The residue is ~0.2–0.4/touch of rolled-loop bookkeeping. That makes the score-half win
   bigger than D57 booked, and it is the reason this is a *measurement* question now, not a probe question.

**Where the census was wrong, and the device corrected it.** Its blocks/CU column was a host-side policy
over `.s` metadata, and its verdict rested on it — `m16_bf16rt` at 235 VGPR was booked as **1 block/CU**.
I ran the real query (`tools/kernel_occupancy_probe.cpp` now carries the five m16 rows; the two-TU link
the worker built but was not permitted to exec), under the lock, `util` recorded, canary asserted in the
same window (`prototype/attn_wmma/evidence/occupancy-m16.txt`):

| kernel | census predicted | **driver says** |
|---|---|---|
| `m16_bf16gq` | 5 blocks / 20 wf | **5 blocks / 20 wf = shipped parity** |
| `m16_roll` | 3 / 12 | 3 / 12 |
| `m16_bf16rt` | **1 / 4** | **3 / 12** |
| shipped `attn_slim_wave` | 5 / 20 | 5 / 20 |

So the parity-preserving spelling is also the cheapest one (global-A, rolled) — no trade-off between the
two currencies, which is the best possible outcome for a gate. But the static prediction was 3× wrong on
one variant, which is the same class of error as `attn_valu_slim`'s LDS booking (D44) and the reason this
repo treats a metadata-derived occupancy as a prediction: the canary is not decorative, and the driver's
number is the one to quote.

**The whole-kernel bill re-prices the prize, and it is smaller than the gate implies.** W2 established the
whole-kernel figure on device: `attn_slim_wave` is **89.9 warp-inst per touch**, not 45 — 45 was the
pass-1 *leg*. Replacing that leg with a 1.98-slot WMMA leg removes ~48 % of the kernel's VALU, and at the
measured slot→time elasticity of `slots^0.39–0.44` (D21/D22) that is **−24 % of `attn`**, i.e. ~13 % of
`attn` wall is the *optimistic bound* before any geometry tax, ~4.6 % of the wave, ~+4.8 % prefill. D30's
kill line for a re-tile is "> ~15 % of `attn`". **The score half alone lands on the line, not above it.**
And clearing it requires both halves: pass 2 (`P·V`) is the other ~50 % of the bill, and a block holding
both halves needs 16 f32 accumulator fragments over `DH = 256` = **128 VGPR/lane against today's 47** —
which is a block redesign, not a spelling change, and is also where the bit-identity forfeit lands (`P`
must become bf16, so a new band and a gate section come with it).

**Decision.**
- **The feeder falsifier's answer is recorded: the feeder does NOT block.** Anyone re-opening 2.6 must
  argue past the two debits (the re-tile costs +15 slots/touch on VALU; pass 2 needs 128 VGPR/lane),
  not past "the feeder is too expensive", which is now disproved.
- **2.6 stays funded-no at the score-half level** — ~13 % of `attn` at the optimistic elasticity bound,
  against a 15 % bar, *before* the geometry tax. Not closed on taste: on a number, against a line the
  repo set itself.
- **The deciding unknown is now the geometry tax at M = 16, and it is a device measurement, not a
  probe**: `prototype/attn_m2` measured the M = 2 tax at **+7 %** on device and modelled **+15–29 %** at
  the funded ruler. Re-tile at M = 16 **with the shipped VALU dot, no WMMA**, measure the tax, and the
  arithmetic closes in one direction or the other: a tax near the measured +7 % funds the WMMA port;
  a tax near the modelled +29 % kills it without any further analysis.
- **R6 stays closed as predicted** — `attn_prep` 3.7 % + `attn_combine` 1.2 % = 4.9 % of family wall
  against a 7.5–15 % threshold. (Its census's "36.3 slots/touch-equivalent" for `attn_prep` is *not*
  comparable to slim's 89.9: the two kernels normalise on different touch counts. The time shares are the
  usable figures.)

## D59. Item 2.6 is closed on measurement **[CONCLUSION OVERTURNED BY D61: the +112 % was timed on the one shape the census had already called unoccupiable, and the register attribution is refuted in §28 — the item is re-opened as unmeasured]**: every ingredient of the WMMA re-tile has now been tested, and each one fails

D58 left one unknown: the geometry tax of a 16-row re-tile, as a device number. It is measured, in
`bench/performance-status.md` §27, paired inside one rep against the kernel the engine ships (`ndiff=0`
on both sides), 40 launches × 5 reps, blocking, zero subagents, admission green first (40/40).

**The re-tile demands 0.28–0.30 of the bytes and runs 2.1× to 14.3× slower** in every fixture the harness
builds — best case **+112 %** at the widest grid, worst **+1,326 %**. Not a marginal result to be argued
down: three independent reasons, all measured rather than modelled.

1. **The traffic win buys nothing here.** D57's instruction cut made `attn` **1.521× faster with the byte
   census unchanged** — the kernel is not traffic-bound at these widths, and a re-tile's only lever is
   demanded bytes.
2. **The cost is registers, and it is coupled to the win.** Full-body VALU re-tile: **134 VGPR / 12,800 B**
   vs shipped **47 / 12,288** → LDS allows 5 blocks/CU, registers allow **3** (12 wavefronts against 20).
   `GRP=8` is refused by the accumulator `static_assert`; `GRP=2` restores parity only by giving up the
   re-use. Re-use and occupancy are one decision in this spell, not two.
3. **The WMMA block would pay more, not less.** D58's 30-VGPR `m16_bf16gq` is a score-tile probe with no
   softmax/rescale state and no `P·V`. The real block must also hold the `P·V` fragments over `DH = 256`,
   **128 VGPR/lane on its own** — so 134 VGPR is an *under*-estimate of the WMMA block's bill, and the
   cheapest form of the geometry already costs +112 % against a route whose entire prize was −15 % to
   −17 % of `attn`.

**Decision: 2.6 closed, not deferred.** D57 re-opened it on the argument that the feeder might be the wall;
D58 disproved that; this disproves the geometry. With §22 (launch count, equal launches and ≥5× slower) and
D46/D30 (tiled GEMM kill band) the route has no untested ingredient left, so there is nothing to fund and
no band to negotiate. Re-open condition, so this is falsifiable rather than a veto: a **512-row** fixture
(the engine's real wave; this harness stops at 64) showing the geometry tax below **~1.15×**, which would
require the re-tile to be traffic-bound — already disproved once, and the burden is on anyone re-filing.

**Corrections carried in the same commit, so the record is not left with the optimistic half.**
- D57's "+5–12 % of `attn`" was priced on a *pass-1-only* 45-slot leg against a whole-kernel bill of
  **89.9 warp-inst/touch** measured in §26 W2; against the whole kernel the score half was worth
  **−15 % to −17 % at the measured elasticity**, and that was already the ceiling before any tax.
- The worker's report stated bit-identity "everywhere" and did not mention **`short-64`**
  (`64 rows @ lanes 64–127`, `split=1`, `full=0`), which returns `ndiff_aout_m16 = 131072` — every element
  differs. `burst_m16.sh` printed `NO-TIMING: admission is red` and refused to time it, so no contaminated
  number entered §27; the closure above does not rest on that case. Unresolved and on the M = 16 form;
  `ndiff == nrows × 2048` exactly reads as a changed work split at `split=1`, not a wrong value.
- Occupancy for `attn_m16` is quoted from the compiler's own resource remarks (VGPR/LDS), because
  `attn_m16.hip.cpp` carries the harness `main()` and cannot be linked into `kernel_occupancy_probe.cpp`
  for a driver-authority reading. The register-cap claim is a static reading, and this repo has already
  been bitten once by a metadata-derived occupancy (D58's `m16_bf16rt`: predicted 1 block/CU, driver 3).

## D60. The pointer-increment follow-up is closed on the census, and the "+156 slots/token" it was priced from never existed

D55's queue line carried "pointer-increment address spelling (−156 slots, bit-neutral follow-up)". That
number is the flat ISA-census delta between `rec_fused_kernel` and `rec_loop_kernel`, and it is an
**accounting artifact**: a flat per-kernel slot count divides by nothing, so it bills the loop kernel's
**203 once-per-launch slots** (kernarg loads, the `xlshl.exec`-guard chain, the `tokens<=0` path) at ×1
per token. Split properly — `prototype/gdn_rec_isa/token_bill.py`, which uses LLVM's own loop annotations
so the flat totals stay comparable to `isa_census.py` — and the same listing says:

| kernel | flat slots/token-step | **true per-token** | address arithmetic |
|---|---|---|---|
| `rec_fused_kernel` (one token per launch) | 1,374 | 1,374 | 96 (7.0 %) |
| `rec_loop_kernel` (`--gdn-loop`) | 1,530 (**+156**) | **1,327 (−47)** | **93 (6.1 %, −3)** |

I re-derived this on a listing built by hand from `engine/kernels_gdn.cpp` at `211eb95`, independently of
the worker's run: same 1,374 / 1,530 / 1,327 / 93, `must-not-catch = 0`. So two claims collapse:

- **The loop did not add instruction cost at all** — it removed 47 slots/token while its wall-time win
  came from submission count (§24's 73,728 → 144 launches). The "+156" was the loop's prologue being
  charged once per token.
- **Address arithmetic is 93 slots/token, not ~156**, and its delta between the two forms is **−3**. Two
  thirds of it (80 slots) is state-walk addressing already strength-reduced and identical in every form.
  The ceiling for anything the spelling could remove is therefore **6.1 % of the instruction bill**, not
  10 % of the wave.

**The MODE, and why it was not measured.** `recptr_kernel` (six row pointers hoisted before the loop and
advanced at the bottom of each iteration) was written, gated for transcription (`check_recptr_transcription.py`:
7 must-catch mutations all caught, a cosmetic step reorder must-not-catch does not fire — its own selftest),
and censused: **flat 1,526, per-token 1,336 (+9, the wrong way)**, VGPR 75 → 69, SGPR 40 → 44, LDS and
spill unchanged, residency unchanged. At §24's elasticity, −4 slots/token is **≈0.17 pp of the wave**; the
measured +9 is a small loss. **No device window was spent on it.** That is a deliberate call, not an
oversight: 0.17 pp is below this box's resolution for a dispatch-bound row (best-of-N spreads of 5–10 % in
`bench/performance-status.md` §6), so a burst could only produce a number that cannot be interpreted —
which is the same reason D30's null was re-taken rather than trusted.

**A category error of mine, recorded where it can be seen.** My brief told this worker "the census already
prints a per-token line (`cast=… other=… total=…`); the shipped looped form reads `cast=97 other=20
total=406`". That triple is **`LaunchCounts::describe()`'s launch-plan bill — launch records per decode
step, classified cast/non-cast** — which is exactly the right canary for a *launch-count* change (§13's
1.7′ A/B used it correctly and still does), and is **not** an instruction census and cannot be reconciled
with one. Sizing an instruction prize from it is a category error, and it is the reason the item was
over-priced by 4×. Two instruments, one name: **launch-plan bill** (records/decode-step) vs **ISA slot
census** (slots/token-step). Quote which one a number came from.

**Where the loop's real surplus lives, if anyone revisits**: not in operand addressing — in the **82-slot
per-launch `xlshl.exec`-guard chain** (which a span of 512 amortises to 0.16 slots/token today, so it is
already cheap) and the **80 slots of state-walk addressing** inside the walks. Both are structural, and
neither is bit-neutral to touch. Item closed; `recptr` stays in the prototype harness as the negative
witness, unadopted and unmeasured, with its transcription checker intact.

## D61. D59's closure of item 2.6 is overturned: the missing side of its own experiment refutes both of its load-bearing claims

Prompted by the user's objection — *"that is suspicious — we know this is possible, we have llama.cpp
implementation that does that"* — which was correct, and correct in a way that turned out not to require any
new analysis of llama.cpp at all, only the second half of our own experiment.

**What was wrong.** §27 timed `GRP=4` (134 VGPR → **3 blocks/CU**) and attributed the tax to register
pressure, then used +112 % to close the item. But `attn_m16_grp2` was already built from the same source at
`-DATTN_M16_GRP=2`: **96 VGPR, 10,496 B → 5 blocks/CU, shipped parity**. It had never been timed. It now
has been, in its own hold with admission in the same run (`bench/performance-status.md` §28, log
`prototype/attn_m2/evidence_burst_grp2.txt`):

- **Registers: refuted.** 3 → 5 blocks/CU (**+66 % wavefronts**) moves the time by **≤5 %** in every
  fixture. The kernel was never occupancy-limited, so the register file is not refusing anything.
- **Traffic: refuted.** GRP2 demands ~1.8× the bytes GRP4 does (ratio 0.50–0.55 vs 0.28–0.30) and costs the
  same — §26/D57's "demanded bytes are not DRAM bytes" for the third time from a third direction.
- **The actual mechanism, and it is not a tax in the sense D59 used.** The m16 wall **tracks lanes and
  ignores rows**: 64 rows @ 4096 and 16 rows @ 4096 cost the same 3.11 vs 3.05 ms, and 128 blocks vs 512
  blocks cost the same, while the shipped form's cost spans 15× across the same fixtures. Making each block
  *longer* while making them *fewer* is what `gdn_collapse` priced in cycles-per-dependency: a round walks a
  dependent dot → 5-step butterfly fold → rescale chain per live pair, 16 live pairs per warp, with nothing
  to interleave against. The shipped form is cheap because it is short and hides that latency behind 20
  wavefronts/CU.

**What survives of D59.** The millisecond figures are real measurements of the prototype on those fixtures,
and the re-tile is genuinely slower there. What does not survive is either explanation, and therefore neither
does the closure: **item 2.6 is re-opened as unmeasured.** The machine is not the obstacle and this decision
should be read as removing a false ceiling, not installing a real one — llama.cpp runs a 256×256-tiled
attention kernel on this same silicon (≤52.5 ms for the family where we spend 264.71 ms), so tiles are
obviously affordable; the difference is intra-round structure, not register capacity.

**The instrument that would actually decide it** — two halves that have never been in one measurement:
1. a **512-row × 4096-lane fixture** (the engine's real wave). The largest fixture here is 64 rows, and at 16
   rows the tile issued 128 blocks on a 40-CU part, so there was nothing to hide latency behind;
2. a tile whose **score comes from a matrix op**, which is precisely the term that deletes the dependent
   fold. D58 censused that form at **1.98 slots/touch**, §27 measured a tile with the VALU fold still in it,
   and the two runs never met.

**Process rules taken from this, both cheap and both violated here.**
- **When a harness can build two shapes, time both before attributing a mechanism.** The unbuilt half of an
  experiment is invisible to a confident explanation: `burst_m16.sh` hardwired `BIN`, so the parity build sat
  on disk while its occupancy was being blamed for a cost it does not have. Now `BIN=${BIN:-…}`.
- **Do not state a device's limits while a rival engine is demonstrating otherwise on it.** "This box's
  register file refuses it" was checkable against a kernel we had already traced, and failed. A claimed
  hardware ceiling should name the measurement that would falsify it *and* the rival that would embarrass it.

## D62. Item 2.6 funded at round scope: their fold is one op per row and ours is a five-step butterfly per score

D61 re-opened 2.6 as unmeasured; `progress/2026-10-18/llama-fattn-anatomy.md` (worker `a313bd01`) is the
read-the-rival step the user ordered first, and it settles what the lever is. **Funded.**

**What the rival actually runs**, census'd from the traced instantiation's own code object with the
`gfx1151`-in-ELF-flags canary: `v_wmma_f32_16x16x16_f16` × 192 + `v_wmma_f16_16x16x16_f16` × 192, **zero
`v_mfma`, zero `v_dot`** — so the class question D58 could only assume is now witnessed, and so is the
fragment map (`mma.cuh:183-215`). Their round is **3.29 warp-inst/touch raw, 7.16 with a WMMA billed at
its 32-clk pipe time**, against our measured **89.9**.

**The four separable reasons, only one of which is the matrix pipe** — and this is what my D59 closure got
backwards:

1. both 256-dim dots retire on tensor pipes (0.125 WMMA/score vs 16 VALU slots);
2. **the fold is 1 cross-lane op per row per round** (`ds_bpermute` × 111 total, reconciled source↔ISA as
   16 for the f32-C→half2-B exchange + 1 for the fold) **against our 5-step butterfly per score**;
3. **10 independent accumulator chains per warp** give the round its own ILP instead of borrowing it from
   occupancy;
4. one rescale per 64 KV rows instead of per live pair.

**They run at 8 wavefronts/CU against our 20, with 358 spilled VGPRs, and are still 5× faster.** That is
the external witness to §28's internal one. Occupancy was never the axis; the dependent chain is.

**Two repo beliefs retired on evidence.** (a) **LDS binds, not the register file**: the "512 lane-slots per
SIMD" in `kernel_occupancy_probe.cpp` is hard-coded arithmetic, the driver returns 5 blocks/CU for
`attn_slim_wave`, and `MaxRegistersPerMultiprocessor = 196,608` ⇒ ≈307 lane-regs/thread at our shape, so
**D30's "16 accumulators = 128 VGPR/lane is a wall" was never a wall at 128 threads** — D58's `m16_bf16gq`
was already LDS-bound at 30 VGPR. (b) The 5× is **not** the matrix roof: they run ~8.8 TFLOP/s, ~17 % of
the 52.63 this box sustains, and are mix-and-stall-limited too, just from a better shape.

**Scope, stated because it is wider than D57's bar.** D57 priced replacing the **dot** (+5–12 %). The
anatomy says replacing the **round** is what buys the 5×: through this repo's own measured elasticity
(α = 0.44 from the 116→45 datum) their round implies **+30 % of prefill**, capped at +39 % by their
observed ratio. That is a scope change, not a re-estimate, and it re-admits the address-aliasing risk
§20.1 of `prototype/attn_decode/results.md` records — the fund authorises the **round**, and the guard is
the falsifiable claim below, not a band.

**The fund, as §4.2 sized it.** One new form, `m16_fattn_score`, changing the M = 16 probe's round in
exactly three ways and nothing else: matrix score tile on the **transcribed** map; fold = 1 cross-lane op
per row per tile; ≥ 8 independent P·V accumulator chains. **Accumulators f32 first** — their half2 O is an
escape from a 256-VGPR ceiling that §3.4 shows does not bind us — so step 1 needs **no numerics decision**
beyond the 16-bit operands the standing rule already permits. Falsifiable claim, three parts: **(i)**
≥ 4 blocks/CU by the occupancy probe, **(ii)** ≤ 20 pipe-weighted slots/touch by census (D57's ≤ 19 bar),
both CPU-only, and **(iii)** on the wall at 4096 lanes, **cut the `attn` family by ≥ 1.5×** — the number
D57 already paid for a 2.4× instruction cut. If (i)+(ii) hold and (iii) fails, the matrix class and the
round shape are both exonerated and 2.6 closes on evidence, not on budget. Taken in the same window: a
**WMMA dependent-chain latency leg** in `prototype/issue_rate_probe` (currency B has a rate but no
latency), and the **512-row × 4096-lane fixture** — as the harness that admits and times, not as the
instrument that decides, since D61 showed the tile's wall is row-flat. Gate work in §4.4 is mandatory
before any ship: a section-(j)-style spelling A/B, a floor probe sized for the bf16→f16 conversion (floor
exactly 0) and the reordered dot (cancellation band, judged by witnesses + cos/meanrel, never by max-ULP),
and a must-catch set **derived** from the tile geometry — including the injection today's butterfly form
structurally cannot see, one lane-half of the C→B parity exchange.

## D63. The zero-spill precondition is re-scoped to a reported cost — the only evidence on this kernel class says spilling is survivable

The port shipped a placement that still spills (**256 VGPR / 49 spill / 72 B scratch / 16,384 B LDS / 4
blocks/CU**), against the rule I set two days ago: *no device minute until spill is zero*. That rule is now
re-scoped, and the honest reason is that it was never measured.

- It was my invention, adopted as hygiene, with no datum behind it on this box or this kernel class.
- The only measured fact we own about spill cost on **exactly this shape** is llama.cpp's traced kernel:
  **358 spilled VGPRs, 900 B/thread scratch, 8 wavefronts/CU — and ≥5× faster than us.** Our best placement
  spills **7× less** than the winning posture. A gate that forbids what the champion does is not a safety
  check, it is a guess with a stern tone.
- Spill is only bad insofar as scratch traffic lands on the dependent chain §28 identified as the mechanism.
  **The wall claim is the instrument that can adjudicate that**; a static spill count cannot.

So: spill becomes a **reported, annotated cost**, not a precondition. Unchanged and stricter than before: the
flag stays **default 0** in all three places, `fattn_ship_census.py` still **exits non-zero on any spill** so no
log may call this form gated, the burst **refuses a `SPILL` side silently** and now prints `wf_per_cu`. What
changes is that a burst may be taken with the spill annotated, and the default may only flip on a measured
≥1.5× win on the `attn` family at 4096 lanes.

**Three measured findings recorded with it, because each retires a claim I made.** (1)
`__launch_bounds__` is worth **191 spilled VGPRs** — without it LLVM budgets 192 registers/thread — so every
register figure in this effort must name the declaration it was compiled under, including mine. (2) The spill
is the **accumulator's live range**, not Q's: register-Q spilled *worse* (115 / 724 B) and `#pragma unroll 8`
trades 49 spills for 544 B of scratch. (3) At one wavefront per block the listing emits **no `s_barrier` at
all**, so the barrier cost I flagged as "unpriced" is priced at zero at this geometry — and the same geometry
means **4 wavefronts/CU**, which is *five times below* the 12–20 band §28 actually measured. That is stated in
the burst's own canary rather than smoothed over, and it is the strongest remaining reason to expect (iii) to
fail. If it does fail, the matrix class and the round shape are both exonerated and **2.6 closes on evidence**.

## D64. The accepted win is withdrawn before shipping: the WMMA form's covered rows are NaN, and the gate could not say so

Measured (worker `80316334`, `tmp/gcc/fattn_band_analysis.md`): rows 512..1023 of `wide1024f` are
`nan = 248320/248320` with the flag on and clean with it off (0 NaN, 0 argmax flips, rows 0..511 bit-identical
between sides). Decision log: **do not flip the default, do not merge.** Claim (iii)'s 1.3665× stands as a
*measurement of a kernel that does not compute the right answer*, which is not a number the ledger may carry as
a win; §29 carries the correction in place.

Why it got through my own burst: canary C was a *difference* witness (anti-stale-binary), and NaN differs from
finite, so it passed **because** of the defect. Difference-from-reference is not the same test as finite-and-
inside-a-band, and from now on any timing window whose canaries are all difference-based is documented as
unverified-for-validity rather than green. Three gate checks were also blind — two compare with `>` so a
non-finite passes a cap silently, one drops the flag on the b2 side, one calls non-finite equality "bit-identical"
— and all three are being fixed to fail loudly, because a check that cannot be false for its target failure is
decoration. Next: localise and fix the NaN, re-run admission section 2, then re-time claim (iii) and only then
re-open the flip decision.
