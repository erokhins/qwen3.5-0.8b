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
