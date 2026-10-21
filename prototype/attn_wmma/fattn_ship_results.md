# The fattn form at the SHIPPED placement — what the census says, and what it forbids

Every number here is a compile-time fact read out of `engine/kernels_attn.cpp`'s GCN listing for the kernel the
engine actually launches (`_ZN6engine7kernels15attn_fattn_waveE8CoreArgs`), reproduced by

```
python3 prototype/attn_wmma/fattn_ship_census.py            # compile-only, no device
```

**No timing appears in this file.** Nothing here was run on a device by the worker who wrote it; the burst is
the main agent's, and per `AGENTS.md` a worker's device run may not enter a timing document.

## 0. The headline, before any of the arithmetic

**Claim (i) — "193 VGPR, 0 spills, 0 scratch, 4 blocks/CU" — is false on the shipping path and was false the
moment it was committed.** At `d7057c4`, the commit that landed the port, `attn_fattn_wave` compiled to

```
.vgpr_count 192   .vgpr_spill_count 242   .private_segment_fixed_size 332   .group_segment_fixed_size 16384
```

Against the probe's `193 / 0 / 0`. The port was never the probe. Nothing in the tree could notice, because no
gate read the note block of the shipping kernel — which is what §5's "put the census numbers at the real
placement before spending a burst" was for, and this file plus `fattn_ship_census.py` is that item done.

## 1. Four placements, all measured at the shipping placement

| placement | VGPR | spill | scratch B | LDS B | blocks/CU (LDS) | verdict |
|---|---|---|---|---|---|---|
| as committed at `d7057c4` (4 warps, one shared `s_q` **shared wrongly**) | 192 | 242 | 332 | 16,384 | 4 claimed | **wrong geometry** (§2) and spilling |
| (A) register-resident Q, 4 warps, V un-aliased | 264 | 115 | 724 | 16,384→24,576 not required | 4 | **rejected on the spill gate** |
| (B) one warp/block, Q staged, V aliased, **no `__launch_bounds__`** | 192 | 240 | 324 | 16,384 | 4 | spilling |
| **(B) one warp/block, Q staged, V aliased, `__launch_bounds__(32)`** | **256** | **49** | **72** | **16,384** | **4** | **shipped, gate NOT cleared** |

Two findings the probe could not produce:

- **`__launch_bounds__` is not decoration, it is 191 spilled VGPRs.** Without it LLVM budgets a thread
  196,608 / 1024 = 192 VGPRs (it prices occupancy against the *target's* maximum block size, not this kernel's
  32) and spills everything above the budget. That single line is most of the difference between the committed
  kernel and the shipped one, and the probe appeared clean partly because it carries its own
  `__launch_bounds__(THREADS)`. Any future claim about this form's registers must state the declaration it was
  compiled under.
- **The unroll pragmas move the scratch, they do not remove it.** Both shapes were compiled at the real
  placement:

  | P·V leg | VGPR | `.vgpr_spill_count` | `.private_segment_fixed_size` |
  |---|---|---|---|
  | `#pragma unroll` (full, 16 tiles) — **shipped** | 256 | 49 | 72 B |
  | `#pragma unroll 8` | 178 | 0 | **544 B** |

  The partial unroll looks better on the spill column and is worse on the one that matters: 544 B/thread of
  scratch is what the probe itself recorded when it rolled its scan legs ("rolling the scan legs costs 544 B
  of scratch", handover §4.2 change (2)), and it reappears here unchanged. Full unroll is shipped because 72 B
  is less traffic than 544 B; neither is zero, and the census fails on either. The lever that could reach zero
  is shortening the 128-register accumulator's live range (drain each half of DH as it completes, or the two
  half-DH passes of handover §5(e)), not retuning this pragma — and `fattn_ship_census.py` re-derives the
  `v_wmma` site count from the pragmas (17 sites at full unroll, 9 at unroll-8), so a retune cannot be silent.
- **The spill is not Q's.** Placement (A) moved Q out of LDS into registers, exactly as designed, and still
  spilled (115). The remaining pressure is the 16 F32 output tiles (128 VGPRs/lane) live across the aliased
  V-stage transpose, plus the score-leg operands. At `FATTN_THREADS=32` the per-thread architectural cap is
  256, and the kernel wants more than that: 49 VGPRs go to scratch, 72 B per thread, and they are on the round's
  dependent chain — the mechanism `progress/2026-10-21/performance-status-2026-10-21.md` §28 identifies as the whole reason a serial
  scan cannot be covered at low occupancy.

Per the gate this work was routed under (spill, not wavefronts): **non-zero spill means `--attn-fattn` stays
default 0** in `engine/decode_run.cpp`, `engine/decode_step.h` and `engine/gate.py`, and it does. The banner
prints `scratch=` and appends ` SPILL` so a burst cannot be read without seeing it, and
`fattn_ship_census.py` exits non-zero, so no log can call this form gated.

## 2. The geometry error the census made visible (fixed here, recorded for the review)

The committed port gave all four warps of a block **one 8 KiB `s_q`** while each warp owned a different query
head (`head0 = ATTN_TILE_G * kv_write + w`), so each warp wrote its own head's 16 rows into the same buffer and
then scored its head against a four-head interleaving of it. The per-warp Q requirement is 8 KiB **per warp**
(16 distinct (row, head) pairs × 256 dims × 2 B), not per block. Placement (B) side-steps the whole question by
making a block one warp, so one stage is exactly one warp's Q and `blockIdx.x` names the head. The probe could
not have caught this because it has no head axis at all: `wmma_fattn_score.cpp` takes one `q` for the block, and
`wmma_fattn_score_kernel.cu`'s only consumer of `out` is a DCE sink (`out[i][j] += j`, `sink[threadIdx.x] = s`),
which is also why §5's LDS figure describes a kernel that cannot be routed.

## 3. Barriers: the third barrier per round costs nothing here, and that is measured

`§5` listed "a third barrier per round, unpriced, because the shipped form fits two" as an open cost of aliasing
V over K. At `FATTN_THREADS = 32` a block **is** one wavefront, and the shipped listing contains **zero
`s_barrier` instructions** (39 `s_waitcnt lgkmcnt(0)` sites do the ordering a single wavefront actually needs:
a staged write must land before another lane reads it). So aliasing V over K is free of sync cost at this
geometry; its cost is the extra LDS round-trip for V, not a barrier. `fattn_ship_census.py` derives the expected
count from `FATTN_THREADS` (0 at a warp, `1 + 3 + 3` above it) and also fails if the LDS-wait sites fall below a
floor — deleting barriers must not be allowed to mean deleting the ordering.

## 4. What the shipping path actually has

- **Grid/wavefronts.** `dim3(N_Q, split, fattn_rtiles_of(rows))` × 32 threads = 8 × split × rtiles warps. LDS
  16,384 B on a 64 KiB part gives **4 blocks/CU = 4 wavefronts/CU**. §28's "occupancy did not matter" evidence
  spans **12→20** wavefronts/CU, so four is outside the range where that was shown: the covered-fraction banner
  prints `wf_per_cu=4` so a burst cannot read these rows against the shipped ~20 wavefronts without the
  annotation. This is the honest counterweight to claim (i)'s "4 blocks/CU", which is arithmetically right and
  strategically misquoted.
- **Derived instruction skeleton, listing matches.** `v_wmma` 17 sites (1 rolled score tile + 16 unrolled P·V
  tiles); 40 `ds_load_b128` against 39 derived from the roll/unroll pragmas plus the three stage fills; 20
  cross-lane sites, exactly the 10 warp min/max + 9 in the round (1 max fold + 8 C→B exchanges) + **the 1
  deferred rowsum combine** the committed port was missing. That last equality is the machine-readable half of
  defect D-4: one fewer butterfly and the census fails, because cosine cannot see a half denominator.
- **Occupancy is host-queried, never guessed.** `fattn_echo` calls `hipOccupancyMaxActiveBlocksPerMultiprocessor`
  at startup and prints `wf_per_cu=`. It queries the occupancy calculator; it does not measure a launch, and it
  is not a timing.

## 5. Retirements — these numbers are dead for the shipping path

| dead number | where it lived | why |
|---|---|---|
| `9.719` pipe-weighted slots/touch | `fattn_census.output.txt`, `fattn_score_handover.md` §4 | probe body: no head axis, no epilogue transpose, no rowsum combine. The shipping form's skeleton is the §4 row above; a slots/touch figure for it is a burst question, not a census one. |
| `lds=16384` as the probe's own achievement | `fattn_census.output.txt` §2 "LDS at the real placement", handover §5 | the probe's `--selftest-lds 16384` was an option; the shipping form reaches 16,384 by a different route (aliased V at one warp) and pays a transpose stage through it. Same bytes, different kernel. |
| "193 VGPR / 0 spills / 0 scratch" | handover §4 claim (i), §5c's banner canary | §0 above. The §5c canary now reads `form=… vgpr=… lds=… scratch=… vtiles=…` and must show `scratch=0`. |
| "third barrier per round, unpriced" | handover §5 | §3 above: 0 `s_barrier` at one warp per block. |

## 6. What is left before a burst is worth anyone's minute

1. **Clear the 49-VGPR spill** (the only gate that is red). The lever is to shorten the live range of the 16
   output tiles — drain each half of DH through the transpose as it completes, or two half-DH passes as the
   handover's §5(e) already sketches — not another Q placement, which §1 shows is not where the pressure is.
2. Only then flip any default, and only in one commit with `gate.py`'s `--attn-fattn` default (the §18 trap).
3. The admission sequence in `run_fattn_admission.sh` is written, not run. It has never been executed by
   anyone, and every count in it is asserted rather than assumed.
