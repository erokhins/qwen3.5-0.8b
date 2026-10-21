# D4 — diagnosing the GDN decode walk

**Status: instrument built, nothing measured by this file.** This is a diagnosis package: the
provenance of both figures in the D4 route line, the bound classification with each bound ruled out
by arithmetic, a candidate list with the ONE arm that adjudicates each, a reproduction gate and
pre-registered thresholds that run **before** the matrix is paid for, and the rig
(`gdn_insitu.hip.cpp`) that runs the shipping GDN decode launches under one-variable arms.
**No millisecond figure in this file is measured here.** Every one is quoted and labelled, and the
rig prints no `ms` at all unless `--time` is present (that flag belongs to the main agent).

Route line, verbatim (`progress/2026-10-21/decode-attribution-2026-10-21.md` §5, D4 row):

> `0.40 ms of wall against a 0.17 ms floor; the state is 18.9 MB r + 18.9 MB w and reads warm at
> ~876 GB/s, so the gap is the per-token chain at 72 launches, not bytes. G4 reg16 shipped the wave
> form; the decode step is T = 1 through the same kernel` — prize −0.23 ms = −2 %;
> **gdn ≤ 0.50 ms WIN, gdn ≥ 0.60 ms KILL**.

## 0. The four things in that sentence do not share an instrument, a shape, or a cache tier

Every clause is separately traceable, and three of the four do not survive the tracing. Nothing here
loosens a gate or moves a floor.

### 0(i) `0.40 ms of wall` is a derived quantity, not a reading, and 0.17 ms is not a floor the family can reach

`0.40` is the attribution's §3 Correction-1 row for `gdn` (`progress/2026-10-21/decode-attribution-2026-10-21.md`):
stage-granularity **0.657 ms/step**, minus
`segments × 7.2 µs` = 36 × 7.2 µs = 0.259 ms, giving an *estimated share of the 11.46 ms burst wall*
of 0.40 ms, against a floor column of 0.170 ms. Both sides have a provenance problem, and they are
different problems.

**The 0.170 ms floor is a byte-rate floor and it is too low by construction.** It is
`bytes / 237 GB/s`, and it makes two errors that run the same way:

1. It prices a stream that is **47 % writes** at the pure-read rate. `tools/step_traffic_model.py`
   (run today, parsed by `census.py --traffic`) gives the `gdn` family **22.642 MB read + 20.425 MB
   written = 43.067 MB/token**, not the 40.3 MB that `40.5 MB` in the attribution's §2 table (line 72)
   implies — so even the raw denominator drifted: `43.067 / 237 = 0.1905 ms`, not 0.170. The tool's own
   no-cliff harmonic mean (`mix_nocliff_gbs`, reads 237 / writes 210) puts the family at
   **225.2 GB/s → 0.2005 ms**.
2. It ignores that **54 of the 72 launches cannot beat their own dispatch floor**. The measured
   empty-kernel dispatch is **1.85 µs/launch** (`devices/strix-halo-memory-bandwidth.md`, and graph
   capture buys 4 %, so it is the GPU's own dispatch rate). Per-launch bytes, from the shipped
   kernels and `engine/state.h` shapes (`census.py --launches`, which buckets the rows
   `step_traffic_model.py` prints rather than retyping a shape):

   | launch | count/step | MB/launch | bytes floor | floor = max(bytes, 1.85 µs) |
   |---|---|---|---|---|
   | `rec` (`rec_fused_kernel`) | 18 | 2.097 (1 MiB S read + 1 MiB written) | 9.278 µs | **9.278 µs → 167.0 µs** |
   | `conv` (`conv_kernel`) | 18 | 0.279 (conv w + window r + u, new_win w) | 1.233 µs | **1.85 µs → 33.3 µs** (dispatch) |
   | `gates` (`gates_kernel`) | 18 | 0.000 | — | **1.85 µs → 33.3 µs** (dispatch) |
   | `gated_norm` | 18 | 0.017 | 0.075 µs | **1.85 µs → 33.3 µs** (dispatch) |
   | **Σ** | 72 | 43.067 | | **0.2669 ms** |

   The ledger already says this about one of the four without drawing the conclusion:
   `engine/performance-status.md` §4 prices
   `gdn_gated_norm` at **0.4273 ms for 144 jobs = 2.967 µs/launch** and labels the row
   **launch-bound** — 42× its own byte floor.

   So the honest floor for this family at 72 launches is **0.267 ms**, not 0.170 ms, and the prize
   against the derived wall share (below) is **0.162 ms, not −0.23 ms**. Deleting launches is worth
   something on this family after all, which the `0.170 ms` floor hid.

**The `0.40` side is right but soft, and one window contains a better constant than 7.2 µs.** The
attribution's own evidence files carry the `gdn` row at *both* serialized granularities
(`progress/2026-10-21/evidence/decode-retake-2026-10-21/stage128_1.txt`, quoted):

```
STAGE gran=stage  family=gdn  ms_per_step=0.6323  launches_per_step=72  segments_per_step=36  us_per_launch=8.78
STAGE gran=kernel family=gdn  ms_per_step=0.8354  launches_per_step=72  segments_per_step=72  us_per_launch=11.60
```

Same launches, same lanes, same run: adding 36 more drains costs **+0.2031 ms**, i.e. **5.64 µs per
drain measured on this family**, not the cross-family average of 7.2 µs. Re-deriving the wall share
with the family's own constant: **0.6323 − 0.2031 = 0.4292 ms** (the ledger's 7.2 µs gives 0.398 ≈
the quoted 0.40, so the quoted figure reproduces with a softer constant). Gap to the byte floor:
**0.4292 − 0.1905 = 0.2387 ms** — so the *size* of the prize survives (quoted 0.23), and its
*denominator* does not (see the 0.267 ms floor above).

### 0(ii) WIN/KILL as written cannot adjudicate on either instrument, at either width

The lines are `gdn ≤ 0.50 ms` / `gdn ≥ 0.60 ms`, and `gdn` names nothing. The two candidates:

| reading | today's value (main agent, quoted) | verdict before any work |
|---|---|---|
| the **stage** row (the only thing one can read off a run) | **0.6297** @128 lanes, **0.6458** @4,032 | **already past KILL** (≥ 0.60) at both widths |
| the **wall share** (the quantity `0.40 ms of wall` names) | 0.6297 − 0.203 = **0.4266** @128, 0.6458 − 0.203 = **0.4427** @4,032 | **already past WIN** (≤ 0.50) at both widths |

So the line as written has no instrument. It is not a matter of taste which one is chosen, and the
prize says which: **−0.23 ms is a wall quantity** (it is the gap from the wall share to the floor), so
WIN/KILL belong to the wall instrument — and then they are vacuous, because 0.40 is already ≤ 0.50.
Choosing the stage row instead would make the route dead on arrival, and it also cannot be won by a
kernel: a stage row below 0.45 ms is unreachable while 0.20–0.26 ms of its own drain is inside it, so
on that instrument D4 is a *fusion* route (D5's territory), not a kernel route.

**Re-stated, explicitly, width included (proposal — re-cutting a pre-registered line is not mine).**

* Instrument: the **paired burst step wall** at **4,032 lanes** (`decode_run --time-stages`' burst
  pass, or `tools/context_sweep.py`'s differential), both arms inside one `tools/gpu_lock.sh` hold,
  with the `gdn` **stage** row from the same window as the attribution cross-check; the wall share is
  then reported as `stage − 36 × measured-drain`, with the drain constant re-measured in the window
  from the stage-vs-kernel pair rather than inherited at 7.2 µs.
* **WIN**: paired Δ(step wall) ≥ **0.08 ms** at 4,032 lanes (half of the re-derived 0.162 ms
  residue) *and* the `gdn` stage row down by a consistent amount.
* **KILL**: paired Δ ≤ **0.02 ms**, or the reproduction gate (§3) fails to synthesize ≥ 0.08 ms, or
  the only arm that wins is a launch-count change (then it is D5's, and it must be re-filed).
* `gdn` is context-flat (0.657 → 0.674 from 128 to 4,032 lanes, attribution §2 table), so the width choice moves
  this family by ~2 %; it is stated anyway, because the campaign's headline rule is 4,032 with
  shorter widths beside it.

### 0(iii) `reads warm at ~876 GB/s` is a demanded-byte, warm-tier figure, and in the step the state cannot be warm

`876 GB/s` is `prototype/gdn_decode/results.md` §Performance row 2 (quoted: **0.0646 ms** for the
18-layer state, "876 GB/s **cache-resident** = 370 % of the probe"). Three things about that number
have to travel with it:

* It counts **demanded** bytes, not DRAM bytes: 56.6 MB = the state read by pass A (18.87 MB),
  re-read by pass B (18.87 MB) and written (18.87 MB). `56.6 / 0.0646 ms = 876 GB/s`. The DRAM
  traffic is 37.75 MB; the second read is a cache hit by construction (a thread re-touches the same
  64 KiB column shard inside one launch). **Demanded ≠ DRAM** — the same error class `AGENTS.md`
  already records for attention ("a traffic claim is a property of the staging loops the source
  contains").
* It is a **warm** figure from a burst that re-reads the same 18 MiB 200 times with nothing else in
  the way, and the tier it is hitting is real but ~32 MiB (`devices/strix-halo-memory-bandwidth.md`:
  876.7 GB/s at 32 MiB, **239.4 at 64 MiB**; and the *copy* column at 32 MiB is only **325.7 GB/s**,
  because "writes pay a fetch/allocate cost the tier does not absorb").
* **In the step it cannot be warm.** Between two consecutive touches of layer *L*'s state the engine
  walks the whole token's **1,532 MB read** (`tools/step_traffic_model.py`). 1,532 MB ≫ 32 MiB, so
  every state byte in a decode step is a cold line. The consequence is a *floor correction, not a
  win*: the rec launches' honest in-situ floor is the DRAM-side one (9.278 µs/launch, §0(i)), which
  is **2.6× the quoted 0.0646 ms burst row** — and the burst row is the only recurrence floor the
  tree has ever printed. That gap is `T_cold`, a term in its own right, and §1's dose arm is what
  measures it instead of asserting it.

### 0(iv) `the decode step is T = 1 through the same kernel` is false, and the tree already knew

At a decode step `n = 1`, so `resolve_gdn_loop(sc.gdn_loop, 1)` returns 1
(`engine/kernels_gdn.cpp`), and `engine/decode_step.cpp` takes the `else` branch: **one
`gdn_rec` call per (layer, token)** → `rec_fused_kernel<<<NH=16, D=128>>>` (spell `fused`, the
default). `--gdn-geom reg16` reaches only the span branches (`gdn_rec_span_reg16`), which
**throw** on `tokens < 2`. G4 recorded this itself and handed it back rather than taking it
(`prototype/gdn_walk/G4-result.md` §4, §9, quoted):

> "the kernel itself **is** correct at `tokens == 1` … so the remedy is a one-line relaxation of the
> launcher guard, and that is a decision about decode routing rather than about G4 — handed back
> rather than taken"; and "**Decode.** … A one-row body never reaches this kernel by design, so the
> decode step is untouched — and untested for this knob."

So D4's mechanism clause ("the per-token chain at 72 launches") describes a defect whose known fix
was measured on the wave path and **structurally never reached decode**. That is the candidate the
route line does not list, and it is the one the rig prices first (§1 C4).

And the shape mismatch is worse than "same kernel, T = 1", which is D2's lesson repeating: the quoted
0.0646 ms burst is `prototype/gdn_decode/gdn_decode.hip.cpp --mode all`, which launches
`rec_kernel<<<L*NH, D>>>` = **one launch of 288 blocks for 18 layers**, while the shipping decode
walk is **18 launches of 16 blocks**. Two shape differences, same direction: 18× the launches, and
18× fewer blocks per launch (16 vs 288). The wavefront arithmetic is the tree's own
(`prototype/gdn_walk/BET.md:317`, quoted): shipped form **0.8 wavefronts/SIMD, 0.80 blocks/CU**,
llama's map **6.40 wf/SIMD, 8×, no idle CU** — with the caveat that BET computed the per-CU figure
against the HIP-reported 20-CU count, which `AGENTS.md` (2026-10-18) retires in favour of KFD
(40 CUs, `simd_per_cu 2`, `simd_count 80`); the **8× ratio is a ratio of block counts (128 vs 16) and
survives either reading**. Also on record and pointing the other way, which is why this is an
experiment and not a foregone conclusion: **G2 took wf/SIMD the other way (0.8 → 0.20–0.40) and lost
122 %** (`prototype/gdn_walk/BET.md` §11).

### 0(v) The residency question, answered from the code object, not a profiler (D3's lesson)

Parsed out of `engine/build/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s` by `census.py --codeobj`
(`.amdhsa_kernel` blocks — **not** `.vgpr_count`, per `AGENTS.md` 2026-10-20), with the shipped block
sizes read off `engine/kernels_gdn.cpp` and reproduced by the rig's `geom_check`:

| kernel | grid × block (decode, T=1) | `.amdhsa_next_free_vgpr` | LDS B | private seg | blocks offered per CU | blocks a CU *could* host (regs / LDS) |
|---|---|---|---|---|---|---|
| `rec_fused_kernel` | (16) × 128 | 59 → 64 | 1,280 | **0** | 0.40 (16 over 40 CU) | 16 / 51 |
| `rec_unroll4_kernel` | (16) × 128 | 59 → 64 | 1,152 | **0** | 0.40 | 16 / 56 |
| `rec_kernel` (the `ship` spelling) | (16) × 128 | 33 → 40 | 1,152 | **0** | 0.40 | 25 / 56 |
| `conv_kernel` | (6, tokens) × 256 | 42 → 48 | 0 | **0** | 0.15 | 10 / — |
| `gates_kernel` | (tokens) × 16 | 18 → 24 | 0 | **0** | 0.025 | ≥ 341 / — |
| `gdn_gated_norm` | (16·tokens) × 128 | 12 → 16 | 128 | **0** | 0.40 | 64 / 512 |
| `rec_reg16_kernel` | (16,8) × (32,4) | 76 → 80 | 0 | **0** | 3.20 (128 over 40 CU) | 12 / — |

Two rulings, both arithmetic:

* **No promote-alloca-to-LDS / no spill candidate exists.** `private_segment_fixed_size = 0` and
  `enable_private_segment = 0` for every GDN kernel, including `conv_kernel`, whose `float
  tap[K][CONV_VEC]` local array LLVM scalarised completely. So the D3-class finding (a resident-resource
  change) has no purchase here: there is nothing in scratch to promote and nothing spilling.
* **Residency is not the limit; supply is.** Register capacity allows ≥ 16 blocks/CU for
  `rec_fused_kernel` on either register-file reading in `prototype/attn_percost/a11_read_vgpr.py`'s
  header (per-SIMD 65,536 → 32 blocks; per-CU 65,536 → 16), and LDS allows 51, while the decode grid
  **offers 0.4 blocks per CU**. The machine could host ~40–100× the work the decode launch supplies.
  (Note `a11_read_vgpr.py`'s `blk/CU` column hardcodes 96 threads/block for the fattn kernel it was
  written for; the table above is recomputed by `census.py --codeobj` at each kernel's real block size,
  and it prints its own register-file assumption on every line. The tool is cited, not forked.)

That leaves **memory-level parallelism and dependent-chain latency at 32 wavefronts and 1 wavefront on
`gates`** as the only machine-side mechanism left standing, which is the route's own claim restated in
units that can be tested.

## 1. The candidates, the ONE arm each, and the reading that rules each out

Notation. `F` = the §0(i) per-launch floor **0.2669 ms** (census sum). `W` = the `gdn` wall share re-derived in the
window (`stage − 36 × measured-drain`), ≈ 0.43 ms quoted. `P = R_ship − R_repro` = what the rig
synthesizes: its own shipped-shape row minus its own shape-matched reproduction row. Shares
`s_x = T_x / P`; **CONFIRMED at s ≥ 0.25, REFUTED at s ≤ 0.10, PARTIAL between; the residue
1 − Σs is printed, never absorbed.** Every arm is a flag on one binary; no arm changes bytes.

| # | candidate | the ONE arm | ruled out if | confirmed if |
|---|---|---|---|---|
| **C1** | bytes are the binding bound | `--kernel rec` standalone vs its 9.278 µs/launch byte floor (`census.py --launches`), DRAM-forced by `--pollute-mib 1536` | the rec row reaches ≥ 80 % of the plateau with the state forced cold — then nothing is left to explain | it sits materially below the plateau, i.e. the family is *not* byte-bound (expected from §0(i): 0.43 ≫ 0.19) |
| **C2** | the state really is warm in situ (the route's `876 GB/s` clause) | `--pollute-mib {0,32,512,1536}`: a read-only interferer of that size, submitted and **drained outside the timed window**, so the row is the recurrence's own time as a function of the cache state it inherits. Four points make it a **slope**, not a difference | `R(1536) − R(0) ≤ 0.10 × P`: warmth is worth nothing in situ and §0(iii)'s DRAM floor is the wrong floor to compare to | `≥ 0.25 × P`, and then the prize is *not* the 0.23 ms against 0.170 but the 0.162 ms against 0.267 |
| **C3** | launch count and drain | two parts: `--kernel {all,rec,conv,gates,gnorm}` at fixed shape (superlinearity of 72 small launches), and `--drain {each,burst}` (the drain term in the rig's own shape). Cross-check: `--drain each − burst` must land in **0.15–0.26 ms** = the 36 × 5.64 µs measured in §0(i), *not* 36 × 7.2 µs | Σ(single-kernel rows) ≈ `--kernel all` **and** the three small kernels sum to ≤ their 3 × 18 × 1.85 µs dispatch floor ⇒ launch count is capped at ~0.10 ms of the residue | either the sum overshoots `all` (superlinear) or the drain term exceeds 0.25 × P |
| **C4** | the per-token chain at 32 wavefronts / 0.4 blocks per CU (**the route's mechanism, correctly located**) | the 2×2 that separates launch count from per-launch parallelism: `--shape {perlayer,combined} × --recmap {col,split8}` — 1 launch × 288 blocks (the shape the quoted burst took), 18 × 16 (shipped), 18 × 128 — identical bytes and identical per-thread work in every cell that exists; **the fourth cell (`combined × split8`) is refused, not skipped**: `rec_reg16_kernel` scopes the state base to `blockIdx.x` and spends `tokens` as a time axis, so it cannot address one block per (layer, head), and a rig that faked it would be measuring a geometry no kernel can execute | `split8 ≈ col` at fixed shape and fixed bytes (**< 10 % on the rec row**): the chain is not occupancy-limited, and D4's mechanism dies the way route A died | the split map closes ≥ 25 % of the rec row's distance to its byte floor. **Pre-declared caveat:** the split map is `rec_reg16_kernel`, which is *not* bit-identical (its norms and state dots re-associate — `engine/gdn_rec_reg16_kernels.h` pre-declares this and its band is the existing §10.1 aggregates cos ≥ 0.9995 / meanrel ≤ 5 %). A win there is a decode-routing decision (relaxing `resolve_gdn_loop`'s one-row rule) plus a `gate.py` admission with no band moved, **not** a D4 patch |
| **C5** | residency / spill / LDS (D3's candidate class) | **no device arm — the code object is the arm** (§0(v), `census.py`, reproducible with the existing `prototype/attn_percost/a11_read_vgpr.py`) | already ruled out: `private_segment = 0` everywhere, and capacity ≥ 40× supply | — |
| **C6** | the barriers inside the launch (5 `s_barrier`/token in `fused`, 11 in `ship`) at only 2 wavefronts per block, i.e. nothing to hide a barrier | `--spell {ship,unroll4,fused}`: three **shipped, mutually bit-identical** spellings, same grid, same operands, same addresses, differing only in barrier/prologue bookkeeping | the three rows agree within the window's spread (D50 already read a near-null: −2.276 % of *slots*, and `#pragma unroll 8` moved nothing in `gdn_collapse` §4) | they spread ≥ 25 % of P — then the fix is a barrier-free decode form, which is `--gdn-norm hoist`'s idea at T = 1 |
| **C7** | placement: 18 separate state blobs / a wide span (D2's B-region and B-span arms, kept because they are cheap) | `--layout {blob1, spanned} --gap-mib {8,42}`. **`--layout blob18` is refused, not skipped**: the engine allocates the state as **one allocation sliced per layer** (`engine/state.h:88 gdn_state` + `gdn_state_layer()`), so the shipped shape *is* `blob1` and a 18-blob arm would price a scatter the engine does not have; `spanned` inserts a real gap between the layer slabs, which is the only scatter the rig can build honestly | `spanned ≈ blob1` (expected: same bytes, same addresses inside each slab, and the recurrence computes its own `S + h*D*D` base so a gap between slabs is invisible to it) | any difference — then placement is live and D2's same-window `mixed_bw` size control comes back into this route |
| **C8** | state dtype (16-bit state halves the 37.75 MB) | **not this route's to run.** It is a pinned-operand change and the ledger already adjudicated it: "GDN state read+write — **not reducible**: it is the recurrence, and F32 is what the reference pins" (`devices/strix-halo-mixed-traffic.md:381`). The `AGENTS.md` delegated-format rule authorises 16-bit *arithmetic* behind green gates; a pinned **state dtype** is a numerics decision, not a format switch | n/a | n/a — recorded so nobody re-proposes it as a D4 sub-item |

**Term none of the three memory candidates is, and it is already measured.** The drain term
`36 segments × 5.64 µs = 0.203 ms` sits inside the stage row the route's line was read against. It is
not a memory mechanism and no kernel edit removes it; only fewer segments do, and the segment count is
2 per layer because a `copy` (the conv-window roll-back) splits the four GDN launches. Deleting that
one `copy` would collapse `gdn` to 18 segments per layer-pair and save ~0.10 ms of stage row —
**credited to `copy`, not to `gdn`**, and named here so it is not double-counted between D4 and D5.

## 2. What the rig is, and what is deliberately not in it

`prototype/gdn_insitu/`, prototype-only. `engine/` untouched, nothing in `progress/` touched, no gate
threshold or floor moved, no commit made, no millisecond measured.

| file | what it is |
|---|---|
| `gdn_insitu.hip.cpp` | the rig. It `#include`s `prototype/gdn_decode/gdn_decode_kernels.h`, `engine/gdn_rec_fused_kernels.h`, `engine/gdn_rec_unroll4_kernels.h`, `engine/gdn_rec_reg16_kernels.h` and `prototype/gdn_gated_norm/gdn_gated_norm_kernels.h` — **the same headers `engine/kernels_gdn.cpp` includes** — so there is no second kernel to drift. Every arm is a flag. |
| `census.py` | the census, parsed: the family byte totals are *consumed from* `tools/step_traffic_model.py` (bucketed by the note each row carries, never retyped), the §0(i) per-launch floor model comes out of those same rows, the residency read comes out of `prototype/attn_percost/a11_read_vgpr.py` plus one field it does not print (`private_segment`), and every rate is labelled DEMANDED or DRAM. Every parser asserts the record count it expected and dies on an empty match. |
| `check_d4.sh` | the ~20 s handback: reproduction + the drain cross-check, before the matrix is paid for. |
| `ab_d4_sweep.sh` | the main agent's window: one lock hold, blocking, canaries inside the window, empty-parse guards, `NO-TIMING:` on any red. |
| `build.sh` | `-O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib`. |

**Existing instruments, listed before adding one** (`AGENTS.md`: *before writing a census tool, list
the census tools*). `prototype/gdn_rec_isa/token_bill.py` owns the per-token/per-launch ISA split
(re-run on the current listing: `rec_fused_kernel` **1,374 slots/token-step per wavefront, 87,936 per
launch**; it is cited, not forked — and it is *not* used for `rec_reg16_kernel`, whose span loop is
fully unrolled so the tool's per-token split does not apply: `prototype/gdn_walk/BET.md` §12's 1.28× is
the figure there). `prototype/attn_percost/a11_read_vgpr.py` owns the code-object register/LDS read (§0(v)).
`tools/step_traffic_model.py` owns the family byte census. `prototype/gdn_decode/check.py` owns this
family's numerics gates, and `engine/gate.py` section (j) owns the shipped-path bit-identity claims;
**this rig re-runs neither and re-bands nothing.** What the rig adds that none of them can: the four
shipped GDN launches, in the shipped decode shapes, at T = 1, with the state cold, with per-launch
drain control, and with a controllable cache carry-over.

**Issue slots are ruled out arithmetically, and it is not close.** 87,936 slots per `rec` launch (32
wavefronts × 1,374). Even granting one issue slot per cycle per *wavefront* — 32 slots/cycle at the
32 wavefronts the launch actually supplies — that is 2,748 cycles ≈ **1.2 µs** at the measured
2.206 GHz (`devices/strix-halo-compute.md:138`), against a 9.278 µs byte floor and a
9.5–17.7 µs in-situ range. This is the same lesson `prototype/gdn_collapse/results.md` §5 already
billed: an instruction-count model over-predicted route A by **9.8×** because cost there is latency
per token at ~1.6 working wavefronts/CU, not issue slots.

### 2.1 The rig's witnesses, and their declared blindnesses

Verification is by **bytes touched**, in the `prototype/mlp_insitu/` discipline: every region is a
separate `hipMalloc` fenced by 64 words of poison on each side, a host snapshot is kept of what was
written *in*, and a verification pass compares device bytes against that snapshot rather than against
a re-derivation of the kernel's arithmetic. Three witnesses, chosen so that each covers the others'
blindness:

* **`--witness land` (did every byte arrive?)**. Every *destination* is pre-filled with a sentinel
  (`-1.2345e37f` for F32, `0xFFFF` for BF16) that no store in these kernels can emit, the operand
  region for the recurrence is set to `g = −20, beta = 1` so `gamma ≠ 1` and the state write-back
  cannot equal its input, and then: every element of the **derived must-land set** differs from the
  sentinel, every element *outside* it is **byte-for-byte what the snapshot says** (stray writes into a
  neighbour are caught, not merely writes into a sentinel field), and the guard bands still hold the
  poison. The `g = −20` choice is the load-bearing part: with `beta = 0, g = 0` a correct kernel leaves
  the state exactly as it found it, and a skipped state write would be invisible.
* **`--witness onehot` (did it land in the *right place*?)**. Tolerance-free and geometry-shaped: per
  head `h`, `q` is one-hot at `p_q(h) = (11h+5) mod 128` and `k` one-hot at a *different* column
  `p_k(h) = (7h+3) mod 128` — positions that vary with the head, so an aliased head shows — while
  `g = 0` and `beta = 0`. Then `gamma = expf(0) = 1` exactly and `dv = beta·(v − gamma·r) = 0` exactly,
  so **the state write-back must be bit-identical to its input**: no tolerance, no summation-order
  question, nothing to re-band, and it adjudicates `split8` (whose norms re-associate) exactly because
  it never looks at a sum. Separately, `o[j]` reduces to `c_q × S[p_q(h)][j]`, and the state is filled
  with signed **powers of two** whose exponent depends on both `i` and `j`, so `|o[j]| / |S[p_q(h)][j]|`
  must equal one constant across every `j`, head and layer — exactly, since scaling by a power of two
  is exact. Blindness: it cannot see a *skipped* state write (that is `land`'s job), which is stated in
  the code next to the check rather than discovered in review.
* **`--witness mirror` (is the operand map the one I think it is?)**. A double mirror of §04.4 for the
  per-layer recurrence, banded at `meanrel ≤ 1e-3` **as a geometry witness only** — the measured value
  on `--layers 3` was `2.2e-07`. It is not a numerics gate and re-bands nothing: `prototype/gdn_decode/check.py`
  F1/F2/F3 and `engine/gate.py` section (j) own those. Its value here is that it reads the operand
  regions the way the *kernel* must, so a wrong slice or a wrong state base shows up as a huge `meanrel`
  rather than as a plausible number: it caught two wrong indices in the witness itself (§5).

**The must-land set is DERIVED, from the same descriptors the launches are built from.** `plan_build()`
emits one descriptor per submission `{kernel, grid, block, dst ranges}` with the shipped populations
re-derived — 18 `conv` launches of `grid (6, tokens) × 256`, 18 `gates` of `(tokens) × 16`, one
recurrence launch per `(layer, token)` of `(16) × 128` (or the `reg16` map's `(16,8) × (32,4)`), and 18
`gated_norm` of `(GN_HEADS·tokens) × GN_D_V` — and the expectation is the union of those ranges.
`verify_run` refuses `FAIL_nothing_expected` when the set is empty, so a `--kernel` subset cannot pass
by expecting the whole family's outputs. `geom_check()` (`--selftest-geom`, host-only) then re-derives
the *index arithmetic* on the host from the same struct: the conv strip must end exactly on `CH`, the
col map must cover `NH·D·D − 1` relative to its own base, the `split8` map must satisfy the
`cols × rows` identity for its own grid, and each arm's grid/block must equal the shipped launch line
`engine/kernels_gdn.cpp` uses — a plan that drifted from the launcher fails there, not in a timing.

**Two kinds of negative, and both perturb the launch and never the expectation.**

* *Launch-level* (`NEG_GRID_SHORT_REC` drops a block, `NEG_STATE_OFF` shifts the state base by one
  element) — judged by the ordinary verify pass, with the case's own witness, because the witness is
  what has to notice. The expected set still comes from the **shipped** plan, so a short grid is caught
  rather than agreed with.
* *Kernel-level*, using the negative enums the shipped headers already carry (`GC_*`, `GG_*`, `GN_*`):
  run the arm clean, snapshot two buffers, run it injected, and compare. Each injection has a
  **derived three-way class per element** — `MUST_DIFF`, `MUST_SAME` (provably bit-identical: the
  injection cannot reach it), `DONT_CARE` — computed from the geometry: `k = t − (K−1) + j` for the
  conv window rows, `k = t − (K−1) + kConvActiveTap` for the conv tile, the token index for the two
  row-zero injections, and for `GC_TAP_STRIDE` an actual host comparison of the two indices
  `raw[k][ch]` vs `raw[ch·tokens + k]` on the fill's 0.1 grid. `MUST_SAME` is the half that makes a
  control mean something: `GC_WIN_LAG` must move the window and leave `u` byte-identical, and
  `GC_GNORM_ROW_ZERO` must move rows `t > 0` of `block_out` and leave row 0 identical.

**A derived set is only sound if the fixture can *prove* it.** The first version of this rig filled
`raw` and the conv window with flat, nearly-uniform values and three of the five kernel-level controls
were un-adjudicable: `win_lag` moved 5 of 18,432 window elements because the window rows were
indistinguishable, `zero_pad` looked blind because the expectation was booked on the wrong buffer, and
the row-zero injections appeared to move nothing. The fix was not to loosen the class but to put the
operands in **disjoint value bands** — carried window in [10, 11.5], `raw` on a 0.1 grid in [1.0, 1.9],
conv weight with exactly **one** non-zero tap (`kConvActiveTap = K−2`, so at token 0 it reads a carried
row and at token ≥ 1 a live one), `o` rows for `gated_norm` in `[1+2t, 2+2t]` — chosen so that
`MUST_DIFF` is arithmetic (a band gap of ≥ 0.09 in, ≥ 0.019 out, against a BF16 ULP of 0.0039 near 0.8)
instead of luck. That is the `partalias` / `must-catch-derived` rule one level over: **derive the set,
then check the fixture can prove the derivation.**

**The blind sets are derived and asserted, not remembered.** At `--tokens 1` both `GC_TAP_STRIDE` and
`GG_ROW_ZERO` have a *provably empty* must-catch set (the two indexings coincide at one row; the
row-zero read has only row 0 to collapse to), and `--selftest-geom` asserts exactly that —
`derived tap_stride must-catch: tokens1=0 tokens2=5529`, `gates row_zero: tokens1=0 tokens2=16` — and
the two device cases are scored `BLIND-OK` only if nothing moved at all. At `--tokens 2` (a *synthetic*
width in this rig, refused under `--time`) the same injections must be caught.

**Refusals are part of the geometry.** `--shape combined` with anything but `rec` (the combined cell has
no per-layer conv tile); `--shape combined --recmap split8` (`rec_reg16_kernel` scopes the state base
to `blockIdx.x` and spends `tokens` as a time axis, so it cannot address one block per (layer, head));
`--recmap split8 --tokens > 1` (above one row it is the wave path's looped span — a different code
path and a different claim); `--layout spanned --gap-mib 0`; `--witness onehot|mirror` with any
non-`rec` kernel in the window (they pin the recurrence's own operands, and `conv`/`gates` would
overwrite them); `--witness mirror --tokens > 1`; `--time --tokens > 1`; `--pages cold` (first touch of
fresh pages is ~7× slow — refused, not averaged). Every one is exercised in `--selftest-geom`.

**Mechanics.** Every device case runs in its own forked child (`mixed_bw`: one faulting kernel poisons
the context and every later case silently never ran), the parent counts parsed children against the
case list, and no HIP call sits on a path reachable before the fork. `hipMalloc` only, never
`hipMallocManaged`. **No `ms` and no GB/s without `--time`**; a window that submits no launch prints
`NO-TIMING … no_launch=1`, never `0.0001`. Any rate above 240 GB/s the rig prints is labelled
`rate_class=demanded-above-DRAM-plateau(L2-resident)` against `DRAM-plausible`. The interferer for C2 is
submitted **and drained outside the clock**, so the row is the kernel's own time as a function of the
cache state it inherits rather than a measurement of the evictor.

## 3. The reproduction gate and the pre-registered thresholds — checked BEFORE the matrix is paid for

`check_d4.sh` runs these in one `tools/gpu_lock.sh` hold, blocking, and stops at the first red with
`NO-DIAGNOSIS: <gate>` — because a matrix measured on an instrument that is not the ledger's is a table
of NO-RESULTs.

* **G-witness (first, because everything else inherits it).** `--selftest` on device: 26 cases — 15
  positives across the witness × shape × map × spelling × layout grid, 9 negatives that must be caught,
  2 that must be provably blind. The driver asserts `parsed == cases`, `caught == |kind 1|` and
  `blind == |kind 2|` **derived from the case list**, not hand-counted. Plus three host-side refusals
  checked in the same script: `--kernel none` must print `NO-TIMING`, and `--pages cold` and
  `--layout blob18` must exit non-zero.
* **G-repro (shape-matched, not loosened).** The rig's `--shape combined --spell ship --kernel rec`
  arm is **one launch of `layers·NH = 288` blocks**, warm, which is the shape *and* the spelling of the
  quoted row in `prototype/gdn_decode/results.md` (**0.0646 ms** for 18 layers). It must reproduce that
  within **±5 %**. (This gate is what caught the rig booking `combined` as 18 launches of 288 blocks —
  18× the shape it is supposed to reproduce; §5.)
* **G-synth (synthesize the penalty before decomposing it).** The shipped shape (`--kernel all`,
  18 launches of 16 blocks, state forced cold with `--pollute-mib 1536`) must cost ≥ **0.08 ms more**
  than G-repro. If it does not, there is no in-situ penalty at the shipped geometry to decompose, and
  the honest report is `NO-DIAGNOSIS` plus "re-cut the route against the step wall" — the D2 outcome,
  named in advance rather than discovered afterwards.
* **G-drain (an independent cross-check of a ledger constant).** `--drain each − --drain burst` on the
  shipped shape must land in **0.15–0.26 ms** = 36 × the 5.64 µs this family itself measured
  (stage-vs-kernel granularity, §0(i)), *not* 36 × the cross-family 7.2 µs. Outside that band, either
  the rig's drain is not the stage clock's drain or the attribution's per-segment constant is wrong for
  this family — worth more than the route, because the whole per-family prize table stands on it.
* **G-witness per row, in the matrix.** Every row carries `verify=ok`; `ab_d4_sweep.sh` refuses to
  print a millisecond figure for a row whose verify is not `ok` (it prints `NO-TIMING` and the row
  text), refuses a row whose `ms_median` is missing or non-numeric, and asserts
  `rows asked == rows parsed`. `--reps` is 5 in the sweep, so a bimodal row shows.
* **A behavioural canary inside the window**, not a binary hash (`engine/build.sh` is not
  bit-reproducible here, and a stale binary agrees with itself): the sweep asserts the shipped row
  reports `launches=72` and the reproduction row reports `launches=1`, because those two numbers *are*
  the shapes the route is about.

## 4. Handback

Both scripts are the main agent's (they take milliseconds). One lock hold each, blocking, nothing
backgrounded. Read `check_d4.sh`'s first red as the answer; do not run the matrix if it did not pass.

```bash
# 1) the ~20 s check: rig selftest + G-repro + G-synth + G-drain. Expect `CHECK-D4 PASS ...`.
prototype/gdn_insitu/check_d4.sh

# 2) only then, the arm matrix (C2 slope, C3 population+drain, C4 shape×map, C6 spellings, C7 layout).
#    Override the burst size if a row is too cheap: REPS=7 ITERS=60.
prototype/gdn_insitu/ab_d4_sweep.sh
```

A worker's own record of what it ran is §5; it ran no timed row. `prototype/gdn_gated_norm/run.py`,
`prototype/gdn_decode/run.py` and `prototype/glue/run.py` self-acquire `tools/gpu_lock.sh`, so neither
script above is ever wrapped inside another lock hold (AGENTS.md: a caller holding the lock and then
running a self-locking gate deadlocks it silently).

## 5. What has actually been run here — correctness only, **no timing taken**

Commands, wall, `tools/gpu_lock.sh util` at acquire, and the words **no timing taken**, per the worker
device rule. All four ran on one hold each, one at a time, blocking.

| command | wall | util at acquire | what it proves |
|---|---|---|---|
| `prototype/gdn_insitu/gdn_insitu --selftest-geom` | < 1 s | n/a (host only, no device) | 7 geometries re-derived from the launch struct, 8 refusals that must refuse, 2 derived blind sets proven empty at `tokens 1` and non-empty at `tokens 2`. `SELFTEST-GEOM ok fails=0`. The two parse-level refusals (`--pages cold`, `--layout blob18` → exit 2) are asserted by `check_d4.sh`, not here. |
| `prototype/gdn_insitu/gdn_insitu --selftest` | 3–4 s over three holds | 19 %, 20 %, 91 % (AGENTS.md: identical clean runs read 24–91 % on this box, so discount it) | `SELFTEST ok cases=26 parsed=26 children_bad=0 caught=9/9 blind=2/2` |
| `gdn_insitu --kernel rec --witness land --time --reps 1 --iters 1` and the same for `--shape combined --spell ship` | < 1 s | 89 % | the `--time` code path and `check_d4.sh`'s field parser agree on a live row: `verify=ok`, `launches=18` / `launches=1`, `ms_median` present and numeric. **The millisecond values were read by the parser and discarded here — no timing taken, and no number from this run may enter any timing doc.** |
| `prototype/gdn_insitu/census.py` | ~2 s | n/a (CPU-only) | family totals 22.642 MB read + 20.425 MB written = **43.067 MB/token** (47.4 % writes); floors 0.1905 ms at 237 GB/s, 0.2005 ms at the no-cliff mix; per-launch floor sum **0.2669 ms**; code object: `private_segment = 0` for all seven GDN kernels, `rec_fused_kernel` 59→64 VGPR / 1,280 B LDS / capacity 16 blocks per CU vs **0.40 offered**. |

Witness detail from the `--selftest` run: `pos_land_all` verified 4,747,296 elements; `pos_onehot_rec`
4,720,640; `pos_onehot_combined` 4,755,456 (the combined cell alone, which is 17 rows more than the
per-layer one); `pos_land_spanned` (gap 8 MiB) 4,720,640 — each with `first_bad=none`;
`pos_mirror_rec3` at `meanrel = 2.2e-07`; `neg_grid_short_col` caught as `unwritten=295040` first at
`state@245760` (i.e. exactly the last head's slab, which is the block that was dropped) and
`neg_grid_short_split8` as `unwritten=590080` at `state@112`; `neg_state_off` caught through the one-hot
ratio (`onehot_ratio_bad=2042`); the six paired conv/gates/gnorm controls caught with their complements
intact (`same = 6144 / 18432 / 18432 / 25191 / 32 / 2048` elements proven **not** to have moved).

**What the rig caught while it was being built** — the reason to believe its no-movement claims, and all
of it caught by a witness rather than by review:

1. `combined` was booked as **18 launches of 288 blocks** instead of one. The reproduction row would have
   been 18× the shape it exists to reproduce, and G-repro would have failed for the wrong reason.
2. The combined cell indexes `o` as `o + h·D` with `h` a **global** head index, so one launch writes
   `layers·kValueDim` floats, not one token row. The witness reported the other 17 rows as 34,816 stray
   writes against a kernel that was correct.
3. The mirror compared every head against **head 0's slot** (it omitted `h·D·D` from the state base). It
   printed `want=0.0435 got=0.0124` at element 128 and the numbers were self-diagnosing: head 1's
   expected value, head 0's address.
4. The mirror read 4 bytes into a `double` for `o`, so the top half was garbage (a denormal 5.1e-310
   where the answer should be 0.066) and `meanrel` came out exactly 1.00 — an aggregate that looks like
   a divergence and is a `memcpy` bug in the checker.
5. A flat operand fill made three kernel-level controls un-adjudicable (§2.1), which is the vacuous
   negative-control trap reached from the fixture side rather than the case-list side.
6. `differ_class` used `kValueDim` as the row width for `beta`, which is `block_out`'s width, so 16
   legitimate movements were counted as violations of the must-not-move set.
7. The host-only `--selftest-geom` segfaulted once a control needed the raw fill's *values*, because those
   lived in a vector only the device path filled. The fill is now one named function (`raw_val(t,c)`)
   shared by the fixture and the expectation, which is also the better spelling.
8. **`--layout blob18` was accepted and silently meant `blob1`.** The route asks for an 18-separate-blob
   arm (D2's B-region control), the enum had the value, the parser accepted the string, and the allocation
   code never looked at it — so a C7 row would have compared one allocation against itself while printing
   `layout=blob1` on both sides. It now refuses at parse time with the reason (the engine allocates the
   state as one allocation sliced per layer, `engine/state.h:gdn_state` + `gdn_state_layer()`), and
   `check_d4.sh` asserts that exit code. Same class as every other silent-default failure here: an option
   that cannot fail is not an option.

**Not run, deliberately:** no `ms` figure of any kind was recorded by this worker, so there is no D4 row
for the ledger; `engine/`, `progress/`, `devices/`, `engine/performance-status.md`, `README.md`, every
gate and every floor are untouched; nothing is committed. If `check_d4.sh` fails at G-synth, the outcome
is the D2 outcome and the route needs re-cutting against the step wall, not a matrix.

---

## 6. The gate window (2026-10-21, main agent): all four gates pass, after two of them were corrected against their own references

Taken by the main agent, blocking, zero subagents, one `tools/gpu_lock.sh` hold per script, util 82-85 %
at acquire, discounted per `AGENTS.md`. Command: `bash prototype/gdn_insitu/check_d4.sh`. Two gates
failed on the first pass and both failures were the gate, not the rig — and in both cases the fix was to
compare against the reference's own instrument, never to widen a band.

**G-repro (fixed by scale, not by tolerance).** At the worker's `--reps 3 --iters 40` the repro row read
best 0.0643 (correct, -0.5 %) but median 0.0741 (+14.7 %) and tripped the +-5 % gate. The quoted row is
`0.0646 ms best / 0.0650 worst, spread 0.5 %` from `gdn_decode --mode all --iters 200 --trials 5`
(`prototype/gdn_decode/results.md:23,57`) — a best-of-N at 5x the iteration scale. Re-run at that scale:
**median 0.0638 / best 0.0636 / worst 0.0638, spread 0.3 %, -1.2 % on the reference**. A launch-bound
sub-0.1 ms row is bimodal when under-sampled, which is exactly why the ledger discounts dispatch-bound
timings, so `REPS/ITERS` now match the reference and the gate additionally refuses a repro row whose
worst/best exceeds 1.03 — a bimodal instrument must fail here rather than poison every row downstream.

**G-drain (fixed by population, not by tolerance).** The band was written as 36 x 5.64 us, but
`gdn_insitu.hip.cpp:649` syncs once per **launch** and the shipped row launches **72**; the engine's stage
clock closes **36 segments** for this family (`STAGE gran=stage family=gdn launches_per_step=72
segments_per_step=36` from today's engine window). Measured across two windows: drain term 0.3827 and
0.4174 ms over 72 syncs = **5.32 and 5.80 us/sync**, both inside the quoted per-drain bounds. So the
attribution's per-segment constant survives and the *assumed population* did not; the band is now derived
from the row's printed `launches=` rather than assumed. The derived engine-comparable reading is the
number that matters downstream: **~0.19-0.21 ms of the 0.63 ms gdn family line is per-segment drain**, not
recurrence. `LC_ALL=C` was also set, because the drain term printed as `0,3827` under this box's locale.

| gate | reading | verdict |
|---|---|---|
| G-witness | `SELFTEST ok cases=26 parsed=26 children_bad=0 caught=9/9 blind=2/2` | PASS |
| G-repro | 0.0638 ms vs quoted 0.0646, spread 0.3 % | PASS (-1.2 %) |
| G-synth | shipped+cold 0.3459 vs repro 0.0638 = **+0.2821 ms** >= 0.08 | PASS |
| G-drain | 0.4174 ms / 72 = 5.80 us/sync, band 4.17-7.20 | PASS |

Two things the rig did NOT find, recorded as negatives. `census.py` reports `private_segment=0` on all
seven GDN kernels, so there is no promote-alloca-to-LDS reclaim available here — the D3 finding does not
repeat on this path. And the census's 43.067 MB/token is 47.4 % **writes**, so unlike the MLP walk this is
not a read-bound stream: any floor built on reads alone would have been half the traffic.
