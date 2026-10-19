# G2 — the looped GDN recurrence as ONE wavefront per block

BET.md §10, pre-registered before any number existed. This file records what was built, what has been
proved on CPU and by a short correctness-only device gate, what the arithmetic says, and what the shipping
decision still needs. **It contains no millisecond figure taken by this worker**: every timing number
below is quoted from §40 of `bench/performance-status.md` and is the baseline the pre-registered lines were
written against, not a new measurement.

## 0. The bet, and the two lines

| pre-registered line (BET.md §10) | live | dead | §40 baseline |
|---|---|---|---|
| `gdn_rec` family row, median of 3, 4,096 lanes | ≤ 180 ms | > 230 ms | 266.60 ms |
| wave `wall_ms`, same window | ≤ 560 ms | > 607 ms | 626.75 ms |

`wall_ms` is the ship gate; the family row is the attribution. Both live bounds must clear to ship, either
dead line kills, and the middle is UNRESOLVED — reported as unresolved, not narrated (§40 of
`bench/performance-status.md` is that precedent: G1's −5.9 % on its own row was −2.2 % of the wave because
+6.06 ms landed in its neighbours — `gdn_gated_norm` +15.6 % and `glue` +8.8 %, for +32 KB/token/layer of
stage traffic).
The A/B in §7 adjudicates **both** lines and prints the externality (ms summed over every family other
than `gdn_rec`) so that failure mode is visible in the same window rather than discovered later.

## 1. What shipped: `--gdn-geom block|wf4|wf2`

`engine/decode_run.cpp` gained `--gdn-geom`, in the style of `--gdn-norm` (`1e92500`):

* **Default is `block`**, the shipped looped body. `engine/gdn_rec_loop_kernels.h` is **untouched** —
  `git diff --stat` shows zero lines there. The launch site branches on the geometry *before* the
  `--gdn-norm` branch (the `gdn_span > 1` span branch of `engine/decode_step.cpp`), so the shipped path is
  the same call it always was.
* **Parse-by-refusal** (`kernels_gdn.cpp::parse_gdn_geom`): an unknown name throws with all three names and
  what each means; there is no fallback to the default, because the flag names which side of an A/B a run
  is and a typo that silently became `block` would price the shipped walk twice.
* **Echoed in `NUMERICS`** as `gdn_geom=<name>`, beside `gdn_norm=` and `gdn_loop=`.
* **Refusals** (each names the reason, not just the pair): with `--gdn-loop 1` (a one-row body is
  `resolve_gdn_loop`'s to route; the decode pins were not pinned through this kernel), with
  `--gdn-scan collapse` (the collapse route owns the recurrence), with `--gdn-norm hoist` (the norms are
  already in registers here, so the flag would be inert and its echo a lie), and — in the launcher —
  `--gdn-spell` other than `fused` and `tokens < 2`. `--gdn-geom block` never reaches the new launcher
  either, so nothing can price the shipped form twice by routing into it.
* **Billed on the same `gdn_rec` plan row, one launch per span, either side.** `launch_stats` and
  `stage_clock`'s bill are unchanged, deliberately: at the A/B shape both sides submit
  `launches=144 plan_launches=144` (8 bodies × 18 GDN layers, cross-checked out-of-binary by
  `prototype/gdn_walk/plan_bill_check.py`). The consequence for the A/B is recorded in
  `engine/stage_clock.h`: unlike G1 (144 vs 288) the launch count cannot discriminate the sides here, so
  canary B discriminates on the `NUMERICS` echo and *asserts* launch-count equality as an invariant the
  geometry must not break.
* **Grid is derived in the launcher from the constant the kernel divides by** (`GDN_WF_LANES`,
  `kernels_gdn.cpp`), with `static_assert`s next to both — the §20 rule of
  `prototype/attn_decode/results.md` (one address expression shared by writer and reader), and the
  AGENTS.md 2026-10-19 row whose whole lesson is that a geometry fact living in two headers gets one half
  committed.

Shapes: `wf4` = `<<<16, 32>>>`, 4 state columns per lane — **the pre-registered side and the one the
verdict is taken on**. `wf2` = `<<<32, 32>>>`, 2 columns per lane — the split leg, reported as a
sensitivity, not as a second bet. Kernel: `engine/gdn_rec_wf_kernels.h::rec_wf_kernel<COLS>`.

### 1b. "The default did not move", measured rather than asserted

A HEAD worktree (no `--gdn-geom` at all) was built beside the working tree and all three binaries run on
the same 46-token prompt under one `gpu_lock` hold (TAG `g2-default-inert`, util 92 % at acquire, **2.5 s
wall for the three runs; correctness only, no timing taken**):

```
baseline(HEAD) vs new default        : 46/46 prefill logit rows byte-identical, 44/44 STATEFP lines identical
baseline(HEAD) vs new --gdn-geom block: 46/46 prefill logit rows byte-identical, 44/44 STATEFP lines identical
new default NUMERICS echo            : gdn_norm=inline gdn_geom=block
baseline NUMERICS contains gdn_geom  : 0 occurrences  (so the two binaries really are different trees)
```

That is the `--gdn-geom block` null of §7 taken in advance: the knob exists, says so on the echo, and moves
nothing. It was taken twice, same result. The built baseline is left in place at `/tmp/base-g2` so the A/B
script skips its 18 s rebuild; delete it and the script re-derives it from `BASE`.

The four parse-time refusals were re-checked the same way (`--gdn-geom bogus`,
`--gdn-geom wf4 --gdn-loop 1`, `--gdn-geom wf4 --gdn-scan collapse`, `--gdn-geom wf4 --gdn-norm hoist` —
each prints its own named refusal and exits before the model loads).

## 2. Constraint 1 — zero new staging bytes, answered in bytes

The reason the geometry is possible at all is a scalar-multiple identity: `k̂ = κ·k` and
`q̂ = Q_SCALE·ι·q` are scalar multiples of the *raw* rows, so with `rq_j = Σ_i k_i S_old[i][j]`,
`tq_j = Σ_i q_i S_old[i][j]` and `kq = Σ_i k_i q_i`,

    o_j = Q_SCALE·ι · ( γ·tq_j + κ·kq·dv_j ),      dv_j = β·(v_j − γ·rq_j)

and `kq` is a **per-head scalar**, not a per-column vector. So walk A reads the old state once and
accumulates *both* dots, and walk B's only remaining job is the update — read `S_old`, write `S_new`, with
nothing to broadcast — which is the only reason the shipped form stages a 128-element `q̂` row through LDS at
all. The norms and `kq` come from the `k` and `q` rows already in hand, reduced by a five-level
`__shfl_xor_sync` butterfly.

Traffic, per head-token, demanded:

| | state reads | state writes | new buffers |
|---|---|---|---|
| shipped `rec_loop_kernel` | 2 × 64 KiB (walk A, then walk B) | 1 × 64 KiB | — |
| `rec_wf_kernel<COLS>` | 2 × 64 KiB (both dots in walk A, update in walk B) | 1 × 64 KiB | **none** |

**192 KiB demanded either side, and no new buffer anywhere**: same operand tiles (`q|k|v`, `g`, `beta`,
`S`, `o`), same addresses, same submission count. The only genuinely new reads are `k[i]`/`q[i]` per walk
row inside walk A — 8 B per row per head, **≈16 KiB/token/layer**, re-reads of rows the kernel already
fetched for the norms, served from L1, and scalarised by LLVM into SGPRs (`s_load_b128`, two per four
rows) so they cost no VGPRs and never touch LDS. For contrast, G1's `--gdn-norm hoist` added
**+32 KiB/token/layer of new global traffic**, which is what ate its −5.9 % row win (§40 of
`bench/performance-status.md`). Constraint 1 is satisfied with **0 new staging bytes**, so the
stop-and-report clause was not needed.

## 3. The structure claim, checked rather than believed

`.venv/bin/python prototype/gdn_walk/walk_isa_diff.py --g2 --asm <listing>` (full capture:
`prototype/gdn_walk/evidence-g2-isa-2026-10-19.txt`, with the re-take commands in its header):

```
kernel            barrier  lds_data   perm  s_load  gld_any  gld_x4  gld_x2  gst_any  gst_x4  gst_x2   fma  rsq_exp
block(shipped)          5        19     20       4       37       0       0       17       0       0    47        3
wf4                     0         0     15       8       11      11       0        5       5       0    39        3
wf2                     0         0     15       8       19       2      17        9       0       9    42        3
RESULT walk_isa_diff --g2 PASS — the wavefront walk is barrier-free and LDS-free, and it still reduces across the warp by permute.
```

* **0 barriers, 0 LDS data accesses** in both wf forms, against 5 and 19 in the shipped form. The
  15 `ds_bpermute_b32` are the three butterflies (5 levels each); a permute moves values between lanes
  through the LDS *unit* without addressing one byte of LDS, which is why `group_segment_fixed_size`
  reads 0 while the permute count is positive. The census therefore splits the buckets and asserts both
  halves — `lds_data == 0` **and** `perm > 0` — because a check that lumps them is either false or
  vacuous depending on the regex. `--selftest` proves a permute is not counted as an LDS access and that
  an LDS read is.
* **The state traffic is still vectorised**: `global_load_b128` / `global_store_b128` for the 4-per-lane
  form, `b64` for the 2-per-lane form, while the shipped form issues 37 scalar `global_load_*` and zero
  vector accesses. RDNA3 spells these `b128`, not GCN's `dwordx4` — the first version of this check
  demanded `dwordx4`, printed a confident **FAIL** over a kernel that had vectorised, and is the reason
  the mnemonic pairs are in the tool and its selftest now (AGENTS.md's `ds_load_b128` row, one class over).
* The per-row `k`/`q` operands arrived as **uniform scalar loads** (`s_load_b128` into SGPRs) and feed
  `v_dual_fmac_f32`, which packs two FMAs per issue slot. `rsq_exp` = 3 per token in every form (two
  `v_rsq_f32_e32`, one `v_exp_f32_e32`), i.e. the norm work is present exactly once per token, not per row.
* Note blocks (`fattn_residency_read.py`, mangled names): vgpr **75 → 54**, spill **0 → 0**,
  `group_segment_fixed_size` **1,280 B → 0**, `private_segment_fixed_size` 0. No register spill, so no
  `--ack-spill` and no band argument to make.

## 4. Occupancy: what the grid realises, not what the resources allow

Device facts from KFD (`/sys/.../nodes/1/properties`: `lds_size_in_kb 64`, `simd_per_cu 2`,
`simd_count 80`) and `rocminfo` ("Compute Unit 40", "Max Waves Per CU 32", "Max Work-item Per CU 1024").
**Not** `hipGetDeviceProperties`, which reports 20 CUs and 64 wavefronts/CU on this SKU (AGENTS.md's
counter row), and **not** clang's `-Rpass-analysis` `waves/SIMD: 16`, which is a constant.

| side | grid | wavefronts | resource capacity | **realised** | idle machine |
|---|---|---|---|---|---|
| `block` (shipped) | 16 × 128 thr (4 wf) | 64 | 8 blocks/CU (threads+wavefronts bind; LDS would allow 51) | 0.80 wf/SIMD, 1.6 wf/CU | 16 of 80 SIMDs |
| `wf4` | 16 × 32 thr (1 wf) | 16 | 32 blocks/CU | 0.20 wf/SIMD, 0.4 wf/CU | 64 of 80 SIMDs |
| `wf2` | 32 × 32 thr (1 wf) | 32 | 32 blocks/CU | 0.40 wf/SIMD, 0.8 wf/CU | 48 of 80 SIMDs |

The grid binds in all three cases, not the resources — which is the whole pricing problem: `wf4` puts
**4× fewer** wavefronts on the device and `wf2` 2× fewer. One correction worth recording: §38 of
`bench/performance-status.md` records "both engines land at 0.80 blocks/CU, four CUs idle", which is HIP's
halved-CU arithmetic; on KFD's 40 CUs the shipped form is **0.40 blocks/CU, with 24 of the 40 CUs holding no
GDN block at all**. The per-SIMD figure (0.80 wf/SIMD) is unchanged, because 40 CUs × 2 SIMDs is the same
80 SIMDs.

The consequence, and it is the one thing this geometry buys nothing on: in **both** forms every busy SIMD
holds exactly one wavefront (64 wavefronts over 80 SIMDs shipped, 16 over 80 at `wf4`), so neither form has
a co-resident wavefront to interleave with while a lane waits on a dependent load. The shipped form pays
three barriers a token to keep the four wavefronts of one head in step; `wf4` deletes the hand-off and with
it the only traffic that was filling those SIMDs. That is the bet stated as a mechanism, and it is why the
answer can only come from a window, not from arithmetic.

## 5. Numerics — the CPU mirror, then the device gate

**Band, unchanged.** The existing aggregates of §10.1 of `specification/10-reference-dumps-and-gates.md`:
`cos ≥ 0.9995`, `meanrel ≤ 5 %`, plus a
cancellation band (`eps = 2⁻¹²` of the tensor max) on the max-relative figure, because a 128-term dot
re-associated moves elements that cancelled to ~1e-6 of the max by large ULP counts: the unband-limited
maxULP column below reaches 4e7 at state elements near zero and means nothing (AGENTS.md's F1–F4 row).
No threshold in `engine/gate.py` was touched, nothing was widened, `--ack-spill` was not used.

CPU mirror `prototype/gdn_walk/g2_mirror.py` (pinned `data/refs/p64` L00/L01 inputs plus random cases;
`--selftest` proves it can move). All five rows are the same case — **L01**, 46 tokens × 16 heads × 128:

| comparison | cos | meanrel | maxrel(band) |
|---|---|---|---|
| shipped **mirror** vs PINNED `o` | 0.999998589 | 1.411e-3 | 3.906e-3 |
| reorder-only floor (`partials=4`, same order, no re-association) | 1.000000000 | 4.775e-7 | 3.676e-5 |
| **geometry `cols=4` vs shipped mirror** | **1.000000000** | **1.176e-6** | 6.174e-5 |
| **geometry `cols=4` vs PINNED `o`** | **0.999998589** | **1.411e-3** | 3.906e-3 |
| geometry state vs shipped mirror | 1.000000000 | 7.816e-7 | 8.565e-5 |

The decisive row pair: the geometry's distance from the *pinned reference* is **the same 1.411e-3** as the
shipped spelling's own CPU mirror — i.e. that residual is `rsqrtf`/`expf`/contraction between this
mirror and torch, not the geometry, and the geometry's own move (1.2e-6 on `o`, 7.8e-7 on state at L01;
5.8e-7 on `o` at L00) sits at the reorder-only floor. `cols=2` is bit-identical to `cols=4` in the mirror
and on device, as predicted: the split changes which lane owns a column, not what that lane computes.

Device gate, run blocking under `tools/gpu_lock.sh` (TAG `g2-gate`), verbatim:

```
$ .venv/bin/python prototype/gdn_walk/gate_g2.py
runs: 6 (2 shapes x 3 sides), ids=46 tokens from /tmp/g2_ids.txt
gpu_lock util@acquire: gpu use: 88%
gpu_lock util@release: gpu use: 26%
wall for all device runs: 5.2 s — CORRECTNESS ONLY, no timing taken (no warm loop, no best-of-N; no ms figure here measures anything)

== shape --wave-rows 512 --gdn-loop 512: reference side `block`, 46 prefill logit rows
   | shipped row 45 vs PINNED logits: cos=0.999806723 meanrel=1.602e-01 rms_rel=1.973e-02
  side wf4: logits compared=46 differing=46 nonfinite=0
        vs shipped (report, not the gate): worst-row cos=0.999307515 meanrel=4.123e-01 maxrel(band)=1.122e+02
        vs PINNED logits, row 45 (THE GATE): cos=0.999803234 meanrel=1.496e-01 rms_rel=1.988e-02   [shipped cos=0.999806723 rms_rel=1.973e-02]
        STATEFP state: rows=18 identical=0 worst_rel=4.59e-03@layer16 conv: rows=18 identical=2 worst_rel=2.85e-03@layer8 kv: rows=6 identical=0 worst_rel=3.18e-04@layer23
        layer-0 state move 1.95e-08 (<= 1e-06: the geometry's direct effect, before the cascade)
  side wf2: ... identical to wf4 in every field ...
== shape --wave-rows 16 --gdn-loop 8: ...
   | shipped row 45 vs PINNED logits: cos=0.999689165 meanrel=2.014e-01 rms_rel=2.521e-02
  side wf4: logits compared=46 differing=46 nonfinite=0
        vs PINNED logits, row 45 (THE GATE): cos=0.999856754 meanrel=1.228e-01 rms_rel=1.693e-02   [shipped cos=0.999689165 rms_rel=2.521e-02]
        layer-0 state move 8.86e-08 (<= 1e-06: the geometry's direct effect, before the cascade)
wf4 vs wf2 bit-identical: True — expected: each COLUMN's arithmetic is identical at either split ...
G2 DEVICE GATE PASS — finite, in band against the pinned reference, layer-0 effect at re-association scale, layer-0 conv untouched.
```

Command, wall, util, and the literal "no timing taken" are all in that block; total wall for the whole
command including model load was **19.6 s**, of which the gate itself reports 5.2 s of device work.

What the gate adjudicates, and the one place it had to be rewritten: the first version applied the floors of
§10.1 of `specification/10-reference-dumps-and-gates.md` to **new-vs-shipped** and demanded byte-identity
for the conv and KV windows. Both were wrong — the
engine already sits ~2 % (RMS) from the pins, so a 0.9995 floor against the *shipped spelling* asks for a
tighter tolerance than the engine lives inside, and conv/KV legitimately move through the cascade. The
corrected gate measures against the **reference** (cos ≥ 0.9995 **and** not further from the pins than the
shipped side, ≤ 1.25× its `rms_rel`), attributes the cascade by layer (layer 0 has no upstream, so its
state move is the geometry's direct, un-amplified effect — measured **1.95e-08** and **8.86e-08**, i.e.
re-association scale; deeper layers inherit chaos up to 4.6e-3, gated at 1e-2 per A-18 of
`specification/ASSUMPTIONS.md`), and keeps layer-0 conv byte-identity as the address witness. Two honest
footnotes: at the ragged shape the wf side
lands *closer* to the pins than shipped (0.999857 vs 0.999689), which is chaos, not an improvement, and
must not be claimed as one; and `STATEFP` moved on every layer, which is the numerics-side witness that
the flag selected something (AGENTS.md's stale-binary null, caught on the arithmetic side).

## 6. The price, in cycles-per-dependency at realised occupancy

Per-token issue floors from the existing census tool (`prototype/gdn_rec_isa/token_bill.py`, no third
census tool written), on one listing:

| side | per-token slots **per wavefront** | wavefronts per head-token | slots per head-token | slots per **busy SIMD** per token |
|---|---|---|---|---|
| `block` | 1,327 (reproduces the number in §24 of `bench/performance-status.md`) | 4 | 5,308 | 1,327 |
| `wf4` | 2,767 | 1 | **2,767 (−48 %)** | 2,767 (**×2.09**) |
| `wf2` | 1,794 | 2 | 3,588 (−32 %) | 1,794 (×1.35) |

The machine-wide instruction demand really does drop by about half — one pass over the state for both
dots, `v_dual_fmac_f32` packing two FMAs per slot, no barrier, no LDS round trip. But wall time here is
set by what a **single** SIMD must issue per token-step, because the recurrence is serial in `t` and each
side keeps at most one wavefront per SIMD: `wf4` puts **2.09×** the per-token demand on a quarter as many
SIMDs. Clock-free, since the shipped side's own wall is 266.60 ms:

* to reach **≤180 ms**, `wf4` must sustain **3.1×** the shipped form's per-SIMD issue duty; `wf2` needs
  **2.0×**;
* to avoid the **>230 ms** dead line, `wf4` needs **2.4×**; `wf2` needs **1.6×**.

Against the shipped side's own 3.6 µs per (token, layer), 1,327 slots per busy SIMD is about a quarter of
the SIMD's issue capacity anywhere in this part's clock range (≈24 % at 1.5 GHz, ≈19 % at 1.9 GHz), so
`wf4` needs roughly **76 % duty** to be live and ~59 % to avoid dead — the same conclusion, divided by
whatever clock the window actually holds. That is the honest price and it looks demanding. The
counter-argument, which is why the bet is still worth the window: the shipped form's
3.6 µs per (token, layer) is several times its own dependency floor — the chain is two 128-long dependent
walks, on the order of 1,000 cycles — so its wall is stalls, not issue slots, and the largest stalls it
contains are the three barriers and the LDS round trip this form deletes. In a 32-thread block a barrier
costs nothing, because there is no sibling wavefront to wait for. AGENTS.md's GDN-collapse row is the
warning that this class of model has been wrong by up to 9.8× on this box, in the *pessimistic* direction,
and the §40 lesson (`bench/performance-status.md`) is that a slot model on a serial kernel is exactly the
thing not to trust. **The A/B decides; nothing in this file is a verdict.**

Externality expectation, for the same reason the A/B prints it: the geometry changes no other kernel, no
launch count, no staging bytes and no plan row, so the externality line should read ≈0. If it does not,
that is a finding about the window (clocks, dispatch) rather than about G2, and it should be read as such
before the `gdn_rec` row is quoted.

## 7. Handover — the timing window (main agent only)

```bash
bash prototype/gdn_walk/run_ab_g2.sh                    # REPS=3, 4,096 lanes, sides base/block/wf4/wf2
REPS=2 SIDES="wf4" bash prototype/gdn_walk/run_ab_g2.sh # ship leg only, drop the split
SUMMARIZE=1 bash prototype/gdn_walk/run_ab_g2.sh        # re-judge existing logs, touch no device
```

It self-acquires `tools/gpu_lock.sh` (TAG `g2-ab`) and runs only non-self-locking programs; do not nest it
inside another lock. **`BASE` defaults to `HEAD` on the assumption that the G2 change is still uncommitted
in the working tree, which is how this package was handed over. If it has been committed by the time the
window runs, pass the pre-G2 sha** (`BASE=785082d`, the tip this package was built on) or canary A will
correctly abort, since the baseline would then accept `--gdn-geom`. Before the hold it asserts, in order:

1. **Canary A** — the baseline binary (built from `BASE=HEAD` into `/tmp/base-g2`) **rejects**
   `--gdn-geom` with `unknown argument`, and the shipping binary accepts `--gdn-geom block`. A stale
   binary fails here, which is AGENTS.md's "a paired A/B against a stale binary reads as a clean null".
2. **Default-inert leg** — `base` vs `new --gdn-geom block`, byte-identical on every prefill logit row and
   every `STATEFP` line. If this moves, the knob leaked into the shipping path and no ms below means
   anything.
3. **Geometry-live leg** — each wf side's `STATEFP` must **differ** from base's; a re-association that
   moved nothing would be the shipped form timed twice.
4. **Mirror leg** — `g2_mirror.py` green (deterministic, CPU, no device).
   `gate_g2.py` is deliberately *not* called inside the window because it self-locks; re-run it outside.
   (Legs 1–2 were already taken once in advance, §1b; re-taking them in the same window as the ms is still
   what the script is for — the point is that the claim is re-proved beside the numbers, not carried.)

Inside the hold, per rep: **canary B** requires each side to echo `gdn_geom=<side>`, every side to show
`family=gdn_rec launches=144 plan_launches=144 counts_match=1` (equality is the invariant; the echo is
the discriminator), and `TOTAL … plan_match=1`. Then the summary prints per-family medians for all four
sides, the base-vs-block NULL, both pre-registered verdicts with LIVE / DEAD / UNRESOLVED, and the
externality sum.

The summary parser was tested against synthetic logs before handover, and that test found four real bugs
(a `med_of` that dropped the family argument, a decimal comma from the locale, an awk `\u` escape printed
literally, and a `med_of` defined after its first use). Re-take it with:

```bash
.venv/bin/python prototype/gdn_walk/mk_parsecheck.py \
  && SUMMARIZE=1 OUT=tmp/gdn_walk/parsecheck bash prototype/gdn_walk/run_ab_g2.sh
```

Expected: `VERDICT LIVE` for the side the fixture puts at 160 ms / 515 ms wall, and `VERDICT UNRESOLVED`
for the one at 213 ms / 568 ms. A verdict that cannot change colour is decoration.

Also green at handover, re-takeable in any order, none of them a device run: `bash engine/build.sh` (both
binaries), `speccheck/run_all.py --quick` (11 gates, 0 failed — the torch reference path, which nothing
here feeds), `prototype/gdn_walk/plan_bill_check.py`, `walk_isa_diff.py --selftest` and `--g2`,
`g2_mirror.py --selftest`.

## 8. What none of these checks can see

* **Compiler contraction and association.** A CPU mirror reproduces the *formula*, so it is blind to FMA
  contraction and to association — the `q̃` trap in AGENTS.md. Where this form claims bit-identity it is
  transcribed character for character; where it re-associates (norms, `o`) it claims a band, not bits.
* **`rsqrtf`/`expf` intrinsics** on device vs torch, and the engine-vs-pin gap that already exists
  (~2 % RMS at the last row). The mirror's 1.411e-3 residual is that gap, not the geometry's.
* **The wavefront-count penalty**, which is the whole risk of this bet and is answerable only at 4,096
  lanes in an exclusive window. The 46-token gate proves the geometry is correct and in band; it says
  nothing about whether 16 wavefronts can hold the machine's attention.
* **Window effects on ms**: cold pages (first pass over fresh pages ~35 GB/s), clock state, and the
  dispatch-bound bimodality of §6 of `bench/performance-status.md`. Quote the medians with the spread and
  the `CLK@start`/`CLK@end` lines; a single rep outside the spread is a discarded window, not a number. No
  ms figure in this file came from this worker.
* **Decode is untouched by construction and therefore untested by necessity**: `resolve_gdn_loop` still
  routes every one-row body to the per-token kernels, so a decode bit never reaches `rec_wf_kernel`.
  The pinned prompts are all one-pass, so no gate here exercises a multi-pass prompt for this kernel
  either — the ragged `16/8` shape exercises a span *tail*, which is the smaller thing.
* **Admission to the shipping path is still `engine/gate.py`'s own fixture run**, in the main agent's
  window, on the derived fixture population. Nothing in this file substitutes for it, and this worker
  changed no band, no threshold and no gate file other than adding the `--g2` mode to
  `walk_isa_diff.py`.

## 9. Files

* `engine/gdn_rec_wf_kernels.h` — new: `rec_wf_kernel<COLS>`, `GdnWfVec<4|2>`, `gdn_bfly_sum_f32`; header
  states the three named things that move, the zero-LDS validity argument, and this pricing question.
* `engine/kernels.h`, `engine/kernels_gdn.cpp` — `GdnGeom`, `parse_gdn_geom`, `gdn_geom_name`,
  `gdn_rec_span_wf` (+ refusals, static_asserts, derived grid).
* `engine/decode_step.{h,cpp}` — `StreamCfg::gdn_geom`, launch-site branch before the `--gdn-norm` branch.
* `engine/decode_run.cpp` — `--gdn-geom` parse/usage/long help, `NUMERICS … gdn_geom=`, four refusals.
* `engine/stage_clock.h` — comment recording why no plan-bill field is needed and what that does to canary B.
* `prototype/gdn_walk/g2_mirror.py`, `gate_g2.py`, `run_ab_g2.sh`, `mk_parsecheck.py`,
  `walk_isa_diff.py --g2`, `evidence-g2-isa-2026-10-19.txt`.

## 10. Traps this package found

* **A band measured against the shipped spelling is a different band.** The first `gate_g2.py` applied the
  floors of §10.1 of `specification/10-reference-dumps-and-gates.md` to new-vs-shipped and demanded
  byte-identity for cascade-fed windows; the engine is already 2 % from the pins, so it failed for reasons
  unrelated to the geometry. An aggregate floor has to name *which* reference it is a distance from.
  See §5.
* **A census whose mnemonics are GCN's FAILs an RDNA3 kernel that vectorised.** `global_load_b128`, not
  `global_load_dwordx4`; the false FAIL is the same class as the false PASS one class over (`ds_load_b128`).
  Keep the mnemonic pairs in the tool, and let the selftest use RDNA3 spellings.
* **`token_bill.py` keys a templated kernel by its unmangled name**, so both instantiations collapse to one
  key and the printed bill silently describes one of them. Slice one instantiation per file (command in
  the evidence header) before quoting per-COLS numbers.
* **A harness that prints `NA` is not a harness.** Four parser bugs in `run_ab_g2.sh` were found only by
  running its summary block against synthetic logs; a summary column of `NA`s reads like a slow engine.
  Any A/B script gets a fixture run whose verdicts must differ.
