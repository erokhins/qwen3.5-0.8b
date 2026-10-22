# prototype/mtp_draft — results

The MTP draft block (§09.2) at its real shapes, gated before it is allowed to touch `engine/`.
Design and reasoning: `DRAFT-HANDOVER.md`. Price and pre-registered bands: `DRAFT-PRICE.md`.
Everything below was produced by the commands in §6; **no timing is recorded here** (worker device runs
may not produce numbers — `run_locked.sh` is the main agent's instrument and its table is still blank).

## 1. What exists

| file | what it is |
|---|---|
| `mtp_draft.hip.cpp` | host entry: block plan + bytes, perturbation, poison-guarded arenas, mode dispatch |
| `mtp_draft_kernels.h` | the rig's own kernels: `td_gather_row`, `td_read_witness`, `td_poison` |
| `td_glue_tu.cpp` `td_proj_tu.cpp` `td_mlp_tu.cpp` `td_attn_tu.cpp` | one TU per shipped op family, each `#include`-ing that op's real kernel header, exposing the engine's own forms |
| `td_modes.cpp` | the four arms: `chain` (14 launches/step), `launch`, `witness`, `xcheck`, `stagecheck` |
| `td_reference.py` | the mirror with two policies + the imported bands + fixture/golden readers |
| `gate.py` | the gate: fixture build, witness, per-observable judgement, derived negatives, `--selftest` |
| `price_block.py`, `DRAFT-PRICE.md` | the pre-registered price model and its WIN/narrow/KILL lines |
| `run_locked.sh` | the main agent's timing harness (admission-gated, canary-asserted, two interleaved passes) |

One draft step is **14 launches**, counted at the launch sites and asserted against `tdplan::
kLaunchesPerDraftStep` at run time: gather → 2 norms (via the fused fc-pair writer) → fc → input_ln →
q → k → v → attention row → o_proj → add+post_attn_ln → gateup → down → add+`mtp.norm`. Activations stay
BF16 end to end (no cast launches), licensed by the `stagecheck` arm below rather than by analogy.

## 2. The two references, and the witness that makes the mirror admissible

`speccheck/mtp.py`'s `MTPDraft.draft` is the shipped MTP reference and the source of the pinned golden,
but it computes §05's attention through F32 rope and F32 scores (`speccheck/specimpl.py:164-181` says so
and records that making it bit-faithful to the pins "was tried and reverted"). The kernels here use the
pinned policy. So the mirror carries both columns and neither judges the other's ground:

```
witness mirror(oracle)==golden step0.u:   ndiff=0/1024   BIT-EXACT
witness mirror(oracle)==golden step0.x_d: ndiff=0/1024   BIT-EXACT   (proposal 2064 = 2064)
witness mirror(oracle)==golden step1.u:   ndiff=0/1024   BIT-EXACT
witness mirror(oracle)==golden step1.x_d: ndiff=0/1024   BIT-EXACT   (proposal 369 = 369)
```

The oracle column is not a paraphrase: it calls `specimpl.attention`/`mlp` verbatim, which is why it can
reproduce `MTPDraft` bit-for-bit and why it has the standing to adjudicate the device at all. The pinned
column (BF16 rope table, rounded products, BF16-rounded scores) differs from it where the policy says it
must — at k=1, `qrop` differs in 512 of 2048 elements while `u` is identical (the policy cannot reach `u`,
and if it did that would be a bug, which is what the selftest asserts).

## 3. The device against the pinned column — 21 observables, both steps

Bands imported, never owned: `prototype/glue/run.py` (norms, adds), `prototype/proj_gemv/run.py` (GEMVs),
`prototype/attn_decode/run.py` (attention row), `prototype/mlp_gemv/run.py` (F1–F4). The per-element test
is `n_bad == 0` in the repo's own comparator — far-apart AND large — not "max ULP ≤ 1", which a legal dot
breaks at a cancelled element.

k=2, `dec8`, the worst of each class (k=1 is the same picture with fewer lanes):

| observable | band | result |
|---|---|---|
| `e`, `pair` | bit-exact | **ndiff 0**/2048, 0/4096 |
| `u` (fc) | proj | **ndiff 0**/2048 |
| `nrm`, `u2` | norm | ndiff 0–1/2048, max 1 ULP, n_bad 0 |
| `qg` `kg` `vg` `gate` | proj | ndiff ≤3/8192, max 1 ULP, n_bad 0 |
| `qrop` `krop` | attn | ndiff ≤2/4096, max 1 ULP, n_bad 0 |
| `aout`, `gout` | attn | ndiff ≤1/4096, max 4 ULP but **n_bad 0** (cancellation band), cos 1.000000 |
| `attn_o` | proj | ndiff 4/2048, max 1 ULP, n_bad 0 |
| `z1` | norm | ndiff 1/2048, max 1 ULP |
| `act`, `mlp_o`, `z2`, `xd` | **F4 aggregates** | cos ≥ 0.999999, meanrel ≤ 3.2e-4 (limits 0.9999 / 5e-3) |
| `cache_k`, `cache_v` | attn / proj | **ndiff 0**/1024 each |
| proposal | exact | device 369 = mirror(pinned) 369 = **golden 369** at step 1 |

Reference-free structural checks, same run: `gate == qg[:,256:]` bit-identical both steps (the §05.1
de-interleave), and `qrop` differs from the un-roped `qg[:, :256]` in 2047–2048 of 2048 elements (rope
actually ran).

Three band *classes* were corrected rather than widened, each by importing what the source op already does:

* `act`, `mlp_o`, `z2` moved from the per-element F1–F3 band to **F4 aggregates**, because `prototype/mlp_gemv/run.py:395,493`
  judges `down_proj`'s output that way for exactly this reason ("an `act` flip puts many ULP onto an `out`
  element", `floor_probe.py`). Without this the gate failed on a legal act flip: max 14,848 ULP at 0.157 of
  tensor max, cos 1.000000.
* `gate` moved from "bit-exact" to the proj band: it *is* a copy, so it was made a **structural** test
  against the device's own `qg` instead — a stronger claim with no reference to argue with.
* `e`/`pair` stayed bit-exact on purpose, and they are the load-bearing strictness: their inputs are pinned
  rows, so a wrong fc column order or a broken pre-FC norm is caught at bit level, not by a band.

## 4. The other arms

```
mode=xcheck  steps=2000 compared=4096000 diff_aout=0 diff_gout=0     → BIT-IDENTICAL
mode=witness uint2_read=5113216 raw=21223116554762222 addr=15646429789258508250 → all three MATCH the host walk
mode=stagecheck fc_diff=0 act_diff=0 down_diff=0
```

`xcheck` runs the transcribed `mtp_attn_row` and the shipped `attn_core` over the same contiguous-from-0
geometry at 10 lane counts (1,2,3,8,9,17,64,128,256,512) and requires bit-identity, which is the only
claim that makes the transcription safe to keep. Contiguous is the *only* honest geometry for it: an MTP
cache is dense in **append index** while `attn_core`'s `slot_of(pos, cap)` couples the slot to the absolute
position, so anywhere else a difference could be the seam rather than a mistake (§3 of the handover).

`witness` reads the block's 40,905,728 B exactly once and its triple is compared against a numpy walk of
the same bytes — the `tools/mixed_bw` rule that the expectation may not be produced by the thing under
test. `stagecheck` runs fc, gateup and down twice, staging the operands F32 versus BF16: **all three
diffs are 0**, which is what licenses keeping activations in BF16 and is a measurement here, not a
citation of `engine/mlp_xbf16_kernels.h`.

## 5. Negative controls, derived not hand-listed

For each injection the must-catch set is what the pinned mirror measurably moves; must-not is the
complement; an empty derived set aborts (the `partalias` 0-of-1,024 lesson). The paired device runs must
move exactly the first set and be byte-identical on the second.

```
fc-swapped      must-catch 20 caught 20 blind []  must-not  1 leaked []
h-source-pre    must-catch 20 caught 20 blind []  must-not  1 leaked []
pos-off-by-one  must-catch 12 caught 12 blind []  must-not  9 leaked []
fc-scale        must-catch 20 caught 20 blind []  must-not  1 leaked []
qproj-scale     must-catch 20 caught 20 blind []  must-not  1 leaked []
qnorm-scale     must-catch 10 caught 10 blind []  must-not 11 leaked []
oproj-scale     must-catch 20 caught 20 blind []  must-not  1 leaked []
down-scale      must-catch 20 caught 20 blind []  must-not  1 leaked []
norm-scale      must-catch 20 caught 20 blind []  must-not  1 leaked []
```

The sets are non-trivial in both directions: `qnorm-scale` is blind to the 11 observables upstream of
`q_norm` and the device agrees byte for byte there; `pos-off-by-one` reaches 12 observables, which is the
evidence that the dense-cache position seam is testable at all rather than invisible.

## 5a. What the gate found by being run

Four real defects, all found by a check rather than by reading:

1. **The scratch arena was under-allocated by one guard and the poison fill wrote 8 KiB past it.** The guard
   is `guard` *words* per side but the allocation added `2*guard` as if they were bytes. The symptom was
   `guard_left=12160` of 12288 — i.e. the guard check had teeth and caught its own setup, which is the only
   reason this was not shipped as a plausible PASS.
2. **The cache dump assumed row-major lanes.** The shipped layout is plane-major,
   `kcache[(kv * cap + lane) * DH]` (`prototype/attn_decode/attn_decode_kernels.h:317`), and the tail-poison
   check said so: `a cache lane outside the chain's own append range was written`. Comparing without the
   transpose would have compared a transposed cache and passed. The dump is per plane now, and `gate.py`
   transposes to the mirror's `[row][kv][256]`.
3. **The mirror paraphrased §05.6 instead of transcribing it.** `attn_core`'s epilogue is
   `bf(av * bf(fb(1/(1+exp(-g)))))` with `av = bf(acc)` — the head output rounded *first*. Computing
   `aout_f32 * bf(sigmoid)` instead left `aout` bit-exact while everything downstream went 0.2 %
   mean-relative off (worst 14,848 ULP at 0.157 of max). Same trap class as a hoisted scalar constant in a
   norm: bit-identity is owed to the source text's association, not to the algebra.
4. **`xcheck` passed host vectors as device output pointers.** The answer was
   `HSA_STATUS_ERROR_MEMORY_FAULT` at a host mapping, after the launches "succeeded" — the same shape as the
   recorded "a staged launch plan with a half-set pointer" fault: the dump names an address and the address
   is the bug.

Two smaller ones: the rig's `--dump` and `--time-burst` are refused together on purpose, and the gate's own
RESULT-line parser had to be told that `mode=xcheck` is not an integer — a parse that silently swallows a
field is the same failure as an empty must-catch set.

## 6. Commands

```bash
export LC_ALL=C
bash prototype/mtp_draft/build.sh                                  # ~20 s, six TUs
.venv/bin/python3 prototype/mtp_draft/gate.py --selftest            # CPU only, no device
.venv/bin/python3 prototype/mtp_draft/gate.py --fixture dec8 --k 2 --negatives   # device, ~8 s
for a in xcheck witness stagecheck; do
  .venv/bin/python3 prototype/mtp_draft/gate.py --fixture dec8 --k 2 --arm $a; done
.venv/bin/python3 prototype/mtp_draft/price_block.py                # the bands, recomputed
prototype/mtp_draft/run_locked.sh                                   # MAIN AGENT ONLY: takes timings
```

`gate.py` does not take `tools/gpu_lock.sh` itself (the three self-locking drivers named in AGENTS are not
this one); `run_locked.sh` holds one lock for the whole admission+timing sequence.

## 7. Not claimed

* No timing of any kind. `DRAFT-PRICE.md` §4 is empty by design.
* `k=4` is admitted structurally only — `dec8`'s golden is two steps, so step 3's inputs would be nobody's
  pin. `run_locked.sh` labels its admission level and refuses to let a structural arm become a headline.
* Nothing was written to `engine/`, and nothing is committed. Graduation needs the price measured first, and
  the accept rate (`a` on the engine's bits, D4) is still the load-bearing unknown that no rig here can
  answer.
