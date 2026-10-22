# P7 / MTP — the draft block: design, bands, and the commands the main agent must run

Prototype-only. **No `engine/` file is touched by this directory**, no timing is taken here (every
millisecond below is quoted and labelled), and nothing is committed. This file is the checkpoint; it is
written before the long analysis steps so a compaction cannot eat the design.

## 0. What is being built, and what decides it

One MTP draft step (§09.2) at the real shapes, in `prototype/mtp_draft/`, gated against the shipped MTP
reference (`speccheck/mtp.py`, `MTPDraft.draft`) on the pinned fixture inputs, so it can graduate into
`engine/` in a later increment. The block is 15 tensors / 40,905,728 B (`mtp_census.json`), currently
priced **by transfer only** at 0.22–0.25 ms; D2's lesson is that a rig which stages it differently from
the engine prices the rig, so every stage here is the engine's own kernel source, included, not retyped.

Per draft step the block is **14 launches** (derived by counting, §5 of this file — D2's "~18" is gone):

```
 1 embed gather   E[y_next]                       (engine/kernels_embed.cpp's shape)
 2 norm  e'       pre_fc_norm_embedding           prototype/glue/glue_kernels.h   glue_add_norm<256>
 3 norm  h'       pre_fc_norm_hidden              ditto, second weight, so a second launch:
                                                  glue's norm weight is deliberately NOT row-indexed
 4 fc             pair (e' ! h') [2048] -> u      prototype/proj_gemv proj_gemv_tokens<2048,1,PACK_X>
 5 norm  u'       input_layernorm                 glue
 6  q_proj        u' -> qg [4096]                 proj_gemv_tokens<1024,...>   (8 heads x (256 q | 256 gate))
 7  k_proj        u' -> kg [512]                  proj_gemv_tokens<1024,...>
 8  v_proj        u' -> vg [512]                  proj_gemv_tokens<1024,...>
 9  attn row      qg,kg,vg + OWN kv, pos p        prototype/attn_decode attn_core arithmetic (see §3)
10  o_proj        gout [2048] -> attn_out         proj_gemv_tokens<2048,...>
11  add + norm    z1, u2  post_attention_ln       glue_add_norm (the §03.2 add fused with the next norm)
12  gate/up+SwiGLU u2 -> act [3584]              engine/mlp_xbf16_kernels.h mlp_gateup_xbf16<1024>
13  down          act -> mlp_out                 mlp_down_xbf16<3584>
14  add + norm    z2, x_d  mtp.norm              glue_add_norm
```

Activations stay BF16 end to end (the engine's `*_xbf16` staging: `hip_bfloat16 -> float` is exact, so
this is bit-identical to the F32-staged form and it deletes a cast launch per use), so the chain is one
enqueue per draft round with **no host sync inside it** — which is what makes the k-step chain priceable
at all.

## 1. The two things the task's "bit-exact" phrase cannot mean, found before writing code

Both are pre-existing, documented properties of this repo, not defects of this rig. They are recorded
here because the load-bearing claim ("a draft layer that drifts from the reference makes the accept rate
meaningless") has to be stated in a form the repo can actually honour.

1. **A GEMV is never bit-identical to torch's dot.** `prototype/proj_gemv/results.md` gates the shipped
   projections against the pins at `ndiff 1..12` elements per row, `max ULP` up to 193, all of it inside
   the cancellation band, and §10.1 F1–F4 is the same story for the MLP: "reordering the reference's own
   F32 sum already moves ~1 element in 1e5 by more than 1 ULP". So the GEMV stages of the block are
   judged by the repo's derived per-element ULP band + aggregates, exactly as those ops are.
2. **`speccheck`'s MTP path uses two F32 plumbings the model does not.** `speccheck/specimpl.py:164-181`
   says it outright: `apply_rope` keeps an F32 table and unrounded products, while "the pins carry a BF16
   rope table with each product rounded before the add (§05.3) ... the F32 form misses 52-97 of every
   2048 elements", and making the oracle bit-faithful "was tried and reverted". Same class one stage
   down: `attention()` keeps its scores in F32, while `attn_core`'s gated `full` form scores as
   `bf(fb(dot)) * SCALE`. Neither is a knob this rig may widen a band for.

Consequence, and the shape the gate takes: the rig judges the device against **two references at once**,
neither of them invented here.

* a CPU mirror written in the **engine's stage structure and the model's pinned policy** (BF16 rope table,
   BF16-rounded scores). Stage-local stages (norms, residual adds, cache writes, the gate vector) are
   required **bit-for-bit** against it — glue's own gate already demands bit-identity of its norm against
   the pins, so this is a real, achieved bar, not a wish — and the GEMV/attn stages under the imported
   bands. The mirror's right to be the reference is itself witnessed: **the mirror run with the oracle's
   F32 plumbings must be bit-identical to `speccheck.mtp.MTPDraft.draft`** on `u`, `x_d` and the proposal.
   A mirror that cannot reproduce the shipped form is not a reference.
* the **pinned golden** from `prototype/mtp_gatezero/golden/` (2 chained steps, `dec8` inputs), judged
   with the published §10.1 bands, plus **exact proposal equality**, which is the accept-rate-bearing
   quantity.

**Open question handed to the supervisor, not decided here:** which rope/score policy MTP is signed off
under. The rig measures the size of the difference instead of guessing at it; if the pinned policy is
chosen, `speccheck/mtp.py`'s golden stays a wider secondary reference, and if the oracle's F32 form is
chosen, `prototype/attn_decode`'s bit-exactness against the model's own pins is what has to be revisited.
Nothing in this increment depends on the answer, because both are gated.

## 2. Bands: imported, never owned

The rig declares **no numeric constant of its own**. Everything is imported by path from the op whose
floor probe measured it:

| observable | rule | source |
|---|---|---|
| norms, residual adds | bit-for-bit (and if not: `bf16_ulp` 1 ULP / cancel 3e-5 / ndiff 2 % / cos 0.9999 / meanrel 1e-2) | `prototype/glue/run.py` |
| `u`, `qg`, `kg`, `vg`, `o_proj` out | 1 ULP outside cancel band, ndiff 2 % (abs 1), cos 0.9999, meanrel 1e-2 | `prototype/proj_gemv/run.py` |
| `act`, `mlp_out` (F1/F2/F3 class) | same shape of rule | `prototype/mlp_gemv/run.py` |
| attention row | same shape, cancel band 2e-4 | `prototype/attn_decode/run.py` |
| `x_d` (chain end) | aggregates only: cos 0.9999, meanrel 5e-3, **no ndiff** | `prototype/mlp_gemv/run.py` `F4_*` |
| proposal token | exact | the golden's own proposals |

## 3. The one stage that cannot reuse a shipped kernel verbatim, and the witness that keeps it honest

`attn_core` couples **rope position** and **cache slot** in one field: `row_pos = a.pos + qrow` feeds both
`head_norm_rope(..., pos, ...)` and `slot_of(row_pos, cap, ...)` = `pos % cap`, which is right for the
target (every position writes a lane, so dense index == position) and wrong for the MTP cache. The MTP
cache is **sparse in absolute position**: at draft depth k the drafter writes one lane per drafted token
and the target consumes k+1 per round, so a k=1 round drafts at positions q+1, q+3, q+5, ... (read from
`speccheck/mtp.py`'s `greedy_mtp`: drafts are fed at `q+1+j` and the verify consumes `q..q+k`). Filling
the gaps is not an option — the gap positions were never drafted, and a zero lane gets real softmax
weight.

So `mtp_attn_row` is a **transcription** of `attn_core`'s fused + one-pass `full` path with exactly one
difference: the appended lane's slot is the dense append index instead of `pos % cap`. It calls the
shipped `head_norm_rope` (so §05.2/§05.3 are the shipped function, not a copy) and transcribes the score
dot, the `bf(fb(dot)) * SCALE` rounding, the tile walk, the max/sum/BF16-weight/PV order and the §05.6
epilogue character for character. The transcription's fidelity is then **disproved independently of the
numbers**: a synthetic case whose positions are contiguous from 0 makes the two addressings coincide, and
there `mtp_attn_row` must be bit-identical to `attn_core` on `aout` and `gout`. Any other difference in
the copy shows up as a bit difference on that case, and only there.

Consequence for the engine increment, stated now so it is not discovered later: the engine needs an
addressing seam — a separate write-slot field in `CoreArgs`, or an MTP cache sized and indexed by absolute
position. That is a numerics-adjacent interface decision and it is not this increment's to make.

## 4. Negative controls: derived by experiment against the reference, not hand-listed

Each injection perturbs **one tensor of this block** (or one input policy), applied on the host copy
before upload — no injection branch inside any shipped kernel, per `hb_kernels.h`'s precedent. For each
injection the *mirror* is run clean and perturbed, and the observable sets are computed from that diff:

* `must_catch(I)` = observables the mirror says can move. Asserted **non-empty** (an injection whose
  must-catch set is empty is vacuous and is dropped with a printed reason, not quietly kept).
* `must_not_catch(I)` = the rest. Asserted non-empty too, which is what kills the "another layer's weights
  catches everything and proves nothing" failure: a perturbation of `q_norm` must not move `u`.

Device side, the paired comparison is **device-clean vs device-perturbed** (same binary, same inputs, only
the perturbation differs), so there is no legal rounding noise to excuse a must-not-catch movement.

Two structural blind spots are *asserted as blind spots*, with a landing witness, rather than pretended
away:

1. **A singleton attention row cannot be moved by its scores.** With one lane in the MTP cache the softmax
   is 1.0 whatever the score arrives as, so `q_norm`/`k_norm`/rope injections are must-not-catch on step 0
   and must-catch on step >= 1. This is why the gated case is >= 2 chained steps (the same derived-blindness
   argument `draft_golden.py` records for itself).
2. **A value a norm divides out cannot be caught by the output.** `--neg scale2` doubles both inputs of the
   final `glue_add_norm` (the add and the norm are one launch), which is an *exact* scaling in binary FP and
   therefore provably cannot move `x_d`. The control is required to have **landed** — the dumped residual and
   the dumped MLP output must both move — and `x_d` is required not to. A gate that reports this as a
   caught bug would be lying; the blind spot is on the record instead.

## 5. The launch census, the byte census, and the canary

* **Launch census**: every launcher call in the rig increments a counter; the rig prints
  `launches=<n>` and it is asserted equal to `14 * k_draft` (+ the witness launches). This replaces D2's
  "~18 launches, not a measured census".
* **Byte census, witnessed**: a dedicated read kernel walks the 15 blobs' byte multiset exactly once with
  an address-weighted checksum and a fetch count derived host-side from the plan, never read back from the
  kernel's own accumulation (`tools/mixed_bw`'s rule: a check that passes by construction cannot fail, and a
  per-thread store to a per-block slot is not a block reduction — the reduction is a 64-bit warp butterfly).
  Asserted `bytes_read == 40,905,728` per draft step.
* **Poison guard bands**: the weight arena and every output region are surrounded by poison words that
  writes never store, and the host requires them untouched, so an out-of-bounds store is a byte fact.
* **Behavioural canary (required to move)**: `tensors_touched`, `weight_bytes_demanded` and `launches` must
  scale exactly with the draft depth k (15 / 40,905,728 / 14 per step), and the block arm must read
  differently from the no-block arm. Read in the same breath as the timing, aborting before any ms is
  printed if it does not move.

## 6. Arms for the main agent (`run_locked.sh`)

One `tools/gpu_lock.sh` hold, acquired and released inside one shell, blocking, never backgrounded; two
interleaved passes; admission-gated (correctness red => `NO-TIMING`, no ms printed). Neither
`mtp_draft` nor `gate.py` contains lock code (`grep -c gpu_lock` is 0 in both), and the script invokes no
self-locking driver inside the hold. Arms: block chains of k = 1, 2, 4 draft steps (the draft chain is
serially dependent, so depth is a chain, not a wider launch — see `DRAFT-PRICE.md` §c), plus the witness
arm that reads the block's bytes and nothing else, which is the transfer-only number D2 asserted.

**No timing was taken by this worker.** No `ms` figure in this file is measured here.

## 7. What the main agent has to run, and in what order

```bash
export LC_ALL=C
bash prototype/mtp_draft/build.sh                                    # 6 TUs, -O3, rpath
.venv/bin/python prototype/mtp_draft/gate.py --selftest               # CPU only: comparator + band imports
.venv/bin/python prototype/mtp_draft/gate.py --fixture dec8           # device: admission set + negatives
bash prototype/mtp_draft/run_locked.sh --admit                        # lock window, no ms
bash prototype/mtp_draft/run_locked.sh                                # lock window, two timed passes
.venv/bin/python prototype/mtp_draft/price_block.py --evidence /tmp/mtp_draft/evidence
```

## 8. Status after the first device pass (supersedes §7 and parts of §2)

The rig built, ran and went green; `results.md` is now the record, and §7's command list was written before
the interfaces settled. The commands that exist are:

```bash
export LC_ALL=C
bash prototype/mtp_draft/build.sh
.venv/bin/python3 prototype/mtp_draft/gate.py --selftest                      # CPU only
.venv/bin/python3 prototype/mtp_draft/gate.py --fixture dec8 --k 2 --negatives
.venv/bin/python3 prototype/mtp_draft/gate.py --fixture dec8 --k 2 --arm xcheck|witness|stagecheck|launch
.venv/bin/python3 prototype/mtp_draft/price_block.py                           # prints the bands
prototype/mtp_draft/run_locked.sh                                              # main agent only
```

Where the design written here turned out to be wrong, in the order it was found:

* **§3's cache seam is real and the dump has to know the layout.** The shipped cache is plane-major,
  `kcache[(kv * cap + lane) * DH]`; §3 assumed a row-major dump. The tail-poison check caught it. `gate.py`
  transposes the device dump to the mirror's `[row][kv][256]`, and the `xcheck` arm — bit-identity against
  the shipped `attn_core` over 10 lane counts, 4,096,000 compared values, zero differences — is what
  licenses the transcription.
* **§2's band table mis-assigned three classes.** `act`, `mlp_o` and `z2` are F4-aggregate observables, not
  F1–F3 per-element ones, because that is what `prototype/mlp_gemv/run.py:395,493` already does for
  `down_proj`'s output; `gate` is a structural equality against the device's own `qg` rather than a
  bit-exact dump. Both changes import an existing op's choice; no threshold is owned here.
* **§1's stage list is right but the epilogue had to be transcribed, not paraphrased.** §05.6 rounds the
  head output to BF16 *before* the gating product; paraphrasing it left `aout` bit-exact and everything
  downstream 0.2 % off. `td_reference.py` now mirrors the shipped expression's association.
* **§6's canary has an extra term.** Besides the plan and behaviour canaries, the rig asserts its own byte
  invariants (`poison_left=0`, `guard_left=12288`, cache-tail poison), and that is what caught the
  under-allocated scratch guard and the cache-layout mistake before either became a plausible PASS.

Still open, unchanged from §5: no timing has been taken, `k=4` is structurally admitted only, `engine/`
graduation waits on the measured price and on `a` (D4).
