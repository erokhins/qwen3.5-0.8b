# CROSSROW — the cross-row dependence in the wide attention body is a shipping prefill defect

**Verdict: it is causal, it is visible in plain prefill with no MTP code anywhere in the run, and it lives
in the wide attention block (`attn_wide_wave` / `attn_wide_llama_wave`), which is the shipping default
(`--attn-fattn-wide 2`).** No engine edit was made and no band, floor, pin or ledger line moved. Nothing is
committed. No timing was taken or printed anywhere; the only seconds in this file are the wall time of a
whole command, which the device-run protocol requires.

## 1. Arm 1 — the prefill reproduction (the headline)

`prototype/mtp_verify/crossrow_arm1.py`. Two prefill runs of the **same prompt length** whose token ids
agree on every position except one. Same length ⇒ same `rows`, same `lanes_last`, same `split`, same wide
grid, same GEMM `M`; the geometry cannot explain a difference, and the state is fresh on both sides
(`--steps 0`, prefix 0). Under a correctly causal engine rows before the perturbed position must be
**byte-identical**.

Fixture `p128` (128 tokens, one 128-row body, default flags), one position perturbed per run, 85
positions sampled:

```
sweep rows=128 cases=85 cross-row edges=44
```

**The law, and it is exact over every one of the 85 sampled positions:**

| perturbed position p | earlier rows that moved | distance |
|---|---|---|
| p odd, p ≥ 3 (44 cases) | exactly one row: **row p−1** | exactly 1 |
| p even (40 cases incl. 0, 2, 16, 32, 64, 96, 120, 124, 126) | none | — |
| p = 1 | none (row 0 is the one exception) | — |

So: **row q attends one lane past its own causal bound, for every even q ≥ 2; odd rows and row 0 are
clean.** Row q's bound is `lanes_of(a, q) = lvalid + q = q + 1` (lanes `0..q`, its own new lane included —
`engine/kernels.h:679-680`), and the influence reach is exactly one lane, never two, never further back.

Sizing, over the 44 movers (`1−cos` on the moved row's logits, `prototype/bf16_ulp.py`):

* range **1.31e-05 … 2.26e-04**;
* **argmax SAME at all 44** — no decision moved in this fixture;
* deterministic: the same perturbation re-run is byte-identical (`logits126.bin` across two runs of the
  same prompt), so this is a fixed arithmetic path, not a race;
* the worst mover sits **~9× inside** `CASCADE_BAND["row_1cos_max"] = 2.0e-3`, which is why nothing
  existing catches it. The MTP handover's 1.9e-05 … 1.19e-04 is this same population, sampled at 8 anchors.

Confirmed at long context and on the streaming route — `wide640f`, 640 rows = waves of 512 + 128, rows
512..639 above `MAXSC = 512` (`split=2`, `lanes_last=640`):

| comparison | result |
|---|---|
| `ref` vs `pert639`, adjudicated row 638 | **MOVED**: 1−cos **1.163e-05**, meanrel 4.417e-03, ndiff 160188, argmax 978/978 SAME |
| `ref` vs `pert638`, adjudicated row 637 | byte-identical (even perturbation, as predicted) |

**This is the answer to the question that mattered: a query row is influenced by the row that comes later
in the same body, on the shipping prefill path, at a magnitude that hides inside every band in force.**
The MTP verify finding is one instance of it — a verify body is exactly the pair (even row 0, odd row 1).

## 2. Arm 2 — localising it: the first differing buffer is the wide kernel's own output

Attribution instrument: the form ladder, `prototype/mtp_verify/crossrow_ladder.sh` (each config re-runs
both sides, so a config in which the edge vanishes is a config the coupling does not live in):

| config | row 126 after perturbing lane 127 |
|---|---|
| default (`attn_fattn_wide=wide_llama`) | **MOVED** 1−cos 9.551e-05, ndiff 214589 |
| `--attn-fattn-wide 1` (`attn_wide_wave`, pre-A13 spelling) | **MOVED** — and its whole 128-row logits set is byte-identical to the default's (128/128 files) |
| `--attn-fattn-wide 0` (`attn_slim_wave` + `attn_fattn_wave` + `attn_combine`) | byte-identical — **clean** |
| `--gemm hand` (hand GEMV instead of hipBLASLt) | **MOVED** 9.551e-05 → not the library GEMM |
| `--wave-rows 1` (every prefill row its own one-row body) | byte-identical — **clean** |

And at 640 rows, where `--attn-fattn-wide 0` routes the streaming rows through `attn_fattn_wave` +
`attn_combine`: **clean** (row 638 byte-identical under the perturbation that moves it under the wide form).

What the exclusions buy, without touching `engine/`: every stage upstream of attention — embedding, the
projection plan, the GDN wave recurrence and conv, the MLP, the glue, `attn_prep`'s staging and the §06.7
appends, and the per-row pricing path `finish_token` — is **shared verbatim** by the leaking and the clean
config and differs between them only in which attention core ran. If any of those stages produced a
row-(p−1) value that depended on row p, the clean config would show it too. Therefore:

* the first buffer that differs is the **wide block's attention output for row q** (its `finish_head` write
  into `aout`), not a projection output, not a cache lane, not a mixer state;
* both wide spellings produce **byte-identical logits for all 128 rows**, which is consistent with A13's
  byte-identity claim and means the defect is in what the two spellings share — the row/round geometry and
  the causal bound — not in A13's ported round.
* the frozen one-row form is clean, and so is the whole non-wide wave route at both one-pass and streaming
  widths. **This is a defect of one form, not of the staged plan.**

## 3. Arm 3 — which mechanism, and which hypotheses are dead

Measured shape: **exactly one lane past the bound; only when the row's lane count `rlanes = q+1` is odd
(i.e. even q); reach never 2; deterministic; magnitude ∝ content.** Against the four hypotheses:

* **softmax normalisation spanning rows** — dead. A shared denominator moves the whole row by a scale
  factor and would show at every p, not on the parity of `q`, and would be far larger than 1e-4.
* **a shared reduction / scratch reused across rows without separation** (`s_out`, the partial slots) —
  dead for the same reason: an aliased slot is either a whole-row corruption (the §20.1 signature, which
  `partalias` exists to catch) or nothing, and it would not be gated on lane-count parity. The wide block
  also writes no partials at all (`combine=0` in the `ATTN_WIDE` echo).
* **the causal mask applied per lane instead of per query row** — dead as stated. The mask *is* per row:
  `engine/attn_wide_kernels.h:200` takes `rlanes = lanes_of(a, rrow)` for the row-slot this lane
  accumulated, and `:259-262` masks `live = row_ok && j < rlanes` per cell, with `j` the true lane index.
  Same law in `engine/attn_wide_llama_kernels.h:391`.
* **what is left, and what the numbers point at** — the row's causal bound is one lane too wide whenever
  that bound is odd. The bound is *derived* from a block-uniform quantity in both spellings —
  `hi = lanes_of(a, rowhi-1)` (`attn_wide_kernels.h:186`, `attn_wide_llama_kernels.h:377`, and the llama
  variant's round-skip `lo_bound = lanes_of(a, row0)` at `:379`) — and the block deliberately over-fetches
  (`"the causal overhang this pays is bounded by WIDE_POS_PER_BLOCK = 32 positions per block … and per-row
  masking below is what absorbs it"`, `attn_wide_kernels.h:183-185`). The observation is that the
  overhang is not fully absorbed: it is absorbed except for one cell, and which cell survives is a
  function of the parity of the row's own lane count. The stage is quantised to `WIDE_STAGE = 64` and the
  C-tile columns to `FATTN_MLANES = 16` with the per-lane column law
  `fattn_c_col(lane, l) = 2l + (lane >> 4)` (`attn_fattn_kernels.h:454`), i.e. **column parity is carried
  by the lane**, and the row-slot law is `fattn_c_row(lane) = lane & 15` with positions at
  `rslot / GQA` — so a bound that is rounded against a 2-lane granularity is exactly the shape that would
  leak one lane on odd bounds and none on even ones.

**Stated plainly, because the brief asks for the honest version:** the *attribution* (which kernel, which
route, which bound, which cell-geometry could produce it) is proved; **the exact predicate that fails is
not yet pinned**, and inspection will not pin it — the mask reads correctly. It needs the operand dump.

## 4. Arm 4 — what this does to the claims in force

* **"a 2-row body verifies a token identically to the one-row path" is false as a byte claim and must be
  retired on the record.** It is the pair case of the law above: row 0 is even, so row 0 reads row 1's
  lane. `prototype/mtp_verify/VERIFY-HANDOVER.md` already observed the movement and already recorded that
  byte identity was never a band this tree bought (`[DEFERRED-FAIL]` on 15 such claims), so the arithmetic
  claim is already in the honest state — but `prototype/mtp_verify/ARMS.md:5` still frames it as "row 0 of
  the body is not bit-identical to the step it replaces", i.e. as a verify-body shortcoming. It is not: it
  is a property of the shipping wide body. That line is the main agent's to rewrite (docs rule).
* **The severity is prefill, not MTP.** Every prefill wave of >1 row runs this form, so every even query
  row in every wave attends one future token. Today no pinned gate can see it: the reach is 1 lane, the
  argmax did not move in 44/44 sampled cases, and the effect is ~9× inside `CASCADE_BAND` at worst. What
  it costs in quality is a separate question from whether it is a leak, and this increment does not answer
  that question.
* **The band that currently tolerates it** is `CASCADE_BAND["row_1cos_max"] = 2.0e-3` (imported by
  `prototype/mtp_verify/gate.py` from `engine/gate.py`, not restated here). Nothing was widened, and
  nothing should be: the leak is inside a band that was bought for a different reason, which is an argument
  for a *causality* check, not for a wider band.

## 5. What would pin the mechanism, and why it is not in this increment

The one datum missing is **which cell of row q's score row is non-−1e30 that should be**: a dump of the
score/`acc` row for a 2-row and a 4-row wave would name the column, and from the column the failing
predicate follows in one step. That needs either a `CoreArgs` field or a kernel-side dump — an `engine/`
edit, which this brief forbids. The pattern to hand to whoever does it:

1. 2-row body (rows 0,1) reproduces the MTP case; 4-row body (0..3) covers both row-slot quarters of a
   wave and shows whether the leak tracks absolute position parity or row-slot group index
   (`rslot/GQA ∈ {0,2}` leak, `{1,3}` do not — `attn_wide_kernels.h:196-200`);
2. dump `acc[L][c]` after the mask for row 0 and check the cell at `j = rlanes(0)`;
3. the parity datum says look for a bound rounded or shifted against a **2**-lane granularity, not 16 or
   64 — check `fattn_c_col`'s parity term `(lane >> 4)` against the comparison `j < rlanes` rather than
   assuming both sides agree about which column a lane holds.
4. `--attn-fattn-wide 0` is a correct, already-shipping route that is **clean at both one-pass and
   streaming widths** — if a decision is needed before the kernel fix, that flag is the off switch, and
   the A12/A13 wide-form timing wins are then off by default.

## 6. Traps this pass hit or confirmed

* **A perturbation boundary that coincides with the geometry's own boundary is a blind spot, not a
  result.** The first arm-1 test split the prompt at K=64 of a 128-row wave and reported "rows 0..63
  byte-identical" — read as a null, and wrong. The wide block's unit is `WIDE_POS_PER_BLOCK = 32`, and the
  leak reach is 1, so a boundary at a multiple of 32 cannot show it; the per-position sweep is what found
  it. A must-catch set has to be derived from the geometry (`AGENTS.md`'s derived-must-catch row, one
  further level: derive it from the *defect's* reach too, not only the fixture's).
* **`--wave-rows 1` being clean does not exonerate anything except the one-row form** — every upstream
  wave stage still runs in the multi-row config; it is the *pair* of configs (wide vs non-wide, everything
  else fixed) that carries the exclusion argument, and a one-flag ladder cannot localise to a line.
* `--attn-tile 0` with `--attn-fattn 1`, and `--attn-touch 0` with `--attn-fattn 0`, are refused — the
  ladder's first versions reported these as RUN FAILED with a `tail -1` that showed a usage line, not the
  refusal. A ladder must print the refusal (`decode_run FAILED:` line), or it reports a config it never ran.

## 7. Commands (all correctness-only; `util` at acquire read 89–92 %, which this box reads high by default; no timing taken)

```bash
export LC_ALL=C
# prompts (CPU only). 85 single-position perturbations of p128, and 2 of wide640f.
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py prep    --fixture p128     --pool wide640f --prefix 64 --work /tmp/crossrow_p128
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py perturb --fixture p128     --pool wide640f --positions <list> --work /tmp/crossrow_p128
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py prep    --fixture wide640f --pool p300     --prefix 639 --work /tmp/crossrow_w640
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py perturb --fixture wide640f --pool p300     --positions 638,639 --work /tmp/crossrow_w640

# device (decode_run does NOT self-lock; the lock is held around the batch)
bash tools/gpu_lock.sh acquire <tag>
./engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file <file>.csv --steps 0 \
    --prefill-logits all --out <dir>            # 29 runs: 35 s wall; 32 runs: 38 s wall; 3 runs @640 rows: 9 s wall
bash tools/gpu_lock.sh release <tag>

# the graph and the bands (CPU only)
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py sweep   --ref /tmp/crossrow_p128/ref --work /tmp/crossrow_p128 --positions <list>
.venv/bin/python prototype/mtp_verify/crossrow_arm1.py compare <ref-dir> <pert-dir> --edge-row <q>
bash prototype/mtp_verify/crossrow_ladder.sh                  # 8 configs x 2 runs; wall ~20 s total per batch
```

---

## Correction appended by the main agent (2026-10-22) — the predicate named here is REFUTED

This file's headline claim — that the wide block admits `lanes_of(a,q)` because the **causal
bound is one lane too wide** — was disproved at cell resolution by `FIX-CROSSROW.md`
(commit `b574f40`). The datum this file discovered is real and reproduced; the mechanism is
not mask widening. Measured at the attention output from the dump: perturbing one token moves
**exactly one earlier row, by ≤ 1 BF16 ULP (1−cos 0.0)**, and only when that row's bound is
odd — a widening would move every lane past the bound, not one. The wide kernels launched
directly with the engine's own `CoreArgs` honour the bound exactly across 345 cases and four
independent encodings, with 39 must-catch injections all red including a host-side
widen-by-one that is exactly the defect refuted here.

Surviving mechanism, as a shaped clue rather than a finding: f16x2 P-pair packing
(`__shfl_xor_sync`) straddles the causal boundary exactly when the bound is odd, so one live
and one masked cell share a pair — which matches the dependence set and the fact that both
wide knobs leak identically. Read `FIX-CROSSROW.md` Results 5, 7 and "what is left" before
using anything in this file as a diagnosis. Kept unedited above for the record of how the
datum was found; a superseded diagnosis is history, not an instruction.
