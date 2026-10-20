# A12 reds — diagnosis and fixes (worker pass, 2026-10-20)

Three reds from `prototype/attn_wide/A12-ADMISSION.md`, worked in the brief's priority order. No band,
threshold, must-catch list or fixture list is widened anywhere in this file; no default is flipped; no
timing is taken (every device run is a numerics gate and says so).

Status: **all three items worked to a verdict. Item 1 ends with no code change and a decision handed up
(four mechanisms tested, three refuted on device); item 2 is a sized fixture/floor artifact, no threshold
touched; item 3 is fixed by declaration — `zz`'s red now names itself and its cause on every surface.**
Detail per item below; §4 carries the witness, the device-run list and the handover.

---

## 0. The measurements the diagnosis rests on

All numbers are F32 logits dumps read on CPU from the gate's own work dirs
(`tmp/a12fix/gate_smoke_w{0,1}/{ab_batched,ab_perrow,ab_carry}`), i.e. the same bytes the gate judged.
Scripts: `tmp/a12fix/rowcontrib.py`, `tmp/a12fix/crossmatrix.py`, `tmp/a12fix/whichrows.py`,
`tmp/a12fix/arms.py`.

### 0.1 No row carries the pooled excess — it is spread over all nineteen

`corpus_drift` pools `cos = Σdot / sqrt(Σna·Σnb)`. Decomposing exactly
(`c_i = (sqrt(na_i·nb_i) − dot_i) / sqrt(Σna·Σnb)`, Σc_i = pooled 1-cos + a Cauchy residual of
−2.104e-05 which is printed and not absorbed), the top three rows of the wide arm's
batched-vs-carry pairing are:

| row | 1-cos | share of the pooled figure | row energy share |
|---|---|---|---|
| 0 | 9.179e-04 | 2.804e-05 | 0.0300 |
| 14 | 6.347e-04 | 2.393e-05 | 0.0375 |
| 10 | 5.434e-04 | 2.286e-05 | 0.0420 |

Σ over the 19 rows = 1.818e-04 against the 2.028e-04 the gate printed. The largest single row owns
15.4 % of Σc_i (13.8 % of the printed pooled figure), the top three together own 41.2 %, and the top
four — rows 0, 14, 10, 1 — sit within a factor of 1.25 of one another. **This is not one row going
bad; it is every element of every row moving a little.** Any fix has to be a change to a rounding that
applies to all rows.

### 0.2 The wide block moves the batched and carry sides, and only them

| pairing | pooled 1-cos |
|---|---|
| knob 0, batched vs one-row | 1.618918e-04 |
| knob 0, batched vs carry | 1.789566e-04 |
| knob 1, batched vs one-row | 1.622213e-04 |
| **knob 1, batched vs carry** | **2.028112e-04** ← red, band 2e-04 |
| knob 0 vs knob 1, batched | 1.241696e-04 |
| knob 0 vs knob 1, one-row | **0.000000e+00** |
| knob 0 vs knob 1, carry | 1.206051e-04 |

Per-row byte-equality across the knob: `ab_batched` 18 of 19 rows moved (every element of each),
`ab_carry` 18 of 19 moved, `ab_perrow` **0 of 19 moved**.

The one-row side cannot move: `engine/kernels_attn.cpp` routes the knob as
`wave_wide = a.fattn_wide != 0 && rows > 1`, and a one-row body is the fused `attn_core` whatever the
knob says (the header states this: "One-row bodies never consult it — the fused"). So `ab_perrow`
at knob 1 is the same binary path as knob 0, and the knob-1 "batched vs one-row" pairing is
wide-vs-fused.

Consequence for sizing the fix: the knob's own move is 1.24e-04 (batched) / 1.21e-04 (carry), and it
is largely common-mode between the two sides — the pairing only grew by 2.39e-05, i.e. ~12 % of the
knob's own movement is differential. **To get 2.028e-04 under 2.0e-04 the differential part has to
shrink by ~12 %.**

### 0.3 The mechanism is the wide form's *formats*, at one round, and the three candidate levers are bounded on pinned data

`smoke`'s widest row is 19 lanes and `WIDE_STAGE` is 64, so **every row of `smoke` is one round**: the
64-position reassociation cannot be the mechanism here, there is nothing to reassociate. That makes
the attribution clean — at one round the wide form differs from the pinned one-pass form
(`§05.5`: `bf16(exp(s−m)/l) @ V`, which `attn_slim_wave` reproduces bit-for-bit; `ref vs the PIN`
below reads 0) in exactly three places:

1. **P**: `f16(exp(s−m))`, un-normalised, vs the pin's `bf16(exp(s−m)/l)`.
2. **the accumulator**: F16, one rounding per 16-position slice-add, vs F32.
3. **the final divide**: `acc_f16/l` once in the epilogue, vs the weight being divided before the sum.

`tmp/a12fix/arms.py` runs those as separate arms over the derived §05 pin population restricted to the
one-round geometry (228 head-rows, lanes 3..32), each measured *against the pinned form*, because the
pinned form is what the two engine sides agree on today:

| arm | worst 1-cos vs the pinned form | worst meanrel | elements differing after BF16 (of 512) | total differing |
|---|---|---|---|---|
| `w0` the shipped wide form | 1.093e-05 | 4.266e-03 | 154 | 129 280 |
| `w1` = `w0` but `l` summed from the F16 P | 1.093e-05 | 4.266e-03 | 153 | 129 256 |
| `w2` = `w0` with an **F32 accumulator** | 9.319e-06 | 4.459e-03 | 153 | 127 888 |
| `w3` = `w1` + `w2` | 9.319e-06 | 4.482e-03 | 153 | 127 702 |
| `ref` (the §05.5 form) vs the PIN | 2.220e-16 | 0 | 0 | 0 |

Three things follow, and two of them kill a hypothesis:

* **The deferred divide is arithmetically exact, so it cannot be the mechanism.** The epilogue computes
  `x = fattn_h_get(acc)/l` in F32 and hands it to `finish_head(..., 1.0f, x, d)`, which evaluates
  `bf(fb(x/1.0f))` = `bf(fb(x))`. Passing `(l, acc)` instead and letting `finish_head` do the single
  divide evaluates `bf(fb(acc/l))` — the same F32 divide, the same single BF16 rounding. Confirmed on
  device in §1.2: the restatement is byte-identical on every dump.
* **The denominator inconsistency is real but ~nothing.** `w1` moves 24 elements out of 933 888
  (0.0026 %). The mismatch it fixes — a term the F16 `P` flushes to zero still sitting in the divisor —
  needs an `exp` below ~6e-8, i.e. a score 16.6 below the row max, and the pinned population does not
  contain one. Confirmed on device in §1.3 (byte-identical `sampled.txt`, pooled figure unmoved).
* **The accumulator format is the only lever of any size, and its ceiling is −15 %, and it is A12's
  design point.** `w2` is the *bound* on what any accumulator change can buy at the attention level:
  1.093e-05 → 9.319e-06. The wide block's F16 accumulator is what `FattnO16` exists for —
  `engine/attn_fattn_kernels.h`: "the SAME DH of output in HALF the accumulator registers … 8 tiles × 8
  VGPRs = 64 a lane, against 16 × 8 = 128", and A10 measured that the reservation does not actually
  shrink when the accumulator does. Swapping it back to F32 is not an epilogue fix, it is a re-funding
  of A12's register premise, and it buys at most 15 % against a figure that is 1.4 % outside the band.

**Read of the whole table:** the wide form's distance to the pinned form is *dominated by the pin's own
rounding* (a `bf16` grid on the normalised weight, which the wide form cannot reproduce because an
online softmax does not know `l` when it rounds `P`, and because the WMMA atom takes F16 operands, not
BF16). `w2` — an F32 accumulator, strictly more accurate than the pin — is *not* closer to the pin than
`w0` by more than 15 %. The excess the gate is charging the wide arm with is therefore the price of
*not being the pinned rounding*, and it is not removable inside `engine/attn_wide_kernels.h` by any
change short of undoing the F16 accumulator that A12 was funded to spend.

<!-- APPENDED BELOW, per item -->

## 1. Item 1 — the wide arm's pooled cosine on `smoke`: four hypotheses, none survives, STOP

**The red** (`gate.py --fixture smoke --attn-fattn-wide 1`, rc=1):

```
[FAIL] batched and carry pool to a cosine inside the cascade band (>= 0.9998) — 19 rows, 0 of them still
       bit-identical; pooled meanrel 1.775e-02 (band 3e-02), pooled 1-cos 2.028e-04 (band 2e-04);
       worst per-row 1-cos 9.179e-04 at logits0.bin (band 2.75e-03)
```

Every other reading of the same run is inside with room: per-row worst 9.179e-04 against 2.75e-3 (3.0×),
pooled meanrel 1.775e-02 against 3e-2 (1.7×), the batched-vs-one-row pairing 1.622e-04 inside the same
2e-04 line. The miss is 1.4 % of the band, and §0.2 says it needs ~12 % of the *differential* removed.

**Which rows carry the excess (the brief's first question): none of them individually.** §0.1, recomputed
from the gate's own dumps: rows **0 / 14 / 10** contribute c = 2.804e-05 / 2.393e-05 / 2.286e-05 of
Σc_i = 1.8178e-04 (15.4 % / 13.2 % / 12.6 %; the three together 41.2 %), and all nineteen rows
contribute. Per-row figures are 3× inside the row band. So there is no row-local defect to fix, and the
excess is the sum of a small, uniform, all-elements movement — which is what the format analysis in §0.3
predicts.

**The mechanism (the brief's second question): the leading hypothesis is refuted by experiment, and so are
the three that replaced it.** Each was implemented, built, and run on device through the gate:

| # | hypothesis | where it was changed | result on device |
|---|---|---|---|
| E1 | the epilogue stages `acc`/`l` with `l = 1.0f` and divides once, where the shipping path divides per round (the brief's leading hypothesis) | divide moved into `finish_head`'s own `acc/l`, the row-slot's `l` fetched by `__shfl_sync` | **byte-identical dumps on all three sides** (0/19 differing logits files), pooled figure unmoved at 2.028e-04, rc=1 |
| E2 | the divisor keeps terms the F16 `P` flushed to zero — numerator/denominator inconsistency | `l` accumulated from the same F16 values the atom saw | **worse**: 2.059e-04, 18/19 files differ |
| E3 | the F16 accumulator | F32 accumulator | **not tried**: A12's funded design point (`FattnO16`, 64 VGPR/lane against 128); §0.3 bounds its ceiling at −15 % |
| E4 | the atom is fed an 11-bit grid where the pin feeds a 7-bit one — the *finer* grid transmits more of the upstream difference | `fattn_wide_p_grid`: round `e` to the BF16 mantissa grid (RNE on the F32 bits), store in the F16 operand | **worse on both pairings**: batched-vs-carry 2.082e-04 *and* batched-vs-one-row 2.153e-04, two FAILs where HEAD had one |

E1 is arithmetically exact, which is why it moved nothing: `bf(fb(acc/l))` is the same F32 divide and the
same single BF16 rounding whichever of the two sites performs it. E2 moved the divisor *away* from the
pin, and the reason is on the reference's side: torch's softmax sums F32 exponentials and casts after,
so the wide form's F32 `l` over the un-rounded exps is already the pin's rule. E4 is the interesting
failure, and it retires the CPU method used to find it:

> The mirror (`tmp/a12fix/arms_grid.py`) said E4 was worth −14 % at the attention stage — worst 1-cos to
> the pinned form 1.093e-05 → 9.423e-06, the same size as the F32 accumulator, and E4+F32-accumulator
> together bought nothing further (9.426e-06), i.e. the two arms were paying for **one** deviation, the P
> grid. On device it moved the *gate's* figure the wrong way. The reason is a metric difference, not an
> arithmetic one: the mirror measures each arm's distance to a **fixed** reference; the gate measures the
> distance **between two cascade-perturbed sides**. A change that moves a form closer to the reference can
> still enlarge the side-to-side difference, because what the pairing sees is how much of the upstream
> difference survives, not how faithful either side is. **Any future A12 rounding work must price the
> differential, not the distance.**

**Why nothing short of a design decision clears it.** The wide form's distance to the pinned one-pass form
is dominated by the pin's own rounding — a `bf16` grid applied to the *normalised* weight — which an online
softmax cannot reproduce (it does not know `l` when it rounds `P`) and an F16 WMMA atom cannot feed (it
takes F16 operands). E3, an F32 accumulator — strictly *more* accurate than the pin — closes only 15 % of
that distance, and 12 % is what the band needs, so E3 is the one lever with enough headroom and it is a
reversal of the premise A12 was funded on. That is a decision for the direction owner, not for a kernel
edit, so per the brief this item **stops here with no code change**. The three options, sized:

1. **A band decision.** The line to beat is 2.028e-04 as HEAD stands. The honest sizing instrument already
   exists — `engine/wave_band_floor_probe.py --gemm-pins all` is what sized the row band, and the gate's own
   PASS text names it as the re-take that would restate the table. Context for whoever decides: at knob 0
   this same fixture's batched-vs-carry reads **1.790e-04, i.e. 89.5 % of the band**, so the band leaves the
   shipping path 10.5 % of headroom and knob 1 costs 13.3 %. Per AGENTS' numeric-format rule a widened band
   must still catch every geometrically derived must-catch injection — that test is the gate's own, not a
   mood.
2. **Re-fund the accumulator** (F32, `FattnO16` → F32 fragments, 64 → 128 VGPR/lane). Measured ceiling
   −15 %, needed ≈12 %, so it *may* clear — but E4 is the cautionary case: the figure must be taken on
   device through the gate, not off the mirror.
3. **Give knob 1 a coverage rule** so rows at or below `MAXSC` stay on the pinned one-pass form. Every
   `smoke` row is ≤19 lanes, so under this rule knob 1 would be byte-identical to knob 0 on `smoke` —
   1.790e-04, PASS, bands untouched. It contradicts `A12-WIDE-BET`'s "the wide form owns every row"
   premise (and the timing rig priced it there), so it re-opens the design question and needs re-pricing.

**rc, both arms, before and after this pass** (`smoke`): wide=0 rc=0 → rc=0; wide=1 rc=1 → rc=1. No engine
file was changed — `git diff --stat -- engine/attn_wide_kernels.h` is empty, the four experiments are all
reverted, and the red stands.

## 2. Item 2 — `chunk64b` default arm, `decode row 120: 8337 vs 20619 with room 2.925 BF16 ULP`

**Verdict: a fixture/floor artifact, not an engine defect. No threshold was moved and no code was
changed.**

Reproduced first: `gate.py --fixture chunk64b --attn-fattn-wide 0` on today's HEAD build gives the identical
FAIL line, rc=1 (wall 13 s, util 39 %, no timing taken) — and the admission had already witnessed it with
`tmp/a12plumb/decode_run.prechange`, so it is pre-existing and nothing in this package caused it.

First thing the row needed was the pin decoded correctly: `decode0.logits_last.npy` stores **BF16 bits in a
`uint16` array** (the `.to()`/`.view()` trap this repo already records), so the magnitudes only appear via
`torch.from_numpy(bits).view(torch.bfloat16)`. Then, from the gate's own dumps
(`tmp/a12fix/chunk_row120.py`, `chunk_vs_pin.py`):

| quantity | value | reading |
|---|---|---|
| pin top-1 / top-2 at row 120 | 8337 = 15.625 / 20619 = 15.500, step 0.0625 | the reference decides by **exactly 2.0000 BF16 ULP** |
| storage uncertainty on that room | ±1.0 ULP (each pin value carries ±½ step) | the 1.50 line sits **inside** the pin's own resolution at this row |
| engine room, batched / one-row | 2.9248 / 2.9514 ULP | both sides spread the gap slightly, symmetrically |
| row 120 per-row 1-cos (batched vs one-row) | 1.159e-04, rank **18 of 150** | 24× inside the 2.75e-3 row band — no stage differs *at this row* |
| median whole-row max\|d\|, this pairing | 0.179 ≈ 2.9 ULP at this magnitude | the legal width movement at row 120's magnitude is **larger than the gap that decides it** |
| max\|d\| vs the pin at row 120, batched / one-row | 0.2089 / 0.2509 | against 30-row medians 0.216 / 0.209 — neither side is an outlier |
| token agreement | batched == pin (8337); one-row == the pin's own runner-up | the flip is the one-row side landing on the pin's second choice |

And the population context: at wide=0 the gate compares **150 rows** (prefill 0–119, decode 120–149) for
this pairing and admits **five thin-room flips** — prefill rows 24 / 70 / 71 / 97 at rooms 0.1327 / 0.02029
/ 0.06161 / 0.05206 and decode row 131 at 0.00911 BF16 ULP — of which row 120 at **2.925** is the only one
above the line. Read the pin's side of the same 30 decode rows: rooms min **0.000** (row 131 — an exact tie
in the reference), median 8.000, max 50.500, four of thirty below the line. Row 120 is the same class one
step up: its deciding gap in the reference is 2 BF16 steps while the *legal* width cascade at that
magnitude moves individual logits by ~2.9 ULP, so the flip is legal by the argument the rule already makes
— it merely reads above a line sized by a census that no longer matches the build.

**The proof of that, taken while filling in the missing arm (wide=1, rc=0 PASS).** The same row, the same
pin, and a different *legal* form on the batched side:

| arm | pairing | row 120 | room |
|---|---|---|---|
| wide=0 | batched vs one-row | 8337 (the pin's token) vs 20619 | **2.925 ULP** → above the line, **FAIL** |
| wide=1 | batched vs one-row | the two sides agree; row 120 is not a flip at all | — |
| wide=1 | batched vs carry | 20619 vs 8337 — the direction **reversed** | **0.812 ULP** → thin-room, admitted |

So the room at this row is not a property of the row: it is a property of the pairing, and it moves by a
factor of 3.6 — enough to cross the line in one direction and sit under it in the other — between two
forms that are both legal under the existing bands. A row whose "room" is that form-dependent cannot be a
defect claim about either side; it is the floor, seen at a row the fixture happens to decide by two BF16
steps. (`prototype/attn_wide/attn_decode`'s §18 lesson applies verbatim: what is missing here is not a
stricter check but *attribution* — the red belongs to the pairing, not to the form or the engine.)

Why the line reads 1.50: the gate's own census text — worst legal cross-width flip 0.9985 ULP over 2951
rows, "the same pairing against the one-row side puts its worst at 0.6224 ULP (**chunk64b**/wide520f row
55)". Same fixture, same pairing, today: **2.925**. That is 4.7× the census's recorded worst, which makes
the census — taken 2026-10-16, before D30's cast reduction, D35/D36's pin promotion and the attention
wave — the stale artifact, not the row.

**What would make it green, and why I did not do it.** Either `flip_room_max_ulp` moves (forbidden, and it
would be loosening the check that the five admitted flips depend on), or `engine/wave_flip_census.py` is
re-taken over the nine fixtures × both routes with today's binary and the line is re-derived from the
result. The re-take is a device pass far above a minute and a measurement, so it is the main agent's; the
command is handed over in §4. Note the re-take is not guaranteed to raise the line: it may instead show
that the legal cascade genuinely grew since the census, which is a *different* finding worth having.

**rc, both arms, before and after this pass** (`chunk64b`): wide=0 rc=1 → rc=1 (the row-120 flip line,
unchanged); wide=1 rc=0 → rc=0 (**PASS** — 1 step reported TIE, 1 reported SLACK; wall 14 s, util 92 % →
37 %, no timing taken). The arm asymmetry is not an accident and is the evidence in the table above.

## 3. Item 3 — `zz`: reproduced, root-caused, and now self-describing

**Reproduced with HEAD's own gate.** `engine/gate.py` on disk is byte-identical to HEAD (`git rev-parse
HEAD:engine/gate.py` = `git hash-object engine/gate.py` = `79fda037…`), so these runs *are* HEAD's gate.
Both arms: rc=1, wall 0–1 s, and **no verdict line at all** — the gate refuses before `decode_run` starts:

```
[FAIL] every declared logit gap is actually absent from the fixture (the declaration and the disk agree,
       in both directions) — declares a step whose logits are present: decode2, decode3
gate: zz declares 2 step(s) with no pinned logits (decode2, decode3), so this gate cannot run the engine
      against it … [expected-red] none recorded for zz, so all 1 failure(s) above are new by construction
```

**Root cause: the fixture contradicts itself, and it is not about attention.**
`data/refs/zz/manifest.json` declares `logits_declared_absent: [decode2, decode3]` and its 524-entry `pins`
carrys logits for exactly three names (`logits_last`, `decode0.logits_last`, `decode1.logits_last`) — but
the directory holds **2500** `.npy` files, 1976 of them undeclared, among them `decode2.logits_last.npy`
and `decode3.logits_last.npy`. Those two are F32 and their argmaxes are 760 and 6511, which are precisely
`greedy_continuation[3]` and `[4]`: real logits the manifest stopped claiming. Their mtimes (13:32:38)
precede the manifest and `decode1`'s logits (13:33:12) by 34 s — the full dump's files, left in place when
the lean pin policy was written over them. The check that fires is correct as written; the consequence,
though, was that `zz` had **never judged a step on any arm**, in any mode, and its rc=1 read as a numerics
result in the admission table. That is the same class as the empty-parse failures this tree keeps recording:
a red that cannot be attributed is a red nobody reads.

**Change made (declaration, not deletion).** Three edits, none of them a band, a threshold, a must-catch
set, the fixture population, or a destructive action on a pinned directory:

1. `engine/gate.py` — `EXPECTED_RED` gains its first entry since it emptied, with the failing check name,
   the numbers above, and why the remedy is not a gate edit. The mechanism's rule is unchanged: `zz` is
   still FAIL and still exits 1 — a label, not a pardon — and if it ever heals the entry prints `healed`.
2. `engine/gate.py` — the declared-gap refusal is a `SystemExit` that walks past `report()`, so it now
   appends the recorded reason where the reader actually is. Before: bare `gate: … cannot run the engine`.
   After: the same refusal plus `[expected-red] zz is on the recorded list: the fixture contradicts
   itself, not the engine: …`.
3. `prototype/attn_wide/a12_engine_admission.py` + `run_a12_admission.sh` — the harness now separates a
   **declared** red from an **undeclared** one (only the latter is a harness failure), prints the reason
   *from* `gate.py`'s own list rather than a second copy of it, prints `healed` if a declaration stops
   firing, and the device table row now carries the refusal/reason instead of a bare `EXIT 1 (no verdict
   line)`. Witness (`tmp/a12fix/check_declared.py`, which runs the real code path over a two-fixture
   population): `zz EXIT 1 [DECLARED] the fixture contradicts itself, not the engine: …` and
   `1 of 2 fixture(s) red at --no-engine and DECLARED above, so the population is understood`.

**Deliberately not done:** no fixture surgery. The two real remedies belong to the pin owner — re-dump zz
with `tools/reference_dump.py --logit-steps all` (a device + HF pass, and A7 declined zz's re-dump because
its `state_fp` re-dump would break the fingerprint re-admission of the other eight), or prune the 1976
undeclared files, which is **irreversible** (`data/` is gitignored at `.gitignore:65`; `git ls-files
data/refs/zz` lists nothing, so there is no git copy to come back from). Clearing `logits_declared_absent`
by hand was rejected: it would make a manifest agree with the exact disagreement its own check exists to
catch. Until one of the two real remedies lands, `zz` judges zero steps on either arm, and every surface
that shows that row now says why.

**rc, both arms, before and after**: wide=0 rc=1 → rc=1, wide=1 rc=1 → rc=1 (unchanged by design); what
changed is that the row is attributed instead of mysterious.

## 4. Knob-0 witness, the device runs, and what is handed over

**Knob-0 byte-identity witness — PASS**, after all experiments were reverted and the tree rebuilt:

```
PRE=tmp/a12plumb/decode_run.prechange FIXTURE=wide520f OUT=tmp/a12fix/witness2 \
  bash prototype/attn_wide/a12_engine_witness.sh
  files compared: 4, differing: 0
  WITNESS PASS: knob 0 is byte-identical to the pre-change binary on every dump
  geometry lines identical
  ATTN_WIDE form=wide128 warps=8 threads=256 pos_per_block=32 rows_per_block=128 stage=64 ktiles=16 lgroups=4 acc=f16 …
  sampled differ B/C (expected: the wide round re-associates the softmax)
util at acquire: gpu use: 91 %   util at release: gpu use: 27 %   — no timing taken
```

**Every device run this pass made** (correctness-only gate runs, one fixture at a time, each inside one
held lock; `no timing taken` — none of these walls is a measurement and none may enter a timing doc):

| command | rc | wall | util at lock |
|---|---|---|---|
| `gate.py --fixture smoke --attn-fattn-wide 1` (HEAD, before) | 1 | not recorded | not recorded (log: `tmp/a12fix/smoke_w1.log`) |
| `gate.py --fixture smoke --attn-fattn-wide 0` (HEAD, before) | 0 | not recorded | not recorded (`smoke_w0.log`) |
| `gate.py --fixture chunk64b --attn-fattn-wide 0` (HEAD, before) | 1 | 14 s | 39 % |
| `gate.py --fixture chunk64b --attn-fattn-wide 1` (restored build) | **0** | 14 s | 92 % → 37 % |
| `gate.py --fixture zz --attn-fattn-wide {0,1}` (before the declaration) | 1 / 1 | 0 s / 0 s | 72 % |
| `gate.py --fixture smoke --attn-fattn-wide 1` (E1, divide in `finish_head`) | 1 | not recorded | not recorded (`gate_e1/`) |
| `gate.py --fixture smoke --attn-fattn-wide 1` (E2, `l` from the F16 P) | 1 | not recorded | not recorded (`gate_e2/`) |
| `gate.py --fixture smoke --attn-fattn-wide {1,0}` (E4, P on the pinned grid) | 1 / 0 | 7 s / 6 s | 21 % |
| `a12_engine_witness.sh` (above) | 0 | ~10 s | 91 % → 27 % |
| `tmp/a12fix/run_final_matrix.sh` = smoke{1,0}, chunk64b{0}, zz{0,1} on the restored build | 1, 0, 1, 1, 1 | 6 / 7 / 13 / 1 / 0 s | 91 % → 31 % |

Every wall above is a gate run's own elapsed seconds, printed to show the run fits a worker's correctness
budget; none is a measurement, and the earlier three rows' attribution lines were not captured — their logs
are named so a reader can see what was run rather than being given a number that was not taken.

**Handed to the main agent (device, blocking, zero subagents):**

1. Item 2's census re-take, the only thing that can legitimately move the flip line:
   `engine/wave_flip_census.py` over the nine fixtures × both routes with today's binary
   (`gate.py --gemm lt` dumps), then re-derive `flip_room_max_ulp` from the result — quoting both the old
   (0.9985 / 0.6224) and new worst flips.
2. If item 1 is going to option 1: `engine/wave_band_floor_probe.py --gemm-pins all` as the re-take that
   restates the pooled stage cosine, with the must-catch set re-run against whatever number comes out.
3. If item 1 is going to option 2: an F32-accumulator wide variant built and gated — the gate's number is
   the verdict, not the mirror's.
4. `zz`: a decision on re-dump versus prune. A prune needs an explicit yes from the user, because
   `data/refs/zz` is untracked and the deletion does not come back.
5. The full both-arms admission when the above lands: `bash prototype/attn_wide/run_a12_admission.sh`
   (it now prints a reason for a no-verdict row).

**Files this pass left behind.** `engine/attn_wide_kernels.h` — untouched (all four experiments reverted,
verified byte-identical to HEAD and to `tmp/a12fix/attn_wide_kernels.h.orig`). `engine/gate.py` — the `zz`
expected-red entry and the refusal-message addition, nothing else. `prototype/attn_wide/a12_engine_admission.py`,
`prototype/attn_wide/run_a12_admission.sh` — the declared-red handling. This file — the diagnosis.
Scratch scripts and dumps live in `tmp/a12fix/` (gitignored): `rowcontrib.py` `crossmatrix.py` `whichrows.py`
`arms.py` `arms_grid.py` `chunk_row120.py` `chunk_vs_pin.py` `zfacts.py` `check_declared.py`
`check_selftest_gaps.py` `run_final_matrix.sh` `run_chunk_w1.sh`, and the gate work dirs `gate_*`, `grid_*`,
`final_*`, `witness2`. The two CPU witnesses for the gate edit are `check_selftest_gaps.py` (the six
declared-gap injections, all six PASS after the edit) and `check_declared.py` (the harness's declared-red
path, run over a two-fixture population so it costs seconds rather than re-reading eighteen fixtures).

**Residual risks.** (a) The pooled band is now known to sit 10.5 % above the shipping path's own reading on
`smoke`; whichever way item 1 resolves, that tightness is the real finding and it will bite the next form
knob too. (b) Item 2's census is stale by four decisions, so every flip verdict in the tree is currently
being judged against a 2026-10-16 margin. (c) `zz` still adjudicates nothing — the declaration makes it
legible, not judged. (d) E4 showed the CPU mirror cannot price the gate's quantity; any A12 rounding claim
made from it should be treated as unpriced until a gate run backs it.

