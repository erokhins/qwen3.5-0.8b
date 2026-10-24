# A9-L-c — the bet: cap the attention split, and the 105 GB/pass of scratch that the split costs stops being paid at 32k

Pre-registered **before any measurement**, in the shape of `prototype/attn_percost/A4-BET.md` and
`prototype/attn_percost/A6-BET.md`. Funded by `progress/2026-10-21/performance-status-2026-10-21.md` §52 (decision: do L-c now, keep the
rewrite pre-gated at a number rather than funding it on hope). Written against tip `8b0a3b4`; the knob is
`engine/decode_run.cpp --attn-split-cap N`, applied at `engine/kernels_attn.cpp`'s one split decision site.

**No timing in this file.** Every millisecond and byte figure is quoted from
`prototype/attn_percost/A9-LONGCTX-CENSUS.md` §2.3/§4.2 (which quotes `progress/2026-10-21/performance-status-2026-10-21.md` §51's traces)
or is arithmetic on those with its formula shown. The main agent measures; the worker who wrote the knob took none.

---

## The claim under test

The split path — `split = ceil(lanes/MAXSC)` lane blocks per wide row, each writing a `(m, l, acc[256])` partial,
`attn_combine` merging them — moves **105.46 GB written + 105.46 GB read back per 32,768-lane pass**, and
`attn_combine` was traced reading its half at **208 GB/s = 88 % of this part's measured 237 GB/s ceiling**
(A9 §2.3, whose witness is `a9_trace_kernels.py` against §51's trace). It is the one item in that census whose
rate is *measured* rather than modelled, which is why this is the only traffic-shaped lever A9 was willing to fund.

Capping the split at `N` keeps every lane (the slice arithmetic is `span = ceil(lanes/split)`, so a shallower
split makes each block's slice **wider** and its round loop longer — verified on CPU, §Gate 0 below) and cuts the
partial round-trip in proportion:

| cap | round-trip GB/pass at 32,768 | widest-body serial depth vs uncapped | source |
|---|---|---|---|
| 64 (= today, and 0 = no cap) | 105.457 | 0.00 % | A9 §4.2, reproduced by `a9_waste_model.py --caps --cap-width 32768` at the time of writing |
| 32 | 78.674 | +2.13 % | ibid. |
| 16 | 45.805 | +2.13 % | ibid. |
| **8** (the funded candidate) | **24.500** | **+2.13 %** | ibid. |

Prediction from the census: **−4.2 … −4.7 % of the 32k wave**, i.e. −5.9 … −6.6 % of attention (denominator:
attention is 70.9 % of the 32k wave, A9 §4).

## Gate 0 — the cheap falsifier, and it runs first (CPU only, already run)

`prototype/attn_percost/a9_lc_geom_check.py` (this bet's only new tool). Four arithmetic claims, one
source-order claim and one band check, each with an injection that must turn it red:

* **[A]** a capped split tiles every row exactly — no lane lost, none double-counted, for every body of every
  width in `{512 … 40,000}` × every cap in `{0,2,3,4,5,8,16,32,64,129}`. **1,970 plans, 0 defects.**
* **[B]** the `part` scratch is sized from `kAttnMaxSplit` (`engine/kernels_attn.cpp`), so the reachable slot
  index stays inside it for every legal plan: worst index 262,143 of 262,144 slots. **0 defects.**
* **[C]** which `(cap,width)` pairs push a block's slice past `MAXSC` — the domain where only the WMMA wave may
  be asked to run a cap. Printed as a census, e.g. cap 8 at 32,768 lanes → **span 4,096**.
* **[D]** the lane-ceiling throw reads the **uncapped** split and precedes the cap, which is what keeps
  `MAXSC × kAttnMaxSplit = 32,768` the engine's ceiling whatever a caller asks for.
* **[E]** the merge re-association itself, on `attn_tile_model`'s imported reductions, judged by the imported
  `CASCADE_BAND`: **7 capped rows, 0 outside** (`meanrel ≈ 3–6e-7`).
* `--inject` turns all four controls red → **INJECT PASS** (floor-divided span, strided slot against a
  run-sized scratch, ceiling compared on the capped value, a partial that divided inside its own block).

Gate 0 is what "does capping the split break the lane ceiling" resolved to; see §Domain below and
`prototype/attn_percost/A9-LC-impl.md` §2 for the citations.

## The candidates

| arm | what it is | funded? |
|---|---|---|
| `--attn-split-cap 0` | **the shipped default**: no cap, `split = split_for(lanes)` verbatim, byte-identical to `8b0a3b4` | the NULL arm of every window |
| `--attn-split-cap 64` | today's behaviour by another spelling: `min(split_for, 64)` cannot bind, because `split_for` above 64 is already a refusal | equivalent to 0, kept so the A9 §4.2 table's row 64 is reproducible |
| `--attn-split-cap 32` | −25 % of the round-trip | a fallback only if cap 8 comes back DISPLACED |
| `--attn-split-cap 16` | −57 % | ibid. |
| **`--attn-split-cap 8`** | **−77 % of the round-trip; A9 §4.2's funded candidate** | **THE candidate this bet ships or kills** |

One candidate per window. `CAP=16 bash prototype/attn_percost/run_ab_a9lc.sh` re-points the same harness at
another row of the table; the canary colours are derived from `$CAP`, so a re-pointed harness cannot silently
judge the old candidate.

## Lines and denominators (both at 32,768 lanes, both measured in ONE window against ONE base)

| line | quantity | live | dead / falsified |
|---|---|---|---|
| **LINE 1** | `WAVE gran=stage family=attn ms` at 32,768 lanes (untraced) | ratio ≤ **0.9700** | ratio > 0.9700 |
| **LINE 2** | `WAVE gran=stage TOTAL wall_ms` at 32,768 lanes | ratio ≤ **0.9700** | ratio > 0.9700 [SHIP GATE] |
| NULL arm | the changed binary with the cap **unnamed**, against base | both lines \|move\| ≤ 2 % | > 2 % ⇒ **UNPAIRED**, no verdict printed |
| spread | per-rep spread of LINE 1, each arm | ≤ 3 % | > 3 % ⇒ window unpaired, re-take |

The verdict is one of four, and the harness prints exactly one (`run_ab_a9lc.sh`'s own summary):

* **SHIP** — both lines ≤ 0.9700.
* **FALSIFIED** — LINE 1 > 0.9700. Then A9 §2.3's 88 %-of-roof reading is retired for this kernel: the scratch
  round-trip is hidden behind compute after all, and D21 ("demanded bytes are not DRAM bytes") wins on this item
  even though it is the measured one. Do not re-run at another cap hoping for a different sign.
* **DISPLACED** — LINE 1 clears and LINE 2 does not: attention got faster and the wave did not, so the saving
  landed in a neighbour. **Not shipped as a wave lever.** This is why there are two lines at all: a single family
  row can be fooled by cost moving next door (the AGENTS trap row on single-family lines). The harness prints the
  externality (Σ over every non-attn family of `cap − base`) so DISPLACED names the neighbour rather than
  describing it.
* **UNPAIRED / spread too wide** — no verdict; report the NULL row and re-take.

Per `progress/2026-10-21/performance-status-2026-10-21.md` §36's convention the base is measured **in the same window** (the harness pins
`BASE=8b0a3b4`, builds a worktree of it, and refuses a base whose binary already knows the flag), both arms are
inside one `tools/gpu_lock.sh` hold, and ≥ 2 reps each.

## Numerics — declared in advance, because this is NOT bit-identical

The cap removes partials, and `attn_combine` merges `a.split` of them; fewer partials is a **different
association of that F32 merge**. So:

* **No bit-identity is owed and none is claimed.** The applicable published band is the one for a *reassociating
  route*: `engine/gate.py:894-904`'s `CASCADE_BAND` — per-row `1 - cos ≤ 2.0e-3`, per-stage
  `meanrel ≤ 3.0e-2`, and a flipped token admissible only with `< 1.5` BF16 ULP of top-1/top-2 room. The
  tighter pin floors (`specification/10-reference-dumps-and-gates.md` §10.1) are the other side of the judge, via
  `engine/gate.py`, `prototype/attn_tile_model.py`, and `prototype/attn_wmma/run_fattn_admission.sh` — all
  unchanged, none edited by this bet.
* **The merge re-association is priced, not assumed**: `a9_lc_geom_check.py` **[E]** runs a row through
  `attn_tile_model`'s own `online`/`merge_partials` at the natural split and at the capped one. The whole
  re-association is **`meanrel ≈ 3–6e-7`, `1-cos ≈ 0`** — four orders of magnitude inside the band. What the
  device measures is larger (worker gate, §`A9-LC-impl.md` §5: `1-cos 8.3e-4`, `meanrel 1.4e-2` at 4,608 lanes
  after a full prefill) because the difference propagates through 48 layers and a BF16 round each — that is the
  chaos floor `ASSUMPTIONS.md` A-18 records, not a merge defect.
* **Which field adjudicates what, learned here**: the mis-normalisation injection in **[E]** (a partial that
  divided inside its own block, the §07.3 bug) leaves **`cos = 1.000000` and moves `meanrel` to ~2** — a pure
  scale error is invisible to cosine. Any judgement of a capped run must read `meanrel` and the top-1 room, not
  cosine alone. Same class as the AGENTS row "max-ULP bands are blind to whole-tensor 1-ULP error".
* **Per the standing numeric-format rule** (`AGENTS.md`, "Numeric format: delegated"): a reassociation at this
  level is admissible **iff the existing gates stay green**. If a gate goes red, the question is whether the band
  can be **widened without losing a must-catch** — and the must-catch set is the geometry-derived one
  (`prototype/attn_tile_model.py`'s `neg_expect`, `partalias`, the wide anchors and the 512/513/514/515-lane wave
  rows), not a hand-listed one. **If the only way to make a gate pass is to loosen a check that currently catches
  a real bug, that is reported, not routed around.** No band is edited by the person who wrote the knob.
* **Nothing may be `--ack-spill`ed**, and nothing is: the change adds no register and no LDS (§Domain).
* The default must be inert, and it is checked as a fact rather than asserted as a claim: cap 0 leaves `split`
  alone, and the harness's NULL arm reads the *geometry* (the `SPLITCAP split=` of the default arm must equal
  base's own banner `split=`) in the same window as the milliseconds.

## Domain — where a cap may be asked for at all (Gate 0's finding, and the bet's real risk)

`split` is not a work knob; it is how many contiguous slices a row is cut into, and a block's slice is
`span = ceil(lanes/split)`. Forcing `split = ceil(lanes/MAXSC)` is what makes `span ≤ MAXSC`, and **two of the
three wave forms stage a whole block slice in LDS**:

* `attn_slim_wave` and the frozen `attn_tile_wave`: `__shared__ hip_bfloat16 s_sc[ATTN_TILE_G][MAXSC]`, indexed
  by the slice-absolute `rr < ns == span` (`engine/attn_slim_kernels.h:157`, written at `:232`; same buffer in
  `attn_tile_body`). A cap that binds there is an LDS overrun whose device signature is a hang, not a crash.
* `attn_fattn_wave*` (the WMMA wave, the **shipping** path since `--attn-fattn` defaulted to 1): stages
  `FATTN_MLANES = 16` lanes a round and its round loop runs `lo_all … hi_all`
  (`engine/attn_fattn_kernels.h:771`), so a wider slice is more rounds and nothing else.
* `attn_core` (every one-row body, and `--attn-tile 0`'s wave): the streaming branch keeps a `tile`-sized window
  in `sc` and the one-pass branch is only reachable at `lanes_of ≤ MAXSC`, so no branch indexes `sc` by the span.

**So a binding cap is legal exactly where the WMMA wave owns the streaming rows, and the engine REFUSES it
otherwise** (`attn_decode_layer`, message naming the staging and the remedy). That is the same rule the `attn_m2`
prototype already enforces (`prototype/attn_m2/attn_m16.hip.cpp:1069-1075`), costs zero registers, zero LDS and
zero instructions, and does not touch the shipping configuration — which is `--attn-fattn 1`. The lane ceiling
`MAXSC × kAttnMaxSplit = 32,768` is **unchanged**: the ceiling throw reads the uncapped split, so 32,768 lanes
stays reachable and 33,280 stays a refusal, and `engine/gate.py::lane_ceiling()` needs no change (§Gate 0 [D]).

**Falsifier of the domain claim itself**, cheap and pre-registered here: a capped run that *loses* lanes would
show as a wrong answer on the widest row, not a slow one. It is caught by Gate 0 [A] (arithmetic) and by the
device witness in `A9-LC-impl.md` (a capped 4,608-lane prefill vs the same run uncapped: same last row, sane
logits, `span=576 > MAXSC`).

## Milestones that gate device time

| # | milestone | gate | status at registration |
|---|---|---|---|
| M0 | Gate 0 above, CPU-only | `a9_lc_geom_check.py` PASS and `--inject` PASS | **green** |
| M1 | engine builds clean at the default | `engine/build.sh`, `tools/attn_ceiling_proof.py` exit 0 (its [A] census must still account for every `kAttnMaxSplit` reader) | build green; ceiling proof is a main-agent command (handover) |
| M2 | the default is inert | the NULL arm of the A/B window (geometry + milliseconds within 2 %) | pending the window |
| M3 | the cap is expressible and correct on device | one short capped run at a width where the cap binds, PASS/FAIL, no timing | see `A9-LC-impl.md` §5 |
| M4 | the gates at the candidate cap | `engine/gate.py` at the default **and** the wide long-context checks at the cap | handover (main agent) |
| M5 | the paired window | `run_ab_a9lc.sh` → SHIP / FALSIFIED / DISPLACED / UNPAIRED | handover (main agent) |

## Standing harness requirements (all of them are in `run_ab_a9lc.sh`, and it self-tests them)

* **Behavioural canary, read in the measuring window.** `engine/build.sh` is not bit-reproducible here, so a
  byte-compare of two binaries proves nothing (the §13 trap: the first D30 1.7′ A/B timed a pre-change binary as
  its `new` side and reported a clean null with every in-binary check green). The knob therefore prints its own
  line — `SPLITCAP split= cap= lanes= natural= rows= span=` — once per distinct geometry, and the harness aborts
  unless: base prints **no** such line, the capped arm prints one whose `cap=`/`split=` equal the harness's own
  `$CAP`, whose `span > MAXSC`, whose `natural > $CAP`, and which agrees with the `ATTN_FATTN` banner's `split=`;
  and the null arm's `split` equals base's. `SELFTEST=1` colour-tests that abort against fixtures generated from
  the harness's own constants.
* **No wide prompt through argv.** 32,768 ids in one argv slot is 191,588 B against `MAX_ARG_STRLEN` = 131,072 B;
  it fails with shell exit 126 in 0 s and reads as "the engine crashes at 32k" (`progress/2026-10-21/performance-status-2026-10-21.md` §39).
  Wide legs use `--ids-file` and the file's field count is asserted.
* **No empty parse.** Every field is `need`-checked, the family table must parse one row per family name, and zero
  `SPLITCAP` records on the capped arm is an abort, not a pass.
* **No verdict from an unpaired window**, nor from reps spread by more than 3 %, nor from arms whose attention
  launch counts differ (§41 closed launch count as a variable; a cap moves a grid extent, not a launch count).

## What is closed, so this list is not re-litigated

* Whether the ceiling must move for L-c: **no** — Gate 0 [D], and `MAXSC × kAttnMaxSplit` is untouched.
* Whether `gate.py::lane_ceiling()` must know about the cap: **no** — same reason; see `A9-LC-impl.md` §4.
* Whether the second spelling of the split rule (`blocks_for_row`) needs a second edit: **no** — it reads
  `a.split`, and the launcher writes the capped value into that one field. There is exactly one decision site.
* Whether the partial scratch must be re-sized: **no** — it is sized from the ceiling, so a shallower split only
  leaves the tail of each row's slab unwritten (Gate 0 [B]).
* Whether L-b or L-a are implicated: **no** — different resources, and A9 §4 killed L-b on declared bytes.

---

## Verdict (appended after measurement; the lines above are unchanged, as pre-registration requires)

**DISPLACED** — LINE 1 cleared (attn family −4.61 %), LINE 2 missed (wave wall −2.38 %). Not shipped as a wave
lever; the knob stays inert and its inertness is witnessed byte-for-byte on device. Full record, including why the
apparent 70 ms `gdn_rec` externality is *unattributed* rather than a displacement:
`prototype/attn_percost/A9-LC-verdict.md`.
