# MAXSC-HANDOVER — the `>MAXSC` instrument gap: evidence and pricing

Worker task, 2026-10-22. Scope: **produce the evidence and the pricing**; the **ruling is the parent's**.
No millisecond figures taken here (quoted ledger figures are labelled as quoted). No commit. No edit to
`engine/performance-status.md`, `README.md`, `engine/generation-limits.md`, `progress/`, or any gate
threshold. No margin written by hand — every margin below comes from `engine/route_allowance.py` and its
band-ledger line.

Status of this file: live checkpoint, written before each long step. Sections are appended in the order the
task fixed them.

---

## 0. The parent's characterization, and where it is right and where it is wrong

The claim under test (`engine/generation-limits.md` §3, restated in the brief):

> the `>MAXSC` streaming route still misses the pinned reference's flat floors by `n_bad` counts at
> 524/648/1,032/4,104 lanes while cos and meanrel are 3–4 decades inside them — two instruments (flat
> §10.1 floors vs the wide fixtures' priced route allowance) disagree, and until that is ruled the
> `--split 18` geometry (worth ~0.36 ms) is held, not spent.

**CONFIRMED** by source and by the numbers in `prototype/attn_decode_wide/HANDOFF-D3.md` §2: the four
flagged widths are exactly the four wide fixtures' widest DECODE row —

| width | fixture | widest decode row |
|---|---|---|
| 524 | `wide520f` (prompt 500 + 24 steps) | `wide520f.decode23.L03` |
| 648 | `wide640f` (640 + 8) | `wide640f.decode7.L03` |
| 1,032 | `wide1024f` (1024 + 8) | `wide1024f.decode7.L03` |
| 4,104 | `wide4096f` (4096 + 8) | `wide4096f.decode7.L03` |

and cos/meanrel are indeed 3–4 decades inside the floors on all four, while `n_bad` (elements over 1 ULP)
and `ndiff` are not: 48/47/45/23 bad elements out of 2,048.

**REFUTED in three places.** Details and evidence in §1–§3 below; the short form, because the ruling should
not be built on the briefed version:

1. **The disagreement is not between the flat floors and a priced allowance.** The failing instrument is
   `prototype/attn_decode_wide/gate.py` section 2 ("AGAINST THE PINS"), which imports `compare` and
   `ndiff_limit` from `prototype/attn_decode/run.py` and applies the **flat op floors of this op** —
   `n_bad == 0`, `ndiff <= ndiff_limit(n)`, `cos >= COS_MIN`, `meanrel <= MEANREL_MAX` — to the
   **stage tensor `attn_pre_gate`**, not to the logits vector. That driver says out loud that it does not
   read a route allowance. So the two instruments that disagree are a per-element ULP/ndiff band on an
   intermediate tensor and a cos/meanrel aggregate on the same tensor; a route allowance is a
   cos/meanrel rule and **cannot adjudicate `n_bad` at all** — see §4.
2. **`wide520f`, one of the four flagged fixtures, already HAS a priced allowance** (2.12 / 0.39,
   per-fixture D-2) and is judged by it in `engine/gate.py` — yet it is still on the flag list, which is a
   second proof that the flag did not come from the allowance-vs-floor precedence in section (b).
3. **`--split 18` is not reachable on the shipping path at all.** `engine/gate.py` has no `--split` flag
   (its full `add_argument` set is at `engine/gate.py:7932-8397`); the engine's only lane-split knob is
   `decode_run --attn-split-cap N`, and `cap_split()` is `min(split, cap)` — it can only make a cut
   SHALLOWER. The natural cut at these widths is `ceil(lanes/512)` = 1/2/3/9, so a cap of 18 is **inert at
   all four** and produces the default bits. `--attn-split-cap 1` is refused outright by both
   `decode_run.cpp:586-597` and `kernels_attn.cpp:405-411`, and for a good reason: one lane block on a
   row wider than `MAXSC` is the mis-normalisation §07.3 exists to make unreachable (and the first probe
   of exactly that printed `cos 0.049` and `−0.013` — plausible numbers from an out-of-bounds LDS run,
   `HANDOFF-D3.md` §2). The deeper cut that won −66.5 % in the rig exists only in
   `prototype/attn_decode_wide/attn_decode_wide --split N`. Task step 3 as written ("run
   `engine/gate.py --fixture X` with the shipped `--split 1` and with `--split 18`") therefore cannot be
   executed, and **nothing in `engine/gate.py` can currently judge a split-18 binary**, because no such
   binary can be built. Escalated as a need-decision; see §5.

---

## 1. STEP 1 — the admitted population, derived by directory walk

Produced by `prototype/maxsc_allowance/census.py` (this directory). Population = every
`data/refs/<dir>/` that carries a `manifest.json`: **21 directories**, enumerated at run time, none typed
from memory. `MAXSC = 512` and `kAttnMaxSplit = 64` are read through `engine/gate.py::lane_geometry()`
from the two files that enforce them, not retyped. Reproduce:

```
.venv/bin/python prototype/maxsc_allowance/census.py --markdown
```

Run 2026-10-22, `MAXSC=512`, `kAttnMaxSplit=64`, lane ceiling 32,768 lanes:

| fixture | kind | prompt | steps | lanes prefill | lanes decode | allowance | margin | margin_rel | margin basis | judged by |
|---|---|---|---|---|---|---|---|---|---|---|
| chunk64b | pin | 120 | 30 | 120 | 150 | NO | — | — | — | one-pass |
| corpus48 | corpus record | — | — | — | — | — | — | — | — | not a pin set (`gate.NOT_A_PIN_FIXTURE`) |
| corpus48engine | corpus record | — | — | — | — | — | — | — | — | engine free-run record, not reference output |
| corpus48sdpa | corpus record | — | — | — | — | — | — | — | — | reference self-consistency floor |
| dec300f | pin | 295 | 128 | 295 | 423 | yes | 7.99 | 0.72 | own (D-2) | one-pass |
| dec64f | pin | 17 | 64 | 17 | 81 | yes | 3.50 | 0.42 | **carried (pre-D-2 global)** | one-pass |
| dec64t | pin | 27 | 64 | 27 | 91 | yes | 3.50 | 0.42 | **carried (pre-D-2 global)** | one-pass |
| dec8 | pin | 17 | 8 | 17 | 25 | NO | — | — | — | one-pass, flat floors |
| mt32f | pin | 232 | 32 | 232 | 264 | yes | 2.93 | 0.24 | own (D-2) | one-pass |
| p128 | pin | 128 | 8 | 128 | 136 | NO | — | — | — | one-pass |
| p300 | pin | 277 | 0 | 277 | — | NO | — | — | — | prefill-only, under MAXSC |
| p64 | pin | 46 | 0 | 46 | — | NO | — | — | — | prefill-only, under MAXSC |
| p8 | pin | 8 | 0 | 8 | — | NO | — | — | — | prefill-only, under MAXSC |
| prefix24dec8 | pin | 24 | 8 | 24 | 32 | NO | — | — | — | one-pass |
| smoke | pin | 17 | 2 | 17 | 19 | NO | — | — | — | one-pass |
| **wide1024f** | pin | 1024 | 8 | 1024 | **1032** | **NO** | — | — | — | **decode streams → FLAT FALLBACK**; prefill rows stream → flat floors, row block pinned |
| **wide4096f** | pin | 4096 | 8 | 4096 | **4104** | **NO** | — | — | — | **decode streams → FLAT FALLBACK**; prefill rows stream → flat floors, row block pinned |
| wide4096p | pin | 4096 | 0 | 4096 | — | NO | — | — | — | prefill-only; prefill rows stream → flat floors, row block pinned |
| wide520f | pin | 500 | 24 | 500 | **524** | **yes** | 2.12 | 0.39 | own (D-2) | decode streams → **priced allowance** |
| **wide640f** | pin | 640 | 8 | 640 | **648** | **NO** | — | — | — | **decode streams → FLAT FALLBACK**; prefill rows stream → flat floors, row block pinned |
| zz | pin | 5 | 4 | 5 | 9 | NO | — | — | — | one-pass (fixture self-contradictory, see `tools/fixture_census.py`) |

(The table above is the state at the census run. The 22:23:59Z pricing in §2 moved `wide640f` and
`wide1024f` from `NO` to priced — 1.79/0.29 and 1.62/0.35 — so re-run `census.py` for the live table
rather than reading this one as current.)

Counts: 18 pin fixtures + 3 corpus records. **5 priced allowances** (`dec300f`, `dec64f`, `dec64t`,
`mt32f`, `wide520f`) — exactly the set the brief named, confirmed rather than assumed.
**4 fixtures have DECODE rows above MAXSC**; of those **3 have no allowance** (`wide640f`, `wide1024f`,
`wide4096f`) and are therefore judged by the flat fallback in sections (b)/(b2) — the section the
allowance exists to serve. **4 fixtures have PREFILL rows above MAXSC**, and all four carry a
`logits_rows.npy` block.

**The structural fact the brief did not name, and it changes what "price the missing ones" can buy.**
`engine/gate.py` section (i) prints its own rule line:

> `no route allowance is priced for prefill rows (route_allowance.json prices decode steps)`

so pricing `wide640f`/`wide1024f`/`wide4096f` changes **nothing** about how their streaming PREFILL rows
are judged — those stay on the flat floors by design. What pricing buys is the rule in force on their 8
DECODE steps, which are indeed streaming rows (648/1032/4104 lanes). Any ruling that assumes pricing will
move the prefill-row verdict is aimed at the wrong section.

---

## 2. STEP 2 — what a properly priced allowance does, and it is not what the brief assumed

Two allowances were missing and are now written (`wide640f` 1.79/0.29, `wide1024f` 1.62/0.35, both by
`engine/route_allowance.py`, each with its `bench/band-history.jsonl` line). `wide4096f` was priced but
**not written**; §3 says why. The tables came from `engine/route_price.py`, which is CPU-only at these
widths — 4 s for `wide640f`, 5 s for `wide1024f`, 22 s / 2.3 GB RSS for `wide4096f`. The briefing expected
`wide4096f` pricing to be the expensive item; it is not, the device gate run is.

**Provenance note for the commit.** `.gitignore:65` is `data/`, so the two new
`data/refs/*/route_allowance.json` files are **not tracked by git** — only `bench/band-history.jsonl` (4
lines: two keys × two fixtures) carries the change into history, and the allowance files themselves travel
as regenerated reference data. A reviewer wanting to see the band files has to read them on disk, not in
the diff.

**Finding F1 — on the shipping route the priced allowance is inert, because the cascade floor is looser.**
`rule_for_step()` calls `cascade_floor()`, which takes `max(cap, CASCADE_BAND.row_1cos_max)` for any
library/`lt` route. Every derived cap here is 2.74e-05..6.44e-05, i.e. 31–73× *below* the cascade floor's
2e-03 / 3 %, so `max()` returns the floor and the allowance never binds:

| fixture | allowance cos cap, worst step | allowance meanrel cap, worst step | what actually binds on `--gemm lt` |
|---|---|---|---|
| wide640f | 6.4436e-05 (step 4) | 1.6761 % (step 4) | cascade floor: 2.0e-03, 3 % |
| wide1024f | 5.9698e-05 (step 6) | 2.2871 % (step 3) | cascade floor: 2.0e-03, 3 % |

The gate says so on its own line: `rule: route allowance for wide640f (… -> cos deficit <= 4.9634e-05 …);
floored at the measured cascade (cos deficit >= 2.0e-03, meanrel <= 3%)`. So the instrument the brief
expected to be "in force" after pricing cannot be in force on the default route at these widths. On the
`--gemm hand` route the floor is not armed and the allowance does bind — see F2.

**Finding F2 — the allowance is much tighter than anything it replaces, and that cuts both ways.** Against
the flat §10.1 G5 floor (5e-04) the derived cos caps are 7.8–18.2× tighter, and against the cascade floor
that a library route substitutes (2e-03) they are 31–73× tighter; on meanrel the factors are 2.2–7.9×
against the flat 5 % ceiling and 1.3–4.7× against the cascade 3 %. Pricing therefore *narrows* the band; it
does not widen it, which is the AGENTS.md test (its "Numeric format: delegated" section,
second bullet): a band that is narrower than the one it replaces keeps every injection the wider band
caught. §4 re-checks this against the §20 must-catch set rather than asserting it.

**Finding F3 — a priced allowance does not admit `--split 18`, because `--split 18` does not exist on the
engine path** (§0 point 3). Pricing changes the rule applied to the natural cut at these widths
(1/2/3/9 lane blocks); it buys no permission for a cut the launcher refuses to build.

---

## 3. STEP 3 — the gate runs, and one finding that outranks the task

Every run below is correctness-only, blocking, one at a time, GPU lock held and released around each; no
timing figure is taken from any of them and none may enter a timing doc.

| # | command (under `tools/gpu_lock.sh`) | wall | verdict |
|---|---|---|---|
| 1 | `engine/gate.py --fixture wide640f` (before pricing) | 41 s | PASS — flat floors, 8/8 steps inside 2e-03 / 3 % |
| 2 | `engine/gate.py --fixture wide1024f` (before pricing) | 60 s | PASS — same |
| 3 | `engine/gate.py --fixture wide520f` | 28 s | **FAIL** — stale provenance, not numerics |
| 4 | `engine/gate.py --fixture dec64f` | 13 s | **FAIL** — same |
| 5 | `engine/gate.py --fixture wide640f` (after pricing) | 40 s | PASS — `rule: route allowance … floored at the measured cascade` |
| 6 | `engine/gate.py --fixture wide1024f` (after pricing) | 60 s | PASS — 298 PASS lines |
| 7 | `engine/gate.py --fixture wide640f --gemm hand` | 43 s | FAIL — 16: 14 band lines caused by the new allowance, 2 pre-existing (proved by #9) |
| 8 | `engine/gate.py --fixture chunk64b --gemm hand` | 13 s | FAIL — 3, **no allowance exists for `chunk64b`**, so these are not allowance-caused |
| 9 | `engine/gate.py --fixture chunk64b --gemm hand --attn-fattn-wide 0` | 13 s | FAIL — 2. Attribution: the (h) bit-identity failures belong to the `hand` route at HEAD, not to any band |

`wide4096f`'s gate run (~240 s) is above the worker device budget and was **not** run. The engine column
used for its pricing table came from an existing log (`tmp/glue_copy_fold/ab/gate_new-on_wide4096f.log`,
2026-10-21 23:38, the fold-on build = HEAD's shipped defaults, engine bit-behaviour unchanged by the three
doc-only commits since). That is why its allowance is **not written**: a band whose engine column I cannot
re-take inside my own budget should not be written blind. Handover commands in §6.

### 3.1 The outranking finding: all five pre-existing allowances are currently STALE, so `engine/gate.py` is red on five of the 18 fixtures at HEAD

`check_inputs.py` (this directory) prints it in one pass:

| fixture | stored `specimpl_sha256` | margins | priced |
|---|---|---|---|
| dec300f | **STALE** | 7.99 / 0.72 | 2026-10-12T03:02:09Z |
| dec64f | **STALE** | 3.50 / 0.42 | 2026-10-12T01:17:04Z |
| dec64t | **STALE** | 3.50 / 0.42 | 2026-10-12T01:17:04Z |
| mt32f | **STALE** | 2.93 / 0.24 | 2026-10-12T03:02:09Z |
| wide520f | **STALE** | 2.12 / 0.39 | 2026-10-14T20:26:06Z |
| wide640f, wide1024f | MATCH (written this session) | 1.79/0.29, 1.62/0.35 | 2026-10-21T22:23:59Z |

The cause is not arithmetic. `allowance_inputs()` hashes the whole bytes of `speccheck/specimpl.py`, and
commit `6e6d88a` ("bench: the build queue, the gate audit and the pinned corpus leave bench/") changed one
**comment line** in that file — a citation re-point from `bench/gate-audit.md` to
`progress/2026-10-11/gate-audit.md`. The reference's behaviour did not move; five band provenances did.
Runs #3 and #4 above are that failure, and they are the only reason those two fixtures are red.

This is the same class as every other empty-provenance failure logged in AGENTS.md, one level over: an
instrument whose validity key includes bytes that cannot affect what it validates. Three repairs, all
band-policy and therefore the parent's call — none is mine to pick:

- **(a) hash the code, not the prose** — hash a normalisation of `specimpl.py` with comment-only lines
  stripped. Cheapest, keeps provenance for every change that can move a bit, and it is what the key is
  *for*. Risk: a comment edit that documents a behaviour change would no longer re-key.
- **(b) re-price all five.** Refuses for `dec64f`/`dec64t` without deleting the files: the tool will not
  overwrite and re-deriving today gives 3.04 against the stored 3.50 (D-2 round-up drift), so a re-price
  is a deliberate band move on three fixtures, not a refresh.
- **(c) leave it red** and accept that 5 of 18 fixtures currently abort on provenance. Anyone running
  `gate.py` on those fixtures gets a red that looks like a numerics result — which is precisely the
  instrument-gap-reported-as-numerics failure already logged twice in AGENTS.md.

Nobody noticed because the recent A/B gates used `dec8` and `wide4096f`, neither of which has an allowance.

---

## 4. STEP 4 — the ruling evidence, per fixture

For each of the four fixtures the flag names: which instrument is in force now, what it says, and what a
ruling would have to rest on.

| fixture | widest decode row | instrument in force for the 8 decode steps **now** | what the numbers say |
|---|---|---|---|
| `wide520f` | 524 | its own priced allowance (2.12 / 0.39), floored to the cascade 2e-03 / 3 % on `lt` | **currently unreachable** — the run aborts on the stale `specimpl_sha256` (§3.1). A ruling cannot cite this fixture until (a) or (b) is chosen |
| `wide640f` | 648 | its own priced allowance (1.79 / 0.29), floored to 2e-03 / 3 % on `lt` | PASS both ways. Deficits 2.0e-05..6.1e-05 cos and 0.60–1.17 % meanrel: 0.9–1.4 decades below the flat floors, 1.5–2.0 decades below the cascade floor actually in force on `lt`. But the binding step sits at **99.8 % of its own cap** (D-3 below) |
| `wide1024f` | 1,032 | its own priced allowance (1.62 / 0.35), floored to 2e-03 / 3 % on `lt` | PASS both ways. Deficits 1.7e-05..6.0e-05 cos, 0.7–2.3 % meanrel; same margins, and the binding step is again at **99.8 % of cap** |
| `wide4096f` | 4,104 | still the flat fallback (no allowance written) | Priced but unwritten: route-vs-route worst cos 0.999919 at step 6, median 1−cos 2.68e-05, 0 of 8 steps under G5; engine-vs-pin worst 0.999805, median 5.9e-05, 0 under G5. Would derive **5.68 / 0.79**. `corr(engine 1−cos, route 1−cos) = 0.834` — the engine's deviation and the reference's own legal spread move together at 4,104 lanes |

**What the flag on all four actually is, restated from the evidence.** The `n_bad` counts of 48/47/45/23
out of 2,048 come from `prototype/attn_decode_wide/gate.py` section 2, which holds the shipped
one-pass form and the streaming form to **per-element ULP agreement with the pins** on
`attn_pre_gate`. Per §07.3 (`specification/07-decode-step.md:79-82`) that route "must be gated on a
cos/meanrel band — §10 G6's thresholds — and NOT on agreement with the pins, which it cannot reach",
and `prototype/attn_decode/run.py` already implements exactly that with `pin_agreement=False`. The wide
driver re-imposes pin agreement, so it is asking a streaming route for bits it is documented as unable to
produce. Under G6's own numbers all four cases pass with 1.5–2 decades of margin.

**The must-catch controls still catch, and they are not adjudicated by these margins.** CPU, no device
(§20's derived case list, including the 512/513/514/515-lane boundary wave and both wide anchors):

```
.venv/bin/python prototype/attn_tile_model.py gate --negative   # rc=0, 567,296 elements, 0 failures
cases: span 0-63, span 64-127, wide640f decode7.L03(wide), wide1024f decode7.L03(wide), boundary 512..515 lanes
readpast 1040/1072 caught   dropcurrent 1072/1072   gqaoff 1072/1072   dimhalf 1064/1072
gqa_head0 134/1072          headcarry 804/1072      partalias 20/1072   must-not-hit 0 everywhere
```

`partalias` — the streaming-partial aliasing injection, the one whose must-catch set is empty without the
wide anchors — still fires, and no injection touches a must-not-catch cell. Since the new bands are
narrower than the flat floors (F2), the AGENTS.md widening test is satisfied by construction *and*
witnessed.

**What pricing does NOT buy, stated plainly for the ruling.** It does not touch the prefill-row verdict (§1);
it does not bind on the default `lt` route at these widths (F1); it does not make `--split 18` buildable
(F3); and on the `hand` route it makes `wide640f` red on 14 band lines, because an allowance carries no
record of which GEMM route produced its engine column while `hand` deficits run 1.1–1.6× the `lt` ones at
every step (side by side, cos deficit then meanrel, `lt` then `hand`: step 0 2.372e-05/0.6222 % then
3.008e-05/0.7335 %; step 4 6.100e-05/1.1664 % then 9.884e-05/1.5056 %, against caps of 6.4436e-05 and
1.6761 %). `--gemm hand` is not a green baseline at HEAD anyway (runs #8, #9), so this is a hazard to
name, not a regression to revert.

**D-3, made concrete — the new allowance has almost no discriminating power at its own binding step.** A
margin is the max engine/route ratio inside the fixture rounded *up* at 2 dp, so every step is admitted by
construction and the worst one lands just under its cap. Deficit as a share of the step's own allowance cap,
`lt` route:

| step | wide640f cos, % of cap | wide640f meanrel, % | wide1024f cos, % | wide1024f meanrel, % |
|---|---|---|---|---|
| 0 | 54.2 | 98.3 | 82.8 | 91.9 |
| 1 | 64.7 | 89.1 | 52.7 | 67.8 |
| 2 | 66.8 | 77.1 | 73.0 | 69.5 |
| 3 | **99.8** | 65.0 | **99.8** | 44.1 |
| 4 | 94.7 | 69.6 | 66.7 | 76.4 |
| 5 | 92.4 | 93.9 | 65.4 | 50.7 |
| 6 | 58.4 | 72.9 | 93.6 | 79.2 |
| 7 | 57.8 | 95.8 | 74.3 | 97.5 |

A check whose worst case sits at 99.8 % of its own limit cannot fail on the data that set the limit — the
same "check that cannot fail" class AGENTS.md logs for empty-parse gates, except that here it is deliberate
and named in the file's own `margin_basis` field. What the allowance does buy is a cap 7.8–18.2× below the
flat floor (31–73× below the cascade floor the `lt` route substitutes), which a worse engine would breach;
what it cannot buy is an independent verdict on the steps it
was priced from.

---

## 5. Escalations (already sent as a progress update; re-listed for the ruling)

1. The briefed mechanism is wrong in three places (§0) — the flag is a stage-tensor pin-agreement
   requirement in `prototype/attn_decode_wide/gate.py` that §07.3 forbids, not a flat-vs-allowance
   precedence in `engine/gate.py`.
2. `--split 18` cannot be built or judged by `engine/gate.py` today (cap semantics, §0 point 3). Any ruling
   that spends the 0.36 ms needs a route decision first: port the deeper cut into the engine launcher, or
   keep the geometry held.
3. **New and blocking for step 4**: five of 18 fixtures are red at HEAD on allowance provenance caused by a
   comment-only edit (§3.1). `wide520f`, one of the four flagged fixtures, is in that set, so the ruling
   cannot read that fixture until (a)/(b)/(c) is chosen.

## 6. Commands left for the parent (device, main-agent window, zero subagents)

```bash
bash tools/gpu_lock.sh acquire maxsc-p1
.venv/bin/python engine/gate.py --fixture wide4096f            # ~240 s, correctness only
# then, if that log path is kept as tmp/maxsc/gate_wide4096f.log:
.venv/bin/python engine/route_price.py --fixture wide4096f \
    --engine-log tmp/maxsc/gate_wide4096f.log --out tmp/maxsc/route_wide4096f_fresh.json
.venv/bin/python engine/route_allowance.py --table tmp/maxsc/route_wide4096f_fresh.json --dry-run
.venv/bin/python engine/route_allowance.py --table tmp/maxsc/route_wide4096f_fresh.json
.venv/bin/python engine/gate.py --fixture wide4096f            # re-gate under the new allowance
.venv/bin/python engine/gate.py --fixture dec300f              # stale now; green only after §3.1 (a) or (b)
bash tools/gpu_lock.sh release maxsc-p1
```

Reproduce everything else in this file:

```bash
.venv/bin/python prototype/maxsc_allowance/census.py --markdown   # §1 population
.venv/bin/python prototype/maxsc_allowance/check_inputs.py        # §3.1 staleness table
.venv/bin/python prototype/attn_tile_model.py gate --negative     # §4 must-catch controls (CPU)
```
