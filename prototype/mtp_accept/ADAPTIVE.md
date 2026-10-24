# ADAPTIVE — speculation depth that adapts to what the rounds actually accept

`--mtp-adaptive 0|1`, **default 0**, accepted only alongside `--mtp-accept`.
Implementation: `engine/mtp_adaptive.h` (the rule) and `engine/mtp_adaptive.h`'s `Tracker`, driven from the
accept loop in `engine/decode_run.cpp`. Gate: `engine/gate.py` section (m) with `--mtp-adaptive 1`, plus
section (m1) inside `engine/gate.py --selftest`. Refusal/default battery: `prototype/mtp_accept/
default_refusals.sh`.

**This document ships a mechanism and its gate. It does not ship a default.** The flag is off, and the only
thing that moves it is the paired A/B in §6, which the main agent takes alone (same split as
`prototype/mtp_accept/DEFAULT-ON.md` for the k=2 flip). No millisecond figure appears anywhere here; every
cost quoted is a dimensionless round-to-step ratio taken from the ledger, cited by section.

---

## 0. Why, in the ledger's own numbers

`engine/performance-status.md` §9, four paired rows at the shipping default `--mtp-accept 1 --mtp-k 2`:

| population | lanes | a (full prefix) | round/step | accept vs base |
|---|---|---|---|---|
| corpus, 12 cases, natural width | ~0.3k | 0.5269 | 1.971 | **+17.87 %** |
| templated long prose | 4,152 | 0.529 | 1.829 | **+23.44 %** |
| raw long prose | 4,142 | 0.400 | 1.841 | **+12.88 %** |
| corpus case that fits 4k (1 case) | 4,032 | 0.0323 | 1.889 | **−28.28 %** |

A round costs ~1.9 steps, so a round that accepts nothing spends two steps to emit one token. The default
is safe on three of those four rows and is a **pure tax** on the fourth, and nothing in the engine knows
which kind of content it has been handed: the draft block cannot predict its own acceptance, and the flags
cannot inspect the prompt. The rounds can, one round at a time. That is the whole feature: **stop paying
for speculation that is not paying.**

llama.cpp has no such policy — `common/speculative.cpp` drafts at a fixed depth for the life of the
request — so this is not a parity item. It is what makes our own default defensible on arbitrary content.

---

## 1. The two CPU questions, answered before the rule was written

### 1.1 Can `k` change between rounds of one run? Yes, and one thing had to be fixed to make it legal.

Everything a round touches was checked against the widest round it might be narrower than:

| buffer / property | sized by | a k=1 round on a k=2 side |
|---|---|---|
| `side.logits` (`rows × kVocab` F32) | `rows_of_k(kAlloc)` = 3 rows | writes rows 0–1 of 3 — a prefix, fine |
| `side.verdict_dev` (`kVerdictWordsFor(rows)` ints) | 7 words at k=2 | a k=1 verdict writes 4 of 7 — a prefix, fine, **but the readback must ask for 4** |
| `side.verdict_partials` | `kernels::kMaxArgmaxRows` = `kVerdictRowsK2` = 3 | independent of the round's k, fine |
| **`side.ck.slots`** | `kMidSlotsFor(rows)` = 2 | **not a prefix.** `submit_wave_body` requires `ckpt->slots == n - 1` EXACTLY (`engine/decode_step.cpp`, the V1 checkpoint guard), so a 2-row body armed with a 2-slot checkpoint throws before any kernel launches |
| `sc.wave_rows`, `st.cap` | the run's scratch / request | per-round check on `rows_of_k(round_k)`, not on `side.rows` |

So the one blocker was the checkpoint's slot count, and the fix is a **per-round view**: `accept_round`
takes `k` as a parameter, derives `rows = rows_of_k(k)`, and arms a local `DecodeScratch::GdnCheckpoint`
whose `mid_gdn`/`mid_conv` are the side's buffers and whose `slots = kMidSlotsFor(rows)`. Slot 0 means the
same thing at either width — "every layer's state after row 0" — so sharing the buffers is correct rather
than merely in-bounds; `engine/mtp_accept.h` states that, because it is the property a future width
depends on. The readback asks for `kVerdictWordsFor(rows)` per round, and `accept_round` refuses a `k`
above the side's allocation instead of walking off the end of the logits tile.

A second defect was **introduced by this change and found while writing it**: `tally()` used to bill
`side.captured_bytes()`, `side.words` and `side.subs_per_round`, all per-side. With one side serving two
depths that bills a 2-row round at the 3-row rate — `captured_bytes`, `status_d2h_bytes` and `split_subs`
on `MTPSTAT` would each have over-reported, silently, on every demoted round, and those are exactly the
figures `ROUND-TAX.md` priced the round tax from. They are now derived from `Round::rows`. Flag-off and at
a fixed `--mtp-k` the round's width *is* the side's, so every one of those numbers is the number the binary
printed before this change.

What is deliberately **not** claimed: that a run whose depth varies is byte-identical to a run at either
fixed depth. It is not, and it cannot be — A-18 measures two legal bf16 continuations agreeing 0.8776 of
the way, and a different body width is a different form (`ACCEPT-PLAN.md` already carries the 2-row-vs-1-row
form gap). The admissible claim is **losslessness against the reference stream**, which is what section (m)
and `m3_loop_rate.sh` adjudicate, and that is what §5 gates.

### 1.2 Where the host already knows acceptance: the verdict readback, and nothing else.

`engine/mtp_accept.cpp` ends every round with one blocking `hipMemcpy` of `kVerdictWordsFor(rows)` ints —
4 at k=1, 7 at k=2 — and `Round::n_accepted`, `Round::emitted` and the hit bits are folded out of those
words on the host. The policy is fed from `ar.emitted` and the round's depth, i.e. from values that are
**already in registers at the moment the decision is made**. Consequences, all deliberate:

* **no new device→host transfer, no new wait, no per-round stall.** The policy is host arithmetic (two
  fixed arrays and a division) inside a window the round already blocked on. Had the decision required
  asking the device anything the verdict did not already answer, the right answer was to stop and report;
  it did not.
* the decision is therefore taken *after* round N and applies to round N+1, which is also when the
  proposals for N+1 are drafted — so the driver drafts `need_next` proposals, never `need`, and after
  abandonment it drafts none (the draft block exists to feed rounds; paying it for candidates no body will
  read would make the "fallback" more expensive than the plain step it is falling back to).
* `DECWALL`'s census stays `ok` by construction: an abandoned iteration is the loop's existing plain-step
  branch, so `bodies == plain + rounds` and the emit-site count still reconciles with the verdict tally.

---

## 2. The rule

One sentence: **each round's depth is drawn from the tokens-per-step-equivalent the recent rounds already
reported; depth may only go down (2 → 1 → plain steps); and each of the two changes needs its own sample.**

```
estimate  est = Σ emitted / Σ cost          over the last `window` rounds, cost = cost(k of that round)
shape     k ∈ {2, 1, plain};  plain = 0 means "no speculation for the rest of this request"
ceiling   the requested --mtp-k. The policy may run narrower, never wider.
feed      once per round, after the verdict fold
demote    k == 2 and rounds-seen ≥ 16 and est < 0.950      → k = 1
abandon   k == 1 and rounds-seen ≥ 24 and est < 0.900      → plain (final)
on change clear the window (so the next decision judges only rounds run under the new geometry)
no re-promotion, ever
```

Four properties, each doing a job the brief asked for:

* **Bounded estimate over a sliding window.** A ring of `window = 32` rounds, and the estimate is a *ratio
  of tallies* (Σ tokens ÷ Σ step-equivalents), not a mean of per-round ratios: it is literally "what this
  window paid per step", which is the quantity the win bar is stated in, and it is unbiased when the window
  mixes depths (which it will, right after a change). An EWMA would have been equally allowed by the brief
  and was not chosen because "last 32 rounds" is checkable by hand and "an exponentially weighted blend" is
  not.
* **A minimum observation window before the first demotion**, so the first rounds are not judged on noise:
  16 at the demote line, 24 before the sticky one. §3 sizes both.
* **Hysteresis.** Two independent mechanisms, and neither depends on a second threshold: (a) the window
  clears on every change, so two changes are never closer together than the next decision's own minimum
  sample; (b) **there is no re-promotion**, so depth is monotone and a request can experience at most two
  policy changes. Round-to-round oscillation is not discouraged, it is not expressible. (Re-promotion was
  optional in the brief and was declined: a promote line above a demote line is a second threshold to tune,
  and the only content it would rescue is content whose acceptance rose mid-request — for which the cost of
  staying at the narrower depth is small and bounded.)
* **Abandonment is sticky.** `plain` is final for the request: after it the loop takes the branch it
  already had for "no proposal pending" — the pre-MTP shipping step, no new code path, no new numerics —
  and stops drafting.

Depths are `{1, 2}` only. k=3 is not implemented (`engine/mtp_accept.h::kMaxK` is 2) and is not invented
here.

---

## 3. The arithmetic that chose every constant

### 3.1 Round costs — the bands, not a number

`cost(k)` is the round's price in plain steps, taken as the **midpoint of a measured band**:

* `cost_k2 = 1.90` ← §9's own round/step column: 1.829, 1.841, 1.889, 1.971 → band [1.829, 1.971].
* `cost_k1 = 1.58` ← `ROUND-TAX.md` §"The pre-registered adopt line": 1.472, 1.481 at templated 4,152
  (1.530 with `--mtp-dense --mtp-prompt-lanes`); `CORPUS-WALL.md` 1.686, 1.701 at the corpus's natural
  width → band [1.47, 1.70].

A midpoint with no band beside it is a taste, so the band is the citation and the *error* it induces is
computed below. A cost read **high** demotes early (safe: the fallback is the shipping step); read **low**
delays a demotion (unsafe), which is why the sensitivity is checked at both ends rather than assumed away.

### 3.2 Break-even, and why the thresholds sit under 1.0

A plain step emits exactly 1 token per step, so a round is worth running iff `tokens/step-equivalent > 1`:

```
k=2 worth running  ⟺  (1 + p1 + p2) / cost_k2 > 1   ⟺  p1 + p2 > 0.90  of 2 candidates
k=1 worth running  ⟺  (1 + p1)      / cost_k1 > 1   ⟺  p1      > 0.58  of 1 candidate
```

(`p1` = P(first candidate accepted), `p2` = P(both). `a` in the ledger is `p2`. This is the k≥2 correction
`§9` asks for: the `a* = round/step − 1` line printed by some harnesses is the **k=1** identity.)

Both policy thresholds sit *below* that 1.0 so that noise at break-even cannot flip a decision:

* `demote_est = 0.950` — the round must be measurably worse than a plain step (>5 %) to lose a candidate.
* `abandon_est = 0.900` — the irreversible decision demands >10 %. A wider gap for the sticky step is the
  cheap half of the anti-oscillation story; the expensive half is §3.4.

### 3.3 The four §9 rows, classified by the rule as it is written

Recovering each row's `p1` from its own three published numbers (`mean_emit = round/step × (1 + win)`,
`E[n] = mean_emit − 1`, `p1 = E[n] − a`):

| row | round/step | win | mean_emit | p2 (=a) | p1 | est at k=2 (÷1.90) | verdict | est at k=1 (÷1.58) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| corpus 12 | 1.971 | +17.87 % | 2.323 | 0.527 | 0.796 | **1.223** | keep | 1.137 | keep |
| templated 4,152 | 1.829 | +23.44 % | 2.258 | 0.529 | 0.729 | **1.188** | keep | 1.094 | keep |
| raw long 4,142 | 1.841 | +12.88 % | 2.078 | 0.400 | 0.678 | **1.094** | keep | 1.062 | keep |
| corpus @ 4,032 | 1.889 | −28.28 % | 1.355 | 0.032 | 0.323 | **0.713** | demote | 0.837 | abandon |

All four classify correctly, and the **model reproduces the measured win to under 2 %** when each row's own
cost is used instead of the midpoint — the estimator is not fitting the ledger, it is a reformulation of
it. Margins: the weakest keep is +15 % clear of its line, the demote is 25 % under it.

Sensitivity, at the *wrong* end of each cost band:

* k=2 with the pessimistic cost 1.971: raw long reads 1.054 — still keeps (the line is 0.95).
* k=2 with the optimistic cost 1.829: the bad row reads 0.741 — still demotes.
* k=1 with the optimistic cost 1.47: the bad row reads 0.900 — **marginal**, and the honest statement is
  that abandonment for a≈0.03 content is the close call in this table, not the decisive one. The measured
  device case (§5.2, `wide4096f`) abandoned at est 0.712, which is not close; but the rule's designers
  should know that if the k=1 round is cheaper than the band's midpoint, staying at k=1 costs about nothing
  (k=1 at break-even is break-even).
* k=1 with the pessimistic cost 1.70: raw long reads 0.987 — still keeps.

### 3.4 Why 16 and 24 rounds

Per-round emitted tokens are bounded, so the window mean's standard error is σ/√N ÷ cost. Using the §9
rows' own implied variances:

* at k=2, templated-like spread (p1 0.729, p2 0.529): per-round σ ≈ 0.86 tokens → se(est) at N=16 ≈ **0.11**.
  The weakest winning row (est 1.094) is 1.3 σ above its line; a false demotion there costs k=1's 1.06
  against k=2's 1.09, i.e. ~3 %, not the 28 % the rule exists to avoid.
* at k=1, low-acceptance spread (p1 ≈ 0.32): per-round σ ≈ 0.47 → se(est) at N=24 ≈ **0.06**. The weakest
  winning k=1 row (est 1.062) is 2.7 σ clear of 0.90. That is the entire reason for a *second, larger*
  minimum: the decision that cannot be undone gets the bigger sample. Raising the demote minimum to 24
  instead would have cut the false demote rate and paid for it by holding bad content at the expensive
  geometry for 8 more rounds, which is the term the −28 % row actually cares about.

Corollary, stated because a gate trips over it: 16 + 24 rounds is a **large number of bodies**, so the
policy cannot be exercised by an 8-token pinned stream. The device cases in section (m) ask for
`MTP_ADAPT_STEPS = 56` bodies and judge the *stream* only over the pinnable depth; the extra bodies exist
to give the policy an observation window and adjudicate no token claim (`gate.py` says this in code, since
"which cases does a claim cover" is the thing that has bitten this repo repeatedly).

---

## 4. Attribution, because every measurement of this will be judged by it

Printed only when the flag is on, so a flag-off run's log is unchanged byte for byte:

```
MTPADAPT on=1 k_start=2 floor=1 ceiling=2 plain=0 window=32 min_observe=16 min_observe_abandon=24
         demote_est=0.950 abandon_est=0.900 cost_k1=1.58 cost_k2=1.90 re_promote=0
MTPADAPT round=4112 k=1 est=0.5263 prev_k=2 window_rounds=16 plain=0
MTPADAPT round=4136 k=0 est=0.7120 prev_k=1 window_rounds=24 plain=1
MTPADAPT-PLAIN reason=abandoned round=4136 draft_work_stopped=1
MTPSTAT  … k=2 depth1=3 depth1_rate=0.075 candidates_accepted=3 k_ran=1..2
MTPADAPTSTAT rounds_k2=16 rounds_k1=24 plain=2 changes=2 final_k=0 ceiling=2 window_rounds=0
             window_emitted=0.000 window_cost_steps=0.000 est=0.000 est_all=0.629
MTPADAPTATTR driver_k2=16 driver_k1=24 driver_changes=2 tracker_k2=16 tracker_k1=24 tracker_changes=2 agree=1
```

* The **config line** states every threshold before the first body runs, so the log carries the rule that
  produced the run's bits and the gate can parse its arithmetic off the engine instead of typing a copy.
* One **event line per change**, stamped with the step, the new depth, the estimate, the previous depth and
  the sample size behind it. No heartbeats.
* `MTPADAPTSTAT` is the per-run attribution the brief asked for: **rounds at k=2 / at k=1 / plain steps**
  (`plain=` is `DECWALL`'s own counter, not a second count of it), plus `changes`, `final_k`, the window's
  raw tallies, and `est_all` — the run's whole-life rate, which is the number to read against §9.
* `MTPADAPTATTR` compares the driver's own counters with the tracker's. They are kept in two places on
  purpose and `agree=1` is the check that neither drifted. (Same reason `DECWALL` cross-checks the emit
  sites against the device verdicts.)
* `MTPSTAT` appends ` k_ran=lo..hi` **only** when the rounds did not all share one depth, so the byte-
  compared k=1 line stays the old line while a mixed run can still say which mixture produced it. Without
  it, `mean_tokens_per_round=` would be a statistic over an unnamed population.
* `DECWALL` is untouched and still prints `census=ok`: 42 bodies = 40 rounds + 2 plain in the excerpt, and
  the emit-site count still reconciles with the verdict tally plus plain minus suppressed.

## 5. The gate

### 5.1 Section (m) with `--mtp-adaptive 1`

Every case in the derived population (`mtp_round_population()`, unchanged — read out of `data/refs/`, never
typed) additionally runs through the policy with `--steps MTP_ADAPT_STEPS`, and gains these assertions:

* the config echo exists, its `k_start` equals the depth the case asked for, and `re_promote=0`;
* **the echoed thresholds equal `engine/mtp_adaptive.h`'s**, which `gate.py` reads out of the C++ source —
  so the mirror's numbers come from the implementation's text and the comparison doubles as a stale-binary
  canary (a binary built before a threshold edit disagrees with the header it is being judged against);
* the engine's event lines equal an **independent Python recomputation** (`mtp_adapt_mirror`) over the
  run's own `ACCEPT` lines, matched on `(round, new depth, est to 4 dp)` — an event at the wrong round
  cannot pass by naming the right depth;
* every round line carries the depth it ran at (the engine stamps ` k=1` on a narrow round only when the
  policy is armed, so flag-off bytes are untouched), and **the depth never rises**;
* no round ran after abandonment; `MTPADAPTSTAT`'s per-depth rounds sum to `MTPSTAT rounds`; `changes`
  equals the number of event lines; `MTPADAPTATTR agree=1`;
* **losslessness against the PIN is asserted for every case**, unchanged predicate, on both sides of a
  change. The policy changes geometry and never a verdict; that is what this row can falsify.

### 5.2 The population coverage, derived

Section (m) classifies each case from its own measured estimate — `demotes` if the policy changed its
depth, `keeps` if it had at least one decision point and never changed — and then asserts:

* at least one case **demoted** (and therefore printed the event line, with `k` never returning to 2, which
  the monotonicity row proves);
* at least one case **kept** the ceiling;
* **losslessness was actually adjudicated in both classes** (a class whose streams were never read is
  named and failed rather than counted);
* a case that never reached a decision point says `NOT ADJUDICABLE` rather than counting as "did not
  demote" — which is the vacuity the brief asked to be defended against.

No fixture name appears in the classification. Which content the model can and cannot predict is a
measurement; a hand-listed "near-zero case" is the admission-list trap (`AGENTS.md`) wearing a new hat.
Measured on device while this was written — seven real fixtures through the adaptive arm, one lock hold, no
timing taken, **zero red verdict rows**, and the population supplied both classes on its own:

| fixture | est at the change | final depth | plain steps left | `est_all` |
|---|---|---|---|---|
| `wide4096f` | demote 0.5263, abandon 0.7120 | plain | 14 | 0.629 |
| `chunk64b` | demote 0.7566, abandon 0.8175 | plain | 3 | 0.790 |
| `dec300f` | demote 0.8553 | k=1 | 1 | 0.978 |
| `mt32f` | — (no change) | k=2 | 3 | 1.137 |
| `p8` | — (no change) | k=2 | 2 | 1.206 |
| `dec64f` | — (no change) | k=2 | 2 | 1.259 |
| `dec64t` | — (no change) | k=2 | 3 | 1.496 |

`wide4096f` is §9's −28 % row in fixture form (accept rate 0.075), and it is the row the feature exists for:
fourteen of its bodies became plain steps. Losslessness against the PIN was adjudicated in both classes —
the demoting fixtures' streams matched the pinned greedy continuation through a demotion and, for
`wide4096f`, through an abandonment as well.

### 5.3 `--selftest` section (m1), and the vacuity witness

`engine/gate.py --selftest --no-engine` runs the policy on synthetic histories, in process, with **no
device and no timing**:

1. the mirror reproduces a **hand-derived** near-zero history's events (16 rejects → `1/1.90 = 0.5263` →
   demote; 24 more rejects at k=1 → `1/1.58 = 0.6329` → abandon). Hand-derived on purpose: a record whose
   events were generated by the code under test would make the comparison a tautology.
2. a high case (every round accepts both candidates, `3/1.90 = 1.5789`) is green with **no** event line.
3. **the vacuity witness**: the one predicate that decides anything (`mtp_adapt_decide`) is replaced with
   an always-keep stub, which must redden the near-zero case; then with an always-demote stub, which must
   redden the high case; then restored, and both must go green again. If either stub changes nothing, that
   assertion is decoration and the selftest fails rather than reporting PASS over it.
4. a round line with no `k=` stamp is *refused as evidence* rather than inferred, and a policy echoing a
   ceiling other than the depth asked for reddens.

### 5.4 What this feature is **not** gated on

* no byte-identity between a mixed-depth run and a fixed-depth run (§1.1);
* no claim that the chosen depth is optimal — this is a floor-protection rule, not a scheduler. `cost(k)`
  is width-only, so a run whose round cost varies with context length is billed the band midpoint;
* no k=3, no per-request priority, no mid-round re-planning, no cross-request learning.

---

## 6. The blocking runs (main agent: exclusive window, zero subagents, blocking, never backgrounded)

Correctness first, then the three measurements that decide the default. All are correctness-only except the
last three, which are the only ones that produce a rate.

**Lock protocol** (`AGENTS.md`: three drivers self-lock and a caller that holds the lock deadlocks them):
`engine/gate.py` and `prototype/mtp_accept/default_refusals.sh` do **not** self-lock, so one hold covers
items 1, 2, 3, 5 and 6; release it before item 4's `m3_loop_rate.sh` and before §6.1's `mtp_wall_ab.sh`,
both of which acquire the lock themselves.

```bash
# 1. the shipped path, unchanged
engine/gate.py --fixture dec8
# 2. the policy's own teeth, in process
engine/gate.py --selftest --no-engine
# 3. section (m) BOTH ways — the second is the new arm
engine/gate.py --fixture dec8 --mtp-adaptive 0
engine/gate.py --fixture dec8 --mtp-adaptive 1
# 4. full-corpus losslessness, both at the default and with the policy on (49/49 both)
prototype/mtp_accept/m3_loop_rate.sh
EXTRA="--mtp-adaptive 1" OUT=/tmp/m3_adaptive prototype/mtp_accept/m3_loop_rate.sh
# 5. the new flag's refusals, and the default proved by absence
prototype/mtp_accept/default_refusals.sh
# 6. speccheck, quick band
speccheck/run_all.py --quick
```

And the three that decide it, each a paired A/B inside ONE lock hold, medians, no cross-window comparisons
(`AGENTS.md`: a ruler re-taken in a different window is not an A/B):

1. **corpus 12-case population** — adaptive-on vs adaptive-off vs plain (`--mtp-accept 0`). Expected shape:
   adaptive ≈ adaptive-off, because the population's acceptance is ~0.53. If adaptive is *below* off here,
   the demote line is too high and §3.2's margin is wrong, not the mechanism.
2. **the a ≈ 0.03 case at 4k** — adaptive-on vs adaptive-off, the same fixture in one window. This is the
   row the feature exists for: adaptive-on must land near the plain arm and far above −28 %. If it does not,
   read `MTPADAPT`'s event rounds — a 16+24-round observation tax on a short generation is the only
   legitimate reason, and `plain=` on `MTPADAPTSTAT` says how many steps were saved.
3. **templated 4,152 adaptive-on vs adaptive-off** — must not lose the +23 %. est here is ~1.19 against a
   0.95 line, so the expected result is *no policy change at all* and a null within the run-to-run spread;
   if it changed depth, the estimator is reading this content worse than §9's numbers say it should, which
   is a bug in the estimator rather than a result.

Quote each with `MTPADAPTSTAT`'s `rounds_k2/rounds_k1/plain` beside the rate. A percentage for an adaptive
run without its depth attribution is not a result — it is a mixture wearing one number.

### 6.1 The exact commands, on the harnesses that already exist

Fixtures come from `prototype/mtp_accept/ids_from_chat.py` (templated) and `ids_from_text.py` (raw), as in
`prototype/mtp_accept/DEFAULT-ON.md` §5. `--ids-file` splits on commas (`AGENTS.md`: a space-joined fixture
reads as ONE token), and every harness below already asserts the field count it wrote.

**Arms 1 and 2 in ONE window.** `corpus_wall_ab.py` does not self-lock (its `.sh` wrapper owns the lock, so
running the wrapper while holding it would deadlock), which is what lets a caller take several arms inside one
window — the paired-A/B rule. `--ctx 0` is the corpus at natural width (the +17.87 % row); `--ctx 4032`
selects the one case that fits 4k (the −28.28 % row). Each invocation already carries the plain arm, so
`off` vs `adaptive` is the pair and `base` is the third point:

```bash
tools/gpu_lock.sh acquire "main: adaptive default A/B"
for ctx in 0 4032; do for arm in "" "--mtp-adaptive 1"; do
  .venv/bin/python3 prototype/mtp_accept/corpus_wall_ab.py --ctx $ctx --out "/tmp/adapt_${ctx}${arm:2}3" \
      $([ -n "$arm" ] && printf '%s' "--accept-extra=$arm")
done; done
tools/gpu_lock.sh release "main: adaptive default A/B"
```

**Arm 3, the templated 4,152 long run.** `mtp_wall_ab.sh` self-locks — do NOT hold the lock around it — and
runs base and accept in one window, so `ACCEPT_EXTRA` is the arm difference exactly as §8 took it:

```bash
ACCEPT_EXTRA="--mtp-adaptive 1" IDS=<templated-4152.csv> prototype/mtp_accept/mtp_wall_ab.sh
```

The decisive reading of arm 3 is window-independent and should be checked first: **it must print no `MTPADAPT`
event line at all.** §3.3 puts its estimate at ~1.19 against a 0.95 line, so the expected result is that the
policy never fires and the rate is the §8 figure again; if it did change depth, the estimator is reading good
content worse than §9 says it should, which is a bug in §3.1's cost bands and must be reported as one rather
than booked as a result. The rate beside §8's carried +23.44 % is a cross-window comparison and must be
quoted with that caveat (`AGENTS.md`: a ruler re-taken in a different window is not an A/B).

**Losslessness over the whole corpus, both modes** (49 lines each, no timing content):

```bash
prototype/mtp_accept/m3_loop_rate.sh
EXTRA="--mtp-adaptive 1" OUT=/tmp/m3_adaptive prototype/mtp_accept/m3_loop_rate.sh
```

Quote every adaptive percentage with `MTPADAPTSTAT`'s `rounds_k2/rounds_k1/plain` beside it. A rate for an
adaptive run without its depth attribution is not a result — it is a mixture wearing one number.
