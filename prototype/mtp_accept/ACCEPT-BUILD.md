# ACCEPT build — the MTP stage family (M1) and the accept loop (M2)

Worker increment, 2026-10-22. Nothing here is committed. Status legend: `[ ]` not started, `[~]` in
progress, `[x]` done and witnessed. This file is an append-only checkpoint: the sections below are
written in the order the work happened, including the dead ends, so a compaction cannot lose the plan.

## 0. The contract as I read it, and the two files that own the bands

- `prototype/mtp_accept/ACCEPT-PLAN.md` is the binding brief. The bands, kill lines and timing protocol
  are pre-registered there and are not re-drawn anywhere below.
- `engine/mtp_verify.h`'s header carries a finding it is "obliged to state, not design around": a
  snapshot taken BEFORE the verify body restores the mixer state to `S_q`, while what a REJECT needs is
  `S_{q+1}`, and the layer-major body has no point in the stream where `S_{q+1}` exists to copy. That
  finding is the crux of M2's rollback bullet — see §4 — and it is not mine to overrule.
- No timing taken or printed anywhere by this worker, per `AGENTS.md`. Every device run below is
  correctness-only, ≤ ~60 s, one at a time, with its command, wall, `util` at acquire, and the words
  "no timing taken".

## 1. M1 — the `mtp` stage family. What is being built, in one paragraph

`decode_run` refuses `--mtp-draft 1` with `--time-stages` because the draft's submissions belong to no
family of §07's launch bill, and charging them to whatever segment preceded them would bill the wrong
stage. The remedy is the one that closed the draft-census red (Topic A, `gate.py::mtp_draft_plan`):
**declare the launches in the plan the instrument reads**. So:

1. `stageclock::Stage` gains `kMtp`, named `mtp` — one word for one concept, beside §07's nine bill rows
   in the enum so the step report's existing row order is untouched.
2. Every submission site in `engine/mtp_draft.cpp` gets a `stageclock::note(kMtp, n)` beside its
   `launchstats::record`, exactly as `decode_step.cpp` does for its own sites. Not one marker around the
   round: a coarse marker would let a launch inside the round go uncounted and still match a hand-typed
   total, which is the empty-parse failure class.
3. The clock learns the draft's per-step bill from the declared §09 counts (`set_plan_mtp`), the same
   precedent as `set_plan_proj` / `set_plan_cast_fold` / `set_plan_copy_fold`, and `plan_per_step` grows
   by exactly that.
4. The staged passes replay the draft as part of the step they replay, so the family is populated by what
   the production loop does, not by a synthetic loop that could drift from it.
5. The refusal is lifted only when the family is actually populated, and stays in force for the schedule
   the clock cannot bill — see §2, because that part is a real constraint and not a preference.

## 2. A constraint found before the code, not after: the clock's per-step division is exact

`Clock::describe()` prints `launches_per_step = subs / steps` and `plan_match = (subs/steps ==
plan_per_step)`, and its own comment says the uniformity "is structural — every generation step submits
the same plan — so an exact integer division is honest here". That premise holds for §07's bill and does
NOT hold for the draft under `--mtp-sched loop`: a `loop` round follows a step only when the previous
round did not advance two positions, so `loop` makes ~1 round per `1+a` steps. There is no integer
per-step bill for that schedule, and a clock that billed `rounds × bill / steps` with integer division
would print `plan_match=0` on a correct run or — worse — truncate to a plausible one.

Decision taken inside this increment, stated rather than hidden: the stage clock bills the `mtp` family
**one round per step**, and `--time-stages` therefore accepts `--mtp-sched every` and refuses
`--mtp-sched loop` with that reason in the message. This is not a narrowing of the timing protocol:
ACCEPT-PLAN's arms are "flag-off baseline, flag-on k=1, flag-on k=2", and the accept loop M2 builds makes
exactly one draft round per round whatever the schedule, so the per-step shape is the shipping shape. The
`loop` schedule stays available for every correctness pass, and it is the schedule the accept-rate witness
is measured at.

## 3. M1 implementation state — DONE, witnessed on device

`[x]` `engine/stage_clock.h` — `kMtp` in `Stage` (after §07's nine rows, so the step report's existing row
    order is untouched); `Clock::set_plan_mtp(long long subs_per_step)`, documented as one-round-per-step.
`[x]` `engine/stage_clock.cpp` — `stage_name(kMtp) = "mtp"`; `plan_bill()` gained a fourth parameter and
    `p[kMtp] = mtp_per_step`, so `plan_per_step` grows by exactly the declared bill; `coarse_stage(kMtp)`
    is `default` → it folds to itself, deliberately (folding it into `kGdn`/`kProj` because a round launches
    glues and projections is the misattribution the row exists to prevent).
`[x]` `engine/mtp_draft.h` — `kStageSubsPerRound = kBlockCopies + kBlockLaunches + kHeadLaunches = 15`,
    alongside `kSubmissionsPerRound = 16`. TWO constants, and the difference is one submission: the
    blocking vocabulary D2H is billed 1 by the `MTPDRAFT` census and 0 by the stage clock, because the
    clock's `kHostTail` row is noted with zero submissions "so the totals still mean launches".
`[x]` `engine/mtp_draft.cpp` — a `RoundTally` marks every submission site (`kMtp`) and the round is then
    **required to add up**: `tally.check()` throws if the marked submissions do not equal
    `kStageSubsPerRound`. That is the anti-drift witness, and it runs on every round whether or not a clock
    is armed. 15 marks (`mtp_attn_row` and `lm_head_full_xbf16` are counted here even though they have no
    `launch_stats` family), and the D2H is marked with 0.
`[x]` `engine/decode_run.cpp` — `stage_pass()` takes the `MtpSide` and replays the draft **inside** the
    measured span (`begin_step` → `decode_step` → `mtp_after_step` → `end_step`), resets the draft's cache
    and round tracker per pass, states the declared bill with `set_plan_mtp` only when the flag is on,
    suppresses the replay's per-round prints (`quiet`, so `gate.py`'s census-vs-`DRAFT`-lines cross-check
    still has exactly one census and one line per round), and prints `STAGEMTP gran=… steps=… rounds=…
    uniform=…` as the in-pass witness. `mtp_after_step` now returns whether it submitted a round.
`[x]` refusal lifted only for what the family can bill: `--time-stages` + `--mtp-draft 1` now requires
    `--mtp-sched every` (reason in the message), and `--time-waves` stays refused because a wave prices a
    prefill pass and no draft submission falls inside that unit — the family would be declared-and-empty.
`[x]` `engine/gate.py` — `mtp_stage_plan()` (15) **derived** from the two tables Topic A already derived
    from: `sum(mtp_draft_plan()) + block_attn_site + head_lm + block_d2d`, i.e. a different PROJECTION of
    one Python count of §09, not a second tally. New verdict in `mtp_plan_verdicts`: declared
    `per_round` minus the run's own declared blocking reads (`head[0]`) must equal it. `--selftest` grew
    two derived cases and the selftest is PASS.
`[x]` inertness and census, as run — §5.
`[x]` wrong-count witness — §6.

## 4. M2 — the rollback question, and why it stopped the increment rather than being designed around

(seeking the decision; written before any M2 code so the reasoning is on disk)

## 4. M2 — the rollback question: DECIDED (supervisor, this session), option V1

M1 is **accepted and recorded as shipped-in-tree** (the witness set in §5–§6 is what was accepted).

The four alternatives the M2 bullets admitted, with the price of each — kept here because the supervisor
said the table is what made it decidable:

| option | what it costs | speedup at `a = 0.6641` |
|---|---|---|
| **V1 — per-layer checkpoint INSIDE the body** (between each GDN layer's row-0 work and its row-1 work, copy that layer's state + conv window; restore on reject) | `rho ≈ 0.16 ms` + `0.16 ms` per reject, +36 launches/round, and a scoped change to `submit_wave_body` | **≈1.42×** |
| V2 — snapshot `S_p`, restore on reject, then run the one-row body for row 0 anyway | +1 whole step per reject | ≈1.10× — this is ACCEPT-PLAN's "recompute" arm, already dead |
| V3 — restore `S_p`, re-anchor the next round at `p` with the verifier's token as the candidate | nothing | **1.00×** — the accept round re-emits an already-emitted token |
| V4 — row 0 = shipping one-row step, row 1 = a second one-row body | 2 steps | **0.83×** — `mtp_delta/D3-RESULT.md`'s KILLED row |

**Decision: V1**, with four conditions, quoted because they are the contract now:

1. **Scope the split, do not generalise it.** The row-0/copy/row-1 insert goes in only when the body is the
   verify body's width AND the accept flag is on, with the `cast_fold`/`copy_fold` precedent — the rule is a
   property of the geometry, so derive it. If the scope cannot be expressed without changing the shipping
   prefill's launch sequence: stop and report, do not make the prefill carry a decode feature's cost.
2. **The admission is a byte-compare, not a band.** Split `row0+row1` must be **bit-identical** to the
   batched single-launch GDN result at the verify body's width (all layers, both parities, real content),
   and flag-off byte-identical to the current binary. A near-match is a fail.
3. **`rho` is 0.16 ms, not the plan's 0.12.** No band or kill line moves — the kill line stays `a < 0.4568`
   and the adopt bar stays +10 % tok/s at 4,000 lanes. But the projected speedup is recomputed from the
   measured `rho` and stated beside 1.42×, and the +36 launches/round is its own line for attribution.
4. **Fold in the D-b pieces now** (device-side argmax, one status word) and build the accept-loop shell
   behind `--mtp-accept` with every rollback call site wired, so the only thing V1 changes later is the copy.

Memory note the supervisor asked to have on the record: the mid-body checkpoint is a **third** ~19.27 MiB
region per request (18.0 MiB GDN + 1.266 MiB F32 conv window), the same size as the existing snapshot's
checkpoint and shadow sides. Per-request model state is 18.66 MiB and the bf16 checkpoint is ~4 % of the
device-visible pool (`AGENTS.md` scope table), so capacity is not a constraint and no paging/eviction
machinery is implicated. The arithmetic, stated so it is not re-derived later: snapshot(checkpoint) +
snapshot(shadow) + checkpoint(mid) = 3 × 19.27 MiB ≈ **57.8 MiB per in-flight request**, plus the draft
block's own 39 MiB of weights and ~15 MiB of lane caches.

## 4b. Why `S_{p+1}` is the state a reject needs, spelled once

The round feeds `[cur at p, candidate d at p+1]`. On a reject the emitted token is the verifier's own
prediction for position `p+1`, so the request must sit where it would have sat after consuming `cur` at
position `p` and nothing else — that is `S_{p+1}`. The body is **layer-major** (all rows of layer `l`, then
layer `l+1`), so "every layer's state after row 0" is a thing that exists at no single point in the batched
stream, which is the finding `engine/mtp_verify.h` states in its own header and hands back. V1 makes it
exist per layer, where it does have a home: the gap between that layer's row-0 recurrence and its row-1
recurrence, 18 copies totalling the same 19.27 MiB a whole-state copy would cost if the state were one
buffer. The conv window needs the same treatment and needs it more sharply: `gdn_conv` at `n = 2` writes
only the END window, so the row-0 window has to be produced by calling the conv per row, which is also what
makes the second row's input window (`conv_new` after row 0) addressable at all.

## 5. M1 device evidence — every run correctness-only, one at a time, `tools/gpu_lock.sh` held by this
## worker, blocking, **no timing taken and no ms figure read** (the `STAGE` lines' ms columns were grepped
## out by the script that ran them, `prototype/mtp_accept/m1_stage_witness.sh`)

Fixture: `dec8`'s own 17-id prompt (`data/refs/dec8/manifest.json` `sources[0].ids`, which contains the
end-of-turn id 248046 mid-prompt and the think-boundary ids 248068/248069), `--steps 8 --state-fp`.

| # | command | util at acquire | wall | result |
|---|---|---|---|---|
| 1 | `PART=AB prototype/mtp_accept/m1_stage_witness.sh` (inside one lock hold, tag `m1 stage family witness AB`) | 92 % | A 1 s, B 1 s | **flag-off inertness**: `STAGE gran=stage TOTAL … launches_per_step=224 plan_per_step=224 plan_match=1` (and the same at `gran=kernel`) from BOTH the new binary and `engine/build/decode_run.premtp`; `sampled.txt` SAME, **9 logits files, 0 differing**, **86 STATEFP rows SAME**, every `STEP` line SAME |
| 2 | same hold, `PART=C` | (same hold) | 2 s | **flag-on census**: `STAGE gran=stage family=mtp launches_per_step=15`, `TOTAL … launches_per_step=239 plan_per_step=239 plan_match=1`, `STAGEMTP gran=stage steps=8 rounds=8 bill_per_round=15 uniform=1`; and the side-computation property holds — flag-on vs flag-off `sampled.txt` SAME, 9 logits files 0 differing, 86 STATEFP rows SAME, STEP lines SAME |
| 3 | same hold, `PART=D` | (same hold) | <1 s | **canaries**: `.premtp` rejects `--mtp-draft` (`unknown argument`), so rows 1 and 2 are two builds and not one build compared with itself; `--time-stages --mtp-sched loop` and `--time-waves` are both refused with the reason in the message |
| 4 | `QWEN_MTP_DRAFT=1 decode_run … --time-stages --mtp-sched every` on the **WRONGCOUNT build** (tag `m1 wrongcount witness`) | 1 % | 1 s | `TOTAL … launches_per_step=239 plan_per_step=240 plan_match=0` at both granularities while `STAGEMTP … uniform=1` — see §6 |
| 5 | `decode_run … --steps 2 --mtp-draft 1` on the **DROPPEDMARK build** (one `tally.mark` commented out) (tag `m1 tally witness`) | 1 % | 2 s | the round refuses: `mtp_draft: draft_round: marked 14 billed submissions across 15 site(s), but the declared §09 bill is 15 …` and rc=1 |
| 6 | `engine/gate.py --fixture dec8` **flag off** (tag `m1 gate dec8 off+on`) | 0 % | 9.6 s | **ENGINE GATE PASS**, and `grep -c "draft " = 0`: with no `MTPDRAFT` line the new verdicts do not run at all, so the flag-off gate is the same gate it was |
| 7 | `QWEN_MTP_DRAFT=1 engine/gate.py --fixture dec8` (same hold) | (same hold) | 9.5 s | **ENGINE GATE PASS**, 0 `[FAIL]`, and the new row present 3×: `[PASS] … draft stage bill matches §09` |
| 8 | `engine/gate.py --fixture wide640f` **flag off** (tag `m1 gate wide640f off`) | 92 % | 40.9 s | **ENGINE GATE PASS** |
| 9 | `QWEN_MTP_DRAFT=1 engine/gate.py --fixture wide640f` (same hold) | (same hold) | 42.0 s | **ENGINE GATE PASS**, new row present 3× — the flag-on correctness pass runs to completion with no timing flags |

All nine are correctness reads. Rows 1–3 and 6–9 compare counts and bytes; the only numbers quoted from a
`STAGE` line are `launches_per_step`, `plan_per_step`, `plan_match`, `segments_per_step` and `uniform`.
Nothing here may enter `engine/performance-status.md`.

## 6. The wrong-count witness, in both directions

The brief asked for a deliberately mis-declared count that goes RED, shown and then restored. Three were
built, because there are three different places a count can be declared and each one's failure looks
different:

1. **The census line lies about the block** (`kBlockLaunches` printed as +1, `per_round` likewise). Result:
   `MTPDRAFT … block=1+14 head=1+1 per_round=17`, and `gate.py`'s own comparison function —
   `mtp_plan_verdicts`, the one section (h) runs — goes RED on exactly two rows:
   `draft block launches match §09`, `draft stage bill matches §09`. Witnessed against **real logs** by
   `prototype/mtp_accept/m1_wrongcount_check.py`: it reads the `MTPDRAFT` line and the `DRAFT round=` count
   out of a run's own stdout rather than a hand-written echo, and it also runs the *must-not-catch* side
   (the clean census → GREEN) and the **internally consistent** mis-declaration — bumping `block` and
   `per_round` together so `per_round == block + head` still holds — which reddens the same two rows. That
   last case is the one that matters, because it is exactly the mis-declaration the in-binary `plan_match`
   cannot see: both sides of its own arithmetic moved.
2. **The stage clock is billed one submission per round too many** (`set_plan_mtp(kStageSubsPerRound + 1)`).
   Result: `plan_per_step=240` against `launches_per_step=239` → `plan_match=0`, at both granularities,
   with `STAGEMTP … uniform=1` (so the red is the declaration, not a schedule that stopped being uniform).
3. **A launch inside the round stops being marked** (one `tally.mark` commented out). The census and the
   clock still agree with each other, and the *round* refuses: `marked 14 billed submissions across 15
   site(s), but the declared §09 bill is 15`. This is the case neither of the other two can catch, and it is
   why `RoundTally::check` exists — a plan that agrees with itself while the dispatch does something else is
   the failure `AGENTS.md` records over and over.

All three were reverted (`grep -c WRONGCOUNT engine/decode_run.cpp` → 0; the `tally.mark` line restored),
the binary rebuilt from the restored tree, and rows 6–9 of §5 were taken on the restored build.
`gate.py --selftest` is PASS with the two new derived cases in `selftest_mtp_census`.

## 7. M2 — the rollback question. Why the loop is not written yet, with the arithmetic

`[ ]` blocked on a decision; the options and their prices are in §4 and were sent to the supervisor.

The short version of why this is a decision and not a detail: in a k=1 speculative round the state a REJECT
needs is `S_{p+1}` — the state after the real token was consumed at position `p` and before the candidate
was consumed at `p+1` — and `engine/mtp_verify.h` already states, in its own header, that the batched 2-row
body cannot hand it over: the body is **layer-major** (all rows of layer `l`, then layer `l+1`), so no
point in its stream holds "every layer's state after row 0" at once. A snapshot taken before the body holds
`S_p`; the end of the body holds `S_{p+2}`. `S_{p+1}` exists only if someone materialises it, and the three
ways to get it have very different prices:

| option | what it costs | speedup at `a = 0.6641`, on the plan's own quoted terms | verdict |
|---|---|---|---|
| (V1) per-layer checkpoint **inside** the body: between each GDN layer's row-0 mixer work and its row-1 work, copy that layer's state + conv window (18 copies totalling 19.27 MiB, ~36 extra launches), restoring them on reject | `rho ≈ 0.16 ms` per round + `0.16 ms` on a reject, and a **new per-row split inside `submit_wave_body`** (P6 batched `gdn_conv`/`gdn_gates`/`gdn_gated_norm` into one launch per layer per BODY, so a per-row form has to be added to the shipped body path) | `1.6641 / (1.1505 + 0.336×0.030) ≈ **1.42×**` | clears the +10 % bar with room; costs an engine change to the path every prefill runs through |
| (V2) snapshot `S_p` before the body; on reject restore it and run the **one-row** body for row 0 anyway | 19.27 MiB restore + a whole step per reject | `1.6641 / (1.1505 + 0.336×1.015) ≈ **1.10×**` | this is the plan's "recompute" arm, and ACCEPT-PLAN says it is dead. It also fails the brief's "do not recompute the recurrence" literally |
| (V3) reject → restore `S_p` and re-anchor the next round at `p` with the candidate replaced by the verifier's own token | nothing extra per round | re-deriving row 0 the next round re-emits an already-emitted token; the pattern is 2 tokens per 2 bodies = **1.00×** | dead on arithmetic |
| (V4) row 0 = the shipping **one-row** step, row 1 = a second one-row body | 2 steps per round | `(1+a)/2 = **0.83×**` | this is `prototype/mtp_delta/D3-RESULT.md`'s "two sequential 1-row bodies — KILLED, structurally" |

The brief's three M2 bullets — row 0 by the unchanged shipping path, the existing 2-row body, and
snapshot/restore without recomputing — are only mutually satisfiable by **V1**, which is not a knob on the
accept loop but a change to the shipping body. So the question sent was: authorise V1 (and its scope), or
accept V2 with its price and its contradiction of "no recompute", or hold M2 until a shape is chosen. M1
landed in the meantime because M1 is needed by every one of those arms.

## 8. Device confirm list for the main agent (the arms are yours; nothing below was run by this worker for timing)

Every arm below is `decode_run` (does NOT self-lock) inside ONE `tools/gpu_lock.sh` hold per arm, blocking,
zero subagents. `IDS=/tmp/dec8_ids.csv` is `dec8`'s 17 ids; the corpus arms use `engine/chat.py`'s fixtures.
The **canary field is named per arm** — `plan_match` is compiled into the binary, so it agrees with a stale
build (`AGENTS.md`), which is why arms that do not use the stage clock must read a different field.

| arm | argv | expected wall | canary that proves the binary is mine |
|---|---|---|---|
| T1 baseline 128 | `decode_run --model models/Qwen3.5-0.8B --ids-file $IDS --steps 128 --time-stages` | ~15 s | `STAGE gran=stage TOTAL … launches_per_step=224 plan_per_step=224 plan_match=1` and **no `family=mtp` row, no `STAGEMTP` line** — their absence is the flag-off canary |
| T2 k=1 draft, 128 | same + `--mtp-draft 1 --mtp-sched every` | ~15 s | `STAGE gran=stage family=mtp launches_per_step=15` **and** `TOTAL … launches_per_step=239 plan_per_step=239 plan_match=1` **and** `STAGEMTP gran=stage steps=128 rounds=128 bill_per_round=15 uniform=1`. `uniform=1` is required: `uniform=0` means a cache-ceiling `DRAFT-SKIP`, and then the ms row is not one round per step |
| T3 k=1 draft, 4,000 lanes | same at a 4,000-lane prompt, `--steps 32 --mtp-cap 512` | ~40 s | as T2, plus `MTPDRAFT … rounds=32 refused=0` (`refused>0` invalidates the per-step bill) |
| T4 depth k=2 (if funded) | T2 + whatever `--mtp-depth` names later | T2 + k× the round | the `mtp` row must move to `45/step` only if the block's declared `block=` grows; `RoundTally::check` throws first if the declaration and the sites disagree |
| C1 flag-off byte identity | `PART=AB prototype/mtp_accept/m1_stage_witness.sh` in one hold | ~10 s | the pre-MTP binary must REJECT `--mtp-draft` (the script asserts this) so the pair is two builds |
| C2 gates | `engine/gate.py --fixture dec8`, `--fixture wide640f`, each with and without `QWEN_MTP_DRAFT=1` | 9.6 s / 41 s per pass | `[PASS] … draft stage bill matches §09` on the flag-on side; **zero** `draft ` verdict lines on the flag-off side |

## 9. Traps hit in this increment (for `AGENTS.md`, if the review wants them)

1. **The clock's per-step division is exact, so a side computation needs a uniform schedule to be billable
   at all.** `Clock::describe()` prints `subs / steps` and compares it to a per-step plan; a `loop`-scheduled
   draft makes ~1 round per `1+a` steps and has no integer per-step bill. The trap is that the failure looks
   like a bug in the counts, and it is a property of the schedule. Declaring the family forced the constraint
   into the open, and `--time-stages` now refuses `--mtp-sched loop` instead of truncating the plan.
2. **"16 submissions per round" and "15 billed submissions per round" are both right, and one constant would
   have been wrong for one of them.** The census line counts the blocking vocabulary D2H; the stage clock
   deliberately bills its `host_tail` row zero submissions. If they had shared a constant, either the census
   would have gone to 15 (breaking Topic A's `per_round == block + head`) or `plan_match` would have read 0 on
   a correct run.
3. **A declared count that is internally consistent is invisible to the binary that declared it.** Bumping
   `block` and `per_round` together keeps every in-binary sum true; only a Python-side re-count of §09
   catches it (`mtp_stage_plan()`), and only the round's own tally catches a launch that stopped being
   marked. Two instruments, two different blind spots, neither sufficient alone.


</content>
</invoke>

---

## 4a. Device pass 2 — what the split admission actually said (worker, uncommitted)

> **This section's diagnosis is superseded — read §4b before using anything below.** The conv was not the
> cause of the split arm's divergence; the recurrence span was. The measurements in this section stand,
> the reading of them in the last three paragraphs does not, and §4b carries the proof (one flag, `--gdn-loop 1`).
> Kept rather than rewritten because the dead figure is the record of how the confound was found.

Source: the tree as it stands, `models/Qwen3.5-0.8B`, dec8 ids, `--steps 4`, correctness only, **no
timing taken**, under `tools/gpu_lock.sh` (tag `mtp2-split-witness`, `util` 93 % at acquire). Two
arms, same shape, differing only in whether the second body is split:

```
engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file /tmp/dec8_ids.csv --steps 4 \
    --out /tmp/ms2 --mtp-verify 1 --mtp-midsplit 2      # control: second body UNSPLIT
    … same with --mtp-midsplit 1                        # split arm
```

| arm | `logits_ndiff` per round | `end_ndiff` | verdict |
|---|---|---|---|
| 2 — control (batched vs batched) | 0, 0, 0, 0 | 0, 0, 0, 0 | `split_eq_batch=1` every round |
| 1 — split (split vs batched) | 0, 248 320, 496 640, 496 638 | 0 | `split_eq_batch=0` from round 2 |

**D-b is admitted.** `verdict_eq_host=1` on every round of both arms: the device argmax of each logits
row equals `argmax_greedy` on the same row, and the accept bit equals `(row 0 argmax == candidate)`,
compared by `memcmp` — cond. 4's conditional met by exact equality, not by a band. The status word is
4 ints / 16 B per round (`MIDSPLIT … status_bytes=16`), against 1 986 560 B of logits downloaded
today by `mtp_delta`'s host fold.

**The control arm is the load-bearing part of this pass.** Two identical batched bodies over the same
anchor reproduce bit for bit — logits, mixer end state, verdict — so the harness, the shadow, the lane
revert and the double call are all sound, and the split arm's numbers are a property of the split and
of nothing else. A `--mtp-midsplit 2` that had also been dirty would have made the split arm
unreadable; keep it.

**V1's conv half is the suspect, and the reading below is the first thing to test, not a proven
cause.** `gdn_conv_step_kernel` derives its window base from the tile width: `shift = n > 3 ? 0 :
n % 3`, and the carried taps are read from `win + ch*3 + (3 - 1 - shift)`. A per-row call is `n = 1`,
`shift = 1`, base `+1`; the batched `n = 2` call is `shift = 2`, base `+0`. Calling that kernel once
per row therefore does not obviously reproduce the batched row's taps, and the mismatch would start at
row 0's conv and propagate. Note the one observation that does NOT fit it: **the first anchor (17) is
bit-exact in both rows** (`logits_ndiff=0`), and only anchors 18+ diverge — so whatever differs is not
uniformly wrong, and the base-offset reading has to be tested against that fact rather than assumed.
The shape of the ndiff (248 320 = `kVocab` exactly, then `2 × kVocab`) is the AGENTS.md signature of a
changed work split, not a wrong value, and §04.5's "the n=1 form is a different code path and is NOT
interchangeable with the batched one" is vindicated one level over: it is not interchangeable with a
per-row chain either.

**The cheaper next move, if V1 is kept.** The after-row-0 window is `roll(carried window, raw row 0)`
— a roll by one — which is the shape `kernels::gdn_conv_roll` (the `copy_fold` one-launch roller)
already implements. So V1 could keep the shipped batched `gdn_conv(n=2)` **exactly as it is**, which
would make both rows' conv bit-identical to the shipping form by construction rather than by
argument, and add ONE extra roller-shaped launch per GDN layer writing the after-row-0 window into the
checkpoint slot. That also removes the reason the recurrence had to be forced to span 1 for the conv's
sake. Whether it is worth doing is now a funding question, not a coding one: V2 (restore `S_p`, which
the snapshot already holds bit-exactly, and re-forward one row on reject) needs none of this. The
recurrence half of the split as implemented copies the buffer the recurrence was handed —
`st.gdn_state_layer(layer)`, which at that call site *is* the live buffer, so the mid copy's source is
not the bug. Second fact the next pass should know: for `n = 2` the batched body never advances
`c->conv_window` at all — the live-window copy sits under `if (n >= kGdnConvWindow)` — which is the
same bookkeeping that makes `QWEN_GDN_CONV_GUARD=0` mandatory for a 2-row body, and it means
"the window after row 0" must be *derived*, never read back off the live window.

Inertness with all of this present and flags off: `.venv/bin/python engine/gate.py --fixture dec8` →
**262 PASS, `ENGINE GATE PASS`**. (`engine/gate.py` needs the venv python and `python3` on PATH has no
numpy; that is worth knowing before the next person spends a minute on it.)

---

## 4b. CORRECTION to §4a — the conv was not the cause; the recurrence span was

Read this before using §4a's diagnosis. §4a put the split arm's whole-row divergence on
`gdn_conv_step_kernel`'s window base (`shift = n % 3`) and called the conv half "wrong". That reading
was taken from a kernel this body does not call. The conv `submit_wave_body` actually runs is
`prototype/gdn_decode/gdn_decode_kernels.h::conv_kernel`, and its window rule has NO dependence on the
tile width: `conv_tap` takes a carried tap from `win[(k + K - 1) * CH]` and a tile tap from
`raw[k * CH]`, so a per-row conv chain and a batched launch do address the same operands. The
divergence was elsewhere, and the proof is one flag:

```
--mtp-midsplit 1                     logits_ndiff per round: 0, 248 320, 496 640, 496 638   RED
--mtp-midsplit 1 --gdn-loop 1        logits_ndiff per round: 0, 0, 0, 0                     CLEAN
```

`--mtp-midsplit` arms a per-token-step recurrence (the checkpoint needs the gap that only a one-row
walk leaves), so at the shipped `--gdn-loop 512` the reference arm resolved to a span of 2 and ran the
LOOPED recurrence while the split arm ran the per-token one — two different recurrences, byte-compared
against each other. `DecodeScratch::gdn_geom`'s own note says the looped block shape "does NOT claim
bit-identity — the state and `o` dots re-associate", so the disagreement was predicted by a note in
the tree and I read the wrong kernel instead of that note. Both halves of §4a's conv paragraph are dead
figures now: the diagnosis, and the "the fix is cheap" paragraph that followed it.

Consequences kept, in order of importance:

1. **The control arm still earned its keep.** `--mtp-midsplit 2` (second body unsplit) was clean in
   every window, including the wrong-span one. That is what proved the harness sound and forced the
   search onto which *form* differed, rather than letting a red split arm be reported as a numerics
   result. It stays in the tree permanently (§4d trap 2).
2. **The split now pins its reference form, in code.** `--mtp-midsplit` without `--gdn-loop 1` is
   refused at parse time, with the measured numbers in the message, because a byte-compare across two
   recurrence forms is not an admission of anything. A pin that lives only in a recipe is a note, not a
   forcing (`AGENTS.md`: "a requirement written after the object it constrains is a note").
3. **The span pin is part of the feature, not of the witness.** An accept round that wants bit-identity
   against the shipped one-row step must run the per-token recurrence, because the one-row step does.

## 4c. V1's conv half, landed as a derivation (option (a), as authorised)

The shipped batched conv is **untouched** — `submit_wave_body`'s `ckpt` branch submits
`kernels::gdn_conv(g_raw, window, w, g_u, conv_new, n)`, the same call the non-checkpoint branch makes,
and then one derivation launch:

```
kernels::gdn_conv_mid_window(g_raw, window_before_body, ckpt->mid_conv + slot * kConvWindow * kConvChannels)
```

`engine/gdn_conv_mid_kernels.h` is the whole new surface: `conv_kernel`'s own end-condition loop
evaluated at `tokens = 1`, writing through the same `conv_tap` into a third buffer. It is bit-identical
by construction rather than by measurement, because it calls the pinned `conv_tap` and stores what it
returns — there is no re-spelled arithmetic to drift. It is issued BEFORE the shipped D2D that advances
the live window (it reads the window as the body found it), and the live window keeps exactly one
writer on every round, checkpointed or not — the constraint §4a recorded as a fact is enforced by
address, not by memory.

Admission, dec8 ids, `--steps 4`, correctness only, **no timing taken**, `tools/gpu_lock.sh` tag
`mtp2-convderive2` (`util` 92 % at acquire):

```
engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file /tmp/dec8_ids.csv --steps 4 \
    --out /tmp/fin1 --mtp-verify 1 --verify-guard 1 --mtp-midsplit 1 --gdn-loop 1
    … same with --mtp-midsplit 2 (control)
```

| arm | per round, all 4 rounds | close |
|---|---|---|
| 1 split | `split_eq_batch=1 logits_ndiff=0 logits_ndiff_row0=0 logits_ndiff_row1=0 end_ndiff=0 verdict_eq_host=1` | `MIDSPLITDONE rounds=4 split_eq_batch=1 logits_ndiff_total=0 end_ndiff_total=0 verdict_eq_host_bad=0 clean=1` |
| 2 control | same fields, all zero | `clean=0` by construction — `clean` requires the split arm |

So cond. 2 is met at anchors 18–20 as well as 17, with real content, both regions (logits and mixer end
state), by `memcmp` — no band, no floor, no pin moved. Flag-off inertness with all of this present:
`.venv/bin/python engine/gate.py --fixture dec8` → **262 PASS, `ENGINE GATE PASS`**.

One field was mis-specified by me and is now labelled rather than deleted: `mid_vs_step`, the mid
checkpoint against the state the shipping ONE-ROW step left. The body's row 0 runs the wave attention
path (`rows = 2` reaches `attn_prep` + staged core) while that step ran the fused single-row form, so
the two row-0 mixer states are argmax-equivalent and never bit-equal — the probe's claim was
`VERIFY2 row0_eq_step=1`, argmax, from the start. It prints `gdn_state_differs` on every round, which
is the expected label and not a rollback verdict; what adjudicates the round is the accept loop's own
continuation against the greedy pins. Deleting it would have removed the only place that number is
visible, which is the `--scope form` lesson applied one level down.

## 4d. Per-round launch and byte budget, itemised (no ms; the timing window attributes these)

Baseline for "extra" is the UNSPLIT 2-row body at the same pinned form (`--gdn-loop 1`), i.e. what the
verify probe already runs. 18 GDN layers per body, `mid_gdn` = 1,048,576 B/layer, `mid_conv` =
73,728 B/layer.

| per accept round | submissions | bytes |
|---|---|---|
| conv-window derivation (`conv_mid_window_kernel`) | +18 | writes 1,327,104 (reads ~1,327,104) |
| mid state capture, D2D after each layer's row-0 recurrence | +18 | 18,874,364 → 18,874,368 written |
| device verdict (`accept_verdict_kernel`, 2 blocks × 256 threads) | +1 | reads 2 × 993,280 B of logits, writes 16 |
| status-word readback, 4 ints D2H | +1 | 16 |
| **round total** | **+38** | **20,201,472 captured, 16 read back** |
| on a REJECT: restore `mid_conv`, restore `mid_gdn`, two bulk D2Ds | +2 | 20,201,472 restored |
| on an ACCEPT: nothing — the body's end state IS S_{p+2} | 0 | 0 |

Two lines to read with care. (i) If a future accept loop runs at a resolved span of 2 instead of
pinning one token-step per launch, add **+18 recurrence launches** to the round total: the split needs
the gap, and the gap is what the extra launch buys. (ii) The 16 B readback replaces what
`prototype/mtp_delta`'s host fold pays for a logits row, 1,986,560 B; that is a 124,160× cut on the
per-round status traffic and it is the only number in this section that is a win rather than a cost.

Memory, on the record: the checkpoint is a third ~19.27 MiB region per in-flight request alongside
`mtp_verify`'s snapshot and shadow when the features coexist (~57.8 MiB total). For this model that is
a rounding error against the ~34 GiB pool and implies nothing about paging or eviction
(`AGENTS.md`'s scope table).

## 4e. Two traps this pass earned its place in the tree

1. **A header whose namespace contains a `__global__` may be included by exactly ONE translation
   unit.** `engine/mtp_verdict_kernels.h` was included by `engine/kernels_lmhead.cpp` (which launches
   it) and by `engine/decode_run.cpp` (which only wanted the status-word offsets), and the link died on
   `duplicate symbol: engine::mtpverdict::__device_stub__accept_verdict_kernel` — a stub the compiler
   emits per TU for every `__global__`. `engine/attn_fattn_kernels.h` already carries this rule for the
   attention family; the general form is that the *integers* a host file needs must live in a
   kernel-free header (`engine/kernels.h` now holds `kVerdictWords`, `kVerdictRow0Token`,
   `kVerdictRow1Token`, `kVerdictAccept`, `kVerdictCandidate`, `kVerdictRows`, with the reason written
   at the declaration) and only the launcher's TU includes the kernel. The failure is a link error, so
   it is cheap to catch — provided the build is run, which is the same reason `AGENTS.md` asks for a
   build of the committed tree rather than a diff review.
2. **A control arm that differs from the feature arm by exactly one flag is what makes a red result
   attributable, so keep it permanently.** `--mtp-midsplit 2` runs the same second body UNSPLIT, so
   when the split arm read whole-row disagreements and the control read zeros, the only conclusion left
   was "a form differs somewhere in the split" — and the search that followed found the recurrence span
   (§4b). Without it, §4a would have shipped its confident and wrong conv diagnosis as the finding.
   `--mtp-midsplit` therefore takes `0` (off), `1` (split arm) and `2` (control arm) and keeps all
   three; the control is not scaffolding, it is the arm that makes the other one mean something. The
   third `MIDSPLITDONE` field pair (`clean=1` only for the split arm) exists so a control run cannot be
   read as a passing admission.

## 4f. The accept-rate witness, re-taken on the tree that has the derivation (and it is 20 s, not 2 min)

`ACCEPT-RESULT.md`'s headline was taken in a main-agent window with `RUN_GATE=1`, which is why it cost
116 s. The number itself does not need the gate or the torch witness, and it fits a worker's
correctness window. Re-taken on THIS tree (derivation kernel present, `--mtp-midsplit` refused without
the span pin, verdict kernel in the link), one lock hold tag `mtp2-convderive2`, `util` 92 % at acquire,
blocking, **no engine timing taken** — the wall quoted is the harness's own:

```
STAGES=corpus SCHED=loop OUT=/tmp/acc1 prototype/mtp_accept/run_accept.sh     # ~20 s wall
```

```
proposals made        953
proposals adjudicated 926
proposals skipped     27
accept OVERALL        615/926 = 0.6641     [denominator: 926/953]
accept by draft position   0-7:0.640(n=25) 8-15:0.578(n=230) 16-31:0.684(n=462) 32-63:0.718(n=209)
engine census (last)    rounds=953 adjudicated=926 accepted=615 refused=0 unadjudicated=27 lanes_max=16 sched=loop
census cross-check    harness 953/926/615 == census 953+0/926/615
```

**615/926 = 0.6641, reproduced exactly**, including the per-position profile and the census
cross-check, on a binary whose only new device work is a kernel the draft path never calls. Two notes
for whoever schedules next: the `STAGES` list is what made this cheap (`corpus` alone skips build, the
gate, the byte-identity legs and the torch witness), and the harness's own stage list is still the right
way to split it if a window is shorter still — `SCHED=every` is a separate 20 s and gives the 1101/1519
number.

## 4g. NOT landed this pass: the `--mtp-accept` loop shell, and exactly what stands between

**SUPERSEDED by §4i — the loop shell is landed.** What follows is kept because the two problems named
below are the reason §4i looks the way it does: (1) became the one-draft-per-round rule, corrected in
§4i.2; (2) became the refusal list and the two context end conditions in §4i.1.

Item 4 of the same brief was to move `prototype/mtp_accept/staging/mtp_accept.{h,cpp}` into the engine
behind `--mtp-accept`. It is not landed, and the reason is two named problems rather than a slip:

1. **Where a round's candidate comes from once a round can advance two positions.** `mtp_draft`'s
   `kLoop` schedule maintains its lane cache by *dropping* the lane on a reject, which is correct for a
   side pass that is only reproducing `greedy_mtp`'s round sequence. An accept loop owns the sequence:
   on an accept the target advanced to `p+2`, so the draft's next proposal must be for `p+3` from a
   cache holding `p+2` lanes; on a reject the target advanced to `p+1` and the draft's proposal for
   `p+1` is stale in a way `truncate_lane` does not express (the lane holds a draft of the *candidate*,
   not of the verifier's emitted token). Staging's `accept_round()` takes `candidate` as an argument and
   so does not name this; a driver that passes the draft's proposal through unchanged would be wrong on
   every accept, silently. The fix is a decision about the draft's state machine (who advances it, and
   what a reject re-drafts) and it belongs in the same commit as the driver, not smuggled into a shell.
2. **The driver shares a loop with three other things.** The generation loop in `decode_run.cpp` also
   owns §07.5's stop rule, the per-step logits dumps, `--replay`'s caller-owned tokens, and the stage
   clock's step records. Turning it into rounds means deciding, for each of those, what a two-token
   round does — e.g. a logits dump file per emitted token or per round, and whether `--replay` can span
   a round that emitted two tokens with the caller's feed listing one. Those are not hard; they are
   four small decisions that must be made in one place and witnessed, and doing them under a deadline
   while unable to run the rate witness is how a default-off path gets committed looking finished.

What IS ready for it: `Side`/`accept_round`/`Stats`/`tally`/`describe` in the staging directory, with
their cost figures now corrected to the derivation form (`subs_per_round = 2 * kNumGdnLayers + 2`,
`subs_per_reject = 2`, and the header's cost block quoting §4d's table), the rollback call sites
identical to what `--mtp-midsplit` just byte-compared, and the route forcing (hand GEMV, `kSeq`,
per-token recurrence) matching the pin §4b established. The remaining work is the driver and those two
decisions; the loop is not blocked on the engine.

Suggested order for the next pass, given the above: land the driver with `--mtp-accept` defaulting off,
run `--mtp-accept 1` on `dec8` for 8 steps as a *correctness* run (accepted + rejected both non-zero,
`MTPSTAT` internally consistent, `sampled.txt` identical to flag-off because greedy speculation cannot
change the output), then take `STAGES=corpus SCHED=loop` again as the rate witness. Do not let the
first witness be the rate: if the candidate bookkeeping is wrong in the silent way (1) describes, the
rate moves and nothing else does.

## 4h. Handed to the main agent's window

1. `wide640f` flag-off gate re-run (not taken here; it does not fit a worker's window):
   `.venv/bin/python engine/gate.py --fixture wide640f` — expected `ENGINE GATE PASS`, 262 PASS lines on
   `dec8` is the sibling number already re-taken here. Run with `tools/gpu_lock.sh` held; no timing
   should be taken from it.
2. The rate witness, when `--mtp-accept` exists (§4g): `STAGES=corpus SCHED=loop
   OUT=/tmp/acc1 prototype/mtp_accept/run_accept.sh`, expected fields `proposals made=953`,
   `proposals adjudicated=926`, `proposals skipped=27`, `accept OVERALL=615/926 = 0.6641`, and the
   `census cross-check` line agreeing with the `engine census` line. Reproduced on this tree as of §4f,
   so any future drift in those five numbers is the loop's, not the harness's.

### 4g.1 The candidate-source rule, resolved — blocker (1) above is answerable, not open

Working the geometry out while writing the handover, so the next pass starts from a decision instead of
a question. The rule that makes the draft cache consistent with the target by construction:

**The draft advances once per EMITTED token, never once per round.**

* Why it closes blocker (1). The draft's cache is "one lane per position the target has consumed", and
  an accept round consumes two positions and a reject one. Drafting per emitted token makes
  `st.nlanes == prefix_len` an invariant again instead of something the round schedule has to reason
  about, and it is the same rule `--mtp-sched every` already implements at width 1 — so it is a rule
  carried over, not one invented for the loop. A reject then costs one extra draft round (the discarded
  candidate's proposal is superseded by a draft of the verifier's own emitted token), which is the
  honest cost of a reject and belongs in `subs_per_reject`, where it is currently missing.
* Why the norm row is available, which was the part worth checking. `draft_round` needs the final-norm
  row of the token just emitted, and both branches already have it in `sc.norm`: an accept emitted row
  1's successor, so row 1 is the row to hand over; a reject emitted row 0's successor, so row 0. No
  re-norm, no extra launch, and `submit_wave_body`'s `norm` rows are the same buffer the head was just
  priced from, so the draft sees the norm the verifier itself used.
* The one consequence to decide, small but real: on a reject the draft for the emitted token runs
  *after* the two restore D2Ds, so it reads the restored state, not the speculative one. That is
  correct and cheap, but it means a reject's cost is `2 restores + 1 draft round`, not `2 restores`, and
  §4d's table should gain the draft line when the loop lands rather than pretend the draft is free.

So the remaining work in (1) is bookkeeping against a fixed rule, and (2) — the four decisions inside
the generation loop (per-token vs per-round logits dumps, `--replay` across a two-token round, §07.5's
stop rule checked per emitted token, and the stage-clock step record) — is listed in §4g with its
shape. Neither is a reason the loop could not have been written this pass; writing it and leaving it
unwitnessed, with the rate witness the only test that would catch the failure mode described in (1), is
the reason it was not written *and committed* this pass.

## 4i. The `--mtp-accept` loop shell is LANDED (worker, uncommitted), and it is lossless on `dec8`

Supervisor item 4, done. `engine/mtp_accept.{h,cpp}` are compiled (`engine/build.sh`), the round is wired
into the `--steps` generation loop of `engine/decode_run.cpp` behind `--mtp-accept 0|1` /
`QWEN_MTP_ACCEPT=0|1`, and `MTPSTAT` prints. Flag-off is inert by the same instrument the other M2 knobs
use: `.venv/bin/python engine/gate.py --fixture dec8` → **`ENGINE GATE PASS`**, taken after the loop was
wired, with the accept code in the binary.

**The admission that matters, and it is byte-compare: the emitted stream is the greedy stream.** Two runs
in one lock hold (`tools/gpu_lock.sh acquire mtp2-al5`, `util` 88 % at acquire, released, `status free`;
correctness only, **no timing taken**, `/tmp/wit_accept.sh`):

```text
engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file /tmp/dec8_ids.csv \
  --steps 16 --gdn-loop 1 --out /tmp/bl0                                       # flag off
engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file /tmp/dec8_ids.csv \
  --steps 16 --gdn-loop 1 --mtp-draft 1 --mtp-accept 1 --out /tmp/bl1          # rounds
```

```text
OFF (STEP kind=decode sampled, 16 tokens):
  6511 314 9338 369 2972 57590 159034 271 2064 369 279 7526 3177 303 9338 321
ON  (`emitted.txt`, first 16 of 17):
  6511 314 9338 369 2972 57590 159034 271 2064 369 279 7526 3177 303 9338 321
```

Identical, and the round mix underneath it was `rounds=8 accepted=7 rejected=1` — so the equality is not
one lucky path: three positions came from an accepted candidate plus a bonus, one reject restored the
mid-checkpoint (1 × 20,201,472 B) and emitted the verifier's token, and the stream still matches the
flag-off greedy stream element for element. The ON run's 17th token is one the flag-off run never
reached, because a round consumes one or two positions and 16 rounds cover more than 16.

`MTPSTAT`, verbatim:

```text
MTPSTAT rounds=8 accepted=7 rejected=1 emitted=15 accept_rate=0.875
  mean_tokens_per_round=1.875 body_rows=16 captured_bytes=161611776
  restored_bytes=20201472 status_d2h_bytes=128 split_subs=306
```

Every field reconciles against §4d's itemised bill, which is the point of printing them together:
`rounds == accepted + rejected` (7+1=8) and `emitted == 2·accepted + rejected` (15) are asserted by
`describe()`, which throws rather than describe an inconsistent tally; `captured_bytes == 8 ×
20,201,472`; `restored_bytes == 1 × 20,201,472`; `status_d2h_bytes == 8 × 16` — 16 bytes of vocabulary
question per round against the 1,986,560 B a host-side fold over both logits rows would download;
`body_rows == 2 × rounds`; `split_subs == 8 × 38 + 1 × 2` (§4d's `+38` per round and `+2` per reject).

**What the round refuses, and why each refusal is a missing object rather than a limitation.**
`--mtp-accept` without `--mtp-draft 1` (no draft block means no proposal, and a guessed candidate is not
the flag); at any `--gdn-loop` other than 1 (§4b's PIN, restated because here a wrong span would restore
a state the body never wrote — a wrong token, not a red comparison); with `--session`/`--sizes-only` (no
step list); with `--time-stages`/`--time-waves` (a round is not a step, so the clock has no per-step plan
to bill); with `--replay`/`--feed` (a round emits 1 or 2 tokens, so a per-step token list has no
alignment); with `--mtp-verify` (the round *is* the 2-row body; arming both would price a third body per
round and make the round's printed cost wrong); and below `kRows` of scratch. All six refusals were
run and their messages read back, not assumed.

### 4i.1 A third trap, found by the run rather than by reading

**A step that can consume two lanes inherits an end condition the one-lane loop never had.** `cap` is
sized as prompt plus `--steps`, one lane per step; speculation reaches `cap` with steps left in the
budget. The first accept run therefore died at the end of a perfectly good run —
`accept_round: anchor 25 + 2 rows is outside cap 26` — which reads as an engine fault and is the request
running out of context. Fixed in two pieces, both in the driver, because `submit_wave_body`'s and
`decode_step`'s refusals to write past `cap` are §08.6's and must stay: ask for room before the round and
pay a plain step when there is none (`ROUND-SKIP reason=context_limit`), and when even one lane is gone,
end the run (`RUN-END reason=context_exhausted pos= cap= budget_left=`). The general form: when you make a
step variable-width, the loop's fixed-size assumptions become your end conditions, and a fixed-size
refusal surfaced mid-run is a bug in the caller wearing the callee's error message.

### 4i.2 Correction to §4g.1 — the draft advances once per ROUND, not once per emitted token

§4g.1's rule was wrong in its detail and is superseded by the shipped one. An accept's second token was
never *fed*, so no final-norm row of the target exists behind it, and a per-emitted-token draft is not
implementable without inventing a norm row. What the loop does, and what `mtp_after_step` already did at
width 1: **one `draft_round` per round, with the norm row of the last position the body FED** (row 1 on an
accept, row 0 on a reject or a plain no-candidate step), `y_next` the last token emitted, and the proposed
position one past that token — `h` at q, `y_next` at q+1, proposal at q+2. A reject additionally truncates
the rejected proposal's draft lane, which is a length field and moves 0 bytes (`engine/mtp_draft.h`), the
same move the side pass makes. The consequence §4g.1 anticipated is real but smaller: a reject costs one
restore plus the ordinary one draft, not an extra one.

### 4i.3 Handed over: the accept rate, which is the priority number and does not fit a worker's window

`MTPSTAT`'s own rate on `dec8` is 7/8 — eight rounds is a sample of eight and is not the fixture. The
fixture number is the one to reproduce, and it needs 49 corpus lines through the new flag
(`prototype/mtp_accept/run_accept.sh` measures the side pass; the loop needs the same stages with
`--mtp-draft 1 --mtp-accept 1 --gdn-loop 1` and a sum over the `MTPSTAT` lines, ~3 min of device time
across the corpus, which is over a worker's budget):

```text
per corpus line L: engine/build/decode_run --model models/Qwen3.5-0.8B --ids-file <L> \
  --steps 32 --gdn-loop 1 --mtp-draft 1 --mtp-accept 1 --no-stop --out <dir>
sum over lines: accepted, rounds   →   a = accepted / rounds
```

Expected fields and the fixed lines, unchanged by this increment: the carried witness is
**`a = 615/926 = 0.6641`** (`ACCEPT-RESULT.md`, reproduced on this tree in §4f through the side pass) and
the kill line remains **`a < 0.4568`**. The loop's own round geometry differs from the side pass's in one
way that must be stated when the number is reported — the side pass drafts after a step and settles a
proposal against the next step, while the loop drafts from the last fed row of the round that consumed the
previous proposal — so agreement with 0.6641 is expected but not guaranteed digit-for-digit, and a loop
rate near but not equal to it is a geometry note, not a regression. No band, floor or pin was touched to
get any of the above.

## 4j. §4i.3's handover is DONE — in `M3-RESULT.md` (main-agent window, 2026-10-23)

All three handed-over items landed, in that order, and each one moved a claim:

- the engine now bills **its own decode window** (`DECWALL`: one wall around the generation loop, the
  prefill leg shown rather than subtracted, the denominator counted at the emit sites, and a `census=`
  that refuses a rate over disagreeing counters);
- the loop's own accept rate is **611/932 = 0.6556** on the corpus-49 population, against the kill line
  **0.4568** — **not tripped**, with the same rising-by-depth profile as the side pass. §4i.3's
  "agreement with 0.6641 is expected but not owed" arrived as 0.6556, and the loop's losslessness was
  re-adjudicated on all 49 lines rather than on `dec8` alone: 41 streams identical to the greedy stream
  over the common 32-token prefix, 8 departures, every one of them the flag-off row's own runner-up at a
  margin below one bf16 ULP of that row, and every one emitted by a round rather than by a plain step;
- the loop **does not clear the adopt line**. At 4,000 lanes one round costs **2.029 decode steps**, so
  k=1's ceiling at that width is **0.986× base at any accept rate** — measured −47.6 % on the timing
  fixture, −18 % projected at the corpus rate; short prose is −2.3 % against a parity requirement of
  a ≥ 0.691. `--mtp-accept` stays default-off and no ledger headline moves.

What the pass leaves as the live question: at the corpus accept rate, +10 % at 4k needs the round to cost
≤ 1.505 steps, and the known per-round extras (mid-checkpoint, restore, verdict readback, one draft) do
not account for the gap. The next measurement is an **attribution of the round**, not more accept-loop code.
Instruments: `prototype/mtp_accept/m3_loop_rate.{py,sh}`, `prototype/mtp_accept/mtp_wall_ab.sh`.
Numbers, protocol and traps: `prototype/mtp_accept/M3-RESULT.md`.
