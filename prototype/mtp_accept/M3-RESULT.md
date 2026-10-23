# M3 — the loop's own rate and the loop's own wall

Measured 2026-10-23 in main-agent windows only: blocking, zero subagents, `tools/gpu_lock.sh` acquired and
released inside one shell per pass, `util` recorded at both ends and then discounted (it read 89–93 % at
every acquire, which is what `flash_serve` polling looks like and is not a gate — `AGENTS.md`).

`ACCEPT-BUILD.md` §4i.3 handed these over in this order, and the order is load-bearing: (1) give the engine
an honest per-run decode wall + emitted + rounds, (2) run the corpus-49 accept-rate witness on the loop
itself, because that is the only number that can kill this, (3) then the timing. Item 3 without item 1 is
how a harness came to print 20,689 tok/s for a stream of 45 tokens.

---

## 1. `DECWALL` — the engine reports its own decode window

**What was invalid.** `mtp_wall_ab.sh` used to subtract a `--steps 0` run from a `--steps 32` run to remove
prefill. A round can end the loop early (§07.5's stop, `RUN-END reason=context_exhausted`,
`ROUND-SKIP reason=context_limit`), so the two windows do not cover the same work. Symptoms in the old log:
2.17 ms of "decode" for 45 emitted tokens (20,689 tok/s), and negative windows on two reps. Separately,
and worse because it was silent: on the accept arm `RESULT ms_total` equals `prefill_ms` exactly (the round
path bills no per-step timer), so the line every harness was reading described only the prompt.

**What exists now.** One wall clock around the generation loop, printed before `RESULT` (so `RESULT` stays
the last line — the `tail -1` trap), on every run with a generation phase, not behind a flag:

```
DECWALL mode=run prefill_ms=784.417 decode_ms=728.983 bodies=32 rounds=31 plain=1 accepted=2 rejected=29
        suppressed=0 emitted=34 stop=0 ended=steps_exhausted ms_per_token=21.4407 tok_per_s=46.640 census=ok
```

In the window: every body the loop ran, plus the driver's own host work per body (its `STEP`/`ACCEMIT`
lines and their fflush — what a server pays anyway, and the same order on both arms). Out of it: the prefill
wave (reported as `prefill_ms` so the split is *shown* rather than subtracted), the `--state-fp` prompt-end
snapshot, and every measurement report below the line — `--time-stages` replays the run three more times
and none of that may enter a wall that claims to be the run's. The window closes with an explicit
`hipDeviceSynchronize`, because a reject's restore is a `hipMemcpyAsync` issued *after* the round's status
readback; without it the last round's copies would be retired by teardown and billed to nothing.

**`census=` is the part that makes it auditable, and it caught a bug in its own author's arithmetic.**
`emitted` is counted at the emit sites; `MTPSTAT`'s `emitted` comes from the device verdicts; `emitted.txt`
is a third count; the `ACCEPT round=` lines are a fourth. The harness and the engine both re-derive
`emitted == MTPSTAT.emitted + plain − suppressed`, where `suppressed` is an accept's bonus behind a stop
(priced by the body, never emitted). The first version incremented `emitted` in the accept arm's *plain*
sub-branch as well as in the shared `emit_one` tail, and the check fired immediately:

```
DECWALL ... rounds=4 plain=2 accepted=3 rejected=1 suppressed=0 emitted=11 ... note=emitted_vs_verdict_tally NO-RATE
                                    (MTPSTAT: rounds=4 accepted=3 rejected=1 emitted=7 → 7+2−0 = 9 ≠ 11)
```

Nine was right, eleven was double-counted. A rate over disagreeing counters prints `NO-RATE` instead — also
for `--steps 0` (`note=no_generation_bodies`), a zero window, a failed sync, or a window smaller than the
sum of its own nested timers. `ended=` names which exit the loop took, because `RESULT stopped=` is one
boolean and both §07.5's stop and a request that ran out of context set it.

**Boundary, stated rather than discovered:** the session REPL prints **no** `DECWALL` line — a session has
no run end (`gen` budgets come from the caller, the state re-anchors at every `feed`). That is noted at the
`--mtp-accept`/`--session` refusal, since "the engine prints a decode wall per run" is otherwise true with
one silent exception.

**Inertness.** `engine/gate.py --fixture dec8` → **`ENGINE GATE PASS`** with the change compiled in (lock
`main-agent decwall-gate`; no timing read from that run). The window's counters are host increments and the
sync retires work the run had already queued.

---

## 2. The shipping loop's accept rate: **a = 611/932 = 0.6556** — the kill line is not tripped

`prototype/mtp_accept/m3_loop_rate.{py,sh}` (98 s harness wall, one lock hold, tag `main-agent m3 loop
accept rate`; log in `prototype/mtp_accept/evidence/m3_loop_rate-2026-10-23.txt`). The population is
imported from `run_accept.load_population()`, which cross-checks `speccheck/fixtures/corpus.txt` against
`data/refs/corpus48/manifest.json` — the AGENTS.md trap row about an admission over a list typed from
memory is why it is imported and not reimplemented. Per line: `--steps 32 --gdn-loop 1 --no-stop`
`--mtp-draft 1 --mtp-accept 1`, and every relation re-derived from four independent counts before the line
is allowed to contribute to the sum.

```
proposals adjudicated (rounds)   932          lines 49/49, every line's census ok
accepted 611   rejected 321      emitted 1617   plain steps 74   suppressed bonus 0
mean tokens per round            1.7350
a = accepted/rounds              611/932 = 0.6556
carried side-pass figure          0.6641 (615/926)
kill line                         a < 0.4568        →  VERDICT: does not trip it (margin +0.1988)
by round ordinal   0-7: 0.594 (n=392)   8-15: 0.686 (n=392)   16-31: 0.736 (n=148)
```

`ACCEPT-BUILD.md` §4i.3 predicted agreement "expected but not owed" because the loop drafts from the last FED row of the
round that consumed the previous proposal while the side pass drafts after a step; 0.6556 against 0.6641,
with the same rising-by-depth profile, is that note arriving as digits. The sample floor is enforced too:
below 400 rounds the harness prints `INSUFFICIENT-SAMPLE` rather than a verdict, because a kill line drawn
on 926 proposals cannot be adjudicated by forty.

**The 0.0645 datum is now explained rather than excused.** The timing fixture (`ids4032.csv`, filler ids)
gives the loop `accept_rate=0.0645161` — the same figure the previous session recorded and read as a
synthetic-fixture artifact. It is one: on real text the same loop on the same binary measures 0.6556. A
fixture that has no next sentence to guess is a fixture the draft cannot guess. Reported here as a property
of the fixture, and *used* as one (§4 prices the arm at its worst accept rate).

---

## 3. The stream admission, re-run on all 49 lines, and what "lossless" is now owed

`ACCEPT-BUILD.md` §4g warns that a wrong candidate moves the *stream* and nothing else, so the rate alone
cannot be the witness. Every corpus line now also runs flag-off at the same `--gdn-loop 1`, and the streams
are compared. The first version of the harness treated any difference as fatal and died on line 3. It was
wrong to, and the reason is already in `ACCEPT-PLAN.md`: row 0 of the round is priced by the **2-row wave
body**, not by the fused 1-row step, and those two forms are not bit-identical (argmax SAME, sub-band
magnitudes). Byte equality of two different-but-legal bf16 continuations at depth is not an admissible ask
— A-18 says they agree 0.8776 of the way even between sdpa and eager.

So the flip is adjudicated as a decision question, against the flag-off run's **own logits at that row**:

> a flip is a legal tie-break iff the token the accept arm emitted is the flag-off row's **runner-up**, and
> the top1−top2 margin there is below **one BF16 ULP of that row's top logit** (`|top1| · 2⁻⁸`) — i.e. the
> two candidates were not distinguishable in the format they were computed in.

```
lines compared 49   identical 41   flipped 8   outside the near-tie band 0
 line 3  idx 22: greedy=1919 accept=2064  rank 1  margin 0.0311 < ulp 0.0853  line median 2.298  emitted_by=round_accept
 line 4  idx 22: greedy=21254 accept=321  rank 1  margin 0.0318 < ulp 0.0680  line median 0.901  emitted_by=round_accept
 line 12 idx  5: greedy=321 accept=314     rank 1  margin 0.0346 < ulp 0.0808  line median 0.924  emitted_by=round_accept
 line 15 idx  4: greedy=16 accept=32       rank 1  margin 0.0233 < ulp 0.0712  line median 2.052  emitted_by=round_accept
 line 24 idx  6: greedy=2434 accept=248068 rank 1  margin 0.0336 < ulp 0.0758  line median 2.410  emitted_by=round_accept
 line 25 idx  3: greedy=11 accept=15       rank 1  margin 0.0025 < ulp 0.0620  line median 4.160  emitted_by=round_reject
 line 26 idx  7: greedy=9175 accept=332    rank 1  margin 0.0583 < ulp 0.0932  line median 2.733  emitted_by=round_reject
 line 47 idx 28: greedy=561 accept=21095   rank 1  margin 0.0208 < ulp 0.0728  line median 1.186  emitted_by=round_accept
```

Two witnesses beyond the margin, both from the same data: every flipped token is the flag-off row's own
**runner-up** (rank 1, never deeper), and every one of them was emitted **by a round** — 6 by an accept, 2
by a reject's row 0 — and **none by a plain 1-row step**, which is the call the flag-off arm makes. A
bookkeeping bug (wrong restore, wrong fed token, mis-positioned lane) would move plain steps too and would
not take care to land on runner-ups at the tightest ties of each line.

What is owed now, stated exactly: with `--mtp-accept 1` the emitted stream **is** the greedy stream on
41/49 corpus lines to the full 32-token common prefix, and on the other 8 it departs only where the
flag-off row itself was decided by less than one bf16 ULP. Past such a flip the continuation is a different
text, not a wrong one — same class as A-18, same class as the wide-vs-one-row divergence already in the
traps table. `ACCEPT-BUILD.md` §4i's byte-compare on `dec8` still stands; it proved equality to 16 tokens, and no flip in this
population happened inside 3 tokens of the start except line 25.

**Disclosure, because the order matters.** Corpus line 3 was inspected *before* the criterion above was
written down, so "runner-up + sub-ULP" is not a band drawn before the first refusal — it is derived from the
format (bf16's 8 mantissa bits) and from the mechanism, and the raw margins, ranks and each line's own
margin distribution are printed so any reader can re-adjudicate with a different one. Nothing that was
already gated was widened to get this green.

---

## 4. The timing: the loop does not clear the adopt line, and at 4k it cannot

`prototype/mtp_accept/mtp_wall_ab.sh`, rewritten on `DECWALL` — no arithmetic across runs anywhere in it.
Canary asserted *inside* every rep's window (AGENTS.md: a paired A/B against a stale binary reads as a
clean null): exactly one `DECWALL`, `census=ok`, `PREFILL tokens=` equal to the field count read off the
fixture file, **no** `MTPSTAT` on base, `MTPSTAT rounds>0` on accept. Any rep failing a canary aborts before
its milliseconds are used. Both arms run `--gdn-loop 1`: the loop pins the per-token recurrence (`ACCEPT-BUILD.md` §4b), and
the first take of this A/B gave the base arm the default route, which showed a 300 ms prefill gap that had
nothing to do with speculation. Medians of 5 reps, one lock hold, no best-of-N.

4,000 lanes, `--steps 32`, `--cap 4300` (`evidence/mtp_wall_ab-4k-2026-10-23.txt`):

| arm | tok/s (median) | decode_ms | bodies | emitted | cost per body |
|---|---|---|---|---|---|
| base | 88.585 | 361.234 | 32 steps | 32 | **11.289 ms/step** |
| accept | 46.395 | 732.831 | 32 rounds | 34 | **22.901 ms/round** |

> one round costs **2.029 ×** one step. k = 1 emits at most 2 tokens per round, so the ceiling at this width,
> at **any** accept rate, is **0.986 × base**. Parity would need a = round/step − 1 = **1.029**, and a is a
> rate. Measured on this fixture: **−47.63 %** tok/s (it is the 0.0645-accept-rate fixture, i.e. the arm
> pays the speculation and gets nothing back). At the corpus rate instead: (1+0.6556)/2.029 = **0.816**, so
> **−18 %** even with a good accept rate.

Short prose (the `dec8` prompt, 17 lanes + 32 steps, `evidence/mtp_wall_ab-short-2026-10-23.txt`) — same
instrument, and here the round is cheaper than two steps:

| arm | tok/s | decode_ms | bodies | emitted | cost per body |
|---|---|---|---|---|---|
| base | 106.968 | 299.155 | 32 steps | 32 | 9.349 ms/step |
| accept | 104.542 | 363.489 | 23 rounds | 38 | 15.804 ms/round |

round/step = **1.691**, so the ceiling is 1.183× and parity needs a ≥ **0.691**; this prompt measures 0.727
and the run comes out at **−2.27 %** (the accept arm ended by `stop_token` at 38 tokens, the base arm by
`steps_exhausted` at 32 — `tok_per_s` is per emitted token on both, which is why the pair is comparable).
So even at short context, where the batched body buys ~16 %, the loop is at best break-even.

Two further numbers so nobody re-derives them wrongly:

- **The recurrence pin is not the cost.** At 4k, flag-off at the default GDN route vs `--gdn-loop 1`: 89.321
  vs 88.945 tok/s median of 3 in one window (`main-agent gdnloop-pin`) — −0.4 %, inside the noise. The
  loop's `ACCEPT-BUILD.md` §4b pin costs nothing; the round costs 2.029 steps.
- **Cross-window, not a paired claim:** today's paired base at 4k is 88.6–89.3 tok/s, where the carried
  ledger headline is 94.2–94.9. Per the standing rule those are different windows and are *not* a
  regression claim; every ratio above is quoted from within a single lock hold.

---

## 5. What this decides, and what it does not

- **The accept rate survives its kill line** (0.6556 vs 0.4568). Nothing in §2 asks for a band, a floor, or
  a re-derivation. The route is not dead because the model can be guessed.
- **The loop as built does not clear the pre-registered adopt line** (≥ +10 % tok/s at 4,000 lanes). Worse
  than missing it narrowly: at 4k the miss is *structural*, because one round already costs more than the
  two tokens it can pay for. `--mtp-accept` stays default-off, nothing is adopted, and no ledger headline
  moves (the base decode figures are unchanged by this work — `DECWALL` is a new line, not a faster engine).
- **The binding constraint is now the round's cost, not the draft's accuracy.** At the corpus accept rate,
  +10 % at 4k needs a round to cost ≤ 1.6556/1.1 = **1.505 steps**; today it costs **2.029**. The gap is
  measurable and unexplained, and it is the interesting question: the 2-row body amortises *nothing* at 4k
  (2.03×) while it amortises ~16 % at short context (1.69×), and the known per-round extras — mid-checkpoint
  capture 20,201,472 B, restore on reject the same, verdict kernel + 16 B readback, one draft round
  ~0.44 ms — do not come close to 11.6 ms. Next measurement is an attribution of the round (body vs
  checkpoint vs draft vs verdict), not more accept-loop code.
- **Not done here, deliberately:** no gate was loosened, no band moved, no `--mtp-*` default changed. The
  flag-on engine gate remains red on the undeclared-launch census for the *draft* side pass and is unrelated
  to this pass; `--mtp-accept` refuses `--time-stages`, so the round has no stage-family attribution and the
  wall A/B is the only instrument that prices it.

## 6. Reproduce

```
bash engine/build.sh                                                       # 26 s, no lock needed
prototype/mtp_accept/m3_loop_rate.sh                       # ~98 s, one lock hold, 49 lines × 2 arms
prototype/mtp_accept/mtp_wall_ab.sh                        # 4k, 5 reps × 2 arms, one lock hold
IDS=/tmp/dec8_ids.csv OUT=tmp/mtp_speed_short prototype/mtp_accept/mtp_wall_ab.sh    # short prose
.venv/bin/python3 engine/gate.py --fixture dec8            # lock held by the caller; ~2 min
```

`m3_loop_rate.sh` holds the lock in the shell for the whole pass (a short-lived subprocess would record a
dead holder pid, and `gpu_lock.sh` deliberately reclaims a dead holder's lock). Nothing here was backgrounded.

## 7. Traps this pass earned

| trap | where |
|---|---|
| **a counter incremented at two sites double-counts, and only a check with an *independent* right-hand side sees it.** `emitted` was bumped both in the accept arm's plain sub-branch and in the shared emit tail; the engine's own `census=` refused the run (`note=emitted_vs_verdict_tally`) precisely because that side came from the device verdicts rather than from the same emit sites. A census computed from the counters it is checking would have printed 11 and been believed | `engine/decode_run.cpp` (`DECWALL` block), §1 above |
| **an A/B that gives one arm a flag the other arm can also take prices the flag.** The first take of the wall A/B passed `--gdn-loop 1` to the accept arm only — it is the loop's *requirement* — and got a 300 ms prefill gap into the base arm's favour before any decode number was read. A required flag belongs on both arms or the pair measures two differences | `prototype/mtp_accept/mtp_wall_ab.sh` (`run()`), §4 |
| **"byte-identical or die" is not an adjudicable admission between two legal bf16 forms, and dying on it hides the real question.** The first M3 harness stopped at corpus line 3 and reported "the loop moved the target's stream" about a flip whose margin was the tightest tie on that line. The adjudicable form names the row's own runner-up and the row's own ULP, and reports margins/ranks/distributions so the reader can disagree | `prototype/mtp_accept/m3_loop_rate.py` (`flip_witness`), §3 |
| **a `key=value` parser whose value pattern is numeric silently drops every word-valued field.** `KV = r"(\w+)=(-?[\d.eE+-]+)"` found no `census` in a line whose census was `ok`, and the harness reported "the engine refused to bill this run" — a refusal that never happened, in a harness whose whole job was to notice refusals | `prototype/mtp_accept/m3_loop_rate.py` (`KV`), §3 |
| **a session prints no run-level line at all, and "the engine prints X per run" is a claim about one code path.** `--session` has its own loop and no run end; `DECWALL` is not printed there. Recorded at the refusal, because a missing line is the worst way for a check to be out of scope | `engine/decode_run.cpp` (session/`DECWALL` note) |
