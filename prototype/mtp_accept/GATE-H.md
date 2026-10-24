# GATE-H — the MTP accept round has a gate: `engine/gate.py` section (m)

`prototype/mtp_accept/K2-RESULT.md` §7 recorded the missing gate — *"no invocation of
`engine/gate.py` runs a round at all"* — and named it the stated blocker on making MTP the default. It is
also the gap that let a route-dependent `emitted.txt` difference survive a whole session. This file is the
record of the gate that closes it: what it adjudicates, the commands that prove it can fail, and what it
cannot see.

Lives in `engine/gate.py` as section **(m)**, `the MTP accept round` (letter (m) because (h) is the batched
path's section). Every predicate is a token id, an integer counter, or an echoed geometry field — **no
millisecond figure appears anywhere in it**, so it is a correctness gate a worker may run under the
device-gate rules in `AGENTS.md`.

## Run it

```bash
tools/gpu_lock.sh acquire <TAG>
.venv/bin/python engine/gate.py --fixture dec8              # whole gate, section (m) included
.venv/bin/python engine/gate.py --fixture dec8 --mtp-k 2     # k follows the flag; default 1
tools/gpu_lock.sh release <TAG>
```

Measured on this box, lock held, `tools/gpu_lock.sh util` read at acquire/release (**no engine timing was
taken and none is recorded here**): section (m) alone over the full population 15–16 s wall at `--mtp-k 1`
and at `--mtp-k 2`; the whole gate 27 s wall (util 26 %→28 %). Both fit the under-a-minute worker budget.

## What it asserts, per case

1. **The population is derived, not typed.** Every directory under `data/refs/` with a `manifest.json`,
   minus four named-and-printed exclusions (`corpus48`, `corpus48engine`, `corpus48sdpa` are carried
   records tagged `NOT_A_PIN_FIXTURE`; `corpus48engine-smoke` has no manifest). 22 directories, 4 excluded,
   **18 cases**, and `cases == |population|` is a check, not an assumption. This is the trap that burned the
   session before: an admission list typed from memory reads plausible while seven of its names point at
   nothing.
2. **Losslessness against the PIN** — the pinned `greedy_continuation`, never a flag-off re-derivation of
   the engine (`prototype/mtp_accept/m3_loop_rate.py` compares against a flag-off arm, which is the right
   *rate* harness and the wrong *adjudicator*: a defect both arms share is invisible to it, `AGENTS.md`'s
   "a check whose reference side is derived from the thing under test cannot adjudicate it").
3. **The per-k tally identity, folded out of the `ACCEPT` lines**: `emitted == candidates + rounds`,
   `rounds == accepted + rejected`, `MTPSTAT emitted == DECWALL emitted`, one `ACCEPT` line per declared
   round, and `emitted.txt`'s line count == `DECWALL emitted`. The witness for each total comes from a
   different source than the total it checks.
4. **`MTPSTAT`/`DECWALL` agreement, with `NO-RATE` a FAIL** — a `census=` refusal is printed as what it is,
   never skipped, and the section reads the counters rather than inferring them.
5. **Rounds actually ran**: `rounds > 0` AND `len(ACCEPT lines) > 0` AND the two equal. A driver that parsed
   zero records cannot print PASS (the `gdn_collapse` empty-parse failure, one layer over).
6. **The geometry echo against the formulas that derive it** — `rows == k+1`, `mid_slots == rows-1`,
   `status_words == 3*rows-2 == status_bytes/4`, `candidates_per_round == k`, all off
   `engine/mtp_accept.h`'s own derivations, so admitting a swept k needs no edit in this file.
7. **The route, read back off the run** (`MTPSTAT route=hand gemm=hand gdn_scan=seq`) rather than assumed
   from the flag — a flag that silently fails to take is otherwise measured as if it had.
8. **Depth is stated, not assumed**: `min(len(greedy_continuation) - 1, FREE_RUN_BOUND = 8)`, because a FREE
   run cannot be pinned past A-18's bound and `--mtp-accept` refuses `--replay`. Four fixtures
   (`p8`, `p64`, `p300`, `wide4096p`) pin only the one token the prompt's last row samples; each prints its
   own `NOT ADJUDICABLE` line instead of passing a comparison it cannot make, and the section's note names
   the denominator (18 cases, 14 stream-adjudicated).

## The alignment: the stream is `[seed] + emitted`, not `emitted`

The obvious alignment — `emitted.txt[i] == greedy_continuation[i+1]` — is **one token short**, and shipping
it would have made the section's first red line a false one. An accept run's first body is a plain step: the
draft block needs a decoded step to draft from, so the prompt's own sample is **fed** to the first round
rather than emitted by it. Proved on device rather than reasoned:

| prefix24dec8 | stream |
|---|---|
| plain greedy path (`--steps 6`, no MTP) | `13 198 760 12263 16985 369 264` |
| plain greedy path **with `--gdn-loop 1`** | `13 198 760 12263 16985 369 264` (identical) |
| accept loop (`--mtp-accept 1 --mtp-k 1`) | `198 760 12263 16985 369 264 2526` — the same stream minus its head |
| pinned `greedy_continuation` | `13 198 760 23010 34130 369 264 2526` |

So the judged stream is `[seed] + emitted` against `greedy_continuation[0:]`, where the seed is read off the
run's own `STEP kind=prefill index=<last> … sampled=` line (`decode_run.cpp` prints one per prefill row and
only the last row has a sample). Consequences, all in `mtp_round_verdicts`:

* `stream[j] == greedy[j]`, i.e. `emitted[i] == greedy[i+1]` — §10.5 spelled out instead of guessed, and
  stream index 0 belongs to §06.5's **y0** pin (`PREFILL_ROW`), not to decode step 0.
* A run that prints no seed line **fails** rather than assuming an offset.
* chunk64b, read off `emitted` alone, looked like a round defect at its very first token (engine 2047, pin
  8337, and *no* band admits 2047 there). With the seed in the stream the fork moves to index 0, where the y0
  pin **TIES over ids [760, 1596] at 19.875 on one payload 0x419f** and D-1's proved tie admits the engine's
  1596 — upstream of any round, which is what the line says in as many words. `AGENTS.md`'s "a free run
  compared against the wrong index of the reference looks like total divergence", caught by stating the
  alignment.
* And the round's own fidelity is visible in the same three rows above: accept == plain-minus-head on both
  fixtures tried, and plain-with-`--gdn-loop-1` == plain-without, so the route the round *requires* moved
  nothing on these two.

## Adjudicating a divergence: `pin_admission`, the pin's own three rules

An accept run writes `emitted.txt` and per-row argmax words and **no logits**, so the engine side of the
tie/slack/room predicates does not exist here. A divergence is judged at the FIRST position only, by the
rules this file already owns, in sections (a)/(b)'s order and reused rather than re-spelled:

* **D-1's proved tie** (`pin_tie`): the pin has no winner, and the run's id must be one of the ids it tied at.
* **D-2's quantization slack** (`step_slack`): the pin's top1/top2 margin is under the element error this
  step's own band licenses, **and** the run's id is one of the pin's own leaders.
* **The cascade room rule** (`reference_flip_admissible`), on a library route only, exactly as section (a)
  arms it.

Downstream positions are reported as one echo, not counted as N defects. Both honest narrower points are
printed on the line: the engine's own gap is unavailable, and the room rule's attribution precondition is
satisfied by the geometry (every emitted token is some body row's argmax — the verifier's prediction, or a
candidate the verdict proved equal to its row's argmax).

Two cases out of 18 needed an admission, and both are rows the band already excuses:

| case | where | the pin's numbers | rule |
|---|---|---|---|
| `chunk64b` | stream index 0 (§06.5's y0) | ties over `[760, 1596]` at 19.875, one payload 0x419f | D-1 |
| `prefix24dec8` | stream index 3 (decode step 2) | 23010 over 12263 by 0.0625 = 1.0 BF16 ULP, licensed element error 0.1377, and 12263 is one of the two inside | D-2 |

Without an admission the section would be redder than the bands it belongs to — a gate nobody runs twice.
`--selftest` proves **both colours** on pin rows the selftest builds (a `loose` row whose margin is inside
its licensed error, a `tight` row whose is not, BF16-exact values so the case measures the rule and not the
reader's rounding), so neither colour depends on which fixture `--selftest` happens to be pointed at.

## k and route coverage — stated, not implied

* **`--mtp-k` is forwarded verbatim** and every geometry assertion derives from the formulas, so a k this
  build admits needs no edit here. **This build admits 1 and 2** (`kMaxKFor`, `kMidSlotsFor`,
  `kVerdictWordsFor`); at `--mtp-k 3` the engine refuses and the section reddens **all 18 cases plus the
  totals** — a refusal is a red gate, never a skip, so a swept k cannot read as coverage.
* **Route: `hand` only**, asserted off each run's `MTPSTAT route=` echo. `amortised` is **not** covered and
  the section prints that in its own output, because `prototype/mtp_accept/CORPUS-WALL.md` and
  `prototype/mtp_accept/ROUTE-FLAG.md` record it as *not output-neutral* — a green section (m) is coverage
  of the shipping path, not of the flag.

## Commands that prove the gate can fail

```bash
# 1. a k the build does not admit is coverage-red, not skipped. Section (m) runs on every fixture in the
#    derived population, so each one must come back RED with the engine's own refusal (`rc=2`), plus the
#    totals rows — measured on device: 19 failed rows and a non-zero exit, at `--mtp-k 3` and above.
tools/gpu_lock.sh acquire <TAG>
.venv/bin/python engine/gate.py --fixture dec8 --mtp-k 3
tools/gpu_lock.sh release <TAG>

# 2. the predicates' teeth, in process, no device. Expect: SELFTEST PASS, and the (m) lines below.
.venv/bin/python engine/gate.py --selftest --no-engine
```

The `(m)` lines `--selftest` prints, each of which is a *witness that the check can change colour*:

* `forcing the losslessness predicate to always-pass CHANGES THE COLOUR of the corrupted case` — the
  always-pass stub of `mtp_stream_matches` (the prefill-row lesson: a numerics predicate that cannot change
  colour under a stub was never a check), plus `and the stub changes nothing else`.
* `a corrupted hit bit (word 2 of rows=2, k=1) reddens the counters of exactly the case it touches` and the
  k=2 sibling (word 3 of rows=3) — an injected verdict word moves **one** case's fold, not the population's.
* `a hit-bit corruption does NOT redden the stream predicate` — the counters and the stream are independent
  predicates, so a PASS from one is not a PASS from the other.
* `a corrupted row-0 argmax word at step 3 of a DECIDED pin reddens the stream` / `and reddens NOTHING
  else` — must-catch, aimed at a step chosen by asking the admission which steps the pin actually decides.
* `the SAME corruption at a pin row that could NOT decide is ADMITTED, not counted as a defect` — the
  must-not-catch half, from the same injection.
* `D-1's proved tie admits an id the pin tied at` / `and refuses an id the pin did not tie at, even though
  the tie means there is no winner`.
* `a moved SEED forks at stream index 0 and is attributed to §06.5's y0 row, not to a decode step` — the
  alignment's own guard: attribution is asserted, not the colour, so it holds on every fixture.
* `a verdict word index at or beyond 3*rows-2 is refused rather than silently aimed at nothing` and
  `row 2's argmax on a round that accepted 0 candidates reddens nothing, and that is a fact about the
  prefix rule, not a missing check` — the blind spot is **derived from the geometry** (`kVerdictWordsFor`,
  the accepted-prefix rule), exactly the lesson of the hand-listed must-catch sets that rot into "everything".
* `deleting the ACCEPT lines reddens the rounds-actually-ran rows`, `a zero-round run is red even when every
  counter agrees with zero`, `a DECWALL that printed NO-RATE reddens instead of skipping`, `MTPSTAT route !=
  hand reddens`.

```bash
# 3. a post-hoc witness on a REAL captured record (CPU; the capture recipe is in the script's own header).
#    It prints the clean verdict, then one injection at a row the pin DECIDES (must redden) and one
#    downstream of the admitted fork (must change nothing), and exits 0 only if all three behave.
.venv/bin/python prototype/mtp_accept/gate_h_witness.py --fixture prefix24dec8 \
    --log <captured log> --out-dir <the run's --out dir>
```

Case (c) is the one worth reading twice: it is the *negative* control on real bytes, and its must-not-catch
status is a consequence of where the fork is, not a hand-written exemption.

## What section (m) cannot see

* **No logits, so no engine-side gap.** The three rules adjudicate the pin; sections (a)/(b) print the
  engine's own vector at the row and this one cannot. The verdict is unchanged (each rule's predicate is a
  statement about the pin), the printed line says which half is missing.
* **A y0 fork is attributed, not adjudicated away.** When the engine's own first token differs from the
  reference's, the line names §06.5's y0 row as the owner and says "the round did not move"; the row itself
  is policed with the engine's logits in sections (a)/(b). Section (m) cannot re-open it, and does not claim to.
* **Depth ≤ `FREE_RUN_BOUND` (8)**: past that a free run is A-18 territory and `--replay` is refused by the
  loop, so a long-generation token match is not and cannot be this gate's question.
* **Four of eighteen cases have no pinnable stream** — `p8`, `p64`, `p300`, `wide4096p`, each with
  `len(greedy_continuation) == 1`. They are counter-and-geometry coverage only and say so, and the note
  prints `14 of 18` rather than implying 18. (`smoke`, at 3 tokens, *is* adjudicated.)
* **`amortised` is not covered**; `--mtp-draft 0` is refused by `--mtp-accept`, so the section cannot
  separate draft quality from accept behaviour — that remains `m3_loop_rate.py`'s flag-off job.
* **k ≥ 3 is not covered by this build.** Extending `--mtp-k` is an engine change (re-derived 3-row pins,
  mid-checkpoint slots, `wave_wide` classification), and the refusal path is deliberately red so nobody
  mistakes the absence for a pass.
* **No timing.** The counters it reads are billed in the same lines a timing harness reads, and it ignores
  every `ms` field; the wall figures above are harness budget, not engine measurements, and no timing doc
  cites them.

## Traps this gate found, in its own words

1. **The emitted stream is missing its head.** `emitted.txt` is the greedy stream minus the seed, because the
   first body feeds that token. Aligning on `[0:]` against `greedy[1:]` is a one-token-short comparison that
   reads as a round defect at index 0 and is actually §06.5's y0 row two doors up.
2. **A pin read under the wrong fixture explains a divergence with the wrong fixture's numbers.** The first
   version restored `REFS` to `--fixture`'s value *before* judging, so a `chunk64b` line quoted `pin top1
   6511` — dec8's token — and was reddened and explained by a dump of another fixture entirely. REFS moves to
   the case and stays there until its verdicts are in.
3. **A bare `logits_last.npy` is not a step pin.** Reading the bare name (last-writer-wins across phases)
   produced two different "rooms" for the same row in the same session and nearly justified an admission that
   did not exist. `logits_pin(step)` is the only spelling that names a step.
4. **A parser that owns the exit status invents one.** `mtp_round_read_run` set `rec["rc"] = None` in its
   initial dict and `mtp_round_run_case`'s `update()` then clobbered the real return code, so every refusal
   parsed as a successful run with empty counters. The parser owns what the log said; the subprocess owns the
   status.
5. **Two self-consistent layouts joined by a branch, again:** an admission that decides by geometry must take
   its pin from the same geometry it is judging (stream index 0 is y0, index j is decode step j-1), or the
   line explains a row the run never stood on.
