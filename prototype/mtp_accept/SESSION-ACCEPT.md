# SESSION-ACCEPT — `--session --mtp-accept 1`: the accept round on the chat path

Status: implemented, builds clean (`bash engine/build.sh`, both binaries), and the lossless witness is
GREEN on the 25 cases of §6 — **but the main agent's device pass found a defect (§8) that blocks
making it chat's default**: a multi-turn session that crosses `MAXSC` during an early turn emits a
different token stream with rounds on than off, root cause NOT established. `engine/chat.sh`
therefore runs the plain step unless you name `--mtp`, and nothing here is committed as a default
change. No timing figure appears in this file: every device pass behind it was correctness-only, with
`no timing taken` recorded (§5), and the numbers that would size the prize stay the main agent's to
take (§7).

What changed, in one line: `decode_run --session` used to refuse `--mtp-accept`, and `engine/chat.py`
therefore pinned `--mtp-accept 0 --mtp-draft 0` to keep the REPL alive. Both halves are gone: the session
runs rounds, and chat runs the session with the engine's own speculative defaults, with `--no-mtp` as the
named plain arm.

The rule itself is not re-spelled. `engine::mtpaccept::session_body` calls the same sources the run loop
calls — `accept_round`, `tally`, `mtpdraft::{draft_round, draft_lane, truncate_lane, refusal}`,
`mtpadapt::Tracker` — and mirrors the run loop's round block (`decode_run.cpp`, the `if (aside)` branch)
step for step. The run loop was deliberately **not** refactored to share it: its prints are the
`flagoff_bytecheck.sh MODE=accept` compared set, and a shared body would have had to reproduce every one of
them byte for byte. See decision 6.

## 1. The arithmetic two of these decisions rest on

**GEN labels.** `session_loop` labels a `GEN` line by `st.prefix_len - 1` at the moment it prints — the
predicting row's position, one behind the sample it shows (this is the convention `tools/corpus_freerun.py`
and `engine/results.md` warn about: `gen[i]` pairs with `greedy[i]`). The run loop's `emit_pos[i]` is the
token's own absolute position instead. For a round anchored at `p = st.prefix_len` the two agree once you
shift by one, and the shift is not a fudge: a round leaves `prefix_len == p + n_emit` on **both** branches
— full accept `p + rows = p + k + 1`, partial `p + n + 1`, and `n_emit == n_accepted + 1` always — so emit
`i` sits at absolute position `p + i + 1` and prints under the label **`p + i`**. That identity is what
lets the session print a round's tokens with the same `GEN` grammar a flag-off session uses, and it is the
reason the lossless witness can compare `(token, pos)` pairs at all.

**One printed token == one consumed lane.** A flag-off `gen N` leaves `prefix_len = L0 + N`: every printed
token is consumed, and the sample produced after the last one is dropped (`sampled_token = -1`). A round
advances `prefix_len` by exactly `n_emit` and prints exactly `n_emit` tokens — the loop prints the token
the *previous* body left pending plus the body's first `n_emit - 1` — so `prefix_len - L0 == emitted` is
maintained per iteration with **no extra step anywhere**. (The first draft of this driver printed all
`n_emit` tokens and then handed the last back as `cur`, which printed every round's last token twice and
made `END prefix_len` disagree. That is what the `i + 1 < bd.n_emit` loop bound is guarding.)

**The stop cut's slot.** If a round's emit `i` is the stop id, the state must end with that token PRINTED
AND PENDING, i.e. `prefix_len = ` its own absolute position `p + i + 1`. `restore_prefix` reaches
`prefix_len = anchor + slot + 1`, so `slot = wanted_prefix_len - p - 1` — for a stop at emit `i`, `slot = i`
(and for a budget cut, had one ever been needed, `slot = j - 1`). The slot always exists: a stop that is not
the last emit has `i ≤ n_emit - 2 = n_accepted - 1 ≤ k - 1 = rows - 2`, which is `rows - 1` slots indexed
from 0. `restore_prefix` throws rather than guessing if that bound is ever violated.

## 2. The six decisions

**1. Budget: a round runs only when the turn can print its widest emit; no budget rollback exists.**
`session_body` runs a round iff `budget_left >= rows_of_k(round_k)` (and the other preconditions below);
otherwise it pays a plain step and prints `ROUND-SKIP reason=budget_tail anchor=… budget_left=… rows=…`.
So the last `rows - 1` tokens of every turn — at the shipped `k=2`, the last two — are plain steps.
Chosen over "run it and print only what fits" because that variant leaves the state ahead of the transcript,
which breaks the `feed` re-anchor, and over "cut at the budget with the mid-checkpoint" because it would
bill a 20 MiB device copy to answer a question the geometry answers for free.

**2. Stop: §07.5 per emitted token, in order, generation phase only; a mid-round stop cuts the state.**
Same rule and the same ordering as the run loop's `emit_one` tail: nothing behind a stop is emitted, because
it was only ever conditional on its predecessor. When the stop is the round's last emit the state already
ends where the transcript does and nothing is restored. When it is a mid-round emit — the draft proposed
248046 and the target's own argmax agreed with it — the surviving emits are suppressed, counted as
`suppressed=` on the `ACCEPT` line and `Session::suppressed` overall, and the state is cut back with
`restore_prefix(st, side, anchor, slot, rows)` (§1). The turn then ends with `stop=1` and
`sampled_token = <stop token>` — pending, unstepped — exactly as flag-off does, so the next `feed` steps it
and the transcripts agree. A prompt may legitimately CONTAIN the id; the stop test is generation-phase only,
and `stop-id-in-prompt` in the witness is the case that says so (§6).

**3. Draft cache and policy state are per-turn.** `session_new_turn` runs at every `feed` (and at `reset`):
`mtpdraft::reset()` on the cache, a fresh `mtpadapt::Tracker(k_ceiling)`, pending proposals cleared,
per-turn `Stats` zeroed with its geometry fields pre-set. Rationale: `engine/mtp_draft.h` has **no
eviction** — `reset` and the LIFO `truncate_lane` only — and `--mtp-cap` 512 lanes with
`mtp_attn_max_lanes() = MAXSC = 512`, so a cache carried across turns fills and then prints
`DRAFT-SKIP` forever; and abandonment is sticky, so a tracker carried across turns would silently kill
speculation for the life of the session after one bad answer. The `reset` verb already did the first half,
which is the precedent. On top of that, when `refusal()` fires mid-turn the driver does one printed
**`DRAFT-RESET reason=… would_propose_for=…`** and retries; only if it still refuses is the older
`DRAFT-SKIP` printed. A long single turn that fills the cache therefore keeps speculating, with every reset
in the log. Turn-boundary reset also means the first body of a turn is always a plain step (no proposal can
exist yet) — that step is what re-seeds the draft cache, exactly as at the start of a run.

**4. Adaptive depth is supported in session, per turn.** `--mtp-adaptive` (on by default with accept) is
armed with a `Tracker` whose ceiling is `--mtp-k`; `round_k = adapt->k()`, `note(round_k, emitted)` after
each round, and `need_next` — the number of proposals to draft — is read from the tracker *after* the note,
so a demotion or abandonment takes effect on the next body. `kPlain` reuses the existing "no proposal
pending" path (no new code path, no new numerics) and prints `MTPADAPT-PLAIN` once per turn. Refusing the
knob in session would have been the honest alternative; leaving it inert would have been the unacceptable
one, and this repo refuses a knob that cannot do anything.

**5. What is still refused in session, and why.** The `:1573` refusal narrowed to `--sizes-only` alone
("a run's generation steps … and `--sizes-only` runs no steps"). Everything retained is an instrument whose
object a session does not have, not a limitation of the round: `--mtp-dense` (the dense lane is staged from
inside the run loop's round iteration; the session driver never stages it, so the flag could only be inert
— NEW refusal, newly reachable), `--mtp-meter` (the meter is armed in the run path, below where
`session_loop` returns — also NEW, also newly reachable, and inert without it), `--mtp-prompt-lanes`
(no single prompt wave), `--mtp-draft-k` (a `feed` re-anchors the position map, so a ladder row would be
settled by a later turn's token), `--mtp-verify`, `--time-stages`/`--time-waves`, `--replay/--feed`,
`--state-fp`, `--prefill-logits all`. Every one exits 2 with a message naming the mode, and
`default_refusals.sh` asserts all of them plus the five positive session echoes (§6). Known pre-existing
gap, not introduced here and not newly reachable: `--session --out DIR` is ignored (a session has never
written dump files), so a lossless-on-dumps session claim is not available — the line-protocol witness
takes its place.

**6. The run loop was not refactored; `restore_prefix` is the single extraction.**
`accept_round`'s partial-accept branch used to spell its two bulk `hipMemcpyAsync` calls and its
`prefix_len` assignment inline. Those exact statements moved, verbatim, into
`long long restore_prefix(RequestState&, const Side&, int anchor, int slot, int rows)`, which both the
partial branch and the session's stop cut now call — one spelling of the byte-offset arithmetic that
`AGENTS.md` bills for (a `float *` plus a byte count walks four bytes per element and faults at slot 1).
The added bound is a throw on a slot the body did not capture. **Loudly stated for the reviewer:** the
claim that this is output-neutral rests on the move being textual — same two copies, same byte counts, same
`prefix_len` value, same `restored_bytes = side.slot_bytes()`, same `PhaseGuard kRollback` wrapping it — and
it is **not** witnessed by anything I ran, because a byte-compare needs two builds. The main agent should
run `MODE=accept prototype/mtp_accept/flagoff_bytecheck.sh <pre-change> <new> …` before believing it.

## 3. What the session driver prints (all non-terminal)

`ACCEPT round=… anchor=… fed=… cand=… cand_pos=… accept=… k=… n_accepted=… emitted=… pos=a->b
restored_bytes=… suppressed=…` (+ ` accept2=… cand2=… cand2_pos=… row2=… restored_slot=…` when `k>1`),
`MTPSTAT <describe(stats)>` once per turn that ran a round, `MTPADAPT`/`MTPADAPT-PLAIN` on a policy change,
`ROUND-SKIP reason=context_limit|short_proposal_set|budget_tail`, `DRAFT-SKIP`, `DRAFT-RESET`. Before
`READY`: `MTPACCEPT on=1 …`, `MTPGEOM k=… rows=… …`, `MTPADAPT on=1 …`. None of these is in `chat.py`'s
`TERMINAL` list, so the protocol is unchanged and an old driver ignores them; a turn that ran no round
prints no `MTPSTAT` at all, so "no speculation available" can never be read as "accept rate 0". Flag-off,
nothing here prints and no region is allocated — `session_loop`'s `else` branches are the pre-round code
paths, including the side-pass `LASTSAMPLE` line, which is emitted only when no session driver is armed.

## 4. Files

- `engine/mtp_accept.h` — `restore_prefix`, `Session`, `Body`, `session_new_turn`, `session_body`, with the
  budget/stop/headroom rules in comments; `+ adjudicated/accepted` census fields.
- `engine/mtp_accept.cpp` — `restore_prefix` (the extraction, and `accept_round`'s partial branch now calls
  it), `refused_after_reset`, `produce_proposals`, `session_new_turn`, `session_body`.
- `engine/decode_run.cpp` — the `:1573` narrowing plus the two new session refusals (`--mtp-meter`,
  `--mtp-dense`); the session-path arming block (side + tracker + `MTPACCEPT`/`MTPGEOM`/`MTPADAPT`) before
  `session_loop(...)`; `session_loop`'s new `gen` branch and the `feed`/`reset`/`stats`/`quit` hooks.
- `engine/chat.py` — `--no-mtp`, no pins by default, `MTPSTAT_FIELDS`/`MTPSTAT_K` + `mtp_annotation()`,
  docstring. `engine/chat.sh` — usage block and the not-a-ledger note.
- `prototype/mtp_accept/{session_lossless.py,session_lossless.sh,default_refusals.sh}`.

## 5. Device passes this worker ran (all correctness-only, `no timing taken`)

Every one: lock acquired/released by the worker, never held around a self-locking driver; `util` recorded at
the time (the box is serving this session's own tokens, so `util` is high and is discounted); no `ms` figure
taken or reported anywhere.

1. `printf 'feed 760 6511 314 9338 369\ngen 8\nstats\nquit\n' | engine/build/decode_run --session --cap 1024 --mtp-accept 1`
   — harness wall 0–2 s, `util` 51 %.
2. Two-arm transcript (`feed/gen 8/feed/gen 6/stats/quit`, `--cap 1024`, arms
   `--mtp-accept 0 --mtp-draft 0` vs `--mtp-accept 1`), one lock hold — harness wall 2 s, `util` 40 %.
   `diff` of the protocol lines: identical (this is the run that caught the double-print bug).
3. `prototype/mtp_accept/session_lossless.sh --only dec8,p8,smoke --gen 6 --long-gen 60` — wall 5 s,
   `util` 88 % at acquire, 55 % at release.
4. `prototype/mtp_accept/default_refusals.sh` — wall 20 s, `util` 12 %. **58 ok, 0 FAIL**, including the
   5 new positive session rows and the 9 retained/new refusals.

## 6. What the lossless witness proved, and its limits

`session_lossless.sh --only dec8,p8,smoke --gen 6 --long-gen 60` →
`LOSSLESS: the mtp arm emitted the same tokens at the same positions on every case`, 7/7 cases:
`fixture:dec8` / `fixture:p8` / `fixture:smoke` (6 GEN each), `multi-turn` (18 GEN, three turns, each of
which anchored rounds), `stop-terminated` (40 GEN, real `stop=1`, followed by another `gen`),
`stop-id-in-prompt` (stop id fed in the prompt, turn still ended `stop=0`), `draft-ceiling` (39 GEN).
Aggregate on that pass: `rounds=43 full-accept=32 reject=11`, `10 turns ran at least one round`.
`END steps= stop= prefix_len=` and `PREFIX p=` agreed on every case, so the state ended where the printed
transcript did each time — including after the stop cut.

What it did **not** prove, and the main agent owes the device pass for:

- **Width.** Prompts ≤ 21 ids, `cap` 1024/4608, so nothing above `MAXSC` and no straddling geometry; no
  templated long-chat fixture (thousands of lanes). The full population run is §7.1.
- **Turn counts.** Multi-turn was 3 turns at `gen 6`; the ceiling case never reached 512 draft lanes at
  `--long-gen 60`, so **`DRAFT-RESET` fired 0 times and decision 3's retry path is UNWITNESSED**. The full
  `--long-gen 700` case is the one that tests it.
- **The mid-round stop cut.** `restore_prefix` from a session round is only exercised if a round's non-final
  emit is 248046. None was on this pass (`suppressed=0` everywhere), so that path is implemented-but-untested
  on device. If it is unreachable in practice that is worth knowing; if it is reachable, a fixture that hits
  it is the missing witness.
- **k=1, `--mtp-adaptive 0`, and `--mtp-route amortised`** in session (only the default shape ran).
- **The run path.** Nothing here adjudicates `restore_prefix`'s effect on a run's byte-compared output (§2,
  decision 6), and no `gate.py` section was run.
- `default_refusals.sh` covers flag behaviour only; it exits at the parser for every `refused` row.

## 7. HANDOVER — commands for the main agent (device, exclusive, zero subagents)

The `.sh` drivers are self-locking; run them alone, not inside a lock hold. `grep -c gpu_lock` before
acquiring anything (AGENTS.md).

```bash
# 7.1 LOSSLESS WITNESS, full derived population (21 fixtures + 4 structural cases, both arms).
prototype/mtp_accept/session_lossless.sh                       # includes --long-gen 700 (DRAFT-RESET)
prototype/mtp_accept/session_lossless.sh --only dec8           # the narrow re-run if a case goes red

# 7.2 MANY-TURN LIVENESS: rounds must still anchor in the last turn, not just the first.
#     Expect `MTPSTAT` lines for ≥4 of the turns and no `DRAFT-SKIP` after a `DRAFT-RESET`.
printf 'feed 760 6511 314 9338 369\ngen 40\nfeed 13 198\ngen 40\nfeed 5156\ngen 40\n' \
  | engine/build/decode_run --session --cap 2048 | grep -cE '^MTPSTAT'
printf 'feed 760 6511 314 9338 369\n' > /dev/null   # (no-op separator, keep the transcript above single-line)

# 7.3 REFUSALS + session admission (58 cases, exits non-zero on any FAIL).
prototype/mtp_accept/default_refusals.sh

# 7.4 THE EXTRACTION'S OWN ADMISSION — run before believing decision 6.
git worktree add /tmp/prewt HEAD --detach && (cd /tmp/prewt && bash engine/build.sh)
MODE=accept prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run \
      engine/build/decode_run /tmp/dcp/dec8.csv 32

# 7.5 TIMING PAIRS — one window each, blocking, zero subagents, no other lock holder.
# WHICH INSTRUMENT: the stage clock is REFUSED with a round running (`--time-stages` → "a round is not a
# step"), so an off-vs-on decode pair has to be a WALL-CLOCK A/B, which is how every accept number in this
# directory was taken. Reuse the existing harness rather than inventing one, and assert a behavioural canary
# inside the window (AGENTS.md: a stale binary agrees with itself):
#   prototype/mtp_accept/m3_loop_rate.sh        # per-round rate + the flip-witness shape
#   prototype/mtp_accept/mtp_wall_ab.sh         # the run-path wall pair; note its `--gdn-loop` lesson
# The pair that matters for THIS change, at the shipped geometry (k=2, adaptive on):
#   OFF: --mtp-accept 0 --mtp-draft 0      ON: (nothing — the engine defaults)
# at 128 lanes and at the 4,032-lane headline width, both arms inside one lock hold, quoting the spread.
# Then the session figure nobody has ever taken: per-`gen` wall for the same transcript through
# `--session`, off vs on — the thing `chat.py`'s rate line is an estimate of.

# 7.6 CHAT-TURN WALL, per turn, on a templated transcript — NOT ledger numbers (chat.py's own rule).
engine/chat.sh --prompt "Summarise what this box is in three sentences." --no-mtp   # plain arm
engine/chat.sh --prompt "Summarise what this box is in three sentences."           # MTP arm; expect the
     # `| MTP k=2 accept=… emit=…/round` field on the per-turn line, absent if no round ran.
```

Timing rule inherited from `AGENTS.md`: quote a pair from one window, state the spread, lead with the
4,032-lane row, and remember `WAVEPASS wall_ms` and `DECWALL prefill_ms` are different rulers (1.22× apart
at 4k) and must never be subtracted. Nothing in this file is a ledger number, and `README.md` /
`engine/performance-status.md` were not touched.

## 8. THE DEFECT — a multi-turn session past `MAXSC` is NOT lossless (main agent, device, exclusive)

**Status: this is why `engine/chat.sh` runs the plain step by default.** `--mtp` opts in. Nothing
here is a timing; every run below was correctness-only with the lock held, no subagents active.

**Repro (durable, and it is the gate):** `prototype/mtp_accept/session_longturn.sh` — self-locking,
no timing, exits non-zero while the defect is present. Its case set IS the localization:

```
ok      A: LOSSLESS (257 protocol lines, 62 rounds)     one turn, generated lanes cross 512
DIFFER  B: first at: < GEN 11041 556  (2 of 265 lines, 73 rounds)   ...then a SECOND feed
ok      C: LOSSLESS (23 lines, 10 rounds)              one long wave, short turn
ok      D: LOSSLESS (145 lines, 88 rounds)             second feed, everything below 512
ok      F: LOSSLESS (187 lines, 103 rounds)            three turns, all below 512
ok      E: LOSSLESS (45 lines, 24 rounds)              wave prefilled ABOVE 512, then a second feed
DIFFER  G: first at: < GEN 3000 556  (2 of 265 lines, 76 rounds)    case B with --gdn-loop 1 on BOTH
```

Prompt: `dec300f` (295 ids, derived from `data/refs/`), `gen 240`, then `feed 13 198 760`, then
`gen 20`. Turn 1 matches on every one of its 240 tokens; the first difference is the token at
position 557, i.e. eighteen tokens into turn 2, and everything after it is a different continuation.

**What differs and what does not.** `END steps= stop= prefix_len=` and `PREFIX p=` agree on every
turn of every case, including the stop-cut case — the state ends exactly where the printed
transcript does. It is the TOKENS that differ. Off says `11041 279 2919 4101 26 424 10751 1040 264
21030 11 6103 6601`; on says `3000 685 279 2919 26 424 10751 1040 424 579 9668 1141 11155`. Those
are two legal continuations, not a shift of one another. The verdict that made it:
`ACCEPT round=73 anchor=554 fed=3070 cand=914 accept=1 k=2 n_accepted=2 emitted=3 row2=3000` — the
round's third row put 3000 where the plain step put 11041, and the two earlier rows of that same
round agreed with the plain step.

**Ruled out by measurement, not by argument.**

* **Not the draft-cache ceiling / not `DRAFT-RESET`.** Case B diverges identically at `--mtp-cap 24`
  and at the default 512. (Both of §6's missing witnesses now exist: at `--mtp-cap 24` a
  `DRAFT-RESET` fired and rounds continued afterwards; and a real mid-round stop was hit —
  `suppressed=1` with `cand2=248046` accepted as a non-final emit, `END steps=39 stop=1
  prefix_len=55`, and the off arm agrees on `prefix_len=55`. The stop cut restored the right slot.)
* **Not width alone.** Case A generates past `MAXSC` with rounds and is lossless; case E prefills
  above it with a wave and is lossless; case D does a second `feed` below it and is lossless. The
  trigger is the *combination*: lanes cross the ceiling during an earlier turn, and a later `feed`
  appends to that state.
* **Not simply the round's forced GDN span.** Case G runs `--gdn-loop 1` on both arms — the span the
  round forces regardless — and they still differ, but they swap branches: the plain arm moves to
  the `3000 685 ...` branch and the round arm to the `11041 ...` one. Whatever the two forms are,
  they are not just `--gdn-loop`.
* **Not the run path.** `MODE=accept prototype/mtp_accept/flagoff_bytecheck.sh <HEAD build> <this
  build> …` over a 120-id fixture: `emitted.txt` plus 78 verdict/emit/tally lines byte-identical,
  over 31 rounds of which **20 were rejects and 25 lines show non-zero `restored_bytes`** — so the
  `restore_prefix` extraction (§2, decision 6) is witnessed on the path that ships, which was
  §6's loudest gap.
* **Not a comparison artifact** — but the trap is real and worth recording: a run-mode accept arm
  writes `emitted.txt` and a plain arm writes `sampled.txt`, and they are **not index-aligned**
  (`emitted.txt` omits the sample the prompt wave itself produced). Compared by index the two look
  totally divergent when they are identical-but-shifted. Compare token SEQUENCES, and say which
  alignment you used — this repo has been bitten by that exact off-by-one before.

**What is NOT established.** — superseded by §9, which ran the experiment named at the end of this paragraph.
The comparison was made, on a session, with logits. What follows is kept as it was written (including the
wrong guess that the multi-row lane/partial addressing was the live candidate) so the record shows what was
known before the measurement.
Which form a round's row owes to the step it replaces above `MAXSC`:
the round body is a 3-row `submit_wave_body` and above the one-pass ceiling it runs the tiled/wide
attention forms (`ATTN_WIDE form=wide128llama … split=2`, `ATTN_FATTN form=fattn16`) while a 1-row
step does not, and `AGENTS.md` already bills "a path that exists only for `rows > 1` is invisible to
every one-row gate the tree has" and "two self-consistent layouts joined by a branch are invisible
to both sides". Case G says the span is not the whole story, which leaves the multi-row staging and
its lane/partial addressing as the live candidates. Nobody has compared a round row's logits against
the same position's step logits above the ceiling; that comparison is the next experiment, and it
needs a logits dump on a session (the session ignores `--out` — a pre-existing gap, §5).

**Consequences in force now.** `engine/chat.sh` / `chat.py` pass `--mtp-accept 0 --mtp-draft 0`
unless `--mtp` is named, and both files say why. The engine's own default is unchanged (a run
inherits speculation), because the run path is byte-witnessed; only the session default moved.

## 9. THE ROOT CAUSE — a sub-ULP tie above `MAXSC`, decided by a round (worker, device, correctness-only)

§8 ended with the experiment that was missing: compare a round's row against the same position's step row
above the ceiling, which needed logits on a session path that had none (§5). That experiment has been run. The
answer is **near-tie, not structural**, and the mechanism is narrower than §8 guessed.

### 9.1 The instrument

`decode_run --session --out DIR` plus a new session command `dump`:

    dump <pos>            arm a single position
    dump <min> <max>      arm a closed position range
    dump off              disarm

Nothing is written unless **both** `--out` and a `dump` command are present, so the protocol bytes of every
existing gate are untouched (`session_lossless.sh`, `session_longturn.sh`, `default_refusals.sh` all compare
transcripts, and a `dump`-less session prints exactly what it printed before). Each body that was priced
writes the F32 `kVocab`-wide logits row it decided, one file per row, and reports it:

    DUMP seq=<n> kind=<feed|step|round> pos=<p> emitted=<0|1> vocab=<n> bytes=<n> file=<name>

The position key is **the position whose row decided the printed token** — for a round, the row at
`anchor + i`; for a step, `prefix_len - 1`; for the `feed`'s own sample, the last fed position. That makes
the two arms' dumps join on `pos` with no shift, which is the whole point: the trap this repo has been bitten
by twice is comparing a stream against the wrong index. `emitted=0` marks a row that was priced and then
thrown away by a reject; `session_rowdiff.py` refuses to adjudicate one and counts them (`priced-then-dropped`)
so a window that covers nothing cannot read as agreement. `vocab`/`bytes` are echoed because a dtype read off
the file size alone is this repo's other standing trap.

`prototype/mtp_accept/session_rowdiff.py` is the comparator. It adjudicates every argmax flip in the tree's
existing form (`m3_loop_rate.py::flip_witness`): a flip is a legal gap between two bf16 forms only if the
token one arm emitted is the **runner-up** in the other arm's row **and** the top1−top2 margin is under one
bf16 ULP of that row's own top logit (`BF16_REL_ULP = 2**-8`, format-derived, not fitted). It prints logits,
margin, ULP, rank, both emitters, 1−cos, ndiff, max_abs, max_rel for each flip, prints `NO VERDICT` when
there are no joined rows, refuses to adjudicate positions past the first divergence (after which the arms are
fed different tokens), and `--selftest` proves the classifier can say STRUCTURAL (4 synthetic rows: a
runner-up at a wide margin and a third-ranked token at a narrow one both must read structural).

### 9.2 The verdict, in numbers (case B, reproduced with the instrument)

    first differing GEN line: index 259  off=GEN 11041 556  on=GEN 3000 556
    position 556  ABOVE the ceiling  emitted_by: off=step(seq 55) on=round(seq 70)
      OFF row:  top1=11041 logit 17.915483   top2=3000 logit 17.910400   margin 0.005083
      bf16 ULP of that top1 = 0.069982   margin/ULP = 0.0726
      ON row emitted 3000: its rank in the OFF row = 1 (0 = top1, 1 = runner-up)
      ON row's own top1=3000 logit 17.922333  margin 0.003332; the OFF token's rank there = 1
      row vs row: 1-cos 3.336e-05  ndiff 248318  max_abs 1.786e-01 at idx 7370  max_rel 6.928e-01
    -> NEAR-TIE

The tie is symmetric — each arm's token is the other row's runner-up, and each row's own margin is under one
ULP — so neither arm misread a distinguishable pair: **the pair was not distinguishable in the format it was
computed in**. 1−cos 3.3e-05 with ndiff ≈ every element is the signature of a re-associated long dot, not of
wrong operands (a wrong layout in exactly this family showed as cos 0.449 on ONE head-pair, not a gradient:
see §20.1 of `prototype/attn_decode/results.md`).

### 9.3 The controls that make that a verdict rather than a story

* **The accept arm's state does not drift.** Of the 37 positions both arms decided with a plain step, 36 are
  **bit-identical**; the exception is 557, one past the flip, where the arms are legitimately fed different
  tokens. The turn-boundary `feed` row at 537 is bit-identical and its argmax agrees (3000 both).
* **The flip is the tightest tie of the run by an order of magnitude.** Over the 18 positions the accept arm's
  rounds decided, margins in ULP units are min 0.0726 / median 21.88 / max 117.88; exactly two are below 1 ULP
  — 556 (0.0726, the flip) and 548 (0.9179, did not flip). Every other round row differed from the step row
  by the same band and moved no token.
* **The gap is not the GDN span the round forces.** With `--gdn-loop 1` on **both** arms (`session_trigger.sh
  --config gdn1both`), the same geometry flips at 556 with margin/ULP 0.1605 — a different tie, same
  reading. That is also why §8's case G swapped branches: a tie whose margin is a sixth of a ULP is broken by
  whichever legal form is standing.
* **The accept machinery does not touch a plain step.** `session_trigger.sh`'s first row, `no-rounds`, is a
  one-token turn whose only body is necessarily a step (a fresh `feed` has no proposals yet): the two arms'
  rows there are bit-identical (`step-pre 0/1`) in every config run. If the flag itself perturbed a step, that
  row goes red and the near-tie reading would be void.

### 9.4 What actually differs, and where

The trigger sweep (`prototype/mtp_accept/session_trigger.sh`, one process pair per config, prompt derived
from `data/refs/`, `cases == |case list|` asserted) says the form gap is **entirely a ceiling phenomenon**:

    case              cross?   verdict          rnd  min/ULP  sub-ULP  flip/ULP  roundgap  step-pre step-post
    no-rounds         no       LOSSLESS           0        -        0         -         -      0/1       -
    at=400 feed=3     no       LOSSLESS          16   42.67       0         -   0.00e+00    0/1  0.00e+00
    at=505 feed=3     no       LOSSLESS          17   24.61       0         -   2.63e-05    0/1  3.20e-05
    at=510 feed=3     no       LOSSLESS          16   48.49       0         -   2.78e-05    0/1  2.83e-05
    at=511 feed=3     no       LOSSLESS          16   39.37       0         -   2.87e-05    0/1  5.00e-05
    at=512 feed=3     no       LOSSLESS          16   37.14       0         -   3.33e-05    0/1  2.37e-05
    at=513 feed=3     crossed  LOSSLESS          17   29.64       0         -   3.14e-05    0/1  3.39e-05
    at=520 feed=3     crossed  LOSSLESS          18    1.62       0         -   3.75e-05    0/1  4.67e-05
    at=535 feed=3     crossed  DIFFERS at 556    18    0.0726      2     0.0726  3.55e-05    0/1       -
    at=535 feed=1     crossed  LOSSLESS          16    1.98       0         -   2.73e-05    0/1  3.11e-05
    at=535 feed=20    crossed  LOSSLESS          17   38.71       0         -   3.47e-05    0/1  2.66e-05
    at=540 feed=3     crossed  LOSSLESS          16    1.32       0         -   2.96e-05    0/1  3.44e-05
    at=535 config=fixed    crossed LOSSLESS      18    0.0726      2         -   3.36e-05    1/1  3.47e-05
    at=535 config=k1fixed  crossed DIFFERS at 556 17   0.9179     1     0.0726  3.34e-05
    at=535 config=k1     crossed DIFFERS at 556  17    0.9179     1     0.0726  3.32e-05
    at=535 config=gdn1both crossed DIFFERS at 556 18   0.1605     1     0.1605  3.49e-05

`roundgap` is the median 1−cos between a round's row and the flag-off step row at the same position;
`step-post` is the same measure for step rows that come after a round has run; `sub-ULP` counts the
round-decided positions whose margin is under one bf16 ULP of their own top logit; `flip/ULP` is that ratio
at the flip. Read together:

1. **Below the ceiling a 3-row body reproduces the 1-row step bit for bit.** `at=400`: 16 round-decided
   positions, `roundgap` exactly 0, and the downstream step rows also 0. The batching itself costs no bits.
2. **Above the ceiling the two forms differ, uniformly and by a hair.** 2.6–4.0e-05 1−cos at every
   above-ceiling width tested, with no outlier, in every config. That is the streaming/tiled attention forms a
   `rows > 1` body must take past `MAXSC` (the `ATTN_WIDE`/`ATTN_FATTN` forms §8 named) meeting the one-pass
   form a single-row step takes.
3. **The difference then lives in the state, not only in the row.** `step-post` is 0 below the ceiling and
   ~3e-05 above it: a step that follows a round inherits the hair, because the row's writes are already the
   wide form's. So the round is not "wrong above the ceiling" — it and the step are two legal spellings, and
   the spelling persists.
4. **Nothing flips unless there is a sub-ULP tie to flip.** Every geometry with `sub-ULP = 0` is LOSSLESS;
   every geometry that differs had a sub-ULP position and flipped exactly there, at that position's ratio. It
   is a necessary condition, not a sufficient one: `config=fixed` has two sub-ULP positions and never flipped,
   because its rounds ran from the start of turn 1 and its own row broke those ties the same way the step did.
   `--mtp-k` (1 / 2), the adaptive policy and the forced GDN span do not change the rule; they only change
   which positions the rounds get to decide. The crossing itself is neither necessary nor sufficient — case A
   crosses with 62 rounds and never met a tie, and `at=535` with a 1-token or a 20-token feed crossed and did
   not. What decides whether a geometry has a tie at all is the **text**: the same crossing with the same feed
   size but different fed ids has no sub-ULP position and stays lossless, which is why the fed ids are fixed
   in this tool (`FEED_TOKENS`) rather than derived, so a table row is the case it names.

So §8's symptom is not a session bug and not a crossing bug. Above `MAXSC` the accept round's row and the
step's row are two legal bf16 spellings of the same quantity; a long turn eventually hands such a spelling
a decision that is tied to within a seventh of a ULP, and the token flips. The session geometry is simply
where a long turn, a re-armed tracker and a wide prefix happen to line up: the per-turn adaptive tracker is
rebuilt at each `feed`, so in case B turn 1 abandoned speculation at round 62 (anchor 390, which is why its
rows were bit-identical steps) and turn 2 speculated from 539 — above the ceiling — from the start.

### 9.5 Why the run gates never saw it

`engine/gate.py` section (m) does carry the above-ceiling fixtures — `wide640f` (640 ids), `wide1024f`,
`wide4096f` are in `mtp_round_population()`, which is derived from `data/refs/` and names every exclusion — but
each is pinnable for only `min(len(greedy_continuation) - 1, FREE_RUN_BOUND = 8)` tokens, i.e. about four
rounds above the ceiling. Four rounds found no sub-ULP tie in any of them, and the probability is per-round
(§9.3: two sub-ULP positions in 18 ≈ 11 %), so eight-token witnesses are structurally blind to this. The
population gap was never "no wide fixture"; it was **no witness long enough above the ceiling**, plus three
concrete holes, now closed:

* `session_lossless.py` gains case **(f) `cross-then-feed`** — the crossing geometry the old case list could
  not express, since every prior case builds its prefix with a `feed` and so crosses in prefill. It asserts
  the turn really crossed (else `FAIL … never crossed`), that rounds ran, and that a round decided a position
  at or above the ceiling (else `FAIL … none decided a position above MAXSC`), so it cannot sit in the
  population as decoration. Measured: `crossed at 548 lanes (MAXSC=512)`, `65 rounds, 3 deciding at or above
  the ceiling`, LOSSLESS.
* `session_longturn.sh` no longer skips case E when its fixture is missing; a hole in the population prints
  `DIFFER … NO VERDICT` and sets the exit code.
* `session_gates.sh` is the entry point that runs `session_lossless.sh`, `session_longturn.sh` and
  `session_trigger.sh` and aggregates their exit codes. It refuses to run a driver that does not self-lock,
  because a caller holding `tools/gpu_lock.sh` around a self-locking driver parks the box silently.
* `session_trigger.sh` is the gate with teeth for this class: it goes red if any flip's margin is **at or above**
  one bf16 ULP of that row's top logit, or if the round-free control is not bit-exact, or if a flipped
  position has no dumped row. Exact-stream equality is no longer the only session assertion, and it is still
  asserted by the two drivers above.

### 9.6 Device passes this worker ran

All through the self-locking drivers, each under a minute per invocation, correctness only — **no timing
taken**, no millisecond figure appears in any of these files. `tools/gpu_lock.sh util` read 84–93 % at
acquire (the serving session's own polling; AGENTS.md says to record and discount it).

    prototype/mtp_accept/session_trigger.sh --at …            (the table above, several slices)
    prototype/mtp_accept/session_lossless.sh --only dec8      (6 cases incl. cross-then-feed: LOSSLESS)
    prototype/mtp_accept/session_longturn.sh                  (red: B and G DIFFER at 556; A C D E F ok)
    MODE=accept prototype/mtp_accept/flagoff_bytecheck.sh /tmp/hwt/engine/build/decode_run \
        engine/build/decode_run /tmp/dcp/dec8.csv 32
        CONTROL: IDENTICAL — emitted.txt byte-identical; DECISIONS IDENTICAL — 78 lines over 31 rounds.
        rc=0, so the `dump` instrument moves nothing on the run path.

### 9.7 The options, priced

The band question is the user's, and nothing here has been widened to make a gate pass: §9's flips are
reported by a gate that stays red.

* **(a) Speculate only while `prefix_len <= MAXSC`.** In `mtp_accept.cpp::session_body`, refuse the round
  branch when the turn's lanes have crossed — a handful of lines, and it restores exact-stream equality above
  the ceiling by construction, because then every above-ceiling row on both arms is a one-pass step and those
  are bit-identical (§9.3's control, and `at=400`'s zero `roundgap`). The price is the whole point of the
  feature: the accept round is worth most exactly where the step is most expensive, and at 4,032 lanes the
  step's attention term is 3.02 ms of the step (`engine/performance-status.md` §6, and `AGENTS.md`'s
  "decode is measured at 4k"), so (a) removes the win from long requests and keeps it only for short ones.
  It also makes the shipping default depend on prompt length in a way no other route does.
* **(b) Make the multi-row body reproduce the 1-row step's form above the ceiling.** The only option that
  keeps both exact equality and the speed. It means the wide/streaming attention forms must produce a
  bit-identical row for a lane that the one-pass form also could have produced — i.e. identical summation
  order and identical combine, per row, above `MAXSC` — while `MAXSC` exists precisely because one-pass above
  it does not fit. §9.4 shows what is owed: nothing is wrong, the two forms simply re-associate, and the
  difference (3e-05 1−cos) is the same class as the chaos floor in `specification/ASSUMPTIONS.md` A-18 (sdpa-vs-eager agrees 0.8776 of the way).
  Priced honestly: this is the §20 lane/partial geometry work plus the same argument for every other kernel the
  body batches, and it buys bit-identity that `specification/ASSUMPTIONS.md` A-18 says is not achievable past short depths. Not recommended.
* **(c) Document the band and adjudicate with it (recommended).** Keep the engine as is; make the session
  gates judge what they can judge: exact stream below the ceiling, decision-level equality above it — a flip is
  admissible only if it is the other arm's runner-up at a sub-ULP margin, which is `flip_witness`'s existing
  rule, format-derived and already the tree's standard. Concretely: `session_trigger.sh` enforces it (§9.5),
  `session_longturn.sh`/`session_lossless.py` stay red on B/G/`cross-then-feed` **until the band is adopted in
  those words**, `chat.py --mtp` help and `SESSION-ACCEPT.md` say that above 512 lanes a round can decide a
  tie the plain step decides the other way, and `engine/gate.py` section (m) is left alone (its 8-token
  witnesses stay exact). Cost: a documented band with an instrument that fails if a flip ever exceeds it. The
  risk it carries is the honest one — long generations are not bit-reproducible between the two forms, which
  A-18 already states as a property of bf16 on this checkpoint rather than a defect of this route.

**Recommendation: (c)**, with (a) kept on the shelf as the fallback if a flip is ever measured at an ordinary
margin — which `session_trigger.sh` would then report as `DIFFERS at an ordinary margin`, red, naming the row.

### 9.8 Files this section added

`engine/decode_run.cpp` (the `dump` command, `SessionDumper`), `engine/mtp_accept.{h,cpp}` (a priced body
carries its logits row, its base position and how many rows were emitted — a pointer copy, zero cost when no
dump is armed), `prototype/mtp_accept/session_rowdiff.py` (comparator + `--selftest`),
`prototype/mtp_accept/session_trigger.{py,sh}` (the margin sweep, `--selftest`, derived population,
round-free control), `prototype/mtp_accept/session_gates.sh` (entry point), plus `session_lossless.py` case
(f), `session_longturn.sh`'s no-silent-skip, and this section.

## 10. THE BAND IS ADOPTED — (c), in words, on 2026-10-25

The user's words were "take c", in the same request that asked for chat to run speculatively by default
("I'm using chat in the longer prompt, so enable it"). That is the decision §9.7 asked for, so §9.7's
proviso — "`session_longturn.sh` stays red on B/G **until the band is adopted in those words**" — is now
spent, and what replaced it is a predicate, not a waiver.

**What was adopted.** Below `MAXSC` the requirement is unchanged and exact: a round's deciding row is
bit-identical to the plain step's row, so a stream difference there is a defect and nothing excuses it.
Above `MAXSC` a difference is admissible exactly when `session_rowdiff.py` can witness it — every
flipped position has dumped rows on **both** arms, the token one arm emitted is the runner-up in the
other's row, and the top1−top2 margin is under one bf16 ULP of that row's top logit. `STRUCTURAL`,
`NO-FLIP` and `NO VERDICT` are all red: the band admits a witnessed tie and nothing else, and a case
whose window missed the flip fails rather than passing unexamined.

**What changed as a result.**

* `session_longturn.sh` now arms the session dump over each case's above-ceiling window (`%DUMP%`
  placeholder, windows computed from the derived prompt length) and adjudicates a differing case
  through `session_rowdiff.py`, printing `BAND` instead of `DIFFER` when the witness holds. Dumps are
  deliberately **not** armed below the ceiling, so the below-ceiling path has no excuse available to it.
* `session_gates.sh` now states three predicates instead of "this file is RED": lossless stays exact
  (and holds — it has never needed a waiver, `cross-then-feed` included), longturn is exact-or-BAND,
  trigger is the margin sweep. All three are green: A/C/D/E/F LOSSLESS, B and G BAND, trigger
  `unexplained by a sub-ULP margin 0`.
* `session_lossless.py` stays exact on purpose, with a note at case (f) saying that if it ever flips the
  answer is to adjudicate it, not to loosen this file.
* `engine/chat.{sh,py}`: speculation is the **default** and `--no-mtp` is the named plain arm
  (`--mtp` remains as an explicit statement of the default for scripts). Verified on device: default
  prints `MTP k=2 accept=0.416667 emit=2.33333/round`, `--no-mtp` prints no `MTP` field at all, and the
  two agreed token-for-token on that short prompt — which is the band's below-ceiling half, seen live.

**What this costs, stated plainly.** A long conversation can take a different branch than `--no-mtp` at
a position the model could not call (the flip that sized this sat at 0.073 of one bf16 ULP). That is not
reproducible between the two forms and no gate will make it so; A-18 already records it as a property of
bf16 on this checkpoint. The thing that would reopen this is a `STRUCTURAL` from either gate, and per
the fix-first rule that outranks speed work the moment it appears.
