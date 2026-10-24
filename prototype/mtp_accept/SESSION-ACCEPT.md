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

**What is NOT established.** Which form a round's row owes to the step it replaces above `MAXSC`:
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
