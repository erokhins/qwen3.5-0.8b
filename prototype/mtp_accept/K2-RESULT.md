# K2-RESULT — the accept loop at k = 2: two candidates per round

**What this is.** `--mtp-k 1|2` (default **1**), a knob on the shipping `--mtp-accept` loop. At `k=2` a round
feeds a **3-row body** `[cur@p, cand1@p+1, cand2@p+2]`, preceded by **two draft rows** — the second conditioned
on the draft block's own `x_d` output row, the same feedback `--mtp-draft-k` measures
(`prototype/mtp_accept/DRAFT-COND.md` D4/D5) — and accepts the **longest correct prefix**: `cand1` only if the
verifier's row-0 argmax equals `cand1`, `cand2` only if it also equals row 1's argmax. Emitted: `n_accepted`
candidates plus the verifier's own token for the position after them, so 1..3 tokens per round.
`--mtp-k 1` is byte-identical to the pre-change binary (invariant 1 below).

**No millisecond figure appears in this file.** Every device run here was a correctness/rate gate under a
minute, with the lock held under its own TAG; the timing pass is the main agent's and the exact commands are
in §6. `tools/gpu_lock.sh util` read 91–93 % at each acquire (this session's own serving) and every number
below was taken blocking, one run at a time.

---

## 1. The geometry, in one place

| | `k=1` (default, shipped) | `k=2` (this change) |
|---|---|---|
| body rows | 2 | 3 |
| candidates per round | 1 | 2 |
| mid-checkpoint slots | 1 | 2 |
| verdict words (status D2H) | 4 ints = 16 B | 7 ints = 28 B |
| one checkpoint slot | 20,201,472 B (GDN 18,874,368 + conv 1,327,104, both from `state_sizes()`) | same |
| captured bytes per round | 20,201,472 | 40,402,944 |
| restored bytes per stop | 20,201,472 (one slot) | 20,201,472 (one slot — the slot the prefix stopped at) |
| declared split subs per round | 39 | 75 |
| draft lanes appended per round | 1 (+1 dense lane if accepted and `--mtp-dense`) | 2 (+1 dense lane) |

Every one of those is derived, not typed: `rows_of_k(k) = k + 1`, `kMidSlotsFor(rows) = rows - 1`,
`kVerdictWordsFor(rows) = 3*rows - 2`, and the byte figures come from `state_sizes()`
(`state_sizes()` in `engine/state.cpp`, which is also where those two sizes are defined). The engine states its
own geometry once per run on a **new line**, so every log says
which geometry produced it:

```
MTPGEOM k=2 rows=3 mid_slots=2 status_words=7 subs_per_round=75 captured_bytes=40402944 candidates_per_round=2
MTPGEOM k=1 rows=2 mid_slots=1 status_words=4 subs_per_round=39 captured_bytes=20201472 candidates_per_round=1
```

`MTPGEOM` is a **new** line rather than new fields on `MTPSTAT`/`MTPACCEPT`, because invariant 1 byte-compares
those; it is not compared by any harness and carries no decision.

**The verdict word, one formula.** `[0 .. rows-1]` per-row argmax, `[rows .. 2*rows-2]` per-candidate hit bit,
`[2*rows-1 .. ]` the candidate echoes the host validates against what it fed. At `rows=2` that is exactly the
legacy 4-word layout with the legacy indices (`static_assert`s in `engine/kernels.h` pin
`kVerdictWordsFor(kVerdictRows) == kVerdictWords` and `kVerdictWordsFor(3) == 7`), so k=1 reads the same four
ints at the same offsets and k=2 reads seven. `kMaxArgmaxRows = kVerdictRowsK2 = 3`, so every caller's partials
buffer follows one constant.

**The prefix is folded on the host from device-computed bits.** The device resolves the tie-rule-dependent
argmax and each row's single integer equality; the leading-run `AND` is arithmetic on ints already inside the
word it just read back, so there is no cross-block dependency and no third launch. `hit[i]` is row `i`'s
equality and `n_accepted` is the leading run: a `hit = [0,1]` (candidate 2 right while candidate 1 was wrong)
accepts **nothing** — see §5's non-nested diagnosis.

**Row-2 enforcement is a refusal, not a hope.** `submit_wave_body` refuses any checkpoint whose `slots != n-1`
("ONE slot per row except the last"), armed by `allocate_side` and by the verify probe (`verify->mid.slots = 1`),
plus `static_assert(kMidSlotsFor(kRowsK2) == 2)`. The width check lives where the width is, per the repo's
rank/shape rule (`engine/weights_contract.py`).

**`--mtp-dense` generalised rather than refused.** The dense lane stages the position the round consumed but did
not stage: it applies iff `n_accepted >= 1`, stages `emit[n_accepted-1]` at `anchor_after`, from norm row
`n_accepted-1`. At k=1 (`n_accepted = 1`) that is the previous code verbatim.

**Rejected candidates' lanes are truncated `k - n_accepted` times** (LIFO — they are the last lanes appended),
which generalises the single truncate k=1 did. A round whose **second proposal is missing** (the draft cache's
lane ceiling refused it) skips the round whole with `ROUND-SKIP reason=short_proposal_set` and truncates the
orphan lane, rather than silently running a 2-row round inside a `k=2` run — `body_rows=` and the win bar
cannot absorb a geometry that changes mid-run.

## 2. Invariant 1 — `--mtp-k 1` is byte-identical: **PASS**

Pre-change binary from a detached worktree at the pre-change tip (`554b89f`), same build script:

```
git worktree add /tmp/prewt HEAD --detach && (cd /tmp/prewt && ./engine/build.sh)
MODE=accept TAG="worker k2-bytecheck" \
  CANARY="--gdn-loop 1 --mtp-draft 1 --mtp-accept 1 --mtp-k 2" CANARY_MARK="^MTPGEOM k=2" \
  prototype/mtp_accept/flagoff_bytecheck.sh /tmp/prewt/engine/build/decode_run \
  engine/build/decode_run /tmp/dcp/dec8.csv 24
```

```
CONTROL: IDENTICAL — 1 dump file(s) byte-identical in accept mode
CONTROL: DECISIONS IDENTICAL — 66 lines (verdicts, emits, proposals, the tally) plus
        emitted.txt, byte-identical on /tmp/dcp/dec8.csv over 23 rounds.
CONTROL: CANARY-OK — the two sides of the byte compare are demonstrably different builds,
        one without the knob and one with it armed.
```

Only the bill fields the harness already normalises were stripped (`split_subs`, `stage_subs_per_round`); no
decision field was touched. One companion check, because this change also touched the draft-seed reserve: the
`--mtp-prompt-lanes` path at k=1 was run through both binaries (`--mtp-prompt-lanes 8 --steps 8` on `dec8`) and
its `MTPPROMPT` line is identical on both sides, `reserve=12` each — the reserve formula's k=1 value did not
move (§9). **The canary needs the whole flag set**, not just `--mtp-k 2`: `--mtp-k` without `--mtp-accept` is a
legitimate refusal (§4), so a bare-`--mtp-k` canary reports `CANARY: FAIL` on a correct binary (that is what
happened here first — a refusal working, not a knob missing).

## 3. Invariant 2 — losslessness at k=2: **PASS**

```
LOSSLESS=1 EXTRA="--mtp-k 2" prototype/mtp_accept/m3_loop_rate.sh          # or directly:
.venv/bin/python prototype/mtp_accept/m3_loop_rate.py --extra "--mtp-k 2" --limit 6 --steps 32 \
       --min-rounds 100 --lossless 1 --out /tmp/k2six
```

`lines compared 6 identical 6 flipped 0 of which outside the near-tie band 0` → `ADMISSION: PASS`. The accept
arm's `emitted.txt` equalled the flag-off greedy stream token for token on every one of the 13 corpus lines run
in this session (four runs, of 2, 2, 3 and 6 lines).

## 4. Invariant 3 — the gate: **PASS**

```
.venv/bin/python engine/gate.py --selftest --no-engine     ->  SELFTEST PASS
tools/gpu_lock.sh acquire <tag>; .venv/bin/python engine/gate.py --fixture dec8  ->  ENGINE GATE PASS
```

(`gate.py` does **not** self-acquire `tools/gpu_lock.sh` — checked with `grep -c gpu_lock engine/gate.py` → 0 —
so the caller must hold the lock around it, unlike `flagoff_bytecheck.sh` and `m3_loop_rate.py`, which do
self-lock and must not be called while holding it.)

**The §09 census `gate.py` adjudicates did not move.** `engine/gate.py`'s §09 count is the **draft block's**
stage bill (`MTPDRAFT`, `per_round=18`), which is the draft round's own plan and is untouched by widening the
*verdict* body. What moved is the accept round's declared bill in `engine/mtp_meter.cpp`'s registry
(slots/rows/word size/one-slot rollback), and `mtp_accept.cpp`'s `subs_per_round = 2*kNumGdnLayers*slots + 3`
→ 39 at k=1 (the figure `ROUND-TAX.md` §4d itemises), 75 at k=2.

## 5. Invariant 4 — the rate lands where the ladder said, and the non-nested count

`prototype/mtp_accept/DRAFTK-RESULT.md` priced the k=2 ladder at **2.143 tokens per round** on the templated
4,152-lane fixture. The harness now prints the same decomposition (`1 + n1 + n2`, over **rounds only**, so
plain steps' tokens are excluded from the denominator, and it cross-checks the engine's own
`mean_tokens_per_round`):

```
k=2 prefix ladder   n1 (cand1 accepted) = 31/43 = 0.7209   n2 (cand2 as well) = 19/43 = 0.4419
tokens per round 1+n1+n2  2.1628   MTPSTAT mean_tokens_per_round 2.1628   consistent
```
(3 corpus lines × 32 steps, `--mtp-k 2`; a 6-line/90-round sample read n1 0.6889, n2 0.3667, 2.0556 — short
prompts that end `context_exhausted`, so the sample is small and the full-population run in §6 is the number
that counts. **No expectation was adjusted for either reading.**) The wrapper path the §6 hand-off uses,
`LOSSLESS=1 EXTRA="--mtp-k 2" LIMIT=4 STEPS=32 prototype/mtp_accept/m3_loop_rate.sh`, read n1 0.7069,
n2 0.4138, **2.1207 tokens per round** — within 0.022 of the ladder's 2.143 on four lines.

At k=1 the harness prints what it printed before this change (`kill line ... judged as: a = accept/rounds`,
no ladder block), which is the harness-side half of invariant 1: a 2-line k=1 sample read a = 0.7037,
lossless, `VERDICT: ... does not trip the kill line`.

`dec8`, 47 rounds: `depth1=41` (0.872), full-prefix accepted 34, `mean_tokens_per_round=2.596`.

**Non-nested hits.** `ACCEPT ... accept=0 ... accept2=1` is candidate 2 agreeing with row 1 while candidate 1
disagreed with row 0. The fold accepts nothing in that case, which is the only lossless reading — the round
cannot skip a position — but it is a diagnosis worth counting, because `DRAFTK-RESULT.md` recorded 14/254 such
rows on the side pass. Counted by splitting the `ACCEPT` line's fields on whitespace and comparing whole fields:
**3 of 47 rounds on `dec8`** (of its 6 rejects, 3 had a row-1 hit under a row-0 miss) and 1 of 11 on the
`--mtp-dense 1` run, i.e. 6.4 % and 9 % against the side pass's 5.5 % — the same order, from a nearby
geometry. Reported, not tuned.

Counting them nearly produced a false zero, twice over, and both ways are worth keeping. A field must be
compared as a **whole field**: substring tests collide across `accept=`, `accept2=` and
`accept_rate=` on one line. And in this box's awk, `n+($0~/re/ && $0~/re/)` adds nothing at all — it prints
`n:` empty — so a tally written that way silently stays 0, which is this file's oldest failure class (a check
whose input list can be empty cannot fail) arriving as an awk idiom. The counts above come from an `if` body
with whole-field equality, and were confirmed by printing the three matching lines.

## 6. Hand-off: the runs that adjudicate k=2 (main agent, blocking, zero subagents, one lock hold)

The pre-registered pair, unchanged and not movable by implementation:

* **win bar** — k=2 at the templated 4,152-lane fixture reaches **≥ +13.6 % tok/s vs base**;
* **kill line** — measured **`round/step` ≥ 1.95** ⇒ stop and report.

```bash
# 1. The paired wall A/B. `ACCEPT_EXTRA` is the arm-specific difference; `--gdn-loop 1` is a REQUIRED flag
#    and therefore stays on both arms (AGENTS.md: a flag one arm requires still belongs on both arms).
#    STEPS=256 needs the cap above prompt+lanes-per-round or the run ends context_exhausted early.
#    This ONE run answers both pre-registered lines: its `accept vs base` percentage is the win bar, and its
#    `cost per body: base X ms/step  accept Y ms/round` plus `parity would need a* = round/step - 1` lines are
#    the kill line's `round/step`.
ACCEPT_EXTRA="--mtp-k 2" IDS=tmp/llama_input/ids_long_chat.csv STEPS=256 REPS=5 CAP=4600 \
    prototype/mtp_accept/mtp_wall_ab.sh
# 2. Optional attribution if (1) lands near either line: the per-phase meter on the same fixture, which is how
#    ROUND-TAX.md itemised a round. `decode_run` does not self-lock, so hold the lock around it.
tools/gpu_lock.sh acquire <tag> && engine/build/decode_run --model models/Qwen3.5-0.8B \
    --ids-file tmp/llama_input/ids_long_chat.csv --gdn-loop 1 --no-stop --mtp-draft 1 --mtp-accept 1 \
    --mtp-k 2 --cap 4600 --steps 256 --mtp-meter 1 --out /tmp/k2meter
# 3. The rate ladder on the whole population (this is the number §5's small sample was pointing at):
LOSSLESS=1 EXTRA="--mtp-k 2" prototype/mtp_accept/m3_loop_rate.sh
# 4. Re-run the byte-check against a fresh pre-change build if the accept path is touched again:
#    the CANARY must carry the whole flag set, per §2.
```

`mtp_wall_ab.sh`'s own canaries (a `DECWALL` with `census=ok`, no `MTPSTAT` on base, `MTPSTAT rounds>0` on the
accept arm, and `PREFILL tokens=` equal to the fixture's comma-field count) are unchanged and will abort
before printing a millisecond figure if the arms are not what they claim to be.

## 7. Refusal table — every combination exits 2 with a message naming the reason

```
--mtp-k 3                                                  --mtp-k takes 1 ... or 2 ..., got '3'. A deeper verify is unfunded
--mtp-k 2 (no --mtp-accept)                                --mtp-k says how many candidates an accept round verifies; with --mtp-accept 0 no round runs
--mtp-accept --mtp-k 2 --wave-rows 2                       needs a scratch at least 3 rows tall ... this run's --wave-rows is 2
--mtp-accept --mtp-k 2 --mtp-draft 0                       --mtp-accept verifies a proposal; with --mtp-draft 0 there is no draft block
--mtp-accept --mtp-k 2 --gdn-loop 512                      --mtp-accept pins --gdn-loop 1
--mtp-accept --mtp-k 2 --time-stages                       --time-stages bills the `mtp` family one draft round per step   [sibling]
--mtp-accept --mtp-k 2 --time-waves                        the wave clock prices a prefill pass                          [sibling]
--mtp-accept --mtp-k 2 --replay 1,2,3                      a round emits 1 or 2 tokens, so the caller's per-step token list has no alignment   [sibling]
--mtp-accept --mtp-k 2 --session                           --session hands the token budget to its caller                [sibling]
--mtp-accept --mtp-k 2 --sizes-only                        same message as --session                                     [sibling]
--mtp-accept --mtp-k 2 --mtp-verify 1                      the read-only --mtp-verify side pass is not also armed        [sibling]
--mtp-accept --mtp-k 2 --mtp-draft-k 2                     --mtp-accept makes its own proposals (one per candidate of its body, widened by --mtp-k)
```

**k=2 inherits `--mtp-accept`'s refusals rather than restating them.** There is deliberately no `--mtp-k`
clause for `--time-stages`/`--time-waves`/`--session`/`--sizes-only`/`--replay`/`--mtp-verify`: `--mtp-k`
requires `--mtp-accept`, and every one of those is already refused for *any* `--mtp-accept` run, so the
combination a new message would name is unreachable. A refusal no caller can reach is an untestable branch —
the reasoning `engine/kernels_attn.cpp` records for the wide-block pair — and the honest form is the note in
`engine/decode_run.cpp` plus the `[sibling]` rows above, each of which was run and produced a non-zero exit
with a message naming its mode.

## 8. What this found on the way (the one that mattered)

**A slot offset in bytes applied to a `float *`.** The rollback's two bulk copies read
`mid_conv + slot * bytes_conv`, where `mid_conv` is a `float *` and `bytes_conv` is a **byte** count from
`state_sizes()`. At `slot = 0` the offset is zero, so every k=1 run — and every gate, since every gate in this
tree runs k=1 — was right. At `slot = 1` the pointer advanced four bytes per element and the copy read a
quarter of the buffer past the end: a SIGSEGV inside `libamdhip`'s own memcpy, after the GDN copy had already
silently restored garbage into the live state (its over-read landed in mapped memory, so it faulted nowhere).
The shape of the failure is the lesson: `hipMalloc` had returned the full two slots, a whole-region `hipMemset`
returned success, the addresses printed at allocation and at restore matched, and a 64 KiB-chunked probe of the
same region read it fine — because the probe divided the same byte count by the same wrong unit only *after*
the bulk copy had already queued the bad one. The fix is byte arithmetic at the use site
(`(const char*)mid_conv + slot * bytes_conv`, `engine/mtp_accept.cpp`), and `restored_bytes` now reports
`slot_bytes()` (one slot) instead of `bytes()` (the whole capture), which is what the field always claimed.
Sibling finding, same category, found by reading the new refusal against its callers: `verify->mid.slots` was
never assigned, so the new `slots == n-1` width check would have refused the `--mtp-midsplit` probe outright —
set to 1 where its buffers are allocated.

## 9. Caveats a reader should hold onto

* **The draft lane ceiling is reached about twice as fast at k=2.** `usable_lanes = min(cap, MAXSC)` with
  `MAXSC = 512`, and a round appends one lane per candidate (plus a dense lane). The seeding reserve now scales
  with the geometry (`rows_of_k` lanes per round instead of a hard-coded 2 — arithmetically identical at k=1,
  and the same integer value, so no existing `MTPPROMPT` line moves: the pre-change and post-change binaries
  both printed `... reserve=12 ...` for `--mtp-prompt-lanes 8 --steps 8` on `dec8`), and a long k=2 run can
  still `DRAFT-SKIP`
  and then skip whole rounds with `ROUND-SKIP reason=short_proposal_set`. Those lines are the tell; a run that
  prints them is a mixture and should not be read as a uniform k=2 population. Witness that the k=2 seed path
  runs at all: on `dec8` with `--mtp-k 2 --mtp-prompt-lanes 8 --steps 8`,
  `MTPPROMPT wanted=8 fitted=8 fan=8 ... ceiling=512 reserve=18 lane_subs=72 clamped=none` — `reserve=18` is
  `(8 × 10 × 3 + 13) / 14`, the k=2 form of the line.
* **`ROUND-SKIP reason=short_proposal_set` pays a plain step.** `plain=` counts it, `DECWALL`'s emit identity
  still holds, but a rate read from a run with many of them is not a k=2 rate.
* **`mean_tokens_per_round` is over rounds, `tokens emitted` is over the run.** The harness prints both and
  says which is which; mixing them was a real inconsistency in an earlier version of the print.
* **Byte-identity claims name their width.** Invariant 1 was taken on `dec8` at 24 steps (23 rounds), the same
  case list `ROUND-TAX.md` used. It is a claim about the verdict/emit/decision lines and `emitted.txt` at that
  width, not about a 4k free run (A-18).
* **`MTPGEOM` is informational.** No harness compares it; `flagoff_bytecheck.sh`'s decision set deliberately
  does not include it, and adding it there would make a correct patch look like a defect.

## 10. Files this touched

`engine/mtp_accept.{h,cpp}` (Side geometry, the verdict fold, the rollback), `engine/mtp_verdict_kernels.h`
(per-row argmax + per-candidate hit + echo, one formula), `engine/kernels.h` + `engine/kernels_lmhead.cpp`
(`kVerdictWordsFor`, `lm_head_batched_xbf16` at 3 rows), `engine/kernels.h` + `engine/kernels_gdn.cpp` +
`engine/gdn_conv_mid_kernels.h` (`gdn_conv_mid_window(..., tokens)` through the pinned `conv_tap`),
`engine/decode_step.{h,cpp}` (`GdnCheckpoint::slots`, `kMaxCheckpointRows`, the `slots == n-1` refusal, the
per-slot capture sites), `engine/decode_run.cpp` (`--mtp-k`, the rewritten accept loop, `MTPGEOM`, the second
draft row, the seed reserve), `engine/mtp_verify.cpp` (`cand2 = -1`), `engine/mtp_meter.cpp` (registry notes);
`prototype/mtp_accept/m3_loop_rate.py` (the k=2 tally identity, the ladder print, and which quantity the
pre-registered kill line judges).

# §7 RESULT — the timing pass: k=2 beats llama.cpp's best MTP number

Bars were fixed before any measurement: **win bar ≥ +13.6 % tok/s at the templated 4,152-lane fixture**
(llama.cpp's `mtp2`, its best), **kill line `round/step ≥ 1.95`**. Main agent, blocking, zero subagents, each
arm paired against base inside one lock hold, medians of 5 reps, `STEPS=256 --cap 5200`.

| arm | round ms | round/step | vs base |
|---|---|---|---|
| k=2, sample 1 | 20.397 | 1.837 | **+16.53 %** |
| k=2, sample 2 | 20.386 | 1.818 | **+17.76 %** |
| k=2 + `--mtp-dense` + `--mtp-prompt-lanes 146` | 20.569 | 1.838 | **+20.68 %** |
| k=2 on **raw** (non-stemplated) 4,142 prose | 20.353 | 1.820 | **+16.09 %** |

Reference points: our k=1 after the round-tax removal was +9.75 %…+10.38 %; llama.cpp's `mtp1` is +9.5 % and
`mtp2` **+13.6 %**; its base step is 10.865 ms against our 11.10–11.21 ms. So on matched templated prose this
engine now speculates **~3 points better than llama.cpp's best**, and on raw prose — the check that keeps this
honest, since llama's ladder is chat-template output — **+16.09 %** against that same +13.6 %.

Kill line not hit: round/step landed at 1.818–1.838 against the 1.95 kill, so the 3-row geometry pays for
itself. Projection versus measured, booked plainly: the funding argument said ~1.80 step and ≈ +19 %; measured
1.82–1.84 step and +16.5/+17.8 % for k=2 alone, so the cost projection was right within 2 % and the gain
projection was ~2 points optimistic. Adding the two draft-conditioning knobs then overran the projection at
+20.68 %, which is the ladder's own story: k=2 is starved by `n₂`, and everything that lifts the draft lifts it
roughly twice.

**Losslessness at full population, not subset** — `LOSSLESS=1 EXTRA="--mtp-k 2" prototype/mtp_accept/m3_loop_rate.sh`:
**49/49 lines emitted stream identical to the plain greedy continuation**, 743 rounds, 1,617 emitted tokens =
**2.176 tokens/round**, against the independently measured ladder's 2.143 (so invariant 4 holds at scale, and
no expectation was moved). Per-depth accept histogram: rounds 0–7 → 0.615, 8–15 → 0.744, 16–31 → 0.478.
`engine/gate.py --fixture dec8` → `ENGINE GATE PASS` on the committed tree.

## What still stands between this and a default flip

1. **Gate section (h).** No gate in `engine/gate.py` runs a round at all — the same gap that let a
   route-dependent emitted-stream difference pass unnoticed for a session. Flipping `--mtp-accept` (at any k)
   default-on without it repeats a known mistake.
2. `kArgmaxChunks = 160` is an unswept guess.
3. k=3 is unmeasured. Our nested ladder suggests `1+n₁+n₂+n₃ = 2.428` at maybe ~2.05 step ⇒ ≈ +18 %, i.e.
   possibly the peak again, but the ladder decays by 0.285→0.179 per position and the draft cost is linear in
   k, so this is a measurement, not an argument.
