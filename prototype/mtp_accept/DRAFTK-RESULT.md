# DRAFTK — the draft block's second position accepts at 0.44 nested / 0.50 marginal on matched prose: k=2 clears break-even there and nothing beyond it

**VERDICT.** On the chat-templated fixture the accept parity was established on
(`tmp/llama_input/ids_long_chat.csv`, 4,152 lanes), our draft block's second proposal accepts at
**m2 = 0.4961 (126/254) marginal, n2 = 0.4409 (112/254) nested, c2 = 0.6292 (112/178) conditional**, against
a first position at **m1 = 0.7020 (179/255)**. Judged against the two bars this ladder was funded to
answer: it **clears the beat-our-k=1 bar** (needs `n1+n2 > 1.093`; measured `1.1429`, projected on the
shipping loop's own `m1` `1.1418`) and **fails the +10 % bar** (needs `n1+n2 > 1.233`; `1.1429`) on **every
population measured, with no exception**. On raw prose and on the corpus it does not clear even
break-even as measured. Converted to throughput at the inherited k=2 round cost of ~2.03 step, the
templated number prices a k=2 round at `tok/round = 1 + n1 + n2 = 2.1429` → **+5.6 %** over base, against
the **+3.1 %** our k=1 arithmetic allows at its `a = 0.7008` (and the +2.85 % the wall actually measured):
**k=2 as measured is worth roughly two points of throughput more than k=1 on this text, not six, and it
loses money on the other two populations.** The `~0.5` the arithmetic needed is met by the marginal column
(0.496) and missed by the column that is apples-to-apples with llama's own counter (0.441 vs llama's 0.518).
That is the number, and it says "build k=2 only if the round body gets cheaper than 2.03 step", which is
the meter's question, not this one.

No millisecond figure appears in this file, in the harness, or in the logs it writes. Costs are stated as
COUNTS (submissions, tokens) and as multiples of a step, all inherited from `ACCEPT-APPLES.md` — this is a
rate pass and measured no cost of its own.

---

## 1. What was built, and which `h` each row consumes (the D8 label)

`decode_run --mtp-draft-k 1|2|3|4`, default-off, engine side (`engine/decode_run.cpp`). At each drafted
position it runs the §09 block k times:

| row | `y_next` | `h` | lane position | cost |
|---|---|---|---|---|
| 1 | the token the step emitted | `sc.norm` row 0 — the **plain one-row step's** final-norm row | `anchor + 2` | one round (16 submissions) |
| 2..k | the previous row's own proposal | the draft block's **own output row `x_d`** (`mtp.norm`'s output, §09.2) | `anchor + 2 .. anchor + 1+k` | one round each |

There is no target row behind a token the target never fed, so `x_d` is the only `h` row 2+ can have; it is
the same *kind* of object (post-norm, 1024 wide, same arena) and it is what llama.cpp feeds back in the
same role (`engines/llama.cpp/common/speculative.cpp:1770-1783` re-adds `id_sampled` with
`llama_get_embeddings_nextn_ith(ctx_dft, i_last)`, at position `pos0 + i + 1`). `x_d` is written only by
stage 14 and read only by stage 15 of its own call, so using it as the next call's `h` is a read that
strictly precedes that call's own write — one stream, no sync, and no silent rate change from a barrier.

**LABEL, and it matters.** Row 1's `h` is the **plain-step** row, which is D8's known difference from the
shipping accept loop (whose accept-case `h` is row 1 of a 2-row body — a legal bf16 *form* of the same
quantity, not its bits). Every rate below is therefore a **side-pass ladder on plain-step `h`**, directly
comparable with the side-pass figure `0.6641` and only projectable onto the accept loop. The projection
used in §6 is stated as one, with `c2` as the transferable quantity.

The side pass cannot take `--mtp-dense` (it never skips a position, so the flag would be inert), which is
why `--mtp-sched` shows up so strongly in the table: under `loop` the accepted round advances the anchor by
2 and the draft cache goes sparse (rope gap 2 between neighbouring lanes), and under `every` it stays
dense. That single axis moves the first column from 0.6306 to 0.7020 on the templated fixture — bigger
than any k effect, and the reason `every` is the headline schedule (a dense draft cache is what the
shipping accept loop has, via `--mtp-dense 1`).

Refusals, all five reachable and each tested to fire: without `--mtp-draft 1`; under `--session` (positions
re-anchor at every `feed`, so a row from one turn would be settled by a token of the next); under
`--replay` (the targets must be this run's own continuation, not a transcript); with `--mtp-accept 1` (its
body is 2 rows — a second proposal would be drafted, paid for, and never verified); with `--time-stages`
(k rows per anchor has no integer per-step `mtp` bill).

## 2. Three definitions, because llama's ladder is not the one that was asked for

llama's `#acc rate/pos` is `n_acc_tokens_per_pos[i] / n_call_accept`, incremented for `i < n_accepted`
(`common/speculative.cpp:3020-3031`). A position is inside `n_accepted` only when every position before it
was accepted, so **llama's ladder is a prefix (nested) rate by construction**, which is exactly why its own
identity `1 + Σ_i rate_i = #mean acc len` closes (long shape: `1 + 0.750 + 0.518 = 2.268` vs printed
`2.27`). Our side pass is not prefix-limited — the block *does* propose row 2 even when row 1 missed — so
it has to say which of three things it is reporting:

| name | definition | what it is for |
|---|---|---|
| MARGINAL `m_i` | `P(hit_i)` over every adjudicable row | row i's own quality. **Overstates** a loop, which never asks row i after a miss. |
| CONDITIONAL `c_i` | `P(hit_i \| hit_1..hit_{i-1})` | what an accept loop realises at row i; the quantity that transfers across `h` forms |
| NESTED `n_i` | `P(hit_1 ∧ .. ∧ hit_i)` | **the llama-comparable column**, and the only one with `tok/round = 1 + Σ n_i` |

`n_i = n_{i-1} · c_i`, so `c_2 = n_2 / n_1` always, and `c_2 = m_2 / m_1` **only if the hits are nested**.
They are not: rows where row 2 hit while row 1 missed are counted and printed (templated-every: 14 at
column 2, 21 at 3, 27 at 4). The practical consequence is that the marginal sum overstates a real k=4
round's tokens by `2.1980 − 2.1429 = 0.0551` on the headline population, so the bars in §6 are applied to
the **nested** sum and the marginal sum is shown beside it, never instead of it.

**The consistency check this repo would normally run, and what replaced it.** `1 + Σ m_i` against
mean-tokens-per-round is not available here: a side pass runs no rounds and emits nothing per round. The
substitutes, all of them aborting rather than warning, are (a) the ladder's first column equals the same
run's own `MTPDRAFT accepted/adjudicated`, per run, asserted in the reader — two independent joins of the
same `(pos, token)` pair; (b) the first column is **invariant in k** (§7, and it is what caught a real bug);
(c) every engine-printed `t`/`hit` is re-derived from the run's own `STEP kind=decode … pos_before=` lines
and compared, so a one-off index slide cannot survive; (d) the non-nested count is printed per column, so
"are the hits nested" is a measured fact with a denominator rather than an assumption.

## 3. The ladders

All runs: greedy, `--gdn-loop 1` (one row of the corpus table at the engine default, labelled),
`--steps 256` on the two 4k fixtures and 32 per corpus line, `--no-stop`, `--mtp-draft-k 4`.
`N|n` = hits/denominator; the denominator is stated per column, and a `-1` field (a target past the run's
last step, or a row the cache ceiling refused) is excluded from it and counted in the exclusion line.

### 3a. templated 4,152 — `ids_long_chat.csv`, `--mtp-sched every` (the headline)

256 drafted positions, 255 adjudicable at column 1; engine census `rounds=256 adjudicated=255
accepted=179 extra_rows=768 row_refused=0`.

| pos | MARGINAL m_i | NESTED n_i | CONDITIONAL c_i | non-nested |
|---|---|---|---|---|
| 1 | 0.7020 = 179/255 | 0.7020 = 179/255 | 0.7020 = 179/255 | — |
| 2 | 0.4961 = 126/254 | 0.4409 = 112/254 | 0.6292 = 112/178 | 14 |
| 3 | 0.3676 = 93/253 | 0.2846 = 72/253 | 0.6486 = 72/111 | 21 |
| 4 | 0.2857 = 72/252 | 0.1786 = 45/252 | 0.6338 = 45/71 | 27 |

exclusions `c1:1/0 c2:2/0 c3:2/1 c4:3/1` (no-target / no-proposal; the one no-proposal is the cache
ceiling refusing one row-4, printed as `DRAFTK-SKIP`, not dropped).

Same fixture, `--mtp-sched loop` (sparse draft cache): 157 positions — m1 `99/157 = 0.6306`,
m2 `71/156 = 0.4551`, n2 `64/156 = 0.4103`, c2 `64/98 = 0.6531`.

### 3b. raw 4,142 — `ids_long.csv`

| sched | pos | m1 | m2 | n2 | c2 | m3 | n3 | c3 | m4 | n4 | c4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| every | 256 | 0.6157 157/255 | 0.3622 92/254 | 0.3346 85/254 | 0.5449 85/156 | 0.2767 70/253 | 0.2213 56/253 | 0.6667 56/84 | 0.2460 62/252 | 0.1627 41/252 | 0.7455 41/55 |
| loop | 175 | 0.4629 81/175 | 0.2356 41/174 | 0.2069 36/174 | 0.4500 36/80 | 0.1667 29/174 | 0.1149 20/174 | 0.5556 20/36 | 0.1618 28/173 | 0.0809 14/173 | 0.7368 14/19 |

### 3c. corpus witness — all 49 lines of `speccheck/fixtures/corpus.txt` (the population
`m3_loop_rate.sh` uses, via `run_accept.load_population()`, derived and text-asserted), 32 steps/line,
`--mtp-sched loop`, k=4. Taken as two device chunks (lines 0-24, 25-48) and summed on counts.

| pos | `--gdn-loop 1`: m_i | n_i | c_i | non-nested | engine-default gdn-loop + §07.5 stop: m_i | n_i | c_i |
|---|---|---|---|---|---|---|---|
| 1 | 0.6645 = 616/927 | 0.6645 | — | — | **0.6641 = 615/926** | 0.6641 | — |
| 2 | 0.4013 = 360/897 | 0.3634 = 326/897 | 0.5507 = 326/592 | 34 | 0.4085 = 366/896 | 0.3717 = 333/896 | 0.5616 = 333/593 |
| 3 | 0.2606 = 228/875 | 0.1954 = 171/875 | 0.5463 = 171/313 | 57 | 0.2818 = 246/873 | 0.2119 = 185/873 | 0.5763 = 185/321 |
| 4 | 0.1751 = 146/834 | 0.1151 = 96/834 | 0.6038 = 96/159 | 50 | 0.1770 = 148/836 | 0.1208 = 101/836 | 0.5838 = 101/173 |

The right-hand three columns are the control population: its first column is **exactly the carried
side-pass rate 615/926 = 0.6641**, so the ladder's denominator machinery reproduces the shipped number to
the token (§7).

## 4. Our ladder beside llama's

llama's long shape is the same prose through its own chat template (4,152 template tokens, 128 emitted),
so §3a is the fair row to read against it. Both columns are **nested** rates, which is what makes the
comparison a comparison:

| position | ours, nested n_i, templated-every | ours, nested, corpus | llama.cpp `mtp4` long (0.698, 0.512, 0.419, 0.326) | llama.cpp `mtp4` short (0.829, 0.561, 0.415, 0.268) |
|---|---|---|---|---|
| 1 | 0.7020 (179/255) | 0.6641 (615/926) | 0.698 | 0.829 |
| 2 | 0.4409 (112/254) | 0.3717 (333/896) | 0.512 | 0.561 |
| 3 | 0.2846 (72/253) | 0.2119 (185/873) | 0.419 | 0.415 |
| 4 | 0.1786 (45/252) | 0.1208 (101/836) | 0.326 | 0.268 |

Each llama row comes from a separate run at its own `n_max`, so its rows do not share a token stream: the
ladder quoted in the brief (0.750, 0.518, 0.447, 0.326) is read across the `mtp1`/`mtp2`/`mtp3`/`mtp4`
runs, and the single-run `mtp4` row is what this table uses. Both spellings are in
`prototype/mtp_llama/evidence/llama_mtp_accept_ladder-2026-10-23.txt`; the difference between them is
run-to-run noise on 128 emitted tokens, which is also why the ordering claim below is stated as "not at
parity" rather than as a difference of 0.07.

Read plainly: **our first position is at parity (0.702 vs 0.698) and every later position is below llama's,
and the gap widens** — position 2 is 0.441 vs 0.512, position 3 is 0.285 vs 0.419. llama's draft degrades
more slowly with depth than ours does. (`ACCEPT-APPLES.md` established parity at position 1; this extends
that to "parity at 1, deficit from 2 onward", which is a draft-conditioning fact we do not yet have an
explanation for. The one measured per-position explanation available here — sparse vs dense draft cache,
`loop` vs `every` — moves position 1 by 0.07 and position 2 by 0.07 in the right direction, and does not
close a 0.07 gap at position 3.)

## 5. What it costs, in counts

One draft row is one round: **16 submissions** = 1 embedding D2D + 13 block launches + 1 head launch + 1
blocking vocabulary D2H (`kSubmissionsPerRound`, `engine/mtp_draft.h`). Per drafted position:

| k | submissions per position | extra vs shipped k=1 | extra rows measured (templated-every) |
|---|---|---|---|
| 1 | 16 | 0 | 0 |
| 2 | 32 | +16 | 256 |
| 3 | 48 | +32 | 512 |
| 4 | 64 | +48 | 768 |

The head pass — a second full walk of the tied 485 MiB table plus a 248,320-float D2H — is per ROW, not per
anchor, so it is the term that makes k linear rather than cheap; the two vocabulary round trips that
`0b3d57e` is about to take off the device would shrink it, and k=2 is cheaper after that commit than before
it. Round cost in step units (inherited, not measured here): k=1 ≈ 1.65 step, k=2 ≈ 2.03 step.

## 6. The two bars, and the verdict against each

Bars as handed down, dimensionless: a k=2 round at ~2.03 step must beat our measured k=1
(`1.7008/1.65 = 1.031`), so it needs `m1 + m2 > 1.093` (`c2 > 0.487`); the +10 % line needs `> 1.233`
(`c2 > 0.678`). Applied to the **nested** sum, with the marginal sum and a llama reference point beside
them. The projection column takes the shipping loop's own first position (`m1 = 0.7008`,
`ACCEPT-APPLES.md`) and our measured `c2`, which is the only honest way to cross the D8 `h` gap — it is a
projection and is labelled as one everywhere.

| population | n1+n2 | marginal m1+m2 | c2 | beat k=1 (>1.093) | +10 % (>1.233) | projected on ship m1 | llama's n1+n2 (long, mtp2) |
|---|---|---|---|---|---|---|---|
| templated, every | **1.1429** | 1.1981 | 0.6292 | **CLEARS** | fails | 1.1418 — **CLEARS** | 1.268 |
| templated, loop | 1.0408 | 1.0857 | 0.6531 | fails | fails | 1.1585 — clears | 1.268 |
| raw, every | 0.9503 | 0.9779 | 0.5449 | fails | fails | 1.0826 — fails | 1.268 |
| corpus, `--gdn-loop 1` | 1.0279 | 1.0658 | 0.5507 | fails | fails | 1.0867 — fails | 1.268 |
| corpus, engine default | 1.0358 | 1.0726 | 0.5616 | fails | fails | 1.0943 — clears by 0.0013, i.e. at the line | 1.268 |

**Saying it without softening.** `c2` lands in 0.54-0.65 across three populations, i.e. **above** the 0.487
break-even conditional and **below** the 0.678 the +10 % line needs. Converted at the shipping loop's first
position, a k=2 round is worth `1.1418/2.03 = +5.5 %` over base on templated prose, against the `+3.1 %`
that k=1's own `a = 0.7008` allows at its 1.65-step round (the wall measured +2.85 % there, and the `k=1`
geometry's hard ceiling is `2/1.65 = +21 %`): **about two points of throughput, not six, and not the
+13.6 % llama's `mtp2` printed.** On raw prose and on the corpus it is at or below break-even, and the
templated pass is the *only* population that clears — the same population whose first position was already
at parity. If our `m_2` were at llama's 0.518 the answer would be "~+4 % machine"; ours is at 0.496
marginal and 0.441 nested, so the answer is smaller and it is fragile: it depends on the round landing at
2.03 step, which this pass did not measure and which the corpus and raw rows say is not paid for on
ordinary text. **Recommendation implied by the numbers, not decided here: k=2 is not worth the 3-row body
at today's round cost.** The lever that would change the verdict is `c2`, and §4 says where it is weak
(positions 2-3, versus llama), while the round-cost lever is the meter's.

## 7. Controls — every one run on device, all green

| control | result |
|---|---|
| flag-off byte identity vs the pre-change binary (`/tmp/prewt` = `HEAD 0b3d57e`, built there) on `dec8`, 32 steps | **IDENTICAL — 34 files** (`sampled.txt`, `emitted.txt`, every `logits*.bin`) |
| behavioural canary inside that pair (`CANARY="--mtp-draft 1 --mtp-draft-k 2" CANARY_MARK="^DRAFTK pos="`) | pre-change binary **rejects** the flag; new binary accepts it and prints `DRAFTK pos=` lines → the two sides are demonstrably different builds, so "identical" is the patch not moving the target and not two copies of one build |
| `engine/gate.py --fixture dec8` | **ENGINE GATE PASS** (rc 0) |
| `k=1` reproduces the shipped side-pass rate, 49 corpus lines, 32 steps, `--mtp-sched loop`, engine-default gdn-loop, §07.5 stop active | **615/926 = 0.6641 exactly** (two chunks: 317/469 + 298/457; `rounds=953`), and the shipped instrument `run_accept.sh STAGES=corpus` printed **953 / 926 / 615 = 0.6641** in the same session on the same binary — two implementations of the join, one number |
| first column **invariant in k** (the control that caught the one real bug) | k=4 corpus-at-defaults gives first column **615/926** as well, asserted per chunk with `--expect-m1` |
| join re-derived independently | every `tN`/`hitN` the engine printed equals the value recomputed from the run's own `STEP kind=decode` lines, or the reader aborts |
| census identities, every run | `DRAFTK lines == anchors == MTPDRAFT rounds`, `hit1` tally `== MTPDRAFT adjudicated/accepted`, `MTPDRAFTK identity=ok` (`rows == anchors + extra_rows`), `DRAFTK-SKIP` count `== row_refused` |

**The bug the invariance control caught, because it is the trap in this file.** The first version returned
the draft cache to `lanes_after_first + 1` instead of `lanes_after_first` after the k rows, so each anchor
left one lane too many behind, at a rope position ahead of the next anchor's first lane. The k=1 control
**passed anyway and could not have caught it** — at k=1 the extra-row path never runs. It showed up as the
first column moving 0.6641 → 0.7100 at k=4. Rule: a k=1 control witnesses the flag-off path, never that
`k>1` leaves row 1 alone; the witness for *that* is the first column being invariant in k, so make it an
assertion (`--expect-m1`) and not an observation. Fixed in `engine/decode_run.cpp` at the truncation, with
the reason written there.

## 8. Commands to reproduce the rates

Device runs, one process at a time, each under `tools/gpu_lock.sh` via the wrapper, correctness/rates only,
**no timing taken** by any of them (the wrapper prints its own harness wall in seconds and nothing else).
`util` at the acquires below read 90-92 % — this box serves its own tokens on the same iGPU, and AGENTS.md
says to record and discount it.

```sh
# 0. build the tree as committed
bash engine/build.sh

# 1. shipped instrument: the side-pass rate the ladder's first column has to match
STAGES=corpus SCHED=loop OUT=/tmp/acc1 prototype/mtp_accept/run_accept.sh     # -> 615/926 = 0.6641

# 2. k=1 control through the ladder's own machinery (two chunks; 317/469 + 298/457 = 615/926)
K=1 STEPS=32 GDN= SCHED=loop prototype/mtp_accept/draftk_ladder.sh \
    --corpus 25 --start 0 --stop-rule engine --tag ctl-k1-a
K=1 STEPS=32 GDN= SCHED=loop prototype/mtp_accept/draftk_ladder.sh \
    --corpus 24 --start 25 --stop-rule engine --tag ctl-k1-b

# 3. the ladder at k=4, corpus, with the first column ASSERTED equal to the k=1 control
K=4 STEPS=32 GDN= SCHED=loop prototype/mtp_accept/draftk_ladder.sh \
    --corpus 25 --start 0 --stop-rule engine --expect-m1 317/469 --tag corpus-k4-default-a
K=4 STEPS=32 GDN= SCHED=loop prototype/mtp_accept/draftk_ladder.sh \
    --corpus 24 --start 25 --stop-rule engine --expect-m1 298/457 --tag corpus-k4-default-b
# ... same two lines with GDN=1 (no --stop-rule) for the --gdn-loop 1 row of §3c

# 4. the two 4k fixtures (every = headline, loop = sparse-cache contrast)
K=4 STEPS=256 SCHED=every GDN=1 prototype/mtp_accept/draftk_ladder.sh \
    --fixture tmp/llama_input/ids_long_chat.csv --tag templated-k4-every
K=4 STEPS=256 SCHED=loop  GDN=1 prototype/mtp_accept/draftk_ladder.sh \
    --fixture tmp/llama_input/ids_long_chat.csv --tag templated-k4
K=4 STEPS=256 SCHED=loop  GDN=1 prototype/mtp_accept/draftk_ladder.sh \
    --fixture tmp/llama_input/ids_long.csv     --tag raw-k4-loop
K=4 STEPS=256 SCHED=every GDN=1 prototype/mtp_accept/draftk_ladder.sh \
    --fixture tmp/llama_input/ids_long.csv     --tag raw-k4-every
# and k=1 on the templated fixture, first column asserted: --expect-m1 179/255

# 5. the three controls
CANARY="--mtp-draft 1 --mtp-draft-k 2" CANARY_MARK="^DRAFTK pos=" \
    prototype/mtp_accept/flagoff_bytecheck.sh          # needs /tmp/prewt built at HEAD
.venv/bin/python3 engine/gate.py --fixture dec8        # ENGINE GATE PASS
```

`GDN=` (empty) means "do not pass `--gdn-loop`", i.e. the engine default, which is how `0.6641` was taken.
Logs and one JSON per population go to `/tmp/draftk/`.

## 9. Limits of this measurement, stated rather than implied

* Side-pass ladder on **plain-step `h`**, no dense lanes, no prompt-lane seeding — not the accept loop. The
  `c2` projection onto the shipping geometry (§6) is a projection across D8's `h` gap; `c2` is the
  transferable quantity chosen deliberately, but it has not been checked against a k=2 loop's own verdicts,
  because there is no k=2 loop to check it against.
* A ladder from a **side pass can overstate a loop** in one direction and understates it in another. It
  overstates by counting non-nested hits (fixed by reading the nested column). It understates because the
  loop's real anchors advance by the accepted length, so a real k=2 loop visits *fewer* anchors than the
  schedule that generated these rows, and the anchors it skips are the ones right after a two-for —
  plausibly easier positions, so the true `n_2` of a built k=2 loop is at least this number and possibly a
  little above it. Nothing here is a reason to round 0.4409 up.
* Cost figures (1.65, 2.03 step; the bars) are **inherited** from `ACCEPT-APPLES.md` and the brief, not
  measured. This pass deliberately measured none — a worker's device number is contaminated by the session
  on the same GPU, and a contaminated number is worse than no number.
* Denominators: 255 / 254 positions (templated), 255 (raw), 926-927 (corpus, 49 lines). Llama's ladder
  rows rest on 128 emitted tokens per run, so its 0.518 is no better resolved than our ±3 % band; the
  position-2 ordering conclusion (0.441 vs 0.512) is inside about 1.5 sigma and should be read as "not at
  parity", not as "0.07 worse".
* 4 is the maximum k because the line carries exactly four triples; `--mtp-draft-k` parses or throws, and a
  `DRAFTK` line that is missing a field aborts the reader rather than being parsed leniently.
