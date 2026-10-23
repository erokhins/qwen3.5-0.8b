# Corpus wall A/B — the first measured MTP gain, and two things that block adopting it

Everything here is main-agent, blocking, one lock hold per pass, `corpus_wall_ab.{py,sh}`. Accept rates are
deterministic and so are quotable from any window; millisecond figures are not, and the harness now refuses
to print one from a window whose base is >25 % off a clean reference.

## What was measured

Population: the derived corpus (`run_accept.load_population()`, `speccheck/fixtures/corpus.txt` cross-checked
against `data/refs/corpus48/manifest.json`), 49 lines, used through `--limit 24`. **The corpus prompts are
~6 lanes each** (`natural length, mean 6 lanes`), so every "corpus" accept rate in this repo — 0.6556, 0.7258,
0.7581 — is a near-empty-context number, and the 4,032-lane reporting width can only be built from this
population by concatenating all 49 lines into ONE synthetic case.

| window | width | arm | base | accept | a | round/step | a* | verdict |
|---|---|---|---|---|---|---|---|---|
| clean | natural (short) | hand (default) | 103.984 tok/s (9.617 ms/step) | 108.436 (16.213 ms/round) | 0.7258 | 1.686 | 0.686 | **+4.28 %** |
| contended | natural | hand | 12.921 ms/step base | — | 0.7258 | 1.56–1.57 | — | REFUSED by guard |
| contended | natural | amortised | 17.138 ms/step base | — | 0.7339 | 1.40–1.48 | — | REFUSED by guard |
| clean | 4,032 lanes (1 synthetic joined case) | hand | 93.979 | 73.090 | **0.2903** | 1.701 | 0.701 | −19.78 % |

So: **MTP now pays on the corpus — +4.28 % — at the width the corpus actually has.** It is still under the
pre-registered +10 % adopt line, and the route flag that might close the gap is not admissible yet (below).

## The corpus cannot answer the 4k question, and that is structural

The standing rule reports decode at 4,032 lanes. The corpus-49 lines are short (49 lines total ~8k tokens,
~170 each), so joining them fills **one** case at 4,032 — that case read `a = 0.2903`, against `a* = 0.701`.
The corpus accept rate is a short-context property: 0.6556 corpus, 0.7258 here, 0.8065 on prose, 0.097 on a
technical document, 0.2903 when the same text is stretched to 4k. Every prior "corpus at 4k" phrasing in this
session was therefore measuring one synthetic concatenated case, and it should be read that way.

## The dense draft-cache lane append, and what it bought

`--mtp-dense` (default **off**; refuses without `--mtp-accept` and under `--time-stages`) stages the draft-cache
lane an ACCEPT skipped. A round appends one lane, so on an accept — which consumes two positions — the draft
KV skips a position and every lane behind it is one rope step further from its neighbour than the block was
trained to see; llama.cpp's draft context has no such hole because its catch-up `process()` feeds one row per
target row (`engines/llama.cpp/common/speculative.cpp:1625-1637`). Staging a lane costs 9 submissions instead
of a round's 16, and it changes the PROPOSED token and nothing else.

Measured, deterministic, same population and window per row:

| width | a, dense off | a, dense on | accept vs base |
|---|---|---|---|
| corpus natural (~6 lanes) | 0.7258 | **0.7581** | +8.54 % → **+10.21 %** (one rep, cleared) |
| 4,032 lanes (1 joined case) | 0.2903 | **0.3548** | −21.87 % → −15.18 % |

So the mechanism is real — +4.5 % relative accept at short width, +22 % at 4k — and it is not enough at the
reporting width: at 4,032 lanes the round still costs ~2 steps, `a` is 0.3548 against `a*` ≈ 0.64+, and the
verdict stays negative. The +10.21 % at natural width is one rep in a window the base-sanity guard would not
have certified, so it is a lead, not an adopted headline.

## What was reported and did not exist

Two subagent reports in this session cited artefacts that were never in the tree. `--mtp-draft-demand` was
reported as tip `60542b2` with witnesses and a three-arm A/B; `76902a0`/`5037b49` were cited for route numbers
and `ROUTE-AB.md`. **No such commit objects, no such files, no second checkout** (`git cat-file`, a
disk-wide `find`, `git worktree list`). What *did* exist was uncommitted engine work in the working tree — the
`--mtp-dense` lane append above, plus `--mtp-prompt-lanes` — which is consistent with a worker that implemented
and measured but never committed before its session ended. The lesson is the one AGENTS.md already states about
uncommitted halves: a tree holding a whole feature unstaged is indistinguishable from one that never had it, so
this file records what was re-verified independently (build, `ENGINE GATE PASS`, the accept rates above) rather
than what was claimed.

## Route numerics finding, and it outranks the speed question

`--mtp-route hand|amortised` was admitted on byte-identity of the **verify body's logits**. On the **loop** it
is not output-neutral: over the same 24 corpus cases × 3 reps, **3 of 42 accept-arm dump sets differ**, all on
case 18, in `emitted.txt` — the emitted token stream, not a tolerance. `a` agrees: 0.7258 (hand) vs 0.7339
(amortised), a deterministic difference of about four acceptances across the population, which is exactly what
different emissions produce.

Consequences, stated as policy not opinion:
- the route flag may **not** be treated as a free timing knob on the accept path; choosing a route there
  changes model output and needs the same admission any output-changing route needs;
- this is invisible to every existing gate: `engine/gate.py` runs no `--mtp-verify`/`--mtp-accept` and no
  `data/refs` fixture pins a round (the gap recorded in `b697843`/ROUND-FORM.md), so nothing in the tree
  could have caught it;
- the cheapest admission is the one this session already has the shape of: pin a round fixture and compare
  `emitted.txt` between routes — deterministic, cheap, no timing involved.

## Why the contended windows are discarded rather than paired-away

`flash_serve` — the engine serving this very session — holds 24 GiB and busy-polls the iGPU; base drifts
9.6 → 12.9 → 17.1 ms/step between windows and only it and our engine are on the device. Pairing arms inside
one hold is not sufficient here because the *ratio* itself moves with contention (+19 % to +24 % in contended
windows against +4.28 % clean), so `--base-ref-ms` now refuses any window >25 % off the clean reference. A
ratio that only appears when the box is busy is not a speedup.
