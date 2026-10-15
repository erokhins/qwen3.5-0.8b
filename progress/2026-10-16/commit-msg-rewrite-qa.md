# Commit-message rewrite — QA record (2026-10-16)

Two independent QA passes over the 405 rewritten commit messages produced by
`tools/commit_msg_rewrite.py`. Pass A authored batch C and ran the primary QA; pass B was
run blind by a second worker as a cross-check, because pass A reported 0 suspects in 196
authored headers and a clean zero needs an adversary. Both conclude the set is safe to
apply; the disagreements are recorded below the seam. Parent-adjudicated follow-ups (9
truncated mid-parenthesis headers rebuilt as authored messages whose body carries the full
original text, 6 header swaps adopted from pass B, 1 area-prefix restored) landed before
the rewrite and are counted in the final class totals: KEEP 29 / SPLIT 171 / HARD 115 /
THIN 89 / NEEDS-BODY 1, verify word_failures=0 lint_failures=0.

---

# Commit-message rewrite — QA report

Repo `qwen3.5-0.8b`, HEAD `3f67ce8`. CPU only: no history rewrite, no `git commit`, no device,
no build, no timing. Working tree untouched (`git status --porcelain` shows only the pre-existing
untracked `tools/commit_msg_rewrite.py`).

## 0. State verified before starting

```
$ python3 tools/commit_msg_rewrite.py --verify /tmp/cr/out/candidates
VERIFY files=405 word_failures=0 lint_failures=0
  authoring queue: 120 authored, 76 still carrying a sentinel
```
Classes as stated: KEEP 29 / SPLIT 180 / HARD 106 / THIN 89 / NEEDS-BODY 1.
Batch A (45) and B (75) headers present. All three facts confirmed.

## 1. PART 1 — batch C authored (76 headers)

Remaining sentinel files, oldest first: **37 HARD + 39 THIN = 76**. For each, only line 1 of
`/tmp/cr/out/candidates/<sha>.msg` was replaced (was `TODO-HEADER`); **every byte below line 1 is
the emit's own text**, proved independently:

```
E1 files changed vs the pre-authoring backup of the emit: 76 (expected 76)
E1 files whose BODY moved: 0 []
```

Headers in `/tmp/cr/out/batchC_headers.txt` as `sha | header | chars` (10-char sha, oldest
first, same convention as batches A/B). Header lengths 53–72.

Rules applied: ≤72 chars, repo area prefix kept, states what the commit did, no trailing
punctuation, no number absent from the original message, body byte-identical. Every row passed a
pre-write gate (`/tmp/cr/work/validate_c.py`) that asserts: length ≤72 and >39 (nothing thin
survives), no padding/trailing punctuation, **each digit-run in the header occurs in the original
commit message**, and the original subject's leading area token is carried over.

Three deliberate, recorded deviations (no silent scope change):

| rows | deviation | why |
|---|---|---|
| d903bd6d28, 58580fd61d, 07976d90cc, ac248fd47f, 7883c09963, bab9766ff0, b9e065ae8a | `performance ledger:` → `ledger:` | `ledger:` is the prefix in the brief's list, and the earlier authored rows (de5252de19, ccafece307, 9ccad93c48, 3af685fa68) already use `ledger:` for the same file. Mechanical SPLIT rows keep `performance ledger:` verbatim, so both spellings exist in the set — aligning them all one way is a one-line call for the reviewer, not mine to make silently. |
| 32dce1e47c | `AGENTS:` added (subject said "two traps from §18 for the index") | diff-informed: `git show --stat` = `AGENTS.md` only |
| bbec27f7fb | `bench:` added (subject said "handoff note for the next session") | diff-informed: `bench/HANDOFF-2026-10-14.md`, `bench/performance-status.md` |

Nothing was marked diff-informed for reading the diff: every subject in this batch was
self-describing; `git show --stat` was used only to confirm the area of the two rows above and to
run the Q2 support column.

After authoring:

```
$ python3 tools/commit_msg_rewrite.py --verify /tmp/cr/out/candidates
VERIFY files=405 word_failures=0 lint_failures=0
  authoring queue: 196 authored, 0 still carrying a sentinel
```

## Q1 — structural gate (independent, all 405)

Tool: `--verify` → `word_failures=0 lint_failures=0` (above). Separately asserted in
`qa/q1_structural.txt` over all 405 files, with my own parser, not the tool's lint:

* no `TODO` token anywhere (this also proves the single NEEDS-BODY row, `0a2e5f6ae4`, carries a
  real authored body, not `TODO-BODY`);
* header non-empty and ≤72 chars;
* line 2 blank;
* body non-empty;
* no trailing whitespace on any line.

**Result: files checked=405, failures=0.** Extra: no final header is ≤39 chars, so no header in
the set is thin by the tool's own definition.

## Q2 — header-truth audit (196 authored headers)

Method in `qa/suspect_headers.txt`: significant tokens = lower-cased words, area prefix stripped,
punctuation trimmed, stopwords and 1–2-char tokens dropped; shared = |header ∩ original subject|.

**Shared-token histogram** (n shared → headers): 2→2, 3→6, 4→31, 5→43, 6→53, 7→34, 8→23, 9→3,
10→1. **Suspects (≤1 shared): 0 of 196** — nothing to fix, and the empty set is the finding.
Because an empty list adjudicates nothing, the 8-row low tail (≤3 shared) is written into the same
file with full fields, judged here:

| sha | verdict |
|---|---|
| 7b255ffc97 HARD `State the one-small-model scope and record this session's traps` | **sound.** Shared only scope/session/state, but "write down every trap found this session" → "record this session's traps" is the same claim; file list (AGENTS.md, README, devices/, specification/) is broader than any single path, so "no file match" is expected, not a contradiction. |
| 0d17ab8972 THIN `AGENTS: one subagent at a time, stopped for measurement` | **sound.** Drops only the second clause ("main agent steers rather than implements"); `AGENTS.md` supports it. |
| 68ba0ef677 HARD `tools/band_history.py: re-derive bands from commits, not a summary` | **sound.** "bands" is the noun the original expresses as "any tolerance"; file list names the tool. |
| d0115a8912 HARD `engine/results.md: where the engine's extra ~4 ms/token is` | **sound but clipped** — the header is a subordinate clause ending on "is"; the body carries the numbers (1.27 ms / ~2.5 ms). Cosmetic only. |
| f67b709977 THIN `tools/reference_dump.py: a pin policy — 64 steps for 60 MB, self-proof` | **sound**; both numbers come from the original ("64-step fixture costs 60 MB"). |
| 0a70c1b5a1 HARD `engine/{pin_bits_probe,curve_probe}.py: make the deep claims runnable` | **sound**; file list names both probes and results/README. |
| 914fa8e920 HARD `engine/route_allowance.py: a fixture's margin from its own rows` | **sound**; the dropped half (the refused re-price) is in the body. |
| 40a87ab680 THIN `spike: W4 scope restated — its bands are width-scoped, not a shape ban` | **sound, and the most interpretive.** "width-scoped, not a shape ban" is a reading of "their bands bound library per-call cost against the decode dispatch floor … do not exclude k/v_proj from wave bodies". Supported by the sentence, and the file (`bench/prefill-route-spike-2026-10-13.md`) is the spike the subject names. If you want zero interpretation, this is the one row to re-word. |

Companion truth checks (`qa/header_truth.txt`, `qa/area_check.txt`):

* **invented numbers: 0 of 196** — every digit-run in an authored header occurs in that commit's
  original message;
* **prefix dropped: 1** — `11f8e80b55` `bench:` → `bench plan:` (batch B, not batch C). Verdict:
  acceptable, the file is a plan file (`bench/plan-…`), but it is the only place the set changes an
  existing prefix besides the 7 `performance ledger:` normalisations;
* **area claim vs `git show --stat` over batch C: 10 PATH OK, 38 names a touched file/op,
  28 content-area (`ledger`, `plan`, `P9 Phase 0`, `attn Phase 2 kernel`), 0 CONTRADICTED.**

## Q3 — duplication (`qa/q3_duplication.txt`)

**19 of 405** final messages have a header that is a verbatim prefix of their own first body
paragraph — 12 HARD + 7 THIN, i.e. entirely an authored-set effect (a SPLIT body opens with the
*rest* of the sentence, so it cannot repeat the header). Of the 19, **10 are the clause-cut shape**
(9 where the remainder opens on a continuation phrase — "instead of", "that came back", "and",
"which" — and 1 where it opens on an em dash), and the other **9 are a complete header restated
verbatim** as the body's opening words. Neither group is a correctness defect; the restated-title
group is inherent to HARD/THIN, whose body carries the whole original subject by contract.

Samples:

* `43c49c3576` — header `attn_decode kernel: three corrections the gate forced`; paragraph 1 is that
  sentence followed by `- rope table used …`. **Judged fine**: the body immediately continues with
  the list, so the repetition is a title-then-detail shape, not a doubled sentence.
* `282dcdcd6c` — header `engine: close experiments E1 and E2 against the device numbers`; body:
  "…the device numbers **that came back** — glue's <256>…". **Reads as one sentence cut in half**;
  the cleanest of the 10 clause-cut rows to re-word.
* `99db2a10f7` — header `tools/stop_probe.py: how far does the reference run before it stops`; body
  continues "…before it stops **on its own** -- the pre-flight…". **Same diagnosis** — a clipped
  question, mildly awkward but accurate.
* `d1a7613a68` — header `results, plan: P4/P4b — the batched kernels are proven to run`; body adds
  ", batching cost zero numerics, and one p300 element is a band question". **Fine.**
* `06082fa4db` — header `route A plumbing: --gdn-scan collapse reaches the resident kernel`; body is
  the full 460-char original subject starting with the same words. **Fine** (HARD carries the whole
  subject by contract).

Related shape counts, which a reviewer eyeballing SPLIT rows will hit: **102 of 405 headers end on a
glued `,` or `;`** and **78 bodies open with a standalone `— `/`-- `**. That is the tool's documented
D4 decision (a glued cut is accepted rather than costing 133 human rewrites), not a regression from
this pass. Also **9 SPLIT headers carry an unclosed `(`** (the tool docstring predicts exactly 9, and
9 are present), and **9 headers end on a function word** ("…gated against it", "…records it",
"…glue does not") — cosmetically clipped, all faithful.

## Q4 — distribution tails (`qa/q4_tails.txt`)

10 longest single paragraphs (chars): 1685 `eac87e8441` SPLIT · 1385 `a0088052cb` SPLIT ·
1318 `05c588212e` HARD · 1194 `9c64df2a2c` SPLIT · 1172 `643fc83ff3` SPLIT · 1034 `2d3c95ac38`
KEEP · 1028 `3862bba312` SPLIT · 952 `26fd753169` THIN · 925 `b2bb84e88a` THIN · 922 `5707883d95`
HARD. These survive because a paragraph is only wall-split at a sentence boundary and these have
none (no `.!?`, no depth-0 `;`); they are folded to ≤76 columns, so lint is right to pass them, but
they are the remaining walls of the set.

10 most-paragraphed: 20 `8eace8497b` · 19 `d2ba883867` · 18 `0f8b81e2cd` · 17 `5d03f24631` ·
17 `20584a76a4` · 16 `fd0c7a886e` · 15 `6db6ee4c9d` · 14 `8b81970cc6` · 14 `7a8dc238fd` ·
14 `3bd46d9882`. Message size min 93 / median 1010 / p90 2473 / max 5174 chars.

## Q5 — sample audit (`qa/q5_sample_audit.txt`)

12 messages: oldest (`d7be6c6c84`, 2026-10-10), newest (`3f67ce860d`, 2026-10-14), three dated
eras 2026-10-11 / 10-12 / 10-13 (first, middle, last of each), plus quartile points. Full old
subject (100 chars) / new header / new body first paragraph per row in the file. Compact:

| date | class | old subject (58) | new header (58) |
|---|---|---|---|
| 10-10 | KEEP | Initialize qwen3.5-0.8b engine: Qwen3.5-0.8B text-only de | Initialize qwen3.5-0.8b engine: Qwen3.5-0.8B text-only de |
| 10-11 | SPLIT | First real HIP kernel: bf16 lm_head GEMV at 242 GB/s, plus | First real HIP kernel: bf16 lm_head GEMV at 242 GB/s, |
| 10-11 | HARD | WIP engine/state + kernel launchers: RequestState sized fr | WIP engine/state: RequestState and launchers, CPU-verified |
| 10-11 | HARD | engine/results.md + route_price.py: the step-1 dip is insi | engine/results.md + route_price.py: the step-1 dip has a l |
| 10-12 | SPLIT | plan Q5a: the prefill implementation plan — T=1 audit (all | plan Q5a: the prefill implementation plan |
| 10-12 | THIN | tools/reference_dump.py: a pin policy, so a 64-step fixtur | tools/reference_dump.py: a pin policy — 64 steps for 60 MB |
| 10-12 | HARD | decision log D-10: P8's engine wiring hangs the device on | decision log D-10: P8's engine wiring is parked in stash@{ |
| 10-12 | THIN | engine/gate.py: section (g) — engine end-state fingerprint | engine/gate.py: section (g) — end-state fingerprints again |
| 10-13 | SPLIT | attn_decode_layer: a wave of query rows through one site, | attn_decode_layer: a wave of query rows through one site, |
| 10-13 | THIN | plan: operator queue order — debt (D-9 sized rule, then th | plan: queue order — D-9 debt, the MLP residue, then a 5k p |
| 10-13 | SPLIT | ledger: D-11 closed and route (c) marked admitted — build- | ledger: D-11 closed and route (c) marked admitted |
| 10-14 | KEEP | docs: the audit's stale-reference table executed, one arti | docs: the audit's stale-reference table executed, one arti |

No sample contradicts its source. The two shapes a reviewer should decide on before the rewrite are
visible here: a SPLIT header that ends on a glued comma and hands the clause to the body, and a HARD
header that restates the opening words of the carried subject.

## Recommendation

**Safe to apply, with one mechanical precondition and no numerics blockers.** Nothing in the 405
files loses, invents or moves a word (the tool's proof), and nothing structural fails.

Must be handled before `--apply`:

1. **`--apply` will refuse to run.** It aborts when `git status --porcelain` is non-empty, and this
   repo currently shows `?? tools/commit_msg_rewrite.py`. Commit that tool (its own topic commit) or
   stash it. That is a working-tree decision and I was told not to touch the tree, so it is left to
   you.
2. **Run the history gate after the rewrite**:
   `python3 tools/commit_msg_rewrite.py --verify-history refs/heads/backup-pre-msgrewrite HEAD`
   (the ref is created by `--apply` on the first run). Expect `VERIFY-HISTORY PASS (405 commits …)`.

Judged acceptable as they stand, and explicitly not fixes-before-apply: the 19 header/body
restatements (§Q3), the 102 glued-comma SPLIT endings, the 9 unclosed-`(` SPLIT headers, the 9
function-word endings, the 10 unbreakable long paragraphs, and the `ledger:` vs
`performance ledger:` split you may want to normalise one way (11 authored rows plus 7 SPLIT rows
would move). If you want a zero-interpretation header, `40a87ab680` is the single row worth
re-wording.

One thing to know rather than fix: **2 candidate files carry a chat-template special token written
literally** — the end-of-turn id (248046) in two of them, and the turn-role opener in one — because
their own original commit messages do, and the rewrite is lossless so it carries them byte for byte.
No batch C row does, and this report quotes none literally per the repo's standing rule; anything
that later quotes those two messages must name the ids, not the token.

Artifacts: `qa/q1_structural.txt`, `qa/suspect_headers.txt`, `qa/header_truth.txt`,
`qa/area_check.txt`, `qa/q3_duplication.txt`, `qa/q4_tails.txt`, `qa/q5_sample_audit.txt`,
`batchC_headers.txt`, scripts under `/tmp/cr/work/`.

---

# Pass B — independent cross-check

# QA_REPORT.workerB.md — independent QA of the rewritten commit-message set

Second opinion, run by the second worker, on the same 405-file set in `/tmp/cr/out/candidates/`.
Nothing here reuses the primary pass's scripts or numbers: every figure below was recomputed with
scripts in `/tmp/cr/out/qa2/` (`qa.py`, `independent_lossless.py`, `truth_extras.py`,
`q2_adversarial.py`, `grounding.py`, `extras.py`, `special_tokens.py`, `bytes_check.py`,
`dups_prefix.py`, `d4_shape_check.py`, `pairs.py`, `verdicts.py`, `swap_check.py`).

Write scope honoured: this file and `/tmp/cr/out/qa2/` only. `candidates/**`, `qa/**` and
`batchC_headers.txt` were read, never written. No repo file was touched, nothing staged, no history
rewrite, no `git commit`, no device, no build, no timing.

## 0. Answer first: is the set safe to apply?

**Yes — safe to apply, and I would apply it.** Independently of the primary pass:

- `python3 tools/commit_msg_rewrite.py --verify /tmp/cr/out/candidates` →
  `files=405 word_failures=0 lint_failures=0`, authoring queue `196 authored, 0 still carrying a
  sentinel`, exit 0 (re-run at the end of my pass, after Batch C landed).
- My own structural gate over all 405 files: **0 files with any failure**.
- My own losslessness check, with my own tokenizer rather than the tool's `word_diff`: **0 failures**
  on 209 KEEP/SPLIT word-multiset equalities, 195 HARD/THIN bodies (every original word present,
  and the body byte-identical to the `--emit` body in 195/195 cases), and the 1 NEEDS-BODY file.
- No header invents a number, a verdict, or a fact: 0 headers share ≤1 significant token with their
  own original subject, 0 with their own whole original message, 0 headers carry a digit absent from
  their own original message.
- Nothing dangerous slipped in: no `TODO` sentinel text, no CR characters, every file ends with a
  newline, no trailing whitespace, no chat-template control-token literal was introduced by the
  rewrite (the two that exist were already in their original messages).

Three things are **not defects but must be seen before the log is judged** — they are in §9. And two
operational prerequisites block `--apply` right now — §8.

## 1. Where I disagree with the primary pass (most valuable first)

**D1 — Batch C: the landed headers are better than mine, and I am not going to pretend otherwise.**
63 of the 76 differ from what I had authored. Judging header quality, not authorship:
`THEIRS-BETTER=32, EQUIVALENT=23, IDENTICAL=13, OURS-BETTER=8` (full 76-row table with a one-clause
reason per row: `qa2/batchC_verdicts.txt`, pair list: `qa2/pairs.txt`). The landed pass usually won
for the same reason: it kept a *second* fact where I trimmed to one (e.g. `ledger: P8's payoff — TTFT
−24 % at 3072 prompt tokens, decode unchanged` where I kept only the absolute ms pair), and it named
mechanisms (`--scope form`, `--mlp-map`, `v_wmma`, `D-9`, `gemm_common.h`) where I paraphrased them.
Eight rows are worth a look before applying, listed in §7.

**D2 — the primary pass's Q2 evidence is thin in a way mine is not.** Its `header_truth.txt` compares
header tokens against the original **subject** only. That is the right first cut, but it cannot see a
header that quietly contradicts the *body*, and it cannot see a header that coins a term the message
never uses. I ran both adversarial variants (header vs the entire original message) and a
morphology-tolerant grounding audit (`qa2/q2_adversarial.txt`, `qa2/q2_grounding.txt`). Result: no
lie found, but the grounding audit does surface **45 of 195** authored headers carrying a word that
appears nowhere in their own original message. I read all 45 (they are printed, with the original
subject beside each). Every one is a compression of something the message says — `RMSNorm`→`norm`,
`can no longer wedge`→`cannot wedge`, `checks out against a double CPU reference`→`matches a double
CPU reference`, `a pin that meant the wrong thing would fail`→`verify what each pin means`. Three
deserve a human eye, none is false:

| sha | header | the coined part | verdict |
|---|---|---|---|
| `40a87ab680` | `spike: W4 scope restated — its bands are width-scoped, not a shape ban` | `width-scoped`, `shape ban` appear in neither the subject nor the body nor the diff | fair paraphrase of "its bands bound library per-call cost against the decode dispatch floor"; the term is used by the sibling commit the same day (`d475b8d93b`), so it is repo vocabulary. Keep. |
| `11f8e80b55` | `bench plan: P1b — dec64t, a deep fixture with a >64-token prompt` | original area label was `bench:`, header says `bench plan:` | the one header in the whole set that alters an existing area prefix. Harmless (still `bench`-rooted), but it is a convention drift, so either accept it explicitly or restore `bench:`. The primary's own audit already flagged it. |
| `eec99a8bbc` | `mfma probe: gfx1151 emits v_wmma and matches a double CPU reference` | `matches` | grounded in the body verbatim ("checks out against a double CPU reference"). Keep. |

**D3 — the brief's Q3 parenthetical is wrong, and the correction matters for how the set is defended.**
The brief guessed the "header repeats as the first body sentence" shape was inherent to SPLIT. It is
the opposite: **0 of 180 SPLIT headers hit it** — a SPLIT header is a glued clause and cannot be a
sentence. All **19 hits are authored HARD (12) / THIN (7)** headers, and they exist for a reason the
contract forces: HARD/THIN bodies must carry the original subject verbatim as their first paragraph,
so any authored header that is a prefix of the original subject *must* duplicate the body opening.
So: not inherent to the tool, but inherent to (losslessness ∧ prefix-style header). §4 quotes the 5
samples and splits the 19 mechanically, 8 sep-lead against 11 word-lead.

**D4 — I add two shape facts the primary pass did not report, one of which a reviewer will notice.**
102 of 405 headers (all SPLIT: 95 end on `,`, 7 on `;`) end on glued punctuation — a quarter of the
log's subjects end mid-sentence. That is the tool's documented accepted decision, not a defect
(`header_shape_reject` deliberately allows it), but the eyeball reviewer must be told, because it is
the first thing that looks wrong in `git log --oneline`. And: **0 duplicate headers** — 405 distinct
headers for 405 messages, so `git log --oneline` stays unambiguous. (`qa2/shape_extras.txt`,
`qa2/dups_prefix.txt`.)

**D5 — paragraph-count tails differ between the two passes, definitionally.** Same top-10 longest
single paragraphs (identical file list and lengths, e.g. `eac87e8441` at 1685 chars); different
top-10 by paragraph count (mine max at 20 paragraphs, theirs at 13) because I count blank-line
paragraphs after re-wrap including list items. Not a discrepancy about the data, only about the
unit — worth stating so the two reports are not read as contradicting.

## 2. Q1 — structural gate (independent)

`python3 /tmp/cr/out/qa2/qa.py` → `qa2/q1_structural.txt`

```
files checked = 405   files with >=1 failure = 0
class counts: KEEP=29  SPLIT=180  NEEDS-BODY=1  THIN=89  HARD=106
header length: max=72 min=40
HARD/THIN bodies byte-identical to the --emit body except line 1: 195 of 195 clean
NEEDS-BODY 0a2e5f6ae469: header is original subject verbatim = True; body still the sentinel = False; body chars = 428
```

Assertions run per file, my own code: literal `TODO` token absent; header non-empty and ≤72 chars;
line 2 blank; body non-empty after the header; no trailing whitespace on any line; final newline
present. Class counts match the emit manifest exactly. Header length: min 40, max 72, 15 headers at
exactly 72 (the tightest fits are at the limit but none over it).

Losslessness, re-derived independently (`qa2/q1_independent_lossless.txt`, own tokenizer):

```
KEEP+SPLIT compared = 209; HARD+THIN = 195; NEEDS-BODY = 1
KEEP/SPLIT not equal: 0
HARD/THIN body missing a word: 0
HARD/THIN body not the emitted body: 0
NEEDS-BODY header not verbatim: 0
NEEDS-BODY missing a word: 0
```

Housekeeping (`qa2/bytes_check.txt`, `qa2/special_tokens.py`): 405 files, 0 with a CR byte, 0 missing
a final newline, 0 that are not valid UTF-8. Control-token scan: 2 candidate files contain a raw
turn-boundary control-token literal — `e028f5cb66` and `e6bf1563dc` — and both carry it **in their
original message too** (`in-candidate == in-original`, `added_by_rewrite = []`), in the body, never in
a header. Those two commits are precisely the ones whose bodies discuss the repo's own rule about not
printing such literals, so the text legitimately names them; the rewrite neither added nor removed
one. Referenced here by file name, not by printing the token.

## 3. Q2 — header-truth audit (independent)

Method, exactly as specified: strip the leading area-prefix token(s) (ending in `:`), split on
whitespace, trim edge punctuation, lowercase, drop tokens with no alphanumeric/`§`, drop tokens of
length ≤2, drop a fixed stopword list; a header is a **suspect** if it shares ≤1 significant token
with its original subject.

Result: **0 suspects of 196 authored headers.** Shared-token histogram:

```
2->3   3->8   4->30   5->46   6->46   7->35   8->23   9->3   10->2
```

Sensitivity, because "significant token" is a judgement call (`qa2/q2_sensitivity.txt`): no stopword
list at all → 0 suspects; no prefix stripping → 0 suspects. The verdict does not depend on the
recipe. Adversarial variant, header vs the whole original message (`qa2/q2_adversarial.txt`): still
0 with ≤1 shared token, minimum shared is 3.

Since the specified threshold yields nothing, I printed the tail instead and adjudicated it against
`git show --stat` — the 11 headers with the fewest shared tokens, each with the files the commit
actually touched (`qa2/q2_adversarial.txt`). The three 2-shared cases:

- `0d17ab8972` `AGENTS: one subagent at a time, stopped for measurement` ← "…and the main agent
  steers rather than implements". Files: `AGENTS.md`. True but incomplete: it drops the second rule.
- `7b255ffc97` `State the one-small-model scope and record this session's traps`. Files: `AGENTS.md`,
  `README.md`, `devices/strix-halo-unified-memory.md`, `prototype/README.md`, `specification/07`,
  `specification/10` — the file list supports "scope + traps" exactly.
- `f67b709977` `tools/reference_dump.py: a pin policy — 64 steps for 60 MB, self-proof`. `self-proof`
  is verbatim from the body ("the dumper grows three switches and a self-proof"); file list is the
  one file named.

Numeric grounding: **0 authored headers contain a number absent from their own original message** —
the specific failure mode the brief worried about (a header asserting a measurement the commit never
made) does not occur anywhere in the set.

Area-prefix audit (`qa2/truth_extras.txt`): 57 path-like prefixes, 133 prose prefixes, 6 headers with
no prefix (all 6 faithful: their originals had none). 15 path-like prefixes do not string-match a
touched path; I checked all 15 against `git show --stat` and every one is a naming variant of a file
the commit touches (`Add bw_probe.py` → `tools/bw_probe.py`; `prototype/glue kernel` →
`prototype/glue/glue.hip.cpp`; `engine/{pin_bits_probe,curve_probe}.py` → the two files with braces
expanded; `gdn_gated_norm results.md §7` → `prototype/gdn_gated_norm/results.md`). No header claims
an area its diff does not touch.

## 4. Q3 — header/body duplication (independent)

19 of 405 messages have a header that is a verbatim prefix of their own first body paragraph.
By class: `KEEP=0 SPLIT=0 NEEDS-BODY=0 THIN=7 HARD=12`. Zero duplicates across messages
(405 distinct headers). Full list: `qa2/dups_prefix.txt`; detail: `qa2/q3_header_prefix.txt`.

Five samples, and the judgement. The split that matters is *sep-lead* (the header is a title and the
body carries on from a separator — reads fine) versus *word-lead* (the header was cut mid-phrase, so
it says nothing on its own and the body immediately re-says it):

| sha | class | header | what follows it in the body |
|---|---|---|---|
| `43c49c3576` | HARD | `attn_decode kernel: three corrections the gate forced` | `- rope table used frequency d instead of d%32 …` — sep-lead: reads as a title plus its list. Fine. |
| `282dcdcd6c` | HARD | `engine: close experiments E1 and E2 against the device numbers` | `that came back — glue's <256> is a measured preference …` — word-lead: the header is cut mid-relative-clause, so it reads twice. |
| `99db2a10f7` | HARD | `tools/stop_probe.py: how far does the reference run before it stops` | `on its own — the pre-flight a no-crossing deep fixture needs` — word-lead, same shape; the header alone is an incomplete question. |
| `d1a7613a68` | THIN | `results, plan: P4/P4b — the batched kernels are proven to run` | `, batching cost zero numerics, and one p300 element is a band question` — sep-lead on a comma: the body is the header's own sentence continued. Mildly repetitive, not wrong. |
| `06082fa4db` | HARD | `route A plumbing: --gdn-scan collapse reaches the resident kernel` | `through the same one place the form is chosen (…)` — word-lead, and the paragraph runs 460 chars, so the repeat is the least of it. |

**Judgement (my own classifier, `qa2/dups_prefix.txt`): sep-lead=8, word-lead=11.** The split is
mechanical, not a matter of taste: 8 of the 19 continuations open on a separator (`,`, `-`, `—`, `(`),
so the header reads as a title and the body carries on from it; the other 11 open on a word, meaning
the header stopped mid-phrase and the body finishes that phrase. The primary pass reports 10 clause-cut
/ 9 restated for the same 19 files — the same division under different wording, so both counts agree
on what matters. The sep-lead group does not read as a repeated sentence and needs no change. The
word-lead group is where "one sentence told twice" actually bites — but even there it is forced by the
losslessness contract: a HARD/THIN body carries the original subject as its first paragraph, so any
authored header that is a ≤72-char prefix of that subject *must* duplicate the body opening. Fixing it
means either rewording headers away from the subject's own words, or breaking the word-preservation
proof. Ship as-is and note the 11.

One extra check in the same family (`qa2/d4_shape_check.txt`): the tool's own D4 helper
(`header_shape_reject`) refuses a header ending on a function word, but `lint()` — what `--verify`
actually runs — never calls it on an authored header. **9 of 405** headers end on a function word
(`…the rule behind it`, `…records it`, `…C=16 is not`, `…where the engine's extra ~4 ms/token is`).
All 9 are idiomatic English and readable; I list it only because it is a place where the emit-time
shape policy and the verify-time lint disagree, so a future batch could land a header that the tool
itself would have rejected. Not a defect in this set.

## 5. Q4 — longest-worst (independent)

`(a)` 10 longest single body paragraphs (chars) — identical file list to the primary pass:
`eac87e8441` 1685, `a0088052cb` 1385, `05c588212e` 1318, `9c64df2a2c` 1194, `643fc83ff3` 1172,
`2d3c95ac38` 1034, `3862bba312` 1028, `26fd753169` 952, `b2bb84e88a` 925, `5707883d95` 922.
Longest-paragraph buckets: 0–159→54, 160–239→68, 240–399→163, 400–799→105, 800+→15.

`(b)` 10 most paragraphs (blank-line paragraphs after re-wrap, list items counted): `8eace8497b` 20,
`d2ba883867` 19, `0f8b81e2cd` 18, `20584a76a4` 17, `5d03f24631` 17, `fd0c7a886e` 16, `6db6ee4c9d` 15,
`7a8dc238fd` 14, `3bd46d9882` 14, `8b81970cc6` 14.
Paragraph-count distribution: `1->117 2->44 3->22 4->25 5->27 6->23 7->29 8->24 9->30 10->17 11->15
12->16 13->6 14->3 15->1 16->1 17->2 18->1 19->1 20->1`.

Read: the long tail is **pre-existing content**, not rewrite damage — the 15 messages with an 800+
char paragraph are SPLIT/KEEP/HARD bodies whose original was one wall of text; the rewriter split
sentences and re-wrapped them but does not delete content (by contract). Nothing in the tails is
malformed (`qa2/claims_check.py`): 0 files with an odd number of code fences, and the only short
colon-terminated body lines (7 of them: `Findings:`, `classes:`, `hooks:`, `polices:`, `same way:`)
are legitimate list labels, not a header stranded in the body.

## 6. Q5 — 12-message sample audit

Full text in `qa2/q5_sample_audit.txt` (old subject / new header / new body P1, per file). Coverage:
newest 2, oldest 2, first+middle of 2026-10-11/-12/-13 (commits/day: 10th=15, 11th=133, 12th=102,
13th=84, 14th=71), the single NEEDS-BODY file, and a KEEP. Verdicts:

- KEEP rows (`3f67ce860d`, `d7be6c6c84`, `d980734015`, `88d2c06eb5`) are byte-identical to their
  originals apart from re-wrap — correct by contract.
- SPLIT rows (`5bc6d04c31`, `f4a1095693`, `ae8c447cca`) show the designed behaviour: header = first
  clause (+`,`) or first clause alone, body P1 = the rest. `5bc6d04c31` reads
  `First real HIP kernel: bf16 lm_head GEMV at 242 GB/s,` / body `plus the memory-mode probe` — the
  glued-comma shape from D4.
- HARD/THIN rows (`27dbaed9e3`, `f67b709977`, `e009989ddb`, `d1703b0d11`) have a new authored header
  and a body whose P1 is the original subject verbatim — hence §4's duplication on some of them.
  The authored headers are faithful and more scannable than the originals, which were truncated
  mid-clause (`… the dtype `, `… then the ~1.09`).
- NEEDS-BODY (`0a2e5f6ae4`): header is the original subject verbatim by contract, and the sentinel
  body was replaced by real text (428 chars) that starts "Both files named id 248044 by printing the
  raw special token itself" — note this is the one file where the *body* discusses the special-token
  rule; the header names the token **by id** (248044), consistent with the repo rule.

## 7. Batch C: landed vs my preserved 76

`/tmp/cr/work/batchC.txt` holds the headers I had authored before the ownership call;
`qa2/batchC_verdicts.txt` has all 76 rows (landed, mine, verdict, reason);
`qa2/batchC_diff.txt` is the raw diff with the file list per commit.
Tally: **IDENTICAL=13, THEIRS-BETTER=32, EQUIVALENT=23, OURS-BETTER=8.** Neither set broke a rule
(`qa2/ending_check.txt`): max header length 72 in both, 0 over the limit, and 0 headers in either set
end on `.`/`:` or on a glued `,`/`;` — none of the 76 is a SPLIT header, so the glued-ending shape
never arises here.

The 8 rows where mine is better. Strings are verbatim, taken from `candidates/<sha>.msg` line 1 and
`/tmp/cr/work/batchC.txt`, and each proposed string was re-checked with the tool's own
`header_shape_reject` plus the ≤72 and trailing-punctuation rules (`qa2/swap_check.txt`; all 8 pass,
lengths 67–70). The pair list itself is `qa2/pairs.txt` — regenerated from those two sources, because
my first pass keyed the verdict reasons off a paginated print and several quoted strings in an
earlier draft of this report were the wrong rows. This section is the corrected version.

| sha | landed | mine — and why |
|---|---|---|
| `88fd712f03` | `AGENTS.md: a device fault on a wave-only path presents as a hang` | `AGENTS.md: trap index gains a wave-only fault that presents as a hang` — landed states the trap, not the change; this commit added a row to the trap index |
| `349e88aef9` | `proj_gemv: --only gdn\|attn, so ledger §2's two rows can be priced` | `proj_gemv: --only gdn\|attn, so ledger §2's unmeasured rows get priced` — keeps *why* those two rows mattered |
| `d1703b0d11` | `engine: the gate/up weight walk follows the row count` | `engine: the gate/up weight walk follows the row count; --mlp-map wins` — landed drops the override that still exists, and its header is also one of §4's verbatim body prefixes |
| `21cf3d1e38` | `tools: route-c audit analyzer — four statistics that separate a bug` | `tools: route-c audit analyzer — four statistics, bug vs reassociation` — landed ends with the second term of the comparison missing |
| `9b75eb0eb9` | `P9 Phase 0: verdict FUND — the chunked delta-rule scan is priced` | `P9 Phase 0: verdict FUND — chunked delta-rule scan priced, no new band` — the no-new-band claim is half of why FUND was cheap |
| `e44ed51d35` | `attn Phase 2 kernel: attn_tile_wave, G=4 heads x M=4 rows per block` | `attn Phase 2 kernel: attn_tile_wave, one K row per block, G=4 x M=4` — "M=4 rows per block" blurs the kernel's defining property, one K row per block |
| `d2ba883867` | `p9 phase1 report: inside budget at C=32/C=64, C=16 is not` | `p9 phase1 report: inside Phase 0's budget at C=32/C=64, not at C=16` — names whose budget; and landed's ending on `is not` is the one swap that also fixes a shape nit (§4) |
| `924937144b` | `D-12: the chunked GDN scan is killed; route A launch-collapse is funded` | `D-12: chunked GDN scan killed; route A funded at 42% less arithmetic` — keeps the cost number that made the funding a decision |

Three representative landed wins, so the tally is not one-sided:

- `d903bd6d28` — landed `ledger: P8's payoff — TTFT −24 % at 3072 prompt tokens, decode unchanged`
  vs mine `performance ledger: P8's payoff — TTFT 5407→4117 ms at 3072 tokens`: two facts where I
  kept one, and both numbers are in the original message.
- `eec99a8bbc` — landed `mfma probe: gfx1151 emits v_wmma and matches a double CPU reference` vs
  mine `…has wavefront matrix instructions and uses them`: landed names the instruction and the
  cross-check.
- `fc5419b563` — landed `agents: execution policy — keep moving, report at milestones, fix-first` vs
  mine `…keep moving, report the honest tokens/s`: landed keeps fix-first, the substantive half of
  the policy change.

My recommendation: **apply the landed set as it stands**; treat the 8 above as an optional menu. Each
swap is a one-line edit to `candidates/<sha>.msg` line 1 that must re-pass `--verify` — header lint
only, since the body is untouched and the word proof cannot break on a header change.

## 8. Two operational prerequisites for `--apply`

1. **`--apply` will refuse right now: the working tree is dirty.** `git status --porcelain` shows
   `?? tools/commit_msg_rewrite.py`, and `cmd_apply` aborts on any non-empty porcelain output
   ("working tree is dirty; commit or stash first"). Either commit the tool first (its own topic) or
   move it out of the repo for the apply run. Worth deciding deliberately, because the tool *should*
   be committed anyway — the `--verify-history` check is part of the apply evidence.
2. **No backup ref exists yet.** `refs/heads/backup-pre-msgrewrite` is absent (checked with
   `git show-ref`); `--apply` creates it at HEAD before rewriting via `update-ref`, so this is
   automatic — but confirm the line prints `created refs/heads/backup-pre-msgrewrite at <sha>` before
   letting the rewrite proceed. `git` is 2.43.0 and `.git/refs/original` is empty (no prior
   filter-branch debris). The apply path is `filter-branch -f --msg-filter` over the whole of `HEAD`,
   so every one of the 405 messages is replaced in one pass; nothing else in the tree is rewritten.

## 9. Residual risks

- **102 headers end on glued `,`/`;`** — all of them SPLIT, 0 elsewhere in the set
  (`qa2/ending_check.txt`). Accepted design, but it is the single most visible oddity in the
  rewritten log; if the reviewer objects after seeing it, the fix is a policy change in
  `header_shape_reject` plus a re-emit of the 180 SPLIT files, not a hand edit.
- **11 messages open their body with the header's own words** (word-lead, §4) — forced by
  losslessness, and only a readability cost.
- **45 authored headers carry a word absent from their own message** (§3). All audited benign;
  the class is invisible to both passes' primary checks, so any *future* authored batch needs the
  grounding audit, not just the subject-overlap test.
- **The tool checks itself in `--verify`** (it reads its own manifest and its own word proof). My
  independent re-derivation is what makes that evidence, which is the main reason this second pass
  was worth running; if the set is ever regenerated, re-run `qa2/qa.py` + `qa2/independent_lossless.py`
  rather than trusting `--verify` alone.
- **A re-take risk, not a content risk:** applying rewrites every SHA, so every historical citation by
  SHA in `bench/`, `progress/`, `results.md` and `devices/` still names objects that exist only as the
  backup ref. That is inherent to history rewrite, not to this message set; the apply run should
  record the backup ref sha next to the ledger entry that says the rewrite happened.
- Nothing here is a timing measurement, and no number in this file came from a device run.
