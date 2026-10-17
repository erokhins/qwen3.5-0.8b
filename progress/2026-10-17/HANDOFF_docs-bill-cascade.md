# HANDOFF — docs cascade for the retired launch bill (`62b73e9`: cast 115→97, total 424→406)

Worker run, base `968bbf9`. **Nothing committed, nothing staged, no device run, `engine/gate.py`
never edited.** The run was cut off by the deadline mid-way through Job 1; Jobs 2 and 3 are analysis
only and are handed over complete below.

## 1. State of the tree (changed files)

| file | state | lines vs HEAD |
|---|---|---|
| `engine/results.md` | **Job 1 complete** — every live 424/115 claim corrected or dated, with arithmetic | 4690 = HEAD, **every diff hunk line-count-balanced** |
| `engine/README.md` | **Job 1 complete** (L10, L372 `(d)` line, L472/L474) | 496 = HEAD, balanced |
| `tools/step_traffic_model.py` | **Job 1 complete**, including the byte-budget defect; runs clean | 322 vs 314 (+8; 0 line-form cites exist into this file, so growth is safe) |
| `tools/instep_family_budget.py` | **NOT EDITED** — remaining (see §3) | untouched |
| `tools/prefill_gemm_shapes.py` | **NOT EDITED** — remaining (see §3) | untouched |
| `tools/mixed_bw.cpp` | **NOT EDITED** — remaining (see §3) | untouched |
| `tmp/cite_shift_probe.py`, `tmp/cite_repoint.py`, `tmp/cites_into.py`, `tmp/bill_lines.py`, `tmp/linecount_check.py`, `tmp/kdump.py`, `tmp/stm_head.py`, `tmp/cite_shift_out.txt` | scratch tools I wrote; untracked, in `tmp/` (gitignored) | n/a |

Verified with `python3 tmp/linecount_check.py <files>` (asserts tree line count == HEAD line count —
needed because `progress/` cites both `engine/results.md` and `engine/README.md` by line number:
results.md at ≥25 sites, README.md at 15 sites: 61/83/126/138/162/182/246/313/314/323/455).

## 2. What was changed, with the arithmetic

### `engine/results.md` (annotate, never delete; historical rows dated)
* **Deferred lever 1** (unpacked projections): bill line now **406 launches/token**, annotated "the
  total read 424 until `62b73e9` retired the packed GDN widening".
* **Deferred lever 2** (dtype adapters) rewritten to the live shape: **97 casts**, derived
  `4·18 + 4·6 + 1 = 97` (4 per GDN layer + 4 per attention layer + 1 before `lm_head`), the retired
  launch named (`cast_bf16_to_f32_3`, the packed `in_proj_qkv`+`a`+`b` widening, one per GDN layer =
  18 launches, worth `18 × 1.85 µs = 33.3 µs/token`), and the note that **48 of the 97 are the MLP's**
  and 1.7′ touched none of them.
* Memory floor: `406 × 1.85 µs = 0.751 ms` (was 0.784 at 424).
* First-device-pass §"424-launch plan verify against §07" → kept as the record, dated + "(cast 97 /
  total 406 today; both derivers moved with the commit, so the two independent readings still agree)".
* `ms_per_step` cause section: 406 launches, `406 × 1.85 = 0.751 ms`, "424 when this paragraph was written".
* Gap table: cast row now `0.364 ms = 97 × 3.75 µs` (2.545 µs anchor: 0.247), priced total
  **+1.27 → +1.20 ms** (`1.27 − 0.067`), traffic half noted (`−0.222 MB read, −0.445 MB written`).
* Short-launch candidate: `237 of the 424` kept, plus `219 of the 406` = `49+97+18+19+36`; cost rows
  `219 × 3.75 µs = 0.82 ms` and `219 × 10 µs = 2.19 ms`.
* **Lever-3 "open opportunity" row (was L571): marked partially retired** — realised part
  `18 × 3.75 µs = −0.068 ms/token`, open part `97 × 3.75 = 0.364 ms`. This is the row the brief
  singled out; it is annotated in place, not removed.
* **Q6 / Q6.1 / Q7 dated sections**: each got an in-section dated correction (Q6 header, Q6 first
  paragraph, the `launches_per_step=424` quoted log line, the pre-registered anchors N/D, the Q7
  header + the two `424 × 42.4 µs` / `cast (115 …)` arithmetic rows). Recomputes: cast dispatch
  `97 × 3.75 = 0.364 ms`; Q6 dispatch column `1.63 − 0.067 = ≈1.56 ms`; Q6.1 `N = 406`, D upper end
  `1.63 − 0.067 = 1.563 ms` (with an explicit note that the 1.1 lower end cannot be shifted the same
  way, because the record does not say whether the deleted casts were in the small-multi-block set);
  Q7 ramp absurdity `42.4 µs × 406 = 17.2 ms`, kernel-granularity waits `406 + 1 = 407`.
  **Recorded log values were left as recorded** — the launch counter reading `424` and the stage-clock
  cast row are facts about a 2026-10-13 window.

Cross-checked live: `.venv/bin/python engine/gate.py --dry-run` section (d) prints
`cast — engine 97 vs §07 97`, `total — 406 vs 406`, other rows unchanged (glue 49, proj 114, gdn 72,
attn 6, mlp 48, other 20). That is the witness for every number above.

### `engine/README.md`
L10 (424 claim, dated + pointed at §13), L372 `(d) … cast=97 other=20 total=406` (matches the live
gate print), L472 `48 of the 97 casts`, L475 `HIP graphs over the 406-launch token (424 before 62b73e9)`.

### `tools/step_traffic_model.py` — the byte-budget defect
* `LAUNCHES["cast"] 115 → 97`, with a comment that `FAMILY_INSITU_MS["cast"] = 0.914` is a **measured**
  2026-10-13 median over 115 launches and must not be divided by the live count.
* `rows()` docstring plan line → `97 cast / 20 other = 406`.
* **Cast byte rows** (the real defect): dropped the retired packed term `(qkv + 2 * vh)` from read
  (`×2`) and write (`×4`) — retired bytes `18 × 6176 × 2 = 222,336 B` read,
  `18 × 6176 × 4 = 444,672 B` written per token (6176 = `kQkvRows 6144 + 2×kNumValueHeads 16`).
* Ladder entry → `cast round-trip (97 launches; 115 before 62b73e9)`, count field 97.
* Before/after, same command `python3 tools/step_traffic_model.py` at 128 lanes:

| row | HEAD | now |
|---|---|---|
| cast read | 0.593 MB | **0.371 MB** |
| cast write | 1.186 MB | **0.741 MB** |
| TOTAL read | 1532.533 | **1532.311** |
| TOTAL all | 1557.387 | **1556.720** |
| read:write | 61.7:1 | **62.8:1** |
| ladder cast row | 1.779 MB / 115 | **1.112 MB / 97** |

(The docstring's headline claim "~60:1" and the module prose survive unchanged — the ratio moved 1.1.)

## 3. Remaining Job 1 work (small, mechanical — hand this to the next worker)

1. `tools/instep_family_budget.py`
   * L14 docstring: `the step walks ~1.5 GB of distinct weights in 424 launches` → 406 (dated).
   * L214: `n_all = int(tot["launches"]) if tot else 424` → `else 406` **and** keep the fallback
     honest: it is a fallback constant, so name the source in a trailing comment.
   * 0 line-form cites exist into this file → line count is free, but the file has a `--selftest`-ish
     path; re-run `python3 tools/instep_family_budget.py` after the edit (CPU-only).
2. `tools/prefill_gemm_shapes.py:206` — `115 cast launches per token` → 97 (dated).
3. `tools/mixed_bw.cpp` — comment-only, **same-line substitutions only** (`mixed_bw.cpp:417` is cited
   from `devices/strix-halo-mixed-traffic.md`): L7 (`~115` cast launches), L288, L620, L944
   (`424`-launch step references). Keep total line count at HEAD.
4. Root `README.md` — grep found **no** bill claim; nothing to do. Recorded so nobody re-hunts it.
5. Re-run before commit: `python3 tmp/linecount_check.py engine/results.md engine/README.md`
   (must read 4690 / 496) and `git diff -U0 engine/results.md | grep '^@@'` through the parity
   checker used in this run — an unbalanced hunk means a cite just rotted.

## 4. Job 2 — citation state (analysis complete, **no edits made**)

`engine/decode_step.cpp` moved **−1/−3 lines** since `62b73e9^` (region-specific: −1 from ~250, −3
from ~254/481, −6 in 333–478); `engine/gate.py` moved **+60 net**. Tools written for this:
`tmp/cite_shift_probe.py` (base-vs-head text at every cited coordinate: 87 cites into the two files,
76 DIFF / 11 SAME) and `tmp/cite_repoint.py` (finds where the cited text actually lives now; skips
`progress/`).

**Verified re-point targets (live files, same-line substitution, safe):**

| citing site | cites | correct line now | note |
|---|---|---|---|
| `prototype/gemm_algo_sweep/README.md:89` | `decode_step.cpp:517` | **514** | text match unique |
| `:90` | `:722` | **719** | text match unique |
| `:91` | `:527` | **524** | text match unique |
| `:92` | `:530` | **527** | text match unique |
| `prototype/gemm_algo_sweep/gemm_algo_sweep.cpp:98,99,100,101` | same four | 514 / 719 / 524 / 527 | same targets |
| `prototype/gemm_algo_sweep/{README.md:88, gemm_algo_sweep.cpp:97}` | `:514` | **hand call needed: 583–584** (`project(... kQprojRows, kHidden)`); `} else {` is not a unique text anchor | the cite names the *call site*, not a brace |
| `prototype/gemm_algo_sweep/results.md:239,257-263` | `:513,516,517,526,529,532,535,721` | **hand rewrite: 583/586/589/596/599/602/605/810** — these cite the eight projection call sites, which moved as a block, not by a uniform shift | same-line edits, keep line count |
| `prototype/gdn_rec_isa/{gdn_rec_isa_kernels.h:372, results.md:337}` | `:677` | **674** (text-verified) | |
| same | `:682` | **679** (text-verified) | |
| `specification/ASSUMPTIONS.md:28` | `gate.py:3692` | **3761** (text-verified) | **OFF-LIMITS → propose only** |

**Could not be verified / must not be mechanically renumbered (report, do not edit):**
* `engine/results.md:2670 → decode_step.cpp:455` — the base line at that coordinate is **blank**, so
  there is no text to match; the cite is a *range* (`455-471`) describing `lm_head` + D2H + argmax.
  Hand judgement: the `lm_head`/D2H block now starts elsewhere; needs a human sentence, not a number.
* `prototype/gdn_rec_isa/traffic_census.output.txt:10,16` — a **captured output file**. Editing it
  falsifies an evidence capture; regenerate `traffic_census.py` instead, or leave and note it.
* `progress/**` cites (24 of the 87) — journals are frozen records; per instructions, same-line
  substitutions only, and the ones from `2026-10-12`/`2026-10-16` predate `62b73e9` and are inherited
  rot already on the main agent's list (`HANDOFF_41a` §6, `HANDOFF_42prime` §6). **No journal edited.**
* Inherited vs new: of the 76 DIFF rows, the ones on `progress/2026-10-12/plan-prefill-2026-10-12.md`
  and `progress/2026-10-16/*` were already broken by earlier commits (embed-gather added 54 lines to
  `decode_step.cpp`); the `62b73e9`-caused set is exactly the 26 rows listed in `HANDOFF_41a` §6.

## 5. Job 3 — proposed dated-correction lines for the off-limits files (write these, do not edit)

`specification/ASSUMPTIONS.md:28` (also the re-point target moved 3692 → 3761):
> *(2026-10-17: the cited `gate.py` line moved from 3692 to 3761 in `62b73e9`; the ceiling claim itself
> is unchanged.)*

`devices/strix-halo-mixed-traffic.md` (L45 cast row, L249–250 `G_step` probe ratio):
> *(2026-10-17, `62b73e9`: the cast family is 97 launches, not 115 — the packed GDN widening lost its
> reader. The row's per-launch figures stand; the family byte totals move to 0.371 MB read / 0.741 MB
> written per token (−0.222 / −0.445), and `G_step`'s read:write denominator moves 61.7:1 → 62.8:1.)*

`bench/performance-status.md` (L618 "115 on both routes", L659 "424 on both routes"):
> *(2026-10-17: both route-symmetric counts moved with `62b73e9` — cast 97, per-token bill 406. §13
> carries the re-take; the §6/§12 rows above are the plan as measured on their own dates.)*

Note the ledger's §1–§5 are banner-marked rather than edited for line-citation reasons (AGENTS.md
citation hygiene), so these belong as appended dated lines, and `bench/performance-status.md` line
numbers are cited from `progress/` — any append there needs the user's explicit words.

## 6. Job 3b — `tools/cite_quote_check.py` false-positive patterns (531 cites checked, 22 suspects)

Two real classes, both from the fragment-grabbing heuristic, neither a defect in the cites:
1. **Wrong-fragment grab on multi-cite lines.** The checker takes the first quoted fragment ≥12 chars
   from the *citing line*, so a line carrying two cites can have fragment A compared against target B,
   producing a spurious `QUOTE NOT NEAR`. `tools/citation_rot_check.py` is blind to the class entirely
   (it compares coordinates to HEAD text), and `tools/doc_ref_check.py` never sees the quote. The fix is
   in the checker, not the docs: bind each extracted quote to the cite whose offset it follows.
2. **Blank/prose base lines.** Where the cited coordinate lands on a blank line or on a line whose text
   the citing prose paraphrases rather than quotes, the checker reports "no quote found" and cannot
   adjudicate — it is not evidence of rot (e.g. `engine/results.md:2670 → decode_step.cpp:455`).
Also worth recording: a `DIFF` from my base-vs-head probe is only evidence that *the text at the
coordinate changed*, not that the cite is wrong — several DIFFs are cites to a moved block whose new
number is not a uniform shift (the eight projection call sites), which is why §4 separates
"text-match unique" from "hand call needed".

## 7. Commands run (all CPU-only; **no timing taken, no device run**)

```
git show 62b73e9 -- engine/decode_step.cpp engine/stage_clock.cpp     # the bill move
grep -rn '424\|115\|406\|97' --include=*.md --include=*.py --include=*.cpp --include=*.h .
python3 tmp/cite_shift_probe.py            # 87 cites into the 2 shifted files; 76 DIFF / 11 SAME
python3 tmp/cite_repoint.py                # where the cited text now lives (live files only)
python3 tmp/cites_into.py engine/README.md # 15 cites from progress/ -> line count is frozen
python3 tmp/linecount_check.py engine/results.md engine/README.md tools/step_traffic_model.py
.venv/bin/python engine/gate.py --dry-run  # section (d): cast 97/97, total 406/406  [CPU dry-run]
python3 tools/step_traffic_model.py        # before/after byte budget table in §2
python3 -m py_compile tools/step_traffic_model.py
python3 tools/cite_quote_check.py          # 531 cites, 22 suspects (run earlier in session)
```

`speccheck/` was not run: nothing under `speccheck/` or any gate file was touched, and `gate.py`
is off-limits for this task.

## 8. Residual risks

* `engine/results.md` and `engine/README.md` are line-count-clean, but the **hunk-parity check must be
  re-run after any further edit** (`tmp/linecount_check.py` + the `-U0` parity loop) — one stray wrap
  rots ≥15 journal cites silently.
* Some corrected paragraphs are now long single lines (L135, L571, the Q6/Q7 headers). The file's own
  style tolerates that, but a later `fmt`-style pass would reflow them and rot the cites. If someone
  rewraps `engine/results.md`, the `progress/` cites into it must be re-pointed in the same commit.
* `tools/step_traffic_model.py` grew 8 lines. If a future cite points at it by line number, note that
  its `LAUNCHES`/`FAMILY_INSITU_MS` block is now the one place where a measured stale ms and a live
  launch count deliberately sit adjacent, with a comment forbidding the division.
* The Q6.1 pre-registered D band was annotated, not recomputed at both ends, because the record does
  not say whether the 18 deleted casts were inside the small-multi-block set the 1.1 end divided by
  1.8×. That is a real gap in the direction, not a fudge — a re-take should re-derive D from the plan.
* Jobs 2 and 3 were **not applied to any file**; §4/§5 are proposals awaiting the main agent's
  decision on whether `prototype/*` re-points land in the same commit as the doc cascade or separately.
