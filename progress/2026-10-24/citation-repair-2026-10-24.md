# Citation repair — the line citations the MTP work rotted (2026-10-24)

CPU-only pass. No device work, no code touched, no gate loosened.

`7e75bc2` → `HEAD` grew `engine/gate.py` by **+829/−22 lines** (net +807), `engine/decode_run.cpp` by
**+459/−128** (net +331) and `engine/decode_step.cpp` by **+14/−6**, while gate (m), the `--mtp-accept`
default flip, the GDN span scoping (`kMtpRoundGdnSpan`) and adaptive depth landed. Every `path:NNNN`
citation below an insertion therefore stopped naming the line it meant: `engine/gate.py:6283` quoted
`MTPGEOM rows == rows_of_k(k) == k + 1` and resolved to prose about a verdict. The repair is
`040ab24` (*docs: re-point the 42 line citations rotted by the MTP work*); this file is its record.

## 1. What rotted, and what "repaired" means here

`tools/citation_rot_check.py --against 7e75bc2 --all-rows` read **65 rotted rows**, which are
**44 distinct `(cited file, line)` coordinates**:

| cited file | rotted coordinates | Δ of the shifts |
|---|---|---|
| `engine/gate.py` | 18 | +2 (before the first insertion), +39, +98, +746, +750, +757 |
| `engine/decode_run.cpp` | 24 (21 applied, 1 withheld, 2 stale) | +1, +11, +43, +145, +146 |
| `engine/decode_step.cpp` | 2 | +8 |

Disposition: **41 coordinates re-pointed** (45 citing lines in 30 files), **1 withheld** (§5),
**2 stale-by-substance** (§4, the part for the main agent to adjudicate).

Every applied re-point is witnessed by the assertion the checker itself cannot make:

> `worktree[new] == 7e75bc2[old]`, byte for byte — 41 of 41 applied coordinates.

## 2. The gate the brief asked for cannot be met, and why (the standing observation)

The brief expected the re-point to drive `--against 7e75bc2` down to "only stale-by-substance".
It cannot, and the reason is the tool's own predicate, not the repair:

```python
if old[n - 1] != new[n - 1]:      # rot_rows(): text at ONE coordinate, two revisions
```

`n` comes from the citing document, but the comparison never consults what the document means — it
only asks whether the cited file's text at line `n` differs between `--against` and the tree. So a
line that **moved** is rotted at *both* numbers: the old one (its text is gone from there) and the new
one (the revision's text there is whatever the shift displaced). Measured on this very repair:
**42 of 42** derivable coordinates read rotted at both the old and the new number; **0** read rotted at
neither. Re-pointing moves a row; it cannot delete one. Driving the count to 0 would need the code put
back at its old line numbers or the citations deleted, and the brief forbids both.

After the repair the tool reads **70 rows**, which decompose exactly:

| rows | what they are |
|---|---|
| 59 | the repaired cites, now naming the moved line at its **new** number — the irreducible residue of §2 |
| 4 | the stale-by-substance coordinates, left alone on purpose (§4) |
| 2 | the withheld range endpoint (§5) |
| 5 | self-referential: a line this pass substituted (`archive-review-2026-10-16.md:296/:321`, `specification/ASSUMPTIONS.md:28`) is itself cited by line elsewhere. `citation_rot_check.py`'s header predicts this — *"a same-line substitution IS reported as drift, deliberately … it should be a decision, not a side effect"*. It is a decision, and it is the whole point of the pass: the fact stands, only the coordinate inside the line moved. |

The two checks that **are** meaningful both pass, and both are stated with their tails in §8:

* `--against HEAD` → **0 rotted** (after the repair commit: the cited code files were never touched, so nothing drifted between HEAD and the tree);
* the witness of §1 → **41/41** (`worktree:new == ref:old`).

The brief's own standing observation still holds and is why this document exists at all:
`citation_rot_check.py` cannot see drift introduced inside the commit being checked, because its
default baseline is `HEAD` and `HEAD` moves with the commit. The only working guard is a coordinate
diff against a pre-change ref — `--against <prev>` **before** the drift is committed, which is what
found these 65. This pass adds the mirror-image half of that sentence: the tool is equally unable to
**certify a repair** made in the citing direction, which is why §1's witness, not the tool, is the
acceptance evidence. `tools/citation_repoint.py` is the pre-commit sibling (it maps `HEAD` → working
tree); giving it an explicit `--against` ref would make it the post-repair witness too, if this class
of drift recurs.

## 3. Method — derived twice, refused rather than guessed

1. `tools/citation_rot_check.py --against 7e75bc2 --all-rows > tmp/rot.txt` (65 rows).
2. For each coordinate, two **independent** derivations had to agree:
   * **alignment** — `difflib.SequenceMatcher` over the whole file between `7e75bc2` and the tree; the
     equal-block opcode containing the old line gives its new coordinate, and that prediction is only
     accepted if the tree's line there carries the identical text (self-witnessing);
   * **exact-string search** — every whole-line exact match of the old text in the current file.
   Agreement: 42 coordinates, **0 disagreements**.
3. `tools/archive_doc.py` was read and could not be used: its two modes rewrite *path tokens*
   (`--pair OLD:NEW`, `--split`), and nothing in it moves a `:NNN` number. Its rule was honoured by
   hand instead — same-line substitution only, and the tool's own refusal ("`REFUSE: … would change
   line count`") was reproduced as an assertion in the repair script.
4. Refusals, which is where an editorial decision would otherwise have been smuggled in:
   * an endpoint whose text is gone is **not** re-pointed, and a range containing one is left whole
     (§4, §5) — a half-moved range like `:1454-1582` would name a 128-line span nobody meant;
   * a number not rotted at all is left alone (its text did not move);
   * rewritten lines were checked to differ **only in digits**: 45 changed line pairs, 0 non-digit
     deltas, 0 change in the count of numeric tokens per line.

## 4. Stale-by-substance — the list for adjudication

Neither of these can be fixed by renumbering, because the text the citation named no longer exists,
and pointing at a lookalike would be a false claim in a frozen journal. The numbers are untouched.

### 4a. `engine/decode_run.cpp:153`

* **Cited by** `prototype/attn_percost/A9-LC-impl.md:42` and `prototype/attn_wmma/E1-result.md:289`.
* **The claim in both**: `--attn-fattn` defaults to 1, so the WMMA/`attn_fattn_wave` path is on the
  shipping path.
* **Text that was there at `7e75bc2`**: `// two tokens. Default off, and off means this pointer is
  never allocated (§4g's inertness argument).` — the third line of the `--mtp-accept` option comment.
  The accept-default flip rewrote that block ("ON by default since the accept loop became the
  default…"), so the text is gone, not moved.
* **The catch**: the coordinate did not name `--attn-fattn` even at the reference revision. Line 153
  was already inside the MTP comment block; `int attn_fattn = 1;` was at `:306` then and is at
  **`:349`** now. So this citation was ~150 lines off *before* today's drift, and re-pointing 153→349
  is a retarget (an editorial act), not a re-number. Both options are open: retire the coordinate, or
  retarget it at `engine/decode_run.cpp:349`.

### 4b. `engine/decode_run.cpp:1454` (and its range partner, §5)

* **Cited by** `prototype/perprompt/A7-CANDIDATES.md:25` and `prototype/perprompt/a7_host_split.sh:9`,
  both as the range `engine/decode_run.cpp:1454-1468`.
* **The claim in both**: `wall_ms` is printed by `wave_pass`, which calls `st.reset()` and runs
  `prefill_wave` again on the same prompt, so §49's subtraction is a second warm pass.
* **Text that was there at `7e75bc2`**: `// token stream, not a red comparison, so it is refused at
  the flag rather than caught later.` — the reason clause of the pre-scoping refusal `--mtp-accept pins
  --gdn-loop 1`. The span scoping replaced that whole-run pin with `kMtpRoundGdnSpan`
  (`engine/decode_step.h:474`, applied per-body in `submit_wave_body`; `engine/decode_run.cpp:1566`
  is the note that says so now), so the sentence is gone.
* **The catch**: at `7e75bc2` the range `1454-1468` bracketed flag-refusals, not `wave_pass` —
  `wave_pass` was at `:3359` then (its `st.reset()` at `:3368`) and is at **`:3557`** now (with
  `st.reset()` at `:3566`). So the A7 cites were already ~1,900 lines off before today; a semantic
  repair means retargeting them at `3557-3576`, which is a decision, not a shift.

## 5. Withheld: `engine/decode_run.cpp:1468`

Its text survived and derives cleanly to **1582**, but it is the far endpoint of §4b's
`1454-1468`. Moving one endpoint of a range whose start is stale would print `1454-1582`, so the whole
cite was left alone. If §4b is retired, this retires with it; if §4b is retargeted at `wave_pass`,
this becomes moot.

## 6. Four re-points whose evidence is a context window, not a unique string

Four cited texts are not unique (`}`, `if stand_in:`, and two **blank lines**), so the exact-string
search could not adjudicate them and the alignment did. The witness is not a hand-picked snippet: it is
the **maximal run of byte-identical lines containing the coordinate**, which is large enough that the
mapping cannot be a coincidence:

| cited | old → new | cited text at the old coordinate | identical run (ref range == work range) |
|---|---|---|---|
| `engine/decode_run.cpp` | 2049 → 2195 | `}` | ref 1530–2613 == work 1676–2759 (+146, 1,084 lines) |
| `engine/gate.py` | 429 → 431 | *(blank)* | ref 142–1154 == work 144–1156 (+2, 1,013 lines) |
| `engine/gate.py` | 7063 → 7813 | *(blank)* | ref 7013–7284 == work 7763–8034 (+750, 272 lines) |
| `engine/gate.py` | 7182 → 7932 | `if stand_in:` | ref 7013–7284 == work 7763–8034 (+750, 272 lines; 4 copies of `if stand_in:` exist, the run picks the one) |

Two of these name a **blank line in both revisions** (the `gate.py` rows at 429 and 7063), which is a
fact about the original citation, not about this repair (see §7). `tools/doc_ref_check.py` reports such a
cite as `line … is blank — the cite points at nothing`, and did so at the old coordinate as well as the
new one.

## 7. Cites already imprecise at the reference revision

A re-point restores text-equality with the reference revision; it cannot repair a citation whose text
was already unrelated to the claim at that revision. Each row below was checked by grepping the claiming
text out of both revisions (all re-pointed faithfully, so each is exactly as loose as it was,
no looser):

| citing line | claim | cited at `7e75bc2` | the claim's actual home then → now |
|---|---|---|---|
| `HANDOFF_ceiling_raise.md:58`, `:256`, `queue2-2026-10-19.md:48` | `kSessionDefaultCap = 4096` | `namespace {` | `:72` → `:73` |
| `generation-limits.md:108`, `route-S1-segments-2026-10-21.md:133` | `bool time_stages = false` | a `--gemm-acc` comment line | `:514` → `:557` |
| `structure-proposal-2026-10-16.md:88`, `:421` | `from tools import band_history` | echo-verdict prose | `:212` → `:214` |
| `A6-G1-impl.md:318` | the help string carrying a literal `5 %` | *(blank)* | `:8555` → `:9321` |
| `ASSUMPTIONS.md:28` (A-32) | the hipBLAS-version provenance check | `moved = wide.copy()` | `numerics_provenance_checks` at `:4759` now |

These are listed, not fixed: choosing a new subject line is a semantic decision on a frozen journal,
the same class of act as retiring §4.

## 8. Commands run, with their tails

```
$ python3 tools/citation_rot_check.py --against 7e75bc2 --all-rows      # before the repair
scanned 1060 tracked files against 7e75bc2; checked 1550 line-anchor(s); skipped 461
(absent-or-untracked-at-one-revision=459, out-of-range-at-one-revision=2)
65 rotted line-citation(s).                                                       # exit 1

$ python3 tools/citation_rot_check.py --against 7e75bc2 --all-rows      # after the repair
70 rotted line-citation(s).                                                       # exit 1 — see §2

$ python3 tools/citation_rot_check.py --all-rows                        # --against HEAD, post-commit
0 rotted line-citation(s).                                                        # exit 0

$ python3 tools/citation_rot_check.py --selftest
SELFTEST PASS — a planted shift is caught, a quiet corpus and a below-only edit stay quiet

$ python3 tools/citation_census.py --anchors      # HEAD~1 → HEAD, same working tree
799 citing files scanned (0 unreadable), 1512 line-form citations found            (both runs)
ANCHORS FAIL — 184 unresolved  →  ANCHORS FAIL — 178 unresolved

$ python3 tools/citation_census.py --check        # HEAD~1 and HEAD, byte-identical output
CHECK PASS — every form matches the census, with each form's record list non-empty
F1 101/101 · F2 171/171 · F4 16/16 · F5 24/24 · F5b 95/95 · F6 66/66 · live-origin F1 5/5

$ python3 tools/doc_ref_check.py                  # HEAD~1 → HEAD, same working tree
ANCHOR-OK  1469 (line anchors: 1469 resolve, 37 broken)  →  ANCHOR-OK  1476 (1476 resolve, 34 broken)
BROKEN-ANCHOR 37 → 34;  BROKEN-LABEL 29 → 29 (unchanged)

$ bash engine/build.sh
built engine/build/loader_test
built engine/build/decode_run                                                   # no code was touched
```

Two notes on those measurements. (a) The `--anchors` and `doc_ref_check` **before** column (184
unresolved, 37 broken) came from a `git worktree add <dir> HEAD~1 --detach` **with this tree's untracked
paths symlinked into it** (`engines/`, `engine/.pi`, `data/`, `prototype/*/out`,
`prototype/perprompt/a7_fixture.log`). Without those links the same tools print 188 unresolved and a
different `BROKEN-PATH` population, because a worktree carries only tracked files — that is the trap to
watch for when measuring a citation tool against an older revision, and it is why a bare worktree read of
“37 → 34” would have been an accident rather than a comparison. (b) `BROKEN-ANCHOR` fell by 3 and
`--anchors` unresolved by 6: several cites that used to land on a blank line now land on text (the three
lines naming `decode_run.cpp` line 65, the two naming `gate.py` line 3834, and the one naming `gate.py`
line 429), while the two blank-line targets of §6 moved from one blank line to another and stay broken.

## 9. Line-count neutrality — proof for every file touched

The diff is 45 changed line pairs; `tmp/prove_digits_only.py` shows **0** lines whose non-digit
characters differ and **0** lines whose count of numeric tokens changed, and `wc -l` is identical
before and after for all 30 files:

| file | `wc -l` before | after | lines changed |
|---|---|---|---|
| `engine/generation-limits.md` | 185 | 185 | 2 |
| `progress/2026-10-12/plan-prefill-2026-10-12.md` | 584 | 584 | 1 |
| `progress/2026-10-16/archive-review-2026-10-16.md` | 528 | 528 | 2 |
| `progress/2026-10-16/performance-status-2026-10-16.md` | 1160 | 1160 | 1 |
| `progress/2026-10-16/structure-proposal-2026-10-16.md` | 424 | 424 | 2 |
| `progress/2026-10-17/HANDOFF_17prime_gdn-cast-deletion.md` | 381 | 381 | 1 |
| `progress/2026-10-17/HANDOFF_41a_cast-deletion.md` | 261 | 261 | 2 |
| `progress/2026-10-17/HANDOFF_ceiling_raise.md` | 310 | 310 | 4 |
| `progress/2026-10-17/glue-cast-copy-bytes-census.md` | 489 | 489 | 1 |
| `progress/2026-10-18/llama-fattn-anatomy.md` | 680 | 680 | 1 |
| `progress/2026-10-19/queue2-2026-10-19.md` | 147 | 147 | 1 |
| `progress/2026-10-21/performance-status-2026-10-21.md` | 2842 | 2842 | 1 |
| `progress/2026-10-21/route-S1-segments-2026-10-21.md` | 161 | 161 | 1 |
| `prototype/attn_decode/results.md` | 1534 | 1534 | 1 |
| `prototype/attn_percost/A6-G1-impl.md` | 330 | 330 | 1 |
| `prototype/attn_percost/A6-result.md` | 409 | 409 | 1 |
| `prototype/attn_percost/A9-BET.md` | 202 | 202 | 1 |
| `prototype/attn_percost/A9-LC-impl.md` | 289 | 289 | 4 |
| `prototype/gemm_algo_sweep/README.md` | 409 | 409 | 1 |
| `prototype/gemm_algo_sweep/gemm_algo_sweep.cpp` | 3141 | 3141 | 1 |
| `prototype/glue_copy_fold/GLUECOPY-HANDOVER.md` | 287 | 287 | 1 |
| `prototype/maxsc_allowance/MAXSC-HANDOVER.md` | 332 | 332 | 1 |
| `prototype/mtp_verify/causal_bound.cpp` | 1184 | 1184 | 2 |
| `prototype/perprompt/A7-CANDIDATES.md` | 300 | 300 | 3 |
| `prototype/perprompt/a7_host_split.sh` | 230 | 230 | 1 |
| `prototype/perprompt/a7_lib.sh` | 50 | 50 | 2 |
| `prototype/perprompt/a7_scratch_bytes.py` | 146 | 146 | 1 |
| `prototype/proj_gemv/results.md` | 843 | 843 | 1 |
| `prototype/wave_shape/wave_shape.hip.cpp` | 2081 | 2081 | 2 |
| `specification/ASSUMPTIONS.md` | 28 | 28 | 1 |

## 10. The re-points

| cited file | old → new | Δ | cited by (pre-repair line) |
|---|---|---|---|
| `engine/decode_run.cpp` | 65 → 66 | +1 | `HANDOFF_ceiling_raise.md:256`, `HANDOFF_ceiling_raise.md:58`, `queue2-2026-10-19.md:48` |
| `engine/decode_run.cpp` | 105 → 116 | +11 | `performance-status-2026-10-16.md:1105`, `attn_decode/results.md:1479` |
| `engine/decode_run.cpp` | 107 → 118 | +11 | `glue-cast-copy-bytes-census.md:31` |
| `engine/decode_run.cpp` | 110 → 121 | +11 | `attn_decode/results.md:1479` |
| `engine/decode_run.cpp` | 111 → 122 | +11 | `glue-cast-copy-bytes-census.md:31` |
| `engine/decode_run.cpp` | 117 → 128 | +11 | `performance-status-2026-10-16.md:1105` |
| `engine/decode_run.cpp` | 176 → 219 | +43 | `A6-result.md:128` |
| `engine/decode_run.cpp` | 205 → 248 | +43 | `A9-LC-impl.md:93` |
| `engine/decode_run.cpp` | 262 → 305 | +43 | `mtp_verify/causal_bound.cpp:1112`, `:51` |
| `engine/decode_run.cpp` | 294 → 337 | +43 | `gemm_algo_sweep/README.md:113`, `gemm_algo_sweep.cpp:2795` |
| `engine/decode_run.cpp` | 340 → 383 | +43 | `generation-limits.md:108`, `route-S1-segments-2026-10-21.md:133` |
| `engine/decode_run.cpp` | 351 → 394 | +43 | `wave_shape.hip.cpp:58` |
| `engine/decode_run.cpp` | 410 → 453 | +43 | `plan-prefill-2026-10-12.md:29` |
| `engine/decode_run.cpp` | 412 → 455 | +43 | `plan-prefill-2026-10-12.md:29`, `GLUECOPY-HANDOVER.md:275` |
| `engine/decode_run.cpp` | 1507 → 1652 | +145 | `a7_lib.sh:21` |
| `engine/decode_run.cpp` | 1649 → 1795 | +146 | `A7-CANDIDATES.md:23`, `a7_scratch_bytes.py:8` |
| `engine/decode_run.cpp` | 1654 → 1800 | +146 | `A7-CANDIDATES.md:23`, `a7_scratch_bytes.py:8` |
| `engine/decode_run.cpp` | 1658 → 1804 | +146 | `a7_lib.sh:30` |
| `engine/decode_run.cpp` | 1851 → 1997 | +146 | `performance-status-2026-10-21.md:2266`, `A7-CANDIDATES.md:22`, `a7_host_split.sh:6` |
| `engine/decode_run.cpp` | 1853 → 1999 | +146 | `performance-status-2026-10-21.md:2266`, `A7-CANDIDATES.md:22`, `a7_host_split.sh:6` |
| `engine/decode_run.cpp` | 2049 → 2195 | +146 | `A7-CANDIDATES.md:37` *(context-derived, §6)* |
| `engine/decode_step.cpp` | 1109 → 1117 | +8 | `wave_shape.hip.cpp:27` |
| `engine/decode_step.cpp` | 1169 → 1177 | +8 | `generation-limits.md:48` |
| `engine/gate.py` | 160 → 162 | +2 | `structure-proposal-2026-10-16.md:421`, `:88` |
| `engine/gate.py` | 396 → 398 | +2 | `HANDOFF_ceiling_raise.md:56` |
| `engine/gate.py` | 397 → 399 | +2 | `A9-LC-impl.md:131` |
| `engine/gate.py` | 429 → 431 | +2 | `A9-LC-impl.md:131` *(context-derived, §6)* |
| `engine/gate.py` | 436 → 438 | +2 | `HANDOFF_ceiling_raise.md:56` |
| `engine/gate.py` | 769 → 771 | +2 | `proj_gemv/results.md:417` |
| `engine/gate.py` | 892 → 894 | +2 | `A9-BET.md:103`, `A9-LC-impl.md:216` |
| `engine/gate.py` | 902 → 904 | +2 | `A9-BET.md:103`, `A9-LC-impl.md:216` |
| `engine/gate.py` | 3569 → 3608 | +39 | `HANDOFF_17prime_gdn-cast-deletion.md:192` |
| `engine/gate.py` | 3834 → 3873 | +39 | `archive-review-2026-10-16.md:321`, `HANDOFF_41a_cast-deletion.md:230` |
| `engine/gate.py` | 3907 → 3946 | +39 | `ASSUMPTIONS.md:28` |
| `engine/gate.py` | 6283 → 6381 | +98 | `archive-review-2026-10-16.md:296`, `HANDOFF_41a_cast-deletion.md:232` |
| `engine/gate.py` | 6762 → 7508 | +746 | `HANDOFF_ceiling_raise.md:57` |
| `engine/gate.py` | 7063 → 7813 | +750 | `A6-G1-impl.md:318` *(context-derived, §6)* |
| `engine/gate.py` | 7182 → 7932 | +750 | `MAXSC-HANDOVER.md:51` *(context-derived, §6)* |
| `engine/gate.py` | 7247 → 7997 | +750 | `A9-LC-impl.md:135` |
| `engine/gate.py` | 7640 → 8397 | +757 | `MAXSC-HANDOVER.md:51` |
| `engine/gate.py` | 8247 → 9004 | +757 | `llama-fattn-anatomy.md:583` |

Two re-pointed lines deserve a flag: `HANDOFF_41a_cast-deletion.md:230` and `:232` sit **inside a
fenced block that quotes an earlier session's rot-report** (§6 of that handoff, "ROTTED line-form
citations — for you to re-point"). They were treated as live citations, because the checker parses
them as such and they then resolve to the right code. If the verbatim quote is worth more than the live
pointer, revert those two lines — it is four digits, and the checker's row count does not change
either way (§2).

## 11. Cite forms this pass could not touch, and their derived numbers

`citation_rot_check.py`'s `CITE` regex requires a path before the colon, so bare continuation cites
(`:6951`) and slash siblings (`:205/397/1770`) are outside its population and outside this repair. They
drifted exactly as the parsed cites did. The table below is **enumerated, not remembered**: every
anchor token on each of the 45 lines this pass edited was extracted from the pre-repair text and run
through the same alignment, so the destinations are derived the same way as §10's.

| citing line (pre-repair) | invisible form | owner | derived destination |
|---|---|---|---|
| `engine/generation-limits.md:108` | bare `:2496` | `engine/decode_run.cpp` | 2496 → 2642 |
| `progress/2026-10-17/HANDOFF_ceiling_raise.md:57` | bare `:6951` | `engine/gate.py` | 6951 → 7697 |
| `prototype/attn_percost/A9-LC-impl.md:93` | slash siblings of the parsed `:205/397/1770` — `:205` was parsed and *is* repaired, so this list is now internally inconsistent | `engine/decode_run.cpp` | 397 → 440, 1770 → 1916 |
| `prototype/attn_percost/A9-LC-impl.md:93` | bare `:878` (usage), `:985` (help) | `engine/decode_run.cpp` | 878 → 956, 985 → 1063 |
| `prototype/attn_percost/A9-LC-impl.md:135` | bare `:5257`, `:7496`, `:8547` | `engine/gate.py` | 5257 → 5304, 7496 → **8253**, 8547 → 9313 |
| `prototype/glue_copy_fold/GLUECOPY-HANDOVER.md:275` | bare `:1853` | `engine/decode_run.cpp` | 1853 → 1999 |
| `prototype/glue_copy_fold/GLUECOPY-HANDOVER.md:275` | bare `:542`, `:545`, `:552`, `:555` | `engine/decode_step.cpp` | **unmoved** — that file's drift starts at 1109, so there is nothing to do |
| `prototype/perprompt/A7-CANDIDATES.md:22` | bare `:1943`, `:1890`, `:2047` | `engine/decode_run.cpp` | 1943 → 2089, 1890 → 2036, 2047 → 2193 |
| `prototype/perprompt/A7-CANDIDATES.md:23` | bare `:1655`, `:1756-1701`, `:1782-1740`, `:1809`, `:1859-1850`, `:2033` | `engine/decode_run.cpp` | 1655→1801, 1756→1902, 1701→1847, 1782→1928, 1740→1886, 1809→1955, 1859→2005, 1850→1996, 2033→2179 |
| `prototype/perprompt/A7-CANDIDATES.md:25` | bare `:1453`, `:1462-1416`, `:1543-1520`, `:1513-1472` — the line whose parsed range is withheld (§5) | `engine/decode_run.cpp` | 1453 **GONE** (same rewritten flag-refusal block as §4b), 1462→1576, 1416→1525, 1543→1689, 1520→1665, 1513→1658, 1472→1586 |
| `prototype/perprompt/a7_host_split.sh:6` | bare `:2047` | `engine/decode_run.cpp` | 2047 → 2193 |

A bare anchor on a line this pass did **not** touch is outside even this list — the population above is
exactly “anchors sharing a line with a repaired cite”, which is what the alignment could be pointed at
without re-deriving the whole corpus. Extending `CITE` in `citation_rot_check.py` to a bare-anchor
continuation (inheriting the nearest preceding path on the line) would put all of them into the gate,
and that is the check worth having before the next big `gate.py` edit.

## 12. Residual risks

1. **§4 is undecided by design.** Two coordinates (four citing lines) still name text that no longer
   exists; they are listed with the coordinate their claim actually lives at now, and retiring or
   retargeting them is the main agent's call.
2. **§7 is unfixed by design.** Roughly a dozen citations were already naming a neighbouring line at
   `7e75bc2`. Any check that compares a citation's prose to the code text (rather than coordinates to
   coordinates) would find more of these; none exists today.
3. **§11 is stale but invisible.** Bare-`:NNN` and slash-sibling cites are not in any checker's
   population, and one of them (`A9-LC-impl.md:93`) is now *internally* inconsistent: its head number
   was repaired and its two siblings were not. Extending `CITE` in `citation_rot_check.py` to a
   bare-anchor continuation (inheriting the nearest preceding path on the line) would put them in, and
   §11 is the data to seed it with.
4. **The checker still reads 70 rows against `7e75bc2`.** That is the residue §2 explains, not
   unfixed work — and it is *expected to stay* for as long as the MTP diff stands. Against `HEAD` it
   reads 0, and any later re-take against `7e75bc2` must classify before it panics.
5. **The two quoted-report lines** (`HANDOFF_41a_cast-deletion.md:230/:232`) are a judgement call
   recorded in §10; revert if verbatim history outranks a live pointer.
