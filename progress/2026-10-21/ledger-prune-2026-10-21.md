# The 2026-10-21 ledger prune — what moved, what stayed, what was rewritten, and what it cannot see

Written alongside the prune. The predecessor is `progress/2026-10-16/ledger-prune-2026-10-16.md`, which
did the same job for the previous five months; the shape of this one follows it because that one's shape
worked.

## A. The two files

| file | lines | role |
|---|---|---|
| `engine/performance-status.md` | **307** | the live ledger. Current measured state, §1–§6 kept at their old numbers, and it moved out of `bench/` in the same pass |
| `progress/2026-10-21/performance-status-2026-10-21.md` | **2,842** = 2,828 frozen + 14 provenance footer | the ledger exactly as it stood at HEAD `2bdea89`, byte-identical to the file it was pruned from |
| `tools/archive_doc.py` | new | moves a document and re-points every citation naming its old path, as same-line substitutions; `--selftest` covers the cite forms and refuses a substitution that would shift a line |
| `tools/citation_census.py` | changed | learned `RENAMES_AT_CENSUS`, so a moved document still contributes its census-era citations to the census check (§D) |

Why the live ledger is short again: it grew from 326 lines at the last prune to 2,828 in five days. The
same two readers were fighting over one file — "what is the engine's number today" wants a table, "how do
I know that number is real" wants the whole record of how it was taken — and by §60 the append-only half
had buried the state half under sixteen days of rounds.

Why the snapshot is a whole-file copy and not an excerpt: a `path:NNN` line cite into the ledger resolves
by line number, and `progress/` is full of them (§2.3 of the archive review counted 101 F1-form cites at
census time). Copying the file byte-for-byte means every one of those coordinates still lands on the line
it landed on. Excerpting would have quietly moved them all.

## B. What moved out, and what stayed

**Moved into the snapshot** — everything from line 340 of the old file onward: the unnumbered
`--attn-touch` section and §7 through §60. That is the GEMM algo flip and its algo pins, the F32-seam
deletion, the gather kernel, the packed-cast deletion, the conv coalesced read and the three admission
corrections, the lane-ceiling change, the `rec2c`/`rec2c4` screens, the llama.cpp comparisons and the
launch-structure refutation, the WMMA attention rounds and their NaN-canary correction, the residency
ladder, the `origin/master` family regression, long context, G1/G2/G4, the per-cost attributions, A9–A13
and the default flip.

**Rewritten, not moved** — §1–§6 kept their subjects (the numbering is a compatibility requirement: live
code, `engine/gate.py` and `AGENTS.md` cite §6 by number), but their tables only stayed if they are still
the newest published number, and four of them were not:

- **§1** now carries the warm headline §60 published (11,636 / 12,047 / **11,070** / 8,542 / 6,154 tok/s at
  512 / 1,024 / 4,096 / 16,384 / 32,000 lanes) instead of the 2026-10-16 prefill table, which three shipped
  rounds had overtaken. The decode marginal-cost rows stayed and are **labelled stale**: they are the
  2026-10-19 window and predate G4, G1 and A13. That is a schedule fact stated as such, not a number
  presented as current.
- **§2 and §3** kept the parts that are method rather than a measurement — the gages are exhaustive, a
  burst is not a step, the exposed gap is the whole launch bill, do not add medians across runs — and
  replaced the attribution tables with a pointer to the snapshot plus "re-take owed, here is the command".
  §60 said the quiet part: the phase table "records an attention row (1,518.91 ms/pass at 16k) that the
  engine no longer produces".
- **§4** keeps a floor row only for a family that has not been rewritten since the row was taken. The GDN
  recurrence (G4) and the attention wave (WMMA, G1, A13) have been rewritten, so their floors are not in
  the live table; they are in the snapshot at §33/§44/§48/§59.
- **§5**'s shipped-default list is re-derived from `engine/decode_run.cpp`'s `Options` rather than copied
  from the old banner, because the banner had gone stale in a way nobody noticed: it listed
  `--gemm lt --gemm-pins all --attn-tile 1 --mlp-map auto --gdn-scan seq --gdn-spell fused --gdn-loop 512
  kRowsPerBlock = 32` and named none of `--attn-fattn`, `--attn-fattn-qstage`, `--attn-fattn-wide 2`,
  `--attn-touch`, `--gdn-geom reg16` or the 32,768-lane ceiling. Reading the struct instead of the banner
  is the same lesson as every other "a census that reads the code but not the preprocessor" row in
  `AGENTS.md`.
- **§6** is a protocol, not a measurement, so it stayed almost verbatim and gained three rules the rounds
  added since the last prune: the ±3 % paired-window noise floor (§37), warm-vs-cold as part of the number
  (§50, §51), and the behavioural-canary requirement for an A/B harness.

**No measurement was taken for this pass.** Every number in the new live ledger was already published in
the snapshot or earlier, with its window and spread; nothing here is a fresh measurement, and the file says
which rows are owed one.

## C. How the citations were re-pointed, and what that proves

`tools/archive_doc.py --split bench/performance-status.md --live engine/performance-status.md --archive
progress/2026-10-21/performance-status-2026-10-21.md --archive-from 7`, over every tracked file except
`engines/`, `tmp/`, `.venv/`, `data/`, `models/`, `logs/`. The routing rule:

| cite form | goes to | why |
|---|---|---|
| `… §1`–`§6` | `engine/performance-status.md` | those headings exist in the live file, so `doc_ref_check.py`'s BROKEN-LABEL stays clean |
| `… §7` and above | the snapshot | same reason, on the other side |
| `§N of \`…\`` | routed by that `N`, whichever side it precedes | the repo spells the label before the path often enough that ignoring it would mis-route |
| `path:NNN`, `:NNN-NNN`, `:NNN,NNN` | the snapshot | the snapshot's line numbers are the old ones, so a line cite survives the move unchanged |
| bare path | the live ledger | it means "the ledger", and the live ledger carries a §-index into the snapshot |

**Counts.** 439 path-form citations existed; 244 routed to the live ledger, 195 to the snapshot, in 157
files — 138 of those occurrences were in non-`.md` files (kernel and gate comments that cite the ledger as
the reason a knob exists). A file's own line count was asserted unchanged for every one of them, which is
`progress/README.md` rule 2's requirement for a same-line substitution.

**Verification, and the one number that matters most here.** `tools/citation_census.py --check` derives
the six census forms over the moved tree and still matches the 2026-10-16 census exactly on all six
(F1 101, F2 171, F4 16, F5 24, F5b 95, F6 66) — that is the witness that the move neither lost nor
duplicated a citation of any counted form. Getting there needed a fix: the census reads each tracked file's
census-era text from the commit that carried the census, keyed by path, so a renamed file read as empty and
its citations silently left the population — 160 counted against an expected 171, a PASS-shaped
undercount. `RENAMES_AT_CENSUS` maps each moved document to the path it had then. `--selftest` still
passes 39 checks (14 of them mutation cases that must FAIL).

The rest of the checker inventory, before and after: BROKEN-ANCHOR 17 → 17, BROKEN-LABEL 20 → 20,
LABEL-UNNAMED 89 → 89, and MISSING-STALE 54 → 67. The 13 new entries are the 13 paths the moved
documents used to have, and each one is cited deliberately from three places: the footer on the moved
document naming where it came from, `progress/README.md`'s crosswalk row, and
`tools/citation_census.py`'s rename map. They are the same kind of row those crosswalk tables have
always contained, and a checker whose MISSING-STALE could not contain an intentional historical path
would have to lose the useful half.

**`README.md` was edited line-count-neutrally, and had to be.** It is cited by line from the frozen
journals (`README.md:47`, `:167`, `:246`, `:313`, `:323`, and four more), so the headline correction that
belongs to this pass — 3,443 → 12,047 warm prompt tok/s at 1,024 lanes, and the pointer to both frozen
eras — had to be written back into exactly the four lines it replaced. The guard for touching a cited
file is `wc -l` before and after, which is the `AGENTS.md` line-citation rule applied to the file this
pass had to correct; `tools/citation_rot_check.py` cannot see the difference, because its own baseline
moved with the commit.
`tools/archive_doc.py --selftest` passes 12 cases; its fixtures assemble their fake paths at runtime so
the checkers do not mistake a fixture for a citation.

## D. What this pass cannot see

- **A `§` with no file is still ambiguous.** 16 pathless `ledger §N` citations exist (the census's F4 form),
  all in frozen documents, and the prune cannot make them name a file without editing journals beyond a
  path. Where an archived document says "ledger §37" it means §37 of the snapshot, because that is what it
  was looking at when it was written; the live ledger's §-index at the top says so in one line.
- **The live ledger's re-take-owed rows are still owed.** §1's decode table and every in-situ attribution
  predate G4/G1/A13. Naming them stale is not the same as re-taking them, and the next device window should
  close them rather than re-take the banner alone.
- **Byte-identity of the frozen part is now a claim about the copy, not about `HEAD`.** One thing is
  recorded against it: the snapshot taken today contains citations whose *paths* were re-pointed earlier in
  this same cleanup pass (the gate audit and build queue had already moved), which is rule 2's permitted
  edit class. The earlier frozen snapshot,
  `progress/2026-10-16/performance-status-2026-10-16.md`, received exactly one such same-line path
  substitution in this pass and nothing else.
- **A move is not a correction.** Nothing in the snapshot was fixed while moving it, including the numbers
  later rounds overturned and §1's stale banner. Those are the record of what was believed when, and the
  corrections live where they were originally written.
