# Pass 4 archive review — what is live data in the main files, and what is the record

Written 2026-10-16. CPU-only: no device command, no build, no gate, no timing, no `gpu_lock.sh`.
It creates **one file** (this one) and modifies nothing; the parent commits. It **builds on**
`progress/2026-10-16/structure-proposal-2026-10-16.md` (§2d priced the journal split in outline)
and `progress/2026-10-16/cleanup-audit-2026-10-16.md` (§4 the stale-reference class) and does not
repeat them. Where a number differs from the structure proposal, this file's number was counted
**today**, after `progress/` existed and after the commit-message rewrite; the proposal counted at
`0f8b81e`, before both. The differences are named in §2.5.

**Headline.** The journal is 4,493 lines in 86 fence-aware sections. Across every tracked file there
are **101 `path:line` citations** into the main docs, and **only 5 come from the live tree** — 2 of
those are a checker's own selftest strings, so **3 real live line-cites exist in the whole repo**.
The other **96 sit inside `progress/`, the one tree that `progress/README.md` rule 2 forbids
editing.** That asymmetry, not the file's length, decides the split: moving any section the archive
points at by line number creates a broken cite the standing rule does not license fixing.
Recommended: **option (a′)** — freeze the journal whole; write the extraction into **new**
`progress/` documents that cite the lines they came from; never move a line.

Conventions: *line-form* = `path:NNNN` / `:NNNN-NNNN` / `:NNNN,NNNN`; *section-sign form* =
`` `path` §N ``; *bare form* = the path or filename with no anchor. All counts are **occurrences**
counted with `grep -o` over `git ls-files`, **code files included** (§2.1 says why). N abbreviates
`engine/results.md`.

## 1. Section census of `engine/results.md`

`wc -l` = 4,493. `grep -c '^# '` = 86, but 72 of those `# ` lines are shell comments inside fenced
blocks (`# cmd: .venv/bin/python engine/gate.py --fixture chunk64b`), so any splitter must be
fence-aware — re-verified independently here: fence-aware **H1 = 14, H2 = 72, H3 = 33**. Rows below
are **non-overlapping and contiguous**: each is that heading's text up to the next heading of any
level, so the line and cite columns add up.

Class key — **LIVE-DATA**: a measurement, band, contract or verdict another file or the gate
consumes. **NARRATIVE**: chronicle / decision-log prose / "what this pass ran" / "left open".
**SUPERSEDED**: the text itself says it was retired, refuted, corrected or replaced.

| lines | n | lvl | class | section (abbrev) | cites¹ |
|---|---:|---|---|---|---:|
| 1-49 | 49 | H1 | LIVE-DATA | `engine/` weight residency (W5 first half) | 0 |
| 50-60 | 11 | H2 | LIVE-DATA | What is resident | 0 |
| 61-80 | 20 | H2 | LIVE-DATA | Phase table, best of 3 gated runs | 0 |
| 81-100 | 20 | H2 | NARRATIVE | The D2H discrepancy, 27.7 vs 63-70 GB/s | 0 |
| 101-108 | 8 | H2 | NARRATIVE | What is deliberately not measured here | 1 |
| 109-113 | 5 | H1 | SUPERSEDED | W5b on disk: state + launcher, "proven, deferred, open" | 0 |
| 114-129 | 16 | H2 | LIVE-DATA | Proven on CPU (+ one launcher on device) | 0 |
| 130-155 | 26 | H2 | NARRATIVE | Deferred, with the price named | 0 |
| 156-184 | 29 | H2 | NARRATIVE | Open questions this phase did not decide | 0 |
| 185-249 | 65 | H1 | SUPERSEDED | W5b token loop "compiled, none of it run" | 0 |
| 250-298 | 49 | H1 | SUPERSEDED | Two device runs in; six failed checks, one cause | 2 |
| 299-335 | 37 | H2 | SUPERSEDED | The first ms/step figure, "not a measurement" | 0 |
| 336-390 | 55 | H2 | SUPERSEDED | The three flagged risks, checked statically | 0 |
| 391-398 | 8 | H1 | LIVE-DATA | First `ENGINE GATE PASS`, 93 checks, 11.978 ms | 0 |
| 399-500 | 102 | H2 | LIVE-DATA | (1) What the pass established | 2 |
| 501-1057 | 557 | H2 | LIVE-DATA | (2) First throughput row; the ~4 ms gap (H3s below) | 9 |
| 1058-1066 | 9 | H1 | LIVE-DATA | P1: the gate 64 steps deep, teacher-forced | 0 |
| 1067-1118 | 52 | H2 | LIVE-DATA | The `dec64f` fixture, 60 MB not 1.7 GB | 0 |
| 1119-1138 | 20 | H2 | LIVE-DATA | Why the deep gate replays | 0 |
| 1139-1185 | 47 | H2 | LIVE-DATA | The curve is flat in depth | 2 |
| 1186-1200 | 15 | H2 | LIVE-DATA | The 300-token free run needs `--no-stop` | 0 |
| 1201-1223 | 23 | H2 | SUPERSEDED | What a worker may not decide, left open on purpose | 0 |
| 1224-1277 | 54 | H1 | LIVE-DATA | P2: `engine/chat.sh`, the engine's REPL | 0 |
| 1278-1306 | 29 | H1 | LIVE-DATA | Decision 2: the 27-34 excursion priced | 3 |
| 1307-1380 | 74 | H2 | LIVE-DATA | The table, all 64 forced steps | 0 |
| 1381-1411 | 31 | H2 | LIVE-DATA | What the table says | 0 |
| 1412-1430 | 19 | H2 | NARRATIVE | What this does not settle | 0 |
| 1431-1458 | 28 | H2 | SUPERSEDED | The proposal, band untouched, "decision belongs to the operator" | 0 |
| 1459-1471 | 13 | H1 | LIVE-DATA | Decision 1: the tie policy is implemented | 0 |
| 1472-1487 | 16 | H2 | LIVE-DATA | The measured ties | 0 |
| 1488-1522 | 35 | H2 | LIVE-DATA | The predicate, and why each clause is that shape | 0 |
| 1523-1536 | 14 | H2 | LIVE-DATA | Where the deep gate stands after it | 0 |
| 1537-1543 | 7 | H1 | LIVE-DATA | `dec64t`: the eos confound, measured | 0 |
| 1544-1565 | 22 | H2 | LIVE-DATA | Pre-flight: `tools/stop_probe.py` | 0 |
| 1566-1595 | 30 | H2 | LIVE-DATA | The fixture and the two comparisons | 4 |
| 1596-1613 | 18 | H2 | LIVE-DATA | The free run, the number never available before | 0 |
| 1614-1649 | 36 | H2 | LIVE-DATA | What this settles about 27-34 | 0 |
| 1650-1677 | 28 | H1 | SUPERSEDED | Q1: band option 1 in force (banner at `:1652`) | 0 |
| 1678-1711 | 34 | H2 | SUPERSEDED | The derivation, re-derived from shipped files | 0 |
| 1712-1739 | 28 | H2 | SUPERSEDED | Why this is a sharpening not a loosening | 0 |
| 1740-1766 | 27 | H2 | SUPERSEDED | The tripwire "retired by D-2, kept here" | 0 |
| 1767-1778 | 12 | H2 | NARRATIVE | What was not chosen, and why it is on the record | 0 |
| 1779-1796 | 18 | H2 | NARRATIVE | What the pass does not buy | 0 |
| 1797-1814 | 18 | H1 | LIVE-DATA | Q2: engine G7 at breadth, 423 lanes | 0 |
| 1815-1843 | 29 | H2 | NARRATIVE | The pre-flight, and making them | 0 |
| 1844-1863 | 20 | H2 | LIVE-DATA | `dec300f`: what 423 lanes changed | 0 |
| 1864-1895 | 32 | H2 | SUPERSEDED | The tripwire fired on `dec300f`, nothing priced | 0 |
| 1896-1937 | 42 | H2 | LIVE-DATA | `mt32f`: the two-turn pin | 0 |
| 1938-1962 | 25 | H2 | LIVE-DATA | The free run did not diverge in 128 tokens | 0 |
| 1963-2004 | 42 | H2 | SUPERSEDED | What this pass left open, named | 2 |
| 2005-2012 | 8 | H1 | LIVE-DATA | D-1 and D-2: tie policy, per-fixture margin | 0 |
| 2013-2078 | 66 | H2 | LIVE-DATA | D-1 — a same-id step is not a tie case | 0 |
| 2079-2134 | 56 | H2 | LIVE-DATA | D-2 — the route margin is keyed per fixture | 4 |
| 2135-2153 | 19 | H2 | LIVE-DATA | What the gate prints now, per step | 0 |
| 2154-2184 | 31 | H2 | NARRATIVE | The pass, and what a worker ran | 5 |
| 2185-2203 | 19 | H2 | NARRATIVE | What this pass left open, named | 2 |
| 2204-2225 | 22 | H1 | LIVE-DATA | D-4: the engine free-runs the corpus | 0 |
| 2226-2260 | 35 | H2 | LIVE-DATA | The numbers, against the chaos floor | 0 |
| 2261-2285 | 25 | H2 | LIVE-DATA | Every first flip is an undecided reference step | 0 |
| 2286-2315 | 30 | H2 | LIVE-DATA | What this licenses, and what it does not | 0 |
| 2316-2339 | 24 | H2 | NARRATIVE | What this pass ran, and how to extend it | 0 |
| 2340-2356 | 17 | H2 | NARRATIVE | What this left open, named | 0 |
| 2357-2369 | 13 | H1 | LIVE-DATA | D-3 — the margin cannot be engine-independent | 0 |
| 2370-2404 | 35 | H2 | LIVE-DATA | The third legal route, two constraints | 0 |
| 2405-2436 | 32 | H2 | LIVE-DATA | The instrument (`route_margin.py`) | 0 |
| 2437-2462 | 26 | H2 | LIVE-DATA | The numbers | 0 |
| 2463-2492 | 30 | H2 | LIVE-DATA | meanrel: the shipped constant mixes two statistics | 0 |
| 2493-2526 | 34 | H2 | LIVE-DATA | Why the route spread cannot price the deficit | 0 |
| 2527-2548 | 22 | H2 | LIVE-DATA | Recommendation (adopted: margins unchanged) | 0 |
| 2549-2561 | 13 | H2 | NARRATIVE | What this pass ran, what was left for device | 4 |
| 2562-2621 | 60 | H2 | SUPERSEDED | The handover: sequence "if a band move is ever decided" | 0 |
| 2622-2662 | 41 | H2 | LIVE-DATA | Q5/P1 — the §08 state is observable | 0 |
| 2663-2703 | 41 | H2 | LIVE-DATA | Q5/P2 — the prompt is now a wave | 0 |
| 2704-2827 | 124 | H2 | LIVE-DATA | Q5/P4+P4b — batched kernels proven | 0 |
| 2828-2919 | 92 | H2 | LIVE-DATA | Q5/P5 — nine projections batch over the wave | 0 |
| 2920-3024 | 105 | H2 | LIVE-DATA | Q5/P6 — the GDN mixer batches | 0 |
| 3025-3169 | 145 | H2 | LIVE-DATA | Q5/P6b — D-7 refuted, prefill row was pinned | 1 |
| 3170-3170 | 1 | H2 | LIVE-DATA | (title continuation of P7) | 0 |
| 3171-3306 | 136 | H2 | LIVE-DATA | Q5/P7 — the `MAXSC` wall is gone | 4 |
| 3307-3307 | 1 | H2 | LIVE-DATA | (title continuation of P8) | 0 |
| 3308-3490 | 183 | H2 | LIVE-DATA | Q5/P8 — attention leaves the token loop | 0 |
| 3491-3674 | 184 | H2 | LIVE-DATA | Q7 — the ~3.2 ms residue is a fresh window | 9 |
| 3675-3891 | 217 | H2 | LIVE-DATA | Q10 — `--mlp-map`, bit-identical on 8 fixtures | 1 |
| 3892-4058 | 167 | H2 | LIVE-DATA | Q10b — the walk follows the row count | 2 |
| 4059-4315 | 257 | H2 | LIVE-DATA | Q11 — D-9's near-tie becomes a sized rule | 10 |
| 4316-4494 | 179 | H2 | LIVE-DATA | Route (c) — hipBLAS for the ten fat prefill matmuls | 10 |

¹ line-form citations pointing at a line inside that section (comma lists counted per number).

**Totals: 59 LIVE-DATA (3,716 lines), 13 NARRATIVE (265), 14 SUPERSEDED (513).** 69 of the 86
sections carry **no** line-form citation at all; 65 line-number references land on 17 sections —
and **14 of those 65 land on sections classed NARRATIVE or SUPERSEDED** (`:101` ×1, `:250` ×2,
`:1963` ×2, `:2154` ×5, `:2549` ×4), i.e. the material a tidy split would move is exactly the
material the frozen archive points into. Three H3 blocks add to the movable set — `565-601` "Levers,
ranked by expected ms/token" (NARRATIVE), `712-769` "Commands for the main agent, in the order that
changes strategy" and `1048-1057` "Context, not action: there is no prefill" (both SUPERSEDED:
`engine/prefill.cpp` and `engine/prefill.h` exist) — **+105 lines, so 883 of 4,493 = 20 % is
chronicle or superseded.** (The 86-row table above and the totals are a mechanical join: the fence-
aware dump below for the ranges, the §2.3 anchor list for the cites.)

Command for the section list (no tool does this today; this is the exact script used, and the one
§5 S1 would add to `tools/`):

```bash
python3 - <<'PY'   # fence-aware heading census: level, start-end, line count, title
import re,sys
lines=open(sys.argv[1]).read().split('\n'); fence=False; h=[]
for i,l in enumerate(lines,1):
    if l.lstrip().startswith('```'): fence = not fence; continue
    if fence: continue
    m=re.match(r'^(#{1,3})\s+(.*)',l)
    if m: h.append((i,len(m.group(1)),m.group(2)))
for k,(i,lv,t) in enumerate(h):
    end=h[k+1][0]-1 if k+1<len(h) else len(lines)
    print(i,end,end-i+1,'#'*lv,t,sep='\t')
PY
grep -c '^# ' engine/results.md    # 86 raw — NOT the section count (72 are fenced shell comments)
```

**Evidence for every NARRATIVE / SUPERSEDED call** (line numbers verified with `grep -n`; the class
of a block whose *heading* is itself the claim is quoted from the heading):

| class | lines | the line that makes it so |
|---|---|---|
| SUPERSEDED | 109-113 | `:111` "Everything in this section is **compiled, never run on a device**." — disproved from `:250` |
| SUPERSEDED | 185-249 | H1 at `:185` "…`decode_step` + `decode_run` + `gate.py` — compiled, none of it run" |
| SUPERSEDED | 250-298 | H1 at `:250` "six failed checks had one cause — §07.5 armed during the prompt" (fixed in `5e7114b`, named there) |
| SUPERSEDED | 299-335 | `:301` "**Superseded the same day, twice over.**" |
| SUPERSEDED | 336-390 | `:411` "both risks are retired by the ids rather than by reading" |
| SUPERSEDED | 1201-1223 | H2 at `:1201` "What a worker may not decide, left open on purpose" — decided at `:1459` |
| SUPERSEDED | 1431-1458 | H2 at `:1431` "The proposal, with numbers — band untouched, decision belongs to the operator" — decided at `:1650` |
| SUPERSEDED | 1650-1677 | `:1652` "**SUPERSEDED IN PART, same day (D-2).** This section's *rule* stands…" |
| SUPERSEDED | 1678-1711, 1712-1739 | the same `:1652` banner governs the whole Q1 block |
| SUPERSEDED | 1740-1766 | H2 at `:1740` "The tripwire this derivation used to carry — retired by D-2, kept here because it is how D-2 was found" |
| SUPERSEDED | 1864-1895 | `:1997` "**That `mt32f` FAIL and the "nothing was priced" state were both superseded later the same day**"; also `:1811` "**They are superseded two sections down**" |
| SUPERSEDED | 1963-2004 | the `:1997` banner sits inside this block |
| SUPERSEDED | 2562-2621 | H2 at `:2562` "The handover: what this pass could not run, and the sequence if a band move is ever decided" — that band move has since been decided and run |
| SUPERSEDED head | 4316-4494 | `:4318` "**Read the end of this section first.** Everything above it describes the route while it was switched off." The tail `:4393` "Admitted (2026-10-14): the default is `library`" is live, so **the section stays** — only its head is stale |
| NARRATIVE | 81-100 | `:84` "The loader's copy-back measures 18–28 GB/s. **The doc is probably right and my number is a different quantity.**" — an attribution the settle-plan (named in the same block) never ran |
| NARRATIVE | 101-108 | H2 at `:101` "What is deliberately not measured here"; `:103` "Nothing in `prototype/` was called…" |
| NARRATIVE | 130-155, 156-184 | H2s at `:130` "Deferred, with the price named", `:156` "Open questions this phase did not decide" |
| NARRATIVE | 1412-1430, 1767-1778, 1779-1796 | those headings: "What this does not settle", "What was not chosen, and why it is on the record", "What the pass does not buy" |
| NARRATIVE | 1815-1843 | H2 at `:1815` "The pre-flight, and making them" |
| NARRATIVE | 2154-2184, 2185-2203, 2316-2339, 2340-2356, 2549-2561 | those headings: "The pass, and what a worker ran", "What this pass left open, named" (twice), "What this pass ran, and how to extend it", "What this left open, named", "What this pass ran, and what was left for the device pass" |

```bash
grep -n 'Superseded\|SUPERSEDED\|retired by\|retracted\|refuted\|Refuted\|Read the end of this' engine/results.md
grep -n '^## What\|^## The pass\|^## The handover\|^## Deferred\|^## Open questions' engine/results.md
```

**What cites N, all forms, by citing file** (commands in §2):
`progress/2026-10-16/cleanup-audit-2026-10-16.md` 26 line-form; `progress/README.md` 19;
`progress/2026-10-16/structure-proposal-2026-10-16.md` 4;
`progress/2026-10-11/commit-review-2026-10-11.md` 1; `tools/doc_ref_check.py` 1 (a selftest fixture);
`AGENTS.md` 2 title-cites (`:64` "`engine/results.md` (Q2 section, …)", `:67` "(“Q5 / P8”)");
`engine/performance-status.md` 5 (incl. `:716` "§Q6.1"); `engine/kernels.h` 2 (`:171`,`:176` "§Q10b");
`engine/decode_run.cpp` 1 (`:513` "§Q10b"); plus **10 bare in-file `§Qn` self-references**
(`§Q10`×5, `§Q9`×2, `§Q6.1`, `§Q7`, `§Q5`) and 4 "section above/below" prose pointers.

## 2. Citation census — exhaustive and reproducible

### 2.1 The citation forms in use (measured, one command per count)

| # | form | real example | occurrences |
|---|---|---|---:|
| F1 | `path:NNNN`, `:NNNN-NNNN`, `:NNNN,NNNN` | `engine/results.md:3739`, `engine/results.md:724,726` | **101** |
| F2 | `` `path` §label `` / `path §N` | `` `engine/performance-status.md` §6 ``, `` `engine/results.md` §Q6.1 `` | **171** |
| F3 | bare path or bare filename, no anchor | `` `engine/results.md` ``, `` `results.md` `` | §2.4 |
| F4 | nickname + section | `ledger §6`, `ledger §5(g)` | 16 (5 live, 11 in `progress/`) |
| F5 | path + parenthetical qualifier, usually a section **title** | `` `engine/results.md` (“Q5 / P8”) ``, `` `engine/performance-status.md` ("Headline tables after the tiled attention wave ships") ``, `` `engine/results.md` ("Decision 2 of the deep gate") `` | 24 |
| F5b | letter-labelled section sign, no digits | `§Q6.1`, `§A2`, `§Q9 Results`, `§"the two real limits"`, `§expectation 6` | 95 (42 `§Q`, 33 `§A`) |
| F6 | label with **no file named on the line** | `engine/gate.py`'s 17 `§A1…§A4`, which live in `progress/2026-10-14/route-c-audit-2026-10-14.md:79,144,204,265` | 66 label-only |
| F7 | bare `§NN.N` into `specification/` | `§04.4`, `§10.5` — thousands; not adjudicable, unaffected by any move here | — |

Two things this exposes that the structure proposal's count could not: **(i) the scanner must include
code files**, because live C++/Python carry citations (`engine/kernels.h` §Q10b,
`prototype/gdn_gated_norm/run.py:107` "the whole of results.md §8"), and `tools/doc_ref_check.py`
reads `git ls-files '*.md'` only; **(ii) F2/F5/F5b/F6 are label-and-title addressing, which survives a
move that F1 cannot survive** — that asymmetry is what makes option (b) more expensive and option
(c) cheaper than they look, and it is why (b) would break silently rather than loudly.

```bash
T='(engine/results\.md|prototype/[^/ ]+/results\.md|bench/performance-status\.md|devices/[A-Za-z0-9-]+\.md|specification/[A-Za-z0-9.-]+\.md|engine/README\.md|AGENTS\.md|README\.md|results\.md|performance-status\.md)'
git ls-files -z | xargs -0 grep -ohE "$T:[0-9]"  2>/dev/null | wc -l   # F1 = 101
git ls-files -z | xargs -0 grep -ohE "$T\`? ?§" 2>/dev/null | wc -l   # F2 = 171
git ls-files -z | xargs -0 grep -ohE 'ledger §[0-9]' 2>/dev/null | wc -l  # F4 = 16
git ls-files -z | xargs -0 grep -ohE "$T\`?,? ?\(" 2>/dev/null | wc -l    # F5 = 23
git ls-files -z | xargs -0 grep -ohE '§[A-Za-z]' 2>/dev/null | wc -l     # F5b = 95
```

### 2.2 F1 by target — the number a split destroys

| target | F1 | which files hold them |
|---|---:|---|
| `engine/results.md` | **51** | cleanup-audit 26, `progress/README.md` 19, structure-proposal 4, commit-review 1, `tools/doc_ref_check.py` 1 |
| bare `results.md` | 12 | commit-review 9, cleanup-audit 2, `engine/kernels_glue.cpp` 1 |
| `engine/README.md` | 8 | commit-review 4, cleanup-audit 3, structure-proposal 1 |
| `engine/performance-status.md` | 5 | cleanup-audit 3, `progress/README.md` 2 |
| `devices/strix-halo-mixed-traffic.md` | 5 | `progress/README.md` 3, cleanup-audit 2 |
| `prototype/attn_decode/results.md` | 4 | cleanup-audit 3, `progress/README.md` 1 |
| `prototype/gdn_gated_norm/results.md` | 2 | cleanup-audit 2 |
| `prototype/glue/results.md` | 1 | `engine/results.md:21` |
| `prototype/proj_gemv/results.md` | 1 | commit-review 1 |
| `specification/*` (5 files) | 7 | structure-proposal 3, cleanup-audit 2, commit-review 1, `prototype/attn_decode/run.py:96` 1 |
| `AGENTS.md` + `README.md` | 3 | cleanup-audit 3 |
| **total** | **101** | |

```bash
git ls-files -z | xargs -0 grep -ohE "$T:[0-9]" 2>/dev/null | sed 's/:.$//' | sort | uniq -c | sort -rn
```

### 2.3 F1 by *origin* — the safety finding

**96 of the 101 line-form citations live inside `progress/`.** Only **5 originate in the live tree,
and 2 of those are the checker's own `--selftest` fixture strings**, leaving **3 real live
line-cites in the entire repository**:

| origin | target | resolves to |
|---|---|---|
| `engine/kernels_glue.cpp:35` | `results.md:113` | `prototype/glue/results.md` §3 — filename only, directory inferred |
| `engine/results.md:21` | `prototype/glue/results.md:170` | glue §4 |
| `prototype/attn_decode/run.py:96` | `specification/10-reference-dumps-and-gates.md:40` | spec §10 G6 |
| `tools/doc_ref_check.py:284,285` | `engine/results.md:3739`, `specification/05-gated-full-attention.md:12` | scanner inputs, not pointers |

A 6th appears if the pattern tolerates a space after the colon —
`prototype/attn_decode/wave_model.py:51` → `devices/strix-halo-compute.md: 24` — so the tolerant
counts are 102 / 6. **State which spelling a count used; 101/102 is not a disagreement.**

**Zero live-tree line-cites point into `engine/results.md`.** That single row is the whole safety
argument, and it inverts the intuitive reading: the journal is *safest* to leave and *most dangerous*
to renumber, because every pointer to its line numbers sits in the tree that rule 2 of
`progress/README.md` freezes — "Nothing in `progress/` is ever edited afterwards, except to fix a
broken path … it must be a same-line substitution, so that `file:NNNN` cites in the journals keep
pointing at the right line." **A split moves no path but moves every line number: the one edit the
archive rule does not license.**

```bash
git ls-files -z | grep -zv '^progress/' | xargs -0 grep -nHE "$T:[0-9]" 2>/dev/null
```

### 2.4 F2/F3 by target — what a renumber destroys

| target | F2 §-form | bare F3 | note |
|---|---:|---:|---|
| `engine/performance-status.md` | 65 | 136 | the most section-addressed doc in the repo |
| bare `results.md` | 41 | 114 | **directory-less**; resolved by the citing file's neighbourhood |
| `prototype/attn_decode/results.md` | 19 | 27 | `AGENTS.md` holds 9 of the journal §-form cites |
| `prototype/gdn_gated_norm/results.md` | 11 | 14 | §1 and §7-§9 cited from `engine/README.md:138,314` |
| `prototype/gdn_collapse/results.md` | 6 | 10 | DEAD route, still section-addressed |
| `engine/results.md` | 6 | 180 | the 6 are `§Q6.1`/`§Q10b` label cites from live code + ledger |
| `devices/strix-halo-mixed-traffic.md` | 6 | 22 | incl. `§Q6.1`, `§Q9 Results`, `§expectation 6` |
| `prototype/mlp_gemv/results.md` | 3 | 46 | the 46 bare are AGENTS.md's F1-F4 trap rows |
| `performance-status.md` (bare) | 5 | 10 | commit-review's nickname-era cites |
| `specification/*` (6 files) | 8 | ~25 | `§10`, `§09`, `§07` label cites |
| `devices/strix-halo-memory-bandwidth.md` | 1 | 18 | — |

Two rot hazards visible only here. **(a) Bare `results.md` is 41+12 anchors with no directory:**
`engine/gate.py:6381` "the default since results.md §21" and `engine/kernels_attn.cpp:330`
"(results.md §20 …)" both mean `prototype/attn_decode/results.md`; `engine/kernels_glue.cpp:35`
means `prototype/glue/results.md`. Nothing resolves this form mechanically today. **(b) Live code
cites archive labels with the file unnamed:** `engine/gate.py` uses `§A1`/`§A2`/`§A3`/`§A4` **17
times** (`:1404,1434,1441,1450,1629,2642,2883,3351,3630,3631,3679,3898,3916,5122,5694,5769,6064`)
and those headings are `## A1 —` … `## A4 —` at
`progress/2026-10-14/route-c-audit-2026-10-14.md:79,144,204,265`. The shipping gate therefore has 17
undocumented pointers into a document the archive rule forbids editing. That is a defect of the same
class as a stale path, and it is invisible to every checker in the tree.

```bash
git ls-files -z | xargs -0 grep -ohE "$T\`? ?§" 2>/dev/null | sed 's/[ `]*§$//' | sort | uniq -c | sort -rn
grep -c '§A[1-4]' engine/gate.py                                  # 17
grep -n '^## A[1-4] ' progress/2026-10-14/route-c-audit-2026-10-14.md   # 79, 144, 204, 265
```

### 2.5 Reconciliation with the structure proposal, plus two id-space holes

The proposal counted 27 line-cites into `engine/results.md`; today it is 51. The delta is **+19 from
`progress/README.md`'s crosswalk table and +4 from the proposal itself becoming tracked** — the
count grows every time the tree documents a move, which is itself an argument against adding one.
Its "36 `results.md §N`" is 41 here, for the same reason (it scanned `*.md` at an earlier commit).

`specification/ASSUMPTIONS.md` carries **A-01…A-18** and no more. Live code cites two ids that are
not defined there: `A-19` (`prototype/gdn_chunk/gdn_chunk_kernels.h:133`) and `A-32`
(`engine/gate.py:3873`, and reproduced verbatim in the committed transcript
`progress/engine/evidence/margin/library_dec300f.log:1362`). Both are `doc_ref_check`-invisible —
ids, not paths — so they belong with F6 in the new check (§5 S7). `A-25` in raw greps is a
`SHA-256` substring, not an assumption id; do not "fix" it.

```bash
grep -c '^| \*\*A-' specification/ASSUMPTIONS.md                       # 18
git ls-files | xargs grep -oh 'A-[0-9][0-9]' 2>/dev/null | sort -u      # …A-18 A-19 A-25 A-32
```

## 3. The other main-tree docs: three-row census and verdict

Sized proportionally — each doc's largest blocks per class, then one verdict. Section ranges come
from the same fence-aware dump as §1 (run that script over each file in `prototype/*/results.md`,
`bench/*.md`, `devices/*.md`, `specification/*.md`, `engine/README.md`); class evidence is the quoted
line.

| doc (lines) | LIVE-DATA | NARRATIVE | SUPERSEDED | verdict |
|---|---|---|---|---|
| `prototype/attn_decode/results.md` (1,509) | all 21 numbered H2s. §12,14,16,18,20 are CPU worker passes, §13,15,17,19,21 the device passes; each is the only record of a gated result and **19 §-cites address them by number** | none separable — every §n is either cited or is the witness for a gate that cites it | §13 `:665` "**RED — the flip is refused**" and §15 `:875` "one still blocking" are answered by §19/§21, **but they are the anti-vacuity witness** for those results | **KEEP AS IS.** The numbers are a citation namespace: `AGENTS.md` cites §18, §18.2, §20.1-§20.3, §20.6. Renumbering would falsify AGENTS.md's own traps table |
| `engine/performance-status.md` (1,149) | §6 `244-1072` (829, the ledger proper) + "Headline tables" `1073-1149` (78) = **907 live** | §3 projection `139-172` (34) | §1-§4 `25-184` (160) + §5 `185-243` (59). `:187` "**Done, 2026-10-12.**" marks §5; `:188-189` says of §1 "**§6 supersedes them wherever it has a row**" | **PRUNE** ~220 lines (§1-§5) → `progress/2026-10-16/ledger-pre-clean-window-2026-10-16.md`. Cost: **5 F1 cites, all in `progress/`**; all 65 §-cites point at §6, which does not move. Do it **after** the line-anchor checker (§5 S4) lands, never before |
| `prototype/mlp_gemv/results.md` (534) | 9 of 12 H2s: gate F1-F4, the cancellation-band width, `floor_probe.py`, the SwiGLU policy, the negatives, Performance, "What graduates", P4, Q10 | "Traps found while building this" `347-377` (31), already harvested into AGENTS.md | none self-declared | **KEEP AS IS.** The only PRUNE candidate is 31 lines against 46 bare cites from AGENTS.md into this file |
| `prototype/gdn_gated_norm/results.md` (517) | §1-§5, §7-§10; §8 `363-446` (the fixture-writer NaN) and §9 `447-484` (the PASS) are load-bearing; `engine/README.md:138,314` cites §1 and §7-§9 | §6 "Not done" `287-295` (9) | H1 `:1` still asserts "**the first device run failed and the cause is not established**" — §8 establishes it | **KEEP AS IS** + one same-line title correction (S6) |
| `prototype/proj_gemv/results.md` (313) | 8 of 10 H2s | "Traps found here (added to the index)" `297-314` (18) | "Performance — **provisional**" `161-228` (68), overtaken by ledger §6 and engine Q6 | **KEEP AS IS** — the `provisional` label already does the supersession work |
| `prototype/glue/results.md` (286) | §0-§6, §8 | §7 "Not done / open" `224-232` (9) | "Timings — **provisional**" `126-171` (46) | **KEEP AS IS.** Its line 113 is 1 of the 3 live line-cites in the repo — glue is a live dependency of `engine/kernels_glue.cpp` |
| `prototype/gdn_collapse/results.md` (220) | §1-§7: the DEAD verdict, bit-identity §3, the 9.8× mispricing §5 | §0 "What happened, in order" `14-28` (15) | route is DEAD (H1: "112.0 ms against a 63 ms win line and an 84 ms kill line") while `engine/gdn_collapse_kernels.h` is still `#include`d by `engine/kernels_gdn.cpp` | **KEEP AS IS** + one `engine/README.md` line saying which side is live (structure proposal §2f already asked) |
| `prototype/gdn_decode` (163), `prototype/lm_head_gemv` (125) | every substantive section | small "Shape of the kernel, for the next one" notes | — | **KEEP AS IS** — too small to split, §-addressed |
| `engine/README.md` (496) | `loader` `15-39`, `state`/`kernels.h` `51-77`, `decode_step` `78-97`, and the CPU command list inside Hand-off `98-459` — that block **is** the run-things-first list, 362 lines | Hand-off preamble `100-107`, dated status prose | three stale blocks: `:11` "**no generation token has come out of this engine yet.**" (the engine now free-runs the corpus, N `:2204`); `:42` "No `prefill.{h,cpp}`" while `engine/prefill.cpp` **and** `engine/prefill.h` exist; `:47` "no batching" while Q5 batched the whole wave; `460-482` "What the next worker owes W5b", items 1-2 both done | **PRUNE in place, no move**: correct `:11`, §"What this is NOT yet", §"What the next worker owes W5b". **Do not split** — 8 F1 cites, 6 §-cites, and it is the on-ramp the parent hands every worker |
| `devices/strix-halo-mixed-traffic.md` (639) | H1, `32-72` what a token moves, `73-116` the probe, `287-361` Results, `362-393` Consequences, `591-640` Q9 Results = 300 | — | two **pre-registration** blocks: `117-286` (170) "Pre-registered sweep, and what is expected — written before the run", `394-590` (197) "Q9 … pre-registered 2026-10-13, before any Q9 device run" | **KEEP AS IS.** 367 lines of pre-registration whose results now exist are the only proof the predictions were written blind — the same reason N `:1740` keeps a retired tripwire. Its 5 F1 cites are all in `progress/` |
| `devices/strix-halo-compute.md` (188) | `39-68` prefill-target arithmetic, `108-145` fp16-vs-bf16 | — | `12-38` "The rows" is corrected **twice in place**: `69-107` "Correction (same day): 'above the hardware' was too strong, and the CU count I printed was wrong"; `146-189` "Second correction, same day: every rate above is an **F32-`C`** rate, and it is 6–8× below the same call" | **KEEP AS IS** — correction-in-place is the house convention AGENTS.md records for spec §07.4. Its live cite (`wave_model.py:51`, space-after-colon spelling) is a real dependency |
| `devices/strix-halo-unified-memory.md` (142), `devices/strix-halo-memory-bandwidth.md` (120) | all sections; AGENTS.md's roofline and `hipMallocManaged` rows depend on them | "What this does *not* buy" `132-143` is a scope note, not chronicle | the `[FUTURE]` passages are tagged, which is the AGENTS.md contract | **KEEP AS IS** |
| `specification/*.md` (13 files, 1,716 lines; largest `10-…` 224) | the whole spec — the requirement document `speccheck/` gates against | — | §07.4 already carries its own correction in place | **KEEP AS IS.** No dated chronicle lives here. Only fix: the **A-19 / A-32 id gap** (§2.5) |
| `AGENTS.md` (228), `README.md` (248) | scope table, traps index, layout block | — | the stale `thinking/` mentions were fixed in pass 2 | **KEEP AS IS.** Any move changing a journal path must edit AGENTS.md in the same commit: it holds 9 §-form journal cites and 3 F1 cites |

**Net: exactly two documents are worth pruning** — `engine/performance-status.md` (220 lines of
superseded pre-clean-window rows) and `engine/README.md` (three stale blocks, corrected in place,
no move). Everything else is a live instruction, a measurement something consumes, or
pre-registration prose whose value *is* that it was written before the result.

## 4. The `engine/results.md` volume split, priced

First the constraint that dominates every option. **A line-number reference is destroyed by any
removal above it, not only by a move of the referenced block.** The first NARRATIVE/SUPERSEDED block
begins at `:81`, so any extraction that deletes lines renumbers everything from 81 down and breaks
**64 of the 65** line-number references into N — not the 14 that happen to sit on moved text. And
96 of the 101 line-form cites in the repo, plus these neighbours, live in `progress/`, which
`progress/README.md` rule 2 freezes to same-line path fixes only:

| archive cite into… | lines referenced |
|---|---|
| `engine/results.md` | 51 references, 65 line numbers, `101…4479` |
| `engine/performance-status.md` | `:988`, `:989`, `:1094` — all below §1-§5, so the ledger PRUNE of §3 shifts all three |
| `engine/README.md` | `:61-62`, `:64`, `:83-84`, `:126`, `:138,314`, `:162`, `:182`, `:246` — all below `:40-50`, so the README PRUNE of §3 shifts all of them |

```bash
grep -ohnE 'bench/performance-status\.md:[0-9, -]*' progress/*.md progress/*/*.md | sort -u
grep -ohnE 'engine/README\.md:[0-9, -]*'          progress/*.md progress/*/*.md | sort -u
```

**So pass 4's premise collides with pass 2's own rule.** *Archive the rest* requires deleting lines
from documents the frozen archive points into by line number. Only the parent/user can resolve that,
either by waiving rule 2 for an approved exception (price: re-validate 62 archive references) or by
holding the line, which is what option (a′) does at zero cost. **This is the one decision in this
document that is not mine to make.**

| option | exact work | citation rewrites required (from §2) | what can still rot |
|---|---|---|---|
| **(a) leave it whole** | nothing | **0** | The 4,493-line file keeps growing (~90 lines/phase ⇒ ~5.4 k by November). A reader cannot tell live contracts from retired prose without reading the banners. Discoverability does not improve at all |
| **(a′) leave it whole, extract by quotation — RECOMMENDED** | (i) write **new** `progress/2026-10-16/journal-chronology-2026-10-16.md` collecting the 883 chronicle/superseded lines **by quotation with their `N:NNN` anchors**, grouped by the §1 class table; (ii) append a **5-line pointer at the very end of N** (append-only moves no line) saying which new file holds the chronology and that line numbers below it are frozen; (iii) add the §1 class table to it. Total: 2 new files + 1 append | **0 rewrites.** 0 path cites touched, 0 of the 65 line numbers moved, 0 §-labels moved. N grows by 5 lines at the tail, which is the only direction that is safe | The extraction is a *copy*: it can drift from the journal it quotes. Mitigation: every row carries its `N:NNN` anchor, so drift is checkable by `sed -n 'Np'`. The journal itself is still 4.5 k lines — this option buys legibility, not brevity |
| **(b) split by phase into `engine/results/` files + a pointer at the old path** | fence-aware splitter (86 raw `# ` lines vs 14 real H1s; `###` needed for Q6/Q6.1/Q7 inside `501-1057`); ~20 usable blocks of 13-257 lines; the natural directory name is a trap: **`git check-ignore -v engine/results/vol1.md` → ignored by `.gitignore:69` (`results/`)**, so a non-trapped name is mandatory (`engine/journal-2026-11.md` is clean, measured) | **64 of 65 line numbers break** and 50 of them are in `progress/` (rule-forbidden to fix). 6 §-label cites from live code + ledger need path edits (labels survive, paths do not). 10 in-file `§Qn` self-references and 4 "section above/below" pointers break *silently* — they name no file, so nothing reports them. break *silently* — they name no file, so nothing reports them. 2 F5 title-cites (`AGENTS.md:64,67`) must be re-pointed. A pointer file at the old path makes the 180 bare F3 cites resolve to a file that no longer contains the cited text — **existence-checked tools still pass** | Everything. A split makes the label-addressing (F2/F6) and the bare-filename form (F3) worse than they already are, and does not fix the 64 line refs it created |
| **(c) move the SUPERSEDED §s to `progress/`, leave live data** | 513 lines (14 H2/H1 blocks) + 105 lines (3 H3 blocks) = **618 lines, 14 % of N**, moved to `progress/…`; requires a **new** `progress/` naming rule, since rule 3 says a filename is *kept unchanged* and journal sections are not files — i.e. rule-making the parent has not approved | **64 of 65 line numbers break** (first deletion at `:81`); the 5 blocks carrying cites (`:101`,`:250`,`:1963`,`:2154`,`:2549` = 14 references) break *inside the frozen tree*. Also removes the audit trail for shipped numbers: `:1650-1766` is the only record of **why** the bands are 3.50/7.99, and `:250-298` is the only record of the §07.5 prompt-phase fix | The live file becomes self-consistent and the archive becomes the truth — the exact failure AGENTS.md records twice ("a check whose reference side is derived from the thing under test cannot adjudicate it"). And it still leaves 3,716 lines |

**Recommendation: (a′).** It is the only option with a zero-rewrite price, the only one that needs no
rule change, and the only one whose own damage is checkable. Pair it with the ledger/README prunes
**only if** the parent waives rule 2; otherwise do them as same-line retirement banners
(`> Moved to progress/… on 2026-10-16; text retained below because 5 archive citations address these
line numbers`) — which is ugly and correct.

## 5. Staged execution plan, ordered by risk (one commit per topic)

| # | commit topic | files | verification command |
|---|---|---|---|
| S1 | `tools: citation_census.py — the line/label/section forms this repo actually uses` — the §1 section dump + F1-F6 counters + `--check` (asserts parsed-record count equals records printed, the AGENTS.md "a check whose input list can be empty cannot fail" rule), `--selftest` with cases for `:724,726`, `:3731-3733`, `` `N` §Q6.1 ``, `ledger §6`, `results.md §21` (bare), and `§A2` (label-only) | new `tools/citation_census.py` only | `python3 tools/citation_census.py --selftest && python3 tools/citation_census.py --check`; expect F1=101, F2=171, live-origin F1=5 |
| S2 | `tools: doc_ref_check validates line anchors and section labels, and reads code` — see the ROOTS note below. Three additions: `LINE_ANCHOR_RE` (path + `:NNN[-NNN][,NNN]`, fail if NNN > `wc -l` of target or the anchored line is blank), a label resolver (`` `path` §L `` must match a fence-aware heading carrying `L`), and `--include-code` (default on) so `engine/gate.py`'s 17 `§A` cites are seen | `tools/doc_ref_check.py`, its `--selftest` fixtures | `python3 tools/doc_ref_check.py --selftest` → `SELFTEST PASS`; then `--list-broken` diffed against the 14 paths recorded in structure-proposal §6 — the delta must be exactly the new anchor/label findings, each explicable |
| S3 | `engine/gate.py, tools, specification: name the file behind every label citation` — append the owning path to the 17 `§A1..§A4` strings in `engine/gate.py`; add `A-19` and `A-32` rows to `specification/ASSUMPTIONS.md` (or retarget the two cites); fix `prototype/gdn_gated_norm/results.md:1` whose H1 states what §8 retracted | `engine/gate.py`, `specification/ASSUMPTIONS.md`, `prototype/gdn_gated_norm/results.md` (**same-line only**) | `python3 tools/citation_census.py --check` shows label-only count drop from 66; `python3 tools/doc_ref_check.py --list-broken` unchanged; `python3 -c "import ast,sys;ast.parse(open('engine/gate.py').read())"` |
| S4 | `docs: engine/README.md retires three stale claims` — `:11` "no generation token has come out of this engine yet", §"What this is NOT yet" (`:42` no prefill, `:47` no batching), §"What the next worker owes W5b" — **same-line retirement text, no line deletion** | `engine/README.md` | `grep -c 'no generation token\|No `prefill`' engine/README.md` → 0; `wc -l engine/README.md` **must be unchanged at 496** (that is the rule-2 guard); `python3 tools/citation_census.py --check` |
| S5 | `progress: the journal chronology, extracted by quotation` — option (a′): new `progress/2026-10-16/journal-chronology-2026-10-16.md` + the 5-line append at the tail of `engine/results.md` + `engine/README.md` pointer line | 2 new files, 1 append (0 deletions) | `wc -l engine/results.md` must be **4493+5**; `python3 tools/citation_census.py --check` → F1 still 101, all 65 anchors still resolvable; `python3 tools/doc_ref_check.py` unchanged + the new paths present |
| S6 | `bench: pre-clean-window rows leave the ledger` — the §3 PRUNE, **only if the parent waives rule 2**. Otherwise this commit is the same-line banner instead | `engine/performance-status.md`, new `progress/2026-10-16/ledger-pre-clean-window-2026-10-16.md`, plus updates to the 3 archive files holding `:988/:989/:1094` | `python3 tools/citation_census.py --check` must report **0 unresolvable anchors**; the ledger's 65 §-cites all still resolve to §6 |
| S7 | `AGENTS.md: two trap rows` — (i) a line-number citation is destroyed by any deletion above it, and 96 of 101 such cites are in the frozen archive; (ii) label-only citations (`§A2`, `results.md §20`) resolve by neighbourhood and are invisible to `git ls`/`doc_ref_check` | `AGENTS.md` | `python3 tools/doc_ref_check.py && python3 tools/citation_census.py --check` |
| S8 | `tools: journal_sections.py — the fence-aware section census` (optional; if S1 already carries `--sections`, fold it in and skip this commit) | new tool | `--selftest` asserts 14 H1 / 72 H2 / 33 H3 for `engine/results.md` |

**Does `tools/doc_ref_check.py` cover the new paths?** **Path existence: yes, already.** `ROOTS` (the list by that name near the top of the file) lists `progress`, added by
pass 2 so archive paths are adjudicated; `git check-ignore` confirms
`progress/2026-10-16/archive-review-2026-10-16.md`, `progress/engine/journal/v1.md` and
`testdata/run.py` are all NOT-IGNORED, while `engine/results/vol1.md` **is** ignored by
`.gitignore:69` (`results/`) — why option (b)'s natural destination is a trap. `--selftest` passes
today (`SELFTEST PASS`, 0.18 s wall). **Coverage it lacks, exactly:** (1) `markdown_files()` =
`git ls-files -z '*.md'`, so **every citation in `engine/*.py`, `*.cpp`, `*.h`, `*.sh` is unseen** —
this census found 5-6 live F1 cites and 17 label cites there; (2) the path pattern deliberately drops
the `:3739` suffix (its own selftest asserts `"line cite engine/results.md:3739 stays whole" →
['engine/results.md']`), so **a shifted line number cannot fail it** — the check that cannot fail;
(3) it does not know §labels exist, so `§Q99` passes vacuously; (4) it never sees `A-19`/`A-32` ids.
S2 is the minimal complete fix: keep `ROOTS`, widen the file set, add the anchor and label resolvers,
and add two selftest cases (`anchor past end of file → broken`, `label not in headings → broken`).

## 6. The second deferred item: `bench/run.py` — EXECUTED 2026-10-16, now `testdata/run.py`

**Ruling: move it to `testdata/run.py`. The census makes it safe.** It is the only non-doc file in
`bench/`, its corpus is `testdata/`, and its blast radius is measured, not guessed:

| what | count |
|---|---|
| `bench.sh` call sites | 3 — `:66`, `:73`, `:80`, all `"$here/bench/run.py"` |
| live doc citations | 2 — `testdata/README.md:35`, `:67` |
| archive citations | 8 — all in `progress/` (structure-proposal ×7, `progress/README.md:43` ×1); **none carries a line anchor** |
| line-form cites into `bench/run.py` | 0 |

So the move is 1 `git mv` + 5 live-path edits, with no line-number at stake anywhere. **The failure
mode is silent**: if `bench.sh` is not edited in the same commit, `./bench.sh` cannot find the
runner. Verification, all CPU: `bash -n bench.sh && grep -c 'bench/run\.py' bench.sh` → 0 after the
commit; `git ls-files bench/run.py` → empty; `python3 tools/doc_ref_check.py --list-broken` must not
list `bench/run.py`. Note the harness itself may be museum-piece (README still quotes its 23 tok/s
CPU baseline while the ledger's numbers come from device sweeps) — that is the open question 5 the
structure proposal already raised, and it does not block the move.

## 7. What this survey could not settle, and residual risks

1. **The rule-2 collision (§4) is the parent's call.** (a′) needs no waiver; every real prune does.
2. **`progress/` has no rule for a journal *section*** (rule 3 speaks of filenames). Option (c)
   would need one; (a′) sidesteps it by making the extraction a new dated document, which rule 3
   already covers.
3. **The 3 real live line-cites are unresolved-by-directory** (`results.md:113`, `results.md §20/§21`
   forms). S2 resolves them by nearest-heading inference; if the parent prefers, a same-line
   qualification in the 4 citing source lines is cheaper than any tooling.
4. **`§A1..§A4` are the shipping gate's evidence for a shipped band** and they live in an archived
   audit. Nothing in the tree proves those labels still match the numbers `CASCADE_BAND` encodes —
   S3 names the file, it does not re-verify the numbers. That needs a device-free re-read of
   `progress/2026-10-14/route-c-audit-2026-10-14.md:144-204`, not a checker.
5. **Classification is content-based, not tool-based.** Every NARRATIVE/SUPERSEDED call in §1 is
   quoted with a verified line number; LIVE-DATA calls rest on "a table/contract sits here" plus the
   cite column, and reasonable disagreement is confined to the zero-cite measurement blocks (e.g.
   `2437-2462` "The numbers"). No step in §5 depends on those particular calls.
6. **No timing was taken and no device command run.** Every number here is `grep`/`wc`/`git ls-files`
   over the working tree at the current HEAD, and every table above carries its command.


## 8. Addendum, same day: what the checkers it asked for came back and said

Appended, not edited in place — an append moves no line, so the 51 line-form
cites into this file and the `N:NNN` anchors it publishes stay valid. Three
corrections, each found by a tool this document asked for (S1
`tools/citation_census.py`, S2 the `tools/doc_ref_check.py` upgrade), and the
cost of building them:

* **§2.1 F5 is 24, not 23.** The published count was one short; the census's own
  `grep -ohE` recipe over the census's own tree gives 24, proven two ways —
  `git grep -ohE '<T>\`?,? ?\(' dece2d9 -- ':(exclude)this-file' | wc -l` → 24,
  and a clean `git archive dece2d9` extraction with this file removed → 24 by
  occurrence count and by matched-line count. Row corrected in place above; the
  same-line edit is the only kind rule 2 allows and the only kind a table cell
  needs.
* **§7.4's `route-c-audit-…:144-203` range was one line long**: line 203 is
  blank and the A3 heading is at `:204`, so the anchor resolver called it
  broken. Corrected to `:144-204`. Found by the anchor resolver, which is the
  check that could not have existed before S2 — a dead line cite in a document
  *about* dead line cites.
* **§7.4's `§A1…§A4` claim survives contact**; the dead-`§` exemplar in
  `tools/cold_restore.sh` does not: `:98` is `engine/performance-status.md §6 —
  fill in the spread and the util pair`, where `§6` resolves fine and "fill" is
  English, not a label. Recorded because "the check found nothing" is a
  different result from "the check was not run".

**The cost, and it is this document's own thesis.** Building S2 moved lines
inside `tools/doc_ref_check.py` — `ROOTS` went `73-77 → 101-105`, the selftest
fixtures `284,285 → 558,559` — which is exactly §4's failure mode arriving
through the front door: an upgrade to a checker silently invalidates the line
cites the archive holds *into that checker*. One of the two moved cites was
detectably dead (line 77 is now blank); the other two now point at the wrong
content and nothing reports it. So `:412` above cites `ROOTS` **by name**
rather than by number: where a citation must outlive edits to the file it
points at, a label is the only form that does, which is the general lesson
§2.1's F2/F5/F5b-versus-F1 asymmetry was already teaching.

**Third correction, found by the extraction S5 produced.** The §1 class table
and its own totals sentence disagreed: the rows sum to **59 LIVE-DATA = 3,716
lines and 14 SUPERSEDED = 513**, where the sentence said 3,745 and 483 — `+30`
and `−29`, netting to −1, which is exactly why the total looked plausible enough
to publish. NARRATIVE (265) and the row counts (59 / 13 / 14) were right, and
the ±1 between the table's 4,494 and the prose's 4,493 is the missing final
newline. Consequences fixed in place: the movable set is **883 lines, 20 %**,
not 853/19 %, and option (c) moves **618 lines, 14 %**, not 588/19 %. The
percentages were carried over from the wrong numerator twice, which a totals
sentence that nobody re-adds is especially good at hiding.

**S6 landed, and §6 is now a plan that was carried out.** `bench/run.py` is
`testdata/run.py`; `bench.sh`, `testdata/README.md` and the eight archive path
cites were updated same-line, so no archive line number moved. §6's own five
remaining mentions of the old path are deliberate history — they describe the
tree before the move, and `tools/doc_ref_check.py` will keep reporting them as
missing paths, which is what an executed plan should look like to a checker.
Where the census miscounted here: its §6 table lists 8 archive cites and misses
the 6 sitting in this document's own §6 prose — the second time in one day that
this file undercounted itself.

Corrected on the same pass, in the root `README.md`, which this review never
enumerated: `:47` still read "**No end-to-end engine number exists yet**" and
`:125` still placed the speed runner in `bench/`. Both fixed same-line, because
`progress/2026-10-16/cleanup-audit-2026-10-16.md:179` cites `README.md:186`.
