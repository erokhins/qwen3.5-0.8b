# Repo structure survey and reorganization proposal (2026-10-16)

Part 2 of the cleanup. Part 1 (`bench/cleanup-audit-2026-10-16.md`) settled *which cited files are
missing*; this settles *where things live and what moving them costs*. CPU-only: no build, no gate,
no device command, no timing, no move, no edit of a tracked file. Two files come out of this pass:
this document and `tools/doc_ref_check.py` (standing checker for stale path citations; real output
in §6). Every number was counted at `0f8b81e`; where the brief's figure differs, the counted figure
is used and the difference named.

**Headline: this tree does not want a reorganization. It wants four index files, one ignore-rule
allowlist, one journal rule, and one tool.** Exactly three moves cost under 5 doc lines each (PLAN B
B1–B3); every other candidate costs 15–435 citation lines and is marked "leave it" with its number
attached. The reason is the shape of the repo: the docs *are* the command lists — **2,146 markdown citation lines
across 376 distinct paths**, plus **449 path mentions in 398 commit messages**, which are immutable —
part 1 §4 is a whole table proving a rename leaves history stale (`engine/specimpl.py` is still
cited by commit `347ed54`).

---

## 1. Inventory

Tracked total **273** (brief said 278; `git ls-files | wc -l` = 273), of which **52 are `*.md`**.
Ignored-but-present: **52,909** files (`.venv/` 31,216; `data/` 20,350; `tmp/` 1,199; `models/` 42;
`prototype/` 31; `tools/` 24; `engine/` 16; `speccheck/` 14; `logs/` 10).

| dir | tracked | ignored-present | size | role here | drift found |
|---|---:|---:|---:|---|---|
| root | 4 | 0 | 40 KB | `AGENTS.md` scope+traps, `README.md` pitch+layout, `bench.sh`, `.gitignore` | README's Layout block (lines 115–131) lists 13 entries and **omits `tools/`, `logs/`, `tmp/`** — `tools/` is 41 tracked files, 15 % of the repo |
| `specification/` | 13 | 0 | 108 KB | the spec: `00`–`10`, `ASSUMPTIONS.md`, README | none; the only dir with a numbering scheme, and it holds it |
| `speccheck/` | 29 | 14 | 416 KB | the independent checker (`specimpl.py`, `gates/`, `run_all.py`, `fixtures/`) | two fixture homes: `speccheck/fixtures/corpus.txt` is **byte-identical** (md5 `08b6de40…`) to the ignored `data/corpus.txt`, which docs cite by name |
| `prototype/` | 65 | 31 | 3.3 MB | one dir per op: kernel + gate + `results.md`, graduating into `engine/` | (i) `prototype/README.md`'s "Current contents" table lists **4 ops**, the dir holds **9**; (ii) `prototype/gdn_chunk/` (6 files) has **no `run.py`, no `results.md`, no `build.sh`** — its verdict lives in `bench/p9-phase1-2026-10-14.md`, so the stated convention is false for 1 of 9; (iii) 3 shared modules sit loose at the root (`bf16_ulp.py`, `attn_tile_model.py`, `weight_names.py`) and are reached through `sys.path.insert(ROOT/"prototype")`, an idiom in **27 tracked `.py` files**, 18 of which import these three modules |
| `engine/` | 42 | 16 | 3.1 MB | the deliverable | flat mix of the engine (16 `.cpp`, 8 `.h`), **13 CPU python tools** (`gate.py`, `route_price.py`, `weights_contract.py`, `curve_probe.py`, …) whose siblings live in `tools/`, the journal, and `numerics_provenance.json`. `engine/gate.py` is the most-cited path in the repo: **185 doc lines in 22 files** |
| `bench/` | 23 | 0 | 668 KB | ledgers + plans + verdicts | four genres in 20 `.md` (1 living ledger, 2 undated one-shots, 17 dated), 2 machine-readable data files, **1 driver script** (`run.py`, invoked by `bench.sh`, cited only by `testdata/README.md`), and one filename whose date field is a sequence number |
| `devices/` | 4 | 0 | 92 KB | measured platform facts, each reproducible with a `tools/` probe | none; the transcripts its tables were read from sit in `logs/` and `tmp/` — correct, but invisible from `devices/` |
| `tools/` | 41 | 24 | 2.1 MB | probes, dumpers, analyzers, shell harnesses | **flat and no index** (`git ls-files 'tools/*.md'` = 0) while 4 kinds are cited by **435 doc lines + 183 code lines** |
| `testdata/` | 5 | 0 | 32 KB | the CPU-era speed corpus (`index.json`, 2 tests, source text) | its runner is in another dir (`bench/run.py`) and its entry point in a third (`./bench.sh`); README still quotes its 23-tok/s CPU baseline |
| `logs/` | 8 | 10 | 264 KB | device-run transcripts; the 8 tracked are the certificates part 1 committed | **44 % tracked, 56 % ignored, all matching `*.log` (`.gitignore:68`) — nothing records which are intentional** |
| `tmp/` | 39 | 1,199 | 83 MB | scratch | the tracked 39 (16 producer scripts + 23 evidence files) live under an ignored path; `tmp/pre-route/` also holds **snapshot copies of 8 tracked `results.md` + READMEs** (4.4 MB) |
| `data/` | 0 | 20,350 | 5.3 GB | the HF pins + generated bench output | correct by decision B; 9 cited paths absent by design (part 1 §3) |
| `models/` | 0 | 42 | 1.7 GB | the checkpoint | correct by decision B |
| `engines/` | 0 | 0 | 4 KB | README: "cloned reference engines for cross-checking" | **empty**, and README's layout block still lists it as a live destination |
| `thinking/` | 0 | 0 | 4 KB | README: "Start a chunk by reading the relevant `thinking/*.md`" | **empty**: the convention migrated to `bench/HANDOFF-2026-10-14.md` + the `results.md` journals. A session following README reads an empty dir |

Journal corpus, since PLAN B touches it: `engine/results.md` **4,493** lines; the 8
`prototype/*/results.md` **3,667** (attn_decode 1,509; mlp_gemv 534; gdn_gated_norm 517; proj_gemv
313; glue 286; gdn_collapse 220; gdn_decode 163; lm_head_gemv 125);
`bench/performance-status.md` **1,149**. Total 9,309.

---

## 2. Problems, with evidence

### (a) 47 tracked files live under ignored paths — 17.2 % of the repo, unmarked

`git ls-files tmp/` = **39**, `git ls-files logs/` = **8** → **47 of 273** tracked files (1.26 MiB,
part 1 groups 1–3), cited by **186 doc lines** (157 `tmp/`, 29 `logs/`) in **15 distinct md files**
(`engine/results.md`, `engine/README.md`, `bench/performance-status.md`,
`devices/strix-halo-mixed-traffic.md`, the part-1 audit, +10). Measured precisely:

- `git clean -nxd` lists **512** removals (471 under `tmp/`, 11 under `logs/`) and would **not**
  delete the tracked ones — `git clean` is not the trap, the plain `rm -rf tmp/` of commit `8c81e21`
  is. That command deletes 39 tracked files; `git status` then shows 39 `D` lines and
  `git checkout -- tmp/` restores them, *if anyone looks*. The proof this is not hypothetical is
  part 1 §4 row 12: `tmp/route_wide520f.json` was purged, is still cited by `engine/results.md:3251`,
  and this pass's checker still reports it missing.
- `git check-ignore tmp/q6` → **not ignored** (git cannot exclude a directory holding tracked files)
  while `git check-ignore tmp/q6/newprobe.py` → **ignored, `.gitignore:70`**. So the ignore machinery
  cannot warn about this class: the dir looks ordinary, its files look forbidden.
- `.gitignore` has **zero `!` lines**: nothing records that any `.log` or `tmp/` file is intentional.

### (b) `tools/` is a flat 41-file pile of four kinds, with no index

All 41 accounted for, none left over:

| group | files | doc lines | md files to rewrite | msg | members |
|---|---:|---:|---:|---:|---|
| device probes (produce `devices/` numbers) | 12 | 38 | 11 | 14 | `bw_probe_hip.cpp`, `bw_probe.py`, `launch_overhead_hip.cpp`, `mem_mode_probe_hip.cpp`, `device_query.cpp`, `gemm_peak.cpp`, `mfma_probe.cpp`, `mfma_isa.sh`, `mixed_bw.cpp`, `prefill_mix.cpp`, `route_c_stages.cpp`, `gemm_common.h` |
| fixture dumpers (produce `data/refs`) | 11 | 70 | 23 | 21 | `reference_dump.py`, `dump_fixtures.sh`, `corpus_dump.py`, `check_dump_reproduction.py`, `check_fixture_prefix.py`, `check_projection_pin_semantics.py`, `stop_probe.py`, `chat_render.py`, `state_fp.py`, `weights_report.py`, `fp16_range_audit.py` |
| analyzers / priced arithmetic | 13 | 113 | 20 | 25 | `band_history.py`, `margin_delta.py`, `step_traffic_model.py`, `instep_family_budget.py`, `prefill_row_rooms.py`, `prefill_gemm_shapes.py`, `route_c_audit.py`, `marker_audit.py`, `corpus_agreement.py`, `corpus_margin.py`, `corpus_freerun.py`, `context_sweep.py`, `prefill_sweep.py` |
| shell harnesses (lock + log discipline) | 5 | 183 | 25 | 10 | `gpu_lock.sh`, `gate_locked_run.sh`, `gate_log_diff.sh`, `cold_restore.sh`, `build.sh` |

The groups are real, but the cost of making them directories is dominated by one file:
**`tools/gpu_lock.sh` alone = 167 doc lines in 22 files + 39 code references**, including the three
self-locking drivers the AGENTS trap row tells operators to grep for (`prototype/glue/run.py` 5,
`prototype/proj_gemv/run.py` 5, `prototype/gdn_collapse/run.py` 2). And it is a build change, not a
docs change: `tools/build.sh` computes `src="$t.cpp"` relative to `tools/` and gives
`route_c_stages` a `../engine/*.cpp` list. Decisive: **`tools/` is a Python package to the gate as well
as a command list** — `engine/gate.py:160` does `from tools import band_history`, and 18 `.py` files in
`tools/` and `engine/` carry `tools/...` path strings — so a subdir split breaks an import, a build
script, and the self-locking gate drivers at once. Verdict: document the four groups (A1), do not
move them.

### (c) `bench/` mixes four genres plus data plus a driver

20 `.md`: **1 living ledger** (`performance-status.md`, cited by **82 doc lines in 29 files**),
**2 undated one-shots** (`gate-audit.md`, `build-queue.md`), **17 dated** in four sub-genres —
plans (`plan-2026-10-12`, `plan-prefill-2026-10-12`, `plan-p9-chunked-scan-2026-10-14`,
`plan-route-c-audit-2026-001`), audits/reviews (`gate-audit`, `commit-review-2026-10-11`,
`review-2026-10-11`, `route-c-audit-2026-10-14`, `cleanup-audit-2026-10-16`), findings
(`attn-wave-diagnosis`, `p3-boundary-fixtures`, `p9-chunked-scan`, `p9-phase1`, `p9-phase1c`,
`p9-route-a`, `prefill-route-spike`, `speedup-routes`), and the **HANDOFF**.

- Moving the 17 dated docs → `bench/2026-10/`: **75 doc lines in 22 md files** (AGENTS.md, README.md,
  18 bench docs, `engine/results.md`, two prototype journals) + ~35 code lines where gates and
  sweeps name their owning plan + **25 commit-message citations that can never be rewritten**. The
  same readability is available from one index file for free.
- New subdir names are a trap: `git check-ignore bench/data/x.jsonl` → ignored by `.gitignore:63`
  (`data/`), `bench/results/x.md` → ignored by `.gitignore:67` (`results/`). Same
  directory-vs-file class that already swallowed `prototype/**/<op>`. `bench/2026-10/` and
  `bench/journal/` are clean; check with `git check-ignore -v <dir>/x.md` before creating anything.
- "Should data sit with data?" — it is **2 files of 23**: `band-history.jsonl` (11 doc + 4 msg +
  **11 code lines**, default path of `tools/band_history.py --check`) and
  `corpus-freerun-2026-10-12.json` (4 doc lines). 15 doc + 11 code lines buys less than the index
  line that explains them, and the natural destination name is ignore-trapped.
- `bench/run.py` is a driver in a docs dir: corpus in `testdata/`, entry point `./bench.sh`, which
  calls `"$here/bench/run.py"` at lines 66/73/80. Cited by **2 doc lines in 1 file**. This is the
  cheapest real move in the repo (B2).
- Naming: `plan-route-c-audit-2026-001.md` is the one file whose date field is a sequence number,
  cited by exactly **1 doc line** (`bench/route-c-audit-2026-10-14.md:12`).

### (d) The journals: growth is real, line-cites are fewer than feared, and the prototype journals must not be renumbered

| measurement | value |
|---|---|
| doc lines citing `engine/results.md` | **86** in 16 files (+42 msg) |
| …carrying a line number `engine/results.md:NNNN` | **27** occurrences / 25 distinct lines (101…4478) |
| …of those, inside `bench/cleanup-audit-2026-10-16.md` | **26** (the other is `bench/commit-review-2026-10-11.md`) |
| line-cites below line 2204 | 15 |
| `prototype/*/results.md:NNNN` | 7 |
| `bench/*.md:NNNN` | 24 · `tools/…:NNNN` 2 |
| `results.md §N` citations | **36**, 25 of them naming a `prototype/` journal |

Structure, which decides *how* a split could work:

1. `grep -c '^# ' engine/results.md` = **86**, but there are **14 real H1 sections** — 72 of those
   lines are shell comments inside fenced blocks (`# cmd: .venv/bin/python engine/gate.py
   --fixture chunk64b`). **A splitter must be fence-aware** or it shreds code blocks, the bug class
   AGENTS.md records three times. Fence-aware H1 starts: 1, 109, 185, 250, 391, 1058, 1224, 1278,
   1459, 1537, 1650, 1797, 2005, 2204.
2. H1 granularity does not fix the growth: the last H1 (`D-4 …`) spans **2204–4493 = 2,290 lines,
   51 % of the file**, because past ~2204 the journal appends at `## ` (`## Q5 / P4+P4b`, `## Q7`,
   `## Q10`, `## Q11`, `## Route (c)`). A usable split unit is "H1, plus the decision-named H2s
   after 2204" → ~20 blocks of 13–257 lines.
3. `engine/results.md`'s own sections are **unnumbered**; its internal references are prose ("see the
   Q6.1 section below") and its `§N` refs point outward (`§04.4` = spec, `ledger §6` =
   `bench/performance-status.md`; 54 `§6` refs). So a split breaks no label addressing — there is none.
4. The **prototype journals are the opposite**: numbered H2s (`## 18. …`, `## 20.5 …`) and 25
   citations by that number — `AGENTS.md` cites `prototype/attn_decode/results.md §20.1–§20.3, §18,
   §18.2, §20.6`; `bench/HANDOFF-2026-10-14.md` cites `§20.4, §21`. **Splitting or renumbering a
   prototype journal silently breaks the traps index.**

Recommendation (A7/B3): **split forward, never backward.** Freeze `engine/results.md` at 4,493 lines
— 0 path rewrites, 0 of the 27 line-cites broken — and start a new volume per phase. Retrospective
splitting costs 27 line-cite rewrites (mechanical, 26 in one file) plus 86 path cites if the head
file moves, needs a fence-aware splitter, and buys nothing the freeze does not.

### (e) Empty dirs, the stray-root class, and the allowlist part 1 raised

- `engines/`, `thinking/`: both empty, both ignored, 0 tracked files — git cannot track a directory,
  so `rmdir` is invisible to git and purely local. The real defect is two README claims (Layout lists
  both as destinations; conventions still say "Start a chunk by reading the relevant
  `thinking/*.md`"). Note why "just add a README inside" is wrong: both are excluded as *directories*,
  so `thinking/README.md` needs `git add -f` — recreating problem (a) by design.
- The stray-root class part 1 deleted (`mlp_gemv.hip-hip-amdgcn-amd-amdhsa-gfx1151.o`, 73,488 B) was
  hidden by **two** `*.o` lines, `.gitignore:38` and `:95` — the duplicate itself is worth removing.
  `prototype/*/<op>` binaries are named exactly and cannot recur; the root `.o` came from a hand-run
  `hipcc`, so the fix is a hygiene line, not an ignore rule.
- Allowlist: part 1 §5(a) stands and is repeated in A4. The `tmp/` half cannot be allowlisted (git
  cannot re-include a child of an excluded directory) — keep `git add -f` there and record the rule.
- Depth trap for any future dir: `data/`, `refs/`, `dumps/`, `results/`, `tmp/`, `build/` are
  **unanchored directory patterns** and bite at any depth (measured: `bench/data/`, `bench/results/`).

### (f) Naming: spec ↔ prototype ↔ engine is not discoverable from names, and nothing tabulates it

The correspondence exists but lives in journal first lines, not filenames:

| spec | prototype op | engine file(s) | where the link is written |
|---|---|---|---|
| §03 glue+norm, **and §03.3 MLP** | `glue`, **`mlp_gemv`** | `kernels_glue.cpp`, `kernels_cast.cpp`, `kernels_mlp.cpp` | `mlp_gemv/results.md` H1 says "§03.3 MLP decode path" — `mlp_gemv` ↔ `specification/03-block-glue-and-normalization.md` is unguessable |
| §04 GDN | `gdn_decode` §04.2–4.4, `gdn_gated_norm` §04.5, `gdn_chunk` §04.7, `gdn_collapse` §04.4 | `kernels_gdn.cpp`, `gdn_collapse_kernels.h` | each results.md H1; `gdn_chunk` has **no journal at all** |
| §05 attention | `attn_decode` §05, `proj_gemv` §04.1+§04.5+§05.1+§05.6 | `kernels_attn.cpp`, `kernels_proj.cpp` | H1s; `proj_gemv` spans four sections, so no 1:1 map exists |
| §06 prefill | — (built straight into the engine) | `prefill.cpp`; `tools/route_c_stages.cpp` | `bench/plan-prefill-2026-10-12.md` |
| §07 decode step | `lm_head_gemv` | `decode_step.cpp`, `decode_run.cpp`, `kernels_lmhead.cpp` | `lm_head_gemv/results.md` H1 names **no §** |
| §08 state | — | `state.cpp`, `engine/state_sizes.py` | — |
| §09 MTP | — (descoped) | `speccheck/mtp.py` | README milestone |
| §10 dumps + gates | `tools/reference_dump.py`, `engine/gate.py`, `speccheck/gates/` | | `specification/10` |

6 of 8 op journals name their spec section in the H1; 2 do not (`glue`, `lm_head_gemv`). No document
maps op → engine file; `engine/README.md:64` states the rule in prose and
`prototype/README.md`'s index is 4 rows against 9 dirs. A second discoverability defect:
`engine/gdn_collapse_kernels.h` is `#include`d by `engine/kernels_gdn.cpp` while its prototype's own
title records the route as **DEAD (112.0 ms against an 84 ms kill line)** — nothing in the tree says
which side is live except that title. Fix = a table (A9), not a rename: renaming ops to carry §
numbers costs **429 doc lines citing `prototype/` paths** + every `.gitignore` binary line.

---

## 3. Target layout

Recommended tree. **Bold = new file**; everything else stays exactly where it is.

```
AGENTS.md                        + 2 trap rows (allowlist + ignore-depth trap)
README.md                        Layout block corrected: add tools/, logs/, tmp/;
                                 drop "read thinking/*.md"; mark engines//thinking/ on-demand
bench.sh                         unchanged entry point
bench/
  README.md                      NEW  index of every file by genre + the data files' producer
  performance-status.md          the ledger, unchanged (82 doc lines cite it)
  gate-audit.md  build-queue.md  unchanged
  <17 dated docs>                LEAVE IN PLACE (move costs 75 doc lines + 25 immutable msgs)
  band-history.jsonl             LEAVE (11 doc + 11 code lines; band_history.py --check default)
  corpus-freerun-2026-10-12.json LEAVE (4 doc lines; explained by bench/README.md)
  run.py                    ->   testdata/run.py          (MOVE-1, the only cheap move)
  plan-route-c-audit-2026-001.md -> -2026-10-13.md        (MOVE-2, fixes the date pattern)
bench/2026-10/                   NOT created (see the 75-line cost in 2c)
devices/                         unchanged (4 files)
data/ models/ .venv/             unchanged, ignored by decision B
engine/
  README.md                      + 3 lines: the journal-volume rule
  results.md                     FROZEN at 4,493 lines; nothing inserted above existing text
  results-2026-11.md             NEW volume when the next phase closes
  <24 C++/H + 13 py, gate.py among them>  all LEFT (gate.py alone is 185 doc lines)
logs/
  README.md                      NEW  the 8 tracked logs and what each certifies (part 1 §2a)
  <8 tracked + 10 ignored .log>  unchanged; .gitignore now carries !logs/*.log
prototype/
  README.md                      refreshed: 9 ops + 3 shared modules, + spec-section and
                                 engine-file columns (the §2f table), + a gdn_chunk row
  bf16_ulp.py attn_tile_model.py weight_names.py   LEAVE (25 doc lines; 27 files' sys.path idiom)
  <9 op dirs>                    names unchanged (renaming costs 429 doc lines)
speccheck/  specification/  devices/   unchanged
testdata/
  README.md index.json *.json summarize_source.txt
  run.py                         MOVED IN from bench/run.py
tools/
  README.md                      NEW  the 4 groups, 41 rows, each with what it produces
  doc_ref_check.py               NEW (this pass) the citation checker
  <41 existing files>            LEAVE (435 doc + 183 code lines; gpu_lock.sh = 167 + 39)
tmp/                             unchanged path; the rule moves into .gitignore + AGENTS.md
tmp/pre-route/                   optional prune (4.4 MB of doc snapshots, MOVE-4)
engines/  thinking/              rmdir locally (git cannot see it); README claims corrected
evidence/                        NOT created: costs 186 doc lines + 24 msgs to relocate 47 files
```

Per-move accounting — the git mv, the citation counts, and the doc files that would need editing.

| move | command | doc cites | msg cites | code refs | doc files to edit | verdict |
|---|---|---:|---:|---:|---|---|
| MOVE-1 | `git mv bench/run.py testdata/run.py` | 2 | 1 | 3 (`bench.sh:66,73,80`) | `testdata/README.md` | **do it** — the last non-doc file leaves `bench/`, the runner sits with its corpus; verify with `./bench.sh --quick` (CPU) |
| MOVE-2 | `git mv bench/plan-route-c-audit-2026-001.md bench/plan-route-c-audit-2026-10-13.md` | 1 | 1 | 0 | `bench/route-c-audit-2026-10-14.md` | **do it** — the dated set then sorts |
| MOVE-3 | journal volume rule: new `engine/results-<phase>.md`; pointer in `engine/README.md` (or appended at the very end of `results.md`) | 0 | 0 | 0 | `engine/README.md` | **do it** — the only journal split that breaks none of the 27 line-cites |
| MOVE-4 | `rm -rf tmp/pre-route/` | 1 | 0 | 0 | the part-1 audit §4 row 5 sentence | optional; it is the only pre-route snapshot of the docs |
| dated → `bench/2026-10/` | 17 × `git mv bench/<dated>.md bench/2026-10/` | **75** | 25 | ~35 | 22 files: AGENTS.md, README.md, 18 bench docs, engine/results.md, prototype/attn_decode/results.md, prototype/gdn_collapse/results.md | **leave it** — the index (A2) gives the same readability at 0 path cost |
| bench data → a data dir | 2 × `git mv` | 15 | 6 | 11 (`tools/band_history.py` default path) | 7 files | **leave it** — 2 files of 23, and `bench/data/` is ignore-trapped |
| evidence `tmp/`+`logs/` → `evidence/` | 47 × `git mv` | **186** | 24 | 4+ (`step_traffic_model.py --probe-file`) | 15 files | **leave it** — replace with A4 (allowlist) + A3 + the AGENTS rule |
| `tools/` → 4 subdirs | 41 × `git mv` | **435** | 80 | **183** | 41 files | **leave it** — `gpu_lock.sh` alone is 167 doc + 39 code cites; `tools/build.sh` resolves `src="$t.cpp"` relative to `tools/` |
| prototype ops → §-prefixed names | 9 dirs | **429** | 53 | every `run.py`/`build.sh` + 9 `.gitignore` binary lines | 39 files | **leave it** — fix discoverability with A9 instead |
| `prototype/` shared 3 → `prototype/_lib/` | 3 | 25 | 5 | 27 files carry `sys.path.insert(ROOT/"prototype")`, 18 import these three | 12 files | **leave it** |
| `engine/` split (C++ vs the 13 py) | 13 | 83+ | 20+ | `engine/gate.py` imports them | 9 files | **leave it** |

---

## 4. Two ranked plans

### PLAN A — nothing moves, every trap gets a marker

| # | step | cost | rollback |
|---|---|---|---|
| A1 | `tools/README.md`: the 41 tools in the four groups of §2b, one line each, with a "produces →" column (`mixed_bw.cpp` → `devices/strix-halo-mixed-traffic.md`; `reference_dump.py` → `data/refs/<tag>/`) and the self-lock warning on the three gate drivers that take the lock themselves | 1 new file ~55 lines; **0 moves, 0 doc lines rewritten** | `git rm tools/README.md` |
| A2 | `bench/README.md`: every bench file labelled **ledger / plan / audit / findings / handoff / data / superseded**, with the naming convention stated (`YYYY-MM-DD`) | 1 new file ~30 lines | `git rm bench/README.md` |
| A3 | `logs/README.md`: the 8 tracked logs and the number each certifies (trimmed from part 1 §2a), plus "the other 10 here are noise; `git add -f` is how a log becomes provenance" | 1 new file ~15 lines | `git rm logs/README.md` |
| A4 | `.gitignore`: part 1 §5(a)'s block (`!logs/*.log`, `logs/pip_*.log`, `logs/download.log`), delete the duplicate `*.o` at line 95, and a `tmp/` comment block ("39 tracked files live here — cited provenance; **never `rm -rf tmp/`**; recover with `git checkout -- tmp/`") | 1 file, **+8 / −1 lines** | `git revert` |
| A5 | `AGENTS.md`: two trap rows — (i) tracked files under ignored paths, why `git clean` is safe and `rm -rf` is not, and the `git check-ignore` blind spot measured in 2a; (ii) unanchored directory patterns (`bench/data/`, `bench/results/` are silently ignored — `git check-ignore -v <dir>/x` before creating any dir) | 2 rows | `git revert` |
| A6 | `README.md`: add `tools/`, `logs/`, `tmp/` to Layout; drop "Start a chunk by reading `thinking/*.md`" (dir empty; handoff memory is `bench/HANDOFF-*.md` + the journals); mark `engines/`/`thinking/` created-on-demand | 1 file ~5 lines | `git revert` |
| A7 | Journal rule: `engine/results.md` frozen; new volume per phase; **never insert above an existing line in a journal** (27 citations carry line numbers, 26 of them in one file) | 1 file, 3 lines | `git revert` |
| A8 | Adopt `tools/doc_ref_check.py` as the docs gate: run before any commit touching `*.md`; fix the stale names it reports (`speccheck/run.py` → `specimpl.py`/`run_all.py`; `prototype/gdn_chunk/run.py` → `phase1.py`; the `engine/gen_watch.py` sentence at `engine/results.md:496`; `specification/ASSUMPTIONS.md:4` → `transformers/models/qwen3_5/…`) | tool already written here + ~5 doc lines | `git revert` |
| A9 | `prototype/README.md`: table to 9 ops + 3 shared modules with **spec-section** and **engine-file** columns (the §2f table) and a `gdn_chunk` row saying where its verdict lives | 1 file ~14 lines | `git revert` |

**Total: 6 files touched/created + ~10 doc lines. Zero moves, zero code changes, zero device work.**
What could break: nothing executable. The honest risk is index rot — the three new indexes go stale
exactly as `prototype/README.md`'s 4-row table did (4 rows, 9 ops), so A1/A2/A9 should be
re-generated whenever `doc_ref_check.py` is run, or generated by a flag instead of hand-written
(open question 6).

### PLAN B — the moves that pay, cheapest first

| # | step | cost | rollback | what could break |
|---|---|---|---|---|
| B1 | MOVE-2, the date-pattern rename | 1 `git mv`, 1 doc line | revert the commit | `bench/route-c-audit-2026-10-14.md:12` + one immutable commit message |
| B2 | MOVE-1, `bench/run.py` → `testdata/run.py` | 1 `git mv`, 2 doc lines, 3 lines of `bench.sh` | revert | `./bench.sh` breaks **silently** if `bench.sh` is not edited in the same commit — run `./bench.sh --quick` in the same pass (CPU, no device) |
| B3 | MOVE-3, journal volume rule | 3 doc lines + a new file when the phase closes | delete the appended line | if the pointer is inserted above existing content all 27 line-cites shift — append at the end or keep it in `engine/README.md` |
| B4 | MOVE-4, prune `tmp/pre-route/` (4.4 MB doc snapshots) | 1 doc line | irreversible (untracked) | it is the only pre-route snapshot; delete only if not wanted as a witness |
| B5 | *only if the user asks*: dated docs → `bench/2026-10/` | 17 `git mv` + **75 doc lines in 22 files** + ~35 code lines; 25 msgs unfixable | one commit per month-dir, `git revert`; `git mv` + rewrite in the **same** commit so the tree is never half-moved | the 24 `bench/*.md:NNNN` line-cites survive (line numbers do not move), but every reproduce block reading `bench/plan-2026-10-12.md` silently stops working — these docs are the command list |
| B6 | everything else in §3's leave-it column | not worth it | — | see the numbers |

The hazard that governs both plans: **a path move without a reference rewrite silently breaks the
command lists the docs exist to be.** 449 commit-message path mentions and 2,146 markdown citation
lines make "rename" and "rewrite the citations" one atomic unit; part 1 §4 is what the tree looks
like when that unit was not respected. `tools/doc_ref_check.py` is the tripwire, which is why PLAN A
ships it before any move is considered.

---

## 5. Open questions for the user

1. **Dated filenames: index or directory?** 17 of 20 bench docs are dated. `bench/2026-10/` costs 75
   doc lines + ~35 code lines and leaves 25 commit-message citations permanently stale; `bench/README.md`
   (A2) costs one file. Do you want dirs or the index?
2. **Does committed evidence get its own dir?** 47 tracked files (39 `tmp/` + 8 `logs/`, 1.26 MiB,
   186 doc lines) live under ignored paths. Relocating to `evidence/` ends the `rm -rf tmp/` exposure
   for good at that price; the alternative (A4 + A3 + an AGENTS row) costs 8 ignore-file lines and
   depends on a human rule. Note the obvious names are already traps: `data/`, `results/`, `refs/`,
   `dumps/`, `tmp/`, `build/` are ignored **at any depth**.
3. **Is `engine/results.md` at 4.5k lines a problem now?** Recommendation: freeze and split forward
   (A7/B3, 3 doc lines, zero broken citations). Retrospective splitting is viable — 26 of the 27
   line-cites are in one file — but needs a fence-aware splitter (86 raw `# ` lines vs 14 real
   sections) and buys nothing the freeze does not. Either way `prototype/*/results.md` must **not** be
   renumbered: 25 citations address them by their internal `§N`.
4. **`engines/` and `thinking/`:** rmdir + correct the two README claims (5 doc lines), or keep
   `thinking/` as a live convention — which needs `git add -f` for anything inside it and so
   recreates problem (a) by design?
5. **Does the CPU speed harness still have a role?** `./bench.sh` + `bench/run.py` + `testdata/`
   (untouched since Oct 10) produce the 23-tok/s speccheck baseline README quotes, while the ledger's
   numbers now come from `tools/context_sweep.py` / `tools/prefill_sweep.py` on device. Museum piece →
   one README line and MOVE-1 is optional; still used → MOVE-1 makes the trio coherent.
6. **Hand-written indexes or generated ones?** A1–A3 are ~100 lines that can rot the way
   `prototype/README.md`'s table already did (4 rows against 9 ops). A `--markdown` mode on
   `doc_ref_check.py` (or a second small tool) keeps them true, at the cost of the first generated
   artifact in a repo that has none.

---

## 6. `tools/doc_ref_check.py` — what it is and what it printed

The only code written by this pass. It reads every tracked `*.md` (`git ls-files '*.md'`, 52 files),
tokenizes every repo-rooted path citation in three shapes (extension-bearing, brace shorthand
`engine/prefill.{h,cpp}`, bare dir/shorthand), and buckets each distinct path against the git index,
the filesystem, and `.gitignore` (one batched `git check-ignore --stdin`). Buckets: `TRACKED`,
`PRESENT-DIR` (a cited directory — git tracks files, not dirs), `PRESENT-IGNORED` (the
cited-provenance class), `PRESENT-DIRTY` (present, untracked, not ignored = `git status` is dirty),
`MISSING-STALE` (absent and not ignored, with nearest tracked candidates printed), `MISSING-SCRATCH`
(absent under `tmp/`/`logs/`), `MISSING-REGENERABLE` (absent but ignored by policy: `data/`,
`models/`, build output), `SHORTHAND` (directories, module stems, `specification/02` section cites —
not adjudicable). Modes: default report, `--all`, `--list-broken` (exit 1 when anything is broken),
`--selftest`.

The part-1 extension bug is fixed twice and asserted: the extension alternation is sorted
**longest-first at import** (`EXTS_SORTED`, so source order cannot reintroduce it) *and* every match
is followed by `(?![\w])(?!\.\w)`, so a truncated extension cannot match even if the list is later
re-sorted wrongly. `--selftest` covers part-1 §1's four cases in both directions (`.jsonl` not
`.json`; `.out` not `.o`; and the two phantoms when a doc writes them literally), plus brace
shorthand, `md:3739` line cites, compound `.hip.cpp`, trailing-slash dirs, `data/refs/p64/…`, the
glob stem `engine/kernels_*.cpp` (→ no path) and the section shorthand `specification/02`.

```
$ python3 tools/doc_ref_check.py --selftest
ok    see bench/band-history.jsonl for the bands           -> ['bench/band-history.jsonl']
ok    full text in tmp/p9/stage.out here                   -> ['tmp/p9/stage.out']
ok    was bench/band-history.json once                     -> ['bench/band-history.json']
ok    and tmp/p9/stage.o was never built                   -> ['tmp/p9/stage.o']
ok    engine/prefill.{h,cpp} are both tracked              -> ['engine/prefill.h', 'engine/prefill.cpp']
ok    line cite engine/results.md:3739 stays whole         -> ['engine/results.md']
ok    specs cite specification/05-gated-full-attention.md:12 -> ['specification/05-gated-full-attention.md']
ok    a bare dir: prototype/gdn_chunk has five scripts     -> ['prototype/gdn_chunk']
ok    trailing slash: write it into tmp/gate_logs/ now     -> ['tmp/gate_logs']
ok    ellipsis dir data/refs/p64/… is a dir cite          -> ['data/refs/p64']
ok    compound suffix prototype/gdn_decode/gdn_decode.hip.cpp -> ['prototype/gdn_decode/gdn_decode.hip.cpp']
ok    a glob is not a path: engine/kernels_*.cpp           -> []
ok    a spec section, not a file: specification/02 says    -> ['specification/02']
ok    two in one line: tools/gpu_lock.sh and tools/build.sh -> ['tools/gpu_lock.sh', 'tools/build.sh']
SELFTEST PASS
```

Real output on this tree (CPU, 0.2 s wall):

```
$ python3 tools/doc_ref_check.py
scanned 52 tracked markdown files, 273 tracked paths
doc_ref_check: 376 distinct path citations, 14 broken
  TRACKED               226
  PRESENT-DIR            61
  PRESENT-IGNORED        26
  PRESENT-DIRTY           0
  MISSING-STALE           6
  MISSING-SCRATCH         6
  MISSING-REGENERABLE     2
  SHORTHAND              49
```

Reconciliation against part 1 §4 — the acceptance test. **All 10 of the audit's genuinely missing
rows are reproduced**: rows 4–7 (`engine/specimpl.py`, `speccheck/run.py`,
`prototype/gdn_chunk/run.py`, `engine/gen_watch.py`), rows 8–9 (`models/qwen3_5/modeling_qwen3_5.py`
and `data/refs/p64/prefill_state_mismatch.json`, correctly bucketed policy-absent), row 10 (both
`tmp/form1024/*/wave_carry_deferred.json`), row 11 (`tmp/gdna/rocprof.txt`), row 12
(`tmp/route_wide520f.json`). **Neither phantom is reported** (row 1 `bench/band-history.json` is not
manufactured from `.jsonl`, row 2 `tmp/p9/stage.o` not from `.out`, and row 3's
`engine/prefill.{h,cpp}` expands to two tracked files). The 14 reported break down as 10 + 4, and
each extra is explained:

| extra path | why |
|---|---|
| `bench/band-history.json`, `tmp/p9/stage.o` | quoted-artifact noise: part 1's own §4 table writes the two phantom names literally, so a path scanner legitimately finds them. Documented, not suppressed — a checker that hides them cannot prove the fix |
| `engine/prefill.h/.cpp` | real prose looseness: `bench/plan-prefill-2026-10-12.md:367` writes `engine/prefill.h/.cpp` (not the brace form). The checker prints `engine/prefill.cpp` and `engine/prefill.h` as the candidates |
| `tmp/engine_gate/logits119.bin` | **new find**: cited by `engine/results.md:3063`; part 1's extension set had no `.bin`, so it could not see it. A purged F32 logits dump — the class the next audit would have lost again |

`--list-broken` prints the 14 paths one per line and exits 1, so it is usable as a gate; on a tree
where only policy-absent paths remain it still exits 1 by design (a cite to `data/refs/p64/…` is
normal on a clean checkout but is still an unresolvable cite, and the report labels which is which).

Two operational notes, both measured rather than assumed:

* **The scanner reads tracked files only** (`git ls-files '*.md'`), so the 14 above do not yet include
  this proposal. Re-run over this file with the tool's own functions and it reports **81 distinct path
  cites, 21 of them missing** — 4 are the stale names it documents, 4 are paths it *proposes to create*
  (`tools/README.md`, `bench/README.md`, `logs/README.md`, `testdata/run.py`), 1 is the MOVE-2 target
  name, and the rest are the examples it quotes (`bench/data/x.jsonl`, `tmp/q6/newprobe.py`). That is
  the expected behavior of a path checker pointed at a plan: read `--list-broken` as a diff against
  the previous run, not as a pass/fail, until PLAN A's files exist. When A1–A3 land, the count drops
  back to the 14 above plus the `logs/README.md` line.
* **`tools/` doubles as a Python package**: `engine/gate.py:160` does `from tools import band_history`,
  and 50 tracked code files (9 `engine/`, 8 `prototype/`, 2 `speccheck/`, 3 cited `tmp/`, the rest in
  `tools/`) contain `tools/...` path strings. That is the concrete reason the §3 leave-it verdicts on
  `tools/` subdirs are not a matter of taste.
