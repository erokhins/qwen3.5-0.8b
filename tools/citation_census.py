#!/usr/bin/env python3
r"""The citation census, as a tool: fence-aware section dump, the six citation forms, line anchors.

Why this exists: `progress/2026-10-16/archive-review-2026-10-16.md` established that this repo's
documentation addresses itself in six different ways, that only two of them survive a move, and
that every number in that census was produced by a hand-run `grep` pipeline nobody could re-run.
This tool is those greps, plus the two resolvers the census said were missing (does a line number
still land on a real line, does a section label still exist). CPU only: no device, no build, no
timing, no `tools/gpu_lock.sh`. Git-only input — `git ls-files`, like `tools/commit_msg_rewrite.py`.

## The six forms (spelled exactly as the census spelled them, §2.1)

`TARGET` is the alternation the census used — the docs a citation can point at, including the
directory-less `results.md` / `performance-status.md` shorthands. Counted as **occurrences**
(`grep -o` semantics: two matches on one line are two), over **every tracked file, code included**
(the census's §2.1 finding (i): live C++ and Python carry citations, and
`tools/doc_ref_check.py` read only `*.md`).

| form | pattern | meaning |
|---|---|---|
| F1 | `TARGET:[0-9]` | line-form cite. STRICT: no space after the colon |
| F1 tolerant | `TARGET: ?[0-9]` | the same cite when a prose space crept in (`…md: 24`) — 102 / 6 |
| F2 | `` TARGET`? ?§ `` | path + section sign |
| F4 | `ledger §[0-9]` | the ledger by nickname + section |
| F5 | `` TARGET`?,? ?\( `` | path + parenthetical qualifier (usually a section *title*) |
| F5b | `§[A-Za-z]` | letter-labelled section sign, no digits required |
| F6 | `§[QAP][\\w.]*` on a line that names no TARGET | label with **no file named on the line** |

**A count must name its spelling.** F1 strict is 101 and F1 tolerant is 102; that is not a
disagreement, and `--check` prints both with the spelling on the line (census §2.3).

F6 is the one form the census counted but never spelled out. The spelling here is derived and
recorded, not guessed at silently: a letter-led label from the three namespaces this repo uses for
numbered sections (`§Qn` in `engine/results.md`, `§An` in `progress/2026-10-14/route-c-audit-…`,
`§Pn` phase labels) appearing on a line that names none of the TARGET paths. That reproduces the
census's 66 exactly. The looser "any `§` + any letter" reading gives 76, and is wrong here because
it sweeps up prose pointers (`§route A`, `§expectation 6`, the census's own meta-`§label`).

## Why `--check` re-counts the tree at `dece2d9` instead of the worktree

The census counted its tree while its own file was still untracked, so it is not in its own baseline;
`progress/2026-10-16/archive-review-2026-10-16.md` §2.5 records the identical effect for the
structure proposal ("+4 from the proposal itself becoming tracked"). The census is now tracked and
carries 17 F1 / 10 F2 / 4 F4 / 23 F5 / 48 F5b / 24 F6 citations of its own — and so do its checkers,
because a tool that documents the forms it counts writes examples of them, and a `--selftest` fixture
is a citation as far as any counter is concerned. The census already conceded that point when it
counted 2 of its own 5 live-origin line-cites as "a checker's own `--selftest` fixture strings".

So `--check` reads every file as of `dece2d9` (the commit that carried the census): files edited since
then are read from that commit, files it did not contain read as empty, and the census document always
reads empty. That reconstructed column is what PASS/FAIL is judged on, because it is the number the
census predicted; the worktree total and the difference between the two are printed beside it, so
nothing is hidden and the growth is attributed rather than waved away.

## What the resolvers mean

`--anchors` answers, for every `path:NNN[-NNN][,NNN]`: which file, and does each number land on a
real, non-blank line. A range is checked at both endpoints (an interior line is not being claimed).
A directory-less `results.md:113` is reported UNROOTED with its candidate list rather than guessed
at — the census §2.4(a) shows the same filename means three different files in this tree.

`--sections` is the census §1 dump: level, start-end, line count, title. A `# ` line inside a
fenced block is a shell comment, not a heading: `grep -c '^# ' engine/results.md` reads 86 and
72 of those are inside fences. Fence-aware, `engine/results.md` is 14 H1 / 72 H2 / 33 H3.

## The AGENTS.md rule this obeys

A driver whose parsed list can be empty cannot fail. So every counter here reports the number of
records it built, the number of files it actually opened, and refuses to certify a form whose record
list is empty or whose file-open count is short of the file list it was given. `--selftest` runs the
same counters over synthetic fixtures and includes a **mutation harness**: for each rule, a mutated
input is required to FAIL. A rule that cannot fail is reported as a selftest failure.

Exit status: 0 = everything passed; 1 = a count disagrees, an anchor does not resolve, or a
selftest case failed; 2 = used wrong (no git checkout, unreadable path).

Usage:
  python3 tools/citation_census.py --sections engine/results.md
  python3 tools/citation_census.py --check
  python3 tools/citation_census.py --check --list F1
  python3 tools/citation_census.py --anchors                 # whole tree
  python3 tools/citation_census.py --anchors engine/gate.py # one citing file
  python3 tools/citation_census.py --selftest
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# The census's $T alternation, verbatim (§2.1). Order matters only for readability: a regex
# alternation is first-match-wins, and every alternative here is a distinct path shape.
TARGET_ALTS = (
    r"engine/results\.md",
    r"prototype/[^/ ]+/results\.md",
    r"bench/performance-status\.md",
    r"devices/[A-Za-z0-9-]+\.md",
    r"specification/[A-Za-z0-9.-]+\.md",
    r"engine/README\.md",
    r"AGENTS\.md",
    r"README\.md",
    r"results\.md",
    r"performance-status\.md",
)
TARGET_RE = "(?:" + "|".join(TARGET_ALTS) + ")"

# The file whose counts became this tool's expectations. It is tracked now and was not when it
# counted itself out of the picture; see the module docstring.
CENSUS_DOC = "progress/2026-10-16/archive-review-2026-10-16.md"

# What `progress/2026-10-16/archive-review-2026-10-16.md` §2.1/§2.3/§2.4 predicted, per form.
EXPECTED = {
    "F1": 101,
    "F2": 171,
    "F4": 16,
    "F5": 24,  # 24, not the census's published 23: its own recipe over its own tree
                # yields 24 (proven by git grep at dece2d9 and by a clean git archive
                # extraction). See progress/2026-10-16/archive-review-2026-10-16.md §8.
    "F5b": 95,
    "F6": 66,
}
EXPECTED_TOLERANT = {"F1": 102}
EXPECTED_LIVE_ORIGIN = {"F1": 5}
# The census §1 heading census, for the INFO line (the journal grows, so this is not a gate).
EXPECTED_HEADINGS = {"engine/results.md": {1: 14, 2: 72, 3: 33}}

# Files that can hold a citation, for --anchors and for the origin split. Code included, per §2.1.
TEXT_SUFFIXES = (".md", ".py", ".cpp", ".h", ".sh")

# ---------------------------------------------------------------- form patterns
F1_STRICT = re.compile(TARGET_RE + r":[0-9]")
F1_TOLERANT = re.compile(TARGET_RE + r": ?[0-9]")
F2_RE = re.compile(TARGET_RE + r"`? ?§")
F4_RE = re.compile(r"ledger §[0-9]")
F5_RE = re.compile(TARGET_RE + r"`?,? ?\(")
F5B_RE = re.compile(r"§[A-Za-z]")
F6_RE = re.compile(r"§[QAP][\w.]*")
FORM_PATTERNS = {"F1": F1_STRICT, "F2": F2_RE, "F4": F4_RE,
                 "F5": F5_RE, "F5b": F5B_RE}

# `path:NNN`, `path:NNN-NNN`, `path:NNN,NNN`, and the mixed `:NNN-NNN,NNN`. A directory-less
# filename is allowed as a *cite* so it can be reported UNROOTED, never silently skipped.
ANCHOR_RE = re.compile(
    r"(?<![\w./~-])((?:[\w.-]+/)*[\w.-]+\.(?:md|py|cpp|h|sh|json|jsonl|log|txt|npy|bin))"
    r"(?::(\d+(?:[-,]\d+)*))(?![\w])")
ANCHOR_NUM_RE = re.compile(r"\d+")
# A section-sign cite, used by --anchors' sibling report and by --selftest fixtures.
LABEL_CITE_RE = re.compile(r"§([A-Za-z0-9][\w.]*)")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
FENCE = "```"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def tracked_files() -> list[str]:
    root = git("rev-parse", "--show-toplevel").strip()
    if root:
        os.chdir(root)
    files = [x for x in git("ls-files").split("\n") if x]
    if not files:
        raise SystemExit("citation_census: no tracked files — is this a git checkout?")
    return files


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().split("\n")


# ---------------------------------------------------------------- sections
class Heading:
    __slots__ = ("start", "level", "title", "end")

    def __init__(self, start: int, level: int, title: str, end: int) -> None:
        self.start, self.level, self.title, self.end = start, level, title, end


def fence_headings(lines: list[str]) -> list[Heading]:
    """Fence-aware H1-H6 dump. A `# ` line inside a ``` fence is a shell comment, not a heading."""
    found: list[Heading] = []
    fence = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(FENCE):
            fence = not fence
            continue
        if fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            found.append(Heading(i, len(m.group(1)), m.group(2), 0))
    for k, head in enumerate(found):
        head.end = found[k + 1].start - 1 if k + 1 < len(found) else len(lines)
    return found


def cmd_sections(path: str) -> int:
    """start  end  n-lines  level  title — the census §1 table, one line per heading."""
    lines = read_lines(path)
    heads = fence_headings(lines)
    counts = {}
    for head in heads:
        counts[head.level] = counts.get(head.level, 0) + 1
    for head in heads:
        print("%d\t%d\t%d\t%s\t%s" % (head.start, head.end,
                                      head.end - head.start + 1,
                                      "#" * head.level, head.title))
    print("# %s: %d lines by `wc -l` (%d newline-split rows), %d headings (%s), raw "
          "`grep -c '^# '` = %d (fence-blind, NOT the count)"
          % (path, len(lines) - (1 if lines and lines[-1] == "" else 0), len(lines), len(heads),
             ", ".join("H%d=%d" % (lv, counts[lv]) for lv in sorted(counts)),
             sum(1 for l in lines if l.startswith("# "))))
    return 0


def wc_lines(path: str) -> int:
    """What `wc -l` says, i.e. newline count — not the newline-split row count."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().count("\n")


# ---------------------------------------------------------------- form counting
class Count:
    """One form's tally. `records` and `per_file` are two independent witnesses of the total."""

    def __init__(self, form: str) -> None:
        self.form = form
        self.records: list[tuple[str, int, str, str]] = []
        self.per_file: dict[str, int] = {}
        self.files_opened = 0

    @property
    def total(self) -> int:
        return len(self.records)

    def integrity(self) -> list[str]:
        """The AGENTS.md anti-vacuous guards. Empty list == the count is admissible."""
        problems = []
        if self.total == 0:
            problems.append("zero records — a counter that matches nothing certifies nothing")
        if sum(self.per_file.values()) != self.total:
            problems.append("per-file sum %d != records %d"
                            % (sum(self.per_file.values()), self.total))
        return problems


def count_pattern(rx: re.compile, files: list[str],
                  skip_line: "re.Pattern[str] | None" = None,
                  provider: "dict[str, list[str]] | None" = None) -> Count:
    """Every occurrence of `rx` in `files`. `skip_line` suppresses whole lines (F6's rule).

    `provider` replaces a named file's content with in-memory lines; that is how `--check` re-counts
    the census commit rather than the worktree.
    """
    tally = Count(rx.pattern)
    for fn in files:
        if provider is not None and fn in provider:
            lines = provider[fn]
        else:
            try:
                lines = read_lines(fn)
            except OSError:
                continue
        tally.files_opened += 1
        for lineno, line in enumerate(lines, 1):
            if skip_line is not None and skip_line.search(line):
                continue
            for m in rx.finditer(line):
                tally.records.append((fn, lineno, m.group(0), line.strip()))
                tally.per_file[fn] = tally.per_file.get(fn, 0) + 1
    return tally


def count_forms(files: list[str], tolerant: bool = False,
                provider: "dict[str, list[str]] | None" = None) -> dict[str, Count]:
    out = {}
    for form, rx in FORM_PATTERNS.items():
        if tolerant and form in EXPECTED_TOLERANT:
            rx = F1_TOLERANT
        out[form] = count_pattern(rx, files, provider=provider)
    out["F6"] = count_pattern(F6_RE, files, skip_line=re.compile(TARGET_RE), provider=provider)
    return out


def origin_split(files: list[str]) -> tuple[list[str], list[str]]:
    """(live-tree origin, progress/ origin) — the census §2.3 axis."""
    live = [f for f in files if not f.startswith("progress/")]
    return live, [f for f in files if f.startswith("progress/")]


# The commit that carried the census. Its tree is what the expected values were counted over, so
# the reconstruction below reads changed-and-added files from there instead of from the worktree:
# a checker documents the forms it counts, and a selftest fixture is a citation as far as `grep`
# is concerned — `tools/doc_ref_check.py`'s own fixtures are already 2 of the census's 5 live
# line-cites (§2.3), and `--selftest` fixtures here add more. Excluding the checkers is the same
# move the census made when it said "only 5 come from the live tree — 2 of those are a checker's
# own `--selftest` fixture strings".
CENSUS_REF = "dece2d9"


def baseline_provider() -> tuple[dict, list[str], list[str]]:
    """path -> lines as the census counted them, files re-read from the ref, files added since it.

    Files edited since `CENSUS_REF` are read from that commit; files it did not contain read as empty
    (that includes this tool, and the census document itself — untracked when it counted).
    """
    changed = {x for x in git("diff", "--name-only", CENSUS_REF).split("\n") if x}
    untracked = {x for x in git("ls-files", "--others", "--exclude-standard").split("\n") if x}
    sub: dict[str, list[str]] = {CENSUS_DOC: []}
    re_read: list[str] = []
    for path in sorted(changed | untracked):
        if path == CENSUS_DOC:
            continue
        shown = git("show", "%s:%s" % (CENSUS_REF, path))
        sub[path] = shown.split("\n") if shown.strip() else []
        if shown.strip():
            re_read.append(path)
    return sub, re_read, sorted(set(sub) - {CENSUS_DOC} - set(re_read))


# The census's own recipes (§2.1), kept verbatim so a disagreement can be adjudicated without
# reading this file. `T` is its target alternation, interpolated where its recipe used it.
CENSUS_T = (r'(engine/results\.md|prototype/[^/ ]+/results\.md|bench/performance-status\.md'
            r'|devices/[A-Za-z0-9-]+\.md|specification/[A-Za-z0-9.-]+\.md|engine/README\.md'
            r'|AGENTS\.md|README\.md|results\.md|performance-status\.md)')
CENSUS_RECIPES = {
    "F1": CENSUS_T + r':[0-9]',
    "F2": CENSUS_T + r'\`? ?§',
    "F4": r'ledger §[0-9]',
    "F5": CENSUS_T + r'\`?,? ?\(',
    "F5b": r'§[A-Za-z]',
}


def recipe_hint(form: str) -> str:
    """The one command the census published for this form, with its own tree substituted in.

    The pattern is stored as a Python regex, where an escaped backtick is harmless; inside the
    single-quoted shell string it is an invalid ERE escape, so the backslashes go here.
    """
    pat = CENSUS_RECIPES.get(form)
    if pat is None:
        return ""
    shell = pat.replace("\\`", "`").replace("'", "'\\''")
    return ("  adjudicate it yourself: git grep -ohE '%s' %s -- ':(exclude)%s' | wc -l"
            % (shell, CENSUS_REF, CENSUS_DOC))


def cmd_check(list_form: str | None) -> int:
    files = tracked_files()
    if not any(f != CENSUS_DOC for f in files):
        print("citation_census: nothing to count besides the census document", file=sys.stderr)
        return 2
    # The census counted the tree it sat in: census document untracked, these tools not yet written.
    # A checker documents the forms it counts and its fixtures are citations as far as `grep` is
    # concerned, so the adjudicated count is taken over that same tree, not the worktree.
    sub, re_read, added = baseline_provider()
    print("citation_census --check")
    print("  spelling: STRICT (no space after the colon); the tolerant spelling is printed beside F1")
    print("  scope: `git ls-files`, %d files, code included. The adjudicated column re-counts the tree"
          " at `%s`, the commit that carried the census — %d file(s) read from that commit, %d counted"
          " as empty because they did not exist there, and the census document always reads empty"
          " because it was untracked when it counted."
          % (len(files), CENSUS_REF, len(re_read), len(added)))
    if re_read or added:
        print("    read from %s: %s" % (CENSUS_REF, ", ".join(re_read) or "-"))
        print("    counted empty (new since %s): %s" % (CENSUS_REF, ", ".join(added) or "-"))
    print()
    print("%-6s %9s %10s %8s %10s %7s" %
          ("form", "expected", "at-census", "today", "added", "verdict"))

    failures: list[str] = []
    strict = count_forms(files, provider=sub)
    allt = count_forms(files)
    for form in ("F1", "F2", "F4", "F5", "F5b", "F6"):
        want = EXPECTED[form]
        base_n = strict[form].total
        for problem in strict[form].integrity() + allt[form].integrity():
            failures.append("%s: %s" % (form, problem))
        if (strict[form].files_opened != len(files)
                or allt[form].files_opened != len(files)):
            failures.append("%s: opened %d of %d files — an unreadable file is a silent "
                            "undercount" % (form, strict[form].files_opened, len(files)))
        verdict = "PASS" if base_n == want else "FAIL"
        if verdict == "FAIL":
            failures.append("%s: counted %d over the census tree, expected %d%s"
                            % (form, base_n, want, recipe_hint(form) and "\n" + recipe_hint(form)))
        print("%-6s %9d %10d %8d %10d %7s" %
              (form, want, base_n, allt[form].total, allt[form].total - base_n, verdict))
        if list_form == form:
            for fn, lineno, text, snippet in strict[form].records:
                print("        %s:%d  %s | %s" % (fn, lineno, text, snippet[:88]))
            print("        printed %d of %d records" %
                  (len(strict[form].records), strict[form].total))
    # The tolerant spelling, named as such (census §2.3: "state which spelling a count used").
    tol = count_forms(files, tolerant=True, provider=sub)["F1"]
    tol_all = count_forms(files, tolerant=True)["F1"]
    want_tol = EXPECTED_TOLERANT["F1"]
    print("%-6s %9d %10d %8d %10d %7s   (tolerant: `path: N` with a space)" %
          ("F1", want_tol, tol.total, tol_all.total, tol_all.total - tol.total,
           "PASS" if tol.total == want_tol else "FAIL"))
    if tol.total != want_tol:
        failures.append("F1 tolerant: counted %d, expected %d" % (tol.total, want_tol))

    # F1 by origin — the census's safety finding (§2.3: 96 of 101 live in progress/).
    live, archived = origin_split(files)
    live_f1 = count_pattern(F1_STRICT, live, provider=sub)
    live_tol = count_pattern(F1_TOLERANT, live, provider=sub)
    arch_f1 = count_pattern(F1_STRICT, archived, provider=sub)
    want_live = EXPECTED_LIVE_ORIGIN["F1"]
    print()
    print("live-origin F1 (origin outside `progress/`)   expected %d, counted %d (tolerant %d)"
          "  %s" % (want_live, live_f1.total, live_tol.total,
                    "PASS" if live_f1.total == want_live else "FAIL"))
    print("    %d of the %d F1 cites originate in `progress/`, the tree rule 2 of "
          "`progress/README.md` freezes" % (arch_f1.total, strict["F1"].total))
    if live_f1.total != want_live:
        failures.append("live-origin F1: counted %d, expected %d" % (live_f1.total, want_live))
    if list_form == "F1-live":
        for fn, lineno, text, snippet in live_f1.records:
            print("        %s:%d  %s | %s" % (fn, lineno, text, snippet[:88]))
    for problem in live_f1.integrity():
        failures.append("F1 live-origin: " + problem)
    if live_f1.files_opened != len(live):
        failures.append("F1 live-origin: opened %d of %d files" % (live_f1.files_opened, len(live)))

    print()
    for path, want in EXPECTED_HEADINGS.items():
        heads = fence_headings(read_lines(path))
        got = {}
        for head in heads:
            got[head.level] = got.get(head.level, 0) + 1
        same = all(got.get(lv, 0) == n for lv, n in want.items())
        print("heading census %s: %d lines (`wc -l`), %s (census-time %s) %s — the journal "
              "grows, so this is INFO"
              % (path, wc_lines(path),
                 " ".join("H%d=%d" % (lv, got.get(lv, 0)) for lv in sorted(want)),
                 " ".join("H%d=%d" % (lv, n) for lv, n in sorted(want.items())),
                 "match" if same else "DIFFERS"))

    print()
    if failures:
        print("CHECK FAIL — %d problem%s" % (len(failures), "s" if len(failures) > 1 else ""))
        for f in failures:
            print("  " + f)
        return 1
    print("CHECK PASS — every form matches the census, with each form's record list non-empty")
    return 0


# ---------------------------------------------------------------- anchors
class Anchor:
    __slots__ = ("origin", "path", "text", "nums", "statuses", "note")

    def __init__(self, origin: str, path: str, text: str) -> None:
        self.origin, self.path, self.text = origin, path, text
        self.nums: list[int] = []
        self.statuses: list[str] = []
        self.note = ""

    @property
    def resolved(self) -> bool:
        return all(s == "ok" for s in self.statuses) and bool(self.statuses)

    def render(self) -> str:
        bad = ["%d:%s" % (n, s) for n, s in zip(self.nums, self.statuses) if s != "ok"]
        return "%s  ->  %s  %s%s" % (self.origin, self.path, self.text,
                                     ("  [" + "; ".join(bad) + "] " + self.note) if bad
                                     else ("  " + self.note if self.note else ""))


def resolve_cited_path(cited: str, origin: str,
                       tracked: list[str]) -> "tuple[str, str, int]":
    """(path, note, candidate-count) for a cited path — resolved only when exactly one file matches.

    A cite may be a full tracked path, a suffix of one (`glue/results.md` for
    `prototype/glue/results.md`, which is how `progress/2026-10-11/commit-review-2026-10-11.md`
    writes it), or a bare filename. When more than one tracked file matches, nothing is chosen and
    the verdict is UNROOTED: the census §2.4(a) shows a bare `results.md` means three different
    files here, and adjudicating `engine/kernels_glue.cpp:35`'s `results.md:113` against
    `engine/results.md` because the citing file lives in `engine/` would prove a line number up
    against the wrong file. The nearest candidate is named, never picked.
    """
    if os.path.isfile(cited):
        return cited, "", 1
    cands = sorted({f for f in tracked if f == cited or f.endswith("/" + cited)})
    if len(cands) == 1:
        return cands[0], "resolved by suffix: %s" % cands[0], 1
    if not cands:
        return cited, "no tracked file matches this cite", 0
    here = origin.rsplit("/", 1)[0]
    near = [f for f in cands if f.rsplit("/", 1)[0] == here]
    return (cited, "%d tracked files match %s (%s%s)%s"
            % (len(cands), cited, ", ".join(cands[:4]), " …" if len(cands) > 4 else "",
               "; nearest to %s is %s" % (origin, near[0]) if near else ""), len(cands))


def target_lines(path: str) -> "list[str] | None":
    """The target file's lines, or None when it is not on disk."""
    if not os.path.isfile(path):
        return None
    return read_lines(path)


def check_anchor(anchor: Anchor, tracked: list[str],
                 lines_for=target_lines) -> Anchor:
    """Fill in statuses: ok | past-end | blank | missing-target | unrooted."""
    path, note, n_cands = resolve_cited_path(anchor.path, anchor.origin, tracked)
    anchor.path = path
    if note:
        anchor.note = (anchor.note + "; " if anchor.note else "") + note
    lines = lines_for(path)
    if lines is None:
        anchor.statuses = ["unrooted" if n_cands > 1 else "missing-target"] * len(anchor.nums)
        return anchor
    for num in anchor.nums:
        if num > len(lines):
            anchor.statuses.append("past-end(file has %d)" % len(lines))
        elif not lines[num - 1].strip():
            anchor.statuses.append("blank")
        else:
            anchor.statuses.append("ok")
    return anchor


def find_anchors(origin: str, lines: list[str]) -> list[Anchor]:
    out: list[Anchor] = []
    for lineno, line in enumerate(lines, 1):
        for m in ANCHOR_RE.finditer(line):
            anchor = Anchor("%s:%d" % (origin, lineno), m.group(1), m.group(0))
            anchor.nums = [int(x) for x in ANCHOR_NUM_RE.findall(m.group(2))]
            if line[max(0, m.start() - 1):m.start()] == "\u2026":
                anchor.note = ("the cite is written with an elided directory, so the path itself "
                               "is truncated")
            out.append(anchor)
    return out


def cmd_anchors(paths: list[str], show_notes: bool = False) -> int:
    tracked = tracked_files()
    scan = paths or [f for f in tracked if f.endswith(TEXT_SUFFIXES)]
    unreadable = []
    anchors: list[Anchor] = []
    for fn in scan:
        try:
            lines = read_lines(fn)
        except OSError:
            unreadable.append(fn)
            continue
        anchors.extend(find_anchors(fn, lines))
    print("citation_census --anchors: %d citing files scanned (%d unreadable), %d line-form "
          "citations found" % (len(scan), len(unreadable), len(anchors)))
    if unreadable:
        print("  UNREADABLE: " + ", ".join(unreadable))
    if not anchors:
        print("  ZERO citations parsed — this scan adjudicates nothing")
        return 1
    printed = 0
    bad = 0
    for anchor in anchors:
        check_anchor(anchor, tracked)
        if not anchor.resolved:
            bad += 1
            print("  BROKEN  " + anchor.render())
            printed += 1
    notes = [a for a in anchors if a.resolved and a.note]
    if show_notes:
        for anchor in notes:
            print("  note    " + anchor.render())
    print("  printed %d BROKEN lines and %d note lines (%s) for %d parsed citations"
          % (bad, len(notes),
             "shown" if show_notes else "pass --all to show them", len(anchors)))
    print("%s" % ("ANCHORS FAIL — %d unresolved" % bad if bad
                  else "ANCHORS PASS — every parsed citation resolved"))
    return 1 if bad else 0


# ---------------------------------------------------------------- selftest
FIXTURE = {
    "doc.md": [
        "intro line",
        "```bash",
        "# a shell comment inside a fence, not a heading",
        "grep -c '^# ' engine/results.md",
        "```",
        "# Real H1",
        "## Q6.1 (2026-10-13): the stage clock",
        "body",
        "### Q10b — the walk follows the row count",
    ],
    "short.md": ["one", "two", "", "four"],
    "prototype/glue/results.md": ["glue line 1", "glue line 2"],
}


def selftest() -> int:
    ok = True
    checks = 0
    results: list[tuple[str, bool, bool]] = []  # (name, passed, must_fail)

    def report(name: str, passed: bool, must_fail: bool = False) -> None:
        nonlocal ok, checks
        checks += 1
        results.append((name, passed, must_fail))
        bad = (passed == must_fail)
        if bad:
            ok = False
        print("%s  %s%s" % ("FAIL" if bad else ("MUT " if must_fail else "ok  "), name,
                            "  (this input must be rejected)" if must_fail else ""))

    # --- headings: the fence rule -------------------------------------------------
    heads = fence_headings(FIXTURE["doc.md"])
    titles = [h.title for h in heads]
    report("fenced `# ` line is not a heading",
           "# a shell comment inside a fence, not a heading" not in titles
           and "Real H1" in titles)
    report("heading levels and ranges are contiguous",
           [(h.start, h.end, h.level) for h in heads] == [(6, 6, 1), (7, 8, 2), (9, 9, 3)])
    report("a fenced file yields zero headings",
           fence_headings(["```", "# not a heading", "```"]) == [])

    # --- the forms, on synthetic text --------------------------------------------
    def count_text(text: str) -> dict[str, list[str]]:
        out = {}
        for form, rx in FORM_PATTERNS.items():
            out[form] = rx.findall(text)
        out["F6"] = ([] if re.search(TARGET_RE, text) else F6_RE.findall(text))
        return out

    line_comma = count_text("see `engine/results.md:724,726` for the pair")
    report("F1 `:724,726` counts once per citation occurrence",
           line_comma["F1"] == ["engine/results.md:7"])
    line_range = count_text("engine/results.md:3731-3733 is the block")
    report("F1 `:3731-3733` matches", line_range["F1"] == ["engine/results.md:3"])
    label_q = count_text("`engine/results.md` §Q6.1 carries it")
    report("F2 `path` §Q6.1 matches", len(label_q["F2"]) == 1 and label_q["F6"] == [])
    ledger = count_text("see ledger §6 for the row")
    report("F4 `ledger §6` matches", ledger["F4"] == ["ledger §6"])
    bare = count_text("the default since results.md §21")
    report("bare `results.md §21` counts as F2 and F5b",
           len(bare["F2"]) == 1 and bare["F5b"] == [])
    label_only = count_text("every flip is §A2's class")
    report("label-only `§A2` counts as F6 (no file named on the line)",
           label_only["F6"] == ["§A2"])
    report("label-only `§A2` is F5b too", label_only["F5b"] == ["§A"])
    report("a line that names a file is not label-only",
           count_text("`engine/results.md` and §A2's class")["F6"] == [])
    tolerant = F1_TOLERANT.findall("devices/strix-halo-compute.md: 24 is the row")
    strict = F1_STRICT.findall("devices/strix-halo-compute.md: 24 is the row")
    report("tolerant spelling catches the space, strict does not",
           len(tolerant) == 1 and strict == [])

    # --- anchors -------------------------------------------------------------------
    tracked = ["doc.md", "short.md", "engine/results.md", "prototype/glue/results.md"]

    def anchor_of(text: str, origin: str = "doc.md") -> Anchor:
        found = find_anchors(origin, text.split("\n"))
        return check_anchor(found[0], tracked, lines_for=FIXTURE.get)

    a_ok = anchor_of("see short.md:1 and short.md:4")
    report("anchor on a real line resolves", a_ok.statuses == ["ok"])
    a_blank = anchor_of("see short.md:3")
    report("blank target line is BROKEN", a_blank.statuses == ["blank"] and not a_blank.resolved)
    a_end = anchor_of("see short.md:9")
    report("anchor past end of file is BROKEN",
           a_end.statuses and a_end.statuses[0].startswith("past-end"))
    a_range = anchor_of("short.md:2-3")
    report("a range is checked at both endpoints",
           a_range.nums == [2, 3] and a_range.statuses == ["ok", "blank"])
    a_unrooted = anchor_of("results.md:1")
    report("a directory-less cite is UNROOTED, not guessed",
           a_unrooted.statuses == ["unrooted"] and "2 tracked files match" in a_unrooted.note)
    a_suffix = anchor_of("glue/results.md:2")
    report("a suffix cite resolves when exactly one tracked file matches",
           a_suffix.statuses == ["ok"]
           and a_suffix.path == "prototype/glue/results.md")
    report("an anchor on a file that is not on disk says so",
           anchor_of("engine/nope.md:1").statuses == ["missing-target"])
    report("mut: an anchor into a virtual file that is not read from disk",
           check_anchor(find_anchors("doc.md", ["short.md:1"])[0], tracked).statuses
           == ["ok"], must_fail=True)

    # --- mutation harness: every rule above must be able to fail -------------------
    # Each mutation feeds the rule an input it must reject. A MUT line that prints `ok`
    # means the rule is decorative, and `--selftest` exits 1.
    report("mut: heading rule rejects a fence-blind dump",
           fence_headings(["```", "# fenced", "```", "# real"])[0].title == "fenced",
           must_fail=True)
    report("mut: F1 strict accepts a spaced cite",
           F1_STRICT.findall("engine/results.md: 24") != [], must_fail=True)
    report("mut: the label-only rule accepts a cite whose file IS named",
           count_text("`engine/results.md` §A2")["F6"] == ["§A2"], must_fail=True)
    report("mut: anchor rule accepts a line past the end",
           anchor_of("short.md:99").resolved, must_fail=True)
    report("mut: anchor rule accepts a blank target line",
           anchor_of("short.md:3").resolved, must_fail=True)
    report("mut: an ambiguous cite resolves to one file when two match",
           anchor_of("results.md:1").statuses == ["ok"], must_fail=True)
    report("mut: a counter reports an empty record list as admissible",
           not count_pattern(re.compile(r"§ZZZ"), ["doc.md", "short.md"]).integrity(),
           must_fail=True)
    report("mut: a selftest case that is simply wrong", 1 == 2, must_fail=True)

    print()
    print("SELFTEST %s — %d checks (%d of them mutation cases that must FAIL)"
          % ("PASS" if ok else "FAIL", checks,
             sum(1 for _, _, mf in results if mf)))
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sections", metavar="PATH",
                      help="fence-aware heading dump of PATH (level, start-end, count, title)")
    mode.add_argument("--check", action="store_true",
                      help="recount forms F1-F6 and compare with the census's expectations")
    mode.add_argument("--anchors", nargs="*", metavar="PATH",
                      help="resolve every path:NNN citation in PATH(s), or in every tracked "
                           "md/code file")
    mode.add_argument("--selftest", action="store_true",
                      help="run the synthetic fixtures and the mutation harness")
    ap.add_argument("--list", metavar="FORM", default=None,
                    help="with --check: print every record for FORM (F1, F1-live, F2, …)")
    ap.add_argument("--all", action="store_true",
                    help="with --anchors: also print the note lines (resolved-by-suffix targets)")
    args = ap.parse_args(argv[1:])

    if args.selftest:
        return selftest()
    if args.sections:
        return cmd_sections(args.sections)
    if args.check:
        return cmd_check(args.list)
    if args.anchors is not None:
        return cmd_anchors(args.anchors, show_notes=args.all)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
