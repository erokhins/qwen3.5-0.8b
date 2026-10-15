#!/usr/bin/env python3
"""Every path cited by a tracked *.md must exist — the stale-reference checker.

Why this exists: the docs in this repo are the command lists. A sentence like
"reproduce with `speccheck/run.py --keep-f32-scalars`" is only worth what the
file behind it is worth, and `progress/2026-10-16/cleanup-audit-2026-10-16.md` §4 had to
hand-tabulate ten such paths. This is that tabulation, as a script, so the next
audit is one command instead of a pass.

Method (CPU only, no device, no build):

1. Read every tracked markdown file (`git ls-files '*.md'`).
2. Tokenize every repo-rooted path-shaped token — roots are the real top-level
   directories, so prose like "the engine" cannot match.
3. Classify each distinct path against the git index, the filesystem, and
   `.gitignore` (`git check-ignore`), then report.

Two scanner bugs fixed here, both from the part-1 audit (§1 finding 1 of
`progress/2026-10-16/cleanup-audit-2026-10-16.md`). A path extension is matched as an
alternation, and a regex alternation is first-match-wins, so `o` before `out`
recorded `progress/bench/evidence/p9/stage.out` as a missing `tmp/p9/stage.o`, and `json` before
`jsonl` recorded tracked `bench/band-history.jsonl` as a missing
`bench/band-history.json`. Neither was a doc error; two of thirteen reported
MISSING rows were phantoms, and the real file went uncommitted. The fix is
twice over: `EXTS` is sorted longest-first, and the extension is followed by
`(?![A-Za-z0-9_])` so a truncated extension cannot match at all. `--selftest`
asserts both cases and their reversal; run it after touching `EXTS`.

Buckets, and why each one is separate: a flat "missing" list would be noise,
because three of these paths are *supposed* to be absent on a clean tree.

  TRACKED            in the git index. Nothing to do.
  PRESENT-UNTRACKED  on disk, ignored by .gitignore. The cited-provenance trap:
                     `data/` and `models/` belong here, and so do the tmp/ and
                     logs/ files the part-1 audit had to rescue.
  MISSING-REGENERABLE  absent, and ignored by policy (data/, models/, build
                     output, .venv). Re-made by a documented command; a cite
                     here is normal, not stale.
  MISSING-SCRATCH    absent, under tmp/ or logs/, with a file extension. Either
                     transient sidecar or genuinely lost evidence — the class
                     that `8c81e21` purged and `route_wide520f.json` died in.
  MISSING-STALE      absent and NOT ignored, i.e. it would be committable. This
                     is the real signal: a renamed or never-built file. Each
                     row carries the nearest tracked path as a candidate.
  SHORTHAND          extensionless, absent, and either resolvable as `<path>.md`
                     or a scratch/output directory a documented command creates
                     (`tmp/gate_logs`). Not counted broken.

Exit status: 0 if nothing is broken, 1 if anything lands in a MISSING bucket, in BROKEN-ANCHOR or
in BROKEN-LABEL (or 2 if a check matched nothing at all and so adjudicates nothing).

Three additions made by the 2026-10-16 archive review (census §5, row S2), because path existence
alone let two whole classes of dead citation through:

1. LINE ANCHORS. `path:NNN`, `path:NNN-NNN`, `path:NNN,NNN` and the spaced spelling `path: NNN`
   resolve against the target: BROKEN if NNN is past the file's end, or if the line it names is
   blank. Before this, `engine/results.md:3739` was checked as `engine/results.md` — a file that
   will exist forever, which is a check that cannot fail. A range is checked at both endpoints.
2. SECTION LABELS. A `` `path` §L `` cite must match a fence-aware heading in the target that
   carries that label: numeric (`§6`), letter (`§A1`, `§Q6.1`), mixed (`§1c.1`), a `§7-§9` range
   (both endpoints), a quoted title (`§"Row corrections after the first attempt"`, matched against
   heading text), with leading zeros folded so `§07.4` and `§7.4` are one label. `§Q99` used to
   pass vacuously; it now fails. A label with no file named anywhere on its line — census form F6,
   which is how `engine/gate.py` carries 17 pointers into an archived audit — is reported as
   LABEL-UNNAMED: named, counted, grouped by citing file, and NOT counted broken, because nothing
   in the line says which file owns it. A label whose target is not a markdown document (a header
   cited as `engine/prefill.h`, §06.1) is neither broken nor resolved: it is not adjudicable, and
   it is counted separately rather than silently dropped.
3. CODE COUNTS. `--include-code` is the default and adds tracked `*.py *.cpp *.h *.sh`, because
   live code carries citations (`engine/gate.py`'s 17 `§A` labels, `engine/kernels.h`'s `§Q10b`);
   `--no-include-code` reproduces the markdown-only sweep this tool used to run.

Gaps recorded rather than papered over. A cite with no directory (`results.md §21`, `results.md:113`)
moves path and label patterns alike, because the bare filename means three different files in this
tree — `tools/citation_census.py --anchors` resolves those by unique suffix and reports the
ambiguous ones. Assumption ids (`A-19`, `A-32`) are ids, not paths, so this tool cannot see them.

Usage:
  python3 tools/doc_ref_check.py                   # summary report, markdown + code
  python3 tools/doc_ref_check.py --list-broken     # broken paths, then tagged anchor/label rows
  python3 tools/doc_ref_check.py --all             # also list PRESENT-* detail
  python3 tools/doc_ref_check.py --no-include-code # markdown only, as before the census
  python3 tools/doc_ref_check.py --selftest        # scanner regression (path + anchor + label)
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
from collections import defaultdict

# Repo-rooted path prefixes. Only these can start a citation, which is what keeps
# prose ("the engine/", "in tools/") from being read as a path.
# `progress/` was added by the 2026-10-16 reorganization (dated one-shot docs and
# committed evidence); `thinking/` stays listed although the directory is gone, so a
# citation to it is still reported rather than silently ignored.
ROOTS = (
    "bench", "data", "devices", "engine", "engines", "logs", "models",
    "progress", "prototype", "speccheck", "specification", "testdata",
    "thinking", "tmp", "tools",
)

# Extension alternation. LONGEST FIRST on purpose — see the module docstring.
# A new extension goes here in length order, and `--selftest` is then re-run.
EXTS = (
    "safetensors", "markdown", "ipynb", "cmake", "ninja", "patch", "svg",
    "jsonl", "json5", "json", "msgpack", "gguf", "onnx", "pyc", "toml",
    "yaml", "yml", "bash", "html", "htm", "cpp", "cxx", "hpp", "hip",
    "log", "txt", "ids", "csv", "tsv", "npy", "npz", "bin", "dat",
    "tmp", "pid", "md5", "out", "md", "py", "sh", "diff", "o", "h", "c",
)
# Sorted longest-first at import, so source order cannot re-introduce the bug.
EXTS_SORTED = sorted(set(EXTS), key=lambda e: (-len(e), e))

# A citation is one of three shapes, all anchored on a known root so that prose
# cannot match. `_SEG` excludes `/` so the root anchor cannot drift mid-token.
_ROOT = "(?:" + "|".join(ROOTS) + ")"
# Dots are allowed inside a segment so a compound name (`gdn_decode.hip.cpp`,
# `model.safetensors.index.json`) is read as one path, not truncated at its
# first dot.
_SEG = r"[\w.+-]+"
# `engine/results.md`, `specification/05-gated-full-attention.md`, `x.jsonl`.
# The extension is NOT part of the captured path; it is required and then
# guarded, so a truncated extension (`json` inside `jsonl`, `o` inside `out`)
# cannot match even if `EXTS_SORTED` is ever re-sorted wrongly.
EXT_CITE_RE = re.compile(
    r"(?<![\w./~-])((?:(?:%s)(?:/%s)*)\.(?:%s))(?![\w])(?!\.\w)"
    % (_ROOT, _SEG, "|".join(re.escape(e) for e in EXTS_SORTED)))
# `tmp/q10`, `prototype/gdn_chunk`, `bench/plan-2026-10-12` (a .md shorthand).
# The trailing guards keep it from matching a stem: `tmp/p9/stage` must not be
# reported beside `progress/bench/evidence/p9/stage.out`.
BARE_CITE_RE = re.compile(
    r"(?<![\w./~-])((?:%s)(?:/%s)+)(?![\w+-])(?!\.\w)" % (_ROOT, _SEG))
# `engine/prefill.{h,cpp}`, `state.{h,cpp}` — brace shorthand, expanded to the
# files it names so it is never reported as one missing path.
BRACE_CITE_RE = re.compile(
    r"(?<![\w./~-])((?:%s)(?:/%s)*)\.\{([\w,]+)\}" % (_ROOT, _SEG))

# ---------------------------------------------------------------- anchors, labels
# `path:NNN`, `path:NNN-NNN`, `path:NNN,NNN`, and the tolerant spaced spelling `path: 24` (the one
# live cite that uses it: `prototype/attn_decode/wave_model.py:51`). The path part is exactly the
# shape `EXT_CITE_RE` matches, so a citation cannot be read as a path in one place and as an anchor
# in another. A path that is absent from disk is NOT checked here: the path buckets already report
# it, and a missing file needs one finding, not one per line number it was cited by.
LINE_ANCHOR_RE = re.compile(
    r"(?<![\w./~-])((?:(?:%s)(?:/%s)*)\.(?:%s))(: ?\d+(?:[-,] ?\d+)*)(?![\w])"
    % (_ROOT, _SEG, "|".join(re.escape(e) for e in EXTS_SORTED)))
ANCHOR_NUMBERS_RE = re.compile(r"\d+")

# A section-sign cite whose path sits next to it: `` `path` §6 ``, `` `path` §Q6.1 ``,
# `path §21`, and the range forms the tree writes (`§7-§9`, `§20.1–§20.3`, `§1–2`) plus the
# quoted-title form (`§"Row corrections after the first attempt"`). Adjacency is the point: a line
# that merely *mentions* a path and also carries an unrelated `§route A` cannot be adjudicated, and
# `progress/2026-10-14/speedup-routes-2026-10-14.md:144` is exactly such a line.
LABEL_CITE_RE = re.compile(
    r"((?:(?:%s)(?:/%s)*)\.(?:%s))`?,?? ?§([A-Za-z0-9][\w.]*|\"[^\"]+\")"
    % (_ROOT, _SEG, "|".join(re.escape(e) for e in EXTS_SORTED)))
# A range endpoint written after a dash: `§7-§9`, `§20.1–§20.3`, `§1–2`, `§F1–F4`. It must LOOK like
# a label (a number, or letters followed by a number), because a plain em dash is prose: the census
# line `` `prototype/glue/results.md` §3 — filename only `` would otherwise adjudicate `filename`.
RANGE_TAIL_RE = re.compile(r"^\s*[–—-]\s*§?([A-Za-z]\d[\w.]*|\d[\w.]*)")
# A label with no file named anywhere on its line — census form F6, and the reason `engine/gate.py`
# could carry 17 pointers into an archived audit that no checker could see. Restricted to this
# repo's section namespaces (an upper-case letter followed by a number: `§A2`, `§Q6.1`, `§Q10b`,
# `§P8`) so prose pointers (`§route A`, `§expectation 6`, `§N`) are not reported as cites.
UNNAMED_LABEL_RE = re.compile(r"§([A-Z]{1,3}\d[\w.]*)(?![\w])")
ANY_PATH_RE = re.compile(_ROOT + r"/")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
FENCE = "```"
# The label a heading carries: an optional `§` (commit-review writes `## §4 —`), then a number
# (`6.`, `10.1`), a letter+number (`A1`, `Q6.1`, `Q10b`, `P4`), or the mixed forms the p9 docs use
# (`1c.1`) and `0b.`. Everything after it is the title.
HEAD_LABEL_RE = re.compile(
    r"^§?\s*([A-Z]{1,3}\d{1,3}[a-z]?(?:\.\d+)?|\d{1,3}[a-z]?(?:\.\d+)?)(?=[\s.).:—–]|$)")
# Extra labels inside a title, for the headings that name two sections at once
# (`## Q5 / P7:`, `### Q10b/Q10c:`, `### F4's aggregate bands`).
HEAD_EXTRA_RE = re.compile(r"(?<![\w/.])([A-Z]{1,3}\d{1,3}[a-z]?(?:\.\d+)?)(?![\w.])")


# quote chars a label can arrive wrapped in: double, single, and the typographic pair
QUOTE_CHARS = chr(34) + chr(8220) + chr(8221) + chr(39)


def canonical_label(label: str) -> str:
    """`§07.4` and `§7.4` are one label; `§0b` is not `§b`. Case-insensitive."""
    lab = label.strip().strip(QUOTE_CHARS).lower()
    return re.sub(r"^0+(\d)", r"\1", lab)


def normalize_cite_label(raw: str) -> str:
    """Strip what prose glues onto a cite: `§21:`, `§4's`, `§20.1/P8`, `§Q10b.`."""
    lab = raw.strip().strip(")]}")
    lab = re.sub(r"'s$", "", lab)
    lab = lab.rstrip(".,:;!?—–-")
    return lab.split("/")[0]      # `3/P7`: the section is `3`; `P7` is a sub-tag, not a heading


def fence_headings(lines: list[str]) -> list[tuple[int, str]]:
    """(level, title) for every heading outside a ``` fence.

    Fence-awareness is not decoration: 72 of the 86 lines matching `^# ` in `engine/results.md` are
    shell comments inside fenced command blocks (`# cmd: .venv/bin/python engine/gate.py …`), so a
    fence-blind dump invents 72 sections and then adjudicates a label against text that is not a
    heading at all.
    """
    out: list[tuple[int, str]] = []
    fence = False
    for line in lines:
        if line.lstrip().startswith(FENCE):
            fence = not fence
            continue
        if fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            out.append((len(m.group(1)), m.group(2)))
    return out


def heading_labels(lines: list[str]) -> tuple[set[str], list[str]]:
    """(label set, title list) of a target's headings — what a `§label` cite is checked against."""
    labels: set[str] = set()
    titles: list[str] = []
    for _, title in fence_headings(lines):
        titles.append(title.lower())
        m = HEAD_LABEL_RE.match(title)
        if m:
            labels.add(canonical_label(m.group(1)))
        for extra in HEAD_EXTRA_RE.finditer(title):
            labels.add(canonical_label(extra.group(1)))
    return labels, titles


def label_resolves(label: str, labels: set[str], titles: list[str]) -> bool:
    """Does a target carry this label? A quoted label is matched against heading titles instead."""
    if label.startswith('"'):
        return canonical_label(label) in " \n".join(titles)
    return canonical_label(label) in labels


def target_lines(path: str, cache: dict) -> "list[str] | None":
    """A cited target's lines, cached; None when it is not on disk."""
    if path not in cache:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                cache[path] = fh.read().split("\n")
        except OSError:
            cache[path] = None
    return cache[path]


def ref_findings(pairs, lines_for=target_lines, cache=None) -> dict:
    """Line-anchor and section-label verdicts for (citing_file, lineno, line) triples.

    Six result classes, and the split between them is the honest part:

      anchor_broken       a number past the target's end, or landing on a blank line
      label_broken        the target has headings and carries no such label (`§Q99`)
      label_unadjudicable the target is not a markdown doc, or has no headings at all
      label_unnamed       a `§A2`-shaped label with no file named anywhere on the line (census F6)
      label_nonadjacent   a label on a line that names a path, but not next to the label
      anchor_ok / label_ok  the counts that did resolve — the witness that the check ran
    """
    cache = {} if cache is None else cache
    out = {"anchor_ok": 0, "label_ok": 0, "label_unadjudicable": 0, "label_nonadjacent": 0,
           "anchor_broken": [], "label_broken": [], "label_unnamed": [], "label_unrooted": []}
    for origin, lineno, line in pairs:
        where = "%s:%d" % (origin, lineno)
        for m in LINE_ANCHOR_RE.finditer(line):
            path = m.group(1)
            lines = lines_for(path, cache)
            if lines is None:
                continue            # absent target: a path finding, already reported elsewhere
            for num in (int(x) for x in ANCHOR_NUMBERS_RE.findall(m.group(2))):
                if num > len(lines):
                    out["anchor_broken"].append(
                        "%s  %s  line %d is past the end (%s has %d lines)"
                        % (where, m.group(2), num, path, len(lines)))
                elif not lines[num - 1].strip():
                    out["anchor_broken"].append(
                        "%s  %s  line %d of %s is blank — the cite points at nothing"
                        % (where, m.group(2), num, path))
                else:
                    out["anchor_ok"] += 1
        adjudicated: list[tuple[int, int]] = []
        for m in LABEL_CITE_RE.finditer(line):
            path, raw = m.group(1), normalize_cite_label(m.group(2))
            adjudicated.append((m.start(2) - 1, m.end(2)))
            tail = RANGE_TAIL_RE.match(line[m.end():])
            labels = [raw] + ([normalize_cite_label(tail.group(1))] if tail else [])
            if tail:
                # The second endpoint of `§7-§9` is adjudicated here, so it must not also be
                # counted as a label that names no file.
                adjudicated.append((m.end() + tail.start(1) - 1, m.end() + tail.end(1)))
            lines = lines_for(path, cache)
            if lines is None or not path.endswith(".md"):
                # A `§label` addresses a *document* section. `engine/prefill.h`, §06.1) in
                # `engine/gate.py:118` cites the spec, not the header, and a header's `#` comments
                # are not headings — so a non-markdown target is not adjudicable, at either branch.
                out["label_unadjudicable"] += len(labels)
                continue
            labels_in_target, titles = heading_labels(lines)
            if not labels_in_target and not titles:
                out["label_unadjudicable"] += len(labels)
                continue
            for lab in labels:
                if label_resolves(lab, labels_in_target, titles):
                    out["label_ok"] += 1
                else:
                    out["label_broken"].append(
                        "%s  `%s` §%s — no heading in %s carries that label (%d headings, %d "
                        "labels)" % (where, path, lab, path, len(fence_headings(lines)),
                                     len(labels_in_target)))
        if not ANY_PATH_RE.search(line):
            for m in UNNAMED_LABEL_RE.finditer(line):
                out["label_unnamed"].append("%s  §%s — no file named on the line"
                                            % (where, m.group(1)))
        else:
            for m in UNNAMED_LABEL_RE.finditer(line):
                if not any(s <= m.start() < e for s, e in adjudicated):
                    out["label_nonadjacent"] += 1
    return out


# Absent under one of these is the intended state: regenerable by a documented
# command, and excluded from git on purpose (part-1 audit §3).
SCRATCH_ROOTS = ("tmp/", "logs/")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def tracked_files() -> set[str]:
    return set(x for x in git("ls-files").splitlines() if x)


def markdown_files() -> list[str]:
    return sorted(x for x in git("ls-files", "-z", "*.md").split("\0") if x)


def code_files() -> list[str]:
    """`*.py *.cpp *.h *.sh`, tracked. Code carries citations: `engine/gate.py` has 17 `§A…`
    labels and `engine/kernels.h` cites `§Q10b`, and a scanner limited to `*.md` cannot see
    either (census §2.1 finding (i))."""
    return sorted(x for x in git("ls-files", "-z", "*.py", "*.cpp", "*.h", "*.sh")
                  .split("\0") if x)


def scanned_files(include_code: bool) -> list[str]:
    return sorted(set(markdown_files()) | (set(code_files()) if include_code else set()))


def ignored_paths(paths: set[str]) -> set[str]:
    """The subset of `paths` that .gitignore would exclude (one batch call)."""
    if not paths:
        return set()
    probe = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        input="\0".join(sorted(paths)) + "\0\0",
        capture_output=True, text=True,
    )
    return {p for p in probe.stdout.split("\0") if p}


def cite_tokens(line: str) -> list[str]:
    """Every path a line cites: brace shorthand expanded, each path once.

    The bare pattern can stop short of a longer citation that shares its prefix
    (`tmp/p9` inside `progress/bench/evidence/p9/stage.out`), and brace shorthand names files rather
    than its stem, so both are filtered against the longer forms found on the
    same line.
    """
    full: list[str] = []
    brace_stems: list[str] = []
    for rx in (EXT_CITE_RE, BRACE_CITE_RE, BARE_CITE_RE):
        for m in rx.finditer(line):
            base = m.group(1)
            if rx is BRACE_CITE_RE:
                brace_stems.append(base)
                full += ["%s.%s" % (base, a)
                         for a in m.group(2).split(",") if a]
            else:
                full.append(base)
    longer = full + [stem + "." for stem in brace_stems]
    kept = [p for p in full
            if not any(q != p and (q.startswith(p + "/") or q == p + ".")
                       for q in longer)]
    # A token ending in `-` or `_` is a glob or a stem the prose froze
    # (`engine/kernels_*.cpp`, `bench/p9-*`), not a path that can be missing.
    kept = [p for p in kept if p[-1] not in "-_.+"]
    out: list[str] = []
    for p in kept:
        if p not in out:
            out.append(p)
    return out


def read_text_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().split("\n")
    except OSError:
        return []


def scan_docs(files: list[str]) -> dict[str, list[str]]:
    """path -> ["file:line", ...] for every citation in `files`."""
    cites: dict[str, list[str]] = defaultdict(list)
    for fn in files:
        for lineno, line in enumerate(read_text_lines(fn), 1):
            for path in cite_tokens(line):
                cites[path].append("%s:%d" % (fn, lineno))
    return cites


def is_filelike(path: str) -> bool:
    """Does the last segment name a file, not a directory, a stem, or a spec
    section? `specification/02` is how docs cite § 02, not a missing file."""
    last = path.rsplit("/", 1)[-1]
    if last.isdigit():
        return False
    return bool(re.search(r"\.[A-Za-z][A-Za-z0-9]*$", last))


def classify(cites: dict[str, list[str]], tracked: set[str]) -> dict[str, dict]:
    ignored = ignored_paths(set(cites) - tracked)
    out: dict[str, dict] = {}
    for path, where in cites.items():
        entry = {"cites": where}
        if path in tracked:
            entry["bucket"] = "TRACKED"
        elif os.path.isdir(path):
            # Git tracks files, not directories, so a cited directory is never
            # "tracked"; `prototype/gdn_chunk` and `tmp/q6` are both fine here.
            entry["bucket"] = "PRESENT-DIR"
        elif os.path.exists(path):
            # On disk but not in git. Ignored is the cited-provenance case
            # (`tmp/`, `logs/`, `data/`); not ignored means `git status` is
            # already showing it, which is a different problem.
            entry["bucket"] = ("PRESENT-IGNORED" if path in ignored
                               else "PRESENT-DIRTY")
        elif not is_filelike(path):
            # A directory cite, a module stem, a spec-section shorthand, or an
            # output directory a documented command creates. Not adjudicable.
            entry["bucket"] = "SHORTHAND"
        elif path.startswith(SCRATCH_ROOTS):
            # Absent evidence under tmp/ or logs/: transient sidecar or lost.
            entry["bucket"] = "MISSING-SCRATCH"
        elif path in ignored:
            # Absent but .gitignore-excluded by policy: `data/` pins, `models/`,
            # build output. A cite here is normal on a clean tree.
            entry["bucket"] = "MISSING-REGENERABLE"
        else:
            # Absent and not ignored: nothing about this tree explains it. Either
            # a rename (a near neighbour exists) or a tool that was never built.
            entry["bucket"] = "MISSING-STALE"
            cands = difflib.get_close_matches(path, sorted(tracked), n=3,
                                              cutoff=0.55)
            if cands:
                entry["near"] = cands
        out[path] = entry
    return out


BROKEN_BUCKETS = ("MISSING-STALE", "MISSING-SCRATCH", "MISSING-REGENERABLE")


def report(classes: dict[str, dict], refs: "dict | None",
           *, show_all: bool, list_broken: bool) -> int:
    buckets: dict[str, list[str]] = defaultdict(list)
    for path, e in classes.items():
        buckets[e["bucket"]].append(path)

    ref_broken = 0 if refs is None else len(refs["anchor_broken"]) + len(refs["label_broken"])

    if list_broken:
        for b in BROKEN_BUCKETS:
            for p in sorted(buckets[b]):
                print(p)
        # The new classes ride along, tagged, so a `--list-broken` diff against an older run
        # IS the delta. The MISSING-bucket rows are unchanged, one path per line as before.
        if refs:
            for kind in ("anchor_broken", "label_broken"):
                for row in refs[kind]:
                    print("[%s] %s" % (kind.upper(), row))
        return 1 if (any(buckets[b] for b in BROKEN_BUCKETS) or ref_broken) else 0

    order = (["TRACKED", "PRESENT-DIR", "PRESENT-IGNORED", "PRESENT-DIRTY"]
             + list(BROKEN_BUCKETS) + ["SHORTHAND"])
    print("doc_ref_check: %d distinct path citations, %d broken"
          % (len(classes), sum(len(buckets[b]) for b in BROKEN_BUCKETS)))
    for b in order:
        print("  %-20s %4d" % (b, len(buckets[b])))
    if refs is not None:
        print("  %-20s %4d   (line anchors: %d resolve, %d broken)"
              % ("ANCHOR-OK", refs["anchor_ok"], refs["anchor_ok"], len(refs["anchor_broken"])))
        print("  %-20s %4d   (section labels: %d resolve, %d broken, %d target has no headings, "
              "%d not adjacent to a path)"
              % ("LABEL-OK", refs["label_ok"], refs["label_ok"], len(refs["label_broken"]),
                 refs["label_unadjudicable"], refs["label_nonadjacent"]))
        print("  %-20s %4d   (F6: a `§A2`-shaped label with no file named on its line)"
              % ("LABEL-UNNAMED", len(refs["label_unnamed"])))

    for b in list(BROKEN_BUCKETS) + (["PRESENT-IGNORED", "PRESENT-DIRTY"]
                                     if show_all else []):
        if not buckets[b]:
            continue
        print("\n%s — %d" % (b, len(buckets[b])))
        for p in sorted(buckets[b]):
            e = classes[p]
            line = "  %s  (%d cite%s: %s)" % (
                p, len(e["cites"]), "" if len(e["cites"]) == 1 else "s",
                ", ".join(e["cites"][:3]) + (" …" if len(e["cites"]) > 3 else ""))
            if e.get("near"):
                line += "\n      nearest tracked: " + ", ".join(e["near"])
            print(line)

    if refs is not None:
        for key, title in (("anchor_broken", "BROKEN-ANCHOR — a line number that does not land"),
                           ("label_broken", "BROKEN-LABEL — a `§label` no heading in the target "
                                            "carries")):
            if not refs[key]:
                continue
            print("\n%s — %d" % (title, len(refs[key])))
            for row in refs[key]:
                print("  " + row)
        if refs["label_unnamed"]:
            by_file: dict[str, list[str]] = defaultdict(list)
            for row in refs["label_unnamed"]:
                origin, _, rest = row.partition("  ")
                by_file[origin.split(":")[0]].append(rest)
            print("\nLABEL-UNNAMED — %d (reported, not counted broken: nothing in the line says "
                  "which file owns the label)" % len(refs["label_unnamed"]))
            for origin in sorted(by_file, key=lambda k: (-len(by_file[k]), k)):
                rows = by_file[origin]
                print("  %-56s %3d  %s" % (origin.split(":")[0], len(rows),
                                           ", ".join(sorted(set(
                                               r.split("—")[0].strip() for r in rows))[:6])))
    return 1 if (any(buckets[b] for b in BROKEN_BUCKETS) or ref_broken) else 0


SELFTEST_CASES = [
    # (text, expected path tokens) — the four audit cases from §1 of
    # `progress/2026-10-16/cleanup-audit-2026-10-16.md`, then the shapes docs really write.
    ("see bench/band-history.jsonl for the bands", ["bench/band-history.jsonl"]),
    ("full text in progress/bench/evidence/p9/stage.out here", ["progress/bench/evidence/p9/stage.out"]),
    ("the probe log is progress/devices/evidence/q8/mixed_bw.log now", ["progress/devices/evidence/q8/mixed_bw.log"]),
    ("was bench/band-history.json once", ["bench/band-history.json"]),
    ("and tmp/p9/stage.o was never built", ["tmp/p9/stage.o"]),
    ("engine/prefill.{h,cpp} are both tracked",
     ["engine/prefill.h", "engine/prefill.cpp"]),
    ("line cite engine/results.md:3739 stays whole", ["engine/results.md"]),
    ("specs cite specification/05-gated-full-attention.md:12",
     ["specification/05-gated-full-attention.md"]),
    ("a bare dir: prototype/gdn_chunk has five scripts", ["prototype/gdn_chunk"]),
    ("trailing slash: write it into tmp/gate_logs/ now", ["tmp/gate_logs"]),
    ("ellipsis dir data/refs/p64/… is a dir cite", ["data/refs/p64"]),
    ("compound suffix prototype/gdn_decode/gdn_decode.hip.cpp",
     ["prototype/gdn_decode/gdn_decode.hip.cpp"]),
    ("a glob is not a path: engine/kernels_*.cpp", []),
    ("a spec section, not a file: specification/02 says",
     ["specification/02"]),
    ("two in one line: tools/gpu_lock.sh and tools/build.sh",
     ["tools/gpu_lock.sh", "tools/build.sh"]),
]


SELFTEST_TARGETS = {
    # A journal-shaped target: H1, an H2 carrying `Q6.1`, a blank line, an H3 carrying `Q10b`.
    "engine/results.md": ["# journal",
                          "## Q6.1 (2026-10-13): the stage clock",
                          "",
                          "### Q10b \u2014 the walk follows the row count",
                          "body"],
    "engine/state.h": ["#ifndef ENGINE_STATE_H", "#define ENGINE_STATE_H"],
    # Every `# ` line here is inside a fence, so the doc has no headings at all: the
    # `grep -c '^# ' engine/results.md` = 86-vs-14 trap, in miniature. The name is deliberately not
    # a real file — see the note under REF_CASES about this tool citing its own fixtures.
    "engine/SELFTEST.md": ["```bash", "# cmd: python3 engine/gate.py", "```", "prose"],
}

# (text, expected class counts, substring that must appear in the reported rows)
# Each case states the COMPLETE verdict vector: `key=count` pairs, and every class not named must be
# zero, so a case cannot pass by under-specifying: `§Q99` must produce exactly one broken label and
# nothing else, and a correct cite must produce zero findings of every kind (the `both fine` case).
# The line numbers are chosen so a case that is broken inside this 5-line fixture still resolves in
# the real `engine/results.md`: this file is tracked, so its own fixture strings are citations as far
# as its scanner is concerned, and a fixture that only fails against itself would print as a finding
# in the shipping tool's own report. The census made the same point about this file when it counted
# 2 of its 5 live line-cites as "a checker's own `--selftest` fixture strings".
REF_CASES = [
    ("see engine/results.md:4 for the row", "anchor_ok=1", None),
    ("an anchor past the end: engine/results.md:7", "anchor_broken=1", "past the end"),
    ("a blank anchored line: engine/results.md:3", "anchor_broken=1", "blank"),
    ("a range checks both endpoints: engine/results.md:3-4",
     "anchor_ok=1 anchor_broken=1", "line 3"),
    ("see `engine/results.md` §Q6.1 for the clock", "label_ok=1", None),
    ("a label that does not exist: `engine/results.md` §Q99", "label_broken=1", "q99"),
    ("trailing punctuation is not part of a label: `engine/results.md` §Q10b:",
     "label_ok=1", None),
    ("a range of labels: `engine/results.md` §Q6.1-§Q10b", "label_ok=2", None),
    ('a title cite: `engine/results.md` §"the stage clock"', "label_ok=1", None),
    # A clean line that resolves on both axes: if a resolver ever degrades to "always broken",
    # this is the case that says so.
    ("engine/results.md:4 and `engine/results.md` §Q6.1 are both fine",
     "anchor_ok=1 label_ok=1", None),
    ("`engine/state.h` §06.1 — a header has no sections", "label_unadjudicable=1", None),
    ("`engine/SELFTEST.md` §cmd — a fenced `# ` line is not a heading",
     "label_unadjudicable=1", None),
    ("the §A2 class, with no file named anywhere", "label_unnamed=1", None),
    ("`engine/results.md` is named here, and §Q6.1 is not next to it",
     "label_nonadjacent=1", None),
    # A recorded gap, asserted so it cannot be mistaken for coverage: a cite with no directory
    # matches neither the ROOT-anchored path pattern nor the label pattern. The census §2.4(a)
    # counts 41+12 anchors of this shape; resolving them needs a basename index, which is
    # `tools/citation_census.py --anchors`' job, not this tool's.
    ("the default since results.md §21", "", None),
]
REF_CLASSES = ("anchor_ok", "anchor_broken", "label_ok", "label_broken",
               "label_unadjudicable", "label_unnamed", "label_nonadjacent")


def ref_verdict(text: str) -> tuple[dict, dict]:
    """The full class-count vector for one synthetic line, plus the raw findings."""
    refs = ref_findings([("t.md", 1, text)],
                        lines_for=lambda path, cache: SELFTEST_TARGETS.get(path))
    counts = {key: (len(refs[key]) if isinstance(refs[key], list) else refs[key])
              for key in REF_CLASSES}
    return counts, refs


def selftest() -> int:
    ok = True
    for text, want in SELFTEST_CASES:
        got = cite_tokens(text)
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            ok = False
        print("%s  %-52s -> %s" % (mark, text, got))

    print()
    for text, spec, needle in REF_CASES:
        want = dict((k, 0) for k in REF_CLASSES)
        for pair in spec.split():
            key, _, n = pair.partition("=")
            want[key] = int(n)
        counts, refs = ref_verdict(text)
        rows = " | ".join(row for key in ("anchor_broken", "label_broken", "label_unnamed")
                          for row in refs[key])
        good = counts == want and (needle is None or needle.lower() in rows.lower())
        if not good:
            ok = False
        print("%s  %-62s -> %s%s" % ("ok  " if good else "FAIL", text,
                                      "as expected" if good else counts,
                                      "" if good else "  rows: " + rows))
    total = len(SELFTEST_CASES) + len(REF_CASES)
    print("\nSELFTEST " + ("PASS" if ok else "FAIL") + " \u2014 %d checks (%d path-token, %d "
          "anchor/label)" % (total, len(SELFTEST_CASES), len(REF_CASES)))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true",
                    help="also list PRESENT-UNTRACKED paths (verbose)")
    ap.add_argument("--list-broken", action="store_true",
                    help="broken paths, one per line, then the anchor/label findings tagged")
    ap.add_argument("--no-include-code", action="store_true",
                    help="scan only tracked *.md, as this tool did before the census found the "
                         "citations living in code")
    ap.add_argument("--selftest", action="store_true",
                    help="check the scanner against the part-1 audit's bugs and the new checks")
    args = ap.parse_args(argv[1:])

    if args.selftest:
        return selftest()

    root = git("rev-parse", "--show-toplevel").strip()
    if root:
        os.chdir(root)
    include_code = not args.no_include_code
    files = scanned_files(include_code)
    if not files:
        print("no tracked files found — is this a git checkout?", file=sys.stderr)
        return 2
    tracked = tracked_files()
    classes = classify(scan_docs(files), tracked)
    refs = ref_findings((fn, lineno, line)
                        for fn in files
                        for lineno, line in enumerate(read_text_lines(fn), 1))
    if not args.list_broken:
        print("scanned %d tracked files (%s), %d tracked paths"
              % (len(files), "markdown + code" if include_code else "markdown only",
                 len(tracked)))
    if refs["anchor_ok"] + len(refs["anchor_broken"]) == 0:
        print("ANCHOR CHECK MATCHED NOTHING — it adjudicates nothing on this input", file=sys.stderr)
        return 2
    return report(classes, refs, show_all=args.all, list_broken=args.list_broken)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
