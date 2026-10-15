#!/usr/bin/env python3
"""Did this change set ROT a `path:NNN` citation? — line-citation drift checker.

Why this exists. A line citation is a claim about a file's line count, not only about its identity.
`tools/doc_ref_check.py` already proves an anchor is in range and non-blank, and it stays green for
citations that point at the wrong sentence: insert one line anywhere above a cited line and the cite
still resolves to a line number that exists and has text in it, while the text under it is a
different sentence. That is how the 2026-10-16 ledger prune rotted 27 citations through a green
`doc_ref_check` run — a citation-migration pass added lines to ten files the frozen archive cites by
number, so `progress/README.md:135` still "resolved" while `devices/strix-halo-mixed-traffic.md:175`
had slid one line down off the mixed_bw command line it names. The rule this enforces is the one in
`AGENTS.md`'s citation-hygiene section — *`wc -l` before and after; journals take same-line
substitutions only* — applied to **every** file any line cite points at, not just the files with
"journal" in their name.

Method (CPU only, no device, no build):

1. Enumerate the citing corpus: every tracked file that exists on disk, markdown AND code.
   `prototype/glue/run.py:65` and `engine/kernels.h` carry line cites, and `doc_ref_check.py`'s
   first version walked only `*.md` — its own header records what that missed.
2. Find every line-form citation: `path:NNN` and the multi-anchor spellings `path:NNN-NNN`,
   `path:NNN,NNN`, `path:NNN,NNN-NNN`. Each named endpoint is checked, which is the same policy
   `doc_ref_check.py` uses for ranges, so the two checkers agree on what an anchor is.
3. Compare the cited text at `--against` (default `HEAD`) against the text at the same coordinate in
   the working tree. Different text = ROTTED, printed as
   `ROTTED file:line | cited by file:line` plus the `was`/`now` lines.

What this CANNOT see, stated plainly because a clean bill here is not a clean bill of citations:

* It drift-checks only what a `path:NNN` cite names. A cite written as `` `path` §4 `` or "see the
  table in X" carries no coordinate and is invisible here — §labels are `doc_ref_check.py`'s
  population, and a §cite to a section that moved is a different defect.
* A cite whose target is **absent at either revision** is skipped and counted, never silently
  passed: the summary prints `skipped (…)`, split into absent-or-untracked and out-of-range. Most of
  this tree's known-broken paths are `tmp/` scratch recorded as history, so "no target at HEAD" is
  the common case and must be countable rather than hidden.
* A same-line substitution IS reported as drift, deliberately: editing a cited line changes what
  that citation quotes. Usually that is intended (the fact stands, the path inside the line got
  re-pointed) — but it should be a decision, not a side effect, so it is printed rather than waved
  through. Read those rows first.
* Equal text at both revisions passes even if the file was rewritten between them. Text-in-place is
  the invariant, not file identity.
* Edits and deletions strictly BELOW a cited line are invisible, correctly: they cannot move it.

Exit status: 0 with no drift; 1 if any citation is rotted; 2 if the checker parsed nothing at all —
a check whose input list can be empty cannot fail, the trap this repo already documents three times
(`prototype/gdn_collapse/run.py`'s `CASE_RE`, `prototype/attn_tile_model.py`'s derived blind sets,
`tools/mixed_bw.cpp`'s `verify_accounts`).

`--selftest` plants a shift in a synthetic two-revision corpus and asserts the detector notices it,
and asserts it stays quiet when nothing moved and when the only change is a deletion below the cited
line. The selftest runs in memory: a selftest that edits a tracked file to prove a point is a
selftest that can corrupt the thing it checks. Run it after touching `CITE`, `anchors_of` or
`rot_rows`.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Extension list is longest-first with a boundary lookahead, for the reason written up in
# doc_ref_check.py's header (a first-match-wins alternation recorded `x.jsonl` as a missing `x.json`).
EXTS = "md|py|cpp|hpp|h|hip|sh|json|jsonl|txt|tsv|csv|yaml|yml|toml|ini|log|out|npy"
# The lookbehind rejects a preceding word char or slash, so a URL tail or a clock reading
# (`12:34`, `https://host/a.md:1`) cannot present as a citation.
CITE = re.compile(rf"(?<![\w/.])((?:[-\w.]+/)*[-\w.]+\.(?:{EXTS}))(?![\w.]):(\d+(?:[,-]\d+)*)")

SKIP_ABSENT = "absent-or-untracked-at-one-revision"
SKIP_RANGE = "out-of-range-at-one-revision"


def git_show(rev: str, path: str) -> list[str] | None:
    """Lines of `path` at `rev`, or None when git has no such blob (untracked, or absent then)."""
    proc = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"],
                          capture_output=True, text=True)
    return proc.stdout.splitlines() if proc.returncode == 0 else None


def git_ls_files() -> list[str]:
    proc = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True)
    return proc.stdout.split()


def resolve(raw: str, citer: str) -> str | None:
    """Resolve a citation path to a repo-relative path; a bare name may mean the file next door.

    Docs here sometimes write `results.md:283` meaning the `results.md` in the same directory, so
    the citing file's own directory is the second candidate. Both candidates are constrained to lie
    inside the repo, so `../../etc/x.md:1` resolves to nothing.
    """
    for cand in (raw, str(Path(citer).parent / raw)):
        p = (REPO / cand).resolve()
        try:
            rel = str(p.relative_to(REPO))
        except ValueError:
            continue
        if (REPO / rel).is_file():
            return rel
    return None


def anchors_of(spec: str) -> list[int]:
    """Every endpoint an anchor list names: `NNN`, `NNN-NNN`, `NNN,NNN`, `NNN,NNN-NNN`."""
    out: list[int] = []
    for part in spec.split(","):
        for num in part.split("-"):
            if num.isdigit():
                out.append(int(num))
    return out


def rot_rows(citer: str, lines: list[str], old_of, new_of) -> tuple[list[tuple], dict, int]:
    """Rotted rows for one citing file, given two `path -> lines | None` revision providers.

    A row is `(target, line, citer:line, was, now)`. Kept free of any I/O so `--selftest` can drive
    it with in-memory revisions and the real run can drive it with git plus the working tree.
    """
    rows: list[tuple] = []
    skips: dict[str, int] = {}
    parsed = 0

    def skip(reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    for i, line in enumerate(lines, 1):
        for m in CITE.finditer(line):
            target = m.group(1)
            old = old_of(target, citer)
            new = new_of(target, citer)
            if old is None or new is None:
                skip(SKIP_ABSENT)
                continue
            shown = resolve(target, citer) or target
            for n in anchors_of(m.group(2)):
                parsed += 1
                if n > len(old) or n > len(new):
                    skip(SKIP_RANGE)
                    continue
                if old[n - 1] != new[n - 1]:
                    rows.append((shown, n, f"{citer}:{i}", old[n - 1], new[n - 1]))
    return rows, skips, parsed


def selftest() -> int:
    """Plant a shift in a synthetic two-revision corpus and prove the detector notices."""
    cited = "the mixed_bw command line"
    target_old = ["# head", cited, "trailing prose"]
    cite_line = "see `x/target.md:2` for the command"
    cases = [
        # name, revision-under-test lines, expected catches
        ("shift", ["inserted", target_old[0], cited, target_old[2]], 1),
        ("quiet", target_old, 0),
        ("below", [target_old[0], cited], 0),
    ]
    failed = []
    for name, new_lines, want in cases:
        rows, _, parsed = rot_rows(
            "x/citer.md", [cite_line],
            lambda p, c, t=target_old: t if p == "x/target.md" else None,
            lambda p, c, t=new_lines: t if p == "x/target.md" else None,
        )
        print(f"selftest {name:6s}: parsed {parsed}, caught {len(rows)}, expected {want}")
        if parsed == 0 or len(rows) != want:
            failed.append(name)
    if failed:
        print(f"SELFTEST FAIL — wrong verdict in: {', '.join(failed)}")
        return 1
    print("SELFTEST PASS — a planted shift is caught, a quiet corpus and a below-only edit stay quiet")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="detect line-citation drift between a revision and the working tree")
    ap.add_argument("--against", default="HEAD", help="revision to compare against (default HEAD)")
    ap.add_argument("--selftest", action="store_true", help="prove the detector can fail")
    ap.add_argument("--list-skips", action="store_true", help="print skip counters by reason")
    ap.add_argument("--all-rows", action="store_true", help="print rotted rows even when there are many")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    cache: dict[str, list[str] | None] = {}

    def old_of(path: str, citer: str) -> list[str] | None:
        rel = resolve(path, citer)
        if rel is None:
            return None
        if rel not in cache:
            cache[rel] = git_show(args.against, rel)
        return cache[rel]

    def new_of(path: str, citer: str) -> list[str] | None:
        rel = resolve(path, citer)
        if rel is None:
            return None
        try:
            return (REPO / rel).read_text(errors="replace").splitlines()
        except OSError:
            return None

    all_rows: list[tuple] = []
    skips: dict[str, int] = {}
    parsed = files_scanned = 0
    for rel in git_ls_files():
        p = REPO / rel
        if not p.is_file():
            continue
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        files_scanned += 1
        rows, sk, n = rot_rows(rel, lines, old_of, new_of)
        all_rows.extend(rows)
        parsed += n
        for k, v in sk.items():
            skips[k] = skips.get(k, 0) + v

    printed = 0
    for target, n, citer, was, now in sorted(all_rows):
        if not args.all_rows and printed >= 60:
            print(f"… {len(all_rows) - printed} more (pass --all-rows)")
            break
        printed += 1
        print(f"ROTTED {target}:{n} | cited by {citer}")
        print(f"    was: {was.strip()[:150]}")
        print(f"    now: {now.strip()[:150]}")

    if args.list_skips:
        for k, v in sorted(skips.items()):
            print(f"skipped {k}: {v}")
    print(f"\nscanned {files_scanned} tracked files against {args.against}; "
          f"checked {parsed} line-anchor(s); skipped {sum(skips.values())} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(skips.items())) or 'nothing'})")
    print(f"{len(all_rows)} rotted line-citation(s).")
    if parsed == 0:
        print("ANCHOR-PARSE-EMPTY: nothing parsed, so this check adjudicated nothing — a failure, not a pass")
        return 2
    return 1 if all_rows else 0


if __name__ == "__main__":
    sys.exit(main())
