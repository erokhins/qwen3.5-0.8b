#!/usr/bin/env python3
"""Where did the line numbers my own edit moved go? The pre-commit sibling of `citation_rot_check.py`.

`tools/citation_rot_check.py` compares the working tree against HEAD, so it does see drift a dirty tree
introduced — and that same drift goes invisible the moment the commit lands, because then both sides of its
comparison moved together (`AGENTS.md`: "citation_rot_check.py cannot see drift introduced inside the change
being checked"). So the guard for editing a cited file is a coordinate diff BEFORE committing, and nothing in
this tree printed the new numbers: `wc -l` before and after says a file grew, not where.

Usage:
    .venv/bin/python3 tools/citation_repoint.py engine/decode_run.cpp          # every cite it can find
    .venv/bin/python3 tools/citation_repoint.py engine/decode_run.cpp 325 1795

For each `path:NNN` anchor that points at the edited file, it takes HEAD's text at N and finds where that
exact line now sits, preferring the candidate nearest `N + (insertions above N)` so a line of text that
occurs more than once (a bare `//`, a closing brace) does not get silently reassigned to the wrong copy.
Candidates are printed with their distance from the estimate and, when a text is ambiguous, the tool says so
instead of guessing: a re-point that lands on a different `//` is worse than no re-point. Lines the diff
deleted are reported as DELETED, because a cite into deleted text is a doc rewrite, not a renumber.

Read-only. It writes nothing, so run it before committing and hand its output to whoever re-points the cites.
Journals under `progress/` are frozen and may take a same-line substitution only — which is what a re-point is.
"""
from __future__ import annotations

import difflib
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANCHOR = re.compile(r"([A-Za-z0-9_./-]+?):(\d+)")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False,
                          cwd=ROOT).stdout


def shift_by_insertions(head: list[str], work: list[str]) -> list[int]:
    """shift[N] = how far HEAD line N moved, from a real diff of the two whole files.

    `difflib`'s opcodes on line lists give exactly the same information as `git diff`'s hunk headers,
    without a parser to get wrong: for every unchanged block the offset is the block's new-minus-old.
    """
    shift = [0] * (len(head) + 2)
    sm = difflib.SequenceMatcher(a=head, b=work, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                shift[k + 1] = j1 - i1
    return shift


def cites_of(target: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    tail = target.split("/")[-1]
    for f in git("grep", "-l", "--", target).split():
        for m in ANCHOR.finditer(git("show", f"HEAD:{f}")):
            p = m.group(1)
            if p in (target, tail) or p.endswith("/" + tail):
                out[int(m.group(2))].append(f)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    head = git("show", f"HEAD:{target}").splitlines()
    if not head:
        raise SystemExit(f"HEAD:{target} is empty or absent; this maps cites into a committed file")
    work = (ROOT / target).read_text().splitlines()
    shift = shift_by_insertions(head, work)
    given = [int(x) for x in sys.argv[2:]]
    found = {} if given else cites_of(target)
    lines = given or sorted(found)
    if not lines:
        print(f"{target}: no line anchor cites it; nothing to re-point")
        return 0
    moved = 0
    for w in lines:
        who = ",".join(sorted(set(found.get(w, [])))) or "command line"
        if not 1 <= w <= len(head):
            print(f"{w:>6} ->  OUT-OF-RANGE-AT-HEAD  (stale before this edit)  <- {who}")
            continue
        text = head[w - 1]
        if text.strip() and text not in work:
            print(f"{w:>6} ->  DELETED BY THIS EDIT  {text.strip()[:60]:<60} <- {who}")
            moved += 1
            continue
        if not text.strip():
            print(f"{w:>6} ->  BLANK AT HEAD, cannot be matched by text; check by eye  <- {who}")
            continue
        est = min(max(1, w + shift[w]), len(work))
        cands = [i + 1 for i, t in enumerate(work) if t == text]
        best = min(cands, key=lambda c: abs(c - est))
        note = "unchanged" if best == w else "MOVED"
        if len(cands) > 1:
            note += f", {len(cands)} identical lines, nearest-first={sorted(cands, key=lambda c: abs(c - est))[:3]}"
        if best != w:
            moved += 1
        print(f"{w:>6} -> {best:<6} {note:<12} {text.strip()[:58]:<58} <- {who}")
    print(f"\n{target}: HEAD {len(head)} lines, working tree {len(work)}; {moved} of {len(lines)} anchor(s) moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
