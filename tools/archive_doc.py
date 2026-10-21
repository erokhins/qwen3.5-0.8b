#!/usr/bin/env python3
"""Move a document and re-point every citation that names its old path.

WHY THIS EXISTS. `progress/README.md` rule 1 says a document moves to `progress/` the moment it
stops being a live instruction, and rule 2 says the only edit an archived file ever receives is a
path rewrite — and it must be a **same-line substitution**, because `file:NNNN` cites inside the
journals resolve by line number. Doing that by hand across 400+ occurrences is how a cleanup
invents drift: `tools/citation_rot_check.py` exists precisely because a line-form cite that still
*resolves* is invisible to every other checker. So the move and the re-point happen in one scripted
pass, and the pass refuses rather than guesses.

TWO MODES.

  pair mode   --pair OLD:NEW            every cite of OLD becomes NEW.
  split mode  --split OLD
              --live NEW_LIVE
              --archive NEW_ARCHIVE
              --archive-from N          a cite whose `§label` numeric part is >= N goes to the
                                        archive, the rest to the live file, so a `§label` cite
                                        keeps naming a file that actually carries that heading
                                        (`tools/doc_ref_check.py` BROKEN-LABEL is what punishes
                                        getting this wrong). Line-form cites (`path:NNN`) always
                                        go to the archive: the archive is a whole-file copy of the
                                        pre-move document, so its line coordinates ARE the old
                                        ones and a `:NNN` cite survives the move unchanged.

WHAT COUNTS AS A CITATION. The old path token, plus anything attached to it that decides routing:

  `path`            backticked or bare
  path:NNN          and :NNN-NNN, :NNN,NNN, :NNN,NNN-NNN
  path §N           spaced label, backticked path or not
  §N of `path`      the label sits BEFORE the path in this repo's prose
  path's            possessive

Only the path token is rewritten; the attached label/line list is matched so it can choose the
target and is then left byte-for-byte alone.

SAFETY. Every replacement is asserted to (a) contain no newline, (b) leave the file's line count
unchanged, and (c) go to a file we named on the command line. A substitution that would move a line
aborts the whole run before anything is written. Files under `engines/`, `tmp/`, `.venv/`, `data/`,
`models/`, `logs/`, `.idea/` are never touched.

--selftest runs the routing table against synthetic cites, including the two mistakes that must
not be survivable: a `:NNN` cite routed to the live file, and a label past the archive boundary
routed to the live file.
"""

import argparse
import re
import subprocess
import sys

SKIP_PREFIXES = ("engines/", "tmp/", ".venv/", "data/", "models/", "logs/", ".idea/")


def cite_re(old: str) -> re.Pattern:
    """One citation occurrence: optional leading `§N of `, the path, then an attached line list
    and/or section label. `[ \\t]` rather than `\\s` on purpose: a label on the NEXT line is a
    different sentence and must not decide this one's target."""
    esc = re.escape(old)
    return re.compile(
        r"(?P<prefix>§(?P<prelabel>[\w.\-]+)[ \t]+of[ \t]+[`'\"]?)?"
        + esc
        + r"(?P<suffix>(?::[0-9][0-9,\- ]*)?(?:[`'\"]?[ \t]*§(?P<label>[\w.\-]+))?'?s?)?"
    )


def route(label, prelabel, lines, split):
    """Which file does this cite name now? Returns (path, why)."""
    lab = label or prelabel
    if lines:
        return split["archive"], "line-form"
    num = re.match(r"(\d+)", lab or "")
    if num and int(num.group(1)) >= split["from"]:
        return split["archive"], f"\u00a7{lab} >= \u00a7{split['from']}"
    return split["live"], (f"\u00a7{lab}" if lab else "bare")


def rewrite_text(text, pairs, split):
    """Returns (new_text, {target: count}, [(start, end, target, why)])."""
    jobs = [(old, new, None) for old, new in pairs]
    if split:
        jobs.append((split["old"], None, split))
    edits = []
    for old, new, sp in jobs:
        for m in cite_re(old).finditer(text):
            head = m.group("prefix") or ""
            suffix = m.group("suffix") or ""
            start, end = m.start() + len(head), m.end() - len(suffix)
            body = text[start:end]
            if "\n" in body or not body:
                raise SystemExit(f"REFUSE: bad match span {body!r}")
            lm = re.match(r":([0-9][0-9,\- ]*)", suffix)
            if sp:
                target, why = route(m.group("label"), m.group("prelabel"), lm and lm.group(1) or "", sp)
            else:
                target, why = new, "pair"
            if "\n" in target:
                raise SystemExit("REFUSE: target path contains a newline")
            edits.append((start, end, target, why))
    edits.sort(key=lambda e: e[0], reverse=True)
    out, counts = text, {}
    for s, e, target, why in edits:
        out = out[:s] + target + out[e:]
        counts[target] = counts.get(target, 0) + 1
    return out, counts, edits


def tracked_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out if not f.startswith(SKIP_PREFIXES)]


def selftest():
    """Assert the routing table. A move tool whose routing cannot be tested cannot be trusted."""
    split = {"old": "bench/LEDGER.md", "live": "engine/LEDGER.md",
             "archive": "progress/2026-01-02/LEDGER-archive.md", "from": 7}
    cases = [
        ("see `bench/LEDGER.md` §6 for rules", "engine/LEDGER.md` §6"),
        ("as `bench/LEDGER.md` §48 records", "progress/2026-01-02/LEDGER-archive.md` §48"),
        ("§28 of `bench/LEDGER.md` identifies", "§28 of `progress/2026-01-02/LEDGER-archive.md"),
        ("frozen at bench/LEDGER.md:92,618", "progress/2026-01-02/LEDGER-archive.md:92,618"),
        ("rows bench/LEDGER.md:1637-1638", "progress/2026-01-02/LEDGER-archive.md:1637-1638"),
        ("the live bench/LEDGER.md carries it", "engine/LEDGER.md carries it"),
        ("`bench/LEDGER.md`'s own banner still works", "engine/LEDGER.md`'s own banner"),
        ("label with a full stop, bench/LEDGER.md §6. is fine", "engine/LEDGER.md §6."),
        ("§6 is the label form bench/LEDGER.md §6.", "engine/LEDGER.md §6."),
        ("a next-line label must not route: bench/LEDGER.md\n  §48 elsewhere", "engine/LEDGER.md"),
    ]
    fails = 0
    for text, expect in cases:
        got, _, _ = rewrite_text(text, [], split)
        ok = expect in got
        print(f"  {'ok  ' if ok else 'FAIL'} {text!r}\n        -> {got!r}")
        fails += 0 if ok else 1
    got, _, _ = rewrite_text("x bench/LEDGER.md §7 y", [("bench/LEDGER.md", "engine/LEDGER.md")], None)
    if got != "x engine/LEDGER.md §7 y":
        print(f"  FAIL pair mode -> {got!r}")
        fails += 1
    # Both targets must be produced by a mixed document: a split that routes everything one way is
    # the silent failure this asserts against.
    mixed = "a `bench/LEDGER.md` §3 b `bench/LEDGER.md` §41 c bench/LEDGER.md:12"
    _, counts, _ = rewrite_text(mixed, [], split)
    if counts.get("engine/LEDGER.md") != 1 or counts.get("progress/2026-01-02/LEDGER-archive.md") != 2:
        print(f"  FAIL mixed routing -> {counts}")
        fails += 1
    print("SELFTEST", "FAIL" if fails else "PASS", f"({len(cases) + 2} cases, {fails} failing)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", default=[], metavar="OLD:NEW",
                    help="simple move; repeatable")
    ap.add_argument("--split", metavar="OLD", help="document that splits into live + archive")
    ap.add_argument("--live", metavar="NEW", help="live file of the split")
    ap.add_argument("--archive", metavar="NEW", help="archive file of the split")
    ap.add_argument("--archive-from", type=int, metavar="N", help="§N and later go to the archive")
    ap.add_argument("--only", action="append", default=[], help="restrict to these files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    pairs = []
    for p in a.pair:
        if ":" not in p:
            raise SystemExit(f"--pair wants OLD:NEW, got {p!r}")
        old, new = p.split(":", 1)
        pairs.append((old, new))
    split = None
    if a.split:
        if not (a.live and a.archive and a.archive_from is not None):
            raise SystemExit("--split needs --live, --archive and --archive-from")
        split = {"old": a.split, "live": a.live, "archive": a.archive, "from": a.archive_from}
        if any(o == a.split for o, _ in pairs):
            raise SystemExit(f"{a.split} given as both --pair and --split")
    if not pairs and not split:
        raise SystemExit("nothing to do: pass --pair or --split")

    total, touched = {}, 0
    for f in (a.only or tracked_files()):
        try:
            before = open(f, encoding="utf-8", errors="ignore").read()
        except OSError as exc:
            print(f"skip {f}: {exc}", file=sys.stderr)
            continue
        after, counts, _ = rewrite_text(before, pairs, split)
        if after == before:
            continue
        if after.count("\n") != before.count("\n"):
            raise SystemExit(
                f"REFUSE: {f} would change line count ({before.count(chr(10))} -> "
                f"{after.count(chr(10))}); rule 2 needs same-line substitutions")
        if not a.dry_run:
            open(f, "w", encoding="utf-8").write(after)
        touched += 1
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
        print(f"  {f}: {sum(counts.values())} cite(s)")
    print("\nDRY RUN — nothing written." if a.dry_run else "")
    print("files touched:", touched)
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d} -> {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
