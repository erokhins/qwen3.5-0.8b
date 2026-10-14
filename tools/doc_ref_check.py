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

Exit status: 0 if nothing is broken, 1 if anything lands in a MISSING bucket.

Usage:
  .venv/bin/python tools/doc_ref_check.py                  # summary report
  .venv/bin/python tools/doc_ref_check.py --list-broken    # one path per line
  .venv/bin/python tools/doc_ref_check.py --all            # every bucket
  .venv/bin/python tools/doc_ref_check.py --selftest       # scanner regression
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

# Absent under one of these is the intended state: regenerable by a documented
# command, and excluded from git on purpose (part-1 audit §3).
SCRATCH_ROOTS = ("tmp/", "logs/")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def tracked_files() -> set[str]:
    return set(x for x in git("ls-files").splitlines() if x)


def markdown_files() -> list[str]:
    return sorted(x for x in git("ls-files", "-z", "*.md").split("\0") if x)


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


def scan_docs(files: list[str]) -> dict[str, list[str]]:
    """path -> ["file:line", ...] for every tracked markdown citation."""
    cites: dict[str, list[str]] = defaultdict(list)
    for fn in files:
        with open(fn, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
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


def report(classes: dict[str, dict], *, show_all: bool, list_broken: bool) -> int:
    buckets: dict[str, list[str]] = defaultdict(list)
    for path, e in classes.items():
        buckets[e["bucket"]].append(path)

    if list_broken:
        for b in BROKEN_BUCKETS:
            for p in sorted(buckets[b]):
                print(p)
        return 1 if any(buckets[b] for b in BROKEN_BUCKETS) else 0

    order = (["TRACKED", "PRESENT-DIR", "PRESENT-IGNORED", "PRESENT-DIRTY"]
             + list(BROKEN_BUCKETS) + ["SHORTHAND"])
    print("doc_ref_check: %d distinct path citations, %d broken"
          % (len(classes), sum(len(buckets[b]) for b in BROKEN_BUCKETS)))
    for b in order:
        print("  %-20s %4d" % (b, len(buckets[b])))

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
    return 1 if any(buckets[b] for b in BROKEN_BUCKETS) else 0


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


def selftest() -> int:
    ok = True
    for text, want in SELFTEST_CASES:
        got = cite_tokens(text)
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            ok = False
        print("%s  %-52s -> %s" % (mark, text, got))
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true",
                    help="also list PRESENT-UNTRACKED paths (verbose)")
    ap.add_argument("--list-broken", action="store_true",
                    help="print only the broken paths, one per line")
    ap.add_argument("--selftest", action="store_true",
                    help="check the scanner against the part-1 audit's bugs")
    args = ap.parse_args(argv[1:])

    if args.selftest:
        return selftest()

    root = git("rev-parse", "--show-toplevel").strip()
    if root:
        os.chdir(root)
    md = markdown_files()
    if not md:
        print("no tracked *.md found — is this a git checkout?", file=sys.stderr)
        return 2
    tracked = tracked_files()
    classes = classify(scan_docs(md), tracked)
    if not args.list_broken:
        print("scanned %d tracked markdown files, %d tracked paths"
              % (len(md), len(tracked)))
    return report(classes, show_all=args.all, list_broken=args.list_broken)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
