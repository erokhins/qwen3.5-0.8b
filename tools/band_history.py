#!/usr/bin/env python3
"""Was any numeric tolerance loosened to make something pass? Re-derivable band history.

    .venv/bin/python tools/band_history.py [--since "2026-10-11 00:00"] [--root .]

Walks every commit after `--since` that touched a gate file (prototype/*/run.py, */check.py,
*/floor_probe.py, */reference.py and speccheck/gates/*.py), reads that revision's tolerance
constants straight out of the file, and reports only the places where a threshold's VALUE moved —
raised, tightened or deleted — with the commit that moved it. Two passes:

  1. named constants (`F4_MEANREL_MAX = 5e-3`, tuple forms included);
  2. thresholds that are not constants: keyword arguments (`ulp_max=2`, `cancel_rel=1e-4`,
     `meanrel_max=1e-3`), argparse defaults on threshold flags, and inline comparisons
     (`cos < 0.999`, `meanrel > 0.05`).

A threshold that appears for the first time in a file created after `--since` is NEW, not loosened,
so it is counted and not reported as a move — that distinction is the whole point: a gate written
today says nothing about whether a gate was widened today. Read-only; touches no device.

The answer this tool was written for is in `bench/commit-review-2026-10-11.md` §2.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess

GLOBS = ["prototype/*/run.py", "prototype/*/check.py", "prototype/*/floor_probe.py",
         "prototype/*/reference.py", "speccheck/gates/*.py"]

CONST_NAME = re.compile(r"(MIN|MAX|REL|FLOOR|TOL|BAND|ACCEPT|AGREE|NDIFF|COS|MEANREL|ULP|EPS)")
ASSIGN = re.compile(r"^([A-Za-z][A-Za-z_0-9]*(?:\s*,\s*[A-Za-z][A-Za-z_0-9]*)*)\s*=\s*"
                    r"([^#]+?)\s*(?:#.*)?$")
KWARG = re.compile(r"(cancel_rel|ulp_max|meanrel_max|cos_min|ndiff_pct_max|cancel_band|"
                   r"min-accept|min-agreement|--floor|--cos-min|--meanrel-max)"
                   r"[^0-9A-Za-z_]{0,4}([0-9]*\.?[0-9]+(?:e-?\d+)?)")
INLINE = re.compile(r"(cos >=|cos <|meanrel <=|meanrel >)\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)")


def git(root, *args):
    return subprocess.run(["git", "-C", root] + list(args),
                          capture_output=True, text=True).stdout


def consts_of(text):
    """Top-level-looking `NAME = number` (or tuple form) whose name looks like a threshold."""
    out = {}
    for line in text.splitlines():
        m = ASSIGN.match(line.strip())
        if not m:
            continue
        names = [n.strip() for n in m.group(1).split(",")]
        vals = [v.strip() for v in m.group(2).split(",")]
        if len(names) != len(vals):
            continue
        for name, val in zip(names, vals):
            if CONST_NAME.search(name):
                try:
                    float(val)
                except ValueError:
                    continue
                out["const:" + name] = val
    return out


def thresholds_of(text):
    """Thresholds that live in call sites and CLI defaults rather than in constants."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hits = set(m.group(0) for m in KWARG.finditer(s)) | set(m.group(0) for m in INLINE.finditer(s))
        if hits:
            out[s[:110]] = hits
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-10-11 00:00")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    touched = git(args.root, "log", "--since=" + args.since, "--name-only", "--format=%x09").split()
    files = sorted({f for f in touched for g in GLOBS if fnmatch.fnmatch(f, g)})

    moved, brand_new = [], set()
    for f in files:
        revs = list(reversed(git(args.root, "log", "--since=" + args.since, "--format=%h",
                                 "--", f).split()))
        prev, first = None, None
        for rev in revs:
            text = git(args.root, "show", "%s:%s" % (rev, f))
            if not text:
                continue
            cur = consts_of(text)
            if first is None:
                first = set(cur)
            if prev is not None:
                for name, val in cur.items():
                    if name in prev and prev[name] != val:
                        moved.append((f, name[5:], prev[name], val, rev))
                for name in prev:
                    if name not in cur:
                        moved.append((f, name[5:], prev[name], "(deleted)", rev))
            prev = cur
        brand_new |= (first or set())

    print("since %s | gate files touched: %d | threshold constants first seen today: %d"
          % (args.since, len(files), len(brand_new)))
    print("\n== pass 1: named threshold constants that MOVED ==")
    if not moved:
        print("(none)")
    for f, name, old, new, rev in moved:
        try:
            way = "RAISED (looser)" if float(new) > float(old) else "tightened"
        except ValueError:
            way = "DELETED the criterion"
        print("  %-30s %-20s %-9s -> %-11s %-20s %s  %s"
              % (f, name, old, new, way, rev,
                 git(args.root, "log", "-1", "--format=%s", rev).strip()[:58]))

    print("\n== pass 2: call-site / CLI thresholds that MOVED (line identity, so expect noise) ==")
    any2 = False
    for f in files:
        revs = list(reversed(git(args.root, "log", "--since=" + args.since, "--format=%h",
                                 "--", f).split()))
        prev = None
        for rev in revs:
            text = git(args.root, "show", "%s:%s" % (rev, f))
            if not text:
                continue
            cur = thresholds_of(text)
            if prev is not None:
                gone = set(prev) - set(cur)
                added = set(cur) - set(prev)
                for g in sorted(gone):
                    pre = [a for a in added if a[:40] == g[:40]]
                    if pre:
                        print("  %-30s %s  %s  ->  %s" % (f, rev, prev[g], cur[pre[0]]))
                    else:
                        print("  %-30s %s  disappeared: %s %s" % (f, rev, prev[g], g[:60]))
                    any2 = True
            prev = cur
    if not any2:
        print("(none)")


if __name__ == "__main__":
    main()
