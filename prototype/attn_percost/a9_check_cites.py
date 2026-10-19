#!/usr/bin/env python3
"""a9_check_cites.py — print the source line behind every `path:NNN` / `path:NNN-MMM` cite in a doc.

A citation in this repo is a claim about a specific line, and nothing else in the tree checks them
(`tools/doc_ref_check.py` checks paths and §labels; `tools/citation_rot_check.py` checks `progress/`
coordinates against HEAD). This is the local version for one markdown file: it prints what each cite
actually points at, so a wrong line is visible instead of plausible.

Usage: python3 prototype/attn_percost/a9_check_cites.py [doc.md ...]
"""
import collections
import re
import sys

CITE = re.compile(r"`?([A-Za-z0-9_./\-]+\.(?:h|hpp|cpp|cu|cuh|json|py|sh|md)):(\d+)(?:[-,](\d+))?`?")


def main(paths):
    bad = 0
    for doc in paths:
        seen = collections.OrderedDict()
        for lineno, line in enumerate(open(doc, errors="replace"), 1):
            for m in CITE.finditer(line):
                path, a, b = m.group(1), int(m.group(2)), m.group(3)
                seen.setdefault((path, a), []).append(lineno)
                if b:
                    seen.setdefault((path, int(b)), []).append(lineno)
        print(f"=== {doc}: {len(seen)} distinct cited lines ===")
        for (path, ln), whos in seen.items():
            try:
                src = open(path, errors="replace").read().splitlines()
            except OSError:
                print(f"  MISSING FILE  {path}:{ln}   (cited at doc lines {whos})")
                bad += 1
                continue
            if not 1 <= ln <= len(src):
                print(f"  OUT OF RANGE  {path}:{ln} (file has {len(src)} lines) at doc lines {whos}")
                bad += 1
                continue
            txt = src[ln - 1].strip()
            flag = "  " if txt else "??"
            print(f"  {flag} {path}:{ln:<5} {txt[:110]}")
    print(f"\n{bad} citations point outside their file; '??' marks a cite landing on a blank line.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["prototype/attn_percost/A9-LONGCTX-CENSUS.md"]))
