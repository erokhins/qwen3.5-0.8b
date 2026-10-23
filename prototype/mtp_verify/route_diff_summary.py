#!/usr/bin/env python3
"""Summarise `route_diff.sh`'s output into one line per (comparison, width).

Takes the script's stdout on stdin. Pure arithmetic over numbers the engine's own instrument printed; it
adds no metric and gates nothing — its only job is to keep a range out of a hand-copy.
"""
import re
import sys

groups: dict[str, list[tuple[str, float, float, int, bool]]] = {}
width = "?"
for ln in sys.stdin:
    m = re.search(r"=== lanes=(\S+)", ln)
    if m:
        width = m.group(1)
        continue
    if "vs-step" not in ln and "hand-vs-amort" not in ln:
        continue
    # `\w+` alone would drop `1-cos` (the hyphen is not a word char), and a parser that silently loses the
    # headline metric is the failure class this repo records over and over.
    d = dict(re.findall(r"([\w-]+)=([^\s]+)", ln))
    if "1-cos" not in d or "argmax" not in d:
        continue
    kind = ("hand vs amort" if "hand-vs-amort" in ln
            else "hand vs step" if re.search(r"\bhand\s+anchor", ln) else "amort vs step")
    a, _, b = d["argmax"].partition("/")
    groups.setdefault(kind, []).append((width, float(d["1-cos"]), float(d["meanrel"]),
                                        int(d["ndiff"]), a == b))
for kind in ("hand vs step", "amort vs step", "hand vs amort"):
    for w in sorted({r[0] for r in groups.get(kind, [])}):
        rows = sorted([r for r in groups[kind] if r[0] == w], key=lambda r: r[1])
        mr = sorted(r[2] for r in rows)
        nd = sorted(r[3] for r in rows)
        agree = sum(1 for r in rows if r[4])
        print(f"{kind:15s} lanes={w:5s} n={len(rows)} argmax_agrees={agree}/{len(rows)}"
              f"  1-cos {rows[0][1]:.3e}..{rows[-1][1]:.3e}"
              f"  meanrel {mr[0]:.3e}..{mr[-1]:.3e}  ndiff {nd[0]}..{nd[-1]}")
