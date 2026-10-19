#!/usr/bin/env python3
"""A4-L1 canary: does the base side of a4_l1_build.sh reproduce the SHIPPED instruction stream?

`engine/build.sh` is not bit-reproducible on this box and a paired A/B against a stale or re-shaped binary reads
as a clean null (AGENTS.md, "a paired A/B against a stale binary reads as a clean null"). Both sides here are
compiled from COPIES of the engine TU, so the base side must be shown to be the shipped code before the
base-vs-new delta means anything. This compares instruction text (addresses stripped) of `attn_fattn_wave`
between any two listings.

    python3 prototype/attn_percost/a4_l1_canary.py LISTING_A LISTING_B [--kernel attn_fattn_wave]

Exit 1 on any difference, with the first few differing lines.
"""
import argparse
import re

FUNC = re.compile(r"^([0-9a-fA-F]{16}) <(.*)>:")
INSN = re.compile(r"^\s+(\S.*?)\s*//\s*[0-9A-Fa-f]{8,16}:")


def body(path, kernel):
    lines = open(path).read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        m = FUNC.match(ln)
        if m and kernel in m.group(2):
            start = i
            break
    if start is None:
        raise SystemExit(f"REFUSE: no function matching {kernel!r} in {path}")
    out = []
    for ln in lines[start + 1:]:
        if FUNC.match(ln):
            break
        m = INSN.match(ln)
        if m:
            out.append(m.group(1).strip())
    if not out:
        raise SystemExit(f"REFUSE: {kernel} parsed to 0 instructions in {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--kernel", default="attn_fattn_wave")
    a = ap.parse_args()
    ba, bb = body(a.a, a.kernel), body(a.b, a.kernel)
    print(f"{a.kernel}: {a.a} = {len(ba)} insns | {a.b} = {len(bb)} insns")
    if ba == bb:
        print(f"IDENTICAL (addresses stripped): {len(ba)} instructions agree")
        return
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(ba, bb)) if x != y]
    print(f"DIFFER: {len(diff)} of {min(len(ba), len(bb))} aligned lines"
          + ("" if len(ba) == len(bb) else f"  (lengths differ: {len(ba)} vs {len(bb)})"))
    for i, x, y in diff[:6]:
        print(f"  [{i}] - {x}\n      + {y}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
