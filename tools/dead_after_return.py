#!/usr/bin/env python3
"""Report unreachable statements: a statement sitting after a return/raise/break/continue in the same
block is dead code, and in this tree that shape has a history (a botched edit once left a stale second
copy of a census body below its `return`, which still referenced a retired CASCADE_BAND key and so could
not have raised its way to attention). CPU only, stdlib only.

    .venv/bin/python tools/dead_after_return.py engine/gate.py engine/wave_flip_census.py
"""
from __future__ import annotations

import ast
import sys

TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def walk(body, path, out):
    seen = None
    for stmt in body:
        if seen is not None:
            out.append(f"{path}: line {stmt.lineno}: unreachable after "
                       f"{type(seen).__name__.lower()} at line {seen.lineno}")
        # Only the first offender per block is reported; the rest are the same mistake.
            continue
        if isinstance(stmt, TERMINATORS):
            seen = stmt
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                walk(inner, f"{path}:{stmt.lineno}", out)
        for field in ("handlers",):
            for h in getattr(stmt, field, []) or []:
                walk(h.body, f"{path}:{stmt.lineno}:handler", out)


def main(argv: list[str]) -> int:
    bad = 0
    for name in argv:
        src = open(name).read()
        out: list[str] = []
        walk(ast.parse(src).body, name, out)
        for line in out:
            print(line)
        bad += len(out)
    print(f"checked {len(argv)} file(s): {bad} unreachable statement(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
