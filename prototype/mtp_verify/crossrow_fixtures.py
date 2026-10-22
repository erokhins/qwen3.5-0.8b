#!/usr/bin/env python3
"""Enumerate the pinned fixtures' prompt lengths (arm-1 fixture census, no GPU).

Written as a file rather than a shell loop because the CC safety net refuses a `python -c` whose source
is built by the shell; the population is derived from `data/refs/*/manifest.json`, never typed
(`AGENTS.md`: an admission over a fixture list typed from memory is not an admission).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    rows = []
    for man in sorted((ROOT / "data" / "refs").glob("*/manifest.json")):
        m = json.loads(man.read_text())
        pi = m.get("prefill_ids") or []
        srcs = [(s.get("kind"), len(s.get("ids") or [])) for s in (m.get("sources") or [])]
        rows.append((man.parent.name, len(pi), srcs))
    for name, n, srcs in rows:
        print(f"{name:20s} prefill_ids={n:6d} sources={srcs}")
    if not rows:
        print("NO FIXTURES FOUND — the population is empty, which cannot fail (AGENTS' empty-parse row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
