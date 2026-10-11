#!/usr/bin/env python3
"""Prove a fresh reference dump reproduces the pinned one before it replaces it.

The rule in `AGENTS.md` / `specification/10`: reference dumps are pinned files, and a
fresh HF run must reproduce them. Extending `tools/reference_dump.py` (new hooks) or
bumping transformers/torch is exactly when this matters — a hook that changes numerics,
or a reference that silently changed its mind, invalidates every gate downstream.

Every tensor present in BOTH directories is compared **byte for byte** (the dumps are
F32, so equal bits means equal values). Names only in the pinned dir are missing pins
(a regression); names only in the fresh dir are additions (fine, reported). The
manifest is compared key by key, minus `pins`, whose note text is expected to change.

    .venv/bin/python tools/check_dump_reproduction.py data/refs/smoke /tmp/refs_smoke_mlp

Exit 0 = every pinned tensor reproduced exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path


def digest(p: Path) -> str:
    h = sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pinned", help="the committed fixture dir (data/refs/<tag>)")
    ap.add_argument("fresh", help="the directory the modified tool just wrote")
    ap.add_argument("--show", type=int, default=8, help="names to list per category")
    args = ap.parse_args()

    pinned, fresh = Path(args.pinned), Path(args.fresh)
    for d in (pinned, fresh):
        if not d.is_dir():
            print("not a directory: %s" % d)
            return 2
    old = {p.name for p in pinned.glob("*.npy")}
    new = {p.name for p in fresh.glob("*.npy")}
    print("%s: %d pins | %s: %d pins" % (pinned, len(old), fresh, len(new)))

    missing = sorted(old - new)
    added = sorted(new - old)
    if missing:
        print("MISSING in fresh dump (%d), e.g.:" % len(missing))
        for n in missing[:args.show]:
            print("  ", n)
    if added:
        kinds: dict[str, int] = {}
        for n in added:
            kinds[n.split(".", 1)[1].rsplit(".", 1)[0] if "." in n else n] = \
                kinds.get(n.split(".", 1)[1].rsplit(".", 1)[0] if "." in n else n, 0) + 1
        print("ADDED by fresh dump (%d): %s" % (len(added), dict(sorted(kinds.items()))))

    differ = []
    for name in sorted(old & new):
        if digest(pinned / name) != digest(fresh / name):
            differ.append(name)
    if differ:
        print("BYTE-DIFFERING pins (%d of %d compared), e.g.:" % (len(differ), len(old & new)))
        for n in differ[:args.show]:
            print("  ", n)
    else:
        print("all %d shared pins are byte-identical" % len(old & new))

    mp_path, mf_path = pinned / "manifest.json", fresh / "manifest.json"
    if mp_path.exists() and mf_path.exists():
        mp, mf = json.loads(mp_path.read_text()), json.loads(mf_path.read_text())
        bad = [k for k in set(mp) | set(mf) if k != "pins" and mp.get(k) != mf.get(k)]
        if bad:
            print("MANIFEST fields that changed: %s" % sorted(bad))
        else:
            print("manifest identical apart from `pins`")
    else:
        print("manifest.json missing on one side - not compared")

    if missing or differ:
        print("\nFAIL: the fresh dump does not reproduce the pinned fixture")
        return 1
    print("\nPASS: pinned fixture reproduced exactly%s"
          % (" (+%d new pins)" % len(added) if added else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
