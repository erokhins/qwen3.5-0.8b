#!/usr/bin/env python3
"""Do the first N pinned rows of one fixture match another fixture's rows, BIT FOR BIT?

Why this is a check and not a curiosity: `speccheck/fixtures/wide640f.txt` is the first 640 tokens of
`speccheck/fixtures/wide1024f.txt` (640 = 10 x the §04.7 chunk size, and 512 + 128 lanes), so the two
pins were made to be nested. If they are, then the narrow fixture's >512-lane rows sit inside the same
corpus the 1024-row dispute lives in, and the two pins corroborate each other. If they are NOT, one of
two things is true and both are news: either the prompts are no longer prefixes of each other (a prompt
file was edited, and the recipe's "narrow half" claim is now false), or the reference's own chunked scan
is not length-local — its output for row i depends on how much text follows — which would undercut every
chunked-prefill claim §04.7 makes. Either way this tool is the thing that says so.

What is compared: the first N rows of `logits_rows.npy` as raw integers (the pin is a uint16 BF16
bitcast, so equality here IS bit equality; no float tolerance is involved), plus the first N of
`prefill_ids`.

THE NEGATIVE CONTROL IS `--shift 1`, and it is part of the contract, not an extra: a comparison that has
never been shown to fail is a comparison that reports PASS whatever it reads. Shifted by one row the two
fixtures are genuinely different rows of the same prompt, so the tool must go red and name where; if it
does not, the byte comparison is comparing something with itself.

    .venv/bin/python tools/check_fixture_prefix.py data/refs/wide640f data/refs/wide1024f --rows 640
    .venv/bin/python tools/check_fixture_prefix.py data/refs/wide640f data/refs/wide1024f --rows 639 \
        --shift 1        # must exit 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def rows_path(d: Path) -> Path:
    p = d / "logits_rows.npy"
    if not p.exists():
        raise SystemExit(f"no {p} — this tool only judges fixtures pinned with"
                         " `reference_dump.py --prefill-logits all`")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("narrow", help="the fixture whose rows should be the prefix")
    ap.add_argument("wide", help="the fixture that should carry it")
    ap.add_argument("--rows", type=int, required=True,
                    help="how many leading rows must agree (the narrow fixture's prompt length)")
    ap.add_argument("--shift", type=int, default=0, metavar="K",
                    help="NEGATIVE CONTROL: compare against the wide fixture's rows K further on"
                         " (K=1 must FAIL for two fixtures that really are nested, because row i and"
                         " row i+1 of one prompt are different rows). It exists so this tool's"
                         " PASS can be shown to mean something; it is not a knob for finding a"
                         " comparison that agrees")
    args = ap.parse_args()

    narrow, wide = Path(args.narrow), Path(args.wide)
    man_n = json.loads((narrow / "manifest.json").read_text())
    man_w = json.loads((wide / "manifest.json").read_text())
    n = args.rows
    ids_n = man_n["prefill_ids"][:n]
    ids_w = man_w["prefill_ids"][:n]
    ids_ok = ids_n == ids_w
    print(f"prompt ids: first {n} {'agree' if ids_ok else 'DO NOT AGREE'}"
          + ("" if ids_ok else f" — first difference at "
             f"{next(i for i, (a, b) in enumerate(zip(ids_n, ids_w)) if a != b)}"))

    a = np.load(rows_path(narrow), mmap_mode="r")
    b = np.load(rows_path(wide), mmap_mode="r")
    print(f"logits_rows: {narrow} is {a.shape}, {wide} is {b.shape}")
    if a.shape[1] != b.shape[1]:
        print(f"FAIL: vocabulary width differs ({a.shape[1]} vs {b.shape[1]}); these are not the same model")
        return 1
    if a.shape[0] < n or b.shape[0] < n + args.shift:
        print(f"FAIL: a fixture pins fewer rows than the comparison needs"
              f" ({a.shape[0]}, {b.shape[0]}, shift {args.shift})")
        return 1
    ra, rb = np.asarray(a[:n]), np.asarray(b[args.shift:args.shift + n])
    if args.shift:
        print(f"  [shift {args.shift}] the wide side starts at row {args.shift}, so a PASS here would"
              " mean two different rows of one prompt are bit-identical — that is the bug, not the goal")
    bad = np.flatnonzero((ra != rb).any(axis=1))
    print(f"rows compared {n} x {a.shape[1]} elements"
          f" ({ra.dtype}, a BF16 bitcast — integer equality is bit equality)")
    if len(bad):
        first = int(bad[0])
        d = np.flatnonzero(ra[first] != rb[first])
        print(f"FAIL: {len(bad)} of {n} rows differ; first is row {first}, {len(d)} elements,"
              f" e.g. index {int(d[0])}: {int(ra[first][d[0]]):#06x} vs {int(rb[first][d[0]]):#06x}")
        if args.shift:
            print("      This is the NEGATIVE CONTROL behaving correctly: with --shift the two sides are"
                  " different rows, so differing is the only acceptable answer. Re-run without --shift"
                  " for the real claim.")
        else:
            print("      Either the prompt files stopped being prefixes of each other, or the reference's"
                  " chunked scan is not length-local. Both are findings; do not file this under 'band'.")
        return 1
    ok = ids_ok
    print(("PASS" if ok else "FAIL") + f": rows 0..{n - 1} of {wide.name} are bit-identical to {narrow.name}"
          + ("" if ok else ", but the prompt ids themselves do not agree, so the row agreement is an"
                           " accident of two different prompts"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
