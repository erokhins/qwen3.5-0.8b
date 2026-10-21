#!/usr/bin/env python3
"""Put the provenance re-pin record where it belongs, and out of the band ledger.

`tools/repin_allowance_provenance.py` first tried to record the re-pin as `applied` lines in
`bench/band-history.jsonl`. That ledger refused, and it was right: it is a NUMERIC band ledger -
`tools.band_history` rejects a non-numeric value for a key - and a provenance digest is not a band, so
the gate then refused to run over those lines ("is not a number (write 'absent' for a key that does not
exist)"). Recording a non-band fact in a band ledger is the same mistake as a check that cannot fail:
it looks like provenance and is really noise that blocks the real judge.

So this script does two things and nothing else:
  1. removes every band-history line whose key is `inputs.specimpl_sha256` (and reports the count it
     removed and the count it kept, so it cannot silently eat legitimate lines);
  2. writes the re-pin note INTO each affected allowance file, beside the field it explains.

    .venv/bin/python tools/fix_allowance_repin_record.py --check   # report only
    .venv/bin/python tools/fix_allowance_repin_record.py --apply
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
import provenance  # noqa: E402

HISTORY = ROOT / "bench" / "band-history.jsonl"
BAD_KEY = "inputs.specimpl_sha256"
NOTE = {
    "by": "tools/repin_allowance_provenance.py",
    "field": "inputs.specimpl_sha256",
    "from": "sha256 of speccheck/specimpl.py bytes",
    "to": "sha256 of speccheck/specimpl.py AST with docstrings stripped (engine/provenance.py)",
    "why": "the gate went red on a comment-only edit to the reference; code digests were identical, so "
           "the cached spread is still the one that was measured. No margin moved. Not recorded in "
           "bench/band-history.jsonl because that ledger is numeric and this is not a band.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    lines = HISTORY.read_text().splitlines(keepends=True)
    keep, drop = [], []
    for ln in lines:
        s = ln.strip()
        if s and json.loads(s).get("key") == BAD_KEY:
            drop.append(ln)
        else:
            keep.append(ln)
    print(f"band-history: {len(drop)} provenance line(s) to remove, {len(keep)} legitimate line(s) kept")

    code = provenance.specimpl_digest()
    noted = []
    for path in sorted((ROOT / "data" / "refs").glob("*/route_allowance.json")):
        doc = json.loads(path.read_text())
        if doc.get("inputs", {}).get("specimpl_sha256") != code:
            continue
        if "provenance_repin" in doc:
            print(f"{path.parent.name:<12} note already present")
            continue
        doc["provenance_repin"] = dict(NOTE, date=datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), digest=code)
        noted.append((path, doc))
        print(f"{path.parent.name:<12} note to add")

    if not args.apply:
        print("\ncheck only; pass --apply")
        return 0
    HISTORY.write_text("".join(keep))
    for path, doc in noted:
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"applied: ledger cleaned, {len(noted)} note(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
