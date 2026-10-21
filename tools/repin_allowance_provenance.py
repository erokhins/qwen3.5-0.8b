#!/usr/bin/env python3
"""Re-pin the provenance field of every cached route allowance: bytes digest -> code digest.

The breakage this repairs: `engine/gate.py` identified the reference by `sha256(file bytes)`, so a commit
that moved documents out of `bench/` and touched ONE COMMENT LINE in `speccheck/specimpl.py` made five
fixtures' allowances "stale" and turned the shipping gate red (`dec300f`, `dec64f`, `dec64t`, `mt32f`,
`wide520f`). Proof it was a false positive: bytes differ `e5acb50c…` -> `cf28b17f…`, code-only AST digests
IDENTICAL `a0d032ed…` both sides. See `engine/provenance.py`.

This script moves NO BAND. It rewrites exactly one field per allowance file -
`inputs.specimpl_sha256` - and refuses to touch anything else:

    .venv/bin/python tools/repin_allowance_provenance.py            # dry run: what would change
    .venv/bin/python tools/repin_allowance_provenance.py --apply    # write, and append band-history

Every re-pin is recorded as an `applied` line in `bench/band-history.jsonl` naming the old bytes digest,
the new code digest, and why - because "nobody edited a band" must be provable from the ledger, not from
this commit message. The script asserts each file's margins are byte-for-byte unchanged after the edit,
and refuses a file whose stored digest is already the code digest.
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
DECISION = ("provenance re-pin: reference identified by code digest, not bytes. Gate went red on a "
            "comment-only edit to speccheck/specimpl.py; code digests proved identical, so the cached "
            "spreads are still the ones that were measured. No margin moved.")


def snapshot(doc: dict) -> str:
    """Everything a re-pin must NOT change: the whole document except the one provenance field.
    Comparing `margin` alone would let `margin_rel`, `steps` or `derivation` drift unnoticed, which is
    exactly the silent-band-drift this script exists to disprove."""
    d = json.loads(json.dumps(doc))
    d.get("inputs", {}).pop("specimpl_sha256", None)
    return json.dumps(d, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    code = provenance.specimpl_digest()
    events, touched = [], []
    for path in sorted((ROOT / "data" / "refs").glob("*/route_allowance.json")):
        doc = json.loads(path.read_text())
        before = snapshot(doc)
        inputs = doc.get("inputs", {})
        old = inputs.get("specimpl_sha256")
        name = path.parent.name
        if old == code:
            print(f"{name:<12} already code-pinned, untouched")
            continue
        if old is None:
            raise SystemExit(f"{name}: no specimpl_sha256 field at all - this script does not invent one")
        inputs["specimpl_sha256"] = code
        doc["inputs"] = inputs
        if snapshot(doc) != before:
            raise SystemExit(f"{name}: internal error, something besides the provenance field moved")
        events.append({
            "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": "applied", "file": str(path.relative_to(ROOT)), "key": "inputs.specimpl_sha256",
            "old": old, "new": code, "by": "tools/repin_allowance_provenance.py",
            "decision": DECISION})
        touched.append((path, doc))
        print(f"{name:<12} re-pin {old[:12]} -> {code[:12]}  (margins untouched)")

    if not args.apply:
        print(f"\ndry run: {len(touched)} file(s) would be re-pinned; pass --apply to write.")
        return 0
    for path, doc in touched:
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    with HISTORY.open("a") as f:
        for ev in events:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
    print(f"applied: {len(touched)} allowance(s) re-pinned, {len(events)} band-history line(s) appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
