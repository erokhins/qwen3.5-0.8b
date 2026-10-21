#!/usr/bin/env python3
"""Remove the two route allowances that bind on no shipping path and reddened one that does.

Priced 2026-10-21 by `prototype/maxsc_allowance`: `wide640f` (1.79/0.29) and `wide1024f` (1.62/0.35).
They were withdrawn the same day, on two measurements taken after they were written:

1. **They cannot bind where the engine runs.** `rule_for_step()` -> `cascade_floor()` takes
   `max(cap, 2e-3 / 3 %)` and every derived cap sits 31-73x below that, so under the shipping
   `--gemm lt` route the gate prints "floored at the measured cascade" and the allowance is inert.
2. **They do bind where the engine does not run, and wrongly there.** `route_allowance.json` records no
   route, so a margin priced from an `lt` log is applied to the `hand` route, whose deficits run
   1.1-1.6x larger - turning 14 band lines red that the fallback used to pass.

So the files bought nothing on the shipping path and cost green on the other one. Deleting an allowance is
strictly a return to the fallback that was already judging those fixtures, which is the safe direction: no
threshold is loosened on the route that ships, and none is tightened on the one that does not.

The removal is recorded in `bench/band-history.jsonl` as `removed` events (the ledger has that event type
for exactly this), so "which band was in force when" stays answerable from the file and not from memory.

    .venv/bin/python tools/withdraw_route_allowance.py --check
    .venv/bin/python tools/withdraw_route_allowance.py --apply
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ("wide640f", "wide1024f")
REASON = ("priced under --gemm lt but the file records no route: inert on lt (cascade floor dominates by "
          "31-73x) and it reddened 14 hand-route band lines. Removed pending a route field on the format.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    hist = ROOT / "bench" / "band-history.jsonl"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    events, removed_files = [], []
    for fx in FIXTURES:
        path = ROOT / "data" / "refs" / fx / "route_allowance.json"
        if not path.exists():
            print(f"{fx:<12} no allowance file (already withdrawn?)")
            continue
        doc = json.loads(path.read_text())
        for key, old in (("margin", doc.get("margin")), ("margin_rel", doc.get("margin_rel"))):
            events.append({"ts": ts, "event": "removed", "file": str(path.relative_to(ROOT)),
                           "key": key, "old": old, "new": "absent",
                           "by": "tools/withdraw_route_allowance.py", "decision": REASON})
        removed_files.append((fx, path))
        print(f"{fx:<12} margin={doc.get('margin')} margin_rel={doc.get('margin_rel')} -> fallback")
    if not args.apply:
        print(f"\ncheck only: {len(removed_files)} file(s), {len(events)} ledger line(s). Pass --apply.")
        return 0
    for fx, path in removed_files:
        path.unlink()
    with hist.open("a") as f:
        for ev in events:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
    print(f"applied: {len(removed_files)} allowance(s) removed, {len(events)} removed-event(s) appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
