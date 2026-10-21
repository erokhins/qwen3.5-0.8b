#!/usr/bin/env python3
"""Does each priced allowance still describe the reference and the pins in front of it?

    .venv/bin/python prototype/maxsc_allowance/check_inputs.py

Why this exists: `engine/gate.py::allowance_problems` re-checks four provenance fields when it loads a
cached route allowance — `prefill_ids`, `feed_ids`, `n_steps`, and the SHA-256 of
`speccheck/specimpl.py` — and a mismatch is a FAIL, not a fallback, because "a stale spread is a band
nobody measured". A fixture whose allowance went stale is therefore judged by the flat floors AND
carries a red line saying so, which is a different situation from a fixture that never had an allowance
at all. The census (`census.py`) prints whether a file exists; this prints whether it still BINDS.

Read-only. `engine/gate.py` must be told which fixture is being read (`set_fixture`) before its
`allowance_inputs`/`allowance_problems` can be reused, exactly as the gate does; nothing here
re-derives a margin, and nothing is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import gate as egate

REFS = ROOT / "data" / "refs"


def main() -> int:
    print(f"`speccheck/specimpl.py` now: "
          f"{(ROOT / 'speccheck' / 'specimpl.py').read_bytes() and ''}"
          f"sha256={__import__('hashlib').sha256((ROOT / 'speccheck' / 'specimpl.py').read_bytes()).hexdigest()}")
    bad_any = 0
    for d in sorted(p for p in REFS.iterdir() if p.is_dir()):
        ap = d / egate.ALLOWANCE_FILE
        if not ap.exists():
            continue
        man = json.loads((d / "manifest.json").read_text())
        n_steps = max(len(man.get("greedy_continuation") or []) - 1, 0)
        egate.set_fixture(d.name)
        doc = json.loads(ap.read_text())
        want = egate.allowance_inputs(n_steps)
        got = doc.get("inputs", {})
        diffs = [f"{k}: priced {str(got.get(k))[:20]!r} != now {str(v)[:20]!r}"
                 for k, v in want.items() if got.get(k) != v]
        self_problems = [p for p in egate.allowance_problems(doc, n_steps)
                         if "inputs." not in p]
        ok = not diffs and not self_problems
        bad_any += 0 if ok else 1
        print(f"  {d.name:<10} priced {doc.get('generated', {}).get('date', '?')}"
              f"  steps {len(doc.get('steps', []))}/{n_steps}"
              f"  margins {doc.get('margin')}/{doc.get('margin_rel')}"
              f"  {'BINDS' if ok else 'DOES NOT BIND'}")
        for x in diffs + self_problems:
            print(f"      - {x}")
    print(f"\n{bad_any} of the priced allowances do not bind the reference/pins as they stand now."
          " engine/gate.py treats each of those as a FAIL line, not a silent fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
