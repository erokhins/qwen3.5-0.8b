#!/usr/bin/env python3
"""Diagnose a subagent 'needs attention' notice before deciding to interrupt the child.

Two worker runs on this box were reported stalled at exactly 300 s of no observed activity, and both
times the child was mid-work. The event log holds the reason: a `compaction_start` lands immediately
before the `needs_attention` control event, because this box serves its own subagents' tokens on the
same iGPU, so a threshold compaction is a multi-minute model call with no tool activity in it.

Usage: tools/subagent_stall_check.py <run-id-prefix> [<run-id-prefix> ...]
Prints compaction count and reason, needs_attention count, and whether the last event before each
attention notice was a compaction. Read-only.
"""
import json
import sys
import glob


def report(prefix: str) -> None:
    matches = glob.glob(f"/tmp/pi-subagents-uid-1000/async-subagent-runs/{prefix}*")
    if not matches:
        print(f"{prefix}: no run directory")
        return
    path = f"{sorted(matches)[-1]}/events.jsonl"
    events = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError as exc:
        print(f"{prefix}: cannot read {path}: {exc}")
        return

    kinds = [e.get("type") for e in events]
    comp = [i for i, k in enumerate(kinds) if k == "compaction_start"]
    attn = [i for i, k in enumerate(kinds) if k == "subagent.control"
            and (events[i].get("event") or {}).get("type") == "needs_attention"]
    reasons = [events[i].get("reason") for i in comp]
    after_comp = sum(1 for i in attn if i > 0 and kinds[i - 1] == "compaction_start")
    print(f"{prefix[:8]}  events={len(events)}  compactions={len(comp)} {reasons}  "
          f"needs_attention={len(attn)}  attention_directly_after_compaction={after_comp}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    for arg in sys.argv[1:]:
        report(arg)
