#!/usr/bin/env python3
"""Diagnose a subagent 'needs attention' notice before deciding to interrupt the child.

Two worker runs on this box were reported stalled at exactly 300 s of no observed activity, and both
times the child was mid-work. The event log holds the reason: a `compaction_start` lands immediately
before the `needs_attention` control event, because this box serves its own subagents' tokens on the
same iGPU, so a threshold compaction is a multi-minute model call with no tool activity in it.

Two causes make a child go quiet on this box, and only one of them is a wedge:

1. **a threshold compaction**, which is a multi-minute generation because the children are served by this same
   checkpoint on the same iGPU; or
2. **a long single generation** with no tool call in it — observed 2026-10-16 as a `high`-thinking turn on a
   large context, immediately after the child had successfully edited a file, i.e. perfectly healthy.

Usage: tools/subagent_stall_check.py <run-id-prefix> [<run-id-prefix> ...]
Prints compaction count and reason, needs_attention count, and how many notices fall inside a compaction window.
A notice *outside* any window is not automatically a wedge: read the last real tool result in
`output-0.log` / `events.jsonl` and see whether it was a successful edit or a fault. Read-only.
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
    # A notice is compaction-caused if its own timestamp falls inside a compaction window. Two things make the
    # naive versions wrong: control events are appended by a watcher, so FILE ORDER is not chronological (one
    # notice sat two lines after a `compaction_start` in the file yet happened 989 s BEFORE any compaction), and
    # the two event kinds carry their time in different fields (`.observedAt` on compaction events, `.event.ts`
    # on control events). An unclosed window — compaction still running when the log was read — extends to now.
    def ts_of(ev):
        if ev.get("type") == "subagent.control":
            return (ev.get("event") or {}).get("ts")
        return ev.get("observedAt")

    windows = []
    for i in comp:
        end = next((ts_of(events[j]) for j in range(i + 1, len(kinds)) if kinds[j] == "compaction_end"), None)
        windows.append((ts_of(events[i]), end))
    no_ts = 0
    in_comp = []
    for i in attn:
        t = ts_of(events[i])
        if t is None:
            no_ts += 1
            continue
        if any(a is not None and t >= a and (b is None or t <= b) for a, b in windows):
            in_comp.append(i)
    print(f"{prefix[:8]}  events={len(events)}  compactions={len(comp)} {reasons}  "
          f"needs_attention={len(attn)}  attention_explained_by_compaction={len(in_comp)}"
          + (f"  windows={[(a, b) for a, b in windows]}" if len(in_comp) != len(attn) or no_ts else "")
          + (f"  notices_without_ts={no_ts}" if no_ts else "")
          + ("" if not no_ts and len(in_comp) == len(attn) else
             "  <-- UNEXPLAINED: at least one notice had no compaction in flight; read the last real tool "
             "result before deciding to interrupt"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    for arg in sys.argv[1:]:
        report(arg)
