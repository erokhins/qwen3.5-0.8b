#!/usr/bin/env python3
"""G7 (S10.1) - end-to-end greedy token equality against the reference.

The tensor gates say "the numbers are close"; this gate says the thing users
actually depend on: the sampled ids are identical, token for token, including the
stop behaviour (A-10). It reads `generated_ids` from a speccheck dump and compares
with `manifest.greedy_continuation`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default="smoke,p64,p300,dec8")
    ap.add_argument("--gdn-prefill", default="chunk", choices=["recurrent", "chunk"])
    ap.add_argument("--attn-softmax", default="full", choices=["full", "running"])
    ap.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    args = ap.parse_args()

    fails = []
    for tag in [t for t in args.fixtures.split(",") if t]:
        m = json.loads((ROOT / "data" / "refs" / tag / "manifest.json").read_text())
        want = list(m["greedy_continuation"])
        ids = m["prefill_ids"]
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=True) as tf:
            cmd = [args.python, str(ROOT / "speccheck" / "specimpl.py"),
                   "--prompt-ids-file", str(ROOT / "data" / "refs" / tag / "manifest.json"),
                   "--decode", str(max(len(want) - 1, 0)), "--gdn-prefill", args.gdn_prefill,
                   "--attn-softmax", args.attn_softmax, "--dump", tf.name]
            subprocess.run(cmd, check=True, capture_output=True)
            got = [int(t) for t in np.load(tf.name)["generated_ids"]]
        # the reference continuation is unbounded; compare the overlap, and require
        # at least the reference length unless the engine stopped on EOS (A-10)
        n = min(len(got), len(want))
        bad = [i for i in range(n) if got[i] != want[i]]
        eos = [i for i, t in enumerate(got) if t in (248046, 248044)]
        stopped = bool(eos) and len(got) <= len(want)
        ok = not bad and (len(got) >= len(want) or stopped)
        print("%-6s ref=%d got=%d  first mismatch=%s  %s"
              % (tag, len(want), len(got), ("idx %d" % bad[0]) if bad else "none",
                 "ok" if ok else "FAIL"))
        if not ok:
            fails.append("%s: %s (got %s vs ref %s)"
                         % (tag, ("mismatch at %s" % bad[0]) if bad else "length",
                            got[:12], want[:12]))
    if fails:
        print("G7 FAILED:")
        for f in fails:
            print("  ", f)
        return 1
    print("G7 PASS: greedy ids identical to the reference on %s "
          "(gdn=%s, attn=%s)" % (args.fixtures, args.gdn_prefill, args.attn_softmax))
    return 0


if __name__ == "__main__":
    sys.exit(main())
