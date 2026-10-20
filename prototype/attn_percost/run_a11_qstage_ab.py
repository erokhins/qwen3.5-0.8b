#!/usr/bin/env python3
"""A11's BET, measured: feed Q from LDS or from global at EQUAL residency, and see what the Q channel is worth.

`prototype/attn_percost/A11-attribution.md` BET: "arm A = shipped unstaged8 padded with a dummy array to
LDS_Block_Size = 32,768 so both arms run the same blocks/CU; arm B = --attn-fattn-qstage stage. If the
LDS-fed arm wins >= 25 % of the traced attention phase at equal residency, the bet is falsified and R2's
premise is live."

Both arms here declare LDS 32,768 and reserve 248 VGPR, so residency is held fixed and the ONLY difference is
where Q comes from each round:

  pad_unstaged  /tmp/ab_res/pad_run   (default qstage = unstaged8: Q re-read from global every round)
  stage         /tmp/ab_res/base_run --attn-fattn-qstage stage (Q staged in LDS, zero Q global traffic)

Reuses tools/prefill_sweep.py's prompt builder and its parse_result so the numbers are the same quantity the
sweep books (ms_total with --steps 1). Interleaved per rep; a rep that fails to parse aborts the run rather
than being skipped, because a check whose input can silently be absent is the recorded failure mode.
"""
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from prefill_sweep import build_prompt, parse_result  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

LANES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "16384").split(",")]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODEL = str(ROOT / "models/Qwen3.5-0.8B")
ARMS = [("pad_unstaged", "/tmp/ab_res/pad_run", []),
        ("stage", "/tmp/ab_res/base_run", ["--attn-fattn-qstage", "stage"])]

tok = AutoTokenizer.from_pretrained(MODEL)
for lanes in LANES:
    ids = [str(t) for t in build_prompt(tok, lanes)]
    vals = {name: [] for name, _, _ in ARMS}
    for rep in range(REPS):
        for name, binary, extra in ARMS:
            out = Path(tempfile.mkdtemp(prefix=f"a11q_{name}_"))
            proc = subprocess.run([binary, "--model", MODEL, "--ids", ",".join(ids),
                                   "--steps", "1", "--out", str(out)] + extra,
                                  capture_output=True, text=True)
            shutil.rmtree(out, ignore_errors=True)
            if proc.returncode != 0:
                sys.exit(f"NO-TIMING: {name} rc={proc.returncode} at rep {rep}\n{proc.stdout[-500:]}")
            try:
                vals[name].append(parse_result(proc.stdout)["ms_total"])
            except Exception as exc:  # a missing parse is not a zero and not a skip
                sys.exit(f"NO-TIMING: {name} unparseable at rep {rep}: {exc}")
    print(f"lanes={lanes} reps={REPS}")
    best = {n: min(v) for n, v in vals.items()}
    for name, _, _ in ARMS:
        print(f"  {name:13s} {vals[name]} best={best[name]:.1f} ms")
    gain = (best["pad_unstaged"] - best["stage"]) / best["pad_unstaged"] * 100.0
    print(f"  Q-from-LDS gain at equal residency: {gain:+.2f} %   "
          f"(BET falsified only at >= +25.00 %; A11's own bound predicted <= 12 %)")
