#!/usr/bin/env python3
"""Non-regression measurement for the wide attention default: wide=0 vs wide=1, same binary, paired.

The kickoff's non-regression line is warm throughput at 512 and 4,096 lanes; the wide form is only supposed to
own long-context rows, so this measures whether flipping the default costs anything at short widths — a cost
there would be a trade to report, not to average away. Same protocol as the pvf16 timing script (it reuses
tools/prefill_sweep.py's prompt builder and parser so the quantity is the one the ledger books), same refusal:
a rep that will not parse aborts rather than being skipped.

Main agent only: blocking, zero subagents, one gpu_lock hold, best-of-N with the spread stated.
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

LANES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "512,4096,16384").split(",")]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODEL = str(ROOT / "models/Qwen3.5-0.8B")
BIN = str(ROOT / "engine/build/decode_run")
ARMS = [("wide0", ["--attn-fattn-wide", "0"]), ("wide1", ["--attn-fattn-wide", "1"])]

tok = AutoTokenizer.from_pretrained(MODEL)
for lanes in LANES:
    ids = [str(t) for t in build_prompt(tok, lanes)]
    vals = {n: [] for n, _ in ARMS}
    for rep in range(REPS):
        for name, extra in ARMS:
            out = Path(tempfile.mkdtemp(prefix=f"wd_{name}_"))
            proc = subprocess.run([BIN, "--mtp-accept", "0", "--mtp-draft", "0", "--model", MODEL, "--ids", ",".join(ids), "--steps", "1",
                                   "--attn-fattn", "1", "--out", str(out)] + extra,
                                  capture_output=True, text=True)
            shutil.rmtree(out, ignore_errors=True)
            if proc.returncode != 0:
                sys.exit(f"NO-TIMING: {name} rc={proc.returncode} rep {rep}\n{proc.stdout[-400:]}")
            try:
                vals[name].append(parse_result(proc.stdout)["ms_total"])
            except Exception as exc:
                sys.exit(f"NO-TIMING: {name} unparseable rep {rep}: {exc}")
    best = {n: min(v) for n, v in vals.items()}
    print(f"lanes={lanes}")
    for name, _ in ARMS:
        tp = lanes / best[name] * 1000.0
        print(f"  {name} {vals[name]} best={best[name]:.1f} ms = {tp:.1f} tok/s")
    d = (best["wide1"] - best["wide0"]) / best["wide0"] * 100.0
    print(f"  wide1 vs wide0: {d:+.2f} %  (negative = wide faster)")
