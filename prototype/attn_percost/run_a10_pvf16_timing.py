#!/usr/bin/env python3
"""Paired throughput for `--attn-fattn-pvf16 0` vs `1`, same binary, interleaved, one lock.

Same protocol as `run_a11_qstage_ab.py` (it reuses `tools/prefill_sweep.py`'s prompt builder and parser so the
number is the quantity the ledger books), and the same refusal: a rep that will not parse aborts the run
instead of being skipped, because a check whose input can be silently absent is the recorded failure mode.
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
BIN = str(ROOT / "engine/build/decode_run")
ARMS = [("pvf16_f32", ["--attn-fattn-pvf16", "0"]),
        ("pvf16_f16", ["--attn-fattn-pvf16", "1"])]

tok = AutoTokenizer.from_pretrained(MODEL)
for lanes in LANES:
    ids = [str(t) for t in build_prompt(tok, lanes)]
    vals = {n: [] for n, _ in ARMS}
    for rep in range(REPS):
        for name, extra in ARMS:
            out = Path(tempfile.mkdtemp(prefix=f"pv_{name}_"))
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
    print(f"lanes={lanes} reps={REPS}")
    for name, _ in ARMS:
        print(f"  {name:11s} {vals[name]} best={best[name]:.1f} ms")
    d = (best["pvf16_f16"] - best["pvf16_f32"]) / best["pvf16_f32"] * 100.0
    print(f"  f16 arm vs f32 arm: {d:+.2f} %  (negative = f16 faster)")
