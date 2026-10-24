#!/usr/bin/env python3
"""TTFT sweep: prompt-phase cost vs context, for the performance ledger (plan Q5 / P10).

`RESULT ms_total` covers the prompt wave plus the requested generation steps, and excludes weight
load (engine/loader_test times that separately, and tools/cold_restore.sh gives it its own row).
With --steps 1 this is as close as the engine gets to time-to-first-token: the prompt wave plus one
priced step. Same protocol as tools/context_sweep.py and the same rule: main agent only, blocking,
zero subagents active, under the GPU lock, best-of-N with the spread stated.

Usage (main agent):
    tools/gpu_lock.sh acquire <tag>
    .venv/bin/python tools/prefill_sweep.py --lanes 64,256,1024,3072 --runs 3
    tools/gpu_lock.sh release <tag>
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_sweep import build_prompt, parse_result  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=str(ROOT / "engine/build/decode_run"))
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-0.8B"))
    ap.add_argument("--lanes", default="64,256,1024,3072")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    print("| prompt lanes | ms to first sampled token, best of %d | all runs | spread | prompt tok/s |" % args.runs)
    print("|---|---|---|---|---|")
    for lanes in [int(x) for x in args.lanes.split(",")]:
        ids = [str(t) for t in build_prompt(tok, lanes)]
        vals = []
        for i in range(args.runs):
            out = Path(tempfile.mkdtemp(prefix=f"ttft_{lanes}_{i}_"))
            proc = subprocess.run(
                [args.binary, "--mtp-accept", "0", "--mtp-draft", "0", "--model", args.model, "--ids", ",".join(ids), "--steps", "1",
                 "--out", str(out)],
                capture_output=True, text=True,
            )
            shutil.rmtree(out, ignore_errors=True)
            if proc.returncode != 0:
                raise SystemExit(f"decode_run rc={proc.returncode}\n{proc.stdout[-600:]}")
            vals.append(parse_result(proc.stdout)["ms_total"])
        best, worst = min(vals), max(vals)
        print("| %d | **%.1f** | %s | %.2f %% | %.1f |" % (
            lanes, best, " / ".join("%.1f" % v for v in vals),
            (worst - best) / best * 100.0, lanes / best * 1000.0))
    print("\nms_total with --steps 1: prompt wave plus one priced step; weight load excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
