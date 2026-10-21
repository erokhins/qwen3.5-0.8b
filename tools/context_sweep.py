#!/usr/bin/env python3
"""Context sweep: engine decode cost vs lane count, for the performance ledger.

Owned by the main agent: this is a timing harness, and per `AGENTS.md` a number only enters
`engine/performance-status.md` if it was taken blocking, with zero subagents active, under the GPU
lock, best-of-N, with its spread stated. A worker must not run this and must not quote its output.

Prompts are synthetic on purpose. The question is how per-token cost grows with the §05 attention
read, which depends on the lane count and not on the tokens' meaning, so a repeated paragraph is
the honest input: it is stated, reproducible, and contains no stop id (checked, not assumed).
Each point runs `--steps` generation steps at the tail of a prompt sized so that the lane count
sits inside `lanes ± steps/2`; the reported `ms_per_step` is `decode_run`'s own steady_clock mean.

Usage (main agent):
    tools/gpu_lock.sh acquire <tag>
    .venv/bin/python tools/context_sweep.py --lanes 1024,2048,4096 --steps 32 --runs 3
    tools/gpu_lock.sh release <tag>
"""
import argparse
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOP_IDS = {248046}
# Two paragraphs of ordinary prose, repeated to length. Kept inline so the sweep is reproducible
# without touching any network or fixture.
PARA = (
    "The laboratory sat at the edge of the field, behind a hedge that had been cut too many times "
    "and had grown thick out of habit rather than design. Inside, the benches were clean, the "
    "instrument cases were labelled in a careful hand, and the kettle had been switched off at the "
    "wall by someone who expected to come back to it. Nobody in the building expected a result "
    "before Thursday, which is why the afternoon unfolded at the pace of the light moving across "
    "the floor rather than at the pace of any particular experiment. "
)


def build_prompt(tok, lanes: int) -> list[int]:
    """Token ids of repeated prose, truncated to exactly `lanes` tokens, with no stop id inside."""
    one = tok.encode(PARA, add_special_tokens=False)
    units = lanes // max(1, len(one)) + 2
    ids = tok.encode(PARA * units, add_special_tokens=False)[:lanes]
    bad = sorted(set(ids) & STOP_IDS)
    if bad:
        raise SystemExit(
            f"prompt for {lanes} lanes contains stop id(s) {bad}; pick different filler text"
        )
    return ids


def parse_result(stdout: str) -> dict:
    """The fields of `decode_run`'s RESULT line, as floats."""
    line = next(l for l in stdout.splitlines() if l.startswith("RESULT"))
    return {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", line)}


def run_once(binary: str, model: str, ids: list[str], steps: int, out: Path) -> tuple[float, float]:
    cmd = [
        binary, "--model", model, "--ids", ",".join(ids),
        "--steps", str(steps), "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"decode_run rc={proc.returncode}\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
    got = parse_result(proc.stdout)
    if float(got.get("stopped", 0)):
        raise SystemExit("run stopped early; lengthen the prompt")
    return float(got["ms_total"]), float(got["prefix_len"])


def lanes_of(got: dict) -> float:
    return float(got.get("prefix_len", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=str(ROOT / "engine/build/decode_run"))
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-0.8B"))
    ap.add_argument("--lanes", default="1024,2048,4096")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    # Differential timing: `RESULT ms_per_step` averages the prompt rows in, and since P2/P4-P6 those
    # rows are a wave with amortised launches, so the mean is no longer a decode cost. Two runs that
    # differ only in --steps isolate it: (ms_total(B) - ms_total(A)) / (steps_B - steps_A) is the cost
    # of a generation step at that context, with the same prefill in both.
    n_lo, n_hi = args.steps, args.steps + args.steps
    print("| lanes at last step | marginal ms/token best of %d | all runs | spread | tok/s |" % args.runs)
    print("|---|---|---|---|---|")
    for lanes in [int(x) for x in args.lanes.split(",")]:
        ids = [str(t) for t in build_prompt(tok, lanes - n_hi)]
        vals = []
        ends = 0.0
        for i in range(args.runs):
            out_a = Path(tempfile.mkdtemp(prefix=f"ctxsweep_{lanes}_{i}a_"))
            out_b = Path(tempfile.mkdtemp(prefix=f"ctxsweep_{lanes}_{i}b_"))
            a, _ = run_once(args.binary, args.model, ids, n_lo, out_a)
            b, end = run_once(args.binary, args.model, ids, n_hi, out_b)
            shutil.rmtree(out_a, ignore_errors=True)
            shutil.rmtree(out_b, ignore_errors=True)
            vals.append((b - a) / (n_hi - n_lo))
            ends = end
        best, worst = min(vals), max(vals)
        spread = (worst - best) / best * 100.0
        print(
            "| %d | **%.3f** | %s | %.2f %% | %.1f |"
            % (
                int(ends),
                best,
                " / ".join("%.3f" % v for v in vals),
                spread,
                1000.0 / best,
            )
        )
    print(
        "\nmarginal cost of a generation step at the stated lane count: two runs differing only in"
        " --steps (%d vs %d), same prompt and same prefill in both. Prompts are repeated prose,"
        " truncated, stop-id free." % (n_lo, n_hi)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
