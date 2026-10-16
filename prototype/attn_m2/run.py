#!/usr/bin/env python3
"""run.py — plan writer and driver for `prototype/attn_m2` (R1: attention row-tiling at M = 2).

WHAT THIS DOES. Writes device plans whose WAVES sit on the boundaries M = 2 moves — a pair of adjacent
query rows at exactly MAXSC and MAXSC+1 lanes (one one-pass, one streaming), pairs straddling lane
blocks 2 and 3, the widest pair a 1024-token context holds, the shortest pairs a wave has, and the
single-row decode steps where the pair's second row does not exist — then runs `./attn_m2`, which
launches BOTH forms over the same operands and prints a per-row bit-diff count.

WHAT THIS DOES NOT DO, AND THAT IS DELIBERATE.

  * It does NOT acquire `tools/gpu_lock.sh`. Three gate drivers in this repo self-lock, and a caller that
    holds the lock and then runs one of them deadlocks it silently (AGENTS.md trap row). This one does
    not, so the CALLER holds the lock around it — HANDOFF.md step 0 says so in those words.
  * It takes no timing and prints no ms. The harness has no timer at all; the paired A/B that prices this
    route is an engine measurement owned by the main agent (HANDOFF.md step 6).
  * It does not widen a check. Bit-identity means zero differing elements per row; anything else prints a
    ROWDIFF line with the row, its lane count, and the count.

Run (repo root, inside one gpu_lock hold, blocking):
    .venv/bin/python prototype/attn_m2/run.py plans
    .venv/bin/python prototype/attn_m2/run.py ab
    .venv/bin/python prototype/attn_m2/run.py negative --neg maskwide
CPU-only, no device, no lock:
    .venv/bin/python prototype/attn_m2/m2_model.py selftest|compare|negative|traffic
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "attn_decode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reference as R                       # noqa: E402
import attn_tile_model as T                 # noqa: E402  MAXSC, windows_of
import m2_model as M                        # noqa: E402  the mirror owns the case builders

HERE = Path(__file__).resolve().parent
WORK = Path("/tmp/attn_m2")   # scratch lives outside the tree, like `attn_decode/run.py`
BIN = HERE / "attn_m2"
MAXSC = T.MAXSC

# The device cases. Content comes from `m2_model.wide_plan` — real pinned rows cycled over a POPULATED
# cache — because no fixture on this box has a prompt long enough to reach these positions (§20.2:
# `wide1024f` has long decode rows but no per-layer prefill stages), and a bit-identity A/B needs
# identical operands, not HF ground truth. The numeric anchor for the streaming route stays `wide1024f`'s
# pinned decode rows, which are the tail cases below.
#
# (tag, start position, nrows, why)
WAVE_SPECS = [
    ("pair-at-maxsc", MAXSC - 1, 4, "lanes 512..515: pair 0 straddles one-pass/streaming"),
    ("maxsc-minus-1", MAXSC - 2, 4, "lanes 511..514: the same boundary one row earlier"),
    ("pair-just-above", MAXSC, 8, "lanes 513..520, split 2, unequal spans: the V-overlap case"),
    ("straddles-block-2", 2 * MAXSC - 1, 4, "lanes 1024..1027, split 3"),
    ("straddles-block-3", 3 * MAXSC - 1, 4, "lanes 1536..1539, split 4"),
    ("pair-widest-1024", 1020, 4, "lanes 1021..1024: the union at block 0 is exactly MAXSC"),
    ("pair-shortest", 0, 8, "lanes 1..8: the pairs with the least to share (0.778, not 0.5)"),
]


def emit(cases, tag: str) -> Path:
    """Write one plan file in the `attn_decode` plan format (12 fields, NROWS optional)."""
    indir = WORK / "in" / tag
    outdir = WORK / "out" / tag
    indir.mkdir(parents=True, exist_ok=True)
    lines = [R.write_case(c, indir, outdir / c["name"]) for c in cases]
    plan = WORK / f"plan-{tag}.txt"
    plan.write_text("\n".join(" ".join(f) for f in lines) + "\n")
    print(f"  plan {plan.name}: {len(lines)} case(s)")
    return plan


def wave_cases(dl: Path, layer: int) -> list[tuple[str, Path]]:
    out = []
    for tag, start, nrows, why in WAVE_SPECS:
        case = M.wide_plan(dl, layer, start, nrows, f"m2.{tag}.L{layer:02d}")
        first, last = case["lvalid"], case["lvalid"] + case["nrows"] - 1
        print(f"  {tag:<19} {nrows} rows, lanes {first}..{last}, split {T.windows_of(last)}"
              f"  | {why}")
        out.append((tag, emit([case], tag)))
    return out


def tail_cases(dl: Path, layer: int) -> list[tuple[str, Path]]:
    """Single-row decode waves at 1025..1028 lanes: the pair's SECOND row does not exist, which is the
    `g.exists == false` path — the one path M = 2 adds that a full pair cannot exercise. These are also
    the only rows above 1024 lanes on this box that carry HF pins at all."""
    cases = []
    for step in [f"decode{i}" for i in range(8)]:
        try:
            c = R.decode_case(dl, step, layer)
        except FileNotFoundError:
            continue
        c["name"] = f"m2.tail.{step}.L{layer:02d}"
        cases.append(c)
    print(f"  {'tail-odd':<19} {len(cases)} single-row decode steps above 1024 lanes")
    return [("tail", emit(cases, "tail"))] if cases else []


def build() -> bool:
    print("$ bash prototype/attn_m2/build.sh   (compile-only; no device is touched)")
    r = subprocess.run(["bash", str(HERE / "build.sh")], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return False
    print("  " + r.stdout.strip())
    return True


def run_plan(plan: Path, neg: str = "none", split: int = 1) -> int:
    (WORK / "dump").mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "--plan", str(plan), "--neg", neg, "--split", str(split),
           "--out", str(WORK / "dump")]
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="device plan writer and A/B driver for attn_m2")
    ap.add_argument("what", choices=["plans", "ab", "negative"])
    ap.add_argument("--fixture", default="p300",
                    help="fixture with per-layer prefill stage pins (p300: 277 rows, cycled to any "
                         "context length by wide_plan)")
    ap.add_argument("--tail-fixture", default="wide1024f",
                    help="fixture with pinned decode rows above 1024 lanes")
    ap.add_argument("--layer", type=int, default=R.ATTN_LAYERS[0])
    ap.add_argument("--neg", default="maskwide",
                    help="injection for `negative`: readpast|dropcurrent|dimhalf|gqaoff|slotk|slotv|"
                         "maskwide|partrow|partalias (headcarry is mirror-only)")
    ap.add_argument("--split", type=int, default=1,
                    help="lane grid depth; raised automatically to split_for_lanes(widest row)")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    plans = wave_cases(Path("data/refs") / args.fixture, args.layer) \
        + tail_cases(Path("data/refs") / args.tail_fixture, args.layer)
    if args.what == "plans":
        print("\nno device touched. HANDOFF.md numbers the runs; this driver does NOT self-lock.")
        return 0
    if not build():
        return 1
    rc = 0
    for tag, plan in plans:
        rc |= run_plan(plan, neg=("none" if args.what == "ab" else args.neg), split=args.split)
    print(f"\n{'FAIL' if rc else 'PASS'}: {args.what} neg={args.neg}"
          " — exit 1 means at least one case differs; see the ROWDIFF lines above.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
