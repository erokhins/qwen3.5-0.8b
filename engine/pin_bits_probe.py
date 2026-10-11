#!/usr/bin/env python3
"""Are the pinned logits BF16-representable? — the claim behind `--logits-bits bf16`.

`tools/reference_dump.py --logits-bits bf16` stores logits as BF16 **bit patterns** in a uint16
payload, which halves the deepest line in the fixture (30.8 MiB instead of 61.5 for dec64f). That is
only a storage decision and not a loosened tolerance if the reference's logits were BF16 values to
begin with — the model is BF16, so its final matmul produces BF16-representable numbers. This checks
it over every pinned logit value of every step of both deep and shallow fixtures, rather than asserting
it from a spot check.

The rounding is done by torch's own `.to(torch.bfloat16)` (the same call `engine/logits_floor.py`
uses), not by hand-rolled bit arithmetic: `prototype/bf16_ulp.py` exists precisely because a BF16
comparator written on a `uint16` view has a silent dead branch (`np.where(x < 0, …)` is never true for
`uint16`), and a bug here would make the fixture look lossless when it is not.

No device, no engine, no timing: ~18 M values round-tripped through torch, a few seconds of CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

import gate                                                    # noqa: E402


def pin_logit_names(man: dict) -> list[str]:
    """`logits_last.npy` (the prefill) plus one `decode{s}.logits_last.npy` per decode step — the
    same set of names `gate.py` judges against, derived from the manifest's own continuation length
    rather than a hardcoded count."""
    return (["logits_last.npy"]
            + [f"decode{s}.logits_last.npy" for s in range(len(man["greedy_continuation"]) - 1)])


def check_fixture(name: str) -> tuple[int, int, float]:
    gate.set_fixture(name)
    man = json.loads((gate.REFS / "manifest.json").read_text())
    total = bad = 0
    worst = 0.0
    for pin_name in pin_logit_names(man):
        if not (gate.REFS / pin_name).exists():
            print(f"  [note] {name}: no pin named {pin_name} — skipped")
            continue
        a = gate.load_pin(pin_name).astype(np.float32)
        # A numeric cast on purpose: this IS the rounding under test, not a bitcast. (`.view()` would
        # be the bitcast, and confusing the two is the trap in prototype/gdn_gated_norm/results.md §8.)
        rounded = torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()
        neq = a != rounded
        total += int(a.size)
        bad += int(neq.sum())
        if neq.any():
            worst = max(worst, float(np.abs(a - rounded)[neq].max()))
    return total, bad, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default="dec8,dec64f")
    args = ap.parse_args()
    failures = []
    for name in args.fixtures.split(","):
        total, bad, worst = check_fixture(name.strip())
        verdict = "bit-lossless" if bad == 0 else f"{bad} values moved, worst {worst:.3g}"
        print(f"{name}: {total} pinned logit values, {bad} not BF16-exact — {verdict}")
        if bad:
            failures.append(name)
    if failures:
        print(f"PIN BITS FAIL — {failures}: --logits-bits bf16 would be a tolerance change, not a"
              " storage choice, and the deep fixture's cos bands would be judged against a pin that"
              " moved. Say so; do not re-widen the pins and call it fixed.")
        return 1
    print("PIN BITS PASS — every pinned logit value is BF16-representable, so the BF16 bit storage"
          " in dec64f loses nothing and the gate compares against the reference's own numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
