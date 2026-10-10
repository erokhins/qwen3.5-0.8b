#!/usr/bin/env python3
"""Negative controls - proof that the gates can fail.

A gate that has never failed proves nothing: it may be comparing nothing, or
normalizing an error away. This gate injects three bugs that are each one plausible
line of C++ away from the real implementation, via `specimpl.NEGATIVE`, and asserts
that a specific reference pin *stops* matching. Then it asserts the same pin does
match with no injection, so the failure is attributable to the bug and not to a
broken comparison.

  deinterleave      S05.1 splits q_proj as [q|gate]; this swaps the halves.
  conv_write_first  S07.2 says read the conv window, then roll it; this overwrites
                    the oldest slot before the taps read it (ring-buffer off-by-one).
  no_state_persist  the bug this project actually shipped for one commit: the GDN
                    recurrent state is not carried into the next decode step.
                    Prefill still matched to 1e-5; the tokens still matched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import speccheck.specimpl as SP  # noqa: E402
from speccheck.gates.run_gates import metrics, norm  # noqa: E402

# injection -> (fixture, canary pin that must stop matching)
CASES = {
    "deinterleave": ("dec8", "L03.attn.attn_out_pre_gate"),
    "conv_write_first": ("dec8", "decode0.L00.gdn.o_pre_norm"),
    "no_state_persist": ("dec8", "decode0.L12.gdn.o_pre_norm"),
}


def run_case(w, fixture: str, inject: frozenset, steps: int) -> dict:
    d = ROOT / "data" / "refs" / fixture
    ids = json.loads((d / "manifest.json").read_text())["prefill_ids"]
    SP.NEGATIVE = set(inject)
    try:
        f = SP.Forward(w, gdn_form="chunk")
        dbg_p, dbg_d = {}, {}
        f.greedy(ids, steps, dbg_p, dbg_d, stop_on_eos=False)
        out = dict(dbg_p)
        for step, sub in dbg_d.items():
            for k, v in sub.items():
                out[f"{step}.{k}"] = v
        return out
    finally:
        SP.NEGATIVE = set()


def cos_with(dbg: dict, pin: str, ref_dir: Path) -> float:
    r = norm(pin, np.load(ref_dir / (pin + ".npy")), True)
    e = norm(pin, dbg[pin].detach().float().numpy(), False)
    if r.shape != e.shape:
        e = e.reshape(r.shape) if r.size == e.size else e
    return metrics(e, r)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--clean-min", type=float, default=0.999,
                    help="clean run must match the canary at least this well")
    ap.add_argument("--inject-max", type=float, default=0.999,
                    help="injected run must fall below this on the canary")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    w = SP.Weights(Path(args.model))
    clean, fails = {}, {}
    for fixture in {f for f, _ in CASES.values()}:
        clean[fixture] = run_case(w, fixture, frozenset(), args.steps)
    for inject, (fixture, pin) in CASES.items():
        ref = ROOT / "data" / "refs" / fixture
        c0 = cos_with(clean[fixture], pin, ref)
        c1 = cos_with(run_case(w, fixture, frozenset({inject}), args.steps), pin, ref)
        ok = c0 >= args.clean_min and c1 < args.inject_max
        print("%-18s canary %-32s clean cos=%.6f  injected cos=%.6f  %s"
              % (inject, pin, c0, c1, "ok" if ok else "FAIL"))
        if c0 < args.clean_min:
            fails.append("%s: clean run does not match %s (cos=%.6f) - comparison "
                         "is broken, gate proves nothing" % (inject, pin, c0))
        if c1 >= args.inject_max:
            fails.append("%s: gate did NOT catch the injected bug (cos=%.6f)" % (inject, c1))
    if fails:
        print("NEGATIVE CONTROLS FAILED:")
        for f in fails:
            print("  ", f)
        return 1
    print("PASS: all %d injected bugs are caught, and the same pins pass when clean"
          % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
