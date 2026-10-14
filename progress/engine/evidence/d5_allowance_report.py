"""Re-derive every shipped route allowance's numbers from the files themselves, for `results.md`.

Each `data/refs/<fx>/route_allowance.json` carries its own `route_*`, `allowance_*` and
`engine_*_when_priced` columns, so every figure quoted in the write-up can be recomputed here instead of
typed from memory. It also reports, per fixture: the stored margins vs the margins its own columns derive
(ceil2), how many steps are judged tighter than the flat §10.1 G5 floor, and the headroom at the worst
ratio step — the boundary-by-construction quantity D-3 is about.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLAT_COS = 5e-4          # 1 - 0.9995, engine/gate.py BAND["cos_hard"]
FLAT_REL = 0.05          # engine/gate.py BAND["meanrel_hard"]


def ceil2(x):
    return math.ceil(round(x, 6) * 100) / 100.0


for fx in ["dec64f", "dec64t", "dec300f", "mt32f"]:
    d = json.loads((ROOT / "data" / "refs" / fx / "route_allowance.json").read_text())
    steps = d["steps"]
    cos_r = [(r["engine_one_cos_when_priced"] / r["route_one_cos"], r["step"]) for r in steps]
    rel_r = [(r["engine_meanrel_when_priced"] / r["route_meanrel"], r["step"]) for r in steps]
    wc, wr = max(cos_r), max(rel_r)
    tight_cos = sum(1 for r in steps if r["allowance_one_cos"] <= FLAT_COS)
    tight_rel = sum(1 for r in steps if r["allowance_meanrel"] <= FLAT_REL)
    worst = [r for r in steps if r["step"] == wc[1]][0]
    print(f"== {fx}: stored margin {d['margin']:.2f}/{d['margin_rel']:.2f}"
          f"  basis={d.get('margin_basis', 'ABSENT (pre-D-2 file)')}"
          f"\n   own derivation from its own columns: cos {wc[0]:.4f} at step {wc[1]} -> ceil2"
          f" {ceil2(wc[0]):.2f}; meanrel {wr[0]:.4f} at step {wr[1]} -> ceil2 {ceil2(wr[0]):.2f}"
          f"\n   steps: {len(steps)}; tighter than flat G5 on cos at {tight_cos}, on meanrel at"
          f" {tight_rel} of {len(steps)}"
          f"\n   cos caps: tightest {min(r['allowance_one_cos'] for r in steps):.4e}"
          f" widest {max(r['allowance_one_cos'] for r in steps):.4e}"
          f"\n   route 1-cos: median {sorted(r['route_one_cos'] for r in steps)[len(steps)//2]:.3e}"
          f" min {min(r['route_one_cos'] for r in steps):.3e}"
          f" max {max(r['route_one_cos'] for r in steps):.3e}"
          f"\n   worst-ratio step {wc[1]}: engine 1-cos {worst['engine_one_cos_when_priced']:.6e} over"
          f" spread {worst['route_one_cos']:.6e} = {wc[0]:.4f}x, cap"
          f" {worst['allowance_one_cos']:.6e}, headroom"
          f" {worst['allowance_one_cos'] - worst['engine_one_cos_when_priced']:.3e}"
          f" ({(worst['allowance_one_cos'] - worst['engine_one_cos_when_priced']) / worst['allowance_one_cos']:.2%}"
          " of the cap)"
          f"\n   flat-G5 verdict at that step: deficit"
          f" {worst['engine_one_cos_when_priced']:.4e} would"
          f" {'FAIL' if worst['engine_one_cos_when_priced'] > FLAT_COS else 'PASS'} the flat floor")
