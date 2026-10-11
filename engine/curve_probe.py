#!/usr/bin/env python3
"""Does the engine's disagreement with `dec64f` scale with the reference's own state magnitude?

Section (e) of `engine/gate.py --fixture dec64f` prints one row per step: the logits cosine on the
left, the REFERENCE's pinned GDN-state moments on the right. The excursion at steps 26..34 is where
the pinned `S n2` (sum of squares of the recurrent state, summed over the 6 GDN layers) roughly
halves, so the question this answers is whether the engine got worse there, or whether the thing it is
being compared against got smaller.

    .venv/bin/python engine/gate.py --fixture dec64f > tmp/gate_dec64f.log   # device: correctness only
    .venv/bin/python engine/curve_probe.py tmp/gate_dec64f.log               # CPU, reads that log

Reading: if the engine's ABSOLUTE error were roughly constant and only the reference's state energy
moved, then (1-cos) should track 1/(S n2) — and `err x (S n2)` should sit at one level across the whole
run. Measured: corr(1-cos, 1/(S n2)) 0.769, corr with the step index -0.033, and err x (S n2) median
0.1994 before the pinned stop token against 0.2074 after (ratio 1.04). So this is scale, not decay.

What this probe CANNOT settle, stated where the number is: it cannot separate "the reference's state got
small" from "the engine failed to maintain its state and the pin is showing that deficit". Both produce
this signature. Separating them needs the engine's own per-step state moments next to the pinned ones —
`decode_run` prints allocation (`STATE cap=…`), not moments. That instrument is named in
`engine/results.md` as the next measurement, not assumed away here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROW = re.compile(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)%\s+\|\s+(\d+)\s+([0-9.]+)")


def parse(log: Path):
    rows = []
    for line in log.read_text().splitlines():
        m = ROW.match(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(4)), float(m.group(6)), float(m.group(7))))
    if not rows:
        raise SystemExit(f"curve_probe: no (e)-table rows in {log}. Produce it with:"
                         " `engine/gate.py --fixture dec64f > that_file` (a device run; the probe"
                         " itself is CPU-only).")
    return rows


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/gate_dec64f.log")
    rows = parse(log)
    step = np.array([r[0] for r in rows])
    cos = np.array([r[1] for r in rows])
    n2 = np.array([r[2] for r in rows])
    dn2 = np.array([r[3] for r in rows])
    err = 1.0 - cos
    print(f"{len(rows)} steps parsed from {log}")
    print("pinned S n2   min %.0f (step %d) max %.0f (step %d)"
          % (n2.min(), step[n2.argmin()], n2.max(), step[n2.argmax()]))
    print("worst 1-cos   %.3e at step %d; median %.3e; median over the first 25 steps %.3e"
          % (err.max(), step[err.argmax()], np.median(err), np.median(err[:25])))
    print("corr(1-cos, 1/(S n2))      %.3f" % np.corrcoef(err, 1.0 / n2)[0, 1])
    print("corr(1-cos, S dn2)         %.3f" % np.corrcoef(err, dn2)[0, 1])
    print("corr(1-cos, step index)    %.3f" % np.corrcoef(err, step)[0, 1])
    # err * (S n2): constant if the absolute error is constant and only the reference shrank.
    prod = err * n2
    print("err x (S n2)  median %.4f  spread min %.4f max %.4f  (ratio max/min %.1fx)"
          % (np.median(prod), prod.min(), prod.max(), prod.max() / prod.min()))
    head, tail = np.median(prod[:25]), np.median(prod[35:])
    print("err x (S n2)  median over the 25 steps up to the pinned stop token %.4f,"
          " over the post-stop steps %.4f  -> ratio %.2f" % (head, tail, tail / head))
    return 0


if __name__ == "__main__":
    sys.exit(main())
