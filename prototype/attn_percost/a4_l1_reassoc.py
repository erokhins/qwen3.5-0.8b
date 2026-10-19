#!/usr/bin/env python3
"""A4-L1 Gate 1 numerics: how far does the two-accumulator roll move the score, on the PINNED data?

NOT a new mirror. It imports `prototype/attn_decode/floor_probe.py` — this tree's existing CPU mirror of the
attention score dot, the instrument §10.1's F1 floor was taken with (`prototype/attn_decode/results.md` F1 row:
two legal re-orderings of the same products, 15,592 scores, 0 flip) — and borrows its `rows()`,
`a_from_scores()` and `bf16_ulp.report()` to ask the one question A4-L1's numerics note needs:

    shipped roll : 16 k-tile partials summed ASCENDING into one accumulator (`:576-583`, Gate 0 §1.2)
    the L1 roll  : `acc0` over the even tiles, `acc1` over the odd, ONE add after the roll

Output is printed in F1's own shape — flips over a named sample size, then the output-element effect with
cos/meanrel and the band floor — so a reader can put the two rows side by side. Reuses floor_probe's exact case
population (dec8 decode steps + p300 long-cache rows), not a list typed from memory (AGENTS' admission trap).

WHAT IT CANNOT SEE, stated rather than hidden: inside ONE 16-dim k-tile the `v_wmma` adds its 16 products in an
order the matrix hardware does not expose, so this models the INTER-tile association — the one L1 changes — and
holds the intra-tile partial at numpy's own sum order. A `0 flip` answer here is evidence about the association
L1 changes; it is not, and cannot be, proof that the device output is bit-identical. CPU only: no device, no
timing, no ms.

    .venv/bin/python prototype/attn_percost/a4_l1_reassoc.py [--fixture dec8]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEC = REPO / "prototype" / "attn_decode"
NO_BAND = 1e-12          # same no-excuse band floor_probe uses, so the floors are comparable


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FP = _load("a4_floor_probe", DEC / "floor_probe.py")
R = FP.R
bf16_ulp = FP.bf16_ulp


def tile_partials(q, k):
    """The k-tile partials of the dot, 16 dims per tile — the granularity `v_wmma` accumulates at."""
    d = (q[None, :] * k).astype(np.float32)                     # [lanes, DH]
    return d.reshape(k.shape[0], d.shape[1] // 16, 16).sum(axis=2, dtype=np.float32)


def roll_one(q, k):
    """SHIPPED association: one accumulator, the 16 tile partials ascending."""
    p = tile_partials(q, k)
    acc = np.zeros(p.shape[0], dtype=np.float32)
    for t in range(p.shape[1]):
        acc = (acc + p[:, t]).astype(np.float32)
    return acc


def roll_two(q, k):
    """L1 association: two accumulators over alternating tiles, one add after the roll."""
    p = tile_partials(q, k)
    a0 = np.zeros(p.shape[0], dtype=np.float32)
    a1 = np.zeros(p.shape[0], dtype=np.float32)
    for t in range(0, p.shape[1], 2):
        a0 = (a0 + p[:, t]).astype(np.float32)
        a1 = (a1 + p[:, t + 1]).astype(np.float32)
    return (a0 + a1).astype(np.float32)


def scores(qr, K, n, roll):
    """The BF16 score rows of every head under one `roll` association — floor_probe's own construction,
    with the WMMA tile association in place of its warp tree."""
    out = []
    for i in range(R.N_Q):
        j = i // (R.N_Q // R.N_KV)
        out.append(torch.tensor(roll(qr[i].float().numpy(), K[j, :n].float().numpy()) * R.SCALE,
                                dtype=torch.bfloat16))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--long-fixture", default="p300")
    ap.add_argument("--long-rows", default="63,127,191,255,276")
    a = ap.parse_args()
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed — the ULP comparator is broken, so no number here means"
                         " anything (AGENTS: a uint16-view comparator has a dead branch)")
    d = R.ROOT / "data" / "refs" / a.fixture
    steps = ["decode%d" % i for i in range(8)]
    cases = [R.decode_case(d, s, l) for l in R.ATTN_LAYERS for s in steps]
    dl = R.ROOT / "data" / "refs" / a.long_fixture
    long_rows = []
    if dl.exists():
        long_rows = [int(x) for x in a.long_rows.split(",")]
        cases += R.prefill_cases(dl, R.ATTN_LAYERS[0], long_rows)
    if not cases:
        raise SystemExit(f"REFUSE: 0 cases from {a.fixture}/{a.long_fixture} — a comparison over an empty"
                         " population cannot fail")
    pre = [(c, *FP.rows(c)) for c in cases]
    print(f"{len(cases)} cases ({len(cases)-len(long_rows)} decode from {a.fixture} + {len(long_rows)}"
          f" long-cache rows from {a.long_fixture}) — the same population F1's floor was taken on")

    for form in ("full", "online"):
        flips = scores_n = elems = adiff = big = 0
        ulp, band, cosmin, mrmax = 0, 0.0, 1.0, 0.0
        for c, qr, kr, vv, gate, K, V, n in pre:
            s1 = scores(qr, K, n, roll_one)
            s2 = scores(qr, K, n, roll_two)
            flips += int(sum(int((x != y).sum()) for x, y in zip(s1, s2)))
            scores_n += sum(x.numel() for x in s1)
            a1 = FP.a_from_scores(s1, V, n, form)
            a2 = FP.a_from_scores(s2, V, n, form)
            rep = bf16_ulp.report(a2, a1, cancel_rel=NO_BAND)
            elems += a1.numel()
            adiff += rep["ndiff"]
            big += rep["n_bad"]
            ulp = max(ulp, rep["maxulp"])
            band = max(band, rep["worst_off_rel"])
            cosmin, mrmax = min(cosmin, rep["cos"]), max(mrmax, rep["meanrel"])
        print(f"\nL1 re-association of the score roll, one accumulator vs two, output form `{form}`:")
        print(f"    {scores_n} scores: {flips} round to a different BF16 ({100.0*flips/scores_n:.5f}%)")
        print(f"    over {elems} output elements: {adiff} differ ({100.0*adiff/elems:.5f}%), {big} LARGE"
              f" (>1 ULP outside a no-excuse band), worst {ulp} ULP, cos >= {cosmin:.8f},"
              f" meanrel <= {mrmax:.3e}, band floor {band:.1e}")
    print("\nBands that would judge it (quoted from §10.1): G4 attention cos >= 0.999, meanrel <= 5 %;"
          " G3i-class isolated-per-layer cos >= 0.9999, meanrel <= 1 %.")


if __name__ == "__main__":
    main()
