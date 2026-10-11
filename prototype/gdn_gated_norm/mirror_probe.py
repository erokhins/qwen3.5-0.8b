#!/usr/bin/env python3
"""Which cast order does §04.5 / A-01 actually pin? A bit-exact mirror probe against the dumps.

The audit that found `rmsnorm_gated` unchecked at bit level is why this file exists: the only
comparison anywhere was `run_isolated.py`'s cos ≥ 0.9999 band, and a cast-order difference cannot
move that number (§10.2 says so for every rounding-policy question in this model). So this probe
compares BF16 BIT PATTERNS of six readings of

    ô = RMSnorm(o) ⊙ w_norm ;  ô = ô ⊙ silu(z)         §04.5, "cast order fixed by A-01"

against `decode{N}.L{ii}.gdn.gated_out`, with `gdn.o_pre_norm` and `in_proj_z.out` as inputs. The
variants are fixed before the data is read; nothing here is tuned to pass.

Result (results.md has the table): the pins carry BF16(n) → BF16( BF16(n)·BF16(w) ) →
BF16( a·silu(z) ) — three BF16 boundaries, the affine in the activation dtype with the weight cast
to it. A-01's "applies w_norm and silu(z) in F32" and `speccheck.specimpl.rmsnorm_gated`, which
implements that reading, both miss 105 683 of 294 912 elements on dec8.

Run: .venv/bin/python prototype/gdn_gated_norm/mirror_probe.py [fixture ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype" / "proj_gemv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import operands as ops  # noqa: E402  (checkpoint + pin readers, already gated by proj_gemv)
from speccheck.specimpl import rmsnorm_gated  # noqa: E402  (the oracle, unchanged)

H_N, D_H = 16, 128          # §00.3: 16 value heads × d_v 128 = the 2048 lanes §04.5 normalizes
EPS = 1e-6                  # §00.2
bf = lambda t: t.to(torch.bfloat16)  # noqa: E731


def variants(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor):
    """(label, BF16 result) per reading. o/z are BF16-valued F32, w is the F32 checkpoint weight."""
    o = o.reshape(H_N, D_H)
    z = z.reshape(H_N, D_H)
    wf = w.float().reshape(D_H)
    silu_z = torch.nn.functional.silu(z)                  # F32, HF's z·σ(z) order
    n_f32 = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
    n_bf = bf(n_f32)
    yield ("POLICY (kernel): bf16(n) · bf16(w) rounds, then ·silu(z) rounds",
           bf(bf(n_bf * bf(wf)).float() * silu_z))
    yield ("A-01 literal: bf16(n), then *w and *silu(z) in F32 (specimpl's reading)",
           bf((n_bf.float() * wf) * silu_z))
    yield ("single cast: f32 norm · w · silu(z), one cast at the end",
           bf((n_f32 * wf) * silu_z))
    yield ("affine stays F32, weight cast: bf16(n·bf16(w)) then ·silu(z)",
           bf((n_bf.float() * bf(wf).float()).float() * silu_z))
    yield ("gate rounded: silu(z) goes through a BF16 slot first",
           bf(bf(n_bf * bf(wf)).float() * bf(silu_z).float()))
    yield ("speccheck.specimpl.rmsnorm_gated (the oracle, unchanged)",
           # Hand it BF16 inputs, the way the model calls it: its final `.to(o.dtype)` is what
           # rounds the result, so passing F32 in would compare an unrounded tensor to the pins.
           rmsnorm_gated(o.to(torch.bfloat16), z.to(torch.bfloat16), w.float()))


def cases(fixture: str, steps: list[str]):
    """(o, z, w, gated_out) per (step, GDN layer), all BF16-valued.

    Which layers are GDN layers is read off the pins themselves (existence of
    `gdn.o_pre_norm`), which is §10.5's rule — the pinned names are the check to trust, and
    `is_attn_layer()` keys on `q_proj.out`, which the dumper does not emit for prefill."""
    d = ROOT / "data" / "refs" / fixture
    for l in range(24):
        for st in steps:
            pre = (st + ".") if st else ""
            try:
                o = ops.pin(d, f"{pre}L{l:02d}.gdn.o_pre_norm")
                z = ops.pin(d, f"{pre}L{l:02d}.in_proj_z.out")
                g = ops.pin(d, f"{pre}L{l:02d}.gdn.gated_out")
            except FileNotFoundError:
                continue          # not a GDN layer (an attention layer has no linear_attn.*)
            w = ops.ckpt_tensor(ops.LAYER_PREFIX % l + "linear_attn.norm.weight")
            yield (o.reshape(-1, D_H).float(), z.reshape(-1, D_H).float(), w, g.reshape(-1))


def sweep(fixture: str, steps: list[str]) -> dict:
    """{variant label: dict(n_elements, ndiff, worst_case_ndiff, cases)}"""
    out = {}
    for (o, z, w, ref) in cases(fixture, steps):
        for name, got in variants(o, z, w):
            r = out.setdefault(name, {"ndiff": 0, "n": 0, "worst": 0, "cases": 0})
            nd = int((got.reshape(-1) != ref).sum())
            r["ndiff"] += nd
            r["n"] += ref.numel()
            r["cases"] += 1
            r["worst"] = max(r["worst"], nd)
    return out


FIXTURES = [("dec8", ["decode%d" % s for s in range(8)]),
            ("prefix24dec8", ["decode%d" % s for s in range(8)]),
            ("smoke", ["decode0", "decode1"]),
            ("p300", [""])]          # prefill positions: bare names, no decode steps in p300


def main(argv: list[str]) -> int:
    want = argv or [f for f, _ in FIXTURES]
    rc = 0
    for fixture, steps in FIXTURES:
        if fixture not in want:
            continue
        res = sweep(fixture, steps)
        if not res:
            print("%-14s no pins found — skipped" % fixture)
            continue
        names = list(res.keys())
        n = res[names[0]]["n"]
        print("\n%s — %d cases, %d BF16 elements compared against gdn.gated_out" %
              (fixture, res[names[0]]["cases"], n))
        for name in names:
            r = res[name]
            print("  %-64s ndiff %7d / %d  worst case %4d  %s"
                  % (name, r["ndiff"], r["n"], r["worst"],
                     "BIT-EXACT" if r["ndiff"] == 0 else "refused by the pins"))
        exact = [nm for nm in names if res[nm]["ndiff"] == 0]
        if exact != [names[0]]:
            print("  FAIL: expected exactly one bit-exact reading, the kernel's; got %s" % exact)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
