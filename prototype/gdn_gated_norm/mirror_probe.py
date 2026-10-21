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
to it, and `speccheck.specimpl.rmsnorm_gated` reproduces that exactly when it is called the way
speccheck calls it: `Weights` casts every model weight to the activation dtype (specimpl.py:70-86,
`keep_f32_scalars=False` by default), so the oracle receives BF16(w), never the raw F32 tensor.

What the weight dtype alone moves, on the same 294 912 dec8 elements with identical `o`/`z`:

    w in BF16, product rounded in BF16  (what speccheck does)      ndiff      0
    w BF16-rounded, then widened to F32 (A-01's letter)            ndiff  74 860
    w straight from the F32 checkpoint  (a dtype speccheck lacks)  ndiff 105 683

An earlier revision of this probe handed the oracle the third of those — `w.float()` off the
checkpoint — and reported its 105 683 as an oracle disagreement with the pins. That claim was
retracted in `progress/2026-10-11/gate-audit.md` (G2 amendment); the row is now the BF16-weight call and the file
asserts it at ndiff 0, so the retraction is a live check rather than prose.

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
    """(key, label, BF16 result) per reading.

    o/z are BF16-valued F32; w is the F32 checkpoint tensor, so each row states the weight dtype it
    uses: `w_bf` is the dtype `speccheck.Weights` holds model weights in, raw `wf` is a dtype the
    model never produces and only exists here to show what that dtype costs.
    """
    o = o.reshape(-1, D_H)
    z = z.reshape(-1, D_H)
    wf = w.float().reshape(D_H)
    w_bf = bf(wf)
    silu_z = torch.nn.functional.silu(z)                  # F32, HF's z·σ(z) order
    n_f32 = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
    n_bf = bf(n_f32)
    yield ("kernel", "POLICY (kernel): bf16(n) · bf16(w) rounds, then ·silu(z) rounds",
           bf(bf(n_bf * w_bf).float() * silu_z))
    yield ("a01_f32_w", "A-01 literal, weight in raw F32 (not a dtype speccheck holds)",
           bf((n_bf.float() * wf) * silu_z))
    yield ("single_cast", "single cast: f32 norm · w · silu(z), one cast at the end",
           bf((n_f32 * wf) * silu_z))
    yield ("a01_bf16_w", "A-01 literal, weight pre-rounded to BF16: *bf16(w) in F32, one cast at the end",
           bf((n_bf.float() * w_bf.float()) * silu_z))
    yield ("gate_rounded", "gate rounded: silu(z) goes through a BF16 slot first",
           bf(bf(n_bf * w_bf).float() * bf(silu_z).float()))
    yield ("specimpl", "speccheck.specimpl.rmsnorm_gated, called as speccheck calls it (BF16 weight)",
           # BF16 in for o and z the way the model calls it — its final `.to(o.dtype)` is what
           # rounds the result, so passing F32 in would compare an unrounded tensor to the pins —
           # and BF16 for w, which is how `Weights` stores it. This row is G2's oracle half.
           rmsnorm_gated(o.to(torch.bfloat16), z.to(torch.bfloat16), w_bf))


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
            # One D_H-wide row is one value head (§04.5 reduces per head, not over the flattened
            # 2048), so a pin is whole tokens of H_N×D_H: H_N rows for a decode step, H_N×T for a
            # prefill. Reshaping to (H_N, D_H) instead works for decode and dies on prefill.
            assert o.numel() % (H_N * D_H) == 0, (fixture, l, st, o.shape)
            yield (o.reshape(-1, D_H).float(), z.reshape(-1, D_H).float(), w, g.reshape(-1))


def sweep(fixture: str, steps: list[str]) -> dict:
    """{variant key: dict(label, n_elements, ndiff, worst_case_ndiff, cases)}, keys in yield order."""
    out = {}
    for (o, z, w, ref) in cases(fixture, steps):
        for key, label, got in variants(o, z, w):
            r = out.setdefault(key, {"label": label, "ndiff": 0, "n": 0, "worst": 0, "cases": 0})
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
        keys = list(res.keys())
        n = res[keys[0]]["n"]
        print("\n%s — %d cases, %d BF16 elements compared against gdn.gated_out" %
              (fixture, res[keys[0]]["cases"], n))
        for key in keys:
            r = res[key]
            print("  %-64s ndiff %7d / %d  worst case %4d  %s"
                  % (r["label"], r["ndiff"], r["n"], r["worst"],
                     "BIT-EXACT" if r["ndiff"] == 0 else "refused by the pins"))
        # Two readings must be bit-exact, and they must be the same formula: the kernel's policy
        # and the oracle called the way speccheck calls it. Any third tie is news, and the dtype
        # ladder must stay ordered — the three reconciliation figures are a weight-dtype effect on
        # identical inputs, so if the raw-F32 row ever stops being worse than the BF16-rounded one,
        # that explanation is wrong and must not be repeated in a docstring.
        exact = [k for k in keys if res[k]["ndiff"] == 0]
        problems = []
        if exact != ["kernel", "specimpl"]:
            problems.append("expected exactly the kernel and the correctly-called oracle "
                            "to be bit-exact; got %s" % [res[k]["label"] for k in exact])
        if not 0 < res["a01_f32_w"]["ndiff"] > res["a01_bf16_w"]["ndiff"]:
            problems.append("weight-dtype ladder is not ordered: raw F32 %d vs BF16-rounded %d"
                            % (res["a01_f32_w"]["ndiff"], res["a01_bf16_w"]["ndiff"]))
        for p in problems:
            print("  FAIL: %s" % p)
            rc = 1
        print("  weight-dtype ladder: bf16 weight %d  |  bf16-rounded-then-F32 %d  |  raw F32 %d"
              % (res["specimpl"]["ndiff"], res["a01_bf16_w"]["ndiff"], res["a01_f32_w"]["ndiff"]))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
