#!/usr/bin/env python3
"""What cos is a step's logits vector legally allowed to sit at? CPU-only; no device, no engine.

The question came from a real gate result: the first end-to-end pass put the engine's step-1 logits at
cos 0.999886 against the pinned `decode1.logits_last` — above the gated G5 floor (0.9995), below the
advisory G6 figure (0.9999). "Expected cascade noise" and "a small real discrepancy" both fit that
sentence, so the sentence needs a number under it: how far apart are two *legal* computations of the
very same logits vector when the only thing that differs is the order the additions happen in?

That is what this measures. It takes the reference's own inputs to the last op — the pinned residual
`decode{s}.L23.s_out` (§03.4's final residual), `lm.norm.weight`, and the tied embedding table
(§01.1: `lm_head.weight` IS `embed_tokens.weight`) — recomputes the logits with the summation order
of the reference's own F32 arithmetic permuted and reassociated, and reports cos against the pinned
tensor. Same method as `prototype/mlp_gemv/floor_probe.py`, applied to the whole-token logits instead
of one MLP dot: a floor is measured from the reference, never chosen by taste.

What is and is not legal here. Every BF16 value is exact in F32, so the *products* are identical in
every variant; only the addition order moves. Reordering the 1024-term dot, summing it in split-K
groups, and summing the norm's mean of squares in another order are all things an engine may do
(§10.1 F1–F4 say exactly this about a chained dot). Nothing here perturbs a value, drops a term, or
changes dtype — those are bugs, and a gate exists to catch them, not to forgive them.

Two floors come out, and they are different quantities:
  * reassociation alone, with the pin's BF16 output rounding reproduced (`bf16(...)`) — a kernel that
    rounds its logits the way the reference does could not do better than this;
  * `no_output_round`, which keeps the logits in F32 — what `lm_head_full` legitimately does, so the
    distance to the pin here is entirely the *reference's own* output rounding (the pin is BF16 values
    stored as F32, asserted below). A value-perfect engine still pays it.
The floor for the gate's comparison is the worse of the two at each step.

What this floor does NOT cover, stated because the reading matters: it is the floor of the LAST op.
The engine's 0.999886 arrives after 24 layers of different kernels and a prompt fed one token per
step instead of the reference's chunked prefill (§04.7 vs §04.4; G3 self-checks chunk-vs-recurrent at
0.9999). If the engine sits *below* this floor, the discrepancy is real and upstream — it cannot be
blamed on summation order in the logits dot, because this file measures what summation order is worth.

Run:  .venv/bin/python engine/logits_floor.py [--steps 8] [--model DIR]

First run (2026-10-11), over the 8 `dec8` steps: the rebuild reproduces the pin at cos 0.999999999, every
reassociation variant also lands at 0.999999999, and the pin's own BF16 output rounding alone sits at
0.999998622 — so the floor of this comparison is 0.99999862 and it is set almost entirely by the
reference rounding its logits while the engine does not. The engine's measured 0.999886 is below it by a
factor of ~83 in distance from 1, which is what "report it as a real upstream difference" rests on.
Recorded with the pass in engine/results.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from speccheck.specimpl import rmsnorm_zc  # the oracle's own norm, not a paraphrase of it  # noqa: E402

REFS = ROOT / "data" / "refs" / "dec8"
EMBED = "model.language_model.embed_tokens.weight"
FINAL_NORM = "model.language_model.norm.weight"
CHUNK = 32768          # vocab rows per block: 32768 × 1024 F32 products = 128 MiB at a time

# The engine's measured per-step cos from the first end-to-end `engine/gate.py` pass (2026-10-11).
# Only the two extremes were quoted with the result, so the range is what gets compared.
ENGINE_OBSERVED = (0.999886, 0.999968)
G5_HARD, G6_ADVISORY = 0.9995, 0.9999
ORDER_KEYS = ("perm_k_1", "perm_k_2", "perm_k_3", "splitk8_rev", "splitk32_rev",
              "splitk32_fwd", "perm_norm_mean")


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    # F64 accumulation, because that is the arithmetic `gate.py::metrics()` uses (`astype(float64)`
    # before the dot). A floor measured in F32 would be comparing a different quantity: the F32 sum of
    # 248 320 products moves cos by ~4e-6 on its own, which is smaller than everything here but is not
    # something the floor should absorb silently.
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64) / (a64.norm() * b64.norm()))


def logits_reordered(emb: torch.Tensor, x: torch.Tensor, order: torch.Tensor | None = None,
                     groups: int = 1, reverse: bool = False) -> torch.Tensor:
    """sum_k emb[i,k]·x[k] for every row, with the addition order chosen by the caller.

    The products are exact: BF16 values in F32, multiplied in F32. `order` permutes K, so the running
    sum visits the same 1024 terms in another sequence; `groups > 1` instead splits K into G partials
    (a split-K kernel's shape) and accumulates them in the sequence `reverse` picks — the reassociation
    §10.1 F1–F4 refuses to call an error. The partials are added in an explicit loop so the order is
    the one named here and not torch's internal reduction order.
    """
    out = torch.empty(emb.shape[0], dtype=torch.float32)
    seq = list(range(groups - 1, -1, -1)) if (reverse and groups > 1) else list(range(groups))
    for lo in range(0, emb.shape[0], CHUNK):
        part = emb[lo:lo + CHUNK]
        prod = (part[:, order] * x[order]) if order is not None else part * x
        if groups <= 1:
            out[lo:lo + CHUNK] = prod.sum(1)
        else:
            acc = torch.zeros(prod.shape[0], dtype=torch.float32)
            for i in seq:
                acc = acc + prod[:, i::groups].sum(1)
            out[lo:lo + CHUNK] = acc
    return out


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def variants(k: int) -> list[tuple[str, dict]]:
    gen = torch.Generator().manual_seed(20261011)               # fixed: the probe is reproducible
    perms = [torch.randperm(k, generator=gen) for _ in range(3)]
    return [("perm_k_1", {"order": perms[0]}),
            ("perm_k_2", {"order": perms[1]}),
            ("perm_k_3", {"order": perms[2]}),
            ("splitk8_rev", {"groups": 8, "reverse": True}),
            ("splitk32_rev", {"groups": 32, "reverse": True}),
            ("splitk32_fwd", {"groups": 32})]


def probe_step(s: int, norm_w: torch.Tensor, emb_f32: torch.Tensor) -> dict[str, float]:
    # §10.5's trap is avoided by construction: every tensor here is named for its step, so no
    # last-writer-wins bare name enters the pairing. L23.s_out is the residual the reference hands the
    # final norm (specimpl `_layer` stores it after `x = residual + mlp(...)`), and the pin is that
    # same step's logits.
    residual = torch.from_numpy(np.load(REFS / f"decode{s}.L23.s_out.npy")).to(torch.bfloat16)
    pin = torch.from_numpy(np.load(REFS / f"decode{s}.logits_last.npy")).float().reshape(-1)
    assert torch.count_nonzero(pin.view(torch.int32) & 0xFFFF) == 0, \
        "the pin is no longer BF16-values-in-F32; the no_output_round floor is then not what it claims"
    residual = residual.reshape(-1)
    normed = rmsnorm_zc(residual, norm_w)                       # BF16 out, exactly §03.1
    x = normed.float()

    out: dict[str, float] = {}
    # The pairing proof first: rebuilding from these three inputs must reproduce the pin. If this is
    # not ≈1 the probe is comparing the wrong tensors and nothing below it means anything.
    out["pair_rebuild"] = cos(bf16(emb_f32 @ x), pin)
    for name, kw in variants(residual.numel()):
        out[name] = cos(bf16(logits_reordered(emb_f32, x, **kw)), pin)
    # The norm's mean of squares summed in another order (same 1024 terms), then the dot exactly as
    # the reference does it — so this row isolates the norm's reassociation.
    perm = torch.randperm(residual.numel(), generator=torch.Generator().manual_seed(7))
    normed_p = rmsnorm_zc(residual[perm], norm_w[perm])[torch.argsort(perm)]
    out["perm_norm_mean"] = cos(bf16(emb_f32 @ normed_p.float()), pin)
    # What the engine legitimately does that the reference does not: keep the logits in F32.
    out["no_output_round"] = cos(emb_f32 @ x, pin)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    shard = next(Path(args.model).glob("*.safetensors"))
    t0 = time.time()
    with safe_open(shard, framework="pt") as f:
        norm_w = f.get_tensor(FINAL_NORM)
        emb_f32 = f.get_tensor(EMBED).float()
    print(f"weights: {FINAL_NORM} {tuple(norm_w.shape)} {norm_w.dtype}, "
          f"{EMBED} {tuple(emb_f32.shape)} F32 (tied head, §01.1) — {time.time() - t0:.1f}s")

    rows = []
    for s in range(args.steps):
        r = probe_step(s, norm_w, emb_f32)
        r["floor"] = min(r["no_output_round"], min(r[k] for k in ORDER_KEYS))
        rows.append(r)
        order = min(r[k] for k in ORDER_KEYS)
        print(f"  step {s}: pair_rebuild {r['pair_rebuild']:.9f}"
              f" | reassociation floor {order:.9f}"
              f" | no-output-round floor {r['no_output_round']:.9f}"
              f" | FLOOR {r['floor']:.9f}")
    del emb_f32

    order_worst = min(min(r[k] for k in ORDER_KEYS) for r in rows)
    round_worst = min(r["no_output_round"] for r in rows)
    worst, best = min(r["floor"] for r in rows), max(r["floor"] for r in rows)
    print("\n== worst cos over the %d steps, per legal variant" % args.steps)
    for k in ORDER_KEYS + ("no_output_round",):
        print(f"  {k:16s} {min(r[k] for r in rows):.9f}")
    print(f"  {'pair_rebuild':16s} {min(r['pair_rebuild'] for r in rows):.9f}")
    print("\n== the legal floor of a step's logits vector")
    print(f"  rebuild reproduces the pin at cos {min(r['pair_rebuild'] for r in rows):.9f}"
          " (a pairing proof, not a result)")
    print(f"  reassociation alone:                 cos >= {order_worst:.9f}")
    print(f"  the pin's own BF16 output rounding:  cos >= {round_worst:.9f}")
    print(f"  FLOOR: {worst:.9f} .. {best:.9f}")

    lo, hi = ENGINE_OBSERVED
    verdict = "ABOVE" if lo >= worst else "BELOW"
    print(f"\n== the engine's observed {lo} .. {hi} against it")
    print(f"  engine worst {lo} is {verdict} the legal floor {worst:.9f}.")
    if lo < worst:
        print("  So the step-1 dip is NOT summation order in the logits dot: a legal reordering of the")
        print("  reference's own arithmetic never leaves that tensor, and the engine left it. The")
        print("  residual is upstream — 24 layers of different kernels, plus a prompt fed")
        print("  token-at-a-time instead of the reference's chunked prefill — i.e. a small real")
        print("  difference, not a floor artefact. Report it as one.")
    else:
        print("  A legal reordering of the reference's own arithmetic gets no better than the floor,")
        print("  so the dip is inside what the last op permits: expected cascade noise.")
    print(f"  G5's hard floor {G5_HARD} and G6's advisory {G6_ADVISORY} are unchanged either way")
    print("  (gate.py's BAND comment carries why G6 stays advisory).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
