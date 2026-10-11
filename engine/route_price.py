#!/usr/bin/env python3
"""What is a GDN prefill route worth on the LOGITS vector — measured inside the reference, with no engine?

Why this exists. `engine/logits_floor.py` prices one op (final norm + tied `lm_head`) with the reference's
own `L23.s_out` held fixed, so its 0.99999862 is the floor for *reassociation-only change on the
reference's own arithmetic at that one op*. It is not a floor for an end-to-end token-level logits vector,
because it removes by construction the 23 other layers, the whole token chain, and the legitimate choice of
how a prompt gets into the GDN state. This script prices that missing term: it runs the dec8 prompt plus its
greedy continuation twice through `speccheck/specimpl.py` — once with the chunked GDN prefill (§04.7, the
form G3 canonicalises) and once with the recurrent form (§04.4, the form an engine with no prefill
necessarily takes, since it feeds the prompt one token per decode step) — and compares the two runs'
logits vector at every step using `engine/gate.py::metrics`' F64 cos. Same weights, same tokens, no engine,
no device. If the spread between two legal reference forms is the same order as the engine's deviation from
the pin, the engine's deviation needs no bug to explain it.

Run:  .venv/bin/python engine/route_price.py            # ~90 s, torch CPU, ~2 GB RSS

Measured 2026-10-11, dec8, the same 8 continuation tokens fed to both forms, argmax agreement all True:

| step (gate numbering) | 1-cos, route vs route | × the last-op floor | engine's own 1-cos vs the pin |
|---|---|---|---|
| 1 | 1.900e-04 | 138× | 1.14e-4 |
| 2 | 5.262e-05 | 38× | 4.1e-5 |
| 3 | 6.989e-05 | 51× | 3.7e-5 |
| 4 | 5.314e-05 | 39× | 3.5e-5 |
| 5 | 4.730e-05 | 34× | 3.5e-5 |
| 6 | 5.628e-05 | 41× | 3.4e-5 |
| 7 | 1.025e-04 | 74× | 3.2e-5 |
| 8 | 1.080e-04 | 78× | 4.8e-5 |

The route spread is larger than the engine's entire deviation **at every step**, and it is worst at step 1
— the same shape the engine shows. So the engine sits inside the band the reference leaves open between two
of its own legal forms, and the fair end-to-end floor is this table, not 0.99999862.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speccheck"))
from specimpl import Weights, Forward  # noqa: E402

torch.set_grad_enabled(False)
FIX = ROOT / "data" / "refs" / "dec8"
ids = json.loads((FIX / "manifest.json").read_text())["prefill_ids"]
w = Weights(ROOT / "models" / "Qwen3.5-0.8B")
print(f"dec8 prefill_ids: {len(ids)} tokens = {ids}")

FLOOR = 1.38e-6  # engine/logits_floor.py: last-op floor, distance from 1


def f64(t):
    return t.double() if t.dtype == torch.bfloat16 else t.float().double()


def cos(a, b):
    a, b = f64(a).ravel(), f64(b).ravel()
    return float((a @ b / (a.norm() * b.norm() + 1e-30)).item())


def meanrel(a, b):
    a, b = f64(a).ravel(), f64(b).ravel()
    return float(((a - b).abs() / b.abs().clamp_min(1e-6)).mean().item())


def run(form, cont):
    """One prefill pass over the whole prompt, then the continuation token-at-a-time. Returns the logits
    vector that predicts each next token: position len(ids)-1, then one per continuation token."""
    f = Forward(w, gdn_form=form)
    lg = [f.logits(f.prefill(torch.tensor(ids)))[-1]]
    for tok in cont:
        lg.append(f.logits(f.decode(tok))[-1])
    return lg


cont = Forward(w, gdn_form="chunk").greedy(ids, 8)
print(f"chunk-form continuation: {cont}")
chunk = run("chunk", cont)
recur = run("recurrent", cont)

print("\nroute-vs-route on the logits vector (chunked prefill vs recurrent prefill, same tokens):")
print("step  cos        meanrel      1-cos        x the last-op floor")
for j in range(8):
    c, mr = cos(recur[j], chunk[j]), meanrel(recur[j], chunk[j])
    print(f"  {j + 1:>2}   {c:.6f}   {mr:.3e}   {1 - c:.3e}   {(1 - c) / FLOOR:>7.1f}x")
print("\nengine observed vs the pinned dump, per step: "
      "0.999886 0.999959 0.999963 0.999965 0.999965 0.999966 0.999968 0.999952")
print("engine 1-cos per step: 1.14e-4 4.1e-5 3.7e-5 3.5e-5 3.5e-5 3.4e-5 3.2e-5 4.8e-5")
print("last-op floor (engine/logits_floor.py): 0.99999862")
print("argmax agreement between the two routes, per step:",
      [int(torch.argmax(a)) == int(torch.argmax(b)) for a, b in zip(chunk, recur)])
