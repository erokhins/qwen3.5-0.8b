"""The state-fingerprint definition — the one place §08's state summary is computed.

Extracted out of `tools/reference_dump.py` (P1 of `progress/2026-10-12/plan-prefill-2026-10-12.md` §2.6) because
two more readers appeared at once: `engine/gate.py` section (g) has to recompute a pinned
fingerprint to prove it still matches the dumper, and `engine/decode_run.cpp --state-fp` has to
emit engine-side rows in the same columns. A formula that lives in three files is a formula that
drifts, so it lives here and the others import it.

`reference_dump.py` imports `fp_row` from here and pins with it unchanged: the arithmetic is
byte-identical to what made every existing fixture, so re-running the dumper still reproduces the
pinned `state_fp` / `conv_fp` / `kv_fp` arrays (§10.5). The F32 accumulation is part of the
definition, not an accident of it — a re-accumulation in F64 differs by ~1e-7 relative and would
therefore no longer be bit-identical to the pins, which is exactly the property (g)'s identity case
asserts. A reader that cannot use torch (engine/gate.py keeps itself torch-free everywhere else)
mirrors the formula instead and says so where it does: `decode_run.cpp`'s `--state-fp` accumulates
in F64 and carries that ~1e-7 floor on the n2/mean columns, which (g) prints as a known floor
rather than folding it into a band.
"""
from __future__ import annotations

import numpy as np
import torch


def fp_row(parts: list[torch.Tensor], prev: list[torch.Tensor] | None) -> list[float]:
    """One row of a state fingerprint: [sum x², max|x|, mean x, dn2], F64.

    `parts` is the tensors that make up one logical state (the GDN row has one; the attention
    row is K and V together, so it is two) and they are summarised as one pooled row. Parts are
    compared PART-WISE against the previous step's parts, never as one flattened blob: K and V
    grow by one lane per step, so a flat `x[:prev.numel()] - prev` would silently compare K's
    tail against the start of V and report a large delta for a cache that is behaving perfectly.
    (It did, at 1e4 on a clean run, which is why the definition is part-wise.)

    Moment sums, deliberately: they are invariant to element ORDER, and the engine lays its GDN
    window and KV out differently from the reference in places (`state.h`: head-major window,
    `[head][slot][dim]` KV), so an order-sensitive summary would read a layout difference as a
    bug. What they catch is a state that decayed, was zeroed, was scaled, or was left behind —
    which is what a depth scan looks for. What they CANNOT catch is a corruption that preserves
    the multiset of values (a rotation within the tensor); per-element state gating needs the
    full tensors, at 18 MiB per step, which is exactly what a deep fixture cannot afford.

    dn2 = Σ_i Σ_{common extent of dim −2} (x_i − prev_i)², i.e. how much the ALREADY-WRITTEN
    entries moved. Append-only storage therefore has an expected dn2 of exactly 0.0, and a
    non-zero kv_fp dn2 means history was rewritten. No `prev` (§08.4: a fresh request's state is
    zeroed) means the delta is the tensor itself, so the first row's dn2 equals its n2.

    "The common extent of dim −2" is the append axis for a growing cache ([1,H,T,D]) and the
    whole extent for a state that never grows (S [1,16,128,128], conv [1,6144,4]), which is why
    one rule serves all three families. Flattening first and slicing a common PREFIX is wrong for
    a growing tensor: when T changes, the per-head boundaries move, so a flat prefix compares
    head 1 against the tail of head 0 — it read 3.0e3 on a cache that is bit-identical.
    """
    flat = torch.cat([p.reshape(-1) for p in parts]).to(torch.float32)
    n2 = float(torch.sum(flat * flat))
    row = [n2, float(torch.max(torch.abs(flat))), float(torch.mean(flat))]
    if prev is None:
        row.append(n2)
        return row
    dn2 = 0.0
    for x, p in zip(parts, prev):
        x = x.detach().to(torch.float32)
        n = min(x.shape[-2], p.shape[-2])
        d = x[..., :n, :] - p[..., :n, :]
        dn2 += float(torch.sum(d * d))
    row.append(dn2)
    return row


def fp_row_np(parts: list[np.ndarray], prev: list[np.ndarray] | None) -> list[float]:
    """The numpy form of `fp_row`, for readers that do not import torch (`engine/gate.py`).

    Same pooling, same part-wise Δ² over the common extent of dim −2, same §08.4 no-previous
    convention — same definition, second implementation, which is a thing one does deliberately or
    not at all. The reason here is that this gate's whole value is independence from the engine and
    it does not import torch (module header), while it still has to derive a fingerprint from a
    pinned tensor: the conv window, which is the newest three columns of a four-column pinned buffer
    (A-04), has no pinned fingerprint at its own extent, so the number it is compared against has to
    be computed from the pinned tensor by whoever does the comparing.

    Accumulation is F64 where `fp_row`'s is F32, so a row computed here sits ~1e-7 relative off a
    pinned row even when the tensors are identical. That is the same floor
    `engine/decode_run.cpp --state-fp` carries, and the reason `(g)` prints an engine-vs-pin
    fingerprint distance with a floor statement rather than calling it closeness.
    """
    flat = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])
    n2 = float(np.sum(flat * flat))
    row = [n2, float(np.max(np.abs(flat))), float(np.mean(flat))]
    if prev is None:
        row.append(n2)
        return row
    dn2 = 0.0
    for x, p in zip(parts, prev):
        x = np.asarray(x, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        n = min(x.shape[-2], p.shape[-2])
        d = x[..., :n, :] - p[..., :n, :]
        dn2 += float(np.sum(d * d))
    row.append(dn2)
    return row
