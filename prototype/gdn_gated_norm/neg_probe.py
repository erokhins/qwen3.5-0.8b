#!/usr/bin/env python3
"""CPU emulation of `gdn_gated_norm`'s injected bugs, so the six negatives are checkable without a device.

The device gate locates each injection by comparing the injected run against the clean run
(`run.py --gate`: "moved gated_out N/294912, norm_pre_gate M/294912"). That is a device command and
device runs belong to the operator (`AGENTS.md`), so this file answers the same question on the CPU:
it replays the kernel's arithmetic — including the two-wavefront shuffle ladder a 128-thread block
actually runs on this target — once per policy, and reports for each injection how many elements of
`gated_out` and of `norm_pre_gate` differ from the clean policy's. An injection that moves nothing
is vacuous, and this prints that verdict instead of leaving a device pass to discover it (the
`mlp_gemv` rule: a bug must show up **where the bug lives**).

It also separates the three weight-dtype worlds a caller can end up in, because the affine's dtype
is the axis this op's policy turns on:

    F32 store / kernel reads F32    — the code as written: `float wv = w[d]`, rounded where it is used
    BF16 store / kernel reads BF16  — numerically identical to the row above: rounding an F32 weight
                                      to BF16 and reading it back is the same value
    F32 store / kernel reads BF16   — the confusion: lane d gets the d-th BF16 **half** of the F32
                                      words, i.e. the low half (~0) on even d

Rows are the kernel's `GdnNeg` names, so this file and `gdn_gated_norm_kernels.h` read side by side;
the expected localisation is copied from `run.py`'s NEGATIVES table so a drift between the two is a
failure. Nothing here is tuned to pass.

Run: .venv/bin/python prototype/gdn_gated_norm/neg_probe.py [fixture]      (CPU only, no GPU)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "proj_gemv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mirror_probe as mp  # noqa: E402  (cases(): the same pins the device gate feeds)
import bf16_ulp            # noqa: E402

BF16 = torch.bfloat16
D_V, HEADS, EPS = 128, 16, 1e-6
WARP = 64                   # wavefront size on gfx1151; one 128-thread block = two of them
STEPS = ["decode%d" % s for s in range(8)]
NEGATIONS = ["w_f32", "affine_uncast", "single_cast", "silu_rounded", "bf16acc", "noeps"]
# (must move, must not move), copied from run.py's NEGATIVES.
EXPECT = {"none": ("-", "-"),
          "w_f32": ("gated_out", "norm_pre_gate"),
          "affine_uncast": ("gated_out", "norm_pre_gate"),
          "single_cast": ("gated_out", "norm_pre_gate"),
          "silu_rounded": ("gated_out", "norm_pre_gate"),
          "bf16acc": ("both", "-"),
          "noeps": ("both", "-")}


def bf(a):
    """Round to the activation dtype and back — the kernel's `gn_bf`."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(BF16).float().numpy()


def warp_sum(v, bf_tree):
    """Lane 0's value after one `__shfl_down_sync` ladder over the last axis (…, 64)."""
    v = np.array(v, dtype=np.float32)
    for off in (WARP // 2, 16, 8, 4, 2, 1):
        head, tail = v[..., :WARP - off], v[..., off:]     # shfl_down: lane i reads lane i + off
        v[..., :WARP - off] = bf(head + tail) if bf_tree else (head + tail).astype(np.float32)
    return v[..., 0]


def block_sum(acc2d, bf_tree):
    """`gn_block_sum` for a 128-thread block: two 64-lane ladders, then one over the two partials."""
    red = warp_sum(acc2d.reshape(HEADS, 128 // WARP, WARP), bf_tree)      # (heads, warps)
    x = np.zeros((HEADS, WARP), dtype=np.float32)
    x[:, :red.shape[1]] = red
    return warp_sum(x, bf_tree)                                          # (heads,)


def w_f32_bytes_as_bf16(w):
    """What a BF16-weight kernel would read if the store handed it the F32 words (little-endian)."""
    words = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32).reshape(-1)
    halves = words.view(np.uint16)[:D_V]                                  # low, high, low, high…
    return torch.from_numpy(halves.copy()).view(BF16).float().numpy()


def f32a(a):
    """numpy float32 from either a torch tensor or an array (pins and checkpoint weights are torch)."""
    return np.ascontiguousarray(a.detach().numpy() if isinstance(a, torch.Tensor) else a,
                                dtype=np.float32)


def policy(o, z, w, neg="none", w_world="f32"):
    """(gated_out, norm_pre_gate) for one [heads][d_v] case, replaying the kernel line by line."""
    o, z = f32a(o).reshape(-1, D_V), f32a(z).reshape(-1, D_V)
    w = f32a(w)
    if w_world == "f32":
        wv, round_w = w.reshape(D_V), True      # the F32 kernel rounds the weight at the point of use
    elif w_world == "bf16_store":
        wv, round_w = bf(w.reshape(D_V)), True  # already an activation value; the round is a no-op
    elif w_world == "f32_as_bf16":
        wv, round_w = w_f32_bytes_as_bf16(w), False   # a BF16 kernel does not re-round its input
    else:
        raise ValueError(w_world)

    x = bf(o)                                                        # activation-dtype cast of o
    acc = bf(x * x) if neg == "bf16acc" else (x * x).astype(np.float32)
    s = block_sum(acc, neg == "bf16acc")
    eps = 0.0 if neg == "noeps" else EPS
    # The device calls rsqrtf (a ~2-ULP F32 approximation); a correctly-rounded reciprocal square
    # root is the closest honest CPU stand-in, and this feeds a <= 1 BF16 ULP band.
    r = (1.0 / np.sqrt(s * (1.0 / D_V) + eps)).astype(np.float32)

    n_f32 = (x * r[:, None]).astype(np.float32)
    n = n_f32 if neg == "single_cast" else bf(n_f32)
    norm = bf(n_f32)                                                 # the norm_pre_gate stage

    if neg == "w_f32":
        # The A-01 literal reading: the pointer's raw value, with no round to the activation dtype.
        # In an F32 world that is the F32 weight; in a BF16 world it is already an activation value,
        # which is exactly why this injection cannot survive a BF16 pointer.
        a = bf(n * wv)
    elif neg == "single_cast":
        a = (n * wv).astype(np.float32)
    else:
        p = (n * (bf(wv) if round_w else wv)).astype(np.float32)
        a = p if neg == "affine_uncast" else bf(p)

    zf = bf(z)
    g = (zf * (1.0 / (1.0 + np.exp(-zf)))).astype(np.float32)        # HF's z·sigma(z), F32
    if neg == "silu_rounded":
        g = bf(g)
    return bf(a * g), norm


def split(a):
    """The zero / NaN / other classification a device output file gets in the gate's report line."""
    zero, nan = bf16_ulp.zero_nan(a)
    n = int(bf16_ulp.bf16_bits(a).size)
    return "n=%5d zeros=%5d nan=%4d other=%5d" % (n, zero, nan, n - zero - nan)


def main(argv):
    fixture = argv[0] if argv else "dec8"
    steps = [""] if fixture == "p300" else STEPS
    cases = list(mp.cases(fixture, steps))
    rows = [("none", "f32")] + [(n, "f32") for n in NEGATIONS] + \
           [("none", "bf16_store"), ("none", "f32_as_bf16"),
            # What happens to the A-01 injection if the pointer is BF16 instead of F32: `gn_bf` of a
            # value that is already an activation value is the identity, so the bug is not injected.
            ("w_f32", "bf16_store")]
    tot = {r: dict(mg=0, mn=0, nd=0, n=0, ulp=0) for r in rows}
    msq = []          # every head's mean(o²): how far the norm is from being eps-dominated
    for o, z, w, ref in cases:
        msq.append((o.reshape(-1, D_V).double() ** 2).mean(-1).numpy())
        clean_g, clean_n = policy(o, z, w)
        for row in rows:
            g, n = policy(o, z, w, row[0], row[1])
            gg, nn = torch.from_numpy(g).to(BF16), torch.from_numpy(n).to(BF16)
            t = tot[row]
            t["mg"] += int((gg.reshape(-1) != torch.from_numpy(clean_g).to(BF16).reshape(-1)).sum())
            t["mn"] += int((nn.reshape(-1) != torch.from_numpy(clean_n).to(BF16).reshape(-1)).sum())
            t["nd"] += int((gg.reshape(-1) != ref.reshape(-1)).sum())
            t["n"] += ref.numel()
            t["ulp"] = max(t["ulp"], int(bf16_ulp.ulp_dist(g.reshape(-1), ref.reshape(-1)).max()))

    print("%s — %d cases, %d elements compared per policy" %
          (fixture, len(cases), tot[("none", "f32")]["n"]))
    rc = 0
    for (neg, world), t in tot.items():
        if world == "f32":
            must = EXPECT[neg][0]
            if neg == "none":
                verdict = ("clean reference: %d/%d off the pin, worst %d ULP%s"
                           % (t["nd"], t["n"], t["ulp"],
                              "" if t["ulp"] <= 1 else " — OUTSIDE THE 1-ULP BAND"))
            else:
                moved_g, moved_n = t["mg"] > 0, t["mn"] > 0
                want_n = (must == "both")
                ok = moved_g and (moved_n == want_n)
                verdict = ("moves both stages, as declared" if ok and want_n else
                           "moves gated_out only, as declared" if ok else
                           "VACUOUS: moves nothing" if not moved_g else
                           "moves both stages, undeclared" if moved_n and not want_n else
                           "does not move gated_out")
            label = neg
        else:
            # The two non-F32 weight worlds are evidence, not verdicts: they report what a different
            # store/kernel dtype pair would do, and only the six declared rows above can fail.
            label = "%s [%s world]" % (neg, world)
            verdict = ("identical to the clean policy: this injection cannot exist in that world"
                       if t["mg"] == 0 and t["mn"] == 0 else
                       "DIFFERS from the clean policy: that world is broken")
        if any(w in verdict for w in ("VACUOUS", "does not", "OUTSIDE")):
            rc = 1
        print("  %-26s gated_out %7d/%d  norm_pre_gate %7d/%d  ndiff vs pin %7d  worst ULP %6d  %s"
              % (label, t["mg"], t["n"], t["mn"], t["n"], t["nd"], t["ulp"], verdict))

    ms = np.concatenate([m.ravel() for m in msq])
    print("\nmean(o²) per value head: min %.3e, median %.3e, against §00.2 eps = %g — eps is %.0f %% of"
          " the median head's variance, so it is part of this normaliser rather than a rounding"
          " detail, which is what sizes the `noeps` row." % (ms.min(), float(np.median(ms)), EPS,
                                                             100.0 * EPS / float(np.median(ms))))

    o, z, w, ref = cases[0]
    print("\ncase c000 — what each weight world writes (%d elements), against a pin that holds" %
          ref.numel())
    for label, world in (("F32 store, F32 read (as written)", "f32"),
                         ("BF16 store, BF16 read (hypothetical)", "bf16_store"),
                         ("F32 store, BF16 read (the confusion)", "f32_as_bf16")):
        g, _ = policy(o, z, w, "none", world)
        print("  %-40s %s  maxULP vs pin %6d"
              % (label, split(g.reshape(-1)),
                 int(bf16_ulp.ulp_dist(torch.from_numpy(g).to(BF16).reshape(-1), ref).max())))
    print("  %-40s %s" % ("the pin", split(ref)))
    print("  %-40s %s" % ("an untouched buffer (all +0)", split(np.zeros(ref.numel()))))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
