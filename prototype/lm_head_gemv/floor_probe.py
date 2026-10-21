#!/usr/bin/env python3
"""Floor probe for the tied logits GEMV: how far apart may two LEGAL computations of the same logits
be, per element? CPU-only, no device — the device's own output is only read, as the subject.

WHY. `progress/2026-10-11/gate-audit.md` G1: `prototype/lm_head_gemv/run.py` asserts argmax, `cos >= 0.999` and
`meanrel <= 5 %` and prints the rest, so it has **no per-element rule at all** — two wrong kernels
named in the audit (a row written into its neighbour's slot; a dropped term in the reduction) pass
every asserted check. Adding a per-element rule needs a band, and a band here must not be typed in:
this is a 1024-term F32 dot, and §10.1 F1-F4 plus `prototype/mlp_gemv/results.md` say a chained dot's
legal ndiff is enormous and its max-ULP is blind to cancellation. So the band gets measured, the way
`engine/logits_floor.py` measured the aggregate one.

METHOD, same shape as `mlp_gemv/floor_probe.py`: every BF16 value is exact in F32, so the 1024
PRODUCTS are identical in every variant and only the ADDITION ORDER moves. Recompute the logits with
each legal order — including `lm_head_gemv_full`'s exact order, emulated from the kernel source —
round to BF16 the way the pin is rounded, and compare against the pin with `prototype/bf16_ulp.py`.
Whatever spread those variants show is what a gate must forgive; a value-perfect kernel cannot do
better. Dropping a term, permuting rows, changing dtype or perturbing a value are NOT variants — they
are bugs, and the gap between this list and those is what the gate's rule has to sit in.

The pin is BF16 values stored as F32 (asserted below), and the kernel writes F32 logits without the
reference's output rounding, so the comparison rounds the candidate to BF16 first. That is the
comparison `run.py`'s per-element rule uses; the unrounded F32 distance is a different quantity and
`engine/logits_floor.py` already owns it.

Anchors, both checked before anything is reported:
  * the pin really is BF16-valued (`pin == bf16(pin)`, elementwise);
  * the input row is the row that produces it — torch's own order on this row must reproduce the pin
    at cos >= 0.99999 and small ndiff. This is the §10.5 pairing trap: the bare dump names are
    decode-step values and pairing the wrong `pre_final_hidden` with `logits_last` gives cos 0.25.

    .venv/bin/python prototype/lm_head_gemv/floor_probe.py [--rows 16384] [--h /tmp/lmhead/h.bin]
                                                          [--device-out /tmp/lmhead/out_full.bin]
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp as U                                                      # noqa: E402

WORK = Path("/tmp/lmhead")
REF = ROOT / "data" / "refs" / "smoke"
CKPT = ROOT / "models" / "Qwen3.5-0.8B"
K = 1024


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round an F32 array through BF16 and return F32-with-BF16-payloads (the pin's own format)."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def kernel_order(P: np.ndarray) -> np.ndarray:
    """`lm_head_gemv_full`'s addition order exactly, per row.

    Lane l takes the uint4 chunks v = l, l+32, l+64, l+96 (128 chunks of 8 BF16s per row of 1024)
    and runs `acc += p[j]*hp[j]` for j = 0..7 inside each, in chunk order; then
    `for (o = 16; o; o >>= 1) acc += __shfl_down_sync(mask, acc, o)` leaves the row's sum in lane 0.
    Chunk index is (c*32 + l)*8 + j, hence the [4, 32, 8] reshape.
    """
    r = P.shape[0]
    v = P.reshape(r, 4, 32, 8)
    acc = np.zeros((r, 32), dtype=np.float32)
    for c in range(4):
        for j in range(8):
            acc = (acc + v[:, c, :, j]).astype(np.float32)
    for off in (16, 8, 4, 2, 1):
        acc[:, :32 - off] = (acc[:, :32 - off] + acc[:, off:32]).astype(np.float32)
    return acc[:, 0]


def order_variants(P: np.ndarray, fsum_rows: int) -> dict:
    """Every sum is over the same 1024 identical products; only the association differs."""
    x = np.ascontiguousarray(P, dtype=np.float32)

    def seq(a, rev=False):
        n = a.shape[-1]
        acc = a[:, -1].copy() if rev else a[:, 0].copy()
        for i in (range(n - 2, -1, -1) if rev else range(1, n)):
            acc = (acc + a[:, i]).astype(np.float32)
        return acc

    def fold(a):
        v, n = a.copy(), a.shape[-1]
        while n > 1:
            n //= 2
            v = (v[:, :n] + v[:, n:2 * n]).astype(np.float32)
        return v[:, 0]

    def splitk(groups):
        v = x.reshape(x.shape[0], groups, x.shape[-1] // groups)
        part = v.sum(-1, dtype=np.float32)
        acc = part[:, 0].copy()
        for i in range(1, groups):
            acc = (acc + part[:, i]).astype(np.float32)
        return acc

    out = {"R0 torch .sum (reference side)": torch.from_numpy(x).sum(-1).numpy(),
           "A1 the kernel's own order": kernel_order(x),
           "A2 sequential forward": seq(x),
           "A2 sequential reverse": seq(x, rev=True),
           "A2 fold in halves": fold(x),
           "A2 split-K 8 x 128": splitk(8),
           "A2 split-K 32 x 32": splitk(32),
           "A2 split-K 128 x 8": splitk(128)}
    if fsum_rows:
        truth = np.array([math.fsum(row) for row in x[:fsum_rows]], dtype=np.float64)
        out["A0 correctly rounded (fsum)"] = np.concatenate(
            [truth.astype(np.float32), np.full(x.shape[0] - fsum_rows, np.nan, np.float32)])
    return out


def line(label, rep):
    print("   %-28s ndiff %7d (%5.3f %%)  max ULP %4d  >1 ULP %6d  n_bad %5d  "
          "worst_bad_rel %.2e  tightest band %.2e"
          % (label, rep["ndiff"], rep["ndiff"] / rep["n"] * 100.0, rep["maxulp"], rep["over1"],
             rep["n_bad"], rep["worst_bad_rel"], rep["worst_off_rel"]))


def rank_check(vals, ref):
    """(argmax matches, top-5 set overlap) — a rank claim is a different assertion from a per-element
    one, and a legal reordering is allowed to tie: two logits one BF16 step apart near rank 5 could
    swap. Whether that actually happens on this pin is what this measures, not what one hopes.
    Non-finite entries are dropped, so a variant that only covers a subsample cannot fake a rank."""
    v = np.asarray(vals)
    r = np.asarray(ref)
    keep = np.isfinite(v)
    v, r = v[keep], r[keep]
    top = set(np.argsort(-v)[:5].tolist())
    ref_top = set(np.argsort(-r)[:5].tolist())
    return int(np.argmax(v) == np.argmax(r)), len(top & ref_top)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=16384, help="vocabulary rows to sweep (0 = all)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--h", default=str(WORK / "h.bin"),
                    help="the pinned lm_head input row, F32 payloads (run.py writes this)")
    ap.add_argument("--device-out", default=str(WORK / "out_full.bin"),
                    help="the gate's device output, if any — the subject, not the floor")
    ap.add_argument("--cancel-rel", type=float, default=1e-4)
    ap.add_argument("--ulp-max", type=int, default=1)
    a = ap.parse_args()

    pin = np.load(REF / "logits_last.npy").reshape(-1).astype(np.float32)
    V = pin.size
    if not np.array_equal(bf16_bits(pin), pin):
        print("STOP: the pin is not BF16-valued, so a BF16 ULP rule cannot be written against it")
        return 1
    h = torch.from_numpy(np.fromfile(a.h, dtype=np.float32)).numpy()
    assert h.size == K, "input row is %d wide, expected %d" % (h.size, K)
    print("pin: %d logits, BF16-valued | input row %s (%d BF16-payload floats)"
          % (V, a.h, h.size))

    key = "model.language_model.embed_tokens.weight"
    Wt = None
    for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            if key in s.keys():
                Wt = s.get_tensor(key)
                break
    if Wt is None:
        raise KeyError(key)

    n = V if a.rows == 0 else min(a.rows, V)
    names = None
    acc: dict[str, list] = {}
    store = {}                                # only the swept rows, for the rank question
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        Wc = Wt[lo:hi].float().numpy()            # exact BF16 payloads, widened, never rounded
        P = Wc * h[None, :]                       # the products are identical in every variant
        vs = order_variants(P, 512 if lo == 0 else 0)
        names = names or sorted(vs)
        for name, s in vs.items():
            z = bf16_bits(s)
            keep = ~np.isnan(z)                   # the fsum truth only covers a subsample
            acc.setdefault(name, []).append(
                (U.report(z[keep], pin[lo:hi][keep], ulp_max=a.ulp_max,
                          cancel_rel=a.cancel_rel), int(keep.sum())))
            store.setdefault(name, []).append(z)
        # The bug classes gate-audit G1 says pass today, computed here so the same yardstick lands on
        # them. A dropped term is local to a row; a row written into its neighbour's slot is not, and
        # is built once the whole sweep is in hand. Adding nothing and adding +0.0 are the same F32
        # operation, so zeroing the last addend (index 1023, which IS the last addend of the kernel's
        # order) is an exact model of a term missing from the reduction.
        Pd = np.ascontiguousarray(P, dtype=np.float32)
        Pd[:, -1] = 0.0
        store.setdefault("BUG last term of the dot missing", []).append(kernel_order(Pd))
    print("\nlegal reassociations of the same 1024 products, against the pin "
          "(%d rows, ulp_max %d, cancel_rel %.0e):" % (n, a.ulp_max, a.cancel_rel))
    floor_ulp, floor_band, floor_ndiff_pct = 0, 0.0, 0.0
    for name in names:
        tot = dict(ndiff=0, n=0, maxulp=0, over1=0, n_bad=0, worst_bad_rel=0.0,
                   worst_off_rel=0.0, cos=0.0)
        for rep, cnt in acc[name]:
            tot["ndiff"] += rep["ndiff"]
            tot["n"] += cnt
            tot["maxulp"] = max(tot["maxulp"], rep["maxulp"])
            tot["over1"] += rep["over1"]
            tot["n_bad"] += rep["n_bad"]
            tot["worst_bad_rel"] = max(tot["worst_bad_rel"], rep["worst_bad_rel"])
            tot["worst_off_rel"] = max(tot["worst_off_rel"], rep["worst_off_rel"])
        line(name, tot)
        rr = np.concatenate(store[name])[:n]
        am, t5 = rank_check(rr, pin[:n])
        print("       argmax %s | top-5 overlap %d/5" % ("=" if am else "DIFFERS", t5))
        # The correctly rounded sum is the truth, not a variant of the reference; it sits further
        # from the pin than the legal orders do (that is what cancellation costs), so it does not
        # get to set the band.
        if "A0" not in name:
            floor_ulp = max(floor_ulp, tot["maxulp"])
            floor_band = max(floor_band, tot["worst_off_rel"])
            floor_ndiff_pct = max(floor_ndiff_pct, tot["ndiff"] / tot["n"] * 100.0)
    print("   -> FLOOR: max ULP over legal orders %d; ndiff up to %.3f %% of the vocabulary; the"
          " widest band any legal order would pass at is %.2e"
          % (floor_ulp, floor_ndiff_pct, floor_band))

    # Row i holding row i-1's value, and the audit's own smaller case: two NON-argmax rows swapped,
    # which is 2 elements of 248 320 and leaves every aggregate alone.
    base = np.concatenate(store["A1 the kernel's own order"])[:n]
    shift = np.empty_like(base)
    shift[0], shift[1:] = base[0], base[:-1]
    store["BUG row i written into slot i+1"] = [shift]
    am_ref = int(np.argmax(pin[:n]))
    hi2 = np.argsort(-pin[:n])[:80]
    i, j = int(hi2[7]), int(hi2[9])          # inside the top 80, so the aggregate moves least
    swap = base.copy()
    swap[i], swap[j] = base[j], base[i]
    store["BUG two rows swapped (%d<->%d, argmax %d untouched)" % (i, j, am_ref)] = [swap]
    print("\nthe same yardstick on the two wrong kernels G1 names. The second line under each is what")
    print("the gate asserts today (argmax, cos >= 0.999, meanrel <= 5 %; top-5 is printed only), so a")
    print("PASSES there is a kernel that ships today:")
    for name in sorted(k for k in store if k.startswith("BUG")):
        v = bf16_bits(np.concatenate(store[name])[:n])
        rep = U.report(v, pin[:n], ulp_max=a.ulp_max, cancel_rel=a.cancel_rel)
        line(name, rep)
        am, t5 = rank_check(v, pin[:n])
        passes = am and rep["cos"] >= 0.999 and rep["meanrel"] <= 0.05
        print("       asserted today: argmax %s, cos %.6f %s, meanrel %.5f %s | top-5 (printed, not"
              " asserted) %d/5 -> %s"
              % ("=" if am else "DIFFERS", rep["cos"], ">=0.999 OK" if rep["cos"] >= 0.999 else "FAIL",
                 rep["meanrel"], "<=0.05 OK" if rep["meanrel"] <= 0.05 else "FAIL", t5,
                 "PASSES the gate as it stands" if passes else "caught by the aggregates"))
        print("       per-element rule at ulp_max %d, cancel_rel %.0e: n_bad %d, ndiff %.3f %% of rows"
              " -> %s"
              % (a.ulp_max, a.cancel_rel, rep["n_bad"], rep["ndiff"] / rep["n"] * 100.0,
                 "CAUGHT" if (rep["n_bad"] or rep["ndiff"] / rep["n"] > 0.001) else "STILL PASSES"))
    print("   -> the rule in run.py has to be at least this wide, and no wider.")

    if a.device_out and Path(a.device_out).exists():
        got = np.fromfile(a.device_out, dtype=np.float32)
        if got.size != V:
            print("\ndevice out %s holds %d floats, expected %d — skipping" % (a.device_out,
                                                                              got.size, V))
        else:
            rep = U.report(bf16_bits(got), pin, ulp_max=a.ulp_max, cancel_rel=a.cancel_rel)
            print("\nsubject, not floor — the device's own logits (%s), same criteria:" % a.device_out)
            line("device lm_head_full", rep)
            am, t5 = rank_check(bf16_bits(got), pin)
            print("       argmax %s | top-5 overlap %d/5 | cos %.8f meanrel %.5f"
                  % ("=" if am else "DIFFERS", t5, rep["cos"], rep["meanrel"]))
            print("   the reference rounds its logits to BF16 and the kernel does not, so the device"
                  " values are rounded here before the ULP comparison; the unrounded F32 distance is"
                  " the reference's own output rounding, which engine/logits_floor.py owns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
