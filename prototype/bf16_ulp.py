"""BF16 ULP distances, done right — shared by every prototype gate (`prototype/*/`).

Why this exists as its own file: the obvious monotone ordering of a BF16 bit pattern is wrong,
and the wrong version looks fine until a value is negative.

BF16 is IEEE binary16-truncated-in-spirit: sign, 8 exponent bits, 7 mantissa bits. The uint16
bit pattern is monotone in the value only within one sign. The usual transform

    key = bits | 0x8000            # WRONG: applied unconditionally

is correct for non-negative values and **inverts the order for negative ones**: -1.0 has the
larger bit pattern, so it sorts above -0.5 instead of below it. Written as
`np.where(x < 0, -x, x | 0x8000)` on a uint16 view it is worse, because a uint16 view is never
negative and the branch is dead. The consequence is not cosmetic: about half of a
weight-gradient-free tensor is negative, so an unconditional `| 0x8000` comparator reports
garbage ULP distances on all of it. Measured on this project's own data, the wrong comparator
claimed that reordering the reference's summation moved 33 % of the elements of a dot by more
than one ULP, which is four orders of magnitude from the truth and would have made every gate
that used it unfalsifiable. Equal bit patterns still compared equal, so ndiff looked right while
max-ULP was nonsense.

Correct transform (the classic one for IEEE order on integers):

    key = bits ^ (0xFFFF if bits is negative else 0x8000)

`prototype/gdn_decode/check.py` gets this right by a different route — it views BF16 as a
*signed* int16, so its `where(a < 0, -a, ...)` branch is live; viewing as uint16 (which is what
`view(torch.uint16)` gives you, and what every dump loader here uses) makes that branch dead.

`selftest()` below asserts monotonicity over real BF16 values, including across zero, which is
exactly the property the wrong version breaks.
"""
from __future__ import annotations

import numpy as np
import torch


def bf16_bits(t) -> np.ndarray:
    """Raw BF16 payload of `t` as uint16-valued int32 (order-agnostic bit pattern)."""
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.asarray(t))
    return t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype(np.int32)


def monotone_key(bits: np.ndarray) -> np.ndarray:
    """Order-preserving integer key for BF16 bit patterns (see module docstring)."""
    neg = bits & 0x8000 != 0
    return np.where(neg, bits ^ 0xFFFF, bits ^ 0x8000).astype(np.int64)


def bits_from_key(key):
    """Inverse of `monotone_key`, so a probe can step a value exactly one representable ULP."""
    key = np.asarray(key, dtype=np.int64) & 0xFFFF
    return np.where(key >= 0x8000, key ^ 0x8000, key ^ 0xFFFF).astype(np.int32)


def ulp_dist(a, b) -> np.ndarray:
    """Elementwise distance in BF16 ULPs between two same-shaped tensors of BF16 values."""
    return np.abs(monotone_key(bf16_bits(a)) - monotone_key(bf16_bits(b)))


def zero_nan(a) -> tuple[int, int]:
    """How many BF16 elements are zero and how many are NaN, by bit pattern.

    A ULP distance cannot tell these two failures apart: an output of +0 against a pin of -2.39 is
    ~49 000 ULP, and so is almost any other value that collapsed to nothing. A comparison that
    prints only `ndiff` and `max ULP` therefore reads the same for "the kernel never wrote real
    values" (wrong dtype, wrong pointer, kernel that did not run) and "the kernel is slightly off"
    (a rounding policy), which are different bugs with different fixes. So the counts are read off
    the pattern: BF16 NaN is exponent all ones with a non-zero mantissa, and ±0 is either sign with
    the other 15 bits zero (denormals are deliberately NOT counted as zero — they are written
    values, and a BF16 denormal is exactly what a wrong-dtype weight read produces).
    """
    bits = bf16_bits(a).ravel()
    exp = (bits >> 7) & 0xFF
    # `bool_mask & int_array` is a bitwise AND, not a logical one: `(exp == 0xFF) & (bits & 0x7F)`
    # ANDs the mask (0/1) into the mantissa and is 0 for every mantissa but 1. Compare first, then
    # combine the two masks.
    nan = (exp == 0xFF) & ((bits & 0x7F) != 0)
    return int((bits & 0x7FFF == 0).sum()), int(nan.sum())


def band_usage(rel, d, ulp_max: int = 1, cancel_rel: float = 1e-4) -> dict:
    """How much of a comparison the cancellation band is actually excusing.

    `rel` is |ref| / max|ref| per element (the scale the band is written against), `d` the ULP
    distances from `ulp_dist`. Returns:

      n               elements compared
      n_band          elements the band covers (|ref| <= cancel_rel x max|ref|)
      n_off           elements further than `ulp_max` (the offenders)
      n_off_in_band   offenders the band excuses — the only ones a scale-aware gate may forgive
      worst_off_rel   the largest rel among offenders: **the tightest band these numbers would
                      pass at**. Anything >= it passes, so a band with no measurement behind it
                      is a band nobody calibrated. Compare it against `cancel_rel` for margin.
      in_band         the band mask, so a caller can build its own `bad` without re-spelling
                      the inequality
    """
    rel = np.asarray(rel).ravel()
    d = np.asarray(d).ravel()
    if rel.size != d.size:
        raise ValueError("band_usage: %d values vs %d distances" % (rel.size, d.size))
    in_band = rel <= cancel_rel
    off = d > ulp_max
    return dict(n=int(d.size), n_band=int(in_band.sum()), n_off=int(off.sum()),
                n_off_in_band=int((off & in_band).sum()),
                worst_off_rel=float(rel[off].max()) if off.any() else 0.0,
                in_band=in_band)


def report(got, ref, ulp_max: int = 1, cancel_rel: float = 1e-4) -> dict:
    """Compare two BF16-valued tensors.

    `ulp_max` is the per-element tolerance for elements big enough for a ULP to mean anything;
    `cancel_rel` is the band below the tensor max in which a ULP distance is NOT meaningful: a
    dot that cancelled to 1e-6 of the tensor max moves many ULPs from ordinary F32 summation
    error, and `floor_probe.py` measures that even the exactly-rounded sum disagrees with the
    reference by 4-16 ULP at such elements. So `n_bad` counts only elements that are both far
    apart and large — a real error shows up there and nothing else does. `n_band` and
    `worst_off_rel` (from `band_usage`) say how much of the comparison the band is excusing and
    how much slack it has; a caller that tightens `cancel_rel` should do it against those two
    numbers and not against taste.
    """
    d = ulp_dist(got, ref)
    gv = got if isinstance(got, np.ndarray) else np.asarray(got.float().numpy())
    rv = ref if isinstance(ref, np.ndarray) else np.asarray(ref.float().numpy())
    gf = gv.astype(np.float64).ravel()
    rf = rv.astype(np.float64).ravel()
    dd = d.ravel()
    scale = float(np.abs(rf).max()) if rf.size else 0.0
    rel = np.abs(rf) / scale if scale else np.zeros(rf.shape, dtype=np.float64)
    bu = band_usage(rel, dd, ulp_max, cancel_rel)
    bad = (dd > ulp_max) & ~bu["in_band"]
    over = np.where(bad, dd, 0)
    denom = np.linalg.norm(gf) * np.linalg.norm(rf)
    return dict(ndiff=int((dd > 0).sum()), n=bu["n"],
                maxulp=int(dd.max()) if dd.size else 0, over1=int((dd > 1).sum()),
                n_bad=int(bad.sum()),
                worst_bad_ulp=int(over.max()) if bad.any() else 0,
                worst_bad_rel=float(np.abs(rf[bad]).max() / scale) if bad.any() else 0.0,
                n_band=bu["n_band"], worst_off_rel=bu["worst_off_rel"],
                cos=float(gf @ rf / denom if denom else 1.0),
                meanrel=float(np.abs(gf - rf).mean() / np.abs(rf).mean()))


def selftest() -> int:
    """The comparator must be monotone and agree with the real ordering of the values."""
    vals = torch.tensor([-300.0, -4.0, -1.0, -0.75, -0.5, -0.25, -0.0078125, -2.0 ** -133,
                         2.0 ** -133, 0.0078125, 0.25, 0.5, 0.75, 1.0, 4.0, 300.0],
                        dtype=torch.bfloat16)
    k = monotone_key(bf16_bits(vals))
    v = vals.double().numpy()
    bad = sum(0 if (k[i] < k[i + 1] and v[i] < v[i + 1]) else 1
              for i in range(len(v) - 1))
    # "1 ULP apart" must mean adjacent representable values, including across zero. Note that
    # torch.nextafter must be called ON BF16 tensors: nextafter in F32 gives a neighbour that
    # rounds straight back to the same BF16, and the test would then measure a distance of 0.
    def nxt(x, toward):
        return torch.nextafter(torch.tensor([x], dtype=torch.bfloat16),
                               torch.tensor([toward], dtype=torch.bfloat16))
    adj = [(torch.tensor([1.0], dtype=torch.bfloat16), nxt(1.0, 2.0), 1),
           (torch.tensor([-1.0], dtype=torch.bfloat16), nxt(-1.0, -2.0), 1),
           (torch.tensor([0.0], dtype=torch.bfloat16), nxt(0.0, 1.0), 1),
           (torch.tensor([-(2.0 ** -133)], dtype=torch.bfloat16),
            # -min_subnormal to +0 is 2 ULP, not 1: the line is -min_sub, -0, +0, +min_sub,
            # and -0/+0 are distinct patterns. Expecting 1 here is the second way to get a
            # ULP comparator wrong.
            torch.tensor([0.0], dtype=torch.bfloat16), 2),
           (torch.tensor([-1.0], dtype=torch.bfloat16), torch.tensor([-1.0078125],
                                                                    dtype=torch.bfloat16), 1)]
    off = [(int(ulp_dist(a, b)[0]), want) for a, b, want in adj]
    print("selftest: %d/%d ordering failures | next-neighbour distances %s (got,want)"
          % (bad, len(v) - 1, off))
    return 0 if (bad == 0 and all(g == w for g, w in off)) else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
