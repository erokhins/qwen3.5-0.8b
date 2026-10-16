#!/usr/bin/env python3
"""CPU-side bit-identity evidence for the two slim levers, and the price in D20's units.

WHAT THIS CAN AND CANNOT PROVE (read before quoting it):
  * It CAN prove that two source spellings of one reduction walk the same operands in the same
    order and therefore produce the same bits. That is the whole content of lever 0 (the rolled dot
    vs the unrolled one) and lever 1 (the compiler's `__shfl_down_sync` lowering vs a hand
    `ds_bpermute` with the same clamp).
  * It CANNOT prove what the compiler does to the expression: FMA contraction, `v_dual` pairing, and
    association choices are invisible to a CPU mirror. This repo has been bitten by exactly that
    blindness twice (`AGENTS.md`: "a hoisted scalar constant is a different number, and a CPU mirror
    is blind to it"; "a CPU-only geometry selftest is not a witness for the kernel"). So this file
    is a necessary check, and the witness for the device is the A/B in `HANDOFF.md`: every plan must
    print `ndiff_aout=0 ndiff_gated=0`.

The float arithmetic model: products and sums are evaluated in F64 and rounded once to F32, which
is the closest numpy gets to a fused `v_fma_f32`. Both spellings use the SAME helper, so a
disagreement here can only come from order or operands, which is what is under test; the residual
double-rounding of the F64 intermediate is identical on both sides and is not a claim about the
device. `--fma-model` prints which model is in use.
"""
from __future__ import annotations

import argparse
import itertools
import numpy as np

LANES = 32
FULL = 0xFFFFFFFFFFFFFFFF


def f32(x) -> np.float32:
    return np.float32(x)


def fma(a, b, c):
    """One-rounding multiply-add, as `v_fma_f32` does it (see the module docstring's caveat)."""
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def rand_touches(n, rng):
    """Magnitudes that stress cancellation: the cases where an order change actually shows up."""
    for _ in range(n):
        q = rng.standard_normal(8).astype(np.float32) * rng.choice([1.0, 1e-4, 1e4], 8).astype(np.float32)
        k = rng.standard_normal(8).astype(np.float32) * rng.choice([1.0, 1e-4, 1e4], 8).astype(np.float32)
        # The shipped dot's operands are BF16 on both sides (q_rope and the staged K row), which is
        # the only place the format matters: bf16 -> f32 is lossless, so the conversion is a no-op
        # on the value and the probe is really about order.
        yield q.astype(np.float32), k.astype(np.float32)


def dot_rolled(q, k, nd):
    """`for (t = 0; t < nd; ++t) dot += bf(k[t]) * q[t];` — the shipped spelling, nd at runtime."""
    dot = np.float32(0.0)
    for t in range(nd):
        dot = fma(q[t], k[t], dot)
    return dot


def dot_unrolled(q, k, nd):
    """The slim spelling: one 128-bit read of the row, constant trip count, same t order."""
    row = np.array(k[:8], dtype=np.float32)      # what one ds_load_b128 delivers, in the same order
    dot = np.float32(0.0)
    for t in range(8 if nd == 8 else 4):
        dot = fma(q[t], row[t], dot)
    return dot


def shfl_down_llvm(v, delta):
    """One `__shfl_down_sync` step as LLVM lowers it: `ds_bpermute`, delta forced to 0 out of range."""
    src = np.empty(LANES, dtype=np.float32)
    for lane in range(LANES):
        d = delta if lane + delta < LANES else 0      # v_cndmask 0, delta, (threshold > lane)
        src[lane] = v[lane + d]                        # bpermute addr = 4*(lane+d); self on clamp
    with np.errstate(over="ignore", invalid="ignore"):
        return v + src


def shfl_down_hand(v, delta):
    """The hand-spelled `vslim_permute_add`: same clamp, same add, no guards, address hoisted."""
    idx = np.array([lane + delta if lane + delta < LANES else lane for lane in range(LANES)])
    with np.errstate(over="ignore", invalid="ignore"):
        return v + v[idx]


def fold_shipped(v):
    for o in (16, 8, 4, 2, 1):
        v = shfl_down_llvm(v, o)
    return v


def fold_slim(v):
    v = shfl_down_llvm(v, 16)          # step 16 is left to the compiler in the slim form too
    for d in (8, 4, 2, 1):
        v = shfl_down_hand(v, d)
    return v


def bits(x):
    return np.float32(x).tobytes()


def edge_vectors(rng):
    """The corners a random draw will not hit: ±0, ±inf, NaN, subnormals, exact cancellation."""
    out = []
    for pat in (
        [0.0] * 32,
        [-0.0] * 32,
        [0.0, -0.0] * 16,
        [1e-38] * 32,
        [5e-44, 0.0] * 16,
        [1e38, -1e38] * 16,
        [float("inf"), 1.0] * 16,
        [float("-inf"), 1.0] * 16,
        [float("nan"), 1.0] * 16,
    ):
        out.append(np.array(pat, dtype=np.float32))
    for _ in range(2000):
        out.append((rng.standard_normal(32) * rng.choice([1.0, 1e-8, 1e8])).astype(np.float32))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--fma-model", action="store_true")
    args = ap.parse_args()
    if args.fma_model:
        print("fma-model: F64 intermediate, single round to F32; identical on both sides of every "
              "comparison here, so it adjudicates order, not contraction")
        return 0
    rng = np.random.default_rng(20261016)
    bad = 0

    for nd in (8, 4):
        diff = 0
        for q, k in itertools.islice(rand_touches(args.trials, rng), args.trials):
            if bits(dot_rolled(q, k, nd)) != bits(dot_unrolled(q, k, nd)):
                diff += 1
        print(f"PROBE dot nd={nd} trials={args.trials} bitdiff={diff} verdict="
              f"{'PASS' if diff == 0 else 'FAIL'}")
        bad += diff

    diff = 0
    vecs = edge_vectors(rng)
    for v in vecs:
        if bits(fold_shipped(v)) != bits(fold_slim(v)):
            diff += 1
    for _ in range(args.trials):
        v = (rng.standard_normal(LANES) * rng.choice([1.0, 1e-20, 1e20])).astype(np.float32)
        if bits(fold_shipped(v)) != bits(fold_slim(v)):
            diff += 1
    print(f"PROBE fold-all-lanes trials={len(vecs) + args.trials} bitdiff={diff} verdict="
          f"{'PASS' if diff == 0 else 'FAIL'}   note: every lane, not only lane 0")
    bad += diff

    # The negative control this probe owes: if the fold comparison cannot fail, it proves nothing.
    # Reversing the butterfly order is a different association, and must show a difference.
    def fold_wrong(v):
        for o in (1, 2, 4, 8, 16):
            v = shfl_down_llvm(v, o)
        return v
    caught = sum(1 for v in vecs if bits(fold_shipped(v)) != bits(fold_wrong(v)))
    print(f"PROBE fold-negative-control must-catch={caught} of {len(vecs)} verdict="
          f"{'PASS' if caught else 'FAIL (control is vacuous)'}")
    bad += 0 if caught else 1

    # Same discipline for the dot, with both sets DERIVED from the geometry rather than hoped for:
    # the terms t = 4..7 are what `nd` removes, so the must-catch set is the touches where those
    # terms are the ones that carry the value, and the structural must-not-catch set is the touches
    # where they are zero — there `nd` cannot matter, and a control that demands a difference there
    # is the hand-listed "everything" set the AGENTS.md trap row warns about.
    def half(qk, hi):
        q, k = qk
        q = q.copy()
        k = k.copy()
        if hi:
            q[:4] = 0.0
            q[4:] = 1.0
            k[4:] = np.float32(2.0 ** np.arange(4))
        else:
            q[4:] = 0.0
        return q, k
    must = sum(1 for qk in itertools.islice(rand_touches(1000, rng), 1000)
               if bits(dot_rolled(*half(qk, True), 8)) != bits(dot_rolled(*half(qk, True), 4)))
    blind = sum(1 for qk in itertools.islice(rand_touches(1000, rng), 1000)
                if bits(dot_rolled(*half(qk, False), 8)) != bits(dot_rolled(*half(qk, False), 4)))
    print(f"PROBE dot-negative-control must-catch={must}/1000 must-not-catch-violations={blind}"
          f" verdict={'PASS' if must == 1000 and blind == 0 else 'FAIL'}")
    bad += (1000 - must) + blind

    print(f"PROBE summary {'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
