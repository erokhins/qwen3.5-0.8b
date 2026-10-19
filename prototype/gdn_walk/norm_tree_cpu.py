#!/usr/bin/env python3
"""The summation association of `block_sum_pair_f32`, written down and then CHECKED.

WHY THIS EXISTS. Bet G1 (`prototype/gdn_walk/BET.md` §3) moves the per-token k-row and q-row
L2-normalisations out of `rec_loop_kernel` into a pre-pass, and its admission is that `o` and the state
come out BIT-IDENTICAL. The thing that decides bit-identity is not the algebra of a row's sum of squares
— any order gives the same real number — but the ASSOCIATION, i.e. which pairs round first: F32 addition
is not associative, and `AGENTS.md`'s `q̃` row (from `prototype/gdn_collapse/results.md` §2) is the case
where changing a product tree moved 93,522 of 131,072 `o` elements on device while a CPU mirror watched
two spellings of one formula agree. So the hoisting pass has to reproduce `block_sum_pair_f32`'s tree
step for step, and "step for step" has to be a sentence somebody can check, not a vibe.

WHAT THIS FILE CHECKS — and what it cannot.

  (1) An OPERATIONAL simulation of the shipped reduction: the 64-bit mask, the 16/8/4/2/1 order, the
      one-add-per-lane-per-step semantics of `__shfl_down_sync` INCLUDING its out-of-segment rule (a
      down-shuffle whose source lane is past the segment returns the caller's own value, which is why
      simulating only lane 0 would be a different program), the `red[warp]` partial layout, the
      `(threadIdx.x < warps) ? red[threadIdx.x] : 0.f` gate and the final `red[0]` fold. `blockDim.x`
      is 128 because that is the walk's block width, and the width is load-bearing: `warps` is derived
      from it, so a different block width is a different tree.
  (2) A STRUCTURAL derivation of the same sum, written from the halving shape rather than from lane
      state: fold the upper half onto the lower half, halving each step, then the four warp partials in
      the order the second tree's offsets imply. Agreement between (1) and (2) is what makes the header's
      association claim checkable: two programs written differently that must agree bit for bit.
  (3) TEETH. On the real pinned post-conv q/k rows, the tree must differ in bits from a naive sum, from
      a reversed-order sum, and from every one of six mutations of the simulation. A witness that no
      change can break is decoration — the rule this repo has now written down four times.

WHAT IT CANNOT SEE, stated here so the handover cannot overclaim. It is a CPU model: it cannot see the
compiler (whether `kq*kq` reaches an add as its own `v_mul_f32` or folded into a `v_fmac_f32_e32` is a
device fact), it cannot see `rsqrtf` or `expf` (device intrinsics, not correctly-rounded, so the
normalisation factor itself is NOT reproduced here — only the sum it is computed from), and it cannot
see the device. Those three are covered differently: the pre-pass calls the shipped `block_sum_pair_f32`
rather than a re-implementation, the walk's arithmetic text is unchanged
(`prototype/gdn_walk/check_hoist_transcription.py`), and the compiled FP-op census is compared by
`prototype/gdn_walk/walk_isa_diff.py`. The final arbiter is the device bit-identity gate, and if THAT
comes back non-zero the honest report is "G1 became a numerics-band question", not a wider band.

Usage:
    prototype/gdn_walk/norm_tree_cpu.py                    # the checks, on pinned data
    prototype/gdn_walk/norm_tree_cpu.py --selftest         # the must-catch set only
    prototype/gdn_walk/norm_tree_cpu.py --fixture data/refs/wide520f
Exit 0 = every check passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = 128                 # one head row, one thread per element
NH = 16
CH = 6144               # the conv tile row: [q 2048 | k 2048 | v 2048]
Q_SCALE = np.float32(0.08838834764831845)
EPS = np.float32(1e-6)
FULL_MASK = 0xFFFFFFFFFFFFFFFF   # the 64-bit mask the shipped source passes

F32 = np.float32


def f32(x) -> np.float32:
    return F32(x)


# ---- (1) the operational simulation ------------------------------------------------------------
def shfl_down(vals: list, delta: int, width: int = 32) -> list:
    """`__shfl_down_sync(mask, v, delta)` over one warp, per the CUDA semantics the compiler honours.

    Lane i reads lane i+delta inside its `width`-lane segment; when that source lane runs past the
    segment, the instruction returns the CALLER'S OWN value, not zero. That quirk never reaches lane 0
    of this reduction (every source lane lane 0 transitively reads is in range), but a simulation that
    quietly substituted 0.0 would be a different program that happens to agree, and the day somebody
    changes the offsets is the day it stops agreeing. So it is spelled out.
    """
    out = list(vals)
    for seg in range(0, len(vals), width):
        for lane in range(seg, seg + width):
            src = lane + delta
            out[lane] = vals[src] if src < seg + width else vals[lane]
    return out


def warp_tree(values: list) -> np.float32:
    """One warp's `v += __shfl_down_sync(FULL_MASK, v, o)` chain for o in 16,8,4,2,1, at lane 0."""
    assert len(values) == 32
    v = list(values)
    for off in (16, 8, 4, 2, 1):
        v = [v[i] + shfl_down(v, off)[i] for i in range(len(v))]
    return F32(v[0])


def block_sum_pair_f32(rows_a: list, rows_b: list, block_threads: int = 128):
    """The shipped pair reduction, one call per (row, row). Returns the two broadcast sums.

    `rows_a` is `kq * kq` for lane j = 0..block_threads-1, i.e. lane j owns element j — the element
    map the walk uses, and the reason the pre-pass must run at the same block width.
    """
    n = len(rows_a)
    assert n == len(rows_b) == block_threads, "one element per lane, same width as the walk"
    warps = block_threads >> 5
    red_a = [F32(0.0)] * 32
    red_b = [F32(0.0)] * 32
    for w in range(warps):
        red_a[w] = warp_tree(rows_a[w * 32:(w + 1) * 32])
        red_b[w] = warp_tree(rows_b[w * 32:(w + 1) * 32])
    # The second stage: only lanes 0..warps-1 hold a partial, the rest hold a literal 0.0, and the same
    # five offsets run again. Only lane 0's result is ever read back (`red[0]`), which is what the walk
    # and the pre-pass both consume.
    t_a = [red_a[i] if i < warps else F32(0.0) for i in range(32)]
    t_b = [red_b[i] if i < warps else F32(0.0) for i in range(32)]
    for off in (16, 8, 4, 2, 1):
        t_a = [t_a[i] + shfl_down(t_a, off)[i] for i in range(32)]
        t_b = [t_b[i] + shfl_down(t_b, off)[i] for i in range(32)]
    return F32(t_a[0]), F32(t_b[0])


# ---- (2) the structural derivation --------------------------------------------------------------
def fold_halves(values: list) -> np.float32:
    """The same warp sum, derived from its shape instead of from lane state.

    `v += shfl_down(v, 16)` then 8 then 4 then 2 then 1 pairs leaves 16 apart first, then 8, and so on:
    after the first step lane i holds `L_i + L_{i+16}`, so the live values are the pairwise sums of the
    two halves, and each further step halves them again. Written that way, the association is a halving
    fold, and nothing here shares code with `warp_tree`.
    """
    cur = list(values)
    while len(cur) > 1:
        h = len(cur) // 2
        cur = [F32(cur[i] + cur[i + h]) for i in range(h)]
    return F32(cur[0])


def fold_partials(partials: list) -> np.float32:
    """The second stage's fold, derived: four live lanes, five offsets, zeros elsewhere.

    Offsets 16 and 4 add lanes that hold a literal 0.0, so the first real pairing is offset 2 —
    `(r0 + r2)` and `(r1 + r3)` — and offset 1 joins them: `(r0 + r2) + (r1 + r3)`. NOT
    `((r0 + r1) + (r2 + r3))`, which is what a reader who assumed a pairwise fold would write, and which
    is mutation M1 below precisely so that the difference is proven to exist.
    """
    assert len(partials) == 4
    return F32(F32(partials[0] + partials[2]) + F32(partials[1] + partials[3]))


def structural_sum(row: list) -> np.float32:
    warps = [fold_halves(row[w * 32:(w + 1) * 32]) for w in range(len(row) // 32)]
    return fold_partials(warps)


# ---- the mutations this witness must catch -------------------------------------------------------
def m1_second_stage_pairwise(rows: list) -> np.float32:
    """Fold the four warp partials `((r0+r1)+(r2+r3))` instead of `(r0+r2)+(r1+r3)`."""
    w = [warp_tree(rows[i * 32:(i + 1) * 32]) for i in range(len(rows) // 32)]
    return F32(F32(w[0] + w[1]) + F32(w[2] + w[3]))


def m3_row_shifted_by_one(rows: list) -> np.float32:
    """Every lane reads one column to the right — a wrong row base or a shifted slice.

    This is the mutation that a REORDERING mutation cannot catch, and the reason it is here instead of a
    reversed leaf order: F32 addition is commutative and the halving pairing maps onto itself under
    reversal (`{i, i+16} -> {15-i, 31-i}`), so reversing the row or flipping the order of the four warp
    partials is BIT-IDENTICAL to doing nothing. Order permutations are invisible to this witness by
    construction; only re-GROUPING is visible. That is a fact about the check, recorded where a future
    reader will not have to rediscover it.
    """
    return warp_tree_all(rows[1:] + rows[:1])


def m4_partial_serial(rows: list) -> np.float32:
    """Fold the four warp partials serially, `((r0+r1)+r2)+r3`, as a single-thread loop would.

    The obvious way to write "sum the partials" in a hoisting pass, and not the shipped tree.
    """
    w = [warp_tree(rows[i * 32:(i + 1) * 32]) for i in range(len(rows) // 32)]
    return F32(F32(F32(w[0] + w[1]) + w[2]) + w[3])


def m2_reversed_offsets(rows: list) -> np.float32:
    """Run the shuffle chain 1,2,4,8,16 instead of 16,8,4,2,1."""
    v = list(rows)
    for off in (1, 2, 4, 8, 16):
        v = [v[i] + shfl_down(v, off)[i] for i in range(len(v))]
    return F32(v[0])


def m3_old_reversed_leaves(rows: list) -> np.float32:
    """FEEDS NOTHING: reversing the row is bit-identical (see m3's note). Kept as the counterexample
    that explains why m3 is a shift and not a reversal — if this ever starts being CAUGHT, something in
    the reasoning above or in `fold_halves` has moved.
    """
    return warp_tree_all(list(reversed(rows)))


def m4_old_partials_flipped(rows: list) -> np.float32:
    """FEEDS NOTHING, same reason as m3_old_reversed_leaves: `(r3+r1)+(r2+r0)` == `(r0+r2)+(r1+r3)`."""
    w = [warp_tree(rows[i * 32:(i + 1) * 32]) for i in range(len(rows) // 32)]
    return fold_partials(list(reversed(w)))


def m4_warp_partials_flipped(rows: list) -> np.float32:
    """Write warp w's partial into `red[warps-1-w]`: the partials land in the wrong order."""
    w = [warp_tree(rows[i * 32:(i + 1) * 32]) for i in range(len(rows) // 32)]
    return fold_partials(list(reversed(w)))


def m5_four_per_lane(rows: list) -> np.float32:
    """Four elements per lane instead of one — the `CONV_VEC` shape, which is a DIFFERENT tree.

    This is the mutation that makes the block-width requirement concrete: a hoisting pass written with 32
    lanes each owning 4 columns computes a legal sum of the same 128 squares and is not bit-identical.
    """
    assert len(rows) == 128
    per_lane = [F32(F32(F32(rows[4 * i] + rows[4 * i + 1]) + rows[4 * i + 2]) + rows[4 * i + 3])
                for i in range(32)]
    return warp_tree(per_lane)


def m6_plain_sum(rows: list) -> np.float32:
    """Whatever numpy's pairwise `sum` does — i.e. no association discipline at all."""
    return F32(np.sum(np.array(rows, dtype=F32), dtype=F32))


def warp_tree_all(rows: list) -> np.float32:
    w = [warp_tree(rows[i * 32:(i + 1) * 32]) for i in range(len(rows) // 32)]
    return fold_partials(w)


MUTATIONS = [
    ("M1 second stage folded pairwise", m1_second_stage_pairwise),
    ("M2 shuffle offsets reversed (1,2,4,8,16)", m2_reversed_offsets),
    ("M3 every lane reads one column to the right", m3_row_shifted_by_one),
    ("M4 warp partials folded serially", m4_partial_serial),
    ("M5 four elements per lane (different block shape)", m5_four_per_lane),
    ("M6 numpy pairwise sum", m6_plain_sum),
]

# Mutations that are provably BIT-IDENTICAL to doing nothing, kept and asserted so the witness cannot
# quietly lose its sense of direction and pretend these are catches.
NO_OPS = [
    ("N1 row reversed (commutativity + reversal-symmetric pairing)", m3_old_reversed_leaves),
    ("N2 warp partials flipped (same reason)", m4_old_partials_flipped),
]


# ---- the data -----------------------------------------------------------------------------------
def load_rows(fixture: Path):
    """The pinned post-conv q and k rows — the real operands of the shipped reduction.

    `L??gdn.{q,k}_post_conv.npy` is [1][tokens][16][128] F32, i.e. exactly the row `rec_loop_kernel`
    reads as `k[h * D + j]`. If the fixture is missing the generator falls back to synthetic values of
    the right kind (bf16-representable, silu-shaped) and SAYS SO, because realistic data is the point.
    """
    out = []
    used = []
    for layer in ("L00", "L01"):
        for name in ("q_post_conv", "k_post_conv"):
            p = fixture / f"{layer}.gdn.{name}.npy"
            if not p.exists():
                continue
            a = np.load(p).astype(F32)
            a = a.reshape(-1, NH, D)
            out.append((f"{layer}.{name}", a))
            used.append(str(p))
    if not out:
        rng = np.random.default_rng(20261019)
        x = rng.standard_normal((4096, NH, D)).astype(F32)
        s = F32(1) / (F32(1) + np.exp(-x).astype(F32))          # silu-ish, like the conv's output
        out.append(("SYNTHETIC(silu of gaussian)", (x * s).astype(F32)))
        used.append("SYNTHETIC — no pinned post-conv rows found")
    return out, used


def bits(x: np.float32) -> int:
    return int(np.float32(x).view(np.uint32))


def ndiff(a: np.ndarray, b: np.ndarray) -> int:
    """Bit-difference count between two float32 arrays (NaN-safe: compares the bit patterns)."""
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="data/refs/p64")
    ap.add_argument("--selftest", action="store_true", help="the must-catch set only")
    args = ap.parse_args()
    fixture = (ROOT / args.fixture) if not Path(args.fixture).is_absolute() else Path(args.fixture)
    datasets, provenance = load_rows(fixture)
    for p in provenance:
        print(f"DATA   {p}")

    rows_all = []           # every 128-element row the checks run over
    for label, arr in datasets:
        n = arr.shape[0] * arr.shape[1]
        print(f"SET    {label}: {n} rows of {D}")
        rows_all.extend(arr.reshape(-1, D))
    assert rows_all, "no rows loaded"

    ok = True

    if not args.selftest:
        # (1)+(2) operational vs structural, every row, both bit patterns equal.
        bad = 0
        for r in rows_all:
            sq_op = block_sum_pair_f32([f32(x * x) for x in r], [f32(0.0)] * D)[0]
            sq_st = structural_sum([f32(x * x) for x in r])
            if bits(sq_op) != bits(sq_st):
                bad += 1
        print(f"CHECK  operational shuffle sim == structural halving derivation: "
              f"{len(rows_all) - bad}/{len(rows_all)} rows bit-equal")
        if bad:
            ok = False
            print(f"FAIL   {bad} rows disagree between the two derivations of the tree")

        # The second stage, spelled as the fold it is: run the operational chain over random partial
        # sets and require it equal the derived `(r0+r2)+(r1+r3)`.
        rng = np.random.default_rng(20261019)
        mism = 0
        for _ in range(20000):
            parts = [F32(rng.random()) for _ in range(4)]
            t = [parts[0], parts[1], parts[2], parts[3]] + [F32(0.0)] * 28
            for off in (16, 8, 4, 2, 1):
                t = [t[i] + shfl_down(t, off)[i] for i in range(32)]
            if bits(t[0]) != bits(fold_partials(parts)):
                mism += 1
        print(f"CHECK  four-partial fold is (r0+r2)+(r1+r3): {20000 - mism}/20000 random partial sets")
        ok = ok and mism == 0

        # Teeth: the association actually matters on this data.
        naive = lo2hi = regroup = 0
        for r in rows_all:
            sq = [f32(x * x) for x in r]
            tree = structural_sum(sq)
            if bits(tree) != bits(F32(np.sum(np.array(sq, dtype=F32), dtype=F32))):
                naive += 1
            acc = F32(0.0)
            for v in sq:
                acc = F32(acc + v)
            if bits(tree) != bits(acc):
                lo2hi += 1
            if bits(tree) != bits(m2_reversed_offsets(sq)):
                regroup += 1
        n = len(rows_all)
        print(f"TEETH  tree differs from numpy sum on {naive}/{n} rows, from left-to-right on "
              f"{lo2hi}/{n}, from the reversed-offset tree on {regroup}/{n}")
        if naive == 0 or lo2hi == 0 or regroup == 0:
            ok = False
            print("FAIL   a naive order reproduced the tree: this data cannot witness the association")

        # The no-op set: bit-identical BY REASON, and asserted so a future edit that makes one of them
        # bite is noticed rather than celebrated.
        for name, fn in NO_OPS:
            moved = sum(1 for r in rows_all
                        if bits(fn([f32(x * x) for x in r])) != bits(structural_sum([f32(x * x) for x in r])))
            print(f"NO-OP  {name}: {moved}/{n} rows moved (expected 0)")
            if moved:
                ok = False

    # The must-catch set: every mutation must move bits somewhere on this data.
    print("MUST-CATCH (a mutation nobody can see means the witness is not witnessing)")
    for name, fn in MUTATIONS:
        moved = 0
        for r in rows_all:
            sq = [f32(x * x) for x in r]
            if bits(fn(sq)) != bits(structural_sum(sq)):
                moved += 1
        status = "caught" if moved else "INVISIBLE"
        print(f"  {status:9s} {name}: {moved}/{len(rows_all)} rows moved")
        if not moved:
            ok = False

    print("RESULT " + ("norm_tree_cpu PASS" if ok else "norm_tree_cpu FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
