#!/usr/bin/env python3
"""CPU bit-identity mirror for route A (D-12): the collapsed, resident GDN recurrence has to be the
SAME computation as the shipped sequential one, not a close approximation of it.

    .venv/bin/python prototype/gdn_collapse/collapse_mirror.py --quick
    .venv/bin/python prototype/gdn_collapse/collapse_mirror.py              # ~4 min, CPU only
    .venv/bin/python prototype/gdn_collapse/collapse_mirror.py --selftest

WHY THIS FILE EXISTS. `engine/gdn_collapse_kernels.h` keeps §04.4's state in LDS for the whole wave
body, so `S` is staged once per (layer, body) instead of loaded and stored once per token. Staging is
exactly where a recurrence can silently change function: the state could be rounded on the way in, a
strip could stage the wrong columns, the norm reduction could be a different tree at a different width,
the serial `i` walk could be re-associated or fused with the update, a per-token scalar could get
hoisted. §04.4 measured what that costs — 1 element in 286 of `o` loses 84 % of its magnitude to
cancellation inside those dots — so "close" is not the gate. The gate is: identical operands and
identical starting state in, every bit of `o` and of the resulting `S` equal out.

WHAT A CPU MIRROR CAN AND CANNOT PROVE, said before it is believed. Both orders are modelled under ONE
host rounding model: numpy F32 scalars, `rsqrtf`/`expf` replaced by the same host call on both sides,
multiply-then-add where the device may contract to an FMA. A host-vs-device difference in any of those
hits both sides identically and cancels in the comparison; what the mirror therefore cannot see is a
difference in FMA contraction or in the intrinsics. Those rest on the collapsed kernel using the shipped
kernel's SOURCE EXPRESSIONS verbatim (a textual claim; `collapse_price.py --isa` prints the instruction
census that supports it) and on the device gate (byte-identical logits — commands in
`bench/p9-route-a-2026-10-14.md`). What the mirror does prove, and no amount of review does: that the
residency, the strip partition, the staging layout, the swizzle, the write-back and the per-token scalar
reads carry no arithmetic consequence — including at a BODY BOUNDARY, which is the second control.

THE WITNESS THAT THIS IS NOT A PRIVATE INVENTION: the sequential side is checked against the pinned HF
layer dumps (`state_new`, `o_pre_norm`) before the two mirrors are compared with each other. A mirror
that only agrees with itself could be agreeing about the wrong recurrence — the trap §10.5 and
`prototype/attn_tile_model.py` are both about.

NEGATIVES: a comparison that cannot fail proves nothing. `--selftest` injects six defects into the
collapsed side and requires each to MOVE at least one bit, then applies two controls that must move
nothing. If a future edit blinds the mirror to any of the six, or makes it paranoid about the two, it
exits non-zero.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "data" / "refs"

D, NH = 128, 16                               # d_k = d_v = 128, 16 value heads (§01.6)
Q_SCALE = np.float32(0.08838834764831845)     # 128^{-1/2}, the kernel's literal
EPS = np.float32(1e-6)                        # the kernel's `+ 1e-6f`
STRIPS = 4                                    # blocks per head in the collapsed form
COLS = D // STRIPS                            # 32 d_v columns per block


# ---- shared primitives ------------------------------------------------------------------------
def shfl_down(v: np.ndarray, delta: int) -> np.ndarray:
    """`__shfl_down_sync(mask, v, delta)` over the last axis (32 lanes) as this device implements it: a
    lane whose source lane (lane + delta) leaves the group gets its OWN value back, so out-of-range
    lanes double. Only lane 0's value is ever consumed, and lane 0's read chain (16, 8, 4, 2, 1) never
    reads a doubled lane — which is what makes `block_sum`'s unguarded `v += shfl_down(v, d)` an exact
    32-wide sum rather than an aliasing accident. `block_sum_scalar` re-derives the same program lane by
    lane, so the vectorised version is checked against something rather than trusted."""
    n = v.shape[-1]
    lane = np.arange(n)
    src = np.where(lane + delta < n, lane + delta, lane)
    return (v + v[..., src]).astype(np.float32)


def block_sum(x: np.ndarray) -> np.ndarray:
    """`gdn_decode_kernels.h::block_sum` at `blockDim.x == 128`, per row: (n, 128) -> (n,).

    Four 32-lane butterflies write red[0..3] from each warp's lane 0; a SECOND butterfly then runs over
    [red0, red1, red2, red3, 0, ...] and lane 0's value is broadcast. Both stages are modelled, zeros
    and doubling included, because the shape of this tree is what a lane-count change silently alters —
    and it is why the collapsed kernel keeps `blockDim.x == 128` and calls the same function instead of
    re-deriving a 32-lane equivalent of it.
    """
    v = x.reshape(-1, 4, 32).astype(np.float32, copy=True)
    for d in (16, 8, 4, 2, 1):
        v = shfl_down(v, d)
    red = v[..., 0]                                   # each warp's lane 0 wrote red[w]
    t = np.zeros((red.shape[0], 32), np.float32)
    t[:, :4] = red
    for d in (16, 8, 4, 2, 1):
        t = shfl_down(t, d)
    return np.ascontiguousarray(t[:, 0])              # red[0], broadcast back to every thread


def block_sum_scalar(row: np.ndarray) -> float:
    """A second, deliberately dumb implementation of the same program: one row, lane-by-lane lists,
    explicit shuffles by index, doubling included. `check_block_sum` requires it to agree BIT-FOR-BIT
    with the vectorised `block_sum` above, which is the only non-vacuous check available here — a tree
    sum is NOT the ascending serial sum bit-for-bit, so comparing against `np.sum` would be wrong.
    What both are checked against is the program the kernel runs."""
    lanes = [np.float32(x) for x in row]

    def butterfly(vals):
        for delta in (16, 8, 4, 2, 1):
            nxt = []
            for ln in range(len(vals)):
                src = ln + delta if ln + delta < len(vals) else ln   # own value when out of range
                nxt.append(np.float32(vals[ln] + vals[src]))
            vals = nxt
        return vals

    warps = [lanes[w * 32:(w + 1) * 32] for w in range(4)]
    red = [np.float32(butterfly(w)[0]) for w in warps]     # lane 0 of each warp wrote red[w]
    second = [np.float32(red[i]) if i < 4 else np.float32(0.0) for i in range(32)]
    return float(butterfly(second)[0])                     # red[0], broadcast to every thread


def check_block_sum() -> None:
    """The two models of the shipped reduction must agree bit-for-bit, and must be near — not equal to —
    a plain F32 sum, which is the point of modelling the tree at all."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for trial in range(40):
        x = (rng.standard_normal((3, D)) * (10.0 ** rng.integers(-4, 3))).astype(np.float32)
        got = block_sum(x * x)
        want = np.array([block_sum_scalar(x[r] * x[r]) for r in range(3)], np.float32)
        if not np.array_equal(got, want):
            raise AssertionError(f"block_sum models disagree at trial {trial}: {got} vs {want}")
        worst = max(worst, float(np.max(np.abs(got - np.sum(x * x, axis=1, dtype=np.float32))
                                       / np.maximum(np.abs(got), 1e-30))))
    print(f"block_sum model: two independent implementations agree bit-for-bit on 40 rows; "
          f"worst gap to a plain F32 sum {worst:.2e} (association, as expected — that is why the "
          f"collapsed kernel reuses the same function instead of re-deriving it)")


def rsqrt(x: np.ndarray) -> np.ndarray:
    return (np.float32(1.0) / np.sqrt(x)).astype(np.float32)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even F32 -> BF16 -> F32, in bits, so numpy stays the mirror's only dependency."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))).astype(np.uint32)
    return (r & np.uint32(0xFFFF0000)).view(np.float32)


def strip_columns(neg=None, strips_rev=False) -> np.ndarray:
    """`gcol[h, x, c]` is the global `d_v` column that block (x, h)'s lane c stages, updates, writes
    back, and writes its `o` at — the kernel spells all four with the same `cbase + tid`, and modelling
    it once is what makes a mapping bug land on every one of them at once.

    `strip_shift` takes the base from the HEAD index instead of the strip index, which is the mapping
    bug this layout is most exposed to: four blocks of a head then own the same 32 columns, four columns
    ranges own nothing, and nothing about either output's shape reveals it."""
    lane = np.arange(COLS, dtype=np.int64)
    if neg == "strip_shift":
        base = (np.arange(NH) % STRIPS) * COLS
        gcol = base[:, None, None] + lane[None, None, :]
        return np.broadcast_to(gcol, (NH, STRIPS, COLS)).copy()
    base = np.arange(STRIPS, dtype=np.int64) * COLS
    if strips_rev:
        base = base[::-1]
    return (base[None, :, None] + lane[None, None, :]).reshape(1, STRIPS, COLS).repeat(NH, axis=0)


# ---- the two orders ---------------------------------------------------------------------------
def seq_order(q, k, v, g, beta, S0):
    """Mirrors `rec_kernel`: one launch per token, 128 threads, thread j owns column j of `S[h]` and
    walks `i` from 0 to 127 ascending in two separate loops. Vectorising over the (head, column) lanes
    does not couple them — each pair is its own scalar chain, run in the same order the threads run in.

    Returns (o[T, NH, D], S[NH, D, D])."""
    S = S0.copy()
    o = np.zeros((q.shape[0], NH, D), np.float32)
    for t in range(q.shape[0]):
        kq, qq = k[t], q[t]                                   # thread j's element is [:, j]
        kn = (kq * rsqrt(block_sum(kq * kq) + EPS)[:, None]).astype(np.float32)
        qn = ((qq * rsqrt(block_sum(qq * qq) + EPS)[:, None]) * Q_SCALE).astype(np.float32)
        gam = np.exp(g[t]).astype(np.float32)[:, None]        # expf(g[h]) per head
        bet = beta[t][:, None]
        r = np.zeros((NH, D), np.float32)
        for i in range(D):                                    # step 2
            r += S[:, i, :] * kn[:, i][:, None]
        dv = (bet * (v[t] - gam * r)).astype(np.float32)
        acc = np.zeros((NH, D), np.float32)
        for i in range(D):                                    # step 3, same ascending walk
            s = (gam * S[:, i, :]).astype(np.float32)
            s += kn[:, i][:, None] * dv
            S[:, i, :] = s
            acc += qn[:, i][:, None] * s
        o[t] = acc
    return o, S


def collapse_order(q, k, v, g, beta, S0, *, neg=None, strips_rev=False):
    """Mirrors `engine/gdn_collapse_kernels.h`: one launch per body, one block per (strip, head), the
    block's state resident in an LDS-shaped array for the whole body, staged once and written back once.

    The staging is a real gather into a real copy, so a wrong mapping stages the wrong cells, updates
    them and writes them back — which shows up as a difference instead of being hidden by vectorising
    over the whole state. `neg` injects exactly one defect; `NEG_EXPECT` is the list and the reason."""
    T = q.shape[0]
    S = S0.copy()
    o = np.zeros((T, NH, D), np.float32)
    gcol = strip_columns(neg, strips_rev)             # (NH, STRIPS, COLS)
    flat = gcol.reshape(NH, D)
    wide = np.broadcast_to(flat[:, None, :], (NH, D, D))

    # __shared__ float Sst[]: stage the strip (128 rows x 32 columns), and touch nothing but this until
    # the write-back. `Sst[:, i]` is the block's row i, in (head, strip, lane) order.
    Sst = np.take_along_axis(S, wide, axis=2).reshape(NH, D, STRIPS, COLS)
    if neg == "state_bf16":
        Sst = bf16_round(Sst)
    vcols = np.take_along_axis(v, np.broadcast_to(flat[None], (T, NH, D)),
                              axis=2).reshape(T, NH, STRIPS, COLS)

    for t in range(T):
        kq, qq = k[t], q[t]
        if neg == "norm_half":
            mask = np.zeros(D, np.float32)
            mask[: D // 2] = 1.0                            # half the elements reach the tree
            kq, qq = kq * mask, qq * mask
        kn = (kq * rsqrt(block_sum(kq * kq) + EPS)[:, None]).astype(np.float32)
        qn = ((qq * rsqrt(block_sum(qq * qq) + EPS)[:, None]) * Q_SCALE).astype(np.float32)
        if neg == "gam_hoist" and t > 0:
            pass                                            # gam/bet below keep token 0's values
        else:
            gam = np.exp(g[t]).astype(np.float32).reshape(NH, 1, 1)
            bet = beta[t].reshape(NH, 1, 1)
        r = np.zeros((NH, STRIPS, COLS), np.float32)
        if neg == "fused_loop":
            # The trap the two-loop structure exists to prevent: dv from a partial r, and the update
            # pass reading state cells it already overwrote this token.
            acc = np.zeros((NH, STRIPS, COLS), np.float32)
            for i in range(D):
                r += Sst[:, i] * kn[:, i][:, None, None]
                dv = (bet * (vcols[t] - gam * r)).astype(np.float32)
                s = (gam * Sst[:, i]).astype(np.float32)
                s += kn[:, i][:, None, None] * dv
                Sst[:, i] = s
                acc += qn[:, i][:, None, None] * s
        else:
            walk = reversed(range(D)) if neg == "i_desc" else range(D)
            for i in walk:                                  # step 2, inside the strip
                r += Sst[:, i] * kn[:, i][:, None, None]
            dv = (bet * (vcols[t] - gam * r)).astype(np.float32)
            acc = np.zeros((NH, STRIPS, COLS), np.float32)
            for i in range(D):                              # step 3
                s = (gam * Sst[:, i]).astype(np.float32)
                s += kn[:, i][:, None, None] * dv
                Sst[:, i] = s
                acc += qn[:, i][:, None, None] * s
        # o follows the same cbase the state does: one thread, one column, one output element.
        np.put_along_axis(o[t], flat, acc.reshape(NH, D), axis=1)
    # ONE write-back per (layer, body), unrounded: an F32 round trip is exact.
    np.put_along_axis(S, wide, Sst.reshape(NH, D, D), axis=2)
    return o, S


# ---- reporting --------------------------------------------------------------------------------
def bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    neq = bits(a) != bits(b)
    out = {"n": int(neq.sum()), "total": int(neq.size)}
    if neq.any():
        out["max_abs"] = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load(fixture: str, layer: int):
    """Real pinned per-layer GDN operands: q|k|v post-conv [T][NH][D], g_decay_log and beta [T][NH].
    Every dump carries a leading batch of 1, which is dropped here (asserted, not assumed)."""
    d = REFS / fixture
    def rd(name):
        a = np.load(d / f"L{layer:02d}.gdn.{name}.npy").astype(np.float32)
        assert a.shape[0] == 1, f"{name}: batch {a.shape[0]} != 1"
        return np.ascontiguousarray(a[0])
    pins = {n: rd(n) for n in ("state_new", "o_pre_norm")
            if (d / f"L{layer:02d}.gdn.{n}.npy").exists()}
    return rd("q_post_conv"), rd("k_post_conv"), rd("v_post_conv"), rd("g_decay_log"), rd("beta"), pins


def tile(a: np.ndarray, rows: int) -> np.ndarray:
    """Lengthen a body by tiling the fixture's own rows: the only honest way to get a 1024-row body out
    of a 277-token prefill. Labelled synthetic-wide everywhere it is used."""
    if rows <= a.shape[0]:
        return a[:rows]
    return np.repeat(a, -(-rows // a.shape[0]), axis=0)[:rows]


NEG_EXPECT = {
    # Each one MUST move bits, or the mirror is not looking at what it claims to look at.
    "state_bf16": "state rounded to BF16 on the way into LDS",
    "strip_shift": "strip column base taken from the head index: strips collide, ranges own nothing",
    "norm_half": "norm reduction fed half of the squared elements",
    "i_desc": "step-2 dot walked with i descending (same terms, re-associated)",
    "fused_loop": "read and update fused into one pass, dv from a partial r",
    "gam_hoist": "gam/beta hoisted out of the token loop (one decay for the whole body)",
}


def run_selftest(q, k, v, g, beta, S0, rows) -> int:
    """Six injections that must break bit-identity, two controls that must not. Short bodies: the
    question is which defects the mirror can see, not how long the state takes to drift.

    THE STARTING STATE IS CARRIED, NOT ZERO, AND THAT IS THE POINT. Staging a BF16-rounded zero is
    still zero, so from a zero start `state_bf16` moves nothing and the negative would pass vacuously —
    the trap `prototype/attn_tile_model.py` and the mlp_gemv results note are both about a must-catch
    set that has to be derived from the geometry, and here the geometry says a defect living in the
    staging is only visible when there is something in the state to stage. The device counterpart of
    that rule is written in `bench/p9-route-a-2026-10-14.md`: the same negative must be run on a wave
    with MORE THAN ONE body, where the second body stages the first one's state."""
    qs, ks, vs, gs, bs = (a[:rows] for a in (q, k, v, g, beta))
    print(f"\n== negatives and controls ({rows} tokens, carried non-zero starting state)")
    rc = 0
    o_seq, S_seq = seq_order(qs, ks, vs, gs, bs, S0)
    for neg, why in NEG_EXPECT.items():
        o_col, S_col = collapse_order(qs, ks, vs, gs, bs, S0, neg=neg)
        do, dS = diff(o_seq, o_col), diff(S_seq, S_col)
        moved = do["n"] > 0 or dS["n"] > 0
        print(f"   [{'CAUGHT ' if moved else 'MISSED '}] {neg:<12} o {do['n']:>7}/{do['total']}"
              f" S {dS['n']:>9}/{dS['total']}  {why}" + ("" if moved else "  <-- INVISIBLE, gate is blind"))
        rc |= 0 if moved else 1
    # CONTROL 1: strip order. Reversing which block owns which columns must change nothing, because
    # every value follows its own column base. If this moves, the negatives move for the wrong reason.
    o_r, S_r = collapse_order(qs, ks, vs, gs, bs, S0, strips_rev=True)
    do, dS = diff(o_seq, o_r), diff(S_seq, S_r)
    clean = do["n"] == 0 and dS["n"] == 0
    print(f"   [{'CLEAN  ' if clean else 'MOVED  '}] {'strips_rev':<12} o {do['n']:>7}/{do['total']}"
          f" S {dS['n']:>9}/{dS['total']}  strips processed in the other order"
          + ("" if clean else "  <-- values did not follow their columns"))
    rc |= 0 if clean else 1
    # CONTROL 2: the body boundary. Splitting the body in two and handing S from one to the other is
    # what a wave does with its 512-row bodies, so the state round trip has to be invisible. This is
    # the residency claim itself, not a proxy for it: F32 out and F32 back is an exact round trip.
    half = rows // 2
    o_a, S_a = collapse_order(qs[:half], ks[:half], vs[:half], gs[:half], bs[:half], S0)
    o_b, S_b = collapse_order(qs[half:], ks[half:], vs[half:], gs[half:], bs[half:], S_a)
    do, dS = diff(o_seq, np.concatenate([o_a, o_b], axis=0)), diff(S_seq, S_b)
    clean = do["n"] == 0 and dS["n"] == 0
    print(f"   [{'CLEAN  ' if clean else 'MOVED  '}] {'body_split':<12} o {do['n']:>7}/{do['total']}"
          f" S {dS['n']:>9}/{dS['total']}  body split in two, state handed between the halves"
          + ("" if clean else "  <-- staging or write-back lost bits at the boundary"))
    rc |= 0 if clean else 1
    print("SELFTEST", "FAIL" if rc else "PASS")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="p300")
    ap.add_argument("--layer", type=int, default=0, help="a GDN layer: 0, 1, 2, 4, 5, 6, ... (§03.5)")
    ap.add_argument("--wide", type=int, default=1024,
                    help="also mirror a body this long (pinned rows tiled); 0 skips it")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fixture rows only, 24-token negatives")
    args = ap.parse_args()

    check_block_sum()
    q, k, v, g, beta, pins = load(args.fixture, args.layer)
    rows = q.shape[0]
    print(f"operands: {args.fixture}/L{args.layer:02d} gdn q|k|v post-conv + g_decay_log + beta, "
          f"{rows} pinned rows")
    rc = 0
    runs = [(rows, False)] + ([(args.wide, True)] if args.wide and not args.quick else [])
    # Both starting states, because they exercise different things: zeros is a fresh request (§06.1) and
    # the one a prefill body 0 actually sees; a carried non-zero state is what body 1 stages, and what a
    # staging defect needs in order to be visible at all.
    for state in (["zeros", "random"] if not args.quick else ["zeros"]):
        rng = np.random.default_rng(11)
        S0 = (np.zeros((NH, D, D), np.float32) if state == "zeros"
              else (rng.standard_normal((NH, D, D)) * 0.1).astype(np.float32))
        for n, synthetic in runs:
            qq, kk, vv, gg, bb = (tile(a, n) for a in (q, k, v, g, beta))
            t0 = time.time()
            o_seq, S_seq = seq_order(qq, kk, vv, gg, bb, S0)
            o_col, S_col = collapse_order(qq, kk, vv, gg, bb, S0)
            do, dS = diff(o_seq, o_col), diff(S_seq, S_col)
            print(f"\n== rows={n} state={state}"
                  + ("  (synthetic-wide: pinned rows tiled to lengthen the body)" if synthetic else ""))
            print(f"   o bit-diff {do['n']}/{do['total']}   S bit-diff {dS['n']}/{dS['total']}"
                  f"   [{time.time() - t0:.1f}s]")
            if do["n"] or dS["n"]:
                print(f"   FAIL {do} {dS}")
                rc = 1
            else:
                print("   BIT-IDENTICAL: every element of o and of the resulting state is the same bit")
            if not synthetic and state == "zeros" and "o_pre_norm" in pins:
                print(f"   pinned witness (mirror vs HF: expected close, not bit-exact — §10.1) "
                      f"o cos={cos(pins['o_pre_norm'], o_seq):.9f} "
                      f"max_abs={np.abs(pins['o_pre_norm'] - o_seq).max():.3e} | "
                      f"state cos={cos(pins['state_new'], S_seq):.9f} "
                      f"max_abs={np.abs(pins['state_new'] - S_seq).max():.3e}")
    if args.selftest:
        rng = np.random.default_rng(11)
        rc |= run_selftest(q, k, v, g, beta,
                           (rng.standard_normal((NH, D, D)) * 0.1).astype(np.float32),
                           24 if args.quick else 48)
    return rc


if __name__ == "__main__":
    sys.exit(main())
