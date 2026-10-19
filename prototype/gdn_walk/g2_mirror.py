#!/usr/bin/env python3
"""g2_mirror.py — CPU witness for bet G2: the GDN recurrence walked by a 32-lane block.

WHAT IT ADJUDICATES. `engine/gdn_rec_wf_kernels.h::rec_wf_kernel` re-spells §04.4's recurrence as ONE
WAVEFRONT PER BLOCK (32 threads, `LDS = 0`, no `__syncthreads()` by construction), each lane owning
several state COLUMNS, with the two row L2-norms reduced by warp butterfly from data the lane already
loaded. That geometry cannot keep the shipped bits, for three named reasons, all of them this file
prices:

  1. the norm sum's ASSOCIATION. The shipped value comes from `block_sum_pair_f32` at `blockDim.x = 128`
     (four warp partials folded in a fixed pattern) — simulated exactly by `norm_tree_cpu.block_sum_pair_f32`,
     which is imported rather than re-derived, per the "list the census tools" rule. A 32-lane block holds
     4 elements per lane and folds with an xor butterfly: a different tree, therefore a different F32 sum.
  2. the state dot's ASSOCIATION. The shipped `r`/`acc` chains are one serial 128-term sum per column;
     here each lane accumulates its own columns and walk A interleaves the k-dot and the q-dot.
  3. the OUTPUT algebra. Without shared memory there is no way to broadcast `q̂[i]` into the second walk, so
     the q-dot over the UPDATED state is taken in closed form instead:
     `o_j = Σ_i q̂_i S_new[i][j] = Q_SCALE·ι·(γ·Σ_i q_i S_old[i][j] + κ·(Σ_i k_i q_i)·dv_j)`.
     That is the distributive law, not an approximation — but it is a different product tree, and
     `AGENTS.md`'s `q̃` row is exactly the precedent that a different product tree is a different number.

So this mirror does NOT chase bit-equality (BET.md §10 constraint 2: bit-identity is not required and the
band is already fixed at cos >= 0.9995 / meanrel <= 5 % through `engine/gate.py`). It answers the question
the band actually needs: how big is the movement, and how much of it is RE-ASSOCIATION ANY REWRITE PAYS
versus what THIS geometry pays. That second half is the floor probe (§10.1's style): `legal` below is the
same algebra summed in 4 partial accumulators, which is what a compiler is free to do to a rolled loop,
and it is the reference for "cannot be helped".

WHAT IT CANNOT SEE, stated so the handover cannot overclaim. It is a CPU model: it cannot see the compiler
(contraction, unrolling, whether the uniform k/q loads scalarize), it cannot see `rsqrtf`/`expf` (device
intrinsics, not correctly rounded — both sides use numpy's, so the intrinsic error CANCELS in this
comparison and the numbers below are purely about association and geometry), and it cannot see the device.
The device witness is the correctness gate named in `prototype/gdn_walk/G2-result.md`.

Inputs are the PINNED post-conv dumps (`data/refs/p64/L00.gdn.*`), not synthetic noise, so the operand
magnitudes and the cancellation structure are the real ones; `--random` adds a seeded random-tensor case
on top. Reference outputs are the pinned `L00.gdn.state_new` / `o_pre_norm`, which the shipped engine
reproduces (`§10.5`), AND a numpy reproduction of the shipped walk in this file's own spelling — the
three-way check is what makes "the mirror implements the shipped recurrence" checkable rather than
assumed.

Usage:
    .venv/bin/python prototype/gdn_walk/g2_mirror.py [--fixture data/refs/p64] [--layers L00,L01]
    .venv/bin/python prototype/gdn_walk/g2_mirror.py --random 8 --seed 7
    .venv/bin/python prototype/gdn_walk/g2_mirror.py --selftest       # the mirror's own teeth
Exit 0 = every case inside the pre-registered band (cos >= 0.9995, meanrel <= 5 %).
"""
from __future__ import annotations

import argparse
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "gdn_walk"))
import norm_tree_cpu as nt   # noqa: E402  the shipped reduction's operational simulation

D = 128
NH = 16
Q_SCALE = np.float32(0.08838834764831845)
EPS = np.float32(1e-6)
LANES = 32
F32 = np.float32

# The band, taken from BET.md §10 constraint 2 — the EXISTING §10.1 aggregates. Not tunable from here:
# a mirror that lets you widen the band from its own command line is not a gate.
COS_FLOOR = 0.9995
MEANREL_CEIL = 0.05
# The cancellation band: an element whose reference magnitude is below this fraction of the tensor max is
# excluded from the max-rel statistic, because at that size one cancelled add owns the quotient
# (`prototype/mlp_gemv/floor_probe.py`, §10.1 F1-F4). ndiff is reported but NOT gated: for a chained dot
# the legal ndiff is 29-47 % (§10.1), so ndiff there measures nothing.
CANCEL_BAND = 1e-4


# ---- F32 ulp helpers ----------------------------------------------------------------------------
def ulp_diff_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ULP distance between two F32 arrays, sign-monotone (the bf16 comparator's trick at F32 width).

    The `uint32` trap AGENTS.md records for the BF16 comparator applies here in a milder form: comparing
    `astype(np.int32)` directly gets the sign wrong. The standard monotone map is: non-negative floats
    sort with their bit pattern; negative ones are reflected. Written with the map explicit rather than
    via `np.where(bits < 0, ...)` on a signed view, which is the dead-branch bug in the index.
    """
    def as_bits(x):
        return np.frombuffer(np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                             dtype=np.uint32).astype(np.int64)

    def mono(bits):
        # +0.0 and -0.0 both land on 0x80000000 (distance 0, as they must be) and the map increases with
        # the VALUE on both sides of zero: -1.0 sits 0x3F800000 steps below it, +1.0 the same above.
        # The reflection must be `0x80000000 - |bits|`; `0xFFFFFFFF - |bits|` (the first draft here) puts
        # -0.0 one step ABOVE +0.0 and fails the comparator's own selftest.
        return np.where((bits & 0x80000000) != 0,
                        0x80000000 - (bits & 0x7FFFFFFF),
                        0x80000000 + bits)

    return np.abs(mono(as_bits(a)) - mono(as_bits(b))).astype(np.int64)


def f32_bits(x: float) -> str:
    return format(struct.unpack("<I", struct.pack("<f", float(x)))[0], "08x")


# ---- the three spellings of one recurrence ------------------------------------------------------
def shipped_norms(k_row: np.ndarray, q_row: np.ndarray):
    """The shipped pair reduction, association verbatim, via norm_tree_cpu's operational simulation.

    `block_sum_pair_f32`'s caller feeds it `kq*kq` and `qq*qq` with lane j owning element j at
    `blockDim.x = D = 128`, so the lists below are exactly the per-lane inputs the kernel hands it.
    """
    sk, sq = nt.block_sum_pair_f32([F32(x * x) for x in k_row], [F32(x * x) for x in q_row],
                                   block_threads=D)
    return sk, sq


def ref_walk(q, k, v, g, beta, S0, partials=1):
    """The shipped walk, one head at a time, in float32 scalar loops, `i` ascending.

    `partials=1` is the shipped serial accumulator. `partials=4` keeps the same ALGEBRA and only re-groups
    the sum — the floor probe's "a legal reorder" side. Both read and write the state in place, in the
    shipped order (update row i, then accumulate `acc += qn[i] * S_new[i]`).
    """
    n, nh, d = q.shape
    S = np.array(S0, dtype=F32, copy=True)
    o = np.zeros((n, nh, d), dtype=F32)
    for t in range(n):
        for h in range(nh):
            sk, sq = shipped_norms(k[t, h], q[t, h])
            kn = k[t, h] * rsqrt_f32(sk + EPS)
            qn = q[t, h] * (rsqrt_f32(sq + EPS) * Q_SCALE)
            gam = F32(np.exp(g[t, h]))
            bet = F32(beta[t, h])
            Sh = S[h]                                   # [d][d], row i, column j
            r = np.zeros(d, dtype=F32)
            if partials == 1:
                for i in range(d):
                    r += F32(Sh[i] * kn[i])             # per-column serial chain, i ascending
            else:
                acc_p = np.zeros((partials, d), dtype=F32)
                for i in range(d):
                    acc_p[i % partials] += F32(Sh[i] * kn[i])
                for p in range(1, partials):
                    acc_p[0] += acc_p[p]
                r = acc_p[0]
            dv = F32(bet * (v[t, h] - gam * r))
            acc = np.zeros(d, dtype=F32)
            for i in range(d):
                Sh[i] = F32(gam * Sh[i] + kn[i] * dv)
                acc += F32(qn[i] * Sh[i])
            o[t, h] = acc
    return o, S


def butterfly(vals, lanes=LANES):
    """`v += __shfl_xor_sync(FULL, v, o)` for o = 16,8,4,2,1: every lane ends with the whole sum.

    XOR-flavoured on purpose: a down-shuffle broadcast needs a second pass, and the kernel uses the xor
    form (it must land in all lanes). Simulated lane-by-lane so the association is a fact of this program,
    not of numpy's vectorised add.
    """
    assert len(vals) == lanes
    cur = list(vals)
    off = lanes // 2
    while off >= 1:
        cur = [F32(cur[i] + cur[i ^ off]) for i in range(lanes)]
        off //= 2
    return cur[0]


def rsqrt_f32(x):
    """F32-width `rsqrtf` stand-in: 1/sqrt(x) at float32 width, no float64 round trip.

    The device intrinsic is NOT correctly rounded, so this is not the device value; BOTH sides of every
    comparison here use it, so the intrinsic error cancels and the numbers measure association only.
    """
    return F32(np.float32(1.0)) / np.sqrt(F32(x))


def wf_walk(q, k, v, g, beta, S0, cols=4):
    """`rec_wf_kernel`'s geometry, simulated: 32 lanes per head, `cols` state columns each.

    The norm is reduced from a fixed 4-elements-per-lane split of the row (32 lanes x 4 = 128), which is
    what the kernel does whatever `cols` is, because the row is 128 wide and a lane loads it 4 at a time.
    The state ownership is `cols` consecutive columns per lane, and with `cols < 4` the head is covered by
    more than one block, each of which redundantly computes the same norm — the kernel does that too, and
    the mirror charges it here by reducing once per block.
    """
    n, nh, d = q.shape
    S = np.array(S0, dtype=F32, copy=True)
    o = np.zeros((n, nh, d), dtype=F32)
    groups = d // (LANES * cols)                       # blocks per head in the kernel's grid
    assert groups * LANES * cols == d
    lane = np.arange(LANES)
    off = np.arange(cols)
    for t in range(n):
        for h in range(nh):
            # The row split the kernel uses for the norm: 4 consecutive elements per lane, 32 lanes.
            kk4 = k[t, h].reshape(LANES, 4)
            qq4 = q[t, h].reshape(LANES, 4)
            sk = butterfly([F32(sum(F32(kk4[l, c] * kk4[l, c]) for c in range(4))) for l in range(LANES)])
            sq = butterfly([F32(sum(F32(qq4[l, c] * qq4[l, c]) for c in range(4))) for l in range(LANES)])
            kpq = butterfly([F32(sum(F32(kk4[l, c] * qq4[l, c]) for c in range(4))) for l in range(LANES)])
            kap = rsqrt_f32(sk + EPS)                      # kappa
            iot = rsqrt_f32(sq + EPS)                      # iota (Q_SCALE is applied last, see the header)
            qsc = F32(Q_SCALE * iot)
            gam = F32(np.exp(g[t, h]))
            bet = F32(beta[t, h])
            # kf carries kappa and kpq ONLY: the iota is already inside qsc, and putting it in both is a
            # second multiplication by 1/||q|| (the mirror caught that: cos(o) fell to 0.666 before the fix).
            kf = F32(kap * kpq)
            Sh = S[h]
            for seg in range(groups):
                js = (seg * LANES * cols + lane * cols)[:, None] + off[None, :]   # [32, cols]
                Sv = Sh[:, js]                             # the OLD state, [d, 32, cols]
                vown = v[t, h][js]
                rq = np.zeros((LANES, cols), dtype=F32)
                tq = np.zeros((LANES, cols), dtype=F32)
                for i in range(d):                         # walk A: one state read, two dots
                    rq += Sv[i] * F32(k[t, h][i])
                    tq += Sv[i] * F32(q[t, h][i])
                dv = bet * (vown - gam * (kap * rq))       # dv_j, in the shipped shape (r_j = kap*rq_j)
                o[t, h][js] = (qsc * (gam * tq + kf * dv)).astype(F32)
                for i in range(d):                         # walk B: the state only
                    Sh[i][js] = (gam * Sv[i] + F32(k[t, h][i] * kap) * dv).astype(F32)
    return o, S


# ---- metrics ------------------------------------------------------------------------------------
def metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    cos = float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-300))
    denom = np.abs(b)
    meanrel = float(np.mean(np.abs(a - b) / np.where(denom > 0, denom, 1.0)))
    live = denom >= CANCEL_BAND * float(np.abs(b).max())
    rel_live = np.abs(a - b)[live] / denom[live] if live.any() else np.array([0.0])
    ul = ulp_diff_f32(np.asarray(got, F32), np.asarray(ref, F32))
    return dict(cos=cos, meanrel=meanrel, maxrel_live=float(rel_live.max()),
                ndiff=int((np.asarray(got, F32) != np.asarray(ref, F32)).sum()),
                maxulp=int(ul.max()), p999ulp=int(np.percentile(ul, 99.9)),
                n_live=int(live.sum()), total=int(ref.size))


def report(label, got, ref, width=64):
    m = metrics(got, ref)
    print(f"    {label:{width}s} cos={m['cos']:.9f} meanrel={m['meanrel']:.3e} "
          f"maxrel(band)={m['maxrel_live']:.3e} ndiff={m['ndiff']}/{m['total']} "
          f"maxULP={m['maxulp']} p99.9ULP={m['p999ulp']}")
    return m


# ---- cases --------------------------------------------------------------------------------------
def load_case(fixture: Path, layer: str):
    """One layer's real operands, straight out of the pinned dumps, plus the zero initial state.

    `state_new` and `o_pre_norm` are the PINNED outputs of the same walk over the same 46 tokens from the
    zero state (§04.4's prefill start), which is what makes this a comparison against the reference rather
    than against a model of it.
    """
    def ld(name):
        return np.load(str(fixture / f"{layer}.gdn.{name}.npy"))

    q = np.ascontiguousarray(ld("q_post_conv")[0])
    k = np.ascontiguousarray(ld("k_post_conv")[0])
    v = np.ascontiguousarray(ld("v_post_conv")[0])
    g = np.ascontiguousarray(ld("g_decay_log")[0])
    beta = np.ascontiguousarray(ld("beta")[0])
    pin_state = np.ascontiguousarray(ld("state_new")[0])
    pin_o = np.ascontiguousarray(ld("o_pre_norm")[0])
    return dict(q=q, k=k, v=v, g=g, beta=beta, pin_state=pin_state, pin_o=pin_o,
                S0=np.zeros((NH, D, D), dtype=F32), label=f"{fixture.name}/{layer}")


def random_case(rng, tokens):
    """A second case with different magnitudes, so nothing rides on the fixture's luck.

    `g` is drawn in the shipped range (a decay log, so negative-ish and up to ~9 in the pinned data) and
    `beta` in (0,1), because the cancellation structure of `dv = beta*(v - gam*r)` depends on those.
    """
    q = (rng.standard_normal((tokens, NH, D)) * 3.0).astype(F32)
    k = (rng.standard_normal((tokens, NH, D)) * 2.0).astype(F32)
    v = (rng.standard_normal((tokens, NH, D)) * 1.5).astype(F32)
    g = (rng.random((tokens, NH)) * 6.0 - 5.0).astype(F32)
    beta = (rng.random((tokens, NH)) * 0.5 + 0.4).astype(F32)
    return dict(q=q, k=k, v=v, g=g, beta=beta, pin_state=None, pin_o=None,
                S0=np.zeros((NH, D, D), dtype=F32), label=f"random/{tokens}tok")


def run_case(case, tokens=None, quick=False):
    q, k, v, g, beta, S0 = case["q"], case["k"], case["v"], case["g"], case["beta"], case["S0"]
    n = q.shape[0] if tokens is None else tokens
    sub = dict(q=q[:n], k=k[:n], v=v[:n], g=g[:n], beta=beta[:n], S0=S0)
    print(f"  case {case['label']}: {n} tokens x {NH} heads x {D}")

    # (0) the mirror's own right to adjudicate: its shipped-side spelling must reproduce the pins.
    #     Skipped when there are no pins (the random case), and reported when there are — a mirror that
    #     cannot reproduce the reference cannot be the witness for a change to it.
    o_ref, s_ref = ref_walk(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], sub["S0"])
    if case["pin_o"] is not None and not quick:
        report("mirror(shipped) vs PINNED o", o_ref, case["pin_o"][:n])
        report("mirror(shipped) vs PINNED state", s_ref, case["pin_state"])

    # (1) the FLOOR: same algebra, only the accumulator re-grouped. Any rewrite pays this much.
    if not quick:
        o_fl, s_fl = ref_walk(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], sub["S0"], partials=4)
        report("floor: reorder-only, o", o_fl, o_ref)
        report("floor: reorder-only, state", s_fl, s_ref)

    # (2) the geometry: 4 columns per lane (BET.md §10's shape), 2 per lane (the split leg).
    out = {}
    for cols in (4, 2):
        o_g, s_g = wf_walk(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], sub["S0"], cols=cols)
        out[cols] = (o_g, s_g)
        report(f"geom cols/lane={cols}: o   vs shipped-mirror", o_g, o_ref)
        report(f"geom cols/lane={cols}: state vs shipped-mirror", s_g, s_ref)
        if case["pin_o"] is not None:
            report(f"geom cols/lane={cols}: o   vs PINNED", o_g, case["pin_o"][:n])
            report(f"geom cols/lane={cols}: state vs PINNED", s_g, case["pin_state"])
    return o_ref, s_ref, out


def in_band(m):
    return m["cos"] >= COS_FLOOR and m["meanrel"] <= MEANREL_CEIL


def ulp_selftest():
    ok = True
    one = np.array([1.0], dtype=F32)
    up = np.array([np.nextafter(np.float32(1.0), np.float32(2.0))], dtype=F32)
    neg = np.array([-1.0], dtype=F32)
    negup = np.array([np.nextafter(np.float32(-1.0), np.float32(-2.0))], dtype=F32)
    zero = np.array([0.0], dtype=F32)
    mzero = np.array([np.float32(-0.0)], dtype=F32)
    big = np.array([np.float32(1.19e38)], dtype=F32)
    cases = [("1.0 vs next-up is 1 ULP", int(ulp_diff_f32(one, up)[0]), 1),
             ("-1.0 vs next-down is 1 ULP", int(ulp_diff_f32(neg, negup)[0]), 1),
             ("+0 vs -0 is 0 ULP", int(ulp_diff_f32(zero, mzero)[0]), 0),
             ("+1 vs -1 is the full span", int(ulp_diff_f32(one, neg)[0]), 2 * 0x3F800000)]
    for name, got, want in cases:
        ok = ok and got == want
        print(f"  {'ok  ' if got == want else 'FAIL'} {name}: got {got} want {want}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="data/refs/p64")
    ap.add_argument("--layers", default="L00,L01")
    ap.add_argument("--tokens", type=int, default=0, help="cap the token count (default: all)")
    ap.add_argument("--random", type=int, default=2, help="random cases, N tokens each (0 to skip)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true",
                    help="geometry legs only: skip the pin-comparison and floor legs (same verdict, less print)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print("SELFTEST g2_mirror")
        ok = ulp_selftest()
        # Teeth on the metrics themselves: a metric that cannot fail cannot adjudicate the band.
        rng = np.random.default_rng(1)
        base = (rng.standard_normal(512)).astype(F32)
        same = metrics(base, base.copy())
        nudged = metrics((base * np.float32(1.0 + 1e-3)).astype(F32), base)
        wrecked = metrics((rng.standard_normal(512)).astype(F32), base)
        for name, cond in [("identical arrays: cos 1.0, ndiff 0",
                            same["cos"] > 1.0 - 1e-12 and same["ndiff"] == 0),
                           ("a 0.1 % scale move is inside the band (cos)", in_band(nudged)),
                           ("noise is OUTSIDE the band", not in_band(wrecked))]:
            ok = ok and cond
            print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        # The geometry must not be silently identical to the reference: if it were, every number below
        # would be vacuous (a mirror that reproduces the reference by accident is the empty-check class).
        r = random_case(rng, 2)
        o_ref, s_ref, out = run_case(r, quick=True)
        cond = out[4][1] is not None and metrics(out[4][1], s_ref)["ndiff"] > 0
        ok = ok and cond
        print(f"  {'ok  ' if cond else 'FAIL'} the 4/lane geometry MOVES the state (the check has something to measure)")
        print("RESULT " + ("g2_mirror SELFTEST PASS" if ok else "g2_mirror SELFTEST FAIL"))
        return 0 if ok else 1

    bad = []
    for layer in [x for x in a.layers.split(",") if x]:
        fx = Path(a.fixture)
        if not (fx / f"{layer}.gdn.q_post_conv.npy").exists():
            print(f"SKIP {fx}/{layer}: no GDN dumps")
            continue
        o_ref, s_ref, out = run_case(load_case(fx, layer), tokens=(a.tokens or None), quick=a.quick)
        for cols, (o_g, s_g) in out.items():
            for kind, got, ref in (("o", o_g, o_ref), ("state", s_g, s_ref)):
                m = metrics(got, ref)
                if not in_band(m):
                    bad.append(f"{layer} cols={cols} {kind}: cos={m['cos']:.6f} meanrel={m['meanrel']:.3e}")
    if a.random:
        rng = np.random.default_rng(a.seed)
        for i in range(a.random):
            case = random_case(rng, a.random)
            o_ref, s_ref, out = run_case(case, quick=a.quick)
            for cols, (o_g, s_g) in out.items():
                for kind, got, ref in (("o", o_g, o_ref), ("state", s_g, s_ref)):
                    m = metrics(got, ref)
                    if not in_band(m):
                        bad.append(f"random#{i} cols={cols} {kind}: cos={m['cos']:.6f} meanrel={m['meanrel']:.3e}")
    print()
    if bad:
        print("G2 MIRROR FAIL — outside the pre-registered band (cos >= 0.9995, meanrel <= 5 %):")
        for b in bad:
            print("   " + b)
        return 1
    print("G2 MIRROR PASS — every case inside the existing §10.1 band; no band was widened here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
