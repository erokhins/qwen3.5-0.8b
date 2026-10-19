#!/usr/bin/env python3
"""g4_mirror.py — CPU witness for bet G4: the GDN recurrence walked by llama's thread map.

WHAT IT ADJUDICATES. `engine/gdn_rec_reg16_kernels.h::rec_reg16_kernel` re-spells §04.4's recurrence with
128 blocks x 128 threads, 8 lanes per state column and **16 state elements per thread held in registers
across the span**. Exactly two things move from the shipped bits, and this file prices both, then shows
that nothing else did:

  1. the ROW NORMS' association. Shipped: `block_sum_pair_f32` at `blockDim.x` = 128 — a 5-level per-warp
     fold, an LDS hop to four partials, a second 5-level fold. Here: a 16-element thread-local partial
     folded in 3 xor levels across the 8 lanes of a column group (`gdn_bfly8_sum_f32`, order 4,2,1, which
     is llama's `warp_reduce_sum<8>`). Different tree, therefore a different F32 sum.
  2. the STATE DOTS' association. `r_j` and `o_j` are one serial 128-term chain per column shipped; here
     each lane folds its own 16 consecutive rows and the 8 lanes butterfly together.

What is deliberately NOT different from the shipped form, and the mirror is written to keep it that way:
the per-element products and the two-walk structure. `k̂_i = k_i·rsqrtf(sk+eps)`,
`q̂_i = (q_i·rsqrtf(sq+eps))·Q_SCALE`, `dv_j = β·(v_j − γ·r_j)`, walk B `S = γ·S + k̂·dv` then
`acc += q̂·S_new` with i ascending — the shipped algebra, character for character. G2 needed a closed-form
rewrite of `o` because it had no way to broadcast `q̂`; this map does not, so its move is smaller than G2's
by construction and that is checkable here rather than claimed.

Band (BET.md §10 constraint 2, unchanged): the EXISTING §10.1 aggregates — cos >= 0.9995, meanrel <= 5 % —
plus the cancellation band on the max-relative figure. Not tunable from a flag: a mirror that lets you
widen the band from its own command line is not a gate. `ndiff` and `maxULP` are REPORTED and not gated,
because for a chained dot the legal ndiff is 29–47 % (§10.1 F1–F4) and an unband-limited max-rel is
dominated by elements that cancelled to ~1e-6 of the max.

WHAT IT CANNOT SEE (stated so the handover cannot overclaim). It is a CPU model: no compiler (contraction,
scheduling, whether the k/q slices really vectorise), no `rsqrtf`/`expf` device intrinsics (both sides use
this file's `rsqrt_f32`, so intrinsic error CANCELS here and the numbers measure association only), and no
device. The device witness is `prototype/gdn_walk/gate_g4.py`.

THE THIRD CHECK IS THE ONE A NUMERICS MIRROR CANNOT MAKE. A re-associated walk that reads the WRONG
elements also produces plausible numbers, which is §20.1 of `prototype/attn_decode/results.md`: two
self-consistent layouts joined by a branch, every per-form check green, half of every streaming row
silently another group's answer. So `--geom` enumerates this kernel's map at every (blockIdx, threadIdx)
and asserts the elements it touches are a permutation of the 262,144-element state, that each column's 8
lanes partition the column exactly, and that no two blocks overlap. `--selftest` re-derives it with one
perturbation and requires the witness to go red, so it is a check and not a printout.

Usage:
    .venv/bin/python prototype/gdn_walk/g4_mirror.py [--fixture data/refs/p64] [--layers L00,L01]
    .venv/bin/python prototype/gdn_walk/g4_mirror.py --random 2 --seed 7
    .venv/bin/python prototype/gdn_walk/g4_mirror.py --geom        # the map witness only
    .venv/bin/python prototype/gdn_walk/g4_mirror.py --selftest    # the mirror's own teeth
Exit 0 = every case inside the pre-registered band and the map a bijection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "gdn_walk"))
import g2_mirror as g2          # noqa: E402  the shipped walk, the metrics, the ulp comparator, the floor

D, NH = g2.D, g2.NH
Q_SCALE, EPS, F32 = g2.Q_SCALE, g2.EPS, g2.F32
E = 16                       # state elements per thread — must equal kGdnR16Elems
LANES_PER_COL = D // E       # 8 — must equal kGdnR16LanesPerCol
COLS_PER_WARP = 32 // LANES_PER_COL
COLS_PER_BLOCK = COLS_PER_WARP * 4
GRID = (NH, D // COLS_PER_BLOCK)
assert E * LANES_PER_COL == D and GRID == (16, 8), \
    "g4_mirror's constants drifted from engine/gdn_rec_reg16_kernels.h — fix the mirror, not the assert"


def butterfly_vec(vals, lanes=LANES_PER_COL):
    """`g2_mirror.butterfly`'s pairing applied to a list of arrays instead of scalars.

    Same offsets (lanes//2 down to 1, xor), same order, so the association is the same tree; the scalar
    form is asserted equal to this one in `--selftest` rather than trusted.
    """
    cur = list(vals)
    off = lanes // 2
    while off >= 1:
        cur = [cur[i] + cur[i ^ off] for i in range(lanes)]
        off //= 2
    return cur


# ---- the map --------------------------------------------------------------------------------------
def reg16_map():
    """Every (h, i, j) this kernel touches, in the order its own index arithmetic produces.

    Mirrors `rec_reg16_kernel`: `h = blockIdx.x`, `lane = threadIdx.x`, `warp = threadIdx.y`,
    `col_in_warp = lane >> 3`, `lane_in_col = lane & 7`, `j = blockIdx.y*COLS_PER_BLOCK +
    warp*COLS_PER_WARP + col_in_warp`, `i = lane_in_col*E + r`.
    """
    for bx in range(GRID[0]):
        for by in range(GRID[1]):
            for warp in range(4):
                for lane in range(32):
                    col_in_warp = lane >> 3
                    lane_in_col = lane & 7
                    j = by * COLS_PER_BLOCK + warp * COLS_PER_WARP + col_in_warp
                    i0 = lane_in_col * E
                    for r in range(E):
                        yield (bx, i0 + r, j)


def reg16_map_of(lane_of, j_of):
    """The same enumeration with injectable index arithmetic — what makes the witness falsifiable."""
    for bx in range(GRID[0]):
        for by in range(GRID[1]):
            for warp in range(4):
                for lane in range(32):
                    j = j_of(by, warp, lane)
                    for r in range(E):
                        yield (bx, lane_of(lane) * E + r, j)


def geom_witness() -> list:
    """(lines, ok): the map is a bijection over the state, and each column's lanes partition the column."""
    bad = []
    triples = list(reg16_map())
    want = NH * D * D
    if len(triples) != want:
        bad.append(f"the map yields {len(triples)} elements, not {want}")
    if len(set(triples)) != want:
        dupes = len(triples) - len(set(triples))
        bad.append(f"{dupes} DUPLICATE (h,i,j) triples: two owners for one element — this is §20.1's"
                   " aliasing trap, two self-consistent layouts writing the same slot")
    heads = sorted({h for h, _i, _j in triples})
    if heads != list(range(NH)):
        bad.append(f"heads touched = {heads}, not all {NH}")
    # Per head: the (i, j) set must be the full 128x128 with no overlap.
    for h in heads:
        cells = {(i, j) for hh, i, j in triples if hh == h}
        if len(cells) != D * D:
            bad.append(f"head {h}: {len(cells)} distinct cells, want {D * D}")
    # Column partition: the 8 lanes that own one column must split its 128 rows exactly once.
    for by in range(GRID[1]):
        for warp in range(4):
            for col_in_warp in range(COLS_PER_WARP):
                j = by * COLS_PER_BLOCK + warp * COLS_PER_WARP + col_in_warp
                rows = []
                for lane_in_col in range(LANES_PER_COL):
                    rows += list(range(lane_in_col * E, lane_in_col * E + E))
                if sorted(rows) != list(range(D)):
                    bad.append(f"column {j}: the {LANES_PER_COL} lanes cover {len(set(rows))} distinct"
                               f" rows of {D}, so a norm over those lanes is not a sum over the column")
    lines = [f"map: grid {GRID} x block (32,4) x {E} elements = {len(triples)} of {NH * D * D};"
             f" distinct {len(set(triples))}; {LANES_PER_COL} lanes/column, {COLS_PER_BLOCK} columns/block"]
    return lines, not bad, bad


# ---- the geometry ---------------------------------------------------------------------------------
def reg16_walk(q, k, v, g, beta, S0):
    """`rec_reg16_kernel` simulated, association for association, one head at a time.

    The shard lives in `S` in place, which is the same thing as living in registers for a span: no other
    agent touches these columns during the span, and the span here is the whole token range.
    """
    n, nh, d = q.shape
    S = np.array(S0, dtype=F32, copy=True)
    o = np.zeros((n, nh, d), dtype=F32)
    for t in range(n):
        for h in range(nh):
            # The row split the 8 lanes of a column group hold: lane p owns rows p*E .. p*E+E-1.
            k8 = k[t, h].reshape(LANES_PER_COL, E)
            q8 = q[t, h].reshape(LANES_PER_COL, E)
            sk = butterfly_vec([sum(F32(k8[p, r] * k8[p, r]) for r in range(E))
                                for p in range(LANES_PER_COL)])[0]
            sq = butterfly_vec([sum(F32(q8[p, r] * q8[p, r]) for r in range(E))
                                for p in range(LANES_PER_COL)])[0]
            rs_k = g2.rsqrt_f32(sk + EPS)
            rs_q = g2.rsqrt_f32(sq + EPS)
            # Normalised in place, like the kernel does to keep its live set at three arrays.
            kh = np.empty((LANES_PER_COL, E), dtype=F32)
            qh = np.empty((LANES_PER_COL, E), dtype=F32)
            for p in range(LANES_PER_COL):
                for r in range(E):
                    kh[p, r] = k8[p, r] * rs_k
                    qh[p, r] = (q8[p, r] * rs_q) * Q_SCALE
            gam = F32(np.exp(g[t, h]))
            bet = F32(beta[t, h])
            S4 = S[h].reshape(LANES_PER_COL, E, d)      # [p][r][j], a view: writes land in S
            # walk A: per-lane partial over its own 16 rows, r ascending, then the 8-lane butterfly.
            part = [np.zeros(d, dtype=F32) for _ in range(LANES_PER_COL)]
            for r in range(E):
                for p in range(LANES_PER_COL):
                    part[p] += S4[p, r, :] * kh[p, r]
            rj = butterfly_vec(part)[0]
            dv = F32(bet * (v[t, h] - gam * rj))         # dv_j, shipped shape
            # walk B: update + output dot, per lane over its own rows, then the butterfly.
            acc = [np.zeros(d, dtype=F32) for _ in range(LANES_PER_COL)]
            for r in range(E):
                for p in range(LANES_PER_COL):
                    s0 = F32(gam * S4[p, r, :])
                    s0 = F32(s0 + kh[p, r] * dv)
                    S4[p, r, :] = s0
                    acc[p] = acc[p] + qh[p, r] * s0
            o[t, h] = butterfly_vec(acc)[0]
    return o, S


def report(label, got, ref, width=64):
    m = g2.metrics(got, ref)
    print(f"    {label:{width}s} cos={m['cos']:.9f} meanrel={m['meanrel']:.3e} "
          f"maxrel(band)={m['maxrel_live']:.3e} ndiff={m['ndiff']}/{m['total']} "
          f"maxULP={m['maxulp']} p99.9ULP={m['p999ulp']}")
    return m


def run_case(case, tokens=None, quick=False):
    q, k, v, g, beta, S0 = case["q"], case["k"], case["v"], case["g"], case["beta"], case["S0"]
    n = q.shape[0] if tokens is None else tokens
    sub = (q[:n], k[:n], v[:n], g[:n], beta[:n], S0)
    print(f"  case {case['label']}: {n} tokens x {NH} heads x {D}")
    o_ref, s_ref = g2.ref_walk(*sub)
    out = {}
    if case["pin_o"] is not None and not quick:
        report("mirror(shipped) vs PINNED o", o_ref, case["pin_o"][:n])
        report("mirror(shipped) vs PINNED state", s_ref, case["pin_state"])
    if not quick:
        o_fl, s_fl = g2.ref_walk(*sub, partials=4)
        report("floor: reorder-only, o", o_fl, o_ref)
        report("floor: reorder-only, state", s_fl, s_ref)
    o_g, s_g = reg16_walk(*sub)
    out = ("reg16", (o_g, s_g))
    report("geom reg16: o     vs shipped-mirror", o_g, o_ref)
    report("geom reg16: state vs shipped-mirror", s_g, s_ref)
    if case["pin_o"] is not None:
        report("geom reg16: o     vs PINNED", o_g, case["pin_o"][:n])
        report("geom reg16: state vs PINNED", s_g, case["pin_state"])
    return o_ref, s_ref, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="data/refs/p64")
    ap.add_argument("--layers", default="L00,L01")
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="skip the pin and floor legs")
    ap.add_argument("--geom", action="store_true", help="the map witness only, no numerics")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print("SELFTEST g4_mirror")
        ok = g2.ulp_selftest()
        rng = np.random.default_rng(1)
        base = (rng.standard_normal(512)).astype(F32)
        same = g2.metrics(base, base.copy())
        wrecked = g2.metrics((rng.standard_normal(512)).astype(F32), base)
        for name, cond in [("identical arrays: cos 1.0, ndiff 0",
                            same["cos"] > 1.0 - 1e-12 and same["ndiff"] == 0),
                           ("noise is OUTSIDE the band", not g2.in_band(wrecked))]:
            ok = ok and cond
            print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        # The vectorised butterfly must be the scalar one g2_mirror already certified.
        vals = [F32(x) for x in rng.standard_normal(LANES_PER_COL)]
        want = g2.butterfly(vals, lanes=LANES_PER_COL)
        got = butterfly_vec(vals)[0]
        cond = float(got) == float(want)
        ok = ok and cond
        print(f"  {'ok  ' if cond else 'FAIL'} butterfly_vec == g2_mirror.butterfly at width"
              f" {LANES_PER_COL} (one tree, spelled twice)")
        # The geometry must MOVE the state, or every number below measures nothing.
        case = g2.random_case(rng, 2)
        o_ref, s_ref, (_name, (o_g, s_g)) = run_case(case, quick=True)
        cond = g2.metrics(s_g, s_ref)["ndiff"] > 0
        ok = ok and cond
        print(f"  {'ok  ' if cond else 'FAIL'} reg16 MOVES the state (the check has something to measure)")
        # MUST-CATCH for the geometry witness: alias two lanes onto the same row block. One perturbation
        # of the index arithmetic must turn the witness red, or it is a printout, not a check.
        _l, ok_plain, _b = geom_witness()
        alias = list(reg16_map_of(lane_of=lambda lane: (lane & 7) >> 1,
                                  j_of=lambda by, warp, lane: by * COLS_PER_BLOCK + warp * COLS_PER_WARP
                                  + (lane >> 3)))
        caught_alias = len(set(alias)) != NH * D * D
        ok = ok and ok_plain and caught_alias
        print(f"  {'ok  ' if caught_alias else 'FAIL'} aliasing two lanes is caught"
              f" ({len(set(alias))} distinct of {NH * D * D})")
        off = list(reg16_map_of(lane_of=lambda lane: lane & 7,
                                j_of=lambda by, warp, lane: by * COLS_PER_BLOCK + warp * COLS_PER_WARP))
        caught_off = len(set(off)) != NH * D * D
        ok = ok and caught_off
        print(f"  {'ok  ' if caught_off else 'FAIL'} dropping col_in_warp is caught"
              f" ({len(set(off))} distinct of {NH * D * D})")
        print("RESULT " + ("g4_mirror SELFTEST PASS" if ok else "g4_mirror SELFTEST FAIL"))
        return 0 if ok else 1

    lines, ok_geom, bad_geom = geom_witness()
    for line in lines:
        print(line)
    for item in bad_geom:
        print("  GEOM FAIL " + item)
    if a.geom:
        print("GEOM " + ("PASS" if ok_geom else "FAIL"))
        return 0 if ok_geom else 1

    bad = []
    for layer in [x for x in a.layers.split(",") if x]:
        fx = Path(a.fixture)
        if not (fx / f"{layer}.gdn.q_post_conv.npy").exists():
            print(f"SKIP {fx}/{layer}: no GDN dumps")
            continue
        case = g2.load_case(fx, layer)
        n = a.tokens or case["q"].shape[0]
        o_ref, s_ref, (_n, (o_g, s_g)) = run_case(case, tokens=(a.tokens or None), quick=a.quick)
        legs = [("vs shipped-mirror", o_g, o_ref), ("vs shipped-mirror", s_g, s_ref)]
        if case["pin_o"] is not None:
            legs += [("vs PINNED", o_g, case["pin_o"][:n]), ("vs PINNED", s_g, case["pin_state"])]
        for kind, got, ref in legs:
            m = g2.metrics(got, ref)
            if not g2.in_band(m):
                bad.append(f"{layer} reg16 {kind} {id(got)}: cos={m['cos']:.6f}"
                           f" meanrel={m['meanrel']:.3e}")
    if a.random:
        rng = np.random.default_rng(a.seed)
        for i in range(a.random):
            case = g2.random_case(rng, a.random)
            o_ref, s_ref, (_n, (o_g, s_g)) = run_case(case, quick=a.quick)
            for kind, got, ref in (("o", o_g, o_ref), ("state", s_g, s_ref)):
                m = g2.metrics(got, ref)
                if not g2.in_band(m):
                    bad.append(f"random#{i} reg16 {kind}: cos={m['cos']:.6f} meanrel={m['meanrel']:.3e}")
    print()
    if not ok_geom:
        bad.append("the map is not a bijection over the state — see GEOM FAIL above")
    if bad:
        print("G4 MIRROR FAIL — outside the pre-registered band (cos >= 0.9995, meanrel <= 5 %):")
        for b in bad:
            print("   " + b)
        return 1
    print("G4 MIRROR PASS — every case inside the existing §10.1 band, and the map is a bijection;"
          " no band was widened here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
