#!/usr/bin/env python3
"""A10-F16 numerics: what does an F16 P*V ACCUMULATOR cost, on the PINNED data?

The bet (`prototype/attn_percost/A10-F16-IMPL.md` D1–D7) is that halving the WMMA wave's P*V accumulator from
128 F32 registers to 64 F16 ones is admissible under the bands `engine/gate.py` already enforces — no new band,
no widening. This answers that on real tensors before any device run, and it is NOT a fourth reference: it
imports `prototype/attn_decode/floor_probe.py` — this tree's existing CPU mirror of §05, the instrument §10.1's
F1 floor was taken with — and reuses its `rows()`, `scores_in_order()` and `bf16_ulp`, and its case population
(every `dec8` decode step × every attention layer, plus `p300`'s long-cache rows). Cases are enumerated, never
typed (AGENTS' admission trap: seven of nine hand-listed fixture names did not exist).

WHAT IS HELD IDENTICAL, because a delta is only attributable if everything else is spelled once: the score
matrix is floor_probe's pinned `matmul` form, computed ONCE per case and fed to both arms; the running max, the
per-round rescale, the exp, the F16 rounding of P, the denominator `l` and the final divide are the same
expressions in both arms. The ONLY difference is where the P*V partial is accumulated:

    f32 arm — the shipping form:  acc = acc*r + partial, F32 throughout, 128 registers a lane.
    f16 arm — the knob:           acc = f16(f16(acc*f16(r)) + f16(partial)), 64 registers a lane.

The round geometry is the kernel's: FATTN_MLANES = 16 KV lanes a round (`engine/attn_fattn_kernels.h`), so one
round is one 16-wide C tile — which is what makes the chain length in section 4 comparable with the probe's.

WHAT IT CANNOT SEE, stated rather than hidden: inside a round the matrix unit sums 16 products into the D block
in an order it does not expose, and may hold guard bits while doing so. `round` models the round's partial as one
F32 dot rounded to F16 once at the add (the OPTIMISTIC end); `step` adds the 16 products sequentially in F16
(the PESSIMISTIC end). The device lies between them, so both are printed and neither is called the answer. This
script takes no device time and cannot certify a device run; it bounds the format change.

Four questions, four sections: the band question (cos/meanrel/ULP, arm-vs-arm and each arm vs the pin), the
decision question (does argmax over DH move?), the range question (fp16 overflow and flush-to-zero are not
precision questions), and the chain question — the longest chain the F16 accumulator carries at THIS dump's
addend scale, against `prototype/issue_rate_probe/results.md` §2b's measured bound (F16 integers exact to
2^11 = 2048; every F16 spelling read back at exactly half the linear value once the chain passed ~256 adds).

CPU only: no device, no timing, no ms.

    .venv/bin/python prototype/attn_percost/a10_pvf16_mirror.py [--fixture dec8] [--long-fixture p300]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEC = REPO / "prototype" / "attn_decode"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FP = _load("a10_floor_probe", DEC / "floor_probe.py")
R = FP.R                                       # the pinned reference module floor_probe itself imports
bf16_ulp = FP.bf16_ulp
NO_BAND = 1e-12                                # floor_probe's own no-excuse band, so floors are comparable

MLANES = 16                                    # FATTN_MLANES: KV lanes per round = one C tile
F16_MAX = 65504.0                              # fp16's largest finite value: overflow, not precision
F16_MIN_NORMAL = 5.96e-08                      # flush-to-zero boundary (the fp16 range census in AGENTS.md)
PROBE_CHAIN = 256                              # §2b: where every F16 spelling began reading back 0.5x
CHAIN_PROBE_MAX = 8192                         # how far the synthetic extension may run to find a drop


def rounds_of(V, scores):
    """The per-round (P, rescale, V-rows) sequence and the denominator, built ONCE and fed to every arm.

    P is `exp(s - m)` rounded to F16 — both arms, because the shipped kernel already packs P as an F16 operand
    (`attn_fattn_kernels.h`, the mask/fold/exp/P block), so P's format is not what this knob changes. The last
    round is short when `n % 16 != 0`, which is why each round carries its OWN V rows: a round is a 16-wide C
    tile and the two must stay the same width or the partial is a lie. `l` is the denominator the kernel carries
    in F32 in BOTH forms; the accumulator change must not move it.
    """
    out, l, m = [], 0.0, np.float32(-1e30)
    for lo in range(0, len(scores), MLANES):
        s = np.float32(scores[lo:lo + MLANES])
        mn = np.maximum(m, s.max())
        r = np.float32(np.exp(m - mn)) if m > -1e29 else np.float32(0.0)
        e = np.exp(s - mn).astype(np.float32)
        l = float(l * r + e.sum(dtype=np.float32))
        out.append((e.astype(np.float16), r, V[lo:lo + MLANES]))
        m = mn
    return out, l


def acc_f32(rl, _):
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float32)
    for P, r, Vc in rl:
        acc = (acc * r).astype(np.float32)
        acc = (acc + (P @ Vc).astype(np.float32)).astype(np.float32)
    return acc


def acc_f16_round(rl, _):
    """Optimistic F16: the round's products sum in F32 (hardware-internal), ONE rounding at the add."""
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float16)
    for (P, r, Vc) in rl:
        partial = (P @ Vc).astype(np.float32).astype(np.float16)
        acc = ((acc * np.float16(r)).astype(np.float16) + partial).astype(np.float16)
    return acc


def acc_f16_step(rl, _):
    """Pessimistic F16: each product rounds and adds in F16 — 16 sequential adds per round."""
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float16)
    for (P, r, Vc) in rl:
        acc = (acc * np.float16(r)).astype(np.float16)
        for j in range(P.shape[0]):
            if P[j] == 0.0:
                continue
            acc = (acc + (P[j] * Vc[j]).astype(np.float16)).astype(np.float16)
    return acc


def chain_no_fold(rl, limit=CHAIN_PROBE_MAX):
    """The counterfactual: an F16 accumulator with NO per-round fold. Returns (rounds until the first add is
    dropped entirely, rounds until MOST adds are dropped). The rounds are CYCLED — the pinned fixtures are
    8–277 lanes and the bound this is compared against is hundreds of adds long, so the round sequence is
    repeated to reach that length rather than extrapolated to it. That repetition holds the addend scale fixed
    and is stated as such wherever the number appears."""
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float16)
    first_drop, most_drop = None, None
    for k in range(limit):
        P, _r, Vc = rl[k % len(rl)]
        partial = (P @ Vc).astype(np.float32).astype(np.float16)
        nxt = (acc + partial).astype(np.float16)
        dropped = int(np.sum((nxt == acc) & (partial != 0)))
        if dropped and first_drop is None:
            first_drop = k
        if dropped * 2 > acc.shape[0] and most_drop is None:
            most_drop = k
            break
        acc = nxt
    return first_drop, most_drop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--long-fixture", default="p300")
    ap.add_argument("--long-rows", default="63,127,191,255,276")
    ap.add_argument("--no-chain", action="store_true", help="skip section 4 (the synthetic extension is the "
                                                           "slow part)")
    a = ap.parse_args()
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed — every number below would be meaningless")

    d = R.ROOT / "data" / "refs" / a.fixture
    named = [("%s/%s/L%s" % (a.fixture, s, l), c)
             for l in R.ATTN_LAYERS for s in ["decode%d" % i for i in range(8)]
             for c in [R.decode_case(d, s, l)]]
    dl = R.ROOT / "data" / "refs" / a.long_fixture
    nlong = 0
    if dl.exists():
        for rn in [int(x) for x in a.long_rows.split(",")]:
            for c in R.prefill_cases(dl, R.ATTN_LAYERS[0], [rn]):
                named.append(("%s/row%d/L%s" % (a.long_fixture, rn, R.ATTN_LAYERS[0]), c))
                nlong += 1
    print("%d cases: %s (8 decode steps x %d layers) + %s (%d long-cache rows); %d heads x %d dims each"
          % (len(named), a.fixture, len(R.ATTN_LAYERS),
             a.long_fixture if dl.exists() else "absent", nlong, R.N_Q, R.DH))

    n_elem = ndiff = nbad = argmv = argtot = 0
    cos_min, mr_max, ulp_max = 1.0, 0.0, 0
    pin = {"f32": [], "f16": []}
    worst = []
    peak_abs, n_over, n_tiny = 0.0, 0, 0
    step_cos, step_mr = 1.0, 0.0
    drops = {"f16": 0, "f32": 0}
    total_adds = 0
    chains = []
    lanes_seen = []

    for name, case in named:
        qr, kr, vv, gate, K, V, n = FP.rows(case)
        sc, _, _ = FP.scores_in_order(qr, K, n, "matmul")      # ONE score matrix, every arm
        ref = case["ref"]["attn_pre_gate"].reshape(R.N_Q, R.DH)
        lanes_seen.append(n)
        for i in range(R.N_Q):
            j = i // (R.N_Q // R.N_KV)
            Vv = V[j, :n].float().numpy().astype(np.float32)
            rl, l = rounds_of(Vv, sc[i].float().numpy())
            a32 = acc_f32(rl, None) / np.float32(l)
            a16 = acc_f16_round(rl, None) / np.float16(l)
            a16s = acc_f16_step(rl, None) / np.float16(l)
            g32 = torch.tensor(a32.astype(np.float32), dtype=torch.bfloat16)
            g16 = torch.tensor(a16.astype(np.float32), dtype=torch.bfloat16)
            g16s = torch.tensor(a16s.astype(np.float32), dtype=torch.bfloat16)
            rep = bf16_ulp.report(g16, g32, cancel_rel=NO_BAND)
            reps = bf16_ulp.report(g16s, g32, cancel_rel=NO_BAND)
            n_elem += g32.numel()
            ndiff += int((g16 != g32).sum())
            nbad += rep["n_bad"]
            cos_min, mr_max = min(cos_min, rep["cos"]), max(mr_max, rep["meanrel"])
            ulp_max = max(ulp_max, rep["maxulp"])
            step_cos = min(step_cos, reps["cos"])
            step_mr = max(step_mr, reps["meanrel"])
            pin["f32"].append(bf16_ulp.report(g32, ref[i], cancel_rel=NO_BAND))
            pin["f16"].append(bf16_ulp.report(g16, ref[i], cancel_rel=NO_BAND))
            argtot += 1
            if int(g16.argmax()) != int(g32.argmax()):
                argmv += 1
            f32n, f16n = g32.float().numpy(), g16.float().numpy()
            if rep["worst_off_rel"] > 1e-2:
                k = int(np.argmax(np.abs(f32n - f16n)))
                worst.append((float(rep["worst_off_rel"]), name, i, k,
                              float(abs(f32n).max()), float(abs(f32n[k])), float(abs(f16n[k]))))
            f32v = a32.astype(np.float32)
            peak_abs = max(peak_abs, float(np.abs(f32v).max()))
            n_over += int(np.sum(np.abs(f32v) > F16_MAX))
            n_tiny += int(np.sum((np.abs(a16.astype(np.float32)) < F16_MIN_NORMAL)
                                 & (a16.astype(np.float32) != 0)))
            # With the fold in force (what ships): how many of the round's adds fail to change the accumulator
            # at all, counted under the SAME rule for both widths so it is a comparison and not an assertion.
            # A dropped add is not an error -- it is an add below half a ULP of the running value, so below
            # 2^-11 relative for F16 and 2^-24 for F32 -- but the COUNT is what says how much the fold carries.
            for widths, seed in (("f16", np.float16), ("f32", np.float32)):
                accw = np.zeros(Vv.shape[1], dtype=seed)
                for (P, r, Vc) in rl:
                    part = (P @ Vc).astype(np.float32).astype(seed)
                    accw = (accw * seed(r)).astype(seed)
                    nxtw = (accw + part).astype(seed)
                    drops[widths] += int(np.sum((nxtw == accw) & (part != 0)))
                    accw = nxtw
                total_adds += len(rl) * Vv.shape[1]
            if not a.no_chain and i == 0:      # one head-row per case: the chain probe is the slow part
                fd, md = chain_no_fold(rl)
                chains.append((len(rl), fd, md))

    print("\n1. THE BAND QUESTION — F16 accumulator vs F32 accumulator (same scores, same P, same l)")
    print("   over %d output elements: %d differ (%.3f%%), %d outside a no-excuse band, worst %d ULP,"
          "\n   cos >= %.8f, meanrel <= %.2e" % (n_elem, ndiff, 100.0 * ndiff / n_elem, nbad, ulp_max,
                                                 cos_min, mr_max))
    print("   pessimistic model (16 sequential F16 adds a round, the device's worst case): cos >= %.8f,"
          " meanrel <= %.2e" % (step_cos, step_mr))
    for tag in ("f32", "f16"):
        print("   the %s arm against the PIN: cos >= %.8f, meanrel <= %.2e, worst %d ULP"
              % (tag, min(r["cos"] for r in pin[tag]), max(r["meanrel"] for r in pin[tag]),
                 max(r["maxulp"] for r in pin[tag])))
    print("   read the last two lines together: the F16 arm is admissible if its distance to the pin is not"
          "\n   materially worse than the F32 arm's, because the F32 arm is what the bands were calibrated on.")

    print("\n2. THE DECISION QUESTION — argmax over DH, per head-row")
    print("   %d/%d head-rows move their argmax when the accumulator becomes F16" % (argmv, argtot))

    print("\n3. THE RANGE QUESTION — fp16 has a finite range, and leaving it is not a precision event")
    print("   largest |acc/l| over the population: %.6g (fp16 max 65504): %d values above it"
          % (peak_abs, n_over))
    print("   nonzero values below the flush-to-zero boundary (~5.96e-08): %d" % n_tiny)

    if worst:
        print("\n   where the relative movement concentrates (worst off_rel first; a small |out| next to a large")
        print("   row peak is cancellation, and that is where a ULP-sized error is a large RELATIVE error):")
        for w in sorted(worst, reverse=True)[:10]:
            print("     off_rel %8.4f  %s head %d dim %3d  |out| %9.4g -> %9.4g (row peak %.4g)"
                  % (w[0], w[1], w[2], w[3], w[5], w[6], w[4]))
    else:
        print("\n   no head-row exceeded off_rel 1e-2, so there is no cancellation row to name.")

    print("\n4. THE CHAIN QUESTION — against issue_rate_probe/results.md §2b (F16 exact to 2^11; reads back")
    print("   half the linear value past ~%d adds at the probe's addend scale)" % PROBE_CHAIN)
    print("   lane counts this population actually exercises: %d..%d lanes = %d..%d rounds of %d"
          % (min(lanes_seen), max(lanes_seen), min(lanes_seen) // MLANES + 1,
             max(lanes_seen) // MLANES + 1, MLANES))
    print("   WITH the per-round fold in force (what ships), adds that never change the accumulator, counted"
          "\n   under one rule for both widths, out of %d adds:" % total_adds)
    print("     F16 accumulator: %5d (%.3f %%)    F32 accumulator: %5d (%.3f %%)"
          % (drops["f16"], 100.0 * drops["f16"] / total_adds,
             drops["f32"], 100.0 * drops["f32"] / total_adds))
    print("   A dropped add is an add below half a ULP of the running value (2^-11 relative for F16, 2^-24 for")
    print("   F32), so this is the fold's residual rather than an error budget; whether the F16 form's larger")
    print("   residual matters is section 1's question, not this one's.")
    if not a.no_chain and chains:
        fds = [c[1] for c in chains if c[1] is not None]
        mds = [c[2] for c in chains if c[2] is not None]
        print("   WITHOUT the fold (the counterfactual that makes the fold mandatory), rounds until the first")
        print("   dropped add, over %d head-rows (one per case), rounds cycled to %d to reach the length:"
              % (len(chains), CHAIN_PROBE_MAX))
        print("     first add dropped: min %s, median %s, max %s"
              % (min(fds) if fds else "never",
                 sorted(fds)[len(fds) // 2] if fds else "never",
                 max(fds) if fds else "never"))
        print("     most adds dropped:   min %s, median %s, max %s"
              % (min(mds) if mds else "never",
                 sorted(mds)[len(mds) // 2] if mds else "never",
                 max(mds) if mds else "never"))
        print("   The cycling repeats the round sequence, so this bounds the chain at the dump's addend scale")
        print("   rather than measuring a real 8k-lane row; §2b's ~%d is the same kind of bound." % PROBE_CHAIN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
