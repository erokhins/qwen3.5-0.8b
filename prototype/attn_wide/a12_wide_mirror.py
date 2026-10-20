#!/usr/bin/env python3
"""A12 numerics: what does the WIDE round geometry cost — 64 KV positions a round with an F16 accumulator?

This is `prototype/attn_percost/a10_pvf16_mirror.py`'s question one step on. That mirror asked what changing the
accumulator's FORMAT costs at the shipping round width; A12 changes the FORMAT *and* the round width together,
and the round width is the more invasive of the two: a 64-position round takes ONE max over 64 columns where the
shipping form takes four maxima over 16, so the running max, the rescale factor, the denominator and the order
of the P·V adds all move. `A10-REWRITE-BRIEF.md` R8 calls this out in advance: "a 64-lane stage re-associates the
softmax merge → not bit-identical", judged by the existing §10.1 aggregates, never by a new band.

It is NOT a fourth reference. It imports `prototype/attn_decode/floor_probe.py` — this tree's CPU mirror of §05,
the instrument §10.1's F1 floor was taken with — and reuses its `rows()`, `scores_in_order()` and `bf16_ulp`, and
floor_probe's pinned score matrix is computed ONCE per case and fed to every arm. What it adds over the A10
mirror is (a) a third arm at the wide geometry, (b) a POPULATION derived from the filesystem instead of two
hand-named fixtures, and (c) the final divide happening after the accumulator is widened to F32, which is what
the kernel does (`s_out[...] = wide_h_get(...) / l`); `a10_pvf16_mirror.py` divides in F16, and the difference is
one rounding on the way out, named here rather than left to a reader who notices the two scripts disagree.

THE THREE ARMS, and what each one is the shipping form of:

    f32@16   the kernel that ships today: F32 accumulator, FATTN_MLANES = 16 positions a round
    f16@16   the inert knob `--attn-fattn-pvf16 1`: F16 accumulator, same 16-position round
    f16@64   A12's wide block: F16 accumulator, WIDE_STAGE = 64 positions a round

Everything else is one shared spelling: the score matrix is floor_probe's pinned `matmul` form; the running max,
the per-round rescale, `exp`, the F16 rounding of P, the F32 denominator `l` and the final divide are the same
expressions in all three. A delta is only attributable if it is the only thing that moved.

THE POPULATION IS DERIVED, not typed: `data/refs/*/` with a `manifest.json`, filtered to the ones that actually
carry §05 attention pins, and every (step, layer) or (layer, row) inside those is enumerated. AGENTS.md has the
admission trap in prose — three gate admissions once claimed "nine canonical fixtures" and seven of the names did
not exist — and `a10_pvf16_mirror.py`'s `dec8`+`p300` list is hand-carried, so this script does not inherit it.
The count of fixtures that were skipped, and why, is printed: a population that silently came up small is the same
failure class as a parser that matched nothing.

WHAT IT CANNOT SEE, stated rather than hidden: inside a round the matrix unit sums its products into the D block
in an order it does expose nowhere, so `round` models the round's partial as one F32 dot rounded to F16 once at
the add (the OPTIMISTIC end) and `step` adds them sequentially in F16 (the PESSIMISTIC end); the device lies
between. The pinned fixtures top out at ~300 lanes, which is FIVE rounds of 64 — the long-sweep behaviour the
chain section cares about is reached by cycling rounds at fixed addend scale, labelled as such. This script takes
no device time and certifies no device run; it bounds the format change.

    .venv/bin/python prototype/attn_wide/a12_wide_mirror.py [--max-prefill-rows 2] [--quiet-skips]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEC = REPO / "prototype" / "attn_decode"
REFS = REPO / "data" / "refs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FP = _load("a12_floor_probe", DEC / "floor_probe.py")
R = FP.R
bf16_ulp = FP.bf16_ulp
NO_BAND = 1e-12                       # floor_probe's own no-excuse band, so floors are comparable

STAGE_SHIPPING = 16                   # FATTN_MLANES: KV lanes per round in the shipping form
STAGE_WIDE = 64                       # WIDE_STAGE in prototype/attn_wide/attn_wide.h
F16_MAX = 65504.0                     # fp16's largest finite value: overflow, not precision
F16_MIN_NORMAL = 5.96e-08             # flush-to-zero boundary (the fp16 range census in AGENTS.md)
PROBE_CHAIN = 256                     # issue_rate_probe §2b: where F16 spellings read back 0.5x
CHAIN_PROBE_MAX = 8192

# The files `R.decode_case` / `R.prefill_cases` read, so a fixture can be admitted by checking that its pin set
# is COMPLETE rather than by hoping the loader does not raise. Partial fixtures exist in `data/refs/`.
DECODE_WANT = ["cache_keys", "cache_values", "q_proj.out", "k_proj.out", "v_proj.out",
               "attn.q_rope", "attn.k_rope", "attn.v", "attn.attn_out_pre_gate", "o_proj.out"]
PREFILL_WANT = ["q_proj.out", "k_proj.out", "v_proj.out", "attn.q_rope", "attn.k_rope", "attn.v",
                "attn.attn_out_pre_gate"]


def population(max_prefill_rows: int, verbose: bool):
    """Every §05 attention case the tree actually has pins for, enumerated from the filesystem."""
    out, skips = [], []
    for d in sorted(p for p in REFS.iterdir() if p.is_dir()):
        if not (d / "manifest.json").is_file():
            skips.append((d.name, "no manifest.json"))
            continue
        names = {f.name for f in d.iterdir()}
        n = 0
        # decode-shaped cases: <step>.L<layer>.<stage>.npy, both axes taken from the filenames
        steps, layers = set(), set()
        for f in names:
            m = re.fullmatch(r"(decode\d+)\.L(\d\d)\.attn\.q_rope\.npy", f)
            if m:
                steps.add(m.group(1))
                layers.add(int(m.group(2)))
        for step in sorted(steps):
            for layer in sorted(layers):
                if all(("%s.L%02d.%s.npy" % (step, layer, w)) in names for w in DECODE_WANT):
                    out.append(("%s/%s/L%02d" % (d.name, step, layer),
                                R.decode_case(d, step, layer)))
                    n += 1
        # prefill-shaped cases: L<layer>.<stage>.npy plus prefill.L<layer>.cache_*.npy
        players = set()
        for f in names:
            m = re.fullmatch(r"L(\d\d)\.attn\.q_rope\.npy", f)
            if m and all(("prefill.L%s.cache_%s.npy" % (m.group(1), w)) in names for w in ("keys", "values")):
                players.add(int(m.group(1)))
        for layer in sorted(players):
            seq = int(np.load(d / ("L%02d.attn.q_rope.npy" % layer)).shape[1])
            # The row list is bounded by the CACHE as well as by the sequence: `prefill_cases` copies lanes 0..p
            # of the pinned cache into a slack buffer, so a row past the pinned cache's own length is not a
            # fixture — it is a size error out of torch. Some dumps carry a shorter cache than q_rope (p8's
            # q_rope has more positions than its pinned prefill cache), and the derived row list must respect
            # that rather than assume the two axes agree.
            kc = np.load(d / ("prefill.L%02d.cache_keys.npy" % layer))
            lanes = kc.shape[1] if kc.ndim == 3 else kc.size // (R.N_KV * R.DH)
            seq = min(seq, int(lanes))
            rows = sorted({max(0, seq - 1), max(0, seq // 2)})[-max_prefill_rows:]
            for case in R.prefill_cases(d, layer, rows):
                out.append(("%s/row%d/L%02d" % (d.name, case["pos"], layer), case))
                n += 1
        if n == 0:
            skips.append((d.name, "manifest present but no complete §05 attention pin set"))
        elif verbose:
            print("  fixture %-18s %3d cases" % (d.name, n))
    return out, skips


def rounds_of(V, scores, lanes):
    """The per-round (P, rescale, V-rows) sequence and the denominator, built once and fed to every arm.

    `lanes` is the round width — the one thing the wide arm changes. P is `exp(s - m)` rounded to F16, which the
    shipping kernel already does (`attn_fattn_kernels.h`, the mask/fold/exp/P block), so P's format is not what
    this moves. The last round is short when n % lanes != 0, which is why each round carries its OWN V rows: a
    round is a `lanes`-wide C tile and the two must stay the same width or the partial is a lie. `l` stays F32 in
    every arm; the accumulator change must not move the denominator's format.
    """
    out, l, m = [], 0.0, np.float32(-1e30)
    for lo in range(0, len(scores), lanes):
        s = np.float32(scores[lo:lo + lanes])
        mn = np.maximum(m, s.max())
        r = np.float32(np.exp(m - mn)) if m > -1e29 else np.float32(0.0)
        e = np.exp(s - mn).astype(np.float32)
        l = float(l * r + e.sum(dtype=np.float32))
        out.append((e.astype(np.float16), r, V[lo:lo + lanes]))
        m = mn
    return out, l


def acc_f32(rl):
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float32)
    for P, r, Vc in rl:
        acc = (acc * r).astype(np.float32)
        acc = (acc + (P @ Vc).astype(np.float32)).astype(np.float32)
    return acc


def acc_f16(rl):
    """The optimistic F16 end: the round's products sum in F32 (hardware-internal), ONE rounding at the add."""
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float16)
    for (P, r, Vc) in rl:
        partial = (P @ Vc).astype(np.float32).astype(np.float16)
        acc = ((acc * np.float16(r)).astype(np.float16) + partial).astype(np.float16)
    return acc


def acc_f16_step(rl):
    """The pessimistic F16 end: each product rounds and adds in F16 — `lanes` sequential adds per round."""
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float16)
    for (P, r, Vc) in rl:
        acc = (acc * np.float16(r)).astype(np.float16)
        for j in range(P.shape[0]):
            if P[j] == 0.0:
                continue
            acc = (acc + (P[j] * Vc[j]).astype(np.float16)).astype(np.float16)
    return acc


def chain_no_fold(rl, limit=CHAIN_PROBE_MAX):
    """An F16 accumulator with NO per-round fold: (first round a drop appears, first round MOST drops, first
    round the accumulator leaves fp16's finite range).

    Rounds are CYCLED, because the pinned fixtures are 1..~5 wide rounds and the bound being compared against is
    hundreds of adds long. Cycling holds the addend scale fixed and is stated as an extrapolation everywhere the
    number appears — the same caveat `A10-PVF16.md` §4 attaches to its own version of this probe.

    The third number is the one that ends the argument: an accumulator that is never rescaled does not merely lose
    low bits, it walks out of fp16's 65,504 ceiling, which is a range failure and not a precision one. The
    overflow is therefore silenced locally and REPORTED, not left as a numpy warning that reads like a bug.
    """
    acc = np.zeros(rl[0][2].shape[1], dtype=np.float16)
    first_drop, most_drop, first_inf = None, None, None
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(limit):
            P, _r, Vc = rl[k % len(rl)]
            partial = (P @ Vc).astype(np.float32).astype(np.float16)
            nxt = (acc + partial).astype(np.float16)
            if first_inf is None and not np.all(np.isfinite(nxt)):
                first_inf = k
            dropped = int(np.sum((nxt == acc) & (partial != 0)))
            if dropped and first_drop is None:
                first_drop = k
            if dropped * 2 > acc.shape[0] and most_drop is None:
                most_drop = k
                break
            acc = nxt
    return first_drop, most_drop, first_inf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-prefill-rows", type=int, default=2,
                    help="rows per prefill fixture, taken as the last (longest) and middle positions")
    ap.add_argument("--quiet-skips", action="store_true")
    ap.add_argument("--no-pessimistic", action="store_true",
                    help="skip the sequential-F16-add model (the slow one) on the wide arm")
    a = ap.parse_args()
    if bf16_ulp.selftest():          # the module's convention: 0 = the comparator itself is sound
        raise SystemExit("bf16_ulp selftest failed — every number below would be meaningless")

    print("0. THE POPULATION, derived from %s (every directory with a manifest.json that carries a complete "
          "§05 attention pin set)" % REFS)
    cases, skips = population(a.max_prefill_rows, not a.quiet_skips)
    for name, why in skips:
        print("  SKIP %-18s %s" % (name, why))
    if not cases:
        raise SystemExit("REFUSAL: zero cases — a population that came up empty cannot adjudicate anything")
    kinds = {}
    for _, c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    lanes_all = [c["pos"] + 1 for _, c in cases]
    print("  %d cases (%s); lanes %d..%d, which is %d..%d rounds at 16 and %d..%d rounds at 64; %d heads x %d dims"
          % (len(cases), ", ".join("%s=%d" % kv for kv in sorted(kinds.items())),
             min(lanes_all), max(lanes_all),
             min(lanes_all) // STAGE_SHIPPING + 1, max(lanes_all) // STAGE_SHIPPING + 1,
             min(lanes_all) // STAGE_WIDE + 1, max(lanes_all) // STAGE_WIDE + 1,
             R.N_Q, R.DH))

    ARMS = ("f32@16", "f16@16", "f16@64")
    pinrep = {k: [] for k in ARMS}
    pair = {("f16@64", "f32@16"): [], ("f16@64", "f16@16"): [], ("f16@16", "f32@16"): []}
    arg = {k: [0, 0] for k in ARMS}          # vs the f32@16 arm
    arg_pin = {k: [0, 0] for k in ARMS}      # vs the pin
    n_elem = 0
    worst = []
    peak_abs, n_over, n_tiny = 0.0, 0, 0
    pess = []
    chains = []
    round_hist = {"f32@16": [], "f16@16": [], "f16@64": []}

    for ci, (name, case) in enumerate(cases):
        qr, kr, vv, gate, K, V, n = FP.rows(case)
        sc, _, _ = FP.scores_in_order(qr, K, n, "matmul")     # ONE score matrix, every arm
        ref = case["ref"]["attn_pre_gate"].reshape(R.N_Q, R.DH)
        for i in range(R.N_Q):
            j = i // (R.N_Q // R.N_KV)
            Vv = V[j, :n].float().numpy().astype(np.float32)
            rl16, l16 = rounds_of(Vv, sc[i].float().numpy(), STAGE_SHIPPING)
            rl64, l64 = rounds_of(Vv, sc[i].float().numpy(), STAGE_WIDE)
            g = {"f32@16": None, "f16@16": None, "f16@64": None}
            g["f32@16"] = torch.tensor((acc_f32(rl16) / np.float32(l16)).astype(np.float32),
                                       dtype=torch.bfloat16)
            for key, rl, lv in (("f16@16", rl16, l16), ("f16@64", rl64, l64)):
                a16 = acc_f16(rl).astype(np.float32) / np.float32(lv)     # widen, THEN divide (kernel order)
                g[key] = torch.tensor(a16.astype(np.float32), dtype=torch.bfloat16)
                round_hist[key].append(len(rl))
            round_hist["f32@16"].append(len(rl16))
            n_elem += R.DH
            for key in ARMS:
                pinrep[key].append(bf16_ulp.report(g[key], ref[i], cancel_rel=NO_BAND))
                rp = int(g[key].argmax()) != int(ref[i].argmax())
                arg_pin[key][0] += rp
                arg_pin[key][1] += 1
            for (ka, kb), lst in pair.items():
                lst.append(bf16_ulp.report(g[ka], g[kb], cancel_rel=NO_BAND))
                arg[ka][0] += int(g[ka].argmax()) != int(g[kb].argmax())
                arg[ka][1] += 1
            rep = pair[("f16@64", "f32@16")][-1]
            if rep["worst_off_rel"] > 1e-2:
                x, y = g["f16@64"].float().numpy(), g["f32@16"].float().numpy()
                k = int(np.argmax(np.abs(x - y)))
                worst.append((float(rep["worst_off_rel"]), name, i, k,
                              float(np.abs(x).max()), float(abs(x[k])), float(abs(y[k]))))
            wide_out = g["f16@64"].float().numpy()
            peak_abs = max(peak_abs, float(np.abs(wide_out).max()))
            n_over += int(np.sum(np.abs(wide_out) > F16_MAX))
            n_tiny += int(np.sum((np.abs(wide_out) < F16_MIN_NORMAL) & (wide_out != 0)))
            if i == 0:                              # one head-row per case: the slow models
                if not a.no_pessimistic and n > STAGE_SHIPPING:
                    # One head-row per case, and only where the round GEOMETRY differs: at or below 16 lanes both
                    # arms are one round and the comparison would be two identical numbers side by side.
                    p16 = acc_f16_step(rl16).astype(np.float32) / np.float32(l16)
                    p64 = acc_f16_step(rl64).astype(np.float32) / np.float32(l64)
                    t16 = torch.tensor(p16.astype(np.float32), dtype=torch.bfloat16)
                    t64 = torch.tensor(p64.astype(np.float32), dtype=torch.bfloat16)
                    pess.append((bf16_ulp.report(t16, g["f32@16"], cancel_rel=NO_BAND),
                                 bf16_ulp.report(t64, g["f32@16"], cancel_rel=NO_BAND),
                                 bool(torch.equal(t16, t64)),
                                 float(np.abs(p16 - p64).max())))
                for lanes, rl in ((STAGE_SHIPPING, rl16), (STAGE_WIDE, rl64)):
                    chains.append((lanes, len(rl)) + chain_no_fold(rl))

    print("\n1. THE BAND QUESTION — the wide form against the forms that exist")
    for (ka, kb), lst in pair.items():
        print("   %-8s vs %-8s  cos >= %.8f  meanrel <= %.2e  worst %d ULP  outside a no-excuse band: %d"
              % (ka, kb, min(r["cos"] for r in lst), max(r["meanrel"] for r in lst),
                 max(r["maxulp"] for r in lst), sum(r["n_bad"] for r in lst)))
    print("   rounds per head-row: f32@16 median %d (max %d), f16@64 median %d (max %d)"
          % (sorted(round_hist["f32@16"])[len(round_hist["f32@16"]) // 2], max(round_hist["f32@16"]),
             sorted(round_hist["f16@64"])[len(round_hist["f16@64"]) // 2], max(round_hist["f16@64"])))
    print("   each arm against the PIN (this is what the bands actually judge; the F32 arm is what the bands were")
    print("   calibrated on, so the question is whether the wide arm's distance is of the same order):")
    for key in ARMS:
        print("     %-8s cos >= %.8f  meanrel <= %.2e  worst %d ULP"
              % (key, min(r["cos"] for r in pinrep[key]), max(r["meanrel"] for r in pinrep[key]),
                 max(r["maxulp"] for r in pinrep[key])))
    if pess:
        excluded = len(cases) * R.N_Q - len(pess)          # one head-row per case was modelled; see above
        print("   pessimistic model (every product of the round rounds and adds in F16, the device's worst end),")
        print("   one head-row per case, over the %d head-rows with more than %d lanes; %d excluded because at or"
              % (len(pess), STAGE_SHIPPING, excluded))
        print("   below that length one round is ONE round in both geometries and the arms cannot be distinguished:")
        for which, label in ((0, "16 lanes a round"), (1, "64 lanes a round")):
            print("     %-18s cos >= %.8f  meanrel <= %.2e"
                  % (label, min(r[which]["cos"] for r in pess), max(r[which]["meanrel"] for r in pess)))
        agree = sum(1 for r in pess if r[2])
        print("     the two pessimistic arms are bit-identical on %d of %d rows and elsewhere differ by at most "
              "%.4g, so this model %s the round geometries" %
              (agree, len(pess), max(r[3] for r in pess),
               "DOES separate" if agree * 2 < len(pess) else "does NOT separate"))
        print("     — which is the honest reading of a min-cos that prints the same number twice: the two arms")
        print("       were never the same arithmetic, but the worst row of the population is one they share.")

    print("\n2. THE DECISION QUESTION — argmax over DH, per head-row")
    for key in ARMS:
        if key == "f32@16":
            print("   %-8s is the reference arm              against the PIN: %d/%d"
                  % (key, arg_pin[key][0], arg_pin[key][1]))
            continue
        print("   %-8s vs the f32@16 arm: %d/%d rows move    against the PIN: %d/%d"
              % (key, arg[key][0], arg[key][1], arg_pin[key][0], arg_pin[key][1]))

    print("\n3. THE RANGE QUESTION — fp16 has a finite range, and leaving it is not a precision event")
    print("   largest |out| in the wide arm: %.6g (fp16 max 65504): %d values above it" % (peak_abs, n_over))
    print("   nonzero values below the flush-to-zero boundary (~5.96e-08): %d" % n_tiny)
    print("   %d output elements compared over %d head-rows" % (n_elem, len(cases) * R.N_Q))
    if worst:
        print("   where the relative movement concentrates (a small |out| beside a large row peak is cancellation,")
        print("   and that is where a ULP-sized error is a large RELATIVE error — AGENTS.md's cancellation row):")
        for w in sorted(worst, reverse=True)[:10]:
            print("     off_rel %8.4f  %s head %d dim %3d  |out| %9.4g -> %9.4g (row peak %.4g)"
                  % (w[0], w[1], w[2], w[3], w[5], w[6], w[4]))
    else:
        print("   no head-row exceeded off_rel 1e-2, so there is no cancellation row to name.")

    print("\n4. THE CHAIN QUESTION — the per-round rescale is what makes the F16 accumulator legal, at BOTH")
    print("   round widths (against issue_rate_probe §2b: F16 integers are exact to 2^11 and every spelling read")
    print("   back half the linear value past ~%d adds at the probe's addend scale)" % PROBE_CHAIN)
    fds = {"f16@16": [], "f16@64": []}
    mds = {"f16@16": [], "f16@64": []}
    ifs = {"f16@16": [], "f16@64": []}
    rounds = {"f16@16": [], "f16@64": []}
    for lanes, nr, fd, md, fi in chains:
        key = "f16@64" if lanes == STAGE_WIDE else "f16@16"
        rounds[key].append(nr)
        if fd is not None:
            fds[key].append(fd)
        if md is not None:
            mds[key].append(md)
        if fi is not None:
            ifs[key].append(fi)

    def q(xs, f):
        return f(xs) if xs else "never"
    print("   WITHOUT the fold (the counterfactual), rounds until the first dropped add / until most adds drop /")
    print("   until the accumulator is no longer FINITE in fp16, rounds cycled to %d to reach a length the" % CHAIN_PROBE_MAX)
    print("   fixtures do not contain:")
    for key in ("f16@16", "f16@64"):
        print("     %-8s %d..%d rounds of %d   first drop: %s   most drop: %s   leaves fp16 range: %s"
              % (key, min(rounds[key]) if rounds[key] else 0,
                 max(rounds[key]) if rounds[key] else 0, 16 if key == "f16@16" else 64,
                 "%s (median %s)" % (q(fds[key], min), q(fds[key], lambda x: sorted(x)[len(x) // 2]))
                 if fds[key] else "never",
                 q(mds[key], min),
                 "%s (median %s)" % (q(ifs[key], min), q(ifs[key], lambda x: sorted(x)[len(x) // 2]))
                 if ifs[key] else "never, within %d rounds" % CHAIN_PROBE_MAX))
    print("   The wide arm reaches the same sweep length in a quarter of the rounds, so each of ITS rounds carries")
    print("   four times the addends into one accumulator step — which is the direction that makes the rescale")
    print("   MORE load-bearing, not less, and it is why `attn_wide.h` keeps the fold exactly where the shipping")
    print("   kernel keeps it: before the round's add, on the accumulator's critical path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
