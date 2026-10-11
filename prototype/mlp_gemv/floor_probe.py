#!/usr/bin/env python3
"""Where is the floor? Reorder the *reference's own* math and see how far BF16 can move.

`run.py` gates the MLP kernel in BF16 ULPs, so it needs a threshold that is not a guess. The
kernel's only license is "same products, F32 accumulation, different summation order", so the
floor is: how far can the reference's own result move when only the summation order changes?

For every layer and every decode step this takes the pinned operands and recomputes each dot
in four orders — torch's bf16 matmul (the reference), numpy F32, a blocked-pairwise F32 sum,
and `math.fsum` (exactly rounded, so it is the truth the others approximate) — and reports the
worst BF16 ULP distance between any two of them. Two things fall out:

  * the distribution of distances, which sizes the per-output threshold;
  * how much of the gate's cancellation band those offenders actually use, and the tightest band
    these measurements would still pass at (a band nobody measures is a band nobody calibrated);
  * whether a large distance ever lands on an element whose magnitude is a tiny fraction of
    the tensor max. It does: a near-zero dot is a cancellation, its absolute F32 error is
    ordinary but its BF16 ULP distance explodes. That is why the gate in run.py is
    scale-aware (ULP on the big elements, `|Δ| ≤ tol × max|ref|` on the small ones) instead
    of a flat ULP bound that a correct kernel can trip by being unlucky.

    .venv/bin/python prototype/mlp_gemv/floor_probe.py [--fixture dec8] [--layers all]
                                                     [--perturb K] [--trials N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402


def _gate():
    """The F4 aggregate bands live in `run.py`, because that is what gates them. This probe is
    what SIZES them, so it reads them from there rather than keeping a second copy that can drift
    into agreeing with nothing."""
    import importlib.util
    p = Path(__file__).resolve().parent / "run.py"
    spec = importlib.util.spec_from_file_location("mlp_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# The band belongs to the gate, not to this probe: reading it from run.py means the probe can
# never report margin against a number the gate no longer uses. Measured here and re-reported by
# run.py: the reference's own reordered sums put their worst >1 ULP element at |ref|/max 1.1e-6
# (gate_pre) and 6.1e-6 (out), and the kernel's own comparisons at 8.2e-6, which is what the
# gate's 3e-5 is sized against (>= 3.6x of margin over both).
CANCEL_BAND = _gate().CANCEL_REL


def ulp_dist(a, b) -> np.ndarray:
    return bf16_ulp.ulp_dist(a, b)


def rel_of(x: np.ndarray) -> np.ndarray:
    """|x| / max|x| — the scale the band is written against (bf16_ulp.band_usage takes it)."""
    scale = float(np.abs(x).max())
    return np.abs(x) / scale if scale else np.zeros(np.shape(x))


def blocked_sum(A: np.ndarray, x: np.ndarray, block: int = 128) -> np.ndarray:
    """Sum in blocks of `block`, then pairwise across the block sums: a different, legitimate
    F32 order (roughly what a warp-shuffle reduction does). Two ways to get this wrong, both
    found in the first version of this function: broadcasting a (rows,) block sum onto an
    (rows, nblocks) accumulator adds it to every column and returns the dot times a power of
    two; and folding an odd number of columns silently drops one (in=3584 gives 28 blocks,
    then 14, 7, and a naive half-fold throws one away). Pad to a power of two and fill
    columns explicitly."""
    n = x.shape[0]
    assert n % block == 0, ("blocked_sum would silently drop a K remainder: n=%d is not a "
                            "multiple of block=%d, and the dropped tail is exactly the "
                            "wrong-reduction-length bug this probe is meant to measure"
                            % (n, block))
    nblocks = n // block
    acc = np.empty((A.shape[0], nblocks), dtype=np.float32)
    for b in range(nblocks):
        s = b * block
        acc[:, b] = (A[:, s:s + block] * x[None, s:s + block]).sum(axis=1, dtype=np.float32)
    width = 1 << (nblocks - 1).bit_length()          # zero pad: 28 -> 32
    if width != nblocks:
        acc = np.concatenate([acc, np.zeros((A.shape[0], width - nblocks), np.float32)], 1)
    while acc.shape[1] > 1:
        h = acc.shape[1] // 2
        acc = acc[:, :h] + acc[:, h:2 * h]
    return acc[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--step", default=None)
    ap.add_argument("--perturb", type=int, default=1,
                    help="also measure how far a 1-ULP error in `act` propagates through "
                         "down_proj, with this many injected elements per trial")
    ap.add_argument("--trials", type=int, default=8,
                    help="perturbation trials per layer x step case (fixed: a worst-of-N is "
                         "not a bound until N is printed next to it)")
    args = ap.parse_args()
    if bf16_ulp.selftest():
        raise SystemExit("bf16_ulp selftest failed - the floor cannot be measured")

    from safetensors import safe_open
    CKPT = ROOT / "models" / "Qwen3.5-0.8B"
    key = {}
    for f in sorted(CKPT.glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                key[k] = s.get_tensor(k)

    d = ROOT / "data" / "refs" / args.fixture
    steps = [args.step] if args.step else sorted(
        {p.name.split(".")[0] for p in d.glob("decode*.L00.mlp.in.npy")})
    layers = range(24) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    pre = "model.language_model.layers.%d.mlp."

    hist = {k: {"max": 0, "over1": 0, "over2": 0, "n": 0, "worst_frac_of_max": 1.0,
                "worst_at": None, "per_variant": {}, "n_band": 0, "n_off": 0,
                "n_off_in_band": 0, "worst_off_rel": 0.0} for k in ("gate_pre", "out")}
    for l in layers:
        for step in steps:
            p = "%s.L%02d.mlp." % (step, l)
            if not (d / (p + "in.npy")).exists():
                continue
            refs = {k: np.load(d / (p + k + ".npy")).astype(np.float32).reshape(-1)
                    for k in ("in", "gate_pre", "act", "out")}
            specs = {"gate_pre": (key[pre % l + "gate_proj.weight"], refs["in"]),
                     "out": (key[pre % l + "down_proj.weight"], refs["act"])}
            for nm, (W, x) in specs.items():
                Wf = W.float().numpy()
                ref = torch.from_numpy(refs[nm])
                h = hist[nm]
                variants = [
                    ("f32-einsum", torch.from_numpy(
                        np.einsum("ij,j->i", Wf, x, dtype=np.float32))),
                    ("f32-blocked", torch.from_numpy(blocked_sum(Wf, x))),
                    # The nearest thing to the truth: bf16 x bf16 products are exact in F64 and
                    # 3584 of them sum without a visible F64 rounding error, so this is the
                    # correctly-rounded dot that both torch and the kernel are approximating.
                    ("f64-exact", torch.from_numpy(
                        np.einsum("ij,j->i", W.double(), x.astype(np.float64)).astype(np.float32))),
                ]
                scale = float(np.abs(refs[nm]).max())
                relref = rel_of(refs[nm])
                for vname, v in variants:
                    dd = ulp_dist(v, ref)
                    bs = bf16_ulp.band_usage(relref, dd, 1, CANCEL_BAND)
                    for key_ in ("n_band", "n_off", "n_off_in_band"):
                        h[key_] += bs[key_]
                    h["worst_off_rel"] = max(h["worst_off_rel"], bs["worst_off_rel"])
                    pv = h.setdefault("per_variant", {})
                    pv[vname] = {"max": max(pv.get(vname, {}).get("max", 0), int(dd.max())),
                                 "over1": pv.get(vname, {}).get("over1", 0) + int((dd > 1).sum()),
                                 "n": pv.get(vname, {}).get("n", 0) + dd.size}
                    i = int(dd.argmax())
                    if int(dd.max()) > h["max"]:
                        h["max"] = int(dd.max())
                        h["worst_frac_of_max"] = float(abs(refs[nm][i]) / scale)
                        h["worst_at"] = "L%02d %s %s via %s" % (l, step, nm, vname)
                    h["over1"] += int((dd > 1).sum())
                    h["over2"] += int((dd > 2).sum())
                    h["n"] += dd.size

    print("fixture %s | steps %s | layers %s" % (args.fixture, len(steps), args.layers))
    for nm, h in hist.items():
        print("\n%s: reordering the reference's own F32 sum moves" % nm)
        print("   worst distance      %d ULP (over %d comparisons)" % (h["max"], h["n"]))
        print("   >1 ULP              %d / %d (%.4f%%)   >2 ULP: %d"
              % (h["over1"], h["n"], 100.0 * h["over1"] / h["n"], h["over2"]))
        print("   worst element |ref|/max|ref| = %.2e  (%s)"
              % (h["worst_frac_of_max"], h["worst_at"]))
        for vname, pv in sorted(h.get("per_variant", {}).items()):
            print("      %-12s max %5d ULP | >1 ULP: %d / %d (%.4f%%)"
                  % (vname, pv["max"], pv["over1"], pv["n"], 100.0 * pv["over1"] / pv["n"]))
        print("   band usage (|ref|/max <= %g): %d / %d elements (%.3f%%) sit in the band; "
              "%d of the %d >1-ULP offenders are in it and %d are outside it"
              % (CANCEL_BAND, h["n_band"], h["n"], 100.0 * h["n_band"] / h["n"],
                 h["n_off_in_band"], h["n_off"], h["n_off"] - h["n_off_in_band"]))
        print("   tightest band these measurements pass at: |ref|/max %.1e  -> the gate's %g "
              "carries %.0fx margin over the worst offender"
              % (h["worst_off_rel"], CANCEL_BAND,
                 CANCEL_BAND / h["worst_off_rel"] if h["worst_off_rel"] else float("inf")))
    print("\nRead: a large ULP distance is a cancellation, not an error — it lands where "
          "|ref| is a rounding-critical fraction of the tensor max. A gate that forbids it "
          "outright forbids correctness.")

    if args.perturb:
        # Chain sensitivity: F4 feeds the kernel's own `act` into the down GEMV, so a legal
        # 1-ULP difference in act is not allowed to produce a 1-ULP difference in out. Inject
        # k one-ULP errors into the reference act and measure what down_proj does with them.
        #
        # Fixed trials per case, and N is printed. The first version stopped each case at its
        # first positive trial, so the quoted "worst" was a worst-of-something-between-1-and-4
        # per case and nothing on the page said which.
        print("\nact -> out propagation, %d injected 1-ULP elements per trial, %d trials per "
              "layer x step case:" % (args.perturb, args.trials))
        rng = np.random.default_rng(7)
        worst = {"ulp": 0, "frac": 1.0, "where": None}
        surv = {"cases": 0, "cases_hit": 0, "trials": 0, "elems": 0, "over1": 0,
                "over1_in_band": 0, "worst_off_rel": 0.0,
                "worst_cos": 1.0, "worst_meanrel": 0.0, "worst_ndiff": 0.0}
        for l in layers:
            for step in steps:
                p = "%s.L%02d.mlp." % (step, l)
                if not (d / (p + "act.npy")).exists():
                    continue
                act = torch.from_numpy(np.load(d / (p + "act.npy"))).to(torch.bfloat16)
                out_ref = torch.from_numpy(np.load(d / (p + "out.npy"))).to(torch.bfloat16)
                Wd = key[pre % l + "down_proj.weight"]
                orf = out_ref.float().reshape(-1).numpy().astype(np.float64)
                oscale = float(np.abs(orf).max())
                orel = rel_of(orf)
                surv["cases"] += 1
                case = {"over1": 0, "max": 0, "worst_off_rel": 0.0}
                # F4 is judged on aggregates, so measure exactly those: what the worst trial of
                # this case does to cos / meanrel / ndiff of `out` against the pin. Nothing else
                # in the repo computes them for a perturbed `out`, which is what F4 gates.
                agg = {"cos": 1.0, "meanrel": 0.0, "ndiff": 0.0}
                for _ in range(args.trials):
                    idx = rng.choice(act.numel(), args.perturb, replace=False)
                    flat = act.view(-1).clone()
                    # step each chosen element exactly one representable BF16 up, via the
                    # monotone key (np.nextafter in F32 would change a value that then rounds
                    # back to the same BF16, i.e. inject nothing at all)
                    k = bf16_ulp.monotone_key(bf16_ulp.bf16_bits(flat[idx])) + 1
                    kb = np.asarray(bf16_ulp.bits_from_key(k), dtype=np.uint16)
                    flat[idx] = torch.from_numpy(np.ascontiguousarray(kb.view(np.uint16))).view(
                        torch.bfloat16)
                    got = flat.reshape(act.shape) @ Wd.t()
                    dd = ulp_dist(got, out_ref)
                    gf = got.float().reshape(-1).numpy().astype(np.float64)
                    den = float(np.linalg.norm(gf) * np.linalg.norm(orf))
                    agg["cos"] = min(agg["cos"], float(gf @ orf / den) if den else 1.0)
                    agg["meanrel"] = max(agg["meanrel"], float(np.abs(gf - orf).mean()
                                                               / np.abs(orf).mean()))
                    agg["ndiff"] = max(agg["ndiff"], 100.0 * float((dd != 0).mean()))
                    bs = bf16_ulp.band_usage(orel, dd, 1, CANCEL_BAND)
                    surv["trials"] += 1
                    surv["elems"] += bs["n"]
                    surv["over1"] += bs["n_off"]
                    surv["over1_in_band"] += bs["n_off_in_band"]
                    surv["worst_off_rel"] = max(surv["worst_off_rel"], bs["worst_off_rel"])
                    case["over1"] += bs["n_off"]
                    case["max"] = max(case["max"], int(dd.max()))
                    case["worst_off_rel"] = max(case["worst_off_rel"], bs["worst_off_rel"])
                    i = int(dd.argmax())
                    if int(dd.max()) > worst["ulp"]:
                        worst = {"ulp": int(dd.max()),
                                 "frac": float(abs(orf[i]) / oscale),
                                 "where": "L%02d %s" % (l, step)}
                surv["worst_cos"] = min(surv["worst_cos"], agg["cos"])
                surv["worst_meanrel"] = max(surv["worst_meanrel"], agg["meanrel"])
                surv["worst_ndiff"] = max(surv["worst_ndiff"], agg["ndiff"])
                if case["over1"]:
                    surv["cases_hit"] += 1
                    print("   L%02d %s: %d >1 ULP out elements over %d trials x %d elements "
                          "(max %d ULP, worst offender |ref|/max %.1e) | cos %.8f meanrel "
                          "%.2e ndiff %.2f%%  [%d 1-ULP flips]"
                          % (l, step, case["over1"], args.trials, dd.size, case["max"],
                             case["worst_off_rel"], agg["cos"], agg["meanrel"], agg["ndiff"],
                             args.perturb))
        print("   survey: %d layer x step cases x %d trials = %d perturbed dots, %d elements"
              % (surv["cases"], args.trials, surv["trials"], surv["elems"]))
        print("   %d/%d cases produced at least one >1 ULP element; %d of %d offenders sit "
              "inside the %g band (the only ones the gate may exempt)"
              % (surv["cases_hit"], surv["cases"], surv["over1_in_band"], surv["over1"],
                 CANCEL_BAND))
        print("   worst seen anywhere: %d ULP at |ref|/max = %.1e (%s), worst offender "
              "|ref|/max = %.1e over %d trials/case"
              % (worst["ulp"], worst["frac"], worst["where"], surv["worst_off_rel"],
                 args.trials))
        g = _gate()
        print("   F4 aggregates (cos / meanrel / ndiff of a perturbed `out` against the pin), "
              "worst over the same survey: cos %.8f, meanrel %.2e, ndiff %.2f%%"
              % (surv["worst_cos"], surv["worst_meanrel"], surv["worst_ndiff"]))
        print("   gate bands: cos >= %.4f (%.0fx of slack in 1-cos), meanrel <= %.1e (%.0fx), "
              "ndiff <= %.0f%% (%.1fx)"
              % (g.F4_COS_MIN, (1 - g.F4_COS_MIN) / max(1 - surv["worst_cos"], 1e-16),
                 g.F4_MEANREL_MAX, g.F4_MEANREL_MAX / max(surv["worst_meanrel"], 1e-300),
                 g.F4_NDIFF_PCT_MAX, g.F4_NDIFF_PCT_MAX / max(surv["worst_ndiff"], 1e-300)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
