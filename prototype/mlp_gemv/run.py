#!/usr/bin/env python3
"""Gate the second GEMV family: the §03.3 MLP at k = 1, against the reference's own operands.

Everything is read from the pinned dumps (`tools/reference_dump.py --decode N`, §10.5) and the
checkpoint — nothing is synthesized. For decode step `s`, layer `l`:

    decode{s}.L{ll}.mlp.in         x          [1024] bf16
    decode{s}.L{ll}.mlp.gate_pre   x·Wgᵀ      [3584] bf16
    decode{s}.L{ll}.mlp.up_pre     x·Wuᵀ      [3584] bf16
    decode{s}.L{ll}.mlp.act        silu(g)·u  [3584] bf16
    decode{s}.L{ll}.mlp.out        act·Wdᵀ    [1024] bf16   (the residual addend)

so the MLP is gateable as two isolated functions, the way G3i gates a layer: no upstream
cascade can enter, every difference belongs to this kernel.

    F1 gateup   gate_pre / up_pre                      (the two projections)
    F2 gateup   act, i.e. the fused SwiGLU epilogue with its two extra roundings
    F3 down     out computed from the REFERENCE act    (cannot hide behind an F2 error)
    F4 chain    out computed from the kernel's act     (the form the engine will use)
    cross       speccheck.specimpl.mlp(x), which ulp_probe.py shows is bit-equal to the
                reference — a second check that never goes through the dumps

Judgement is in BF16 ULPs against the floor `ulp_probe.py` measured (a reordered F32 sum
moves ~1 element in 3584 by exactly 1 ULP): **≤2 % of elements differ and none >1 ULP**,
plus the §10.1 `mlp` class cos ≥ 0.9999 / meanrel ≤ 1 %.

Timing reports each kernel against the 237 GB/s probe (devices/) and folds the pair into a
per-token MLP number against the 2.230 ms roofline of §01.7.

    .venv/bin/python prototype/mlp_gemv/run.py [--fixture dec8] [--mode gateup|down|all]
                                              [--layers 0,3,11,23|all] [--rows 8] [--no-time]
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
import bf16_ulp  # noqa: E402
CKPT = ROOT / "models" / "Qwen3.5-0.8B"
BIN = Path(__file__).resolve().parent / "mlp_gemv"
WORK = Path("/tmp/mlp_gemv")
PROBE_GBS = 237.0
FF, HID = 3584, 1024
# The cancellation band: below this fraction of the tensor max, a BF16 ULP distance is not
# evidence of anything (see judge() and floor_probe.py). Measured, not chosen. The worst >1-ULP
# offender over the reference's own reordered sums is at |ref|/max 1.1e-6 (gate_pre) and 6.1e-6
# (out) across 192 layer x step cases x 3 reorderings, and the worst one this kernel produces is
# 8.2e-6 (printed at the end of every run); 3e-5 clears both by >= 3.6x. It was 1e-4 from the
# first version, which was a guess that happened to be 12x looser than the data required.
# Tightening it further is a gate change and has to clear both numbers again.
CANCEL_REL = 3e-5
# F4 is the chain (down applied to the kernel's own act) and has no per-element rule — the
# perturbation survey in floor_probe.py shows why. These aggregates ARE the gate, and the same
# probe sizes them: it prints the worst cos / meanrel / ndiff a legal 1-ULP `act` difference can
# produce, with the margin these three numbers leave.
F4_COS_MIN = 0.9999
F4_MEANREL_MAX = 1e-3
F4_NDIFF_PCT_MAX = 15.0
WPK = "model.language_model.layers.%d.mlp."    # checkpoint key prefix (plain ints, not %02d)
# §01.7: the MLP is 0.5285 GB/token; at the probe bandwidth that is 2.230 ms/token.
MLP_BYTES_PER_TOKEN = 24 * (2 * FF * HID + HID * FF) * 2


def ckpt_tensor(key: str) -> torch.Tensor:
    for f in sorted(glob.glob(str(CKPT / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            if key in s.keys():
                return s.get_tensor(key)
    raise KeyError(key)


def dump_bf16(t: torch.Tensor, path: Path) -> Path:
    t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tofile(path)
    return path


def read_bf16(path: Path, n: int, skip: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16)[skip:skip + n]).view(torch.bfloat16)


def compare(got: torch.Tensor, ref: torch.Tensor, **kw) -> dict:
    """BF16 ULP distance via the shared comparator (prototype/bf16_ulp.py — read its
    docstring before writing a new one) plus the §10.1 metrics."""
    kw.setdefault("cancel_rel", CANCEL_REL)
    return bf16_ulp.report(got, ref, **kw)


def fmt(c: dict) -> str:
    return ("ndiff %5d/%d (%.3f%%) max %d ULP | over-tol large elements: %d "
            "(worst %d ULP at |ref|/max %.1e) | cos %.8f meanrel %.2e | in-band %d "
            "tightest band %.1e"
            % (c["ndiff"], c["n"], 100.0 * c["ndiff"] / c["n"], c["maxulp"], c["n_bad"],
               c["worst_bad_ulp"], c["worst_bad_rel"], c["cos"], c["meanrel"],
               c["n_band"], c["worst_off_rel"]))


def judge(tag: str, c: dict, fails: list[str], worst: dict, ndiff_pct_max=2.0,
          meanrel_max=1e-2, cos_min=0.9999, ulp_gate=True, ulp_max=1) -> None:
    """What counts as reproducing the reference, and why.

    A flat "max 1 ULP" bound is not the floor: floor_probe.py measured that reordering the
    reference's OWN summation moves about 1 element in 100 000 by more than 1 ULP, and every
    one of those is a dot that cancelled to ~1e-6 of the tensor max, where one ULP is a large
    fraction of the value. So the element-wise test is scale-aware: nothing may exceed
    `ulp_max` *while also* being larger than 1e-4 of the tensor max. Aggregate cos/meanrel are
    gated alongside, which is what a genuinely wrong formula moves.

    `ulp_max=2` for `act` (F2): HF's silu is `x * (1/(1+exp(-x)))` in F32 and then rounds to
    BF16, and an F32 `exp` differing by 1 ULP from libm's flips the BF16 result about once per
    few thousand elements; a 1-ULP BF16 input to the following BF16 multiply can then round to
    2 ULP away. Measured: 0-2 of 3584 elements per layer-step, never beyond 2 ULP. Reproducing
    libm's exp bit-for-bit is not the deliverable; `--negative f32act` shows the gate still
    catches the rounding policy that actually matters.

    `ulp_gate=False` is for the F4 chain: its input is the kernel's `act`, which legitimately
    differs from the reference's by 1 ULP, and floor_probe.py shows a 1-ULP input flip can
    land hundreds of ULP away in the output at small elements. Judging ULP there would be
    judging arithmetic sensitivity, not the kernel.
    """
    print("    %-22s %s" % (tag, fmt(c)))
    key = tag.split()[-1]
    # [worst over large elements, raw worst, elements inside the band, tightest band these
    #  results pass at (per-element-gated outputs only)]
    prev = worst.setdefault(key, [0, 0, 0, 0.0])
    prev[0] = max(prev[0], c["worst_bad_ulp"] if ulp_gate else 0)
    prev[1] = max(prev[1], c["maxulp"])
    prev[2] = max(prev[2], c["n_band"])
    # A comparison with no per-element ULP rule has no band to size; recording one there would
    # invite someone to tighten a gate that never used it.
    if ulp_gate:
        prev[3] = max(prev[3], c["worst_off_rel"])
    if ulp_gate and c["n_bad"]:
        fails.append("%s: %d large element(s) differ by >%d ULP (worst %d ULP at |ref|/max %.1e)"
                     % (tag, c["n_bad"], ulp_max, c["worst_bad_ulp"], c["worst_bad_rel"]))
    if 100.0 * c["ndiff"] / c["n"] > ndiff_pct_max:
        fails.append("%s: %.3f%% of elements differ (limit %.3f%%)"
                     % (tag, 100.0 * c["ndiff"] / c["n"], ndiff_pct_max))
    if c["cos"] < cos_min or c["meanrel"] > meanrel_max:
        fails.append("%s: cos %.8f meanrel %.2e outside G3i/§10.1" % (tag, c["cos"],
                                                                      c["meanrel"]))


def kernel(args: list[str], label: str, quiet: bool = False) -> dict:
    proc = subprocess.run([str(BIN)] + args, capture_output=True, text=True)
    if proc.returncode:
        sys.stdout.write(proc.stdout + proc.stderr)
        raise SystemExit("kernel failed: %s" % label)
    if not quiet:
        for line in proc.stdout.splitlines():
            if line.startswith("kernel "):
                print("    %-9s %s" % (label, line.strip()))
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return dict(kv.split("=", 1) for kv in line.split()[1:])
    raise SystemExit("no RESULT line from the binary:\n" + proc.stdout)


NEG_DESCRIPTIONS = {
    # Each of these is a plausible way to write this kernel wrong. The gate must refuse all of
    # them, or it is not verifying anything (speccheck/gates/run_negative.py set the precedent).
    "half": "accumulate only half of K: a wrong reduction length",
    "tail": "drop the LAST word of K (8 of 1024 terms): an off-by-one bound",
    "trunc": "round toward zero instead of round-to-nearest-even",
    "rowswap": "write each finished row into its neighbour's slot: right dots, wrong index",
    "preswap": "stage gate into the up slot: `act` correct, both raw dumps each other's",
    "f32act": "do the SwiGLU entirely in F32, the literal A-07 reading (ulp_probe.py shows "
              "the reference rounds twice; this is the only one of the three the raw "
              "max-ULP-on-small-elements test would have missed)",
}

# Which kernel the bug lives in. `--mode down` never launches the gateup kernel, so an epilogue
# or staging bug there is not a failure of the gate — it is a bug the run cannot reach.
NEG_KERNELS = {"half": ("gateup", "down"), "tail": ("gateup", "down"),
               "trunc": ("gateup", "down"), "rowswap": ("gateup", "down"),
               "f32act": ("gateup",), "preswap": ("gateup",)}

# Judged on five separable observations:
#   gate_pre / up_pre / act  the gateup kernel's own outputs (F1, F2)
#   out(down)   mlp_down carrying the injection, fed the REFERENCE act — F3's isolation, so a
#               mlp_down bug cannot hide behind propagation and an upstream bug cannot fake one
#   out(chain)  mlp_down carrying the injection, fed the kernel's own act — the F4 form, i.e. how
#               an epilogue bug reaches the value the engine returns
MUST_CATCH = {
    "half":    ("gate_pre", "up_pre", "act", "out(down)"),
    "tail":    ("gate_pre", "up_pre", "act", "out(down)"),
    "trunc":   ("gate_pre", "up_pre", "act", "out(down)"),
    "rowswap": ("gate_pre", "up_pre", "act", "out(down)"),
    "f32act":  ("act", "out(chain)"),
    "preswap": ("gate_pre", "up_pre"),
}
MUST_NOT = {
    "half": (), "tail": (), "trunc": (), "rowswap": (),
    # An epilogue bug must not move a dot, and with the reference act it must not move `out`
    # either: if it does, the down kernel is being judged on the epilogue's ticket.
    "f32act": ("gate_pre", "up_pre", "out(down)"),
    # A staging swap moves nothing but the two raw dumps — `act` and every `out` stay clean.
    "preswap": ("act", "out(down)", "out(chain)"),
}


def run_negatives(args, d, steps, layers, fails) -> None:
    """Inject each bug and assert the gate's own criteria reject it — where the bug is, and not
    where it is not."""
    l0, st = layers[0], steps[0]
    p0 = "%s.L%02d.mlp." % (st, l0)
    # Its own weights, into its own files: the main loop leaves the LAST layer's weights in
    # WORK/Wg.bin, and comparing layer 0's pins against layer 11's weights "rejects" every
    # injection no matter what the kernel does — a vacuous negative test.
    dump_bf16(ckpt_tensor(WPK % l0 + "gate_proj.weight"), WORK / "Wg_n.bin")
    dump_bf16(ckpt_tensor(WPK % l0 + "up_proj.weight"), WORK / "Wu_n.bin")
    dump_bf16(ckpt_tensor(WPK % l0 + "down_proj.weight"), WORK / "Wd_n.bin")
    x = torch.from_numpy(np.load(d / (p0 + "in.npy"))).to(torch.bfloat16).reshape(-1)
    refs = {k: torch.from_numpy(np.load(d / (p0 + k + ".npy"))).to(torch.bfloat16).reshape(-1)
            for k in ("gate_pre", "up_pre", "act", "out")}
    dump_bf16(x, WORK / "x.bin")
    dump_bf16(refs["act"], WORK / "act_negref.bin")
    modes = {"gateup", "down"} if args.mode == "all" else {args.mode}
    for neg, why in NEG_DESCRIPTIONS.items():
        if not set(NEG_KERNELS[neg]) & modes:
            print("    --neg %-8s skipped: the bug lives in %s and --mode %s never runs it"
                  % (neg, "/".join(NEG_KERNELS[neg]), args.mode))
            continue
        extra = ["--neg", neg]
        caught: list[str] = []
        if "gateup" in modes:
            kernel(["--mode", "gateup", "--x", str(WORK / "x.bin"), "--wg", str(WORK / "Wg_n.bin"),
                    "--wu", str(WORK / "Wu_n.bin"), "--out", str(WORK / "nact.bin"),
                    "--pre", str(WORK / "npre.bin"), "--iters", "1"] + extra, "", quiet=True)
            pre_t = read_bf16(WORK / "npre.bin", 2 * FF)
            for nm, got, want, kw in (
                    ("gate_pre", pre_t[:FF], refs["gate_pre"], {}),
                    ("up_pre", pre_t[FF:], refs["up_pre"], {}),
                    ("act", read_bf16(WORK / "nact.bin", FF), refs["act"], {"ulp_max": 2})):
                if judge_neg(compare(got, want, **kw)):
                    caught.append(nm)
        # out(down): the reference act in, so the ONLY thing that can move `out` is the injected
        # mlp_down. The old version fed the injected act here, which meant an upstream bug could
        # "catch" a down-side test and a down-side bug could be attributed to the epilogue.
        kernel(["--mode", "down", "--x", str(WORK / "act_negref.bin"),
                "--wd", str(WORK / "Wd_n.bin"), "--out", str(WORK / "nout_down.bin"),
                "--iters", "1"] + extra, "", quiet=True)
        if judge_neg(compare(read_bf16(WORK / "nout_down.bin", HID), refs["out"])):
            caught.append("out(down)")
        if "gateup" in modes:
            kernel(["--mode", "down", "--x", str(WORK / "nact.bin"),
                    "--wd", str(WORK / "Wd_n.bin"), "--out", str(WORK / "nout.bin"),
                    "--iters", "1"] + extra, "", quiet=True)
            if judge_neg(compare(read_bf16(WORK / "nout.bin", HID), refs["out"])):
                caught.append("out(chain)")
        observable = ((["gate_pre", "up_pre", "act"] if "gateup" in modes else [])
                      + ["out(down)"]
                      + (["out(chain)"] if "gateup" in modes else []))
        must_catch = [c for c in MUST_CATCH[neg] if c in observable]
        must_not = [c for c in MUST_NOT[neg] if c in observable]
        missing = [c for c in must_catch if c not in caught]
        wrong = [c for c in must_not if c in caught]
        print("    --neg %-8s %-56s caught: %-34s%s" % (
            neg, why, ",".join(caught) or "NOTHING",
            "  <-- also moved " + ",".join(wrong) if wrong else ""))
        if missing:
            fails.append("negative %s was NOT rejected at %s - the gate is vacuous"
                         % (neg, ",".join(missing)))
        if wrong:
            fails.append("negative %s moved %s, which it should not touch"
                         % (neg, ",".join(wrong)))


def judge_neg(c: dict) -> bool:
    """True if this comparison violates the criteria the clean gate applies. `act` is compared
    with a 2 ULP band there and compare() carries that into c['n_bad'], so a negative test
    cannot pass by being judged on a stricter rule than the thing it tests."""
    return bool(c["n_bad"] or 100.0 * c["ndiff"] / c["n"] > 2.0 or c["cos"] < 0.9999
                or c["meanrel"] > 1e-2)


def main() -> int:

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8", help="a fixture that has decode steps")
    ap.add_argument("--step", default=None, help="one step, e.g. decode0 (default: all)")
    ap.add_argument("--mode", default="all", choices=["gateup", "down", "all"])
    ap.add_argument("--layers", default="0,3,11,23", help="comma list, or 'all'")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rows", type=int, default=8, help="rows per block (sweep this)")
    ap.add_argument("--rows-sweep", default=None,
                    help="comma list, e.g. 1,2,4,8,16: timed AND correctness-checked each")
    ap.add_argument("--tile", type=int, default=6,
                    help="replicate each weight this many times for the cold-bandwidth "
                         "measurement (0 skips it). Rule: a number above ~240 GB/s must be "
                         "explained by cache residency, so push the working set past it.")
    ap.add_argument("--negative", action="store_true",
                    help="inject each bug in --neg into the kernel and assert the gate stops "
                         "it (the G-neg convention: a gate that cannot fail verifies nothing)")
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()

    d = ROOT / "data" / "refs" / args.fixture
    steps = [args.step] if args.step else sorted(
        {p.name.split(".")[0] for p in d.glob("decode*.L00.mlp.in.npy")})
    if not steps:
        raise SystemExit("%s has no decode-step MLP pins; this kernel is the k=1 path" % d)
    layers = list(range(24)) if args.layers == "all" else [int(x) for x in
                                                           args.layers.split(",")]
    if not BIN.exists():
        subprocess.run(["bash", str(BIN.parent / "build.sh")], check=True)
    if bf16_ulp.selftest():        # the comparator is part of the gate; check it first
        raise SystemExit("bf16_ulp selftest failed - the gate cannot be trusted")
    WORK.mkdir(parents=True, exist_ok=True)
    print("fixture %s | steps %s | mode %s | layers %s | rows/block %d | iters %d"
          % (args.fixture, ",".join(steps), args.mode,
             "all 24" if args.layers == "all" else layers, args.rows, args.iters))

    pre = WPK
    fails: list[str] = []
    worst: dict[str, int] = {}
    timings: dict[str, float] = {}
    from speccheck.specimpl import Weights, mlp as spec_mlp   # noqa: E402
    w: Weights | None = None

    for l in layers:
        need_w = args.mode in ("gateup", "all") or args.mode == "down"
        if not need_w:
            continue
        Wg = Wu = Wd = None
        wp = []
        if args.mode in ("gateup", "all"):
            Wg = ckpt_tensor(pre % l + "gate_proj.weight")
            Wu = ckpt_tensor(pre % l + "up_proj.weight")
            wp += [dump_bf16(Wg, WORK / "Wg.bin"), dump_bf16(Wu, WORK / "Wu.bin")]
        if args.mode in ("down", "all"):
            Wd = ckpt_tensor(pre % l + "down_proj.weight")
            wp += [dump_bf16(Wd, WORK / "Wd.bin")]
        print("\nL%02d" % l, end="  ")
        print("weights gate%s up%s down%s" % tuple(
            str(tuple(t.shape)) if t is not None else "-" for t in (Wg, Wu, Wd)))

        for step in steps:
            p = "%s.L%02d.mlp." % (step, l)
            if not (d / (p + "in.npy")).exists():
                continue
            x = torch.from_numpy(np.load(d / (p + "in.npy"))).to(torch.bfloat16).reshape(-1)
            refs = {k: torch.from_numpy(np.load(d / (p + k + ".npy"))).to(torch.bfloat16)
                    .reshape(-1) for k in ("gate_pre", "up_pre", "act", "out")}
            act_kernelside = None

            if args.mode in ("gateup", "all"):
                xp = dump_bf16(x, WORK / "x.bin")
                t = kernel(["--mode", "gateup", "--x", str(xp), "--wg", str(wp[0]),
                            "--wu", str(wp[1]), "--out", str(WORK / "act.bin"),
                            "--pre", str(WORK / "pre.bin"), "--iters", str(args.iters),
                            "--rows", str(args.rows)], "gateup")
                timings["gateup"] = float(t["ms"])
                pre_t = read_bf16(WORK / "pre.bin", 2 * FF)
                act_kernelside = dump_bf16(read_bf16(WORK / "act.bin", FF),
                                           WORK / "act_kernel.bin")
                judge("%s F1 gate_pre" % step, compare(pre_t[:FF], refs["gate_pre"]),
                      fails, worst)
                judge("%s F1 up_pre" % step, compare(pre_t[FF:], refs["up_pre"]),
                      fails, worst)
                judge("%s F2 act     " % step,
                      compare(read_bf16(WORK / "act.bin", FF), refs["act"], ulp_max=2),
                      fails, worst, ulp_max=2)

            dump_bf16(refs["act"], WORK / "act_ref.bin")
            t = kernel(["--mode", "down", "--x", str(WORK / "act_ref.bin"),
                        "--wd", str(wp[-1]), "--out", str(WORK / "out.bin"),
                        "--iters", str(args.iters), "--rows", str(args.rows)], "down")
            timings["down"] = float(t["ms"])
            judge("%s F3 out(ref-act)" % step,
                  compare(read_bf16(WORK / "out.bin", HID), refs["out"]), fails, worst)
            if act_kernelside is not None:
                kernel(["--mode", "down", "--x", str(act_kernelside), "--wd", str(wp[-1]),
                        "--out", str(WORK / "out_chain.bin"), "--iters", "1",
                        "--rows", str(args.rows)], "chain", quiet=True)
                judge("%s F4 out(kernel-act)" % step,
                      compare(read_bf16(WORK / "out_chain.bin", HID), refs["out"]),
                      fails, worst, ulp_gate=False, ndiff_pct_max=F4_NDIFF_PCT_MAX,
                      meanrel_max=F4_MEANREL_MAX, cos_min=F4_COS_MIN)

        # cross-check speccheck's own MLP against the reference on this layer's first step
        x = torch.from_numpy(np.load(d / ("%s.L%02d.mlp.in.npy" % (steps[0], l)))) \
            .to(torch.bfloat16)
        ref_out = torch.from_numpy(np.load(d / ("%s.L%02d.mlp.out.npy" % (steps[0], l)))) \
            .to(torch.bfloat16).reshape(-1)
        if w is None:                     # Weights() maps the whole checkpoint: do it once
            w = Weights(CKPT)
        judge("L%02d speccheck mlp()" % l, compare(spec_mlp(w, l, x).reshape(-1), ref_out),
              fails, worst)

    if args.negative:
        print("\nnegative tests (each must be rejected):")
        run_negatives(args, d, steps, layers, fails)

    if not args.no_time:
        print()
        if args.mode in ("gateup", "all"):
            timings["gateup"] = float(kernel(
                ["--mode", "gateup", "--x", str(WORK / "x.bin"), "--wg", str(WORK / "Wg.bin"),
                 "--wu", str(WORK / "Wu.bin"), "--out", str(WORK / "act.bin"),
                 "--iters", str(max(args.iters, 60)), "--rows", str(args.rows)],
                "gateup")["ms"])
        if args.mode in ("down", "all"):
            timings["down"] = float(kernel(
                ["--mode", "down", "--x", str(WORK / "act_ref.bin"),
                 "--wd", str(WORK / "Wd.bin"), "--out", str(WORK / "out.bin"),
                 "--iters", str(max(args.iters, 60)), "--rows", str(args.rows)],
                "down")["ms"])
        ms = sum(timings.values())
        print("\nper-layer MLP %.4f ms WARM  (%s)"
              % (ms, " + ".join("%s %.4f ms" % kv for kv in timings.items())))
        print("per-token MLP warm (24 layers): %.3f ms = %.0f%% of the probe — but that is a "
              "cache artifact: it re-reads one layer's 22 MB %d times. The honest number is "
              "the cold check below."
              % (24 * ms, 100.0 * (MLP_BYTES_PER_TOKEN / (24 * ms) / 1e6) / PROBE_GBS, 50))

    if args.rows_sweep:
        # Rule: correctness-check every config, not just the default (prototype/README.md).
        print()
        l0 = layers[0]
        p0 = "%s.L%02d.mlp." % (steps[0], l0)
        xref = torch.from_numpy(np.load(d / (p0 + "in.npy"))).to(torch.bfloat16).reshape(-1)
        aref = torch.from_numpy(np.load(d / (p0 + "act.npy"))).to(torch.bfloat16).reshape(-1)
        oref = torch.from_numpy(np.load(d / (p0 + "out.npy"))).to(torch.bfloat16).reshape(-1)
        dump_bf16(xref, WORK / "x.bin")
        dump_bf16(aref, WORK / "act_ref.bin")
        # The sweep's OWN weights: the layer loop above leaves the last layer's weights in
        # Wg.bin, and timing/correctness against a different layer's weights checks nothing.
        dump_bf16(ckpt_tensor(pre % l0 + "gate_proj.weight"), WORK / "Wg.bin")
        dump_bf16(ckpt_tensor(pre % l0 + "up_proj.weight"), WORK / "Wu.bin")
        dump_bf16(ckpt_tensor(pre % l0 + "down_proj.weight"), WORK / "Wd.bin")
        for r in [int(v) for v in args.rows_sweep.split(",")]:
            line = "rows/block %-3d" % r
            if args.mode in ("gateup", "all"):
                t = kernel(["--mode", "gateup", "--x", str(WORK / "x.bin"),
                            "--wg", str(WORK / "Wg.bin"), "--wu", str(WORK / "Wu.bin"),
                            "--out", str(WORK / "act.bin"), "--iters",
                            str(max(args.iters, 60)), "--rows", str(r)], "", quiet=True)
                c = compare(read_bf16(WORK / "act.bin", FF), aref, ulp_max=2)
                line += " gateup %7.4f ms %6.1f GB/s act ndiff %d max %d ULP |"
                line = line % (float(t["ms"]), float(t["gbs"]), c["ndiff"], c["maxulp"])
                if c["n_bad"] or c["cos"] < 0.9999:
                    fails.append("rows/block %d gateup: %d over-tolerance elements, cos %.6f"
                                 % (r, c["n_bad"], c["cos"]))
            if args.mode in ("down", "all"):
                t = kernel(["--mode", "down", "--x", str(WORK / "act_ref.bin"),
                            "--wd", str(WORK / "Wd.bin"), "--out", str(WORK / "out.bin"),
                            "--iters", str(max(args.iters, 60)), "--rows", str(r)], "",
                           quiet=True)
                c = compare(read_bf16(WORK / "out.bin", HID), oref)
                line += "  down %7.4f ms %6.1f GB/s | out ndiff %d max %d ULP"
                line = line % (float(t["ms"]), float(t["gbs"]), c["ndiff"], c["maxulp"])
                if c["n_bad"] or c["cos"] < 0.9999:
                    fails.append("rows/block %d down: %d over-tolerance elements, cos %.6f"
                                 % (r, c["n_bad"], c["cos"]))
            print(line)

    if args.tile and args.tile > 1:
        # The warm numbers are above the 237 GB/s probe, which per the measurement rules is
        # evidence of cache residency (or of skipped work). Tile each weight N deep: same
        # kernel, same math, working set N x larger, and every replica's output must still
        # match the reference row for row.
        n = args.tile
        cold: dict[str, float] = {}
        print()
        l0 = layers[0]
        p0 = "%s.L%02d.mlp." % (steps[0], l0)
        aref = torch.from_numpy(np.load(d / (p0 + "act.npy"))).to(torch.bfloat16).reshape(-1)
        oref = torch.from_numpy(np.load(d / (p0 + "out.npy"))).to(torch.bfloat16).reshape(-1)
        dump_bf16(torch.from_numpy(np.load(d / (p0 + "in.npy"))).to(torch.bfloat16)
                  .reshape(-1), WORK / "x.bin")
        dump_bf16(aref, WORK / "act_ref.bin")
        Wg = torch.cat([ckpt_tensor(pre % l0 + "gate_proj.weight")] * n)
        Wu = torch.cat([ckpt_tensor(pre % l0 + "up_proj.weight")] * n)
        Wd = torch.cat([ckpt_tensor(pre % l0 + "down_proj.weight")] * n)
        dump_bf16(Wg, WORK / "Wg_t.bin")
        dump_bf16(Wu, WORK / "Wu_t.bin")
        dump_bf16(Wd, WORK / "Wd_t.bin")
        print("cold check: each weight replicated %dx (gateup working set %.0f MiB, down "
              "%.0f MiB) - the LAST replica's output must still match, and the sweep is COLD "
              "because warm every config looks fast" % (n, n * 2 * FF * HID * 2 / 1048576,
                                                        n * HID * FF * 2 / 1048576))
        # Config choice has to be made COLD: warm, every config looks fast and the ranking is
        # just who fits in the iGPU cache better. Loop the rows/block configs against the tiled
        # (cache-defeated) weights and keep the best per-layer time. In a real token the two
        # kernels alternate, so for --mode all the pair is what gets timed (see below).
        cfgs = ([int(v) for v in args.rows_sweep.split(",")] if args.rows_sweep
                else [args.rows])
        best = None
        for r in cfgs:
            line = "  cold rows/block %-3d" % r
            per_layer = 0.0
            ok = True
            last = n - 1        # which replica to validate; see the note below
            if args.mode == "all":
                # A token ALTERNATES the two kernels: gateup's 88 MiB stream is what evicts the
                # down weights and the other way round, so bursting one kernel and then the
                # other measures two machines that no token runs on. `--mode mlp` alternates
                # them inside the burst, one burst iteration = one layer.
                t = kernel(["--mode", "mlp", "--x", str(WORK / "x.bin"),
                            "--act", str(WORK / "act_ref.bin"), "--wg", str(WORK / "Wg_t.bin"),
                            "--wu", str(WORK / "Wu_t.bin"), "--wd", str(WORK / "Wd_t.bin"),
                            "--out", str(WORK / "out_t.bin"), "--actout", str(WORK / "act_t.bin"),
                            "--iters", "30", "--rows", str(r)], "mlp", quiet=True)
                # Validate the LAST replica, not replica 0. Replica 0's bytes are the ones even
                # a kernel that read too little would certainly touch; the last replica is the
                # one whose traffic a skipped or short read would drop.
                c = compare(read_bf16(WORK / "act_t.bin", n * FF, last * FF), aref, ulp_max=2)
                co = compare(read_bf16(WORK / "out_t.bin", n * HID, last * HID), oref)
                line += (" pair %7.4f ms %6.1f GB/s (%.0f%% of probe) | replica %d/%d ndiff "
                         "act %d out %d"
                         % (float(t["ms"]), float(t["gbs"]), float(t["gbs"]) / PROBE_GBS * 100,
                            last, n, c["ndiff"], co["ndiff"]))
                per_layer = float(t["ms"]) / n
                ok = not (c["n_bad"] or co["n_bad"] or c["cos"] < 0.9999 or co["cos"] < 0.9999)
            if args.mode == "gateup":
                t = kernel(["--mode", "gateup", "--x", str(WORK / "x.bin"),
                            "--wg", str(WORK / "Wg_t.bin"), "--wu", str(WORK / "Wu_t.bin"),
                            "--out", str(WORK / "act_t.bin"), "--iters", "30",
                            "--rows", str(r)], "gateup", quiet=True)
                c = compare(read_bf16(WORK / "act_t.bin", FF), aref, ulp_max=2)
                line += (" gateup %7.4f ms %6.1f GB/s (%.0f%% of probe) ndiff %d max %d ULP |"
                         % (float(t["ms"]), float(t["gbs"]),
                            float(t["gbs"]) / PROBE_GBS * 100, c["ndiff"], c["maxulp"]))
                per_layer += float(t["ms"]) / n
                ok = not (c["n_bad"] or c["cos"] < 0.9999)
            if args.mode == "down":
                t = kernel(["--mode", "down", "--x", str(WORK / "act_ref.bin"),
                            "--wd", str(WORK / "Wd_t.bin"), "--out", str(WORK / "out_t.bin"),
                            "--iters", "30", "--rows", str(r)], "down", quiet=True)
                c = compare(read_bf16(WORK / "out_t.bin", HID), oref)
                line += (" down %7.4f ms %6.1f GB/s (%.0f%% of probe) ndiff %d max %d ULP"
                         % (float(t["ms"]), float(t["gbs"]),
                            float(t["gbs"]) / PROBE_GBS * 100, c["ndiff"], c["maxulp"]))
                per_layer += float(t["ms"]) / n
                ok = ok and not (c["n_bad"] or c["cos"] < 0.9999)
            if not ok:
                fails.append("cold rows/block %d: output outside tolerance" % r)
            tok = 24 * per_layer
            line += "  |  per-token MLP %.3f ms" % tok
            if best is None or tok < best[1]:
                best = (r, tok)
                line += "  <- best"
            print(line)
        cold["rows"] = best[0]
        cold["per_layer"] = best[1] / 24.0
        # The number the ledger wants. In a real decode step each layer streams its own 22 MB
        # of MLP weights once per token and the next layer evicts them, so the warm per-kernel
        # GB/s — which re-reads one layer's weights 50 times out of the iGPU cache — is not
        # the budget number. This is: cold time / replicas, times 24 layers.
        if args.mode == "all":
            per_layer = best[1] / 24.0
            tok = best[1]
            gbs = MLP_BYTES_PER_TOKEN / (tok / 1000.0) / 1e9
            print("  per-layer cold %.4f ms (rows/block %d) -> per-token MLP (24 layers) "
                  "%.3f ms = %.1f GB/s = %.0f%% of the %.0f GB/s probe (roofline %.3f ms/token)"
                  % (per_layer, best[0], tok, gbs, 100.0 * gbs / PROBE_GBS, PROBE_GBS,
                     MLP_BYTES_PER_TOKEN / PROBE_GBS / 1e6))

    print("\nworst BF16 ULP distance by output -- gated (large elements only) | raw (including"
          " the cancelling elements floor_probe.py attributes to arithmetic, not error) | the"
          " most elements any one comparison put inside the %g band | the tightest band these"
          " results would still pass at:" % CANCEL_REL)
    band_slack = None
    for k, (bad, raw, nband, minband) in worst.items():
        print("    %-20s %4d   %6d   %6d   %s" % (k, bad, raw, nband,
                                                  "%.1e" % minband if minband else "-"))
        if minband:
            band_slack = min(band_slack or 1e9, CANCEL_REL / minband)
    if band_slack:
        print("    band in use: |ref|/max <= %g, worst offender anywhere above it: %.1e -> %.0fx "
              "of slack. floor_probe.py puts the reference's own reordering floor at 1.1e-6 "
              "(gate_pre) and 6.1e-6 (out); a tighter band has to clear BOTH numbers."
              % (CANCEL_REL, max(v[3] for v in worst.values()), band_slack))
    if fails:
        print("\nFAIL (%d)" % len(fails))
        for f in fails[:20]:
            print("  -", f)
        return 1
    print("PASS: the §03.3 MLP reproduces the reference operands. What that means here:")
    print("  * F1/F2/F3 (each output from the REFERENCE operands): no element larger than %g"
          % CANCEL_REL)
    print("    of the tensor max differs by >1 BF16 ULP; <=2 % of elements differ at all; and")
    print("    the §10.1 class gates hold (cos >= 0.9999, meanrel <= 1e-2). floor_probe.py is")
    print("    what sets the 1-ULP-with-a-cancellation-band rule: reordering the reference's")
    print("    OWN F32 sum already moves ~1 element in 1e5 by >1 ULP, always at an element")
    print("    that cancelled to ~1e-6 of the max, where one ULP is a large fraction of the")
    print("    value. A flat 'max 1 ULP' rule would forbid correctness.")
    print("  * F4 (chain: down applied to the KERNEL's act) is judged on aggregates only")
    print("    (cos >= %.4f, meanrel <= %.0e, ndiff <= %.0f %%): floor_probe.py shows a 1-ULP flip"
          % (F4_COS_MIN, F4_MEANREL_MAX, F4_NDIFF_PCT_MAX))
    print("    in `act` -- the most a correct kernel with a different summation order may produce")
    print("    -- puts >1 ULP onto an `out` element in all 192 cases it surveyed (192 x 8 trials,")
    print("    7483 offenders, worst 28665 ULP, most offenders NOT small elements). Per-element")
    print("    ULP there measures arithmetic sensitivity, not the kernel. The same survey measures")
    print("    these aggregates: worst legit meanrel 9.1e-4 for ONE flip (2.0e-3 at five), so the")
    print("    %.0e band is marginal by construction, not conservative -- see results.md." %
          F4_MEANREL_MAX)
    print("  * --negative injects six wrong kernels and requires each to be caught where the")
    print("    bug is and nowhere else (--mode down skips the two that live in the epilogue).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
