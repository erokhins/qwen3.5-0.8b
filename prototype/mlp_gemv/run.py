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

Q10 added the WEIGHT-WALK switch. `mlp_gateup` can read its two weight streams interleaved (what
has always shipped) or as two contiguous passes; `--mapping` picks which, and `--mapping-identity`
is the stage that proves the two are BIT-identical — same `act`, same `pre`, same `out`, layer by
layer — which is what makes the walk a free knob rather than a numerical decision. The claim, its
measurement and the reason it holds are `MlpMap` in `mlp_gemv_kernels.h` and
`devices/strix-halo-mixed-traffic.md` §Q9.

    .venv/bin/python prototype/mlp_gemv/run.py [--fixture dec8] [--mode gateup|down|all]
                                              [--layers 0,3,11,23|all] [--rows 8] [--no-time]
                                              [--mapping interleave|contig] [--mapping-identity]
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
# perturbation survey in floor_probe.py shows why (a legal 1-ULP `act` flip puts >1 ULP onto an
# `out` element in all 192 layer x step cases it surveyed). So F4 is judged on aggregates ONLY and
# these two numbers ARE the gate. Both are measured, not argued: `floor_probe.py --perturb N
# --trials 8` over 192 cases x 8 trials = 1 536 perturbed dots says the worst a LEGAL chain can
# do is cos 0.99999661 / meanrel 9.12e-4 with one 1-ULP `act` flip, and cos 0.99999615 /
# meanrel 2.01e-3 with five (F2 measures up to 5 legal flips per layer-step). meanrel gets 2.5x
# over the worse of those because 2.01e-3 is a SAMPLE worst, not an upper bound — if this ever
# needs more room, enlarge the survey rather than the multiplier.
#
# ndiff is not here any more. It was <= 15 %, and one legal flip measures 29.1 % (five: 46.8 %),
# so that band rejected correct kernels; a band wide enough to accept them (~50 %) would carry no
# information cos and meanrel do not already carry, so it was removed rather than inflated.
#
# What this band can and cannot detect: it is a gross-error net with a floor of ~5e-3. `--negative
# f32act` moves the chain to meanrel 4.14e-3, i.e. BELOW that floor, and is rejected at F2 `act`
# instead (40.4 % ndiff against a 2 % rule). Gross errors are still caught here: `trunc` 1.0e-2,
# `tail` 9.8e-2, `half` 7.4e-1, `rowswap` 1.4e+0 (all `out(chain)` meanrel, layer 0 decode0).
F4_COS_MIN = 0.9999      # measured legal worst 0.99999615 -> 26x of slack in 1 - cos
F4_MEANREL_MAX = 5e-3    # measured legal worst 2.01e-3 -> 2.5x
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
    judging arithmetic sensitivity, not the kernel. `ndiff_pct_max=None` goes with it: a legal
    chain measures 29-47 % differing elements (measured, same survey), so cos and meanrel — sized
    by that same measurement, see the F4_* constants — are F4's only criteria.
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
    if ndiff_pct_max is not None and 100.0 * c["ndiff"] / c["n"] > ndiff_pct_max:
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
    margs = mapping_args(args)
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
                    "--pre", str(WORK / "npre.bin"), "--iters", "1"] + margs + extra,
                   "", quiet=True)
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


def mapping_args(args) -> list[str]:
    """The `--mapping` flags for a GATEUP launch. Only gate/up has two weight streams, and the
    binary refuses `--mapping` on `--mode down`, so down-side launches must never grow this list
    (a flag that silently vanished from the gateup calls would make the A/B compare one walk against
    itself)."""
    return ["--mapping", args.mapping]


def run_mapping_identity(args, d, steps, layers, fails, worst) -> None:
    """Q10: the two gate/up weight walks must be BIT-identical, layer by layer.

    The claim being proved is the one that makes the walk a free knob: `MAP_INTERLEAVE` and
    `MAP_CONTIG` fetch the same `uint4` words in the same order into the same accumulators, so every
    output bit must agree — `act`, both raw `pre` dumps, the batched form's `act`, and therefore the
    chain's `out` (one `mlp_down`, which has no walk to choose, fed each side's own `act`). Any
    differing byte is the claim failing, and there is no tolerance to argue about here: this compares
    bits, not ULPs.

    It is not vacuous by construction: the same two runs are ALSO judged against the reference pins
    with the ordinary F1/F2/F3 rules, so "identical" is never allowed to mean "identically wrong".
    """
    print("\nmapping identity: interleave vs contig, on %s, layers %s, step %s"
          % (args.fixture, "all 24" if args.layers == "all" else layers, steps[0]))
    if args.mode == "down":
        print("    skipped: --mode down never launches the gate/up kernel, so it has no walk to A/B")
        return
    st = steps[0]
    tot = {"act": 0, "pre": 0, "out": 0, "act_batched": 0}
    for l in layers:
        p0 = "%s.L%02d.mlp." % (st, l)
        if not (d / (p0 + "in.npy")).exists():
            continue
        dump_bf16(ckpt_tensor(WPK % l + "gate_proj.weight"), WORK / "Wg_m.bin")
        dump_bf16(ckpt_tensor(WPK % l + "up_proj.weight"), WORK / "Wu_m.bin")
        dump_bf16(ckpt_tensor(WPK % l + "down_proj.weight"), WORK / "Wd_m.bin")
        x = torch.from_numpy(np.load(d / (p0 + "in.npy"))).to(torch.bfloat16).reshape(-1)
        refs = {k: torch.from_numpy(np.load(d / (p0 + k + ".npy"))).to(torch.bfloat16).reshape(-1)
                for k in ("gate_pre", "up_pre", "act", "out")}
        dump_bf16(x, WORK / "x_m.bin")
        outs = {}
        for mp in ("interleave", "contig"):
            tag = mp[0:3]
            kernel(["--mode", "gateup", "--x", str(WORK / "x_m.bin"), "--wg", str(WORK / "Wg_m.bin"),
                    "--wu", str(WORK / "Wu_m.bin"), "--out", str(WORK / f"act_{tag}.bin"),
                    "--pre", str(WORK / f"pre_{tag}.bin"), "--iters", "1",
                    "--mapping", mp], "", quiet=True)
            # `down` takes each side's OWN act, so `out` identity is the chain's, not a rerun of the
            # same launch on the same input.
            kernel(["--mode", "down", "--x", str(WORK / f"act_{tag}.bin"),
                    "--wd", str(WORK / "Wd_m.bin"), "--out", str(WORK / f"out_{tag}.bin"),
                    "--iters", "1"], "", quiet=True)
            outs[mp] = (read_bf16(WORK / f"act_{tag}.bin", FF),
                        read_bf16(WORK / f"pre_{tag}.bin", 2 * FF),
                        read_bf16(WORK / f"out_{tag}.bin", HID))
        (act_i, pre_i, out_i), (act_c, pre_c, out_c) = outs["interleave"], outs["contig"]
        # The batched form too: that is what a prefill wave launches, and one extra token-tile run at
        # T=1 under each walk closes the square — batched@contig against single@interleave is the
        # engine's own before/after pair, not two variants of the same instantiation.
        for mp in ("interleave", "contig"):
            tag = mp[0:3]
            kernel(["--mode", "gateup", "--x", str(WORK / "x_m.bin"), "--in", str(HID),
                    "--tokens", "1", "--wg", str(WORK / "Wg_m.bin"), "--wu", str(WORK / "Wu_m.bin"),
                    "--out", str(WORK / f"actb_{tag}.bin"), "--iters", "1",
                    "--mapping", mp], "", quiet=True)
        actb_c = read_bf16(WORK / "actb_con.bin", FF)
        actb_i = read_bf16(WORK / "actb_int.bin", FF)
        bits = {"act": int((act_i != act_c).sum()), "pre": int((pre_i != pre_c).sum()),
                "out": int((out_i != out_c).sum()),
                # batched@contig vs single@interleave is the engine's own before/after pair; the
                # second term keeps the batched interleaved form honest against the same baseline.
                "act_batched": int((actb_c != act_i).sum()) + int((actb_i != act_i).sum())}
        for k in tot:
            tot[k] += bits[k]
        print("    L%02d  act %d/%d differ | pre %d/%d | out %d/%d | act(batched) %d/%d"
              % (l, bits["act"], FF, bits["pre"], 2 * FF, bits["out"], HID,
                 bits["act_batched"], 2 * FF))
        if any(bits.values()):
            fails.append("L%02d %s: the two gate/up weight walks differ (%s) — the mapping is not"
                         " bit-neutral" % (l, st, bits))
        # ...and both sides still reproduce the reference, under the ordinary rules. `act`/`pre` are
        # F1/F2 material (each side's own dots); `out` here is down fed the KERNEL's act, so it is the
        # F4 chain and must be judged the way F4 is judged — aggregates only, because a legal 1-ULP
        # `act` flip puts many ULP onto an `out` element (floor_probe.py; see the F4_* constants).
        for mp, (act, pre, out) in (("interleave", outs["interleave"]), ("contig", outs["contig"])):
            judge("L%02d %s F1 gate_pre" % (l, mp), compare(pre[:FF], refs["gate_pre"]),
                  fails, worst)
            judge("L%02d %s F1 up_pre " % (l, mp), compare(pre[FF:], refs["up_pre"]),
                  fails, worst)
            judge("L%02d %s F2 act    " % (l, mp),
                  compare(act, refs["act"], ulp_max=2), fails, worst, ulp_max=2)
            judge("L%02d %s F4 out(chain)" % (l, mp), compare(out, refs["out"]), fails, worst,
                  ulp_gate=False, ndiff_pct_max=None, meanrel_max=F4_MEANREL_MAX,
                  cos_min=F4_COS_MIN)
    print("    total across %d layers: act %d, pre %d, out %d, act(batched) %d differing"
          " elements (all must be 0)"
          % (len(layers), tot["act"], tot["pre"], tot["out"], tot["act_batched"]))


def judge_neg(c: dict) -> bool:
    """True if this comparison violates the criteria the clean gate applies. `act` is compared
    with a 2 ULP band there and compare() carries that into c['n_bad'], so a negative test
    cannot pass by being judged on a stricter rule than the thing it tests."""
    return bool(c["n_bad"] or 100.0 * c["ndiff"] / c["n"] > 2.0 or c["cos"] < 0.9999
                or c["meanrel"] > 1e-2)


def worst_over_rows(got: torch.Tensor, ref: torch.Tensor, **kw) -> dict:
    """Judge a [rows, dim] tile one row at a time and return the worst row's comparison.

    Per row rather than over the flattened tile because the cancellation band and the 2 % ndiff
    rule are relative to the tensor being judged: flattened, one loud token could hide a quiet
    one, and the F1–F4 rules were sized per token by floor_probe.py. `kw` (e.g. ulp_max=2 for
    `act`) is passed through unchanged, so the batched stage judges with exactly the decode
    stage's rules.
    """
    bad = None
    for i in range(got.shape[0]):
        c = compare(got[i], ref[i], **kw)
        if bad is None or (c["n_bad"], c["ndiff"], c["maxulp"], -c["cos"], -c["meanrel"]) > (
                bad["n_bad"], bad["ndiff"], bad["maxulp"], -bad["cos"], -bad["meanrel"]):
            bad = c
            bad["row"] = i
    return bad


def run_tokens(args, layers, fails, worst) -> None:
    """P4: the token-batched MLP, gated against the prefill pins and against the k=1 kernel.

    Two claims, in order of how much the engine depends on them:

    1. **bit-identity vs the shipped single-token kernel.** The batched kernel re-reads each
       weight row once per TILE of tokens instead of once per token; if that changed any
       output bit the legal route would move and every margin `route_allowance.py` priced would
       be stale. So for sampled tokens the k=1 kernel's answer must BE the batched kernel's row,
       bit for bit.
    2. **the pins.** F1/F2/F3/F4 on the whole [T, dim] tile, same rules, judged per row.
    """
    dp = ROOT / "data" / "refs" / args.tokens
    if not dp.exists():
        raise SystemExit("no prefill fixture %s (try p64)" % dp)
    T0 = int(np.load(dp / ("L%02d.mlp.in.npy" % layers[0])).shape[1])
    print("\ntoken-batched stage on %s: %d tokens x layers %s (shared-memory tile, not registers,"
          " is what bounds the batch) | mapping %s"
          % (dp.name, T0, layers, args.mapping))
    margs = mapping_args(args)
    sample = sorted({0, 1, T0 // 2, max(0, T0 - 2), T0 - 1})
    for l in layers:
        pre = "L%02d.mlp." % l
        Wg = dump_bf16(ckpt_tensor(WPK % l + "gate_proj.weight"), WORK / "Wg_b.bin")
        Wu = dump_bf16(ckpt_tensor(WPK % l + "up_proj.weight"), WORK / "Wu_b.bin")
        Wd = dump_bf16(ckpt_tensor(WPK % l + "down_proj.weight"), WORK / "Wd_b.bin")
        xref = torch.from_numpy(np.load(dp / (pre + "in.npy"))).to(torch.bfloat16).reshape(-1, HID)
        refs = {k: torch.from_numpy(np.load(dp / (pre + k + ".npy"))).to(torch.bfloat16)
                .reshape(-1, FF if k in ("gate_pre", "up_pre", "act") else HID)
                for k in ("gate_pre", "up_pre", "act", "out")}
        T = xref.shape[0]
        print("\nL%02d  x %s | refs gate_pre/up_pre/act %s, out %s"
              % (l, tuple(xref.shape), tuple(refs["act"].shape), tuple(refs["out"].shape)))
        # stage R for the tile: the pins must be self-consistent as a tile under the REFERENCE's
        # rounding policy — gate/up round to BF16, silu rounds again, the product rounds a third
        # time (mlp_gemv.hip.cpp's header, and NEG_F32ACT measures what the one-rounding reading
        # costs: 26 % of `act` moves by 1 ULP). Without this the batched kernel could be gated
        # against a mispaired file (§10.5).
        mirror = (torch.nn.functional.silu(refs["gate_pre"].float()).to(torch.bfloat16)
                  * refs["up_pre"]).to(torch.bfloat16)
        dump_bf16(xref, WORK / "x_tile.bin")
        dump_bf16(refs["act"], WORK / "act_tile.bin")
        kernel(["--mode", "gateup", "--x", str(WORK / "x_tile.bin"), "--in", str(HID),
                "--tokens", str(T), "--wg", str(Wg), "--wu", str(Wu),
                "--out", str(WORK / "act_batched.bin"), "--pre", str(WORK / "pre_batched.bin"),
                "--iters", "1"] + margs, "gateup x%d" % T, quiet=True)
        kernel(["--mode", "down", "--x", str(WORK / "act_tile.bin"), "--in", str(FF),
                "--tokens", str(T), "--wd", str(Wd), "--out", str(WORK / "out_batched.bin"),
                "--iters", "1"], "down x%d (ref act)" % T, quiet=True)
        kernel(["--mode", "down", "--x", str(WORK / "act_batched.bin"), "--in", str(FF),
                "--tokens", str(T), "--wd", str(Wd), "--out", str(WORK / "out_chain.bin"),
                "--iters", "1"], "down x%d (kernel act)" % T, quiet=True)
        pre_t = read_bf16(WORK / "pre_batched.bin", 2 * FF * T).reshape(T, 2 * FF)
        act_b = read_bf16(WORK / "act_batched.bin", FF * T).reshape(T, FF)
        out_b = read_bf16(WORK / "out_batched.bin", HID * T).reshape(T, HID)
        chain_b = read_bf16(WORK / "out_chain.bin", HID * T).reshape(T, HID)

        for tag, got, ref, ckw, jkw in (
                ("stage R act", mirror, refs["act"], {"ulp_max": 2}, {"ulp_max": 2}),
                ("F1 gate_pre", pre_t[:, :FF], refs["gate_pre"], {}, {}),
                ("F1 up_pre", pre_t[:, FF:], refs["up_pre"], {}, {}),
                ("F2 act", act_b, refs["act"], {"ulp_max": 2}, {"ulp_max": 2}),
                ("F3 out(ref-act)", out_b, refs["out"], {}, {}),
                ("F4 out(kernel-act)", chain_b, refs["out"], {},
                 {"ulp_gate": False, "ndiff_pct_max": None, "meanrel_max": F4_MEANREL_MAX,
                  "cos_min": F4_COS_MIN})):
            c = worst_over_rows(got, ref, **ckw)
            judge("L%02d r%03d %s" % (l, c["row"], tag), c, fails, worst, **jkw)

        # claim 1: the k=1 kernel's answer for a token IS the batched kernel's row for that token.
        # The k=1 side runs under `args.mapping` too, so `--mapping contig` checks the batched contig
        # walk against the single-token contig walk AND (by the identity stage) against the shipped
        # interleaved one — one more place a walk that moved bits would show up.
        bad_bits = 0
        for t in sample:
            dump_bf16(xref[t:t + 1], WORK / "x_one.bin")
            kernel(["--mode", "gateup", "--x", str(WORK / "x_one.bin"), "--wg", str(Wg),
                    "--wu", str(Wu), "--out", str(WORK / "act_one.bin"), "--iters", "1"] + margs,
                   "", quiet=True)
            dump_bf16(read_bf16(WORK / "act_one.bin", FF), WORK / "act_one_f.bin")
            kernel(["--mode", "down", "--x", str(WORK / "act_one_f.bin"), "--wd", str(Wd),
                    "--out", str(WORK / "out_one.bin"), "--iters", "1"], "", quiet=True)
            for nm, one, many in (("act", read_bf16(WORK / "act_one.bin", FF), act_b[t]),
                                  ("out", read_bf16(WORK / "out_one.bin", HID), chain_b[t])):
                n = int((one != many).sum())
                bad_bits += n
                if n:
                    fails.append("L%02d token %d: the %d-token kernel's `%s` row differs from the"
                                 " single-token kernel in %d elements — batching changed the"
                                 " arithmetic" % (l, t, T, nm, n))
        print("    bit-identity vs the k=1 kernel: tokens %s, `act` + `out`, %d elements"
              " compared, %d differ" % (sample, len(sample) * (FF + HID), bad_bits))


def main() -> int:

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="dec8", help="a fixture that has decode steps")
    ap.add_argument("--step", default=None, help="one step, e.g. decode0 (default: all)")
    ap.add_argument("--mode", default="all", choices=["gateup", "down", "all"])
    ap.add_argument("--layers", default="0,3,11,23", help="comma list, or 'all'")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rows", type=int, default=8, help="rows per block (sweep this)")
    ap.add_argument("--tokens", default=None, metavar="FIXTURE",
                    help="P4 stage: run the token-batched gateup/down over this PREFILL fixture's"
                         " L{ll}.mlp.* pins (p64/p300), require F1-F4 to hold per row, and"
                         " require the batched rows to be BIT-IDENTICAL to the k=1 kernel run"
                         " token by token. Multi-token arithmetic is otherwise untested.")
    ap.add_argument("--rows-sweep", default=None,
                    help="comma list, e.g. 1,2,4,8,16: timed AND correctness-checked each")
    ap.add_argument("--tile", type=int, default=6,
                    help="replicate each weight this many times for the cold-bandwidth "
                         "measurement (0 skips it). Rule: a number above ~240 GB/s must be "
                         "explained by cache residency, so push the working set past it.")
    ap.add_argument("--negative", action="store_true",
                    help="inject each bug in --neg into the kernel and assert the gate stops "
                         "it (the G-neg convention: a gate that cannot fail verifies nothing)")
    ap.add_argument("--mapping", default="interleave", choices=["interleave", "contig"],
                    help="which gate/up weight walk to launch (Q10). `interleave` is the shipped"
                         " one and the default; every gateup launch in this run uses the value, so"
                         " `--mapping contig` re-runs the whole gate under the alternative walk")
    ap.add_argument("--mapping-identity", action="store_true",
                    help="Q10's proof stage: run gate/up under BOTH walks for each layer and"
                         " require act/pre/out to be byte-identical, then judge each side against"
                         " the pins so identity cannot mean 'identically wrong'")
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
    print("fixture %s | steps %s | mode %s | layers %s | rows/block %d | iters %d | mapping %s"
          % (args.fixture, ",".join(steps), args.mode,
             "all 24" if args.layers == "all" else layers, args.rows, args.iters, args.mapping))
    margs = mapping_args(args)

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
                            "--rows", str(args.rows)] + margs, "gateup")
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
                      fails, worst, ulp_gate=False, ndiff_pct_max=None,
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

    if args.mapping_identity:
        run_mapping_identity(args, d, steps, layers, fails, worst)

    if args.tokens:
        run_tokens(args, layers, fails, worst)

    if not args.no_time:
        print()
        if args.mode in ("gateup", "all"):
            timings["gateup"] = float(kernel(
                ["--mode", "gateup", "--x", str(WORK / "x.bin"), "--wg", str(WORK / "Wg.bin"),
                 "--wu", str(WORK / "Wu.bin"), "--out", str(WORK / "act.bin"),
                 "--iters", str(max(args.iters, 60)), "--rows", str(args.rows)] + margs,
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
                            str(max(args.iters, 60)), "--rows", str(r)] + margs, "", quiet=True)
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
                            "--iters", "30", "--rows", str(r)] + margs, "mlp", quiet=True)
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
                            "--rows", str(r)] + margs, "gateup", quiet=True)
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
    print("    (cos >= %.4f, meanrel <= %.0e, NO ndiff rule): a 1-ULP flip in `act` -- the most"
          % (F4_COS_MIN, F4_MEANREL_MAX))
    print("    a correct kernel with a different summation order may produce -- puts >1 ULP onto")
    print("    an `out` element in all 192 cases floor_probe.py surveyed (192 x 8 trials, 7483")
    print("    offenders, worst 28665 ULP, most of them NOT small elements), so per-element ULP")
    print("    there measures arithmetic sensitivity, not the kernel. The same 1536-dot survey")
    print("    measures what a LEGAL chain reaches: worst cos 0.99999615 / meanrel 2.01e-3 at five")
    print("    legal flips (0.99999661 / 9.12e-4 at one), ndiff 29-47 %. meanrel is banded at")
    print("    %.0e, 2.5x over that measured worst; ndiff was DROPPED, not inflated -- a band"
          % F4_MEANREL_MAX)
    print("    wide enough to admit a legal chain (~50 %) carries no information that cos and")
    print("    meanrel do not already carry. F4 is therefore a gross-error net with a floor of")
    print("    ~5e-3: --negative f32act moves the chain by 4.14e-3, under that floor, and is")
    print("    caught at F2 `act` instead; trunc/tail/half/rowswap move it 1.0e-2..1.4e+0 and")
    print("    are caught here too. Measurements: results.md and specification/10 §10.1.")
    print("  * --negative injects six wrong kernels and requires each to be caught where the")
    print("    bug is and nowhere else (--mode down skips the two that live in the epilogue).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
