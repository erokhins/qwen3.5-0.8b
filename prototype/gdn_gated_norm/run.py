#!/usr/bin/env python3
"""Gate for §04.5's gated RMSNorm: a CPU mirror that needs no device, and a device gate.

    .venv/bin/python prototype/gdn_gated_norm/run.py --mirror     # CPU only, no GPU touched
    .venv/bin/python prototype/gdn_gated_norm/run.py --gate       # device: run + negatives
    .venv/bin/python prototype/gdn_gated_norm/run.py --gate --time

--mirror is the half that decides the arithmetic: it compares BF16 bit patterns of six readings of
§04.5 against `gdn.gated_out` on four fixtures and requires that the bit-exact set is exactly the
kernel's reading plus `speccheck.specimpl.rmsnorm_gated` called the way speccheck calls it (the two
are the same formula: the affine lands in the activation dtype because the weight arrives BF16).
That is the whole reason this op exists as its own directory: A-01's cast order was asserted at no
bit level anywhere (§10.2's bands are blind to it). A-01's letter — affine in F32 — is what the pins
refuse, and how far it differs depends only on the weight dtype it is handed (results.md §1).

--gate is the device half: the binary gets the same pins the mirror gets, must reproduce
`gdn.gated_out`, and each of six injected bugs must move the stage it lives in and leave the other
one alone. Run it under `tools/gpu_lock.sh` only if you want the timing row; the correctness half
needs no lock (AGENTS.md).

A plan names its weight the way `WeightStore` is keyed — the §02.5 engine name
(`lm.layers.N.linear_attn.norm.weight`), never the checkpoint key `mirror_probe.py` reads the file
with. Getting that wrong aborts the subprocess in `store.at()` with an uncaught `std::out_of_range`,
which is how the first-ever device run of this gate died, so `write_cases()` builds every plan name
through `weight_names.normalize_name` (the same §02.5 mirror the guard below checks against) and
each plan is verified against the store's key set **before** the binary is launched: a bad name now
prints the offending name on the CPU instead of a SIGABRT from a child process.

Band for `gated_out`: max 1 BF16 ULP with ≤1 % of elements differing, which is the band
prototype/gdn_decode/check.py already uses for this exact op class (an F32 norm whose pins are
BF16-valued: reordering the mean-of-squares is legal and flips a handful of elements by 1 ULP —
prototype/glue/floor_probe.py measures 3-4 of 393 216). `bf16_ulp.report` prints what this run
actually used, so the band can be tightened the day it lands at 0.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "prototype"), str(ROOT / "prototype" / "proj_gemv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mirror_probe as mp  # noqa: E402
import operands as ops  # noqa: E402
import bf16_ulp  # noqa: E402
import weight_names as wn  # noqa: E402

BIN = Path(__file__).parent / "gdn_gated_norm"
# Column of the weight name in a plan line: `NAME OFILE ZFILE WNAME OUT [NORM]`.
WEIGHT_FIELD = 3
STEPS = ["decode%d" % s for s in range(8)]
FIXTURE = "dec8"

# Each injection: (name, stages it must move, stages it must NOT move).
NEGATIVES = [
    ("w_f32", ["gated_out"], ["norm_pre_gate"]),
    ("affine_uncast", ["gated_out"], ["norm_pre_gate"]),
    ("single_cast", ["gated_out"], ["norm_pre_gate"]),
    ("silu_rounded", ["gated_out"], ["norm_pre_gate"]),
    ("bf16acc", ["gated_out", "norm_pre_gate"], []),
    ("noeps", ["gated_out", "norm_pre_gate"], []),
]
STAGES = ("gated_out", "norm_pre_gate")


def mirror_only() -> int:
    print("== CPU mirror gate (no device): §04.5 cast order against gdn.gated_out ==")
    rc = mp.main([])
    # Both selftests are part of the verdict, not commentary: bf16_ulp is what judges the bit
    # distance and weight_names is what the plan guard and the plan writer both rely on.
    ulp_ok = bf16_ulp.selftest() == 0
    names_ok = wn.selftest() == 0
    print("\nbf16_ulp selftest:", "PASS" if ulp_ok else "FAIL")
    print("weight_names selftest:", "PASS" if names_ok else "FAIL")
    if not (ulp_ok and names_ok):
        rc |= 1
    print("MIRROR GATE:", "PASS" if rc == 0 else "FAIL")
    return rc


def write_cases(d: Path, fixture: str) -> list[dict]:
    """One case per (step, GDN layer): pin values as raw files + the expected outputs."""
    fx = ROOT / "data" / "refs" / fixture
    cases = []
    for (o, z, w, ref) in mp.cases(fixture, STEPS):
        l = len(cases)
        name = "c%03d" % l
        of, zf = d / (name + ".o.f32"), d / (name + ".z.bin")
        o.numpy().astype(np.float32).tofile(of)
        z.to(torch.uint16).numpy().tofile(zf)
        cases.append(dict(name=name, ofile=of, zfile=zf, weight=w, ref=ref, layer=None,
                          wname=None))
    # map cases back to layers by re-reading the pin order (cases() iterates layer-major)
    idx = 0
    for l in range(24):
        for st in STEPS:
            if (fx / ("%s.L%02d.gdn.o_pre_norm.npy" % (st, l))).exists():
                cases[idx]["layer"] = l
                # The one place this gate turns a layer number into the weight the plan asks for.
                # Derived through the §02.5 mirror from the checkpoint key `mirror_probe.py`
                # already used, so producer and guard cannot drift apart.
                cases[idx]["wname"] = wn.normalize_name(
                    ops.LAYER_PREFIX % l + "linear_attn.norm.weight")[1]
                idx += 1
    return cases


def run_binary(plan: Path, outdir: Path, neg: str = "none") -> None:
    if BIN.exists() is False and Path(str(BIN) + "/gdn_gated_norm").exists() is False:
        pass
    exe = BIN if BIN.is_file() else Path(__file__).parent / "gdn_gated_norm"
    cmd = [str(exe), "--mode", "once", "--plan", str(plan), "--neg", neg]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("  $ " + " ".join(cmd[0:1]) + " --mode once --plan <plan> --neg " + neg)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit("binary failed (rc %d)" % r.returncode)
    for line in r.stdout.strip().splitlines():
        if line.startswith(("loader:", "plan ", "RESULT")):
            print("  " + line)


def read_out(path: Path, n: int) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16)[:n]).view(torch.bfloat16)


def device_gate(with_time: bool) -> int:
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cases = write_cases(d, FIXTURE)
        print("== device gate: %d cases (%d GDN layers × %d decode steps of %s) =="
              % (len(cases), len({c['layer'] for c in cases}), len(STEPS), FIXTURE))
        plan = d / "plan.txt"
        outs = {}
        with plan.open("w") as f:
            for c in cases:
                c["out"] = d / (c["name"] + ".gated.bin")
                c["norm"] = d / (c["name"] + ".norm.bin")
                f.write("%s %s %s %s %s %s\n" % (
                    c["name"], c["ofile"], c["zfile"], c["wname"],
                    c["out"], c["norm"]))
        wn.check_plan(plan, WEIGHT_FIELD)
        print("\n-- clean run --")
        run_binary(plan, d)
        worst_ulp, worst_ndiff, n_el, tightest = 0, 0, 0, 0.0
        clean = {}
        for c in cases:
            got = read_out(c["out"], 2048)
            clean[c["name"]] = dict(gated=got, norm=read_out(c["norm"], 2048))
            rep = bf16_ulp.report(got.float(), c["ref"].float(), ulp_max=1, cancel_rel=1e-4)
            worst_ulp = max(worst_ulp, rep["maxulp"])
            worst_ndiff = max(worst_ndiff, rep["ndiff"])
            tightest = max(tightest, rep["worst_off_rel"])
            n_el += rep["n"]
            if rep["n_bad"] or rep["ndiff"] > 0.01 * rep["n"]:
                fails.append("%s: ndiff %d/%d maxulp %d n_bad %d"
                             % (c["name"], rep["ndiff"], rep["n"], rep["maxulp"], rep["n_bad"]))
        print("  gated_out vs pins over %d elements: worst case ndiff %d/2048, max ULP %d, "
              "tightest band these results pass at %.1e" % (n_el, worst_ndiff, worst_ulp, tightest))
        if worst_ulp > 1:
            fails.append("a gated_out element is more than 1 BF16 ULP from the pin")

        print("\n-- negatives (each must move only the stage it lives in) --")
        for neg, must, must_not in NEGATIVES:
            plan_neg = d / ("plan_%s.txt" % neg)
            with plan_neg.open("w") as f:
                for c in cases:
                    c["nout"] = d / (c["name"] + ".%s.gated.bin" % neg)
                    c["nnorm"] = d / (c["name"] + ".%s.norm.bin" % neg)
                    f.write("%s %s %s %s %s %s\n" % (
                        c["name"], c["ofile"], c["zfile"], c["wname"],
                        c["nout"], c["nnorm"]))
            wn.check_plan(plan_neg, WEIGHT_FIELD)
            run_binary(plan_neg, d, neg)
            moved = {s: 0 for s in STAGES}
            for c in cases:
                g = read_out(c["nout"], 2048)
                nm = read_out(c["nnorm"], 2048)
                moved["gated_out"] += int((g != clean[c["name"]]["gated"]).sum())
                moved["norm_pre_gate"] += int((nm != clean[c["name"]]["norm"]).sum())
            bad = [s for s in must if moved[s] == 0] + [s for s in must_not if moved[s] > 0]
            verdict = "OK" if not bad else "MISCATCHED " + ",".join(bad)
            print("  %-14s moved gated_out %7d/294912, norm_pre_gate %7d/294912 -> %s"
                  % (neg, moved["gated_out"], moved["norm_pre_gate"], verdict))
            if bad:
                fails.append("negative %s mislocalised: %s" % (neg, verdict))

        if with_time:
            print("\n-- timing (take tools/gpu_lock.sh first; rule 7: wall-clock over a burst) --")
            exe = BIN if BIN.is_file() else Path(__file__).parent / "gdn_gated_norm"
            r = subprocess.run([str(exe), "--mode", "burst", "--plan", str(plan),
                                "--iters", "200", "--trials", "3"],
                               cwd=ROOT, capture_output=True, text=True)
            for line in (r.stdout + r.stderr).splitlines():
                if line.startswith(("TIMING", "RESULT")):
                    print("  " + line)

    print("\n" + ("DEVICE GATE FAILED:\n  " + "\n  ".join(fails) if fails
                  else "DEVICE GATE PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true", help="CPU-only cast-order gate (no GPU)")
    ap.add_argument("--gate", action="store_true", help="device gate (needs the GPU idle)")
    ap.add_argument("--time", action="store_true", help="with --gate: burst timing too")
    a = ap.parse_args()
    if not (a.mirror or a.gate):
        ap.print_help()
        raise SystemExit(2)
    rc = 0
    if a.mirror:
        rc |= mirror_only()
    if a.gate:
        rc |= device_gate(a.time)
    raise SystemExit(rc)
