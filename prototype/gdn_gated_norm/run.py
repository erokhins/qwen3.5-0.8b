#!/usr/bin/env python3
"""Gate for §04.5's gated RMSNorm: a CPU mirror that needs no device, and a device gate.

    .venv/bin/python prototype/gdn_gated_norm/run.py --mirror     # CPU only, no GPU touched
    .venv/bin/python prototype/gdn_gated_norm/run.py --preflight  # CPU only: plan + name check
    .venv/bin/python prototype/gdn_gated_norm/run.py --gate       # device: run + negatives
    .venv/bin/python prototype/gdn_gated_norm/run.py --selftest   # CPU: fixture round-trip guard
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
which is how the first-ever device run of this gate died. So plan files are written by
`write_plan()`, the only function here that produces one, and it refuses to hand back a plan whose
weight names are not store keys: no launch path can skip the check, because no launch path can get
an unchecked plan. `--preflight` is that same producer-and-guard path with the launch taken out, so
the contract is checkable without a device — which matters, because `--mirror` (the CPU half that
decides the arithmetic) never asks the store for anything and could not have seen the bug.

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


def write_f32_raw(t: torch.Tensor, path: Path) -> None:
    """Write `t`'s F32 payload. The kernel's `o` parameter is `const float*`, so F32 is the file
    format, not a choice — and `mirror_probe.cases()` already hands over `o` as float32, so this is
    a widening no-op rather than a conversion. `astype` copies, which is what makes the naive version
    of this line safe where the BF16 one below was not: numpy has one float32 and one uint16, and it
    cannot reinterpret one as the other by accident. `verify_cases` checks it anyway."""
    assert t.dtype == torch.float32, (path, t.dtype)
    t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).tofile(path)


def write_bf16_raw(t: torch.Tensor, path: Path) -> None:
    """Write `t`'s BF16 payload bytes.

    `.view(torch.uint16)` reinterprets the two bytes; `.to(torch.uint16)` **converts** them, and the
    difference is the whole of results.md §8: `mirror_probe.cases()` yields `z` as float32, so
    `.to(torch.uint16)` truncated 0.195 to the integer 0 and wrapped -1.3 to 0xFFFF, which the kernel
    then read back as +0 and NaN — 1525 zeros and 320 NaN per case, from a pin holding neither. The
    assert is here because `.view` needs equal element sizes, which is exactly the thing that was
    silently wrong when a wider dtype was passed in.
    """
    assert t.dtype == torch.bfloat16, (path, t.dtype)
    t.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(path)


def verify_cases(cases: list[dict]) -> list[str]:
    """Read every fixture file back and prove it holds the tensor that was meant to go in.

    A gate that manufactures its own inputs has to check the manufacturing. This one did not, and the
    first device run is the cost: `write_cases` manufactured NaN out of a clean pin, the gate compared
    that NaN against the pin for 144 cases, and the log blamed the kernel (`ndiff 2048/2048, max ULP
    49 174`) while the kernel was right. Three checks, cheapest failure first:

      length      — a short or long file is a writer bug, not a numerical one.
      NaN         — NaN in the file where the pin has none. Real activations do not contain NaN, and
                    a numeric cast instead of a bitcast is the only way this gate can invent one, so
                    the message says that rather than merely reporting a count.
      bit pattern — every element identical, not merely close: these are payload bytes, so "close"
                    has no meaning here and a one-bit difference is a different tensor.
    """
    problems: list[str] = []
    for c in cases:
        for kind, path, want, size in (("o", c["ofile"], c["src_o"], 2048), ("z", c["zfile"],
                                                                       c["src_z"], 2048)):
            raw = np.fromfile(path, dtype=np.uint16 if kind == "z" else np.float32)
            if raw.size != size:
                problems.append("%s.%s: %d elements on disk, %d expected" % (c["name"], kind,
                                                                             raw.size, size))
                continue
            if kind == "z":
                gb = raw.astype(np.int32)
                wb = bf16_ulp.bf16_bits(want)
                bad_nan = int((((gb & 0x7FFF) >= 0x7F81) & ((wb & 0x7FFF) < 0x7F81)).sum())
            else:
                gb = np.frombuffer(raw.tobytes(), dtype=np.uint32).astype(np.int64)
                wb = np.frombuffer(want.numpy().astype(np.float32).tobytes(),
                                   dtype=np.uint32).astype(np.int64)
                bad_nan = int((np.isnan(raw) & ~np.isnan(want.numpy())).sum())
            if bad_nan:
                problems.append("%s.%s: %d NaN in the file where the pin has none — a numeric cast "
                                "(Tensor.to(uint16)) instead of a bitcast (.view(uint16)) is what "
                                "does that" % (c["name"], kind, bad_nan))
            diff = int((gb != wb).sum())
            if diff:
                first = int(np.flatnonzero(gb != wb)[0])
                problems.append("%s.%s: %d/%d elements differ from the tensor meant to go in "
                                "(first at lane %d: file 0x%x, tensor 0x%x)"
                                % (c["name"], kind, diff, size, first, gb[first] & 0xFFFF if kind
                                   == "z" else gb[first], wb[first] & 0xFFFF if kind == "z"
                                   else wb[first]))
    return problems


def fixture_selftest() -> int:
    """CPU only: prove `verify_cases` catches the writer bug it was written for.

    A guard nobody has seen fail is a guard that can rot. Both halves are required: the good write must
    pass (or the guard rejects everything and looks like it works), and the injected numeric cast must
    be rejected *for the right reason* — the NaN clause, naming the cast.
    """
    o, z, _w, _ref = next(mp.cases(FIXTURE, STEPS))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        case = dict(name="t000", ofile=d / "t000.o.f32", zfile=d / "t000.z.bin",
                    src_o=o.reshape(-1), src_z=z.reshape(-1))
        write_f32_raw(case["src_o"], case["ofile"])
        write_bf16_raw(case["src_z"].to(torch.bfloat16), case["zfile"])
        clean = verify_cases([case])
        print("  bitcast write  -> %s" % ("accepted" if not clean else "REJECTED: %s" % clean))
        # The injection: the writer this gate shipped with, numeric cast on a float32 tensor.
        case["zfile"].unlink()
        case["src_z"].to(torch.uint16).numpy().tofile(case["zfile"])
        broken = verify_cases([case])
        caught = bool(broken) and "numeric cast" in broken[0]
        print("  numeric-cast   -> %s" % ("rejected: %s" % broken[0] if broken else "ACCEPTED"))
    ok = (not clean) and caught
    print("FIXTURE SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def write_cases(d: Path, fixture: str) -> list[dict]:
    """One case per (step, GDN layer): pin values as raw files + the expected outputs.

    The files are written through the two `write_*_raw` helpers and then read back by `verify_cases`,
    which cannot be skipped from here: this is the only producer of gate inputs, so the alternative
    to checking here is remembering to check at every call site.
    """
    fx = ROOT / "data" / "refs" / fixture
    cases = []
    for (o, z, w, ref) in mp.cases(fixture, STEPS):
        l = len(cases)
        name = "c%03d" % l
        of, zf = d / (name + ".o.f32"), d / (name + ".z.bin")
        write_f32_raw(o.reshape(-1), of)
        write_bf16_raw(z.reshape(-1).to(torch.bfloat16), zf)
        cases.append(dict(name=name, ofile=of, zfile=zf, weight=w, ref=ref, layer=None,
                          wname=None, src_o=o.reshape(-1), src_z=z.reshape(-1)))
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
    problems = verify_cases(cases)
    if problems:
        raise SystemExit("fixture writer produced %d bad file(s):\n  %s"
                         % (len(problems), "\n  ".join(problems[:8])))
    print("fixture verified: %d cases read back from disk, %d elements per case, bit-identical to"
          " the pins (o as F32, z as BF16 payload)" % (len(cases), 2048))
    return cases


def plan_rows(c, out, norm):
    """One plan line's fields: `NAME OFILE ZFILE WNAME OUT [NORM]`, the format the binary parses."""
    return (c["name"], c["ofile"], c["zfile"], c["wname"], out, norm)


def write_plan(path: Path, rows) -> Path:
    """The only function here that produces a plan file — and it never hands back an unchecked one.

    Every row's weight name must be a `WeightStore` key (column `WEIGHT_FIELD`) before the path is
    returned, so no launch path can reach `run_binary` with a name the store would refuse: there is
    nothing to remember to call, which is the difference between a guard and a habit.
    """
    with path.open("w") as f:
        for row in rows:
            f.write(" ".join(str(x) for x in row) + "\n")
    wn.check_plan(path, WEIGHT_FIELD)
    return path


def preflight() -> int:
    """CPU only: run the real producer and the real guard, and launch nothing.

    `--gate` needs the device and a quiet box; the naming contract it rests on does not, and
    `--mirror` — the CPU half that decides the arithmetic — never asks the store for anything, so it
    could watch the naming bug ship. This is the check that closes that gap on the CPU.
    """
    print("== pre-flight (no device): assemble the gate's plan and resolve its weight names ==")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cases = write_cases(d, FIXTURE)
        plan = write_plan(d / "plan.txt",
                          [plan_rows(c, d / (c["name"] + ".gated.bin"),
                                      d / (c["name"] + ".norm.bin")) for c in cases])
        n = len(plan.read_text().rstrip().splitlines())
    print("PREFLIGHT:", "PASS" if n == len(cases) else "FAIL (%d rows, %d cases)" % (n, len(cases)))
    return 0 if n == len(cases) else 1


def run_binary(plan: Path, outdir: Path, neg: str = "none") -> None:
    exe = BIN if BIN.is_file() else Path(__file__).parent / "gdn_gated_norm"
    cmd = [str(exe), "--mode", "once", "--plan", str(plan), "--neg", neg]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("  $ " + " ".join(cmd[0:1]) + " --mode once --plan <plan> --neg " + neg)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit("binary failed (rc %d)" % r.returncode)
    for line in r.stdout.strip().splitlines():
        # `loader:`/`plan `/`RESULT` are the historical lines; the three after them are the driver's
        # own account of what reached the device: the weight rows it resolved, and whether the launch
        # actually wrote the buffers it was given (sentinel prefill). The subprocess output is
        # captured, so a line not echoed here is invisible to the gate log.
        if line.startswith(("loader:", "plan ", "RESULT", "weights:", "write-back:", "sentinel:")):
            print("  " + line)


def read_out(path: Path, n: int) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16)[:n]).view(torch.bfloat16)


def print_profile(n_el: int, n_zero: int, n_nan: int, n_pzero: int, n_pnan: int) -> None:
    """Print the zeros/NaN profile of a gate run, and what that profile means.

    `ndiff 2048/2048, max ULP 49174` is the same headline for two different bugs: a +0 against a pin
    of -2.39 is ~49 000 ULP, and so is a large-magnitude value that drifted a few thousand ULP. The
    counts separate them — "the epilogue never saw real values" (weight dtype, wrong pointer, a
    launch that did not run) from "the rounding policy is wrong" — and they change no verdict.
    """
    print("  output profile over %d elements: zeros %d, NaN %d (%.1f %% of the output is neither);"
          " the pins hold zeros %d, NaN %d"
          % (n_el, n_zero, n_nan, 100.0 * (n_el - n_zero - n_nan) / max(n_el, 1), n_pzero, n_pnan))
    if n_zero + n_nan > n_el // 2:
        print("  MOST of the output is zero or NaN: the epilogue never saw real values, so check"
              " what the kernel was handed (weight dtype, pointers, whether the launch ran) before"
              " touching the cast order.")
    elif n_zero + n_nan > 0:
        print("  some zeros/NaN in the output: read the per-case counts above before the ULP figure.")


def device_gate(with_time: bool) -> int:
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cases = write_cases(d, FIXTURE)
        print("== device gate: %d cases (%d GDN layers × %d decode steps of %s) =="
              % (len(cases), len({c['layer'] for c in cases}), len(STEPS), FIXTURE))
        for c in cases:
            c["out"] = d / (c["name"] + ".gated.bin")
            c["norm"] = d / (c["name"] + ".norm.bin")
        plan = write_plan(d / "plan.txt", [plan_rows(c, c["out"], c["norm"]) for c in cases])
        print("\n-- clean run --")
        run_binary(plan, d)
        worst_ulp, worst_ndiff, n_el, tightest = 0, 0, 0, 0.0
        n_zero = n_nan = n_pzero = n_pnan = n_fail_cases = 0
        clean = {}
        for c in cases:
            got = read_out(c["out"], 2048)
            clean[c["name"]] = dict(gated=got, norm=read_out(c["norm"], 2048))
            rep = bf16_ulp.report(got.float(), c["ref"].float(), ulp_max=1, cancel_rel=1e-4)
            zero, nan = bf16_ulp.zero_nan(got)
            p_zero, p_nan = bf16_ulp.zero_nan(c["ref"])
            n_zero += zero
            n_nan += nan
            n_pzero += p_zero
            n_pnan += p_nan
            worst_ulp = max(worst_ulp, rep["maxulp"])
            worst_ndiff = max(worst_ndiff, rep["ndiff"])
            tightest = max(tightest, rep["worst_off_rel"])
            n_el += rep["n"]
            if rep["n_bad"] or rep["ndiff"] > 0.01 * rep["n"]:
                n_fail_cases += 1
                fails.append("%s: ndiff %d/%d maxulp %d n_bad %d | zeros %d nan %d"
                             % (c["name"], rep["ndiff"], rep["n"], rep["maxulp"], rep["n_bad"],
                                zero, nan))
        print("  gated_out vs pins over %d elements: worst case ndiff %d/2048, max ULP %d, "
              "tightest band these results pass at %.1e"
              % (n_el, worst_ndiff, worst_ulp, tightest))
        print_profile(n_el, n_zero, n_nan, n_pzero, n_pnan)
        if worst_ulp > 1:
            fails.append("a gated_out element is more than 1 BF16 ULP from the pin")

        print("\n-- negatives (each must move only the stage it lives in) --")
        for neg, must, must_not in NEGATIVES:
            plan_neg = d / ("plan_%s.txt" % neg)
            for c in cases:
                c["nout"] = d / (c["name"] + ".%s.gated.bin" % neg)
                c["nnorm"] = d / (c["name"] + ".%s.norm.bin" % neg)
            write_plan(plan_neg, [plan_rows(c, c["nout"], c["nnorm"]) for c in cases])
            run_binary(plan_neg, d, neg)
            moved = {s: 0 for s in STAGES}
            neg_zero = neg_nan = 0
            for c in cases:
                g = read_out(c["nout"], 2048)
                nm = read_out(c["nnorm"], 2048)
                # Bit patterns, not `!=`: NaN is not equal to itself, so a float compare charges
                # every NaN position in the clean output as "moved" in every negative, whether the
                # injection touched it or not. The first device run proved the size of that error --
                # its clean output held 66 557 NaNs and `w_f32` reported exactly 66 557 moved, and
                # the other five landed within 57 of it, i.e. all six negatives looked like they
                # moved the same thing. Comparing the uint16 payloads makes an identical NaN pair
                # compare equal, which is the claim `moved` is supposed to be making.
                moved["gated_out"] += int((bf16_ulp.bf16_bits(g)
                                           != bf16_ulp.bf16_bits(clean[c["name"]]["gated"])).sum())
                moved["norm_pre_gate"] += int(
                    (bf16_ulp.bf16_bits(nm)
                     != bf16_ulp.bf16_bits(clean[c["name"]]["norm"])).sum())
                z, na = bf16_ulp.zero_nan(g)
                neg_zero += z
                neg_nan += na
            bad = [s for s in must if moved[s] == 0] + [s for s in must_not if moved[s] > 0]
            verdict = "OK" if not bad else "MISCATCHED " + ",".join(bad)
            # The zeros/NaN pair says the injection moved the value into a wrong-but-real one,
            # which is a different claim from "moved" alone.
            print("  %-14s moved gated_out %7d/294912, norm_pre_gate %7d/294912"
                  " (output zeros %d nan %d) -> %s"
                  % (neg, moved["gated_out"], moved["norm_pre_gate"], neg_zero, neg_nan, verdict))
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
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only: prove the fixture round-trip guard rejects a numeric cast")
    ap.add_argument("--preflight", action="store_true",
                    help="CPU-only: build the real plan and resolve its weight names, launch nothing")
    ap.add_argument("--gate", action="store_true", help="device gate (needs the GPU idle)")
    ap.add_argument("--time", action="store_true", help="with --gate: burst timing too")
    a = ap.parse_args()
    if not (a.mirror or a.preflight or a.gate or a.selftest):
        ap.print_help()
        raise SystemExit(2)
    rc = 0
    if a.mirror:
        rc |= mirror_only()
    if a.selftest:
        rc |= fixture_selftest()
    if a.preflight:
        rc |= preflight()
    if a.gate:
        rc |= device_gate(a.time)
    raise SystemExit(rc)
