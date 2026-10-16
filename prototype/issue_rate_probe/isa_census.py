#!/usr/bin/env python3
"""isa_census.py — where the warp-instruction-per-iteration numbers come from.

THE RULE THIS EXISTS TO SATISFY. The brief for this probe says it plainly: a count copied from the
source text is not acceptable, because "the loop does N FMAs and a few odds and ends" is exactly the
kind of claim that has been wrong twice in this repo (the §12 "M = 4x" traffic booking that was 1x,
and the route-A cost model priced at 11.4 ms that measured 112.04 ms). So the number the probe bills
is read out of the DISASSEMBLY of the object the compiler actually produced, and the driver refuses to
run if it disagrees with the number recorded in `run.py`.

Method, all of it CPU-only and compile-only — no device is touched and no timing is taken:

    hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 -c irp_kernels_isa.cpp \
        -o /tmp/.../one.o --save-temps

which is the same method `tools/mfma_isa.sh` and `prototype/attn_decode/wave_model.py` use. The driver
leaves the device assembly (`*-hip-amdgcn-amd-amdhsa-gfx1151.s`) behind; this slices it per kernel,
finds each kernel's HOT LOOP (the backward branch whose span contains the arithmetic), and counts the
instruction lines inside it. Everything is printed: the loop body, the mnemonic histogram, the chosen
span, so a reader can check the census rather than trust it.

Why the loop body and not the whole kernel: the billed quantity is warp-instructions PER ITERATION, and
the per-iteration body is what runs `iters` times. `#pragma unroll 1` in the kernels keeps exactly one
logical iteration inside the backedge; if the compiler ever unrolls one of those loops anyway, the
span grows by the unroll factor, `--check` fails against `run.py`'s declared table, and a human gets
to decide which of the two is wrong.

    ./isa_census.py                 # print the census for every kernel
    ./isa_census.py --check         # ... and fail if a count differs from run.py's declared table
    ./isa_census.py --emit irp_isa.gen.h    # what build.sh calls, to generate the billing header
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = "gfx1151"

# The kernels whose per-iteration warp-instruction count is billed. Anything else in the header
# (fill kernels) is host plumbing and is not measured.
KERNELS = [
    "irp_fma_1", "irp_fma_4", "irp_fma_8", "irp_fma_16", "irp_fma_32",
    "irp_serial_8", "irp_serial_32",
    "irp_cvt_8", "irp_cvt_16",
    "irp_lds_tile", "irp_kv_read",
]

# What makes a span count as the hot loop, in priority order. A loop that only decrements a counter is
# not the work; the loop that carries the arithmetic is.
ARITH = re.compile(r"^(v_fma_f32|v_mad_f32|v_fmac_f32|v_add_f32|v_cvt_f32_bf16|v_cvt_bf16_f32"
                   r"|v_pk_cvt|v_bfe|v_lshl|v_or3|v_cndmask|v_mov_b32)\b")
MEMORY = re.compile(r"^(ds_load|ds_store|global_load|global_store|flat_load)\w*")
# Instructions that are part of the loop's bookkeeping rather than its work, and are therefore counted
# but printed separately: the trip counter and its predicate live in SGPRs on this target.
SCALAR = re.compile(r"^s_")

LABEL_RE = re.compile(r"^([A-Za-z0-9_.$]+):")
BRANCH_RE = re.compile(r"^\s*(s_cbranch\S*)\s+([A-Za-z0-9_.\$]+)")


def compile_asm(tmp: Path) -> Path:
    src = tmp / "irp_kernels_isa.cpp"
    src.write_text('#include "%s"\n' % (HERE / "irp_kernels.h"))
    r = subprocess.run(
        ["hipcc", "-O3", "-Wl,-rpath,/opt/rocm/lib", f"--offload-arch={ARCH}", "-c", str(src),
         "-o", str(tmp / "one.o"), "--save-temps"],
        capture_output=True, text=True, cwd=tmp)
    if r.returncode:
        raise SystemExit("COMPILE FAILED (compile-only, no device touched):\n" + r.stderr[-3000:])
    hits = sorted(tmp.glob(f"*{ARCH}.s"))
    if not hits:
        raise SystemExit("no device assembly emitted; files were: "
                         + ", ".join(p.name for p in tmp.iterdir()))
    return hits[0]


def slice_kernels(text: str) -> dict[str, list[str]]:
    """Per-kernel instruction lines, in emit order. Blank/label/comment lines are kept as markers.

    The symbols are Itanium-mangled inside `namespace irp`, i.e. `_ZN3irp9irp_fma_1ENS_4ArgsE`, so
    the name is recovered from the length-prefixed form rather than by demangling a whole signature.
    """
    out: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        m = re.search(r"-- Begin function _ZN3irp(\d+)", line)
        if m:
            n = int(m.group(1))
            cand = line[m.end():m.end() + n]
            if len(cand) == n and re.fullmatch(r"\w+", cand):
                name = cand
                out[name] = []
                continue
        if name is None:
            continue
        if line.strip().startswith(".size"):
            name = None
            continue
        out[name].append(line)
    return out


def loops_of(body: list[str]) -> list[dict]:
    """Every backward branch, with the instruction span it closes."""
    labels: dict[str, int] = {}
    for i, line in enumerate(body):
        m = LABEL_RE.match(line.strip())
        if m and m.group(1) not in labels:
            labels[m.group(1)] = i
    found = []
    for i, line in enumerate(body):
        m = BRANCH_RE.match(line.strip())
        if not m:
            continue
        tgt = labels.get(m.group(2))
        if tgt is None or tgt >= i:
            continue                       # forward branch: not a loop
        n_arith = n_mem = n_inst = n_scalar = 0
        for j in range(tgt, i + 1):
            s = body[j].strip()
            if not s or s.startswith((".", ";", "#", "@", ":")) or LABEL_RE.match(s):
                continue
            n_inst += 1
            if SCALAR.match(s):
                n_scalar += 1
            if ARITH.match(s):
                n_arith += 1
            if MEMORY.match(s):
                n_mem += 1
        found.append({"start": tgt, "stop": i, "target": m.group(2), "branch": m.group(1),
                      "insts": n_inst, "arith": n_arith, "mem": n_mem, "scalar": n_scalar})
    return found


def pick_hot(loops: list[dict]) -> dict | None:
    """The hot loop is the one that does the arithmetic; ties go to the OUTERMOST (widest) span,
    because a nested loop inside the work loop is bookkeeping, not a second body."""
    if not loops:
        return None
    best = max(loops, key=lambda L: (L["arith"] + L["mem"], L["insts"]))
    outer = [L for L in loops
             if L["start"] <= best["start"] and L["stop"] >= best["stop"]
             and L["arith"] + L["mem"] >= best["arith"] + L["mem"]]
    return min(outer, key=lambda L: L["insts"]) if outer else best


def census(sfile: Path) -> dict[str, dict]:
    text = sfile.read_text()
    kernels = slice_kernels(text)
    out: dict[str, dict] = {}
    for name in KERNELS:
        body = kernels.get(name)
        if body is None:
            raise SystemExit(f"kernel {name} not found in {sfile.name}; saw "
                             + ", ".join(sorted(kernels)))
        loops = loops_of(body)
        hot = pick_hot(loops)
        if not loops:
            raise SystemExit(f"kernel {name} has no backward branch: nothing was billed. "
                             "Did the compiler eat the loop?")
        hist: Counter[str] = Counter()
        for j in range(hot["start"], hot["stop"] + 1):
            s = body[j].strip()
            if not s or s.startswith((".", ";", "#", "@", ":")) or LABEL_RE.match(s):
                continue
            hist[s.split()[0]] += 1
        out[name] = {"loop": hot, "hist": hist, "body": body, "loops": loops}
    return out


def print_census(res: dict[str, dict], show_body: bool) -> None:
    for name, r in res.items():
        L = r["loop"]
        print(f"\n== {name}")
        print(f"   backward branch {L['branch']} -> {L['target']} closes lines "
              f"{L['start']}..{L['stop']} of the kernel body")
        print(f"   WARP_INST_PER_ITER = {L['insts']}   (of which {L['scalar']} s_*, "
              f"{L['arith']} arithmetic-ish, {L['mem']} memory)")
        print("   mnemonic histogram: " + ", ".join(f"{k}x{v}" for k, v in
                                                    sorted(r["hist"].items(),
                                                           key=lambda kv: (-kv[1], kv[0]))))
        others = [x for x in r["loops"] if x is not L]
        if others:
            print("   other backward branches in this kernel (not billed): "
                  + ", ".join(f"{x['target']}@{x['start']}..{x['stop']} insts={x['insts']} "
                              f"arith={x['arith']}" for x in others))
        if show_body:
            print("   ---- loop body ----")
            for j in range(L["start"], L["stop"] + 1):
                print("   " + r["body"][j].rstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if a derived count differs from run.py's declared table")
    ap.add_argument("--emit", type=Path, default=None,
                    help="write a C header with the derived per-iteration counts")
    ap.add_argument("--show-body", action="store_true", help="dump each hot loop's assembly")
    ap.add_argument("--json", type=Path, default=None, help="also write a machine-readable copy")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sfile = compile_asm(tmp)
        res = census(sfile)
        print(f"# compiled --offload-arch={ARCH} (compile only, no device, no timing)")
        print(f"# assembly: {sfile.name}")
        print_census(res, args.show_body)

        if args.emit:
            lines = [
                "/* GENERATED by isa_census.py — do not edit.",
                " * Per-iteration warp-instruction counts read out of the disassembly of THIS",
                " * header's kernels, not counted from the source text. Rerun:",
                " *   prototype/issue_rate_probe/isa_census.py --emit irp_isa.gen.h",
                " */",
                "#ifndef IRP_ISA_GEN_H",
                "#define IRP_ISA_GEN_H",
            ]
            for name, r in res.items():
                lines.append(f"#define IRP_WI_{name} {r['loop']['insts']}ULL")
            lines += ["#endif  // IRP_ISA_GEN_H", ""]
            args.emit.write_text("\n".join(lines))
            print(f"\nwrote {args.emit}")

        rc = 0
        if args.check:
            sys.path.insert(0, str(HERE))
            import run as irp_run  # the declared table lives with the driver that uses it
            bad = []
            for name, want in irp_run.DECLARED_WI.items():
                got = res[name]["loop"]["insts"]
                flag = "ok" if got == want else "MISMATCH"
                print(f"   check {name:16s} declared={want:5d} disassembly={got:5d} {flag}")
                if got != want:
                    bad.append(name)
            missing = set(KERNELS) - set(irp_run.DECLARED_WI)
            if missing:
                print(f"   check: kernels with no declared count: {sorted(missing)}")
                bad += sorted(missing)
            rc = 1 if bad else 0
            print("ISA CENSUS " + ("ok" if rc == 0 else "FAILED — the compiler changed a loop body"))
        if args.json:
            import json
            args.json.write_text(json.dumps(
                {n: {"warp_inst_per_iter": r["loop"]["insts"], "hist": dict(r["hist"])}
                 for n, r in res.items()}, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
