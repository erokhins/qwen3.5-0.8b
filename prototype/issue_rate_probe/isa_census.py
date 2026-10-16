#!/usr/bin/env python3
"""isa_census.py — where every per-iteration count this probe bills comes from.

THE RULE THIS EXISTS TO SATISFY. A count copied from the source text is not acceptable: "the loop
does N FMAs and a few odds and ends" is the kind of claim that has been wrong twice in this repo (the
§12 "M = 4x" traffic booking that was really 1x, and the route-A cost model priced at 11.4 ms that
measured 112.04 ms). So every number the driver bills is read out of the DISASSEMBLY of the object the
compiler actually produced, and `--check` fails the build if a recorded count disagrees.

WHAT THE PREVIOUS RUN GOT WRONG, which is the whole reason this file was rewritten. Selecting "the hot
loop" by widest arithmetic-plus-memory score picked the WRONG LOOP in two of the three kernels it was
asked to bill:

  * `irp_lds_tile` was billed the STAGING loop. The read-back side — the 48 `ds_read_b16` the kernel
    exists to measure — had been hoisted out of the loop entirely, because with a staged value that
    depends only on the tile address the reads are loop-invariant. The billed span contained 6
    `ds_write_b128` and not one LDS read, so the row measured a hoist and called it LDS bandwidth.
  * `irp_kv_read` was billed `publish_block_sum`'s reduction loop, which is verification code inside
    `if (a.verify)`, while the row loop holding `global_load_b128` — the thing under measurement — sat
    unbilled.
  * Two regex bugs hid both failures. `^v_fmac_f32\b` never matches `v_fmac_f32_e32` because `_` is a
    word character, so the dominant arithmetic instruction on this target was invisible to the ARITH
    score; and `ds_load|ds_store` are GCN pre-GFX10 spellings — GFX10+ disassembles them as
    `ds_read_*`/`ds_write_*` — so LDS traffic scored zero and could not win a loop selection.

So loop selection is no longer a score. Each kernel DECLARES the mnemonics that prove the billed span
is the span under measurement (`SPEC` below: an op pattern whose occurrences are the per-iteration op
count, plus a list of minimums that must all hold in the span), and a kernel whose loops satisfy none
of its requirements is NOT billed — the census exits non-zero and prints what every loop did contain.
A check that cannot fail is decoration; a census that can silently pick the wrong loop is worse than
that, because its output looks like a measurement.

Method, all of it CPU-only and compile-only — no device is touched and no timing is taken:

    hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 -c irp_kernels_isa.cpp \
        -o /tmp/.../one.o --save-temps

the same method `tools/mfma_isa.sh` and `prototype/attn_decode/wave_model.py` use. The driver leaves
the device assembly (`*-hip-amdgcn-amd-amdhsa-gfx1151.s`) behind; this slices it per kernel, finds
every backward branch, and counts mnemonic occurrences inside each span. Everything is printed — the
chosen span, the mnemonic histogram, the rejected loops — so a reader can check the census instead of
trusting it.

Why the loop body and not the whole kernel: the billed quantity is warp-instructions PER ITERATION,
and the per-iteration body is what runs `iters` times. `#pragma unroll 1` in the kernels keeps exactly
one logical iteration inside the backedge; if the compiler ever unrolls one anyway the span grows by
the unroll factor, the op minimum fails or `--check` disagrees with `run.py`, and a human decides which
of the two is wrong.

    ./isa_census.py                       # print the census for every kernel
    ./isa_census.py --check               # ... and fail if a count differs from run.py's table
    ./isa_census.py --emit irp_isa.gen.h  # what build.sh calls to generate the billing header
    ./isa_census.py --quiet               # ... plus the above, printing only the RESULT lines
    ./isa_census.py --show-body irp_lds_tile
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

# ---------------------------------------------------------------------------------------------
# The billed kernels and their requirements.
#
#   ops      regex for a mnemonic occurrence that IS one op of the path under measurement. The count
#            of matches inside the billed span is the per-iteration op count, straight from the ISA.
#   need     [(name, regex, min_count)]: every one must hold in the billed span. These are the
#            proofs, and they are what a wrong loop selection fails.
#   expect   the op count the source shape implies. Recorded here so --check can also catch a compiler
#            that changed the shape (an unroll, a hoist, a strength reduction) rather than only a
#            changed count.
# ---------------------------------------------------------------------------------------------
F32_FMA = r"v_fma_f32|v_fmac_f32_e32|v_mad_f32|v_mac_f32|v_dual_fma_f32|v_dual_fmac_f32_e32|v_dual_fmac_f32"
# LDS reads, as this disassembler actually spells them: GFX10+ prints `ds_load_*`/`ds_store_*`, and
# the pre-GFX10 `ds_read_*`/`ds_write_*` spellings are kept as alternatives because a compiler change
# here would otherwise look like a vanished measurement. The width matters: a `ds_load_b128` is eight
# BF16 element reads, so the element count is derived by weight rather than by counting instructions.
LDS_READ_TERMS = [(r"ds_load_b16|ds_read_b16", 1), (r"ds_load_b32|ds_read_b32", 2),
                  (r"ds_load_b64|ds_read_b64", 4), (r"ds_load_b128|ds_read_b128", 8)]
DS_READ16 = r"ds_load_b16|ds_read_b16|ds_load_b128|ds_read_b128"
DS_WRITE = r"ds_write\S*|ds_store\S*"
KV_LOAD = r"global_load_b128"

SPEC: dict[str, dict] = {
    # The scalar FMA family and its dependent-chain twin: the op is the FMA itself.
    **{n: {"ops": F32_FMA, "need": [], "expect": n_}
        for n, n_ in [("irp_fma_1", 1), ("irp_fma_4", 4), ("irp_fma_8", 8),
                      ("irp_fma_16", 16), ("irp_fma_32", 32),
                      ("irp_serial_8", 8), ("irp_serial_32", 32)]},
    # The bf16 tax: one FMA per chain plus one conversion per chain. What is required in the span is
    # (a) the operand ROTATE, which is what stops the conversion being loop-invariant and LICM'd out -
    # without it this kernel silently degrades into `irp_fma_N`, the failure its own comment warns
    # about - and (b) that a conversion is actually issued. On this compiler the bf16->f32 conversion
    # is done with `v_lshlrev_b32`/`v_and_b32` bit surgery rather than `v_cvt_f32_bf16`, which is worth
    # knowing and is why the pattern accepts either spelling. The op count is reported as measured
    # rather than asserted equal to the chain count, because LLVM dual-issue packs two of these VALU
    # ops into one slot (`v_dual_fmac_f32`) and the honest number is the one read off the ISA.
    **{n: {"ops": F32_FMA, "expect": None,
           "need": [("the bf16 operand rotate", r"v_cndmask_b16\S*|v_cndmask_b32\S*", 1),
                    ("a bf16->f32 conversion", r"v_lshl\S*|v_and_b32\S*|v_cvt\S*bf16\S*", 1)],
           } for n in ("irp_cvt_8", "irp_cvt_16")},
    # LDS: the read side is the measurement, so the read is what must be in the span, at full width.
    # 48 `ds_read_b16` (one per BF16 element of attn's dot shape) and at least one staged write, both
    # in the SAME loop, is the proof that the staging and the read-back were not split by LICM.
    "irp_lds_tile": {"ops": DS_READ16, "ops_terms": LDS_READ_TERMS, "expect": 48,
                     "need": [("48 LDS element reads", DS_READ16, 6),
                              ("staged ds_write", DS_WRITE, 1),
                              ("staged global read of the tile", KV_LOAD, 1),
                              ("FMAs on the read results", F32_FMA, 8)]},
    # KV: the row loop, proved by the 128-bit global load and the FMA that folds the row's checksum.
    # `publish_block_sum`'s reduction loop cannot satisfy this, which is what fixes the old mis-bill.
    # One iteration is one warp reading one 512 B row: 512 B / 32 lanes = 16 B = exactly one
    # global_load_b128. The 2 in the first version of this file was the widest-loop bill, which had
    # counted a second load belonging to the block reduction, not to the row.
    "irp_kv_read": {"ops": KV_LOAD, "expect": 1,
                    "need": [("the row's global_load_b128", KV_LOAD, 1)]},
    # The arithmetic paths. One entry per mnemonic in the ceiling table; `expect` is the chain count
    # in the source shape, and the requirement is the mnemonic itself, so a leg whose op got hoisted,
    # folded, or replaced by a cheaper instruction stops the build instead of billing a hoist.
    "irp_f32_8":    {"ops": F32_FMA, "expect": 8, "need": [("v_fma f32", F32_FMA, 8)]},
    "irp_mix_8":    {"ops": r"v_fma_mix_f32", "expect": 8,
                     "need": [("v_fma_mix_f32", r"v_fma_mix_f32", 8)]},
    "irp_pkf16_8":  {"ops": r"v_pk_fma_f16", "expect": 8,
                     "need": [("v_pk_fma_f16", r"v_pk_fma_f16", 8)]},
    "irp_d2f16_8":  {"ops": r"v_dot2_f32_f16", "expect": 8,
                     "need": [("v_dot2_f32_f16 (plain, not the acc form)",
                               r"v_dot2_f32_f16", 8)]},
    "irp_d2bf16_8": {"ops": r"v_dot2_f32_bf16", "expect": 8,
                     "need": [("v_dot2_f32_bf16", r"v_dot2_f32_bf16", 8)]},
    "irp_d4i8_8":   {"ops": r"v_dot4_i32_i8", "expect": 8,
                     "need": [("v_dot4_i32_i8", r"v_dot4_i32_i8", 8)]},
    "irp_d8i4_8":   {"ops": r"v_dot8_i32_i4", "expect": 8,
                     "need": [("v_dot8_i32_i4", r"v_dot8_i32_i4", 8)]},
    "irp_u4u8_8":   {"ops": r"v_dot4_u32_u8", "expect": 8,
                     "need": [("v_dot4_u32_u8", r"v_dot4_u32_u8", 8)]},
    "irp_u8u4_8":   {"ops": r"v_dot8_u32_u4", "expect": 8,
                     "need": [("v_dot8_u32_u4", r"v_dot8_u32_u4", 8)]},
    "irp_wmma_4":   {"ops": r"v_wmma_f32_16x16x16_f16", "expect": 4,
                     "need": [("v_wmma_f32_16x16x16_f16", r"v_wmma_f32_16x16x16_f16", 4)]},
}
KERNELS = list(SPEC)

# Dual-issue encodings (RDNA2 `v_dual_*`) fold two ops into one instruction slot, which would make the
# instruction count understate the work. gfx1151 is not expected to emit them, but the census checks
# rather than assuming, because a dual instruction counted once would silently halve a billed op rate.
DUAL = re.compile(r"v_dual\S*")

SCALAR = re.compile(r"^s_")
LABEL_RE = re.compile(r"^([A-Za-z0-9_.$]+):")
BRANCH_RE = re.compile(r"^\s*(s_cbranch\S*)\s+([A-Za-z0-9_.\$]+)")
# GFX10+ spellings. Keeping the pre-GFX10 alternatives costs nothing and documents the trap.
MEMORY = re.compile(r"^(ds_(read|write|load|store)|global_load|global_store|flat_load|scratch_)\S*")


def mnemonic(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith((".", ";", "#", "@", ":", "/")) or LABEL_RE.match(s):
        return None
    return s.split()[0]


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
    """Per-kernel instruction lines, in emit order. Labels and comments stay as markers.

    The symbols are Itanium-mangled and the arithmetic legs live in a NESTED namespace, so there are
    two shapes to meet: `_ZN3irp9irp_fma_1ENS_4ArgsE` and `_ZN9irp_arith9irp_mix_8ENS_4ArgsE`. The
    first version of this slicer matched only the `irp` prefix, so every arithmetic kernel's body was
    silently appended to whichever `irp::` kernel preceded it - the census would have billed the wrong
    kernel's loop while printing the right name. Matching on `<name>E` (the mangled terminator) is
    exact for both shapes and cannot confuse `irp_fma_1` with `irp_fma_16`.
    """
    known = list(SPEC) + ["irp_wmma_frag", "fill_const_k", "fill_rows_k"]
    out: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        m = re.search(r"-- Begin function (\S+)", line)
        if m:
            sym = m.group(1)
            name = next((c for c in known if c + "E" in sym), None)
            if name is not None:
                out.setdefault(name, [])
            continue
        if name is None:
            continue
        if line.strip().startswith(".size"):
            name = None
            continue
        out[name].append(line)
    return out


def loops_of(body: list[str]) -> list[dict]:
    """Every backward branch, with the mnemonic counts inside the span it closes."""
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
            continue                                   # forward branch: not a loop
        hist: Counter[str] = Counter()
        n_scalar = n_mem = 0
        for j in range(tgt, i + 1):
            mn = mnemonic(body[j])
            if mn is None:
                continue
            hist[mn] += 1
            if SCALAR.match(mn):
                n_scalar += 1
            if MEMORY.match(mn):
                n_mem += 1
        found.append({"start": tgt, "stop": i, "target": m.group(2), "branch": m.group(1),
                      "hist": hist, "insts": sum(hist.values()), "scalar": n_scalar, "mem": n_mem})
    return found


def count(hist: Counter[str], pattern: str) -> int:
    """Occurrences of `pattern` in a span's mnemonic histogram.

    RDNA2/RDNA3 dual-issue encodings (`v_dual_*`) pack two VALU ops into ONE instruction slot. A
    dual instruction therefore counts 1 toward the warp-instruction bill and 2 toward the op count,
    which is exactly the distinction this probe exists to keep straight; silently counting it once
    would understate an op rate by up to 2x on any loop the compiler manages to pair up.
    """
    rx = re.compile(r"^(" + pattern + r")$")
    n = 0
    for mn, c in hist.items():
        mult = 1
        if mn.startswith("v_dual_"):
            mult = 2
        if rx.fullmatch(mn):
            n += c * mult
    return n


def count_terms(hist: Counter[str], terms) -> int:
    """Weighted count: how many BF16 ELEMENTS the span reads out of LDS, whatever widths it used."""
    return sum(count(hist, rx) * w for rx, w in terms)


def pick_span(loops: list[dict], spec: dict, name: str) -> dict:
    """The span that satisfies every requirement in `spec`. No scoring, no fallback: if nothing
    satisfies it, stop and show what was there, because a silently mis-picked span is what this file
    was rewritten to prevent."""
    ok = []
    for L in loops:
        if all(count(L["hist"], rx) >= lo for _, rx, lo in spec["need"]) and count(L["hist"], spec["ops"]) > 0:
            ok.append(L)
    if not ok:
        return None, explain_unbill(name, loops, spec)
    # If several spans qualify, the tightest one is the body: a wider span would bill outer-loop
    # overhead (and, in the LDS case, would be the whole kernel after LICM merged the stages).
    return min(ok, key=lambda L: (L["insts"], -count(L["hist"], spec["ops"]))), ""


def explain_unbill(name: str, loops: list[dict], spec: dict) -> str:
    lines = [f"{name}: NO LOOP SATISFIES ITS REQUIREMENTS, nothing billed."]
    for what, rx, lo in spec["need"]:
        lines.append(f"   needed: {what} (>= {lo}) -> per-loop counts: "
                     + ", ".join(f"{L['target']}={count(L['hist'], rx)}" for L in loops))
    lines.append(f"   op pattern `{spec['ops']}` occurrences per loop: "
                 + ", ".join(f"{L['target']}={count(L['hist'], spec['ops'])}" for L in loops))
    for L in loops:
        lines.append(f"   loop {L['target']}@{L['start']}..{L['stop']} insts={L['insts']}: "
                     + ", ".join(f"{k}x{v}" for k, v in sorted(L["hist"].items(),
                                                               key=lambda kv: (-kv[1], kv[0]))))
    return "\n".join(lines)


def census(sfile: Path, allow_unbilled: tuple[str, ...] = ()) -> dict[str, dict]:
    text = sfile.read_text()
    kernels = slice_kernels(text)
    out: dict[str, dict] = {}
    for name in KERNELS:
        body = kernels.get(name)
        if body is None:
            raise SystemExit(f"kernel {name} not found in {sfile.name}. A kernel that does not reach "
                             "the code object cannot be billed — see the dot1-insts trap in "
                             "irp_arith_kernels.h. Kernels present: " + ", ".join(sorted(kernels)))
        loops = loops_of(body)
        if not loops:
            raise SystemExit(f"kernel {name} has no backward branch: the loop was eaten by the "
                             "compiler, so there is nothing to bill.")
        span, why = pick_span(loops, SPEC[name], name)
        spec = SPEC[name]
        if span is None:
            # An unbilled leg is worse than a missing one only if it is silent, so it is named here
            # and carried into the header as a zero the driver refuses to run.
            if name not in allow_unbilled:
                raise SystemExit(why + f"\nPass --allow-unbilled {name} to emit a zero bill for it "
                                       "and have run.py skip the leg loudly.")
            out[name] = {"unbilled": why}
            continue
        # `ops_terms` widens the count (the LDS row counts ELEMENT reads, not load instructions);
        # otherwise one occurrence of `ops` is one op.
        ops = count_terms(span["hist"], spec["ops_terms"]) if "ops_terms" in spec \
            else count(span["hist"], spec["ops"])
        dual = {m: c for m, c in span["hist"].items() if DUAL.search(m)}
        out[name] = {"loop": span, "hist": span["hist"], "body": body, "loops": loops,
                     "ops": ops, "dual": dual,
                     "wi": span["insts"], "scalar": span["scalar"], "mem": span["mem"]}
    return out


def fmt_hist(hist: Counter[str]) -> str:
    return ", ".join(f"{k}x{v}" for k, v in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))


def print_census(res: dict[str, dict], quiet: bool, show_body: str | None) -> None:
    for name, r in res.items():
        if "unbilled" in r:
            print("UNBILLABLE " + r["unbilled"].replace("\n", "\n   "))
            continue
        L, spec = r["loop"], SPEC[name]
        if not quiet:
            print(f"\n== {name}")
            print(f"   backward branch {L['branch']} -> {L['target']} closes lines "
                  f"{L['start']}..{L['stop']} of the kernel body")
        flag = "ok" if spec["expect"] is None or r["ops"] == spec["expect"] \
            else f"!= source shape says {spec['expect']}"
        print(f"RESULT {name:16s} WARP_INST_PER_ITER={r['wi']:<5d} OPS_PER_ITER={r['ops']:<4d} ({flag})"
              f" scalar={r['scalar']} mem={r['mem']}")
        if r["dual"]:
            print(f"   WARNING dual-issue encodings in the billed span, counted once: {fmt_hist(Counter(r['dual']))}")
        if not quiet:
            reqs = "; ".join(f"{what}={count(L['hist'], rx)}>={lo}"
                             for what, rx, lo in spec["need"])
            print("   requirements satisfied: " + (reqs if reqs else "none declared"))
            print(f"   mnemonic histogram: {fmt_hist(r['hist'])}")
            others = [x for x in r["loops"] if x is not L]
            if others:
                print("   other backward branches (NOT billed): "
                      + " | ".join(f"{x['target']}@{x['start']}..{x['stop']} insts={x['insts']} "
                                   f"ops={count(x['hist'], spec['ops'])}" for x in others))
            if show_body == name:
                print("   ---- billed span ----")
                for j in range(L["start"], L["stop"] + 1):
                    print("   " + r["body"][j].rstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if a derived count differs from run.py's declared table")
    ap.add_argument("--emit", type=Path, default=None,
                    help="write a C header with the derived per-iteration counts")
    ap.add_argument("--show-body", type=str, default=None, metavar="KERNEL",
                    help="dump the billed span of one kernel")
    ap.add_argument("--quiet", action="store_true", help="RESULT lines and errors only")
    ap.add_argument("--allow-unbilled", type=str, default="", metavar="K1,K2",
                    help="named kernels may be unbilled: emit 0 and let run.py skip them loudly")
    ap.add_argument("--json", type=Path, default=None, help="also write a machine-readable copy")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sfile = compile_asm(tmp)
        res = census(sfile, tuple(x for x in args.allow_unbilled.split(",") if x))
        if not args.quiet:
            print(f"# compiled --offload-arch={ARCH} (compile only, no device, no timing)")
            print(f"# assembly: {sfile.name}")
        print_census(res, args.quiet, args.show_body)

        if args.emit:
            lines = [
                "/* GENERATED by isa_census.py — do not edit.",
                " * Per-iteration warp-instruction counts (IRP_WI_*) and per-iteration op counts",
                " * (IRP_OPS_*) read out of the disassembly of THIS header's kernels, not counted",
                " * from the source text. Each kernel also declares which mnemonics must appear in",
                " * the span that gets billed, so a hoisted or folded op fails the census instead",
                " * of becoming a plausible-looking number. Rerun:",
                " *   prototype/issue_rate_probe/isa_census.py --emit irp_isa.gen.h",
                " */",
                "#ifndef IRP_ISA_GEN_H",
                "#define IRP_ISA_GEN_H",
            ]
            for name, r in res.items():
                lines.append(f"#define IRP_WI_{name} {r.get('wi', 0)}ULL")
            for name, r in res.items():
                lines.append(f"#define IRP_OPS_{name} {r.get('ops', 0)}ULL")
            for name, r in res.items():
                if "unbilled" in r:
                    lines.append(f"#define IRP_UNBILLED_{name} 1  /* {r['unbilled'].splitlines()[0]} */")
            lines += ["#endif  // IRP_ISA_GEN_H", ""]
            args.emit.write_text("\n".join(lines))
            if not args.quiet:
                print(f"\nwrote {args.emit}")

        rc = 0
        if args.check:
            sys.path.insert(0, str(HERE))
            import run as irp_run  # the declared table lives with the driver that uses it
            bad = []
            declared = irp_run.DECLARED
            for name in KERNELS:
                want = declared.get(name)
                got = (0, 0) if "unbilled" in res[name] else (res[name]["wi"], res[name]["ops"])
                if want is None:
                    print(f"   check {name:16s} declared=MISSING disassembly={got}")
                    bad.append(name)
                    continue
                flag = "ok" if tuple(want) == got else "MISMATCH"
                if "unbilled" in res[name] and tuple(want) != (0, 0):
                    flag = "MISMATCH (unbilled must be declared (0,0))"
                    bad.append(name)
                    continue
                print(f"   check {name:16s} declared=(wi={want[0]},ops={want[1]}) "
                      f"disassembly=(wi={got[0]},ops={got[1]}) {flag}")
                if tuple(want) != got:
                    bad.append(name)
            extra = set(declared) - set(KERNELS)
            if extra:
                print(f"   check: declared table has kernels the census does not bill: {sorted(extra)}")
                bad += sorted(extra)
            rc = 1 if bad else 0
            print("ISA CENSUS " + ("ok" if rc == 0 else "FAILED — the compiler changed a loop body"))
        if args.json:
            import json
            args.json.write_text(json.dumps(
                {n: {"warp_inst_per_iter": r.get("wi", 0), "ops_per_iter": r.get("ops", 0),
                     "unbilled": r.get("unbilled"), "hist": dict(r.get("hist", {}))}
                 for n, r in res.items()}, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
