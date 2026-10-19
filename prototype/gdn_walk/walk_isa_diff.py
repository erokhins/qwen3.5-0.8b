#!/usr/bin/env python3
"""walk_isa_diff.py — does the hoisted walk still COMPILE to the shipped walk's arithmetic?

WHY THIS EXISTS. `check_hoist_transcription.py` proves the SOURCE TEXT of the state walk did not move.
That is necessary and not sufficient: what bit-identity is actually owed is the compiled expression tree,
and `AGENTS.md`'s `q̃` row is the case where source-level agreement and device agreement came apart in
the other direction too — the compiler's FMA contraction is a property of what LLVM is handed, and an
`lds` read replaced by a `global` read is exactly the sort of change that could, in principle, let it
reassociate or duplicate something. The device bit-identity gate is the only real arbiter, but it is a
device run, and between "the text matches" and "the bits match" there is a cheap CPU check that would
catch a contraction change without spending a window: count the floating-point instructions each kernel
is compiled to, and require the shipped kernel's FP-op multiset to be exactly the union of the two new
kernels'. Contraction changes the CLASS mix (`v_mul_f32` + `v_add_f32` ↔ `v_fmac_f32_e32`), so the census
moves when contraction moves.

IT ALSO CHECKS THE STRUCTURAL CLAIM, which is the whole bet: `rec_loop_normed_kernel` must contain ZERO
`s_barrier` and ZERO LDS data accesses, while `rec_loop_kernel` contains both. That is what "no barrier
between token t and token t+1" means at the ISA level, and it is checked rather than asserted
(`prototype/gdn_walk/BET.md` §3). Those two counts are unambiguous in a listing, which is why they are the
gate here and the FP mix is only a report.

CPU-only. Reads a `.s` produced by:
    hipcc -O3 -Wno-unused-command-line-argument -I. -Iengine --save-temps=obj \
        -c engine/kernels_gdn.cpp -o tmp/gdn_walk/s/kernels_gdn.o
which leaves `tmp/gdn_walk/s/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s`. Launches nothing, times
nothing, prints no millisecond figure.

Slicing note (the trap `AGENTS.md` records for the third census tool on this kernel family): clang emits a
kernel body under a bare column-0 label whose symbol is the mangled name (`_Z15rec_loop_kernelPKf…:`), so
the slicer matches on the mangled prefix and stops at the next column-0 label.

Usage:
    prototype/gdn_walk/walk_isa_diff.py --asm tmp/gdn_walk/s/kernels_gdn-...-gfx1151.s
    prototype/gdn_walk/walk_isa_diff.py --selftest             # the classifier's own must-catch
Exit 0 = the structural claims hold (zero barriers and zero LDS in the hoisted walk, and a shipped body
big enough that the check could have failed). The FP table is printed as a report, not adjudicated: see
the dual-packet note in `main`.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

SHIPPED = "rec_loop_kernel"
NORMED = "rec_loop_normed_kernel"
PREPASS = "gdn_norm_hoist_kernel"

# The three kernels this bet is about, all present in one binary: `rec_loop_kernel` is the shipped form
# (untouched by the change), so the baseline and the two new forms are read off ONE compiler invocation.
KERNELS = (SHIPPED, NORMED, PREPASS)

FP_RE = re.compile(r"^v_[a-z0-9_]*f32[a-z0-9_]*(_e32|_e64)?$")
# Arithmetic mnemonics, i.e. the ones whose COUNT is a statement about the expression tree. Compares,
# `v_exp_f32`, `v_rsq_f32` and `v_ldexp_f32` are excluded: they are special-function plumbing, they appear
# once per `expf`/`rsqrtf` in both forms, and counting them alongside the arithmetic only muddies the one
# question this census can answer.
# Which mnemonics count as ARITHMETIC (a count of these is a statement about the expression tree).
# Compares, `v_exp_f32`, `v_rsq_f32` and `v_ldexp_f32` are left out: they are special-function plumbing and
# appear once per `expf`/`rsqrtf` in both forms. Nothing gates on this bucket today — see the note in
# `main` about rolled loops and dual packets — but the table it feeds is what a reader weighs.
ARITH_PREFIXES = ("v_add_", "v_mul_", "v_sub_", "v_fma", "v_dual_add_", "v_dual_mul_", "v_dual_fma")
BARRIER_RE = re.compile(r"^s_barrier$")
# On RDNA3 a `__shfl_down_sync` lowers to a LDS swizzle/bpermute, so "cross-lane" and "LDS data" are the
# same census here — which is exactly why the two counts below must BOTH be zero in the hoisted walk.
LDS_RE = re.compile(r"^(ds_[a-z0-9_]+|ds_read[a-z0-9_]*|ds_write[a-z0-9_]*|ds_bpermute_b32"
                    r"|ds_sw_b32|ds_add_u32)$")
GLS_RE = re.compile(r"^global_(load|store)[a-z0-9_]*$")


def kernel_labels(text: str) -> list:
    """Column-0 labels of the form `_Z<...>name<...>:`, in file order."""
    out = []
    for i, line in enumerate(text.splitlines()):
        m = re.match(r"^(_Z[0-9]+([A-Za-z_][A-Za-z0-9_]*)[A-Za-z0-9_]*):", line)
        if m:
            out.append((i, m.group(1), m.group(2)))
    return out


def slice_kernel(text: str, kernel: str) -> list:
    """The instruction lines of one kernel body."""
    lines = text.splitlines()
    labels = kernel_labels(text)
    for idx, (line_no, sym, name) in enumerate(labels):
        # The mangled form is `_Z<len><name><args>`, so `_Z<len><name>` is an unambiguous symbol prefix:
        # the length rules out one name being a prefix of another, and matching on the bare name would let
        # `rec_loop_kernel` also match a longer name that contains it.
        if sym.startswith(f"_Z{len(kernel)}{kernel}"):
            end = labels[idx + 1][0] if idx + 1 < len(labels) else len(lines)
            return [l.strip() for l in lines[line_no + 1:end]]
    raise SystemExit(f"SLICE FAIL: no kernel `{kernel}` in the listing (labels found:"
                     f" {', '.join(s for _, s, _ in labels[:12])})")


def classify(lines: list) -> Counter:
    c = Counter()
    for line in lines:
        if not line or line.startswith((".", ";", "/")):
            continue
        mn = line.split()[0].rstrip(":")
        if FP_RE.match(mn):
            c["fp:" + mn] += 1
            c["fp:TOTAL"] += 1
            if mn.startswith(ARITH_PREFIXES):
                c["arith:TOTAL"] += 1
        if BARRIER_RE.match(mn):
            c["barrier"] += 1
        if LDS_RE.match(mn):
            c["lds"] += 1
        if GLS_RE.match(mn):
            c["global"] += 1
        if mn.startswith("v_exp") or mn == "v_exp_f32_e32":
            c["exp"] += 1
        if mn.startswith("v_rsqrt") or mn.startswith("v_rsq"):
            c["rsqrt"] += 1
    return c


def census(text: str) -> dict:
    return {k: classify(slice_kernel(text, k)) for k in KERNELS}


def selftest() -> bool:
    """The classifier must be able to say 0 and 1, and it must not be reading nothing."""
    fake = """
_Z15rec_loop_kernelPKfS0_i:
    v_fmac_f32_e32 v1, v1, v2
    v_add_f32_e32 v1, v1, v2
    s_barrier
    ds_write_b32 v3, v4
    global_load_b32 v5, v6
_Z9other_onei:
    s_nop 0
"""
    c = classify(slice_kernel(fake, "rec_loop_kernel"))
    ok = True
    checks = [
        ("fma counted", c["fp:v_fmac_f32_e32"] == 1),
        ("add counted", c["fp:v_add_f32_e32"] == 1),
        ("FP total is 2, not 0", c["fp:TOTAL"] == 2),
        ("barrier counted", c["barrier"] == 1),
        ("LDS counted", c["lds"] == 1),
        ("global counted", c["global"] == 1),
        ("the next label ends the slice", "s_nop" not in " ".join(slice_kernel(fake,
                                                                               "rec_loop_kernel"))),
    ]
    for name, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        ok = ok and passed
    try:
        slice_kernel(fake, "no_such_kernel")
        raised = False
    except SystemExit:
        raised = True
    print(f"  {'ok  ' if raised else 'FAIL'} a missing kernel raises rather than slicing nothing")
    ok = ok and raised
    for name, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        ok = ok and passed
    # A contraction change must move the class mix: two ops becoming one is the whole point of the census.
    merged = classify(["v_fmac_f32_e32 v1, v1, v2"])
    split = classify(["v_mul_f32_e32 v1, v1, v2", "v_add_f32_e32 v1, v1, v2"])
    moved = merged != split
    print(f"  {'ok  ' if moved else 'FAIL'} a mul+add pair and an fma are DIFFERENT censuses"
          " (the census can see contraction)")
    return ok and moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("SELFTEST walk_isa_diff")
        ok = selftest()
        print("RESULT " + ("walk_isa_diff SELFTEST PASS" if ok else "walk_isa_diff SELFTEST FAIL"))
        return 0 if ok else 1
    if not args.asm or not args.asm.exists():
        raise SystemExit("need --asm PATH (see the file header for the hipcc command that makes one)")

    text = args.asm.read_text()
    cen = census(text)
    shipped = cen[SHIPPED]
    print(f"LISTING {args.asm.name}")
    print(f"{'class':34s} {'shipped':>9s} {'normed':>9s} {'prepass':>9s} {'sum-shipped':>12s}")
    keys = sorted(set(shipped) | set(cen[NORMED]) | set(cen[PREPASS]))
    ok = True
    for k in keys:
        a, b, c = shipped.get(k, 0), cen[NORMED].get(k, 0), cen[PREPASS].get(k, 0)
        mark = "" if a == b + c else "   <- delta"
        print(f"{k:34s} {a:9d} {b:9d} {c:9d} {b + c - a:12d}{mark}")

    # The structural claims of the bet, checked rather than asserted.
    if shipped["barrier"] < 3:
        ok = False
        print(f"FAIL   the shipped walk shows {shipped['barrier']} barriers; the census expected >= 3,"
              " so the structural check below cannot be believed")
    if shipped["lds"] == 0:
        ok = False
        print("FAIL   the shipped walk shows no LDS traffic: the slicer is reading the wrong body")
    if cen[NORMED]["barrier"] != 0:
        ok = False
        print(f"FAIL   `rec_loop_normed_kernel` still has {cen[NORMED]['barrier']} `s_barrier`s:"
              " the norm machinery is not out of the walk")
    if cen[NORMED]["lds"] != 0:
        ok = False
        print(f"FAIL   `rec_loop_normed_kernel` still has {cen[NORMED]['lds']} LDS accesses: the"
              " ks/qs round-trip is not gone")
    if shipped["fp:TOTAL"] < 50:
        ok = False
        print(f"FAIL   only {shipped['fp:TOTAL']} FP instructions in the shipped body: this census"
              " cannot adjudicate anything")
    print(f"CHECK  shipped walk: {shipped['barrier']} barriers, {shipped['lds']} LDS ops"
          f" | normed walk: {cen[NORMED]['barrier']} barriers, {cen[NORMED]['lds']} LDS ops")

    # The accounting the listing DOES support exactly. The shipped body's three special-function sites
    # (`expf` once, `rsqrtf` twice) and its barrier budget split in a way that says where each half of the
    # prologue went: `block_sum_pair_f32`'s three barriers are the pre-pass's, the two around the `ks`/
    # `qs` writes are simply GONE (not moved), which is the round-trip the bet was about.
    if shipped["exp"] != cen[NORMED]["exp"] + cen[PREPASS]["exp"] \
            or shipped["rsqrt"] != cen[NORMED]["rsqrt"] + cen[PREPASS]["rsqrt"]:
        ok = False
        print(f"FAIL   the special-function sites did not move intact: exp "
              f"{shipped['exp']} vs {cen[NORMED]['exp']}+{cen[PREPASS]['exp']}, rsqrt "
              f"{shipped['rsqrt']} vs {cen[NORMED]['rsqrt']}+{cen[PREPASS]['rsqrt']}")
    else:
        print(f"CHECK  exp {shipped['exp']}={cen[PREPASS]['exp']} in the pre-pass, rsqrt"
              f" {shipped['rsqrt']}={cen[PREPASS]['rsqrt']} in the pre-pass: the norm arithmetic moved,"
              " it was not rewritten")
    shipped_b, normed_b, pre_b = shipped["barrier"], cen[NORMED]["barrier"], cen[PREPASS]["barrier"]
    if shipped_b != pre_b + 2 or normed_b != 0:
        ok = False
        print(f"FAIL   barrier budget does not account: shipped {shipped_b}, pre-pass {pre_b},"
              " expected shipped = pre-pass + 2 (the ks/qs pair, now deleted)")
    else:
        print(f"CHECK  barriers {shipped_b} = {pre_b} (the pair reduction, in the pre-pass) + 2"
              " (the ks/qs write pair, deleted)")

    fp_classes = [k for k in keys if k.startswith("fp:") and k != "fp:TOTAL"]
    deltas = [(k, shipped.get(k, 0), cen[NORMED].get(k, 0) + cen[PREPASS].get(k, 0))
              for k in fp_classes
              if shipped.get(k, 0) != cen[NORMED].get(k, 0) + cen[PREPASS].get(k, 0)]
    # WHY THE FP TABLE IS A REPORT AND NOT A GATE. RDNA3 encodes dual-issue packets as ONE mnemonic
    # (`v_dual_fmac_f32`) that carries an FMA *plus an unrelated second op*, so a listing-level count is
    # not additive across kernels: one `v_dual_fmac_f32` in the shipped body may be an FMA paired with a
    # move, and in the hoisted walk it may be that same FMA paired with a different move. Unfolding them
    # would be a guess about the encoding, and a guess that could be talked into agreeing — which is the
    # failure mode this repo has written up four times. So the census's SOUND claims are the structural
    # ones above (barriers and LDS, which are unambiguous), and the FP table is printed for a human to
    # read: a large shift in the FMA-to-mul mix is a reason to expect the device gate to bite, not a
    # verdict in itself.
    print("FP MIX (report only — dual packets make the counts non-additive across kernels)")
    for k in fp_classes:
        a, b, c = shipped.get(k, 0), cen[NORMED].get(k, 0), cen[PREPASS].get(k, 0)
        if a or b or c:
            print(f"  {k:28s} shipped {a:5d}   normed {b:5d}   prepass {c:5d}   sum-shipped "
                  f"{b + c - a:+d}")
    moved = [k for k, _, _ in deltas]
    print(f"NOTE   {len(moved)} of {len(fp_classes)} FP classes do not line up as a partition"
          f"{': ' + ', '.join(moved) if moved else ''}")
    print("NOTE   both loops are ROLLED (`#pragma unroll 4`), so a body's instruction count depends on"
          " how far LLVM chose to unroll it in each caller, and a dual packet hides a second op inside"
          " one mnemonic. Neither makes a cross-kernel count additive: this table is a MIX report, and"
          " the device gate is what adjudicates the bits.")

    print("RESULT " + ("walk_isa_diff PASS" if ok else "walk_isa_diff FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
