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


# ----------------------------------------------------------------------------------------------------
# bet G2's census: the wavefront geometry, checked structurally rather than believed.
#
# WHY THIS IS HERE AND NOT IN A NEW FILE: this is the third census over the same kernel family and
# `AGENTS.md` has a trap row about exactly that ("before writing a census tool, list the census
# tools"), and the slicing rules — clang emits a kernel body under a bare column-0 mangled label, and
# the templated instantiations sort under `_Z13rec_wf_kernelILi4…` where the length prefix counts the
# UNMANGLED name — are already written down and tested above. What is new here is only the question: the
# G1 mode asks "did the FP-op multiset survive the split", which is the bit-identity question. G2 does
# not claim bit-identity; it claims a SHAPE, and the shape is checkable in a listing:
#
#   * ZERO `s_barrier` in the walk (the bet's whole point: 32 lanes, so a block IS a wavefront and there
#     is no cross-warp hand-off left to order);
#   * ZERO LDS *data* accesses (`ds_read_*`/`ds_write_*`/`ds_load_*`/`ds_store_*`) — the state stays in
#     global memory and the norms stay in registers;
#   * the cross-lane reduction is a PERMUTE, not a memory round-trip: `ds_bpermute_b32`/`ds_swizzle_b32`
#     move values between lanes through the LDS *unit* without addressing one byte of LDS, which is why
#     the note block's `group_segment_fixed_size` reads 0 while the permute count is positive. So the
#     `lds` bucket above MUST be split, or the claim "0 LDS ops" is either false or vacuous depending on
#     which regex you wrote — and a check that cannot distinguish those two is the empty-parse class of
#     bug in AGENTS.md.
#   * the butterfly must actually be PRESENT (permutes > 0). A kernel that had no cross-lane traffic at
#     all would pass a "0 barriers, 0 LDS" test while being a completely different program.
#   * the per-row k/q operands arrive as UNIFORM loads; whether LLVM scalarised them (`s_load_b32`) or
#     issues a per-lane broadcast `global_load_dword` is a listing fact the A/B has to be read against,
#     so it is reported per kernel rather than asserted in a header.
#   * the state traffic must still be VECTORED: `global_load_dwordx4`/`global_store_dwordx4` for the
#     4-columns-per-lane form. If the float4 became four scalar loads the geometry is the same shape and
#     a different machine cost, and the per-token bill claim in BET.md §10 would be wrong.
# ----------------------------------------------------------------------------------------------------
G2_REF = "rec_loop_kernel"
G2_TARGETS = (("rec_loop_kernel", "block(shipped)"),
              ("rec_wf_kernelILi4", "wf4"),
              ("rec_wf_kernelILi2", "wf2"))
G2_LDS_DATA_RE = re.compile(r"^ds_(read|write|load|store)")
G2_PERM_RE = re.compile(r"^ds_(bpermute|swizzle|sw_)")


def slice_symbol(text: str, needle: str) -> list:
    """One kernel body by a substring of its label — the templated instantiations need this."""
    lines = text.splitlines()
    labels = kernel_labels(text)
    hits = [(i, sym) for i, sym, _ in labels if needle in sym]
    if len(hits) != 1:
        raise SystemExit(f"SLICE FAIL: {needle!r} matched {len(hits)} labels (want exactly 1):"
                         f" {', '.join(s for _, s in hits[:8])}")
    idx = [i for i, (line_no, sym, _) in enumerate(labels) if sym == hits[0][1]][0]
    end = labels[idx + 1][0] if idx + 1 < len(labels) else len(lines)
    return [l.strip() for l in lines[hits[0][0] + 1:end]]


def g2_census(lines: list) -> Counter:
    c = Counter()
    for line in lines:
        if not line or line.startswith((".", ";", "/")):
            continue
        mn = line.split()[0].rstrip(":")
        if BARRIER_RE.match(mn):
            c["barrier"] += 1
        elif G2_LDS_DATA_RE.match(mn):
            c["lds_data"] += 1
        elif G2_PERM_RE.match(mn):
            c["perm"] += 1
        if mn.startswith("s_load"):
            c["s_load"] += 1
        # RDNA3 spells a vector global access `global_load_b128`/`global_store_b64`, NOT the GCN
        # `dwordx4`/`dwordx2` names — the mnemonic-class trap AGENTS.md records for `ds_load_b128`. A
        # census that matches only the old names reports "not vectorised" on a kernel that is, which is a
        # false FAIL, and the only worse outcome than a false PASS.
        if mn.startswith(("global_load_b128", "global_load_dwordx4")):
            c["gld_x4"] += 1
        if mn.startswith(("global_load_b64", "global_load_dwordx2")):
            c["gld_x2"] += 1
        if mn.startswith("global_load_"):
            c["gld_any"] += 1
        if mn.startswith(("global_store_b128", "global_store_dwordx4")):
            c["gst_x4"] += 1
        if mn.startswith(("global_store_b64", "global_store_dwordx2")):
            c["gst_x2"] += 1
        if mn.startswith("global_store_"):
            c["gst_any"] += 1
        if mn.startswith(("v_fma", "v_fmac", "v_dual_fma")):
            c["fma"] += 1
        if mn.startswith(("v_exp", "v_rsq")):
            c["rsq_exp"] += 1
    return c


def g2_report(text: str) -> bool:
    print("G2 STRUCTURAL CENSUS (0 barriers / 0 LDS data ops / butterfly present / state vectorised)")
    cens = {}
    for needle, label in G2_TARGETS:
        cens[label] = g2_census(slice_symbol(text, needle))
    hdr = f"{'kernel':16s} {'barrier':>8s} {'lds_data':>9s} {'perm':>6s} {'s_load':>7s} " \
          f"{'gld_any':>8s} {'gld_x4':>7s} {'gld_x2':>7s} {'gst_any':>8s} {'gst_x4':>7s} " \
          f"{'gst_x2':>7s} {'fma':>5s} {'rsq_exp':>8s}"
    print(hdr)
    for _needle, label in G2_TARGETS:
        c = cens[label]
        print(f"{label:16s} {c['barrier']:8d} {c['lds_data']:9d} {c['perm']:6d} {c['s_load']:7d} "
              f"{c['gld_any']:8d} {c['gld_x4']:7d} {c['gld_x2']:7d} {c['gst_any']:8d} "
              f"{c['gst_x4']:7d} {c['gst_x2']:7d} {c['fma']:5d} {c['rsq_exp']:8d}")
    ok = True
    ref = cens["block(shipped)"]
    if ref["barrier"] < 3 or ref["lds_data"] == 0:
        ok = False
        print(f"FAIL   the shipped walk reads barrier={ref['barrier']} lds_data={ref['lds_data']};"
              " the reference side must show the machinery the bet removes, or the checks below are"
              " decoration")
    for label in ("wf4", "wf2"):
        c = cens[label]
        if c["barrier"]:
            ok = False
            print(f"FAIL   {label} still issues {c['barrier']} `s_barrier`")
        if c["lds_data"]:
            ok = False
            print(f"FAIL   {label} still issues {c['lds_data']} LDS data accesses — the state or the"
                  " norms are being staged, which BET.md §10 constraint 1 forbids")
        if c["perm"] == 0:
            ok = False
            print(f"FAIL   {label} has no cross-lane permute: the row-norm butterfly is not in the body,"
                  " so this is not the geometry the bet is about")
        if cens[label]["gld_x4"] + cens[label]["gld_x2"] == 0:
            ok = False
            print(f"FAIL   {label} has no vectorised global LOAD: the state is being walked one scalar"
                  " column at a time, so the per-token cost is not the one BET.md §10 priced")
    if not ok:
        print("RESULT walk_isa_diff --g2 FAIL")
    else:
        print("RESULT walk_isa_diff --g2 PASS — the wavefront walk is barrier-free and LDS-free, and it"
              " still reduces across the warp by permute.")
    return ok


def g2_selftest() -> bool:
    """The G2 buckets must separate a permute from an LDS access, and count both ways."""
    fake = """
_Z13rec_wf_kernelILi4EEvPKfS1_S1_S1_S1_PfS2_ii:
    ds_bpermute_b32 v1, v2, v3
    ds_swizzle_b32 v1, v2
    global_load_b128 v[4:7], v5, off
    global_store_b128 v[4:7], v5, off
    s_load_b32 s4, s[4:5], 0x10
    v_fmac_f32_e32 v1, v1, v2
_Z15rec_loop_kernelPKfS0_S0_S0_S0_PfS1_ii:
    s_barrier
    ds_write_b32 v3, v4
    ds_read_b32 v3, v4
_Z9other_onei:
    s_nop 0
"""
    wf = g2_census(slice_symbol(fake, "rec_wf_kernelILi4"))
    ref = g2_census(slice_symbol(fake, "rec_loop_kernel"))
    checks = [
        ("a permute is NOT an LDS data access", wf["lds_data"] == 0),
        ("the permutes are counted (2)", wf["perm"] == 2),
        ("an LDS read/write IS an LDS data access (2)", ref["lds_data"] == 2),
        ("a barrier in the reference is counted", ref["barrier"] == 1),
        ("the RDNA3 vector load name counts (b128)", wf["gld_x4"] == 1 and wf["gst_x4"] == 1),
        ("the scalar load is seen (s_load)", wf["s_load"] == 1),
        ("the slice stops at the next label", wf["barrier"] == 0),
    ]
    for name, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    try:
        slice_symbol(fake, "no_such_kernel")
        raised = False
    except SystemExit:
        raised = True
    print(f"  {'ok  ' if raised else 'FAIL'} a missing symbol raises rather than censusing nothing")
    return all(p for _n, p in checks) and raised


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--g2", action="store_true",
                    help="bet G2's structural census instead of G1's: the wavefront geometry must ship"
                         " 0 barriers and 0 LDS DATA accesses, and must still contain its butterfly")
    args = ap.parse_args()

    if args.selftest:
        print("SELFTEST walk_isa_diff")
        ok = selftest()
        ok = g2_selftest() and ok
        print("RESULT " + ("walk_isa_diff SELFTEST PASS" if ok else "walk_isa_diff SELFTEST FAIL"))
        return 0 if ok else 1
    if not args.asm or not args.asm.exists():
        raise SystemExit("need --asm PATH (see the file header for the hipcc command that makes one)")

    text = args.asm.read_text()
    if args.g2:
        return 0 if g2_report(text) else 1
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
