#!/usr/bin/env python3
"""A2 census aid: class histogram INSIDE a loop span that `prototype/gdn_walk/g3_isa_bill.py` already found.

This is not a fourth ISA census tool (AGENTS.md, "before writing a census tool, list the census tools").
`g3_isa_bill.py` does the hard part — it locates every backward branch and prints its span — and its per-span
dict only reports a fixed key set (`lds/barrier/load/fma/exp/store`), which cannot see `v_wmma` at all (the
class list predates a matrix rung) and cannot separate the round's 128 accumulator multiplies from its other
float work. This script takes that span as an argument and answers exactly one further question: what is the
matrix-to-non-matrix mix inside it.

    python3 prototype/attn_percost/a2_round_bill.py LISTING.S LOOP_HEX_LO LOOP_HEX_HI [--whole-kernel-only]

Reads a disassembly listing already on disk (carved out of the shipping binary by
`tools/carve_codeobjects.py` + `llvm-objdump -d`, never a --save-temps rebuild — g3_isa_bill.py's docstring).
CPU-only: launches nothing, times nothing. Refuses a span that contains zero instructions.
"""
import argparse
import collections
import re

INSN = re.compile(r"^\s+([a-z_][a-z0-9_.]*)\s(.*?)//\s*([0-9A-Fa-f]{8,16}):")

# Ordered: first prefix wins, so `v_fmac_f32_e32` is fma and not fmul, `ds_bpermute_b32` is permute not lds.
BUCKETS = [
    ("wmma", ("v_wmma", "v_mfma")),
    ("fma32", ("v_fma_f32", "v_fmac_f32_e32", "v_mad_f32", "v_madrs_f32")),
    ("fmul", ("v_mul_f32", "v_mac_f32")),
    ("fadd", ("v_add_f32", "v_fma", "v_addci")),
    ("fmax", ("v_max_f32", "v_min_f32")),
    ("exp", ("v_exp",)),
    ("cvt", ("v_cvt_", "v_pk_cnt", "v_cvt_pk")),
    ("permute_xlane", ("ds_bpermute", "permlane", "v_readfirstlane")),
    ("lds", ("ds_",)),
    ("vmem_load", ("global_load",)),
    ("vmem_store", ("global_store",)),
    ("barrier", ("s_barrier",)),
    ("wait", ("s_waitcnt",)),
    ("delay", ("s_delay_alu",)),
    ("salu", ("s_",)),
    ("move", ("v_mov_b32", "v_dual_mov", "v_readfirstlane")),
    ("select_mask", ("v_cndmask", "v_cselect", "v_cmp_")),
    ("int_vec", ("v_",)),
]


def bucket(mn):
    for name, pats in BUCKETS:
        if mn.startswith(pats):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    ap.add_argument("lo", nargs="?", help="hex address of the loop's first instruction (from g3_isa_bill.py)")
    ap.add_argument("hi", nargs="?", help="hex address of the loop's last instruction")
    a = ap.parse_args()

    addrs, mons = [], []
    for line in open(a.listing):
        m = INSN.match(line)
        if m:
            addrs.append(int(m.group(3), 16))
            mons.append(m.group(1))
    if not mons:
        raise SystemExit("REFUSE: no disassembled instructions in %s" % a.listing)

    if a.lo:
        lo, hi = int(a.lo, 16), int(a.hi, 16)
        sel = [mn for ad, mn in zip(addrs, mons) if lo <= ad <= hi]
        where = "loop %s..%s" % (a.lo, a.hi)
    else:
        sel, where = mons, "whole kernel"
    if not sel:
        raise SystemExit("REFUSE: span %s matched 0 instructions — a bill over an empty list cannot fail" % where)

    c = collections.Counter(bucket(m) for m in sel)
    print("# %s: %d instructions" % (where, len(sel)))
    for name, _ in BUCKETS + [("other", ())]:
        if c.get(name):
            print("#   %-13s %5d  (%.1f %%)" % (name, c[name], 100.0 * c[name] / len(sel)))
    print("#   top mnemonics: " + "  ".join("%s=%d" % kv for kv in
                                            collections.Counter(sel).most_common(12)))
    mat = c.get("wmma", 0)
    if mat:
        print("# matrix-class issue slots %d of %d = %.1f %%; non-matrix slots per matrix slot = %.2f"
              % (mat, len(sel), 100.0 * mat / len(sel), (len(sel) - mat) / float(mat)))
    else:
        print("# NOTE: 0 matrix-class slots in this span. If the span is supposed to hold a v_wmma loop, the "
              "span is wrong — this census does not read that as a rate.")


if __name__ == "__main__":
    main()
