#!/usr/bin/env python3
"""loop_bill.py — per-loop instruction bill for one kernel of a `hipcc --save-temps` listing.

CPU-only: reads a `.s`, launches nothing, times nothing, prints no ms.

Slicing rules are the three AGENTS.md census rows, all of them applied:
  * a kernel body runs from its bare `_Z...:` label to the next such label (clang emits bodies
    under that column-0 label, not under `.globl`);
  * loop membership comes from LLVM's own annotations, which appear in THREE spellings here:
    on a `; %bb.N:` label line, on a standalone comment line right after a `.LBBN_M:` label, and
    as `=>This Loop Header: Depth=N` on the header block's own label. A parser that reads only one
    spelling under-bills the loop, which is the empty-parse failure class rather than a smaller
    answer;
  * mnemonics are matched as RDNA3 emits them (`v_fmac_f32_e32`, `v_dual_fmac_f32`,
    `global_load_ushort`, `v_ds_shfl_b32`, `s_waitcnt`).

A loop's bill is the instruction count of every basic block LLVM places inside it — a STATIC bill,
so blocks the geometry skips (the `if (w < nt)` / `if (j == row_pos)` arms) are counted once per
trip. It is an upper bound on what a trip issues, and it is the number to compare against
`prototype/issue_rate_probe`'s issue capacity, never to quote as a measurement.

Usage: loop_bill.py <file.s> <kernel-name-without-mangling-count> [mnemonics]
"""
import collections
import re
import sys

CATS = [
    ("vmem", r"^global_(load|store)|^buffer_"),
    ("lds", r"^ds_(read|write|load|store|bpermute|swizzle|add)"),
    ("shuffle", r"shfl"),
    ("exp", r"^v_exp_f32"),
    ("rcp_div", r"^v_(rcp|div)"),
    ("fma32", r"^v_fma|^v_fmac|^v_dual_fmac"),
    ("add_f32", r"^v_(add|sub)_f32|^v_add3"),
    ("mul_f32", r"^v_mul_f32"),
    ("mov", r"^v_mov_b32|^v_dual_mov"),
    ("cvt", r"^v_cvt"),
    ("maxmin", r"^v_(max|min)"),
    ("cmp", r"^v_cmp"),
    ("barrier", r"^s_barrier"),
    ("sched", r"^s_waitcnt|^s_delay|^s_nop"),
    ("cbranch", r"^s_cbranch|^s_branch|^s_cselect"),
    ("salu", r"^s_"),
]
LABEL = re.compile(r"^(?:\.LBB\d+_\d+|; %bb\.\d+):")
LOOPON = re.compile(r"in Loop: Header=(BB\d+_\d+) Depth=(\d+)")
HDR = re.compile(r"=>This Loop Header: Depth=(\d+)")
INSN = re.compile(r"([a-z][\w.]*)")


def cat(m):
    for name, pat in CATS:
        if re.match(pat, m):
            return name
    return "other"


def kernel_body(lines, kernel):
    start = next(i for i, ln in enumerate(lines)
                 if re.match(r"^_Z\d+" + re.escape(kernel) + r"\w*:", ln))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^_Z\d+\w+:", lines[j]):
            end = j
            break
    return lines[start:end]


def bill(body):
    """[(loop_key, [mnemonic,...])] in listing order; loop_key None = outside any loop."""
    blocks = []
    cur, curloop, i = [], None, 0
    while i < len(body):
        s = body[i].strip()
        if LABEL.match(s):
            if cur:
                blocks.append((curloop, cur))
            cur, m = [], LOOPON.search(s)
            curloop = (m.group(1), int(m.group(2))) if m else None
            i += 1
            while i < len(body) and body[i].strip().startswith(";"):
                t = body[i].strip()
                m = LOOPON.search(t)
                if m:
                    curloop = (m.group(1), int(m.group(2)))
                i += 1
            continue
        m = HDR.search(s)
        if m and re.match(r"^\.LBB\d+_\d+:", s):
            hdr = "BB" + s.split(".LBB", 1)[1].split(":", 1)[0].split("_", 1)[1].join(
                ("", ""))
            hdr = "BB" + s.split(".LBB", 1)[1].split(":", 1)[0]
            curloop = (hdr, int(m.group(1)))
            i += 1
            continue
        mm = INSN.match(re.sub(r";.*", "", s).strip())
        if mm and not s.startswith((".", ";", "@")):
            cur.append(mm.group(1))
        i += 1
    if cur:
        blocks.append((curloop, cur))
    out = collections.defaultdict(list)
    for loop, insns in blocks:
        out[loop].extend(insns)
    return out


def main(path, kernel, show):
    body = kernel_body(open(path).read().splitlines(), kernel)
    meta = {}
    for f in ("amdhsa_next_free_vgpr", "amdhsa_next_free_sgpr",
              "amdhsa_group_segment_fixed_size", "amdhsa_private_segment_fixed_size",
              "amdhsa_kernarg_size"):
        m = re.search(r"\." + f + r"\s+(\d+)", "\n".join(body))
        if m:
            meta[f.replace("amdhsa_", "")] = int(m.group(1))
    out = bill(body)
    total = sum(len(v) for v in out.values())
    inloop = sum(len(v) for k, v in out.items() if k)
    print("== %s  %s" % (kernel, meta))
    print("   instructions: %d total, %d inside annotated loops, %d outside"
          % (total, inloop, total - inloop))
    for (loop, insns) in sorted(out.items(), key=lambda kv: -len(kv[1])):
        if loop is None:
            print("  (outside loops): %d insns" % len(insns))
            continue
        hdr, depth = loop
        hist = collections.Counter(cat(m) for m in insns)
        print("  loop %s d%d: %5d insns | %s" % (hdr, depth, len(insns),
              " ".join("%s=%d" % kv for kv in sorted(hist.items()))))
        if show:
            print("      " + " ".join("%s:%d" % kv for kv in
                  collections.Counter(insns).most_common(20)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3)
