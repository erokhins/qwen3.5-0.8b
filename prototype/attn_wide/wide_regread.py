#!/usr/bin/env python3
"""Read a kernel's register/LDS/scratch reservation out of an LLVM `.s`, with the BLOCK SIZE as an argument.

This is `prototype/attn_percost/a11_read_vgpr.py` with ONE thing changed and ONE thing added, and it exists in
`prototype/attn_wide/` rather than as an edit to that file because A12's kernels are 256-thread blocks and the
file it descends from multiplies by a hardcoded 96 threads per block (the shipping `attn_fattn_wave_u8`'s
extent). Reusing its parsing and its field choice is the point; the tool list it belongs to is
`a11_read_vgpr.py` (note block vs code object, and why), `a11_alloc_check.py` (the dispatch-record cross-check)
and `prototype/attn_wmma/fattn_residency_read.py` (threads/warps as flags, off the same `.s`).

WHAT IT READS, and why these fields:

  * `.amdhsa_next_free_vgpr` — the register reservation the DRIVER uses. NOT the LLVM note block's
    `.vgpr_count`, which is LLVM's own count, is not in definition order, and is what put a false `185` in
    this repo's docs (AGENTS.md, the note-block/register-file row; `a11_read_vgpr.py`'s header).
  * `.amdhsa_private_segment_fixed_size` — private (scratch) bytes per THREAD. This is the number A12's spill
    budget is stated in (<= 1,024 B/thread, `A12-WIDE-BET.md` §1), and `a11_read_vgpr.py` does not read it.
  * `.amdhsa_group_segment_fixed_size` — declared LDS per block. Dynamic `extern __shared__` shows as 0 here,
    which is why llama's kernel reads 0 and its real 36,096 B has to come from source (`A2-census.md` §1,
    finding F3). Our own kernels declare static `__shared__` arrays, so theirs is the real number.
  * `.amdhsa_max_flat_workgroup_size` — read and printed, so the threads-per-block used for blocks/CU is the
    kernel's own claim rather than a flag someone forgot to pass. `--threads` overrides it for the ladder,
    where the point is what a hypothetical block size WOULD allow.

The register file per CU is NOT 65,536 on this part: `A11-attribution.md` §6-RESULT (a) measured 3 blocks x
96 threads x 248 VGPR = 71,424 resident, so the file is bigger — the reading that fits is 65,536 per SIMD,
i.e. 131,072 per CU. Printed on every line, because a reader must re-derive, not inherit. The per-THREAD
ceiling of 256 is a separate limit and is what the reservation is checked against (`A10-RF-LADDER.md` §3).

Usage:
    wide_regread.py <kernel.s> [--match SUB] [--threads N] [--regs-per-cu N]

Exit code is 1 if nothing matched — a reader that matches zero kernels and prints an empty table is the
failure class this repo keeps meeting (a check whose input list can be empty cannot fail).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

GRAN = 8                   # the driver's VGPR allocation granularity, observed in `a11_alloc_check.py`
DEFAULT_REGS_PER_CU = 131072

LDS_PER_CU = 65536         # KFD `lds_size_in_kb 64`, read not quoted (never `hipGetDeviceProperties`)
PER_THREAD_CAP = 256       # RDNA's own per-thread VGPR ceiling; no code object in this tree exceeds it


def kernels(path):
    """Every `.amdhsa_kernel` block in an `.s`, with the four fields that decide a residency claim."""
    text = open(path, errors="replace").read()
    out = []
    for mangled, body in re.findall(r"\.amdhsa_kernel\s+(\S+)(.*?)\.end_amdhsa_kernel", text, re.S):
        nv = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", body)
        if not nv:
            continue

        def num(field):
            m = re.search(r"\.amdhsa_%s\s+(\d+)" % field, body)
            return int(m.group(1)) if m else 0

        out.append({
            "mangled": mangled,
            "next_free_vgpr": int(nv.group(1)),
            "next_free_sgpr": num("next_free_sgpr"),
            "lds": num("group_segment_fixed_size"),
            "scratch": num("private_segment_fixed_size"),
            "flat_wg": num("max_flat_workgroup_size"),   # absent from a `.s`; present in the ELF metadata
            "uses_private_segment": "1" if re.search(r"\.amdhsa_enable_private_segment 1", body) else "0",
        })
    return out


def demangle(names):
    try:
        return subprocess.run(["c++filt"] + list(names), capture_output=True, text=True).stdout.split("\n")
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asm")
    ap.add_argument("--match", default="", help="substring of the demangled or mangled name")
    ap.add_argument("--threads", type=int, default=0,
                    help="override threads/block for the blocks/CU arithmetic (0 = the object's own "
                         "max_flat_workgroup_size)")
    ap.add_argument("--regs-per-cu", type=int, default=DEFAULT_REGS_PER_CU)
    a = ap.parse_args()

    rows = kernels(a.asm)
    n_blocks = len(rows)
    names = [r["mangled"] for r in rows]
    demo = demangle(names)
    for r, d in zip(rows, demo):
        r["pretty"] = (d.strip() if d and d.strip() else r["mangled"])
    if a.match:
        rows = [r for r in rows if a.match in r["mangled"] or a.match in r["pretty"]]
    if not rows:
        print("REFUSAL: no kernel block in %s matches %r (%d blocks read)"
              % (a.asm, a.match, n_blocks), file=sys.stderr)
        return 1

    print("# %s   [regs/CU %d (per-SIMD reading, see header); LDS/CU %d; per-thread cap %d; granularity %d]"
          % (a.asm, a.regs_per_cu, LDS_PER_CU, PER_THREAD_CAP, GRAN))
    print("%-52s %5s %6s %6s %8s %7s %6s  binder" % ("kernel", "vGPU", "alloc", "over256", "scratch",
                                                     "LDS", "blk/CU"))
    for r in rows:
        thr = a.threads or r["flat_wg"]
        alloc = ((r["next_free_vgpr"] + GRAN - 1) // GRAN) * GRAN
        over = r["next_free_vgpr"] - PER_THREAD_CAP
        if thr:
            by_reg = a.regs_per_cu // (alloc * thr) if alloc else 0
            by_lds = (LDS_PER_CU // r["lds"]) if r["lds"] else 99
            blk = min(by_reg, by_lds)
            binder = "registers" if by_reg <= by_lds else "LDS"
            wf = blk * thr // 32
        else:
            by_reg = by_lds = blk = wf = 0
            binder, thr = "unknown-wgsize", 0
        note = ""
        if r["lds"] == 0:
            note = "  [LDS declared 0: an `extern __shared__` request made at launch is INVISIBLE in the " \
                   "code object — llama's real 36,096 B comes from source, A2-census.md finding F3]"
        if not thr:
            note += "  [no workgroup size in the object: pass --threads N (it is the kernel's " \
                    "__launch_bounds__ in source)]"
        print("%-52s %5d %6d %6s %8d %7d %6s  %s  (%s thr: regs %s, LDS %s -> %s wf/CU)%s"
              % (r["pretty"][:52], r["next_free_vgpr"], alloc,
                 ("+%d" % over) if over > 0 else "no", r["scratch"], r["lds"],
                 blk if thr else "?", binder if thr else "?",
                 thr or "-", by_reg or "-", by_lds or "-", wf or "-", note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
