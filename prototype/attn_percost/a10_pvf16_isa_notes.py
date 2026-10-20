#!/usr/bin/env python3
"""Per-kernel register reservation AND instruction tallies for the eight `attn_fattn*` spellings.

Why this exists and `a11_read_vgpr.py` does not already do it: the register claim of the F16 P*V accumulator
(`--attn-fattn-pvf16 1`, `prototype/attn_percost/A10-F16-IMPL.md` D1/D7) is "half the accumulator, 64 VGPR/lane
freed", and the only thing that can adjudicate that is the code object the assembler emitted. Two names are
prefixes of two others once the `_pvf16` spellings exist (`attn_fattn_wave` ⊂ `attn_fattn_wave_pvf16`), so any
reader that greps a name and takes the last line reports the wrong kernel — `a11_read_vgpr.py` itself documents
that trap in its header, and `| tail -1` walks straight into it. This script keys on the FULL mangled name and
prints one row per kernel, always, so a comparison is a diff of two named rows.

The second column set is why the same script counts instructions rather than registers only. D1 of the bet
prices the F16 side as "16 `v_wmma` issues either way and half the VALU in the rescale loop"; the fold being
`v_mul_f32` → `v_mul_f16` is checkable per kernel body, and whether the freed registers show up in the
reservation is checkable in the same pass. If the two disagree — accumulator 64 registers narrower, reservation
 unmoved — that is the finding, and it is printed as one rather than smoothed over.

    python3 prototype/attn_percost/a10_pvf16_isa_notes.py /tmp/pv_new/kernels_attn-*-gfx1151.s
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# One RE per instruction class that matters to this bet, applied to the kernel's own body text.
# `v_mac_f16`/`v_mad_f16` are counted with `v_mul_f16` because LLVM may contract the fold into a multiply-
# accumulate form; on gfx11 `v_fmac_f32_e32` (not `v_fma_f32`) is the spelling that exists, which is the same
# mnemonic-class trap recorded for `ds_load_b128` in AGENTS.md.
CLASSES = {
    "wmma_f32": r"\bv_wmma_f32\w*",
    "wmma_f16": r"\bv_wmma_f16\w*",
    "mul_f32": r"\bv_mul_f32\w*",
    "mul_f16": r"\b(v_mul_f16\w*|v_mac_f16\w*|v_mad_f16\w*|v_pk_mul_f16\w*)",
    "cvt_f32_f16": r"\b(v_cvt_f32_f16\w*|v_cvt_f16_f32\w*|v_pk_cvt_f32_f16\w*|v_cvt_off_f32_i4\w*)",
    "lds": r"\b(ds_read_\w+|ds_write_\w+|ds_read2_\w+|ds_write2_\w+|global_load_\w+)",
    "spill": r"\b(spill_\w+|stack_\w+)",
}


def bodies(text: str):
    """Yield (mangled_label, body_text) for every function body in the listing.

    Clang emits a kernel body under a bare column-0 label (`_Z17attn_fattn_waveu8v...:`), NOT under `.globl`
    (AGENTS.md, the "three census tools" row: a `.globl`-based slicer finds a 57-slot stub and reports it as
    the kernel). Labels starting with `.` are basic blocks and are skipped.
    """
    marks = []
    for m in re.finditer(r"^([A_][A-Za-z0-9_]*):(\s*;.*)?$", text, re.M):
        marks.append((m.start(), m.group(1)))
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        yield name, text[pos:end]


def reservations(text: str):
    """mangled name -> (next_free_vgpr, lds bytes, note spill count) from the `.amdhsa_kernel` blocks."""
    out = {}
    for mangled, body in re.findall(r"\.amdhsa_kernel\s+(\S+)(.*?)\.end_amdhsa_kernel", text, re.S):
        vg = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", body)
        ld = re.search(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", body)
        if vg:
            out[mangled] = (int(vg.group(1)), int(ld.group(1)) if ld else 0, 0)
    return out


def yaml_notes(text: str):
    """mangled name -> (vgpr_count, vgpr_spill_count, private B, group segment B) from the YAML note block.

    Paired INSIDE each `- ` entry, never by walking the file and grabbing the next `.vgpr_count`: that is the
    mis-pairing `a11_read_vgpr.py`'s header records, and it is how `185 VGPR / 0 spill` got attributed to a
    kernel that reserved 241. The note block is LLVM's own claim; the `.amdhsa_kernel` block below is the
    driver's, and both are printed because a spill that the reservation hides is exactly what the spill gate
    exists to catch.
    """
    out = {}
    for entry in re.split(r"\n  - \.args:", text):
        nm = re.search(r"^\s+\.name:\s+(\S+)", entry, re.M)
        vc = re.search(r"^\s+\.vgpr_count:\s+(\d+)", entry, re.M)
        if not nm or not vc:
            continue
        sp = re.search(r"^\s+\.vgpr_spill_count:\s+(\d+)", entry, re.M)
        pv = re.search(r"^\s+\.private_segment_fixed_size:\s+(\d+)", entry, re.M)
        gs = re.search(r"^\s+\.group_segment_fixed_size:\s+(\d+)", entry, re.M)
        out[nm.group(1)] = (int(vc.group(1)), int(sp.group(1)) if sp else -1,
                            int(pv.group(1)) if pv else -1, int(gs.group(1)) if gs else -1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asm")
    ap.add_argument("--filter", default="attn_fattn")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the slicer finds real bodies and the tallies are not all zero")
    a = ap.parse_args()

    text = Path(a.asm).read_text(errors="replace")
    res = reservations(text)
    dem = subprocess.run(["c++filt"] + list(res), capture_output=True, text=True).stdout.split("\n")
    short = {}
    for i, (m, _) in enumerate(res.items()):
        d = dem[i].strip() if i < len(dem) else m
        short[m] = d.split("(")[0].split("::")[-1]

    if a.selftest:
        bad = []
        seen = 0
        for label, body in bodies(text):
            nm = short.get(label)
            if not nm or a.filter not in nm:
                continue
            seen += 1
            n_w = len(re.findall(CLASSES["wmma_f32"], body)) + len(re.findall(CLASSES["wmma_f16"], body))
            if len(body.splitlines()) < 500:
                bad.append(f"{nm}: body is {len(body.splitlines())} lines — the slicer found a stub")
            if n_w == 0:
                bad.append(f"{nm}: zero v_wmma in its own body — a WMMA kernel with no matrix instruction")
        print("selftest: %d %s bodies sliced; %s" % (seen, a.filter, "; ".join(bad) or "no stub, no empty"))
        return 1 if bad or seen == 0 else 0

    cols = list(CLASSES)
    notes = yaml_notes(text)
    print("%-30s %6s %6s %6s %6s %6s %8s   %s"
          % ("kernel", "vGPU", "alloc", "LDS", "n.vgpr", "spill", "private", "  ".join("%9s" % c for c in cols)))
    rows = []
    for label, body in bodies(text):
        nm = short.get(label)
        if not nm or a.filter not in nm:
            continue
        vg, lds, _ = res.get(label, (0, 0, 0))
        nvg, nsp, npv, _n_gs = notes.get(label, (-1, -1, -1, -1))
        tallies = {c: len(re.findall(rx, body)) for c, rx in CLASSES.items()}
        rows.append((nm, vg, lds, nvg, nsp, npv, tallies))
    for nm, vg, lds, nvg, nsp, npv, t in sorted(rows):
        print("%-30s %6d %6d %6d %6d %6d %8d   %s"
              % (nm, vg, ((vg + 7) // 8) * 8, lds, nvg, nsp, npv,
                 "  ".join("%9d" % t[c] for c in cols)))
    if not rows:
        print("no %s bodies found in %s — the slicer matched nothing, which is not a result"
              % (a.filter, a.asm), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
