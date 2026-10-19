"""G3 census aid: per-kernel ISA bill for a carved AMDGPU code object.

Answers, for one kernel symbol out of a code object carved out of the binary that actually ran
(`tools/carve_codeobjects.py`): the resource allocation from the object's own amdhsa metadata
(VGPR/SGPR/spills/LDS/scratch), the instruction-class mix over the whole kernel, and the ISA span of
every backward branch — which is how the token loop is located without guessing. The loop span is
what a per-token bill must be read off: llama's state loops have compile-time trip counts and are
fully unrolled, so the only backward branch left in the body is the token loop.

Why the object and not a re-compile: AGENTS.md ("Input is the GPU assembly carved out of the SHIPPING
binary, never a --save-temps re-build").

Canaries, in the spirit of "a check whose input list can be empty cannot fail":
  * a symbol absent from the listing is a hard error, not a zero row;
  * a kernel that disassembles to 0 instructions is a failed disassembly, not a result;
  * if no backward branch is found the line says NO LOOP and no per-token figure is printed.

CPU-only. Reads files, runs llvm-objdump / llvm-readelf. No device, no timing.

    python3 prototype/gdn_walk/g3_isa_bill.py <listing.s> <code-object> <symbol-substring>
"""
import collections
import re
import subprocess
import sys

READELF = "/opt/rocm/llvm/bin/llvm-readelf"

# llvm-objdump's AMDGPU line: "\tmnemonic operands   // ADDR: rawhex"
INSN = re.compile(r"^\s+([a-z_][a-z0-9_.]*)\s(.*?)//\s*([0-9A-Fa-f]{8,16}):")
FUNC = re.compile(r"^([0-9a-fA-F]{16}) <(.*)>:")

CLASSES = [
    ("vmem_load", re.compile(r"^global_load")),
    ("vmem_store", re.compile(r"^global_store")),
    ("lds", re.compile(r"^ds_(read|write|load|store|swizzle|add)")),
    ("xlane_permute", re.compile(r"^ds_bpermute")),
    ("barrier", re.compile(r"^s_barrier")),
    ("shuffle", re.compile(r"^v_permlane")),
    ("fma", re.compile(r"^v_(fma_f32|fmac_f32_e32|mad_f32|madrs_f32)")),
    ("fmul", re.compile(r"^v_mul_f32")),
    ("fadd", re.compile(r"^v_add_f32")),
    ("exp", re.compile(r"^v_exp")),
    ("cvt", re.compile(r"^v_(cvt|cvte|pk_|ptras|rev)")),
    ("xlane_logic", re.compile(r"^v_(lshl|lshr|bfe|and|or|med3|prmt)")),
    ("salu", re.compile(r"^s_")),
    ("vother", re.compile(r"^v_")),
]


def metadata(path):
    """Per-kernel amdhsa metadata, read one ENTRY at a time.

    The YAML puts `.name:` in the MIDDLE of a kernel's block (after `.args`, before `.wavefront_size`),
    so a line-by-line scan that keys on `.name:` mis-attributes every field that precedes it — which is
    exactly how `group_segment_fixed_size` gets billed to the wrong kernel. An entry starts at `- .args:`,
    so chunk on that and read the whole chunk.
    """
    out = subprocess.run([READELF, "-n", path], capture_output=True, text=True).stdout
    info = {}
    for chunk in out.split("\n  - .args:")[1:]:
        m = re.search(r"^\s*\.name:\s*(\S+)", chunk, re.M)
        if not m:
            continue
        fields = {}
        for f in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
                  "group_segment_fixed_size", "private_segment_fixed_size",
                  "max_flat_workgroup_size", "wavefront_size", "kernarg_segment_size"):
            fm = re.search(rf"^\s*\.{f}:\s*(\d+)", chunk, re.M)
            if fm:
                fields[f] = int(fm.group(1))
        info[m.group(1)] = fields
    return info


def slice_function(lines, symbol):
    start = None
    for i, ln in enumerate(lines):
        m = FUNC.match(ln)
        if m and symbol in m.group(2):
            start = i
            base = int(m.group(1), 16)
            name = m.group(2)
            break
    else:
        sys.exit(f"REFUSE: no function matching {symbol!r} in the listing. A census over the wrong "
                 f"symbol is a census over nothing.")
    body = []
    for ln in lines[start + 1:]:
        if FUNC.match(ln):
            break
        m = INSN.match(ln)
        if m:
            body.append((int(m.group(3), 16), m.group(1), m.group(2), ln))
    if not body:
        sys.exit(f"REFUSE: {name} parsed to 0 instructions — failed disassembly, not a zero result.")
    return name, base, body


def main():
    listing_path, co, filt = sys.argv[1], sys.argv[2], sys.argv[3]
    meta = metadata(co)
    lines = open(listing_path).read().splitlines()
    name, base, body = slice_function(lines, filt)
    md = meta.get(name, {})
    if not md:
        print("WARN: no amdhsa metadata block matched this symbol; resource figures come from the "
              "trace's own VGPR/LDS columns instead.")
    print(f"==== {name}")
    print(f"     metadata: {md}")
    cnt = collections.Counter()
    mnem = collections.Counter()
    for addr, op, rest, ln in body:
        for cname, rx in CLASSES:
            if rx.match(op):
                cnt[cname] += 1
                break
        else:
            cnt["other"] += 1
        mnem[op] += 1
    print(f"     {len(body)} instructions; classes: " + "  ".join(f"{k}={v}" for k, v in cnt.most_common()))
    print("     top mnemonics: " + "  ".join(f"{k}={v}" for k, v in mnem.most_common(14)))
    print("     load/store widths: " + "  ".join(
        f"{k}={v}" for k, v in sorted(mnem.items()) if k.startswith(("global_", "ds_"))))

    back = []
    for idx, (addr, op, rest, ln) in enumerate(body):
        if not (op.startswith("s_cbranch") or op.startswith("s_cjump")):
            continue
        tm = re.search(r"\+0x([0-9A-Fa-f]+)>", ln)
        if tm:
            tgt = base + int(tm.group(1), 16)
        else:
            m = re.match(r"\s*(\d+)\s*$", rest)
            if not m:
                continue
            v = int(m.group(1))
            if v >= 32768:
                v -= 65536          # llvm prints the 16-bit displacement unsigned
            tgt = addr + 4 + v
        if tgt >= addr:
            continue
        inner = collections.Counter()
        for a2, op2, _, _ in body:
            if tgt <= a2 <= addr:
                if op2.startswith(("v_fma_f32", "v_fmac_f32_e32", "v_mad_f32")):
                    inner["fma"] += 1
                elif op2.startswith("global_load"):
                    inner["load"] += 1
                elif op2.startswith("global_store"):
                    inner["store"] += 1
                elif op2.startswith("ds_"):
                    inner["lds"] += 1
                elif op2.startswith(("v_permlane", "ds_bpermute")):
                    inner["xlane"] += 1
                elif op2.startswith("v_exp"):
                    inner["exp"] += 1
                elif op2.startswith("s_barrier"):
                    inner["barrier"] += 1
                inner["total"] += 1
        back.append((addr - tgt, tgt, addr, dict(inner)))
    if not back:
        print("     NO LOOP: no backward branch, so no per-token bill can be read off this kernel.")
        return
    back.sort(reverse=True)
    for span, tgt, frm, inner in back[:5]:
        print(f"     loop 0x{tgt:x}..0x{frm + 4:x}  span={span} B  per-iteration ISA: {inner}")


if __name__ == "__main__":
    main()
