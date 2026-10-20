"""Read the register/LDS reservation the DRIVER sees, not the LLVM note block.

Why: two fields disagree and only one is authoritative.

  * the note-block line `.vgpr_count: 185` is LLVM's own count, and the notes list kernels in an order that is
    NOT the order of their definitions -- a parser that walks the file and pairs a name with the next
    `.vgpr_count` it finds will silently report another kernel's number. That is how `185 VGPR / 0 spill` got
    into A10 and the ledger for `attn_fattn_wave_u8`.
  * the `.amdhsa_kernel <mangled>` metadata block's `.amdhsa_next_free_vgpr` is what the assembler emits into
    the code-object header and what the KFD reserves. For `attn_fattn_wave_u8` it reads 241, the driver rounds
    that to 248 at its allocation granularity, and `VGPR_Count = 248` is exactly what a pmc dispatch record
    reports (`prototype/attn_percost/a11_alloc_check.py` over `tmp/a11/pmc2/*.csv`). So this field, not the
    note block, is the register reservation.

WHAT THE RESERVATION DOES *NOT* DECIDE is blocks/CU. The first version divided the register file as 65,536 per
CU and concluded "2 blocks/CU, register-bound, LDS allowing 3" for every fattn kernel. Measurement falsifies
that: padding declared LDS 20,480 -> 32,768 costs +3.36 % at 3,072 lanes and +4.42 % at 16,384 with disjoint
per-rep ranges, which is only possible if the kernel was resident at 3 blocks/CU (`evidence/a11_pad_ab.txt`,
ledger §55) -- and 3 blocks x 96 threads x 248 = 71,424 registers therefore fit, so the per-CU file is not
65,536. At least 71,424, and most plausibly **65,536 per SIMD** = 131,072/CU, which is the value used below.
The per-*thread* ceiling of 256 is untouched, and that ceiling -- not the register-file division -- is what
A10's source kill rests on.

Usage: a11_read_vgpr.py <kernel.s> [name-substring]
Prints, per kernel matching the substring (default: fattn): next_free_vgpr, the driver-rounded reservation at
granularity 8, LDS bytes, and the resulting blocks/CU from BOTH the register file (65,536/CU) and LDS (64 KiB).
"""
import re
import sys
import subprocess

path = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else "fattn"
# Registers available to a whole CU for the block-count bound. NOT 65,536 -- see the header: the pad A/B
# (ledger §55) shows 3 x 96 x 248 = 71,424 resident, so the file is bigger. 65,536 per SIMD is the reading
# that fits, so 2 x 65,536 per CU. Printed on every run so a reader re-derives rather than inherits.
REGS_PER_CU = 131072
LDS_PER_CU = 65536

text = open(path, errors="replace").read()
# Demangle the names the metadata blocks carry so the output is readable next to pmc's Kernel_Name.
blocks = re.findall(r"\.amdhsa_kernel\s+(\S+)(.*?)\.end_amdhsa_kernel", text, re.S)
names = []
for mangled, body in blocks:
    m_v = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", body)
    m_l = re.search(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", body)
    if not m_v:
        continue
    names.append((mangled, int(m_v.group(1)), int(m_l.group(1)) if m_l else 0))

if names:
    try:
        demo = subprocess.run(["c++filt"] + [n[0] for n in names], capture_output=True, text=True).stdout.split("\n")
    except Exception:
        demo = []
else:
    demo = []

print(f"{'kernel':<46} {'vGPU':>5} {'alloc':>6} {'LDS':>7} {'blk/CU':>7} {'wf/CU':>6}  binder "
      f"[at {REGS_PER_CU} regs/CU]")
for i, (mangled, vgpr, lds) in enumerate(names):
    pretty = (demo[i].strip() if i < len(demo) and demo[i].strip() else mangled)
    if sub not in pretty and sub not in mangled:
        continue
    alloc = ((vgpr + 7) // 8) * 8
    by_reg = REGS_PER_CU // (alloc * 96)
    by_lds = (LDS_PER_CU // lds) if lds else 99
    blocks_cu = min(by_reg, by_lds)
    binder = "registers" if by_reg <= by_lds else "LDS"
    print(f"{pretty[:46]:<46} {vgpr:>5} {alloc:>6} {lds:>7} {blocks_cu:>7} {blocks_cu*3:>6}  {binder}"
          f"  (regs allow {by_reg}, LDS allows {by_lds})")
