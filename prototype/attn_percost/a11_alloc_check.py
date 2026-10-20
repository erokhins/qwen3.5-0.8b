"""Does the dispatch reserve VGPR + Accum_VGPR, and what does that do to blocks/CU?

The compile listing says `.vgpr_count 185` for `attn_fattn_wave_u8` and the §53 residency probe read 3 blocks/CU
(9 wf/CU) from the driver, but the pmc dispatch record reserves 248 with the same LDS. The candidate
explanation is that WMMA's D operand lives in Accum VGPRs, which `.vgpr_count` does not include and the
hardware allocator does — and if that is right, the register file binds at 2 blocks/CU, not LDS at 3, which
reopens the question A10-F16-IMPL.md D2 closed.
"""
import csv
import sys

path = sys.argv[1]
seen = set()
REGS_PER_CU = 65536
for r in csv.DictReader(open(path, newline="")):
    k = r.get("Kernel_Name", "")
    if "fattn_wave" not in k:
        continue
    key = (r.get("Dispatch_Id"), k)
    if key in seen:
        continue
    seen.add(key)
    v = int(r["VGPR_Count"])
    av = int(r.get("Accum_VGPR_Count", 0) or 0)
    sg = int(r.get("SGPR_Count", 0) or 0)
    lds = int(r.get("LDS_Block_Size", 0) or 0)
    scratch = int(r.get("Scratch_Size", 0) or 0)
    threads = int(r.get("Workgroup_Size", 96) or 96)
    for name, per_thread in (("vgpr only", v), ("vgpr+accum", v + av)):
        by_reg = REGS_PER_CU // (per_thread * threads) if per_thread else 0
        by_lds = (65536 // lds) if lds else 99
        print(f"{k[:40]:40s} wg={threads} v={v} accum={av} sgpr={sg} lds={lds} scratch={scratch} "
              f"| {name:11s} -> {by_reg} blocks/CU (LDS allows {by_lds}) "
              f"=> {min(by_reg, by_lds)} blocks = {min(by_reg, by_lds) * threads // 32} wf/CU")
    break
