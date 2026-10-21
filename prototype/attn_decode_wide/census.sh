#!/usr/bin/env bash
# census.sh — the compile-only witness for the residency claim. CPU only: it reads the code object
# metadata out of the assembly `./build.sh` leaves behind and never touches the device.
#
# WHY THIS FILE EXISTS. The route rests on two numbers in a code object: `attn_core` reserved 38,944 B
# of LDS per block while its source declares 6,176 B, and `attn_core_lean` reserved 6,176 B. On this
# part 64 KiB/CU, 38,944 B is ONE resident block per CU. Those are properties of the build, so the
# check belongs beside the build and must be re-runnable after any compiler change.
#
# THE RESIDENCY LINES ARE DERIVED, NOT RECALLED. They used to be a hard-coded pair (38,944 -> 1,
# 6,176 -> 4) sitting under readings parsed from the file, so after the port they would have gone on
# printing "attn_core 38944 B -> 1 block/CU" beside a header that now reserves 6,176 — a census that
# contradicts its own output above it and reads plausible. Every number below comes out of the `.s`.
#
# LDS and the SIMD/wave counts come from KFD, never from a HIP capability query (AGENTS.md, the
# counter-and-occupancy row: a HIP query on this SKU reports a halved CU count and a doubled wave
# ceiling).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
s=$(ls "$here"/attn_decode_wide.hip-hip-amdgcn-amd-amdhsa-*.s 2>/dev/null | head -1)
if [[ -z "$s" ]]; then echo "no generated .s — run ./build.sh (--save-temps=obj)"; exit 1; fi
props=/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties
awk '/lds_size_in_kb|simd_per_cu|max_waves_per_simd|simd_count/{printf "%s=%s ", $1, $2}' "$props"
echo ""
python3 - "$s" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
# The note block is the only place these live, and the kernel name is the mangled symbol: match the
# `.amdhsa_kernel` line exactly and read the fields that follow it, in that block only.
blocks = re.split(r"\n\s*\.amdhsa_kernel ", txt)
WANT = ("_Z9attn_core8CoreArgs", "_Z14attn_core_lean8CoreArgs")
read = {}
for b in blocks[1:]:
    name = b.split("\n")[0].strip()
    if name not in WANT:
        continue
    lds = re.search(r"\.amdhsa_group_segment_fixed_size (\d+)", b)
    vg = re.search(r"\.amdhsa_next_free_vgpr (\d+)", b)
    sg = re.search(r"\.amdhsa_next_free_sgpr (\d+)", b)
    read[name] = (int(lds.group(1)), int(vg.group(1)), sg.group(1))
# A census that found nothing prints a clean nothing, which reads as a pass. The two names are the
# whole point of this file, so their absence is an error exit. (AGENTS.md: a check whose input list
# can be empty cannot fail.)
missing = [n for n in WANT if n not in read]
if missing:
    sys.exit("census refuses to report: no .amdhsa_kernel block for " + ", ".join(missing)
             + " in " + sys.argv[1] + " — rebuild with --save-temps=obj")
for name in WANT:
    lds, vg, sg = read[name]
    print(f"{name:36s} lds={lds:6d} B  next_free_vgpr={vg:>4}  next_free_sgpr={sg}")
print("residency on 64 KiB LDS/CU, 4 blocks/CU at the 1024-thread ceiling (THREADS=256), "
      "derived from the readings above:")
FRIENDLY = {"_Z9attn_core8CoreArgs": "attn_core",
            "_Z14attn_core_lean8CoreArgs": "attn_core_lean"}
for name in WANT:
    lds = read[name][0]
    print(f"  {FRIENDLY[name]:16s} {lds:6d} B -> {65536 // lds:2d} blocks/CU by LDS -> "
          f"{min(65536 // lds, 4)} block(s)/CU resident")
PY
