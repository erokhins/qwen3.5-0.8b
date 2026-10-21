#!/usr/bin/env bash
# census.sh — the compile-only witness for the residency claim. CPU only: it reads the code object
# metadata out of the assembly `./build.sh` leaves behind and never touches the device.
#
# WHY THIS FILE EXISTS. The route rests on two numbers in a code object: `attn_core` reserves 38,944 B
# of LDS per block while its source declares 6,176 B, and `attn_core_lean` reserves 6,176 B. On this
# part 64 KiB/CU, 38,944 B is ONE resident block per CU. Those are properties of the build, so the
# check belongs beside the build and must be re-runnable after any compiler change.
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
for b in blocks[1:]:
    name = b.split("\n")[0].strip()
    if not re.match(r"_Z\d+(attn_core|attn_core_lean)8CoreArgs$", name):
        continue
    lds = re.search(r"\.amdhsa_group_segment_fixed_size (\d+)", b)
    vg = re.search(r"\.amdhsa_next_free_vgpr (\d+)", b)
    sg = re.search(r"\.amdhsa_next_free_sgpr (\d+)", b)
    print(f"{name:36s} lds={int(lds.group(1)):6d} B  next_free_vgpr={vg.group(1):>4}"
          f"  next_free_sgpr={sg.group(1)}")
print("residency on 64 KiB LDS/CU, 4 blocks/CU at the 1024-thread ceiling:")
for nm, b in (("attn_core", 38944), ("attn_core_lean", 6176)):
    print(f"  {nm:16s} {b:6d} B -> {65536 // b:2d} blocks/CU by LDS -> "
          f"{min(65536 // b, 4)} block(s)/CU resident")
PY
