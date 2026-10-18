#!/usr/bin/env bash
# build_m16.sh — compile the M = 16 re-tile and print the compiler's own resource census.
#
# Separate from `build.sh` on purpose: that one builds `attn_m2`, the artifact of §1-§13, and this
# directory must keep building its predecessor unchanged while the M = 16 tax is being priced.
#
# -O3 is not optional (never benchmark an -O0 build; prototype/README.md), the rpath is what lets the
# binary load libamdhip64.so.7, and --offload-arch=gfx1151 is what makes the resource remarks below be
# the SHIPPED kernel's rather than a fat-pack's.
#
# NO LOCK, NO DEVICE: this only compiles.
#
#   bash prototype/attn_m2/build_m16.sh              # GRP = 4 (16 live pairs), the shipped shape
#   GRP=2 bash prototype/attn_m2/build_m16.sh        # 8 live pairs: the register-parity variant
set -euo pipefail
cd "$(dirname "$0")"
GRP=${GRP:-4}
OUT=attn_m16
[ "$GRP" = "4" ] || OUT="attn_m16_grp${GRP}"
DEF=""
[ "$GRP" = "4" ] || DEF="-DATTN_M16_GRP=${GRP}"

echo "\$ hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 ${DEF} -o ${OUT} attn_m16.hip.cpp"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 ${DEF} -o "${OUT}" attn_m16.hip.cpp
echo "built ./${OUT}"

# The compiler's own census, printed at build time because the one number this item turns on is here:
# VGPRs per lane. `AGENTS.md` retired clang's `Occupancy [waves/SIMD]` line as a constant, so the two
# lines printed below are VGPRs and ScratchSize only, and the derived blocks/CU is arithmetic on the
# 65,536-VGPR register file and 4 warps per block — the DRIVER's occupancy number is the authority and
# comes from tools/kernel_occupancy_probe.cpp, which is a device run this script does not take.
echo "\$ hipcc ... -Rpass-analysis=kernel-resource-usage -c attn_m16.hip.cpp   (attn_m16_wave only)"
hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 ${DEF} -Rpass-analysis=kernel-resource-usage \
     -c attn_m16.hip.cpp -o /tmp/attn_m16_census.o 2>&1 \
  | sed -n '/Function Name: _Z13attn_m16_wave/,/^$/p' \
  | sed -E 's/.*remark: +//; s/ \[-Rpass[^]]*\]//' \
  | grep -E '^(Function Name|TotalSGPRs|VGPRs|ScratchSize|VGPRs Spill)' || true
