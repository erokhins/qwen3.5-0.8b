#!/usr/bin/env bash
# Build the gdn_rec MODE harness. -O3 is not optional: this binary exists to A/B two spellings of the
# same prologue, and an -O0 build is a different kernel (AGENTS.md: never benchmark an unoptimized
# build). The rpath is required or the binary dies at load looking for libamdhip64.so.7
# (prototype/README.md), and --offload-arch=gfx1151 matches the Framework Desktop target; without it
# the build fat-packs for everything and the kernel that runs is not the one the census billed.
#
#   ./build.sh                # -> prototype/gdn_rec_isa/gdn_rec_isa (the binary .gitignore already names)
#   SAVE_TEMPS=1 ./build.sh   # also leaves the device .s next to the object, for the census row
#
# -mfma is a HOST flag (-Xcompiler), and it is here for one reason: the CPU reference spells the two
# multiply-add sites the device fuses as explicit fmaf() calls, and without hardware FMA those become
# libm calls — correct, but slow enough that four cases of the mirror would dominate the run. It does
# not touch device codegen; the census row in isa_census.harness.output.txt is the witness that the
# kernel side of this TU is still the shipped 1,523-slot bill.
set -euo pipefail
cd "$(dirname "$0")/../.."

FLAGS=(-O3 -Wno-unused-command-line-argument -Xcompiler -mfma -Wl,-rpath,/opt/rocm/lib
       --offload-arch=gfx1151)
if [[ -n ${SAVE_TEMPS:-} ]]; then
    mkdir -p tmp/gdn_isa
    hipcc "${FLAGS[@]}" --save-temps -c prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp \
        -o tmp/gdn_isa/gdn_rec_isa.o
    # --save-temps names the intermediates after the SOURCE stem in the CWD, not after the -o path
    # (the trap noted in HANDOFF.md for the engine TU: read the .s where it actually landed). Collect them
    # under tmp/ so a build leaves the tree as it found it.
    mv -f gdn_rec_isa.hip-hip-amdgcn-amd-amdhsa-gfx1151.s tmp/gdn_isa/gdn_rec_isa.s
    rm -f gdn_rec_isa.hip-host-x86_64-unknown-linux-gnu.s gdn_rec_isa.hip-*.bc \
        gdn_rec_isa.hip-*.hipi gdn_rec_isa.hip-*.hipfb gdn_rec_isa.hip-*.o \
        gdn_rec_isa.hip-*.out gdn_rec_isa.hip-*.out.resolution.txt
    echo "kept tmp/gdn_isa/gdn_rec_isa.s (census input; disposable)"
fi

hipcc "${FLAGS[@]}" prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp \
    -o prototype/gdn_rec_isa/gdn_rec_isa
echo "built prototype/gdn_rec_isa/gdn_rec_isa"
