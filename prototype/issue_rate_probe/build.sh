#!/usr/bin/env bash
# Build the issue-rate probe: one clean binary and one binary per defect.
#
# WHY THE DEFECTS ARE SEPARATE BINARIES. Every `if (IRP_NEG == ...)` in irp_kernels.h is a comparison
# against a literal, so it folds out of the shipping binary and the timed loop is instruction-for-
# instruction identical with and without the flag. A runtime --inject branch would have put a select
# inside the counted loop and changed the instruction stream this probe exists to count — the same
# rule prototype/gdn_collapse/run.py states for its negatives.
#
# The clean binary's instruction counts are re-derived from the disassembly on every build
# (isa_census.py --check) and written to irp_isa.gen.h; the bill the host prints comes from that file,
# never from a count of the source text.
set -euo pipefail
cd "$(dirname "$0")"
: "${HIPCC:=hipcc}"
ARCH="${ARCH:-gfx1151}"
HFLAGS="-O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=${ARCH}"

test -x /opt/rocm/bin/rocm-core/rocm_agent_enumerator || echo "note: rocm agent enumerator missing" >&2

# 1. The census: compile-only, no device, and it refuses to emit if a hot loop cannot be found.
python3 isa_census.py --emit irp_isa.gen.h --quiet
python3 isa_census.py --check

# 2. The shipping binary.
${HIPCC} ${HFLAGS} -o issue_rate_probe issue_rate_probe.cpp

# 3. One per defect. The name is compiled in so a defect is scored only against a build that can
#    express it (`IRP_NEG_NAME`), which is what keeps a negative from passing vacuously.
declare -A NEG=(
  [drop_chain]=1     # A family: the last chain is not computed
  [kv_short]=2       # kv: the last row of the grid is not read
  [kv_alias]=3       # kv: the last warp re-reads row 0 instead of its own last row
  [kv_stray]=4       # kv: a read lands in the guard band past the buffer
  [lds_slot_off]=5   # lds: two readers, one slot (not a permutation)
  [lds_short]=6      # lds: the last row of the tile is not read
  [lds_swap]=7       # lds: a pure permutation: invisible to the block sum, caught per thread
)
for name in "${!NEG[@]}"; do
  ${HIPCC} ${HFLAGS} -DIRP_NEG="${NEG[$name]}" -DIRP_NEG_NAME="\"${name}\"" \
      -o "neg_${name}" issue_rate_probe.cpp
done

# 4. Nothing above may run a kernel. This proves it: the census is compile-only and every binary built
#    here has touched no device. The gate driver (run.py) is what talks to the device, and it does not
#    take the GPU lock — see its header comment for the deadlock it would otherwise cause.
echo "BUILD ok: issue_rate_probe + ${#NEG[@]} defect binaries"
