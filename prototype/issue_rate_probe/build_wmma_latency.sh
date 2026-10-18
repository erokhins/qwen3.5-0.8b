#!/usr/bin/env bash
# Build the WMMA latency leg (`wmma_latency.cpp`). Compile-only: this script must never launch a kernel,
# so that a worker can prepare the leg and hand the device pass to the main agent.
#
#   bash prototype/issue_rate_probe/build_wmma_latency.sh [outdir]
#
# Leaves `$outdir` (default `tmp/issue_rate_probe_latency`) with the binary and its GCN listing.
#
# WHY THE LISTING IS A BUILD FAILURE, NOT A NOTE. This binary's claim is about one instruction. If the
# compiler turned a WMMA into a scalar fallback, folded a dependent chain into a multiply, or spilled the
# accumulators, the leg would time something else and still print a plausible clock figure. Two recorded
# traps are being guarded: the rolled-accumulator failure that put 128 accumulator registers into 544 B of
# scratch per thread while clang's own remark still said "VGPRs Spill: 0"
# (`prototype/attn_wmma/fattn_score_handover.md` §1), and the rate probe's own rule that a bill must be
# re-derived from the disassembly rather than from the source text (`isa_census.py --check`). The
# accumulator-spill half is checked at RUN time as well, via `hipFuncGetAttributes().local_spilled`,
# because the remark and the attribute are different readers of the same note and only one of them is the
# one the launcher sees.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/issue_rate_probe_latency}
mkdir -p "$out"
cd "$out"

hipcc -O3 --offload-arch=gfx1151 --save-temps -Wno-unused-command-line-argument \
      -c "$here/wmma_latency.cpp" -o wmma_latency.o
# -Wl,-rpath is not decoration: without it the linked binary dies at exec with
# "error while loading shared libraries: libamdhip64.so.7" (the standing HIP-build trap in AGENTS.md).
hipcc --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib wmma_latency.o -o wmma_latency

s=wmma_latency-hip-amdgcn-amd-amdhsa-gfx1151.s
test -f "$s" || { echo "ABORT: no GCN listing produced ($s)" >&2; exit 1; }

# Witness 2 for both widths: the instructions must be there, and there must be enough of them that no leg
# collapsed. Nine legs x their ILP: f32 (1+2+4+8) + f16 (1+2+4+8) + the empty baseline = 30 wmma sites at
# minimum, and LLVM may duplicate a site under unrolling, so the bound is a floor, never an exact count.
for mn in v_wmma_f32_16x16x16_f16 v_wmma_f16_16x16x16_f16; do
    n=$(grep -c "$mn" "$s" || true)
    [ "$n" -ge 15 ] || {
        echo "ABORT: listing has $n '$mn' lines; the four ILP legs of that width need >= 15." >&2
        echo "       A leg whose instruction is missing would time loop overhead and call it matrix time." >&2
        exit 1
    }
    echo "  $mn: $n sites"
done

# The empty baseline must be EMPTY of matrix instructions, or it is not a baseline.
awk '/^wl_empty_i1/{f=1} f&&/^\.Lfunc_end/{f=0} f' "$s" > wl_empty_i1.body.s
if grep -q "v_wmma" wl_empty_i1.body.s; then
    echo "ABORT: wl_empty_i1 contains a v_wmma — the loop-overhead baseline would then contain the thing" >&2
    echo "       it exists to subtract, and every wmma row would be read against itself." >&2
    exit 1
fi
# Non-emptiness, stated separately: an awk range that matched nothing would produce an empty body file and a
# vacuously clean "0 v_wmma" above it — the empty-must-catch trap in `prototype/attn_tile_model.py`, one
# level over. The baseline is only a baseline if the extractor found it.
[ "$(wc -l < wl_empty_i1.body.s)" -ge 20 ] || {
    echo "ABORT: wl_empty_i1.body.s is $(wc -l < wl_empty_i1.body.s) lines; the label range did not match," >&2
    echo "       so the check above was vacuous." >&2; exit 1; }
echo "  wl_empty_i1: 0 v_wmma over $(wc -l < wl_empty_i1.body.s) lines (loop-overhead baseline is clean)"

# Per-kernel register cost, from the assembler metadata, attributed to a name. Printed for the record: the
# runtime also prints vgpr/lds/private per leg from hipFuncGetAttributes, and it is the runtime figure that
# carries the spill witness. clang's -Rpass-analysis occupancy remark is NOT used anywhere here: it prints
# the same constant for every kernel in the TU (AGENTS.md, the counter/occupancy row).
echo "  per-kernel cost, from .amdhsa metadata:"
awk ' /^[ \t]*\.name:/            {name=$2}
      /^[ \t]*\.vgpr_count:/      {vgpr=$2}
      /^[ \t]*\.sgpr_count:/      {sgpr=$2}
      /^[ \t]*\.vgpr_spill_count:/ {print "    " name " vgpr=" vgpr " sgpr=" sgpr " vgpr_spill=" $2}
    ' "$s" | head -12 || true

echo "wrote $out/wmma_latency"
