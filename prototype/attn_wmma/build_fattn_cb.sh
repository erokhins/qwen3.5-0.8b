#!/usr/bin/env bash
# Build the C->B relabelling witness (`fattn_cb_probe.cpp`): one clean binary and one per defect.
#
#   bash prototype/attn_wmma/build_fattn_cb.sh [outdir]
#
# Compile-only — this script must never launch a kernel, so a worker can prepare the gate and hand the
# device pass to the main agent (or run it themselves as a short correctness gate, which is what
# `check_fattn_cb.py` is for).
#
# WHY SEPARATE BINARIES. Every `CB_NEG == ...` test is a comparison against a literal, so it folds out of
# the clean build and the clean `cb_probe` contains the shipping `p_to_operand` and nothing else. A runtime
# `--inject` branch would have put a select inside the very code whose register allocation the probe
# depends on — the same rule `prototype/issue_rate_probe/build.sh` states for its defects.
#
# WHY THE LISTING IS A BUILD FAILURE AND NOT A NOTE. A probe that stopped doing matrix work would report a
# clean empty result and call it a pass, so the witness is per-kernel and per-binary: `cb_probe`'s own body
# must contain 32 `v_wmma_f32_16x16x16_f16` (16 score + 16 P*V, `KTILES = VTILES = 16`) and
# `private_segment_size 0`. Counting inside `cb_probe` rather than in the whole file matters: the included
# source also compiles `m16_fattn_score` and its legs, and a file-wide count would pass even if the probe's
# own matrix work had been optimised away — the `wl_empty_i1` vacuity trap in
# `prototype/issue_rate_probe/build_wmma_latency.sh`, same shape.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wmma_cb}
mkdir -p "$out"
cd "$out"

ARCH="${ARCH:-gfx1151}"
declare -A NEG=(
  [clean]=0                # the shipping p_to_operand, nothing else
  [lane_half_swap]=1       # butterfly writes the partner's slot
  [pack_swap]=2            # make_half2 argument order
  [elem_reverse]=3         # register order inside a lane's 8 values
  [no_shfl]=4              # the cross-lane exchange replaced by the lane's own value
  [score_slot_swap]=5      # the pre-correction score leg: the regression, as a case
  [vrow_off]=6             # pv_leg tile -> row map off by 8
  [kv_both_relabel]=7      # the derived BLIND case: cancels, must still PASS
  [v_odd_kv_zero]=8        # prediction test for the parity reading cb_no_shfl measured
)

for name in "${!NEG[@]}"; do
    hipcc -O3 --offload-arch="$ARCH" --save-temps -Wno-unused-command-line-argument \
          -DCB_NEG="${NEG[$name]}" -I"$repo" \
          -c "$here/fattn_cb_probe.cpp" -o "cb_${name}.o"
    # -Wl,-rpath is not decoration: without it the linked binary dies at exec with
    # "error while loading shared libraries: libamdhip64.so.7" (the standing HIP-build trap in AGENTS.md).
    hipcc --offload-arch="$ARCH" -Wl,-rpath,/opt/rocm/lib "cb_${name}.o" -o "cb_${name}"
    rm -f "cb_${name}.o"
    # `--save-temps` names the intermediate after the SOURCE, not after -o, so seven builds would all
    # overwrite one listing and only the last defect would ever have been witnessed. Move it aside.
    mv "fattn_cb_probe-hip-amdgcn-amd-amdhsa-gfx1151.s" "cb_${name}.s"
done
rm -f fattn_cb_probe-hip-amdgcn-amd-amdhsa-gfx1151.* fattn_cb_probe-host-x86_64-unknown-linux-gnu.* \
      fattn_cb_probe.cpp-hip-amdgcn-amd-amdhsa.hipfb

# ---- witnesses, per binary, inside cb_probe's own body --------------------------------------------
for name in "${!NEG[@]}"; do
    s="cb_${name}.s"
    test -f "$s" || { echo "ABORT: no listing for cb_${name}" >&2; exit 1; }
    awk '/^cb_probe:/{f=1} f && /^\.Lfunc_end/{f=0} f' "$s" > "cb_probe_${name}.body.s"
    lines=$(wc -l < "cb_probe_${name}.body.s")
    [ "$lines" -ge 40 ] || {
        echo "ABORT: cb_probe body in cb_${name} is $lines lines — the extractor matched nothing, so" >&2
        echo "       the matrix-work count below would be vacuously satisfiable." >&2; exit 1; }
    n=$(grep -c "v_wmma_f32_16x16x16_f16" "cb_probe_${name}.body.s" || true)
    # SITES, not executions — the denominator question AGENTS.md (D60) insists on. `cb_probe`'s body holds
    # 17 v_wmma SITES: 1 in the rolled score loop (`#pragma unroll 1`, executed KTILES = 16 times per
    # round) and 16 in the fully unrolled P*V chains. 17 sites, 32 executions per round, and the census
    # that prices the round reports the second while the listing shows the first. The floor is on SITES
    # because that is what a deletion would remove.
    [ "$n" -ge 17 ] || {
        echo "ABORT: cb_${name}'s cb_probe contains $n v_wmma_f32_16x16x16_f16 sites; the form needs 17"
        echo "       (1 rolled score site + 16 unrolled P*V sites). A probe whose matrix work was deleted"
        echo "       would report zeros and look like a numerical failure, not like a broken probe." >&2
        exit 1; }
    # Accumulators in registers, not scratch. The `#pragma unroll 1` failure is the reason this is checked
    # per binary: a defect that pushes the 16 output tiles into private memory changes what is measured.
    # The metadata block is INDENTED, and an awk pattern anchored to column 1 would match nothing and make
    # this check pass by finding no private segment at all — the empty-parse trap, in the checker itself.
    priv=$(awk '/^[ \t]*\.name:/{n=$2} n=="cb_probe" && /^[ \t]*\.private_segment_fixed_size:/{print $2; exit}' "$s")
    [ -n "$priv" ] || {
        echo "ABORT: could not read cb_probe's .private_segment_fixed_size from $s — the metadata pattern did"
        echo "       not match, so a '0' here would have meant nothing." >&2; exit 1; }
    [ "$priv" = "0" ] || {
        echo "ABORT: cb_${name} cb_probe private_segment_size=${priv} B/thread: the accumulators are in" >&2
        echo "       scratch, so the relabelling being witnessed is a memory round trip." >&2; exit 1; }
    vgpr=$(awk '/^[ \t]*\.name:/{n=$2} n=="cb_probe" && /^[ \t]*\.vgpr_count:/{print $2; exit}' "$s")
    [ -n "$vgpr" ] || { echo "ABORT: no .vgpr_count for cb_probe in $s (metadata pattern)" >&2; exit 1; }
    echo "  cb_${name}: v_wmma=${n} sites in ${lines} lines, vgpr=${vgpr}, private=${priv} B"
done

echo "wrote $out/cb_{clean,...,kv_both_relabel,v_odd_kv_zero} — 8 binaries"
