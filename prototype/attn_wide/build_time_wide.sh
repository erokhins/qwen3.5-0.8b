#!/usr/bin/env bash
# build_time_wide.sh — build the A12 timing rig (`time_wide.cpp`).
#
#   bash prototype/attn_wide/build_time_wide.sh [outdir]
#
# ONE hipcc invocation produces BOTH the linked binary and the code-object listing the register read parses
# (`--save-temps`, then link the same object), so the listing can never describe a different build than the
# binary it is compared against. That is the point of the arrangement: `engine/build.sh` is not bit-reproducible
# on this box and every self-consistency field inside a binary agrees with itself, so a stale `new` side is only
# catchable by tying the artifact to the listing taken from the same compile (AGENTS.md, 2026-10-17, "a paired
# A/B against a stale binary reads as a clean null"). `run_a12_window.sh` re-checks the timestamps and then
# compares the rig's self-reported LDS against this listing's.
#
# The ladder axes are the same five env vars `build_wide.sh` takes — WIDE_WARPS / WIDE_STAGE / WIDE_QREG /
# WIDE_PVF16 / WIDE_MINB — and they MUST be passed to both scripts in one environment, because `time_wide`
# compares the correctness driver's banner axes with its own and refuses on a mismatch (R1 in time_wide.cpp).
#
# Two refusals, both earned elsewhere in this tree:
#   * the listing must contain BOTH kernels the rig launches. A rig whose baseline kernel was DCE'd out measures
#     one kernel twice and reports it as a pair (`prototype/attn_percost/build_a10_pvf16_map.sh` refuses the same
#     way; `prototype/attn_wide/build_wide.sh` refuses on the wide kernel's WMMA site counts).
#   * `-Wl,-rpath,/opt/rocm/lib` is not decoration (the standing HIP-build trap).
#
# `--save-temps` writes into the CURRENT directory, which is why this cds into the outdir. The default outdir is
# `tmp/`, not `prototype/attn_wide/`, so nothing built here can fall into the `.gitignore` binary-pattern trap
# recorded in AGENTS.md (a `prototype/**/<op>` pattern matches the op DIRECTORY and untracks the whole thing).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wide_time}
mkdir -p "$out"
cd "$out"

FLAGS=(-O3 --offload-arch=gfx1151 -Wno-unused-command-line-argument
       -I"$repo" -I"$repo/engine" -I"$here")
for v in WIDE_WARPS WIDE_STAGE WIDE_QREG WIDE_PVF16 WIDE_MINB; do
  if [[ -n ${!v:-} ]]; then FLAGS+=(-D$v=${!v}); fi
done

hipcc "${FLAGS[@]}" --save-temps -c "$here/time_wide.cpp" -o time_wide.o

L=time_wide-hip-amdgcn-amd-amdhsa-gfx1151.s
[ -f "$L" ] || { echo "ABORT: no $L in $out; the register read needs it and it is not optional" >&2; exit 1; }

# Both `__global__`s must exist in the code object, by their metadata blocks (not by a text grep that a comment
# could satisfy): a missing block means the launcher's kernel was optimised out and the rig would time a stub.
for k in attn_wide_wave attn_fattn_wave_u8; do
  grep -q "\.amdhsa_kernel.*$k" "$L" \
    || { echo "ABORT: $L has no .amdhsa_kernel block for $k; the rig would be timing a shell of it." >&2; exit 1; }
done
n32=$(grep -c 'v_wmma_f32_16x16x16_f16' "$L" || true)
n16=$(grep -c 'v_wmma_f16_16x16x16_f16' "$L" || true)
[ "$((n32 + n16))" -gt 0 ] \
  || { echo "ABORT: no v_wmma sites in $L: neither attention form survived compilation." >&2; exit 1; }
echo "wmma sites in the TU: $n32 v_wmma_f32_16x16x16_f16 + $n16 v_wmma_f16_16x16x16_f16"
echo "axes: WIDE_WARPS=${WIDE_WARPS:-8} WIDE_STAGE=${WIDE_STAGE:-64} WIDE_QREG=${WIDE_QREG:-1} WIDE_PVF16=${WIDE_PVF16:-1} WIDE_MINB=${WIDE_MINB:-1}"

hipcc -Wl,-rpath,/opt/rocm/lib time_wide.o -o time_wide
echo "wrote $out/time_wide   (listing $out/$L — the register read parses THIS file)"
echo "next: bash prototype/attn_wide/run_a12_window.sh   (main agent only: it takes the gpu_lock)"
